"""`train.candidate_parallelism: sequential` trains at ONE torch thread, like
every forked worker, so the two schedules execute the same arithmetic.

The rule is unconditional: `parallel` must stay bit-identical to
`sequential`. The forked paths pin one thread (`_pin_one_thread` in
`_child_main` and `_seed_child_main`). A batch launcher typically exports
OMP/MKL/TORCH_NUM_THREADS equal to the allocated CPU count, so if
`parallelism_sequential` pinned nothing, a sequential sb3 run would train at 8
or 16 threads while a parallel one trained at 1 -- different reduction orders,
different numbers (training.py's own comment (d): "sequential SB3 at 16 threads
and at 8 threads produce different gt_return"), and the parity test would need
`torch.set_num_threads(1)` in the parent by hand to hold. fasttd3 pins inside
its seed loop; sb3 relies on the dispatcher.

Two guards that need no learner and one that does: the dispatcher enters the
single-thread context around each backend call (recorded); with torch present
the call really runs at one thread and the count is restored after; and, slow
and sb3-gated, an unpinned parent at several threads produces the same rows
under `sequential` as under `parallel`.
"""

from __future__ import annotations

import contextlib

import pytest

from test_parallelism import scrub_volatile   # the ONE volatile list, recursively
from bird import registry
from bird.budget import Budget
from bird.components import training
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, TrainResult

_REWARD = ("def compute_reward(state, action, next_state):\n"
           "    return -float(abs(next_state[0]))\n")


def _ctx(profile: str = "tester", **overrides) -> Context:
    registry.load_all()
    cfg = load("eureka", profile=profile, overrides=overrides or None)
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    return ctx


def _cands(n: int):
    return [Candidate(cand_id=f"c{i:04d}", iteration=0, reward_code=_REWARD) for i in range(n)]


def test_the_sequential_dispatcher_pins_one_thread_around_each_backend_call(monkeypatch) -> None:
    ctx = _ctx()
    entered = []
    active = {"pinned": False}

    @contextlib.contextmanager
    def recording():
        entered.append(True)
        active["pinned"] = True
        try:
            yield
        finally:
            active["pinned"] = False
    monkeypatch.setattr(training, "_single_thread_torch", recording)

    seen_pinned = []

    def run_one(c):
        seen_pinned.append(active["pinned"])
        return TrainResult(cand_id=c.cand_id, candidate=c, trained=True)

    cands = _cands(3)
    cands[1].screened_out, cands[1].failure_kind = True, "screened"
    out = training.parallelism_sequential(ctx, RunState(), cands, run_one, lambda c, r: None)

    assert [r.trained for r in out] == [True, False, True]
    assert seen_pinned == [True, True], "a backend call ran outside the single-thread pin"
    assert len(entered) == 2, "the pin is per backend call; a skipped candidate makes none"
    assert active["pinned"] is False


def test_with_torch_present_the_backend_call_really_runs_at_one_thread() -> None:
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        ctx = _ctx()
        inside = []

        def run_one(c):
            inside.append(torch.get_num_threads())
            return TrainResult(cand_id=c.cand_id, candidate=c, trained=True)

        training.parallelism_sequential(ctx, RunState(), _cands(1), run_one, lambda c, r: None)
        assert inside == [1], f"the sequential backend call ran at {inside} torch threads"
        assert torch.get_num_threads() == 2, "the parent's count must be restored afterwards"
    finally:
        torch.set_num_threads(before)


@pytest.mark.slow
def test_sequential_matches_parallel_on_sb3_without_pinning_the_parent_by_hand(tmp_path) -> None:
    """The claim on the real backend, with the parent left at several threads
    -- the state a batch launcher puts it in. `tests/test_parallelism.py`'s sb3
    case sets `torch.set_num_threads(1)` first; this one must not need to."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    before = torch.get_num_threads()
    rows = {}
    try:
        torch.set_num_threads(4)
        for sched in ("sequential", "parallel"):
            ctx = _ctx(profile="dev", **{
                "seed": 0, "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
                "output.tracker": "none", "train.env_steps": 600,
                "evaluate.rollouts_per_candidate": 1,
                "train.candidate_parallelism": sched, "loop.max_parallel_trainings": 1})
            assert ctx.cfg["train.backend"] == "sb3"
            dispatch = registry.get("candidate_parallelism", sched)
            backend = registry.get("train_backend", "sb3")
            out = dispatch(ctx, RunState(), _cands(1),
                           lambda c: backend(ctx, RunState(), c, n_seeds=1), lambda c, r: None)
            res = out[0]
            assert res.trained, res.error
            # The checkpoint curve is scrubbed as well as the seed metrics: it
            # carries wall-clock fields, so an unscrubbed curve differs between
            # the two schedules. This test `importorskip`s torch AND sb3, so it
            # only runs where both are installed.
            rows[sched] = scrub_volatile(res.seed_metrics[0])
            rows[sched]["checkpoints"] = scrub_volatile(res.checkpoints)
    finally:
        torch.set_num_threads(before)
    assert rows["sequential"] == rows["parallel"]
