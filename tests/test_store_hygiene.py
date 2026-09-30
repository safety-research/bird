"""The TPE store pool must not reach the returned fitness.

`TrainResult.store_trajectories` is CARD's dedicated check pool
(`verify.tpe.trajectories_per_iteration`), collected for `update.memory.
trajectory_store` and read in exactly one place -- `components/update.py`'s
`dedicated` branch. It rides home on every result under candidate parallelism
because the fork pickles the whole `TrainResult`, so it is present, non-empty
and in scope at every consumer that takes a result.

The claim under test is that the RETURN PATH ignores it: the fitness a run
reports, and the native signal recorded beside it, must be byte-identical
whether the pool is empty or full. If they are not, the reported result depends
on a verification-stage buffer, two stages away, and the dependence is silent.

WHY THIS IS A CONSUMER TEST AND NOT A GREP. `grep -rn
store_trajectories bird/components/evaluation.py` is empty today, and that is
the whole problem with it as evidence: it is a statement about one file's
current text, it says nothing about the scorers' transitive reads, and it goes
stale the moment a scorer starts calling a helper that does look. So these
tests DRIVE the two functions `run_final_retrain` actually calls --
`registry.get("fitness_source", cfg["evaluate.fitness.source"])` and
`phases._score_native` -- and compare their outputs across the two inputs.

WHAT THE TEST WOULD MISS IF IT ONLY COMPARED `.fitness`. A scorer could leave
the scalar alone and still put a store-derived number into `meta`, which is
what `result.json` publishes and what a reader quotes. So the comparison is
over the whole report surface a consumer can see, and the negative control
below leaks through `meta` specifically, not through the scalar -- otherwise
the control would prove only that the comparison notices the easy case.
"""
from __future__ import annotations

import copy

import pytest

from bird import registry
from bird.budget import Budget
from bird.components import phases
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, Trajectory

_REWARD = ("def compute_reward(s, a, s2):\n"
           "    import numpy as np\n"
           "    o = np.asarray(s2, dtype=float)\n"
           "    return float(-np.linalg.norm(o[0:2] - o[2:4]))\n")


def _ctx_and_result():
    """One real training on the tester profile, exactly as stage 3 produces it."""
    registry.load_all()
    cfg = load("limen", profile="tester", overrides={
        "seed": 0,
        "train.env_steps": 200,
        "train.seeds_per_candidate": 1,
        "evaluate.rollouts_per_candidate": 2,
    })
    ctx = Context(cfg=cfg, budget=Budget(),
                  env=registry.get("env", cfg["problem.env_id"])({}))
    state = RunState(iteration=0, restart=0)
    backend = registry.get("train_backend", cfg["train.backend"])
    result = backend(ctx, state, Candidate("c0000", 0, _REWARD), n_seeds=1)
    assert result.trained and not result.error, result.error
    assert not result.store_trajectories, (
        "this config should collect no TPE store pool; the fixture's 'without' "
        "arm is not actually without")
    return cfg, ctx, state, result


def _with_stores(result):
    """The same result, plus a full store pool. Nothing else differs.

    The pool is built from the result's OWN rollouts and then perturbed, so it
    is well-formed enough for a scorer to use by accident -- an empty-shaped
    stub could be ignored for the wrong reason (it raised, or it was skipped as
    malformed) and the test would read clean without proving anything.
    """
    out = copy.deepcopy(result)
    pool = []
    for i, t in enumerate(result.trajectories or []):
        s = copy.deepcopy(t)
        # Values a scorer would visibly pick up if it reduced over this pool.
        s.ret = float(t.ret) + 1000.0 * (i + 1)
        s.success = not t.success
        pool.append(s)
    if not pool:                      # a backend that kept no rollouts
        pool = [Trajectory(rewards=[1000.0], success=True, length=1, ret=1000.0)]
    out.store_trajectories = pool
    assert out.store_trajectories, "the 'with' arm has no store pool"
    return out


def _surface(rep):
    """Everything a consumer can read off a report, not just the scalar."""
    if rep is None:
        return None
    return (rep.fitness, rep.fitness_source, rep.feedback, dict(rep.meta or {}))


# --------------------------------------------------------------------------
# the two consumers `run_final_retrain` calls

@pytest.mark.parametrize("consumer", ["fitness_source", "native"])
def test_the_return_path_ignores_the_tpe_store_pool(consumer):
    cfg, ctx, state, bare = _ctx_and_result()
    stored = _with_stores(bare)

    if consumer == "fitness_source":
        score = registry.get("fitness_source", cfg["evaluate.fitness.source"])
        a = _surface(score(ctx, state, [bare])[0])
        b = _surface(score(ctx, state, [stored])[0])
    else:
        a = _surface(phases._score_native(ctx, state, bare)[0])
        b = _surface(phases._score_native(ctx, state, stored)[0])

    assert a == b, (
        f"{consumer} changed when store_trajectories was populated.\n"
        f"  without: {a}\n  with:    {b}\n"
        "The returned fitness depends on CARD's verification-stage check pool, "
        "which is two stages away and empty on most configs -- so the same "
        "reward scores differently depending on `update.memory.trajectory_store`.")


def test_score_native_reports_the_same_channel_either_way():
    """`_score_native` returns `(report, channel, error)` and the channel is
    what the artifact records. A store-sensitive scorer could keep the number
    and change the channel, which is a different wrong answer with the same
    fitness."""
    _cfg, ctx, state, bare = _ctx_and_result()
    stored = _with_stores(bare)
    _, chan_a, err_a = phases._score_native(ctx, state, bare)
    _, chan_b, err_b = phases._score_native(ctx, state, stored)
    assert (chan_a, err_a) == (chan_b, err_b), (chan_a, err_a, chan_b, err_b)


# --------------------------------------------------------------------------
# teeth

def test_the_comparison_can_actually_fail(monkeypatch):
    """THE NEGATIVE CONTROL. Without it, the two tests above pass equally well
    against a comparison that cannot distinguish anything -- and a control that
    cannot fail is not evidence.

    The injected scorer leaks through `meta`, not through `.fitness`, because
    that is the case the cheap version of this test would miss: `result.json`
    publishes `meta`, so a store-derived number there is quoted by readers
    exactly as a fitness would be.
    """
    _cfg, ctx, state, bare = _ctx_and_result()
    stored = _with_stores(bare)

    real = registry.get("fitness_source", "ground_truth_metric")

    def leaky(c, s, results):
        reports = real(c, s, results)
        for rep, res in zip(reports, results):
            rep.meta = dict(rep.meta or {})
            rep.meta["n_store"] = len(getattr(res, "store_trajectories", ()) or ())
        return reports

    a = _surface(leaky(ctx, state, [bare])[0])
    b = _surface(leaky(ctx, state, [stored])[0])
    assert a != b, (
        "a scorer that demonstrably reads store_trajectories produced identical "
        "output -- the comparison in this file is not measuring anything")
    assert a[3]["n_store"] == 0 and b[3]["n_store"] > 0, (a[3], b[3])


def test_the_two_pools_are_actually_different_objects():
    """Guards the fixture, not the code. `_with_stores` deep-copies, so a
    scorer mutating one arm's result cannot silently make both arms equal by
    editing a shared object -- which would be a green test for a false reason.
    """
    _cfg, _ctx, _state, bare = _ctx_and_result()
    stored = _with_stores(bare)
    assert stored is not bare
    assert stored.trajectories is not bare.trajectories
    assert not bare.store_trajectories and stored.store_trajectories
    # and the pool really is distinguishable from the feedback pool
    assert [t.ret for t in stored.store_trajectories] != [
        t.ret for t in (stored.trajectories or [])]
