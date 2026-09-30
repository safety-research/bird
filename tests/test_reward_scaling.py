"""`train.reward_scaling: elite_moments` -- LaRes Eq. 3, applied where LaRes runs.

WHY THE STORED ACTION IS DECODED BY TYPE. The tester surrogates store actions
as INDICES into `env.action_set`; the sb3 export stores the action itself, a
float vector on a continuous env. Decoding every stored action as an index
would make every tester run apply the scaling and read normal, while on
`configs/methods/lares.yaml`'s own SAC config `action_set[vector]` raises `IndexError`
outside the relabel `try`: `_scaled_reward` would raise, the backend's compile
guard would turn it into `trained=False`, and every non-elite candidate of
every round after the first would fail to train. The stabiliser the paper's
collapse claim rests on would not merely be inert: it would break the
candidates it is meant to steady, and nothing in the artifact would name Eq. 3
as the cause.

WHY THE TRANSFORM IS PLANNED IN THE PARENT. Computed INSIDE the backend call,
i.e. inside a forked worker under `train.candidate_parallelism: parallel`, the
moment sample would be drawn from `ctx.rng` (the parent's Thompson allocator
draws from the same stream, so `parallel` would diverge from `sequential`: 7
waves against 9 on the tester probe) and journalled through a `ctx.rundir` the
worker has set to None (a `parallel` run's journal would carry no
`reward_scaling` record at all). It would also re-sample once per SLICE under
`shared_population`, where the release computes the scale once per evolution.
So stage 3 PLANS the transform in the parent, once per candidate, before
anything forks, and the backends apply the plan.

WHAT THIS FILE PINS:

  1. on a continuous sb3 buffer the scaled reward DIFFERS from the raw one by
     exactly the affine map Eq. 3 states, no row fails to relabel, and the
     seed row carries the moments the learner trained under;
  2. a `lares` tester run under `parallel` is bit-identical to `sequential`
     with the sample draw firing, both journals carry the `reward_scaling`
     events, and each candidate is planned exactly once per round.

The sb3 tests use pendulum SAC at a few hundred steps: the assertions are on
the transform's parameters and the row's accounting, not on a fitness.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import REPO  # noqa: F401  (path setup)
from bird import registry
from bird.budget import Budget
from bird.components import training as T
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport

pytestmark = pytest.mark.slow

try:
    import stable_baselines3  # noqa: F401

    HAVE_SB3 = True
except ImportError:  # pragma: no cover - depends on the machine
    HAVE_SB3 = False

needs_sb3 = pytest.mark.skipif(not HAVE_SB3, reason="train.backend: sb3 needs sb3+gymnasium")

UPRIGHT = """
def compute_reward(state, action):
    import numpy as np
    th = np.arctan2(state[1], state[0])
    return -(th ** 2) - 0.1 * state[2] ** 2, {"upright": -(th ** 2)}
"""

#: A reward on a different scale and offset from UPRIGHT, so Eq. 3 has
#: something to correct and the affine map is not the identity by accident.
COSINE = """
def compute_reward(state, action):
    import numpy as np
    th = np.arctan2(state[1], state[0])
    return 10.0 * np.cos(th) + 3.0, {"cos": float(np.cos(th))}
"""


@pytest.fixture(autouse=True)
def _clean_stores():
    for store in (T._POLICY_STORE, T._REPLAY_STORE, T._ROUND_REPLAY):
        store.clear()
    if HAVE_SB3:
        import torch
        torch.set_num_threads(1)
    yield
    for store in (T._POLICY_STORE, T._REPLAY_STORE, T._ROUND_REPLAY):
        store.clear()


def _ctx(**overrides):
    """`lares` on the dev profile (sb3/SAC), pendulum, Eq. 3 on, Eq. 4 off."""
    base = {"seed": 0, "problem.env_id": "pendulum", "output.tracker": "none",
            "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "evaluate.rollouts_per_candidate": 1,
            "train.env_steps": 400,
            "train.interaction_cfg.slices_per_candidate": 2,
            "train.interaction_allocator": "uniform",
            "train.reward_scaling": "elite_moments",
            "train.elite_constraint.kind": "none",
            "loop.carry": ["best_reward", "replay_buffer", "policy_checkpoint", "archive"]}
    base.update(overrides)
    cfg = load("lares", overrides=base, profile="dev")
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", "pendulum")(ctx)
    events = []
    ctx.event = lambda stage, **fields: events.append({"stage": stage, **fields})
    return ctx, events


def _elite_round(ctx, state, steps=200):
    """Train one candidate as the round's elite and publish it into `state`."""
    backend = registry.get("train_backend", ctx.cfg["train.backend"])
    elite = Candidate(cand_id="c0000", reward_code=UPRIGHT, iteration=0)
    res = backend(ctx, state, elite, 1, env_steps=steps)
    assert res.trained and res.replay_ref, res.error
    state.best = CandidateReport(cand_id=elite.cand_id, candidate=elite, result=res,
                                 fitness=0.5)
    state.replay_ref = res.replay_ref
    state.policy_ref = res.policy_ref
    return elite


# ==========================================================================
# 1. Eq. 3 on a continuous sb3 buffer
# ==========================================================================


@needs_sb3
def test_eq3_scales_a_reward_over_a_continuous_sb3_buffer():
    """The faithful setting: SAC on a continuous env, a real replay export.

    Three assertions, each of which would fail under index decoding at the
    first -- where `_scaled_reward` would raise `IndexError` off the first
    buffer row:

      * the scaled reward differs from the raw one and equals
        `(r - mu_new) * scale + mu_elite` on a stored transition;
      * every sampled row relabelled (`n_failed == 0`), so the moments are
        moments of the buffer and not of the rows that happened to survive;
      * the seed row of the training that followed carries the transform.
    """
    ctx, events = _ctx()
    state = RunState()
    _elite_round(ctx, state)
    buf = T._REPLAY_STORE[state.replay_ref]
    assert np.ndim(buf[0][1]) == 1, "pendulum's sb3 export stores action VECTORS"

    cand = Candidate(cand_id="c0001", reward_code=COSINE, iteration=1)
    # Stage 3's parent-side step, then the backend's apply step -- the two
    # halves `bird.py::train` and `_scaled_reward` run.
    plan = registry.get("reward_scaling", "elite_moments")(ctx, state, cand)
    assert plan["applied"] is True and cand.meta["reward_scaling"] is plan
    raw = T.compile_reward(cand.reward_code, cand)
    scaled = T._scaled_reward(ctx, state, cand, raw)

    applied = [e for e in events if e["stage"] == "reward_scaling"]
    # The elite's own training skipped ("no shared replay buffer yet"), rightly;
    # the candidate under test must not have.
    skipped = [e for e in events if e["stage"] == "reward_scaling_skipped"
               and e["cand_id"] == cand.cand_id]
    assert applied and not skipped, (applied, skipped)
    assert applied[-1]["n_failed"] == 0 and applied[-1]["n_samples"] >= 2, applied[-1]
    assert scaled is not raw and abs(scaled.scale - 1.0) > 1e-6, (
        "COSINE and UPRIGHT have different spreads; scale must not be 1")

    s, a, s2, _done = buf[0]
    a = np.asarray(a, dtype=float).ravel()
    r_raw = raw(s, a, s2)[0]
    r_scaled, comps = scaled(s, a, s2)
    assert r_scaled != pytest.approx(r_raw)
    assert r_scaled == pytest.approx((r_raw - scaled.mu_new) * scaled.scale + scaled.mu_elite)
    assert comps == raw(s, a, s2)[1], "components are the program's own claim, never scaled"

    backend = registry.get("train_backend", "sb3")
    res = backend(ctx, state, cand, 1, env_steps=100)
    assert res.trained, res.error
    row = res.seed_metrics[0]["reward_scaling"]
    assert row["applied"] is True and row["mode"] == "elite_moments"
    assert row["scale"] == pytest.approx(scaled.scale)
    assert row["mu_new"] == pytest.approx(scaled.mu_new)
    assert row["mu_elite"] == pytest.approx(scaled.mu_elite)
    assert row["sigma_new"] > 0 and row["sigma_elite"] > 0
    assert row["n_samples"] >= 2 and row["n_failed"] == 0
    assert row["elite_id"] == "c0000" and row["buffer_ref"] == state.replay_ref


# ==========================================================================
# 2. planned in the parent: parallel == sequential, and the journal has it
# ==========================================================================


def _journal(root):
    import json
    from pathlib import Path
    (run,) = [p for p in Path(root).iterdir() if p.is_dir()]
    return run, [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()
                 if line.strip()]


def test_parallel_matches_sequential_under_elite_moments_and_the_journal_has_it(tmp_path):
    """`tests/test_parallelism.py`'s invariant, on the one family that broke it.

    `moment_samples: 50` is below the mock buffer's row count, so the sample
    draw FIRES -- with the default 4096 it does not under the tester profile,
    so a worker-side draw would go unnoticed there. With the draw in the
    worker the two fingerprints would differ (the parent's `ctx.rng` stream is
    short of the worker's draws, so the Thompson allocator's waves differ) and
    the parallel journal would have no `reward_scaling` event at all; here both
    hold, and every candidate is planned exactly ONCE per round rather than
    once per slice.
    """
    from collections import Counter

    from test_parallelism import _run

    extra = {"train.interaction_cfg.moment_samples": 50}
    seq = _run("lares", tmp_path / "seq", extra=extra,
               **{"train.candidate_parallelism": "sequential"})
    par = _run("lares", tmp_path / "par", extra=extra,
               **{"train.candidate_parallelism": "parallel",
                  "loop.max_parallel_trainings": 2})
    assert set(seq) == set(par)
    differing = [k for k in sorted(seq) if seq[k] != par[k]]
    assert not differing, f"parallel changed {differing}"

    for root in (tmp_path / "seq", tmp_path / "par"):
        run, journal = _journal(root)
        applied = [e for e in journal if e["stage"] == "reward_scaling"]
        decided = [e for e in journal if e["stage"] in
                   ("reward_scaling", "reward_scaling_skipped", "reward_scaling_exempt")]
        assert applied, f"{root.name}: no reward_scaling event reached the journal"
        assert all(e["n_samples"] == 50 for e in applied), "the draw must have fired"
        assert max(Counter(e["cand_id"] for e in decided).values()) == 1, (
            "one plan per candidate per round, not one per slice")
        slices = [e for e in journal if e["stage"] == "interaction_slice"]
        assert len(slices) > len(decided), "the round was sliced, the plan was not"
        # And the plan is in the artifact, on the candidate, where a reader of
        # the run directory finds it.
        import json
        metas = [json.loads(p.read_text()) for p in sorted(run.glob("candidates/iter01_*/meta.json"))]
        assert metas and any((m.get("meta") or {}).get("reward_scaling", {}).get("applied")
                             for m in metas), "no iteration-1 candidate carries an applied plan"
