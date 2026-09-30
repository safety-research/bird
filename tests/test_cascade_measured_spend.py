"""The cascade's `short_budget_honoured` and `steps_actually_run` are MEASURED
off the TrainResult, not asserted off the backend's signature.

A signature predicate -- `honoured = True` because one of
`env_steps`/`steps`/`max_steps`/`budget_steps` appears in the backend's
signature -- and `steps_actually_run = steps`, a copy of the REQUEST, cannot
see a backend that accepts `env_steps` and reads `train.env_steps` anyway: it
would journal `honoured: true` at full cost, with every signature-level test
(tests/test_cascade_short_budget.py) green. The honest case would be
mis-recorded too: 3,000,000 recorded for a mock run the step cap clamps to
6,000.

So both fields come from the backend's own per-seed rows:
`train_steps_requested` is the bound the learner loop actually ran against and
`train_steps` the count it executed. Honoured means the backend adopted a bound
no larger than the ask. Runs on the mock backend under the tester profile.
"""

from __future__ import annotations

import pytest

from bird import registry
from bird.budget import Budget
from bird.components import screens, training
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate

_REWARD = ("def compute_reward(state, action, next_state):\n"
           "    import numpy as np\n"
           "    return -float(np.linalg.norm(next_state[4:6] - next_state[0:2]))\n")

#: Below the tester profile's `train.env_steps` (400) so a backend that reads
#: the config instead of the override is measurably OVER the ask -- a screen
#: costing more than it claimed.
_SHORT = 300


def _ctx(**over):
    registry.load_all()
    cfg = load("limen_reward_only", profile="tester", overrides={
        "verify.cascade.short_budget_steps": _SHORT,
        # > 1.0 so every candidate is REJECTED and the rejection record -- the
        # artifact a reader of `candidates/` sees -- carries the measured fields.
        "verify.cascade.min_success_threshold": 1.01,
        **over})
    assert cfg["train.env_steps"] > _SHORT, "precondition: the ask must be under the training budget"
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    events = []
    ctx.event = lambda stage, **fields: events.append({"stage": stage, **fields})
    return ctx, events


def _candidates(n: int):
    return [Candidate(cand_id=f"c{i:04d}", iteration=0, reward_code=_REWARD) for i in range(n)]


def _screen_record(c: Candidate) -> dict:
    recs = [r for r in c.verify_records if r.get("screen") == "cascade"]
    assert len(recs) == 1, c.verify_records
    return recs[0]


def _capturing(real):
    """The registered backend, with every TrainResult it returns kept."""
    results = []

    def wrapped(ctx, state, candidate, n_seeds=1, env_steps=None, resume_ref=None):
        res = real(ctx, state, candidate, n_seeds=n_seeds, env_steps=env_steps,
                   resume_ref=resume_ref)
        results.append(res)
        return res
    return wrapped, results


def test_an_honest_backend_is_measured_honoured_and_the_record_says_what_ran(monkeypatch) -> None:
    real = registry.get("train_backend", "mock")
    wrapped, results = _capturing(real)
    monkeypatch.setitem(registry._REGISTRY, ("train_backend", "mock"), wrapped)
    ctx, events = _ctx()
    cands = _candidates(2)

    screens.screen_cascade(ctx, RunState(), cands)

    assert len(results) == 2
    for c, res in zip(cands, results):
        row = res.seed_metrics[0]
        assert row["train_steps_requested"] == _SHORT, row
        rec = _screen_record(c)
        assert c.screened_out and c.failure_kind == "screened"
        assert rec["short_budget_honoured"] is True
        assert rec["short_budget_steps"] == _SHORT
        assert rec["short_budget_adopted"] == row["train_steps_requested"] == _SHORT
        assert rec["steps_actually_run"] == row["train_steps"], (
            "steps_actually_run must be the executed count, not the request")
        assert "NOT HONOURED" not in rec["detail"]
    assert not [e for e in events if e["stage"] == "screen_pass"]


def test_a_backend_that_accepts_and_ignores_the_override_is_reported_unhonoured(monkeypatch) -> None:
    """The class the signature predicate cannot see: `env_steps` in the
    signature, `train.env_steps` in the loop."""
    def leaky(ctx, state, candidate, n_seeds=1, env_steps=None, resume_ref=None):
        return training._run_backend(ctx, state, candidate, n_seeds, kind="auto",
                                     step_cap=training.MOCK_STEP_CAP, label="mock",
                                     env_steps=None, resume_ref=resume_ref)
    wrapped, results = _capturing(leaky)
    monkeypatch.setitem(registry._REGISTRY, ("train_backend", "mock"), wrapped)
    ctx, _events = _ctx()
    cands = _candidates(1)

    screens.screen_cascade(ctx, RunState(), cands)

    row = results[0].seed_metrics[0]
    full = int(ctx.cfg["train.env_steps"])
    assert row["train_steps_requested"] == full, "precondition: the leaky backend ran the full budget"
    rec = _screen_record(cands[0])
    assert rec["short_budget_honoured"] is False, (
        "a backend that took env_steps and ignored it was journalled as honoured")
    assert rec["short_budget_adopted"] == full
    assert rec["steps_actually_run"] == row["train_steps"] >= full
    assert "NOT HONOURED" in rec["detail"]


def test_a_passing_candidate_carries_the_same_measured_fields(monkeypatch) -> None:
    """The `screen_pass` event is the only record a KEPT candidate leaves."""
    real = registry.get("train_backend", "mock")
    wrapped, results = _capturing(real)
    monkeypatch.setitem(registry._REGISTRY, ("train_backend", "mock"), wrapped)
    ctx, events = _ctx(**{"verify.cascade.min_success_threshold": 0.0})
    cands = _candidates(1)

    screens.screen_cascade(ctx, RunState(), cands)

    passes = [e for e in events if e["stage"] == "screen_pass"]
    assert len(passes) == 1 and cands[0].trainable
    row = results[0].seed_metrics[0]
    assert passes[0]["short_budget_honoured"] is True
    assert passes[0]["short_budget_adopted"] == _SHORT
    assert passes[0]["steps_actually_run"] == row["train_steps"]


def test_the_measurement_reads_the_rows_not_the_signature() -> None:
    """Unit form of the rule, on a bare TrainResult."""
    from bird.types import TrainResult
    c = Candidate(cand_id="c", iteration=0, reward_code=_REWARD)
    res = TrainResult(cand_id="c", candidate=c, seed_metrics=[
        {"train_steps_requested": 300, "train_steps": 400}])
    assert screens._measured_short_run(res, 300) == (300, 400, True)
    assert screens._measured_short_run(res, 299) == (300, 400, False)
    res.seed_metrics = []
    assert screens._measured_short_run(res, 300) == (None, None, False), (
        "no rows means nothing was measured, and unmeasured is not honoured")
    res.seed_metrics = [{"train_steps_requested": 300, "train_steps": 300},
                        {"train_steps_requested": 6000, "train_steps": 6000}]
    assert screens._measured_short_run(res, 300) == (6000, 6300, False)
