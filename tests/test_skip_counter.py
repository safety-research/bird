"""`policy_trainings_skipped` counts trainings a screen or allocator DECIDED not
to launch; `candidates_invalid` counts candidates that never compiled or
forfeited their slot. They must be two numbers.

A `_skip_result` that charges `record_skip()` for every `not c.trainable`
candidate, with `trainable = valid and not screened_out`, makes
`failure_kind: "invalid"` and `failure_kind: "screened"` increment the same
counter -- the one reserved for CARD's cost claim (RL runs it did NOT launch),
"a saved RL run, not a failure". Measured under that rule: the eureka tester
point, which has no quality screen at all, reports `policy_trainings_skipped:
16` for sixteen forfeited slots. The per-candidate artifact (`failure_kind`)
keeps the populations apart; the budget summary must too.
"""

from __future__ import annotations

import importlib.util

from conftest import REPO
from bird import registry
from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate

_REWARD = ("def compute_reward(state, action, next_state):\n"
           "    return -float(abs(next_state[0]))\n")


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx() -> Context:
    registry.load_all()
    cfg = load("eureka", profile="tester")
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    return ctx


def test_screened_and_invalid_candidates_are_counted_apart() -> None:
    entry = _entry()
    ctx = _ctx()
    good = Candidate(cand_id="c0000", iteration=0, reward_code=_REWARD)
    invalid = Candidate(cand_id="c0001", iteration=0, reward_code="", valid=False,
                        failure="provider returned an empty sample", failure_kind="invalid")
    screened = Candidate(cand_id="c0002", iteration=0, reward_code=_REWARD, screened_out=True,
                         failure="cascade: success 0.0000 < min_success_threshold 0.0100",
                         failure_kind="screened")

    results = entry.train(ctx, RunState(), [good, invalid, screened])

    assert [r.trained for r in results] == [True, False, False]
    b = ctx.budget
    assert b.policy_trainings == 1
    assert b.policy_trainings_skipped == 1, (
        f"policy_trainings_skipped={b.policy_trainings_skipped}: an invalid "
        f"candidate was counted as a training a screen chose not to launch")
    assert b.candidates_invalid == 1


def test_the_new_counter_travels_with_the_others() -> None:
    """Derived from the dataclass, so it merges across a fork, restores from a
    checkpoint and reaches budget.json without anyone extending a tuple."""
    b = Budget()
    b.record_invalid()
    assert "candidates_invalid" in Budget.counter_names()
    assert b.report()["candidates_invalid"] == 1
    other = Budget()
    other.merge_delta(b.delta_since(Budget().snapshot()))
    assert other.candidates_invalid == 1
