"""`loop.carry` is applied at EVERY iteration boundary, an all-failed round included.

`run_iteration` returns None before `update()` when no report is valid. If
`state.apply_carry` were called only at the end of `update()`, a carry slot
written in stages 1-4 -- `subtasks` by the decompose phase, `preferences` by
evaluate, `failure_memory` by stage 2's `teach_next_iteration`, `critics` by
R* -- would survive a total-failure iteration regardless of `loop.carry`,
against `apply_carry`'s docstring and the `loop.carry` contract ("state not
named here is dropped at the iteration boundary"). On rda with `subtask_list`
uncarried, the iteration after a forced all-fail would enter generate with the
failed round's four subtasks and skip its decompose call.

No shipped config carries a slot it does not also write before stage 6, so no
shipped config exercises it; the rule is what this pins. `generate` is stubbed to write
non-carried slots and hand back only invalid candidates, which is the shape of
an all-failed round without depending on the mock's failure rate.
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


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx(entry, **overrides) -> Context:
    registry.load_all()
    cfg = load("eureka", profile="tester", overrides=overrides or None)
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.generator = registry.get("llm", "mock")(ctx, role="generator")
    ctx.evaluator = registry.get("llm", "mock")(ctx, role="evaluator")
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    ctx.human = registry.get("phase", "human_oracle")(ctx)
    return ctx


def _all_invalid_generate(ctx, state):
    """Stage 1 that writes three carry slots and forfeits every slot."""
    state.subtasks = ["leaked from the failed round"]
    state.failure_memory.append({"cand_id": "c0000", "trace": "NameError"})
    state.dialogue.append({"role": "assistant", "content": "kept: dialogue is carried"})
    return [Candidate(cand_id=f"c{i:04d}", iteration=state.iteration, reward_code="",
                      valid=False, failure="provider returned an empty sample",
                      failure_kind="invalid")
            for i in range(int(ctx.cfg["generate.n_candidates"]))]


def test_a_total_failure_round_still_drops_the_slots_it_does_not_carry() -> None:
    entry = _entry()
    ctx = _ctx(entry)
    assert ctx.cfg["loop.carry"] == ["best_reward", "dialogue"], "precondition: eureka's carry"
    entry.generate = _all_invalid_generate
    state = RunState()

    selection = entry.run_iteration(ctx, state)

    assert selection is None, "precondition: the round has to be a total failure"
    assert state.subtasks == [], (
        f"subtasks {state.subtasks!r} survived an all-failed round though "
        f"`subtask_list` is not in loop.carry")
    assert state.failure_memory == [], "failure_memory is not carried by eureka"
    # the carried slot is untouched, on this path as on the success path
    assert [m["content"] for m in state.dialogue] == ["kept: dialogue is carried"]


def test_a_carried_slot_survives_the_same_boundary() -> None:
    """The mirror: name the slot and the same round keeps it."""
    entry = _entry()
    ctx = _ctx(entry, **{"loop.carry": ["best_reward", "dialogue", "subtask_list",
                                        "failure_memory"]})
    entry.generate = _all_invalid_generate
    state = RunState()

    assert entry.run_iteration(ctx, state) is None
    assert state.subtasks == ["leaked from the failed round"]
    assert len(state.failure_memory) == 1
