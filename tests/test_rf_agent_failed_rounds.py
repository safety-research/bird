"""RF-Agent backs failed rounds up in their own iteration, first and last
rounds included.

The release backs every completed child up, failed ones at
`reward_fail_bound = 0` (`refs/code/RF-Agent/RF_Agent/rf_agent_algo/
rfagent_algo.py:673`, `:800-805`); whether the next round initialises or
expands depends only on whether the root has children (`:781-790`). BIRD's
`run_iteration` returns before §6 when every candidate fails. A tree that only
caught up with `_catch_up` one iteration late, guarded on `if state.tree`, would
leave the tree empty after an all-failed INITIALISATION round, so the next round
would re-issue zero-shot init calls (fourteen init children instead of six, and
a different sampling mode), and an all-failed FINAL round would never enter the
tree at all. A tree short of a round renders as a plausible tree.

Two mechanisms, two tests. `_descend` catches up into an empty tree (the probe:
six failed init reports, for which a tree without the catch-up returns `[]`
from `parent_uct_leaf`). And `loop.on_total_failure: continue_after_update` --
which `configs/methods/rf_agent.yaml` pins -- runs §6 on the failed round itself with no
winner, so it joins the tree in its own iteration and `_catch_up` is only the
backstop.
"""
from __future__ import annotations

import importlib.util
import random
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from conftest import REPO

from bird import registry
from bird.budget import Budget
from bird.components import tree
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult

_N = 6  # the probe: six failed initial children


def _driver():
    spec = importlib.util.spec_from_file_location("bird_entry_failed_rounds", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _failed(cid: str, iteration: int, slot: int) -> CandidateReport:
    """What §2-§4 hand on for a candidate that never compiled: invalid, untrained,
    `fitness` at RF-Agent's `select.failure_value` (0.0, the release's
    `reward_fail_bound`)."""
    cand = Candidate(cid, iteration, "def compute_reward(s, a, s2):\n    return 1/0\n",
                     meta={"sampling_mode": "tree_actions", "action": "init",
                           "sample_index": slot}).failed("ZeroDivisionError: division by zero")
    result = TrainResult(cid, cand, trained=False,
                         skip_reason="not trained: candidate failed verification")
    return CandidateReport(cid, cand, result, fitness=0.0, fitness_source="failure_value")


def _ctx(**overrides) -> Context:
    registry.load_all()
    cfg = load("rf_agent", profile="tester", overrides=overrides)
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(0))
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    ctx.tracker = SimpleNamespace(log_iteration=lambda *a, **k: None,
                                  log_media=lambda *a, **k: None)
    return ctx


def test_an_all_failed_initialisation_round_joins_the_tree_before_the_next_descent():
    """Iteration 1 begins with six failed init reports from iteration 0 and an
    empty tree: the descent must catch them up (root + six children at 0) and
    return a parent, so the next round expands a child with the five actions
    instead of re-initialising. Without the catch-up: `[]`, tree empty."""
    ctx = _ctx()
    state = RunState(iteration=1)
    state.all_reports = [_failed(f"c{i:04d}", 0, i) for i in range(_N)]

    parents = tree.parent_uct_leaf(ctx, state)

    assert len(state.tree) == _N + 1, "root plus one node per failed child"
    assert set(state.tree[tree.ROOT].children) == {f"c{i:04d}" for i in range(_N)}
    assert [p.cand_id for p in parents] == ["c0000"], "all q equal: ties keep the lowest index"


@pytest.mark.parametrize("policy,nodes_after", [
    ("continue", 0),                  # §6 skipped; `_catch_up` fills it in next round
    ("continue_after_update", _N + 1),  # §6 ran on the round: root + six, same iteration
])
def test_continue_after_update_backs_the_failed_round_up_in_its_own_iteration(policy, nodes_after):
    """`run_iteration` with every stage stubbed to an all-failed round. Under
    `continue_after_update` the round is in the tree when `run_iteration` returns -- which is
    the only way an all-failed FINAL round ever gets in -- and no winner was
    elected on the way (the sentinel-parent trap the early return exists for)."""
    driver = _driver()
    ctx = _ctx(**{"loop.on_total_failure": policy})
    state = RunState(iteration=0)
    reports = [_failed(f"c{i:04d}", 0, i) for i in range(_N)]

    with patch.object(driver, "generate", return_value=[r.candidate for r in reports]), \
            patch.object(driver, "verify", side_effect=lambda ctx, state, cands: cands), \
            patch.object(driver, "train", return_value=[r.result for r in reports]), \
            patch.object(driver, "evaluate", return_value=reports):
        selection = driver.run_iteration(ctx, state)

    assert selection is None, "still 'every candidate failed' to the loop"
    assert len(state.all_reports) == _N
    assert len(state.tree) == nodes_after
    assert state.best is None and state.iteration_best is None, "no winner was elected"
    if nodes_after:
        assert all(state.tree[c].score == 0.0 for c in state.tree[tree.ROOT].children)


def test_rf_agent_pins_the_same_iteration_backup():
    cfg = load("rf_agent")
    assert cfg["loop.on_total_failure"] == "continue_after_update"
    assert cfg["select.failure_value"] == 0.0, "the release's reward_fail_bound"
