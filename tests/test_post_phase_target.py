"""`post:` phases act on the restart that produced the RETURNED reward.

`bird.py` returns the argmax over the restarts' final artifacts. Handing every
post phase `states[-1]` -- the last restart's state -- would, with
`loop.n_restarts > 1` and the winner from an earlier restart, make
`final_retrain.json` (and RDA's `alignment_rate.json`) describe a reward
`result.json` does not name, while `run_final_retrain`'s own contract says it
retrains the returned reward. The wrong answer is a well-formed report about
the wrong candidate, which is why it gets a test.

The setup: two finished searches with fitness 0.9 and 0.1; the driver returns
the first, and the phase must receive the first. Moot at one restart.
"""
from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest
from conftest import REPO

from bird import registry
from bird.budget import Budget
from bird.config import load
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult


def _driver():
    spec = importlib.util.spec_from_file_location("bird_entry_post_target", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(cid: str, fitness: float) -> CandidateReport:
    cand = Candidate(cid, 0, "def reward(obs, action, next_obs):\n    return 0.0, {}\n")
    return CandidateReport(cid, cand, TrainResult(cid, cand, policy_ref="policy:" + cid),
                           fitness=fitness)


@pytest.mark.parametrize("winner_restart", [0, 1])
def test_post_phases_receive_the_restart_that_produced_the_returned_reward(winner_restart):
    registry.load_all()
    driver = _driver()
    cfg = load("eureka", profile="tester", overrides={
        "loop.n_restarts": 2, "post": ["final_retrain"], "output.tracker": "none"})
    scores = [0.1, 0.1]
    scores[winner_restart] = 0.9
    reports = [_report(f"restart_{i}_winner", s) for i, s in enumerate(scores)]
    states = [RunState(restart=i, best=r, latest=r) for i, r in enumerate(reports)]

    seen = []
    real_get = registry.get

    def get(family, name):
        if family == "phase" and name == "final_retrain":
            return lambda ctx, state: seen.append(state)
        return real_get(family, name)

    with patch.object(driver, "run_search", side_effect=states), \
            patch.object(registry, "get", side_effect=get):
        result = driver._execute(cfg, Budget(), None)

    winner = reports[winner_restart]
    assert result["returned_cand_id"] == winner.cand_id
    assert len(seen) == 1, "one post phase, called once"
    assert seen[0] is states[winner_restart], (
        f"final_retrain was handed restart {seen[0].restart}'s state; the returned "
        f"reward is restart {winner_restart}'s")
    assert seen[0].best.cand_id == result["returned_cand_id"]
