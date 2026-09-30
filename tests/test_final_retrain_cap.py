"""`final_retrain.max_env_steps`: the profile-owned cap on the report protocol's budget.

WHY THIS EXISTS. `configs/methods/eureka.yaml` pins its published final-retrain budget,
2,621,440,000 env steps per seed (the released driver's ShadowHand default), and
every config that `extends:` it inherits the number. A profile cannot move a
method's `final_retrain.env_steps` (a method config wins on every key it states), so
without a cap a `dev` or `full` run of the eureka tree would train 5 x 2.62 B steps. The cap
is the guard: the published file stays faithful, the profile says what a run may
spend, and the artifact records both, so the two numbers can never be confused.

The cap is `same_as_train` in every profile, not a literal: a literal is the profile's
budget at authoring time, and would cut HumanoidBench's 2 M retrains
(`--set train.env_steps=2000000` under `humanoid`, which extends `dev`) to the dev
profile's 20 k. The sentinel follows the launch-resolved `train.env_steps`.

Things a wrong answer would render as a plausible one, hence a test each:

  * the published point (no profile) is UNCAPPED -- a cap leaking into it would
    silently un-pin the paper's budget under the paper's own run id;
  * under a profile the effective budget is min(pinned, cap);
  * the phase writes `env_steps_pinned`, `env_steps_cap` and `env_steps_per_seed`,
    so a reader of `phases/final_retrain.json` sees which number trained.
"""
from __future__ import annotations

import json

import pytest

from bird import registry
from bird.artifacts import RunDir
from bird.budget import Budget
from bird.components.phases import _retrain_budget, run_final_retrain
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult

EUREKA_PINNED = 2_621_440_000


def test_the_published_point_is_uncapped_and_pins_the_drivers_budget():
    pinned, cap, effective = _retrain_budget(load("eureka"))
    assert (pinned, cap, effective) == (EUREKA_PINNED, None, EUREKA_PINNED)
    for name in ("rstar", "dreureka", "eureka_no_evolution", "roska"):
        assert _retrain_budget(load(name))[2] == EUREKA_PINNED, f"{name} extends eureka"


def test_every_execution_profile_caps_at_its_own_search_budget():
    for profile in ("tester", "dev", "full"):
        cfg = load("eureka", profile=profile)
        pinned, cap, effective = _retrain_budget(cfg)
        assert pinned == EUREKA_PINNED, "the profile must not move the method's pin"
        assert cap == cfg["train.env_steps"], (profile, cap, cfg["train.env_steps"])
        assert effective == cap
    # A method whose own pin is below the cap keeps its own number.
    pinned, cap, effective = _retrain_budget(load("gt", profile="tester"))
    assert pinned == 9_000_000 and effective == cap == 400


def test_the_cap_follows_the_launch_resolved_search_budget_not_the_profile_file():
    """The launch-override case: HumanoidBench runs `--set train.env_steps=2000000` under
    `humanoid` (extends `dev`, whose file says 20 000). A literal cap would freeze 20 000
    onto that launch and cut RDA's 2 M retrain to 20 k; `same_as_train` moves with the launch."""
    cfg = load("eureka", profile="dev", overrides={"train.env_steps": 2_000_000})
    assert _retrain_budget(cfg) == (EUREKA_PINNED, 2_000_000, 2_000_000)
    cfg = load("rda_humanoidbench", profile="humanoid", overrides={"train.env_steps": 2_000_000})
    pinned, cap, effective = _retrain_budget(cfg)
    assert cap == effective == 2_000_000, (pinned, cap, effective)
    # The full profile at 1 M caps gt's 9 M pin at 1 M, the same budget an explicit
    # `final_retrain.env_steps=1000000` override would ask for.
    assert _retrain_budget(load("gt", profile="full"))[2] == 1_000_000


def test_only_the_one_sentinel_string_is_accepted():
    import pytest
    from bird.config import ConfigError
    with pytest.raises(ConfigError):
        load("eureka", profile="dev", overrides={"final_retrain.max_env_steps": "lots"})
    with pytest.raises(ConfigError):
        load("eureka", profile="dev", overrides={"final_retrain.max_env_steps": 0})
    assert load("eureka", profile="dev", overrides={"final_retrain.max_env_steps": 12345})["final_retrain.max_env_steps"] == 12345


def test_the_phase_records_pinned_cap_and_what_trained(tmp_path):
    registry.load_all()
    cfg = load("eureka", profile="tester", overrides={
        "seed": 0, "final_retrain.n_seeds": 2, "evaluate.rollouts_per_candidate": 1,
    })
    ctx = Context(cfg=cfg, budget=Budget(), env=registry.get("env", cfg["problem.env_id"])({}))
    ctx.rundir = RunDir(tmp_path, "t", "0" * 8)
    reward = ("def compute_reward(s, a, s2):\n"
              "    import numpy as np\n"
              "    o = np.asarray(s2, dtype=float)\n"
              "    return float(-np.linalg.norm(o[0:2] - o[2:4]))\n")
    last = cfg["loop.n_iterations"] - 1
    cand = Candidate("c_cap", last, reward)
    state = RunState(iteration=last, restart=0)
    backend = registry.get("train_backend", cfg["train.backend"])
    result = backend(ctx, state, cand, n_seeds=1)
    score = registry.get("fitness_source", cfg["evaluate.fitness.source"])
    state.best = score(ctx, state, [result])[0]

    run_final_retrain(ctx, state)

    body = json.loads((ctx.rundir.path / "phases" / "final_retrain.json").read_text())
    assert body["env_steps_pinned"] == EUREKA_PINNED
    assert body["env_steps_cap"] == cfg["train.env_steps"] == 400
    assert body["env_steps_per_seed"] == 400


# ==========================================================================
# the cap binds on the WHOLE retrain, before the first seed, on every path
# ==========================================================================


def _retrain_ctx(tmp_path, n_seeds, parallelism, cap):
    registry.load_all()
    cfg = load("eureka", profile="tester", overrides={
        "seed": 0, "final_retrain.n_seeds": n_seeds,
        "evaluate.rollouts_per_candidate": 1,
        "train.candidate_parallelism": parallelism,
    })
    ctx = Context(cfg=cfg, budget=Budget(max_policy_trainings=cap),
                  env=registry.get("env", cfg["problem.env_id"])({}))
    ctx.rundir = RunDir(tmp_path, "t", "0" * 8)
    reward = ("def compute_reward(s, a, s2):\n"
              "    import numpy as np\n"
              "    o = np.asarray(s2, dtype=float)\n"
              "    return float(-np.linalg.norm(o[0:2] - o[2:4]))\n")
    last = cfg["loop.n_iterations"] - 1
    state = RunState(iteration=last, restart=0)
    state.best = CandidateReport(
        cand_id="c_cap", candidate=Candidate("c_cap", last, reward),
        result=TrainResult(cand_id="c_cap",
                           candidate=Candidate("c_cap", last, reward)),
        fitness=0.0)
    return ctx, state


@pytest.mark.parametrize("parallelism", ["sequential", "parallel"])
def test_a_retrain_that_does_not_fit_is_refused_whole_on_every_path(
        tmp_path, parallelism):
    """Three seeds asked for, room for two: nothing trains, nothing is spent.

    ASSERTED ON THE BUDGET, NOT ON THE RAISE. `run_final_retrain` swallows
    `BudgetExceeded` by design -- the search is over and a spent budget must
    not turn the run's reported result into a crash -- so a test that only
    checked for an exception would check nothing at all here. What matters is
    that the COUNTERS do not move, because the failure this prevents is two
    runs of one config recording different numbers of paid trainings
    depending on the schedule.

    WHY THE WHOLE RETRAIN RATHER THAN k OF n. `final_retrain.n_seeds` is a
    report protocol, not a replicate count -- it is the third and distinct
    seed count for that reason -- so k seeds of a requested n is not a partial
    result. Under a per-seed check the sequential path would train two of the
    three and then raise, recording two paid trainings where a fanned-out run
    of the same config records none.

    BOTH PATHS IN ONE TEST BECAUSE THE CHECK IS ABOVE THE BACKEND. It sits in
    `run_final_retrain` between `n_seeds` and the backend call, so the
    sequential path and the forked fan-out both reach it before anything is
    dispatched -- which is the property being pinned, and why parametrising
    the schedule is the honest way to state it.
    """
    ctx, state = _retrain_ctx(tmp_path, n_seeds=3, parallelism=parallelism,
                              cap=2)
    run_final_retrain(ctx, state)

    assert ctx.budget.policy_trainings == 0, (
        f"the retrain spent {ctx.budget.policy_trainings} trainings under a "
        "cap that cannot fit it; it must be refused whole, before the first "
        "seed, so both schedules record the same zero")
    body = json.loads((ctx.rundir.path / "phases" / "final_retrain.json").read_text())
    assert body["retrained_fitness"] is None, (
        "a measurement that did not happen must read as one that did not "
        "happen, never as a number over fewer seeds")
    assert "budget" in (body.get("error") or "").lower()


@pytest.mark.parametrize("parallelism", ["sequential", "parallel"])
def test_a_retrain_that_fits_still_runs(tmp_path, parallelism):
    """The other half: a pre-flight that refused everything would pass the
    test above while disabling the phase on both schedules."""
    ctx, state = _retrain_ctx(tmp_path, n_seeds=2, parallelism=parallelism,
                              cap=2)
    run_final_retrain(ctx, state)

    assert ctx.budget.policy_trainings == 2, ctx.budget.policy_trainings
    body = json.loads((ctx.rundir.path / "phases" / "final_retrain.json").read_text())
    assert not (body.get("error") or "")
