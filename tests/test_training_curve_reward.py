"""The training curve is TWO series: the task metric and the candidate's own return.

Both trainers already evaluate the policy at every `train.checkpoint_interval`
and put both numbers on the checkpoint point (`_evaluate_policy`):

    fitness        the ENVIRONMENT's task metric at that point in training
    reward_return  what the CANDIDATE's own reward paid those same rollouts
    gt_return      what the REFERENCE reward paid them

Both land in `train_result.json`, and together they answer the comparison a
reader of a run actually wants: does the reward this method wrote go up in step
with the task metric, or away from it? A reward climbing while fitness stays flat
is the reward-hacking signature -- the one failure the whole benchmark exists to
see.

WHY THIS IS PARAMETRISED OVER THE BACKENDS rather than testing one: the two
trainers are separate implementations of the same stage that can grow separate
answers to the same config key (`tests/test_pruning.py` has the example). A curve
reader reads four keys off a checkpoint point; a
backend that stops writing one of them yields an empty series with nothing going
red, so the contract is asserted per backend rather than assumed shared.
"""

from __future__ import annotations

import types as pytypes

import pytest

from conftest import backend_param_unimplemented

from bird import registry
from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.types import Candidate

#: Not in the tester-tier smoke suite (heavyweight execution: a real learner run per backend).
#: Deselected by `-m "not slow"`; run it with `-m slow`. See pyproject.toml.
pytestmark = pytest.mark.slow

_REWARD = """
def compute_reward(s, a, s2):
    import numpy as np
    o = np.asarray(s2, dtype=float)
    return float(-np.linalg.norm(o[0:2] - o[2:4]))
"""

#: The keys a training-curve reader needs off one checkpoint. `step` is the
#: x-axis; `fitness` is the task metric; `reward_return` and `gt_return` are the
#: candidate's and the reference reward's returns.
_CHART_KEYS = ("step", "fitness", "reward_return", "gt_return")


def _backends():
    registry.load_all()
    return sorted(n for (k, n) in registry._REGISTRY if k == "train_backend")


def _skip_if_unavailable(name: str) -> None:
    if name == "sb3":
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("gymnasium")
    if name == "fasttd3":
        pytest.importorskip("torch")
    # `simba_v2` is the FOURTH learner backend and this arm exists because
    # REGISTERING A BACKEND CREATES CASES IN FILES THE CHANGE NEVER TOUCHES.
    # `_backends()` reads the registry, so registering `train_backend: simba_v2`
    # gives every parametrised body here a `[simba_v2]` case. Without the arm it falls through
    # to the backend, which raises "train.backend: simba_v2 needs torch, which
    # is not installed", and because this file is slow-marked the default
    # `-m "not slow"` selection deselects it and cannot see the failure. `torch` and nothing else: the
    # port is torch, not a wrapper round upstream's JAX, so there is no
    # `jax`/`flax` to name and this is not the never-opening gate
    # `tests/test_no_silent_skip.py` refuses.
    if name == "simba_v2":
        pytest.importorskip("torch")

#: `train.backend: fasttd3` runs its upstream defaults otherwise -- 128 envs,
#: batch 32768, 1024-wide distributional critics, a GPU-scale set -- and this
#: file tests the CONTRACT, not learning. Injected only for that backend; the
#: others keep the config as written.
_FASTTD3_TINY = {"num_envs": 2, "batch_size": 32, "buffer_size": 64,
                 "critic_hidden_dim": 8, "actor_hidden_dim": 8, "num_atoms": 5,
                 "v_min": -10.0, "v_max": 10.0, "learning_starts": 1,
                 "compile": False}

#: `train.backend: simba_v2` runs UPSTREAM SimbaV2's defaults otherwise --
#: 512-wide hyperspherical blocks, a 101-bin distributional critic, batch 256 --
#: and this file tests the CONTRACT, not learning, for the same reason the
#: fasttd3 set above exists: the assertions are about checkpoint rows, pruning
#: decisions and budget accounting, none of which reads a layer width.
#:
#: SIZE KEYS ONLY, AND `learning_starts` IS DELIBERATELY NOT HERE.
#: `tests/test_simba_v2.py::TINY` also pins `learning_starts: 8` and
#: `buffer_size: 256`, and copying it verbatim makes these cases SLOWER than no
#: injection at all -- measured on one case
#: (`test_a_tie_at_the_peak_ships_the_later_checkpoint[simba_v2]`, cold, same
#: machine, same load): 31.5 s with upstream defaults, 109.0 s with TINY verbatim,
#: 14.7 s with the size keys alone. `learning_starts` is not a size knob, it is
#: a WORK knob: upstream's default exceeds these files' whole step budget, so
#: lowering it turns a near-zero-update run into thousands of updates and buys
#: nothing an assertion here reads. Copying a constant because its name says
#: "tiny" is the trap; the keys that matter are the ones that shrink a tensor.
#: Over all 21 `[simba_v2]` cases in the three files: 455 s with no injection,
#: 195 s with this one.
_SIMBA_V2_TINY = {"batch_size": 32, "critic_hidden_dim": 16,
                  "actor_hidden_dim": 16, "critic_num_bins": 11,
                  "compile": False, "actor_num_blocks": 1,
                  "critic_num_blocks": 1}


def _run(name: str):
    registry.load_all()
    overrides = {
        "seed": 0,
        "train.backend": name,
        "train.env_steps": 3000,
        "train.seeds_per_candidate": 1,
        "evaluate.rollouts_per_candidate": 1,
    }
    if name == "fasttd3":
        overrides["train.hyperparameters"] = dict(_FASTTD3_TINY)
    if name == "simba_v2":
        overrides["train.hyperparameters"] = dict(_SIMBA_V2_TINY)
    if name == "sb3":
        # PPO's rollout (2048) would otherwise be most of this budget: `_sb3_run`
        # charges measured steps and stops at the budget, so pin a rollout that
        # divides the chunk and the curve keeps its five rows (tests/test_pruning.py).
        overrides["train.hyperparameters"] = {"n_steps": 600, "batch_size": 100}
    cfg = load("eureka", profile="tester", overrides=overrides)
    ctx = Context(cfg=cfg, budget=Budget(),
                  env=registry.get("env", "toy_reacher")({}))
    return registry.get("train_backend", name)(
        ctx, pytypes.SimpleNamespace(restart=0, iteration=0),
        Candidate(cand_id="c0000", iteration=0, reward_code=_REWARD), 1)


def test_there_are_backends_to_check() -> None:
    """Guard the guard: a parametrised test over an empty list passes vacuously."""
    assert _backends()


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_every_checkpoint_carries_both_the_metric_and_the_reward(name):
    """One point, both quantities, indexed by env step. This is what makes a
    training curve of the reward possible at all."""
    _skip_if_unavailable(name)
    res = _run(name)

    assert res.checkpoints, f"{name} produced no checkpoint curve"
    for i, point in enumerate(res.checkpoints):
        missing = [k for k in _CHART_KEYS if k not in point]
        assert not missing, (
            f"{name} checkpoint {i} is missing {missing}; a curve reader reads "
            f"these off every point and would get an empty series instead")
        assert all(isinstance(point[k], float) for k in _CHART_KEYS), point


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_every_checkpoint_carries_the_wall_clock_its_evaluation_took(name):
    """`eval_wall_s`, per checkpoint, on the seed row -- measured.

    Without it, a vectorised run's train wall can only be split into
    training and evaluation by FITTING, e.g. as `iterations x per-iteration
    cost + a FIXED term` over several env counts, with the fixed term
    attributed to checkpoints x rollouts x steps through a single-env
    evaluation. That is a fit to a few totals and an arithmetic coincidence
    that matches it, with the eval never timed. On a heavily vectorised run
    such a fit can put most of the wall clock in evaluation rather than
    training -- far too load-bearing a number to stay fitted.

    PARAMETRISED OVER THE BACKENDS for this file's own reason: the two
    trainers are separate implementations that can grow separate answers to
    one config key. `_evaluate_policy_timed` gives them one definition of the
    timing, and this asserts both actually call it -- a backend that called
    the untimed evaluator would leave the key absent, and a cost argument
    would again rest on a fit.

    NOT AN UPPER BOUND, deliberately. Wall clock on a shared machine is not
    reproducible and a ceiling here would flake for reasons that have nothing
    to do with the code. The contract is that the number exists, is a float,
    and is positive -- 0.0 is the value a broken timer returns, and it is the
    one value that would let the fit survive unchallenged.
    """
    _skip_if_unavailable(name)
    res = _run(name)

    assert res.checkpoints, f"{name} produced no checkpoint curve"
    for i, point in enumerate(res.checkpoints):
        assert "eval_wall_s" in point, (
            f"{name} checkpoint {i} carries no `eval_wall_s`; the checkpoint "
            "eval can be most of the wall clock on a vectorised run and nothing "
            "else measures it")
        assert isinstance(point["eval_wall_s"], float), point
        assert point["eval_wall_s"] > 0.0, (
            f"{name} checkpoint {i} reports eval_wall_s={point['eval_wall_s']!r}; "
            "an evaluation that ran took nonzero time, so this is a broken "
            "timer rather than a fast one")


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_the_reward_series_is_not_the_metric_series(name):
    """They are different quantities on different scales, so they must never
    share a y-axis.

    Asserted as a RANGE fact rather than "they differ at some index": on a
    degenerate toy run both can be flat, and the claim that matters is that the
    reward's own return is recorded as a return -- unbounded, negative here --
    and not silently aliased to the [0, 1] rate beside it.
    """
    _skip_if_unavailable(name)
    res = _run(name)

    rewards = [p["reward_return"] for p in res.checkpoints]
    assert any(r != 0.0 for r in rewards), (
        f"{name}: every reward_return is 0.0, so the series carries nothing")
    assert all(r <= 0.0 for r in rewards), (
        f"{name}: `_REWARD` is a negative distance, so its return cannot be "
        f"positive -- this series is not the candidate's reward: {rewards}")


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_the_reference_reward_curve_is_the_same_length(name):
    """`gt_reward_curve` is plotted against the SAME x-axis as the checkpoints,
    so a mismatch would misalign every point."""
    _skip_if_unavailable(name)
    res = _run(name)

    assert len(res.gt_reward_curve) == len(res.checkpoints)
    assert res.gt_reward_curve == [p["gt_return"] for p in res.checkpoints]
