"""The per-step `info["success"]` is an EVENT, on every adapter, and it is present.

`EnvAdapter.step` documents `info["success"]` as the ground-truth per-step success flag.
Its one in-loop consumer, `train.bc_prior.accept: success` (`anchored_ppo.collect_demos`),
reads the FIRST step with `info.get("success", 0) > 0.5` as the demonstration's success,
cuts the episode `success_margin` steps later and drops an episode that never flags. That
gives the flag two obligations:

  * it must be the task's success condition on the arriving state -- the comparison
    `task_metric` reduces -- and never a proxy for progress. A `vx > 0` flag on a crawl
    task is True at step 1 and on nearly every step of any crawling demonstration, so
    every 1000-step demonstration would be cut a few dozen pairs in while `bc_prior.json`
    reported all of them succeeded.
  * it must be PRESENT. An `Acrobot` info of `{"tip_height": ...}` with no key would make
    the same rule raise "none of N demonstrations succeeded" for a teacher whose tip
    crosses the bar on 41 of 300 steps.

Two kinds of test. The contract sweep constructs every registered env this interpreter
can build, steps once from the reset distribution and asserts the key; the simulator tiers
skip per case exactly as `tests/test_env_spec_leak.py` does, with the HumanoidBench and
jax ids MARKED (deselected with `-m "not humanoid"` / `-m "not jax"` rather than
skipped). The per-adapter tests then pin the EVENT on the flag functions
directly, with synthetic states where the adapter needs MuJoCo -- `object.__new__` is this
repo's idiom for a metric-only instance (`tests/test_humanoid_hand.py::_metric_only`) --
so the three events are held in every profile, not only where a simulator is installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bird import registry
from conftest import env_param


def _registered_env_ids():
    """Every registered env NAME, marked where the tier needs a simulator no shared
    interpreter has. Names only -- this runs at collection and constructs nothing."""
    registry.load_all()
    return [env_param(n) for n in sorted(registry.names("env"))]


def _build(name):
    """Construct AND materialise, or skip: the default CI job installs `--extra test` and cannot
    build the simulator-backed adapters, and the parametrisation keys on names.

    CONSTRUCTION IS NOT THE ONLY MOMENT A SIMULATOR CAN GO MISSING. An adapter whose
    `__init__` is deliberately simulator-FREE (so `--validate-all` works in a venv with
    no runtime) constructs fine on a box without its simulator, and the ImportError
    arrives from the first `reset()` instead -- which would surface here as a FAILURE.

    So the probe is a reset rather than a constructor call, which catches an ImportError
    deferred past construction; the skip reason names the exception either way.
    """
    try:
        env = registry.get("env", name)({})
        env.reset(np.random.default_rng(0))
        return env
    except ImportError as exc:
        pytest.skip(f"{name}: adapter needs an optional simulator ({exc})")


def _nominal_action(env):
    """A zero action inside the adapter's own bounds: index 0 on a discrete set, no
    torque on a continuous one. Nominal, not random, so the step is the same every run."""
    lo = np.asarray(env.action_low, dtype=float).ravel()
    hi = np.asarray(env.action_high, dtype=float).ravel()
    return np.clip(np.zeros(lo.shape[0]), lo, hi)


# --------------------------------------------------------------------------
# the contract, over the registry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", _registered_env_ids())
def test_every_env_emits_the_flag_and_it_is_not_constant_true(env_id):
    """`step()` returns a `success` key, always -- False where there is no bar -- and
    on a bar-bearing env the flag is not True on every state probed: four first steps
    from the reset distribution plus eight coverage states from `random_state` (the
    same draw `sample_transitions` steps from, so stepping from it is already part of
    the contract). Both populations, because either alone is wrong for some task: a
    balance task's reset state IS its success state (a cart-pole starts upright on
    every seed), and a random pool can land on the goal (`toy_gridworld`). What this
    refuses is a flag that is constant True. The event itself is pinned per adapter
    below; this sweep is the presence and the sanity."""
    env = _build(env_id)
    a = _nominal_action(env)

    def flag_of(s):
        s2, done, info = env.step(s, a)
        assert isinstance(info, dict) and "success" in info, (
            f"{env_id}: step() returned info {sorted(info) if isinstance(info, dict) else info!r} "
            "with no `success` key. `EnvAdapter.step` promises one on every adapter; "
            "`train.bc_prior.accept: success` reads its absence as an episode that never "
            "succeeded and drops the demonstration.")
        flag = info["success"]
        assert isinstance(flag, (bool, float, int, np.bool_, np.floating, np.integer)), (
            f"{env_id}: info['success'] is {type(flag).__name__}, not a bool/float the "
            "`> 0.5` consumer can read")
        assert float(flag) in (0.0, 1.0), f"{env_id}: info['success'] = {flag!r} is not binary"
        return float(flag)

    flags = [flag_of(env.reset(np.random.default_rng(seed))) for seed in range(4)]
    coverage = np.random.default_rng(0)
    flags += [flag_of(env.random_state(coverage)) for _ in range(8)]
    if math.isfinite(float(env.success_threshold)):
        assert not all(f > 0.5 for f in flags), (
            f"{env_id}: the per-step success flag was True on every reset first step AND "
            "every coverage state -- a flag that is constant is a proxy for something, "
            "not a success event")
    else:
        assert not any(f > 0.5 for f in flags), (
            f"{env_id}: success_threshold is inf (no bar) yet the flag fired; the two "
            "reductions disagree about whether a success can exist here")


# --------------------------------------------------------------------------
# acrobot: the key exists, and it is the metric's own comparison
# --------------------------------------------------------------------------


def test_acrobot_flags_the_tip_above_the_bar_and_agrees_with_its_metric():
    """Hanging: present and False. Balanced on top and stepped
    once with no torque: True. And on every step the flag equals the comparison
    `task_metric` averages, so a one-state trajectory's metric IS the flag."""
    registry.load_all()
    env = registry.get("env", "acrobot")(None)
    s = env.reset(np.random.default_rng(0))
    s2, done, info = env.step(s, np.zeros(1))
    assert "success" in info and info["success"] is False and done is False
    assert info["tip_height"] == env.tip_height(s2) < env.goal_height

    up = env._obs(np.array([math.pi, 0.0, 0.0, 0.0]))          # both links straight up
    s3, done, info = env.step(up, np.zeros(1))
    assert info["success"] is True and env.tip_height(s3) > env.goal_height

    for state, flag in ((s2, False), (s3, True)):
        assert env.task_metric(np.asarray([state])) == float(flag), (
            "the per-step flag and the metric's per-step comparison disagree")


def test_acrobot_bc_prior_no_longer_drops_a_teacher_that_crosses_the_bar():
    """`simple_suite/acrobot_pump` (registered `negative`: it does not HOLD the tip
    up) still crosses the bar on 41 of 300 steps of seed 0, first at step 75. With no
    key, `collect_demos(accept="success")` would raise for it; the crossing is the
    event the rule keys on, and the kept demonstration
    is cut `success_margin` steps after it rather than running to the horizon."""
    from bird.components.anchored_ppo import collect_demos
    from bird.policy_api import load_policy

    registry.load_all()
    env = registry.get("env", "acrobot")(None)
    pol = load_policy("simple_suite/acrobot_pump")
    X, Y, counts = collect_demos(env, pol, 2, 25, np.random.default_rng(0), accept="success")
    assert counts["success_gated"] is True and counts["demos_used"] >= 1, counts
    assert counts["pairs"] < counts["demos_used"] * env.horizon, (
        "a demonstration that flagged was not truncated after its first success")


# --------------------------------------------------------------------------
# h1hand_crawl: inside the tunnel past its mouth, not moving forward
# --------------------------------------------------------------------------


def _metric_only(cls):
    """An instance with no simulator behind it: the flag reads only the state array and
    module constants. `tests/test_humanoid_hand.py::_metric_only` carries the argument
    for the idiom."""
    return object.__new__(cls)


def test_h1hand_crawl_flag_is_entering_the_tunnel_not_moving_forward():
    """`x > _TUNNEL_X0` and `|y| <= _CRAWL_CORRIDOR_HALF_WIDTH`. A forward-velocity
    flag would be True at the spawn while moving and False at rest inside the
    corridor; the event is the reverse."""
    from bird.envs import humanoid_hand as HH
    env = _metric_only(HH.H1HandCrawl)
    x0, half = HH._TUNNEL_X0, HH._CRAWL_CORRIDOR_HALF_WIDTH

    def flag(x, y):
        return env._success_flag(np.array([x, y]))

    assert flag(0.0, 0.0) is False, "the spawn is not inside the tunnel"
    assert flag(x0 + 0.8, 0.0) is True, "inside the corridor IS the event"
    assert flag(x0, 0.0) is False, "the mouth itself is not past it"
    assert flag(x0 + 0.8, half + 0.01) is False, "beside the tunnel"
    assert flag(x0 + 0.8, -half) is True, "the bound is inclusive"
    assert flag(float("nan"), 0.0) is False
