"""`components.training.policy_from_ref` and the base-class `EnvAdapter.dr_probe`.

`_POLICY_STORE` holds three kinds of blob under one key shape, and `policy_from_ref`
is the one reader that knows all three (`training.py`, its docstring). These tests
hand-build each kind on `toy_reacher` and check the rebuilt callable against the
arithmetic the kind defines -- pure numpy, no learner, except the SB3 cases, which
`importorskip` and train a 200-step SAC once per module.

`dr_probe` lives on `EnvAdapter`, built on top of it, which is what makes DrEureka's
RAPP sweep measurable on any adapter and on `train.backend: sb3`. The last tests pin
that the base implementation answers a number, restores the DR it found, and is what
`phases._dr_probe` actually reaches.
"""

from __future__ import annotations

import math
from typing import Any, List

import numpy as np
import pytest

from bird import registry
from bird.components import training as T


def _env() -> Any:
    registry.load_all()
    return registry.get("env", "toy_reacher")(None)


def _states(env: Any, n: int = 5, seed: int = 3) -> List[np.ndarray]:
    """`n` states off one episode driven by random actions.

    Not `n` resets: `toy_reacher._reset` draws from the far quadrant only, so five
    resets would test the policy on five nearly identical states and never exercise
    a moving mass or the action clip.
    """
    rng = np.random.default_rng(seed)
    s = env.reset(rng)
    out = [s]
    while len(out) < n:
        s, _done, _info = env.step(s, rng.uniform(-1.0, 1.0, size=env.action_dim))
        out.append(s)
    return out


# ==========================================================================
# the two surrogate kinds
# ==========================================================================


def test_policy_from_ref_rebuilds_a_q_table() -> None:
    env = _env()
    rng = np.random.default_rng(0)
    q = rng.normal(size=(env.n_disc_states, env.n_actions))
    policy = T.policy_from_ref(None, env, q, ref="policy:q")
    for s in _states(env):
        expected = env.action_set[int(np.argmax(q[env.discretise(s)]))]
        np.testing.assert_array_equal(policy(s), expected)


def test_policy_from_ref_rebuilds_a_linear_policy() -> None:
    env = _env()
    rng = np.random.default_rng(1)
    # scale 2 so the clip is exercised, and asserted below to be exercised: an
    # unclipped comparison would pass with the clip deleted.
    w = rng.normal(scale=2.0, size=(env.action_dim, env.obs_dim + 1))
    policy = T.policy_from_ref(None, env, w, ref="policy:w")
    clipped = 0
    for s in _states(env):
        raw = w @ np.append(np.asarray(s, dtype=float), 1.0)
        expected = np.clip(raw, env.action_low, env.action_high)
        clipped += int(np.any(raw != expected))
        np.testing.assert_array_equal(policy(s), expected)
    assert clipped, "no state reached the action bounds; the clip went untested"


def test_policy_from_ref_honours_a_co_designed_observation() -> None:
    """LIMEN: a linear policy trained over phi has phi's width, and the rebuilt
    callable still takes the RAW state -- it features internally, as
    `_linear_policy` over an `_ObsView` does."""
    env = _env()
    code = ("def get_observation(state):\n"
            "    return np.asarray([state[4] - state[0], state[5] - state[1], state[2]])\n")
    rng = np.random.default_rng(4)
    w = rng.normal(size=(env.action_dim, 3 + 1))
    policy = T.policy_from_ref(None, env, w, code, ref="policy:phi")
    for s in _states(env):
        phi = np.asarray([s[4] - s[0], s[5] - s[1], s[2]], dtype=float)
        expected = np.clip(w @ np.append(phi, 1.0), env.action_low, env.action_high)
        np.testing.assert_allclose(policy(s), expected, rtol=0, atol=1e-12)
    # The same blob WITHOUT its observation code fits nothing on the raw env, and the
    # error says what is missing rather than guessing a feature map.
    with pytest.raises(ValueError, match="observation_code"):
        T.policy_from_ref(None, env, w, ref="policy:phi")


def test_policy_from_ref_refuses_a_blob_that_does_not_fit() -> None:
    env = _env()
    with pytest.raises(ValueError, match="policy:bad"):
        T.policy_from_ref(None, env, np.zeros((3, 3)), ref="policy:bad")
    with pytest.raises(ValueError, match="policy:none"):
        T.policy_from_ref(None, env, None, ref="policy:none")
    with pytest.raises(ValueError, match="policy:str"):
        T.policy_from_ref(None, env, "not a policy", ref="policy:str")


# ==========================================================================
# the sb3 kind
# ==========================================================================

#: Small enough that 200 steps train in about a second; `learning_starts` well below
#: the step count so the actor's weights MOVE -- the rebuild test below relies on the
#: trained network predicting differently from a fresh one with the same seed.
_SB3_HYPER = {"learning_starts": 20, "batch_size": 16, "buffer_size": 512,
              "train_freq": 1, "gradient_steps": 1,
              "policy_kwargs": {"net_arch": [8, 8]}}


def _sb3_cfg(hyper: dict) -> Any:
    from bird.config import load
    return load("eureka", profile="tester", overrides={
        "seed": 0, "train.backend": "sb3", "train.algorithm": "sac",
        "train.hyperparameters": dict(hyper)})


@pytest.fixture(scope="module")
def sb3_trained():
    """`(cfg, env, model, blob, make_gym)`: one SAC trained the way `_sb3_run` builds
    its model -- `_sb3_algo_and_hyper(cfg)` for the class and kwargs (which carry
    `verbose`, defaulted to 0), `algo("MlpPolicy", gym_env, seed=..., **hyper)` -- and
    serialised with `_sb3_policy_blob`."""
    pytest.importorskip("stable_baselines3")
    gym = pytest.importorskip("gymnasium")
    from gymnasium import spaces

    env = _env()
    cfg = _sb3_cfg(_SB3_HYPER)
    algo, hyper, _anchor = T._sb3_algo_and_hyper(cfg)

    class _Gym(gym.Env):
        """`_sb3_run._Gym`'s shape with a fixed goal-distance reward: the point is a
        network whose weights have moved off their initialisation, not a good policy."""
        metadata: dict = {}

        def __init__(self) -> None:
            self.observation_space = spaces.Box(np.asarray(env.obs_low, dtype=np.float32),
                                                np.asarray(env.obs_high, dtype=np.float32))
            self.action_space = spaces.Box(np.asarray(env.action_low, dtype=np.float32),
                                           np.asarray(env.action_high, dtype=np.float32))
            self._s = None
            self._t = 0
            self._episode = 0

        def reset(self, *, seed=None, options=None):
            self._s = env.reset(np.random.default_rng(self._episode))
            self._episode += 1
            self._t = 0
            return np.asarray(self._s, dtype=np.float32), {}

        def step(self, action):
            s2, done, info = env.step(self._s, np.asarray(action, dtype=float))
            r = -float(np.hypot(s2[0] - s2[4], s2[1] - s2[5]))
            self._s = s2
            self._t += 1
            return np.asarray(s2, dtype=np.float32), r, bool(done), self._t >= env.horizon, info

    model = algo("MlpPolicy", _Gym(), seed=0, **hyper)
    model.learn(200)
    blob = T._sb3_policy_blob(model)
    assert blob is not None
    return cfg, env, model, blob, _Gym


def _predict(model: Any, s: np.ndarray) -> np.ndarray:
    # Under the same thread pin the rebuilt callable predicts under, so the
    # comparison is weights against weights and not thread count against thread count.
    with T._single_thread_torch():
        act, _ = model.predict(np.asarray(s, dtype=np.float32), deterministic=True)
    return np.asarray(act, dtype=float).ravel()


def test_policy_from_ref_rebuilds_an_sb3_blob(sb3_trained) -> None:
    cfg, env, model, blob, make_gym = sb3_trained
    assert isinstance(blob, T._PolicyBlob) and blob.dtype == np.uint8 and blob.ndim == 1
    rebuilt = T.policy_from_ref(cfg, env, blob, ref="policy:sb3")
    states = _states(env)
    for s in states:
        np.testing.assert_array_equal(rebuilt(s), _predict(model, s))
    # And the equality means something: a fresh network under the same seed and
    # config predicts differently, so a rebuild that skipped the load (or degraded
    # `_sb3_apply_policy`'s False into a cold start) would have failed above.
    algo, hyper, _anchor = T._sb3_algo_and_hyper(cfg)
    fresh = algo("MlpPolicy", make_gym(), seed=0, **hyper)
    assert any(not np.array_equal(rebuilt(s), _predict(fresh, s)) for s in states), (
        "200 training steps left the actor at its initialisation; the rebuild check "
        "above cannot distinguish a loaded policy from an untrained one")


def test_policy_from_ref_refuses_an_sb3_blob_that_does_not_fit(sb3_trained) -> None:
    """The `False` from `_sb3_apply_policy` is a refusal here, never a fresh network."""
    _cfg, env, _model, blob, _make_gym = sb3_trained
    other = _sb3_cfg({**_SB3_HYPER, "policy_kwargs": {"net_arch": [16, 16]}})
    with pytest.raises(ValueError, match="policy:sb3"):
        T.policy_from_ref(other, env, blob, ref="policy:sb3")
    # An sb3 blob with no config cannot name its algorithm, so it is refused too.
    with pytest.raises(ValueError, match="config"):
        T.policy_from_ref(None, env, blob, ref="policy:sb3")


# ==========================================================================
# `EnvAdapter.dr_probe`
# ==========================================================================


def _homing_controller(env: Any) -> np.ndarray:
    """`a = k (goal - pos) - c vel`, as a `(action_dim, obs_dim + 1)` matrix over
    `toy_reacher`'s `[px, py, vx, vy, gx, gy]`: a linear policy that actually reaches
    the goal, so the probe has a success to count and the test can tell a probe that
    measured from one that installed the wrong dynamics and scored nothing."""
    k, c = 1.5, 1.0
    w = np.zeros((env.action_dim, env.obs_dim + 1))
    w[0, [0, 2, 4]] = (-k, -c, k)
    w[1, [1, 3, 5]] = (-k, -c, k)
    return w


def test_base_dr_probe_is_not_nan_for_a_linear_policy(monkeypatch) -> None:
    env = _env()
    assert "mass" in env.dr_parameters, "the probe test overrides an axis the env declares"
    ref = "policy:test-dr-probe"
    T._POLICY_STORE[ref] = _homing_controller(env)
    installed: List[Any] = []
    real_set_dr = env.set_dr

    def recording_set_dr(params):
        installed.append(None if params is None else dict(params))
        return real_set_dr(params)

    monkeypatch.setattr(env, "set_dr", recording_set_dr)
    try:
        assert math.isnan(env.dr_probe("policy:absent", {"mass": 2.0}, 2))
        assert installed == [], "no policy to roll out must not touch the DR"

        rate = env.dr_probe(ref, {"mass": 2.0}, 2)
        assert math.isfinite(rate) and 0.0 <= rate <= 1.0
        assert rate > 0.0, "a homing controller at 2x mass should still succeed sometimes"
        assert installed == [{"mass": 2.0}, None], "override installed, then nominal restored"
        assert env.dr_config == {}
        assert env.dr_probe(ref, {"mass": 2.0}, 2) == rate, "fixed rng: the probe is repeatable"

        # A distribution that was already installed comes back, not nominal.
        installed.clear()
        real_set_dr({"damping": (0.3, 0.6)})
        before = env.dr_config
        assert before, "the precondition of this check is an installed distribution"
        env.dr_probe(ref, {"mass": 0.5}, 2)
        assert env.dr_config == before
        assert installed[-1] == before
    finally:
        T._POLICY_STORE.pop(ref, None)
        real_set_dr(None)


def test_base_dr_probe_is_nan_where_no_success_criterion_exists(monkeypatch) -> None:
    """An adapter whose `success_threshold` is not finite (`GymMujoco`: `success()` is False
    unconditionally) has no rate to report: NaN, which RAPP reads as "decline to narrow",
    never a measured 0.0 that would mark every axis `degenerate` under a measurement's name.
    The DR is not touched. Mutation: drop the guard (a finite 0.0 comes
    back and `set_dr` is called)."""
    env = _env()
    ref = "policy:test-dr-probe-nobar"
    T._POLICY_STORE[ref] = _homing_controller(env)
    calls: List[Any] = []
    monkeypatch.setattr(env, "set_dr", lambda params: calls.append(params))
    monkeypatch.setattr(type(env), "success_threshold", float("inf"), raising=False)
    try:
        assert math.isnan(env.dr_probe(ref, {"mass": 2.0}, 2))
        assert calls == [], "an unsupported probe must not install anything"
    finally:
        T._POLICY_STORE.pop(ref, None)


def test_phases_dr_probe_reaches_the_base_implementation() -> None:
    """`phases._dr_probe` tries `env.dr_probe` first and passes `ctx.cfg`; a stored
    linear policy on the tester profile therefore probes to a number -- without the base
    implementation the RAPP sweep is `degenerate` on every non-Meta-World adapter. A blob
    that cannot be rebuilt is NaN there, not an exception out of the phase."""
    from bird.budget import Budget
    from bird.components.phases import _dr_probe
    from bird.config import load
    from bird.context import Context

    env = _env()
    cfg = load("dreureka", profile="tester")
    ctx = Context(cfg=cfg, budget=Budget(), env=env)
    ref, bad = "policy:test-phase-probe", "policy:test-phase-garbage"
    T._POLICY_STORE[ref] = _homing_controller(env)
    T._POLICY_STORE[bad] = np.zeros(16, dtype=np.uint8)
    try:
        rate = _dr_probe(ctx, ref, "mass", 1.0, 2)
        assert math.isfinite(rate) and rate > 0.0
        assert math.isnan(_dr_probe(ctx, bad, "mass", 1.0, 2))
        assert math.isnan(_dr_probe(ctx, "policy:absent", "mass", 1.0, 2))
        assert env.dr_config == {}
    finally:
        T._POLICY_STORE.pop(ref, None)
        T._POLICY_STORE.pop(bad, None)
        env.set_dr(None)
