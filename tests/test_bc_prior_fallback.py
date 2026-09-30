"""`train.bc_prior.accept: success_else_top` -- the `success` rule with a fallback.

A teacher registered `negative` never succeeds, so `success` has no demonstration and
raises. The fallback rolls `fallback_oversample` x `n_demos` fresh episodes and clones
the top `fallback_keep` fraction by the adapter's own `task_metric` (whole episodes).
The metric ranks the TEACHER's episodes only. Everything below is a fake env whose
metric is a known function of the episode's seed, so the kept set is checkable.
"""
from __future__ import annotations

import numpy as np
import pytest

from bird.components.anchored_ppo import collect_demos


class _Env:
    """Success bar 0.5. `succeed_at` < 0: never succeeds. task_metric = the episode's
    first-observation value, which `reset` derives from the seed -- so ranking by metric is
    ranking by a number the test can recompute."""
    action_low = np.array([-1.0]); action_high = np.array([1.0])
    horizon = 12

    def __init__(self, succeed_at: int = -1):
        self.success_threshold = 0.5; self.succeed_at = succeed_at

    def reset(self, rng):
        self._v = float(rng.random()); self._t = 0
        return np.array([self._v])

    def step(self, s, a):
        self._t += 1
        info = {"success": 1.0 if (0 <= self.succeed_at <= self._t) else 0.0}
        return np.array([self._v]), self._t >= self.horizon, info

    def task_metric(self, states):
        return float(np.asarray(states)[0, 0])


class _Teacher:
    def reset(self, rng): pass
    def act(self, s, *, t, env=None): return np.array([0.5])


def test_fallback_keeps_the_top_quarter_of_four_times_the_demos_by_metric():
    env = _Env(succeed_at=-1)
    X, Y, counts = collect_demos(env, _Teacher(), n_demos=8, success_margin=2,
                                 rng=np.random.default_rng(0), accept="success_else_top",
                                 fallback_oversample=4, fallback_keep=0.25)
    assert counts["accept"] == "success_else_top" and counts["fallback_used"] is True
    assert counts["fallback_pool"] == 32 and counts["fallback_kept"] == 8 == counts["demos_used"]
    assert counts["demos_skipped"] == 8 + 24, "8 failed first-pass demos + 24 unkept pool episodes"
    kept = counts["fallback_kept_metrics"]; pool = counts["fallback_pool_metrics"]
    assert kept == sorted(kept, reverse=True) == sorted(pool, reverse=True)[:8]
    assert counts["fallback_metric_floor"] == pytest.approx(min(kept), abs=1e-4)  # kept metrics are rounded to 4 dp
    # whole episodes: 8 kept x 12 steps
    assert X.shape == (8 * 12, 1) and Y.shape == (8 * 12, 1)
    # the pairs really are the kept episodes' observations (metric == first obs == every obs)
    # X is float32; the recorded metrics are rounded floats -- compare with a tolerance
    assert np.allclose(sorted(set(X[:, 0].tolist())), sorted(kept), atol=1e-4)


def test_a_teacher_that_succeeds_never_falls_back_and_matches_success_exactly():
    a = collect_demos(_Env(succeed_at=3), _Teacher(), 6, 2, np.random.default_rng(1), accept="success")
    b = collect_demos(_Env(succeed_at=3), _Teacher(), 6, 2, np.random.default_rng(1), accept="success_else_top")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert b[2]["fallback_used"] is False and b[2]["accept"] == "success_else_top"
    # the fallback keys travel only with a fallback rule: absent under `success`, present
    # (and False) under `success_else_top` when the teacher succeeded
    drop = ("accept", "fallback_used", "fallback_rule")
    assert {k: v for k, v in a[2].items() if k not in drop} == {k: v for k, v in b[2].items() if k not in drop}
    assert "fallback_used" not in a[2] and b[2]["fallback_used"] is False


def test_plain_success_still_raises_with_no_successful_demo():
    with pytest.raises(RuntimeError, match="none of 4 demonstrations succeeded"):
        collect_demos(_Env(succeed_at=-1), _Teacher(), 4, 2, np.random.default_rng(0), accept="success")


def test_fallback_is_reproducible_from_the_seed():
    r1 = collect_demos(_Env(), _Teacher(), 5, 1, np.random.default_rng(7), accept="success_else_top")
    r2 = collect_demos(_Env(), _Teacher(), 5, 1, np.random.default_rng(7), accept="success_else_top")
    assert np.array_equal(r1[0], r2[0]) and r1[2]["fallback_kept_seeds"] == r2[2]["fallback_kept_seeds"]


def test_a_pool_success_is_still_cut_at_success_plus_margin():
    """Only the FIRST pass gates on success; a pool episode that happens to succeed is
    kept whole up to success + margin, like `metric_floor` does."""
    class _Late(_Env):
        # never succeeds in the first pass' 3 episodes, then succeeds at step 4 for
        # every later episode (call-counted), so the pool sees successes
        def __init__(self):
            super().__init__(-1); self.calls = 0
        def reset(self, rng):
            self.calls += 1
            self.succeed_at = 4 if self.calls > 3 else -1
            return super().reset(rng)
    env = _Late()
    X, Y, counts = collect_demos(env, _Teacher(), 3, 2, np.random.default_rng(0),
                                 accept="success_else_top", fallback_oversample=2, fallback_keep=0.5)
    assert counts["fallback_used"] and counts["fallback_kept"] == 3
    # success is first observed after the step that takes _t to 4, i.e. at t=3; the cut
    # lands when t >= 3 + margin(2), after 6 pairs have been recorded
    assert all(n == 6 for n in counts["fallback_kept_lengths"]), counts["fallback_kept_lengths"]
    assert X.shape[0] == 3 * 6


# -- success_else_best: keep sampling, capped ------------------------------------------

def test_best_keeps_sampling_until_n_demos_clear_the_floor():
    """Metric = first obs (uniform); floor 0.7 -> ~30% of episodes qualify, so the rule
    rolls ~3x n_demos and stops as soon as it has n_demos above the floor."""
    X, Y, c = collect_demos(_Env(succeed_at=-1), _Teacher(), 6, 2, np.random.default_rng(3),
                            accept="success_else_best", fallback_min_metric=0.7, fallback_max_episodes=40)
    assert c["accept"] == "success_else_best" and c["fallback_used"] and c["fallback_rule"] == "success_else_best"
    assert c["fallback_kept"] == 6 == c["demos_used"] and not c["fallback_hit_cap"]
    assert all(m > 0.7 for m in c["fallback_kept_metrics"])
    assert c["fallback_above_floor"] >= 6 and c["fallback_pool"] >= 6 and c["fallback_pool"] <= 240
    # it stopped at the first moment 6 were above the floor: the pool's last episode is above it
    assert c["fallback_pool_metrics"][-1] > 0.7
    assert X.shape == (6 * 12, 1)


def test_best_hits_the_cap_and_keeps_what_cleared_the_floor():
    class _Rare(_Env):
        # the first pass consumes resets 1-4 (none succeed); in the pool only reset #7
        # scores above 0.9, every other episode is 0.1
        def __init__(self): super().__init__(-1); self.n = 0
        def reset(self, rng):
            self.n += 1; self._v = 0.95 if self.n == 7 else 0.1; self._t = 0
            return np.array([self._v])
    X, Y, c = collect_demos(_Rare(), _Teacher(), 4, 2, np.random.default_rng(0),
                            accept="success_else_best", fallback_min_metric=0.9, fallback_max_episodes=5)
    # first pass: 4 episodes (none succeed) -> pool: cap = 5 x 4 = 20 episodes, one above the floor
    assert c["fallback_hit_cap"] is True and c["fallback_pool"] == 20 and c["fallback_kept"] == 1
    assert c["fallback_kept_metrics"] == [0.95] and c["fallback_above_floor"] == 1


def test_best_raises_when_nothing_clears_the_floor_by_the_cap():
    with pytest.raises(RuntimeError, match="none of 3 demonstrations succeeded"):
        collect_demos(_Env(succeed_at=-1), _Teacher(), 3, 2, np.random.default_rng(0),
                      accept="success_else_best", fallback_min_metric=2.0, fallback_max_episodes=2)


def test_best_is_the_success_rule_when_the_teacher_succeeds():
    a = collect_demos(_Env(succeed_at=3), _Teacher(), 6, 2, np.random.default_rng(1), accept="success")
    b = collect_demos(_Env(succeed_at=3), _Teacher(), 6, 2, np.random.default_rng(1), accept="success_else_best")
    assert np.array_equal(a[0], b[0]) and b[2]["fallback_used"] is False


# --- the config surface ------------------------------------------------------------
#
# Everything above calls `collect_demos` with kwargs. That proves the rules and not the
# keys: revert `ensure_bc_prior`'s pass-through and every test above stays green while
# `train.bc_prior.fallback_*` and `clone_eval_episodes` become declared-and-unread pins
# (a key that validates, moves the hash, and changes nothing). The tests below drive
# each key from a LOADED config to the artifact `ensure_bc_prior` writes, torch-free: the
# real `collect_demos`, `_rollout` and the `bc_prior.json` writer run; only the SB3 class,
# the fit and the registry policy are doubles, since neither `stable_baselines3` nor
# `gymnasium` is in the `test` extra and a skip here would be invisible on a PR.

import io
import json
import sys
import types

from bird import config as C
from bird.components import anchored_ppo as A
from bird.components import training as T

_BASE = {"train.init": "bc_prior", "problem.env_id": "mt10_reach-v3", "train.backend": "sb3",
         "train.algorithm": "ppo", "output.tracker": "none",
         "llm.generator.provider": "mock", "llm.evaluator.provider": "mock"}


def _dev(**over):
    return C.load("eureka", profile="dev", overrides={**_BASE, **over})


class _RollEnv(_Env):
    """`_Env` plus what `_rollout` (the clone eval) needs and `_Spaces` reads."""
    obs_low = np.array([0.0]); obs_high = np.array([1.0])

    def reference_reward(self, s, a):
        return 0.0

    def success(self, traj):
        return False

    def task_metric(self, x):
        return float(np.asarray(getattr(x, "states", x))[0, 0])


class _FakeActor:
    """Stands in for the SB3 class `_sb3_algo_and_hyper` returns: constructible the way
    `ensure_bc_prior` constructs it, `predict`s a constant, `save`s bytes."""
    def __init__(self, policy, env, seed=0, verbose=0, **hyper):
        self.hyper = hyper

    def predict(self, obs, deterministic=True):
        return np.zeros(1), None

    def save(self, buf):
        buf.write(b"fake-actor")


def _prior_record(monkeypatch, tmp_path, cfg, env=None):
    """Run `ensure_bc_prior(ctx, None, cfg)` against `env` and return what it wrote to
    `<rundir>/bc_prior.json`."""
    if "gymnasium" not in sys.modules:
        try:
            import gymnasium  # noqa: F401
        except ImportError:
            gym = types.ModuleType("gymnasium"); gym.Env = object
            gym.spaces = types.ModuleType("gymnasium.spaces")
            gym.spaces.Box = lambda lo, hi: (lo, hi)
            monkeypatch.setitem(sys.modules, "gymnasium", gym)
            monkeypatch.setitem(sys.modules, "gymnasium.spaces", gym.spaces)
    monkeypatch.setattr(T, "_sb3_algo_and_hyper", lambda cfg: (_FakeActor, {}, "none"))
    monkeypatch.setattr(A, "bc_fit", lambda model, X, Y, epochs, seed: {"final_mse": 0.0, "log_std_fit": None})
    monkeypatch.setattr("bird.policy_api.load_policy", lambda pid, **kw: _Teacher())
    T._POLICY_STORE.pop(T.BC_PRIOR_REF, None)
    ctx = types.SimpleNamespace(env=env or _RollEnv(succeed_at=-1),
                                rundir=types.SimpleNamespace(path=tmp_path))
    try:
        T.ensure_bc_prior(ctx, None, cfg)
    finally:
        T._POLICY_STORE.pop(T.BC_PRIOR_REF, None)
    return json.loads((tmp_path / "bc_prior.json").read_text())


def test_success_else_top_reads_its_pool_and_fraction_from_the_config(monkeypatch, tmp_path):
    cfg = _dev(**{"train.bc_prior.accept": "success_else_top", "train.bc_prior.n_demos": 4,
                  "train.bc_prior.fallback_oversample": 2, "train.bc_prior.fallback_keep": 0.5})
    demos = _prior_record(monkeypatch, tmp_path, cfg)["demos"]
    assert demos["fallback_used"] is True and demos["fallback_rule"] == "success_else_top"
    # 2 x 4 rolled, half kept -- neither is the module constant (4 x, a quarter)
    assert demos["fallback_oversample"] == 2 and demos["fallback_keep"] == 0.5
    assert demos["fallback_pool"] == 8 and demos["fallback_kept"] == 4 == demos["demos_used"]


def test_success_else_best_reads_its_floor_and_cap_from_the_config(monkeypatch, tmp_path):
    # a floor no episode clears, so the collector rolls to the cap: 3 x 2 = 6, not 40 x 2
    cfg = _dev(**{"train.bc_prior.accept": "success_else_best", "train.bc_prior.n_demos": 2,
                  "train.bc_prior.fallback_min_metric": 0.9, "train.bc_prior.fallback_max_episodes": 3})
    demos = _prior_record(monkeypatch, tmp_path, cfg)["demos"]
    assert demos["fallback_rule"] == "success_else_best"
    assert demos["fallback_min_metric"] == 0.9 and demos["fallback_max_episodes"] == 3
    assert demos["fallback_cap"] == 6 and demos["fallback_pool"] <= 6
    assert all(m > 0.9 for m in demos["fallback_kept_metrics"])


def test_clone_eval_episodes_rolls_exactly_that_many_and_records_them(monkeypatch, tmp_path):
    cfg = _dev(**{"train.bc_prior.clone_eval_episodes": 2, "train.bc_prior.n_demos": 2})
    rec = _prior_record(monkeypatch, tmp_path, cfg, env=_RollEnv(succeed_at=3))
    ev = rec["clone_eval"]
    assert ev["episodes"] == 2 and len(ev["task_metrics"]) == 2 == len(ev["episode_lengths"])
    assert ev["success_rate"] == 0.0 and 0.0 <= ev["task_metric_mean"] <= 1.0
    # off by default: the block is absent, not present-and-empty
    assert "clone_eval" not in _prior_record(monkeypatch, tmp_path, _dev(**{"train.bc_prior.n_demos": 2}),
                                             env=_RollEnv(succeed_at=3))


def test_fallback_keys_are_tied_to_their_rule_at_load():
    """Mirror of `accept_min_metric`'s rule: each `fallback_*` pair is required under the
    rule that reads it and refused under every other, so a value nothing reads cannot
    sit in a resolved config."""
    # required under its own rule (the defaults are null)
    with pytest.raises(C.ConfigError, match="success_else_top requires train.bc_prior.fallback_oversample"):
        _dev(**{"train.bc_prior.accept": "success_else_top"})
    with pytest.raises(C.ConfigError, match="success_else_best requires train.bc_prior.fallback_min_metric"):
        _dev(**{"train.bc_prior.accept": "success_else_best"})
    # refused under any other rule, each key by name
    for key, val in (("fallback_oversample", 2), ("fallback_keep", 0.5)):
        with pytest.raises(C.ConfigError, match=f"{key}=.* is read only under train.bc_prior.accept=success_else_top"):
            _dev(**{f"train.bc_prior.{key}": val})
        with pytest.raises(C.ConfigError, match="read only under train.bc_prior.accept=success_else_top"):
            _dev(**{"train.bc_prior.accept": "success_else_best", "train.bc_prior.fallback_min_metric": 0.0,
                    "train.bc_prior.fallback_max_episodes": 40, f"train.bc_prior.{key}": val})
    for key, val in (("fallback_min_metric", 0.0), ("fallback_max_episodes", 40)):
        with pytest.raises(C.ConfigError, match=f"{key}=.* is read only under train.bc_prior.accept=success_else_best"):
            _dev(**{f"train.bc_prior.{key}": val})
    # and the complete pairs load
    a = _dev(**{"train.bc_prior.accept": "success_else_top", "train.bc_prior.fallback_oversample": 4,
                "train.bc_prior.fallback_keep": 0.25})
    b = _dev(**{"train.bc_prior.accept": "success_else_best", "train.bc_prior.fallback_min_metric": 0.0,
                "train.bc_prior.fallback_max_episodes": 40})
    assert a["train.bc_prior.fallback_min_metric"] is None and b["train.bc_prior.fallback_oversample"] is None
