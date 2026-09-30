"""`train.init: bc_prior` and `train.anchor.*`: the schema, the coherence rules, and
the pure pieces of `bird/components/anchored_ppo.py`. The
torch-shaped half is under `importorskip` and runs where sb3 and metaworld are
installed; the rest runs everywhere."""
from __future__ import annotations

import pytest

from bird import config as C
from bird.components import anchored_ppo as A
from bird.components import training as T


def _load(name: str, **overrides):
    return C.load(name, profile="tester", overrides=overrides)


# --- the schedules --------------------------------------------------------------

def test_kl_schedule_fixed_never_reaches_zero_and_annealed_does():
    const = A.kl_schedule(5.0, 0.25, const=True)
    assert const(1.0) == 5.0 and const(0.5) == 5.0 and const(0.0) == 5.0
    ann = A.kl_schedule(5.0, 0.25)
    assert ann(1.0) == 5.0 and ann(0.875) == pytest.approx(2.5) and ann(0.75) == 0.0 and ann(0.0) == 0.0


def test_gaussian_kl_is_zero_at_identity_and_matches_the_closed_form():
    import numpy as np
    mu = np.array([[0.1, -0.3, 0.7]]); ls = np.array([[-1.0, 0.0, 0.5]])
    assert A.gaussian_kl(mu, ls, mu, ls).sum() == pytest.approx(0.0)
    # one dimension, std_c = 1, std = 2, means 1 apart: log(2) + (1 + 1)/8 - 1/2
    kl = A.gaussian_kl(np.array([[0.0]]), np.array([[0.0]]), np.array([[1.0]]), np.array([[np.log(2.0)]]))
    assert float(kl[0]) == pytest.approx(np.log(2.0) + 2.0 / 8.0 - 0.5)


def test_the_reward_form_of_the_anchor_takes_any_algorithm():
    for alg in ("sac", "td3", "ppo"):
        cfg = _load("eureka", **{"train.init": "bc_prior", "train.anchor.kind": "kl_reward",
                                 "train.algorithm": alg})
        assert cfg["train.anchor.kind"] == "kl_reward"
    with pytest.raises(C.ConfigError, match="from_scratch has none"):
        _load("eureka", **{"train.anchor.kind": "kl_reward"})


# --- the registry lookup --------------------------------------------------------

def test_auto_resolves_the_unique_runnable_policy_for_the_env():
    assert A.find_registry_policy("mt10_reach-v3") == "mt10_suite/reach"


def test_a_named_policy_must_belong_to_the_env():
    with pytest.raises(ValueError):
        A.find_registry_policy("mt10_reach-v3", "mt10_suite/pick_place")
    with pytest.raises(ValueError):
        A.find_registry_policy("mt10_reach-v3", "no_such/policy")


def test_auto_refuses_an_env_with_no_runnable_policy():
    with pytest.raises(ValueError):
        # `gym_inverted_pendulum_balance`, not `pendulum`: every pure-numpy task has a
        # runnable registry policy (policies/simple_suite), so `pendulum` does not
        # qualify as "an env with no runnable policy".
        A.find_registry_policy("gym_inverted_pendulum_balance")


# --- the config surface ---------------------------------------------------------

def test_training_init_hands_back_the_fixed_prior_ref():
    cfg = _load("eureka", **{"train.init": "bc_prior"})
    plan = T._training_init(None, None, cfg)
    ref, secondary, ratio = plan.ref, plan.secondary, plan.ratio
    assert ref == T.BC_PRIOR_REF and secondary == [] and ratio == 0.0


def test_the_anchor_refuses_a_cold_start_and_a_non_ppo_learner():
    with pytest.raises(C.ConfigError, match="from_scratch has none"):
        _load("eureka", **{"train.anchor.kind": "kl_clone"})
    with pytest.raises(C.ConfigError, match="ppo only"):
        _load("eureka", **{"train.init": "bc_prior", "train.anchor.kind": "kl_clone",
                           "train.algorithm": "sac"})
    with pytest.raises(C.ConfigError, match="target > 0"):
        _load("eureka", **{"train.init": "bc_prior", "train.anchor.kind": "kl_clone",
                           "train.algorithm": "ppo", "train.anchor.target": 0.0})


def test_the_prior_refuses_at_load_an_env_with_no_policy_on_the_backend_that_builds_it():
    with pytest.raises(C.ConfigError, match="does not resolve"):
        # `gym_inverted_pendulum_balance`: `pendulum` has a runnable policy
        # (policies/simple_suite).
        C.load("eureka", profile="dev", overrides={"problem.env_id": "gym_inverted_pendulum_balance",
                                                   "train.backend": "sb3", "train.init": "bc_prior"})
    # a named policy for the wrong env is refused the same way
    with pytest.raises(C.ConfigError, match="does not resolve"):
        C.load("eureka", profile="dev", overrides={"problem.env_id": "mt10_reach-v3",
                                                   "train.backend": "sb3", "train.init": "bc_prior",
                                                   "train.bc_prior.policy": "mt10_suite/pick_place"})
    # and the one that resolves loads
    C.load("eureka", profile="dev", overrides={"problem.env_id": "mt10_reach-v3",
                                               "train.backend": "sb3", "train.init": "bc_prior"})


def test_the_prior_refuses_a_co_designed_observation():
    with pytest.raises(C.ConfigError, match="raw observation"):
        _load("limen", **{"train.init": "bc_prior"})


def test_the_prior_prelude_degrades_off_sb3_without_touching_the_store():
    """On a backend that cannot build a clone the prelude warns and leaves the store
    alone, so `_training_init`'s reader takes the cold-start path it already has."""
    import types
    T._POLICY_STORE.pop(T.BC_PRIOR_REF, None)
    cfg = _load("eureka", **{"train.init": "bc_prior"})  # tester profile: mock backend
    ctx = types.SimpleNamespace(env=None, rundir=None)
    T.ensure_bc_prior(ctx, None, cfg)
    assert T.BC_PRIOR_REF not in T._POLICY_STORE
    assert getattr(ctx, "_bc_prior_warned", False) is True


# --- the torch half --------------------------------------------------------------

@pytest.mark.slow
def test_the_prior_clones_the_scripted_policy_and_loads_into_a_fresh_ppo():
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    pytest.importorskip("metaworld")
    import io
    import numpy as np
    from bird import registry
    registry.load_all()
    env = registry.get("env", "mt10_reach-v3")(None)
    cfg = C.load("eureka", profile="dev", overrides={
        "problem.env_id": "mt10_reach-v3", "train.backend": "sb3", "train.algorithm": "ppo",
        "train.init": "bc_prior", "train.bc_prior.n_demos": 4, "train.bc_prior.epochs": 40,
        "output.tracker": "none", "llm.generator.provider": "mock", "llm.evaluator.provider": "mock"})
    import types
    ctx = types.SimpleNamespace(env=env, rundir=None)
    T._POLICY_STORE.pop(T.BC_PRIOR_REF, None)
    T.ensure_bc_prior(ctx, None, cfg)
    blob = T._POLICY_STORE.get(T.BC_PRIOR_REF)
    assert blob is not None, "the prelude must leave the clone in the store"
    # the clone loads, exactly, into a model built the way _sb3_run builds one
    algo, hyper, anchor = T._sb3_algo_and_hyper(cfg)
    from gymnasium import spaces
    import gymnasium as gym

    class _E(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(np.asarray(env.obs_low, np.float32), np.asarray(env.obs_high, np.float32))
            self.action_space = spaces.Box(np.asarray(env.action_low, np.float32), np.asarray(env.action_high, np.float32))

        def reset(self, *, seed=None, options=None):
            return np.zeros(self.observation_space.shape, np.float32), {}

        def step(self, a):
            raise RuntimeError
    model = algo("MlpPolicy", _E(), seed=0, **hyper)   # `hyper` carries `verbose` (its default is 0)
    assert T._sb3_apply_policy(model, blob, T.BC_PRIOR_REF)
    # and acts like the scripted policy on a fresh episode: the clone must at least
    # move the hand toward the goal (reach's expert is a proportional controller)
    s = env.reset(np.random.default_rng(3))
    a, _ = model.predict(np.asarray(s, np.float32), deterministic=True)
    assert np.isfinite(a).all() and a.shape == (env.action_dim,)


def test_the_reward_form_penalises_the_state_the_action_was_chosen_in(monkeypatch):
    """`kl_reward` is r'(s, a) = r - beta * D(s): the divergence at the state the
    policy ACTED from, which is the state the loss-side form (`kl_clone`) penalises
    too. Subtracting D(s') -- the state the action led to -- would be invisible
    everywhere else: the reward stays finite, the counters stay normal, and only
    the shaping semantics move.

    The pin: an episode's reset state is the first state handed to the penalty
    under D(s), and is never handed to it under D(s'), because s' is never the
    state an episode began in. Two assertions carry it: the very first penalised
    state IS the first reset (SB3 resets the gym before its first step), and at
    least two reset states are penalised at all (256 steps on a 200-step horizon
    is two episodes). Not every reset is a gym episode -- `_evaluate_policy` resets
    the same adapter off the gym path after every chunk -- which is why the count
    is a floor rather than "every reset". The anchor is a recording double, so the
    algorithm and the clone are irrelevant here; only WHICH observation crosses
    into it is.
    """
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    import numpy as np
    from bird import registry
    from bird.budget import Budget
    from bird.context import Context
    from bird.state import RunState
    from bird.types import Candidate, CandidateReport
    from bird.components import anchored_ppo
    from bird.components.update import _carry_inner_loop_refs

    registry.load_all()
    cfg = C.load("rda", profile="dev", overrides={
        "seed": 0, "problem.env_id": "pendulum", "output.tracker": "none",
        "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
        "evaluate.rollouts_per_candidate": 1,
        "train.backend": "sb3", "train.algorithm": "ppo", "train.env_steps": 256,
        "train.hyperparameters": {"n_steps": 64, "batch_size": 32, "n_epochs": 1},
        "train.init": "warm_start_from_best", "train.anchor.kind": "kl_reward",
        "train.anchor.warmup_steps": 0})
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", "pendulum")(ctx)

    resets, penalised = [], []
    real_reset = ctx.env.reset

    def recording_reset(rng):
        s = real_reset(rng)
        resets.append(np.asarray(s, dtype=np.float32).copy())
        return s

    class _Recorder:
        def penalty(self, obs):
            penalised.append(np.asarray(obs, dtype=np.float32).copy())
            return 0.0

        def adapt(self):
            return {}

    monkeypatch.setattr(ctx.env, "reset", recording_reset)
    monkeypatch.setattr(anchored_ppo, "reward_anchor", lambda model, **kw: _Recorder())

    T._POLICY_STORE.clear()
    backend = registry.get("train_backend", "sb3")
    code = ("def compute_reward(state, action):\n"
            "    return -float(state[2] ** 2), {'v': 1.0}\n")
    first = backend(ctx, RunState(), Candidate(cand_id="c0", reward_code=code, iteration=0), 1)
    assert first.policy_ref, "iteration 0 must store the policy the anchor will hang off"
    assert not penalised, "no incumbent, so nothing to anchor to and nothing penalised"

    state = RunState()
    state.best = CandidateReport(cand_id="c0", candidate=first.candidate, result=first, fitness=1.0)
    _carry_inner_loop_refs(state)
    resets.clear()
    second = backend(ctx, state, Candidate(cand_id="c1", reward_code=code, iteration=1), 1)
    assert second.seed_metrics[0]["warm_started_from"] == first.policy_ref
    assert second.seed_metrics[0]["anchor"]["kind"] == "kl_reward"
    assert resets and penalised, "the anchored run must have started episodes and been penalised"
    assert np.allclose(penalised[0], resets[0]), (
        f"the first penalised state {penalised[0]} is not the first reset state {resets[0]}: "
        "the penalty is being computed at s' rather than at s")
    hits = sum(any(np.allclose(p, s0) for p in penalised) for s0 in resets)
    assert hits >= 2, f"only {hits} reset state(s) were ever penalised across {len(resets)} resets"


def test_the_prior_and_the_penalty_build_their_tensors_on_the_policy_device(monkeypatch):
    """`train.hyperparameters.device: cuda` puts SB3's parameters on the GPU, and a
    demonstration batch or an observation built on torch's default device would meet
    them mid-fit. The suite cannot assume a GPU, so the pin is on construction: every `as_tensor` on the two paths passes the
    policy's own device. `_device_of` reads SB3's `policy.device`, falling back to the
    first parameter's."""
    pytest.importorskip("stable_baselines3")
    th = pytest.importorskip("torch")
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class _E(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-np.ones(3, np.float32), np.ones(3, np.float32))
            self.action_space = spaces.Box(-np.ones(2, np.float32), np.ones(2, np.float32))

        def reset(self, *, seed=None, options=None):
            return np.zeros(3, np.float32), {}

        def step(self, a):
            raise RuntimeError

    model = PPO("MlpPolicy", _E(), seed=0, verbose=0, n_steps=8, batch_size=8, device="cpu")
    want = A._device_of(model.policy)
    assert want == th.device("cpu") == next(model.policy.parameters()).device

    seen = []
    real = th.as_tensor

    def spy(*args, **kw):
        seen.append(kw.get("device"))
        return real(*args, **kw)

    monkeypatch.setattr(th, "as_tensor", spy)
    X = np.random.default_rng(0).uniform(-1, 1, (16, 3)).astype(np.float32)
    Y = np.random.default_rng(1).uniform(-1, 1, (16, 2)).astype(np.float32)
    A.bc_fit(model, X, Y, epochs=1, batch=8, seed=0)
    import copy
    pen = A.AnchorPenalty(model.policy, copy.deepcopy(model.policy), beta=1.0,
                          schedule="fixed", target=0.05)
    assert pen.penalty(X[0]) >= 0.0
    assert seen, "the spy saw no tensor construction at all"
    assert all(d == want for d in seen), (
        f"a tensor was built on {[d for d in seen if d != want]} rather than on the policy's {want}")


def test_until_iteration_releases_the_anchor_after_the_first_round(monkeypatch):
    """`train.anchor.until_iteration: 1` anchors iteration 0 and lets iteration 1 train
    from the clone unanchored, so that after the first iteration the policy is free to
    explore. The seed row says which regime
    ran: `anchor` non-empty and `anchor_released` False in iteration 0; `anchor == {}`
    and `anchor_released` True in iteration 1. Same harness as the reward-form test."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from bird import registry
    from bird.budget import Budget
    from bird.context import Context
    from bird.state import RunState
    from bird.types import Candidate, CandidateReport
    from bird.components import anchored_ppo
    from bird.components.update import _carry_inner_loop_refs

    registry.load_all()
    cfg = C.load("rda", profile="dev", overrides={
        "seed": 0, "problem.env_id": "pendulum", "output.tracker": "none",
        "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
        "evaluate.rollouts_per_candidate": 1,
        "train.backend": "sb3", "train.algorithm": "ppo", "train.env_steps": 256,
        "train.hyperparameters": {"n_steps": 64, "batch_size": 32, "n_epochs": 1},
        "train.init": "warm_start_from_best", "train.anchor.kind": "kl_reward",
        "train.anchor.warmup_steps": 0, "train.anchor.until_iteration": 1})
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", "pendulum")(ctx)
    calls = []

    class _Recorder:
        def penalty(self, obs):
            calls.append(1); return 0.0

        def adapt(self):
            return {"mean_divergence": 0.0, "beta": 1.0}
    monkeypatch.setattr(anchored_ppo, "reward_anchor", lambda model, **kw: _Recorder())
    T._POLICY_STORE.clear()
    backend = registry.get("train_backend", "sb3")
    code = "def compute_reward(state, action):\n    return -float(state[2] ** 2), {'v': 1.0}\n"
    first = backend(ctx, RunState(), Candidate(cand_id="c0", reward_code=code, iteration=0), 1)
    state = RunState(); state.iteration = 0
    state.best = CandidateReport(cand_id="c0", candidate=first.candidate, result=first, fitness=1.0)
    _carry_inner_loop_refs(state)
    anchored_round = backend(ctx, state, Candidate(cand_id="c1", reward_code=code, iteration=0), 1)
    row0 = anchored_round.seed_metrics[0]
    assert row0["warm_started_from"] and row0["anchor"].get("kind") == "kl_reward" and row0["anchor_released"] is False
    assert calls, "iteration 0 must be anchored (the penalty was never consulted)"
    calls.clear()
    state.best = CandidateReport(cand_id="c1", candidate=anchored_round.candidate, result=anchored_round, fitness=1.0)
    _carry_inner_loop_refs(state)
    state.iteration = 1
    free_round = backend(ctx, state, Candidate(cand_id="c2", reward_code=code, iteration=1), 1)
    row1 = free_round.seed_metrics[0]
    assert row1["warm_started_from"], "iteration 1 still STARTS from the carried policy"
    assert row1["anchor"] == {} and row1["anchor_released"] is True
    assert not calls, "iteration 1 must train unanchored (the penalty was consulted)"


def _stub_env_and_policy():
    """A minimal stub env and policy, plus a `task_metric` the `metric_floor` rule can read: the fraction of
    visited states past the success step (0 for an episode that never succeeds)."""
    import numpy as np

    class _Env:
        horizon = 6
        action_low = np.array([-1.0]); action_high = np.array([1.0])

        def __init__(self, bar, succeed_at=None):
            self.success_threshold = bar; self.succeed_at = succeed_at

        def reset(self, rng):
            return np.zeros(2)

        def step(self, s, a):
            t = int(s[0]) + 1
            info = {"success": 1.0} if (self.succeed_at is not None and t >= self.succeed_at) else {}
            return np.array([t, 0.0]), False, info

        def task_metric(self, states):
            states = np.asarray(states)
            if self.succeed_at is None:
                return 0.0
            return float(np.mean(states[:, 0] >= self.succeed_at))

    class _Pol:
        def reset(self, rng): pass

        def act(self, s, *, t, env=None): return np.array([0.5])

    return _Env, _Pol


def test_accept_any_keeps_every_episode_whole_even_on_an_env_with_a_bar():
    """`train.bc_prior.accept: any` -- a base policy registered as partial/negative
    never (or rarely) satisfies success(); under `success` the prior raises on it, under
    `any` every episode is cloned, whole, and the counts say the gate was off."""
    import numpy as np
    from bird.components.anchored_ppo import collect_demos
    _Env, _Pol = _stub_env_and_policy()
    X, Y, c = collect_demos(_Env(0.5, succeed_at=None), _Pol(), 3, 1, np.random.default_rng(0), accept="any")
    assert c["demos_used"] == 3 and c["demos_skipped"] == 0 and c["pairs"] == 18
    assert c["success_gated"] is False and c["accept"] == "any" and c["accept_min_metric"] is None
    # a succeeding episode is NOT truncated under `any` either (nothing is cut)
    X, Y, c = collect_demos(_Env(0.5, succeed_at=2), _Pol(), 3, 1, np.random.default_rng(0), accept="any")
    assert c["pairs"] == 18, c


def test_accept_metric_floor_keeps_by_the_adapters_metric_and_still_truncates_at_success():
    """`metric_floor`: the adapter's own task_metric over the episode's states decides; a
    kept episode that succeeds is still cut at success + margin (the hold-phase argument)."""
    import numpy as np
    from bird.components.anchored_ppo import collect_demos
    _Env, _Pol = _stub_env_and_policy()
    # succeed_at=2 on a 6-step episode: the FULL episode's states 0..6 are judged (5 of
    # 7 past the success step -> 0.714), then the kept pairs are cut at success + 1
    X, Y, c = collect_demos(_Env(0.5, succeed_at=2), _Pol(), 3, 1, np.random.default_rng(0),
                            accept="metric_floor", min_metric=0.5)
    assert c["demos_used"] == 3 and c["pairs"] == 9 and c["accept_min_metric"] == 0.5, c   # truncated at success + 1
    # The prefix and the whole episode must DISAGREE about the floor for this to guard
    # against judging the prefix: succeed_at=3 -> first success at t=2, the cut
    # at t=3 leaves the prefix states 0..4 (2 of 5 past the success step = 0.40) while
    # the whole episode's states 0..6 give 4/7 = 0.571. A floor of 0.5 must KEEP it --
    # judging the prefix dropped it -- and the kept pairs are still cut at success + margin
    X, Y, c = collect_demos(_Env(0.5, succeed_at=3), _Pol(), 2, 1, np.random.default_rng(0),
                            accept="metric_floor", min_metric=0.5)
    assert c["demos_used"] == 2 and c["pairs"] == 8, c
    # a metric that is not a number is not over any floor
    class _NanEnv(_Env):
        def task_metric(self, states): return float("nan")
    with pytest.raises(RuntimeError, match="reached task_metric"):
        collect_demos(_NanEnv(0.5, succeed_at=2), _Pol(), 2, 1, np.random.default_rng(0),
                      accept="metric_floor", min_metric=0.0)
    # a floor above what the episode reaches drops it; the error names the rule
    with pytest.raises(RuntimeError, match="reached task_metric >= 0.9"):
        collect_demos(_Env(0.5, succeed_at=2), _Pol(), 3, 1, np.random.default_rng(0),
                      accept="metric_floor", min_metric=0.9)
    # a floor of 0.0 keeps a never-succeeding episode, whole
    X, Y, c = collect_demos(_Env(0.5, succeed_at=None), _Pol(), 2, 1, np.random.default_rng(0),
                            accept="metric_floor", min_metric=0.0)
    assert c["demos_used"] == 2 and c["pairs"] == 12
    # the rule set is closed, and the floor is required by the rule that reads it
    with pytest.raises(ValueError, match="accept must be one of"):
        collect_demos(_Env(0.5), _Pol(), 1, 1, np.random.default_rng(0), accept="best_k")
    with pytest.raises(ValueError, match="needs min_metric"):
        collect_demos(_Env(0.5), _Pol(), 1, 1, np.random.default_rng(0), accept="metric_floor")


def test_accept_min_metric_is_tied_to_its_rule_at_load():
    """`accept_min_metric` set under any rule but `metric_floor` is a declared-but-unread
    pin; `metric_floor` without it has no floor to apply. Both are refused at load."""
    base = {"train.init": "bc_prior", "problem.env_id": "mt10_reach-v3", "train.backend": "sb3",
            "train.algorithm": "ppo", "output.tracker": "none",
            "llm.generator.provider": "mock", "llm.evaluator.provider": "mock"}
    with pytest.raises(C.ConfigError, match="requires train.bc_prior.accept_min_metric"):
        C.load("eureka", profile="dev", overrides={**base, "train.bc_prior.accept": "metric_floor"})
    with pytest.raises(C.ConfigError, match="read only under train.bc_prior.accept=metric_floor"):
        C.load("eureka", profile="dev", overrides={**base, "train.bc_prior.accept_min_metric": 0.3})
    cfg = C.load("eureka", profile="dev", overrides={**base, "train.bc_prior.accept": "metric_floor",
                                                      "train.bc_prior.accept_min_metric": 0.3})
    assert cfg["train.bc_prior.accept"] == "metric_floor"


def test_collect_demos_keeps_whole_episodes_on_an_env_with_no_success_bar():
    """The gym MuJoCo adapters declare `success_threshold = inf` and never report a
    success, so a demonstration gate keyed on success would drop all 25 demos of every
    gym MuJoCo policy and `bc_prior` would raise. With no bar, the whole episode is
    the demonstration; with a finite bar the gate is unchanged, failures included."""
    import numpy as np
    from bird.components.anchored_ppo import collect_demos

    class _Env:
        horizon = 6
        action_low = np.array([-1.0]); action_high = np.array([1.0])

        def __init__(self, bar, succeed_at=None):
            self.success_threshold = bar; self.succeed_at = succeed_at

        def reset(self, rng):
            return np.zeros(2)

        def step(self, s, a):
            t = int(s[0]) + 1
            info = {"success": 1.0} if (self.succeed_at is not None and t >= self.succeed_at) else {}
            return np.array([t, 0.0]), False, info

    class _Pol:
        def reset(self, rng): pass

        def act(self, s, *, t, env=None): return np.array([0.5])

    X, Y, c = collect_demos(_Env(float("inf")), _Pol(), 3, 1, np.random.default_rng(0))
    assert c == {"demos_used": 3, "demos_skipped": 0, "pairs": 18, "success_gated": False,
                 "accept": "success", "accept_min_metric": None}
    assert len(X) == 18 and float(Y.mean()) == 0.5
    X, Y, c = collect_demos(_Env(0.5, succeed_at=2), _Pol(), 3, 1, np.random.default_rng(0))
    assert c["success_gated"] is True and c["demos_used"] == 3 and c["pairs"] == 9, c   # truncated at success + 1
    with pytest.raises(RuntimeError, match="none of 3 demonstrations succeeded"):
        collect_demos(_Env(0.5, succeed_at=None), _Pol(), 3, 1, np.random.default_rng(0))


def test_the_adaptive_controller_never_goes_below_beta_min():
    """`train.anchor.beta_min` floors the adaptive coefficient. Both anchor forms take
    their controller step from ONE function, `adapt_beta` (KLPPO.train and
    AnchorPenalty.adapt both call it), so the floor is checked on that function and
    then through the reward form's object. Unfloored, beta was measured decaying 5.0 ->
    ~0.25 on MT10 while divergence sat under target, after which designed rewards pulled
    the clone away; a floor of 0.5 holds it."""
    import types
    beta = 5.0
    for _ in range(1000):
        beta = A.adapt_beta(beta, 0.0, 0.05, 0.1, 0.5)      # divergence 0: the rule shrinks beta
    assert beta == pytest.approx(0.5)
    beta = 5.0
    for _ in range(1000):
        beta = A.adapt_beta(beta, 0.0, 0.05, 0.1, 1e-3)
    assert beta == pytest.approx(1e-3), "the default floor is the code's old hard-coded 1e-3"
    assert A.adapt_beta(1.0, 1.0, 0.05, 0.1, 0.5) == pytest.approx(1.02), "divergence over target grows beta by at most 2%"
    pen = A.AnchorPenalty(types.SimpleNamespace(), types.SimpleNamespace(), beta=5.0, schedule="adaptive",
                          target=0.05, beta_min=0.5)
    for _ in range(1000):
        pen._sum, pen._n = 0.0, 10
        pen.adapt()
    assert pen.beta == pytest.approx(0.5)
