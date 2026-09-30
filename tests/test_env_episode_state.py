"""`EnvAdapter.episode_state` / `restore_episode_state`: interleaving N episodes
on ONE adapter must equal N adapters, bit for bit.

`step(state, action)` is state-passing, but `_step` also reads what `reset()`
left on the INSTANCE -- the episode RNG (`action_noise`, `slip_prob`, ...) and
the domain-randomisation draw (`_dr_now`). `components.fasttd3._VecEnvView`
runs `num_envs` episode slots on `ctx.env` and relies on this pair of hooks to
give each slot its own stream and draw rather than constructing `num_envs`
adapters (on HumanoidBench a construction is a MuJoCo model plus an offscreen
GL context). The hooks are only as honest as the adapters that implement
them, so this file runs the contract against every env the toy and
classic-control modules register -- picked off the registry, so a new cheap
adapter is covered without being named here -- under the WIDEST DR range each
declares, with a foreign `reset()` thrown in mid-run the way a between-chunk
evaluation does it. The MuJoCo-backed tiers are not constructible in CI, and
each is covered for its own reason: `metaworld`'s `_step` restores the
simulator from the state's episode snapshot, which holds every model field its
DR writes, so there the draw travels with the state; HumanoidBench has no draw
to restore -- every h1hand/h1strong spec says `domain_randomization: null` and
neither `humanoid.py` nor `humanoid_hand.py` reads `_dr_now` -- so the base
pair is the whole of its instance state. A per-episode quantity an adapter
derives from the draw is best read off `_dr_now` at step time rather than
cached as an attribute in `_reset`, so the base pair covers it too.

The control at the bottom is what makes the equality mean something: the
SAME interleave with the restore left out diverges on the envs whose dynamics
read the RNG or the draw, so a hook that silently restored nothing would fail
here rather than pass by coincidence.
"""

import numpy as np
import pytest

from conftest import REPO  # noqa: F401  (path setup)
from bird import registry
from bird.budget import Budget
from bird.config import load
from bird.context import Context

registry.load_all()
CHEAP_MODULES = ("bird.envs.toy", "bird.envs.control")
CHEAP = sorted(n for n in registry.names("env")
               if getattr(registry.get("env", n), "__module__", "") in CHEAP_MODULES)
assert len(CHEAP) >= 6, CHEAP   # toy_{reacher,gridworld,hungry_thirsty}, pendulum{,_discrete}, acrobot

SEED = 7
T = 40


def _make(env_id):
    cfg = load("eureka", overrides={"problem.env_id": env_id, "output.tracker": "none",
                                    "llm.generator.provider": "mock",
                                    "llm.evaluator.provider": "mock"}, profile="tester")
    env = registry.get("env", env_id)(Context(cfg=cfg, budget=Budget()))
    # the widest range the adapter declares, on every axis: the draw has to
    # be able to matter for the equality to be a statement about the draw
    try:
        env.set_dr({k: [float(lo), float(hi)] for k, (lo, hi) in env.dr_parameters.items()})
    except TypeError:
        # `Acrobot.set_dr` takes point values, not ranges, and its `_sample_dr`
        # draws the whole `_DR` table on every reset whatever was installed --
        # so its draw varies per episode here regardless.
        pass
    return env


def _actions(env, n_slots):
    g = np.random.default_rng(1234)
    if getattr(env, "exact_states", None) is not None:
        return [[env.action_set[int(g.integers(env.n_actions))].copy() for _ in range(n_slots)]
                for _ in range(T)]
    return [[g.uniform(env.action_low, env.action_high) for _ in range(n_slots)] for _ in range(T)]


def _isolated(env_id, actions, n_slots=2):
    """The reference: one adapter PER slot, stepped in lockstep."""
    envs = [_make(env_id) for _ in range(n_slots)]
    ep = [0] * n_slots
    s = [np.asarray(e.reset(np.random.default_rng((SEED, i, 0))), dtype=float)
         for i, e in enumerate(envs)]
    out = []
    for t in range(T):
        for i, e in enumerate(envs):
            s2, done, _ = e.step(s[i], actions[t][i])
            out.append(np.asarray(s2, dtype=float))
            if done:
                ep[i] += 1
                s2 = e.reset(np.random.default_rng((SEED, i, ep[i])))
            s[i] = np.asarray(s2, dtype=float)
    return out


def _shared(env_id, actions, restore=True, n_slots=2):
    """One adapter, N slots; `restore` is the hook under test."""
    env = _make(env_id)
    ep = [0] * n_slots
    s, blob = [None] * n_slots, [None] * n_slots
    for i in range(n_slots):
        s[i] = np.asarray(env.reset(np.random.default_rng((SEED, i, 0))), dtype=float)
        blob[i] = env.episode_state()
    out = []
    for t in range(T):
        if t == T // 2:
            env.reset(np.random.default_rng(99))       # a between-chunk evaluation's reset
        for i in range(n_slots):
            if restore:
                env.restore_episode_state(blob[i])
            s2, done, _ = env.step(s[i], actions[t][i])
            blob[i] = env.episode_state()
            out.append(np.asarray(s2, dtype=float))
            if done:
                ep[i] += 1
                s2 = env.reset(np.random.default_rng((SEED, i, ep[i])))
                blob[i] = env.episode_state()
            s[i] = np.asarray(s2, dtype=float)
    return out


@pytest.mark.parametrize("env_id", CHEAP)
def test_interleaving_episodes_on_one_adapter_equals_one_adapter_per_episode(env_id):
    actions = _actions(_make(env_id), 2)
    ref = _isolated(env_id, actions)
    got = _shared(env_id, actions, restore=True)
    assert len(ref) == len(got)
    for k, (a, b) in enumerate(zip(ref, got)):
        assert np.array_equal(a, b), f"{env_id}: transition {k} differs\n{a}\n{b}"


def test_the_equality_is_not_a_coincidence_of_the_probe():
    """Drop the restore and the interleave diverges on at least one covered env --
    on the envs whose `_step` reads the RNG or the draw. If this stops
    failing, the probe above cannot see the leak it exists to catch."""
    diverged = []
    for env_id in CHEAP:
        actions = _actions(_make(env_id), 2)
        ref = _isolated(env_id, actions)
        naive = _shared(env_id, actions, restore=False)
        if any(not np.array_equal(a, b) for a, b in zip(ref, naive)):
            diverged.append(env_id)
    assert "pendulum" in diverged and "toy_reacher" in diverged, diverged


def test_the_base_pair_is_the_rng_and_the_draw():
    """What the base hooks carry, stated: the live `Generator` (by reference,
    so a step advances the blob's stream) and the draw dict."""
    env = _make("pendulum")
    env.reset(np.random.default_rng(3))
    rng, draw = env.episode_state()
    assert rng is env._rng and draw is env._dr_now
    other = _make("pendulum")
    other.reset(np.random.default_rng(4))
    other.restore_episode_state((rng, draw))
    assert other._rng is rng and other._dr_now is draw
