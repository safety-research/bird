"""`verification.sample_transitions` must actually reach the env's sampler.

This is the file for one specific silent failure: the EPIC / STARC /
`policy_rank_corr` screens are pseudometrics between reward *functions*, so they
need a distribution over transitions. `sample_transitions` probes for the
richest interface the adapter offers and degrades to Gaussian draws. A ladder
whose every rung swallows its exception hides the FIRST rung failing to bind, so
the first rung does not swallow: a sampler that raises or returns nothing is
`SamplerFailed`, loud (the tests at the foot of this file), and the ladder's
lower rungs serve only an env that has no sampler at all. Every adapter in
`bird/envs/` inherits `EnvAdapter.sample_transitions(self, rng, n)`; a
`_call_flexible` that assumed `n` came first and passed the rng by keyword would
emit `fn(n, rng=nprng)` and get `TypeError: got multiple values for argument
'rng'` on all of them.

On the toy and control envs the consequence would be a screen scoring
off-distribution noise. On Meta-World it is worse and not a degradation at all:
a 39-D Gaussian is not a state the simulator ever emitted, so `reference_reward` raises
`UnknownStateError` on every draw and the screen reports itself inert --
`verify.alignment_filter.keep_top_n` never applies and the method point is not
reachable.
"""

import numpy as np
import pytest

from bird.components import verification


class _Ctx:
    def __init__(self, env):
        self.env = env
        self.rng = __import__("random").Random(0)


def _adapters():
    from bird import registry
    from conftest import env_param

    registry.load_all()
    # `env_param` marks the HumanoidBench ids so `-m "not humanoid"` DESELECTS
    # them rather than reaching the `pytest.skip` below: a skip there could never be
    # removed by any single install, since no interpreter can hold both
    # HumanoidBench and metaworld.
    # Meta-World's fifty (`mt10_`/`mt50_`) share one adapter and one sampler; the
    # module tests it once below rather than fifty times.
    return [env_param(n) for n in registry.names("env")
            if not n.startswith(("mt10_", "mt50_"))]


@pytest.mark.parametrize("env_name", _adapters())
def test_the_env_sampler_is_reached_and_not_the_gaussian_fallback(env_name, caplog):
    """The fallback is legitimate and must stay; silently landing on it is not.

    Asserted through the log the fallback itself emits, because the shapes of the
    two rungs agree by construction -- the synthetic draws are shaped from the
    adapter's advertised dimensions -- so a value check cannot tell them apart.
    """
    from bird import registry

    try:
        env = registry.get("env", env_name)({})
    except ImportError as exc:
        # An adapter that needs a simulator this install does not have. CI runs
        # `--extra test` (pyyaml + numpy + pytest), so `gym_inverted_pendulum_balance`
        # cannot be constructed there and the parametrisation above -- which filters by NAME --
        # cannot know that. Skipping is the honest outcome: the sampler contract is
        # still asserted for it on any install that can build the env, and a hard
        # failure here would only report a missing optional dependency as a broken
        # screen. `mt10_*` is excluded by name above instead of reaching this
        # branch; both mean the same thing.
        pytest.skip(f"{env_name}: adapter needs an optional simulator ({exc})")
    with caplog.at_level("DEBUG", logger="bird"):
        transitions = verification.sample_transitions(_Ctx(env), 5)

    assert len(transitions) == 5
    assert "env.sample_transitions failed" not in caplog.text, (
        f"{env_name}: the harness could not call EnvAdapter.sample_transitions, so "
        "the screens are running on synthetic draws:\n" + caplog.text)


def test_call_flexible_binds_the_base_signature_by_name():
    """`(rng, n)` is the order the base class uses; `(n, rng)` is what callers
    outside this repo tend to write. Both have to work, or the ladder in
    `sample_transitions` silently drops a rung."""
    seen = {}

    def rng_first(rng, n):
        seen["shape"] = ("rng_first", n, rng)
        return []

    def n_first(n, rng=None):
        seen["shape"] = ("n_first", n, rng)
        return []

    def count_only(n):
        seen["shape"] = ("count_only", n, None)
        return []

    def positional_pair(a, b):
        seen["shape"] = ("positional_pair", a, b)
        return []

    nprng = np.random.default_rng(0)
    for fn, tag in ((rng_first, "rng_first"), (n_first, "n_first"),
                    (count_only, "count_only"), (positional_pair, "positional_pair")):
        verification._call_flexible(fn, 7, nprng)
        assert seen["shape"][0] == tag
        assert seen["shape"][1] == 7, (
            f"{tag}: the count landed in the wrong parameter")
        if tag in ("rng_first", "n_first", "positional_pair"):
            assert seen["shape"][2] is nprng, (
                f"{tag}: the rng did not arrive, so the sample is not reproducible")


# -- a sampler that fails is refused, never degraded ----------------------------


class _Broken:
    """An `EnvAdapter`-shaped env whose sampler raises: state-passing `step`, so the
    gym-shaped rollout rung cannot even call it, and dims the synthetic rung would
    happily use. A swallowing ladder would swallow the raise, try the rollout
    (TypeError: `step() missing 1 required positional argument: 'action'`, also
    swallowed) and score the screens on Gaussian draws."""

    obs_dim = 6
    action_dim = 2

    def sample_transitions(self, rng, n):
        raise RuntimeError("simulator not attached")

    def reset(self, rng=None):
        return np.zeros(6)

    def step(self, state, action):
        return np.zeros(6), False, {}


class _Empty(_Broken):
    def sample_transitions(self, rng, n):
        return []


class _NoSampler:
    """No sampler at all: the documented ladder still applies (rollout, then
    synthetic draws), because GT's validity check on random draws is a faithful
    fallback for an env that offers nothing better."""

    obs_dim = 3
    action_dim = 1


def test_a_sampler_that_raises_is_refused_loudly_not_degraded():
    with pytest.raises(verification.SamplerFailed, match=r"_Broken\.sample_transitions failed "
                                                         r"\(RuntimeError: simulator not attached\)"):
        verification.sample_transitions(_Ctx(_Broken()), 8)


def test_a_sampler_that_returns_nothing_is_refused_too():
    with pytest.raises(verification.SamplerFailed, match=r"returned no transitions for n=5"):
        verification.sample_transitions(_Ctx(_Empty()), 5)


def test_an_env_with_no_sampler_still_gets_the_documented_fallback():
    out = verification.sample_transitions(_Ctx(_NoSampler()), 4)
    assert len(out) == 4 and all(np.asarray(t.state).shape == (3,) for t in out)
    assert all(np.asarray(t.action).shape == (1,) for t in out)


def test_the_refusal_reaches_the_dynamic_checks_as_an_error_not_a_verdict():
    """A harness fault propagates out of `execution_smoke`; it is not filed against
    the candidate as `invalid`, which would blame the program for the sampler."""
    from types import SimpleNamespace

    from bird import registry
    from bird.types import Candidate

    registry.load_all()
    cfg = {"verify.smoke_test_steps": 3, "verify.timeout_s": 60,
           "generate.output.format": "component_dict_return", "generate.reward_language": "numpy"}
    ctx = SimpleNamespace(cfg=SimpleNamespace(get=lambda k, d=None: cfg.get(k, d)),
                          env=_Broken(), rng=__import__("random").Random(0))
    cand = Candidate(cand_id="c", iteration=0,
                     reward_code="def compute_reward(state, action=None, next_state=None):\n    return 0.0, {}\n")
    fn, err = verification.compile_reward(ctx, cand)
    assert fn is not None, err
    with pytest.raises(verification.SamplerFailed):
        registry.get("dynamic_check", "execution_smoke")(ctx, cand, fn)
