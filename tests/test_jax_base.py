"""The jax tier's skeleton: the key, the id rule, the suite row, the adapter.

These tests deliberately do NOT need jax. The one
property that matters most here is that importing the tier costs a machine
with no GPU nothing at all, and a test that skipped without jax could not
assert that -- it would skip on exactly the machine the claim is about.

The cases that WOULD need a device (a real `step_batch`, a DLPack hand-off,
a jitted reward) carry the `jax` marker and skip in the default CI job:
`--extra test` installs no jax, because the `jax` extra is ~3 GB of CUDA
wheels a CPU runner would download to run nothing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from bird.config import Config, ConfigError, load, validate
from bird.envs.base import EnvAdapter
from bird.envs.jax_base import BatchedEnvAdapter
from bird.envs.suites import env_suite

REPO = Path(__file__).resolve().parents[1]


# -- the property the whole tier rests on -----------------------------------


def test_importing_the_adapter_does_not_import_jax():
    """`registry.load_all()` imports every component module on every run.

    On a laptop, in CI, and inside a cluster worker that may hold no GPU. An
    import of jax at module scope here would make a ~3 GB optional extra a
    hard dependency of `bird.py --list-configs`, and the failure would be an
    ImportError at registry load rather than an actionable error in the one
    code path that needs it -- which is the rule `pyproject.toml`'s extras
    comment states for every other extra.

    A SUBPROCESS, because by the time this file runs, another test may
    already have imported jax for its own reasons; asking `sys.modules` in
    THIS process would answer a question about the suite rather than about
    the module.
    """
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            import bird.envs.jax_base  # noqa: F401
            bad = sorted(m for m in sys.modules
                         if m == "jax" or m.startswith("jax."))
            print(",".join(bad))
        """)],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", (
        f"importing bird.envs.jax_base pulled in {out.stdout.strip()}; the "
        f"jax extra is optional and load_all() imports this module on every "
        f"run, including on machines that will never have a GPU")


def test_the_adapter_is_an_EnvAdapter():
    """The trainer selects the batched path with `isinstance`, not a config flag.

    The capability belongs to the adapter; the science knob is the env id.
    If this ever stopped being a subclass, every eval, screen, trace and
    video path would need a second implementation.
    """
    assert issubclass(BatchedEnvAdapter, EnvAdapter)
    assert BatchedEnvAdapter.physics == "mjx"


def test_a_subclass_missing_either_device_method_cannot_be_built():
    """A skeleton must fail loudly, and as EARLY as it can.

    Both are `@abstractmethod`, so the failure is a TypeError from the
    CONSTRUCTOR naming the method -- not, as is easy to assume, an error at
    class definition. That distinction is why the two stubs below can still
    subclass to exercise one half of the pair.

    The alternative, a bare `raise NotImplementedError` as `EnvAdapter` uses
    for its own five, fails on the first transition instead: for this family
    that is inside a training, after a generate call and a jit compile on a
    charged GPU.
    """
    class _NoStep(BatchedEnvAdapter):
        def reset_batch(self, key, n):
            return None

    class _NoReset(BatchedEnvAdapter):
        def step_batch(self, state, action, physics):
            return None, None, {}

    with pytest.raises(TypeError, match="step_batch"):
        _NoStep()
    with pytest.raises(TypeError, match="reset_batch"):
        _NoReset()


def test_the_single_env_bridge_goes_through_the_batched_path():
    """`_step` must not have its own implementation.

    Two code paths for one transition is how a batched answer and a
    single-env answer drift, and the single-env one is what every eval,
    screen and trace reads -- so the drift would show up as a fitness that
    disagrees with the training curve for no visible reason.

    Exercised with a stub `step_batch` and NO JAX: `_step` converts with
    `np.asarray` rather than `jnp.asarray` -- a jitted `step_batch` transfers
    a host array on call anyway, so the two cost the same one transfer and
    the numpy spelling is what lets this case run in an ordinary CI job
    instead of a `jax`-marked one that skips there (no jax installed).
    """
    import numpy as np

    seen = {}

    class _Stub(BatchedEnvAdapter):
        # `EnvAdapter.__init__` calls both of these, so a stub that skipped
        # them would fail in the CONSTRUCTOR and never reach the bridge.
        def _build_action_set(self):
            return [[0.0]]

        def _bounds(self):
            return np.full(2, -1.0), np.full(2, 1.0)

        # `source_parents` and `random_state` are ABSTRACT on the ABC, so a
        # double must say too -- that is the intended cost of making them
        # abstract rather than defaulting them. A default would let this stub
        # stay silent, which is exactly how a delegating adapter can render
        # subclass-only and pass the leak guard while checking nothing.
        def source_parents(self):
            return ()

        def random_state(self, rng):
            return np.zeros(2)

        def reset_batch(self, key, n):
            return np.zeros((n, 2), dtype=np.float32)

        def step_batch(self, state, action, physics):
            seen["shape"] = np.asarray(state).shape
            s2 = np.asarray(state, dtype=np.float32) + 1.0
            return s2, np.array([True]), {"time_out": np.array([False])}

    s2, done, info = _Stub()._step(np.array([1.0, 2.0]), np.array([0.0]))
    assert seen["shape"] == (1, 2), "step_batch was not given a leading axis"
    assert s2.dtype == np.float64, (
        "the bridge returned float32; every numpy consumer in tasks/ is "
        "float64 and silently narrowing moves the last digits of a metric")
    assert done is True and info["time_out"].shape == ()


# -- the env family ---------------------------------------------------------


@pytest.mark.parametrize("env_id", ["jax_toy", "upstream_assistax_feeding",
                                    "upstream_assistax_scratchitch"])
def test_a_jax_id_lands_in_the_jax_suite(env_id):
    """Both prefixes of the device-batched family resolve to the `jax` suite:
    a family that landed in `other` would silently under-report wherever runs
    are grouped by suite."""
    assert env_suite(env_id) == "jax"


def test_the_cpu_ids_are_untouched():
    """The prefix separates two SOLVERS, so the pair must not merge."""
    assert env_suite("gym_half_cheetah") == "gym_mujoco"
    assert env_suite("assistax_feeding") == "assistax"


# -- the reward language ----------------------------------------------------


def test_the_default_is_numpy_and_every_shipped_config_keeps_it():
    """None of the published configs is a jax run."""
    assert load("eureka").get("generate.reward_language") == "numpy"


def _problems(env_id, lang):
    cfg = Config({"problem": {"env_id": env_id},
                  "generate": {"reward_language": lang}})
    try:
        validate(cfg)
    except ConfigError as exc:
        return [l for l in str(exc).splitlines() if "reward_language" in l]
    return []


def test_a_jax_env_with_a_numpy_reward_is_refused():
    """Refused, not downgraded.

    A numpy reward on a device batch would be evaluated row by row on the
    host -- the batching this family exists for, silently undone. Every
    counter would read normal and the only symptom is a run that is
    inexplicably slow -- the same silent shape as a GPU job training on a CPU.
    """
    assert _problems("jax_toy", "numpy")


def test_a_numpy_env_with_a_jax_reward_is_refused():
    """Nothing would run it: the numpy backends compile with the numpy
    namespace, so a jnp body either fails to import or computes on host
    arrays under a name that says device."""
    assert _problems("gym_half_cheetah", "jax")


@pytest.mark.parametrize("env_id,lang", [("jax_toy", "jax"),
                                         ("gym_half_cheetah", "numpy")])
def test_the_matching_pairs_are_clean(env_id, lang):
    """The pair. A rule that refused everything would pass both tests above."""
    assert _problems(env_id, lang) == []


def test_the_refusals_are_scoped_through_the_suite_table_not_a_prefix_test():
    """A second copy of the prefix list drifts from the suite table.

    Read off the rule's CODE with comments stripped, and that detail is the
    point: a probe over the raw source is failed by the rule's own comment,
    which contains the sentence
    `never an id.startswith("jax_") here`. A guard that a promise NOT to do
    something trips is a guard reading prose, not behaviour.
    """
    import inspect

    from bird import config as config_mod

    lines = inspect.getsource(config_mod._check_coherence).splitlines()
    start = next(n for n, l in enumerate(lines)
                 if l.strip().startswith('lang = g("generate.reward_language")'))
    block = "\n".join(l.split("#", 1)[0] for l in lines[start:start + 20])

    assert "env_suite(" in block, "the rule no longer reads the suite table"
    assert "startswith(" not in block, (
        "the rule grew its own prefix test; `envs.suites.env_suite` is the "
        "one place the `jax_` prefix lives, and two copies would diverge")


# -- physics is REQUIRED, and the bridge passes it ----------------------------


def test_step_batch_takes_physics_as_a_required_third_positional_on_every_adapter():
    """`step_batch(state, action, physics)`: three positionals, no default, on the
    abstract and on every concrete adapter.
    A default would let a caller omit the draw and step under whichever slot reset
    last; the signature is the guard, checked by `inspect` so a new adapter cannot
    quietly reintroduce one. Classes are found by subclassing, which needs no jax:
    `registry.load_all()` imports every adapter module without constructing."""
    import inspect

    from bird import registry

    registry.load_all()

    def concrete(cls):
        out = []
        for sub in cls.__subclasses__():
            out.append(sub)
            out.extend(concrete(sub))
        return out

    classes = [BatchedEnvAdapter] + [c for c in concrete(BatchedEnvAdapter)
                                     if not c.__name__.startswith("_")]
    assert any(c.__name__ == "JaxToyReacher" for c in classes), "the toy is the first concrete adapter"
    for cls in classes:
        params = [p for p in inspect.signature(cls.step_batch).parameters.values()
                  if p.name != "self"]
        names = [p.name for p in params]
        assert names[:3] == ["state", "action", "physics"] and len(params) == 3, (
            f"{cls.__name__}.step_batch{inspect.signature(cls.step_batch)}: three positionals "
            "(state, action, physics) and nothing else")
        for p in params:
            assert p.default is inspect.Parameter.empty, (
                f"{cls.__name__}.step_batch: `{p.name}` has a default; the draw is required")
            assert p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD), (cls.__name__, p.name)


def test_the_bridge_passes_the_blobs_draw_as_a_non_none_pytree():
    """Bridge side: on a DR-capable adapter, `_step` hands `step_batch` the blob's
    draw as `physics_rows(1)` -- a non-None pytree of one-row arrays keyed like
    `dr_parameters` -- and never leaves the argument to the adapter. Exercised with
    a recording double and no jax."""
    import numpy as np

    seen = {}

    class _Double(BatchedEnvAdapter):
        dr_parameters = {"mass": (0.3, 3.0), "damping": (0.2, 0.9)}
        _dr_nominal = {"mass": 1.0, "damping": 0.5}

        def _build_action_set(self):
            return [[0.0]]

        def _bounds(self):
            return np.full(2, -1.0), np.full(2, 1.0)

        def source_parents(self):
            return ()

        def random_state(self, rng):
            return np.zeros(2)

        def reset_batch(self, key, n):
            return np.zeros((n, 2), dtype=np.float32)

        def step_batch(self, state, action, physics):
            seen["physics"] = physics
            return (np.asarray(state, dtype=np.float32) + 1.0, np.array([False]),
                    {"success": np.array([False])})

    env = _Double()
    env._dr_now = {"mass": 2.0, "damping": 0.3}          # the episode's draw, as reset() would set it
    env._step(np.array([0.0, 0.0]), np.array([0.0]))
    physics = seen["physics"]
    assert physics is not None and set(physics) == {"mass", "damping"}
    assert physics["mass"].shape == (1,) and float(physics["mass"][0]) == 2.0
    assert float(physics["damping"][0]) == pytest.approx(0.3)  # float32 rows
    # ... and restoring another episode's blob changes what the next call receives
    blob = env.episode_state()
    env._dr_now = {"mass": 0.5, "damping": 0.9}
    env._step(np.array([0.0, 0.0]), np.array([0.0]))
    assert float(seen["physics"]["mass"][0]) == 0.5
    env.restore_episode_state(blob)
    env._step(np.array([0.0, 0.0]), np.array([0.0]))
    assert float(seen["physics"]["mass"][0]) == 2.0
    # a family with no DR passes None explicitly: physics_rows is None when nothing is declared
    env._dr_now = {}
    assert env.physics_rows(1) is None


def test_a_subclass_with_no_concrete_random_state_is_refused_at_class_creation():
    """The check fires, and this is the evidence it does.

    `random_state` is NOT `@abstractmethod` on `BatchedEnvAdapter`: it is
    checked at class creation instead. An abstract declaration on the ABC would
    shadow a concrete `random_state` a subclass inherits from a later base, so
    such a class would not instantiate without re-declaring it; the
    class-creation check refuses a subclass with no concrete `random_state`
    while accepting one that inherits it.

    WHAT IT MUST NOT BE: a presence check. `dir(cls)` always contains
    `random_state`, because `EnvAdapter` defines the bare
    `raise NotImplementedError` one -- so "exists somewhere in the MRO" is
    trivially true and would never fire. The check resolves the attribute
    through the MRO and compares it BY IDENTITY to that bare base.

    What it is protecting: `EnvAdapter.random_state` raising surfaces as
    `env.sample_transitions failed ()`, an empty message, after which a
    verifier that fell back to SYNTHETIC draws would score rewards on states
    the environment never produced.
    """
    from bird.envs.jax_base import BatchedEnvAdapter

    with pytest.raises(TypeError, match="random_state"):
        class _NoRandomState(BatchedEnvAdapter):
            name = "_no_random_state"

            def source_parents(self):
                return ()

            def reset_batch(self, key, n):
                return np.zeros((n, 2), dtype=np.float32)

            def step_batch(self, state, action, physics):
                return state, np.array([False]), {}


def test_a_subclass_inheriting_a_concrete_random_state_is_created_fine():
    """The mirror: the check must not fire on the case it exists to allow.

    Without this, a check that raised on EVERYTHING would pass the test
    above and look like protection.
    """
    from bird.envs.jax_base import BatchedEnvAdapter

    class _WithParentRandomState(BatchedEnvAdapter):
        name = "_with_parent_random_state"

        def source_parents(self):
            return ()

        def random_state(self, rng):          # concrete, as a CPU parent would supply
            return np.zeros(2)

        def reset_batch(self, key, n):
            return np.zeros((n, 2), dtype=np.float32)

        def step_batch(self, state, action, physics):
            return state, np.array([False]), {}

    assert _WithParentRandomState.random_state is not None
