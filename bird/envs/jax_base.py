"""`BatchedEnvAdapter`: an `EnvAdapter` that can also step a device batch.

WHAT THIS IS FOR. A device simulator (MJX, or a model written directly in
`jnp`) steps thousands of environments on one GPU, and the learner already
lives there -- so the
only way to spend a GPU on physics rather than on transfers is for the env to
hand the batch over without leaving the device. This class is the seam: it
keeps EVERY `EnvAdapter` method, so evaluation, screens, traces and videos run
byte-identically against a batched env, and adds two pure device functions
that `fasttd3._JaxVecEnvView` drives.

THE CAPABILITY IS THE ADAPTER'S, NOT A CONFIG FLAG. `fasttd3` selects the
batched path with `isinstance(env, BatchedEnvAdapter)`. The science knob is
the env id -- a device implementation of a task is a different experiment
from the CPU task, because a device solver such as MJX is a different SOLVER
and not a faster one. A flag on one env id would let two solvers share a row
in a results table, which is the comparison a separate env id exists to
prevent.

WHY THE STATE IS A FLAT ARRAY AND NOT AN MJX PIPELINE OBJECT. Every numpy
`task_metric`, `success` and `reference_reward` in `tasks/` reads
`concat(qpos, qvel[, slots, tail])` positionally. Keeping that layout means
those functions apply verbatim to `np.asarray(state)` and a batched adapter
inherits the whole spec catalogue rather than needing a second copy of it --
which is the same argument `bird/tasks.py` makes for specs being data.

NOTHING HERE IMPORTS JAX AT MODULE SCOPE. `registry.load_all()` imports every
component module on every run, including on a laptop or any machine that may
hold no GPU; an import here would make `jax` a hard
dependency of the whole repo. Imports live inside the factory of each
concrete adapter, and this module is importable with nothing installed
-- `tests/test_jax_base.py` asserts exactly that, because it is the property
that stops the tier costing anything to the tiers that do not use it.

SCOPE. This is the skeleton concrete batched adapters build on: the two
signatures, the single-env bridge, the episode-state contract and the render
contract. No concrete env lives here; `bird/envs/jax_toy.py` is one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import EnvAdapter


class BatchedEnvAdapter(EnvAdapter, ABC):
    """An adapter that can step `n` environments at once on a device.

    Subclasses implement `reset_batch` and `step_batch` and get `_reset` /
    `_step` for free through the n=1 bridge below. Everything else they
    inherit unchanged, which is the point: an adapter that also answers
    `task_metric` and `render` the old way costs the evaluation half of the
    system nothing.
    """

    #: What physics produced these numbers. Written onto the seed row so
    #: a results table can never silently average a device-solver run with a
    #: CPU MuJoCo one -- the env id already separates them, and this is the
    #: second copy that survives an id being renamed. The default names the
    #: MJX solver this base is designed for; a subclass that steps something
    #: else says so (`jax_toy` sets `"jnp"`).
    physics = "mjx"

    # -- the two device functions subclasses must provide -------------------
    #
    # ABSTRACT, which is the one place this class departs from `EnvAdapter`'s
    # convention: the base marks its five required methods with a bare
    # `raise NotImplementedError`. The departure is deliberate and narrow. A
    # missing `_step` fails on
    # the first transition of the first local eval; a missing `step_batch`
    # fails on the first transition of a TRAINING, which for this family is
    # after a generate call and a jit compile on a charged GPU. `abstractmethod`
    # moves that to the adapter's CONSTRUCTOR -- not, as is often assumed, to
    # class definition: a partial subclass still DEFINES fine and raises
    # TypeError naming the missing method when someone tries to build one.
    # That is early enough to be free and late enough that test stubs can
    # still subclass for one half of the pair by supplying the other.

    @abstractmethod
    def reset_batch(self, key: Any, n: int) -> Any:
        """`(n, obs_dim)` float32 device array of fresh initial states.

        `key` is a JAX PRNG key and is CONSUMED: a caller that needs another
        batch splits it first. Taking a key rather than a seed keeps the
        adapter pure -- there is no mutable generator to fork, which is what
        makes `episode_state()` picklable and a resumed run reproducible.
        """
        raise NotImplementedError

    @abstractmethod
    def step_batch(self, state: Any, action: Any,
                   physics: Any) -> Tuple[Any, Any, Dict[str, Any]]:
        """`(n, *) -> (next_state, done, info)`, jitted and vmapped.

        `done` is `(n,)` boolean. `info` is a dict of `(n,)` arrays carried to
        the seed row untouched.

        AN ADAPTER NEVER SETS `time_out`. `time_out` is the HOST's to decide
        from its own step budget -- an adapter setting
        it tells the host a row hit a horizon the adapter cannot see. The
        divergence guard travels as `info["nonfinite"]`, which every jax
        adapter emits so the view's `nonfinite_rows` count works uniformly.

        `done` MIRRORS THE CPU PARENT where the parent defines one: a CPU
        adapter whose `_step` sets done (on non-finite state, say) has a
        batched twin that reports it in the same field, so a cross-tier
        reduction over `done` agrees. Where the parent defines none, `done` is
        constant False. Parity within a family outranks uniformity across the
        batched adapters.

        MUST BE PURE. It is traced, so no Python branching on values, no
        host callbacks and no mutation of `self`. A subclass that caches
        anything per step breaks `vmap` in a way that shows up as a wrong
        number rather than an error.

        `physics` is REQUIRED -- three positionals, no default:
        the domain-randomisation draw as a pytree of PER-ROW `(n,)` device
        arrays keyed like `dr_parameters`. A family without DR passes `None`
        EXPLICITLY, every call. THE DRAW LIVES WITH THE CALLER and
        `self._dr_now` is NEVER read in here: the n=1 bridge below passes
        `physics_rows(1)` from the blob; the batched view holds a per-row pytree,
        redraws a row's entry on the host through `_sample_dr` when that row
        resets, and passes it every call. A default would let a caller omit
        the draw and step under whichever slot reset last -- silent-wrong --
        which is why there is none: `tests/test_jax_base.py` asserts the
        signature on every concrete adapter.
        """
        raise NotImplementedError

    #: Does this adapter's step REQUIRE a CUDA learner device?
    #:
    #: TRUE FOR AN MJX-BACKED ADAPTER and false for a jax env that merely
    #: computes in `jnp`. `bird/config.py`'s coherence rule refuses a jax env
    #: configured without `train.hyperparameters.device=cuda`, and its own
    #: justification is "the env steps on the GPU and hands the learner
    #: device tensors through DLPack; there is no cpu path" -- TRUE of an
    #: MJX-backed adapter, FALSE of `jax_toy`, which is an offline test env in
    #: `jnp` with no device hand-off at all. Keyed on the suite NAME, the rule would
    #: catch the toy and make `configs/examples/jax_reward.yaml` unloadable.
    #:
    #: Declared here rather than listed in the rule so a future CPU-capable
    #: jax env needs no second exemption -- an exemption list is a second
    #: place the fact lives, and the rule would keep being wrong by default.
    requires_cuda: bool = True

    #: TRUE FOR EVERY SUBCLASS: this is the batched base. Declared as its own
    #: attribute rather than inferred, because the two questions this tier
    #: raises are NOT the same question and `jax_toy` is the proof -- it IS
    #: batched and does NOT require cuda. Collapsing them would make both
    #: consumers (the device coherence rule, and `xla_env.prepare_for_env`)
    #: wrong on `jax_toy`, which is why each reads a declared capability rather
    #: than the `jax_` NAME.
    #:
    #: Read by `_check_coherence` off the REGISTERED OBJECT, which for some
    #: jax ids is a factory function -- so `isinstance` is not
    #: available at config-load time and `_declare` carries this the same way
    #: it carries `requires_cuda`.
    batched: bool = True

    @abstractmethod
    def source_parents(self) -> Tuple[type, ...]:
        """The CPU classes whose task this adapter runs, rendered FIRST.

        ABSTRACT, AND DELIBERATELY NOT DEFAULTED TO THE MRO. The obvious
        default -- "the CPU adapter classes in my MRO" -- returns an empty
        tuple for an adapter that composes rather than inherits, and an
        empty tuple means `describe("full_source")` renders the subclass
        alone. That is not a degraded rendering, it is the LEAK GUARD
        SILENTLY GUARDING NOTHING: `tests/test_env_spec_leak.py` withholds
        ground-truth methods from the rendered source, and a subclass that
        defines none has nothing to withhold, so the test passes while
        checking nothing. A missing `random_state` raises; a defaulted-empty
        `source_parents` reads as success, which is why this one cannot have
        a default.

        An empty tuple is therefore a STATEMENT -- "this adapter has no CPU
        twin in code" -- made by an adapter that has considered it, and
        `JaxToyReacher` is the case. `_cpu_classes_in_mro()` is available
        for an adapter that inherits its CPU twin (it returns it in one line);
        one that composes returns the composed class, e.g.
        `(type(self._cpu()),)`.
        """

    def _cpu_classes_in_mro(self) -> Tuple[type, ...]:
        """The CPU `EnvAdapter` classes this adapter inherits from, if any.

        A convenience for `source_parents()`, never its default -- see there.
        """
        from bird.envs.base import EnvAdapter
        return tuple(c for c in type(self).__mro__
                     if issubclass(c, EnvAdapter) and not issubclass(c, BatchedEnvAdapter)
                     and c is not EnvAdapter)

    def __init_subclass__(cls, **kw: Any) -> None:
        """Every concrete adapter must have a REAL `random_state` somewhere.

        NOT `@abstractmethod`, and the difference is load-bearing. An
        abstract `random_state` here SHADOWS a concrete one inherited from a
        CPU parent, because an adapter whose MRO is
        `(BatchedEnvAdapter, <CPU parent>)` gets the ABC first -- that order is
        what makes the adapter override the physics and inherit everything
        else, so it cannot be swapped. The class then will not instantiate
        unless it re-declares the method, while the design requires the
        opposite: the batched adapter's `random_state` IS the CPU parent's,
        by identity, because a second copy lets the two tiers drift about
        what they measure. Abstract-plus-delegate cannot satisfy both; a
        delegation is not the same object.

        So the requirement is checked here, at class creation, against the
        WHOLE MRO. An adapter that inherits a real one passes and keeps
        identity; one that inherits only `EnvAdapter`'s
        `raise NotImplementedError` fails loudly with its own name -- which
        is what the abstract was for. That bare raise surfaces as
        `env.sample_transitions failed ()`, an empty message, after which
        the verifier scores rewards on SYNTHETIC draws.
        """
        super().__init_subclass__(**kw)
        # ABSTRACTNESS IS COMPUTED HERE, NOT READ. `__init_subclass__` runs
        # BEFORE `ABCMeta` populates `__abstractmethods__`, so the obvious
        # `if cls.__abstractmethods__: return` guard is always false and the
        # check fires on classes that are still abstract -- it would reject a
        # test double that deliberately omits a device method, at class
        # creation, before its own test could assert anything.
        if any(getattr(getattr(cls, n, None), "__isabstractmethod__", False)
               for n in dir(cls)):
            return                      # still abstract itself; nothing to check yet
        from bird.envs.base import EnvAdapter
        fn = getattr(cls, "random_state", None)
        if fn is None or fn is EnvAdapter.random_state:
            raise TypeError(
                f"{cls.__name__} has no concrete `random_state`: it resolves to "
                f"`EnvAdapter.random_state`, which raises NotImplementedError. The "
                f"screens call it through `sample_transitions`, and the failure is "
                f"SILENT -- the verifier falls back to synthetic draws and scores "
                f"the reward on states the environment never produced. Inherit it "
                f"from the CPU parent, or define it.")

    def sample_transitions(self, rng: Any, n: int) -> Any:
        """`n` (s, a, s2) triples in ONE batched dispatch.

        `EnvAdapter`'s version loops `_step` n times, which on this tier is n
        jit dispatches of a batch of one. The screens are pseudometrics
        between reward FUNCTIONS, so the draw must not see a candidate's
        domain randomisation -- hence `self._dr_nominal` explicitly rather
        than whatever `_dr_now` happens to hold.

        THE DR-PYTREE BRANCH is reached only by an adapter that declares DR
        axes: `physics_rows` then returns a per-row pytree and `step_batch`
        receives it. An adapter with no axes gets None (and may refuse a
        non-None physics by design). `jax_toy` is the adapter that declares
        axes, and `tests/test_jax_toy.py` exercises the pytree path there.
        """
        import numpy as _np
        from bird.envs.base import as_np_rng
        g = as_np_rng(rng)
        saved = self.episode_state()
        try:
            states = _np.stack([_np.asarray(self.random_state(g), dtype=_np.float64)
                                for _ in range(int(n))])
            acts = _np.stack([self.action_set[int(g.integers(self.n_actions))].copy()
                              for _ in range(int(n))])
            s2, _done, _info = self.step_batch(
                states, acts, self.physics_rows(int(n), self._dr_nominal))
            s2 = _np.asarray(s2, dtype=_np.float64)
            out = [(states[i], acts[i], s2[i]) for i in range(int(n))]
            # COUNTED ON THE ASSEMBLED LIST, not inferred from the batch
            # shape: `step_batch` returns `(n, obs_dim)` by construction, so
            # a shape check would be satisfied by its own input. The verifier
            # refuses a sampler that returns nothing, and a silent short
            # return is what that refusal is for.
            if len(out) != int(n):
                raise RuntimeError(
                    f"{self.name}: batched sample_transitions assembled {len(out)} "
                    f"triples for n={n}")
            return out
        finally:
            self.restore_episode_state(saved)

    def physics_rows(self, n: int, draw: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """`draw` (default `self._dr_now`) broadcast to `n` rows, or None when the
        adapter declares no DR axes. The one place a host dict becomes the
        per-row pytree `step_batch` takes, so the batched view and the bridge
        agree."""
        draw = self._dr_now if draw is None else draw
        if not draw:
            return None
        return {str(k): np.full((n,), float(v), dtype=np.float32) for k, v in draw.items()}

    # -- the single-env bridge ----------------------------------------------

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        """One state, for evals, screens and traces.

        Goes through `reset_batch` with n=1 rather than having its own code
        path, so the single-env and batched answers cannot drift. The cost is
        a jit dispatch per call, which is real and is why this is never on the
        training path -- `fasttd3._JaxVecEnvView` calls `reset_batch` directly.
        """
        key = self._key_from(rng)
        # STORED, not just used. `episode_state()` returns `self._key`, so a
        # reset that derived a key and dropped it would leave the episode
        # state `None` until a subclass happened to set the attribute -- and
        # `restore_episode_state(episode_state())` would then be an identity
        # on nothing, which is a restore that silently restores no episode.
        self._key = key
        batch = self.reset_batch(key, 1)
        return np.asarray(batch[0], dtype=np.float64)

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """One transition. Same bridge, same reason, same cost."""
        # `np.asarray`, not `jnp.asarray`, and this is not an oversight.
        # `step_batch` is jitted, and a jitted function transfers a host array
        # to the device on call exactly as an explicit `jnp.asarray` would --
        # same one transfer, same cost. Doing it with numpy is what keeps this
        # whole module importable AND EXERCISABLE with no jax installed, so
        # the bridge's wiring is covered by an ordinary CI case rather than by
        # a `jax`-marked one that skips wherever jax is not installed.
        # The blob's draw goes in as ONE-row arrays, so `episode_state()` /
        # `restore_episode_state()` keep the base contract for every numpy
        # consumer while `step_batch` itself never reads `self._dr_now`.
        s2, done, info = self.step_batch(np.asarray(s)[None, :],
                                         np.asarray(a)[None, :],
                                         self.physics_rows(1))
        # float64 on the way out: every numpy consumer in `tasks/` and every
        # metric in `bird/components/` is float64, and silently handing them
        # float32 would move the last digits of a task metric for no reason a
        # reader could find.
        # SCALARS out of the info dict, not 0-d arrays: `info["success"]` is read
        # as a bool by every consumer and `tests/test_env_success_flag.py` holds
        # it to a Python/numpy scalar type. `np.asarray(x)[()]` is a numpy
        # scalar (`np.bool_`, `np.float32`) for a 0-d value and the array
        # itself for anything with a shape, so a per-row vector still passes.
        return (np.asarray(s2[0], dtype=np.float64),
                bool(np.asarray(done[0])),
                {k: np.asarray(v[0])[()] for k, v in (info or {}).items()})

    # -- the contracts the base class already has ---------------------------

    def _key_from(self, rng: Any) -> Any:
        """A JAX PRNG key from whatever the caller had.

        The suite hands adapters a `np.random.Generator`; the batched path
        hands a key. Deriving one from the other HERE, once, is what lets a
        single-env eval and a batched training start from the same seed
        material instead of two unrelated streams.
        """
        import jax  # local: see the module docstring

        if hasattr(rng, "integers"):
            return jax.random.PRNGKey(int(rng.integers(0, 2 ** 31 - 1)))
        if rng is None:
            return jax.random.PRNGKey(0)
        return rng  # already a key

    def episode_state(self) -> Any:
        """`(key, rng, dr)`: the base contract (`EnvAdapter.episode_state`, "the
        episode RNG and the DR draw") with the PRNG key added, not replaced.

        The key ALONE would be a silent-wrong contract: a `step_batch` that
        read `self._dr_now` per call, with `_VecEnvView` restoring this blob
        per slot, would step two interleaved slots under whichever reset LAST,
        and nothing would say so. `step_batch` takes the draw as an argument
        (above) and the blob carries it for the n=1 bridge and every numpy
        consumer. The numpy generator rides along for the same reason it does
        on the base:
        `sample_transitions` swaps it out and puts the blob back, and
        `random_state` reads it. The key is immutable, the dict is copied, so
        a restored blob is the episode as it was, not a live alias of it.
        """
        return (getattr(self, "_key", None), self._rng, dict(self._dr_now))

    def restore_episode_state(self, state: Any) -> None:
        self._key, self._rng, dr = state
        self._dr_now = dict(dr)

    # NO `render` ON THIS BASE, deliberately.
    #
    # `observability.py` fetches it with `getattr(env, "render", None)` and
    # treats absence as "this env has no renderer": recording is skipped with
    # one debug line and the run is otherwise identical (its module docstring,
    # `OPTIONAL. Duck-typed`). Defining a `render` that returns None would make
    # `getattr` answer with a bound method, so the recorder would proceed and
    # hand None to `_as_rgb8`, which raises and is caught as "env.render
    # returned something unrenderable" -- a warning per candidate per
    # iteration, for an env that simply has no renderer. Absent is the way to
    # say absent.
    #
    # An MJX-backed concrete adapter would define it through a CPU
    # `mujoco.MjData` written from the state array's `qpos`/`qvel`: MJX has no
    # renderer of its own, and the flat layout is what makes that route
    # possible at all.
