"""`jax_toy` -- `ToyReacher`'s dynamics in `jax.numpy`: the offline point of the jax tier.

WHY IT EXISTS. `generate.reward_language: jax` is refused on every numpy
env (`config._check_coherence`), so without a `jax_*` env the whole contract --
the prompt clause, the jnp namespaces, the traced probe, the mock's jnp program
-- can be unit-tested but never RUN end to end by the suite. This is the env
that makes `configs/examples/jax_reward.yaml` a legal config: the same
damped point mass as `toy_reacher`, the same goal, radius, horizon, action grid
and success threshold, with `step_batch` and `reset_batch` written as pure
`jnp` functions under `jax.jit`. It is a TEST ENV, not a benchmark:
CPU jax is fine, and a run on it says "the tier's plumbing holds", nothing
about physics.

THE PORT IS LINE FOR LINE. `_step_pure` below is `ToyReacher._step` with the
scalar `if` chains replaced by `jnp.clip` and the per-row loop replaced by a
leading batch axis; `tests/test_jax_toy.py` holds the two within float32 of
each other on random transitions under nominal AND moved physics. Two
deliberate differences, both stated rather than hidden: the state is float32
on device (the bridge in `BatchedEnvAdapter._step` hands float64 back to every
numpy consumer), and the `action_noise` DR axis is ABSENT -- `step_batch` is
pure and takes no key, so a noisy actuator would need a key threaded through
the step, which the batched view owns and this toy does not need. `dr_parameters`
therefore declares three axes where `toy_reacher` declares four; RAPP never
sweeps `action_noise` on this env (it sweeps `dr_parameters`), and `set_dr` drops
the key as it drops any axis the adapter does not declare.

A SPEC, BECAUSE THE NATIVE-SIGNAL RULE READS ONE. Without `tasks/jax_toy/` the
env could not run: rule N
(`bird/native_signal.py`) resolves what counts as the env's own signal from
the SPEC alone (`discrete_success.kind`, `reward.human.kind`), and "no spec"
resolves to REFUSE, at load for `native_success` and at §4 for `native`, so a
spec-less env cannot be scored by any supervised source at all -- which is
the rule working as written, not a gap. `tasks/jax_toy/shared_spec.yaml` is
therefore `toy_reacher`'s spec re-derived for this adapter: the same surface
and prose, the citations re-pointed at this file, the DR axes reduced to the
three that exist here, and the random anchor and the 24-reset ranges
RE-MEASURED through this adapter rather than copied.

NOTHING HERE IMPORTS JAX AT MODULE SCOPE. `registry.load_all()` imports this
module on every run; the factory imports jax FIRST and raises `ImportError`
naming the extra when it is absent, so every catalogue test that constructs
the registry skips or deselects this id on a machine without the extra
(`tests/conftest.py::_SIM_MARKERS` marks it `jax`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..registry import register
from .base import _bin, _states_of
from .jax_base import BatchedEnvAdapter
from .spec import SpecEnvAdapter

__all__ = ["JaxToyReacher", "JAX_EXTRA_HINT"]

#: What the factory says when jax is missing. One string, so the message a user
#: reads and the message a test matches are the same bytes.
JAX_EXTRA_HINT = ("needs the `jax` extra (`uv sync --extra jax`, or scripts/setup_jax.sh); "
                  "CPU jax is enough for this env")


class JaxToyReacher(BatchedEnvAdapter, SpecEnvAdapter):
    """2-D damped point mass that must reach a fixed goal and stay there -- in jnp.

    See the module docstring. Every number below is `ToyReacher`'s. The MRO puts
    the batched bridge first (`_reset`/`_step` through `reset_batch`/`step_batch`)
    and the spec second (prose, fields, DR axes off `tasks/jax_toy`); both are
    `EnvAdapter`s and neither overrides the other's members.
    """

    name = "jax_toy"
    #: Not MJX: this is `jax.numpy` arithmetic, and the seed row should say so.
    physics = "jnp"
    obs_dim = 6
    action_dim = 2
    horizon = 25
    success_threshold = 0.20

    goal = (0.5, 0.5)
    goal_radius = 0.22
    dt = 0.18
    _pos_bins = 9
    _vel_bins = 2
    n_disc_states = 9 * 9 * 2 * 2
    exact_states = None  # continuous: `discretise` is a binning, not a bijection

    #: Three of `toy_reacher`'s four axes -- no `action_noise` (module docstring).
    #: The spec's `domain_randomization` block states the same three and wins
    #: (`SpecEnvAdapter._apply_spec`); these are the class's own statement of them.
    dr_parameters = {"mass": (0.3, 3.0), "damping": (0.2, 0.9), "force_scale": (0.3, 1.2)}
    _dr_nominal = {"mass": 1.0, "damping": 0.5, "force_scale": 0.8}

    def __init__(self) -> None:
        import jax  # local: the factory has already checked it imports
        import jax.numpy as jnp

        self._jax, self._jnp = jax, jnp
        super().__init__()
        # Jitted ONCE per adapter. Physics parameters are ARGUMENTS, not
        # closed-over attributes: a jit that captured `self._dr_now` at trace
        # time would keep the first episode's draw for every later one, and
        # nothing would say so. Passed as 0-d device arrays so a new draw is a
        # new value in the same trace, not a retrace.
        self._step_fn = jax.jit(self._step_pure)
        self._reset_fn = jax.jit(self._reset_pure, static_argnums=1)

    # -- the shape of the problem -------------------------------------------

    def _build_action_set(self) -> Any:
        # 3x3 grid of forces, as `ToyReacher`: enough to brake on either axis.
        return [[fx, fy] for fx in (-1.0, 0.0, 1.0) for fy in (-1.0, 0.0, 1.0)]

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array([-1, -1, -1, -1, -1, -1], dtype=float),
                np.array([1, 1, 1, 1, 1, 1], dtype=float))

    # -- the two device functions --------------------------------------------

    def _reset_pure(self, key: Any, n: int) -> Any:
        """`(n, 6)` float32: the far quadrant, at rest, goal fixed -- `ToyReacher._reset`."""
        jax, jnp = self._jax, self._jnp
        kx, ky = jax.random.split(key)
        x = jax.random.uniform(kx, (n,), minval=-0.85, maxval=-0.25)
        y = jax.random.uniform(ky, (n,), minval=-0.85, maxval=-0.25)
        zero = jnp.zeros((n,))
        gx = jnp.full((n,), self.goal[0])
        gy = jnp.full((n,), self.goal[1])
        return jnp.stack([x, y, zero, zero, gx, gy], axis=1).astype(jnp.float32)

    def reset_batch(self, key: Any, n: int) -> Any:
        return self._reset_fn(key, int(n))

    def _step_pure(self, s: Any, a: Any, mass: Any, damping: Any,
                   force_scale: Any) -> Tuple[Any, Any, Dict[str, Any]]:
        """`ToyReacher._step`, batched: the same clips in the same order.

        `mass`, `damping`, `force_scale` are PER-ROW `(n,)` arrays (or scalars
        that broadcast): the draw is a required argument, never `self._dr_now`, so two
        rows may step under two draws in one call (the `step_batch` contract)."""
        jnp = self._jnp
        s = jnp.asarray(s, dtype=jnp.float32)
        a = jnp.asarray(a, dtype=jnp.float32)
        mass = jnp.asarray(mass, dtype=jnp.float32)
        damping = jnp.asarray(damping, dtype=jnp.float32)
        force_scale = jnp.asarray(force_scale, dtype=jnp.float32)
        ax = jnp.clip(a[:, 0], -1.0, 1.0)
        ay = jnp.clip(a[:, 1], -1.0, 1.0)
        gain = force_scale / jnp.maximum(mass, 1e-3)
        vx = jnp.clip(damping * s[:, 2] + gain * ax, -1.0, 1.0)
        vy = jnp.clip(damping * s[:, 3] + gain * ay, -1.0, 1.0)
        px = jnp.clip(s[:, 0] + self.dt * vx, -1.0, 1.0)
        py = jnp.clip(s[:, 1] + self.dt * vy, -1.0, 1.0)
        s2 = jnp.stack([px, py, vx, vy, s[:, 4], s[:, 5]], axis=1)
        dist = jnp.hypot(px - s[:, 4], py - s[:, 5])
        # WHO OWNS HORIZON TRUNCATION. Not
        # the adapter. `done` is "the env ended the episode on its own terms"
        # and `info["time_out"]` is "the env's OWN clock ran out" -- the toy has
        # neither, so both are always False, exactly as `ToyReacher._step`
        # returns `False`. The `horizon` cut is the HARNESS's: `_rollout` and
        # every eval loop step `env.horizon` times, and `fasttd3._JaxVecEnvView`
        # does the same per slot in its `_tail`, reading `env.horizon`, not
        # this flag. An adapter that set `time_out` at its horizon would make
        # the cut happen twice, one step apart, under two names.
        done = jnp.zeros((s.shape[0],), dtype=bool)
        return s2, done, {"success": dist <= self.goal_radius, "distance": dist,
                          "time_out": done}

    def step_batch(self, state: Any, action: Any,
                   physics: Any) -> Tuple[Any, Any, Dict[str, Any]]:
        """`physics` (required): `{mass, damping, force_scale}` as `(n,)` arrays, the
        caller's draw; an explicit `None` means nominal (the class constants,
        never `self._dr_now`)."""
        p = physics if physics is not None else self._dr_nominal
        return self._step_fn(state, action, p["mass"], p["damping"], p["force_scale"])

    # -- the numpy half every stage reads ------------------------------------

    def discretise(self, s: np.ndarray) -> int:
        p, v = self._pos_bins, self._vel_bins
        bx = _bin(float(s[4]) - float(s[0]), -2.0, 2.0, p)
        by = _bin(float(s[5]) - float(s[1]), -2.0, 2.0, p)
        bvx = _bin(float(s[2]), -1.0, 1.0, v)
        bvy = _bin(float(s[3]), -1.0, 1.0, v)
        return ((bx * p + by) * v + bvx) * v + bvy

    #: NO DEVICE HAND-OFF: this is an offline test env that computes in `jnp`
    #: and returns host arrays through the n=1 bridge. The CUDA refusal in
    #: `_check_coherence` exists for MJX-backed adapters, whose `_JaxVecEnvView`
    #: passes the learner device tensors over DLPack; there is no such path
    #: here, so requiring cuda would refuse a config that runs correctly on
    #: any machine.
    requires_cuda = False

    def source_parents(self) -> tuple:
        """EMPTY, AND THAT IS A STATEMENT RATHER THAN A DEFAULT.

        `jax_toy` has no CPU twin in code: it is a toy written directly in
        `jax.numpy`, so there is no delegated class whose source a reader of
        `full_source` would otherwise miss, and its own ground-truth methods
        are on this class where the stripper can already reach them.

        `source_parents` is abstract on `BatchedEnvAdapter` precisely so this
        emptiness has to be asserted by an adapter that has considered it. A
        defaulted empty tuple would render a DELEGATING adapter subclass-only,
        leaving the leak guard with nothing to withhold and passing while
        guarding nothing.
        """
        return ()

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        s = np.empty(6)
        s[0:4] = rng.uniform(-1.0, 1.0, size=4)
        s[4], s[5] = self.goal
        return s

    def task_metric(self, traj: Any) -> float:
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        dist = np.linalg.norm(states[1:, 0:2] - states[1:, 4:6], axis=1)
        return float(np.mean(dist <= self.goal_radius))

    def legend_lines(self, state: np.ndarray) -> List[str]:
        s = np.asarray(state, dtype=float).ravel()
        lines = ["REACH THE GOAL AND STAY"]
        if s.size >= 6:
            gx, gy = (np.round(s[4:6], 2) + 0.0).tolist()
            lines.append(f"GOAL X {gx:.2f} Y {gy:.2f}")
        lines.append(f"GOAL RADIUS {self.goal_radius:.2f}")
        return lines

    # --- BEGIN reference reward (ground truth; strip before showing an LLM) ---
    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """`ToyReacher.reference_reward`: get close, arrive slowly, stay."""
        s = np.asarray(s, dtype=float)
        dist = float(np.hypot(s[0] - s[4], s[1] - s[5]))
        r = -dist - 0.05 * float(np.hypot(s[2], s[3]))
        if dist <= self.goal_radius:
            r += 1.0
        if a is not None:
            aa = np.asarray(a, dtype=float).ravel()
            r -= 0.01 * float(np.dot(aa, aa))
        return r
    # --- END reference reward ---


def _declare(**kw: Any) -> Any:
    """Copy an adapter's declarations onto the FACTORY the registry holds.

    `registry.get("env", ...)` returns whatever was decorated -- the CLASS for
    some adapters, a FACTORY FUNCTION here -- and a
    function inherits nothing from the class it returns. `_check_coherence`
    must read `requires_cuda` WITHOUT constructing the adapter (construction
    needs the `jax` extra, and config resolution must work in any venv), so
    the declaration has to be on the registered object itself.

    That is two homes for one fact, so `tests/test_jax_toy.py` asserts the
    factory's copy equals the class's.
    """
    def deco(fn: Any) -> Any:
        for k, v in kw.items():
            setattr(fn, k, v)
        return fn
    return deco


@register("env", "jax_toy")
@_declare(requires_cuda=JaxToyReacher.requires_cuda, batched=JaxToyReacher.batched)
def jax_toy(ctx: Any) -> JaxToyReacher:
    """`ToyReacher` in jnp; the offline test env of `generate.reward_language: jax`.

    ImportError FIRST, naming the extra, so a machine without jax fails here in
    milliseconds with a sentence rather than inside `EnvAdapter.__init__`.
    """
    # THE FLAGS, BEFORE THE IMPORT BELOW. `bird.py::run` and the spawn child
    # both call this earlier and this one is then a no-op; the point is the
    # THIRD path -- any other code that constructs a jax env -- for which
    # neither of those runs. The factory is where every construction passes
    # through while jax is still unimported, which is the only moment
    # `XLA_FLAGS` and `XLA_PYTHON_CLIENT_PREALLOCATE` can still bite.
    # `strict=False`: see `prepare_for_env`. (No-op for this env in any case,
    # which declares `requires_cuda = False`.)
    from bird.xla_env import prepare_for_env
    prepare_for_env("jax_toy", strict=False)
    try:
        import jax  # noqa: F401  (lazy: an optional dependency)
    except ImportError as exc:
        raise ImportError(f"problem.env_id: jax_toy {JAX_EXTRA_HINT} ({exc})") from exc
    return JaxToyReacher()
