"""Shared MuJoCo plumbing for the gymnasium-backed tiers, and the terminal-state finding.

This module registers no environment. `bird/envs/gym_mujoco.py` and
`bird/envs/assistax.py` import `_default_mujoco_gl` and `_preload_llvm_before_mujoco`
from here, and `gym_mujoco.py` takes `DEFAULT_CAMERA_CONFIG` for its
`InvertedPendulum-v5` family. Nothing here imports a simulator at module scope, so
there is no install in which those imports can fail. (`bird/envs/humanoid.py` keeps
its own copies of the two functions: the HumanoidBench tier runs in a venv of its own
with a different mujoco pin.)

--------------------------------------------------------------------------------
THE STATELESS RESTORE IS EXACT, BUT IT IS NOT FREE
--------------------------------------------------------------------------------

On a gymnasium MuJoCo env whose observation is its full simulator state,
`EnvAdapter.step(state, action)`'s stateless contract is satisfiable by `set_state` +
`do_simulation` with no cache of any kind. But `set_state` alone does not make the
step pure: MuJoCo seeds its constraint solver from `data.qacc_warmstart`, which
`set_state` leaves alone. MEASURED on `InvertedPendulum-v5`, whose hinge limit is an
active constraint once the pole is down: 163/200 shuffled-replay transitions differed,
at 4.4e-16, with the warm-start left alone (1.2e-13 on `HalfCheetah-v5`), and zeroing
it before each step takes the measurement to **0/200 at max |obs deviation|
0.000e+00**. Anyone porting this pattern to another MuJoCo env should assume the same
is required until measured otherwise, and should measure with SHUFFLED replay rather
than round-trips: a round-trip repeats the same state and hides a history dependence
entirely. `gym_mujoco.py` and `humanoid.py` carry the per-env measurements.

--------------------------------------------------------------------------------
A TERMINAL STATE CORRELATED WITH THE METRIC LEAKS THE METRIC INTO THE LEARNER
--------------------------------------------------------------------------------

`Acrobot` in `control.py` removes gymnasium's terminal state, and the reason recorded
there is that terminating on *success* makes "got there" the only measurable thing.
`InvertedPendulum-v5` terminates on *failure* -- `|theta| > 0.2` -- and that is the
same defect with the opposite sign, but it is much harder to see, because it does not
distort the metric's definition. It distorts what the learner optimises.

MEASURED: SAC (`stable_baselines3` 2.9.0) 20,000 steps, 20 evaluation episodes, on
`InvertedPendulum-v5` with its termination LEFT ON: a reward that is identically zero,
and one that pays the agent to drop the pole, both reach the survival ceiling.

Reproduced on a second seed: 10/10 episodes of exactly 1000 steps under `reward = 0`,
with deterministic-policy action statistics of mean -0.000, **std 0.025** and
excursions to +/-0.9 -- an active stabilising controller making small corrections, not
a collapsed or lucky policy. SAC learned to balance an inverted pendulum from a reward
containing no information.

The candidate explanation, stated as a candidate: SB3's SAC target is
`r + gamma*(1-d)*(Q(s',a') - alpha*log pi(a'|s'))`. The entropy bonus `-alpha*log pi`
is positive and accrues on every NON-terminal transition, while a terminated
transition bootstraps to exactly 0. With `r == 0` the agent therefore still sees more
value in not terminating: the learner supplies the survival incentive that the reward
never had to. A pure effort penalty is the one degenerate reward that fails, precisely
because `-u^2` pays for the zero action and fights it.

This explanation is **not directly confirmed**. Two attempts at a learner control
(`ent_coef=0.0`, and PPO) both came back uninformative -- in each the positive control
collapsed too, so "the degenerate reward does not survive" was a true statement about
a learner that could not learn. **A control without a positive control is not a
control, it is a result waiting to be misread.**

What IS established is the fix, and it is a targeted one rather than a workaround: with
`d == 0` the `(1-d)` factor is constant, any per-step additive becomes a uniform offset
to every Q-value, and a uniform offset changes no argmax. With termination REMOVED the
goal-directed rewards sit two orders of magnitude above the degenerate ones. v5's
native `+1 per alive step` collapsing once nothing can die is the same finding seen
from the reward side: it IS a constant function.

Where the finding is applied: `bird/envs/assistax.py` removes `bedbathing`'s upstream
terminal for this reason. `gym_mujoco.py` keeps the gymnasium envs' own terminals,
because those tasks are defined by them, and says on which tasks the trap therefore
applies; the HumanoidBench tier keeps its terminals too (`humanoid_hand.py`).

Provenance of the camera constants below: Gymnasium's `InvertedPendulum-v5`
(`gymnasium/envs/mujoco/inverted_pendulum_v5.py` and `assets/inverted_pendulum.xml`:
rail half-length 1.0, pole half-length 0.3, cart capsule `size="0.1 0.1"`). Nothing
here is a pin taken from a reward-design paper -- this is the substrate they get run
on -- so it carries no dagger/double-dagger markers.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


#: Explicit camera for `render`. Every field is here because of a measured defect;
#: do NOT simplify this back to MuJoCo's default free camera.
#:
#: (1) THE DEFAULT CAMERA MAKES `render(state)` IMPURE. Its `lookat` is initialised
#:     once, from whatever the simulator held when the viewer was first created, and
#:     never updated. Shuffled-replay over 120 transitions with bit-identical states
#:     gave **120/120 frame-hash mismatches, max per-pixel delta 129** -- a viewer born
#:     at cart x=-0.8 keeps `lookat=[-0.7845,0,0]` after a `set_state` to x=+0.6. It is
#:     not GL nondeterminism: the same env at the same state twice is byte-identical.
#:     Pinning `lookat` fixes it, re-verified at these values: **0/150 mismatches,
#:     max |dpx| 0**. This is invisible in every sequential test, which is why it has
#:     to be pinned rather than noticed.
#:
#: (2) `distance=2.04` PUTS THE CART OUT OF FRAME AT BOTH RAIL ENDS -- 0 pixels of
#:     cart-or-pole rendered at |x| = 1.0, measured; the pixel count does not move when
#:     the pole moves, because the pole is not drawn. The sharpest justification is a
#:     PD policy that stabilises *about the wall*: it survives all 1000 steps, a
#:     perfect balance score, while **93.6% of its frames would have been an
#:     empty rail**. `rda`'s `vlm_score` and `gt`'s `llm_on_vlm_captions` are scored
#:     off these frames, so a candidate could score perfectly and be invisible for
#:     almost its whole rollout with nothing in the artifact saying so.
#:
#:     The binding extent is the CART, not the pole: the cart capsule is `size="0.1
#:     0.1"`, so the body reaches x = +/-1.2, where the pole tip at theta=0.2 only
#:     reaches 1.129. At `fovy=45` that needs `d >= 2.90`; 3.3 leaves 0.167 m ~ 19 px
#:     of margin, which the pixel measurement reproduces exactly.
#:
#: (3) `elevation=0` rather than -45. Pole angle is the quantity a VLM has to read, and
#:     looking down at 45 degrees foreshortens exactly that axis. Side-on, the angle
#:     reads at true scale and the cart's position reads against the full rail. -15
#:     costs nothing in margin if a depth cue is ever wanted; -45 costs legibility.
DEFAULT_CAMERA_CONFIG: Dict[str, Any] = {
    "trackbodyid": -1,
    "distance": 3.3,
    "lookat": np.array([0.0, 0.0, 0.30]),
    "azimuth": 90.0,
    "elevation": 0.0,
}


def _default_mujoco_gl() -> str:
    """The MuJoCo GL backend to use when the operator has not chosen one.

    A hard-coded `or "osmesa"` is right on some machines and **wrong on a GPU node**
    where `scripts/setup_gl.sh` installs `libEGL.so.1` plus Mesa's EGL vendor JSON
    and **no OSMesa at all** -- there the default names a backend that does not
    exist. An adapter should not depend on an operator script having exported
    `MUJOCO_GL` first.

    So pick a backend that is actually present, preferring the one MEASURED to work
    where both are:

      * osmesa first. Measured on a machine with both libraries: `MUJOCO_GL=osmesa`
        renders at ~3.3 ms/frame while `MUJOCO_GL=egl` raises
        `OpenGL.raw.EGL._errors.EGLError` -- both libraries load, but only OSMesa
        produces a context. Loadability is necessary, not sufficient, which is why the
        order is fixed by measurement rather than derived.
      * egl second. On a headless GPU node this may be the only backend that exists,
        and it is the faster one there (measured 2.6 ms/frame, quicker than the
        software path).
      * neither -> return "" and leave `MUJOCO_GL` unset, so MuJoCo's own autodetect
        raises and names the real problem. Asserting a backend we know is absent would
        replace a missing-system-library error with a confusing backend-specific one,
        which is precisely the failure mode `scripts/setup_gl.sh`'s header describes.

    An operator's `MUJOCO_GL` always wins; this is only consulted when it is unset or
    empty.
    """
    import ctypes
    for backend, libs in (("osmesa", ("libOSMesa.so.8", "libOSMesa.so")),
                          ("egl", ("libEGL.so.1", "libEGL.so"))):
        for lib in libs:
            try:
                ctypes.CDLL(lib)
            except OSError:
                continue
            return backend
    return ""


def _preload_llvm_before_mujoco() -> None:
    """Load Triton's C extension before the software GL stack. See the long form in
    `bird/envs/metaworld.py::_preload_llvm_before_mujoco`: the system `libOSMesa.so.8`
    (llvmpipe) and `triton/_C/libtriton.so` each embed an LLVM, whichever loads second
    gets the other's symbols interposed, and `train.backend: sb3` reaches Triton lazily
    from `torch.optim.Adam` -- so the crash lands at stage [3] with exit 139 and no
    traceback. Loading Triton first is the order that survives. Duplicated here rather
    than imported because `metaworld.py` needs the `metaworld` package to be installed
    and this module must not.
    """
    try:
        import triton  # noqa: F401  (imported for its side effect on load order)
    except Exception:  # pragma: no cover - depends on the machine
        pass
