"""HumanoidBench: the tier's recorded measurements, and the GL helpers its adapters import.

This module registers no environment. The tier's adapters -- the `h1hand_*` tasks and
`h1strong_highbar_hard` -- are in `bird/envs/humanoid_hand.py`, which imports
`_default_mujoco_gl` and `_preload_llvm_before_mujoco` from here and relies on the
measurements below rather than repeating them. They were taken on upstream's plain-H1
`h1-crawl-v0` (a free root joint plus nineteen hinges, a 51-D `qpos ++ qvel`), and each
is a property of HumanoidBench and MuJoCo rather than of that one task, so it applies to
every adapter on the tier.

--------------------------------------------------------------------------------
READ THIS FIRST: `_step` ZEROES `qacc_warmstart`, AND HERE THAT IS LOAD-BEARING
--------------------------------------------------------------------------------

`set_state` alone does NOT make a MuJoCo `step` pure. MuJoCo seeds its constraint
solver from `data.qacc_warmstart`, `set_state` does not reset it, and the seed steers
the answer whenever the solver has an active constraint set to work on.

`bird/envs/mujoco_control.py` records the same effect on gymnasium's cart-pole and
cheetah, where it is a hygiene note. MEASURED HERE, 300 shuffled-replay transitions on
`h1-crawl-v0`:

    zeroed        0/300 mismatches   max |deviation|  0.000e+00
    NOT zeroed  115/300 mismatches   max |deviation|  4.816e-01

Four point eight times ten to the minus one, in a joint coordinate. For scale, the
same probe on `InvertedPendulum-v5` gives 4.4e-16 and on `HalfCheetah-v5` 1.2e-13 --
this is **thirteen orders of magnitude larger** and is not a rounding artifact. It is a
different trajectory. The cause is contact: `nefc` is 12 with the robot merely at rest,
because a body lying under a tunnel has persistent resting contacts, so the solver runs
with a large active set on every single step rather than occasionally.

The consequence is what makes it worth the top of the file. **An adapter on this tier
that omitted one line would have a non-deterministic `step`** -- and `step` being a pure
function of `(state, action)` is what `tests/test_parallelism.py`'s bit-identity claim,
`tests/test_resume.py`'s determinism claim and every shuffled-replay evaluation rest on.
All three would be quietly wrong while every test still passed, because none of them
re-derives the transition; they compare two runs that would both be wrong the same way
only if the warmstart happened to align, and it does not.

**Measure with SHUFFLED replay, never round-trips.** A round-trip re-steps the state it
just arrived at, so the warmstart is already the right one and the defect is invisible.

--------------------------------------------------------------------------------
THE OBSERVATION MUST BE THE WHOLE SIMULATOR STATE, AND UPSTREAM CAN QUIETLY MAKE IT NOT
--------------------------------------------------------------------------------

With `obs == concatenate([data.qpos, data.qvel])`, `EnvAdapter`'s stateless
`step(state, action)` contract is satisfiable with `set_state` and nothing else. That
invertibility is **one keyword argument away from being false**, which is why the
adapters assert it rather than trust it: `humanoid_bench.wrappers.ObservationWrapper`
returns `robot.joint_angles() + robot.joint_velocities()`, i.e. `qpos[7:]` and
`qvel[6:]` -- it drops the free root joint entirely and is genuinely non-invertible. It
is selected by `kwargs["obs_wrapper"]`, compared as the *string* `"true"`.

THE STATE IS READ BACK OUT OF THE SIMULATOR, NOT RETURNED AS CONSTRUCTED. `set_state`
runs `mj_forward`, which normalises the free joint's quaternion -- so the four numbers
handed in are not the four numbers the simulator then holds (measured: 0.99027 in,
0.99977 out, after a componentwise perturbation of a unit quaternion). Returning the
pre-normalisation array would make `reset()` emit a state this env cannot be in, and
every consumer that assumes `reset()` and `step()` speak the same language --
`sample_transitions`, the replay buffer, any stored initial condition -- would be
working from a point off the manifold. Upstream's `get_obs` reads `data` for the same
reason.

--------------------------------------------------------------------------------
A DIFFERENT INITIAL DISTRIBUTION IS A DIFFERENT ENVIRONMENT
--------------------------------------------------------------------------------

HumanoidBench's initial distribution is the `qpos0` keyframe plus
`U(-randomness, randomness)` on every position coordinate, velocities left at zero
(`humanoid_bench.env.HumanoidEnv.reset_model`, `DEFAULT_RANDOMNESS` = 0.01). It perturbs
the quaternion componentwise and does not renormalise it. Where upstream's reset is odd
in this way it is REPRODUCED, NOT CORRECTED: quietly fixing it would make an adapter's
numbers incomparable with anything anyone else runs on the same gym id. The draw comes
from the RNG an adapter is handed rather than the simulator's, so an episode is
reproducible from `ctx.rng` alone.

--------------------------------------------------------------------------------
THE SHIPPED REWARD IS CALLED, NOT PORTED
--------------------------------------------------------------------------------

HumanoidBench's rewards read `data.actuator_force` and site positions through MuJoCo's
named indexing (and on some tasks the live contact list), none of which are in the state
vector. A reimplementation would need forward kinematics anyway and would be a second
copy free to drift from upstream. So an adapter's `reference_reward` restores the
state, sets `ctrl`, runs `mj_forward` and asks upstream's task object. The cost is one
`mj_forward` per call; the benefit is that it cannot disagree with the published
baselines.

--------------------------------------------------------------------------------
`success_bar` IS NOT OUR METRIC, AND THE REASON GENERALISES
--------------------------------------------------------------------------------

The obvious move is to adopt the benchmark's own success criterion, the way
`bird/envs/metaworld.py` adopts Meta-World's -- an environment's success check is better
provenance than one written here. **Here it is worse, and the difference is worth stating
because it is easy to get backwards.**

Meta-World's flag is a *state* fact: the object is within epsilon of the goal, true or
false regardless of what reward you trained on. HumanoidBench's `success_bar` is a
threshold on **the episode return of HumanoidBench's own hand-written reward**. Adopting
it would make `task_metric` a monotone function of `reference_reward`, so every §4 number
would measure how closely a candidate reproduces the behaviour that one specific
hand-written reward already rewards -- a reward-metric circularity, arriving through a
different door.

So each adapter's `task_metric` is built from **state** (`humanoid_hand.py` states
each one) and `success_bar` is carried only as a comparison number. Two facts about it
that a later reader will want:

  * `crawl`'s bar is **700**, on a reward whose per-step value is in [0, 1] over a
    1000-step episode.
  * `success_bar` is **read by nothing in the upstream repo.** It is declared on 20 task
    classes and consumed in zero lines of code -- a plotting constant from the paper. It
    is a declared-but-unread constant, and anyone reaching for it should know it was
    never load-bearing upstream either.

The distinction that actually matters, since an adapter's `reference_reward` here IS
the native reward: a native reward is dangerous **when the metric is derived from it**,
not inherently. `crawl`'s shipped reward, for instance, is a genuine hand-tuned expert
cost with per-task bounds and the reward the published SAC baseline was trained on.
Dropping `success_bar` is what makes adopting a native reward safe. Do not apply
"designed, not native" mechanically to the next env without asking which of the two is
true there.

--------------------------------------------------------------------------------
RENDERING
--------------------------------------------------------------------------------

The models define `cam_default` as a `mode="trackcom"` camera on the pelvis, and
`humanoid_bench.tasks.Task.camera_name` selects it by name -- which gymnasium's renderer
honours in preference to any config dict, so HumanoidBench's own
`env.DEFAULT_CAMERA_CONFIG` is dead code. Tracking that lives in the MODEL derives its
pose from `data`, which `set_state` sets, so `render(state)` is pure by construction.
Pinning a fixed `lookat` would be actively wrong: the robot translates metres across a
scene and a fixed look-at point would lose it.

THE FRAME SIZE IS REQUESTED, NOT INHERITED. `HumanoidEnv.__init__` defaults to 256x256;
the adapters ask for 448x320 instead, because `rda`'s `vlm_score` and `gt`'s captions
are graded off these frames and a humanoid at 256 px is materially less legible than
the cart-pole this repo's other adapters render. A `trackcom` camera keeps the robot
centred, so the only cue that it is travelling is the scenery moving behind it --
measured on upstream's plain H1 in the crawl tunnel, an 8 m translation at fixed pose changes
6.92% of pixels because the tunnel walls sweep past, against 0.4% on HalfCheetah-v5's
open checkerboard. The robot renders as a near-black silhouette against pale grey, so
gross posture reads clearly and fine joint detail does not -- worth knowing before
trusting a VLM caption about, say, which arm is forward.

--------------------------------------------------------------------------------
GL IS REQUIRED AT CONSTRUCTION, EVEN FOR A RUN THAT NEVER RENDERS
--------------------------------------------------------------------------------

`humanoid_bench.env.HumanoidEnv.__init__` defaults `render_mode="rgb_array"`, and
`humanoid_bench.tasks.Task.__init__` builds an offscreen viewer whenever `render_mode is
not None`. So constructing any env on this tier opens a GL context before `bird.py` has
printed its banner.

On a headless node without `libEGL.so.1` that is an `AttributeError: 'NoneType' object has
no attribute 'eglQueryString'` at 20-47 s elapsed with no run output -- a run that dies
before it has written anything. **A job script for this tier must source
`scripts/setup_gl.sh --env`.** `MUJOCO_GL=disable` is not an
escape hatch: gymnasium's renderer rejects anything outside `{glfw, egl, osmesa}`.

--------------------------------------------------------------------------------
DEPENDENCIES: THIS TIER CANNOT SHARE AN INTERPRETER WITH THE REST OF THE REPO
--------------------------------------------------------------------------------

`humanoid_bench` is not on PyPI (install `--no-deps` from a git clone) and needs
**`mujoco==3.1.6`**, which is a real pin and not author caution: its vendored
`dmc_deps/dmc_index.py` reads `MjModel.flex_xvert0`, removed after 3.1.6. `metaworld`
3.1.1 pins `mujoco==3.3.0` exactly. **The two cannot coexist**, so this tier runs in its
own venv (`scripts/setup_humanoid.sh`).

Only `dm_control` (for `rewards.tolerance`), `torch` and `jax` are additionally required
at import, the last two via `wrappers.py` -> `mjx/flax_to_torch.py` at module scope, even
for a task that never touches MJX.

WHICH PINS ARE REAL, because "the pins are author-caution" is true of some and false of
others:

  * `mujoco==3.1.6` -- **REAL**, and not for the reason `setup.py` suggests. 3.3.0 fails
    inside HumanoidBench's *vendored* `dmc_deps/dmc_index.py`, which reads
    `MjModel.flex_xvert0`; that attribute was removed after 3.1.6. A throughput benchmark
    that only calls `mj_step` never touches `dmc_index` and will run happily on 3.3.0 --
    so a green physics benchmark is NOT evidence the package works on 3.3.0.
  * `gymnasium==0.29.1` for **import and stepping** -- **not real**. Measured working on
    1.3.0.
  * `gymnasium==0.29.1` for **rendering** -- **real**. `Task.render` calls the 0.29-era
    `MujocoRenderer.render(mode, camera_id, camera_name)` and raises `TypeError` on 1.x.
    `humanoid_hand.py::_H1HandBase.render` bypasses it; see there.

The moral, since this tier will be re-tested: a pin is real *for a code path*, not for a
package. Test the path you are about to use.

"""

from __future__ import annotations


def _default_mujoco_gl() -> str:
    """Pick a MuJoCo GL backend that is actually installed. See the long form in
    `bird/envs/mujoco_control.py::_default_mujoco_gl` -- osmesa first because it is the
    one measured to produce a context where both are installed, egl second because it
    may be the only one on a headless GPU node, and "" when neither is present so MuJoCo's own
    autodetect raises and names the real problem instead of a backend-specific one.

    Duplicated rather than imported because this tier runs in its own `mujoco==3.1.6`
    venv and is kept independent of the modules the `mujoco==3.3.0` tiers use; the two
    copies implement the same rule.
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
    """Load Triton's C extension before the simulator's GL stack. See the long form in
    `bird/envs/metaworld.py::_preload_llvm_before_mujoco`: the GL stack and
    `triton/_C/libtriton.so` each embed an LLVM, whichever loads second gets the other's
    symbols interposed, and the crash is a SIGSEGV with no traceback at stage [3].

    MEASURED ON THIS TIER (an A100 node, driver 580.178.04, the venv
    `scripts/setup_humanoid.sh` builds, mujoco 3.1.6 / torch 2.14.0+cu130 / triton
    3.8.0). Four one-line processes, the discriminator being which import comes first:

        import triton                  -> ok
        import torch;  import triton   -> ok
        import jax;    import triton   -> ok
        import mujoco; import triton   -> SIGSEGV 139, in `triton/knobs.py:15`

    and the same fault through the real path -- construct `h1hand_package`, then
    `torch.compile(...)` on cuda -- which is what `train.backend: simba_v2` does on
    every training. So it is the mujoco-first order -- not the machine, not jax, not a
    numpy ABI, and not a CPU-only torch install.

    WHY IT MATTERS ON THIS TIER: `train.backend: simba_v2` is a torch learner that
    `torch.compile`s its update closure on cuda, which loads Triton after the
    simulator unless this preload has run first.

    Duplicated rather than imported for the same reason `_default_mujoco_gl` above is.
    """
    try:
        import triton  # noqa: F401  (imported for its side effect on load order)
    except Exception:  # pragma: no cover - depends on the machine
        pass
