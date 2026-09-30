"""Meta-World MT50: fifty manipulation tasks whose ground truth is *not ours* (§0).

`bird.envs.toy` makes the loop run; `bird.envs.control` makes it mean something on a
task the RL literature quotes. This module exists for a third reason: on Pendulum
the success check is **written by us**, so "did the method succeed" is graded by an
author of this repo. Meta-World ships `info["success"]` -- a geometric check published with
the benchmark, used by every MT10 number in the literature -- and this adapter's
`task_metric` is a re-derivation of it that we *test for equality* against the
simulator's own flag. That is the first ground-truth signal in the repo we did not
invent, and it is the whole reason this file is worth 900 lines.

Fifty entries in the `env` family, one per MT50 task (`metaworld.env_dict.MT50_V3`):
the ten MT10 tasks as `mt10_<key>` and the other forty as `mt50_<key>`. The prefix
names the smallest published protocol the task belongs to (MT10 is a subset of MT50,
so a task never carries two ids); `bird/tasks.py::METAWORLD_MT10` is the one place the
ten are listed and the id rule lives there. Where a measurement below says "the
ten", the success-check equality, the snapshot restore and the reset draws were also
re-measured on all fifty, and the two transcriptions that approximate (`reach`'s
tcp, and `stick-push`'s contact conjunct) are the only ones that do.

WHAT IS HARD HERE, AND WHY THE FILE LOOKS THE WAY IT DOES
---------------------------------------------------------
`EnvAdapter._step(s, a)` is a *function of the state passed in* (see the contract in
`bird.envs.toy`'s docstring). MuJoCo is a stateful simulator and a 39-D Meta-World
observation is not invertible to (qpos, qvel) -- half of it is the *previous* frame
and none of it is a velocity. Three call sites in the harness genuinely hand `_step`
a state the simulator is not currently in, so "ignore `s` and step the sim" is wrong,
not merely impure:

  * `training._PlannerLearner` (`train.algorithm: none`, L2R) branches `n_samples`
    lookahead rollouts from one real state. Measured on a toy env instrumented for
    it: 100 of 325 `_step` calls receive a non-sequential state.
  * `training._sb3_run` interleaves `_evaluate_policy` -- which *resets* the shared
    adapter -- between `model.learn` chunks, then resumes from `_Gym._s`. Measured
    exactly `n_chunks - 1` = 4 non-sequential calls per seed.
  * `EnvAdapter.sample_transitions` steps from `random_state()` `n` independent
    times; that is what feeds the EPIC/STARC screens.

So `_step` restores the simulator from a snapshot keyed by the observation it is
handed. Two consequences run through everything below:

  1. **A cache miss is an error, never a guess.** `UnknownStateError` names the three
     legal origins of a state. Returning 0.0, re-resetting, or "stepping from
     wherever the sim happens to be" would produce a plausible number computed on
     physics that never happened -- and it would be attributed to the candidate
     reward, which is the one failure this repo cannot tolerate. Where it lands is
     already handled: `training.py:1078` records it as `result.error` per seed and
     `observability._render_frames` logs it and skips THAT CANDIDATE's video (the
     other candidates in the iteration are still attempted, and the count of
     skipped rollouts reaches the journal as a `record_skipped` event). The cause
     that would otherwise make that path FIRE ROUTINELY -- the LRU dropping a
     rollout before the recorder replays it -- is closed by `retain_states`.
  2. **Emitted observations are read-only** (`setflags(write=False)`). This is not
     hygiene theatre: Meta-World's own scripted door policy mutates the observation
     in place (`policies/sawyer_door_open_v3_policy.py:38` does `pos_door[0] -= 0.05`
     on a view), which would silently poison the cache. Read-only
     turns that into an error at the mutation site instead of a miss three frames
     later. Measured harmless across all four training backends.

WHY THE STATE IS THE 39-D OBSERVATION AND NOT A "PURE" SIM STATE
----------------------------------------------------------------
The tempting alternative is `concat(obs39, qpos16, qvel15)`. It does not work, and
that -- not the loss of comparability with published MT10 numbers -- is the decisive
argument. Dropping `data.mocap_pos` alone costs 0.047 max|dobs| on a 20-step replay
(the Sawyer is mocap-welded and `set_xyz_action` *integrates* the action into
`mocap_pos`, so it is accumulated state), and `_prev_obs` is literally half the
observation. Worse, `reset_model` rewrites **mjModel** every episode -- `body_pos`
for drawer/window/door/box/button and `site_pos("goal")` on all ten tasks -- so a
vector that restored the sim across episodes would have to carry a slice of the model
too. A genuinely pure Meta-World state does not exist at any reasonable dimension.

GL AND THE PROCESS ENVIRONMENT
------------------------------
`MUJOCO_GL` is set inside `__init__`, not at module scope: `registry.load_all()`
imports this module for `--validate-all`, `--list-configs` and every pytest run, and
a validation path must not mutate the process environment. It is written with `or`
rather than `setdefault` because a present-but-empty `MUJOCO_GL` -- exactly what
exporting the environment of a shell with `MUJOCO_GL=` produces -- makes
`setdefault` a no-op and gymnasium then raises `RuntimeError: ... must be one of
dict_keys(['glfw','egl','osmesa']): got ''`. Default is `osmesa`: measured within 6%
of egl on a CPU machine (48.8 vs 46.0 ms/frame -- rendering is CPU-bound in
`mjv_updateScene`), it produces no `EGL_NOT_INITIALIZED` teardown noise, and GPU
nodes often carry `libEGL_nvidia.so.0` but not the vendor-neutral `libEGL.so.1`
(see `scripts/setup_gl.sh`). An operator's `MUJOCO_GL` always wins.

The software GL stack and Triton cannot be loaded in either order: llvmpipe (which
is what both OSMesa and a Mesa EGL on a CPU node are) brings its own `libLLVM`, and
`libtriton.so` embeds another. `__init__` therefore calls
`_preload_llvm_before_mujoco()` immediately before the simulator imports -- read its
docstring before touching the GL default, because "just use egl" does not fix this.

Provenance markers: nothing in this file is a pin from one of the reward-design
papers -- it is substrate, like `control.py` -- so there are no †/‡ markers on the
physics. The one approximation that *is* ours is flagged on `_success_reach`.
"""

from __future__ import annotations

import ast
import copy
import inspect
import logging
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..registry import register
# Imported rather than re-implemented: `_bin`'s half-open clamped semantics are
# load-bearing for `train.backend: tabular` (a second definition that rounds
# differently at a cell edge silently moves transitions between Q-table rows), and
# `_states_of` is what lets `task_metric` score a trajectory that has been through a
# JSON round trip.
from ..tasks import (METAWORLD_MT10, TaskSpec, TaskSpecError, by_env_id,
                     index as task_index)
from .spec import (ANY_STEP, PER_STEP, SpecEnvAdapter,
                   baselines_of as _baselines_of)
from .base import UnknownStateError as _BaseUnknownStateError
from .base import View, _bin, _states_of, as_np_rng
from .cameras import MujocoViews

__all__ = ["MetaWorld", "UnknownStateError"]

log = logging.getLogger("bird")


class UnknownStateError(_BaseUnknownStateError):
    """`_step`/`render`/`reference_reward` was handed a state this adapter never
    emitted, so no simulator state can be restored for it. Subclasses
    `base.UnknownStateError`, the family-neutral name a caller outside `bird/envs/`
    catches.

    A `RuntimeError` and not a `ValueError` on purpose: `training._run_backend`
    catches broad exceptions per seed and records them as `result.error`, so this
    reads in the artifact as an adapter fault with an unmistakable message rather
    than as a bad candidate reward.
    """


def _preload_llvm_before_mujoco() -> None:
    """Load Triton's C extension *before* the software GL stack, or the process
    segfaults later inside SB3.

    Measured 3/3 each: with `MUJOCO_GL=osmesa`,
    `python -c 'import mujoco; import triton'` exits 139 (SIGSEGV) while the same
    two imports in the OTHER order exit 0 -- and so does `MUJOCO_GL=egl` in either
    order, and so does `ctypes.CDLL('.../triton/_C/libtriton.so')` in place of the
    `import triton`, with no GL context ever created. It is a plain dynamic-linker
    collision, not a rendering bug: the system `libOSMesa.so.8` (llvmpipe) pulls in
    `libLLVM.so.20.1` with global visibility and `triton/_C/libtriton.so` (462 MB)
    embeds its own LLVM, so whichever loads second gets the other's symbols
    interposed. Loading Triton first is the order that survives.

    Where it lands without this: `train.backend: sb3` reaches Triton lazily from
    `torch.optim.Adam`'s constructor inside `stable_baselines3/sac/policies.py`, so
    every Meta-World run on that backend dies at stage [3] train with exit 139 --
    no traceback, no `result.json`, an artifact directory that stops mid-journal.

    Why not just default to `egl` instead: the collision is with *llvmpipe*, not
    with OSMesa specifically, and a Mesa EGL on a CPU node is llvmpipe too -- which
    is exactly what `scripts/setup_gl.sh` installs. Fixing the load
    order fixes both; switching the default would only move the crash to the nodes
    that have no GPU.

    Cheap and optional by construction: `import triton` is 0.2 s and it does map
    `libtriton.so` (checked in `/proc/self/maps`), and Triton is absent from the
    hard dependency set, so the `except` is the normal path for a `--extra test`
    install with no torch.
    """
    try:
        import triton  # noqa: F401  (imported for its side effect on load order)
    except Exception:  # pragma: no cover - depends on the machine
        # No torch/triton here, or a Triton that cannot load at all. Either way the
        # collision this guards against cannot happen, and an env adapter must not
        # fail to construct because an unrelated optional package is unhappy.
        pass


# --------------------------------------------------------------------------
# snapshot field lists
# --------------------------------------------------------------------------
#
# Verified bitwise exact (0.0e+00 on obs, reward AND every info field over a 20-step
# replay) on all ten MT10 tasks, both within an episode and after an intervening full
# reset onto a different task instance. The second case is the `_sb3_run` checkpoint
# interleave, and it is the case the mjMODEL half exists for.
#
# Established by drop-one ablation on pick-place-v3 (max|dobs| when the field is
# omitted): data.mocap_pos 0.047, env._prev_obs 0.182. The rest were harmless in that
# test and are kept because they are cheap and matter elsewhere -- `curr_path_length`
# governs truncation, `_target_pos` is read by every reward, `data.time` by anything
# that integrates.

#: Per-step mjDATA fields. `qacc_warmstart` is restored separately and *after*
#: `mj_forward` -- see `_restore`.
_MJ_DATA = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat",
            "qfrc_applied", "xfrc_applied")

#: Per-episode mjMODEL fields. `reset_model` rewrites these every episode (grep of
#: the ten task files: sawyer_drawer_open_v3.py:106,112; sawyer_window_open_v3.py
#: :116,120; sawyer_window_close_v3.py:122,126; sawyer_door_v3.py:119,120;
#: sawyer_peg_insertion_side_v3.py:145,147; sawyer_button_press_topdown_v3.py:109;
#: and `model.site("goal").pos` in every one), so they are episode state, not
#: constants. The DR fields ride along here too, which is what makes a restored
#: state carry its own domain-randomisation draw with it.
_MJ_MODEL = ("body_pos", "body_quat", "site_pos", "body_mass", "dof_damping",
             "geom_friction", "actuator_gainprm", "eq_solref")

#: Per-episode python attributes of the Sawyer env.
#:
#: **`init_left_pad` / `init_right_pad` are deliberately absent.** They are assigned
#: `get_body_com('leftpad')`, and gymnasium's `get_body_com` returns
#: `data.body(name).xpos` -- a LIVE VIEW into `data.xpos`, not a copy. Deep-copying
#: them on save and writing them back on restore replaces the view with a frozen
#: array and permanently changes `_gripper_caging_reward`; that was the sole cause of
#: a 5.9e-3 reward mismatch that survived an otherwise bitwise-exact obs restore.
#: `mj_forward` refreshes them for free. Do not "fix" this.
_PY_EPISODE = ("_target_pos", "obj_init_pos", "obj_init_angle", "_last_rand_vec",
               "init_tcp", "_obj_to_target_init", "peg_head_pos_init",
               "window_handle_pos_init", "maxDist", "objHeight", "heightTarget",
               "target_reward")

#: Per-step python attributes. `_prev_obs` is half the observation (obs[18:36]).
_PY_STEP = ("curr_path_length", "_prev_obs", "_last_stable_obs",
            "_did_see_sim_exception")


#: Never snapshotted, whatever assigns them. `init_left_pad`/`init_right_pad` are the
#: two `_PY_EPISODE` deliberately leaves out (see the note above it): they are LIVE
#: VIEWS into `data.xpos`, and `sawyer_pick_place_v3.py:155-156` assigns them in
#: `reset_model`, so the `ast` scan below would collect them and re-create exactly the
#: 5.9e-3 caging-reward mismatch that note records. Measured: with them collected, a
#: pick-place restore reads max|dreward| = 5.4e-2 and
#: `np.shares_memory(env.init_left_pad, env.data.xpos)` goes False; with this
#: deny-list, 0.0 and True. `tests/test_metaworld.py` restores a mid-episode state on
#: all fifty tasks, pick-place included.
_PY_NEVER: Tuple[str, ...] = ("init_left_pad", "init_right_pad")


def _assigned_attrs(cls: type, methods: Tuple[str, ...]) -> Tuple[str, ...]:
    """Names `self.<name>` is ASSIGNED in the given methods of `cls` (the task class
    itself, not its bases), by `ast`, in source order and without duplicates. Names
    already in the base lists are left out so the tuples stay disjoint, and `_PY_NEVER`
    is left out because it must be. Attribute writes on something else
    (`self.model.body(...).pos = ...`) are not `self` attributes and are not collected --
    the mjMODEL half of the snapshot covers them.
    """
    try:
        tree = ast.parse(inspect.getsource(cls))
    except (OSError, TypeError, SyntaxError):  # pragma: no cover - a class with no source
        return ()
    out: List[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name in methods):
            continue
        for sub in ast.walk(node):
            targets: List[ast.expr] = []
            if isinstance(sub, ast.Assign):
                targets = list(sub.targets)
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                targets = [sub.target]
            for t in targets:
                for leaf in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                    if (isinstance(leaf, ast.Attribute) and isinstance(leaf.value, ast.Name)
                            and leaf.value.id == "self" and leaf.attr not in out
                            and leaf.attr not in _PY_EPISODE and leaf.attr not in _PY_STEP
                            and leaf.attr not in _PY_NEVER):
                        out.append(leaf.attr)
    return tuple(out)

#: `frame_skip=5` x `opt.timestep=0.0025` (metaworld/sawyer_xyz_env.py). Velocity is
#: not observed, so this is the divisor a candidate reward needs to finite-difference
#: obs[0:18] - obs[18:36]; it is quoted in `_state_fields` and `_helpers` for exactly
#: that reason.
_DT = 0.0125


# --------------------------------------------------------------------------
# geometry helpers used by the success checks
# --------------------------------------------------------------------------


def _rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate `vec` by the unit quaternion `quat`, given in **x, y, z, w** order.

    Order matters and is not the MuJoCo one: the peg class's `_get_quat_objects` returns
    `scipy.spatial.transform.Rotation.as_quat()`, which is scalar-LAST, while
    `data.xquat` is scalar-first. (The base method raises NotImplementedError; reach,
    push, pick_place, door_open and peg_insert_side return the scipy order, drawer_open
    and button_press_topdown return the raw `xquat`, and drawer_close and the two window
    tasks return zeros. `_rotate` is only ever applied to peg_insert_side's observation,
    which is scipy-ordered.) Implemented here rather than imported from scipy
    because it is eight lines, it runs inside `task_metric` (which must stay a pure
    function of the stored states), and one fewer optional import is one fewer way
    for a config-validation path to fail.

    v' = v + 2w(u x v) + 2(u x (u x v)),  u = (x, y, z). Verified to 1.2e-16 against
    `Rotation.from_quat(q).as_matrix() @ v` over 30 random peg orientations.
    """
    q = np.asarray(quat, dtype=float).ravel()
    u, w = q[0:3], float(q[3])
    cross = np.cross(u, np.asarray(vec, dtype=float).ravel())
    return np.asarray(vec, dtype=float).ravel() + 2.0 * w * cross + 2.0 * np.cross(u, cross)


def _rotate_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """`_rotate` for a quaternion in MuJoCo's **w, x, y, z** order.

    Which order a task's obs[7:11] carries is a property of that task's
    `_get_quat_objects`: the ones that convert a rotation matrix through scipy emit
    x, y, z, w (`_rotate`); the ones that read `data.body(...).xquat` emit w, x, y, z
    (this). The two are listed per task in `tasks/<id>/shared_spec.yaml` under
    `state_surface.fields[s.obj_quat]`; getting it wrong is silent (a rotated offset
    lands a few centimetres out) and the equality test against `info["success"]` is
    what catches it.
    """
    q = np.asarray(quat, dtype=float).ravel()
    return _rotate(np.array([q[1], q[2], q[3], q[0]]), vec)


# The success checks. Each is a re-derivation of what the task's `evaluate_state`
# computes, from the OBSERVATION alone -- see `MetaWorld.task_metric` for why it may
# not simply read `info["success"]`. Signature is `(obs, obj, goal)` where
# `obj = obs[4:7]` and `goal = obs[36:39]`, both passed in so fifty checks do not each
# re-slice. Measured: 0 mismatches against `info["success"]` over 900 steps per task
# (3 scripted-expert + 3 random 150-step episodes across 6 pinned instances; the MT10
# ten also over 600 steps) on 48 of 50 tasks. The two exceptions, `reach`
# (and `reach-wall`, the same tcp approximation) and `stick-push` (a contact
# conjunct), are documented on their functions and in their specs.


def _success_reach(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_reach_v3.py:86 -- `float(reach_dist <= 0.05)`, reach_dist being
    `||tcp_center - target||` (compute_reward:151).

    **The one approximation in this file, and the only one.** `tcp_center` -- the
    midpoint of the two EndEffector sites -- is genuinely absent from the 39-D
    observation; obs[0:3] is `data.body('hand').xpos`, 4.5 cm above it in z. Measured
    against the live `env.tcp_center` over 300 scripted-expert + 300 random steps per
    seed (seeds 0-5; seed s means `env.reset(np.random.default_rng(s))`
    picking the pinned instance): `hand + (0, 0, -0.045)` is 0.3-6.4 mm out on average
    depending on the seed and policy and up to ~8.6 mm on a single step. The error is
    NOT the fingers: the finger slide joints move under 0.05 mm; the hand BODY tilts on
    its soft mocap weld (eq solref 0.02) by up to ~9 degrees relative to the mocap's
    fixed orientation, and the 4.5 cm lever arm to the fingertip sites turns that tilt
    into 0-6 mm of +y lead (posture-dependent: it builds during the approach, relaxes
    below 1 degree once the hand settles at a mid or high goal, and persists at ~9
    degrees while holding the lowest goals, z ~ 0.054). It disagrees with
    `info["success"]` on 1 to 8 of every 300 expert steps depending on the seed (8, 1,
    2, 3 on seeds 0-3 under that convention; pinned instance s gives 4, 3, 2, 2; 0 on random
    steps) -- typically within 2 mm of the 0.05 m boundary (21 of 23 mismatches over
    seeds 0-5), up to 4.2 mm on seed 3, whose goal is the lowest and whose hand carries
    the largest tilt as it crosses the boundary (the size of the miss is set by the
    tilt, not the crossing speed), and always one way: the port under-reports while the
    hand approaches from -y. Every other task's check is exact. Regressing the offset on
    the gripper opening does not remove it -- the residual is the hand body tilting on
    its weld, not the fingers' aperture or inertia -- so it is shipped documented rather
    than fudged: reach is the anchor task, not a ranking task. If a published-number
    comparison on reach is ever planned, revisit. This is
    the file's one ‡-style provenance note.
    """
    tcp = obs[0:3] + np.array([0.0, 0.0, -0.045])
    return float(np.linalg.norm(tcp - goal) <= 0.05)


def _success_push(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_push_v3.py:101 -- `target_to_obj <= TARGET_RADIUS`, with
    `target_to_obj = ||obs[4:7] - _target_pos||` (:179) and TARGET_RADIUS 0.05 (:31)."""
    return float(np.linalg.norm(obj - goal) <= 0.05)


def _success_pick_place(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_pick_place_v3.py:98 -- `obj_to_target <= 0.07`, with
    `obj_to_target = ||obs[4:7] - target||` (:261)."""
    return float(np.linalg.norm(obj - goal) <= 0.07)


def _success_door_open(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_door_v3.py:79 -- `abs(obs[4] - _target_pos[0]) <= 0.08`. One axis
    only: the handle swings, so upstream scores the x displacement, not a radius."""
    return float(abs(float(obs[4]) - float(goal[0])) <= 0.08)


def _success_drawer_open(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_drawer_open_v3.py:79 -- `handle_error <= 0.03`, with
    `handle_error = ||obs[4:7] - _target_pos||` (compute_reward:122)."""
    return float(np.linalg.norm(obj - goal) <= 0.03)


def _success_drawer_close(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_drawer_close_v3.py:81 -- `target_to_obj <= TARGET_RADIUS + 0.015`
    with `target_to_obj = ||obs[4:7] - target||` (:132).

    0.065, not 0.055: the class declares `_TARGET_RADIUS = 0.04` at :16 but that name
    is never read -- `self.TARGET_RADIUS` resolves to the base class's 0.05
    (sawyer_xyz_env.py:156). The underscored attribute is dead code, and reading it
    instead is the obvious way to get this task's threshold wrong.
    """
    return float(np.linalg.norm(obj - goal) <= 0.065)


def _success_button_press(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_button_press_topdown_v3.py:74 -- `obj_to_target <= 0.024`, with
    `obj_to_target = abs(_target_pos[2] - obs[4:7][2])` (:135). The REMAINING gap to the
    target height, not the depth pressed -- the unpressed gap is 0.0935 m, so success is
    a press of at least 0.0695 m, which exceeds the button joint's declared 6 cm travel
    (buttonbox.xml:14 `range="-0.06 0"`) and is reached only by pushing through the soft
    joint limit (the expert sits at qpos ~ -0.07 at first success) -- and a single axis
    again."""
    return float(abs(float(goal[2]) - float(obj[2])) <= 0.024)


def _success_peg_insert(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_peg_insertion_side_v3.py:115 -- `obj_to_target <= 0.07` with
    `obj_to_target = ||(pegHead - target) * [1, 2, 2]||` (:177): lateral error is
    weighted double -- upstream's comment says to force a lift-then-insert; the hole is
    a 6 cm square channel (both lateral axes get the same weight), not a slot.

    `pegHead` is not in the observation but is recoverable from it. Both sites are on
    the peg body (assets/sawyer_xyz/sawyer_peg_insertion_side.xml:14,16 --
    pegHead at (-0.1, 0, 0), pegGrasp at (0.03, 0, 0.01)), and obs[4:7] is pegGrasp,
    so pegHead = obs[4:7] + R(obs[7:11]) @ (-0.13, 0, -0.01). Verified 1.2e-16 against
    `_get_site_pos("pegHead")` under 30 random peg orientations.
    """
    head = obj + _rotate(obs[7:11], np.array([-0.13, 0.0, -0.01]))
    return float(np.linalg.norm((head - goal) * np.array([1.0, 2.0, 2.0])) <= 0.07)


def _success_window(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_window_open_v3.py:92 and sawyer_window_close_v3.py:96 --
    `target_to_obj <= TARGET_RADIUS` (0.05, :27/:28) with
    `target_to_obj = |obs[4:7][0] - target[0]|` (:136/:143). The window slides in x,
    so both directions share one check."""
    return float(abs(float(obj[0]) - float(goal[0])) <= 0.05)


# --------------------------------------------------------------------------
# the task table
# --------------------------------------------------------------------------


#: The MT10 protocol, frozen -- `bird/tasks.py::METAWORLD_MT10`, which is also what
#: decides the `mt10_`/`mt50_` prefix. Re-exported under the name the tests import.
_EXPECTED_MT10: Tuple[str, ...] = METAWORLD_MT10

#: The other forty tasks of `metaworld.env_dict.MT50_V3`, in that dict's order (the
#: MT50 order is the file order of `env_dict.py`, which is alphabetical except for two
#: swaps upstream never fixed; it is copied verbatim rather than sorted so that the
#: registry order matches the benchmark's). Registered as `mt50_<key>`.
_EXPECTED_MT50_ONLY: Tuple[str, ...] = (
    "assembly-v3", "basketball-v3", "bin-picking-v3", "box-close-v3",
    "button-press-topdown-wall-v3", "button-press-v3", "button-press-wall-v3",
    "coffee-button-v3", "coffee-pull-v3", "coffee-push-v3", "dial-turn-v3",
    "disassemble-v3", "door-close-v3", "door-lock-v3", "door-unlock-v3",
    "hand-insert-v3", "faucet-open-v3", "faucet-close-v3", "hammer-v3",
    "handle-press-side-v3", "handle-press-v3", "handle-pull-side-v3", "handle-pull-v3",
    "lever-pull-v3", "pick-place-wall-v3", "pick-out-of-hole-v3", "plate-slide-v3",
    "plate-slide-side-v3", "plate-slide-back-v3", "plate-slide-back-side-v3",
    "peg-unplug-side-v3", "soccer-v3", "stick-push-v3", "stick-pull-v3", "push-wall-v3",
    "push-back-v3", "reach-wall-v3", "shelf-place-v3", "sweep-into-v3", "sweep-v3",
)

#: Every Meta-World task this module registers: MT10 first, then the other forty.
#: `tasks/` is a hard repo artifact: this module reads the catalogue at import time, so
#: a missing or short one would otherwise register forty-nine env ids and surface a
#: hundred lines away as "unknown key `problem.env_id`" errors on the configs that name
#: the fiftieth. Named here, it fails once and says which task is missing.
_EXPECTED_METAWORLD: Tuple[str, ...] = _EXPECTED_MT10 + _EXPECTED_MT50_ONLY
assert len(set(_EXPECTED_METAWORLD)) == 50, "MT50 is fifty tasks"

#: `s[36:39]` -- `pos_goal`, the LAST block of `SawyerXYZEnv._get_obs`'s
#: `hstack((curr_obs, self._prev_obs, pos_goal))` (see the layout note above
#: `_CURRENT_FRAME_FIELDS`), which is `_target_pos`: the one per-episode quantity in a
#: Meta-World task and the `goal` every check in `_SUCCESS_CHECKS` is handed. Named once
#: so `_success_of` and `discretise` cannot disagree about where it is.
_GOAL = slice(36, 39)



def _metaworld_specs() -> "OrderedDict[str, TaskSpec]":
    """The fifty task specs, MT10 first, cross-checked against the protocol above."""
    have = {s.env_id: s for s in task_index().values() if s.library == "metaworld"}
    missing = [t for t in _EXPECTED_METAWORLD if t not in have]
    extra = sorted(set(have) - set(_EXPECTED_METAWORLD))
    if missing or extra:
        raise TaskSpecError(
            "tasks/: the Meta-World catalogue does not match MT50 -- "
            f"missing {missing}, unexpected {extra}")
    return OrderedDict((t, have[t]) for t in _EXPECTED_METAWORLD)


#: The ONE thing a task spec cannot carry, and the one place a spec file is
#: REFERENCED rather than read.
#:
#: `_success_peg_insert` needs a quaternion rotation and a (-0.13, 0, -0.01) site
#: offset read out of an XML asset, verified to 1.2e-16 against `_get_site_pos`
#: over 30 random orientations. An expression string carrying that is source code
#: with none of Python's tooling, and its provenance docstring -- the most valuable
#: thing in this file -- would have nowhere to live.
#:
#: So each spec pins `discrete_success.shipped.reimplementation.symbol` and it is
#: resolved HERE, through a CLOSED allow-list. Never `importlib` on a string from a
#: data file: run directories on a shared mount may be world-writable, and
#: `checkpoint._DECODABLE` is the precedent -- "decoding never resolves a name
#: outside this dict".


# -- the forty MT50-only tasks. Same signature, same rule: a threshold
# on a quantity recoverable from obs[4:7], obs[7:11], obs[11:14] and obs[36:39].
# Where a task's `evaluate_state` returns a SCALED distance to the reward and an
# unscaled one to `info`, the check follows `info` (coffee-pull/push and soccer
# recompute `norm(obj - target)` in their return tuple "to avoid `scale` above").


def _success_within(radius: float, strict: bool = False
                    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """`||obs[4:7] - goal|| <= radius` (or `<` when `strict`) -- the shape most MT50
    checks take. One factory rather than twenty-one near-identical defs; the radius is
    the task's own literal from its `evaluate_state`, cited on the entry below."""
    def check(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
        d = float(np.linalg.norm(obj - goal))
        return float(d < radius) if strict else float(d <= radius)
    check.__name__ = f"_success_within_{radius}{'_strict' if strict else ''}"
    return check


def _success_axis(axis: int, tol: float) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """`|obs[4:7][axis] - goal[axis]| <= tol`: a press depth or a slide along one axis."""
    def check(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
        return float(abs(float(obj[axis]) - float(goal[axis])) <= tol)
    check.__name__ = f"_success_axis{axis}_{tol}"
    return check


def _success_assembly(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_assembly_peg_v3.py:221 -> `_reward_pos(wrench_center, target)`
    (:155-164): `aligned = ||(target - center)[:2]|| < 0.02` and `hooked =
    (target - center)[2] > 0`. `wrench_center` is the `RoundNut` site at the nut body's
    origin, while obs[4:7] is the `RoundNut-8` site at (0, -0.13, 0) in the nut frame
    (assets/objects/assets/assembly_peg.xml:16-17), so center = obs[4:7] - R(q)(0,
    -0.13, 0) with q = `data.body("RoundNut").xquat` in MuJoCo w,x,y,z order (:107).
    Measured: 0 of 900 mismatches."""
    center = obj - _rotate_wxyz(obs[7:11], np.array([0.0, -0.13, 0.0]))
    err = goal - center
    return float(float(np.linalg.norm(err[:2])) < 0.02 and float(err[2]) > 0.0)


def _success_basketball(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_basketball_v3.py:87 -- `obj_to_target <= TARGET_RADIUS` (0.08, :17)
    with `target[2] = 0.3` and `target_to_obj = ||(obj - target) * [1, 1, 2]||`
    (:150-156): the hoop's height replaces the goal's, and height error counts double.
    The scaled figure IS the one `info` reports here, unlike coffee/soccer."""
    target = np.array(goal, dtype=float).copy()
    target[2] = 0.3
    return float(np.linalg.norm((obj - target) * np.array([1.0, 1.0, 2.0])) <= 0.08)


def _success_disassemble(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_disassemble_peg_v3.py:215 -- `obs[6] > _target_pos[2]`: the nut is
    lifted clear of the peg top. Strict, and a single axis."""
    return float(float(obj[2]) > float(goal[2]))


def _success_faucet_open(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_faucet_open_v3.py:76 -- `target_to_obj <= 0.07` with `obj =
    obs[4:7] + (-0.04, 0, 0.03)` (:133): the reward reads the handle 4 cm inboard and
    3 cm up from the observed site. faucet-close (:77) has no such offset and is a
    plain `_success_within(0.07)`."""
    return float(np.linalg.norm(obj + np.array([-0.04, 0.0, 0.03]) - goal) <= 0.07)


#: `nail_link` y when `NailSlideJoint.qpos == 0`: the box is placed at (0.24, 0.85, 0)
#: on every reset (sawyer_hammer_v3.py:112) and the nail body sits at (0, -0.21, 0.11)
#: in it (assets/objects/assets/hammerblock.xml:8), sliding along +y (:9). Measured
#: `obs[12] - qpos` = 0.64000 on all six instances probed.
_NAIL_Y0 = 0.64


def _success_hammer(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_hammer_v3.py:204 -- `data.joint("NailSlideJoint").qpos > 0.09`. The
    joint is not observed but the nail body is: obs[11:14] is `get_body_com("nail_link")`
    (:91-94, the second object slot), and the slide axis is +y from a fixed box, so
    qpos = obs[12] - _NAIL_Y0. Measured: 0 of 900 mismatches."""
    return float(float(obs[12]) - _NAIL_Y0 > 0.09)


#: `LEVER_RADIUS` (sawyer_lever_pull_v3.py:31): the `leverStart` site is 0.2 m from
#: the hinge (assets/objects/assets/lever.xml:17).
_LEVER_RADIUS = 0.2


def _success_lever_pull(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_lever_pull_v3.py:92 -- `lever_error <= pi/24` with `lever_angle =
    -data.joint("LeverAxis").qpos` (:165) and the desired angle pi/2. The angle is not
    observed but the handle position is: obs[4:7] is the `leverStart` site, which
    rotates on a circle of radius LEVER_RADIUS about a hinge at `_target_pos - (0, 0,
    LEVER_RADIUS)` (reset_model:119-124 places the goal straight above the hinge), so
    the angle is `atan2(dz, -dy)` of the site relative to the hinge. Measured: 0 of 900
    mismatches."""
    pivot = goal - np.array([0.0, 0.0, _LEVER_RADIUS])
    d = obj - pivot
    angle = float(np.arctan2(float(d[2]), -float(d[1])))
    return float(abs(angle - np.pi / 2.0) <= np.pi / 24.0)


def _success_reach_wall(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_reach_wall_v3.py:88 -- `tcp_to_object <= 0.05`, where the quantity
    is `||tcp_center - target||` (:156). The SAME approximation as `_success_reach`,
    for the same reason (tcp_center is not in the observation): 3 of 900 steps
    disagree (0.33%), documented rather than fudged."""
    return _success_reach(obs, obj, goal)


#: The stick's resting height, `init_config["stick_init_pos"][-1]`
#: (sawyer_stick_push_v3.py:45); reset_model keeps z and draws only xy (:152-155).
_STICK_Z0 = 0.02


def _success_stick_push(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_stick_push_v3.py:82-91 -- `grasp_success and ||obs[11:14] -
    target|| <= 0.12`, with `grasp_success = touching_main_object and obs[3] > 0 and
    obs[6] - 0.01 > stick_init_pos[2]`.

    **The second approximation in this file** (after `reach`), and a different kind:
    `touching_main_object` is a CONTACT query (sawyer_xyz_env.py:393-410) and no
    contact is in the 39-D observation. Measured over 900 steps against the live
    query: every touching step has `||tcp_center - stick|| <= 0.0375`, so the
    transcription stands in `tcp_to_stick <= 0.04` for the contact and disagrees with
    `info["success"]` on 9 of 900 steps (1.0%), all of them the gripper hovering
    within 4 cm of a lifted stick without touching it. Tighter thresholds lose
    touching steps (0.03: 125 mismatches); looser ones gain hovering ones. Shipped
    documented: stick-push's ceiling is set by the container, and the conjunct
    the approximation stands in for is the one a policy cannot exploit without also
    holding the stick.
    """
    tcp = obs[0:3] + np.array([0.0, 0.0, -0.045])
    grasped = (float(np.linalg.norm(tcp - obj)) <= 0.04 and float(obs[3]) > 0.0
               and float(obj[2]) - 0.01 > _STICK_Z0)
    return float(grasped and float(np.linalg.norm(obs[11:14] - goal)) <= 0.12)


def _success_stick_pull(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_stick_pull_v3.py:81-84 -- `||handle - target|| <= 0.12 and
    _stick_is_inserted(handle, end_of_stick)` (:186-193: end x >= handle x, |dy| <=
    0.04, |dz| <= 0.06). `handle` is obs[11:14] (the `insertion` site, :107-113) and
    `end_of_stick` is the `stick_end` site at (0.05, 0, 0) in the stick frame
    (assets/objects/assets/stick.xml:5), recovered from obs[4:7] and the scipy-order
    quaternion obs[7:11] (:115-129). Measured: 0 of 900 mismatches."""
    handle = obs[11:14]
    end = obj + _rotate(obs[7:11], np.array([0.05, 0.0, 0.0]))
    inserted = (float(end[0]) >= float(handle[0])
                and abs(float(end[1]) - float(handle[1])) <= 0.04
                and abs(float(end[2]) - float(handle[2])) <= 0.06)
    return float(float(np.linalg.norm(handle - goal)) <= 0.12 and inserted)


def _success_sweep_into(obs: np.ndarray, obj: np.ndarray, goal: np.ndarray) -> float:
    """envs/sawyer_sweep_into_goal_v3.py:81 -- `obj_to_target <= 0.05` with `target =
    (goal_x, goal_y, obj_z)` (:228): height is ignored, so this is an xy radius."""
    return float(np.linalg.norm((obj - goal)[:2]) <= 0.05)


#: Named so a spec can cite the literal. `_success_within`'s factory names are what
#: the specs' `discrete_success.shipped.reimplementation.symbol` carry.
_success_within_0_05 = _success_within(0.05)
_success_within_0_07 = _success_within(0.07)
_success_within_0_08 = _success_within(0.08)
_success_within_0_08_strict = _success_within(0.08, strict=True)
_success_axis_y_0_02 = _success_axis(1, 0.02)
_success_axis_y_0_03 = _success_axis(1, 0.03)
_success_axis_z_0_02 = _success_axis(2, 0.02)
_success_axis_z_0_024 = _success_axis(2, 0.024)
_success_axis_z_0_05 = _success_axis(2, 0.05)
_success_axis_x_0_02 = _success_axis(0, 0.02)


#: The closed allow-list `discrete_success.shipped.reimplementation.symbol` resolves
#: through. A surjection onto the fifty tasks, not a bijection: window-open and
#: window-close are one goal-distance test against two goals, twenty-one MT50 tasks
#: are `||obj - goal|| <= r` for one of four radii and nine are one axis of it within a
#: band. Every entry is a module-level
#: name, so `getattr(module, symbol) is _SUCCESS_CHECKS[symbol]` holds for each.
_SUCCESS_CHECKS: Dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    # -- MT10 --
    "_success_button_press": _success_button_press,
    "_success_door_open": _success_door_open,
    "_success_drawer_close": _success_drawer_close,
    "_success_drawer_open": _success_drawer_open,
    "_success_peg_insert": _success_peg_insert,
    "_success_pick_place": _success_pick_place,
    "_success_push": _success_push,
    "_success_reach": _success_reach,
    "_success_window": _success_window,
    # -- MT50: bespoke --
    "_success_assembly": _success_assembly,
    "_success_basketball": _success_basketball,
    "_success_disassemble": _success_disassemble,
    "_success_faucet_open": _success_faucet_open,
    "_success_hammer": _success_hammer,
    "_success_lever_pull": _success_lever_pull,
    "_success_reach_wall": _success_reach_wall,
    "_success_stick_push": _success_stick_push,
    "_success_stick_pull": _success_stick_pull,
    "_success_sweep_into": _success_sweep_into,
    # -- MT50: a radius on ||obj - goal|| -- the line cited is the `success = ...` /
    #    `"success": ...` line of each task's evaluate_state (read by ast, 3.1.1 wheel)
    #   0.05: bin-picking (:107), hand-insert (:84), sweep (:83)
    #   0.07: coffee-pull (:77), coffee-push (:77), dial-turn (:76), faucet-close (:77),
    #         pick-place-wall (:98), pick-out-of-hole (:79), plate-slide (:80),
    #         plate-slide-side (:78), plate-slide-back (:78), plate-slide-back-side (:93),
    #         peg-unplug-side (:75), soccer (:82), push-wall (:102), push-back (:82),
    #         shelf-place (:81)
    #   0.08: door-close (:112), handle-pull-side (:79)
    #   0.08 strict: box-close (compute_reward:225 and :303, `< 0.08`; info at :83)
    "_success_within_0_05": _success_within_0_05,
    "_success_within_0_07": _success_within_0_07,
    "_success_within_0_08": _success_within_0_08,
    "_success_within_0_08_strict": _success_within_0_08_strict,
    # -- MT50: one axis --
    #   y 0.02: button-press (:74), coffee-button (:80); y 0.03: button-press-wall (:76)
    #   z 0.02: door-lock (:78), handle-press (:78), handle-press-side (:91)
    #   z 0.024: button-press-topdown-wall (:76); z 0.05: handle-pull (:78, TARGET_RADIUS
    #            = the base class's 0.05, sawyer_xyz_env.py:156)
    #   x 0.02: door-unlock (:76)
    "_success_axis_y_0_02": _success_axis_y_0_02,
    "_success_axis_y_0_03": _success_axis_y_0_03,
    "_success_axis_z_0_02": _success_axis_z_0_02,
    "_success_axis_z_0_024": _success_axis_z_0_024,
    "_success_axis_z_0_05": _success_axis_z_0_05,
    "_success_axis_x_0_02": _success_axis_x_0_02,
}


# --------------------------------------------------------------------------
# describe() source material -- shared across all ten tasks
# --------------------------------------------------------------------------
#
# Getting these right is the whole point of the `generate.context.env_spec` axis. An
# empty `_state_fields` -- which is what `Acrobot` ships today -- renders a
# syntactically valid, information-free prompt with no error and no warning, so a
# `--diff` between two methods that differ only on `env_spec` would show a difference
# that has no effect. On a 39-D vector the field docs are the ONLY thing that tells an
# LLM what `s[36:39]` is.
#
# Layout verified against SawyerXYZEnv._get_obs (sawyer_xyz_env.py:513-527):
# `hstack((curr_obs, self._prev_obs, pos_goal))` with `_obs_obj_max_len = 14`.

_CURRENT_FRAME_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("hand_x", "gripper body x, `data.body('hand').xpos`. NOTE this is NOT tcp_center: "
               "the tool centre point used by the built-in rewards sits 4.5 cm below "
               "this in z"),
    ("hand_y", "gripper body y"),
    ("hand_z", "gripper body z; the table surface is near z = 0"),
    ("gripper_opening", "distance between the two claws, normalised to [0, 1]; "
                        "1.0 = fully open, 0.0 = fully closed"),
    ("object_x", "manipulated object x (the handle/button/peg position for the "
                 "fixture tasks)"),
    ("object_y", "manipulated object y"),
    ("object_z", "manipulated object z"),
    ("object_qx", "object orientation quaternion, slot 1 of 4 (the name is a slot label, not "
                  "a component). The component ORDER is per task and is stated in the task's "
                  "spec: scipy (x, y, z, w) where the task converts a rotation matrix, MuJoCo "
                  "(w, x, y, z) where it reads the body's xquat; all four zero on the fixture "
                  "tasks that report none. On the MT10 ten: scipy on reach, push, pick_place, "
                  "door_open and peg_insert_side; MuJoCo on drawer_open and "
                  "button_press_topdown -- where it is also CONSTANT, (1, 0, 0, 0) and "
                  "(0.7074, -0.7068, 0, 0), because those bodies translate on slide joints and "
                  "never rotate; all four components constant zeros on drawer_close, "
                  "window_open and window_close"),
    ("object_qy", "object orientation quaternion, slot 2 of 4 (order per task, see object_qx)"),
    ("object_qz", "object orientation quaternion, slot 3 of 4 (order per task, see object_qx)"),
    ("object_qw", "object orientation quaternion, slot 4 of 4: the scalar w on the scipy-order "
                  "tasks, z on the MuJoCo-order tasks, 0 where the task reports no orientation"),
    ("object2_x", "second object x -- exactly zero (padding) on every MT10 task and on every "
                  "other task with one object; hammer carries the nail here, stick-push and "
                  "stick-pull the container's insertion point"),
    ("object2_y", "second object y -- zero padding (always on MT10) except on hammer, "
                  "stick-push, stick-pull"),
    ("object2_z", "second object z -- zero padding (always on MT10) except on hammer, "
                  "stick-push, stick-pull"),
    ("object2_qx", "second object quaternion, slot 1 of 4 -- zero (always on MT10) except on "
                   "hammer (the nail body's xquat, w, x, y, z)"),
    ("object2_qy", "second object quaternion, slot 2 of 4 -- zero except on hammer"),
    ("object2_qz", "second object quaternion, slot 3 of 4 -- zero except on hammer"),
    ("object2_qw", "second object quaternion, slot 4 of 4 -- zero except on hammer"),
)

_STATE_FIELDS: Tuple[Tuple[str, str], ...] = (
    _CURRENT_FRAME_FIELDS
    + tuple((f"prev_{name}",
             f"the PREVIOUS step's {name}, verbatim. Velocity is not observed: "
             f"finite-difference s[0:18] - s[18:36] and divide by dt = {_DT}")
            for name, _doc in _CURRENT_FRAME_FIELDS)
    + (("goal_x", "goal position x for this episode; re-randomised every reset"),
       ("goal_y", "goal position y"),
       ("goal_z", "goal position z"))
)

_ACTION_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("delta_x", "commanded gripper displacement along x, in [-1, 1]; scaled and "
                "integrated into the mocap target by `set_xyz_action`, so this is a "
                "velocity command, not a position"),
    ("delta_y", "commanded gripper displacement along y, in [-1, 1]"),
    ("delta_z", "commanded gripper displacement along z, in [-1, 1]"),
    ("gripper", "gripper command in [-1, 1]; -1 opens, +1 closes"),
)

#: Populated ONLY for `pythonic_class_abstraction` -- the base class renders
#: `_helpers` in that one rendering and omits it from the api stub, which is the
#: distinction between the GT-style stub and the Text2Reward/CARD abstraction.
#:
#: They are advertised as INLINEABLE EXPRESSIONS and deliberately not implemented as
#: methods. `training.CompiledReward._make_binder` (training.py:167) passes None for a
#: `self` parameter, while `verification.call_reward` passes `_SelfProxy(ctx.env)`
#: (verification.py:307) -- so a helper that existed on the adapter would let a
#: `self.hand_pos(s)` candidate pass §2 verification and then die in §3 with an
#: AttributeError on None, i.e. a wasted 500-step training run misattributed to the
#: reward. Advertising expressions keeps the method distinction real without building
#: that trap.
_HELPERS: Tuple[Tuple[str, str], ...] = (
    ("hand_pos(s)", "= s[0:3]"),
    ("object_pos(s)", "= s[4:7]"),
    ("object_quat(s)", "= s[7:11]; component order is per task, see the state table "
                       "(object_qx). On MT10: scipy x,y,z,w on reach/push/pick_place/door_open/"
                       "peg_insert_side, MuJoCo w,x,y,z on drawer_open/button_press_topdown, "
                       "zeros on drawer_close/window_open/window_close"),
    ("goal_pos(s)", "= s[36:39]"),
    ("gripper_opening(s)", "= s[3], 1.0 open"),
    ("hand_velocity(s)", f"= (s[0:3] - s[18:21]) / {_DT}"),
    ("tolerance(x, bounds, margin, sigmoid='long_tail')",
     "Meta-World's shaping kernel: 1.0 inside `bounds`, decaying over `margin`"),
    ("hamacher_product(a, b)", "= a*b / (a + b - a*b); Meta-World's soft AND"),
)


# --------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------


class MetaWorld(SpecEnvAdapter):
    """One Meta-World MT50 task, selected by the `task` constructor argument.

    One class and a fifty-row table, not fifty subclasses: everything that differs
    between the tasks is data (prose, success check, symbol mapping, measured
    baselines), and a subclass per task would be fifty places for the shared
    snapshot/cache logic to drift.

    The three EnvAdapter conventions this env introduces, all forced by MuJoCo being
    stateful and all documented at module level: emitted states are read-only, an
    unrecognised state is an error rather than a guess, and `_step` restores before it
    steps.

    Measured (pick-place, 500 steps each): a sequential step costs
    0.754 ms and a step that has to restore costs 0.822 ms, so the WORST case -- every
    single step forced to restore -- is +9%, and the harness only forces one at a
    planner branch or a checkpoint interleave. `reset()` is 10.7 ms, which is 2% of a
    500-step episode. The functional interface costs a few percent of throughput, not
    a rewrite.
    """

    # 39-D observation, 4-D action, 500-step episode: `SawyerXYZEnv.max_path_length`.
    obs_dim = 39
    action_dim = 4
    horizon = 500

    #: MUST stay None. `_sb3_run` derives `continuous` solely from
    #: `exact_states is None` (training.py:5512), so any integer here silently hands
    #: SAC a `Discrete` action space over the 15-row proxy set and turns "real SAC on
    #: Meta-World" into a 15-armed bandit.
    exact_states = None

    #: Mixed-radix binning of six TASK-RELATIVE scalars; see `discretise`.
    n_disc_states = 5 * 5 * 3 * 4 * 3 * 4  # = 3600

    #: 1 / (2 * horizon) = 0.001, and this small number is the whole trick.
    #: `EnvAdapter.success()` is inherited as `task_metric >= success_threshold`, and
    #: `task_metric` here is the FRACTION of steps in the goal region. One successful
    #: step out of <= 501 gives 0.002 > 0.001; zero gives 0.0 < 0.001. So `success()`
    #: is exactly Meta-World's published any-step criterion `max_t success_t` while
    #: `task_metric` stays dense -- and the two cannot disagree, because one is a
    #: threshold on the other. Both are reachable from config with no new key:
    #: `evaluate.fitness.source: ground_truth_metric` takes the mean,
    #: `success_rate` takes the binary.
    #:
    #: Any-step is the right criterion and final-step is not: measured with the
    #: benchmark's own scripted experts over full episodes, success@final_step is 0 on
    #: 3 of 10 tasks (push, door-open, peg-insert) because the object drifts back out.
    success_threshold = 1.0 / (2.0 * 500.0)

    #: Both, because `task_metric` here IS a reduction of a per-step success check and the
    #: sparse reading is the one the benchmark publishes. `__init__` picks between them.
    supported_reductions = (PER_STEP, ANY_STEP)

    #: 8192 entries; measured 26.5 MB (9.9 MB pickled at 3070 entries, scaled). Never cleared on reset -- clearing it is exactly
    #: what breaks the `_sb3_run` eval interleave, where `_Gym._s` is a state emitted
    #: before another consumer reset the shared adapter.
    #:
    #: THE ONE HONEST DEGRADATION, and it is a recording one, never a training one.
    #: Training measured 0 misses on all four backends (sb3/SAC 600 steps through the
    #: eval interleave, mock/CEM and tabular/Q-learning at 3000, planner at 3000).
    #: What does miss is a LATE render: `observability.record_rollouts` replays a
    #: trajectory stored at the end of a candidate's training, and everything the
    #: NEXT candidate emits pushes it out of the LRU. Measured end to end
    #: (Eureka with a mock LLM, sb3/SAC): at `generate.n_candidates: 1` the rollout renders
    #: eight distinct frames; at 2 the OLDER candidate's rollout is already gone.
    #: Roughly 11k states are emitted per candidate (2000 training steps plus 5
    #: checkpoints x 3 x 500 evaluation steps plus the recorded rollouts), so this
    #: fires at the SMALLEST budget the harness can run and on every backend -- it is
    #: not scoped to Eureka-sized budgets or to the planner. Under
    #: `train.algorithm: none` the planner emits ~25x more states than it executes and
    #: hits the same wall inside one candidate.
    #:
    #: What it costs is one candidate's video and not the iteration's:
    #: `observability._render_frames` returns `[]` for the evicted trajectory,
    #: carries on with the remaining picks, and emits a `record_skipped` journal
    #: event -- so the loss is visible to a collector rather than only in the log.
    #:
    #: Raising the cap does not fix it: 16384 (54 MB) still misses, and no cap
    #: survives `generate.n_candidates: 16` at Eureka budgets. Neither does a
    #: trunk/branch two-tier cache -- a lookahead chain is a sequential chain, so no
    #: adapter-level heuristic distinguishes the executed trajectory from a planned or
    #: a finished one.
    #:
    #: The fix is `retain_states`, below: not a heuristic and not a bigger LRU, but
    #: the CALLER saying which rows will be replayed. See that method.
    snapshot_cache_size = 8192

    #: Entries in the RETAINED store, which the LRU cannot evict. Sized from what
    #: is retained -- `training._n_replayed_rollouts(cfg)` rollouts of each
    #: candidate in the iteration being recorded, 501 rows each on this env's
    #: 500-step horizon -- so 32768 is 65 such rollouts, against a wide
    #: configuration of `generate.n_candidates: 8` x 3 replayed rollouts
    #: (8 x 3 x 501 = 12,024). At the measured 3169 B/state that ceiling is
    #: ~104 MB and the 12,024-row point is ~38 MB, but the store is CLEARED every
    #: iteration by `observability.record_rollouts`, so the steady state is one
    #: iteration's rollouts and not the cap.
    #:
    #: Overflow is FIFO and therefore drops the OLDEST retained rollout first,
    #: which is the same "the last N candidates keep their video" degradation the
    #: LRU has -- deliberately, because a retained store that raised would turn a
    #: missing video into a dead run.
    pinned_cache_size = 32768

    #: `MT1(task, seed)` pre-generates 50 goals; fixing the seed means every run
    #: shares the same 50 and only the per-episode CHOICE varies with `cfg.seed`.
    #: That is the right default -- a cross-seed comparison should vary initial
    #: conditions, not the benchmark -- and it also matters that it is not per-run:
    #: `MT1` costs 714 ms and briefly reseeds the GLOBAL numpy RNG while it works.
    benchmark_seed = 0

    #: Rendered at 480 and downscaled by `observability._resize_nearest`. Cost is flat
    #: in resolution (45.0 ms at 84x84, 48.8 at 480x480): it is `mjv_updateScene` over
    #: 37 geoms plus the Sawyer meshes, not rasterisation, so there is nothing to gain
    #: by rendering small.
    render_width = 480

    #: `corner` is the canonical third-person Meta-World view -- and it is defined
    #: UPSIDE DOWN, which `render` corrects. See the derivation there. Measured frame
    #: std over the seven cameras (a rough proxy for how much of the frame is scene
    #: rather than backdrop): corner3 68.3, topview 60.0, corner2 56.6, corner4 47.6,
    #: gripperPOV 39.7, corner 37.4, behindGripper 35.3. The two upright alternatives
    #: were tried and rejected on evidence: `topview` flattens away the lift axis,
    #: which is the whole task on pick-place and peg-insert, and `behindGripper` is
    #: `mode="track"` (xyz_base.xml:152) so it is occluded by the arm's own links --
    #: measured 3 of 6 recorded frames at 72% red robot on a pick-place rollout.
    #: `corner` alone is exactly 180-degree rolled (below), so it is the one that can
    #: be corrected without guessing.
    camera_name = "corner"

    #: Six multiplicative scales on the values captured at construction; nominal 1.0.
    #: Declared as OUTER bounds -- finding the feasible bracket inside them is RAPP's
    #: job (`pre: [rapp]`, DrEureka §3). Ranges chosen from a measured sweep:
    #:
    #:   object_mass_scale    body_mass of the manipulated bodies. mass 0.75 -> 5.0
    #:                        takes the scripted expert from sum_reward 1541 to 1156.
    #:   arm_damping_scale    dof_damping[0:7], the seven Sawyer joints (nominal 10.0).
    #:                        x4 takes the expert from 1541 to 22.9 with success 0 --
    #:                        this axis alone spans solvable -> unsolvable, which is
    #:                        exactly what a feasibility sweep needs.
    #:   friction_scale       geom_friction[:,0] on the object AND the gripper pads
    #:                        TOGETHER. Perturbing the object alone measures exactly
    #:                        0.00000 effect: MuJoCo mixes contact friction as the
    #:                        elementwise MAX and the pads at 2.0 dominate the object
    #:                        at 1.0. Scaled together, 0.05 takes the expert from 1541
    #:                        to 323. The single most misleading DR axis here.
    #:   gravity_scale        model.opt.gravity. Weak under random actions (the arm is
    #:                        position-controlled through a mocap weld) but it is the
    #:                        standard sim2real axis and it is cheap.
    #:   actuator_gain_scale  actuator_gainprm[:,0] (nominal 400). Largest single-axis
    #:                        observation effect measured: 400 -> 250 gives 0.227.
    #:   weld_softness_scale  eq_solref[:,0], the mocap weld = arm tracking compliance.
    #:                        The closest thing Meta-World has to actuator lag; 0.02 ->
    #:                        0.05 gives max|dobs| 0.042.
    #:
    #: Geometry is deliberately absent: the success checks are defined against
    #: fixed object dimensions, so randomising those moves the goalposts rather than
    #: the world.
    #:
    #: MEASUREMENT WARNING for anyone running RAPP here: two of these axes measure as
    #: no-ops under a random policy, because a random policy never makes contact. A
    #: sweep must be driven by `metaworld.policies.ENV_POLICY_MAP[task]`, which
    #: reaches success on all ten tasks.
    #:
    #: THREE OF THE SIX ARE LESS LIVE THAN THEIR NAMES SAY, measured on the installed
    #: metaworld 3.1.1 models:
    #:
    #:   object_mass_scale    matches NO body on door-open, drawer-open, drawer-close,
    #:                        window-open and window-close (`_name_matched_bodies`'s keys
    #:                        name nothing there), and on button-press it scales the
    #:                        11.24 kg case as well as the 0.01 kg button. It scales
    #:                        `body_mass` only, never `body_inertia` -- translational.
    #:   friction_scale       scales a NON-COLLIDING visual mesh on door/drawer/button and
    #:                        nothing object-side on the window pair; and the table, walls
    #:                        and fixture colliders all sit at friction 1.0 with no geom
    #:                        priority, so under MuJoCo's max-mixing a scale below 1 never
    #:                        lowers the friction the puck slides on. The lower half of
    #:                        this axis is inert on every task.
    #:   actuator_gain_scale  `nu = 2`: the only actuators are the gripper's two position
    #:                        servos (kp 400); the arm is unactuated and dragged by the
    #:                        mocap weld. Scaling `gainprm[:, 0]` without `biasprm[:, 1]`
    #:                        moves the servo's SETPOINT / grip force, not its stiffness.
    #:
    #: These six are DrEureka's published axes and are left exactly as published. The
    #: two below are BIRD additions, made rather than redefining a published one:
    #:
    #:   contact_friction_scale  geom_friction[:, 0] on EVERY colliding geom that is not
    #:                        a Sawyer arm link: the object/peg, the pads and claws, the
    #:                        table top, the retaining walls, the floor and the fixture
    #:                        links' colliders. Scaling every side of every contact is the
    #:                        only way a factor below 1 survives max-mixing.
    #:   fixture_damping_scale   dof_damping of the fixture's own joint -- `doorjoint`,
    #:                        `goal_slidey`, `btnbox_joint`, `window_slide` (nominal 2.0,
    #:                        2.0, 1.0, 2.0) -- the one axis the six fixture tasks
    #:                        actually need; empty (structurally inert) on reach, push,
    #:                        pick-place and peg-insert.
    dr_parameters = {
        "object_mass_scale": (0.2, 4.0),
        "arm_damping_scale": (0.5, 4.0),
        "friction_scale": (0.05, 4.0),
        "gravity_scale": (0.7, 1.3),
        "actuator_gain_scale": (0.5, 1.75),
        "weld_softness_scale": (0.75, 3.0),
        "contact_friction_scale": (0.1, 4.0),
        "fixture_damping_scale": (0.1, 10.0),
    }
    _dr_nominal = {k: 1.0 for k in dr_parameters}

    #: The fixture joints `fixture_damping_scale` scales, by name. One per fixture task;
    #: the free-object tasks have none, so the axis writes nothing there.
    _FIXTURE_JOINTS = ("doorjoint", "goal_slidey", "btnbox_joint", "window_slide")
    #: Body-name prefixes of the Sawyer's own links, which `contact_friction_scale` leaves
    #: alone: arm-on-arm contact is not part of any task, and the pads/claws are named
    #: separately (they collide with the object and ARE scaled).
    _ROBOT_LINK_PREFIXES = ("right_", "base", "controller_box", "pedestal", "torso",
                            "screen", "head")

    #: Defaults only. `_apply_spec` overrides all three with the spec's own tables at
    #: construction, and `tests/test_task_specs.py` pins the two equal. They stay here
    #: only so that the identity is checkable; the spec is the definition.
    _state_fields = _STATE_FIELDS
    _action_fields = _ACTION_FIELDS
    _helpers = _HELPERS

    #: mjMODEL arrays DR writes to. Kept separate from `_MJ_MODEL` (the snapshot list)
    #: because `_apply_dr` must rewrite from the NOMINAL copy every time -- scaling in
    #: place would compound across episodes.
    _DR_MODEL_FIELDS = ("body_mass", "dof_damping", "geom_friction",
                        "actuator_gainprm", "eq_solref")

    # -- construction -------------------------------------------------------

    def __init__(self, task: str, reduction: str = PER_STEP) -> None:
        if reduction not in self.supported_reductions:
            raise ValueError(
                f"reduction {reduction!r} not in {self.supported_reductions}")
        self.reduction = reduction
        specs = _metaworld_specs()
        if task not in specs:
            raise KeyError(f"unknown Meta-World task {task!r}; MT50 is {list(specs)}")
        self.task = task
        self.name = _env_id(task)

        # The success check the spec NAMES, resolved through the closed allow-list above.
        _shipped = specs[task].discrete_success.get("shipped") or {}
        _reimpl = _shipped.get("reimplementation") or {}
        symbol = _reimpl.get("symbol")
        if not symbol:
            raise TaskSpecError(
                f"{specs[task].path}: no discrete_success.shipped.reimplementation.symbol. "
                "Every Meta-World spec must name the success check this adapter runs; without "
                "it there is no ground truth and `task_metric` would score nothing")
        if symbol not in _SUCCESS_CHECKS:
            raise TaskSpecError(
                f"{specs[task].path}: names success check {symbol!r}, which is not in this "
                "module's _SUCCESS_CHECKS allow-list. Names from a data file are resolved "
                "against that dict and nowhere else")
        self._success_check = _SUCCESS_CHECKS[symbol]

        # Before `import mujoco`, and inside __init__ rather than at module scope --
        # see the module docstring. `or` rather than setdefault: an empty MUJOCO_GL is
        # a value, and gymnasium rejects it.
        os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL") or "osmesa"
        # Strictly before the simulator imports (`import metaworld` pulls in mujoco
        # itself, so this cannot sit between them) -- see the docstring: the reverse
        # order is a SIGSEGV at stage [3] train.
        _preload_llvm_before_mujoco()
        try:
            import metaworld
            import mujoco
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise ImportError(
                f"problem.env_id: {self.name} needs metaworld and mujoco, which are not "
                "installed. Either\n"
                "    uv sync --extra metaworld\n"
                "    (or: pip install 'metaworld>=3.0' 'mujoco==3.3.0' 'gymnasium>=1.1')\n"
                "or run the same method point on an env that needs no simulator, such as\n"
                "the pendulum control task:\n"
                "    problem.env_id: pendulum\n"
                f"(underlying import error: {exc})"
            ) from exc
        self._mj = mujoco

        # THE construction route, and the only one that works. `gym.make('Meta-World/
        # MT1', ...)` wraps the env in a `RandomTaskSelectWrapper` that RESAMPLES the
        # goal on every reset and ignores `reset(seed=)` -- measured four different
        # goals from four `reset(seed=0)` calls, which makes a determinism test
        # unpassable and means two candidates are not compared on the same task. The
        # `*_GOAL_OBSERVABLE` route pins the goal but its generated `__init__` is only
        # `(env, seed, render_mode)`, so width/height/camera_name cannot be passed --
        # and setting `env.width` afterwards is a no-op, because `MujocoRenderer`
        # captured it. MT1 + set_task is the only route that pins the goal AND takes
        # constructor kwargs.
        bench = metaworld.MT1(task, seed=self.benchmark_seed)
        self._tasks = list(bench.train_tasks)      # 50, `_N_GOALS` in metaworld/__init__.py
        self._env = bench.train_classes[task](
            render_mode="rgb_array", camera_name=self.camera_name,
            width=self.render_width, height=self.render_width)
        self._task_cls = type(self._env)
        self._env.set_task(self._tasks[0])
        # Per-task snapshot attributes, on top of the base-class lists: whatever THIS
        # task's `reset_model` assigns on `self` is episode state, whatever its
        # `compute_reward`/`evaluate_state` assign is step state (bin-picking latches
        # `_target_to_obj_init` on the first reward call; the v1 reward paths keep
        # `pickCompleted`/`placeCompleted`). Read off the class by `ast` at
        # construction, so a task the lists were never hand-tuned for restores
        # everything its own reward reads; `_PY_NEVER` keeps the two live views out.
        # Held bitwise on all fifty by `test_every_metaworld_task_restores_bitwise`.
        self._py_episode = _PY_EPISODE + _assigned_attrs(self._task_cls, ("reset_model",))
        self._py_step = _PY_STEP + _assigned_attrs(
            self._task_cls, ("compute_reward", "evaluate_state", "step"))

        # Nominal physics, captured once. `_apply_dr` always writes scale x nominal,
        # never scale x current, so DR draws cannot compound across episodes.
        self._nominal = {f: np.copy(getattr(self._env.model, f))
                         for f in self._DR_MODEL_FIELDS}
        self._nominal["gravity"] = np.copy(self._env.model.opt.gravity)
        self._dr_bodies = self._name_matched_bodies()
        self._dr_geoms = self._name_matched_geoms()
        self._dr_colliders = self._colliding_geoms()
        self._fixture_dofs = self._fixture_joint_dofs()

        # obs -> (episode record, step record). One LRU, never cleared.
        self._cache: "OrderedDict[bytes, Tuple[dict, dict]]" = OrderedDict()
        # The same records, for states a caller declared it will replay later.
        # The LRU cannot reach into here; `retain_states` documents why.
        self._pinned: "OrderedDict[bytes, Tuple[dict, dict]]" = OrderedDict()
        self._cur: Optional[bytes] = None
        self._episode: Optional[dict] = None
        self._pool: List[np.ndarray] = []
        self.cache_misses = 0
        self._reward_source: Optional[str] = None

        # Per-task describe()/T2R material: prose, the observation and action tables,
        # the helper vocabulary, the symbol map and the DR axes all come from
        # `tasks/<id>/shared_spec.yaml` -- one home, and a diffable one. What stays in
        # Python is what a data file cannot carry: the dynamics, and the check.
        task_spec = by_env_id(self.name)
        if task_spec is None:
            raise TaskSpecError(
                f"no task spec backs {self.name!r}. The catalogue defines every "
                "Meta-World task; without it this adapter has no description, no symbol "
                "table and no randomisation axes, and would render an information-free "
                "prompt rather than fail")
        self._apply_spec(task_spec)
        if self.primary_view.name != self.camera_name:
            raise TaskSpecError(
                f"{task_spec.path}: judge.camera.name is {self.primary_view.name!r} but "
                f"this adapter renders {self.camera_name!r}. The spec NAMES the primary "
                "panel to the judge (multiview.describe), so the two must agree")
        #: The (random, human) anchors `evaluate.fitness.normalisation:
        #: human_normalised` divides by (evaluation.py:664-676, via `_env_baselines`).
        #: Keyed by REDUCTION, because `task_metric` (fraction of steps) and `success()`
        #: (the same check on at least one step) are different numbers -- 0.15 vs
        #: 0.25 for a uniform-random policy on drawer-close (n=100). A null anchor stays None, which
        #: makes `_normalise_pool` leave fitness raw rather than invent a scale.
        self.baselines = _baselines_of(task_spec, self.reduction)

        #: Under the sparse reduction `task_metric` is already 0/1 per episode, so the
        #: threshold that keeps `success()` the SAME check is a half, not 1/(2H).
        #: One is a threshold on the other under either reading -- that invariant is
        #: what stops the two ever disagreeing about whether a task was done.
        if self.reduction == ANY_STEP:
            self.success_threshold = 0.5

        super().__init__()

    def _name_matched_bodies(self) -> List[int]:
        """mjBODY ids of the manipulated things, by name.

        By name and not by index because the indices differ per task (obj is body 33
        on pick-place and does not exist under that name on the fixture tasks). The
        Sawyer's own links are excluded, which is the point: `object_mass_scale` must
        mean "the thing being manipulated", not "the robot".
        """
        # Two rules, unioned. The NAME row is the MT10 set and is not edited: changing
        # it would move the DR semantics of every existing MT10 config point. The
        # FREE-JOINT rule adds every body in the subtree of a body that owns a free
        # joint -- the free objects the forty MT50 tasks add (basketball, nut, lid,
        # mug, hammer, plug, ball, stick) carry their mass on an unnamed or
        # differently-named CHILD of the named free body (`stick` is 0 kg and its
        # unnamed child 0.02 kg; `plug1` 0 kg and `plug` 0.08 kg), so a name match
        # would scale nothing. Measured on all fifty models: on the ten MT10 tasks the
        # union equals the name-only set exactly (their free bodies `obj`/`peg` are
        # named), so the free-joint rule leaves MT10 DR unchanged. Still dead, and
        # stated: the four plate-slides (the puck is a geom on a slide-jointed body,
        # not a free one; the name row matches only the two 0 kg channel fixtures) and
        # the fixtures with no free body -- window, door, drawer, faucet, dial, lever.
        keys = ("obj", "peg", "puck", "block", "handle", "btn", "button")
        m = self._env.model
        out = set()
        for i in range(1, m.nbody):
            name = self._mj.mj_id2name(m, self._mj.mjtObj.mjOBJ_BODY, i) or ""
            if any(k in name for k in keys):
                out.add(i)
        free_roots = {int(m.jnt_bodyid[j]) for j in range(m.njnt)
                      if m.jnt_type[j] == self._mj.mjtJoint.mjJNT_FREE}
        subtree = set(free_roots)
        grown = bool(subtree)
        while grown:
            grown = False
            for i in range(1, m.nbody):
                if i not in subtree and int(m.body_parentid[i]) in subtree:
                    subtree.add(i)
                    grown = True
        #: Kept for `_name_matched_geoms`: the free objects' collision geoms are as
        #: unnamed as their mass-carrying bodies, so friction follows the same subtree.
        self._dr_free_subtree = sorted(subtree)
        return sorted(out | subtree)

    def _name_matched_geoms(self) -> List[int]:
        """mjGEOM ids whose sliding friction `friction_scale` scales.

        Includes the gripper PADS, not just the object. MuJoCo mixes contact friction
        as the elementwise max of the two geoms, and the pads (nominal 2.0) dominate
        the object (nominal 1.0), so scaling the object alone measures exactly 0.00000
        change in the observation -- an axis that looks live in the config and is dead
        in the physics.

        The NAME row is the MT10 set and is not edited. Unioned with the collision
        geoms (`contype | conaffinity` nonzero) of every body in the free-joint
        subtree `_name_matched_bodies` recorded: the MT50 free objects' geoms are
        unnamed on assembly/disassemble (the nut), bin-picking, box-close (plus the
        named `BoxHandleGeom`), the three coffee tasks (the mug), hammer and
        peg-unplug-side, so a name match alone would leave `friction_scale` dead on
        them. Measured on all fifty models: on the ten MT10 tasks the union equals the
        name-only set (their free bodies `obj`/`peg` carry exactly the named
        `objGeom`/`peg` collision geom). Still dead (pads only), and measured rather
        than assumed: the four handle tasks (the handle geoms are unnamed and the
        handle is hinged, not free), door-lock and door-unlock (the lock lever's geom
        is unnamed), and the fixtures with no named or free geom -- window, faucet,
        dial, the four plate-slides. door-open/close's `handle`, the drawers'
        `objGeom` and the buttons' `btnGeom` are named and live.
        """
        keys = ("objGeom", "pad", "peg", "handle", "btnGeom")
        m = self._env.model
        out = set()
        for i in range(m.ngeom):
            name = self._mj.mj_id2name(m, self._mj.mjtObj.mjOBJ_GEOM, i) or ""
            if any(k in name for k in keys):
                out.add(i)
        free = set(getattr(self, "_dr_free_subtree", ()))
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) in free and (int(m.geom_contype[g]) | int(m.geom_conaffinity[g])):
                out.add(g)
        return sorted(out)

    def _colliding_geoms(self) -> List[int]:
        """mjGEOM ids `contact_friction_scale` scales: every geom that can collide and
        does not belong to a Sawyer arm link.

        By collision flags and not by name, because the names are the problem:
        `_name_matched_geoms` keys the door handle MESH (contype 0) and misses the five
        unnamed `dl_col` cylinders the pads actually touch. The table top, walls and
        floor are included on purpose -- MuJoCo takes the elementwise MAX of the two
        geoms' friction, so a factor below 1 that leaves the table at 1.0 changes
        nothing the puck slides on (measured).
        """
        model = self._env.model
        out = []
        for g in range(model.ngeom):
            if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
                continue
            body = self._mj.mj_id2name(model, self._mj.mjtObj.mjOBJ_BODY,
                                       int(model.geom_bodyid[g])) or ""
            if body.startswith(self._ROBOT_LINK_PREFIXES):
                continue
            out.append(g)
        return out

    def _fixture_joint_dofs(self) -> List[int]:
        """dof addresses of `_FIXTURE_JOINTS` present in this task's model (0 or 1)."""
        model = self._env.model
        out = []
        for j in range(model.njnt):
            name = self._mj.mj_id2name(model, self._mj.mjtObj.mjOBJ_JOINT, j) or ""
            if name in self._FIXTURE_JOINTS:
                out.append(int(model.jnt_dofadr[j]))
        return out

    def _build_action_set(self) -> Any:
        """15 rows, and each part of that number is load-bearing.

        `EnvAdapter.__init__` DERIVES `action_low`/`action_high` from this set's
        per-axis min/max, and `_sb3_run` builds SAC's `Box` from those -- so the set
        must hit -1 and +1 on all four axes or the continuous policy can never command
        a corner. It must also be able to translate WHILE gripping, or the tabular and
        planner backends cannot express "carry" and pick-place is unsolvable by
        construction for them. Row 0 is the zero action because `_run_backend` uses
        `action_set[0]` as its no-policy fallback (training.py:1129) and
        `_PlannerLearner` as `best_first` -- "do nothing" is the right default there,
        "slam -x with the gripper closed" is not.
        """
        rows = [[0.0, 0.0, 0.0, 0.0],       # the fallback / planner default
                [0.0, 0.0, 0.0, -1.0],      # open in place
                [0.0, 0.0, 0.0, 1.0]]       # close in place
        for axis in range(3):
            for sign in (-1.0, 1.0):
                for grip in (-1.0, 1.0):
                    row = [0.0, 0.0, 0.0, grip]
                    row[axis] = sign
                    rows.append(row)
        return np.asarray(rows, dtype=float)

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """`sawyer_observation_space`, NOT `observation_space`.

        metaworld 3.1.1 bug: `MujocoEnv.__init__` stores the space computed while
        `_partially_observable` was still True, so `env.observation_space` reports the
        goal dims as low == high == 0 and `observation_space.contains(obs)` is False
        on every construction route. `sawyer_observation_space` is a cached_property
        that recomputes and is the one `step()` itself clips against
        (sawyer_xyz_env.py:625-631). Measured through this adapter: 0 degenerate dims
        and 0 out-of-bounds observations over 153 obs x 3 tasks, against 3 degenerate
        dims via the stale space.

        Clipped to +/-10, and NOT for SB3. The raw box is +/-inf on 28 of its 39 dims --
        the two 14-wide object blocks, dims 4-17 and 22-35 (measured on `reach-v3` and
        `pick-place-v3`, metaworld 3.1.1); hand xyz and gripper in both frames, and the
        goal, are finite and pass through the clip unchanged. The clip is NOT there
        because "`spaces.Box` with an infinite bound upsets SB3's normalisation";
        measured (SB3 2.9.0, gymnasium 1.3.0, torch 2.13.0, numpy 2.4.6): SAC,
        PPO and SAC+VecNormalize, each trained 512 steps from seed 0 on one toy env under
        `Box(-inf, inf)`, `Box(-10, 10)` and `Box(-20, 20)` -- float32, as
        `training.py::_Gym` casts -- produce BIT-IDENTICAL deterministic actions on a
        fixed batch of 64 observations, finite parameters, finite `obs_rms`, and
        `check_env` warns on none. `MlpPolicy` never reads a Box's bounds, this repo uses
        no `VecNormalize`, and gymnasium's own MuJoCo envs declare `Box(-inf, inf)`.

        Nothing in this repo needs the clip either. `random_state` below is a pool of
        reachable states, not `rng.uniform(obs_low, obs_high)` -- the draw is where an
        infinite bound does bite (it raises `OverflowError`; see `humanoid_hand._bounds`,
        whose box IS finite for that reason). `discretise` bins six task-relative features
        with their own `(lo, hi)`. `training.CompiledObservation._measure` clips the
        declared bounds to `_OBS_PROBE_CLIP` (also 10) before building its probe lattice.
        `fasttd3` reads `.size` only.

        What the clip IS: a finite declaration of where an observation can be, so that
        `obs_low <= obs <= obs_high` is a real check on all 39 dims instead of vacuous on
        28 -- `tests/test_metaworld.py` holds every emitted and every sampled state to it.
        10 m is two orders of magnitude outside the 1 m workspace, so nothing real is clipped, and
        by the measurement above +/-10 against +/-inf is unobservable in training. The
        number stays because moving it would change the declared observation space for
        no measurable gain.
        """
        space = self._env.sawyer_observation_space
        return (np.clip(np.asarray(space.low, dtype=float), -10.0, 10.0),
                np.clip(np.asarray(space.high, dtype=float), -10.0, 10.0))

    # -- the observation -> simulator-snapshot cache ------------------------

    @staticmethod
    def _key(obs: Any) -> bytes:
        """Cache key: the observation as float32 bytes.

        float32 and not float64 on purpose -- `_Gym` hands SB3 `np.float32` and hands
        it back, so a float32 round trip must still hit. 39 float32 = 1248 bits, so
        collisions are negligible, and a collision costs a ~1e-7 perturbation rather
        than a wrong episode.
        """
        return np.asarray(obs, dtype=np.float32).ravel().tobytes()

    @staticmethod
    def _emit(obs: Any) -> np.ndarray:
        """Every observation leaving this adapter is a frozen copy. See the module
        docstring: Meta-World's own scripted policies mutate observations in place."""
        arr = np.array(obs, dtype=float, copy=True)
        arr.setflags(write=False)
        return arr

    def _episode_snapshot(self) -> dict:
        """The ~4.6 KB that is constant within an episode; shared by reference across
        every step record of that episode, so the per-step cost stays ~3.3 KB."""
        env = self._env
        return {"model": {f: np.copy(getattr(env.model, f)) for f in _MJ_MODEL},
                "gravity": np.copy(env.model.opt.gravity),
                "py": {a: copy.deepcopy(getattr(env, a))
                       for a in self._py_episode if hasattr(env, a)}}

    def _step_snapshot(self) -> dict:
        env = self._env
        return {"mj": {f: np.copy(getattr(env.data, f)) for f in _MJ_DATA},
                "warmstart": np.copy(env.data.qacc_warmstart),
                "time": float(env.data.time),
                "py": {a: copy.deepcopy(getattr(env, a))
                       for a in self._py_step if hasattr(env, a)}}

    def _store(self, obs: np.ndarray) -> bytes:
        key = self._key(obs)
        self._cache[key] = (self._episode, self._step_snapshot())
        self._cache.move_to_end(key)
        while len(self._cache) > self.snapshot_cache_size:
            self._cache.popitem(last=False)
        self._cur = key
        return key

    def _lookup(self, obs: Any) -> Tuple[bytes, Tuple[dict, dict]]:
        key = self._key(obs)
        record = self._cache.get(key)
        if record is not None:
            # Keeps the planner's branch root hot: it is re-visited `n_samples` times
            # while its children are visited once each.
            self._cache.move_to_end(key)
            return key, record
        # Second, and only second: a retained row the LRU has since dropped. The
        # LRU is checked first because it is where every hot state lives, and the
        # retained store only ever holds a rollout the recorder has not replayed yet.
        record = self._pinned.get(key)
        if record is not None:
            return key, record

        self.cache_misses += 1
        # Diagnostic. Deliberately does NOT name a cause from the distance alone: the
        # three real causes -- LRU eviction of an older trajectory, an in-place
        # mutation of an emitted array, and a synthesised state -- are not separable
        # by magnitude. Measured over 400 steps on drawer-open, two CONSECUTIVE
        # genuine states differ by as little as 1.2e-3 in L-inf (median 0.010), which
        # is the same order as Meta-World's own scripted door policy mutating an
        # observation by 0.05. Reporting "something mutated the array" for a 0.15
        # neighbour would send the reader to the wrong file. Only the sub-1e-6 case is diagnostic on its own, because the
        # cache key is float32 bytes and anything closer than that should have hit.
        hint = ""
        probe = np.asarray(obs, dtype=float).ravel()
        nearest = None
        for recent in list(self._cache)[-8:]:
            other = np.frombuffer(recent, dtype=np.float32).astype(float)
            if other.shape == probe.shape:
                delta = float(np.abs(probe - other).max())
                nearest = delta if nearest is None else min(nearest, delta)
        if nearest is not None and nearest < 1e-6:
            hint = (f" The nearest recently emitted state is {nearest:.3g} away, which is "
                    "below float32 resolution: the observation was re-derived (a cast, a "
                    "mean, a JSON round trip) rather than passed through.")
        elif nearest is not None:
            hint = (f" Nearest recently emitted state: {nearest:.3g} away -- not diagnostic "
                    "on its own, since consecutive genuine states here differ by as little "
                    "as 1.2e-3. With the cache full the usual cause is LRU eviction of an "
                    "older trajectory (see MetaWorld.snapshot_cache_size); with room to "
                    "spare it is an in-place mutation of an emitted array.")
        raise UnknownStateError(
            f"{self.name}: asked to act on an observation this adapter never emitted "
            f"(cache {len(self._cache)}/{self.snapshot_cache_size} entries, "
            f"{len(self._pinned)} retained, "
            f"{self.cache_misses} misses so far). MuJoCo is a stateful simulator and a "
            "39-D Meta-World observation cannot be inverted to (qpos, qvel), so every "
            "state passed to step/render/reference_reward must have come from "
            "reset(), a previous step() or random_state()." + hint)

    def _restore(self, key: bytes, record: Tuple[dict, dict]) -> None:
        """Put the simulator into the state `key` was emitted from.

        The ORDER is load-bearing and was established by measurement: model fields ->
        data fields -> data.time -> mj_forward -> qacc_warmstart -> python attrs.
        Restoring `qacc_warmstart` BEFORE `mj_forward` leaves a 6.7e-16 residual that
        `mj_forward` re-derives into obs[7:10] -- the object quaternion, computed
        through `Rotation.from_matrix`, which amplifies eps-level changes in
        `geom_xmat`. After, the replay is exactly 0.0.
        """
        if key == self._cur:
            return
        episode, step = record
        env = self._env
        for f, v in episode["model"].items():
            getattr(env.model, f)[:] = v
        env.model.opt.gravity[:] = episode["gravity"]
        for a, v in episode["py"].items():
            setattr(env, a, copy.deepcopy(v))
        for f, v in step["mj"].items():
            getattr(env.data, f)[:] = v
        env.data.time = step["time"]
        self._mj.mj_forward(env.model, env.data)
        env.data.qacc_warmstart[:] = step["warmstart"]
        for a, v in step["py"].items():
            setattr(env, a, copy.deepcopy(v))
        self._episode = episode
        self._cur = key

    def _record(self, key: bytes) -> Optional[Tuple[dict, dict]]:
        """The snapshot for `key` from either store, or None. No LRU bump."""
        record = self._cache.get(key)
        return record if record is not None else self._pinned.get(key)

    def _evict_pinned(self) -> None:
        while len(self._pinned) > self.pinned_cache_size:
            self._pinned.popitem(last=False)

    # -- surviving until the recorder replays them ----------------------------

    def retain_states(self, state_arrays: Any) -> None:
        """Move these states out of the LRU's reach; see `EnvAdapter.retain_states`.

        THE PROBLEM THIS SOLVES, measured on `mt10_window-open-v3` with real
        SB3/SAC. One candidate at `train.env_steps: 2000` emits 11,126
        states -- 2,108 training, 7,515 checkpoint evaluation (`_evaluate_policy`
        replays whole 500-step episodes at every checkpoint), 1,503 rollout --
        so 87% of the flood is training and evaluation, and it scales with
        `train.env_steps` while `snapshot_cache_size` does not: at 1,000,000
        steps one candidate emits ~1.05 M states, which is ~3 GB of
        snapshots and no cache size answers it. The rollouts are produced LAST, then the
        next candidate wipes them: at 2 candidates the recorder finds candidate
        0's rollout `0/501` present and candidate 1's `501/501`, i.e. exactly one
        video from two. Two more shapes in the same family:
        `evaluate.rollouts_per_candidate: 50` (LIMEN's setting) evicts
        rollout 0 with its own rollouts 1..49 inside a SINGLE candidate, and the
        parallel path imports `n_candidates x rollouts x 501` rows, which
        overflows at 8 candidates x 3.
        None of those is a cache-size problem, which is why raising the cap to
        16384 still misses.
        And no rule the adapter can apply from the inside separates a trajectory
        that will be replayed from a planner's lookahead -- so the caller, which
        knows, says so, and that is the whole mechanism.

        `components.training` retains rollout 0 of each candidate (the one
        `observability.record_rollouts` and `preferences._clip_for` both replay)
        as soon as it exists, and `record_rollouts` calls `release_states` once
        the iteration's videos are written. So the store holds one iteration's
        rollouts and not a run's.

        A state that is in neither store is skipped rather than raising: it was
        never emitted by this adapter, and `render` will say so with a far better
        message than a retention call could.
        """
        for rows in state_arrays or ():
            if rows is None:
                continue
            arr = np.asarray(rows)
            for row in (arr if arr.ndim > 1 else arr.reshape(1, -1)):
                key = self._key(row)
                record = self._record(key)
                if record is None:
                    continue
                self._pinned[key] = record
                self._pinned.move_to_end(key)
        self._evict_pinned()

    def release_states(self) -> None:
        """Drop every retention claim. The rows stay in the LRU if it still has
        them -- this gives back the protection, not the memory, and the memory
        follows on the next eviction."""
        self._pinned.clear()

    # -- crossing a process boundary -----------------------------------------

    def export_states(self, state_arrays: Any) -> Any:
        """Ship the snapshots for exactly the states a worker is handing back.

        The base-class default is `None` because toy/control invert an
        observation arithmetically; here the cache IS the inverse, it lives in
        the worker's address space, and a `fork`ed child's writes never reach
        the parent. Measured: a child produced a (31, 39) trajectory and the
        parent's cache afterwards had 0 entries, so `reference_reward` and
        `render` both raised `UnknownStateError` while `task_metric` -- a pure
        function of the rows -- still returned a number. Fitness survives a
        fork; the reference reward, the videos, and every VLM-graded method do
        not.

        Only trajectory states are carried, never the training states: a
        rollout set is ~500 rows at 3169 B/state = ~1.6 MB and ~16 ms, against
        candidate trainings measured in minutes, while the full cache is capped
        at `snapshot_cache_size` (8192) entries and would be ~27 MB. Keeping it
        narrow is also what STOPS the parent's cache being flooded by training
        states, because each worker's cache holds only its own candidate and
        the parent imports only trajectory rows.

        Without retention the sequential path loses rollouts that this narrow
        export preserves. MEASURED with Eureka on `mt10_window-open-v3`, 4
        candidates, mock LLM, one iteration, no `retain_states`:

            sequential  1 of 4 rollout videos written, and one
                        `UnknownStateError: ... (cache 8192/8192 entries) ...
                        the usual cause is LRU eviction of an older trajectory`
                        followed by `3 of 4 selected rollout(s) could not be
                        rendered`
            parallel    4 of 4 written, no UnknownStateError at all

        Every JSON artifact matched between the two; only `videos/` differed.
        `retain_states` is what closes that gap on the sequential path, so that
        `rda`'s `vlm_score` and `gt`'s `preference_bt` do not grade blind for
        all but the last candidate.

        The episode half of each record is shared by reference across every
        step of an episode (see `_episode_snapshot`), so it is deduplicated by
        `id()` here and rewired on import -- otherwise a 500-step rollout would
        carry 500 copies of the same ~4.6 KB.
        """
        keys: list = []
        seen: set = set()
        for rows in state_arrays or ():
            arr = np.asarray(rows)
            for row in (arr if arr.ndim > 1 else arr.reshape(1, -1)):
                key = self._key(row)
                if key in seen or self._record(key) is None:
                    continue
                seen.add(key)
                keys.append(key)
        episodes: Dict[int, Any] = {}
        out = []
        for key in keys:
            episode, step = self._record(key)
            slot = id(episode)
            if slot not in episodes:
                episodes[slot] = (len(episodes), episode)
            out.append((key, episodes[slot][0], step))
        # A retention claim is a worker output like the budget delta and the
        # policy blob: made in the child, meaningless unless it is sent home.
        # Without this the parent imports 8 candidates x 3 rollouts x 501 rows
        # into an 8192-entry LRU and the first three candidates' videos are gone
        # before `record_rollouts` runs (measured).
        return {"episodes": [e for _slot, e in sorted(episodes.values())], "steps": out,
                "retained": [k for k in keys if k in self._pinned]}

    def import_states(self, blob: Any) -> None:
        """Fold a worker's snapshots into this process's cache, in trajectory order.

        Insertion order matters: `_store`'s LRU evicts oldest-first, so
        importing in the order the states were produced leaves the newest
        trajectory hottest -- the same ordering the parent would have had if it
        had produced them itself.
        """
        if not blob:
            return
        episodes = list(blob.get("episodes") or ())
        steps = list(blob.get("steps") or ())
        # PIN FIRST, from the blob itself. Filling the LRU, trimming, then pinning
        # via `_record` would silently drop exactly the records the exporter
        # promised to protect whenever the blob outruns `snapshot_cache_size` (a
        # checkpoint resume carries the whole trajectory store; 8 candidates x 3
        # rollouts x 501 rows = 12,024 > 8192): the trim evicts the oldest rows
        # and `_record` then returns None for them.
        # `.get`, so a blob without this key simply retains nothing:
        # `checkpoint.py` reuses this format, so a resume from a checkpoint
        # written without it must still import rather than raise.
        records = {key: (episodes[slot], step) for key, slot, step in steps}
        for key in blob.get("retained") or ():
            record = records.get(key)
            if record is None:
                record = self._record(key)
            if record is not None:
                self._pinned[key] = record
                self._pinned.move_to_end(key)
        self._evict_pinned()
        for key, slot, step in steps:
            self._cache[key] = (episodes[slot], step)
            self._cache.move_to_end(key)
        while len(self._cache) > self.snapshot_cache_size:
            self._cache.popitem(last=False)

    # -- domain randomisation ------------------------------------------------
    #
    # `set_dr` and `_sample_dr` are NOT overridden, and that is deliberate. The base
    # class already parses `(lo, hi)` pairs, `{"low":, "high":}` mappings and bare
    # scalars, clips into `dr_parameters` and drops unknown keys -- all of which an
    # LLM-written `dr_config` blob needs (`train.domain_randomization.mode: generated`
    # produces ranges, and `env.set_dr(...)` at training.py:1000 sits OUTSIDE
    # `_run_backend`'s try/finally, so a TypeError there aborts the whole train stage
    # instead of being recorded as a candidate failure). The base also writes the
    # per-episode draw into `self._dr_now`, which is the attribute `_apply_dr` reads:
    # an override that samples into one attribute while the physics reads another is
    # DR that is silently switched off, and every DrEureka number on such an env is
    # meaningless.

    def _apply_dr(self) -> None:
        """Write the current episode's DR draw into mjModel, from nominal.

        Called from `_reset` before `env.reset()`, so the episode snapshot taken
        afterwards captures the draw and a restored state carries its own
        randomisation with it.
        """
        draw = self._dr_now or {}
        model, nominal = self._env.model, self._nominal
        model.body_mass[:] = nominal["body_mass"]
        scale = float(draw.get("object_mass_scale", 1.0))
        for i in self._dr_bodies:
            model.body_mass[i] = nominal["body_mass"][i] * scale
        model.dof_damping[:] = nominal["dof_damping"]
        # [0:7] are right_j0..right_j6, the Sawyer arm joints (nominal 10.0 each);
        # [7:9] are the gripper's r_close/l_close at 1000.0 and are left alone -- a
        # gripper that cannot hold makes every grasping task unsolvable rather than
        # harder, which is not what a DR axis is for.
        model.dof_damping[0:7] = nominal["dof_damping"][0:7] * float(
            draw.get("arm_damping_scale", 1.0))
        model.geom_friction[:] = nominal["geom_friction"]
        friction = float(draw.get("friction_scale", 1.0))
        for g in self._dr_geoms:
            model.geom_friction[g, 0] = nominal["geom_friction"][g, 0] * friction
        # `contact_friction_scale` multiplies IN PLACE after `friction_scale`, so where the
        # two overlap (pads, object) the result is nominal x friction x contact -- and it
        # cannot compound across episodes, because the array was just reset from nominal.
        contact = float(draw.get("contact_friction_scale", 1.0))
        if contact != 1.0:
            for g in self._dr_colliders:
                model.geom_friction[g, 0] *= contact
        fixture = float(draw.get("fixture_damping_scale", 1.0))
        for dof in self._fixture_dofs:
            model.dof_damping[dof] = nominal["dof_damping"][dof] * fixture
        model.opt.gravity[:] = nominal["gravity"] * float(draw.get("gravity_scale", 1.0))
        model.actuator_gainprm[:] = nominal["actuator_gainprm"]
        model.actuator_gainprm[:, 0] = nominal["actuator_gainprm"][:, 0] * float(
            draw.get("actuator_gain_scale", 1.0))
        model.eq_solref[:] = nominal["eq_solref"]
        model.eq_solref[:, 0] = nominal["eq_solref"][:, 0] * float(
            draw.get("weld_softness_scale", 1.0))

    # -- dynamics -----------------------------------------------------------

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        """Start an episode on one of the 50 pinned task instances.

        The instance is chosen with the EPISODE rng, so initial conditions vary with
        `cfg.seed` while the benchmark itself does not. `set_task` freezes the goal
        (`_freeze_rand_vec = True`), which is what makes this deterministic: two
        independently constructed adapters with the same seeds produce bitwise
        identical 60-step trajectories -- measured.

        The cache is NOT cleared here. Clearing it is what breaks the `_sb3_run`
        interleave, where `_evaluate_policy` resets this shared adapter while SB3 is
        still holding a training state it will step from on the next chunk.
        """
        self._apply_dr()
        self._env.set_task(self._tasks[int(rng.integers(len(self._tasks)))])
        obs = self._emit(np.asarray(self._env.reset()[0], dtype=float))
        self._episode = self._episode_snapshot()
        self._store(obs)
        return obs

    @property
    def n_tasks(self) -> int:
        """How many pinned task instances `MT1(task, seed=benchmark_seed)` provides."""
        return len(self._tasks)

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Restore the simulator from `s`, step it, cache and return the successor.

        `done` is ALWAYS False. Meta-World does not terminate on success -- the
        harness truncates at `horizon`, and returning `terminated=True` to SB3 on what
        is really a truncation would cut value bootstrapping and quietly bias every
        learned value function on this env.

        `info` carries ONLY `success`. `unscaled_reward` is byte-identical to the
        shaped reward the candidate is competing to replace, and `in_place_reward` /
        `grasp_reward` are its two factors -- `_Gym` forwards `info` straight to SB3,
        so anything left in here is a leak with a wide blast radius.
        """
        key, record = self._lookup(s)
        self._restore(key, record)
        env = self._env
        # The planner branches past step 500 and Meta-World then raises
        # `ValueError("You must reset the env manually once truncate==True")`, which
        # aborted the whole train stage. Clamping is correct rather than merely
        # convenient: `curr_path_length` is Meta-World's own truncation counter and
        # BIRD owns truncation (`_Gym` truncates at `env.horizon`), so the simulator's
        # copy of it must not be allowed to end an episode behind the harness's back.
        if env.curr_path_length >= env.max_path_length:
            env.curr_path_length = env.max_path_length - 1
        action = np.clip(np.asarray(a, dtype=float).ravel()[:self.action_dim], -1.0, 1.0)
        obs, _reward, _term, _trunc, info = env.step(action)
        obs = self._emit(np.asarray(obs, dtype=float))
        self._store(obs)
        return obs, False, {"success": float(info.get("success", 0.0))}

    # -- ground truth --------------------------------------------------------

    def _success_of(self, obs: np.ndarray) -> float:
        return float(self._success_check(obs, obs[4:7], obs[_GOAL]))

    def task_metric(self, traj: Any) -> float:
        """Fraction of the episode's arriving states inside the goal region.

        RECOMPUTED FROM THE OBSERVATION, not read from a cache and not read from
        `info`. That is what makes it a pure function of `traj.states`: it survives
        LRU eviction and a JSON round trip, so `phases._reference_score`'s
        `env.task_metric` probe and §4 re-scoring both work on records the simulator
        has long since forgotten.

        NO LEAK, by three arguments. (1) The check reads only hand/object/goal
        positions out of the observation plus a fixed threshold; the shaped reward
        never enters it. (2) We never monkey-patch `compute_reward`, so Meta-World's
        own `info["success"]` stays valid -- which is what lets a test assert this
        check EQUALS it rather than merely resembles it. That is the trap in the
        obvious design: `success` is computed inside `evaluate_state`, which CALLS
        `compute_reward` and reads geometric intermediates out of its return tuple, so
        an adapter that swapped the reward would destroy its own ground truth.
        (3) `_step`'s info carries only the flag, never the reward.

        Scored over rows[1:] -- the states an action arrived in -- matching
        Meta-World's own per-step accounting and `reference_reward`'s convention.
        """
        states = _states_of(traj)
        if states.shape[0] == 0 or states.shape[1] < self.obs_dim:
            return 0.0
        rows = states[1:] if states.shape[0] > 1 else states
        per_step = [self._success_of(np.asarray(r, dtype=float)) for r in rows]
        if self.reduction == ANY_STEP:
            # The benchmark's published criterion: done at least once in the episode.
            return float(max(per_step) > 0.0) if per_step else 0.0
        return float(np.mean(per_step))

    # --- BEGIN reference reward (ground truth; strip before showing an LLM) ---
    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """Meta-World's own dense shaped reward for arriving in `s` having taken `a`.

        Restores the snapshot for `s` and calls the UNTOUCHED `evaluate_state`, so
        this is not a re-implementation that can drift: measured byte-identical
        (0.0e+00) to the reward `env.step` returned, on all ten tasks x 600 steps.

        Is exposing it a leak? It flows to (i) `gt_return` -> `gt_reward_curve` -> the
        Gran Turismo human-oracle proxy, which is the intended meaning of "an expert
        scores this candidate"; (ii) the EPIC/STARC reference, which is the published
        construction; (iii) never into fitness by default; (iv) never into `info`. The
        real leak vector is the PROMPT, and that is what `reward_source` below handles.

        One honesty note: a config that runs `verify.quality_screen: epic` on this
        env is measuring "distance from Meta-World's own hand-tuned reward", which is
        a much stronger supervisory signal than the same key on Pendulum, and should
        say so rather than inheriting it silently.
        """
        key, record = self._lookup(s)
        self._restore(key, record)
        action = (np.zeros(self.action_dim) if a is None
                  else np.clip(np.asarray(a, dtype=float).ravel()[:self.action_dim], -1.0, 1.0))
        reward, _info = self._env.evaluate_state(np.asarray(s, dtype=float), action)
        return float(reward)

    def gt_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None,
                  s2: Optional[np.ndarray] = None) -> float:
        """Alias required by `screens.reference_reward`, which probes
        `gt_reward|ground_truth_reward|true_reward|reward_fn|compute_reward` and NOT
        `reference_reward` (screens.py:245). Without it EPIC, STARC and
        policy_rank_corr are inert on this env -- they log "env adapter exposes no
        ground-truth reward" and return nothing. `verification.call_reward`
        arity-adapts the three-parameter form, and the BIRD convention is that a reward
        scores the state it ARRIVES in, hence `s2 or s`.
        """
        return self.reference_reward(s if s2 is None else s2, a)
    # --- END reference reward ---

    # -- coverage and discretisation ----------------------------------------

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        """A REACHABLE state, drawn from a lazily refilled coverage pool.

        "Coverage, not the initial distribution" is the contract, but on a simulator
        it collides with reachability: a state `_step` cannot continue from is useless
        to the EPIC/STARC screens that consume this. Uniform sampling of a 39-D box
        would also produce quaternions of norm 3 and negative gripper openings, i.e.
        it would evaluate reward functions at physically impossible points.

        One 200-step random rollout yields 100 pool entries, which keeps
        `sample_transitions(128)` at a measured 0.43 s for 128 distinct in-bounds
        states rather than 128 full resets at 10.7 ms each plus the walk.

        `_reset` is called directly rather than `reset()` so the current DR draw is
        left alone -- `sample_transitions` installs NOMINAL dynamics around this call
        precisely so a screen never sees the candidate's randomisation, and
        `reset()` would re-draw from the installed ranges and undo that.
        """
        # VALIDATED ON POP. Pool entries hold observations whose snapshots live
        # only in the un-pinned LRU, and the pool outlives a training flood that
        # evicts them (one SB3 candidate emits ~11k `_store`s against 8192 slots).
        # Returning such an entry would hand the EPIC/STARC screens a state
        # `_lookup` must refuse; `verification.sample_transitions` would degrade to
        # Gaussian draws, logged only at DEBUG, and the screens would go silently
        # inert from iteration 2 on. A stale entry is discarded; the pool refills
        # when dry.
        while True:
            if not self._pool:
                obs = self._reset(rng)
                for i in range(200):
                    obs, _done, _info = self._step(
                        obs, self.action_set[int(rng.integers(self.n_actions))])
                    if i % 2 == 0:
                        self._pool.append(obs)
            obs = self._pool.pop(int(rng.integers(len(self._pool))))
            if self._record(self._key(obs)) is not None:
                return obs

    #: Six task-RELATIVE scalars and their (lo, hi, n_bins). Relative rather than
    #: absolute because the goal is re-randomised every episode: an absolute grid over
    #: workspace coordinates would put the same physical situation in different cells
    #: on different episodes, and the Q-table would learn a location rather than a
    #: task. Bounds are the workspace extents (the Sawyer reaches ~0.5 m).
    _DISC_BINS = (5, 5, 3, 4, 3, 4)   # product = n_disc_states = 3600

    def discretise(self, s: np.ndarray) -> int:
        """Mixed-radix index in [0, 3600). A binning, not a bijection -- hence
        `exact_states is None`, and hence `_make_learner` warns that
        `train.backend: tabular` is approximate here.

        3600 rows x 15 actions is 432 KB per seed. The obvious alternative -- a
        product of absolute per-axis bins -- was 10**6 x 9 = 72 MB per seed for a
        reachable set of ~97k cells, allocated eagerly in `_QLearner.__init__` once
        per seed. Measured over 400 random steps on pick-place: indices span
        [973, 2533], well inside the range and not clustered at one end.
        """
        o = np.asarray(s, dtype=float).ravel()
        hand, obj, goal = o[0:3], o[4:7], o[_GOAL]
        n = self._DISC_BINS
        features = (
            (float(np.linalg.norm(hand - obj)), 0.0, 0.5, n[0]),   # can it reach the object
            (float(np.linalg.norm(obj - goal)), 0.0, 0.6, n[1]),   # is the object there yet
            (float(o[3]), 0.0, 1.0, n[2]),                         # open / closing / closed
            (float(hand[2]), 0.0, 0.4, n[3]),                      # hand height
            (float(obj[2]), 0.0, 0.3, n[4]),                       # is the object lifted
            (float(np.linalg.norm(hand - goal)), 0.0, 0.6, n[5]),  # approach phase
        )
        index = 0
        for value, lo, hi, bins in features:
            index = index * bins + _bin(value, lo, hi, bins)
        return int(index)

    # -- rendering -----------------------------------------------------------

    def render(self, state: np.ndarray, width: int = 480) -> np.ndarray:
        """One `(480, 480, 3)` uint8 frame of the simulator restored to `state`.

        Exactly one positional parameter, which is what `observability.
        _accepts_a_state`'s `signature.bind(object())` gate requires. `width` is
        accepted and ignored: `MujocoRenderer` captured the size at construction and
        rendering cost is flat in resolution anyway, so the frame is produced at 480
        and `observability._resize_nearest` takes it down to `output.video.width`.

        The restore is the entire reason this method is not one line: rendering
        replays a STORED trajectory after training has finished and the simulator has
        moved on many episodes. Measured: ten out-of-order states from a 60-step
        trajectory produce ten distinct frames. 345 ms for the first frame (GL context
        creation), 42 ms/frame steady -- about 55x a sim step, so
        `output.video.max_frames` is a real cost knob on this tier.

        If headless GL is unavailable this RAISES, and that is a considered rejection
        of `control.py`'s numpy fallback rather than an omission. control.py rasterises
        a rod on a plain field, which is exactly reproducible; a hand-rolled Sawyer,
        drawer and puck would be plausible-looking and wrong, and `rda` /
        `gt_reward_design` do not log frames, they GRADE on them. A confidently wrong
        frame handed to a VLM grader is strictly worse than the "no video" degradation
        `observability._render_frames` already implements.
        """
        key, record = self._lookup(state)
        self._restore(key, record)
        frame = np.asarray(self._env.render(), dtype=np.uint8)
        # Rotate 180 degrees, because Meta-World's `corner` camera is defined upside
        # down: `objects/assets/xyz_base.xml:17` gives it
        # `xyaxes="-1 1 0  -0.2 -0.2 -1"`, and the angle between that up-vector and
        # the NEGATED world up projected into the image plane measures 0.000 degrees
        # -- the camera is exactly inverted, not merely tilted.
        #
        # Not ours to fix upstream and not a renderer bug: gymnasium's frame is
        # bitwise equal to `mujoco.Renderer`'s for the same camera, and that same
        # renderer puts a sphere at z=1.5 in the TOP half of the frame on a control
        # model. It still has to be corrected here, because `rda` and
        # `gt_reward_design` do not log these frames, they GRADE on them, and a VLM
        # shown an inverted scene is a silent degradation of the method.
        #
        # `[::-1, ::-1]` (a rotation) and NOT `[::-1]` (a mirror): an inverted camera
        # negates both image axes, so a plain vertical flip would leave the scene
        # left-right reversed and make "is the puck left of the goal" answerable
        # backwards -- exactly the confidently-wrong frame this method refuses to
        # produce elsewhere.
        return frame[::-1, ::-1]

    def render_view(self, state: np.ndarray, view: View) -> np.ndarray:
        """One `(480, 480, 3)` uint8 frame of `state` from an extra viewpoint.

        The same restore as `render` -- the frame is a function of the state and not
        of whatever the simulator last did -- then `envs.cameras.MujocoViews` renders
        the named model camera and corrects its orientation by measurement: every
        `corner*` camera in `xyz_base.xml` is defined with its up-vector pointing into
        the floor (measured, all four; `render`'s derivation covers
        `corner` alone) while `topview`, `behindGripper` and `gripperPOV` are upright.

        `pose` views are accepted but every MT10 spec names model cameras: the seven
        Meta-World defines cover the task from every side that matters, and a posed
        camera here would be a camera nobody measured.
        """
        key, record = self._lookup(state)
        self._restore(key, record)
        return self._views().render(view)

    def _views(self) -> MujocoViews:
        """The extra-view renderer, built on first use (a second GL context)."""
        views = getattr(self, "_extra_view_renderer", None)
        if views is None:
            views = MujocoViews(self._mj, self._env.model, self._env.data,
                                self.render_width, self.render_width)
            self._extra_view_renderer = views
        return views

    def task_images(self) -> List[bytes]:
        """PNG bytes of the reset state, for `problem.instruction_modality:
        text+image|video` (generation.py:249). `anthropic_client._image_block` takes
        raw PNG bytes. Returns [] -- which `generation._images_for` already tolerates
        -- when imageio or GL is missing, because an instruction modality that cannot
        be honoured must degrade to text rather than abort a run."""
        try:
            import imageio.v3 as iio
            frame = self.render(self.reset(0))
            return [bytes(iio.imwrite("<bytes>", frame, extension=".png"))]
        except Exception as exc:  # noqa: BLE001 - optional dep or missing GL context
            log.debug("%s: task_images unavailable (%s)", self.name, exc)
            return []

    # -- describe(): `generate.context.env_spec` -----------------------------

    @property
    def reward_source(self) -> str:
        """The exact source span `generate.context.strip_existing_reward` must cut.

        This is the AUTHORITATIVE gate that `generation._strip_reward` prefers over
        its regex (generation.py:183), and it is `evaluate_state` -- not
        `compute_reward`. `evaluate_state` is the worse of the two leaks: it contains
        the SUCCESS CHECK, i.e. the metric the method is scored on, and
        `_REWARD_DEF` would not catch it (that regex matches `def \\w*reward\\w*`,
        `def \\w*success\\w*` and the EnvAdapter metric names). `compute_reward` and
        `_gripper_caging_reward` are caught by the regex, so between the two
        mechanisms all three go.

        Located by AST rather than `inspect.getsource(cls.evaluate_state)`, which is
        not a nicety: the method carries the `@_Decorators.assert_task_is_set`
        decorator, so `getsource` on the bound function returns the body of
        `assert_task_is_set.<locals>.inner` -- a string that is not a substring of the
        class source at all, and `_strip_reward`'s `in text` test would silently fail.
        """
        if self._reward_source is None:
            self._reward_source = _span_of(self._task_cls, "evaluate_state")
        return self._reward_source

    def _render_full_source(self) -> str:
        """The ACTUAL Meta-World source, assembled by `inspect`.

        The base implementation returns `inspect.getsource(type(self))` -- this
        adapter -- which would show an LLM BIRD's plumbing instead of the environment
        it is writing a reward for. What it needs is: the task class, the shared
        Sawyer base's observation/control/contact helpers (the per-task file does not
        describe the 39-D observation at ALL), and `reward_utils.tolerance` /
        `hamacher_product`, out of which every v2 Meta-World reward is built.

        The task class is resolved with `inspect.getsourcefile(type(self._env))`,
        never a string transform of the task id: `door-open-v3` lives in
        `sawyer_door_v3.py` and `peg-insert-side-v3` in
        `sawyer_peg_insertion_side_v3.py`, so the obvious slug transform 404s on 2 of
        10 tasks -- and 8 of 10 configs would look fine.

        Measured 19.5-26.4 KB across the ten tasks, i.e. roughly 5-7k prompt tokens.
        That is what
        "full_source" costs on this tier and the configs should say so.
        """
        from metaworld import sawyer_xyz_env
        from metaworld.utils import reward_utils

        parts = [
            f"# environment `{self.name}` -- Meta-World task `{self.task}`, full source.",
            "#",
            "# NOTE: every task file contains a dead `if self.reward_function_version ==",
            "# 'v2': ... else: ...` branch. v2 is the default on every construction route",
            "# and the v1 branch below is never executed.",
            "",
            f"# --- the task environment: {self._task_cls.__name__} ---",
            _safe_source(self._task_cls),
            "# --- the shared Sawyer base: observation, control and contact helpers ---",
        ]
        for member in ("_get_obs", "_get_curr_obs_combined_no_goal", "tcp_center",
                       "set_xyz_action", "_gripper_caging_reward", "touching_main_object"):
            obj = getattr(sawyer_xyz_env.SawyerXYZEnv, member, None)
            if obj is None:
                continue
            parts.append(_safe_source(obj.fget if isinstance(obj, property) else obj))
        parts.append("# --- the shaping primitives every Meta-World reward is built from ---")
        parts.append(_safe_source(reward_utils.tolerance))
        parts.append(_safe_source(reward_utils.hamacher_product))
        return "\n\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# source-extraction helpers
# --------------------------------------------------------------------------


def _safe_source(obj: Any) -> str:
    """`inspect.getsource` that degrades to a comment instead of raising. A prompt
    that is missing one helper is worth more than a run that dies assembling one."""
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError) as exc:  # pragma: no cover - zipimport / builtins
        return f"# (source for {getattr(obj, '__name__', obj)!r} unavailable: {exc})"


def _span_of(cls: type, method: str) -> str:
    """The verbatim source span of `cls.method`, decorators included.

    Verbatim matters: `_strip_reward` does `text.replace(span, WITHHELD)`, so a span
    that differs from the class source by even one character silently does nothing.
    """
    try:
        src = inspect.getsource(cls)
        lines = src.splitlines(keepends=True)
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):  # pragma: no cover
        return ""
    for node in getattr(tree.body[0], "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == method:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            return "".join(lines[start - 1:node.end_lineno])
    return ""


# --------------------------------------------------------------------------
# registration: fifty ids, one per MT50 task, mechanically derived
# --------------------------------------------------------------------------
#
# NAMING. The id is `mt10_` + the MT10_V3 key VERBATIM for the ten MT10 tasks
# (`mt10_pick-place-v3`, not `metaworld_pick_place`) and `mt50_` + the key for the
# other forty (`mt50_hammer-v3`). The prefix is the SMALLEST published protocol the
# task belongs to, decided by `bird/tasks.py::METAWORLD_MT10`; MT10 is a subset of
# MT50 so no task gets two ids. Two reasons for the shape, in order of weight:
#
#   1. It cannot drift. The suffix is the exact string `metaworld.env_dict.MT10_V3` is
#      keyed by and the exact string `MT1(task)` takes, so an id that resolves is an
#      id the benchmark accepts. A prettified underscore form is a second spelling of
#      the same fact, and the two can disagree (which they already do on `door-open`,
#      whose source file is `sawyer_door_v3.py`).
#   2. The prefix names the BENCHMARK SUITE, which is the thing published numbers are
#      comparable against. `metaworld_` would name the package; MT10 is the protocol.
#
# The registration is a loop over the CATALOGUE, not ten hand-written decorators: ten
# copy-pasted factories are ten chances for an id to disagree with the task it builds.
# This is data-driven dispatch, not method branching -- `tests/test_no_method_branching`
# keys on METHOD names (the method config stems), and no task id is one.


def success_check_for(task_id: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """The success check for one Meta-World task, resolved the way the adapter resolves it.

    Public so the allow-list resolution can be exercised without constructing the adapter
    (it runs with numpy alone; `MetaWorld` imports mujoco).

    Going through the spec rather than exposing `_SUCCESS_CHECKS` directly is the point: the
    spec is what NAMES the check, so a caller that reaches past it could disagree
    with the adapter about which function a task runs, and nothing would say so.
    """
    spec = by_env_id(_env_id(task_id))
    if spec is None:
        raise TaskSpecError(f"no task spec for Meta-World task {task_id!r}")
    shipped = spec.discrete_success.get("shipped") or {}
    symbol = (shipped.get("reimplementation") or {}).get("symbol")
    if symbol not in _SUCCESS_CHECKS:
        raise TaskSpecError(
            f"{spec.path}: names success check {symbol!r}, which is not in this module's "
            "_SUCCESS_CHECKS allow-list")
    return _SUCCESS_CHECKS[symbol]


def _env_id(task_id: str) -> str:
    """`mt10_<key>` for the MT10 ten, `mt50_<key>` for the other forty -- the rule is
    `bird/tasks.py`'s, restated through the same tuple so the two cannot disagree."""
    return ("mt10_" if task_id in _EXPECTED_MT10 else "mt50_") + task_id


def _factory(task_id: str) -> Callable[[Any], SpecEnvAdapter]:
    def make(ctx: Any) -> SpecEnvAdapter:
        cfg = getattr(ctx, "cfg", None)
        reduction = PER_STEP if cfg is None else (
            cfg.get("evaluate.fitness.reduction") or PER_STEP)
        return MetaWorld(task_id, reduction=reduction)

    make.__name__ = _env_id(task_id).replace("-", "_")
    # Advertised on the FACTORY, not just the class: `_check_coherence` must be able to
    # ask "does this env honour the sparse reduction?" during `--validate-all`, and
    # constructing a MetaWorld to find out would import mujoco and take 1.2 s per config.
    make.supported_reductions = MetaWorld.supported_reductions
    # The first sentence of the task's own prose, rather than a `noun` field carried in
    # the spec purely to build this string -- a schema field for a docstring would be a
    # declared value the algorithm never honours, i.e. a fabricated pin.
    tier = "MT10" if task_id in _EXPECTED_MT10 else "MT50"
    make.__doc__ = (f"Meta-World {tier} `{task_id}`. "
                    + str(_METAWORLD[task_id].description["env_prose"]).split(". ")[0].strip()
                    + ". Benchmark-supplied success check (§0).")
    return make


_METAWORLD = _metaworld_specs()

for _task_id in _METAWORLD:
    register("env", _env_id(_task_id))(_factory(_task_id))
del _task_id
