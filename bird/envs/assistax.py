"""Assistax -- assistive-robotics tasks (a Panda arm helping a seated or bedridden
human), driven through plain `mujoco` on CPU.

Env ids `assistax_<task>`: one per robot-human task upstream ships
(`assistive-autonomy/assistax`, Hinckeldey et al., RLC 2026; JAX + Brax + MuJoCo-MJX):

    scratchitch      scratch a randomly placed itch on the human's right arm
    bedbathing       wipe 52 points spread over the human's right arm
    armmanipulation  hook the human's weak right arm and lift it to waist level
    feeding          bring a spoon to the human's mouth, level and gently
    teethbrushing    brush the human's teeth: bristles to the mouth, in contact, moving

(Upstream's two robot-robot tasks, `pushcoop` and `handover`, have no human and are not
ported.)

The scene files and meshes of the five are the SAME files upstream loads --
`bird/envs/assets/assistax/`, vendored from upstream commit
`PROVENANCE.json:upstream.commit` by `scripts/vendor_assistax_assets.py` -- and each id is
a ROW in `_TASKS`: which scene, where the tool is, what the per-episode draw is, what
counts as doing the task, and what upstream's own reward paid for. Nothing branches on
the task key: every per-task difference is a value or a callable in the row (`sites`,
`camera`, `metric`, `place_markers` included), and `_construct_model` resolves the names a
row declares -- never a reward or a check.

WHAT IS FAITHFUL AND WHAT IS NOT, stated up front because the differences are the kind that
look like the original from the outside.

  * Physics: the vendored MJCF at its own `timestep=0.002` (armmanipulation: 0.001, the
    value upstream overrides in code), 4 substeps per control step, the XML's default
    solver (Newton, 100 iterations). Upstream runs the same model under MJX with
    `opt.iterations=1`, `opt.ls_iterations=4` (1 on armmanipulation) and EULERDAMP
    disabled -- a deliberately loose solve chosen for GPU throughput. Same model, different solve; the two
    trajectories part within a few control steps on the contact-rich human, and no
    number from the paper is comparable to one from here.
  * Actuation: the Panda's seven `general` actuators are stiff position servos in
    normalised units -- force = gain*ctrl - kp*q - kd*qd with gain/kp ~ pi, so a ctrl of
    u drives the joint towards pi*u radians -- with `ctrlrange` clipped to the joint
    limits. `step` hands the 7-D action in [-1, 1] to those actuators as MuJoCo does
    upstream (ctrl clipping included). NOT torques, whatever the paper says.
  * The human is PASSIVE. Upstream trains the human as a second agent (three right-arm
    actuators on the arm tasks, two on the head tasks) and ships a zoo of JAX partner
    policies; this port commands every humanoid actuator with zero, so the body rests
    under gravity, its joint stiffness/damping and the chair or bed, and the robot can
    push it. That is upstream with a zero-action partner, not a different model: the
    uncommanded humanoid actuators are zero upstream too. On `armmanipulation` the human
    arm actuators have gear 0 in the XML (the weak arm), so there the two agree exactly.
  * Observation: `concat(qpos, qvel, task slots, derived fields)`, NOT upstream's per-agent
    vectors. The contract here is `step(state, action)` with no hidden state, so the
    observation must be the whole simulator state; the per-episode target (the itch, the
    52 wiped flags) rides in it as the task slots. The derived fields -- tool pose,
    tool velocity, the contact force the tool applies to the human, distances to the
    targets -- are functions of that state, appended so a reward can read them without a
    kinematic model. Upstream's own observation slices carry an indexing defect this port
    does not reproduce (`_UPSTREAM_OBS_NOTE`).
  * Termination: NONE. Upstream ends `bedbathing` the step every point is wiped; here the
    episode runs its full 1000 steps and a finished wipe leaves the robot idle. The
    reasoning is `mujoco_control.py`'s: a terminal under SAC's entropy bonus is a reward
    term the candidate never wrote (here a penalty on finishing early). `done` is the
    non-finite guard every adapter carries.
  * `task_metric`: a per-step CHECK this repo wrote, reduced to the fraction of the
    episode's arriving states on which it holds (`bedbathing`: the wiped fraction at the
    final state -- a count, not a per-step check). Upstream ships no success criterion
    for any task: it reports the return of its own reward. Each check is stated in its
    row and in the spec's `continuous_success.raw.expr`; each is a threshold on
    quantities the reference reward is also shaped on, so `success_threshold` is
    PROVISIONAL and the metric is not independent of the reference reward the way a
    benchmark's success flag would be.
  * `reference_reward`: upstream's reward transcribed term by term with four stated
    substitutions: the tool speed is the simulator's instantaneous site velocity rather
    than the finite difference of two consecutive frames; contact forces are summed over
    the actual tool-human contact pairs rather than read off fixed MJX contact-array
    indices (`_UPSTREAM_CONTACT_NOTE`); the control cost squares the 7-D robot action
    (clipped to [-1, 1], as the actuators see it) where upstream squares the concatenated
    robot+human action vector -- identical while the human's part is zero and the action
    is in range; and `bedbathing`'s per-point bonus is dropped because it needs the
    previous state (it is exactly what `task_metric` counts -- see `_ref_bedbathing` for
    what that leaves). It is called, never shown: `_CONSUMER_FORBIDDEN` bans it from
    candidates and `_render_full_source` withholds its span.

PURITY. `_step` restores `qpos`/`qvel` from the row, zeroes `qacc_warmstart` (the line
`humanoid.py` measured: 115/300 shuffled-replay mismatches at max |dev| 0.48 without it),
runs one `mj_forward` so the constraint state describes the restored pose and not the
previous caller's, and only then substeps. Measure with shuffled replay, never round-trips
-- `tests/test_assistax.py` does.

RENDERING. Each scene's own `default` camera: `targetbodycom` on the wheelchair scenes
(it follows the human, who barely moves), fixed on the bed scenes. On `scratchitch` the
itch marker geom (`target-u` / `target-l`, non-colliding) is moved to the drawn itch
before each frame, as upstream's `get_sys_for_render` does. The renderer is created
lazily and per-process: `train.candidate_parallelism: parallel` forks after
construction, and a GL context is not something a child may inherit.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..registry import register
from ..tasks import TaskSpec, TaskSpecError, index as task_index
from .base import _bin, _states_of
from .hud import draw_panel
from .metaworld import _span_of
from .mujoco_control import _default_mujoco_gl, _preload_llvm_before_mujoco
from .spec import PER_STEP, SpecEnvAdapter

log = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent / "assets" / "assistax"

#: The consumer anti-leak gate, advertised on every factory and inherited by
#: `config._inherit_from_task_spec` when the spec supplies nothing.
_CONSUMER_FORBIDDEN: Tuple[str, ...] = ("task_metric", "success", "reference_reward",
                                        "_model", "_data", "_forward", "_Derived",
                                        "_derived_now", "_TASKS", "_FAMILIES",
                                        # the `reference_reward` alias, and the one row metric
                                        "gt_reward", "_metric_bedbathing",
                                        )

#: Upstream's observation slices are indexed as if `qpos` were indexed by joint id. The
#: humanoid root is a FREE joint occupying `qpos[0:7]`, so joint j (j >= 1) lives at
#: `qpos[j + 6]`. On scratchitch that makes `qpos[panda_joint_id_start:panda_joint_id_end]`
#: = `qpos[18:24]` the human's shoulder and elbow angles (joints 12-17), not the Panda's
#: (qpos 24-31), and `qpos[1:18]` ("human_joint_angles") six root-pose components and
#: eleven trunk and leg joints. The exact slices differ per scene -- on feeding and
#: teethbrushing the "robot" slice is six human arm angles plus Panda joint1 (each spec's
#: `upstream_divergence` has its own numbers) -- but the defect is the same one. Recorded
#: here because a reader comparing this port's state layout to upstream's will find them
#: disagreeing and should know which one is the model's.
_UPSTREAM_OBS_NOTE = ("upstream reads joint angles at qpos[joint_id], but the root free "
                      "joint shifts every hinge by six: its 'robo_joint_angles' are the "
                      "human's arm joints. This port reads the model's own addresses.")
#: Upstream's `_get_force_on_tool` indexes MJX's static contact array by hard-coded ids
#: (273/274 on scratchitch; 58/59 and 62/63 on bedbathing; 56-62 on armmanipulation;
#: 186/187 on feeding; 19/20 -- with a `TODO: update these` -- on teethbrushing). Which
#: geom pair an index denotes is a property of the MJX build, not of the task, so this
#: port sums `mj_contactForce` over every live contact between the row's tool geoms and
#: the row's human geoms instead. The rows are this repo's reading of what those ids
#: were meant to select, not a proof of it: on the three arm tasks they include the right
#: HAND beside the two arm segments, which upstream's slot names never mention.
_UPSTREAM_CONTACT_NOTE = ("upstream reads contact forces at fixed MJX contact-array "
                          "indices; this port sums the live tool-human contacts by geom pair.")

# --------------------------------------------------------------------------
# constants shared by the rows (upstream constructor defaults, in SI)
# --------------------------------------------------------------------------

SUBSTEPS = 4
RESET_NOISE = 5e-3           # upstream `reset_noise_scale`: U(-5e-3, 5e-3) on every qpos and qvel
ACTION_DIM = 7
N_WIPE_POINTS = 52           # bedbathing `n_targets`
WIPE_THRESHOLD = 0.1         # bedbathing `target_threshold`, metres
HORIZON = 1000               # upstream `episode_length`
#: What `dist_tool_nearest_unwiped` reads once every point is wiped: a finite sentinel
#: (the distances are +inf internally), so the observation stays finite and the
#: non-finite guard does not end a FINISHED episode.
WIPED_OUT_DIST = 10.0

#: Render size: a 448x320 frame the harness already resizes.
RENDER_W, RENDER_H = 448, 320
_HUD_SCALE = 2
_HUD_RGB = (235, 235, 235)

# --------------------------------------------------------------------------
# small math
# --------------------------------------------------------------------------

def _quat_rotate(q: np.ndarray, v: np.ndarray, xp: Any = np) -> np.ndarray:
    """Rotate `v` by the unit quaternion `q = (w, x, y, z)` -- brax `math.rotate`.

    CALLED BY A SECOND TIER, so it must stay traceable. A jax tier calls this
    with `xp=jax.numpy` to build the
    orientation fields its tail reads -- feeding's `spoon_up_dot_world_up`
    among them -- rather than re-deriving the rotation from `xmat`, because a
    second copy of one formula is what makes a parity gap ambiguous between
    physics and transcription. With bare `np.array` on traced scalars this
    would raise TracerArrayConversionError, so the array API is a parameter and
    `jnp` is never imported here (numpy + pyyaml only, hard-deps rule).
    """
    w, x, y, z = q
    u = xp.stack([x, y, z]) if hasattr(xp, "stack") else xp.array([x, y, z])
    s = w
    return 2.0 * xp.dot(u, v) * u + (s * s - xp.dot(u, u)) * v + 2.0 * s * xp.cross(u, v)


def _unit(v: Any, eps: float = 1e-6, xp: Any = np) -> Any:
    """`v / (|v| + eps)`, for a numpy OR a jnp-backed derive.

    `xp` is the same pattern `_quat_rotate` carries: ONE
    definition both tiers call, rather than a jnp copy of the formula in the
    upstream (MJX) adapter. The `eps` matters and is not cosmetic -- it is what keeps a
    zero vector from producing NaN, and under `jax.jit` a NaN propagates
    silently into the observation tail instead of raising.
    """
    return v / (xp.linalg.norm(v) + eps)


def _boltzmann(x: float, target: float) -> float:
    """`(x/t) * exp(-x/t)`: upstream's speed and force shaping, peaking at 1/e when x = t."""
    r = x / target
    return float(r * math.exp(-r))


# --------------------------------------------------------------------------
# what the checks and the reference rewards read, computed once from one state
# --------------------------------------------------------------------------

class _Derived:
    """Every quantity the check, the reference reward and the observation tail read,
    computed from the simulator restored to one `(qpos, qvel)`.

    Built by `_step` after the substeps (the simulator is already in the arriving state,
    so `forward=False`, and the contact forces are computed live under the step's own
    action) and by `reference_reward`/`render`/`task_metric` from a stored row
    (`forward=True`, which restores `(qpos, qvel)`, runs `mj_forward` for the kinematic
    fields, and reads the contact forces back off the row -- see the note inside on why a
    restored solve cannot reproduce them).
    """

    #: THE ARRAY MODULE THE TAIL LAMBDAS CALL THROUGH. numpy here; a jax tier
    #: (`bird/envs/upstream_assistax.py`) builds an equivalent `_Derived` with
    #: `xp = jax.numpy` and evaluates THESE SAME LAMBDAS on it, so both tiers
    #: share one formula set and one layout and can differ only in the
    #: simulator quantities the formulas read -- which is what a cross-tier parity
    #: measurement is for. Two lambdas below need it: Python's `min` and
    #: `float` demand a concrete value and refuse under `jax.jit`. `jnp` cannot
    #: be imported here (this module is numpy + pyyaml only, by the hard-deps
    #: rule), so the module stays jax-free and the caller supplies the array
    #: API instead.
    xp = np

    def __init__(self, env: "Assistax", s: np.ndarray, forward: bool = True) -> None:
        s = np.asarray(s, dtype=float).ravel()
        nq, nv = env.n_q, env.n_v
        self.qpos, self.qvel = s[:nq], s[nq:nq + nv]
        self.slots = s[nq + nv:nq + nv + env.n_slots]
        if forward:
            env._forward(self.qpos, self.qvel)
        m, d, mj = env._model, env._data, env._mj
        row = env._row
        # -- the tool --
        self.tool_pos = d.site_xpos[env._tool_site].copy()
        self.tool_quat = d.xquat[env._tool_body].copy()
        vel6 = np.zeros(6)
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_SITE, env._tool_site, vel6, 0)
        self.tool_vel = vel6[3:6].copy()          # [angular, linear]: linear, world frame
        self.tool_speed = float(np.linalg.norm(self.tool_vel))
        # -- contact force the tool applies to the human --
        # THE FORCE IS STATE, NOT A FUNCTION OF THE POSE. A contact force comes out of the
        # constraint solve, and that solve reads the actuator forces: the same (qpos, qvel)
        # pressed by a different `d.ctrl` gives a different force (measured on a
        # stored armmanipulation row: 210 N recomputed as 51 N or 0 N under another
        # caller's leftover action). So the force the tail carries is the one the step
        # that produced the row computed -- under that step's own action -- and a
        # re-derivation from a stored row reads it BACK OFF THE ROW rather than solving
        # again. Only the live path (`_step`, forward=False, ctrl just written) recomputes.
        # `_forward` restores qpos/qvel only; `d.ctrl` is not in the observation.
        if forward and s.shape[0] >= env.obs_dim:
            ti = env._tail_index()
            self.tool_force = s[ti["tool_fx"]:ti["tool_fx"] + 3].copy()
            self.tool_force_mag = float(np.linalg.norm(self.tool_force))
            self.tip_force = None                    # the vector is not in the tail
            # 0.0 where the tail has no tip field: those rows have no tip geoms either
            # (`_construct_model` holds that), so the live path's tip vector is zero too
            self.tip_force_mag = float(s[ti["tip_force_mag"]]) if "tip_force_mag" in ti else 0.0
        else:
            self.tool_force, self.tip_force = env._contact_forces(d)
            self.tool_force_mag = float(np.linalg.norm(self.tool_force))
            self.tip_force_mag = float(np.linalg.norm(self.tip_force))
        # -- the human --
        self.uarm_pos = d.xpos[env._uarm_body].copy()
        self.larm_pos = d.xpos[env._larm_body].copy()
        # -- per-task --
        row["derive"](env, self, d)


# -- per-task derived fields --------------------------------------------------

def _derive_scratchitch(env: "Assistax", D: _Derived, d: Any) -> None:
    arm = int(D.slots[0] > 0.5)                      # 0 upper, 1 lower
    geom = env._larm_geom if arm else env._uarm_geom
    local = D.slots[1:4]
    D.target_pos = d.geom_xpos[geom] + d.geom_xmat[geom].reshape(3, 3) @ local
    D.dist = float(np.linalg.norm(D.target_pos - D.tool_pos))


def _wipe_points_world(env: "Assistax", d: Any) -> np.ndarray:
    """The 52 points in the world: 26 on each arm capsule, mapped through the GEOM frame.

    Upstream maps its cylinder points through the arm BODY's frame (`xmat[body]`,
    `xpos[body]`), whose z axis is not the capsule's axis -- the capsule runs diagonally in
    the body frame (`fromto="0 0 0 .16 -.16 -.16"`) -- so its points lie on a cylinder that
    intersects the arm rather than on its surface. The intent the README states (line 164:
    points "distributed along the surface of the human's arm") is what this port
    implements: the same 26 points per arm, in the geom frame, whose z axis IS the
    capsule axis. Recorded in the spec's `upstream_divergence`.
    """
    out = []
    for geom, pts in ((env._uarm_geom, env._wipe_local_u), (env._larm_geom, env._wipe_local_l)):
        R = d.geom_xmat[geom].reshape(3, 3)
        out.append(pts @ R.T + d.geom_xpos[geom])
    return np.vstack(out)


def _derive_bedbathing(env: "Assistax", D: _Derived, d: Any) -> None:
    pts = _wipe_points_world(env, d)
    unwiped = D.slots[:N_WIPE_POINTS] > 0.5
    dists = np.linalg.norm(pts - D.tool_pos, axis=1)
    D.wipe_points = pts
    D.wipe_dists = dists
    D.n_wiped = int(N_WIPE_POINTS - unwiped.sum())
    if unwiped.any():
        masked = np.where(unwiped, dists, np.inf)
        i = int(np.argmin(masked))
        D.nearest_pos = pts[i]
        D.dist = float(masked[i])
    else:
        D.nearest_pos = D.tool_pos.copy()
        D.dist = float("inf")


def _derive_armmanipulation(env: "Assistax", D: _Derived, d: Any) -> None:
    D.hook_target_pos = d.site_xpos[env._hook_target_site].copy()
    D.waist_target_pos = d.site_xpos[env._arm_target_site].copy()
    D.dist = float(np.linalg.norm(D.hook_target_pos - D.tool_pos))
    D.dist_forearm_waist = float(np.linalg.norm(D.waist_target_pos - D.hook_target_pos))
    # upstream `tool_target_dist_angular`: elementwise difference of the two site rotation
    # matrices, and `r_rot` its Frobenius norm
    diff = d.site_xmat[env._hook_target_site] - d.site_xmat[env._tool_site]
    D.rot_err = float(np.sqrt(np.sum(diff ** 2)))


def _derive_feeding(env: "Assistax", D: _Derived, d: Any) -> None:
    D.mouth_pos = d.site_xpos[env._mouth_site].copy()
    D.dist = float(np.linalg.norm(D.mouth_pos - D.tool_pos))
    # upstream: the spoon's "up" is its local -x, compared with world up
    D.spoon_up = _quat_rotate(D.tool_quat, np.array([-1.0, 0.0, 0.0]))
    D.spoon_up_dot_world_up = float(D.spoon_up[2])
    D.spoon_north = _quat_rotate(D.tool_quat, np.array([0.0, 0.0, 1.0]))


def _derive_teethbrushing(env: "Assistax", D: _Derived, d: Any) -> None:
    D.mouth_pos = d.site_xpos[env._mouth_site].copy()
    D.dist = float(np.linalg.norm(D.mouth_pos - D.tool_pos))
    to_mouth = _unit(D.mouth_pos - D.tool_pos)
    D.bristle = _quat_rotate(D.tool_quat, np.array([-1.0, 0.0, 0.0]))
    D.tilt_axis = _quat_rotate(D.tool_quat, np.array([0.0, -1.0, 0.0]))
    D.bristle_dot_to_mouth = float(np.dot(to_mouth, D.bristle))
    D.tilt_dot_to_mouth = float(np.dot(to_mouth, D.tilt_axis))
    normal = _unit(D.tool_pos - D.mouth_pos)
    v_tan = D.tool_vel - np.dot(D.tool_vel, normal) * normal
    D.tangential_speed = float(np.linalg.norm(v_tan))


# -- the observation tail: (name, doc, extractor, low, high) per task -----------------

#: IMPORTED BY A SECOND TIER. `bird/envs/upstream_assistax.py` evaluates these
#: same lambdas on a jnp-backed `_Derived` (via `D.xp`), so an edit to one
#: of them moves BOTH the CPU and the upstream MJX tier -- which is the point (one
#: formula set, so a parity gap can only be physics) and is also the trap:
#: a change made for the CPU tier silently changes what the GPU tier
#: measures. Keep every lambda traceable: no Python `min`/`float`/`if` on
#: a value, use `D.xp`.
_COMMON_TAIL = (
    ("tool_x", "tool reference point position x, metres, world frame", lambda D: D.tool_pos[0], -3.0, 3.0),
    ("tool_y", "tool reference point position y, metres, world frame", lambda D: D.tool_pos[1], -3.0, 3.0),
    ("tool_z", "tool reference point position z (height), metres, world frame", lambda D: D.tool_pos[2], -1.0, 3.0),
    ("tool_qw", "tool body orientation quaternion component w (w, x, y, z; unit norm)", lambda D: D.tool_quat[0], -1.0, 1.0),
    ("tool_qx", "tool body orientation quaternion component x", lambda D: D.tool_quat[1], -1.0, 1.0),
    ("tool_qy", "tool body orientation quaternion component y", lambda D: D.tool_quat[2], -1.0, 1.0),
    ("tool_qz", "tool body orientation quaternion component z", lambda D: D.tool_quat[3], -1.0, 1.0),
    ("tool_vx", "tool reference point linear velocity x, m/s, world frame", lambda D: D.tool_vel[0], -10.0, 10.0),
    ("tool_vy", "tool reference point linear velocity y, m/s, world frame", lambda D: D.tool_vel[1], -10.0, 10.0),
    ("tool_vz", "tool reference point linear velocity z, m/s, world frame", lambda D: D.tool_vel[2], -10.0, 10.0),
    ("tool_fx", "net contact force the row's tool geoms apply to the row's human geoms (the right arm on the arm tasks, the head on the head tasks), x, newtons, world frame; the range is typical, a stiff-contact transient can exceed it", lambda D: D.tool_force[0], -2000.0, 2000.0),
    ("tool_fy", "net contact force the row's tool geoms apply to the row's human geoms (the right arm on the arm tasks, the head on the head tasks), y, newtons, world frame; the range is typical, a stiff-contact transient can exceed it", lambda D: D.tool_force[1], -2000.0, 2000.0),
    ("tool_fz", "net contact force the row's tool geoms apply to the row's human geoms (the right arm on the arm tasks, the head on the head tasks), z, newtons, world frame; the range is typical, a stiff-contact transient can exceed it", lambda D: D.tool_force[2], -2000.0, 2000.0),
)

_ARM_TAIL = (
    ("uarm_x", "human right upper arm body position x, metres, world frame", lambda D: D.uarm_pos[0], -3.0, 3.0),
    ("uarm_y", "human right upper arm body position y, metres, world frame", lambda D: D.uarm_pos[1], -3.0, 3.0),
    ("uarm_z", "human right upper arm body position z, metres, world frame", lambda D: D.uarm_pos[2], -1.0, 3.0),
    ("larm_x", "human right lower arm body position x, metres, world frame", lambda D: D.larm_pos[0], -3.0, 3.0),
    ("larm_y", "human right lower arm body position y, metres, world frame", lambda D: D.larm_pos[1], -3.0, 3.0),
    ("larm_z", "human right lower arm body position z, metres, world frame", lambda D: D.larm_pos[2], -1.0, 3.0),
)

#: IMPORTED BY A SECOND TIER. `bird/envs/upstream_assistax.py` evaluates these
#: same lambdas on a jnp-backed `_Derived` (via `D.xp`), so an edit to one
#: of them moves BOTH the CPU and the upstream MJX tier -- which is the point (one
#: formula set, so a parity gap can only be physics) and is also the trap:
#: a change made for the CPU tier silently changes what the GPU tier
#: measures. Keep every lambda traceable: no Python `min`/`float`/`if` on
#: a value, use `D.xp`.
_TAIL_SCRATCHITCH = _COMMON_TAIL + (
    ("target_x", "the itch target position x, metres, world frame (on the arm's surface)", lambda D: D.target_pos[0], -3.0, 3.0),
    ("target_y", "the itch target position y, metres, world frame", lambda D: D.target_pos[1], -3.0, 3.0),
    ("target_z", "the itch target position z, metres, world frame", lambda D: D.target_pos[2], -1.0, 3.0),
    ("dist_tool_target", "distance from the scratcher tip to the itch target, metres", lambda D: D.dist, 0.0, 5.0),
) + _ARM_TAIL

#: IMPORTED BY A SECOND TIER. `bird/envs/upstream_assistax.py` evaluates these
#: same lambdas on a jnp-backed `_Derived` (via `D.xp`), so an edit to one
#: of them moves BOTH the CPU and the upstream MJX tier -- which is the point (one
#: formula set, so a parity gap can only be physics) and is also the trap:
#: a change made for the CPU tier silently changes what the GPU tier
#: measures. Keep every lambda traceable: no Python `min`/`float`/`if` on
#: a value, use `D.xp`.
_TAIL_BEDBATHING = _COMMON_TAIL + (
    ("nearest_unwiped_x", "position x of the nearest still-unwiped point, metres, world frame (the tool's own position when none remain)", lambda D: D.nearest_pos[0], -3.0, 3.0),
    ("nearest_unwiped_y", "position y of the nearest still-unwiped point, metres, world frame", lambda D: D.nearest_pos[1], -3.0, 3.0),
    ("nearest_unwiped_z", "position z of the nearest still-unwiped point, metres, world frame", lambda D: D.nearest_pos[2], -1.0, 3.0),
    ("dist_tool_nearest_unwiped", "distance from the wiper centre to the nearest still-unwiped point, metres (10.0, a sentinel, once every point is wiped)", lambda D: D.xp.minimum(D.dist, WIPED_OUT_DIST), 0.0, WIPED_OUT_DIST),
    ("n_wiped", "how many of the 52 points have been wiped so far", lambda D: D.xp.asarray(D.n_wiped, dtype=float), 0.0, float(N_WIPE_POINTS)),
) + _ARM_TAIL

#: IMPORTED BY A SECOND TIER. `bird/envs/upstream_assistax.py` evaluates these
#: same lambdas on a jnp-backed `_Derived` (via `D.xp`), so an edit to one
#: of them moves BOTH the CPU and the upstream MJX tier -- which is the point (one
#: formula set, so a parity gap can only be physics) and is also the trap:
#: a change made for the CPU tier silently changes what the GPU tier
#: measures. Keep every lambda traceable: no Python `min`/`float`/`if` on
#: a value, use `D.xp`.
_TAIL_ARMMANIPULATION = _COMMON_TAIL + (
    ("hook_target_x", "position x of the point on the human forearm the hook should engage, metres, world frame", lambda D: D.hook_target_pos[0], -3.0, 3.0),
    ("hook_target_y", "position y of the forearm hook point, metres, world frame", lambda D: D.hook_target_pos[1], -3.0, 3.0),
    ("hook_target_z", "position z of the forearm hook point, metres, world frame", lambda D: D.hook_target_pos[2], -1.0, 3.0),
    ("waist_target_x", "position x of the waist-level target the forearm should be brought to, metres, world frame", lambda D: D.waist_target_pos[0], -3.0, 3.0),
    ("waist_target_y", "position y of the waist-level target, metres, world frame", lambda D: D.waist_target_pos[1], -3.0, 3.0),
    ("waist_target_z", "position z of the waist-level target, metres, world frame", lambda D: D.waist_target_pos[2], -1.0, 3.0),
    ("dist_tool_hook_target", "distance from the hook's platform centre to the forearm hook point, metres", lambda D: D.dist, 0.0, 5.0),
    ("dist_forearm_waist_target", "distance from the forearm hook point to the waist-level target, metres", lambda D: D.dist_forearm_waist, 0.0, 5.0),
    ("hook_rot_err", "Frobenius norm of the difference between the hook site's and the forearm hook point's rotation matrices (0 when aligned, at most 2*sqrt(2) = 2.83; the README's 5.66 doubles it)", lambda D: D.rot_err, 0.0, 3.0),
) + _ARM_TAIL

#: IMPORTED BY A SECOND TIER. `bird/envs/upstream_assistax.py` evaluates these
#: same lambdas on a jnp-backed `_Derived` (via `D.xp`), so an edit to one
#: of them moves BOTH the CPU and the upstream MJX tier -- which is the point (one
#: formula set, so a parity gap can only be physics) and is also the trap:
#: a change made for the CPU tier silently changes what the GPU tier
#: measures. Keep every lambda traceable: no Python `min`/`float`/`if` on
#: a value, use `D.xp`.
_TAIL_FEEDING = _COMMON_TAIL + (
    ("mouth_x", "the human's mouth position x, metres, world frame", lambda D: D.mouth_pos[0], -3.0, 3.0),
    ("mouth_y", "the human's mouth position y, metres, world frame", lambda D: D.mouth_pos[1], -3.0, 3.0),
    ("mouth_z", "the human's mouth position z, metres, world frame", lambda D: D.mouth_pos[2], -1.0, 3.0),
    ("dist_tool_mouth", "distance from the spoon centre to the mouth, metres", lambda D: D.dist, 0.0, 5.0),
    ("spoon_up_dot_world_up", "cosine between the spoon's own up axis and world up: 1 level, 0 on its side, -1 upside down", lambda D: D.spoon_up_dot_world_up, -1.0, 1.0),
    ("tip_force_mag", "magnitude of the contact force between the spoon's contact face and the head, newtons (range typical, not a bound)", lambda D: D.tip_force_mag, 0.0, 2000.0),
)

#: IMPORTED BY A SECOND TIER. `bird/envs/upstream_assistax.py` evaluates these
#: same lambdas on a jnp-backed `_Derived` (via `D.xp`), so an edit to one
#: of them moves BOTH the CPU and the upstream MJX tier -- which is the point (one
#: formula set, so a parity gap can only be physics) and is also the trap:
#: a change made for the CPU tier silently changes what the GPU tier
#: measures. Keep every lambda traceable: no Python `min`/`float`/`if` on
#: a value, use `D.xp`.
_TAIL_TEETHBRUSHING = _COMMON_TAIL + (
    ("mouth_x", "the human's mouth position x, metres, world frame", lambda D: D.mouth_pos[0], -3.0, 3.0),
    ("mouth_y", "the human's mouth position y, metres, world frame", lambda D: D.mouth_pos[1], -3.0, 3.0),
    ("mouth_z", "the human's mouth position z, metres, world frame", lambda D: D.mouth_pos[2], -1.0, 3.0),
    ("dist_tool_mouth", "distance from the brush-head plate centre (the tool site) to the mouth, metres", lambda D: D.dist, 0.0, 5.0),
    ("bristle_dot_to_mouth", "cosine between the bristle direction and the direction from the brush head to the mouth: 1 when the bristles point at the mouth", lambda D: D.bristle_dot_to_mouth, -1.0, 1.0),
    ("tip_force_mag", "magnitude of the contact force between the brush's bristle face and the head, newtons (range typical, not a bound)", lambda D: D.tip_force_mag, 0.0, 2000.0),
    ("brush_tangential_speed", "speed of the brush head across the mouth's surface (the velocity component perpendicular to the head-to-mouth direction), m/s", lambda D: D.tangential_speed, 0.0, 10.0),
)


# -- task slots: (name, doc, low, high) ---------------------------------------------

_SLOTS_SCRATCHITCH = (
    ("target_arm", "which arm segment carries the itch: 0 = upper arm, 1 = lower arm; fixed for the episode", 0.0, 1.0),
    ("target_local_x", "the itch position in the target arm segment's own frame, x, metres (on the capsule surface); fixed for the episode", -0.3, 0.3),
    ("target_local_y", "the itch position in the target arm segment's own frame, y, metres", -0.3, 0.3),
    ("target_local_z", "the itch position in the target arm segment's own frame, z (along the segment), metres", -0.3, 0.3),
)
_SLOTS_BEDBATHING = tuple(
    (f"unwiped_{i}", f"wipe point {i} ({'upper' if i < 26 else 'lower'} arm): 1 = still to wipe, 0 = wiped; a point stays wiped", 0.0, 1.0)
    for i in range(N_WIPE_POINTS))


# --------------------------------------------------------------------------
# per-task success checks (state -> bool), reference rewards, resets
# --------------------------------------------------------------------------

def _all3(D: _Derived, a: Any, b: Any, c: Any) -> Any:
    """`a and b and c` for a D that may be numpy-backed or jnp-backed.

    `bool(... and ...)` CANNOT TRACE. Python's `and` short-circuits, which
    needs the truth value of a traced array, and `bool()` needs it outright --
    both raise TracerBoolConversionError under `jax.jit`. Written that way,
    all five checks fail to trace (measured with every stub attribute a slice
    of the traced argument, so nothing could pass by being a concrete constant).

    Same idiom the observation tail already uses (`D.xp`), applied to the
    sibling functions, so ONE definition serves both tiers and the jax tier
    does not carry a second copy of five thresholds.

    RETURNS AN ARRAY, NOT A bool. Every CPU caller already wraps it --
    `bool(self._row["check"](self, D))` at the `info["success"]` site and
    `int(bool(...))` in `task_metric` -- so the numpy path gives the same
    value; that is verified as CPU bit-identity on all five tasks, not argued.
    """
    return D.xp.logical_and(D.xp.logical_and(a, b), c)


def _chk_scratchitch(env: "Assistax", D: _Derived) -> Any:
    return _all3(D, D.dist < 0.05, D.tool_force_mag >= 0.5, D.tool_speed >= 0.02)


def _chk_bedbathing(env: "Assistax", D: _Derived) -> Any:
    """Not used by `task_metric` (bedbathing's metric is the final wiped fraction, see
    `Assistax.task_metric`) but EMITTED as `info["success"]` by `_step`, which
    `anchored_ppo`'s `train.bc_prior.accept: success` reads. It is therefore the same threshold `success()` applies to the metric -- `n_wiped / 52 >=
    success_threshold`, 26 points at the row's 0.5 -- and not "every point wiped": with the
    stricter flag a teacher that wiped 26-51 points would be counted as failed by the one
    consumer while `success()` called the episode a success."""
    return D.n_wiped / N_WIPE_POINTS >= env.success_threshold


def _chk_armmanipulation(env: "Assistax", D: _Derived) -> Any:
    return D.dist_forearm_waist < 0.08


def _chk_feeding(env: "Assistax", D: _Derived) -> Any:
    return D.xp.logical_and(D.dist < 0.04, D.tip_force_mag < 5.0)


def _chk_teethbrushing(env: "Assistax", D: _Derived) -> Any:
    return _all3(D, D.dist < 0.03, D.tip_force_mag >= 0.1, D.tangential_speed >= 0.01)


def _ref_scratchitch(env: "Assistax", D: _Derived, a: np.ndarray) -> float:
    """scratchitch.py:203 (ctrl cost) and 229-244, constructor defaults: dist 1.0 (scale 0.1),
    scratching 4.0 (v* 0.1 m/s, f* 3 N), ctrl 1e-6. The scratching gate is on the DISTANCE
    (< 0.1 m): the README reads it as a gate on r_dist, the code compares |dist| with
    dist_scale."""
    r_dist = math.exp(-D.dist ** 2 / 0.1)
    gate = float(D.dist < 0.1)
    r_scratch = gate * _boltzmann(D.tool_speed, 0.1) * _boltzmann(D.tool_force_mag, 3.0)
    return 1.0 * r_dist + 4.0 * r_scratch - 1e-6 * float(np.sum(a ** 2))


def _ref_bedbathing(env: "Assistax", D: _Derived, a: np.ndarray) -> float:
    """bedbathing.py:229-250: dist 1.0 (scale 0.1) on the nearest UNWIPED point, ctrl 0,
    plus 3.0 per newly wiped point -- dropped here (needs the previous flags; it is
    what `task_metric` counts). Zero once every point is wiped (all distances inf).

    Two consequences, stated because they are the kind that look like a faithful
    transcription from the outside. (1) "Nearest unwiped" is evaluated on the ARRIVING
    state's flags, where upstream evaluates it on the pre-step mask: on a step that wipes
    a point the two disagree, and here the reward DROPS as the nearest point becomes a
    farther one (measured: it fell on 32 of 34 wipe events, by up to 0.96). (2) Without
    the bonus nothing in this function pays for wiping: its per-episode maximiser hovers
    3 cm from an unwiped point (0.991/step, 946 over 1000 steps) and out-scores a policy
    that wipes all 52 (305). It is upstream's shaping term and is fit for a reference in
    a screen; it is NOT a reward to train an expert on, and the spec says so."""
    if not math.isfinite(D.dist):
        return 0.0
    return 1.0 * math.exp(-D.dist ** 2 / 0.1)


def _ref_armmanipulation(env: "Assistax", D: _Derived, a: np.ndarray) -> float:
    """armmanipulation.py:218 (ctrl cost) and 244-254: waist 10.0 * exp(-d^2/0.1), hook 1.0 * (1 - tanh(d/0.1)),
    rot 0.1 * ||R_target - R_tool||_F, ctrl 1e-6 (the rotation term is a bonus on
    MISalignment, as upstream: it is transcribed, not corrected)."""
    r_hook = 1.0 - math.tanh(D.dist / 0.1)
    r_waist = math.exp(-D.dist_forearm_waist ** 2 / 0.1)
    return 10.0 * r_waist + 1.0 * r_hook + 0.1 * D.rot_err - 1e-6 * float(np.sum(a ** 2))


def _ref_feeding(env: "Assistax", D: _Derived, a: np.ndarray) -> float:
    """feeding.py:181 (ctrl cost) and 204-285: dist 2.0 * exp(-3 d); orientation, velocity
    and force at 1.0 each; ctrl 1e-6. The speed is the tool's instantaneous speed here."""
    dist = D.dist
    r_dist = math.exp(-3.0 * dist)
    to_mouth = _unit(D.mouth_pos - D.tool_pos)
    proximity = math.exp(-150.0 * dist ** 2)
    target_vec = _unit((1.0 - proximity) * np.array([0.0, 0.0, 1.0]) + proximity * to_mouth)
    r_pour = float(np.dot(target_vec, D.spoon_up))
    r_aim = float(np.dot(D.spoon_north, to_mouth))
    r_orientation = 1.0 * r_pour + (0.3 + 0.7 * proximity) * r_aim
    braking = math.exp(-150.0 * dist ** 2)
    target_speed = (1.0 - braking) * 0.20
    r_velocity = math.exp(-((D.tool_speed - target_speed) ** 2) / 0.1 ** 2)
    r_force = _boltzmann(D.tip_force_mag, 1.0)
    return (2.0 * r_dist + 1.0 * (r_orientation + r_velocity + r_force)
            - 1e-6 * float(np.sum(a ** 2)))


def _ref_teethbrushing(env: "Assistax", D: _Derived, a: np.ndarray) -> float:
    """teethbrushing.py:181 (ctrl cost), 204-253 and get_brush_reward (321): dist 2.0 *
    exp(-3 d); brushing 1.0 * (r_brush + r_align * r_dist); ctrl 1e-6. r_brush gates on tip
    force > 0.1 N and peaks at a tangential speed of 0.05 m/s."""
    dist = D.dist
    r_dist = math.exp(-3.0 * dist)
    r_align = (D.bristle_dot_to_mouth + D.tilt_dot_to_mouth) / 2.0
    r_align_weighted = r_align * r_dist
    active = float(D.tip_force_mag > 0.1)
    r_brush = active * _boltzmann(D.tangential_speed, 0.05)
    return 2.0 * r_dist + 1.0 * (r_brush + r_align_weighted) - 1e-6 * float(np.sum(a ** 2))


def _reset_common(env: "Assistax", rng: np.random.Generator) -> None:
    """Upstream `reset`: the `init` keyframe plus U(-5e-3, 5e-3) on every qpos entry and
    every qvel entry (the root quaternion included), then the root quaternion is
    normalised EXPLICITLY. Upstream leaves the raw draw in qpos and its first step
    normalises it; `mj_forward` does NOT (measured: it normalises only its own body
    quaternions), so without this line the reset observation would carry a non-unit root
    quaternion (|q| - 1 up to 6.6e-3) that every later observation lacks."""
    m, d = env._model, env._data
    env._mj.mj_resetDataKeyframe(m, d, env._key)
    d.qpos[:] = d.qpos + rng.uniform(-RESET_NOISE, RESET_NOISE, size=m.nq)
    d.qvel[:] = rng.uniform(-RESET_NOISE, RESET_NOISE, size=m.nv)
    env._mj.mju_normalize4(d.qpos[3:7])
    d.ctrl[:] = 0.0


def _rs_scratchitch(env: "Assistax", rng: np.random.Generator) -> np.ndarray:
    """After the pose: Bernoulli(0.5) picks the arm (1 = lower), then U(-1, 1) along the
    capsule (a fraction of its half-length) and U(0, 2 pi) around it -- upstream's
    `reset`, lines 129-156, with the geom's own radius and half-length."""
    _reset_common(env, rng)
    lower = bool(rng.uniform() < 0.5)
    height = float(rng.uniform(-1.0, 1.0))
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    geom = env._larm_geom if lower else env._uarm_geom
    radius, half = float(env._model.geom_size[geom][0]), float(env._model.geom_size[geom][1])
    return np.array([float(lower), radius * math.cos(angle), radius * math.sin(angle),
                     half * height])


def _rs_bedbathing(env: "Assistax", rng: np.random.Generator) -> np.ndarray:
    _reset_common(env, rng)
    return np.ones(N_WIPE_POINTS)


def _rs_plain(env: "Assistax", rng: np.random.Generator) -> np.ndarray:
    _reset_common(env, rng)
    return np.zeros(0)


def _pool_slots_bedbathing(env: "Assistax", slots: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Flags for a `random_state` pool entry: half the entries keep the rollout's own
    (all unwiped -- a random walk never touches the arm), the rest wipe a random subset
    at a random density. Without this the pool carried 52 ones in every state, and a
    screen comparing candidates on pool states (EPIC, STARC, policy rank) could not tell a
    reward that pays for wiping from one that does not: the term was constant on every
    sample. The tail is recomputed for the new flags by the caller."""
    if rng.uniform() < 0.5:
        return slots
    density = rng.uniform()
    out = slots.copy()
    out[:N_WIPE_POINTS] = np.where(rng.uniform(size=N_WIPE_POINTS) < density, 0.0, 1.0)
    return out


def _adv_bedbathing(env: "Assistax", slots: np.ndarray, D: _Derived) -> np.ndarray:
    """Upstream `_update_contact_vector`: a point is marked wiped when the wiper centre is
    within `target_threshold` of it AND the wiper is in contact with the human -- any
    non-zero summed force between either wiper geom (pad or base cube) and the right
    upper arm, forearm or hand (`human_geoms`). Applied to the arriving state, as upstream."""
    out = slots.copy()
    if D.tool_force_mag > 0.0:
        close = D.wipe_dists < WIPE_THRESHOLD
        out[:N_WIPE_POINTS] = np.where(close, 0.0, out[:N_WIPE_POINTS])
    return out


# --------------------------------------------------------------------------
# row callables: the bedbathing final-state count and scratchitch's itch marker are the
# `metric` / `place_markers` row keys, not `if self._task ==` branches in `task_metric`
# and `render`.
# --------------------------------------------------------------------------

def _metric_bedbathing(env: "Assistax", states: np.ndarray) -> float:
    """The wiped fraction at the FINAL arriving state (`n_wiped / 52`): the flags are
    cumulative, so the last state carries the whole episode's work. Dispatched through
    the row's `metric`; `task_metric` has already refused < 2 rows and non-finite state."""
    c0 = env.n_q + env.n_v
    unwiped = states[-1, c0:c0 + N_WIPE_POINTS] > 0.5
    return float(N_WIPE_POINTS - int(unwiped.sum())) / N_WIPE_POINTS


def _pm_scratchitch(env: "Assistax", D: _Derived) -> None:
    """scratchitch: move the (non-colliding) marker geom of the drawn arm onto the itch,
    in its body's frame -- upstream's `get_sys_for_render`. Model fields, so the change
    persists; it affects nothing but the picture (contype/conaffinity 0)."""
    m = env._model
    arm = int(D.slots[0] > 0.5)
    geom = env._larm_geom if arm else env._uarm_geom
    marker = env._markers[arm]
    # geom frame -> body frame: geom_pos + R_geom @ local
    R = np.zeros(9)
    env._mj.mju_quat2Mat(R, m.geom_quat[geom])
    m.geom_pos[marker] = m.geom_pos[geom] + R.reshape(3, 3) @ D.slots[1:4]
    other = env._markers[1 - arm]
    m.geom_pos[other] = np.array([0.0, 0.0, -10.0])   # the other marker out of sight
    env._mj.mj_forward(m, env._data)


# --------------------------------------------------------------------------
# the task line `render` draws on every frame
# --------------------------------------------------------------------------
#
# `render`'s panel (`Assistax._legend_lines`) names the task and nothing else: the tool's
# pose, every `dist_*`, the force and the wiped count sit in the same tail and none is
# drawn, because each is the answer to the question the frame poses.

#: The panel's width budget: `bird/envs/hud.py` draws a 5x7 glyph at 2x, six scaled pixels
#: a character, so about 28 fit a 360-wide frame.
_LEGEND_COLS = 28


def _task_lines(task: str) -> List[str]:
    """The panel's task line, `ASSISTAX <TASK>`, as a one-element list (every task key
    fits inside `_LEGEND_COLS` on one line)."""
    return [f"ASSISTAX {task.upper()}"]


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

#: One row per task. `scene` is the vendored scene file; `timestep` the MuJoCo timestep the
#: row runs at; `tool_site`/`tool_body` the tool's reference point and body; `tool_geoms` the
#: geoms whose contacts with `human_geoms` are the "force on the human"; `tip_geoms` the
#: subset whose contacts are the "tip" force, on exactly the rows whose tail carries
#: `tip_force_mag` (feeding/teethbrushing distinguish the contact face) and EMPTY elsewhere,
#: so that a row re-derived from the observation agrees with the live step about it;
#: `slots` the task's per-episode memory; `tail` the derived fields;
#: `check`/`ref`/`reset`/`advance` the row's callables; `fitness` names the metric
#: and `threshold` is the PROVISIONAL `success_threshold`; `pool_slots` is how `random_state`
#: varies the task slots a random walk never changes (None where it has none or they vary).
#: Further keys, one key set across the table: `metric` a `(env, states) -> float` reduction where the row's is not the plain per-step
#: fraction (None: the fraction); `camera` the scene camera `render` draws from; `sites` the
#: named sites `_construct_model` resolves onto `self._<key>_site`; `place_markers` an
#: `(env, D)` callable `render` runs before a frame (None: nothing to place).
_TASKS: Dict[str, Dict[str, Any]] = {
    "scratchitch": dict(
        scene="wheelchair_scene.xml", timestep=0.002, upstream="ScratchItch",
        tool_site="scratcher_point", tool_body="scratcher", tool_geoms=("scratcher_stick",),
        tip_geoms=(), human_geoms=("right_uarm1", "right_larm", "right_hand"),
        uarm_geom="right_uarm1", larm_geom="right_larm", markers=("target-u", "target-l"),
        slots=_SLOTS_SCRATCHITCH, tail=_TAIL_SCRATCHITCH,
        reset=_rs_scratchitch, advance=None, derive=_derive_scratchitch,
        check=_chk_scratchitch, ref=_ref_scratchitch, threshold=0.3,
        fitness="scratching_fraction", pool_slots=None,
        metric=None, camera="default", sites={}, place_markers=_pm_scratchitch),
    "bedbathing": dict(
        scene="bed_scene.xml", timestep=0.002, upstream="BedBathing",
        tool_site="wiper_centre", tool_body="wiper", tool_geoms=("wiper_pad", "wiper_base"),
        tip_geoms=(), human_geoms=("right_uarm", "right_larm", "right_hand"),
        uarm_geom="right_uarm", larm_geom="right_larm", markers=(),
        slots=_SLOTS_BEDBATHING, tail=_TAIL_BEDBATHING,
        reset=_rs_bedbathing, advance=_adv_bedbathing, derive=_derive_bedbathing,
        check=_chk_bedbathing, ref=_ref_bedbathing, threshold=0.5,
        fitness="wiped_fraction", pool_slots=_pool_slots_bedbathing,
        metric=_metric_bedbathing, camera="default", sites={}, place_markers=None),
    "armmanipulation": dict(
        scene="bed_scene_armmanip.xml", timestep=0.001, upstream="ArmManipulation",
        tool_site="platform_center", tool_body="hook", tool_geoms=("hook_platform", "hook_end"),
        tip_geoms=(), human_geoms=("right_uarm", "right_larm", "right_hand"),
        uarm_geom="right_uarm", larm_geom="right_larm", markers=(),
        slots=(), tail=_TAIL_ARMMANIPULATION,
        reset=_rs_plain, advance=None, derive=_derive_armmanipulation,
        check=_chk_armmanipulation, ref=_ref_armmanipulation, threshold=0.3,
        fitness="arm_at_waist_fraction", pool_slots=None,
        metric=None, camera="default", sites={"hook_target": "hook_target", "arm_target": "arm_target"},
        place_markers=None),
    "feeding": dict(
        scene="feeding_scene.xml", timestep=0.002, upstream="Feeding",
        tool_site="spoon_center", tool_body="spoon", tool_geoms=("spoon_bowl", "spoon_right_side"),
        tip_geoms=("spoon_right_side",), human_geoms=("head",),
        uarm_geom="right_uarm1", larm_geom="right_larm", markers=(),
        slots=(), tail=_TAIL_FEEDING,
        reset=_rs_plain, advance=None, derive=_derive_feeding,
        check=_chk_feeding, ref=_ref_feeding, threshold=0.3,
        fitness="spoon_at_mouth_fraction", pool_slots=None,
        metric=None, camera="default", sites={"mouth": "mouth"}, place_markers=None),
    "teethbrushing": dict(
        scene="teethbrushing_scene.xml", timestep=0.002, upstream="TeethBrushing",
        tool_site="toothbrush_center", tool_body="toothbrush",
        tool_geoms=("toothbrush_head", "toothbrush_right_side"),
        tip_geoms=("toothbrush_right_side",), human_geoms=("head",),
        uarm_geom="right_uarm1", larm_geom="right_larm", markers=(),
        slots=(), tail=_TAIL_TEETHBRUSHING,
        reset=_rs_plain, advance=None, derive=_derive_teethbrushing,
        check=_chk_teethbrushing, ref=_ref_teethbrushing, threshold=0.2,
        fitness="brushing_fraction", pool_slots=None,
        metric=None, camera="default", sites={"mouth": "mouth"}, place_markers=None),
}

#: The name `tests/test_task_specs.py`'s pasted-block detector looks for: a family-dispatch
#: adapter names each task's reset by the ROW's function (`_rs_scratchitch`), and the
#: detector finds the row through `module._FAMILIES[env_id]['reset']`.
_FAMILIES = _TASKS

#: Every row's key, sorted: upstream's five task keys verbatim (`assistax.registered_envs`
#: minus the two robot-robot tasks); the suffix of every env id.
_EXPECTED: Tuple[str, ...] = tuple(sorted(_TASKS))


def _env_id(task: str) -> str:
    """`tasks._BIRD_ID_RULES["assistax"]` is the definition; this must agree with it."""
    return "assistax_" + task


def _assistax_specs() -> Mapping[str, TaskSpec]:
    found = {s.env_id: s for s in task_index().values() if s.library == "assistax"}
    missing = sorted(set(_EXPECTED) - set(found))
    if missing:
        raise TaskSpecError(
            f"bird/envs/assistax.py has rows for tasks whose specs are gone from the "
            f"catalogue: {missing}. Restore tasks/assistax_<task>/shared_spec.yaml or drop "
            "the row")
    return found


# --------------------------------------------------------------------------
# the adapter
# --------------------------------------------------------------------------

class Assistax(SpecEnvAdapter):
    """One Assistax task; see the module docstring. `name` is `assistax_<task>`."""

    exact_states = None          # MUST stay None: `_sb3_run` derives `continuous` from it
    #: None at class level: the shapes are per-row and `_apply_spec` asserts the spec
    #: against whatever the CLASS declares, so a base-class default would collide.
    obs_dim = None               # type: ignore[assignment]
    horizon = None               # type: ignore[assignment]
    action_dim = ACTION_DIM
    supported_reductions = (PER_STEP,)
    render_width, render_height = RENDER_W, RENDER_H
    frame_skip = SUBSTEPS
    #: Coarse binning for `discretise`: the distance the row's check gates on
    #: (`_tail_index()["dist"]`: tool-to-target, except forearm-to-waist on
    #: armmanipulation), tool height, tool speed, force magnitude.
    _bins = (8, 6, 4, 4)
    n_disc_states = 8 * 6 * 4 * 4

    def __init__(self, task: str, reduction: str = PER_STEP) -> None:
        if task not in _TASKS:
            raise KeyError(f"assistax: no task {task!r}; rows: {sorted(_TASKS)}")
        if reduction not in self.supported_reductions:
            raise TaskSpecError(f"assistax_{task}: evaluate.fitness.reduction={reduction!r} "
                                f"is not one of {self.supported_reductions}")
        self._row = _TASKS[task]
        self._task = task
        self.name = _env_id(task)
        specs = _assistax_specs()
        if task not in specs:
            raise TaskSpecError(f"{self.name}: no tasks/{self.name}/shared_spec.yaml in the "
                                f"catalogue (have {sorted(specs)})")
        spec = specs[task]
        want = spec.raw["continuous_success"]["raw"]["name"]
        if want != self._row["fitness"]:
            raise TaskSpecError(f"{spec.path}: continuous_success.raw.name={want!r} but the "
                                f"adapter row implements {self._row['fitness']!r}")
        self.success_threshold = float(self._row["threshold"])

        # The dependency probe first, with no side effect on the process (the rule
        # `gym_mujoco.py` states: MUJOCO_GL is process-global).
        import importlib.util
        try:
            missing = importlib.util.find_spec("mujoco") is None
        except (ImportError, ValueError):
            missing = True
        if missing:
            raise ImportError(
                f"problem.env_id: {self.name} needs the `mujoco` package, which is not "
                "installed. Either\n    uv sync --extra assistax\n    (or: pip install "
                "'mujoco==3.3.0')\nor run the same method point on an env that needs no "
                "simulator:\n    problem.env_id: pendulum\n(missing: mujoco)")
        chosen = os.environ.get("MUJOCO_GL") or _default_mujoco_gl()
        if chosen:
            os.environ["MUJOCO_GL"] = chosen
        _preload_llvm_before_mujoco()
        import mujoco
        self._mj = mujoco
        self._construct_model(mujoco)
        self.obs_dim = self.n_q + self.n_v + self.n_slots + len(self._row["tail"])
        self.horizon = HORIZON
        self._apply_spec(spec)
        super().__init__()

    # -- model ------------------------------------------------------------------

    def _construct_model(self, mujoco: Any) -> None:
        """Load the vendored scene and resolve every name the row uses. Shared with the
        spec generator (which needs the live model before the spec exists), so nothing
        here reads the spec."""
        row = self._row
        path = ASSETS / row["scene"]
        if not path.is_file():
            raise FileNotFoundError(
                f"{self.name}: {path} is missing. The Assistax scenes are vendored by "
                "scripts/vendor_assistax_assets.py; run it, or check that bird/envs/assets "
                "is present in this checkout or installation")
        m = mujoco.MjModel.from_xml_path(str(path))
        m.opt.timestep = float(row["timestep"])
        self._model, self._data = m, mujoco.MjData(m)
        self.n_q, self.n_v = int(m.nq), int(m.nv)
        self.n_slots = len(row["slots"])
        self.dt = float(row["timestep"]) * SUBSTEPS

        def _id(kind: Any, name: str) -> int:
            i = mujoco.mj_name2id(m, kind, name)
            if i < 0:
                raise RuntimeError(f"{self.name}: scene {row['scene']} has no {kind} named "
                                   f"{name!r}; the asset changed underneath this adapter")
            return int(i)

        O = mujoco.mjtObj
        self._key = _id(O.mjOBJ_KEY, "init")
        self._tool_site = _id(O.mjOBJ_SITE, row["tool_site"])
        self._tool_body = _id(O.mjOBJ_BODY, row["tool_body"])
        self._tool_geoms = frozenset(_id(O.mjOBJ_GEOM, g) for g in row["tool_geoms"])
        self._tip_geoms = frozenset(_id(O.mjOBJ_GEOM, g) for g in row["tip_geoms"])
        has_tip_field = any(name == "tip_force_mag" for name, *_rest in row["tail"])
        if has_tip_field != bool(row["tip_geoms"]):
            raise RuntimeError(f"{self.name}: a row has tip geoms exactly when its tail carries "
                               "tip_force_mag -- otherwise a state re-derived from the "
                               "observation would disagree with the live step about the tip force")
        self._human_geoms = frozenset(_id(O.mjOBJ_GEOM, g) for g in row["human_geoms"])
        self._uarm_geom = _id(O.mjOBJ_GEOM, row["uarm_geom"])
        self._larm_geom = _id(O.mjOBJ_GEOM, row["larm_geom"])
        self._uarm_body = int(m.geom_bodyid[self._uarm_geom])
        self._larm_body = int(m.geom_bodyid[self._larm_geom])
        self._markers = tuple(_id(O.mjOBJ_GEOM, g) for g in row["markers"])
        self._camera = _id(O.mjOBJ_CAMERA, row["camera"])
        # the sites a row's derive reads, by the row's own names (unresolved: -1)
        self._hook_target_site = self._arm_target_site = self._mouth_site = -1
        for attr, site in row["sites"].items():
            setattr(self, f"_{attr}_site", _id(O.mjOBJ_SITE, site))

        # The Panda actuators: the last seven, named actuator1..7 (asserted, not trusted).
        names = [mujoco.mj_id2name(m, O.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        want = [f"actuator{i}" for i in range(1, 8)]
        if names[-7:] != want:
            raise RuntimeError(f"{self.name}: the last seven actuators are {names[-7:]}, "
                               f"expected {want}")
        self._robot_act = np.arange(m.nu - 7, m.nu)
        jids = [int(m.actuator_trnid[i, 0]) for i in self._robot_act]
        self._robot_qadr = np.array([int(m.jnt_qposadr[j]) for j in jids])
        self._robot_dadr = np.array([int(m.jnt_dofadr[j]) for j in jids])
        # bedbathing: 26 points per arm capsule, in the GEOM frame (`_wipe_points_world`)
        self._wipe_local_u = self._cylinder_points(m, self._uarm_geom)
        self._wipe_local_l = self._cylinder_points(m, self._larm_geom)
        self._renderer: Any = None
        self._renderer_pid: Optional[int] = None
        self._pool: List[np.ndarray] = []

    @staticmethod
    def _cylinder_points(m: Any, geom: int, n: int = N_WIPE_POINTS // 2) -> np.ndarray:
        """Upstream `_initialize_targets(n, half_length, radius)`: `n` points at equally
        spaced angles AND equally spaced heights (a helix over the capsule)."""
        radius, half = float(m.geom_size[geom][0]), float(m.geom_size[geom][1])
        angles = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        heights = np.linspace(-half, half, n)
        return np.stack([radius * np.cos(angles), radius * np.sin(angles), heights], axis=-1)

    # -- contacts ----------------------------------------------------------------

    def _contact_forces(self, d: Any) -> Tuple[np.ndarray, np.ndarray]:
        """(force the TOOL applies to the human, force the TIP geoms apply), world frame.

        `mj_contactForce` returns the wrench in the contact frame, whose normal points
        from geom1 to geom2 and whose force is the one acting ON geom2 (equal and
        opposite on geom1). Rotated to the world frame and signed so the vector is the
        force on the HUMAN geom.
        """
        m, mj = self._model, self._mj
        total = np.zeros(3)
        tip = np.zeros(3)
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self._tool_geoms and g2 in self._human_geoms:
                tool, sign = g1, 1.0
            elif g2 in self._tool_geoms and g1 in self._human_geoms:
                tool, sign = g2, -1.0
            else:
                continue
            mj.mj_contactForce(m, d, i, f6)
            frame = np.asarray(c.frame, dtype=float).reshape(3, 3)
            f_world = sign * (frame.T @ f6[:3])
            total += f_world
            if tool in self._tip_geoms:
                tip += f_world
        return total, tip

    # -- action set, bounds -----------------------------------------------------------

    def _build_action_set(self) -> Any:
        """One actuator at a time at +/-1, plus all-zero and both corners: a finite set
        for `sample_transitions`' coverage that hits -1 and +1 on every axis, which is
        what `EnvAdapter.__init__` derives the continuous Box from."""
        n = ACTION_DIM
        rows = [[0.0] * n, [1.0] * n, [-1.0] * n]
        for i in range(n):
            for v in (1.0, -1.0):
                a = [0.0] * n
                a[i] = v
                rows.append(a)
        return rows

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        m = self._model
        lo = np.full(self.obs_dim, -math.pi)
        hi = np.full(self.obs_dim, math.pi)
        for j in range(int(m.njnt)):
            if int(m.jnt_type[j]) == 0:                               # the free joint (the human root)
                adr = int(m.jnt_qposadr[j])
                lo[adr:adr + 3], hi[adr:adr + 3] = (-3.0, -3.0, -1.0), (3.0, 3.0, 3.0)
                lo[adr + 3:adr + 7], hi[adr + 3:adr + 7] = -1.0, 1.0  # unit quaternion
            elif int(m.jnt_limited[j]) and int(m.jnt_type[j]) == 3:  # limited hinge
                adr = int(m.jnt_qposadr[j])
                lo[adr], hi[adr] = float(m.jnt_range[j][0]) - 0.5, float(m.jnt_range[j][1]) + 0.5
        v0 = self.n_q
        lo[v0:v0 + 3], hi[v0:v0 + 3] = -10.0, 10.0                 # root linear
        lo[v0 + 3:v0 + 6], hi[v0 + 3:v0 + 6] = -30.0, 30.0         # root angular
        lo[v0 + 6:v0 + self.n_v], hi[v0 + 6:v0 + self.n_v] = -50.0, 50.0
        for j in range(int(m.njnt)):
            if int(m.jnt_limited[j]) and int(m.jnt_type[j]) == 3:   # limited hinge
                adr = int(m.jnt_qposadr[j])
                lo[adr], hi[adr] = float(m.jnt_range[j][0]) - 0.5, float(m.jnt_range[j][1]) + 0.5
        c0 = v0 + self.n_v
        for i, (_n, _d, slo, shi) in enumerate(self._row["slots"]):
            lo[c0 + i], hi[c0 + i] = slo, shi
        t0 = c0 + self.n_slots
        for i, (_n, _d, _f, tlo, thi) in enumerate(self._row["tail"]):
            lo[t0 + i], hi[t0 + i] = tlo, thi
        return lo, hi

    # -- simulator plumbing ---------------------------------------------------------

    def _forward(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        """Restore `(qpos, qvel)` and compute the kinematic fields FROM IT: the warmstart
        every MuJoCo adapter here zeroes, then one `mj_forward`. It does not touch
        `d.ctrl`, which is why `_Derived` reads contact forces off the row on this path;
        and it does not normalise `qpos`'s quaternion (`_reset_common` does that once)."""
        d = self._data
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        d.qacc_warmstart[:] = 0.0
        self._mj.mj_forward(self._model, d)

    def _obs(self, slots: np.ndarray, D: _Derived) -> np.ndarray:
        d = self._data
        tail = np.array([float(f(D)) for _n, _d, f, _lo, _hi in self._row["tail"]])
        return np.concatenate([d.qpos, d.qvel, slots, tail]).astype(float)

    def _derived_now(self, slots: np.ndarray) -> _Derived:
        """`_Derived` for the state the simulator is IN (after `mj_forward`), with `slots`."""
        d = self._data
        s = np.concatenate([d.qpos, d.qvel, slots, np.zeros(len(self._row["tail"]))])
        return _Derived(self, s, forward=False)

    # -- the interface -----------------------------------------------------------------

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        slots = np.asarray(self._row["reset"](self, rng), dtype=float)
        d = self._data
        d.qacc_warmstart[:] = 0.0
        self._mj.mj_forward(self._model, d)          # kinematics of the reset pose
        D = self._derived_now(slots)                 # (ctrl is 0: the tail's forces are the
        return self._obs(slots, D)                   # passive body's own, 0 N off contact)

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        s = np.asarray(s, dtype=float).ravel()
        if s.shape[0] < self.obs_dim:
            raise ValueError(f"{self.name}: state of length {s.shape[0]}; expected {self.obs_dim}")
        u = np.clip(np.asarray(a, dtype=float).ravel(), -1.0, 1.0)
        if u.shape[0] != ACTION_DIM:
            raise ValueError(f"{self.name}: action of length {u.shape[0]}; this env takes "
                             f"exactly {ACTION_DIM} joint-position targets")
        nq, nv = self.n_q, self.n_v
        slots = s[nq + nv:nq + nv + self.n_slots].copy()
        m, d = self._model, self._data
        d.ctrl[:] = 0.0                             # the passive human -- set BEFORE the
        d.ctrl[self._robot_act] = u                 # forward: the warmstart it leaves for
        self._forward(s[:nq], s[nq:nq + nv])       # the first substep reads the actuation
                                                    # (MuJoCo clips ctrl to ctrlrange)
        for _ in range(SUBSTEPS):
            self._mj.mj_step(m, d)
        # Site poses and contacts of the ARRIVING state, with the warmstart zeroed so the
        # contact forces the tail carries are the ones a later restore (`_forward`, which
        # zeroes it too) recomputes -- otherwise a stored row and its restore disagree.
        d.qacc_warmstart[:] = 0.0
        self._mj.mj_forward(m, d)
        D = self._derived_now(slots)
        advance = self._row["advance"]
        if advance is not None:
            slots = advance(self, slots, D)
            D.slots = slots
            self._row["derive"](self, D, d)         # the tail reads the advanced slots
        s2 = self._obs(slots, D)
        done = not bool(np.isfinite(s2).all())
        if done:
            return s2, True, {"success": False}
        return s2, False, {"success": bool(self._row["check"](self, D))}

    def discretise(self, s: np.ndarray) -> int:
        s = np.asarray(s, dtype=float).ravel()
        tail = self._tail_index()
        d = float(s[tail["dist"]]) if math.isfinite(s[tail["dist"]]) else 5.0
        feats = ((d, 0.0, 1.0, self._bins[0]),
                 (float(s[tail["tool_z"]]), 0.0, 1.5, self._bins[1]),
                 (float(np.linalg.norm(s[tail["tool_vx"]:tail["tool_vx"] + 3])), 0.0, 1.0, self._bins[2]),
                 (float(np.linalg.norm(s[tail["tool_fx"]:tail["tool_fx"] + 3])), 0.0, 20.0, self._bins[3]))
        idx = 0
        for value, lo, hi, n in feats:
            idx = idx * n + _bin(value, lo, hi, n)
        return int(idx)

    def _tail_index(self) -> Dict[str, int]:
        """name -> observation index for the derived fields (`dist` is whichever distance
        the row's check gates on: on armmanipulation that is forearm-to-waist, not the
        tool-to-hook-point distance `_Derived.dist` holds -- `discretise` reads the gated
        one)."""
        t0 = self.n_q + self.n_v + self.n_slots
        idx = {name: t0 + i for i, (name, *_rest) in enumerate(self._row["tail"])}
        for cand in ("dist_tool_target", "dist_tool_nearest_unwiped", "dist_forearm_waist_target",
                     "dist_tool_mouth"):
            if cand in idx:
                idx["dist"] = idx[cand]
                break
        return idx

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        """A REACHABLE state from a lazily refilled rollout pool -- uniform draws over the
        box put the human inside the chair at 30 rad/s and MuJoCo answers with a QACC
        warning, and a drawn quaternion is not a rotation. The pool refills with
        200-step random walks. A row with
        `pool_slots` re-draws its task slots on each pooled state (bedbathing's wipe
        flags, which a random walk leaves at 52 ones) and recomputes the tail for them
        from the same pose -- the contact force is the row's own, so the entry is exactly
        what `_step` would have emitted had those points been wiped earlier."""
        pool_slots = self._row["pool_slots"]
        c0 = self.n_q + self.n_v
        while len(self._pool) < 64:
            obs = self._reset(rng)
            for i in range(200):
                obs, done, _info = self._step(obs, self.action_set[int(rng.integers(self.n_actions))])
                if done:
                    break
                if i % 2 == 0:
                    entry = obs
                    if pool_slots is not None and self.n_slots:
                        slots = pool_slots(self, obs[c0:c0 + self.n_slots], rng)
                        if not np.array_equal(slots, obs[c0:c0 + self.n_slots]):
                            probe = obs.copy()
                            probe[c0:c0 + self.n_slots] = slots
                            entry = self._obs(slots, _Derived(self, probe, forward=True))
                    self._pool.append(entry)
        return self._pool.pop(int(rng.integers(len(self._pool))))

    # -- ground truth -----------------------------------------------------------------

    def task_metric(self, traj: Any) -> float:
        """Fraction of the episode's arriving states on which the row's check holds --
        or, where the row carries a `metric`, that reduction (bedbathing's wiped fraction
        at the FINAL arriving state).

        Per-step, from state only, and `nan` -- never 0.0 -- for a non-finite or empty
        trajectory: a 0.0 for an unmeasurable episode would sit inside the metric's range
        and outrank nothing honestly. The guard runs BEFORE the dispatch, so no row's
        reduction sees garbage.
        """
        states = _states_of(traj)
        if states.shape[0] < 2 or states.shape[1] < self.obs_dim \
                or not bool(np.isfinite(states[:, :self.n_q + self.n_v + self.n_slots]).all()):
            return float("nan")
        metric = self._row["metric"]
        if metric is not None:
            return float(metric(self, states))
        hits = 0
        for row in states[1:]:
            hits += int(bool(self._row["check"](self, _Derived(self, row, forward=True))))
        return hits / float(states.shape[0] - 1)

    def success(self, traj: Any) -> bool:
        v = self.task_metric(traj)
        return bool(math.isfinite(v) and v >= self.success_threshold)

    # --- BEGIN reference reward (ground truth; strip before showing an LLM) ---
    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """Upstream's reward for this task, transcribed term by term (the module docstring
        lists the four substitutions). Sites come from the simulator restored to `s` and
        the contact forces from `s`'s own tail (`_Derived`), so this is a pure function of
        its arguments; `a` is clipped to [-1, 1] before it is squared."""
        s = np.asarray(s, dtype=float).ravel()
        if s.shape[0] < self.obs_dim or not bool(np.isfinite(s[:self.n_q + self.n_v]).all()):
            return 0.0
        u = np.zeros(ACTION_DIM) if a is None else np.clip(np.asarray(a, dtype=float).ravel(), -1.0, 1.0)
        D = _Derived(self, s, forward=True)
        r = float(self._row["ref"](self, D, u))
        return r if math.isfinite(r) else 0.0
    # --- END reference reward ---

    def gt_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None,
                  s2: Optional[np.ndarray] = None) -> float:
        """Alias required by `screens.reference_reward`'s probe list -- see
        `metaworld.gt_reward`. BIRD convention: a reward scores the state it ARRIVES in."""
        return self.reference_reward(s if s2 is None else s2, a)

    # -- describe() -------------------------------------------------------------------

    def _render_full_source(self) -> str:
        """The state and action this adapter emits, the scene in words, and this task's
        constants -- not ~900 lines of five-task plumbing. The reference reward's span is
        included so `strip_existing_reward` has something to cut; the check that scores
        the task is never shown."""
        row = self._row
        # every row names its upstream task (`tests/test_assistax.py` asserts it)
        lineage = f"upstream {row['upstream']}"
        parts = [
            f"# environment `{self.name}` -- Assistax task `{self._task}` ({lineage}), driven through "
            "plain MuJoCo.",
            "#",
            "# The observation is concat(qpos, qvel, task slots, derived fields), documented",
            "# field by field below. Actions are 7 normalised joint-position targets for the",
            "# Panda's actuators in [-1, 1] (target angle ~ pi * a, clipped to the joint's",
            "# control range); the human is passive.",
            "",
            "# --- the state and action this adapter emits ---",
            self._render_api_stub(),
            "",
            "# --- constants ---",
            f"DT, SUBSTEPS, HORIZON = {self.dt}, {SUBSTEPS}, {self.horizon}",
            f"N_WIPE_POINTS, WIPE_THRESHOLD = {N_WIPE_POINTS}, {WIPE_THRESHOLD}",
            "",
            "# --- this adapter's reference reward ---",
            _span_of(Assistax, "reference_reward"),
        ]
        return "\n".join(parts)

    @property
    def reward_source(self) -> str:
        return _span_of(Assistax, "reference_reward")

    # -- rendering ----------------------------------------------------------------------

    #: Whether `render` draws the task-name panel (`_legend_lines`). The pixel-identity
    #: test turns it off to compare BARE frames.
    draw_legend: bool = True

    def render(self, state: np.ndarray, width: int = 320) -> np.ndarray:
        """One `(render_height, render_width, 3)` uint8 frame of the simulator restored to
        `state`, from the row's `camera`, with a small legend. `width` is accepted and
        ignored (the harness resizes). Exactly one positional parameter beyond `self`,
        which `observability._accepts_a_state` requires. Pure in `state`: the row's
        `place_markers` puts every marker where the state says (the itch marker, from the
        slots)."""
        s = np.asarray(state, dtype=float).ravel()
        D = _Derived(self, s, forward=True)
        place = self._row["place_markers"]
        if place is not None:
            place(self, D)
        r = self._renderer_for_this_process()
        r.update_scene(self._data, camera=self._camera)
        frame = np.array(r.render(), dtype=np.uint8, copy=True)
        if self.draw_legend:
            draw_panel(frame, 6, 6, self._legend_lines(D), scale=_HUD_SCALE, rgb=_HUD_RGB)
        return frame

    def _legend_lines(self, D: _Derived) -> List[str]:
        """The task's name, and nothing that depends on `D`: the QUESTION, never the answer.

        These frames are the judge's evidence -- `observability.sample_frames` feeds
        `render` to `evaluate.fitness.source: vlm_score` and to the GT pairwise judge --
        so the legend may state the task's input, never `task_metric`, `success()` or a
        per-step verdict. A panel printing
        `WIPED n/52` (on bedbathing exactly `task_metric * 52`), `DIST` (the distance
        every row's `_chk_*` gates on) or `FORCE` (scratchitch's gated force) would make
        a VLM verdict here a transcription of the metric. The itch on scratchitch is a drawn marker
        (`_pm_scratchitch`), which is how the question reaches the pixels. `D` stays in the
        signature so `render`'s call and the test that pins independence from it
        (`tests/test_assistax.py`) both hold. `_task_lines` returns the family word and
        the upper-cased task key on one line.
        """
        return list(_task_lines(self._task))

    def _renderer_for_this_process(self) -> Any:
        """Lazy and per-process: a GL context must not cross the fork
        `train.candidate_parallelism: parallel` performs after construction."""
        if self._renderer is None or self._renderer_pid != os.getpid():
            self._renderer = self._mj.Renderer(self._model, height=self.render_height,
                                               width=self.render_width)
            self._renderer_pid = os.getpid()
            self._renderer.update_scene(self._data, camera=self._camera)
            self._renderer.render()     # a fresh renderer's first frame differs; spend it here
        return self._renderer

    def task_images(self) -> List[bytes]:
        """PNG bytes of a reset state, for `problem.instruction_modality: text+image`.
        `[]` when imageio or GL is missing: an instruction modality that cannot be
        honoured must degrade to text rather than abort a run."""
        try:
            import imageio.v3 as iio
            frame = self.render(self.reset(0))
            return [bytes(iio.imwrite("<bytes>", frame, extension=".png"))]
        except Exception as exc:  # noqa: BLE001 - optional dep or missing GL context
            log.debug("%s: task_images unavailable (%s)", self.name, exc)
            return []


# --------------------------------------------------------------------------
# registration: one id per row, mechanically derived
# --------------------------------------------------------------------------

def _factory(task: str) -> Callable[[Any], Assistax]:
    def make(ctx: Any) -> Assistax:
        cfg = getattr(ctx, "cfg", None)
        reduction = PER_STEP
        if cfg is not None:
            try:
                reduction = str(cfg.get("evaluate.fitness.reduction", PER_STEP) or PER_STEP)
            except Exception:  # noqa: BLE001 - a ctx without a config
                reduction = PER_STEP
        return Assistax(task, reduction)

    make.__name__ = _env_id(task)
    make.supported_reductions = Assistax.supported_reductions
    make.consumer_forbidden_symbols = _CONSUMER_FORBIDDEN
    make.defines_success = True
    # The spec is NOT consulted at registration (contrast `gym_mujoco._factory`): the spec
    # generator imports this module to read the model before the spec exists, so a
    # missing spec fails at CONSTRUCTION, naming the catalogue, and in
    # `tests/test_task_specs.py`'s partition test -- both loud, neither at import.
    up = _TASKS[task]["upstream"]
    make.__doc__ = (f"Assistax `{task}` (upstream {up}): a Panda arm "
                    "helping a passive human, plain-MuJoCo port.")
    return make


for _task in _EXPECTED:
    register("env", _env_id(_task))(_factory(_task))
