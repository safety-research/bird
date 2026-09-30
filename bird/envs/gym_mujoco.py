"""Ten gymnasium/MuJoCo tasks, each defined by its task spec (§0, §3).

One adapter class, ten registrations, five simulators. The registration is a loop over
the CATALOGUE (`tasks/<id>/shared_spec.yaml`, `env.library.name: mujoco` -- run through
gymnasium, whose own directory name for these is `mujoco`), exactly as
`bird/envs/metaworld.py` registers MT10: ten hand-written factories would be ten chances
for an id to disagree with the task it builds, and an eleventh hand-maintained registry
is exactly what the `tasks/` catalogue replaces. What is hand-written here is per-SIMULATOR
(five `_FAMILIES` rows: observation layout, bounds, cameras, reference reward, which
qpos coordinates are cyclic); everything per-TASK (which env to construct, the horizon,
the ground-truth fitness, the instruction) comes from the spec. A row is the COMPLETE
definition of a family -- `__init__`, `_step`, `random_state` and `render` consume row
keys and never branch on an env id -- so the extension recipe in `_gym_specs`'s error
message is the whole recipe.

NAMING. The id is `gym_` + the spec id VERBATIM: `gym_hopper_hop`, not `gym_Hopper-v5`.
The benchmark key cannot be the suffix -- `HalfCheetah-v5` backs three specs and
`Reacher-v5`, `Swimmer-v5` and `Hopper-v5` two each, because a task is an
(environment, objective) pair -- and the spec id is the catalogue's only unique key,
so it is the one name that picks out a single objective.
The prefix names the suite, as `mt10_` does. The
derivation lives in `tasks._BIRD_ID_RULES["mujoco"]` so the partition test and this
loop cannot disagree.

THE TEN SPECS ARE MAINTAINED IN THIS REPOSITORY (`tasks/SOURCE.md`) and keep a flatter
shape than the other suites' specs: no `flat_fields`, no `env_prose`/`l_task` split,
flat anchors with no reduction label, and `reward.forbidden_symbols: []` with no BIRD
consumer entry. Four consequences, each deliberate rather than deferred:

  * This class is a plain `EnvAdapter`, not a `SpecEnvAdapter` -- the base that reads
    `flat_fields` and `env_prose` would refuse every one of these files, and adding
    those fields is catalogue work (`tasks/SCHEMA_DELTA.md`), not adapter work. The
    observation tables below are authored per family, by hand.
  * `spec.anchors` is never read for normalisation. `bird/envs/spec.py::baselines_of`
    is the only correct reader in the tree and it returns None for a flat pair -- an
    anchor that cannot say which reduction it measured cannot normalise anything. So
    `evaluation._env_baselines` finds nothing, fitness stays RAW (metres, m/s,
    fractions), and that is the schema's absence-is-explicit rule reaching the
    dashboard, not a bug. Eight of the ten carry a MEASURED `expert` anchor
    (PPO trained on `reference_reward`); `hopper_hop_in_place`'s is null because
    its measured value came in BELOW its random anchor, and
    `half_cheetah_backward`'s is null because its recorded number cannot have
    come from its recorded recipe (next paragraph). That changes nothing
    here: the pair is still flat and unlabelled, so `baselines_of` still refuses it.
  * THE REFERENCE REWARD IS PER TASK, NOT PER SIMULATOR. Five specs -- `half_cheetah`,
    `hopper_hop`, `swimmer_forward`, `reacher_reach`, `inverted_pendulum_balance` --
    declare `reward.human.kind: published_dense` and pin the wheel's own reward method;
    the other five -- `half_cheetah_backward`, `half_cheetah_target_speed`,
    `hopper_hop_in_place`, `swimmer_heading`, `reacher_hold` -- declare `kind: none`
    and say in words that the built-in reward pays "the exact thing this task
    penalises" (hop_in_place) or that recording it "would launder that choice into a
    published artifact" (backward). `reference_reward` reads that kind
    (`_reference_reward_shape`) and RAISES `NotImplementedError` on the five, with
    `has_reference_reward` False, so `gt_return`/`gt_reward_curve` are None there,
    `pearson_curve` is undefined rather than a correlation against forward speed,
    and `phases._reference_score` falls through to `task_metric`. Returning the
    FAMILY row's reward on every task would hand a backward task the base
    simulator's forward-velocity reward -- exactly the laundering the specs refuse.
    The derived tasks' recorded "expert" anchors were trained by
    `scripts/train_expert.py` on that family reward; their `source`/`reason` fields
    say what the policy was actually trained on and that it is not a specialist at
    the task.
  * `success()` is overridden to False -- see the method for the three costs that
    carries and where each shows up.
  * The anti-leak gate cannot come from the spec, so it comes from the FACTORY:
    the specs' empty `forbidden_symbols` declares no gate for a generic consumer of
    the spec, not a review of what a candidate can reach in THIS repo (`task_metric`,
    `reference_reward`, `self._env` -- adapter internals, exactly the four symbols
    every Meta-World spec pins for BIRD via `forbidden_symbols_by_consumer`).
    `_CONSUMER_FORBIDDEN` below is advertised on each factory and
    `config._inherit_from_task_spec` falls back to it when the spec supplies
    nothing, so `-s problem.env_id=gym_half_cheetah` over any config that leaves
    `verify.forbidden_symbols: null` inherits the same gate an `mt10_*` override
    would. An overlay that AUTHORS a list (even `[]`, the `singh_orp` idiom) still
    wins.

THE ANCHOR NUMBERS THEMSELVES CARRY TWO RECORDED DEFECTS. Both were verified
by reproduction and are accepted as known limitations:

  * The ten `anchors.random` were measured under TWO mujoco versions while every spec
    states 3.3.0 uniformly in `env.library.stack`. The split: 3.3.0 for
    `half_cheetah`, `reacher_hold`, `hopper_hop_in_place`, `swimmer_heading`; 3.11.0
    for the other six. The attribution is by REPRODUCTION, not inference:
    `half_cheetah`'s committed -4.5772 reproduces to every recorded digit under
    3.3.0 and is not close under 3.11.0. This repo pins `mujoco==3.3.0` (metaworld
    requires it exactly), so six of the ten committed randoms were measured under a
    different engine from the one this adapter runs.
  * `half_cheetah` (-4.5772) and `half_cheetah_backward` (+4.46) are mutually
    contradictory: the backward fitness is defined upstream as the exact negation of
    the forward one, so the pair must negate, and it does not -- that +4.46 is
    3.11.0's forward measurement -4.4554 negated, i.e. the two halves of one
    identity were measured under different physics. Anyone
    normalising against either must re-measure BOTH under one engine first; the
    numbers above say which half to trust under which engine, so the upstream
    reproduction does not have to be redone to find out.

THE STATELESS-STEP CONTRACT, and what it costs per env. Every one of these five
simulators satisfies `obs == concat(qpos, qvel)` once the observation is taken from the
simulator rather than from `_get_obs` -- which is what this adapter does, so `_step` is
a pure restore-then-step, no snapshot cache, and `export_states` / `import_states` /
`retain_states` stay the base class's no-ops truthfully. Two known deviations from the
specs' documented observation, both declared:

  * The state INCLUDES the root position(s) the native observation excludes (cheetah,
    hopper, swimmer x; swimmer y): rendering a stored state must not teleport the
    robot to x=0, and every displacement fitness below reads it directly.
  * Hopper's native observation clips velocities to [-10, 10]; the state here is the
    simulator's, unclipped. The clip is a property of the OBSERVATION, and a state
    that had been clipped could not be restored exactly.
  * Reacher's native observation encodes joint angles as cos/sin and carries derived
    fingertip geometry; the state here is the four qpos (two arm angles, two target
    coordinates) and four qvel. The fingertip is recovered by planar two-link forward
    kinematics whose link constants are asserted against the live model in
    `tests/test_gym_mujoco.py`.

`_step` zeroes `qacc_warmstart`, following the measurement trail in
`mujoco_control.py` (unzeroed error 4.4e-16 on InvertedPendulum-v5, 1.2e-13 on
HalfCheetah-v5) and `humanoid.py` (4.8e-1 on H1). Measured here (mujoco 3.3.0),
shuffled replay, 60 random-action transitions per family, max |state deviation|,
zeroed vs warmstart-left-alone:

    gym_half_cheetah               0.000e+00  vs  5.418e-14
    gym_hopper_hop                 0.000e+00  vs  8.882e-16
    gym_swimmer_forward            0.000e+00  vs  4.441e-16
    gym_reacher_reach              0.000e+00  vs  2.220e-16
    gym_inverted_pendulum_balance  0.000e+00  vs  0.000e+00

The pendulum row is honest rather than a typo: over those 60 transitions the solver
never engaged (the pole stays near upright, no limit is active), so the control shows
nothing there and the zeroing is prophylactic -- kept because whether the solver runs
depends on where the policy goes, not on the env, and `mujoco_control`'s same probe
DID measure a nonzero unzeroed error on the same model once the pole rests on its
stop. Measure with SHUFFLED replay, never round-trips: a round-trip repeats the same
state and cannot see a history dependence at all.

TERMINATION IS THE ENV'S OWN, NOT A TRANSCRIPTION -- on the families that HAVE one.
Each row declares `terminates`; where True (Hopper, InvertedPendulum) `_step` routes
through `self._env.step(u)` so the healthy check and the 0.2 rad limit run upstream's
code and cannot drift from it. Where False (HalfCheetah, Reacher, Swimmer -- no
terminal state exists upstream) `_step` calls `do_simulation` directly:
`env.step` would also build the native observation, compute
the native reward and allocate the info dict, all discarded, on every SAC step of a
run -- an overhead the anti-drift argument does not buy anything for when there is no
termination logic to drift. Episodes on these two families genuinely terminate early,
and that is a property of the TASKS (`balance_fraction` is "steps survived /
horizon"; the hopper instruction says falling ends the episode), not a leak to remove:
contrast `bird/envs/mujoco_control.py`, whose module docstring records why a terminal
is otherwise removed. The entropy-bonus trap recorded there (SAC balances the
pole under `reward = 0` when termination is on) therefore APPLIES on
`gym_inverted_pendulum_balance` and `gym_hopper_*`, and reward-free control runs must be
made before trusting a ranking from those tasks.

FITNESS comes from `continuous_success.raw.name`, resolved through the closed
`_FITNESSES` allow-list -- the spec names the measure, this module implements it, and
a name outside the list is a `TaskSpecError` at construction (the `success_check_for`
pattern). The implementations are the `_fit_*` functions below (each spec's
`provenance.authored_from` names its own), written in telescoped form: per-step
velocity records would hold the env's finite differences `(x_t - x_{t-1}) / dt`,
and a mean of
them telescopes to `(x_T - x_0) / (T * dt)`; the forms below use the telescoped
positions directly, which is algebraically identical and exact in the state this
adapter already carries. A trajectory that cannot be measured -- too short, wrong
width, or containing a non-finite value (the appended post-divergence state) -- scores
`nan`, NEVER 0.0: on the signed fitnesses 0.0 is a CEILING (`neg_speed_error` calls it
a perfect cruise; the displacement tasks go negative on backward drift), so a
malformed record returning it would outrank every honest episode. `nan` is "no
measurement": the seed aggregations drop it and evaluation's collapse layer maps a
candidate with nothing finite to `select.failure_value`, which is the convention
`evaluation._failure_value` documents for exactly this trap.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..registry import register
from ..tasks import TaskSpec, TaskSpecError, index as task_index
from .base import EnvAdapter, _bin, _states_of
from .spec import dr_of
# Shared plumbing, imported rather than duplicated: `mujoco_control` needs only the
# standard library and numpy at import, so there is no install in which these imports
# can fail.
from .mujoco_control import (DEFAULT_CAMERA_CONFIG as _IP_CAMERA,
                             _default_mujoco_gl, _preload_llvm_before_mujoco)
from .metaworld import _safe_source, _span_of

log = logging.getLogger(__name__)


#: The anti-leak gate this CONSUMER needs, advertised on every factory below and
#: inherited by `config._inherit_from_task_spec` when the spec supplies nothing.
#: The four symbols are adapter internals -- how a candidate could reach the metric
#: it is scored on IN THIS REPO -- and match what every Meta-World spec pins for
#: BIRD via `forbidden_symbols_by_consumer`. It cannot live in the specs: their
#: `forbidden_symbols: []` is a statement about a generic consumer of the spec
#: (they carry no BIRD consumer entry), not a review of this repo's adapter internals.
_CONSUMER_FORBIDDEN: Tuple[str, ...] = ("task_metric", "success",
                                        "reference_reward", "_env")


def _reference_reward_shape(spec: TaskSpec, family: Mapping[str, Any]
                            ) -> Optional[Tuple[Any, ...]]:
    """The family's `ref_reward` row IF the task claims a human reward, else None.

    The spec decides, not the simulator: `reward.human.kind` is `published_dense`
    on the five base tasks, which pin the wheel's `_get_rew` and get the row, and
    `none` on the five derived objectives, whose notes disown the built-in reward
    in words -- so they get nothing, and `GymMujoco.reference_reward` raises. Pure
    and simulator-free so `tests/test_gym_reference_reward.py` can hold every
    registered gym task to the rule without constructing one.
    """
    kind = ((spec.reward or {}).get("human") or {}).get("kind")
    if kind == "none":
        return None
    return tuple(family["ref_reward"])


# --------------------------------------------------------------------------
# the ground-truth fitnesses the specs name
# --------------------------------------------------------------------------

#: `half_cheetah_target_speed`'s cruise target; the spec's `raw.expr` quotes the
#: same 2.0.
TARGET_SPEED_MPS = 2.0

#: `reacher_hold`'s dwell radius.
REACH_HOLD_RADIUS_M = 0.05

#: `swimmer_heading`'s commanded direction: the 45-degree diagonal between +x and +y,
#: pi/4 radians.
TARGET_HEADING_RAD = float(np.pi / 4)

#: Reacher's two link lengths in metres, from `assets/reacher.xml` (`body1` at 0.1,
#: `fingertip` at 0.11). ONE home for the constants both the dwell/reach fitnesses
#: and the `fingertip_pos` helper use; asserted against the live model's
#: `get_body_com("fingertip")` in `tests/test_gym_mujoco.py`.
_REACHER_L1, _REACHER_L2 = 0.1, 0.11

#: Where the reacher's target sits in the state: `qpos[2:4]` -- `target_x`, `target_y`
#: in the Reacher-v5 row's `state_fields` -- the one per-episode quantity any of these
#: ten tasks varies. Named once so `target_pos` and `_reacher_tip_distances` cannot
#: disagree about where it is; `metaworld._GOAL` is the precedent.
_REACHER_TARGET = slice(2, 4)


def _reacher_tip_distances(states: np.ndarray) -> np.ndarray:
    """Fingertip-target distance per row, vectorised (planar two-link FK)."""
    t0 = states[:, 0]
    t01 = states[:, 0] + states[:, 1]
    tx, ty = states[:, _REACHER_TARGET].T
    fx = _REACHER_L1 * np.cos(t0) + _REACHER_L2 * np.cos(t01) - tx
    fy = _REACHER_L1 * np.sin(t0) + _REACHER_L2 * np.sin(t01) - ty
    return np.hypot(fx, fy)


def _fit_forward_displacement(env: "GymMujoco", states: np.ndarray) -> float:
    """Final x minus reset x: metres travelled along +x over one episode."""
    return float(states[-1, 0]) - float(states[0, 0])


def _fit_backward_displacement(env: "GymMujoco", states: np.ndarray) -> float:
    """Metres travelled along -x: the forward measure, negated (upstream defines it
    as exactly that, which is why the two random anchors must negate -- see the
    module docstring for why the committed pair does not)."""
    return -_fit_forward_displacement(env, states)


def _fit_target_speed(env: "GymMujoco", states: np.ndarray) -> float:
    """-|episode-mean forward velocity - 2 m/s|: 0 is a perfect cruise, else < 0.

    Upstream means the per-step finite differences; the telescoped form here is the
    same number computed from the endpoints."""
    n_steps = states.shape[0] - 1
    v_bar = (float(states[-1, 0]) - float(states[0, 0])) / (n_steps * env.dt)
    return -abs(v_bar - TARGET_SPEED_MPS)


def _fit_hop_minus_drift(env: "GymMujoco", states: np.ndarray) -> float:
    """std of the torso height minus |net x drift|, both metres.

    z over ARRIVING states (`states[1:]`), matching upstream's one record per
    executed step; the drift origin is the reset state, matching its `x_start`.
    Known scale asymmetry (a 0.10 m hop gives std(z) ~ 0.07 while drift reaches
    metres) is upstream's, recorded in its docstring, and reproduced rather than
    rebalanced."""
    z = states[1:, 1]
    drift = abs(float(states[-1, 0]) - float(states[0, 0]))
    return float(np.std(np.asarray(z, dtype=np.float64))) - drift


def _fit_balance_fraction(env: "GymMujoco", states: np.ndarray) -> float:
    """Fraction of the nominal horizon survived before the pole fell.

    No clamp: every producer that reaches `task_metric` is horizon-capped, so a
    value above 1.0 is not a case to absorb but a bug to surface -- a wrong
    `horizon` or concatenated episodes -- and a `min(1.0, ...)` here would report
    it as a perfect score instead."""
    return (states.shape[0] - 1) / float(env.horizon)


def _fit_reach_hold(env: "GymMujoco", states: np.ndarray) -> float:
    """Fraction of the episode with the fingertip inside REACH_HOLD_RADIUS_M.

    The deliberate contrast with `neg_final_reach_distance`, which reads the last
    step only: sweeping through the target and drifting off scores well there and
    near zero here."""
    return float(np.mean(_reacher_tip_distances(states[1:]) <= REACH_HOLD_RADIUS_M))


def _fit_final_reach_distance(env: "GymMujoco", states: np.ndarray) -> float:
    """Negated FINAL fingertip-target distance: 0 is on-target, else < 0."""
    return -float(_reacher_tip_distances(states[-1:])[0])


def _fit_heading_speed(env: "GymMujoco", states: np.ndarray) -> float:
    """Mean speed along TARGET_HEADING_RAD (m/s); negative if it swims the wrong way.

    Upstream means per-step `(vx, vy) . (cos, sin)` where vx, vy are the env's
    finite differences, so the mean telescopes to the projected net displacement
    over elapsed time -- the form used here."""
    n_steps = states.shape[0] - 1
    ux, uy = math.cos(TARGET_HEADING_RAD), math.sin(TARGET_HEADING_RAD)
    dx = float(states[-1, 0]) - float(states[0, 0])
    dy = float(states[-1, 1]) - float(states[0, 1])
    return (dx * ux + dy * uy) / (n_steps * env.dt)


def _fit_gated_mean_com_velocity(env: "GymMujoco", states: np.ndarray) -> float:
    """REvolve's Auto fitness for Humanoid Locomotion -- App. B.2 Eq. 3
    (`refs/tex/revolve/main.tex:1121-1129`): the episode-mean forward velocity of
    the CENTRE OF MASS, and **0 unless the episode survived all `T_max` steps**.

        sigma = (1/T_max) * sum_t v_t^x   if T == T_max,   else 0

    Three things about it that a transcription of the formula alone gets wrong:

    * `v_t^x` IS THE MASS CENTRE'S VELOCITY, NOT `qvel[0]`. Both gymnasium's
      `humanoid_v5.step` and REvolve's own vendored copy
      (`refs/code/Revolve/rl_agent/HumanoidEnv.py:441-450`, whose `mass_center`
      at `:58` is gymnasium's `:17` with `expand_dims` for `einsum`) difference
      `mass_center()` -- the body-mass-weighted mean of `data.xipos` -- across
      the step. The root's own x velocity is a DIFFERENT quantity and not a
      usable stand-in: measured over the zero-action trajectory it correlates
      **-0.39** with the env's value (the torso counter-rotates against the
      legs), which is why `com_x` runs a forward pass instead of reading
      `s[env.n_q]`. Getting this wrong would have scored the task on a signal
      anti-correlated with the one the paper reports.
    * THE MEAN TELESCOPES, so the whole episode costs two forward passes rather
      than one per step: upstream's `v_t` is `(com_x(t) - com_x(t-1)) / dt`, so
      the sum collapses to `(com_x(T) - com_x(0)) / (T * dt)`. Exact in theory;
      measured residual against the mean of the env's own logged per-step values
      **3.2e-07 m/s on 0.141 m/s** (21-step rollout), and the
      residual is not float noise -- `mass_center` read straight after
      `do_simulation` sees `xipos` from before the last integration substep
      (gymnasium's v5 notes cite `deepmind/mujoco#889` for exactly this), while
      `com_x` re-runs `mj_forward` and sees the arrived pose. The re-forwarded
      value is the more correct of the two; the difference is declared, not
      hidden.
    * `0.0` IS A MEASUREMENT HERE, NOT THE MALFORMED-RECORD VALUE, which is the
      one place this fitness departs from the module docstring's `nan` rule --
      and it does not collide with it: `task_metric` has already returned `nan`
      for a record with under two rows, too narrow, or carrying a non-finite
      value, so a short well-formed episode reaching this function means exactly
      what Eq. 3's `T < T_max` branch means -- the humanoid fell. Note the
      consequence, because it is a property of the PAPER's fitness and not of
      this port: a reward that makes the humanoid fall at once scores 0.0 and so
      **outranks one that keeps it upright for 1000 steps walking backwards**
      (mean COM vx < 0). The spec records it; nothing here rebalances it.
    """
    n_steps = states.shape[0] - 1
    if n_steps < int(env.horizon):
        return 0.0
    return (env.com_x(states[-1]) - env.com_x(states[0])) / (n_steps * env.dt)


#: `continuous_success.raw.name` -> implementation. CLOSED: the spec names the
#: measure and this table is the only resolver, so a spec whose fitness this module
#: has not implemented fails at construction with the file named, never by silently
#: scoring with a lookalike. `balance_fraction` and `dwell_fraction` are in [0, 1]
#: by construction; every other entry is signed and unbounded on at least one side,
#: which is why `task_metric`'s failure value is `nan` and not 0.0.
_FITNESSES: Dict[str, Callable[["GymMujoco", np.ndarray], float]] = {
    "forward_displacement": _fit_forward_displacement,
    "backward_displacement": _fit_backward_displacement,
    "neg_speed_error": _fit_target_speed,
    "hop_minus_drift": _fit_hop_minus_drift,
    "balance_fraction": _fit_balance_fraction,
    "dwell_fraction": _fit_reach_hold,
    "neg_final_reach_distance": _fit_final_reach_distance,
    "heading_speed": _fit_heading_speed,
    "gated_mean_com_velocity": _fit_gated_mean_com_velocity,
}


# --------------------------------------------------------------------------
# the five simulator families
# --------------------------------------------------------------------------
#
# Everything in a row is a fact about ONE gymnasium env, shared by every objective
# defined on it, and the row is the COMPLETE per-family definition: `__init__`
# consumes `camera_config` and asserts `n_q`/`n_v`/`dt` against the live model,
# `_step` consumes `terminates`, `random_state` consumes `pin_qpos` and `goal_disk`.
# Constants are read from the installed wheel (`gymnasium 1.3.0`, `mujoco 3.3.0` --
# the artifact the run executes, the same rationale as the specs' whole-file
# sha256s); `frame_skip` is read off the env at construction rather than pinned.
#
# Row keys beyond the obvious tables:
#   terminates     True => `_step` runs `env.step` so upstream's termination logic
#                  executes verbatim; False => `do_simulation` directly (the env has
#                  no terminal state, and `env.step` would compute an unused native
#                  observation, reward and info dict per step).
#   pin_qpos       qpos indices held at 0 by `random_state`: cyclic coordinates that
#                  provably cannot change a reward difference (measured on
#                  HalfCheetah-v5: a 37.5 m x-shift moves the stepped
#                  result by 6.2e-13).
#   goal_disk      (start_index, radius) of a goal drawn uniformly in a disk, or
#                  None. Reacher's env rejection-samples its target uniformly inside
#                  r=0.2 (`reacher_v5.py::reset_model`); `random_state` matches that
#                  distribution rather than projecting box draws onto the rim, which
#                  would put 57% of screen-coverage mass exactly on the boundary
#                  band where `dwell_fraction`'s threshold sits.
#   camera_config  passed to `gym.make` when the model defines no camera and the
#                  default free camera needs pinning; None where the model (or
#                  gymnasium's fixed config) already yields a pure render.
#   quat_qpos      index at which a UNIT quaternion starts in qpos, or None. Humanoid's
#                  free root puts one at qpos[3:7], and `set_state` does NOT
#                  renormalise it -- measured: a norm-2.0 quaternion goes
#                  in and comes back out at norm 2.0 -- so `random_state`'s uniform
#                  box draw would hand the simulator an orientation that is not a
#                  rotation at all. `random_state` normalises the four components
#                  after the draw. None on the five planar rows, whose roots are
#                  slide and hinge joints only.
#   height_index   index of the root's height coordinate, or None where the row's
#                  `helpers` does not advertise `torso_height(s)` (pendulum, reacher,
#                  swimmer -- on those, s[1] is a pole angle, an elbow angle and a y
#                  position respectively, and `torso_height` raises rather than
#                  returning one of them under a name it does not have). 1 on the two
#                  planar locomotion rows, 2 on Humanoid, whose free root spends
#                  qpos[0:2] on x and y.
#   com_velocity   True => `forward_velocity(s)` is the CENTRE OF MASS's x velocity,
#                  computed by a forward pass (`com_vx`); False => it is `s[n_q]`,
#                  the root's own first velocity component. A per-family fact and not
#                  a preference: on Humanoid the two correlate **-0.39** over a
#                  zero-action episode, so the planar rows' reading is not an
#                  approximation there but the wrong sign. See
#                  `_fit_gated_mean_com_velocity`.
#   healthy        the row's own termination test as a tagged tuple, or None where the
#                  env has none. `("hopper_upright", z_min, angle_max, state_max)` is
#                  Hopper-v5's three-clause check (`is_healthy`); `("z_range", lo, hi)`
#                  is Humanoid-v5's single strict height band (`is_upright`). Two
#                  names rather than one method for both, following this module's
#                  existing per-family helper naming (`pole_angle`, `heading`,
#                  `fingertip_pos`): the checks are not the same predicate, and
#                  advertising one name would tell a model Hopper's pitch clause
#                  applies to a humanoid.
#
# `ref_reward`: which shape the SIMULATOR's built-in reward takes -- a per-family
# fact. Whether a TASK exposes it as `reference_reward` is the spec's
# `reward.human.kind` (`_reference_reward_shape`): a `kind: none` spec on this
# family gets no reference reward at all, not this row. Shapes --
#   ("velocity", w_fwd, w_ctrl, healthy_bonus): w_fwd * qvel_x - w_ctrl * |a|^2
#     (+1 while healthy, hopper only). The velocity term uses the instantaneous
#     qvel where the env uses its per-step finite difference, a declared
#     approximation: a (state, action) reward cannot see the next state.
#   ("reacher",): -|fingertip - target| * 1.0 - |a|^2 * 1.0. NOTE the weights: the
#     wheel's docstring table says `reward_control_weight` defaults to 0.1, but the
#     v5 signature says 1.0 (`reacher_v5.py:150-151`) and the signature is what
#     executes.
#   ("alive",): +1 per step while the pole is up. Informative HERE because this
#     adapter keeps the terminal state: an episode that ends early collects strictly
#     less of it. With the terminal removed it would be a constant
#     (`mujoco_control.py`'s module docstring).
#: The DR axes this adapter can honour, and the mjModel arrays each one
#: writes. WHICH axes a task declares is the SPEC's business (`domain_randomization` in
#: `tasks/<id>/shared_spec.yaml`, read through `bird.envs.spec.dr_of` in `__init__`); a
#: spec naming an axis absent from this table is refused at construction. Every axis is a
#: multiplicative scale on the arrays captured at construction (nominal 1.0), written on
#: every `_reset` from `scale x nominal` so nothing compounds -- `metaworld.py::_apply_dr`'s
#: contract.
#:
#:   body_mass_scale         body_mass AND body_inertia of every non-world body (a mass
#:                           without its inertia is not a physical mass)
#:   torso_mass_scale        body_mass/body_inertia of the row's `dr_torso_body` only --
#:                           the legged "payload" axis (DrEureka's added_mass)
#:   contact_friction_scale  geom_friction[:, 0] of EVERY geom, floor included: MuJoCo
#:                           max-mixes the two sides of a contact, so a factor below 1
#:                           that left the floor alone would change nothing
#:   gravity_scale           opt.gravity
#:   motor_strength_scale    actuator_gear[:, 0] -- for a `<motor>` the applied force is
#:                           gear x ctrl (gainprm is 1, biasprm 0 on all five models)
#:   joint_damping_scale     dof_damping of every dof (root dofs carry 0.0 and stay 0.0)
#:   viscosity_scale         opt.viscosity -- the swimmer's medium is its ground; 0.0 on
#:                           the other four models, which therefore do not declare it
_GYM_AXES: Dict[str, Tuple[str, ...]] = {
    "body_mass_scale": ("body_mass", "body_inertia"),
    "torso_mass_scale": ("body_mass", "body_inertia"),
    "contact_friction_scale": ("geom_friction",),
    "gravity_scale": ("opt.gravity",),
    "motor_strength_scale": ("actuator_gear",),
    "joint_damping_scale": ("dof_damping",),
    "viscosity_scale": ("opt.viscosity",),
}
#: mjModel arrays `_apply_dr` rewrites from the captured nominal on every reset.
_DR_MODEL_FIELDS = ("body_mass", "body_inertia", "geom_friction", "actuator_gear", "dof_damping")

_FAMILIES: Dict[str, Dict[str, Any]] = {
    "HalfCheetah-v5": dict(
        n_q=9, dt=0.05, action_clip=1.0,
        terminates=False, pin_qpos=(0,), goal_disk=None, camera_config=None,
        dr_torso_body="torso",
        ref_reward=("velocity", 1.0, 0.1, False),
        quat_qpos=None, height_index=1, com_velocity=False, healthy=None,
        render=(480, 240),
        # root x, z, pitch: x cyclic and unbounded, pitch a full turn (unlimited in
        # the model; a fallen cheetah rotates past vertical).
        root_lo=(-1e3, -1.0, -math.pi), root_hi=(1e3, 1.0, math.pi),
        vel_lo=(-20.0, -10.0, -20.0) + (-30.0,) * 6,
        vel_hi=(20.0, 10.0, 20.0) + (30.0,) * 6,
        bins=(1, 9, 9, 5, 5, 5, 5, 5, 5),
        state_fields=(
            ("x_position", "s[0] -- horizontal position of the torso in metres, "
             "increasing forwards. Unbounded; it grows without limit as the robot "
             "runs. NOTE this coordinate is absent from this environment's usual "
             "observation and is included here"),
            ("torso_height", "s[1] -- vertical position of the torso in metres, "
             "relative to its resting height; 0 is nominal, negative is lower"),
            ("torso_pitch", "s[2] -- pitch angle of the torso in radians; 0 is "
             "level, positive is nose-up. UNLIMITED -- the robot can rotate past "
             "vertical and land on its back"),
            ("bthigh_angle", "s[3] -- back thigh joint angle in radians"),
            ("bshin_angle", "s[4] -- back shin joint angle in radians"),
            ("bfoot_angle", "s[5] -- back foot joint angle in radians"),
            ("fthigh_angle", "s[6] -- front thigh joint angle in radians"),
            ("fshin_angle", "s[7] -- front shin joint angle in radians"),
            ("ffoot_angle", "s[8] -- front foot joint angle in radians"),
            ("x_velocity", "s[9] -- forward velocity of the torso in m/s; positive "
             "is forwards"),
            ("z_velocity", "s[10] -- vertical velocity of the torso in m/s"),
            ("pitch_velocity", "s[11] -- angular velocity of the torso in rad/s"),
            ("bthigh_velocity", "s[12] -- back thigh angular velocity in rad/s"),
            ("bshin_velocity", "s[13] -- back shin angular velocity in rad/s"),
            ("bfoot_velocity", "s[14] -- back foot angular velocity in rad/s"),
            ("fthigh_velocity", "s[15] -- front thigh angular velocity in rad/s"),
            ("fshin_velocity", "s[16] -- front shin angular velocity in rad/s"),
            ("ffoot_velocity", "s[17] -- front foot angular velocity in rad/s"),
        ),
        action_fields=(
            ("bthigh_torque", "a[0] -- torque at the back thigh, in [-1, 1]"),
            ("bshin_torque", "a[1] -- torque at the back shin, in [-1, 1]"),
            ("bfoot_torque", "a[2] -- torque at the back foot, in [-1, 1]"),
            ("fthigh_torque", "a[3] -- torque at the front thigh, in [-1, 1]"),
            ("fshin_torque", "a[4] -- torque at the front shin, in [-1, 1]"),
            ("ffoot_torque", "a[5] -- torque at the front foot, in [-1, 1]"),
        ),
        helpers=(
            ("forward_velocity(s)", "s[9], the torso's forward velocity in m/s"),
            ("torso_height(s)", "s[1], the torso's height relative to nominal"),
            ("joint_angles(s)", "the six actuated joint angles, s[3:9]"),
            ("joint_velocities(s)", "the six actuated joint velocities, s[12:18]"),
            ("control_cost(a)", "sum of squared torques, a measure of effort"),
        ),
        joint_slice=(3, 9), joint_vel_slice=(12, 18),
        symbol_mapping={
            "forward_velocity": "s[9]", "velocity": "s[9]", "x_velocity": "s[9]",
            "speed": "s[9]", "x_position": "s[0]", "position": "s[0]",
            "torso_height": "s[1]", "height": "s[1]", "torso_angle": "s[2]",
            "pitch": "s[2]", "joint_angles": "s[3:9]",
            "joint_velocities": "s[12:18]",
            "control_cost": "np.sum(np.square(a))", "torques": "a",
        },
        prose=(
            "A two-dimensional cheetah-like robot on a flat, endless floor, seen "
            "from the side. It has a torso and two legs -- one at the back and one "
            "at the front -- each made of a thigh, a shin and a foot, six "
            "torque-controlled joints in total; it moves only by pushing against "
            "the floor with its feet. The torso is free to slide forwards and "
            "backwards, rise and fall, and pitch without limit, so the robot can "
            "and does end up on its back. The episode never ends early: there is "
            "no way to fail permanently, and a robot that has flipped over simply "
            "spends the rest of the episode flipped over."
        ),
    ),
    "Hopper-v5": dict(
        n_q=6, dt=0.008, action_clip=1.0,
        terminates=True, pin_qpos=(0,), goal_disk=None, camera_config=None,
        dr_torso_body="torso",
        ref_reward=("velocity", 1.0, 1e-3, True),
        quat_qpos=None, height_index=1, com_velocity=False,
        healthy=("hopper_upright", 0.7, 0.2, 100.0),
        render=(480, 320),
        root_lo=(-1e3, 0.0, -math.pi), root_hi=(1e3, 2.0, math.pi),
        vel_lo=(-15.0, -15.0, -15.0) + (-25.0,) * 3,
        vel_hi=(15.0, 15.0, 15.0) + (25.0,) * 3,
        bins=(1, 7, 7, 5, 5, 5),
        state_fields=(
            ("x_position", "s[0] -- horizontal position of the torso in metres, "
             "increasing forwards. NOTE this coordinate is absent from this "
             "environment's usual observation and is included here"),
            ("z_height", "s[1] -- torso height in metres; the episode ends the "
             "moment it drops below 0.7"),
            ("torso_angle", "s[2] -- torso pitch angle in radians; the episode "
             "ends the moment |angle| exceeds 0.2"),
            ("thigh_angle", "s[3] -- thigh joint angle in radians"),
            ("leg_angle", "s[4] -- leg joint angle in radians"),
            ("foot_angle", "s[5] -- foot joint angle in radians"),
            ("x_velocity", "s[6] -- forward velocity of the torso in m/s. NOTE the "
             "environment's usual observation clips velocities to [-10, 10]; this "
             "state is the simulator's, unclipped"),
            ("z_velocity", "s[7] -- vertical velocity of the torso in m/s"),
            ("angle_velocity", "s[8] -- torso pitch angular velocity in rad/s"),
            ("thigh_velocity", "s[9] -- thigh joint angular velocity in rad/s"),
            ("leg_velocity", "s[10] -- leg joint angular velocity in rad/s"),
            ("foot_velocity", "s[11] -- foot joint angular velocity in rad/s"),
        ),
        action_fields=(
            ("thigh_torque", "a[0] -- torque at the thigh (hip), in [-1, 1]"),
            ("leg_torque", "a[1] -- torque at the leg (knee), in [-1, 1]"),
            ("foot_torque", "a[2] -- torque at the foot (ankle), in [-1, 1]"),
        ),
        helpers=(
            ("forward_velocity(s)", "s[6], the torso's forward velocity in m/s"),
            ("torso_height(s)", "s[1], the torso's height in metres"),
            ("is_healthy(s)", "1.0 while the robot is up (height above 0.7, "
             "|torso angle| below 0.2, and every value from the angle onward "
             "inside (-100, 100)); the episode ends the step this becomes 0.0"),
            ("joint_angles(s)", "the three actuated joint angles, s[3:6]"),
            ("joint_velocities(s)", "the three actuated joint velocities, s[9:12]"),
            ("control_cost(a)", "sum of squared torques, a measure of effort"),
        ),
        joint_slice=(3, 6), joint_vel_slice=(9, 12),
        symbol_mapping={
            "forward_velocity": "s[6]", "velocity": "s[6]", "x_velocity": "s[6]",
            "speed": "s[6]", "x_position": "s[0]", "position": "s[0]",
            "z_height": "s[1]", "height": "s[1]", "torso_height": "s[1]",
            "torso_angle": "s[2]", "pitch": "s[2]", "joint_angles": "s[3:6]",
            "joint_velocities": "s[9:12]", "control_cost": "np.sum(np.square(a))",
        },
        prose=(
            "A planar one-legged hopping robot -- a torso atop a single leg with "
            "three torque-controlled joints (thigh, leg and foot) -- stands "
            "upright on flat ground, seen from the side. The root is free to "
            "slide forwards, rise and fall, and pitch. The episode ends the "
            "moment the robot falls: torso height below 0.7 m or torso pitch "
            "beyond 0.2 rad from vertical, so staying up is a precondition for "
            "doing anything at all."
        ),
    ),
    "InvertedPendulum-v5": dict(
        n_q=2, dt=0.04, action_clip=3.0,
        terminates=True, pin_qpos=(), goal_disk=None, camera_config=_IP_CAMERA,
        ref_reward=("alive",),
        quat_qpos=None, height_index=None, com_velocity=False, healthy=None,
        render=(480, 240),
        # cart slide limited to +/-1 in the model; pole hinge limited to +/-pi/2 --
        # both read from the model at construction, so these two conventions cover
        # nothing (kept for the uniform row shape).
        root_lo=(-1.0, -math.pi / 2), root_hi=(1.0, math.pi / 2),
        vel_lo=(-10.0, -10.0), vel_hi=(10.0, 10.0),
        bins=(7, 9),
        state_fields=(
            ("cart_position", "s[0] -- cart position on the rail in metres; 0 is "
             "the centre"),
            ("pole_angle", "s[1] -- pole angle from vertical in radians; 0 is "
             "upright, and the episode ends the moment |angle| exceeds 0.2"),
            ("cart_velocity", "s[2] -- cart velocity along the rail in m/s"),
            ("pole_angvel", "s[3] -- pole angular velocity in rad/s"),
        ),
        action_fields=(
            ("cart_force", "a[0] -- force on the cart along the rail, in [-3, 3]"),
        ),
        helpers=(
            ("pole_angle(s)", "s[1], the pole's angle from vertical in radians"),
            ("cart_position(s)", "s[0], the cart's position on the rail in metres"),
            ("is_balanced(s)", "1.0 while the pole is up (|s[1]| <= 0.2); the "
             "episode ends the step this becomes 0.0"),
            ("control_cost(a)", "sum of squared forces, a measure of effort"),
        ),
        joint_slice=(0, 2), joint_vel_slice=(2, 4),
        # No single-letter keys (such as "x"): a mapping that claims one
        # rewrites every throwaway variable a candidate might use --
        # `tests/test_env_spec_leak.py::test_no_symbol_mapping_key_is_a_single_letter`.
        symbol_mapping={
            "pole_angle": "s[1]", "angle": "s[1]", "theta": "s[1]",
            "cart_position": "s[0]", "position": "s[0]",
            "cart_velocity": "s[2]", "pole_angvel": "s[3]",
            "angular_velocity": "s[3]", "control_cost": "np.sum(np.square(a))",
        },
        prose=(
            "A cart slides along a horizontal rail with a rigid pole hinged to "
            "its top. The pole starts nearly vertical and gravity wants to topple "
            "it; the only control is a force pushing the cart left or right along "
            "the rail. The episode ends the moment the pole leans more than 0.2 "
            "radians from vertical, so every additional step the pole stays up is "
            "an achievement in itself."
        ),
    ),
    "Reacher-v5": dict(
        n_q=4, dt=0.02, action_clip=1.0,
        terminates=False, pin_qpos=(), goal_disk=(2, 0.2), camera_config=None,
        ref_reward=("reacher",),
        quat_qpos=None, height_index=None, com_velocity=False, healthy=None,
        render=(480, 480),
        # target x/y conventions; the model's own jnt_range (+/-0.27) overrides
        # them below, and `goal_disk` is what keeps sampled targets inside the
        # r=0.2 disk the env actually draws goals from.
        root_lo=(-math.pi, -math.pi, -0.2, -0.2),
        root_hi=(math.pi, math.pi, 0.2, 0.2),
        vel_lo=(-25.0, -25.0, 0.0, 0.0), vel_hi=(25.0, 25.0, 0.0, 0.0),
        bins=(5, 5, 3, 3),
        state_fields=(
            ("shoulder_angle", "s[0] -- shoulder joint angle in radians "
             "(unlimited; the arm can spin)"),
            ("elbow_angle", "s[1] -- elbow joint angle in radians"),
            ("target_x", "s[2] -- the target dot's x coordinate in metres; fixed "
             "for the whole episode"),
            ("target_y", "s[3] -- the target dot's y coordinate in metres; fixed "
             "for the whole episode"),
            ("shoulder_velocity", "s[4] -- shoulder angular velocity in rad/s"),
            ("elbow_velocity", "s[5] -- elbow angular velocity in rad/s"),
            ("target_x_velocity", "s[6] -- always 0.0; the target does not move"),
            ("target_y_velocity", "s[7] -- always 0.0; the target does not move"),
        ),
        action_fields=(
            ("shoulder_torque", "a[0] -- torque at the shoulder, in [-1, 1]"),
            ("elbow_torque", "a[1] -- torque at the elbow, in [-1, 1]"),
        ),
        helpers=(
            ("fingertip_pos(s)", "the fingertip's (x, y) in metres, from the two "
             "joint angles"),
            ("target_pos(s)", "the target's (x, y) in metres, s[2:4]"),
            ("to_target_dist(s)", "the fingertip-target distance in metres; 0 "
             "means on target"),
            ("control_cost(a)", "sum of squared torques, a measure of effort"),
        ),
        joint_slice=(0, 2), joint_vel_slice=(4, 6),
        symbol_mapping={
            "joint_angles": "s[0:2]", "joint_velocities": "s[4:6]",
            "target_position": "s[2:4]", "goal_position": "s[2:4]",
            "control_cost": "np.sum(np.square(a))",
        },
        prose=(
            "A two-joint planar robot arm is anchored at the centre of a small "
            "arena, seen from above, and a target dot sits at a random reachable "
            "spot that does not move for the rest of the episode. The shoulder "
            "and elbow joints are torque-controlled; the fingertip is the free "
            "end of the outer link. The episode always runs its full length."
        ),
    ),
    "Swimmer-v5": dict(
        n_q=5, dt=0.04, action_clip=1.0,
        terminates=False, pin_qpos=(0, 1), goal_disk=None, camera_config=None,
        ref_reward=("velocity", 1.0, 1e-4, False),
        quat_qpos=None, height_index=None, com_velocity=False, healthy=None,
        render=(480, 480),
        root_lo=(-1e3, -1e3, -math.pi), root_hi=(1e3, 1e3, math.pi),
        vel_lo=(-10.0, -10.0, -10.0) + (-20.0,) * 2,
        vel_hi=(10.0, 10.0, 10.0) + (20.0,) * 2,
        bins=(1, 1, 7, 7, 7),
        state_fields=(
            ("x_position", "s[0] -- world x of the front tip in metres; grows as "
             "it swims forward. NOTE this coordinate is absent from this "
             "environment's usual observation and is included here"),
            ("y_position", "s[1] -- world y of the front tip in metres: sideways "
             "drift. Also absent from the usual observation"),
            ("heading", "s[2] -- the front tip's heading angle in radians; 0 "
             "points along +x"),
            ("rotor1_angle", "s[3] -- first rotor joint angle in radians"),
            ("rotor2_angle", "s[4] -- second rotor joint angle in radians"),
            ("x_velocity", "s[5] -- the front tip's x velocity in m/s"),
            ("y_velocity", "s[6] -- the front tip's y velocity in m/s"),
            ("heading_velocity", "s[7] -- heading angular velocity in rad/s"),
            ("rotor1_velocity", "s[8] -- first rotor angular velocity in rad/s"),
            ("rotor2_velocity", "s[9] -- second rotor angular velocity in rad/s"),
        ),
        action_fields=(
            ("rotor1_torque", "a[0] -- torque at the first rotor, in [-1, 1]"),
            ("rotor2_torque", "a[1] -- torque at the second rotor, in [-1, 1]"),
        ),
        helpers=(
            ("forward_velocity(s)", "s[5], the front tip's x velocity in m/s"),
            ("heading(s)", "s[2], the front tip's heading angle in radians"),
            ("joint_angles(s)", "the two rotor joint angles, s[3:5]"),
            ("joint_velocities(s)", "the two rotor angular velocities, s[8:10]"),
            ("control_cost(a)", "sum of squared torques, a measure of effort"),
        ),
        joint_slice=(3, 5), joint_vel_slice=(8, 10),
        symbol_mapping={
            "x_position": "s[0]", "y_position": "s[1]", "heading": "s[2]",
            "x_velocity": "s[5]", "y_velocity": "s[6]",
            "forward_velocity": "s[5]", "velocity": "s[5]",
            "joint_angles": "s[3:5]", "joint_velocities": "s[8:10]",
            "control_cost": "np.sum(np.square(a))",
        },
        prose=(
            "A three-link planar swimmer suspended in a viscous fluid, seen from "
            "above. Two torque-controlled rotor joints connect the three rigid "
            "links, and the fluid's drag is the only thing to push against. The "
            "front tip is free to translate in the plane and to rotate; nothing "
            "can fall over and the episode never ends early."
        ),
    ),
    "Humanoid-v5": dict(
        # REvolve's own MuJoCo environment (`refs/tex/revolve/main.tex:416`, Table
        # `tab:rl_tasks_summary`: "Run upright for 1000 steps"). The SIXTH family,
        # and the first here that is not planar: a FREE root joint, so qpos carries
        # a quaternion and `n_q` (24) exceeds `n_v` (23) -- which is why this row is
        # the first to need `quat_qpos`, `height_index` and `com_velocity`.
        #
        # v5 VS THE PAPER'S v4. The paper reports a 376-D observation; this wheel's
        # Humanoid-v5 emits 348-D, and the difference is not ours. Gymnasium's own
        # changelog (`gymnasium/envs/mujoco/humanoid_v5.py:271-289`) lists, of the
        # changes that touch this task:
        #   * "Excluded the cinert & cvel & cfrc_ext of worldbody and root/freejoint
        #     qfrc_actuator from the observation space, as it was always 0" -- 376 ->
        #     348 exactly: cinert 140 -> 130, cvel 84 -> 78, cfrc_ext 84 -> 78,
        #     qfrc_actuator 23 -> 17 (`env.observation_structure`, measured). Every
        #     dropped entry was a constant zero, so NO INFORMATION was removed and
        #     the paper's 376 and this 348 describe the same state.
        #   * "Fixed bug: healthy_reward was given on every step (even if the Humanoid
        #     was unhealthy)". This one DOES change the shipped reward, and REvolve's
        #     vendored copy has the v4 behaviour
        #     (`refs/code/Revolve/rl_agent/HumanoidEnv.py:388-392`:
        #     `float(self.is_healthy or self._terminate_when_unhealthy)`, which with
        #     termination on is the constant 1). It reaches `reference_reward` here
        #     and not the fitness -- Eq. 3 reads velocity and survival only.
        #   * "Restored contact_cost ... (was removed in v4)". Also reference-reward
        #     only, and NOT negligible: measured, it reaches |2.44|, up to
        #     94% of |reward| on an impact step (mean 2.9-6.5%), so it is computed
        #     rather than dropped -- see `reference_reward`'s "humanoid" arm.
        # NONE of the three changes the observation's information content or the
        # dynamics; the model file is unchanged since `gym==0.21.0` (v3 note). What
        # this row must NOT claim is that a number measured here is comparable to a
        # v4 number on the SHIPPED reward -- the healthy-bonus fix alone moves an
        # episode return by up to 5.0 per unhealthy step. Eq. 3's fitness is
        # unaffected, which is why the spec's ground truth is Eq. 3 and not the reward.
        n_q=24, dt=0.015, action_clip=0.4,
        terminates=True, pin_qpos=(0, 1), goal_disk=None, camera_config=None,
        dr_torso_body="torso",
        ref_reward=("humanoid", 1.25, 0.1, 5.0, 5e-7, 10.0),
        render=(480, 480),
        # `quat_qpos`: qpos[3:7] is a UNIT quaternion and `set_state` does NOT
        # renormalise it (measured: a norm-2.0 quaternion goes in and comes back out
        # at norm 2.0), so `random_state`'s box draw would hand the simulator an
        # invalid orientation. See the key's note in the row-key list above.
        quat_qpos=3,
        # `height_index`: the root's z, which for a free joint is qpos[2] and not the
        # qpos[1] the five planar rows use.
        height_index=2,
        # `com_velocity`: `forward_velocity(s)` must run a forward pass instead of
        # reading `s[n_q]`. Measured -- see `_fit_gated_mean_com_velocity`.
        com_velocity=True,
        # `healthy`: Humanoid-v5's own termination test, `1.0 < qpos[2] < 2.0`
        # (`_healthy_z_range`, strict on both sides), reproduced with 0 mismatches
        # against upstream's `terminated` over 24 steps of a falling episode. NOT
        # Hopper's check: no pitch clause and no +/-100 state-range clause exist here.
        healthy=("z_range", 1.0, 2.0),
        # x and y are cyclic and pinned; z spans the drop; the quaternion components
        # span [-1, 1] and are renormalised after the draw. The 17 hinge joints are
        # all `jnt_limited` so `_bounds` overwrites their +/-pi pad with the model's
        # own ranges -- the free root is unlimited, which is what keeps that loop
        # correct on a 7-qpos joint.
        root_lo=(-1e3, -1e3, 0.0, -1.0, -1.0, -1.0, -1.0),
        root_hi=(1e3, 1e3, 2.5, 1.0, 1.0, 1.0, 1.0),
        # Root linear then root angular (a free joint's dof order), then the 17
        # hinges. Conventions for COVERAGE, deliberately wider than a random policy
        # reaches (measured over 40 random episodes: root linear within +/-3.4,
        # hinges within +/-30) because a trained runner is the regime that matters --
        # REvolve's best reward on this task reached an average velocity of 5.67
        # (main.tex:2457, which states the figure WITHOUT a unit; the "m/s" gloss
        # appears only in a commented-out draft line at :553, so it is not quoted
        # here as the paper's -- see the spec's `normalized.formula`).
        vel_lo=(-10.0, -10.0, -10.0, -10.0, -10.0, -10.0) + (-30.0,) * 17,
        vel_hi=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0) + (30.0,) * 17,
        # x, y pinned to 1; z to 5; the quaternion to 1 each (a four-component
        # orientation has no meaningful axis-aligned grid, and no config on this tier
        # selects `train.backend: tabular` -- `discretise`'s docstring says why the
        # method exists at all); the 17 hinges to 2. n_disc_states = 5 * 2^17 =
        # 655,360, the same order as HalfCheetah's 1.27e6.
        bins=(1, 1, 5, 1, 1, 1, 1) + (2,) * 17,
        state_fields=(
            # --- the free root: 3 positions, 4 quaternion components (qpos[0:7]) ---
            ("x_position", "s[0] -- the torso's world x position in metres, "
             "increasing forwards. Unbounded; it grows without limit as the robot "
             "runs. NOTE this coordinate is absent from this environment's usual "
             "observation and is included here. NOTE ALSO that it is the TORSO's x, "
             "not the whole body's centre of mass -- the two differ by up to 0.10 m "
             "in a 21-step episode, and the forward velocity this task is scored on "
             "is the centre of mass's"),
            ("y_position", "s[1] -- the torso's world y position in metres: "
             "sideways drift, which this task neither rewards nor penalises. Also "
             "absent from the usual observation"),
            ("z_height", "s[2] -- the torso's height in metres; about 1.4 standing, "
             "and the episode ends the moment it leaves the open interval (1.0, "
             "2.0)"),
            ("torso_quat_w", "s[3] -- w component of the torso's orientation "
             "quaternion. s[3:7] is a UNIT quaternion in MuJoCo's (w, x, y, z) "
             "order, NOT scipy's (x, y, z, w)"),
            ("torso_quat_x", "s[4] -- x component of the torso's orientation "
             "quaternion"),
            ("torso_quat_y", "s[5] -- y component of the torso's orientation "
             "quaternion"),
            ("torso_quat_z", "s[6] -- z component of the torso's orientation "
             "quaternion"),
            # --- the 17 hinge joint angles (qpos[7:24]), in JOINT order ---
            ("abdomen_z_angle", "s[7] -- abdomen yaw (twist) angle in radians, in "
             "[-0.785, 0.785]"),
            ("abdomen_y_angle", "s[8] -- abdomen pitch (bend forward/back) angle in "
             "radians, in [-1.309, 0.524]"),
            ("abdomen_x_angle", "s[9] -- abdomen roll (lean sideways) angle in "
             "radians, in [-0.611, 0.611]"),
            ("right_hip_x_angle", "s[10] -- right hip abduction angle in radians, "
             "in [-0.436, 0.087]"),
            ("right_hip_z_angle", "s[11] -- right hip rotation angle in radians, in "
             "[-1.047, 0.611]"),
            ("right_hip_y_angle", "s[12] -- right hip flexion angle in radians, in "
             "[-1.920, 0.349]; the main swing of the right leg"),
            ("right_knee_angle", "s[13] -- right knee angle in radians, in [-2.793, "
             "-0.035]; always negative, 0 would be a hyperextended leg"),
            ("left_hip_x_angle", "s[14] -- left hip abduction angle in radians, in "
             "[-0.436, 0.087]"),
            ("left_hip_z_angle", "s[15] -- left hip rotation angle in radians, in "
             "[-1.047, 0.611]"),
            ("left_hip_y_angle", "s[16] -- left hip flexion angle in radians, in "
             "[-1.920, 0.349]; the main swing of the left leg"),
            ("left_knee_angle", "s[17] -- left knee angle in radians, in [-2.793, "
             "-0.035]"),
            ("right_shoulder1_angle", "s[18] -- first right shoulder axis in "
             "radians, in [-1.484, 1.047]"),
            ("right_shoulder2_angle", "s[19] -- second right shoulder axis in "
             "radians, in [-1.484, 1.047]"),
            ("right_elbow_angle", "s[20] -- right elbow angle in radians, in "
             "[-1.571, 0.873]"),
            ("left_shoulder1_angle", "s[21] -- first left shoulder axis in radians, "
             "in [-1.047, 1.484]"),
            ("left_shoulder2_angle", "s[22] -- second left shoulder axis in "
             "radians, in [-1.047, 1.484]"),
            ("left_elbow_angle", "s[23] -- left elbow angle in radians, in [-1.571, "
             "0.873]"),
            # --- the free root's 6 velocities (qvel[0:6] -> s[24:30]) ---
            ("x_velocity", "s[24] -- the TORSO's forward velocity in m/s. NOT the "
             "quantity this task is scored on and not a usable proxy for it: over a "
             "zero-action episode it correlates -0.39 with the centre of mass's "
             "forward velocity, because the torso counter-rotates against the legs. "
             "Use `forward_velocity(s)`"),
            ("y_velocity", "s[25] -- the torso's sideways velocity in m/s"),
            ("z_velocity", "s[26] -- the torso's vertical velocity in m/s"),
            ("roll_velocity", "s[27] -- the torso's angular velocity about x in "
             "rad/s"),
            ("pitch_velocity", "s[28] -- the torso's angular velocity about y in "
             "rad/s"),
            ("yaw_velocity", "s[29] -- the torso's angular velocity about z in "
             "rad/s"),
            # --- the 17 hinge joint velocities (qvel[6:23] -> s[30:47]) ---
            ("abdomen_z_velocity", "s[30] -- abdomen yaw angular velocity in rad/s"),
            ("abdomen_y_velocity", "s[31] -- abdomen pitch angular velocity in "
             "rad/s"),
            ("abdomen_x_velocity", "s[32] -- abdomen roll angular velocity in "
             "rad/s"),
            ("right_hip_x_velocity", "s[33] -- right hip abduction angular velocity "
             "in rad/s"),
            ("right_hip_z_velocity", "s[34] -- right hip rotation angular velocity "
             "in rad/s"),
            ("right_hip_y_velocity", "s[35] -- right hip flexion angular velocity "
             "in rad/s"),
            ("right_knee_velocity", "s[36] -- right knee angular velocity in rad/s"),
            ("left_hip_x_velocity", "s[37] -- left hip abduction angular velocity "
             "in rad/s"),
            ("left_hip_z_velocity", "s[38] -- left hip rotation angular velocity in "
             "rad/s"),
            ("left_hip_y_velocity", "s[39] -- left hip flexion angular velocity in "
             "rad/s"),
            ("left_knee_velocity", "s[40] -- left knee angular velocity in rad/s"),
            ("right_shoulder1_velocity", "s[41] -- first right shoulder axis "
             "angular velocity in rad/s"),
            ("right_shoulder2_velocity", "s[42] -- second right shoulder axis "
             "angular velocity in rad/s"),
            ("right_elbow_velocity", "s[43] -- right elbow angular velocity in "
             "rad/s"),
            ("left_shoulder1_velocity", "s[44] -- first left shoulder axis angular "
             "velocity in rad/s"),
            ("left_shoulder2_velocity", "s[45] -- second left shoulder axis angular "
             "velocity in rad/s"),
            ("left_elbow_velocity", "s[46] -- left elbow angular velocity in rad/s"),
        ),
        # ACTUATOR ORDER IS NOT JOINT ORDER, and the first two are the ones that
        # catch you: a[0] drives abdomen_y (s[8]) and a[1] drives abdomen_z (s[7]).
        # Read off the model, not transcribed. Each entry is in [-0.4,
        # 0.4] and is multiplied by the actuator's gear ratio to give a torque: 100
        # on the abdomen, hips_x/z and shoulders' neighbours, 300 on hip_y, 200 on
        # the knees, 25 on the arms -- so equal action components are NOT equal
        # torques, and a control cost over the raw action prices a knee and a wrist
        # the same.
        action_fields=(
            ("abdomen_y_torque", "a[0] -- abdomen pitch actuator (drives s[8]), in "
             "[-0.4, 0.4]; gear 100"),
            ("abdomen_z_torque", "a[1] -- abdomen yaw actuator (drives s[7]), in "
             "[-0.4, 0.4]; gear 100"),
            ("abdomen_x_torque", "a[2] -- abdomen roll actuator (drives s[9]), in "
             "[-0.4, 0.4]; gear 100"),
            ("right_hip_x_torque", "a[3] -- right hip abduction, in [-0.4, 0.4]; "
             "gear 100"),
            ("right_hip_z_torque", "a[4] -- right hip rotation, in [-0.4, 0.4]; "
             "gear 100"),
            ("right_hip_y_torque", "a[5] -- right hip flexion, in [-0.4, 0.4]; gear "
             "300, the strongest actuator on the robot"),
            ("right_knee_torque", "a[6] -- right knee, in [-0.4, 0.4]; gear 200"),
            ("left_hip_x_torque", "a[7] -- left hip abduction, in [-0.4, 0.4]; gear "
             "100"),
            ("left_hip_z_torque", "a[8] -- left hip rotation, in [-0.4, 0.4]; gear "
             "100"),
            ("left_hip_y_torque", "a[9] -- left hip flexion, in [-0.4, 0.4]; gear "
             "300"),
            ("left_knee_torque", "a[10] -- left knee, in [-0.4, 0.4]; gear 200"),
            ("right_shoulder1_torque", "a[11] -- first right shoulder axis, in "
             "[-0.4, 0.4]; gear 25"),
            ("right_shoulder2_torque", "a[12] -- second right shoulder axis, in "
             "[-0.4, 0.4]; gear 25"),
            ("right_elbow_torque", "a[13] -- right elbow, in [-0.4, 0.4]; gear 25"),
            ("left_shoulder1_torque", "a[14] -- first left shoulder axis, in [-0.4, "
             "0.4]; gear 25"),
            ("left_shoulder2_torque", "a[15] -- second left shoulder axis, in "
             "[-0.4, 0.4]; gear 25"),
            ("left_elbow_torque", "a[16] -- left elbow, in [-0.4, 0.4]; gear 25"),
        ),
        helpers=(
            ("forward_velocity(s)", "the forward (x) velocity of the whole body's "
             "CENTRE OF MASS in m/s -- the quantity this task is scored on. It is "
             "NOT s[24], which is the torso's own x velocity and correlates -0.39 "
             "with this one"),
            ("torso_height(s)", "s[2], the torso's height in metres"),
            ("is_upright(s)", "1.0 while the robot is up (torso height strictly "
             "inside (1.0, 2.0)); the episode ends the step this becomes 0.0"),
            ("joint_angles(s)", "the seventeen hinge joint angles, s[7:24]"),
            ("joint_velocities(s)", "the seventeen hinge joint velocities, "
             "s[30:47]"),
            ("control_cost(a)", "sum of squared action components, a measure of "
             "effort. Note the gear ratios above: this prices every actuator alike"),
        ),
        joint_slice=(7, 24), joint_vel_slice=(30, 47),
        # NO `forward_velocity`/`velocity`/`speed` ENTRY, deliberately, and it is the
        # one row where that is a correctness matter rather than an omission: on this
        # body the forward velocity the task scores is not any `s[i]` (see
        # `_fit_gated_mean_com_velocity`), so a `"forward_velocity": "s[24]"` row
        # would silently rewrite a candidate's correct symbol into a quantity
        # measured at -0.39 correlation with the one it named. A symbol whose value
        # is not a state slice gets no slice.
        symbol_mapping={
            "x_position": "s[0]", "y_position": "s[1]",
            "z_height": "s[2]", "height": "s[2]", "torso_height": "s[2]",
            "torso_quat": "s[3:7]", "orientation": "s[3:7]",
            "joint_angles": "s[7:24]", "joint_velocities": "s[30:47]",
            "control_cost": "np.sum(np.square(a))", "torques": "a",
        },
        prose=(
            "A three-dimensional bipedal humanoid -- a torso with a twisting "
            "abdomen, two legs (hip and knee) and two arms (shoulder and elbow), "
            "seventeen torque-controlled joints in total -- stands on a flat, "
            "endless floor. Its root is completely free: it can translate in any "
            "direction, rise and fall, and rotate to any orientation, so balance is "
            "something the robot has to produce rather than something the "
            "environment provides. It moves only by pushing against the floor with "
            "its feet. The episode ends the moment the torso leaves the height band "
            "between 1.0 and 2.0 metres -- that is, the moment it falls over or "
            "leaves the ground -- so staying upright is a precondition for doing "
            "anything at all."
        ),
    ),
}


class GymMujoco(EnvAdapter):
    """One gymnasium/MuJoCo task, selected by the `task` constructor argument.

    One class and a five-row family table, not ten subclasses: everything that
    differs between the tasks is data (which env to build, the horizon, the prose,
    the ground-truth fitness), and a subclass per task would be ten places for the
    restore logic to drift.

    The observation is exactly `concatenate([data.qpos, data.qvel])`, so `_step` is
    a pure function of the state passed in: restore, zero the solver warmstart,
    step. On the families with a terminal state (Hopper, InvertedPendulum) the step
    runs the environment's own `step` so upstream's termination logic executes
    verbatim; the other three have no terminal state to run and step the physics
    directly.
    """

    #: MUST stay None: `_sb3_run` derives `continuous` solely from
    #: `exact_states is None`, and an integer here would hand SAC a Discrete
    #: space over the coarse proxy action set.
    exact_states = None

    #: The one entry is the schema's INERT DEFAULT, not a capability claim: none
    #: of the ten metrics is a reduction of a per-step success check -- they are
    #: metres, m/s and episode-level fractions -- so no reduction value moves
    #: anything here. Coherence validation reads this tuple as "the values a
    #: config may carry for this env": listing
    #: only `per_step_fraction` keeps every config's untouched default valid
    #: while `any_step_episode_fraction` is refused at validation. Stated here
    #: so nobody reads a resolved artifact's `reduction:` line as a claim about
    #: these metrics.
    supported_reductions: Tuple[str, ...] = ("per_step_fraction",)

    #: The specs define success as 0.9 of the random->expert span
    #: (`continuous_success.normalized.success_threshold`). Eight specs carry a
    #: measured expert anchor, but the span is STILL not computable here: the
    #: pair is flat and unlabelled so `baselines_of` refuses it, and the `random`
    #: half of six of them was measured under a different mujoco than the one
    #: BIRD pins (see the module docstring). A bar derived from
    #: that span would be a fabricated pin. Kept as documentation of that
    #: reasoning; `success()` below is overridden rather than left to threshold
    #: against this, because `task_metric >= inf` is False for every real metric
    #: but TRUE for an inf a diverged trajectory could produce -- a divergence
    #: must not launder into `success_rate = 1.0`.
    success_threshold = float("inf")

    # -- construction --------------------------------------------------------

    def __init__(self, task: str, spec: Optional[TaskSpec] = None) -> None:
        # `spec` may be passed explicitly; the default None resolves the task
        # from the catalogue (`_gym_specs()`, the `library: mujoco` slice).
        if spec is None:
            specs = _gym_specs()
            if task not in specs:
                raise KeyError(
                    f"unknown gymnasium/MuJoCo task {task!r}; the catalogue holds "
                    f"{sorted(specs)}")
            spec = specs[task]
        family = _FAMILIES[spec.env_id]

        self.task = task
        # The spec's OWN rule (`tasks._BIRD_ID_RULES`), not a second copy of it
        # here. On a `library: mujoco` spec `bird_env_id` IS `_env_id(task)` --
        # the rule is `"gym_" + id`. `_env_id` stays as the fallback for a spec
        # whose library has no rule.
        self.name = spec.bird_env_id or _env_id(task)
        self._family = family
        self._gym_id = spec.env_id
        # Per TASK, from the spec -- not `family["ref_reward"]` directly. See
        # `_reference_reward_shape` and `reference_reward`.
        self._ref_reward = _reference_reward_shape(spec, family)
        self.horizon = int(spec.env["horizon"])
        self.n_q = int(family["n_q"])
        self.dt = float(family["dt"])
        self.action_clip = float(family["action_clip"])
        self.render_width, self.render_height = family["render"]
        self._bins = tuple(family["bins"])
        self.n_disc_states = int(np.prod(self._bins))

        raw = spec.continuous_success["raw"]
        fitness = _FITNESSES.get(str(raw.get("name")))
        if fitness is None:
            raise TaskSpecError(
                f"{spec.path}: continuous_success.raw.name {raw.get('name')!r} is "
                "not in gym_mujoco's _FITNESSES allow-list. The spec names the "
                "measure; this module must implement it before the task is runnable")
        self._fitness = fitness

        # The instruction (the spec's `natural_language`, which on these gym specs
        # is written at instruction granularity) reaches prompts through
        # `problem.task_description` inheritance (`TaskSpec.instruction`), NOT
        # through `_prose` -- `_prose` is the ENVIRONMENT paragraph, and putting the
        # instruction in it would hand the task to methods whose `env_spec`
        # deliberately withholds environment text. Same split
        # `description.env_prose` / `l_task` draws.
        self._prose = str(family["prose"])
        self._state_fields = tuple(family["state_fields"])
        self._action_fields = tuple(family["action_fields"])
        self._helpers = tuple(family["helpers"])
        self.symbol_mapping = dict(family["symbol_mapping"])

        # The dependency probe comes FIRST, before anything touches the process.
        # `MUJOCO_GL` is process-global and the Triton preload changes library load order, so
        # a missing-extra failure must have no side effect beyond the raised
        # error -- a later construction, or the debugging session reading the
        # traceback, would otherwise inherit a GL choice made for an adapter that
        # never existed. `find_spec` rather than a try-import so the probe itself
        # loads nothing; the ValueError arm covers a module poisoned to None in
        # `sys.modules`, which is how the no-simulator test simulates absence.
        import importlib.util
        missing = []
        for mod in ("gymnasium", "mujoco"):
            try:
                if importlib.util.find_spec(mod) is None:
                    missing.append(mod)
            except (ImportError, ValueError):
                missing.append(mod)
        if missing:
            raise ImportError(
                f"problem.env_id: {self.name} needs gymnasium's MuJoCo extra, "
                "which is not installed. Either\n"
                "    uv sync --extra metaworld\n"
                "    (or: pip install 'gymnasium[mujoco]>=1.1' 'mujoco==3.3.0')\n"
                "or run the same method point on an env that needs no simulator:\n"
                "    problem.env_id: pendulum\n"
                f"(missing: {', '.join(missing)}; gymnasium and mujoco are both "
                "required)")

        chosen = os.environ.get("MUJOCO_GL") or _default_mujoco_gl()
        if chosen:
            os.environ["MUJOCO_GL"] = chosen
        _preload_llvm_before_mujoco()

        import gymnasium as gym
        import mujoco
        self._mj = mujoco

        # `camera_config` is a ROW KEY, not a branch on the env id, so a sixth
        # family cannot silently inherit the wrong camera. InvertedPendulum ships
        # no model camera, and gymnasium's default free camera makes
        # `render(state)` impure (its `lookat` is captured once and never follows
        # `set_state`) and frames the cart out of the picture at the rail ends --
        # both measured, and recorded beside the pinned config its row reuses
        # (`mujoco_control.DEFAULT_CAMERA_CONFIG`). The other four render pure
        # without help, for two DIFFERENT reasons worth keeping apart:
        # cheetah/hopper/swimmer have `track` (trackcom) cameras IN THE MODEL,
        # whose pose is derived from `data` on every frame; reacher's is
        # gymnasium's default free camera,
        # which is pure because it is fixed at construction and never re-derived
        # (a FREE camera ignores the `trackbodyid` its config carries).
        kwargs: Dict[str, Any] = {}
        if family["camera_config"] is not None:
            kwargs["default_camera_config"] = family["camera_config"]
        self._env = gym.make(
            spec.env_id, render_mode="rgb_array",
            width=self.render_width, height=self.render_height, **kwargs,
        ).unwrapped

        # The row's model claims, asserted against the live wheel -- `n_q` because
        # every state index depends on it, `dt` because two fitnesses divide by it
        # (a wheel that changed a timestep would otherwise silently rescale
        # `neg_speed_error` and `heading_speed` while everything stayed green),
        # and `n_v` against the velocity-bound vectors so a short row fails here
        # rather than as a distant shape error. `frame_skip` is read off the env
        # rather than pinned: `do_simulation` below needs it, and a pin nothing
        # validates is how a sixth family would invent a wrong one.
        nq, nv = int(self._env.model.nq), int(self._env.model.nv)
        if nq != self.n_q or abs(float(self._env.dt) - self.dt) > 1e-12:
            raise RuntimeError(
                f"{self.name}: family row says n_q={self.n_q} dt={self.dt}, model "
                f"says nq={nq} dt={float(self._env.dt)}. The wheel changed "
                "underneath this adapter; every state index and both dt-scaled "
                "fitnesses are now suspect.")
        if nv != len(family["vel_lo"]) or nv != len(family["vel_hi"]):
            raise RuntimeError(
                f"{self.name}: model nv={nv} but the family row carries "
                f"{len(family['vel_lo'])} velocity bounds")
        self.n_v = nv
        self.obs_dim = nq + nv
        self.action_dim = int(self._env.model.nu)
        self.frame_skip = int(self._env.frame_skip)
        self._env.reset(seed=0)

        # The DR contract. The SPEC declares the axes
        # (`domain_randomization`, read by `dr_of`), this module says what each writes
        # (`_GYM_AXES`), and the two are held together here rather than in a test: a spec
        # naming an axis this adapter cannot write would otherwise be an axis `set_dr`
        # drops silently.
        # Rebound on the INSTANCE: the class-level dicts are shared by all ten envs.
        ranges, nominal = dr_of(spec)
        unknown = sorted(set(ranges) - set(_GYM_AXES))
        if unknown:
            raise TaskSpecError(
                f"{spec.path}: domain_randomization names axes {unknown} that "
                f"bird/envs/gym_mujoco.py cannot write; it knows {sorted(_GYM_AXES)}")
        self.dr_parameters = dict(ranges)
        self._dr_nominal = dict(nominal)
        model = self._env.model
        # Nominal physics, captured once. `_apply_dr` always writes scale x nominal,
        # never scale x current, so DR draws cannot compound across episodes.
        self._nominal = {f: np.copy(getattr(model, f)) for f in _DR_MODEL_FIELDS}
        self._nominal["gravity"] = np.copy(model.opt.gravity)
        self._nominal["viscosity"] = float(model.opt.viscosity)
        torso = family.get("dr_torso_body")
        self._torso_body = (int(self._mj.mj_name2id(model, self._mj.mjtObj.mjOBJ_BODY, torso))
                            if torso else -1)
        if torso and self._torso_body < 0:
            raise RuntimeError(f"{self.name}: family row names torso body {torso!r}, "
                               "which the model does not have")

        super().__init__()

    # -- subclass hooks ------------------------------------------------------

    def _build_action_set(self) -> Any:
        """A coarse finite set for `sample_transitions`' coverage: one joint at a
        time at +/-clip, plus all-zero and both all-on corners, DEDUPLICATED --
        at action_dim 1 the corners coincide with the per-joint rows, and the
        duplicate rows would weight |a| = clip at 4/5 instead of 2/3 in the
        screens' uniform draw while advertising N_ACTIONS = 5 for a set of three.
        Continuous control additionally accepts anything inside
        [action_low, action_high]."""
        n, c = self.action_dim, self.action_clip
        acts = [[0.0] * n, [c] * n, [-c] * n]
        for i in range(n):
            for v in (c, -c):
                a = [0.0] * n
                a[i] = v
                acts.append(a)
        unique: List[List[float]] = []
        for a in acts:
            if a not in unique:
                unique.append(a)
        return unique

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Box bounds for the sb3 observation space and `random_state` coverage.

        Conventions first (`root_lo` covers the unactuated root; +/-pi stands in
        for every joint after it), then the model overrides each LIMITED joint
        with its own `jnt_range`.

        The loop is safe on a multi-DoF root, and the reason is worth stating:
        Humanoid's root is a FREE joint occupying qpos[0:7], so `jnt_range` would describe seven
        coordinates and `lo[adr], hi[adr] = ...` would write only the first. It never
        runs on that joint -- a free joint carries `jnt_limited == 0` (asserted on
        the live model) -- so the pad `root_lo`/`root_hi` supplies is what stands,
        which is exactly what a row is for. Every LIMITED joint in all six models is
        1-DoF (slide or hinge), and that is the property the indexing needs.
        """
        model = self._env.model
        fam = self._family
        pad = self.n_q - len(fam["root_lo"])
        lo = list(fam["root_lo"]) + [-math.pi] * pad
        hi = list(fam["root_hi"]) + [math.pi] * pad
        for j in range(int(model.njnt)):
            if int(model.jnt_limited[j]):
                adr = int(model.jnt_qposadr[j])
                lo[adr], hi[adr] = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
        lo += list(fam["vel_lo"])
        hi += list(fam["vel_hi"])
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)

    # -- domain randomisation -----------------------------------------------------

    def _apply_dr(self) -> None:
        """Write the current nominal-or-draw scales into mjModel, from the captured
        nominal. Called at the top of `_reset`, so every episode starts on the physics
        `_dr_now` names and nothing compounds. An axis the spec does not declare is
        never in `_dr_now`, so `.get(axis, 1.0)` is the identity for it."""
        draw = self._dr_now or {}
        model, nominal = self._env.model, self._nominal
        for f in _DR_MODEL_FIELDS:
            getattr(model, f)[:] = nominal[f]
        mass = float(draw.get("body_mass_scale", 1.0))
        if mass != 1.0:
            model.body_mass[1:] = nominal["body_mass"][1:] * mass
            model.body_inertia[1:] = nominal["body_inertia"][1:] * mass
        torso = float(draw.get("torso_mass_scale", 1.0))
        if torso != 1.0 and self._torso_body >= 0:
            model.body_mass[self._torso_body] *= torso
            model.body_inertia[self._torso_body] *= torso
        friction = float(draw.get("contact_friction_scale", 1.0))
        if friction != 1.0:
            model.geom_friction[:, 0] = nominal["geom_friction"][:, 0] * friction
        model.opt.gravity[:] = nominal["gravity"] * float(draw.get("gravity_scale", 1.0))
        motor = float(draw.get("motor_strength_scale", 1.0))
        if motor != 1.0:
            model.actuator_gear[:, 0] = nominal["actuator_gear"][:, 0] * motor
        damping = float(draw.get("joint_damping_scale", 1.0))
        if damping != 1.0:
            model.dof_damping[:] = nominal["dof_damping"] * damping
        model.opt.viscosity = nominal["viscosity"] * float(draw.get("viscosity_scale", 1.0))

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        """The environment's OWN initial distribution, seeded from the RNG this
        adapter is handed.

        Deliberately `env.reset(seed=...)` rather than a per-family transcription
        of five `reset_model` bodies: Reacher rejection-samples its goal inside a
        0.2 m disk, Hopper draws uniform noise where HalfCheetah draws Gaussian
        `qvel` noise, and five transcriptions are five chances to drift from the
        wheel. Deriving the seed from `rng` keeps the episode reproducible from
        `ctx.rng` alone, which is the property the transcriptions bought elsewhere.
        """
        self._apply_dr()
        self._env.reset(seed=int(rng.integers(2 ** 31 - 1)))
        return np.concatenate([self._env.data.qpos, self._env.data.qvel]).ravel().copy()

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Stateless: restore `s`, step, read the arriving state.

        The `qacc_warmstart` zeroing is required for purity whenever MuJoCo's
        constraint solver runs -- the measured deviations per env are in the module
        docstring. Measure with SHUFFLED replay, never round-trips.

        Two step paths, chosen by the row's `terminates` key. Where the env HAS a
        terminal state (Hopper, InvertedPendulum), `self._env.step(u)` runs so
        upstream's termination logic executes verbatim and cannot drift; `done` is
        its `terminated` plus the non-finite guard every adapter carries. Where it
        has none (HalfCheetah, Reacher, Swimmer), `do_simulation` steps the physics
        directly and skips the native observation build, reward computation and
        info allocation `env.step` would discard -- there is no termination logic
        for the anti-drift argument to protect there. `truncated` never applies:
        the env is unwrapped and the harness owns the horizon.

        `info["success"]` is False on every step, and that is the spec speaking
        rather than a stub: all ten carry `discrete_success.kind:
        continuous_only` -- gymnasium ships no success check for these tasks,
        so there is no per-step flag to report and inventing one would put a
        number in the artifact that nothing published defines.
        """
        s = np.asarray(s, dtype=float).ravel()
        u = np.clip(np.asarray(a, dtype=float).ravel(),
                    -self.action_clip, self.action_clip)
        self._env.set_state(s[:self.n_q], s[self.n_q:])
        self._env.data.qacc_warmstart[:] = 0.0
        if self._family["terminates"]:
            _obs, _r, terminated, _truncated, _info = self._env.step(u)
        else:
            self._env.do_simulation(u, self.frame_skip)
            terminated = False
        s2 = np.concatenate([self._env.data.qpos, self._env.data.qvel]).ravel().copy()
        done = bool(terminated) or not bool(np.isfinite(s2).all())
        return s2, done, {"success": False}

    def discretise(self, s: np.ndarray) -> int:
        """qpos-only binning; a binning, not a bijection, hence `exact_states is
        None`. Present because `EnvAdapter` requires it -- no config on this tier
        selects `train.backend: tabular`."""
        s = np.asarray(s, dtype=float).ravel()
        lo, hi = self.obs_low, self.obs_high
        idx = 0
        for i, n in enumerate(self._bins):
            idx = idx * n + (0 if n == 1 else _bin(float(s[i]), float(lo[i]), float(hi[i]), n))
        return idx

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        """Coverage over the box, not the initial distribution -- shaped by two
        row keys so a sixth family cannot silently miss either. `pin_qpos` holds
        cyclic root coordinates at 0 (they provably cannot change a reward
        difference -- measured on HalfCheetah-v5, a 37.5 m shift
        moves the stepped result by 6.2e-13; sampling them would spend the
        screens' coverage on coordinates that carry none). `goal_disk` redraws a
        goal UNIFORMLY IN THE DISK the env itself samples from (r = R*sqrt(u)):
        the box bounds for Reacher's target come from the model's +/-0.27
        `jnt_range`, and projecting box draws back onto the r=0.2 boundary would
        concentrate 57% of the mass exactly on the rim, where
        `dwell_fraction`'s threshold sits; the env rejection-samples
        uniformly INSIDE (`reacher_v5.py::reset_model`), and the screens should
        judge rewards under that distribution, not a rim-weighted one. The
        target's zero velocities come from the row's `vel_lo`/`vel_hi`, their one
        home."""
        s = np.asarray(rng.uniform(self.obs_low, self.obs_high), dtype=float)
        for i in self._family["pin_qpos"]:
            s[i] = 0.0
        # A unit quaternion cannot be drawn componentwise from a box: measured,
        # a box draw of norm 1.44 survives `set_state` at norm 1.44 --
        # MuJoCo does not renormalise `data.qpos` -- so without this the screens'
        # coverage would be taken over orientations that are not rotations.
        # Normalising the draw gives a direction uniform on the 3-sphere's box
        # projection rather than Haar-uniform on SO(3); that is the same kind of
        # "coverage over the box" every other row here samples, not a claim about
        # the rotation group. The zero-norm draw is impossible in floating point but
        # guarded anyway, since the fallback (identity) is the only defined answer.
        qi = self._family["quat_qpos"]
        if qi is not None:
            q = np.asarray(s[qi:qi + 4], dtype=float)
            norm = float(np.linalg.norm(q))
            s[qi:qi + 4] = (q / norm) if norm > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])
        disk = self._family["goal_disk"]
        if disk is not None:
            start, radius = disk
            r = radius * math.sqrt(float(rng.uniform(0.0, 1.0)))
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            s[start] = r * math.cos(theta)
            s[start + 1] = r * math.sin(theta)
        return s

    # -- ground truth --------------------------------------------------------

    def task_metric(self, traj: Any) -> float:
        """The spec's own continuous fitness (`continuous_success.raw`), RAW --
        or `nan` when there is nothing to measure.

        Raw means unnormalised: metres for the displacement tasks, m/s for the
        speed tasks, fractions for balance and dwell -- NOT the [0, 1] the base
        class's docstring describes, and not clipped into it. The specs normalise
        through anchors, and `baselines_of` refuses the flat pair (see the module
        docstring) even though eight of the ten carry a measured expert -- so the
        honest number is the raw one, negative values included: a cheetah that drifts backwards
        scores negative metres here where a clipped metric would call it
        indistinguishable from standing still.

        `nan`, NEVER 0.0, for a trajectory that cannot be measured -- too few
        rows, too narrow, or carrying a non-finite value (a diverged episode's
        appended final state). 0.0 sits INSIDE these metrics' ranges and on the
        signed ones it is a CEILING: a perfect cruise on `neg_speed_error`, on
        target on `neg_final_reach_distance`, and above the measured random
        anchor (~ -4.58 m) on the displacement tasks -- so a malformed record
        returning it would outrank every honest episode with `failure_kind`
        empty. `nan` is "no measurement": the seed aggregations drop it, and a
        candidate with nothing finite lands on `select.failure_value`, losing
        every comparison while staying recorded -- `evaluation._failure_value`'s
        contract for exactly this trap. The same guard is why a diverged
        trajectory cannot launder an `inf` metric into the curves, the pruner or
        `success()`.
        """
        states = _states_of(traj)
        if (states.shape[0] < 2 or states.shape[1] < self.obs_dim
                or not bool(np.isfinite(states).all())):
            return float("nan")
        value = float(self._fitness(self, states))
        return value if math.isfinite(value) else float("nan")

    def success(self, traj: Any) -> bool:
        """False, always: no usable random->expert span exists, so no success bar
        does either (see `success_threshold` -- eight specs carry a measured expert,
        but the flat pair is still refused). Overridden rather than thresholded
        because the base `task_metric >= inf` comparison is False for every real
        metric but True for an `inf` a diverged episode could produce -- a
        divergence must not become `success_rate = 1.0`.

        Three named costs, so none of them surprises an artifact reader:
        `verify.quality_screen: tpe` (CARD) fails open on a store with no
        successes; `problem.fitness_access: success_indicator` reads a constant;
        and `evaluate.fitness.source: success_rate` (LIMEN's pin) would collapse
        every candidate to fitness 0.0 -- an all-tie in which map-elites' strict
        `>` lets the first occupant of every cell keep it forever -- which is why
        `_check_coherence` REFUSES that pairing outright rather than letting the
        tie be discovered in a finished sweep."""
        return False

    @property
    def has_reference_reward(self) -> bool:
        """False on the five `reward.human.kind: none` tasks; see `reference_reward`."""
        return self._ref_reward is not None

    # --- BEGIN reference reward (ground truth; strip before showing an LLM) ---
    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """The environment's own published reward, transcribed from the installed
        wheel (gymnasium 1.3.0) -- ON THE TASKS WHOSE SPEC CLAIMS IT, and
        `NotImplementedError` on the tasks whose spec disowns it.

        Which is which is the spec's `reward.human.kind`, read once at construction
        (`_reference_reward_shape`). The five base tasks (`half_cheetah`,
        `hopper_hop`, `swimmer_forward`, `reacher_reach`,
        `inverted_pendulum_balance`) declare `published_dense` and pin the wheel's
        own reward method by file and whole-file sha256; they get the family row. The
        five derived objectives (`half_cheetah_backward`, `half_cheetah_target_speed`,
        `hopper_hop_in_place`, `swimmer_heading`, `reacher_hold`) declare `none` --
        "the env's built-in reward pays forward velocity, the exact thing this task
        penalises" -- and get nothing: `has_reference_reward` is False and this
        raises, which `training._rollout` turns into `gt_return = None`,
        `policy_api.run_episode` into `reference_return: None`, and
        `scripts/train_expert.py` into a refusal. No per-task shape is invented in
        their place -- the backward spec names negated `vx` as "our choice", and a
        reference this repo chose would be BIRD's reward wearing the benchmark's
        label. Returning the FAMILY row regardless of the spec would make a
        backward run's `gt_reward_curve` measure forward speed and `pearson_curve`
        on those tasks correlate candidates against an objective the task
        penalises.

        Per family: HalfCheetah `vx - 0.1 |a|^2`; Hopper `healthy + vx - 1e-3
        |a|^2` (the alive term paid only while healthy, per the v5 fix); Swimmer
        `vx - 1e-4 |a|^2`; Reacher `-|fingertip - target| - |a|^2` (both weights
        1.0 -- the wheel's docstring table says 0.1 for the control term but the v5
        signature says 1.0, and the signature is what executes); InvertedPendulum
        `+1` while the pole is up, which is informative here precisely because this
        adapter keeps the terminal state -- contrast `mujoco_control`'s variant,
        where removing the terminal turned it into a constant.

        The velocity term reads the instantaneous `qvel` where the env uses its
        per-step finite difference -- the same declared approximation as
        `mujoco_control.HalfCheetah.reference_reward`, for the same reason: a
        (state, action) reward cannot see the next state.

        NO `gt_reward` ALIAS; the analysis in `mujoco_control.py` applies verbatim.
        The consumer gate for this tier is `_CONSUMER_FORBIDDEN` at module top,
        inherited through the factory; an overlay may still author its own list.
        """
        kind = self._ref_reward
        if kind is None:
            raise NotImplementedError(
                f"{self.name}: no reference reward. tasks/{self.task}/shared_spec.yaml "
                "declares `reward.human.kind: none` -- the built-in reward of "
                f"{self._gym_id} is not this task's objective, and this adapter "
                "invents none in its place. Consumers record absence (gt_return None), "
                "not 0.0.")
        s = np.asarray(s, dtype=float).ravel()
        u = None if a is None else np.clip(
            np.asarray(a, dtype=float).ravel(), -self.action_clip, self.action_clip)
        effort = 0.0 if u is None else float(np.sum(np.square(u)))
        if kind[0] == "velocity":
            _tag, w_fwd, w_ctrl, healthy_bonus = kind
            r = w_fwd * float(s[self.n_q]) - w_ctrl * effort
            if healthy_bonus:
                r += self.is_healthy(s)
            return float(r)
        if kind[0] == "humanoid":
            # The shipped Humanoid-v5 reward: the healthy and control terms exactly,
            # and TWO declared approximations. The wheel's reward method is NOT
            # named here, and the omission is deliberate rather than untidy: this
            # method's source is rendered verbatim into EVERY gym task's
            # `full_source`, so a name written here appears in eleven renders --
            # including the pendulum's, whose env has no such method at all (its
            # reward is a line inside `step`). `test_the_strip_removes_both_copies_
            # of_the_reward` asserts exactly that. The per-task citation lives
            # in each spec's `reward.human.reference`, which is where a reader
            # looking for it will be.
            #
            # The forward term is the approximation every row above declares, but it
            # costs more here and so is measured rather than asserted: the env
            # differences the mass centre across the step and a (state, action)
            # reward cannot see the previous state, so `com_vx` supplies the
            # INSTANTANEOUS centre-of-mass velocity via `mj_subtreeVel`. Measured
            # against the env's own finite difference (this wheel): max
            # |diff| 1.8e-2 m/s and corr 0.9997 on a zero-action episode, 5.8e-2 and
            # 0.9927 on a random one. The obvious cheaper reading, `s[self.n_q]`, was
            # REJECTED on measurement rather than on taste -- it correlates -0.39
            # with the env's value, so it is not a worse approximation but a wrong
            # one.
            #
            # THE CONTACT TERM IS ALSO AN APPROXIMATION. `mj_forward` reaches
            # `cfrc_ext` only through the acceleration-stage sensor path and this
            # model has `nsensor == 0`, so `contact_cost` calls
            # `mj_rnePostConstraint` to compute it; without that call it would read
            # whatever the buffer last held. Comparing against the wheel's own
            # `info["reward_contact"]` on the same `self._env` is CIRCULAR -- both
            # read the single buffer the wheel's `step` just filled -- so agreement
            # there proves nothing. Computed from the stored state, the term differs
            # for the same reason the velocity term does (post-step buffer against
            # re-forwarded arrived pose), and the residual is measured rather than
            # asserted: max |diff| 1.7e-03 over the zero-action
            # trajectory and 3.0e-03 over a random one, mean 1.9e-04 on both, against
            # a live range reaching 2.62.
            # It is still not droppable -- v5 restored it (v4 had none, and v4's was
            # always 0 anyway per the wheel's own bug note) and it reaches |2.62|, up
            # to 94% of |reward| on an impact step.
            #
            # WHOLE-REWARD AGREEMENT, which is the number a consumer actually wants:
            # max |diff| 2.1e-02 (zero action) and 7.6e-02 (random) against a live
            # range of 0.45 to 5.50, i.e. ~1.4% worst case, dominated by the velocity
            # term rather than by this one.
            #
            # The healthy bonus follows THIS wheel, i.e. the v5 fix: paid only while
            # upright. REvolve's vendored copy pays it unconditionally
            # (`HumanoidEnv.py:388-392`); the spec records that † and nothing here
            # reproduces the bug, because this method's contract is "the reward the
            # installed wheel pays", not "the reward REvolve's fork paid".
            _tag, w_fwd, w_ctrl, healthy_r, contact_w, contact_cap = kind
            return float(w_fwd * self.com_vx(s)
                         - w_ctrl * effort
                         + healthy_r * self.is_upright(s)
                         - self.contact_cost(s, u, contact_w, contact_cap))
        if kind[0] == "reacher":
            return float(-self.to_target_dist(s) - effort)
        # "alive": InvertedPendulum-v5's `int(not terminated)`.
        return float(self.is_balanced(s))
    # --- END reference reward ---

    # -- helpers advertised in `_helpers`; these must stay real methods -------
    # (`state_action_api_stub` and `pythonic_class_abstraction` advertise them to
    # the generator, and an advertised method that does not exist is a fabricated
    # pin: `verification._SelfProxy` forwards a candidate's `self.<name>` here, so a
    # missing one fails the candidate for using the API its prompt gave it.)

    def forward_velocity(self, s: Any) -> float:
        """The forward (x) velocity in m/s -- WHOSE forward velocity is the row's
        `com_velocity` key, and the two are different quantities rather than two
        precisions of one.

        False (the five planar rows): the root's own first qvel entry, which is what
        those envs difference for their reward and their fitness. True (Humanoid):
        the whole body's centre of mass, because that is what `humanoid_v5.step` and
        REvolve's own fork difference -- and the root's entry correlates **-0.39**
        with it there. A single unconditional `s[self.n_q]` is correct for five
        families and would score the sixth on a signal of the opposite sign.
        """
        if self._family["com_velocity"]:
            return self.com_vx(s)
        return float(np.asarray(s, dtype=float).ravel()[self.n_q])

    def com_x(self, s: Any) -> float:
        """The whole body's centre-of-mass x in metres, re-derived from a stored
        state by a forward pass.

        `mass_center` in both gymnasium (`humanoid_v5.py:17`) and REvolve's fork
        (`HumanoidEnv.py:58`) is the body-mass-weighted mean of `data.xipos`, which
        `mj_forward` computes from qpos alone -- so this is a pure function of the
        state even though it needs the simulator to evaluate. Restoring into
        `self._env` is safe here for the same reason `_step` is stateless: `_step`
        re-restores qpos/qvel and zeroes `qacc_warmstart` before every step, and
        that zeroing is exactly what makes an intervening call unable to move the
        next one (measured: shuffled replay deviation 0.000e+00 zeroed against
        7.520e-07 left alone, the largest unzeroed error in this family by four
        orders of magnitude -- see the module docstring).

        NOT `info["x_position"]`, which in v5 is qpos-based rather than
        `xipos`-based (the wheel's v5 notes record that fix) and differs from this
        by up to 0.10 m over 21 steps.
        """
        s = np.asarray(s, dtype=float).ravel()
        self._env.set_state(s[:self.n_q], s[self.n_q:])
        self._mj.mj_forward(self._env.model, self._env.data)
        mass = self._env.model.body_mass
        return float(np.einsum("b,bj->j", mass, self._env.data.xipos)[0] / mass.sum())

    def com_vx(self, s: Any) -> float:
        """The centre of mass's INSTANTANEOUS forward velocity in m/s
        (`mj_subtreeVel`'s whole-model subtree linear velocity).

        The instantaneous analogue of what the env differences across a step; see
        `reference_reward`'s "humanoid" arm for the measured agreement and for why
        the cheap reading was rejected. `mj_subtreeVel` needs `mj_forward` first,
        which `com_x` has already done on the same state.
        """
        self.com_x(s)
        self._mj.mj_subtreeVel(self._env.model, self._env.data)
        return float(self._env.data.subtree_linvel[0][0])

    def contact_cost(self, s: Any, a: Any, weight: float, cap: float) -> float:
        """`min(weight * sum(cfrc_ext**2), cap)` at this (state, action) -- the
        shipped contact penalty, to within the re-forwarding residual below.

        `mj_rnePostConstraint` IS REQUIRED. `cfrc_ext` is computed in the
        ACCELERATION stage and `mj_forward` only reaches it through the
        acceleration-stage sensor path -- `Humanoid-v5` declares `nsensor == 0`, so
        `mj_forward` does not touch the array at all. Measured by poisoning it:
        `cfrc_ext[:] = nan` before `mj_forward` comes back `nan`, and `= 1e6` comes
        back capped at 10.0. So without this call the method would read whatever the
        buffer last held and be a function of call HISTORY rather than of `(s, a)` --
        the same defect the `ctrl` write guards against, in the bigger of the two
        instances.

        THE TERM IS AN APPROXIMATION, NOT EXACT. Comparing this method's reading
        against the wheel's own `info["reward_contact"]` on the same `self._env` is
        circular -- both read the one buffer the wheel's `step` just filled -- so
        agreement there is no evidence of exactness. The contact term is an
        approximation for exactly the reason the forward term is: `mj_step` leaves
        `cfrc_ext` describing the pose from
        before the last integration substep, while this re-forwards from the stored
        qpos and describes the ARRIVED pose (`deepmind/mujoco#889`, the same note
        gymnasium's v5 changelog cites for `info["x_position"]`). Residual measured
        over a 40-step falling trajectory: see `reference_reward`'s "humanoid" arm.

        The cap is the upper end of v5's `contact_cost_range`; the lower end is `-inf`
        and so cannot bind on a sum of squares. Takes the weight and cap as arguments
        rather than reading the row, so the one place they are written is the row's
        `ref_reward` tuple.

        `ctrl` IS WRITTEN UNCONDITIONALLY, zeros for `a is None`, and that is a
        purity requirement rather than tidiness: `cfrc_ext` is a solver output and
        depends on the actuator forces, so leaving `data.ctrl` at whatever the
        previous call happened to put there would make this a function of call
        HISTORY -- the same `(s, None)` returning different numbers depending on what
        ran before it. Zeros is the right stand-in because it is the same convention
        the effort term already uses one frame up (`effort = 0.0 if u is None`), so a
        reward asked for without an action is priced as costing nothing to apply
        rather than as costing whatever the last action cost.
        """
        self.com_x(s)
        self._env.data.ctrl[:] = (0.0 if a is None
                                  else np.asarray(a, dtype=float).ravel())
        self._mj.mj_forward(self._env.model, self._env.data)
        # Computes `cfrc_ext`, which `mj_forward` does not; see the docstring.
        self._mj.mj_rnePostConstraint(self._env.model, self._env.data)
        raw = float(weight) * float(np.sum(np.square(self._env.data.cfrc_ext)))
        return min(raw, float(cap))

    def is_upright(self, s: Any) -> float:
        """1.0 while Humanoid-v5's own healthy check passes: torso height strictly
        inside the row's `("z_range", lo, hi)` band, which for this wheel is
        `(1.0, 2.0)`.

        A SEPARATE name from Hopper's `is_healthy`, deliberately, and the module's
        own precedent for it is `is_balanced`: these are three different predicates
        and `is_healthy`'s extra clauses (a 0.2 rad pitch limit, a +/-100 state
        range) do not exist on this model. One shared name would advertise them to
        a model that reads the helper table. Strict `<` on both sides, matching
        `humanoid_v5.py`'s `min_z < qpos[2] < max_z`; reproduced against upstream's
        `terminated` with 0 mismatches over a falling episode.
        """
        band = self._family["healthy"]
        if band is None or band[0] != "z_range":
            raise NotImplementedError(
                f"{self.name}: `is_upright` is the `('z_range', lo, hi)` check and "
                f"this family declares healthy={band!r}. Hopper's three-clause "
                "check is `is_healthy`; the pendulum's is `is_balanced`.")
        s = np.asarray(s, dtype=float).ravel()
        if not bool(np.isfinite(s).all()):
            return 0.0
        return float(band[1] < float(s[self._family["height_index"]]) < band[2])

    def torso_height(self, s: Any) -> float:
        """The root's height coordinate in metres, at the row's `height_index`
        (qpos[1] on the two planar locomotion bodies, qpos[2] on Humanoid's free
        root).

        Raises where the row declares None -- the pendulum, reacher and swimmer,
        whose s[1] is a pole angle, an elbow angle and a y position. Returning one
        of those under this name would be a plausible number that measures
        something else.
        """
        idx = self._family["height_index"]
        if idx is None:
            raise NotImplementedError(
                f"{self.name}: {self._gym_id} has no torso height -- its s[1] is "
                "not a height and this adapter will not report one under that name.")
        return float(np.asarray(s, dtype=float).ravel()[int(idx)])

    def heading(self, s: Any) -> float:
        """The front tip's heading angle in radians (swimmer; qpos[2])."""
        return float(np.asarray(s, dtype=float).ravel()[2])

    def pole_angle(self, s: Any) -> float:
        """The pole's angle from vertical in radians (inverted pendulum; s[1])."""
        return float(np.asarray(s, dtype=float).ravel()[1])

    def cart_position(self, s: Any) -> float:
        """The cart's position on the rail in metres (inverted pendulum; s[0])."""
        return float(np.asarray(s, dtype=float).ravel()[0])

    def is_balanced(self, s: Any) -> float:
        """1.0 while the pole is up: |angle| <= 0.2 and every value finite --
        InvertedPendulum-v5's own termination test, negated."""
        s = np.asarray(s, dtype=float).ravel()
        return float(bool(np.isfinite(s).all()) and abs(float(s[1])) <= 0.2)

    def is_healthy(self, s: Any) -> float:
        """1.0 while Hopper-v5's own healthy check passes on this state: height
        (s[1]) above 0.7, |torso angle| (s[2]) below 0.2, and everything FROM THE
        ANGLE ONWARD (s[2:]) finite and inside (-100, 100). The +/-100 clause
        deliberately excludes the height, exactly as the wheel does --
        `hopper_v5.py` applies `healthy_state_range` to `state_vector()[2:]`, so a
        torso at z >= 100 is (bizarrely but truly) still alive upstream and must
        still collect the alive bonus here. Computed on the unclipped simulator
        state, so -- unlike the spec's observation-based recomputation -- the
        clauses it does apply are exact."""
        s = np.asarray(s, dtype=float).ravel()
        return float(bool(np.isfinite(s).all())
                     and bool(np.all(np.abs(s[2:]) < 100.0))
                     and float(s[1]) > 0.7 and abs(float(s[2])) < 0.2)

    def joint_angles(self, s: Any) -> np.ndarray:
        """The actuated joint angles, as an array (family-specific slice)."""
        lo, hi = self._family["joint_slice"]
        return np.asarray(s, dtype=float).ravel()[lo:hi].copy()

    def joint_velocities(self, s: Any) -> np.ndarray:
        """The actuated joint velocities, as an array (family-specific slice)."""
        lo, hi = self._family["joint_vel_slice"]
        return np.asarray(s, dtype=float).ravel()[lo:hi].copy()

    def fingertip_pos(self, s: Any) -> np.ndarray:
        """The reacher fingertip's (x, y) in metres: planar two-link forward
        kinematics on `_REACHER_L1`/`_REACHER_L2` (see those constants for the
        asset provenance and the live-model assertion that keeps them honest)."""
        s = np.asarray(s, dtype=float).ravel()
        t0, t01 = float(s[0]), float(s[0]) + float(s[1])
        return np.array([_REACHER_L1 * math.cos(t0) + _REACHER_L2 * math.cos(t01),
                         _REACHER_L1 * math.sin(t0) + _REACHER_L2 * math.sin(t01)])

    def target_pos(self, s: Any) -> np.ndarray:
        """The reacher target's (x, y) in metres, s[2:4] (`_REACHER_TARGET`)."""
        return np.asarray(s, dtype=float).ravel()[_REACHER_TARGET].copy()

    def to_target_dist(self, s: Any) -> float:
        """The fingertip-target distance in metres; 0 means on target."""
        return float(np.linalg.norm(self.fingertip_pos(s) - self.target_pos(s)))

    def control_cost(self, a: Any) -> float:
        """Sum of squared action components -- a measure of effort."""
        return float(np.sum(np.square(np.asarray(a, dtype=float).ravel())))

    # -- describe(): what `full_source` shows -------------------------------

    #: `_span_of` on the underlying env class, handed to `generation._strip_reward`
    #: verbatim (the MetaWorld `reward_source` contract): gymnasium names its
    #: reward method `_get_rew`, which the `def \\w*reward\\w*` regex does NOT
    #: match, so without this the one thing `strip_existing_reward` exists to
    #: withhold -- the published human reward -- would ride into the prompt inside
    #: the env source. InvertedPendulum has no `_get_rew` (its reward is a line
    #: inside `step`), so its span is the whole `step` method; the termination
    #: bound it also removes is still stated twice in the surviving text (the
    #: field table and the prose).
    _reward_source: Optional[str] = None

    @property
    def reward_source(self) -> str:
        if self._reward_source is None:
            cls = type(self._env)
            span = _span_of(cls, "_get_rew") or _span_of(cls, "step")
            self._reward_source = span
        return self._reward_source

    def _render_full_source(self) -> str:
        """The ACTUAL gymnasium source, assembled by `inspect` -- plus the two
        things it cannot say.

        The base implementation returns `inspect.getsource(type(self))` -- this
        adapter, ~1000 lines of five-family plumbing in which no per-task fact
        appears (the field tables live in `_FAMILIES`, module level; the horizon
        in the spec), so the default render would name which `s[i]` is what for
        NO family while advertising every family's helpers. What the model needs
        is: the state table THIS adapter emits (which differs from the env's own
        `_get_obs` -- root positions included, no clipping, no cos/sin encoding),
        the environment's real dynamics/termination source, and this adapter's
        reference reward (stripped, with the env's `_get_rew`, by
        `strip_existing_reward` -- see `reward_source`).
        """
        parts = [
            f"# environment `{self.name}` -- gymnasium `{self._gym_id}`, full source.",
            "#",
            "# NOTE: the observation THIS environment emits is concat(qpos, qvel),",
            "# documented field by field below. It is NOT the `_get_obs()` of the",
            "# gymnasium class that follows: root positions are included here,",
            "# velocities are not clipped, and angles are raw radians.",
            "",
            "# --- the state and action this adapter emits ---",
            self._render_api_stub(),
            "",
            f"# --- the gymnasium environment: {type(self._env).__name__} ---",
            _safe_source(type(self._env)),
            "# --- this adapter's reference reward ---",
            _span_of(GymMujoco, "reference_reward"),
        ]
        return "\n\n".join(p for p in parts if p)

    # -- rendering -----------------------------------------------------------

    def render(self, state: np.ndarray, width: int = 320) -> np.ndarray:
        """One `(render_height, render_width, 3)` uint8 frame of the simulator
        restored to `state`.

        `width` is accepted and ignored -- the renderer captured its size at
        construction and `observability._resize_nearest` scales to
        `output.video.width`. Exactly one positional parameter beyond `self`,
        which is what `observability._accepts_a_state` requires.

        No `qacc_warmstart` zeroing here, deliberately: the warmstart seeds the
        constraint solver of a future `mj_step`, this method never steps, and
        every `_step` re-zeroes after its own `set_state` -- so a zeroing here
        could not affect a frame or a stepped result and would read as if render
        purity depended on it. Purity comes from where each camera's pose lives;
        see the `camera_config` note in `__init__`.
        """
        s = np.asarray(state, dtype=float).ravel()
        self._env.set_state(s[:self.n_q], s[self.n_q:])
        self._mj_forward()
        return np.asarray(self._env.render(), dtype=np.uint8)

    def _mj_forward(self) -> None:
        """`set_state` already calls `mj_forward`; this is here so the render path
        says so explicitly rather than depending on that remaining true."""
        import mujoco
        mujoco.mj_forward(self._env.model, self._env.data)

    def task_images(self) -> List[bytes]:
        """PNG bytes of the reset state, for `problem.instruction_modality:
        text+image`. Returns `[]` when imageio or GL is missing, because an
        instruction modality that cannot be honoured must degrade to text rather
        than abort a run."""
        try:
            import imageio.v3 as iio
            frame = self.render(self.reset(0))
            return [bytes(iio.imwrite("<bytes>", frame, extension=".png"))]
        except Exception as exc:  # noqa: BLE001 - optional dep or missing GL context
            log.debug("%s: task_images unavailable (%s)", self.name, exc)
            return []


# --------------------------------------------------------------------------
# registration: ten ids, one per gymnasium spec, mechanically derived
# --------------------------------------------------------------------------

#: The ten task ids this module implements. The check below is ONE-SIDED on
#: purpose: an id pinned here whose spec is GONE raises at import (an adapter
#: without its definition renders nothing and must fail loudly), but a NEW
#: `library: gymnasium` spec does NOT raise -- this loop runs inside
#: `registry.load_all()`, which forgives nothing but anthropic's ImportError, so
#: a symmetric check would let a pure-data commit (or a regenerated catalogue)
#: take down every config load, `--validate-all` and most of the test suite.
#: The unwired-extra case is still loud, just in the right place:
#: `tests/test_task_specs.py`'s coverage partition fails until the new spec is
#: either wired here (family row + fitness + this tuple) or listed in
#: `tasks/_no_adapter.json`. Contrast `metaworld._EXPECTED_MT10`, which checks
#: both directions because MT10 is a CLOSED protocol; this slice is open and
#: grows by authorship (`tasks/SOURCE.md`).
_EXPECTED_GYM = (
    "half_cheetah", "half_cheetah_backward", "half_cheetah_target_speed",
    "hopper_hop", "hopper_hop_in_place", "inverted_pendulum_balance",
    "reacher_hold", "reacher_reach", "swimmer_forward", "swimmer_heading",
    # REvolve's own MuJoCo task. The ELEVENTH spec and the SIXTH family --
    # every earlier row shares a simulator with at least one sibling; this one does not.
    "humanoid_run",
)


def _gym_specs() -> Mapping[str, TaskSpec]:
    """The `library: mujoco` slice of the catalogue (run through gymnasium).

    NOT memoised: `tasks.index()` already is, so this is a dict comprehension
    over 27 entries -- and a private second cache would hold import-time
    `TaskSpec` objects forever while `tasks.index(refresh=True)` rebuilt the real
    one, giving any in-process catalogue editor (a test using a temporary
    catalogue, say) two answers with no error.
    """
    found = {tid: s for tid, s in task_index().items()
             if s.library == "mujoco"}
    missing = sorted(set(_EXPECTED_GYM) - set(found))
    if missing:
        raise TaskSpecError(
            f"bird/envs/gym_mujoco.py registers adapters for specs that are gone "
            f"from the catalogue: {missing}. An adapter without its definition "
            "renders nothing; remove it from _EXPECTED_GYM or restore the spec")
    return found


def _env_id(task_id: str) -> str:
    """`tasks._BIRD_ID_RULES["mujoco"]` is the definition; this must agree with
    it, and `tests/test_task_specs.py`'s partition test fails if it stops."""
    return "gym_" + task_id


def _factory(task_id: str) -> Callable[[Any], GymMujoco]:
    def make(ctx: Any) -> GymMujoco:
        return GymMujoco(task_id)

    make.__name__ = _env_id(task_id)
    # Advertised on the FACTORY so `_check_coherence` can ask without constructing
    # a simulator -- see `metaworld._factory` for the pattern. Three facts travel
    # this way: the (empty) reduction set, the consumer anti-leak gate the specs
    # cannot carry, and that no success bar exists -- which is what lets coherence
    # refuse `evaluate.fitness.source: success_rate` at validation instead of an
    # all-tie being discovered in a finished sweep.
    make.supported_reductions = GymMujoco.supported_reductions
    make.consumer_forbidden_symbols = _CONSUMER_FORBIDDEN
    make.defines_success = False
    # The first sentence of the task's own instruction, rather than a schema field
    # carried purely to build a docstring.
    make.__doc__ = (f"Gymnasium/MuJoCo `{task_id}`. "
                    + str(_gym_specs()[task_id].description["natural_language"])
                    .split(". ")[0].strip()
                    + ". Spec-defined continuous fitness; no discrete success (§0).")
    return make


for _task_id in _EXPECTED_GYM:
    register("env", _env_id(_task_id))(_factory(_task_id))
del _task_id
