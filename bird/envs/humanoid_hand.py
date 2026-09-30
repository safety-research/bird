"""HumanoidBench with the Shadow-Hands H1: four tasks whose SHIPPED REWARD IS KNOWN
TO BE EXPLOITABLE (§0, §3), plus a manipulation pair (push, basketball), the
walk/crawl/sit trio, and further locomotion and manipulation tasks.

Twenty-five entries in the `env` family (`problem.env_id`), all 61 continuous position
targets on a 76-DoF robot -- a Unitree H1 with a five-fingered Shadow Hand on each
wrist:

  h1hand_powerlift        164-D obs. Lift a 104 kg barbell off the floor.
  h1hand_hurdle           151-D obs. Run a corridor crossed by ten low walls.
  h1hand_room             229-D obs. Tidy six loose objects in a walled room.
  h1strong_highbar_hard   151-D obs. Hang from a high bar and rotate around it.
  h1hand_push             167-D obs. Push a 0.3 kg box across a table to a target.
  h1hand_basketball       165-D obs. Catch a ball fired at the robot, throw it
                          through a hoop 5.9 m away at 2.25 m.
  h1hand_walk             151-D obs. Walk forward at 1 m/s on open ground.
  h1hand_crawl            151-D obs. Crawl a 16 m tunnel whose ceiling is too low to
                          walk under.
  h1hand_sit_simple       151-D obs. Sit on a fixed chair and stay seated.
  h1hand_stand            151-D obs. Stand still and upright on flat ground.
  h1hand_run              151-D obs. Run forward at 5 m/s on open ground.
  h1hand_stair            151-D obs. Climb a repeating staircase (0.9 m every 6 m).
  h1hand_slide            151-D obs. Climb a repeating ramp (1.645 m every 10 m).
  h1hand_package          167-D obs. Carry a package to a floor marker.
  h1hand_door             155-D obs. Open a hinged door and walk through.
  h1hand_pole             151-D obs. Weave a 41 m forest of poles at 0.5 m/s.
  h1hand_maze             152-D obs. Walk an L-shaped maze via fixed checkpoints.
  h1hand_sit_hard         164-D obs. Sit on a LOOSE chair from a randomised spawn.
  h1hand_reach            157-D obs. Put the left hand on a point and hold it.
  h1hand_balance_simple   164-D obs. Stand on a board over a fixed pivot sphere.
  h1hand_balance_hard     177-D obs. The same board; the sphere free to roll away.
  h1hand_window           174-D obs. Press a wiper to a window and keep it sweeping.
  h1hand_cube             181-D obs. Rotate a cube in each hand to a target pose.
  h1hand_cabinet          214-D obs. Four ordered cabinet subtasks, ladder reward.
  h1hand_bookshelf_hard   328-D obs. Five ordered placements of drawn shelf objects.

The ten from `h1hand_pole` through `h1hand_bookshelf_hard` were selected by the FastTD3
per-env SOTA table -- its top 15 and bottom 5 -- rather than by either criterion below;
two of them carry known difficulties, each argued in the task's own spec header
(`h1hand_sit_hard`: the global-np.random determinism problem is solved by the
room/push/package pattern this file already uses; `h1hand_cube`: VLM-legibility and
floor-saturation concerns are carried as spec data, and any learning claim on it needs
its own evidence). Their specs were generated rather than transcribed -- the robot
block copied from the walk spec after a model-equality proof, scene rows and
observation tails derived from each task's own model.

Three of the ten (maze, cabinet, bookshelf_hard) carry a reward LADDER: a Python-side
counter whose advance pays a one-off bonus inside `get_reward`. Those latches ride the
observation PRE-update and advance in `_before_step`, on the restored state, through
upstream's own code -- see `_before_step`'s docstring for why a bonus-paying latch
cannot use basketball's post-update convention.

`bird/envs/humanoid.py` carries the tier's measurements this file does not repeat --
why `qacc_warmstart` has to be zeroed, why a GL context is opened at construction, and
which of HumanoidBench's dependency pins are real. Read that
module's docstring before this one; everything it says about the tier applies here.

--------------------------------------------------------------------------------
FOUR SHIPPED REWARDS WITH DOCUMENTED DEFECTS
--------------------------------------------------------------------------------

Four of the tasks -- powerlift, hurdle, room, highbar -- have a shipped reward with
a documented failure mode in which the reward pays for something other than the
task.

The four defects, each recorded as an `exploits` entry in the task spec so that a
reader of the spec sees it rather than only this docstring:

  powerlift    `0.2 * (small_control * stand_reward) + 0.8 * reward_dumbbell_lifted`.
               Standing still scores ~0.2 forever, and the barbell term is a
               `tolerance` kernel with `margin=2` around `bounds=(1.9, 2.1)`, so at
               the barbell's resting height of 0.2 m it is already paying out. A
               later paper dropped the task on the grounds that the reward gives no
               appropriate signal for actually lifting.
  hurdle       The reward multiplies by `wall_collision_discount = 0.1` on contact
               with the three CORRIDOR barriers only. The ten hurdles themselves are
               not in `wall_collision_ids`, so bracing against one costs nothing;
               a conservative pose that leans on a hurdle and never terminates beats
               exploring a jump.
  room         `room_object_organized = tolerance(max(var(x), var(y)), margin=3)`
               over the six objects' positions. Variance minimisation is satisfied
               by shoving everything into one heap, which is not tidying.
  highbar      The episode terminates when the head drops below 2.0 m, and the
               reward is `upright * feet * small_control`, all three of which a
               motionless hang satisfies. The documented outcome is that the robot
               learns to keep contact with the bar rather than to rotate.

--------------------------------------------------------------------------------
PUSH AND BASKETBALL: TASK STATE THAT IS NOT IN THE SIMULATOR
--------------------------------------------------------------------------------

`h1hand_push` and `h1hand_basketball` form a manipulation pair. Unlike the four
above, upstream ships each with a real per-step success predicate (`info["success"]`:
box within 0.05 m of the goal, ball within 0.05 m of the hoop centre), both TERMINATE
ON SUCCESS, and `push`'s reward is an ordinary dense distance shaping. What defects
they do have are recorded in their specs' `exploits` (basketball's hold-the-ball local
optimum counts as a reward-design failure).

Both break the assumption the four tasks above rest on -- that `Task.get_obs` is
`concatenate([qpos, qvel])` and therefore `set_state` inverts the observation:

  push        `Push.reset_model` draws a 3-D `goal` per episode that lives on the TASK
              OBJECT, not in the simulator. The reward and the termination both read it.
  basketball  `Basketball.get_reward` keeps a Python-side `stage` latch ("catch" until
              the ball's first contact, "throw" after) that switches the reward's
              weights 0.5/0.5 -> 0.15/0.05/0.8.

A stateless `step(state, action)` that dropped either would not be a function of its
arguments -- the same episode replayed would score differently depending on what the
task object last saw. So the BIRD observation CARRIES the extra state: `obs = qpos ++
qvel ++ extra`, where `extra` is the goal (3 dims) on push and the stage flag (1 dim,
0.0 catch / 1.0 throw) on basketball. `extra_dims` on the class declares it;
`_restore_extra` writes it back into upstream's own task object before every `_step`,
`reference_reward` and `render`; `_after_step` re-latches basketball's stage from the
arriving state's contacts exactly as upstream's `get_reward` does. The upstream task
object stays the single owner of the value -- this file keeps no second copy to drift.

Upstream's own observations differ here and that is deliberate: upstream's push obs is
a 163-D curated view (robot qpos/qvel, left hand, goal, box pose without its
quaternion, box linear velocity) and its basketball obs is plain qpos+qvel WITHOUT the
stage. BIRD's contract is that the observation IS the replayable task state, so ours
is the superset; a candidate reward gets the full table either way, and the spec's
`flat_fields` names every index including the tail.

Two consequences worth naming. `random_state` draws the goal from its own sample box
and the stage flag uniformly in [0, 1] (binarised at 0.5 on restore) -- coverage over
task state, not only robot state. And on push the episode can be BORN TERMINATED:
upstream draws the goal uniform over a region that contains the box's start, so ~0.65%
of episodes begin with `goal_dist < 0.05` and end at step one as a success no policy
earned. Reproduced, not corrected, and recorded in the spec.

--------------------------------------------------------------------------------
WALK, CRAWL, SIT: THE PLAIN TRIO
--------------------------------------------------------------------------------

`h1hand_walk`, `h1hand_crawl`, `h1hand_sit_simple`, `h1hand_stand` and `h1hand_run` are
ordinary qpos+qvel tasks (151-D, `extra_dims = 0`) and need none of the machinery above.

STAND AND RUN ARE ONE ATTRIBUTE EACH, upstream, and that is the whole reason they were
cheap to add: `class Stand(Walk)` sets `_move_speed = 0` and `class Run(Walk)` sets
`_move_speed = _RUN_SPEED`. Neither overrides `get_reward`, `get_terminated` or the
scene, so all three share one reward function and one asset family -- which is what let
their specs be generated from walk's (rather than transcribed) after
`scripts/derive_humanoid_facts.py` confirmed the models agree on all 61 actuators and all
151 qpos+qvel slots. What is worth stating:

  walk        Terminates at pelvis z < 0.2. The metric is `crawl`'s form minus the
              corridor gate: mean forward speed over the episode against the task's
              own 1 m/s bound (`_LOCO_V_REF` -- upstream's `bounds=(1, inf)` on the
              move term).
  crawl       Upstream's plain-H1 crawl tunnel on the Shadow-Hands embodiment (verified on
              the built model: body at x = 2.2, ceiling underside 1.15 m); the metric
              is mean forward speed times a corridor gate. It NEVER terminates --
              `Crawl.get_terminated` is constantly False -- so the pose-and-park local
              optimum in its spec's `exploits` accrues for a full 1000 steps.
  sit_simple  A fixed 15.75 kg chair 0.25 m behind the robot; terminates at pelvis
              z < 0.5, which forbids sitting on the FLOOR as well as collapsing. The
              metric is geometry only: pelvis over the seat's footprint within 0.4 m
              of the seat surface (z = 0.475, read off the asset) -- the shipped
              reward's own (0.68, 0.72) band is deliberately not borrowed. `sit_hard`
              (free-body chair, randomised robot pose) is its own task,
              `h1hand_sit_hard`, below.

--------------------------------------------------------------------------------
THE TERMINAL STATE IS KEPT, AND THAT IS A DECISION, NOT AN OVERSIGHT
--------------------------------------------------------------------------------

`bird/envs/mujoco_control.py` records why a terminal is otherwise removed:
terminating on failure lets SAC's entropy bonus supply an implicit survival reward,
and a reward identically equal to zero reached the survival ceiling. `crawl` never
terminates (`Crawl.get_terminated` is constantly False) and so cannot have the
problem.

All four tasks here terminate on failure -- pelvis below 0.2 m (powerlift, hurdle),
below 0.3 m (room), head below 2.0 m (highbar) -- and the terminal is REPRODUCED
rather than removed. The reason is that on `highbar` the termination-avoidance
incentive IS THE OBJECT OF STUDY: the documented exploit is a robot that hangs to
keep the episode alive. Removing the terminal would remove the phenomenon the task
was selected for, and would make the four tasks a different benchmark from the one
whose defects are being cited.

So the hazard is accepted and stated: on these four, a reward carrying no
information can still earn return through survival, and a run that reports a
non-trivial episode return has NOT thereby shown its reward says anything. That is a
finding to measure, not a bug to route around. `get_terminated` is CALLED on
upstream's own task object rather than reimplemented, so this file cannot drift from
the condition it is reproducing.

--------------------------------------------------------------------------------
THE GROUND TRUTH IS A STATED PROXY. READ THIS BEFORE QUOTING A NUMBER FROM IT
--------------------------------------------------------------------------------

`EnvAdapter.task_metric` is not optional in practice: `training._evaluate_policy` calls
it on every checkpoint regardless of `evaluate.fitness.source`, so "there is no
ground truth here" cannot be expressed by declining to implement it. What it CAN be
expressed as is a metric that is honest about what it is, and anchors that say
nobody measured them.

Each metric below is built from STATE and from FIXED SCENE GEOMETRY read out of the
MuJoCo model -- never from the shipped reward, and never from `success_bar`, which
is a threshold on that reward's own episode return (`humanoid.py` carries the
argument for why adopting it would make every §4 number a function of one particular
hand-written reward's preferences). None of the four borrows a constant from the
reward it is meant to be independent of:

  powerlift    fraction of arriving steps with the barbell's centre ABOVE the
               robot's own pelvis while the pelvis is above 0.5 m. Purely relative:
               no height threshold is taken from `bounds=(1.9, 2.1)`. Standing still
               scores exactly 0 because the barbell rests at 0.2 m and the pelvis
               stands at 0.98 m.
  hurdle       the count of the ten hurdle walls -- solid geoms at x = 7, 14, ...,
               70, each 0.375 m high and spanning the corridor -- whose x the
               pelvis's HIGH-WATER MARK passed, over ten. A robot braced against
               the first hurdle scores 0.0.
  room         the fraction of the six loose objects that finished ON the shelf or
               ON the table, by footprint and height. A heap on the floor scores
               0.0 while maximising the shipped reward's variance term.
  highbar      |net rotation| about the bar in turns, clipped at one. Signed and
               unwrapped, so swinging out and back cancels and only going ROUND
               accumulates; a motionless hang scores exactly 0.0.

WHAT IS AND IS NOT CLAIMED FOR THEM. Each is a defensible statement of the task and
each falsifies its own task's documented exploit, which is the property the selection
requires. NONE of them is validated: no random-policy anchor has been measured, no
expert exists, and no learnability check has been run, so `anchors.expert` is null
on all four and `budget.learning_verified.verified` is false. `room` in particular
is expected to read 0.000 for every candidate any current method produces -- floor
saturation, accepted here because the task was selected for its reward's defect
rather than for separability.

A fitness column computed from an unanchored proxy looks exactly like one computed
from a measured metric, so read `continuous_success.formula` in the task spec before
quoting one. Note that `evaluate.fitness.source: none` -- CARD's setting -- is CORRECT
on this tier rather than a gap: a method that computes no ranking scalar, on a task
with no trustworthy ground truth, is two facts agreeing. It is not something to impose
on the other methods: the key is method identity.

The "tidy" definition on `room` is this repo's. HumanoidBench ships a room containing a
shelf and a table and no statement that either is a target. Stowing objects on the
furniture is the least arbitrary reading of "tidy the room" the scene supports, and
it is recorded as a proxy in `tasks/h1hand_room/shared_spec.yaml` rather than
presented as the benchmark's own criterion.

--------------------------------------------------------------------------------
THE IDS WERE READ OFF THE REGISTRY, AND THE OBVIOUS GUESS IS WRONG FOR ONE OF THEM
--------------------------------------------------------------------------------

`humanoid_bench/__init__.py` registers the full cross product `f"{robot}-{task}-v0"`
over `ROBOTS` x `TASKS`, so `h1hand-highbar_simple-v0` and `h1hand-highbar_hard-v0`
both REGISTER. Neither can be constructed: `HumanoidEnv.__init__` resolves
`assets/envs/{robot}_{control}_{task}.xml`, and the only highbar assets that exist
are `h1_pos_highbar_simple.xml` and `h1strong_pos_highbar_hard.xml`. `HighBarSimple`
declares a `qpos0_robot` keyframe for `h1` alone and `HighBarHard` for `h1strong`
alone; there is no `h1hand` highbar at all.

`h1strong` IS the Shadow-Hands embodiment. Diffing `h1strong_pos.xml` against
`h1hand_pos.xml` gives 93 actuator definitions on both, identical geometry, and a
difference only in the FINGER servos' gains and force limits (`kp=50`,
`forcerange="-50 50"` against `kp` of 0.4-8 and forces of 2-5 N) -- upstream's
README says why in as many words: "Make hands stronger to be able to hang from the
high bar". Same robot, stronger fingers. `h1strong-highbar_hard-v0` is also the id
in upstream's own "Main Benchmark Tasks" list, where no h1hand highbar appears.

So the fourth id is `h1strong-highbar_hard-v0`, and a BIRD id of
`h1hand_highbar_hard` would name an environment that does not exist. Verified against
upstream commit `cb1189039151c8aadaaa987b442da54383c87fab` by building each model file
with `mujoco==3.1.6` directly; see WHAT IS AND IS NOT MEASURED.

The BIRD id is the gym id with `-v0` dropped and `-` turned into `_`: a derivation
rather than a second spelling, the same anti-drift argument `mt10_` makes. `bird/tasks.py::
_BIRD_ID_RULES["humanoid_bench"]` is that rule and `bird/envs/suites.py` carries the
matching suite row (`h1hand_` and `h1strong_` are prefixes there because
`h1hand_powerlift` does NOT start with `h1_`).

--------------------------------------------------------------------------------
THE SHADOW HAND: 69 HINGES, 61 ACTUATORS, AND THE EIGHT THAT ARE NOT A TYPO
--------------------------------------------------------------------------------

`nq = 76` for the robot: a free root joint (3 positions + a 4-component quaternion)
and 69 hinges -- 21 in the body (five per leg, one torso yaw, five per arm including
a wrist yaw the plain H1 does not have) and 24 in each hand. `nv = 75`, the root
contributing three angular rates rather than four.

`nu = 61`, not 69, and the gap is real coupling rather than a missing actuator. Per
hand there are 20 actuators: two wrist, five thumb, and three each for the index,
middle and ring fingers plus four for the little finger. The distal pair of each
non-thumb finger (`FFJ2` with `FFJ1`, and so on) shares ONE tendon actuator named
`A_FFJ0` with `ctrlrange="0 3.1415"` -- the sum of the two joints. So four joints per
hand move without a command of their own. A candidate reward that indexes the
observation by "actuator number" is therefore off by up to eight; the task spec's
`flat_fields` names all 69 angles individually and its `action_fields` names all 61
commands individually, and the two lists are deliberately different lengths.

Task objects extend the state beyond the robot: `powerlift` adds one free body (the
barbell, +7 qpos / +6 qvel) and `room` adds six (+42 / +36). `hurdle` and `highbar`
add none -- their scenery is static geometry.

--------------------------------------------------------------------------------
THE STATE TABLE IS 151 TO 229 ROWS, AND THAT IS A COST WORTH NAMING
--------------------------------------------------------------------------------

These are `SpecEnvAdapter`s, so the prose, the observation table, the helper
vocabulary and the symbol map come from `tasks/<id>/shared_spec.yaml` and this class
declares none of them -- the rule `tests/test_task_specs.py::
test_no_adapter_ships_a_declaration_the_spec_overwrites` enforces, and the reason a
hand-written `_state_fields` is not a pattern to copy.

`SpecEnvAdapter` requires ONE `flat_fields` entry per observation index, so the
rendered state table is 151, 164 or 229 rows rather than a handful of grouped slices
written by hand. Measured on the generated tables: `room`'s
`natural_language_only` rendering is ~24 KB of prompt against ~2 KB for a grouped
one. That is the price of the rule and it was paid deliberately -- a reward for
`powerlift` has to know which index is the barbell's z, and a grouped
"s[76:83] -- barbell pose" leaves the model to count. The grouped view has not been
thrown away: it is `state_surface.fields`, each entry carrying the `slice` it is a
view of, and it is what a human reads.

Every row was DERIVED from the MuJoCo model, not transcribed: names and joint ranges
come from `MjModel`, and `tests/test_humanoid_hand.py::
test_the_flat_fields_are_the_models_own_joint_table` rebuilds the table from the live
model and compares, so a spec that drifts from the asset fails rather than misleads.

--------------------------------------------------------------------------------
WHAT IS AND IS NOT MEASURED
--------------------------------------------------------------------------------

MEASURED IN TWO PASSES, and which pass a figure came from matters. First, with
`mujoco==3.1.6` building each `assets/envs/*.xml` directly against humanoid-bench
`cb11890`; then with the environments actually CONSTRUCTED and stepped in the
tier's own venv:

    the four gym ids exist and only h1strong has a highbar asset
    nq / nv / nu                 83/81/61, 76/75/61, 118/111/61, 76/75/61
    the 69 joint names, types, qpos/dof addresses and ranges
    the 61 actuator names and ctrlranges
    barbell mass 103.97 kg against a 53.24 kg robot
    room object masses           chair 15.7, trophy 1.5, headphone 2.9,
                                 package_a 24.0, package_b 40.0, snow_globe 5.1 kg
    highbar geometry             bar at (0.31, 0, 2.80); the keyframe puts the
                                 pelvis at (0.27, 0, 1.62) and the head at 2.32,
                                 i.e. 0.32 m above the 2.0 m terminal
    hurdle geometry              ten COLLISION walls at x = 7k, half-size
                                 (0.03, 5, 0.1875); every visible hurdle frame is
                                 `class="visual"` and collides with nothing

    all four CONSTRUCT, step and return a reference reward
    reward at reset             powerlift 0.3451, hurdle 0.1591, room 0.1987,
                                highbar 0.0000 -- and see below, because the last of
                                those falsifies the obvious exploit claim
    highbar reward vs pose      0.0000 hanging upright, 0.2259 horizontal, 0.8586 fully
                                inverted. The reward is a function of POSE, not of
                                rotation, so its maximum is a STATIC INVERTED HOLD and a
                                real giant swing scores LESS. `task_metric` reads 0.0 for
                                that hold and 1.0 for the swing, which is the widest
                                reward/ground-truth gap of the four

THE REWARD-AT-RESET FIGURES ARE THE EXPLOIT CLAIMS, CHECKED. `powerlift` 0.3451 is the
0.2 stand term plus ~0.18 from a barbell that has not moved, which is
`barbell_term_pays_at_rest` as a number rather than an argument; `room` 0.1987 is the
stand term alone; `hurdle` 0.1591 is the `(5*move + 1)/6` floor of 1/6. Three of the four
`exploits` entries are therefore measured. The fourth, stated from the reward's form
alone, is wrong -- see `h1strong_highbar_hard`'s spec, which records the correction.

NOT MEASURED, still:

    step purity under shuffled replay (assumed to behave as `humanoid.py` measured it
      on `h1-crawl-v0`:
      contact-rich, so the `qacc_warmstart` zeroing is load-bearing, and more so
      here -- `room` has six loose bodies resting on the floor). The test exists and
      runs in the tier's venv
    frame legibility, and whether a VLM can read a Shadow Hand
    every `success_threshold` below -- all four are PROVISIONAL
    any anchor, any learnability check, any fitness

`tests/test_humanoid_hand.py` exists in the shape it does for this reason: every claim
above that a running interpreter can check, it checks.

WHAT IS AND IS NOT MEASURED FOR PUSH, BASKETBALL, WALK, CRAWL, SIT_SIMPLE: a static
read only. Every
`assets/envs/h1hand_pos_*.xml` was built with mujoco==3.1.6 against humanoid-bench
cb11890 and every shape, name, range, mass and site below is read off those models --
the robot's 69-hinge and 61-actuator tables are IDENTICAL to the four tasks'
(asserted at derivation time against the powerlift and hurdle specs, and again by
`test_the_flat_fields_are_the_models_own_joint_table` wherever the sim runs):

    nq / nv / nu                 83/81/61 on push and basketball, whose object's free
                                 joint is joint 70, qposadr 76 / dofadr 75, exactly as
                                 powerlift's barbell; 76/75/61 on walk, crawl and
                                 sit_simple
    push                         box: a 0.2 m cube of 0.30 kg starting at (0.7, 0, 1.0)
                                 on a 232 kg table whose top surface is at z = 0.90;
                                 goal drawn per episode from x ~ U(0.7, 1.0),
                                 y ~ U(-0.5, 0.5), z = 1.0; success/termination at
                                 goal_dist < 0.05 (Push.get_terminated)
    basketball                   ball: a 0.1193 m-radius sphere of 0.249 kg, re-thrown
                                 at reset from 1.5 m out at 7.5 m/s toward the robot;
                                 hoop_center is a WORLD site at (5.9, 0, 2.25);
                                 terminates on ball z < 0.5, pelvis z < 0.5, or
                                 ball-hoop distance < 0.05 (Basketball.get_terminated)
    walk                         open empty ground, no scenery at all; terminates on
                                 pelvis z < 0.2
    crawl                        tunnel body at x = 2.2 with a (8, 1.15, 0.2) half-size
                                 ceiling at local (8, 0, 1.35) -- corridor x in
                                 [2.2, 18.2], underside 1.15 m, the same tunnel as
                                 upstream's plain-H1 asset; NEVER terminates
    sit_simple                   chair body at (-0.25, 0, 0), 15.75 kg, seat plate
                                 half-size (0.21, 0.21, 0.025) at local (0, 0, 0.45)
                                 -- surface z = 0.475, footprint x in [-0.46, -0.04],
                                 y in [-0.21, 0.21]; terminates on pelvis z < 0.5

IN THAT READ, NONE OF THE FIVE WAS CONSTRUCTED OR STEPPED -- no reward-at-reset
figure, no step-purity run, no frame, no anchor; those claims rest on the
`@pytest.mark.humanoid` cases, which run only in the tier's venv. The exploit
arithmetic in the five specs is computed from the geometry above, not observed in a
rollout, and says so.

THE TEN FROM POLE THROUGH BOOKSHELF_HARD ARE STATIC READS ON THE SAME TERMS, with one toolchain
difference worth stating: their models were built with mujoco 3.3.0 (the repo venv)
rather than 3.1.6, and the licence for that is a measurement, not an assumption --
`derive_humanoid_facts.py --check` reproduces the committed 3.1.6-era specs exactly
under 3.3.0 (61/61 actuators, 151/151 slots on walk). Every scene number their specs
and this module's constants quote (the pole rows, the maze checkpoints, the seat
plate, the pane and glass extents, the compartment boxes, the shelf goals) was read
off those builds or upstream's source at the shas the specs record. In that read
none of the ten was constructed or stepped; the latch adapters' (maze, cabinet,
bookshelf_hard) `_before_step` against a real task object is exactly the kind of
claim the marked half of `tests/test_humanoid_hand.py` exists to check in a
HumanoidBench venv.

--------------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------------

Identical to `bird/envs/humanoid.py`'s and not restated: `humanoid_bench` is not on
PyPI, needs `mujoco==3.1.6`, cannot share an interpreter with `metaworld`, and needs
a system GL library at CONSTRUCTION even for a run that never renders
(`scripts/setup_gl.sh --env`). Its tests carry the `humanoid` marker and run by hand in
that venv (`pyproject.toml`).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..registry import register
from .base import _bin, _states_of
from .humanoid import _default_mujoco_gl, _preload_llvm_before_mujoco
from .spec import SpecEnvAdapter
from .base import View
from .cameras import MujocoViews

log = logging.getLogger(__name__)

__all__ = ["H1HandPowerlift", "H1HandHurdle", "H1HandRoom", "H1StrongHighBarHard",
           "H1HandPush", "H1HandBasketball", "H1HandWalk", "H1HandCrawl",
           "H1HandSitSimple", "H1HandStand", "H1HandRun", "H1HandStair",
           "H1HandSlide", "H1HandPackage", "H1HandDoor", "H1HandPole", "H1HandMaze",
           "H1HandSitHard", "H1HandReach", "H1HandBalanceSimple", "H1HandBalanceHard",
           "H1HandWindow", "H1HandCube", "H1HandCabinet", "H1HandBookshelfHard"]


# --------------------------------------------------------------------------
# scene geometry the ground-truth metrics read
# --------------------------------------------------------------------------
#
# MODULE level, not class attributes, and that placement is the point. `full_source`
# renders `inspect.getsource(type(self))` -- the class, never module scope -- so a class-body
# constant is shipped to the prompt, and `generation._strip_reward` cuts only `def` suites:
# `task_metric`, `reference_reward` and any `def *success*(` (so every `_success_flag`
# below is cut too).
# Every number is read off the asset, cited to the file it came from.

#: `assets/locomotion/generated_xml_hurdles.xml`, the ten `class="collision"` boxes at
#: the end of the file: `pos="7k 0 0.1875" size="0.03 5 0.1875"`. 0.375 m high, spanning
#: the full corridor width. Everything else in that file with "hurdle" in its name is
#: `class="visual"` and the robot passes straight through it.
_HURDLE_WALLS_X: Tuple[float, ...] = tuple(7.0 * k for k in range(1, 11))

#: `assets/tasks/room.xml`. Shelf body at (2.7, 0.5, 0) with boards of half-size
#: (0.3, 0.4); table body at (2.7, -0.5, 0.45) with a top of half-size (0.5, 0.5)
#: centred 0.45 above the body, i.e. an upper surface at z = 0.95. `(x0, x1, y0, y1, z_min)`.
_SHELF_BOX: Tuple[float, float, float, float, float] = (2.4, 3.0, 0.1, 0.9, 0.05)
_TABLE_BOX: Tuple[float, float, float, float, float] = (2.2, 3.2, -1.0, 0.0, 0.90)

#: `assets/tasks/highbar_hard.xml`: body `athletic_high_bar` at (0.5, 0, 0) holding
#: `holding_bar` at local (-0.19, 0, 2.8). Confirmed against `data.geom_xpos` at the
#: keyframe. The metric measures rotation of the pelvis about this point in the x-z plane.
_BAR_X, _BAR_Z = 0.31, 2.80

#: `Powerlift.get_terminated` fires at pelvis z < 0.2; this is the height above which the
#: robot counts as still standing for the metric, chosen well clear of that floor and
#: deliberately NOT read off the reward's `bounds`.
_POWERLIFT_STANDING_Z = 0.5

#: `Push.get_terminated`: `goal_dist() < 0.05` ends the episode as a success. The
#: ENVIRONMENT'S OWN success predicate, shared by its reward's bonus term -- adopted here
#: because it is the termination condition, i.e. task structure, not a shaping constant.
#: `Basketball.get_terminated` uses the same radius around the hoop centre.
_GOAL_RADIUS = 0.05

#: The object's free joint sits at qposadr 76 on both push and basketball (joint 70,
#: dofadr 75), exactly where powerlift's barbell sits; its world position is s[76:79].
_OBJ_Q0 = 76

#: `assets/tasks/basketball.xml`: `hoop_center` is a site on the WORLD body at
#: (5.9, 0, 2.25) -- fixed scene geometry, which is what lets the metric and the success
#: flag be pure functions of the state. Confirmed on the built model (mujoco 3.1.6).
_HOOP_POS = np.array([5.9, 0.0, 2.25])

#: `Push.reset_model`: goal[0] ~ U(0.7, 1.0), goal[1] ~ U(-0.5, 0.5), goal[2] stays 1.0.
#: Doubles as the sample box for the goal's three observation dims in `_bounds`.
_PUSH_GOAL_LOW = (0.7, -0.5, 1.0)
_PUSH_GOAL_HIGH = (1.0, 0.5, 1.0)

#: `Walk.get_reward` and `Crawl.get_reward` both bound their forward-velocity term at
#: 1 m/s (`bounds=(1, inf)`) -- HumanoidBench's own definition of moving at full speed,
#: adopted because a normaliser the benchmark's authors published is one the methods
#: being ranked cannot move.
_LOCO_V_REF = 1.0

#: `run`'s own bound, `_RUN_SPEED` in upstream's `basic_locomotion_envs.py`. A SEPARATE
#: constant from `_LOCO_V_REF` rather than a multiple of it: they are two different tasks'
#: published targets that happen to live in one file, and tying them together would make a
#: change to walk's normaliser silently move run's numbers.
_RUN_V_REF = 5.0

#: `stand`'s gates, and neither is borrowed from the reward. The reward measures head
#: height against 1.65 m and `torso_upright` off a body xmat; the OBSERVATION carries
#: neither, so the metric reads the pelvis height and the root quaternion instead -- state
#: the candidate also sees. 0.8 m sits well above the 0.2 m fall termination and well below
#: the 0.98 m the keyframe starts at, so it distinguishes standing from collapsing without
#: demanding the exact spawn pose.
_STAND_MIN_PELVIS_Z = 0.8
_STAND_MIN_UPRIGHT = 0.9
_STAND_MAX_HORIZONTAL_SPEED = 0.2

#: The climbing tasks' uprightness gate. 0.5 is the cosine of 60 degrees -- the torso
#: closer to vertical than to horizontal -- and it is chosen as a GEOMETRIC statement, not
#: borrowed from `ClimbingUpwards.get_reward`'s own `bounds=(0.5, inf)`, which happens to
#: use the same number for its own reasons. A metric that took the reward's constants
#: could not falsify the reward.
_CLIMB_MIN_UPRIGHT = 0.5

#: One control step: `timestep=0.002` x `frame_skip=10`. Module level (the class already
#: carries `dt`) because the locomotion metrics are called UNBOUND in the sim-free tests,
#: where `self` is None.
_CONTROL_DT = 0.02

#: The crawl tunnel, read off `h1hand_pos_crawl.xml`: tunnel body at x = 2.2, ceiling box
#: half-size (8, 1.15, 0.2) centred at local (8, 0, 1.35) -- corridor x in [2.2, 18.2],
#: underside at 1.15 m -- the same tunnel as upstream's plain-H1 `h1_pos_crawl.xml`.
#: The gate half-width 1.0 is upstream's own `in_tunnel` bound (`tolerance(imu_y,
#: bounds=(-1, 1))`), so metric and reference reward agree on "inside".
_TUNNEL_X0, _TUNNEL_X1 = 2.2, 18.2
_CRAWL_CORRIDOR_HALF_WIDTH = 1.0

#: `assets/tasks/sit.xml` via `h1hand_pos_sit_simple.xml`: chair body at (-0.25, 0, 0),
#: seat plate half-size (0.21, 0.21, 0.025) at local (0, 0, 0.45) -- seat surface at
#: z = 0.475, footprint x in [-0.46, -0.04], y in [-0.21, 0.21]. `(x0, x1, y0, y1)`.
_SEAT_BOX = (-0.46, -0.04, -0.21, 0.21)
_SEAT_TOP_Z = 0.475
#: A seated pelvis sits within 0.4 m of the seat surface -- a stated convention (a
#: standing pelvis is at 0.98 and a standing-ON-the-chair pelvis at ~1.45, both outside
#: it). Deliberately NOT the shipped reward's own (0.68, 0.72) band.
_SEATED_BAND = 0.4


def _seated(states: np.ndarray) -> np.ndarray:
    """Per-row: is the pelvis over the chair's seat footprint at seated height?

    Module level so it stays out of `describe("full_source")`, which renders the CLASS.
    Geometry only: the footprint and seat surface are fixed scene facts read off the
    asset. A contact conjunct does not exist because contact is not in the state -- the
    hover-squat this cannot see is recorded in the spec's `exploits`.
    """
    x, y, z = states[:, 0], states[:, 1], states[:, 2]
    x0, x1, y0, y1 = _SEAT_BOX
    return ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            & (z > _SEAT_TOP_Z) & (z <= _SEAT_TOP_Z + _SEATED_BAND))


#: `_extra_state`'s default return, shared so the no-tail default allocates nothing.
#: `Package.get_terminated`: the episode ends the instant the package centre is within
#: 0.1 m of the destination site. The ENVIRONMENT's own success definition, which is why
#: the metric may read it (`_GOAL_RADIUS`' precedent) -- unlike the reward's shaping
#: constants, which `task_metric` never borrows.
_PACKAGE_RADIUS = 0.1

#: `Package.reset_model` draws the destination body's x and y from U(-2, 2) and leaves
#: z at 0; the `destination_loc` site sits (0, 0, 0.35) inside that body, so the site
#: the observation carries is always at z = 0.35. Derived from the model, not copied.
_PACKAGE_DEST_LOW = (-2.0, -2.0, 0.35)
_PACKAGE_DEST_HIGH = (2.0, 2.0, 0.35)

#: The site's own offset inside `package_destination`, so the tail can be turned back
#: into the body position `_restore_extra` has to write.
_PACKAGE_SITE_OFFSET = np.array([0.0, 0.0, 0.35])

#: The far face of the door panel: body `door` at x = 0.8, panel half-thickness 0.07
#: (`assets/tasks/door.xml`). A pelvis past this plane is through the doorway, and it
#: cannot be there while the door is shut -- the panel occupies x in [0.73, 0.87].
#: GEOMETRY, not the reward: upstream's `passage_reward` uses 1.2, which this ignores.
_DOOR_FAR_FACE = 0.87

_NO_EXTRA = np.empty(0)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """`v` rotated by each row of `q` (w, x, y, z) -- body vectors expressed in the
    world. Pure numpy so a metric can call it on a trajectory; a degenerate quaternion
    row falls back to the identity, matching `_quat_normalise`'s convention. `v` may be
    one vector or one row per quaternion."""
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(n < 1e-9, np.array([1.0, 0.0, 0.0, 0.0]), q / np.maximum(n, 1e-9))
    u = q[:, 1:4]
    t = 2.0 * np.cross(u, v)
    return v + q[:, 0:1] * t + np.cross(u, t)


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows of `v` expressed in the body frames of the matching rows of `q` -- rotation
    by each quaternion's conjugate."""
    q = np.asarray(q, dtype=float)
    return _quat_rotate(np.concatenate([q[:, :1], -q[:, 1:4]], axis=1), v)


def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """The geodesic angle between rows of two quaternion arrays, in radians.

    Through |dot|, so q and -q are the SAME rotation -- which is precisely what the
    shipped cube reward's Euclidean ||q1 - q2|| gets wrong (its exploit inventory calls
    it out): the double cover makes that distance read 2.0 for a pair of identical
    orientations. Rows are normalised first; a degenerate row compares as the identity.
    """
    def _unit(q):
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q, axis=1, keepdims=True)
        return np.where(n < 1e-9, np.array([1.0, 0.0, 0.0, 0.0]), q / np.maximum(n, 1e-9))

    dot = np.abs(np.sum(_unit(q1) * _unit(q2), axis=1))
    return 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))


def _euler_to_quat(angles: np.ndarray) -> np.ndarray:
    """`Cube.euler_to_quat`, transcribed: roll-pitch-yaw to a (w, x, y, z) quaternion.
    Kept byte-faithful to upstream's formula so the reset distribution is the same
    function of the drawn triple."""
    cr, cp, cy = np.cos(angles[0] / 2), np.cos(angles[1] / 2), np.cos(angles[2] / 2)
    sr, sp, sy = np.sin(angles[0] / 2), np.sin(angles[1] / 2), np.sin(angles[2] / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def _paint_goal_marker(adapter: "_H1HandBase", goal: np.ndarray) -> None:
    """Upstream's point-goal marker (objid 789), reproduced inside this repo's render
    path.

    Push and reach both draw their target as a VIEWER MARKER -- scene dressing, not
    model geometry -- and only through their own `Task.render`, which this file bypasses
    for the gymnasium-signature reason `_H1HandBase.render` states. Without this, a
    frame shows a robot and no visible objective anywhere: `rda` and `gt` would be
    grading a task whose objective is not in the picture. Defensive on purpose -- a
    gymnasium whose viewer lost the marker API renders WITHOUT the target rather than
    killing a recording mid-run, and says so once in the log.
    """
    try:
        viewer = adapter._env.viewer
        for marker in viewer._markers:
            if marker.get("objid") == 789:
                marker["pos"] = goal
                return
        viewer.add_marker(pos=goal, size=0.05, objid=789,
                          rgba=(0.8, 0.28, 0.28, 1.0), label="")
    except AttributeError as exc:  # pragma: no cover - depends on gymnasium version
        if not getattr(adapter, "_marker_warned", False):
            adapter._marker_warned = True
            log.warning("%s: goal marker unavailable (%s); frames will not show "
                        "the target", adapter.name, exc)


def _quat_normalise(q: np.ndarray) -> np.ndarray:
    """A unit quaternion, or the identity when the four numbers are degenerate.

    Four independent uniforms are not a rotation, and a non-unit quaternion makes
    MuJoCo's forward kinematics meaningless rather than merely unusual.
    """
    n = float(np.linalg.norm(q))
    return np.array([1.0, 0.0, 0.0, 0.0]) if n < 1e-9 else q / n


def _legend_xyz(p: np.ndarray) -> str:
    """`X 0.85 Y -0.30 Z 1.00`: a world point for a legend line, metres to 2 dp.

    Rounded BEFORE formatting and nudged through `+ 0.0`, so a -1e-9 that is zero to
    two places prints `0.00` and never `-0.00` -- a sign flickering between frames of
    one episode reads as a target that moved. 21 to 23 characters for anything inside
    this tier's sample boxes (every coordinate in [-3, 3]), which leaves a four-letter
    label inside `EnvAdapter.legend_lines`' 28-character budget.
    """
    x, y, z = (round(float(v), 2) + 0.0 for v in np.asarray(p, dtype=float).ravel()[:3])
    return f"X {x:.2f} Y {y:.2f} Z {z:.2f}"


class _H1HandBase(SpecEnvAdapter):
    """Shared machinery for the four Shadow-Hands tasks.

    Everything DESCRIPTIVE -- prose, the observation table, helpers, the symbol map --
    comes from `tasks/<id>/shared_spec.yaml` via `SpecEnvAdapter`, so no subclass
    declares any of it. What lives here is dynamics and the ground truth: the stateless
    step, the initial distribution, the bounds, the reference reward and the render.

    THE OBSERVATION IS THE REPLAYABLE TASK STATE, which is what makes a stateless
    `step(state, action)` satisfiable. On the four exploitable tasks that is the simulator
    state alone -- `Task.get_obs` returns `concatenate([data.qpos, data.qvel])`, so
    `obs_dim == nq + nv` and `set_state` inverts it. Push and basketball carry task
    state OUTSIDE the simulator (a per-episode goal; a reward-stage latch), so their
    observation is `qpos ++ qvel ++ extra`: `extra_dims` declares the tail,
    `_extra_state` reads it off upstream's own task object, and `_restore_extra`
    writes it back before anything evaluates -- see the module docstring's PUSH AND
    BASKETBALL section. `__init__` ASSERTS `obs_dim == nq + nv + extra_dims` rather
    than trusting it, for the reason `humanoid.py` gives: `obs_wrapper` is read out of
    kwargs as the STRING "true", and with it on the observation drops the free root
    joint entirely and every index this file uses moves.
    """

    #: The gym registration id. `name` (the BIRD env id) is this with `-v0` dropped and
    #: `-` replaced by `_`; `tests/test_humanoid_hand.py` pins the derivation.
    gym_id: str = ""

    n_q: int = 0
    n_v: int = 0

    #: Observation dims BEYOND qpos+qvel: task state the simulator does not hold (push's
    #: goal, basketball's stage flag). 0 on the four exploitable tasks. A subclass that sets
    #: it must implement `_extra_state` / `_restore_extra` as a pair -- the tail is read
    #: off upstream's task object and written back into it, never stored on this class.
    extra_dims: int = 0

    #: `assets/robots/h1hand_pos.xml` `timestep=0.002` and `Task.frame_skip = 10`, so one
    #: control step is 20 ms (50 Hz) and a 1000-step episode is 20 s of simulated time.
    dt = 0.02
    frame_skip = 10

    #: Bigger than upstream's 256x256 default for the reason `humanoid.py` states: `rda`
    #: and `gt_*` grade candidates off these frames, and a humanoid at 256 px is
    #: materially less legible than the cart-pole this repo's other adapters render.
    render_width = 448
    render_height = 320

    exact_states = None

    def __init__(self) -> None:
        chosen = os.environ.get("MUJOCO_GL") or _default_mujoco_gl()
        if chosen:
            os.environ["MUJOCO_GL"] = chosen
        _preload_llvm_before_mujoco()

        try:
            import gymnasium as gym
            import humanoid_bench  # noqa: F401  (registers the gym ids on import)
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise ImportError(
                f"problem.env_id: {self.name} needs HumanoidBench, which is not "
                "installed. It is NOT on PyPI and it cannot share an interpreter with "
                "the metaworld tier -- it needs mujoco==3.1.6, metaworld pins 3.3.0. "
                "In a venv of its own:\n"
                "    git clone https://github.com/carlosferrazza/humanoid-bench\n"
                "    uv pip install 'gymnasium[mujoco]>=1.1' 'mujoco==3.1.6' \\\n"
                "                   dm_control 'jax[cpu]' torch stable-baselines3\n"
                "    uv pip install --no-deps -e humanoid-bench\n"
                "or run the same method point on an env that needs no simulator:\n"
                "    problem.env_id: pendulum\n"
                f"(underlying import error: {exc})"
            ) from exc

        # NO `default_camera_config`. Each task's own `camera_name` (`cam_default`, and
        # `cam_hurdle` on Hurdle) is passed by `HumanoidEnv.__init__` to gymnasium's
        # renderer, which resolves it to a camera id ONCE at construction -- so
        # `mujoco_renderer.render(mode)` later honours it. See `render` below for why
        # this file never calls `self._env.render()`.
        self._env = self._make_env()

        if getattr(self._env, "obs_wrapper", False):
            raise RuntimeError(
                f"{self.name}: humanoid_bench was constructed with obs_wrapper on. That "
                "observation is joint_angles + joint_velocities only -- it drops the "
                "free root joint, so it is not the simulator state and this adapter's "
                "stateless step() would be unsound. Construct without obs_wrapper.")

        nq, nv = int(self._env.model.nq), int(self._env.model.nv)
        if (nq, nv) != (self.n_q, self.n_v):
            raise RuntimeError(
                f"{self.name}: expected nq={self.n_q} nv={self.n_v}, got nq={nq} nv={nv}. "
                "The model changed underneath this adapter; every hard-coded state index "
                "in it, and every row of its task spec's flat_fields, is now wrong.")

        # The keyframe, read directly rather than via `reset()`. `HumanoidEnv.reset_model`
        # is `mj_resetDataKeyframe` followed by the randomness draw, so a `reset()` here
        # would capture ONE perturbed sample as the nominal pose and every later episode
        # would be perturbed twice, around a point that is not the keyframe.
        key = int(getattr(self._env, "keyframe", 0))
        self._init_qpos = np.asarray(self._env.model.key_qpos[key], dtype=float).copy()
        self._init_qvel = np.asarray(self._env.model.key_qvel[key], dtype=float).copy()

        super().__init__()          # resolves the task spec, then EnvAdapter.__init__

        if self.obs_dim != nq + nv + self.extra_dims:
            raise RuntimeError(
                f"{self.name}: the task spec declares obs_dim={self.obs_dim} but the model "
                f"has nq+nv={nq + nv} and the class declares extra_dims={self.extra_dims}. "
                "The observation is qpos ++ qvel ++ extra, so those must agree or every "
                "flat_fields index is off.")

    # -- subclass hooks ------------------------------------------------------

    def _make_env(self) -> Any:
        """The unwrapped `HumanoidEnv`, constructed through upstream's own gym registration.

        `gym.make(gym_id)` and `.unwrapped`, kept as a separate method so a subclass
        can construct the environment another way without touching `__init__`.
        Everything after construction
        (the obs_wrapper refusal, the nq/nv check, the keyframe read) is shared and stays
        in `__init__`.
        """
        import gymnasium as gym
        return gym.make(
            self.gym_id,
            render_mode="rgb_array",
            width=self.render_width, height=self.render_height,
        ).unwrapped

    def _build_action_set(self) -> Any:
        """A coarse finite set for `sample_transitions`' coverage.

        All 61 actuators at {-1, 0, +1} is 3**61, so this is the 122 one-actuator-at-a-time
        commands plus all-zero and both corners. Continuous control additionally accepts
        anything inside [action_low, action_high]; this exists only so the base class has
        something finite to draw from.
        """
        n = self.action_dim
        acts = [[0.0] * n, [1.0] * n, [-1.0] * n]
        for i in range(n):
            for v in (1.0, -1.0):
                a = [0.0] * n
                a[i] = v
                acts.append(a)
        return acts

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Box bounds for the sb3 observation space and `random_state` coverage.

        READ OFF THE MODEL rather than tabulated. At 69 hinges across four models a
        hard-coded `jnt_range` table would be 276 numbers maintained by hand against an asset that already
        states them, which is the drift `flat_fields` is generated to avoid.

        Free joints are unlimited in the model, so their entries are CONVENTIONS rather
        than physical facts: positions get a room-sized box, the four quaternion
        components get exactly [-1, 1] because a unit quaternion's components do, and
        velocities are wide on purpose -- anything narrower would clip a fall, and a
        fallen humanoid is the state this env spends most of early training in.

        Every entry is FINITE, and for two consumers in this file rather than for SB3:
        `random_state` draws `rng.uniform(obs_low, obs_high)`, which raises `OverflowError`
        on an infinite range, and `discretise` bins the pelvis pose through `_bin`, which
        divides by `hi - lo` -- NaN, so `ValueError`, on an infinite width, and on a
        half-infinite one every state lands silently in bin 0. SB3 itself is indifferent:
        an infinite `Box` trains bit-identically to a finite one (measured on SB3 2.9.0;
        the record is in `metaworld._bounds`).
        """
        import mujoco

        m = self._env.model
        lo = np.full(self.obs_dim, -np.inf)
        hi = np.full(self.obs_dim, np.inf)

        for j in range(int(m.njnt)):
            adr, dadr = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                lo[adr:adr + 3] = self._free_pos_low
                hi[adr:adr + 3] = self._free_pos_high
                lo[adr + 3:adr + 7], hi[adr + 3:adr + 7] = -1.0, 1.0
                lo[self.n_q + dadr:self.n_q + dadr + 3] = -20.0
                hi[self.n_q + dadr:self.n_q + dadr + 3] = 20.0
                lo[self.n_q + dadr + 3:self.n_q + dadr + 6] = -50.0
                hi[self.n_q + dadr + 3:self.n_q + dadr + 6] = 50.0
                continue
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_BALL):
                # Four quaternion slots and three angular rates -- window's wiper head
                # is the tier's first. Left to the scalar path below, 3/4 quaternion
                # components and 2/3 rates would stay at +/-inf, and `random_state`'s
                # uniform draw over them would raise `OverflowError` (an sb3 Box would
                # accept them either way -- the docstring above has the record). A unit
                # quaternion's components live in exactly [-1, 1]; the ball's
                # `jnt_range` is a SWING limit, not a per-component bound, so it is
                # deliberately not used here.
                lo[adr:adr + 4], hi[adr:adr + 4] = -1.0, 1.0
                lo[self.n_q + dadr:self.n_q + dadr + 3] = -40.0
                hi[self.n_q + dadr:self.n_q + dadr + 3] = 40.0
                continue
            if m.jnt_limited[j]:
                lo[adr], hi[adr] = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            else:                                   # no hinge here is unlimited today
                lo[adr], hi[adr] = -np.pi, np.pi
            lo[self.n_q + dadr], hi[self.n_q + dadr] = -40.0, 40.0

        if self.extra_dims:
            xlo, xhi = self._extra_bounds()
            lo[self.n_q + self.n_v:] = xlo
            hi[self.n_q + self.n_v:] = xhi

        return lo, hi

    def _extra_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Box bounds for the `extra_dims` tail. Required exactly when `extra_dims > 0` --
        a tail left at +/-inf would make every `random_state` draw raise (`rng.uniform`
        over an infinite range). The sb3 Box itself would not mind: see `_bounds`."""
        raise NotImplementedError

    #: The world box a free body's position is sampled in, per task. A convention, and
    #: the only per-task input `_bounds` needs.
    _free_pos_low: Tuple[float, float, float] = (-5.0, -5.0, -0.5)
    _free_pos_high: Tuple[float, float, float] = (5.0, 5.0, 3.0)

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        """HumanoidBench's initial distribution, drawn from the RNG this adapter is handed.

        `HumanoidEnv.reset_model`: `mj_resetDataKeyframe`, then
        `U(-randomness, randomness)` on every position coordinate with velocities left at
        zero, then the task's own `reset_model`. `DEFAULT_RANDOMNESS` is 0.01; HighBar sets
        it to 0 in its own `reset_model` and `_randomness` reproduces that per subclass.

        The keyframe perturbs the quaternion componentwise and does not renormalise it.
        That is upstream's behaviour and is reproduced rather than corrected, because a
        different initial distribution is a different environment.

        THE STATE IS READ BACK OUT OF THE SIMULATOR, NOT RETURNED AS CONSTRUCTED.
        `set_state` runs `mj_forward`, which normalises the free joints' quaternions, so
        the numbers handed in are not the numbers the simulator then holds (`humanoid.py`
        measured 0.99027 in, 0.99977 out). Returning the pre-normalisation array would
        make `reset()` emit a state this env cannot be in, and every consumer that assumes
        `reset()` and `step()` speak the same language -- `sample_transitions`, the replay
        buffer, a stored initial condition -- would start off the manifold. Upstream's
        `get_obs` reads `data` for the same reason.

        WHY THE RNG IS OURS AND NOT THE SIMULATOR'S. `ctx.rng` alone must reproduce an
        episode. That also repairs one upstream defect by construction rather than by
        choice: `Room.reset_model` draws from the GLOBAL `np.random`, so upstream's own
        object placement is not reproducible from a seed at all.
        """
        r = float(self._randomness)
        qpos = self._init_qpos + (rng.uniform(-r, r, size=self.n_q) if r > 0 else 0.0)
        qvel = self._init_qvel.copy()
        qpos, qvel = self._task_reset_state(np.asarray(qpos, dtype=float), qvel, rng)
        self._env.set_state(qpos, qvel)
        # A reset must not carry the previous episode into what a controller reads at t=0.
        # `set_state` forwards from whatever `ctrl` and `qacc_warmstart` the last step left in
        # `MjData`, so the contact, constraint and actuator-force arrays after this call depended
        # on the episode before it while the observation did not. MEASURED without it: 31
        # MjData arrays differed between "seed 13 alone" and "seed 13 after seed 12", a stand
        # law reading foot contact forces at t=0 chose a first action 0.17 different, and
        # eval_policy's per-seed rows depended on the seeds that ran before them. `_step`
        # already re-sets ctrl and zeroes the warm-start every step (the line above
        # `do_simulation`), so dynamics are pure in (s, a); only the t=0 read would leak.
        # Zeroing here makes a reset from any history identical to a reset from a
        # fresh adapter, which is what every record in policies/ implicitly claims.
        import mujoco  # noqa: PLC0415 -- lazy, like every other simulator import in this file
        self._env.data.ctrl[:] = 0.0
        self._env.data.qacc_warmstart[:] = 0.0
        mujoco.mj_forward(self._env.model, self._env.data)
        # `cfrc_ext`/`cfrc_int` are filled by `mj_rnePostConstraint`, which `mj_step` runs and
        # `mj_forward` does not, so without this line they would still be the previous
        # episode's last-step contact wrenches (measured: the one array still differing).
        mujoco.mj_rnePostConstraint(self._env.model, self._env.data)
        return self._observe()

    #: `HumanoidEnv.DEFAULT_RANDOMNESS`. HighBar overrides it to 0.
    _randomness: float = 0.01

    def _task_reset(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The task's own `reset_model`, where it has one. Identity by default."""
        return qpos

    def _task_reset_state(self, qpos: np.ndarray, qvel: np.ndarray,
                          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """The task's own `reset_model` where it touches VELOCITIES too (basketball
        throws the ball). Defaults to the position-only hook so the four exploitable tasks
        keep their `_task_reset` overrides untouched."""
        return self._task_reset(qpos, rng), qvel

    # -- the extra-state tail (push's goal, basketball's stage) ---------------

    def _extra_state(self) -> np.ndarray:
        """The `extra_dims` tail of the observation, read off upstream's task object.
        Empty on the four exploitable tasks."""
        return _NO_EXTRA

    def _restore_extra(self, tail: np.ndarray) -> None:
        """Write the tail back into upstream's task object so the reward, termination
        and render see the task state the observation carries. No-op when
        `extra_dims == 0`."""

    def _before_step(self) -> None:
        """Task-side latch update on the RESTORED state, for latches whose ADVANCE PAYS
        a one-off reward inside `get_reward` (maze's +100*stage, cabinet's and
        bookshelf's +100*index).

        Why this is a different hook from `_after_step`, and not a style choice. For
        those tasks the observation tail must carry the latch as it stood BEFORE the
        reward's own update, because `reference_reward(s)` reproduces upstream's reward
        by calling `get_reward` -- which re-runs the update from whatever the tail
        restored, and pays the advance bonus exactly when the restored latch is the
        pre-advance one. A tail holding the post-advance value (basketball's convention,
        where the latch pays nothing) would make every transition state's reference
        reward silently drop its +100*k. So these latches advance HERE, on the restored
        state, and the arriving observation carries the value upstream's step-t
        `get_reward` STARTS from -- the counter as it stood before upstream processed the
        arriving state -- into the next step. Runs after `set_state`, before
        `do_simulation`. The TERMINAL is a different reader: upstream's `get_terminated`
        sees the counter after that step's advance, so a latch task's `_terminated` is
        `_terminated_after_advance`, never the base's."""

    def _after_step(self) -> None:
        """Task-side state transition on the ARRIVING state, mirroring what upstream
        computes once per wrapper step (basketball's catch->throw latch reads the
        arriving contacts). Runs after `do_simulation`, before `_observe`."""

    def _observe(self) -> np.ndarray:
        obs = np.concatenate([self._env.data.qpos, self._env.data.qvel]).ravel()
        if self.extra_dims:
            return np.concatenate([obs, np.asarray(self._extra_state(), dtype=float).ravel()])
        return obs.copy()

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Stateless: restore `s`, apply `a`, read the arriving state.

        THE `qacc_warmstart` LINE IS LOAD-BEARING AND IS NOT HYGIENE. MuJoCo seeds its
        constraint solver from `data.qacc_warmstart`; `set_state` does not reset it and
        neither does `mj_forward`, so without this line the function is not a function --
        it returns different values for the same arguments depending on what was stepped
        before. `humanoid.py` measured 115/300 mismatches and 4.8e-1 of deviation in a
        joint coordinate on `h1-crawl-v0`, thirteen orders of magnitude worse than
        `InvertedPendulum-v5`, because a body in persistent contact gives the solver a large
        active set on every step. All four tasks here are at least as contact-rich --
        `room` has six loose bodies resting on a floor -- so the same argument applies with
        more force. Measure it with SHUFFLED replay, never round-trips: a round-trip
        re-steps the state it just arrived at, so the warmstart is already the right one
        and the defect is invisible.

        `a` goes through the task's own `unnormalize_action` because the H1 is built with
        `control="pos"`: the [-1, 1] a policy emits is an affine encoding of each
        actuator's `ctrlrange`, not a torque, and a raw `ctrl` value would be wrong by
        that transform.

        `done` is UPSTREAM'S OWN `get_terminated`, called rather than reimplemented. All
        four of these tasks terminate on failure and the terminal is deliberately kept --
        see the module docstring; on `highbar` the incentive it creates is the object of
        study.
        """
        s = np.asarray(s, dtype=float).ravel()
        u = np.clip(np.asarray(a, dtype=float).ravel(), -1.0, 1.0)
        self._restore_extra(s[self.n_q + self.n_v:])
        self._env.set_state(s[:self.n_q], s[self.n_q:self.n_q + self.n_v])
        self._env.data.qacc_warmstart[:] = 0.0
        self._before_step()
        self._env.do_simulation(self._env.task.unnormalize_action(u), self.frame_skip)
        self._after_step()
        s2 = self._observe()
        finite = bool(np.isfinite(s2).all())
        done = (not finite) or self._terminated()
        return s2, done, {"success": finite and self._success_flag(s2)}

    def _terminated(self) -> bool:
        """Upstream's own `get_terminated`, on the arriving state, called not ported.

        A `_before_step` latch task overrides this (`_terminated_after_advance`): its
        terminal reads the counter its `get_reward` advances, and upstream reads it AFTER
        that advance on the arriving state.
        """
        return bool(self._env.task.get_terminated()[0])

    def _terminated_after_advance(self) -> bool:
        """Upstream's terminal, read where upstream reads it. `Task.step` is
        `do_simulation -> get_reward -> get_terminated`, so the counter `get_terminated`
        sees has been advanced on the ARRIVING state. `_before_step` advanced it on the
        RESTORED state so the observation tail stays pre-advance (its contract); that
        observation has been taken by the time this runs, so advancing again here is a
        PEEK -- the next `_restore_extra` (in `_step`, `reference_reward`, `render`)
        overwrites it. Reading the terminal with the counter one advance behind
        upstream's would run the episode one step longer, carry the terminal value in its
        extra state and make its `reference_reward` upstream's never-reached final arm (a
        completed cabinet's `reference_return` would read 2004 against upstream's 1004),
        and bookshelf's drop clause -- `xpos[objects[task_index], z] < 0.5` -- would name
        the object JUST PLACED, so a placement at either bottom-shelf goal (z = 0.35)
        would end the episode as a drop where upstream advances. Checked by reading
        against the fetched upstream source, not by stepping the real model.
        `tests/test_humanoid_hand.py` drives this and a transcription of `Task.step` over
        one scripted sequence and asserts agreement."""
        with contextlib.redirect_stdout(io.StringIO()):      # cabinet's "Completed subtask"
            self._env.task.get_reward()
        return bool(self._env.task.get_terminated()[0])

    def _success_flag(self, s: np.ndarray) -> bool:
        """The per-step ground-truth flag `step()` puts in `info["success"]`.

        Per task, and never the shipped reward's opinion. Defined here so that a subclass
        that forgets one fails loudly rather than reporting False for a whole run.
        """
        raise NotImplementedError

    def discretise(self, s: np.ndarray) -> int:
        """A coarse binning over the pelvis pose only -- a binning, not a bijection, hence
        `exact_states is None`. Present because `EnvAdapter` requires it; no shipped
        config selects `train.backend: tabular`, and a 151-to-229-D tabular index would
        be meaningless if one did."""
        s = np.asarray(s, dtype=float).ravel()
        lo, hi = self.obs_low, self.obs_high
        idx = 0
        for i, n in enumerate((12, 5, 5)):
            idx = idx * n + _bin(float(s[i]), float(lo[i]), float(hi[i]), n)
        return idx

    n_disc_states = 12 * 5 * 5

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        """Coverage over the box, not the initial distribution -- EPIC/STARC need the whole
        space, not the on-policy slice. EVERY quaternion is renormalised, free joints' and
        ball joints' alike, not just the pelvis's: `room` has six more free bodies and
        `window`'s wiper head is a ball joint, and a non-unit quaternion makes forward
        kinematics meaningless."""
        import mujoco

        s = np.asarray(rng.uniform(self.obs_low, self.obs_high), dtype=float)
        m = self._env.model
        for j in range(int(m.njnt)):
            adr = int(m.jnt_qposadr[j])
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                s[adr + 3:adr + 7] = _quat_normalise(s[adr + 3:adr + 7])
            elif int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_BALL):
                s[adr:adr + 4] = _quat_normalise(s[adr:adr + 4])
        return s

    # -- ground truth --------------------------------------------------------

    def task_metric(self, traj: Any) -> float:
        """Ground-truth fitness in [0, 1]. A STATED PROXY on all four tasks -- read the
        module docstring's section on it before quoting a number."""
        raise NotImplementedError

    # --- BEGIN reference reward (ground truth; strip before showing an LLM) ---
    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """HumanoidBench's own shipped reward for this task, evaluated by CALLING it.

        IT IS NOT AN EXPERT BASELINE HERE, and that is the whole reason these four tasks
        were selected. `Crawl`'s shipped reward is a genuine hand-tuned expert cost
        (`humanoid.py`); each reward reached through this method is instead a reward with
        a documented failure mode, recorded in the task spec's `exploits`. Keep the
        distinction when reading anything computed from it:

          * `gt_return` on this tier is "return under the reward being criticised",
            not "return under a good reward".
          * `evaluate.similarity.metric` against `gt_reward` measures how closely a
            candidate reproduces THE EXPLOITABLE REWARD. A high value is a diagnosis, not
            a quality score, and reading it as one inverts the finding.

        CALLED, NOT PORTED, for the reason `humanoid.py` gives: these rewards read
        `data.actuator_force`, `named.data.xpos` and, on `hurdle`, the live contact list --
        none of which is in the state vector, so a reimplementation would need forward
        kinematics anyway and would be a second copy free to drift.

        Mutating `self._env` here is safe for the SCORE because `_step` restores the state
        vector it is given and zeroes `qacc_warmstart` on every call, so the NEXT physics step
        is bit-identical whether or not this was called in between (probed on mujoco
        3.3.0: same `qpos`/`qvel` after a step with and without an interleaved call).
        That is not the same as saying a ROLLOUT is unaffected. MEASURED: summing this
        reward between the steps of a rollout moved contact-rich task scores of scripted
        policies on identical seeds. The physics did not diverge on its own; the likely residue -- unmeasured -- is what a rig-based policy
        reads from the live `mjData` between steps: this call's `set_state` + `mj_forward`
        re-derives the contact, constraint and actuator-force arrays from the same state
        instead of leaving the step's own contact solve in place, and a controller gating on
        those fields (a seat window on contact forces, a double-support freeze on foot loads)
        then acts differently. Whatever the mechanism, the rule is the same: never call this
        inside a rollout loop; record states and actions and sum it AFTERWARDS, which is what
        `bird.policy_api.run_episode` does for `reference_return`. If `_step` ever stops restoring
        the state vector, even the score becomes cross-contamination between the rollout and
        its evaluation.

        With `a` omitted the effort term is evaluated at the neutral command -- the
        midpoint of every actuator's range -- rather than at zero force, since these are
        position servos and there is no "no effort" command.
        """
        import mujoco

        s = np.asarray(s, dtype=float).ravel()
        u = (np.zeros(self.action_dim) if a is None
             else np.clip(np.asarray(a, dtype=float).ravel(), -1.0, 1.0))
        self._restore_extra(s[self.n_q + self.n_v:])
        self._env.set_state(s[:self.n_q], s[self.n_q:self.n_q + self.n_v])
        self._env.data.qacc_warmstart[:] = 0.0
        self._env.data.ctrl[:] = self._env.task.unnormalize_action(u)
        mujoco.mj_forward(self._env.model, self._env.data)
        return float(self._env.task.get_reward()[0])
    # --- END reference reward ---

    # -- rendering -----------------------------------------------------------

    def render(self, state: np.ndarray, width: int = 320) -> np.ndarray:
        """One `(render_height, render_width, 3)` uint8 frame of the simulator restored to
        `state`.

        `width` is accepted and ignored -- the renderer captured its size at construction
        and `observability._resize_nearest` scales to `output.video.width`. Exactly one
        positional parameter beyond `self`, which is what `observability._accepts_a_state`
        requires.

        DO NOT CALL `self._env.render()`. `HumanoidEnv.render` delegates to
        `humanoid_bench.tasks.Task.render`, which calls
        `mujoco_renderer.render(mode, camera_id, camera_name)` -- the gymnasium 0.29
        signature. On gymnasium 1.x that is a `TypeError`. Going to the renderer directly
        keeps the fix here rather than in a patch to a vendored clone somebody will
        re-pull, and the camera still comes out right because `MujocoRenderer.__init__`
        resolved `camera_name` to an id once, at construction.
        """
        s = np.asarray(state, dtype=float).ravel()
        self._restore_extra(s[self.n_q + self.n_v:])
        self._env.set_state(s[:self.n_q], s[self.n_q:self.n_q + self.n_v])
        self._env.data.qacc_warmstart[:] = 0.0
        self._mj_forward()
        self._pre_render()
        self._make_gl_current()
        return np.asarray(self._env.mujoco_renderer.render(self._env.render_mode),
                          dtype=np.uint8)

    def _make_gl_current(self) -> None:
        """Make THIS env's off-screen GL context current before drawing into it.

        gymnasium's `OffScreenViewer` calls `make_current()` once, in its constructor, and
        never again -- so the moment a second MuJoCo env is constructed in the same process
        (a second adapter, a `gym.make` beside it, an extra-view renderer) that env's context
        is the current one and every later frame from the first env is drawn into the wrong
        context and comes back ALL ZEROS. Measured: with two adapters alive, the second's
        `render()` returned a uniform black frame right after a
        `gym.make` of the same id; alone it rendered. `mujoco.Renderer` makes its own context
        current on every `render()`; this does the same for the gymnasium path.
        """
        try:
            viewer = self._env.mujoco_renderer._get_viewer(self._env.render_mode)
            ctx = getattr(viewer, "opengl_context", None)
            if ctx is not None:
                ctx.make_current()
        except Exception as exc:  # noqa: BLE001 - a renderer without this API renders as before
            log.debug("%s: could not make the GL context current (%s)", self.name, exc)

    def render_view(self, state: np.ndarray, view: View) -> np.ndarray:
        """One `(render_height, render_width, 3)` uint8 frame of `state` from an extra
        viewpoint -- the same restore as `render`, then `envs.cameras.MujocoViews`.

        Two kinds of view reach here. A MODEL camera (`cam_hurdle`, `cam_tabletop`,
        `cam_hand_visible`, ... -- `h1hand_pos.xml` defines seven `trackcom` cameras on
        the pelvis and every task's asset inherits them) follows the robot exactly as
        `cam_default` does. A POSED camera (`pose: {track_body: pelvis, azimuth,
        elevation, distance}`) is built at render time, which is how a task gets a
        top-down or front-on view the asset never defined without editing the vendored
        XML. `_pre_render` runs for both, so push's goal marker is in every panel.
        """
        s = np.asarray(state, dtype=float).ravel()
        self._restore_extra(s[self.n_q + self.n_v:])
        self._env.set_state(s[:self.n_q], s[self.n_q:self.n_q + self.n_v])
        self._env.data.qacc_warmstart[:] = 0.0
        self._mj_forward()
        self._pre_render()
        return self._views().render(view)

    def _views(self) -> MujocoViews:
        """The extra-view renderer, built on first use (a second GL context)."""
        views = getattr(self, "_extra_view_renderer", None)
        if views is None:
            import mujoco
            views = MujocoViews(mujoco, self._env.model, self._env.data,
                                self.render_height, self.render_width)
            self._extra_view_renderer = views
        return views

    def _pre_render(self) -> None:
        """Task-side scene dressing before the frame is taken (push's goal marker, which
        upstream draws only through its own `Task.render` -- the path this file bypasses
        for the gymnasium-signature reason above)."""

    def _mj_forward(self) -> None:
        """`set_state` already calls `mj_forward`; this is here so the render path says so
        explicitly rather than depending on that remaining true."""
        import mujoco
        mujoco.mj_forward(self._env.model, self._env.data)

    def task_images(self) -> List[bytes]:
        """PNG bytes of the reset state, for `problem.instruction_modality: text+image`.
        Returns `[]` when imageio or GL is missing, because an instruction modality that
        cannot be honoured must degrade to text rather than abort a run."""
        try:
            import imageio.v3 as iio
            frame = self.render(self.reset(0))
            return [bytes(iio.imwrite("<bytes>", frame, extension=".png"))]
        except Exception as exc:  # noqa: BLE001 - optional dep or missing GL context
            log.debug("%s: task_images unavailable (%s)", self.name, exc)
            return []

    # -- what the task asks, as a short label (`EnvAdapter.legend_lines`) -------

    #: The task's FIXED ask, as the line(s) `legend_lines` opens with: the command
    #: ("WALK +X AT 1.0 M/S"), the fixed goal ("HOOP AT X 5.9 Z 2.25"), the route
    #: ("MAZE: 3,0 THEN 3,6 THEN 6,6"). Declared by every concrete class the way each
    #: already declares `gym_id` -- there is no table of task names to branch on and
    #: this file must not grow one. Where the number IS the task's published constant
    #: (`_LOCO_V_REF`, `_RUN_V_REF`, `_POLE_V_REF`, `_MAZE_CHECKPOINTS`, `_HOOP_POS`,
    #: `_HURDLE_WALLS_X`) the line is built from it, so the legend cannot say 1.0 m/s
    #: while the metric normalises by 5.
    #:
    #: Every line is upper-case, at most 28 characters, and drawn only from the
    #: characters `bird/envs/hud.py`'s font has (`hud._SHEET_CHARS` -- digits, A-Z,
    #: `. - + / : % < > = [ ] ? ,` and space), pinned by `tests/test_humanoid_hand.py`
    #: on every class.
    _ask: Tuple[str, ...] = ()

    def legend_lines(self, state: np.ndarray) -> List[str]:
        """`_ask`, then whatever the observation TAIL says the task asks right now.

        THE ASK, NEVER THE ANSWER. `EnvAdapter.legend_lines` states the rule; what it
        means on this tier is concrete. Push's per-episode goal (`s[164:167]`), reach's target,
        package's destination, bookshelf's current placement spot: those are what the
        policy was TOLD, and they go in the label. Basketball's stage latch and
        cabinet's ladder rung select which command is in force, so the command they
        select is named ("CATCH THE BALL" / "THROW TO THE HOOP", "NOW: PULL THE DRAWER
        OUT"). Barbell height, distance to the goal, whether the pelvis is past the
        door, how many checkpoints fell: those are what `task_metric` computes, and none
        appears here, because these frames are the input to `evaluate.fitness.source:
        vlm_score` and a legend carrying the verdict would make that score a copy of
        the ground truth it is meant to be independent of.

        Three tails this deliberately does NOT restate. Maze's stage, because
        `_restore_extra` already paints the junction markers green/red from the same
        number, so the frame carries it as upstream draws it. Cube's target quaternion,
        because the target cube is rendered in the scene at exactly that orientation,
        and an Euler triple is neither legible at a glance nor unique (double cover).
        Window's `head_pos0`, which is not an ask at all -- it is the reward's own
        anchor -- and is left out on that ground.

        A PURE FUNCTION OF `state`. Every number comes from the array, through
        `_legend_extra(tail)` -- never from `self._env`, whose task object holds
        whatever the last restore left there, and this may be called with no restore
        before it. That is also what lets `tests/test_humanoid_hand.py` check the legend
        on an adapter built with `object.__new__`, in the interpreter that cannot build
        the model. `render` here returns the renderer's frame untouched: it draws no
        label.
        """
        s = np.asarray(state, dtype=float).ravel()
        return list(self._ask) + self._legend_extra(s[self.n_q + self.n_v:])

    def _legend_extra(self, tail: np.ndarray) -> List[str]:
        """The per-episode or per-phase ASK the `extra_dims` tail carries, as label
        lines -- `_restore_extra`'s reader, for `legend_lines`. Empty by default, and empty
        on purpose for a tail that is not an ask (window's reward anchor, maze's
        painted stage, cube's rendered target)."""
        return []


# --------------------------------------------------------------------------
# powerlift
# --------------------------------------------------------------------------


class H1HandPowerlift(_H1HandBase):
    """Lift a 104 kg barbell off the floor. `h1hand-powerlift-v0`.

    The robot weighs 53 kg and the barbell weighs twice that, which is worth knowing
    before reading a flat learning curve as a reward problem.
    """

    name = "h1hand_powerlift"
    gym_id = "h1hand-powerlift-v0"
    obs_dim = 164
    action_dim = 61
    horizon = 1000
    n_q, n_v = 83, 81
    _ask = ("LIFT THE BARBELL, HOLD IT UP",)

    #: A pelvis height, not a barbell height. See `task_metric`.
    success_threshold = 0.1

    _free_pos_low = (-3.0, -3.0, -0.5)
    _free_pos_high = (3.0, 3.0, 3.0)

    def task_metric(self, traj: Any) -> float:
        """Fraction of arriving steps with the barbell's centre above the pelvis while the
        robot is still standing.

        RELATIVE, so it borrows no constant from the reward it has to be independent of.
        The barbell rests at z = 0.2 m and a standing pelvis is at 0.98 m, so the
        do-nothing policy the shipped reward pays ~0.2 per step for scores exactly 0 here.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        arrived = states[1:]
        up = arrived[:, 78] > arrived[:, 2]
        standing = arrived[:, 2] >= _POWERLIFT_STANDING_Z
        return float(np.mean(np.logical_and(up, standing)))

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(s[78] > s[2] and s[2] >= _POWERLIFT_STANDING_Z)


# --------------------------------------------------------------------------
# hurdle
# --------------------------------------------------------------------------


class H1HandHurdle(_H1HandBase):
    """Run a walled corridor crossed by ten low walls. `h1hand-hurdle-v0`."""

    name = "h1hand_hurdle"
    gym_id = "h1hand-hurdle-v0"
    obs_dim = 151
    action_dim = 61
    horizon = 1000
    n_q, n_v = 76, 75
    _ask = (f"RUN +X, CLEAR THE {len(_HURDLE_WALLS_X)} HURDLES",)

    #: One hurdle out of ten. PROVISIONAL like every threshold in this file.
    success_threshold = 0.1

    _free_pos_low = (-2.0, -5.0, -0.5)
    _free_pos_high = (75.0, 5.0, 3.0)

    def task_metric(self, traj: Any) -> float:
        """The fraction of the ten hurdle walls the pelvis's high-water mark passed.

        HIGH-WATER, not final: a hurdle jumped and then fallen back behind was still
        cleared, and the alternative would score a robot that stumbles after hurdle three
        below one that never left the start.

        No corridor gate is needed -- unlike `crawl`'s tunnel, the side barriers here are
        real collision geometry 4.725 m out, so a policy physically cannot walk around the
        course. The resolution is deliberately coarse (0.1): this counts a task-defined
        event rather than smoothing distance, and a robot braced against the first hurdle
        -- the documented exploit -- reads 0.0 rather than "nearly there".
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        reach = float(np.max(states[1:, 0]))
        return float(sum(reach > x for x in _HURDLE_WALLS_X)) / len(_HURDLE_WALLS_X)

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(s[0] > _HURDLE_WALLS_X[0])


# --------------------------------------------------------------------------
# room
# --------------------------------------------------------------------------

#: `assets/tasks/room.xml`, in the order `Room.get_reward` lists them. Each is a free
#: body, so its 7 qpos start at 76 + 7k of the flat observation.
_ROOM_OBJECTS: Tuple[str, ...] = ("chair", "trophy", "headphone",
                                  "package_a", "package_b", "snow_globe")
_ROOM_OBJ_Q0 = 76


class H1HandRoom(_H1HandBase):
    """Tidy six loose objects in a walled room containing a shelf and a table.
    `h1hand-room-v0`."""

    name = "h1hand_room"
    gym_id = "h1hand-room-v0"
    obs_dim = 229
    action_dim = 61
    horizon = 1000
    n_q, n_v = 118, 111
    #: "Tidy" here is the spec's stated proxy -- the furniture -- so the ask says which
    #: furniture, not merely "tidy".
    _ask = (f"TIDY: PUT THE {len(_ROOM_OBJECTS)} OBJECTS", "ON THE SHELF OR THE TABLE")

    #: One object out of six. PROVISIONAL, and expected to be unreachable -- see the
    #: module docstring on floor saturation.
    success_threshold = 0.16

    _free_pos_low = (-4.0, -4.0, -0.5)
    _free_pos_high = (4.0, 4.0, 3.0)

    def _task_reset(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`Room.reset_model`, reproduced INCLUDING its off-by-one.

        Upstream scatters the loose objects with

            for i in range(-7, 0):
                position[i * 7]     = U(-3.5 + (i + 7), -3.5 + (i + 8))
                position[i * 7 + 1] = U(1.2, 3.5) * choice([1, -1])
                if i == -4: position[i*7+3 : i*7+7] = <a fixed quaternion>

        There are SIX loose bodies and the loop runs SEVEN times. `nq` is 118, so the
        first iteration writes `position[-49]` and `position[-48]`, which are indices 69
        and 70 -- the right hand's `rh_LFJ2` and `rh_LFJ1`, both limited to [0, 1.571] rad
        and both set here to roughly -3 rad and +/-[1.2, 3.5] rad. Every episode therefore
        begins with the right little finger folded far outside its own range.

        REPRODUCED, NOT CORRECTED, on `humanoid.py`'s rule: a different initial
        distribution is a different environment, and quietly fixing it would make this
        adapter's numbers incomparable with anything anyone else runs on `h1hand-room-v0`.
        It is recorded in `tasks/h1hand_room/shared_spec.yaml` and pinned by
        `tests/test_humanoid_hand.py` so that a future upstream fix shows up as a failing
        test rather than as a silent change of environment.

        The one deliberate departure is the RNG: upstream draws from the global
        `np.random`, so its placement is not reproducible from a seed. This draws from
        `ctx.rng` and is.
        """
        q = np.asarray(qpos, dtype=float).copy()
        n = q.shape[0]
        for i in range(-7, 0):
            base = n + i * 7
            q[base] = rng.uniform(-3.5 + (i + 7), -3.5 + (i + 8))
            q[base + 1] = rng.uniform(1.2, 3.5) * (1.0 if rng.random() < 0.5 else -1.0)
            if i == -4:
                q[base + 3:base + 7] = np.array(
                    [0.0733422, 0.0519076, -0.240058, -0.966591])
        return q

    def task_metric(self, traj: Any) -> float:
        """The fraction of the six loose objects that finished stowed on the furniture.

        THE DEFINITION OF "TIDY" IS OURS AND THE SPEC SAYS SO. HumanoidBench ships a room
        with a shelf and a table in it and never states that either is a target; the only
        criterion it ships is the reward's variance term, and adopting that would make the
        metric a function of the reward being criticised. Stowing on the furniture is the
        least arbitrary reading the scene supports, and it is exactly what the heap
        exploit fails: six objects pushed into one pile on the floor drive the variance to
        zero and score 0.0 here.

        Expected to read 0.000 for anything current methods produce. That is recorded
        rather than mitigated -- see the module docstring.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        final = states[-1]
        return float(sum(_stowed(final[_ROOM_OBJ_Q0 + 7 * k:_ROOM_OBJ_Q0 + 7 * k + 3])
                         for k in range(len(_ROOM_OBJECTS)))) / len(_ROOM_OBJECTS)

    def _success_flag(self, s: np.ndarray) -> bool:
        return any(_stowed(s[_ROOM_OBJ_Q0 + 7 * k:_ROOM_OBJ_Q0 + 7 * k + 3])
                   for k in range(len(_ROOM_OBJECTS)))


def _stowed(pos: np.ndarray) -> bool:
    """Is this object's centre over the shelf, or over the table top?

    Module level so it stays out of `describe("full_source")`, which renders the CLASS.
    The z bound differs between the two: the shelf's lowest board is 0.05 m off the floor,
    so anything above it is on a shelf, while the table's top surface is at 0.95 m and a
    footprint test alone would count an object shoved UNDER the table as tidied.
    """
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    for x0, x1, y0, y1, zmin in (_SHELF_BOX, _TABLE_BOX):
        if x0 <= x <= x1 and y0 <= y <= y1 and z >= zmin:
            return True
    return False


# --------------------------------------------------------------------------
# highbar
# --------------------------------------------------------------------------


class H1StrongHighBarHard(_H1HandBase):
    """Hang from a high bar and rotate around it. `h1strong-highbar_hard-v0`.

    `h1strong` is the same Shadow-Hands robot as `h1hand` with `kp=50` /
    `forcerange="-50 50"` finger servos instead of the stock 0.4-8 gains, which is what
    lets it hold its own weight. There is no `h1hand` highbar asset -- see the module
    docstring, because the id that does not exist is the one everyone reaches for first.
    """

    name = "h1strong_highbar_hard"
    gym_id = "h1strong-highbar_hard-v0"
    obs_dim = 151
    action_dim = 61
    horizon = 1000
    n_q, n_v = 76, 75
    _ask = ("SWING ALL THE WAY AROUND", "THE HIGH BAR")

    #: Half a turn -- the body carried above horizontal on the far side of the bar.
    #: PROVISIONAL.
    success_threshold = 0.5

    #: NO initial perturbation: the keyframe places the hands on the bar and a
    #: perturbation large enough to see would drop the robot off it.
    #:
    #: A KNOWN AND DELIBERATE DIVERGENCE FROM UPSTREAM, in the only direction that keeps
    #: `reset()` a function. `HighBarBase.reset_model` sets `self._env.randomness = 0`,
    #: but it runs AFTER `HumanoidEnv.reset_model` has already drawn the perturbation --
    #: so upstream's FIRST episode on a fresh env is perturbed by 0.01 and every episode
    #: after it is not. Reproducing that would make the initial state depend on how many
    #: resets preceded it, which is exactly the impurity `_step`'s `qacc_warmstart` line
    #: exists to remove and which `tests/test_resume.py` and `tests/test_parallelism.py`
    #: both rest on. This adapter takes the steady state -- upstream's episodes 2..N --
    #: for every episode, and says so here rather than in a comment nobody reads.
    _randomness = 0.0

    _free_pos_low = (-3.0, -3.0, -0.5)
    _free_pos_high = (3.0, 3.0, 4.0)

    def task_metric(self, traj: Any) -> float:
        """|net rotation| of the pelvis about the bar, in turns, clipped at one.

        The angle is `atan2(z - 2.80, x - 0.31)` in the x-z plane, unwrapped across steps
        and SUMMED SIGNED before the absolute value is taken. Signed is the whole point:
        a robot swinging out and back accumulates nothing, and only going round the bar
        adds up. A motionless hang -- the documented exploit, and the behaviour the
        shipped `upright * feet * small_control` reward pays full price for -- scores
        exactly 0.0.

        No "still on the bar" gate is needed. `HighBarBase.get_terminated` ends the
        episode as soon as the head drops below 2.0 m, so a robot that lets go stops
        producing states rather than sweeping a meaningless angle.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        theta = np.arctan2(states[:, 2] - _BAR_Z, states[:, 0] - _BAR_X)
        net = float(np.sum(np.diff(np.unwrap(theta))))
        return float(min(abs(net) / (2.0 * np.pi), 1.0))

    def _success_flag(self, s: np.ndarray) -> bool:
        """Pelvis above the bar: the top of a giant swing, which a hang never reaches."""
        return bool(s[2] > _BAR_Z)


# --------------------------------------------------------------------------
# push
# --------------------------------------------------------------------------


class H1HandPush(_H1HandBase):
    """Push a 0.2 m, 0.3 kg box across a table to a per-episode target.
    `h1hand-push-v0`.

    The first task in this file whose observation is MORE than the simulator state:
    the goal is drawn per episode, lives on upstream's task object, and rides in
    s[164:167] -- read the module docstring's PUSH AND BASKETBALL section before
    touching any index here. Upstream draws the goal from the GLOBAL `np.random`
    (Room's defect exactly); this adapter draws it from `ctx.rng`, so unlike
    upstream's, ours is reproducible from a seed.
    """

    name = "h1hand_push"
    gym_id = "h1hand-push-v0"
    obs_dim = 167
    action_dim = 61
    horizon = 500
    n_q, n_v = 83, 81
    _ask = ("PUSH THE BOX TO THE GOAL",)
    extra_dims = 3

    #: Best approach halved the initial box-goal gap. PROVISIONAL like every
    #: threshold in this file.
    success_threshold = 0.5

    _free_pos_low = (-3.0, -3.0, -0.5)
    _free_pos_high = (3.0, 3.0, 3.0)

    def _extra_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(_PUSH_GOAL_LOW), np.array(_PUSH_GOAL_HIGH)

    def _extra_state(self) -> np.ndarray:
        return np.asarray(self._env.task.goal, dtype=float).ravel().copy()

    def _restore_extra(self, tail: np.ndarray) -> None:
        self._env.task.goal[:] = np.asarray(tail, dtype=float).ravel()

    def _task_reset(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`Push.reset_model`: goal x ~ U(0.7, 1.0), y ~ U(-0.5, 0.5), z stays 1.0.

        qpos is untouched -- the box starts wherever the keyframe perturbation left it;
        only the target moves. Note the sample region contains the box's own start, so
        ~0.65% of episodes are BORN TERMINATED (goal_dist < 0.05 at reset) -- upstream's
        behaviour, reproduced rather than corrected, and recorded in the spec.
        """
        goal = self._env.task.goal
        goal[0] = rng.uniform(0.7, 1.0)
        goal[1] = rng.uniform(-0.5, 0.5)
        # z is written too, and not because upstream writes it (upstream never does: its
        # goal z is 1.0 from construction and nothing else touches it). Here `_restore_extra`
        # writes WHATEVER z a carried tail holds into the same array -- a `random_state`
        # draw, or a test's goal-on-the-box state -- so without this line a reset after such
        # a restore would keep that z and `reset(seed)` would stop being a function of the
        # seed (`test_push_carries_its_goal_in_the_state...` on the real stack reads 1.0089
        # where 1.0 is expected).
        goal[2] = _PUSH_GOAL_LOW[2]
        return qpos

    def task_metric(self, traj: Any) -> float:
        """Best approach of the box to the goal, as a fraction of the initial gap --
        and 1.0 outright the moment any arriving state is inside the environment's own
        success radius, which is also the moment the episode ends.

        RELATIVE and HIGH-WATER: a do-nothing policy scores 0.0 (the box never moves),
        and shoving the box off the table scores 0.0 too (the goal sits at table height,
        so the floor is further from it than the start was). The denominator is floored
        at the success radius so an episode born solved divides by nothing.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        arrived = states[1:]
        d = np.linalg.norm(arrived[:, 76:79] - arrived[:, 164:167], axis=1)
        if bool(np.any(d < _GOAL_RADIUS)):
            return 1.0
        d0 = float(np.linalg.norm(states[0, 76:79] - states[0, 164:167]))
        return float(np.clip(1.0 - float(np.min(d)) / max(d0, _GOAL_RADIUS), 0.0, 1.0))

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(np.linalg.norm(s[76:79] - s[164:167]) < _GOAL_RADIUS)

    def _legend_extra(self, tail: np.ndarray) -> List[str]:
        """The per-episode goal, off the same three slots `_restore_extra` writes back."""
        return ["GOAL " + _legend_xyz(tail)]

    def _pre_render(self) -> None:
        """Upstream's goal marker, reproduced -- see `_paint_goal_marker` for why a
        bypassed `Task.render` makes this necessary."""
        _paint_goal_marker(self, np.asarray(self._env.task.goal, dtype=float).copy())


# --------------------------------------------------------------------------
# basketball
# --------------------------------------------------------------------------


class H1HandBasketball(_H1HandBase):
    """Catch a basketball fired at the robot, then throw it through a hoop 5.9 m away
    at 2.25 m. `h1hand-basketball-v0`.

    The reward is a TWO-STAGE machine: 0.5*stand + 0.5*hand-proximity until the ball's
    first contact, then 0.15*stand + 0.05*hand-proximity + 0.8*ball-to-hoop after. The
    stage is a Python-side latch on upstream's task object, not simulator state, so it
    rides in s[164] (0.0 catch, 1.0 throw) -- the module docstring's PUSH AND
    BASKETBALL section is the contract.
    """

    name = "h1hand_basketball"
    gym_id = "h1hand-basketball-v0"
    obs_dim = 165
    action_dim = 61
    horizon = 500
    n_q, n_v = 83, 81
    #: The fixed half of the ask: where the hoop is, off the same `_HOOP_POS` the metric
    #: measures to. The stage decides the other half -- see `_legend_extra`.
    _ask = (f"HOOP AT X {_HOOP_POS[0]:.1f} Z {_HOOP_POS[2]:.2f}",)
    extra_dims = 1

    #: Best approach halved the launch-to-hoop gap -- a real throw or carry, where the
    #: hold exploit reads ~0.0. PROVISIONAL like every threshold in this file.
    success_threshold = 0.5

    _free_pos_low = (-3.0, -5.0, -0.5)
    _free_pos_high = (7.0, 5.0, 5.0)

    def __init__(self) -> None:
        super().__init__()
        import mujoco

        #: The geom id `Basketball.get_reward` derives per call through dm_control's
        #: named indexing; resolved once here, same integer.
        self._ball_geom_id = int(mujoco.mj_name2id(
            self._env.model, mujoco.mjtObj.mjOBJ_GEOM, "basketball_collision"))

    def _extra_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(1), np.ones(1)

    def _extra_state(self) -> np.ndarray:
        return np.array([1.0 if self._env.task.stage == "throw" else 0.0])

    def _restore_extra(self, tail: np.ndarray) -> None:
        """Binarised at 0.5: the flag is 0.0 or 1.0 everywhere this adapter writes it,
        and the threshold only decides for `random_state`'s coverage draws."""
        self._env.task.stage = "throw" if float(tail[0]) >= 0.5 else "catch"

    def _legend_extra(self, tail: np.ndarray) -> List[str]:
        """The stage's own command -- the half of the ask that changes within an
        episode -- binarised at 0.5 exactly as `_restore_extra` is, so the legend and
        the reward agree about which stage a state is in."""
        return ["THROW TO THE HOOP" if float(tail[0]) >= 0.5 else "CATCH THE BALL"]

    def _after_step(self) -> None:
        """Upstream's catch->throw latch on the ARRIVING contacts.

        `Basketball.get_reward` flips `stage` when any contact pair involves the ball
        -- ANY contact, so the ball glancing off the robot's chest ends the catch
        shaping just as a clean catch does. Upstream runs this scan once per wrapper
        step inside `get_reward`; this adapter does not call `get_reward` on the hot
        path, so the same scan lives here. Monotone within an episode by construction
        -- nothing writes "catch" back except reset and `_restore_extra`.

        `data.contact.geom` IS the supported API on this tier's mujoco: an (ncon, 2)
        int array (verified on 3.1.6 by stepping this very model to a ball-floor
        contact and finding the pair), and the exact expression upstream's own
        `get_reward` iterates. `contact[i].geom1/geom2` is the same data element-wise.
        """
        task = self._env.task
        if task.stage != "catch":
            return
        gid = self._ball_geom_id
        for pair in self._env.data.contact.geom:
            if gid in pair:
                task.stage = "throw"
                return

    def _task_reset_state(self, qpos: np.ndarray, qvel: np.ndarray,
                          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """`Basketball.reset_model`: the ball is re-thrown at the robot.

        Launch angle theta ~ U(0, 1.45) * choice(+-1); the ball starts 1.5 m out at
        (1.5 cos, 1.5 sin, 0.3924 + left_hand_z) with velocity -7.5 * (cos, sin, 0) --
        7.5 m/s straight at the robot, no vertical component. `left_hand_z` is read off
        the ALREADY-SET perturbed state, which is upstream's ordering
        (`HumanoidEnv.reset_model` calls `set_state` before `task.reset_model` reads
        `site_xpos`), so the state is set here first and again by the base -- the
        second set is upstream's own double-set. The RNG is ours, not the global
        `np.random`, for Room's reason.
        """
        import mujoco

        self._env.set_state(qpos, qvel)
        sid = int(mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_SITE, "left_hand"))
        hand_z = float(self._env.data.site_xpos[sid][2])
        angle = rng.random() * 1.45 * (1.0 if rng.random() < 0.5 else -1.0)
        qpos, qvel = qpos.copy(), qvel.copy()
        qpos[-7] = 1.5 * np.cos(angle)
        qpos[-6] = 1.5 * np.sin(angle)
        qpos[-5] = 0.3924 + hand_z
        qvel[-6] = -7.5 * np.cos(angle)
        qvel[-5] = -7.5 * np.sin(angle)
        self._env.task.stage = "catch"
        return qpos, qvel

    def task_metric(self, traj: Any) -> float:
        """Best approach of the ball to the hoop centre, as a fraction of the initial
        gap -- and 1.0 outright the moment any arriving state is inside the
        environment's own success radius, which is also the moment the episode ends.

        The LAUNCH GEOMETRY is what makes this falsify the hold exploit: the ball is
        fired from ~1.5 m in front of the robot TOWARD it -- away from the hoop -- so a
        catch-and-hold leaves the closest approach at roughly the launch distance and
        the metric at ~0.0. Only carrying or throwing the ball hoopward moves it. The
        hoop centre is fixed world geometry (`_HOOP_POS`), so the metric is a pure
        function of the states.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        arrived = states[1:]
        d = np.linalg.norm(arrived[:, 76:79] - _HOOP_POS, axis=1)
        if bool(np.any(d < _GOAL_RADIUS)):
            return 1.0
        d0 = float(np.linalg.norm(states[0, 76:79] - _HOOP_POS))
        return float(np.clip(1.0 - float(np.min(d)) / max(d0, _GOAL_RADIUS), 0.0, 1.0))

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(np.linalg.norm(s[76:79] - _HOOP_POS) < _GOAL_RADIUS)


# --------------------------------------------------------------------------
# walk / crawl / sit
# --------------------------------------------------------------------------


class H1HandWalk(_H1HandBase):
    """Walk forward at 1 m/s on open, empty ground. `h1hand-walk-v0`."""

    name = "h1hand_walk"
    gym_id = "h1hand-walk-v0"
    obs_dim = 151
    action_dim = 61
    horizon = 1000
    n_q, n_v = 76, 75
    _ask = (f"WALK +X AT {_LOCO_V_REF:.1f} M/S",)

    #: A quarter of the task's own full pace, matching `h1_crawl`'s threshold on the
    #: same normaliser. PROVISIONAL like every threshold in this file.
    success_threshold = 0.25

    _free_pos_low = (-2.0, -10.0, -0.5)
    _free_pos_high = (50.0, 10.0, 3.0)

    def task_metric(self, traj: Any) -> float:
        """Mean forward speed against the task's own 1 m/s bound -- `H1Crawl.task_metric`'s
        form minus the corridor gate, because there is no corridor: the ground is open in
        every direction and any path forward is walking.

        Endpoints over elapsed steps, not a high-water mark: `walk` is a pace task, so a
        sprint-and-collapse and a steady shuffle should read as the speeds they averaged.
        Standing still -- the `(5*move+1)/6` floor's local optimum, see the spec's
        `exploits` -- reads exactly 0.0. Backwards travel clips to 0.0 rather than going
        negative, for the reason `H1Crawl.task_metric` gives.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        n_steps = states.shape[0] - 1
        mean_v = (float(states[-1, 0]) - float(states[0, 0])) / (n_steps * _CONTROL_DT)
        return float(min(max(mean_v, 0.0), _LOCO_V_REF) / _LOCO_V_REF)

    def _success_flag(self, s: np.ndarray) -> bool:
        """One metre of travel from the spawn -- the robot demonstrably walked."""
        return bool(s[0] >= 1.0)


def _uprightness(states: np.ndarray) -> np.ndarray:
    """The world-z component of the root frame's own z axis, from the orientation
    quaternion at `s[3:7]`.

    MuJoCo's free joint stores `(x, y, z, qw, qx, qy, qz)`, so the rotation matrix entry
    `R[2, 2]` is `1 - 2*(qx^2 + qy^2)` -- +1 upright, 0 on its side, -1 inverted. Derived
    from the observation rather than read off a body xmat because a `task_metric` may only
    use what the state carries; `Walk.get_reward`'s own `torso_upright()` reaches into the
    simulator, which the metric deliberately cannot.
    """
    return 1.0 - 2.0 * (states[:, 4] ** 2 + states[:, 5] ** 2)


class H1HandStand(_H1HandBase):
    """Stand still and upright on flat ground. `h1hand-stand-v0`.

    `class Stand(Walk)` upstream, and the whole task is one attribute: `_move_speed = 0`
    selects the other branch of `Walk.get_reward`, which drops the `move` term and prices
    `dont_move` instead. Same scene as walk, same robot, same termination.
    """

    name = "h1hand_stand"
    gym_id = "h1hand-stand-v0"
    obs_dim = 151
    action_dim = 61
    horizon = 1000
    n_q, n_v = 76, 75
    _ask = ("STAND STILL AND UPRIGHT",)

    #: Half the episode held. PROVISIONAL like every threshold in this file, and higher
    #: than the travelling tasks' 0.25 because the robot STARTS in the scoring state --
    #: a policy that does nothing at all already scores well above zero here, which is the
    #: opposite of walk.
    success_threshold = 0.5

    _free_pos_low = (-2.0, -2.0, -0.5)
    _free_pos_high = (2.0, 2.0, 2.0)

    def task_metric(self, traj: Any) -> float:
        """Fraction of arriving states that are standing still AND upright.

        Three gates, all read from the observation: the pelvis above
        `_STAND_MIN_PELVIS_Z`, the root frame's uprightness above `_STAND_MIN_UPRIGHT`,
        and horizontal speed below `_STAND_MAX_HORIZONTAL_SPEED`. None of them is the
        reward's own constant -- the reward gates on head height and a body xmat, neither
        of which is in the state.

        THE HEIGHT GATE IS WHAT FALSIFIES THE RECORDED EXPLOIT. `dont_move` reads only
        `center_of_mass_velocity()[[0, 1]]`, so vertical motion is unpriced and a policy
        that hops in place collects nearly the full reward. This metric loses a step every
        time the pelvis dips below the gate, so a hopping policy and a standing one are
        different numbers here and the same number to the reward.

        ARRIVING states, as the spec says and as every other fraction metric in this file
        does: row 0 is the reset state, no action produced it, and the keyframe reset
        passes all three gates. Counted, it would make a policy that collapsed in T steps
        read 1/(T+1) for standing on none of them.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        arrived = states[1:]
        tall = arrived[:, 2] >= _STAND_MIN_PELVIS_Z
        upright = _uprightness(arrived) >= _STAND_MIN_UPRIGHT
        still = np.hypot(arrived[:, 76], arrived[:, 77]) <= _STAND_MAX_HORIZONTAL_SPEED
        return float(np.mean(tall & upright & still))

    def _success_flag(self, s: np.ndarray) -> bool:
        """One arriving state that is standing still and upright."""
        return bool(s[2] >= _STAND_MIN_PELVIS_Z
                    and (1.0 - 2.0 * (s[4] ** 2 + s[5] ** 2)) >= _STAND_MIN_UPRIGHT
                    and float(np.hypot(s[76], s[77])) <= _STAND_MAX_HORIZONTAL_SPEED)


class H1HandRun(_H1HandBase):
    """Run forward at 5 m/s on open, empty ground. `h1hand-run-v0`.

    `class Run(Walk)` upstream, and like stand the whole task is one attribute:
    `_move_speed = _RUN_SPEED`. Same scene as walk, same reward code, same termination --
    only the velocity the `move` tolerance is measured against differs.
    """

    name = "h1hand_run"
    gym_id = "h1hand-run-v0"
    obs_dim = 151
    action_dim = 61
    horizon = 1000
    n_q, n_v = 76, 75
    _ask = (f"RUN +X AT {_RUN_V_REF:.1f} M/S",)

    #: A fifth of the task's own full pace -- deliberately NOT walk's 0.25, because the
    #: bound underneath it is five times larger and reusing the fraction would quietly
    #: demand five times the speed. PROVISIONAL like every threshold in this file.
    success_threshold = 0.2

    _free_pos_low = (-2.0, -10.0, -0.5)
    _free_pos_high = (250.0, 10.0, 3.0)

    def task_metric(self, traj: Any) -> float:
        """Mean forward speed against the task's own 5 m/s bound.

        `H1HandWalk.task_metric`'s form against a different normaliser, and the normaliser
        is the task: upstream changes exactly one attribute to turn walk into run, so the
        metric changes exactly one constant. Endpoints over elapsed steps, so a
        sprint-and-collapse reads as the speed it averaged; standing still reads 0.0, which
        is what falsifies the `(5*move+1)/6` floor the spec records; backwards travel
        clips to 0.0 rather than going negative.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        n_steps = states.shape[0] - 1
        mean_v = (float(states[-1, 0]) - float(states[0, 0])) / (n_steps * _CONTROL_DT)
        return float(min(max(mean_v, 0.0), _RUN_V_REF) / _RUN_V_REF)

    def _success_flag(self, s: np.ndarray) -> bool:
        """Five metres of travel from the spawn -- one second at the task's own pace."""
        return bool(s[0] >= 5.0)


class _H1HandClimb(_H1HandBase):
    """Shared by `stair` and `slide`, because upstream shares everything.

    `class Stair(ClimbingUpwards): pass` and `class Slide(ClimbingUpwards): pass` -- the
    two tasks are byte-identical code differing only in which asset they load, so a
    metric written twice would be two copies of one fact. Subclasses supply nothing but
    the id and the scene constants their prose is derived from.

    THE TERRAIN UNDULATES, which is what the metric has to answer for. Ray-cast along
    y = 0, stair peaks at 0.9 m every 6 m and slide at 1.645 m every 10 m, each returning
    almost to ground level between peaks. So the far side of every rise is a descent, and
    `ClimbingUpwards`'s `move` term pays for forward velocity there exactly as it does on
    the climb -- with gravity supplying it. Pace alone would score a policy that pitches
    over each crest and slides down.
    """

    obs_dim = 151
    action_dim = 61
    horizon = 1000
    n_q, n_v = 76, 75

    #: A quarter of the task's own pace while upright. PROVISIONAL, and the same figure
    #: walk uses, because the `move` term underneath it is bounded at the same 1 m/s.
    success_threshold = 0.25

    def task_metric(self, traj: Any) -> float:
        """Mean forward pace against the task's own 1 m/s bound, GATED ON STAYING UPRIGHT.

        Two factors, and the second is the one that does work here. The first is
        `H1HandWalk.task_metric` unchanged -- endpoints over elapsed steps, clipped at the
        `_WALK_SPEED` bound `ClimbingUpwards` measures against, so standing still reads 0.0
        and falsifies the `(5*move+1)/6` floor. The second is the fraction of arriving
        states whose uprightness clears `_CLIMB_MIN_UPRIGHT`, and it is what separates
        traversing this terrain from falling down it: a policy that pitches over each crest
        collects the same `move` reward as one that walks down, and scores its uprightness
        here instead.

        The gate is deliberately not `ClimbingUpwards.get_terminated`'s 0.1. That terminal
        is 84 degrees from vertical -- it ends an episode only once the robot is nearly
        horizontal -- so reusing it would make the metric agree with the reward about
        exactly the behaviour the spec records as an exploit.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        n_steps = states.shape[0] - 1
        mean_v = (float(states[-1, 0]) - float(states[0, 0])) / (n_steps * _CONTROL_DT)
        pace = min(max(mean_v, 0.0), _LOCO_V_REF) / _LOCO_V_REF
        # Arriving states only (the spec's wording): the upright reset row is not the
        # policy's doing, and counted it would buy 1/(T+1) of the pace for a robot that
        # fell on step one.
        upright = float(np.mean(_uprightness(states[1:]) >= _CLIMB_MIN_UPRIGHT))
        return float(pace * upright)

    def _success_flag(self, s: np.ndarray) -> bool:
        """Past the first crest, still upright. The x is the subclass's scene fact."""
        return bool(s[0] >= self._first_crest_x
                    and (1.0 - 2.0 * (s[4] ** 2 + s[5] ** 2)) >= _CLIMB_MIN_UPRIGHT)


class H1HandStair(_H1HandClimb):
    """Climb a repeating staircase. `h1hand-stair-v0`.

    Terrain measured by ray-casting the asset: 0.18 m treads rising to a 0.9 m landing at
    x = 3, 9, 15, 21 m and back to ground level between, 10 m wide so it cannot be skirted.
    """

    name = "h1hand_stair"
    gym_id = "h1hand-stair-v0"

    #: First landing, ray-cast from `h1hand_pos_stair.xml`.
    _first_crest_x = 3.0
    _ask = (f"STAIRS: CLIMB +X AT {_LOCO_V_REF:.1f} M/S",)

    _free_pos_low = (-2.0, -5.0, -0.5)
    _free_pos_high = (25.0, 5.0, 3.0)


class H1HandSlide(_H1HandClimb):
    """Climb a repeating ramp. `h1hand-slide-v0`.

    Terrain measured by ray-casting the asset: a smooth climb to a 1.645 m crest at
    x = 5, 15, 25 m and beyond, falling to 0.105 m between, 10 m wide.
    """

    name = "h1hand_slide"
    gym_id = "h1hand-slide-v0"

    #: First crest, ray-cast from `h1hand_pos_slide.xml`.
    _first_crest_x = 5.0
    _ask = (f"RAMP: CLIMB +X AT {_LOCO_V_REF:.1f} M/S",)

    _free_pos_low = (-2.0, -5.0, -0.5)
    _free_pos_high = (90.0, 5.0, 4.0)


class H1HandCrawl(_H1HandBase):
    """Crawl a 16 m tunnel whose ceiling is too low to walk under. `h1hand-crawl-v0`.

    `h1_crawl`'s task on the Shadow-Hands embodiment -- same tunnel geometry, same
    reward family, and a `task_metric` IDENTICAL IN FORM to `H1Crawl.task_metric`, so
    the plain-H1 and Shadow-Hands numbers are directly comparable. Unlike every other
    task in this file it NEVER terminates: `Crawl.get_terminated` is constantly False.
    """

    name = "h1hand_crawl"
    gym_id = "h1hand-crawl-v0"
    obs_dim = 151
    action_dim = 61
    horizon = 1000
    n_q, n_v = 76, 75
    _ask = (f"CRAWL +X AT {_LOCO_V_REF:.1f} M/S", "STAY INSIDE THE TUNNEL")

    #: `h1_crawl`'s own threshold on the same metric. PROVISIONAL there and here.
    success_threshold = 0.25

    _free_pos_low = (-2.0, -3.0, -0.5)
    _free_pos_high = (20.0, 3.0, 2.0)

    def task_metric(self, traj: Any) -> float:
        """Mean forward speed times the corridor gate -- `H1Crawl.task_metric`, in the
        same units, against the same `v_ref`, with the same `in_tunnel` half-width.
        Its docstring carries the full argument; what matters here is only that the two
        embodiments stay comparable, so any change to one metric is a change to both.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        n_steps = states.shape[0] - 1
        mean_v = (float(states[-1, 0]) - float(states[0, 0])) / (n_steps * _CONTROL_DT)
        progress = min(max(mean_v, 0.0), _LOCO_V_REF) / _LOCO_V_REF
        arrived = states[1:]
        in_corridor = float(np.mean(np.abs(arrived[:, 1]) <= _CRAWL_CORRIDOR_HALF_WIDTH))
        return float(progress * in_corridor)

    def _success_flag(self, s: np.ndarray) -> bool:
        """Inside the tunnel, past its mouth -- the robot got down and in."""
        return bool(s[0] > _TUNNEL_X0 and abs(s[1]) <= _CRAWL_CORRIDOR_HALF_WIDTH)


#: `H1HandSitSimple.task_metric`'s denominator. Module level for `_BALANCE_HORIZON`'s
#: reason: the metric is called UNBOUND in the sim-free tests (`self` is None), and a
#: constant inside the class body would ship through `describe("full_source")`. The
#: class binds its `horizon` to this so the two cannot disagree.
_SIT_HORIZON = 1000


class H1HandSitSimple(_H1HandBase):
    """Sit on a fixed chair 0.25 m behind the spawn, and stay seated.
    `h1hand-sit_simple-v0`.

    `sit_hard` (a FREE-BODY chair plus a robot spawn randomised through the global
    `np.random`) is a separate task, `H1HandSitHard`.
    """

    name = "h1hand_sit_simple"
    gym_id = "h1hand-sit_simple-v0"
    obs_dim = 151
    action_dim = 61
    horizon = _SIT_HORIZON
    n_q, n_v = 76, 75
    _ask = ("SIT ON THE CHAIR BEHIND YOU",)

    #: Seated for a quarter of the episode. PROVISIONAL like every threshold here.
    success_threshold = 0.25

    _free_pos_low = (-2.0, -2.0, -0.5)
    _free_pos_high = (2.0, 2.0, 2.0)

    def task_metric(self, traj: Any) -> float:
        """Steps spent seated, over the FULL horizon: pelvis over the seat footprint
        within `_SEATED_BAND` of the seat surface, by geometry.

        Fixed scene facts only -- the shipped reward's own (0.68, 0.72) height kernel is
        deliberately not borrowed. Standing beside the chair reads 0.0 (outside the
        footprint); standing ON the chair reads 0.0 (above the band); a pelvis on the
        floor is below the band and the 0.5 m terminal ends the episode there. What it
        cannot see is contact -- a squat hovering over the seat reads as seated, exactly
        as it does to the shipped reward, and the spec's `exploits` records that as
        `undetectable_here`.

        The denominator is the horizon, not the rows visited, for `_H1HandBalance`'s
        reason: leaving the scoring posture is what ends the episode. Both consumers --
        `training._rollout` and `policy_api.run_episode` -- stop at `done`, and a robot
        toppling BACKWARDS off the spawn passes through the seat band over the footprint
        on its way to the 0.5 m terminal, so the fall itself is the visited trajectory.
        `np.mean(_seated(states[1:]))`, a fraction of whatever rows happened to be
        visited, would let 9 of 30 uniform-random episodes clear `success_threshold`
        (mean 0.147, max 0.604, every "seated" step BEFORE the terminal): "sitting on the
        floor reads 0.0" would be true of the floor and false of the fall that reaches
        it. A terminated episode forfeits the steps it did not run; a full-horizon
        trajectory is unaffected by the choice.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        return float(min(np.count_nonzero(_seated(states[1:])) / _SIT_HORIZON, 1.0))

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(_seated(np.asarray(s, dtype=float).reshape(1, -1))[0])


# --------------------------------------------------------------------------
# pole, maze
# --------------------------------------------------------------------------

#: `pole.py`'s own `_WALK_SPEED = 0.5` -- HALF of walk's 1.0, two tasks' published
#: targets that happen to share a variable name. A separate constant for run's reason:
#: tying them together would let a change to one task's normaliser silently move the
#: other's numbers.
_POLE_V_REF = 0.5

#: The pole forest's lateral extent, read off the built model: 144 collision cylinders
#: (radius 0.05) in 42 rows at x = 0..41 m, outermost centres at |y| = 2.25. The gate is
#: centre plus radius. The ground beyond it is open, which is exactly the exploit the
#: spec records: nothing in the reward stops a robot walking AROUND the forest.
_POLE_FOREST_HALF_WIDTH = 2.3

#: `Maze.checkpoints` minus the degenerate first entry (the spawn itself, which
#: registers on the first reward call with no action taken -- the spec records it as a
#: free +100). Three real checkpoints, visited in this order, and the leg length each
#: approach is normalised by: spawn->(3,0) is 3 m, ->(3,6) is 6 m, ->(6,6) is 3 m.
_MAZE_CHECKPOINTS: Tuple[Tuple[float, float], ...] = ((3.0, 0.0), (3.0, 6.0), (6.0, 6.0))
_MAZE_LEG_LENGTHS: Tuple[float, ...] = (3.0, 6.0, 3.0)

#: `Maze.update_move_direction`'s own advance radius. Adopted because it is the stage
#: machine's arrival test -- task structure, the same argument `_GOAL_RADIUS` makes for
#: push's termination radius -- not a shaping constant.
_MAZE_CP_RADIUS = 0.4

#: The commanded direction per stage, read off `Maze.update_move_direction`'s branches:
#: stages 0 and 1 head +x, 2 heads +y, 3 heads +x, and the 3->4 transition leaves the
#: direction at +y (where the move term is then overridden to full marks anyway).
_MAZE_DIRECTIONS: Tuple[Tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

#: Upstream's junction-marker colouring, as a function of the stage (cumulative --
#: derived from `update_move_direction`'s branch bodies: a reached marker turns green,
#: the next turns red, and nothing is ever repainted black).
_MAZE_GREEN = (0.0, 1.0, 0.0, 1.0)
_MAZE_RED = (1.0, 0.0, 0.0, 1.0)
_MAZE_BLACK = (0.0, 0.0, 0.0, 1.0)
_MAZE_MARKERS = ("intersection_a", "intersection_b", "intersection_c", "intersection_d")


class H1HandPole(_H1HandBase):
    """Walk forward at 0.5 m/s through a 41 m forest of thin vertical poles.
    `h1hand-pole-v0`.

    Walk's reward shape against half walk's pace, times a x0.1 discount on any pole
    contact -- and unlike hurdle, the discount really does watch the obstacles. The
    termination is also stricter than walk's: pelvis below 0.5 m, not 0.2.
    """

    name = "h1hand_pole"
    gym_id = "h1hand-pole-v0"
    obs_dim = 151
    action_dim = 61
    horizon = 1000
    n_q, n_v = 76, 75
    _ask = (f"POLES: WEAVE +X AT {_POLE_V_REF:.1f} M/S",)

    #: A quarter of the task's own full pace, walk's fraction against pole's own bound.
    #: PROVISIONAL like every threshold in this file.
    success_threshold = 0.25

    _free_pos_low = (-2.0, -5.0, -0.5)
    _free_pos_high = (45.0, 5.0, 3.0)

    def task_metric(self, traj: Any) -> float:
        """Mean forward pace against the task's own 0.5 m/s bound, GATED ON STAYING IN
        THE FOREST.

        The gate is the factor that does the work here: the reward's move term pays
        forward centre-of-mass velocity wherever the robot is, the collision discount
        fires only on contact, and the ground beside the forest is open -- so walking
        AROUND the field at |y| > 2.3 collects the full move term forever while never
        engaging the task. That skirt and a genuine weave are the same number to the
        reward and different numbers here. Standing still reads 0.0 (the `(5*move+1)/6`
        floor's local optimum), and backwards travel clips to 0.0.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        n_steps = states.shape[0] - 1
        mean_v = (float(states[-1, 0]) - float(states[0, 0])) / (n_steps * _CONTROL_DT)
        pace = min(max(mean_v, 0.0), _POLE_V_REF) / _POLE_V_REF
        in_forest = float(np.mean(np.abs(states[1:, 1]) <= _POLE_FOREST_HALF_WIDTH))
        return float(pace * in_forest)

    def _success_flag(self, s: np.ndarray) -> bool:
        """One metre of travel, made inside the forest rather than around it."""
        return bool(s[0] >= 1.0 and abs(s[1]) <= _POLE_FOREST_HALF_WIDTH)


class H1HandMaze(_H1HandBase):
    """Walk an L-shaped walled maze to its far corner via fixed checkpoints.
    `h1hand-maze-v0`.

    The third task in this file whose observation is more than the simulator state:
    `Maze.update_move_direction` keeps a checkpoint counter (0..4) on the task object,
    and the reward reads it three ways -- it selects the commanded direction, it selects
    the checkpoint the proximity kernel measures to, and each advance pays a one-off
    `+100 * stage`. The counter rides in s[151].

    THE TAIL CARRIES THE LATCH AS OF THE PREVIOUS STEP'S UPDATE (`_before_step`'s
    contract), because upstream pays the advance bonus inside `get_reward`:
    `reference_reward(s)` restores the pre-advance value and upstream's own update then
    re-fires at exactly the states where upstream paid. The one divergence is the
    degenerate spawn checkpoint, whose +100 lands in the RESET state's reference reward
    rather than the first step's -- upstream never evaluates its reward at the reset
    state, so there is no upstream number for that state to disagree with; the spec
    records the spawn checkpoint as free either way.
    """

    name = "h1hand_maze"
    gym_id = "h1hand-maze-v0"
    obs_dim = 152
    action_dim = 61
    horizon = 1000
    n_q, n_v = 76, 75
    #: The fixed route, from the checkpoints the metric counts. The stage in s[151] is
    #: deliberately not restated: `_restore_extra` paints the junction markers from it.
    _ask = ("MAZE: " + " THEN ".join(f"{cx:.0f},{cy:.0f}" for cx, cy in _MAZE_CHECKPOINTS),)
    extra_dims = 1

    #: One of the three real checkpoints reached. PROVISIONAL like every threshold here.
    success_threshold = 0.33

    _free_pos_low = (-2.0, -2.0, -0.5)
    _free_pos_high = (8.0, 8.0, 3.0)

    def _extra_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(1), np.full(1, 4.0)

    def _extra_state(self) -> np.ndarray:
        return np.array([float(self._env.task.maze_stage)])

    def _restore_extra(self, tail: np.ndarray) -> None:
        """Stage, commanded direction AND the junction markers' colours, all derived
        from the one carried number.

        The colours matter because frames are judged on this tier: upstream paints a
        reached marker green and the next red, but only through `get_reward`, and never
        resets them -- episode two starts wearing episode one's progress. Deriving them
        from the restored stage makes a rendered frame a pure function of the state and
        is recorded in the spec as a deliberate divergence in reset dressing.
        """
        stage = int(np.clip(round(float(tail[0])), 0, 4))
        task = self._env.task
        task.maze_stage = stage
        task.move_direction = np.array(_MAZE_DIRECTIONS[stage])
        rgba = self._env.named.model.geom_rgba
        for k, marker in enumerate(_MAZE_MARKERS):
            if k < stage or (k == 0 and stage == 0):
                rgba[marker] = _MAZE_GREEN
            elif k == stage:
                rgba[marker] = _MAZE_RED
            else:
                rgba[marker] = _MAZE_BLACK

    def _before_step(self) -> None:
        """Upstream's own stage machine, run on the restored state -- called rather than
        reimplemented, so the advance radius, the direction switches and the marker
        painting cannot drift from `Maze.update_move_direction`. The returned bonus is
        upstream's to pay inside `get_reward`; here only the side effect is wanted."""
        self._env.task.update_move_direction()

    def _task_reset(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`Maze.reset_model`: stage and direction back to the start. The markers are
        deliberately repainted too (upstream leaves them stale across episodes -- see
        `_restore_extra`)."""
        self._restore_extra(np.zeros(1))
        return qpos

    def task_metric(self, traj: Any) -> float:
        """Checkpoints reached in order, out of three, with partial credit for the best
        approach toward the first unreached one -- push's high-water argument applied
        leg by leg.

        GEOMETRY ONLY, recomputed from the pelvis track rather than read off the carried
        stage: the metric must hold whatever a candidate did to the tail. The spawn
        checkpoint is not counted (the robot starts inside it -- the +100 it pays is the
        spec's `spawn_checkpoint_is_free` exploit, worth nothing here), and each leg's
        approach is normalised by that leg's own length, so parking beside a junction
        freezes the score at the progress already made while the reward's proximity
        kernel keeps paying per step. Standing still at the spawn reads exactly 0.0.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        xy = states[:, 0:2]
        reached, frm = 0, 0
        for cx, cy in _MAZE_CHECKPOINTS:
            d = np.hypot(xy[frm:, 0] - cx, xy[frm:, 1] - cy)
            hits = np.nonzero(d <= _MAZE_CP_RADIUS)[0]
            if hits.size == 0:
                break
            frm += int(hits[0])
            reached += 1
        if reached == len(_MAZE_CHECKPOINTS):
            return 1.0
        cx, cy = _MAZE_CHECKPOINTS[reached]
        d = np.hypot(xy[frm:, 0] - cx, xy[frm:, 1] - cy)
        approach = 1.0 - float(np.min(d)) / _MAZE_LEG_LENGTHS[reached]
        approach = min(max(approach, 0.0), 1.0)
        return float((reached + approach) / len(_MAZE_CHECKPOINTS))

    def _success_flag(self, s: np.ndarray) -> bool:
        """At the goal corner, by the stage machine's own arrival radius."""
        gx, gy = _MAZE_CHECKPOINTS[-1]
        return bool(np.hypot(s[0] - gx, s[1] - gy) <= _MAZE_CP_RADIUS)


# --------------------------------------------------------------------------
# sit_hard, reach
# --------------------------------------------------------------------------

#: The chair's seat plate, in the CHAIR'S OWN frame this time: half-size 0.21 m square,
#: surface at local z = `_SEAT_TOP_Z` -- the same plate `assets/tasks/sit.xml` gives
#: sit_simple, read off `sit_hard.xml`. sit_simple's `_SEAT_BOX` is this footprint
#: translated to that task's FIXED chair at (-0.25, 0, 0); here the chair is a free body,
#: so the footprint has to live in its frame and follow it.
_CHAIR_SEAT_HALF = 0.21

#: The metric's "hand at the target" radius: roughly the span of the Shadow Hand's palm.
#: A stated convention, and deliberately NOT the reward's own 0.05 bonus radius -- reach
#: has no terminal, so upstream's 0.05 is pure shaping and borrowing it would make the
#: metric a function of the reward it is meant to falsify.
_REACH_TOUCH_RADIUS = 0.10

#: `Reach.__init__`: target_low / target_high, the per-episode goal's sample box.
#: Doubles as the sample box for the goal's three observation dims in `_extra_bounds`.
_REACH_TARGET_LOW = (-2.0, -2.0, 0.2)
_REACH_TARGET_HIGH = (2.0, 2.0, 2.0)


def _seated_on_chair(states: np.ndarray) -> np.ndarray:
    """Per-row: is the pelvis over the FREE chair's seat footprint at seated height,
    in the chair's own frame?

    `_seated` (sit_simple) is this same test against a chair that cannot move, so its
    footprint is world-fixed. Here the chair's pose rides in s[76:83] and the pelvis is
    expressed in its frame first -- which is what makes the toppled-chair case honest:
    tip the chair on its side and the seat plane tips with it, so a robot squatting in
    mid-air where the seat USED to be reads 0.0 (the spec's
    `sitting_band_is_absolute_while_the_chair_moves` exploit, falsified).
    """
    local = _quat_rotate_inverse(states[:, 79:83], states[:, 0:3] - states[:, 76:79])
    return ((np.abs(local[:, 0]) <= _CHAIR_SEAT_HALF)
            & (np.abs(local[:, 1]) <= _CHAIR_SEAT_HALF)
            & (local[:, 2] > _SEAT_TOP_Z) & (local[:, 2] <= _SEAT_TOP_Z + _SEATED_BAND))


class H1HandSitHard(_H1HandBase):
    """Sit on a LOOSE chair from a randomised spawn, and stay seated.
    `h1hand-sit_hard-v0`.

    `class SitHard(Sit)`: same reward, same terminal, two changes of circumstance --
    the chair gains a free joint (thirteen observation slots at s[76:83] and s[158:164])
    and `reset_model` randomises the robot's x, y and heading. That reset uses the
    GLOBAL `np.random`; the fix is the room/push/package one this file already uses three times --
    the same three uniforms, drawn from ctx.rng, reproducible from a seed where
    upstream's are not.
    """

    name = "h1hand_sit_hard"
    gym_id = "h1hand-sit_hard-v0"
    obs_dim = 164
    action_dim = 61
    horizon = 1000
    n_q, n_v = 83, 81
    _ask = ("SIT ON THE CHAIR AND STAY",)

    #: Seated for a quarter of the episode, sit_simple's own threshold on the matching
    #: metric. PROVISIONAL like every threshold in this file.
    success_threshold = 0.25

    _free_pos_low = (-3.0, -3.0, -0.5)
    _free_pos_high = (3.0, 3.0, 2.0)

    def _task_reset(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`SitHard.reset_model`: x ~ U(0.2, 0.4), y ~ U(-0.15, 0.15), yaw ~ U(-1.8, 1.8)
        -- overriding the base perturbation on exactly those coordinates, as upstream
        does. The quaternion is the pure-yaw case of upstream's `euler_to_quat`. The RNG
        is ours, not the global `np.random`, for Room's reason."""
        qpos = qpos.copy()
        qpos[0] = rng.uniform(0.2, 0.4)
        qpos[1] = rng.uniform(-0.15, 0.15)
        yaw = rng.uniform(-1.8, 1.8)
        qpos[3:7] = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
        return qpos

    def task_metric(self, traj: Any) -> float:
        """The fraction of arriving steps spent seated, in the chair's own frame.

        sit_simple's metric with the footprint attached to the chair instead of the
        world: slide the chair and the scoring region slides with it; tip it over and
        the region tips too, so the reward's mid-air squat above a toppled chair reads
        0.0 here. What it still cannot see is contact -- the hover-squat reads as seated
        exactly as it does on sit_simple, and the spec records that as
        `undetectable_here`.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        return float(np.mean(_seated_on_chair(states[1:])))

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(_seated_on_chair(np.asarray(s, dtype=float).reshape(1, -1))[0])


class H1HandReach(_H1HandBase):
    """Move the left hand to a per-episode target point and hold it there.
    `h1hand-reach-v0`.

    The observation is upstream's own: qpos ++ qvel ++ left_hand ++ target, 157-D. The
    six-slot tail is HALF derived and HALF task state, and the difference matters:
    the target lives on the task object and must be carried (push's goal exactly), while
    the left hand is forward kinematics of the qpos block -- upstream appends it so a
    reward need not do FK, and `_restore_extra` deliberately ignores it, because the
    simulator recomputes it from the restored joints. No terminal of any kind: episodes
    always run the full 1000 steps.
    """

    name = "h1hand_reach"
    gym_id = "h1hand-reach-v0"
    obs_dim = 157
    action_dim = 61
    horizon = 1000
    n_q, n_v = 76, 75
    _ask = ("LEFT HAND TO THE GOAL, HOLD",)
    extra_dims = 6

    #: At the target for a quarter of the episode -- with no terminal, the task is to
    #: get there and STAY. PROVISIONAL like every threshold in this file.
    success_threshold = 0.25

    _free_pos_low = (-3.0, -3.0, -0.5)
    _free_pos_high = (3.0, 3.0, 2.5)

    def _extra_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.concatenate([[-3.0, -3.0, 0.0], _REACH_TARGET_LOW]),
                np.concatenate([[3.0, 3.0, 2.5], _REACH_TARGET_HIGH]))

    def _extra_state(self) -> np.ndarray:
        task = self._env.task
        return np.concatenate([
            np.asarray(task.robot.left_hand_position(), dtype=float).ravel(),
            np.asarray(task.goal, dtype=float).ravel()])

    def _restore_extra(self, tail: np.ndarray) -> None:
        """Only the target half restores. The hand half is derived state -- writing it
        anywhere would be inventing a second copy of the arm's forward kinematics for
        the simulator to disagree with; it is recomputed from qpos on the next read.
        The assignment raises on a wrong-sized tail, push's convention."""
        self._env.task.goal[:] = np.asarray(tail, dtype=float).ravel()[3:6]

    def _task_reset(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`Reach.reset_model`: goal ~ U(target_low, target_high), drawn from ctx.rng
        rather than upstream's global `np.random` (Room's reason). qpos is untouched --
        only the target moves."""
        task = self._env.task
        task.goal[:] = rng.uniform(task.target_low, task.target_high)
        return qpos

    def task_metric(self, traj: Any) -> float:
        """Fraction of arriving states with the left hand AT the target -- inside
        `_REACH_TOUCH_RADIUS`, a stated hand-sized convention, not the reward's 0.05.

        Nothing terminates this task, so the reward's dense terms accrue for the whole
        1000 steps whatever the hand does: `reward_close` pays within a full METRE and
        `healthy_reward` pays for standing anywhere. Both documented exploits -- the
        loiterer and the statue -- spend their episode outside the radius and read
        exactly 0.0 here, while a policy that reaches and holds converges to 1.0. Both
        positions come off the observation tail, which on a rolled-out trajectory is the
        simulator's own forward kinematics.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        arrived = states[1:]
        d = np.linalg.norm(arrived[:, 151:154] - arrived[:, 154:157], axis=1)
        return float(np.mean(d <= _REACH_TOUCH_RADIUS))

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(np.linalg.norm(s[151:154] - s[154:157]) <= _REACH_TOUCH_RADIUS)

    def _legend_extra(self, tail: np.ndarray) -> List[str]:
        """The TARGET half of the tail only (`tail[3:6]`, the slots `_restore_extra`
        reads). The hand half is where the hand IS -- the answer, not the ask -- and a
        legend that printed both would be the distance to go in two lines."""
        return ["GOAL " + _legend_xyz(tail[3:6])]

    def _pre_render(self) -> None:
        """The target marker, push's mechanism: upstream's `Reach.render` draws objid
        789 through the viewer and this file bypasses `Task.render`."""
        _paint_goal_marker(self, np.asarray(self._env.task.goal, dtype=float).copy())


# --------------------------------------------------------------------------
# balance, window, cube
# --------------------------------------------------------------------------

#: The balance metric's denominator. A MODULE constant (the class already carries
#: `horizon`) because the metric is called unbound in the sim-free tests, where `self`
#: is None -- `_CONTROL_DT`'s reason.
_BALANCE_HORIZON = 1000

#: Stand's 0.8 m ground gate, measured from the BOARD's carried z instead of the world:
#: the board is what the robot stands on, and it moves.
_BALANCE_MIN_PELVIS_ABOVE_BOARD = 0.8

#: `assets/tasks/window.xml`, via the window body at x = 1: the collision slab spans
#: x in [0.95, 1.05] (near face 0.95), and the visible glass panes span y in
#: [-0.95, 0.95], z in [0.70, 1.91]. The wiper head counts as AT the pane within a
#: 0.10 m standoff of the near face -- a stated convention (contact is not in the
#: state), deliberately not the reward's own site bound of 0.92.
_WINDOW_PANE_NEAR_X = 0.95
_WINDOW_WIPE_STANDOFF = 0.10
_WINDOW_GLASS_HALF_Y = 0.95
_WINDOW_GLASS_Z_LO, _WINDOW_GLASS_Z_HI = 0.70, 1.91

#: A wipe is a SWEEP: the tool moving vertically at least this fast, a stated
#: convention (one pass over the 1.2 m glass in ~5 s). Deliberately not the reward's
#: `bounds=(0.5, 0.5)` kernel peak.
_WINDOW_SWEEP_SPEED = 0.25

#: The wiper head body's mount point on the handle, local (0, 0.5, 0.02) -- read off
#: `window.xml`. The head's world position is the handle pose (s[76:83]) rotated
#: through this offset; the ball joint moves the BLADE around the head, not the head
#: around the handle, so the head is the right proxy for where the wipe is happening.
_WIPER_HEAD_OFFSET = np.array([0.0, 0.5, 0.02])

_WINDOW_HORIZON = 1000


def _wiper_head(states: np.ndarray) -> np.ndarray:
    """The wiper head's world position: the handle pose (s[76:83]) rotated through the
    head's fixed mount offset. Forward kinematics of the carried state, pure numpy.

    MODULE level for both of the reasons the scene constants are: `describe
    ("full_source")` renders the class, and the metric is called UNBOUND in the
    sim-free tests, where `self` is None.
    """
    return states[:, 76:79] + _quat_rotate(states[:, 79:83], _WIPER_HEAD_OFFSET)

#: Both cubes within 20 degrees of the target counts as matched -- a stated convention
#: on the geodesic angle, deliberately not the reward's `margin=0.3` on raw quaternion
#: L2 distance (which is not a rotation metric; see `_quat_angle`).
_CUBE_ALIGN_TOL_RAD = np.deg2rad(20.0)

_CUBE_HORIZON = 500


class _H1HandBalance(_H1HandBase):
    """Shared by `balance_simple` and `balance_hard`, because upstream shares
    everything but one `<freejoint/>`: one reward, one three-clause terminal, one board.
    Subclasses supply the id, the widths and the spec.

    THE REWARD NEVER READS THE BOARD -- it is stand's posture product verbatim -- so the
    entire balance content of the task lives in the terminal, and the honest metric is a
    DURATION: how much of the full horizon was spent standing tall on the board. A
    visited-states mean would be degenerate here, because leaving the scoring posture is
    what ends the episode: ten balanced steps then a fall would read ~1.0 over visited
    states while failing the task almost entirely.
    """

    action_dim = 61
    horizon = 1000
    #: One ask for both variants, as upstream has one reward and one terminal for both;
    #: whether the sphere underneath is fixed or free is the scene, not the command.
    _ask = ("BALANCE ON THE BOARD",)

    #: Half the horizon held. PROVISIONAL like every threshold in this file, and higher
    #: than the travelling tasks' 0.25 for stand's reason: the robot starts in the
    #: scoring state.
    success_threshold = 0.5

    _free_pos_low = (-2.0, -2.0, -0.5)
    _free_pos_high = (2.0, 2.0, 2.5)

    def task_metric(self, traj: Any) -> float:
        """Steps spent standing tall on the board, over the FULL HORIZON.

        Two gates per arriving state, both from the observation: pelvis at least 0.8 m
        above the board's own carried z (stand's ground gate, made board-relative --
        the board moves, and on `balance_hard` so does what it rests on), and stand's
        uprightness gate. Dividing by the horizon rather than the states visited is
        what makes an early fall read as the failure it is; it also scores the
        documented hop exploit down, because every step the pelvis dips is a step not
        counted.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        arrived = states[1:]
        tall = arrived[:, 2] - arrived[:, 78] >= _BALANCE_MIN_PELVIS_ABOVE_BOARD
        upright = _uprightness(arrived) >= _STAND_MIN_UPRIGHT
        return float(min(np.count_nonzero(tall & upright) / _BALANCE_HORIZON, 1.0))

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(s[2] - s[78] >= _BALANCE_MIN_PELVIS_ABOVE_BOARD
                    and (1.0 - 2.0 * (s[4] ** 2 + s[5] ** 2)) >= _STAND_MIN_UPRIGHT)


class H1HandBalanceSimple(_H1HandBalance):
    """Stand on a see-saw board over a FIXED pivot sphere. `h1hand-balance_simple-v0`."""

    name = "h1hand_balance_simple"
    gym_id = "h1hand-balance_simple-v0"
    obs_dim = 164
    n_q, n_v = 83, 81


class H1HandBalanceHard(_H1HandBalance):
    """Stand on a board whose pivot sphere is FREE to roll away underneath.
    `h1hand-balance_hard-v0`. The board's rows sit at the same indices as on
    balance_simple; the sphere adds thirteen more."""

    name = "h1hand_balance_hard"
    gym_id = "h1hand-balance_hard-v0"
    obs_dim = 177
    n_q, n_v = 90, 87


class H1HandWindow(_H1HandBase):
    """Wipe a window: hold a long-handled wiper, press its head to the glass, keep it
    sweeping. `h1hand-window-v0`.

    The fourth task here whose observation exceeds the simulator, and the tail is the
    oddest of them: `Window.reset_model` stores WHERE THE HEAD WAS AT RESET on the task
    object, and the reward measures every step's head position against it. It rides in
    s[171:174]. The scene also brings the tier's first BALL joint -- the wiper head,
    anonymous in the XML -- whose four quaternion slots sit at s[83:87].
    """

    name = "h1hand_window"
    gym_id = "h1hand-window-v0"
    obs_dim = 174
    action_dim = 61
    horizon = 1000
    n_q, n_v = 87, 84
    _ask = ("WIPE THE WINDOW: KEEP THE", "WIPER SWEEPING ON THE GLASS")
    extra_dims = 3

    #: A quarter of the horizon spent wiping. PROVISIONAL like every threshold here.
    success_threshold = 0.25

    _free_pos_low = (-2.0, -2.0, -0.5)
    _free_pos_high = (2.0, 2.0, 2.5)

    def _extra_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """The spawn head position varies only by the +/-0.01 reset perturbation around
        (0, 0, ~1.66); a small box around the standing head covers it."""
        return np.array([-0.5, -0.5, 1.0]), np.array([0.5, 0.5, 2.0])

    def _extra_state(self) -> np.ndarray:
        return np.asarray(self._env.task.head_pos0, dtype=float).ravel().copy()

    def _restore_extra(self, tail: np.ndarray) -> None:
        """RAISES on a wrong-sized tail, package's argument exactly: `head_pos0` is a
        REBOUND attribute, so a skipped restore would not fail -- it would leave the
        previous episode's anchor on the task object and make the reward's
        head-distance term a function of what ran before."""
        pos = np.asarray(tail, dtype=float).ravel()
        if pos.size != self.extra_dims:
            raise ValueError(
                f"{type(self).__name__}._restore_extra needs the {self.extra_dims} "
                f"values of the reset head position and got {pos.size}. Skipping the "
                "restore would leave the previous episode's anchor in task.head_pos0 "
                "and make `reference_reward` depend on what ran before it")
        self._env.task.head_pos0 = pos.copy()

    def _task_reset_state(self, qpos: np.ndarray, qvel: np.ndarray,
                          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """`Window.reset_model` reads the head site off the ALREADY-SET perturbed state
        (basketball's ordering exactly), so the state is set here first and again by the
        base -- the second set is upstream's own double-set."""
        import mujoco

        self._env.set_state(qpos, qvel)
        sid = int(mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_SITE, "head"))
        self._env.task.head_pos0 = np.asarray(
            self._env.data.site_xpos[sid], dtype=float).copy()
        return qpos, qvel

    def task_metric(self, traj: Any) -> float:
        """Steps spent WIPING, over the full horizon: wiper head at the glass AND the
        tool sweeping vertically.

        Both documented exploits die on one conjunct each: air-wiping (the reward's
        `moving_wipe` pays vertical tool speed anywhere in space) fails the position
        gate, and the static press (its contact half pays for touching, motionless)
        fails the sweep gate. The denominator is the horizon because dropping the tool
        TERMINATES -- a visited-states mean would score a wipe-then-drop near 1.0. What
        the position gate cannot see is pressure: a blade sweeping nine centimetres off
        the glass reads as wiping, and the spec records that as `undetectable_here`.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        arrived = states[1:]
        head = _wiper_head(arrived)
        at_pane = ((head[:, 0] >= _WINDOW_PANE_NEAR_X - _WINDOW_WIPE_STANDOFF)
                   & (np.abs(head[:, 1]) <= _WINDOW_GLASS_HALF_Y)
                   & (head[:, 2] >= _WINDOW_GLASS_Z_LO)
                   & (head[:, 2] <= _WINDOW_GLASS_Z_HI))
        sweeping = np.abs(arrived[:, 164]) >= _WINDOW_SWEEP_SPEED
        return float(min(np.count_nonzero(at_pane & sweeping) / _WINDOW_HORIZON, 1.0))

    def _success_flag(self, s: np.ndarray) -> bool:
        s = np.asarray(s, dtype=float).reshape(1, -1)
        head = _wiper_head(s)[0]
        return bool(head[0] >= _WINDOW_PANE_NEAR_X - _WINDOW_WIPE_STANDOFF
                    and abs(head[1]) <= _WINDOW_GLASS_HALF_Y
                    and _WINDOW_GLASS_Z_LO <= head[2] <= _WINDOW_GLASS_Z_HI
                    and abs(float(s[0, 164])) >= _WINDOW_SWEEP_SPEED)


class H1HandCube(_H1HandBase):
    """Rotate a cube in each hand to a floating target cube's orientation.
    `h1hand-cube-v0`.

    Included despite known difficulties -- the spec's header carries the reasoning. The
    target orientation is MODEL state (`model.body_quat[-1]`, package's destination
    pattern: the target cube has no joint), drawn afresh each episode; it rides in
    s[177:181] and is restored into the model before anything evaluates or renders.
    The one task in this batch with a 500-step horizon.
    """

    name = "h1hand_cube"
    gym_id = "h1hand-cube-v0"
    obs_dim = 181
    action_dim = 61
    horizon = 500
    n_q, n_v = 90, 87
    #: The target orientation in s[177:181] is not restated as numbers: the target cube
    #: is drawn in the scene at exactly that pose (`_restore_extra`), and an Euler triple
    #: is neither legible at a glance nor unique (`_quat_angle`'s double cover).
    _ask = ("TURN BOTH CUBES TO MATCH", "THE FLOATING TARGET CUBE")
    extra_dims = 4

    #: A quarter of the (short) horizon spent matched. PROVISIONAL like every threshold
    #: in this file.
    success_threshold = 0.25

    _free_pos_low = (-2.0, -2.0, -0.5)
    _free_pos_high = (2.0, 2.0, 2.5)

    def _extra_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.full(4, -1.0), np.full(4, 1.0)

    def _extra_state(self) -> np.ndarray:
        return np.asarray(self._env.model.body_quat[-1], dtype=float).ravel().copy()

    def _restore_extra(self, tail: np.ndarray) -> None:
        """Write the target back into the MODEL, where upstream keeps it -- package's
        pattern, including the explicit raise: model state survives across calls, so a
        skipped restore silently leaves the previous episode's target in place.
        Normalised on the way in because `random_state` draws four raw uniforms and a
        non-unit body_quat makes the target cube's rendered pose meaningless; on a
        rolled-out tail the quaternion is already unit and this is the identity."""
        quat = np.asarray(tail, dtype=float).ravel()
        if quat.size != self.extra_dims:
            raise ValueError(
                f"{type(self).__name__}._restore_extra needs the {self.extra_dims} "
                f"components of the target quaternion and got {quat.size}. Skipping "
                "the restore would leave the previous episode's target in "
                "`model.body_quat[-1]` and make `_step` depend on what ran before it")
        self._env.model.body_quat[-1] = _quat_normalise(quat)

    def _task_reset(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`Cube.reset_model`: both hand cubes AND the target re-drawn as random Euler
        triples -- through upstream's own conversion, on ctx.rng rather than the global
        `np.random` (Room's reason). The draw order matches upstream's: left, right,
        target."""
        qpos = qpos.copy()
        qpos[79:83] = _euler_to_quat(rng.uniform(-3.14, 3.14, 3))
        qpos[86:90] = _euler_to_quat(rng.uniform(-3.14, 3.14, 3))
        self._env.model.body_quat[-1] = _euler_to_quat(rng.uniform(-3.14, 3.14, 3))
        return qpos

    def task_metric(self, traj: Any) -> float:
        """Steps with BOTH cubes matched to the target, over the full 500-step horizon.

        Matched means geodesic angle within 20 degrees -- `_quat_angle` goes through
        |dot|, so the quaternion sign upstream's Euclidean distance trips over cannot
        matter here. Both at once, because the task is both hands; the freeze-and-hold
        exploit (the closeness term is satisfied at spawn) scores 0.0 barring a lucky
        double draw. The denominator is the horizon because dropping a cube TERMINATES
        -- a visited-states mean would score an align-then-drop near 1.0.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        arrived = states[1:]
        left = _quat_angle(arrived[:, 79:83], arrived[:, 177:181])
        right = _quat_angle(arrived[:, 86:90], arrived[:, 177:181])
        both = (left <= _CUBE_ALIGN_TOL_RAD) & (right <= _CUBE_ALIGN_TOL_RAD)
        return float(min(np.count_nonzero(both) / _CUBE_HORIZON, 1.0))

    def _success_flag(self, s: np.ndarray) -> bool:
        s = np.asarray(s, dtype=float).reshape(1, -1)
        return bool((_quat_angle(s[:, 79:83], s[:, 177:181])[0] <= _CUBE_ALIGN_TOL_RAD)
                    and (_quat_angle(s[:, 86:90], s[:, 177:181])[0] <= _CUBE_ALIGN_TOL_RAD))


# --------------------------------------------------------------------------
# cabinet, bookshelf_hard (the two big object sets)
# --------------------------------------------------------------------------

#: `Cabinet.get_reward`'s four rung tests, adopted as task structure (the ladder's
#: terminal value IS the success terminal): the pull door and drawer joints past 95% of
#: their ranges, then each cube's centre inside its compartment box. `(x0, x1, y0, y1,
#: z0, z1)`; the top compartment shares the footprint at a higher shelf.
_CABINET_MID_BOX = (0.6, 1.2, -0.6, 0.6, 0.79, 1.09)
_CABINET_TOP_BOX = (0.6, 1.2, -0.6, 0.6, 1.39, 1.69)

#: What each rung of the ladder asks, in `Cabinet.get_reward`'s order -- the line
#: `legend_lines` names while that rung is current (`H1HandCabinet._legend_extra`). Indexed by
#: rung - 1; rung five is total success and asks nothing more. The wording is the
#: subtask's, with no `k/4`: a count of rungs climbed is progress, not an ask.
_CABINET_STEP_ASKS: Tuple[str, ...] = (
    "NOW: SLIDE THE DOOR OPEN",
    "NOW: PULL THE DRAWER OUT",
    "NOW: CUBE INTO THE MIDDLE",
    "NOW: CUBE UP INTO THE TOP",
)


def _cabinet_rung_done(rung: int, s: np.ndarray) -> bool:
    """Upstream's completion test for one ladder rung, on one state row.

    The joint addresses and the /0.4, /0.45 normalisers are `Cabinet.get_reward`'s own
    (its negative qpos indices resolved against the 109-qpos model): rung one is the
    pulling cabinet's sliding door (qpos 79), two the drawer (76), three the drawer
    cube (81:84) into the middle box, four the lateral cube (88:91) into the top box.
    """
    if rung == 1:
        return bool(abs(float(s[79]) / 0.4) > 0.95)
    if rung == 2:
        return bool(abs(float(s[76]) / 0.45) > 0.95)
    box = _CABINET_MID_BOX if rung == 3 else _CABINET_TOP_BOX
    pos = s[81:84] if rung == 3 else s[88:91]
    x0, x1, y0, y1, z0, z1 = box
    return bool(x0 <= pos[0] <= x1 and y0 <= pos[1] <= y1 and z0 <= pos[2] <= z1)


class H1HandCabinet(_H1HandBase):
    """Work a wall cabinet through four ordered subtasks: slide a door, pull a drawer,
    stow two cubes. `h1hand-cabinet-v0`.

    The reward is a LADDER whose position lives on the task object (1..5), pays
    +100*rung on each advance, and whose terminal value is the episode's only terminal
    -- there is NO fall terminal here, alone in the tier. The ladder rides in s[213],
    PRE-update (`_before_step`'s contract, maze's precedent), so `reference_reward`
    reproduces the advance bonuses at the states where upstream pays them. The latch is
    advanced by upstream's OWN `get_reward` called on the restored state -- called, not
    ported, so the thresholds cannot drift -- with its "Completed subtask" print
    muzzled.
    """

    name = "h1hand_cabinet"
    gym_id = "h1hand-cabinet-v0"
    obs_dim = 214
    action_dim = 61
    horizon = 1000
    n_q, n_v = 109, 104
    _ask = ("CABINET: 4 STEPS IN ORDER",)
    extra_dims = 1

    #: One rung of four. PROVISIONAL like every threshold in this file, and the
    #: benchmark's own SOTA sits at 8% of its success bar, so expect the floor.
    success_threshold = 0.25

    _free_pos_low = (-2.0, -2.0, -0.5)
    _free_pos_high = (2.0, 2.0, 2.5)

    def _extra_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.ones(1), np.full(1, 5.0)

    def _extra_state(self) -> np.ndarray:
        return np.array([float(self._env.task.current_subtask)])

    def _restore_extra(self, tail: np.ndarray) -> None:
        self._env.task.current_subtask = int(np.clip(round(float(tail[0])), 1, 5))

    def _before_step(self) -> None:
        """Upstream's ladder, advanced by upstream's own code on the restored state.

        `Cabinet.get_reward` is the only place the advance rule exists, so it is CALLED
        and its value discarded -- reimplementing the four thresholds here would be a
        second copy free to drift (basketball's scan was reimplemented for the opposite
        reason: its rule is one line and its reward pays no advance bonus). The print
        that upstream fires on each advance is muzzled; four lines per episode times a
        sweep is log spam with no reader.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            self._env.task.get_reward()

    def _legend_extra(self, tail: np.ndarray) -> List[str]:
        """The current rung's command, from `_CABINET_STEP_ASKS`. The ladder is
        upstream's phase machine -- which of the four subtasks is asked of the policy
        NOW -- so the line names that subtask and nothing else. Rung five is the success
        terminal and adds no line; the fixed ask stands alone on that last frame."""
        rung = int(np.clip(round(float(tail[0])), 1, 5))
        return list(_CABINET_STEP_ASKS[rung - 1:rung])

    def _task_reset(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`Cabinet.reset_model`: the ladder back to rung one. No randomness of its own
        -- the cubes and doors start at the keyframe."""
        self._env.task.current_subtask = 1
        return qpos

    def _terminated(self) -> bool:
        return self._terminated_after_advance()

    def task_metric(self, traj: Any) -> float:
        """Completed rungs over four, recomputed from the states in order.

        DELIBERATELY DISCRETE: a door 94% open reads 0.0 here while the reward pays
        0.8 * 0.94 per step for it -- which is precisely the documented
        `park_below_the_advance_threshold` exploit, where holding a rung's shaping
        income out-earns completing it. Partial credit would re-derive that shaping.
        One advance per arriving state, as upstream's once-per-step check does; the
        rung tests are upstream's own completion predicates (task structure -- the
        ladder's terminal value is the success terminal).
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        rung = 1
        for s in states[1:]:
            if rung <= 4 and _cabinet_rung_done(rung, s):
                rung += 1
        return float((rung - 1) / 4.0)

    def _success_flag(self, s: np.ndarray) -> bool:
        """This ARRIVING state completes the ladder -- upstream's own success, the event
        that ends the episode. The carried rung is pre-advance, so on the terminal state
        it reads 4 and the flag is rung four's own completion test on that state; a
        carried 5 (a state after the terminal, which `_step` does not emit) counts too.
        A `s[213] >= 4.5` test would be true only on an extra state past the terminal."""
        rung = int(round(float(s[213])))
        return bool(rung >= 5 or (rung == 4 and _cabinet_rung_done(4, s)))


#: `BookshelfBase.reset_model`'s five goal spots, its 0.15 m advance radius, and the
#: signed-body-index encoding its object list uses: id i in [-24, -13] is object
#: k = i + 24 of the twelve, whose free-joint block starts at qpos 76 + 7k. The ghost
#: markers are the last twelve bodies (i + 12, i.e. -12..-1).
_BOOKSHELF_GOALS = ((0.75, -0.25, 1.55), (0.8, 0.05, 0.95), (0.8, -0.25, 0.95),
                    (0.85, 0.05, 0.35), (0.85, -0.25, 0.35))
_BOOKSHELF_PLACE_RADIUS = 0.15
_BOOKSHELF_ID_LOW, _BOOKSHELF_ID_HIGH = -24, -13
_BOOKSHELF_OBJ_Q0 = 76

#: `BookshelfBase.carried_order_rgb`: the placement-order tint applied to the chosen
#: objects and their ghosts, pink through dark red.
_BOOKSHELF_ORDER_RGB = ((1.0, 0.8, 0.8), (1.0, 0.5, 0.5), (0.9, 0.2, 0.2),
                        (0.5, 0.0, 0.0), (0.2, 0.0, 0.0))


def _bookshelf_object_qpos(obj_id: int) -> slice:
    """The position slice of the free body a signed object index names."""
    k = int(obj_id) - _BOOKSHELF_ID_LOW
    q0 = _BOOKSHELF_OBJ_Q0 + 7 * k
    return slice(q0, q0 + 3)


class H1HandBookshelfHard(_H1HandBase):
    """Move five randomly chosen shelf objects to five drawn spots, one at a time, in
    order. `h1hand-bookshelf_hard-v0`.

    The largest model in the tier (84 scene DoF) and the largest observation tail: the
    reward's counter, its five chosen object ids and their five goals -- 21 slots at
    s[307:328] -- are all Python-side task state that every reward term reads. The
    counter rides PRE-update (maze's convention, for the +100*index bonuses); the ids
    and goals are constant within an episode. Upstream draws both from the GLOBAL
    `np.random`; this adapter draws them from ctx.rng, Room's reason.
    """

    name = "h1hand_bookshelf_hard"
    gym_id = "h1hand-bookshelf_hard-v0"
    obs_dim = 328
    action_dim = 61
    horizon = 1000
    n_q, n_v = 160, 147
    _ask = ("PLACE THE 5 TINTED OBJECTS", "ON THEIR GHOSTS, IN ORDER")
    extra_dims = 21

    #: One placement of five. PROVISIONAL like every threshold in this file.
    success_threshold = 0.2

    _free_pos_low = (-2.0, -2.0, -0.5)
    _free_pos_high = (2.0, 2.0, 2.5)

    def _extra_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.concatenate([[0.0], np.full(5, float(_BOOKSHELF_ID_LOW)),
                             np.tile([0.6, -0.5, 0.2], 5)])
        hi = np.concatenate([[5.0], np.full(5, float(_BOOKSHELF_ID_HIGH)),
                             np.tile([1.1, 0.3, 1.8], 5)])
        return lo, hi

    def _extra_state(self) -> np.ndarray:
        task = self._env.task
        return np.concatenate([
            [float(task.task_index)],
            np.asarray(task.bookshelf_objects[:5], dtype=float),
            np.asarray(task.placement_goals[:5], dtype=float).ravel()])

    def _restore_extra(self, tail: np.ndarray) -> None:
        """Counter, object list, goal list AND the scene dressing, all from the tail.

        The dressing half matters for the same two reasons it did on maze and package:
        the ghost markers' positions are MODEL state (`body_pos` of the last twelve
        bodies) that upstream writes only at reset, so a skipped restore leaves the
        previous episode's markers standing; and the placement-order tints are what a
        judged frame shows of the objective. Both are pure functions of the carried
        ids and goals. Upstream keeps a sixth duplicate entry on both lists (read once
        at task_index 5, in the step the terminal ends); rebuilt here the same way.
        """
        v = np.asarray(tail, dtype=float).ravel()
        if v.size != self.extra_dims:
            raise ValueError(
                f"{type(self).__name__}._restore_extra needs the {self.extra_dims} "
                f"values of the counter, object ids and goals and got {v.size}. "
                "Skipping the restore would leave the previous episode's task standing")
        task = self._env.task
        task.task_index = int(np.clip(round(v[0]), 0, 5))
        ids = np.clip(np.round(v[1:6]).astype(int),
                      _BOOKSHELF_ID_LOW, _BOOKSHELF_ID_HIGH)
        goals = v[6:21].reshape(5, 3)
        task.bookshelf_objects = np.append(ids, ids[-1])
        task.placement_goals = np.vstack([goals, goals[-1]])
        self._paint_targets(ids, goals)

    def _paint_targets(self, ids: np.ndarray, goals: np.ndarray) -> None:
        """`BookshelfBase.reset_model`'s dressing loops, replayed for a given draw:
        every object back to neutral grey with its ghost hidden, then each chosen
        object and its ghost tinted in placement order and the ghost moved to the
        goal."""
        task = self._env.task
        named = self._env.named
        for obj_name in task.object_names:
            for geom_name in task.object_names[obj_name]:
                named.model.geom_rgba[geom_name] = np.array([0.8, 0.8, 0.8, 0.0])
                named.model.geom_rgba[geom_name.replace("_vision", "")] = np.array(
                    [0.8, 0.8, 0.8, 1.0])
        keys = list(task.object_names.keys())
        for obj_id, goal, rgb in zip(ids, goals, _BOOKSHELF_ORDER_RGB):
            self._env.model.body_pos[int(obj_id) + 12] = goal
            for geom_name in task.object_names[keys[int(obj_id) + 12]]:
                named.model.geom_rgba[geom_name] = np.append(rgb, 0.2)
                named.model.geom_rgba[geom_name.replace("_vision", "")] = np.append(
                    rgb, 1.0)

    def _before_step(self) -> None:
        """Upstream's counter, advanced by upstream's own `get_reward` on the restored
        state -- cabinet's argument, without the print to muzzle."""
        self._env.task.get_reward()

    def _legend_extra(self, tail: np.ndarray) -> List[str]:
        """Where the CURRENT object is asked to go, read off the carried counter and
        goal list -- the same slots `task_metric` reads (`s[307]`, `s[313:328]`). Past
        the fifth placement the counter sits at its terminal value and the last goal
        stands, as upstream's own duplicate sixth entry does. The object ids are not
        printed: the chosen objects wear their placement-order tint in the scene."""
        k = int(np.clip(round(float(tail[0])), 0, 4))
        goals = np.asarray(tail[6:21], dtype=float).reshape(5, 3)
        return ["GOAL " + _legend_xyz(goals[k])]

    def _task_reset(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`BookshelfHard.reset_model`: five objects drawn without replacement, the
        five goal spots shuffled, the counter zeroed, the dressing painted -- from
        ctx.rng where upstream uses the global `np.random`."""
        goals = rng.permutation(np.asarray(_BOOKSHELF_GOALS, dtype=float))
        ids = rng.choice(np.arange(_BOOKSHELF_ID_LOW, _BOOKSHELF_ID_HIGH + 1), 5,
                         replace=False)
        task = self._env.task
        task.task_index = 0
        task.bookshelf_objects = np.append(ids, ids[-1])
        task.placement_goals = np.vstack([goals, goals[-1]])
        self._paint_targets(ids, goals)
        return qpos

    def _terminated(self) -> bool:
        return self._terminated_after_advance()

    def task_metric(self, traj: Any) -> float:
        """Ordered placements completed, over five, recomputed from the states.

        The ids and goals are read off the FIRST state's tail (constant within an
        episode) and the counter is re-derived: one advance per arriving state when
        the CURRENT object's centre is inside upstream's own 0.15 m radius of its
        goal. The hover-hand and prepaid-proximity exploits read 0.0 -- nothing pays
        short of the radius -- and a placed object later knocked off STAYS counted,
        exactly as upstream's counter keeps it; the spec's fly-through entry carries
        the honest reading of that.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        ids = np.clip(np.round(states[0, 308:313]).astype(int),
                      _BOOKSHELF_ID_LOW, _BOOKSHELF_ID_HIGH)
        goals = states[0, 313:328].reshape(5, 3)
        placed = 0
        for s in states[1:]:
            if placed >= 5:
                break
            obj = s[_bookshelf_object_qpos(ids[placed])]
            if float(np.linalg.norm(obj - goals[placed])) < _BOOKSHELF_PLACE_RADIUS:
                placed += 1
        return float(placed / 5.0)

    def _success_flag(self, s: np.ndarray) -> bool:
        """This ARRIVING state completes the fifth placement -- upstream's own success,
        the event that ends the episode. The carried counter is pre-advance, so on the
        terminal state it reads 4 and the flag is the fifth object inside upstream's
        radius of its goal, both read off this state's own tail; a carried 5 counts too.
        A `s[307] >= 4.5` test would be true only on an extra state past the terminal."""
        k = int(round(float(s[307])))
        if k >= 5:
            return True
        if k != 4:
            return False
        ids = np.clip(np.round(s[308:313]).astype(int), _BOOKSHELF_ID_LOW, _BOOKSHELF_ID_HIGH)
        goal = s[313:328].reshape(5, 3)[4]
        obj = s[_bookshelf_object_qpos(ids[4])]
        return bool(float(np.linalg.norm(obj - goal)) < _BOOKSHELF_PLACE_RADIUS)


# --------------------------------------------------------------------------
# package, door
# --------------------------------------------------------------------------


class H1HandPackage(_H1HandBase):
    """Carry a 0.28 kg package to a destination marker on the floor. `h1hand-package-v0`.

    The second task here whose observation is more than the simulator state, and the
    extra dimension is NOT on upstream's task object the way push's goal is -- it is
    MODEL state. `Package.reset_model` writes `self._env.model.body_pos[-1]`, so the
    destination lives in the model, survives `set_state` untouched, and would silently
    persist across a `reset()` if this adapter did not restore it. Body -1 is
    `package_destination` (id 72 of 73, verified on the composed model), and the
    `destination_loc` site sits `_PACKAGE_SITE_OFFSET` inside it.

    Ordering is what makes the restore correct rather than merely present: `_step` calls
    `_restore_extra` BEFORE `set_state`, and `set_state` runs `mj_forward`, which is what
    recomputes `site_xpos` from the body position just written. A restore after
    `set_state` would leave the site one step stale.

    Upstream draws both the package's x, y and the destination's from the GLOBAL
    `np.random` (Room's defect again); this adapter draws them from `ctx.rng`, so ours
    is reproducible from a seed and upstream's is not.
    """

    name = "h1hand_package"
    gym_id = "h1hand-package-v0"
    obs_dim = 167
    action_dim = 61
    horizon = 1000
    n_q, n_v = 83, 81
    _ask = ("CARRY THE PACKAGE TO GOAL",)
    extra_dims = 3

    #: Delivered on any arriving state, as a fraction of the episode. PROVISIONAL like
    #: every threshold in this file, and low because success TERMINATES: a delivering
    #: policy is inside the radius for one step of a thousand.
    success_threshold = 0.0005

    _free_pos_low = (-3.0, -3.0, -0.5)
    _free_pos_high = (3.0, 3.0, 3.0)

    def _extra_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(_PACKAGE_DEST_LOW), np.array(_PACKAGE_DEST_HIGH)

    def _extra_state(self) -> np.ndarray:
        """The destination site's world position, read out of `data` after the forward
        pass rather than reconstructed from `model.body_pos` -- the site offset is the
        model's to define, not ours to assume."""
        return np.asarray(
            self._env.named.data.site_xpos["destination_loc"], dtype=float).ravel().copy()

    def _restore_extra(self, tail: np.ndarray) -> None:
        """Write the destination back into the MODEL, where upstream keeps it.

        RAISES on a wrong-sized tail instead of skipping, and the difference is not
        defensive programming. `_step` is documented stateless -- it restores `s`, acts,
        and reads the arriving state -- and for this task the destination lives in
        `model.body_pos[-1]`, which is MODEL state that survives across calls. A skipped
        restore therefore does not fail; it silently leaves the PREVIOUS episode's
        destination in place, so `_step(s, a)` returns something that depends on what
        was stepped before it. That is the one property `test_step_is_a_function_of_its
        _arguments_and_the_warmstart_is_why` exists to hold, and a no-op is invisible to
        it because the test never supplies a malformed tail.

        The two sibling implementations already fail fast without saying so: `H1HandPush`
        assigns `self._env.task.goal[:] = ...`, which raises on a length mismatch, and
        `H1HandBasketball` indexes `tail[0]`, which raises when the tail is empty; this
        one must not be the only one that swallows it.

        Sized from `extra_dims` rather than the literal 3, so the check cannot drift
        away from the declared contract the class attribute states.
        """
        pos = np.asarray(tail, dtype=float).ravel()
        if pos.size != self.extra_dims:
            raise ValueError(
                f"{type(self).__name__}._restore_extra needs the {self.extra_dims} "
                f"values of the destination position and got {pos.size}. Skipping the "
                "restore would leave the previous episode's destination in "
                "`model.body_pos[-1]` and make `_step` depend on what ran before it")
        self._env.model.body_pos[-1] = pos - _PACKAGE_SITE_OFFSET

    def _legend_extra(self, tail: np.ndarray) -> List[str]:
        """The destination, off the same three slots `_restore_extra` writes back
        (the site's world position, z always 0.35)."""
        return ["GOAL " + _legend_xyz(tail)]

    def _task_reset(self, qpos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """`Package.reset_model`: the package's x, y ~ U(-2, 2) with z pinned to 0.35,
        and the destination body's x, y ~ U(-2, 2) with z 0.

        Both are moved, which is why a do-nothing policy cannot rely on the pair being
        close: the sample regions coincide, so about 0.2% of episodes begin already
        delivered (a 0.1-radius disc over a 4 x 4 square). Upstream's behaviour,
        reproduced rather than corrected, and recorded in the spec.
        """
        qpos = qpos.copy()
        qpos[76] = rng.uniform(-2.0, 2.0)
        qpos[77] = rng.uniform(-2.0, 2.0)
        qpos[78] = 0.35
        self._env.model.body_pos[-1] = np.array(
            [rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), 0.0])
        return qpos

    def task_metric(self, traj: Any) -> float:
        """Fraction of ARRIVING states with the package inside the destination radius.

        Delivery, not approach, and that choice is what falsifies this task's strongest
        local optimum. The shipped reward pays `package_height` (up to +1 per step for
        holding the package at or above z = 1) on top of `stand_reward * small_control`
        (about another 1), while success TERMINATES the episode for a one-off +1000. So
        a policy that carries the package to just outside the radius and stands there
        holding it up collects roughly 1.7 per step for a thousand steps -- about 1700,
        past the task's own `success_bar` of 1500 and well past what succeeding pays.
        Any metric shaped like "best approach" would score that hover near 1.0. This one
        scores it exactly 0.0.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        arrived = states[1:]
        d = np.linalg.norm(arrived[:, 76:79] - arrived[:, 164:167], axis=1)
        return float(np.mean(d < _PACKAGE_RADIUS))

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(np.linalg.norm(s[76:79] - s[164:167]) < _PACKAGE_RADIUS)


class H1HandDoor(_H1HandBase):
    """Open a hinged door and walk through the doorway. `h1hand-door-v0`.

    No extra dimensions: both task degrees of freedom are real joints in `qpos` --
    `door_hinge` at index 76 (range [0, 1.4], stiffness 50) and `door_hatch_hinge` at 77
    (range [0, 2], stiffness 2). Restoring `qpos` restores the door, which is why this
    task needs none of package's machinery.

    THE STRUCTURAL FACTS, measured on the adapter's own model (reading the joint list
    alone suggests otherwise): the
    panel IS latched by CONTACT -- the far-side `door_lock` cylinder on the hatch body
    enters the frame after 0.02 rad of hinge travel, and 60 N.m on the hinge opens the
    door 0.019 rad. Pressing the lever (`door_hatch_hinge`) past about 0.9 rad frees it;
    a lever-up door released from any angle >= 0.35 rad rests at 0.275 rad, its bolt on
    the frame's inner face. And the door swings TOWARD the robot -- the free edge is at
    x = 0.30 at 0.6 rad and x = -0.20 at 1.4 rad -- so it is a pull door that sweeps the
    spawn. The shipped reward pays 0.45 for hinge angle and 0.35 for passage while asking
    only 0.05 for the hatch, which is the reward-structure fact the spec's exploit
    inventory records; a scripted policy frees the latch on 10/10 seeds and cracks the
    door 0.1 rad, and passage (theta >= ~1.05 for a 0.45 m robot, with the robot
    standing where the door sweeps) is out of its reach.
    """

    name = "h1hand_door"
    gym_id = "h1hand-door-v0"
    obs_dim = 155
    action_dim = 61
    horizon = 1000
    n_q, n_v = 78, 77
    _ask = ("OPEN THE DOOR, WALK THROUGH",)

    #: A tenth of the episode spent through the doorway. PROVISIONAL like every
    #: threshold in this file.
    success_threshold = 0.1

    _free_pos_low = (-3.0, -3.0, -0.5)
    _free_pos_high = (3.0, 3.0, 3.0)

    def task_metric(self, traj: Any) -> float:
        """Fraction of arriving states with the pelvis past the far face of the door.

        Read from state and fixed scene geometry: the panel body sits at x = 0.8 with
        half-thickness 0.07, so `_DOOR_FAR_FACE` is 0.87 and a pelvis beyond it is
        through the doorway. It cannot be there while the door is shut, so traversal
        implies the door opened -- the metric needs no separate angle term to be honest
        about that. Upstream's `passage_reward` bound of 1.2 is deliberately NOT used:
        that is the reward's constant, and this file does not borrow them.

        What this falsifies: standing still scores 0.0 despite the shipped reward paying
        `0.1 * stand_reward * small_control` for it, and approaching the door scores 0.0
        despite `passage_reward`'s margin of 1 paying partial credit from x = 0.2 onward.
        What it deliberately does NOT falsify is the shoulder-barge -- see the class
        docstring and the spec: getting through the doorway IS the task, and whether the
        handle was used is the reward's preference rather than the task's definition.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        return float(np.mean(states[1:, 0] > _DOOR_FAR_FACE))

    def _success_flag(self, s: np.ndarray) -> bool:
        return bool(s[0] > _DOOR_FAR_FACE)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


@register("env", "h1hand_powerlift")
def h1hand_powerlift(ctx: Any) -> _H1HandBase:
    """HumanoidBench: lift a 104 kg barbell. Shipped reward pays for standing (§0, §3)."""
    return H1HandPowerlift()


@register("env", "h1hand_hurdle")
def h1hand_hurdle(ctx: Any) -> _H1HandBase:
    """HumanoidBench: run a hurdle course. Shipped reward does not penalise hurdle
    contact (§0, §3)."""
    return H1HandHurdle()


@register("env", "h1hand_room")
def h1hand_room(ctx: Any) -> _H1HandBase:
    """HumanoidBench: tidy a room. Shipped reward minimises positional variance (§0, §3)."""
    return H1HandRoom()


@register("env", "h1strong_highbar_hard")
def h1strong_highbar_hard(ctx: Any) -> _H1HandBase:
    """HumanoidBench: rotate around a high bar. Shipped reward pays for hanging (§0, §3)."""
    return H1StrongHighBarHard()


@register("env", "h1hand_push")
def h1hand_push(ctx: Any) -> _H1HandBase:
    """HumanoidBench: push a box across a table to a per-episode target (§0, §3)."""
    return H1HandPush()


@register("env", "h1hand_basketball")
def h1hand_basketball(ctx: Any) -> _H1HandBase:
    """HumanoidBench: catch a basketball and throw it through a hoop (§0, §3)."""
    return H1HandBasketball()


@register("env", "h1hand_walk")
def h1hand_walk(ctx: Any) -> _H1HandBase:
    """HumanoidBench: walk forward at 1 m/s on open ground (§0, §3)."""
    return H1HandWalk()


@register("env", "h1hand_crawl")
def h1hand_crawl(ctx: Any) -> _H1HandBase:
    """HumanoidBench: crawl a 16 m tunnel whose ceiling is too low to walk under, on
    the Shadow-Hands embodiment (§0, §3)."""
    return H1HandCrawl()


@register("env", "h1hand_stand")
def h1hand_stand(ctx: Any) -> _H1HandBase:
    """HumanoidBench: stand still and upright. `_move_speed = 0` selects the reward's
    `dont_move` branch, which prices only horizontal motion (§0, §3)."""
    return H1HandStand()


@register("env", "h1hand_run")
def h1hand_run(ctx: Any) -> _H1HandBase:
    """HumanoidBench: run forward at 5 m/s -- walk's reward against a five-times-larger
    bound, with walk's `success_bar` left unraised (§0, §3)."""
    return H1HandRun()


@register("env", "h1hand_stair")
def h1hand_stair(ctx: Any) -> _H1HandBase:
    """HumanoidBench: climb a repeating staircase. Descending the far side pays the same
    as climbing (§0, §3)."""
    return H1HandStair()


@register("env", "h1hand_slide")
def h1hand_slide(ctx: Any) -> _H1HandBase:
    """HumanoidBench: climb a repeating ramp -- Stair's reward byte for byte, a different
    asset (§0, §3)."""
    return H1HandSlide()


@register("env", "h1hand_sit_simple")
def h1hand_sit_simple(ctx: Any) -> _H1HandBase:
    """HumanoidBench: sit on a fixed chair and stay seated (§0, §3)."""
    return H1HandSitSimple()


@register("env", "h1hand_cabinet")
def h1hand_cabinet(ctx: Any) -> _H1HandBase:
    """HumanoidBench: four ordered cabinet subtasks. Parking below a rung's advance
    threshold out-earns completing it (§0, §3)."""
    return H1HandCabinet()


@register("env", "h1hand_bookshelf_hard")
def h1hand_bookshelf_hard(ctx: Any) -> _H1HandBase:
    """HumanoidBench: place five randomly chosen shelf objects at five drawn spots, in
    order. A parked hand near the current object earns per-step income (§0, §3)."""
    return H1HandBookshelfHard()


@register("env", "h1hand_balance_simple")
def h1hand_balance_simple(ctx: Any) -> _H1HandBase:
    """HumanoidBench: balance on a see-saw board over a fixed sphere. The reward never
    reads the board (§0, §3)."""
    return H1HandBalanceSimple()


@register("env", "h1hand_balance_hard")
def h1hand_balance_hard(ctx: Any) -> _H1HandBase:
    """HumanoidBench: balance on a board over a free-rolling sphere -- the same
    board-blind reward on a harder model (§0, §3)."""
    return H1HandBalanceHard()


@register("env", "h1hand_window")
def h1hand_window(ctx: Any) -> _H1HandBase:
    """HumanoidBench: wipe a window with a long-handled tool. Shaking the tool in the
    air collects the motion term anywhere (§0, §3)."""
    return H1HandWindow()


@register("env", "h1hand_cube")
def h1hand_cube(ctx: Any) -> _H1HandBase:
    """HumanoidBench: rotate a cube in each hand to a target orientation. The reward's
    alignment kernel is Euclidean over quaternions and can punish exact success (§0,
    §3)."""
    return H1HandCube()


@register("env", "h1hand_sit_hard")
def h1hand_sit_hard(ctx: Any) -> _H1HandBase:
    """HumanoidBench: sit on a loose, tippable chair from a randomised spawn. The
    reward's height band is absolute while the chair moves (§0, §3)."""
    return H1HandSitHard()


@register("env", "h1hand_reach")
def h1hand_reach(ctx: Any) -> _H1HandBase:
    """HumanoidBench: put the left hand on a per-episode target point. Pays within a
    full metre and never terminates (§0, §3)."""
    return H1HandReach()


@register("env", "h1hand_pole")
def h1hand_pole(ctx: Any) -> _H1HandBase:
    """HumanoidBench: weave through a 41 m pole forest at 0.5 m/s. Nothing stops a
    robot walking around it (§0, §3)."""
    return H1HandPole()


@register("env", "h1hand_maze")
def h1hand_maze(ctx: Any) -> _H1HandBase:
    """HumanoidBench: walk an L-shaped maze via fixed checkpoints. The first checkpoint
    is the spawn and pays +100 for existing (§0, §3)."""
    return H1HandMaze()


@register("env", "h1hand_package")
def h1hand_package(ctx: Any) -> _H1HandBase:
    """HumanoidBench: carry a package to a floor marker. Shipped reward pays per step
    for holding it up, while success terminates (§0, §3)."""
    return H1HandPackage()


@register("env", "h1hand_door")
def h1hand_door(ctx: Any) -> _H1HandBase:
    """HumanoidBench: open a door and walk through. Nothing latches the hinge to the
    handle (§0, §3)."""
    return H1HandDoor()
