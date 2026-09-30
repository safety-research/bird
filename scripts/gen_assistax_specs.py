#!/usr/bin/env python3
"""Write `tasks/assistax_<task>/shared_spec.yaml` for every row in `bird/envs/assistax.py`.

    uv run --no-sync python3 scripts/gen_assistax_specs.py            # (re)write every row
    uv run --no-sync python3 scripts/gen_assistax_specs.py --check    # exit 1 if any is stale
    uv run --no-sync python3 scripts/gen_assistax_specs.py --tasks feeding   # one row

Every row is one of upstream's five tasks: the reward block is upstream's transcribed, the
provenance cites upstream's files, and the MJX comparison lines are emitted.

WHY GENERATED: a spec whose `flat_fields` table is 84-137 entries long, each naming one
`qpos`/`qvel` entry in the order the MODEL lays them out, is a spec that drifts from its
adapter with nothing failing -- `SpecEnvAdapter._apply_spec` checks the COUNT of flat_fields
against `obs_dim`, not the names. So the table is read off the live model here, the
`env.reset` block is MEASURED (24 seeded resets: which columns move is what the block says
moves), and the authored part -- the prose, the goal sentence, the exploits -- is the
`_AUTHORED` dict below. The output is committed; `--check` verifies the committed specs are
what the generator emits, as
`tests/test_assistax.py::test_the_committed_specs_are_what_the_generator_emits` does.

The rules the output obeys are the catalogue's (`tasks/README.md`):
`env_prose` describes the ENVIRONMENT and `l_task` states the GOAL and the action
semantics -- never an index, a tolerance, or a sign convention, because `l_task` reaches
every method while `env_prose` is gated on `generate.context.env_spec`.
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spec_provenance import (  # noqa: E402
    check_under_foreign_wheel, measured_mujoco, override_entries,
    recorded_mujoco, refuse_foreign_write, unjudgeable_under)
from bird.envs import assistax as ax  # noqa: E402
from bird.envs.base import EnvAdapter  # noqa: E402
from bird.policies import reference_for_env  # noqa: E402

#: This tier's `pyproject.toml` extra -- the one place it names itself. The mujoco
#: VERSION is never written here: it is read from that extra at run time
#: (`spec_provenance.pinned_mujoco`), so a pin move is one edit, in pyproject.
EXTRA = "assistax"

UPSTREAM_COMMIT = "a7d94f4e20636b9b0c07370344b8a2b521db58b2"
DATE = "2026-09-08"
_AX_PATH = "bird/envs/assistax.py"

_UPSTREAM_FILE = {
    "scratchitch": "assistax/envs/scratchitch.py", "bedbathing": "assistax/envs/bedbathing.py",
    "armmanipulation": "assistax/envs/armmanipulation.py", "feeding": "assistax/envs/feeding.py",
    "teethbrushing": "assistax/envs/teethbrushing.py",
}
_SUPPORT = {"scratchitch": "wheelchair", "feeding": "wheelchair", "teethbrushing": "wheelchair",
            "bedbathing": "bed", "armmanipulation": "bed"}
_CAMERA_MODE = {"wheelchair": "tracking", "bed": "fixed"}
_CAMERA_NOTE = {
    "wheelchair": "the scene's `default` camera: `targetbodycom` on the humanoid, from (1.3, 2.1, "
                  "1.8) -- it re-aims at the human's centre of mass, so the framing follows the "
                  "human's slow settling into the chair and the Panda's base stays in frame",
    "bed": "the scene's `default` camera: a fixed camera at (1.5, -2.8, 1.5) looking obliquely "
           "across and down the bed from the robot's side and beyond its foot (view direction "
           "(-0.47, 0.81, -0.34) from its xyaxes), the whole Panda and the lying human in frame",
}
#: The metric each row implements, in the words the spec carries (`continuous_success.raw`).
_FITNESS_EXPR = {
    "scratching_fraction": "fraction of arriving states with dist_tool_target < 0.05 m, the "
                           "contact force the scratcher applies to the human's right arm or hand "
                           ">= 0.5 N and the scratcher tip's speed >= 0.02 m/s",
    "wiped_fraction": "n_wiped / 52 at the FINAL arriving state: a point is marked wiped on any "
                      "step where the wiper centre is within 0.1 m of it while the wiper (pad or "
                      "base cube) is in contact with the human's right upper arm, forearm or hand "
                      "(any non-zero summed tool-human contact force; upstream's marking rule), "
                      "and stays wiped",
    "arm_at_waist_fraction": "fraction of arriving states with dist_forearm_waist_target < 0.08 m",
    "spoon_at_mouth_fraction": "fraction of arriving states with dist_tool_mouth < 0.04 m and "
                               "tip_force_mag < 5 N",
    "brushing_fraction": "fraction of arriving states with dist_tool_mouth < 0.03 m, tip_force_mag "
                         ">= 0.1 N and brush_tangential_speed >= 0.01 m/s",
}
_FITNESS_UNIT = {"wiped_fraction": "fraction of the 52 points"}

#: Units of each task's `task_fields` block (the tail after the 13 common columns), in
#: the order the fields carry them -- not a five-task union.
_TASK_FIELD_UNITS = {
    "scratchitch": "m (positions and distance)",
    "bedbathing": "m (positions and distance), count (n_wiped), m (arm positions)",
    "armmanipulation": "m (positions and distances), dimensionless (hook_rot_err, a Frobenius norm), m (arm positions)",
    "feeding": "m (positions and distance), cosine (spoon_up_dot_world_up), N (tip_force_mag)",
    "teethbrushing": "m (positions and distance), cosine (bristle_dot_to_mouth), N (tip_force_mag), m/s (brush_tangential_speed)",
}
#: The `task_slots` block's description where the slots are too many to list one by one.
_SLOT_BLOCK_DESC = {
    "bedbathing": "the 52 wipe flags, 1 = still to wipe (26 upper-arm points then 26 lower-arm points)",
}

#: MEASURED: mean `task_metric` over n=30
#: uniform-random episodes, `scripts/measure_anchors.py --env assistax_<task> --n 30
#: --no-write`, seed 0; `(mean, half-width, estimator)`. None = not yet measured (the generator then writes an explicit absence).
_RANDOM_ANCHORS: Dict[str, Any] = {
    # All five all-zero over 30 episodes (every episode scored exactly 0.0, Wilson at n=30
    # is 0.0568): a random Panda waves beside the human and never brings the tool to the
    # target -- the smoke rollouts measured zero tool-human contact force in 300 steps on
    # every task. Measured through `measure_anchors._episode`/`interval` on the
    # spec-less adapter (the bootstrap path
    # `measure_anchors.measure_env` documents: measure first, then write the file).
    "scratchitch": (0.0, 0.0568, "wilson"), "bedbathing": (0.0, 0.0568, "wilson"),
    "armmanipulation": (0.0, 0.0568, "wilson"), "feeding": (0.0, 0.0568, "wilson"),
    "teethbrushing": (0.0, 0.0568, "wilson"),
}
#: MEASURED against upstream's own package (jax
#: 0.8.0, mujoco 3.4.0 CPU): the MJX control-step time, and the max |qpos difference| between
#: the MJX solve and this port's C-MuJoCo solve from the same keyframe under zero actions,
#: after 1, 10 and 100 control steps. Provenance for the "same model, different physics" claim.
_MJX_STEP_MS = {"scratchitch": 506.0, "bedbathing": 513.2, "armmanipulation": 644.4,
                "feeding": 828.4, "teethbrushing": 729.7}
_MJX_VS_C = {"scratchitch": (0.0063, 0.0245, 0.0369), "bedbathing": (0.0026, 0.0234, 0.0682),
             "armmanipulation": (0.0090, 0.1263, 0.1881), "feeding": (0.0033, 0.0201, 0.0605),
             "teethbrushing": (0.0033, 0.0197, 0.0606)}

#: task -> the authored prose. Every key is
#: REQUIRED and the generator refuses a task without one rather than writing a placeholder.
#:   env_prose, l_task, scene, success_criterion_prose, exploits [(name, mechanism)],
#:   telemetry [str], legibility_note, upstream_divergence, goal_sentence,
#:   prompt (field, verbatim quote), mentions [roles the quote says are drawn]
_AUTHORED: Dict[str, Dict[str, Any]] = (  # noqa: E501 -- prose, verbatim
    {'armmanipulation': {'env_prose': "A Franka Panda arm stands beside a bed at the human's right. "
                                      'The human lies supine, right arm out toward the robot, forearm '
                                      'at the mattress edge and pressed into it. The human is passive: '
                                      'the body rests under gravity, held by joint stiffness and its '
                                      "idle trunk and hip actuators' restoring torques, and can be "
                                      'pushed; the right arm resists only lightly. The tool is an '
                                      "L-shaped hook whose reference point (the crook) sits in the L's "
                                      'angle beneath the lip; it collides with the mattress. Two body '
                                      'points matter: a hook point midway along the right forearm and '
                                      'a waist target beside the right hip, about 4 cm above the '
                                      'mattress, initially about 33 cm away toward the feet and in '
                                      'toward the body; the crook starts about 31 cm from the hook '
                                      'point. The action is seven numbers in [-1, 1], one per Panda '
                                      'joint: normalised position targets for stiff servos (about pi '
                                      "times the value in radians, clipped to the joint's range), not "
                                      'torques. Only the wrist links, hand, fingers and hook collide; '
                                      'other links pass through everything. A control step is 4 ms '
                                      '(250 Hz); an episode is 1000 steps (4 s) and nothing ends it '
                                      'early. The observation is the whole simulator state -- the '
                                      "human's root position and quaternion, every hinge angle (human, "
                                      'then the seven Panda joints), then all velocities, same order '
                                      '-- then derived fields: crook position, hook orientation '
                                      'quaternion, crook velocity, the contact force in newtons the '
                                      "hook applies to the right arm and hand, the hook point's and "
                                      "waist target's positions, the two distances (crook to hook "
                                      'point, hook point to waist target), a hook-to-hook-point '
                                      'rotation mismatch (zero when aligned), and the right upper-arm '
                                      'and forearm positions. Every episode starts from the same lying '
                                      'pose and robot pose with small random jitter on every position '
                                      'and velocity coordinate; both target points are fixed to the '
                                      'body.',
                         'exploits': [('drag_not_hook',
                                       "The check reads only the forearm hook point's distance to the "
                                       'waist site; dist_tool_hook_target and hook_rot_err are in the '
                                       "observation and in upstream's reward but in no clause. The "
                                       'forearm can therefore be shoved across the mattress with the '
                                       "flat of the plate, the outside of the lip, or the Panda's own "
                                       'wrist, hand and finger collision geoms (which carry the same '
                                       'collision masks as the hook, so they touch the human exactly '
                                       'as it does, and whose force on the human is not in the '
                                       "observation's force field, which sums only the two hook geoms' "
                                       'contacts), and the arm is never hooked; the check pays the '
                                       'same.'),
                                      ('slide_along_the_mattress_no_lift',
                                       'The waist site is at z 0.713 m, about 4 cm above the mattress '
                                       'top at 0.67 m, and a forearm of radius 0.031 m lying on the '
                                       'mattress has its centreline near 0.67 + 0.031 m -- inside the '
                                       '0.08 m ball. A purely horizontal drag of the forearm across '
                                       'the mattress toward the hip, roughly the 0.33 m initial '
                                       'separation, satisfies the check on every step it holds, with '
                                       'the arm never leaving the mattress, although the task is '
                                       'stated as a lift to waist level.'),
                                      ('pin_with_unbounded_force',
                                       'The check has no force term and pays per step, so once the '
                                       'hook point is inside the ball, pressing the forearm down into '
                                       'the mattress or wedging it against the hip holds it there for '
                                       "the remaining steps, and doing so at the Panda's full joint "
                                       'torque (up to 87 N.m on the first four joints) scores exactly '
                                       "as a gentle placement does; upstream's ee_force is info-only, "
                                       'so the reference reward does not penalise it either.'),
                                      ('move_the_hips_to_the_arm',
                                       'The waist site is fixed to the pelvis of a free-floating '
                                       'passive body with no weld to the bed, and the check is a '
                                       'relative distance, so pushing the pelvis or legs sideways '
                                       'bends the trunk at the abdomen joints -- held only by the idle '
                                       "actuators' restoring torque of 40-45 N.m per radian plus 10-20 "
                                       'N.m per radian of joint stiffness, well within what the '
                                       "Panda's 87 N.m proximal joints can apply -- and swings the "
                                       'pelvis-fixed waist site toward the arm without the arm moving; '
                                       'combined with a shorter drag, the point enters the ball while '
                                       'the arm stays well out at the side.')],
                         'factory_doc': 'Assistax `armmanipulation` (ArmManipulation): a Panda arm '
                                        'with an L-shaped hook lifts the weak right forearm of a '
                                        'passive human lying on a bed to a waist-level target beside '
                                        'the hip, plain-MuJoCo port.',
                         'goal_sentence': 'none -- both targets are sites fixed to the body and '
                                          'identical in every episode: hook_target midway along the '
                                          'right forearm and arm_target on the pelvis beside the right '
                                          'hip at waist level, moving only with the body; '
                                          "arm_at_waist_fraction compares the forearm's hook_target "
                                          'point against the arm_target site',
                         'l_task': 'Help the person lying in the bed move their weak right arm: slide '
                                   "the L-shaped hook on the Panda's hand under the forearm where it "
                                   'rests at the edge of the mattress, then lift and carry the forearm '
                                   'in across the bed and bring it to rest beside the right hip at '
                                   'waist level, and keep it there for as much of the episode as you '
                                   'can. The arm is commanded with seven joint-position targets, one '
                                   'per Panda joint, each a normalised target angle that a stiff '
                                   'position servo tracks rather than a torque; the person is passive '
                                   'and will neither help nor resist beyond the light stiffness and '
                                   'damping of their own arm.',
                         'legibility_note': "The scene's `default` camera is fixed in the world at "
                                            '(1.5, -2.8, 1.5) and looks obliquely across and down the '
                                            "bed from the robot's side, beyond the foot end, so the "
                                            'lying human, the whole Panda and the hook are in frame '
                                            'with the right arm on the near side, about 3 m from the '
                                            'camera. Both targets are rendered: an opaque green '
                                            'capsule of radius 0.04 m wrapping the mid-forearm '
                                            '(hook_target) and a half-transparent cyan sphere of the '
                                            'same radius beside the right hip (arm_target), so a VLM '
                                            'can read where the arm lies (out at the mattress edge or '
                                            'in beside the hip), whether the green marker has reached '
                                            'the cyan sphere, whether the arm was lifted or dragged '
                                            'along the mattress, whether the hook is at the forearm or '
                                            'elsewhere in the air, and whether the torso or legs have '
                                            'been shoved. It cannot tell whether the hook is engaged '
                                            'under the forearm or merely beside it -- the plate is 12 '
                                            'mm thick and 60 mm wide and the gap under the arm is '
                                            'below what 448x320 resolves at that distance -- nor read '
                                            'contact force or speed from a still. The legend prints '
                                            'only the task name -- neither the hook-to-forearm '
                                            'distance, the forearm-to-waist distance the check gates '
                                            'on nor any force is printed -- so the gated distance '
                                            "reaches a judge only through the two markers' "
                                            'separation.',
                         'mentions': ['agent_pose', 'agent_velocity'],
                         'prompt': ('env_prose',
                                    'Every episode starts from the same lying pose and robot pose with '
                                    'small random jitter on every position and velocity coordinate; '
                                    'both target points are fixed to the body.'),
                         'scene': "A single bed seen obliquely from the robot's side and from beyond "
                                  'its foot, looking down toward the head: a person lying on their '
                                  'back with their head on a pillow, right arm out at the mattress '
                                  'edge with a green marker around the forearm and a translucent cyan '
                                  'ball beside the right hip, and a Franka Panda arm standing on the '
                                  'floor beside the bed holding a blue L-shaped hook with a red lip.',
                         'success_criterion_prose': 'arm_at_waist_fraction is a check this repository '
                                                    'wrote, not a benchmark criterion: upstream '
                                                    'Assistax ships no success test for this task and '
                                                    'reports only the return of its own reward. It is '
                                                    "the fraction of the episode's 1000 arriving "
                                                    'states on which the hook_target point midway '
                                                    "along the human's right forearm is within 0.08 m "
                                                    'of the arm_target site beside the right hip at '
                                                    'waist level -- one distance clause, with no '
                                                    'requirement that the hook be engaged, on contact '
                                                    'force, or on how the arm got there. success() is '
                                                    'this fraction reaching 0.3, a provisional '
                                                    'threshold. The check gates on the same '
                                                    'forearm-to-waist distance that dominates '
                                                    "upstream's reward (its 10 x exp(-d^2/0.1) term), "
                                                    'so it is not independent of that reward the way a '
                                                    "benchmark's success flag would be; and the waist "
                                                    'target sits only about 4 cm above the mattress '
                                                    'top, so the check does not by itself require the '
                                                    'forearm to leave the mattress. A uniform-random '
                                                    'policy scores exactly 0 over 30 episodes.',
                         'telemetry': ['dist_forearm_waist_target',
                                       'dist_tool_hook_target',
                                       'hook_rot_err',
                                       'contact_force_magnitude',
                                       'hook_target_z'],
                         'upstream_divergence': 'Physics: the same vendored MJCF at the 0.001 s '
                                                'timestep armmanipulation.py sets in code (the XML '
                                                'default is 0.002 s), 4 substeps per control step (250 '
                                                'Hz, 4 s episodes), solved by C MuJoCo 3.3.0 with the '
                                                "XML's defaults (Newton, 100 iterations) where "
                                                'upstream runs MJX with iterations=1, ls_iterations=1 '
                                                "and EULERDAMP disabled; measured against upstream's "
                                                'own package from the keyframe under zero actions, the '
                                                'two solves part by a max |dq| of 0.009 after one '
                                                'control step, 0.126 after 10 and 0.188 after 100, and '
                                                'MJX costs 644 ms per control step on CPU against 0.5 '
                                                'ms here. The human is a zero-action passive body, and '
                                                "on this task alone that is exactly upstream's human "
                                                'too: its three trained right-arm actuators '
                                                '(right_shoulder1, right_shoulder2, right_elbow) have '
                                                'gear 0 in this scene and apply no torque, so '
                                                "upstream's partner policy touches nothing but the "
                                                '1e-6 control cost. Neither side terminates early '
                                                "(upstream's step sets done = 0 and its EpisodeWrapper "
                                                'truncates at 1000 steps). The observation is the '
                                                'whole state (qpos, qvel, derived tool, force, target '
                                                "and arm fields) rather than upstream's 66-dim vector "
                                                'sliced into a 29-index robot view and a 40-index '
                                                'human view that overlap at index 28 and run past the '
                                                "end; upstream's qpos slices are indexed by joint id, "
                                                "so its 'robo_joint_angles' are the human's six arm "
                                                "hinges, the Panda's own angles appear nowhere, and "
                                                'both normalisations are sign-inverted by swapped '
                                                'range columns -- none of which this port reproduces. '
                                                "The reference reward transcribes upstream's four "
                                                'terms at the constructor defaults -- waist 10 x '
                                                'exp(-d^2/0.1), hook 1 x (1 - tanh(d/0.1)), rotation '
                                                '0.1 x the Frobenius norm of the rotation-matrix '
                                                'difference (a bonus on MISalignment, transcribed and '
                                                'not corrected) and a 1e-6 control cost over the 7 '
                                                "robot actions rather than upstream's 24-vector -- "
                                                'with the contact force summed over the live hook-arm '
                                                'contacts (mj_contactForce, world frame, hook geoms '
                                                'against the right upper arm, forearm and hand) '
                                                'instead of the norm of a 6-component wrench (forces '
                                                'and torques summed together) read from four fixed MJX '
                                                'contact slots 56/57/60/62, hard-coded under a TODO, '
                                                'whose geom pair is a property of the MJX collision '
                                                'enumeration and was not resolved here; the force is '
                                                'info-only upstream and enters neither reward. '
                                                "Do-nothing earns about 3.63 per step of upstream's "
                                                'reward at the keyframe pose (the waist term alone '
                                                'pays 3.40 with the arm at rest) -- a snapshot rather '
                                                'than a held pose, since a zero action commands the '
                                                "Panda's servos toward their own zero angles and moves "
                                                'the arm off the keyframe -- a floor the check does '
                                                'not share: a random policy scores 0.'},
     'bedbathing': {'env_prose': "A Franka Panda arm stands beside a single bed, at the human's right, "
                                 'level with the shoulder and elbow. The human lies supine, head on a '
                                 "pillow, right arm on the mattress on the robot's side, upper arm "
                                 'angled outward, forearm along the body. The human is passive: no '
                                 'joint is commanded, so the body rests on the mattress under gravity '
                                 "and its joint stiffness, and the robot can push it. The Panda's tool "
                                 'is a wiper: a red square pad, 10 cm on a side and 1.2 cm thick, on a '
                                 "small base cube, its reference point at the pad's centre. Fifty-two "
                                 "wipe points lie on the arm's surface, twenty-six in one helical turn "
                                 'around each of the upper arm and the forearm; they move with the arm '
                                 'and are not drawn. The action is seven numbers in [-1, 1], one per '
                                 'Panda joint: normalised position targets for stiff servos (about pi '
                                 "times the value in radians, clipped to the joint's range), not "
                                 'torques. A control step is 8 ms; an episode is 1000 steps (8 s) and '
                                 'nothing ends it early. The observation is the whole simulator state, '
                                 "then derived fields: the human's root pose (position and "
                                 'quaternion), every hinge angle (human then Panda), all velocities, '
                                 'the 52 wipe flags (1 still to wipe, 0 wiped, upper arm first; a flag '
                                 'flips when the wiper, touching the human, passes close over its '
                                 "point, and never back), the wiper's position, orientation quaternion "
                                 'and velocity, the contact force it applies to the human in newtons, '
                                 "the nearest unwiped point's position and distance (a large sentinel "
                                 'once all are wiped), the wiped count, and the upper-arm and forearm '
                                 'body positions. Every episode starts from the same lying pose and '
                                 'robot pose with small random jitter on every position and velocity '
                                 "coordinate, the human's root pose included; the 52 points are fixed "
                                 'to the arm and identical in every episode.',
                    'exploits': [('press_dont_sweep',
                                  'The marking radius is 0.1 m from the wiper centre while the arm '
                                  'capsules have radii of only 0.04 m (upper arm) and 0.031 m '
                                  '(forearm) and adjacent points along each helix sit 0.0147 m and '
                                  '0.0134 m apart, so one pad pressed onto the top of the arm marks a '
                                  'whole run of points at once, including those on the underside it '
                                  'never touched. A few firm presses spaced along each segment, with '
                                  'no sweeping motion at all, can clear the 0.5 threshold.'),
                                 ('base_cube_or_hand_contact_gate',
                                  'The contact clause is any non-zero summed force between either tool '
                                  'geom (the red pad or the base cube behind it) and the upper-arm, '
                                  'forearm or hand geoms, with no requirement on direction, magnitude '
                                  'or which face touches. A corner of the base cube jabbed edge-on '
                                  'into the arm, or the tool resting against the hand while its centre '
                                  'sits within 0.1 m of forearm points, marks those points exactly as '
                                  'a flat wipe would.'),
                                 ('shove_the_passive_arm',
                                  'The human is uncommanded and the mattress is the only thing under '
                                  'the arm, so the robot can press the arm into the mattress or drag '
                                  'the forearm sideways across it; the check has no force ceiling, no '
                                  'gentleness term and no notion of the arm staying put, and points '
                                  "carried to the wiper by the arm's own displacement count the same "
                                  'as points the wiper travelled to.'),
                                 ('one_segment_clears_the_bar',
                                  'The threshold is 0.5 of 52 points and each segment carries exactly '
                                  '26, so wiping only the forearm (or only the upper arm), or only the '
                                  "26 points nearest the robot's base, passes success with the other "
                                  'half of the arm untouched: the check counts flags, not coverage of '
                                  'both segments.')],
                    'factory_doc': 'Assistax `bedbathing` (BedBathing): a Panda arm with a wiper pad '
                                   'wipes 52 points on the right arm of a passive human lying on a '
                                   'bed, plain-MuJoCo port.',
                    'goal_sentence': "none -- the 52 points are fixed in the two arm capsules' own "
                                     'frames (26 per segment, one helical turn on each surface) and '
                                     'identical in every episode; wiped_fraction counts how many of '
                                     'their flags have flipped by the final step',
                    'l_task': 'Give the person a bed bath: bring the wiper pad into contact with their '
                              'right arm and sweep it over the whole surface of the upper arm and the '
                              'forearm, so that every one of the fifty-two wipe points has been passed '
                              'over while the wiper was pressing on the arm. The robot is commanded '
                              'with seven joint-position targets, one per Panda joint, which stiff '
                              'position servos track; the person is passive and will neither help nor '
                              'move out of the way.',
                    'legibility_note': "The scene's `default` camera is fixed in the world at (1.5, "
                                       "-2.8, 1.5) and looks across and down the bed from the robot's "
                                       'side, beyond the foot end, so the lying human, the whole Panda '
                                       'and the red pad are all in frame and the right arm is on the '
                                       'near side rather than behind the torso. A VLM can read whether '
                                       'the pad is over the arm or elsewhere in the air, roughly where '
                                       'along the arm it is, whether the arm has been shoved off its '
                                       'resting line or the body disturbed, and gross pad motion '
                                       'between frames. It cannot see the 52 wipe points (they are '
                                       'virtual: no geom, no site, nothing is rendered for them here '
                                       'or upstream), which of them remain, or anything on the '
                                       'underside of the arm; a small blue sphere on the upper arm '
                                       '(radius 0.01 m) is a leftover marker from the scratchitch '
                                       'scene, not a wipe point, and reads as a target. The legend '
                                       'prints only the task name: the wiped count (which IS the '
                                       'metric, `task_metric * 52`), the distance to the nearest '
                                       'unwiped point and the contact force are deliberately not '
                                       'printed, so a judge has no route to the count at all and '
                                       'wiping progress is not legible from a frame on this task. '
                                       'At 448x320 from that distance '
                                       "the 10 cm pad is small in frame and a gap of the pad's own 1.2 "
                                       'cm thickness between pad and skin is below what the frame '
                                       'resolves, so touching versus hovering just above is not '
                                       'legible from the pixels.',
                    'mentions': ['agent_pose', 'agent_velocity'],
                    'prompt': ('env_prose',
                               'Every episode starts from the same lying pose and robot pose with '
                               'small random jitter on every position and velocity coordinate, the '
                               "human's root pose included; the 52 points are fixed to the arm and "
                               'identical in every episode.'),
                    'scene': "A single bed seen obliquely from the robot's side and from beyond its "
                             'foot, looking down toward the head: a person lying on their back with '
                             'their head on a pillow, and a Franka Panda arm standing on the floor '
                             "beside the bed by the person's right arm, holding a small red square pad "
                             'that points downward.',
                    'success_criterion_prose': 'wiped_fraction: the number of the 52 wipe points '
                                               'marked wiped by the final step of the episode, divided '
                                               'by 52. A point is marked wiped on any step where the '
                                               'wiper centre is within 0.1 m of it while the wiper '
                                               '(the pad or its base cube) is in contact with the '
                                               "human's right upper arm, forearm or hand -- any "
                                               "non-zero summed tool-human contact force, upstream's "
                                               'own marking rule -- and it stays wiped for the rest of '
                                               'the episode. Success is a provisional threshold of 0.5 '
                                               'on that fraction, i.e. at least 26 of the 52 points '
                                               'wiped by the end. This is a check this repo wrote: '
                                               'upstream ships no success criterion for the task and '
                                               'reports only the return of its own dense reward.',
                    'telemetry': ['n_wiped',
                                  'dist_tool_nearest_unwiped',
                                  'contact_force_magnitude',
                                  'ee_speed',
                                  'larm_z'],
                    'upstream_divergence': 'Physics: the same vendored MJCF at its own 0.002 s '
                                           'timestep with 4 substeps per control step (125 Hz), solved '
                                           "by C MuJoCo 3.3.0 with the XML's defaults (Newton, 100 "
                                           'iterations) where upstream runs MJX with iterations=1, '
                                           'ls_iterations=4 and EULERDAMP disabled; measured against '
                                           "upstream's own package from the keyframe under zero "
                                           'actions, the two solves part by a max |dq| of 0.0026 after '
                                           'one control step, 0.0234 after 10 and 0.0682 after 100, '
                                           'and MJX costs 513 ms per control step on CPU against 0.5 '
                                           'ms here. The human is a zero-action passive body in place '
                                           "of upstream's trained partner on right_shoulder1, "
                                           'right_shoulder2 and right_elbow, and the episode always '
                                           'runs its 1000 steps where upstream sets done the step the '
                                           '52nd point is wiped (and, at its brax-level API, carries '
                                           'the unwiped vector across an auto-reset because state.info '
                                           'is not restored -- here the flags live in the state and '
                                           'reset to ones). The observation is the whole state (qpos, '
                                           'qvel, the 52 flags, derived tool, force and target fields) '
                                           "rather than upstream's 25-D robot and 36-D human vectors, "
                                           'whose qpos slices are indexed by joint id and so carry the '
                                           "human's arm angles as 'robo_joint_angles', no Panda joint "
                                           'at all, no velocities and not the wiped vector. The 52 '
                                           'points are the same 26-point helix per capsule but placed '
                                           "in the capsule GEOM frame, on the arm's surface, where "
                                           'upstream maps them through the arm BODY frame so its '
                                           'points lie 0-0.12 m off the arm surface with 8 of 52 '
                                           'inside the mattress and the upper-arm set reaching 13 cm '
                                           'above the shoulder; the marking rule is the same 0.1 m '
                                           'radius but gated on a non-zero 3-vector world-frame force '
                                           'summed over the live contacts between the two wiper geoms '
                                           "and the human's right upper arm, forearm and hand "
                                           '(mj_contactForce) rather than on any component of a '
                                           '6-vector wrench read at fixed MJX contact slots '
                                           "58/59/62/63. The reference reward keeps upstream's "
                                           'exp(-d^2/0.1) distance term at weight 1 on the nearest '
                                           'unwiped point, omits the zero-weight control cost and '
                                           'drops the +3 per newly-wiped point bonus because it needs '
                                           'the previous flags; it therefore pays 0 once every point '
                                           'is wiped. Upstream sets done the step the last point is '
                                           'wiped, but its reward does see that regime through its '
                                           'auto-reset defect (the flags persist, so r_dist is 0 and '
                                           'done is set on every further step until the wrapper '
                                           'resets them). Two more consequences of dropping the bonus '
                                           'are stated in the adapter: the nearest-unwiped distance '
                                           'is taken on the arriving flags, so the reference DROPS on '
                                           'a wipe step (32 of 34 measured), and nothing in it pays '
                                           'for wiping -- hovering near an unwiped point scores 946 '
                                           'over an episode against 305 for wiping all 52.'},
     'feeding': {'env_prose': 'A Franka Panda arm on a low table stands at the right of a person '
                              'seated in a wheelchair, facing them. Its hand carries a spoon: a '
                              'light-grey oval bowl about 7 cm by 6 cm and 4 mm thick on a short '
                              'handle, its food side a translucent green face; the reference point is '
                              "the bowl's centre, 20 cm from the hand. Each action is seven numbers in "
                              '[-1, 1], one per Panda joint: a normalised position target for a stiff '
                              "servo (about pi times the value, clipped to the joint's range), not a "
                              'torque. The person is passive: every humanoid actuator receives zero, '
                              'so the body rests on its joint stiffness, the chair and a '
                              'pelvis-to-chair anchor, settles back into the chair over the first two '
                              'seconds, and can be pushed; the head rests on the headrest, held only '
                              'by light neck damping and a weak spring, so the spoon can turn it. The '
                              'mouth is a fixed point on the face. The spoon collides only with the '
                              "head and left arm; the Panda's hand and fingers pass through the head. "
                              'Every episode starts from one seated keyframe pose with small random '
                              'jitter on every position and velocity coordinate; the spoon hangs '
                              'bowl-down from the hand, on its side, about half a metre to the '
                              "person's right at roughly mouth height. A control step is 8 ms; an "
                              'episode is 1000 steps (8 s) and nothing ends it early. The observation '
                              'is the whole simulator state then derived fields: root position and '
                              'quaternion, every hinge angle (human then Panda), all velocities '
                              "likewise, then the spoon's position (bowl centre), orientation "
                              'quaternion and velocity, the world-frame force the bowl applies to the '
                              "head, the mouth's position, the spoon-to-mouth distance, the cosine "
                              "between the spoon's up axis and world up (1 level, 0 sideways, -1 "
                              'inverted), and the force magnitude through the green face alone. No '
                              'task slots.',
                 'exploits': [('hover_without_touching',
                               'The force clause is an upper bound with no floor and there is no '
                               'contact clause, so a spoon parked with its centre a few centimetres in '
                               'front of the mouth and never touching the face satisfies both clauses '
                               'on every step it stays there. A policy that reaches that hover early '
                               'and freezes scores close to 1.0 without feeding anyone.'),
                              ('any_orientation_counts',
                               "The spoon's level-ness (spoon_up_dot_world_up) is in the observation "
                               "and in upstream's reward but not in the check, and the spoon starts on "
                               'its side; a bowl that arrives edge-on or inverted, which would have '
                               'spilled, scores exactly as a level one does.'),
                              ('press_with_the_uncounted_geoms',
                               'Only the thin green face (spoon_right_side) feeds tip_force_mag: the '
                               "bowl body's head contacts enter the tool-force field but not that one, "
                               'the handle and the 6 cm mounting cube do collide with the head yet '
                               "enter neither, and the Panda's hand and finger geoms cannot collide "
                               "with the head at all. Pressing the bowl's underside, the handle or the "
                               'mounting cube into the face pushes the head with a force the check '
                               'never sees, so long as the penetration does not reach the green face, '
                               'and a gripper driven into the head passes through without contact -- '
                               'the counted force stays at 0 N while the spoon centre sits within '
                               'range.'),
                              ('carry_the_mouth_along',
                               'The mouth is fixed to a passive head whose centre of mass sits at the '
                               'neck joints (no gravity torque) and which is held only by damping 1, '
                               'armature 0.1 and a 0.625 N m/rad actuator spring, on an uncommanded '
                               'trunk; a spoon that turns the head toward the robot or presses head '
                               'and trunk further into the chair carries the mouth with it, and the '
                               'distance is measured to wherever the mouth has gone. Shoving and '
                               'holding scores identically to a careful approach as long as the '
                               'counted face force stays under 5 N or the pushing is done with an '
                               'uncounted geom.')],
                 'factory_doc': 'Assistax `feeding` (upstream Feeding): a Panda holding a spoon brings '
                                'it, level and gently, to the mouth of a passive human seated in a '
                                'wheelchair; plain-MuJoCo port.',
                 'goal_sentence': 'none -- the mouth site is fixed on the head (8 cm forward of and 4 '
                                  "cm below the head centre, just inside the head's collision sphere) "
                                  'and moves only with the head; spoon_at_mouth_fraction compares the '
                                  'spoon centre against it',
                 'l_task': "Feed the seated person: bring the spoon fixed to the Panda's hand up to "
                           'their mouth with the bowl level and its green food side facing up so '
                           'nothing spills, slowing as it arrives, and hold it there for as much of '
                           'the episode as you can, touching the face only lightly if at all and never '
                           'pushing the head or the body about. The arm is commanded with seven '
                           'joint-position targets, one per Panda joint, each a normalised target '
                           "angle for that joint's stiff servo rather than a torque; the person is "
                           'passive and will not lean in to meet the spoon.',
                 'legibility_note': "A seated adult and a Panda at 448x320 from the scene's `default` "
                                    "camera at (1.3, 2.1, 1.8), which re-aims at the person's centre "
                                    'of mass from above and in front of their right side, so the face, '
                                    'the chair and the whole robot share the frame and the framing '
                                    'drifts slightly as the person settles. Whether the spoon is up at '
                                    'the face or elsewhere in the air, whether the gripper has been '
                                    'driven into the head (it cannot collide with it and visibly '
                                    'passes inside), whether the head has been turned or the trunk '
                                    'shoved, and in a clip whether the spoon is still or moving all '
                                    "read from the pixels. The spoon's most visible part is the blue 6 "
                                    'cm mounting cube at the gripper; the mouth is a green dot of '
                                    'radius 0.01 m and the bowl a light-grey ellipse about 7 cm long '
                                    "with a translucent green face, both small in frame, so 'at the "
                                    "mouth' against 'a few centimetres off', hovering against touching "
                                    '(the bowl is 4 mm thick) and level to within a few degrees are '
                                    'not legible, though inverted or edge-on against level is. Two red '
                                    'spheres on the right upper arm and forearm are leftover itch '
                                    'markers from the scratchitch humanoid, not targets. The legend '
                                    'prints only the task name: neither the spoon-to-mouth distance '
                                    'nor any contact force is printed, so those numbers have no '
                                    "route into a VLM's verdict beyond the pixels described above.",
                 'mentions': ['agent_pose', 'agent_velocity'],
                 'prompt': ('env_prose',
                            'Every episode starts from one seated keyframe pose with small random '
                            'jitter on every position and velocity coordinate; the spoon hangs '
                            "bowl-down from the hand, on its side, about half a metre to the person's "
                            'right at roughly mouth height.'),
                 'scene': 'A person sits reclined in a wheelchair, head against the headrest and '
                          'forearms coming to rest on the armrests; a Franka Panda arm stands on a '
                          'small pedestal table at their right, a spoon -- a blue mounting cube at the '
                          'gripper, a thin grey handle and a light-grey bowl with a translucent green '
                          'face -- hanging bowl-down from its hand about half a metre from the face, '
                          "and a small green dot marks the mouth. The scene's camera sits above and in "
                          "front of the person's right side, aimed at their body's centre of mass, so "
                          'the face, the chair and the robot share the frame.',
                 'success_criterion_prose': 'spoon_at_mouth_fraction is a check this repository wrote, '
                                            'not a benchmark criterion: upstream Assistax ships no '
                                            'success test for feeding and reports only the return of '
                                            "its own dense reward. It is the fraction of the episode's "
                                            '1000 arriving states on which both of the following hold '
                                            "at once: the spoon centre (the bowl's centre) is within "
                                            '0.04 m of the mouth site, and the contact force through '
                                            "the spoon's green contact face onto the head is below 5 "
                                            'N. Zero force passes -- no contact is required -- and '
                                            "only that one face is counted: force from the bowl's "
                                            'underside, the handle, the mounting cube or the gripper '
                                            'enters the check nowhere. success() is this fraction '
                                            'reaching 0.3, a provisional threshold. The check gates on '
                                            "the distance upstream's reward is shaped on and reads a "
                                            "force upstream's reward also shapes, so it is not "
                                            "independent of that reward the way a benchmark's success "
                                            "flag would be; it does not test the spoon's orientation "
                                            '(level or not), its speed, or whether the head was moved. '
                                            'A uniform-random policy scores exactly 0 over 30 '
                                            'episodes.',
                 'telemetry': ['dist_tool_mouth',
                               'tip_force_mag',
                               'contact_force_magnitude',
                               'spoon_up_dot_world_up',
                               'mouth_z'],
                 'upstream_divergence': 'Same MJCF, different solve: this port steps the vendored '
                                        "feeding scene through C MuJoCo 3.3.0 with the XML's defaults "
                                        '(Newton solver, 100 iterations, timestep 0.002 s, 4 substeps '
                                        'per 125 Hz control step), where upstream runs it under MJX '
                                        'with iterations 1, ls_iterations 4 and EULERDAMP disabled; '
                                        "measured against upstream's own package from the keyframe "
                                        'under zero actions, the two solves part by a max |dq| of '
                                        '0.0033 after one control step, 0.0201 after 10 and 0.0605 '
                                        'after 100, and MJX costs 828 ms per control step on CPU '
                                        "against 0.5 ms here. Upstream's human is a second trained "
                                        "agent on two actuators base_env.py labels 'head' but which "
                                        'are abdomen_y and abdomen_z (torso pitch and yaw; the neck '
                                        'actuators are commanded by nobody upstream either), so its '
                                        'mouth moves by leaning; here every humanoid actuator receives '
                                        'zero and the mouth moves only with the settling transient or '
                                        'when the robot pushes the head or trunk. Neither side '
                                        'terminates early: upstream hard-codes done = 0 and truncates '
                                        'at 1000 steps, which is the horizon here. The observation is '
                                        'the whole state (qpos, qvel, derived spoon, mouth, force and '
                                        "orientation fields) rather than upstream's 58-value vector, "
                                        'whose qpos slices are indexed by joint id so that its '
                                        "'robo_joint_angles' are six human arm joints plus Panda "
                                        'joint1, Panda joints 2-7 appear nowhere and its declared '
                                        'per-agent ranges misroute one value and drop three -- defects '
                                        'this port does not reproduce. The reference reward '
                                        "transcribes upstream's terms and weights (2 exp(-3 d); the "
                                        'pour/aim orientation blend with proximity scale 150; the '
                                        'speed term with targets 0.20 m/s far and 0 at the mouth and '
                                        'sigma 0.1, as the code hard-codes them rather than the unread '
                                        'target_spoon_speed argument; the force term peaking at 1 N; '
                                        'ctrl 1e-6) with three substitutions: the control cost squares '
                                        'the 7-D robot action (clipped to [-1, 1]) where upstream '
                                        'squares the concatenated robot+human action, identical while '
                                        "the human's part is zero; the spoon speed is the "
                                        "simulator's instantaneous site velocity rather than a finite "
                                        'difference over dt 0.008 s; and the force is the world-frame '
                                        'sum over the live spoon_right_side-head contacts '
                                        '(mj_contactForce) rather than the norm of fixed MJX '
                                        'contact-array row 187, whose geom pair is a property of the '
                                        'MJX build and was not resolved to a named pair here '
                                        "(upstream's 6-vector norm equals its 3-force norm, every geom "
                                        'being condim 3, so the row selection is the real difference). '
                                        "Both sides share the keyframe's defects -- Panda joint4 "
                                        'starts outside its own range and is driven to its limit over '
                                        'about a quarter of a second, swinging the spoon, and the '
                                        'pelvis anchor is violated by 2 cm so the person snaps and '
                                        "slumps -- and both integrate the Panda's actuator velocity "
                                        'feedback explicitly under Euler, so five of its joints '
                                        'chatter at the substep rate: measured here in the C engine, '
                                        'expected but not run under MJX.'},
     'scratchitch': {'env_prose': 'A Panda arm stands 0.75 m to the right of a person seated in a '
                                  'wheelchair, facing them. Its hand carries a scratcher: a cube with '
                                  'a thin 0.1 m stick of radius 0.01 m whose far end is the tip. Each '
                                  'action is seven numbers in [-1, 1], one per Panda joint: a '
                                  'normalised position target for a stiff servo (about pi times the '
                                  'value), not a torque. The person is passive (every humanoid '
                                  'actuator receives zero): the body rests under gravity on its joint '
                                  "stiffness, its actuators' springs, the chair and a pelvis-to-chair "
                                  'anchor, settles about three centimetres over the first two seconds, and '
                                  'can be pushed about. The right upper arm and forearm are capsules '
                                  'of radius 0.04 m and 0.031 m; the stick collides only with that '
                                  'arm, the hand and the chair, passing through everything else. Each '
                                  'episode draws a fresh itch: a coin toss picks the upper arm or the '
                                  "forearm, then a point on that segment's skin at a random height "
                                  'along it and a random angle around it; the itch rides with the arm '
                                  'for the rest of the episode. Human and robot start from one seated '
                                  'keyframe pose with a few millimetres of jitter on every position '
                                  'coordinate and a small random velocity; the tool hangs beside the '
                                  "robot's upright column, its tip a little under half a metre from "
                                  'the arm. The observation is the whole simulator state then derived '
                                  "fields: the human's root position and quaternion, every hinge angle "
                                  '(human then Panda), every velocity likewise, the task slots (which '
                                  'segment carries the itch and where, fixed for the episode), tip '
                                  'position, tool quaternion, tip velocity, the world-frame contact '
                                  'force in newtons the stick alone applies to the right arm and hand, '
                                  "the itch's world position, the tip-to-itch distance and both arm "
                                  'segment positions. Control is 125 Hz; an episode is 1000 steps (8 '
                                  's); nothing ends it early.',
                     'exploits': [('drag_the_arm',
                                   "The itch is a point fixed in the arm segment's frame and the human "
                                   'is passive (the right arm has almost no joint stiffness), so a '
                                   'stick that hooks or shoves the forearm carries the itch along with '
                                   'it: the tip stays inside the 0.05 m gate, the shove supplies the '
                                   '0.5 N, and the drag supplies the 0.02 m/s. Pushing the arm around '
                                   'the chair scores as scratching on every step it lasts.'),
                                  ('tremble_in_contact',
                                   "The speed clause reads the tip's total speed, not its motion along "
                                   'the skin, so a tip that is in range and stays pressed on the arm '
                                   'clears it with any 0.02 m/s oscillation -- including a '
                                   'push-and-ease straight into the arm that never breaks contact. A '
                                   'vibrating touch scores a full-episode scratch without ever '
                                   'rubbing.'),
                                  ('stick_side_contact',
                                   'Force is summed over the whole stick capsule while distance is '
                                   'measured from the tip site, so laying the stick along the arm and '
                                   'pressing with its side while the tip point sits within 0.05 m of '
                                   'the itch satisfies both clauses with nothing touching the itch '
                                   'itself; the same tolerance accepts contact anywhere within 0.05 m '
                                   'of the itch, including on the hand when the itch is near the wrist '
                                   'end of the forearm.'),
                                  ('unbounded_pressure',
                                   'The force clause has a floor and no ceiling, so once the tip is '
                                   'near the itch and moving, pinning the arm hard against the armrest '
                                   'and grinding scores identically to a gentle scratch; any force '
                                   "above 0.5 N reads the same. Upstream's reward peaks at 3 N and "
                                   'falls off above it; the check does not.')],
                     'factory_doc': 'Assistax `scratchitch` (upstream ScratchItch): a Panda holding a '
                                    'scratcher rubs a per-episode itch on the right arm of a passive '
                                    'human seated in a wheelchair; plain-MuJoCo port.',
                     'goal_sentence': "the itch is drawn per episode on the right arm's surface -- a "
                                      'coin toss picks the upper arm or the forearm, then a height '
                                      'along that capsule and an angle around it -- and '
                                      'scratching_fraction compares the scratcher tip against its '
                                      'world position, which rides with the arm',
                     'l_task': "Scratch the itch on the seated person's right arm: bring the tip of "
                               "the scratcher fixed to the Panda's hand onto the itch, whose location "
                               'is given, press it gently against the skin and rub, and keep '
                               'scratching there for as much of the episode as you can without shoving '
                               'the arm about. The arm is commanded by seven joint-position targets, '
                               "one per Panda joint, each a normalised target angle for that joint's "
                               'servo rather than a torque.',
                     'legibility_note': "A seated adult and a Panda at 448x320 from the scene's camera "
                                        "above and in front of the person's right side: which arm "
                                        'segment the blue stick is near, whether the arm is still '
                                        'resting on the armrest or has been shoved off it, and (in a '
                                        'clip) whether the stick is rubbing all read from the pixels. '
                                        "The itch is a red marker of the tip's own radius, a few "
                                        "pixels wide, so 'on the itch' against 'a few centimetres off' "
                                        'cannot be told from pixels, nor can light contact from a near '
                                        'miss, and neither contact force nor tip speed is visible in a '
                                        "still. The frame's legend prints only the task name: the "
                                        'tip-to-itch distance and the contact force the check gates on '
                                        "are not printed, so neither has a route into a VLM's verdict "
                                        'beyond the pixels. The camera re-aims at the '
                                        "human's centre of mass, so the framing drifts slightly as the "
                                        'person settles.',
                     'mentions': ['agent_pose', 'agent_velocity', 'goal'],
                     'prompt': ('env_prose',
                                'Each episode draws a fresh itch: a coin toss picks the upper arm or '
                                "the forearm, then a point on that segment's skin at a random height "
                                'along it and a random angle around it; the itch rides with the arm '
                                'for the rest of the episode. Human and robot start from one seated '
                                'keyframe pose with a few millimetres of jitter on every position '
                                'coordinate and a small random velocity; the tool hangs beside the '
                                "robot's upright column, its tip a little under half a metre from the "
                                'arm.'),
                     'scene': 'A person sits in a wheelchair, trunk leaning forward, forearms on the '
                              'armrests and head on the headrest; a Panda arm stands on the floor at '
                              'their right holding a blue scratcher stick, and a small red dot on the '
                              "right upper arm or forearm marks the itch. The scene's camera sits "
                              "above and in front of the person's right side, aimed at their body's "
                              'centre of mass, so the chair and the robot beside it share the frame.',
                     'success_criterion_prose': 'scratching_fraction is a check this repository wrote, '
                                                'not a benchmark criterion: upstream Assistax ships no '
                                                'success test for this task and reports only the '
                                                'return of its own reward. It is the fraction of the '
                                                "episode's 1000 arriving states on which all three of "
                                                'the following hold at once: the scratcher tip is '
                                                'within 0.05 m of the itch, the stick presses on the '
                                                "person's right arm or hand with a net contact force "
                                                'of at least 0.5 N, and the tip is moving at 0.02 m/s '
                                                'or faster. success() is this fraction reaching 0.3, a '
                                                'provisional threshold; the check gates on the same '
                                                "quantities upstream's reward is shaped on (distance, "
                                                'contact force, tip speed), so it is not independent '
                                                "of that reward the way a benchmark's success flag "
                                                'would be. A uniform-random policy scores exactly 0 '
                                                'over 30 episodes.',
                     'telemetry': ['dist_tool_target',
                                   'contact_force_magnitude',
                                   'ee_speed',
                                   'target_z',
                                   'larm_z'],
                     'upstream_divergence': 'Same MJCF, different solve: this port steps the vendored '
                                            "wheelchair scene through C MuJoCo 3.3.0 with the XML's "
                                            'defaults (Newton solver, 100 iterations, timestep 0.002 '
                                            's, 4 substeps per 125 Hz control step), where upstream '
                                            'runs it under MJX with iterations 1, ls_iterations 4 and '
                                            "EULERDAMP disabled; measured against upstream's own "
                                            'package, MJX takes 500-830 ms per control step on CPU '
                                            'against 0.5 ms here, and from one keyframe under zero '
                                            'actions the two solves part by a max |dq| of 0.02-0.13 '
                                            "within 10 control steps. Upstream's human is a second "
                                            'trained agent driving right_shoulder1, right_shoulder2 '
                                            'and right_elbow (up to 10 N.m each) from a zoo of JAX '
                                            'partner policies; here every humanoid actuator receives '
                                            'zero, so the person is passive and the itch moves only '
                                            'when gravity, the reset transient or the robot moves the '
                                            "arm. Neither side terminates early: upstream's step "
                                            'hard-codes done = 0 and its EpisodeWrapper truncates at '
                                            '1000, which is the horizon here. The observation is the '
                                            'whole simulator state (positions and velocities) plus '
                                            "derived fields rather than upstream's 69-dim global "
                                            'vector sliced into a 29-dim robot view and a 40-dim human '
                                            'view, which carries no velocities and has an indexing '
                                            'defect this port does not reproduce: qpos is sliced by '
                                            "joint id, so the robot's 'joint angles' are the human's "
                                            "six arm joints and the Panda's joints appear nowhere. The "
                                            "reference reward transcribes upstream's terms at the "
                                            'constructor defaults (dist 1.0 with scale 0.1, scratching '
                                            '4.0 with targets 0.1 m/s and 3.0 N, ctrl 1e-6, gate on '
                                            'distance below 0.1 m as the code has it and not on r_dist '
                                            'as the README says) with two substitutions: the tip speed '
                                            "is the simulator's instantaneous site velocity rather "
                                            'than a finite difference over dt 0.008 s, and the contact '
                                            'force is the world-frame sum over the live stick contacts '
                                            'with the right upper arm, forearm and hand rather than '
                                            'the 6-norm of two fixed MJX contact slots (273/274) whose '
                                            'geom pair is a property of the MJX collision enumeration '
                                            "and was not resolved here. Upstream's AutoResetWrapper "
                                            'never re-draws the itch within a rollout; here every '
                                            'reset draws a fresh arm and position.'},
     'teethbrushing': {'env_prose': 'A Franka Panda arm on a small table 0.9 m to the right of a '
                                    'seated wheelchair user faces them, holding a toothbrush hanging '
                                    'straight down. The person is a capsule-limbed humanoid of about '
                                    '42 kg; the head is a 0.09 m radius sphere on the headrest, the '
                                    'mouth a marked point low on its front and no teeth. The brush '
                                    'head is a 12 by 12 by 36 mm box on a thin handle; one long side '
                                    "is the high-friction bristle face, its centre the tool's "
                                    'reference point. The person is passive (no humanoid actuator is '
                                    'commanded): the body rests on the chair under gravity, pelvis '
                                    'anchored to the seat, and the head sits on a soft neck the brush '
                                    'can push. The brush collides only with the head and the left arm, '
                                    'passing through all else; only head contact is measured. Each '
                                    'action is seven numbers in [-1, 1], one per Panda joint: a '
                                    'normalised position target for a stiff servo (about pi times the '
                                    'value, clipped to the joint range), not a torque. The observation '
                                    "is the whole simulator state then derived fields: the human's "
                                    'root position and quaternion, every hinge angle (human then '
                                    "Panda), every velocity, the brush reference point's position, the "
                                    "brush body's quaternion, the reference point's velocity, the "
                                    'world-frame contact force the brush applies to the head, the '
                                    "mouth's position, the brush-to-mouth distance, the bristle "
                                    "direction's cosine with the brush-to-mouth direction, the force "
                                    "the bristle face alone applies to the head, and the brush's speed "
                                    "across the mouth's surface. Every episode starts from one seated "
                                    'keyframe pose with the brush hanging about 0.8 m from the mouth, '
                                    'with small random jitter on every position and velocity '
                                    'coordinate of both the human and the arm; the mouth is a fixed '
                                    'point on the head and is never re-placed. Control is 125 Hz; an '
                                    'episode is 1000 steps (8 s); nothing ends it early.',
                       'exploits': [('rub_the_chin_or_cheek',
                                     'The distance clause is a 0.03 m ball around a single mouth point '
                                     'on a featureless sphere of radius 0.09 m, and the model has no '
                                     'teeth, lips or jaw, so pressing the bristle face onto any patch '
                                     'of the head inside that ball -- above the mouth marker, beside '
                                     'it or below it -- and sliding it there satisfies every clause '
                                     'exactly as brushing the teeth would; nothing in the check '
                                     "distinguishes the mouth's surface from the skin around it, and "
                                     'the bristle-direction cosine the observation carries is not part '
                                     'of the check.'),
                                    ('dither_in_place',
                                     'The speed clause is a floor of 0.01 m/s on the tangential '
                                     "component of the brush reference point's instantaneous velocity, "
                                     'read every 8 ms with no requirement on stroke length or '
                                     'reversal, so a wrist tremor of under a tenth of a millimetre per '
                                     'control step while the brush stays pressed on one spot clears it '
                                     'on every step; a vibrating touch scores a full-episode brush '
                                     'without any stroke.'),
                                    ('drag_the_head_with_the_brush',
                                     "The speed clause reads the brush's world-frame velocity, not its "
                                     'motion relative to the face, and the passive head rides on a '
                                     'neck whose only restoring torque is a spring well under one '
                                     'newton-metre per radian plus light damping, while the bristle '
                                     'face carries a friction coefficient of 2: a brush pressed into '
                                     "the face from the person's right and swept along carries the "
                                     'head with it on its neck, so the distance and force clauses '
                                     'hold, the world speed reads as brushing, and bristles and skin '
                                     'never slide against each other. Pushing the head around scores '
                                     'as brushing on every step it lasts.'),
                                    ('lean_on_the_face',
                                     'The force clause has a floor and no ceiling and the human cannot '
                                     'resist, so once the brush is at the mouth and moving, grinding '
                                     'it into the face as hard as the arm can push and shoving the '
                                     'head sideways on its neck scores identically to a gentle brush, '
                                     'and a head pinned by the brush is easier to keep the distance '
                                     "clause on. Upstream's own force shaping peaks at 1 N and is "
                                     'metrics-only; the check has nothing above 0.1 N.')],
                       'factory_doc': 'Assistax `teethbrushing` (upstream TeethBrushing): a Panda '
                                      'holding a toothbrush brushes the mouth of a passive human '
                                      'seated in a wheelchair; plain-MuJoCo port.',
                       'goal_sentence': 'none -- the mouth is a site fixed on the head body (0.08 m '
                                        'forward of and 0.04 m below the head centre, at the front of '
                                        'the head sphere) and is never re-placed; brushing_fraction '
                                        "compares the brush head's reference point against its world "
                                        'position, which moves only with the head -- by the reset '
                                        'jitter and by whatever the brush pushes',
                       'l_task': "Brush the seated person's teeth: bring the toothbrush across to "
                                 'their mouth, press its bristle face gently against the front of the '
                                 'mouth, and keep the brush moving back and forth over it for as much '
                                 'of the episode as you can, without shoving the head about. The arm '
                                 'is commanded by seven joint-position targets, one per Panda joint, '
                                 'each a normalised target angle that a stiff position servo tracks '
                                 'rather than a torque; the person is passive and will neither help '
                                 'nor turn toward the brush.',
                       'legibility_note': "A seated adult and a Panda at 448x320 from the scene's "
                                          '`default` camera at (1.3, 2.1, 1.8), which re-aims at the '
                                          "person's centre of mass from above and from their "
                                          'front-right, about 2.7 m from the head: whether the arm has '
                                          'crossed to the face, which side of the head the brush is '
                                          'on, whether the head has been pushed sideways or turned on '
                                          'its neck, and (in a clip) whether the brush is moving all '
                                          'read from the pixels. The brush head is a 12 mm box -- a '
                                          'pixel or two wide at that distance -- and the head is a '
                                          'featureless sphere with a green mouth marker of radius 0.01 '
                                          'm a few pixels across, drawn in the same green as the '
                                          "bristle plate, so 'at the mouth' against 'on the skin "
                                          "beside it' cannot be told from pixels, nor can light "
                                          'contact from a near miss, and neither the bristle-face '
                                          'force nor the tangential speed is visible in a still. The '
                                          'person also settles about three centimetres into the chair '
                                          'over the first two seconds on their own, so early head motion is '
                                          'not evidence of a push. The legend prints only the task '
                                          'name: neither the brush-to-mouth distance nor any force is '
                                          "printed, so those numbers have no route into a VLM's "
                                          'verdict beyond the pixels.',
                       'mentions': ['agent_pose', 'agent_velocity'],
                       'prompt': ('env_prose',
                                  'Every episode starts from one seated keyframe pose with the brush '
                                  'hanging about 0.8 m from the mouth, with small random jitter on '
                                  'every position and velocity coordinate of both the human and the '
                                  'arm; the mouth is a fixed point on the head and is never '
                                  're-placed.'),
                       'scene': 'A person sits in a wheelchair with their head on the headrest, seen '
                                'from above and from their front-right; a Franka Panda arm stands on a '
                                'small pedestal table 0.9 m to their right holding a toothbrush on a '
                                'blue mounting block that hangs straight down, and a small green dot '
                                'low on the front of the plain spherical head marks the mouth. The '
                                "camera re-aims at the person's centre of mass, so the chair, the "
                                'table and the robot share the frame.',
                       'success_criterion_prose': 'brushing_fraction is a check this repository wrote, '
                                                  'not a benchmark criterion: upstream Assistax ships '
                                                  'no success test for this task and reports only the '
                                                  'return of its own reward. It is the fraction of the '
                                                  "episode's 1000 arriving states on which all three "
                                                  "of the following hold at once: the brush head's "
                                                  'reference point (the centre of the bristle face) is '
                                                  'within 0.03 m of the mouth site, the bristle-face '
                                                  'plate presses on the head with a contact force of '
                                                  'at least 0.1 N (the magnitude of the summed '
                                                  'plate-to-head contact force in the world frame, '
                                                  'friction included), and the brush head is moving '
                                                  "across the mouth's surface at 0.01 m/s or faster "
                                                  "(the component of the reference point's world-frame "
                                                  'velocity perpendicular to the line from the brush '
                                                  'head to the mouth). success() is this fraction '
                                                  'reaching 0.2, a provisional threshold; all three '
                                                  "gates are on the same quantities upstream's reward "
                                                  'is shaped on (distance, bristle-face force, '
                                                  'tangential speed), so the check is not independent '
                                                  "of that reward the way a benchmark's success flag "
                                                  'would be. A uniform-random policy scores exactly 0 '
                                                  'over 30 episodes.',
                       'telemetry': ['dist_tool_mouth',
                                     'tip_force_mag',
                                     'brush_tangential_speed',
                                     'bristle_dot_to_mouth',
                                     'mouth_z'],
                       'upstream_divergence': 'Same MJCF, different solve: this port steps the '
                                              'vendored teethbrushing scene through C MuJoCo 3.3.0 '
                                              "with the XML's defaults (Newton solver, 100 iterations, "
                                              'timestep 0.002 s, 4 substeps per 125 Hz control step), '
                                              'where upstream runs it under MJX with iterations 1, '
                                              'ls_iterations 4 and EULERDAMP disabled; measured '
                                              "against upstream's own package from the keyframe under "
                                              'zero actions, the two solves part by a max |dq| of '
                                              '0.0033 after one control step, 0.0197 after 10 and '
                                              '0.0606 after 100, and MJX costs 729.7 ms per control '
                                              "step on CPU against 0.5 ms here. Upstream's human is a "
                                              'second trained agent whose two actuators are abdomen_y '
                                              'and abdomen_z -- torso pitch and yaw, despite '
                                              "base_env.py's `# head` comment; the neck actuators are "
                                              'commanded by nobody -- so its human moves the mouth by '
                                              'bending the trunk, while here every humanoid actuator '
                                              "receives zero, which is what upstream's 17 uncommanded "
                                              'ones already get. Neither side terminates early '
                                              '(upstream hard-codes done = 0 and its EpisodeWrapper '
                                              'truncates at 1000, the horizon here). The observation '
                                              "is the whole simulator state read at the model's own "
                                              "qpos addresses plus derived fields, not upstream's "
                                              '58-wide concat sliced into a 22-D robot view and a 33-D '
                                              'human view, which carry no velocities and whose qpos '
                                              'slices are indexed by joint id -- its '
                                              "'robo_joint_angles' are six human arm angles plus Panda "
                                              'joint1, its robot agent sees no Panda joint at all and '
                                              'Panda joints 2-7 appear nowhere -- and contact forces '
                                              'are summed over the live toothbrush-head contacts by '
                                              'geom pair rather than read at MJX contact-array slots '
                                              '19 and 20 (marked `TODO: update these` upstream; a '
                                              "reconstruction of MJX 3.4.0's collision enumeration for "
                                              'this scene puts the head-vs-brush-head and '
                                              'head-vs-bristle-plate pairs at those rows, which was '
                                              'not confirmed against a stepped MJX Data and is '
                                              'recorded as unverified). The '
                                              'reference reward transcribes 2.0*exp(-3 d) + '
                                              '1.0*(r_brush + r_align*r_dist) - 1e-6*|a|^2 as the code '
                                              'has it -- the hard-coded 0.05 m/s brushing speed and '
                                              '0.1 N force gate, no distance gate and no force factor '
                                              "in r_brush whatever the README says, the constructor's "
                                              'target_toothbrush_speed 0.1 never read, r_force '
                                              'metrics-only -- with three substitutions: the brush '
                                              "speed is the simulator's instantaneous site velocity "
                                              'rather than a finite difference over dt 0.008 s, the '
                                              'plate force is the world-frame sum over the live '
                                              'plate-head contacts rather than the norm of slot 20, '
                                              'and the control cost squares the 7-D action rather than '
                                              'the 26-D global one. Reproduced on purpose: the `init` '
                                              'keyframe puts Panda joint4 at 0, outside its range, so '
                                              'on both sides the arm snaps to that limit at t = 0.'}}
)


#: The keys every `_AUTHORED` entry must carry.
_REQUIRED_AUTHORED = ("env_prose", "l_task", "scene", "success_criterion_prose", "exploits",
                      "telemetry", "legibility_note", "upstream_divergence", "goal_sentence",
                      "prompt", "mentions")


# --------------------------------------------------------------------------
# the live model, without the spec
# --------------------------------------------------------------------------

def construct_without_spec(task: str) -> ax.Assistax:
    """`Assistax.__init__` up to the spec lookup, then `EnvAdapter.__init__`: the generator
    needs `reset()` on the live model before the spec it is about to write exists."""
    import importlib
    import os
    mujoco = importlib.import_module("mujoco")
    env = ax.Assistax.__new__(ax.Assistax)
    env._row = ax._TASKS[task]
    env._task = task
    env.name = ax._env_id(task)
    env._mj = mujoco
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = ax._default_mujoco_gl() or "egl"
    env._construct_model(mujoco)
    env.obs_dim = env.n_q + env.n_v + env.n_slots + len(env._row["tail"])
    env.horizon = ax.HORIZON
    env.success_threshold = float(env._row["threshold"])
    env._state_fields = tuple((f"s{i}", "") for i in range(env.obs_dim))
    env._action_fields = tuple((f"a{i}", "") for i in range(ax.ACTION_DIM))
    EnvAdapter.__init__(env)
    return env


def _force_field_names(row: Dict[str, Any]) -> set:
    """The tail columns that are contact forces -- a typical range, not a bound, and 0 N at
    reset: the family's `tool_f*` and `tip_force_mag`."""
    return {n for n, *_r in row["tail"] if n.startswith("tool_f")} | {"tip_force_mag"}


def _joint_owner(name: str) -> str:
    """`robot` for the Panda's joint1..7, `human` for everything else."""
    if name.startswith("joint") and name[5:].isdigit():
        return "robot"
    return "human"


def _flat_fields(env: ax.Assistax) -> List[Dict[str, Any]]:
    import mujoco
    m = env._model
    nq, nv = env.n_q, env.n_v
    names: List[str] = [""] * (nq + nv)
    docs: List[str] = [""] * (nq + nv)
    for j in range(m.njnt):
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        qa, da = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            for k, axis in enumerate("xyz"):
                names[qa + k] = f"human_root_{axis}"
                docs[qa + k] = (f"the humanoid's root free-joint position {axis}, metres, world frame "
                                "(the `humanoid_container` body, about 1.4 m from the torso along the "
                                "body axis -- not the torso itself)")
            for k, c in enumerate("wxyz"):
                names[qa + 3 + k] = f"human_root_q{c}"
                docs[qa + 3 + k] = (f"the humanoid's root free-joint orientation quaternion component {c} "
                                    "(w, x, y, z; unit norm -- normalised explicitly at reset after the noise draw)")
            for k, axis in enumerate("xyz"):
                names[nq + da + k] = f"human_root_v{axis}"
                docs[nq + da + k] = f"the human's root linear velocity {axis}, m/s, world frame"
            for k, axis in enumerate("xyz"):
                names[nq + da + 3 + k] = f"human_root_w{axis}"
                docs[nq + da + 3 + k] = f"the human's root angular velocity about its own {axis} axis, rad/s (body frame)"
        else:
            owner = _joint_owner(jn)
            lim = m.jnt_range[j]
            label = f"Panda {jn}" if owner == "robot" else f"human {jn}"
            names[qa] = f"{owner}_{jn}_pos"
            docs[qa] = f"{label} joint angle, radians, limited to [{lim[0]:.3f}, {lim[1]:.3f}]"
            names[nq + da] = f"{owner}_{jn}_vel"
            docs[nq + da] = f"{label} joint velocity, rad/s"
    for name, doc, lo, hi in env._row["slots"]:
        names.append(name)
        docs.append(f"{doc}; in [{lo:g}, {hi:g}]")
    force_names = _force_field_names(env._row)
    for name, doc, _f, lo, hi in env._row["tail"]:
        names.append(name)
        # a contact force has a typical range, not a bound (a stiff-contact transient can
        # exceed it, and the observation is not clipped); the other fields are bounded
        word = ("typically in" if name in force_names or name.endswith("_force")
                or name.endswith("_force_mag") else "in")
        docs.append(f"{doc}; {word} [{lo:g}, {hi:g}]")
    assert all(names), names
    assert len(names) == env.obs_dim, (len(names), env.obs_dim)
    return [{"index": i, "name": n, "description": f"s[{i}] -- {d}"}
            for i, (n, d) in enumerate(zip(names, docs))]


_DIST_FIELD = {"scratchitch": "dist_tool_target", "bedbathing": "dist_tool_nearest_unwiped",
               "armmanipulation": "dist_forearm_waist_target", "feeding": "dist_tool_mouth",
               "teethbrushing": "dist_tool_mouth"}
_GOAL_FIELD = {"scratchitch": "target", "bedbathing": "nearest_unwiped",
               "armmanipulation": "waist_target", "feeding": "mouth", "teethbrushing": "mouth"}


def _helpers(flat: List[Dict[str, Any]], env: ax.Assistax) -> List[Dict[str, Any]]:
    idx = {f["name"]: f["index"] for f in flat}
    task = env._task
    robot_pos = [idx[f"robot_joint{k}_pos"] for k in range(1, 8)]
    robot_vel = [idx[f"robot_joint{k}_vel"] for k in range(1, 8)]
    human_pos = [i for n, i in idx.items() if n.startswith("human_") and n.endswith("_pos")]
    g = _GOAL_FIELD[task]
    hs = [
        ("tool_position", "tool_position(s)", "the tool reference point's position, metres, world frame",
         f"s[{idx['tool_x']}:{idx['tool_x'] + 3}]"),
        ("tool_velocity", "tool_velocity(s)", "the tool reference point's linear velocity, m/s, world frame",
         f"s[{idx['tool_vx']}:{idx['tool_vx'] + 3}]"),
        ("tool_speed", "tool_speed(s)", "the tool reference point's speed, m/s",
         f"np.linalg.norm(s[{idx['tool_vx']}:{idx['tool_vx'] + 3}])"),
        ("tool_force", "tool_force(s)", "the net contact force the tool applies to the human, newtons, world frame",
         f"s[{idx['tool_fx']}:{idx['tool_fx'] + 3}]"),
        ("tool_force_magnitude", "tool_force_magnitude(s)", "how hard the tool presses on the human, newtons",
         f"np.linalg.norm(s[{idx['tool_fx']}:{idx['tool_fx'] + 3}])"),
    ] + ([
        ("goal_position", "goal_position(s)", f"the {g.replace('_', ' ')} position the task is about, metres, world frame",
         f"s[{idx[g + '_x']}:{idx[g + '_x'] + 3}]"),
    ] if g is not None else []) + [
        ("distance_to_goal", "distance_to_goal(s)", "the distance the task's check gates on, metres",
         f"s[{idx[_DIST_FIELD[task]]}]"),
        ("robot_joint_angles", "robot_joint_angles(s)", "the Panda's seven joint angles in actuator order, radians",
         f"s[[{', '.join(map(str, robot_pos))}]]"),
        ("robot_joint_velocities", "robot_joint_velocities(s)", "the Panda's seven joint velocities in actuator order, rad/s",
         f"s[[{', '.join(map(str, robot_vel))}]]"),
        ("human_joint_angles", "human_joint_angles(s)", "every humanoid hinge joint angle in model order, radians",
         f"s[[{', '.join(map(str, sorted(human_pos)))}]]"),
    ]
    if task == "bedbathing":
        hs.append(("n_wiped", "n_wiped(s)", "how many of the 52 points have been wiped", f"s[{idx['n_wiped']}]"))
        hs.append(("unwiped_flags", "unwiped_flags(s)", "the 52 flags, 1 = still to wipe",
                   f"s[{idx['unwiped_0']}:{idx['unwiped_0'] + ax.N_WIPE_POINTS}]"))
    if "tip_force_mag" in idx:
        hs.append(("tip_force_magnitude", "tip_force_magnitude(s)",
                   "how hard the tool's contact face presses on the head, newtons", f"s[{idx['tip_force_mag']}]"))
    if task == "armmanipulation":
        hs.append(("hook_target_position", "hook_target_position(s)",
                   "the point on the human forearm the hook should engage, metres, world frame",
                   f"s[{idx['hook_target_x']}:{idx['hook_target_x'] + 3}]"))
        hs.append(("distance_tool_to_forearm", "distance_tool_to_forearm(s)",
                   "distance from the hook's platform centre to the forearm hook point, metres",
                   f"s[{idx['dist_tool_hook_target']}]"))
    return [{"name": n, "signature": sig, "description": d, "kind": "inline_expression",
             "expression": e} for n, sig, d, e in hs]


def _symbol_mapping(flat: List[Dict[str, Any]], env: ax.Assistax) -> Dict[str, str]:
    idx = {f["name"]: f["index"] for f in flat}
    task = env._task
    g = _GOAL_FIELD[task]
    tp = idx["tool_x"]
    out = {
        "distance_to_goal": f"s[{idx[_DIST_FIELD[task]]}]",
        "ee_pos": f"s[{tp}:{tp + 3}]", "end_effector_position": f"s[{tp}:{tp + 3}]",
        "tool_pos": f"s[{tp}:{tp + 3}]",
        "ee_vel": f"s[{idx['tool_vx']}:{idx['tool_vx'] + 3}]",
        "ee_speed": f"np.linalg.norm(s[{idx['tool_vx']}:{idx['tool_vx'] + 3}])",
        **({"goal_pos": f"s[{idx[g + '_x']}:{idx[g + '_x'] + 3}]",
            "target_pos": f"s[{idx[g + '_x']}:{idx[g + '_x'] + 3}]"} if g is not None else {}),
        "contact_force": f"s[{idx['tool_fx']}:{idx['tool_fx'] + 3}]",
        "contact_force_magnitude": f"np.linalg.norm(s[{idx['tool_fx']}:{idx['tool_fx'] + 3}])",
        "joint_pos": "s[[" + ", ".join(str(idx[f"robot_joint{k}_pos"]) for k in range(1, 8)) + "]]",
        "joint_vel": "s[[" + ", ".join(str(idx[f"robot_joint{k}_vel"]) for k in range(1, 8)) + "]]",
        "control_cost": "np.sum(np.square(a))",
        "action_magnitude": "np.linalg.norm(a)",
    }
    if "n_wiped" in idx:
        out["n_wiped"] = f"s[{idx['n_wiped']}]"
    if "tip_force_mag" in idx:
        out["tip_force"] = f"s[{idx['tip_force_mag']}]"
    return out


# --------------------------------------------------------------------------
# env.reset: MEASURED, then described
# --------------------------------------------------------------------------

_RESET_SAMPLES = 24


def _measured_block(task: str) -> str:
    return _MEASURED_BLOCK


_MEASURED_BLOCK = (
    f"MEASURED {DATE}: every `fields` list in this block was held against {_RESET_SAMPLES} resets "
    "with np.random.default_rng(0..23) through the adapter -- a draw's fields are exactly the "
    "observation columns whose range across those resets exceeded 0, a constant's entries did "
    "not move, and reset(default_rng(3)) twice returned identical observations. Measured by the "
    "generator itself on every run and re-measured on every suite run by "
    "tests/test_task_specs.py::test_the_declared_reset_draws_are_exactly_what_moves. Distribution "
    "numbers and bounds are READ off each entry's `source` lines, not measured.")
_MEASURED = ("`fields` measured over the seeded resets named in the block note; the numbers are "
             "read off `source`")
_MEASURED_CONST = "held constant over the seeded resets named in the block note"


def _lines(*fns: Any) -> str:
    starts, ends = [], []
    for fn in fns:
        src, start = inspect.getsourcelines(fn)
        starts.append(start)
        ends.append(start + len(src) - 1)
    return f"{min(starts)}-{max(ends)}"


def _src(*fns: Any) -> Dict[str, Any]:
    """A `source_ref` into bird/envs/assistax.py. The symbol is the QUALIFIED name, so a
    method cites `Assistax._contact_forces` (which `tests/test_task_specs.py` resolves
    through the class) and a module-level function or class cites its bare name."""
    return {"path": _AX_PATH, "lines": _lines(*fns),
            "symbol": fns[0].__qualname__ if len(fns) == 1 else None}


def _draw(role: str, quantity: str, fields: List[str], family: str, text: str,
          source: Dict[str, Any], *, basis: str = "measured", low: Any = None, high: Any = None,
          p: Any = None, condition: Any = None, derived_from: Any = None,
          not_observed_because: Any = None, note: Any = None) -> Dict[str, Any]:
    dist: Dict[str, Any] = {"family": family, "text": text}
    if low is not None:
        dist["low"], dist["high"] = low, high
    if p is not None:
        dist["p"] = p
    d: Dict[str, Any] = {"role": role, "quantity": quantity, "fields": list(fields)}
    if not fields:
        d["not_observed_because"] = not_observed_because
    d["distribution"] = dist
    if condition is not None:
        d["condition"] = condition
    if derived_from is not None:
        d["derived_from"] = derived_from
    d["source"], d["basis"] = dict(source), basis
    if note is not None:
        d["note"] = note
    return d


def _const(role: str, quantity: str, value: str, source: Dict[str, Any], *,
           fields: Any = None, basis: str = "measured", note: Any = None) -> Dict[str, Any]:
    c: Dict[str, Any] = {"role": role, "quantity": quantity, "value": value}
    if fields is not None:
        c["fields"] = list(fields)
    c["source"], c["basis"] = dict(source), basis
    if note is not None:
        c["note"] = note
    return c


def _measure_reset(env: ax.Assistax) -> Tuple[set, set]:
    rows = [np.asarray(env.reset(np.random.default_rng(i)), dtype=float).ravel()
            for i in range(_RESET_SAMPLES)]
    S = np.stack(rows)
    # "moved" is `tests/test_task_specs.py::test_the_declared_reset_draws_are_exactly_what_moves`'s
    # own criterion (ptp > 1e-6), not exact inequality: float jitter across seeds is not a draw
    # and must not be declared as one
    moving = {i for i in range(S.shape[1]) if np.ptp(S[:, i]) > 1e-6}
    a = env.reset(np.random.default_rng(3))
    b = env.reset(np.random.default_rng(3))
    assert a.tobytes() == b.tobytes(), "reset is not reproducible from its seed"
    return moving, set(range(S.shape[1])) - moving


#: Per-task additions to the draw ORDER sentence (`rng_note`): what runs after `_reset_common`.
_RESET_ORDER_NOTE = {
    "scratchitch": "; then `_rs_scratchitch`: u for the arm, the height fraction, the angle",
}
def reset_of(task: str, flat: List[Dict[str, Any]], env: ax.Assistax,
             description: Dict[str, Any]) -> Dict[str, Any]:
    """The `env.reset` block: what one `_reset(rng)` draws, measured against the adapter."""
    row = ax._TASKS[task]
    au = _AUTHORED[task]
    names = [f["name"] for f in flat]
    nq, nv, ns = env.n_q, env.n_v, env.n_slots
    moving, still = _measure_reset(env)
    q_cols = list(range(nq))
    v_cols = list(range(nq, nq + nv))
    slot_cols = list(range(nq + nv, nq + nv + ns))
    tail_cols = list(range(nq + nv + ns, env.obs_dim))
    common = _src(ax._reset_common)
    reset_fn = row["reset"]

    draws: List[Dict[str, Any]] = []
    consts: List[Dict[str, Any]] = []

    # -- the pose and velocity noise, on every generalised coordinate --------------
    qm = [names[i] for i in q_cols if i in moving]
    vm = [names[i] for i in v_cols if i in moving]
    # the noise touches every entry
    assert len(qm) == len(q_cols) and len(vm) == len(v_cols), (task, len(qm), len(vm))
    draws.append(_draw("agent_pose", "human_and_robot_pose", qm, "uniform",
                       f"the scene's `init` keyframe plus U(-{ax.RESET_NOISE:g}, {ax.RESET_NOISE:g}) on "
                       "EVERY generalised position -- the human's root position and quaternion "
                       "(normalised again after the draw; mj_forward does not do it), every human "
                       "joint and every Panda joint; the "
                       "human's and the robot's poses are one draw of one vector (upstream `reset`)",
                       common, low=[-ax.RESET_NOISE] * len(qm), high=[ax.RESET_NOISE] * len(qm),
                       note=_MEASURED))
    draws.append(_draw("agent_velocity", "human_and_robot_velocity", vm, "uniform",
                       f"U(-{ax.RESET_NOISE:g}, {ax.RESET_NOISE:g}) on every generalised velocity "
                       "(the keyframe's velocities are zero)", common,
                       low=[-ax.RESET_NOISE] * len(vm), high=[ax.RESET_NOISE] * len(vm),
                       note=_MEASURED))

    # -- the task's own draws -------------------------------------------------------
    if task == "scratchitch":
        src = _src(ax._rs_scratchitch)
        assert all(i in moving for i in slot_cols), task
        draws.append(_draw("goal", "itch_segment", ["target_arm"], "bernoulli",
                           "1 (the lower arm) with p = 0.5, else 0 (the upper arm): one U(0, 1) "
                           "against 0.5", src, p=0.5, note=_MEASURED))
        draws.append(_draw("goal", "itch_position", ["target_local_x", "target_local_y", "target_local_z"],
                           "other", "on the drawn segment's capsule surface, in the capsule's own frame: "
                           "height U(-1, 1) times the capsule's half-length along its axis (z), angle "
                           "U(0, 2 pi) around it at the capsule's radius (x, y) -- upstream `reset` "
                           "lines 128-146 with the geom's own radius and half-length", src,
                           condition="the radius and half-length depend on the itch_segment draw",
                           note=_MEASURED))
    elif task == "bedbathing":
        consts.append(_const("goal", "unwiped_flags", "1 on all 52 points: every point is still to "
                             "wipe (upstream `contact_vector = ones`)", _src(ax._rs_bedbathing),
                             fields=[names[i] for i in slot_cols], note=_MEASURED_CONST))
    else:
        assert not slot_cols, (task, ns)

    # -- the derived tail: which of it moves is a consequence of the draws above -------
    tail_moving = [names[i] for i in tail_cols if i in moving]
    tail_still = [names[i] for i in tail_cols if i not in moving]
    derived_from = "itch_position" if task == "scratchitch" else "human_and_robot_pose"
    draws.append(_draw("agent_pose", "derived_fields", tail_moving, "derived",
                       "the tool's pose and velocity, the human arm/mouth/target positions and the "
                       "distances: recomputed from the drawn state by `_Derived`, so they move "
                       "exactly as much as the noise (and, on scratchitch, the itch) moves them",
                       _src(ax._Derived), derived_from=derived_from, note=_MEASURED))
    # the still force columns: the tool force, `tip_force_mag`, and any `*_force_mag` /
    # `*_force` spelling
    force_cols = [n for n in tail_still if n.startswith("tool_f") or n.endswith("_force_mag")
                  or n.endswith("_force")]
    other_still = [n for n in tail_still if n not in force_cols]
    if force_cols:
        consts.append(_const("agent_pose", "contact_forces", "0 N: the tool starts clear of the human "
                             "in every scene, so no tool-human contact exists at reset",
                             _src(ax.Assistax._contact_forces), fields=force_cols, note=_MEASURED_CONST))
    if task == "bedbathing":
        assert other_still == ["n_wiped"], other_still
        consts.append(_const("goal", "n_wiped", "0: nothing is wiped at reset", _src(ax._derive_bedbathing),
                             fields=["n_wiped"], note=_MEASURED_CONST))
    else:
        assert not other_still, (task, other_still)
    consts.append(_const("goal", "goal", au["goal_sentence"], _src(row["check"]), basis="read"))

    # -- the block's own bookkeeping, checked here so a wrong table fails at generation --
    drawn = {f for d in draws for f in d["fields"]}
    assert drawn == {names[i] for i in moving}, (task, sorted(drawn ^ {names[i] for i in moving}))
    quantities = [d["quantity"] for d in draws]
    assert len(set(quantities)) == len(quantities), (task, quantities)
    for d in draws:
        if d.get("derived_from") is not None:
            assert d["derived_from"] in quantities, (task, d["quantity"], d["derived_from"])
    roles = {d["role"] for d in draws} | {c["role"] for c in consts}
    assert {"agent_pose", "goal"} <= roles, (task, roles)
    field, quote = au["prompt"]
    norm = lambda s: " ".join(str(s).split())  # noqa: E731
    assert norm(quote) in norm(description[field]), (task, field, quote)
    drawn_roles: List[str] = []
    for d in draws:
        if d["role"] not in drawn_roles:
            drawn_roles.append(d["role"])
    omitted = [r for r in drawn_roles if r not in au["mentions"]]
    assert set(au["mentions"]) <= set(drawn_roles), (task, au["mentions"], drawn_roles)

    order = (f"`_reset_common`: {nq} uniforms for the positions, then {nv} for the velocities"
             + _RESET_ORDER_NOTE.get(task, ""))
    return {
        "kind": "stochastic",
        "mechanism": "adapter_draws",
        "entry": {"path": _AX_PATH, "symbol": reset_fn.__name__, "lines": _lines(reset_fn)},
        "rng_note": (f"One Generator, consumed in order: `EnvAdapter.reset` samples "
                     f"`domain_randomization` first (no axes are declared, so it consumes nothing), "
                     f"then `Assistax._reset` (lines {_lines(ax.Assistax._reset)}) runs "
                     f"`{reset_fn.__name__}` -- {order} -- normalises the root quaternion "
                     "(mju_normalize4; mj_forward would not), and after one `mj_forward` the "
                     "derived fields are computed under zero ctrl; nothing else draws."),
        "draws": draws,
        "constants": consts,
        "stochastic_transitions": None,
        "upstream_divergence": au["upstream_divergence"],
        "prompt_states": quote,
        "prompt_field": field,
        "prompt_omits": omitted,
        "note": _measured_block(task),
    }


# --------------------------------------------------------------------------
# anchors
# --------------------------------------------------------------------------

def _scripted(task: str) -> Any:
    """`anchors.scripted`: this repo's own registry policy for the row, COPIED from its
    campaign manifest by the one rule `bird.policies.reference_for_env` states (best
    verdict, then score). Null where the
    registry has no entry (no Assistax row has one), explicitly, so a row without one
    cannot inherit another's on regeneration."""
    ref = reference_for_env(ax._env_id(task))
    if ref is None:
        return None
    pid, rec = ref
    sc = rec.score or {}
    campaign = pid.split("/", 1)[0]
    return {
        "value": float(sc["value"]), "method": "scripted_policy", "policy_id": pid,
        "status": rec.status, "unit": sc.get("unit"), "reduction": sc.get("reduction"),
        "n_seeds": sc.get("n_seeds"), "n_episodes": sc.get("n_seeds"), "reason": None,
        "date": str(sc.get("date")) if sc.get("date") is not None else None,
        "source": (f"REGISTRY {sc.get('date')}: policies/{campaign}/policies.yaml entry `{pid}` "
                   f"(status {rec.status}), score {sc['value']} {sc.get('unit')}, n={sc.get('n_seeds')} "
                   f"(seeds {sc.get('seeds')}), verified through {sc.get('verified_through')} at "
                   f"{sc.get('verified_at_commit')}. What THIS repo's scripted policy scores on the "
                   + "task's own metric -- the manifest is the primary record; "
                     "regenerate this file when it moves."),
    }


def _anchors(task: str) -> Dict[str, Any]:
    unavailable = {"value": None, "method": "unavailable", "source": None, "n_seeds": None,
                   "n_episodes": None, "date": None}
    rec = _RANDOM_ANCHORS.get(task)
    date = DATE
    if rec is None:
        random = dict(unavailable, reason="Not yet measured: scripts/measure_anchors.py --env "
                                          f"assistax_{task} --n 30 has not been run on this row. A "
                                          "random Panda waves beside the human and rarely touches "
                                          "it, so the floor is expected to be ~0 -- expected, not "
                                          "measured, which is why this is null.")
    else:
        mean, half, est = rec
        est_text = {"wilson": "95% Wilson interval half-width (every episode scored exactly 0)",
                    "sem": "95% t/normal SEM interval half-width"}[est]
        random = {"value": mean, "method": "random_policy",
                  "source": f"MEASURED {date}: mean `task_metric` over n=30 uniform-random episodes, "
                            f"`scripts/measure_anchors.py --env assistax_{task} --n 30`, seed 0; "
                            f"{est_text} {half}.",
                  "n_seeds": 30, "n_episodes": 30, "reason": None, "date": date}
    expert_reason = ("No scripted expert ships with Assistax and its trained policies are JAX "
                     "partner zoos this repo does not run. The remaining route is "
                     "scripts/train_expert.py on this env's own reference_reward at a budget "
                     "matched to the search's policy trainings, recorded as method: trained_policy.")
    if task == "bedbathing":
        expert_reason = ("No scripted expert ships with Assistax and its trained policies are JAX "
                         "partner zoos this repo does not run. scripts/train_expert.py on this env's "
                         "reference_reward is NOT a route here: the port drops upstream's +3 per "
                         "newly wiped point (it needs the previous flags), so the reference carries "
                         "no term that pays for wiping -- its maximiser hovers about 3 cm from an "
                         "unwiped point (measured 0.991/step, 946 over 1000 steps) and out-scores a "
                         "policy that wipes all 52 points (305). A policy trained on it would not be "
                         "an expert at the task the metric measures.")
    expert = dict(unavailable, reason=expert_reason)
    return {
        "reduction": "per_step_fraction",
        "by_reduction": {"per_step_fraction": {"random": random, "expert": expert}},
        "random": dict(random, source="See anchors.by_reduction.") if rec is not None
        else dict(random, reason="See anchors.by_reduction."),
        "expert": dict(unavailable, reason="See anchors.by_reduction."),
        # A COPY of the registry score iff `bird.policies.reference_for_env` names an entry
        # for the env, else explicitly null -- the schema's own spelling of "no registry
        # entry yet" (a block with `method: unavailable` would be an anchor with nothing
        # behind it). No Assistax row has a registry policy.
        "scripted": _scripted(task),
    }


# --------------------------------------------------------------------------
# exploits
# --------------------------------------------------------------------------

def _exploit(entry: Any, task: str) -> Dict[str, Any]:
    """`(name, mechanism)` -> open; `(name, mechanism, closed_by)` -> closed, naming the conjunct
    and the test that pins it; `(name, mechanism, status, closed_by)` as given. A closed entry
    has no residual."""
    if len(entry) == 2:
        name, mech = entry
        status, closed_by = "open", None
    elif len(entry) == 3:
        name, mech, closed_by = entry
        status = "closed"
    else:
        name, mech, status, closed_by = entry
        assert status in ("open", "closed", "undetectable_here"), (name, status)
        assert (status == "closed") == bool(closed_by), (name, status, closed_by)
    return {"name": name, "status": status, "closed_by": closed_by, "mechanism": mech,
            "note": "Present in the check this repo wrote, not in a benchmark's."}


#: The family's sentence on `discrete_success.no_discrete_success_because`.
_NO_DISCRETE_VENDORED = ("Assistax ships no boolean success check for any task -- "
                         "its agents are judged by the return of its own dense "
                         "reward -- so there is no benchmark criterion to state. "
                         "`EnvAdapter.success()` here is a threshold WE chose on a "
                         "per-step check WE wrote, and recording it as a `shipped` "
                         "check would claim an independence it does not have.")


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def build(task: str) -> Dict[str, Any]:
    if task not in _AUTHORED:
        raise SystemExit(f"gen_assistax_specs.py: no authored prose for {task!r}; refusing to write "
                         "a placeholder")
    au = _AUTHORED[task]
    missing = [k for k in _REQUIRED_AUTHORED if k not in au]
    if missing:
        raise SystemExit(f"gen_assistax_specs.py: {task} authored prose lacks {missing}")
    env = construct_without_spec(task)
    row = ax._TASKS[task]
    flat = _flat_fields(env)
    idx = {f["name"]: f["index"] for f in flat}
    env_id = ax._env_id(task)
    fitness = row["fitness"]
    support = _SUPPORT[task]
    scene_file = row["scene"]
    nq, nv, ns = env.n_q, env.n_v, env.n_slots
    control_hz = round(1.0 / env.dt)
    description = {
        "env_prose": au["env_prose"],
        "l_task": au["l_task"],
        "natural_language": au["env_prose"],
        "scene": au["scene"],
        "success_criterion_prose": au["success_criterion_prose"],
        "full_source": {"path": _AX_PATH, "lines": None},
    }
    human_hinge = [i for i in range(7, nq) if flat[i]["name"].startswith("human_")]
    robot_hinge = [i for i in range(nq) if flat[i]["name"].startswith("robot_")]
    fields = [
        {"name": "human_root_pose", "shape": "(7,)", "units": "m, unit quaternion",
         "description": "the human's root position and orientation (w, x, y, z)", "source": scene_file,
         "slice": "0:7"},
        {"name": "human_joint_angles", "shape": f"({len(human_hinge)},)", "units": "rad",
         "description": "every humanoid hinge joint angle in model order",
         "source": scene_file, "slice": f"{min(human_hinge)}:{max(human_hinge) + 1}"},
        {"name": "robot_joint_angles", "shape": "(7,)", "units": "rad",
         "description": "the Panda's seven joint angles, joint1..joint7",
         "source": scene_file, "slice": f"{min(robot_hinge)}:{max(robot_hinge) + 1}"},
        {"name": "human_root_velocity", "shape": "(6,)", "units": "m/s, rad/s",
         "description": "the human's root linear velocity (world frame) then angular velocity (body frame)",
         "source": scene_file, "slice": f"{nq}:{nq + 6}"},
        {"name": "joint_velocities", "shape": f"({nv - 6},)", "units": "rad/s",
         "description": "every hinge joint velocity in model order (human then Panda)",
         "source": scene_file, "slice": f"{nq + 6}:{nq + nv}"},
    ]
    if ns:
        fields.append({"name": "task_slots", "shape": f"({ns},)", "units": "task-specific",
                       "description": _SLOT_BLOCK_DESC.get(task) or "; ".join(f"{s[0]}: {s[1]}" for s in row["slots"]),
                       "source": f"{_AX_PATH}::_TASKS", "slice": f"{nq + nv}:{nq + nv + ns}"})
    t0 = nq + nv + ns
    fields.append({"name": "tool_pose_velocity_force", "shape": "(13,)", "units": "m, unit quaternion, m/s, N",
                   "description": "tool position (3), tool orientation quaternion (4), tool linear velocity "
                                  "(3), net contact force the row's tool geoms apply to the row's human "
                                  "geoms (3; the right arm on the arm tasks, the head on the head tasks; "
                                  "a range, not a bound: stiff-contact transients can exceed it), world frame",
                   "source": f"{_AX_PATH}::_COMMON_TAIL", "slice": f"{t0}:{t0 + 13}"})
    fields.append({"name": "task_fields", "shape": f"({env.obs_dim - t0 - 13},)", "units": _TASK_FIELD_UNITS[task],
                   "description": "; ".join(f"{t[0]}: {t[1]}" for t in row["tail"][13:]),
                   "source": f"{_AX_PATH}::_TASKS", "slice": f"{t0 + 13}:{env.obs_dim}"})
    return {
        "schema_version": 1,
        "id": env_id,
        "env": {
            "env_id": task,
            "library": {"name": "assistax", "version_verified": UPSTREAM_COMMIT[:12],
                        "commit": UPSTREAM_COMMIT,
                        "stack": {"mujoco": measured_mujoco(),
                                  **override_entries(EXTRA, measured_mujoco()),
                                  "numpy": ">=1.24", "python": ">=3.10"}},
            "horizon": int(ax.HORIZON),
            "control_hz": control_hz,
            "termination": {
                "early_terminate": False, "autoreset": False, "partial_reset": None,
                "prompt_states": (f"The episode always runs the full {ax.HORIZON} steps "
                                  f"({ax.HORIZON * env.dt:g} s at {control_hz} Hz); nothing ends it early."
                                  + (" Upstream ends the episode the step every point is wiped; this "
                                     "repo removes that terminal (see bird/envs/mujoco_control.py)."
                                     if task == "bedbathing" else
                                     " Upstream has no terminal on this task either."))},
            "reset": reset_of(task, flat, env, description),
            "spaces": {
                "obs_dim": int(env.obs_dim), "action_dim": ax.ACTION_DIM,
                "action_low": [-1.0] * ax.ACTION_DIM, "action_high": [1.0] * ax.ACTION_DIM,
                "action_semantics": "7 normalised joint-position TARGETS, one per Panda actuator "
                                    "(actuator1..actuator7, joints 1..7), in [-1, 1]: each drives a "
                                    "stiff PD servo towards about pi times the value in radians, "
                                    "clipped to the actuator's control range (the joint's own "
                                    "limits). Targets for a position controller, not torques. The "
                                    "human's actuators receive zero.",
                "action_fields": [
                    {"name": f"joint{k}_target", "description": f"a[{k - 1}] -- normalised position "
                     f"target for Panda joint {k} (actuator{k}): target angle ~ pi * a[{k - 1}] rad, "
                     "clipped to the actuator's control range", "low": -1.0, "high": 1.0}
                    for k in range(1, 8)]},
        },
        "description": description,
        "state_surface": {
            "fields": fields,
            "helpers": _helpers(flat, env),
            "helpers_note": "Advertised as inlineable EXPRESSIONS and deliberately not implemented "
                            "as methods: verification binds a proxy `self` and training binds "
                            "`self=None`, so a helper that existed would let a `self.x(s)` "
                            "candidate pass the gate and then die in training -- a wasted policy "
                            "run charged to the reward that did not cause it.",
            "flat_fields": flat,
        },
        "discrete_success": {
            "kind": "continuous_only",
            "no_discrete_success_because": _NO_DISCRETE_VENDORED,
            "shipped": None, "hardened": None,
        },
        "continuous_success": {
            "raw": {"name": fitness, "expr": _FITNESS_EXPR[fitness],
                    "unit": _FITNESS_UNIT.get(fitness, "fraction of arriving states"),
                    "direction": "maximize", "saturates_at": 1.0},
            "normalized": {"method": "explicit",
                           "formula": "identity -- the raw metric is already a fraction in [0, 1].",
                           "success_threshold": float(row["threshold"])},
        },
        "anchors": _anchors(task),
        "budget": {"train_steps": 2000000,
                   "learning_verified": {"verified": False, "reward_used": "none", "algo": None,
                                         "seeds": None,
                                         "result": ("Not established as a floor by a dedicated sweep. "
                                                    "Upstream trains its MARL baselines for 40 M "
                                                    "steps on a GPU; nothing here is comparable."),
                                         "date": None},
                   "saturates_at": None},
        "reward": {
            "human": {
                "kind": "published_dense", "scope": "value_determining",
                "reference": {"path": _UPSTREAM_FILE[task], "lines": None,
                              "symbol": f"{row['upstream']}.step", "commit": UPSTREAM_COMMIT,
                              "sha256": None},
                "depends_on": [
                    {"path": "assistax/envs/base_env.py", "lines": None, "symbol": None,
                     "commit": UPSTREAM_COMMIT, "sha256": None},
                    {"path": "assistax/envs/README.md", "lines": None, "symbol": None,
                     "commit": UPSTREAM_COMMIT, "sha256": None}],
                "import_path": None,
                "verified_against": f"assistive-autonomy/assistax @ {UPSTREAM_COMMIT[:12]}, read {DATE}",
                "notes": f"Upstream's {row['upstream']}.step reward, transcribed into "
                         "`bird/envs/assistax.py` term by term at the constructor defaults, with "
                         "four stated substitutions: the tool speed is the simulator's instantaneous "
                         "site velocity rather than the finite difference of two frames; contact forces "
                         "are summed over the live tool-human contact pairs (and read back off the "
                         "observation when a stored state is re-scored) rather than read at fixed MJX "
                         "contact-array indices; the control cost squares the 7-D robot action clipped "
                         "to [-1, 1] where upstream squares the concatenated robot+human action; "
                         "bedbathing's per-point bonus (needs the previous flags) is dropped. "
                         "Transcribed, not imported: upstream is JAX/MJX.",
            },
            "contract": {
                "signature": "def compute_reward(state, action=None, next_state=None):",
                "framework": "numpy", "batched": False, "allowed_modules": ["numpy", "math"],
                "returns": "(scalar, components)", "max_reward": "<float>, declared by the candidate",
                "idiom_rule": "One scalar out, over the FLAT observation array. There is no `self` "
                              "and no env handle: the compiler binds arguments by name and maps "
                              "`self`/`env` to None deliberately.",
                "reward_utils": [],
            },
            "forbidden_symbols": [],
            "forbidden_symbols_by_consumer": {"bird": list(ax._CONSUMER_FORBIDDEN)},
            "forbidden_symbols_note": "The BIRD entries name adapter internals -- how a candidate "
                                      "could reach the metric it is scored on IN THIS REPO -- which "
                                      "are not properties of the task and cannot be expressed in the "
                                      "flat list above (empty: upstream ships no forbidden-symbol set).",
        },
        "judge": dict({
            "vlm_legible": True,
            "legibility_note": au["legibility_note"],
            "camera": {"name": "default", "mode": _CAMERA_MODE[support],
                       "note": _CAMERA_NOTE[support]},
            "subtasks_sequential": False,
            "telemetry_fields": list(au["telemetry"]),
        }),
        "domain_randomization": None,
        "exploits": [_exploit(e, task) for e in au["exploits"]],
        "symbol_mapping": _symbol_mapping(flat, env),
        "provenance": {
            "authored_from": [
                {"path": f"bird/envs/assets/assistax/{scene_file}", "lines": None,
                 "what": "the joint inventory and every `range=`, the actuator gains/ctrlrange, the "
                         "geom sizes, the sites and the camera (vendored verbatim from upstream by "
                         "scripts/vendor_assistax_assets.py, meshes decimated)"},
                {"path": _AX_PATH, "lines": None,
                 "what": "the row: tool, human geoms, task slots, derived fields, reset, the per-step "
                         "check, the threshold and the reference reward"},
                {"path": f"refs: assistive-autonomy/assistax @ {UPSTREAM_COMMIT[:12]}, "
                         f"{_UPSTREAM_FILE[task]}, assistax/envs/base_env.py, assistax/envs/README.md",
                 "lines": None,
                 "what": "upstream's reset, observation, reward terms and weights, termination and "
                         "agent/actuator mapping for this task"},
            ],
            "rule": "source and asset, not docstrings",
            "measured": [
                f"Step purity: bitwise over shuffled replay, tests/test_assistax.py ({DATE}); 0/40 "
                "mismatches on every task once the actuators are written before the restoring "
                "forward (11-28/40 at ~1e-14 with them written after it).",
                "Random-action rollouts: the tool "
                "touches the human within 300 steps only on armmanipulation, where the hook starts "
                "under the forearm (about one seed in three, forces up to 286 N); on the other four "
                "tasks the two never meet. Under zero actuation the seated human's bodies move down "
                "by at most 3.3-3.6 cm over the first 2.4 s on the wheelchair scenes and rise "
                "slightly (no settling) on the bed scenes; 0.4-0.5 ms per control step.",
                f"Upstream under its own stack (jax 0.8.0, mujoco 3.4.0, CPU, {DATE}): "
                f"{_MJX_STEP_MS[task]:.0f} ms per control step after a 20-40 s JIT; from the same "
                "keyframe under zero actions the MJX solve (iterations 1) and this port's C solve "
                "(iterations 100) part by a max |dq| of "
                f"{_MJX_VS_C[task][0]:.3f} after 1 step, {_MJX_VS_C[task][1]:.3f} after 10 and "
                f"{_MJX_VS_C[task][2]:.3f} after 100 -- same model, different physics.",
                "Upstream reads the tool-human force at hard-coded MJX contact-array indices, which "
                "denote whatever geom pair the MJX collision enumeration puts there (a property of the "
                "build, not of the task; mjx.make_data reports 250-282 static contact rows per scene, "
                "and this repo has not confirmed which pairs the indices denote against a stepped MJX "
                "Data -- a reconstruction of the enumeration puts them on tool-vs-human pairs, which is "
                "recorded as unverified); this port sums the live tool-human contacts by geom pair "
                "instead.",
            ] + (["Upstream's 52 bedbathing points, mapped through the arm BODY frames as its code "
                  "does, lie off the capsule surfaces at the keyframe: the 26 upper-arm points "
                  "0.031-0.135 m off the axis (radius 0.04 m) and up to 0.19 m along it "
                  "(half-length 0.139 m), the 26 forearm points 0.027-0.128 m off the axis "
                  "(radius 0.031 m) and up to 0.22 m along it; 8 of the 52 sit inside the mattress. "
                  "This port lays both sets in the GEOM frame, on the surface."] if task == "bedbathing" else []),
            "verified_by": "generated by scripts/gen_assistax_specs.py",
            "date": DATE,
        },
    }


# --------------------------------------------------------------------------
# write / check
# --------------------------------------------------------------------------

_HEADER = ("# GENERATED by scripts/gen_assistax_specs.py -- edit the generator, not this file.\n"
           "# AUTHORED FOR BIRD from assistive-autonomy/assistax (Apache-2.0), commit\n"
           f"# {UPSTREAM_COMMIT}. The flat_fields table is read off the live MuJoCo model and\n"
           "# the env.reset block is measured against the adapter; the prose is authored in the\n"
           "# generator. Same rules as the rest of the catalogue: absence is explicit, and every\n"
           "# claim came off the model, the adapter or upstream's source rather than a docstring.\n")


def _dump(doc: Dict[str, Any]) -> str:
    return _HEADER + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed spec is stale")
    ap.add_argument("--allow-foreign-wheel", action="store_true",
                    help="write under a wheel this tier does not pin; the spec then "
                         "records the crossing and the provenance test goes red")
    ap.add_argument("--tasks", nargs="*", default=None)
    args = ap.parse_args(argv)
    tasks = args.tasks or list(ax._EXPECTED)
    stale, unjudgeable = [], []
    installed = measured_mujoco()
    # REFUSE TO WRITE under a wheel this tier does not pin. The read side above already
    # refuses to JUDGE across wheels; without this, the same run can still silently
    # CREATE the record it would then refuse to check. Compared against the tier's PIN,
    # never against the committed spec's recorded value: the pin is the declaration of
    # intent, so moving a pin stays one change (move it, regenerate under the newly
    # pinned wheel, every `stack.mujoco` moves beside its re-measured numbers) instead
    # of a tier-wide red demanding an override on every spec.
    if not args.check:
        refusal = refuse_foreign_write(EXTRA, installed, args.allow_foreign_wheel)
        if refusal:
            print(refusal, file=sys.stderr)
            return 2
    for task in tasks:
        path = REPO / "tasks" / ax._env_id(task) / "shared_spec.yaml"
        # REFUSE BEFORE REGENERATING when the wheel differs from the one the committed
        # spec was measured under. Every measured block here is a property of the SOLVER
        # as much as of the scene, so a change of wheel can flip a column between mover
        # and constant. Reporting "stale" invites the one repair that destroys the record -- a
        # regeneration that overwrites correct measurements with this wheel's, after
        # which every test is green because the generator and the guard agree with each
        # other about a wheel the tier does not run. Non-zero, not a warning: a warning
        # in a regeneration script is read by someone already committed to regenerating.
        #
        # The comparison is ALWAYS spec-against-the-interpreter-checking-it. It is never
        # a derived spec against its parent: `upstream_assistax_*` is derived from these CPU
        # specs by `scripts/derive_jax_spec.py`, and a measurement there records ITS OWN
        # measuring version, so a 3.3.0 parent is not a disagreement.
        if args.check and path.is_file():
            recorded = recorded_mujoco(path)
            if unjudgeable_under(recorded, installed):
                unjudgeable.append((str(path.relative_to(REPO)), recorded))
                continue
        text = _dump(build(task))
        if args.check:
            if not path.is_file() or path.read_text() != text:
                stale.append(str(path.relative_to(REPO)))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print("wrote", path.relative_to(REPO))
    if args.check:
        note = check_under_foreign_wheel(EXTRA, installed)
        if note:
            print(note)
        if unjudgeable:
            for rel, recorded in unjudgeable:
                print(f"cannot check under mujoco {installed}, spec measured under "
                      f"{recorded}: {rel}")
            print(f"{len(unjudgeable)} spec(s) not checked. Install the recorded wheel "
                  "(`uv sync --extra assistax`, which pins it) and re-run; do NOT "
                  "regenerate to make this go away -- that rewrites measurements taken "
                  "under the other wheel.")
        if stale:
            print("stale:", *stale)
        if stale or unjudgeable:
            return 1
        print(f"ok: {len(tasks)} specs match the generator")
    return 0


if __name__ == "__main__":
    sys.exit(main())
