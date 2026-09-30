#!/usr/bin/env python3
"""Derive a HumanoidBench task's model facts FROM THE MODEL, and prove it against a spec.

WHY THIS IS A COMMITTED SCRIPT AND NOT A SCRATCH FILE. Every h1hand task spec in `tasks/`
carries ~1,500 lines describing an observation layout, an actuator table and a set of
scene bodies. Those are facts about a MuJoCo model, and there are exactly two ways to get
them: read the model, or copy a sibling spec and edit the parts you noticed. The second
is faster and is wrong in a way review does not catch -- a `s.target_pos` description
copied verbatim from push into the package spec would be false in BOTH halves (the
destination is on the floor, not the tabletop, and it is model state, not task-object
state). **A renamed splice is not a derivation.**

This is the deriving tooling, generalised over the task list and given a REPRODUCTION
CONTROL.

    uv run python3 scripts/derive_humanoid_facts.py --task walk --check tasks/h1hand_walk
    uv run python3 scripts/derive_humanoid_facts.py --all --out /tmp/facts.json

THE CONTROL IS THE POINT. `--check <spec dir>` re-derives a task whose spec is already
committed and asserts the derived actuator table and observation width match what that
spec records. Run it before trusting a derivation for a NEW task: it is the difference
between "this script ran" and "this script agrees with a spec a human reviewed". It also
catches the toolchain drifting -- the committed specs record `mujoco 3.1.6` and this
repo's venv currently resolves 3.3.0, and the only reason to believe that is safe is that
`--check` passes 61/61 actuators and 151/151 observation slots on `walk`.

WHAT IT DELIBERATELY DOES NOT DO. It does not write YAML. Everything a spec needs beyond
these facts -- the prose, the reward defects in `exploits:`, the `task_metric` that
falsifies them, the anchors -- comes from READING the upstream reward code, and a
generator that emitted a plausible-looking spec for those would be manufacturing exactly
fabricated pins -- declared values nothing measured. This produces the half that can be
checked mechanically, so a human spends their attention on the half that cannot.

ASSETS. Needs HumanoidBench's `assets/envs/*.xml` and `mujoco`, but NOT `humanoid_bench`
importable -- the XML is loaded by path, so the specs can be derived correctly in an
interpreter that cannot construct the environment.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

#: Each robot's own qpos width, MEASURED, never guessed. Joints at or past this address
#: belong to the SCENE (the package, the door, the cabinet shelves), not the robot, and
#: they are what makes one task's observation wider than another's. Recorded once here
#: rather than per task because every model of one robot shares that robot -- which
#: `--check` verifies. h1hand is this repo's whole tier; h1strong is the same
#: Shadow-Hands embodiment with stronger fingers (76 qpos, verified in
#: bird/envs/humanoid_hand.py's docstring). A `--robot` value with no entry here is
#: REFUSED rather than silently computed against the wrong width.
_ROBOT_NQ = {"h1hand": 76, "h1strong": 76}


def robot_nq(robot: str) -> int:
    try:
        return _ROBOT_NQ[robot]
    except KeyError:
        raise SystemExit(
            f"--robot {robot}: this script knows the robot-only qpos width for "
            f"{sorted(_ROBOT_NQ)} and nothing else. scene_joints (and the splice gate "
            "built on it) are computed from that width, so a guess here mislabels "
            "robot joints as scene or vice versa -- measure the robot's own block and "
            "add it to _ROBOT_NQ rather than defaulting.")


#: h1hand's width, kept as a plain constant for readers of the h1hand-only callers.
ROBOT_NQ = _ROBOT_NQ["h1hand"]

#: The upstream checkout `scripts/setup_humanoid.sh` clones (`.forks/humanoid-bench`); pass
#: `--assets` to read another one.
DEFAULT_ASSETS = str(pathlib.Path(__file__).resolve().parent.parent / ".forks" / "humanoid-bench"
                     / "humanoid_bench" / "assets" / "envs")


def derive(task: str, assets: str, robot: str = "h1hand") -> dict:
    """Every model fact a spec records, read off the model."""
    import mujoco

    path = pathlib.Path(assets) / f"{robot}_pos_{task}.xml"
    if not path.exists():
        raise SystemExit(f"no asset for {robot}/{task}: {path}")
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    def name(kind, i):
        return mujoco.mj_id2name(m, kind, i)

    def joint_name(j: int) -> str:
        """The joint's own name, or one SYNTHESISED from its body when the XML leaves it
        anonymous -- `window`'s wiper head is `<joint type="ball" range="0 1.4"/>` with no
        name at all, and a spec row cannot be nameless. The synthesis is deterministic
        (body name + "_joint"), so a delta can reference it and a re-derivation agrees."""
        n = name(mujoco.mjtObj.mjOBJ_JOINT, j)
        if n:
            return n
        return f"{name(mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j]))}_joint"

    joints = [{
        "name": joint_name(j),
        "type": int(m.jnt_type[j]),
        "qposadr": int(m.jnt_qposadr[j]),
        "dofadr": int(m.jnt_dofadr[j]),
        "range": [round(float(x), 6) for x in m.jnt_range[j]],
    } for j in range(int(m.njnt))]

    actuators = [{
        "name": name(mujoco.mjtObj.mjOBJ_ACTUATOR, i),
        # The RAW range. The action a policy emits is normalised to [-1, 1] by
        # upstream's `Task.normalize_action`, so a spec records +/-1 as the bound and
        # this range in the field's description. Both, or the description lies.
        "ctrlrange": [round(float(x), 6) for x in m.actuator_ctrlrange[i]],
    } for i in range(int(m.nu))]

    return {
        "task": task,
        "robot": robot,
        "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
        "qpos_qvel_width": int(m.nq) + int(m.nv),
        "joints": joints,
        "actuators": actuators,
        # The scene half: what this task adds on top of the robot.
        "scene_joints": [j for j in joints if j["qposadr"] >= robot_nq(robot)],
        "bodies": {name(mujoco.mjtObj.mjOBJ_BODY, b): {
            "id": b,
            "pos": [round(float(x), 6) for x in m.body_pos[b]],
            "xpos": [round(float(x), 6) for x in d.xpos[b]],
            "mass": round(float(m.body_mass[b]), 6),
        } for b in range(int(m.nbody))},
        "sites": {name(mujoco.mjtObj.mjOBJ_SITE, s): {
            "id": s,
            "xpos": [round(float(x), 6) for x in d.site_xpos[s]],
            "local_pos": [round(float(x), 6) for x in m.site_pos[s]],
            "body": name(mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[s])),
        } for s in range(int(m.nsite))},
        "last_body": name(mujoco.mjtObj.mjOBJ_BODY, int(m.nbody) - 1),
    }


def check(facts: dict, spec_dir: str, expected_extra: int = 0) -> list[str]:
    """Re-derived facts vs a COMMITTED spec. Returns the disagreements.

    Deliberately checks the two things a splice gets wrong silently: the actuator table
    (names AND the raw range each field's description quotes) and the observation width.
    A spec that passes this was not copied from a task with a different scene.
    """
    import yaml

    spec = yaml.safe_load((pathlib.Path(spec_dir) / "shared_spec.yaml").read_text())
    problems = []

    fields = spec["env"]["spaces"]["action_fields"]
    if len(fields) != len(facts["actuators"]):
        problems.append(f"action_fields: spec {len(fields)} vs model {len(facts['actuators'])}")
    for got, want in zip(fields, facts["actuators"]):
        if got["name"] != want["name"]:
            problems.append(f"actuator name: spec {got['name']!r} vs model {want['name']!r}")
            continue
        lo, hi = want["ctrlrange"]
        quoted = f"[{lo:g}, {hi:g}] rad"
        if quoted not in got.get("description", ""):
            problems.append(
                f"{got['name']}: description does not quote the model's range {quoted}")
        if (got.get("low"), got.get("high")) != (-1.0, 1.0):
            problems.append(f"{got['name']}: action bounds are not the normalised +/-1")

    flat = spec.get("state_surface", {}).get("flat_fields") or []
    declared = spec["env"]["spaces"].get("obs_dim")
    if len(flat) != declared:
        problems.append(f"flat_fields {len(flat)} != declared obs_dim {declared}")
    # THE OBSERVATION WIDTH IS THE ONLY THING HERE THAT DISCRIMINATES BETWEEN TASKS, so
    # it is a hard failure and not a note. Every h1hand model shares one robot and one
    # 61-actuator table -- verified across all 25 committed h1hand/h1strong specs -- which means the
    # actuator checks above pass for ANY pairing of task and spec. Measured: derived
    # `walk` (151) against the committed `door` spec (155) reported AGREES while that
    # surplus was only a note, so the control could not catch a spec spliced from the
    # wrong task, which is the exact failure it exists for.
    #
    # `expected_extra` is 0 unless the spec carries task state outside qpos+qvel, which
    # happens two ways: upstream overrides `get_obs` (reach, cube, bookshelf), or BIRD
    # declares `extra_dims` for state upstream keeps on the task object (push 3,
    # package 3, basketball 1). Passing it explicitly is the point -- the caller states
    # the surplus they expect and the model either agrees or does not.
    surplus = (declared or 0) - facts["qpos_qvel_width"]
    if surplus != expected_extra:
        problems.append(
            f"obs_dim {declared} minus qpos+qvel {facts['qpos_qvel_width']} is {surplus}, "
            f"expected {expected_extra}. Either this spec was derived from a different "
            f"task's model, or upstream overrides get_obs and --extra-dims is wrong")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", help="one HumanoidBench task name, e.g. walk")
    ap.add_argument("--all", action="store_true", help="every asset present for --robot")
    ap.add_argument("--robot", default="h1hand")
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    ap.add_argument("--check", metavar="SPEC_DIR",
                    help="re-derive and compare against a committed spec directory")
    ap.add_argument("--extra-dims", type=int, default=0, metavar="N",
                    help="observation slots the spec carries BEYOND qpos+qvel. Two "
                         "sources: upstream overriding get_obs (reach, cube, bookshelf) "
                         "and BIRD's own extra_dims for task state outside the "
                         "simulator (push 3, package 3, basketball 1). Default 0 -- a "
                         "surplus is then a failure, which is what makes this check "
                         "able to tell two tasks apart.")
    ap.add_argument("--out", help="write the derived facts here as JSON")
    args = ap.parse_args()

    if args.all:
        tasks = sorted(p.stem.split("_pos_", 1)[1]
                       for p in pathlib.Path(args.assets).glob(f"{args.robot}_pos_*.xml"))
    elif args.task:
        tasks = [args.task]
    else:
        ap.error("one of --task or --all")

    # `--all` SKIPS an asset the installed mujoco cannot build, and says so, rather than
    # dying on the first one: measured, `h1hand_pos_kitchen.xml`'s family fails
    # under 3.3.0 with "top-level default class 'main' cannot be renamed", and one broken
    # sibling should not hide the facts of the twenty-nine that build. A single --task is
    # still a hard failure -- the caller asked about exactly that asset.
    facts = {}
    for t in tasks:
        try:
            facts[t] = derive(t, args.assets, args.robot)
        except Exception as exc:  # noqa: BLE001 - report and continue under --all only
            if not args.all:
                raise
            print(f"{t:16} UNBUILDABLE under this mujoco: {exc}", file=sys.stderr)

    if args.check:
        if len(tasks) != 1:
            ap.error("--check takes a single --task")
        problems = check(facts[tasks[0]], args.check, args.extra_dims)
        hard = [p for p in problems if not p.startswith("NOTE")]
        for p in problems:
            print(("  " if p.startswith("NOTE") else "  MISMATCH ") + p)
        print(f"{args.check}: {'AGREES' if not hard else str(len(hard)) + ' MISMATCH(ES)'} "
              f"with the model ({facts[tasks[0]]['nu']} actuators, "
              f"{facts[tasks[0]]['qpos_qvel_width']} qpos+qvel)")
        if hard:
            return 1

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(facts, indent=1))
        print(f"wrote {args.out} ({len(facts)} task(s))")
    elif not args.check:
        for t, f in facts.items():
            print(f"{t:16} nq={f['nq']:4} nv={f['nv']:4} nu={f['nu']:3} "
                  f"qpos+qvel={f['qpos_qvel_width']:4} scene_joints={len(f['scene_joints'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
