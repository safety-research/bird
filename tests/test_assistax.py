"""The Assistax adapter (`bird/envs/assistax.py`).

Two halves, split by what they need installed, and only the properties whose failure is
SILENT:

**Without the simulator** (a subprocess with `mujoco` poisoned, via `conftest.run_probe`):
`registry.load_all()` imports this module on every CLI path and forgives an ImportError
only for `anthropic_client`, so the module must import, register its ids and leave
`MUJOCO_GL` alone with no simulator present -- and constructing one must fail with the extra
named AND no side effect on the process.

**With the simulator** (`uv sync --extra assistax`): `_step(s, a)` is a
function of the state passed in, asserted BITWISE over shuffled replay -- an approximate
restore has no other symptom, the screens just quietly score physics that never happened.
The one measured way this adapter loses that property: setting the actuators AFTER the
restoring `mj_forward`, so the warmstart the first substep reads carries the previous
caller's action (11-28 of 40 shuffled replays off by ~1e-14). The
test carries that as a CONTROL: with the order reversed, replay must FAIL, or the purity
assertion is passing on an env whose solver never reads the warmstart.

The rest are the claims the module docstring makes and a reader would otherwise take on
trust: the observation IS the simulator state (restoring it reproduces every derived
field), nothing ends an episode early, the passive human gets zero on every actuator, the
itch lies on the arm's surface, a wiped point stays wiped and the bedbathing metric is the
wiped count, and the reference reward is a pure function of the row it is handed.

**The render panel** (`Assistax._legend_lines`): the task line `render` draws must be
legible in the HUD font and must not depend on the state -- pinned without the simulator.

**In contact**: uniform-random actions never
bring the tool to the human -- 0 N on every state of every `_rollout` above -- so none of
that exercised the contact-force sign or frame, the force and speed terms of any reference
reward, the marking rule that IS bedbathing, any success-check clause, or a stored row's
force being re-solved under whatever `d.ctrl` the previous caller left. The `contact` fixture drives the tool INTO the human on every task with a greedy
descent and the tests after it read the contact states it produces.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pytest

from bird import registry
from bird.config import ConfigError, load
from bird.envs import hud
from bird.envs.assistax import (_CONSUMER_FORBIDDEN, _EXPECTED, _LEGEND_COLS, _TASKS, ACTION_DIM,
                                N_WIPE_POINTS, WIPE_THRESHOLD, WIPED_OUT_DIST, _Derived,
                                _env_id, _task_lines)

pytestmark = pytest.mark.slow

ALL_IDS = [_env_id(t) for t in _EXPECTED]


def _need_simulator() -> None:
    """Skip unless `mujoco` is installed -- WITHOUT importing it (an import here can
    segfault on an OSMesa host: the adapter preloads Triton and picks the GL backend
    itself)."""
    import importlib.util

    try:
        if importlib.util.find_spec("mujoco") is None:
            pytest.skip("mujoco is not installed (uv sync --extra assistax)")
    except (ImportError, ValueError):  # pragma: no cover - a broken install
        pytest.skip("mujoco is not importable")


# --------------------------------------------------------------------------
# without the simulator
# --------------------------------------------------------------------------

_NO_SIMULATOR_PROBE = r"""
import json, os, sys

sys.modules["mujoco"] = None
os.environ.pop("MUJOCO_GL", None)

import bird.envs.assistax as ax
from bird import registry

out = {
    "gl_after_import": os.environ.get("MUJOCO_GL"),
    "imported": sorted(m for m in ("mujoco",) if sys.modules.get(m) is not None),
    "ids": sorted(n for n in registry.names("env") if n.startswith("assistax_")),
}
try:
    ax.Assistax("scratchitch")
except Exception as exc:
    out["error_type"] = type(exc).__name__
    out["error"] = str(exc)
else:
    out["error_type"] = None
    out["error"] = ""
out["gl_after_failed_construction"] = os.environ.get("MUJOCO_GL")
print("PROBE" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def without_simulator() -> dict:
    from conftest import run_probe

    return run_probe(_NO_SIMULATOR_PROBE)


def test_the_module_imports_and_registers_with_no_simulator(without_simulator):
    assert without_simulator["imported"] == []
    assert without_simulator["gl_after_import"] is None
    assert without_simulator["ids"] == sorted(ALL_IDS)


def test_constructing_without_the_dependency_names_the_extra_and_touches_nothing(without_simulator):
    assert without_simulator["error_type"] == "ImportError"
    for expected in ("problem.env_id", "assistax_scratchitch", "--extra assistax"):
        assert expected in without_simulator["error"], without_simulator["error"]
    assert without_simulator["gl_after_failed_construction"] is None


def test_the_id_rule_and_the_module_agree():
    from bird import tasks

    seen = 0
    for spec in tasks.index().values():
        if spec.library == "assistax":
            assert tasks._BIRD_ID_RULES["assistax"](spec) == _env_id(spec.env_id)
            seen += 1
    assert seen == len(_EXPECTED) == 5       # the five vendored rows; an EQUALITY
    registry.load_all()
    assert set(ALL_IDS) <= set(registry.names("env"))


def test_the_rows_are_complete_and_name_the_same_things_the_specs_do():
    """Every row carries the same keys, its check/ref/reset are callables and the spec
    names the row's metric -- the pairing `Assistax.__init__` refuses otherwise."""
    from bird import tasks

    keys = {frozenset(r) for r in _TASKS.values()}
    assert len(keys) == 1, "the rows do not share one key set"
    for task, row in _TASKS.items():
        assert callable(row["check"]) and callable(row["ref"]) and callable(row["reset"])
        # tip geoms exactly where the tail carries tip_force_mag: a re-derived row reads the
        # tip force off the tail (0.0 without the field), and the live step must agree
        has_tip_field = any(name == "tip_force_mag" for name, *_rest in row["tail"])
        assert has_tip_field == bool(row["tip_geoms"]), task
        assert set(row["tip_geoms"]) <= set(row["tool_geoms"]), task
        spec = tasks.by_env_id(_env_id(task))
        assert spec is not None, f"no spec backs {_env_id(task)}"
        assert spec.raw["continuous_success"]["raw"]["name"] == row["fitness"]
        assert 0.0 < float(row["threshold"]) < 1.0


def test_a_mistyped_id_is_a_config_error_and_a_real_one_inherits_its_spec():
    with pytest.raises(ConfigError):
        load("eureka", profile="tester", overrides={"problem.env_id": "assistax_scratchich"})
    cfg = load("eureka", profile="tester", overrides={"problem.env_id": "assistax_scratchitch"})
    assert "itch" in str(cfg["problem.task_description"]).lower()
    assert set(_CONSUMER_FORBIDDEN) <= set(cfg["verify.forbidden_symbols"])
    # the subset check above is blind to an omission: the reference alias must be named
    assert {"gt_reward", "reference_reward", "task_metric", "success"} <= set(_CONSUMER_FORBIDDEN)


# --------------------------------------------------------------------------
# with the simulator
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scratch():
    _need_simulator()
    return registry.get("env", "assistax_scratchitch")(None)


@pytest.fixture(scope="module")
def bath():
    _need_simulator()
    return registry.get("env", "assistax_bedbathing")(None)


def _rollout(env, n: int = 40, seed: int = 3):
    rng = np.random.default_rng(seed)
    s = env.reset(np.random.default_rng(1))
    out = []
    for _ in range(n):
        a = rng.uniform(-1.0, 1.0, size=ACTION_DIM)
        s2, _done, _info = env.step(s, a)
        out.append((s, a, s2))
        s = s2
    return out, rng


def _mismatches(env, transitions, rng) -> int:
    return sum(int(not np.array_equal(s2, env.step(s, a)[0]))
               for s, a, s2 in (transitions[i] for i in rng.permutation(len(transitions))))


@pytest.mark.parametrize("task", sorted(_TASKS))
def test_step_is_bitwise_pure_over_shuffled_replay(task):   # every row
    _need_simulator()
    env = registry.get("env", _env_id(task))(None)
    transitions, rng = _rollout(env)
    assert _mismatches(env, transitions, rng) == 0


def test_the_purity_control_setting_ctrl_after_the_forward_breaks_replay(scratch, monkeypatch):
    """The measured failure: with the actuators written after the restoring
    forward, the warmstart the first substep reads depends on the PREVIOUS caller's action
    and shuffled replay drifts by ~1e-14. Without this control a 0-mismatch assertion
    would pass just as happily on a scene whose solver never read the warmstart."""
    from bird.envs.assistax import SUBSTEPS, Assistax

    def leaky_step(self, s, a):
        s = np.asarray(s, dtype=float).ravel()
        u = np.clip(np.asarray(a, dtype=float).ravel(), -1.0, 1.0)
        nq, nv = self.n_q, self.n_v
        slots = s[nq + nv:nq + nv + self.n_slots].copy()
        m, d = self._model, self._data
        self._forward(s[:nq], s[nq:nq + nv])
        d.ctrl[:] = 0.0
        d.ctrl[self._robot_act] = u
        for _ in range(SUBSTEPS):
            self._mj.mj_step(m, d)
        d.qacc_warmstart[:] = 0.0
        self._mj.mj_forward(m, d)
        D = self._derived_now(slots)
        s2 = self._obs(slots, D)
        return s2, False, {"success": False}

    monkeypatch.setattr(Assistax, "_step", leaky_step)
    transitions, rng = _rollout(scratch)
    assert _mismatches(scratch, transitions, rng) > 0


def test_the_observation_is_the_simulator_state_and_restoring_it_reproduces_the_tail(scratch):
    """The derived fields are FUNCTIONS of `(qpos, qvel, slots)`: rebuilding them from a
    stored row must give the row back. If this fails, a stored trajectory and its restore
    disagree and the reference reward is scored on numbers the policy never saw."""
    from bird.envs.assistax import _Derived

    transitions, _ = _rollout(scratch, n=25)
    nq, nv = scratch.n_q, scratch.n_v
    for _s, _a, s2 in transitions[::5]:
        D = _Derived(scratch, s2, forward=True)
        rebuilt = scratch._obs(D.slots, D)
        assert np.array_equal(rebuilt[:nq + nv], s2[:nq + nv])
        assert np.allclose(rebuilt, s2, atol=1e-9, rtol=0.0), np.max(np.abs(rebuilt - s2))
    assert scratch.export_states([transitions[0][0]]) is None, "no snapshot cache: the obs is the state"


def test_nothing_ends_an_episode_early_and_the_human_is_passive(scratch):
    """`done` is the non-finite guard and nothing else, and every humanoid actuator sees
    zero: the two facts the module docstring calls out as departures from upstream."""
    transitions, _ = _rollout(scratch, n=60)
    s = transitions[-1][2]
    _s2, done, info = scratch.step(s, np.ones(ACTION_DIM))
    assert done is False and set(info) == {"success"}
    ctrl = scratch._data.ctrl.copy()
    human = np.setdiff1d(np.arange(ctrl.size), scratch._robot_act)
    assert np.all(ctrl[human] == 0.0)
    assert np.allclose(ctrl[scratch._robot_act], 1.0)


def test_the_itch_lies_on_the_drawn_arm_segments_surface(scratch):
    """The target is drawn in the arm capsule's OWN frame at its radius: the world-frame
    target the tail reports must sit at exactly the capsule's radius off its axis."""
    from bird.envs.assistax import _Derived

    m = scratch._model
    for seed in range(8):
        s = scratch.reset(np.random.default_rng(seed))
        D = _Derived(scratch, s, forward=True)
        geom = scratch._larm_geom if D.slots[0] > 0.5 else scratch._uarm_geom
        R = scratch._data.geom_xmat[geom].reshape(3, 3)
        rel = D.target_pos - scratch._data.geom_xpos[geom]
        along = float(rel @ R[:, 2])
        radial = float(np.linalg.norm(rel - along * R[:, 2]))
        assert abs(radial - float(m.geom_size[geom][0])) < 1e-9
        assert abs(along) <= float(m.geom_size[geom][1]) + 1e-9


def test_a_wiped_point_stays_wiped_and_the_metric_is_the_wiped_count(bath):
    """The flags are the episode's memory: monotone under `step`, and `task_metric` reads
    the LAST state's count over 52 -- a random 60-step episode wipes nothing."""
    c0 = bath.n_q + bath.n_v
    transitions, _ = _rollout(bath, n=60)
    prev = transitions[0][0][c0:c0 + N_WIPE_POINTS]
    for _s, _a, s2 in transitions:
        cur = s2[c0:c0 + N_WIPE_POINTS]
        assert np.all(cur <= prev + 1e-12), "a wiped point came back"
        prev = cur
    states = np.array([transitions[0][0]] + [t[2] for t in transitions])
    assert bath.task_metric(states) == float(N_WIPE_POINTS - int((prev > 0.5).sum())) / N_WIPE_POINTS
    # a state with every flag cleared scores 1.0 whatever the rest of the episode did. Built
    # the way `random_state` builds a re-flagged pool entry -- the tail RECOMPUTED for the
    # new flags -- not by zeroing the flags under a stale tail, which would assert nothing
    # about the sentinel (`test_bedbathing_all_wiped_is_a_finite_sentinel_state` does).
    done_state = _with_flags(bath, states[-1], np.zeros(N_WIPE_POINTS))
    assert bath.task_metric(np.vstack([states[0], done_state])) == 1.0
    assert done_state[bath._tail_index()["dist_tool_nearest_unwiped"]] == WIPED_OUT_DIST


def test_task_metric_is_nan_for_garbage_and_a_fraction_otherwise(scratch):
    assert np.isnan(scratch.task_metric(np.zeros((1, scratch.obs_dim))))
    assert np.isnan(scratch.task_metric(np.full((3, scratch.obs_dim), np.inf)))
    transitions, _ = _rollout(scratch, n=20)
    v = scratch.task_metric(np.array([transitions[0][0]] + [t[2] for t in transitions]))
    assert 0.0 <= v <= 1.0


def test_the_reference_reward_is_a_pure_finite_function_of_the_row(scratch):
    transitions, _ = _rollout(scratch, n=30)
    for s, a, s2 in transitions[::6]:
        r1 = scratch.reference_reward(s2, a)
        _ = scratch.step(transitions[0][0], transitions[0][1])       # disturb the simulator
        r2 = scratch.reference_reward(s2, a)
        assert math.isfinite(r1) and r1 == r2
        assert scratch.gt_reward(s, a, s2) == r1


def test_the_committed_specs_are_what_the_generator_emits():
    """`scripts/gen_assistax_specs.py --check` is the drift gate ("edit the generator,
    never the YAML"); a byte-equality between the committed text and the generator's."""
    import subprocess
    import sys
    from pathlib import Path

    _need_simulator()
    repo = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, str(repo / "scripts" / "gen_assistax_specs.py"), "--check"],
                          cwd=repo, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, (
        "the committed tasks/assistax_*/shared_spec.yaml differ from what "
        "scripts/gen_assistax_specs.py emits; regenerate rather than editing the YAML\n"
        + proc.stdout[-2000:] + proc.stderr[-2000:])


def test_the_vendored_assets_carry_their_provenance():
    """Every file under the vendored tree is named in PROVENANCE.json with the digest it
    has -- the half of `vendor_assistax_assets.py --check` that needs no network."""
    import hashlib
    import json
    import re
    from pathlib import Path

    from bird.envs.assistax import ASSETS

    prov = json.loads((ASSETS / "PROVENANCE.json").read_text())
    expected = {rel: rec["sha256"] for rel, rec in prov["xml"].items()}
    expected.update({f"meshes/{n}.stl": rec["sha256"] for n, rec in prov["meshes"].items()})
    on_disk = {str(p.relative_to(ASSETS)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in ASSETS.rglob("*") if p.is_file() and p.name != "PROVENANCE.json"}
    # every file under the tree is vendored from upstream and recorded: no unrecorded file
    assert on_disk == expected
    assert all(row["upstream"] is not None for row in _TASKS.values())
    assert prov["upstream"]["repo"] == "assistive-autonomy/assistax"
    assert all(rec["triangles"] <= rec["triangles_upstream"] for rec in prov["meshes"].values())
    assert all(Path(ASSETS / rel).stat().st_size < 5 * 1024 * 1024 for rel in on_disk)
    # the physics half of the manifest: a mesh a colliding geom is fitted from
    # is never decimated, and the compiled scenes match upstream's on everything the
    # physics reads. The five hand/finger meshes are the measured set; a sixth appearing
    # here means a scene changed, a mesh vanishing means the rule stopped seeing it.
    assert sorted(prov["protected"]) == ["finger_0", "hand_3", "link6_16", "link7_0", "link7_7"]
    for name, rec in prov["meshes"].items():
        assert rec["protected"] == (name in prov["protected"]), name
        if rec["protected"]:
            assert rec["triangles"] == rec["triangles_upstream"], f"{name}: protected yet decimated"
    assert set(prov["fidelity"]) == {"wheelchair_scene.xml", "bed_scene.xml", "bed_scene_armmanip.xml",
                                     "feeding_scene.xml", "teethbrushing_scene.xml"}
    for scene, rec in prov["fidelity"].items():
        for key in ("colliding_geom_size_max_abs_diff", "colliding_geom_pos_max_abs_diff",
                    "colliding_geom_quat_max_abs_diff", "moving_body_mass_max_abs_diff",
                    "moving_body_inertia_max_abs_diff", "moving_body_ipos_max_abs_diff"):
            assert rec[key] == 0.0, (scene, key, rec[key])
        assert rec["colliding_geoms"] > 20 and rec["moving_bodies"] > 20, scene


@pytest.mark.parametrize("task", sorted(_TASKS))
def test_rendering_the_same_state_twice_is_identical(task, gl):
    _need_simulator()
    env = registry.get("env", _env_id(task))(None)
    s = env.reset(np.random.default_rng(0))
    a = env.render(s)
    b = env.render(s)
    assert a.shape == (env.render_height, env.render_width, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert float(a.std()) > 1.0, "a uniform frame is a wedged renderer, not a scene"


# --------------------------------------------------------------------------
# the render panel's task line (sim-free)
# --------------------------------------------------------------------------


def test_every_task_line_is_legible_in_the_hud_font():
    """The panel `render` draws (`_legend_lines` -> `_task_lines`): a non-empty list of
    non-empty upper-case strings, each at most `_LEGEND_COLS`, drawn only from the
    characters `bird/envs/hud.py`'s sheet has a glyph for -- anything else renders as a
    tofu box, which is visible on a frame and invisible to every other test. Pinned
    WITHOUT the simulator, like the registration above."""
    ok = set(hud._SHEET_CHARS)
    for task in _TASKS:
        lines = _task_lines(task)
        assert isinstance(lines, list) and lines, (task, lines)
        for line in lines:
            assert isinstance(line, str) and line.strip(), (task, line)
            assert line == line.upper(), f"{task}: not upper-case: {line!r}"
            assert len(line) <= _LEGEND_COLS, f"{task}: {len(line)} chars: {line!r}"
            stray = sorted(set(line) - ok)
            assert not stray, f"{task}: no glyph for {stray} in {line!r}"


# --------------------------------------------------------------------------
# with the simulator, IN CONTACT
# --------------------------------------------------------------------------
#
# A greedy descent: hold the current 7-D action and each step try +/- `_DELTA` on every
# dimension, keeping the candidate whose ARRIVING state puts the tool closest to a point
# `_DEPTH[task]` metres INSIDE the human surface at the target (the itch, the nearest
# unwiped point, the forearm hook point, the mouth). Every human geom is convex, so a point
# on the segment from a surface point to an interior point is inside it and the tool can
# only get there by penetrating -- which is to say, by touching. Aiming AT the surface
# point does not work: the scratcher tip hovered 1 cm off the itch for 384 of 400 steps
# (measured; the descent cannot go past zero distance and the action grid is coarse) and
# the sole contact came from the body settling under the tool.
#
# Measured at seed 0, first contact at step: scratchitch 167,
# bedbathing 40, armmanipulation 31, feeding 131, teethbrushing 112. The depths are per
# task because the tools differ (a 1 cm scratcher tip against a 5 cm wiper pad) and were
# chosen for coverage: at 0.01 the scratcher meets the check's every clause on all 31 of its
# contact states; at 0.02 the spoon comes within 4 cm of the mouth; at 0 the brush's
# bristle FACE carries the force (its head box does at 0.01, and `tip_force_mag` reads 0).
_DELTA = 0.05
_DEPTH = {"scratchitch": 0.01, "bedbathing": 0.03, "armmanipulation": 0.03, "feeding": 0.02,
          "teethbrushing": 0.0}
#: (steps before the fixture gives up, steps to keep going after first contact). Bedbathing
#: runs on with no early stop: its nearest-unwiped target moves as points are wiped, so the
#: same descent sweeps the arm -- 26 of 52 wiped by step 259, plateauing at 30 (the rest
#: lie against the mattress) -- and the trajectory carries both sides of its threshold.
_BUDGET = {"bedbathing": (700, None)}
_DEFAULT_BUDGET = (400, 40)
#: The tail field naming the target the descent aims at (its x; y, z follow).
_TARGET_FIELD = {"scratchitch": "target_x", "bedbathing": "nearest_unwiped_x",
                 "armmanipulation": "hook_target_x", "feeding": "mouth_x",
                 "teethbrushing": "mouth_x"}
_MJ_CAPSULE = 3


def _tail(env, s, name: str, n: int = 1):
    i = env._tail_index()[name]
    return np.asarray(s[i:i + n], dtype=float) if n > 1 else float(s[i])


def _tool_force(env, s) -> np.ndarray:
    return _tail(env, s, "tool_fx", 3)


def _in_contact(env, s) -> bool:
    return bool(np.linalg.norm(_tool_force(env, s)) > 0.0)


def _hold_action(env, s) -> np.ndarray:
    """The action that holds the Panda where `s` has it: the position servo's equilibrium
    `gain * u + bias1 * q = 0` per actuator (so u = q / pi on this model)."""
    m = env._model
    q = np.asarray(s, dtype=float)[env._robot_qadr]
    gain = m.actuator_gainprm[env._robot_act, 0]
    bias1 = m.actuator_biasprm[env._robot_act, 1]
    return np.clip(-bias1 * q / gain, -1.0, 1.0)


def _human_geoms(env) -> list:
    mj = env._mj
    return [int(mj.mj_name2id(env._model, mj.mjtObj.mjOBJ_GEOM, g)) for g in env._row["human_geoms"]]


def _spine_point(env, geom: int, p: np.ndarray) -> np.ndarray:
    """The point of the geom's SPINE closest to `p`: the axis segment of a capsule, the
    centre of a sphere. A capsule is the set of points within its radius of that segment
    and a sphere of its centre, so the direction from a surface point to its closest spine
    point is the exact inward normal there -- friction, which acts in the tangent plane,
    cannot flip a dot product with it. The geom CENTRE is not that direction on a capsule:
    from a contact near one end it points along the arm, and the force the wiper applies
    while pressing the arm into the mattress dotted with it came out NEGATIVE on 28 of 381
    bedbathing contact states (measured) -- a heuristic, not a test."""
    m, d = env._model, env._data
    c = d.geom_xpos[geom]
    if int(m.geom_type[geom]) == _MJ_CAPSULE:
        axis = d.geom_xmat[geom].reshape(3, 3)[:, 2]
        half = float(m.geom_size[geom][1])
        t = float(np.clip(np.dot(p - c, axis), -half, half))
        return c + t * axis
    return c.copy()


def _outward(env, geoms: list, p: np.ndarray) -> np.ndarray:
    """Unit outward normal of the human surface nearest `p`, read off the simulator's
    current pose (the caller has it in the state `p` came from)."""
    best = None
    for g in geoms:
        sp = _spine_point(env, g, p)
        dd = float(np.linalg.norm(p - sp))
        if best is None or dd < best[0]:
            best = (dd, sp)
    v = p - best[1]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def _descend(env, task: str, seed: int = 0):
    """Greedy descent (module comment above). Returns `(transitions, first_contact)`:
    `(s, a, s2, info)` per step and the index of the first arriving state whose tail
    carries a non-zero tool force -- None if the budget ran out first."""
    max_steps, after = _BUDGET.get(task, _DEFAULT_BUDGET)
    depth = _DEPTH[task]
    ti = env._tail_index()
    tx = ti[_TARGET_FIELD[task]]
    geoms = _human_geoms(env)
    s = env.reset(np.random.default_rng(seed))
    a = _hold_action(env, s)
    out, first = [], None

    def objective(s2):
        # the simulator IS in s2 here (the step just produced it), so the spine read is s2's
        tool = s2[ti["tool_x"]:ti["tool_x"] + 3]
        tgt = s2[tx:tx + 3]
        return float(np.linalg.norm(tgt - depth * _outward(env, geoms, tgt) - tool))

    for k in range(max_steps):
        cands = [a.copy()]
        for i in range(ACTION_DIM):
            for sgn in (1.0, -1.0):
                c = a.copy()
                c[i] = float(np.clip(c[i] + sgn * _DELTA, -1.0, 1.0))
                cands.append(c)
        best = None
        for c in cands:
            s2, _done, _info = env.step(s, c)
            v = objective(s2)
            if best is None or v < best[0]:
                best = (v, c)
        a = best[1]
        s2, done, info = env.step(s, a)              # leaves the simulator in the chosen state
        assert not done, f"{task}: the descent hit a non-finite state at step {k}"
        out.append((s, a, s2, info))
        if first is None and _in_contact(env, s2):
            first = k
        s = s2
        if first is not None and after is not None and k - first >= after:
            break
    return out, first


@pytest.fixture(scope="module")
def contact() -> dict:
    """task -> {env, transitions, contact, first_contact}. FAILS -- never skips -- if a
    task's descent does not touch the human within its budget: every test below reads
    these states, and a fixture that quietly produced none would pass them all."""
    _need_simulator()
    out = {}
    for task in sorted(_TASKS):
        env = registry.get("env", _env_id(task))(None)
        transitions, first = _descend(env, task)
        assert first is not None, (
            f"{task}: the greedy descent made no contact with the human in "
            f"{_BUDGET.get(task, _DEFAULT_BUDGET)[0]} steps (min gated distance "
            f"{min(_tail(env, t[2], 'dist') for t in transitions):.4f} m)")
        touching = [t for t in transitions if _in_contact(env, t[2])]
        out[task] = {"env": env, "transitions": transitions, "contact": touching,
                     "first_contact": first}
    return out


def _with_flags(env, s, flags) -> np.ndarray:
    """`s` with bedbathing's wipe flags replaced and the tail RECOMPUTED for them, exactly
    as `random_state` builds a re-flagged pool entry (`_obs` over `_Derived(forward=True)`
    of the row with the new slots). The contact force is the row's own."""
    c0 = env.n_q + env.n_v
    probe = np.asarray(s, dtype=float).copy()
    probe[c0:c0 + N_WIPE_POINTS] = np.asarray(flags, dtype=float)
    slots = probe[c0:c0 + env.n_slots].copy()
    return env._obs(slots, _Derived(env, probe, forward=True))


@pytest.mark.parametrize("task", sorted(_TASKS))
def test_the_greedy_descent_reaches_contact_and_holds_it(contact, task):
    """The fixture's own claim, stated per task so a regression in ONE task's contact
    (a re-vendored mesh moving a fitted collision box, a changed timestep) names it."""
    c = contact[task]
    max_steps, after = _BUDGET.get(task, _DEFAULT_BUDGET)
    assert c["first_contact"] < max_steps
    assert len(c["contact"]) >= 3, f"{task}: only {len(c['contact'])} contact states"
    forces = [float(np.linalg.norm(_tool_force(c["env"], t[2]))) for t in c["contact"]]
    assert all(math.isfinite(f) and f > 0.0 for f in forces)


@pytest.mark.parametrize("task", sorted(_TASKS))
def test_the_tail_force_is_the_live_contact_force_and_points_into_the_human(contact, task):
    """Two claims the tail's `tool_fx..fz` docstring makes -- "the force the TOOL applies to
    the HUMAN, world frame" -- neither of which any random state could test.

    (1) Sign and frame, convention-free: at every live tool-human contact the inward normal
    is the direction from the contact point to the closest point of the human geom's spine
    (`_spine_point`); the tail force must have a positive component along the sum of those
    directions. That statement uses positions only. It is checked beside the frame-based
    one (the contact frame's normal points geom1 -> geom2, oriented tool -> human), so a
    force rotated by the wrong frame or signed for the wrong geom fails one or both. With
    the sign flipped -- the force on the TOOL -- every contact state fails.
    (2) The tail is what the live solve computed: re-stepping `(s, a)` and reading
    `_contact_forces` gives the tail's vector bitwise, and the tip magnitude on the head
    tasks. Without this the tail could carry a rounded, stale or re-solved number and the
    `_Derived` read-back-off-the-row path would faithfully preserve it."""
    env = contact[task]["env"]
    d = env._data
    nq, nv = env.n_q, env.n_v
    ti = env._tail_index()
    for s, a, s2, _info in contact[task]["contact"]:
        f = _tool_force(env, s2)
        env.step(s, a)
        live, tip = env._contact_forces(d)
        assert np.array_equal(live, f)
        if "tip_force_mag" in ti:
            assert float(np.linalg.norm(tip)) == s2[ti["tip_force_mag"]]
        else:                                        # no tip field: no tip geoms, zero live too
            assert not tip.any() and _Derived(env, s2, forward=True).tip_force_mag == 0.0
        env._forward(s2[:nq], s2[nq:nq + nv])
        into_geom, into_normal, pairs = np.zeros(3), np.zeros(3), 0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in env._tool_geoms and g2 in env._human_geoms:
                human, sign = g2, 1.0
            elif g2 in env._tool_geoms and g1 in env._human_geoms:
                human, sign = g1, -1.0
            else:
                continue
            pairs += 1
            pos = np.asarray(c.pos, dtype=float)
            into_normal += sign * np.asarray(c.frame, dtype=float)[:3]
            v = _spine_point(env, human, pos) - pos
            into_geom += v / np.linalg.norm(v)
        assert pairs >= 1, f"{task}: a {np.linalg.norm(f):.1f} N tail force with no live tool-human contact"
        assert float(np.dot(f, into_geom)) > 0.0, (task, f, into_geom)
        assert float(np.dot(f, into_normal)) > 0.0, (task, f, into_normal)


# -- upstream's rewards, transcribed independently from the tail ---------------------
#
# Each function below is a second reading of the upstream `step` (upstream assistax at
# the vendored commit: scratchitch.py:203,229-244; bedbathing.py:229-250;
# armmanipulation.py:218,244-254; feeding.py:181,204-285; teethbrushing.py:181,204-253
# and get_brush_reward at 321) written from the TAIL FIELDS, with the adapter's four
# declared substitutions and nothing else: the tool's instantaneous speed for the
# finite difference, the tail's summed tool-human force for the fixed MJX contact rows,
# the 7-D action clipped to [-1, 1] in the control cost, bedbathing's +3 per new point
# dropped. The rotation goes through a rotation matrix, not the adapter's brax-style
# `_quat_rotate`, so the two agree only if both are right.

def _rotate(q, v) -> np.ndarray:
    w, x, y, z = (float(c) for c in q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, dtype=float)


def _bz(x: float, target: float) -> float:
    """upstream's `x/t * exp(-x/t)`."""
    return (x / target) * math.exp(-x / target)


def _ctrl_cost(a) -> float:
    u = np.clip(np.asarray(a, dtype=float).ravel(), -1.0, 1.0)
    return float(np.sum(np.square(u)))


def _up_scratchitch(env, s, a) -> float:
    dist = _tail(env, s, "dist_tool_target")
    speed = float(np.linalg.norm(_tail(env, s, "tool_vx", 3)))
    force = float(np.linalg.norm(_tail(env, s, "tool_fx", 3)))
    r_dist = math.exp(-dist ** 2 / 0.1)                              # dist_scale 0.1
    r_scratching = float(abs(dist) < 0.1) * _bz(speed, 0.1) * _bz(force, 3.0)
    return 1.0 * r_dist + 1e-6 * (-_ctrl_cost(a)) + 4.0 * r_scratching


def _up_bedbathing(env, s, a) -> float:
    if _tail(env, s, "n_wiped") >= N_WIPE_POINTS:
        return 0.0                              # every distance masked to inf: exp(-inf) = 0
    d = _tail(env, s, "dist_tool_nearest_unwiped")
    return 1.0 * math.exp(-d ** 2 / 0.1) + 0.0 * (-_ctrl_cost(a))   # ctrl weight 0; +3/point dropped


def _up_armmanipulation(env, s, a) -> float:
    hook = _tail(env, s, "dist_tool_hook_target")
    waist = _tail(env, s, "dist_forearm_waist_target")
    rot = _tail(env, s, "hook_rot_err")
    r_hook = 1.0 - math.tanh(hook / 0.1)
    r_waist = math.exp(-waist ** 2 / 0.1)
    return 10.0 * r_waist + 1.0 * r_hook + 1e-6 * (-_ctrl_cost(a)) + rot * 0.1


def _up_feeding(env, s, a) -> float:
    tool = _tail(env, s, "tool_x", 3)
    mouth = _tail(env, s, "mouth_x", 3)
    q = _tail(env, s, "tool_qw", 4)
    speed = float(np.linalg.norm(_tail(env, s, "tool_vx", 3)))
    tip = _tail(env, s, "tip_force_mag")
    distance = float(np.linalg.norm(tool - mouth))
    r_dist = math.exp(-3.0 * distance)
    to_mouth = (mouth - tool) / (np.linalg.norm(mouth - tool) + 1e-6)
    spoon_up = _rotate(q, [-1.0, 0.0, 0.0])
    r_aim = float(np.dot(_rotate(q, [0.0, 0.0, 1.0]), to_mouth))
    proximity = math.exp(-150.0 * distance ** 2)
    target_vec = (1.0 - proximity) * np.array([0.0, 0.0, 1.0]) + proximity * to_mouth
    target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-6)
    r_pour = float(np.dot(target_vec, spoon_up))
    r_orientation = 1.0 * r_pour + (0.3 + 0.7 * proximity) * r_aim
    braking = math.exp(-150.0 * distance ** 2)
    target_speed = (1.0 - braking) * 0.20 + braking * 0.0
    r_velocity = math.exp(-((speed - target_speed) ** 2) / 0.1 ** 2)
    r_force = _bz(tip, 1.0)
    return 2.0 * r_dist + 1.0 * (r_orientation + r_velocity + r_force) + 1e-6 * (-_ctrl_cost(a))


def _up_teethbrushing(env, s, a) -> float:
    tool = _tail(env, s, "tool_x", 3)
    mouth = _tail(env, s, "mouth_x", 3)
    q = _tail(env, s, "tool_qw", 4)
    vel = _tail(env, s, "tool_vx", 3)
    tip = _tail(env, s, "tip_force_mag")
    distance = float(np.linalg.norm(tool - mouth))
    r_dist = math.exp(-3.0 * distance)
    to_mouth = (mouth - tool) / (np.linalg.norm(mouth - tool) + 1e-6)
    r_align_surface = float(np.dot(to_mouth, _rotate(q, [-1.0, 0.0, 0.0])))
    r_tilt = float(np.dot(to_mouth, _rotate(q, [0.0, -1.0, 0.0])))
    r_align_weighted = (r_align_surface + r_tilt) / 2.0 * r_dist
    # get_brush_reward: gate on the bristle-face force, tangential speed against the
    # mouth-to-brush normal, peak at 0.05 m/s (upstream's r_force is computed and unused)
    active = 1.0 if tip > 0.1 else 0.0
    normal = (tool - mouth) / (distance + 1e-6)
    v_tan = vel - float(np.dot(vel, normal)) * normal
    r_brush = active * _bz(float(np.linalg.norm(v_tan)), 0.05)
    return 2.0 * r_dist + 1.0 * (r_brush + r_align_weighted) + 1e-6 * (-_ctrl_cost(a))


_UPSTREAM = {"scratchitch": _up_scratchitch, "bedbathing": _up_bedbathing,
             "armmanipulation": _up_armmanipulation, "feeding": _up_feeding,
             "teethbrushing": _up_teethbrushing}


def _force_term_live(env, task: str, s) -> bool:
    """Whether the task's force-dependent term is non-zero on `s` (the three tasks that
    have one): the scratching product, the spoon's `r_force`, the brush's gated `r_brush`."""
    if task == "scratchitch":
        return (_tail(env, s, "dist_tool_target") < 0.1
                and np.linalg.norm(_tail(env, s, "tool_fx", 3)) > 0
                and np.linalg.norm(_tail(env, s, "tool_vx", 3)) > 0)
    if task == "feeding":
        return _tail(env, s, "tip_force_mag") > 0
    if task == "teethbrushing":
        return (_tail(env, s, "tip_force_mag") > 0.1
                and _tail(env, s, "brush_tangential_speed") > 0)
    return False


@pytest.mark.parametrize("task", sorted(_TASKS))
def test_the_reference_reward_is_upstreams_reward_transcribed(contact, task):
    """`reference_reward(s2, a)` equals an independent transcription of upstream's reward
    to 1e-9 on contact states AND random states -- including, on the three tasks that have
    one, states where the force term is non-zero, which no random state reaches. The
    purity test alone asserts only that the reward is finite and repeatable: a
    transcription with the wrong constant, a missing gate or a term dropped in the force
    path passes it. Also pins the declared clip: an action outside [-1, 1] costs what its
    clipped value costs (and the cost term is live -- ones and zeros differ -- on every
    task but bedbathing, whose upstream ctrl weight is 0)."""
    env = contact[task]["env"]
    up = _UPSTREAM[task]
    states = [(s, a, s2) for s, a, s2, _i in contact[task]["transitions"]]
    states += _rollout(env, n=30, seed=11)[0]
    n_force = 0
    for s, a, s2 in states:
        got, want = env.reference_reward(s2, a), up(env, s2, a)
        assert math.isfinite(got) and abs(got - want) <= 1e-9, (task, got, want)
        assert env.gt_reward(s, a, s2) == got
        n_force += int(_force_term_live(env, task, s2))
    if task in ("scratchitch", "feeding", "teethbrushing"):
        assert n_force > 0, f"{task}: no state exercised the force term"
    s2 = contact[task]["contact"][0][2]
    ones = np.ones(ACTION_DIM)
    assert env.reference_reward(s2, 3.0 * ones) == env.reference_reward(s2, ones)
    assert abs(env.reference_reward(s2, 3.0 * ones) - up(env, s2, 3.0 * ones)) <= 1e-9
    if task != "bedbathing":
        assert env.reference_reward(s2, ones) != env.reference_reward(s2, np.zeros(ACTION_DIM))


# -- a stored row's force is the row's, not the leftover ctrl's ------------------------------

def _rederive(env, s2, a):
    D = _Derived(env, s2, forward=True)
    return (env.reference_reward(s2, a), bool(env._row["check"](env, D)), D.tool_force.copy(),
            D.tool_force_mag, D.tip_force_mag, tuple(env._legend_lines(D)))


@pytest.mark.parametrize("task", sorted(_TASKS))
def test_re_deriving_a_stored_row_does_not_read_the_leftover_ctrl(contact, task):
    """If `_Derived(forward=True)` re-solves the contact forces under whatever `d.ctrl` the
    previous caller left, `reference_reward(s2, a)`, `task_metric` and the render legend on
    the SAME row depend on call order -- a 210 N contact reads back as 51 N or 0 N. All of
    them, re-derived
    after (a) stepping another state under a saturated action, (b) `d.ctrl[:] = 0`,
    (c) `d.ctrl[:] = 1`, are bit-identical to the first reading, and the force they read is
    the row's own `tool_fx..fz` exactly."""
    env = contact[task]["env"]
    d = env._data
    other = contact[task]["transitions"][0][0]
    for s, a, s2, _info in contact[task]["contact"][:12]:
        base = _rederive(env, s2, a)
        env.step(other, np.ones(ACTION_DIM))
        after_step = _rederive(env, s2, a)
        d.ctrl[:] = 0.0
        after_zero = _rederive(env, s2, a)
        d.ctrl[:] = 1.0
        after_one = _rederive(env, s2, a)
        for got in (after_step, after_zero, after_one):
            assert got[0] == base[0] and got[1] == base[1]
            assert np.array_equal(got[2], base[2])
            assert got[3] == base[3] and got[4] == base[4] and got[5] == base[5]
        assert np.array_equal(base[2], _tool_force(env, s2))
        assert base[3] > 0.0


@pytest.mark.parametrize("task", sorted(_TASKS))
def test_the_control_a_restored_solve_does_depend_on_the_leftover_ctrl(contact, task):
    """The unsafe path, shown to be order-dependent on this fixture's states: restore
    the row with `_forward` and re-solve `_contact_forces` under `d.ctrl` all 0 and all 1.
    The two must DIFFER on at least one contact state, or the invariance test above is
    passing on states where the solve never read the actuation and would pass with the bug
    reinstated. Under the step's OWN action the restored solve reproduces the tail bitwise
    -- the tail is that solve, which is why reading it back off the row is the fix and not
    an approximation."""
    env = contact[task]["env"]
    d = env._data
    nq, nv = env.n_q, env.n_v
    differing = 0
    for s, a, s2, _info in contact[task]["contact"]:
        q, v = s2[:nq], s2[nq:nq + nv]
        d.ctrl[:] = 0.0
        env._forward(q, v)
        f_zero = env._contact_forces(d)[0]
        d.ctrl[:] = 1.0
        env._forward(q, v)
        f_one = env._contact_forces(d)[0]
        d.ctrl[:] = 0.0
        d.ctrl[env._robot_act] = np.clip(a, -1.0, 1.0)
        env._forward(q, v)
        f_own = env._contact_forces(d)[0]
        assert np.array_equal(f_own, _tool_force(env, s2))
        differing += int(not np.array_equal(f_zero, f_one))
    assert differing > 0, f"{task}: the restored solve ignored ctrl on all {len(contact[task]['contact'])} contact states"


# -- bedbathing: the marking rule, its threshold, the all-wiped sentinel -------------------

def test_bedbathing_marks_a_point_only_within_the_threshold_and_only_in_contact(contact):
    """Upstream's `_update_contact_vector`, applied to the arriving state: a point is
    marked when the wiper centre is within 0.1 m of it AND the summed tool-human force is
    non-zero; a marked point stays marked; nothing else changes. Asserted as the exact
    flag vector on every step of the descent, which carries both halves of the rule:
    steps that mark points (the descent sweeps the arm), and steps where the wiper hovers
    inside 0.1 m of an unwiped point with ZERO force -- the approach -- on which nothing
    may be marked. Without the second the rule `close -> wiped` passes."""
    env = contact["bedbathing"]["env"]
    c0 = env.n_q + env.n_v
    marking_steps = hovering_steps = 0
    for s, _a, s2, _info in contact["bedbathing"]["transitions"]:
        before = s[c0:c0 + N_WIPE_POINTS]
        after = s2[c0:c0 + N_WIPE_POINTS]
        D = _Derived(env, s2, forward=True)                 # wipe_dists at the ARRIVING pose
        force = float(np.linalg.norm(_tool_force(env, s2)))
        close = D.wipe_dists < WIPE_THRESHOLD
        want = np.where(close & (force > 0.0), 0.0, before)
        assert np.array_equal(after, want)
        assert np.all(after <= before)
        newly = (before > 0.5) & (after < 0.5)
        if newly.any():
            marking_steps += 1
            assert force > 0.0 and bool(np.all(D.wipe_dists[newly] < WIPE_THRESHOLD))
        if force == 0.0 and bool(np.any(close & (before > 0.5))):
            hovering_steps += 1
            assert np.array_equal(after, before)
    assert marking_steps > 0, "the descent never wiped a point"
    assert hovering_steps > 0, "no step hovered inside 0.1 m without touching"


def test_bedbathing_success_flag_flips_at_26_of_52(contact):
    """`_step`'s `info["success"]` is `n_wiped / 52 >= success_threshold` (0.5), the same
    threshold `success()` applies -- 26 points, not 52 (a 52-point flag would make
    `train.bc_prior.accept: success` reject a 26-51-point teacher). Built from a reset state,
    where the wiper is far from the arm so the step marks nothing and the count is exactly
    the one constructed; and observed on the descent, where the flag first turns True on
    the step the count reaches 26."""
    env = contact["bedbathing"]["env"]
    assert env.success_threshold == 0.5
    s0 = contact["bedbathing"]["transitions"][0][0]
    hold = _hold_action(env, s0)
    for n_zero, want in ((26, True), (25, False), (52, True), (0, False)):
        flags = np.ones(N_WIPE_POINTS)
        flags[:n_zero] = 0.0
        state = _with_flags(env, s0, flags)
        assert _tail(env, state, "n_wiped") == n_zero
        s2, done, info = env._step(state, hold)
        assert done is False
        assert _tail(env, s2, "n_wiped") == n_zero, "the step marked a point from the reset pose"
        assert info["success"] is want
    flips = [(int(_tail(env, t[2], "n_wiped")), bool(t[3]["success"]))
             for t in contact["bedbathing"]["transitions"]]
    assert all(flag == (n >= 26) for n, flag in flips)
    assert {flag for _n, flag in flips} == {False, True}


def test_bedbathing_all_wiped_is_a_finite_sentinel_state(contact):
    """Once every point is wiped the distances are +inf internally and the observation must
    stay finite so the non-finite guard does not end a FINISHED episode: from a state with
    all 52 flags cleared (tail recomputed, as `random_state` does), `_step` returns
    `done=False`, `dist_tool_nearest_unwiped` reads the 10.0 sentinel, the nearest-unwiped
    position is the tool's own, the reference reward is 0.0, `info["success"]` is True and
    a two-state trajectory scores `task_metric` 1.0. The base row is a real CONTACT state
    (wiper on the arm), so the step's marking rule runs and has nothing left to mark. (The
    descent itself plateaus at 30 wiped -- the rest lie on the arm's underside against the
    mattress -- so no naturally all-wiped row exists to hold to the same facts.)"""
    env = contact["bedbathing"]["env"]
    ti = env._tail_index()

    def sentinel_facts(state):
        assert bool(np.isfinite(state).all())
        assert _tail(env, state, "n_wiped") == N_WIPE_POINTS
        assert _tail(env, state, "dist_tool_nearest_unwiped") == WIPED_OUT_DIST
        assert np.array_equal(_tail(env, state, "nearest_unwiped_x", 3), _tail(env, state, "tool_x", 3))
        assert env.reference_reward(state) == 0.0
        assert _Derived(env, state, forward=True).dist == math.inf

    base = contact["bedbathing"]["contact"][0][2]                 # in contact, on the arm
    state = _with_flags(env, base, np.zeros(N_WIPE_POINTS))
    sentinel_facts(state)
    s2, done, info = env._step(state, _hold_action(env, state))
    assert done is False and info["success"] is True
    sentinel_facts(s2)
    assert env.task_metric(np.vstack([state, s2])) == 1.0
    assert env.success(np.vstack([state, s2])) is True
    assert ti["dist"] == ti["dist_tool_nearest_unwiped"]      # the gated distance IS this one


def test_random_state_varies_the_wipe_flags_and_each_entry_is_self_consistent(contact):
    """`random_state`'s pool re-draws bedbathing's flags on half its entries: over 64 draws
    `n_wiped` is not constant and some are > 0 (a random walk never touches the arm, so a
    pool of 52 ones cannot tell a reward that pays for wiping from one that does not), and
    every entry is what `_step` would have emitted with those flags -- recomputing its tail
    from its own `(qpos, qvel, slots)` reproduces the row BITWISE (the stepped rows restore
    bitwise too: 0 mismatches over 797 stepped rows across the five tasks, measured)."""
    env = contact["bedbathing"]["env"]
    rng = np.random.default_rng(5)
    draws = [env.random_state(rng) for _ in range(64)]
    n_wiped = [int(_tail(env, x, "n_wiped")) for x in draws]
    assert len(set(n_wiped)) > 1 and any(n > 0 for n in n_wiped) and any(n == 0 for n in n_wiped)
    c0 = env.n_q + env.n_v
    for x in draws:
        assert x.shape == (env.obs_dim,) and bool(np.isfinite(x).all())
        assert int(N_WIPE_POINTS - (x[c0:c0 + N_WIPE_POINTS] > 0.5).sum()) == int(_tail(env, x, "n_wiped"))
        D = _Derived(env, x, forward=True)
        assert np.array_equal(env._obs(D.slots, D), x)


# -- the success checks are the spec's words -------------------------------------------

def _spec_clause(env, task: str, s) -> bool:
    """`scripts/gen_assistax_specs.py::_FITNESS_EXPR`, read literally off the tail."""
    if task == "scratchitch":
        return bool(_tail(env, s, "dist_tool_target") < 0.05
                    and np.linalg.norm(_tail(env, s, "tool_fx", 3)) >= 0.5
                    and np.linalg.norm(_tail(env, s, "tool_vx", 3)) >= 0.02)
    if task == "bedbathing":
        return bool(_tail(env, s, "n_wiped") / 52 >= 0.5)
    if task == "armmanipulation":
        return bool(_tail(env, s, "dist_forearm_waist_target") < 0.08)
    if task == "feeding":
        return bool(_tail(env, s, "dist_tool_mouth") < 0.04 and _tail(env, s, "tip_force_mag") < 5.0)
    if task == "teethbrushing":
        return bool(_tail(env, s, "dist_tool_mouth") < 0.03
                    and _tail(env, s, "tip_force_mag") >= 0.1
                    and _tail(env, s, "brush_tangential_speed") >= 0.01)
    raise KeyError(task)


@pytest.mark.parametrize("task", sorted(_TASKS))
def test_the_success_check_is_the_specs_clause_on_the_tail(contact, task):
    """On every state of the descent -- contact and approach -- the row's check re-derived
    from the row equals the spec's expression evaluated on the tail, and so does the
    `info["success"]` the step emitted. Both values appear on scratchitch (every clause
    holds on all its contact states: 1-3 cm from the itch, tens of newtons, moving) and on
    bedbathing (the count crosses 26). Only False appears on armmanipulation (the forearm
    never reaches the waist target: the descent hooks it, it does not lift it), feeding
    (< 4 cm from the mouth needs the spoon there at under 5 N; the greedy press lands at
    tens to hundreds of newtons) and teethbrushing (< 3 cm with the bristle FACE loaded and moving). Not forced:
    a controller built to satisfy a clause tests the controller."""
    env = contact[task]["env"]
    seen = set()
    for _s, _a, s2, info in contact[task]["transitions"]:
        want = _spec_clause(env, task, s2)
        assert bool(env._row["check"](env, _Derived(env, s2, forward=True))) == want
        assert info["success"] == want
        seen.add(want)
    assert False in seen
    if task in ("scratchitch", "bedbathing"):
        assert True in seen, f"{task}: the check never held on the descent"


# -- reset ------------------------------------------------------------------------------

#: `test_reset_normalises_the_root_quaternion`'s bound on |q - key_q|, one line per row (a MEASURED
#: statement, not a floor): the vendored five draw U(-5e-3, 5e-3) on the four components and nothing
#: else moves them (measured on the six seeds this test draws: 0.0051-0.0066).
_ROOT_Q_BOUND = {"scratchitch": 0.02, "bedbathing": 0.02, "armmanipulation": 0.02, "feeding": 0.02,
                 "teethbrushing": 0.02}


@pytest.mark.parametrize("task", sorted(_TASKS))
def test_reset_normalises_the_root_quaternion(contact, task):
    """Upstream adds U(-5e-3, 5e-3) to every qpos entry, the root quaternion included, and
    leaves the raw draw for its first step to normalise; `mj_forward` does NOT normalise
    `qpos` (measured), so without an explicit normalisation the reset observation carries
    |q| - 1 of up to 6.6e-3 that no later observation has. `obs[3:7]` is unit to 1e-12 on
    every seed -- and it is
    the noisy draw, not the keyframe's, so the normalisation is of a perturbed quaternion."""
    env = contact[task]["env"]
    key_q = np.asarray(env._model.key_qpos[env._key][3:7], dtype=float)
    for seed in range(6):
        s = env.reset(np.random.default_rng(seed))
        q = s[3:7]
        assert abs(float(np.linalg.norm(q)) - 1.0) <= 1e-12, (task, seed, np.linalg.norm(q))
        assert not np.array_equal(q, key_q)
        assert 0.0 < float(np.linalg.norm(q - key_q)) < _ROOT_Q_BOUND[task], (task, seed, np.linalg.norm(q - key_q))


def test_the_legend_states_the_task_and_never_the_verdict():
    """The legend-independence check on the QUANTITIES rather than a word list, and
    without the simulator: `_legend_lines` is called unbound on a stub
    adapter with two `_Derived` stand-ins that differ in every quantity a `_chk_*` gates on
    or `task_metric` reduces. The legend must be the same text for both and carry none of
    their values -- the question (which task), never the answer.

    A panel printing `WIPED n/52` (on bedbathing exactly `task_metric * 52`), `DIST` (the
    distance every row's check gates on) or `FORCE` (scratchitch's gated force), with the
    same `render` feeding `evaluate.fitness.source: vlm_score` and the GT pairwise judge,
    makes a VLM verdict on this tier a transcription of the metric. A word ban alone would
    not catch the numbers; the state-independence assertion does.
    """
    from types import SimpleNamespace

    from bird.envs.assistax import Assistax

    class _Stand(SimpleNamespace):
        def __getattr__(self, name):            # any other derived field: a 3-vector
            return np.zeros(3)

    def derived(**kw):
        base = dict(dist=0.0123, n_wiped=37, tool_force_mag=4.7, tip_force_mag=2.9,
                    tool_speed=0.31, tangential_speed=0.17, dist_forearm_waist=0.456,
                    rot_err=0.3, spoon_up_dot_world_up=0.9, slots=np.zeros(N_WIPE_POINTS))
        base.update(kw)
        return _Stand(**base)

    d1 = derived()
    d2 = derived(dist=0.987, n_wiped=3, tool_force_mag=0.0, tip_force_mag=0.0, tool_speed=0.0,
                 tangential_speed=0.0, dist_forearm_waist=0.011, rot_err=1.2,
                 spoon_up_dot_world_up=-0.4)
    for task, row in _TASKS.items():
        stub = SimpleNamespace(_task=task, _row=row, n_q=0, n_v=0, n_slots=0)
        stub._tail_index = lambda s=stub: Assistax._tail_index(s)
        l1, l2 = Assistax._legend_lines(stub, d1), Assistax._legend_lines(stub, d2)
        assert l1 == l2, f"{task}: the legend depends on the state -- {l1} vs {l2}"
        text = " ".join(l1).upper()
        # the key's underscores draw as spaces (the sheet has none), so match the words
        assert task.upper().replace("_", " ") in text, f"{task}: the legend does not even name the task: {l1}"
        for banned in ("WIPED", "DIST", "FORCE", "TOOL", "PASS", "FAIL", "SUCCESS", "SCORE",
                       "METRIC", "FITNESS"):
            assert banned not in text, f"{task}: legend says {banned!r}: {l1}"
        for value in ("37", "/52", "0.012", "4.7"):
            assert value not in text, f"{task}: legend carries a gated quantity {value!r}: {l1}"


# -- the vendored scenes (sim-free: the scene XML) ---------------------------------------------------

#: `torquescale` of the pelvis weld per scene: the vendored 0 on the wheelchair-layout scenes, None
#: where the scene has no weld (the bed layout, neq 0).
_WELD_TORQUESCALE = {"wheelchair_scene.xml": "0", "feeding_scene.xml": "0",
                     "teethbrushing_scene.xml": "0", "bed_scene.xml": None,
                     "bed_scene_armmanip.xml": None}


def _scene_text(name: str) -> str:
    from bird.envs.assistax import ASSETS
    return (ASSETS / name).read_text()


def _weld_attr(text: str, attr: str):
    m = re.search(r'<weld\b[^>]*\b' + attr + r'="([^"]*)"', text)
    return m.group(1) if m else None


def test_every_scene_is_named_by_the_torquescale_table():
    scenes = {row["scene"] for row in _TASKS.values()}
    assert scenes == set(_WELD_TORQUESCALE), (sorted(scenes ^ set(_WELD_TORQUESCALE)))


@pytest.mark.parametrize("scene", sorted(_WELD_TORQUESCALE))
def test_the_pelvis_weld_torquescale_per_scene(scene):
    """The vendored weld: `torquescale="0"` (position only, no relpose) on the wheelchair
    layout; no weld at all on the bed layout."""
    text = _scene_text(scene)
    want = _WELD_TORQUESCALE[scene]
    if want is None:
        assert "<weld" not in text, scene
        return
    assert text.count("<weld") == 1, scene
    assert _weld_attr(text, "torquescale") == want, (scene, _weld_attr(text, "torquescale"))
    assert (_weld_attr(text, "relpose") is not None) == (want == "1"), scene


def test_the_vendored_five_read_euler():
    """The five vendored scenes carry no integrator attribute (MuJoCo's default, Euler), as
    upstream's do."""
    vendored = {row["scene"] for row in _TASKS.values() if row["upstream"] is not None}
    assert len(vendored) == 5 and len(_TASKS) == 5, sorted(vendored)
    for scene in sorted(vendored):
        assert 'integrator="' not in _scene_text(scene), scene
