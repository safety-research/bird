"""The gymnasium/MuJoCo spec adapter (`bird/envs/gym_mujoco.py`).

Three halves, split by what they need installed, and the split is the point:

**Nothing** (pyyaml + numpy, the PR job): eight of the nine ground-truth fitnesses
are pure functions of a trajectory array, so they are tested on synthetic episodes
against hand-computed values -- including the orderings RDA's own docstrings record
(a hopper that stands still scores 0.0 on `hop_minus_drift`, one that hops away
scores metres negative). The stub they run against is built FROM `_FAMILIES`, not
hand-copied, so a corrected constant cannot leave this half testing a
configuration that no longer exists.

`gated_mean_com_velocity` (REvolve's Eq. 3) is the NINTH and the exception: it
reads the mass centre, which is not in the trajectory array and needs a forward
pass. It is still tested here -- the stub is given a `com_x` that reads a known
column, which is what lets the GATE and the mean be checked as arithmetic -- and
the claim that `com_x` really is the wheel's own mass centre is a
with-the-simulator test. Splitting it that way is deliberate: the gate's
behaviour and the identity of the quantity it integrates are two different
claims, and only the second needs a model compiled.

**Without the simulator** (a subprocess with `gymnasium`/`mujoco` poisoned, via
`conftest.run_probe`): `registry.load_all()` imports this module on every
CLI path and forgives an ImportError only for `anthropic_client`, so the module
must import, register its eleven ids and leave `MUJOCO_GL` alone with no simulator
present -- and constructing one must fail with the extra named AND no side effect
on the process.

**With the simulator** (`uv sync --extra metaworld`, run by hand): the one
property everything else silently depends on -- `_step(s, a)` is a function *of
the state passed in*. Asserted BITWISE over shuffled replay, never `allclose`: an
approximate restore has no other symptom, the screens just quietly score physics
that never happened. Plus the claims that justify this adapter's shape: the
observation IS `concat(qpos, qvel)`, termination is the env's own on the families
that have one, the reacher fingertip trig matches the live model, `full_source`
shows the environment rather than this adapter's plumbing (and the strip removes
the reward from it), and the spec's flat anchors are refused rather than
normalised against. And for `gym_humanoid_run`: its `reference_reward` is the
one in this family that TOUCHES THE SIMULATOR (a
forward pass for the mass-centre velocity and the contact term), and
`training._rollout` calls it BETWEEN two `step`s -- so an interleaved call must
leave the next step bitwise unchanged, which is asserted rather than argued.
"""

import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

from bird import registry, tasks
from bird.config import ConfigError, load
from bird.envs.gym_mujoco import (_CONSUMER_FORBIDDEN, _EXPECTED_GYM, _FAMILIES,
                                  _FITNESSES, GymMujoco, REACH_HOLD_RADIUS_M,
                                  TARGET_HEADING_RAD, TARGET_SPEED_MPS, _env_id,
                                  _gym_specs)

#: The eleven ids, derived the same way the module derives them, so this file cannot
#: drift from the registration loop without one of them disappearing.
ALL_IDS = [_env_id(t) for t in _EXPECTED_GYM]


# --------------------------------------------------------------------------
# no simulator installed
# --------------------------------------------------------------------------

_NO_SIMULATOR_PROBE = r"""
import json, os, sys

sys.modules["gymnasium"] = None
sys.modules["mujoco"] = None
os.environ.pop("MUJOCO_GL", None)

import bird.envs.gym_mujoco as gm
from bird import registry

out = {
    "gl_after_import": os.environ.get("MUJOCO_GL"),
    "imported": sorted(m for m in ("gymnasium", "mujoco")
                       if sys.modules.get(m) is not None),
    "ids": sorted(n for n in registry.names("env") if n.startswith("gym_")),
}
try:
    gm.GymMujoco("hopper_hop")
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


def test_the_module_imports_with_no_simulator_installed(without_simulator):
    assert without_simulator["imported"] == [], (
        "importing the adapter module pulled in "
        f"{without_simulator['imported']}; the simulator imports belong inside "
        "GymMujoco.__init__, not at module scope")


def test_importing_the_module_does_not_touch_the_process_environment(without_simulator):
    assert without_simulator["gl_after_import"] is None


def test_all_eleven_ids_register_without_the_simulator(without_simulator):
    assert without_simulator["ids"] == sorted(ALL_IDS)


def test_constructing_without_the_dependency_names_the_extra(without_simulator):
    """A bare `ModuleNotFoundError: No module named 'gymnasium'` from four frames
    down tells an operator nothing about which config key asked for it."""
    assert without_simulator["error_type"] == "ImportError"
    message = without_simulator["error"]
    for expected in ("problem.env_id", "gym_hopper_hop", "gymnasium", "mujoco",
                     "--extra metaworld"):
        assert expected in message, f"the install hint omits {expected!r}:\n{message}"


def test_a_failed_construction_leaves_the_process_untouched(without_simulator):
    """The dependency probe runs before the `MUJOCO_GL` choice and the Triton
    preload, so a missing-extra failure has no side effect beyond the raised
    error: a later adapter -- or the debugging
    session reading the traceback -- must not inherit a GL backend chosen for a
    construction that never happened."""
    assert without_simulator["gl_after_failed_construction"] is None


# --------------------------------------------------------------------------
# registration and the config surface
# --------------------------------------------------------------------------


def test_the_eleven_ids_are_registered_as_envs():
    names = set(registry.names("env"))
    assert len(ALL_IDS) == 11
    missing = [n for n in ALL_IDS if n not in names]
    assert not missing, f"registered env ids are missing {missing}"


def test_the_id_rule_and_the_module_agree():
    """`tasks._BIRD_ID_RULES["mujoco"]` is the definition and `_env_id` the
    mirror; the coverage partition holds them together, and this failure names
    the two files directly rather than through that test's aggregate message."""
    for task_id, spec in _gym_specs().items():
        assert spec.bird_env_id == _env_id(task_id) == "gym_" + task_id


def test_a_vanished_spec_fails_loud_and_a_new_spec_does_not(monkeypatch):
    """The catalogue check is ONE-SIDED by design: it runs inside
    `registry.load_all()`, which forgives nothing, so a symmetric check would
    let a pure-data commit -- an eleventh gymnasium spec -- take down every
    config load and most of the suite. A MISSING pinned spec still raises (an
    adapter without its definition renders nothing); an EXTRA one is deferred
    to the coverage partition test, which fails until it is wired or listed as
    an exemption -- loud in the right place."""
    import bird.envs.gym_mujoco as gm

    real = dict(tasks.index())
    without_one = {k: v for k, v in real.items() if k != "hopper_hop"}
    monkeypatch.setattr(gm, "task_index", lambda: without_one)
    with pytest.raises(tasks.TaskSpecError, match="hopper_hop"):
        gm._gym_specs()

    extra = dict(real)
    extra["walker2d_walk"] = real["hopper_hop"]  # any spec with library mujoco
    monkeypatch.setattr(gm, "task_index", lambda: extra)
    found = gm._gym_specs()                      # must NOT raise
    assert "walker2d_walk" in found


@pytest.mark.parametrize("env_id", ALL_IDS)
def test_problem_env_id_accepts_every_gym_id(env_id):
    """`problem.env_id` is `F("§0", (S,), kind="env")`, so registering the adapter
    IS the schema edit. This asserts the two agree in the direction that matters:
    a config naming a gymnasium task validates. The carrier is `zeroshot`, an
    UNSUPERVISED point (its fitness source selects nothing native), because the
    question here is registration, not signal: a supervised config is refused on
    the five gym tasks that ship no native signal -- see the next test."""
    cfg = load("zeroshot", profile="tester", overrides={"problem.env_id": env_id})
    assert cfg["problem.env_id"] == env_id


#: The five gymnasium tasks with neither a shipped success nor a shipped reward
#: (`reward.human.kind: none`; see bird/native_signal.py).
_NO_NATIVE_SIGNAL = {"gym_half_cheetah_backward", "gym_half_cheetah_target_speed",
                     "gym_hopper_hop_in_place", "gym_reacher_hold", "gym_swimmer_heading"}


@pytest.mark.parametrize("env_id", ALL_IDS)
def test_a_supervised_config_is_refused_on_exactly_the_gym_tasks_with_no_native_signal(env_id):
    """`eureka.yaml` pins `evaluate.fitness.source: native`, so the native-signal
    rule refuses it at load on the five gym tasks that ship nothing it can read
    -- a supervised search has no admissible signal there. Every other gym task
    loads. Without the rule `eureka` would load on all ten and score the BIRD
    custom metric on the five."""
    if env_id in _NO_NATIVE_SIGNAL:
        with pytest.raises(ConfigError, match="ships neither a native success"):
            load("eureka", profile="tester", overrides={"problem.env_id": env_id})
    else:
        assert load("eureka", profile="tester", overrides={"problem.env_id": env_id})


def test_a_mistyped_gym_id_is_still_fatal():
    with pytest.raises(ConfigError):
        load("eureka", profile="tester", overrides={"problem.env_id": "gym_hopper-hop"})


def test_the_instruction_inherits_from_the_spec():
    """`TaskSpec.instruction` owns the l_task -> natural_language fallback; this
    asserts the whole chain: the RDA-owned gym specs have no `l_task`, so
    `problem.task_description` materialises upstream's single field -- which on
    these specs IS the instruction -- rather than `null` rendering "None" as the
    prompt's first line."""
    cfg = load("eureka", profile="tester", overrides={"problem.env_id": "gym_hopper_hop"})
    spec = tasks.load("hopper_hop")
    assert spec.description.get("l_task") is None, "the fallback case must be real"
    assert cfg["problem.task_description"] == spec.instruction \
        == spec.description["natural_language"]
    assert "hop forward" in cfg["problem.task_description"]


def test_the_consumer_gate_is_inherited_from_the_factory():
    """The specs are RDA-owned and carry `forbidden_symbols: []` with no BIRD
    consumer entry, so the anti-leak gate comes from the FACTORY -- otherwise
    `-s problem.env_id=gym_half_cheetah` over a config leaving the key null
    resolves an empty gate, `static_forbidden_symbols` returns immediately, and
    a candidate returning `reference_reward` (a real expert reward on this tier)
    tops the pool. Without the factory gate the asymmetry is exact: the same
    override onto an `mt10_*` env inherits six symbols; onto a `gym_*` env,
    nothing."""
    cfg = load("eureka", profile="tester", overrides={"problem.env_id": "gym_half_cheetah"})
    assert cfg["verify.forbidden_symbols"] == list(_CONSUMER_FORBIDDEN)
    # and the spec really supplies nothing, so the factory is the live source:
    from bird.config import forbidden_symbols_of

    assert forbidden_symbols_of(tasks.load("half_cheetah")) == []


def test_a_non_default_reduction_is_refused():
    """`supported_reductions` lists only the schema's inert default -- none of
    the ten metrics is a reduction of a per-step success check -- so any other
    `evaluate.fitness.reduction` must be refused at validation rather than
    resolving into an artifact as a key the metric never honours. (Coherence
    validates the DEFAULT too, which is why the tuple cannot simply be empty:
    it means "the values a config may carry".)"""
    with pytest.raises(ConfigError, match="reduction"):
        load("eureka", profile="tester", overrides={
            "problem.env_id": "gym_hopper_hop",
            "evaluate.fitness.reduction": "any_step_episode_fraction"})


def test_success_rate_fitness_is_refused():
    """`success()` is False by construction on this tier (no expert anchor, no
    bar), so `evaluate.fitness.source: success_rate` would score every candidate
    exactly 0.0 -- a real value, not the failure sentinel, so selection becomes
    an all-tie and map-elites' strict `>` freezes the first occupant of every
    cell. Refused where the config is read, not discovered in a finished sweep."""
    with pytest.raises(ConfigError, match="success"):
        load("eureka", profile="tester", overrides={
            "problem.env_id": "gym_swimmer_forward",
            "evaluate.fitness.source": "success_rate"})


def test_an_unknown_task_lists_the_catalogue():
    """Raised before the simulator import, so this holds everywhere."""
    with pytest.raises(KeyError) as excinfo:
        GymMujoco("half-cheetah")
    assert "hopper_hop" in str(excinfo.value)


def test_every_factory_advertises_its_contract():
    """`_check_coherence` asks the FACTORY, because constructing the adapter to
    answer would import mujoco per validated config. Three facts travel this
    way; a factory that lost one silently re-opens the coherence hole it closes."""
    for env_id in ALL_IDS:
        factory = registry.get("env", env_id)
        assert factory.supported_reductions == ("per_step_fraction",)
        assert factory.consumer_forbidden_symbols == _CONSUMER_FORBIDDEN
        assert factory.defines_success is False
        assert (factory.__doc__ or "").strip(), f"{env_id}: factory has no doc"


# --------------------------------------------------------------------------
# the fitnesses the specs name, on synthetic trajectories (no simulator)
# --------------------------------------------------------------------------
#
# `_FITNESSES` entries read nothing off the adapter but `n_q`, `dt`, `horizon`
# and `obs_dim`, so a stub carrying those is enough -- which is what lets the
# fitness CONTRACT run in the PR job, where the simulator half of this file
# skips. The stub is built FROM `_FAMILIES`: a hand-copied `(n_q, dt)` table here
# would let the PR-job half quietly test a configuration that no longer exists.


def _stub(gym_env_id: str, horizon: int = 1000,
          fitness: str = "forward_displacement") -> SimpleNamespace:
    fam = _FAMILIES[gym_env_id]
    return SimpleNamespace(n_q=int(fam["n_q"]), dt=float(fam["dt"]),
                           obs_dim=int(fam["n_q"]) + len(fam["vel_lo"]),
                           horizon=horizon, _fitness=_FITNESSES[fitness])


def test_the_specs_name_exactly_the_fitnesses_the_module_implements():
    """A surjection, not a bijection: `forward_displacement` backs three tasks
    (cheetah forward, hopper_hop, swimmer_forward), so nine measures back eleven
    tasks. A spec naming a measure outside this set must fail at construction
    with the file named -- the `success_check_for` pattern."""
    named = {str(s.continuous_success["raw"]["name"]) for s in _gym_specs().values()}
    assert named == set(_FITNESSES)
    assert len(named) == 9 and len(_gym_specs()) == 11


def test_displacement_is_final_minus_reset_x():
    env = _stub("HalfCheetah-v5")
    states = np.zeros((11, 18))
    states[:, 0] = np.linspace(0.0, 5.0, 11)
    assert _FITNESSES["forward_displacement"](env, states) == pytest.approx(5.0)
    assert _FITNESSES["backward_displacement"](env, states) == pytest.approx(-5.0)


def test_target_speed_telescopes_to_the_endpoint_form():
    """Upstream means the per-step finite differences `(x_t - x_{t-1}) / dt`; the
    adapter uses the telescoped endpoints. Same number, asserted by computing the
    upstream form here."""
    env = _stub("HalfCheetah-v5")
    rng = np.random.default_rng(0)
    states = np.zeros((21, 18))
    states[:, 0] = np.cumsum(rng.uniform(-0.1, 0.4, size=21))
    upstream_vbar = float(np.mean(np.diff(states[:, 0]) / env.dt))
    got = _FITNESSES["neg_speed_error"](env, states)
    assert got == pytest.approx(-abs(upstream_vbar - TARGET_SPEED_MPS))


def test_hop_in_place_orders_the_three_behaviours():
    """The ordering RDA's own docstring records from synthetic episodes: stand
    still 0.000, hop in place small-positive, hop away metres-negative. The
    fitness cannot rank methods if these three collapse."""
    env = _stub("Hopper-v5")
    T = 101
    stand = np.zeros((T, 12)); stand[:, 1] = 1.25
    hop = np.zeros((T, 12))
    hop[:, 1] = 1.25 + 0.07 * np.sin(np.linspace(0, 20 * np.pi, T))
    away = hop.copy(); away[:, 0] = np.linspace(0.0, 5.0, T)
    f = _FITNESSES["hop_minus_drift"]
    assert f(env, stand) == 0.0
    assert f(env, hop) > 0.03
    assert f(env, away) < -4.0


def test_hop_in_place_reads_arriving_states_and_the_reset_origin():
    """z over `states[1:]` (upstream records one row per EXECUTED step) and drift
    from `states[0]` (upstream's `x_start` is the reset value, not the first
    record). A fitness off by one row here scores a different episode."""
    env = _stub("Hopper-v5")
    states = np.zeros((3, 12))
    states[0, 1] = 99.0          # the reset z must NOT enter std(z)
    states[1:, 1] = 1.25         # arriving z constant -> std 0
    states[0, 0] = 2.0           # drift measured from reset x...
    states[1:, 0] = 5.0          # ...to final x: |5 - 2| = 3
    assert _FITNESSES["hop_minus_drift"](env, states) == pytest.approx(-3.0)


def test_balance_fraction_has_no_clamp_to_hide_a_wrong_horizon():
    """`(steps survived) / horizon`, unclamped: every real producer is
    horizon-capped, so a value above 1.0 is a bug (wrong horizon, concatenated
    episodes) that must SURFACE as >1 rather than be absorbed into a perfect
    1.000 -- a `min(1.0, ...)` would report the one case it could ever fire on
    as a top score."""
    env = _stub("InvertedPendulum-v5")
    f = _FITNESSES["balance_fraction"]
    assert f(env, np.zeros((251, 4))) == pytest.approx(0.25)
    assert f(env, np.zeros((1001, 4))) == pytest.approx(1.0)
    assert f(env, np.zeros((1501, 4))) == pytest.approx(1.5), \
        "an over-horizon trajectory must expose itself, not score 1.000"


def test_reacher_fitnesses_disagree_exactly_where_the_tasks_do():
    """`reacher_hold` exists because `reacher_reach`'s final-step fitness cannot
    see the hover hack -- sweep through the target late and drift off. Built
    here: on-target for the last step only scores well on reach and near zero on
    hold; parked on target the whole time scores well on both."""
    env = _stub("Reacher-v5", horizon=50)
    # fingertip at angles (0, 0) is (0.21, 0); target there means distance 0
    sweep = np.zeros((11, 8)); sweep[:, 2] = 0.5           # far all episode...
    sweep[-1, 2] = 0.21                                    # ...on target at the end
    parked = np.zeros((11, 8)); parked[:, 2] = 0.21
    reach, hold = _FITNESSES["neg_final_reach_distance"], _FITNESSES["dwell_fraction"]
    assert reach(env, sweep) == pytest.approx(0.0)
    assert hold(env, sweep) == pytest.approx(0.1)          # 1 of 10 arriving steps
    assert reach(env, parked) == pytest.approx(0.0)
    assert hold(env, parked) == pytest.approx(1.0)
    # and the radius is REACH_HOLD_RADIUS_M, not "close":
    edge = np.zeros((3, 8)); edge[:, 2] = 0.21 + REACH_HOLD_RADIUS_M * 1.01
    assert hold(env, edge) == 0.0


def test_heading_speed_projects_and_penalises_the_wrong_way():
    env = _stub("Swimmer-v5")
    diag = np.zeros((11, 10))
    diag[:, 0] = np.linspace(0.0, 2.0, 11)
    diag[:, 1] = np.linspace(0.0, 2.0, 11)
    want = (2 * math.cos(TARGET_HEADING_RAD) + 2 * math.sin(TARGET_HEADING_RAD)) / (10 * env.dt)
    f = _FITNESSES["heading_speed"]
    assert f(env, diag) == pytest.approx(want)
    # pure +x motion scores only its projection...
    x_only = np.zeros((11, 10)); x_only[:, 0] = np.linspace(0.0, 2.0, 11)
    assert f(env, x_only) == pytest.approx((2 * math.cos(TARGET_HEADING_RAD)) / (10 * env.dt))
    # ...and against-heading motion is negative, not clipped to zero
    assert f(env, x_only[::-1].copy()) < 0


#: `Humanoid-v5`'s joint and actuator addressing, CAPTURED FROM THE MODEL and committed
#: as a literal. The live-model check
#: (`test_the_humanoid_field_table_matches_the_live_model`) is the authority, but it needs
#: the simulator, and the simulator is exactly what the PR job does not have -- so on its
#: own the abdomen_y/abdomen_z transposition would be caught by nothing that EXECUTES in
#: CI. This literal closes that: the simulator-free test below compares the row's table
#: against it, and the live-model test compares THIS against the model, so neither can rot
#: without the other failing. Two copies of one fact, deliberately, with a test holding
#: them equal -- the `_FAMILIES`-vs-live-model pattern the row's `n_q`/`dt` assertions
#: already use.
#:
#: (joint name, jnt_qposadr, jnt_dofadr), joint order, root excluded.
HUMANOID_JOINTS = (
    ("abdomen_z", 7, 6), ("abdomen_y", 8, 7), ("abdomen_x", 9, 8),
    ("right_hip_x", 10, 9), ("right_hip_z", 11, 10), ("right_hip_y", 12, 11),
    ("right_knee", 13, 12),
    ("left_hip_x", 14, 13), ("left_hip_z", 15, 14), ("left_hip_y", 16, 15),
    ("left_knee", 17, 16),
    ("right_shoulder1", 18, 17), ("right_shoulder2", 19, 18), ("right_elbow", 20, 19),
    ("left_shoulder1", 21, 20), ("left_shoulder2", 22, 21), ("left_elbow", 23, 22),
)
#: Actuator order, which is NOT joint order -- the trap this literal most exists for:
#: `a[0]` drives `abdomen_y` (s[8]) and `a[1]` drives `abdomen_z` (s[7]), so a table built
#: by zipping the two lists is wrong in exactly its first two rows and right everywhere
#: else. Third element is `actuator_gear[:, 0]`, the factor between an action component
#: and a torque.
HUMANOID_ACTUATORS = (
    ("abdomen_y", 8, 100), ("abdomen_z", 7, 100), ("abdomen_x", 9, 100),
    ("right_hip_x", 10, 100), ("right_hip_z", 11, 100), ("right_hip_y", 12, 300),
    ("right_knee", 13, 200),
    ("left_hip_x", 14, 100), ("left_hip_z", 15, 100), ("left_hip_y", 16, 300),
    ("left_knee", 17, 200),
    ("right_shoulder1", 18, 25), ("right_shoulder2", 19, 25), ("right_elbow", 20, 25),
    ("left_shoulder1", 21, 25), ("left_shoulder2", 22, 25), ("left_elbow", 23, 25),
)


def test_the_humanoid_field_table_sits_where_the_model_addresses_it():
    """The index half of the field table, WITHOUT the simulator -- so it runs in the job
    that runs.

    `test_the_row_tables_describe_the_row_they_sit_in` pins the table's WIDTH and
    `test_the_humanoid_field_table_matches_the_live_model` pins its indices against the
    model, but the second needs `--extra metaworld` and the default CI job skips it, so a
    transposition would be caught only by a run with the extra installed. This test
    is the same check against `HUMANOID_JOINTS` / `HUMANOID_ACTUATORS`, which the
    live-model test holds equal to the model.

    Three traps, all easy to hit when writing the row: the free root spends qpos[0:7] so the
    hinges start at 7 and not 3; qvel's root block is 3 linear THEN 3 angular so the hinge
    velocities start at n_q + 6 = 30; and actuator order is not joint order."""
    row = _FAMILIES["Humanoid-v5"]
    state, action = list(row["state_fields"]), list(row["action_fields"])
    n_q = int(row["n_q"])
    assert (n_q, len(state), len(action)) == (24, 47, 17)

    # every entry self-labels its own index, in order: a transposition shows up as
    # `s[31]` sitting at position 30 even if both names are plausible there
    for i, (name, desc) in enumerate(state):
        assert desc.startswith(f"s[{i}]"), (i, name, desc[:40])
    for i, (name, desc) in enumerate(action):
        assert desc.startswith(f"a[{i}]"), (i, name, desc[:40])

    for jname, qadr, dadr in HUMANOID_JOINTS:
        assert state[qadr][0] == f"{jname}_angle", (qadr, jname, state[qadr][0])
        assert state[n_q + dadr][0] == f"{jname}_velocity", (
            n_q + dadr, jname, state[n_q + dadr][0])

    for i, (jname, drives, gear) in enumerate(HUMANOID_ACTUATORS):
        assert action[i][0] == f"{jname}_torque", (i, jname, action[i][0])
        assert f"gear {gear}" in action[i][1], (i, jname, action[i][1])
        m = re.search(r"drives s\[(\d+)\]", action[i][1])
        if m:
            assert int(m.group(1)) == drives, (i, jname, m.group(1), drives)

    # the a[0]/a[1] transposition, called out by name because it is the one an author
    # reproduces rather than invents
    assert action[0][0] == "abdomen_y_torque" and "drives s[8]" in action[0][1]
    assert action[1][0] == "abdomen_z_torque" and "drives s[7]" in action[1][1]

    js, vs = row["joint_slice"], row["joint_vel_slice"]
    assert js == (7, 24) and vs == (30, 47), (js, vs)
    assert row["symbol_mapping"]["joint_angles"] == f"s[{js[0]}:{js[1]}]"
    assert row["symbol_mapping"]["joint_velocities"] == f"s[{vs[0]}:{vs[1]}]"
    assert row["symbol_mapping"]["torso_height"] == f"s[{row['height_index']}]"


def _humanoid_stub(horizon=1000, com_col=0):
    """The humanoid fitness stub, plus the one thing `_stub` cannot synthesise.

    `gated_mean_com_velocity` reads the MASS CENTRE, which is not a column of the
    trajectory array, so the stub is given a `com_x` that reads column `com_col`.
    That substitution is exactly what makes the gate and the mean testable as
    arithmetic with no model compiled; the separate claim -- that the real `com_x`
    is the wheel's own `mass_center` -- is asserted in
    `test_com_velocity_is_the_wheels_own_and_the_root_velocity_is_not`, with the
    simulator. Keeping them apart is deliberate: conflating them would leave the
    gate untested on the PR job, which is the job that runs.
    """
    stub = _stub("Humanoid-v5", horizon=horizon, fitness="gated_mean_com_velocity")
    stub.com_x = lambda s: float(np.asarray(s, dtype=float).ravel()[com_col])
    return stub


def test_the_gate_zeroes_a_short_episode_and_means_a_full_one():
    """REvolve App. B.2 Eq. 3, both branches. T == T_max means the episode-mean COM
    velocity; T < T_max means 0 -- and 0 is a MEASUREMENT here, the one place this
    table departs from the module's `nan` rule, because `task_metric` has already
    returned `nan` for a record that is short, narrow or non-finite."""
    env = _humanoid_stub(horizon=10)
    full = np.zeros((11, 47))
    full[:, 0] = np.linspace(0.0, 1.5, 11)            # 1.5 m of COM travel
    # 10 steps at dt 0.015 -> 0.15 s -> 10 m/s
    assert _FITNESSES["gated_mean_com_velocity"](env, full) == pytest.approx(
        1.5 / (10 * env.dt))

    for n_steps in (1, 5, 9):                          # fell before the horizon
        short = full[:n_steps + 1]
        assert _FITNESSES["gated_mean_com_velocity"](env, short) == 0.0, n_steps


def test_the_gate_ranks_a_fall_above_a_reverse_run_and_that_is_the_papers_fitness():
    """A property of Eq. 3, pinned so nobody "fixes" it into a different metric.

    A policy that falls at once scores exactly 0.0; one that survives the whole
    horizon running BACKWARDS scores negative. So falling strictly outranks
    reversing, and ties with standing perfectly still. With the random anchor also a
    hard 0.0 (a random policy never survives 1000 steps), the bottom of this scale
    is a plateau on which "fell", "stood still" and "never ran" are
    indistinguishable -- which is why a flat early generation on this task is
    expected rather than evidence of a broken search.

    This is the PUBLISHED fitness and the whole point of the task is to compare
    against the published figure, so it is recorded as an open exploit on the spec
    and NOT rebalanced here. The assertion is the guard: a well-meaning change that
    clipped the metric at zero, or dropped the gate, breaks this test by name."""
    env = _humanoid_stub(horizon=10)
    fell = np.zeros((3, 47))
    fell[:, 0] = [0.0, 0.4, 0.8]                       # fast, then gone at step 2
    reverse = np.zeros((11, 47))
    reverse[:, 0] = np.linspace(0.0, -0.3, 11)         # upright all 10, going -x
    still = np.zeros((11, 47))                         # upright all 10, no motion

    f = _FITNESSES["gated_mean_com_velocity"]
    assert f(env, fell) == 0.0
    assert f(env, reverse) < 0.0
    assert f(env, still) == 0.0
    assert f(env, fell) > f(env, reverse)              # the quirk, stated as an order
    assert f(env, fell) == f(env, still)               # and the plateau


def test_every_family_row_carries_every_row_key():
    """A row is the COMPLETE definition of a family -- the module docstring's claim,
    and the thing that lets `__init__`, `_step`, `random_state` and `render` consume
    row keys and never branch on an env id.

    It is asserted rather than trusted because the humanoid family needs four keys
    the planar rows do not use (`quat_qpos`, `height_index`, `com_velocity`,
    `healthy`), and a row that simply
    omitted one would not fail loudly: `fam["com_velocity"]` raises KeyError only on
    the code path that reads it, which for five of the six rows is never. A seventh
    family inheriting a planar assumption by omission is exactly the failure this
    forbids.

    `dr_torso_body` is the ONE key read with `.get()` rather than `[]`, because only
    the three legged rows have a torso to put a payload on -- and it is pinned here
    as an EQUALITY rather than waved past, so a SECOND optional key cannot join it
    without this failing. That matters more than it looks: an optional row key is
    exactly how "every row is the complete definition" erodes, one `.get()` at a
    time, and the erosion is invisible until a family needs the key."""
    keys = [set(row) for row in _FAMILIES.values()]
    common = set.intersection(*keys)
    optional = set().union(*keys) - common
    assert optional == {"dr_torso_body"}, (
        f"row keys not present on every row: {sorted(optional)}. Only "
        "`dr_torso_body` may be optional (see this test's docstring); either give "
        "every row the new key or justify a second exemption here.")
    for required in ("quat_qpos", "height_index", "com_velocity", "healthy",
                     "terminates", "pin_qpos", "goal_disk", "camera_config",
                     "ref_reward", "bins", "state_fields",
                     "action_fields", "helpers", "symbol_mapping", "prose",
                     "n_q", "dt", "action_clip", "render", "root_lo", "root_hi",
                     "vel_lo", "vel_hi", "joint_slice", "joint_vel_slice"):
        assert required in common, f"not every row declares {required!r}"


def test_the_row_tables_describe_the_row_they_sit_in():
    """`state_fields` has one entry per state slot and `action_fields` one per
    actuator, on every row. Cheap, and it is the check that would have caught a
    47-entry humanoid table written against a 46-entry state."""
    for gym_env_id, row in _FAMILIES.items():
        width = int(row["n_q"]) + len(row["vel_lo"])
        assert len(row["state_fields"]) == width, (
            f"{gym_env_id}: {len(row['state_fields'])} state fields for a "
            f"{width}-wide state")
        assert len(row["vel_lo"]) == len(row["vel_hi"]), gym_env_id
        assert len(row["root_lo"]) == len(row["root_hi"]) <= int(row["n_q"]), gym_env_id
        assert len(row["bins"]) == int(row["n_q"]), gym_env_id
        names = [n for n, _d in row["state_fields"]]
        assert len(set(names)) == len(names), f"{gym_env_id}: duplicate state field name"


def test_humanoid_is_upright_is_the_wheels_z_band_and_not_hoppers_check():
    """Humanoid-v5's healthy test is `min_z < qpos[2] < max_z` and NOTHING else --
    no pitch clause, no +/-100 state range. Strict on both sides, and reading s[2]
    rather than the s[1] the planar rows use.

    Two names for two predicates, following the module's own per-family helper
    naming: `is_healthy` is Hopper's three-clause check and `is_balanced` the
    pendulum's. Sharing one name would advertise Hopper's pitch limit to a model
    reading a humanoid's helper table, and `is_upright` therefore REFUSES a row
    whose `healthy` is not a `z_range` rather than silently answering for it."""
    fam = _FAMILIES["Humanoid-v5"]
    assert fam["healthy"] == ("z_range", 1.0, 2.0) and fam["height_index"] == 2
    stub = SimpleNamespace(_family=fam, name="gym_humanoid_run",
                           _gym_id="Humanoid-v5")

    def s_at(z, **kw):
        s = np.zeros(47)
        s[2] = z
        for i, v in kw.items():
            s[int(i)] = v
        return s

    assert GymMujoco.is_upright(stub, s_at(1.4)) == 1.0
    assert GymMujoco.is_upright(stub, s_at(1.0)) == 0.0     # strict, not <=
    assert GymMujoco.is_upright(stub, s_at(2.0)) == 0.0     # strict, not >=
    assert GymMujoco.is_upright(stub, s_at(0.99)) == 0.0
    assert GymMujoco.is_upright(stub, s_at(2.01)) == 0.0
    # Hopper's clauses do NOT apply: a wildly pitched, fast-spinning humanoid whose
    # torso is still in the band is alive upstream and must be alive here.
    assert GymMujoco.is_upright(stub, s_at(1.4, **{"5": 3.0, "30": 250.0})) == 1.0
    # non-finite is never alive
    assert GymMujoco.is_upright(stub, s_at(np.nan)) == 0.0
    assert GymMujoco.is_upright(stub, s_at(1.4, **{"31": np.inf})) == 0.0

    for gym_env_id, row in _FAMILIES.items():
        if gym_env_id == "Humanoid-v5":
            continue
        other = SimpleNamespace(_family=row, name=gym_env_id, _gym_id=gym_env_id)
        with pytest.raises(NotImplementedError):
            GymMujoco.is_upright(other, np.zeros(60))


def test_torso_height_refuses_the_rows_that_have_no_height():
    """`height_index` is None on the pendulum, reacher and swimmer, whose s[1] is a
    pole angle, an elbow angle and a y position. Returning one of those under the
    name `torso_height` would be a plausible number measuring something else, so
    it raises.

    And where the row DOES declare an index, the advertised helper and the
    `symbol_mapping` entry must agree about which slot it is: two claims about the
    same thing in one row is exactly what a row is supposed to prevent."""
    for gym_env_id, row in _FAMILIES.items():
        stub = SimpleNamespace(_family=row, name=gym_env_id, _gym_id=gym_env_id)
        s = np.arange(60, dtype=float)
        idx = row["height_index"]
        advertised = any(n == "torso_height(s)" for n, _d in row["helpers"])
        if idx is None:
            assert not advertised, f"{gym_env_id} advertises a helper its row refuses"
            with pytest.raises(NotImplementedError):
                GymMujoco.torso_height(stub, s)
        else:
            assert GymMujoco.torso_height(stub, s) == float(idx)
            mapped = row["symbol_mapping"].get("torso_height")
            if mapped is not None:
                assert mapped == f"s[{idx}]", (gym_env_id, mapped, idx)


def test_the_humanoid_row_maps_no_velocity_symbol_to_a_state_slice():
    """The one row where an omission from `symbol_mapping` is a CORRECTNESS matter.

    On this body the forward velocity the task is scored on is the mass centre's,
    which is not any `s[i]`; the nearest slice, `s[n_q]`, correlates -0.39 with it.
    `_apply_symbol_mapping` is a text substitution into candidate code, so a
    `"forward_velocity": "s[24]"` entry would silently rewrite a candidate's correct
    symbol into a quantity of the opposite sign. A symbol whose value is not a state
    slice gets no slice -- and the five planar rows, where it IS a slice, keep theirs."""
    humanoid = _FAMILIES["Humanoid-v5"]["symbol_mapping"]
    for banned in ("forward_velocity", "velocity", "speed", "x_velocity"):
        assert banned not in humanoid, (
            f"{banned!r} is mapped on Humanoid-v5, but the quantity it names is not a "
            "state slice on this body -- see _fit_gated_mean_com_velocity")
    # the helper table is where it is reachable, and it says so
    helpers = dict(_FAMILIES["Humanoid-v5"]["helpers"])
    assert "forward_velocity(s)" in helpers
    assert "CENTRE OF MASS" in helpers["forward_velocity(s)"]
    # and the planar rows still map theirs, so this is a per-row fact not a retreat
    assert _FAMILIES["HalfCheetah-v5"]["symbol_mapping"]["forward_velocity"] == "s[9]"
    assert _FAMILIES["Hopper-v5"]["symbol_mapping"]["forward_velocity"] == "s[6]"


def test_no_family_symbol_mapping_key_is_a_single_letter():
    """The rule `tests/test_env_spec_leak.py` enforces, asserted here WITHOUT a
    simulator: that file's guard iterates buildable adapters, so under
    `--extra test` -- where these five families cannot be constructed -- it
    cannot see their mappings at all, and this test is what keeps the rule
    non-vacuous here. A single-letter key turns `_apply_symbol_mapping`'s
    replacement loose on every throwaway variable a candidate might use
    (`zero` -> `s[1]ero`)."""
    from bird.envs.gym_mujoco import _FAMILIES

    for gym_id, family in _FAMILIES.items():
        short = sorted(k for k in family["symbol_mapping"] if len(k) < 2)
        assert not short, f"{gym_id}: single-letter symbol_mapping keys {short}"


def test_the_flat_anchors_stay_refused():
    """The ten specs carry a measured `random` with no reduction label and a null
    expert; `baselines_of` must return None so fitness stays RAW rather than
    normalised against a scale with one end missing. `tests/test_task_specs.py`
    pins the refusal across every spec; this pins it for the ten envs this file is
    about, through the same reader the adapter's consumers use."""
    from bird.envs.spec import baselines_of
    from bird.tasks import ANY_STEP, PER_STEP

    for task_id, spec in _gym_specs().items():
        assert (spec.anchors.get("random") or {}).get("value") is not None, task_id
        for reduction in (PER_STEP, ANY_STEP):
            assert baselines_of(spec, reduction) is None, task_id


def test_a_trajectory_that_cannot_be_measured_scores_nan_not_zero():
    """On the signed fitnesses 0.0 is a CEILING, not a floor: it is a perfect
    cruise on `neg_speed_error`, on-target on `neg_final_reach_distance`, and
    above the measured random anchor (~ -4.58 m) on the displacement tasks -- so
    a malformed record scoring it would outrank every honest episode with
    `failure_kind` empty. `nan` routes through the seed aggregations' _clean and
    lands on `select.failure_value`, the convention `evaluation._failure_value`
    documents for exactly this trap."""
    env = _stub("HalfCheetah-v5", fitness="neg_speed_error")
    metric = GymMujoco.task_metric
    assert math.isnan(metric(env, None))                       # no states at all
    assert math.isnan(metric(env, np.zeros((1, 18))))          # a single row
    assert math.isnan(metric(env, np.zeros((5, 4))))           # wrong width
    honest = np.zeros((5, 18))                                 # a real, bad episode
    honest[:, 0] = np.linspace(0.0, -1.0, 5)                   # drifting backwards
    assert metric(env, honest) < 0.0, "the sentinel must not shadow real scores"


def test_a_diverged_trajectory_cannot_launder_into_the_metric():
    """`_rollout` appends the post-divergence state before it breaks, so the
    final row can carry inf -- which would make the displacement fitness inf, and
    `inf >= inf` would make `success()` True: a finite `success_rate = 1.0` from
    exactly the candidates most likely to diverge, on the tier documented never
    to succeed. Both paths must come back closed."""
    env = _stub("HalfCheetah-v5")
    diverged = np.zeros((5, 18))
    diverged[-1, 0] = float("inf")
    assert math.isnan(GymMujoco.task_metric(env, diverged))
    assert GymMujoco.success(env, diverged) is False


def test_success_is_false_by_construction():
    """No expert anchor -> no success bar; overridden rather than thresholded so
    no metric value of any kind -- inf included -- can clear it. The three costs
    are named on the method; the third (an all-tie under `fitness.source:
    success_rate`) is refused by coherence, tested above."""
    env = _stub("InvertedPendulum-v5", fitness="balance_fraction")
    full = np.zeros((1001, 4))
    assert GymMujoco.task_metric(env, full) == pytest.approx(1.0)
    assert GymMujoco.success(env, full) is False
    assert GymMujoco.success_threshold == float("inf")


def test_hopper_is_healthy_matches_the_wheel_clause_for_clause():
    """The +/-100 range clause applies to `state_vector()[2:]` upstream -- the
    HEIGHT is excluded, bounded only below by 0.7 -- so applying it to `s[1:]`
    would be wrong: a finite torso at z >= 100 is (bizarrely but
    truly) still alive upstream, still collects the alive bonus, and this
    adapter's reference reward must agree."""
    healthy = np.zeros(12); healthy[1] = 1.25
    assert GymMujoco.is_healthy(None, healthy) == 1.0
    sky_high = healthy.copy(); sky_high[1] = 150.0        # z outside +/-100: ALIVE
    assert GymMujoco.is_healthy(None, sky_high) == 1.0
    tipped = healthy.copy(); tipped[2] = 0.3              # angle beyond 0.2: dead
    assert GymMujoco.is_healthy(None, tipped) == 0.0
    blown = healthy.copy(); blown[7] = 150.0              # a velocity at 150: dead
    assert GymMujoco.is_healthy(None, blown) == 0.0
    low = healthy.copy(); low[1] = 0.5                    # fallen: dead
    assert GymMujoco.is_healthy(None, low) == 0.0


# --------------------------------------------------------------------------
# with the simulator: fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simulator():
    """A fixture and NOT a module-level importorskip: at module scope the skip
    fires during collection and takes the no-simulator half of this file -- the
    half that must run on a machine without gymnasium -- with it."""
    pytest.importorskip("gymnasium", reason="needs `uv sync --extra metaworld`")
    return pytest.importorskip("mujoco", reason="needs `uv sync --extra metaworld`")


#: One adapter per simulator family carries the expensive proofs; the eleven tasks
#: are six models under nine objectives, and the restore contract is a property
#: of the MODEL. `gym_half_cheetah` and `gym_hopper_hop` between them cover both
#: `_step` paths (direct `do_simulation` and the env's own `step`).
#: `gym_humanoid_run` is here because it is the model the contract is HARDEST on:
#: its unzeroed-warmstart deviation is 7.5e-07 against HalfCheetah's 5.4e-14, so
#: it is the one row where a broken restore would show up as physics rather than
#: as the last bits of a float.
FAMILY_REPS = ("gym_half_cheetah", "gym_hopper_hop", "gym_inverted_pendulum_balance",
               "gym_reacher_reach", "gym_swimmer_forward", "gym_humanoid_run")


@pytest.fixture(scope="module")
def adapters(simulator) -> dict:
    """All ten adapters, one construction each for the whole module: a fresh set
    here plus a fresh set in the observation test would be 15 model compiles for
    10 models' worth of coverage."""
    return {name: registry.get("env", name)({}) for name in ALL_IDS}


def _shuffled_transitions(env, n=60, seed=0):
    rng = np.random.default_rng(seed)
    s = env.reset(rng)
    tr = []
    for _ in range(n):
        a = rng.uniform(-env.action_clip, env.action_clip, size=env.action_dim)
        s2, done, _info = env.step(s, a)
        tr.append((s, a, s2))
        s = env.reset(rng) if done else s2
    order = rng.permutation(len(tr))
    return tr, order


def test_the_observation_is_the_whole_simulator_state(adapters):
    """The claim the entire adapter rests on, checked against the live model. If
    this fails, `EnvAdapter`'s stateless `step(state, action)` is unsatisfiable
    here and this family needs `metaworld.py`'s obs -> snapshot cache -- a
    different and much larger object."""
    for env_id in ALL_IDS:
        env = adapters[env_id]
        s = env.reset(3)
        sim = env._env
        assert s.shape == (env.obs_dim,) == (int(sim.model.nq) + int(sim.model.nv),)
        assert np.allclose(s, np.concatenate([sim.data.qpos, sim.data.qvel])), env_id
        # and reproducible from the rng alone, which is what `_reset`'s derived
        # seed buys in place of transcribing five `reset_model` bodies
        assert np.array_equal(s, env.reset(3)), env_id


@pytest.mark.parametrize("name", FAMILY_REPS)
def test_step_is_a_bitwise_function_of_its_arguments(adapters, name):
    """Shuffled replay, never round-trips (a round-trip repeats the same state
    and cannot see a history dependence), asserted at 0.0 exactly: an approximate
    restore is the failure with no other symptom."""
    env = adapters[name]
    tr, order = _shuffled_transitions(env)
    dev = max(float(np.max(np.abs(env.step(tr[i][0], tr[i][1])[0] - tr[i][2])))
              for i in order)
    assert dev == 0.0, f"{name}: max |deviation| {dev:.3e} over shuffled replay"


def test_the_warmstart_zeroing_is_what_buys_the_purity(adapters):
    """The control that keeps the test above meaningful: the same replay with the
    warmstart left alone must NOT come back exact on the contact-rich family.
    Measured on this model: 5.4e-14 unzeroed against 0.0 zeroed. Run
    on the cheetah because whether the solver engages depends on where the policy
    goes -- on `gym_inverted_pendulum_balance` random actions never activate a
    constraint and the control measures 0.0 there too (recorded in the module
    docstring), so asserting it per-family would assert a vacuity."""
    env = adapters["gym_half_cheetah"]
    tr, order = _shuffled_transitions(env)
    worst = 0.0
    for i in order:
        s_i, a_i, s2_i = tr[i]
        env._env.set_state(s_i[:env.n_q], s_i[env.n_q:])
        env._env.do_simulation(np.clip(a_i, -env.action_clip, env.action_clip),
                               env.frame_skip)
        got = np.concatenate([env._env.data.qpos, env._env.data.qvel]).ravel()
        worst = max(worst, float(np.max(np.abs(got - s2_i))))
    assert worst > 1e-15, (
        "removing the qacc_warmstart zeroing did NOT change the result, so the "
        "purity assertion above proves nothing on this env. Either the contact "
        "regime changed or the control stopped bypassing the line it means to.")


def test_termination_is_the_environments_own(adapters):
    """The first envs in this repo whose episodes genuinely end early. Pushing
    the pendulum cart hard one way tips the pole past 0.2 rad within a few steps;
    a random hopper falls. Both `done` flags must come back, because a fitness
    defined as `steps survived / horizon` measures nothing if nothing ends."""
    ip = adapters["gym_inverted_pendulum_balance"]
    s, done, steps = ip.reset(0), False, 0
    while not done and steps < ip.horizon:
        s, done, _ = ip.step(s, [3.0])
        steps += 1
    assert done and steps < ip.horizon, f"pendulum never terminated ({steps} steps)"

    hop = adapters["gym_hopper_hop"]
    rng = np.random.default_rng(0)
    s, done, steps = hop.reset(rng), False, 0
    while not done and steps < hop.horizon:
        s, done, _ = hop.step(s, rng.uniform(-1, 1, hop.action_dim))
        steps += 1
    assert done and steps < hop.horizon, f"hopper never terminated ({steps} steps)"
    # and the three no-terminal families really have none: the guard is only the
    # non-finite catch, so a short random roll must come back all-False
    for name in ("gym_half_cheetah", "gym_swimmer_forward", "gym_reacher_reach"):
        env = adapters[name]
        s = env.reset(rng)
        for _ in range(20):
            s, done, _ = env.step(s, rng.uniform(-env.action_clip, env.action_clip,
                                                 env.action_dim))
            assert done is False, f"{name} terminated; it has no terminal state"


def test_the_fingertip_trig_matches_the_live_model(adapters):
    """`fingertip_pos` hardcodes the two link lengths from `reacher.xml` so the
    dwell and reach fitnesses stay pure numpy; this is the assertion that keeps
    those constants honest against the installed asset."""
    import mujoco

    env = adapters["gym_reacher_reach"]
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        s = env.random_state(rng)
        env._env.set_state(s[:env.n_q], s[env.n_q:])
        mujoco.mj_forward(env._env.model, env._env.data)
        sim_tip = env._env.get_body_com("fingertip")[:2]
        worst = max(worst, float(np.max(np.abs(env.fingertip_pos(s) - sim_tip))))
    assert worst < 1e-10, f"trig fingertip disagrees with the model by {worst:.3e}"


def test_reacher_coverage_targets_match_the_envs_goal_distribution(adapters):
    """`random_state`'s goal comes from `goal_disk`: uniform IN the r=0.2 disk
    the env itself rejection-samples from. Projecting box draws onto the rim
    instead would put 57% of the screens' coverage mass exactly on the boundary
    band where `dwell_fraction`'s threshold sits. Uniform-in-disk puts
    ~9.75% in the outer 0.19-0.20 annulus; asserting < 0.2 separates the two
    regimes with a wide margin on 2000 draws."""
    env = adapters["gym_reacher_reach"]
    rng = np.random.default_rng(7)
    norms = np.asarray([float(np.linalg.norm(env.random_state(rng)[2:4]))
                        for _ in range(2000)])
    assert float(norms.max()) <= 0.2 + 1e-12
    rim = float(np.mean(norms > 0.19))
    assert rim < 0.2, f"{rim:.3f} of targets in the outer annulus; rim atom is back"
    # target velocities come from the row's bounds, their one home
    assert np.all(env.random_state(rng)[6:8] == 0.0)


def test_the_action_set_has_no_duplicate_rows(adapters):
    """At action_dim 1 the all-on/all-off corners coincide with the per-joint
    rows; undeduplicated, the screens' uniform draw weighted |a| = 3 at 4/5
    instead of 2/3 while the rendered abstraction advertised N_ACTIONS = 5 for a
    set of three."""
    for name in ALL_IDS:
        env = adapters[name]
        rows = [tuple(r) for r in env.action_set]
        assert len(rows) == len(set(rows)), f"{name}: duplicate action rows"
    assert adapters["gym_inverted_pendulum_balance"].n_actions == 3
    assert adapters["gym_half_cheetah"].n_actions == 15


def test_task_metric_scores_a_live_episode(adapters):
    """End to end on the pendulum, whose fitness has an exact expectation: the
    fraction of the horizon survived, computed from the same trajectory shape
    `training._rollout` builds (T+1 states)."""
    env = adapters["gym_inverted_pendulum_balance"]
    s, states, done = env.reset(0), [], False
    states.append(s)
    while not done and len(states) <= env.horizon:
        s, done, _ = env.step(s, [3.0])
        states.append(s)
    got = env.task_metric(np.asarray(states))
    assert got == pytest.approx((len(states) - 1) / env.horizon)
    assert 0.0 < got < 0.05, "a full constant push should fall almost immediately"


def test_reference_reward_is_the_wheels_own_shape(adapters):
    """Spot-checks against the transcribed formulas, one per shape."""
    cheetah = adapters["gym_half_cheetah"]
    s = np.zeros(18); s[9] = 2.0
    a = np.asarray([0.5] * 6)
    assert cheetah.reference_reward(s, a) == pytest.approx(2.0 - 0.1 * 6 * 0.25)

    hop = adapters["gym_hopper_hop"]
    s = np.zeros(12); s[1], s[6] = 1.25, 1.5          # healthy, moving forward
    assert hop.reference_reward(s) == pytest.approx(1.0 + 1.5)
    s[1] = 0.5                                        # fallen: no alive bonus
    assert hop.reference_reward(s) == pytest.approx(1.5)

    ip = adapters["gym_inverted_pendulum_balance"]
    up, down = np.zeros(4), np.zeros(4)
    down[1] = 0.5
    assert ip.reference_reward(up) == 1.0 and ip.reference_reward(down) == 0.0

    reach = adapters["gym_reacher_reach"]
    s = np.zeros(8); s[2] = 0.21                      # fingertip on target
    a = np.asarray([0.3, -0.4])
    assert reach.reference_reward(s, a) == pytest.approx(-(0.09 + 0.16))


def test_the_humanoid_reference_reward_is_the_wheels_own_four_terms(adapters):
    """The shipped reward, term by term, against the live env's own decomposition.

    Read off `_get_rew` (humanoid_v5.py:499-518) and checked against
    `info["reward_*"]` rather than against a transcription of the formula: three of
    the four terms are reproduced EXACTLY and the fourth, the forward term, is this
    family's declared finite-difference approximation. So the assertion is
    per-term, not on the total -- a total would hide which term moved."""
    env = adapters["gym_humanoid_run"]
    live = env._env
    rng = np.random.default_rng(0)
    s = env.reset(rng)
    a = rng.uniform(-env.action_clip, env.action_clip, size=env.action_dim)
    s2, _done, _info = env.step(s, a)

    live.set_state(s[:env.n_q], s[env.n_q:])
    live.data.qacc_warmstart[:] = 0.0
    _obs, shipped, _term, _trunc, info = live.step(a)

    # the two exactly-reproduced terms
    assert env.is_upright(s2) == pytest.approx(
        1.0 if info["reward_survive"] > 0 else 0.0)
    assert -0.1 * env.control_cost(a) == pytest.approx(info["reward_ctrl"])

    # THE CONTACT TERM, AGAINST A POISONED BUFFER. The poison is the whole point and
    # must never be removed: `cfrc_ext` is an acceleration-stage array and
    # `mj_forward` reaches it only through the acceleration-stage sensor path, which
    # this model (`nsensor == 0`) does not have. `contact_cost` and `live` are the
    # SAME object, so without the poison this comparison reads the one buffer the
    # wheel's own `step` just filled on both sides and agrees to 0.0 whether or not
    # the method computes anything, and would "verify" an exactness claim that is
    # false. With nan in the buffer, a `contact_cost` that has lost its
    # `mj_rnePostConstraint` returns nan and fails here.
    live.data.cfrc_ext[:] = np.nan
    ours_contact = env.contact_cost(s2, a, 5e-7, 10.0)
    assert np.isfinite(ours_contact), (
        "contact_cost returned a non-finite value from a poisoned cfrc_ext, i.e. it "
        "read the buffer instead of computing it -- mj_rnePostConstraint is missing")
    # and once computed honestly it is an APPROXIMATION, for the same reason the
    # forward term is: the wheel's value describes the pose before the last
    # integration substep, ours the arrived pose. Bound from the measured residual
    # (max 3.0e-03 over a random trajectory), with room to spare and NOT `abs=0.0`.
    assert ours_contact == pytest.approx(-info["reward_contact"], abs=1e-2), (
        f"contact term {ours_contact} against the wheel's {-info['reward_contact']}")

    # the whole reward, whose gap is the two approximations together
    live.data.cfrc_ext[:] = np.nan
    ours = env.reference_reward(s2, a)
    assert np.isfinite(ours)
    assert ours == pytest.approx(shipped, abs=0.2), (
        f"reference_reward {ours} against the wheel's {shipped}; the measured "
        "worst-case gap is 7.6e-02, so a 0.2 miss is a real divergence")


def test_com_velocity_is_the_wheels_own_and_the_root_velocity_is_not(adapters):
    """The measurement the whole row turns on, re-measured here rather than quoted.

    Both gymnasium and REvolve's fork difference `mass_center()` across the step, so
    that -- not `qvel[0]` -- is the quantity Eq. 3 integrates. `com_vx` tracks it
    closely; the root's own x velocity does not merely track it worse, it tracks it
    with the WRONG SIGN, which is why `forward_velocity` pays for a forward pass.

    Asserted as a correlation gap rather than a tolerance: the point is not that
    `s[n_q]` is imprecise but that it is anti-correlated, and a tolerance would pass
    the day someone "simplified" the helper on a trajectory that happened to agree."""
    env = adapters["gym_humanoid_run"]
    live = env._env
    fd, com, root = [], [], []
    s = env.reset(np.random.default_rng(0))
    for _ in range(40):
        live.set_state(s[:env.n_q], s[env.n_q:])
        live.data.qacc_warmstart[:] = 0.0
        _o, _r, term, _tr, info = live.step(np.zeros(env.action_dim))
        s2 = np.concatenate([live.data.qpos, live.data.qvel]).copy()
        fd.append(float(info["x_velocity"]))
        com.append(env.com_vx(s2))
        root.append(float(s2[env.n_q]))
        s = s2
        if term:
            break
    assert len(fd) >= 20, "too short a rollout to say anything about a correlation"
    fd, com, root = np.array(fd), np.array(com), np.array(root)
    assert np.corrcoef(com, fd)[0, 1] > 0.99, np.corrcoef(com, fd)[0, 1]
    assert np.corrcoef(root, fd)[0, 1] < 0.5, (
        "s[n_q] agreed with the env's own forward velocity on this trajectory; the "
        "row's com_velocity key and the -0.39 measurement it rests on need re-checking")
    assert np.max(np.abs(com - fd)) < np.max(np.abs(root - fd))
    # and `forward_velocity` is the helper that routes to it, on this row only
    assert env.forward_velocity(s) == pytest.approx(env.com_vx(s))
    cheetah = adapters["gym_half_cheetah"]
    s_c = cheetah.reset(np.random.default_rng(0))
    assert cheetah.forward_velocity(s_c) == float(s_c[cheetah.n_q]), (
        "the planar rows must keep reading the root's own velocity -- com_velocity "
        "is a per-family key, not a change of definition for everyone")


def test_the_reference_reward_does_not_perturb_the_next_step(adapters):
    """`gym_humanoid_run` is the first row here whose `reference_reward` restores
    state into the simulator, and `training._rollout` calls it BETWEEN two `step`s
    (`gt_return += env.reference_reward(s2, a)` inside the loop). So the question is
    not whether the call is pure but whether it leaves the NEXT step unchanged.

    It does, and the reason is the same one the stateless contract rests on: `_step`
    re-restores qpos/qvel and zeroes `qacc_warmstart` before every step. Asserted
    BITWISE over a real rollout, because an approximate answer here has no other
    symptom -- the curves would just quietly measure physics that never happened,
    and on a contact-rich body an intervening call is exactly what moved scores on
    the HumanoidBench tier (see the comment beside the in-loop
    `reference_reward` call in `bird/policy_api.py`).

    Run on every family, not just this one: the property is what makes the in-loop
    call safe at all, and a future row that broke it would break it silently."""
    for name in FAMILY_REPS:
        env = adapters[name]
        rng = np.random.default_rng(5)
        s = env.reset(rng)
        clean, interleaved = [], []
        acts = [rng.uniform(-env.action_clip, env.action_clip, size=env.action_dim)
                for _ in range(12)]
        pokes = 0
        for label, poke in (("clean", False), ("interleaved", True)):
            cur = env.reset(np.random.default_rng(5))
            out = []
            for a in acts:
                s2, done, _info = env.step(cur, a)
                out.append(s2.copy())
                if poke and env.has_reference_reward:
                    env.reference_reward(s2, a)
                    pokes += 1
                cur = s2
                if done:
                    break
            (clean if label == "clean" else interleaved).append(out)
        # The guard against a VACUOUS pass: the
        # `has_reference_reward` condition is there because five gym specs declare
        # `reward.human.kind: none` and `reference_reward` raises on them -- but if it
        # were ever False on a row in FAMILY_REPS, this test would compare two
        # identical un-poked runs and pass while proving nothing. Every row in
        # FAMILY_REPS has a reference reward today, and this is what says so.
        assert pokes > 0, (
            f"{name}: no interleaved reference_reward call was made, so the two runs "
            "are identical by construction and this test proved nothing")
        a_run, b_run = clean[0], interleaved[0]
        assert len(a_run) == len(b_run), name
        for k, (x, y) in enumerate(zip(a_run, b_run)):
            dev = float(np.max(np.abs(x - y)))
            assert dev == 0.0, (
                f"{name}: an interleaved reference_reward moved step {k} by {dev:.3e}")


def test_random_state_hands_the_simulator_a_real_rotation(adapters):
    """A unit quaternion cannot be drawn componentwise from a box, and MuJoCo will
    not fix it: measured, `set_state` round-trips a norm-2.0 quaternion at norm 2.0.
    So without the row's `quat_qpos` normalisation the screens' coverage would be
    taken over orientations that are not rotations at all.

    Also asserts the five planar rows are UNCHANGED by the key -- they declare None,
    and a normalisation applied to a slide/hinge root would corrupt three real
    coordinates."""
    env = adapters["gym_humanoid_run"]
    qi = _FAMILIES["Humanoid-v5"]["quat_qpos"]
    assert qi == 3
    for seed in range(8):
        s = env.random_state(np.random.default_rng(seed))
        assert s.shape == (env.obs_dim,)
        assert float(np.linalg.norm(s[qi:qi + 4])) == pytest.approx(1.0, abs=1e-12), seed
        # x and y are pinned (cyclic), z is not
        assert s[0] == 0.0 and s[1] == 0.0, seed
        # and the state restores and steps without blowing up
        s2, _done, _info = env.step(s, np.zeros(env.action_dim))
        assert np.isfinite(s2).all(), seed

    # MuJoCo really does not renormalise for us -- the measurement the key exists for
    live = env._env
    probe = env.reset(np.random.default_rng(0))
    bad = probe.copy()
    bad[qi:qi + 4] = np.array([2.0, 0.0, 0.0, 0.0])
    live.set_state(bad[:env.n_q], bad[env.n_q:])
    assert float(np.linalg.norm(live.data.qpos[qi:qi + 4])) == pytest.approx(2.0)

    for name in ("gym_half_cheetah", "gym_hopper_hop", "gym_swimmer_forward",
                 "gym_reacher_reach", "gym_inverted_pendulum_balance"):
        row = _FAMILIES[adapters[name]._gym_id]
        assert row["quat_qpos"] is None, name


def test_the_humanoid_field_table_matches_the_live_model(adapters):
    """Every one of the 47 state entries and 17 action entries at the index the MODEL
    puts it, read off `jnt_qposadr` / `jnt_dofadr` / the actuator names rather than
    transcribed.

    `test_the_row_tables_describe_the_row_they_sit_in` pins the table's WIDTH, and
    width is not correctness: a 47-entry table with two entries transposed fails
    nothing, and a candidate is then handed the wrong slot under a confident name --
    the same class of error as the reference reward that measured forward speed on a
    backward task, and just as invisible. This is the `fingertip_trig_matches_the_
    live_model` pattern applied to a table instead of a constant.

    Three specific traps it exists for, all of which the author hit while writing the
    row: the free root spends qpos[0:7] so the hinges start at 7 and not at 3;
    qvel's root block is 3 linear THEN 3 angular, so the hinge velocities start at
    n_q + 6 = 30; and ACTUATOR ORDER IS NOT JOINT ORDER -- a[0] drives abdomen_y
    (s[8]) while a[1] drives abdomen_z (s[7]), so a table built by zipping the two
    lists is wrong in exactly its first two rows and nowhere else.
    """
    env = adapters["gym_humanoid_run"]
    model, mj = env._env.model, env._mj
    row = _FAMILIES["Humanoid-v5"]
    state = list(row["state_fields"])
    action = list(row["action_fields"])

    def jname(i):
        return mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)

    # every entry self-labels its own index, in order
    for i, (name, desc) in enumerate(state):
        assert desc.startswith(f"s[{i}]"), (i, name, desc[:40])
    for i, (name, desc) in enumerate(action):
        assert desc.startswith(f"a[{i}]"), (i, name, desc[:40])

    # FIRST: the committed literal against the model, which is what makes the
    # simulator-free test above trustworthy. Without this the literal is just a second
    # copy of the same authoring mistake.
    assert len(HUMANOID_JOINTS) == int(model.njnt) - 1
    assert len(HUMANOID_ACTUATORS) == int(model.nu)
    for (jn, qadr, dadr), j in zip(HUMANOID_JOINTS, range(1, int(model.njnt))):
        assert jname(j) == jn, (j, jname(j), jn)
        assert int(model.jnt_qposadr[j]) == qadr, (jn, int(model.jnt_qposadr[j]), qadr)
        assert int(model.jnt_dofadr[j]) == dadr, (jn, int(model.jnt_dofadr[j]), dadr)
    for i, (an, drives, gear) in enumerate(HUMANOID_ACTUATORS):
        nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        assert nm == an, (i, nm, an)
        j = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, nm))
        assert int(model.jnt_qposadr[j]) == drives, (an, int(model.jnt_qposadr[j]), drives)
        assert int(model.actuator_gear[i][0]) == gear, (an, model.actuator_gear[i][0], gear)

    # the hinges, at the model's own addresses. Joint 0 is the free root: skipped,
    # and skipped by TYPE rather than by index so a remodelled root fails here.
    assert int(model.jnt_type[0]) == int(mj.mjtJoint.mjJNT_FREE)
    assert int(model.jnt_limited[0]) == 0, (
        "the free root gained a limit -- `_bounds` writes only jnt_range[0] and would "
        "now describe one of seven coordinates")
    for j in range(1, int(model.njnt)):
        nm = jname(j)
        qadr, dadr = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        assert state[qadr][0] == f"{nm}_angle", (qadr, nm, state[qadr][0])
        assert state[env.n_q + dadr][0] == f"{nm}_velocity", (
            env.n_q + dadr, nm, state[env.n_q + dadr][0])

    # the actuators, in the model's order, with the gear each description quotes
    for i in range(int(model.nu)):
        nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        assert action[i][0] == f"{nm}_torque", (i, nm, action[i][0])
        assert f"gear {int(model.actuator_gear[i][0])}" in action[i][1], (i, nm, action[i][1])
        # where a description claims which slot it drives, the model decides
        m = re.search(r"drives s\[(\d+)\]", action[i][1])
        if m:
            j = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, nm))
            assert int(m.group(1)) == int(model.jnt_qposadr[j]), (i, nm, m.group(1))

    # and the slices bracket exactly those two blocks
    js, vs = row["joint_slice"], row["joint_vel_slice"]
    assert js == (7, 24) and vs == (30, 47), (js, vs)
    assert js[1] - js[0] == vs[1] - vs[0] == int(model.nu) == 17
    assert row["symbol_mapping"]["joint_angles"] == f"s[{js[0]}:{js[1]}]"
    assert row["symbol_mapping"]["joint_velocities"] == f"s[{vs[0]}:{vs[1]}]"
    assert row["symbol_mapping"]["torso_height"] == f"s[{row['height_index']}]"


def test_the_humanoid_reset_is_the_wheels_own_non_unit_quaternion(adapters):
    """The H1-tier trap, checked NOT to occur here.

    `reset_model` adds its uniform noise to all 24 qpos including the quaternion, so
    the reset state's |q| is genuinely off 1.0 -- that is upstream's behaviour. It is
    harmless here only because `set_state` leaves `data.qpos` exactly as given, so
    the state `_reset` returns is restorable bit for bit. A restore path that DID
    normalise would start every episode off the manifold, silently changing the
    reset distribution of every run. If a future mujoco starts normalising in
    `set_state`, this test is what says so."""
    env = adapters["gym_humanoid_run"]
    qi = 3
    norms = []
    for seed in range(8):
        s = env.reset(np.random.default_rng(seed))
        norms.append(float(np.linalg.norm(s[qi:qi + 4])))
        # the whole point: the sim holds exactly what reset handed back
        assert np.array_equal(
            s, np.concatenate([env._env.data.qpos, env._env.data.qvel])), seed
        assert np.array_equal(s, env.reset(np.random.default_rng(seed))), seed
    assert any(abs(n - 1.0) > 1e-6 for n in norms), (
        "every reset quaternion was unit -- either the wheel stopped perturbing it "
        "or set_state started normalising; both change the reset distribution the "
        "spec documents")
    assert all(abs(n - 1.0) < 0.05 for n in norms), norms


# --------------------------------------------------------------------------
# what the model is shown
# --------------------------------------------------------------------------


class _Ctx:
    def __init__(self, env):
        self.env = env


def test_full_source_shows_the_environment_not_the_plumbing(adapters):
    """The base render is `inspect.getsource(type(self))` -- ~1000 lines of
    five-family machinery in which no per-task fact appears, shown to eureka and
    rda by default (`env_spec: full_source`). The override must carry: the field
    table for THIS family (which `s[i]` is what), the gymnasium class's real
    source, and this adapter's reference reward -- and must NOT carry the other
    families' helpers."""
    env = adapters["gym_hopper_hop"]
    text = env.describe("full_source")
    assert "z_height" in text, "the hopper field table is missing"
    assert "HopperEnv" in text, "the gymnasium class source is missing"
    assert "def reference_reward" in text
    assert "def _get_rew" in text, "the env source should arrive intact pre-strip"
    assert "fingertip_pos" not in text, "another family's helpers leaked in"
    assert "_FAMILIES" not in text, "the adapter's plumbing leaked in"


def test_the_strip_removes_both_copies_of_the_reward(adapters):
    """Two mechanisms, both needed: `_REWARD_DEF` catches `def reference_reward`
    by name, and `reward_source` hands over the verbatim `_get_rew` span --
    which the reward-name regex does NOT match, so without it the one thing
    `strip_existing_reward` exists to withhold (the published human reward)
    rides into the prompt inside the env source."""
    from bird.components.generation import _strip_reward

    env = adapters["gym_hopper_hop"]
    text = env.describe("full_source")
    stripped = _strip_reward(_Ctx(env), text)
    assert "def reference_reward" not in stripped
    assert "def _get_rew" not in stripped
    assert "forward_reward_weight * x_velocity" not in stripped

    # InvertedPendulum's reward is a line inside `step` (no `_get_rew` exists),
    # so its span is the whole step method
    ip = adapters["gym_inverted_pendulum_balance"]
    assert "_get_rew" not in ip.describe("full_source")
    ip_stripped = _strip_reward(_Ctx(ip), ip.describe("full_source"))
    assert "int(not terminated)" not in ip_stripped


def test_the_other_renderings_stay_task_specific(adapters):
    for name in ("gym_hopper_hop", "gym_reacher_hold"):
        env = adapters[name]
        for kind in ("natural_language_only", "state_action_api_stub",
                     "pythonic_class_abstraction"):
            assert env.describe(kind).strip(), (name, kind)
    assert env.describe("none") == ""


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_render_returns_a_frame_of_the_state_it_was_given(adapters, gl):
    for name, env in adapters.items():
        rng = np.random.default_rng(7)
        s0 = env.reset(rng)
        s1, _done, _ = env.step(s0, env.action_set[1])
        # move the simulator somewhere else, then render the STORED states
        env.step(env.reset(np.random.default_rng(8)), env.action_set[0])
        frames = [env.render(s) for s in (s1, s0)]
        for frame in frames:
            assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
            assert frame.std() > 1.0, f"{name}: a uniform frame means nothing rendered"


def test_render_takes_exactly_one_positional_state(adapters):
    """`observability._accepts_a_state` gates on `signature.bind(object())`; a
    renderer that fails it is skipped with a warning and the run records no
    video at all."""
    from bird.observability import _accepts_a_state

    for env in adapters.values():
        assert _accepts_a_state(env.render)


# --------------------------------------------------------------------------
# the DR contract
# --------------------------------------------------------------------------


def test_every_gym_spec_declares_axes_this_module_can_write():
    """Spec == adapter, the gym half of `test_task_specs.py`'s MT10 rule -- checked here
    without a simulator, because `_GYM_AXES` is a module table and a spec block is data.
    Each spec declares exactly the axes with a mechanical path on ITS model (no fluid on a
    legged model, no collider on Reacher), and the nominal sits strictly inside its bounds
    so a DR band can still be centred on it."""
    from bird.envs.gym_mujoco import _GYM_AXES

    for tid, spec in _gym_specs().items():
        dr = spec.domain_randomization
        assert dr, f"{tid}: every gym spec must carry a DR block"
        assert set(dr["parameters"]) <= set(_GYM_AXES), tid
        assert set(dr["parameters"]) == set(dr["nominal"]), tid
        for axis, (lo, hi) in dr["parameters"].items():
            assert lo < dr["nominal"][axis] < hi, (tid, axis)
        fam = spec.env_id
        if fam == "Swimmer-v5":
            assert "viscosity_scale" in dr["parameters"] and "contact_friction_scale" not in dr["parameters"]
        else:
            assert "viscosity_scale" not in dr["parameters"]
        if fam in ("Reacher-v5", "InvertedPendulum-v5", "Swimmer-v5"):
            assert "contact_friction_scale" not in dr["parameters"], "no colliding geoms there"


def test_the_instance_tables_are_the_specs(adapters):
    from bird.envs.spec import dr_of

    for env_id in ALL_IDS:
        env = adapters[env_id]
        ranges, nominal = dr_of(env._gym_spec if hasattr(env, "_gym_spec") else _gym_specs()[env.task])
        assert env.dr_parameters == ranges and env._dr_nominal == nominal, env_id


def test_a_nominal_adapter_is_bitwise_the_compiled_model(adapters):
    """`_apply_dr` runs on every reset writing `1.0 x nominal`; at nominal that must
    change no model entry, or every gym run would train on physics that differs from the
    compiled model."""
    from bird.envs.gym_mujoco import _DR_MODEL_FIELDS

    for env_id in ALL_IDS:
        env = registry.get("env", env_id)({})
        model = env._env.model
        compiled = {f: np.copy(getattr(model, f)) for f in _DR_MODEL_FIELDS}
        gravity, viscosity = np.copy(model.opt.gravity), float(model.opt.viscosity)
        env.reset(np.random.default_rng(0))
        for f, arr in compiled.items():
            assert np.array_equal(getattr(model, f), arr), (env_id, f)
        assert np.array_equal(model.opt.gravity, gravity), env_id
        assert float(model.opt.viscosity) == viscosity, env_id


def test_a_fixed_draw_moves_exactly_the_arrays_its_axis_names(adapters):
    """mass x2 on the whole plant: every non-world body's mass AND inertia; the floor's
    friction scaled, the gear untouched -- and the draw holds through a step. A bare
    scalar is a fixed DR value (`EnvAdapter.set_dr`)."""
    env = registry.get("env", "gym_hopper_hop")({})
    env.set_dr({"body_mass_scale": 2.0, "contact_friction_scale": 0.5})
    s = env.reset(np.random.default_rng(0))
    model = env._env.model
    assert np.allclose(model.body_mass[1:], 2.0 * env._nominal["body_mass"][1:])
    assert np.allclose(model.body_inertia[1:], 2.0 * env._nominal["body_inertia"][1:])
    assert np.allclose(model.geom_friction[:, 0], 0.5 * env._nominal["geom_friction"][:, 0])
    assert np.array_equal(model.actuator_gear, env._nominal["actuator_gear"])
    env.step(s, np.zeros(env.action_dim))
    assert np.allclose(model.body_mass[1:], 2.0 * env._nominal["body_mass"][1:])
    swim = registry.get("env", "gym_swimmer_forward")({})
    swim.set_dr({"viscosity_scale": 2.0})
    swim.reset(np.random.default_rng(0))
    assert swim._env.model.opt.viscosity == 2.0 * swim._nominal["viscosity"]


# --------------------------------------------------------------------------
# the per-step success flag, with the simulator
# --------------------------------------------------------------------------


def test_step_reports_a_constant_false_success_flag(adapters):
    """`_step` returns a constant `{"success": False}` on every task -- its docstring:
    "that is the spec speaking rather than a stub". Every spec here is
    `discrete_success.kind: continuous_only` and gymnasium ships no success check, so
    the flag is not a measurement and must not start pretending to be one."""
    rng = np.random.default_rng(3)
    for task_id in _EXPECTED_GYM:
        env = adapters[_env_id(task_id)]
        s = env.reset(rng)
        _s2, _done, info = env.step(s, env.action_set[0])
        assert info == {"success": False}, (task_id, info)
