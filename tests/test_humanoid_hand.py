"""The twenty-five Shadow-Hands HumanoidBench tasks -- four selected for their defective
shipped rewards, and twenty-one more -- and the claims they are wrong without.

TWO TIERS OF TEST IN ONE FILE, and the split is deliberate.

`HumanoidBench cannot be installed alongside metaworld` -- it needs `mujoco==3.1.6` and
metaworld 3.1.1 pins `mujoco==3.3.0` exactly -- so nothing that CONSTRUCTS one of these
envs can run in the repo's main venv. Those tests carry `@pytest.mark.humanoid`
individually and are DESELECTED there rather than skipped (`tests/conftest.py::env_param`
and `pyproject.toml` carry that argument in full).

What is NOT individually marked is everything checkable from the task spec and the class
body without a simulator: the id derivation, the spec/class shape agreement, the metric
arithmetic on synthetic trajectories, and the leak gate. Those RUN in the main venv
whenever `slow` is selected, and they are the only automated cover this tier has. That
is why they are here rather than folded into the marked fixture: a module marked as a
whole has no unmarked half at all, and the price is that a spec edit that contradicts
the adapter is caught by nothing until someone stages the venv.

The module is `slow` (a dependency-gated file; deselected by `-m "not slow"`).
The metric tests are the ones to read first: each asserts that its task's own DOCUMENTED
EXPLOIT scores zero, which is the property the four tasks were selected for and the one
a plausible-looking refactor would quietly remove.

Running the marked half is a manual act in the tier's own venv
(`bash scripts/setup_humanoid.sh` builds it):

    PYTHONPATH=. .venv-humanoid/bin/python -m pytest tests/test_humanoid_hand.py
"""

from types import SimpleNamespace

import numpy as np
import pytest

from bird import registry, tasks
from bird.envs import humanoid_hand as HH
from bird.envs import hud

#: Not in the tester-tier smoke suite. A LIST, not two assignments: `pytestmark = a`
#: followed by `pytestmark = b` keeps only `b`, silently dropping the first mark. See
#: pyproject.toml for what each marker buys.
pytestmark = [pytest.mark.slow]

#: (BIRD env id, adapter class). The registry is the authority on the ids; this pairs
#: them with the classes so a sim-free test can read class attributes.
ADAPTERS = (
    ("h1hand_powerlift", HH.H1HandPowerlift),
    ("h1hand_hurdle", HH.H1HandHurdle),
    ("h1hand_room", HH.H1HandRoom),
    ("h1strong_highbar_hard", HH.H1StrongHighBarHard),
    ("h1hand_push", HH.H1HandPush),
    ("h1hand_basketball", HH.H1HandBasketball),
    ("h1hand_walk", HH.H1HandWalk),
    ("h1hand_crawl", HH.H1HandCrawl),
    ("h1hand_sit_simple", HH.H1HandSitSimple),
    ("h1hand_package", HH.H1HandPackage),
    ("h1hand_door", HH.H1HandDoor),
    ("h1hand_pole", HH.H1HandPole),
    ("h1hand_maze", HH.H1HandMaze),
    ("h1hand_sit_hard", HH.H1HandSitHard),
    ("h1hand_reach", HH.H1HandReach),
    ("h1hand_balance_simple", HH.H1HandBalanceSimple),
    ("h1hand_balance_hard", HH.H1HandBalanceHard),
    ("h1hand_window", HH.H1HandWindow),
    ("h1hand_cube", HH.H1HandCube),
    ("h1hand_cabinet", HH.H1HandCabinet),
    ("h1hand_bookshelf_hard", HH.H1HandBookshelfHard),
)
IDS = [e for e, _ in ADAPTERS]

#: The four selected FOR their defective shipped rewards -- BY NAME, not by position
#: in ADAPTERS, so reordering or inserting a task cannot silently move the stricter
#: requirement onto the wrong envs. The other tasks were not selected on that
#: criterion, so only these four are REQUIRED to carry
#: an `open` exploit -- sit_simple's one honest entry is `undetectable_here`, and
#: inventing an open one to satisfy a test would be exactly the fabrication the specs
#: refuse.
DEFECT_SELECTED = frozenset(
    ("h1hand_powerlift", "h1hand_hurdle", "h1hand_room", "h1strong_highbar_hard"))
assert DEFECT_SELECTED <= set(IDS)

#: Which tasks a COLLAPSED robot terminates on. Upstream's own `get_terminated`s:
#: push and package end only on success (no failure terminal at all), crawl never ends,
#: and reach has NO terminal of any kind -- its `get_terminated` is the base class's
#: constant False.
FALLS_TERMINATE = frozenset(IDS) - {"h1hand_push", "h1hand_crawl", "h1hand_package",
                                    "h1hand_reach", "h1hand_cabinet"}

#: Tasks whose episodes can never end before the horizon at all -- the spec's
#: `termination.early_terminate` must be False for exactly these.
NEVER_TERMINATES = frozenset({"h1hand_crawl", "h1hand_reach"})

#: `reference_reward`'s codomain, per task. The four defect-selected rewards and the three
#: locomotion/posture rewards are products/sums of tolerance() kernels in [0, 1];
#: push is a dense NEGATIVE distance shaping with a +1000 success bonus (so nothing
#: bounds it below), and basketball's kernel sum carries the same +1000 spike.
REFERENCE_REWARD_RANGE = {
    "h1hand_push": (-np.inf, 1000.0),
    "h1hand_basketball": (0.0, 1001.0),
    # Package is ADDITIVE with an unbounded-below `-3 * dist` term and a +1000 success
    # bonus on top of stand_reward * small_control (<=1) and package_height (<=1).
    "h1hand_package": (-np.inf, 1002.0),
    # Door is a weighted sum of tolerance kernels whose weights total exactly 1.0.
    "h1hand_door": (0.0, 1.0),
    # Maze's kernels total 1.0 but each stage advance pays a one-off +100*stage on top,
    # so the largest single step is the 3->4 transition: 1 + 400.
    "h1hand_maze": (0.0, 401.0),
    # Reach is ADDITIVE: healthy in [-5, 5], an unbounded-below motion penalty, +5
    # within a metre and +10 within 0.05 -- so at most 20 and no floor.
    "h1hand_reach": (-np.inf, 20.0),
    # Cabinet's rung shaping stays in [0, 1] and each advance adds +100*rung (the
    # largest transition step is 1 + 400), but a rung-five state pays a flat 1000.
    "h1hand_cabinet": (0.0, 1000.0),
    # Bookshelf's kernels total 1.0 and the final placement's one-off is +100*5.
    "h1hand_bookshelf_hard": (0.0, 501.0),
}
_DEFAULT_RANGE = (0.0, 1.0)


@pytest.fixture(scope="module")
def sim_envs():
    """Every one of the four, constructed. Module-scoped: each is a MuJoCo model build."""
    pytest.importorskip("humanoid_bench")
    registry.load_all()
    return {e: registry.get("env", e)({}) for e in IDS}


# --------------------------------------------------------------------------
# what the ids are, and the one that does not exist
# --------------------------------------------------------------------------


def test_the_four_tasks_are_registered_and_the_ids_derive_from_the_gym_ids():
    """`h1hand-powerlift-v0` -> `h1hand_powerlift`: `-v0` dropped, `-` to `_`.

    A DERIVATION, not a second spelling, for the reason `mt10_` is a derivation: an id
    that resolves is an id `gym.make` accepts. This pins both halves
    -- the rule in `bird/tasks.py::_BIRD_ID_RULES` and the `gym_id` each class carries --
    against each other, so a class whose `gym_id` drifts from its registry name fails
    here rather than at `gym.make` in hour one of a sweep.
    """
    registry.load_all()
    names = set(registry.names("env"))
    for env_id, cls in ADAPTERS:
        assert env_id in names, f"{env_id} is not registered"
        assert cls.gym_id.removesuffix("-v0").replace("-", "_") == env_id
        assert tasks.by_env_id(env_id).bird_env_id == env_id


def test_the_highbar_task_is_h1strong_because_no_h1hand_highbar_asset_exists():
    """Why the fourth defect-selected id is not the obvious one.

    `humanoid_bench/__init__.py` registers the full cross product of ROBOTS x TASKS, so
    `h1hand-highbar_hard-v0` and `h1hand-highbar_simple-v0` both REGISTER and neither can
    be constructed: `assets/envs/` has only `h1_pos_highbar_simple.xml` and
    `h1strong_pos_highbar_hard.xml`. Guessing `h1hand-highbar_hard-v0` from the pattern
    would have produced an id that passes every registry check in this repo and then dies
    at `gym.make` on a cluster node.

    Asserted here as a NAME check rather than a construction, so it runs without the
    simulator -- which is the configuration in which the wrong guess would have been
    made.
    """
    assert HH.H1StrongHighBarHard.gym_id == "h1strong-highbar_hard-v0"
    assert "h1hand" not in HH.H1StrongHighBarHard.gym_id
    assert HH.H1StrongHighBarHard.n_q == 76, (
        "h1strong shares the Shadow-Hands embodiment with h1hand: 76 qpos and 61 "
        "actuators, differing only in the finger servos' gains. If this is 26 the id has "
        "silently become the plain-H1 highbar, which is a different robot.")


@pytest.mark.parametrize("env_id,cls", ADAPTERS, ids=IDS)
def test_the_class_and_its_task_spec_agree_on_shape(env_id, cls):
    """`obs_dim`, `action_dim` and `horizon` live in both places legitimately -- the class
    sizes its action set and bounds from them, the spec renders the API stub from them --
    so the contract is that they AGREE.

    `SpecEnvAdapter._apply_spec` asserts the same thing, but only at CONSTRUCTION, which
    on this tier means never in CI. This is the copy that runs.
    """
    spec = tasks.load(env_id)
    spaces = spec.env["spaces"]
    assert cls.obs_dim == int(spaces["obs_dim"]) == cls.n_q + cls.n_v + cls.extra_dims
    assert cls.action_dim == int(spaces["action_dim"]) == 61
    assert cls.horizon == int(spec.env["horizon"])
    assert len(spec.state_surface["flat_fields"]) == cls.obs_dim
    assert len(spaces["action_fields"]) == cls.action_dim


@pytest.mark.parametrize("env_id", IDS)
def test_each_spec_records_its_reward_defect_as_data(env_id):
    """The defect is in `exploits`, not only in prose.

    The loader exposes these entries as data, so a defect recorded only as a paragraph in
    a docstring is invisible to anything that reads the spec. Every spec on this tier must carry at least one entry with a real
    mechanism; the four DEFECT_SELECTED tasks must additionally keep one `open` -- they
    were selected FOR their open exploits, and an all-`closed` list would mean the task
    no longer does the job it was added for. The other tasks are NOT held to that:
    sit_simple's one honest entry is `undetectable_here`, and requiring `open` there
    would force an invented defect into a data file.
    """
    spec = tasks.load(env_id)
    exploits = spec.exploits
    assert exploits, f"{env_id}: no exploits recorded"
    if env_id in DEFECT_SELECTED:
        assert any(e["status"] == "open" for e in exploits), (
            f"{env_id}: every exploit is closed. These four tasks exist because their "
            "shipped reward is gameable; if that is no longer true the task should be "
            "re-justified rather than kept.")
    for e in exploits:
        assert e["mechanism"].strip(), f"{env_id}: exploit {e['name']} has no mechanism"


@pytest.mark.parametrize("env_id", IDS)
def test_the_ground_truth_is_gated_out_of_the_generation_prompt(env_id):
    """`verify.forbidden_symbols` is inherited from the spec, and it must name every
    route to the fitness.

    `success_bar` is the one that is easy to leave out and the one that matters most
    here: HumanoidBench declares it on every task class, nothing upstream reads it, and a
    candidate that found it would be reading a threshold on the shipped reward's own
    episode return -- the circularity these specs reject by construction.

    AND `success` MUST NOT BE IN IT, which is the half this test exists to hold. Gating
    it on a live run on this tier rejected three of four candidates at iteration 1, every
    one for naming its own proxy `success` -- the same result `tasks/drawer_close`
    records measuring at 15 of 15. A word gate here rejects
    honest candidates and stops nothing: the defence is structural, because `success()`
    is a threshold on `task_metric`, which IS gated.
    """
    from bird.config import forbidden_symbols_of

    gate = set(forbidden_symbols_of(tasks.load(env_id)))
    assert {"task_metric", "reference_reward", "success_bar"} <= gate
    assert "success" not in gate, (
        f"{env_id}: `success` is back in the leak gate. Measured twice now -- 3 of 4 "
        "candidates here and 15 of 15 on Meta-World -- it rejects candidates for naming "
        "their own proxy and blocks no route the structural argument does not already "
        "cover. See the spec's forbidden_symbols_note before re-adding it.")


# --------------------------------------------------------------------------
# the metrics, on synthetic trajectories -- no simulator needed
# --------------------------------------------------------------------------
#
# These are the tests to read. Each builds the behaviour its task's `exploits` entry
# describes and asserts the ground truth scores it ZERO, then builds the behaviour the
# task asks for and asserts it scores above zero. A metric that stopped distinguishing
# the two would leave every run on this tier reporting a number that agrees with the
# reward it was built to disagree with.


def _metric(cls, states):
    """`task_metric` called UNBOUND, with `self` as None.

    Legitimate, and load-bearing for this half of the file: all four metrics read only
    the trajectory and the module-level scene constants, so none of them touches `self`
    and none needs a simulator. That is not an accident -- the constants are at module
    scope precisely so `describe("full_source")`, which renders the CLASS, does not ship
    the metric's geometry to the generator -- and calling unbound is what lets the
    metric arithmetic be tested in a configuration where the env cannot be built.

    If a metric ever grows a `self.` reference these calls raise `AttributeError`, which
    is the right failure: it says the sim-free cover just went away.
    """
    return cls.task_metric(None, states)


def _blank(cls, n=5):
    """`n` copies of a plausible resting state for `cls`: pelvis standing at 0.98 m,
    identity quaternion, everything else zero."""
    s = np.zeros((n, cls.obs_dim))
    s[:, 2] = 0.98
    s[:, 3] = 1.0
    return s


def test_powerlift_pays_nothing_for_standing_still_beside_the_barbell():
    states = _blank(HH.H1HandPowerlift)
    states[:, 78] = 0.20                       # the barbell where the keyframe leaves it
    assert _metric(HH.H1HandPowerlift, states) == 0.0

    states[1:, 78] = 1.40                      # lifted above the standing pelvis
    assert _metric(HH.H1HandPowerlift, states) == pytest.approx(1.0)

    # AND THE STANDING CONJUNCT, which the two cases above do not exercise: a robot
    # lying on the floor under a barbell propped on something satisfies "barbell above
    # the pelvis" trivially, because the pelvis is at ankle height. Without this the
    # metric would pay for collapsing.
    collapsed = _blank(HH.H1HandPowerlift)
    collapsed[:, 2] = 0.10
    collapsed[:, 78] = 1.40
    assert _metric(HH.H1HandPowerlift, collapsed) == 0.0


def test_hurdle_pays_nothing_for_bracing_short_of_the_first_wall():
    states = _blank(HH.H1HandHurdle)
    states[:, 0] = 6.9                         # up against the x = 7 wall, not over it
    assert _metric(HH.H1HandHurdle, states) == 0.0

    states[:, 0] = 15.0                        # past the walls at 7 and 14
    assert _metric(HH.H1HandHurdle, states) == pytest.approx(0.2)


def test_hurdle_counts_the_high_water_mark_and_not_the_final_position():
    """A hurdle jumped and then fallen back behind was still cleared.

    Scoring the final x would rank a robot that stumbles after hurdle three below one
    that never left the start, which inverts the ordering the metric exists to produce.
    """
    states = _blank(HH.H1HandHurdle, n=4)
    states[:, 0] = [0.0, 8.0, 22.0, 3.0]
    assert _metric(HH.H1HandHurdle, states) == pytest.approx(0.3)


def test_room_pays_nothing_for_a_heap_on_the_floor():
    """The heap is exactly what maximises the shipped reward's variance term."""
    states = _blank(HH.H1HandRoom, n=3)
    for k in range(6):                          # all six in one pile, on the floor
        states[:, HH._ROOM_OBJ_Q0 + 7 * k: HH._ROOM_OBJ_Q0 + 7 * k + 3] = [1.0, 1.0, 0.05]
    assert _metric(HH.H1HandRoom, states) == 0.0

    for k in range(3):                          # three stowed on the shelf
        states[-1, HH._ROOM_OBJ_Q0 + 7 * k: HH._ROOM_OBJ_Q0 + 7 * k + 3] = [2.7, 0.5, 0.65]
    assert _metric(HH.H1HandRoom, states) == pytest.approx(0.5)


def test_room_does_not_count_an_object_shoved_under_the_table():
    """The table's footprint test needs its height conjunct: the table top is at 0.95 m
    and the floor under it is inside the same x-y box."""
    assert HH._stowed(np.array([2.7, -0.5, 0.05])) is False
    assert HH._stowed(np.array([2.7, -0.5, 1.00])) is True


def test_highbar_pays_nothing_for_hanging_and_nothing_for_swinging_back_and_forth():
    """The signed sum is what makes the second half true, and it is the one piece of
    metric design in this file that does real work.

    A metric summing |dtheta| would score a vigorous swing highly and hide precisely the
    behaviour the task's `exploits` entry names as the natural local optimum.
    """

    hang = _blank(HH.H1StrongHighBarHard, n=10)
    hang[:, 0], hang[:, 2] = 0.27, 1.62
    assert _metric(HH.H1StrongHighBarHard, hang) == 0.0

    phase = np.concatenate([np.linspace(-np.pi / 2, 0.0, 20),
                            np.linspace(0.0, -np.pi, 20)])
    swing = _blank(HH.H1StrongHighBarHard, n=len(phase))
    swing[:, 0] = HH._BAR_X + 1.18 * np.cos(phase)
    swing[:, 2] = HH._BAR_Z + 1.18 * np.sin(phase)
    assert _metric(HH.H1StrongHighBarHard, swing) < 0.3, "a swing that returns is not a rotation"


def test_highbar_scores_a_full_rotation_at_one():
    phase = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi, 60)
    turn = _blank(HH.H1StrongHighBarHard, n=len(phase))
    turn[:, 0] = HH._BAR_X + 1.18 * np.cos(phase)
    turn[:, 2] = HH._BAR_Z + 1.18 * np.sin(phase)
    assert _metric(HH.H1StrongHighBarHard, turn) == pytest.approx(1.0, abs=1e-6)


def _push_states(n=5):
    """Standing robot, box at its keyframe, a target 0.4243 m away on the table."""
    s = _blank(HH.H1HandPush, n)
    s[:, 76:79] = [0.7, 0.0, 1.0]
    s[:, 164:167] = [1.0, 0.3, 1.0]
    return s


def test_push_pays_nothing_for_never_touching_the_box():
    assert _metric(HH.H1HandPush, _push_states()) == 0.0

    # and off the table is WORSE than untouched, not partial credit: the target sits at
    # table height, so the floor is further from it than the start was
    knocked = _push_states()
    knocked[2:, 76:79] = [0.9, 0.3, 0.10]
    assert _metric(HH.H1HandPush, knocked) == 0.0


def test_push_scores_one_exactly_where_the_env_would_have_terminated():
    """Inside the 0.05 m radius the env ends the episode as a success, so the metric
    saturates there rather than asking a terminated episode to keep improving."""
    states = _push_states()
    states[-1, 76:79] = [0.98, 0.29, 1.0]
    assert _metric(HH.H1HandPush, states) == 1.0


def test_push_credits_the_best_approach_not_the_final_state():
    """d0 = |(0.3, 0.3, 0)| and the midpoint halves it, so a push that reached halfway
    and was then knocked back reads exactly 0.5 -- the high-water argument, hurdle's."""
    states = _push_states(n=6)
    states[2, 76:79] = [0.85, 0.15, 1.0]
    states[3:, 76:79] = [0.7, 0.0, 1.0]
    assert _metric(HH.H1HandPush, states) == pytest.approx(0.5, abs=1e-9)


def test_push_born_solved_is_one_not_a_division_by_zero():
    """Upstream can draw the target on top of the box (~0.65% of episodes). The metric
    must read that as the env does -- an immediate success -- and never divide by the
    zero-length initial gap."""
    states = _push_states()
    states[:, 164:167] = states[:, 76:79]
    assert _metric(HH.H1HandPush, states) == 1.0


#: Where `Basketball.reset_model` puts the ball at theta = 0: 1.5 m out at roughly
#: hand height. z is approximate; the metric reads only positions, not the launch law.
_BB_LAUNCH = (1.5, 0.0, 1.68)


def _bb_states(n=6):
    s = _blank(HH.H1HandBasketball, n)
    s[:, 76:79] = _BB_LAUNCH
    return s


def test_basketball_pays_nothing_for_catch_and_hold():
    """The documented exploit: catch the incoming ball and stand still collecting the
    ball-to-hoop kernel forever. The ball is launched TOWARD the robot -- away from the
    hoop -- so a hold's best approach IS the launch distance and the metric reads 0.0."""
    states = _bb_states(n=40)
    states[2:, 76:79] = [0.3, 0.0, 1.3]        # caught, held at the chest ever after
    assert _metric(HH.H1HandBasketball, states) == 0.0


def test_basketball_scores_one_at_the_hoop():
    states = _bb_states()
    states[-1, 76:79] = HH._HOOP_POS + [0.0, 0.0, 0.03]
    assert _metric(HH.H1HandBasketball, states) == 1.0


def test_basketball_credits_a_throw_by_its_closest_approach():
    """A throw that dies halfway to the hoop reads exactly 0.5, whatever the ball does
    afterwards -- best approach over the episode, normalised by the launch gap."""
    states = _bb_states()
    launch = np.array(_BB_LAUNCH)
    d0 = np.linalg.norm(HH._HOOP_POS - launch)
    states[3, 76:79] = HH._HOOP_POS + (launch - HH._HOOP_POS) / d0 * (d0 / 2.0)
    assert _metric(HH.H1HandBasketball, states) == pytest.approx(0.5, abs=1e-9)


def test_walk_pays_nothing_for_standing_and_full_marks_at_the_tasks_own_pace():
    """The spec's exploit: the reward's `(5*move+1)/6` floor pays a motionless stand
    ~0.16/step for the whole episode. The metric reads mean forward speed and must
    score that exactly 0.0 -- and 1.0 at the task's own 1 m/s bound."""
    still = _blank(HH.H1HandWalk, n=50)
    assert _metric(HH.H1HandWalk, still) == 0.0

    pace = _blank(HH.H1HandWalk, n=51)
    pace[:, 0] = np.arange(51) * HH._LOCO_V_REF * HH._CONTROL_DT
    assert _metric(HH.H1HandWalk, pace) == pytest.approx(1.0)

    backwards = _blank(HH.H1HandWalk, n=50)
    backwards[:, 0] = -np.arange(50) * 0.1
    assert _metric(HH.H1HandWalk, backwards) == 0.0, "reversing must clip to 0, not go negative"


def test_crawl_metric_is_speed_times_the_corridor_gate():
    """Mean speed over v_ref times the in-corridor fraction. A pose-and-park
    (the spec's exploit, amplified by the absent terminal) reads 0.0; walking the same
    distance OUTSIDE the corridor earns the speed and loses the gate."""
    parked = _blank(HH.H1HandCrawl, n=50)
    parked[:, 0], parked[:, 2] = 5.0, 0.6      # posed mid-tunnel, never moving
    assert _metric(HH.H1HandCrawl, parked) == 0.0

    crawl = _blank(HH.H1HandCrawl, n=51)
    crawl[:, 0] = np.arange(51) * HH._LOCO_V_REF * HH._CONTROL_DT
    assert _metric(HH.H1HandCrawl, crawl) == pytest.approx(1.0)

    outside = crawl.copy()
    outside[:, 1] = 2.0                        # same travel, beside the tunnel
    assert _metric(HH.H1HandCrawl, outside) == 0.0

    half_out = crawl.copy()
    half_out[1::2, 1] = 2.0                    # weaving in and out halves the gate
    assert _metric(HH.H1HandCrawl, half_out) == pytest.approx(
        np.mean(np.abs(half_out[1:, 1]) <= HH._CRAWL_CORRIDOR_HALF_WIDTH))


def test_sit_scores_the_seat_and_nothing_that_merely_resembles_it():
    """Geometry, not the reward's kernels: seated over the footprint counts, and the
    three near-misses -- floor, beside the chair, standing ON the chair -- all read 0.

    The denominator is the horizon (`tests/test_h1hand_sit_metric.py` holds the
    terminated-episode half), so full marks need a full-horizon trajectory."""
    seated = _blank(HH.H1HandSitSimple, n=HH._SIT_HORIZON + 1)
    seated[:, 0:3] = [-0.25, 0.0, 0.70]
    assert _metric(HH.H1HandSitSimple, seated) == pytest.approx(1.0)

    standing_beside = _blank(HH.H1HandSitSimple, n=10)      # spawn pose: x = 0
    assert _metric(HH.H1HandSitSimple, standing_beside) == 0.0

    floor = _blank(HH.H1HandSitSimple, n=10)
    floor[:, 0:3] = [-0.25, 0.0, 0.15]
    assert _metric(HH.H1HandSitSimple, floor) == 0.0

    on_top = _blank(HH.H1HandSitSimple, n=10)
    on_top[:, 0:3] = [-0.25, 0.0, 1.45]
    assert _metric(HH.H1HandSitSimple, on_top) == 0.0

    half = _blank(HH.H1HandSitSimple, n=11)
    half[6:, 0:3] = [-0.25, 0.0, 0.70]         # sits down from step 6 of 10 arrivals
    assert _metric(HH.H1HandSitSimple, half) == pytest.approx(5 / HH._SIT_HORIZON), (
        "five seated arrivals over the 1000-step horizon -- not 0.5 of the ten rows "
        "visited, which is what a visited-states mean would read for a terminated topple")


@pytest.mark.parametrize("env_id,cls", ADAPTERS, ids=IDS)
def test_every_metric_stays_inside_the_unit_interval(env_id, cls):
    """`EnvAdapter` requires [0, 1] and `success()` is a threshold on it, so a metric that
    ran over would make the boolean and the continuous ground truth disagree."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        states = rng.uniform(-4.0, 4.0, size=(8, cls.obs_dim))
        assert 0.0 <= _metric(cls, states) <= 1.0


def test_the_hurdle_walls_are_the_ten_this_repo_thinks_they_are():
    """Seven-metre spacing, ten of them. A constant nothing else pins, and the metric is
    a pure function of it."""
    assert HH._HURDLE_WALLS_X == tuple(float(7 * k) for k in range(1, 11))


# --------------------------------------------------------------------------
# the simulator half: deselected in every CI configuration
# --------------------------------------------------------------------------


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", IDS)
def test_a_reset_carries_nothing_of_the_previous_episode(env_id, sim_envs):
    """`reset()` from any history must leave `MjData` exactly as a reset from a fresh
    adapter would. The observation always did (qpos ++ qvel are what `set_state` writes);
    what leaked was everything the constraint solve derives from the leftover `ctrl` and
    `qacc_warmstart` -- `efc_force`, `cfrc_ext`, `actuator_force`, `qacc` -- which a
    controller reading contact forces at t=0 sees. Measured on h1hand_cube without the
    reset: 31 MjData arrays differ between "seed 13 alone" and "seed 13 after seed 12",
    the first action differs by 0.17, and eval_policy's per-seed rows depend on the seeds
    before them. Every registry record implicitly claims this equality, so it is pinned here."""
    env = sim_envs[env_id]
    sim = env._env
    fields = ("ctrl", "qacc_warmstart", "qacc", "actuator_force", "efc_force", "cfrc_ext", "qfrc_constraint")

    def snapshot():
        return {f: np.array(getattr(sim.data, f), dtype=float, copy=True) for f in fields}

    env.reset(np.random.default_rng(13))
    fresh = snapshot()
    # A history: a different seed, then a handful of large actions, then the same reset.
    s = env.reset(np.random.default_rng(12))
    rng = np.random.default_rng(0)
    for _ in range(5):
        s, done, _ = env.step(s, rng.uniform(-1.0, 1.0, size=env.action_dim))
        if done:
            break
    env.reset(np.random.default_rng(13))
    after = snapshot()
    for f in fields:
        assert after[f].shape == fresh[f].shape, f
        assert np.array_equal(after[f], fresh[f]), (
            f"{env_id}: MjData.{f} after reset depends on the previous episode "
            f"(max |diff| {np.max(np.abs(after[f] - fresh[f])):.3g}); reset must clear ctrl "
            f"and the warm-start before forwarding")


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", IDS)
def test_the_observation_is_the_whole_replayable_state(env_id, sim_envs):
    """The claim the whole adapter rests on: the observation is qpos ++ qvel ++ extra,
    so restoring it (set_state plus `_restore_extra`) makes a stateless
    `step(state, action)` satisfiable.

    Asserted rather than assumed because `humanoid_bench.wrappers.ObservationWrapper`
    returns `qpos[7:] + qvel[6:]` -- dropping the free root joint -- and is selected by a
    STRING compare against "true".
    """
    env = sim_envs[env_id]
    sim = env._env
    s = env.reset(0)
    nqnv = int(sim.model.nq) + int(sim.model.nv)
    assert s.shape == (env.obs_dim,) == (nqnv + type(env).extra_dims,)
    assert np.allclose(s[:nqnv], np.concatenate([sim.data.qpos, sim.data.qvel]))
    assert not getattr(sim, "obs_wrapper", False)


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", IDS)
def test_the_flat_fields_are_the_models_own_joint_table(env_id, sim_envs):
    """The spec's 151-to-229-row observation table, rebuilt from the live model.

    The tables were GENERATED from these models rather than transcribed, which removes
    the transcription error and leaves the drift: an upstream asset that gains a joint,
    reorders one, or changes a range makes every index in the spec wrong while every
    other test in this repo stays green. This is the check that sees it.

    Names, not descriptions: the prose is authored and the names are derived, so only the
    names can be compared against the model without asserting a writing style.
    """
    import mujoco

    env = sim_envs[env_id]
    m = env._env.model
    spec = tasks.load(env_id)
    got = [f["name"] for f in sorted(spec.state_surface["flat_fields"],
                                     key=lambda f: f["index"])]

    # THE ROBOT'S HINGES, not the model's. `h1hand_door` and `h1hand_cabinet` put
    # HINGED SCENERY in the same model -- a door and its hatch, two cabinet doors and
    # a pull-up drawer -- and counting every hinge would fail this assertion on those
    # two tasks with the message "the robot gained or lost a hinge joint", which is
    # wrong about its own subject: the robot is unchanged at 69 in all six, as
    # `h1hand_basketball` (69 hinges, no scenery) shows directly.
    #
    # Filtered by KINEMATIC ROOT rather than by name or by position. A joint belongs
    # to the robot iff walking `body_parentid` from its body reaches `pelvis`; the
    # scenery roots at `door`, `lateral_cabinet` or `pullup_drawer`. Measured:
    # door 69 + 2, cabinet 69 + 3, and the robot's 69 are identical across
    # all six tasks. Taking `hinges[:69]` would also pass today -- the robot is
    # declared first in every one of these XMLs -- and would silently stop meaning
    # "the robot" the day an asset reorders, which is the exact drift the docstring
    # above says this test exists to catch.
    def _roots_at_pelvis(joint_id: int) -> bool:
        body = int(m.jnt_bodyid[joint_id])
        while body > 0:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)
            parent = int(m.body_parentid[body])
            if parent == 0:
                return name == "pelvis"
            body = parent
        return False

    all_hinges = [j for j in range(int(m.njnt))
                  if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)]
    hinges = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
              for j in all_hinges if _roots_at_pelvis(j)]
    scenery = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
               for j in all_hinges if not _roots_at_pelvis(j)]
    assert len(hinges) == 69, (
        f"the robot gained or lost a hinge joint: {len(hinges)} rooted at pelvis "
        f"(task scenery, excluded: {scenery or 'none'})")
    assert got[7:76] == hinges, "the spec's joint-angle block is not the model's order"
    assert got[:7] == ["pelvis_x", "pelvis_y", "pelvis_z", "pelvis_quat_w",
                       "pelvis_quat_x", "pelvis_quat_y", "pelvis_quat_z"]
    assert got[env.n_q + 6: env.n_q + 75] == [h + "_vel" for h in hinges]

    for f in spec.state_surface["flat_fields"]:
        assert f["description"].strip(), f"{env_id}: s[{f['index']}] has no description"


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", IDS)
def test_the_action_fields_are_the_models_own_actuators(env_id, sim_envs):
    """61 commands, not 69. The eight-joint gap is tendon coupling -- the middle and
    fingertip joints of four fingers per hand share one `A_<finger>J0` actuator -- and a
    spec that listed 69 actions would hand the model eight commands that do not exist."""
    import mujoco

    env = sim_envs[env_id]
    m = env._env.model
    want = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(int(m.nu))]
    got = [f["name"] for f in tasks.load(env_id).env["spaces"]["action_fields"]]
    assert got == want
    assert len(got) == 61
    assert sum(n.endswith("J0") for n in got) == 8


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", IDS)
def test_step_is_a_function_of_its_arguments_and_the_warmstart_is_why(env_id, sim_envs):
    """Shuffled replay, both with and WITHOUT the `qacc_warmstart` zeroing.

    THE SECOND HALF IS THE POINT. "0 mismatches" alone would pass just as happily on an
    env whose solver never runs, so its passing would say nothing about whether the line
    under test does anything. Asserting that removing the line BREAKS purity is what
    makes the first assertion mean something. `bird/envs/humanoid.py` measured 115/300
    mismatches at 4.8e-1 on `h1-crawl-v0`; these four are at least as contact-rich.

    SHUFFLED, never round-trips: a round-trip re-steps the state the simulator is already
    in, so the warmstart is already the right one and the defect is invisible.
    """
    env = sim_envs[env_id]
    rng = np.random.default_rng(0)

    s = env.reset(0)
    pairs = []
    for _ in range(30):
        a = rng.uniform(-1, 1, size=env.action_dim)
        s2, done, _ = env.step(s, a)
        pairs.append((s.copy(), a.copy(), s2.copy()))
        s = env.reset(0) if done else s2

    order = rng.permutation(len(pairs))
    for i in order:
        s0, a, want = pairs[i]
        got, _, _ = env.step(s0, a)
        assert np.array_equal(got, want), f"{env_id}: step is not a function of its args"

    # and now the control: the same replay with the warmstart left alone
    orig = type(env)._step

    def _impure(self, s, a):
        u = np.clip(np.asarray(a, dtype=float).ravel(), -1.0, 1.0)
        self._restore_extra(s[self.n_q + self.n_v:])
        self._env.set_state(s[:self.n_q], s[self.n_q:self.n_q + self.n_v])
        self._env.do_simulation(self._env.task.unnormalize_action(u), self.frame_skip)
        self._after_step()
        return self._observe(), False, {}

    mismatches = 0
    try:
        type(env)._step = _impure
        for i in order:
            s0, a, want = pairs[i]
            got, _, _ = env.step(s0, a)
            mismatches += int(not np.array_equal(got, want))
    finally:
        type(env)._step = orig
    assert mismatches > 0, (
        f"{env_id}: dropping the qacc_warmstart zeroing changed nothing, so this test "
        "proves nothing about the line it is here to protect. Either the solver has "
        "stopped running on this env or the control is no longer exercising it.")


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", IDS)
def test_the_terminal_is_upstreams_own_and_has_not_been_removed(env_id, sim_envs):
    """Each task keeps exactly upstream's terminal -- including the two that DON'T have
    a failure terminal.

    `bird/envs/mujoco_control.py` records why a terminal is otherwise removed:
    terminating on failure lets SAC's entropy bonus supply a survival reward. It
    is kept here because on `highbar` that incentive IS the documented exploit -- the
    object of study -- and removing it would make these tasks a different benchmark from
    the one whose defects are cited. The other direction is pinned too: `push` ends only
    on success and `crawl` never ends, and a well-meaning collapse terminal ADDED to
    either would be just as silent a change of benchmark.
    """
    env = sim_envs[env_id]
    spec = tasks.load(env_id)
    assert spec.env["termination"]["early_terminate"] is (env_id not in NEVER_TERMINATES)

    s = env.reset(0)
    fallen = s.copy()
    fallen[2] = 0.0                            # pelvis on the floor
    if env_id == "h1hand_push":
        # keep the success terminal out of the reading: park the goal away from the box
        fallen[164:167] = [1.0, 0.5, 1.0]
    _, done, _ = env.step(fallen, np.zeros(env.action_dim))
    if env_id in FALLS_TERMINATE:
        assert done, f"{env_id}: a collapsed robot did not terminate"
    else:
        assert not done, (
            f"{env_id}: a collapsed robot terminated, but upstream has no failure "
            "terminal here (push ends only on success; crawl never ends). A terminal "
            "has been added, which changes the benchmark.")


@pytest.mark.humanoid
def test_the_room_reset_reproduces_upstreams_off_by_one(sim_envs):
    """`Room.reset_model` loops `range(-7, 0)` over SIX objects, so its first iteration
    writes qpos 69 and 70 -- the right hand's `rh_LFJ2` and `rh_LFJ1`, both limited to
    [0, 1.571] rad.

    REPRODUCED, NOT CORRECTED: a different initial distribution is a different
    environment, and a quiet fix would make BIRD's numbers incomparable with anyone
    else's on `h1hand-room-v0`. Pinned so that an upstream fix arrives as a failing test
    rather than as a silent change of environment.
    """
    env = sim_envs["h1hand_room"]
    s = env.reset(0)
    assert not (0.0 <= s[69] <= 1.5708), (
        "rh_LFJ2 is inside its own range at reset, so upstream's off-by-one is gone. "
        "Confirm against humanoid_bench/envs/room.py before relaxing this: the adapter "
        "reproduces the loop deliberately.")
    assert env.reset(0).tolist() == s.tolist(), "reset is not a function of the seed"
    assert env.reset(1).tolist() != s.tolist(), "the object scatter ignores the seed"


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", IDS)
def test_reference_reward_is_the_shipped_reward_and_moves(env_id, sim_envs):
    """It is HumanoidBench's own `get_reward`, called rather than ported. What is
    asserted is that it is real: finite, inside its own codomain (the tolerance-kernel
    rewards are bounded to [0, 1]; push is a dense negative shaping and basketball
    carries a +1000 bonus, so REFERENCE_REWARD_RANGE widens for those two), and
    different at two different states. A constant would make every EPIC/STARC number on
    the tier meaningless while every test stayed green.

    THE PERTURBATION IS AN ORIENTATION, not a pelvis HEIGHT, which is a genuinely
    different question: lowering the pelvis moves three of the four rewards and leaves
    `h1strong_highbar_hard`'s at 0.0000 both times, because that reward is
    `upright_reward * feet * small_control` and `upright_reward` is a `tolerance` on
    NEGATIVE torso uprightness -- it is exactly zero for an upright robot and stays zero
    however far it is lowered. Every one of the four reads `torso_upright`, so inverting
    the pelvis is the perturbation that is guaranteed to reach all of them.
    """
    env = sim_envs[env_id]
    lo, hi = REFERENCE_REWARD_RANGE.get(env_id, _DEFAULT_RANGE)
    s = env.reset(0)
    r0 = env.reference_reward(s)
    assert np.isfinite(r0) and lo <= r0 <= hi

    inverted = s.copy()
    inverted[3:7] = [0.0, 0.0, 1.0, 0.0]        # pitched 180 deg: feet up
    r1 = env.reference_reward(inverted)
    assert np.isfinite(r1) and lo <= r1 <= hi
    assert r1 != r0, (
        f"{env_id}: the shipped reward reads the same ({r0}) upright and inverted, so it "
        "is constant over the one axis every one of these rewards is built on")


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", IDS)
def test_render_is_a_pure_function_of_the_state(env_id, sim_envs):
    """Frames are graded by a VLM on this tier, so a render that depended on step history
    would make a caption a statement about the rollout order."""
    env = sim_envs[env_id]
    a = env.reset(0)
    # NOT `reset(1)`. `h1strong_highbar_hard` sets `_randomness = 0` -- upstream's
    # `HighBarBase.reset_model` does, and the adapter reproduces the steady state -- so
    # two seeds give the BIT-IDENTICAL state there and "two different states rendered
    # identically" would be a true statement about one state. The zero randomness is
    # asserted in its own test below.
    b = a.copy()
    b[2] += 0.6                                  # a pose, not a seed
    fa, fb = env.render(a), env.render(b)
    assert fa.shape == (env.render_height, env.render_width, 3)
    assert np.array_equal(env.render(a), fa), "render is not pure"
    assert not np.array_equal(fa, fb), "two different states rendered identically"


@pytest.mark.humanoid
def test_the_highbar_reset_really_is_deterministic(sim_envs):
    """`_randomness = 0` on that adapter, and the other three keep upstream's 0.01.

    Asserted because it is load-bearing in both directions: it is a deliberate divergence
    from upstream (whose first episode alone is perturbed -- see the attribute), and it is
    what makes two seeds give one state, which is why the render test above uses a pose,
    not a seed, to get a different state.
    """
    hb = sim_envs["h1strong_highbar_hard"]
    assert np.array_equal(hb.reset(0), hb.reset(7)), "highbar reset is not seed-independent"
    other = sim_envs["h1hand_powerlift"]
    assert not np.array_equal(other.reset(0), other.reset(7)), (
        "powerlift reset ignores its seed, so upstream's 0.01 randomness is not reaching it")


# --------------------------------------------------------------------------
# the extra-state mechanics: push's goal and basketball's stage latch
# --------------------------------------------------------------------------


@pytest.mark.humanoid
def test_push_carries_its_goal_in_the_state_and_step_honours_it(sim_envs):
    """The whole extra-dims contract on one task: the goal rides in s[164:167], is drawn
    from ctx.rng inside its sample box, stays fixed within an episode, and a state
    carrying a DIFFERENT goal is a different task instance -- `_restore_extra` must make
    the termination read the carried goal, not whatever the task object last held."""
    env = sim_envs["h1hand_push"]
    s = env.reset(0)
    goal = s[164:167].copy()
    assert 0.7 <= goal[0] <= 1.0 and -0.5 <= goal[1] <= 0.5
    assert goal[2] == pytest.approx(1.0)

    s2, _, _ = env.step(s, np.zeros(env.action_dim))
    assert np.array_equal(s2[164:167], goal), "the goal moved inside an episode"

    solved = s.copy()
    solved[164:167] = solved[76:79]            # the goal ON the box: success by definition
    _, done, info = env.step(solved, np.zeros(env.action_dim))
    assert info["success"] and done, (
        "a state carrying goal == box did not terminate as a success, so step() is not "
        "reading the goal off the state it was handed")

    assert env.reset(0)[164:167].tolist() == goal.tolist(), "goal is not a function of the seed"
    assert env.reset(1)[164:167].tolist() != goal.tolist(), "the goal draw ignores the seed"


@pytest.mark.humanoid
def test_push_reference_reward_is_negative_until_the_bonus(sim_envs):
    """`-0.1 * hand_dist - goal_dist` is strictly negative wherever the +1000 has not
    fired -- the property the spec's notes lean on. And with the box AT the goal the
    bonus dominates: the two regimes must both be reachable through reference_reward."""
    env = sim_envs["h1hand_push"]
    s = env.reset(0)
    assert env.reference_reward(s) < 0.0

    solved = s.copy()
    solved[164:167] = solved[76:79]
    assert env.reference_reward(solved) > 900.0


@pytest.mark.humanoid
def test_basketball_reset_rethrows_the_ball_from_the_seeded_rng(sim_envs):
    """`Basketball.reset_model`, reproduced on ctx.rng: ball on the 1.5 m ring at
    hand-relative height, velocity -7.5 * (cos, sin, 0) -- i.e. exactly -5x its own
    planar position -- and the whole draw a function of the seed."""
    env = sim_envs["h1hand_basketball"]
    s = env.reset(0)
    assert s[164] == 0.0, "an episode must start in the catch stage"
    assert float(np.hypot(s[76], s[77])) == pytest.approx(1.5, abs=1e-9)
    assert float(np.hypot(s[158], s[159])) == pytest.approx(7.5, abs=1e-9)
    assert np.allclose(s[158:160], -5.0 * s[76:78], atol=1e-9)
    assert env.reset(0).tolist() == s.tolist(), "reset is not a function of the seed"
    assert env.reset(1).tolist() != s.tolist(), "the launch angle ignores the seed"


@pytest.mark.humanoid
def test_basketball_stage_latch_flips_on_contact_and_rides_in_the_state(sim_envs):
    """The latch is the reward's own state machine, so it must (a) flip when the ball
    touches something, (b) never flip back within an episode, and (c) be read off the
    STATE, not off whatever the task object remembers."""
    env = sim_envs["h1hand_basketball"]

    probe = env.reset(0)
    probe[76:79] = [0.0, 0.0, 1.1]             # ball overlapping the torso: contact
    probe[158:164] = 0.0
    s2, _, _ = env.step(probe, np.zeros(env.action_dim))
    assert s2[164] == 1.0, "a ball in contact with the torso did not flip the latch"

    thrown = env.reset(0)
    thrown[164] = 1.0                          # a throw-stage state, ball in free air
    s3, _, _ = env.step(thrown, np.zeros(env.action_dim))
    assert s3[164] == 1.0, "the latch reverted to catch, which upstream's never does"

    fresh = env.reset(0)                       # ball in free air, stage from the state
    s4, done, _ = env.step(fresh, np.zeros(env.action_dim))
    if not done:
        assert s4[164] in (0.0, 1.0)

    # and the stage changes the reward the way upstream's weights say it should: with
    # the ball held at the chest, catch pays ~0.5 * stand + 0.5 * proximity while throw
    # pays the hoop kernel instead -- the two must simply DIFFER at one state
    held = env.reset(0)
    held[76:79] = [0.5, 0.0, 1.3]              # near the robot but clear of its geoms:
    held[158:164] = 0.0                        # get_reward's own contact scan must not
    held[164] = 0.0                            # flip the stage we are testing
    r_catch = env.reference_reward(held)
    held[164] = 1.0
    r_throw = env.reference_reward(held)
    assert r_catch != r_throw, (
        "the stage flag changed nothing in the reward, so _restore_extra is not "
        "reaching Basketball.get_reward")


# --------------------------------------------------------------------------
# package and door: the metric against the documented exploit
# --------------------------------------------------------------------------


def _traj(states):
    """A trajectory object shaped the way `_states_of` reads one."""
    return SimpleNamespace(states=np.asarray(states, dtype=float))


def _metric_only(cls):
    """An adapter with its `__init__` skipped, so `task_metric` can be exercised without
    the simulator.

    `_H1HandBase.__init__` imports `humanoid_bench` and builds a MuJoCo model, which is
    installed in no interpreter the default suite uses -- but `task_metric` reads nothing but the
    state array and class attributes. `object.__new__` is this repo's existing idiom for
    exactly that (`RunDir` in the artifact tests), and it is what keeps these checks
    running at all in an interpreter that cannot build the model, rather than behind an
    `importorskip` that would make them vanish.

    WHAT IT DOES NOT BUY. These checks are NOT in the default CI selection: this module
    is `pytestmark = [pytest.mark.slow]` and `ci.yml` runs `-m "not slow"`, so the whole
    file is deselected there. Being sim-free makes them runnable in any interpreter that
    selects `slow` -- which is real, and is why they are written this way -- but it does
    not put them in the default run.

    That deselection is deliberate and is not a bug to route around: `pyproject.toml`
    assigns `slow` BY FILE AND BY KIND, and this file's kind is dependency-gated. Roughly
    69 sim-free cases in this file sit behind that choice, not the five below.
    """
    return object.__new__(cls)


def _standing(n, *, pelvis_z=0.98, vx=0.0, vy=0.0, qx=0.0, qy=0.0):
    """`n` states of a robot at a given height, tilt and horizontal velocity.

    Row 0 of every trajectory the loop builds is the RESET state (`training._rollout`,
    `policy_api.run_episode`: `states = [env.reset(rng)]`, then one row per step) and the
    metrics score rows 1..n, so a caller that wants T arriving states asks for T + 1 rows.

    The quaternion is kept UNIT-LENGTH (qw absorbs whatever qx/qy leave over), so
    `qx = sin(45 deg)` really is a 90-degree tilt and the rows stay honest states if a
    metric ever starts reading the full quaternion.
    """
    st = np.zeros((n, 151))
    st[:, 2] = pelvis_z
    st[:, 3] = np.sqrt(max(1.0 - qx * qx - qy * qy, 0.0))
    st[:, 4] = qx
    st[:, 5] = qy
    st[:, 76] = vx
    st[:, 77] = vy
    return st


def test_stand_scores_the_hop_the_reward_cannot_see():
    """The exploit this task's spec records, falsified by construction.

    `Stand` prices `dont_move = tolerance(center_of_mass_velocity()[[0, 1]]).mean()` --
    the HORIZONTAL components only -- so vertical motion is not in the reward at all. A
    policy that hops in place keeps horizontal velocity at zero and collects very nearly
    the full return while plainly not standing.

    The metric gates on pelvis HEIGHT as well, so the two behaviours are different numbers
    here and the same number to the reward. Pinned with two trajectories identical except
    in the one axis the reward ignores.
    """
    env = _metric_only(HH.H1HandStand)
    assert env.task_metric(_traj(_standing(20))) == 1.0

    # 21 rows: the reset state and 20 arriving states, so `1::2` crouches exactly half
    # of the ARRIVING rows (10 of 20) and the metric reads 0.5 with no rounding.
    hop = _standing(21)
    hop[1::2, 2] = 0.45                      # airborne/crouched on every other step
    assert env.task_metric(_traj(hop)) == 0.5, (
        "a hop that halves the time spent standing must halve the metric -- the reward "
        "cannot see this at all, which is the point")


def test_stand_pays_nothing_for_drifting_or_for_lying_down():
    """The two other ways to look like standing without standing.

    Drifting: upright and tall but translating, which `dont_move` DOES price -- included
    so the metric is not merely a height gate wearing a posture name.
    Lying down: a robot flat on its back has a pelvis above the 0.2 m fall termination, so
    upstream never ends the episode; only the uprightness gate separates it from standing.
    """
    env = _metric_only(HH.H1HandStand)
    assert env.task_metric(_traj(_standing(10, vx=1.5))) == 0.0
    # qx = sin(45 degrees) -> uprightness 1 - 2*0.5 = 0.0, flat on its side
    assert env.task_metric(_traj(_standing(10, qx=np.sqrt(0.5)))) == 0.0


def test_run_is_walks_metric_against_the_five_times_larger_bound():
    """`Run` is `Walk` with one attribute changed, so the metric changes one constant.

    The pairing is the assertion: the SAME trajectory must read 1.0 on walk and 0.2 on
    run, because 1 m/s is walk's full pace and a fifth of run's. A metric that reused
    walk's normaliser would score both 1.0 and quietly claim a walk was a run.
    """
    st = np.zeros((51, 151))
    st[:, 0] = np.arange(51) * 1.0 * 0.02      # exactly 1 m/s for 50 steps
    assert _metric_only(HH.H1HandWalk).task_metric(_traj(st)) == pytest.approx(1.0)
    assert _metric_only(HH.H1HandRun).task_metric(_traj(st)) == pytest.approx(0.2)


def test_run_pays_nothing_for_the_move_floor_or_for_going_backwards():
    """`(5*move+1)/6` pays a standing robot 1/6 of the maximum for all 1000 steps. The
    metric reads 0.0, and backwards travel clips rather than going negative -- otherwise a
    retreat would be worth less than a fall, which is not a claim this metric makes.
    """
    env = _metric_only(HH.H1HandRun)
    assert env.task_metric(_traj(np.zeros((30, 151)))) == 0.0
    back = np.zeros((30, 151))
    back[:, 0] = -np.arange(30) * 2.0 * 0.02
    assert env.task_metric(_traj(back)) == 0.0


@pytest.mark.parametrize("cls", [HH.H1HandStair, HH.H1HandSlide],
                         ids=lambda c: c.__name__)
def test_climbing_pays_for_pace_only_while_upright(cls):
    """The exploit both climbing specs record, falsified by construction.

    Their terrain UNDULATES -- ray-cast, stair peaks at 0.9 m every 6 m and slide at
    1.645 m every 10 m -- so the far side of every rise is a descent, and
    `ClimbingUpwards`'s `move` term pays for forward velocity there exactly as on the
    climb, with gravity supplying it. Two trajectories with IDENTICAL forward progress,
    differing only in uprightness, must therefore be different numbers here and are the
    same number to the reward.
    """
    env = _metric_only(cls)
    # 51 rows: the reset state and 50 arriving states, so `1::2` below tilts exactly half
    # of the ARRIVING rows (25 of 50) -- 50 rows would be 25/49 and the assertion would be
    # testing this file's arithmetic rather than the metric.
    walking = np.zeros((51, 151))
    walking[:, 0] = np.arange(51) * 1.0 * 0.02      # 1 m/s, upright (qx = qy = 0)
    assert env.task_metric(_traj(walking)) == pytest.approx(1.0)

    tumbling = walking.copy()
    tumbling[:, 4] = np.sqrt(0.5)                   # 90 degrees over: uprightness 0.0
    assert env.task_metric(_traj(tumbling)) == 0.0, (
        "sliding down the far side at the same speed must not score as traversal")

    half = walking.copy()
    half[1::2, 4] = np.sqrt(0.5)
    assert env.task_metric(_traj(half)) == pytest.approx(0.5)


@pytest.mark.parametrize("cls", [HH.H1HandStair, HH.H1HandSlide],
                         ids=lambda c: c.__name__)
def test_climbing_pays_nothing_for_the_move_floor(cls):
    """`(5*move+1)/6` pays a standing robot 1/6 of the maximum for 1000 steps. Upright and
    motionless is the best case for that exploit -- full uprightness, zero pace -- and the
    metric still reads 0.0, because the product needs both.
    """
    assert _metric_only(cls).task_metric(_traj(np.zeros((30, 151)))) == 0.0


def test_stand_and_the_climbs_score_arriving_states_and_not_the_reset_row():
    """Row 0 is the reset state on every path that builds a trajectory
    (`training._rollout`, `policy_api.run_episode`: `states = [env.reset(rng)]`, then one
    row per step), and no action produced it. The three specs say so -- h1hand_stand
    "mean over arriving states", h1hand_stair and h1hand_slide "the fraction of arriving
    states whose uprightness ..." -- as does every other fraction-of-states metric in the
    file (`arrived = states[1:]`). Averaging Stand, or the climbs' uprightness factor,
    over ALL rows would make a keyframe reset that passes every gate (pelvis 0.98 m,
    identity quaternion, at rest) worth (k+1)/(T+1) where the spec says k/T: a policy
    that collapsed in T = 10 steps would read 0.091 for standing on none of them, and a
    reset row with no step after it would read 1.0.
    """
    stand = _metric_only(HH.H1HandStand)
    T = 10
    reset_row = _standing(1)
    # the reset state passes every gate; every ARRIVING state is on its side
    fallen = np.vstack([reset_row, _standing(T, qx=np.sqrt(0.5))])
    assert stand.task_metric(_traj(fallen)) == 0.0, "the spec's formula reads k/T = 0/10"
    # k of T arriving states standing reads exactly k/T, with no reset-row term
    mixed = np.vstack([reset_row, _standing(T)])
    mixed[1:4, 2] = 0.45                          # arriving rows 1-3 crouched: 7 of 10 stand
    assert stand.task_metric(_traj(mixed)) == pytest.approx(0.7)
    # a reset row with no step after it is not an episode, on either reduction
    assert stand.task_metric(_traj(reset_row)) == 0.0
    assert stand.success(_traj(reset_row)) is False

    for cls in (HH.H1HandStair, HH.H1HandSlide):
        climb = _metric_only(cls)
        st = np.zeros((T + 1, 151))
        st[:, 0] = np.arange(T + 1) * 1.0 * 0.02   # 1 m/s: the pace factor is exactly 1.0
        st[1:, 4] = np.sqrt(0.5)                   # every ARRIVING state 90 degrees over
        assert climb.task_metric(_traj(st)) == 0.0, (
            f"{cls.__name__}: the upright reset row must not buy 1/(T+1) of the pace")


# --------------------------------------------------------------------------
# cabinet and bookshelf: the ladder terminal fires when upstream's does
# --------------------------------------------------------------------------


class _ScriptedSim:
    """A fake `HumanoidEnv` for the two `_before_step` latch tasks. The 'physics' is a
    script -- `do_simulation` advances a state index kept in qpos[0] and `qpos_of` says
    what the scene looks like at each index -- and `task` is upstream's ladder / counter,
    completion test and terminal transcribed from HumanoidBench @cb11890 (`envs/cabinet.py`,
    `envs/bookshelf.py`; the stabilisation kernels, which need MuJoCo kinematics and touch
    neither the counter nor the terminal, are held at 1.0). The adapter's REAL `_step` and
    a literal transcription of upstream's `Task.step` are driven over the same script, so
    what is asserted is agreement, not a number."""

    frame_skip = 10

    def __init__(self, qpos_of, nv, task_cls):
        self._qpos_of = qpos_of
        self.data = SimpleNamespace(qpos=qpos_of(0), qvel=np.zeros(nv), qacc_warmstart=np.zeros(nv))
        self.model = SimpleNamespace(body_pos=np.zeros((40, 3)))
        self.named = SimpleNamespace(
            data=SimpleNamespace(xpos=_Xpos(self.data)),
            model=SimpleNamespace(geom_rgba={}))
        self.task = task_cls(self)

    def set_state(self, q, v):
        self.data.qpos, self.data.qvel = np.array(q, dtype=float), np.array(v, dtype=float)

    def do_simulation(self, ctrl, n):
        self.data.qpos = self._qpos_of(int(round(self.data.qpos[0])) + 1)


class _Xpos:
    """`named.data.xpos[obj]` and `[obj, "z"]` for a bookshelf object id, off the fake qpos."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        obj, axis = key if isinstance(key, tuple) else (key, None)
        pos = np.asarray(self._data.qpos[HH._bookshelf_object_qpos(int(obj))], dtype=float)
        return pos if axis is None else float(pos["xyz".index(axis)])


def _upstream_episode(sim, max_steps=12):
    """`humanoid_bench/tasks.py::Task.step`: do_simulation -> get_obs -> get_reward ->
    get_terminated, until terminated. Returns each step's reward and terminal."""
    log = []
    for _ in range(max_steps):
        sim.do_simulation(None, sim.frame_skip)
        r, _info = sim.task.get_reward()
        term, _tinfo = sim.task.get_terminated()
        log.append((float(r), bool(term)))
        if term:
            break
    return log


def _adapter_episode(cls, sim, tail0, max_steps=12):
    """The real `_step` from a reset-shaped state, until `done`; the trajectory, the
    per-step `info["success"]` flags, and the post-hoc reference terms exactly as
    `policy_api.run_episode` computes them (restore the tail, restore the state, ask
    `get_reward`)."""
    ad = object.__new__(cls)
    ad._env = sim
    nq, nv = cls.n_q, cls.n_v
    s = np.concatenate([sim.data.qpos, sim.data.qvel, np.asarray(tail0, dtype=float)])
    states, flags, done = [s.copy()], [], False
    for _ in range(max_steps):
        s, done, info = ad._step(s, np.zeros(cls.action_dim))
        states.append(s.copy())
        flags.append(bool(info["success"]))
        if done:
            break
    terms = []
    for t in states[1:]:
        ad._restore_extra(t[nq + nv:])
        sim.set_state(t[:nq], t[nq:nq + nv])
        terms.append(float(sim.task.get_reward()[0]))
    return ad, np.asarray(states), flags, done, terms


def _cabinet_qpos(i):
    q = np.zeros(HH.H1HandCabinet.n_q)
    q[0] = float(i)
    if i >= 1:
        q[79] = 0.4                     # rung 1: the pulling door, |0.4/0.4| > 0.95
    if i >= 2:
        q[76] = 0.45                    # rung 2: the drawer
    if i >= 3:
        q[81:84] = (0.9, 0.0, 0.9)      # rung 3: drawer cube in the middle box
    if i >= 4:
        q[88:91] = (0.9, 0.0, 1.5)      # rung 4: lateral cube in the top box
    return q


class _CabinetTask:
    """`Cabinet.get_reward` / `get_terminated` @cb11890, counter and terminal verbatim."""

    def __init__(self, env):
        self._env, self.current_subtask = env, 1

    def unnormalize_action(self, a):
        return a

    def get_reward(self):
        if self.current_subtask >= 5:                       # `lambda: (1000, {}, False)`
            return 1000.0, {"success": True}
        done = HH._cabinet_rung_done(self.current_subtask, self._env.data.qpos)
        reward = 0.2 * 1.0 + 0.8 * (1.0 if done else 0.5)
        if done:
            reward += 100 * self.current_subtask
            self.current_subtask += 1
        return reward, {"success": self.current_subtask == 5}

    def get_terminated(self):
        return self.current_subtask == 5, {}


_BS_IDS = np.array([-24, -23, -22, -21, -20])
#: goal 0 is a BOTTOM-shelf spot (z = 0.35 < the drop terminal's 0.5), deliberately first.
_BS_GOALS = np.array([(0.85, 0.05, 0.35), (0.75, -0.25, 1.55), (0.8, 0.05, 0.95),
                      (0.8, -0.25, 0.95), (0.85, -0.25, 0.35)])


def _bookshelf_qpos(i):
    q = np.zeros(HH.H1HandBookshelfHard.n_q)
    q[0], q[2] = float(i), 1.0                                   # pelvis above the fall terminal
    for k in range(5):
        q[HH._bookshelf_object_qpos(_BS_IDS[k])] = (0.3, 0.0, 1.0)   # on a shelf
        if i >= 3 + k:
            q[HH._bookshelf_object_qpos(_BS_IDS[k])] = _BS_GOALS[k]  # placed, in order
    return q


class _BookshelfTask:
    """`BookshelfBase.get_reward` / `get_terminated` @cb11890: the counter, the 0.15 m
    advance, the fall / success / drop terminals."""

    def __init__(self, env):
        self._env, self.task_index = env, 0
        self.bookshelf_objects = np.append(_BS_IDS, _BS_IDS[-1])
        self.placement_goals = np.vstack([_BS_GOALS, _BS_GOALS[-1]])
        self.object_names = {f"obj{k}": [f"g{k}_vision"] for k in range(12)}

    def unnormalize_action(self, a):
        return a

    def get_reward(self):
        obj = self._env.named.data.xpos[self.bookshelf_objects[self.task_index]]
        d = float(np.linalg.norm(obj - self.placement_goals[self.task_index]))
        reward = 0.2 * 1.0 + 0.4 * (1.0 if d < 0.15 else 0.0) + 0.4 * 0.5
        if d < 0.15:
            self.task_index += 1
            reward += 100 * self.task_index
        return reward, {"success": self.task_index == 5}

    def get_terminated(self):
        if self._env.data.qpos[2] < 0.58:
            return True, {"terminated_reason": 0}
        if self.task_index == 5:
            return True, {"terminated_reason": 1}
        if self._env.named.data.xpos[self.bookshelf_objects[self.task_index], "z"] < 0.5:
            return True, {"terminated_reason": 2}
        return False, {}


def test_cabinet_terminates_the_step_the_last_rung_completes_as_upstream_does():
    """Upstream's `Task.step` is do_simulation -> get_reward (advances the ladder on the
    ARRIVING state) -> get_terminated, so it ends the episode on the step whose arriving
    state completes rung four. The adapter advances the ladder in `_before_step` on the
    RESTORED state (so the observation tail stays pre-advance, which is what lets
    `reference_reward` reproduce each of upstream's advance bonuses); reading
    `get_terminated` with the counter one advance behind would run the episode one step
    longer than upstream's, with the extra state's tail at 5 and its `reference_reward`
    upstream's never-reached flat-1000 arm -- a completed cabinet's `reference_return`
    would read 2004 against upstream's 1004. So the tail stays pre-advance, and only the
    terminal reads the counter where upstream does.
    """
    nq, nv = HH.H1HandCabinet.n_q, HH.H1HandCabinet.n_v
    up = _upstream_episode(_ScriptedSim(_cabinet_qpos, nv, _CabinetTask))
    ad, states, flags, done, terms = _adapter_episode(
        HH.H1HandCabinet, _ScriptedSim(_cabinet_qpos, nv, _CabinetTask), [1.0])
    assert done and len(flags) == len(up) == 4, (
        f"upstream ends after {len(up)} steps, the adapter after {len(flags)}")
    assert flags == [False, False, False, True], "info['success'] is the event that ends the episode"
    assert [int(round(t)) for t in states[:, 213]] == [1, 1, 2, 3, 4], (
        "the carried ladder is PRE-advance on every emitted state (reference_reward's contract)")
    assert terms == pytest.approx([r for r, _ in up]), "post-hoc reference terms are upstream's rewards, step for step"
    assert sum(terms) == pytest.approx(1004.0), "no flat-1000 rung-five state is ever scored"
    assert ad.task_metric(states) == 1.0


def test_bookshelf_terminates_as_upstream_does_and_a_bottom_shelf_placement_is_not_a_drop():
    """Same lag as cabinet, and one more consequence: `get_terminated`'s drop clause tests
    `xpos[objects[task_index], z] < 0.5` on the CURRENT object, and with the counter one
    advance behind it would name the object just placed -- so placing an object at either
    bottom-shelf goal (z = 0.35) would end the adapter's episode as a drop where upstream
    advances and continues. Two of five goals are bottom-shelf and the order is a random
    permutation, so every adapter episode would end at or before the fourth placement and
    the success terminal would be unreachable. Here goal 0 is a bottom spot: upstream advances at
    that step and ends the episode on the fifth placement with success.
    """
    nq, nv = HH.H1HandBookshelfHard.n_q, HH.H1HandBookshelfHard.n_v
    up = _upstream_episode(_ScriptedSim(_bookshelf_qpos, nv, _BookshelfTask))
    tail0 = np.concatenate([[0.0], _BS_IDS.astype(float), _BS_GOALS.ravel()])
    ad, states, flags, done, terms = _adapter_episode(
        HH.H1HandBookshelfHard, _ScriptedSim(_bookshelf_qpos, nv, _BookshelfTask), tail0)
    assert done and len(flags) == len(up) == 7, (
        f"upstream ends after {len(up)} steps (the fifth placement), the adapter after "
        f"{len(flags)} -- a bottom-shelf placement must not read as a drop")
    assert ad._env.task.get_terminated()[1] == {"terminated_reason": 1}, "success, not a drop"
    assert flags == [False] * 6 + [True]
    assert [int(round(t)) for t in states[:, 307]] == [0, 0, 0, 0, 1, 2, 3, 4], "the counter stays PRE-advance"
    assert terms == pytest.approx([r for r, _ in up])
    assert ad.task_metric(states) == 1.0


def test_the_two_climbing_tasks_share_one_metric_because_upstream_shares_one_reward():
    """`class Stair(ClimbingUpwards): pass` and `class Slide(ClimbingUpwards): pass`.

    Upstream's two tasks are byte-identical code differing only in the asset, so a metric
    written twice would be two copies of one fact -- and the copies would drift, which is
    the failure keeping one copy exists to prevent. Pinned as identity of the
    function object, not as equal numbers on a sample: equal numbers would also hold for
    two implementations that happen to agree today.
    """
    assert HH.H1HandStair.task_metric is HH.H1HandSlide.task_metric
    assert HH.H1HandStair.task_metric is HH._H1HandClimb.task_metric
    # ...and the scene constants they do NOT share are the ones the ray-cast measured.
    assert HH.H1HandStair._first_crest_x != HH.H1HandSlide._first_crest_x


def test_pole_pays_for_pace_only_inside_the_forest():
    """The exploit the spec records, falsified by construction: the pole field spans
    |y| <= 2.3 and the ground beside it is open, so a robot that walks AROUND the field
    collects the full move term and never risks the contact discount. Two trajectories
    with identical forward progress, differing only in lane, must be different numbers
    here and are the same number to the reward."""
    env = _metric_only(HH.H1HandPole)
    weave = np.zeros((51, 151))
    weave[:, 0] = np.arange(51) * HH._POLE_V_REF * HH._CONTROL_DT
    assert env.task_metric(_traj(weave)) == pytest.approx(1.0)

    skirt = weave.copy()
    skirt[:, 1] = 3.0                          # same travel, beside the forest
    assert env.task_metric(_traj(skirt)) == 0.0

    assert env.task_metric(_traj(np.zeros((30, 151)))) == 0.0, "the move floor's optimum"
    back = np.zeros((30, 151))
    back[:, 0] = -np.arange(30) * 0.1
    assert env.task_metric(_traj(back)) == 0.0, "reversing clips to 0, not negative"


def test_pole_uses_its_own_half_speed_bound_not_walks():
    """pole.py sets `_WALK_SPEED = 0.5` -- half of walk's 1.0 -- so the same 0.5 m/s
    trajectory is full marks here and half marks on walk. A metric that borrowed walk's
    normaliser would quietly demand twice the task's published pace."""
    st = np.zeros((51, 151))
    st[:, 0] = np.arange(51) * 0.5 * HH._CONTROL_DT
    assert _metric_only(HH.H1HandPole).task_metric(_traj(st)) == pytest.approx(1.0)
    assert _metric_only(HH.H1HandWalk).task_metric(_traj(st)) == pytest.approx(0.5)


def _maze_track(*legs, n_per=10):
    """A pelvis track through the maze: straight-line segments between waypoints, all
    other observation dims zero."""
    pts = [np.array([0.0, 0.0])]
    for x, y in legs:
        a, b = pts[-1], np.array([x, y], dtype=float)
        pts += [a + (b - a) * t for t in np.linspace(0, 1, n_per)[1:]]
    xy = np.stack(pts)
    st = np.zeros((xy.shape[0], 152))
    st[:, 0:2] = xy
    return st


def test_maze_counts_checkpoints_in_order_and_only_in_order():
    """The metric mirrors the stage machine: (3,0), then (3,6), then (6,6). A robot that
    cuts across to a LATER checkpoint without passing the earlier one is still on leg
    one, exactly as upstream's counter would say -- crediting checkpoints in any order
    would score a wall-clipping shortcut upstream's own stage machine refuses."""
    env = _metric_only(HH.H1HandMaze)
    assert env.task_metric(_traj(_maze_track((3, 0), (3, 6), (6, 6)))) == 1.0

    # parking exactly on checkpoint one: leg two untouched, so exactly one third
    assert env.task_metric(_traj(_maze_track((3, 0)))) == pytest.approx(1 / 3)

    # straight to (3,6) without touching (3,0): still leg one; its best approach to
    # (3,0) happens mid-path. Nowhere near a third.
    skip = env.task_metric(_traj(_maze_track((3, 6))))
    assert skip < 1 / 3, "checkpoint two must not count while checkpoint one is unvisited"

    assert env.task_metric(_traj(np.zeros((20, 152)))) == 0.0, "standing at the spawn"


def test_maze_pays_the_open_leg_by_best_approach_not_by_lingering():
    """Half of leg one is one sixth of the maze -- and STAYING there does not grow it.
    The reward's proximity kernel pays per step for being near the current checkpoint
    (the spec's parking exploit); the metric is a high-water fraction, so ten more
    steps parked at the same spot move it not at all."""
    env = _metric_only(HH.H1HandMaze)
    half = _maze_track((1.5, 0.0))
    m = env.task_metric(_traj(half))
    assert m == pytest.approx((1.5 / 3.0) / 3.0)

    parked = np.vstack([half, np.repeat(half[-1:], 30, axis=0)])
    assert env.task_metric(_traj(parked)) == pytest.approx(m), "lingering must not pay"


def _sit_hard_states(n=10, chair=(-0.25, 0.0, 0.0), chair_quat=(1.0, 0.0, 0.0, 0.0),
                     pelvis=None):
    st = np.zeros((n, 164))
    st[:, 3] = 1.0
    st[:, 76:79] = chair
    st[:, 79:83] = chair_quat
    st[:, 0:3] = pelvis if pelvis is not None else (0.3, 0.0, 0.98)
    return st


def test_sit_hard_follows_the_chair_wherever_it_goes():
    """The free chair is the whole difference from sit_simple, so the metric's scoring
    region must MOVE with it: seated over the keyframe chair counts, seated over the
    spot the chair slid away from does not, and seated over the chair's new spot does.
    """
    env = _metric_only(HH.H1HandSitHard)
    seated = _sit_hard_states(pelvis=(-0.25, 0.0, 0.70))
    assert env.task_metric(_traj(seated)) == pytest.approx(1.0)

    ghost = _sit_hard_states(chair=(2.0, 1.0, 0.0), pelvis=(-0.25, 0.0, 0.70))
    assert env.task_metric(_traj(ghost)) == 0.0, (
        "squatting where the chair USED to be must not read as seated")

    followed = _sit_hard_states(chair=(2.0, 1.0, 0.0), pelvis=(2.0, 1.0, 0.70))
    assert env.task_metric(_traj(followed)) == pytest.approx(1.0)

    standing = _sit_hard_states()               # spawn pose, pelvis at 0.98
    assert env.task_metric(_traj(standing)) == 0.0


def test_sit_hard_scores_the_toppled_chair_squat_zero():
    """The spec's `sitting_band_is_absolute_while_the_chair_moves` exploit, falsified.

    Tip the chair 90 degrees onto its side and the reward's `sitting` kernel still pays
    its full height band at 0.68-0.72 m of ALTITUDE -- above a seat that is now
    vertical. The metric works in the chair's own frame, so the seat plane tips with the
    chair and the mid-air squat stops counting.
    """
    env = _metric_only(HH.H1HandSitHard)
    on_side = (np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0)      # 90 degrees about x
    squat = _sit_hard_states(chair=(1.0, 0.0, 0.2), chair_quat=on_side,
                             pelvis=(1.0, 0.0, 0.70))
    assert env.task_metric(_traj(squat)) == 0.0

    # and the control: the same relative offset, expressed in the tipped frame, still
    # counts -- the region moved rather than vanished. Local (0, 0, 0.70) in a frame
    # tipped 90 degrees about +x points along world -y.
    reseated = _sit_hard_states(chair=(1.0, 0.0, 0.2), chair_quat=on_side,
                                pelvis=(1.0, -0.70, 0.2))
    assert env.task_metric(_traj(reseated)) == pytest.approx(1.0)


def test_reach_pays_nothing_for_the_loiterer_or_the_statue():
    """The two documented exploits: `reward_close` pays +5/step a full METRE out, and
    `healthy_reward` pays up to +5/step for standing anywhere, forever, with no
    terminal. Both spend the episode outside the hand-sized radius and must read 0.0."""
    env = _metric_only(HH.H1HandReach)
    n = 157
    loiter = np.zeros((20, n))
    loiter[:, 151:154] = [0.5, 0.0, 1.2]       # hand held 0.5 m from the target
    loiter[:, 154:157] = [1.0, 0.0, 1.2]
    assert env.task_metric(_traj(loiter)) == 0.0

    statue = np.zeros((20, n))
    statue[:, 151:154] = [0.3, 0.3, 1.1]       # resting hand, target far away
    statue[:, 154:157] = [-1.5, 1.5, 0.3]
    assert env.task_metric(_traj(statue)) == 0.0

    held = loiter.copy()
    held[10:, 151:154] = held[10:, 154:157]    # reaches at step 10, holds
    assert env.task_metric(_traj(held)) == pytest.approx(10 / 19)


def test_reach_radius_is_the_hand_convention_not_the_rewards():
    """0.10 m is a stated hand-span convention; 0.05 is the reward's bonus radius. A
    hand at 0.07 m therefore counts here and earns no bonus there -- the gap is what
    keeps the metric independent of the shaping constant it must be able to falsify."""
    env = _metric_only(HH.H1HandReach)
    st = np.zeros((10, 157))
    st[:, 151:154] = [1.07, 0.0, 1.0]
    st[:, 154:157] = [1.0, 0.0, 1.0]
    assert env.task_metric(_traj(st)) == pytest.approx(1.0)
    st[:, 151] = 1.11                          # just outside the hand radius
    assert env.task_metric(_traj(st)) == 0.0


def test_balance_is_a_duration_because_leaving_the_posture_ends_the_episode():
    """The one metric in this file whose denominator is the HORIZON rather than the
    states visited, and the test is the reason: on balance, falling off is what ends
    the episode, so a visited-states mean scores a ten-step wobble-and-fall near 1.0.
    Duration is the task; a full episode reads 1.0, an early fall reads as the failure
    it is."""
    env = _metric_only(HH.H1HandBalanceSimple)

    full = np.zeros((1001, 164))
    full[:, 2] = 1.38                          # pelvis at the keyframe height
    full[:, 3] = 1.0
    full[:, 78] = 0.37                         # board at its rest height
    assert env.task_metric(_traj(full)) == pytest.approx(1.0)

    brief = full[:11].copy()                   # balanced ten steps, then the fall ended it
    assert env.task_metric(_traj(brief)) == pytest.approx(10 / 1000)

    hop = full.copy()
    hop[1::2, 2] = 0.9                         # dipping below board + 0.8 every other step
    assert env.task_metric(_traj(hop)) == pytest.approx(0.5)


def test_balance_measures_height_above_the_board_not_altitude():
    """The board is what the robot stands on and the board MOVES -- a pelvis at
    standing altitude above a board that has ridden up under it is a crouch, not a
    stand."""
    env = _metric_only(HH.H1HandBalanceSimple)
    crouch = np.zeros((101, 164))
    crouch[:, 2] = 1.1                         # would clear a world-fixed 0.8 gate
    crouch[:, 3] = 1.0
    crouch[:, 78] = 0.5                        # but the board is high: only 0.6 above it
    assert env.task_metric(_traj(crouch)) == 0.0


def test_the_two_balance_tasks_share_one_metric_because_upstream_shares_one_reward():
    """`BalanceSimple` and `BalanceHard` differ upstream by one `<freejoint/>` -- one
    reward, one terminal -- so one metric, pinned by function identity for the climbing
    pair's reason: equal numbers on a sample is what two copies do right before they
    drift."""
    assert HH.H1HandBalanceSimple.task_metric is HH.H1HandBalanceHard.task_metric
    assert HH.H1HandBalanceSimple.task_metric is HH._H1HandBalance.task_metric


def _window_states(n, tool=(0.93, -0.5, 1.2), quat=(1.0, 0.0, 0.0, 0.0), vz=0.5):
    """Tool held so its head (mounted at local (0, 0.5, 0.02)) sits against the glass,
    sweeping at `vz`."""
    st = np.zeros((n, 174))
    st[:, 2] = 0.98
    st[:, 3] = 1.0
    st[:, 76:79] = tool
    st[:, 79:83] = quat
    st[:, 164] = vz
    return st


def test_window_pays_only_for_sweeping_at_the_glass():
    """Each documented exploit dies on one conjunct: the air-wiper (vertical tool speed
    is paid anywhere) fails the position gate, the static press (the contact half pays
    for touching, motionless) fails the sweep gate, and only the two together count."""
    env = _metric_only(HH.H1HandWindow)
    wiping = _window_states(1001)
    assert env.task_metric(_traj(wiping)) == pytest.approx(1.0)

    air = _window_states(1001, tool=(0.2, 0.0, 1.2))     # shaking the tool by the robot
    assert env.task_metric(_traj(air)) == 0.0

    press = _window_states(1001, vz=0.0)                 # pressed to the glass, motionless
    assert env.task_metric(_traj(press)) == 0.0

    drop = _window_states(101)                           # wiped 100 steps, then dropped it
    assert env.task_metric(_traj(drop)) == pytest.approx(100 / 1000), (
        "the denominator is the horizon: dropping the tool ends the episode, and a "
        "visited-states mean would have scored this near 1.0")


def test_window_bounds_cover_the_ball_joint():
    """A `_bounds` that special-cased free joints only would send window's ball joint
    (the tier's first) into the scalar path and leave three of its four quaternion slots
    and two of its three angular rates at +/-inf -- a `random_state` drawing uniform
    over infinity, which raises. (SB3 itself trains bit-identically on an infinite Box,
    measured; the bounds have to be finite for the draw.) The fake model
    below is window's joint layout (free root, 69 hinges, free tool, ball wiper head);
    every bound must come back finite, with the ball's quaternion at exactly +/-1."""
    mujoco = pytest.importorskip("mujoco", reason="needs `uv sync --extra metaworld` (mujoco)")

    env = _metric_only(HH.H1HandWindow)
    free, ball, hinge = (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL),
                         int(mujoco.mjtJoint.mjJNT_HINGE))
    env._env = SimpleNamespace(model=SimpleNamespace(
        njnt=72,
        jnt_type=np.array([free] + [hinge] * 69 + [free, ball]),
        jnt_qposadr=np.array([0] + list(range(7, 76)) + [76, 83]),
        jnt_dofadr=np.array([0] + list(range(6, 75)) + [75, 81]),
        jnt_limited=np.array([0] + [1] * 69 + [0, 1]),
        jnt_range=np.array([[0.0, 0.0]] + [[-1.0, 1.0]] * 69 + [[0.0, 0.0], [0.0, 1.4]]),
    ))
    lo, hi = env._bounds()
    assert np.isfinite(lo).all() and np.isfinite(hi).all(), (
        "a +/-inf survived into the observation bounds -- the ball joint has fallen "
        "back into the scalar path")
    assert np.array_equal(lo[83:87], -np.ones(4)) and np.array_equal(hi[83:87], np.ones(4)), (
        "the ball's quaternion components must be bounded to exactly [-1, 1]; its "
        "jnt_range is a swing limit, not a per-component bound")
    assert np.array_equal(lo[168:171], np.full(3, -40.0))
    assert np.array_equal(hi[168:171], np.full(3, 40.0))


def test_window_reads_the_head_through_the_handles_orientation():
    """The head hangs half a metre along the handle's own y axis, so where the wipe is
    happening depends on the handle's ORIENTATION, not just its centre. The same centre
    rotated a quarter turn moves the head half a metre off the glass."""
    env = _metric_only(HH.H1HandWindow)
    yaw90 = (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
    turned = _window_states(101, quat=yaw90)
    assert env.task_metric(_traj(turned)) == 0.0


def _cube_states(n, left=None, right=None, target=(1.0, 0.0, 0.0, 0.0)):
    st = np.zeros((n, 181))
    st[:, 2] = 0.98
    st[:, 3] = 1.0
    st[:, 79:83] = left if left is not None else target
    st[:, 86:90] = right if right is not None else target
    st[:, 177:181] = target
    return st


def test_cube_needs_both_cubes_matched_and_ignores_the_quaternion_sign():
    """Two halves. BOTH: one matched cube is not the task, so it reads 0.0 while the
    reward's per-cube average pays half. SIGN: q and -q are the same rotation, and the
    metric must say so -- upstream's Euclidean ||q1 - q2|| reads the antipodal twin as
    maximally misaligned (distance 2.0), which is the recorded defect."""
    env = _metric_only(HH.H1HandCube)
    matched = _cube_states(501)
    assert env.task_metric(_traj(matched)) == pytest.approx(1.0)

    flipped = _cube_states(501, left=(-1.0, 0.0, 0.0, 0.0))
    assert env.task_metric(_traj(flipped)) == pytest.approx(1.0), (
        "-q IS q as a rotation; a metric that trips over the double cover would "
        "inherit exactly the defect the spec records against the reward")

    one_off = _cube_states(501, left=HH._euler_to_quat(np.array([1.2, 0.0, 0.0])))
    assert env.task_metric(_traj(one_off)) == 0.0, "one cube matched of two is not matched"


def test_cube_horizon_denominator_defeats_align_then_drop():
    """Dropping a cube TERMINATES, so a visited-states mean banks whatever fraction was
    aligned before the drop -- 25 aligned arrivals of a 500-step task must read 0.05,
    not ~1.0."""
    env = _metric_only(HH.H1HandCube)
    brief = _cube_states(26)
    assert env.task_metric(_traj(brief)) == pytest.approx(25 / 500)


def test_cube_tolerance_is_twenty_degrees_geodesic():
    """15 degrees off counts, 25 does not -- the boundary is the stated convention, not
    the reward's margin=0.3 on a quantity that is not even a rotation metric."""
    env = _metric_only(HH.H1HandCube)
    inside = _cube_states(11, left=HH._euler_to_quat(np.array([np.deg2rad(15), 0, 0])),
                          right=HH._euler_to_quat(np.array([0, np.deg2rad(15), 0])))
    assert env.task_metric(_traj(inside)) == pytest.approx(10 / 500)
    outside = _cube_states(11, left=HH._euler_to_quat(np.array([np.deg2rad(25), 0, 0])))
    assert env.task_metric(_traj(outside)) == 0.0


def test_cabinet_counts_completed_rungs_and_refuses_the_parked_door():
    """The task's central exploit, falsified by discreteness: the reward pays
    0.8 * openness per step for a door held at 94%, and completing the rung SWAPS that
    income for the next rung's near-zero shaping -- so parking below the threshold
    out-earns advancing. The metric pays only on completion, so the parked door and the
    untouched cabinet are the same 0.0."""
    env = _metric_only(HH.H1HandCabinet)
    n = 214

    still = np.zeros((30, n))
    assert env.task_metric(_traj(still)) == 0.0

    parked = np.zeros((30, n))
    parked[:, 79] = -0.4 * 0.94                # the pull door held just short
    assert env.task_metric(_traj(parked)) == 0.0

    opened = np.zeros((30, n))
    opened[:, 79] = -0.39                      # past 95% of the 0.4 range
    assert env.task_metric(_traj(opened)) == pytest.approx(0.25)


def test_cabinet_rungs_complete_in_order_only():
    """The drawer being open counts for nothing while rung one is incomplete --
    upstream's ladder asks the questions one at a time, and so does the metric. Once
    the door opens, the already-open drawer completes on the NEXT arrival (one advance
    per state, upstream's once-per-step check)."""
    env = _metric_only(HH.H1HandCabinet)
    n = 214
    st = np.zeros((30, n))
    st[:, 76] = 0.44                           # drawer open from the start
    assert env.task_metric(_traj(st)) == 0.0, "an out-of-order drawer must not count"

    st[6:, 79] = -0.39                         # the pull door opens at arrival 6
    assert env.task_metric(_traj(st)) == pytest.approx(0.5)

    # and the cube rungs, chained: drawer cube into the middle box, lateral cube up top
    st[10:, 81:84] = [0.9, 0.0, 0.94]
    st[14:, 88:91] = [0.9, 0.0, 1.54]
    assert env.task_metric(_traj(st)) == pytest.approx(1.0)


def _bookshelf_states(n, ids=(-24, -20, -18, -16, -13)):
    """States whose tail carries a fixed draw: five object ids and the five base goals
    in order. Objects themselves start at the origin (nowhere near any goal)."""
    st = np.zeros((n, 328))
    st[:, 2] = 0.98
    st[:, 3] = 1.0
    st[:, 308:313] = ids
    st[:, 313:328] = np.asarray(HH._BOOKSHELF_GOALS).ravel()
    return st


def test_bookshelf_counts_ordered_placements_and_pays_nothing_for_hovering():
    """The reward's hover-hand term pays ~0.4 per step for a palm parked against the
    current object, and its proximity term is partly prepaid at spawn. The metric pays
    only inside the 0.15 m placement radius, object by object, in the drawn order."""
    env = _metric_only(HH.H1HandBookshelfHard)
    hover = _bookshelf_states(40)
    assert env.task_metric(_traj(hover)) == 0.0

    placed = _bookshelf_states(40)
    q = HH._bookshelf_object_qpos(-24)         # the first drawn object is book_a
    placed[10:, q] = HH._BOOKSHELF_GOALS[0]
    assert env.task_metric(_traj(placed)) == pytest.approx(0.2)


def test_bookshelf_places_the_drawn_objects_in_the_drawn_order():
    """Placing a LATER object at its spot early earns nothing until its turn comes --
    upstream's counter asks about exactly one object at a time, and the metric reads
    the same ids and goals off the observation tail."""
    env = _metric_only(HH.H1HandBookshelfHard)
    st = _bookshelf_states(40)
    q3 = HH._bookshelf_object_qpos(-18)        # the THIRD drawn object, placed early
    st[:, q3] = HH._BOOKSHELF_GOALS[2]
    assert env.task_metric(_traj(st)) == 0.0

    q1 = HH._bookshelf_object_qpos(-24)
    q2 = HH._bookshelf_object_qpos(-20)
    st[10:, q1] = HH._BOOKSHELF_GOALS[0]       # then one and two arrive in order,
    st[20:, q2] = HH._BOOKSHELF_GOALS[1]       # and three is already in place
    assert env.task_metric(_traj(st)) == pytest.approx(3 / 5)


def test_package_metric_scores_the_hover_exploit_zero():
    """The strongest local optimum on this task, falsified by construction.

    `Package.get_reward` pays `stand_reward * small_control` (about 1) plus
    `package_height = min(package_z, 1)` (up to another 1) every step, while success
    TERMINATES for a one-off +1000. So holding the package up just outside the success
    radius is worth about 1700 over the episode -- more than delivering, and past the
    task's own success_bar of 1500.

    A best-approach metric would score that near 1.0, which is why this one is DELIVERY.
    Pinned with two trajectories that differ only in whether the package ever enters the
    radius: the hover sits 0.15 m away holding the package at z = 1.2 (maximal reward
    height) and must read exactly 0.0.
    """
    env = _metric_only(HH.H1HandPackage)
    n = env.n_q + env.n_v + env.extra_dims
    dest = np.array([1.0, 0.0, 0.35])

    hover = np.zeros((11, n))
    hover[:, 164:167] = dest
    hover[:, 76:79] = dest + np.array([0.15, 0.0, 0.85])   # 0.15 m out, held at z=1.2
    assert env.task_metric(_traj(hover)) == 0.0, (
        "holding the package up outside the radius must score 0.0 -- it is the "
        "behaviour the shipped reward pays most for")

    delivered = hover.copy()
    delivered[5:, 76:79] = dest                            # set down at the marker
    m = env.task_metric(_traj(delivered))
    assert m > 0.0, "a delivered package must score above zero"
    assert m == pytest.approx(6 / 10), (
        "the metric is the fraction of ARRIVING states delivered: 6 of the 10 arriving "
        f"states here, not {m}")


def test_package_refuses_a_tail_it_cannot_restore_rather_than_skipping_it():
    """A silent no-op here is invisible to every other test in this file.

    `_step` is documented stateless, and for this task the destination lives in
    `model.body_pos[-1]` -- MODEL state that survives across calls. So a `_restore_extra`
    that quietly does nothing does not fail: it leaves the PREVIOUS episode's destination
    in place and makes `_step(s, a)` depend on what was stepped before it.
    `test_step_is_a_function_of_its_arguments_and_the_warmstart_is_why` cannot catch that,
    because it never supplies a malformed tail -- the defect only exists on the path that
    test does not walk.

    The second assertion is the control, and it is the half that makes the first mean
    something. A guard that raised unconditionally would satisfy the first assertion just
    as well, so the right-sized tail must get PAST the size check -- reaching `self._env`,
    which `_metric_only` deliberately never sets. `AttributeError` there is the proof that
    the refusal is about the SIZE and not about the call.
    """
    env = _metric_only(HH.H1HandPackage)

    for bad in (np.zeros(2), np.zeros(4), np.zeros(0)):
        with pytest.raises(ValueError, match=r"_restore_extra needs the 3 values"):
            env._restore_extra(bad)

    with pytest.raises(AttributeError):
        env._restore_extra(np.zeros(HH.H1HandPackage.extra_dims))


def test_package_metric_ignores_height_and_standing():
    """The other two entries in the inventory, falsified by the same metric.

    `package_height` pays for lifting anywhere in the arena and the additive stand term
    pays for standing still; neither moves the package to the marker, and the metric
    reads both as 0.0.
    """
    env = _metric_only(HH.H1HandPackage)
    n = env.n_q + env.n_v + env.extra_dims
    lifted = np.zeros((6, n))
    lifted[:, 164:167] = np.array([2.0, 2.0, 0.35])
    lifted[:, 76:79] = np.array([-2.0, -2.0, 1.5])         # high, and as far as it gets
    assert env.task_metric(_traj(lifted)) == 0.0

    still = np.zeros((6, n))
    still[:, 164:167] = np.array([2.0, 2.0, 0.35])
    still[:, 76:79] = np.array([0.5, 0.0, 0.35])           # package untouched at spawn
    assert env.task_metric(_traj(still)) == 0.0


def test_door_metric_needs_the_doorway_not_the_approach():
    """Traversal, from geometry: the panel is at x = 0.8 with half-thickness 0.07.

    Falsifies the two entries it claims to. Standing still scores 0.0 despite
    `0.1 * stand_reward * small_control` paying for it, and walking up to the closed door
    scores 0.0 despite `passage_reward`'s margin of 1 paying partial credit from x = 0.2.

    The bound is the panel's far face and NOT upstream's `passage_reward` bound of 1.2 --
    borrowing the reward's constant is what this file's metrics refuse. Pinned at 0.86
    (inside the panel's own thickness, so not through) against 0.88.
    """
    env = _metric_only(HH.H1HandDoor)
    n = env.n_q + env.n_v
    assert env.extra_dims == 0, "door's two DOFs are real joints; it needs no tail"

    for x in (0.0, 0.5, 0.8, 0.86):
        s = np.zeros((6, n))
        s[:, 0] = x
        assert env.task_metric(_traj(s)) == 0.0, f"pelvis at x={x} is not through"

    through = np.zeros((11, n))
    through[6:, 0] = 0.88
    assert env.task_metric(_traj(through)) == pytest.approx(5 / 10)


def test_door_metric_does_not_pretend_to_catch_the_barge():
    """The honesty half, on sit_simple's precedent.

    `barge_without_handle` is recorded as an OPEN exploit that the metric deliberately
    does not close: getting through the doorway is the task, and handle use is the
    reward's preference. This asserts that posture rather than a detection -- a
    traversing state scores the same whether the handle moved or not -- so that a future
    change which quietly started requiring the handle would fail here and have to argue
    for itself in the spec too.
    """
    env = _metric_only(HH.H1HandDoor)
    n = env.n_q + env.n_v
    barge = np.zeros((6, n))
    barge[:, 0] = 1.0            # through the doorway
    barge[:, 76] = 1.2           # panel open
    barge[:, 77] = 0.0           # handle NEVER touched
    used = barge.copy()
    used[:, 77] = 1.5            # handle fully worked
    assert env.task_metric(_traj(barge)) == env.task_metric(_traj(used)) == 1.0, (
        "the metric is traversal and must not silently start requiring the handle; the "
        "spec records the barge as open and unclosed by it")

    spec = tasks.load("h1hand_door")
    barge_entry = [e for e in spec.exploits if e.get("name") == "barge_without_handle"]
    assert barge_entry and barge_entry[0].get("status") == "open"
    assert barge_entry[0].get("closed_by") is None, (
        "the inventory must not claim the metric closes it")


# --------------------------------------------------------------------------
# extra viewpoints (`output.video.n_views`, spec `judge.extra_views`)
# --------------------------------------------------------------------------

#: Every camera `assets/robots/h1hand_pos.xml` (and the h1 / h1strong variants) define,
#: verbatim from the asset. A spec view with no `pose` must name one of these, and the
#: check runs with no simulator, so a typo cannot wait for a cluster job to surface.
H1_MODEL_CAMERAS = frozenset({
    "cam_kitchen", "cam_default", "cam_maze", "cam_tabletop", "cam_hurdle",
    "cam_basketball", "cam_hand_visible", "cam_inhand", "left_eye_camera",
    "right_eye_camera",
})


def _h1_specs():
    from bird.envs.spec import views_of

    out = {}
    for task_id in tasks.available():
        if task_id.startswith("h1"):
            out[task_id] = views_of(tasks.load(task_id))
    return out


def test_every_humanoid_spec_offers_extra_views_the_asset_can_render():
    """A view that names a camera the asset lacks, or repeats the primary, would be
    refused at render time on a cluster node with nothing in CI having looked."""
    specs = _h1_specs()
    assert len(specs) >= 25, sorted(specs)   # the 24 h1hand specs and h1strong_highbar_hard
    for task_id, (primary, extras) in specs.items():
        cameras = H1_MODEL_CAMERAS
        assert primary.name in cameras, (task_id, primary.name)
        assert extras, f"{task_id}: offers no extra view; every task on this tier has a measured one"
        names = [v.name for v in extras]
        assert len(set(names)) == len(names) and primary.name not in names, task_id
        for v in extras:
            assert v.note, f"{task_id}/{v.name}: a view with no note tells the judge nothing"
            if v.pose is None:
                assert v.name in cameras, (task_id, v.name)
            else:
                pose = v.pose_dict
                assert pose.get("track_body") == "pelvis" or pose.get("lookat") is not None, (
                    task_id, v.name)


def test_the_two_trackcom_primaries_are_declared_as_tracking():
    """`cam_hurdle` and `cam_maze` are `mode="trackcom"` in the asset; the spec's
    `mode` is what the judge is TOLD about the primary panel, so `fixed` there would be
    a false statement."""
    specs = _h1_specs()
    for task_id in ("h1hand_hurdle", "h1hand_maze"):
        primary, _ = specs[task_id]
        assert primary.mode == "tracking", (task_id, primary.mode)


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", IDS)
def test_every_spec_view_renders_the_state_it_was_given(env_id, sim_envs):
    """Each extra view is a real frame of the STORED state, distinct from the primary
    and from the other views, and a pure function of the state -- the same three
    claims `tests/test_metaworld.py` makes for the Meta-World views."""
    env = sim_envs[env_id]
    a = env.reset(0)
    b = a.copy()
    b[2] += 0.6
    primary = env.render(a)
    seen = [primary]
    assert env.extra_views, env_id
    for view in env.extra_views:
        frame = env.render_view(a, view)
        assert frame.shape == primary.shape and frame.dtype == np.uint8, view.name
        assert frame.std() > 1.0, f"{view.name}: a uniform frame means the scene never rendered"
        for other in seen:
            assert not np.array_equal(frame, other), f"{view.name} duplicates another panel"
        assert np.array_equal(env.render_view(a, view), frame), f"{view.name} is not pure"
        assert not np.array_equal(env.render_view(b, view), frame), (
            f"{view.name}: two different states rendered identically")
        seen.append(frame)


@pytest.mark.humanoid
def test_a_posed_view_follows_the_robot(sim_envs):
    """A `track_body` view is aimed at the body's centre of mass for THIS frame: the
    robot translated 4 m renders the same silhouette in the same place, where a
    MuJoCo tracking camera built per frame would have stayed aimed at the origin."""
    from bird.envs.base import View

    env = sim_envs[IDS[0]]
    view = View.from_mapping({"name": "probe", "mode": "tracking",
                              "pose": {"track_body": "pelvis", "azimuth": 90,
                                       "elevation": -10, "distance": 3}})
    a = env.reset(0)
    b = a.copy()
    b[0] += 4.0
    fa, fb = env.render_view(a, view), env.render_view(b, view)
    # the robot is the dark silhouette: same number of dark pixels, in the same columns
    dark_a, dark_b = (fa.mean(axis=-1) < 60), (fb.mean(axis=-1) < 60)
    assert abs(int(dark_a.sum()) - int(dark_b.sum())) < 0.1 * max(1, int(dark_a.sum()))
    cols_a = np.flatnonzero(dark_a.any(axis=0))
    cols_b = np.flatnonzero(dark_b.any(axis=0))
    assert abs(int(cols_a.mean()) - int(cols_b.mean())) < 8


# --------------------------------------------------------------------------
# `legend_lines`: what the task ASKS, never what the metric answers
# --------------------------------------------------------------------------
#
# `EnvAdapter.legend_lines` is the contract: the lines name the command, the target,
# the phase -- and never a verdict, because a label on a frame scored by
# `evaluate.fitness.source: vlm_score` must not carry the answer. The verdict vocabulary
# below is banned unconditionally.
#
# Every check here but the last two is sim-free, through `_metric_only`: `legend_lines`
# reads class attributes and the state array only, which is the property that makes it
# a pure function of the state -- and an adapter with no `_env` at all is the sharpest
# test of that property there is. The two marked tests check the one thing the sim-free
# half cannot: that the tail slots the legend prints are the ones upstream's task object
# actually fills.

#: EVERY concrete adapter in the module, from its own `__all__` -- not `ADAPTERS`, which
#: lists 21 of the 25 (stand, run, stair and slide are tested unbound elsewhere). The
#: registry test below pins this list against `registry.names("env")`, so a task
#: registered without a legend fails here by name.
LEGEND_CLASSES = tuple((getattr(HH, n).name, getattr(HH, n)) for n in HH.__all__)
LEGEND_IDS = [n for n, _ in LEGEND_CLASSES]
assert set(IDS) < set(LEGEND_IDS)

#: The font sheet: the only characters a legend line may use. Anything else draws as a
#: box (`hud._TOFU`), which is visible but says nothing.
LEGEND_CHARS = frozenset(hud._SHEET_CHARS)
#: `EnvAdapter.legend_lines`: a 5x7 glyph at 2x, six scaled pixels a character, so about
#: 28 fit a 360-wide frame.
LEGEND_MAX_CHARS = 28
#: Words that name an ANSWER. No line may carry one; `DIST` catches `DISTANCE` too.
LEGEND_VERDICT_WORDS = ("TASK METRIC", "SUCCESS", "DIST", "FITNESS", "REWARD", "SCORE",
                        "DONE", "FAIL", "REACHED", "CLEARED")

#: (env id, the tail slots that ARE the per-episode or per-phase ask, a value for them
#: that differs from the resting state's zeros). The legend must move when these move
#: and must not move when anything else does.
LEGEND_TARGET_SLOTS = (
    ("h1hand_push", slice(164, 167), (0.9, -0.4, 1.0)),
    ("h1hand_reach", slice(154, 157), (1.5, -1.0, 1.7)),
    ("h1hand_package", slice(164, 167), (-1.5, 1.2, 0.35)),
    ("h1hand_bookshelf_hard", slice(313, 316), (0.85, 0.05, 0.35)),   # goal of object 0
    ("h1hand_basketball", slice(164, 165), (1.0,)),                    # the stage latch
    ("h1hand_cabinet", slice(213, 214), (3.0,)),                       # the ladder rung
)

#: (env id, tail slots that are NOT an ask, a non-zero value, why). The legend must be
#: identical across these -- each is documented in `legend_lines` as left out on
#: purpose, and a test is what keeps "on purpose" from decaying into "forgot".
LEGEND_NOT_AN_ASK = (
    ("h1hand_reach", slice(151, 154), (0.5, 0.5, 1.2),
     "the hand's own position is the answer, not the ask"),
    ("h1hand_window", slice(171, 174), (0.1, 0.1, 1.7),
     "head_pos0 is the reward's anchor, not a target"),
    ("h1hand_cube", slice(177, 181), (0.0, 1.0, 0.0, 0.0),
     "the target cube is drawn in the scene at that orientation"),
    ("h1hand_maze", slice(151, 152), (2.0,),
     "the junction markers are painted from the stage"),
    ("h1hand_bookshelf_hard", slice(308, 313), (-13.0, -14.0, -15.0, -16.0, -17.0),
     "the chosen objects wear their placement tint in the scene"),
)


def _legend_states(cls):
    """A resting state for `cls`, plus one per corner and the midpoint of the tail's own
    sample box -- the widest numbers a legitimate state will ever ask the legend to
    print, which is where a 28-character budget breaks first."""
    base = _blank(cls, 1)[0]
    out = [base]
    if cls.extra_dims:
        lo, hi = _metric_only(cls)._extra_bounds()
        for tail in (lo, hi, (lo + hi) / 2.0):
            s = base.copy()
            s[cls.n_q + cls.n_v:] = tail
            out.append(s)
    return out


def _assert_legible(env_id, lines):
    assert isinstance(lines, list) and lines, f"{env_id}: an empty legend"
    for line in lines:
        assert isinstance(line, str) and line.strip(), (env_id, line)
        assert len(line) <= LEGEND_MAX_CHARS, (env_id, line, len(line))
        assert set(line) <= LEGEND_CHARS, (env_id, line, sorted(set(line) - LEGEND_CHARS))
        assert line == line.upper(), (env_id, line)


def _other_robot_pose(cls, s):
    """`s` with the ROBOT somewhere else entirely -- translated, crouched, joints bent,
    moving -- and the tail untouched. Nothing about the ask has changed."""
    t = s.copy()
    t[0] += 3.0
    t[2] = 0.3
    t[7:30] = 0.4
    t[cls.n_q:cls.n_q + 3] = 2.0
    return t


def test_every_registered_h1_env_has_a_class_here_and_a_non_empty_legend():
    """The registry is the authority on which envs exist; every `h1hand_*` / `h1strong_*`
    entry must map to a class in `HH.__all__` that says what it asks. A task registered
    without an ask would have `legend_lines` say nothing, silently."""
    registry.load_all()
    registered = {n for n in registry.names("env") if n.startswith(("h1hand_", "h1strong_"))}
    assert registered == set(LEGEND_IDS), (sorted(registered ^ set(LEGEND_IDS)))
    for env_id, cls in LEGEND_CLASSES:
        assert isinstance(cls._ask, tuple) and cls._ask, f"{env_id} declares no `_ask`"
        lines = _metric_only(cls).legend_lines(_blank(cls, 1)[0])
        assert lines and all(isinstance(x, str) for x in lines), env_id


@pytest.mark.parametrize("env_id,cls", LEGEND_CLASSES, ids=LEGEND_IDS)
def test_the_legend_fits_the_font_on_every_class_at_the_tails_own_extremes(env_id, cls):
    """Upper-case, at most 28 characters, only `hud._SHEET_CHARS` -- on the resting state
    AND at the corners of the tail's sample box, where a `-2.00` is five characters
    wide and a `GOAL X -2.00 Y -2.00 Z -0.50` line is exactly the budget."""
    env = _metric_only(cls)
    for s in _legend_states(cls):
        _assert_legible(env_id, env.legend_lines(s))


@pytest.mark.parametrize("env_id,cls", LEGEND_CLASSES, ids=LEGEND_IDS)
def test_the_legend_names_the_ask_and_never_a_verdict(env_id, cls):
    """No line carries `task_metric`'s vocabulary. The frames are the VLM judge's input;
    a legend that said SUCCESS or DIST 0.12 would make its score a copy of the ground
    truth. Unconditional: nothing is grandfathered."""
    env = _metric_only(cls)
    for s in _legend_states(cls):
        for line in env.legend_lines(s):
            for word in LEGEND_VERDICT_WORDS:
                assert word not in line, (env_id, line, word)


@pytest.mark.parametrize("env_id,slots,value", LEGEND_TARGET_SLOTS,
                         ids=[e for e, _, _ in LEGEND_TARGET_SLOTS])
def test_the_legend_moves_with_the_target_and_only_with_the_target(env_id, slots, value):
    """Two states differing ONLY in the target slots give two legends; two states
    differing in everything BUT the target give one. Together they say the per-episode
    ask is read from the state's tail and from nothing else -- not from `self._env`
    (there is none here) and not from where the robot happens to be."""
    cls = dict(LEGEND_CLASSES)[env_id]
    env = _metric_only(cls)
    a = _blank(cls, 1)[0]
    b = a.copy()
    b[slots] = value
    assert env.legend_lines(a) != env.legend_lines(b), env_id
    assert env.legend_lines(_other_robot_pose(cls, a)) == env.legend_lines(a), env_id
    assert env.legend_lines(_other_robot_pose(cls, b)) == env.legend_lines(b), env_id
    # and the fixed lines are the SAME lines in both -- only the ask's moving part moved
    assert env.legend_lines(b)[:len(cls._ask)] == list(cls._ask)


@pytest.mark.parametrize("env_id,cls", LEGEND_CLASSES, ids=LEGEND_IDS)
def test_a_fixed_ask_does_not_move_with_the_robot(env_id, cls):
    """For every class: the robot elsewhere, crouched and moving leaves the legend
    identical. A legend that tracked the pelvis would be printing the answer to walk."""
    env = _metric_only(cls)
    a = _blank(cls, 1)[0]
    assert env.legend_lines(_other_robot_pose(cls, a)) == env.legend_lines(a)


@pytest.mark.parametrize("env_id,slots,value,why", LEGEND_NOT_AN_ASK,
                         ids=[e for e, _, _, _ in LEGEND_NOT_AN_ASK])
def test_the_legend_ignores_tail_slots_that_are_not_an_ask(env_id, slots, value, why):
    """Each of these tails is carried state and NOT a command, for the reason in the
    table -- and `legend_lines`' docstring names every one. If this fails, either a
    slot became an ask (update the table and the docstring together) or the legend has
    started printing an answer."""
    cls = dict(LEGEND_CLASSES)[env_id]
    env = _metric_only(cls)
    a = _blank(cls, 1)[0]
    b = a.copy()
    b[slots] = value
    assert env.legend_lines(a) == env.legend_lines(b), (env_id, why)


def test_the_fixed_asks_are_built_from_the_metrics_own_constants():
    """A speed on the frame that disagreed with the speed the metric normalises by would
    be a comment contradicting the code, on every frame of every run. So the lines are
    built FROM the constants, and this pins that they still are."""
    assert HH.H1HandWalk._ask == (f"WALK +X AT {HH._LOCO_V_REF:.1f} M/S",)
    assert HH.H1HandRun._ask == (f"RUN +X AT {HH._RUN_V_REF:.1f} M/S",)
    assert f"{HH._POLE_V_REF:.1f} M/S" in HH.H1HandPole._ask[0]
    assert f"{HH._LOCO_V_REF:.1f} M/S" in HH.H1HandStair._ask[0]
    assert f"{HH._LOCO_V_REF:.1f} M/S" in HH.H1HandSlide._ask[0]
    assert f"{HH._LOCO_V_REF:.1f} M/S" in HH.H1HandCrawl._ask[0]
    assert "M/S" not in " ".join(HH.H1HandStand._ask)       # stand asks for NO speed
    assert str(len(HH._HURDLE_WALLS_X)) in HH.H1HandHurdle._ask[0]
    assert str(len(HH._ROOM_OBJECTS)) in HH.H1HandRoom._ask[0]
    assert HH.H1HandMaze._ask == ("MAZE: 3,0 THEN 3,6 THEN 6,6",)
    assert HH.H1HandMaze._ask[0] == "MAZE: " + " THEN ".join(
        f"{cx:.0f},{cy:.0f}" for cx, cy in HH._MAZE_CHECKPOINTS)
    assert HH.H1HandBasketball._ask == (
        f"HOOP AT X {HH._HOOP_POS[0]:.1f} Z {HH._HOOP_POS[2]:.2f}",)
    # stair and slide share a metric and differ in the ASSET, so their asks name it
    assert HH.H1HandStair._ask != HH.H1HandSlide._ask
    assert HH.H1HandBalanceSimple._ask == HH.H1HandBalanceHard._ask


def test_basketball_legend_says_which_stage_is_in_force():
    """The stage latch IS an ask -- catch until first contact, throw after -- and the
    legend binarises it at 0.5 exactly as `_restore_extra` does, so the frame and the
    reward can never disagree about which stage a state is in."""
    env = _metric_only(HH.H1HandBasketball)
    s = _bb_states(1)[0]
    s[164] = 0.0
    assert env.legend_lines(s) == [HH.H1HandBasketball._ask[0], "CATCH THE BALL"]
    s[164] = 1.0
    assert env.legend_lines(s) == [HH.H1HandBasketball._ask[0], "THROW TO THE HOOP"]
    s[164] = 0.5
    assert env.legend_lines(s)[-1] == "THROW TO THE HOOP"
    s[164] = 0.49
    assert env.legend_lines(s)[-1] == "CATCH THE BALL"


def test_cabinet_legend_names_the_current_rung_and_falls_silent_at_success():
    """One command per rung, in `_cabinet_rung_done`'s order, and NO count: `3/4` would
    be progress, which is the metric's business. Rung five is the success terminal --
    the episode's last frame -- and adds nothing to the fixed ask."""
    assert len(HH._CABINET_STEP_ASKS) == 4        # one per rung test in _cabinet_rung_done
    env = _metric_only(HH.H1HandCabinet)
    s = _blank(HH.H1HandCabinet, 1)[0]
    for rung in range(1, 5):
        s[213] = rung
        assert env.legend_lines(s) == [*HH.H1HandCabinet._ask, HH._CABINET_STEP_ASKS[rung - 1]]
        assert "/4" not in env.legend_lines(s)[-1]
    s[213] = 5.0
    assert env.legend_lines(s) == list(HH.H1HandCabinet._ask)
    s[213] = 0.0                                   # below the ladder: clipped to rung one
    assert env.legend_lines(s)[-1] == HH._CABINET_STEP_ASKS[0]


def test_bookshelf_legend_follows_the_counter_through_the_goal_list():
    """The current object's spot, selected by the carried counter from the carried goal
    list -- the same slots `task_metric` reads. At the terminal count the last goal
    stands, as upstream's duplicate sixth entry does, rather than an index error."""
    env = _metric_only(HH.H1HandBookshelfHard)
    s = _bookshelf_states(1)[0]
    goals = np.asarray(HH._BOOKSHELF_GOALS)
    for k in range(5):
        s[307] = k
        assert env.legend_lines(s)[-1] == "GOAL " + HH._legend_xyz(goals[k])
    s[307] = 5.0
    assert env.legend_lines(s)[-1] == "GOAL " + HH._legend_xyz(goals[4])


def test_legend_xyz_rounds_to_centimetres_and_never_prints_negative_zero():
    assert HH._legend_xyz(np.array([0.85, -0.3, 1.0])) == "X 0.85 Y -0.30 Z 1.00"
    assert HH._legend_xyz(np.array([-1e-9, 0.004, -0.004])) == "X 0.00 Y 0.00 Z 0.00"
    assert HH._legend_xyz(np.array([-2.0, -2.0, -0.5])) == "X -2.00 Y -2.00 Z -0.50"
    assert len("GOAL " + HH._legend_xyz(np.array([-2.0, -2.0, -0.5]))) == LEGEND_MAX_CHARS


#: Where upstream's task object holds the goal the legend prints, per env -- the one
#: fact the sim-free half cannot check: that the TAIL SLOTS are the right ones.
_LEGEND_UPSTREAM_GOAL = {
    "h1hand_push": lambda env: env._env.task.goal,
    "h1hand_reach": lambda env: env._env.task.goal,
    "h1hand_package": lambda env: env._env.named.data.site_xpos["destination_loc"],
    "h1hand_bookshelf_hard": lambda env: env._env.task.placement_goals[0],
}


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", sorted(_LEGEND_UPSTREAM_GOAL))
def test_the_legend_prints_the_goal_upstream_actually_holds(env_id, sim_envs):
    """After a seeded reset the GOAL line equals the goal upstream's own task object
    (or model site) holds -- so the slots the legend reads are the slots `_extra_state`
    fills. Then the simulator is moved to a DIFFERENT state and the same `s` is asked
    again: the legend must not have noticed, because it never read the simulator."""
    env = sim_envs[env_id]
    s = env.reset(3)
    goal = np.asarray(_LEGEND_UPSTREAM_GOAL[env_id](env), dtype=float).copy()
    lines = env.legend_lines(s)
    assert "GOAL " + HH._legend_xyz(goal) in lines, (env_id, lines, goal)
    env.render(env.reset(4))
    assert env.legend_lines(s) == lines, env_id


@pytest.mark.humanoid
@pytest.mark.parametrize("env_id", IDS)
def test_a_live_reset_states_legend_fits_and_ignores_what_the_simulator_holds(env_id, sim_envs):
    env = sim_envs[env_id]
    a = env.reset(0)
    lines = env.legend_lines(a)
    _assert_legible(env_id, lines)
    b = a.copy()
    b[0] += 4.0
    b[2] += 0.6
    env.render(b)
    assert env.legend_lines(a) == lines, env_id
