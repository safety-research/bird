"""`loop.carry` is how a config says what kind of search it is running."""

from bird.state import CARRY_COMPANIONS, CARRY_SLOTS, RunState
from bird.types import Preference, Trajectory


def _populated() -> RunState:
    s = RunState()
    s.dialogue = [{"role": "user", "content": "hi"}]
    s.preferences = [Preference("a", "b", 1)]
    s.trajectory_store = [Trajectory(success=True, length=3, ret=3.0)]
    s.subtasks = ["reach", "grasp"]
    s.replay_ref = "buf-0"
    s.policy_ref = "ckpt-0"
    s.failure_memory = [{"code": "x", "traceback": "boom"}]
    s.descriptor_stats = {"reward_ast_node_count": {"min": 3.0, "max": 40.0}}
    return s


def test_carry_keeps_only_what_is_named():
    s = _populated()
    s.apply_carry(["dialogue", "preference_dataset"])
    assert s.dialogue and s.preferences
    assert s.trajectory_store == []
    assert s.subtasks == []
    assert s.replay_ref is None
    assert s.policy_ref is None
    assert s.failure_memory == []


def test_descriptor_stats_travel_with_the_archive_slot():
    """The running binning stats are archive state (CARRY_COMPANIONS): kept when
    `archive` is carried, cleared when it is dropped. The silent failure this
    pins: stats surviving a dropped archive would bin a FRESH grid against a
    dead run's min/max, which renders as a plausible (just oddly skewed)
    archive -- the one-binning contract."""
    assert CARRY_COMPANIONS["archive"] == ("descriptor_stats",)
    kept = _populated()
    kept.apply_carry(["archive"])
    assert kept.descriptor_stats, "stats must survive when the archive does"
    dropped = _populated()
    dropped.apply_carry(["dialogue"])
    assert dropped.archive == {} and dropped.descriptor_stats == {}, (
        "stats outlived the archive they were measured for")


def test_empty_carry_clears_everything():
    s = _populated()
    s.apply_carry([])
    for attr in CARRY_SLOTS.values():
        v = getattr(s, attr)
        assert v in (None, [], {}), f"{attr} survived an empty carry: {v!r}"


def test_every_carry_slot_is_reachable():
    """The schema's loop.carry enum and CARRY_SLOTS must agree."""
    from bird.schema import SCHEMA
    allowed = set(SCHEMA["loop.carry"].item_enum)
    assert allowed == set(CARRY_SLOTS), (
        f"schema/state disagree: schema-only={allowed - set(CARRY_SLOTS)}, "
        f"state-only={set(CARRY_SLOTS) - allowed}")
