"""`records/iterNN.json`: the carried objects, not only their counts.

`state/iterNN.json` is `RunState.summary()` -- six of a run's richest objects
survive it only as a LENGTH (`n_dialogue: 9`, `n_preferences: 14`, ...), so a
read-only reader of run dirs could say a run held fourteen preferences and never
show one. The records file is the other half: the objects
themselves, verbatim where they are cheap (dialogue, subtasks, failure memory),
as labels plus trajectory summaries for preferences, and as pointer+stats rows
for the trajectory store and the archive.

Two invariants carry the file's meaning, and both fail silently if unpinned:

  * POINTERS, NEVER PAYLOADS. A trajectory's states/actions/rewards must not
    land here -- `save_train_result` drops them on size grounds and the same
    argument holds unchanged. If those keys appear, the artifact has quietly
    become the dump that decision refused.
  * EMPTY IS NOT MISSING. Every slot is present in every file, `[]` where the
    method carries nothing, so "this run predates records/" and "this method
    carries nothing" stay distinguishable -- the `save_report` convention.
"""

import json

from bird.artifacts import RunDir
from bird.state import ArchiveCell, RunState
from bird.types import Candidate, CandidateReport, Preference, TrainResult, Trajectory

#: The trajectory-summary keys that would mean the payload leaked in.
_PAYLOAD_KEYS = {"states", "actions", "rewards", "component_values"}


def _traj(video=None):
    return Trajectory(states=[[0.0], [1.0]], actions=[[0.1]], rewards=[1.0, 2.0],
                      component_values={"prox": [0.5, 0.6]},
                      success=True, length=2, ret=3.0, video_path=video)


def _report(cand_id="c0001", iteration=2, fitness=0.7):
    cand = Candidate(cand_id=cand_id, iteration=iteration, reward_code="x = 1")
    res = TrainResult(cand_id=cand_id, candidate=cand, trajectories=[_traj()])
    return CandidateReport(cand_id=cand_id, candidate=cand, result=res, fitness=fitness)


def _state():
    s = RunState(iteration=3, restart=1)
    s.dialogue = [{"role": "assistant", "content": "def r(): ..."},
                  {"role": "user", "content": "fitness 0.7; tighten the goal term"}]
    s.preferences = [Preference("c0001", "c0002", 1,
                                left_traj=_traj("videos/c0001"), right_traj=None,
                                source="selection", iteration=2)]
    s.trajectory_store = [_traj("videos/c0001"), _traj()]
    s.subtasks = ["reach", "grasp"]
    s.failure_memory = [{"cand_id": "c0000", "failure": "SyntaxError: bad",
                         "failure_kind": "invalid", "reward_code": "def r(: ..."}]
    s.archive = {
        (0, 1, 2): ArchiveCell(coords=(1, 2), island=0, report=_report(), fitness=0.7),
        # An unscored occupant keeps ArchiveCell's -inf default.
        (1, "elite", 0): ArchiveCell(coords=("elite", 0), island=1, report=None),
    }
    return s


def _records(rd, iteration):
    return json.loads((rd.path / "records" / f"iter{iteration:02d}.json").read_text())


def test_the_records_file_holds_the_objects_the_counts_summarise(tmp_path):
    rd = RunDir(tmp_path, "t", "deadbeef")
    rd.save_records(_state(), 3)
    rd.close()
    out = _records(rd, 3)

    assert out["schema"] == 1
    assert out["iteration"] == 3 and out["restart"] == 1

    # Verbatim slots: the prose IS the artifact, not its length.
    assert out["dialogue"] == [
        {"role": "assistant", "content": "def r(): ..."},
        {"role": "user", "content": "fitness 0.7; tighten the goal term"}]
    assert out["subtasks"] == ["reach", "grasp"]
    assert out["failure_memory"][0]["failure"] == "SyntaxError: bad"
    assert out["failure_memory"][0]["reward_code"] == "def r(: ..."

    (pref,) = out["preferences"]
    assert (pref["left_id"], pref["right_id"], pref["label"]) == ("c0001", "c0002", 1)
    assert pref["source"] == "selection" and pref["iteration"] == 2
    assert pref["left"]["video_path"] == "videos/c0001"
    assert pref["left"]["mean_per_step_return"] == 1.5
    assert pref["right"] is None, "an absent side is null, not a fabricated summary"

    rows = out["trajectory_store"]
    assert [r["index"] for r in rows] == [0, 1]
    assert rows[0]["video_path"] == "videos/c0001" and rows[1]["video_path"] is None
    assert rows[0]["success"] is True and rows[0]["length"] == 2
    assert rows[0]["components"] == ["prox"]
    assert rows[0]["has_states"] and rows[0]["has_actions"]


def test_pointers_never_payloads(tmp_path):
    """The size argument that drops trajectories from train_result.json holds here."""
    rd = RunDir(tmp_path, "t", "deadbeef")
    rd.save_records(_state(), 3)
    rd.close()
    out = _records(rd, 3)

    summaries = list(out["trajectory_store"])
    summaries += [p[side] for p in out["preferences"] for side in ("left", "right")
                  if p[side] is not None]
    assert summaries, "the fixture must exercise at least one summary"
    for s in summaries:
        leaked = _PAYLOAD_KEYS & set(s)
        assert not leaked, (
            f"trajectory payload leaked into records/: {sorted(leaked)}. "
            "save_train_result drops these on size grounds; records/ must too")


def test_every_slot_is_present_even_when_empty(tmp_path):
    """`[]` where the method carries nothing -- absent would mean 'predates records/'."""
    rd = RunDir(tmp_path, "t", "deadbeef")
    rd.save_records(RunState(), 0)
    rd.close()
    out = _records(rd, 0)
    for slot in ("dialogue", "preferences", "trajectory_store", "subtasks",
                 "failure_memory", "archive"):
        assert out[slot] == [], f"{slot} must be an empty list, got {out[slot]!r}"


def test_archive_rows_are_pointers_and_minus_inf_is_null(tmp_path):
    """The occupant's program lives under candidates/; the cell names it.

    A `-inf` fitness is ArchiveCell's 'unscored' sentinel, and `Infinity` is not
    valid JSON -- strict JSON consumers (anything but Python's lenient reader)
    reject it -- so it is recorded as null, keeping None-vs-number distinct
    exactly as `.fitness` does.
    """
    rd = RunDir(tmp_path, "t", "deadbeef")
    rd.save_records(_state(), 3)
    rd.close()
    rows = {tuple(r["coords"]): r for r in _records(rd, 3)["archive"]}

    scored = rows[(1, 2)]
    assert scored["cand_id"] == "c0001" and scored["iteration"] == 2
    assert scored["fitness"] == 0.7 and scored["island"] == 0

    elite = rows[("elite", 0)]
    assert elite["cand_id"] is None and elite["iteration"] is None
    assert elite["fitness"] is None, "-inf must land as null, never as Infinity"
    assert elite["island"] == 1


# --------------------------------------------------------------------------
# End to end: the file appears beside the snapshot, and the two agree
# --------------------------------------------------------------------------


def _run(tmp_path, **overrides):
    import importlib.util

    from conftest import REPO
    from bird.config import load

    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.run(load("eureka", profile="tester",
                   overrides={"seed": 0, "loop.n_iterations": 2, **overrides}),
              out_root=str(tmp_path))
    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    return run


def test_counts_and_records_agree_on_a_real_run(tmp_path):
    """The snapshot's counts are kept ON PURPOSE: they are the alarm.

    Written from the same RunState in the same call sequence, never from each
    other -- so a disagreement here means one of the two writers drifted,
    which is exactly what the redundancy exists to catch.
    """
    run = _run(tmp_path)
    states = sorted((run / "state").glob("iter*.json"))
    assert states, "the tester run should snapshot every iteration"

    pairs = 0
    for snap_path in states:
        rec_path = run / "records" / snap_path.name
        assert rec_path.exists(), f"{snap_path.name} has a snapshot but no records file"
        snap = json.loads(snap_path.read_text())
        rec = json.loads(rec_path.read_text())
        for count_key, slot in (("n_dialogue", "dialogue"),
                                ("n_preferences", "preferences"),
                                ("n_trajectories", "trajectory_store"),
                                ("n_subtasks", "subtasks"),
                                ("n_failures_remembered", "failure_memory"),
                                ("archive_occupied", "archive")):
            assert snap[count_key] == len(rec[slot]), (
                f"{snap_path.name}: {count_key}={snap[count_key]} but records/ "
                f"holds {len(rec[slot])} {slot} -- the count and the records "
                "it summarises disagree")
        pairs += 1
    assert pairs == 2

    # eureka carries `dialogue`, so the richest slot must actually be there --
    # a records file that is all `[]` on a dialogue-carrying method would pass
    # every shape check above while recording nothing.
    last = json.loads((run / "records" / states[-1].name).read_text())
    assert last["dialogue"], (
        "eureka under the tester profile carries loop.carry: [best_reward, dialogue]; an empty "
        "dialogue in records/ means the objects were dropped before writing")


def test_records_respect_the_save_state_gate(tmp_path):
    """One gate for the snapshot and its records: output.save_state_every_iteration."""
    run = _run(tmp_path, **{"output.save_state_every_iteration": False,
                            "loop.n_iterations": 1})
    assert not list((run / "state").glob("iter*.json"))
    assert not (run / "records").exists(), (
        "records/ must ride the same gate as state/ -- a run that asked for "
        "no snapshots must not pay for their records either")


def test_a_non_finite_return_lands_as_null_not_as_bare_nan(tmp_path):
    """`json.dumps` emits bare NaN/Infinity by default -- legal for Python's
    reader, rejected by strict JSON consumers (anything but Python's lenient
    reader). A reward returning nan is a normal thing for this project to
    produce, so the summary records null rather than a token that breaks a
    strict parser (the archive rows' -inf fitness takes the same route).
    """
    s = RunState()
    s.trajectory_store = [Trajectory(length=2, ret=float("nan")),
                          Trajectory(length=2, ret=float("inf"))]
    rd = RunDir(tmp_path, "t", "deadbeef")
    rd.save_records(s, 0)
    rd.close()

    text = (rd.path / "records" / "iter00.json").read_text()
    assert "NaN" not in text and "Infinity" not in text, (
        "bare NaN/Infinity in the artifact is invalid strict JSON")
    rows = json.loads(text)["trajectory_store"]
    assert all(r["return"] is None for r in rows)
    assert all(r["mean_per_step_return"] is None for r in rows), (
        "a mean derived from a non-finite return is not a measurement either")


def test_the_records_write_leaves_no_tmp_and_parses_whole(tmp_path):
    """The write is tmp + os.replace (the `_write_status` convention): a
    concurrent reader must never see a torn file. The tmp name does not match `iter*.json`, and nothing
    of it survives a completed write.
    """
    rd = RunDir(tmp_path, "t", "deadbeef")
    rd.save_records(_state(), 3)
    rd.close()
    assert not list((rd.path / "records").glob("*.tmp")), (
        "a completed save_records must leave no tmp sibling behind")
    assert json.loads(
        (rd.path / "records" / "iter03.json").read_text())["schema"] == 1
