"""`preferences.jsonl`: the comparisons a run made, as a dataset rather than a count.

Without it the only thing about a preference that reaches disk is
`RunState.summary()`'s `n_preferences` -- a number. The records would die with
the process, making the VLM-vs-human agreement rate a research project rather
than a query.

The tests that carry the weight are the ones about what a POOLED file can be
asked, because that is the whole point and it fails silently: a dataset that
mixes a model's judgement with an inferred one, or that cannot say which run a
candidate id belongs to, still loads fine and still answers -- wrongly.
"""

from __future__ import annotations

import json

import pytest

from conftest import load_gt_published
from bird import preference_log, registry
from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import (Candidate, CandidateReport, Preference, Trajectory,
                        TrainResult)


class _Rundir:
    def __init__(self, path):
        self.path = path


def _ctx(tmp_path, **over):
    cfg = load("gt", profile="tester", overrides=over)
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.rundir = _Rundir(tmp_path)
    return ctx


class _State:
    iteration = 0
    restart = 0


def _pref(**over):
    kw = dict(left_id="c0001", right_id="c0002", label=1, source="vlm", iteration=0)
    kw.update(over)
    return Preference(**kw)


def test_a_record_is_written_when_the_comparison_is_made(tmp_path):
    """Append-only and immediate, which is the whole reason it is a log.

    A snapshot written at the iteration boundary loses every comparison a run
    had already paid for when it is killed mid-round; a complete line written
    at the moment of judgement does not.
    """
    ctx = _ctx(tmp_path)
    preference_log.record(ctx, _pref(), _State())
    preference_log.record(ctx, _pref(left_id="c0003"), _State())
    lines = (tmp_path / "preferences.jsonl").read_text().splitlines()
    assert len(lines) == 2, "each comparison is its own line, written as it is made"
    assert all(json.loads(l)["schema"] == preference_log.SCHEMA for l in lines)


def test_judge_says_what_source_cannot(tmp_path):
    """The headline, made mechanical.

    `source` is the COMPARATOR's name: `llm_on_vlm_captions` does not advertise
    that a model judged it, and `human` names a comparator whose oracle may be
    a script. A pooled analysis filtering on `source` therefore mixes a model's
    judgement, a person's and one inferred from the ranking. `judge` is the
    field that can be filtered, and the producer states it.
    """
    ctx = _ctx(tmp_path)
    preference_log.record(ctx, _pref(source="llm_on_vlm_captions"), _State(),
                          judge="model")
    preference_log.record(ctx, _pref(source="selection"), _State(), judge="inferred")
    recs = preference_log.read(tmp_path)
    assert [r["judge"] for r in recs] == ["model", "inferred"]
    # …and the raw value is kept verbatim beside it, so the mechanism is still
    # recoverable rather than normalised away.
    assert [r["source"] for r in recs] == ["llm_on_vlm_captions", "selection"]


def test_a_producer_that_says_nothing_is_unknown_never_human(tmp_path):
    """Silence is not evidence that a person judged something."""
    ctx = _ctx(tmp_path)
    preference_log.record(ctx, _pref(), _State())
    assert preference_log.read(tmp_path)[0]["judge"] == "unknown"


def test_the_scripted_oracle_is_not_a_human(tmp_path):
    """`_ScriptedHumanOracle` is a stand-in, and the dataset has to say so.

    Letting it pass as a person's judgement would corrupt exactly the
    VLM-vs-human agreement number this log exists to make computable -- and no
    real in-run human preference has ever been collected in this repo.
    """
    registry.load_all()
    from bird.components import phases
    assert preference_log.judge_of(phases._ScriptedHumanOracle.__new__(
        phases._ScriptedHumanOracle)) == "scripted"
    assert preference_log.judge_of(phases._NullHumanOracle.__new__(
        phases._NullHumanOracle)) == "none"
    assert preference_log.judge_of(None) == "none"
    assert preference_log.judge_of(object()) == "unknown"


def test_an_abstention_is_not_a_side(tmp_path):
    """`preferred: null` means asked and not answered.

    Distinct from `"tie"`, which is a judgement that the two are equal, and
    distinct from a side -- coercing an abstention manufactures data. No in-run
    producer emits it yet; the field is shaped for it now rather than after
    there is a dataset whose schema is already fixed.
    """
    assert preference_log.side_of(1) == "left"
    assert preference_log.side_of(0) == "right"
    assert preference_log.side_of(-1) == "tie"
    assert preference_log.side_of(None) is None
    assert preference_log.side_of(7) is None, "an unknown label is not a side"

    ctx = _ctx(tmp_path)
    preference_log.record(ctx, _pref(label=None), _State())
    rec = preference_log.read(tmp_path)[0]
    assert rec["preferred"] is None
    assert preference_log.preferred_id(rec) is None
    assert preference_log.preferred_id({"preferred": "left", "left_id": "a"}) == "a"


def test_clips_are_pointers_not_payloads(tmp_path):
    """States, actions and frames never land here.

    `save_train_result`'s size argument holds unchanged: the episodes are large
    and the clips are already on disk under `videos/`.
    """
    import numpy as np
    traj = Trajectory(states=np.zeros((4, 3)), actions=np.zeros((3, 1)),
                      rewards=[0.0] * 3, success=True, length=3, ret=1.5)
    traj.video_path = "videos/c0001"
    ctx = _ctx(tmp_path)
    preference_log.record(ctx, _pref(left_traj=traj), _State())
    rec = preference_log.read(tmp_path)[0]
    assert rec["left"] == {"length": 3, "return": 1.5, "success": True,
                           "video": "videos/c0001"}
    blob = json.dumps(rec)
    assert "states" not in blob and "actions" not in blob and "rewards" not in blob


def test_records_carry_the_run_so_pooling_is_concatenation(tmp_path):
    """Candidate ids mean nothing once two runs sit in one file.

    This is what makes `pool` a generator over sources rather than a merge:
    every record already names its own run, iteration and restart, so
    concatenating is a valid dataset with no rewriting.
    """
    one, two = tmp_path / "run-a", tmp_path / "run-b"
    one.mkdir(); two.mkdir()
    for path in (one, two):
        preference_log.record(_ctx(path), _pref(), _State())
    pooled = list(preference_log.pool([one, two]))
    assert [r["run"] for r in pooled] == ["run-a", "run-b"]
    assert all(r["iteration"] == 0 and r["restart"] == 0 for r in pooled)
    # A source that made no comparisons contributes nothing and is not an error:
    # a sweep is full of runs that never ran a pairwise comparator.
    assert list(preference_log.pool([tmp_path / "never-ran"])) == []


def test_a_torn_last_line_is_skipped_not_raised_on(tmp_path):
    """The normal end of a killed run. Everything before it is still good."""
    ctx = _ctx(tmp_path)
    preference_log.record(ctx, _pref(), _State())
    with open(tmp_path / "preferences.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"schema": 1, "left_id": "c9')       # killed mid-write
    recs = preference_log.read(tmp_path)
    assert len(recs) == 1 and recs[0]["left_id"] == "c0001"


def test_a_write_failure_never_ends_a_search(tmp_path):
    """Best-effort, like the judge trace and the trajectory trace.

    A lost record is a gap in a dataset. Raising here would turn it into a lost
    run, and nothing in this module is on a correctness path.
    """
    ctx = _ctx(tmp_path)
    ctx.rundir = _Rundir(tmp_path / "file-not-a-dir")
    (tmp_path / "file-not-a-dir").write_text("in the way")
    preference_log.record(ctx, _pref(), _State())          # must not raise
    # No rundir at all -- `--dry-run`, and every hand-built Context in this suite.
    ctx.rundir = None
    preference_log.record(ctx, _pref(), _State())          # must not raise
    assert preference_log.read(tmp_path / "nothing-here") == []


def test_a_callers_extra_field_cannot_redefine_the_schema(tmp_path):
    """`**extra` is merged last but cannot take a built field.

    `judgments.note_query`'s reason: a stray key must not be able to silently
    redefine `source` or `preferred` for every reader downstream.
    """
    ctx = _ctx(tmp_path)
    preference_log.record(ctx, _pref(source="vlm"), _State(),
                          judge="model", source="forged", preferred="left",
                          t=0, annotator="someone")
    rec = preference_log.read(tmp_path)[0]
    assert rec["source"] == "vlm" and rec["preferred"] == "left"
    assert rec["t"] != 0, "the computed timestamp is not a caller's to set"
    assert rec["annotator"] == "someone", "a field the schema declares IS settable"

    # DERIVED from the record, not restated: a `_PROTECTED` list that still
    # names `ts` after the field is renamed to `t` disagrees with the writer,
    # and only this shape of check sees that. Asserting a hand-written key list
    # here would pass with that bug in place.
    computed = set(rec) - preference_log._CALLER_SETTABLE
    assert computed <= preference_log._PROTECTED, (
        "these fields are computed but a caller's **extra can overwrite them: "
        + repr(sorted(computed - preference_log._PROTECTED)))


def test_the_human_is_asked_about_the_bt_winner():
    """GT Alg. 1 l.15-17: feedback <- human(pi_best), the Bradley-Terry argmax.

    The human phase runs before deferred `preference_bt` fitness is resolved,
    so a `_rank_key` reading fitness sees None (-inf) for every trained
    candidate, the sort collapses to cand_id order, and the one human query per
    iteration -- GT's defining mechanism, injected next round as "ground truth"
    -- narrates the lexicographically-first candidate (wrong in 12/20 measured
    rounds). Worse, the -10000 failure sentinel outranks None, so a
    valid-but-training-FAILED report would be preferred over every trained one.
    Nothing else in the suite can see it: the canonical
    wrong-number-rendering-as-plausible case, which is why this test exists.
    """
    registry.load_all()
    cfg = load_gt_published(profile="tester")
    ctx = Context(cfg=cfg, budget=Budget())
    asked = []

    class _Human:
        def feedback(self, report):
            asked.append(report.cand_id)
            return f"narration about {report.cand_id}"

    ctx.human = _Human()

    def rep(cid, strength, *, fitness=None):
        """A report as the preferences phase leaves it for preference_bt."""
        cand = Candidate(cand_id=cid, iteration=0, reward_code="def f(): ...")
        res = TrainResult(cand_id=cid, candidate=cand, trained=fitness is None)
        r = CandidateReport(cand_id=cid, candidate=cand, result=res,
                            fitness=fitness, fitness_source="preference_bt")
        r.meta["bt_strength"] = strength
        if fitness is None:
            r.meta["fitness_pending"] = "bt_strength"
        return r

    reports = [
        # Lexicographically first AND carrying the sentinel: a fitness-ranked
        # rule's target on both of its failure axes at once.
        rep("c0000", 0.0, fitness=-10000.0),
        rep("c0001", 0.10),
        rep("c0002", 0.55),  # the BT argmax -- Alg. 1's pi_best
        rep("c0003", 0.35),
    ]
    state = RunState()
    registry.get("phase", "human_feedback")(ctx, state, reports)
    assert asked == ["c0002"], (
        "the one human query must land on the strength argmax, not the "
        f"lexicographically-first id or the -10000 report: asked {asked}")
    assert "c0002" in state.human_feedback


@pytest.mark.slow
def test_a_real_run_logs_every_comparison_and_the_count_agrees(tmp_path):
    """End to end, with the count-versus-records integrity check.

    `n_preferences` in the state snapshot is kept precisely so it can be
    compared against the number of persisted records: a count that disagrees
    with the records it summarises is itself an alarm. They agree here because
    `gt` carries `preference_dataset`, so state accumulates exactly
    what the log appended -- a config that does NOT carry it would reset the
    snapshot while the log kept going, which is the log being right.
    """
    import subprocess, sys, glob, pathlib
    out = subprocess.run(
        [sys.executable, "bird.py", "-c", "gt", "-p", "tester",
         "-s", "seed=11", "--out", str(tmp_path),
         # The scripted oracle too, so all three judge kinds are in one file.
         # `all_candidates` because `selected_only` yields one candidate and a
         # pair needs two.
         "-s", "evaluate.human.mode=preference_labels",
         "-s", "evaluate.human.queries_per_iteration=2",
         "-s", "evaluate.human.applies_to=all_candidates",
         # The `inferred` judge kind comes only from this loser action, which
         # gt does not pin (Alg. 1 line 21 grows D_pref by the VLM labels
         # alone). The subject here is the LOG keeping
         # three judge kinds separable, so ask for the third explicitly.
         "-s", "update.loser.action=add_to_preference_dataset"],
        capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    run = pathlib.Path(sorted(glob.glob(str(tmp_path / "*")))[0])
    recs = preference_log.read(run)
    assert recs, "a gt run makes pairwise comparisons"
    states = sorted(glob.glob(str(run / "state" / "*.json")))
    with open(states[-1], encoding="utf-8") as fh:
        last = json.load(fh)
    assert last["n_preferences"] == len(recs), (
        "the snapshot count and the log disagree, which is the alarm this check is")
    # All three kinds present and separable -- the pooled-dataset property.
    kinds = {r["judge"] for r in recs}
    assert {"model", "inferred", "scripted"} <= kinds, kinds
    assert all(r["run"] == run.name for r in recs)

    # THE case that matters, on real data rather than in the abstract:
    # every record whose `source` says "human" was in fact judged by a SCRIPT,
    # so a pooled analysis filtering on `source` would count these as human
    # data. `judge` is what makes them separable, and the annotator names the
    # stand-in.
    human_sourced = [r for r in recs if r["source"] == "human"]
    assert human_sourced, "the scripted oracle should have labelled some pairs"
    assert all(r["judge"] == "scripted" for r in human_sourced)
    assert all(r["annotator"] == "oracle:_ScriptedHumanOracle"
               for r in human_sourced)
    assert not [r for r in recs if r["judge"] == "human"], (
        "no real in-run human preference has ever been collected in this repo, "
        "and the dataset must not claim one")


def test_a_critic_vote_run_logs_every_segment_label_with_its_span(tmp_path):
    """R*'s labels reach the log, as segments, under their own judge kind.

    Appending the critic population's `Preference`s straight to
    `state.preferences` without `preference_log.record` would leave an rstar
    run with NO `preferences.jsonl` while its state snapshot counts 20
    preferences a round -- the count-versus-records alarm the test above
    describes, firing in the other direction. And a record with no place for a
    segment's span would make even a logged R* label read as a whole-rollout
    judgement.
    """
    import subprocess, sys, glob, pathlib
    out = subprocess.run(
        [sys.executable, "bird.py", "-c", "rstar", "-p", "tester",
         "-s", "seed=0", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    run = pathlib.Path(sorted(glob.glob(str(tmp_path / "*")))[0])
    recs = preference_log.read(run)
    votes = [r for r in recs if str(r["source"]).startswith("critic_vote_")]
    assert votes, "the critic population labelled segments and none reached the log"
    assert {r["judge"] for r in votes} == {"program"}, (
        "a critic is a program, not a model queried per pair and not a person")
    for r in votes:
        lo, hi = r["left_span"]
        assert isinstance(lo, int) and isinstance(hi, int) and hi > lo
        assert r["right_span"] == r["left_span"], "R* compares the same step range on both sides"
        assert r["preferred"] in ("left", "right")
    states = sorted(glob.glob(str(run / "state" / "*.json")))
    with open(states[-1], encoding="utf-8") as fh:
        last = json.load(fh)
    assert last["n_preferences"] == len(recs), (
        "the snapshot count and the log disagree, which is the alarm this check is")
