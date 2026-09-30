"""`output.judge_trace`: what the (V)LM judge was SHOWN, not just what it said.

`report.json` keeps the verdict and kept none of the evidence. Three losses,
one shape: `evaluation._vlm_frames` rendered its frames fresh (nothing is on
disk at scoring time -- `record_rollouts` runs after stage 4), sent them as
bytes and dropped them; the EVALUATOR's prompt was written nowhere, though the
generator's lands in `candidates/*/prompt.json`; and Gran Turismo's two-step
comparator consumed its contrastive caption -- the decider's entire input -- in
flight.

The test that matters most here is the blind one. A judgment made with no
frames and a judgment made on twenty healthy frames produced IDENTICAL artifacts
apart from a warning line, so the distinction has to be pinned rather than
trusted: `n_frames: 0` with a stated `blind_reason` is a different record from a
frame set that happens to be empty, and both differ again from a run that
recorded nothing because it was not asked to.
"""

from __future__ import annotations

import collections
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from test_parallelism import _entry

from bird import judgments, registry
from bird.artifacts import RunDir
from bird.budget import Budget, BudgetExceeded
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, Trajectory, TrainResult


class _VLM:
    """A client honouring `bird/llm/base.py`, which records what it was sent.

    `messages` is a message LIST by the time it reaches a client, so the prompt
    is flattened the way `test_vlm_sees_frames._SpyVLM` flattens it -- the
    artifact records the prompt STRING the component built, and the two have to
    be compared on the same footing for that claim to mean anything.
    """

    modality = "vlm"

    def __init__(self, reply: str = "Score: 0.5\nReason: it approached.") -> None:
        self.reply = reply
        self.calls = []

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        text = messages if isinstance(messages, str) else "\n".join(
            str(m.get("content", "")) for m in messages)
        self.calls.append({"prompt": text, "n_images": len(images or ())})
        return [self.reply] * n


class _TextOnly(_VLM):
    """Same contract, `modality: text` -- the provider would drop any image."""

    modality = "text"


class _RenderingEnv:
    """Renders a stored state -- the contract `sample_frames` needs.

    The frame is a function of the state so two different states cannot produce
    byte-identical PNGs: the content hash and the frame-set id are only
    meaningful if distinct pixels hash distinctly, and a constant frame would
    let a broken digest pass.
    """

    def __init__(self, width: int = 6) -> None:
        self.width = width

    def render(self, state):
        v = int(abs(float(np.ravel(np.asarray(state, dtype=float))[0]))) % 200
        frame = np.zeros((self.width, self.width, 3), dtype=np.uint8)
        frame[:, :, 0] = v
        frame[0, 0, 1] = 255  # so luma is never exactly the red channel's mean
        return frame


def _traj(n_steps: int = 40, *, states=True) -> Trajectory:
    return Trajectory(
        states=[[float(i)] for i in range(n_steps)] if states else None,
        rewards=[1.0] * n_steps,
        success=False,
        length=n_steps,
        ret=float(n_steps),
    )


def _ctx(tmp_path, *, record="inputs", images=4, **over) -> Context:
    cfg = load("rda", profile="tester", overrides={
        "output.judge_trace.record": record,
        "evaluate.vlm.images_per_query": images,
        "output.video.timeout_s": None,
        "output.video.width": 6,
        "verify.forbidden_symbols": [],
        **over,
    })
    ctx = Context(cfg=cfg, budget=Budget(),
                  rundir=RunDir(tmp_path, "judge", "hash0"),
                  env=_RenderingEnv())
    # ONE client on both roles: `fitness_vlm_score` judges with `ctx.generator`
    # -- the Agent VLM does RDA's in-loop scoring --
    # while `alignment_rate` and the pairwise paths judge with `ctx.evaluator`,
    # and this file's assertions read whichever `.calls` list applies.
    ctx.generator = ctx.evaluator = _VLM()
    return ctx


def _score(ctx, n_rollouts: int = 2, iteration: int = 0):
    cand = Candidate(cand_id="c0", iteration=iteration,
                     reward_code="def compute_reward(state, action=None): return 0.0")
    res = TrainResult(cand_id="c0", candidate=cand,
                      trajectories=[_traj() for _ in range(n_rollouts)])
    state = RunState()
    state.iteration = iteration
    return registry.get("fitness_source", "vlm_score")(ctx, state, [res])


def _recs(ctx, kind: str = "", **where):
    out = [r for r in judgments.read(ctx.rundir) if not kind or r["kind"] == kind]
    for key, value in where.items():
        out = [r for r in out if r.get(key) == value]
    return out


# --------------------------------------------------------------------------
# The default
# --------------------------------------------------------------------------

def test_the_default_writes_nothing_at_all(tmp_path):
    """Every published config inherits `none`, so no published point changes.

    `trajectory_trace`'s rule, for its reason: the key exists to be turned on
    for a run somebody intends to audit. A default that wrote would make every
    sweep pay for an artifact nobody asked for.
    """
    ctx = _ctx(tmp_path, record="none")
    reports = _score(ctx)

    assert reports and reports[0].fitness is not None, "scoring must still work"
    assert not (ctx.rundir.path / "judgments").exists(), (
        "`none` created judgments/; the default must cost nothing")
    assert judgments.mode(ctx) == ""


def test_no_rundir_is_not_an_error(tmp_path):
    """`--dry-run` and every hand-built test Context have no rundir.

    `Context.event` guards the same way for the same reason. A measurement about
    a judgment must never be the thing that ends a search.
    """
    cfg = load("rda", profile="tester", overrides={
        "output.judge_trace.record": "inputs+frames",
        "verify.forbidden_symbols": [],
    })
    ctx = Context(cfg=cfg, budget=Budget(), rundir=None, env=_RenderingEnv())
    ctx.evaluator = _VLM()

    # Neither raises, and neither creates anything: the id still comes back, so
    # a caller can link records it is equally free to have not written.
    judgments.note_query(ctx, iteration=0, role="unit", prompt="hello")
    assert judgments.note_frames(ctx, {"id": "x"}, iteration=0) == "x"
    assert not (tmp_path / "judgments").exists()


# --------------------------------------------------------------------------
# 1 -- the prompt
# --------------------------------------------------------------------------

def test_the_evaluators_prompt_is_written_verbatim(tmp_path):
    """The headline gap. `save_candidate` writes the GENERATOR's prompt; this is
    the evaluator's half."""
    ctx = _ctx(tmp_path)
    _score(ctx, n_rollouts=1)

    queries = _recs(ctx, "query", role="vlm_subtask_score")
    assert queries, "no judge query was recorded"
    sent = [c["prompt"] for c in ctx.evaluator.calls]
    for q in queries:
        assert q["prompt"] in sent, (
            "the recorded prompt is not one the client was actually called with")
    assert len(queries) == len(sent), (
        "every query the judge was asked must appear exactly once")


def test_the_prompt_is_not_truncated(tmp_path):
    """A capped prompt turns a parity record into an approximate one, which is
    the state this artifact exists to leave."""
    long_task = "balance the rod. " * 400
    ctx = _ctx(tmp_path, **{"problem.task_description": long_task})
    _score(ctx, n_rollouts=1)

    q = _recs(ctx, "query")[0]
    assert long_task.strip() in q["prompt"]
    assert len(q["prompt"]) > 6000, len(q["prompt"])


# --------------------------------------------------------------------------
# 2 -- the frames
# --------------------------------------------------------------------------

def test_a_frame_set_records_the_stride_the_frames_came_from(tmp_path):
    """`sample_frames` must KEEP the frame-to-step mapping.

    Which instants a judgment was made from is the one question the frames
    cannot answer once they are bytes, and re-deriving the stride later from a
    `max_frames` that has since moved is wrong by exactly the amount that looks
    like a rollout diverging late -- `save_trajectory_trace`'s argument, applied
    to the judge's own input.
    """
    ctx = _ctx(tmp_path, images=4)
    _score(ctx, n_rollouts=1)

    sets = _recs(ctx, "frames")
    assert len(sets) == 1, sets
    fs = sets[0]
    assert fs["source"] == "render"
    assert fs["n_frames"] == 4
    assert fs["steps"] == [0, 13, 26, 39], fs["steps"]
    assert fs["width"] == 6 and fs["height"] == 6
    assert len(fs["sha256"]) == 4
    assert len(set(fs["sha256"])) == 4, "distinct pixels must hash distinctly"
    assert fs["blind_reason"] == ""


def test_per_frame_luma_is_recorded_and_is_a_measurement(tmp_path):
    """Per frame, not just a mean: a clip that goes black halfway has the same
    mean as one dim throughout, and the difference is the diagnosis
    (`observability.clip_stats`' argument). And no verdict is asserted -- there
    is no `dark: true`, because no absolute threshold generalises across envs."""
    ctx = _ctx(tmp_path, images=4)
    _score(ctx, n_rollouts=1)

    fs = _recs(ctx, "frames")[0]
    assert len(fs["luma"]) == 4
    assert all(isinstance(v, float) for v in fs["luma"])
    assert fs["luma_mean"] == pytest.approx(float(np.mean(fs["luma"])), abs=1e-3)
    assert not any(k in fs for k in ("dark", "healthy", "verdict", "ok"))


def test_queries_point_at_a_frame_set_instead_of_repeating_it(tmp_path):
    """The frames are sampled ONCE per rollout and reused across every subtask
    and repeat (`fitness_vlm_score` says why); the record says the same, or it
    would carry twenty hashes per subtask per repeat."""
    ctx = _ctx(tmp_path, images=3, **{"evaluate.vlm.repeats": 2})
    _score(ctx, n_rollouts=2)

    sets = _recs(ctx, "frames")
    assert len(sets) == 2, "one frame set per rollout, not per query"
    ids = {s["rollout"]: s["id"] for s in sets}
    assert all(ids.values()), ids

    queries = _recs(ctx, "query")
    assert len(queries) > len(sets)
    for q in queries:
        assert q["frame_set"] == ids[q["rollout"]], (
            "a query points at a frame set that is not its rollout's")
        assert "sha256" not in q, "the hashes belong to the frame set, once"
    assert {q["repeat"] for q in queries} == {0, 1}, (
        "each repeat is its own query; a 2-1 vote must not look like 3-0")


def test_the_frame_set_id_is_a_function_of_the_pixels(tmp_path):
    """It is what lets two judgments say they saw byte-identical frames, and
    what lets `inputs+frames` store one set once."""
    a = judgments.frame_set([b"one", b"two"])
    b = judgments.frame_set([b"one", b"two"])
    c = judgments.frame_set([b"one", b"three"])
    assert a["id"] == b["id"] and a["id"] != c["id"]
    assert judgments.frame_set([])["id"] == ""


# --------------------------------------------------------------------------
# 3 -- absent is not empty
# --------------------------------------------------------------------------

def test_a_blind_judgment_records_that_it_was_blind(tmp_path):
    """The whole point. Without it, a judge with its eyes shut and a judge
    shown twenty healthy frames leave the same artifact plus a log line."""
    ctx = _ctx(tmp_path, **{"evaluate.artifacts": ["scalar_metrics"]})
    _score(ctx, n_rollouts=1)

    fs = _recs(ctx, "frames")[0]
    assert fs["n_frames"] == 0
    assert fs["sha256"] == []
    assert "videos" in fs["blind_reason"], fs["blind_reason"]
    assert fs["id"] == ""
    q = _recs(ctx, "query")[0]
    assert q["n_images"] == 0
    assert q["blind_reason"], "the query record does not say it was made blind"


def test_a_blind_reason_names_which_degrade_it_was(tmp_path):
    """Three ways to end up with no pixels, and a run has to be able to tell a
    config that declined footage from a renderer that could not produce it --
    the remedies are opposite."""
    ctx = _ctx(tmp_path, **{"evaluate.artifacts": ["scalar_metrics"]})
    _score(ctx, n_rollouts=1)
    assert "evaluate.artifacts" in _recs(ctx, "frames")[0]["blind_reason"]

    ctx = _ctx(tmp_path / "b")
    # Both roles: the in-loop scorer's frames are gated on the GENERATOR
    # (the judging client).
    ctx.generator = ctx.evaluator = _TextOnly()
    _score(ctx, n_rollouts=1)
    assert "modality" in _recs(ctx, "frames")[0]["blind_reason"]

    ctx = _ctx(tmp_path / "c")
    ctx.env = object()  # no render()
    _score(ctx, n_rollouts=1)
    assert "render" in _recs(ctx, "frames")[0]["blind_reason"]


def test_unmeasured_brightness_is_null_with_a_stated_reason(tmp_path):
    """"Nobody measured this" and "this measured zero" are different facts, and
    only one of them means the frames were dark. Frames that reach the judge
    already encoded (read off a directory somebody else wrote) are the first."""
    prov = judgments.frame_set([b"\x89PNG-ish", b"\x89PNG-too"], None)
    assert prov["luma"] is None and prov["luma_mean"] is None
    assert "luma_absent" in prov and "encoded" in prov["luma_absent"]
    # A set with no frames at all makes no such claim: there was nothing to
    # measure, which is what `blind_reason` already says.
    assert "luma_absent" not in judgments.frame_set([])


# --------------------------------------------------------------------------
# 4 -- the caption, and the pairwise judge
# --------------------------------------------------------------------------

def _pair_ctx(tmp_path, comparator="llm_on_vlm_captions"):
    cfg = load("gt", profile="tester", overrides={
        "output.judge_trace.record": "inputs",
        "evaluate.preferences.comparator": comparator,
        "verify.forbidden_symbols": [],
    })
    ctx = Context(cfg=cfg, budget=Budget(),
                  rundir=RunDir(tmp_path, "pair", "hash0"), env=_RenderingEnv())
    ctx.evaluator = _VLM()
    return ctx


def _pair_reports():
    out = []
    for cid in ("cL", "cR"):
        cand = Candidate(cand_id=cid, iteration=0, reward_code="def f(): ...")
        res = TrainResult(cand_id=cid, candidate=cand, trajectories=[_traj(30)])
        out.append(CandidateReport(cand_id=cid, candidate=cand, result=res, fitness=0.5))
    return out


def test_the_contrastive_caption_is_kept(tmp_path):
    """GT's step 2 sees the caption and the task and NEVER the clips, so the
    caption IS the decider's entire input -- and must not live only in a local
    variable.
    Kept as its own field, not merely embedded in the prompt it is pasted into,
    so a reader can take it without parsing prose back out of a template."""
    ctx = _pair_ctx(tmp_path)
    left, right = _pair_reports()
    registry.get("comparator", "llm_on_vlm_captions")(ctx, RunState(), left, right)

    caps = _recs(ctx, "query", role="pair_caption")
    decs = _recs(ctx, "query", role="pair_decide")
    assert caps and decs
    assert caps[0]["vlm"] is True, "the captioner is the VLM"
    assert decs[0]["vlm"] is False, "the decider must never see the clips"
    caption = decs[0]["caption"]
    assert caption and caption in decs[0]["prompt"], (
        "the recorded caption is not the one pasted into the decider's prompt")

    # ...and kept on BOTH reports, not only in the judgment record. App. D's
    # no-human feedback is an LLM summary of exactly these App. C descriptions;
    # if they die in a local variable, `analyzer: llm` summarises numeric traces
    # instead -- a feedback section that plausibly summarises the wrong channel
    # is a silent failure.
    for rep, side, other in ((left, "LEFT", "cR"), (right, "RIGHT", "cL")):
        entries = rep.meta.get("pair_captions")
        assert entries and len(entries) == 1, (
            f"{rep.cand_id}: one comparison leaves one persisted caption")
        assert entries[0] == {"other": other, "side": side, "caption": caption}


def test_a_pairwise_comparison_points_at_both_clips(tmp_path):
    """A clip's window is recorded ONCE where it is cut, not on every pair that
    reads it: 55 comparisons x 2 sides x 150 step indices would be mostly a
    repeated copy of the same list.

    Each side leaves TWO records and they make different claims. `clip_set` is
    the 15-second window in EPISODE instants, `pixels: false`, no digest --
    what a judge was told about. `frame_set` is the pixels a judge was handed,
    with a content digest and a measured brightness.
    """
    ctx = _pair_ctx(tmp_path, comparator="vlm")
    # Two sides whose CLIPS DIFFER, not the shared `_pair_reports()` pair. The
    # frame-set id is a digest of the pixels, so two agents that behaved
    # identically legitimately share one -- correct, and it would collapse the
    # join this test is checking into a tautology.
    left, right = _pair_reports()
    right.result.trajectories[0] = _traj(31)
    registry.get("comparator", "vlm")(ctx, RunState(), left, right)

    recs = _recs(ctx, "frames")
    assert {c["cand_id"] for c in recs} == {"cL", "cR"}
    clips = [c for c in recs if c["source"] == "clip"]
    shown = [c for c in recs if c["source"] != "clip"]
    assert len(clips) == len(shown) == 2, (
        "each side leaves one clip record and one frame set: %r"
        % [(c["cand_id"], c["source"]) for c in recs])

    for c in clips:
        assert c["pixels"] is False and c["id"] == "" and c["luma"] is None, (
            "the clip record describes what the judge was TOLD; claiming "
            "pixels or a digest there would be a second, false, claim")
        assert c["steps"], "the window is what only this record carries"
    for f in shown:
        assert f["id"] and len(f["sha256"]) == f["n_frames"] > 0, (
            "a frame set with no digest cannot say two judges saw the same "
            "pixels, which is the only thing its id is for")
        assert f["luma_mean"] is not None, "these pixels passed through us"

    q = _recs(ctx, "query", role="pair_vlm")[0]
    assert q["left_id"] == "cL" and q["right_id"] == "cR"
    assert "steps" not in q["left_clip"], "the window belongs to the clip record"
    assert q["left_clip"]["n_frames"] == clips[0]["n_frames"]
    # The join must RESOLVE. A query naming a frame set no record carries is a
    # broken artifact, and it is exactly what happens if the frames are recorded
    # by the phase instead of where they are cut: a comparator called directly
    # -- as here -- never runs the phase.
    by_id = {f["id"]: f for f in shown}
    for side, cand in (("left_clip", "cL"), ("right_clip", "cR")):
        got = by_id.get(q[side]["frame_set"])
        assert got is not None and got["cand_id"] == cand, (
            "%s points at frame set %r, which no record carries"
            % (side, q[side]["frame_set"]))
        assert q[side]["n_images"] == got["n_frames"]
    assert q["n_images"] == sum(f["n_frames"] for f in shown), (
        "the query must count every image it sent, both sides together")


# --------------------------------------------------------------------------
# 5 -- the pixels, and where records land
# --------------------------------------------------------------------------

def test_the_pixels_are_opt_in_and_deduplicated(tmp_path):
    """`inputs` is pointer-and-stats; the bytes are a second, larger ask. A
    metaworld run at 16 cands x 3 rollouts x 20 images x iterations reaches GBs,
    so the set is stored once per content hash however many queries read it."""
    ctx = _ctx(tmp_path, record="inputs", images=3)
    _score(ctx, n_rollouts=1)
    assert not (ctx.rundir.path / "judgments" / "frames").exists(), (
        "`inputs` wrote pixels; only `inputs+frames` may")

    ctx = _ctx(tmp_path / "px", record="inputs+frames", images=3,
               **{"evaluate.vlm.repeats": 2})
    _score(ctx, n_rollouts=2)
    fs = _recs(ctx, "frames")
    root = ctx.rundir.path / "judgments" / "frames"
    assert sorted(p.name for p in root.iterdir()) == sorted({s["id"] for s in fs})
    for s in fs:
        pngs = sorted((root / s["id"]).glob("frame_*.png"))
        assert len(pngs) == s["n_frames"]
        assert [judgments._digest(p.read_bytes()) for p in pngs] == s["sha256"], (
            "the bytes on disk are not the ones the record hashed")
        assert not list((root / s["id"]).glob("*.part")), "a torn write was left behind"


def test_records_are_filed_by_iteration_and_post_phases_are_not(tmp_path):
    """`state/` and `records/` are keyed by iteration because that is what they
    describe. `alignment_rate` grades the FINAL retrained policy and belongs to
    no iteration; filing it under the last one would claim it was part of that
    iteration's search."""
    ctx = _ctx(tmp_path)
    _score(ctx, n_rollouts=1, iteration=3)
    assert (ctx.rundir.path / "judgments" / "iter03.jsonl").is_file()

    judgments.note_query(ctx, iteration=-1, role="alignment_rate", prompt="rate this")
    post = ctx.rundir.path / "judgments" / "post.jsonl"
    assert post.is_file()
    assert json.loads(post.read_text().splitlines()[0])["role"] == "alignment_rate"


def test_a_torn_line_is_skipped_not_raised_on(tmp_path):
    """This file is appended while a run runs, so a reader can catch a partial
    last line -- the tolerance `_read_jsonl` already applies to the journal."""
    ctx = _ctx(tmp_path)
    _score(ctx, n_rollouts=1)
    path = ctx.rundir.path / "judgments" / "iter00.jsonl"
    n = len(judgments.read(ctx.rundir))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"kind": "query", "prompt": "truncated mid-w')

    assert len(judgments.read(ctx.rundir)) == n, "a torn line changed the count"


def test_a_write_failure_never_ends_a_search(tmp_path):
    """Best-effort, like the trace write. Nothing here is on a correctness path:
    a run whose judge trace failed has a gap in an artifact, not a wrong number."""
    ctx = _ctx(tmp_path)
    # A file where the directory has to go: every write below must fail.
    (ctx.rundir.path / "judgments").write_text("not a directory")

    reports = _score(ctx, n_rollouts=1)
    assert reports and reports[0].fitness is not None, (
        "an unwritable judge trace ended the evaluation")


# -- the write lock ---------------------------------------------------------
#
# `_LOCK` is the module's only concurrency claim, and without the tests below,
# bypassing `with _LOCK:` at every site leaves this whole file green. It is not
# hypothetical: with `llm.max_concurrent_requests` and the judge fan-out,
# `judge_concurrency` overlaps exactly the round-trips these records are
# written from.


class _CountingLock:
    """A lock that records how many times it was ENTERED."""

    def __init__(self) -> None:
        self._inner = threading.Lock()
        self.entries = 0

    def __enter__(self):
        self._inner.acquire()
        self.entries += 1          # under the lock: this count is exact
        return self

    def __exit__(self, *exc) -> bool:
        self._inner.release()
        return False


def test_every_record_write_takes_the_lock(tmp_path, monkeypatch):
    """One lock entry per record written, asserted as a fact about the code.

    Counting the LINES cannot show this. A record is one `write()` of one
    complete line to an append-mode file, and on a GIL build that is atomic
    enough that the file comes out intact whether or not the lock is there --
    so a line-count assertion passes with `_LOCK` removed and proves nothing.
    Entry count is deterministic and does not: delete the `with` and this goes
    red on the first record.
    """
    counting = _CountingLock()
    monkeypatch.setattr(judgments, "_LOCK", counting)
    ctx = _ctx(tmp_path)
    n_threads, per = 8, 25

    def worker(t: int) -> None:
        for i in range(per):
            judgments.note_query(ctx, iteration=0, role="unit",
                                 prompt=f"thread {t} query {i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counting.entries == n_threads * per, (
        "a judge-trace record was written without taking _LOCK")


def test_concurrent_records_are_whole_lines(tmp_path):
    """The property the lock buys: no record is torn by another thread's write.

    Weaker than the entry-count test above (it would pass unlocked on this
    build) and kept for the thing it does show -- that the file a concurrent
    judge produces is readable at all, every line parses, and none is lost.
    """
    ctx = _ctx(tmp_path)
    n_threads, per = 8, 25

    def worker(t: int) -> None:
        for i in range(per):
            judgments.note_query(ctx, iteration=0, role="unit",
                                 prompt=f"thread {t} query {i}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    recs = _recs(ctx, "query", role="unit")
    assert len(recs) == n_threads * per, "records were lost under concurrent writes"
    assert len({r["prompt"] for r in recs}) == n_threads * per, "a record was duplicated or torn"


# --------------------------------------------------------------------------
# 6 -- the answer
# --------------------------------------------------------------------------
#
# Recording the input alone leaves the artifact one step short of what it is
# for. `report.json` keeps the aggregate -- `fitness_vlm_score` folds one
# judgment per (repeat, rollout, subtask) to a mean before anything is written
# -- so without a response record the per-item judge answer, the only thing a
# human answer could be scored against, exists in a local variable and nowhere
# else. A human could be shown the judge's exact prompt and exact frames,
# answer perfectly, and there would still be no number to compare with: an
# agreement rate would not be computable under ANY configuration, which makes
# this a blocker for any human baseline rather than a gap in an artifact.

def test_a_query_returns_the_id_its_answer_is_filed_under(tmp_path):
    """The pair, and the join. `note_frames` hands back the frame-set id so a
    query can point at its evidence; `note_query` hands back the query id for
    the same reason and the same shape -- the caller is the only thing holding
    both halves."""
    ctx = _ctx(tmp_path)
    qid = judgments.note_query(ctx, iteration=0, role="unit", prompt="how good?")
    assert qid, "note_query must return an id to file the answer under"
    judgments.note_response(ctx, iteration=0, query_id=qid,
                            raw="Score: 0.75", parsed=0.75)

    q = _recs(ctx, "query", role="unit")[0]
    r = _recs(ctx, "response")[0]
    assert q["query_id"] == r["query_id"] == qid
    assert r["raw"] == "Score: 0.75" and r["parsed"] == 0.75


def test_the_raw_text_and_the_parsed_value_are_both_kept(tmp_path):
    """They differ whenever parsing is lossy, and which one a human is compared
    against is a real question: a human shown `raw` is answering the judge's
    question, while a human scored against `parsed` is being graded on the
    parser as much as on the judge. Keeping one and deriving the other later
    would make that choice for every future reader."""
    ctx = _ctx(tmp_path)
    prose = "Well, on balance I would say roughly 0.6 out of 1, though the grasp slipped."
    qid = judgments.note_query(ctx, iteration=0, role="unit", prompt="p")
    judgments.note_response(ctx, iteration=0, query_id=qid, raw=prose, parsed=0.6)

    r = _recs(ctx, "response")[0]
    assert r["raw"] == prose, "the answer must be kept verbatim, not as its parse"
    assert r["parsed"] == 0.6
    assert r["raw"] != r["parsed"], "a test that cannot see the difference proves nothing"
    assert r["failure_kind"] == "" and r["failure"] == ""


def test_no_answer_and_an_unparseable_answer_are_different_failures(tmp_path):
    """`artifacts`' `failure`/`failure_kind` convention, here for its reason.

    A provider that returned nothing (a refusal, a timeout, a client with no
    such method) and a provider that answered in prose the parser could not
    read are different events with different remedies -- one is an outage, the
    other a prompt or a regex. A bare `parsed: null` conflates them, and would
    also conflate both with a judgment that legitimately parsed to nothing.
    """
    ctx = _ctx(tmp_path)
    for raw, kind in ((None, "no_answer"), ("   ", "no_answer"),
                      ("I would rather not say.", "unparsed"), ("0.5", "")):
        qid = judgments.note_query(ctx, iteration=0, role="unit", prompt=str(raw))
        judgments.note_response(ctx, iteration=0, query_id=qid, raw=raw,
                                parsed=None if kind else 0.5)

    kinds = [r["failure_kind"] for r in _recs(ctx, "response")]
    assert kinds == ["no_answer", "no_answer", "unparsed", ""]
    for r in _recs(ctx, "response"):
        # Absent is not empty, on the answer as on the frames: a failure states
        # itself in prose as well as in a kind, and a success states that it
        # had none rather than omitting the key.
        assert ("failure" in r) and bool(r["failure"]) == bool(r["failure_kind"])


def test_the_query_id_is_derived_so_a_resumed_leg_re_keys_nothing(tmp_path):
    """A counter would give one judge call two ids across a kill and a resume.

    `loop.resume_from` re-pays the iteration that was in flight
    (`budget.resume_discarded_trainings` counts it), so the second leg asks
    questions the first already asked. Under a counter those land as unrelated
    rows and the artifact reads as twice as many decisions as were made. The id
    is a digest of the call's identity, so it resolves identically in any
    process, in any order, on any leg -- and, being derived, it is the same id
    computed by a reader who never saw the run.
    """
    ident = {"role": "vlm_subtask_score", "iteration": 3, "cand_id": "c0007",
             "rollout": 1, "repeat": 0, "subtask": "grasp the handle",
             "prompt": "Score how well the rollout accomplishes THIS subtask."}
    first = judgments.query_id(ident)
    assert first == judgments.query_id(dict(reversed(list(ident.items())))), (
        "the id must not depend on the order the caller built its dict in")

    # Every identity field must MOVE it, or two distinct calls share a key and
    # the join silently merges them.
    for key in ("role", "iteration", "cand_id", "rollout", "repeat", "subtask",
                "prompt"):
        assert judgments.query_id(dict(ident, **{key: "moved"})) != first, key

    # Absent and None are the same claim ("not applicable here"); a VALUE is
    # not, so `-1` -- which `_stem` reads as "outside the loop" -- is its own id.
    bare = {"role": "r", "prompt": "p"}
    assert judgments.query_id(bare) == judgments.query_id(dict(bare, rollout=None))
    assert judgments.query_id(bare) != judgments.query_id(dict(bare, rollout=-1))


def test_the_default_records_no_answers_either(tmp_path):
    """`none` must stay free on the response path too, and `note_query`
    returning `""` is what lets a call site skip its own `mode` check: an
    answer filed against an empty id is a no-op rather than an orphan."""
    ctx = _ctx(tmp_path, record="none")
    assert judgments.note_query(ctx, iteration=0, role="unit", prompt="p") == ""
    judgments.note_response(ctx, iteration=0, query_id="", raw="0.5", parsed=0.5)
    assert not (ctx.rundir.path / "judgments").exists()


# --------------------------------------------------------------------------
# 6b -- the answer, at each of the three call sites
# --------------------------------------------------------------------------

def test_the_per_item_vlm_score_is_kept_before_the_mean(tmp_path):
    """The headline gap. RDA extracts one judgment per (repeat, rollout,
    subtask) -- they arrive as ONE structured call per (repeat, rollout)
    covering the whole subtask list -- and `save_report` writes the MEAN; the
    per-item score, exactly what a human answer would be compared against, is
    otherwise folded away in memory.

    `score_scale` is pinned back to `[0,1]` here so the assertion keeps proving
    the per-item PASS-THROUGH: under the rda default `three_point` a
    0.75 would snap to 0.5 and the test would be checking the quantiser."""
    ctx = _ctx(tmp_path, **{"evaluate.feedback.score_scale": "[0,1]"})
    ctx.generator = ctx.evaluator = _VLM(
        '```json\n{"subtasks": [{"number": 1, "score": 0.75, '
        '"analysis": "it approached."}]}\n```')
    reports = _score(ctx, n_rollouts=2)

    answers = _recs(ctx, "response", role="vlm_subtask_score")
    assert answers, "no judge answer was recorded"
    assert all(a["parsed"] == 0.75 for a in answers), (
        "the per-item score is not the one the judge gave")
    # Filterable on the same labels as the query it answers, so "what did the
    # judge say about this candidate's rollout 1" is not a two-step join.
    assert {a["cand_id"] for a in answers} == {"c0"}
    assert {a["rollout"] for a in answers} == {0, 1}
    assert {a["subtask"] for a in answers}, "the subtask is what was judged"
    # The claim that makes it worth keeping: the report kept one number and the
    # trace kept the judgments behind it. `vlm_judgments` counts verdicts
    # (queries x subtasks); `vlm_queries` counts round-trips.
    assert len(answers) == reports[0].meta["vlm_judgments"] > 1
    assert reports[0].meta["vlm_queries"] == 2, "one call per (repeat, rollout)"


def test_a_fallback_score_is_not_recorded_as_the_judges_answer(tmp_path):
    """`_vlm_trajectory_analysis` never records a fallback number AS the judge's
    answer. Under a REAL provider (this `_VLM` double carries no `provider:
    mock`) that returns no parseable score, it ABSTAINS: no number reaches the
    fitness mean, and the record says `parsed: null, used: null, fallback:
    abstain`. Falling back PER SUBTASK to the ENVIRONMENT's success flag
    (`hi if traj.success else lo`) would read ground truth, which RDA's
    `problem.fitness_access: none` forbids touching, and would be arithmetically
    indistinguishable from a judgment of the same value once the mean is taken. `provider: mock` keeps a deterministic synthetic stand-in
    (`fallback: synthetic_mock`), covered in test_judge_failure_fallbacks.
    """
    ctx = _ctx(tmp_path)
    ctx.generator = ctx.evaluator = _VLM("I decline to rate this rollout.")
    _score(ctx, n_rollouts=1)

    answers = _recs(ctx, "response", role="vlm_subtask_score")
    assert answers
    for a in answers:
        assert a["parsed"] is None, "an unparseable answer must not report a score"
        assert a["failure_kind"] == "unparsed"
        assert a["used"] is None, "a real provider abstains; no fabricated number"
        assert a["fallback"] == "abstain"


def test_used_defaults_to_the_judges_own_answer(tmp_path):
    """The other side of the same claim: on the ordinary path the loop used what
    the judge said, and the record states that rather than leaving it as an
    inference from a missing key."""
    ctx = _ctx(tmp_path, **{"evaluate.feedback.score_scale": "[0,1]"})
    ctx.generator = ctx.evaluator = _VLM(
        '```json\n{"subtasks": [{"number": 1, "score": 0.25}]}\n```')
    _score(ctx, n_rollouts=1)

    answers = _recs(ctx, "response", role="vlm_subtask_score")
    assert answers and all(a["used"] == a["parsed"] == 0.25 for a in answers)
    assert all("fallback" not in a for a in answers)


def test_the_pairwise_verdict_is_kept_per_query_not_per_comparison(tmp_path):
    """A majority over three unparseable answers and a 2-1 split both leave one
    comparison in `report.json`. Only the per-query parsed verdict tells them
    apart, and only it can be set beside a human's."""
    ctx = _pair_ctx(tmp_path, comparator="vlm")
    ctx.evaluator = _VLM("After watching both, LEFT is clearly better.")
    left, right = _pair_reports()
    registry.get("comparator", "vlm")(ctx, RunState(), left, right)

    answers = _recs(ctx, "response", role="pair_vlm")
    assert answers
    for a in answers:
        assert a["parsed"] == 1, "LEFT is verdict 1; the record kept no verdict"
        assert a["raw"] == ctx.evaluator.reply
        assert a["left_id"] == "cL" and a["right_id"] == "cR"


def test_the_captioners_answer_is_the_caption_itself(tmp_path):
    """Step 1 parses nothing -- the caption is not a verdict read out of prose,
    it is the whole product and step 2's entire input. A blank one is still a
    failed step (`_decide_offline` treats it as such), so the trace must not
    file whitespace as a successful answer."""
    ctx = _pair_ctx(tmp_path)
    left, right = _pair_reports()
    registry.get("comparator", "llm_on_vlm_captions")(ctx, RunState(), left, right)

    cap = _recs(ctx, "response", role="pair_caption")[0]
    dec = _recs(ctx, "query", role="pair_decide")[0]
    assert cap["parsed"] == ctx.evaluator.reply.strip()
    assert cap["parsed"] == dec["caption"], (
        "the captioner's recorded answer is not the caption the decider read")

    blank = _pair_ctx(tmp_path / "blank")
    blank.evaluator = _VLM("   \n  ")
    registry.get("comparator", "llm_on_vlm_captions")(blank, RunState(), *_pair_reports())
    got = _recs(blank, "response", role="pair_caption")[0]
    assert got["parsed"] is None and got["failure_kind"] == "no_answer"


def test_a_comparison_the_provider_never_answered_still_records_one(tmp_path):
    """It cost a counted call and it routed the comparator to `_decide_offline`.
    A missing record would read as a query nobody got round to asking, which is
    the one thing the join must not be ambiguous about."""
    ctx = _pair_ctx(tmp_path, comparator="vlm")
    ctx.evaluator = None  # `_client_call` degrades to None with no provider
    left, right = _pair_reports()
    registry.get("comparator", "vlm")(ctx, RunState(), left, right)

    a = _recs(ctx, "response", role="pair_vlm")
    assert a, "an unanswered comparison recorded no response at all"
    assert a[0]["raw"] is None and a[0]["parsed"] is None
    assert a[0]["failure_kind"] == "no_answer"
    assert len(a) == len(_recs(ctx, "query", role="pair_vlm"))


# --------------------------------------------------------------------------
# 7 -- the join, end to end
# --------------------------------------------------------------------------

def _search(name, out, **over):
    """One whole tester-tier search, and its run directory.

    The acceptance criterion is about a RUN, not about three functions called by
    hand: the join can only be total if every judge call in the method reaches
    both halves, and a unit test proves that for the sites it remembered to
    call. This is what catches a forgotten site -- `phases.alignment_rate` is
    the easily missed third.
    """
    over.setdefault("seed", 0)
    over.setdefault("output.judge_trace.record", "inputs")
    over.setdefault("generate.n_candidates", 5)  # gt's keep_top_n floor
    _entry().run(load(name, profile="tester", overrides=over), out_root=str(out))
    return next(p for p in Path(out).iterdir() if p.is_dir())


#: One config per judge CALL SITE, which is not the same as one per method.
#: `rda` reaches `_vlm_trajectory_analysis` in the loop (its trace ROLE is
#: named `_vlm_subtask_score`, for the per-subtask scorer it supersedes) and
#: `alignment_rate` in a post
#: phase -- and a post phase belongs to no iteration, so its records land in
#: `post.jsonl` and a reader that globbed `iter*.jsonl` would call the join
#: total while missing 60 judgments. `gt` reaches the two-step pairwise
#: comparator, whose halves are asked of different modalities.
_JOIN_CONFIGS = [
    ("rda", {"vlm_subtask_score", "alignment_rate"}),
    ("gt", {"pair_caption", "pair_decide"}),
]


@pytest.mark.parametrize("name,roles", _JOIN_CONFIGS, ids=[c for c, _ in _JOIN_CONFIGS])
def test_every_judge_call_in_a_run_has_both_halves(tmp_path, name, roles):
    """THE acceptance criterion: no query without a response, no response
    without a query.

    Counted per `query_id` rather than matched one-to-one, and the module says
    why: the id is DERIVED from the call's identity so that a resumed leg
    re-keys nothing, and the price of that is that two calls nothing
    distinguishes share an id. A set comparison alone would hide a site that
    recorded two queries and one answer whenever the ids collided; the
    per-id floor below cannot.

    Answers may legitimately OUTNUMBER queries per id: one trajectory-analysis call covers the whole subtask list, filed as
    ONE query and one response per subtask against the same id -- so the
    invariant is "every query answered at least once, no orphaned answers",
    not multiset equality.
    """
    records = judgments.read(_search(name, tmp_path))
    queries = collections.Counter(r["query_id"] for r in records if r["kind"] == "query")
    answers = collections.Counter(r["query_id"] for r in records if r["kind"] == "response")

    assert queries, f"{name} recorded no judge queries at all"
    assert set(queries) == set(answers), (
        f"unanswered: {sorted(set(queries) - set(answers))[:3]}; "
        f"orphaned: {sorted(set(answers) - set(queries))[:3]}")
    short = [q for q in queries if answers[q] < queries[q]]
    assert not short, f"ids with more queries than answers: {short[:3]}"
    assert {r["role"] for r in records if r["kind"] == "query"} == roles
    # The join is what a consumer actually wants: one row per decision carrying
    # both what the judge was shown and what it said.
    by_id = {r["query_id"]: r for r in records if r["kind"] == "query"}
    rows = [(by_id[r["query_id"]], r) for r in records if r["kind"] == "response"]
    assert len(rows) == sum(answers.values())
    for q, a in rows:
        assert q["prompt"] and a["role"] == q["role"]


def test_a_post_phase_files_both_halves_in_the_same_file(tmp_path):
    """`_stem` puts a post phase in `post.jsonl` because it belongs to no
    iteration. An answer recorded under the last iteration instead would still
    join -- `read()` concatenates every file -- while claiming the report
    protocol was part of the search, and would break any reader that scoped
    itself to one iteration."""
    rd = _search("rda", tmp_path)
    # `.get`, not `[]`: a frames record carries no role, and this must fail on
    # a misfiled answer rather than on the evidence sitting beside it.
    post = [r for r in judgments.read(rd, "post")
            if r.get("role") == "alignment_rate"]
    kinds = collections.Counter(r["kind"] for r in post)
    assert kinds["query"] and kinds["query"] == kinds["response"]
    assert not [r for r in judgments.read(rd)
                if r.get("role") == "alignment_rate" and r["file"] != "post.jsonl"]


class _ConcurrentVLM(_VLM):
    """A client that DECLARES concurrency, which the tester mock never does.

    This class exists because the obvious version of the test below is a false
    comfort. `judge_concurrency` returns 1 unless the client sets
    `concurrent_samples`, and the mock LLM deliberately never sets it -- its
    per-sample RNG draws consume the run's stream in order, so concurrency
    would reorder them and break every tester-tier determinism claim. A
    tester-tier run with `llm.max_concurrent_requests: 4` is therefore still
    SERIAL, and a test that sets that key and asserts the join holds has
    asserted nothing whatsoever about concurrency.

    Answers are returned slowest-first, so completion order is the REVERSE of
    job order and a positional pairing cannot survive. The separation is tens
    of milliseconds rather than a few, because the machine this runs on may be
    shared and a 4 ms stagger is not reliably an inversion.
    """

    concurrent_samples = True
    max_concurrent_requests = 4

    def __init__(self, reply: str = "Score: 0.5\nReason: it approached.") -> None:
        super().__init__(reply)
        self.order: list = []          # the order calls actually COMPLETED in
        self._issued = 0
        self._lock = threading.Lock()

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        with self._lock:
            i, self._issued = self._issued, self._issued + 1
        time.sleep(max(0, 8 - i) * 0.015)
        with self._lock:  # `_VLM.__call__` appends to a plain list
            out = super().__call__(messages, n=n, temperature=temperature,
                                   images=images, tag=tag)
            self.order.append(i)
        return out


def test_the_join_holds_when_the_judge_answers_out_of_order(tmp_path):
    """`judge_concurrency` overlaps the round-trips, so responses are appended
    in completion order while the queries went out in job order. The join is by
    id and must not care -- an implementation that paired them positionally
    would pass every serial test in this file and mislabel every concurrent run.

    The probe is verified BEFORE the claim, because the way this test fails
    silently is by not being concurrent at all: `judge_concurrency` degrades to
    1 on any client that does not declare itself, and then the assertion below
    passes for the wrong reason -- a false negative of the probe's own making.
    """
    ctx = _ctx(tmp_path)
    ctx.generator = ctx.evaluator = _ConcurrentVLM()
    # Six rollouts, so six independent jobs against four workers: enough that
    # the inversion is unambiguous rather than a two-element coin flip, and
    # enough that the pool has to reuse a worker.
    _score(ctx, n_rollouts=6)

    order = ctx.evaluator.order
    assert len(order) > 1, "no judge calls were made"
    assert order != sorted(order), (
        f"answers arrived in job order ({order}); the pool never engaged, so "
        "this test did not exercise the thing it is named for")

    queries = collections.Counter(r["query_id"] for r in _recs(ctx, "query"))
    answers = collections.Counter(r["query_id"] for r in _recs(ctx, "response"))
    assert queries and queries == answers, (
        f"unanswered: {sorted((queries - answers).elements())[:3]}; "
        f"orphaned: {sorted((answers - queries).elements())[:3]}")
    # Every answer still carries the labels of the query it answers -- the fold
    # that writes them runs on a pool thread, so a shared mutable trace dict
    # would show up here as two responses claiming one rollout.
    got = {(r["cand_id"], r["rollout"], r["subtask"]) for r in _recs(ctx, "response")}
    assert len(got) == len(_recs(ctx, "response")), (
        "two answers claim the same (candidate, rollout, subtask)")


def test_a_query_the_budget_refused_is_still_answered_in_the_artifact(tmp_path):
    """`_query` writes the query record BEFORE it reserves budget, so a
    comparison the cap refuses leaves a query behind on its way out.

    An orphaned query is not a neutral gap: it reads as a call still in flight,
    which is exactly the wrong thing for the artifact to say about the run that
    ended because it ran out of budget. The exception itself is re-raised
    unchanged -- the loop owns it.
    """
    ctx = _pair_ctx(tmp_path, comparator="vlm")
    ctx.budget.max_llm_calls = 0
    left, right = _pair_reports()
    with pytest.raises(BudgetExceeded):
        registry.get("comparator", "vlm")(ctx, RunState(), left, right)

    q = _recs(ctx, "query", role="pair_vlm")
    a = _recs(ctx, "response", role="pair_vlm")
    assert q and len(a) == len(q), "the refused comparison left an orphaned query"
    assert a[0]["failure_kind"] == "no_answer"
    assert "budget" in a[0]["failure"], a[0]["failure"]


def test_an_absent_label_is_absent_rather_than_invented(tmp_path):
    """`repeat` must not be defaulted to `-1` on a pairwise response such as
    `pair_caption`, which has no repeats at all.

    A sentinel is worse than a gap here for the reason `query_id`'s own
    docstring gives in as many words: a field with a VALUE is a positive claim.
    `-1` is `_stem`'s "outside the loop", so an invented `repeat: -1` reads as a
    real index to anything filtering on one, and it would also disagree with
    the QUERY record for the same call, which carries no `repeat` key. Two
    records about one judge call cannot describe it differently.
    """
    ctx = _pair_ctx(tmp_path)
    left, right = _pair_reports()
    registry.get("comparator", "llm_on_vlm_captions")(ctx, RunState(), left, right)

    cap_q = _recs(ctx, "query", role="pair_caption")[0]
    cap_a = _recs(ctx, "response", role="pair_caption")[0]
    assert "repeat" not in cap_q, "the captioner's query gained a repeat index"
    assert "repeat" not in cap_a, (
        f"the captioner has no repeats, but its answer claims repeat="
        f"{cap_a.get('repeat')!r}")

    # The other half: where a repeat genuinely exists it must still be carried,
    # or this test would pass just as well against a version that dropped the
    # label everywhere.
    dec_a = _recs(ctx, "response", role="pair_decide")
    assert dec_a and all(d["repeat"] == 0 for d in dec_a), (
        "the decide step votes per repeat; its answers must say which")


def test_a_scored_answer_carries_the_labels_its_query_carries(tmp_path):
    """The same rule on the in-loop path: copied where the trace has them,
    never defaulted. `_vlm_trajectory_analysis`'s trace always carries all
    three, so the guard is about the day a caller stops passing one -- at which
    point a fabricated `rollout: -1` would be indistinguishable from a
    post-phase."""
    ctx = _ctx(tmp_path)
    _score(ctx, n_rollouts=2)

    for a in _recs(ctx, "response", role="vlm_subtask_score"):
        assert a["rollout"] in (0, 1) and a["repeat"] == 0
        assert a["cand_id"] == "c0"
