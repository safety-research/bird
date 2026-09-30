"""What the (V)LM judge was SHOWN and what it ANSWERED.

`candidates/<id>/report.json` keeps the judge's verdict -- the ranking scalar,
RDA's per-subtask analysis, the prose the next prompt is built from -- but not
its INPUT. Without this module three inputs are lost, one shape:

  * **The frames.** `evaluation._vlm_frames` calls `observability.sample_frames`
    per scored rollout, and because `record_rollouts` runs AFTER stage 4 there is
    nothing on disk at scoring time: the pixels are rendered fresh, sent as
    bytes and dropped. Only counts would survive -- `budget.vlm_calls` (there is no
    `budget.vlm_queries`; the per-report `report.meta["vlm_queries"]` is
    repeats x rollouts), `report.meta["vlm_images_per_rollout"]`, and a warning
    line when a rollout was scored blind.
  * **The prompt.** The GENERATOR's prompt lands in `candidates/*/prompt.json`.
    The EVALUATOR's is written nowhere else.
  * **The caption.** Gran Turismo's two-step comparator
    (`preferences.comparator_llm_on_vlm_captions`) has a VLM caption both clips
    contrastively and a text model decide from the caption ALONE. That caption is
    the decider's entire input, and it would otherwise live only in a local
    variable.

The consequence is not abstract: without this record a judge that scored with
its eyes shut and a judge shown twenty healthy frames produce identical
artifacts apart from a log line, and any comparison against a human judge --
meaningful only if the human judges under exactly the judge's conditions --
could only APPROXIMATE them, re-subsampling `videos/<cand>/` (rollout 0 only, capped by
`output.video.max_frames`) and showing `report.json`'s `feedback` where a caption
belongs, i.e. the judge's OUTPUT in the slot reserved for its input.

**Measurements, never verdicts.** Per-frame brightness is recorded; "this clip
was black" is not asserted. The reasoning is `observability.clip_stats`'s and
`budget.blind_comparisons`': no absolute threshold generalises across
environments -- a reacher clip is legitimately darker than a half-cheetah one,
and near-black compressed video still yields ~1,480 distinct colours -- so the
run states the numbers and the reader draws the conclusion, comparing a run's
own same-environment siblings.

**Pointer and stats, not payload.** `records/`' rule, for `records/`' reason. A
frame set is recorded as the episode step indices it was drawn from, its size,
and a per-frame luma and content hash -- a few floats a frame. The exact PNG
bytes are `output.judge_trace.record: inputs+frames` only, off by default: a
metaworld run at 16 candidates x 3 rollouts x 20 images a query x iterations
reaches GBs, and the step indices plus the hashes already make those bytes
reproducible from `videos/` wherever the strides overlap.

**Absent is not empty.** A judgment made blind records THAT IT WAS BLIND and why
-- `n_frames: 0` with a `blind_reason` -- rather than an empty frame list, which
is the same distinction `_vlm_frames`'s `note` draws in memory and the one
`save_report` draws between a missing key and `""`.

**What it was shown AND what it said.** `note_query` and `note_response` are a
pair joined on a `query_id`, and the pair is the point. Recording the input
alone would leave the artifact one step short of the thing it is built for: a
human can be shown the judge's exact prompt and exact frames, answer perfectly,
and there is still nothing to compare the answer against. `report.json` keeps
only the aggregate -- `fitness_vlm_score` issues one judgment per
`(repeat, rollout, subtask)` and folds them to a mean before the report is
written -- so the per-item judge answer, which is exactly what a human answer
would be scored against, would exist only in a local variable, and an agreement
rate would not be computable under any configuration.

The response record keeps BOTH the raw text and the parsed value, because they
differ whenever parsing is lossy and which of them a human is being compared
against is a real question, not a formatting detail. Where the loop used
neither -- `_vlm_trajectory_analysis` falls back to the environment's own
success flag on an unparseable answer -- the record says so in `used` and
`failure_kind`, so
a rate computed over judge answers cannot silently include numbers the judge
never gave.

**The `query_id` is DERIVED, never a counter.** A digest over the fields that
identify a call (`_QUERY_KEY`), so it resolves identically on a resumed leg --
the same argument `loop.resume_from`'s sentinel makes for the run directory. The
cost of that choice is that two genuinely indistinguishable calls share an id;
the join is therefore defined on COUNTS per id (n queries == n responses), not
on a unique key, and the module claims no more than that.

**Absent is not empty**, on the answer as much as on the frames. A provider that
returned nothing and a provider whose answer would not parse are different
failures with different remedies, so they are `failure_kind: no_answer` and
`failure_kind: unparsed` -- `artifacts`' `failure`/`failure_kind` convention,
here for its reason -- rather than a missing key or a null that could mean
either.

**Best-effort, always.** Every entry point swallows its own exceptions, for the
reason `observability.save_trajectory_trace`'s call site does: a lost record must
never end a search. Nothing in this module is on a correctness path, and a run
whose judge trace failed to write is a run with a gap in an artifact, not a run
with a wrong number.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger("bird.judgments")

#: One writer at a time. A record is one `write()` of one complete line to a
#: file opened for append -- `RunDir.event`'s convention for `journal.jsonl`,
#: which is what a reader of this tree already expects -- and that is atomic
#: enough between processes only because there is one process per run dir. It is
#: NOT automatically safe between THREADS, and `llm.max_concurrent_requests`
#: exists to overlap exactly the judge round-trips this module sits inside, so
#: the lock is held across open-write-close rather than assumed away.
_LOCK = threading.Lock()

#: Truncation of the per-frame and per-set digests. 64 bits is far past what
#: distinguishing frames within one run needs, and the point of the hash is
#: identity, not integrity: it is what lets two queries that were shown
#: byte-identical pixels say so, and what lets `inputs+frames` deduplicate a
#: frame set reused across `evaluate.vlm.repeats` and every subtask.
_DIGEST_CHARS = 16

#: The fields a judge call is IDENTIFIED by, in a fixed order, digested into
#: `query_id`. Fixed rather than "whatever the caller passed" because the id has
#: to be stable across a resume, and a set derived from the fields present would
#: change the day a call site started passing one more of them -- re-keying every
#: record written after the change and silently breaking the join for the
#: iteration in flight.
#:
#: `cand_id` covers the per-candidate sites; `left_id`/`right_id` cover the
#: pairwise ones, which have no single candidate. `prompt` is last and is what
#: separates two calls that agree on every label -- which is most of what makes
#: this unique in practice, since a call site that varies nothing else varies
#: the prompt.
_QUERY_KEY = ("role", "iteration", "cand_id", "rollout", "repeat", "subtask",
              "left_id", "right_id", "prompt")

#: `failure_kind` for a response, mirroring `artifacts`' convention for a
#: candidate: `""` when the judge answered and the answer parsed. The two
#: non-empty values are not interchangeable -- `no_answer` is a provider that
#: returned nothing (a refusal, a timeout, a client that does not implement the
#: call), `unparsed` is a provider that answered in prose the parser could not
#: turn into a value -- and their remedies differ, so a reader must not have to
#: infer which one a null `parsed` meant.
_NO_ANSWER = "no_answer"
_UNPARSED = "unparsed"

#: Rec. 601 luma weights: 0 is black, 255 is white. The per-frame series is kept
#: whole because a clip that goes black halfway has the same mean as one dim
#: throughout, and the difference is the diagnosis.
#:
#: THE definition, and public for that reason: `observability.clip_stats`
#: measures the same quantity at recording time and imports this rather than
#: restating it. Two copies of three floats is record drift -- the numbers
#: would agree until one of them was
#: "corrected", and the artifact that disagreed would be the citable one. The
#: direction is the one the package already has (`observability` imports
#: `frame_set` from here), so nothing new is coupled by it.
LUMA_601 = (0.299, 0.587, 0.114)


def mode(ctx: Any) -> str:
    """`""` when this run records no judge inputs, else the configured mode.

    Read through `ctx.cfg` on every call rather than cached: this module is
    imported by three component modules and a test builds a `Context` by hand,
    so there is no run-scoped place to cache it that a test would not have to
    reset. The lookup is a dict get.
    """
    try:
        value = str(ctx.cfg.get("output.judge_trace.record", "none") or "none")
    except Exception:  # noqa: BLE001 - a Context without a config records nothing
        return ""
    return "" if value == "none" else value


def _root(ctx: Any) -> Optional[Path]:
    """`<rundir>/judgments/`, or None when there is nowhere to write.

    `ctx.rundir` is None for `--dry-run` and throughout the test suite's
    hand-built contexts, and that is not an error: it is the same guard
    `Context.event` applies before touching the journal.
    """
    rundir = getattr(ctx, "rundir", None)
    path = getattr(rundir, "path", None)
    if path is None:
        return None
    return Path(path) / "judgments"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:_DIGEST_CHARS]


def _luma(frame: Any) -> Optional[float]:
    """One frame's Rec. 601 mean brightness, or None if it will not coerce.

    Channel means first and the luma weights after: the mean is linear, so this
    is the same number as weighting every pixel, without materialising an
    H x W x 3 float64 copy of the frame in the judge's hot path -- the shape
    `clip_stats` settled on for the same reason.
    """
    try:
        arr = np.asarray(frame)
        if arr.ndim == 2:  # grayscale: every channel is the same plane
            return round(float(arr.mean(dtype=np.float64)), 3)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return None
        r, g, b = (arr[..., i].mean(dtype=np.float64) for i in range(3))
        return round(float(r * LUMA_601[0] + g * LUMA_601[1]
                          + b * LUMA_601[2]), 3)
    except Exception:  # noqa: BLE001 - an unmeasurable frame is null, not a raise
        return None


def frame_set(png: Sequence[bytes],
              rgb: Optional[Sequence[Any]] = None,
              *,
              steps: Optional[Sequence[int]] = None,
              source: str = "",
              blind_reason: str = "",
              views: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The provenance of one judge-facing frame set. Never raises.

    `png` are the bytes the client was actually handed, so the content hashes are
    over what the provider received and not over an intermediate. `rgb` are the
    same frames as arrays where this process ever held them -- the render path
    and the decode-a-clip path do, and the read-PNGs-off-disk path does not.

    `steps` is the EPISODE index each frame came from, and it is recorded rather
    than recomputed for `save_trajectory_trace`'s reason: frames are strided
    across an episode, not truncated, so frame 15 of 20 is step 150 of a
    200-step episode, and a consumer that re-derives the stride from a
    `max_frames` that has since changed is wrong by exactly the amount that
    looks like a reward diverging late. `None` means the stride is genuinely
    unknown to this process (frames read from a directory somebody else wrote),
    which is a different claim from `[]`.

    `id` is the set's identity: a digest over the per-frame digests, so it is a
    pure function of the pixels. Two queries shown byte-identical frames carry
    the same id and say so, and `inputs+frames` writes those bytes once.
    """
    try:
        shas = [_digest(bytes(b)) for b in (png or ())]
    except Exception:  # noqa: BLE001
        shas = []
    prov: Dict[str, Any] = {
        "id": _digest("".join(shas).encode()) if shas else "",
        "source": source,
        "n_frames": len(shas),
        "sha256": shas,
        # `""` when the judge had eyes. The non-empty case is the whole reason
        # this key exists: a blind judgment must be legible as blind, not as a
        # frame set that happens to be empty.
        "blind_reason": blind_reason,
    }
    prov["steps"] = [int(s) for s in steps] if steps is not None else None
    # ONLY when the frame carries more than one viewpoint (`output.video.n_views`):
    # a single-view record is byte-identical to what it was before the key existed,
    # and an absent key means one view, which is the reading every older record
    # already has. The layout is what says which pixel columns are which camera,
    # so a reader that re-derives a panel boundary from `width / n` is wrong by
    # the gutter -- read `panels[].x0/x1`.
    if views and int(views.get("n_views") or 0) > 1:
        try:
            prov["views"] = json.loads(json.dumps(views))
        except (TypeError, ValueError):
            prov["views"] = {"n_views": int(views.get("n_views") or 0)}

    frames = list(rgb or ())
    if frames:
        lumas = [_luma(f) for f in frames]
        prov["luma"] = lumas
        measured = [v for v in lumas if v is not None]
        prov["luma_mean"] = round(float(np.mean(measured)), 3) if measured else None
        try:
            shape = np.asarray(frames[0]).shape
            prov["height"], prov["width"] = int(shape[0]), int(shape[1])
        except Exception:  # noqa: BLE001
            pass
    else:
        # Null with a stated reason, never a silently missing key: "nobody
        # measured this" and "this was measured at zero" are different facts,
        # and only one of them means the frames were dark.
        prov["luma"] = None
        prov["luma_mean"] = None
        if shas:
            prov["luma_absent"] = ("the frames reached this process already "
                                   "encoded, so no pixels passed through it")
    return prov


def clip_set(steps: Optional[Sequence[int]], *, n_frames: int,
             video_path: Optional[str] = None,
             blind_reason: str = "") -> Dict[str, Any]:
    """The provenance of a clip a judge was shown IN WORDS rather than in pixels.

    `frame_set`'s sibling, and deliberately not the same function. This one is
    for the case where a judge was told about a clip and shown none of it, so
    there are no bytes to hash and no brightness to measure. `pixels: false`
    states that as a fact about the judge's CHANNEL, which is a different claim
    from `frame_set`'s `luma: null`: there, pixels existed and this process did
    not hold them; here they were never part of the judgment at all.

    A frames-attaching comparator writes a real `frame_set` beside this one,
    and the two are both true and answer different questions. This record says which EPISODE
    INSTANTS the 15-second window covers; the `frame_set` says which pixels of
    that window the judge was handed, with a digest that makes "these two
    queries saw byte-identical frames" checkable. `comparator_human` still
    produces only this one, because its judge reads a file path.

    What survives, and what nothing kept, is `steps`: which EPISODE instants the
    clip covers, per `evaluate.preferences.clip_length_s`/`fps`/`max_video_s`.
    After `_clip_for` the clip is a fresh Trajectory whose own indices run 0..n
    and say nothing about the episode behind it, so re-deriving the window later
    means re-reading three config keys and hoping they have not moved.
    """
    return {
        "id": "",  # no pixels, so no content digest; (cand_id, iteration) is the key
        "source": "clip",
        "pixels": False,
        "n_frames": int(n_frames),
        "steps": [int(i) for i in steps] if steps is not None else None,
        "video_path": video_path,
        "sha256": [],
        "luma": None,
        "luma_mean": None,
        "blind_reason": blind_reason,
    }


def note_frames(ctx: Any, prov: Optional[Dict[str, Any]], *,
                iteration: int, cand_id: str = "", rollout: int = -1,
                png: Optional[Sequence[bytes]] = None) -> str:
    """Record one frame set and return its id (`""` when nothing was recorded).

    The id is returned so the query records that follow can point at this one
    instead of repeating twenty hashes per subtask per repeat: the frames are
    sampled ONCE per rollout and reused across every question asked of it
    (`fitness_vlm_score` says why), and the record should say the same.

    Under `inputs+frames` the exact bytes land in `frames/<id>/`, written once
    per id: a set reused across subtasks and repeats is stored once, which is
    the deduplication the content hash was for.
    """
    if not prov:
        return ""
    m = mode(ctx)
    if not m:
        return ""
    set_id = str(prov.get("id") or "")
    if m == "inputs+frames" and set_id and png:
        _write_pixels(ctx, set_id, png)
    # The structural keys go LAST so they cannot be shadowed. `prov` is built by
    # `frame_set`/`clip_set` and carries none of them today, but a record whose
    # `kind` came from measured data would be unfilterable, and the cost of
    # ruling that out is the ordering of three lines.
    _append(ctx, iteration, {
        **prov,
        "kind": "frames",
        "cand_id": cand_id,
        "rollout": rollout,
    })
    return set_id


def query_id(ident: Dict[str, Any]) -> str:
    """The id a query and its response are joined on. A pure function of `ident`.

    Derived, never a counter, and that is the whole of the design: a resumed leg
    re-asks the iteration that was in flight when it died (`budget.
    resume_discarded_trainings` counts it), and a counter would give the same
    judge call a different id on the second leg, leaving the artifact with two
    unrelated-looking rows for one decision. A digest over `_QUERY_KEY` resolves
    identically in any process, in any order, on any leg.

    The price is stated rather than engineered away: two calls that agree on
    every field in `_QUERY_KEY` -- same role, same candidate, same rollout, same
    repeat, same subtask, byte-identical prompt -- share an id, because by
    construction nothing distinguishes them. The join is therefore over COUNTS
    per id, not a lookup, and `read()`'s consumers are told so.

    A field that is absent and a field that is `None` both normalise to `""`,
    so a call site is free to pass `rollout=None` for "not applicable" without
    that reading as a distinct call. A field with a VALUE is stringified, so
    `rollout=-1` and `rollout` absent are deliberately different ids: `-1` is a
    positive claim (`_stem`'s "outside the loop"), not a way of saying nothing.
    """
    payload = json.dumps(
        ["" if ident.get(k) is None else str(ident.get(k, "")) for k in _QUERY_KEY],
        ensure_ascii=False, separators=(",", ":"))
    return _digest(payload.encode("utf-8"))


def note_query(ctx: Any, *, iteration: int, role: str, prompt: str,
               **fields: Any) -> str:
    """Record one judge call: what was asked, of what evidence, verbatim.

    Returns the `query_id` the answer must be filed under -- `note_frames`'
    convention, for `note_frames`' reason: the caller is the only thing that
    holds both halves, and handing it the key is what lets the answer be written
    where it is known rather than plumbed back into this module. `""` when this
    run records nothing, so a call site needs no `mode` check of its own and
    `note_response` on that id is a no-op.

    The prompt is NOT truncated. It is the artifact's only claim -- that this is
    what the judge was shown -- and a cap would quietly make a parity record
    into an approximate one, which is the state this module exists to leave.
    Every prompt here is built from a template bounded by the config
    (`evaluate.vlm.images_per_query` bounds the state lines,
    `update.prompt.max_length_tokens` the prose), so there is no unbounded case
    to defend against.

    `role` names the CALL SITE, not the method: `vlm_subtask_score`,
    `alignment_rate`, `pair_caption`, `pair_decide`, `pair_vlm`. A reader
    filtering this file is asking which judge spoke, and a method name here
    would be the branch this repo does not have.
    """
    # BEFORE the digest, not after. `query_id` hashes the whole prompt, and on
    # the default (`none`) the answer is `""` however the hash comes out -- so
    # computing it first would put a sha256 over every judge prompt on the path
    # of every run that asked for no trace at all. `none` costs nothing is a
    # property this module states twice; it has to survive the id.
    if not mode(ctx):
        return ""
    # `**fields` first, for `note_frames`' reason: `kind`, `role` and `prompt`
    # are what a reader filters and reads on, and a caller's field must not be
    # able to take their names. The identity dict follows the same order, so a
    # caller cannot move an id by passing `role=` twice either.
    qid = query_id({**fields, "role": role, "iteration": iteration,
                    "prompt": prompt})
    _append(ctx, iteration, {**fields, "kind": "query", "role": role,
                             "query_id": qid, "prompt": prompt})
    return qid


def note_response(ctx: Any, *, iteration: int, query_id: str, raw: Any,
                  parsed: Any, **fields: Any) -> None:
    """Record what the judge ANSWERED, keyed to the query that asked it.

    Both halves, always. `raw` is the provider's text verbatim -- untruncated,
    for the prompt's reason -- and `parsed` is the value the loop read out of
    it. They differ whenever parsing is lossy, and an agreement rate computed
    against the wrong one of them is a different measurement wearing the same
    name: a human shown `raw` is answering the judge's question, while a human
    compared against `parsed` is being scored on the parser as much as on the
    judge.

    Where the loop used NEITHER -- `evaluation._vlm_trajectory_analysis` falls
    back to the environment's own success flag when no score parses -- the caller
    passes
    `used=`, and the pair (`parsed: null`, `used: 0.0`) is the artifact saying
    that the number in `report.json` did not come from the judge at all. Without
    it a fallback is indistinguishable from a judgment, which is precisely the
    class of silent substitution the frame records exist to stop.

    `failure_kind` is DERIVED here rather than at each call site, so the three
    sites cannot drift about what "the judge did not answer" means. `raw` that
    is None or blank is `no_answer`; text that yielded no value is `unparsed`;
    anything else is `""`. A caller that knows better -- a refusal it can name,
    a budget exception -- passes `failure=` for the prose and the kind still
    describes the shape.
    """
    if not mode(ctx):
        return
    text = "" if raw is None else str(raw)
    if not text.strip():
        kind = _NO_ANSWER
    elif parsed is None:
        kind = _UNPARSED
    else:
        kind = ""
    record: Dict[str, Any] = {
        # `used` defaults to `parsed`: at every site but one they are the same
        # value, and a key that appeared only on the fallback path would make
        # "the loop used the judge's answer" an inference from an absence.
        "used": parsed,
        **fields,
        "kind": "response",
        "query_id": str(query_id or ""),
        "raw": None if raw is None else text,
        "parsed": parsed,
        "failure_kind": kind,
    }
    record.setdefault("failure", "")
    if kind and not record["failure"]:
        record["failure"] = ("the provider returned no text"
                             if kind == _NO_ANSWER else
                             "the answer carried no value the parser could read")
    _append(ctx, iteration, record)


# --------------------------------------------------------------------------
# the writers
# --------------------------------------------------------------------------

def _stem(iteration: int) -> str:
    """`iterNN` in the loop, `post` outside it.

    `state/` and `records/` are keyed by iteration because that is what they
    describe. A `post:` phase judgment -- RDA's `alignment_rate` grades the
    FINAL retrained policy -- belongs to no iteration, and filing it under the
    last one would claim it was part of that iteration's search. `iteration < 0`
    is the caller saying so.
    """
    return f"iter{int(iteration):02d}" if int(iteration) >= 0 else "post"


def _append(ctx: Any, iteration: int, record: Dict[str, Any]) -> None:
    """One complete line, appended. Swallows everything."""
    root = _root(ctx)
    if root is None:
        return
    try:
        line = json.dumps(record, default=str) + "\n"
    except Exception:  # noqa: BLE001 - an unserialisable record is dropped, not raised
        log.debug("could not serialise a judge-trace record", exc_info=True)
        return
    try:
        with _LOCK:
            root.mkdir(parents=True, exist_ok=True)
            with open(root / f"{_stem(iteration)}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception:  # noqa: BLE001 - a lost record is not a lost run
        log.debug("could not append to the judge trace", exc_info=True)


def note_composed(ctx: Any, set_id: str, sheet: bytes) -> None:
    """Store the COMPOSED image a policy actually sent, beside its source frames.

    Only under `inputs+frames`, and only for a policy that composes. The frames
    `note_source_frames` stores under `frames/<set_id>/` -- `observability.
    _deliver` calls it immediately before this -- are what the sheet was built
    FROM; they are not what crossed the wire. Under `evaluate.vlm.frame_policy:
    contact_sheet` the judge saw one tiled image at reduced per-cell resolution,
    and every question worth asking of the record afterwards -- was a marker
    legible at this cell size, did the judge misread a cell, would a human agree
    looking at the same thing -- is a question about the sheet and not about its
    inputs. Storing the sources and discarding the payload would leave
    `inputs+frames` claiming to hold the judge's input while holding a
    reconstruction of it.

    `sheet.png`, not `frame_NNNN.png`, so a reader listing the frame set cannot
    mistake the composite for one of its own cells.
    """
    if mode(ctx) != "inputs+frames" or not set_id or not sheet:
        return
    root = _root(ctx)
    if root is None:
        return
    dest = root / "frames" / set_id
    try:
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / "sheet.png"
        if out.exists():
            return                      # the id is a content hash; same id, same sheet
        tmp = dest / ".sheet.png.tmp"
        tmp.write_bytes(sheet)
        os.replace(tmp, out)
    except Exception:  # noqa: BLE001 - a lost record is not a lost judgment
        log.debug("could not store a composed frame set", exc_info=True)


def note_source_frames(ctx: Any, set_id: str, png: Sequence[bytes]) -> None:
    """Store a frame set's SOURCE frames under `inputs+frames`.

    Called by `observability._deliver` for a composing policy, BEFORE
    `note_composed`, with the frames the sheet was built from -- the ones
    `frame_set` hashed into the record's `sha256` and `n_frames`. The callers'
    own `note_frames(..., png=<delivered>)` runs later and is handed the
    DELIVERED list, which under a composing policy is the sheet, so without
    this call nothing would store the sources: `judgments/frames/<id>/` would
    hold `sheet.png` alone while the record beside it described four frames.
    Under `even` the policy delivers the sources themselves and `note_frames`
    stores them.
    """
    if mode(ctx) != "inputs+frames" or not set_id or not png:
        return
    _write_pixels(ctx, set_id, png)


def _write_pixels(ctx: Any, set_id: str, png: Sequence[bytes]) -> None:
    """`frames/<set_id>/frame_NNNN.png`, once per set. Swallows everything.

    tmp + `os.replace` per frame, `_write_status`'s convention: a reader may
    read these trees live, and one that listed a half-written PNG could only
    report it as corrupt rather than as absent. The tmp name does not match
    `frame_*.png`, so a crash mid-write leaves nothing a reader would list.

    A set whose frames are already present is left alone rather than
    rewritten: the id IS the content hash, so a second set with the same id
    holds the same bytes, and re-encoding them would be work with no possible
    different outcome. Presence is judged per FILE (`frame_0000.png`), not per
    directory: `note_composed` creates the directory for `sheet.png`, and a
    directory test would read that as "the frames are here".
    """
    root = _root(ctx)
    if root is None:
        return
    dest = root / "frames" / set_id
    try:
        if (dest / "frame_0000.png").exists():
            return
        with _LOCK:
            dest.mkdir(parents=True, exist_ok=True)
            for i, blob in enumerate(png):
                tmp = dest / f"frame_{i:04d}.png.part"
                tmp.write_bytes(bytes(blob))
                os.replace(tmp, dest / f"frame_{i:04d}.png")
    except Exception:  # noqa: BLE001 - lost pixels are not a lost run
        log.debug("could not write judge-trace pixels for %s", set_id, exc_info=True)


def read(rundir: Any, stem: str = "") -> List[Dict[str, Any]]:
    """Every record in a run's judge trace. THE format's reader, not a helper.

    Here rather than in each consumer for the reason `bird/tasks.py` is the only
    reader of `tasks/`: a second implementation of "how do you parse this" drifts
    from the first, and the drift shows up as a consumer quietly seeing fewer
    records than the run wrote. Any external reader should follow these rules
    exactly.

    Three kinds come back: `frames` (evidence), `query` (what was asked of it)
    and `response` (what came back). A query points at its evidence by
    `frame_set`, and a response at its query by `query_id` -- **grouped, not
    looked up**. `query_id` is a digest of the call's identity (`query_id()`),
    so two indistinguishable calls share one, and a resumed leg re-asking the
    iteration it died in writes the same id a second time. The invariant a
    reader may rely on is that within a completed run each id carries as many
    responses as queries; a shortfall is a call that did not return, which is a
    finding rather than a parse error.

    Malformed lines are SKIPPED, not raised on: the file is appended while a run
    runs, so a reader can catch a torn last line -- the same tolerance a reader
    of `journal.jsonl` needs.
    """
    root = Path(getattr(rundir, "path", rundir)) / "judgments"
    if not root.is_dir():
        return []
    names = [f"{stem}.jsonl"] if stem else sorted(
        p.name for p in root.glob("*.jsonl"))
    out: List[Dict[str, Any]] = []
    for name in names:
        path = root / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(dict(rec, file=name))
    return out
