"""The preference dataset: one line per comparison, written as it is made.

`RunState.preferences` holds every pairwise comparison a run makes, and
without this module nothing but a COUNT would reach disk -- `RunState.summary()`'s
`n_preferences` -- and the records would die with the process. That is worse
than a wrong format: a VLM-vs-human agreement rate is exactly a join of
`source: vlm` against `source: human` over the same pairs, which is a query
against a dataset that exists and a research project against one that does not.

WHY A LOG AND NOT A SNAPSHOT. `records/iterNN.json` also carries preferences,
and it is not this: it is the state AFTER `apply_carry`, rewritten whole every
iteration. Three things follow from being a log
instead:

  * a run killed mid-iteration keeps every comparison it had already made,
    because each line is complete when it is written;
  * a comparison appears ONCE, at the moment it was made, rather than again in
    every later snapshot that still carries it;
  * the file is the unit of pooling. Records carry their own run id, so
    concatenating two runs' logs is a valid dataset with no rewriting.

NOT GATED BY A CONFIG KEY, deliberately, and this is the one place this module
departs from `judgments.py`'s shape. The judge trace is off by default because
its `inputs+frames` mode reaches gigabytes; a preference is one short line and a
whole `gt` run makes a few hundred. A dataset you have to remember to
switch on is a dataset you will not have when you want to analyse last week's
sweep, and the cost of always writing it is a few tens of KB.

WHAT A RECORD CARRIES, and each field is here because leaving it out breaks a
question someone will ask:

  * `run`/`iteration`/`restart` -- ids are candidate ids, which mean nothing
    once two runs' records sit in one file. This is what makes pooling safe.
  * `source` -- `vlm`, `human`, `selection`. GT stores an EXTRA preference
    per loser with a Bradley-Terry-derived label that
    can contradict the head-to-head verdict, and `screen_tac` consumes them
    without filtering. Any pooled analysis that ignores `source` mixes a model's
    judgement, a person's and an inferred one; the field is only useful if it is
    always written, so it always is.
  * `preferred` -- `"left"`, `"right"`, `"tie"`, or `null`. `null` is an
    ABSTENTION: asked, and no judgement made. It is not `"tie"`, which is a
    judgement that the two are equal, and it is not a side, because coercing an
    abstention manufactures data. No in-run producer emits `null` today; a human
    UI offering "can't tell" is why the field is shaped for it now rather than
    after there is a dataset whose schema is already fixed.
  * `left`/`right` -- POINTERS and summaries, never the episodes. A
    `Trajectory` carries states and actions and the clips are already on disk
    under `videos/`; `save_train_result`'s size argument holds unchanged here.
  * `left_span`/`right_span` -- the half-open step range each side was judged
    over, or `null` for the whole rollout. R* labels SEGMENTS (`Preference.
    left_span`), and a segment label pooled as a whole-rollout label is a
    different claim about the same pair; `null` and `[0, length]` stay
    distinct for the reason `types.Preference` gives.
  * `annotator`/`session` -- null for a machine judgement. Pooling human
    judgements without being able to separate the people who made them makes
    inter-annotator agreement uncomputable, and agreement is the only way to
    tell a preference signal from noise.
  * `task` -- the same pair judged under a different task description is a
    different judgement, and the description is a config value that moves.

BEST-EFFORT, ALWAYS. Every entry point swallows its own exceptions, for
`judgments.py`'s reason and `observability.save_trajectory_trace`'s: a lost
record is a gap in a dataset, never a failed search. Nothing here is on a
correctness path.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

log = logging.getLogger("bird.preference_log")

#: One writer at a time. A record is one `write()` of one complete line to a
#: file opened for append -- `RunDir.event`'s convention -- which is atomic
#: enough between processes because there is one process per run dir, and is
#: NOT automatically safe between threads. `evaluate.preferences` runs under
#: `judge_concurrency`, so the lock is held rather than assumed away.
_LOCK = threading.Lock()

#: Bump when a field's MEANING changes, never for an addition. A reader that
#: understands 1 must keep working against a file that grew a column, which is
#: the whole reason the version is in every line rather than in a header.
SCHEMA = 1

FILENAME = "preferences.jsonl"

#: `Preference.label` as the search uses it, mapped to the stored word. The
#: integer is an implementation detail of `D_pref` and Bradley-Terry; the word
#: is what survives into a dataset someone reads in a year.
_LABEL_TO_SIDE = {1: "left", 0: "right", -1: "tie"}

#: Fields this module COMPUTES, which a caller's `**extra` must not be able to
#: take. The distinction is not "every field we write": `annotator` and
#: `session` are declared here with a null default precisely so a producer that
#: knows them can fill them in, and protecting those would silently drop the
#: human oracle's annotator.
#: The two deliberate omissions are `annotator` and `session`. Everything else
#: `to_record` builds belongs here, and `test_a_callers_extra_field_cannot_
#: redefine_the_schema` derives that from the built record rather than trusting
#: this list: a hand-maintained list of computed fields drifts silently when one
#: is renamed (a stale `ts` here beside a written `t` would let `t=` in
#: `**extra` overwrite the computed timestamp for a whole dataset).
_PROTECTED = frozenset({
    "schema", "t", "run", "iteration", "restart", "left_id", "right_id",
    "preferred", "source", "judge", "comparator", "task", "left", "right",
    "left_span", "right_span",
})

#: Built fields a producer MAY set, and the reason `_PROTECTED` is not simply
#: "every key `to_record` writes": protecting these would silently drop the
#: human oracle's annotator.
_CALLER_SETTABLE = frozenset({"annotator", "session"})


def side_of(label: Any) -> Optional[str]:
    """`1|0|-1` -> `"left"|"right"|"tie"`; anything else -> `None`.

    `None` in, `None` out, and an unrecognised integer also maps to `None`
    rather than to a side: a label this module does not understand is an
    abstention as far as the dataset is concerned, which is the reading that
    cannot invent a preference that was never expressed.
    """
    if label is None:
        return None
    try:
        return _LABEL_TO_SIDE.get(int(label))
    except (TypeError, ValueError):
        return None


def _int(value: Any, default: int = 0) -> int:
    """An int, or the default. A record must not fail to be written over a field."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def judge_of(oracle: Any) -> str:
    """What KIND of judge an oracle is, for the record's `judge` field.

    Asked of the oracle rather than inferred from the call site, because both
    human-ish paths -- the `human` comparator and the human-feedback phase --
    go through the same `ctx.human`, so the call site cannot tell a person from
    a script. The oracle can, and declares it.

    An oracle that declares nothing is `"unknown"`, never `"human"`: a wrong
    claim that a person judged something is the one error this field exists to
    prevent, and silence is not evidence of a person.
    """
    if oracle is None:
        return "none"
    kind = getattr(oracle, "judge_kind", None)
    return str(kind) if kind else "unknown"


def _traj_pointer(traj: Any) -> Optional[Dict[str, Any]]:
    """A clip as a pointer plus scalars -- never its states, actions or frames."""
    if traj is None:
        return None
    out: Dict[str, Any] = {}
    for key, attr in (("length", "length"), ("return", "ret"),
                      ("success", "success"), ("video", "video_path")):
        value = getattr(traj, attr, None)
        if value is None:
            continue
        if key == "return":
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
        elif key == "length":
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
        elif key == "success":
            value = bool(value)
        else:
            value = str(value)
        out[key] = value
    return out or None


def _span(value: Any) -> Optional[List[int]]:
    """A `(lo, hi)` step range as a two-int list, or None for the whole rollout."""
    if value is None:
        return None
    try:
        lo, hi = value
        return [int(lo), int(hi)]
    except (TypeError, ValueError):
        return None


def _run_id(ctx: Any) -> str:
    """The run directory's NAME -- `<config>-<confighash>-<stamp>`.

    The name and not the path: it is what a reader keys a run on and what a
    person recognises, and a pooled dataset that carried absolute paths
    would be tied to the machine that wrote it.
    """
    path = getattr(getattr(ctx, "rundir", None), "path", None)
    return Path(path).name if path else ""


def _cfg(ctx: Any, key: str, default: Any = None) -> Any:
    try:
        return ctx.cfg.get(key, default)
    except Exception:  # noqa: BLE001 - a Context without a config carries no fields
        return default


def to_record(ctx: Any, pref: Any, state: Any = None, **extra: Any) -> Dict[str, Any]:
    """One `Preference` as the line that will be written. Pure; no I/O.

    Separated from `record` so a producer that is not a search -- a human
    annotation session, which belongs to no single run -- can build the same
    shape and write it wherever it lives. That is the flexibility the format is
    for: one record type, several producers, one reader.

    `**extra` is merged LAST but cannot take a field this builds, for
    `judgments.note_query`'s reason: a caller's stray key must not be able to
    silently redefine `source` or `preferred` for every reader downstream.
    """
    built = {
        "schema": SCHEMA,
        # `t`, not `ts`: `journal.jsonl` already calls a record's wall-clock
        # stamp `t`, and the parallelism and resume guards scrub that name as
        # volatile. A second spelling for the same idea would make this file
        # the one artifact those guards could not compare.
        "t": time.time(),
        "run": _run_id(ctx),
        "iteration": _int(getattr(pref, "iteration", -1), -1),
        # From the STATE, not the config: `n_restarts` is how many restarts a
        # run does, `state.restart` is which one this is. Without it two
        # restarts of the same config are indistinguishable in a pooled file,
        # since they share a run directory.
        "restart": _int(getattr(state, "restart", 0)),
        "left_id": str(getattr(pref, "left_id", "") or ""),
        "right_id": str(getattr(pref, "right_id", "") or ""),
        "preferred": side_of(getattr(pref, "label", None)),
        # VERBATIM, never normalised: `source` is whatever the search set, and
        # for a `D_pref` record that is the COMPARATOR's name
        # (`llm_on_vlm_captions`, `vlm`, `human`) rather than a category. It is
        # kept exactly as the run saw it so a reader can always recover the
        # mechanism.
        "source": str(getattr(pref, "source", "") or ""),
        # …and the category beside it, because `source` alone cannot be
        # filtered on. `llm_on_vlm_captions` does not advertise that a model
        # judged it, and `human` means the human COMPARATOR whose oracle may be
        # a script. A pooled dataset that ignores the distinction mixes a
        # model's judgement, a person's and an
        # inferred one. The producer states it; this module never guesses.
        "judge": str(extra.pop("judge", "") or "unknown"),
        "comparator": str(_cfg(ctx, "evaluate.preferences.comparator", "") or ""),
        "task": str(_cfg(ctx, "problem.task_description", "") or ""),
        "left": _traj_pointer(getattr(pref, "left_traj", None)),
        "right": _traj_pointer(getattr(pref, "right_traj", None)),
        "left_span": _span(getattr(pref, "left_span", None)),
        "right_span": _span(getattr(pref, "right_span", None)),
        # Machine judgements have no annotator and no session. Present and null
        # rather than absent: "no person made this" and "this file predates the
        # field" are different facts, and only one of them is about the data.
        "annotator": None,
        "session": None,
    }
    for key, value in extra.items():
        if key not in _PROTECTED:
            built[key] = value
    return built


def record(ctx: Any, pref: Any, state: Any = None, **extra: Any) -> None:
    """Append one preference to `<rundir>/preferences.jsonl`. Swallows everything."""
    path = _path(ctx)
    if path is None:
        return
    try:
        line = json.dumps(to_record(ctx, pref, state, **extra), default=str) + "\n"
    except Exception:  # noqa: BLE001 - an unserialisable record is dropped, not raised
        log.debug("could not serialise a preference record", exc_info=True)
        return
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line)
    except Exception:  # noqa: BLE001 - a lost record is not a lost run
        log.debug("could not append to the preference log", exc_info=True)


def _path(ctx: Any) -> Optional[Path]:
    """`<rundir>/preferences.jsonl`, or None when there is nowhere to write.

    `ctx.rundir` is None for `--dry-run` and for the suite's hand-built
    contexts, which is not an error -- the same guard `Context.event` applies
    before touching the journal.
    """
    path = getattr(getattr(ctx, "rundir", None), "path", None)
    return None if path is None else Path(path) / FILENAME


def read(source: Any) -> List[Dict[str, Any]]:
    """Every record in one log. A torn line is SKIPPED, never raised on.

    Accepts a run directory or the file itself, because a caller pooling a
    sweep has directories and a caller debugging one has a path.

    A partial last line is the normal end of a killed run, and the whole point
    of the format is that everything before it is still good.
    """
    path = Path(getattr(source, "path", source) or "")
    if path.is_dir():
        path = path / FILENAME
    out: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def pool(sources: Iterable[Any]) -> Iterator[Dict[str, Any]]:
    """Every record across many runs, in the order the sources are given.

    No rewriting and no run argument: each record already names its own run,
    which is what makes concatenation a valid dataset rather than a lossy one.
    A source with no log contributes nothing and is not an error -- a sweep
    contains runs that made no comparisons at all.
    """
    for src in sources:
        for rec in read(src):
            yield rec


def preferred_id(rec: Dict[str, Any]) -> Optional[str]:
    """The winning candidate's id, or None for a tie or an abstention.

    Derived on READ rather than stored: a stored copy is a second source of
    truth that can disagree with the sides it was derived from, and this is one
    line of arithmetic.
    """
    side = rec.get("preferred")
    if side == "left":
        return rec.get("left_id")
    if side == "right":
        return rec.get("right_id")
    return None
