"""Run artifacts: directory layout, JSONL journal, checkpoints.

Three conventions worth stating:
  * the fully-resolved config is written out and hashed; the hash is the run ID;
  * a candidate that never compiled and a candidate a screen rejected are
    different populations, so `failure` and `failure_kind` are always recorded
    and never collapsed into a single "didn't work" bucket;
  * **a run says whether it finished.** `status.json` is written `running` when
    the directory is created and rewritten `ok`/`failed` on the way out, so a
    run that died in a way Python never saw -- SIGSEGV, a scheduler cancel, a
    batch walltime -- is positively identifiable rather than merely missing
    files. Without it, a crashed search leaves `config.resolved.yaml`, a journal
    that simply stops, `candidates/` and `state/`, and nothing else:
    structurally indistinguishable to any collector not specifically testing for
    the absence of `result.json`, which is exactly the shape a large job array
    produces when its stdout lives on a compute node nobody reads. See
    `RunDir.finish`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import Candidate, CandidateReport, Selection

log = logging.getLogger("bird")


_CODE_VERSION: Optional[Tuple[str, bool]] = None


def code_version() -> Tuple[str, bool]:
    """``(commit, dirty)`` of the checkout ``bird.py`` was loaded from.

    ``config_hash`` says WHAT ran; ``commit`` records which code produced the run,
    so results can be tied to a code version. ``commit`` is the 40-char ``HEAD`` sha, or ``"unknown"``
    when this is not a git checkout / git is absent / the tree has no ``HEAD``.
    ``dirty`` is True when the working tree was not clean at run start (tracked
    edits OR untracked files), because then ``commit`` names code that is not
    exactly what ran -- and also when cleanliness could not be verified (the
    ``git status`` fork failed or timed out): unverified is not clean.

    Cached: computed once per process, never a ``git`` fork per status write.
    Deliberately NOT a config key: it identifies the RUN, not the experiment, so
    it must not move the config hash (`Config.hash` covers only the config).
    """
    global _CODE_VERSION
    if _CODE_VERSION is not None:
        return _CODE_VERSION
    # LEFT AS `parents[1]`, DELIBERATELY, and this is the one of the six
    # sites that does not go through `bird.paths`.
    #
    # It feeds `git -C <repo> rev-parse HEAD`. Under a non-editable install
    # that directory is site-packages, which is not a git repository, so the
    # subprocess fails and the existing fallback records `"unknown"` -- which
    # is the TRUE answer: an installed wheel has no commit. Routing it
    # through the resolver would turn a correct "unknown" into an exception
    # on a path whose whole job is to never fail a run, and the docstring
    # above already says this identifies the run rather than the experiment.
    #
    # The one thing it must not do is report some OTHER repo's sha, and it
    # cannot: site-packages is not a checkout, and if it somehow were, the
    # sha would be that project's and visibly not ours.
    repo = str(Path(__file__).resolve().parents[1])
    commit, dirty = "unknown", False
    try:
        commit = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    if commit != "unknown":
        try:
            porcelain = subprocess.run(
                ["git", "-C", repo, "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            dirty = bool(porcelain.strip())
        except (OSError, subprocess.SubprocessError):
            # FAIL CLOSED. A `status` that timed out or errored verified
            # nothing, and `False` here would read as "clean" -- the one
            # wrong answer a provenance stamp must not give. `True` means
            # "not verified clean", which is what a consumer excluding runs
            # by commit needs.
            dirty = True
    _CODE_VERSION = (commit, dirty)
    return _CODE_VERSION


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _curriculum_record(cur: Any) -> Optional[Dict[str, Any]]:
    """One curriculum as the three questions a reader asks of it, or None.

    **None, not `{}` and not `[]`.** Every other slot in `save_records`'
    payload is a collection whose empty value is honestly empty; a curriculum
    either exists or does not. `[]` there would make "this method carries no
    curriculum" -- the correct, normal state of every published config --
    indistinguishable from "a curriculum with no stages", which no author can
    produce (`curriculum._MIN_STAGES`) and which would read as a broken run
    to anyone inspecting the artifact. The third state, the key being ABSENT, then means what it
    should: the run was written by a version that did not record curricula.
    Three states, none of them graded.

    `summary()` supplies n_stages/index/current/complete/passed/splits/patience
    and the whole `history`. THREE things it does not carry are added back
    verbatim, because they are what a reader needs and they are small (one
    short string and one float per passed stage). Naming them exactly matters:
    a reader who believes a key comes from `summary()` will look for a bug in
    the wrong file:

      * `stages` -- the ordered list, which `summary()` reduces to `n_stages`;
      * `checkpoints` -- which `policy_ref` was current when each stage passed,
        i.e. what a regression rollback restores. `summary()` reduces it to
        `passed`, a sorted list of the stage indices, which is a count of
        which stages rather than what they handed on;
      * `pass_scores` -- the gate score each stage passed WITH, which is what
        a later regression is measured against. `summary()` does not carry it
        in any form. The gate threshold is not here
        and does not belong here: it is a config value, read from
        `config.resolved.yaml`.

    Duck-typed on `summary()` rather than isinstance-checked, so this module
    keeps importing nothing from `bird.state` -- and a run whose state object
    predates the method degrades to None with the key present, which is a
    different and honest claim from the key being absent.
    """
    if cur is None:
        return None
    summary = getattr(cur, "summary", None)
    if not callable(summary):
        return None
    try:
        out = dict(summary())
    except Exception:  # pragma: no cover - a state object we do not own
        return None
    # The ORDERED LIST verbatim, which `summary()` reduces to `n_stages`.
    # Added here and NOT to `summary()` on this tree's standing division:
    # `records/` is the verbatim artifact and `state/iterNN.json` is the
    # deliberately lossy report, so growing the summary would move the shape of
    # a file whose whole job is to be cheap. A reader needs the strings.
    out["stages"] = [str(x) for x in (getattr(cur, "stages", None) or [])]
    out["checkpoints"] = {str(k): str(v) for k, v in
                          (getattr(cur, "checkpoints", None) or {}).items()}
    scores: Dict[str, Any] = {}
    for k, v in (getattr(cur, "pass_scores", None) or {}).items():
        try:
            scores[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    out["pass_scores"] = scores
    return out


def _traj_summary(t: Any) -> Optional[Dict[str, Any]]:
    """A Trajectory as its cheap facts plus a pointer, never its payload.

    `records/iterNN.json` must not become the trajectory dump
    `save_train_result` deliberately refuses to write: states and actions are
    large, and the frames a reader actually wants are already under `videos/`
    -- so `video_path` is the pointer, and the presence flags say what the
    live object held, so what was dropped is stated rather than inferred.
    """
    if t is None:
        return None
    length = int(getattr(t, "length", 0) or 0)
    try:
        ret = float(getattr(t, "ret", 0.0) or 0.0)
    except (TypeError, ValueError):
        ret = None
    if ret is not None and not math.isfinite(ret):
        # A reward returning nan is a normal thing for this project to
        # produce, and `json.dumps` emits bare NaN/Infinity by default --
        # legal for Python's reader, rejected by strict JSON parsers. Recorded
        # as null: None is not a number, here as everywhere in this tree.
        ret = None
    components = getattr(t, "component_values", None) or {}
    return {
        "length": length,
        "return": ret,
        # Both TAC and TPE rank on this, for the reason `Trajectory` defines
        # it: successful episodes terminate early, so cumulative return is
        # length-biased toward failures.
        "mean_per_step_return": (ret / max(length, 1)) if ret is not None else None,
        "success": bool(getattr(t, "success", False)),
        "video_path": getattr(t, "video_path", None),
        "has_states": getattr(t, "states", None) is not None,
        "has_actions": getattr(t, "actions", None) is not None,
        "components": sorted(str(k) for k in components),
    }


class RunDir:
    """One run's output tree.

        runs/<name>-<confighash>-<timestamp>/
          config.resolved.yaml   the exact config that ran
          status.json            running / ok / failed  (written FIRST)
          journal.jsonl          one line per stage event
          candidates/            every generated program, valid or not
          state/                 per-iteration RunState snapshots (lossy REPORT --
                                 counts, ids, histories; see RunState.summary)
          records/               the carried objects those counts summarise,
                                 verbatim or as pointer+stats (see
                                 save_records for the shape and the exclusions)
          judgments/             what the (V)LM judge was SHOWN, per query
                                 (`output.judge_trace.record`, off by
                                 default -- see bird/judgments.py for the shape,
                                 and why keeping the verdict was not enough)
          budget.json            the cost column
          result.json            the run's return value
    """

    def __init__(self, root: str | Path, name: str, config_hash: str,
                 out_root_source: str = "cli"):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = Path(root) / f"{name}-{config_hash}-{stamp}"
        #: Where `root` came from -- "cli" for an explicit `--out`, "config" for
        #: the resolved `output.dir` -- so a run directory can say why it is
        #: where it is. Both roots are legal.
        self.out_root_source = out_root_source
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "candidates").mkdir(exist_ok=True)
        (self.path / "state").mkdir(exist_ok=True)
        self._journal = open(self.path / "journal.jsonl", "a", buffering=1)
        self._started = time.time()
        self.name = name
        self.config_hash = config_hash
        #: Which invocation of the same command line this is. 1 for a fresh run;
        #: `leg > 1` is one of the three independent markers that say a run was
        #: resumed (the others are `result.json.legs` and the journal's `resume`
        #: lines).
        self.leg = 1
        self._resumes: List[Dict[str, Any]] = []
        #: `make_tracker`'s in-process fallback, when there was one (`record_tracker_degraded`).
        self._tracker_degraded: Optional[Dict[str, Any]] = None
        #: Journal lines written so far. A COUNT, not a clock: it is what lets a
        #: resumed leg find exactly the events the dead leg wrote after the last
        #: checkpoint, without putting a timestamp inside a checkpoint body.
        self.n_events = 0
        # Written before anything else can fail, and it records WHERE the run is,
        # because on a cluster the answer to "what happened" lives in a scheduler
        # log on a filesystem the submitting node may not see.
        self._write_status("running", name=name, config_hash=config_hash)

    # -- adoption (`loop.resume_from`) --

    @classmethod
    def adopt(cls, path: str | Path) -> "RunDir":
        """Continue an existing run directory instead of creating a new one.

        No `mkdir` of a new stamped name, and deliberately no `write_config`:
        `config.resolved.yaml` is written ONCE and never rewritten, so it stays
        the exact configuration the run started under; `bird.py` asserts the
        existing file still matches the config being resumed.

        `started` is preserved from the previous leg so `elapsed_s` keeps
        meaning "since this search began", and the journal is opened for append
        so the seam is a `resume` line rather than a second file.
        """
        self = cls.__new__(cls)
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"cannot adopt {self.path}: not a directory")
        prev: Dict[str, Any] = {}
        status_path = self.path / "status.json"
        if status_path.exists():
            try:
                prev = json.loads(status_path.read_text())
            except (OSError, json.JSONDecodeError):
                prev = {}
        (self.path / "candidates").mkdir(exist_ok=True)
        (self.path / "state").mkdir(exist_ok=True)
        journal = self.path / "journal.jsonl"
        self.n_events = sum(1 for line in journal.read_text().splitlines()
                            if line.strip()) if journal.exists() else 0
        self._journal = open(journal, "a", buffering=1)
        self._started = float(prev.get("started") or time.time())
        self.name = prev.get("name") or self.path.name
        self.config_hash = prev.get("config_hash") or ""
        self.leg = int(prev.get("leg") or 1) + 1
        self._resumes = list(prev.get("resumes") or [])
        self.out_root_source = str(prev.get("out_root_source") or "")
        # Not carried over: whether THIS leg's tracker degrades is decided by this
        # leg's environment, and `make_tracker` records it again if it does.
        self._tracker_degraded = None
        self._write_status("running", name=self.name, config_hash=self.config_hash)
        return self

    def record_resume(self, record: Dict[str, Any]) -> None:
        """Put this leg's seam in `status.json` where a collector will find it."""
        self._resumes.append(record)
        self._write_status("running", name=self.name, config_hash=self.config_hash)

    def record_tracker_degraded(self, record: Dict[str, Any]) -> None:
        """Put the tracker's in-process fallback (`make_tracker`: `{from, to, reason}`)
        in `status.json`, beside the resume seams and for the same reason: a run whose
        dashboard never existed must say so in the artifact, not only on a compute
        node's stderr. Kept on every later status write, the terminal one included."""
        self._tracker_degraded = dict(record)
        self._write_status("running", name=self.name, config_hash=self.config_hash)

    # -- status --

    def _write_status(self, status: str, **extra: Any) -> None:
        code_commit, code_dirty = code_version()
        payload: Dict[str, Any] = {
            "status": status,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "argv": list(sys.argv),
            "started": round(self._started, 3),
            "updated": round(time.time(), 3),
            "elapsed_s": round(time.time() - self._started, 3),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            # WHICH CODE ran -- the commit `bird.py` was loaded from, and whether
            # the tree was clean (see `code_version`). On EVERY write, so a run
            # that died still names its code. "unknown"/false when not a git tree.
            "code_commit": code_commit,
            "code_dirty": code_dirty,
            "leg": getattr(self, "leg", 1),
            "resumes": list(getattr(self, "_resumes", ())),
            # Carried on EVERY write, terminal ones included: `RunDir.adopt`
            # reads them back, and a run that died left its last status as
            # `failed`, not as the opening `running` line.
            "name": getattr(self, "name", ""),
            "config_hash": getattr(self, "config_hash", ""),
            # Whether the CLI flag or `output.dir` chose the run root. The root
            # itself is where this file sits and is NOT recorded: an absolute
            # path here differs between two otherwise identical runs, which is
            # exactly what `tests/test_resume.py` compares.
            "out_root_source": getattr(self, "out_root_source", ""),
        }
        # Only when it happened: a run under `none` has no dashboard to have lost.
        degraded = getattr(self, "_tracker_degraded", None)
        if degraded is not None:
            payload["tracker_degraded"] = degraded
        payload.update(extra)
        # `os.replace` so a reader never sees a half-written file, and a crash
        # during the rewrite leaves the previous status rather than an empty one.
        tmp = self.path / "status.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
        os.replace(tmp, self.path / "status.json")

    def finish(self, status: str = "ok", exc: Optional[BaseException] = None) -> None:
        """Stamp the terminal status. `status.json` still saying `running` after
        the process is gone is the crash marker -- there is no way to write one
        from inside a SIGSEGV, so the only workable convention is to write the
        *optimistic* state up front and clear it on the way out.

        Called from a `finally`, so it must not raise: an artifact-writing fault
        that masked the original exception would be strictly worse than no
        status at all."""
        extra: Dict[str, Any] = {}
        if exc is not None:
            extra["error_type"] = type(exc).__name__
            extra["error"] = str(exc)
            extra["traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__))
        try:
            self._write_status(status, **extra)
            self.event("run_finished", status=status,
                       error_type=extra.get("error_type", ""),
                       error=extra.get("error", ""))
        except Exception:  # pragma: no cover - never mask the real failure
            log.exception("could not write status.json for %s", self.path)

    # -- writes --

    def write_config(self, yaml_text: str, lineage: List[str]) -> None:
        (self.path / "config.resolved.yaml").write_text(
            f"# resolved config -- lineage: {' <- '.join(reversed(lineage))}\n" + yaml_text)

    def event(self, stage: str, **fields: Any) -> None:
        rec = {"t": round(time.time(), 3), "stage": stage, **fields}
        self._journal.write(json.dumps(rec, default=_json_default) + "\n")
        self.n_events = getattr(self, "n_events", 0) + 1

    def write_task_spec(self, spec_id: str, yaml_text: str, resolved: Dict[str, Any]) -> None:
        """The task definition this run used, copied VERBATIM, plus what it resolved to.

        Written once and never rewritten, for the same reason as
        `config.resolved.yaml`: it is the other half of "what did this run think the task
        was". `Config.hash()` covers the resolved config dict and nothing else, so two
        runs whose spec CONTENT differs would otherwise share a hash, a directory name,
        and adoptability -- a silently wrong resume, which is worse than a crash.
        Materialising the spec into the config instead would make `load()` depend on
        `tasks/` and move every existing config hash; recording it beside the config
        and refusing to adopt across a change costs neither.

        Verbatim rather than re-dumped so that `diff` against `tasks/` means something.
        """
        (self.path / "task_spec.yaml").write_text(yaml_text)
        (self.path / "tasks.resolved.json").write_text(
            json.dumps({"task_id": spec_id, **resolved}, indent=2, default=_json_default))

    def save_candidate(self, c: Candidate) -> None:
        d = self.path / "candidates" / f"iter{c.iteration:02d}_{c.cand_id}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "reward.py").write_text(c.reward_code or "")
        if c.observation_code:
            (d / "observation.py").write_text(c.observation_code)
        if c.raw_response:
            (d / "response.txt").write_text(c.raw_response)
        if c.prompt_messages:
            (d / "prompt.json").write_text(
                json.dumps(c.prompt_messages, indent=2, default=_json_default))
        (d / "meta.json").write_text(json.dumps({
            "cand_id": c.cand_id,
            "iteration": c.iteration,
            "parent_id": c.parent_id,
            "valid": c.valid,
            "screened_out": c.screened_out,
            # Never-compiled vs screened: keep these distinguishable. "" means
            # no failure.
            "failure": c.failure,
            "failure_kind": c.failure_kind,
            "weights": c.weights,
            "component_names": c.component_names,
            "dr_config": c.dr_config,
            # The design thought a candidate was generated with: GT's two-stage
            # prose, or RF-Agent's brace. Recorded here so the idea a tree
            # prompt showed for a node need not be recovered from
            # `response.txt`. "" when the method asks for none.
            "nl_spec": c.nl_spec,
            "verify_records": c.verify_records,
            "meta": c.meta,
        }, indent=2, default=_json_default))

    def save_train_result(self, result: Any) -> None:
        """Persist one candidate's TrainResult beside its program.

        `save_candidate` writes the Candidate -- the code, the prompt, the
        verdict. Without this, everything the *learner* produced would live
        only in memory and in a log line: the per-seed curves, the checkpoint
        trace `evaluate.fitness.checkpoint_aggregation` reduces, the skip
        reason, and `train.hyperparameter_search`'s grid scoreboard, which is
        the whole point of Singh's protocol. A run could then report "best
        (alpha, epsilon)" on stdout and leave no way to check it afterwards.

        Trajectories are dropped: they are large, they are already summarised by
        the curves, and the frames a human or a VLM looks at are written by
        `observability.record_rollouts` under `videos/`.
        """
        d = self.path / "candidates" / f"iter{result.candidate.iteration:02d}_{result.cand_id}"
        d.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in asdict(result).items()
                   if k not in ("candidate", "trajectories", "store_trajectories")}
        payload["n_trajectories"] = len(getattr(result, "trajectories", ()) or ())
        # The dedicated TPE-store pool is dropped for the same size reason
        # (100 x horizon arrays per candidate); the count is what the artifact
        # needs to say the collection happened.
        payload["n_store_trajectories"] = len(
            getattr(result, "store_trajectories", ()) or ())
        (d / "train_result.json").write_text(
            json.dumps(payload, indent=2, default=_json_default))

    def save_report(self, report: Any) -> None:
        """Persist one candidate's CandidateReport beside its program.

        `save_candidate` keeps what the generator wrote and `save_train_result`
        what the learner produced; this keeps the *evaluation* -- the ranking
        scalar, the prose the next prompt is built from, and RDA's per-subtask
        credit assignment. The journal records only `feedback_chars`, i.e. its
        LENGTH: three thousand characters of VLM analysis, stored as an
        integer. Without this file it would survive only indirectly, inside the
        NEXT iteration's `prompt.json`, and the final iteration's evaluation
        would be unrecoverable.

        `subtasks` uses the field names RDA's own appendix (§7.3) prints --
        `number`, `name`, `behavior`, `score`, `analysis` -- so the artifact is
        diffable against the paper rather than against a translation of it.
        `behavior` is emitted per subtask when `report.subtask_behaviors`
        carries it: the judge IS asked for the
        paper's describe-behaviour step (App. 7.3's two-step, in
        `_vlm_trajectory_analysis`), so a present key is a real answer. An
        ABSENT key still means the judge did not supply one -- an older run, a
        reply that skipped the field, a method with no VLM judge -- and stays
        absent rather than becoming `""`, which would assert the VLM looked
        and saw nothing. Missing means unanswered, which is the true statement
        and the one that keeps this file honest as a fidelity record.

        Written for every method, not just RDA: `subtasks` is simply empty where
        stage 4 produced no per-subtask evidence, which is the same shape the
        rest of the artifact tree uses for an absent population.
        """
        c = report.candidate
        d = self.path / "candidates" / f"iter{c.iteration:02d}_{report.cand_id}"
        d.mkdir(parents=True, exist_ok=True)
        scores = report.subtask_scores or {}
        rationales = report.subtask_rationales or {}
        behaviors = getattr(report, "subtask_behaviors", None) or {}
        subtasks = []
        for i, (name, score) in enumerate(scores.items(), start=1):
            entry = {"number": i, "name": name, "score": score}
            did = behaviors.get(name)
            if did:
                entry["behavior"] = did
            why = rationales.get(name)
            if why:
                entry["analysis"] = why
            subtasks.append(entry)
        (d / "report.json").write_text(json.dumps({
            "cand_id": report.cand_id,
            "iteration": c.iteration,
            "fitness": report.fitness,
            "fitness_source": report.fitness_source,
            "per_seed_fitness": report.per_seed_fitness,
            "subtasks": subtasks,
            "feedback": report.feedback,
            "feedback_channel": report.feedback_channel,
            "similarity": report.similarity,
            "safety": report.safety,
            "descriptors": report.descriptors,
            "meta": report.meta,
        }, indent=2, default=_json_default))

    def save_state(self, state: Any, iteration: int) -> None:
        (self.path / "state" / f"iter{iteration:02d}.json").write_text(state.to_json())

    def save_records(self, state: Any, iteration: int) -> None:
        """The carried objects themselves, beside the counts that summarise them.

        `state/iterNN.json` is `RunState.summary()`: six of a run's richest
        objects survive it only as a LENGTH (`n_dialogue: 9`,
        `n_preferences: 14`, ...). A reader of run dirs could then say a run
        held fourteen preferences and never show one -- the objects are real
        dataclasses in memory, discarded at serialisation rather than absent
        from the model. `checkpoints/` is not the answer: it exists only under
        `loop.resume_from` (null in every published config), and the snapshot
        is a report, not a checkpoint -- bird/state.py's
        docstring owns the rule that those two never merge, and this file is
        a third report, not a blend of them.

        One file per iteration, `records/iterNN.json`, mirroring `RunState`
        AFTER `apply_carry` -- a slot the config does not carry is honestly
        `[]` here, exactly as the search itself will see it next iteration.
        Every slot is always present, `[]` where the method holds nothing, so
        "this run predates records/" and "this method carries nothing" stay
        distinguishable (the `save_report` convention).

        Verbatim or pointer, decided per object by what it costs:

          * `dialogue`, `subtasks`, `failure_memory` -- verbatim. Prose and
            code strings, bounded by `update.prompt.max_length_tokens` and by
            the population itself.
          * `preferences` -- the labels verbatim (ids, label, source,
            iteration); each side's Trajectory reduced by `_traj_summary`.
          * `trajectory_store` -- one `_traj_summary` per entry.
            States/actions/frames never land here: they are why
            `save_train_result` drops trajectories, and the same reasoning
            holds unchanged.
          * `archive` -- one row per cell: island, coords (`"elite"`-first
            coords are elite slots, int-only coords are MAP-Elites bins --
            bird/components/update.py's shared-store convention), occupant
            `cand_id` (the program itself is under `candidates/`), its
            iteration, and fitness. A `-inf` fitness (an unscored occupant)
            is recorded as null: None is not a number here, as everywhere in
            this tree.
          * `curriculum` -- verbatim, via `CurriculumState.summary()` plus the
            three things the summary does not carry (`stages`, which it
            reduces to `n_stages`; `checkpoints`, which it reduces to the
            `passed` key list; and `pass_scores`, which it omits), and
            **`None` rather than `[]` when the method carries no curriculum.**
            It is the one slot here whose empty value is not a list: every
            other slot is a collection that can be honestly empty, while a
            curriculum either exists or does not, and `[]` would make "no
            curriculum" indistinguishable from "a curriculum with no stages"
            -- a state `author`s cannot produce and `_MIN_STAGES` exists to
            forbid. The `history` inside it is the whole per-stage trace, and
            it is what a reader must be given rather than `index` and
            `entered_at`: a resplit rewrites the stage list mid-run and a
            regression rollback moves the policy backwards, so a
            stage-to-iteration mapping recomputed from the two counters is
            wrong exactly where the interesting runs are (the rule
            `output.trajectory_trace` states as "the alignment is recorded,
            never recomputed").

            EVERY `CARRY_SLOTS` ENTRY MUST APPEAR HERE. A slot added to
            `CARRY_SLOTS` and not to this payload breaks the invariant the
            paragraph above asserts: a run carrying it would write a
            `records/iterNN.json` byte-identical in SHAPE to a run that does
            not, so "this run ran no curriculum" and "this artifact does not
            record curricula" could not be told apart -- precisely the
            confusion this file exists to end.
          * `critics` -- R*'s critic population verbatim (carry slot
            `critic_population`), one source string per LLM-written comparison
            program. Verbatim rather than counted for the reason `save_report`
            gives about `feedback_chars`: these programs decide every
            preference label the parameter alignment then fits to, so "five
            critics voted" without the five is a number nobody can check.
            `[]` when the method authors none.
          * `tree` -- one row per node of RF-Agent's search tree (carry slot
            `search_tree`), sorted by `sim_index` so the virtual root comes
            first and the rest read in the order they were trained: `node_id`,
            `parent_id`, `children`, `depth`, `action`, `iteration`, `score`,
            `q`, `visits`, `self_verify`, `sim_index`. Statistics only -- the
            report a node summarises is under `candidates/<iter>_<id>/`, and
            `bird/state.py::TreeNode` is the reason it is not duplicated
            here. `[]` when the method carries no tree, on the same rule as
            `archive`: a tree either has nodes or it does not, and an empty
            list is the honest empty value.

        The counts in the snapshot are deliberately KEPT and never derived
        from this file: they are cheap, and a count that disagrees with the
        records it summarises is itself a useful alarm.

        Like `save_state`, keyed by iteration only, so with
        `loop.n_restarts > 1` the last restart to write owns the file -- the
        same caveat as for snapshots.
        """
        prefs = []
        for p in getattr(state, "preferences", None) or []:
            prefs.append({
                "left_id": getattr(p, "left_id", None),
                "right_id": getattr(p, "right_id", None),
                "label": getattr(p, "label", None),
                "source": getattr(p, "source", ""),
                "iteration": getattr(p, "iteration", -1),
                "left": _traj_summary(getattr(p, "left_traj", None)),
                "right": _traj_summary(getattr(p, "right_traj", None)),
            })

        archive = []
        for cell in (getattr(state, "archive", None) or {}).values():
            report = getattr(cell, "report", None)
            fitness = getattr(cell, "fitness", None)
            try:
                fitness = (float(fitness) if fitness is not None
                           and math.isfinite(float(fitness)) else None)
            except (TypeError, ValueError):
                fitness = None
            archive.append({
                "island": getattr(cell, "island", 0),
                "coords": list(getattr(cell, "coords", ()) or ()),
                "cand_id": getattr(report, "cand_id", None) if report else None,
                "iteration": (getattr(getattr(report, "candidate", None),
                                      "iteration", None) if report else None),
                "fitness": fitness,
            })

        tree = []
        for node in (getattr(state, "tree", None) or {}).values():
            score = getattr(node, "score", None)
            try:
                score = (float(score) if score is not None
                         and math.isfinite(float(score)) else None)
            except (TypeError, ValueError):
                score = None
            self_verify = getattr(node, "self_verify", None)
            try:
                self_verify = float(self_verify) if self_verify is not None else None
            except (TypeError, ValueError):
                self_verify = None
            tree.append({
                "node_id": getattr(node, "node_id", None),
                "parent_id": getattr(node, "parent_id", None),
                "children": list(getattr(node, "children", None) or []),
                "depth": int(getattr(node, "depth", 0) or 0),
                "action": getattr(node, "action", ""),
                "iteration": int(getattr(node, "iteration", -1)),
                "score": score,
                "q": float(getattr(node, "q", 0.0) or 0.0),
                "visits": int(getattr(node, "visits", 0) or 0),
                "self_verify": self_verify,
                "sim_index": int(getattr(node, "sim_index", 0) or 0),
                # `mean = total / visits` is the backup's third term; without it a
                # recorded `q` cannot be re-derived from the row.
                "total": float(getattr(node, "total", 0.0) or 0.0),
            })
        tree.sort(key=lambda row: row["sim_index"])

        payload: Dict[str, Any] = {
            "schema": 1,
            "iteration": int(iteration),
            "restart": int(getattr(state, "restart", 0) or 0),
            "dialogue": list(getattr(state, "dialogue", None) or []),
            "preferences": prefs,
            "trajectory_store": [
                dict(_traj_summary(t) or {}, index=i)
                for i, t in enumerate(getattr(state, "trajectory_store", None) or [])],
            "subtasks": [str(s) for s in getattr(state, "subtasks", None) or []],
            "failure_memory": list(getattr(state, "failure_memory", None) or []),
            "archive": archive,
            "curriculum": _curriculum_record(getattr(state, "curriculum", None)),
            "tree": tree,
            # R*'s critic population (carry slot `critic_population`): the
            # PROGRAMS verbatim, because a critic is a judge and a judge that
            # cannot be read cannot be checked. They are bounded -- five short
            # functions per run, authored once -- and the alternative is the
            # `feedback_chars` mistake again: a count of how many judges voted,
            # with no way to see what any of them tested for.
            "critics": [str(c) for c in getattr(state, "critics", None) or []],
        }
        d = self.path / "records"
        d.mkdir(exist_ok=True)
        # tmp + os.replace, the `_write_status` convention: a reader may poll
        # this directory while the run writes it, and a bare write_text has a
        # window in which a reader sees a torn file, which it could only
        # report as unreadable rather than as content. The tmp name does not match
        # `iter*.json`, so even a crash mid-write leaves nothing a reader
        # would list.
        tmp = d / f"iter{iteration:02d}.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
        os.replace(tmp, d / f"iter{iteration:02d}.json")

    def save_budget(self, budget: Any) -> None:
        (self.path / "budget.json").write_text(
            json.dumps(budget.report(), indent=2, default=_json_default))

    def save_result(self, result: Dict[str, Any]) -> None:
        (self.path / "result.json").write_text(
            json.dumps(result, indent=2, default=_json_default))

    def close(self) -> None:
        try:
            self._journal.close()
        except Exception:  # pragma: no cover
            pass


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


def summarise_reports(reports: List[CandidateReport]) -> str:
    rows = []
    for r in reports:
        if not r.candidate.valid:
            mark = f"INVALID({r.candidate.failure_kind})"
        elif r.candidate.screened_out:
            mark = "SCREENED"
        elif r.result.timed_out:
            # NOT "SKIPPED". A skip is CARD deciding not to train (the whole
            # point of `policy_trainings_skipped`); a timeout is a training that
            # was cut off mid-flight and whose curve is truncated. Both arrive
            # here as `trained=False` and printing them the same way is how a
            # wedged node reads as a method's design.
            mark = "TIMEOUT"
        elif not r.result.trained and r.result.error:
            mark = "FAILED"
        elif not r.result.trained:
            mark = "SKIPPED"
        elif r.result.pruned:
            # Trained, but stopped early by `train.pruning`: the fitness beside
            # it is a LOWER BOUND, not a measurement of the reward's ceiling.
            mark = "pruned"
        else:
            mark = "trained"
        fit = "n/a" if r.fitness is None else f"{r.fitness:.4f}"
        rows.append(f"    {r.cand_id:<14} {mark:<16} fitness={fit}")
    return "\n".join(rows)


def execute_rate(reports: List[CandidateReport]) -> float:
    """Fraction of candidates whose reward was a runnable program.

    Eureka reports this per run.

    "Runnable" means BOTH halves: it compiled (`candidate.valid`) *and* it did
    not blow up the first time the learner called it (`not result.error`).
    Counting only the compile is a real overcount -- a reward can parse
    perfectly and then raise `FloatingPointError: reward returned nan` or
    `TypeError: compute_reward() takes 0 positional arguments` on the first
    transition. Measured on a tester-profile eureka run: 2 of 24 rewards crashed and a
    compile-only rate read 1.0 for every iteration. It is the number that answers "is
    generation working at all", so an inflated value is costly exactly where it
    is most trusted.

    A candidate that was SCREENED or whose training was deliberately SKIPPED
    (CARD's whole contribution) still counts as executed: its code was fine and
    nothing about it failed to run. Only a candidate that could not be compiled
    or could not be called counts against the rate.
    """
    if not reports:
        return 0.0
    return sum(1 for r in reports
               if r.candidate.valid and not r.result.error) / len(reports)
