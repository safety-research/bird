"""Run recording: the experiment tracker and the rollout video encoder.

Two registry families live here, both hanging off `output.*` (§0 -- the
loop-invariant block that no single stage owns).

**`tracker`** (`output.tracker`). A run already writes everything it knows to
`ctx.rundir` (`bird/artifacts.py`): the resolved config, a JSONL journal, every
candidate valid or not, and `budget.json`. A tracker is therefore never the
system of record -- it is a *second* sink for the same numbers, for watching a
run that executes somewhere else without copying its run dir. That framing
decides two things:

  * `none` is the default and must be *free*: the whole test suite and every
    `--dry-run` runs under it, so it may not import, connect, or allocate.
  * a tracker failure mid-run must never destroy the search. `start()` raises
    loudly (a misconfigured tracker should fail in the first second, not the
    sixth hour) but every subsequent call swallows its exception and warns:
    the run dir still has the data, so the correct trade is to keep training.

**`video_format`** (`output.video.format`). `output.video.enabled` defaults to
**true**, so the default encoder may not depend on a package that might not be
installed -- hence `frames`, a stdlib PNG writer (`zlib` + `struct`), as the
default. PNG frames are also exactly what a VLM consumes: §4's `vlm_score`
fitness and the `vlm` /
`llm_on_vlm_captions` preference comparators all read the artifact a human
would watch, and `bird/llm/anthropic_client.py` base64s `.png` straight into a
message. So `frames` is the primary artifact and `gif`/`mp4` are conveniences
for humans, not the other way round. When their encoder is missing they degrade
*loudly* to frames rather than losing the recording, because a missing codec is
a packaging accident and a lost rollout is lost evidence.

Interfaces this module consumes but does not own:

  * `ctx.env.render(state) -> (H, W, 3) uint8` -- OPTIONAL. Duck-typed, because
    the env adapter's contract is `bird/envs/`'s to pin, and none of the three
    toy envs implement it. A missing `render` is not an error: recording is
    skipped with a debug line and the run is otherwise identical.
  * `TrainResult.trajectories` (`bird/types.py`) -- rollout 0 is the recorded
    one, the same convention `bird/components/preferences.py::_clip_for` uses,
    and for the same reason: recording each candidate's *best* rollout would
    show a comparison of bests rather than of agents.
"""

from __future__ import annotations

import faulthandler
import hashlib
import inspect
import json
import logging
import math
import os
import struct
import sys
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from . import paths
from .judgments import LUMA_601, frame_set
from .registry import get as registry_get
from .registry import register
from .types import Candidate, CandidateReport, Trajectory
from . import multiview

log = logging.getLogger("bird.observability")

#: Frames pushed to a tracker per media event. A 300-frame rollout as 300
#: images is a slow upload of a nearly static picture; the artifact on disk
#: keeps every frame, so this cap costs nothing that matters.
_TRACKER_IMAGE_CAP = 16

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Which imageio import failures have already been reported this process.
_IMAGEIO_WARNED: set = set()


# ==========================================================================
# §0 `output.tracker` -- the tracker family
# ==========================================================================


class Tracker:
    """The tracker surface, and the `none` implementation at the same time.

    Six methods, all returning None, all safe to call in any order and any
    number of times. Callers never test the concrete type: `make_tracker`
    always returns something with all six, so `bird.py` has no `if tracker is
    not None` anywhere.

        start(ctx)                                  once, before iteration 0.
                                                    Idempotent -- `make_tracker`
                                                    already called it.
        log_iteration(ctx, state, reports, selection)
                                                    once per loop iteration,
                                                    after stage 5.
        log_candidate(ctx, candidate, report)       per candidate; `report` may
                                                    be None when the candidate
                                                    never reached stage 4.
        log_media(ctx, name, path)                  a frames dir or a gif/mp4,
                                                    whatever `record_rollouts`
                                                    returned.
        log_summary(ctx, result, budget)            once, with `bird.run`'s
                                                    return dict and the Budget.
        finish()                                    once, in a finally.

    Subclasses override what they support; anything they do not override stays
    a no-op, which is why adding a seventh method here cannot break an existing
    tracker.
    """

    def __init__(self, ctx: Any = None) -> None:  # noqa: D401 - trivial
        pass

    def start(self, ctx: Any) -> None:
        pass

    def log_iteration(self, ctx: Any, state: Any, reports: Sequence[CandidateReport],
                      selection: Any = None) -> None:
        pass

    def log_candidate(self, ctx: Any, candidate: Candidate,
                      report: Optional[CandidateReport] = None) -> None:
        pass

    def log_media(self, ctx: Any, name: str, path: Any) -> None:
        pass

    def log_summary(self, ctx: Any, result: Dict[str, Any], budget: Any = None) -> None:
        pass

    def finish(self) -> None:
        pass


@register("tracker", "none")
class NullTracker(Tracker):
    """No tracking at all: the run is entirely local (the default).

    Deliberately inherits every no-op rather than defining its own, so "the
    interface" and "the free implementation" cannot drift apart. Imports
    nothing, opens nothing, and in particular never touches wandb.
    """


def _import_wandb() -> Any:
    """Import wandb at *call* time, never at module import time.

    `registry.load_all()` imports this module for every `--dry-run`,
    `--validate-all` and `--list-configs`, exactly as it does
    `bird/components/training.py` (whose `sb3` backend imports
    stable-baselines3 the same way, for the same reason): a module-level import
    would make a machine without wandb unable to *validate* a config it never
    intended to run.
    """
    try:
        import wandb  # noqa: PLC0415 - deliberately deferred
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError(
            "output.tracker: wandb is selected but not installed. Either\n"
            "    pip install wandb\n"
            "or turn tracking off in the config with\n"
            "    output.tracker: none\n"
            "which is the default and keeps the run entirely local (the run dir "
            "under output.dir already holds the config, journal, candidates and "
            "budget).\n"
            f"(underlying import error: {exc})"
        ) from exc
    return wandb


@register("tracker", "wandb")
class WandbTracker(Tracker):
    """Weights & Biases, imported lazily inside `start`.

    What it logs, and why each one is not optional:

      * the **resolved** config as the wandb config. The resolved config is the
        run's identity (its sha256 prefix is the run ID), so logging the
        hand-written file instead would make two runs that differ by an
        inherited default indistinguishable in the UI.
      * every candidate's fitness for the iteration, as a table plus
        distribution scalars. `fitness` is `Optional[float]` and `None` is not
        `0.0`: unscored candidates are *excluded* from the statistics and
        counted separately in `fitness/n_scored`, so a CARD run (which computes
        no ranking scalar at all) shows an honest zero there rather than a
        fabricated flat line at 0.0.
      * the execute rate, which is the statistic Eureka reports per run.
      * the whole `budget.report()`, `policy_trainings_skipped` included --
        CARD's entire contribution is RL runs it did *not* launch, and a cost
        panel without that counter makes its claim invisible.

    Per-candidate fitness is a table rather than one series per `cand_id`
    because ids are minted per run (`ctx.next_id`) and never recur across
    iterations, so a series keyed by id would be a chart of one point each.
    """

    def __init__(self, ctx: Any = None) -> None:
        super().__init__(ctx)
        self._wandb: Any = None
        self._run: Any = None
        self._step = 0

    # -- credentials: degrade IN PROCESS, never in the config --

    #: What `start()` cannot authenticate without in `online` mode. Read by
    #: `make_tracker`, which falls back to `none` when one is unset rather than
    #: let `wandb.init` fail on a compute node hours from anyone -- and does so
    #: in process, so the resolved config, `Config.hash()` and therefore the run
    #: directory `loop.resume_from: auto` looks for are the same with or without
    #: the key. Doing it in the config instead (`--set output.tracker=none`
    #: whenever the key is absent from the submitting shell) would move the
    #: hash, which covers every key by design, so a requeue of the same submit
    #: line from a shell that differed only in WANDB_API_KEY would hash to a
    #: different directory and start a fresh search beside the unfinished one.
    #: The rule: key absent, no wandb. `wandb login`'s netrc is deliberately
    #: not consulted -- a rule with two sources is two rules.
    REQUIRED_ENV: Tuple[str, ...] = ("WANDB_API_KEY",)

    @classmethod
    def degradation(cls, ctx: Any) -> Optional[Dict[str, Any]]:
        """None when `start()` can authenticate, else the record `make_tracker`
        journals and writes to `status.json`: which tracker was asked for, which
        one runs instead, and why. `offline` mode needs no key and never
        degrades."""
        if (ctx.cfg.get("output.wandb.mode") or "online") == "offline":
            return None
        missing = [v for v in cls.REQUIRED_ENV if not os.environ.get(v)]
        if not missing:
            return None
        return {"from": ctx.cfg.get("output.tracker") or "wandb", "to": "none",
                "reason": f"{' and '.join(missing)} unset"}

    # -- lifecycle --

    def start(self, ctx: Any) -> None:
        if self._run is not None:
            return  # idempotent: make_tracker starts, bird.py may start again
        wandb = _import_wandb()
        cfg = ctx.cfg
        # `group` falls back to the config's name so a `loop.n_restarts` sweep
        # of one method lands in one group without anyone setting a key. This
        # is labelling, not control flow.
        group = cfg.get("output.wandb.group") or cfg.get("name") or "bird"
        run_name = ctx.rundir.path.name if getattr(ctx, "rundir", None) else str(group)
        self._wandb = wandb
        self._run = wandb.init(
            project=cfg.get("output.wandb.project") or "bird",
            entity=cfg.get("output.wandb.entity"),
            group=group,
            tags=list(cfg.get("output.wandb.tags") or []),
            mode=cfg.get("output.wandb.mode") or "online",
            name=run_name,
            dir=str(ctx.rundir.path) if getattr(ctx, "rundir", None) else None,
            config=cfg.to_dict(),
        )
        # Declare the x-axis ONCE, for every panel. Without this, wandb charts
        # against its internal log-call counter, which interleaves
        # `log_candidate` (commit=False) with `log_iteration` (commit=True) and
        # produces an axis that means nothing. `iteration` is the only x that
        # makes a cross-run overlay legible, which is the entire reason a sweep
        # is logged to one project.
        self._guard("define_metric", lambda: (
            self._run.define_metric("iteration"),
            self._run.define_metric("*", step_metric="iteration")))
        try:
            self._run.summary["config_hash"] = cfg.hash()
            self._run.summary["config_lineage"] = list(getattr(cfg, "lineage", []) or [])
        except Exception as exc:  # pragma: no cover - backend-dependent
            log.warning("wandb: could not record config provenance: %s", exc)
        if cfg.get("output.wandb.log_code"):
            # REFUSE RATHER THAN UPLOAD THE WRONG DIRECTORY.
            # `Path(__file__).parent.parent` is the repo in a checkout and
            # **site-packages** under a non-editable install -- so using it,
            # `log_code` would ship every installed package in the
            # environment to wandb, under this run's name, with no error.
            # That is worse than a wrong path: it is an upload to an external
            # service of files that are not ours to send.
            #
            # A missing checkout means we cannot honour the key, so the run
            # says so and continues. Failing the run would be the wrong
            # trade: `log_code` is provenance, and provenance may not be the
            # thing that kills a training.
            root = paths.repo_root(required=False)
            if root is None:
                log.warning(
                    "wandb: output.wandb.log_code is set but bird is not "
                    "running from a checkout, so there is no source tree to "
                    "log; skipping. (Set %s to a checkout to enable it. The "
                    "directory above the package is site-packages here, and "
                    "uploading that is not what the key asks for.)",
                    paths.DATA_ROOT_ENV)
            else:
                self._guard("log_code", lambda: self._run.log_code(str(root)))

    def finish(self) -> None:
        if self._run is None:
            return
        self._guard("finish", self._wandb.finish)
        self._run = None

    # -- logging --

    def log_iteration(self, ctx: Any, state: Any, reports: Sequence[CandidateReport],
                      selection: Any = None) -> None:
        if self._run is None:
            return
        reports = list(reports or [])
        scored = [r.fitness for r in reports if r.fitness is not None]
        payload: Dict[str, Any] = {
            "iteration": int(getattr(state, "iteration", self._step)),
            "restart": int(getattr(state, "restart", 0)),
            "candidates/n": len(reports),
            "candidates/n_valid": sum(1 for r in reports if r.candidate.valid),
            "candidates/n_screened": sum(1 for r in reports if r.candidate.screened_out),
            "candidates/n_trained": sum(1 for r in reports if r.result.trained),
            "candidates/execute_rate": _execute_rate(reports),
            "fitness/n_scored": len(scored),
        }
        if scored:
            payload["fitness/best"] = max(scored)
            payload["fitness/worst"] = min(scored)
            payload["fitness/mean"] = float(sum(scored) / len(scored))
        if selection is not None:
            winner = getattr(selection, "winner", None)
            if winner is not None:
                payload["selection/winner_fitness"] = winner.fitness
            payload["selection/tie_broken"] = bool(getattr(selection, "tie_broken", False))
        incumbent = getattr(state, "best", None)
        if incumbent is not None and incumbent.fitness is not None:
            payload["fitness/incumbent"] = incumbent.fitness
        payload.update(self._budget_payload(getattr(ctx, "budget", None)))
        try:
            payload["candidates/table"] = self._candidate_table(reports)
        except Exception as exc:  # noqa: BLE001 - a table is not worth a run
            log.warning("wandb: could not build the candidate table (%s)", exc)
        self._log(payload)

    def log_candidate(self, ctx: Any, candidate: Candidate,
                      report: Optional[CandidateReport] = None) -> None:
        if self._run is None:
            return
        self._log({
            "candidate/iteration": candidate.iteration,
            "candidate/valid": bool(candidate.valid),
            "candidate/screened_out": bool(candidate.screened_out),
            # "" means no failure; the kind keeps `invalid` and `screened`
            # distinguishable populations.
            "candidate/failure_kind": candidate.failure_kind or "",
            "candidate/fitness": None if report is None else report.fitness,
        }, commit=False)

    def log_media(self, ctx: Any, name: str, path: Any) -> None:
        """Log one recording under a panel keyed by ROLE, not by candidate id.

        The distinction is the whole value of the panel. `cand_id` is minted per
        candidate and never recurs, so a panel keyed by it holds exactly one
        image forever and a 3-iteration x 8-candidate run produces 24 dead
        panels. Keyed by role -- `video/best`, `video/worst` -- one panel
        accumulates a frame per iteration and gets a slider, which is the thing
        anyone actually wants to look at. `_candidate_table` avoids the same
        trap for fitness.

        The candidate id is not lost: it goes into the caption, so a frame is
        still traceable to `candidates/<id>/` in the run tree.

        NOTE on `output.video.format: frames` (the default): what reaches wandb
        is up to 16 sampled stills, not a playable clip. That is correct for the
        artifact's primary consumer -- a VLM takes images, and §4's comparators
        read the same PNGs -- but if you want to press play in the browser, set
        `output.video.format: gif`.
        """
        if self._run is None or not ctx.cfg.get("output.wandb.log_videos"):
            return
        p = Path(path)
        if not p.exists():
            return
        caption = f"{name} ({p.name})"
        key = f"video/{name}"
        if p.is_dir():
            frames = sorted(p.glob("*.png"))
            if not frames:
                return
            picks = [frames[i] for i in _even_indices(len(frames), _TRACKER_IMAGE_CAP)]
            self._guard("log frames", lambda: self._log(
                {key: [self._wandb.Image(str(f), caption=caption) for f in picks]},
                commit=False))
        else:
            # `format=` is explicit because wandb warns that it becomes REQUIRED
            # in v0.20.0, and a warning on every upload is one more line a reader
            # of a 20-hour log has to learn to skip. Taken from the suffix the
            # `video_format` encoder actually produced rather than from
            # `output.video.format`, which can degrade to `frames` when imageio
            # is missing (`_encode_with_imageio`) -- so the config key and the
            # file on disk are allowed to disagree and the FILE is authoritative.
            fmt = p.suffix.lstrip(".").lower() or "mp4"
            self._guard("log video", lambda: self._log(
                {key: self._wandb.Video(str(p), caption=caption, format=fmt)},
                commit=False))

    def log_summary(self, ctx: Any, result: Dict[str, Any], budget: Any = None) -> None:
        if self._run is None:
            return

        def _apply() -> None:
            for key, value in (result or {}).items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    self._run.summary[key] = value
            for key, value in self._budget_payload(budget).items():
                self._run.summary[key] = value

        self._guard("summary", _apply)

    # -- internals --

    def _candidate_table(self, reports: Sequence[CandidateReport]) -> Any:
        return self._wandb.Table(
            columns=["cand_id", "parent_id", "valid", "screened_out", "trained",
                     "failure_kind", "fitness", "fitness_source"],
            data=[[r.cand_id, r.candidate.parent_id or "", bool(r.candidate.valid),
                   bool(r.candidate.screened_out), bool(r.result.trained),
                   r.candidate.failure_kind or "", r.fitness, r.fitness_source]
                  for r in reports],
        )

    @staticmethod
    def _budget_payload(budget: Any) -> Dict[str, Any]:
        """Every counter, prefixed. `policy_trainings_skipped` rides along by
        construction rather than by an explicit list, so a counter added to
        `bird/budget.py` cannot be silently dropped from the dashboard."""
        if budget is None:
            return {}
        try:
            return {f"budget/{k}": v for k, v in budget.report().items()}
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("wandb: could not read the budget report: %s", exc)
            return {}

    def _log(self, payload: Dict[str, Any], commit: bool = True) -> None:
        self._guard("log", lambda: self._run.log(payload, commit=commit))
        if commit:
            self._step += 1

    @staticmethod
    def _guard(what: str, fn: Any) -> None:
        """A tracker outage costs a chart, not a search.

        Everything the tracker would have said is already on disk in the run
        dir, so raising here would trade recoverable data loss for
        unrecoverable compute loss. `start()` is the deliberate exception: a
        tracker that cannot even initialise is a configuration error and should
        surface in the first second.
        """
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - see docstring
            log.warning("wandb: %s failed (%s); the run dir still has the data", what, exc)


def _execute_rate(reports: Sequence[CandidateReport]) -> float:
    """Fraction of candidates whose reward was a runnable program.

    Duplicated in spirit with `bird.artifacts.execute_rate`, but computed here
    from the same definition rather than imported, because `bird/artifacts.py`
    is the artifact writer and importing it for one arithmetic line would tie
    the tracker to the on-disk layout it is explicitly a second sink for. Two
    definitions can drift, so `tests/test_artifacts.py` pins them to each other.

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


# ==========================================================================
# §0 `output.video.format` -- the video_format family
# ==========================================================================


def _even_indices(n: int, cap: int) -> List[int]:
    """`cap` indices spread over `range(n)`, or all of them if there are fewer.

    Even-stride subsampling, not truncation: the interesting part of a rollout
    is usually the end (the goal reached, or the fall), and a head-truncated
    recording would show the reset transient and nothing else. This is the same
    choice, for the same reason, that `preferences._clip_for` documents.
    """
    n = max(int(n), 0)
    cap = max(int(cap), 1)
    if n <= cap:
        return list(range(n))
    return sorted({int(round(k)) for k in np.linspace(0, n - 1, cap)})


def _states_of(traj: Trajectory) -> List[Any]:
    """`traj.states` as a list of per-step rows; `[]` when it carries none.

    Extracted rather than inlined because two callers index this list with the
    same indices and must agree on its length: `_render_frames` turns those
    rows into frames, and `save_trajectory_trace` samples the reward's own
    numbers at the same positions. If the two disagreed about how many steps an
    episode had, every frame would be paired with the wrong instant and the
    disagreement the trace exists to show would be manufactured by the reader.
    """
    states = traj.states
    if states is None:
        return []
    try:
        return list(np.asarray(states, dtype=float))
    except (TypeError, ValueError):
        try:
            return list(states)
        except TypeError:
            return []


def _as_rgb8(frame: Any) -> np.ndarray:
    """Coerce whatever `env.render` returned into contiguous (H, W, 3) uint8.

    Accepts grayscale (H, W), RGB, and RGBA (alpha dropped -- PNG colour type 2
    has no alpha channel and a rollout frame has nothing to be transparent
    against), and floats, which are treated as [0, 1] and scaled. Being liberal
    here is what keeps `render` duck-typed: an env author should not have to
    read this file to know what dtype to return.
    """
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"render() returned an array of shape {arr.shape}; expected (H, W[, C])")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] >= 4:
        arr = arr[..., :3]
    elif arr.shape[-1] != 3:
        raise ValueError(f"render() returned {arr.shape[-1]} channels; expected 1, 3 or 4")
    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=float)
        if np.nanmax(np.abs(arr), initial=0.0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.nan_to_num(arr), 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _resize_nearest(frame: np.ndarray, width: int) -> np.ndarray:
    """Nearest-neighbour rescale to `output.video.width`, preserving aspect.

    Nearest rather than bilinear on purpose, and not only because bilinear
    would want scipy: these renders are diagrammatic (a grid cell, an arm, a
    target dot), and interpolating them blurs the exact thing a VLM is being
    asked to read.
    """
    h, w = frame.shape[:2]
    width = int(width)
    if width <= 0 or w == 0 or h == 0 or width == w:
        return frame
    height = max(1, int(round(h * width / w)))
    ys = np.minimum((np.arange(height) * h // height), h - 1)
    xs = np.minimum((np.arange(width) * w // width), w - 1)
    return np.ascontiguousarray(frame[ys][:, xs], dtype=np.uint8)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def encode_png(frame: np.ndarray) -> bytes:
    """Encode (H, W, 3) uint8 as a PNG, using only `zlib` and `struct`.

    The whole reason `output.video.enabled` can default to true. The format is
    the minimum legal file: signature, IHDR (8-bit, colour type 2 = truecolour,
    no interlace), one IDAT holding zlib-compressed scanlines each prefixed by
    filter type 0, and IEND. No adaptive filtering -- it would shrink the file
    and cost readability, and these frames are small and few.
    """
    rgb = _as_rgb8(frame)
    h, w = rgb.shape[:2]
    raw = bytearray()
    for row in rgb:
        raw.append(0)  # PNG filter type 0 (None) for this scanline
        raw += row.tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"".join((
        _PNG_SIGNATURE,
        _png_chunk(b"IHDR", ihdr),
        _png_chunk(b"IDAT", zlib.compress(bytes(raw), 6)),
        _png_chunk(b"IEND", b""),
    ))


@register("video_format", "frames")
def frames_format(frames: Sequence[Any], dest_dir: Path, fps: int = 20) -> Path:
    """Write `frame_0000.png ...` into `dest_dir` and return the directory.

    Stdlib only, which is the point: this is the default format and recording
    is on by default, so the default path may not depend on an encoder. The
    frames are also the VLM-facing artifact (§4), so this is the primary output
    and `gif`/`mp4` are humans' convenience wrappers over the same pixels.

    `fps` is not dropped -- it goes into `meta.json` beside the frames, so a
    later encoder (or a reader asking "how fast was this?") does not have to
    guess the playback rate the config asked for.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, frame in enumerate(frames):
        (dest_dir / f"frame_{i:04d}.png").write_bytes(encode_png(frame))
        written += 1
    (dest_dir / "meta.json").write_text(json.dumps(
        {"format": "frames", "n_frames": written, "fps": int(fps)}, indent=2))
    return dest_dir


def _imageio(need_ffmpeg: bool = False) -> Optional[Any]:
    """Import imageio lazily; return None (never raise) if it is unavailable.

    Unlike wandb, a missing encoder is not a configuration error worth aborting
    a run over: the caller falls back to `frames` and keeps the recording.
    """
    try:
        try:
            import imageio.v2 as imageio  # noqa: PLC0415 - deliberately deferred
        except ImportError:
            import imageio  # type: ignore[no-redef]  # noqa: PLC0415
        if need_ffmpeg:
            import imageio_ffmpeg  # noqa: F401,PLC0415
    except ImportError as exc:
        # Loud, but once: the fallback happens per *candidate*, and a warning
        # repeated ten times an iteration is a warning nobody reads.
        emit = log.warning if need_ffmpeg not in _IMAGEIO_WARNED else log.debug
        _IMAGEIO_WARNED.add(need_ffmpeg)
        emit("output.video.format needs imageio%s, which is not installed "
             "(%s). Install it with\n"
             "    pip install imageio%s\n"
             "Falling back to output.video.format: frames -- the recording is "
             "kept as PNGs, only the container is lost.",
             " and imageio-ffmpeg" if need_ffmpeg else "", exc,
             "[ffmpeg]" if need_ffmpeg else "")
        return None
    return imageio


def _encode_with_imageio(frames: Sequence[Any], dest_dir: Path, fps: int,
                         suffix: str, need_ffmpeg: bool) -> Path:
    """Shared body of `gif` and `mp4`: try imageio, else degrade to frames."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    imageio = _imageio(need_ffmpeg=need_ffmpeg)
    if imageio is None:
        return frames_format(frames, dest_dir, fps)

    rgb = [_as_rgb8(f) for f in frames]
    out = dest_dir / f"rollout{suffix}"
    try:
        try:
            imageio.mimwrite(str(out), rgb, fps=int(fps))
        except TypeError:
            # imageio's GIF plugin took `duration` (seconds/frame) before it
            # took `fps`, and takes milliseconds after; seconds is the form
            # both old and new accept.
            imageio.mimwrite(str(out), rgb, duration=1.0 / max(int(fps), 1))
    except Exception as exc:  # noqa: BLE001 - e.g. the ffmpeg binary is absent
        log.warning("output.video.format=%s failed to encode (%s); writing frames instead",
                    suffix.lstrip("."), exc)
        return frames_format(frames, dest_dir, fps)
    return out


@register("video_format", "gif")
def gif_format(frames: Sequence[Any], dest_dir: Path, fps: int = 20) -> Path:
    """`rollout.gif` via imageio; degrades to PNG frames if imageio is absent."""
    return _encode_with_imageio(frames, dest_dir, fps, ".gif", need_ffmpeg=False)


@register("video_format", "mp4")
def mp4_format(frames: Sequence[Any], dest_dir: Path, fps: int = 20) -> Path:
    """`rollout.mp4` via imageio + imageio-ffmpeg; degrades to PNG frames.

    Also degrades when imageio imports but the ffmpeg *binary* it shells out to
    is missing, which is a distinct failure that raises at write time rather
    than at import time.
    """
    return _encode_with_imageio(frames, dest_dir, fps, ".mp4", need_ffmpeg=True)


#: Frames sampled for the distinct-colour count in `clip_stats`. Counting over
#: every pixel of every frame is O(all pixels) for a statistic whose useful
#: range is "hundreds vs tens of thousands"; the even stride keeps the sample
#: describing the whole clip rather than its head (`_even_indices`'s argument).
_STATS_COLOUR_FRAMES = 8


def clip_stats(frames: Sequence[Any]) -> Dict[str, Any]:
    """Measure one recording at write time, while the pixels are in memory.

    A headless renderer can silently produce black frames: the run finishes
    `status: ok`, the journal says `n_frames: 300`, the files are on disk,
    the render watchdog never fires -- and the candidates have already been
    VLM-scored off frames that show nothing. Without these statistics nothing
    in the artifact distinguishes a good clip from a black one, so nothing
    downstream can.

    MEASUREMENTS, NEVER VERDICTS. A fixed detector cannot work: near-black compressed
    video still yields ~1,480 distinct colours, so a colour-count floor used
    as a conjunct missed the real case -- and a reacher clip is legitimately
    darker and smaller than a half-cheetah one, so no absolute threshold
    generalises across environments. Comparison must be within-environment,
    and with `output.video.record: all` a run's own candidates are same-env
    siblings, so this records the numbers per clip and leaves the comparison
    to the reader -- the same split `budget.blind_comparisons` uses: the run
    states the fact, the consumer draws the conclusion.

    Luma is Rec. 601 over the uint8 frame -- the weights are
    `judgments.LUMA_601`, imported rather than restated so the recording-time
    measurement and the judge-trace one cannot drift into two different
    numbers wearing the same name. 0 is black, 255 is white. The per-frame
    series is kept whole -- capped by
    `output.video.max_frames`, it is a few KB -- because a clip that goes
    black halfway has the same mean as one dim throughout, and the
    difference is the diagnosis. Distinct colours are counted over an
    even-strided sample of `_STATS_COLOUR_FRAMES` frames; the count
    saturates quickly and the stride keeps it honest about the whole clip.
    """
    lumas: List[float] = []
    for f in frames:
        # `_as_rgb8` first: the recording path hands this function frames it
        # has already normalised, but grayscale, RGBA and float frames are
        # valid render outputs everywhere else in the pipeline, and a
        # measurement must coerce with the encoders' own rule rather than
        # assume (H, W, 3) uint8 -- on an already-normalised frame it is a
        # cheap passthrough. Then per-channel means and the luma weights: the
        # mean is linear, so this is the same number as weighting every pixel
        # first, without materialising an H x W x 3 float64 copy of each
        # frame in the recording hot path. numpy accumulates a uint8 mean in
        # float64 on its own.
        r, g, b = _as_rgb8(f).mean(axis=(0, 1), dtype=np.float64)
        lumas.append(round(float(r * LUMA_601[0] + g * LUMA_601[1]
                                 + b * LUMA_601[2]), 3))
    picks = [_as_rgb8(frames[i]).reshape(-1, 3)
             for i in _even_indices(len(frames), _STATS_COLOUR_FRAMES)]
    if picks:
        # ONE uint32 per pixel rather than `np.unique(..., axis=0)` over an
        # (N, 3) array: the row-wise form sorts by structured dtype and is
        # markedly more expensive in both CPU and peak memory, which at
        # 480x480 x 8 frames is a spike in the recording hot path for a
        # statistic whose useful range is "hundreds vs tens of thousands".
        # A 24-bit code is injective over
        # RGB triples, so the COUNT is identical by construction -- pinned by
        # a test whose colours collide under any lossier packing.
        px = np.concatenate(picks, axis=0).astype(np.uint32)
        codes = (px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]
        distinct = int(np.unique(codes).size)
    else:
        distinct = 0
    h, w = (np.asarray(frames[0]).shape[:2] if len(frames) else (0, 0))
    return {
        "schema": 1,
        "n_frames": len(frames),
        "width": int(w),
        "height": int(h),
        "luma_mean": round(float(np.mean(lumas)), 3) if lumas else None,
        "luma_min_frame": min(lumas) if lumas else None,
        "luma_max_frame": max(lumas) if lumas else None,
        "luma_per_frame": lumas,
        "distinct_colours": distinct,
        "distinct_colours_frames_sampled": len(picks),
    }


# ==========================================================================
# Plumbing called from bird.py
# ==========================================================================


def make_tracker(ctx: Any) -> Tracker:
    """Construct (and start) the tracker named by `output.tracker`.

    Returns a started tracker so no caller has to remember the two-step, and
    always returns an object -- `none` resolves to `NullTracker`, so `bird.py`
    never guards a call site.

    A tracker class may declare `degradation(ctx)` (see `WandbTracker`): a
    record naming the tracker to run INSTEAD and why, or None. When it returns
    one, that tracker is constructed, the swap is logged as a warning, and the
    record is written where the artifact will be read -- a `tracker_degraded`
    journal event and `status.json.tracker_degraded` -- so a run with no
    dashboard says so in its own directory. The config is not touched: the
    degradation is a fact about this process's environment, and putting it in
    the config would move `Config.hash()` and, with it, the run directory a
    resumed leg looks for.
    """
    name = ctx.cfg.get("output.tracker") or "none"
    cls = registry_get("tracker", name)
    degrade = getattr(cls, "degradation", None)
    record = degrade(ctx) if degrade is not None else None
    if record is not None:
        log.warning("output.tracker: %s -> %s (%s); the run dir stays the record and the "
                    "config hash is unchanged", record["from"], record["to"], record["reason"])
        ctx.event("tracker_degraded", **record)
        rundir = getattr(ctx, "rundir", None)
        if rundir is not None and hasattr(rundir, "record_tracker_degraded"):
            rundir.record_tracker_degraded(record)
        cls = registry_get("tracker", record["to"])
    tracker = cls(ctx)
    tracker.start(ctx)
    return tracker


def _recordable(reports: Sequence[CandidateReport]) -> List[CandidateReport]:
    """Candidates with something to show: trained, with at least one rollout."""
    return [r for r in (reports or []) if r.result.trained and r.result.trajectories]


def select_for_recording(mode: str, reports: Sequence[CandidateReport]) -> List[CandidateReport]:
    """Which candidates `output.video.record` asks us to record.

    `best` and `best_and_worst` rank on `fitness`, and a candidate whose
    fitness is `None` is *not a candidate for "best"*: `None` is distinct from
    `0.0` everywhere in this codebase, and treating an unranked candidate as a
    zero would silently make it the worst. Where nothing is scored at all --
    CARD computes no ranking scalar -- there is no best and no worst, so
    nothing is recorded, and the config that wants footage anyway should say
    `output.video.record: all`.
    """
    pool = _recordable(reports)
    if mode == "all":
        return pool
    scored = [r for r in pool if r.fitness is not None]
    if not scored:
        if pool:
            log.debug("output.video.record=%s ranks on fitness and no candidate has one; "
                      "use output.video.record: all to record an unranked population", mode)
        return []
    best = max(scored, key=lambda r: r.fitness)
    if mode == "best":
        return [best]
    if mode == "best_and_worst":
        worst = min(scored, key=lambda r: r.fitness)
        if worst is best and len(scored) > 1:
            # A FLAT COLUMN still records TWO clips.
            #
            # `max` and `min` both return the FIRST extreme, so they return the
            # same object exactly when every scored fitness is equal -- and
            # collapsing the pair to a single recording there would be
            # backwards: a column where every candidate scores the same is the
            # case where the SCALAR has stopped discriminating, so footage of
            # two different agents is the only evidence left about whether they
            # actually behave differently. It is not hypothetical -- RDA on
            # Pendulum returned nine candidates all scoring exactly 1.000.
            #
            # The last candidate rather than a random one: `scored` is in
            # candidate order, which is stable across
            # `train.candidate_parallelism` (reassembly is by index, never by
            # completion order), so `best_and_worst` stays deterministic and
            # `tests/test_parallelism.py`'s byte-identical run dirs still hold.
            worst = scored[-1]
        return [best] if worst is best else [best, worst]
    return []


def _accepts_a_state(render: Any) -> bool:
    """Does `render` take one positional argument?

    Asked by introspection, up front, rather than inferred from a `TypeError`
    raised by the call: `render(state)` may itself raise `TypeError` for a
    perfectly good reason (a state the env cannot interpret), and blaming that
    on the signature sends whoever reads the log to the wrong file. Callables
    whose signature cannot be read at all (C builtins, some `functools`
    wrappers) get the benefit of the doubt -- the call itself is still guarded.
    """
    try:
        inspect.signature(render).bind(object())
    except TypeError:
        return False
    except Exception:  # noqa: BLE001 - unreadable signature; let the call decide
        return True
    return True


@dataclass
class _RenderClock:
    """Wall clock for ONE `record_rollouts` call, shared across every pick.

    Per call rather than per candidate: `output.video.record: all` renders
    `generate.n_candidates` rollouts, and a per-candidate cap would silently
    multiply `output.video.timeout_s` by that number -- the operator asked how
    long recording may take, not how long one candidate may take.

    `frames` is tallied here because an expired call returns `None` and throws
    its partial frames away, so the caller has no other way to say in the
    artifact how far the renderer actually got before it stopped answering.
    """

    expires_at: Optional[float] = None
    frames: int = 0
    expired: bool = False

    def out_of_time(self) -> bool:
        if self.expires_at is None:
            return False
        if time.monotonic() >= self.expires_at:
            self.expired = True
        return self.expired


def _render_frames(render: Any, traj: Trajectory, max_frames: int, width: int,
                   warned: Optional[set] = None,
                   clock: Optional[_RenderClock] = None
                   ) -> Optional[Tuple[List[np.ndarray], List[int]]]:
    """Replay a stored trajectory through `env.render`, capped and rescaled.

    The env contract this assumes is `render(state) -> (H, W, 3)`: rendering a
    *stored* state, not "whatever the env is showing now", because the
    trajectory being recorded was produced during training and the env has long
    since moved on. An env whose `render` takes no state cannot honour that, so
    it is skipped with a warning rather than silently recording the wrong
    episode.

    Returns `(frames, steps)`, where `steps` is the episode index each frame
    was rendered from. The stride is handed back rather than left to be
    recomputed: `save_trajectory_trace` samples at exactly these indices, and
    two call sites computing `_even_indices` separately would agree only by
    coincidence.

    Three answers, and the distinction between the last two is load-bearing:

      * `(frames, steps)` -- rendered.
      * `([], [])` -- *this trajectory* could not be rendered: skip the
        candidate and carry on. An empty render has an empty stride.
      * `None` -- the RENDERER is unusable for every trajectory, so stop;
        otherwise every candidate repeats the same warning.

    A raise out of `render(state)` is the first kind, not the second, and the
    difference is the whole point of the distinction. On `bird.envs.metaworld` a
    stored trajectory whose snapshots the LRU has since evicted raises
    `UnknownStateError` -- a property of *that* trajectory's age, not of the
    renderer: the candidate trained most recently renders fine. Treating it as a
    dead renderer would let one evicted rollout cost the whole iteration's
    video (measured at `generate.n_candidates: 2` on every training backend). `warned`
    is a per-`record_rollouts`-call set that keeps the warning to one line per
    iteration when several candidates fail the same way.

    `clock` expires the whole call, not this trajectory, so a spent clock is
    the second kind and returns `None`: the next pick would be rendered by the
    same slow renderer against a deadline that has already passed. It is
    checked BETWEEN frames, which bounds "every frame returns, far too slowly
    to ever finish" and bounds nothing else -- a `render` that never returns is
    the case this guard exists for and no Python-level check can see
    it (`record_rollouts` arms a faulthandler watchdog for that half).
    """
    rows = _states_of(traj)
    if not rows:
        return [], []

    if not _accepts_a_state(render):
        log.warning("env.render does not accept a state; no rollout recorded. The expected "
                    "signature is render(state) -> (H, W, 3) uint8, because the trajectory "
                    "being recorded finished during training and the env has moved on.")
        return None

    # The stride is computed ONCE, here, and handed back with the frames.
    # Recomputing it in `record_rollouts` from the same expression would agree
    # only by coincidence, not by guarantee. The trace's whole claim is that a frame and
    # a trace point are the same instant, and a cursor wrong by one stride is
    # indistinguishable on screen from a reward diverging late, so the mapping
    # travels with the thing it maps.
    steps = _even_indices(len(rows), max_frames)
    frames: List[np.ndarray] = []
    for i in steps:
        if clock is not None and clock.out_of_time():
            _warn_once(warned, "render-deadline",
                       "output.video.timeout_s elapsed after %d frame(s); rollout "
                       "recording abandoned and the search continues without it",
                       clock.frames)
            return None
        try:
            frame = render(rows[i])
        except Exception as exc:  # noqa: BLE001 - a broken renderer is not a broken run
            _warn_once(warned, "render-raised",
                       "env.render(state) raised %s: %s; this candidate's rollout is not "
                       "recorded (other candidates are still attempted)",
                       type(exc).__name__, exc)
            return [], []
        try:
            frames.append(_resize_nearest(_as_rgb8(frame), width))
            if clock is not None:
                clock.frames += 1
        except (ValueError, TypeError) as exc:
            log.warning("env.render returned something unrenderable (%s); "
                        "no rollout recorded", exc)
            return None
    # Every index produced a frame, or we returned above: `frames` and `steps`
    # are the same length by construction, which is what lets the caller pair
    # them without a slice.
    return frames, steps


def _warn_once(warned: Optional[set], key: str, msg: str, *args: Any) -> None:
    """`log.warning` the first time this key is seen in a call, `log.debug` after.

    Recording runs over every pick in an iteration, so an env-wide condition
    would otherwise print N identical paragraphs; the count is what an operator
    needs, not the repetition."""
    if warned is None or key not in warned:
        if warned is not None:
            warned.add(key)
        log.warning(msg, *args)
    else:
        log.debug(msg, *args)


#: How long past `output.video.timeout_s` the faulthandler watchdog waits before
#: dumping. Non-zero so a dump *means* something: the between-frames check has
#: already had a frame boundary's worth of chances to abandon recording cleanly,
#: so a dump says recording did not come back out of whatever it was in.
#:
#: WHICH call it was in is the traceback's job to say, not this constant's. The
#: armed window covers the whole of `record_rollouts`, and the deadline is
#: checked only between frames, so the two other unbounded things inside it can
#: dump here too: the in-flight `render` call, and `encode(...)` writing the
#: artifact (300 PNGs onto the shared mount, or a pipe to imageio-ffmpeg). Read
#: the frame the dump names before concluding it was GL.
_WATCHDOG_GRACE_S = 30.0


def _arm_stack_dump(timeout_s: float) -> bool:
    """Ask faulthandler to dump every thread's stack in `timeout_s` seconds.

    This is the half of the guard that diagnoses a wedged renderer: the log
    stops one line before `record_rollouts`, nothing is written for many
    minutes, and the scheduler eventually tears the job down with no traceback
    and `status.json` still reading `running`. The main thread is blocked
    inside MuJoCo's EGL call, where no Python bytecode runs, so the deadline in
    `_render_frames` cannot fire -- but faulthandler's timer lives in a
    separate C thread and writes straight to the stderr fd, so it dumps
    anyway.

    It DIAGNOSES, it does not rescue: `exit=False`, because an uninterruptible
    call is not something Python can free, and killing the process would throw
    away an iteration's completed training to save a video. What it adds is
    that the log names the call -- whichever call it turns out to be: the dump
    reports the frame the main thread is actually in, which is usually the
    renderer but can equally be the encoder or
    the write of the frames themselves (see `_WATCHDOG_GRACE_S`).

    Returns whether the timer was armed; a no-op here must never be an error,
    since `redirect_stdout`-style plumbing (and pytest's capture) can leave
    `sys.stderr` as an object with no real fd, which faulthandler requires.
    That degradation is silent by design and therefore pinned by a test
    (`tests/test_render_timeout.py`): unnoticed, it would remove the only
    diagnostic the wedged run produces.
    """
    try:
        sys.stderr.fileno()
    except Exception:  # noqa: BLE001 - captured/replaced stderr: degrade, do not raise
        log.debug("stderr has no fileno; the render stack-dump watchdog is off")
        return False
    try:
        faulthandler.dump_traceback_later(timeout_s, exit=False)
    except Exception as exc:  # noqa: BLE001 - platform/threading refusal is not fatal
        log.debug("could not arm the render stack-dump watchdog: %s", exc)
        return False
    return True


@contextmanager
def render_watchdog(timeout_s: Optional[float]):
    """Arm `_arm_stack_dump` for the duration of a block; `None` is the off switch.

    A context manager because the cancel is not optional and not local to
    `record_rollouts`: the faulthandler timer is process-global and there is
    exactly ONE, so a live one left behind by a finished block dumps every
    thread's stack in the middle of the next iteration's training.

    Exported because rendering does not only happen in `record_rollouts`.
    `generation._images_for` calls `env.task_images()` during stage 1 for
    `problem.instruction_modality: text+image`, and for RDA on Meta-World that
    is where the GL context is first created -- i.e. the likeliest place in the
    whole run to wedge, with the same signature (no output, no traceback,
    `status.json` still `running`).
    """
    armed = _arm_stack_dump(float(timeout_s)) if timeout_s is not None else False
    try:
        yield armed
    finally:
        if armed:
            faulthandler.cancel_dump_traceback_later()


class Recording(NamedTuple):
    """One written rollout.

    Three fields because recording and uploading are different decisions:

    * `role` is the DISK role and is stable across iterations under `best` /
      `best_and_worst`, but is `cand0..candN` under `record: all` -- there is no
      "best" when everything is kept.
    * `cand_id` is minted per candidate and never recurs. Useful as a caption
      and as the path on disk; useless as a panel key.
    * `upload_role` is the WANDB role, computed independently from
      `output.wandb.video_record`, and is always `best`/`worst` (or `candN` when
      uploading everything) regardless of the disk role. `None` means
      disk-only. Keeping it separate is what lets `record: all` write every
      rollout while the dashboard still shows two panels with a slider instead
      of N panels holding one image each.
    """

    role: str
    cand_id: str
    path: Path
    upload_role: Optional[str] = None


def _sample_series(values: Any, idx: Sequence[int]) -> List[Optional[float]]:
    """`values[i]` for each `i` in `idx`, as plain floats, `None` where absent.

    Absent means: past the end of the series, `None` in it (a step where the
    reward claimed nothing -- `_rollout` packs component values per step, so a
    component the reward sets only on some steps reads `None` on the others),
    or not a finite number (JSON has no NaN, and a strict JSON parser would
    refuse the whole trace). A short or gapped series must
    stay absent *at the right indices*: padding it would draw a line that
    agrees with the measurement exactly where there was no claim, which is the
    one shape this artifact must never render.
    """
    try:
        seq = list(values)
    except TypeError:
        return [None] * len(idx)
    out: List[Optional[float]] = []
    for i in idx:
        if 0 <= i < len(seq) and seq[i] is not None:
            try:
                v = float(seq[i])
            except (TypeError, ValueError):
                v = float("nan")
            if math.isfinite(v):
                out.append(v)
                continue
        out.append(None)
    return out


def save_trajectory_trace(root: Path, cand_id: str, traj: Trajectory,
                          frame_steps: Sequence[int], *, mode: str,
                          max_steps: int) -> Optional[Path]:
    """Write one episode's claimed-vs-measured trace; return the path, or None.

    The artifact `save_train_result` deliberately does not write. It drops
    `TrainResult.trajectories` because they are large and "already summarised by
    the curves" -- true of everything except this one question. A reward that
    inflates its own component values agrees with itself perfectly in
    `component_traces`, because that is a mean over the very numbers being
    inflated. Only the per-step pair disagrees: what the reward CLAIMED it was
    paying (`component_values`) beside what the harness MEASURED it paid
    (`rewards`).

    **Alignment is recorded, never recomputed.** Frames are strided across the
    episode by `_even_indices`, not truncated, so frame 150 of 300 is step 250
    of a 500-step episode and not step 150. Each sample therefore carries both
    its episode `step` and the `frame` it belongs to, so a reader that lines
    frames up with steps reads a mapping instead of re-deriving one -- a
    re-derivation would silently go wrong the day `max_frames` differs from
    what it was at write time, and an alignment wrong by a stride looks
    exactly like a reward diverging late in the episode.

    Samples land on a SUBSET of the frame indices, so every trace point has a
    frame behind it. `max_steps` below the frame count thins that subset evenly
    rather than clipping its tail, for the reason `_even_indices` gives.
    """
    if mode == "none":
        return None
    rows = _states_of(traj)
    if not rows or not frame_steps:
        return None

    # A subset of the frame indices, never an independent sampling: a trace
    # point with no frame behind it has nothing to align to.
    keep = _even_indices(len(frame_steps), max_steps)
    steps = [int(frame_steps[k]) for k in keep]

    payload: Dict[str, Any] = {
        "schema": 1,
        "cand_id": cand_id,
        # Rollout 0, matching `record_rollouts` and `preferences._clip_for`:
        # the frames on disk are this episode, so the trace must be too.
        "episode": 0,
        "length": int(getattr(traj, "length", 0) or 0),
        "return": float(getattr(traj, "ret", 0.0) or 0.0),
        "success": bool(getattr(traj, "success", False)),
        "n_states": len(rows),
        "n_frames": len(frame_steps),
        "samples": {"step": steps, "frame": [int(k) for k in keep]},
        # What the harness paid out, per step.
        "measured": {"reward": _sample_series(getattr(traj, "rewards", ()) or (), steps)},
        # What the reward said it was paying, per step, per component.
        "claimed": {
            str(name): _sample_series(series, steps)
            for name, series in (getattr(traj, "component_values", {}) or {}).items()
        },
    }
    if mode == "components+states":
        # `None` for an out-of-range index, never a filter. Dropping the entry
        # shifts every later one, so `states[j]` would stop describing
        # `samples.step[j]` with nothing in the payload saying so -- the
        # mirror image of the padding `_sample_series` refuses, and the same
        # damage. Unreachable while `steps` comes from `frame_steps`, which is
        # why it is a null rather than a raise: the branch exists, so it should
        # fail the way the rest of the payload fails.
        payload["states"] = [
            ([float(x) for x in np.atleast_1d(np.asarray(rows[i], dtype=float)).ravel()]
             if 0 <= i < len(rows) else None)
            for i in steps
        ]

    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{cand_id}.json"
    dest.write_text(json.dumps(payload))
    return dest


class _ViewPlan(NamedTuple):
    """What `record_rollouts` and `sample_frames` render with -- one decision, two callers.

    `render` is the env's own `render` on the single-view path (`output.video.n_views:
    1`, every published config) and a closure composing panels otherwise; `width` is
    the pixel width `_render_frames` should treat as final (the composite is already
    `layout["width"]` wide, so its resize is a no-op); `layout` is `multiview.layout`
    or None; `note` is a shortfall the caller journals, or `""`.
    """

    render: Optional[Any]
    width: int
    layout: Optional[Dict[str, Any]]
    note: str


def _view_plan(ctx: Any) -> _ViewPlan:
    """Resolve `output.video.n_views` against what the env can actually show.

    THE PRIMARY PANEL IS THE SINGLE-VIEW FRAME, PIXEL FOR PIXEL. Each panel is
    `_as_rgb8` + `_resize_nearest(width)`, exactly what `_render_frames` does to a
    single-view frame, and the primary is drawn first at x=0. So `n_views: 2` adds a
    panel to the right of the frame every published config records and moves nothing
    in it -- which is what makes an ablation over the key read as "the judge was
    additionally shown X" rather than "the judge was shown something else".
    `tests/test_multi_view.py` holds the two byte-equal.
    """
    env = getattr(ctx, "env", None)
    cfg = ctx.cfg
    render = getattr(env, "render", None)
    width = int(cfg.get("output.video.width") or 320)
    if not callable(render):
        return _ViewPlan(None, width, None, "")
    views, note = multiview.select(env, cfg)
    if len(views) <= 1:
        return _ViewPlan(render, width, None, note)
    render_view = env.render_view
    extras = views[1:]
    lay = multiview.layout(views, width, requested_n=multiview.requested(cfg),
                           available=1 + len(getattr(env, "extra_views", ()) or ()))

    def composed(state: Any) -> np.ndarray:
        panels = [_resize_nearest(_as_rgb8(render(state)), width)]
        for view in extras:
            panels.append(_resize_nearest(_as_rgb8(render_view(state, view)), width))
        return multiview.compose(panels, lay["gutter_px"])

    return _ViewPlan(composed, int(lay["width"]), lay, note)


def record_rollouts(ctx: Any, reports: Sequence[CandidateReport]) -> List[Recording]:
    """Render and write this iteration's rollouts, then release the states.

    A thin wrapper so the release cannot be skipped. `components.training`
    asked the env adapter to RETAIN rollout 0 of every candidate the moment it
    existed (`EnvAdapter.retain_states` -- on `MetaWorld` a rollout is otherwise
    evicted from the obs->snapshot LRU long before this function runs), and this
    is the point at which those rows have served their purpose. Releasing here
    and not at the top of the next iteration is what keeps the retained store at
    ONE iteration's rollouts: the `finally` covers the disabled-video and
    no-rundir early returns, a raise out of the encoder, and the `record_timeout`
    path, all of which otherwise leak a claim per iteration.
    """
    try:
        return _record_rollouts(ctx, reports)
    finally:
        release = getattr(getattr(ctx, "env", None), "release_states", None)
        if callable(release):
            # A retained row is an optimisation, never a correctness property:
            # failing to give it back must not fail the iteration.
            try:
                release()
            except Exception:  # noqa: BLE001
                log.debug("release_states failed", exc_info=True)


def _record_rollouts(ctx: Any, reports: Sequence[CandidateReport]) -> List[Recording]:
    """Render and write this iteration's rollouts; return what was written.

    Honours the whole `output.video.*` block: `enabled`, `record` (which
    candidates), `format` (the encoder, a registry lookup), `fps`, `max_frames`
    and `width`. Output lands in `<rundir>/videos/<cand_id>/`.

    Each recorded trajectory has its `video_path` set, which is what makes the
    recording *usable* rather than merely archived: `evaluate.artifacts:
    [videos]` feedback and the preference comparators read
    `Trajectory.video_path` to tell a judge where the frames are.

    Three silent no-ops, all of them normal rather than exceptional:
    recording disabled, no run dir (`--dry-run`), and an env without `render`
    -- the three toy envs have none, so the entire test suite takes that path.

    `output.video.timeout_s` bounds the whole call by two different mechanisms
    because they catch two different failures. The `_RenderClock` deadline
    catches a renderer that still returns frames but far too slowly to finish,
    and abandons recording so the search keeps going. It cannot catch a
    renderer that never returns -- a Python-level check only runs between
    frames -- so a faulthandler watchdog covers that case, and covers it by
    printing every thread's stack rather than by freeing anything. The watchdog
    is armed over the whole call, so it also covers the two other places this
    function can block indefinitely: the in-flight render and `encode`.
    """
    cfg = ctx.cfg
    if not cfg.get("output.video.enabled", True):
        return []
    mode = cfg.get("output.video.record") or "none"
    if mode == "none":
        return []
    if getattr(ctx, "rundir", None) is None:
        log.debug("no run dir (dry run); rollout recording skipped")
        return []
    plan = _view_plan(ctx)
    render, width = plan.render, plan.width
    if not callable(render):
        log.debug("%s has no render(); rollout recording skipped",
                  type(ctx.env).__name__)
        return []

    picks = select_for_recording(mode, reports)
    if not picks:
        return []

    encode = registry_get("video_format", cfg.get("output.video.format") or "frames")
    fps = int(cfg.get("output.video.fps") or 20)
    max_frames = int(cfg.get("output.video.max_frames") or 300)
    root = Path(ctx.rundir.path) / "videos"
    # Once per call, not per candidate, and only when the key asked for more than
    # one view: a run that asked for three panels and could draw one must read as
    # that in the journal, not as a run that never asked.
    if multiview.requested(cfg) > 1:
        ctx.event("views", requested=multiview.requested(cfg),
                  n_views=multiview.panel_count(plan.layout),
                  panels=[p["name"] for p in (plan.layout or {}).get("panels", [])]
                  or [getattr(getattr(ctx.env, "primary_view", None), "name", "primary")],
                  note=plan.note)
        if plan.note:
            log.warning("%s", plan.note)
    # Read here rather than inside the loop: a trace is written per recorded
    # candidate, but whether to write one at all is a property of the run.
    trace_mode = cfg.get("output.trajectory_trace.record") or "none"
    trace_max = int(cfg.get("output.trajectory_trace.max_steps") or 300)
    trace_root = Path(ctx.rundir.path) / "traces"

    # `select_for_recording` returns best-first, then worst; naming the roles
    # here is what lets the tracker key a panel by role rather than by an id
    # that never recurs. `all` has no roles to speak of, so candidates are
    # numbered -- still stable across iterations, which is what matters.
    if mode == "all":
        roles = [f"cand{i}" for i in range(len(picks))]
    else:
        roles = ["best", "worst"][:len(picks)]

    # The upload selection is computed FROM THE SAME REPORTS but under its own
    # mode, not filtered out of `picks` by role name -- under `record: all` the
    # disk roles are `candN` and carry no best/worst information at all. Keyed by
    # `id()` because CandidateReport is not hashable and two candidates can
    # legitimately tie on every field.
    upload_mode = cfg.get("output.wandb.video_record") or "none"
    upload_picks = select_for_recording(upload_mode, reports) if upload_mode != "none" else []
    upload_roles = (["best", "worst"] if upload_mode != "all"
                    else [f"cand{i}" for i in range(len(upload_picks))])
    upload_by_id = {id(r): role for role, r in zip(upload_roles, upload_picks)}

    # `null` and only `null` disables the guard -- tested for identity rather
    # than truthiness so a `0` reads as the cap it literally is.
    timeout_s = cfg.get("output.video.timeout_s")
    capped = timeout_s is not None
    started = time.monotonic()
    clock = _RenderClock(expires_at=started + float(timeout_s) if capped else None)

    written: List[Recording] = []
    warned: set = set()
    skipped = 0
    # `render_watchdog` cancels on the way out unconditionally, including on the
    # `return written` below: the timer is process-global and a single one, so a
    # live one from a finished call would dump during the next iteration's
    # training.
    with render_watchdog(float(timeout_s) + _WATCHDOG_GRACE_S if capped else None):
        for role, report in zip(roles, picks):
            # Rollout 0, matching `preferences._clip_for`: recording each
            # candidate's best rollout would compare bests, not agents.
            traj = report.result.trajectories[0]
            rendered = _render_frames(render, traj, max_frames, width, warned, clock)
            if rendered is None:
                frames = None
            else:
                frames, frame_steps = rendered
            if frames is None:
                if clock.expired:
                    # In the artifact for the same reason `record_skipped` is:
                    # the fitness that comes back looks complete either way, and
                    # a run that quietly stopped producing footage after the
                    # third iteration is otherwise only visible to whoever
                    # counts the directories.
                    ctx.event("record_timeout", timeout_s=float(timeout_s),
                              elapsed_s=round(time.monotonic() - started, 3),
                              n_frames=clock.frames, n_recorded=len(written),
                              n_abandoned=len(picks) - len(written) - skipped)
                return written  # the renderer already warned; do not repeat it per candidate
            if not frames:
                skipped += 1
                continue  # this candidate's rollout carried no states; others may
            dest = root / report.cand_id
            path = encode(frames, dest, fps)
            traj.video_path = str(path)
            # `frame_steps` comes back FROM `_render_frames`, so the trace is
            # sampled at the indices the frames were actually rendered from
            # rather than at a second computation of the same expression. No
            # slice: that function returns the complete list or nothing at all
            # (a deadline returns None, a raising renderer returns [], an
            # unrenderable frame returns None) and every one of those paths is
            # handled above, so a prefix guard here would defend a state the
            # code cannot reach -- and would be the wrong repair anyway, since
            # surviving frames are not the first N indices.
            #
            # Writing the trace must never end a search. Everything else in this
            # loop degrades -- a renderer that raises, returns garbage, or blows
            # the deadline all leave the run going, because a video is not worth
            # a training run. This is I/O onto a possibly shared mount that can
            # hang, for an artifact that is off by default, and it runs AFTER
            # the training is paid for.
            trace_path = None
            try:
                trace_path = save_trajectory_trace(
                    trace_root, report.cand_id, traj, frame_steps,
                    mode=trace_mode, max_steps=trace_max)
            except Exception as exc:  # noqa: BLE001 - a lost trace is not a lost run
                _warn_once(warned, "trace-write",
                           "could not write the trajectory trace (%s: %s); the search "
                           "continues without it", type(exc).__name__, exc)
            # The clip's own health, measured HERE because this is the one
            # moment the pixels are in memory: re-deriving brightness later
            # means decoding 300 PNGs per candidate per run, and black frames
            # (`clip_stats`) go unnoticed precisely when nothing records them
            # at write time. Best-effort for the reason
            # the trace write is: a lost statistic is not a lost run.
            stats_path: Optional[Path] = None
            stats: Dict[str, Any] = {}
            try:
                stats = clip_stats(frames)
                if plan.layout:
                    # Beside the clip, because the clip is what a later reader
                    # opens: `_frames_from_disk` hands this back to the judge
                    # path so a frame sampled off disk is described to the judge
                    # exactly as a freshly rendered one is.
                    stats["views"] = plan.layout
                stats.update({"cand_id": report.cand_id, "fps": fps,
                              # Off the FILE the encoder produced, not the
                              # config: the format degrades when imageio is
                              # missing, and the file is authoritative
                              # (`WandbTracker.log_media` argues this in full).
                              "format": Path(path).suffix.lstrip(".").lower() or "frames"})
                # tmp + os.replace, the `_write_status` / `save_records`
                # convention: a reader may poll a run dir while it is written,
                # and a torn file reads as NO STATS -- which for this artifact
                # means a clip whose health is unknown reads exactly like one
                # never measured. The tmp name is not `render_stats.json`, so
                # no reader picks it up.
                stats_path = dest / "render_stats.json"
                tmp_stats = dest / "render_stats.json.tmp"
                tmp_stats.write_text(json.dumps(stats, indent=2))
                os.replace(tmp_stats, stats_path)
            except Exception as exc:  # noqa: BLE001 - a lost statistic is not a lost run
                _warn_once(warned, "render-stats",
                           "could not write render_stats.json (%s: %s); the "
                           "search continues without it", type(exc).__name__, exc)
                stats_path, stats = None, {}
                # And take the tmp file with it. THIS writer is the one that
                # can accumulate: its failure is swallowed and the loop goes
                # on to the next candidate and the next iteration, so a
                # failing `os.replace` would leave one `.tmp` per attempt
                # under `videos/<cand>/` for a reader to trip over later.
                # The other two atomic
                # writers here and in `artifacts.py` fail into paths where
                # the run is ending, so they have nothing to accumulate.
                #
                # Recomputed from `dest` rather than referencing the name
                # bound inside the `try`: `clip_stats` raising leaves it
                # unbound, and a NameError in the cleanup would replace a
                # degraded statistic with a dead run. Cleanup that cannot
                # itself fail, for the same reason the write is best-effort.
                try:
                    (dest / "render_stats.json.tmp").unlink(missing_ok=True)
                except OSError:
                    log.debug("could not remove the render_stats tmp file", exc_info=True)
            # Absent rather than null when the measurement failed, the same
            # convention `subtask_scores` follows in the journal.
            health = ({"luma_mean": stats.get("luma_mean"),
                       "distinct_colours": stats.get("distinct_colours"),
                       "stats": str(stats_path)} if stats_path else {})
            upload_role = upload_by_id.get(id(report))
            written.append(Recording(role=role, cand_id=report.cand_id, path=Path(path),
                                     upload_role=upload_role))
            ctx.event("record", cand_id=report.cand_id, role=role, upload_role=upload_role,
                      video=str(path), n_frames=len(frames),
                      n_views=multiview.panel_count(plan.layout),
                      trace=str(trace_path) if trace_path else "",
                      # In the journal as well as beside the clip: `record` is
                      # the line a reader greps when footage looks wrong, and
                      # two scalars are what make a dark sibling visible
                      # without opening 300 PNGs.
                      **health)
    if skipped:
        # In the artifact, not only in the log: a VLM-graded method whose frames
        # went missing falls through to a text-only comparison, and the fitness
        # that comes back looks complete either way.
        ctx.event("record_skipped", n_skipped=skipped, n_recorded=len(written),
                  n_picked=len(picks))
        log.warning("%d of %d selected rollout(s) could not be rendered; %d recorded",
                    skipped, len(picks), len(written))
    return written


# ==========================================================================
# §4's frames -- the pixels a VLM judge is actually sent
#
# `record_rollouts` runs AFTER stage 4 (`bird.py`), because it needs the reports
# to know which candidates to record. So at scoring time `Trajectory.video_path`
# is still None and there is nothing on disk to attach. `sample_frames` goes to
# the SOURCE the recorder goes to -- the stored states plus `env.render` --
# rather than to the recorder's output, so a judge does not depend on
# `output.video.record` having happened to pick its candidate. That
# independence is the point: narrowing `output.video.record` must stay an
# artifact choice that no fitness reads.
# ==========================================================================


def sample_frames(ctx: Any, traj: Trajectory, n_images: int, *,
                  prov: Optional[Dict[str, Any]] = None,
                  step_index: Optional[Sequence[int]] = None) -> List[bytes]:
    """Up to `n_images` evenly-spaced PNG frames of `traj`, as raw bytes.

    Bytes rather than paths because that is what `LLMClient(images=...)` takes
    without touching the filesystem (`bird/llm/anthropic_client.py::_image_block`
    base64s a `bytes` straight into an image block), and because the frames a
    judge sees must exist whether or not this trajectory was ever recorded.

    Three sources, in order, and the order is the cheap-first one:

      1. an already-written frames directory (`output.video.format: frames`),
      2. an already-written `gif`/`mp4`, decoded with imageio if it is present,
      3. `env.render(state)` replayed over `traj.states`.

    Returns `[]` -- never raises -- when there is nothing to sample: no states
    and no file, an env with no `render`, a renderer that raised, or a
    trajectory whose snapshots the adapter's LRU has since evicted. Every one of
    those is a *degrade*, and the caller is expected to record that it happened
    rather than to fail: a judgment made blind still returns a number.

    The render path is bounded by `output.video.timeout_s`, the same key and the
    same two mechanisms `record_rollouts` uses, because it is the same renderer
    and can wedge the same way.

    `prov` is an OUT-PARAMETER: pass a dict and it is filled with
    `judgments.frame_set`'s provenance for whatever was sampled -- which source
    answered, the episode step indices behind the frames, their size, and a
    per-frame luma and content hash. It is a mutable argument used as a channel
    for the same reason `_render_frames`'s `warned` set is one: this function's
    three sources are the only code that knows which of them ran and what stride
    it used, the return type is what every caller already unpacks, and the
    identity a judgment record needs (`cand_id`, which rollout) is knowable only
    to the caller. Filled even on the degraded paths, carrying the reason -- a
    judgment made blind must be legible as blind rather than as an empty list.

    `step_index` is for a `traj` that is a CLIP cut from a longer episode
    (`preferences._clip_for`): `step_index[i]` is the episode step behind clip
    index `i`. The rendered steps are translated through it HERE, before `_note`
    records them and before `_deliver` hands them to a `frame_policy` that burns
    them into a contact sheet's cells and lists them in its manifest -- so the
    frame record, the labels and the manifest carry ONE index space, the
    episode's. Remapping `prov["steps"]` after the fact would leave the record
    saying 0, 90, 181 while the pixels and manifest said 0, 30, 60.
    """
    def _blind(reason: str) -> List[bytes]:
        _note(prov, (), None, steps=None, source="", blind_reason=reason)
        return []

    n = max(0, int(n_images or 0))
    if n == 0:
        return _blind("evaluate.vlm.images_per_query is 0")
    if traj is None:
        return _blind("there is no trajectory to sample")

    png, rgb, source, disk_views = _frames_from_disk(traj.video_path, n)
    if png:
        # The on-disk stride is not this process's to claim: the recording may
        # have been written under a different `output.video.max_frames`, and
        # `judgments.frame_set` keeps `None` ("unknown") distinct from `[]`.
        # The panel layout IS the recorder's to claim -- it wrote it beside the
        # clip -- so it travels with the frames it describes.
        _note(prov, png, rgb, steps=None, source=source, views=disk_views)
        return _deliver(ctx, prov, png, rgb, None)

    plan = _view_plan(ctx)
    render, width = plan.render, plan.width
    if not callable(render):
        return _blind("the environment adapter has no render()")
    timeout_s = ctx.cfg.get("output.video.timeout_s")
    capped = timeout_s is not None
    started = time.monotonic()
    clock = _RenderClock(expires_at=started + float(timeout_s) if capped else None)
    with render_watchdog(float(timeout_s) + _WATCHDOG_GRACE_S if capped else None):
        rendered = _render_frames(render, traj, n, width, warned=None, clock=clock)
    if not rendered:
        # `_render_frames` returns None both for a spent clock and for a
        # renderer that cannot render (no-state `render`, an unrenderable
        # frame). The two call for different remedies -- a bigger
        # `output.video.timeout_s` against a different adapter -- so the record
        # carries the specific one, the rule `evaluation._vlm_frames` states
        # for every other degrade, and the journal names the timeout the way
        # `_record_rollouts` names its own (`record_timeout`). Otherwise a
        # judgment blinded by the deadline would be filed as "the renderer is
        # unusable for every trajectory" and no journal event would say the
        # guard had fired during scoring at all.
        if clock.expired:
            elapsed = round(time.monotonic() - started, 3)
            try:
                ctx.event("sample_timeout", timeout_s=float(timeout_s), elapsed_s=elapsed,
                          n_frames=clock.frames, n_requested=n,
                          traj_length=int(getattr(traj, "length", 0) or 0))
            except Exception:  # noqa: BLE001 - a lost event is not a lost judgment
                log.debug("could not journal a sample_frames timeout", exc_info=True)
            return _blind(f"output.video.timeout_s={float(timeout_s):g} elapsed after "
                          f"{clock.frames} frame(s) ({elapsed}s); the renderer was slow, "
                          f"not shown to be unusable")
        return _blind("the renderer is unusable for this trajectory (render() takes no "
                      "state, or a frame could not be rendered)")
    # `_render_frames` hands back `(frames, steps)` so a caller that pairs frames
    # with the reward's own numbers cannot recompute the stride and disagree. The
    # judge is handed the frames alone -- it is asked what the posture looks
    # like, not which step it came from -- but the mapping is kept:
    # `prov` carries it, because "which instants was this judgment made from" is
    # the one question the frames themselves cannot answer once they are bytes.
    frames, steps = rendered
    if not frames:
        return _blind("the rollout carries no states to render "
                      "(evicted, or never retained)")
    if step_index is not None and steps is not None:
        window = list(step_index)
        steps = [int(window[int(k)]) if 0 <= int(k) < len(window) else int(k) for k in steps]
    png = [encode_png(f) for f in frames]
    _note(prov, png, frames, steps=steps, source="render", views=plan.layout)
    if prov is not None and plan.note:
        prov["views_note"] = plan.note
    return _deliver(ctx, prov, png, frames, steps)


def _deliver(ctx: Any, prov: Optional[Dict[str, Any]], png: Sequence[bytes],
             rgb: Optional[Sequence[Any]],
             steps: Optional[Sequence[int]]) -> List[bytes]:
    """Hand the sampled frames to `evaluate.vlm.frame_policy` and return what to send.

    Called AFTER `_note`, and the order is deliberate. `frame_set` measures the
    frames one at a time -- per-frame step, luma and content hash -- and that
    record is what a reader of the judgment resolves against and must never
    have to re-derive. A policy that tiles those frames into one sheet does not get to
    erase them from the record: the individual frames stay measured exactly as
    before, and the composition is recorded ALONGSIDE under `composed`.

    The one place this leaves an honest seam is `frame_set`'s own docstring
    claim that its hashes are over "the bytes the client was actually handed".
    Under a composing policy they are over the frames the sheet was built from,
    which is the more useful record -- a sheet's hash cannot tell you which
    instant a cell came from -- but it is no longer literally what crossed the
    wire, so `composed.sheet_sha256` records that separately.

    Never raises. A delivery format is a presentation choice, and losing a whole
    judgment because a tiling failed would be the tail wagging the dog.
    """
    frames = list(png)
    cfg = getattr(ctx, "cfg", None)
    if cfg is None:
        return frames
    try:
        name = cfg["evaluate.vlm.frame_policy"]
    except Exception:  # noqa: BLE001 - an absent key is the default policy
        return frames
    if not name or name == "even":
        return frames
    try:
        from .registry import get  # noqa: PLC0415 - deferred: components import this module
        policy = get("frame_policy", str(name))
        lay = (prov or {}).get("views") if prov else None
        # The layout is passed only when there is one, so a policy written to the
        # four-argument contract keeps working on every single-view run; under
        # multi-view a policy that cannot take it degrades below, and says so.
        images, note, layout = (policy(ctx, png, rgb, steps, views=lay) if lay
                                else policy(ctx, png, rgb, steps))
        # Materialise and check HERE, inside the guard: this is a registry
        # family, so `images` is whatever some implementation returned, and a
        # non-iterable (or a policy that quietly delivered nothing) raising at
        # the `return` below would break the never-raises contract the
        # docstring promises -- and blind the judge, respectively.
        images = list(images)
        if not images or not all(isinstance(b, (bytes, bytearray)) for b in images):
            raise TypeError(f"frame_policy {name!r} returned "
                            f"{'no images' if not images else 'non-bytes images'}")
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.debug("frame_policy %r failed; sending frames individually", name, exc_info=True)
        # The degrade is RECORDED, not only logged: without this, a config
        # that names a policy and a record with no `composed` entry are
        # indistinguishable from a policy that was never invoked -- "crashed
        # and fell back" must be auditable as itself. The policies record
        # their own reasons when they return cleanly; this covers the raise.
        if prov is not None:
            try:
                prov["composed"] = {"frame_policy": str(name),
                                    "degraded": "policy raised "
                                                f"{type(exc).__name__}; frames "
                                                "sent individually"}
            except Exception:  # noqa: BLE001 - a lost note is not a lost judgment
                pass
        return frames
    if prov is not None and layout:
        try:
            rec = dict(layout)
            if images and len(images) == 1:
                rec["sheet_sha256"] = hashlib.sha256(images[0]).hexdigest()[:16]
            if note:
                rec["manifest"] = note
            prov["composed"] = rec
            # Both halves of what `inputs+frames` promises, when the run keeps
            # pixels at all: the SOURCE frames the record hashes, then the
            # sheet that actually crossed the wire. Stored HERE and in this
            # order because the caller's `note_frames(..., png=<delivered>)`
            # runs after this and is handed the sheet, not the sources --
            # without these calls nothing would store the sources under a
            # composing policy.
            if images and len(images) == 1:
                try:
                    from .judgments import note_composed, note_source_frames  # noqa: PLC0415
                    set_id = str((prov or {}).get("id") or "")
                    note_source_frames(ctx, set_id, png)
                    note_composed(ctx, set_id, images[0])
                except Exception:  # noqa: BLE001 - see the docstring
                    log.debug("could not store the composed sheet", exc_info=True)
        except Exception:  # noqa: BLE001 - a lost measurement is not a lost judgment
            log.debug("could not record a composed frame set", exc_info=True)
    return images


def _note(prov: Optional[Dict[str, Any]], png: Sequence[bytes],
          rgb: Optional[Sequence[Any]], **kw: Any) -> None:
    """Fill `sample_frames`'s out-parameter, or do nothing. Never raises.

    A measurement about a judgment must not be able to break the judgment: the
    whole module this delegates to is best-effort for that reason, and the guard
    belongs here rather than at four call sites above.
    """
    if prov is None:
        return
    try:
        prov.update(frame_set(png, rgb, **kw))
    except Exception:  # noqa: BLE001 - a lost measurement is not a lost judgment
        log.debug("could not measure a judge frame set", exc_info=True)


def _views_beside(path: Path) -> Optional[Dict[str, Any]]:
    """The panel layout `record_rollouts` wrote into `render_stats.json`, or None.

    None for a single-view recording AND for a recording that predates the key, and
    that is the right reading of both: a frame with no layout beside it is described
    to the judge as one view, which is what it is.
    """
    stats = path / "render_stats.json" if path.is_dir() else path.parent / "render_stats.json"
    try:
        blob = json.loads(stats.read_text())
    except (OSError, ValueError):
        return None
    lay = blob.get("views") if isinstance(blob, dict) else None
    return lay if isinstance(lay, dict) and multiview.panel_count(lay) > 1 else None


def _frames_from_disk(video_path: Optional[str], n: int
                     ) -> Tuple[List[bytes], Optional[List[np.ndarray]], str,
                                Optional[Dict[str, Any]]]:
    """Sample an already-written recording. `([], None, "", None)` when there is not one.

    Returns the bytes, the same frames as ARRAYS where this process decoded them,
    which source answered, and the panel layout the recorder wrote beside the clip
    (`_views_beside`). The arrays are what lets a judgment record carry
    per-frame brightness; the read-PNGs-off-a-directory path never holds any, so
    it hands back `None` and the record says the pixels never passed through here
    rather than reporting a brightness it did not measure.
    """
    if not video_path:
        return [], None, "", None
    path = Path(video_path)
    if path.is_dir():
        pngs = sorted(path.glob("*.png"))
        return ([pngs[i].read_bytes() for i in _even_indices(len(pngs), n)],
                None, "frames_dir", _views_beside(path))
    if not path.is_file():
        return [], None, "", None
    imageio = _imageio(need_ffmpeg=path.suffix.lower() == ".mp4")
    if imageio is None:
        return [], None, "", None
    try:
        frames = list(imageio.mimread(str(path), memtest=False))
    except Exception:  # noqa: BLE001 -- an unreadable clip degrades to no frames
        log.debug("sample_frames could not decode %s", path, exc_info=True)
        return [], None, "", None
    picked = [_as_rgb8(frames[i]) for i in _even_indices(len(frames), n)]
    return ([encode_png(f) for f in picked], picked, path.suffix.lstrip(".").lower(),
            _views_beside(path))
