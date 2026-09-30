"""Pairwise-preference machinery -- Gran Turismo Sophy's evaluation loop.

Everything here belongs to GT (§4, `evaluate.preferences.*`),
the one published method that ranks reward candidates by *comparing agents*
rather than by scoring them. Three families live in this file:

    comparator      who decides a single head-to-head    (§4)
    pair_strategy   which head-to-heads get run at all   (§4)
    pref_aggregator how a bag of comparisons becomes a ranking (§4)

and one plain function, `run_preferences`, which sequences them. It is NOT
registered here: `bird/components/phases.py` registers it as `phase:preferences`
so that `evaluate()` in `bird.py` can reach it. Keeping the registration there
and the implementation here is only a file-ownership split; the callable is the
one `registry.get("phase", "preferences")` returns.

Three faithfulness points that are easy to get wrong, all recorded in §4:

* **`llm_on_vlm_captions` is not a synonym for `vlm`.** It is GT's actual
  design: a VLM captions the two clips *contrastively* (both agents described
  in one pass, side by side), and then a separate **text** LLM decides from the
  captions alone. It exists because the clips do not fit in the decider's
  context, so the second step must never see the frames. Both steps are
  implemented, and the decision prompt is built from the caption only. The
  caption also OUTLIVES the comparison: it is appended to both sides'
  `report.meta["pair_captions"]`, because App. D's no-human feedback is an LLM
  summary of exactly these descriptions, and `analyzer: llm` reads them there
  rather than re-analysing numeric traces.
* **`allow_ties` is false in GT, deliberately.** The comparator is forced to
  produce a decision; `TIE` is a sentinel that only survives when a config
  turns ties on, and even then it is recorded as two mirrored preferences so
  that `Preference.label` keeps its frozen `{0, 1}` contract.
* **`evaluate.artifacts` changes what the judge sees, and that is measurable.**
  GT reports 74.14% VLM-human agreement with trajectory+video against 62.96%
  for video alone (§4, Table 2; human-human is 78.54%). So the prompt carries
  state/action text when `state_trajectories` is in `evaluate.artifacts`, and
  not otherwise. A config that asks for videos only gets the weaker judge, on
  purpose -- that ablation is the point.

  **`videos` means PIXELS.** Frames go to `comparator_vlm` and to step 1 of
  `comparator_llm_on_vlm_captions`; step 2 gets none, because that is the
  design and not an oversight. A pairwise judge handed only a line naming a
  file path -- with `llm.evaluator.modality: vlm` and
  `evaluate.vlm.images_per_query` read only to interpolate that string --
  would turn both of GT's video conditions into text conditions, and an
  agreement rate computed between one of them and a human shown a frame strip
  would measure a difference in MODALITY, not in judge. Every pairwise query
  record carries `n_images`, and `budget.blind_comparisons` counts the sides
  judged without pixels.

  **The rows are the ENVIRONMENT's, and they are documented.** App. C hands
  the VLM "frame-by-frame data coming directly from the environment's API"
  and, once, ahead of both clips, "Documentation about each frame's data"
  (refs/tex/gt_reward_design/neurips_2025.tex:682-688). A reward is not an
  environment field, the paper's judge never sees one, and a reward can
  inflate it: a caption that cites the candidate's own reward trace as the
  evidence of task success makes the preference labels -- GT's stand-in for
  the inaccessible F -- partly a function of the candidate's self-report. So
  `_render` writes `t= s= a=` only, and `_variables_documentation` renders
  the adapter's `_state_fields` / `_action_fields` into the paper's slot,
  placed by each comparator once ahead of both sides.

Scope discipline (§4, `evaluate.preferences.scope`): GT renormalises
Bradley-Terry strengths **per round** and never compares them across rounds,
which is the `within_iteration` default -- the aggregator is handed only *this*
iteration's preferences and only *this* iteration's ids. The aggregator's
signature still has no `state`, and that is still load-bearing: widening is a
decision `_run_preferences` reads off a config key and hands in, never one an
aggregator can take for itself. `cumulative` (REvolve, Appendix B.1 p.20 --
comparisons "update the fitness scores of all individuals at the end of each
generation") widens the **aggregation** pool and NOT the **pair** pool; the
decision, the resume hazard behind it and what the refit therefore does and does
not reproduce are in `_cumulative_ids`. `state.preferences` accumulates
unconditionally when `evaluate.preferences.store_dataset` is set, because that
store is what §2's TAC screen consumes NEXT iteration (contrast CARD's
self-gated `verify.tpe.store: append_on_pass`, which gates the growth of its
own evidence) -- and under `cumulative` it is additionally the entire evidence
base the ranking is fitted over, so that one list now has two readers with two
different failure modes.

Unpublished members of these families are marked in their docstrings: GT names
Bradley-Terry only and runs all-pairs, so `borda`/`copeland`, `tournament` and
`active` are reachable points that no paper occupies. `elo_raw` is REvolve's
pin down to the constants (`refs/tex/revolve/main.tex:993-1007`;
`refs/code/Revolve/human_feedback/elo_scoring.py:15-27`) and returns the raw
rating as fitness; `elo` replays the same games and maps the ratings onto a
10^(R/400) simplex, which nobody publishes -- it is kept as the BT-comparable
image and as a strictly stricter admission gate than `elo_raw`.
`active` is a Bayesian-experimental-design hook, an open research direction,
and it is implemented for real -- not as a random subset with a better name.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace as _dc_replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .evaluation import client_sees_images, judge_concurrency

import numpy as np

from .. import judgments, multiview
from .. import preference_log, registry
from ..budget import BudgetExceeded
from ..observability import sample_frames
from .frames import sheet_cells, sheet_manifest
from ..registry import register
from ..types import CandidateReport, Preference, Trajectory

log = logging.getLogger("bird")

#: Comparator return value for "these two are indistinguishable". Legal only
#: when `evaluate.preferences.allow_ties`; GT sets that false on purpose, and
#: `run_preferences` resolves any tie that leaks through rather than dropping
#: the pair (a dropped pair is silently lost evidence).
TIE = -1
#: A judge that produced no usable verdict, under a REAL
#: provider. The pair is DROPPED from D_pref rather than decided by the
#: agents' own reward magnitudes, which is neither a VLM preference nor a
#: behavioural substitute. Only `provider:
#: mock` still gets a deterministic stand-in -- there is no real judge and the
#: offline suite must exercise the path.
ABSTAIN = -2

#: Private cache key on `CandidateReport.meta`. A clip is a pure function of
#: the report plus config, but the comparator and the phase both need it, and
#: the frozen `comparator_fn(ctx, state, left, right)` signature gives no way
#: to pass one in. `run_preferences` pops this before returning.
_CLIP_KEY = "_pref_clip"

#: The EPISODE step indices `_clip_for` drew the clip from, cached beside it and
#: dropped with it. `Trajectory` has no field for a stride and should not grow
#: one -- it describes an episode, not how somebody sampled it -- but "which
#: instants was this comparison judged from" is exactly what a judgment record
#: has to be able to answer, and after `_take` the clip is a new Trajectory whose
#: own indices are 0..n and say nothing about the episode behind it. Recorded
#: rather than recomputed, for `save_trajectory_trace`'s reason.
_CLIP_STEPS_KEY = "_pref_clip_steps"

#: The PNG bytes a frames-attaching comparator hands the judge, plus their
#: provenance, cached beside the clip and dropped with it. Cached for the reason
#: `evaluation._vlm_trajectory_analysis` samples once per rollout rather than once per
#: subtask: on `all_pairs` over ten candidates each side is compared nine times,
#: and on Meta-World a frame is a MuJoCo render. Same pixels, nine questions.
_FRAMES_KEY = "_pref_frames"


# ==========================================================================
# What the comparator is shown
# ==========================================================================


def _int(value: Any, default: int) -> int:
    """Schema marks these keys nullable; a null must not crash the judge."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _n_frames(traj: Optional[Trajectory]) -> int:
    if traj is None:
        return 0
    for seq in (traj.rewards, traj.states, traj.actions):
        try:
            return len(seq)  # type: ignore[arg-type]
        except TypeError:
            continue
    return max(int(traj.length or 0), 0)


def _take(seq: Any, idx: Sequence[int]) -> Any:
    """Subsample any of the `Any`-typed trajectory fields without assuming
    numpy. Returns the original object if it cannot be indexed -- a comparator
    losing resolution is recoverable, a crash is not."""
    if seq is None:
        return None
    if hasattr(seq, "shape") and hasattr(seq, "__getitem__"):
        try:
            return seq[list(idx)]
        except Exception:  # pragma: no cover - exotic array types
            return seq
    try:
        return [seq[i] for i in idx]
    except Exception:
        return seq


#: The FULL-RESOLUTION sub-trajectory TAC re-scores, cached beside the view clip.
#: `_clip_for` even-strides the source window down to the frames the judge SEES;
#: this keeps the window's real `(s, a, s')` transitions so a candidate's own
#: reward can be re-applied to actual adjacent steps. Storing the view clip in
#: `D_pref` instead would pair states up to 13 real steps apart with a single
#: earlier action and drop the terminal next-state.
_PREF_TRAJ_KEY = "_pref_tac_traj"


def _head(seq: Any, n: int) -> Any:
    """The first `n` elements of an `Any`-typed trajectory field, contiguous and
    in order -- a real slice, not an even-stride sample. `seq[:n]` for anything
    sliceable (list, tuple, ndarray); otherwise `_take` with a range, which
    returns the original object when even that fails."""
    if seq is None:
        return None
    try:
        return seq[:n]
    except (TypeError, KeyError):
        return _take(seq, list(range(n)))


def _pref_traj_for(ctx: Any, report: CandidateReport) -> Optional[Trajectory]:
    """The sub-trajectory stored in `D_pref` for TAC (§2), at FULL resolution.

    Same SOURCE WINDOW the judge's clip is drawn from -- the first
    `max_video_s * fps` steps ("first lap or three minutes", §4) -- but its
    states/actions/rewards are kept contiguous and unsampled, so `screens._traj_
    transitions` reconstructs REAL `(s, a, s')` triples. The window carries
    `W` transitions, so `W + 1` states and `W` actions, preserving the
    T-actions/T+1-states rollout convention (`training._rollout`); a candidate's
    own reward is then re-scored on the terminal next-state and on every action,
    neither of which survived the view clip's `_take`.

    The judge still watches `_clip_for`'s even-strided frames; TAC re-scores the
    behaviour those frames were sampled from. Re-scoring the subsample itself
    would measure a candidate against transitions that never occurred."""
    cached = report.meta.get(_PREF_TRAJ_KEY)
    if cached is not None:
        return cached
    trajs = report.result.trajectories or []
    if not trajs:
        return None
    source = trajs[0]
    cfg = ctx.cfg
    fps = max(1, _int(cfg.get("evaluate.preferences.fps"), 10))
    max_s = max(1, _int(cfg.get("evaluate.preferences.max_video_s"), 180))
    total = _n_frames(source)  # T transitions == len(rewards)
    if total <= 0:
        report.meta[_PREF_TRAJ_KEY] = source
        return source
    window = min(total, max_s * fps)  # transitions kept

    taken = _head(source.rewards, window)
    try:
        rewards = [float(r) for r in taken] if taken is not None else []
    except (TypeError, ValueError):
        rewards = []
    # `window + 1` states for `window` actions: the terminal next-state is a
    # real transition and `_traj_transitions` builds all `window` of them.
    n_states = _n_frames(Trajectory(states=source.states)) if source.states is not None else 0
    try:
        n_states = len(source.states)  # type: ignore[arg-type]
    except TypeError:
        n_states = window + 1
    sub = Trajectory(
        states=_head(source.states, min(window + 1, n_states)),
        actions=_head(source.actions, window),
        rewards=rewards,
        component_values={k: _head(v, window)
                          for k, v in (source.component_values or {}).items()},
        success=source.success,
        length=window,
        ret=float(sum(rewards)),
        video_path=source.video_path,
    )
    report.meta[_PREF_TRAJ_KEY] = sub
    return sub


def _clip_for(ctx: Any, report: CandidateReport) -> Optional[Trajectory]:
    """Cut the clip the comparator judges, per `clip_length_s`/`fps`/`max_video_s`.

    GT shows 15s clips at 10fps drawn from the first lap or the first three
    minutes (§4). The toy envs have no wall-clock rate, so one env step is
    treated as one frame at `fps`: the source window is the first
    `max_video_s * fps` steps ("first lap or 3 minutes") and the clip carries
    `clip_length_s * fps` frames drawn from it.

    App. C is NOT silent here, and the divergence is recorded rather than
    excused: the paper captions EVERY 15s clip of the
    3-minute video and stitches all the descriptions into the decider's prompt
    ("The result of comparing every pair of clips in the complete video is
    shown below"). BIRD instead makes ONE caption call over one clip-budget of
    frames evenly subsampled from the whole source window -- a single-window
    simplification of the stitched-clips protocol. On the toy envs the two
    coincide: an episode shorter than one clip makes `window <= budget`, so
    the whole episode IS the one clip. Even-stride subsampling is chosen over
    a contiguous opening window because an opening window on a short episode
    would show the judge nothing but the reset transient. Both agents get the
    same rule and therefore the same view, which is what keeps the comparison
    contrastive. (The related 20-frames-per-side cap vs the paper's 150/clip
    is recorded at configs/methods/gt.yaml, the
    images_per_query note.)
    """
    cached = report.meta.get(_CLIP_KEY)
    if cached is not None:
        return cached

    trajs = report.result.trajectories or []
    if not trajs:
        return None
    # Rollout 0 for both sides: GT races the two agents on the same course, and
    # picking each side's *best* rollout would compare bests, not agents.
    source = trajs[0]

    cfg = ctx.cfg
    fps = max(1, _int(cfg.get("evaluate.preferences.fps"), 10))
    clip_s = max(1, _int(cfg.get("evaluate.preferences.clip_length_s"), 15))
    max_s = max(1, _int(cfg.get("evaluate.preferences.max_video_s"), 180))

    total = _n_frames(source)
    if total <= 0:
        return source

    window = min(total, max_s * fps)
    budget = max(1, clip_s * fps)
    if window <= budget:
        idx = list(range(window))
    else:
        idx = sorted({int(round(k)) for k in np.linspace(0, window - 1, budget)})

    taken = _take(source.rewards, idx)
    try:
        rewards = [float(r) for r in taken] if taken is not None else []
    except (TypeError, ValueError):
        rewards = []
    clip = Trajectory(
        states=_take(source.states, idx),
        actions=_take(source.actions, idx),
        rewards=rewards,
        component_values={k: _take(v, idx) for k, v in (source.component_values or {}).items()},
        success=source.success,
        length=len(idx),
        # The clip's own return, not the episode's: `mean_per_step_return` on
        # this object is then the length-corrected statistic TAC re-scores in
        # §2, so the judge and the screen agree on what "better" means.
        ret=float(sum(rewards)),
        video_path=source.video_path,
    )
    report.meta[_CLIP_KEY] = clip
    report.meta[_CLIP_STEPS_KEY] = idx
    # ONCE, here, on the cold path -- not on each comparison that reads it. A
    # clip is cut once per report per phase and then judged in as many pairs as
    # the pair strategy asks for, and `evaluate.vlm.repeats` multiplies that
    # again; writing its 150 step indices onto every one of those records made
    # the comparison lines mostly a repeated copy of this one. `judgments`'
    # frames-and-queries split is exactly this shape, for exactly this reason:
    # the evidence is recorded once and the queries point at it.
    judgments.note_frames(
        ctx, judgments.clip_set(idx, n_frames=len(idx),
                                video_path=source.video_path),
        iteration=int(getattr(report.candidate, "iteration", -1)),
        cand_id=report.cand_id, rollout=0)
    return clip


#: How many entries of a state/action vector one trajectory row shows. The legend
#: `_variables_documentation` writes covers exactly these and says how many more
#: the adapter declares, so a row and its documentation cannot disagree.
_VEC_WIDTH = 6


def _vec(x: Any, width: int = _VEC_WIDTH) -> str:
    try:
        arr = np.asarray(x, dtype=float).ravel()
    except (TypeError, ValueError):
        return str(x)[:60]
    if arr.size == 0:
        return "[]"
    tail = "..." if arr.size > width else ""
    return "[" + " ".join(f"{v:.3g}" for v in arr[:width]) + tail + "]"


def _at(seq: Any, i: int, scalar: bool = False) -> str:
    """One frame's worth of a trajectory field, or "n/a". The fields are typed
    `Any`, so every access here is a guess that must not be able to crash."""
    if seq is None:
        return "n/a"
    try:
        item = seq[i]
    except Exception:
        return "n/a"
    if scalar:
        try:
            return f"{float(item):.4f}"
        except (TypeError, ValueError):
            return "n/a"
    return _vec(item)


def _pair_frames(ctx: Any, report: CandidateReport) -> Tuple[List[bytes], Dict[str, Any]]:
    """The pixels one side of a head-to-head is judged on, and why not more.

    `evaluation._vlm_frames`'s sibling, and gated the same three ways -- the
    config must declare rollout footage as evidence (`evaluate.artifacts`
    containing `videos`, already coherence-checked against
    `output.video.enabled`), the client must be one that would actually look
    (`client_sees_images`: an `images=` keyword AND `modality: vlm`, because
    `AnthropicClient._complete` drops images handed to a text client with a
    warning that scrolls past in a 20-hour log), and there must be a rollout.

    Returns `(png, prov)`. `prov` is `judgments.frame_set`'s provenance and is
    filled on every path INCLUDING the blind ones, carrying the reason: a
    comparison made with no pixels must be legible as blind rather than as a
    frame set that happens to be empty, and this method's whole mechanism is
    watching rollouts.

    **The frames are rendered from the clip's own states, never from
    `video_path`, and that is not an optimisation.** `_clip_for` cuts GT's
    15-second window (`clip_length_s`/`fps`/`max_video_s`, §4) and carries the
    SOURCE episode's `video_path` along with it, so handing that path to
    `sample_frames` would sample evenly across the whole episode and quietly
    make `clip_length_s` inert -- the same key-read-only-to-interpolate-a-string
    failure this function exists to end. Worse, it would be inconsistent
    BETWEEN candidates: under `output.video.record: best_and_worst` two of ten
    have a file and eight do not, so two sides of a tournament would be judged
    on the episode and eight on the clip, with nothing saying so.
    `_vlm_frames` makes the same argument for the same reason -- the judge's
    input must not depend on what the recorder happened to pick.

    `sample_frames` is handed the clip's EPISODE step window (`step_index`,
    from `_CLIP_STEPS_KEY`): it indexes the trajectory it is given, which here
    is the clip, whose own indices run 0..n and say nothing about the episode
    behind it, so the translation happens INSIDE it -- before the frame record
    is written and before a `frame_policy` burns the steps into a contact
    sheet's cells and lists them in its manifest. One index space: remapping
    `prov["steps"]` afterwards would leave the record carrying episode indices
    while the sheet's labels and manifest carried clip-local ones. Recorded
    rather than recomputed, for
    `save_trajectory_trace`'s reason: a consumer that re-derived the stride
    from config keys that have since moved is wrong by exactly the amount that
    looks like a real finding.
    """
    cached = report.meta.get(_FRAMES_KEY)
    if cached is not None:
        return cached

    prov: Dict[str, Any] = {}

    def keep(png: List[bytes]) -> Tuple[List[bytes], Dict[str, Any]]:
        """Cache the result and record it, ONCE, here where the frames are cut.

        `_clip_for` writes its clip record at the same point and for the same
        reason. Recording in `_warm_frames` instead would leave the join
        dangling for every caller that is not `run_preferences` -- a comparator
        invoked directly writes a query record naming a `frame_set` id, and
        there would be no `frames` record under it. A query pointing at nothing
        is worse than a query pointing at an empty set: one is a broken artifact
        and the other is a stated fact.
        """
        judgments.note_frames(
            ctx, prov, iteration=int(getattr(report.candidate, "iteration", -1)),
            cand_id=report.cand_id, rollout=0, png=png)
        out: Tuple[List[bytes], Dict[str, Any]] = (png, prov)
        report.meta[_FRAMES_KEY] = out
        return out

    def blind(reason: str) -> Tuple[List[bytes], Dict[str, Any]]:
        prov.update(judgments.frame_set((), None, blind_reason=reason))
        return keep([])

    if "videos" not in (ctx.cfg.get("evaluate.artifacts") or []):
        return blind("evaluate.artifacts does not list videos")
    if not client_sees_images(getattr(ctx, "evaluator", None)):
        return blind("the evaluator client takes no images or is not modality: vlm")
    clip = _clip_for(ctx, report)
    if clip is None:
        return blind("no rollout was recorded for this agent")

    n_images = max(0, _int(ctx.cfg.get("evaluate.vlm.images_per_query"), 20))
    window = report.meta.get(_CLIP_STEPS_KEY)
    png = list(sample_frames(ctx, _dc_replace(clip, video_path=None),
                             n_images, prov=prov, step_index=window or None))
    # NOT `blind(...)` on an empty result: `sample_frames` has already filled
    # `prov` with the SPECIFIC reason (which of its sources failed), and
    # overwriting it with a summary would throw away the only part a reader can
    # act on. `_vlm_frames` draws the same line.
    return keep(png)


def _count_blind(ctx: Any) -> None:
    """One blind side of one comparison, in the cost column and not only a log.

    Counted, not just printed. The judge still receives the text lines and still
    returns a preference, so a run whose pixels never landed produces a
    complete-looking fitness for a method whose entire mechanism is watching
    rollouts. Nothing errors; the number simply means something else.

    **The condition is that no pixels reached the judge**, which is the fact
    `blind_comparisons` is named for -- not that `clip.video_path` is empty,
    which is a statement about the RECORDER: on `output.video.record:
    best_and_worst` that would fire for eight of ten candidates in every
    comparison even when all ten were judged blind. One entry per
    call, per side, per comparison: `tests/test_vlm_concurrency.py` derives its
    lock-entry arithmetic from this counter, so the cardinality is load-bearing.
    """
    budget = getattr(ctx, "budget", None)
    if budget is not None and hasattr(budget, "record_blind_comparison"):
        with _JUDGE_BUDGET_LOCK:
            budget.record_blind_comparison()


def _render(ctx: Any, report: CandidateReport, side: str,
            attach: bool = False) -> str:
    """Describe one agent to the judge, governed by `evaluate.artifacts`.

    This function is where GT's 74.14%-vs-62.96% finding is expressed as a
    config knob (§4): `state_trajectories` adds the per-step state/action text
    that lifts VLM-human agreement, `videos` alone leaves the judge with the
    clip. Nothing here inspects the method name.

    The rows are `t= s= a=` and nothing else -- the environment's observation
    and the action, App. C's "frame-by-frame data coming directly from the
    environment's API" (tex:682). Rows deliberately omit `r=<the candidate's
    own per-step reward>`: it is not an environment field and is the one
    number in the prompt the thing being judged controls (see the module
    docstring). What
    the entries MEAN is `_variables_documentation`, which the comparators place
    once ahead of both sides rather than per side, as the paper's template does.

    `attach` says whether the caller is about to send the clip's PIXELS
    alongside this text (`_pair_frames`), and it changes only the `videos` line.
    A frames-attaching comparator states what is attached; `comparator_human`
    passes False and keeps the file path, because the file path is the one
    channel through which its judge -- a person at a terminal -- can actually
    watch the rollout.
    """
    cfg = ctx.cfg
    artifacts = set(cfg.get("evaluate.artifacts") or [])
    clip = _clip_for(ctx, report)
    lines = [f"[{side}]"]

    if clip is None:
        lines.append("  (no rollout was recorded for this agent)")
        return "\n".join(lines)

    fps = max(1, _int(cfg.get("evaluate.preferences.fps"), 10))
    n = _n_frames(clip)

    if "videos" in artifacts:
        secs = n / float(fps)
        if attach:
            png, prov = _pair_frames(ctx, report)
            if png:
                cells = sheet_cells(prov)
                if cells:
                    # One image that is a GRID: say so, and hand over the manifest
                    # (which numbers are burned into which cell) -- otherwise the
                    # judge is told one frame is attached and shown a tiled sheet
                    # with unexplained numbers on it. The non-sheet sentence below
                    # is unaffected.
                    lines.append(f"  video: {n} frames @ {fps}fps ({secs:.1f}s); "
                                 f"one contact sheet tiling {cells} evenly-spaced "
                                 f"frames of this clip is attached")
                    manifest = sheet_manifest(prov)
                    if manifest:
                        lines.append("  " + manifest.replace("\n", "\n  "))
                else:
                    lines.append(f"  video: {n} frames @ {fps}fps ({secs:.1f}s); "
                                 f"{len(png)} evenly-spaced frames of this clip "
                                 f"are attached")
                # Under `output.video.n_views > 1` each attached frame is several
                # cameras side by side; the judge is told which, per side, in
                # the spec's own words (`multiview.describe`). Empty otherwise,
                # so the single-view prompt is byte-identical to one without it.
                panels = multiview.describe(prov.get("views"), short=True)
                if panels:
                    lines.append(f"  frames: {panels}")
            else:
                # The REASON is not in the prompt. A prompt is an input to a
                # model, not a log: "the evaluator client takes no images or is
                # not modality: vlm" tells the judge nothing it can use and
                # invites it to reason about our plumbing. It travels in
                # `prov["blind_reason"]`, which reaches the judgment trace, the
                # `pair_frames` journal event and the query record.
                lines.append(f"  video: {n} frames @ {fps}fps ({secs:.1f}s); "
                             f"no frames could be attached")
                _count_blind(ctx)
        else:
            # No pixels are going out with this text, so the path is the only
            # handle on the clip there is -- and for `comparator_human` it is a
            # working one. `budget.blind_comparisons` still records the case
            # where even that is missing.
            where = clip.video_path or "(not rendered)"
            lines.append(f"  video: {n} frames @ {fps}fps ({secs:.1f}s), file={where}")
            if not clip.video_path:
                _count_blind(ctx)

    if "scalar_metrics" in artifacts or not artifacts:
        lines.append(
            f"  outcome: {'SUCCESS' if clip.success else 'no success flag'}; "
            f"clip return {clip.ret:.4f} over {n} frames "
            f"(mean per step {clip.mean_per_step_return:.4f})")

    if "component_traces" in artifacts and clip.component_values:
        for name, series in sorted(clip.component_values.items()):
            try:
                # `None` marks a step with no claim (per-step packing)
                vals = np.asarray([v for v in series if v is not None], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                mean = float(np.mean(vals))
            except (TypeError, ValueError):
                continue
            lines.append(f"  component {name}: mean {mean:.4f}")

    if "state_trajectories" in artifacts:
        # `evaluate.vlm.images_per_query` is the frames-per-query knob (RDA: 20
        # for the agent view); it bounds how much of the clip actually reaches
        # the judge, exactly as it bounds how many frames reach a VLM.
        per_query = max(1, _int(cfg.get("evaluate.vlm.images_per_query"), 20))
        picks = sorted({int(round(k)) for k in np.linspace(0, max(n - 1, 0), min(per_query, n))})
        lines.append(f"  trajectory ({len(picks)} of {n} frames):")
        for i in picks:
            # No `r=`: `clip.rewards` is what the candidate PAID itself, and a
            # judge shown it reads task success off the reward it is meant to be
            # checking (tex:682 gives the VLM environment-API data only).
            lines.append(f"    t={i:<5} s={_at(clip.states, i)} a={_at(clip.actions, i)}")

    return "\n".join(lines)


def _variables_documentation(ctx: Any) -> str:
    """App. C's `{variables_documentation}`: what each `s[i]` / `a[i]` in the
    trajectory rows IS, from the environment's own declaration.

    GT's caption prompt gives the VLM "frame-by-frame data coming directly from
    the environment's API" and states "Documentation about each frame's data is
    given here" ONCE, ahead of both clips (refs/tex/gt_reward_design/
    neurips_2025.tex:682-688). The source is the adapter's `_state_fields` /
    `_action_fields` -- the same `(name, doc)` pairs `EnvAdapter._render_api_stub`
    turns into the generator's `state_action_api_stub` -- so the judge reads the
    rows with the names the reward was written against.

    Empty, and therefore byte-neutral for the prompt, when `state_trajectories`
    is not an artifact (the block belongs to the channel Table 2 measures; the
    video-only arm is the WEAKER judge on purpose) or when the adapter declares
    no fields. `_vec` shows the first `_VEC_WIDTH` entries of a longer vector,
    so the legend covers exactly the entries a row shows and says how many more
    exist rather than documenting numbers the judge cannot see.
    """
    artifacts = set(ctx.cfg.get("evaluate.artifacts") or [])
    if "state_trajectories" not in artifacts:
        return ""
    env = getattr(ctx, "env", None)

    def pairs(attr: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for item in list(getattr(env, attr, None) or []):
            if isinstance(item, (tuple, list)) and len(item) == 2:
                out.append((str(item[0]), str(item[1])))
            else:
                out.append((str(item), ""))
        return out

    s_fields, a_fields = pairs("_state_fields"), pairs("_action_fields")
    if not s_fields and not a_fields:
        return ""
    lines = ["DOCUMENTATION of the trajectory rows (frame-by-frame data from the "
             "environment's API; t is the frame index within the clip):"]
    for label, what, fields in (("s", "state / observation vector", s_fields),
                                ("a", "action vector", a_fields)):
        if not fields:
            continue
        lines.append(f"  {label} = {what}:")
        for i, (name, doc) in enumerate(fields[:_VEC_WIDTH]):
            lines.append(f"    {label}[{i}] {name}" + (f": {doc}" if doc else ""))
        if len(fields) > _VEC_WIDTH:
            lines.append(f"    ({len(fields) - _VEC_WIDTH} further {label} entries are "
                         f"declared and not shown in the rows)")
    return "\n".join(lines)


# ==========================================================================
# Talking to the judge
# ==========================================================================

_LLM_METHODS = ("complete", "chat", "generate", "ask", "respond", "__call__")


def _as_text(out: Any) -> Optional[str]:
    if out is None:
        return None
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for key in ("text", "content", "completion", "response", "message"):
            if isinstance(out.get(key), str):
                return out[key]
        return None
    for attr in ("text", "content", "completion"):
        val = getattr(out, attr, None)
        if isinstance(val, str):
            return val
    if isinstance(out, (list, tuple)):
        for item in out:
            got = _as_text(item)
            if got:
                return got
    return None


def _client_call(client: Any, prompt: str,
                 images: Optional[Sequence[Any]] = None
                 ) -> Tuple[Optional[str], int]:
    """Duck-type the LLM client.

    `llm.*.provider` selects the client (`registry.get("llm", ...)`) and its
    call convention is that module's business, not this one's. Every failure
    below degrades to `None`, which routes the comparator to its offline
    decision rule -- a preference round must stay deterministic and offline
    even with no provider at all (repo rule: the toy env + mock LLM run the
    whole method end to end).

    `images` is passed as the `images=` keyword `bird/llm/base.py` declares, and
    it is tried FIRST for each call shape -- a client that accepts the keyword
    is never called without it, so a conformant provider cannot answer blind
    because we took a shortcut. A client with no such keyword (a duck-typed stub,
    an older provider) is still asked, without images, and the caller learns the
    pixels did not land from `client_sees_images` and from the `n_images` on its
    own query record rather than from silence. (`evaluation._client_text` is
    STRICTER: its callers gate frames on `client_sees_images` beforehand, so it
    never retries a sighted call blind at all and returns `""` instead -- a
    refused or truncated image request there must not become a text verdict
    filed as an image-graded one.)

    **Returns how many images the answering call ACTUALLY carried, not how many
    it was given**, and the distinction is the whole reason for the second
    return value. `client_sees_images` says True for a client whose signature it
    cannot read ("let the call itself decide"), and the `except TypeError:
    continue` below then falls through to the same call without images. That is
    the right behaviour -- an answer beats a refusal -- but if the caller
    recorded its own `len(images)` the judgment record would state that twenty
    frames were sent when none were, which is precisely the false artifact the
    count exists to prevent. The count travels back with the text.

    A call that RAISED reports zero for the same reason, and it is the same
    trap one step along: the frames left this process, but no verdict was
    reached on them -- the comparator falls through to `_decide_offline` -- so
    counting them would file a provider outage as a sighted judgment.
    """
    if client is None:
        return None, 0
    messages = [{"role": "user", "content": prompt}]
    kwargs: List[Dict[str, Any]] = [{"images": list(images)}] if images else []
    kwargs.append({})
    for name in _LLM_METHODS:
        fn = getattr(client, name, None)
        if not callable(fn):
            continue
        for payload in (prompt, messages):
            for kw in kwargs:
                try:
                    out = fn(payload, **kw)
                except TypeError:
                    continue  # wrong arity/shape: try the next convention
                except BudgetExceeded:
                    raise  # the loop owns this one; never swallow it
                except Exception as exc:
                    # ZERO, not `len(kw["images"])`, and the difference is the
                    # same false artifact.
                    # This call did not answer -- the comparator falls through
                    # to `_decide_offline` -- so no verdict was reached on those
                    # frames, and a nonzero count here would report the
                    # provider's failure as a successful sighted judgment. The
                    # second return value is what the ANSWERING call carried,
                    # and there was no answering call.
                    log.debug("preference judge call failed (%s): %s", name, exc)
                    return None, 0
                return _as_text(out), len(kw.get("images") or ())
    return None, 0


#: Serialises this module's direct `ctx.budget` mutations when comparisons run
#: concurrently (`parallel_safe` comparators under an evaluator that declares
#: `concurrent_samples`). `Budget` counters are unsynchronised `+=`s, and the
#: budget is a first-class output. Uncontended cost on the serial path is
#: nanoseconds.
_JUDGE_BUDGET_LOCK = threading.Lock()


def _query(ctx: Any, prompt: str, *, vlm: bool, role: str = "pair",
           images: Optional[Sequence[Any]] = None,
           trace: Optional[Dict[str, Any]] = None,
           parse: Optional[Callable[[Optional[str]], Any]] = None) -> Optional[str]:
    """One judge query, counted.

    The *comparison* is the unit GT pays for, so it is recorded here whether or
    not a provider answered -- a cost column that reads zero VLM calls for a
    method whose entire evaluation is VLM calls would be worse than useless.
    Token counts are a 4-chars-per-token estimate so the token column
    is not degenerate offline.

    This is also the single funnel every pairwise judge call goes through, which
    is why `output.judge_trace.record` is honoured here rather than at each
    comparator: pass `trace` and the prompt is written VERBATIM to the run's
    judge trace. Without it the pairwise judge's input would exist nowhere --
    the `_render` body naming clip lengths and file paths, and, for
    `llm_on_vlm_captions`, the contrastive caption that IS the decider's entire
    input. Counting the call and keeping nothing it said is what
    `bird/judgments.py` exists to prevent.

    `n_images` is written on EVERY query record, on the good path as well as the
    blind one, for the reason `fitness_vlm_score` journals its frame count
    either way: "no warning in the log" is not evidence that the judge had eyes.
    It is recorded
    here rather than by the caller because this is the function that knows what
    was actually handed to the client -- a comparator can intend to attach
    frames and be handed none.

    Two numbers, and both are kept. The query record is written BEFORE the call
    -- a record written only on the way back is missing for exactly the run that
    died mid-judgment -- so it states the INTENT; `_client_call` reports what the
    answering call actually carried, and a shortfall becomes a
    `judge_images_dropped` journal event rather than being left to be inferred
    from a client-side warning nobody reads. The pre-call record is not amended
    to match, because a judgments file is append-only and a rewritten record is
    a worse artifact than two honest ones.

    `role` is an ARGUMENT rather than a field inside `trace`, because `trace`
    is `None` unless `output.judge_trace.record` is on while the journal is
    always on: read out of the trace, the `judge_images_dropped` event below
    would say `role: "pair"` for every judge in every default-configured run, and the
    one thing that event has to say is which call lost its frames. `_answer`'s
    labels take it from the same argument, so a `response` record cannot
    disagree with the `query` record it answers about which judge spoke.

    Image tokens are deliberately NOT added to the estimate below. A real
    client bills them itself (`LLMClient._record` adds `IMAGE_TOKENS *
    n_images`), and the 4-chars-per-token figure here exists so the token column
    is not degenerate offline, where there are no images either.

    `parse` closes the other half of that. The ANSWER is recorded here too, in
    the same funnel and against the same `query_id`, because a query record with
    no matching response is exactly the gap this whole trace was built to remove
    -- a human can be shown this prompt, answer it, and have nothing to be
    compared against. The caller supplies its own reader (`_parse_choice` for a
    verdict, the caption's own text for step 1) because what counts as "the
    value" differs per role and this function must not learn the roles.

    The parse runs twice per call -- once here for the record, once in the
    caller that acts on it -- and that is deliberate rather than tolerated:
    every reader passed here is a pure function of the text, so the two cannot
    disagree, and threading a parsed value back out would change this
    function's return type at four call sites to save one regex match.
    """
    n_images = len(images or ())
    query_id = ""
    fields: Dict[str, Any] = {}
    if trace is not None:
        fields = dict(trace)  # never mutate the caller's dict
        fields.setdefault("n_images", n_images)
        query_id = judgments.note_query(ctx, role=role, prompt=prompt,
                                        vlm=vlm, **fields)

    def _answer(raw: Optional[str], parsed: Any, failure: str = "") -> None:
        """File the answer against the query above. Every exit goes through it.

        A closure rather than two call sites, because the two exits are the
        normal one and an exception one and the invariant is that they cannot
        differ: a query record with no response is indistinguishable from a
        judge call that is still in flight, so EVERY path out of this function
        past `note_query` has to leave one behind.
        """
        if trace is None:
            return
        # The role again on the answer, as on every other response record: a
        # reader filtering `kind: response` for "what did the captioner say"
        # must not have to join back to the query to find out which judge spoke.
        # From the ARGUMENT, not from `trace` -- the role does not travel in
        # the trace dict (see above), so reading it there would label every
        # response `pair`.
        labels: Dict[str, Any] = {
            "role": role,
            "left_id": trace.get("left_id", ""),
            "right_id": trace.get("right_id", ""),
        }
        # `repeat` ONLY when the caller has one. `_decide` adds it per vote;
        # `pair_caption` is a single call with no repeats, and defaulting it to
        # -1 there would invent an index -- which `judgments.query_id`'s own
        # docstring rules out in as many words: a value is a positive claim, not
        # a way of saying nothing. Absent is not empty, on a label as much as on
        # a frame set.
        if "repeat" in trace:
            labels["repeat"] = trace["repeat"]
        judgments.note_response(
            ctx, iteration=int(fields.get("iteration", -1)), query_id=query_id,
            raw=raw, parsed=parsed, vlm=vlm, failure=failure, **labels)

    try:
        with _JUDGE_BUDGET_LOCK:
            # Reserve, then record: under concurrent judging every in-flight and
            # queued comparison would otherwise increment past
            # `budget.max_llm_calls` before the first BudgetExceeded surfaced to
            # stop the pool -- the artifact would overshoot the cap by up to the
            # whole plan. Checking exhaustion BEFORE recording keeps the
            # overshoot to the calls already in flight, and a queued task fails
            # here without bumping a counter.
            #
            # The trace write above is deliberately OUTSIDE this lock: it is
            # best-effort and file-local (`judgments._LOCK` serialises it), and
            # holding the budget lock across a write would serialise the very
            # round-trips `judge_concurrency` exists to overlap.
            if ctx.budget.exhausted():
                raise BudgetExceeded("llm call budget exhausted before this comparison")
            # RESERVE one call before the round-trip, so a concurrent flock of
            # queued comparisons cannot each slip past the cap before the first
            # BudgetExceeded surfaces. The estimate is ALSO the charge for a
            # judge that does not account for itself (a duck-typed callable) --
            # but a real LLMClient records its OWN exact usage inside
            # `_client_call`, and charging both would double-count every GT
            # judge round-trip (one call, one record). Snapshot so the
            # reservation can be rolled back when the client recorded.
            _calls_reserved = ctx.budget.llm_calls
            ctx.budget.record_llm(prompt_tokens=len(prompt) // 4,
                                  completion_tokens=0, vlm=vlm)
        # Inside the same `try` as the reservation, not after it: `_client_call`
        # re-raises BudgetExceeded by design ("the loop owns this one"), so both
        # the refusal before the call and the one during it leave through here.
        text, sent = _client_call(ctx.evaluator, prompt, images)
    except BudgetExceeded as exc:
        # The query record is already written, so the refusal has to be written
        # too: an orphaned query reads as a call still in flight, and a run that
        # ended on an exhausted budget is precisely the one somebody will audit.
        # Re-raised unchanged -- a trace must never swallow it.
        _answer(None, None, failure=f"the llm call budget ended this comparison: {exc}")
        raise
    # `> _calls_reserved + 1`: our own reservation moved the counter by one;
    # anything beyond that is the client accounting for the SAME call, so the
    # reservation was a double-count and is undone -- only the client's exact
    # figures remain.
    _client_self_recorded = ctx.budget.llm_calls > _calls_reserved + 1
    if _client_self_recorded:
        with _JUDGE_BUDGET_LOCK:
            ctx.budget.llm_calls -= 1
            ctx.budget.llm_prompt_tokens -= len(prompt) // 4
            if vlm:
                ctx.budget.vlm_calls -= 1
    if text and sent < n_images:
        # The client answered, but not the call that carried the frames. Named,
        # because the alternative is a query record claiming twenty images
        # against a judgment made on none -- indistinguishable from a sighted
        # one, which is the state this event exists to prevent.
        #
        # `text and`, not `sent < n_images` alone, because this event makes ONE
        # claim -- "a verdict was reached, and it was reached without the
        # pixels" -- and a provider that raised or returned nothing reached no
        # verdict at all: the comparator falls through to `_decide_offline`,
        # which is a different degradation with a different remedy (a dead
        # provider, not a client that cannot take `images=`). Firing here on an
        # outage would put the two in one stream under the name of the rarer
        # one, and the count of blind-but-answered comparisons is the number
        # this event exists to make countable. An outage is not silent either:
        # `_client_call` logs it, the query record stands with its `n_images`,
        # and `_answer` files a response with no text against that same
        # `query_id` -- which is the row that says a call was made and returned
        # nothing. (`blind_comparisons` says nothing about it: it counts sides
        # whose PIXELS were never cut, and here they were.)
        #
        # The JOURNAL, not a second judgments record: `kind: query` means "a
        # judge was asked this", and a row with an empty prompt standing for a
        # degradation would have to be excluded by every reader of that stream.
        # `budget.blind_comparisons` is left alone for the same reason -- it
        # counts sides of comparisons, and mixing a per-query event into it
        # would make its units unreadable.
        log.warning("the preference judge answered without %d of %d attached "
                    "image(s); this comparison was decided on text",
                    n_images - sent, n_images)
        event = getattr(ctx, "event", None)
        if callable(event):
            event("judge_images_dropped", role=role,
                  intended=n_images, sent=sent)
    if text and not _client_self_recorded:
        # Only estimate completion tokens for a judge that recorded nothing; a
        # self-recording client already logged its own exact count.
        with _JUDGE_BUDGET_LOCK:
            ctx.budget.llm_completion_tokens += len(text) // 4
    # AFTER the budget bookkeeping and outside its lock, for the trace write's
    # reason above: this is a file-local best-effort append and holding the
    # budget lock across it would serialise the round-trips `judge_concurrency`
    # exists to overlap.
    #
    # Recorded even when `text` is None. A comparison the provider never
    # answered still cost a counted call and still routed the comparator to
    # `_decide_offline`, and a missing record would read as a query nobody got
    # round to asking -- the one thing the join must not be ambiguous about.
    try:
        # No reader supplied means the answer IS its own value -- otherwise a
        # perfectly good reply would be filed as `unparsed`, which names a
        # provider failure that did not happen.
        parsed = parse(text) if parse is not None else text
    except Exception:  # noqa: BLE001 - a reader that throws is not a lost run
        log.debug("a judge-trace parser raised", exc_info=True)
        parsed = None
    _answer(text, parsed)
    return text


_STRONG = re.compile(
    r"\b(left|right|agent\s*a|agent\s*b|option\s*a|option\s*b|tie|draw|equal|equivalent)\b",
    re.IGNORECASE)
_BARE = re.compile(r"^\W*(a|b|1|2|left|right)\W*$", re.IGNORECASE)


def _parse_choice(text: Optional[str]) -> Optional[int]:
    """Pull a verdict out of free prose. `None` means "no verdict found"."""
    if not text:
        return None
    hits = _STRONG.findall(text)
    if hits:
        # The last decisive token wins: judges reason first and conclude last.
        tok = re.sub(r"\s+", "", hits[-1]).lower()
        if tok in ("left", "agenta", "optiona"):
            return 1
        if tok in ("right", "agentb", "optionb"):
            return 0
        return TIE
    bare = _BARE.match(text.strip())
    if bare:
        tok = bare.group(1).lower()
        return 1 if tok in ("a", "1", "left") else 0
    return None


def _caption_value(text: Optional[str]) -> Optional[str]:
    """A caption's parsed value: itself, stripped, or None if there is none.

    `comparator_llm_on_vlm_captions` treats a blank caption as a failed step 1
    and decides offline rather than falling through to the one-step comparator,
    which would quietly turn the config into `vlm`. The judge trace has to draw
    the same line: `parsed: null` says the captioner produced nothing usable,
    which is a different record from a caption that happens to be short.
    """
    return text.strip() if text and text.strip() else None


def _fallback_decision(ctx: Any, left: CandidateReport, right: CandidateReport,
                       *, role: str, reason: str) -> int:
    """What a comparison returns when the judge produced nothing usable.
    `provider: mock` -> the deterministic offline stand-in (no real judge exists,
    and the tester tier must still exercise the fold), recorded
    `fallback=offline_synthetic`. Any real provider -> `ABSTAIN`, recorded
    `fallback=abstain`: the pair leaves D_pref rather than being decided by the
    agents' own reward scale, which is neither a VLM preference nor a
    behavioural substitute. Different reward programs share no calibrated
    scale, so `mean_per_step_return` cannot rank them here."""
    is_mock = str(ctx.cfg.get("llm.evaluator.provider")) == "mock"
    ctx.event("judge_fallback", role=role, left=left.cand_id, right=right.cand_id,
              reason=reason, fallback="offline_synthetic" if is_mock else "abstain")
    if is_mock:
        return _decide_offline(ctx, left, right)
    return ABSTAIN


def _decide_offline(ctx: Any, left: CandidateReport, right: CandidateReport) -> int:
    """The forced decision, from exactly the evidence the judge was shown.

    `allow_ties: false` (GT's deliberate choice, App. C -- neurips_2025.tex:720
    "The LLM was not given the option to provide ties", where the forced choice
    is the LLM's) means a comparison must
    return a verdict, so an unparseable or absent judgment cannot abstain. The
    ordering is (task outcome, length-corrected clip return, id) -- outcome
    first because a human watching the clip also sees whether the lap finished,
    mean-per-step return second because that is the statistic TAC re-scores,
    and the id last so the tie-break consumes no RNG and stays reproducible
    across pair strategies.
    """
    def key(report: CandidateReport) -> Tuple[int, float]:
        clip = _clip_for(ctx, report)
        if clip is None:
            return (0, float("-inf"))
        return (1 if clip.success else 0, float(clip.mean_per_step_return))

    kl, kr = key(left), key(right)
    if kl != kr:
        return 1 if kl > kr else 0
    return 1 if left.cand_id <= right.cand_id else 0


def _decide(ctx: Any, prompt: str, *, vlm: bool, role: str = "pair",
            images: Optional[Sequence[Any]] = None,
            trace: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Ask, optionally several times, and take the majority verdict.

    `evaluate.vlm.repeats` is the "score each artifact k times and average"
    knob; the pairwise analogue of averaging is a majority vote. † The RDA
    claim behind that key (each video scored 4x) is unverified, so binding it
    here is this repo's reading, not a pin. The default is 1 and costs nothing.
    """
    repeats = max(1, _int(ctx.cfg.get("evaluate.vlm.repeats"), 1))
    votes: List[int] = []
    for repeat in range(repeats):
        # One trace record per REPEAT, not per decision: a majority vote is
        # `repeats` separate queries to the provider, and collapsing them into
        # one record would make a 2-1 verdict indistinguishable from a 3-0 one
        # in the artifact.
        # `parse=_parse_choice` so the response record carries the VERDICT this
        # vote contributed, not only the prose it was read out of. A majority
        # over three unparseable answers and a 2-1 split both leave one
        # comparison in `report.json`; only the per-query parsed value tells
        # them apart, and only it can be compared against a human's.
        got = _parse_choice(_query(ctx, prompt, vlm=vlm, role=role, images=images,
                                   trace=dict(trace, repeat=repeat)
                                   if trace is not None else None,
                                   parse=_parse_choice))
        if got is not None:
            votes.append(got)
    if not votes:
        return None
    left = votes.count(1)
    right = votes.count(0)
    if left == right:
        return TIE if TIE in votes or left == 0 else None
    return 1 if left > right else 0


def _sides(ctx: Any, left: CandidateReport, right: CandidateReport,
           *, attach: bool) -> Tuple[str, List[bytes]]:
    """Both agents as the judge receives them: one body of text, one image list.

    The images are a FLAT list, because that is what `LLMClient(images=...)`
    takes and what `AnthropicClient._attach_images` turns into an ordered run of
    image blocks ahead of the text. Order is therefore the only thing that says
    whose clip is whose, so when there are images the body carries a key naming
    the split. Without it the judge is handed forty unlabelled frames of two
    agents doing almost the same thing and asked which agent was better, which
    is not a harder version of the question -- it is a different one, and its
    answer is noise. The counts come from the lists actually built, never from
    `images_per_query`: a side whose render half-failed contributes fewer.

    **One side may have pixels and the other none, and that comparison is still
    sent.** It is worth stating because the alternative -- drop both when either
    fails -- is tempting and wrong in a way that costs evidence: a comparison
    where only LEFT can be seen is genuinely a third condition, neither the
    video one nor the text one, and it would bias a *pairwise* verdict rather
    than merely weaken it. The answer is to make it legible, not to normalise it
    away or to throw away the good side: the body says which side has none,
    `_count_blind` fires for it, and the trace records `n_images` per side. A
    reader can exclude those comparisons; a reader cannot recover a dropped one.
    """
    ltext = _render(ctx, left, "LEFT", attach=attach)
    rtext = _render(ctx, right, "RIGHT", attach=attach)
    if not attach:
        return ltext + "\n\n" + rtext, []
    lpng, lprov = _pair_frames(ctx, left)
    rpng, rprov = _pair_frames(ctx, right)
    body = ltext + "\n\n" + rtext
    lcells, rcells = sheet_cells(lprov), sheet_cells(rprov)
    if (lpng or rpng) and (lcells or rcells):
        # A sheet is one image tiling many frames; counted as "1" without saying
        # so, the judge would be told one frame was attached per side. The
        # non-sheet sentence below is unaffected.
        def _what(png: List[bytes], cells: Optional[int]) -> str:
            return (f"one contact sheet tiling {cells} frames" if cells
                    else f"{len(png)} evenly-spaced frame(s)")
        body += (f"\n\nATTACHED IMAGES, in order: the first {len(lpng)} "
                 f"({_what(lpng, lcells)}) are LEFT's clip, the next {len(rpng)} "
                 f"({_what(rpng, rcells)}) are RIGHT's. The frames behind them are "
                 f"evenly spaced over the same window of each agent's episode; a "
                 f"sheet's cells are labelled with the EPISODE step, as its manifest "
                 f"above lists.")
    elif lpng or rpng:
        body += (f"\n\nATTACHED IMAGES, in order: the first {len(lpng)} are "
                 f"LEFT's clip, the next {len(rpng)} are RIGHT's. Both are "
                 f"evenly spaced over the same window of each agent's episode.")
    return body, list(lpng) + list(rpng)


def _pair_prompt(ctx: Any, left: CandidateReport, right: CandidateReport, *,
                 body: str, docs: str = "") -> str:
    """The judge's question around `body`. `docs` is `_variables_documentation`
    for a body that carries trajectory rows (the paper puts it once, before the
    first clip -- tex:682-684); the decide step of `llm_on_vlm_captions` passes
    none, because its body is a caption and has no rows to document."""
    cfg = ctx.cfg
    task = cfg.get("problem.task_description") or "(no task description)"
    ties = bool(cfg.get("evaluate.preferences.allow_ties"))
    closing = ("Answer with exactly one word: LEFT, RIGHT, or TIE."
               if ties else
               "You must pick one. Ties are not allowed. "
               "Answer with exactly one word: LEFT or RIGHT.")
    return (
        "Two agents attempted the same task. Judge them on the task "
        "description alone.\n\n"
        f"TASK: {task}\n\n"
        + (f"{docs}\n\n" if docs else "")
        + f"{body}\n\n"
        "Which agent better fulfils the task description?\n"
        f"{closing}"
    )


def _pair_trace(ctx: Any, state: Any, left: CandidateReport,
                right: CandidateReport,
                *, attach: bool = False) -> Optional[Dict[str, Any]]:
    """The identity of one head-to-head, or None when nothing is being recorded.

    Each side is a POINTER: the candidate id, its clip's frame count, whether it
    had a rollout at all, and -- when this comparator attaches pixels -- the id
    of the frame set it was handed. The heavy provenance is one `frames` record
    per candidate, written where the frames are cut and joined on
    `(cand_id, iteration)`; repeating a 150-element step list and twenty content
    hashes on every pair and every `evaluate.vlm.repeats` vote is exactly the
    duplication that join exists to avoid.

    Two frame-set flavours, and they are different claims rather than two
    spellings of one. `judgments.clip_set` describes the clip the judge was
    shown IN WORDS -- `pixels: false`, no digest, only which episode instants
    the window covers. `judgments.frame_set` describes the pixels a judge was
    actually handed, with a content digest that makes "these two queries saw
    byte-identical frames" a checkable statement.
    """
    if not judgments.mode(ctx):
        return None

    def side(report: CandidateReport) -> Dict[str, Any]:
        # `_clip_for` is what WRITES the clip record (and caches it), so calling
        # it here is what guarantees there is one to point at -- and it is the
        # same call the comparator is about to make, so it costs nothing.
        clip = _clip_for(ctx, report)
        out = {
            "cand_id": report.cand_id,
            "n_frames": _n_frames(clip) if clip is not None else 0,
            # `""` when the side had a rollout. GT's judge is shown a clip; a
            # side with none is judged on `_render`'s "(no rollout was recorded
            # for this agent)" line, which is the pairwise analogue of a blind
            # judgment and must be as legible in the artifact.
            "blind_reason": "" if clip is not None else "no rollout was recorded",
        }
        if attach:
            png, prov = _pair_frames(ctx, report)
            out["frame_set"] = str(prov.get("id") or "")
            out["n_images"] = len(png)
            # The frames' reason WINS over the clip's when there is one: a side
            # that had a rollout and still reached the judge blind is the case
            # worth naming, and `sample_frames` knows which of its sources
            # failed while this function only knows that none did.
            out["blind_reason"] = str(prov.get("blind_reason") or out["blind_reason"])
        return out

    return {
        "iteration": int(getattr(state, "iteration", -1)),
        "left_id": left.cand_id,
        "right_id": right.cand_id,
        "left_clip": side(left),
        "right_clip": side(right),
    }


# ==========================================================================
# comparator -- who decides one head-to-head
# ==========================================================================


@register("comparator", "vlm")
def comparator_vlm(ctx: Any, state: Any, left: CandidateReport,
                   right: CandidateReport) -> int:
    """Single VLM call sees both clips and decides (GT, §4, simplified).

    This is the one-step comparator: the clips and the verdict live in the same
    context. GT does *not* do this -- it cannot, the clips do not fit -- so this
    member exists as the ablation against `llm_on_vlm_captions`, isolating the
    caption bottleneck from the judgment itself. "The clips do not fit" is a
    claim about pixels, so both members send them; otherwise the ablation
    would compare one text prompt against two.
    """
    body, images = _sides(ctx, left, right, attach=True)
    trace = _pair_trace(ctx, state, left, right, attach=True)
    prompt = _pair_prompt(ctx, left, right, body=body, docs=_variables_documentation(ctx))
    verdict = _decide(ctx, prompt, vlm=True, role="pair_vlm", images=images, trace=trace)
    if verdict is None:
        return _fallback_decision(ctx, left, right, role="pair_vlm",
                                  reason="no parseable verdict")
    return verdict


#: One comparison touches nothing shared but the budget (locked above) and the
#: per-report clip and frame caches (warmed before the pool; identical on a
#: racing write), so independent pairs may be judged in flight together.
#: Warming the FRAMES outside the pool is not an optimisation: `sample_frames`
#: drives the shared env renderer, and two threads on one MuJoCo adapter
#: segfault 3/3. `run_preferences` does it.
comparator_vlm.parallel_safe = True  # type: ignore[attr-defined]

#: This comparator sends the clip's PIXELS, so `run_preferences` renders every
#: pooled candidate's frames before it opens the pool. Declared as an attribute
#: for `parallel_safe`'s reason: the phase has to know a property of the
#: component it was handed, and the alternative is a branch on the component's
#: NAME, which is the one thing this repo does not have.
comparator_vlm.attaches_frames = True  # type: ignore[attr-defined]


@register("comparator", "llm_on_vlm_captions")
def comparator_llm_on_vlm_captions(ctx: Any, state: Any, left: CandidateReport,
                                   right: CandidateReport) -> int:
    """VLM captions both clips contrastively, then a text LLM decides (GT, §4).

    GT's actual design, and a context-length workaround rather than a modelling
    choice: the two clips will not fit in the decider's context, so a VLM is
    asked to describe *both agents in one pass, side by side* -- contrastive
    captioning, so the descriptions are commensurable -- and a separate text
    model reads only that description. Step 2's prompt therefore carries the
    caption and the task, and never the clips; if it saw them the component
    would collapse into `vlm` and the ablation between the two would measure
    nothing.

    ‡ The schema exposes one evaluator client (`llm.evaluator.*`), so both
    steps use `ctx.evaluator`; GT uses a different text model for step 2. That
    is a missing knob (a second evaluator role), not a behaviour to branch on,
    and it is recorded here rather than worked around.

    † RECORDED DIVERGENCE IN THE CAPTION INSTRUCTION.
    The `caption_prompt` literal below ends "Do not say which is better" -- split
    across two string literals ("...Do not say which " / "is better.\n\n"), which
    is why a grep for the whole phrase finds only this docstring. Cite it by
    symbol, not by line: a line number inside this file moves whenever this
    docstring does. **That sentence is OURS**, not the paper's (checked
    against `refs/tex/`).

    What diverges is the VERDICT, not the comparison. Our prompt does ask for a
    "single contrastive paragraph ... so a reader who cannot see the clips can
    compare them"; it withholds only the ranking. GT withholds neither:

    * `refs/tex/gt_reward_design/neurips_2025.tex:690` -- "Describe and compare
      in detail any and all of the agents' behaviors in each clip as it pertains
      to the desired goal", with `:696` requiring the answer to end in "a
      comparison between the two clips".
    * `:720` is the sharp one, because it states the PURPOSE: "the VLM was
      always fed two clips side-by-side for more accurate descriptions of an
      agent's behavior, since inaccuracies are not as important as getting a
      relative sense of which agent was better." A relative sense of which agent
      was better is exactly what our last sentence forbids.
    * The only objectivity ask is `:694-695`, "Remain as objective as possible
      for all descriptions, as to not bias anyone towards whether the goal was
      achieved or not for each agent" -- about not prejudging the GOAL, not
      about withholding a verdict between the two agents.

    A defensible reading of our sentence is that it keeps the captioner from
    pre-empting the judge, which is what makes the `vlm` /
    `llm_on_vlm_captions` ablation measure anything. That may be the right
    engineering call; it is not what the paper does. Left unchanged on
    purpose: altering the prompt of a published method point is a fidelity
    change that moves every GT result.
    """
    body, images = _sides(ctx, left, right, attach=True)
    task = ctx.cfg.get("problem.task_description") or "(no task description)"
    docs = _variables_documentation(ctx)
    caption_prompt = (
        "You are annotating two clips of different agents attempting the same "
        "task. Describe BOTH agents in a single contrastive paragraph: use the "
        "same sentence structure for each so a reader who cannot see the clips "
        "can compare them. Refer to them as LEFT and RIGHT. Do not say which "
        "is better.\n\n"
        f"TASK: {task}\n\n"
        # App. C: "Documentation about each frame's data is given here:
        # {variables_documentation}" precedes "{frames_and_observations_1}"
        # (tex:682-684). Empty unless the rows are shown.
        + (f"{docs}\n\n" if docs else "")
        + f"{body}"
    )
    trace = _pair_trace(ctx, state, left, right, attach=True)
    # Step 1 is the ONLY step that gets the pixels, and step 2 below passes none
    # -- not by omission but because a decider that could see the clips would
    # collapse this component into `vlm` and the ablation between the two would
    # measure nothing. `_query`'s `images` defaults to None; it is written out
    # at the step-2 call anyway, because the absence is the design.
    #
    # Step 1 parses nothing into a VERDICT: the caption is not a verdict to be
    # read out of prose, it is the whole product, and step 2's entire input.
    # `_caption_value` keeps the distinction the record needs anyway -- a
    # provider that returned only whitespace produced no caption, and
    # `_decide_offline` below treats it as such, so the trace must not file it
    # as a successful answer.
    caption = _query(ctx, caption_prompt, vlm=True, role="pair_caption",
                     images=images, trace=trace, parse=_caption_value)

    if not caption or not caption.strip():
        # No caption: the decider has no clip description to read, and falling
        # through to the one-step comparator would quietly turn this config into
        # `vlm`. Abstain instead of deciding by reward scale.
        return _fallback_decision(ctx, left, right, role="pair_caption",
                                  reason="no caption")

    # These are App. C's contrastive clip descriptions -- the artifact App. D's
    # no-human feedback re-reads ("following the agent descriptions obtained in
    # Appendix C"). Plain strings, deliberately NOT in
    # `_drop_clips`'s pop list: the persistence is the point, and they ride
    # `report.meta` into `report.json`, `RunState` and the checkpoints, where
    # `evaluation._section_analyzer` summarises them. `list.append` is atomic
    # under the judge ThreadPoolExecutor's GIL; the order two racing pairs
    # land in is not deterministic under a concurrent live judge, so consumers
    # sort by partner id before reading.
    for rep, side, other in ((left, "LEFT", right), (right, "RIGHT", left)):
        rep.meta.setdefault("pair_captions", []).append(
            {"other": other.cand_id, "side": side, "caption": caption.strip()})

    decide_prompt = _pair_prompt(
        ctx, left, right,
        body=("You cannot see the clips. A video annotator watched both and "
              "wrote this description:\n\n" + caption.strip()))
    # The caption as its OWN field, not only embedded in the prompt it is pasted
    # into. It is the decider's entire input -- step 2 sees the caption and the
    # task and never the clips -- so a reader auditing this comparison, or a
    # tool reconstructing its caption-only condition, must be able to take it
    # without parsing prose back out of a template.
    verdict = _decide(ctx, decide_prompt, vlm=False, role="pair_decide",
                      images=None,  # the decider never sees the clips (§4)
                      trace=dict(trace, caption=caption.strip()) if trace else None)
    if verdict is None:
        return _fallback_decision(ctx, left, right, role="pair_decide",
                                  reason="no parseable verdict")
    return verdict


#: The caption->decide sequence WITHIN one pair stays ordered on its thread;
#: only distinct pairs overlap.
comparator_llm_on_vlm_captions.parallel_safe = True  # type: ignore[attr-defined]

#: Step 1 is a VLM looking at two clips, so the frames are rendered before the
#: pool opens, exactly as for `comparator_vlm`. Step 2 gets none by design.
comparator_llm_on_vlm_captions.attaches_frames = True  # type: ignore[attr-defined]


@register("comparator", "human")
def comparator_human(ctx: Any, state: Any, left: CandidateReport,
                     right: CandidateReport) -> int:
    """A person labels the pair (GT keeps a human in the loop; T2R-human, §4).

    GT is explicit that "all human feedback is collected from one of the
    authors" and caps free-text feedback at one query per iteration; the number
    of *comparisons* a human is asked for is set by the pair strategy instead,
    which is why this component does not read
    `evaluate.human.queries_per_iteration` -- that key belongs to the
    human-feedback phase.

    The oracle behind `ctx.human` is scripted, interactive, or absent
    (`registry.get("phase", "human_oracle")`); its call convention belongs to
    that module, so it is duck-typed here and degrades to the offline rule.
    Human queries are counted separately from LLM calls -- they are the
    scarcest resource any of these methods spends -- and THE ORACLE CHARGES
    THEM, not this function: it is the one place that knows a person was
    actually asked (`phases._ScriptedHumanOracle`), and a comparison nobody
    answered, which falls to `_decide_offline` below, is not a human query.
    Charging `budget.record_human()` here as well would count every comparison
    twice (a human-arm tester run would report `human_queries: 512` for 240
    comparisons + 32 feedback queries), on the counter that is the only
    measurement of REvolve's central cost claim.
    """
    # `attach=False`, and deliberately: this judge is a person at a terminal, so
    # the artifact that reaches them is the file path `_render` prints, not a
    # base64 image block on an API call nobody is making.
    body, _images = _sides(ctx, left, right, attach=False)
    prompt = _pair_prompt(ctx, left, right, body=body, docs=_variables_documentation(ctx))

    oracle = ctx.human
    answer: Any = None
    if oracle is not None:
        for name, args in (("compare", (left, right)), ("preference", (left, right)),
                           ("ask", (prompt,)), ("query", (prompt,)), ("__call__", (prompt,))):
            fn = getattr(oracle, name, None)
            if not callable(fn):
                continue
            try:
                answer = fn(*args)
            except TypeError:
                continue
            except BudgetExceeded:
                raise
            except Exception as exc:  # pragma: no cover - oracle-specific
                log.debug("human oracle call failed (%s): %s", name, exc)
                answer = None
            break

    if isinstance(answer, bool):
        return 1 if answer else 0
    if isinstance(answer, int):
        return 1 if answer == 1 else (TIE if answer == TIE else 0)
    verdict = _parse_choice(_as_text(answer))
    if verdict is None:
        return _decide_offline(ctx, left, right)
    return verdict


# ==========================================================================
# pair_strategy -- which head-to-heads get run
# ==========================================================================


def _all_pairs(n: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _subset_budget(n: int, total: int) -> int:
    """Comparison budget for the sub-linear strategies.

    ‡ The schema has no pair-count knob, so the budget is fixed here at one
    comparison per agent -- linear in the pool where all-pairs is quadratic.
    `random_subset` and `active` share it deliberately: holding the budget
    equal is what makes the two a clean ablation of *which* comparisons to buy
    (the Bayesian-experimental-design direction). If a config ever needs to
    set this, the fix is a schema key, not a branch here.
    """
    return max(1, min(total, n))


@register("pair_strategy", "all_pairs")
def pairs_all(ctx: Any, state: Any, reports: List[CandidateReport]) -> List[Tuple[int, int]]:
    """Every unordered pair, once per iteration (GT, §4).

    N-choose-2 over the trained pool (`_eligible`): 10 per iteration for GT,
    whose alignment filter keeps 5 of the 10 written (§5.3: "only collected 10
    preferences per iteration"). Quadratic cost is exactly why the other three
    members of this family exist.
    """
    return _all_pairs(len(reports))


@register("pair_strategy", "random_subset")
def pairs_random_subset(ctx: Any, state: Any,
                        reports: List[CandidateReport]) -> List[Tuple[int, int]]:
    """A uniform random sample of pairs -- the null baseline for `active`.

    Published by nobody. Its only job is to be the control condition: same
    budget as `active`, comparisons chosen without looking at the posterior, so
    any difference is attributable to the design rule and not to spend.
    """
    pairs = _all_pairs(len(reports))
    if not pairs:
        return []
    k = _subset_budget(len(reports), len(pairs))
    if k >= len(pairs):
        return pairs
    return ctx.rng.sample(pairs, k)


@register("pair_strategy", "tournament")
def pairs_tournament(ctx: Any, state: Any,
                     reports: List[CandidateReport]) -> List[Tuple[int, int]]:
    """A seeded knockout bracket: N-1 comparisons instead of N-choose-2.

    Published by nobody; a reachable point of this family. Round r pairs the
    leaders of adjacent blocks of size 2^r, which is a bracket's schedule and a
    bracket's cost.

    ‡ A genuinely adaptive bracket -- round r+1's pairings determined by round
    r's *results* -- cannot be expressed under the frozen
    `pair_fn(ctx, state, reports) -> [(i, j)]` signature, which must return
    every pair before any comparison runs. So the block leader stands in for
    the winner of that block: the bracket is seeded by input order rather than
    resolved. The resulting comparison graph is still connected (index 0
    appears in every round), which is what Bradley-Terry needs to identify a
    single ranking.
    """
    n = len(reports)
    pairs: List[Tuple[int, int]] = []
    step = 1
    while step < n:
        for base in range(0, n, 2 * step):
            if base + step < n:
                pairs.append((base, base + step))
        step *= 2
    return pairs


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    return math.exp(max(x, -60.0)) / (1.0 + math.exp(max(x, -60.0)))


def _laplace_bt(n: int, obs: Sequence[Tuple[int, int, float]],
                tau2: float = 1.0, iters: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """Laplace posterior over Bradley-Terry log-strengths.

    `obs` is a list of (winner, loser, count). The prior N(0, tau2 I) both
    identifies the model (BT is shift-invariant) and keeps the Hessian positive
    definite on disconnected data, so Newton never has to be guarded. Returns
    (theta_MAP, Sigma) with Sigma the inverse Hessian at the mode.
    """
    theta = np.zeros(n, dtype=float)
    hess = np.eye(n) / tau2
    for _ in range(iters):
        grad = -theta / tau2
        hess = np.eye(n) / tau2
        for a, b, w in obs:
            p = _sigmoid(float(theta[a] - theta[b]))
            grad[a] += w * (1.0 - p)
            grad[b] -= w * (1.0 - p)
            v = w * p * (1.0 - p)
            hess[a, a] += v
            hess[b, b] += v
            hess[a, b] -= v
            hess[b, a] -= v
        step = np.linalg.solve(hess, grad)
        theta = theta + step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return theta, np.linalg.inv(hess)


@register("pair_strategy", "active")
def pairs_active(ctx: Any, state: Any,
                 reports: List[CandidateReport]) -> List[Tuple[int, int]]:
    """Pick comparisons by expected information gain under the BT posterior.

    Published by nobody, and no shipped config selects it, so no config
    exercises it end to end. Together with `select.allocation:
    bayesian_experimental_design` it is an unexplored axis, so it is
    implemented as real Bayesian experimental design rather than as a
    heuristic:

    1. Fit a Laplace posterior N(theta, Sigma) over log-strengths from whatever
       preferences already mention these ids.
    2. Score each candidate comparison x = e_i - e_j by the Gaussian EIG
       approximation `0.5 * log(1 + w * x' Sigma x)`, w = p(1-p) at the mode --
       the entropy the observation is expected to remove.
    3. Take the argmax, then fold its *expected* Fisher information w x x' into
       Sigma by Sherman-Morrison and repeat. The expected information of a
       logistic observation does not depend on its outcome, which is precisely
       what licenses choosing a whole batch before observing any of it -- the
       classical D-optimal batch-design argument, and the reason this works
       under the frozen signature that defeated `tournament`.

    Step 1 will usually find nothing: `state.preferences` accumulates across
    iterations but its ids are the previous rounds' candidates, and this round's
    are fresh. That is not a bug -- it is GT's per-round renormalisation
    boundary (§5, `select.scope`) showing up in the design problem. Starting
    from the prior, the greedy still spreads its budget over a balanced design
    rather than clustering.
    """
    n = len(reports)
    pairs = _all_pairs(n)
    if not pairs:
        return []
    k = _subset_budget(n, len(pairs))
    if k >= len(pairs):
        return pairs

    ids = [r.cand_id for r in reports]
    index = {cid: i for i, cid in enumerate(ids)}
    counted: Dict[Tuple[int, int], float] = {}
    for pref in (getattr(state, "preferences", None) or []):
        a = index.get(pref.left_id)
        b = index.get(pref.right_id)
        if a is None or b is None or a == b:
            continue
        key = (a, b) if pref.label == 1 else (b, a)
        counted[key] = counted.get(key, 0.0) + 1.0
    obs = [(a, b, w) for (a, b), w in sorted(counted.items())]

    theta, sigma = _laplace_bt(n, obs)
    chosen: List[Tuple[int, int]] = []
    remaining = list(pairs)
    for _ in range(k):
        best_pair: Optional[Tuple[int, int]] = None
        best_gain = float("-inf")
        best_vec: Optional[np.ndarray] = None
        best_w = 0.0
        for (i, j) in remaining:
            x = np.zeros(n)
            x[i], x[j] = 1.0, -1.0
            var = float(x @ sigma @ x)
            p = _sigmoid(float(theta[i] - theta[j]))
            w = p * (1.0 - p)
            gain = 0.5 * math.log1p(max(w * var, 0.0))
            if gain > best_gain + 1e-12:  # strict: ties keep lexicographic order
                best_gain, best_pair, best_vec, best_w = gain, (i, j), x, w
        if best_pair is None or best_vec is None:
            break
        chosen.append(best_pair)
        remaining.remove(best_pair)
        sx = sigma @ best_vec
        denom = 1.0 + best_w * float(best_vec @ sx)
        if denom > 1e-12:
            sigma = sigma - (best_w / denom) * np.outer(sx, sx)
    return chosen


# ==========================================================================
# pref_aggregator -- a bag of comparisons becomes a ranking
# ==========================================================================


def _tally(ids: Sequence[str], prefs: Sequence[Preference]) -> Tuple[np.ndarray, np.ndarray]:
    """(wins per id, comparison counts per unordered pair)."""
    n = len(ids)
    index = {cid: i for i, cid in enumerate(ids)}
    wins = np.zeros(n, dtype=float)
    counts = np.zeros((n, n), dtype=float)
    for pref in prefs:
        a = index.get(pref.left_id)
        b = index.get(pref.right_id)
        if a is None or b is None or a == b:
            continue
        wins[a if pref.label == 1 else b] += 1.0
        counts[a, b] += 1.0
        counts[b, a] += 1.0
    return wins, counts


def _normalise(ids: Sequence[str], strength: np.ndarray) -> Dict[str, float]:
    """Renormalise to a simplex over `ids`.

    GT renormalises strengths per round and never compares them across rounds,
    so under the `within_iteration` default every aggregator returns a
    distribution over *this* round's candidates and nothing that survives the
    iteration boundary.

    **`ids` is therefore the scale, and `evaluate.preferences.scope` chooses
    it.** Under `cumulative` the caller hands in every individual with a game on
    record, so the same function returns a distribution over the whole history
    and the numbers DO cross the boundary -- that is the point of the key, and
    it is also why a strength is only ever comparable to another strength from
    the same call. Nothing here changes; the widening is entirely in what the
    caller passes.
    """
    strength = np.asarray(strength, dtype=float)
    strength = np.where(np.isfinite(strength), strength, 0.0)
    strength = np.clip(strength, 0.0, None)
    total = float(strength.sum())
    if total <= 0.0 or not math.isfinite(total):
        return {cid: 1.0 / max(len(ids), 1) for cid in ids}
    return {cid: float(strength[i] / total) for i, cid in enumerate(ids)}


@register("pref_aggregator", "bradley_terry")
def aggregate_bradley_terry(ctx: Any, ids: Sequence[str],
                            prefs: Sequence[Preference]) -> Dict[str, float]:
    """Bradley-Terry strengths by MM iteration (GT, §4).

    GT's aggregator, and the source of the `preference_bt` fitness scalar.
    Zermelo/Hunter MM: p_i <- W_i / sum_j n_ij / (p_i + p_j), renormalised each
    sweep -- monotone in the likelihood and free of a step size, which matters
    because this runs unattended inside the loop.

    ‡ GT names Bradley-Terry and stops there; the MM iteration and the prior
    are implementation choices this repo is making, not pins. The prior is
    `alpha` virtual comparisons split evenly between every pair: it completes
    the comparison graph, which is what guarantees a finite maximiser when a
    candidate went unbeaten or unwon (all-pairs cannot produce that, but
    `tournament` and `active` can), and it shrinks toward equal strength rather
    than toward any particular ranking.
    """
    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: 1.0}

    wins, counts = _tally(ids, prefs)
    alpha = 0.5
    wins = wins + alpha * (n - 1) / 2.0
    counts = counts + alpha * (1.0 - np.eye(n))

    p = np.ones(n, dtype=float) / n
    for _ in range(500):
        denom = (counts / (p[:, None] + p[None, :])).sum(axis=1)
        nxt = wins / np.maximum(denom, 1e-12)
        total = float(nxt.sum())
        if total <= 0.0 or not math.isfinite(total):
            break
        nxt = nxt / total
        if float(np.max(np.abs(nxt - p))) < 1e-12:
            p = nxt
            break
        p = nxt
    return _normalise(ids, p)


def _elo_replay(ids: Sequence[str], prefs: Sequence[Preference], *,
                fold_ties: bool) -> Dict[str, float]:
    """REvolve's Elo replay, in emission order: K = 32, start 1500, /400 logistic.

    `refs/tex/revolve/main.tex:993-1007` (`\\subsection{REvolve Fitness}`);
    `refs/code/Revolve/human_feedback/elo_scoring.py:15` is `K = 32`, `:17` the
    `/ 400` expectation, `:27` seeds every video at `1500`. Shared by `elo` and
    `elo_raw`, which differ only in what they return -- the float expressions
    and their order are exactly `aggregate_elo`'s, so sharing the replay does
    not move `elo`'s numbers.

    `fold_ties` is the one difference in what is replayed. `run_preferences`
    writes a tie as two mirrored records (label 1 then 0) with `tie=True` on
    both; with `fold_ties` the label-1 record is ONE game at f = 0.5 (tex:999-
    1004; `elo_scoring.py:36-40`, `result = 0.5` and a single `update_elo`) and
    its label-0 mirror is skipped. Without it both records replay as decisive
    games, which is what `elo` does.
    """
    rating = {cid: 1500.0 for cid in ids}
    k_factor = 32.0
    for pref in prefs:
        if pref.left_id not in rating or pref.right_id not in rating:
            continue
        if fold_ties and getattr(pref, "tie", False):
            if pref.label != 1:
                continue
            score = 0.5
        else:
            score = 1.0 if pref.label == 1 else 0.0
        ra, rb = rating[pref.left_id], rating[pref.right_id]
        expected = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        rating[pref.left_id] = ra + k_factor * (score - expected)
        rating[pref.right_id] = rb + k_factor * ((1.0 - score) - (1.0 - expected))
    return rating


@register("pref_aggregator", "elo")
def aggregate_elo(ctx: Any, ids: Sequence[str],
                  prefs: Sequence[Preference]) -> Dict[str, float]:
    """Sequential Elo updates, mapped back onto BT strengths.

    Elo is the online approximation to Bradley-Terry -- one gradient step per
    comparison instead of a fit -- so this is the ablation of *batch vs online*
    aggregation on identical preference data, and the member that is
    order-dependent (the pair strategy's emission order becomes load-bearing,
    which is exactly the sensitivity worth measuring).

    K = 32, the 1500 start and the /400 logistic are REvolve's
    (`refs/code/Revolve/human_feedback/elo_scoring.py:15` is `K = 32`, `:17`
    the `/ 400` expectation, `:27` seeds every video at `1500`; `_elo_replay`).
    The ratings are then centred and converted by the Elo-BT identity p_i
    proportional to 10^(R_i / 400), so the value written to `bt_strength` is
    the BT-comparable IMAGE of the ratings and means the same thing whichever
    aggregator produced it.

    That image is published by nobody. REvolve's fitness sigma is the rating
    itself (`refs/tex/revolve/main.tex:993-1007`, gated raw at `:314`), and
    the exponential map is not affine, so under `evaluate.preferences.scope:
    cumulative` it changes what `above_island_mean` decides: by Jensen the
    deme mean of 10^(R/400) exceeds 10^(mean R/400), so this variant is a
    STRICTLY stricter gate than the paper's -- deme {1400, 1600}, offspring
    1500: the paper admits (1500 >= 1500), this rejects (0.2993 < 0.3503
    through `_normalise`), a lean toward `above_island_best`. A tie reaches it
    as two mirrored K=32 updates (about twice the paper's single f = 0.5
    step, order-dependent). REvolve's published config therefore pins
    `elo_raw`; this member stays as a variant for comparison.
    """
    n = len(ids)
    if n == 0:
        return {}
    rating = _elo_replay(ids, prefs, fold_ties=False)
    vals = np.array([rating[cid] for cid in ids], dtype=float)
    vals = vals - float(vals.mean())  # centre before exponentiating
    return _normalise(ids, np.power(10.0, vals / 400.0))


@register("pref_aggregator", "elo_raw")
def aggregate_elo_raw(ctx: Any, ids: Sequence[str],
                      prefs: Sequence[Preference]) -> Dict[str, float]:
    """REvolve's fitness: the Elo rating itself, one f = 0.5 game per tie.

    `refs/tex/revolve/main.tex:985-1007` (`\\subsection{REvolve Fitness}`)
    defines sigma AS the rating -- K = 32, every individual starts at 1500 --
    and `:314` gates admission `sigma >= sigma^P` on that number. The release
    replays the same updates (`elo_scoring.py:13-27`) and then min-max
    normalises the pooled ratings to [0, 1] for its report (`:42-47`); that map
    is affine in R, so every `above_island_mean` decision it makes is the raw
    rating's, and its human driver never ran (`:11`, `:198`, `:207` call
    `RewardsDatabase` / `add_reward_to_group`, which do not exist in
    `rewards_database.py`). The paper's raw rating is therefore the pin and no
    dagger applies. Raw rather than min-max also because min-max pins the worst
    raced individual at exactly 0.0, which `run_preferences` reserves for
    "never raced"; a raced rating stays far above that floor -- 1500
    minus at most 32 per game, with a shrinking step.

    Ties. A record with `tie=True` is one game at f = 0.5 (tex:999-1004;
    `elo_scoring.py:36-40`, `result = 0.5` then a single `update_elo`): the
    update is applied on the label-1 record and its label-0 mirror is skipped.
    `_pref` is the only producer of `tie=True` and always emits the (1, 0) pair,
    so a lone hand-built `Preference(label=0, tie=True)` contributes nothing.

    An id with no game on record returns 1500.0, the paper's start rating --
    including on `run_preferences`' one-agent early return, where a simplex
    aggregator would answer 1.0. Under `scope: cumulative` this is the number
    `_refit_carried` rewrites onto carried reports and `topo_island_lineage`
    refreshes into `cell.fitness`, so the admission gate compares raw ratings
    exactly as tex:314 does. `_normalise` is not applied: these are ratings,
    not a distribution, and `bt_strength` here means "Elo rating"
    (`evaluation._section_preference` labels it so).
    """
    if len(ids) == 0:
        return {}
    rating = _elo_replay(ids, prefs, fold_ties=True)
    return {cid: float(rating[cid]) for cid in ids}


@register("pref_aggregator", "borda")
def aggregate_borda(ctx: Any, ids: Sequence[str],
                    prefs: Sequence[Preference]) -> Dict[str, float]:
    """Smoothed win rate -- Borda count over pairwise ballots.

    Published by nobody. The cheap, model-free member: no transitivity
    assumption, no fit, just how often each candidate won. Under `all_pairs` it
    is the raw Borda score; under the unbalanced schedules (`tournament`,
    `active`) the count is divided by appearances, because a candidate that was
    compared eight times should not outrank one that won its only comparison
    purely for having been on screen more. That normalisation is a deliberate
    deviation from textbook Borda, forced by this family's uneven schedules.

    Jeffreys smoothing (alpha = 0.5) keeps a candidate that lost everything at
    a small positive strength rather than at zero, which matters because
    downstream these are read as a distribution.
    """
    n = len(ids)
    if n == 0:
        return {}
    wins, counts = _tally(ids, prefs)
    appearances = counts.sum(axis=1)
    alpha = 0.5
    rate = (wins + alpha) / (appearances + 2.0 * alpha)
    return _normalise(ids, rate)


@register("pref_aggregator", "copeland")
def aggregate_copeland(ctx: Any, ids: Sequence[str],
                       prefs: Sequence[Preference]) -> Dict[str, float]:
    """Copeland score: opponents beaten minus opponents lost to.

    Published by nobody. The Condorcet-flavoured member -- it counts *distinct
    opponents* rather than comparisons, so it is the one aggregator that a
    repeated comparison cannot move, and the one that keeps working when the
    preferences are intransitive (the case Bradley-Terry quietly averages
    away). The raw score becomes a smoothed rate over opponents beaten --
    `borda`'s formula, per opponent rather than per comparison -- so that an
    agent which lost every head-to-head still lands strictly above 0.0. That
    floor matters: 0.0 is reserved for a candidate with no policy to race, and
    the two populations must stay distinguishable.
    """
    n = len(ids)
    if n == 0:
        return {}
    index = {cid: i for i, cid in enumerate(ids)}
    tally = np.zeros((n, n), dtype=float)  # tally[i, j] = i's wins over j
    for pref in prefs:
        a = index.get(pref.left_id)
        b = index.get(pref.right_id)
        if a is None or b is None or a == b:
            continue
        tally[a if pref.label == 1 else b, b if pref.label == 1 else a] += 1.0

    score = np.zeros(n, dtype=float)
    faced = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if tally[i, j] == 0.0 and tally[j, i] == 0.0:
                continue
            faced[i] += 1.0
            faced[j] += 1.0
            if tally[i, j] > tally[j, i]:
                score[i] += 1.0
                score[j] -= 1.0
            elif tally[j, i] > tally[i, j]:
                score[j] += 1.0
                score[i] -= 1.0
    beaten = (score + faced) / 2.0  # opponents strictly beaten
    alpha = 0.5
    return _normalise(ids, (beaten + alpha) / (faced + 2.0 * alpha))


# ==========================================================================
# The phase -- exported, registered as `phase:preferences` by phases.py
# ==========================================================================


def _eligible(reports: List[CandidateReport]) -> List[CandidateReport]:
    """Only trained agents can race.

    A candidate that never compiled or that a screen rejected has no policy, so
    there is nothing to show a judge. Keeping the two populations apart here is
    the same discipline the artifacts keep.
    """
    return [r for r in reports if r.candidate.trainable and r.result.trained]


# ==========================================================================
# `evaluate.preferences.scope` -- what the ranking is fitted over
#
# Everything below is reached only when the key says `cumulative`. The default
# `within_iteration` path calls none of it, which is the safety property this
# axis must have: GT's numbers must not move because REvolve needed a knob.
# ==========================================================================


def _held_reports(state: Any) -> List[CandidateReport]:
    """Every individual still held in `RunState`, in a deterministic order.

    `cumulative` refits "the fitness scores of all individuals" (REvolve,
    Appendix B.1 p.20), and in this repo an individual that survived an
    iteration is not one container but five slots: an `ArchiveCell.report` (the
    population §6 keeps), `state.best` (the global incumbent
    `_update_global_best` compares each fresh winner against,
    `bird/components/update.py:101`), `state.latest`, `state.iteration_best`
    and `state.all_reports`. All five are walked, because leaving any one of
    them on the previous round's normalisation is the same silent bias wearing
    a different name -- a `state.best` still holding a simplex-over-16 value
    while this round's winner arrives on a simplex-over-64 one is an incumbent
    that can never be displaced again, and no artifact would say so.

    Deduped by IDENTITY and not by `cand_id`: these slots share objects -- the
    archive cell and `state.best` are usually the very same report -- and a
    fitness written onto one copy but not the other is a two-copies-of-one-fact
    failure. Order is slot order, then each container's
    own insertion order, so the id list the caller builds from this is a
    function of the state and never of a set's iteration order.

    Read through `getattr` throughout: this runs against whatever object the
    phase was handed, and a slot a test's stub state does not carry must read
    as "no individuals there", never as an AttributeError inside stage 4.
    """
    out: List[CandidateReport] = []
    seen: set = set()
    cells = [getattr(cell, "report", None)
             for cell in (getattr(state, "archive", None) or {}).values()]
    singles = [getattr(state, name, None)
               for name in ("best", "latest", "iteration_best")]
    for rep in cells + singles + list(getattr(state, "all_reports", None) or ()):
        if rep is None or id(rep) in seen:
            continue
        seen.add(id(rep))
        out.append(rep)
    return out


def _cumulative_prefs(state: Any, prefs: List[Preference],
                      stored: bool) -> List[Preference]:
    """The comparison record a cumulative refit is fitted over.

    `state.preferences` when `store_dataset` is on, because the extend a few
    lines up already put this round's comparisons in it and appending them
    again would double every game this generation played. When the store is
    off, this round's list is concatenated instead, so the refit still sees the
    comparisons the budget just paid for rather than silently dropping them.

    That second arm should be unreachable -- `cumulative` without
    `store_dataset` (and without `preference_dataset` in `loop.carry`) is a
    config that asks for a cumulative fit over a store it is throwing away, and
    the coherence rules refuse it. It is written anyway because the degraded
    answer here is a *number*, not a crash: with an empty store the refit rates
    only whoever happens to be in this round and then rewrites the whole
    archive from it. `_refit_carried` is the other half of that defence -- it
    refuses to rewrite an individual whose games are not in the list.
    """
    store = list(getattr(state, "preferences", None) or ())
    return store if stored else store + list(prefs)


def _cumulative_ids(pool: List[CandidateReport],
                    prefs: Sequence[Preference]) -> List[str]:
    """The ids a `scope: cumulative` refit is fitted over.

    This round's pool first in pool order (so the default path's ordering is a
    prefix of this one), then every id appearing anywhere in the accumulated
    comparison record, in first-appearance order.

    **Every compared id, not just the surviving population.** Naming only the
    pool plus the archive looks like the right set and quietly corrupts the
    fit: `_elo_replay` (behind both `elo` and `elo_raw`) skips any preference
    with a side it was not given (`if pref.left_id not in rating or
    pref.right_id not in rating: continue`) and `_tally` does the same for the
    batch aggregators, so an archived individual would lose every game it
    played against a round-mate that was never admitted. Its rating would then
    move as a function of *who else got admitted* -- a path effect on a number
    that `update.archive.admission` gates on and that `select.final_artifact:
    archive_best` takes an argmax over. Including the losers costs one dict
    entry each and buys the property that makes the refit meaningful: because
    the pair pool is never widened (below), each generation's comparisons form
    a clique disjoint from every other generation's, so replaying all of them
    from a 1500 start reproduces every individual's raw Elo rating EXACTLY as
    its own round computed it. What `cumulative` then changes depends on the
    aggregator:

    * `elo_raw` -- what REvolve's human-feedback config pins -- returns that
      rating as the strength with no `_normalise`, so the refit is an
      identity: every number `_refit_carried` writes back is the number
      already there (the `preference_refit` event still counts them as
      rewritten -- it counts writes, not changes -- and a tester run measures
      15 of 15 bit-identical). The key is inert in value for this aggregator.
      It stays pinned because REvolve's release refits from generation 0
      every generation (`elo_scoring.py:124-135`) and because the moment the
      pair pool IS widened (see below) the cliques stop being disjoint and
      the refit stops being an identity.
    * `elo` centres the ratings over the widened id set and pushes them
      through `_normalise`, so there the only thing `cumulative` changes is
      the centring and the simplex denominator.
    * The batch aggregators do not share the clique property at all:
      `bradley_terry`'s alpha prior is spread over every pair, so a larger id
      set shrinks every strength toward uniform. A `cumulative` +
      `bradley_terry` config is reachable and is a different estimator, not
      the same one on more data.

    **The PAIR pool is not widened with it, and that is deliberate.** Widening it
    would mean pairing an archived report, which needs `_clip_for` -- and
    `_clip_for` reads `report.result.trajectories` (:203), which
    `RunDir.save_train_result` drops from the artifact and `bird/checkpoint.py`
    drops from the checkpoint. Nothing clears them *in memory*, so a live
    process would compare cross-generation pairs happily and a resumed one
    would render "(no rollout was recorded for this agent)" (:424) for every
    archived side and degrade to `_decide_offline` -- identical before the
    resume, silently different after it, which is precisely the divergence
    `bird/state.py:13-27` says no static test can catch.

    The obstacle is a real one but it is NOT "this repo cannot represent a
    cross-generation clip", and the difference matters for extending it:
    `Preference.left_traj`/`right_traj` hold the clip the judge actually
    saw, `state.preferences` is a carry slot, and `bird/checkpoint.py:1329-1333`
    spills those clips' states through `export_states` on purpose. Under
    `pairs: all_pairs` every individual is compared at least once, so a
    `cand_id -> clip` index over the store would supply a checkpoint-durable
    clip for every archived individual. That is an implementation with a cost
    (a lookup, and an Elo graph that stops being a disjoint union of cliques),
    not an impossibility -- so REvolve's retroactive *rescoring* is reproduced
    here and its retroactive *re-pairing* is not, and the second is a design
    decision rather than a limitation of the representation.
    """
    ids = [r.cand_id for r in pool]
    seen = set(ids)
    for pref in prefs:
        for cid in (pref.left_id, pref.right_id):
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
    return ids


def _as_strength(value: Any) -> Optional[float]:
    """A finite float, or None. `bt_strength` survives a checkpoint as JSON."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _refit_carried(state: Any, carried: List[CandidateReport],
                   strengths: Dict[str, float]) -> List[str]:
    """Rewrite the fitness of every carried individual the refit re-rated.

    **The direct `rep.fitness = ...` is mandatory and is a §4 write to reports
    §4 does not own for this round.** `_resolve_deferred` refuses to run twice
    (`bird/components/evaluation.py:913`, `or rep.fitness is not None`), and
    `ArchiveCell.fitness` is only ever written as `_score(report)` at insert
    time, so an archived report keeps the fitness its first round gave it for
    ever unless this assignment happens. Without it the whole key is
    decoration: the pool would move onto the widened normalisation and the
    population it is gated against would not, which is *worse* than not having
    the key at all. The matching `ArchiveCell.fitness` refresh -- the cell
    carries its own float beside the report -- belongs to the `topology`
    component that owns the archive, and is an identity unless this ran first.

    Two refusals, and each one is a wrong number this would otherwise write:

    * **no strength** -- the individual has no game in the record the refit was
      fitted over, so there is nothing to say about it. Its old fitness stands.
    * **no fitness, or a fitness that is not its own `bt_strength`** -- this
      report's scalar did not come from a preference round. A ground-truth
      fitness under `preferences.enabled`, a `min_max`-normalised one
      (`evaluation.py:492`, which records `raw_fitness` and rewrites the
      scalar), and a `human_veto` sentinel (`phases.py:1070`) all land here,
      and overwriting any of them with a simplex strength would be a scale
      error dressed as a refresh. Asking the report directly -- *is your
      fitness the strength this phase last wrote?* -- is deliberately not the
      same as testing `fitness_source == "preference_bt"`: that would put a
      second copy of `evaluation._DEFERRED`'s table in this file, and the copy
      is what goes stale.

    `pref_summary`, `pref_rank` and the win/loss counts are NOT rewritten. They
    describe the round the individual raced in, and that round happened; a
    refit rescales the verdict, it does not re-run the contest.

    Two limitations, stated rather than papered over:

    * **A round with fewer than two trained agents does not refit.**
      `_run_preferences` returns early there: it asks the aggregator for its
      uncontested answer over the pool alone (`{id: 1.0}` from a simplex
      aggregator, the 1500.0 start rating from `elo_raw`) and never reaches
      the `cumulative` block, so this function is not called and a generation
      in which nothing trained leaves the carried population on the previous
      round's scale for one iteration. Refitting from a contest that did not
      happen would be the worse answer; a generation with no contest produced
      no evidence.
    * **The artifact a carried report already wrote is not rewritten.**
      `candidates/iterNN_<id>/report.json` is written once at the §4 of the
      iteration that produced it, so it keeps the number that round computed
      while `records/iterNN.json` shows the refit. That asymmetry is inherent
      to `save_report`, and it means a
      `report.json` fitness and a later `records` fitness for the same
      candidate are BOTH right and are not comparable. `bt_refit_iteration` on
      the live report is the only thing that says which one you are reading.
    """
    touched: List[str] = []
    iteration = int(getattr(state, "iteration", -1))
    for rep in carried:
        fresh = _as_strength(strengths.get(rep.cand_id))
        if fresh is None:
            continue
        prior = _as_strength(rep.meta.get("bt_strength"))
        current = _as_strength(rep.fitness)
        if prior is None or current is None:
            continue
        if not math.isclose(current, prior, rel_tol=1e-12, abs_tol=1e-15):
            continue
        rep.meta["bt_strength"] = fresh
        # What it was and when it moved, as plain scalars -- `checkpoint.encode`
        # has tags for those and none for anything richer, and a rewritten
        # number with no record of the rewrite is the thing that makes
        # `records/iterNN.json` incomparable across iterations without saying so.
        rep.meta["bt_strength_prior"] = prior
        rep.meta["bt_refit_iteration"] = iteration
        rep.fitness = fresh
        touched.append(rep.cand_id)
    return touched


def _warm_frames(ctx: Any, pool: List[CandidateReport], comparator: Any) -> None:
    """Render every pooled candidate's clip once, before anything overlaps.

    Three jobs, and only the first is about speed.

    1. **Once per candidate, not once per comparison.** On `all_pairs` over ten
       agents each side is judged nine times, and `evaluate.vlm.repeats`
       multiplies that again; the pixels do not change with the question. Same
       argument, same shape, as sampling once per rollout rather than once per
       subtask in `fitness_vlm_score`.
    2. **Before the pool.** `sample_frames` drives the shared env renderer and
       two threads on one MuJoCo adapter segfault 3/3, so the renders have to be
       finished by the time `judge_concurrency` opens a ThreadPoolExecutor. The
       cache warm-up below is what makes the comparators' `parallel_safe` claim
       true rather than merely asserted.
    3. **On record before the first query.** `_pair_frames` writes the `frames`
       record where it cuts the frames, so calling it here is what puts every
       candidate's pixels on record before any comparison is asked -- a run
       killed mid-round still says what its judge was looking at.

    The journal event is written on the GOOD path too. "No warning in the log"
    is not evidence that the judge had eyes, so the count travels with the run
    either way and a collector can filter on it.
    """
    if not getattr(comparator, "attaches_frames", False):
        return
    counts: List[int] = []
    reasons: List[str] = []
    # The loop is UNCONDITIONAL once the comparator attaches frames, including
    # for a config that lists no videos and will get none. It costs nothing
    # there -- `_pair_frames` returns before it renders -- and it is what makes
    # the cache populated for every report before a thread can exist. Gated on
    # `videos` instead, two comparators racing the same cold report would each
    # write a `frames` record for it, and the artifact would carry a duplicate
    # that reads as two separate frame sets.
    for report in pool:
        png, prov = _pair_frames(ctx, report)
        counts.append(len(png))
        reason = str(prov.get("blind_reason") or "")
        if reason:
            reasons.append(reason)
    if "videos" not in (ctx.cfg.get("evaluate.artifacts") or []):
        # Nothing was asked for, so nothing is missing. Journalling a
        # `pair_frames` event here would report "10 of 10 blind" for a
        # deliberately scalars-only config -- the same false alarm
        # `_count_blind` refuses to raise for the same case, and the reason the
        # gate is on the REPORTING and not on the work.
        return
    blind = sum(1 for c in counts if not c)
    ctx.event("pair_frames", n_candidates=len(pool), n_blind=blind,
              images_per_candidate=counts, reason=reasons[0] if reasons else "")
    if blind:
        log.warning("%d of %d agent(s) enter this preference round with NO "
                    "frames (%s); the pairwise judge is text-only for every "
                    "comparison they appear in",
                    blind, len(pool), reasons[0] if reasons else "unknown")


def run_preferences(ctx: Any, state: Any,
                    reports: List[CandidateReport]) -> List[CandidateReport]:
    """Run this iteration's preference round, and always drop the caches.

    A thin wrapper over `_run_preferences` for one reason: a RAISE must drop
    the per-report clip and frame caches just as the clean exits do, and
    `BudgetExceeded` from `_query` is a raise that happens in normal operation
    -- it is how a round ends when the LLM budget runs out mid-comparison. The
    caches would otherwise ride out on `CandidateReport.meta` into `RunState`,
    and a checkpoint written on the way out would base64 twenty PNGs per
    candidate -- megabytes -- into its JSON (`checkpoint.encode` handles
    `bytes`, so this bloats rather than crashes -- which is why nothing would
    report it).
    """
    try:
        return _run_preferences(ctx, state, reports)
    finally:
        _drop_clips(reports)


def _run_preferences(ctx: Any, state: Any,
                     reports: List[CandidateReport]) -> List[CandidateReport]:
    """Run this iteration's preference round (GT, §4, §2).

    Registered as `phase:preferences` by `bird/components/phases.py`; `evaluate()`
    reaches it through `registry.get("phase", "preferences")` when
    `evaluate.preferences.enabled`.

    Sequence: pick the comparisons (`pairs`), run each one (`comparator`),
    accumulate the labels into `D_pref` (`store_dataset`), fit a ranking
    (`aggregator`), and write the strength to `report.meta["bt_strength"]` --
    the exact key `evaluate.fitness.source: preference_bt` reads.

    Two invariants worth stating because they are easy to break:

    * **What the aggregator sees is `evaluate.preferences.scope`, and nothing
      else in this function.** Under the default `within_iteration` it is this
      round's ids and this round's preferences: GT renormalises per round and
      its strengths cross no round boundary, so reaching for the accumulated
      store there would silently turn a per-round ranking into a cumulative
      one. Under `cumulative` the widening is deliberate, it is confined to the
      one block below, and it covers the AGGREGATION only -- the pair pool is
      `_eligible(reports)` under either value, for the resume reason set out in
      `_cumulative_ids`. Nothing about *which* comparisons get bought, who
      judges them, or how they are recorded depends on this key.
    * `state.preferences` grows unconditionally when `store_dataset` is set,
      losers included. That store is §2's TAC evidence for the NEXT iteration,
      and its unconditional growth is the recorded contrast with CARD's
      self-gated `verify.tpe.store: append_on_pass`, which gates the growth of
      its own evidence (§2). GT's own §5.3 notes TAC only beats random past
      ~100 preferences while its runs gathered 40, so how fast this fills is
      the whole ballgame.
    """
    cfg = ctx.cfg
    comparator_name = cfg["evaluate.preferences.comparator"]
    pairs_name = cfg["evaluate.preferences.pairs"]
    aggregator_name = cfg["evaluate.preferences.aggregator"]
    allow_ties = bool(cfg.get("evaluate.preferences.allow_ties"))

    comparator = registry.get("comparator", comparator_name)
    pair_fn = registry.get("pair_strategy", pairs_name)
    aggregator = registry.get("pref_aggregator", aggregator_name)

    pool = _eligible(reports)
    # A candidate with no policy loses every comparison it could not enter.
    # 0.0 is below every aggregator's raced value -- the floor of a simplex
    # strength, and far under a raw Elo rating (1500 minus at most 32 a game)
    # -- so it plays the role `select.failure_value` plays for a fitness
    # scalar: recorded, and last. Every aggregator keeps a raced candidate
    # strictly above this floor, so "never raced" and "lost everything" stay
    # distinguishable.
    for report in reports:
        report.meta["bt_strength"] = 0.0

    if len(pool) < 2:
        # The aggregator's own answer for an uncontested pool, not a literal:
        # every simplex aggregator returns {id: 1.0} for one id and {} for
        # none (pinned by tests/test_revolve_elo_raw.py), and `elo_raw`
        # returns the paper's 1500.0 start rating -- a 1.0 there would, under
        # `scope: cumulative`, sit 1499 below every peer and fail every gate.
        strengths = aggregator(ctx, [r.cand_id for r in pool], [])
        _write_back(ctx, pool, strengths, [], comparator_name, aggregator_name)
        ctx.event("preferences", comparator=comparator_name, pairs=pairs_name,
                  aggregator=aggregator_name, n_comparisons=0, n_candidates=len(pool),
                  note="no contest: fewer than two trained agents")
        return reports

    raw_pairs = pair_fn(ctx, state, pool) or []
    seen: set = set()
    plan: List[Tuple[int, int]] = []
    for pair in raw_pairs:
        i, j = int(pair[0]), int(pair[1])
        if not (0 <= i < len(pool) and 0 <= j < len(pool)) or i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue  # a strategy may propose a rematch; buying it twice is waste
        seen.add(key)
        plan.append((i, j))

    # Every pair strategy returns its FULL plan before any comparison runs (the
    # frozen `pair_fn` signature forces it -- see `pairs_tournament`), so the
    # comparisons are independent by construction and may overlap in flight.
    # Gated three ways: the comparator declares itself `parallel_safe` (the
    # human oracle never is), the evaluator client declares it fans out
    # (`concurrent_samples` -- the mock never does, so tester-tier determinism
    # is untouched), and `llm.max_concurrent_requests` bounds the width.
    # Serially, one judge round-trip at a time, the preferences block of one
    # GT iteration measured 1,990 s.
    # Labels are gathered IN PLAN ORDER; the fold below is unchanged and
    # sequential, so tie resolution, `_pref` construction and `D_pref` order
    # are byte-identical to the serial loop's.
    # Frames FIRST, and unconditionally -- not inside the `workers > 1` arm the
    # clip warm-up lives in. `sample_frames` drives the shared env renderer, and
    # two threads on one MuJoCo adapter segfault 3/3, so rendering must be
    # finished before a pool can exist. Doing
    # it here also means the frames are on record BEFORE the first query, so a
    # run killed mid-round still says what its judge was looking at --
    # `fitness_vlm_score` orders it the same way for the same reason.
    _warm_frames(ctx, pool, comparator)
    workers = (judge_concurrency(ctx, len(plan))
               if getattr(comparator, "parallel_safe", False) else 1)
    if workers > 1:
        for r in pool:
            _clip_for(ctx, r)  # warm the per-report clip cache outside the pool
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="bird-judge") as ex:
            labels = list(ex.map(
                lambda ij: comparator(ctx, state, pool[ij[0]], pool[ij[1]]), plan))
    else:
        # A generator: the serial path judges as the fold consumes, with no
        # materialised label list.
        labels = (comparator(ctx, state, pool[i], pool[j]) for i, j in plan)

    prefs: List[Preference] = []
    n_ties = 0
    n_abstained = 0
    for (i, j), label in zip(plan, labels):
        left, right = pool[i], pool[j]
        if label == TIE and not allow_ties:
            # GT forbids ties on purpose (§4), so a judge that answers TIE anyway
            # has disregarded the instruction. Resolving it with `_decide_offline`
            # would rank the two agents by their OWN reward scale -- the
            # reward-scale fallback through the tie door. Under
            # a real provider ABSTAIN (the pair leaves D_pref); only `provider:
            # mock`, which has no real judge, keeps the deterministic offline
            # resolution. Same split as every other judge fallback here.
            label = _fallback_decision(ctx, left, right, role="pair_tie_leak",
                                       reason="judge returned TIE under allow_ties:false")
        if label == ABSTAIN:
            # The judge produced nothing usable, or leaked a tie under
            # allow_ties:false, under a real provider: the pair leaves D_pref
            # rather than being fabricated from reward magnitude.
            # `_fallback_decision` already journalled it.
            n_abstained += 1
            continue
        if label == TIE:
            # A tie is two mirrored records, which keeps `Preference.label`
            # inside its frozen {0, 1} contract: for the tally aggregators
            # (`borda`, `copeland`) the pair IS a half-win each way, and for
            # `bradley_terry` it is a double-weight tie in the right
            # direction. `tie=True` on both lets `elo_raw` fold the pair into
            # the paper's single f = 0.5 game (REvolve tex:999-1004,
            # elo_scoring.py:36-40). `elo` still replays both records -- two
            # K=32 updates, about twice the movement, order-dependent -- and
            # is not the variant REvolve's config pins.
            n_ties += 1
            prefs.append(_pref(ctx, state, left, right, 1, comparator_name, tie=True))
            prefs.append(_pref(ctx, state, left, right, 0, comparator_name, tie=True))
        else:
            prefs.append(_pref(ctx, state, left, right, 1 if label == 1 else 0,
                               comparator_name))

    if cfg.get("evaluate.preferences.store_dataset"):
        state.preferences.extend(prefs)

    # AGGREGATION SCOPE (§4, `evaluate.preferences.scope`). Read with an
    # explicit default rather than `cfg[...]` because the two answers must not
    # be equally likely: `within_iteration` is what every published config
    # before REvolve means, and a tree in which the key is absent -- an older
    # `_default.yaml`, a hand-built config dict in a test -- has to resolve to
    # GT's rule and not to an exception in the middle of stage 4. `or` rather
    # than a two-argument `get`, because the schema marks these keys nullable
    # and `scope: null` must land on the default too, not on `str(None)`.
    scope = str(cfg.get("evaluate.preferences.scope") or "within_iteration")
    carried: List[CandidateReport] = []
    if scope == "cumulative":
        # REvolve refits from generation 0 every generation
        # (`refs/code/Revolve/human_feedback/elo_scoring.py:124-135` reloads
        # every CSV and re-runs the whole Elo sweep). The pair pool above is
        # untouched; only what the ranking is fitted over widens.
        agg_prefs = _cumulative_prefs(
            state, prefs, bool(cfg.get("evaluate.preferences.store_dataset")))
        ids = _cumulative_ids(pool, agg_prefs)
        pooled = {r.cand_id for r in pool}
        carried = [r for r in _held_reports(state) if r.cand_id not in pooled]
    else:
        ids = [r.cand_id for r in pool]
        agg_prefs = prefs
    strengths = aggregator(ctx, ids, agg_prefs)
    _write_back(ctx, pool, strengths, prefs, comparator_name, aggregator_name)
    if scope == "cumulative":
        # A SEPARATE event, not three more fields on the one below: the
        # `preferences` record describes a contest that was run, and this
        # describes numbers that were rewritten on individuals which raced in
        # earlier generations. Keeping them apart is also what leaves the
        # default path's journal byte-identical.
        refit = _refit_carried(state, carried, strengths)
        ctx.event("preference_refit", scope=scope, aggregator=aggregator_name,
                  n_rated=len(ids), n_pool=len(pool), n_carried=len(carried),
                  n_preferences=len(agg_prefs), refit=refit)
        log.info("      preferences -> cumulative refit over %d individual(s) "
                 "from %d comparison(s); %d carried fitness value(s) rewritten",
                 len(ids), len(agg_prefs), len(refit))

    ctx.event("preferences", comparator=comparator_name, pairs=pairs_name,
              aggregator=aggregator_name, n_comparisons=len(plan), n_ties=n_ties,
              n_abstained=n_abstained,
              n_candidates=len(pool), stored=bool(cfg.get("evaluate.preferences.store_dataset")),
              store_size=len(state.preferences),
              strengths={cid: round(v, 6) for cid, v in strengths.items()})
    log.info("      preferences -> %d comparison(s) over %d agent(s) via %s/%s; "
             "strengths via %s (D_pref=%d)",
             len(plan), len(pool), pairs_name, comparator_name, aggregator_name,
             len(state.preferences))

    return reports


def _pref(ctx: Any, state: Any, left: CandidateReport, right: CandidateReport,
          label: int, source: str, *, tie: bool = False) -> Preference:
    """One `D_pref` record, carrying the clips the judge actually saw.

    The SOURCE WINDOW the judge watched, at full resolution -- not the even-
    strided view clip. §2's TAC re-scores these sub-trajectories with the
    candidate's own reward, which needs real `(s, a, s')` transitions; the
    frames the judge saw are an even-stride SAMPLE of this window
    (`_clip_for`), and re-scoring that sample would pair non-adjacent states
    with a single action and drop the terminal next-state. The window is
    the same on both paths, so TAC still re-scores the behaviour the judge
    assessed -- it just does it on the transitions that actually occurred.
    """
    pref = Preference(
        left_id=left.cand_id,
        right_id=right.cand_id,
        label=label,
        left_traj=_pref_traj_for(ctx, left),
        right_traj=_pref_traj_for(ctx, right),
        source=source,
        iteration=int(getattr(state, "iteration", -1)),
        tie=tie,
    )
    # Written HERE, where the comparison is made, rather than at the iteration
    # boundary: a run killed mid-round keeps the comparisons it had already
    # paid for. `preference_log` swallows its own failures -- a lost line is a
    # gap in a dataset, never a failed search.
    preference_log.record(
        ctx, pref, state,
        # The `human` comparator delegates to `ctx.human`, which may be a
        # script -- so the ORACLE says what kind of judge this was, not the
        # comparator's name.
        judge=(preference_log.judge_of(getattr(ctx, "human", None))
               if source == "human" else "model"))
    return pref


def _write_back(ctx: Any, pool: List[CandidateReport], strengths: Dict[str, float],
                prefs: List[Preference], comparator_name: str,
                aggregator_name: str) -> None:
    """Publish the round's verdict onto the reports.

    `meta["bt_strength"]` is the contract with `evaluate.fitness.source:
    preference_bt`; everything else is evidence for the feedback builder and
    the journal, and no stage is required to read it.
    """
    wins: Dict[str, int] = {r.cand_id: 0 for r in pool}
    losses: Dict[str, int] = {r.cand_id: 0 for r in pool}
    for pref in prefs:
        winner = pref.left_id if pref.label == 1 else pref.right_id
        loser = pref.right_id if pref.label == 1 else pref.left_id
        if winner in wins:
            wins[winner] += 1
        if loser in losses:
            losses[loser] += 1

    order = sorted(pool, key=lambda r: (-strengths.get(r.cand_id, 0.0), r.cand_id))
    rank = {r.cand_id: k + 1 for k, r in enumerate(order)}

    for report in pool:
        cid = report.cand_id
        strength = float(strengths.get(cid, 0.0))
        played = wins[cid] + losses[cid]
        report.meta["bt_strength"] = strength
        report.meta["pref_wins"] = wins[cid]
        report.meta["pref_losses"] = losses[cid]
        report.meta["pref_comparisons"] = played
        report.meta["pref_win_rate"] = (wins[cid] / played) if played else None
        report.meta["pref_rank"] = rank[cid]
        report.meta["pref_comparator"] = comparator_name
        report.meta["pref_aggregator"] = aggregator_name
        report.meta["pref_summary"] = (
            f"{wins[cid]} win(s) / {losses[cid]} loss(es) over {played} pairwise "
            f"comparison(s) judged by {comparator_name}; {aggregator_name} strength "
            f"{strength:.4f}, rank {rank[cid]} of {len(pool)} this round.")


def _drop_clips(reports: List[CandidateReport]) -> None:
    """Clips are a within-phase cache; they must not ride out on the reports.

    The frames go with them, and that one matters more than the clips did.
    Twenty PNGs per candidate is megabytes and `CandidateReport.meta` is carried
    into `RunState`, so a frame list left behind here reaches `save_report` and
    `checkpoint.encode`. **Neither of them fails**, and that is the problem:
    `encode` has a `bytes` tag and base64s them straight into the checkpoint
    JSON, whose digest then covers megabytes of pixels that no reader wants and
    no resume uses. A crash would at least be reported. `run_preferences` calls
    this from a `finally` for the same reason -- the leak path that matters is
    the one nothing would notice.
    """
    for report in reports:
        report.meta.pop(_CLIP_KEY, None)
        report.meta.pop(_CLIP_STEPS_KEY, None)
        report.meta.pop(_FRAMES_KEY, None)
        # The full-resolution TAC trajectory carries the window's states and
        # actions -- heavier than the view clip -- and for exactly the same
        # reason it must not reach `checkpoint.encode`.
        report.meta.pop(_PREF_TRAJ_KEY, None)
