"""Stage 4 components -- Reward Evaluation (§4).

Stage 4 has **two outputs, not one**, and they are configured apart:

  * ``report.fitness``  -- the scalar. Read only by stage 5.
  * ``report.feedback`` -- the prose. Read only by stage 6.

A method may have one without the other. Eureka has both. CARD has only the
prose: it computes *no* ranking scalar at all (§4, `evaluate.fitness.source`
note -- "`none` has two readings: fitness comes from VLM/human elsewhere in
this block -- or from nowhere"). That second reading is why the sentinel and
`None` must stay distinguishable: `select.failure_value` means "this candidate
lost", `None` means "no comparison exists". Collapsing them into `0.0` would
silently turn CARD into a hill-climber.

Component families implemented here:

    fitness_source          ground_truth_metric | success_rate | vlm_score
                            | preference_bt | human_score | none
    checkpoint_aggregation  final | max_over_checkpoints | auc | last_k_mean | iqm
    seed_aggregation        mean | median | iqm | min
    feedback_builder        default
    similarity              none | pearson_curve | epic | starc

Everything below is stdlib + numpy and runs offline. Where a component needs an
LLM (`evaluate.feedback.analyzer: llm` goes through ``ctx.evaluator``;
`fitness.source: vlm_score` through ``ctx.generator``, RDA's in-loop Agent VLM)
or a human (`human_score`, ``ctx.human``) it degrades
to a documented deterministic stand-in if the client returns nothing usable --
tests run whole methods end to end with the mock LLM, and a parse failure must
not change which branch of the config was exercised.
"""

from __future__ import annotations

import inspect
import itertools
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

#: numpy renamed `trapz` to `trapezoid`; bind once so `auc` is not a hasattr
#: check per call.
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz

from .. import judgments, multiview
from ..parsing import extract_json
from ..observability import sample_frames
from ..registry import register
from ..types import CandidateReport, Trajectory, TrainResult

log = logging.getLogger("bird")

# --------------------------------------------------------------------------
# Constants for choices the schema has no knob for.
#
# Each of these is a place where a published method has a number but the
# schema does not give it a leaf key. They are named here rather than inlined
# so that "the schema is missing a knob" stays a visible, gradeable claim
# instead of a magic literal three levels down a helper.
# --------------------------------------------------------------------------

#: Checkpoints verbalised in numeric reflection. Eureka's released code samples
#: roughly ten evenly spaced points off each component trace.
_REFLECT_POINTS = 10

#: `k` for `checkpoint_aggregation: last_k_mean`. The schema has no
#: `evaluate.fitness.last_k` key; CARD's ~5 eval checkpoints (§4) is the
#: nearest published precedent, and it is half of Eureka's ~10-point grid.
_LAST_K = 5

#: The rendering stride is a config key
#: (`evaluate.feedback.trajectory_sample_interval`; CARD publishes it in
#: App. B.2 Table 8). These two are NOT published numbers:
#: the row cap keeps one rollout from eating the whole prompt budget that §6
#: then has to trim -- and when it binds, `render_trajectory` re-strides evenly
#: across the WHOLE episode rather than head-truncating, because a rendering
#: that stops at t=230 of a 500-step rollout hides the terminal state, which is
#: the one step that says whether the task was solved.
_TRAJ_MAX_STEPS = 24
_TRAJ_MAX_VARS = 8
#: The JUDGE's per-step component table (`_traj_component_table`) is not under
#: `_TRAJ_MAX_VARS`: App. 7.3 hands the analyst every `self.reward_info` entry,
#: and real RDA rewards carry 7-20 components -- an 8-column slice would drop
#: the tail (success bonus, penalties), which is exactly what "one component
#: overshadowing others" has to be diagnosed FROM.
#: This is a prompt-budget ceiling only, and whenever it binds the table says
#: "showing the first N of M", so neither the analyst nor the judgment record
#: can mistake a partial table for the whole reward.
_JUDGE_TABLE_MAX_VARS = 32

#: Per-step channels that hold the *reference* reward, not the candidate's.
#: They are the similarity metrics' input and must never reach the prose: §1's
#: `generate.context.strip_existing_reward` exists so the LLM cannot see the
#: reward it is benchmarked against, and a rendered rollout is a prompt.
_REFERENCE_KEYS = ("gt_reward", "ground_truth", "gt", "reference_reward")


# ==========================================================================
# Numeric helpers
# ==========================================================================


def _num(x: Any) -> Optional[float]:
    """Coerce to a finite float, or None. Booleans count: env success flags
    arrive as bool and `success_rate` is their mean."""
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if math.isfinite(v) else None
    return None


def _clean(values: Sequence[float]) -> np.ndarray:
    a = np.asarray([v for v in (_num(x) for x in values) if v is not None], dtype=float)
    return a


def _iqm(a: np.ndarray) -> float:
    """Interquartile mean: drop the outer 25% of each tail, average the core.

    Robust to the single catastrophic seed that `mean` lets dominate and that
    `median` throws away information to avoid (§4, `seed_aggregation`)."""
    s = np.sort(a)
    n = s.size
    lo, hi = int(math.floor(n * 0.25)), int(math.ceil(n * 0.75))
    core = s[lo:hi]
    return float(core.mean()) if core.size else float(s.mean())


def _sample_evenly(seq: Sequence[float], n: int) -> List[float]:
    """`n` evenly spaced samples (endpoints included) -- how a trace becomes a
    line of prose short enough to survive the prompt budget."""
    xs = list(seq)
    if len(xs) <= n:
        return xs
    idx = np.linspace(0, len(xs) - 1, num=n)
    return [xs[int(round(i))] for i in idx]


def _fmt(x: Optional[float], places: int = 3) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{places}f}"


def _fmt_list(xs: Sequence[float], places: int = 2) -> str:
    return "[" + ", ".join(_fmt(_num(x), places) for x in xs) + "]"


def _paired_finite(a: Sequence[float], b: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Index-aligned pairs over the common prefix, keeping a pair only when BOTH
    values are finite.

    Not `_clean(a), _clean(b)` then a positional zip: that drops each series'
    non-finite entries INDEPENDENTLY, so one NaN in `a` at index k pairs a[k+1]
    with b[k] for every later point and the "correlation" is over misaligned
    samples (`_pearson([1, nan, 2, 3, 4], [1, 5, 2, 3, 4])` would read 0.227
    where the four valid pairs correlate at 1.0).
    Pearson's r and the EPIC/STARC pseudometrics are defined over PAIRED
    observations; pairwise deletion drops the pair, never the element."""
    xs = [_num(v) for v in a]
    ys = [_num(v) for v in b]
    m = min(len(xs), len(ys))
    keep = [(x, y) for x, y in zip(xs[:m], ys[:m]) if x is not None and y is not None]
    if not keep:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    return (np.asarray([p[0] for p in keep], dtype=float),
            np.asarray([p[1] for p in keep], dtype=float))


def _pearson(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Pearson r over the common prefix, pairwise-finite (`_paired_finite`).
    None when it is undefined (fewer than two paired points, or a constant
    series) -- a constant reward correlates with nothing, and reporting 0.0 for
    that would be a fabricated number."""
    x, y = _paired_finite(a, b)
    if x.size < 2:
        return None
    sx, sy = float(x.std()), float(y.std())
    if sx < 1e-12 or sy < 1e-12:
        return None
    r = float((((x - x.mean()) * (y - y.mean())).mean()) / (sx * sy))
    # Clamp: float error puts a perfect correlation at 1.0000000000000002, and
    # `similarity.role: select` would promote that straight into a fitness.
    return max(-1.0, min(1.0, r))


# ==========================================================================
# checkpoint_aggregation -- values -> float
#
# This is a real finding, not bookkeeping. Eureka selects on
# `max_over_checkpoints`, i.e. the best of ~10 noisy evaluations of one seed,
# which is optimistically biased -- the reported number is the maximum of a
# sample, not an estimate of the policy's value. `auc` and `last_k_mean` cost
# nothing extra (the checkpoints are already there) and nobody has run them.
# The figures rule that follows: plot the quantity the search selected on.
# ==========================================================================


@register("checkpoint_aggregation", "final")
def ckpt_final(values: Sequence[float]) -> float:
    """Last checkpoint only. The unbiased-but-noisy baseline: it estimates the
    end-of-training policy, which is the thing actually shipped.

    "The thing actually shipped" is the definition, and `train.checkpoint_selection:
    best_by_reward` can make that an EARLIER checkpoint: stage 3 restores it into
    the live model and records which in the seed row. `_per_seed_values` honours
    that through `scores_shipped` below -- this rule then reads the shipped
    checkpoint's row rather than the curve's last -- so the scored number and the
    shipped policy agree, which is the whole reason the key exists. The other
    rules describe the curve and are left alone.

    "Unbiased" then needs its qualifier: the restored row was CHOSEN as the argmax
    of the candidate's own reward over ~20 noisy evaluations, so its task metric
    inherits whatever the correlation between the two carries of the winner's
    curse `ckpt_max` documents. It is not a max over the task metric -- selection
    never reads that -- but it is no longer a row nobody looked at either."""
    a = _clean(values)
    return float(a[-1]) if a.size else float("nan")


#: The rule scores the policy that was SHIPPED. Read by `_per_seed_values`, which
#: substitutes the restored checkpoint's row when a seed restored one. An
#: attribute rather than a name test, the `_PRUNERS_WITH_CFG` idiom: a rule
#: declares the property instead of being special-cased at the lookup.
ckpt_final.scores_shipped = True  # type: ignore[attr-defined]


@register("checkpoint_aggregation", "max_over_checkpoints")
def ckpt_max(values: Sequence[float]) -> float:
    """Eureka's rule: the max over EVERY logged epoch of the training curve
    (code; the paper reports "the maximum task metric values achieved from 10
    policy checkpoints sampled at fixed intervals", experiments.tex:34, †).

    `eureka.py:253` is `metric_cur_max = max(tensorboard_logs[metric])` over the
    full per-epoch log (~3000 points at `max_iterations: 3000`) and :256 appends
    that to `successes`; the `[::epoch_freq]` stride at :252 (~10 values) is
    only the list VERBALISED in the reflection. The 5 runs are the final
    evaluation (:382, each run's own max). BIRD applies the same max over the
    5-20 SB3 evaluation checkpoints `train.checkpoint_interval` produces -- an
    operational approximation of the statistic, not the same one (it is not
    "max over ~10 checkpoints x 5 runs").

    **Optimistically biased** (§4). E[max] > max E whenever checkpoint
    evaluation is noisy, so a reward whose training is merely *unstable* scores
    like one that is good, and the bias grows with the number of checkpoints --
    i.e. with `train.checkpoint_interval`, a §3 key. Kept because it is the
    published point, flagged because ablating it is free."""
    a = _clean(values)
    return float(a.max()) if a.size else float("nan")


@register("checkpoint_aggregation", "auc")
def ckpt_auc(values: Sequence[float]) -> float:
    """Normalised area under the learning curve (trapezoidal, divided by span).

    Rewards *fast* learning as well as final performance and is far less
    sensitive to per-checkpoint noise than `max`. Published by none of the
    methods here -- a free improvement over Eureka's rule (§4)."""
    a = _clean(values)
    if a.size == 0:
        return float("nan")
    if a.size == 1:
        return float(a[0])
    return float(_TRAPZ(a) / (a.size - 1))


@register("checkpoint_aggregation", "last_k_mean")
def ckpt_last_k_mean(values: Sequence[float]) -> float:
    """Mean of the last `k` checkpoints (k=_LAST_K; the schema has no knob).

    The cheap de-biased alternative to `max_over_checkpoints`: it estimates the
    same converged policy `final` does, with the variance of k evaluations
    instead of one. Also unrun (§4)."""
    a = _clean(values)
    if a.size == 0:
        return float("nan")
    return float(a[-min(_LAST_K, a.size):].mean())


@register("checkpoint_aggregation", "iqm")
def ckpt_iqm(values: Sequence[float]) -> float:
    """Interquartile mean over the checkpoint curve -- `auc` with the tails
    trimmed, so one collapsed evaluation cannot set the score."""
    a = _clean(values)
    return _iqm(a) if a.size else float("nan")


@register("checkpoint_aggregation", "max_over_training_epochs")
def ckpt_max_training_epochs(values: Sequence[float]) -> float:
    """Eureka's ACTUAL statistic: the max over the full per-update training
    series, not over the held-out checkpoint curve.

    `max_over_checkpoints` above is the operational approximation this replaces
    for `eureka.yaml`: it maxes the 5-20 rows `MIN_CHECKPOINTS`/`MAX_CHECKPOINTS`
    allow, each a held-out greedy evaluation. The source maxes every entry of
    the training log -- `eureka.py:253` is `metric_cur_max =
    max(tensorboard_logs[metric])` over ~3000 per-epoch points at
    `max_iterations: 3000`, and `:256` appends exactly that to `successes`, the
    ranking input. The `[::epoch_freq]` stride at `:252` is only the ~10 values
    VERBALISED in the reflection and never enters the number.

    **The arithmetic is the same max; the SERIES is the whole difference**, so
    this function is `ckpt_max`'s body and the work is in the routing. It
    carries `reads_training_epochs`, which `_per_seed_values` checks to feed it
    `seed_metrics[i]["training_epoch_metrics"]` instead of the checkpoint curve
    -- the same mechanism `ckpt_final.scores_shipped` uses to read shipped rows.
    Handing it the checkpoint curve by accident would silently make it
    `max_over_checkpoints` under a different name, which is why the routing is
    an attribute on the function rather than a name comparison at the call site.

    **More optimistically biased than `max_over_checkpoints`, not less.** E[max]
    grows with the number of draws, and this maxes hundreds-to-thousands of
    update windows where the other maxes 5-20 evaluations. That is not a
    regression: it is what the published method does, and reproducing it is the
    point. `auc`/`last_k_mean`/`iqm` remain the unbiased comparisons."""
    a = _clean(values)
    return float(a.max()) if a.size else float("nan")


#: Read by `_per_seed_values`: this rule scores the per-update TRAINING series
#: (`seed_metrics[i]["training_epoch_metrics"]`), not the checkpoint curve.
ckpt_max_training_epochs.reads_training_epochs = True  # type: ignore[attr-defined]


# ==========================================================================
# seed_aggregation -- values -> float
#
# `train.seeds_per_candidate` is 1 in every published search loop except LIMEN
# (§3), so this axis is usually aggregating a list of length one. It matters
# exactly when someone fixes that, which is cheap and probably worth an
# ablation on its own.
# ==========================================================================


@register("seed_aggregation", "mean")
def seed_mean(values: Sequence[float]) -> float:
    """Arithmetic mean across inner-loop seeds -- the default everywhere."""
    a = _clean(values)
    return float(a.mean()) if a.size else float("nan")


@register("seed_aggregation", "median")
def seed_median(values: Sequence[float]) -> float:
    """Median across seeds: ignores a single diverged run entirely."""
    a = _clean(values)
    return float(np.median(a)) if a.size else float("nan")


@register("seed_aggregation", "iqm")
def seed_iqm(values: Sequence[float]) -> float:
    """Interquartile mean across seeds (rliable's estimator). Needs >=4 seeds
    to differ from the mean, which no published method here supplies."""
    a = _clean(values)
    return _iqm(a) if a.size else float("nan")


@register("seed_aggregation", "min")
def seed_min(values: Sequence[float]) -> float:
    """Worst seed. The risk-averse variant nobody has tried (§4): it
    selects rewards that train *reliably* rather than rewards that once
    trained well, which is the failure mode 1-seed argmax is exposed to."""
    a = _clean(values)
    return float(a.min()) if a.size else float("nan")


# ==========================================================================
# Fitness plumbing shared by every source
# ==========================================================================


def _failure_value(ctx: Any) -> float:
    """`select.failure_value` -- Eureka's -10000 (§5).

    The contract: a failed or screened-out candidate must **lose every
    comparison but stay recorded**. `None` would drop it out of the ranking
    entirely (and out of the execute-rate denominator's meaning), `-inf` breaks
    means and significance tests, `0.0` can win on a task with negative
    rewards -- and is nonetheless a published value: RF-Agent backs a failed
    node up as `reward_fail_bound = 0` (rfagent_algo.py:386 ‡; its declared
    `DUMMY_FAILURE = -10000` at :109 is read by nothing), so it has to be
    reachable. `is None`, not `or`: `0.0 or -10000.0` is -10000.0, which ran
    that method's failures at Eureka's sentinel while its config said 0."""
    v = ctx.cfg.get("select.failure_value")
    return -10000.0 if v is None else float(v)


def _has_evidence(ctx: Any, result: TrainResult) -> bool:
    """Is there anything to score?

    Four ways there is not, and they are different populations: the
    program never compiled; a screen rejected it and `evaluate.skip_if_screened`
    says a screened candidate is never evaluated (CARD -- its feedback comes
    from the screen itself); training was skipped; training raised."""
    c = result.candidate
    if not c.valid:
        return False
    if c.screened_out and ctx.cfg.get("evaluate.skip_if_screened", False):
        return False
    if not result.trained:
        return False
    if result.error:
        return False
    return True


def _screen_measurement(candidate: Any) -> Optional[Tuple[float, str]]:
    """The success rate the rejecting screen itself measured, if it measured one.

    Reads the LAST screen-phase rejection in `verify_records` (the cascade's
    `_reject` writes `success_rate=rate` when the short run completed and
    nothing when it crashed -- `screens.py::screen_cascade`). Returns
    `(rate, screen_name)`, or None when the screen measured nothing: a crash
    is the release's `add_failure -> return None` population and keeps the
    sentinel.
    """
    for rec in reversed(candidate.verify_records):
        if rec.get("phase") != "screen" or rec.get("ok") is not False:
            continue
        rate = rec.get("success_rate")
        if rate is None:
            return None
        return float(rate), str(rec.get("screen") or rec.get("check") or "screen")
    return None


def _blank(ctx: Any, result: TrainResult, source: str) -> CandidateReport:
    """A report with the failure sentinel already applied where it applies.

    One carve-out (`evaluate.screened_fitness: screen_measurement`, LIMEN ‡):
    a screen-rejected candidate whose screen
    MEASURED something carries that measurement as its fitness instead of the
    sentinel. The release archives exactly this population --
    `SHORT_TRAIN_REJECTED` keeps `metrics=short_metrics` so `passed` stays True
    (`evaluator.py:45-49,66-68,237-258`) and `database.py::add` (L225-299)
    grid-places any passed, non-CRASHED result -- so a 0.005-rate cascade
    reject can honestly occupy a niche and parent, while a crash (no metrics)
    still sentinels. `meta["fitness_from_screen"]` is the flag §5's archive
    eligibility gate reads; the value is never compared against the sentinel.
    The default `failure_value` applies the sentinel to every screened
    candidate.

    `None` stays distinct from 0.0 and from `select.failure_value` throughout:
    a measured 0.0 rate is a fitness of 0.0, not a sentinel.
    """
    rep = CandidateReport(cand_id=result.cand_id, candidate=result.candidate,
                          result=result, fitness_source=source)
    if not _has_evidence(ctx, result):
        c = result.candidate
        if c.valid and c.screened_out and \
                ctx.cfg.get("evaluate.screened_fitness", "failure_value") == "screen_measurement":
            measured = _screen_measurement(c)
            if measured is not None:
                rate, screen = measured
                rep.fitness = rate
                rep.meta["fitness_from_screen"] = True
                rep.meta["fitness_note"] = (
                    f"short-run measurement from the '{screen}' screen that "
                    f"rejected this candidate (evaluate.screened_fitness)")
                return rep
        rep.fitness = _failure_value(ctx)
        rep.meta["fitness_note"] = (
            result.skip_reason or result.error or result.candidate.failure
            or "no training evidence")
    return rep


def _curve_key(d: Dict[str, Any]) -> Optional[float]:
    for k in ("step", "steps", "iteration", "epoch", "env_steps"):
        v = _num(d.get(k))
        if v is not None:
            return v
    return None


def _ordered(curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A curve in step order, when every row carries a step key."""
    if all(_curve_key(d) is not None for d in curve):
        return sorted(curve, key=lambda d: _curve_key(d) or 0.0)
    return curve


def _checkpoint_curves(result: TrainResult) -> List[List[Dict[str, Any]]]:
    """`result.checkpoints`, grouped by a `seed` field when its rows carry one.

    THIS IS THE POOLED CURVE, NOT ONE PER SEED, on every backend in the tree:
    both backends set `result.checkpoints = _mean_curve(per_seed_curves)`, the
    seed-MEAN curve, whose rows carry an AVERAGED `seed` field -- so the
    grouping below yields exactly one curve however many seeds trained. It is
    what the reflection prose and `_candidate_curve` read (one curve, the
    published shape), and it is NOT what §4's fitness pipeline reads:
    `_per_seed_values` goes through `_seed_curves`, because collapsing seeds
    here would make `evaluate.fitness.seed_aggregation` inert on every
    multi-seed run. The `seed` grouping is kept for a hand-built or foreign
    result that labels a flat list per seed."""
    ckpts = [d for d in result.checkpoints if isinstance(d, dict)]
    if not ckpts:
        return []
    if not any("seed" in d for d in ckpts):
        return [ckpts]
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for d in ckpts:
        groups.setdefault(str(d.get("seed", 0)), []).append(d)
    # str keys: deterministic without assuming int seeds
    return [_ordered(groups[key]) for key in sorted(groups)]


def _seed_curves(result: TrainResult) -> List[List[Dict[str, Any]]]:
    """One checkpoint curve PER SEED -- the input to §4's two-step collapse.

    Both backends leave each seed's own curve intact under
    `seed_metrics[i]["checkpoints"]` (the `_mean_curve` docstring's promise),
    and that is what this reads. `result.checkpoints` is the seed-mean curve
    and cannot be un-averaged, so it is the fallback only when no seed row
    carries a curve (a hand-built result, an artifact from before the seed
    rows existed) -- then `_checkpoint_curves` hands back one pooled curve and
    `seed_aggregation` sees one value, which the report's `n_seeds` records.

    With one seed the two readings are the same numbers: the mean of one row
    is that row, so a single-seed run scores bit-identically either way."""
    out: List[List[Dict[str, Any]]] = []
    for m in (result.seed_metrics or []):
        if not isinstance(m, dict):
            continue
        curve = [d for d in (m.get("checkpoints") or []) if isinstance(d, dict)]
        # A seed with an EMPTY curve is dropped, so `n_seeds` on the report
        # counts one fewer for this candidate. Deliberate: a seed that trained
        # and logged nothing is a `seed_error`, and `fold_seed` sets
        # `result.error` for it, which fails the candidate as a whole
        # (`training.py`, `fatal_error`) -- the aggregate is never read.
        if curve:
            out.append(_ordered(curve))
    return out or _checkpoint_curves(result)


def _pick(d: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    """First of `keys` present in `d` with a finite numeric value."""
    for k in keys:
        if k in d:
            v = _num(d[k])
            if v is not None:
                return v
    return None


def _series(records: Sequence[Dict[str, Any]], keys: Sequence[str]) -> List[float]:
    out = []
    for d in records:
        v = _pick(d, keys)
        if v is not None:
            out.append(v)
    return out


def _env_defines_success(ctx: Any) -> bool:
    """Whether the env's `success()` measures anything. The gym MuJoCo family
    ships no success check and returns False on every step (`defines_success =
    False` on its factory, the same flag `_check_coherence` reads to refuse
    `fitness.source: success_rate` there). Printing "success: mean 0.000"
    beside a task_score of 1.00 for that env would tell every reflecting LLM
    the policy had failed a check that does not exist -- and RF-Agent's own tip
    (1) reads task_score AS the success rate, so the two lines would contradict
    each other in every prompt."""
    try:
        from ..registry import get as _get
        factory = _get("env", ctx.cfg["problem.env_id"])
    except Exception:  # noqa: BLE001 -- no registry entry means no claim either way
        return True
    return getattr(factory, "defines_success", True) is not False


def _shipped_rows(result: TrainResult) -> List[Dict[str, Any]]:
    """Per seed, the checkpoint row of the policy stage 3 SHIPPED -- but only
    when at least one seed restored an earlier checkpoint (`restored_checkpoint`
    non-null under `train.checkpoint_selection: best_by_reward`). `[]` otherwise,
    so a run that shipped its last checkpoints takes the ordinary curve path and
    its artifact is unchanged. A seed whose shipped round is missing from its own
    curve is skipped rather than guessed at."""
    seeds = [m for m in (result.seed_metrics or []) if isinstance(m, dict)]
    if not any(m.get("restored_checkpoint") is not None for m in seeds):
        return []
    rows: List[Dict[str, Any]] = []
    for m in seeds:
        want = m.get("shipped_checkpoint")
        curve = [d for d in (m.get("checkpoints") or []) if isinstance(d, dict)]
        if want is None or not curve:
            continue
        hit = [d for d in curve if _num(d.get("round")) == float(want)]
        if hit:
            rows.append(hit[-1])
    return rows


def _per_seed_values(ctx: Any, result: TrainResult, keys: Sequence[str],
                     rep: CandidateReport) -> List[float]:
    """checkpoint_aggregation applied per seed -> one value per seed.

    The two-step collapse (checkpoints, then seeds) is the whole of §4's
    fitness pipeline, and the order is load-bearing: aggregating seeds first
    and then taking `max_over_checkpoints` would maximise over an average and
    hide exactly the instability the max is biased by. So the curves come
    from `_seed_curves`, not from `result.checkpoints`: off the seed-MEAN
    curve `seed_aggregation` would always receive one value, `min` would equal
    `mean`, `per_seed_fitness` would have length 1 and `select.significance`
    could never see two seeds.

    `rep.meta["n_checkpoints"]` is the number of checkpoint rows the collapse
    read -- summed over seeds, so three seeds of ten rounds report 30."""
    ckpt_agg = _component("checkpoint_aggregation",
                               ctx.cfg.get("evaluate.fitness.checkpoint_aggregation", "final"))
    curves = _seed_curves(result)
    # `train.checkpoint_selection` restored an earlier checkpoint on some seed,
    # and this rule scores the policy that was shipped (`ckpt_final.scores_shipped`):
    # each seed's own row at `shipped_checkpoint`, off its own curve. One value
    # per seed, the same shape the curve path below hands `seed_aggregation`,
    # so a candidate that restored and a pool-mate that did not are scored
    # under the same seed semantics. Only when a restore happened: with nothing
    # restored the two readings agree and the ordinary path runs.
    # `max_over_training_epochs` scores the per-update TRAINING series, not the curve.
    # Placed before the shipped-checkpoint branch because the two describe
    # different objects -- `scores_shipped` picks a ROW of the checkpoint curve,
    # this replaces the series outright -- and no rule sets both.
    if getattr(ckpt_agg, "reads_training_epochs", False):
        vals = []
        for m in (result.seed_metrics or []):
            if not isinstance(m, dict):
                continue
            s = [float(v) for v in (m.get("training_epoch_metrics") or [])
                 if isinstance(v, (int, float))]
            if s:
                v = _num(ckpt_agg(s))
                if v is not None:
                    vals.append(v)
        if vals:
            rep.meta["checkpoint_aggregation_applied"] = True
            rep.meta["fitness_from"] = "training_epoch_series"
            rep.meta["n_training_epochs"] = sum(
                len(m.get("training_epoch_metrics") or [])
                for m in (result.seed_metrics or []) if isinstance(m, dict))
            return vals
        # No series on any seed row -- a backend or an older artifact that does
        # not record it. Fall through to the checkpoint curve rather than
        # returning nothing, and SAY SO: a silent fallback here would make this
        # rule indistinguishable from `max_over_checkpoints` in the report,
        # which is the one confusion that would invalidate a comparison between
        # the two.
        rep.meta["training_epoch_series_missing"] = True

    if getattr(ckpt_agg, "scores_shipped", False):
        shipped = _shipped_rows(result)
        if shipped:
            vals = [v for v in (_pick(row, keys) for row in shipped) if v is not None]
            if vals:
                rep.meta["checkpoint_aggregation_applied"] = True
                rep.meta["fitness_from"] = "shipped_checkpoint"
                rep.meta["restored_checkpoints"] = [
                    m.get("restored_checkpoint") for m in result.seed_metrics
                    if isinstance(m, dict)]
                rep.meta["shipped_per_seed"] = [float(v) for v in vals]
                rep.meta["n_checkpoints"] = sum(len(c) for c in curves)
                return [float(v) for v in vals]
    vals = []
    for curve in curves:
        s = _series(curve, keys)
        if s:
            v = _num(ckpt_agg(s))
            if v is not None:
                vals.append(v)
    if vals:
        rep.meta["checkpoint_aggregation_applied"] = True
        rep.meta["n_checkpoints"] = sum(len(c) for c in curves)
        return vals

    # No checkpoint curve carried the metric. Fall back to the per-seed summary
    # dicts -- and record that `checkpoint_aggregation` was a no-op, because a
    # config that believes it is running Eureka's `max_over_checkpoints` while
    # the backend reports one number per seed is running `final` and does not
    # know it.
    rep.meta["checkpoint_aggregation_applied"] = False
    out = []
    for m in result.seed_metrics:
        if isinstance(m, dict):
            v = _pick(m, keys)
            if v is not None:
                out.append(v)
    return out


def _component(kind: str, name: str) -> Callable[..., Any]:
    """Local alias for `registry.get`, imported lazily.

    Lazy because `registry.load_all()` imports this module: a module-level
    `from ..registry import get` would be fine, but resolving the *name* at
    call time is what keeps `evaluate.fitness.checkpoint_aggregation` a config
    value rather than a closure captured at import."""
    from ..registry import get
    return get(kind, name)


def _normalise_pool(ctx: Any, reports: List[CandidateReport]) -> None:
    """`evaluate.fitness.normalisation` -- applied last, over the whole pool.

    Failure sentinels are excluded from the statistics and left untouched: a
    -10000 inside a min-max range would compress every real candidate into the
    top 1e-4 of [0,1], and rescaling the sentinel itself would let it stop
    losing. It is already below any normalised value, which is the invariant
    §5 needs."""
    mode = ctx.cfg.get("evaluate.fitness.normalisation", "none")
    if mode == "none":
        return
    sentinel = _failure_value(ctx)
    live = [r for r in reports
            if r.fitness is not None and math.isfinite(r.fitness) and r.fitness != sentinel]
    if not live:
        return

    if mode == "min_max":
        vals = [r.fitness for r in live if r.fitness is not None]
        lo, hi = min(vals), max(vals)
        span = hi - lo
        for r in live:
            r.meta["raw_fitness"] = r.fitness
            r.fitness = 0.5 if span < 1e-12 else (float(r.fitness) - lo) / span
        return

    if mode == "human_normalised":
        # Eureka's human-normalised score, (x - random) / (human - random).
        # The two baselines are task properties, so they belong to the env
        # adapter; when it does not publish them we leave the raw number rather
        # than invent a scale, and say so once.
        base = _env_baselines(ctx)
        if base is None:
            log.debug("evaluate.fitness.normalisation=human_normalised but the env "
                      "publishes no human/random baselines; leaving fitness raw")
            return
        rand, human = base
        span = human - rand
        for r in live:
            r.meta["raw_fitness"] = r.fitness
            r.fitness = 0.0 if abs(span) < 1e-12 else (float(r.fitness) - rand) / span


def _env_baselines(ctx: Any) -> Optional[Tuple[float, float]]:
    env = getattr(ctx, "env", None)
    if env is None:
        return None
    d = getattr(env, "baselines", None)
    if isinstance(d, dict):
        rand = _num(d.get("random", 0.0))
        human = _num(d.get("human", d.get("expert")))
        if human is not None:
            return (rand if rand is not None else 0.0), human
    human = _num(getattr(env, "human_score", None))
    if human is not None:
        return _num(getattr(env, "random_score", 0.0)) or 0.0, human
    return None


def _finish(ctx: Any, reports: List[CandidateReport]) -> List[CandidateReport]:
    for r in reports:
        r.meta.setdefault("seed_aggregation", ctx.cfg.get("evaluate.fitness.seed_aggregation"))
        r.meta.setdefault("checkpoint_aggregation",
                          ctx.cfg.get("evaluate.fitness.checkpoint_aggregation"))
    _normalise_pool(ctx, reports)
    return reports


def _collapse(ctx: Any, rep: CandidateReport, per_seed: List[float]) -> None:
    """seed_aggregation over the per-seed values -> `rep.fitness`."""
    seed_agg = _component("seed_aggregation",
                               ctx.cfg.get("evaluate.fitness.seed_aggregation", "mean"))
    rep.per_seed_fitness = [float(v) for v in per_seed]
    if not per_seed:
        # Evidence was expected and did not arrive: that is a failed candidate,
        # not an unranked one.
        rep.fitness = _failure_value(ctx)
        rep.meta.setdefault("fitness_note", "no metric found in checkpoints or seed_metrics")
        return
    v = _num(seed_agg(per_seed))
    rep.fitness = v if v is not None else _failure_value(ctx)
    rep.meta["n_seeds"] = len(per_seed)


# ==========================================================================
# fitness_source
# ==========================================================================


@register("fitness_source", "ground_truth_metric")
def fitness_ground_truth(ctx: Any, state: Any, results: List[TrainResult]) -> List[CandidateReport]:
    """Eureka / DrEureka / RDA(+Ckpt) / LIMEN: the task metric `F` (§4).

    `evaluate.fitness.metric` names it -- Eureka's is `consecutive_successes` on
    IsaacGym, a sparse metric deliberately *not* shown to the generator
    (`generate.context.strip_existing_reward`, §1). The pipeline is fixed and
    the two knobs are the interesting part: `checkpoint_aggregation` over
    `result.checkpoints`, then `seed_aggregation` over seeds, then
    `evaluate.fitness.normalisation`."""
    metric = ctx.cfg.get("evaluate.fitness.metric", "task_success")
    # Ordered fallbacks: the configured metric first, then the conventional
    # names a backend uses for "the fitness I was asked to log" (`train.log`
    # includes `fitness`). Deliberately no return-like key -- `F` is the task
    # metric, never the candidate's own reward, or the search grades its own
    # homework.
    keys = (metric, "fitness", "gt_metric", "task_metric", "score")
    reports = []
    for res in results:
        rep = _blank(ctx, res, "ground_truth_metric")
        if rep.fitness is None:
            rep.meta["metric"] = metric
            _collapse(ctx, rep, _per_seed_values(ctx, res, keys, rep))
        reports.append(rep)
    return _finish(ctx, reports)


@register("fitness_source", "success_rate")
def fitness_success_rate(ctx: Any, state: Any, results: List[TrainResult]) -> List[CandidateReport]:
    """LIMEN: **trajectory-level success**, never the candidate's own reward.

    The distinction is the point. A reward is being searched over; scoring it by
    the return it itself defines is circular and is exactly the reward hacking
    the whole stage exists to detect. So this reads the environment's success
    flag off stored rollouts (`Trajectory.success`) or the backend's logged
    success rate -- both ground truth -- and never `Trajectory.ret`.

    Same two-step collapse as `ground_truth_metric` when success is logged at
    checkpoints; otherwise the rollouts themselves are the evidence and
    `checkpoint_aggregation` has nothing to bite on (recorded in meta)."""
    keys = ("success_rate", "success", "successes", "task_success", "is_success")
    reports = []
    for res in results:
        rep = _blank(ctx, res, "success_rate")
        if rep.fitness is None:
            per_seed = _per_seed_values(ctx, res, keys, rep)
            if not per_seed:
                trajs = _rollouts(ctx, res)
                if trajs:
                    per_seed = [float(np.mean([1.0 if t.success else 0.0 for t in trajs]))]
                    rep.meta["success_from"] = "trajectories"
                    rep.meta["n_rollouts"] = len(trajs)
            _collapse(ctx, rep, per_seed)
        reports.append(rep)
    return _finish(ctx, reports)


def _native_kinds(ctx: Any):
    """`(discrete_success.kind, reward.human.kind)` for this run, via the shared
    resolver -- the SAME read `_check_coherence` makes at load time."""
    from ..native_signal import kinds_for_env
    return kinds_for_env(env_id=ctx.cfg.get("problem.env_id"),
                         task_id=ctx.cfg.get("problem.task_id"))


def _native_success_per_seed(ctx: Any, res: TrainResult,
                             rep: CandidateReport) -> List[float]:
    """The env SUCCESS FLAG, per seed -- never a `custom_metric` alias."""
    from ..native_signal import NATIVE_SUCCESS_KEYS
    per_seed = _per_seed_values(ctx, res, NATIVE_SUCCESS_KEYS, rep)
    if not per_seed:
        trajs = _rollouts(ctx, res)
        if trajs:
            per_seed = [float(np.mean([1.0 if t.success else 0.0 for t in trajs]))]
            rep.meta["success_from"] = "trajectories"
            rep.meta["n_rollouts"] = len(trajs)
    return per_seed


@register("fitness_source", "native_success")
def fitness_native_success(ctx: Any, state: Any, results: List[TrainResult]) -> List[CandidateReport]:
    """N: the benchmark's SHIPPED binary success (native_success).

    Only where the task ships one -- `discrete_success.kind` in
    {discrete, native_authored}, which is exactly the tasks whose spec pins a
    verified reimplementation of the vendor's own success. Reads the env success
    FLAG (`NATIVE_SUCCESS_KEYS`), never a `task_metric` alias. On a task with no
    native success it REFUSES rather than fall back on the BIRD metric (coherence
    refuses the pin at load; this is the runtime belt-and-suspenders for an
    env-id-only run)."""
    from ..native_signal import has_native_success
    ds, _rw = _native_kinds(ctx)
    if not has_native_success(ds):
        raise ValueError(
            "evaluate.fitness.source='native_success' but the task for "
            f"problem.env_id={ctx.cfg.get('problem.env_id')!r} ships no native "
            "success (discrete_success.kind is not 'discrete'/'native_authored'); "
            "the native-signal rule refuses to score a supervised search on the "
            "BIRD-authored task_metric")
    reports = []
    for res in results:
        rep = _blank(ctx, res, "native_success")
        if rep.fitness is None:
            _collapse(ctx, rep, _native_success_per_seed(ctx, res, rep))
        reports.append(rep)
    return _finish(ctx, reports)


@register("fitness_source", "native_reward")
def fitness_native_reward(ctx: Any, state: Any, results: List[TrainResult]) -> List[CandidateReport]:
    """N: the benchmark's SHIPPED reward, as its RETURN (native_reward).

    The reference-reward return (`gt_return`), never the candidate's OWN reward
    (`return`/`reward_return`) and never `task_metric`. Available only where the
    task ships a reference reward (`reward.human.kind` != none); refuses
    otherwise (coherence at load, here at runtime)."""
    from ..native_signal import has_native_reward, NATIVE_REWARD_KEYS
    _ds, rw = _native_kinds(ctx)
    if not has_native_reward(rw):
        raise ValueError(
            "evaluate.fitness.source='native_reward' but the task for "
            f"problem.env_id={ctx.cfg.get('problem.env_id')!r} ships no reference "
            "reward (reward.human.kind: none); nothing native to score against")
    reports = []
    for res in results:
        rep = _blank(ctx, res, "native_reward")
        if rep.fitness is None:
            _collapse(ctx, rep, _per_seed_values(ctx, res, NATIVE_REWARD_KEYS, rep))
        reports.append(rep)
    return _finish(ctx, reports)


@register("fitness_source", "native")
def fitness_native(ctx: Any, state: Any, results: List[TrainResult]) -> List[CandidateReport]:
    """N (default): the env's OWN signal -- native_success where the task ships
    one, else the native_reward return, else REFUSE. Never the BIRD task_metric.

    The resolved leaf (native_success / native_reward) is recorded as each
    report's `fitness_source`, so the artifact says which quantity the fitness
    IS -- and the refuse set is exactly the five gymnasium tasks with neither
    signal (`reward.human.kind: none`), which coherence blocks at load."""
    from ..native_signal import (resolve_channel, NATIVE_SUCCESS, NATIVE_REWARD,
                                  NATIVE_REWARD_KEYS)
    ds, rw = _native_kinds(ctx)
    channel = resolve_channel(ds, rw)
    if channel == NATIVE_SUCCESS:
        reports = []
        for res in results:
            rep = _blank(ctx, res, "native_success")
            if rep.fitness is None:
                _collapse(ctx, rep, _native_success_per_seed(ctx, res, rep))
            reports.append(rep)
        return _finish(ctx, reports)
    if channel == NATIVE_REWARD:
        reports = []
        for res in results:
            rep = _blank(ctx, res, "native_reward")
            if rep.fitness is None:
                _collapse(ctx, rep, _per_seed_values(ctx, res, NATIVE_REWARD_KEYS, rep))
            reports.append(rep)
        return _finish(ctx, reports)
    raise ValueError(
        "evaluate.fitness.source='native' but the task for "
        f"problem.env_id={ctx.cfg.get('problem.env_id')!r} ships neither a native "
        "success nor a reference reward (reward.human.kind: none), so there is no "
        "native signal to select on; it must not fall back on the BIRD task_metric "
        "(the five gymnasium unsupervised-only tasks are unsupervised by construction)")


@register("fitness_source", "demo_margin")
def fitness_demo_margin(ctx: Any, state: Any, results: List[TrainResult]) -> List[CandidateReport]:
    """`evaluate.fitness.source: demo_margin`: rank on the verifier's score alone --
    nothing was trained.

    `candidate.meta["demo"]["score"]` = margin + monotonicity from the
    `demo_margin` screen (§2), in [-2, 2]. A candidate the sign test rejected
    (`meta["demo"]["passed_sign_test"] is False`) or that was never scored
    gets the failure sentinel through `_blank`, exactly like a
    candidate that produced no training evidence anywhere else. No task metric
    is read: this is CARD's "skip the RL" made total, with the single
    `final_retrain` at the end as the only policy the run ever trains.
    """
    reports = []
    for res in results:
        rep = _blank(ctx, res, "demo_margin")
        demo = (getattr(res.candidate, "meta", None) or {}).get("demo") or {}
        if rep.fitness is None or "score" in demo:
            # `trainable is not None` cannot be the gate: it is always true
            # (`trainable` is a bool property), so a sign-test REJECT would
            # keep its negative score as fitness and could win a
            # verifier-only round. `trainable` itself cannot be the gate either:
            # under keep=skip_all every scored passer is un-trainable by
            # design. The screen writes `passed_sign_test` for exactly this.
            if "score" in demo and demo.get("passed_sign_test") is True:
                rep.fitness = float(demo["score"])
                rep.meta["demo_margin"] = float(demo.get("margin", 0.0))
                rep.meta["demo_monotonicity"] = float(demo.get("monotonicity", 0.0))
                rep.meta["fitness_note"] = "verifier score (margin + monotonicity); no policy trained"
            elif rep.fitness is None:
                rep.fitness = _failure_value(ctx)
                rep.meta.setdefault("fitness_note", "no verifier score on this candidate")
        reports.append(rep)
    return _finish(ctx, reports)


@register("phase", "rejudge_incumbent")
def phase_rejudge_incumbent(ctx: Any, state: Any, reports: List[CandidateReport]) -> List[CandidateReport]:
    """Re-score the incumbent (`state.best`) under THIS round's subtasks and repeats.

    `evaluate.rejudge_incumbent` (unpublished). RDA's decomposition
    moves between rounds (`update.co_evolve.subtasks`), so a fitness the judge
    gave the incumbent in round 0 is on a different scale from the one it gives
    round 2's candidates -- and `select.require_improvement_over_incumbent` then
    compares two different rubrics, so one lucky early score can pin the
    incumbent for the rest of the run. This phase re-judges the incumbent's
    STORED rollouts --
    the same behaviour, the current rubric, `evaluate.vlm.repeats` times -- and
    overwrites `state.best.fitness` (and the per-subtask breakdown) before §5
    runs. It spends `rollouts_per_candidate` x `repeats` judge calls and no RL,
    and it reads nothing but frames and the reward code: no ground truth.

    Skipped, and said so in the journal, when there is no incumbent yet, when the
    incumbent's rollouts are not in memory (a resumed run: `checkpoint.py` does not
    carry trajectories), or when the fitness source is not the VLM.
    """
    best = getattr(state, "best", None)
    it = int(getattr(state, "iteration", -1))
    if best is None or getattr(best, "result", None) is None:
        # Round 0, or a run whose incumbent slot was never filled: nothing to re-score.
        # Journalled like every other skip, so "did the phase run" is answerable from
        # the artifact for every round.
        ctx.event("rejudge_incumbent", cand_id=None, skipped=True, iteration=it,
                  reason="no incumbent yet")
        return reports
    if ctx.cfg.get("evaluate.fitness.source") != "vlm_score":
        ctx.event("rejudge_incumbent", cand_id=getattr(best, "cand_id", None), skipped=True,
                  reason="evaluate.fitness.source is not vlm_score")
        return reports
    trajs = list(getattr(best.result, "trajectories", None) or [])
    if not trajs:
        ctx.event("rejudge_incumbent", cand_id=best.cand_id, skipped=True,
                  reason="incumbent has no stored rollouts (resumed run?)")
        log.info("  [4] rejudge   -> skipped: %s has no stored rollouts", best.cand_id)
        return reports
    before = best.fitness
    before_sub = dict(best.subtask_scores or {})
    # A fresh TrainResult carrying only what the judge reads, so `_blank` starts
    # from `fitness=None` and the scorer runs; `best.result` itself is untouched.
    probe = TrainResult(cand_id=best.cand_id, candidate=best.candidate, trained=True,
                        trajectories=trajs, seed_metrics=list(best.result.seed_metrics or []),
                        component_traces=dict(best.result.component_traces or {}),
                        policy_ref=best.result.policy_ref)
    rep = fitness_vlm_score(ctx, state, [probe])[0]
    best.fitness = rep.fitness
    best.per_seed_fitness = list(rep.per_seed_fitness)
    best.subtask_scores = dict(rep.subtask_scores)
    best.subtask_rationales = dict(rep.subtask_rationales)
    best.subtask_behaviors = dict(rep.subtask_behaviors)
    hist = list(best.meta.get("rejudged", []) or [])
    hist.append({"iteration": it, "fitness_before": before, "fitness_after": rep.fitness,
                 "subtasks": list(_subtasks(ctx, state)),
                 "vlm_queries": rep.meta.get("vlm_queries"), "n_rollouts": len(trajs)})
    best.meta["rejudged"] = hist
    ctx.event("rejudge_incumbent", cand_id=best.cand_id, skipped=False, iteration=it,
              fitness_before=before, fitness_after=rep.fitness,
              subtask_scores_before=before_sub, subtask_scores_after=dict(rep.subtask_scores),
              n_rollouts=len(trajs), vlm_queries=rep.meta.get("vlm_queries"))
    log.info("  [4] rejudge   -> incumbent %s: %s -> %s under %d subtask(s)", best.cand_id,
             _fmt(before) if before is not None else "none", _fmt(rep.fitness),
             len(rep.subtask_scores or {}))
    return reports


@register("fitness_source", "vlm_score")
def fitness_vlm_score(ctx: Any, state: Any, results: List[TrainResult]) -> List[CandidateReport]:
    """RDA: ONE structured VLM call per trajectory scores ALL subtasks in order;
    the mean over every (rollout, repeat, subtask) score is the fitness.

    RDA's generalisation of `F` to an instruction-conditioned `S: Pi x I -> R`
    (Problem setting). The call protocol is App. 7.3's -- a single sequential
    analysis per trajectory with the failure-cascade rule, judged by
    `ctx.generator` (the Agent VLM), with the reward code and per-step component
    values as evidence; `_vlm_trajectory_analysis` carries the argument. The
    score is averaged over
    `evaluate.rollouts_per_candidate` rollouts and `evaluate.vlm.repeats`
    repeats -- the paper's 4 repeats are the post-hoc alignment metric's (§5.1
    main.tex:367, §10.1 appendix.tex:1452) and live in `alignment_rate.repeats`;
    the in-loop count is unstated (§4, `evaluate.vlm.repeats`; no ‡, RDA has no
    release), so the default is 1 and the knob exists to test the claim rather
    than to assert it.

    `subtask_scores` / `subtask_rationales` / `subtask_behaviors` are the
    per-subtask breakdown §6 renders when `evaluate.feedback.granularity:
    per_subtask`; RDA attributes its gains to exactly that credit assignment.
    The rationale kept per subtask is ALL of them, one per (rollout, repeat),
    labeled -- §4.2's `p_{n,j}` aggregates the whole set `{rho_{n,k,j}}_{k=1..K}`
    into recurring failure modes, and a min-score-only fold would compute
    'recurring' from a sample of one.
    Rationales cost tokens, so they are requested only when the config will use
    them.

    `checkpoint_aggregation` is *not* applied: a VLM judges a rollout, not a
    learning curve, so there is no curve to collapse. `seed_aggregation` still
    is -- one value per seed's rollout set."""
    subtasks = _subtasks(ctx, state)
    repeats = max(1, int(ctx.cfg.get("evaluate.vlm.repeats", 1) or 1))
    want_reason = bool(ctx.cfg.get("evaluate.feedback.require_rationale", False)) or \
        ctx.cfg.get("evaluate.feedback.granularity") == "per_subtask"
    lo, hi = _scale_bounds(ctx)

    # `-1` files the records under `judgments/post.jsonl` rather than claiming an
    # iteration this stage was not called from: every in-loop caller passes a
    # RunState, and a hand-built context in a test need not.
    it = int(getattr(state, "iteration", -1))

    reports = []
    # TWO PASSES, AND THE SPLIT IS THE POINT. Pass 1 below is per candidate
    # and stays SERIAL because `sample_frames` drives the shared env renderer
    # (see the note beside `jobs`); pass 2 runs every candidate's judge
    # round-trips through ONE pool. With a pool per candidate, the next
    # candidate's frames would wait on the previous candidate's last verdict,
    # with the pool idle whenever a candidate was between its own jobs
    # (measured at ~1 h per evaluate stage).
    #
    # NOT a pool per candidate around the existing one: nesting would make the
    # in-flight round-trip count the PRODUCT of two limits and silently exceed
    # `llm.max_concurrent_requests`, which is the one number the judge's
    # transport budget is expressed in.
    prepared: List[Tuple[Any, Optional[Dict[str, Any]]]] = []
    for res in results:
        rep = _blank(ctx, res, "vlm_score")
        if rep.fitness is not None:
            prepared.append((rep, None))
            continue

        trajs = _rollouts(ctx, res)
        # Sampled once per trajectory and reused across subtasks and repeats:
        # the pixels do not change with the question, and on Meta-World a frame
        # is a MuJoCo render.
        #
        # One provenance dict per rollout, filled in place by `sample_frames`
        # (see `_vlm_frames`). It is the only channel that can carry which
        # instants the pixels came from: the frames themselves are PNG bytes by
        # the time this loop sees them, and the stride that produced them is
        # known to nothing else.
        provs: List[Dict[str, Any]] = [{} for _ in trajs]
        frames_by_traj = [_vlm_frames(ctx, t, provs[i],
                                      client=getattr(ctx, "generator", None))
                          for i, t in enumerate(trajs)]
        blind = sum(1 for f, _note in frames_by_traj if not f)
        # Written BEFORE the first query, so a run killed mid-judgment still
        # says what its judge was looking at. `note_frames` returns the frame
        # set's id -- a digest of the pixels -- and every query below points at
        # it rather than repeating twenty hashes per subtask per repeat.
        frame_sets = [judgments.note_frames(ctx, provs[i], iteration=it,
                                            cand_id=rep.cand_id, rollout=i,
                                            png=frames_by_traj[i][0])
                      for i in range(len(trajs))]
        # ALWAYS journalled, not only on the bad path. "No warning" is not
        # evidence that the judge had eyes, so the artifact records the frame
        # count per rollout either way and a collector can filter on it.
        ctx.event("vlm_frames", cand_id=rep.cand_id, n_rollouts=len(trajs),
                  n_blind_rollouts=blind,
                  images_per_rollout=[len(f) for f, _n in frames_by_traj],
                  reason=next((n for f, n in frames_by_traj if not f), ""))
        if blind:
            note = next(n for f, n in frames_by_traj if not f)
            ctx.budget.record_blind_judgment(blind * repeats * len(subtasks))
            log.warning("%s: %d of %d rollout(s) scored with NO frames (%s); the VLM "
                        "judge is a text-only comparator for those",
                        rep.cand_id, blind, len(trajs), note)
        per_subtask: Dict[str, List[float]] = {s: [] for s in subtasks}
        rationales: Dict[str, List[str]] = {s: [] for s in subtasks}
        behaviors: Dict[str, List[str]] = {s: [] for s in subtasks}
        # One judgment per (repeat, rollout) -- ONE call covers every subtask
        # (App. 7.3's protocol; see `_vlm_trajectory_analysis`'s docstring)
        # -- independent by construction, so the
        # round-trips may overlap (`judge_concurrency`; 1 = the serial double
        # loop). Frames were sampled ONCE above, before anything overlaps:
        # `sample_frames` drives the shared env renderer. Measured serially on
        # RDA, a per-(trajectory, subtask) block took 4,410 s in one iteration,
        # one round-trip at a time; the single-call protocol also divides the
        # round-trip count by J.
        #
        # The job tuple carries `roll`/`repeat`/`note` as well as the inputs
        # because the judge TRACE needs them: a query record names
        # which rollout and which repeat it was, and the concurrent fold cannot
        # recover either from a bare enumerate once the order is a pool's.
        jobs = [(traj, frames, roll, repeat, note)
                for repeat in range(repeats)
                for roll, (traj, (frames, note)) in enumerate(zip(trajs, frames_by_traj))]

        # BOUND NOW, NOT READ LATE. `_analyse` runs in pass 2, after this loop
        # has moved on, so a free `rep` / `provs` / `frame_sets` would resolve
        # to the LAST candidate's: every judgment would see its own frames
        # beside the last candidate's reward code, reward trace and frame
        # labels. Default arguments freeze this candidate's.
        def _analyse(job: Tuple[Any, ...], rep: Any = rep, provs: Any = provs,
                     frame_sets: Any = frame_sets
                     ) -> List[Tuple[Optional[float], str, str]]:
            traj, frames, roll, repeat, note = job
            return _vlm_trajectory_analysis(
                ctx, traj, subtasks, rep.candidate.reward_code or "",
                want_reason, frames=frames, prov=provs[roll],
                candidate=rep.candidate,
                trace={"iteration": it, "cand_id": rep.cand_id,
                       "rollout": roll, "repeat": repeat,
                       "frame_set": frame_sets[roll],
                       # The frame set's reason in preference to `note`:
                       # `_vlm_frames` collapses every sampling failure into
                       # "no frames could be sampled", while `sample_frames`
                       # knows which of them it was ("the rollout carries no
                       # states to render", "the renderer is unusable"). The
                       # remedies differ, so the query record carries the
                       # specific one.
                       "blind_reason": (provs[roll].get("blind_reason") or note)})

        prepared.append((rep, {"jobs": jobs, "analyse": _analyse, "trajs": trajs,
                               "frames_by_traj": frames_by_traj,
                               "per_subtask": per_subtask, "rationales": rationales,
                               "behaviors": behaviors}))

    # -- pass 2: ONE pool over every candidate's jobs ---------------------
    #
    # `flat` is in candidate order and, within a candidate, in that candidate's
    # own job order, so both folds below stay JOB-ORDERED -- never
    # completion-ordered. `pool.map` preserves submission order by contract.
    flat: List[Tuple[int, Any]] = [
        (ci, j) for ci, (_rep, st) in enumerate(prepared) if st for j in st["jobs"]]

    def _run(item: Tuple[int, Any]) -> List[Tuple[Optional[float], str, str]]:
        ci, job = item
        return prepared[ci][1]["analyse"](job)

    workers = judge_concurrency(ctx, len(flat), client=getattr(ctx, "generator", None))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="bird-vlm") as pool:
            # MATERIALISED INSIDE THE `with`: `Executor.map` returns a lazy
            # generator, and letting it
            # escape the block would have the fold consume it after
            # `shutdown()`. It happens to work, because `map` submits eagerly,
            # but it relies on that rather than on the contract.
            outcomes_all: Iterator[Any] = iter(list(pool.map(_run, flat)))
    else:
        # A generator, not a list: the serial path still asks as it folds and
        # never holds every outcome at once -- byte-for-byte the serial loop,
        # which is what the parity test pins.
        outcomes_all = (_run(t) for t in flat)

    # -- pass 3: fold per candidate, in candidate index order --------------
    for rep, st in prepared:
        if st is None:
            reports.append(rep)
            continue
        jobs = st["jobs"]; trajs = st["trajs"]; frames_by_traj = st["frames_by_traj"]
        per_subtask = st["per_subtask"]; rationales = st["rationales"]
        behaviors = st["behaviors"]
        outcomes = itertools.islice(outcomes_all, len(jobs))
        label_them = repeats * len(trajs) > 1
        for (traj, frames, roll, repeat, note), verdicts in zip(jobs, outcomes):
            for sub, (score, behavior, analysis) in zip(subtasks, verdicts):
                if score is not None:  # an abstained subtask carries no value
                    per_subtask[sub].append(score)
                # ALL rationales, labeled per rollout/repeat -- never only the
                # min-score one. §4.2's summarisation aggregates the whole set
                # {rho_{n,k,j}} into recurring failure modes, and a fold that
                # kept one rationale would compute 'recurring' from a sample of
                # one.
                label = (f"rollout {roll}" + (f" (repeat {repeat})" if repeats > 1
                                              else "") + ": ") if label_them else ""
                if want_reason and analysis:
                    rationales[sub].append(label + analysis)
                if behavior:
                    behaviors[sub].append(label + behavior)

        means = {s: float(np.mean(v)) for s, v in per_subtask.items() if v}
        rep.subtask_scores = {s: round(v, 4) for s, v in means.items()}
        if want_reason:
            rep.subtask_rationales = {s: "\n".join(v) for s, v in rationales.items() if v}
        rep.subtask_behaviors = {s: "\n".join(v) for s, v in behaviors.items() if v}
        rep.meta["score_scale"] = ctx.cfg.get("evaluate.feedback.score_scale", "[0,1]")
        # Round-trips actually made vs verdicts extracted from them: one call
        # per (repeat, rollout) carries len(subtasks) judgments.
        rep.meta["vlm_queries"] = repeats * len(trajs)
        rep.meta["vlm_judgments"] = repeats * len(trajs) * len(subtasks)
        rep.meta["n_rollouts"] = len(trajs)
        # In the artifact, not only in the log: a score produced from frames and
        # one produced from a text digest are the same float and mean different
        # things, so the count of pixels behind each report travels with it.
        rep.meta["vlm_images_per_rollout"] = [len(f) for f, _n in frames_by_traj]

        if means:
            # Normalise the configured scale into [0,1] so `select.failure_value`
            # keeps its meaning across `score_scale` settings.
            raw = float(np.mean(list(means.values())))
            _collapse(ctx, rep, [(raw - lo) / (hi - lo) if hi > lo else raw])
        else:
            _collapse(ctx, rep, [])
        reports.append(rep)
    return _finish(ctx, reports)


@register("fitness_source", "preference_bt")
def fitness_preference_bt(ctx: Any, state: Any, results: List[TrainResult]) -> List[CandidateReport]:
    """Gran Turismo: the Bradley-Terry strength fitted over VLM preferences.

    GT has no ground-truth fitness at all (`problem.fitness_access: none` in
    spirit); candidates are ranked only against each other.

    **What the scalar is comparable to is `evaluate.preferences.scope`, and
    this source cannot tell.** Under the default `within_iteration` -- GT's
    rule -- the strengths are renormalised per round and never cross rounds, so
    two numbers from different iterations are on the same scale only by
    accident. Under `cumulative` (REvolve) one fit spans every individual with
    a comparison on record, so they do cross rounds, and
    `bird/components/preferences.py::_refit_carried` rewrites the fitness of
    carried individuals each round to keep them on it. Either way the number
    reaching `rep.fitness` here is whatever `bt_strength` last said; a reader
    comparing two of them across iterations has to check the key first.

    **Ordering note.** `bird.evaluate` calls the fitness source *before* the
    preferences phase, so at this point `report.meta["bt_strength"]` -- the key
    `bird/components/preferences.py` agrees to write -- does not exist yet. This
    source therefore registers itself as *deferred*: it emits reports with
    `fitness = None`, and the scalar is resolved from meta by the default
    feedback builder, which is the last component `evaluate` runs. That is
    ordering, not method dispatch: the trigger is `report.fitness_source`, a
    registry value, and any future source with the same dependency joins the
    table below rather than adding a branch."""
    reports = []
    for res in results:
        rep = _blank(ctx, res, "preference_bt")
        if rep.fitness is None:
            rep.meta["fitness_pending"] = "bt_strength"
            # Some pair strategies stamp strengths onto the candidate before
            # stage 4 (e.g. a screen that already ran BT); take it if present.
            _resolve_bt(ctx, rep)
        reports.append(rep)
    return _finish(ctx, reports)


@register("fitness_source", "human_score")
def fitness_human_score(ctx: Any, state: Any, results: List[TrainResult]) -> List[CandidateReport]:
    """Text2Reward-human: the human *is* the evaluation signal (§4).

    The only method whose entire fitness comes from a person. `ctx.human` is the
    oracle built by the `human_oracle` phase -- scripted offline, interactive
    when a run asks for it -- and it owns its own `budget.record_human`
    accounting, so this component does not double-count.

    `evaluate.human.queries_per_iteration` caps how many candidates a person is
    asked about (GT deliberately caps at 1). A candidate past the cap was never
    shown to anyone, so it gets `None` -- **not** the failure sentinel: it did
    not lose a comparison, it was never in one (§5's none rule)."""
    cap = ctx.cfg.get("evaluate.human.queries_per_iteration", 0) or 0
    asked = 0
    reports = []
    for res in results:
        rep = _blank(ctx, res, "human_score")
        if rep.fitness is None:
            if cap and asked >= cap:
                rep.meta["fitness_note"] = "beyond evaluate.human.queries_per_iteration"
                reports.append(rep)
                continue
            asked += 1
            v = _human_score(ctx, rep)
            if v is None:
                rep.meta["fitness_note"] = "human returned no score"
            else:
                lo, hi = _scale_bounds(ctx)
                _collapse(ctx, rep, [(v - lo) / (hi - lo) if hi > lo else v])
        reports.append(rep)
    return _finish(ctx, reports)


@register("fitness_source", "none")
def fitness_none(ctx: Any, state: Any, results: List[TrainResult]) -> List[CandidateReport]:
    """No ranking scalar is produced here. Two readings, both supported (§4).

    1. *Filled elsewhere in this block.* `evaluate.preferences.enabled` or
       `evaluate.human.mode` runs after this component and may write a scalar
       onto the same reports. Leaving `fitness = None` is what lets it.
    2. *From nowhere at all.* **CARD computes no ranking scalar whatsoever** --
       K=1, so no contest exists (`select.rule: none`); its curves feed only the
       prompt, and the TPE verdict is an admission gate in §2, not a fitness.

    Deliberate exception to the failure-sentinel rule: under `none` an invalid
    candidate keeps `None` too. There is no ordering for it to lose, and
    stamping -10000 would manufacture the ranking scalar this source exists to
    deny. Its failure stays recorded where failures always are -- `candidate.failure`
    and `failure_kind`, which the run artifact keeps distinguishable."""
    return [CandidateReport(cand_id=r.cand_id, candidate=r.candidate, result=r,
                            fitness=None, fitness_source="none")
            for r in results]


# --------------------------------------------------------------------------
# Deferred fitness: sources whose input lands after `bird.evaluate` calls them.
# --------------------------------------------------------------------------


def _resolve_bt(ctx: Any, rep: CandidateReport) -> Optional[float]:
    """Read the Bradley-Terry strength `bird/components/preferences.py` writes."""
    v = _num(rep.meta.get("bt_strength"))
    if v is None:
        v = _num(getattr(rep.candidate, "meta", {}).get("bt_strength"))
    if v is None:
        return None
    rep.fitness = v
    rep.meta.pop("fitness_pending", None)
    return v


#: fitness_source name -> resolver run after the rest of stage 4 has filled in
#: its evidence. Keyed by a registry value, never by a method name.
_DEFERRED: Dict[str, Callable[[Any, CandidateReport], Optional[float]]] = {
    "preference_bt": _resolve_bt,
}


def _resolve_deferred(ctx: Any, rep: CandidateReport) -> None:
    resolver = _DEFERRED.get(rep.fitness_source)
    if resolver is None or rep.fitness is not None:
        return
    if resolver(ctx, rep) is None and rep.candidate.valid:
        log.debug("%s: fitness source %s found no %s; fitness stays None",
                  rep.cand_id, rep.fitness_source, rep.meta.get("fitness_pending"))


# ==========================================================================
# LLM / human plumbing
#
# `ctx.evaluator` and `ctx.human` are built by other component families
# (`bird/llm/*`, the `human_oracle` phase). All this module needs from either is
# text, so it duck-types across the plausible call shapes and treats "no usable
# answer" as a degraded-but-deterministic path rather than an exception: an
# offline end-to-end test must exercise the same config branch whether or not
# the mock happens to emit a parseable score.
# ==========================================================================

_CALL_NAMES = ("complete", "chat", "ask", "query", "respond", "generate", "__call__")


def _client_text(client: Any, prompt: str,
                 images: Optional[Sequence[Any]] = None) -> str:
    """Ask `client` one question and return its text.

    `images` is passed as the `images=` keyword `bird/llm/base.py` declares, and
    when frames are supplied EVERY attempt carries them: the call is never
    retried without its images. A ladder of [with images, without images] per
    call shape would let any exception or empty answer on the sighted call fall
    through to the blind one -- which is exactly what `AnthropicClient`
    produces on a `refusal` or `max_tokens` stop, on a non-retryable 4xx such
    as a too-large image payload, and on an exhausted retry ladder. The judge
    would then re-answer the identical prompt from the text tables alone, the
    caller (which has already written `n_images=len(frames)` to its query
    record, before the call) would file the verdict as image-graded,
    `budget.vlm_calls` would count two calls, and RDA's subtask scores, the
    curriculum gate and `alignment_rate` would be partly text-graded with every
    counter reading normal: images attached, refused, re-asked blind.

    So a sighted call that raises or answers nothing returns `""`. Every caller
    has a documented degraded path for empty text (RDA falls back to the env's
    success flag and records `fallback`; the curriculum gate scores `None`;
    `alignment_rate` counts `n_unparsed`), and `judgments.note_response` files
    the empty answer as `no_answer` against a query that says how many images
    it carried -- "asked with N frames, no verdict", which is the truth. The
    caller decides whether to attach frames at all through `client_sees_images`;
    a client that takes no `images=` keyword, or a text-modality one, is handed
    none and asked without them, unchanged.

    The call SHAPE is still duck-typed: each name in `_CALL_NAMES` is tried
    with a message list and with the bare prompt, the images riding along on
    every attempt. But a sighted call that HAPPENED -- returned anything, or
    raised anything but the `TypeError` of a wrong arity -- is the one and only
    ask: a second sighted attempt on the other shape would be a retry of a
    refused request, billed as a second VLM call (`preferences._client_call`
    draws the same line). The text path still hunts over shapes on an empty
    answer or an error.
    """
    if client is None:
        return ""
    messages = [{"role": "user", "content": prompt}]
    kw: Dict[str, Any] = {"images": list(images)} if images else {}
    for name in _CALL_NAMES:
        fn = getattr(client, name, None)
        if not callable(fn):
            continue
        for args in ((messages,), (prompt,)):
            try:
                out = fn(*args, **kw)
            except TypeError:
                continue  # wrong arity/shape: try the next convention
            except Exception:  # noqa: BLE001 -- any client failure is a degrade
                if kw:
                    return ""  # the sighted call happened and failed: no re-ask
                continue
            text = _as_text(out)
            if text or kw:
                return text  # sighted: an empty answer IS the answer
    return ""


def client_sees_images(client: Any) -> bool:
    """Would `client` actually look at attached frames?

    Two conditions, and both are silent failures on their own. The client must
    take an `images=` keyword (`bird/llm/base.py`'s contract), and it must
    declare `modality: vlm` -- `AnthropicClient._complete` DROPS images handed to
    a text-modality client, with a warning that scrolls past in a 20-hour log.
    Callers use this to decide whether to spend renders and, when it is false,
    to say so in the artifact.
    """
    if client is None:
        return False
    if str(getattr(client, "modality", "text")) != "vlm":
        return False
    for name in _CALL_NAMES:
        fn = getattr(client, name, None)
        if not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return True  # unreadable signature: let the call itself decide
        if "images" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return True
    return False


def _as_text(out: Any) -> str:
    """Whatever a client returned, as text.

    A LIST IS THE NORMAL CASE, not an edge case: `bird/llm/base.py` states the
    contract as `client(messages, n=1, ...) -> list[str]`, and every client in
    this repo honours it. Omitting the list branch would make `_client_text`
    return "" for every conformant client, and both callers degrade SILENTLY on
    empty text -- `_vlm_pair_score` falls back to the environment's success
    flag and `_human_score` returns None. So a config asking for
    `evaluate.fitness.source: vlm_score` would get a blind success-flag
    comparator that never once called the VLM, and nothing in the artifact
    would say so: a VLM-graded method would quietly become a blind
    comparator.

    `n=1` is what these callers ask for, so a longer list means the client
    ignored `n`; joining is wrong (it would concatenate two independent
    samples), so the first sample wins.
    """
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)):
        for item in out:
            text = _as_text(item)
            if text:
                return text
        return ""
    if isinstance(out, dict):
        for k in ("text", "content", "completion", "message", "response"):
            v = out.get(k)
            if isinstance(v, str):
                return v
    for attr in ("text", "content", "completion"):
        v = getattr(out, attr, None)
        if isinstance(v, str):
            return v
    return ""


def judge_concurrency(ctx: Any, n_jobs: int, client: Any = None) -> int:
    """How many independent judge round-trips may be in flight at once.

    `llm.max_concurrent_requests`, read off the judging client, honoured only
    for providers that declare `concurrent_samples` -- the anthropic client
    does; the mock never does, because its per-sample RNG draws consume the
    run's stream IN ORDER and concurrency would reorder them, so every
    tester-profile determinism claim is untouched. Returns 1 -- the serial loop
    -- when the client lacks those attributes.

    `client` names WHOSE transport budget gates the fan-out and defaults to
    `ctx.evaluator` -- the judge at every call site (preferences, curriculum,
    alignment_rate) except RDA's in-loop scorer, which judges with
    `ctx.generator` (the paper's Agent VLM) and passes it explicitly. The
    default must stay the evaluator: silently reading the other role's limits
    here would misroute the judge's transport budget.

    Only the ROUND-TRIPS overlap. Everything order-sensitive around them --
    rendering frames on the shared env adapter, folding scores, building
    `D_pref` -- stays sequential at the call sites, which each say so.
    """
    if n_jobs <= 1:
        return 1
    if client is None:
        client = getattr(ctx, "evaluator", None)
    if not getattr(client, "concurrent_samples", False):
        return 1
    return max(1, min(int(getattr(client, "max_concurrent_requests", 1) or 1), n_jobs))


def _scale_bounds(ctx: Any) -> Tuple[float, float]:
    """`evaluate.feedback.score_scale` as numeric bounds."""
    scale = ctx.cfg.get("evaluate.feedback.score_scale", "[0,1]")
    if scale == "likert":
        return 1.0, 5.0
    # "[0,1]", "binary", "three_point" and "staged" share the range; binary
    # and three_point quantise (see `_snap`), staged stays continuous.
    return 0.0, 1.0


#: `three_point`'s codomain -- RDA's App. 7.3 rubric (1.0 success / 0.5 partial
#: / 0.0 failure). §4.2's `s in [0,1]` is the CODOMAIN this discretises, which
#: is why the bounds above stay (0, 1).
_THREE_POINT = (0.0, 0.5, 1.0)


def _snap(score: float, scale: str) -> float:
    """Quantise a parsed score onto its declared scale.

    `three_point` snaps to the nearest of {0.0, 0.5, 1.0} -- a judge that
    answers 0.8 despite the rubric must not smuggle a continuous scale back in
    through the parser, or the ablation `score_scale` exists for is unmeasurable.
    Ties round DOWN (0.75 -> 0.5, the first of two equidistant values), so the
    quantiser cannot inflate a borderline judgment. Every other scale is
    returned untouched: `[0,1]` and `likert` are what the judge said, and
    `binary`'s integers already parse as themselves.
    """
    if scale == "three_point":
        return min(_THREE_POINT, key=lambda v: abs(v - score))
    return score


#: A score token: a number, optionally written as a fraction (`4/5`, `8 / 10`).
_SCORE_TOKEN = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")
#: The same token anchored to a stated score: `Score: 4/5`, `score 0.8`,
#: `rating = 2`, `rated 3`, and a JSON key (`"score": 4`, the shape the mock
#: judge and `chat_json` answers take). Case-insensitive.
_SCORE_ANCHOR = re.compile(
    r"\b(?:score|scored|scores|rating|rated|rate|grade)\b[\"']?\s*(?:is|of|was|[:=\-])?\s*"
    r"(-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?)", re.IGNORECASE)


def _score_token(tok: str, lo: float, hi: float) -> Optional[float]:
    """One token as a score on [lo, hi], or None.

    A fraction is a score only when its denominator IS the scale: `den == hi`
    reads `4/5` on likert as 4 (and `1/1` on the unit scale as 1), and on the
    unit scale [0, 1] any proper fraction is the proportion it states (`8/10`
    -> 0.8, `4/5` -> 0.8). Anything else -- `7/10` on a 1-5 scale, `12/50` --
    is a miss: `lo + num/den * (hi - lo)` would turn a likert `4/5` into 4.2
    and an incidental `step 12/50` into 0.24. A fraction
    that misses is consumed WHOLE; its digits are never re-read as plain
    numbers, or `0/5` on likert would parse as 5."""
    eps = 1e-9
    if "/" in tok:
        num_s, den_s = tok.split("/", 1)
        num, den = float(num_s), float(den_s)
        if den <= 0:
            return None
        if abs(den - hi) < eps and lo - eps <= num <= hi + eps:
            return num
        if abs(lo) < eps and abs((hi - lo) - 1.0) < eps and 0.0 <= num <= den:
            return num / den
        return None
    v = float(tok)
    return v if lo - eps <= v <= hi + eps else None


def _parse_score(text: str, lo: float, hi: float) -> Optional[float]:
    """The score stated in `text`, on [lo, hi].

    A number anchored to a score word (`Score: 4/5`, `score 0.8`, `rating: 2`)
    wins over anything earlier in the text; failing that, the first token that
    is a score on this scale (`0.8`, `8/10`, `4`). Anything else is a miss, and
    a miss is reported as such rather than coerced -- an invented score would
    silently become a fitness. An UNANCHORED fraction regex run first would
    parse `At step 12/50 the arm moved; score 0.8` as 0.24 and a likert
    `Score: 4/5` as 4.2."""
    if not text:
        return None
    anchored = _SCORE_ANCHOR.search(text)
    if anchored:
        v = _score_token(anchored.group(1), lo, hi)
        if v is not None:
            return v
    for m in _SCORE_TOKEN.finditer(text):
        v = _score_token(m.group(0), lo, hi)
        if v is not None:
            return v
    return None


def _parse_reason(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"(?:reason|rationale|because|why)\s*[:\-]\s*(.+)", text,
                  re.IGNORECASE | re.DOTALL)
    body = m.group(1) if m else text
    return " ".join(body.split())[:400]


def _human_score(ctx: Any, rep: CandidateReport) -> Optional[float]:
    """Ask `ctx.human` to score one candidate."""
    human = getattr(ctx, "human", None)
    if human is None:
        return None
    lo, hi = _scale_bounds(ctx)
    prompt = (f"Rate the behaviour produced by candidate {rep.cand_id} on the task "
              f"{ctx.cfg.get('problem.task_description', '')!r}, on a {lo}-{hi} scale.\n"
              f"{_behaviour_digest(ctx, rep.result)}")
    for name in ("score", "rate", "evaluate"):
        fn = getattr(human, name, None)
        if callable(fn):
            for args in ((rep,), (prompt,), ()):
                try:
                    v = _num(fn(*args))
                except Exception:  # noqa: BLE001
                    continue
                if v is not None:
                    return v
    return _parse_score(_client_text(human, prompt), lo, hi)


# ==========================================================================
# Evidence rendering shared by the VLM path and the feedback builder
# ==========================================================================


def _subtasks(ctx: Any, state: Any) -> List[str]:
    """RDA's subtask list `T^i`, or a single pseudo-subtask standing for the
    whole task when `generate.decomposition.enabled` is false -- so per-subtask
    machinery degrades to per-task rather than to nothing."""
    subs = list(getattr(state, "subtasks", None) or [])
    return subs or [ctx.cfg.get("problem.task_description", "the task")]


def _rollouts(ctx: Any, result: TrainResult) -> List[Trajectory]:
    """`evaluate.rollouts_per_candidate` rollouts (RDA's K=3, Table 1)."""
    k = int(ctx.cfg.get("evaluate.rollouts_per_candidate", 3) or 3)
    return list(result.trajectories)[:max(1, k)]


def _show_task_metric(ctx: Any) -> bool:
    """`evaluate.feedback.include_task_metric`, read in ONE place.

    GT's §3 assumption ("the fitness function F is not accessible while
    searching") is this key set false. Every renderer that could name the
    flag gates on this -- `_section_numeric`, the analyzer's no-caption branch,
    the rendered example rollouts and the summariser's digest -- or a
    fitness-blind method would see the environment's ground-truth flag
    (`success=<flag>`, "the agent solves the task") through a side door.

    It is NOT the whole gate on those renderers: this key
    says what the METHOD may see, `_show_env_success` adds whether the
    BENCHMARK ships the flag at all. Read that one before adding a caller."""
    return bool(ctx.cfg.get("evaluate.feedback.include_task_metric", True))


def _show_env_success(ctx: Any) -> bool:
    """`include_task_metric`, AND the benchmark actually ships the success this
    rendering would name (native-signal rule N, `bird/native_signal.py`).

    `_show_task_metric` asks what the METHOD is allowed to see;
    `_env_defines_success` asks whether the adapter computes a flag at all.
    Neither asks the question that matters to a reflecting LLM: is the flag the
    BENCHMARK'S, or one we wrote? On a `continuous_only` task `EnvAdapter.
    success()` is a threshold BIRD chose on a per-step check BIRD wrote (every
    assistax spec says so in `no_discrete_success_because`), so `success=False`
    and "the agent still has not solved the task" assert a criterion the
    benchmark does not define -- the same falsehood `_env_defines_success` was
    added to stop on the gym family, one level up: there the env computed no
    flag, here it computes OUR flag and the prose calls it the task's.

    Observed with CARD (fitness.source: none, include_task_metric: true) on
    `assistax_feeding`: without this gate every iteration's reflection carries
    `task_success: [0.00, ..., 0.93, 0.67]` plus a `success=` flag per
    rollout.

    The gate is `has_native_success`, NOT `native_curve_keys is not None`: a
    task can ship a reference reward and no success check, and assistax is
    exactly that (`reward.human.kind: published_dense`,
    `discrete_success.kind: continuous_only`), so a "has any native signal"
    gate would leave the reported line in place. No spec resolves -> refuse,
    the same default `resolve_channel` takes when nativeness is unverifiable.
    """
    if not (_show_task_metric(ctx) and _env_defines_success(ctx)):
        return False
    from ..native_signal import has_native_success, kinds_for_env
    ds, _rw = kinds_for_env(env_id=ctx.cfg.get("problem.env_id"),
                            task_id=ctx.cfg.get("problem.task_id"))
    return has_native_success(ds)


def _declared_state_fields(ctx: Any) -> Tuple[str, ...]:
    """The adapter's declared observation field names, in observation order --
    the legend `_traj_state_table` prints for the judge, reused to name the
    columns `render_trajectory` shows the generator."""
    env = getattr(ctx, "env", None)
    return tuple(str(name) for name, _desc in (getattr(env, "_state_fields", ()) or ()))


def _traj_digest(t: Trajectory, show_success: bool = True) -> str:
    flag = f"success={t.success} " if show_success else ""
    return (f"{flag}length={t.length} return={_fmt(_num(t.ret))} "
            f"mean_per_step={_fmt(t.mean_per_step_return)}")


def _behaviour_digest(ctx: Any, result: TrainResult) -> str:
    trajs = _rollouts(ctx, result)
    if not trajs:
        return "(no rollouts were stored for this candidate)"
    show = _show_env_success(ctx)
    lines = [f"  rollout {i}: {_traj_digest(t, show)}" for i, t in enumerate(trajs)]
    return "\n".join(lines)


def judge_digest(t: Trajectory) -> str:
    """The rollout summary a judge may be told IN WORDS: its length.

    Deliberately not `_traj_digest`, and the two must never be merged. That one
    prints `success=<bool> ... return=<float>`, and handing either to a judge
    breaks the method in a different way:

      * `success` is the ENVIRONMENT'S GROUND TRUTH -- the very success check
        `verify.forbidden_symbols` goes to lengths to keep out of the reward
        code on Meta-World (the benchmark's `evaluate_state`). Letting it in
        through the judge's prompt hands the
        metric over through the other door, and the resulting "VLM score" is
        mostly a restatement of the flag.
      * `return` is the CANDIDATE'S OWN TOTAL REWARD. A judge shown it scores
        monotonically in it, so a candidate wins by inflating its reward scale
        -- the reward grading itself. `_vlm_trajectory_analysis`'s degraded
        branch refuses `traj.ret` for exactly this reason.

    The withholding argument covers EXACTLY those two, and must not be read
    wider than it is: per-step reward COMPONENT values and the reward function
    code are not withheld -- they are evidence RDA's own protocol hands the
    judge on purpose (App. 7.3: 'Support your evaluation with both image and
    reward evidence', with a diagnosis taxonomy -- gating, overshadowing,
    saturation -- that is impossible without them; §5.2 credits the
    mis-specification detection to exactly that channel, and every §8 worked
    example cites component values at named steps). A judge shown neither
    would select on strictly less evidence than the published method's.
    `_traj_component_table` renders them
    -- components only, never a per-step total, an episode return or the
    success flag, and never `_REFERENCE_KEYS`.

    Length stays because it is a property of the episode and not of the reward
    or of the metric, and because "the arm did nothing for 500 steps" is not
    legible from 20 sampled frames.
    """
    return f"length={int(t.length)} steps"


def _vlm_frames(ctx: Any, traj: Trajectory,
                prov: Optional[Dict[str, Any]] = None,
                client: Any = None) -> Tuple[List[bytes], str]:
    """The frames one `(rollout, *)` judgment is entitled to, and why not more.

    `client` is the judge the frames are destined for, defaulting to
    `ctx.evaluator`; `fitness_vlm_score` passes `ctx.generator`, the Agent VLM
    that does RDA's in-loop scoring.

    Gated on `evaluate.artifacts` containing `videos` -- the config's own
    statement that rollout footage is evidence for this method, and already
    coherence-checked against `output.video.enabled` (`config._check_coherence`),
    so this cannot render frames a config asked not to have -- and NOT on
    `output.video.record`, which chooses what reaches disk for a human. Those
    have to stay separate: a config may narrow `record` from `all` to
    `best_and_worst` on rendering cost, and no fitness notices only because
    the judge renders its own frames.

    Returns `(frames, note)`; `note` is non-empty exactly when the judgment is
    going to be made blind, and says which of the three reasons it was.

    `prov` is `sample_frames`' out-parameter, threaded through so a judgment
    record can say WHICH pixels this judgment was made from -- the episode step
    indices, the frame size, a per-frame luma and content hash -- rather than
    only how many there were. It is filled on the blind paths too, carrying the
    reason: `output.judge_trace.record`'s whole point is that a judgment made
    with no frames must be legible as such afterwards, and `note` says so only
    in memory (`bird/judgments.py`).
    """
    def blind(reason: str) -> Tuple[List[bytes], str]:
        if prov is not None:
            prov.update(judgments.frame_set((), None, blind_reason=reason))
        return [], reason

    if "videos" not in (ctx.cfg.get("evaluate.artifacts") or []):
        return blind("evaluate.artifacts does not list videos")
    if client is None:
        client = getattr(ctx, "evaluator", None)
    if not client_sees_images(client):
        return blind("the judging client takes no images or is not modality: vlm")
    n_images = int(ctx.cfg.get("evaluate.vlm.images_per_query", 20) or 0)
    frames = sample_frames(ctx, traj, n_images, prov=prov)
    if not frames:
        # NOT `blind(...)`: `sample_frames` has already filled `prov` with the
        # reason, and its reason is the specific one (which of its three sources
        # failed and how). Overwriting it with this summary line would throw
        # away the only part a reader could act on.
        return [], "no frames could be sampled from the rollout"
    return frames, ""


def _note_answer(ctx: Any, trace: Optional[Dict[str, Any]], query_id: str,
                 **fields: Any) -> None:
    """File one judge answer against the query that asked for it.

    A wrapper so the two exits of `_vlm_trajectory_analysis` cannot disagree about
    which iteration the answer belongs to: `trace` carries it (`-1` for a
    context with no RunState, which `_stem` files under `post.jsonl`), and
    reading it in one place is what keeps a response in the same file as its
    query. Callers with no trace never recorded a query either, so there is
    nothing to answer and this returns.

    The identity labels come off `trace` rather than being re-passed, so a
    response is filterable on the same fields as its query -- `cand_id`,
    `rollout`, `repeat`, as `alignment_rate`'s answers already are. Without
    them "what did the judge say about c0007" is a two-step join through
    `query_id` for a question the record can simply answer. `frame_set` is NOT
    copied: the evidence hangs off the query, and duplicating the pointer would
    invite the two to disagree.
    """
    if trace is None:
        return
    # Copied only where the trace actually carries them, never defaulted: a
    # `rollout: -1` invented here would read as `_stem`'s "outside the loop"
    # rather than as "this caller did not say", and `judgments.query_id`'s
    # docstring draws exactly that line.
    labels = {k: trace[k] for k in ("cand_id", "rollout", "repeat") if k in trace}
    judgments.note_response(
        ctx, iteration=int(trace.get("iteration", -1)), query_id=query_id,
        **labels, **fields)


#: Per-step rows in the judge's component table when the frame provenance
#: carries no step mapping (a blind call, or frames read off disk with an
#: unknown stride). Matches `_TRAJ_MAX_STEPS`' prompt-budget argument.
_JUDGE_TABLE_ROWS = 24


def _traj_component_table(traj: Trajectory, steps: Optional[Sequence[int]]) -> str:
    """App. 7.3's 'Trajectory data per step': step index + reward component values.

    `steps` is the frame provenance's own episode-step list (`sample_frames`
    fills it), so each table row pairs with the attached frame taken at that
    instant -- RECORDED alignment, never recomputed, `save_trajectory_trace`'s
    argument applied to the judge's prompt. When no mapping exists (a blind
    judgment, or frames whose on-disk stride is unknown) the rows re-stride
    evenly across the episode instead.

    What the rows carry is the candidate's own per-step COMPONENT values --
    `self.reward_info` in the paper's terms, the evidence App. 7.3's diagnosis
    taxonomy needs -- and nothing else: no per-step total, no episode return,
    no success flag (see `judge_digest` for the line), and never
    `_REFERENCE_KEYS` (the hidden reference reward stays hidden, the same
    guard `render_trajectory` applies).
    """
    n = int(traj.length or 0)
    every = [(k, list(v)) for k, v in (traj.component_values or {}).items()
             if k not in _REFERENCE_KEYS]
    # `_JUDGE_TABLE_MAX_VARS`, not `_TRAJ_MAX_VARS`: the analyst gets every
    # component up to a prompt-budget ceiling, and is TOLD when it binds -- a
    # silent `[:8]` would drop the success bonus and the penalties from a
    # 9-component reward and leave the header claiming a full table.
    comps = every[:_JUDGE_TABLE_MAX_VARS]
    if n <= 0 or not comps:
        return "(no per-step reward component values were logged for this rollout)"
    rows = [int(s) for s in (steps or []) if 0 <= int(s) < n]
    # `paired` keys on the rows ACTUALLY USED, not on the `steps` argument: a
    # non-empty mapping whose every index filtered out of range falls back to
    # the linspace rows below, and a header still claiming frame pairing there
    # would assert an alignment the table no longer has.
    paired = bool(rows)
    if not rows:
        rows = list(dict.fromkeys(
            int(round(x)) for x in np.linspace(0, n - 1, min(n, _JUDGE_TABLE_ROWS))))
    lines = ["Per-step reward component values (each row pairs with the attached "
             "frame sampled at that step):" if paired else
             "Per-step reward component values:"]
    if len(every) > len(comps):
        lines.append(f"  (showing the first {len(comps)} of {len(every)} reward "
                     f"components; the remaining {len(every) - len(comps)} are "
                     f"not in this table)")
    header_n = len(lines)
    for s in rows:
        # `None` is a step the reward claimed nothing for (per-step packing):
        # no cell, rather than a cell reading "n/a" beside the frame.
        bits = [f"{name}={_fmt(_num(seq[s]))}" for name, seq in comps
                if s < len(seq) and seq[s] is not None]
        if bits:
            lines.append(f"  step={s:<4} " + "  ".join(bits))
    return "\n".join(lines) if len(lines) > header_n else \
        "(no per-step reward component values were logged for this rollout)"


def _score_rubric(scale: str, lo: float, hi: float) -> str:
    """The scoring instruction, per `evaluate.feedback.score_scale`.

    `three_point` renders App. 7.3's rubric verbatim; the other scales keep a
    bounds sentence so the enum stays honest (a `likert` config really is asked
    for 1-5, not silently for the paper's rubric).

    `staged` anchors the continuous scale to named stages of progress
    (unpublished). The rationale: an unanchored judge tends to collapse onto
    a few habitual values, and equal scores are dropped as ties by everything
    that consumes an ordering. Continuous on purpose: `_snap` leaves it
    untouched, and the prompt asks for interpolation, because rank resolution
    is the whole point.
    """
    if scale == "three_point":
        return ("- Use a 3-point success scale:\n"
                "  - 1.0 = success\n"
                "  - 0.5 = partial success\n"
                "  - 0.0 = failure")
    if scale == "staged":
        return (f"- Score each subtask on a {lo}-{hi} scale, anchored to "
                "stages of progress and interpolating freely between anchors:\n"
                f"  - {lo} = no progress toward it at any point\n"
                f"  - {lo + 0.25 * (hi - lo)} = approaches or orients toward "
                "the relevant object or direction, but no useful interaction "
                "or movement happens\n"
                f"  - {lo + 0.5 * (hi - lo)} = the intended interaction "
                "clearly begins (contact, grasp, movement in the right "
                "direction) but falls well short of the goal\n"
                f"  - {lo + 0.75 * (hi - lo)} = most of the way there: clear "
                "sustained progress, goal nearly reached\n"
                f"  - {hi} = the goal state is achieved at some point\n"
                "  Use the full scale: rank-order matters more than the "
                "absolute value, so prefer 0.15 vs 0.3 over calling both 0.2.")
    return f"- Score each subtask on a {lo}-{hi} scale."


def _demo_trace_table(ctx: Any, cand: Any) -> str:
    """What the candidate's reward PAID a graded set of known policies, as numbers
    -- gated on `demo_reward_traces` in `evaluate.artifacts`, empty otherwise.

    Rendered from `candidate.meta["demo"]`, written by the `demo_margin` screen
    (`bird/demos.py`). Per policy: the mean per-step return under THIS reward
    and a strided per-step series of episode 0, components included. Nothing
    about how the policy did on the task -- no metric, no success flag, no
    reference reward -- and never the policy itself: the reader learns that
    "the solution policy" earned 0.31/step and "random" earned 0.29/step under
    this reward, which is the whole point (a reward that cannot tell them apart
    cannot teach the difference), and nothing else. The judge sees the same
    table the reflecting LLM sees, so the two reason from one record.
    """
    if "demo_reward_traces" not in set(ctx.cfg.get("evaluate.artifacts") or []):
        return ""
    demo = (getattr(cand, "meta", None) or {}).get("demo") if cand is not None else None
    if not demo or not demo.get("policies"):
        return ""
    lines = ["## Reward under known policies",
             "The candidate reward re-scored rollouts of policies of KNOWN relative "
             "quality (best first). Numbers are what THIS reward paid them -- not how "
             "they did on the task."]
    # `_fmt(_num(x))`, never `float(x)`: a component series is packed per
    # step and carries None on every step the reward did not claim
    # that key (a conditional bonus, a step the `demo_margin` screen's
    # `on_error` swallowed into `cv = {}`). `strided` passes the gap through so
    # the list stays aligned with `reward_series`; it is rendered `n/a` at its
    # own index, never coerced to 0.0 -- a payment of nothing and no claim at
    # all are different findings, and this table exists to show which.
    for pol in demo["policies"]:
        series = ", ".join(_fmt(_num(v)) for v in (pol.get("reward_series") or []))
        lines.append(f"  {pol.get('name')}: mean per-step reward "
                     f"{_fmt(float(pol.get('mean_per_step_return', 0.0)))} "
                     f"over {int(pol.get('episodes', 0))} episode(s); "
                     f"per-step reward, episode 0 (strided): [{series}]")
        comps = pol.get("component_series") or {}
        for k, v in list(comps.items())[:_TRAJ_MAX_VARS]:
            if k in _REFERENCE_KEYS:
                continue
            lines.append(f"      {k}: [{', '.join(_fmt(_num(x)) for x in v)}]")
    if "margin" in demo:
        lines.append(f"  best-vs-worst margin {demo['margin']:+.3f} (scale-free, [-1, 1]); "
                     f"quality-order monotonicity {demo.get('monotonicity', 0.0):+.2f}")
    return "\n".join(lines) + "\n"


def _traj_state_table(ctx: Any, traj: Trajectory, steps: Optional[Sequence[int]]) -> str:
    """Per-step state/action rows for the judge, gated on `state_trajectories`
    in `evaluate.artifacts` -- empty string otherwise.

    This is GT's Table 2 lever (74.14% vs 62.96% VLM-human agreement with
    trajectory text beside the clip, §4) applied to the ABSOLUTE judge: the
    pairwise comparator honours the same key (`preferences._render`), and a
    vlm_score judge ignoring the same declared artifact would leave the key
    declared but unread. The need is measured: on `mt10_push-v3` the judge's
    mean pick sat at 0.26 against a 0.66 oracle over its own candidates -- a
    centimetre-scale
    puck-to-goal gap that 20 stride-sampled 320px frames cannot resolve and one
    `obs` row states outright.

    Same alignment contract as `_traj_component_table`: rows key on the frame
    provenance's own episode steps, falling back to an even re-stride. What a
    row carries is the environment's OBSERVATION and the action -- never the
    success flag, never a task metric, never the hidden reference reward. A
    legend names the observation fields when the adapter declares them
    (`_state_fields`), because `obs=[0.02, -0.31, ...]` is only evidence if the
    judge knows which entry is the puck."""
    if "state_trajectories" not in set(ctx.cfg.get("evaluate.artifacts") or []):
        return ""
    n = int(traj.length or 0)
    states = traj.states
    try:
        n_states = len(states)
    except TypeError:
        n_states = 0
    if n <= 0 or n_states == 0:
        return ""
    rows = [int(s) for s in (steps or []) if 0 <= int(s) < min(n, n_states)]
    paired = bool(rows)
    if not rows:
        rows = list(dict.fromkeys(
            int(round(x)) for x in np.linspace(0, min(n, n_states) - 1,
                                               min(n, _JUDGE_TABLE_ROWS))))

    def _vec(x: Any) -> str:
        try:
            arr = np.asarray(x, dtype=float).ravel()
        except (TypeError, ValueError):
            return str(x)
        return "[" + ", ".join(_fmt(float(v)) for v in arr) + "]"

    lines = []
    fields = [name for name, _desc in (getattr(ctx.env, "_state_fields", ()) or ())]
    if fields:
        lines.append("Observation fields, in order: " + ", ".join(fields))
    lines.append("Per-step observation and action (each row pairs with the "
                 "attached frame sampled at that step):" if paired else
                 "Per-step observation and action:")
    actions = traj.actions
    try:
        n_actions = len(actions)
    except TypeError:
        n_actions = 0
    for s in rows:
        row = f"  step={s:<4} obs={_vec(states[s])}"
        if s < n_actions:
            row += f"  act={_vec(actions[s])}"
        lines.append(row)
    return "\n" + "\n".join(lines)


def _vlm_trajectory_analysis(ctx: Any, traj: Trajectory, subtasks: Sequence[str],
                             reward_code: str, want_reason: bool,
                             frames: Optional[Sequence[bytes]] = None,
                             prov: Optional[Dict[str, Any]] = None,
                             trace: Optional[Dict[str, Any]] = None,
                             candidate: Any = None,
                             ) -> List[Tuple[float, str, str]]:
    """ONE structured judgment per trajectory covering ALL subtasks, in order
    (RDA, App. 7.3) -- from `ctx.generator`, the paper's Agent VLM.

    One call, not one per `(trajectory, subtask)` pair: App. 7.3's prompt hands
    the judge the whole ordered subtask list and asks for one JSON report with a
    `subtasks` array, evaluated 'independently and in order' under the explicit
    failure cascade ('If a subtask fails, all later subtasks are assumed failed
    (0.0)...'). Per-pair calls would give the judge no ordering and no
    cascade, so late subtasks of a failed rollout would be scored on their own
    merits and the mean fitness would rank candidates differently from the
    paper's (§4.2's per-pair `(s, rho)` notation is the score SET, not the call
    protocol). The cascade is the MODEL's to apply, not
    post-applied here: its own 'unless there is a clear visual or reward-based
    evidence of later success' escape hatch makes a mechanical zeroing
    unfaithful.

    The prompt carries the paper's evidence: the candidate's reward function
    code and a per-step component-value table aligned to the attached frames'
    own episode steps (see `judge_digest` for what stays withheld and why),
    and the judge is `ctx.generator` because GPT-5,
    the Agent VLM, does all of RDA's in-loop work; GPT-4.1 exists only for the
    post-hoc alignment metric, 'to mitigate potential bias between the
    generator and evaluator' (§5.1, §10.1).

    RECORDED DEVIATION: the paper's two-step
    Describe-Behavior / Classify-Success block and the `behavior`/`analysis`
    output fields are requested only under `want_reason`
    (`require_rationale` or `granularity: per_subtask` -- `configs/methods/rda.yaml`
    sets both, so the pinning method point runs the full App. 7.3 protocol).
    App. 7.3 itself mandates the behavior step unconditionally; a `vlm_score`
    ablation with rationales off therefore gets a reduced prompt
    (number/name/score only) that is BIRD's token-economy choice, not the
    paper's protocol.

    Returns one `(score, behavior, analysis)` per subtask, in subtask order. A
    subtask the reply misses -- or a reply that never parses -- ABSTAINS with a
    `None` score (only the `mock` provider substitutes a synthetic stand-in,
    recorded as such; see below), and never falls back to `traj.ret`, which is
    the candidate's own reward and would let the reward grade itself.

    `frames` are the PNG bytes of the rollout, sampled ONCE per trajectory by
    the caller and reused across repeats; they are passed as `images=`. `trace`
    is this query's identity, and when present the prompt is written to the
    judge trace HERE, where it is built (role `vlm_subtask_score`, as trace
    consumers expect); each subtask's verdict is filed as
    its own response against the one query.
    """
    lo, hi = _scale_bounds(ctx)
    scale = ctx.cfg.get("evaluate.feedback.score_scale", "[0,1]")
    subtasks = list(subtasks)
    frames = list(frames or ())
    listing = "\n".join(f"{i}. {s}" for i, s in enumerate(subtasks, start=1))
    seen = (f"\n{len(frames)} frames sampled evenly from the rollout are attached."
            if frames else "")
    steps = list((prov or {}).get("steps") or [])
    out_fields = ('{"subtasks": [{"number": <int>, "name": "<subtask>", '
                  '"behavior": "<what the agent does>", "score": <score>, '
                  '"analysis": "<reasoning citing reward values>"}, ...]}'
                  if want_reason else
                  '{"subtasks": [{"number": <int>, "name": "<subtask>", '
                  '"score": <score>}, ...]}')
    two_step = (
        "For every subtask, perform two steps: Describe Behavior, then Classify "
        "Success.\n"
        "(i) Describe Behavior: observe the key frames and describe what the agent "
        "is doing during this subtask, checking for unnatural or unintended "
        "behavior.\n"
        "(ii) Classify Success: assign a score with concise reasoning that "
        "references specific reward values at key steps and explains how they "
        "justify the chosen level. If performance was poor or unstable, identify "
        "contributing factors, such as:\n"
        "  - Reward gating or missing signal flow.\n"
        "  - One component overshadowing others.\n"
        "  - Reward magnitudes too small or saturated.\n"
        if want_reason else "")
    # `evaluate.vlm.judge_guidance` (ours, no paper pin): extra scoring guidance,
    # appended INSIDE the instructions block so the output contract stays last.
    # What the images ARE, when they are not one-frame-per-image. Empty under
    # `evaluate.vlm.frame_policy: even`, so that path's prompt carries no frames
    # section -- the byte-identity the default rests on. `note_query` records
    # `prompt`, so the manifest reaches
    # the judgment record by construction rather than by a second write that
    # could disagree with what was sent.
    sheet = ((prov or {}).get("composed") or {}).get("manifest") or ""
    # What each frame's PANELS are, under `output.video.n_views > 1`
    # (`multiview.describe`): the spec's own mode and note per camera, in
    # reading order. Empty for a single-view frame, so that prompt is unchanged.
    panels = multiview.describe((prov or {}).get("views"))
    frames_block = "\n".join(part for part in (panels, sheet) if part)
    sheet = f"## Frames\n{frames_block}\n\n" if frames_block else ""
    extra_guidance = ctx.cfg.get("evaluate.vlm.judge_guidance")
    extra_guidance = (str(extra_guidance).strip() + "\n"
                      if extra_guidance and str(extra_guidance).strip() else "")
    prompt = (
        "You are a robotics evaluation analyst reviewing a trajectory produced by "
        "an RL policy. Score the agent's performance for each subtask using both "
        "visual observations and reward values.\n\n"
        f"## Task Instruction\n{ctx.cfg.get('problem.task_description', '')}\n\n"
        f"## Subtask List\n{listing}\n\n"
        f"## Reward Function\n{reward_code or '(no reward code available)'}\n\n"
        f"## Trajectory\nRollout: {judge_digest(traj)}{seen}\n"
        f"{_traj_component_table(traj, steps)}"
        f"{_traj_state_table(ctx, traj, steps)}\n\n"
        f"{_demo_trace_table(ctx, candidate)}"
        f"{sheet}"
        "## Instructions\n"
        "Evaluate each subtask independently and in order.\n"
        f"{_score_rubric(scale, lo, hi)}\n"
        "- If a subtask fails, all later subtasks are assumed failed (0.0) unless "
        "there is a clear visual or reward-based evidence of later success.\n"
        "- Support your evaluation with both image and reward evidence for each "
        "subtask.\n"
        f"{two_step}"
        f"{extra_guidance}"
        "## Output\n"
        "Report every subtask, in order, in a single JSON object inside a "
        f"```json fence:\n{out_fields}")
    query_id = ""
    if trace is not None:
        query_id = judgments.note_query(ctx, role="vlm_subtask_score", prompt=prompt,
                                        subtasks=subtasks, n_images=len(frames),
                                        **trace)
    client = getattr(ctx, "generator", None)
    text = _client_text(client, prompt, images=frames)
    entries = _analysis_entries(text, subtasks)

    def _valid(entry: Dict[str, Any]) -> Optional[float]:
        v = _num((entry or {}).get("score"))
        return v if (v is not None and lo - 1e-9 <= v <= hi + 1e-9) else None

    # ABSTAIN, never fabricate. Replacing a missing/out-of-range score with the
    # environment's OWN success flag (`hi if traj.success else lo`) would read
    # ground truth -- and RDA's `problem.fitness_access: none` forbids touching
    # it, so a task-successful but instruction-violating policy could score
    # perfectly whenever judging failed. A missing subtask therefore ABSTAINS
    # -- it carries no score into the fitness mean -- unless the provider is
    # `mock`, where a deterministic synthetic stand-in is acceptable because
    # there is no real judge and the offline suite must still exercise the
    # path. (No re-query is made: a second analysis call is its own judge
    # round-trip that has to be independently budgeted, recorded as its own
    # `note_query`, and frame-attached -- it does not fit the
    # one-query-many-subtask-answers trace model, and abstaining already
    # removes the ground-truth leak.)
    is_mock = getattr(client, "provider", None) == "mock"
    results: List[Tuple[Optional[float], str, str]] = []
    for sub in subtasks:
        entry = entries.get(sub) or {}
        score = _valid(entry)
        behavior = " ".join(str(entry.get("behavior") or "").split())[:600]
        analysis = (" ".join(str(entry.get("analysis") or "").split())[:600]
                    if want_reason else "")
        if score is None:
            if is_mock:
                # No real judge: a deterministic stand-in, RECORDED as synthetic
                # so it is never mistaken for a judgment. Confined to `mock`
                # -- a real provider abstains below.
                used = hi if traj.success else lo
                _note_answer(ctx, trace, query_id, raw=text, parsed=None, used=used,
                             role="vlm_subtask_score", subtask=sub,
                             reason=analysis, fallback="synthetic_mock")
                results.append((used, behavior, analysis))
            else:
                # ABSTAIN: no judgment was returned, and fabricating one from
                # ground truth would leak it. The subtask contributes nothing to
                # the mean; if every subtask abstains the report collapses to
                # `select.failure_value`, not to a GT-derived score.
                _note_answer(ctx, trace, query_id, raw=text, parsed=None, used=None,
                             role="vlm_subtask_score", subtask=sub,
                             reason=analysis, fallback="abstain")
                results.append((None, behavior, analysis))
            continue
        # `parsed` is what the judge SAID, `used` is what reached the fitness --
        # the trace convention the fallback above already follows. Snapping
        # before recording would file a judge's 0.8 as `parsed: 1.0`, leaving
        # the literal answer only in `raw`.
        parsed = float(score)
        used = _snap(parsed, scale)
        extra = {"used": used, "snapped_by": scale} if used != parsed else {}
        _note_answer(ctx, trace, query_id, raw=text, parsed=parsed,
                     role="vlm_subtask_score", subtask=sub, reason=analysis,
                     **extra)
        results.append((used, behavior, analysis))
    return results


def _analysis_entries(text: str, subtasks: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Match the reply's `subtasks` array back onto the subtask list.

    By `number` first (1-based, the paper's own field) and by `name` second, so
    a judge that renumbers but names correctly -- or names sloppily but counts
    correctly -- still lands its verdicts. An entry that matches neither is
    dropped; the caller's per-subtask fallback covers the hole.
    """
    data = extract_json(text or "")
    if isinstance(data, dict):
        items = data.get("subtasks")
    elif isinstance(data, list):
        items = data
    else:
        items = None
    out: Dict[str, Dict[str, Any]] = {}
    by_name = {str(s).strip().lower(): s for s in subtasks}
    for item in items or []:
        if not isinstance(item, dict):
            continue
        sub = None
        num = _num(item.get("number"))
        if num is not None and 1 <= int(num) <= len(subtasks):
            sub = subtasks[int(num) - 1]
        if sub is None:
            sub = by_name.get(str(item.get("name") or "").strip().lower())
        if sub is not None and sub not in out:
            out[sub] = item
    return out


def render_trajectory(t: Trajectory, label: str, *, stride: int = 10,
                      horizon: Optional[int] = None,
                      rewards_override: Optional[Sequence[float]] = None,
                      show_success: bool = True,
                      state_fields: Sequence[str] = ()) -> str:
    """CARD's rollout rendering: per-step named variables + sub-reward values,
    at a stride, **assembled offline** (§4).

    The offline part is the design claim. CARD explicitly declines to hand the
    rollout to an LLM analyzer (`evaluate.feedback.analyzer: none`) on token
    cost, and prints the numbers instead; Auto MC-Reward is the opposite point
    on that same axis.

    The step sequence is the release's exactly (utils.py:184-186):
    `[0] + range(stride-1, n, stride)`, with the
    FINAL step appended unconditionally -- the terminal state is where a goal
    bonus lives, and a rendering that omits it hides the one step that says
    whether the task was solved. When the `_TRAJ_MAX_STEPS` prompt-budget cap
    binds, the sequence is re-strided EVENLY across the whole episode keeping
    both endpoints, never head-truncated (a `[:24]` would stop a 500-step
    Meta-World rollout at t=230). A terminal sentence in the release's own
    words (utils.py:213-216) closes the rendering.

    `rewards_override` replaces the `reward=` channel: §6's preference report
    passes the CANDIDATE's re-scored per-step values there, because
    `t.rewards` is what the reward in force at COLLECTION time paid.
    Component and observation channels stay the stored ones, as the release's
    convert_traj_to_str renders them.

    `state_fields` names the observation columns: the adapter's declared
    `_state_fields`, in observation order. Every backend stores `states` as an
    ndarray (`training._rollout`), so rendering named variables only for
    dict-shaped states would leave CARD's §4.2.2 "observation parameters"
    channel (release utils.py:189-195, the `mapping_dicts` keys per sampled
    step) empty on every backend.
    The first `_TRAJ_MAX_VARS` fields are shown, and a legend line says "N of
    M" whenever that cap -- or the component cap -- omits anything.

    `show_success=False` is `_show_env_success` reaching this renderer: no
    `success=` in the header, and a terminal sentence that says the episode
    ended, never whether the task was solved. Two facts close it --
    `evaluate.feedback.include_task_metric: false` (what the method may see)
    and a task whose benchmark ships no success check (rule N, what there is
    to see) -- and the caller decides, not this function.

    Public (no underscore): `update._preference_report` imports it.
    """
    if rewards_override is not None:
        # No collection-time return/mean in the header: the surrounding prose
        # quotes the CANDIDATE's re-scored aggregates, and an unlabelled stored
        # `return=` two lines above them is two contradictory returns for one
        # rollout -- the exact collection-time-vs-current-reward ambiguity the
        # re-scored report exists to remove (the release's rendering has no
        # digest header at all, utils.py:180-212).
        flag = f"success={t.success} " if show_success else ""
        head = f"  [{label}] {flag}length={t.length}"
    else:
        head = f"  [{label}] {_traj_digest(t, show_success)}"
    n = int(t.length or len(list(t.rewards)) or 0)
    if n <= 0:
        return head
    stride = max(1, int(stride))
    steps = [0] + list(range(stride - 1, n, stride))
    if steps[-1] != n - 1:
        steps.append(n - 1)
    if len(steps) > _TRAJ_MAX_STEPS:
        # Even coverage of [0, n-1] with both endpoints kept. `linspace` +
        # dict.fromkeys dedupes while preserving order.
        steps = list(dict.fromkeys(int(round(x))
                                   for x in np.linspace(0, n - 1, _TRAJ_MAX_STEPS)))
    every = [(k, v) for k, v in t.component_values.items() if k not in _REFERENCE_KEYS]
    comps = every[:_TRAJ_MAX_VARS]
    fields = tuple(state_fields) if _array_states(t.states) else ()
    rewards = list(rewards_override) if rewards_override is not None else list(t.rewards)
    lines = [head]
    # Say what the caps left out, so a partial rendering never reads as the
    # whole reward or the whole observation (the same rule as the judge's table).
    omitted = []
    if len(every) > len(comps):
        omitted.append(f"reward components: the first {len(comps)} of {len(every)}")
    if len(fields) > _TRAJ_MAX_VARS:
        omitted.append(f"observation columns: the first {_TRAJ_MAX_VARS} of "
                       f"{len(fields)} declared fields")
    if omitted:
        lines.append("      (showing " + "; ".join(omitted) + ")")
    for s in steps:
        bits = []
        for name, series in comps:
            seq = list(series)
            if s < len(seq) and seq[s] is not None:  # None: nothing claimed at s
                bits.append(f"{name}={_fmt(_num(seq[s]))}")
        for name, value in _state_vars(t, s, fields):
            bits.append(f"{name}={_fmt(value)}")
        if s < len(rewards):
            bits.append(f"reward={_fmt(_num(rewards[s]))}")
        if bits:
            lines.append(f"      t={s:<4} " + "  ".join(bits))
    # The release's terminal sentence, verbatim where its precondition holds
    # (utils.py:213-216). Its failure line asserts "maximum trajectory length
    # ... reached", which is true only for episodes that ran to the horizon --
    # the release's fixed-length eval envs always do, so it cannot hit the
    # third state; BIRD episodes can terminate early unsuccessfully, and a
    # truthful variant beats a false published sentence.
    if not show_success:
        # `include_task_metric: false`: the sentence may say the episode ended
        # and how long it ran, never whether the task was solved.
        lines.append(f"      At step {n}, the trajectory ends.")
    elif t.success:
        lines.append(f"      At step {n}, the agent solves the task, so the "
                     f"trajectory ends.")
    elif horizon is None or n >= int(horizon):
        lines.append(f"      At step {n}, the agent still has not solved the task "
                     f"and the maximum trajectory length of {n} is reached, so the "
                     f"trajectory ends.")
    else:
        lines.append(f"      At step {n}, the episode ended without success, before "
                     f"the maximum trajectory length was reached.")
    return "\n".join(lines)


def _array_states(states: Any) -> bool:
    """True for the shape every backend stores: an array (or list) of per-step
    observation vectors -- not a dict of named channels, not a list of dicts."""
    if states is None or isinstance(states, dict):
        return False
    try:
        n = len(states)
    except TypeError:
        return False
    return n > 0 and not isinstance(states[0], dict)


def _state_vars(t: Trajectory, step: int,
                fields: Sequence[str] = ()) -> List[Tuple[str, Optional[float]]]:
    """Named state variables at one step.

    `Trajectory.states` is `Any` on purpose (types.py) -- a dict of named
    channels, an array, or nothing. A dict renders its own names. An array --
    what `training._rollout` stores on every backend -- renders through
    `fields`, the adapter's declared observation names in observation order;
    with no `fields` an anonymous array is still left out
    rather than printed as `s[0]..s[173]`, which is prompt noise. The first
    `_TRAJ_MAX_VARS` in either case; `render_trajectory` states the cap."""
    states = getattr(t, "states", None)
    if isinstance(states, dict):
        out = []
        for k, v in list(states.items())[:_TRAJ_MAX_VARS]:
            seq = list(v) if isinstance(v, (list, tuple)) else None
            if seq is not None and step < len(seq):
                out.append((k, _num(seq[step])))
        return out
    if isinstance(states, (list, tuple)) and step < len(states) and isinstance(states[step], dict):
        return [(k, _num(v)) for k, v in list(states[step].items())[:_TRAJ_MAX_VARS]]
    if fields and _array_states(states) and step < len(states):
        try:
            row = np.asarray(states[step], dtype=float).ravel()
        except (TypeError, ValueError):
            return []
        return [(str(name), _num(row[i]))
                for i, name in enumerate(list(fields)[:_TRAJ_MAX_VARS]) if i < row.size]
    return []


def _pick_examples(ctx: Any, result: TrainResult) -> List[Tuple[str, Trajectory]]:
    """`evaluate.feedback.trajectory_examples` -> the rollouts to render."""
    mode = ctx.cfg.get("evaluate.feedback.trajectory_examples", "none")
    trajs = _rollouts(ctx, result)
    if mode == "none" or not trajs:
        return []
    if mode == "best_and_worst":
        # CARD: highest- and lowest-return rollouts. `ret` is the candidate's own
        # reward here, which is correct -- these examples are prose for §6, not a
        # fitness, and the point is to show the LLM what its own reward paid for.
        order = sorted(trajs, key=lambda t: _num(t.ret) if _num(t.ret) is not None else 0.0)
        if len(order) == 1:
            return [("only rollout", order[0])]
        return [("highest return", order[-1]), ("lowest return", order[0])]
    if mode == "failures_only":
        return [(f"failure {i}", t) for i, t in enumerate(trajs) if not t.success]
    if mode == "random_k":
        # k is `evaluate.rollouts_per_candidate`; ctx.rng keeps it reproducible.
        k = min(len(trajs), int(ctx.cfg.get("evaluate.rollouts_per_candidate", 3) or 3))
        chosen = ctx.rng.sample(list(range(len(trajs))), k) if k < len(trajs) else \
            list(range(len(trajs)))
        return [(f"rollout {i}", trajs[i]) for i in sorted(chosen)]
    return []


# ==========================================================================
# feedback_builder
# ==========================================================================


@register("feedback_builder", "default")
def feedback_default(ctx: Any, state: Any, report: CandidateReport) -> Tuple[str, str]:
    """Assemble stage 6's prose from `evaluate.feedback.*`, and name its channel.

    One builder, composed from leaf keys, rather than one builder per method:
    Eureka's reflection is `numeric_reflection + per_component`, CARD's process
    feedback is the same two keys plus `trajectory_examples: best_and_worst`,
    RDA's is `visual_analysis + per_subtask + summarisation`, Auto MC-Reward's
    is `analyzer: llm`. The sections below are those keys, in a fixed order.

    The return value's second half is the **channel name** -- what actually went
    in. §6's `update.feedback.routing: tpe_verdict` switches on it (CARD: pass
    -> process + trajectory feedback; fail -> a preference report *instead*,
    exclusive), so the name has to describe content, not intent."""
    _resolve_deferred(ctx, report)

    cand = report.candidate
    channels: List[str] = []

    # -- the two short-circuits: nothing was trained, so nothing can be reflected on --
    if not cand.valid:
        # A program that never compiled is its own population. The
        # traceback *is* the feedback (Eureka's failure sentinel, LIMEN's
        # `teach_next_iteration`, GT's resample-with-trace).
        return (f"Candidate {report.cand_id} failed verification "
                f"({cand.failure_kind or 'invalid'}):\n  {cand.failure}"), "failure"

    if cand.screened_out and ctx.cfg.get("evaluate.skip_if_screened", False):
        return _screen_feedback(ctx, report)

    sections: List[str] = []
    sections.append(_section_header(ctx, report))

    if ctx.cfg.get("evaluate.feedback.numeric_reflection", False):
        block = _section_numeric(ctx, report)
        if block:
            # Single source of truth for §1's NUMERIC REFLECTION section:
            # generation._numeric_reflection prefers this meta key; without it
            # that function's head-truncating fallback would fire and the prompt
            # would carry two disagreeing renditions of the same curves.
            report.meta["numeric_reflection"] = block
            sections.append(block)
            channels.append("process")

    if ctx.cfg.get("evaluate.feedback.visual_analysis", False) or report.subtask_scores:
        block = _section_subtasks(ctx, report)
        if block:
            sections.append(block)
            channels.append("visual")

    traj_block = _section_trajectories(ctx, report)
    if traj_block:
        sections.append(traj_block)
        channels.append("trajectory")

    demo_block = _demo_trace_table(ctx, report.candidate)
    if demo_block:
        # The verifier's evidence goes to the reflecting LLM as well as to the
        # judge: "your reward paid the solution policy 0.31/step and random
        # 0.29/step" is the most direct diagnosis of a flat or gamed reward
        # there is, and it exists before any training was paid for.
        sections.append(demo_block.rstrip())
        channels.append("demo")

    if ctx.cfg.get("evaluate.feedback.analyzer", "none") == "llm":
        block = _section_analyzer(ctx, report)
        if block:
            sections.append(block)
            channels.append("analysis")

    pref_block = _section_preference(ctx, report)
    if pref_block:
        sections.append(pref_block)
        channels.append("preference")

    human = (getattr(state, "human_feedback", "") or "").strip()
    if human and ctx.cfg.get("evaluate.human.mode", "none") != "none":
        sections.append("Human feedback (treated as ground truth):\n  " + human)
        channels.append("human")

    summary = _section_summary(ctx, report)
    if summary:
        sections.append(summary)

    prose = "\n\n".join(s for s in sections if s)
    return prose, ("+".join(channels) if channels else "none")


def _screen_feedback(ctx: Any, report: CandidateReport) -> Tuple[str, str]:
    """CARD: a screened-out candidate is never evaluated -- its feedback comes
    from the screen itself (`evaluate.skip_if_screened`, §4).

    The screen names its own channel on `candidate.meta["feedback_channel"]`
    when it has one. Absent that we call it `preference`, because the only
    method that sets `skip_if_screened` is the one whose screen substitutes a
    preference report (order-accuracy % + the most-violating success/failure
    pair) for process feedback, and §6's `tpe_verdict` routing is switching on
    exactly that."""
    cand = report.candidate
    meta = getattr(cand, "meta", {}) or {}
    text = ""
    for key in ("screen_feedback", "feedback", "tpe_report", "report"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            text = v.strip()
            break
    if not text:
        for rec in cand.verify_records or []:
            if isinstance(rec, dict):
                for key in ("feedback", "report", "message", "detail"):
                    v = rec.get(key)
                    if isinstance(v, str) and v.strip():
                        text = v.strip()
                        break
            if text:
                break
    if not text:
        text = cand.failure or "the quality screen rejected this candidate"
    channel = meta.get("feedback_channel") or "preference"
    return (f"Candidate {report.cand_id} did not pass the quality screen; training was "
            f"skipped and the screen's own report replaces the usual evaluation.\n  {text}"), \
        str(channel)


def _section_header(ctx: Any, report: CandidateReport) -> str:
    # `state_selection_scalar: false` keeps the fitness label, its aggregation
    # rule and the per-seed values out of the prompt: Eureka's feedback is
    # statistics + tips only, the selection scalar appearing solely as the
    # unlabeled Max of the task_score curve (eureka.py:250-267), and no global
    # best is ever stated. Prose only -- the scalar itself is untouched, `None`
    # stays distinct from 0.0, and stage 5 still selects on it. The CARD
    # no-scalar sentence below stays
    # unconditional: it reports an absence, not the scalar's value.
    show_scalar = bool(ctx.cfg.get("evaluate.feedback.state_selection_scalar", True))
    bits = [f"Candidate {report.cand_id}"]
    if report.fitness is not None:
        if show_scalar:
            bits.append(f"fitness={_fmt(report.fitness, 4)} "
                        f"(source={report.fitness_source}, "
                        f"{ctx.cfg.get('evaluate.fitness.checkpoint_aggregation')}/"
                        f"{ctx.cfg.get('evaluate.fitness.seed_aggregation')})")
    else:
        # The CARD reading: say so, so the prompt does not imply a ranking that
        # the search never computed.
        bits.append("no ranking scalar was computed for this candidate")
    if show_scalar and report.per_seed_fitness and len(report.per_seed_fitness) > 1:
        bits.append(f"per-seed={_fmt_list(report.per_seed_fitness)}")
    if report.similarity is not None and \
            ctx.cfg.get("evaluate.similarity.role", "report") != "report":
        # `role: report` means "records only" -- the number goes to the journal
        # and the artifact, never into the prompt. Verbalising it here would leak
        # a correlation-with-the-hidden-reference scalar to the generator, which
        # eureka.py explicitly withholds (its feedback loop skips gt_reward and
        # gpt_reward).
        bits.append(f"{ctx.cfg.get('evaluate.similarity.metric')} vs "
                    f"{ctx.cfg.get('evaluate.similarity.reference')}="
                    f"{_fmt(report.similarity)}")
    return "; ".join(bits) + "."


def training_caveat(ctx: Any, report: Any) -> str:
    """A sentence saying the previous policy's training was CUT SHORT, when it was.

    WITHOUT THIS THE MODEL IS ASKED TO EXPLAIN A NUMBER IT CANNOT INTERPRET.
    `TrainResult` already argues the point for the artifact: a pruned run's
    fitness "is therefore a LOWER BOUND on what the reward would have reached",
    and treating it as a completed one is "exactly the confusion the
    `failure`/`failure_kind` split exists to prevent one level up".

    It lives HERE, and `_section_numeric` prepends it, because that block is
    the one rendition both prompt-side readers share: `feedback_default` writes
    it to `report.meta["numeric_reflection"]` (§1's NUMERIC REFLECTION section)
    and into `report.feedback` (the dialogue turn §6 carries). Emitting it
    only from `generation._behavioural_analysis`, which eureka never renders,
    would let a pruned Eureka run (e.g. `-s train.pruning=median_stop`) reflect
    on winners pruned at round 8-13 of 20 as complete curves under "Training
    statistics, sampled at checkpoints:", with Eureka's tips ("near identical
    throughout", "always near zero") applied to a curve that stopped at 40% of
    budget. Under pruning, without the caveat "this reward is bad" and "this
    reward was never trained" are the same observation.

    The sentence names the curve `train.pruning_metric` actually watched
    (`own_reward`: the policy's return under this reward; `task_metric`: the
    environment's task metric) rather than asserting "its own learning curve"
    for both. Two outcomes, kept apart for the reason the type does: `pruned`
    is a DELIBERATE cut by `train.pruning`, `timed_out` is `train.timeout_s`
    running out on a run nobody chose to stop. The remedies differ, so the
    sentences do.

    Public (no underscore): `generation._training_caveat` imports it for the
    behavioural channel of a method with no numeric one.
    """
    result = getattr(report, "result", None)
    if result is None:
        return ""
    bits = []
    rows = getattr(result, "seed_metrics", None) or []
    halving = [m for m in rows if m.get("halving_dropped")]
    if halving:
        # `train.pruning: successive_halving_pool` (rounds.py) dropped this
        # candidate on the POOL RANKING, not on its own curve -- which may have
        # been rising. Saying "stopped improving" here would be false for exactly
        # the candidates the pool ranks below the median.
        rung = max(int(m.get("rungs", 1) or 1) for m in halving)
        metric = str(ctx.cfg.get("train.pruning_metric", "own_reward") or "own_reward")
        bits.append(
            f"NOTE: this policy's training was STOPPED EARLY after rung {rung} of the "
            f"successive-halving pool: its `{metric}` ranked below the pool's cut at that "
            "rung (train.pruning: successive_halving_pool), NOT because its own curve had "
            "stopped improving -- it may still have been rising. It did not spend its full "
            "step budget, so the score below is a LOWER BOUND on what this reward would "
            "have reached, not a measurement of what it did reach.")
    elif getattr(result, "pruned", False):
        rounds = sorted({m.get("pruned_at_round") for m in rows if m.get("pruned_at_round")})
        where = f" at checkpoint {rounds[0]}" if len(rounds) == 1 else ""
        metric = str(ctx.cfg.get("train.pruning_metric", "own_reward") or "own_reward")
        watched = {
            "own_reward": "the policy's return under its own reward "
                          "(train.pruning_metric: own_reward)",
            "task_metric": "the environment's task metric "
                           "(train.pruning_metric: task_metric)",
        }.get(metric, f"the `{metric}` curve (train.pruning_metric)")
        bits.append(
            "NOTE: this policy's training was STOPPED EARLY" + where + " because "
            + watched + " stopped improving, so it did not spend its full step "
            "budget. The score below is a LOWER BOUND on what this reward would have "
            "reached, not a measurement of what it did reach.")
    if getattr(result, "timed_out", False):
        bits.append(
            "NOTE: this policy's training hit its wall-clock limit before its step "
            "budget, so its curve is truncated and the score below is a lower bound "
            "for a reason that has nothing to do with the reward.")
    return ("\n".join(bits) + "\n\n") if bits else ""


#: Eureka's `policy_feedback.txt`, VERBATIM, with its one slot.
#:
#: `refs/code/Eureka/eureka/utils/prompts/policy_feedback.txt` is a single
#: sentence and `eureka.py:251` renders it as `policy_feedback.format(
#: epoch_freq=epoch_freq)`. It is kept here as the paper's own text because the
#: reflection IS part of the next prompt, and a changed prompt is a different
#: method.
#:
#: `bird/components/tree.py::_TRAINED_FRAMING` is the same sentence with
#: "after every {epoch_freq} epochs" replaced by "at each training checkpoint",
#: for the checkpoint-curve rendering, which has no epoch stride to name. Under
#: `evaluate.feedback.series_source: training_epochs` there is one, and this
#: constant is the exact form. tree.py's constant is deliberately separate --
#: it feeds RF-Agent's action prompts, whose release renders the slot as the
#: literal string "None", and whether to reproduce that is a separate choice
#: for the RF-Agent point.
EUREKA_POLICY_FEEDBACK = (
    "We trained a RL policy using the provided reward function code and "
    "tracked the values of the individual components in the reward function as "
    "well as global policy metrics such as success rates and episode lengths "
    "after every {epoch_freq} epochs and the maximum, mean, minimum values "
    "encountered:"
)


def _section_epoch_series(ctx: Any, report: CandidateReport) -> str:
    """`evaluate.feedback.series_source: training_epochs` -- Eureka's block.

    `eureka.py:246-268`, operation for operation:

        max_iterations = len(tensorboard_logs['gt_reward'])     :249
        epoch_freq     = max(int(max_iterations // 10), 1)      :250
        for metric in tensorboard_logs:                         :254
            if "/" in metric: continue                          :255
            values  = metric[::epoch_freq]                      :256
            Max/Mean/Min over the FULL series                   :257-259
            consecutive_successes renders as "task_score"       :261-264
            gt_reward / gpt_reward are SKIPPED from the listing  :265
            ...unless there is no consecutive_successes, when
            gt_reward renders as "ground-truth score"           :266-268

    THREE THINGS IN THAT LIST ARE EASY TO GET WRONG AND ALL THREE CHANGE WHAT
    THE LLM IS TOLD.

    (1) The stride is a STRIDE, not an even sample. `[::epoch_freq]` starts at
    index 0 and may omit the last epoch; `_sample_evenly` includes both
    endpoints. On a 3000-epoch log the two pick different points, and the last
    point is the one a reader weights most.

    (2) Max/Mean/Min are over the FULL series, never over the strided list.
    Computing them on the ~10 shown values is a different statistic that looks
    identical in the prose.

    (3) `gpt_reward` is skipped. It is the candidate's OWN reward, and showing
    it back is showing the model its own claim as if it were evidence -- which
    is the same argument `_REFERENCE_KEYS` makes for never verbalising the
    reference reward, applied to the other side. The correlation between the
    two is what Eureka computes from them (`:243-248`), and that is a scalar
    the similarity family already reports.

    `evaluate.feedback.include_task_metric: false` still withholds the metric
    line, and `metric_alias` still overrides its label -- both apply here
    exactly as they do on the checkpoint path, because a method that may not
    see the ground-truth curve may not see it in either rendering.
    """
    res = report.result
    series = dict(getattr(res, "epoch_series", {}) or {})
    if not series:
        # NOT an empty block, and not silence either: a config asked for a
        # rendering its backend cannot supply, and saying so is the difference
        # between "the learner logged nothing" and "this backend has no
        # per-epoch log". The `gt_reward_curve` convention -- no channel is a
        # state, never zero.
        return ("Training statistics: this backend records no per-epoch series, "
                "so evaluate.feedback.series_source=training_epochs has nothing "
                "to render (it needs a backend that records a per-epoch named "
                "log under train.log: epoch_scalars).\n")

    # `max_iterations` is the LOG'S OWN LENGTH, as eureka.py:249 reads it off
    # `gt_reward`. Longest series rather than a named one, so a task that
    # publishes no `gt_reward` still strides at its own epoch count.
    n_epochs = max((len(v) for v in series.values()), default=0)
    freq = max(int(n_epochs // 10), 1)
    show_metric = _show_task_metric(ctx)
    alias = ctx.cfg.get("evaluate.feedback.metric_alias") or "task_score"
    gran = ctx.cfg.get("evaluate.feedback.granularity", "scalar_only")

    lines: List[str] = [EUREKA_POLICY_FEEDBACK.format(epoch_freq=freq)]
    has_success = bool(series.get("consecutive_successes"))

    def _row(label: str, values: Sequence[float]) -> str:
        full = _clean(values)
        shown = [float(v) for v in list(values)[::freq]]
        return (f"{label}: {_fmt_list(shown)}, "
                f"Max: {_fmt(float(full.max()) if full.size else None, 2)}, "
                f"Mean: {_fmt(float(full.mean()) if full.size else None, 2)}, "
                f"Min: {_fmt(float(full.min()) if full.size else None, 2)}")

    for name in series:
        if "/" in name:
            continue           # eureka.py:255
        values = series[name]
        if not values:
            continue
        if name == "consecutive_successes":
            if show_metric:
                lines.append(_row(alias, values))
            continue
        if name == "gpt_reward":
            continue           # eureka.py:265 -- the candidate's own claim
        if name == "gt_reward":
            # eureka.py:266-268: rendered ONLY as the fallback metric when the
            # task publishes no success rate, and it is a ground-truth channel,
            # so `include_task_metric: false` withholds it too.
            if not has_success and show_metric:
                lines.append(_row("ground-truth score", values))
            continue
        if gran in ("per_component", "per_subtask"):
            lines.append(_row(name, values))
    if gran == "scalar_only":
        n_comp = sum(1 for k, v in series.items()
                     if "/" not in k and v and k not in
                     ("consecutive_successes", "gt_reward", "gpt_reward"))
        if n_comp:
            lines.append(f"({n_comp} component series withheld: "
                         f"evaluate.feedback.granularity=scalar_only)")
    if len(lines) <= 1:
        return ""
    return training_caveat(ctx, report) + "\n".join(lines)


def _section_numeric(ctx: Any, report: CandidateReport) -> str:
    """Eureka's reward reflection / CARD's process feedback (§4).

    Both are the same family: per-component scalar traces sampled at checkpoints
    and verbalised, plus return / episode length / success mean+std. What
    `evaluate.feedback.granularity` changes is whether the components are
    itemised at all -- and component visibility is a §1 precondition
    (`generate.output.format` must return a named-component dict), which
    `config._check_coherence` already enforces."""
    # `evaluate.feedback.series_source` routes BEFORE anything is rendered:
    # the two paths read different series and stride them differently, so they
    # are two renderings and not one with an option (see
    # `_section_epoch_series`). Routed on the CONFIG VALUE, never on the
    # backend -- a backend test here would be the method branching
    # `tests/test_no_method_branching.py` forbids one level up, and it would
    # make the key unable to say "render the checkpoint curve even though a
    # per-epoch log exists".
    if ctx.cfg.get("evaluate.feedback.series_source", "checkpoints") == "training_epochs":
        return _section_epoch_series(ctx, report)

    res = report.result
    gran = ctx.cfg.get("evaluate.feedback.granularity", "scalar_only")
    lines: List[str] = ["Training statistics, sampled at checkpoints:"]

    # `include_task_metric: false` is GT's §3 assumption made real: "the
    # fitness function F is not accessible while searching", so the reflection
    # carries component traces, the return and (beyond App. B's eta, :636-642)
    # the episode length -- but not the environment's
    # ground-truth curve or a success rate. The checkpoint "fitness" key below
    # is filled by _evaluate_policy from env.task_metric, so with the line in,
    # a fitness_access: none method could hill-climb the hidden metric
    # in-context. Default true: Eureka and CARD both
    # verbalise task success to the LLM.
    show_metric = _show_task_metric(ctx)
    # Native-signal rule (N): the numeric reflection verbalises to the generator
    # the SAME signal the search is scored on -- `evaluate.fitness.source`. When
    # that is native/native_success/native_reward, the metric line is the native
    # channel's checkpoint curve (`success_rate` / `gt_return`), NEVER the BIRD
    # `task_metric` (custom_metric); a task with no native signal shows no metric
    # line (the generator sees no env metric). When the source is a non-native one
    # (`ground_truth_metric` / `success_rate`), the line stays the metric it is
    # scored on.
    # Setting `evaluate.fitness.source: native` for a supervised method therefore
    # switches BOTH the fitness and this feedback to native.
    # `evaluate.feedback.metric_alias` still overrides the label (Eureka renames
    # its metric before verbalising it).
    _fsrc = ctx.cfg.get("evaluate.fitness.source", "native")
    if _fsrc in ("native", "native_success", "native_reward"):
        from ..native_signal import native_curve_keys, NATIVE_SUCCESS
        _nkeys, _nchannel = native_curve_keys(env_id=ctx.cfg.get("problem.env_id"),
                                              task_id=ctx.cfg.get("problem.task_id"))
        if show_metric and _nkeys:
            _default_label = ("success rate" if _nchannel == NATIVE_SUCCESS
                              else "reference reward return")
            label = ctx.cfg.get("evaluate.feedback.metric_alias") or _default_label
            for curve in _checkpoint_curves(res):
                series = _series(curve, _nkeys)
                if series:
                    lines.append(f"  {label}: {_fmt_list(_sample_evenly(series, _REFLECT_POINTS))}"
                                 f"  (max {_fmt(max(series))}, mean {_fmt(float(np.mean(series)))}, "
                                 f"min {_fmt(min(series))})")
                    break
    elif _show_env_success(ctx):
        metric = ctx.cfg.get("evaluate.fitness.metric", "task_success")
        alias = ctx.cfg.get("evaluate.feedback.metric_alias") or metric
        for curve in _checkpoint_curves(res):
            series = _series(curve, (metric, "fitness", "score"))
            if series:
                lines.append(f"  {alias}: {_fmt_list(_sample_evenly(series, _REFLECT_POINTS))}"
                             f"  (max {_fmt(max(series))}, mean {_fmt(float(np.mean(series)))}, "
                             f"min {_fmt(min(series))})")
                break

    if gran in ("per_component", "per_subtask") and res.component_traces:
        lines.append("  per-component reward values:")
        for name, trace in res.component_traces.items():
            if name in _REFERENCE_KEYS:
                continue  # never verbalise the reference reward (see _REFERENCE_KEYS)
            vals = _clean(trace)
            if not vals.size:
                continue
            lines.append(
                f"    {name}: {_fmt_list(_sample_evenly(list(vals), _REFLECT_POINTS))}"
                f"  (max {_fmt(float(vals.max()))}, mean {_fmt(float(vals.mean()))}, "
                f"min {_fmt(float(vals.min()))})")
    elif gran == "scalar_only" and res.component_traces:
        # Deliberate: the traces exist but the config asked for scalar-only
        # feedback. Honouring that is the whole point of the granularity axis --
        # RDA attributes its gains to the credit assignment this key controls,
        # and an ablation of it is worthless if the components leak in anyway.
        lines.append(f"  ({len(res.component_traces)} component traces withheld: "
                     f"evaluate.feedback.granularity=scalar_only)")

    trajs = _rollouts(ctx, res)
    if trajs:
        rets = _clean([t.ret for t in trajs])
        lens = _clean([t.length for t in trajs])
        succ = _clean([1.0 if t.success else 0.0 for t in trajs])
        if rets.size:
            # A known Eureka fidelity gap, deliberately not keyed: this line is
            # computed under the CANDIDATE'S OWN reward, which upstream's
            # stringifier excludes (gpt_reward skipped, eureka.py:258). A config
            # key per wording detail is worse than recording the gap.
            lines.append(f"  episode return: mean {_fmt(float(rets.mean()))} "
                         f"+/- {_fmt(float(rets.std()))}")
        if lens.size:
            lines.append(f"  episode length: mean {_fmt(float(lens.mean()), 1)} "
                         f"+/- {_fmt(float(lens.std()), 1)}")
        if succ.size and _show_env_success(ctx):
            lines.append(f"  success: mean {_fmt(float(succ.mean()))} "
                         f"+/- {_fmt(float(succ.std()))} over {succ.size} rollout(s)")
    if len(lines) <= 1:
        return ""
    # A pruned or timed-out curve is announced BEFORE it is shown, here and
    # nowhere else on the numeric channel: this block is both §1's NUMERIC
    # REFLECTION section (via `meta["numeric_reflection"]`) and part of the
    # carried dialogue turn (via `report.feedback`), so one emission reaches
    # both renditions (see `training_caveat`).
    return training_caveat(ctx, report) + "\n".join(lines)


def _section_subtasks(ctx: Any, report: CandidateReport) -> str:
    """RDA's per-subtask credit assignment (§4, `granularity: per_subtask`)."""
    if not report.subtask_scores:
        return ""
    scale = ctx.cfg.get("evaluate.feedback.score_scale", "[0,1]")
    lines = [f"Per-subtask evaluation (scores on {scale}):"]
    for sub, score in sorted(report.subtask_scores.items(), key=lambda kv: kv[1]):
        line = f"  {sub}: {_fmt(score)}"
        reason = report.subtask_rationales.get(sub)
        if reason and ctx.cfg.get("evaluate.feedback.require_rationale", False):
            line += f"\n      reason: {reason}"
        lines.append(line)
    return "\n".join(lines)


def _traj_stride(ctx: Any) -> int:
    """`evaluate.feedback.trajectory_sample_interval`: the per-step stride at
    which an example rollout is verbalised. CARD publishes it (App. B.2 Table 8:
    100 on Meta-World, 25 on ManiSkill2 = `--log_interval`); the default is
    10."""
    return max(1, int(ctx.cfg.get("evaluate.feedback.trajectory_sample_interval", 10) or 10))


def _section_trajectories(ctx: Any, report: CandidateReport) -> str:
    examples = _pick_examples(ctx, report.result)
    if not examples:
        return ""
    mode = ctx.cfg.get("evaluate.feedback.trajectory_examples")
    # Hoisted: the gate reads the task spec (a scan of the memoised index), and
    # it is one answer for the whole section, not one per rollout.
    show = _show_env_success(ctx)
    body = "\n".join(render_trajectory(t, label, stride=_traj_stride(ctx),
                                       horizon=getattr(ctx.env, "horizon", None),
                                       show_success=show,
                                       state_fields=_declared_state_fields(ctx))
                     for label, t in examples)
    return f"Example rollouts ({mode}), rendered offline:\n{body}"


def _section_analyzer(ctx: Any, report: CandidateReport) -> str:
    """`evaluate.feedback.analyzer: llm` -- an LLM writes the behaviour prose (§4).

    TWO CHANNELS, dispatched on what this round's evidence actually is -- data
    presence, the same pattern as `report.subtask_scores` gating
    `_section_subtasks`, never a method name:

    * **`report.meta["pair_captions"]` non-empty** -- GT's no-human substitution
      (App. D): the preference step's contrastive clip descriptions (App. C) are
      summarised by an LLM under the paper's explicitly OBJECTIVE instruction
      ("Remain objective ... Do no[t] give opinions on whether certain behaviors
      are 'good' or 'bad' ... Make no mention of the other agent ... refer to
      the agent as 'the agent'"). Only GT's preference step writes that key;
      without persisted captions this path would make a fresh EVALUATIVE call
      over numeric traces -- a different input channel and a framing App. D
      forbids. ‡ App. D uses a text LLM distinct from the captioning
      VLM; the schema exposes one evaluator role, so both are `ctx.evaluator`
      (same recorded gap as `comparator_llm_on_vlm_captions` step 2). The
      paper's "Do no give" [sic] is not carried into the prompt. Two more
      divergences are recorded: the paper writes
      ONE App. D summary per iteration (for the selected agent only), where
      `feedback_default` runs this for EVERY report carrying captions -- 5 live
      calls per iteration on `gt`, though only the winner's summary
      reaches the next prompt -- and the summary is clipped to the analyzer's
      1200-char convention, where the paper injects it whole.

    * **No captions** -- the Auto MC-Reward axis: hand the worst rollout to
      `ctx.evaluator` as numbers and ask what the agent is doing and where it
      goes wrong. That evaluative framing is that method's behaviour.

    The section heading names which channel ran, so a reader of the next prompt
    can tell a caption summary from a numeric-trace description.

    Worth stating the contrast the config makes visible: **CARD explicitly
    refuses this**, on token cost -- it renders the same rollouts as numbers
    offline (`trajectory_examples: best_and_worst`, `analyzer: none`) and cites
    ~14.2k tokens/run against Eureka's ~663k (§0). The two are one key apart,
    and no paper has measured what the LLM analyzer buys."""
    task = ctx.cfg.get("problem.task_description", "")
    caps = report.meta.get("pair_captions") or []
    if caps:
        # Deterministic under a concurrent live judge: appends race, so the
        # ORDER in meta is not a fact about the round -- sort before reading.
        caps = sorted(caps, key=lambda c: (str(c.get("other", "")),
                                           str(c.get("side", ""))))
        body = "\n\n".join(
            f"Description {k} (this agent is {c.get('side', '?')} in it):\n"
            f"{c.get('caption', '')}"
            for k, c in enumerate(caps, start=1))
        # App. D, "Getting Agent Behavior Descriptions" (p.17), adapted: the
        # {agent_to_describe} slot becomes the side labels above, and the
        # Gran-Turismo-specific out-of-bounds note is the task description's
        # job here.
        prompt = (
            f"Stated goal of the task: {task}\n\n"
            f"Carefully analyze the descriptions of each agent's behavior for "
            f"each clip. Then, provide a detailed summary of this agent's "
            f"behavior only, especially as it relates to the stated goal.\n"
            f"Additional notes:\n"
            f"- Remain objective for all descriptions. Do not give opinions on "
            f"whether certain behaviors are 'good' or 'bad'.\n"
            f"- Make no mention of the other agent in the summary.\n"
            f"- Start your response with '## Agent Summary'. Simply refer to "
            f"the agent as 'the agent'.\n\n"
            f"{body}")
        text = _client_text(getattr(ctx, "evaluator", None), prompt).strip()
        if not text:
            return ""
        return ("Behaviour analysis (LLM, from the VLM clip descriptions):\n  "
                + " ".join(text.split())[:1200])
    trajs = _rollouts(ctx, report.result)
    if not trajs:
        return ""
    worst = min(trajs, key=lambda t: _num(t.ret) if _num(t.ret) is not None else 0.0)
    rendered = render_trajectory(worst, "rollout", stride=_traj_stride(ctx),
                                 horizon=getattr(ctx.env, "horizon", None),
                                 show_success=_show_env_success(ctx),
                                 state_fields=_declared_state_fields(ctx))
    prompt = (f"Task: {task}\n"
              f"Describe, in two or three sentences, what the agent is doing in this rollout "
              f"and where it goes wrong.\n"
              f"{rendered}")
    text = _client_text(getattr(ctx, "evaluator", None), prompt).strip()
    if not text:
        return ""
    return ("Behaviour analysis (LLM, from numeric traces; no clip "
            "descriptions were produced this round):\n  "
            + " ".join(text.split())[:1200])


#: What `meta["bt_strength"]` IS, by the `pref_aggregator` registry value that
#: wrote it. Every simplex aggregator's strength is BT-comparable and keeps the
#: Bradley-Terry label; `elo_raw` writes the
#: raw rating (REvolve tex:993-1007), and calling 1531.26 a Bradley-Terry
#: strength in the prompt would be wrong. Keyed on a registry value, never on a
#: method name. `_section_preference` reads that value off the report
#: (`preferences._write_back` stamps it on every raced report) and falls back
#: to the configured `evaluate.preferences.aggregator` when the meta is silent
#: -- a report built by hand, or one carrying no stamp --
#: so the label can never disagree with the aggregator the run is configured
#: to use.
_STRENGTH_LABEL = {"elo_raw": "Elo rating"}


def _section_preference(ctx: Any, report: CandidateReport) -> str:
    """The preference phase's strength and tally -- prose, gated like the header.

    For `evaluate.fitness.source: preference_bt` the SELECTION SCALAR is the
    Bradley-Terry strength: `_resolve_bt` copies `meta["bt_strength"]` into
    `report.fitness`, and the win/loss/comparison tally is the data that
    strength was fitted from. So `evaluate.feedback.state_selection_scalar:
    false` hides this block exactly as it hides the `fitness=` header
    (`_section_header`): GT's App. B prompt (neurips_2025.tex:618-662) shows
    the LLM no strength, rank or tally, and b_1:N feeds only the argmax
    (:318-320). Ungated, the block would reach gt's generator twice per round
    (once in the carried user turn via `update._turns`, once in BEHAVIOURAL
    ANALYSIS via `generation._behavioural_analysis`'s `report.feedback`
    fallback) while the config states the prompt is scalar-free. Prose only:
    the numbers stay in report.json
    meta (`preferences._write_back` writes bt_strength / pref_wins /
    pref_losses / pref_comparisons / pref_rank / pref_summary) and stage 5
    still selects on `report.fitness`. There is no qualitative text here to
    keep -- App. D's objective summary is `_section_analyzer`, human feedback
    the `human` section. The caller appends the `preference` channel label
    only `if pref_block`, so `feedback_channel` drops `+preference` with it --
    the content-describing name `feedback_default`'s docstring promises.

    **A candidate that never raced has no strength to print.**
    `preferences._run_preferences` writes `bt_strength = 0.0` onto EVERY
    report as the never-raced floor before `_write_back` fills in the pool
    ("never raced" and "lost everything" stay distinguishable because
    every aggregator keeps a raced candidate strictly above 0.0), and every
    key `_write_back` stamps on a pooled report is `pref_*` (`pref_wins`
    through `pref_summary`; nothing else in `bird/` writes one). So a strength
    beside no `pref_*` key at all is the floor sentinel on a candidate that
    had no trained policy to race, not a number any aggregator produced -- the
    marker is the absence of all of them, not of one, because a report built
    by hand carries whichever subset its author typed. Printing it as a
    strength anyway would be doubly wrong: under REvolve's human variant a
    failed training would read "Preference evidence: Bradley-Terry strength
    0.000." beside peers reading "Elo rating 1672.029" -- the wrong label (the
    meta carries no `pref_aggregator` to look up, so the literal fallback
    wins) on a fabricated number (an unraced POOL member under `elo_raw` reads
    1500.0, the start rating; 0.0 is 1500 below every peer). The honest line states
    the absence, as `_section_header` does for CARD's missing scalar, and
    carries no digits. The meta keeps the 0.0: this is prose, the artifact is
    the artifact, and stage 5 still sees `select.failure_value`.
    """
    if not bool(ctx.cfg.get("evaluate.feedback.state_selection_scalar", True)):
        return ""
    meta = report.meta or {}
    v = _num(meta.get("bt_strength"))
    if v is not None and not any(k.startswith("pref_") for k in meta):
        return "Preference evidence: not compared (no trained policy to race this round)."
    bits = []
    if v is not None:
        aggregator = str(meta.get("pref_aggregator")
                         or ctx.cfg.get("evaluate.preferences.aggregator") or "")
        label = _STRENGTH_LABEL.get(aggregator, "Bradley-Terry strength")
        bits.append(f"{label} {_fmt(v)}")
    for key, label in (("pref_wins", "wins"), ("pref_losses", "losses"),
                       ("n_comparisons", "comparisons")):
        n = _num(meta.get(key))
        if n is not None:
            bits.append(f"{int(n)} {label}")
    if not bits:
        return ""
    return "Preference evidence: " + ", ".join(bits) + "."


def _section_summary(ctx: Any, report: CandidateReport) -> str:
    """`evaluate.feedback.summarisation` -- RDA's `A^i_{n,k}` -> `G^i_n` (§4).

    RDA aggregates per-subtask scores across rollouts into one summary of
    (mean score, recurring failure modes). `per_candidate` is that summary for
    the candidate as a whole; `per_candidate_per_subtask` keeps it split, which
    is what makes the feedback actionable per component.

    The aggregation itself is arithmetic and is done offline; only the
    "recurring failure modes" phrasing needs a model, and it degrades to
    the arithmetic alone when the client gives nothing back. Two points of
    fidelity live here: the summariser queries `ctx.generator` -- the Agent
    VLM does all of RDA's in-loop work, §5.1; GPT-4.1 serves only the post-hoc
    alignment metric -- and the `per_candidate_per_subtask` digest carries the
    FULL rationale set per subtask (every rollout and repeat, labeled, as
    `fitness_vlm_score` keeps it), so 'recurring' is computed over §4.2's
    whole `{rho_{n,k,j}}` rather than a sample of one."""
    mode = ctx.cfg.get("evaluate.feedback.summarisation", "none")
    if mode == "none":
        return ""
    scores = report.subtask_scores
    if not scores:
        digest = _behaviour_digest(ctx, report.result)
    elif mode == "per_candidate_per_subtask":
        digest = "\n".join(f"  {s}: mean {_fmt(v)}"
                           + (f" -- {report.subtask_rationales[s]}"
                              if s in report.subtask_rationales else "")
                           for s, v in sorted(scores.items(), key=lambda kv: kv[1]))
    else:
        digest = (f"  mean subtask score {_fmt(float(np.mean(list(scores.values()))))} "
                  f"over {len(scores)} subtask(s); weakest: "
                  f"{min(scores, key=lambda k: scores[k])}")

    text = _client_text(
        getattr(ctx, "generator", None),
        "Summarise this evaluation into the recurring failure modes it shows, in at "
        f"most three sentences.\n{digest}").strip()
    head = f"Summary ({mode}):"
    if text:
        return f"{head}\n{digest}\n  recurring failure modes: " + " ".join(text.split())[:600]
    return f"{head}\n{digest}"


# ==========================================================================
# similarity
#
# `evaluate.similarity.role` decides what the number is FOR (§4):
#   report  -- the default, and what Eureka actually does: it computes curve
#              Pearson against the GT reward and only prints it.
#   select  -- promote it into the scalar §5 reads.
#   screen  -- flag it for rejection. "Similarity as a screen, not a
#              report" is one key away; note that the schema has no
#              `evaluate.similarity.threshold`, so the flag is all this stage
#              can set -- that missing knob is where the cut-off would live.
# ==========================================================================


def _promote(ctx: Any, report: CandidateReport, name: str, value: Optional[float],
             extra: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Record the pseudometric, then honour `evaluate.similarity.role`."""
    role = ctx.cfg.get("evaluate.similarity.role", "report")
    report.meta["similarity_metric"] = name
    report.meta["similarity_role"] = role
    for k, v in (extra or {}).items():
        report.meta[k] = v
    if value is None:
        return None
    if role == "select" and report.fitness is None and report.candidate.valid:
        # Promote only into a vacancy. Overwriting a real fitness would make
        # `role` silently disable `evaluate.fitness.source`, which is two axes
        # colliding; a config that wants a blend says so in §5 with
        # `select.rule: weighted_evidence`, and finds the number in meta.
        report.fitness = float(value)
        report.meta["fitness_from_similarity"] = True
    elif role == "screen":
        report.meta["similarity_screen"] = True
    return float(value)


@register("similarity", "none")
def similarity_none(ctx: Any, state: Any, report: CandidateReport) -> Optional[float]:
    """No pseudometric. (`bird.evaluate` short-circuits before calling this; it
    is registered so the schema's enum for `evaluate.similarity.metric` and the
    registry family stay equal in both directions.)"""
    return None


def _candidate_curve(result: TrainResult) -> List[float]:
    """The candidate's own reward over training.

    Named-total first, then the elementwise sum of the component traces (a
    `component_dict_return` reward is exactly that sum), then whatever the
    checkpoints called the reward."""
    for key in ("total", "total_reward", "reward", "episode_reward"):
        trace = result.component_traces.get(key)
        if trace:
            return [float(v) for v in _clean(trace)]
    traces = [list(v) for v in result.component_traces.values() if list(v)]
    if traces:
        n = min(len(t) for t in traces)
        return [float(sum(_num(t[i]) or 0.0 for t in traces)) for i in range(n)]
    for curve in _checkpoint_curves(result):
        s = _series(curve, ("reward", "episode_reward", "mean_reward", "return"))
        if s:
            return s
    return []


def _reference_curve(ctx: Any, state: Any, report: CandidateReport) -> List[float]:
    """`evaluate.similarity.reference`: the ground-truth reward's curve under the
    same policy (Eureka), or the incumbent's curve."""
    ref = ctx.cfg.get("evaluate.similarity.reference", "gt_reward")
    if ref == "gt_reward":
        # `_clean` drops the None entries a task with no reference reward
        # writes (`training._rollout`), so the curve is EMPTY there and
        # `_pearson` answers None -- undefined, not a correlation against zeros.
        return [float(v) for v in _clean(report.result.gt_reward_curve)]
    if ref == "previous_best":
        best = getattr(state, "best", None)
        if best is None or best.result is None:
            return []
        return _candidate_curve(best.result)
    return []


@register("similarity", "pearson_curve")
def similarity_pearson(ctx: Any, state: Any, report: CandidateReport) -> Optional[float]:
    """Eureka: Pearson correlation of the candidate's reward curve against the
    ground-truth reward's, over the same training run (§4).

    Eureka computes this and uses it for **reporting only** -- it is the paper's
    evidence that LLM rewards are often *uncorrelated* with the human-written
    one and still train better policies. Promoting `evaluate.similarity.role` to
    `select` or `screen` is one key and turns the same number into a
    reward-hacking gate."""
    cand = _candidate_curve(report.result)
    ref = _reference_curve(ctx, state, report)
    r = _pearson(cand, ref)
    return _promote(ctx, report, "pearson_curve", r,
                    {"curve_points": min(len(cand), len(ref))})


def _paired_rewards(ctx: Any, state: Any, report: CandidateReport) -> Tuple[List[float], List[float]]:
    """Paired (candidate reward, reference reward) samples.

    EPIC and STARC compare two reward *functions* over a coverage distribution,
    so the samples must be paired on the same transitions. Per-step rollout
    rewards are the only place that pairing exists here: the candidate's is
    `Trajectory.rewards`, and the reference is whatever the backend logged
    alongside it (`train.log: [..., gt_reward]`). Falls back to the training
    curves so the metric still returns a number on a backend that stores no
    per-step reference."""
    cand: List[float] = []
    ref: List[float] = []
    for t in report.result.trajectories:
        series = None
        for key in _REFERENCE_KEYS:
            if key in t.component_values:
                series = list(t.component_values[key])
                break
        if series is None or not t.rewards:
            continue
        rew = list(t.rewards)
        n = min(len(series), len(rew))
        # Paired on the SAME step, so index `i` of both is step `i`; a step
        # where the reference was not claimed (`None`, per-step packing) is no
        # sample at all rather than a shifted one.
        for i in range(n):
            if series[i] is None:
                continue
            cand.append(float(rew[i]))
            ref.append(float(series[i]))
    if cand and ref:
        return cand, ref
    return _candidate_curve(report.result), _reference_curve(ctx, state, report)


def _canonical(x: np.ndarray) -> np.ndarray:
    """Centre and scale to unit L2 norm.

    This is EPIC's canonicalisation with the shaping terms dropped. The full
    definition subtracts the expected next-state and current-state potentials
    under the transition distribution, which needs a reward *model* callable on
    (s, a, s') plus a coverage distribution -- neither exists at this point in
    the pipeline, where a reward is a training run's worth of scalars. What
    survives is the equivalence class the metric actually quotients out here:
    positive affine rescaling."""
    if x.size == 0:
        return x
    c = x - x.mean()
    n = float(np.linalg.norm(c))
    return c / n if n > 1e-12 else c


def _screen_helper(name: str) -> Optional[Callable[..., Any]]:
    """Borrow a pseudometric from `bird/components/screens.py` if it has one.

    EPIC/STARC live in §2 as screens too (`verify.quality_screen: epic|starc`),
    and there should be exactly one implementation of each. Imported lazily and
    optionally: screens.py is a sibling registry module, so a module-level
    import would couple two independently-loaded modules, and a missing helper
    must degrade to the local version rather than break config validation."""
    try:
        from . import screens  # noqa: PLC0415 -- lazy on purpose, see docstring
    except Exception:  # noqa: BLE001
        return None
    fn = getattr(screens, name, None)
    return fn if callable(fn) else None


def _pseudometric(ctx: Any, state: Any, report: CandidateReport, kind: str) -> Optional[float]:
    cand, ref = _paired_rewards(ctx, state, report)
    # Pairwise-finite, never `_clean` per side: the samples are paired on the
    # same transition, and a one-sided drop shifts every later pair.
    x, y = _paired_finite(cand, ref)
    if x.size < 2:
        return None

    helper = _screen_helper(f"{kind}_distance")
    if helper is not None:
        try:
            d = _num(helper(x, y))
            if d is not None:
                return d
        except Exception:  # noqa: BLE001 -- a screen's signature is not ours to assume
            log.debug("screens.%s_distance was unusable; using the local version", kind)

    cx, cy = _canonical(x), _canonical(y)
    if kind == "epic":
        # EPIC distance = sqrt((1 - rho) / 2) over canonicalised rewards.
        rho = _pearson(cx, cy)
        if rho is None:
            return None
        return float(math.sqrt(max(0.0, (1.0 - rho) / 2.0)))
    # STARC: standardise, then half the Euclidean distance between the unit
    # vectors -- the same quotient, measured as an angle rather than a
    # correlation, which is what makes it a bounded pseudometric.
    return float(np.linalg.norm(cx - cy) / 2.0)


@register("similarity", "epic")
def similarity_epic(ctx: Any, state: Any, report: CandidateReport) -> Optional[float]:
    """EPIC-style distance between the candidate reward and the reference,
    returned as a **similarity** in [0, 1] (1 = equivalent).

    Published by none of the methods here: EPIC/STARC are the obvious
    unexplored members of the §2 screen family, a label-free prefilter that
    avoids both TAC's cold start and TPE's fail-open. This entry
    is its §4 twin -- the same number, reported instead of gating.

    Direction matters: `evaluate.similarity.role: select` promotes this into
    `report.fitness`, and every §5 rule maximises, so the registered value is
    `1 - distance`. The raw pseudometric is kept at `meta["epic_distance"]`.

    The local implementation is the *degraded* EPIC (see `_canonical`): with a
    reward available only as sampled scalars, the shaping terms are not
    computable and it reduces to the Pearson distance between canonicalised
    per-step rewards. A faithful EPIC needs `ctx.env` to expose a transition
    distribution, which no adapter in this repo does yet."""
    d = _pseudometric(ctx, state, report, "epic")
    sim = None if d is None else max(0.0, 1.0 - d)
    return _promote(ctx, report, "epic", sim, {"epic_distance": d})


@register("similarity", "starc")
def similarity_starc(ctx: Any, state: Any, report: CandidateReport) -> Optional[float]:
    """STARC-style distance, returned as a similarity in [0, 1] (1 = equivalent).

    Same standing as `epic` -- an unpublished and obvious point.
    STARC differs from EPIC in the canonicalisation (it quotients potential
    shaping explicitly and then normalises), which makes it a true pseudometric
    with bounds on regret; the same data limitation applies here, so the local
    version quotients only the affine class. Raw distance at
    `meta["starc_distance"]`."""
    d = _pseudometric(ctx, state, report, "starc")
    sim = None if d is None else max(0.0, 1.0 - d)
    return _promote(ctx, report, "starc", sim, {"starc_distance": d})
