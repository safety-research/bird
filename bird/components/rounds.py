"""Stage 3 in RUNGS -- `train.pruning: successive_halving_pool`.

NOT A PUBLISHED MECHANISM. An efficiency variant introduced by BIRD.

Every rule in `training._PRUNERS` is WITHIN one training: it reads that
training's own curve and stops it. `successive_halving` there is the run
against itself, because inside one training there is no population. This
module is the population version. One iteration's candidates are trained in
rungs -- everyone for `rung_fraction` x `train.env_steps`, then the pool is
ranked on `train.pruning_metric` at that checkpoint, the bottom `1 - 1/eta` are
dropped, and the survivors are trained `eta` x as far again from their OWN
stored policy (`_POLICY_STORE["policy:<cand_id>"]`, written by the backend at
the end of every rung) -- until the cumulative slice reaches the budget. The
CAP ends the rungs, not a survivor count: at K=8 the third rung is capped at
0.25 and TWO survivors reach the full budget.

Why (measured with the hill-climb's v4 configuration on MT10): the
within-training plateau rule cut every training at ~0.47M of 1.03M steps, and
in every dead seed no trained candidate reached task success. The saving was
real and it was spent on the wrong thing -- the same ~0.5M on each of K
candidates, most of which were never going to learn. Halving spends the same
round budget on the candidates that ARE learning: for K=8 at eta 2 and the
default `rung_fraction` 0.25 the round costs 8 x 0.25 + 4 x 0.5 + 2 x 0.25
(capped) = 4.5 budgets ~ 0.56 per candidate, and the two last survivors get the
full 1.03M that the plateau rule never granted anyone.
(`tests/test_halving_pool.py` pins the schedule for K=8/6/4/2.)

The ranking metric has to be comparable ACROSS candidates, and a candidate's own
return is not (each reward pays in its own units). `demo_fraction` is: the share
of the random->expert gap the training has closed under its own reward, the
expert and random returns being the demonstration rollouts `bird.demos` makes
for the screen (`training._demo_ceiling`). A candidate whose ceiling is
UNINFORMATIVE -- the reward pays the expert no more than random, the demo
screen's sign-test failure -- is ranked on its own-range progress, as the
curve field already holds, and in a LOWER TIER than every informative one:
a reward under which solving the task is not worth more than flailing has not
earned a comparison with rewards under which it is. Both facts are in the
`train_halving_rung` event and in `candidate.meta["ceiling"]["status"]`.
UNAVAILABLE is different: an env with no demonstration policy at all
(`bird.demos.policy_set` finds no expert) gives every candidate the own-range
reading and puts them all in tier 0, where ties break on index -- each rung
would keep the lowest-index half and the run would read as a ranking. So
`demo_fraction` REFUSES before rung 0 on such an env, where the ceiling GUARD
(`pruning_cfg.ceiling`) falls open to the plain rule and journals it: a guard
that cannot fire is the unguarded rule, a metric that cannot rank is not one.

HOW A RUNG CONTINUES, and why nothing new was added for it. A later rung is
the backend's `resume_ref` -- the same argument `train.interaction:
shared_population` (LaRes) hands a candidate to continue its previous slice by
-- and runs under `Budget.slice_mode()`, the same latch, so the rung's steps
and time are charged and its `record_training` lands in `training_slices`
rather than `policy_trainings`. `budget.max_policy_trainings` therefore keeps
counting CANDIDATES trained (24 under an [8,6,4,2,2,2] candidate taper), a rung
is not a new candidate, and a nonzero `training_slices` on a config with
`train.interaction: independent` says the curves in that run are stitched
across rung boundaries. `_check_coherence` refuses the two drivers together:
a round has one.

How this stays bit-identical between `train.candidate_parallelism: sequential`
and `parallel` (the bar `tests/test_parallelism.py` holds every schedule to):
each rung is ONE call of the configured round pass over the survivors, with
the same per-candidate arguments whichever schedule runs it, and every
decision here is a pure function of the rung's returned results in
candidate-index order. Ties break on index. The schedule's own
reassembly-by-index and store merge (`_merge_child` publishes
`policy:<cand_id>` before the next rung forks) are what carry a survivor's
policy across the rung boundary.

Two honest limits. A rung continuation re-seeds the learner from
`_seed_base` (no rung term), so exploration noise repeats its sequence from
a new starting policy -- deterministic, mildly correlated across rungs, and
the artifact's `rungs` field is how a reader knows the curve is stitched. And
"continue from the stored policy" restores what the store holds -- weights, and
the optimiser state SB3 saves with them; a surrogate learner's exploration
schedule restarts (`_CEMLearner.sigma`), and PPO's rollout buffer is rebuilt.
A stitched training is therefore not the training an uninterrupted one would
have been; it is the training this method performs, and it is recorded as such
(`seed_metrics[*].rungs`, `rung_train_steps`, `Budget.training_slices`).

This module runs in the PARENT only -- it calls the round pass, it never runs
inside a worker -- which is why it may journal through `ctx.event` where
`training.py` may not (`test_the_train_stage_touches_only_cfg_env_budget`).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..types import Candidate, TrainResult
from .training import _PRUNING_FIELDS

log = logging.getLogger(__name__)

RunOne = Callable[..., TrainResult]
OnResult = Callable[[Candidate, TrainResult], None]
RoundPass = Callable[..., List[TrainResult]]

__all__ = ["training_rounds"]


def training_rounds(ctx: Any, state: Any, candidates: Sequence[Candidate],
                    run_one: RunOne, on_result: OnResult, round_pass: RoundPass,
                    ) -> List[TrainResult]:
    """Stage 3's one entry point: the round driver `train.pruning` selects.

    `round_pass(ctx, state, cands, run_fn, on_res)` is one pass of the round
    over `cands` -- in `bird.py` the `train.interaction` component over the
    `train.candidate_parallelism` schedule. Every rule but
    `successive_halving_pool` is a single pass, so the default path is one
    call of the round pass and nothing else.
    """
    rule = str(ctx.cfg.get("train.pruning", "none") or "none")
    metric = str(ctx.cfg.get("train.pruning_metric", "own_reward") or "own_reward")
    if rule != "none" and metric == "demo_fraction":
        # Any rule, not only the pool: a within-training rule watching
        # `demo_fraction` on an env with no expert would watch own-range
        # progress under that name (`_demo_fraction_point`'s fallback is for
        # UNINFORMATIVE ceilings). Refused here, in the parent, before the
        # first training of the run.
        _require_demonstrations(ctx)
    driver = _ROUND_DRIVERS.get(rule, _one_round)
    return driver(ctx, state, candidates, run_one, on_result, round_pass)


def _one_round(ctx: Any, state: Any, candidates: Sequence[Candidate], run_one: RunOne,
               on_result: OnResult, round_pass: RoundPass) -> List[TrainResult]:
    return round_pass(ctx, state, candidates, run_one, on_result)


def _noop(candidate: Candidate, result: TrainResult) -> None:  # noqa: ARG001
    """Per-rung `on_result`: the journal gets ONE `train` event per candidate,
    for the merged result, after its last rung -- not one per rung."""


def _require_demonstrations(ctx: Any) -> None:
    """`pruning_metric: demo_fraction` needs an expert to measure against.

    Refused by `training_rounds`, before anything trains and under every rule
    that watches `demo_fraction`, on the same lookup the
    backend's `_demo_ceiling` makes per candidate (`demos.policy_set`), so
    the answer is the one the ceilings would have carried. `_check_coherence`
    cannot see the demonstration catalogue -- the env decides -- which is why
    this is a runtime refusal and not a config one.
    """
    from .. import demos
    from ..config import ConfigError
    env_id = str(ctx.cfg.get("problem.env_id") or "")
    pols, why = demos.policy_set(ctx.env, env_id, ["expert", "random"])
    if not pols:
        raise ConfigError(
            f"train.pruning_metric=demo_fraction ranks candidates on the share of the "
            f"random->expert gap closed, and {env_id!r} has no demonstration policy "
            f"({why}); every candidate would fall to own-range progress and the rungs "
            f"would keep the lowest-index half -- use a metric the env can supply, or "
            f"an env with an expert (policies/, Meta-World's scripted policy, or "
            f"EnvAdapter.expert_policy)")


def _rank_key(parts: Sequence[TrainResult], field: str, candidate: Candidate,
              ) -> Tuple[int, float, str]:
    """`(tier, score, basis)` for one candidate after its latest rung.

    `score` is the LAST checkpoint's `field` over the stitched curve. Under
    `demo_fraction` the tier separates informative ceilings (1) from
    uninformative or unavailable ones (0); any other field is one tier. A
    non-finite score is tier -1: a curve that went NaN ranks below everything.
    """
    last: Optional[float] = None
    for part in reversed(parts):
        if part.checkpoints:
            last = part.checkpoints[-1].get(field)
            break
    if last is None or not np.isfinite(float(last)):
        return -1, float("-inf"), "non_finite"
    if field == "demo_fraction":
        ceiling = candidate.meta.get("ceiling") or {}
        if ceiling.get("informative"):
            return 1, float(last), "demo_fraction"
        return 0, float(last), f"own_range (ceiling {ceiling.get('status', 'unavailable')})"
    return 1, float(last), field


def _successive_halving_pool(ctx: Any, state: Any, candidates: Sequence[Candidate],
                             run_one: RunOne, on_result: OnResult, round_pass: RoundPass,
                             ) -> List[TrainResult]:
    cfg = ctx.cfg
    total = int(cfg.get("train.env_steps", 20000) or 20000)
    frac = float(cfg.get("train.pruning_cfg.rung_fraction", 0.25) or 0.25)
    eta = max(2, int(cfg.get("train.pruning_cfg.eta", 2) or 2))
    metric = str(cfg.get("train.pruning_metric", "own_reward") or "own_reward")
    field = _PRUNING_FIELDS[metric]
    first = max(1, int(round(frac * total)))

    n = len(candidates)
    parts: List[List[TrainResult]] = [[] for _ in range(n)]
    skipped: Dict[int, TrainResult] = {}
    dropped_at: Dict[int, int] = {}
    alive: List[int] = list(range(n))  # rung 0 is everyone; the schedule skips `not trainable`
    cum, rung, slice_ = 0, 0, first

    while alive:
        steps = min(slice_, total - cum)
        if steps <= 0:
            break
        continuing = rung > 0

        def run_rung(c: Candidate, _steps: int = steps, _cont: bool = continuing) -> TrainResult:
            kw: Dict[str, Any] = {"env_steps": _steps}
            if _cont:
                kw["resume_ref"] = f"policy:{c.cand_id}"
            return run_one(c, **kw)

        batch = [candidates[i] for i in alive]
        if continuing:
            # A later rung continues a training already counted: steps and time
            # are spend, the training is not (`Budget.slice_mode`, the LaRes
            # latch). Set in the parent before the pass forks, so a child's
            # budget delta lands in the same column.
            with ctx.budget.slice_mode():
                outs = round_pass(ctx, state, batch, run_rung, _noop)
        else:
            outs = round_pass(ctx, state, batch, run_rung, _noop)
        assert len(outs) == len(batch), "a schedule returned a result count that is not its input's"
        cum += steps

        ok: List[int] = []
        for i, res in zip(alive, outs):
            if not candidates[i].trainable:
                skipped[i] = res  # `_skip_result`, already counted by the schedule
                continue
            parts[i].append(res)
            if res.trained and not res.error:
                ok.append(i)
        keys = {i: _rank_key(parts[i], field, candidates[i]) for i in ok}
        # tier desc, score desc, then candidate INDEX asc -- never completion order
        order = sorted(ok, key=lambda i: (-keys[i][0], -keys[i][1], i))
        finished = cum >= total
        if finished:
            keep: List[int] = []
        elif len(order) <= 1:
            keep = list(order)  # the last survivor trains on to the budget
        else:
            keep = order[:max(1, len(order) // eta)]
        dropped = [] if finished else [i for i in order if i not in keep]
        for i in dropped:
            dropped_at[i] = rung
        ctx.event("train_halving_rung", rung=rung, env_steps_each=int(steps),
                  cumulative_steps=int(cum), budget=int(total), n_trained=len(batch),
                  n_ok=len(ok), n_failed=len(batch) - len(ok) - sum(1 for i in alive if i in skipped),
                  metric=metric, eta=eta, finished=finished,
                  kept=[candidates[i].cand_id for i in keep],
                  dropped=[candidates[i].cand_id for i in dropped],
                  ranking=[{"cand_id": candidates[i].cand_id, "tier": keys[i][0],
                            "score": keys[i][1], "basis": keys[i][2]} for i in order])
        log.info("  [3] halving rung %d: %d trained for %d steps (cum %d/%d) -> keep %d, drop %d",
                 rung, len(batch), steps, cum, total, len(keep), len(dropped))
        alive = keep
        rung += 1
        slice_ *= eta

    results: List[TrainResult] = []
    for i, c in enumerate(candidates):
        if i in skipped:
            results.append(skipped[i])
            continue
        merged = _merge_rungs(parts[i], total, dropped=i in dropped_at)
        results.append(merged)
        on_result(c, merged)
    return results


_ROUND_DRIVERS: Dict[str, Callable[..., List[TrainResult]]] = {
    "successive_halving_pool": _successive_halving_pool,
}


# --------------------------------------------------------------------------
# Stitching one candidate's rungs into the TrainResult stage 4 reads
# --------------------------------------------------------------------------


def _shift(points: Sequence[Dict[str, float]], step_off: float, round_off: float,
           ) -> List[Dict[str, float]]:
    out = []
    for pt in points:
        q = dict(pt)
        q["step"] = float(pt.get("step", 0.0)) + step_off
        q["round"] = float(pt.get("round", 0.0)) + round_off
        out.append(q)
    return out


def _shift_round(value: Any, round_off: float) -> Any:
    """A seed row's checkpoint ROUND (`restored_checkpoint`, `shipped_checkpoint`,
    written by `training._select_checkpoint`'s callers relative to the rung's own
    curve) re-indexed onto the stitched curve. Laid end to end unshifted those
    would name the wrong checkpoint of the merged curve -- an off-by-a-rung that
    reads as a plausible row. Worked example: rungs of 2 and 3 checkpoints, the
    last rung restored its round 1 -> merged round 3 of 5. None stays None."""
    if value is None:
        return None
    return int(int(value) + int(round_off))


def _merge_rungs(parts: Sequence[TrainResult], total_requested: int, *, dropped: bool,
                 ) -> TrainResult:
    """One `TrainResult` whose curve is the rungs laid end to end.

    Steps and rounds are offset by what the earlier rungs spent, so the curve is
    monotone in `step` and §4's aggregations read it as one training. Counters
    add; the final rollouts, policy ref and replay ref are the LAST rung's (the
    policy that exists). `pruned` is set when the pool dropped the candidate --
    the third outcome `TrainResult.pruned` documents, a lower bound rather than
    a finished training -- with `pruned_at_round` at the stitched checkpoint
    count. `train_steps_requested` stays the whole budget: against
    `train_steps` it is where the saving is read.
    """
    assert parts, "a trained candidate has at least its rung-0 result"
    if len(parts) == 1 and not dropped:
        return parts[0]
    last = parts[-1]
    merged = TrainResult(cand_id=last.cand_id, candidate=last.candidate)
    merged.trained = all(p.trained for p in parts)
    merged.error = next((p.error for p in parts if p.error), "")
    merged.skip_reason = parts[0].skip_reason
    merged.timed_out = any(p.timed_out for p in parts)
    merged.pruned = bool(dropped or any(p.pruned for p in parts))
    merged.trajectories = list(last.trajectories)
    merged.store_trajectories = list(last.store_trajectories)
    merged.policy_ref = last.policy_ref
    merged.replay_ref = last.replay_ref
    # Where the training STARTED is rung 0's answer (`train.init` resolved once;
    # every later rung is a `continuation` of this candidate's own policy).
    merged.init_source = parts[0].init_source
    merged.init_from_cand_id = parts[0].init_from_cand_id
    merged.init_similarity = parts[0].init_similarity
    merged.wallclock_s = float(sum(p.wallclock_s for p in parts))
    merged.env_steps_used = int(sum(p.env_steps_used for p in parts))

    step_off = 0.0
    round_off = 0.0
    comps: Dict[str, List[float]] = {}
    for p in parts:
        merged.checkpoints.extend(_shift(p.checkpoints, step_off, round_off))
        merged.gt_reward_curve.extend(p.gt_reward_curve)
        for k, v in p.component_traces.items():
            comps.setdefault(k, []).extend(v)
        # what this rung spent LEARNING, summed over its seed rows (one, by coherence)
        step_off += float(sum(int(sm.get("train_steps", 0)) for sm in p.seed_metrics))
        round_off += float(len(p.checkpoints))
    merged.component_traces = comps

    # Seed rows: one merged row per seed index, the last rung's row as the base
    # (it describes the policy that exists) with the additive fields summed and
    # the per-seed curve stitched the same way.
    n_rows = min(len(p.seed_metrics) for p in parts) if all(p.seed_metrics for p in parts) else 0
    for s in range(n_rows):
        rows = [p.seed_metrics[s] for p in parts]
        row = dict(rows[-1])
        row["env_steps"] = int(sum(int(r.get("env_steps", 0)) for r in rows))
        row["train_steps"] = int(sum(int(r.get("train_steps", 0)) for r in rows))
        row["train_steps_requested"] = int(total_requested)
        row["wallclock_s"] = float(sum(float(r.get("wallclock_s", 0.0)) for r in rows))
        row["n_checkpoints"] = int(sum(int(r.get("n_checkpoints", 0)) for r in rows))
        maxes = [float(r["max"]) for r in rows if "max" in r]
        row["max"] = float(max(maxes)) if maxes else row.get("max", 0.0)
        weights = [max(1, int(r.get("n_checkpoints", 0))) for r in rows]
        aucs = [float(r.get("auc", 0.0)) for r in rows]
        row["auc"] = float(np.average(aucs, weights=weights)) if aucs else 0.0
        curve: List[Dict[str, float]] = []
        so, ro = 0.0, 0.0
        for r in rows:
            pts = r.get("checkpoints") or []
            curve.extend(_shift(pts, so, ro))
            so += float(int(r.get("train_steps", 0)))
            ro += float(len(pts))
        row["checkpoints"] = curve
        row["init"] = rows[0].get("init", row.get("init"))
        row["warm_started_from"] = rows[0].get("warm_started_from", "")
        # `train.checkpoint_selection` as executed on the LAST rung -- the
        # policy that exists -- with its rounds moved onto the stitched curve.
        last_ro = ro - float(len(rows[-1].get("checkpoints") or []))
        for key in ("restored_checkpoint", "shipped_checkpoint"):
            if key in rows[-1]:
                row[key] = _shift_round(rows[-1].get(key), last_ro)
        row["rungs"] = len(rows)
        row["rung_train_steps"] = [int(r.get("train_steps", 0)) for r in rows]
        row["rung_warm_started_from"] = [r.get("warm_started_from", "") for r in rows]
        row["halving_dropped"] = bool(dropped)
        row["pruned_at_round"] = (int(row["n_checkpoints"]) if dropped
                                  else rows[-1].get("pruned_at_round"))
        decisions = [d for r in rows for d in (r.get("ceiling_decisions") or [])]
        if any("ceiling_decisions" in r for r in rows):
            row["ceiling_decisions"] = decisions
        merged.seed_metrics.append(row)
    return merged
