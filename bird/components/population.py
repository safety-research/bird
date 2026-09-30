"""Joint population training -- LaRes (Li et al., NeurIPS 2025).

Four registry families, all of them §3, all of them degenerate-by-default so
that every config written before LaRes is bit-identical under them:

    interaction            train.interaction              independent | shared_population
    interaction_allocator  train.interaction_allocator    uniform | thompson_success
    reward_scaling         train.reward_scaling           none | elite_moments
    elite_constraint       train.elite_constraint.kind    none | l2_params

**Why a new family rather than another `candidate_parallelism` entry.** That
family chooses only the SCHEDULE and is held bit-identical to `sequential` by
`tests/test_parallelism.py`, because a config that moved a number by changing
its schedule would be changing the method. `shared_population` changes the
work: candidates stop being independent. Putting it in that family would
either break the determinism test or quietly weaken what it asserts, and both
are worse than one more key.

**What LaRes actually does, and where this is coarser.** LaRes reallocates at
the granularity of ONE EPISODE: it draws an arm, runs one episode, updates the
Beta posterior from that episode's success, and repeats until the step budget
is gone (`refs/code/LaRes/LaRes_from_scratch.py:799-827`). A BIRD backend call
carries fixed set-up and a checkpoint evaluation, so per-episode calls would be
dominated by overhead. The slice here is therefore
`train.env_steps / interaction_cfg.slices_per_candidate` and the posterior is
updated once per slice from that slice's final evaluation.
`slices_per_candidate: 1` collapses the whole mechanism to a single allocation
pass, which is a useful control and not the method.

**The buffer is pooled too** (`train.interaction_cfg.shared_buffer`, §4.3).
Without it every slice would build a fresh SAC on an EMPTY replay buffer, so
the method that ran would be "SAC restarted five times per candidate",
`shared_buffer: false` would be bit-identical to `true`, and every counter
would read normal. `_RoundBuffers` below keeps the round's raw transitions --
one ring under `true` (the release's single `All_buffer`,
`LaRes_from_scratch.py:533`, into which every agent's episode is written and
from which every agent samples uniformly under its own reward column,
`replay_buffer.py:39-42`), one per arm under `false` --
and each slice's learner is pre-filled with the rows it may see, relabelled
under its own reward, before it learns (`training._sb3_prefill_replay`). The
pool is the whole history: it starts from the carried `state.replay_ref` and
becomes the carried buffer at round close. The seed is salted by the slice
index (`training._seed_for`). What is still coarser than the release is the
GRANULARITY: the release appends to the buffer every episode and every agent
trains from it every round; here the pool grows once per slice, and the arm
sees other arms' rows collected since its previous slice only at its next one.

**The budget is pooled, not enlarged.** A round's budget is
`train.env_steps x (trainable candidates)` under both modes, and the wave loop
runs until it is spent. So `shared_population` + `uniform` gives every
candidate exactly `train.env_steps`, identical to `independent`, and the only
thing `thompson_success` changes is who gets which share. That is what makes
the pair the paper's own Fig. 6a ablation ("removing Thompson sampling leads to
a performance decline") rather than a comparison at two different budgets, and
it is what `tests/test_population.py` pins.

**The allocator is a ground-truth channel.** `thompson_success` counts the
ENVIRONMENT's success flag, not the designed reward -- `TS.update(worker_idx,
float(success))` at `LaRes_from_scratch.py:827`. So a config using it has GT
access inside the inner loop, not merely at the selection boundary, and a
GT-free config must leave it `uniform`. `config._check_coherence` refuses the
combination rather than letting it leak.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .. import registry
from ..registry import register
from ..state import RunState
from ..types import Candidate, TrainResult

log = logging.getLogger("bird.population")


# ==========================================================================
# interaction -- fn(ctx, state, candidates, run_one, on_result, schedule,
#                   backend) -> List[TrainResult]
#
# Called from STAGE 3 in place of the bare `schedule(...)` call. `independent`
# IS that bare call, so nothing about any pre-LaRes config changes.
# ==========================================================================


@register("interaction", "independent")
def interaction_independent(ctx: Any, state: RunState, candidates: List[Candidate],
                            run_one: Callable[[Candidate], TrainResult],
                            on_result: Callable[[Candidate, TrainResult], None],
                            schedule: Callable[..., List[TrainResult]],
                            backend: Callable[..., TrainResult]) -> List[TrainResult]:
    """Every candidate trains alone for `train.env_steps`. Every method but LaRes.

    A tail call, deliberately: this path must stay the code that ran before the
    family existed, so that adding the key cannot have moved a number.
    """
    return schedule(ctx, state, candidates, run_one, on_result)


def _slice_success(res: TrainResult) -> float:
    """The environment's own success on this slice's last checkpoint.

    LaRes counts an interaction a success on the env's flag, never on the
    designed reward (`LaRes_from_scratch.py:827`, `TS.update(..., float(success))`).
    Three keys are tried because the backends do not agree on one name and a
    silent 0.0 would make every arm look equally bad -- i.e. would turn the
    allocator into `uniform` while still reporting `thompson_success`.

    The native-signal rule (N): the keys are `NATIVE_SUCCESS_KEYS` -- genuine
    env success-flag names only. `task_success`/`fitness` are deliberately not
    tried: they are ALIASES of the BIRD `task_metric` (custom_metric) in the
    curve rows, so thresholding them would let `thompson_success` steer the
    inner loop by the number we wrote rather than the env's own flag. Both
    backends always emit `success_rate` in every checkpoint row, so that key
    wins in practice.
    """
    if not res.checkpoints:
        return 0.0
    last = res.checkpoints[-1]
    from ..native_signal import NATIVE_SUCCESS_KEYS
    for key in NATIVE_SUCCESS_KEYS:
        if key in last:
            try:
                return float(last[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _merge_slices(slices: List[TrainResult]) -> TrainResult:
    """One candidate's several backend calls, back into the one training they were.

    An arm trained across K slices is ONE run of ONE seed lineage: its
    checkpoint history is its slices' curves concatenated in slice order, and
    the terminal facts -- the policy, the rollouts §4 will read, the last
    slice's row fields -- describe the policy that actually leaves the round.
    Summing the curve or averaging over slices would both report a policy that
    never existed.

    THE SHAPE MATTERS. Each slice's checkpoints are stamped with that slice's
    own salted seed (`training._seed_for`). Keeping the LAST slice's seed row
    (a one-checkpoint curve) while pooling all K slices' checkpoints as they
    were would be read two wrong ways: Stage 4 groups a pooled curve by its
    `seed` field, so it would read K slices as K SEEDS -- `n_seeds: 5`, a
    fitness aggregated over slices most of which scored 0 -- and a reader that
    trusts the row would see one checkpoint. Neither is what an arm means. So:

      * the arm has ONE seed row: the last slice's, with `checkpoints` = the
        slices' curves concatenated in slice order on ONE cumulative axis
        (`round` and `step` offset by the slices before; the slice's own
        values kept as `slice_round` / `slice_step`), every checkpoint stamped
        `slice_index` and `slice_seed`, `seed` = the lineage seed (the first
        seeded slice's), `slice_seeds` listing every slice's, and the spend
        fields (`train_steps`, `env_steps`, `n_checkpoints`, `wallclock_s`,
        `train_steps_requested`) summed over slices;
      * the pooled `result.checkpoints` carries the same entries with the same
        single `seed`, so no reader can mistake slices for seeds;
      * `shipped_checkpoint` / `restored_checkpoint` are re-expressed on the
        cumulative `round` axis so a lookup by round still finds the last
        slice's row and not an earlier slice's.

    The per-slice salt stays what it is -- a training fact recorded on the
    slice -- and is not a seed identity for evaluation.
    """
    head = slices[-1]
    out = TrainResult(cand_id=head.cand_id, candidate=head.candidate)
    out.trained = all(s.trained for s in slices)
    out.skip_reason = head.skip_reason
    out.component_traces = head.component_traces
    out.trajectories = head.trajectories
    out.store_trajectories = head.store_trajectories
    out.policy_ref = head.policy_ref
    out.replay_ref = head.replay_ref
    for s in slices:
        out.gt_reward_curve.extend(s.gt_reward_curve)
        out.env_steps_used += int(s.env_steps_used)
        out.wallclock_s += float(s.wallclock_s)
    out.pruned = any(s.pruned for s in slices)
    out.timed_out = any(getattr(s, "timed_out", False) for s in slices)
    errs = [s.error for s in slices if s.error]
    out.error = errs[0] if errs else ""

    rows = [(s.seed_metrics[-1] if s.seed_metrics and isinstance(s.seed_metrics[-1], dict)
             else None) for s in slices]
    if all(r is None for r in rows):
        # No slice produced a seed row (nothing compiled): the pooled curves,
        # in slice order, are all there is.
        for s in slices:
            out.checkpoints.extend(s.checkpoints)
        out.seed_metrics = list(head.seed_metrics)
        return out

    slice_seeds = [r.get("seed") if r is not None else None for r in rows]
    lineage = next((v for v in slice_seeds if v is not None), None)
    curve: List[Dict[str, Any]] = []
    round_offset = 0.0
    step_offset = 0.0
    head_round_offset = 0.0
    for k, (s, r) in enumerate(zip(slices, rows)):
        pts = [d for d in ((r or {}).get("checkpoints") or s.checkpoints) if isinstance(d, dict)]
        if k == len(slices) - 1:
            head_round_offset = round_offset
        max_round = 0.0
        max_step = 0.0
        for d in pts:
            e = dict(d)
            e["slice_index"] = k
            e["slice_seed"] = slice_seeds[k] if slice_seeds[k] is not None else d.get("seed")
            if d.get("round") is not None:
                e["slice_round"] = d["round"]
                e["round"] = round_offset + float(d["round"])
                max_round = max(max_round, float(d["round"]))
            if d.get("step") is not None:
                e["slice_step"] = d["step"]
                e["step"] = step_offset + float(d["step"])
                max_step = max(max_step, float(d["step"]))
            if lineage is not None:
                e["seed"] = lineage
            else:
                e.pop("seed", None)
            curve.append(e)
        round_offset += max_round
        # The slice's LEARNING steps move the axis, not its last checkpoint's
        # `step` alone -- the two agree on every backend that checkpoints at the
        # end, and `train_steps` is the recorded fact either way.
        learned = (r or {}).get("train_steps")
        step_offset += float(learned) if learned is not None else max_step

    out.checkpoints = [dict(e) for e in curve]
    merged = dict(rows[-1] if rows[-1] is not None else {})
    merged["checkpoints"] = curve
    merged["seed"] = lineage
    merged["slice_seeds"] = slice_seeds
    merged["n_slices"] = len(slices)
    for key in ("train_steps", "env_steps", "n_checkpoints", "train_steps_requested"):
        vals = [(r or {}).get(key) for r in rows]
        if any(v is not None for v in vals):
            merged[key] = sum(int(v or 0) for v in vals)
    walls = [(r or {}).get("wallclock_s") for r in rows]
    if any(v is not None for v in walls):
        merged["wallclock_s"] = float(sum(float(v or 0.0) for v in walls))
    for key in ("shipped_checkpoint", "restored_checkpoint"):
        v = merged.get(key)
        if v is not None:
            merged[key] = float(v) + head_round_offset
    out.seed_metrics = [merged]
    return out


class _RoundBuffers:
    """The round's RAW transitions, grouped the way `shared_buffer` says.

    `shared=True` is LaRes §4.3's one buffer: every arm's rows in arrival order
    (waves ascending; a carried history first), of which the NEWEST `capacity`
    are the ring -- the release's `replay_buffer.add` overwrites oldest-first
    at `--buffer-size` (`refs/code/LaRes/replay_buffer.py:12-19`,
    `arguments.py:19`). `shared=False` is one such ring PER ARM, i.e. an agent
    that keeps its own buffer between slices and never sees another's -- the
    ablation, and the reason the key can move a number at all.

    Rows are stored as the chunks the slices exported (`_ReplaySlice`, four
    columns, no reward) and concatenated only when a wave needs a prefill, so
    the parent holds at most the ring plus one materialised copy. `trim` drops
    whole chunks that fell out of their group's ring so the round's memory is
    bounded by `capacity` per group, not by the round's pooled steps.
    """

    def __init__(self, shared: bool, capacity: int, history: Any = None) -> None:
        self.shared = bool(shared)
        self.capacity = max(1, int(capacity))
        #: (wave, arm or None for carried history, rows) in arrival order
        self.chunks: List[Tuple[int, Optional[int], Any]] = []
        self.history_rows = 0
        if self.shared and history is not None and len(history) > 0:
            # `loop.carry: [replay_buffer]` under LaRes means the pool
            # persists across evolutions (the release relabels the SAME
            # `All_buffer` at every LLM round, `:732-762`); an ablation that
            # drops the carry starts every round's pool empty.
            self.chunks.append((-1, None, history))
            self.history_rows = int(len(history))

    def add(self, wave: int, arm: int, rows: Any) -> None:
        if rows is None or len(rows) == 0:
            return
        self.chunks.append((int(wave), int(arm), rows))
        self.trim()

    def _group(self, arm: Optional[int]) -> List[Tuple[int, Optional[int], Any]]:
        if self.shared:
            return list(self.chunks)
        return [c for c in self.chunks if c[1] == arm]

    def trim(self) -> None:
        """Drop chunks entirely outside their group's newest-`capacity` rows."""
        keep: List[Tuple[int, Optional[int], Any]] = []
        groups = [None] if self.shared else sorted({c[1] for c in self.chunks
                                                    if c[1] is not None})
        alive = set()
        for g in groups:
            acc = 0
            for idx in range(len(self.chunks) - 1, -1, -1):
                c = self.chunks[idx]
                if not self.shared and c[1] != g:
                    continue
                if acc >= self.capacity:
                    continue
                alive.add(idx)
                acc += len(c[2])
        for idx, c in enumerate(self.chunks):
            if idx in alive:
                keep.append(c)
        self.chunks = keep

    def materialise(self, arm: Optional[int]) -> Tuple[Optional[Any], Dict[int, int]]:
        """The ring one arm may see, as one `_ReplaySlice`, plus own-row counts.

        Returns `(rows, own)` where `own[a]` is how many of `rows` arm `a`
        collected itself -- what the seed row's `replay_prefill_own` /
        `_pooled` split reports. Under `shared` the rows are the same for
        every arm and only `own` differs, so the driver calls this once per
        wave and reads `own[arm]` per arm.
        """
        from .training import _ReplaySlice, _replay_columns  # local: no import cycle

        sel = self._group(arm)
        if not sel:
            return None, {}
        cols = []
        owners: List[Tuple[Optional[int], int]] = []
        for _w, a, rows in sel:
            c = _replay_columns(rows)
            if c is None or c[0].shape[0] == 0:
                continue
            cols.append(c)
            owners.append((a, int(c[0].shape[0])))
        if not cols:
            return None, {}
        if len(cols) > 1:
            try:
                obs = np.concatenate([c[0] for c in cols], axis=0)
                act = np.concatenate([c[1] for c in cols], axis=0)
                nxt = np.concatenate([c[2] for c in cols], axis=0)
                done = np.concatenate([c[3] for c in cols], axis=0)
            except ValueError as exc:
                # A carried buffer from another env or action space cannot be
                # pooled with this round's rows; the round's own rows can.
                log.warning("      shared_population: could not pool %d buffer chunks (%s) "
                            "-- using this round's rows only", len(cols), exc)
                cols = [c for c, (a, _n) in zip(cols, owners) if a is not None]
                owners = [(a, n) for a, n in owners if a is not None]
                if not cols:
                    return None, {}
                obs = np.concatenate([c[0] for c in cols], axis=0)
                act = np.concatenate([c[1] for c in cols], axis=0)
                nxt = np.concatenate([c[2] for c in cols], axis=0)
                done = np.concatenate([c[3] for c in cols], axis=0)
        else:
            obs, act, nxt, done = cols[0]
        total = int(obs.shape[0])
        cut = max(0, total - self.capacity)
        own: Dict[int, int] = {}
        start = 0
        for a, n in owners:
            end = start + n
            kept = max(0, end - max(start, cut))
            if a is not None and kept:
                own[a] = own.get(a, 0) + kept
            start = end
        if cut:
            obs, act, nxt, done = obs[cut:], act[cut:], nxt[cut:], done[cut:]
        return _ReplaySlice(np.ascontiguousarray(obs), np.ascontiguousarray(act),
                            np.ascontiguousarray(nxt), np.ascontiguousarray(done)), own

    def rows(self) -> int:
        return sum(len(c[2]) for c in self.chunks)


def _consecutive_passes(picks: Sequence[int]) -> List[List[int]]:
    """Split one wave's draws into PASSES of distinct arms, first occurrences first.

    `[0, 0, 1]` -> `[[0, 1], [0]]`. Under `interaction_cfg.skip_duplicates:
    true` every wave is one pass; under `false` an arm drawn k times in a
    wave trains k CONSECUTIVE slices, each resuming the one before it -- the
    release re-runs the same individual sequentially and keeps every episode
    -- and never two slices from one resume point. Scheduling both copies in
    the same wave from the same `resume[arm]` would charge both in `issued`
    while `resume[arm]` kept only the last, so one slice's learning would be
    thrown away and still billed.
    """
    passes: List[List[int]] = []
    for a in picks:
        for pas in passes:
            if a not in pas:
                pas.append(a)
                break
        else:
            passes.append([a])
    return passes


@register("interaction", "shared_population")
def interaction_shared_population(ctx: Any, state: RunState, candidates: List[Candidate],
                                  run_one: Callable[[Candidate], TrainResult],
                                  on_result: Callable[[Candidate, TrainResult], None],
                                  schedule: Callable[..., List[TrainResult]],
                                  backend: Callable[..., TrainResult]) -> List[TrainResult]:
    """LaRes: one pooled interaction budget, divided online among the round.

    Waves run until the pool is spent. Each wave asks the allocator which arms
    interact; each selected arm trains one slice, RESUMING its own policy from
    its previous slice (`_training_init`'s `resume_ref`) and its REPLAY BUFFER
    from the round's rows it may see (`training.SliceHandoff`: the shared ring
    under `interaction_cfg.shared_buffer: true`, its own ring under `false`),
    at a seed salted by the slice index, and reports that slice's success back
    to the allocator. What the buffer did is on every `interaction_slice`
    event (`buffer_prefill_own`, `buffer_prefill_pooled`, `buffer_added`) and
    in the seed row (`replay_prefill_*`, `replay_added`,
    `learning_starts_effective`), so a run can show the mechanism executed --
    or, on a backend that does not wire it (mock, tabular), that it did not
    (`interaction_buffer_unsupported`, once per round).

    The WAVES are sequential by construction -- the allocator's next draw
    depends on this wave's outcomes -- but the arms WITHIN a wave (a pass, under
    `skip_duplicates: false`) are independent and go through
    `train.candidate_parallelism` exactly like an ordinary round, so `parallel`
    executes here and is bit-identical to `sequential`
    (`tests/test_parallelism.py`'s cross-section carries a `lares` entry;
    `tests/test_reward_scaling.py` pins it with Eq. 3's sample draw firing).
    There is deliberately NO coherence rule against `parallel` beside this
    mode -- `bird/config.py` says why: the `full` profile pins it, and a
    profile wins over a method config by design.
    """
    cfg = ctx.cfg
    trainable = [c for c in candidates if getattr(c, "trainable", True)]
    if len(trainable) < 2:
        # One arm is not a bandit and the pooled budget equals the per-candidate
        # one, so this is `independent` exactly. Say so rather than running a
        # one-armed wave loop that reads as if something was allocated.
        log.info("      interaction: shared_population with %d trainable candidate(s) "
                 "-- identical to independent, running that", len(trainable))
        ctx.event("interaction_degenerate", mode="shared_population",
                  n_trainable=len(trainable), reason="fewer than two arms")
        return schedule(ctx, state, candidates, run_one, on_result)

    per_cand = int(cfg["train.env_steps"])
    pooled = per_cand * len(trainable)
    n_slices = max(1, int(cfg["train.interaction_cfg.slices_per_candidate"]))
    slice_steps = max(1, per_cand // n_slices)

    make_alloc = registry.get("interaction_allocator", cfg["train.interaction_allocator"])
    alloc = make_alloc(ctx, len(trainable))

    # A slice shorter than one episode cannot be honoured: every backend here
    # steps whole episodes, so it will overshoot and the allocation stops
    # describing what was spent. Loud, because the symptom is a budget that
    # looks divided and is not -- and it is the same shape as the bug this
    # driver's `issued` counter exists to prevent.
    horizon = int(getattr(ctx.env, "horizon", 0) or 0)
    if horizon and slice_steps < horizon:
        log.warning("      train.interaction_cfg.slices_per_candidate=%d makes a slice "
                    "of %d steps, below this env's %d-step horizon -- the backend will "
                    "overshoot every slice and the allocation will not describe the "
                    "spend. Lower slices_per_candidate or raise train.env_steps.",
                    n_slices, slice_steps, horizon)
        ctx.event("interaction_slice_below_horizon", slice_steps=slice_steps,
                  horizon=horizon, slices_per_candidate=n_slices)

    log.info("      interaction: shared_population -- %s over %d arms, "
             "%d pooled steps in slices of %d",
             cfg["train.interaction_allocator"], len(trainable), pooled, slice_steps)
    ctx.event("interaction_open", mode="shared_population",
              allocator=cfg["train.interaction_allocator"], n_arms=len(trainable),
              pooled_env_steps=pooled, slice_steps=slice_steps)

    from .training import (_REPLAY_STORE, _REPLAY_STORE_MAX_BYTES, _ROUND_REPLAY,
                           SliceHandoff, _replay_capacity, _store,
                           _wants_replay)  # local: no import cycle

    shared_buffer = bool(cfg["train.interaction_cfg.shared_buffer"])
    capacity = _replay_capacity(cfg)
    carried = _REPLAY_STORE.get(getattr(state, "replay_ref", None) or "")
    buffers = _RoundBuffers(shared_buffer, capacity, history=carried)
    # A previous round's leftovers can only exist after a crash mid-round;
    # they must not be read as this round's rows.
    _ROUND_REPLAY.clear()
    if buffers.history_rows:
        log.info("      shared_population: pool starts from the carried buffer "
                 "(%d rows, %s)", buffers.history_rows, state.replay_ref)
    ctx.event("interaction_buffer_open", shared_buffer=shared_buffer,
              capacity=capacity, history_rows=buffers.history_rows,
              history_ref=(getattr(state, "replay_ref", None) or "")
              if buffers.history_rows else "")
    any_supported = False
    any_slice = False
    # The round's spend, DECOMPOSED. `env_steps_spent` alone cannot say
    # whether a gap against `env_steps_allocated` is the learner over-training
    # a slice (a real over-spend against the budget match) or the per-call
    # overhead every backend pays -- checkpoint evaluations and the §4
    # rollouts -- which a slice pays as often as a whole training does, so
    # `slices_per_candidate` slices pay it `slices_per_candidate` times. Both
    # are read off the seed rows the backend already writes (`train_steps`,
    # `env_steps` = train + eval) and off `env_steps_used` (+ rollouts), so no
    # backend changes for this to be exact.
    train_spent = 0
    eval_spent = 0
    rollout_spent = 0
    checkpoint_evals = 0

    by_arm: Dict[int, List[TrainResult]] = {i: [] for i in range(len(trainable))}
    resume: Dict[int, Optional[str]] = {i: None for i in range(len(trainable))}
    # `issued` counts the steps ALLOCATED, not the steps a backend reported
    # spending, and the loop is bounded on it. The distinction is the whole
    # correctness of the budget match: a backend may overshoot a small
    # `env_steps` (the mock rounds up to whole episodes; sb3 rounds to a
    # rollout), and bounding on actual spend would let one arm's overshoot
    # starve a later arm of its share -- silently, and differently on every
    # backend. Measured on the tester profile, bounding on spend makes
    # `shared_population` + `uniform` spend 4500 steps against `independent`'s
    # 8000, i.e. the "matched" comparison runs at 56% of the budget.
    issued = 0
    spent = 0
    wave = 0
    # A wave that selects nothing would spin forever; the allocators cannot do
    # that today (both always return at least one arm) and this is the guard
    # that keeps a future one from hanging a run instead of failing.
    empty_waves = 0

    # `slice_mode` wraps only the BACKEND CALLS, never the whole round, and a
    # candidate's training is charged on its FIRST slice rather than at the end.
    # Charging at the end would let `budget.max_policy_trainings` sit unexamined
    # for a whole round: every slice runs under `slice_mode`, so
    # `policy_trainings` would not move and `_check()` could not fire until the
    # waves were over -- a cap that bites a round late is a declared bound that
    # does not bound. Charging on first selection makes the check fire exactly
    # where `independent` fires it.
    while issued < pooled:
        picks = alloc.select()
        if not picks:
            empty_waves += 1
            if empty_waves > 3:
                log.warning("      allocator %s selected no arm in %d consecutive "
                            "waves; stopping the round at %d/%d steps",
                            cfg["train.interaction_allocator"], empty_waves,
                            spent, pooled)
                break
            continue
        empty_waves = 0
        # A PASS's arms are distinct and independent of one another -- only
        # the NEXT pass or wave depends on this one's outcomes -- so each
        # pass goes through `train.candidate_parallelism` exactly like an
        # ordinary round. That matters twice: the release trains its
        # population in worker processes too (`queue.get()` at `:896`), and
        # the `full` profile pins `parallel`, which a config cannot override
        # (a profile wins by design, `bird/config.py`, WHY A PROFILE WINS). Reusing `schedule`
        # is also what keeps the fork plumbing -- budget deltas, the
        # `_POLICY_STORE` writes a resume depends on -- in one place. A wave
        # is one pass unless `skip_duplicates: false` drew an arm twice; then
        # the repeats run in later passes, each resuming the slice before it
        # (`_consecutive_passes`).
        for wave_pass, pass_picks in enumerate(_consecutive_passes(picks)):
            if issued >= pooled:
                break
            wave_arms = []
            budgeted = issued
            for a in pass_picks:
                if budgeted >= pooled:
                    break
                wave_arms.append(a)
                budgeted += slice_steps
            wave_cands = [trainable[a] for a in wave_arms]
            arm_of = {c.cand_id: a for c, a in zip(wave_cands, wave_arms)}

            # The replay half of the hand-off, published to `_ROUND_REPLAY` in the
            # PARENT before the wave (a forked worker inherits it): one ring for
            # the wave under `shared_buffer`, one per arm otherwise. Popped after
            # the wave so the store never holds more than one wave's worth.
            prefill_keys: List[str] = []
            handoffs: Dict[int, SliceHandoff] = {}
            if shared_buffer:
                pool_rows, own_by_arm = buffers.materialise(None)
                pool_key = None
                if pool_rows is not None and len(pool_rows):
                    pool_key = "pool"
                    _ROUND_REPLAY[pool_key] = pool_rows
                    prefill_keys.append(pool_key)
                for a in wave_arms:
                    handoffs[a] = SliceHandoff(
                        index=len(by_arm[a]), prefill_ref=pool_key,
                        prefill_own=int(own_by_arm.get(a, 0)),
                        export_ref=f"new:{trainable[a].cand_id}:{wave}")
            else:
                for a in wave_arms:
                    own_rows, own_by_arm = buffers.materialise(a)
                    key = None
                    if own_rows is not None and len(own_rows):
                        key = f"own:{trainable[a].cand_id}"
                        _ROUND_REPLAY[key] = own_rows
                        prefill_keys.append(key)
                    handoffs[a] = SliceHandoff(
                        index=len(by_arm[a]), prefill_ref=key,
                        prefill_own=int(own_by_arm.get(a, 0)),
                        export_ref=f"new:{trainable[a].cand_id}:{wave}")

            def _run_slice(c: Candidate) -> TrainResult:
                # `n_seeds=1` by construction, not by omission: a slice resumes
                # ONE policy (`resume_ref`), so `train.seeds_per_candidate` and
                # the `select.allocation` plan have no meaning here and
                # `_check_coherence` refuses any value but 1 / `uniform` beside
                # this mode rather than letting a resolved config claim
                # replicates that never ran.
                arm = arm_of[c.cand_id]
                return backend(ctx, state, c, 1, env_steps=slice_steps,
                               resume_ref=resume[arm], handoff=handoffs[arm])

            def _noop(c: Candidate, r: TrainResult) -> None:
                # Stage 3's own `on_result` journals one `train` event per
                # candidate. A slice is not a training (see
                # `Budget.training_slices`), so the per-slice record is the
                # `interaction_slice` event below and the `train` event is
                # emitted once, at the end, off the merged result.
                return None

            # One training per candidate, charged the first time it is selected --
            # outside `slice_mode`, so `_check()` sees it and a cap can refuse the
            # round here rather than a round later.
            for a in wave_arms:
                if not by_arm[a]:
                    ctx.budget.record_training(env_steps=0)

            with ctx.budget.slice_mode():
                wave_results = schedule(ctx, state, wave_cands, _run_slice, _noop)
            for c, res in zip(wave_cands, wave_results):
                arm = arm_of[c.cand_id]
                by_arm[arm].append(res)
                resume[arm] = res.policy_ref or resume[arm]
                issued += slice_steps
                spent += int(res.env_steps_used or slice_steps)
                success = _slice_success(res)
                alloc.update(arm, success)
                # Fold the slice's export into the round's rows. `None` means the
                # backend did not export (on-policy, or a surrogate), and the seed
                # row says which; a slice that exported nothing on an off-policy
                # sb3 learner is a slice that took no step, and its `replay_added`
                # reads 0 beside `trained`.
                h = handoffs[arm]
                delta = _ROUND_REPLAY.pop(str(h.export_ref), None)
                n_added = int(len(delta)) if delta is not None else 0
                if n_added:
                    buffers.add(wave, arm, delta)
                row = res.seed_metrics[-1] if res.seed_metrics else {}
                any_slice = any_slice or bool(res.seed_metrics)
                any_supported = any_supported or bool(row.get("replay_prefill_supported"))
                s_train = sum(int(m.get("train_steps", 0) or 0) for m in res.seed_metrics)
                s_train_eval = sum(int(m.get("env_steps", 0) or 0) for m in res.seed_metrics)
                s_eval = max(0, s_train_eval - s_train)
                s_roll = max(0, int(res.env_steps_used or 0) - s_train_eval)
                s_ckpts = sum(int(m.get("n_checkpoints", 0) or 0) for m in res.seed_metrics)
                train_spent += s_train
                eval_spent += s_eval
                rollout_spent += s_roll
                checkpoint_evals += s_ckpts
                ctx.event("interaction_slice", wave=wave, arm=arm,
                          cand_id=c.cand_id, env_steps=res.env_steps_used,
                          allocated=slice_steps, success=success,
                          issued=issued, spent=spent, pooled=pooled,
                          train_steps=s_train, eval_steps=s_eval, rollout_steps=s_roll,
                          checkpoint_evaluations=s_ckpts,
                          wave_pass=wave_pass, slice_index=h.index, seed=row.get("seed"),
                          buffer_prefill_own=int(row.get("replay_prefill_own", 0) or 0),
                          buffer_prefill_pooled=int(row.get("replay_prefill_pooled", 0) or 0),
                          buffer_added=n_added,
                          buffer_rows=buffers.rows())
            for key in prefill_keys:
                _ROUND_REPLAY.pop(key, None)
        wave += 1

    # The round's pool becomes the CARRIED buffer (`loop.carry: [replay_buffer]`
    # -> `state.replay_ref`, via `update._carry_inner_loop_refs` off the
    # winner): under `shared_buffer` every merged result points at the one
    # ring, so whichever arm wins, the next round's pool and Eq. 3's moments
    # (`scaling_elite_moments`) read the WHOLE pool and not the winner's own
    # tail -- the release takes both over `All_buffer`. Under `false` each
    # merged result keeps its last slice's own export. Written only
    # when something can dereference it (`_wants_replay`), for the reason that
    # helper gives.
    pool_ref: Optional[str] = None
    pool_rows_n = 0
    if shared_buffer and _wants_replay(cfg):
        pool_rows, _own = buffers.materialise(None)
        if pool_rows is not None and len(pool_rows):
            pool_rows_n = int(len(pool_rows))
            pool_ref = _store(_REPLAY_STORE,
                              f"replay:pool:r{getattr(state, 'restart', 0)}"
                              f"i{getattr(state, 'iteration', 0)}",
                              pool_rows, _REPLAY_STORE_MAX_BYTES)
    _ROUND_REPLAY.clear()
    if any_slice and not any_supported:
        # Once per round, not per slice: the tester tier's surrogate learners
        # take the policy and the seed salt and ignore the buffer, and the
        # artifact has to say so where a reader of a LaRes tester run will look.
        # Through `effective_backend`, the one reader every "which learner?"
        # rule uses.
        from ..config import effective_backend
        backend = effective_backend(cfg)
        log.info("      shared_population: train.backend=%s does not restore or pool a "
                 "replay buffer; interaction_cfg.shared_buffer=%s did not execute here",
                 backend, shared_buffer)
        ctx.event("interaction_buffer_unsupported", backend=backend,
                  algorithm=cfg.get("train.algorithm"), shared_buffer=shared_buffer)

    results: List[TrainResult] = []
    for c in candidates:
        if c in trainable:
            arm = trainable.index(c)
            if not by_arm[arm]:
                # Selected by no wave. A real outcome under `thompson_success`
                # and not an error: it is the allocator saying this member was
                # never worth an episode. It must NOT read as a crash, so it
                # gets an untrained result with the reason spelled out.
                res = TrainResult(cand_id=c.cand_id, candidate=c)
                res.trained = False
                res.skip_reason = ("never selected by "
                                   f"{cfg['train.interaction_allocator']}")
                ctx.budget.record_skip()
                ctx.event("interaction_unselected", cand_id=c.cand_id)
            else:
                # Already charged one training on this arm's first selection
                # (above); the slices themselves counted their steps but not
                # themselves. See `Budget.training_slices`.
                res = _merge_slices(by_arm[arm])
                if pool_ref is not None:
                    res.replay_ref = pool_ref
        else:
            res = run_one(c)
        results.append(res)
        on_result(c, res)

    shares = {trainable[i].cand_id: sum(int(r.env_steps_used) for r in by_arm[i])
              for i in range(len(trainable))}
    n_slices = sum(len(v) for v in by_arm.values())
    # WHO the round's elite was -- the member Eq. 3's moments and Eq. 4's
    # parameters point at (`state.best`, fixed for the round) -- and which
    # elites `update.topology: elitist_population` carried INTO this round
    # (`state.archive`'s elite slots, by island then rank). Recorded here
    # because this is where a LaRes round is read, and because a config that
    # omits `archive` from `loop.carry` empties that set at every boundary,
    # with nothing else in the artifact to show it.
    from .update import _ELITE  # local: no import cycle
    best = getattr(state, "best", None)
    retained = [c.report.cand_id for _k, c in
                sorted(getattr(state, "archive", {}).items(),
                       key=lambda kv: (int(kv[1].island), tuple(kv[1].coords[1:])))
                if c.report is not None and c.coords[:1] == (_ELITE,)]
    ctx.event("interaction_close", mode="shared_population", waves=wave,
              env_steps_allocated=issued, env_steps_spent=spent,
              pooled_env_steps=pooled, shares=shares,
              elite_id=(best.cand_id if best is not None else ""),
              retained_elites=retained,
              # `env_steps_spent` = train + eval + rollout, and only the first
              # term is what the pooled budget allocates; the other two are
              # per-backend-call overhead paid once per SLICE here and once
              # per TRAINING under `independent`, i.e. `slices_per_candidate`
              # times more. A budget-matched comparison reads these, not
              # `env_steps_spent`.
              env_steps_train_spent=train_spent, env_steps_eval_spent=eval_spent,
              env_steps_rollout_spent=rollout_spent,
              checkpoint_evaluations=checkpoint_evals, backend_calls=n_slices,
              shared_buffer=shared_buffer, buffer_rows=buffers.rows(),
              buffer_capacity=capacity, buffer_pool_ref=pool_ref or "",
              buffer_pool_rows=pool_rows_n)
    log.info("      interaction: %d waves, %d/%d allocated (%d spent), shares %s",
             wave, issued, pooled, spent, shares)
    return results


# ==========================================================================
# interaction_allocator -- fn(ctx, n_arms) -> allocator object
#
# The object is stateful for the length of ONE round: `.select()` names the
# arms that interact next and `.update(arm, success)` feeds back what happened.
# ==========================================================================


class _UniformAllocator:
    """Round robin: every arm, every wave, in index order.

    Makes `shared_population` spend exactly what `independent` spends, per
    candidate, which is the control the Thompson arm is measured against.
    """

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms

    def select(self) -> List[int]:
        return list(range(self.n_arms))

    def update(self, arm: int, success: float) -> None:  # noqa: D401 - no state
        return None


class _ThompsonSuccessAllocator:
    """Beta-Bernoulli Thompson sampling over a sliding window of episode success.

    `LaRes_from_scratch.py:326-360`, and the paper's Eq. 2 -- `alpha'_i =
    alpha_i + n_{s,i}`, `beta'_i = beta_i + n_{f,i}` with alpha, beta = 1 (App.
    B). A draw takes `argmax_i theta_i`, `theta_i ~ Beta(...)`; the window
    truncates each arm's ONE list of outcomes so the posterior forgets (the
    release keeps successes and failures separately, each capped, `:354-360` --
    identical until one saturates), which is the knob LaRes sets per task and
    the paper never mentions.

    `skip_duplicates` reproduces App. B's "If the same individual is sampled
    again, we will skip the interaction": a wave draws `n_arms` times and the
    duplicates collapse, so a wave hands out between one and `n_arms` slices.
    The pooled budget is unchanged -- the wave loop simply runs longer -- which
    is how the release behaves too (its `while global_timesteps <
    total_timesteps`). Under `false` the duplicates are KEPT and the driver
    runs them as consecutive slices of that arm (`_consecutive_passes`), the
    ablation of the paper's rule rather than a double charge for one slice.
    """

    def __init__(self, ctx: Any, n_arms: int) -> None:
        cfg = ctx.cfg
        self.n_arms = n_arms
        self.rng = ctx.rng
        self.window = max(1, int(cfg["train.interaction_cfg.window"]))
        self.a0 = float(cfg["train.interaction_cfg.prior_alpha"])
        self.b0 = float(cfg["train.interaction_cfg.prior_beta"])
        self.skip_duplicates = bool(cfg["train.interaction_cfg.skip_duplicates"])
        self.history: List[List[float]] = [[] for _ in range(n_arms)]

    def _theta(self) -> List[float]:
        # `ctx.rng` is a stdlib `random.Random` (bird/context.py), not a numpy
        # Generator, so this is `betavariate` and not `.beta`. Drawing from
        # ctx.rng rather than a private stream is what keeps the run
        # reproducible from `seed` alone.
        out: List[float] = []
        for h in self.history:
            wins = sum(h)
            out.append(float(self.rng.betavariate(self.a0 + wins,
                                                  self.b0 + (len(h) - wins))))
        return out

    def select(self) -> List[int]:
        picks: List[int] = []
        for _ in range(self.n_arms):
            arm = int(np.argmax(self._theta()))
            if self.skip_duplicates and arm in picks:
                continue
            picks.append(arm)
        return picks

    def update(self, arm: int, success: float) -> None:
        # LaRes thresholds at 0.5 (`:355`, `if reward > 0.5`). The value it
        # thresholds is a per-episode 0/1 flag, so the threshold only bites
        # here, where `success` is a success RATE over the slice's evaluation
        # episodes -- and the paper's own quantity is the episode outcome, so
        # keeping the threshold is the faithful reading.
        h = self.history[arm]
        h.append(1.0 if float(success) > 0.5 else 0.0)
        del h[:-self.window]


@register("interaction_allocator", "uniform")
def alloc_uniform(ctx: Any, n_arms: int) -> _UniformAllocator:
    """Every arm every wave. The Fig. 6a control, and the default."""
    return _UniformAllocator(n_arms)


@register("interaction_allocator", "thompson_success")
def alloc_thompson_success(ctx: Any, n_arms: int) -> _ThompsonSuccessAllocator:
    """LaRes §4.3. Reads the environment's success flag -- a GT channel."""
    return _ThompsonSuccessAllocator(ctx, n_arms)


def _is_elite(candidate: Candidate, elite: Optional[Candidate]) -> Tuple[bool, str]:
    """Is `candidate` the elite -- by id, or by carrying the elite's reward PROGRAM?

    Both exemptions below (Eq. 3's scale-1 for elites, Eq. 4's `apply_to:
    non_elite`) need more than `cand_id`. Under `configs/methods/lares.yaml` every
    member is freshly minted each round (`ctx.next_id`) and the elite is never
    re-admitted by id, so an id test alone would never be true: `apply_to:
    non_elite` would be indistinguishable from `all` and
    `reward_scaling_exempt` could not fire. The release exempts by SLOT: an
    elite slot keeps its program verbatim across the evolution
    (`LaRes_from_scratch.py:664-668`) and its agent continues unconstrained and
    unscaled. The BIRD equivalent of that slot is a member
    whose program is the elite's, unchanged -- same reward code, same
    co-designed observation, same weights -- warm-started from the elite's
    policy: the elite continuing under a new id. That is what this matches,
    and the reason is returned so the journal can say which of the two it was.
    """
    if elite is None or candidate is None:
        return False, ""
    if candidate.cand_id == elite.cand_id:
        return True, "is the elite"
    if (candidate.reward_code == elite.reward_code
            and (candidate.observation_code or None) == (elite.observation_code or None)
            and dict(candidate.weights or {}) == dict(elite.weights or {})):
        return True, "carries the elite's reward program unchanged"
    return False, ""


# ==========================================================================
# reward_scaling -- fn(ctx, state, candidate) -> Optional[dict]
#
# PLANNED in the parent, APPLIED in the backend, and the split is the
# determinism argument of `train.candidate_parallelism`:
# the plan draws its moment sample from `ctx.rng` and journals the moments,
# both of which only the parent may do, so stage 3 calls this once per
# trainable candidate BEFORE anything can fork -- exactly where it calls
# `select.allocation` -- and leaves the result on
# `candidate.meta["reward_scaling"]`, which crosses the fork with the
# candidate and lands in `candidates/*/meta.json`. Every backend then calls
# `apply_reward_scaling` on the compiled reward, a pure function of that plan.
#
# Running the transform INSIDE the backend call, i.e. inside a forked worker
# under `parallel`, would break both: the worker's `ctx.rng` draw never reaches
# the parent, so the Thompson allocator's next draw would differ from
# `sequential` (measured: 7 waves against 9 on the tester, different shares),
# and the worker's `ctx.event` goes to a `rundir` of None, so a `parallel`
# run's journal would carry no `reward_scaling` record at all. It would also
# run once per SLICE under `shared_population`, re-sampling the moments each
# time; the release computes `Final_reward_scale` once per evolution
# (`LaRes_from_scratch.py:715-730`) and applies it for the whole interval,
# which is what one plan per candidate per round does.
# ==========================================================================


class _AffineReward:
    """`r -> (r - mu_new) * scale + mu_elite`, components untouched.

    The COMPONENTS are deliberately not scaled. They are what the reward
    claimed it was paying, and §4's per-component reflection and
    `output.trajectory_trace`'s claimed-vs-measured comparison both read them
    as the program's own numbers. The affine term is a harness stabiliser
    applied to the scalar the learner sees; folding it into the components
    would make the artifact attribute our transform to the model's code.
    """

    def __init__(self, inner: Any, mu_new: float, scale: float, mu_elite: float,
                 record: Optional[Dict[str, Any]] = None) -> None:
        self.inner = inner
        self.mu_new = float(mu_new)
        self.scale = float(scale)
        self.mu_elite = float(mu_elite)
        self.name = getattr(inner, "name", "")
        self.weights = getattr(inner, "weights", {})
        #: What the transform IS, for the seed row (`seed_metrics[*].reward_scaling`):
        #: the affine parameters plus the moments and the sample they came from.
        #: Written by the backend, not journalled, because a seed body can run in
        #: a forked worker where `ctx.event` is lost -- the same reason
        #: `elite_constraint_optimisers` lives in the row. A reward that trained
        #: under a transform the artifact does not name is the "declared and not
        #: measurable" shape; this is the measurement.
        self.record: Dict[str, Any] = dict(record or {})
        self.record.update({"applied": True, "scale": self.scale, "mu_new": self.mu_new,
                            "mu_elite": self.mu_elite})

    @property
    def component_names(self) -> List[str]:
        return getattr(self.inner, "component_names", [])

    def __call__(self, s: Any, a: Any, s_next: Any) -> Tuple[float, Dict[str, float]]:
        total, comps = self.inner(s, a, s_next)
        return (total - self.mu_new) * self.scale + self.mu_elite, comps

    def __getattr__(self, item: str) -> Any:
        return getattr(self.inner, item)


#: Where the plan lives on the candidate. Not a config key: it is what stage 3
#: decided for this candidate this round, and `_merge_child` copies it back
#: from a worker with the rest of `Candidate.meta`.
SCALING_META = "reward_scaling"


def apply_reward_scaling(candidate: Candidate, reward: Any) -> Any:
    """The backend half: wrap `reward` in the plan stage 3 left, or return it.

    Pure -- no config, no RNG, no journal -- so it is the same function in a
    forked worker as in the parent, which is what `tests/test_parallelism.py`
    needs of everything a backend runs.
    """
    plan = (candidate.meta or {}).get(SCALING_META) if candidate is not None else None
    if not isinstance(plan, dict) or not plan.get("applied"):
        return reward
    return _AffineReward(reward, plan["mu_new"], plan["scale"], plan["mu_elite"],
                         record=plan)


@register("reward_scaling", "none")
def scaling_none(ctx: Any, state: RunState, candidate: Candidate) -> Optional[Dict[str, Any]]:
    """No transform, no plan. Every method but LaRes."""
    return None


@register("reward_scaling", "elite_moments")
def scaling_elite_moments(ctx: Any, state: RunState,
                          candidate: Candidate) -> Optional[Dict[str, Any]]:
    """LaRes Eq. 3: match this reward's first two moments to the elite's.

    `r_scaled = (sigma_elite / sigma_new) (r_new - mu_new) + mu_elite`, both
    moments measured over the shared replay buffer as a surrogate for the true
    ones (§4.3) -- which is why this needs `loop.carry: [replay_buffer]` and a
    learner that fills one.

    Runs in the PARENT, once per candidate per round (see the family note
    above), and returns the plan it wrote to `candidate.meta["reward_scaling"]`:
    `applied: True` with the affine parameters and the moments, or
    `applied: False` with the reason. Three skip states, all ANNOUNCED rather
    than silently passed: no buffer yet (iteration 0, or an on-policy learner
    that keeps none), no elite yet, or a degenerate elite spread. Each leaves
    the reward untouched and writes the reason to the journal, because a
    stabiliser that quietly did not run is indistinguishable from one that ran
    and was not needed -- and the paper's whole claim for this mechanism is
    that without it the policy collapses.
    """
    from .training import _REPLAY_STORE, compile_reward  # local: no import cycle

    def _plan(**fields: Any) -> Dict[str, Any]:
        plan = {"mode": "elite_moments", **fields}
        candidate.meta[SCALING_META] = plan
        return plan

    def _skip(reason: str) -> Dict[str, Any]:
        log.info("      reward_scaling: elite_moments not applied (%s)", reason)
        ctx.event("reward_scaling_skipped", cand_id=candidate.cand_id, reason=reason)
        return _plan(applied=False, reason=reason)

    ref = getattr(state, "replay_ref", None)
    buf = _REPLAY_STORE.get(ref or "", [])
    if not buf:
        return _skip("no shared replay buffer yet")

    best = getattr(state, "best", None)
    elite_cand = getattr(best, "candidate", None) if best is not None else None
    if elite_cand is None:
        return _skip("no elite selected yet")
    exempt, why = _is_elite(candidate, elite_cand)
    if exempt:
        # LaRes exempts the elite outright: scale 1.0, offset its own mean
        # (`LaRes_from_scratch.py:729-730`). That is the identity, so plan none.
        # By id OR by program (`_is_elite`); the reason says which.
        ctx.event("reward_scaling_exempt", cand_id=candidate.cand_id, reason=why,
                  elite_id=elite_cand.cand_id)
        return _plan(applied=False, reason=why, elite_id=elite_cand.cand_id)

    try:
        from ..config import reward_language  # local, as effective_backend above
        lang = reward_language(ctx.cfg)
        reward = compile_reward(candidate.reward_code, candidate, language=lang)
        elite_reward = compile_reward(elite_cand.reward_code, elite_cand, language=lang)
    except Exception as exc:  # noqa: BLE001
        # The candidate's own compile error is the backend's to report (it will
        # hit the same one and fail the candidate); the elite's is a skip.
        return _skip(f"a reward did not compile: {exc}")

    n = min(len(buf), int(ctx.cfg["train.interaction_cfg.moment_samples"]))
    idx = ctx.rng.sample(range(len(buf)), n) if n < len(buf) else range(len(buf))
    new_vals: List[float] = []
    elite_vals: List[float] = []
    n_failed = 0
    for i in idx:
        s, ai, s2, _done = buf[int(i)]
        # `ai` is a DISCRETE index (a scalar, from `_ReplaySlice.__getitem__`'s
        # `int(act)`) or a CONTINUOUS action vector (returned as-is for a 2-D
        # act array). Index `action_set` only in the first case; a continuous
        # `ai` is already the action the reward wants, and indexing an array
        # with it would raise IndexError before SAC ran. The release passes
        # stored actions straight into the reward
        # (`LaRes_from_scratch.py:688-691`); there is no continuous->index step.
        # A continuous `ai` is in ENV space (`_sb3_export_replay` unscales SB3's
        # [-1, 1] buffer convention), so Eq. 3's moments are over the actions
        # the env took, not the scaled ones (which differ on a non-+/-1 env
        # such as pendulum's +/-2).
        if hasattr(ctx.env, "action_set") and np.ndim(ai) == 0:
            a = ctx.env.action_set[int(ai)]
        else:
            a = ai
        try:
            new_vals.append(float(reward(s, a, s2)[0]))
            elite_vals.append(float(elite_reward(s, a, s2)[0]))
        except Exception:  # noqa: BLE001
            n_failed += 1
            continue
    if len(new_vals) < 2:
        # The count is in the reason on purpose: an EMPTY buffer and a buffer
        # every row of which failed to relabel are different defects, and
        # "fewer than two" alone reads the same for both.
        return _skip(f"fewer than two usable transitions in the buffer "
                     f"({len(new_vals)} relabelled, {n_failed} failed, of {n} sampled)")

    sd_new = float(np.std(new_vals))
    sd_elite = float(np.std(elite_vals))
    if not math.isfinite(sd_new) or sd_new <= 0.0:
        return _skip("this reward is constant over the buffer (sigma_new = 0)")
    if not math.isfinite(sd_elite):
        return _skip("elite reward is not finite over the buffer")

    scale = sd_elite / sd_new
    mu_new = float(np.mean(new_vals))
    mu_elite = float(np.mean(elite_vals))
    log.info("      reward_scaling: elite_moments  scale=%.4g  mu %.4g -> %.4g  (n=%d)",
             scale, mu_new, mu_elite, len(new_vals))
    ctx.event("reward_scaling", cand_id=candidate.cand_id, scale=scale,
              mu_new=mu_new, mu_elite=mu_elite, sigma_new=sd_new,
              sigma_elite=sd_elite, n_samples=len(new_vals), n_failed=n_failed)
    return _plan(applied=True, scale=scale, mu_new=mu_new, mu_elite=mu_elite,
                 elite_id=elite_cand.cand_id, sigma_new=sd_new, sigma_elite=sd_elite,
                 n_samples=len(new_vals), n_failed=n_failed,
                 buffer_ref=str(ref), buffer_rows=int(len(buf)))



# ==========================================================================
# elite_constraint -- fn(ctx, state, candidate) -> dict | None
#
# Returns the keep-close term stage 3 should install, or None. The learner
# side is the backend's: only those with parameters can honour it.
# ==========================================================================


@register("elite_constraint", "none")
def constraint_none(ctx: Any, state: RunState, candidate: Candidate) -> Optional[Dict[str, Any]]:
    """No parameter constraint. Every method but LaRes."""
    return None


@register("elite_constraint", "l2_params")
def constraint_l2_params(ctx: Any, state: RunState,
                         candidate: Candidate) -> Optional[Dict[str, Any]]:
    """LaRes Eq. 4: `||theta_i - theta_elite||^2` on the actor and BOTH critics.

    Added to the actor and critic losses at a constant weight -- App. B says
    the weight "is set to the default value of 1.0 across all tasks" and
    `refs/code/LaRes/sac.py:459` is a literal `factor_weight = 1.0`.

    **Not `train.anchor`.** That key holds a policy near the one this candidate
    STARTED from, as an action-space divergence, on an adaptive schedule. This
    one tracks a DIFFERENT policy (the elite's, which moves as the elite does),
    in parameter space, at a fixed weight. A config may set both; they are
    different terms and nothing here reads the anchor's fields.

    Returns a descriptor rather than touching a learner: the two are wired in
    the backends that have parameters to constrain. `mock` and `tabular` have
    none, so there the constraint degrades to a no-op with a warning, exactly
    as `warm_start_from_best` and `bc_prior` already do on those backends.

    **The reference is `state.policy_ref` -- the elite checkpoint carried into
    this round -- and it is fixed for the round.** `state.best` and
    `state.policy_ref` move only in stage 6, so every slice of every arm in a
    round is pulled toward the SAME parameters, whatever that slice resumed
    from. That is the release's reading exactly: each worker loads
    `elite_actor` / `elite_q1` / `elite_q2` from `best_*.pth` once per
    evolution, when `update_model_flag` is set (`refs/code/LaRes/utils.py:
    2119-2125`), and holds them for the interval while the elite's own agent
    keeps training. The backend therefore reads this descriptor's
    `reference_policy_ref` rather than snapshotting whatever `train.init` has
    just loaded, which under `shared_population` is the arm's OWN previous
    slice for every slice after its first -- a proximal term toward itself,
    while `elite_constraint_optimisers` would still read 2.
    """
    best = getattr(state, "best", None)
    if best is None:
        return None
    ref = getattr(state, "policy_ref", None)
    if not ref:
        return None
    apply_to = ctx.cfg["train.elite_constraint.apply_to"]
    if apply_to == "non_elite":
        # By id OR by program (`_is_elite`): a member carrying the elite's
        # program unchanged is the elite continuing under a new id, and the
        # release constrains only the non-elite slots (Alg. 1 line 10).
        exempt, _why = _is_elite(candidate, getattr(best, "candidate", None))
        if exempt:
            return None
    return {"kind": "l2_params",
            "weight": float(ctx.cfg["train.elite_constraint.weight"]),
            "reference_policy_ref": ref}


# ==========================================================================
# The learner side of `elite_constraint: l2_params`.
#
# Kept here rather than in `anchored_ppo` because it is algorithm-agnostic and
# that module is PPO-shaped: it hooks the OPTIMISERS, so PPO, SAC and TD3 all
# get the term without a subclass apiece.
# ==========================================================================


def attach_l2_constraint(model: Any, weight: float) -> int:
    """Add `weight * ||theta - theta_0||^2` to every loss the model optimises.

    `theta_0` is whatever the model holds when this is called. The sb3 caller
    (`training._sb3_attach_elite_constraint`) loads the ELITE's stored
    parameters into the model for exactly the duration of this call and
    restores the slice's own afterwards, so the snapshot is theta_elite
    whatever the slice resumed from. Relying on `train.init` having just
    loaded the elite would hold only for an arm's FIRST slice under
    `shared_population`, where every later slice resumes its own previous
    slice and would be anchored to itself.

    **Implemented on the gradient, not by a proximal step.** The exact
    gradient of `w||theta - theta_0||^2` is `2w(theta - theta_0)`, so adding it
    to `p.grad` before the optimiser steps is *identical* to putting the term
    in the loss, and it composes correctly with Adam -- Adam sees one total
    gradient, as it would have. A post-step pull `theta -= lr*2w(theta-theta_0)`
    would NOT be identical under an adaptive optimiser, and the difference
    would be a silent deviation from the paper's loss.

    Returns the number of optimisers patched, so a caller can tell "constrained"
    from "found nothing to constrain" -- a zero here means the term is declared
    and not running.

    This is the SB3 shape of the term (a `model.policy` whose optimisers hang
    off `policy.optimizer` / `.actor` / `.critic` / `.critic_target`);
    `attach_l2_to_optimisers` below is the learner-agnostic core it delegates
    to, and the one `train.backend: fasttd3` calls with its own actor and
    critic AdamW optimisers.
    """
    policy = getattr(model, "policy", None)
    if policy is None:
        return 0
    optimisers: List[Any] = []
    for attr in ("optimizer", "actor", "critic", "critic_target"):
        obj = getattr(policy, attr, None)
        opt = obj if attr == "optimizer" else getattr(obj, "optimizer", None)
        if opt is not None and not any(opt is o for o in optimisers):
            optimisers.append(opt)
    return attach_l2_to_optimisers(policy.parameters(), optimisers, weight)


def attach_l2_to_optimisers(parameters: Iterable[Any], optimisers: Sequence[Any],
                            weight: float) -> int:
    """The core of `attach_l2_constraint`, for any torch learner.

    Snapshots every `requires_grad` parameter in `parameters` as `theta_0` NOW
    -- so the caller must call it after `train.init` has loaded the elite --
    and wraps each optimiser's `step` to add `2w(theta - theta_0)` to the
    gradient of every snapshotted parameter it owns before stepping (the exact
    gradient of `w||theta - theta_0||^2`; see the wrapper's docstring for why
    this and not a proximal step). Parameters are matched by identity, not by
    name, so two modules with the same parameter names (fasttd3's actor and
    critic both start at `net.0.weight`) cannot shadow each other's reference.
    An optimiser already patched is skipped, so calling twice does not double
    the term. Returns how many optimisers were patched.
    """
    import types
    import torch  # local: --validate-all imports this file where torch is absent

    by_param = {id(p): p.detach().clone() for p in parameters
                if getattr(p, "requires_grad", False)}
    if not by_param:
        return 0

    def _patch(optim: Any) -> bool:
        if optim is None or getattr(optim, "_bird_l2_elite", False):
            return False
        original = optim.step

        def step(_optim: Any, *args: Any, **kwargs: Any) -> Any:
            with torch.no_grad():
                for group in optim.param_groups:
                    for p in group["params"]:
                        base = by_param.get(id(p))
                        if base is None or p.grad is None:
                            continue
                        p.grad.add_(2.0 * float(weight) * (p.detach() - base))
            return original(*args, **kwargs)

        # A BOUND METHOD, not a bare function on the instance: torch's
        # `LRScheduler.__init__` wraps `optimizer.step` through `.__func__` to
        # track call order (fasttd3's cosine schedulers), and a plain attribute
        # has none. Bound, the scheduler wraps this step like any other.
        optim.step = types.MethodType(step, optim)  # type: ignore[method-assign]
        optim._bird_l2_elite = True  # type: ignore[attr-defined]
        return True

    n = 0
    seen: List[Any] = []
    for opt in optimisers:
        if opt is None or any(opt is o for o in seen):
            continue
        seen.append(opt)
        n += 1 if _patch(opt) else 0
    return n
