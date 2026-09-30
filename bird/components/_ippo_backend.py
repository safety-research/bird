"""Result assembly and the EVAL ROLLOUT PAYLOAD contract for `train.backend: assistax_ppo`.

WHY THIS IS A SEPARATE MODULE FROM `assistax_ppo.py`. Two reasons, and the
second is the load-bearing one:

  * `assistax_ppo.py` owns the `@register("train_backend", "assistax_ppo")`
    line and the learner. This module owns the two things that OUTLIVE the
    learner -- the `TrainResult` shape stage 4 reads, and the eval payload a
    *different* component computes its metric from. A learner rewrite must
    not be able to move a published contract by accident, and a contract that
    lives in the same file as a training loop moves every time the loop does.
  * The payload is data for code that does not run this backend (an offline
    computation of upstream's preference metric, say). Its field names are
    therefore an interface, not an implementation detail, and they are pinned
    here as module constants (`PARTNER_FIELDS`, `EPISODE_ARRAYS`) so a consumer
    can import them rather than re-typing names it believes are right.

NO JAX AT MODULE SCOPE, and nothing that imports torch. `registry.load_all()`
imports every component module for EVERY config in the repo, including
`--dry-run`, `--validate-all` and `--print-config`, so a module-level `import
jax` would put a GPU initialisation on the path of a command that executes
nothing (`training.sb3_backend` makes exactly this argument for
stable-baselines3). Every jax import in this backend
belongs inside the one function that needs it; this module needs none at all,
because everything here operates on host arrays that have already left the
device. `numpy` is a hard dependency of the repo and is fine.

THE PAYLOAD IS HOST DATA, DELIBERATELY. A `jax.Array` handed to another
process is a handle to a device buffer that process may not be able to reach,
and under `train.candidate_parallelism: parallel` the whole `TrainResult` is
pickled across a fork -- a device array either fails there or silently
round-trips through a copy nobody budgeted. Everything this module emits is
`np.ndarray` / `float` / `str`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..types import Candidate, TrainResult

# ==========================================================================
# The contract: names that must not drift
# ==========================================================================

#: `train.backend` value this module assembles for. Recorded on every seed row
#: as `backend`, beside `algorithm_cited` -- the two are different axes and
#: `simba_v2_backend`'s seed row explains at length why a row must keep them
#: apart.
BACKEND_LABEL = "assistax_ppo"

#: Bump when a field is ADDED, REMOVED or RESHAPED. A consumer pins this and
#: refuses a payload it was not written against; without it the failure mode is
#: a silently-missing array read as a zero, which is precisely the
#: `last_contact_force` hazard below, one level up.
EVAL_PAYLOAD_VERSION = 1

#: The partner's SEVEN preference parameters -- the per-partner half of the
#: signature: three weights and two two-sided ranges.
PARTNER_PREFERENCE_PARAMS: Tuple[str, ...] = (
    "w_speed",
    "w_force",
    "w_touch",
    "speed_range_min",
    "speed_range_max",
    "force_range_min",
    "force_range_max",
)

#: The three remaining scalars of the signature. Kept in their OWN tuple and
#: not folded into the seven: `reward_budget` / `overall_weight` are the
#: normalisation the wrapper applies on top of the preference, and
#: `touch_threshold` is a detector setting, so a sweep over "the partner's
#: preferences" and a sweep over "how the preference is scaled" are different
#: experiments even though every one of these rides on the same partner record.
PARTNER_SCALARS: Tuple[str, ...] = (
    "reward_budget",
    "overall_weight",
    "touch_threshold",
)

#: Every per-partner field, one vector of length `n_episodes` each.
PARTNER_FIELDS: Tuple[str, ...] = PARTNER_PREFERENCE_PARAMS + PARTNER_SCALARS

#: The ordered per-step arrays, shape `(n_episodes, T)`.
#:
#: `contact_force` IS A MAGNITUDE IN NEWTONS, not a wrench. The wrapper reduces
#: it with `jp.linalg.norm(contact_forces)` before it ever reaches the tracking
#: dict (`training.py:509` and `:531`), and `last_contact_force` is that same
#: reduced scalar, so a payload that carried the 3-vector here would be
#: comparing a vector against a scalar threshold one step later.
EPISODE_ARRAYS: Tuple[str, ...] = ("tool_speed", "contact_force")

#: Top-level keys every payload must carry.
REQUIRED_PAYLOAD_KEYS: Tuple[str, ...] = (
    "version",
    "backend",
    "n_episodes",
    "horizon",
    "partner_id",
    "steps",
    "last_contact_force",
    "partner",
)


class EvalPayloadError(ValueError):
    """A payload that a consumer must refuse rather than repair.

    Its own exception type because the repair is never local: a missing field
    means the ROLLOUT did not emit it, and a caller that caught `ValueError`
    broadly would be a caller that could plausibly fill in a default -- which
    is the one thing this contract exists to prevent.
    """


# ==========================================================================
# 1. TrainResult assembly
# ==========================================================================


@dataclass
class SeedOutcome:
    """What one seed of this backend produced, already off the device.

    ONE SEED, NOT ONE CALL. `train.seeds_per_candidate` replicates the inner
    loop and the seeds are reassembled BY SEED INDEX, never by completion order
    -- so the transport between the
    learner and `build_train_result` is a list of these in seed order, and this
    dataclass deliberately carries `seed` so a caller that got the order wrong
    is detectable in the artifact instead of invisible in it.

    `policy_params` is the learner's pytree ALREADY PULLED TO HOST
    (`jax.device_get`). It is accepted here and never written onto the
    `TrainResult`: `TrainResult` has no params field, only `policy_ref`, and
    that is not an oversight -- `RunState` stays checkpointable by storing
    handles, and the arrays live in the module-level policy store
    (`training._store`, `training._POLICY_STORE`).
    """

    seed: int
    #: Per-checkpoint curve rows, oldest first. Build them with `checkpoint_row`.
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    #: Per-checkpoint reward-component means, parallel to `checkpoints`.
    components: List[Dict[str, float]] = field(default_factory=list)
    policy_params: Any = None
    #: Which checkpoint SHIPPED. `None` means the last one, which is
    #: `train.checkpoint_selection: final`, the default. The caller chooses it
    #: because the criterion is the candidate's OWN reward and never the task
    #: metric -- selecting on fitness leaks ground truth into training --
    #: and this module, which can see
    #: `fitness` on every row, must not be the thing that picks.
    shipped_index: Optional[int] = None
    env_steps: int = 0          # training + evaluation, what the budget is charged
    train_steps: int = 0        # training only
    train_steps_requested: int = 0
    wallclock_s: float = 0.0
    #: Environment steps per second, as MEASURED over this seed's training
    #: (not a configured target). Distinct from `compile_s`: on an XLA backend
    #: the first iteration pays the trace and would drag an sps averaged over
    #: it down by an amount that varies with the step budget, so a reader
    #: comparing two runs' throughput needs the compile term reported beside
    #: it rather than mixed into it.
    sps: float = 0.0
    #: Seconds spent in XLA tracing/compilation for this seed. Reported even
    #: when zero: zero means "measured and there was none", absent would mean
    #: "nobody looked", and those are different claims about a jitted backend.
    compile_s: float = 0.0
    num_envs: int = 0
    horizon_effective: int = 0
    error: str = ""
    timed_out: bool = False
    pruned_at_round: Optional[int] = None
    warm_started_from: str = ""
    #: The evaluation rollouts as `Trajectory` objects (rows, rewards, done), the
    #: first `evaluate.rollouts_per_candidate` episodes of upstream's eval. Empty
    #: when the eval logged no rows. Shipped from the LAST seed, the same rule as
    #: `policy_ref` ("the last seed's learner is the one that is kept").
    trajectories: List[Any] = field(default_factory=list)
    #: THE HELD-OUT TRIPLE, as fields rather than `extra` entries.
    #:
    #: As `extra` entries they would reach the seed row only through a key
    #: somebody remembered to write, so `gt_return` could read a real value
    #: on the CHECKPOINT row (which has the field) and `null` on the seed
    #: row. A number that exists one level down and is null where a reader
    #: looks is worse than absent.
    #:
    #: `None` IS MEANINGFUL ON ALL THREE and must stay distinct from 0.0:
    #: `gt_return` None means no reference was computable, and `r_pref` None
    #: means no partner was loaded -- a 0.0 there says a partner ran and
    #: earned nothing, which is a different experiment.
    gt_return: Optional[float] = None
    r_pref: Optional[float] = None
    r_task: Optional[float] = None
    #: The EFFECTIVE XLA flag string, read from the environment at run time --
    #: the flags actually in effect. `XLA_FLAGS` may be extended by the caller
    #: (e.g. `--xla_gpu_autotune_level=0` beside the worker's own
    #: `--xla_gpu_enable_triton_gemm=false`), and since autotune-off alone does
    #: NOT fix the Triton refusal, a record of the intent would attribute a
    #: pass to the wrong flag.
    xla_flags: str = ""
    #: Steps the gt_return reduction dropped for pre/post misalignment. ON THE
    #: SEED ROW, not only in `meta`: the reader looks at the row, and a count
    #: that lives only in metadata is a count nobody checks.
    #: NOTE THAT THE NAME ASSERTS MORE THAN THE VALUE COUNTS. It reads as
    #: "steps missing from this payload"; what it counts is steps excluded
    #: from ONE SUM -- the step arrays are complete. A clearer name would be
    #: `gt_return_steps_excluded`. Non-zero is NORMAL -- one per episode
    #: column -- and `validate_eval_payload` checks the multiple rather than
    #: leaving that in prose.
    gt_return_steps_dropped: int = 0
    #: This seed's eval rollout payload, already validated. Carried on the
    #: outcome rather than returned separately so a seed that produced no
    #: payload is visible as `None` instead of as a shorter list.
    eval_payload: Optional[Dict[str, Any]] = None
    #: Backend facts with no home above. Merged into the seed row last, so it
    #: can never overwrite a contract field by colliding with its name.
    extra: Dict[str, Any] = field(default_factory=dict)


def checkpoint_row(*, step: float, round_: int, seed: int, fitness: float,
                   success_rate: float, reward_return: float,
                   gt_return: Optional[float],
                   eval_wall_s: float = 0.0,
                   **extra: Any) -> Dict[str, Any]:
    """One curve row, with the alias keys every downstream reader expects.

    THE ALIASES ARE NOT REDUNDANCY. `fitness` / `score` / `task_success` /
    `consecutive_successes` are four names the readers in `evaluation.py` and
    `search.py` use for the BIRD task metric, and a backend that emitted only
    one of them would make a `select.rule` or a prune field read an empty
    series and silently rank on nothing. Copied from the shape the `simba_v2`
    and `fasttd3` checkpoint rows already emit, so a cross-backend diff of two
    curves compares rows and not vocabularies.

    `gt_return` IS THE ONE KEY THAT MAY BE `None`, AND IT MUST NOT BE 0.0.
    It is the only member of `native_signal.NATIVE_REWARD_KEYS`, so on this
    tier -- where the task is `continuous_only` and ships no native success --
    `evaluate.fitness.source: native` resolves to `native_reward` and reads
    exactly this key (`native_signal.py:69`, `evaluation.py:907-925`). A `None`
    there means "this task has no reference reward" and is skipped; a `0.0`
    means "the reference reward paid nothing", which is a MEASUREMENT. The
    Assistax adapter defines `reference_reward` precisely so this key is real
    on this tier (`bird/envs/upstream_assistax.py`), so a `None` here would be
    this backend contradicting the adapter.
    """
    row: Dict[str, Any] = {
        "step": float(step), "round": float(round_), "seed": float(seed),
        "fitness": float(fitness), "score": float(fitness),
        "task_success": float(fitness), "consecutive_successes": float(fitness),
        "success_rate": float(success_rate),
        "gt_return": (None if gt_return is None else float(gt_return)),
        "return": float(reward_return), "reward_return": float(reward_return),
        "eval_wall_s": float(eval_wall_s),
    }
    row.update(extra)
    return row


def seed_metric_row(outcome: SeedOutcome, *, algorithm_cited: str = "ppo",
                    backend: str = BACKEND_LABEL,
                    init_source: str = "scratch") -> Dict[str, Any]:
    """One `TrainResult.seed_metrics` entry: this seed AS EXECUTED.

    The row answers questions the curve cannot. `fitness` here is the SHIPPED
    checkpoint's, `max` and `auc` are over the whole curve, and all three are
    on the row because a reward whose best checkpoint is early and whose final
    checkpoint collapsed is a different result from one that never got there --
    and `train.checkpoint_selection` decides which of them the search sees.

    `algorithm_cited` IS A CITATION, NOT AN EXECUTION FACT, and it keeps that
    name for the reason `fasttd3.py` gives: the method's paper named an
    algorithm, this backend runs the one it runs, and a row where the two
    coincide is the row that teaches a reader to read them as one field.

    EMPTY-VALUE DISCIPLINE: `error` is `""` when the seed was fine, never
    `None`; `pruned_at_round` is `None` when nothing pruned, never `0` -- round
    zero is a real round. This mirrors the `failure` / `failure_kind`
    convention one level up: the populations must stay separable in the
    artifact.
    """
    curve = list(outcome.checkpoints)
    fits = [float(p.get("fitness", 0.0)) for p in curve]
    ship_i = _shipped_index(outcome, len(curve))
    row: Dict[str, Any] = {
        "seed": int(outcome.seed),
        "fitness": (fits[ship_i] if fits else 0.0),
        "final": (fits[-1] if fits else 0.0),
        "max": (float(np.max(fits)) if fits else 0.0),
        "auc": (float(np.mean(fits)) if fits else 0.0),
        "env_steps": int(outcome.env_steps),
        "train_steps": int(outcome.train_steps),
        "train_steps_requested": int(outcome.train_steps_requested),
        "wallclock_s": float(outcome.wallclock_s),
        # THE THREE THROUGHPUT FACTS, together. `sps` alone cannot be compared
        # across two runs with different step budgets on a jitted backend,
        # because the trace is a fixed cost amortised over the budget; with
        # `compile_s` and `wallclock_s` beside it the reader can do the
        # subtraction that makes them comparable.
        "sps": float(outcome.sps),
        "compile_s": float(outcome.compile_s),
        "num_envs": int(outcome.num_envs),
        "horizon_effective": int(outcome.horizon_effective),
        "n_checkpoints": len(curve),
        "shipped_checkpoint": (int(curve[ship_i]["round"]) if curve else None),
        "final_checkpoint_fitness": (fits[-1] if fits else 0.0),
        "pruned_at_round": outcome.pruned_at_round,
        "timed_out": bool(outcome.timed_out),
        "learner": BACKEND_LABEL,
        "algorithm_cited": algorithm_cited,
        "backend": backend,
        "init": init_source,
        "warm_started_from": outcome.warm_started_from,
        # DID THIS SEED EMIT A PAYLOAD, as a fact on the row rather than an
        # inference from a `None` somewhere else. A seed that trained fine and
        # emitted nothing is a rollout bug; a seed that errored and emitted
        # nothing is expected. Without this the two look identical downstream.
        "eval_payload": bool(outcome.eval_payload),
        "eval_payload_version": (
            None if not outcome.eval_payload
            else int(outcome.eval_payload.get("version", -1))),
        "error": outcome.error,
        "checkpoints": curve,
    }
    # BEFORE `extra`, so a stray `extra` key of the same name cannot overwrite
    # a contract field with something that was never computed.
    row.update({"gt_return_steps_dropped": outcome.gt_return_steps_dropped,
                "gt_return": outcome.gt_return,
                "r_pref": outcome.r_pref,
                "r_task": outcome.r_task,
                "xla_flags": outcome.xla_flags})
    row.update(outcome.extra)
    return row


def build_train_result(candidate: Candidate, outcomes: Sequence[SeedOutcome], *,
                       algorithm_cited: str = "ppo",
                       init_source: str = "scratch",
                       init_from_cand_id: Optional[str] = None,
                       store_policy: Optional[Callable[[Any], str]] = None,
                       wallclock_s: Optional[float] = None,
                       trained: bool = True) -> TrainResult:
    """Fold n seed outcomes into the one `TrainResult` stage 4 reads.

    SEED ORDER IS THE ARGUMENT ORDER, ALWAYS. `outcomes` is consumed as given
    and never sorted by completion, wall clock or fitness: candidate
    parallelism reassembles by index precisely so a scheduling knob cannot
    become a science knob, and a fold that reordered here would reintroduce
    that at the seed level.

    WHAT IS DELIBERATELY LEFT EMPTY, because an empty field is a statement:

      * `epoch_series` -- this backend has no per-training-epoch scalar log to
        report. `TrainResult.epoch_series` fixes the convention: empty on every backend
        that does not fill it, and the renderer says so rather than drawing an
        empty block. Filling it with the checkpoint curve would be a second,
        differently-sampled copy of `checkpoints` under a name that promises a
        different axis.
      * `store_trajectories` -- the CARD TPE pool is collected by the caller
        under `verify.tpe.trajectories_per_iteration`, not by the learner.

    `trained=False` IS DERIVED, NOT ASSUMED. Any seed with a non-empty `error`
    makes the whole call a failed candidate and the first such error is the
    one reported, because the stage above treats `trained` as the switch
    between "this reward produced a policy" and "this reward produced nothing";
    a partial result that still claimed `trained=True` would be ranked against
    complete ones.

    `store_policy` TAKES THE PARAMS AND RETURNS A HANDLE. Passing it is how the
    caller decides the params survive; without it `policy_ref` stays `None` and
    the run has no warm start to offer, which is the honest outcome rather than
    a dangling reference. Only the LAST seed's params are stored -- "the last
    seed's learner is the one that is kept", the same rule the other backends
    in `training` follow -- because `policy_ref` is one handle and a silent choice among n
    would be worse than a stated one.
    """
    from .training import _mean_curve, _mean_components  # lazy: keeps this module leaf-ish

    result = TrainResult(cand_id=candidate.cand_id, candidate=candidate)
    outcomes = list(outcomes)
    result.seed_metrics = [
        seed_metric_row(o, algorithm_cited=algorithm_cited, init_source=init_source)
        for o in outcomes
    ]
    result.checkpoints = _mean_curve([o.checkpoints for o in outcomes])
    result.component_traces = _mean_components([o.components for o in outcomes])
    # `.get`, not `[...]`: a row assembled by hand elsewhere may lack the key,
    # and `None` is the correct "no channel" entry -- never 0.0
    # (`TrainResult.gt_reward_curve`).
    result.gt_reward_curve = [p.get("gt_return") for p in result.checkpoints]
    result.env_steps_used = int(sum(int(o.env_steps) for o in outcomes))
    # DEFAULT IS THE SUM, WHICH IS NOT THE WALL CLOCK WHEN SEEDS ARE FORKED.
    # The sum is the compute spent; the elapsed time of a forked fan-out is
    # less. The caller measures the elapsed time if it wants it, and passes it,
    # rather than this function guessing which schedule ran.
    result.wallclock_s = (float(sum(float(o.wallclock_s) for o in outcomes))
                          if wallclock_s is None else float(wallclock_s))
    result.pruned = any(o.pruned_at_round is not None for o in outcomes)
    result.timed_out = any(o.timed_out for o in outcomes)
    errors = [o.error for o in outcomes if o.error]
    result.error = errors[0] if errors else ""
    result.trained = bool(trained and not errors)
    result.init_source = init_source
    result.init_from_cand_id = init_from_cand_id
    if outcomes:
        # The rollouts stage 4 reads (`evaluation._rollouts`): the last seed's, as
        # `policy_ref` is -- one stated choice rather than a silent one among n.
        result.trajectories = list(outcomes[-1].trajectories or [])
    if store_policy is not None:
        last = outcomes[-1] if outcomes else None
        if last is not None and last.policy_params is not None:
            result.policy_ref = store_policy(last.policy_params)
    return result


def _shipped_index(outcome: SeedOutcome, n: int) -> int:
    """Which curve row shipped; the last one when the caller named none.

    Clamped rather than raising: an out-of-range index is a caller bug, but
    losing an entire trained seed to an `IndexError` during bookkeeping costs
    more than recording the final checkpoint and letting the row's
    `shipped_checkpoint` show what happened.
    """
    if n <= 0:
        return 0
    idx = n - 1 if outcome.shipped_index is None else int(outcome.shipped_index)
    return max(0, min(n - 1, idx))


# ==========================================================================
# 2. The eval rollout payload
# ==========================================================================


def _terminating_tasks() -> Any:
    """Upstream's own `TERMINATES`, imported -- never restated here.

    A list in this file would be a second copy of a fact
    `upstream_assistax.TERMINATES` already owns, and two copies of one list
    drift. Imported lazily so this module stays importable without the env
    package.
    """
    from ..envs.upstream_assistax import TERMINATES       # noqa: WPS433

    return TERMINATES


def _require_partner_fields(partner: Mapping[str, Any]) -> None:
    """Refuse a partner block missing any of the ten, AT BUILD TIME.

    Filtering with `if k in partner` would silently drop a missing field and
    build a knowingly incomplete payload, leaving the refusal to the consumer
    -- after a full training, and far from the line where the block was
    assembled.
    """
    missing = [k for k in PARTNER_FIELDS if k not in partner]
    if missing:
        raise EvalPayloadError(
            f"partner block is missing {missing}; these are the arguments of "
            f"compute_preference_reward and a payload without them cannot be "
            f"scored. Refusing at assembly rather than building an incomplete "
            f"payload for a consumer to reject.")


def eval_rollout_payload(*, partner_id: Sequence[str],
                         tool_speed: Any,
                         contact_force: Any,
                         last_contact_force: Any,
                         partner: Mapping[str, Any],
                         task: str = "",
                         seed: Optional[int] = None,
                         checkpoint_round: Optional[int] = None,
                         meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the published eval-rollout payload for one evaluation.

    THIS IS A CONTRACT, NOT A CONVENIENCE. The payload carries what upstream's
    preference metric (`compute_preference_reward`,
    `refs/code/assistax/assistax/wrappers/training.py:537`) takes, and this
    module computes no metric at all: re-deriving it here would give the repo
    two implementations of one number -- two copies of one fact, which drift.

    The field names are upstream's ARGUMENT names, not ours. `tool_speed` and
    `contact_force` map onto `speed` and `force`; every partner field is
    already spelled exactly as the parameter it feeds. Renaming for local
    taste would make the consumer maintain a translation table whose drift
    nothing can detect.

    `last_contact_force` IS REQUIRED AND HAS NO DEFAULT. It is the contact
    force carried INTO step 0 of each episode -- `prev_contact_force` for the
    first step -- and the wrapper keeps it in its tracking dict for exactly
    this purpose (`training.py:521`, `_update_tracking`). Defaulting it to 0.0
    manufactures a rising edge: the touch detector fires on `(prev <
    touch_threshold) & (current >= touch_threshold)`, so an episode that
    resumes already in contact would be charged `w_touch` a second time for a
    touch that never ended. A genuine reset may legitimately REPORT 0.0; what
    is forbidden is 0.0 arriving because nobody emitted anything. Hence the key
    is required, and `validate_eval_payload` refuses its absence by name.

    Shapes: `tool_speed` and `contact_force` are `(n_episodes, T)`, ordered in
    time along axis 1 and aligned step-for-step with each other;
    `last_contact_force` and every `partner` field are `(n_episodes,)`, one
    value per episode, so a run that evaluated several partners is one payload
    and not several. `partner_id` is the episode's partner as a STRING -- an
    identity, never an index into a table the consumer would have to also be
    given.
    """
    speed = _as_2d(tool_speed, "tool_speed")
    force = _as_2d(contact_force, "contact_force")
    n_ep = int(speed.shape[0])
    # BEFORE the assembly below, so a missing field is named here rather than
    # raising a bare KeyError out of a dict comprehension.
    _require_partner_fields(partner)
    horizon = int(speed.shape[1])
    payload: Dict[str, Any] = {
        "version": EVAL_PAYLOAD_VERSION,
        "backend": BACKEND_LABEL,
        "task": str(task),
        "seed": (None if seed is None else int(seed)),
        "checkpoint_round": (None if checkpoint_round is None else int(checkpoint_round)),
        "n_episodes": n_ep,
        "horizon": horizon,
        "partner_id": [str(p) for p in partner_id],
        "steps": {"tool_speed": speed, "contact_force": force},
        # TOP LEVEL, NOT INSIDE `steps` AND NOT INSIDE `partner`. It is neither:
        # a per-step array would imply it is recomputable from the series (it
        # is not -- step 0's predecessor is outside the window), and a partner
        # field would imply it is a property of the partner rather than of this
        # episode's entry condition. Its own key is the only placement that
        # makes "it was not emitted" visible.
        "last_contact_force": _as_1d(last_contact_force, "last_contact_force", n_ep),
        "partner": {k: _as_1d(partner[k], f"partner.{k}", n_ep)
                    for k in PARTNER_FIELDS},
        "meta": dict(meta or {}),
    }
    # ALWAYS, with no opt-out: a `validate=False` switch is one production can
    # clear by accident.
    if True:
        require_valid_eval_payload(payload)
    return payload


# ==========================================================================
# 3. The validator
# ==========================================================================


def validate_eval_payload(payload: Any) -> List[str]:
    """Every problem with this payload, as a list of sentences; empty when valid.

    ALL OF THEM AT ONCE, not the first. `config.validate()` makes the argument
    for the config space and it holds here: a consumer that fixes one missing
    field, re-runs an evaluation and discovers a second has paid for two
    rollouts to learn what one report could have told it.

    EVERY MESSAGE NAMES THE FIELD. A validator that says "malformed payload" is
    a validator whose output has to be debugged, and the whole point of
    refusing is to make the emitter's bug local.

    WHAT IT CHECKS BEYOND PRESENCE, and why each earns its place:
      * shape agreement -- `(n_episodes, T)` against `n_episodes` everywhere,
        because a payload whose partner vector is one short silently broadcasts
        or truncates under numpy rather than failing;
      * finiteness -- a NaN in `contact_force` propagates through the gaussian
        into the total and arrives as a NaN metric with no origin;
      * DEGENERATE RANGES -- `speed_range_min == speed_range_max` makes
        upstream's `width` zero and `_gaussian_pref` divides by it, so every
        out-of-range step becomes NaN while every in-range step is fine. That
        is a payload whose metric is computable for some policies and not
        others, which is the worst version of this failure.
    It does NOT check `w_touch < 0`: a non-negative touch weight is a strange
    partner, not a malformed payload, and a validator that refused it would be
    making a science decision.
    """
    problems: List[str] = []
    if not isinstance(payload, Mapping):
        return [f"payload must be a mapping, got {type(payload).__name__}"]

    for key in REQUIRED_PAYLOAD_KEYS:
        if key not in payload:
            problems.append(f"missing required field {key!r}")
    if problems:
        # Shape checks below would raise KeyError; report the absences first --
        # they are the cause, and a cascade of shape complaints would bury them.
        return problems

    version = payload.get("version")
    if int(version if isinstance(version, int) else -1) != EVAL_PAYLOAD_VERSION:
        problems.append(
            f"field 'version' is {version!r}, this reader implements "
            f"{EVAL_PAYLOAD_VERSION}")

    n_ep = _int_or_none(payload.get("n_episodes"))
    horizon = _int_or_none(payload.get("horizon"))
    if n_ep is None or n_ep <= 0:
        problems.append(f"field 'n_episodes' must be a positive int, got "
                        f"{payload.get('n_episodes')!r}")
    if horizon is None or horizon <= 0:
        problems.append(f"field 'horizon' must be a positive int, got "
                        f"{payload.get('horizon')!r}")

    steps = payload.get("steps")
    if not isinstance(steps, Mapping):
        problems.append("field 'steps' must be a mapping of per-step arrays")
    else:
        for name in EPISODE_ARRAYS:
            if name not in steps:
                problems.append(f"missing required per-step array 'steps.{name}'")
                continue
            problems += _check_2d(steps[name], f"steps.{name}", n_ep, horizon)

    problems += _check_1d(payload.get("last_contact_force"), "last_contact_force", n_ep)

    ids = payload.get("partner_id")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        problems.append("field 'partner_id' must be a sequence of episode partner ids")
    elif n_ep is not None and len(ids) != n_ep:
        problems.append(f"field 'partner_id' has {len(ids)} entries, expected "
                        f"n_episodes={n_ep}")
    elif any(not str(p) for p in ids):
        problems.append("field 'partner_id' contains an empty id; an unnamed partner "
                        "cannot be joined to a partner record")

    partner = payload.get("partner")
    if not isinstance(partner, Mapping):
        problems.append("field 'partner' must be a mapping of the preference parameters")
    else:
        for name in PARTNER_FIELDS:
            if name not in partner:
                problems.append(
                    f"missing required partner parameter 'partner.{name}' "
                    f"(an argument of compute_preference_reward)")
                continue
            problems += _check_1d(partner[name], f"partner.{name}", n_ep)
        problems += _check_range(partner, "speed_range_min", "speed_range_max")
        problems += _check_range(partner, "force_range_min", "force_range_max")
    # THE JOIN, and the non-terminating precondition. Both are about the same
    # premise: one partner per payload.
    problems += _partner_join_problems(payload)
    # TWO COPIES OF ONE FACT, MADE LOAD-BEARING. `partner_distinct_ids` is
    # derivable from `partner_id`, which is already length-checked, so the
    # redundancy is only worth carrying if a disagreement is an error --
    # otherwise the join check believes `partner_id` while a reader believes
    # the meta number. One line rather than dropping the key, because the key
    # survives into the artefact where the array does not get re-derived.
    #
    # The producer derives it from the step-0 `partner_id` list, so both
    # copies are the same KIND of quantity -- a redundancy check is only as
    # good as that. The all-step count is a different quantity and has its
    # own key (below).
    _meta = payload.get("meta") or {}
    # EVERY COLUMN DROPS THE SAME NUMBER -- that is the invariant, and it is
    # NOT "one per episode", which is the intuitive reading and the wrong
    # one. A window holding two episodes per column drops two per column and
    # is still a multiple. What makes the count a multiple of n_episodes is
    # that the columns agree with each other, so this check is testing THE
    # RECTANGLE from another angle: it costs nothing on anything we expect to
    # emit and fires on the thing that would actually be wrong.
    #
    # WHY IT HOLDS, AND WHAT BREAKS IT. Columns disagree only if episodes
    # have different lengths, which requires early termination -- and the
    # terminating-task refusal above excludes exactly that. So this check is
    # valid BECAUSE of that one. A check whose validity rests on another
    # check is the thing that gets left behind in a later change, so: if
    # per-episode lengths are added and terminating tasks are admitted, this
    # goes at the same moment.
    #
    # Checked rather than stated, the same trade as the distinct-ids check
    # above: an invariant left as prose is a fact about the code that nothing
    # re-derives.
    n_dropped = _meta.get("gt_return_steps_dropped")
    if n_dropped is not None and n_ep and int(n_dropped) % int(n_ep):
        problems.append(
            f"meta.gt_return_steps_dropped is {n_dropped}, not a multiple of "
            f"n_episodes ({n_ep}). One done step per episode column is the "
            f"mechanism, so a remainder means episodes of differing length "
            f"reached a reduction that assumes a rectangle.")
    actual = len({str(p) for p in (payload.get("partner_id") or [])})
    if "partner_distinct_ids" in _meta:
        if int(_meta["partner_distinct_ids"]) != actual:
            problems.append(
                f"meta.partner_distinct_ids is {_meta['partner_distinct_ids']} "
                f"but partner_id holds {actual} distinct ids")
    # THE COUNT THAT IS NOT DERIVABLE, AND WHY IT IS A SECOND FIELD.
    # `LoadAgentWrapper` resamples the partner at every mid-rollout
    # auto-reset (`aht.py:900-903`), so a rollout contains partners no step-0
    # row names. The step-0 and all-step counts diverge once a rollout is long
    # enough to auto-reset (e.g. 57 all-step against 31 step-0 distinct ids
    # over 32 episodes). Both quantities are worth keeping, so they are two
    # keys.
    # The relation is an INEQUALITY and only in this direction -- a resample
    # can add partners, never remove one -- so the equality above is
    # untouched rather than weakened to cover both.
    if "partner_distinct_ids_all_steps" in _meta:
        resampled = int(_meta["partner_distinct_ids_all_steps"])
        if resampled < actual:
            problems.append(
                f"meta.partner_distinct_ids_all_steps is {resampled}, fewer "
                f"than the {actual} distinct ids partner_id already holds at "
                f"step 0. Resampling on auto-reset can only add partners, so "
                f"a smaller all-step count means the two were measured over "
                f"different rollouts.")
    task = str(payload.get("task") or "")
    if task and task in _terminating_tasks():
        problems.append(
            f"task {task!r} can terminate early, and this payload carries no "
            f"per-episode length or valid mask. Episodes of different lengths "
            f"are padded into a rectangular (n_episodes, T), and zero padding "
            f"is finite and in range -- so padded steps would score as "
            f"simulated time. Refusing until the payload carries lengths.")

    return problems


def _partner_join_problems(payload: Mapping[str, Any]) -> List[str]:
    """Episodes sharing a `partner_id` must share all ten parameter values.

    THE CHECK THAT ACCEPTING (n_episodes,) ARRAYS DOES NOT BUY. An emitter
    that lists 32 distinct partner ids while broadcasting ONE partner's
    parameters across all 32 entries produces a payload valid on every other
    check -- right lengths, finite, ranges non-degenerate -- whose metric
    scores 32 episodes against the wrong partner. The ids and the parameters
    are both per episode and nothing else ties them together.
    """
    problems: List[str] = []
    ids = list(payload.get("partner_id") or [])
    partner = payload.get("partner") or {}
    if not ids or not partner:
        return problems

    groups: Dict[str, List[int]] = {}
    for i, pid in enumerate(ids):
        groups.setdefault(str(pid), []).append(i)

    for name in PARTNER_FIELDS:
        vals = np.asarray(partner.get(name, []), dtype=float).reshape(-1)
        if vals.size != len(ids):
            continue                      # a length problem is reported elsewhere
        for pid, rows in groups.items():
            if len({float(vals[i]) for i in rows}) > 1:
                problems.append(
                    f"partner.{name} differs between episodes that share "
                    f"partner_id {pid!r}: one partner cannot have two values "
                    f"for one preference parameter")
    return problems


def require_valid_eval_payload(payload: Any) -> Any:
    """`validate_eval_payload`, raising `EvalPayloadError` on the first call.

    Two entry points on purpose: a caller that wants to REPORT gets the list, a
    caller that must not proceed gets the exception. The exception carries
    every problem, one per line, for the same reason the list does.
    """
    problems = validate_eval_payload(payload)
    if problems:
        raise EvalPayloadError(
            f"eval rollout payload is not valid ({len(problems)} problem(s)):\n  - "
            + "\n  - ".join(problems))
    return payload


# ==========================================================================
# small helpers -- host arrays only
# ==========================================================================


def _as_2d(value: Any, name: str) -> np.ndarray:
    """`(n_episodes, T)` float64 on the host.

    `np.asarray` on a `jax.Array` copies it off the device without this module
    importing jax, which is the whole reason the conversion happens at
    assembly time rather than being left to the consumer.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise EvalPayloadError(
            f"{name} must be 2-D (n_episodes, T), got shape {arr.shape}")
    return arr


def _as_1d(value: Any, name: str, n_ep: Optional[int] = None) -> np.ndarray:
    """A per-episode vector. A SCALAR IS REFUSED BY NAME, never reshaped.

    Reshaping a 0-d input to `(1,)` would not make it per-episode: shape (1,)
    is not (n_episodes,), so a scalar would survive assembly and be refused
    later by a LENGTH check -- and at n_episodes == 1 the two are
    indistinguishable and it would validate clean.

    Refusing rather than broadcasting is the stronger form: under a partner
    resampled per episode, a scalar is not a value to spread across episodes,
    it is a MISSING per-episode measurement. Broadcasting would make the
    one-partner premise permanent and invisible instead of loud.
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        raise EvalPayloadError(
            f"{name} is a scalar; it must be per episode, shape (n_episodes,)"
            f"{'' if n_ep is None else f' = ({n_ep},)'}. Under a partner "
            f"resampled per episode a single value is a missing measurement, "
            f"not one to broadcast.")
    if arr.ndim != 1:
        raise EvalPayloadError(
            f"{name} must be 1-D (n_episodes,), got shape {arr.shape}")
    return arr


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _check_2d(value: Any, name: str, n_ep: Optional[int],
              horizon: Optional[int]) -> List[str]:
    arr = np.asarray(value)
    if arr.ndim != 2:
        return [f"field '{name}' must be 2-D (n_episodes, T), got shape {arr.shape}"]
    out: List[str] = []
    if n_ep is not None and arr.shape[0] != n_ep:
        out.append(f"field '{name}' has {arr.shape[0]} episodes, expected "
                   f"n_episodes={n_ep}")
    if horizon is not None and arr.shape[1] != horizon:
        out.append(f"field '{name}' has {arr.shape[1]} steps, expected "
                   f"horizon={horizon}")
    if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
        out.append(f"field '{name}' contains a non-finite value; a NaN here "
                   f"reaches the metric with no origin")
    return out


def _check_1d(value: Any, name: str, n_ep: Optional[int]) -> List[str]:
    if value is None:
        return [f"field '{name}' is None; it is required and has no default "
                f"(a defaulted last_contact_force manufactures a touch)"]
    arr = np.asarray(value)
    if arr.ndim != 1:
        return [f"field '{name}' must be 1-D (n_episodes,), got shape {arr.shape}"]
    out: List[str] = []
    if n_ep is not None and arr.shape[0] != n_ep:
        out.append(f"field '{name}' has {arr.shape[0]} entries, expected "
                   f"n_episodes={n_ep}")
    if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
        out.append(f"field '{name}' contains a non-finite value")
    return out


def _check_range(partner: Mapping[str, Any], lo_key: str, hi_key: str) -> List[str]:
    """Refuse a zero-width or inverted preference range.

    Upstream's `_gaussian_pref` centres on `(min + max) / 2` with `width =
    (max - min) / 2` and divides by `width ** 2`, so `min == max` is a division
    by zero that only shows up on the steps that fall OUTSIDE the range -- the
    in-range branch returns 1.0 and hides it. An inverted range is worse: every
    step is "outside" and the gaussian still evaluates, so the metric is a
    finite number computed against a range nobody meant.
    """
    if lo_key not in partner or hi_key not in partner:
        return []
    lo = np.asarray(partner[lo_key], dtype=float).reshape(-1)
    hi = np.asarray(partner[hi_key], dtype=float).reshape(-1)
    if lo.shape != hi.shape:
        return [f"'partner.{lo_key}' and 'partner.{hi_key}' have different shapes "
                f"({lo.shape} vs {hi.shape})"]
    out: List[str] = []
    if np.any(hi < lo):
        out.append(f"'partner.{hi_key}' is below 'partner.{lo_key}' for at least one "
                   f"episode; the preference range is inverted")
    if np.any(np.isclose(hi, lo)):
        out.append(f"'partner.{hi_key}' equals 'partner.{lo_key}' for at least one "
                   f"episode; the gaussian width is zero and every out-of-range step "
                   f"becomes NaN")
    return out


# ============================ THE TRAINING CALL ============================
#
# Assembled here rather than in `assistax_ppo.py` so that module stays the
# registration, and so this half can be imported by a test without pulling the
# registry in.


#: Upstream's own IPPO hyperparameters, `assistax/baselines/IPPO/config/
#: ippo.yaml` at a7d94f4e. COPIED WITH THEIR VALUES rather than re-tuned: the
#: tier's claim is that this is upstream's learner on upstream's settings, and
#: a number changed here quietly makes that false. `TOTAL_TIMESTEPS`,
#: `NUM_SEEDS` and `SEED` are per-run and are supplied by the caller.
UPSTREAM_IPPO_CONFIG: Dict[str, Any] = {
    "ALG": "IPPO",
    "NUM_STEPS": 64,
    "NUM_ENVS": 1024,
    "NUM_EVAL_EPISODES": 32,
    "UPDATE_EPOCHS": 16,
    "NUM_MINIBATCHES": 16,
    "ANNEAL_LR": False,
    "LR": 1.37e-4,
    "ENT_COEF": 0.0017260804,
    "CLIP_EPS": 0.225,
    "SCALE_CLIP_EPS": False,
    "RATIO_CLIP_EPS": False,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "VF_COEF": 1.0,
    "MAX_GRAD_NORM": 0.5,
    "ADAM_EPS": 1e-8,
    "ADVANTAGE_UNROLL_DEPTH": 8,
    # UPSTREAM'S NETWORK GROUP, `assistax/baselines/IPPO/config/network/ff_nps.yaml`
    # at a7d94f4e, selected by `defaults: - network: ff_nps` in `config/ippo.yaml`
    # and COPIED WITH ITS VALUES. READ BY UPSTREAM, not by us: `MultiActorCritic`
    # in `ippo_ff_nps.py` does `config["network"]["activation"]` (:68),
    # `["actor_hidden_dim"]` (:77, :84) and `["critic_hidden_dim"]` (:108, :115).
    # Without this block `make_train` raises `KeyError('network')`.
    "network": {"name": "ff_nps", "recurrent": False, "agent_param_sharing": False,
                "actor_hidden_dim": 128, "critic_hidden_dim": 128,
                "activation": "relu"},
}


def upstream_ippo_config(*, env_name: str, env_kwargs: Mapping[str, Any],
                         total_timesteps: int, zoo_path: str,
                         num_envs: Optional[int] = None,
                         num_steps: Optional[int] = None) -> Dict[str, Any]:
    """`UPSTREAM_IPPO_CONFIG` plus the per-run fields `make_train` reads.

    `NUM_ENVS` and `NUM_STEPS` are overridable because a smoke run cannot
    afford 1024 envs, and the override is EXPLICIT for that reason -- a
    default that quietly shrank them would make a cheap run look like the
    paper's regime. Every override is recorded on the seed row.

    Refuses a budget smaller than one update rather than rounding up to one:
    `NUM_UPDATES = TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS` floors to 0 and
    upstream's `jax.lax.scan` over zero updates returns the initial params
    with no error at all -- an untrained policy reported as a trained one.
    """
    config = dict(UPSTREAM_IPPO_CONFIG)
    if num_envs is not None:
        config["NUM_ENVS"] = int(num_envs)
    if num_steps is not None:
        config["NUM_STEPS"] = int(num_steps)
    per_update = config["NUM_STEPS"] * config["NUM_ENVS"]
    if int(total_timesteps) < per_update:
        raise ValueError(
            f"TOTAL_TIMESTEPS={int(total_timesteps)} is less than one update "
            f"({config['NUM_STEPS']} steps x {config['NUM_ENVS']} envs = "
            f"{per_update}). NUM_UPDATES would floor to 0 and upstream's scan "
            f"would return the INITIAL parameters without erroring -- an "
            f"untrained policy reported as trained. Raise the budget or lower "
            f"NUM_ENVS/NUM_STEPS explicitly.")
    config.update({
        "ENV_NAME": env_name,
        "ENV_KWARGS": dict(env_kwargs),
        "TOTAL_TIMESTEPS": int(total_timesteps),
        # READ BY UPSTREAM, NOT BY US, and live rather than decorative:
        # `ippo_ff_nps.py:257` and `:731` both do
        # `ZooManager(config["ZOO_PATH"])`. Noted because a grep of
        # `bird/` alone finds no reader and invites the conclusion that
        # the key is dead -- the reader is the vendored package.
        "ZOO_PATH": str(zoo_path),
    })
    return config
