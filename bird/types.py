"""Typed records passed between the six stages.

These are the API of the whole project. Stage functions have fixed signatures
(see `bird.py`) and communicate only through the records below, so no stage ever
reaches backwards into another stage's internals.

The 4->5->6 contract is the one that matters most:
`evaluate` produces a `CandidateReport` carrying BOTH a scalar (`fitness`) and
prose (`feedback`). `select` consumes only the scalar; `update` consumes only
the prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Stage 1 output
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    """A generated reward program (optionally with co-designed artifacts)."""

    cand_id: str
    iteration: int
    reward_code: str
    # --- co-designed artifacts (`generate.co_design.*`) ---
    observation_code: Optional[str] = None
    dr_config: Optional[Dict[str, Any]] = None
    # --- provenance ---
    parent_id: Optional[str] = None
    raw_response: str = ""
    nl_spec: str = ""  # `generate.output.two_stage_nl_then_code` stage-1 prose
    prompt_messages: List[Dict[str, str]] = field(default_factory=list)
    # --- structured reward representation (`component_dict_plus_weights`) ---
    component_names: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    subtask_index: Dict[str, int] = field(default_factory=dict)
    # --- verification outcome, filled by stage 2 ---
    valid: bool = True
    screened_out: bool = False
    failure: str = ""  # "" means no failure; non-empty is the reason
    failure_kind: str = ""  # "" | "invalid" | "screened"
    verify_records: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def trainable(self) -> bool:
        return self.valid and not self.screened_out

    def failed(self, reason: str, kind: str = "invalid") -> "Candidate":
        return replace(self, valid=False, failure=reason, failure_kind=kind)


# --------------------------------------------------------------------------
# Stage 3 output
# --------------------------------------------------------------------------


@dataclass
class Trajectory:
    """One rollout. `states`/`actions` are kept for screens that re-score them."""

    states: Any = None
    actions: Any = None
    rewards: Sequence[float] = field(default_factory=list)
    #: Per STEP: `component_values[k][t]` is what the reward claimed for `k` at
    #: step `t`, `None` where it claimed nothing there, so index `t` pairs with
    #: `rewards[t]` (the claimed-vs-measured contract; see
    #: `observability.save_trajectory_trace`).
    component_values: Dict[str, Sequence[Optional[float]]] = field(default_factory=dict)
    success: bool = False
    length: int = 0
    ret: float = 0.0
    video_path: Optional[str] = None

    @property
    def mean_per_step_return(self) -> float:
        """Length-normalised return. Both TAC and TPE specify this, for the same
        reason: successful episodes terminate early, so cumulative return is
        length-biased toward failures."""
        return float(self.ret) / max(int(self.length), 1)


@dataclass
class TrainResult:
    """The output of one inner-loop run `pi = A_M(R)`."""

    cand_id: str
    candidate: Candidate
    trained: bool = True  # False when stage 2 skipped training (CARD's TPE fork)
    skip_reason: str = ""
    seed_metrics: List[Dict[str, Any]] = field(default_factory=list)
    checkpoints: List[Dict[str, float]] = field(default_factory=list)
    component_traces: Dict[str, List[float]] = field(default_factory=dict)
    # THE LEARNER'S OWN PER-EPOCH NAMED SERIES -- Eureka's `tensorboard_logs`
    # equivalent, and the input its reward reflection is built from
    # (`eureka.py:236-268` reads the scalar log, strides it at
    # `max_iterations // 10` and reports Max/Mean/Min over the full series).
    # Distinct from `component_traces` in both axes and both matter: this is
    # keyed by EVERY scalar the learner logged -- `consecutive_successes`,
    # `gt_reward`, `gpt_reward` and each reward component -- and it is sampled
    # per TRAINING EPOCH, where `component_traces` is components only at the
    # held-out checkpoint stride. `evaluate.feedback.series_source:
    # training_epochs` reads this; `checkpoints` reads the curve.
    #
    # EMPTY ON EVERY BACKEND THAT DOES NOT FILL IT, and that is a statement
    # rather than a gap: no backend in this release has a per-epoch named log
    # to report (`train.log: epoch_scalars`), and the renderer says so and falls
    # back rather than drawing an empty block -- the `gt_reward_curve`
    # convention, where every entry None means "no channel" and never zero.
    epoch_series: Dict[str, List[float]] = field(default_factory=dict)
    trajectories: List[Trajectory] = field(default_factory=list)
    # The dedicated TPE-store pool (CARD's `--trajectory_num` check pool,
    # collected at `verify.tpe.trajectories_per_iteration`, distinct from the
    # `evaluate.rollouts_per_candidate` feedback pool above -- the release keeps
    # the two pools in two envs, metaworld_exp_one_step.py:230-232). Empty on
    # every config whose
    # `update.memory.trajectory_store` is `none`. Ships home for free under
    # candidate parallelism: the fork pickles the whole TrainResult.
    store_trajectories: List[Trajectory] = field(default_factory=list)
    policy_ref: Optional[str] = None
    replay_ref: Optional[str] = None
    # WHERE THIS POLICY STARTED, as resolved by `train.init` for THIS candidate
    # (`training._training_init`): `parent` (its lineage parent's stored
    # policy, `warm_start_from_parent`), `similar` (the stored policy whose
    # reward is structurally closest, `warm_start_from_similar`), `best` (the
    # incumbent's), `bc_prior`, `continuation` (its OWN last policy: a later
    # slice of `train.interaction: shared_population` or a later rung of
    # `train.pruning: successive_halving_pool`), or `scratch` -- which is also
    # what an evicted or never-written ref resolves to, so the field names what
    # was loadable and not what was asked for. `init_from_cand_id` is the
    # candidate whose policy it was. `seed_metrics[*].warm_started_from` is
    # the per-seed complement (did the blob actually load into this learner);
    # together they make a warm start that fell through distinguishable from
    # the ablation in the artifact.
    init_source: str = ""
    #: ROSKA's fusion search AS EXECUTED (`train.init: fused_warm_start`):
    #: which search ran, the alpha it chose, every alpha it probed with that
    #: probe's score, and `probe_env_steps` -- the env steps the search itself
    #: spent, which the paper's TTS arithmetic counts and which are invisible
    #: in `env_steps_used` BESIDE IT: that field is this candidate's own
    #: training, and the probes are not it. They are not invisible in
    #: `budget.json`, which counts them in `env_steps` and again, on their own,
    #: in `rollout_env_steps` -- the two artifacts count different things on
    #: purpose and `tests/test_roska_fusion.py` pins both halves. Empty for
    #: every other init mode, and
    #: empty in round 1, where there is no policy to fuse with and the paper
    #: runs no search. Without it a searched run and a fixed-alpha run leave
    #: identical artifacts.
    fusion: Dict[str, Any] = field(default_factory=dict)
    init_from_cand_id: Optional[str] = None
    # `warm_start_from_similar` only: how alike the donor's reward and this
    # one are (`selection._structural_similarity`, [0, 1]); None otherwise.
    init_similarity: Optional[float] = None
    # The reference reward's checkpoint curve (`checkpoints[i]["gt_return"]`).
    # Every entry is None on a task whose adapter has no reference reward
    # (`EnvAdapter.has_reference_reward`); the consumers -- `pearson_curve`,
    # `phases._reference_score` -- treat that as "no channel", never as zero.
    gt_reward_curve: List[Optional[float]] = field(default_factory=list)
    wallclock_s: float = 0.0
    env_steps_used: int = 0
    error: str = ""
    # `train.pruning` stopped at least one seed before its step budget was
    # spent. It is a THIRD outcome and must not collapse into either of the
    # other two: `error` non-empty means the run broke, `pruned` means it was
    # deliberately cut short and its final fitness is therefore a lower bound
    # on what the reward would have reached. Selecting on a pruned candidate's
    # fitness as if it were a completed one is exactly the confusion the
    # `failure`/`failure_kind` split exists to prevent one level up, so the
    # same rule applies here: keep the populations separable in the artifact.
    # Per-seed detail lives in `seed_metrics[i]["pruned_at_round"]`.
    pruned: bool = False
    # `train.timeout_s` cut at least one seed off mid-training. Unlike `pruned`
    # this sets `trained=False` and fills `error`, because nobody CHOSE to stop
    # it -- but `error` alone is not enough to act on. A reward that raises on
    # the first transition and a healthy reward on a wedged node both arrive as
    # `trained=False` with a string, and the remedy is opposite: one is a bad
    # program, the other is a bad node and the candidate deserves a re-run. The
    # curve a timed-out seed leaves behind is TRUNCATED, so any fitness computed
    # off it is a lower bound on a reward that was never allowed to finish --
    # which must never be indistinguishable from a reward that genuinely
    # learned less.
    timed_out: bool = False


# --------------------------------------------------------------------------
# Stage 4 output
# --------------------------------------------------------------------------


@dataclass
class CandidateReport:
    """Stage 4's single typed record. Read `fitness` in stage 5, `feedback` in 6."""

    cand_id: str
    candidate: Candidate
    result: TrainResult
    # --- the scalar (stage 5 only) ---
    fitness: Optional[float] = None  # None when no ranking scalar exists (CARD)
    fitness_source: str = "none"
    per_seed_fitness: List[float] = field(default_factory=list)
    # --- the prose (stage 6 only) ---
    feedback: str = ""
    feedback_channel: str = "none"  # e.g. "process+trajectory" | "preference"
    # --- auxiliary evidence ---
    subtask_scores: Dict[str, float] = field(default_factory=dict)
    subtask_rationales: Dict[str, str] = field(default_factory=dict)
    # RDA's describe-behaviour output (App. 7.3 step (i)): what the agent DID
    # during the subtask, kept apart from `subtask_rationales` (the reasoning
    # that justifies the score). Empty everywhere the judge was not asked --
    # `artifacts.save_report` keeps absent distinct from "" for exactly this
    # field.
    subtask_behaviors: Dict[str, str] = field(default_factory=dict)
    similarity: Optional[float] = None
    safety: Dict[str, float] = field(default_factory=dict)
    descriptors: Dict[str, float] = field(default_factory=dict)  # archive coords
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_failure(self) -> bool:
        return not self.candidate.valid


@dataclass
class Preference:
    """One pairwise comparison; the unit of GT's `D_pref`."""

    left_id: str
    right_id: str
    label: int  # 1 = left preferred, 0 = right preferred
    left_traj: Optional[Trajectory] = None
    right_traj: Optional[Trajectory] = None
    source: str = "vlm"
    iteration: int = -1
    # A half-open step range within each trajectory, or None for "the whole
    # rollout" -- which is what every comparator in `preferences.py` produces.
    # R* compares SEGMENTS: "quality differences typically exist only in
    # specific segments of a trajectory, and the above methods struggle to
    # effectively extract these fine-grained segments" (R* S4.3, p.5), so a
    # preference over a whole rollout is precisely the thing it argues against.
    # None and (0, length) are NOT the same statement: the first says nobody
    # asked about spans, the second says a labeller looked and chose the whole
    # episode.
    left_span: Optional[Tuple[int, int]] = None
    right_span: Optional[Tuple[int, int]] = None
    # True on BOTH of the mirrored records `run_preferences` writes for a tie
    # (one with label 1, one with label 0). `label` keeps its frozen {0, 1}
    # contract, so `_tally`, `aggregate_copeland`, `_write_back`'s win/loss
    # counts, `preference_log.side_of` and `selection._bt_strengths` are
    # untouched; only an aggregator that folds a tie into one game reads this
    # (`elo_raw`, REvolve's single f = 0.5 update, tex:999-1004). A record
    # written before the field existed decodes to False (`checkpoint.decode`
    # fills a missing field from its default), so an older tie replays as two
    # decisive games.
    tie: bool = False


# --------------------------------------------------------------------------
# Stage 5 output
# --------------------------------------------------------------------------


@dataclass
class Selection:
    """Stage 5's verdict for this iteration."""

    winners: List[CandidateReport] = field(default_factory=list)
    losers: List[CandidateReport] = field(default_factory=list)
    rule: str = "none"
    contested: bool = True  # False when K == 1 and no contest can exist
    tie_broken: bool = False
    tied_ids: List[str] = field(default_factory=list)
    improved_over_incumbent: Optional[bool] = None
    notes: str = ""

    @property
    def winner(self) -> Optional[CandidateReport]:
        return self.winners[0] if self.winners else None
