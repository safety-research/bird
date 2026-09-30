"""Budget accounting -- a first-class output, not a log line.

These methods differ by orders of magnitude in
inner-loop cost, so a cross-method comparison without a cost column is
meaningless. In particular CARD's *entire* contribution is RL runs it did NOT
launch, which is invisible unless `policy_trainings_skipped` is counted.
"""

from __future__ import annotations

from contextlib import contextmanager

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    """Counters + optional hard caps from `budget.*`."""

    max_policy_trainings: Optional[int] = None
    max_llm_calls: Optional[int] = None
    max_gpu_hours: Optional[float] = None

    policy_trainings: int = 0
    #: RL runs a screen, an allocator or the loop shape DECIDED not to launch --
    #: CARD's whole pitch (TPE), LIMEN's cascade, LaRes's never-selected
    #: members, and the generate-only last iteration of `loop.termination:
    #: fixed_generations` (CARD again: Alg. 1 returns R_N straight from the
    #: Coder, so its training is one the method never launches). A candidate
    #: that never compiled or forfeited its slot is NOT one of these: that is a
    #: defect, and it is counted in `candidates_invalid` below. Pooled here, an
    #: Eureka run whose provider forfeited three slots would report three
    #: "skipped trainings" in the very column CARD's saving is read from (per
    #: candidate, `failure_kind` keeps them apart; the summary must too).
    policy_trainings_skipped: int = 0
    #: Candidates stage 3 could not launch because they were `invalid`
    #: (`failure_kind: "invalid"`: no code, unparseable, failed verification).
    #: The other half of the split above: a saving and a defect are different
    #: numbers, and a cross-method cost table reading `policy_trainings_skipped`
    #: as "RL runs the method chose not to run" must not be crediting parse
    #: failures.
    candidates_invalid: int = 0
    env_steps: int = 0
    #: The part of `env_steps` that is NOT a candidate's own training: today the
    #: ROSKA fusion probes (`train.fusion.ratio_search: sc_bo`) and card's TPE
    #: collection. Recorded separately because `env_steps` is one aggregate and
    #: the summed `train_result.env_steps_used` is another -- without a third
    #: number the gap between them cannot be attributed, and an unattributed gap
    #: is how a large collection cost goes unreported (see
    #: `record_rollout_steps`). A reader
    #: should not have to subtract two artifacts to learn what a method's
    #: largest cost term was.
    rollout_env_steps: int = 0
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    #: A SPLIT of `llm_prompt_tokens`, never an addition to it (`record_llm`).
    #: `llm.prompt_caching` is transparent to the model but not to the bill --
    #: a cache read is ~0.1x an ordinary input token and a cache write ~1.25x --
    #: so without these two the cost column cannot tell a cheap method from a
    #: well-cached one, which is precisely the comparison this file exists to
    #: make.
    llm_cache_read_tokens: int = 0
    llm_cache_write_tokens: int = 0
    vlm_calls: int = 0
    human_queries: int = 0
    verify_resamples: int = 0
    #: Provider round-trips that produced NO usable sample: the model refused
    #: (`stop_reason: refusal`, after the client's own retries) or was cut at
    #: `max_tokens`. Both are counted in `llm_calls` and both hand the generate
    #: stage an empty string, which it records as a forfeited slot -- so without
    #: these two a run the provider refused wholesale is indistinguishable from
    #: one whose model wrote unparseable code.
    llm_refusals: int = 0
    llm_truncations: int = 0
    #: Comparisons a judge made with NO frames available, while the config asked
    #: for `videos` in `evaluate.preferences.artifacts`. For `rda` and `gt`
    #: the whole distinguishing mechanism is a model looking at rollouts, so a
    #: comparison made blind still returns a preference, still produces a
    #: fitness, and is indistinguishable in the artifact from one made with
    #: eyes. That is the failure this counts: not an error, a number that means
    #: something other than what it claims. Nonzero here invalidates a
    #: video-grounded method's result even though nothing crashed.
    blind_comparisons: int = 0
    #: (rollout, subtask) judgments `evaluate.fitness.source: vlm_score` made
    #: with NO frames attached, while `evaluate.artifacts` asked for `videos`.
    #: The sibling of `blind_comparisons` above and justified by the same
    #: sentence: RDA's whole mechanism is a model watching a rollout, and a
    #: judgment made from a text digest still returns a score, still becomes a
    #: fitness, and is indistinguishable in the artifact from one made with
    #: eyes. Nonzero here means the reported ranking is partly or wholly
    #: text-only, whatever the config says it is.
    blind_judgments: int = 0
    #: Slots a resumed run could NOT restore (`loop.resume_from: auto_degraded`),
    #: one per dropped slot. Modelled directly on `blind_comparisons` and
    #: justified by the same sentence: not an error, a number that means
    #: something other than what it claims. A run that resumed without its
    #: replay buffer ran the ablation, not the method, and a cross-method table
    #: can filter on this where it cannot filter on a log line.
    resume_degradations: int = 0
    #: Work the leg that died had already paid for after its last checkpoint and
    #: which the resumed leg re-does. Recovered EXACTLY from the journal's
    #: `train` events, which already carry `seeds` and `env_steps`. Kept apart
    #: from `policy_trainings` for the same reason `policy_trainings_skipped` is:
    #: `policy_trainings` must stay "trainings this search's result depends on"
    #: so caps and the cross-method cost column keep meaning that. Add the two
    #: for machine cost; read the first alone to compare methods.
    resume_discarded_trainings: int = 0
    resume_discarded_env_steps: int = 0
    #: Times an operator ran this search PAST the rule that stopped it, with
    #: `--full-iterations`. Its sibling above counts what a resume LOST; this
    #: counts what one ADDED, and both are here for the same reason: the run's
    #: own `config.resolved.yaml` cannot explain the number in the journal.
    #: Nonzero means the iterations above the early-stop point are spend the
    #: configured `loop.termination` declined to authorise -- so an extended run
    #: and one that reached `loop.n_iterations` under its own rule are different
    #: treatments, and this counter is what separates them in the cost column.
    resume_extensions: int = 0
    #: Backend calls that CONTINUED a training rather than starting one.
    #: `train.interaction: shared_population` spends one pooled budget in waves,
    #: so a candidate's single training arrives as several backend calls, each
    #: resuming the last. Counting those as separate trainings would inflate
    #: `policy_trainings` by the slice count and -- worse -- trip
    #: `max_policy_trainings` part way through a round that is within its budget,
    #: which is this repo's recurring "a cap bites and does not announce itself"
    #: failure. They are counted HERE instead, so the slicing stays visible
    #: without being charged twice.
    training_slices: int = 0
    gpu_seconds: float = 0.0
    wallclock_start: float = field(default_factory=time.time)

    # -- record --

    # NB `_in_slice` is deliberately NOT a dataclass field. `counter_names()`
    # is derived from the fields, and every counter is reported in
    # `budget.json` -- correctly, since a counter nobody can see is one nobody
    # can act on. A private latch is not a counter, so it lives outside the
    # field list rather than being special-cased out of it.

    @contextmanager
    def slice_mode(self):
        """Inside this block a `record_training` counts STEPS but not TRAININGS.

        Two callers: `train.interaction: shared_population`, which charges the
        round's trainings itself, once per candidate, after the waves are done;
        and `train.pruning: successive_halving_pool` (`rounds.py`), whose later
        rungs continue a training already counted at rung 0.
        """
        prev = getattr(self, "_in_slice", False)
        self._in_slice = True
        try:
            yield
        finally:
            self._in_slice = prev

    @contextmanager
    def uncounted_llm(self):
        """Inside this block a `record_llm` records NOTHING.

        `verify.repairs_count_against_budget: false` is the one caller. The
        LLM client contract (`bird/llm/base.py`) is that a client records its
        own round-trip and no stage calls `record_llm`, so the only way for a
        stage to EXCLUDE a call it makes through a client is to latch the
        counter around the call -- the same shape as `slice_mode`, and a
        private latch for the same reason. `verify_resamples` is not under
        the latch: it counts repairs, not LLM cost, and stays honest either way.
        """
        prev = getattr(self, "_llm_uncounted", False)
        self._llm_uncounted = True
        try:
            yield
        finally:
            self._llm_uncounted = prev

    def record_training(self, env_steps: int = 0, gpu_seconds: float = 0.0) -> None:
        if getattr(self, "_in_slice", False):
            self.training_slices += 1
        else:
            self.policy_trainings += 1
        self.env_steps += int(env_steps)
        self.gpu_seconds += float(gpu_seconds)
        self._check()

    def record_rollout_steps(self, env_steps: int = 0) -> None:
        """Env interaction that is NOT a training: the dedicated TPE-store
        collection (`verify.tpe.trajectories_per_iteration`). Those rollouts
        also reach `TrainResult.env_steps_used`; left out of here,
        `budget.json` would under-report CARD's env interaction ~4x while
        `train_result.json` counted it, two artifacts silently disagreeing
        about the same quantity. Steps only: no `policy_trainings` increment
        (collection is not an RL run) and no cap check moves, since no cap
        covers env steps."""
        self.env_steps += int(env_steps)
        self.rollout_env_steps += int(env_steps)

    def record_skip(self) -> None:
        """A training NOT launched by decision: a screen's verdict or an
        allocator's. Never for an invalid candidate -- see `record_invalid`."""
        self.policy_trainings_skipped += 1

    def record_invalid(self) -> None:
        """A candidate stage 3 could not launch: it never compiled or forfeited
        its slot (`failure_kind: "invalid"`). A defect, not a saving."""
        self.candidates_invalid += 1

    def record_llm(self, prompt_tokens: int = 0, completion_tokens: int = 0,
                   vlm: bool = False, cache_read_tokens: int = 0,
                   cache_write_tokens: int = 0, refused: bool = False,
                   truncated: bool = False) -> None:
        """`prompt_tokens` is the whole input; the two cache figures SPLIT it.

        Not added to it. The Anthropic API reports `input_tokens`,
        `cache_read_input_tokens` and `cache_creation_input_tokens` as three
        DISJOINT buckets, so the client sums them before calling this and passes
        the two cached parts through for the cost column. Adding them again here
        is the double-count to avoid; dropping them is the ~4x under-report.
        """
        if getattr(self, "_llm_uncounted", False):
            return
        self.llm_calls += 1
        if vlm:
            self.vlm_calls += 1
        self.llm_prompt_tokens += int(prompt_tokens)
        self.llm_completion_tokens += int(completion_tokens)
        self.llm_cache_read_tokens += int(cache_read_tokens)
        self.llm_cache_write_tokens += int(cache_write_tokens)
        self.llm_refusals += int(bool(refused))
        self.llm_truncations += int(bool(truncated))
        self._check()

    def record_human(self, n: int = 1) -> None:
        self.human_queries += n

    def record_resample(self, n: int = 1) -> None:
        self.verify_resamples += n

    def record_blind_comparison(self, n: int = 1) -> None:
        self.blind_comparisons += n

    def record_blind_judgment(self, n: int = 1) -> None:
        self.blind_judgments += int(n)

    # -- crossing a process boundary (`train.candidate_parallelism: parallel`) --
    #
    # A forked worker inherits a *copy* of this object, so every
    # `record_training` it makes lands in an address space the parent never
    # sees. Left alone that is not a slow counter, it is a silent one:
    # `budget.json` would report zero trainings and `max_policy_trainings` /
    # `max_gpu_hours` would stop firing altogether, i.e. a capped run would
    # quietly become uncapped. The fix is a delta merge, and it is a delta
    # rather than a lock because there is nothing shared to lock -- the child's
    # counters are a private copy that dies with it.
    #
    # Addition is commutative, so the totals do not depend on merge order. The
    # one order-sensitive thing is *where* `BudgetExceeded` fires, which is why
    # `bird/components/training.py::_parallel` merges strictly in candidate
    # order and calls `_check()` after each candidate.
    #
    # `max_*` are excluded (they are configuration, not counts) and so is
    # `wallclock_start` (a timestamp; subtracting two of them is meaningless).

    #: The fields a worker can move. Derived from the dataclass rather than
    #: listed, so a new counter is merged automatically instead of being
    #: dropped by whoever forgot to extend a hard-coded tuple.
    @classmethod
    def counter_names(cls) -> tuple:
        return tuple(k for k in cls.__dataclass_fields__
                     if not k.startswith("max_") and k != "wallclock_start")

    @classmethod
    def cap_names(cls) -> tuple:
        """The cap fields, DERIVED, for the same reason `counter_names` is.

        Each maps 1:1 onto the config key `budget.<name>`, which is what
        `configs/_default.yaml` carries. Deriving rather than listing means a
        fourth cap is honoured everywhere the moment it is a field; the
        listing version had to be edited in every construction site, and a
        site that was missed does not fail -- it builds a Budget whose new
        cap is None, which is a worker running UNCAPPED and reporting normal.
        """
        return tuple(k for k in cls.__dataclass_fields__ if k.startswith("max_"))

    @classmethod
    def from_config(cls, cfg: Any) -> "Budget":
        """The one place a Budget's caps are read out of a config.

        There were two: `bird.py`'s run setup, and -- once candidate workers
        could be SPAWNED rather than forked -- the worker's own rehydration,
        because a spawned worker has no inherited Budget object to copy. A
        forked worker never needed this at all, which is why the duplication
        only appeared when the second process-start path did.

        ADDING A `max_` FIELD NOW REQUIRES A CONFIG KEY. `cap_names()` reads
        the dataclass, so a new cap with no `budget.<name>` in
        `configs/_default.yaml` raises here at run start. That is the
        intended direction: the alternative -- skipping keys that are
        missing -- is a cap that silently never binds, which is the failure
        `counter_names`'s own comment describes one field over. Loud at
        startup beats uncapped at runtime, and a cap nothing can configure
        is not a cap.
        """
        return cls(**{k: cfg[f"budget.{k}"] for k in cls.cap_names()})

    def snapshot(self) -> Dict[str, float]:
        """Counter values right now; pair with `delta_since` around a fork."""
        return {k: getattr(self, k) for k in self.counter_names()}

    def delta_since(self, before: Dict[str, float]) -> Dict[str, float]:
        return {k: getattr(self, k) - before[k] for k in self.counter_names()
                if getattr(self, k) != before[k]}

    def merge_delta(self, delta: Dict[str, float]) -> None:
        """Add one worker's counter deltas, then re-check the caps.

        Raises `BudgetExceeded` exactly where the sequential run would have,
        *to candidate granularity*: a worker cannot see the counts of the
        candidates forked beside it, so it enforces the cap against its own
        fork-time snapshot and can therefore overrun by at most the seeds of
        the one candidate it owns. With `train.seeds_per_candidate: 1` -- every
        published config in `configs/` except the GT points -- the two are
        exactly equal, which is what `tests/test_parallelism.py` pins.
        """
        for k, v in delta.items():
            setattr(self, k, getattr(self, k) + v)
        self._check()

    # -- crossing a PROCESS LIFETIME (`loop.resume_from`) --

    def restore(self, snapshot: Dict[str, float], prior_elapsed_s: float = 0.0) -> None:
        """Adopt a checkpoint's counters. **Sets, never adds.**

        `merge_delta` adds because it folds a fork of the *same* process into a
        live parent. Here the target is a fresh `Budget` at zero and the
        snapshot is the authoritative total, so setting is IDEMPOTENT: a requeue
        loop that adopts the same directory five times cannot double-count.
        Adding would. Nor can the skipped iterations be counted twice, and that
        is structural rather than careful -- iterations 0..k are never
        re-executed, so nothing re-records them.

        `_check()` fires immediately, on purpose: a run that resumes already
        past `budget.max_llm_calls` should raise at second one rather than pay
        for one more 18-minute iteration to find out. The caps come from the
        resumed config and are never read from the checkpoint -- they are
        configuration, not counts.

        `wallclock_start` is BACK-DATED rather than stored as a separate
        `prior_wallclock_s` field, so `counter_names()` (which already excludes
        it) is untouched and `wallclock_s` automatically keeps meaning "wall
        time this search has consumed" -- which is what `budget.json` claims.
        """
        names = set(self.counter_names())
        for k, v in (snapshot or {}).items():
            if k in names:
                setattr(self, k, v)
        self.wallclock_start = time.time() - float(prior_elapsed_s or 0.0)
        self._check()

    def record_resume_degradation(self, n: int = 1) -> None:
        self.resume_degradations += n

    def record_resume_extension(self, n: int = 1) -> None:
        self.resume_extensions += n

    def record_discarded(self, trainings: int = 0, env_steps: int = 0) -> None:
        self.resume_discarded_trainings += int(trainings)
        self.resume_discarded_env_steps += int(env_steps)

    # -- query --

    @property
    def total_tokens(self) -> int:
        return self.llm_prompt_tokens + self.llm_completion_tokens

    @property
    def wallclock_s(self) -> float:
        return time.time() - self.wallclock_start

    def exhausted(self) -> bool:
        try:
            self._check()
        except BudgetExceeded:
            return True
        return False

    #: The two COUNTER caps, each as (counter attribute, cap attribute).
    #: `max_gpu_hours` is deliberately not here: its counter is seconds and
    #: its cap is hours, so it needs the unit conversion `would_exceed` does
    #: for it separately rather than a third row that looks uniform and is
    #: not.
    _CAPS = (("policy_trainings", "max_policy_trainings"),
             ("llm_calls", "max_llm_calls"))

    def would_exceed(self, **increments: float) -> Optional[str]:
        """Would these increments cross a cap? Returns the reason, or None.

        THE CAPS BIND BEFORE THE WORK. Every `record_*` increments and only
        then calls `_check`, which raises on `>`, so checking only there lets a
        cap of N permit N+1: the crossing unit runs in full and the run stops
        afterwards. `max_policy_trainings`, `max_llm_calls` and
        `max_gpu_hours` all have that shape and all three mean "caps the
        total", so this is one off-by-one in three places rather than three
        semantics.

        ASK BEFORE DOING THE WORK. The fix cannot be `>` -> `>=` inside
        `_check`: `record_training` is called AFTER the training, with
        `gpu_seconds = time.monotonic() - t0`, so a flipped predicate would
        raise after the work had run and before it was counted -- N+1
        trainings of compute reported as N, which is worse still. `_check`
        keeps `>` as the backstop for paths that cannot ask first, such as a
        forked child folding its delta.
        """
        for counter, cap_name in self._CAPS:
            cap = getattr(self, cap_name)
            add = increments.get(counter, 0)
            if cap is not None and add and getattr(self, counter) + add > cap:
                return (f"{counter} {getattr(self, counter)} + {add} would "
                        f"exceed cap {cap}")
        hours = increments.get("gpu_seconds", 0) / 3600.0
        if self.max_gpu_hours is not None and hours and \
                self.gpu_seconds / 3600.0 + hours > self.max_gpu_hours:
            return (f"gpu hours {self.gpu_seconds / 3600.0:.2f} + {hours:.2f} "
                    f"would exceed cap {self.max_gpu_hours}")
        return None

    def gpu_hours_exhausted(self) -> Optional[str]:
        """Is `max_gpu_hours` already spent? Returns the reason, or None.

        SEPARATE FROM `would_exceed`, and the asymmetry is the honest part.
        A training's duration is not knowable before it runs, so "would this
        unit cross the cap" is a question nobody can answer in advance --
        unlike a training (one) or an API call (one). What CAN be answered is
        whether there is any budget left at all, so the rule this enforces is
        "do not START a unit once the cap is spent" rather than "do not cross
        it", and a unit already running may still overrun by its own length.
        Saying so here rather than pretending the two caps behave alike.

        `>=`, not `>`: at exactly the cap there is nothing left to spend, and
        starting a unit that can only overrun is the thing being prevented.
        """
        if self.max_gpu_hours is None:
            return None
        if self.gpu_seconds / 3600.0 >= self.max_gpu_hours:
            return (f"gpu hours {self.gpu_seconds / 3600.0:.2f} has reached "
                    f"cap {self.max_gpu_hours}; no unit may start")
        return None

    def would_exceed_training(self, n: int = 1) -> Optional[str]:
        """`would_exceed` for the common case: n more policy trainings."""
        return self.would_exceed(policy_trainings=n)

    def _check(self) -> None:
        if self.max_policy_trainings is not None and self.policy_trainings > self.max_policy_trainings:
            raise BudgetExceeded(
                f"policy trainings {self.policy_trainings} > cap {self.max_policy_trainings}")
        if self.max_llm_calls is not None and self.llm_calls > self.max_llm_calls:
            raise BudgetExceeded(f"llm calls {self.llm_calls} > cap {self.max_llm_calls}")
        if self.max_gpu_hours is not None and self.gpu_seconds / 3600.0 > self.max_gpu_hours:
            raise BudgetExceeded(
                f"gpu hours {self.gpu_seconds / 3600.0:.2f} > cap {self.max_gpu_hours}")

    def report(self) -> Dict[str, float]:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("max_")}
        d.pop("wallclock_start", None)
        d["total_tokens"] = self.total_tokens
        d["wallclock_s"] = round(self.wallclock_s, 2)
        return d
