#!/usr/bin/env python3
"""BIRD -- Benchmark for Iterative Reward Design.

    python bird.py --config eureka --profile tester
    python bird.py --config configs/methods/rda.yaml --print-config

The six stages below are the whole algorithm. Their bodies do exactly two
things: read config values, and dispatch to registry components keyed by those
values. There is deliberately no `if cfg.name == "eureka"` anywhere in this
file or any other -- if a method needs behaviour that cannot be reached from a
config value, the schema is missing a knob and *that* is the bug to fix.

Stage contract:

    generate(ctx, state)             -> list[Candidate]
    verify(ctx, state, candidates)   -> list[Candidate]     (identity if disabled)
    train(ctx, state, candidates)    -> list[TrainResult]
    evaluate(ctx, state, results)    -> list[CandidateReport]
    select(ctx, state, reports)      -> Selection
    update(ctx, state, selection)    -> RunState

`CandidateReport` carries both a scalar (`.fitness`) and prose (`.feedback`).
Stage 5 reads only the scalar; stage 6 reads only the prose. No stage reaches
backwards.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bird import checkpoint, preference_log, registry
from bird.artifacts import RunDir, execute_rate, setup_logging, summarise_reports
from bird.budget import Budget, BudgetExceeded
from bird.components.rounds import training_rounds
from bird.config import Config, ConfigError, load, parse_override, parse_overrides
from bird.context import Context
from bird.observability import make_tracker, record_rollouts
from bird.state import FINAL_ARTIFACT_RULES, RunState
from bird.types import Candidate, CandidateReport, Selection, TrainResult

log = logging.getLogger("bird")


# ==========================================================================
# Stage 1 -- Reward Generation
# ==========================================================================


def generate(ctx: Context, state: RunState) -> List[Candidate]:
    """Produce this iteration's candidate pool."""
    cfg = ctx.cfg
    backend = registry.get("generator_backend", cfg["generate.generator_backend"])
    sampler = registry.get("sampling_mode", cfg["generate.sampling_mode"])

    # Subtask decomposition runs once per search, not once per iteration (RDA).
    if cfg["generate.decomposition.enabled"] and not state.subtasks:
        decompose = registry.get("phase", "decompose")
        state.subtasks = decompose(ctx, state)
        ctx.event("decompose", n_subtasks=len(state.subtasks), subtasks=state.subtasks)

    # The curriculum is authored ONCE per search, on the same lazy guard and for
    # the same reason as the decomposition above: it must exist before the first
    # prompt is built, and a `pre:` entry the operator has to remember would make
    # its absence look like a config choice. Authoring AFTER decomposition is
    # deliberate -- `author: subtask_list` orders that list.
    if cfg["loop.curriculum.enabled"] and getattr(state, "curriculum", None) is None:
        author = registry.get("curriculum_author", cfg["loop.curriculum.author"])
        state.curriculum = author(ctx, state)
        ctx.event("curriculum_author", author=cfg["loop.curriculum.author"],
                  **state.curriculum.summary())

    # The critic population is authored ONCE per search, on the same lazy guard
    # as the decomposition and the curriculum above -- R* Alg. 1 line 3 puts it
    # in the initialisation stage and Stage II never regenerates it.
    if cfg["generate.alignment.critics"] and not state.critics:
        author = registry.get("phase", "author_critics")
        state.critics = author(ctx, state)
        ctx.event("critics", n=len(state.critics))

    candidates = sampler(ctx, state, backend)

    # R*: part of the population is recombined rather than sampled. The sampler
    # was already asked for the remainder (`_n_llm_this_iteration`), so this
    # tops the pool back up -- and it costs no LLM call, which is the point.
    crossover = registry.get("crossover_operator", cfg["generate.crossover.operator"])
    candidates = crossover(ctx, state, candidates)

    # R*: label pairs drawn from the carried trajectory store (every iteration's
    # rollouts, not only the last) into segments, then fit each candidate's own
    # numbers to them. Alg. 1 lines 10-12, which sit before the
    # training on line 14 -- so this runs on the pool about to be trained, and
    # at iteration 1 there is nothing to label and it is a no-op.
    labeller = registry.get("segment_labeller", cfg["generate.alignment.labeller"])
    labelled = labeller(ctx, state)
    if labelled:
        state.preferences.extend(labelled)
        # Logged HERE, where the labels are made -- `bird/preference_log.py`'s
        # rule for every producer; logging anywhere later would leave an rstar
        # run's `preferences.jsonl` empty while its state snapshot counts the
        # preferences. `judge="program"`: a critic is an LLM-WRITTEN
        # PROGRAM voting, not a model queried per pair and not a person, and
        # the dataset has to keep those apart to say anything about agreement.
        for pref in labelled:
            preference_log.record(ctx, pref, state, judge="program")
    align = registry.get("param_alignment", cfg["generate.alignment.method"])
    candidates = align(ctx, state, candidates)

    for c in candidates:
        ctx.event("generate", cand_id=c.cand_id, parent_id=c.parent_id,
                  chars=len(c.reward_code))
    n_x = sum(1 for c in candidates if c.meta.get("produced_by") == "module_insert")
    log.info("  [1] generate  -> %d candidate(s) via %s/%s%s",
             len(candidates), cfg["generate.sampling_mode"], cfg["generate.parent_source"],
             f" (+{n_x} crossover, {len(labelled)} segments)" if n_x or labelled else "")
    return candidates


# ==========================================================================
# Stage 2 -- Reward Verification
# ==========================================================================


def verify(ctx: Context, state: RunState, candidates: List[Candidate]) -> List[Candidate]:
    """Validity checks, then the quality screen. Identity when disabled."""
    cfg = ctx.cfg
    # The generate-only last iteration of a `loop.termination: fixed_generations`
    # run (CARD) keeps the VALIDITY half -- the release validates every
    # generation, `max_try_num` retries -- and drops the quality screen: Alg. 1
    # l.17 hands R_N to `Ensure R` unscreened, and a TPE verdict here would gate
    # a training that is not going to happen. Substituting `screen none`, which
    # is a registered component, rather than skipping the call keeps this a
    # dispatch on a value; `screen_label` says which value ran.
    screen_name = "none" if state.generation_only else cfg["verify.quality_screen"]
    screen_label = (cfg["verify.quality_screen"] if not state.generation_only
                    else "none: generation-only final iteration")
    if not cfg["verify.enabled"]:
        if screen_name == "none":
            log.info("  [2] verify    -> disabled (identity)")
            return candidates
    else:
        validity = registry.get("phase", "validity")
        candidates = validity(ctx, state, candidates)

    screen = registry.get("screen", screen_name)
    candidates = screen(ctx, state, candidates)

    n_ok = sum(1 for c in candidates if c.valid)
    n_screened = sum(1 for c in candidates if c.screened_out)
    for c in candidates:
        if c.failure:
            ctx.event("verify", cand_id=c.cand_id, kind=c.failure_kind, failure=c.failure)
    log.info("  [2] verify    -> %d valid, %d screened out (%s)",
             n_ok, n_screened, screen_label)
    return candidates


# ==========================================================================
# Stage 3 -- Policy Training
# ==========================================================================


def _repair_training_errors(ctx: Context, state: RunState,
                            results: List[TrainResult], backend, hpsearch, plan
                            ) -> Tuple[List[TrainResult], int]:
    """A reward that passed verification's smoke/shape checks but RAISED on a
    state reached later in policy training is re-generated with its TRAINING traceback and
    re-trained on a fresh seed stream, within the same iteration, so §6 and the tree
    back up the REPAIRED result rather than the crash. This is RF-Agent's shared
    retry loop (`rfagent_algo.py:561-563`, `:661-670`; neurips_2025 §205), which
    verification alone (`verification.run_validity`) applies only BEFORE training.

    Runs PARENT-SIDE, after `interaction()` has finished, over `results` in index order:
    the parallel training is already done, so this consumes `ctx.rng` deterministically and
    keeps `parallel` bit-identical to `sequential`. Each re-generation is one LLM call
    (`_regenerate` charges it once) and each re-train is one policy training
    (`hpsearch` charges it) -- both visible in `budget.json`. The re-train's seed stream is
    salted per attempt (`seed_phase=f"repair{attempt}"` -> `_seed_base`/`_phase_salt`), so a
    repaired candidate is a FRESH trial, disjoint from the crashed attempt and the search.
    Bounded by `verify.max_repair_attempts` (else `UNCAPPED_SAFETY_LIMIT`). Each re-train
    writes the same `train` journal event `on_result` writes, plus `repaired_from`.
    """
    from bird.components.verification import (
        _regenerate, run_validity, _REPAIR_TRACE, UNCAPPED_SAFETY_LIMIT)
    cfg = ctx.cfg
    cap = int(cfg.get("verify.max_repair_attempts", 0) or 0)
    limit = cap if cap > 0 else UNCAPPED_SAFETY_LIMIT
    out = list(results)
    n_repairs = 0
    n_crashed = sum(1 for r in results if r.error)
    for i, res in enumerate(results):
        if not res.error:
            continue
        cur = res
        attempt = 0
        while cur.error and attempt < limit:
            attempt += 1
            cand = cur.candidate
            cand.meta[_REPAIR_TRACE] = cur.error          # the TRAINING traceback
            fresh, _err = _regenerate(ctx, state, cand, attempt)   # one LLM call
            if fresh is None:
                break                                     # cannot resample; the crash stands
            verified = run_validity(ctx, state, [fresh])  # training only ever sees valid code
            if not verified or not verified[0].valid:
                settled = verified[0] if verified else fresh
                cur = TrainResult(cand_id=settled.cand_id, candidate=settled, trained=False,
                                  error=(settled.failure or "training-error resample did not compile"))
                break
            good = verified[0]
            # The allocator's plan is keyed by the CRASHED candidate's id; `good`
            # carries the fresh id `_regenerate` minted, so look the seeds up by
            # `res.cand_id` -- the trial the allocation was made for.
            n_seeds = plan.get(res.cand_id, cfg["train.seeds_per_candidate"])
            cur = hpsearch(ctx, state, good, backend, n_seeds=n_seeds,
                           seed_phase=f"repair{attempt}")  # one policy training, fresh seeds
            n_repairs += 1
            # The same `train` journal line `on_result` writes for every other
            # training -- `interaction()` has already recorded the ORIGINAL id
            # as crashed, and per-candidate readers of the journal read this
            # event, so a repaired training with no line would show as never
            # trained beside a report.json that says `trained: true`. `repaired_from` links the
            # two ids (the candidate's `meta["repaired_from"]` links each
            # attempt to its predecessor).
            ctx.event("train", cand_id=cur.cand_id, trained=cur.trained,
                      pruned=cur.pruned, timed_out=cur.timed_out,
                      env_steps=cur.env_steps_used, seeds=len(cur.seed_metrics),
                      repaired_from=res.cand_id, repair_attempt=attempt)
        out[i] = cur
    if n_repairs or n_crashed:
        state.n_train_repairs = int(getattr(state, "n_train_repairs", 0)) + n_repairs
        ctx.event("train_repair", iteration=state.iteration,
                  n_crashed=n_crashed, n_repairs=n_repairs)
    return out, n_repairs


def train(ctx: Context, state: RunState, candidates: List[Candidate]) -> List[TrainResult]:
    """The inner loop: pi = A_M(R), once per (candidate, seed)."""
    cfg = ctx.cfg
    if state.generation_only:
        # `loop.termination: fixed_generations`, last iteration: nothing is
        # launched. Each trainable candidate is booked as a SKIPPED training
        # (`policy_trainings_skipped`, the column CARD's saving is read from)
        # with a `skip_reason` naming this route; an untrainable one keeps its
        # own failure and counter. No backend, hpsearch or allocation dispatch
        # runs -- deliberately, and `allocate` is the one that matters: it
        # consumes `ctx.rng` (`bandit`, `bayesian_experimental_design`), so not
        # drawing here means the generate-only iteration spends no RNG the
        # trained loop would, and the stream a resumed leg restores is the
        # stream the same code would have produced.
        from bird.components import training
        results = [training.generation_only_result(ctx, c) for c in candidates]
        for r in results:
            ctx.event("train", cand_id=r.cand_id, trained=False, pruned=False,
                      timed_out=False, env_steps=0, seeds=0)
        n_skipped = sum(1 for r in results
                        if not r.trained and (not r.candidate.trainable or r.skip_reason))
        log.info("  [3] train     -> 0 trained, %d failed, %d skipped "
                 "(generation-only final iteration)", len(results) - n_skipped, n_skipped)
        return results
    backend = registry.get("train_backend", cfg["train.backend"])
    # Protect the blobs this round's warm start will ask for, BEFORE any
    # candidate is stored. `_store` evicts oldest-first, and the entry it
    # reaches first is the INCUMBENT -- the one entry the run cannot lose. A
    # large enough store cap makes that unreachable for a given blob size; the
    # pin makes it unreachable for any blob size. Per stage, because
    # reachability is: see `training.pin_refs`.
    from bird.components import training as _training
    _training.pin_refs(state)
    # `train.hyperparameter_search` WRAPS the backend rather than replacing it:
    # `none` is a bare tail call, `grid`/`per_candidate_best` run the backend once
    # per point and keep the best. Singh 2009 judges every reward under its own
    # best (alpha, epsilon), so fitness there is a max over a learner grid; every
    # LLM-era method pins the learner instead. Both are now sayable in config.
    hpsearch = registry.get("hyperparameter_search", cfg["train.hyperparameter_search"])
    allocate = registry.get("allocation", cfg["select.allocation"])
    # `train.candidate_parallelism` chooses only the SCHEDULE, never the work:
    # `sequential` runs the candidates in a plain loop and `parallel`
    # forks one process per candidate. Iterations stay strictly sequential
    # either way -- reflect-and-mutate is the method. See the long section at
    # the foot of `bird/components/training.py` for why the parallel one is
    # processes rather than threads, and for the determinism argument.
    schedule = registry.get("candidate_parallelism", cfg["train.candidate_parallelism"])
    # `train.interaction` sits BETWEEN the stage and the schedule. `independent`
    # is a bare tail call to `schedule`, so every config written before LaRes
    # runs the code it always ran; `shared_population` (LaRes, NeurIPS 2025)
    # pools the round's budget and divides it online, which makes the round's
    # candidates coupled and is therefore a different family from
    # `candidate_parallelism` -- see the header of
    # `bird/components/population.py` for why it could not be another entry
    # there without weakening the determinism test.
    interaction = registry.get("interaction", cfg["train.interaction"])

    # `allocate` consumes `ctx.rng` (the `bandit` and
    # `bayesian_experimental_design` rules do), so it runs ONCE, here, in the
    # parent, before anything can fork. A worker must never touch `ctx.rng`.
    plan = allocate(ctx, state, candidates)  # {cand_id: n_seeds}; uniform by default

    # `train.reward_scaling` (LaRes Eq. 3) is the other parent-only act in this
    # stage: `elite_moments` draws its moment sample from `ctx.rng` and
    # journals the moments, so it too runs HERE, once per trainable candidate
    # and before anything can fork, and leaves its plan on
    # `candidate.meta["reward_scaling"]` for every backend call (and every
    # `shared_population` slice) to APPLY. `none` plans nothing. Run inside
    # the backend, it would draw `ctx.rng` in a worker and lose its journal
    # line -- see the family note in `bird/components/population.py`.
    scaler = registry.get("reward_scaling", cfg["train.reward_scaling"])
    for c in candidates:
        if c.trainable:
            scaler(ctx, state, c)

    # ROSKA's per-round candidate budget. THE one reader of these three keys --
    # deliberately literal `cfg.get("...")` calls rather than a prefix helper, so
    # an AST search for the key name finds them and no allowlist entry is needed
    # (the declared-but-unread-key test).
    #
    # The published schedule trains a candidate for a FRACTION of a full
    # training, never a full training plus extras: round 1 short (no previous
    # policy exists to fuse with), rounds 2+ only the post-probe top-up, with
    # the search's own probes charged separately by the backend. Modelling it
    # as Eureka-plus-probes makes ROSKA cost MORE than Eureka, which inverts
    # its headline claim.
    def _round_budget() -> Optional[int]:
        if cfg.get("train.init") != "fused_warm_start":
            return None
        full = int(cfg.get("train.env_steps", 0) or 0)
        frac = float(cfg.get("train.fusion.first_round_fraction", 0.0) or 0.0) \
            if state.iteration == 0 \
            else float(cfg.get("train.fusion.sc_bo.post_probe_fraction", 0.0) or 0.0)
        return max(1, int(round(frac * full))) if frac > 0 and full > 0 else None

    round_budget = _round_budget()
    if round_budget is not None:
        # Per-component spend, journalled in the PARENT (never from a worker --
        # the journal is one append-only file and `ctx.rundir` is None in a
        # child). `slot` names which term of the published schedule this round's
        # per-candidate budget is, so the artifact's totals can be checked
        # against the config's fractions instead of taken on trust.
        ctx.event("round_budget", iteration=state.iteration,
                  slot="round1_train" if state.iteration == 0 else "post_probe",
                  env_steps_per_candidate=round_budget,
                  n_candidates=len([c for c in candidates if c.trainable]))

    def run_one(c: Candidate, **backend_kw: Any) -> TrainResult:
        """Everything a single candidate costs. This is what a worker runs.

        `backend_kw` is `rounds.py`'s: a rung of `train.pruning:
        successive_halving_pool` passes `env_steps` and `resume_ref` through
        to the backend. Empty on every other config.
        """
        if round_budget is not None:
            # `setdefault`, never an override: a `successive_halving_pool` rung
            # passes its own `env_steps` and owns the budget for that rung.
            # `_check_coherence` refuses the two schedules together anyway, so
            # this is a belt on a braces rather than a live precedence rule.
            backend_kw.setdefault("env_steps", round_budget)
        return hpsearch(ctx, state, c, backend,
                        n_seeds=plan.get(c.cand_id, cfg["train.seeds_per_candidate"]),
                        **backend_kw)

    # THE SAME FIVE THINGS, DECLARED AS DATA. `run_one` closes over ctx,
    # state, the backend, the hpsearch wrapper and the seed plan. A FORKED
    # worker inherits that closure; a SPAWNED one cannot be sent a closure at
    # all, so the identical five are named here -- at the one site that holds
    # them -- and `training.worker_inputs` turns them into the payload. The
    # two callables resolve from these same names by registry lookup, so the
    # worker runs the same construction rather than a second implementation
    # of it.
    #
    # ATTACHED TO `run_one` rather than added to the `candidate_parallelism`
    # signature: that signature is shared by every member of the family,
    # `sequential` included, and by their tests, so a spawn-only parameter on
    # all of them would be a wider change than the feature that needs it.
    # Declared on the lines immediately below the closure it describes, so
    # the two cannot drift apart without it being visible in one screen.
    run_one.worker_spec = {
        "backend_name": cfg["train.backend"],
        "hpsearch_name": cfg["train.hyperparameter_search"],
        "seed_plan": dict(plan),
        "default_n_seeds": cfg["train.seeds_per_candidate"],
    }

    def on_result(c: Candidate, res: TrainResult) -> None:
        """Parent-side bookkeeping. Never called from inside a worker: the
        journal is one append-only file and `ctx.rundir` is None in a child."""
        # `pruned` is journalled beside `trained` because they are three
        # outcomes, not two: ran to completion, cut short by `train.pruning`,
        # or broke. A pruned candidate's fitness is a lower bound, so a
        # ranking that cannot see the flag is reading two different quantities
        # off the same column.
        # `init_source` / `init_from_cand_id` beside them for the same reason:
        # under `train.init: warm_start_from_parent` a candidate whose parent had
        # no stored policy trains as the `warm_start_from_best` arm, and only
        # this pair (and `train_result.json`) says which arm each policy ran.
        ctx.event("train", cand_id=c.cand_id, trained=res.trained,
                  pruned=res.pruned, timed_out=res.timed_out,
                  env_steps=res.env_steps_used, seeds=len(res.seed_metrics),
                  init_source=res.init_source, init_from_cand_id=res.init_from_cand_id,
                  init_similarity=res.init_similarity)
        # The demonstration ceiling (`train.pruning_cfg.ceiling: demo_return` /
        # `train.pruning_metric: demo_fraction`): the expert's and the random
        # policy's return under THIS candidate's reward, and every stop the
        # guard blocked or allowed. Written by the backend onto the candidate,
        # journalled here because a worker has no journal.
        ceiling = c.meta.get("ceiling")
        if isinstance(ceiling, dict):
            ctx.event("train_ceiling", cand_id=c.cand_id, **ceiling,
                      decisions=[d for sm in res.seed_metrics
                                 for d in (sm.get("ceiling_decisions") or [])])

    def one_round(ctx_: Context, state_: RunState, cands: List[Candidate],
                  run_fn: Callable[..., TrainResult],
                  on_res: Callable[[Candidate, TrainResult], None]) -> List[TrainResult]:
        """One pass of the round over `cands`: the interaction over the schedule."""
        return interaction(ctx_, state_, cands, run_fn, on_res, schedule, backend)

    # ONE pass on every config but `train.pruning: successive_halving_pool`,
    # which trains the round in rungs and calls `one_round` once per rung
    # (`bird/components/rounds.py`). The dispatch is on the config value, the
    # way `_pruner_for` resolves the rule, so `train()` still holds no branch.
    results: List[TrainResult] = training_rounds(ctx, state, candidates, run_one,
                                                 on_result, one_round)

    # A reward can pass verification and still RAISE deeper in training. When
    # `train.on_runtime_error: repair`, re-generate + re-train it here, in-iteration, so
    # the repaired result (not the crash) reaches §4/§6 and the tree. Parent-side, so
    # `parallel` stays bit-identical to `sequential`.
    n_train_repairs = 0
    if cfg["train.on_runtime_error"] == "repair":
        results, n_train_repairs = _repair_training_errors(
            ctx, state, results, backend, hpsearch, plan)

    # FAILED AND SKIPPED ARE DIFFERENT POPULATIONS. Reporting
    # `len(results) - n_trained` as "skipped" would make a training that
    # launched and died read exactly like one that was never launched.
    # `budget.json` keeps them apart (`policy_trainings` counts the spend, and
    # `policy_trainings_skipped` is CARD's entire contribution), and so must the
    # banner a human watches a run through: a `limen` candidate whose
    # co-designed observation failed to install is a defect, not a cost saving.
    # Same convention as `failure_kind` in the artifact: `skipped` means the RL
    # job never launched.
    n_trained = sum(1 for r in results if r.trained)
    # `skip_reason` counts too, not just `trainable`. A candidate
    # `train.interaction: shared_population` never selected is trainable, was
    # not launched, and did not break -- counting it by `trainable` alone would
    # report it as FAILED, the exact confusion the comment above exists to
    # prevent (a scheduling decision reading as a defect).
    n_skipped = sum(1 for r in results
                    if not r.trained and (not getattr(r.candidate, "trainable", True)
                                          or r.skip_reason))
    n_failed = len(results) - n_trained - n_skipped
    log.info("  [3] train     -> %d trained, %d failed, %d skipped%s (%s, %s seed(s) each)",
             n_trained, n_failed, n_skipped,
             (" [%d repaired after a training crash]" % n_train_repairs) if n_train_repairs else "",
             cfg["train.backend"], cfg["train.seeds_per_candidate"])
    return results


# ==========================================================================
# Stage 4 -- Reward Evaluation
# ==========================================================================


def evaluate(ctx: Context, state: RunState, results: List[TrainResult]) -> List[CandidateReport]:
    """Turn training runs into (scalar, prose). Two outputs, configured apart."""
    cfg = ctx.cfg
    if state.generation_only:
        # `loop.termination: fixed_generations`, last iteration: nothing was
        # measured, so no configured source has anything to reduce and a
        # failure sentinel here would fabricate a scalar for a program nobody
        # scored. `fitness_source: none` is the honest report (`fitness=None`,
        # and `result.json: returned_fitness_source` reads "none" off it); the
        # feedback is the fixed skip constant on the `none` channel. No
        # preferences, similarity, human, thought-alignment, self-verify or
        # feedback builder -- each reads a training. `update.prompt.mode: append`
        # will still add that constant as the last user turn of the carried
        # dialogue, dead text in `state/iterNN` that no later prompt reads. The
        # per-report `evaluate` event below is emitted as usual, so the journal
        # holds one line per candidate in every iteration.
        from bird.components import training
        reports = registry.get("fitness_source", "none")(ctx, state, results)
        for r in reports:
            r.feedback, r.feedback_channel = training.GENERATION_ONLY_SKIP, "none"
    else:
        score = registry.get("fitness_source", cfg["evaluate.fitness.source"])
        reports = score(ctx, state, results)
    build_feedback = registry.get("feedback_builder", "default")

    # Every block from here to `build_feedback` reads a training; the
    # generate-only iteration has none, so `measured` gates them all.
    measured = not state.generation_only
    # The incumbent is re-scored under this round's rubric BEFORE anything reads
    # `state.best.fitness` -- §5's incumbent gate and §6's elitism both do. It
    # reads a training too, so the generate-only iteration skips it.
    if measured and cfg["evaluate.rejudge_incumbent"]:
        reports = registry.get("phase", "rejudge_incumbent")(ctx, state, reports)

    if measured and cfg["evaluate.preferences.enabled"]:
        prefs_phase = registry.get("phase", "preferences")
        reports = prefs_phase(ctx, state, reports)

    if measured and cfg["evaluate.similarity.metric"] != "none":
        sim = registry.get("similarity", cfg["evaluate.similarity.metric"])
        for r in reports:
            r.similarity = sim(ctx, state, r)

    if measured and cfg["evaluate.human.mode"] != "none":
        human_phase = registry.get("phase", "human_feedback")
        reports = human_phase(ctx, state, reports)

    # RF-Agent's two per-candidate LLM judgments. They are §4, not §1 or §5:
    # each reads one TRAINED candidate and writes evidence onto its report
    # (`meta["design_thought"]`, `meta["self_verify"]`) that §1's parent prompt
    # and §6's tree backup consume later; neither ranks anything. Alignment
    # runs first because verify judges the ALIGNED idea -- the release
    # re-describes the thought from the compiled code and only then scores it
    # (rfagent_algo.py:566-592; tex:205, tex:812-820) -- so `self_verify` reads
    # the `design_thought` the phase before it just wrote. Both sit before
    # `build_feedback` so a feedback builder can render either.
    if measured and cfg["evaluate.thought_alignment.enabled"]:
        reports = registry.get("phase", "thought_alignment")(ctx, state, reports)
    if measured and cfg["evaluate.self_verify.enabled"]:
        reports = registry.get("phase", "self_verify")(ctx, state, reports)

    for r in reports:
        if measured:
            r.feedback, r.feedback_channel = build_feedback(ctx, state, r)
        # `subtask_scores` is journalled, not merely rendered. It reaches the
        # generator (as `## BEHAVIOURAL ANALYSIS`, via
        # `evaluate.feedback.summarisation`) and `phases.run_subtask_
        # reflection`, which picks the LOWEST-scoring subtask to revise -- so
        # these numbers decide both what the next prompt says and which subtask
        # gets rewritten. Unjournalled, they would be recoverable only from a
        # LATER candidate's `prompt.json`, and `journal.jsonl` could not answer
        # "which subtask was weakest, and why was that one revised".
        # Empty for every config that is not per-subtask, so this costs nothing
        # where it means nothing.
        extra = {"subtask_scores": r.subtask_scores} if r.subtask_scores else {}
        # `similarity` is journalled for the same reason as `subtask_scores`.
        # `evaluate.similarity.metric` is computed per candidate per iteration,
        # and otherwise reaches only `summarise_reports` (the prose the
        # generator reads) and `selection._objective_value` (a ranking key) --
        # so under `evaluate.similarity.role: select`, or any `select.rule`
        # naming `similarity` as an objective, THIS NUMBER PICKS THE WINNER and
        # must leave a record of having done so. It is also the instrument a
        # reward-design ablation depends on -- EPIC distance of each candidate
        # against ground truth, tracked over iterations -- and
        # `configs/methods/eureka.yaml` publishes `evaluate.similarity.metric:
        # pearson_curve`, so every Eureka run computes it.
        #
        # `None` for every config leaving `metric: none` (the default), so this
        # costs nothing where it means nothing.
        if r.similarity is not None:
            extra["similarity"] = r.similarity
            # POLARITY, FIRST, BECAUSE A COMMENT THAT INVERTS AN AXIS IS WORSE
            # THAN NONE.
            #
            #     `similarity` is HIGHER-IS-BETTER for every metric.
            #     The raw EPIC/STARC distance is `1 - similarity`, and this does
            #     not journal it -- it lives at `meta["epic_distance"]`.
            #
            # `similarity_epic` returns `max(0.0, 1.0 - d)` (`bird/components/evaluation.py`)
            # precisely so section 5's rules, which all maximise, can read it. So
            # `epic` here points the same way as `pearson_curve`, not the
            # opposite way: `"similarity": 0.9762` beside a task metric of 0.82
            # is coherent, and read as a distance it would look inverted. A
            # plotting harness that took the axis backwards would name the worst
            # candidate the best.
            #
            # THE METRIC NAME TRAVELS WITH THE VALUE, against the house style.
            # It does NOT record which implementation ran -- it reads the config,
            # and every EPIC path runs under `metric: epic`. What it buys is that
            # the two records of an EPIC number are not oriented the same way:
            # this journals a similarity, while `similarity_epic` keeps the raw
            # distance at `meta["epic_distance"]`. A row that can be read
            # BACKWARDS is worse than a row that repeats itself.
            #
            # That is the contrast with `fitness`: it is not that
            # `evaluate.fitness.source` uniquely pins the implementation, it is
            # that the quantity has ONE ORIENTATION and this one does not.
            extra["similarity_metric"] = cfg["evaluate.similarity.metric"]
            # WHICH IMPLEMENTATION ACTUALLY RAN, when the component records it.
            # This cannot come from the config and it cannot come from a
            # run-level artifact either: `_pseudometric` falls back PER
            # CANDIDATE, so a candidate whose program will not recompile yields
            # the local path while its siblings in the same iteration yield the
            # canonical one. The implementation can differ between two rows of
            # one run, which makes it a per-row fact by construction rather
            # than a repetition of the config.
            #
            # Emitted only when present, so this is inert for a component that
            # does not set it. Absent rather than null for the same reason as
            # the fields above.
            mode = (r.meta or {}).get("epic_mode") or (r.meta or {}).get("starc_mode")
            if mode:
                extra["similarity_mode"] = mode

        ctx.event("evaluate", cand_id=r.cand_id, fitness=r.fitness,
                  channel=r.feedback_channel, feedback_chars=len(r.feedback),
                  **extra)

    log.info("  [4] evaluate  -> fitness via %s; feedback granularity=%s",
             cfg["evaluate.fitness.source"], cfg["evaluate.feedback.granularity"])
    log.debug("\n%s", summarise_reports(reports))
    return reports


# ==========================================================================
# Stage 5 -- Reward Selection
# ==========================================================================


def select(ctx: Context, state: RunState, reports: List[CandidateReport]) -> Selection:
    """Collapse the evidence into this iteration's verdict."""
    cfg = ctx.cfg
    rule = registry.get("select_rule", cfg["select.rule"])
    selection = rule(ctx, state, reports)

    if selection.contested and len(selection.winners) > 0:
        sig = registry.get("significance", cfg["select.significance"])
        selection = sig(ctx, state, selection, reports)

    if cfg["select.require_improvement_over_incumbent"] and state.best is not None:
        w = selection.winner
        incumbent = state.best.fitness
        challenger = w.fitness if w else None
        if incumbent is not None and challenger is not None:
            selection.improved_over_incumbent = challenger > incumbent
            # The Occam gate (`select.incumbent_simplicity_margin`, ours): a
            # challenger that is not strictly better still counts as improving
            # when it sits within the margin of the incumbent's fitness AND its
            # reward has fewer AST nodes than the chain head it would replace.
            # Fitness is compared against the GLOBAL incumbent (so the chain can
            # drift at most one margin below the best ever seen, not one margin
            # per round); complexity against `state.latest` (the program the
            # winner would actually displace). Without the gate the strict rule
            # is a ratchet: a simpler reward at equal fitness can never take the
            # chain, so complexity only ever grows.
            margin = float(cfg["select.incumbent_simplicity_margin"] or 0.0)
            if margin > 0.0 and not selection.improved_over_incumbent:
                from bird.components.selection import _ast_node_count
                head = state.latest if state.latest is not None else state.best
                c_nodes = _ast_node_count(w.candidate.reward_code)
                h_nodes = _ast_node_count(head.candidate.reward_code)
                within = challenger >= incumbent - margin
                simpler = c_nodes < h_nodes
                selection.improved_over_incumbent = bool(within and simpler)
                ctx.event("select_occam_gate", cand_id=w.cand_id, head_id=head.cand_id,
                          challenger=challenger, incumbent=incumbent, margin=margin,
                          challenger_ast_nodes=c_nodes, head_ast_nodes=h_nodes,
                          within_margin=within, simpler=simpler,
                          accepted=selection.improved_over_incumbent)
        elif challenger is None:
            selection.improved_over_incumbent = False

    ctx.event("select", rule=cfg["select.rule"],
              winners=[w.cand_id for w in selection.winners],
              tie_broken=selection.tie_broken, tied=selection.tied_ids)
    w = selection.winner
    log.info("  [5] select    -> %s (%s)%s",
             w.cand_id if w else "no contest", cfg["select.rule"],
             "  [TIE BROKEN]" if selection.tie_broken else "")
    return selection


# ==========================================================================
# Stage 6 -- Reward Update
# ==========================================================================


def update(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """Winners, losers, and the state that survives into the next iteration."""
    cfg = ctx.cfg
    topology = registry.get("topology", cfg["update.topology"])
    state = topology(ctx, state, selection)

    routing = registry.get("feedback_routing", cfg["update.feedback.routing"])
    state = routing(ctx, state, selection)

    prompt_mode = registry.get("prompt_mode", cfg["update.prompt.mode"])
    state = prompt_mode(ctx, state, selection)

    for loser in selection.losers:
        registry.get("loser_action", cfg["update.loser.action"])(ctx, state, loser)

    if cfg["update.co_evolve.subtasks"] and selection.winner is not None:
        co = registry.get("phase", "subtask_reflection")
        state.subtasks = co(ctx, state, selection.winner)

    # AFTER topology (a stage pass or a regression rollback overwrites the
    # `policy_ref` topology just set from `state.best`, and would be overwritten
    # by it the other way round) and BEFORE apply_carry (the curriculum is a carry slot like any
    # other; a config that does not carry it does not have one).
    if cfg["loop.curriculum.enabled"]:
        step = registry.get("phase", "curriculum_step")
        state = step(ctx, state, selection)

    # The run's RETURN, recorded while it is still knowable. `final_artifact`
    # reads a memory slot (`best`, `archive`) or the chain head, and the carry
    # below clears every memory slot the config does not name -- so under
    # `loop.carry: []` (`zeroshot`: one iteration, inherited `global_best`)
    # reading `state.best` after the carry would find it nulled and return
    # `returned_cand_id: None` while its candidate sits in `candidates/`.
    # What the search produced must not depend on what the
    # config forgets between iterations; `RunState.returned` is the record and
    # `apply_carry` cannot touch it. Kept from the previous iteration when this
    # one resolves to nothing, so a return once known is never un-known.
    resolved = state.resolve_final_artifact(cfg["select.final_artifact"])
    if resolved is not None:
        state.returned = resolved
    state.apply_carry(cfg["loop.carry"] or [])
    ctx.event("update", topology=cfg["update.topology"], **state.summary())
    log.info("  [6] update    -> %s; carry=%s",
             cfg["update.topology"], cfg["loop.carry"])
    return state


# ==========================================================================
# The loop
# ==========================================================================


def _begin(ctx: Context, state: RunState, of: str) -> None:
    """Record that a stage was ENTERED, not that it finished.

    Every other journal line is written on completion -- `generate` emits its
    candidates only once `sampler()` has returned all of them, `train` one line
    per candidate as each finishes. That makes a journal that cannot answer the
    question `bird/artifacts.py` was written to answer. Measured: a healthy
    16-candidate `generate` wrote its first journal byte **twelve minutes**
    after the run started, and for those twelve minutes the artifact is
    byte-for-byte indistinguishable from a run that died the moment it began.
    A renderer wedged inside `record_rollouts` produces the same silence, and
    without an entry line it can be diagnosed only after the fact, by matching
    elapsed times against the code.

    An entry line closes it: the last one names the stage a run is inside, so a
    silent gap is attributable rather than merely long. This is a fact about the
    run, recorded where the run's other facts are recorded; nothing about it is
    for any particular reader.

    Emitted from the parent loop and nowhere else, which is what keeps
    `train.candidate_parallelism: parallel` byte-identical to `sequential` -- a
    worker has no `ctx.rundir` and must never write here (see `on_result`).

    ON RESUME, BYTE-EQUALITY IS TESTED FOR ONE CRASH AND NOT THE OTHER.
    `test_resume.py`'s `_crash_then_resume` replaces `run_iteration` wholesale
    and raises BEFORE calling the real one, so the dead leg emits no entry line
    and the journals still compare exactly -- the synthetic crash lands outside
    the instrumented region. A real kill lands inside it: the dead leg has
    already written `stage_begin`, the resumed leg re-runs that iteration and
    writes a second, and `stage_begin` is not in `_VOLATILE`. That is not a
    defect, it is the journal-level shadow of a re-payment the repo already
    declares and counts (`budget.resume_discarded_trainings`) -- but it is
    untested, and this line is here so nobody reads the passing test as
    covering it.
    """
    ctx.event("stage_begin", of=of, iteration=state.iteration,
              restart=state.restart)


def run_iteration(ctx: Context, state: RunState) -> Optional[Selection]:
    """One iteration. None => every candidate failed.

    Stages §1-§4 run once per `loop.waves` pass and §5-§6 run ONCE over the
    union of them. With no waves configured -- every method but R* -- there is
    exactly one pass and this is the six-stage sequence it always was.

    WHY THE SPLIT IS WHERE IT IS. R*'s first iteration evaluates its LLM
    individuals, then builds crossover individuals FROM THOSE RESULTS and
    evaluates them, "and finally aggregate[s] all evaluation results to select
    the best individual" (App. A, p.12). So the boundary is exactly §4/§5: a
    pass must reach evaluation for the next pass to have parents, and selection
    must not run until every pass has. Splitting anywhere else would either
    starve the second pass or give the iteration two winners.

    ORDER IS BY WAVE, THEN BY CANDIDATE INDEX WITHIN A WAVE -- never by
    completion. `select.tie_break: first` decides real rounds, so an
    aggregation that concatenated in finishing order would change winners
    while every fitness stayed identical, and
    `tests/test_parallelism.py`'s bit-identity claim would quietly stop holding.
    """
    from bird.components.generation import active_waves

    # Evaluated BEFORE any wave runs: `when: no_archive` asks about the state
    # the iteration STARTED in. Wave 1 fills the archive, so re-reading the
    # predicate later would suppress the pass it exists to enable.
    state.wave_bootstrap = len(state.all_reports) < 2
    waves = active_waves(ctx, state)

    candidates: List[Candidate] = []
    results: List[TrainResult] = []
    reports: List[CandidateReport] = []
    for w_i, _wave in enumerate(waves):
        state.wave = w_i
        if len(waves) > 1:
            ctx.event("wave_begin", iteration=state.iteration, wave=w_i,
                      of=len(waves), llm=_wave["llm"], crossover=_wave["crossover"])
        _begin(ctx, state, "generate")
        w_cands = generate(ctx, state)
        _begin(ctx, state, "verify")
        w_cands = verify(ctx, state, w_cands)
        _begin(ctx, state, "train")
        w_results = train(ctx, state, w_cands)
        _begin(ctx, state, "evaluate")
        w_reports = evaluate(ctx, state, w_results)
        # Appended here, inside the loop, and this is the mechanism rather than
        # bookkeeping: `all_reports` IS the archive a later wave's crossover
        # draws parents from (alignment._archive_reports), so wave 2 can only
        # recombine wave 1's individuals because wave 1 landed here first.
        state.all_reports.extend(w_reports)
        candidates.extend(w_cands)
        results.extend(w_results)
        reports.extend(w_reports)
    state.wave = 0
    if state.generation_only:
        # One line per generate-only iteration, in the parent, beside the
        # per-candidate `train` (`trained: false`) and `evaluate` lines: the
        # journal's own statement that this iteration ended at §1 by design
        # (`loop.termination: fixed_generations`), so an untrained chain end
        # reads as the method and not as a learner that never launched.
        ctx.event("generation_only", iteration=state.iteration, restart=state.restart,
                  cand_ids=[c.cand_id for c in candidates],
                  trainable=sum(1 for c in candidates if c.trainable),
                  termination=ctx.cfg["loop.termination"])

    # ONE execute-rate per iteration, over the union. Per wave it would make
    # `loop.termination: execute_rate` and every plot that reads the history
    # count a two-wave iteration twice, so a config that changed only its wave
    # structure would appear to change its convergence.
    rate = execute_rate(reports)
    state.execute_rate_history.append(rate)
    for c in candidates:
        if ctx.rundir:
            ctx.rundir.save_candidate(c)
    for r in results:
        if ctx.rundir:
            # The Candidate is the program; the TrainResult is what the learner
            # did with it -- per-seed curves, the checkpoint trace §4 reduces,
            # the skip reason, and the hyperparameter scoreboard. Writing only
            # the first left the second as a log line nobody could check.
            ctx.rundir.save_train_result(r)
    for r in reports:
        if ctx.rundir:
            # And the third: what stage 4 CONCLUDED. The journal carries the
            # scalar and the per-subtask scores, but not the rationales behind
            # them nor the prose the next prompt is built from -- only its
            # length. `report.json` is RDA's §7.3 trajectory analysis in the
            # field names the paper prints, and for every other method the same
            # record with an empty `subtasks`.
            ctx.rundir.save_report(r)

    # Recording happens here, between §4 and §5, because it needs the reports
    # (to know which candidates to record) and because §4's VLM and preference
    # comparators read the frames it writes -- the artifact a human watches and
    # the artifact a judge scores are deliberately the same one.
    _begin(ctx, state, "record")
    for rec in record_rollouts(ctx, reports):
        # Every recording reaches disk; only the subset named by
        # `output.wandb.video_record` reaches the tracker, and it is keyed by
        # its UPLOAD role ("best"/"worst") rather than by cand_id -- an id is
        # minted once and never recurs, so an id-keyed panel holds one frame
        # forever instead of gaining a slider.
        if rec.upload_role:
            ctx.tracker.log_media(ctx, rec.upload_role, rec.path)

    # "Every sample failed" means execute_rate's predicate, not just the parse:
    # a reward can compile perfectly and then raise on the first transition
    # (eureka.py L228-230 gates exec_success on the RL run finishing without a
    # traceback). Testing `candidate.valid` alone let an all-runtime-crash
    # round fall through to §5, where the -10000 sentinels would elect a
    # crashed reward as parent and global best.
    if reports and not any(r.candidate.valid and not r.result.error
                           for r in reports):
        if ctx.cfg["loop.on_total_failure"] == "continue_after_update":
            # §6 with no winner: the failed round is offered to the topology as
            # losers, so one that backs failures up (RF-Agent's search tree,
            # `reward_fail_bound = 0`) sees them in their own iteration rather
            # than one late via `tree._catch_up` -- and sees an all-failed first
            # or final round at all. Nothing here elects a winner: §5
            # is still skipped, so a sentinel fitness cannot become the parent.
            _begin(ctx, state, "update")
            state = update(ctx, state, Selection(
                winners=[], losers=list(reports), rule="none",
                contested=len(reports) > 1,
                notes="every candidate failed; §6 ran without a winner "
                      "(loop.on_total_failure: continue_after_update)"))
        else:
            # The iteration boundary is a boundary whichever way the round went.
            # `apply_carry` lives at the end of `update()`, which every other value skips,
            # so without this call a carry slot written in stages 1-4 -- `subtasks`
            # (decompose), `preferences` (evaluate), `failure_memory` (stage-2
            # `teach_next_iteration`), `critics` -- would survive an all-failed
            # round whatever `loop.carry` says (rda with `subtask_list` uncarried
            # would enter the next iteration with the failed round's subtasks and
            # skip its decompose call). No shipped config carries a slot it
            # does not also write before stage 6, so this is the rule, not a number.
            state.apply_carry(ctx.cfg["loop.carry"] or [])
        ctx.tracker.log_iteration(ctx, state, reports, None)
        return None  # loop.on_total_failure decides what happens next

    _begin(ctx, state, "select")
    selection = select(ctx, state, reports)
    _begin(ctx, state, "update")
    state = update(ctx, state, selection)
    ctx.tracker.log_iteration(ctx, state, reports, selection)
    return selection


def _ckpt(ctx: Context, phase: str, state: Optional[RunState] = None,
          restart: int = 0, next_iteration: int = 0, retries: int = 0) -> None:
    """Write one checkpoint, if `loop.resume_from` asked for them.

    A no-op when the key is null (every published config) or when there is no
    run directory, and it never raises: `checkpoint.write` swallows its own
    failures, because killing a 20-hour search over a serialisation fault would
    be strictly worse than not resuming it.
    """
    cp = ctx.checkpointer
    if cp is None:
        return
    cp.save(ctx, phase=phase, state=state, restart=restart,
            next_iteration=next_iteration, retries=retries)


def _reenter_for_full_iterations(plan: Optional["checkpoint.ResumePlan"],
                                 cfg: Config, flag: str = "--full-iterations",
                                 ) -> "checkpoint.ResumePlan":
    """Make an ALREADY-ENDED search resumable for its remaining iterations.

    SHARED BY TWO FLAGS, and `flag` only names the one in the error messages.
    `--full-iterations` re-enters a search that stopped EARLY; `--extend-iterations`
    re-enters one that ran to a budget which has since been raised. The mechanical
    problem is identical in both cases and is stated below: a loop that ended looks
    finished, so without this the leg skips the loop entirely, runs the post phases
    and reports success having added no iterations -- a silent no-op, which is
    worse than a refusal: the extension adopts the directory, writes its audit
    record, exits 0, and runs nothing.

    Why this is needed at all: when `should_stop` breaks the loop, no iteration
    checkpoint is written -- `_execute` writes `restart-NN.json` instead -- and
    `_execute` resumes at `len(plan.restart_states)`. So a restart that stopped
    early looks *finished*: the loop is skipped entirely and the run goes
    straight to its post phases. Dropping that last restart state re-enters the
    restart at the newest iteration checkpoint, which is where the search
    actually got to.

    What this deliberately does NOT do is touch the config. `loop.termination`
    stays `fitness_plateau` in `config.resolved.yaml`, so the file remains
    byte-identical to the config being resumed (`_assert_same_config` still
    holds) and the hash still names this directory. Overriding the key instead
    would leave a directory called `<name>-H-<stamp>` whose config no longer
    produces H -- the exact failure `--resume-degraded` is a flag to avoid.

    SCOPE, neither half of which is guessable from the flag's name:

      * it extends ONE restart -- the one whose loop ended last, the only one
        whose continuation point is still on disk. A search in which an
        already-completed restart stopped short is REFUSED below rather than
        partly extended, because a partial extension reports as a whole one.
      * the suppression it turns on is passed to every restart the resumed leg
        runs (`_execute` -> `run_search`), not only to the re-entered one, so a
        restart that had not started yet also runs under the suppressed rule.
        That is the coherent reading of "give me the configured budget" -- a
        fresh restart plateauing at 2 of 10 would leave exactly the gap this
        flag exists to close -- but it does change the method for work that had
        not happened yet, which is why it is written down here.
    """
    if plan is None:
        raise checkpoint.CheckpointError(
            f"{flag}: no usable checkpoint to continue from")
    total = cfg["loop.n_iterations"]
    states = list(plan.restart_states)
    if not states:
        raise checkpoint.CheckpointError(
            f"{flag}: no completed restart to continue; this search did "
            "not reach the end of its loop, so plain --resume already continues it")
    # THE ITERATION CURSOR IS GONE, and that is why this cannot just drop the
    # restart and reuse `plan.next_iteration`. Retention keeps only the newest two
    # checkpoints, so once a loop ends those two are `restart_done` and
    # `loop_done` -- both of which carry `next_iteration: 0`. Reusing it re-runs
    # the search from zero INTO THE SAME DIRECTORY: measured on a tester run, the
    # journal came out as iterations [0,1,2,3, 0,1,2,3,4], which is the "two
    # searches concatenated in one journal" this module refuses elsewhere.
    #
    # The end-of-restart state is the right continuation point instead: it is
    # exactly what the next iteration would have read, and the accompanying
    # checkpoint's rng/counters are the end-of-loop ones. `state.iteration` is
    # the last index executed, so the next one is that plus one.
    resume_state = states[-1]
    states = states[:-1]
    # ONLY THE RE-ENTERED RESTART CAN BE EXTENDED, so refuse rather than extend
    # one restart of five. `_execute` seeds `states` from
    # `plan.restart_states` and loops from `len(states)`, so every restart before
    # this one is REPLAYED FROM ITS STORED END STATE and never re-enters its
    # loop. It cannot: a completed restart's write-once state carries no rng or
    # counters, and retention keeps only the newest two iteration checkpoints, so
    # the continuation point for an earlier restart is simply not on disk.
    #
    # Extending anyway would write one full restart beside short ones into a
    # single `result.json` -- `per_restart` a mix of the two, post phases reading
    # `states[-1]`, the only extended one -- with nothing in the artifact saying
    # the extension was partial. `configs/methods/eureka.yaml` is `n_restarts: 5` and
    # `configs/methods/rda.yaml` is 3, so under a plateau rule the operator would ask for
    # the full budget, be told it was delivered, and get 20% of it.
    #
    # A restart that reached the budget under its own rule is NOT short and is no
    # obstacle, which is why this counts iterations rather than restarts.
    #
    # CHECKED BEFORE THE RESTART-CURSOR GUARD BELOW, which on a cleanly finished
    # multi-restart run fires first and blames the wrong thing: `loop_done`
    # records `restart: 0` while `restart_states` holds one entry per restart, so
    # `plan.restart != len(states)` and it refuses "to guess which loop to
    # re-enter" -- ambiguity, when the actual obstacle is that restart 0 stopped
    # short and cannot be continued. Relying on that accident would also be
    # fragile: when the newest checkpoint is `restart_done` rather than
    # `loop_done` (a leg killed between the two writes) the cursor guard passes
    # and the partial extension goes through silently, which is the case this
    # check exists for.
    short = [(i, int(getattr(s, "iteration", 0)) + 1) for i, s in enumerate(states)
             if int(getattr(s, "iteration", 0)) + 1 < total]
    if short:
        ran = ", ".join(f"restart {i} ran {n}" for i, n in short)
        raise checkpoint.CheckpointError(
            f"--full-iterations: {ran} of {total} iterations and cannot be "
            f"re-entered -- only restart {plan.restart}, whose loop ended last, "
            "keeps the rng and counters a continuation needs. Extending would "
            "report one full restart beside short ones as a single result. Re-run "
            "this cell from zero to bring every restart to its full budget")
    if plan.restart != len(states):
        raise checkpoint.CheckpointError(
            f"--full-iterations: the newest checkpoint belongs to restart "
            f"{plan.restart} but {len(states)} restarts read as complete; refusing "
            "to guess which loop to re-enter")
    next_it = int(getattr(resume_state, "iteration", 0)) + 1
    if next_it >= total:
        raise checkpoint.CheckpointError(
            f"--full-iterations: restart {plan.restart} already ran {next_it} of "
            f"{total} iterations; there is nothing to extend")
    return dataclasses.replace(plan, restart_states=states,
                               completed_restarts=list(range(len(states))),
                               state=resume_state, next_iteration=next_it)


def run_search(ctx: Context, restart: int = 0,
               resume: Optional["checkpoint.ResumePlan"] = None,
               full_iterations: bool = False) -> RunState:
    """One independent run of the outer loop (`loop.n_restarts` of these).

    `resume` supplies ONLY the initial `(state, iteration, retries, rng)`. After
    that this is indistinguishable from a fresh run -- there is deliberately no
    second code path through the loop.
    """
    cfg = ctx.cfg
    if resume is None:
        state = RunState(restart=restart, rng_seed=cfg["seed"] + restart)
        # A restart is an independent run of the loop: the module-level policy
        # / reward-code / replay stores are emptied so `warm_start_from_similar`
        # and `warm_start_from_parent` cannot reach the previous restart's
        # policies (`training.reset_policy_stores`).
        from bird.components.training import reset_policy_stores
        reset_policy_stores()
        ctx.rng = random.Random(state.rng_seed)
        retries, it = 0, 0
    else:
        state = resume.state
        # setstate, NOT a reseed: `ctx.rng` is one stream consumed at a
        # data-dependent rate (selection bootstrap, archive parent sampling,
        # generation's shuffle, preference-pair sampling, and MockLLM's
        # per-completion salt). Reseeding here would rewind all of it.
        ctx.rng.setstate(resume.rng)
        retries, it = resume.retries, resume.next_iteration
        log.info("resuming restart %d at iteration %d (retries=%d)", restart, it, retries)
    should_stop = registry.get("termination", cfg["loop.termination"])
    # Read off the CONFIGURED rule, before `full_iterations` may rebind the
    # name to a closure: `fixed_generations` (CARD) declares that the iteration
    # the declared budget ends on is generate-only (§1 + validity, no screen,
    # no training, no evaluation). The bound is the config's `loop.n_iterations`
    # -- the one `--extend-iterations` moves -- so the generate-only step is
    # always the last iteration of the budget as declared, never a fixed index.
    ends_on_generation = bool(getattr(should_stop, "ends_on_generation", False))
    if full_iterations:
        # The operator asked for the CONFIGURED budget in full. The rule is still
        # consulted once that budget is reached, so a termination that would fire
        # exactly at `n_iterations` still fires; only stopping SHORT is suppressed.
        # This can never run past `loop.n_iterations` -- the `while` bound is the
        # config's, which is what keeps the declared budget authoritative.
        #
        # THIS APPLIES TO EVERY RESTART THIS LEG RUNS, not only the re-entered
        # one: `_execute` passes the flag down for all of them,
        # so a restart that never ran -- and never stopped short -- also runs
        # with its early-stop rule suppressed. Deliberate, and recorded in
        # `result.json` as `extended_full_iterations` for exactly that reason:
        # the run's own `config.resolved.yaml` still names the rule and now
        # points the wrong way for it.
        _configured_stop = should_stop

        def should_stop(ctx_: Context, state_: RunState) -> bool:
            if state_.iteration + 1 < cfg["loop.n_iterations"]:
                return False
            return _configured_stop(ctx_, state_)

    while it < cfg["loop.n_iterations"]:
        state.iteration = it
        # Recomputed every pass from `it`, so a resumed leg re-entering the last
        # iteration gets the right value from the counter, not from the file.
        state.generation_only = ends_on_generation and it + 1 >= cfg["loop.n_iterations"]
        log.info("iteration %d/%d  (restart %d)%s", it + 1, cfg["loop.n_iterations"], restart,
                 "  [generation only]" if state.generation_only else "")
        try:
            selection = run_iteration(ctx, state)
        except BudgetExceeded as exc:
            log.warning("budget exhausted: %s", exc)
            break

        if selection is None:
            state.consecutive_failures += 1
            policy = cfg["loop.on_total_failure"]
            log.warning("every candidate failed this iteration -> %s", policy)
            if policy == "abort":
                break
            if policy == "retry_iteration" and retries < cfg["loop.max_iteration_retries"]:
                # Repeat from the current message checkpoint, do not advance
                # the counter and do not touch the parent. (No published
                # config selects this: upstream Eureka `continue`s and
                # consumes the iteration.)
                retries += 1
                _ckpt(ctx, "iteration_retry", state, restart, it, retries)
                continue
            it += 1
            # The cap is PER FAILED ITERATION (`loop.max_iteration_retries`):
            # giving up on this one hands the next a fresh budget. Resetting
            # only after a success would make a streak of failed iterations
            # share one cap, advancing every iteration after the first after a
            # single attempt. Reset BEFORE the checkpoint so a leg resumed from
            # `iteration_failed` agrees.
            retries = 0
            _ckpt(ctx, "iteration_failed", state, restart, it, retries)
            continue

        retries = 0
        state.consecutive_failures = 0
        w = selection.winner
        state.fitness_history.append(w.fitness if w else None)

        if ctx.rundir and cfg["output.save_state_every_iteration"]:
            ctx.rundir.save_state(state, it)
            # The snapshot is the COUNTS; this is the objects they count
            # (`records/iterNN.json`). Same gate, same cadence, same restart caveat -- and
            # both written after `update()` ran `apply_carry`, so a slot the
            # config does not carry is honestly empty in both files.
            ctx.rundir.save_records(state, it)

        it += 1
        if should_stop(ctx, state):
            log.info("terminating early: %s", cfg["loop.termination"])
            break
        # AFTER `should_stop`, and the ordering is load-bearing twice over.
        # `update()` ends with `state.apply_carry(...)`, so only now is the
        # state exactly what the next iteration will read; and `should_stop`
        # may consume `ctx.rng`, so checkpointing after it means the saved
        # stream includes that draw and the resumed leg correctly does NOT
        # re-ask a termination question that already answered "no". When
        # `should_stop` breaks, no iteration checkpoint is written -- `_execute`
        # writes `restart-NN.json` immediately afterwards, which is better.
        _ckpt(ctx, "iteration_done", state, restart, it, retries)

    return state


def final_artifact(ctx: Context, state: RunState) -> Optional[CandidateReport]:
    """What the whole run returns -- a different rule from the per-iteration
    winner, and one nobody in the literature ablates.

    Read through `RunState.final_artifact`: the record `update()` wrote before
    the last boundary carry when there is one, else the live slots. The rules
    themselves live in `RunState.resolve_final_artifact`, shared with the post
    phases so what they retrain is what this returns."""
    mode = ctx.cfg["select.final_artifact"]
    if mode not in FINAL_ARTIFACT_RULES:
        raise ConfigError(f"unknown select.final_artifact: {mode}")
    return state.final_artifact(mode)


def _restore_status_after_failed_extension(rundir, prior_status, exc) -> None:
    """Put back the terminal status an extension attempt overwrote.

    `--full-iterations` adopts a run that has ALREADY FINISHED. Adoption rewrites
    `status.json` to "running", and `run`'s handler then writes "failed" if
    anything raises -- so a crash in the extension (an API outage, an OOM, the
    walltime) downgrades a search that completed successfully hours earlier.
    Nothing about that search changed; only the attempt to continue it did. An
    exception after adoption must not rewrite an adopted run's status.

    The failed attempt is not hidden: it is kept as `status.extend-failed.json`
    and recorded in `resume.jsonl`, exactly as `result.pre-extend.json` keeps the
    outcome the extension supersedes.
    """
    if not prior_status or prior_status.get("status") not in ("ok", "failed"):
        return
    try:
        path = rundir.path / "status.json"
        attempt = json.loads(path.read_text()) if path.exists() else {}
        (rundir.path / "status.extend-failed.json").write_text(
            json.dumps(attempt, indent=2, sort_keys=True) + "\n")
        # The restored status CARRIES the failure rather than hiding it: a
        # consumer reading only `status.json` still learns the extension died,
        # while a consumer selecting `status == "ok"` still sees the completed
        # search. Both needs are real and neither has to lose.
        restored = dict(prior_status)
        restored["extension"] = {
            "attempted": True, "outcome": "failed",
            "error": f"{type(exc).__name__}: {exc}"[:300],
            "attempt_status": "status.extend-failed.json"}
        path.write_text(json.dumps(restored, indent=2, sort_keys=True) + "\n")
        checkpoint.append_resume_record(rundir.path, {
            "event": "extension_failed",
            "restored_status": prior_status.get("status"),
            "error": f"{type(exc).__name__}: {exc}"[:300]})
    except OSError:
        pass          # never let bookkeeping mask the original exception


def run(cfg: Config, out_root: Optional[str] = None, dry_run: bool = False,
        adopt: Optional[str] = None, force: bool = False,
        allow_degraded: bool = False,
        full_iterations: bool = False,
        extend_to: Optional[int] = None) -> Dict[str, Any]:
    """Entrypoint: pre-phases, restarts of the loop, post-phases.

    `out_root` is the run root. `None` -- the CLI given no `--out` -- resolves
    to the config's `output.dir`, so `-s output.dir=...` both moves the hash
    and moves the run. An explicit `--out` wins, and `status.json` records
    which it was (`out_root_source`: cli | config).

    The body is `_execute`; this wrapper exists only so that the run directory
    carries a terminal status. A search that raises must leave the reason IN THE
    ARTIFACT and not only on a stdout that, under a batch scheduler, may be
    written to a compute node's local disk -- and a run killed outside Python
    (SIGSEGV, a scheduler cancel, the walltime) must leave a directory that
    reads as `running` forever rather
    than as a healthy run that happens to be missing two files.
    """
    setup_logging(cfg["output.log_level"])
    if out_root is None:
        out_root, out_root_source = str(cfg["output.dir"] or "runs"), "config"
    else:
        out_root_source = "cli"
    prior_status = None          # terminal status an extension is superseding
    registry.load_all()

    budget = Budget.from_config(cfg)
    rundir: Optional[RunDir] = None
    lock = None
    plan = None
    leg = 1
    target: Optional[Path] = None
    if not dry_run:
        # `loop.resume_from` is a SENTINEL, so leg 1 and leg 4 resolve to the
        # same config, hash to the same H, and belong in the same
        # `runs/<name>-H-<stamp>/`. `Config.hash()` is untouched and nothing is
        # excluded from it: a run started WITHOUT the sentinel simply has a
        # different H and a different directory, which is correct and visible
        # in `--diff` rather than a bug.
        mode = cfg["loop.resume_from"]
        if adopt is not None:
            target = Path(adopt)
            if not target.is_dir():
                raise ConfigError(f"--resume: no such run directory: {target}")
        elif mode is not None:
            target = checkpoint.find_adoptable(Path(out_root), cfg["name"], cfg.hash())
        if target is not None:
            _assert_same_config(target, cfg, extend_to=extend_to)
            lock = checkpoint.claim(target, force)       # refuses or steals
            # An extension is about to overwrite `status.json` with "running".
            # Keep what it said first: if the extension then fails, the search it
            # was extending had ALREADY FINISHED, and its terminal status is the
            # record of that. Losing it turns a successful search into a failed
            # one for reasons that have nothing to do with the search.
            if full_iterations:
                try:
                    prior_status = json.loads((target / "status.json").read_text())
                except (OSError, ValueError):
                    prior_status = None
            rundir = RunDir.adopt(target)
            leg = rundir.leg
            # A run directory written without a frozen task_spec.yaml (for example by
            # an older version of BIRD) may be adopted -- see `_assert_same_config` --
            # but WITHOUT this it would never gain a copy, so the exemption would become
            # permanent for that directory rather than covering one leg, and every
            # future resume of it would go unchecked, which is the opposite of the
            # intent. Freeze it now, from leg 2 onward, and say in the journal that
            # leg 1 was not covered -- an unguarded leg that leaves no trace is
            # indistinguishable from a guarded one.
            if _task_spec_for(cfg) is not None and \
                    not (rundir.path / "task_spec.yaml").exists():
                _write_task_spec(rundir, cfg)
                rundir.event("task_spec_backfilled", leg=leg, reason=(
                    "this run directory has no frozen task_spec.yaml, so no leg before "
                    "this one was checked against a task definition"))
        else:
            rundir = RunDir(out_root, cfg["name"], cfg.hash(), out_root_source=out_root_source)
            rundir.write_config(cfg.to_yaml(), cfg.lineage)
            _write_task_spec(rundir, cfg)
            if mode is not None:
                lock = checkpoint.claim(rundir.path, force=False)
                checkpoint.append_resume_record(
                    rundir.path, {"event": "fresh_start",
                                  "reason": "no adoptable run dir for this config"})
        # The lock is a heartbeat too, and it has to beat on a CLOCK rather
        # than at iteration boundaries -- one metaworld iteration is hours,
        # which is many times `RESUME_MIN_IDLE_S`, so a per-checkpoint
        # heartbeat leaves a healthy run looking dead for most of its life.
        checkpoint.start_heartbeat(lock)
        log.info("run dir: %s (leg %d)", rundir.path, leg)

    if rundir is None:
        return _execute(cfg, budget, rundir)
    try:
        if target is not None:
            # The strict refusal happens HERE -- before a single LLM call --
            # and lands in `status.json` as `failed`, because the rundir is
            # already adopted by this point.
            #
            # Strictness has two sources and they mean different things.
            # `loop.resume_from: auto_degraded` is a CONFIG choice -- "this
            # search is willing to run lenient", made up front, hashed, and
            # therefore its own point in the config space. `--resume-degraded`
            # is a one-shot OPERATOR choice about one directory in front of
            # them, and it is a flag precisely so it cannot move the hash: the
            # alternative (an override on the resumed config) would leave a
            # directory named `<name>-H-<stamp>` whose config was no longer the
            # one that produced H.
            strict = cfg["loop.resume_from"] != "auto_degraded" and not allow_degraded
            plan = checkpoint.load_plan(rundir, strict=strict)
            if extend_to is not None:
                # BESIDE the frozen artifact, never over it. `config.resolved.yaml`
                # is write-once, so it stays the exact configuration the run started
                # under; the new budget goes in its own file and in the audit log --
                # the same shape as `result.pre-extend.json`.
                (rundir.path / "config.extended.yaml").write_text(cfg.to_yaml())
                checkpoint.append_resume_record(rundir.path, {
                    "event": "extended_iteration_budget",
                    "to_iteration": int(extend_to),
                    "from_iteration": (plan.next_iteration if plan is not None else None)})
            if extend_to is not None and plan is not None:
                # The same re-entry `--full-iterations` needs, for the same reason:
                # a loop that ENDED looks finished, so without this the extension
                # skips the loop and reports success having added nothing. The two
                # flags stay orthogonal -- this one raises the budget, that one
                # suppresses an early-stop rule -- and compose when both are given.
                plan = _reenter_for_full_iterations(plan, cfg, "--extend-iterations")
            if full_iterations and plan is not None:
                plan = _reenter_for_full_iterations(plan, cfg)
                # An extension that leaves no trace is indistinguishable from a
                # search that simply ran longer, so say it in the audit log AND
                # keep the outcome it is superseding.
                checkpoint.append_resume_record(rundir.path, {
                    "event": "extended_full_iterations",
                    "from_iteration": plan.next_iteration,
                    "to_iteration": cfg["loop.n_iterations"],
                    "suppressed_termination": cfg["loop.termination"]})
                # COPIED, not renamed. `_execute` overwrites `result.json` on
                # success, so the copy costs nothing there -- and if the
                # extension leg dies instead, the run still HAS a `result.json`.
                # Renaming would leave `status.json: failed` beside no result at
                # all, with the real prior outcome under a name nothing looks
                # for, so a reader would take the failure of the extension for
                # the loss of the search it was extending.
                prior = rundir.path / "result.json"
                if prior.exists():
                    (rundir.path / "result.pre-extend.json").write_bytes(
                        prior.read_bytes())
            if plan is None:
                # NOT gated on `adopt is not None`.
                # The directory is already adopted -- `leg` is 2 and
                # `journal.jsonl` is open in append mode -- so falling through
                # here runs a COMPLETE fresh search into it: two searches
                # concatenated in one journal, `state/iterNN.json` overwritten,
                # `budget.json` holding only the second half, and `result.json`
                # reporting `resumed: true, legs: 2, degraded: []`. That is the
                # "silent restart at zero" this module's docstring says cannot
                # happen, and guarding only `--resume` would leave the `auto`
                # path open to it.
                #
                # `find_adoptable` accepts a directory on the mere EXISTENCE of
                # a `ckpt-*.json`; whether one is readable is decided here.
                # Refusing is also what keeps a requeue honest -- bumping
                # `SCHEMA_VERSION` under a live sweep should stop it, not
                # silently re-buy every hour it had already paid for.
                raise checkpoint.CheckpointError(
                    f"{target}: no usable checkpoint under {target}/checkpoints, and the "
                    "directory is already adopted -- restarting at iteration 0 inside it "
                    "would concatenate two searches into one set of artifacts. See "
                    "resume.jsonl for every file that was rejected and why; move the "
                    "directory aside to start this search fresh.")
        result = _execute(cfg, budget, rundir, plan=plan, leg=leg, lock=lock,
                          full_iterations=full_iterations,
                          extend_to=extend_to)
    except BaseException as exc:          # KeyboardInterrupt included, on purpose
        rundir.finish("failed", exc)
        _restore_status_after_failed_extension(rundir, prior_status, exc)
        rundir.close()
        checkpoint.release(lock)
        raise
    rundir.finish("ok")
    rundir.close()
    checkpoint.release(lock)
    return result


def _task_spec_for(cfg: Config):
    """The spec backing this config's environment, or None if it has none.

    Every registered environment has one today, so None means the catalogue and the
    registry disagree -- which `_check_coherence` reports by name. It is still a real
    answer rather than an exception, because a run directory written without a frozen
    task_spec.yaml must stay adoptable.
    """
    from bird import tasks

    try:
        return tasks.by_env_id(str(cfg["problem.env_id"]))
    except tasks.TaskSpecError:
        return None


def _write_task_spec(rundir: RunDir, cfg: Config) -> None:
    from bird import tasks

    spec = _task_spec_for(cfg)
    if spec is None:
        return
    reduction = cfg.get("evaluate.fitness.reduction")
    anchors = ((spec.anchors or {}).get("by_reduction") or {}).get(reduction) or {}
    rundir.write_task_spec(spec.id, spec.path.read_text(), {
        "spec_sha256": spec.sha256,
        "tasks_root": str(tasks.TASKS_ROOT),
        "env_id": spec.env_id,
        "library": spec.library,
        "reduction": reduction,
        "anchors": {k: v.get("value") for k, v in anchors.items()},
        "anchor_n_episodes": {k: v.get("n_episodes") for k, v in anchors.items()},
        # Recorded, never inherited: the spec's budget is a claim about the TASK, and
        # `train.env_steps` is a decision about this experiment. A run far below the
        # task's stated budget is a fact to write down, not a refusal -- and
        # `learning_verified` says whether that budget was ever established at all.
        "spec_budget_train_steps": (spec.budget or {}).get("train_steps"),
        "config_train_env_steps": cfg.get("train.env_steps"),
        "learning_verified": (spec.budget or {}).get("learning_verified"),
    })
    rundir.event("task_spec_resolved", task_id=spec.id, sha256=spec.sha256,
                 reduction=reduction,
                 spec_budget_train_steps=(spec.budget or {}).get("train_steps"),
                 config_train_env_steps=cfg.get("train.env_steps"))


def _assert_only_the_iteration_budget_moved(path: Path, cfg: Config, extend_to: int) -> None:
    """`--extend-iterations`: the ONE divergence from the frozen config that is allowed.

    WHY THIS NEEDS A HOLE AT ALL. `loop.n_iterations` is inside `Config.hash()`,
    and `_assert_same_config` compares the whole resolved YAML -- so a search
    created with a budget of 10 could never be continued to 20 by any route.
    `--full-iterations` does not help: it runs an early-stopped search out to the
    budget it already has and says so ("it cannot move the config hash"). The
    budget was therefore fixed at directory-creation time forever, which is a
    property nobody chose; it fell out of the hash covering everything.

    WHY IT IS SAFE TO OPEN EXACTLY THIS ONE. Raising the iteration count adds
    iterations to the same search under the same method -- it does not change
    what the method IS, which is the property the frozen config protects. Every
    other key stays compared byte for byte, and the check is one-directional: a
    LOWER budget is refused, because truncating a finished search and calling it
    the same run would silently discard iterations the artifact still contains.

    THE ARTIFACT IS NOT REWRITTEN. `config.resolved.yaml` stays write-once, so it
    keeps meaning "this is what the run was configured with". The new budget is
    recorded beside it in `config.extended.yaml` and in `resume.jsonl`, the same
    way `result.pre-extend.json` keeps the outcome an extension supersedes.
    """
    import yaml as _yaml

    stored = _yaml.safe_load(path.read_text())
    current = _yaml.safe_load(cfg.to_yaml())
    was = ((stored or {}).get("loop") or {}).get("n_iterations")
    if was is None:
        raise checkpoint.CheckpointError(
            f"{path} has no loop.n_iterations to extend")
    if int(extend_to) <= int(was):
        raise checkpoint.CheckpointError(
            f"--extend-iterations {extend_to} does not extend {path}: it already ran "
            f"to {was}. A budget may only go UP -- lowering it would drop iterations "
            "this run directory still holds while calling itself the same run")
    # Compare with the budget normalised away. Anything else that moved means this
    # is a different experiment wearing the same directory.
    probe = _yaml.safe_load(path.read_text())
    probe["loop"]["n_iterations"] = int(extend_to)
    if probe != current:
        diff = sorted(k for k in set(_flat_keys(probe)) | set(_flat_keys(current))
                      if _flat_keys(probe).get(k) != _flat_keys(current).get(k))
        raise checkpoint.CheckpointError(
            f"--extend-iterations may only change loop.n_iterations, but "
            f"{len(diff)} other key(s) differ from {path}: {diff[:8]}. Refusing to "
            "adopt: the resolved config is a frozen input and this would be a "
            "different experiment in the same directory")


def _flat_keys(d, prefix=""):
    """`{'a.b': 1}` from `{'a': {'b': 1}}`, so a refusal can NAME what moved."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat_keys(v, key + "."))
        else:
            out[key] = v
    return out


def _assert_same_config(target: Path, cfg: Config, extend_to: Optional[int] = None) -> None:
    """`config.resolved.yaml` is written ONCE and never rewritten.

    It cannot differ given the hash matched, unless the YAML dumper changed
    underneath us -- in which case refusing is right. Refusing also keeps the file
    the exact configuration the run started under.
    """
    path = target / "config.resolved.yaml"
    if not path.exists():
        raise checkpoint.CheckpointError(f"{target} has no config.resolved.yaml to adopt")
    if extend_to is not None:
        _assert_only_the_iteration_budget_moved(path, cfg, extend_to)
    elif not path.read_text().endswith(cfg.to_yaml()):
        raise checkpoint.CheckpointError(
            f"{path} does not match the config being resumed; refusing to adopt "
            "(the resolved config is an artifact and is never rewritten)")

    # The task definition is the other frozen input, and it is NOT in the config hash --
    # so this is the only place a changed spec can be caught. Adopting across one is
    # adopting across a changed experiment.
    frozen, current = target / "task_spec.yaml", _task_spec_for(cfg)
    if current is not None and frozen.exists():
        # A MISSING copy is not evidence of a changed task -- it means the directory
        # was written without a frozen task_spec.yaml (for example by an older
        # version of BIRD), and the config hash (the primary identity) still
        # matched. Refusing would strand such directories to guard against nothing.
        # A copy that EXISTS and differs is the case worth refusing.
        if frozen.read_text() != current.path.read_text():
            raise checkpoint.CheckpointError(
                f"{frozen} is not the task definition this run would now load "
                f"({current.id}, sha256 {current.sha256[:12]}); refusing to adopt. The "
                "task spec is a frozen input, exactly like config.resolved.yaml")


def _restore(ctx: Context, plan: "checkpoint.ResumePlan", leg: int) -> Dict[str, Any]:
    """Adopt a checkpoint. Supplies the initial state and NOTHING else.

    Whatever could not be restored is loud in four places -- the log, the
    `resume` journal event, `status.json.resumes[]` / `result.json.degraded[]`,
    and `budget.resume_degradations` -- in the spirit of `record_skip` and
    `failure_kind`: a silently degraded run is a run that reports the ablation
    as if it were the method.
    """
    cfg = ctx.cfg
    ctx.counters = dict(plan.counters)          # candidate ids, cursors, RAPP bounds
    ctx.rng.setstate(plan.rng)
    ctx.budget.restore(plan.budget, plan.prior_elapsed_s)
    ctx.budget.record_discarded(plan.discarded_trainings, plan.discarded_env_steps)
    if plan.degraded:
        # DISTINCT losses, not distinct sentences: one evicted `policy_ref`
        # is named once at checkpoint time and once per holder at load time,
        # and `budget.py` defines this counter as one per dropped slot.
        ctx.budget.record_resume_degradation(checkpoint.degradation_count(plan.degraded))
    checkpoint.import_stores(plan.policy_store, plan.replay_store)
    if plan.env_states is not None and getattr(ctx.env, "import_states", None) is not None:
        ctx.env.import_states(plan.env_states)

    record = {"leg": leg, "seq": plan.seq, "phase": plan.phase,
              "from_restart": plan.restart, "from_iteration": plan.next_iteration,
              "degraded": plan.degraded, "dropped": plan.dropped,
              "rejected": plan.rejected,
              "discarded_trainings": plan.discarded_trainings,
              "discarded_env_steps": plan.discarded_env_steps}
    if cfg["llm.generator.provider"] != "mock" or cfg["llm.evaluator.provider"] != "mock":
        # Two FRESH runs against a real provider are already not identical, so
        # resume adds nothing there and there is no equality claim to make.
        # Said out loud rather than left as an assumption about the reader.
        log.warning("resume_determinism: provider_nondeterministic -- a resumed run is as "
                    "reproducible as two fresh runs against this provider, and no more")
        record["resume_determinism"] = "provider_nondeterministic"
    for entry in plan.degraded:
        log.warning("resume DEGRADED: %s -- %s", entry.get("slot"), entry.get("reason"))
    ctx.event("resume", **record)
    if ctx.rundir is not None:
        ctx.rundir.record_resume(record)
        checkpoint.append_resume_record(ctx.rundir.path, {"event": "resumed", **record})
    log.info("resumed leg %d from checkpoint %d (%s): restart %d, iteration %d",
             leg, plan.seq, plan.phase, plan.restart, plan.next_iteration)
    return record


def _ground_truth_of(report: Any) -> Optional[Dict[str, Any]]:
    """The returned candidate's GROUND-TRUTH score, on the one scale every env shares.

    `returned_fitness` is whatever `evaluate.fitness.source` produced, and across the
    methods that is four different quantities -- `ground_truth_metric`, `vlm_score`,
    `preference_bt`, or nothing at all. They are not comparable to each other, and a
    results table that puts them in one column named `fitness` invites exactly the
    error it looks like it is preventing: a `preference_bt` score beside a
    `ground_truth_metric` one can read as several times worse while, on ground truth,
    both runs returned equally good candidates.

    So this records `env.task_metric` for the returned candidate alongside it. Nothing
    new is computed -- the learner already measured it per seed -- but otherwise it
    lives only in `candidates/*/train_result.json`, so any consumer wanting a common
    scale would have to re-derive it and know which field to reach for.

    EVERY STATISTIC IS NAMED, and that is the point rather than a detail. There are two
    fields called `fitness` in this artifact -- one per seed, one per report -- reduced
    by different rules, and `evaluate.fitness.checkpoint_aggregation` applies only to
    the second. A ground-truth number that did not say whether it was a final, a max or
    an area under the curve would recreate the trap in the file written to close it.
    """
    res = getattr(report, "result", None) if report is not None else None
    seeds = [m for m in (getattr(res, "seed_metrics", None) or [])
             if isinstance(m, dict)]
    if not seeds:
        return None

    def _mean(key: str) -> Optional[float]:
        vals = [float(m[key]) for m in seeds
                if isinstance(m.get(key), (int, float))]
        return (sum(vals) / len(vals)) if vals else None

    return {
        # Mean over `train.seeds_per_candidate`; `n_seeds` is here so a single-seed
        # number is never mistaken for a replicated one.
        "n_seeds": len(seeds),
        "final": _mean("final"),          # the SHIPPED checkpoint: the last one unless
                                          # `train.checkpoint_selection` restored an
                                          # earlier one -- also `seed_metrics.fitness`
        "max": _mean("max"),              # best checkpoint reached
        "auc": _mean("auc"),              # mean over the checkpoint curve
        "success_rate": _mean("success_rate"),
        "source": "env.task_metric",
        "aggregation_over_seeds": "mean",
    }


def _execute(cfg: Config, budget: Budget, rundir: Optional[RunDir],
             plan: Optional["checkpoint.ResumePlan"] = None, leg: int = 1,
             lock: Optional["checkpoint.Lock"] = None,
             full_iterations: bool = False,
             extend_to: Optional[int] = None) -> Dict[str, Any]:
    """The run itself. Everything that can fail lives here so that `run` can be
    a wrapper thin enough to be obviously correct."""
    ctx = Context(cfg=cfg, budget=budget, rundir=rundir, rng=random.Random(cfg["seed"]))
    ctx.generator = registry.get("llm", cfg["llm.generator.provider"])(ctx, role="generator")
    ctx.evaluator = registry.get("llm", cfg["llm.evaluator.provider"])(ctx, role="evaluator")
    # BEFORE THE ENV IS BUILT, AND ONLY FOR THE JAX FAMILY. The adapter's
    # `mjx.put_model` initialises the XLA backend inside its own `__init__`,
    # so this is the last moment the determinism flag can still bite -- one
    # line later is already too late, and the flag would be recorded on every
    # seed row while doing nothing.
    #
    # Gated on the suite rather than set unconditionally: `XLA_FLAGS` is
    # process-global and a numpy-tier run has no business changing kernel
    # selection for anything else in the process.
    #
    # `bird.xla_env`, not `bird.components.fasttd3`, because every run on
    # every tier passes this line and fasttd3 imports torch at module scope.
    from bird.xla_env import prepare_for_env
    prepare_for_env(cfg["problem.env_id"])
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    # `problem.horizon`: truncate the episode to the method's own length, or
    # keep the environment's. `apply_horizon` is the ONE place the rule lives
    # (bird/envs/base.py) and it runs immediately after construction, before
    # anything reads `env.horizon`.
    from bird.envs.base import apply_horizon as _apply_horizon
    _apply_horizon(ctx.env, cfg)  # (effective, requested); the seed rows read the adapter
    ctx.human = registry.get("phase", "human_oracle")(ctx)
    ctx.tracker = make_tracker(ctx)
    ctx.checkpointer = checkpoint.Checkpointer(
        rundir, cfg, lock=lock, enabled=cfg["loop.resume_from"] is not None,
        seq=plan.seq if plan is not None else 0)

    resume_info = _restore(ctx, plan, leg) if plan is not None else None
    if full_iterations:
        # AFTER `_restore`, and that ordering is load-bearing: `Budget.restore`
        # SETS counters from the checkpoint snapshot rather than adding to them,
        # so a counter incremented before it is silently wiped by the restore it
        # is supposed to accompany.
        budget.record_resume_extension()

    log.info("=" * 72)
    log.info("BIRD  %s   (%s)", cfg["name"], " <- ".join(reversed(cfg.lineage)))
    log.info("  %s | %d iter x %d cand x %d seed | screen=%s | fitness=%s",
             cfg["problem.env_id"], cfg["loop.n_iterations"], cfg["generate.n_candidates"],
             cfg["train.seeds_per_candidate"], cfg["verify.quality_screen"],
             cfg["evaluate.fitness.source"])
    log.info("=" * 72)

    # A pre-phase's outputs live in `ctx.counters` (RAPP writes `rapp_bounds` /
    # `rapp_sweep`, DR generation writes `dr_configs` / `dr_selected`) and that
    # dict is restored, so a completed pre-phase is skipped rather than re-paid.
    # Under DrEureka the RAPP sweep is the most expensive thing in the run.
    done_pre = set(plan.pre_phases_done) if plan is not None else set()
    cp = ctx.checkpointer
    cp.pre_phases_done = list(done_pre)
    for phase_spec in cfg["pre"] or []:
        name = phase_spec["name"] if isinstance(phase_spec, dict) else phase_spec
        if name in done_pre:
            log.info("pre-phase: %s (restored from checkpoint)", name)
            continue
        log.info("pre-phase: %s", name)
        registry.get("phase", name)(ctx, None)
        cp.pre_phases_done.append(name)
    if cfg["pre"]:
        _ckpt(ctx, "pre_done")

    # The budget as the restart loop begins, carried forward across legs. It is
    # the only wind-back point for "restart 0 is gone and will be re-run", and
    # a running total cannot supply it -- see `checkpoint._plan_from`.
    cp.loop_start_budget = (plan.loop_start_budget if plan is not None
                            and plan.loop_start_budget else checkpoint.budget_blob(budget))

    states: List[RunState] = list(plan.restart_states) if plan is not None else []
    cp.completed_restarts = list(range(len(states)))
    for restart in range(len(states), cfg["loop.n_restarts"]):
        if cfg["loop.n_restarts"] > 1:
            log.info("-" * 72)
            log.info("restart %d/%d", restart + 1, cfg["loop.n_restarts"])
        use = plan if (plan is not None and plan.state is not None
                       and plan.restart == restart) else None
        states.append(run_search(ctx, restart, resume=use,
                                 full_iterations=full_iterations))
        written = None
        if cp.enabled:
            # Write-once and never garbage-collected: a completed restart is
            # immutable, and `final_artifact` needs the whole `RunState`
            # (`best` / `latest` / `archive`), not its summary. Its budget goes
            # with it, because that is the only record of what the search had
            # spent by the end of this restart.
            written = checkpoint.write_restart(rundir, restart, states[-1],
                                               checkpoint.budget_blob(budget))
        # The return value is CHECKED. `write_restart` swallows its own
        # failures, so recording the restart regardless left every later
        # checkpoint naming a file that does not exist.
        cp.record_restart(restart, written)
        _ckpt(ctx, "restart_done", restart=restart)
        plan = None
    _ckpt(ctx, "loop_done")

    # The returned reward is the argmax over the restarts' final artifacts, and
    # the state handed to every `post:` phase is THAT restart's, so what the
    # phases retrain, rate and report is the reward `result.json` names.
    # Handing them `states[-1]`, the LAST restart, would make
    # `final_retrain.json` describe a different reward from `returned_cand_id`
    # whenever `loop.n_restarts > 1` and the winner came from an earlier
    # restart. Moot at one restart; the contract at `phases.run_final_retrain`
    # ("retrain the returned reward") is not.
    finals = [(s, final_artifact(ctx, s)) for s in states]
    finals = [(s, f) for s, f in finals if f is not None]
    best_state, best = max(
        finals, key=lambda sf: (sf[1].fitness if sf[1].fitness is not None else float("-inf"))
    ) if finals else ((states[-1] if states else None), None)

    # On the eureka tester point with two restarts, retraining the last restart
    # instead picks a different candidate on 7 of 10 seeds. The phase records the
    # restart it was handed (`final_retrain.json: restart`) and `returned_restart`
    # below names the same one, so the join is checkable from the artifacts.
    #
    # A post phase is a REPORT PROTOCOL over a search that has already finished,
    # and its failure must not replace the search's outcome: an exception
    # escaping here before `result` below is composed would make `run()` stamp
    # `status.json: failed` and write no `result.json` -- a completed search
    # indistinguishable from one that never finished. Recorded
    # per phase in `result.json: post_phase_errors` (a phase that handles its own
    # failure, as `final_retrain` does, files it in `ctx.counters` under the same
    # key), journalled, and the remaining phases still run: `alignment_rate`
    # already reports `not_run` when the retrain left it nothing to rate.
    post_errors: Dict[str, str] = {}
    for phase_spec in cfg["post"] or []:
        name = phase_spec["name"] if isinstance(phase_spec, dict) else phase_spec
        log.info("post-phase: %s", name)
        try:
            registry.get("phase", name)(ctx, best_state)
        except Exception as exc:  # noqa: BLE001 -- recorded on the result, see above
            post_errors[name] = f"{type(exc).__name__}: {exc}"
            log.exception("post-phase %s failed; the search's own result is kept", name)
            ctx.event("post_phase_failed", phase=name, error=post_errors[name],
                      error_type=type(exc).__name__)
    for name, err in (ctx.counters.get("post_phase_errors") or {}).items():
        post_errors.setdefault(str(name), str(err))

    result: Dict[str, Any] = {
        "name": cfg["name"],
        "config_hash": cfg.hash(),
        "lineage": cfg.lineage,
        "final_artifact_rule": cfg["select.final_artifact"],
        "returned_cand_id": best.cand_id if best else None,
        "returned_fitness": best.fitness if best else None,
        # Which restart the returned artifact came from -- the one the post
        # phases acted on. `per_restart` below summarises every restart; this
        # names the winner among them.
        "returned_restart": (int(best_state.restart) if best is not None else None),
        # `{phase: "ExcType: message"}` for every `post:` phase that failed. The
        # search's own numbers above stand regardless; a consumer reading a
        # report protocol's output (`phases/*.json`) checks here first.
        "post_phase_errors": post_errors,
        # WHAT `returned_fitness` IS. Four different quantities share that field
        # across methods; without this a reader would have to open
        # `config.resolved.yaml` to find out which one a run produced -- and a
        # table built from `result.json` alone could not find out at all.
        "returned_fitness_source": (best.fitness_source if best else None)
                                   or cfg["evaluate.fitness.source"],
        # The rule that reduced the checkpoint curve to that scalar. It applies to
        # the REPORT-level fitness only; `seed_metrics[].fitness` is always the
        # SHIPPED checkpoint -- the last one unless `train.checkpoint_selection`
        # restored an earlier one -- regardless of this value.
        "fitness_checkpoint_aggregation": cfg["evaluate.fitness.checkpoint_aggregation"],
        # The common scale, so cross-method comparison never needs the above.
        "returned_task_metric": _ground_truth_of(best),
        # A null task metric and a null fitness already say nothing was
        # measured; these two say WHY. `returned_trained: false` with a
        # `returned_skip_reason` is a training that was NEVER LAUNCHED -- the
        # generate-only last iteration of `loop.termination: fixed_generations`
        # (CARD; `skip_reason` names it) or a TPE rejection under
        # `verify.tpe.on_failure: skip_training` (the candidate's
        # `failure_kind: "screened"` names that one). `returned_trained: true`
        # with a null metric is a training that launched and measured nothing.
        # The two populations `failure_kind` keeps apart per candidate, kept
        # apart here too.
        "returned_trained": (bool(best.result.trained)
                             if best is not None and best.result is not None else None),
        "returned_skip_reason": (best.result.skip_reason
                                 if best is not None and best.result is not None else None),
        "returned_reward_code": best.candidate.reward_code if best else None,
        "per_restart": [s.summary() for s in states],
        "budget": budget.report(),
        # Three independent greppable markers say a run was resumed; this is
        # one, `status.json.leg` is the second and the journal's `resume` lines
        # are the third.
        "legs": leg,
        "resumed": leg > 1,
        "degraded": list(resume_info["degraded"]) if resume_info else [],
        # A ONE-SHOT OPERATOR ACTION IS RECORDED IN BOTH PLACES, which is what
        # `--resume-degraded`'s own help promises for its own ("counted in
        # budget.resume_degradations and named in result.json"). `resume.jsonl`
        # says it happened;
        # this says THIS NUMBER was produced under it. Without the second, a
        # finished extended run reads as a config that says stop at plateau, a
        # journal showing the full budget, and `resumed: true` -- which is also
        # exactly what an ordinary resumed run says, leaving the only tell the
        # ABSENCE of a file (`result.pre-extend.json`) nothing looks for.
        #
        # It matters beyond tidiness: an extended run and a run that reached
        # `n_iterations` under its own rule are different TREATMENTS, and pooling
        # them silently mixes two treatments with nothing saying so.
        "extended_full_iterations": bool(full_iterations),
        # None when the budget was never raised, so a reader can tell "ran its
        # original 20" from "was created at 10 and extended".
        "extended_iteration_budget": (int(extend_to) if extend_to is not None else None),
        # The rule that was overridden, named here because
        # `config.resolved.yaml` still carries it and now points the wrong way
        # for this run. Empty rather than the rule's name when nothing was
        # suppressed, so the key never asserts an override that did not happen.
        "suppressed_termination": (cfg["loop.termination"] if full_iterations
                                  else ""),
        # The bound this design does NOT pretend to close: the iteration in
        # flight when a leg died spent LLM calls and tokens that no journal
        # event carries (`llm/base.py` records them into the Budget, not the
        # journal), so `budget.json` under-reports LLM spend by AT MOST one
        # iteration per leg. Printed rather than hidden. The RL half of the same
        # iteration IS recovered, into `resume_discarded_*`.
        "budget_unaccounted_iterations": leg - 1,
    }
    log.info("=" * 72)
    log.info("returned %s (rule=%s) fitness=%s",
             result["returned_cand_id"], cfg["select.final_artifact"],
             result["returned_fitness"])
    log.info("cost: %d trainings (%d never launched: screen/allocator/generate-only, "
             "%d invalid candidates), "
             "%d LLM calls, %d tokens, %d human queries",
             budget.policy_trainings, budget.policy_trainings_skipped,
             budget.candidates_invalid,
             budget.llm_calls, budget.total_tokens, budget.human_queries)
    log.info("=" * 72)

    ctx.tracker.log_summary(ctx, result, budget)
    ctx.tracker.finish()

    if rundir:
        rundir.save_budget(budget)
        rundir.save_result(result)
    return result


# ==========================================================================
# CLI
# ==========================================================================


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="bird.py",
        description="Benchmark for Iterative Reward Design -- one algorithm, one config space.")
    p.add_argument("--config", "-c",
                   help="a config file path, a path relative to configs/ "
                        "(methods/eureka, hillclimb/v2_verify), or a bare name "
                        "(eureka, era_u, v2_verify) when exactly one file under "
                        "configs/ has it; see --list-configs")
    p.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config key (repeatable)")
    p.add_argument("--profile", "-p", metavar="NAME",
                   help="layer configs/_profiles/NAME.yaml between the defaults and the "
                        "method config: 'how to execute' as a second parent, which "
                        "extends: (single inheritance) cannot express. The name is "
                        "recorded in the resolved config, so it is inside the run hash")
    p.add_argument("--out", default=None,
                   help="output root (default: the config's output.dir, itself runs/)")
    p.add_argument("--dry-run", action="store_true",
                   help="validate and print the resolved config; run nothing")
    p.add_argument("--print-config", action="store_true",
                   help="print the resolved config and exit")
    p.add_argument("--list-configs", action="store_true",
                   help="every config you can pass to --config, in labelled "
                        "sections: the published methods (configs/methods/), the "
                        "ERA recipes, the hill-climb (configs/hillclimb/) and "
                        "examples (configs/examples/)")
    p.add_argument("--diff", nargs=2, metavar=("A", "B"),
                   help="show the keys on which two configs differ")
    p.add_argument("--validate-all", action="store_true",
                   help="load and validate every config, then exit")
    # Flags, NOT config keys: they are one-shot operator actions about a lock
    # and about printing, not properties of the search, so they cannot touch the
    # config hash.
    p.add_argument("--resume", metavar="RUNDIR",
                   help="adopt an existing run directory, using ITS "
                        "config.resolved.yaml verbatim (mutually exclusive with -c)")
    p.add_argument("--resume-force", action="store_true",
                   help="steal the resume lock even if its heartbeat is fresh")
    p.add_argument("--resume-degraded", action="store_true",
                   help="accept a checkpoint that lost method-defining state, once. "
                        "The loss is counted in budget.resume_degradations and named "
                        "in result.json, exactly as under loop.resume_from=auto_degraded")
    p.add_argument("--full-iterations", action="store_true",
                   help="resume an EARLY-STOPPED search and run it out to "
                        "loop.n_iterations, suppressing the early-stop rule for the "
                        "iterations below that budget. Requires --resume. The "
                        "suppression applies to every restart this leg runs, not "
                        "only the one being continued; a search in which an "
                        "already-completed restart stopped short is refused rather "
                        "than partly extended. Counted in budget.resume_extensions "
                        "and named in result.json. A flag, not a config key, so it "
                        "cannot move the config hash")
    p.add_argument("--extend-iterations", type=int, metavar="N",
                   help="continue a finished or interrupted search with a HIGHER "
                        "loop.n_iterations. Requires --resume. The only key allowed "
                        "to differ from the run's frozen config -- every other "
                        "divergence is refused and named -- and it may only go UP. "
                        "config.resolved.yaml is NOT rewritten; the new budget lands "
                        "in config.extended.yaml, resume.jsonl and result.json's "
                        "extended_iteration_budget. A flag, not a config key, for the "
                        "reason --full-iterations is: it must not move the hash, or "
                        "the extended run would land in a different directory from "
                        "the one it is extending")
    p.add_argument("--resume-info", metavar="RUNDIR",
                   help="print a run directory's checkpoint state and exit; runs nothing")
    args = p.parse_args(argv)

    from bird.config import CONFIG_ROOT, PROFILE_DIR

    if args.resume_info:
        print(json.dumps(checkpoint.describe(Path(args.resume_info)), indent=2, default=str))
        return 0

    if args.list_configs:
        # LABELLED SECTIONS, because the files are not interchangeable:
        # `methods/` holds the published methods (each file's header cites where
        # its values come from), while the ERA recipes, the hill-climb points and
        # the examples are OURS by construction, and citing one of those as a
        # published method is the single worst thing this listing could invite.
        # Every line is something `--config` accepts: a bare name for a method or
        # a recipe, and `<dir>/<stem>` for the hill-climb and the examples so the
        # directory stays visible where it says what a file is. The comment lines
        # are `#`-prefixed so `--list-configs | grep -v '^#'` is still a list of
        # `--config` arguments. `_profiles/` is hidden by its leading underscore;
        # `sweeps/` (written on demand by `scripts/ablate.py`) is derived rather
        # than authored and is not listed.
        _sections = (
            ("methods", "", ("configs/methods/ -- the published methods and their published",
                             "variants, plus `zeroshot`, the one-shot root every config",
                             "extends. Each file's header cites where its values come from.")),
            ("", "", ("configs/ -- the two recipes introduced in the paper: ERA-U and",
                      "ERA-S (ERA-U with the fitness source switched to the environment's",
                      "own signal). Ours, not published methods.")),
            ("hillclimb", "hillclimb/", (
                "configs/hillclimb/ -- OUR hill-climb points (v1 ... v4_peak_noes40)",
                "and their chain parents. NOT published points: each extends another",
                "config, so `bird.py --diff <point> <parent>` is the whole statement",
                "of what it changes.")),
            ("examples", "examples/",
             ("configs/examples/ -- small illustrative configs, not methods.",)),
        )
        first = True
        for sub, prefix, blurb in _sections:
            files = sorted(f for f in (CONFIG_ROOT / sub).glob("*.yaml")
                           if not f.name.startswith("_"))
            if not files:
                continue
            # `print("#")`, not `print()`: a bare blank line would survive
            # `| grep -v '^#'` as an empty `--config` argument.
            if not first:
                print("#")
            first = False
            for line in blurb:
                print(f"# {line}")
            for f in files:
                print(f"{prefix}{f.stem}")
        return 0

    if args.diff:
        a, b = (load(args.diff[0], profile=args.profile),
                load(args.diff[1], profile=args.profile))
        d = a.diff(b)
        print(f"{len(d)} key(s) differ between {a['name']} and {b['name']}:\n")
        width = max((len(k) for k in d), default=0)
        for k, (va, vb) in d.items():
            print(f"  {k:<{width}}  {va!r:>28}  ->  {vb!r}")
        return 0

    if args.validate_all:
        ok = True
        for f in sorted(CONFIG_ROOT.rglob("*.yaml")):
            # A profile is a LAYER, not a config: on its own it has no method to
            # be, so loading one as if it were would validate an overlay against
            # the defaults and report a pass for something nobody can run. The
            # directory check is on the path parts, not on `f.name`, because a
            # profile is named for the tier (`tester.yaml`) and nothing about the
            # filename says which directory it came out of.
            if f.name.startswith("_") or PROFILE_DIR in f.relative_to(CONFIG_ROOT).parts:
                continue
            try:
                load(f, profile=args.profile)
                print(f"  ok    {f.relative_to(CONFIG_ROOT)}")
            except ConfigError as exc:
                ok = False
                print(f"  FAIL  {f.relative_to(CONFIG_ROOT)}\n{exc}")
        return 0 if ok else 1

    if args.resume and args.config:
        p.error("--resume and --config are mutually exclusive: --resume uses the run "
                "directory's own config.resolved.yaml")
    if args.resume and args.profile:
        # A resolved config is already fully layered, so a profile under it would
        # be overridden key for key and change nothing -- except the recorded
        # `profile`, and therefore the hash, and therefore which run directory
        # this is. Refuse rather than silently adopt the wrong one.
        p.error("--resume and --profile are mutually exclusive: the run directory's "
                "config.resolved.yaml already carries the profile it was run under")
    if not args.config and not args.resume:
        p.error("--config is required (or use --resume / --list-configs / --diff "
                "/ --validate-all)")

    if args.extend_iterations is not None and not args.resume:
        p.error("--extend-iterations requires --resume <RUNDIR>: it raises the budget "
                "of one existing search, it does not start one")
    if args.extend_iterations is not None and args.extend_iterations < 1:
        p.error("--extend-iterations must be a positive iteration count")
    if args.full_iterations and not args.resume:
        p.error("--full-iterations requires --resume <RUNDIR>: it continues one "
                "existing early-stopped search, it does not start one")

    overrides = parse_overrides(args.set)
    if args.resume:
        # The resolved config is fully resolved (no `extends:`) and already
        # carries `resume_from`, so the hash matches by construction and no
        # override can be forgotten at 3 a.m.
        resolved = Path(args.resume) / "config.resolved.yaml"
        if not resolved.exists():
            raise ConfigError(f"--resume: {resolved} does not exist")
        if args.extend_iterations is not None:
            # Set HERE rather than asking the operator for `-s` as well: the flag is
            # the statement of intent, and a mismatch between the two would be a
            # silent way to extend to a number nobody asked for.
            overrides["loop.n_iterations"] = int(args.extend_iterations)
        cfg = load(resolved, overrides=overrides)
    else:
        cfg = load(args.config, overrides=overrides, profile=args.profile)

    if args.print_config:
        print(cfg.to_yaml())
        return 0

    if args.dry_run:
        setup_logging(cfg["output.log_level"])
        log.info("config %s is valid (%d keys, profile %s, hash %s)",
                 cfg["name"], len(cfg.flat()), cfg.get("profile") or "none", cfg.hash())
        return 0

    run(cfg, out_root=args.out, adopt=args.resume, force=args.resume_force,
        allow_degraded=args.resume_degraded, full_iterations=args.full_iterations,
        extend_to=args.extend_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
