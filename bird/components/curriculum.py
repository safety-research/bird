"""Curriculum: an ORDERED progression of skills, gated stage by stage (unpublished).

**This is not RDA's decomposition, and the difference is the whole module.**
`generate.decomposition` splits a task into J subtasks *once per search*
(`configs/methods/rda.yaml`'s own comment says so), and every iteration then trains one
reward covering all of them while `_vlm_trajectory_analysis` asks the judge how each
aspect looks. The subtasks are CONCURRENT ASPECTS and the policy is always
attempting the whole task. A curriculum is SEQUENTIAL STAGES with a progression
gate: train stage k to competence, keep the policy, then move to stage k+1.
Same J, different loop -- RDA asks *"how is each aspect going?"*, a curriculum
asks *"which aspect are we allowed to work on yet?"*.

None of the reproduced methods does the second thing, and the ordering plus the
gate is the new axis. What already existed and is reused rather than rebuilt:
`train.init: warm_start_from_best` (carry the policy across a handover),
`train.init: secondary_replay_buffer`, `generate.decomposition.*` (a list to
order), `_vlm_trajectory_analysis`'s judge, and `pre`/`post` phases.

## Related work

The published methods and recent work are not silent on curricula, and the
distinctions matter, because each entry below differs from this axis on at
least one key:

* **Eureka §4.3 / App. D.1 uses a curriculum**, and it is the nearest published
  relative of this axis. Pen spinning is split in two -- first re-orient the pen
  to random targets under the final Eureka reward, then fine-tune THAT policy on
  the spinning task under the same reward -- against a `Scratch` control that
  skips stage 1. So: **human-authored, two stages, no gate (each stage runs to
  its budget), one reward reused across both**, with warm-start as the only
  handover mechanism. It is expressed here as `author: subtask_list` over a
  two-item list with `gate: fixed_budget`.
  That is the acceptance criterion this repo is built on -- a method that cannot
  be expressed is a missing knob -- and it is the reason `fixed_budget` is
  registered rather than dismissed as the weak option.
* **DrEureka** says "DrEureka does not need a reward curriculum" and its
  Human-Designed baseline uses a *velocity* curriculum. That is a different
  sense of the word -- a schedule INSIDE one reward, not an ordering OVER
  skills -- and conflating the two would put a knob in the wrong section.
* **LEACL** (arXiv 2607.23515, July 2026) is the closest recent work: an LLM
  decomposes a long-horizon manipulation task and an automatic-curriculum
  algorithm solves the subtasks in sequence. It differs from this axis in four
  ways that are each a key below -- one sparse reward is reused for every stage
  rather than a reward generated per stage, progression is an ACL algorithm's
  competence estimate rather than a judge, there is no stated policy carry, and
  it has neither a stall path nor a forgetting check.

So the combination here -- **a reward generated per stage, a VLM ensemble as the
gate, re-splitting a stalled stage, and a regression rollback** -- is unclaimed,
and this module is an unpublished direction rather than a published method
point. For that reason there is deliberately no config file for it; the
axis is exercised offline by `tests/test_curriculum.py` instead.

## The four forks, and how each was settled

1. **Who authors it.** The LLM, with a second agent reviewing until they agree
   (`author: llm_reviewed`). The curriculum is therefore a searched artifact and
   is recorded per run -- `CurriculumState.history` carries the rounds. The
   offline/deterministic point is `author: subtask_list`, which orders RDA's own
   decomposition and asks no model anything.
2. **What gates progression.** The judge, as in RDA, but as an **ensemble of
   `gate_votes` calls** rather than one -- a handover is irreversible for the
   rest of the search and a single noisy score is not evidence for one.
3. **What happens when a stage never passes.** `on_stall`. The default is
   `resplit`: the LLM is asked to break the stalled stage into smaller ones,
   because the stated priority is to work THROUGH the stages in order -- a
   policy that does part 1 well beats one that fails at everything. `advance`
   and `abort` are the other two points, and `advance` plus `fixed_budget` is
   the pair that structurally cannot stall.
4. **Budget.** Not matched against RDA or Eureka, on purpose: this is a
   different question (long-horizon multi-stage tasks), not a cheaper way to
   answer theirs. A curriculum arm and a vanilla arm are not comparable and no
   config here invites the comparison.

## The powerlift trap, which this module makes VISIBLE and does not fix

`h1hand_powerlift`'s shipped reward is `0.2*(small_control*stand_reward) +
0.8*reward_dumbbell_lifted`, so **standing still scores ~0.2 for ever**. A
curriculum whose stage 1 is "stand up" trains the policy to competence at
precisely the behaviour that already absorbs this task's reward and then
warm-starts stage 2 from it. That could be the fix or it could entrench the
local optimum with a principled-sounding reason, and **a gate cannot tell you
which** -- a judge asked "did it stand up?" is satisfied by standing still, and
correctly so, because standing IS stage 1. What separates the two outcomes is
evidence, so every handover writes one: the gate's score and votes, the frames
it saw (`judgments`), the policy ref taken forward, and every later stage's
score on the same rollout. "Stage 1 passed" is never the only thing recorded.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import judgments
from ..observability import sample_frames
from ..registry import register
from ..state import CurriculumState
from ..types import CandidateReport
from .evaluation import _parse_reason, _parse_score, client_sees_images, judge_concurrency
from .phases import _ask, _parse_list

log = logging.getLogger("bird")

#: Ceiling on what a re-split may grow the curriculum to. `resplit` calls an LLM
#: and inserts what it returns, so an unbounded loop of stalled stages could
#: grow the list without limit and never reach the end -- which is the failure
#: `on_stall` exists to prevent, reintroduced by its own remedy.
MAX_STAGES = 24

#: Offline stand-in when nothing usable comes back. Two stages, not one: a
#: one-stage curriculum is not a curriculum and would make every gate, stall and
#: regression path unreachable while still reporting `curriculum: enabled`.
_MIN_STAGES = 2


# ==========================================================================
# helpers
# ==========================================================================


def _cfg(ctx: Any, key: str, default: Any = None) -> Any:
    return ctx.cfg.get(f"loop.curriculum.{key}", default)


def _int(value: Any, default: int) -> int:
    """Schema marks several of these nullable; a null must not crash the loop."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _stage_patience(ctx: Any, cur: CurriculumState) -> int:
    """Iterations one stage may spend before it counts as stalled.

    `null` means DERIVED, and derived means an even split of the run:
    `n_iterations // n_stages`, floored at 1. It is null by default because the
    useful value is a function of two other keys, and a hard-coded integer would
    silently become wrong the moment either moved.

    **Derived ONCE and then frozen on the curriculum** (`CurriculumState.
    patience`), and that is not an optimisation. `on_stall: resplit` grows
    `n_stages`, so a live derivation makes the remedy for a stalled stage shrink
    the budget of every stage it creates -- each then stalls sooner and splits
    again. Measured on this axis's first end-to-end run: 7 -> 9 -> 13 stages in
    three iterations with nothing trained to completion and no stage passed. An
    explicit `stage_patience` is read live, because an operator who set a number
    means that number.
    """
    explicit = _cfg(ctx, "stage_patience")
    if explicit is not None:
        return max(1, _int(explicit, 1))
    if cur.patience > 0:
        return cur.patience
    total = _int(ctx.cfg.get("loop.n_iterations"), 1)
    cur.patience = max(1, total // max(1, len(cur.stages)))
    return cur.patience


def _stage_frames(ctx: Any, report: CandidateReport) -> Tuple[List[bytes], Dict[str, Any]]:
    """The pixels the gate judges one handover on, and why not more.

    `_pair_frames`' sibling and gated the same three ways -- `videos` in
    `evaluate.artifacts`, a client that would actually look (`client_sees_images`
    : an `images=` keyword AND `modality: vlm`), and a rollout to render. A
    blind gate is legal and is recorded as blind: the judge still answers from
    the text digest, and a handover decided without pixels must not be
    indistinguishable in the artifact from one decided with them. The pairwise
    judge records the same distinction.

    Sampled ONCE per report and cached on it, because the ensemble asks
    `gate_votes` questions of the same rollout and every previously-passed stage
    asks one more -- the pixels do not change with the question.
    """
    cached = report.meta.get(_FRAMES_KEY)
    if cached is not None:
        return cached

    prov: Dict[str, Any] = {}

    def out(png: List[bytes]) -> Tuple[List[bytes], Dict[str, Any]]:
        judgments.note_frames(
            ctx, prov, iteration=int(getattr(report.candidate, "iteration", -1)),
            cand_id=report.cand_id, rollout=0, png=png)
        pair: Tuple[List[bytes], Dict[str, Any]] = (png, prov)
        report.meta[_FRAMES_KEY] = pair
        return pair

    def blind(reason: str) -> Tuple[List[bytes], Dict[str, Any]]:
        prov.update(judgments.frame_set((), None, blind_reason=reason))
        return out([])

    if "videos" not in (ctx.cfg.get("evaluate.artifacts") or []):
        return blind("evaluate.artifacts does not list videos")
    if not client_sees_images(getattr(ctx, "evaluator", None)):
        return blind("the evaluator client takes no images or is not modality: vlm")
    trajs = list(getattr(report.result, "trajectories", None) or ())
    if not trajs:
        return blind("no rollout was recorded for this agent")
    n_images = max(0, _int(ctx.cfg.get("evaluate.vlm.images_per_query"), 20))
    # NOT `blind(...)` on an empty result: `sample_frames` has already put the
    # SPECIFIC reason in `prov` and a summary would throw away the only part a
    # reader can act on. `_pair_frames` and `_vlm_frames` draw the same line.
    return out(list(sample_frames(ctx, trajs[0], n_images, prov=prov)))


#: Private cache key on `CandidateReport.meta`, dropped by the phase before it
#: returns. Frames on a report reach `RunState`, `save_report` and
#: `checkpoint.encode` -- which has a `bytes` tag, so twenty PNGs per candidate
#: would BLOAT a checkpoint rather than crash it, and nothing would report it
#: (`preferences._drop_clips` guards the same trap).
_FRAMES_KEY = "_curriculum_frames"


def _score_stage(ctx: Any, state: Any, report: CandidateReport, stage: str,
                 *, index: int, role: str) -> Tuple[Optional[float], str, int, str]:
    """One judge call: how completely does this rollout achieve THIS stage?

    A score on `evaluate.feedback.score_scale`, normalised to [0,1] here so a
    threshold in the config means one thing across scales -- `fitness_vlm_score`
    normalises for the same reason. `None` means the judge produced nothing
    parseable, and it is kept distinct from `0.0`: a gate that read an
    unanswered call as "stage failed" would hold a curriculum at stage 0 for a
    whole run on a provider outage, which is exactly the never-passes case
    `on_stall` was written for and NOT the case it should be reporting.

    The question is a score rather than a PASS/FAIL word on purpose. The
    regression check needs a MAGNITUDE -- "is part 1 still working" is a
    comparison against the score it passed with, not an absolute -- and one
    parser shared with RDA's judge cannot disagree with itself about what the
    model said.
    """
    from .evaluation import _scale_bounds, judge_digest

    lo, hi = _scale_bounds(ctx)
    png, prov = _stage_frames(ctx, report)
    trajs = list(getattr(report.result, "trajectories", None) or ())
    digest = judge_digest(trajs[0]) if trajs else "(no rollout)"
    seen = (f"\n{len(png)} frames sampled evenly from the rollout are attached."
            if png else "")
    prompt = (
        f"Task: {ctx.cfg.get('problem.task_description', '')}\n"
        f"This agent is being trained on ONE STAGE of that task, in order.\n"
        f"Stage {index + 1} under judgement: {stage}\n"
        f"Rollout: {digest}{seen}\n"
        f"Score how completely the agent achieves THIS STAGE -- not the whole "
        f"task, and not a later stage -- on a {lo}-{hi} scale.\n"
        f"Then give a one-sentence reason for the score.")
    query_id = judgments.note_query(
        ctx, iteration=int(getattr(state, "iteration", -1)), role=role,
        prompt=prompt, vlm=bool(png), cand_id=report.cand_id, stage=stage,
        stage_index=index, n_images=len(png),
        frame_set=str(prov.get("id") or ""),
        blind_reason=str(prov.get("blind_reason") or ""))
    from .evaluation import _client_text

    text = _client_text(getattr(ctx, "evaluator", None), prompt, images=png)
    raw = _parse_score(text, lo, hi)
    reason = _parse_reason(text)
    score = None if raw is None else (
        (raw - lo) / (hi - lo) if hi > lo else float(raw))
    judgments.note_response(
        ctx, iteration=int(getattr(state, "iteration", -1)), query_id=query_id,
        raw=text, parsed=score, vlm=bool(png), role=role, stage=stage,
        stage_index=index, reason=reason)
    return score, reason, len(png), str(prov.get("blind_reason") or "")


def _ensemble(ctx: Any, state: Any, report: CandidateReport, stage: str,
              *, index: int, role: str, votes: int) -> Dict[str, Any]:
    """`votes` independent judgments of one stage, folded in ASK ORDER.

    Order matters even though the fold is a mean: the reason kept is the
    lowest-scoring one (RDA's rule -- the failure mode is what the next prompt
    has to act on), and a completion-order fold would swap which of two equal
    scores supplies it. `ThreadPoolExecutor.map` preserves input order, which is
    the property being relied on.
    """
    jobs = list(range(max(1, votes)))

    def one(_i: int) -> Tuple[Optional[float], str, int, str]:
        return _score_stage(ctx, state, report, stage, index=index, role=role)

    workers = judge_concurrency(ctx, len(jobs))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="bird-curriculum") as pool:
            outcomes = list(pool.map(one, jobs))
    else:
        outcomes = [one(i) for i in jobs]

    scores = [s for s, _r, _n, _b in outcomes if s is not None]
    worst = ""
    low = None
    for s, r, _n, _b in outcomes:
        if s is not None and r and (low is None or s < low):
            low, worst = s, r
    n_images = outcomes[0][2] if outcomes else 0
    blind_reason = outcomes[0][3] if outcomes else ""
    return {
        "votes": [None if s is None else round(s, 4) for s, _r, _n, _b in outcomes],
        "score": (sum(scores) / len(scores)) if scores else None,
        "n_answered": len(scores),
        "reason": worst,
        "n_images": n_images,
        "blind_reason": blind_reason,
    }


# ==========================================================================
# curriculum_author -- who writes the ordered stage list
# ==========================================================================




#: Fence artifacts a stage must never be. `_parse_list` tries the fenced JSON
#: first and falls through to its prose parser when the payload is EMPTY -- so
#: `{"stages": []}` comes back as the literal lines `json` and `{"stages": []}`.
#: That hazard is `_parse_list`'s and belongs to all five of its callers;
#: `run_decompose` is merely masked from it because a short list gets padded
#: from `_fallback_subtasks`. Here it would splice fence text into the
#: curriculum as stage names -- a stage the judge is then asked to score. The
#: fix stays local rather than changing a parser five other call sites depend on.
_NOT_A_STAGE = {"json", "python", "py", "stages", "```"}


def _clean_stages(items: Sequence[str]) -> List[str]:
    """Keep only items that could be a described BEHAVIOUR.

    Deliberately crude and one-directional: it drops obvious non-prose rather
    than trying to validate a stage, because the alternative to a wrong stage
    here is a shorter curriculum, and the alternative to a dropped good stage is
    nothing worse than that either.
    """
    out: List[str] = []
    for raw in items:
        item = str(raw).strip().strip("`").strip()
        if len(item) < 4 or item.lower() in _NOT_A_STAGE:
            continue
        if item[0] in "{}[]\"'" or ('"stages"' in item):
            continue
        if item not in out:
            out.append(item)
    return out


def _seal(ctx: Any, state: Any, cur: CurriculumState) -> CurriculumState:
    """Freeze the derived patience and say out loud if the run cannot finish.

    Every author ends here, so the two facts a curriculum needs before its first
    gate are established in one place rather than once per author.

    The feasibility warning is not a `_check_coherence` rule and cannot be: the
    stage COUNT does not exist until a model has been asked, so the config is
    valid and the run is still infeasible. With `n_iterations` below `n_stages`
    the derived patience floors at 1 and the last stages are unreachable no
    matter what the policy does -- a run that will report `curriculum: enabled`,
    spend its whole budget on the first stage or two, and look like a method
    result. Journalled as well as logged, because a warning buried in a
    20-hour log is easily missed.
    """
    # The EFFECTIVE patience, which is not always the frozen one.
    # `_stage_patience` returns an explicit `stage_patience` without freezing it
    # -- freezing is only for the derived value, whose whole purpose is not to
    # move when a resplit changes `n_stages`. So `cur.patience` stays 0 under an
    # explicit config, and recording THAT here would say "patience 0" for a run
    # whose patience is 2. A record of 0 is not a smaller number than 2, it is a
    # false one.
    effective = _stage_patience(ctx, cur)
    total = _int(ctx.cfg.get("loop.n_iterations"), 1)
    feasible = total >= len(cur.stages)
    cur.history[-1]["patience"] = effective
    cur.history[-1]["feasible"] = feasible
    if not feasible:
        log.warning("    curriculum: %d stage(s) over %d iteration(s) -- at "
                    "patience %d the last stage(s) cannot be reached whatever "
                    "the policy does",
                    # `effective`, not `cur.patience`, for the same reason as
                    # the record above: under an explicit `stage_patience`,
                    # `cur.patience` would print "at patience 0" for a run whose
                    # patience is 2, and the warning is the only thing an
                    # operator sees at submit time.
                    len(cur.stages), total, effective)
    return cur


@register("curriculum_author", "subtask_list")
def author_subtask_list(ctx: Any, state: Any) -> CurriculumState:
    """Order RDA's own decomposition, and ask no model anything.

    The offline, deterministic point on this axis, and the one that makes the
    tester tier able to exercise every gate, stall and regression path without a
    provider. It is also how a HUMAN-authored curriculum is expressed today: a
    spec or a config supplies the list, this member only fixes its order.

    Reads `state.subtasks` when a decomposition has already run and falls back
    to splitting the instruction, exactly as `run_decompose` does -- a poor
    curriculum on purpose, rather than pretending a model said something.
    """
    from .phases import _fallback_subtasks

    stages = [s for s in (getattr(state, "subtasks", None) or []) if str(s).strip()]
    if len(stages) < _MIN_STAGES:
        stages = _fallback_subtasks(str(ctx.cfg.get("problem.task_description") or ""),
                                    _MIN_STAGES)
    stages = stages[:MAX_STAGES]
    cur = CurriculumState(stages=list(stages), entered_at=int(getattr(state, "iteration", 0)))
    cur.history.append({"event": "authored", "author": "subtask_list",
                        "iteration": int(getattr(state, "iteration", 0)),
                        "stages": list(stages), "rounds": 0})
    return _seal(ctx, state, cur)


@register("curriculum_author", "llm_reviewed")
def author_llm_reviewed(ctx: Any, state: Any) -> CurriculumState:
    """One model proposes an ordered curriculum, a second reviews, repeat to agreement.

    The curriculum is then a SEARCHED artifact rather than a task fact, which is
    why every round is recorded in `history`: two runs of the same config can
    disagree about what the stages were, and a comparison that cannot see the
    stage list is comparing two different experiments.

    ‡ "A second agent" is `ctx.evaluator`, because the schema exposes exactly two
    LLM roles (`llm.generator.*`, `llm.evaluator.*`). A genuinely independent
    reviewer would be a third role, which is a missing knob and is recorded as
    one here rather than faked by calling the same client twice and calling the
    second call a reviewer.

    The proposer is asked to REASON FIRST and then emit the list, which is a
    requirement and not decoration: the ordering claim ("B needs A") is
    the thing the reviewer has to check, and a bare list gives it nothing to
    check. Agreement is the literal token `AGREE`; anything else is a revision
    request and its text becomes the next round's context. `author_max_rounds`
    bounds it, and hitting the bound accepts the last proposal rather than
    failing -- a search that dies because two models kept arguing would be
    worse than a curriculum one of them still disliked, and `history` records
    that it ended on the bound.
    """
    task = str(ctx.cfg.get("problem.task_description") or "")
    rounds = max(1, _int(_cfg(ctx, "author_max_rounds"), 3))
    known = [s for s in (getattr(state, "subtasks", None) or []) if str(s).strip()]
    critique = ""
    stages: List[str] = []
    trace: List[Dict[str, Any]] = []

    for r in range(rounds):
        proposal = _ask(getattr(ctx, "generator", None),
                        _propose_prompt(task, known, critique, r), tag="curriculum_decompose")
        got = _clean_stages(_parse_list(proposal, key="stages"))[:MAX_STAGES]
        if got:
            stages = got
        if not stages:
            trace.append({"round": r, "verdict": "no-proposal", "n_stages": 0})
            continue
        verdict = _ask(getattr(ctx, "evaluator", None),
                       _review_prompt(task, stages), tag="curriculum_review_critique")
        agreed = "AGREE" in (verdict or "").upper()
        trace.append({"round": r, "verdict": "agree" if agreed else "revise",
                      "n_stages": len(stages),
                      "critique": "" if agreed else (verdict or "").strip()[:400]})
        if agreed:
            break
        critique = verdict or ""

    if len(stages) < _MIN_STAGES:
        # Every round produced nothing usable. Degrade to the offline author
        # rather than to a one-stage list: a curriculum of one is not a
        # curriculum, and it would report `enabled` while no gate could fire.
        log.warning("    curriculum: no usable stage list after %d round(s); "
                    "falling back to the offline author", rounds)
        cur = author_subtask_list(ctx, state)
        cur.history[-1]["fallback_from"] = "llm_reviewed"
        cur.history[-1]["rounds"] = trace
        return cur  # already sealed by the author it degraded to

    cur = CurriculumState(stages=list(stages),
                          entered_at=int(getattr(state, "iteration", 0)))
    cur.history.append({"event": "authored", "author": "llm_reviewed",
                        "iteration": int(getattr(state, "iteration", 0)),
                        "stages": list(stages), "rounds": trace,
                        "agreed": bool(trace and trace[-1].get("verdict") == "agree")})
    return _seal(ctx, state, cur)


def _propose_prompt(task: str, known: Sequence[str], critique: str, round_index: int) -> str:
    head = (
        "Design a CURRICULUM for a reinforcement-learning agent: an ordered list "
        "of stages, each a skill that can be trained to competence on its own, "
        "and each one a prerequisite for the next. The agent keeps its policy "
        "between stages, so a later stage builds on the behaviour the earlier "
        "one established.\n\n"
        f"Task: {task}\n")
    if known:
        head += ("\nThe task has already been decomposed into these aspects "
                 "(they are NOT ordered and NOT necessarily your stages):\n"
                 + "\n".join(f"  - {s}" for s in known) + "\n")
    if critique:
        head += ("\nA reviewer rejected your previous proposal and said:\n"
                 + critique.strip() + "\n")
    return head + (
        "\nFirst reason briefly about the ORDER: for each stage after the first, "
        "say which earlier stage it depends on and why it cannot be learned "
        "before it. Then give the final list.\n"
        'Return the list as fenced JSON: {"stages": ["...", "..."]}, first stage '
        "first. Between 2 and 8 stages. Each stage one sentence, describing "
        "OBSERVABLE BEHAVIOUR a judge could see in a video, never a reward term.")


def _review_prompt(task: str, stages: Sequence[str]) -> str:
    listing = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(stages))
    return (
        "You are reviewing a proposed training curriculum written by another "
        "model. Judge only the ORDER and the achievability of each stage.\n\n"
        f"Task: {task}\n\nProposed curriculum:\n{listing}\n\n"
        "Check: is every stage individually achievable by an agent that has "
        "mastered the ones before it and nothing after? Is any stage really the "
        "whole task in disguise? Is any ordering constraint wrong or missing? Is "
        "any stage satisfied by an agent that does nothing?\n"
        "If the curriculum is sound, reply with exactly the word AGREE and "
        "nothing else. Otherwise say what must change, in two sentences.")


# ==========================================================================
# stage_gate -- what advances a stage
# ==========================================================================


@register("stage_gate", "vlm_ensemble")
def gate_vlm_ensemble(ctx: Any, state: Any, report: CandidateReport,
                      stage: str, index: int) -> Dict[str, Any]:
    """`gate_votes` judgments of the current stage; the MEAN decides.

    The judge, as in RDA -- but an ensemble, because a handover is irreversible
    for the rest of the search while a single VLM score is not. The fold is a
    mean over the answering calls rather than a majority over per-call
    thresholds: a majority discards the magnitude, and the magnitude is what the
    regression check compares against later.

    A gate with NO answering call does not pass and does not fail: `score` is
    None and `passed` is False, so the stage is held -- but `n_answered: 0` is on
    the record, so a curriculum stuck behind a dead provider is distinguishable
    from one stuck behind an agent that cannot do the task. Those need opposite
    responses and `on_stall` cannot tell them apart on its own.
    """
    votes = max(1, _int(_cfg(ctx, "gate_votes"), 3))
    out = _ensemble(ctx, state, report, stage, index=index,
                    role="curriculum_gate", votes=votes)
    threshold = float(_cfg(ctx, "gate_threshold") or 0.0)
    out["threshold"] = threshold
    out["passed"] = out["score"] is not None and out["score"] >= threshold
    out["gate"] = "vlm_ensemble"
    return out


@register("stage_gate", "fixed_budget")
def gate_fixed_budget(ctx: Any, state: Any, report: CandidateReport,
                      stage: str, index: int) -> Dict[str, Any]:
    """Advance after `stage_patience` iterations, whatever the policy did.

    The weakest gate and the only one that cannot stall, which is why it is
    registered rather than argued away: paired with `on_stall: advance` it makes
    a curriculum whose progression is a pure schedule, and **that is exactly
    Eureka's published pen-spinning curriculum** (§4.3: stage 1 to its budget,
    then fine-tune). It asks the judge nothing and costs no LLM calls; `score`
    is null because none was measured, never 0.0, which would be a claim.

    It passes on the same boundary `on_stall` fires at, so the stall path is
    unreachable under it by construction rather than by a special case.
    """
    from ..state import CurriculumState as _CS  # noqa: F401  (typing only)

    cur = getattr(state, "curriculum", None)
    spent = int(getattr(state, "iteration", 0)) - int(getattr(cur, "entered_at", 0)) + 1
    budget = _stage_patience(ctx, cur) if cur is not None else 1
    return {"gate": "fixed_budget", "passed": spent >= budget, "score": None,
            "votes": [], "n_answered": 0, "reason": "",
            "n_images": 0, "blind_reason": "this gate asks the judge nothing",
            "spent": spent, "budget": budget}


# ==========================================================================
# stall_action -- what happens when a stage never passes
# ==========================================================================


@register("stall_action", "resplit")
def stall_resplit(ctx: Any, state: Any, cur: CurriculumState) -> str:
    """Ask the LLM to break the stalled stage into smaller ones, in place.

    The stated priority is to work THROUGH the stages in order: a policy that
    does part 1 well is worth more than one that fails at everything, so a stage
    the agent cannot reach is evidence that the stage was too big, not that the
    curriculum should be abandoned. The replacements are spliced in AT THE SAME
    INDEX, so the ordering claim the reviewer checked still holds either side of
    them, and `entered_at` is reset so the first replacement gets a full
    patience of its own.

    Bounded twice, because the remedy can reintroduce the disease. `MAX_STAGES`
    caps the list, and a re-split that returns fewer than two usable stages
    degrades to `advance` -- a stage that stalls and cannot be split is a stage
    to move past, not one to sit on for ever.
    """
    stage = cur.current
    reply = _ask(getattr(ctx, "generator", None), _resplit_prompt(
        str(ctx.cfg.get("problem.task_description") or ""), cur, stage),
        tag="curriculum_decompose")
    parts = _clean_stages(_parse_list(reply, key="stages"))
    room = MAX_STAGES - (len(cur.stages) - 1)
    parts = parts[:max(0, room)]
    if len(parts) < 2:
        log.warning("    curriculum: stage %d stalled and could not be split; "
                    "advancing past it", cur.index + 1)
        return stall_advance(ctx, state, cur)
    cur.stages[cur.index:cur.index + 1] = parts
    cur.splits += 1
    cur.entered_at = int(getattr(state, "iteration", 0)) + 1
    cur.note = ("The previous stage was not reached and has been split into "
                "smaller ones. Design a reward for the FIRST of them only.")
    return "resplit"


@register("stall_action", "advance")
def stall_advance(ctx: Any, state: Any, cur: CurriculumState) -> str:
    """Move to the next stage anyway, and say in the record that it was not passed.

    `history` distinguishes this from a pass, and that distinction is the whole
    reason the trace exists: both are `index += 1`, and a run whose every stage
    was advanced past rather than passed is a failed curriculum that looks
    finished from the index alone.
    """
    cur.index += 1
    cur.entered_at = int(getattr(state, "iteration", 0)) + 1
    cur.note = ("The previous stage was NOT completed and the curriculum moved "
                "on regardless. Do not assume the earlier behaviour is present.")
    return "advance"


@register("stall_action", "abort")
def stall_abort(ctx: Any, state: Any, cur: CurriculumState) -> str:
    """Stop the search. `loop.on_total_failure: abort` is the precedent.

    Raising `BudgetExceeded` is deliberate rather than a new exception type: it
    is the one signal `run_search` already treats as "this search is over, write
    what you have and stop", so a curriculum that gives up leaves the same
    complete artifact as one that ran out of budget.
    """
    from ..budget import BudgetExceeded

    cur.note = "The curriculum aborted here."
    raise BudgetExceeded(
        f"curriculum stage {cur.index + 1}/{len(cur.stages)} never passed "
        f"({cur.current!r}); loop.curriculum.on_stall: abort")


def _resplit_prompt(task: str, cur: CurriculumState, stage: str) -> str:
    return (
        "A reinforcement-learning agent is being trained through a curriculum "
        "and is STUCK: it has spent its whole budget on one stage without "
        "achieving it. Break that stage into smaller ordered stages the agent "
        "can climb.\n\n"
        f"Task: {task}\n"
        f"Full curriculum: " + "; ".join(
            f"{i + 1}. {s}" for i, s in enumerate(cur.stages)) + "\n"
        f"The stage it cannot reach: {stage}\n\n"
        "Return between 2 and 4 replacement stages, in order, the last of which "
        "is equivalent to the stuck stage. Each must be observable behaviour a "
        "judge could see in a video.\n"
        'Fenced JSON: {"stages": ["...", "..."]}')


# ==========================================================================
# the per-iteration step
# ==========================================================================


@register("phase", "curriculum_step")
def run_curriculum_step(ctx: Any, state: Any, selection: Any) -> Any:
    """Gate the current stage, check for regression, and advance or hold.

    Called from `update()` behind `loop.curriculum.enabled`, in the shape
    `update.co_evolve.subtasks` already uses: a config key selects it and the
    registry supplies it, so no stage branches on a method name.

    It runs AFTER `update.topology` and BEFORE `apply_carry`, and both halves
    matter. After topology, because topology's `_carry_inner_loop_refs` sets
    `policy_ref` from `state.best` -- the incumbent, which under elitism is not
    necessarily this round's winner -- and both a stage pass (the judged
    winner's own ref) and a regression rollback overwrite it; the other way
    round, topology would overwrite them. Before `apply_carry`, because the curriculum is a carry
    slot like any other -- a config that does not carry it does not have one,
    and that is the correct reading of `loop.carry`, not a bug to work around.

    **Every exit drops the frame cache**, in a `finally`. Twenty PNGs per report
    ride out on `CandidateReport.meta` into `RunState` otherwise, and
    `checkpoint.encode` has a `bytes` tag -- so the leak bloats a checkpoint
    rather than raising, which is why nothing would report it. `on_stall:
    abort` raises straight through this function, so the `finally` is load-
    bearing rather than tidy.
    """
    cur = getattr(state, "curriculum", None)
    winner = getattr(selection, "winner", None)
    if cur is None or not cur.stages or cur.complete or winner is None:
        return state
    try:
        return _step(ctx, state, cur, winner)
    finally:
        for rep in ([winner] + list(getattr(selection, "losers", None) or [])):
            try:
                rep.meta.pop(_FRAMES_KEY, None)
            except Exception:  # noqa: BLE001 - a report with no meta is not a lost run
                pass


def _step(ctx: Any, state: Any, cur: CurriculumState, winner: CandidateReport) -> Any:
    it = int(getattr(state, "iteration", 0))
    gate = ctx.cfg["loop.curriculum.gate"]
    gate_fn = _get("stage_gate", gate)
    verdict = gate_fn(ctx, state, winner, cur.current, cur.index)
    # `stage_text`, not `stage`: `Context.event(self, stage, **fields)` names its
    # first positional parameter `stage`, so a field of that name is a TypeError
    # at the one moment the journal is the only witness.
    ctx.event("curriculum_gate", iteration=it, stage_index=cur.index,
              stage_text=cur.current,
              **{k: v for k, v in verdict.items() if k != "reason"})

    regressed = _regression(ctx, state, cur, winner)

    if verdict.get("passed"):
        # The policy that PASSED is the one the gate JUDGED: `gate_fn` scored
        # the winner's own rollout, so the winner's own `policy_ref` is what the
        # stage checkpoint records, what `train.init: warm_start_from_best`
        # starts the next stage from, and what a regression rollback restores.
        # NOT `state.policy_ref`: topology's `_carry_inner_loop_refs` has just
        # set that from `state.best`, and under `update.elitism.keep_global_best`
        # (the default) `state.best` moves only on whole-task fitness -- so a
        # winner that passed the stage without beating the incumbent would be
        # recorded, carried and later restored as the INCUMBENT's policy, one
        # the gate never saw. A report with no ref records "" rather than
        # somebody else's ref.
        ref = str(getattr(getattr(winner, "result", None), "policy_ref", None) or "")
        cur.checkpoints[str(cur.index)] = ref
        if ref and not regressed:
            # A rollback this same iteration already chose an older checkpoint
            # and told the generator so; it keeps precedence over the handover.
            state.policy_ref = ref
        if verdict.get("score") is not None:
            cur.pass_scores[str(cur.index)] = float(verdict["score"])
        cur.history.append({
            "event": "passed", "iteration": it, "stage_index": cur.index,
            "stage": cur.current, "score": verdict.get("score"),
            "votes": verdict.get("votes"), "gate": verdict.get("gate"),
            "n_images": verdict.get("n_images"),
            "blind_reason": verdict.get("blind_reason"),
            "reason": verdict.get("reason"),
            "policy_ref": cur.checkpoints[str(cur.index)],
            "spent_iterations": it - cur.entered_at + 1})
        cur.index += 1
        cur.entered_at = it + 1
        if not regressed:
            # Cleared on the ordinary path so a re-split notice or an old
            # regression warning does not follow the generator into a stage it
            # no longer describes -- but NOT when this iteration also found a
            # regression, whose warning is about the policy being carried
            # forward and is therefore exactly what the next stage must read.
            cur.note = ""
        ctx.event("curriculum_advance", iteration=it, to_stage=cur.index,
                  n_stages=len(cur.stages), complete=cur.complete)
        log.info("      curriculum -> stage %d/%d passed (%s); %s",
                 cur.index, len(cur.stages), gate,
                 "curriculum complete" if cur.complete else f"now {cur.current!r}")
        return state

    spent = it - cur.entered_at + 1
    patience = _stage_patience(ctx, cur)
    if spent >= patience:
        before = list(cur.stages)
        action = ctx.cfg["loop.curriculum.on_stall"]
        cur.history.append({
            "event": "stalled", "iteration": it, "stage_index": cur.index,
            "stage": cur.current, "spent_iterations": spent, "patience": patience,
            "score": verdict.get("score"), "n_answered": verdict.get("n_answered"),
            "action": action})
        ctx.event("curriculum_stall", iteration=it, stage_index=cur.index,
                  stage_text=cur.current, spent=spent, patience=patience,
                  action=action)
        taken = _get("stall_action", action)(ctx, state, cur)
        cur.history[-1]["taken"] = taken
        if taken == "resplit":
            # BOTH lists, recorded rather than left to be reconstructed.
            # `resplit` rewrites `cur.stages` mid-run, so the final
            # list is not the list this run was working from at iteration k, and
            # a reader handed only the last one cannot see that the curriculum
            # was re-authored at all. `stages_after` alone is recoverable-in-
            # principle -- chain back through the previous event -- and that is
            # exactly the recomputation `output.trajectory_trace` refuses for
            # the same reason: a chain that goes wrong is indistinguishable on
            # screen from a curriculum that advanced late.
            cur.history[-1]["stages_before"] = before
            cur.history[-1]["stages_after"] = list(cur.stages)
            log.info("      curriculum -> stage %d stalled; split %d -> %d stage(s)",
                     cur.index + 1, len(before), len(cur.stages))
        else:
            log.info("      curriculum -> stage %d stalled; %s", cur.index + 1, taken)
    return state


def _regression(ctx: Any, state: Any, cur: CurriculumState,
                winner: CandidateReport) -> bool:
    """Has the policy forgotten a stage it already passed? If so, roll back.

    Asked of EVERY stage already passed, not only the last one, and with the
    same ensemble the gate uses. A rollback throws away the training since that
    handover, which is destructive enough that a single noisy call is not
    evidence for it -- the cost is `gate_votes` calls per passed stage per
    iteration, it is counted in the budget like every other judge call, and it
    is what `loop.curriculum.regression_check: false` turns off.

    The comparison is against the score the stage PASSED with, not against the
    gate threshold. A stage that passed at 0.95 and now reads 0.75 has lost most
    of the skill while still clearing a 0.7 bar, and an absolute test would
    call that healthy.

    Two things happen on a regression and neither is optional. The policy is
    restored to the checkpoint recorded when the earliest regressed stage passed
    -- earliest, because rolling back to a later one keeps a policy that has
    already forgotten something. And the generator is TOLD, through
    `CurriculumState.note`: a rollback the prompt cannot see is a silent
    rewrite of what the next reward is being written against.
    """
    if not bool(_cfg(ctx, "regression_check", False)) or not cur.pass_scores:
        return False
    tol = float(_cfg(ctx, "regression_tolerance") or 0.0)
    votes = max(1, _int(_cfg(ctx, "gate_votes"), 3))
    it = int(getattr(state, "iteration", 0))
    worst_index: Optional[int] = None
    findings: List[Dict[str, Any]] = []
    for key in sorted(cur.pass_scores, key=lambda k: int(k)):
        k = int(key)
        if k >= cur.index:
            continue  # not yet passed: that is the gate's question, not this one
        out = _ensemble(ctx, state, winner, cur.stages[k], index=k,
                        role="curriculum_regression", votes=votes)
        now = out.get("score")
        if now is None:
            continue  # unanswered is not evidence of forgetting; see `_score_stage`
        was = float(cur.pass_scores[key])
        if now < was - tol:
            findings.append({"stage_index": k, "stage": cur.stages[k],
                             "was": round(was, 4), "now": round(now, 4),
                             "votes": out.get("votes"), "reason": out.get("reason")})
            if worst_index is None:
                worst_index = k
    if worst_index is None:
        return False

    restored = cur.checkpoints.get(str(worst_index), "")
    state.policy_ref = restored or getattr(state, "policy_ref", None)
    cur.history.append({"event": "regressed", "iteration": it,
                        "stage_index": cur.index, "findings": findings,
                        "restored_from_stage": worst_index,
                        "restored_policy_ref": restored})
    cur.note = (
        "REGRESSION: the agent has lost a skill it had already established. "
        + "; ".join(f"stage {f['stage_index'] + 1} ({f['stage']}) scored "
                    f"{f['now']} against {f['was']} when it passed"
                    for f in findings)
        + ". The policy has been restored to the checkpoint from stage "
        f"{worst_index + 1}. Your reward must not trade the earlier behaviour "
        "away for the current stage.")
    ctx.event("curriculum_regression", iteration=it, stages=[f["stage_index"] for f in findings],
              restored_from_stage=worst_index, restored=bool(restored))
    log.warning("      curriculum -> regression on stage(s) %s; policy restored to "
                "the stage-%d checkpoint",
                [f["stage_index"] + 1 for f in findings], worst_index + 1)
    return True


def _get(kind: str, name: str) -> Any:
    from ..registry import get  # local: registry imports this module

    return get(kind, name)
