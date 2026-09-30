"""REvolve's island evolutionary search -- one mechanism, read at two stages.

`generate.sampling_mode: evolutionary_operators` (§1) and
`update.topology: island_lineage` (§6) are two halves of one thing, and they
talk to each other through a single integer stamped on `candidate.meta`. They
live in one module for the same reason §1's `parent_source: archive_sample` and
§6's `sample_archive_parent` share `update.py`: the pair is only correct
together, and a reader who finds one has to be able to find the other.

WHERE THE STATE LIVES. Nowhere new. A deme is `state.archive`'s `_ELITE`
population slots, keyed `(island, "elite", rank)` and written through
`update._set_population` / read through `update._population`
(`bird/components/update.py:670-698`) -- the same one carried dict every other
topology uses, because `RunState` has exactly one carried population slot and
inventing a second outside `CARRY_SLOTS` would make it invisible to
`loop.carry`. A config that runs this method and omits `archive` from
`loop.carry` therefore starts every generation from an empty population, which
is a config error nothing here can catch. The three other places state moves
between the two halves:

    candidate.meta["island"]              §1 -> §6, per offspring; the deme
                                          the cohort was drawn from, or a
                                          generation-0 seed
    ctx.counters["last_sampled_island"]   update.sample_archive_parents -> §1,
                                          within one stage-1 call, cleared on
                                          entry so a stale integer can never be
                                          read as a measured one
    ctx.counters["archive_inserts"]       the migration cadence, incremented
                                          once per ITERATION here

WHAT IS NOT REPRODUCED. Six things, all of them the paper's or the release's
and none of them expressible in this schema today. Each is repeated at the
component that would have owned it, so nobody has to read this list to find
the caveat that applies to them.

  1. **Cross-generation preference pairs.** The paper compares rollouts "not
     only within the same generation but also across different generations,
     updating the fitness scores of all individuals at the end of each
     generation" (Appendix B.1, p.20; the chess-eras footnote 5). BIRD's pair
     pool is this round's eligible reports and stays that way --
     `bird/state.py:13-27` forbids reading rollouts off a carried report, and
     `evaluate.preferences.scope: cumulative` widens the AGGREGATION pool only.
     So an archived individual's games never change here; what `cumulative`
     buys is one normalisation across rounds, which is what makes
     `above_island_mean` a threshold on commensurable numbers at all.
  2. **The softmax parent wheel.** REvolve weights islands and members with
     `exp(x/T)/sum` (`refs/code/Revolve/rewards_database.py:12-14`); BIRD's
     `island_avg_weighted` branch is a shifted linear roulette. Recorded in
     full at `update._pick_island_avg_weighted`.
  3. **REvolve's migration.** `reset_islands`
     (`refs/code/Revolve/rewards_database.py:178-200`) fires at the end of a
     generation with probability 0.3, deletes all but the fittest of the weaker
     half of the islands, and re-seeds each from a random survivor, REMOVING
     the migrant from its source. `update._maybe_migrate` is LIMEN's ring: a
     fixed cadence, a copy rather than a move, and no deme is ever emptied. A
     config states `migration_interval` x `migration_rate` and gets the ring;
     the destructive reset is not expressible and is not approximated.
  4. **The retry that makes a crossover always find two parents.** The release
     resamples islands until `island.size >= num_in_context_samples`
     (`rewards_database.py:253-258`, with its author's own "# TODO: getting
     trapped in the while loop in the initial phases" at `:254`). This sampler
     draws once and falls back to mutation, recorded per candidate in
     `meta["operator_fallback"]`; see `sample_evolutionary_operators`.
  5. **Two separate few-shot counts.** `cfg/generate.yaml:29-30` is
     `few_shot: {mutation: 1, crossover: 2}`. BIRD has one `generate.n_parents`,
     so mutation is pinned at one parent (a multi-parent "mutation" is a
     crossover) and crossover draws `generate.n_parents`. At the published
     `n_parents: 2` the two agree exactly.
  6. **The operator prompt is a SECTION here, not the whole user turn.**
     Upstream swaps `prompts/mutation` for `prompts/crossover` wholesale
     (`refs/code/Revolve/main.py:179`). This module lifts the operative
     sentences of each into one prompt section; the surrounding material --
     environment spec, parent code, component traces, output contract -- is
     BIRD's and is shared by every method. See `_operator_instruction`.

Nothing here reads `cfg["name"]`. The two components are selected by config
value like everything else, and either is usable without the other: the
topology falls back to `update._island_for` for an offspring this sampler did
not stamp, and the sampler works under any `parent_source`.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..config import ConfigError
from ..context import Context
from ..registry import register
from ..state import RunState
from ..types import Candidate, CandidateReport, Selection
from .generation import (
    _TIPS_ANALYSIS,
    _build_candidate,
    _build_messages,
    _get,
    _n_parents,
    _n_this_iteration,
    _n_llm_this_iteration,
    _numeric_reflection,
    _parent_ids,
    _parents_for,
    _two_stage_on,
    _two_stage_meta,
    _two_stage_sample,
)
from .update import (
    NEG_INF,
    _dispatch_operator,
    _dispatch_winner,
    _island_for,
    _maintain_memory,
    _maybe_migrate,
    _population,
    _promotion_blocked,
    _score,
    _set_population,
    _truncate,
    _update_global_best,
    measured_fitness,
)

log = logging.getLogger("bird")


# ==========================================================================
# §1 -- the genetic operators (`generate.sampling_mode: evolutionary_operators`)
# ==========================================================================
#
# The operator instruction is lifted from the two vendored prompt files rather
# than paraphrased, because it is the only thing in this repo that distinguishes
# a mutation from a crossover in words the model reads. `refs/code/Revolve/main.py:179`
# is `operator_prompt = prompts.types[operator]` -- upstream swaps the ENTIRE
# user turn, and the two files disagree about far more than parent count.

#: `refs/code/Revolve/prompts/mutation:5` and `:11-14`, verbatim except for the
#: human-feedback clause (see `_HUMAN_CLAUSE`). Upstream's own preceding
#: sentences describe the fitness score and the per-component traces; those are
#: BIRD's PARENT REWARD and NUMERIC REFLECTION sections and are not repeated
#: here, and upstream's four writing tips are BIRD's
#: `generate.context.include_reward_engineering_tips`.
_MUTATION = (
    "Your task is to iterate on the reward function by mutating a single "
    "component to enhance the agent's performance.\n\n"
    "The mutation process involves the following steps:\n"
    "First, select a reward component based on the given guidelines{human} the "
    "values of individual reward components.\n"
    "Next, clearly explain how you intend to mutate the selected component and "
    "your rationale behind improving the performance. This could involve "
    "adjusting its scale, rewriting its formula, or other modifications.\n"
    "Finally, write the mutated reward function code."
)

#: `refs/code/Revolve/prompts/crossover:5` and `:8-11`. The doubled "to to" in
#: the first sentence is upstream's and is kept, on the same principle as every
#: other verbatim pin here: a prompt we paraphrase is a prompt that cannot be
#: diffed against the paper's Box 3 (p.29).
_CROSSOVER = (
    "Your task is to combine high-performing reward components from different "
    "reward functions and combine them to to enhance the agent's performance.\n\n"
    "The crossover process involves the following steps:\n"
    "First, select high-performing reward components based on{human} the values "
    "of individual reward components.\n"
    "Next, clearly explain how you intend to combine them and your rationale "
    "behind improving the performance.\n"
    "Finally, write the combined reward function code."
)

#: REvolve Auto's prompt delta. `prompts/mutation_auto` differs from
#: `prompts/mutation` in exactly one clause -- the mention of human feedback in
#: the "First, select ..." step (`diff prompts/mutation prompts/mutation_auto`
#: is line 12 alone). The crossover pair does NOT: `diff prompts/crossover
#: prompts/crossover_auto` is FOUR hunks -- the human file's opening sentence
#: ("For each reward function, we collected human feedback on what aspects of
#: the agent are satisfactory and what aspects need improvement.", :1); the
#: clause (:9 in both); the output-format lines and a "Some helpful tips" header
#: (:14-17 vs :14-20); and tip 4's `_get_obs(self)` wording plus a fifth tip
#: ("Make sure you dont give contradictory reward components.", :21) that the
#: auto file lacks. The paper describes the difference in one sentence, and it
#: is not a claim that the boxes differ by one sentence of prompt text: Auto
#: replaces the human feedback "with automatically generated feedback
#: comprising statistics on the reward components tracked during RL training"
#: (refs/tex/revolve/main.tex:441-442). The prompt delta is therefore read off
#: the released files above, not off the paper, and the paper's boxes are not
#: the authority for it.
#:
#: Only the clause is reproduced here (`_HUMAN_CLAUSE`): under
#: `include_human_feedback: true` the human crossover file's opening sentence
#: and fifth tip are NOT rendered -- a recorded gap, not keyed.
#:
#: So the auto arm is NOT a second prompt file here. It is what
#: `generate.context.include_human_feedback: false` already means, which keeps
#: the paper's own ablation a single-key diff instead of a second component.
#: Both alternatives are written out rather than one being the other minus a
#: substring, because the punctuation moves with the clause: dropping only the
#: words leaves mutation reading "the given guidelines, and the values", which
#: is neither file.
_HUMAN_CLAUSE = {
    #             with human feedback        without (the `_auto` prompts)
    "mutation": (", human feedback, and", " and"),
    "crossover": (" the human feedback and", ""),
}


def _operator_instruction(ctx: Context, operator: str) -> str:
    """The mutate-vs-combine instruction, gated exactly as upstream gates it."""
    key = "crossover" if operator == "crossover" else "mutation"
    template = _CROSSOVER if key == "crossover" else _MUTATION
    with_human, without_human = _HUMAN_CLAUSE[key]
    return template.format(
        human=with_human if ctx.cfg["generate.context.include_human_feedback"]
        else without_human)


def _cohort_capable(fn: Callable[..., Any]) -> bool:
    """Does this `parent_source` accept a cohort size?

    The contract this replaces was a duck type -- call `fn(ctx, state, n=want)`
    and catch `TypeError` -- copied from `comparator_human`'s probe of
    `ctx.human` (`preferences.comparator_human`). It is the wrong instrument here.
    A `TypeError` raised INSIDE a cohort-capable parent source (a malformed
    weight in `update.archive.parent_sampling`, a report whose fitness is a
    string) would be caught by that `except` and silently downgrade REvolve's
    one-deme cohort to LIMEN's per-parent draw -- two different methods, no
    trace anywhere, and a plausible run at the end of it. Reading the signature
    cannot confuse "does not take n" with "raised while running".

    `**kwargs` counts, so a future parent source that forwards its arguments is
    not excluded on a technicality.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        # A builtin or C callable, whose signature cannot be introspected.
        # `False` means "not cohort-capable", i.e. fall back to the
        # per-parent `_parents_for` draw -- the conservative direction,
        # because guessing the cohort form wrong would silently hand a
        # crossover two parents from two demes and call it REvolve.
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "n" in params


def _draw_parents(ctx: Context, state: RunState, want: int) -> List[CandidateReport]:
    """`want` parents from `generate.parent_source`, deduped, at most `want`.

    Dedup is this function's, not the source's: `parent_archive_sample`
    (`generation.py:534-562`) makes `n_parents` INDEPENDENT draws and so can
    return the same cell twice, and a "crossover" of a reward with itself is a
    mutation that says otherwise in the artifact. `archive_island_cohort` dedups
    already and this pass is then an identity.
    """
    fn = _get("parent_source", ctx.cfg["generate.parent_source"])
    if _cohort_capable(fn):
        drawn = list(fn(ctx, state, n=want) or [])
    else:
        drawn = _parents_for(ctx, state)
    out: List[CandidateReport] = []
    seen = set()
    for report in drawn:
        if report is None or report.cand_id in seen:
            continue
        seen.add(report.cand_id)
        out.append(report)
    return out[:want]


def _numeric_section(ctx: Context, parents: Sequence[CandidateReport]) -> str:
    """Per-parent component traces -- the one section a cohort genuinely needs.

    `_build_messages:1618-1619` renders `_numeric_reflection(ctx, parents[0])`,
    parent 0 only, which is correct for every sampler that has one parent and
    silently hides half the evidence of a two-parent crossover. REvolve's
    prompt shows the traces of every in-context sample
    (`prepare_in_context_prompt` appends the serialized component history per
    parent; paper §3.2 p.5, Box 3 p.29), and the crossover instruction above
    tells the model to choose between components "based on ... the values of
    individual reward components" -- an instruction that cannot be followed
    when one of the two parents' values are absent.

    The tips / `reflection_guidance` tail is reproduced rather than left to
    `_build_messages`, because `bodies.update(extra)` replaces this section
    wholesale and would drop it. It is reproduced under the SAME condition
    `_build_messages:1620-1633` uses -- parent 0's own rendition being non-empty
    -- so a run cannot end up with the guidance twice (once here and once on
    BEHAVIOURAL ANALYSIS) or with it nowhere.
    """
    ctxcfg = "generate.context."
    bodies = [(p, _numeric_reflection(ctx, p)) for p in parents]
    blocks = [f"Reward {p.cand_id}:\n{body.strip()}" for p, body in bodies if body.strip()]
    if not blocks:
        return ""
    text = "\n\n".join(blocks)
    guidance = ctx.cfg.get(ctxcfg + "reflection_guidance")
    guidance = str(guidance) if guidance and str(guidance).strip() else None
    # THE TIPS BELONG TO EXACTLY ONE SECTION, and which one depends on the
    # parent count -- so this mirrors `_build_messages`' own `grouped` test
    # (`generation.py:1738`) rather than assuming the single-parent shape.
    #
    #   grouped (>1 parent, i.e. a crossover): the base builder appends the tips
    #   to PARENT REWARD (`generation.py:1749-1757`), which `extra` does NOT
    #   overwrite -- so appending them here too renders them TWICE.
    #   ungrouped (a mutation): the base builder would have put them on NUMERIC
    #   REFLECTION, which `extra` DOES overwrite -- so they must be re-added here
    #   or they vanish.
    #
    # Appending unconditionally would render the analysis tips twice in every
    # crossover prompt.
    if len(parents) == 1 and bodies[0][1].strip():   # `_build_messages`' own gate
        if ctx.cfg[ctxcfg + "include_reward_engineering_tips"]:
            text += "\n\n" + (guidance or _TIPS_ANALYSIS)
        elif guidance:
            text += "\n\n" + guidance
    return text


def _human_section(parents: Sequence[CandidateReport]) -> str:
    """Each parent's own human feedback, or "" when no parent has any.

    REvolve's feedback is per-INDIVIDUAL: `format_human_feedback` renders the
    stored positive/negative tag lists for the sampled parent and the prompt
    carries one `human feedback:` field per in-context sample (paper §3.2 p.5,
    Box 3 p.29). `_build_messages:1634-1640` renders `state.human_feedback`,
    which is GT's one-comment-per-iteration channel -- the right thing for a
    single-parent method and the wrong thing for a cohort, where it would
    attribute one comment to two different agents.

    Returning "" (the key is then never put in `extra`) is deliberate and is
    not the same as returning a header with nothing under it: `phases.py:1018`
    writes `report.meta["human_feedback"]` only under
    `evaluate.human.mode: feedback_text`, only for `queries_per_iteration`
    reports, and `_human_targets` returns `live[:1]` under the default
    `applies_to: selected_only` (`phases.py:1007-1008`), so most parents have
    none. An empty section reads to the model as "the human said nothing",
    which is a claim about the human.
    """
    texts = [(p, str(p.meta.get("human_feedback") or "")) for p in parents]
    blocks = [f"Reward {p.cand_id}:\n{t.strip()}" for p, t in texts if t.strip()]
    if not blocks:
        return ""
    return ("A human watched each agent below and said, verbatim -- treat this as "
            "ground truth:\n\n" + "\n\n".join(blocks))


def _seed_island(ctx: Context) -> Optional[int]:
    """Generation 0's island, drawn per INDIVIDUAL.

    `refs/code/Revolve/main.py:158-161` draws `random.choice(range(num_islands))`
    inside the per-individual loop, and `seed_islands` files each individual
    where that draw put it (paper §3.1, p.4). The alternative -- letting §6's
    `_island_for` place a parentless offspring -- is round-robin BY ITERATION
    (`update.py:465`, `int(state.iteration) % n`), so at generation 0 all K
    offspring would land in island 0 and, at `n_islands: 13`, twelve demes would
    never come into existence. On a 2-iteration run that placement gives
    `update_island` island 0 size 16, then island 1 size 16.

    This CONSUMES `ctx.rng`, once per offspring, and
    that is deliberate: `_age_key`'s docstring (`update.py:374-391`) is the
    standing warning that an extra draw shifts every later draw in the stream.
    With one island there is nothing to choose and no draw is made, so the
    degenerate case leaves the stream exactly where it was.
    """
    n = max(int(ctx.cfg.get("update.archive.n_islands", 1) or 1), 1)
    if n < 2:
        return 0
    return int(ctx.rng.choice(range(n)))


@register("sampling_mode", "evolutionary_operators")
def sample_evolutionary_operators(ctx: Context, state: RunState,
                                  backend: Callable[..., List[str]]) -> List[Candidate]:
    """REvolve: one Bernoulli draw per offspring picks mutation or crossover (§1).

    `generate.crossover_rate` is `p_c = 1 - p_m` (paper §3.2, p.5; `p_m = 0.5`
    at §5.2, p.7), drawn ONCE PER OFFSPRING -- which is the reason the key had to
    exist. `update.operator` picks one operator for the whole round and
    `generate.n_parents` is a fixed count; neither can say "half of this
    generation is a recombination". The release agrees on the direction of the
    comparison: `rewards_database.py:244` is
    `operator = "mutation" if random.random() >= self.crossover_prob else "crossover"`,
    i.e. `P(crossover) = crossover_prob` at every value, not only at 0.5.

    **Parents are drawn INSIDE the per-offspring loop, and no other sampler does
    that.** All four samplers in `generation.py` call `_parents_for` once before
    their loop (`:2240, :2299, :2341, :2379`), so their K candidates share one
    parent set and one `parent_id`; here each offspring gets its own cohort out
    of its own deme, which is what an island model means. Two consequences that
    are not obvious:

      * `ctx.rng` consumption differs from every other sampler -- one
        `.random()` per offspring for the operator, plus the parent source's
        draws per offspring, plus one island draw per offspring at generation 0.
        The `tests/test_parallelism.py` and `tests/test_resume.py`
        cross-sections must therefore be MEASURED against this sampler if a row
        is ever added for it, never inherited from another method's row.
      * `candidate.parent_id` is `parents[0].cand_id` per offspring
        (`_parent_ids`, `:2176-2179`), so the lineage in `candidates/*/meta.json`
        is per-candidate. The full cohort is `meta["parent_ids"]`.

    **Iteration 0 is zero-shot, by construction.** The archive is empty until §6
    runs, so no cohort can be drawn, and an offspring with no parent gets NO
    operator instruction and is stamped `operator: init` -- the release's
    generation 0 sends an EMPTY user turn (`refs/code/Revolve/main.py:159-163`
    sets `operator_prompt = ""` and `in_context_samples = (None, None)`;
    `modules.py:69-71` returns "" from `prepare_in_context_prompt` when
    `evolve` is False) and the paper's Initialization subsection
    (`refs/tex/revolve/main.tex:285-287`) calls the seed population GPT-4's
    "zero-shot instruction following". Without this rule every generation-0
    candidate would be told to "iterate on the reward function by mutating a
    single component" with no PARENT REWARD to mutate and stamped
    `operator: mutation`, often with an EMPTY `operator_fallback` (the
    fallback branch only fires for a drawn crossover), so the artifact could
    not tell a forced mutation from a drawn one. The
    Bernoulli draw IS still made at generation 0 and recorded in
    `meta["operator_drawn"]`, so the stream position of every later draw is
    unchanged. Do not read a ~50/50 operator split off a 2-iteration run:
    half of it is `init`.

    **What is overridden in the prompt, and under which flag.** Three sections
    reach `_build_messages` through `extra=`, and `bodies.update(extra)` runs
    last and unconditionally (`generation.py:1648`) -- so an `extra` section
    would otherwise appear even with its `include_*` flag off, which would
    delete the single-key ablation those flags exist for. Each is therefore
    re-gated here on the same flag `_build_messages` would have used:

        GUIDANCE           whenever a parent was drawn -- the operator
                           instruction, and the ONLY section here with no flag
                           of its own, because the operator is this sampler's
                           whole contribution. With no parent (`operator:
                           init`) the key is NOT overridden and
                           `_build_messages` renders `generate.context.guidance`
                           alone (`generation.py:1819-1821`): the zero-shot
                           turn is the base prompt and nothing more.
                           `generate.context.guidance` is preserved VERBATIM
                           above it rather than displaced, so that key stays a
                           real pin. Two divergences are recorded rather than
                           hidden: upstream's operator text is the entire user
                           turn and sits AFTER the parents, where GUIDANCE sits
                           before PARENT REWARD; and it borrows a title rather
                           than adding one, because `_SECTION_ORDER`
                           (`generation.py:1150-1168`) is a closed tuple in a
                           file this module does not own and `_build_messages`
                           SILENTLY DROPS an `extra` title that is not in it
                           (`:1650`). A dedicated "GENETIC OPERATOR" entry
                           there would be better.
        NUMERIC REFLECTION generate.context.include_numeric_reflection
        HUMAN FEEDBACK     generate.context.include_human_feedback, and omitted
                           entirely when no parent carries any

    One prompt-level defect is INHERITED and is not this module's to fix:
    `_parent_section`'s lead line keys off `len(blocks)`, and `blocks` gains a
    second entry for ONE parent's weights (`generation.py:1407-1410`), so a
    single-parent mutation under an output format that carries weights is
    introduced with "These are the parent rewards to combine and improve on:"
    -- against a GUIDANCE section that says to mutate one component. The
    operator section is the authoritative one; the lead line wants a
    `len(parents)` test.

    Returns exactly `_n_this_iteration(ctx, state)` candidates, failures
    included -- `_build_candidate`'s rule, and the reason Eureka's execute rate
    is (executable / requested) rather than (executable / returned).
    """
    cfg = ctx.cfg
    ctxcfg = "generate.context."
    n = _n_llm_this_iteration(ctx, state)
    rate = float(cfg["generate.crossover_rate"])
    out: List[Candidate] = []

    for i in range(n):
        # One draw per offspring, before anything else in the iteration, so the
        # operator is a pure function of the stream position and not of how many
        # parents the archive happened to hold. It is drawn at generation 0 too,
        # where the release draws nothing (`sample_in_context` is never called
        # there, `main.py:159-163`): consuming it keeps every later draw's stream
        # position where it was, and `meta["operator_drawn"]` records it.
        drawn = "crossover" if ctx.rng.random() < rate else "mutation"
        # `few_shot: {mutation: 1, crossover: 2}` (cfg/generate.yaml:29-30) is two
        # counts and BIRD has one key. Mutation is pinned at one parent -- a
        # multi-parent mutation is a crossover -- and crossover reads
        # `generate.n_parents`, so that key is honoured rather than decorative
        # and `_check_coherence`'s `crossover_rate>0 needs n_parents>=2` is a
        # statement about this sampler and not only about the two keys.
        want = _n_parents(ctx) if drawn == "crossover" else 1
        parents = _draw_parents(ctx, state, want)

        operator, fallback = drawn, ""
        if not parents:
            # ZERO-SHOT. No parent, so no operator: there is nothing to mutate
            # and nothing to combine, and an instruction to do either is a
            # claim about a PARENT REWARD section that is not in the prompt.
            # The release's generation 0 sends an EMPTY user turn
            # (`main.py:159-163`, `modules.py:69-71`; paper main.tex:285-287,
            # see the docstring). Without this branch every parentless
            # offspring would fall through to the mutation instruction below.
            # The condition is "no parent
            # was drawn", not "iteration 0", for the reason `island_source`
            # gives: under `parent_source: none` every round is parentless.
            operator = "init"
        elif drawn == "crossover" and len(parents) < 2:
            # The paper never says what a crossover does on a deme with fewer
            # than two members; the release never asks, because it resamples
            # islands until one is big enough (`rewards_database.py:253-258`,
            # and its author's own "# TODO: getting trapped in the while loop in
            # the initial phases" against exactly this case). At I=13 and K=16 a
            # deme with fewer than two members is the MODAL case in early
            # generations, so this branch is the common path, not the corner --
            # which is why it is recorded per candidate rather than logged.
            fallback = (f"crossover drew {len(parents)} distinct parent(s) from "
                        f"{cfg['generate.parent_source']}; ran mutation instead")
            operator = "mutation"
            parents = parents[:1]

        extra: Dict[str, str] = {}
        guidance = cfg.get(ctxcfg + "guidance")
        guidance = str(guidance) if guidance and str(guidance).strip() else ""
        # The configured guidance keeps the TOP of the section, so
        # `## GUIDANCE\n<generate.context.guidance>` still appears verbatim and
        # that key renders exactly where its contract says
        # (`tests/test_prompt_overrides.py:57` asserts that prefix, for a
        # sampler that does not reach here -- but the invariant is the key's,
        # not the test's). The operator instruction follows it: standing
        # guidance first, then what to do with these particular parents. With
        # no parent there is no operator and the key is left alone --
        # `_build_messages` then renders the configured guidance by itself.
        if operator != "init":
            extra["GUIDANCE"] = (guidance + "\n\n" if guidance else "") + \
                _operator_instruction(ctx, operator)
        if cfg[ctxcfg + "include_numeric_reflection"] and parents:
            section = _numeric_section(ctx, parents)
            if section:
                extra["NUMERIC REFLECTION"] = section
        human_parents = 0
        if cfg[ctxcfg + "include_human_feedback"]:
            section = _human_section(parents)
            human_parents = sum(1 for p in parents
                                if str(p.meta.get("human_feedback") or "").strip())
            # UNCONDITIONAL inside the flag, and the empty case is the reason.
            # `_build_messages` falls back to `state.human_feedback` -- GT's ONE
            # comment for the whole iteration, rendered under the same heading
            # with the wording "A human watched the current agent and said" --
            # whenever this key is absent. For a cohort that is a false
            # attribution of one person's comment to two different agents, which
            # is exactly what `_human_section` exists to prevent; leaving the key
            # out on an empty bundle would state the invariant and not enforce it.
            # `""` is safe to assign: `_build_messages` drops a body that is
            # empty after strip, so this suppresses the section rather than
            # printing a blank heading. Under `applies_to: selected_only` (the
            # DEFAULT), omitting the key hands 13 of 16 offspring the GT blob.
            extra["HUMAN FEEDBACK"] = section

        # PARENT REWARD is deliberately NOT overridden. `_parent_section`
        # (`generation.py:1389-1411`) already loops over ALL parents, already
        # switches its lead line at more than one block, and already gates the
        # "(fitness x.xxxx)" label on `evaluate.feedback.state_selection_scalar`.
        # A hand-rolled parent bundle here would bypass that gate and leak the
        # fitness label in a place nobody would look for it.
        parent_id, parent_code = _parent_ids(ctx, parents)

        empty_reason = ""
        two_stage_meta: Dict[str, Any] = {}
        if _two_stage_on(ctx, backend):
            # Four values: the fourth is `empty_reason`, for a Thinker that
            # answered nothing.
            raw, prose, messages, empty_reason = _two_stage_sample(
                ctx, state, backend, parents, None, extra)
            two_stage_meta = _two_stage_meta(raw, empty_reason)
        else:
            messages = _build_messages(ctx, state, parents, extra=extra)
            got = backend(ctx, state, messages, 1)
            raw, prose = ((got[0], "") if got else (None, ""))

        # The deme this offspring belongs to, in the three states it can be in.
        # `last_sampled_island` is cleared on entry to `sample_archive_parents`
        # (update.py), so "absent or None" can never be the PREVIOUS offspring's
        # deme read as this one's.
        stamped = ctx.counters.get("last_sampled_island")
        if stamped is not None:
            island, island_source = int(stamped), "cohort"
        elif not parents:
            # Generation 0 -- REvolve's own rule, per individual. The condition
            # is "no parent was drawn", not "iteration 0": under
            # `parent_source: none` there is never a lineage to inherit, and a
            # fresh random deme every round is the only honest answer there.
            island, island_source = _seed_island(ctx), "seeded"
        else:
            # A parent exists but the branch that drew it names no deme (the
            # global branches, or a non-archive `parent_source`). Leaving the
            # stamp off is not a gap: §6's `_island_for` then inherits the
            # parent's island, which is the lineage rule this topology is named
            # for, and it consumes no `ctx.rng`.
            island, island_source = None, "inherited"

        meta: Dict[str, Any] = {
            "sampling_mode": "evolutionary_operators",
            "sample_index": i,
            "operator": operator,
            "operator_drawn": drawn,
            "operator_fallback": fallback,
            "island": island,
            "island_source": island_source,
            "parent_ids": [p.cand_id for p in parents],
            "prompt_sections_overridden": sorted(extra),
        }
        meta.update(two_stage_meta)  # `thinker_*` fields, as the four generation.py callers record
        if cfg[ctxcfg + "include_human_feedback"]:
            meta["human_feedback_parents"] = human_parents

        # Never construct a Candidate directly: `_build_candidate` is the only
        # caller of `ctx.next_id("c")` and it owns `generate.parse.max_retries`,
        # `_apply_edit_mode`, `_symbol_mapping`, component names/weights and
        # `_apply_co_design` -- six §1 keys a direct construction bypasses.
        out.append(_build_candidate(ctx, state, backend, messages, raw=raw,
                                    parent_id=parent_id, parent_code=parent_code,
                                    nl_spec=prose, meta=meta,
                                    empty_reason=empty_reason or None))
    return out


# ==========================================================================
# §6 -- the lineage islands (`update.topology: island_lineage`)
# ==========================================================================
#
# Table-driven for the same reason `_TRAJECTORY_SELECTORS` (update.py:240) is:
# `update.archive.admission` is an `enum=` key, not a `kind=` key, so its three
# values are selectors over one rule and not three implementations anyone would
# select independently. An unknown value raises rather than defaulting -- the
# gates decide who enters a population, and a typo that quietly became
# `truncate` would be a different method with an identical artifact.


def _admit_truncate(ctx: Context, members: Sequence[CandidateReport],
                    report: CandidateReport) -> bool:
    """Everyone in; `_truncate` alone decides who stays. The pre-existing
    behaviour of every topology, and the default."""
    return True


def _deme_scores(members: Sequence[CandidateReport]) -> List[float]:
    """The finite fitnesses of a deme -- the same `> NEG_INF` and `f == f` test
    `update._fitness_proportional` uses. An unranked occupant (`fitness is
    None`, i.e. `_score` = -inf) does not drag the threshold to -inf, where
    every gate would admit everything."""
    return [s for s in (_score(r) for r in members) if s > NEG_INF and s == s]


def _admit_above_island_mean(ctx: Context, members: Sequence[CandidateReport],
                             report: CandidateReport) -> bool:
    """REvolve's survival rule: insert into island P iff `sigma >= sigma^P`,
    the island AVERAGE (paper §3.3, p.6; `rewards_database.py:118-126`).

    Explicitly the average and not the maximum, "to ensure genetic diversity"
    (footnote 1, p.6) -- which is the paper making its own design argument, and
    is why `above_island_best` exists beside it as a single-key ablation.

    GREATER-OR-EQUAL, so an offspring exactly at the mean enters:
    `rewards_database.py:126` is `if fitness_score >= island_avg_fitness_score`.
    An EMPTY deme admits everything -- upstream reaches the same place by
    handing an empty island `-sys.maxsize - 1` (`:118-123`, and
    `Island.fitness_scores` returns `[-sys.maxsize - 1]` when empty,
    `entities.py:166-168`).
    """
    scores = _deme_scores(members)
    return not scores or _score(report) >= sum(scores) / len(scores)


def _admit_above_island_best(ctx: Context, members: Sequence[CandidateReport],
                             report: CandidateReport) -> bool:
    """The rule footnote 1 (p.6) argues AGAINST, so that the paper's own
    reasoning is reachable as a one-key diff. Published by nobody, and in no
    release -- it is the anti-value, not a second reading."""
    scores = _deme_scores(members)
    return not scores or _score(report) >= max(scores)


_ADMISSION: Dict[str, Callable[[Context, Sequence[CandidateReport], CandidateReport], bool]] = {
    "truncate": _admit_truncate,
    "above_island_mean": _admit_above_island_mean,
    "above_island_best": _admit_above_island_best,
}


def _admission_rule(ctx: Context, state: RunState) -> str:
    """Which gate runs this iteration.

    GENERATION 0 IS UNGATED, whatever the config says. `main.py:236-253` runs
    `add_individuals_to_islands` -- the `sigma >= sigma^P` path -- only
    `if generation_id > 0`; generation 0 goes to `seed_islands`, under the
    comment "for initialization, we don't use this step", and that function's
    own docstring is "for initialization step (generation_id = 0) all
    individuals are added". The paper says the same at §3.1, p.4.

    It is not a detail. Iteration 0 is where a deme first has one member, and a
    gate applied there would measure the second arrival against a mean of one:
    at `n_islands: 13` and `n_candidates: 16` roughly half of generation 0
    would be refused where the release admits all of it, and the population the
    whole search runs on would be half the size the config says.
    """
    rule = str(ctx.cfg.get("update.archive.admission", "truncate") or "truncate")
    if rule not in _ADMISSION:
        raise ConfigError(
            f"update.archive.admission: unknown value {rule!r}; "
            f"expected one of {sorted(_ADMISSION)}")
    return "truncate" if int(state.iteration) == 0 else rule


def _unmeasured(report: CandidateReport) -> str:
    """Why this survivor must NOT be filed into a deme, or "" when it may be.

    A deme holds individuals with a MEASURED fitness and nothing else. §5 does
    not guarantee that: under REvolve's `select.n_survivors: 16` every report
    with a numeric fitness is a winner, and a candidate that never compiled or
    whose training raised reaches §6 carrying `select.failure_value` (-10000,
    `evaluation._blank`), which `selection._scored` keeps because it drops
    `None` alone. At generation 0 the admission rule is forced to `truncate`,
    and at every later generation an EMPTY deme admits anything under both
    gates, so without this check the sentinel would be filed (for example,
    `verify.on_failure: record_neg_inf` puts a -10000.0 candidate into a deme
    at iteration 0). Once in, it poisons both wheels:
    `update._pick_island_avg_weighted` and `update._fitness_proportional`
    shift their weights by the MINIMUM, so one -10000 member makes every
    other deme's weight ~10000 + epsilon -- island selection probabilities of
    0.439/0.195/0.172/0.195 flatten to ~0.20 each -- and within its own deme a
    0.045-vs-0.0 pair goes from 1.00/0.00 to 0.507/0.493.

    The release has no path on which such an individual is filed: an invalid
    function ends `generate_valid_reward` with `None`
    (`refs/code/Revolve/main.py:87-90`) and a training exception propagates
    out of `future.result()` (`modules.py:256`), so `seed_islands` /
    `add_individuals_to_islands` (`main.py:236-253`) only ever see the fitness
    scores of trained policies (`main.py:230-232`).

    Tested via FLAGS (`update.measured_fitness` -- trained cleanly, or carrying
    the screen's own short-run measurement under `evaluate.screened_fitness:
    screen_measurement`), never by comparing the value against
    `select.failure_value`, for the reason that function gives; it is the
    same predicate `select.rule: map_elites_insert` gates the MAP-Elites
    archive with. A report whose fitness is `None`
    but whose training was clean is NOT caught here: it is unranked, not
    failed, and `_score` already sorts it below everything. The return value
    is the reason, for the journal -- a silent skip is indistinguishable from
    a candidate that was never a winner.
    """
    if measured_fitness(report):
        return ""
    c, res = report.candidate, report.result
    if not c.valid:
        return f"{c.failure_kind or 'invalid'}: {c.failure or 'no reason recorded'}"
    error = getattr(res, "error", "")
    if error:
        return f"training raised: {error}"
    if res is not None and not getattr(res, "trained", True):
        return f"training skipped: {getattr(res, 'skip_reason', '') or 'no reason recorded'}"
    if c.screened_out:
        return f"screened: {c.failure or 'no measurement from the screen'}"
    return str(report.meta.get("fitness_note") or "no fitness measurement")


def _candidate_order(reports: Sequence[CandidateReport]) -> List[CandidateReport]:
    """Candidate-INDEX order, undoing §5's fitness sort for an order-dependent
    gate. `meta["sample_index"]` is the slot an offspring was generated into
    (`sample_evolution` stamps it) and survives repair, where a repaired id
    `r0000` would sort after every `c####` sibling. The same key as
    `tree.py`'s `_in_slot_order`; a copy rather than an import, so two
    topologies do not couple through a private name."""
    def _slot(r: CandidateReport):
        idx = r.candidate.meta.get("sample_index")
        return (int(r.candidate.iteration),
                int(idx) if isinstance(idx, int) else 10 ** 9, r.cand_id)
    return sorted(reports, key=_slot)


@register("topology", "island_lineage")
def topo_island_lineage(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """REvolve: every offspring enters the deme it was drawn from, if it earns it.

    The difference from `topo_island` is one line of its source and the whole of
    the method. `topo_island` feeds ONE deme per round, chosen round-robin by
    iteration (`update.py:810`, `isl = int(state.iteration) % n_islands`), so a
    generation of 16 offspring lands in a single island and, at
    `n_islands: 13`, twelve demes never come into existence. Here each offspring
    carries its own deme on `candidate.meta["island"]` -- stamped by
    `sample_evolutionary_operators` from the cohort's island (paper Alg. 1 line
    11, p.4: `sample_in_context` returns `sampled_island_id`, and `main.py:171`
    uses that, not a lookup of the parent's stored island) or, at generation 0,
    from a per-individual draw. The artifact tell is that there is one
    `update_island` event PER TOUCHED DEME rather than one per round.

    An offspring with no stamp falls back to `update._island_for`, so the
    topology is usable under any `sampling_mode`; `_island_for` is deliberately
    not patched, because `topo_island` and `insert_into_archive` depend on its
    current behaviour and its docstring promises it consumes no `ctx.rng`.

    Three departures from the neighbouring topologies, each measured:

      * **Capacity is `update.archive.population_size` verbatim, NOT
        `_population_size(ctx)`.** That helper (`update.py:187-218`) returns
        `max(population_size, select.n_survivors, generate.n_parents, 1)` -- a
        FLOOR, never a cap -- and its argument for the floor ("individuals the
        other stages have already committed to keeping") is a claim about ONE
        population. This topology splits the round across up to `n_islands`
        demes, and REvolve needs `select.n_survivors: 16` so that all of
        `D_temp` reaches §6 while a deme holds 8 (`cfg/generate.yaml:14`,
        `max_island_size: 8`). Under the floor, `population_size: 8` with
        `n_survivors: 16` measures 16 -- every deme twice the size the resolved
        config states. `_maybe_migrate` is passed the same number for the same
        reason.
      * **`ctx.counters["archive_inserts"]` moves ONCE PER ITERATION**, as
        `topo_island:814` does and unlike `insert_into_archive:530-531`, which moves
        it once per insert. That is what keeps `update.archive.migration_interval`
        meaning ITERATIONS under this topology.
      * **The admission mean is recomputed after every admission**, so the order
        in which a deme's offspring are offered decides who gets in. That is the
        release (`rewards_database.py:117-172` recomputes
        `average_fitness_score` inside the loop and truncates to `max_size`
        inside it too), and batching it into one threshold would be a silent
        change of method, not a fix. **The order is CANDIDATE order** -- the
        slot each offspring was generated into (`main.py:237-245` passes the
        candidate arrays as built; `rewards_database.py:102-126` admits in
        zipped candidate/slot order) -- and NOT §5's. `select.rule:
        argmax_fitness` hands over `winners` sorted by fitness, and admitting in
        that order rejected offspring the release admits: incumbent 0, offspring
        slots [0.25, 1.0]; the release admits 0.25 against mean 0, then 1.0
        against 0.125, population [0, 0.25, 1]; fitness order admitted 1.0
        first and then rejected 0.25 against mean 0.5. §5 still names the
        winner; it does not order the gate.

    And one rule that is not a departure but the release's own: **a survivor
    with no MEASURED fitness is never filed**, at any generation and under any
    admission rule (`_unmeasured`). §5 hands
    over every numeric fitness, sentinel included; the topology, not the
    selector, is where the release's "nothing untrained reaches the database"
    has to be enforced, because `truncate` at generation 0 and an empty deme
    at every later one would otherwise admit -10000 and poison both wheels.
    The skip is journaled as `update_island_skipped`.

    Not gated by `_promotion_blocked`? It is -- unlike `archive_map_elites`. A
    lineage island IS a chain: the winner becomes the parent through
    `_dispatch_winner`, and the two regression guards (`update.rollback_if_worse`
    and §5's `require_improvement_over_incumbent`, published by nobody) mean the
    same thing here as under hill climbing.
    """
    cfg = ctx.cfg
    winner = selection.winner
    state.iteration_best = winner
    _maintain_memory(ctx, state, selection)
    if winner is None:
        return state

    _update_global_best(ctx, state, winner)
    blocked = _promotion_blocked(ctx, state, selection, winner)
    if blocked:
        log.info("      islands NOT updated -- %s", blocked)
        state.last_selection_notes = blocked
        ctx.event("update_rollback", cand_id=winner.cand_id, reason=blocked)
        return state

    # `evaluate.preferences.scope: cumulative` refits the Elo ranking over every
    # individual still held in state and writes the refit back onto the carried
    # reports (`preferences._refit_carried`). `ArchiveCell.fitness` is a float
    # BESIDE the report, written as `_score(report)` at insert
    # (`update.py:527`, `:683`) and never again, so without this pass the cells
    # keep their first-round numbers while their reports carry the current ones
    # -- and `_admit_above_island_mean`, `select.final_artifact: archive_best`
    # (`phases.py:1410`, `max(cells, key=lambda c: c.fitness)`) and the
    # migration ordering all read the float. Under the default
    # `within_iteration` this loop is an identity, and under `cumulative` it is
    # an identity too unless §4's cumulative branch ran: that branch is the
    # prerequisite, because `_resolve_deferred` refuses a second resolution
    # (`evaluation.py:913`, `or rep.fitness is not None`) and a report's fitness
    # is otherwise frozen at the round that produced it.
    if cfg.get("evaluate.preferences.scope") == "cumulative":
        for cell in state.archive.values():
            if cell.report is not None:
                cell.fitness = _score(cell.report)

    n_islands = max(int(cfg.get("update.archive.n_islands", 1) or 1), 1)
    capacity = max(int(cfg.get("update.archive.population_size", 1) or 1), 1)
    rule = _admission_rule(ctx, state)
    gate = _ADMISSION[rule]

    # Grouped in CANDIDATE order, and visited in island order: the within-deme
    # order is the method (see the docstring), the between-deme order must not
    # depend on dict churn.
    by_island: Dict[int, List[CandidateReport]] = {}
    for report in _candidate_order(selection.winners):
        stamped = report.candidate.meta.get("island")
        why = _unmeasured(report)
        if why:
            # Not filed, at ANY generation and under ANY admission rule -- see
            # `_unmeasured`. Journaled per report: the `select` event above
            # lists it among the winners, and without this line the reader
            # would have to infer from its absence in every `update_island`
            # event that it was dropped rather than mis-stamped.
            log.info("      island_lineage: %s not filed -- %s", report.cand_id, why)
            ctx.event("update_island_skipped", cand_id=report.cand_id, island=stamped,
                      reason=why, fitness=report.fitness)
            continue
        isl = int(stamped) if stamped is not None else _island_for(ctx, state, report)
        if not 0 <= isl < n_islands:
            # `n_islands` was lowered between the stamp and here, or a stamp
            # came from somewhere else. Fold rather than drop -- the ring in
            # `_maybe_migrate` is `% n_islands`, so an out-of-range deme would
            # never send or receive and would sit outside the topology while
            # every artifact counted it as part of it.
            log.warning("island_lineage: %s is stamped island %d but "
                        "update.archive.n_islands is %d; folding to %d",
                        report.cand_id, isl, n_islands, isl % n_islands)
            isl = isl % n_islands
        by_island.setdefault(isl, []).append(report)

    for isl in sorted(by_island):
        members = _population(state, isl)
        admitted: List[str] = []
        rejected: List[str] = []
        for report in by_island[isl]:
            if gate(ctx, members, report):
                # Truncated inside the loop, not after it, because the next
                # offspring is measured against what the deme holds NOW --
                # `rewards_database.py:164-172` truncates per individual for the
                # same reason. A gate-passer can therefore still be evicted by
                # the truncation; `members` in the event below is the truth,
                # `admitted` is who cleared the gate.
                members = _truncate(members + [report], capacity)
                admitted.append(report.cand_id)
            else:
                rejected.append(report.cand_id)
        _set_population(state, isl, members)
        ctx.event("update_island", island=isl, size=len(members),
                  members=[r.cand_id for r in members], admitted=admitted,
                  rejected=rejected, admission=rule, capacity=capacity)

    ctx.counters["archive_inserts"] = ctx.counters.get("archive_inserts", 0) + 1
    _maybe_migrate(ctx, state, capacity)

    _dispatch_winner(ctx, state, winner)
    _dispatch_operator(ctx, state, selection)
    return state


# Capability markers, read by `config._check_coherence` (`:1342`, `:1374`).
# Rather than the rules naming their readers in a literal tuple, which goes stale
# because no component author opens that file, the topology declares what it
# consults -- the same shape as an env factory's
# `supported_reductions`. Both are load-bearing for `configs/methods/revolve.yaml`:
# without them, `--validate-all` refuses `migration_rate>0` and
# `admission: above_island_mean` under this topology.
topo_island_lineage.has_demes = True
#: This topology files every MEASURED survivor into a deme ITSELF (`_unmeasured`
#: skips the rest), so a `winner_action` that also inserts would file the winner
#: twice. `_check_coherence` refuses that pairing rather than letting it read as a
#: richer archive than the run produced.
topo_island_lineage.files_survivors = True
topo_island_lineage.honours_admission = True
