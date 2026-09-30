"""Allowed values and types for every leaf key.

Split of responsibility (documented in README):
  * `configs/_default.yaml` -- the set of keys and their defaults. Readable;
    it is what you copy from when writing a new config.
  * this file -- the allowed values / types for those keys, plus the
    section each key belongs to (§0 for problem, loop and run settings,
    §1-§6 for the six stages).

`tests/test_schema_coverage.py` asserts the two agree in both directions, so
neither can drift.

Where a key's allowed values are exactly the members of a registry family, the
entry names the family (`kind=...`) instead of listing them. That makes the
guarantee in registry.py structural: you cannot add a config value without
registering an implementation, and you cannot register one the schema rejects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, Type

# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    section: str  # "§0" ... "§6"
    types: Tuple[Type, ...] = (object,)
    enum: Optional[Tuple[Any, ...]] = None
    kind: Optional[str] = None  # registry family supplying the enum
    item_enum: Optional[Tuple[Any, ...]] = None  # for list-valued keys
    item_kind: Optional[str] = None
    nullable: bool = True
    note: str = ""


F = Field
I, S, B, FL = int, str, bool, float
NUM = (int, float)

SCHEMA: dict[str, Field] = {
    # ---------------------------------------------------------------- §0 ---
    "name": F("§0", (S,), nullable=False),
    "extends": F(
        "§0", (S,),
        note="A meta key: popped during resolution, so it never appears in a resolved "
             "config"),
    "seed": F("§0", (I,)),
    "profile": F(
        "§0", (S,),
        note="The execution profile this run was loaded under: `--profile NAME` applies "
             "`configs/_profiles/NAME.yaml`. Recorded because it is part of the resolved "
             "config and therefore of `Config.hash()`; null means no profile layer was "
             "applied. Set by the operator at launch and never authored in a method "
             "config: a config that named its own profile would be claiming a layer the "
             "loader did not apply. Not a kind= key, because profiles are files, not a "
             "registry family"),
    "problem.env_id": F("§0", (S,), kind="env"),
    "problem.task_id": F(
        "§0", (S,),
        note="The task spec backing `problem.env_id`, named by id. With null the spec "
             "is derived from the env id, which is unambiguous today; name it when one "
             "environment backs several tasks. Not a kind= key: the catalogue is data, "
             "not a registry family of callables, so a missing id is caught in "
             "_check_coherence rather than by an enum that would need editing on every "
             "task added"),
    "problem.task_description": F(
        "§0", (S,),
        note="The task instruction l_task shown to the model. With null the run "
             "inherits the task spec's instruction: `description.l_task`, falling back "
             "to upstream's single `natural_language` field on specs that predate the "
             "split (`TaskSpec.instruction` owns that rule). A string overrides the "
             "spec. The override is the live path at the paper tier -- GT's instruction "
             "is deliberately vague and L2R's describes a robot dog -- so a task file "
             "may supply this value but must never force it"),
    "problem.horizon": F("§0", (I,),
                         note="Episode length in ENV steps, or null for the "
                              "environment's own (a spec's `env.horizon`). "
                              "TRUNCATION ONLY: an integer must be <= the env's own, "
                              "because shortening an episode is the trainer's choice "
                              "and lengthening one is a claim about the simulator. "
                              "RDA's HumanoidBench column pins 500 against "
                              "HumanoidBench's shipped 1000"),
    "problem.instruction_modality": F("§0", (S,), enum=("text", "text+image", "video")),
    "problem.search_space": F(
        "§0", (list,), item_enum=("reward", "observation", "dr"),
        note="LIMEN searches observation and reward jointly; DrEureka searches the "
             "reward first, then domain randomisation"),
    "problem.search_space_mode": F(
        "§0", (S,), enum=("joint", "staged"),
        note="A DESCRIPTIVE axis, read by no stage: staging is implemented by the `pre` "
             "phases (DrEureka's `dr_generation` searches the dr surface after the reward "
             "loop). `_check_coherence` refuses a value that disagrees with them -- "
             "`staged` with nothing that stages, `joint` beside `dr_generation` -- so the "
             "key cannot be ablated alone into a byte-identical run"),
    "problem.reward_representation": F(
        "§0", (S,),
        enum=("free_form_code", "weighted_components", "template_dsl", "tabular"),
        note="A DESCRIPTIVE axis, read by no stage: the form is dispatched by "
             "`generate.output.format` (free_form_code <-> component_dict_return | "
             "scalar_only; weighted_components <-> component_dict_plus_weights; "
             "template_dsl <-> template_params) and by `generate.generator_backend` "
             "(tabular <-> exhaustive_enumeration). `_check_coherence` refuses a pair that "
             "disagrees, naming both keys"),
    "problem.fitness_access": F(
        "§0", (S,), enum=("ground_truth_metric", "success_indicator", "none",
                          "demonstrations"),
        note="With none, evaluation reads no ground-truth signal: fitness comes from a "
             "VLM or human comparator in §4, or the method computes no fitness at all "
             "(CARD ranks nothing). `demonstrations` declares access to a SOLUTION "
             "POLICY's rollouts (policies/) and still no task metric: the verifier "
             "screen re-scores those rollouts with the candidate's own reward"),
    "llm.generator.provider": F("§0", (S,), kind="llm"),
    "llm.generator.model": F("§0", (S,)),
    "llm.generator.temperature": F("§0", NUM),
    "llm.generator.reasoning_effort": F("§0", (S,), enum=("low", "medium", "high", "none")),
    "llm.generator.modality": F("§0", (S,), enum=("text", "vlm")),
    "llm.generator.program": F(
        "§0", (S,),
        note="`llm.generator.provider: fixed` only: the reward program file the generator "
             "returns verbatim on every call (a control: one authored program through "
             "stages 2-6). Absolute, or relative to the checkout. Refused under any other "
             "provider. K candidates under it are the same reward under K learner seeds."),
    "llm.evaluator.provider": F("§0", (S,), kind="llm"),
    "llm.evaluator.model": F("§0", (S,)),
    "llm.evaluator.temperature": F("§0", NUM),
    "llm.evaluator.modality": F("§0", (S,), enum=("text", "vlm")),
    "llm.max_context_tokens": F("§0", (I,)),
    # Provider-side billing, not a design choice: a cached prefix and an uncached
    # one produce the same distribution over completions. Declared anyway because
    # it is honoured (bird/llm/anthropic_client.py::_cache_breakpoints) and because
    # a run's cost column is a first-class output.
    "llm.prompt_caching": F("§0", (B,)),
    "llm.max_concurrent_requests": F("§0", (I,)),
    "loop.n_iterations": F("§0", (I,), nullable=False),
    "loop.n_restarts": F("§0", (I,), nullable=False),
    "loop.termination": F("§0", (S,), kind="termination"),
    "loop.termination_cfg.patience": F("§0", (I,)),
    "loop.termination_cfg.min_delta": F("§0", NUM),
    "loop.termination_cfg.target_fitness": F("§0", NUM),
    "loop.waves": F(
        "§0", (list,), nullable=False,
        note="generate->verify->train->evaluate passes per iteration; [] is one pass "
             "(every method but R*). Each entry {llm, crossover, when}; `when: "
             "no_archive` runs the pass only in an iteration that began with fewer "
             "than two evaluated candidates. §5 and §6 run ONCE over the union, so an "
             "iteration keeps one winner. R* App. A p.12: \"we adopt a two-step "
             "iteration approach ... and finally aggregate all evaluation results to "
             "select the best individual\""),
    "loop.carry": F(
        "§0", (list,),
        item_enum=("best_reward", "dialogue", "preference_dataset", "trajectory_store",
                   "subtask_list", "replay_buffer", "policy_checkpoint",
                   "failure_memory", "archive", "curriculum", "search_tree",
                   "critic_population")),
    "loop.curriculum.enabled": F(
        "§0", (B,),
        note="An ORDERED progression of skills with a gate, not RDA's concurrent "
             "subtasks: the search works on one stage until something says it is done "
             "(unpublished; ours). Requires `curriculum` and `policy_checkpoint` in "
             "loop.carry -- a curriculum whose position is dropped restarts at stage 0 "
             "every iteration, and one that does not carry the policy retrains each "
             "stage from scratch, which is the mechanism"),
    "loop.curriculum.author": F("§0", (S,), kind="curriculum_author"),
    "loop.curriculum.author_max_rounds": F(
        "§0", (I,),
        note="Cap on the propose/review loop in `author: llm_reviewed`. Hitting it "
             "accepts the last proposal and records that it ended on the bound"),
    "loop.curriculum.gate": F("§0", (S,), kind="stage_gate"),
    "loop.curriculum.gate_votes": F(
        "§0", (I,),
        note="Judgments per gate decision. A handover is irreversible for the rest of "
             "the search and one VLM score is not evidence for one"),
    "loop.curriculum.gate_threshold": F(
        "§0", NUM,
        note="Mean normalised judge score at or above which a stage passes. Read on "
             "[0,1] whatever evaluate.feedback.score_scale is, as fitness_vlm_score "
             "normalises, so the number means one thing across scales"),
    "loop.curriculum.stage_patience": F(
        "§0", (I,),
        note="Iterations one stage may spend before on_stall fires. Null means derived "
             "-- loop.n_iterations // n_stages, floored at 1 -- because the useful "
             "value is a function of two other keys and a literal goes stale when "
             "either moves. Also the budget `gate: fixed_budget` advances on"),
    "loop.curriculum.on_stall": F("§0", (S,), kind="stall_action"),
    "loop.curriculum.regression_check": F(
        "§0", (B,),
        note="Re-score every already-passed stage each iteration and roll the policy "
             "back to that stage's checkpoint if it has been forgotten. Costs "
             "gate_votes judge calls per passed stage per iteration, counted in the "
             "budget like any other"),
    "loop.curriculum.regression_tolerance": F(
        "§0", NUM,
        note="How far below the score a stage PASSED with counts as forgetting it. "
             "Against the pass score, not the threshold: a stage that passed at 0.95 "
             "and now reads 0.75 has lost most of the skill while still clearing 0.7"),
    "loop.on_total_failure": F("§0", (S,), enum=("retry_iteration", "abort", "continue",
                                                  "continue_after_update")),
    "loop.max_iteration_retries": F("§0", (I,)),
    "loop.max_parallel_trainings": F("§0", (I, S)),
    "loop.resume_from": F(
        "§0", (S,), enum=("auto", "auto_degraded"),
        note="With null the run never resumes. The value is a sentinel, never a path: "
             "a fixed string resolves identically on every leg of a resumed run, so "
             "every leg lands in the same run dir and Config.hash() is untouched. A "
             "non-null value also enables checkpoint writing. `auto` refuses to resume "
             "a checkpoint that lost method-defining state; `auto_degraded` resumes "
             "anyway and counts the loss in budget.resume_degradations"),
    "budget.max_policy_trainings": F("§0", (I,)),
    "budget.max_llm_calls": F("§0", (I,)),
    "budget.max_gpu_hours": F("§0", NUM,
                              note="hours after which no new unit may START; "
                                   "not a hard total, since a training's duration "
                                   "is unknowable in advance and the last unit may "
                                   "overrun by its own length"),
    "pre": F("§0", (list,), item_kind="phase", note="stages that run before the loop"),
    "post": F("§0", (list,), item_kind="phase", note="stages that run after the loop"),
    # ---------------------------------------------------------------- §1 ---
    "generate.n_candidates": F("§1", (I,), nullable=False),
    "generate.candidate_schedule": F("§1", (S,), kind="candidate_schedule",
                                     nullable=False),
    "generate.candidate_schedule_values": F("§1", (list,), nullable=False),
    "generate.sampling_mode": F("§1", (S,), kind="sampling_mode"),
    "generate.sampling.chunk_size": F("§1", (I,)),
    "generate.parent_source": F("§1", (S,), kind="parent_source"),
    "generate.n_parents": F("§1", (I,)),
    "generate.crossover_rate": F(
        "§1", NUM, nullable=False,
        note="Per-offspring P(crossover) = 1 - p_m (REvolve §3.2, p.5). Read ONLY "
             "by generate.sampling_mode=evolutionary_operators, which draws one operator "
             "per offspring where update.operator picks one for the whole round; every "
             "other sampler ignores it, so the degenerate 0.0 changes nothing. The range "
             "is checked in _check_coherence rather than by an enum -- it is a "
             "probability, not a menu. nullable=False because the sampler compares it "
             "with < and null would raise TypeError at iteration 0"),
    # RF-Agent's search tree. Read by parent_source: uct_leaf | puct_leaf, sampling_mode:
    # tree_actions and update.topology: search_tree only; every default is the
    # degenerate one, so a config that leaves them alone runs no tree.
    "generate.tree.horizon_trainings": F(
        "§1", (I,),
        note="Alg. 1's N: denominator of the UCT lambda schedule and of the backup "
             "decay, with t = non-root tree nodes so far. A horizon, not a cap -- the "
             "release checks it only between rounds, so 80 realises 86 trainings "
             "(rfagent_algo.py:776-804). null is the no-tree default and is refused under "
             "parent_source uct_leaf/puct_leaf, which need the horizon"),
    "generate.tree.uct_lambda0": F(
        "§1", NUM, nullable=False,
        note="lambda_0 of Eq. 2: multiplies both the sqrt exploration term and the "
             "softmax(self-verify) term (rfagent_algo.py:70-89); c_puct under "
             "parent_source puct_leaf. RF-Agent 0.4"),
    "generate.tree.uct_lambda_final": F(
        "§1", NUM, nullable=False,
        note="lambda decays linearly from uct_lambda0 to this over horizon_trainings. "
             "† the paper's schedule reaches 0 (neurips_2025_arxiv.tex:1038); the code "
             "stops at 0.1 (rfagent_algo.py:117,778)"),
    "generate.tree.max_depth": F(
        "§1", (I,),
        note="uct_leaf/puct_leaf stop descending at this depth and expand the node they stand "
             "on, so children reach max_depth+1 as in the release (rfagent_algo.py:149). "
             "null = uncapped. ‡ paper silent; the release's cfg pins 16"),
    "generate.tree.group_size": F(
        "§1", (list,), nullable=False,
        note="[lo, hi], inclusive range of the TOTAL nodes a crossover_elite / "
             "different_thought prompt shows, parent included; k drawn uniformly per "
             "prompt. RF-Agent [2, 4] (neurips_2025_arxiv.tex:829; the release draws "
             "randint(2,4)-1 extras plus the parent, rfagent_algo.py:203-205,223)"),
    "generate.tree.elite_size": F(
        "§1", (I,), nullable=False,
        note="Top-N tree nodes by score, failures eligible, derived from the tree at "
             "read time, ties by insertion order. ‡ paper silent "
             "(neurips_2025_arxiv.tex:199); code 10 (rfagent_algo.py:128)"),
    "generate.tree.path_window": F(
        "§1", (I,), nullable=False,
        note="Nearest ancestors, parent included and oldest first, a path_reasoning "
             "prompt shows. † paper 'a k-length tree path', k in [2,4] "
             "(neurips_2025_arxiv.tex:201,829); code a fixed 4 "
             "(rfagent_algo.py:131,209-220)"),
    # -- R* (ICML 2025): module-level crossover, which costs no API call --
    "generate.crossover.n": F(
        "§1", (I,), nullable=False,
        note="How many of `generate.n_candidates` are built by PROGRAMMATIC crossover "
             "rather than asked of the LLM. R* n_c = 4 of a population of 16 (App. A, "
             "p.12; the split is stated as 12 LLM + 4 crossover on p.5), i.e. a 3:1 "
             "RATIO -- at any other population size the faithful value is n/4, not 4. "
             "0 disables, and every other config leaves it there"),
    "generate.crossover.operator": F(
        "§1", (S,), kind="crossover_operator",
        note="`module_insert` is R* Eq. 3: F_new = {M_i,1..M_i,m, M_j,m}, one module of "
             "one parent added to another, split by AST on the returned reward dict "
             "(App. A, p.12). Unlike REvolve's crossover and RF-Agent's crossover_elite "
             "it issues NO LLM call -- that is the mechanism, not an optimisation"),
    "generate.crossover.parent_selection": F(
        "§1", (S,), kind="crossover_parent_selection",
        note="`softmax_fitness`: \"we use a softmax function based on the success rates "
             "to select parent individuals\" (App. A, p.12)"),
    "generate.crossover.temperature": F(
        "§1", NUM, nullable=False,
        note="Softmax temperature over archive fitness. OURS (R* has no release, so no "
             "‡): R* states the softmax and not its temperature; 1.0 is the plain "
             "softmax the sentence describes"),
    "generate.crossover.backfill_failures": F(
        "§1", (B,),
        note="\"If the reward function generation by the LLM encounters runtime failures "
             "... the failed individuals are supplemented with those generated via "
             "crossover ... No extra API calls are required in this process\" (R* App. A, "
             "p.12). Counts candidates already invalid when §1 ends, which is a "
             "generation failure only -- a §2 screen has not run yet (see configs/methods/rstar.yaml)"),

    # -- R*: parameter alignment, the half that fits numbers to preferences --
    "generate.alignment.method": F(
        "§1", (S,), kind="param_alignment",
        note="`bradley_terry_segments` fits a candidate's own numeric parameters to "
             "labelled segment pairs under R* Eq. 1-2. Distinct from "
             "evaluate.preferences.aggregator=bradley_terry, which fits a RANKING OVER "
             "CANDIDATES: same model, different unknowns, different output"),
    "generate.alignment.labeller": F(
        "§1", (S,), kind="segment_labeller",
        note="`critic_population_vote` is R* §4.3: the LLM writes rule-based critic "
             "PROGRAMS once, they label every step of every pair for free, and a vote "
             "over the population turns steps into segments"),
    "generate.alignment.critics": F(
        "§1", (I,), nullable=False,
        note="Size of the critic population. R* c = 5 (App. A, p.12). These are LLM "
             "calls and are charged to the budget's aggregate llm_calls (not separated) "
             "-- a cost comparison against Eureka that counts only reward generation "
             "understates R* by at least this many"),
    "generate.alignment.vote_ladder": F(
        "§1", (list,), nullable=False,
        note="Agreement thresholds tried in order, strictest first, until "
             "target_segments are collected. R* [5, 4, 3] over 5 critics (App. A, p.12). "
             "†p §4.3 states only the floor (\"at least half\"); the appendix states the "
             "ladder, and the ladder is the specific claim"),
    "generate.alignment.min_segment_len": F(
        "§1", (I,), nullable=False,
        note="Shortest run of one label that becomes a segment, INCLUSIVE. R* App. A "
             "\"a length of at least 5\" -> 5. †p §4.3 says \"greater than 5\"; the two "
             "differ by one step and nothing downstream would report which ran"),
    "generate.alignment.target_segments": F(
        "§1", (I,), nullable=False,
        note="How many labelled segments to collect before the ladder stops relaxing. "
             "R* 20 (App. A, p.12)"),
    "generate.alignment.max_pairs": F(
        "§1", (I,), nullable=False,
        note="Cap on trajectory pairs offered to the critics; 0 = every pair in the "
             "store. OURS (no ‡): R* says only \"Sample trajectories\" (Alg. 1 line 10) "
             "and never "
             "how many"),
    "generate.alignment.iterations": F(
        "§1", (I,), nullable=False,
        note="Optimisation steps on the preference loss. R* 1000 (App. A, p.12)"),
    "generate.alignment.train_fraction": F(
        "§1", NUM, nullable=False,
        note="Train share of the labelled segments; the rest is validation and selects "
             "the answer. R* 0.7 (App. A, p.12)"),
    "generate.alignment.learning_rate": F(
        "§1", NUM, nullable=False,
        note="OURS (no ‡): R* names no optimiser and no step size, only \"1000 iterations of "
             "optimization\". The component uses SPSA (two objective evaluations per "
             "step at any parameter count); this is its base step"),
    "generate.alignment.tunable": F(
        "§1", (S,), enum=("none", "float_literals"),
        note="Which numbers in a reward program are psi. OURS (no ‡): R* says \"the parameters "
             "within the reward module and the weights between modules\" (§2, p.3) and "
             "never how one is recognised. `float_literals` is OURS: every float literal "
             "is a parameter, every int is not, which separates 2.0*dist from obs[3] "
             "with no per-task configuration"),
    "generate.actions": F(
        "§1", (dict,), nullable=False,
        note="tree_actions only: per-expansion counts over mutation_mechanism, "
             "mutation_param, crossover_elite, path_reasoning, different_thought, "
             "summing to n_candidates. RF-Agent {2,2,2,1,1} "
             "(neurips_2025_arxiv.tex:829; rfagent_algo.py:123). Must be {} under any "
             "other sampling_mode"),
    "generate.generator_backend": F("§1", (S,), kind="generator_backend"),
    "generate.search_grid": F("§1", (dict,), note="exhaustive_enumeration backend only"),
    "generate.context.env_spec": F("§1", (S,), kind="env_spec"),
    "generate.context.strip_existing_reward": F("§1", (B,)),
    "generate.context.include_safety_instruction": F("§1", (B,)),
    "generate.context.safety_instruction": F("§1", (S,)),
    "generate.context.include_task_decomposition": F("§1", (B,)),
    "generate.context.include_curriculum_stage": F(
        "§1", (B,),
        note="Put the CURRENT curriculum stage in the prompt, so the generator writes a "
             "reward for one stage instead of the whole task -- which is the axis's "
             "whole hypothesis. Requires loop.curriculum.enabled; on its own the "
             "section would render nothing"),
    "generate.context.include_reward_recipe_hints": F(
        "§1", (B, S), enum=(False, True, "t2r_metaworld", "t2r_panda"),
        note="false renders nothing. true renders BIRD's own environment-neutral "
             "four-part paraphrase (generation._RECIPE_HINTS; ours, unpublished -- the "
             "only recipe with a referent on a non-manipulator env). t2r_metaworld "
             "renders the three-item Meta-World template T2R and CARD both send "
             "(MetaworldPrompt.py:8-12 = CARD metaworld_prompt.py:6-10 = "
             "card/main.tex:728-732). t2r_panda renders the five-item ManiSkill2/Panda "
             "template both send (PandaPrompt.py:9-15 = CARD panda_prompt.py:9-15); "
             "dagger: T2R's appendix.tex:189-195 drops the release's hedge and 'the'. "
             "The value is keyed by benchmark, not method, and nothing ties it to "
             "problem.env_id -- the operator chooses. Eureka explicitly excludes "
             "reward templates (intro.tex:41)"),
    "generate.context.include_reward_engineering_tips": F(
        "§1", (B,),
        note="Eureka's instructional blocks: code_feedback's analyze-the-policy-feedback "
             "procedure (appended after the numeric reflection) and code_output_tip's "
             "transformation/temperature tips (appended to the system message and to "
             "every output contract, error path included). True also drops the 'code, "
             "not commentary' sentence from the default system message -- it works "
             "against the analyze-first behaviour the tips elicit "
             "(refs/code/Eureka/eureka/eureka.py:57,268-269,276)"),
    "generate.context.include_parent_code": F("§1", (B,)),
    "generate.context.include_parent_thought": F(
        "§1", (B,),
        note="Renders `Design Idea: <thought>` above each parent's code in PARENT "
             "REWARD: report.meta['design_thought'] when evaluate.thought_alignment "
             "ran, else the candidate's nl_spec. RF-Agent true (base_thought_code.txt; "
             "rfagent_algo.py:261-338)"),
    "generate.context.include_numeric_reflection": F("§1", (B,)),
    "generate.context.include_behavioural_analysis": F("§1", (B,)),
    "generate.context.include_human_feedback": F("§1", (B,)),
    "generate.context.include_failure_traces": F("§1", (B,)),
    "generate.context.include_archive_elites": F("§1", (B,)),
    "generate.context.shuffle_sections": F("§1", (B,)),
    "generate.context.system_prompt": F(
        "§1", (S,),
        note="Non-null replaces the generator's system message verbatim (nothing is "
             "appended, including the reward-engineering tips). Refused under "
             "sampling_mode: personas. Published: RF-Agent, LIMEN and CARD pin their "
             "own system prompts through it"),
    "generate.context.guidance": F(
        "§1", (S,),
        note="Non-null renders as a `## GUIDANCE` section before EXAMPLES. Ours; no "
             "paper pin"),
    "generate.context.reflection_guidance": F(
        "§1", (S,),
        note="Non-null replaces Eureka's code_feedback analysis block where that would "
             "render, and is otherwise appended after whichever feedback section "
             "renders (NUMERIC REFLECTION or BEHAVIOURAL ANALYSIS) when parents exist. "
             "Ours; no paper pin"),
    "generate.history_mode": F("§1", (S,), kind="history_mode"),
    "generate.stage_history_modes": F(
        "§1", (list,), item_kind="history_mode",
        note="Overrides history_mode per LLM call, in call order; an empty list "
             "applies history_mode to every call. L2R keeps the dialogue for the "
             "Thinker and drops it for the Coder, which one global history_mode "
             "cannot express (conversation.py: keep_message_history=[True, False])"),
    "generate.history_max_turns": F("§1", (I,)),
    "generate.history_max_tokens": F("§1", (I,)),
    "generate.fewshot.enabled": F("§1", (B,)),
    "generate.fewshot.k": F("§1", (I,)),
    "generate.fewshot.retriever": F("§1", (S,), enum=("semantic_similarity", "random", "fixed")),
    "generate.fewshot.corpus": F("§1", (S,)),
    "generate.decomposition.enabled": F("§1", (B,)),
    "generate.decomposition.n_subtasks": F("§1", (I, S), enum=None),
    "generate.decomposition.reward_must_map_to_subtasks": F("§1", (B,)),
    "generate.decomposition.guidance": F(
        "§1", (S,),
        note="Non-null is appended to the subtask-generation prompt as extra "
             "instruction. Ours; no paper pin"),
    # §1 because it changes what a method can EXPRESS, not how fast it runs:
    # a jitted reward has no Python control flow on values. An enum rather
    # than a registry family -- there is no component to
    # dispatch, the value selects a prompt clause, an exec namespace and a
    # verifier path, all inside `generate`/`verify`.
    # `torch`: the language Eureka's Isaac Gym prompt asks for verbatim
    # (`@torch.jit.script def compute_reward(...)`, reward_signature.txt).
    # Accepted on the fasttd3 and sb3 backends only (`_check_coherence`).
    "generate.reward_language": F("§1", (S,), enum=("numpy", "jax", "torch")),
    "generate.output.format": F("§1", (S,), kind="output_format"),
    "generate.output.edit_mode": F("§1", (S,), enum=("full_rewrite", "diff", "weights_only")),
    "generate.output.signature": F("§1", (S,), enum=(
        "compute_reward_state_action_next", "t2r_compute_dense_reward",
        "card_compute_dense_reward")),
    "generate.output.two_stage_nl_then_code": F("§1", (B,)),
    "generate.output.design_thought": F(
        "§1", (S,), enum=("none", "inline_brace"), nullable=False,
        note="inline_brace asks for a one-sentence design idea inside {braces} before "
             "the code, in the SAME LLM turn, and parses the first {...} into "
             "Candidate.nl_spec. Refused with two_stage_nl_then_code (both write "
             "nl_spec). RF-Agent inline_brace (thought_code_output.txt:1-2; "
             "rfagent_algo.py:426-433)"),
    "generate.output.forbid_helper_functions": F("§1", (B,)),
    "generate.parse.patterns": F("§1", (list,)),
    "generate.parse.max_retries": F("§1", (I, S), note="'inf' allowed"),
    "generate.postprocess.symbol_mapping": F(
        "§1", (dict, S),
        note="T2R's general->specific converter (`RewardFunctionConverter."
             "general_to_specific`), applied longest-key-first on word boundaries; CARD "
             "inherits it. A dict is the table itself; a STRING names a member of the "
             "`symbol_table` registry family -- `per_task` reads the task spec's own "
             "`symbol_mapping:` block (T2R's ManiSkill2 shape), `t2r_metaworld_global` is "
             "T2R's single Meta-World table (`metaworld_exp.py:19-27`) with BIRD `s[...]` "
             "right-hand sides, and is the companion `env_spec: t2r_class_abstraction` "
             "requires. NOT a kind= field, because a dict is also legal: an unregistered "
             "name is refused by _check_coherence instead of by an enum"),
    "generate.co_design.observation_fn": F("§1", (B,)),
    "generate.co_design.observation_max_dim": F("§1", (I,)),
    "generate.co_design.dr_config": F("§1", (B,)),
    "generate.co_design.dr_prior": F("§1", (S,), enum=("none", "default_sim_ranges", "rapp")),
    "generate.co_design.dr_n_configs": F("§1", (I,)),
    # ---------------------------------------------------------------- §2 ---
    "verify.enabled": F("§2", (B,)),
    "verify.static_checks": F("§2", (list,), item_kind="static_check"),
    "verify.forbidden_symbols": F("§2", (list,)),
    "verify.dynamic_checks": F("§2", (list,), item_kind="dynamic_check"),
    "verify.smoke_test_steps": F("§2", (I,)),
    "verify.timeout_s": F("§2", (I,)),
    "verify.check_order": F("§2", (S,), kind="check_order",
                            note="LIMEN short-circuits cheap-static -> import -> execute "
                                 "with a per-stage timeout; a flat list cannot order that"),
    "verify.stage_timeouts_s": F("§2", (dict,), note="per-check override of timeout_s"),
    "verify.human_confirm_before_execute": F(
        "§2", (B,),
        note="Puts a human approval step in the validity gate: generated code must "
             "be confirmed before it executes (L2R confirmation_safe_executor.py). "
             "Distinct from select.human_override, which acts after training"),
    "verify.on_failure": F("§2", (S,), kind="verify_failure"),
    "verify.max_repair_attempts": F("§2", (I,),
                                    note="0 = uncapped (100-resample backstop, "
                                         "verification.UNCAPPED_SAFETY_LIMIT = 100, "
                                         "then verify.on_exhaustion)"),
    "verify.on_exhaustion": F("§2", (S,), enum=("record_and_continue", "abort_run",
                                                 "teach_next_iteration")),
    "verify.repairs_count_against_budget": F("§2", (B,)),
    "verify.dedup.method": F("§2", (S,), kind="dedup"),
    "verify.dedup.threshold": F("§2", NUM),
    "verify.quality_screen": F("§2", (S,), kind="screen"),
    "verify.alignment_filter.enabled": F("§2", (B,)),
    "verify.alignment_filter.metric": F("§2", (S,), enum=("trajectory_alignment_coefficient",)),
    "verify.alignment_filter.preference_dataset": F("§2", (S,)),
    "verify.alignment_filter.keep_top_n": F("§2", (I,)),
    "verify.alignment_filter.subtraj_len": F("§2", (I,)),
    "verify.cascade.enabled": F("§2", (B,)),
    "verify.cascade.short_budget_steps": F("§2", (I,)),
    "verify.cascade.short_budget_fraction": F(
        "§2", NUM,
        note="null = use short_budget_steps verbatim; a float in (0,1] sizes "
             "the screen as a fraction of train.env_steps (LIMEN pins 0.6/0.5/0.5)"),
    "verify.cascade.min_success_threshold": F("§2", NUM),
    # The demonstration verifier (`verify.quality_screen: demo_margin`): graded
    # policies from `policies/` rolled out under each candidate's reward, before
    # any training. Ours; no paper pin. Refused unless
    # `problem.fitness_access: demonstrations`.
    "verify.demo_screen.policies": F("§2", (list,)),
    "verify.demo_screen.episodes": F("§2", (I,), nullable=False),
    "verify.demo_screen.keep": F("§2", (S,), enum=("all", "fraction", "top_k", "adaptive",
                                                   "skip_all"),
                                 nullable=False),
    "verify.demo_screen.keep_fraction": F("§2", NUM, nullable=False),
    "verify.demo_screen.keep_top_k": F("§2", (I,), nullable=False),
    "verify.demo_screen.keep_cfg.cap": F("§2", (I,), nullable=False),
    "verify.demo_screen.keep_cfg.floor": F("§2", (I,), nullable=False),
    "verify.demo_screen.keep_cfg.patience": F("§2", (I,), nullable=False),
    "verify.demo_screen.keep_cfg.decay": F("§2", NUM, nullable=False),
    "verify.demo_screen.keep_cfg.min_spread": F("§2", NUM, nullable=False),
    "verify.demo_screen.reject_nonpositive_margin": F("§2", (B,)),
    "verify.demo_screen.max_reward_error_rate": F("§2", NUM, nullable=False),
    "verify.complexity_cap.max_components": F(
        "§2", (I,), nullable=False,
        note="quality_screen: complexity_cap -- a candidate returning more reward-dict "
             "entries than this is screened, never trained; 0 = no bound. Ours, no paper pin"),
    "verify.complexity_cap.max_ast_nodes": F(
        "§2", (I,), nullable=False,
        note="quality_screen: complexity_cap -- a candidate whose reward FunctionDef has more "
             "AST nodes than this (LIMEN's reward_ast_node_count) is screened; 0 = no bound. "
             "Ours, no paper pin"),
    "verify.tpe.enabled": F("§2", (B,)),
    "verify.tpe.rule": F(
        "§2", (S,), enum=("pair_accuracy", "frac_above_max_failure", "min_over_max"),
        note="three published readings of 'order-preserving'"),
    "verify.tpe.threshold": F("§2", NUM),
    "verify.tpe.discount": F("§2", NUM,
                             note="null scores stored trajectories undiscounted. "
                                  "† paper: Def. 4.1 Eq. (1) writes a γ^t discount; "
                                  "code: undiscounted (utils.py:277); the config "
                                  "follows the code"),
    "verify.tpe.trajectories_per_iteration": F("§2", (I,)),
    "verify.tpe.store": F("§2", (S,), enum=("append_on_pass", "append_always")),
    "verify.tpe.on_failure": F("§2", (S,), enum=("skip_training", "train_anyway"),
                               note="The fork CARD's cost claim rests on. † paper: "
                                    "skips RL training on a TPE failure (§4.2.3, "
                                    "Alg. 1 l.9-10); code: trains every iteration "
                                    "(metaworld_exp_one_step.py:374-381)"),
    "verify.llm_self_critique.enabled": F("§2", (B,)),
    # ---------------------------------------------------------------- §3 ---
    "train.backend": F("§3", (S,), kind="train_backend",
                       note="Selects the learner implementation that executes training "
                            "in this repo. `train.algorithm` records the algorithm the "
                            "method's paper used and is kept as a citation even when "
                            "no backend here can run it"),
    "train.algorithm": F("§3", (S,),
                         enum=("ppo", "sac", "qr_sac", "td3", "q_learning", "none",
                               "fasttd3"),
                         note="fasttd3 (Seo et al., 2025) is runnable via "
                              "train.backend: fasttd3 and cannot run on sb3"),
    "train.architecture": F("§3", (S,)),
    "train.hyperparameters": F("§3", (dict,)),
    "train.reward_source": F(
        "§3", (S,), kind="reward_source",
        note="WHICH reward the learner trains on. `candidate` (default) is the program "
             "the generator wrote, which is every published METHOD. `reference` is the "
             "environment's own shipped reward -- the ORACLE baseline both Text2Reward "
             "(Fig. 2, experiments.tex:55 'the expert-written reward function provided "
             "by the environment') and CARD (Fig. 3) plot their searched curves against. "
             "It is not a program: see training.py's reward_source block for why a "
             "transcription cannot be faithful on Meta-World (tcp_center, init_tcp and "
             "_gripper_caging_reward are simulator state a reward program cannot reach). "
             "_check_coherence requires an adapter with has_reference_reward and "
             "generate.n_candidates 1 -- an oracle population is one program"),
    "train.hyperparameter_search": F(
        "§3", (S,), kind="hyperparameter_search",
        note="Singh 2009 scores each reward under its own best (alpha, epsilon), so "
             "fitness is a max over a learner grid; every LLM-era method holds "
             "hyperparameters fixed and is 'none'"),
    "train.hyperparameter_grid": F("§3", (dict,), note="{param: [values]}, per candidate"),
    "train.env_steps": F("§3", (I,)),
    "train.n_parallel_envs": F("§3", (I,)),
    "train.seeds_per_candidate": F("§3", (I,), nullable=False),
    "train.candidate_parallelism": F("§3", (S,), kind="candidate_parallelism"),
    # LaRes (NeurIPS 2025). `independent` is a tail call to candidate_parallelism
    # and is what every other config inherits; `shared_population` pools the
    # round's budget and divides it online. Same total either way.
    "train.interaction": F("§3", (S,), kind="interaction",
                           note="independent = every method before LaRes; "
                                "shared_population trains one agent per member "
                                "(seeds_per_candidate must be 1, select.allocation "
                                "uniform) and its waves go through "
                                "candidate_parallelism unchanged"),
    "train.interaction_allocator": F("§3", (S,), kind="interaction_allocator",
                                     note="thompson_success reads the env success flag "
                                          "-- a ground-truth channel, refused unless "
                                          "problem.fitness_access declares ground truth "
                                          "(ground_truth_metric | success_indicator) and "
                                          "under evaluate.fitness.source: none"),
    "train.interaction_cfg.slices_per_candidate": F("§3", (I,), nullable=False,
                                                    note="1 = a single allocation pass"),
    "train.interaction_cfg.moment_samples": F("§3", (I,), nullable=False,
                                              note="buffer sample for Eq. 3's moments"),
    "train.interaction_cfg.window": F("§3", (I,), nullable=False,
                                      note="episodes per arm in the Beta posterior; "
                                           "LaRes sets it per task and the paper never "
                                           "names the knob"),
    "train.interaction_cfg.prior_alpha": F("§3", NUM, nullable=False),
    "train.interaction_cfg.prior_beta": F("§3", NUM, nullable=False),
    "train.interaction_cfg.skip_duplicates": F("§3", (B,), nullable=False,
                                               note="App. B: a re-sampled member is "
                                                    "skipped, not re-run; false = a "
                                                    "member drawn k times in one wave "
                                                    "trains k consecutive slices, each "
                                                    "resuming the previous, never two "
                                                    "from one resume point"),
    "train.interaction_cfg.shared_buffer": F("§3", (B,), nullable=False,
                                             note="true: one raw-transition ring per round, "
                                                  "pre-filled into each arm's replay buffer "
                                                  "relabelled under its own reward (LaRes "
                                                  "§4.3); false: per-arm continuity only. "
                                                  "Off-policy sb3 learners only"),
    "train.reward_scaling": F("§3", (S,), kind="reward_scaling",
                              note="LaRes Eq. 3, align to the elite's mean/std"),
    "train.elite_constraint.kind": F("§3", (S,), kind="elite_constraint",
                                     note="LaRes Eq. 4, L2 to the elite's parameters"),
    "train.elite_constraint.weight": F("§3", NUM, nullable=False),
    "train.elite_constraint.apply_to": F("§3", (S,), enum=("non_elite", "all"),
                                         nullable=False),
    "train.init": F("§3", (S,),
                    enum=("from_scratch", "warm_start_from_best", "warm_start_from_parent",
                          "warm_start_from_similar", "secondary_replay_buffer",
                          "bc_prior", "bc_prior_then_warm_start", "fused_warm_start")),
    "train.on_runtime_error": F(
        "§3", (S,), enum=("record", "repair"),
        note="What happens when a reward passes verification's smoke/shape checks but "
             "RAISES on a state reached later in policy training. `record` (default) files "
             "the crash on the TrainResult and lets §4/§6 see it (a crashed reward can then "
             "be selected against). `repair` is RF-Agent (rfagent_algo.py:561-563, :661-670, "
             "neurips_2025 §205): the training traceback is sent back to the generator inside "
             "the same iteration, the candidate is re-generated and re-trained on a FRESH seed "
             "stream, and the repaired result (not the crash) reaches the tree. Bounded by "
             "verify.max_repair_attempts."),
    "train.secondary_buffer.ratio": F("§3", NUM),
    "train.secondary_buffer.size": F("§3", (I,)),
    "train.fusion.ratio_search": F("§3", (S,), kind="fusion_ratio_search",
                                   note="ROSKA: how the parameter-fusion ratio alpha is "
                                        "chosen. `fixed` reads train.fusion.alpha."),
    "train.fusion.alpha": F("§3", NUM,
                            note="theta_f = alpha*theta_best_prev + (1-alpha)*theta_0. "
                                 "1.0 == warm_start_from_best, 0.0 == from_scratch."),
    "train.fusion.sc_bo.init_points": F("§3", (list,), nullable=False),
    "train.fusion.sc_bo.n_evaluations": F("§3", (I,), note="J, including the init points"),
    "train.fusion.sc_bo.probe_fraction": F(
        "§3", NUM, note="T_BO as a fraction of train.env_steps. The paper's 200 is EPOCHS "
                        "(appendix.tex:98: an epoch is a fixed number of transitions), so "
                        "200/3000 is a sample-size ratio, not 200 env steps. 0 disables."),
    "train.fusion.sc_bo.post_probe_fraction": F(
        "§3", NUM, note="ROSKA's +300/3000 after the search, per candidate."),
    "train.fusion.first_round_fraction": F(
        "§3", NUM, note="ROSKA's round-1 candidate budget as a fraction of train.env_steps "
                        "(500/3000, appendix.tex:7). Round 1 has no policy to fuse with, so "
                        "it trains short rather than fully. 0 = off."),
    "train.winner_extension_fraction": F(
        "§3", NUM, note="ROSKA's +2500/3000 for the round's winner after selection, as a "
                        "fraction of train.env_steps. 0 = off."),
    "train.bc_prior.policy": F("§3", (S,),
                               note="`auto` = the unique runnable `policies/` entry for "
                                    "problem.env_id, else a registry id. Unpublished."),
    "train.bc_prior.n_demos": F("§3", (I,)),
    "train.bc_prior.epochs": F("§3", (I,)),
    "train.bc_prior.success_margin": F("§3", (I,),
                                       note="steps kept after a demonstration's first success"),
    "train.bc_prior.accept": F("§3", (S,), enum=("success", "any", "metric_floor", "success_else_top",
                                                  "success_else_best"), nullable=False,
                               note="which episodes of the scripted policy are demonstrations: "
                                    "`success` truncates at the first success and drops failures "
                                    "(whole episodes on an env with no success bar); `any` keeps "
                                    "every episode whole; `metric_floor` keeps those whose "
                                    "task_metric >= accept_min_metric. Unpublished."),
    "train.bc_prior.fallback_oversample": F("§3", (I,),
                                            note="`accept: success_else_top` only: teacher episodes rolled "
                                                 "when none of n_demos succeeded, as a multiple of n_demos; "
                                                 "required under that rule, must be null under any other"),
    "train.bc_prior.fallback_keep": F("§3", NUM,
                                      note="`accept: success_else_top` only: fraction of that pool kept, "
                                           "best task_metric first; required under that rule, must be "
                                           "null under any other"),
    "train.bc_prior.fallback_min_metric": F("§3", NUM,
                                            note="`accept: success_else_best` only: a teacher episode counts "
                                                 "when its task_metric exceeds this; required under that "
                                                 "rule, must be null under any other"),
    "train.bc_prior.fallback_max_episodes": F("§3", (I,),
                                              note="`accept: success_else_best` only: the sampling cap, as a "
                                                   "multiple of n_demos; required under that rule, must be "
                                                   "null under any other"),
    "train.bc_prior.clone_eval_episodes": F("§3", (I,),
                                            note="greedy episodes of the finished clone scored by task_metric "
                                                 "and success, recorded in bc_prior.json only; 0 = off"),
    "train.bc_prior.accept_min_metric": F("§3", NUM,
                                          note="the floor for accept: metric_floor, in the metric's "
                                               "own units; must be null under any other rule"),
    "train.anchor.kind": F("§3", (S,), enum=("none", "kl_clone", "kl_reward"),
                           note="keep-close term toward the candidate's initial policy during "
                                "RL. `kl_clone` adds it to PPO's loss (the measured form; PPO "
                                "only); `kl_reward` subtracts it from the reward (the RLHF "
                                "form; any algorithm). Unpublished."),
    "train.anchor.schedule": F("§3", (S,), enum=("fixed", "adaptive")),
    "train.anchor.beta": F("§3", NUM),
    "train.anchor.target": F("§3", NUM, note="per-state KL budget the adaptive schedule tracks"),
    "train.anchor.warmup_steps": F("§3", (I,), note="actor frozen while the critic warms up"),
    "train.anchor.target_kl": F("§3", NUM, note="PPO's per-update early stop"),
    "train.anchor.log_std_init": F("§3", NUM,   # nullable, like every field not marked otherwise
                                   note="overrides the start policy's log-std; null keeps it"),
    "train.anchor.beta_min": F("§3", NUM, note="floor under the adaptive controller's beta"),
    "train.anchor.until_iteration": F("§3", (I,),   # nullable: null anchors every iteration
                                      note="anchor only loop iterations below this index; "
                                           "later ones start from the clone but train free"),
    # Inner-loop early stopping is ONE mechanism: `pruning` is the RULE (when a
    # training has stopped paying), `pruning_metric` is WHAT it watches. A rule
    # reading env.task_metric reads ground truth, and would hand it to a method
    # that declares none (a `median_stop` override on RDA, say), so the default
    # metric is the candidate's own reward -- legal for every method;
    # `task_metric` is refused under `problem.fitness_access: none`.
    "train.pruning": F("§3", (S,),
                       enum=("none", "successive_halving", "hyperband", "median_stop",
                             "plateau", "successive_halving_pool")),
    "train.pruning_metric": F("§3", (S,), enum=("own_reward", "task_metric", "demo_fraction"),
                              nullable=False),
    "train.pruning_cfg.patience": F("§3", (I,), nullable=False),
    "train.pruning_cfg.min_delta": F("§3", NUM, nullable=False),
    "train.pruning_cfg.min_checkpoints": F("§3", (I,), nullable=False),
    # The demonstration ceiling (not published): the expert's and the
    # random policy's return under each candidate's OWN reward, from the demo
    # screen's rollouts. `ceiling: demo_return` lets the rule above stop a
    # training only once it has closed `ceiling_fraction` of that gap; the
    # `demo_fraction` metric is the same quantity as a curve, comparable across
    # rewards, which `successive_halving_pool` ranks a pool on in rungs of
    # `rung_fraction` x env_steps, keeping 1/`eta`. All demonstration access.
    "train.pruning_cfg.ceiling": F("§3", (S,), enum=("none", "demo_return"), nullable=False),
    "train.pruning_cfg.ceiling_fraction": F("§3", NUM, nullable=False),
    "train.pruning_cfg.rung_fraction": F("§3", NUM, nullable=False),
    "train.pruning_cfg.eta": F("§3", (I,), nullable=False),
    "train.reward_norm": F("§3", (S,), enum=("none", "running_std", "clip")),
    # A KL-style pull toward the task's demonstration policy during TRAINING only
    # (Gaussian KL to a deterministic reference = ||a - a_ref||^2 / 2 sigma^2,
    # weighted by beta). Reads the same solution policy the demo_margin screen
    # rolls out; requires `problem.fitness_access: demonstrations`. Ours.
    "train.reference_policy.enabled": F("§3", (B,)),
    "train.reference_policy.beta": F("§3", NUM, nullable=False),
    "train.reference_policy.sigma": F("§3", NUM, nullable=False),
    # Which checkpoint is SHIPPED once training ends (unpublished).
    # `best_by_reward` restores the checkpoint whose greedy evaluation paid the
    # candidate's own reward the most, so the policy §4 scores, the judge sees
    # and `policy_ref` carries is that one rather than the last. The task metric
    # is deliberately not an option: selecting on it leaks ground truth into
    # training. `_cfg.min_delta` is the noise guard, a fraction of the curve's
    # own range like `train.pruning_cfg.min_delta`.
    "train.checkpoint_selection": F("§3", (S,), enum=("final", "best_by_reward", "best_own_return"),
                                    nullable=False),
    "train.checkpoint_selection_cfg.min_delta": F("§3", NUM, nullable=False),
    "train.checkpoint_interval": F("§3", (I,)),
    # NUM, not (I,), and for the same reason `output.video.timeout_s` is NUM: it
    # is a DURATION. An int-only wall clock cannot say "half a second", which is
    # the granularity a cascade screen (`verify.cascade.short_budget_steps`) or a
    # test needs, and `float(... or 3600)` already turns the only int that could
    # have expressed it -- 0 -- back into the default.
    "train.timeout_s": F("§3", NUM),
    "train.failure_detection": F("§3", (S,), enum=("exception", "stdout_grep")),
    "train.log": F("§3", (list,),
                   item_enum=("fitness", "reward_component_values", "episode_stats",
                              "gradient_stats", "video", "gt_reward",
                              "epoch_scalars")),
    "train.domain_randomization.mode": F("§3", (S,), enum=("none", "fixed_human", "generated")),
    "train.domain_randomization.params": F("§3", (dict,)),
    "final_retrain.enabled": F("§3", (B,)),
    "final_retrain.n_seeds": F("§3", (I,)),
    "final_retrain.env_steps": F("§3", (I,)),
    # The profile-owned execution cap on the line above: min(env_steps, cap) is
    # what trains; both are recorded. Null = no cap (the published point); an
    # int; or the sentinel "same_as_train" = the launch-resolved train.env_steps
    # (every profile's value, so `--set train.env_steps=N` moves the cap too).
    "final_retrain.max_env_steps": F("§3", (I, S), note="'same_as_train' allowed"),
    # A reporting-protocol block, like final_retrain: it measures the run, it
    # does not steer it. Kept apart from `evaluate.vlm.*` because the paper
    # pins different numbers for the two evaluators (RDA §5.1).
    "alignment_rate.enabled": F("§4", (B,)),
    "alignment_rate.n_videos": F("§4", (I,)),
    "alignment_rate.repeats": F("§4", (I,)),
    "alignment_rate.images_per_query": F("§4", (I,)),
    "alignment_rate.scale": F("§4", (S,), enum=("[0,1]", "likert", "binary")),
    "alignment_rate.normalisation": F("§4", (S,),
                                      enum=("divide_by_max", "min_max", "none")),
    # ---------------------------------------------------------------- §4 ---
    "evaluate.rollouts_per_candidate": F("§4", (I,)),
    "evaluate.checkpoint_eval_episodes": F(
        "§4", (I,), nullable=True,
        note="episodes behind each checkpoint of the training curve -- the series "
             "`evaluate.fitness.checkpoint_aggregation` collapses, NOT the feedback "
             "pool `evaluate.rollouts_per_candidate` sizes. null = each backend's own "
             "constant (surrogate `EVAL_EPISODES` 4 with `EVAL_EPISODES_FINAL` 8 on the "
             "last row; sb3 a bare literal 3). An integer applies to every intermediate "
             "checkpoint on both paths and the surrogate's last row takes "
             "max(value, EVAL_EPISODES_FINAL). Matters most under a GATED fitness read "
             "by `max_over_checkpoints`, where a low count biases the maximum UP rather "
             "than merely widening it"),
    "evaluate.artifacts": F("§4", (list,),
                            item_enum=("scalar_metrics", "state_trajectories", "videos",
                                       "component_traces", "demo_reward_traces")),
    "evaluate.fitness.source": F("§4", (S,), kind="fitness_source",
        note="Which signal §4 ranks candidates on. The native-signal rule: a "
             "supervised search may only read the env's OWN signal, never the "
             "BIRD-authored task_metric (custom_metric). Default `native` resolves to "
             "the shipped binary success (discrete_success.kind: discrete) if the task "
             "ships one, else the shipped reward's return (native_reward/gt_return, "
             "unless reward.human.kind: none), else REFUSES -- never task_metric. "
             "`native_success`/`native_reward` pin one channel; `ground_truth_metric` "
             "reads the task_metric (custom_metric) and is what several published method "
             "configs pin (rf_agent, revolve, singh_orp, zeroshot). _check_coherence refuses a "
             "native source on a task lacking the signal (the five reward.human.kind: "
             "none gymnasium tasks refuse bare `native`)."),
    "evaluate.fitness.metric": F("§4", (S,)),
    "evaluate.fitness.reduction": F(
        "§4", (S,), nullable=False,
        enum=("per_step_fraction", "any_step_episode_fraction"),
        note="How a per-step ground-truth success check reduces to one number per episode. "
             "This is an enum rather than a registry family because the two values differ "
             "by one aggregation the env adapter applies, not by a component a stage "
             "dispatches to; a registry family here would be two one-line functions "
             "expressing one choice. The default is `per_step_fraction`, which is what "
             "every existing number on the Meta-World tier was measured under. The "
             "sparse `any_step_episode_fraction` reading collapses that tier's scores "
             "to {0, 1} per seed at the budget it can afford, so it is an ablation "
             "someone runs deliberately, not the default"),
    "evaluate.fitness.normalisation": F("§4", (S,), enum=("none", "human_normalised", "min_max")),
    "evaluate.fitness.checkpoint_aggregation": F("§4", (S,), kind="checkpoint_aggregation"),
    "evaluate.fitness.seed_aggregation": F("§4", (S,), kind="seed_aggregation"),
    "evaluate.feedback.numeric_reflection": F("§4", (B,)),
    "evaluate.rejudge_incumbent": F("§4", (B,),
                                    note="re-score state.best's stored rollouts every iteration under the "
                                         "current subtasks and repeats, so the incumbent comparison is on "
                                         "one scale; vlm_score only"),
    "evaluate.feedback.include_task_metric": F("§4", (B,)),
    "evaluate.feedback.metric_alias": F(
        "§4", (S,),
        note="Prompt-visible name for the fitness metric's curve in the numeric "
             "reflection; null shows the real name. Label only -- the series lookup "
             "keeps the real metric name. Eureka renames consecutive_successes to "
             "'task_score' before verbalising it (eureka.py:258-263)"),
    "evaluate.feedback.state_selection_scalar": F(
        "§4", (B,),
        note="Whether prompt-visible feedback labels the selection scalar: the "
             "'fitness=' header with its aggregation rule, per-seed values, and the "
             "'best so far' line in the carried dialogue turn. True is BIRD's default, "
             "not a published prompt: GT's Alg. 1 line 19 appends feedback : R_best : "
             "eta_best, eta_best being per-component training diagnostics (tex:336), "
             "never a strength (:318-320, App. B :618-662); false is Eureka's "
             "stats-only feedback (eureka.py:250-267). Prose only -- the scalar itself "
             "is untouched and stage 5 still selects on it. For "
             "evaluate.fitness.source: preference_bt the selection scalar is the "
             "Bradley-Terry strength, so false also removes the 'Preference evidence: "
             "Bradley-Terry strength, N wins, N losses' block "
             "(evaluation._section_preference); the numbers stay in report.json meta"),
    "evaluate.feedback.visual_analysis": F("§4", (B,)),
    "evaluate.feedback.granularity": F("§4", (S,),
                                       enum=("scalar_only", "per_component", "per_subtask")),
    # WHICH SERIES THE NUMERIC REFLECTION VERBALISES, and it is not a
    # presentation choice.
    #
    # `checkpoints` (default, unchanged behaviour) reads `result.checkpoints` --
    # the 5-20 held-out greedy evaluations `train.checkpoint_interval` produces
    # -- and samples ~10 points evenly. That is BIRD's operational
    # approximation of Eureka's block and `ckpt_max`'s docstring already says
    # so.
    #
    # `training_epochs` reads the LEARNER'S OWN per-epoch log, named series and
    # all, and strides it exactly as `eureka.py:252` does:
    # `epoch_freq = max(max_iterations // 10, 1)` then `[::epoch_freq]`, with
    # Max/Mean/Min over the FULL series and not over the stride. It is the same
    # routing `checkpoint_aggregation: max_over_training_epochs` already needs
    # (`reads_training_epochs`) applied to the prose channel, and it
    # exists because the two series are different measurements of different
    # things: a held-out greedy evaluation every ~150 epochs is not the
    # training curve Eureka's LLM was shown, and an LLM asked to diagnose a
    # reward component from the wrong one is being shown the wrong evidence.
    # No backend in this release fills the named per-epoch series
    # (`train.log: epoch_scalars`); on a backend that does not, the reflection
    # says so and falls back rather than rendering an empty block.
    "evaluate.feedback.series_source": F(
        "§4", (S,), enum=("checkpoints", "training_epochs"),
        note="checkpoints = the held-out evaluation curve, sampled evenly (BIRD's "
             "approximation). training_epochs = the learner's per-epoch log, "
             "strided at max_epochs // 10, which is Eureka's reward reflection "
             "verbatim (eureka.py:246-268)"),
    "evaluate.feedback.trajectory_examples": F(
        "§4", (S,), enum=("none", "best_and_worst", "random_k", "failures_only")),
    "evaluate.feedback.trajectory_sample_interval": F(
        "§4", (I,),
        note="Per-step stride at which an example rollout is verbalised; the final "
             "step is always included and rows re-stride evenly across the whole "
             "episode when the prompt-budget row cap binds. CARD publishes it "
             "(App. B.2 Table 8: 100 Meta-World / 25 ManiSkill2 = --log_interval, "
             "consumed by convert_traj_to_str); the default of 10 is BIRD's own"),
    "evaluate.feedback.analyzer": F("§4", (S,), enum=("none", "llm")),
    "evaluate.feedback.score_scale": F(
        "§4", (S,), enum=("[0,1]", "likert", "binary", "three_point", "staged"),
        note="three_point is RDA's in-loop rubric (App. 7.3: 1.0 success / 0.5 "
             "partial / 0.0 failure; parsed scores snap to those three values). "
             "staged anchors the continuous [0,1] scale to named stages of "
             "progress -- unpublished. An opt-in value that no config selects; "
             "it is not shown to improve ranking"),
    "evaluate.feedback.require_rationale": F("§4", (B,)),
    "evaluate.feedback.summarisation": F(
        "§4", (S,), enum=("none", "per_candidate", "per_candidate_per_subtask")),
    "evaluate.similarity.metric": F("§4", (S,), kind="similarity"),
    "evaluate.similarity.reference": F("§4", (S,), enum=("gt_reward", "previous_best", "none")),
    "evaluate.similarity.role": F("§4", (S,), enum=("report", "select", "screen")),
    "evaluate.preferences.enabled": F("§4", (B,)),
    "evaluate.preferences.comparator": F("§4", (S,), kind="comparator"),
    "evaluate.preferences.pairs": F("§4", (S,), kind="pair_strategy"),
    "evaluate.preferences.aggregator": F("§4", (S,), kind="pref_aggregator"),
    "evaluate.preferences.allow_ties": F("§4", (B,)),
    "evaluate.preferences.clip_length_s": F("§4", (I,)),
    "evaluate.preferences.fps": F("§4", (I,)),
    "evaluate.preferences.max_video_s": F("§4", (I,)),
    "evaluate.preferences.store_dataset": F("§4", (B,)),
    "evaluate.preferences.scope": F(
        "§4", (S,), enum=("within_iteration", "cumulative"),
        note="The AGGREGATION pool, and NOT the contest pool -- that is §5's "
             "select.scope, a different key with the same two values. `cumulative` "
             "refits the aggregator over every individual still held in state plus "
             "state.preferences, so ratings sit on one normalisation across rounds; it "
             "therefore requires store_dataset and `preference_dataset` in loop.carry. "
             "The PAIR pool stays this round's eligible reports either way. Re-read "
             "in §6 by update.topology=island_lineage, which refreshes each "
             "ArchiveCell.fitness from the refit: §4 recomputes the ranking and §6 "
             "is what acts on it, so without that the admission gate would compare "
             "against frozen numbers"),
    "evaluate.vlm.images_per_query": F("§4", (I,)),
    "evaluate.vlm.frame_policy": F("§4", (S,), kind="frame_policy"),
    "evaluate.vlm.contact_sheet.cell_px": F("§4", (I,)),
    "evaluate.vlm.contact_sheet.grid": F("§4", (S,)),
    "evaluate.vlm.contact_sheet.label_frames": F("§4", (B,)),
    "evaluate.vlm.judge_guidance": F(
        "§4", (S,),
        note="Non-null is appended to the VLM subtask judge's `## Instructions` block; "
             "the rubric, cascade rule and output contract stay. Ours; no paper pin"),
    "evaluate.vlm.repeats": F("§4", (I,),
                              note="Whether RDA's in-loop scorer repeats each query is "
                                   "unverified: 'the VLM evaluates each video 4 times' "
                                   "(RDA §5.1) describes the reported alignment metric, "
                                   "not the search, so the 4 belongs to "
                                   "alignment_rate.repeats"),
    "evaluate.human.mode": F("§4", (S,),
                             enum=("none", "feedback_text", "preference_labels", "veto")),
    "evaluate.human.queries_per_iteration": F("§4", (I,)),
    "evaluate.human.applies_to": F("§4", (S,), enum=("selected_only", "all_candidates")),
    # RF-Agent's two per-candidate LLM calls after training, both `phase` plugins
    # dispatched from stage 4 after the human phase, alignment first.
    "evaluate.thought_alignment.enabled": F(
        "§4", (B,),
        note="One generator call per trained candidate re-describing the reward from "
             "its idea + code in under 4 sentences -> report.meta['design_thought'] "
             "(initial_thought_alignment.txt; rfagent_algo.py:566-588). RF-Agent true"),
    "evaluate.self_verify.enabled": F(
        "§4", (B,),
        note="One evaluator call per trained candidate scoring its similarity to an "
             "imagined expert strategy -> report.meta['self_verify'] (float, or None "
             "when unparsed; the release falls back to 0, rfagent_algo.py:610-616). "
             "Enters uct_leaf as a softmax over siblings (Eq. 2; ‡ the softmax set is "
             "the code's, rfagent_algo.py:78), and puct_leaf as the prior P. RF-Agent true"),
    "evaluate.self_verify.range": F(
        "§4", (list,), nullable=False,
        note="[lo, hi] stated in the prompt and used to parse the `[x]` answer; not "
             "clamped (the release does not, rfagent_algo.py:611-614). An unverified "
             "node enters the sibling softmax at lo. RF-Agent [-1, 1] "
             "(neurips_2025_arxiv.tex:829)"),
    "evaluate.skip_if_screened": F("§4", (B,)),
    "evaluate.screened_fitness": F("§4", (S,), enum=("failure_value", "screen_measurement"),
                                   note="What fitness a screen-rejected candidate "
                                        "carries into §5/§6: the select.failure_value "
                                        "sentinel, or the rate the rejecting screen "
                                        "itself measured (LIMEN ‡ screen_measurement -- "
                                        "evaluator.py:237-258 keeps short_metrics on "
                                        "SHORT_TRAIN_REJECTED, database.py:225-299 "
                                        "grid-places it; a screen that measured nothing "
                                        "keeps the sentinel)"),
    # ---------------------------------------------------------------- §5 ---
    "select.rule": F("§5", (S,), kind="select_rule"),
    "select.n_survivors": F("§5", (I,)),
    "select.scope": F("§5", (S,), enum=("within_iteration", "cumulative")),
    "select.final_artifact": F("§5", (S,),
                               enum=("global_best", "chain_end", "archive_best", "all_survivors")),
    "select.failure_value": F("§5", NUM),
    "select.tie_break": F("§5", (S,), kind="tie_break"),
    "select.significance": F("§5", (S,), kind="significance"),
    "select.allocation": F("§5", (S,), kind="allocation"),
    "select.require_improvement_over_incumbent": F("§5", (B,)),
    "select.incumbent_simplicity_margin": F(
        "§5", NUM, nullable=False,
        note="With require_improvement_over_incumbent: a challenger within this margin below "
             "the incumbent's fitness still counts as improving when its reward has fewer AST "
             "nodes than the chain head it replaces (the Occam gate). 0.0 = strict. Refused > 0 "
             "without the incumbent rule. Ours, no paper pin"),
    "select.objectives": F("§5", (list,)),
    "select.retrain_before_select": F(
        "§5", (B,),
        note="DECLARED, NOT YET IMPLEMENTED: no stage reads it, so `_check_coherence` "
             "refuses `true` rather than let a run claim a mechanism that never "
             "executed (a declared-but-unread key)"),
    "select.human_override": F("§5", (B,)),
    # ---------------------------------------------------------------- §6 ---
    "update.topology": F("§6", (S,), kind="topology"),
    "update.operator": F("§6", (S,), kind="update_operator"),
    "update.winner.action": F("§6", (S,), kind="winner_action"),
    "update.loser.action": F("§6", (S,), kind="loser_action"),
    "update.prompt.mode": F("§6", (S,), kind="prompt_mode"),
    "update.prompt.assistant_content": F(
        "§6", (S,), enum=("reward_code", "raw_response", "nl_spec"),
        note="What the carried assistant turn contains: the extracted reward code, the "
             "model's full raw response including its own reasoning prose, or the "
             "candidate's stage-1 English design (Candidate.nl_spec, written by "
             "generate.output.two_stage_nl_then_code or design_thought=inline_brace). "
             "Eureka carries the raw response (eureka.py:331,335); GT carries the "
             "English -- App. B accumulates {all_english_rewards} and shows "
             "{reward_code} once (neurips_2025.tex:627-634). reward_code/raw_response "
             "fall back to each other; nl_spec falls back to the code and journals "
             "`assistant_content_fallback`"),
    "update.prompt.max_length_tokens": F("§6", (I,)),
    "update.feedback.routing": F("§6", (S,), kind="feedback_routing"),
    "update.elitism.keep_global_best": F("§6", (B,)),
    "update.rollback_if_worse": F("§6", (B,)),
    "update.archive.enabled": F("§6", (B,)),
    "update.archive.descriptors": F("§6", (list,)),
    "update.archive.bins_per_descriptor": F("§6", (list,)),
    "update.archive.n_islands": F("§6", (I,)),
    "update.archive.migration_interval": F("§6", (I,)),
    "update.archive.migration_clock": F(
        "§6", (S,), enum=("global_inserts", "island_generations"),
        note="Which clock `migration_interval` counts against. `island_generations` is "
             "the LIMEN release (`database.py::should_migrate` :532-535, "
             "`controller.py:534-537`): every iteration increments the CURRENT island's "
             "generation, the current island rotates every n_islands iterations, and "
             "migration fires when max(island_generations) - last_migration_gen >= "
             "interval -- so 3 islands x 30 single-candidate iterations end [12,9,9] and "
             "NEVER migrate (first event at iteration 56). `global_inserts` is BIRD's "
             "alternative clock: migration every `interval` archive insertions (once at "
             "30), a panmictic population wearing an island structure. The two are "
             "qualitatively different searches; LIMEN pins the release clock"),
    "update.archive.migration_rate": F("§6", NUM, note="fraction of a population that migrates"),
    "update.archive.population_size": F("§6", (I,), note="per island"),
    "update.archive.admission": F(
        "§6", (S,), enum=("truncate", "above_island_mean", "above_island_best"),
        note="Who may enter a deme. `truncate` keeps the top population_size and is "
             "every topology's default behaviour; the two gates refuse an "
             "offspring that does not clear the deme's current average / best. REvolve "
             "admits on the island AVERAGE (§3.3, p.6) and its footnote 1 argues "
             "against the max 'to ensure genetic diversity', which makes the paper's "
             "own design argument a single-key diff. Read by "
             "update.topology=island_lineage only"),
    "update.archive.archive_size": F("§6", (I,), note=(
        "caps the SAMPLING POOL the global parent branches draw from "
        "(top-K cells by fitness), never the grid; null = uncapped")),
    "update.archive.parent_sampling": F("§6", (dict,)),
    # `topology: search_tree` backup (Eq. 3): each ancestor of a new node takes
    # q = (1 - bcw - mw*decay)*q + bcw*max_child_q + mw*decay*mean, with
    # decay = max(0, 1 - sim_index/horizon_trainings).
    "update.tree.best_child_weight": F(
        "§6", NUM, nullable=False,
        note="eta of Eq. 3, the weight of the max child Q (neurips_2025_arxiv.tex:207-214; "
             "rfagent_algo.py:43). RF-Agent 0.7; 1.0 with mean_weight 0 is a plain "
             "max-backup"),
    "update.tree.mean_weight": F(
        "§6", NUM, nullable=False,
        note="Weight of the running mean of scores backed up through the node, times "
             "decay. † absent from Eq. 3; the code carries 0.15 "
             "(rfagent_algo.py:44,66-68). The negative decay the release reaches in its "
             "overshoot round (:694) is clamped at 0"),
    "update.memory.trajectory_store": F("§6", (S,),
                                        enum=("none", "append_on_pass", "append_always")),
    "update.co_evolve.subtasks": F("§6", (B,)),
    "update.co_evolve.observation_fn": F(
        "§6", (B,),
        note="Whether the co-designed observation function PERSISTS across iterations -- "
             "the §6 half of §1's `generate.co_design.observation_fn`, which it requires. "
             "true (LIMEN): a child is shown its parent's whole program, `OBS_DIM` and "
             "`get_observation` included, and under a non-rewriting "
             "`generate.output.edit_mode` inherits it. false: the parent's observation is "
             "cut out of everything the child is handed (PARENT REWARD, ARCHIVE ELITES, the "
             "program an edit mode patches), so each candidate carries the observation its "
             "own generation produced. Read by `generation._inheritable_program`; inert "
             "when §1's key is false"),
    "update.co_evolve.dr_config": F("§6", (B,)),
    "update.meta.prompt_optimizer": F(
        "§6", (S,), enum=("none", "gepa", "manual"),
        note="DECLARED, NOT YET IMPLEMENTED: nothing rewrites the templates, so "
             "`_check_coherence` refuses any value but `none` rather than let a run "
             "claim a prompt-optimised method (a declared-but-unread key)"),
    "update.restart.on_stagnation": F("§6", (B,)),
    "update.restart.patience": F("§6", (I,)),
    # ------------------------------------------------- pre/post phase cfg ---
    "rapp.enabled": F("§0", (B,), note="DrEureka pre-phase"),
    "rapp.policy": F("§0", (S,)),
    "rapp.parameters": F("§0", (list,)),
    "rapp.rollouts_per_value": F("§0", (I,)),
    "rapp.success_criterion": F("§0", (S,)),
    # ---------------------------------------------------------- run/output ---
    "output.dir": F(
        "§0", (S,),
        note="The run root when the CLI is given no `--out`; an explicit `--out` wins, "
             "and status.json records which chose it (`out_root_source`: cli | config)"),
    "output.save_state_every_iteration": F("§0", (B,)),
    "output.log_level": F("§0", (S,), enum=("DEBUG", "INFO", "WARNING", "ERROR")),
    "output.tracker": F("§0", (S,), kind="tracker",
                        note="experiment tracker; 'none' keeps a run entirely local"),
    "output.wandb.project": F("§0", (S,)),
    "output.wandb.entity": F("§0", (S,)),
    "output.wandb.group": F("§0", (S,), note="null = the config's `name`"),
    "output.wandb.tags": F("§0", (list,)),
    "output.wandb.mode": F("§0", (S,), enum=("online", "offline")),
    "output.wandb.log_videos": F("§0", (B,)),
    "output.wandb.video_record": F(
        "§0", (S,), enum=("none", "best", "best_and_worst", "all"),
        note="Which recordings are uploaded to the tracker; output.video.record says "
             "which are written to disk. They are separate keys because the disk "
             "artifact is evidence and the dashboard is a summary. An enum rather than "
             "a registry family, because no component implements the choice"),
    "output.wandb.log_code": F("§0", (B,)),
    "output.video.enabled": F("§0", (B,)),
    "output.video.record": F("§0", (S,), enum=("none", "best", "best_and_worst", "all")),
    "output.video.format": F("§0", (S,), kind="video_format",
                             note="'frames' writes PNGs with stdlib only; gif/mp4 need imageio"),
    "output.video.fps": F("§0", (I,)),
    "output.video.max_frames": F("§0", (I,)),
    "output.video.width": F("§0", (I,)),
    "output.video.n_views": F(
        "§0", (I,), nullable=False,
        note="Viewpoints per recorded frame. 1 = the task's primary camera, the frame "
             "every published config records and the judge is sent; N draws the "
             "first N-1 of the task spec's judge.extra_views beside it, left to "
             "right in one image, each panel output.video.width wide, and states "
             "the layout to the judge. The primary panel is pixel-identical to the "
             "single-view frame, so an ablation over this key changes only what the "
             "judge is additionally shown. Ours, no paper pin"),
    "output.judge_trace.record": F(
        "§0", (S,), enum=("none", "inputs", "inputs+frames"),
        note="Records what the (V)LM judge was shown, written at query time to "
             "judgments/iterNN.jsonl: the prompt verbatim, the contrastive caption "
             "that llm_on_vlm_captions consumes in flight, and for each frame set "
             "the episode step indices, image size, per-frame luma and per-frame "
             "content hash. Without this, report.json keeps the verdict and none of "
             "the evidence, so a judgment made blind and one made on twenty healthy "
             "frames are the same artifact. `inputs+frames` also stores the PNG "
             "bytes"),
    "output.trajectory_trace.record": F(
        "§0", (S,), enum=("none", "components", "components+states"),
        note="Per-episode trace pairing what the reward claimed to pay (its own "
             "component values) with what the harness measured it paying, for the "
             "episode record_rollouts rendered and at the same strided indices. It "
             "exists because `component_traces` in train_result.json is already a "
             "mean, and a mean is exactly where an inflated component hides"),
    "output.trajectory_trace.max_steps": F(
        "§0", (I,),
        note="per-episode sample cap, strided evenly like frames rather than "
             "truncated -- a trace of an episode's first N steps describes a "
             "different episode than the one recorded"),
    "output.video.timeout_s": F(
        "§0", NUM,
        note="A wedge detector rather than a budget: without it, a renderer that "
             "blocks forever burns the whole walltime with no traceback. Wall clock "
             "for one record_rollouts call, covering every pick together; null "
             "disables it. A set value also arms the stack-dump watchdog, which "
             "additionally wraps the stage-1 task_images() render"),
}


#: Keys that exist only in an unresolved file and are consumed by the loader.
META_KEYS = frozenset({"extends"})


def sections() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, f in SCHEMA.items():
        out.setdefault(f.section, []).append(key)
    return out
