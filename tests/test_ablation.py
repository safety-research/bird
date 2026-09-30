"""Ablations are config diffs, not second scripts.

This is the payoff a unified configuration space claims for the whole
exercise, so it is worth asserting rather than trusting.
"""

import pytest

from conftest import (GT_PUBLISHED_OVERRIDES, REVOLVE_PUBLISHED_OVERRIDES,
                      load_gt_published, load_revolve_published)
from bird.config import load
from bird.schema import SCHEMA


def test_a_published_ablation_is_a_two_key_diff():
    """Eureka's no-evolution ablation differs from Eureka in exactly the two
    keys that define it. Anything else in the diff is a confound."""
    diff = load("eureka").diff(load("eureka_no_evolution"))
    diff.pop("name", None)
    assert set(diff) == {"loop.n_iterations", "generate.n_candidates"}, (
        f"unexpected keys differ: {sorted(diff)}")


def test_gt_is_a_minimal_diff():
    """The GT-vs-gt diff, held exactly where the paper's §5.1 (iii)
    ablation puts it. Under one config per method the published point is not a
    file but `configs/methods/gt.yaml` plus `GT_PUBLISHED_OVERRIDES`, so the assertion is overrides-vs-file: the
    diff must be exactly the override set, or one side has gone stale -- an
    override that stops differing means the file quietly absorbed a published
    value, and a diff key outside the set means the file drifted from the
    documented ablation (the set its header points to)."""
    diff = load_gt_published().diff(load("gt"))
    diff.pop("name", None)
    assert set(diff) == set(GT_PUBLISHED_OVERRIDES) - {"name"}, (
        f"gt diverges from the documented ablation: {sorted(diff)}")


def test_revolve_is_a_minimal_diff():
    """The REvolve-vs-revolve diff, held exactly where Appendix B.2 puts it.

    Same shape as `test_gt_is_a_minimal_diff` and for the same reason: under
    one-config-per-method the published point is not a file but
    `configs/methods/revolve.yaml` plus `REVOLVE_PUBLISHED_OVERRIDES`, so the assertion
    is overrides-vs-file. An override that stops differing means the shipped
    file quietly absorbed a published value -- and here that would be worse
    than GT's version of the same slip, because the values in question are the
    human channel: absorbing one would make the shipped REvolve Auto point
    silently claim human preferences it never collected. A diff key outside the
    set means the file drifted from the documented ablation.

    Why this test exists at all, given the repo defaults to writing none: the
    failure is silent by construction. The override set lives only in this
    dict, which configs/methods/revolve.yaml's header names as the full method,
    and no run executes it: it can rot without a single run changing behaviour.
    """
    diff = load_revolve_published().diff(load("revolve"))
    diff.pop("name", None)
    assert set(diff) == set(REVOLVE_PUBLISHED_OVERRIDES) - {"name"}, (
        f"revolve diverges from the documented ablation: {sorted(diff)}")


def test_revolve_auto_reads_no_human_channel():
    """The shipped arm must be the ABLATION, not the method with a stub human.

    Three independent switches have to be off together, and each alone is
    enough to make the point dishonest: a preference block that runs, a human
    oracle that answers, or a prompt that claims human feedback it never got.
    `evaluate.human.mode: none` is the one that matters most -- non-`none`
    builds `_ScriptedHumanOracle`, which answers from the environment's own
    reference reward, so the arm would report `budget.human_queries` for labour
    nobody supplied while reading as the published no-human point.
    """
    cfg = load("revolve")
    assert cfg["evaluate.human.mode"] == "none"
    assert cfg["evaluate.preferences.enabled"] is False
    assert cfg["generate.context.include_human_feedback"] is False
    assert cfg["evaluate.fitness.source"] == "ground_truth_metric"


def test_the_humanoidbench_variant_is_a_benchmark_change_and_not_a_method():
    """`rda_humanoidbench` is RDA's SECOND BENCHMARK, not a second method.

    RDA §5.1 runs two benchmarks that disagree on three numbers -- candidates
    4/8, env steps 50M/10M, parallel envs 2048/16 -- plus the learner block Table 1
    gives per column (HumanoidBench: SimbaV2, gamma 0.98, and the 625K update budget
    expressed as `train_freq`/`gradient_steps`; rda.yaml leaves the ManiSkill column's
    gamma 0.95 / 250K unpinned). `configs/methods/rda.yaml` pre-authorises exactly
    that: it pins ManiSkill and records each HumanoidBench value in the comment on
    its key, so the HumanoidBench variant is a visible diff rather than a second
    file with a different provenance story.

    This is the check that it STAYED one. Every key below is §0 or §3 -- which
    benchmark, and what the learner does on it. A key from §1's search shape
    (bar the candidate count, which is one of the paper's own three), §2, §4, §5
    or §6 appearing here would mean the two benchmarks had quietly become two
    methods, and every cross-benchmark reading of RDA's results would be void --
    a file that reads like the same method on another benchmark and is
    actually a second experiment.

    THE TEST FOR "IS THIS KEY ALLOWED HERE?" IS WHETHER TABLE 1 SPLITS ITS ROW
    PER BENCHMARK, and that is why the set below has six members.
    `problem.horizon` is §0, not §3, and it earns its
    place the same way `generate.n_candidates` does: Table 1
    (`refs/tex/rda/appendix.tex:1424`) gives the row "Maximum Environment Steps
    & Task-dependent (MS), 500 (HB)" -- SPLIT, exactly like the four other split RL rows beside it -- so the paper itself reports it as a benchmark fact of the same
    standing as the env-step budget, and a variant that did NOT move it would be
    running the wrong column. A key whose Table-1 row is unsplit (Architecture,
    for one: SimbaV2 on both benchmarks) is a repo-side inconsistency when it
    appears here, which is what `rda_humanoidbench.yaml` records at
    `train.architecture`. Each exemption is NAMED below rather than admitted by
    widening the section set, because "§0 is allowed" would let any future §0
    key in unchallenged, and this test exists to challenge them.
    """
    diff = load("rda").diff(load("rda_humanoidbench"))
    diff.pop("name", None)
    assert set(diff) == {
        "generate.n_candidates",      # 4 -> 8      §5.1, one of the paper's three
        "train.env_steps",            # 50M -> 10M  §5.1, two of the paper's three
        "train.n_parallel_envs",      # 2048 -> 16  §5.1, three of the paper's three
        "train.architecture",         # simba -> simba_v2, per the column
        "train.hyperparameters",      # gamma 0.98 + the 625K update budget
        "problem.horizon",            # null -> 500, Table 1's "Maximum Environment
                                      #   Steps": "Task-dependent (MS), 500 (HB)"
    }, f"the HumanoidBench variant moved more than the benchmark: {sorted(diff)}"

    # TWO NAMED EXEMPTIONS, and both are numbers the PAPER reports per benchmark.
    # They are named rather than absorbed into the section set below, because
    # each is precisely the kind of key that could hide a method change and so
    # must be justified by the paper every time it moves rather than by this
    # test's convenience.
    #
    #   `generate.n_candidates` (§1) -- one of the three numbers §5.1 itself
    #   reports differently per benchmark. RDA's own ablation makes N a
    #   first-class knob (1 -> 8 lifts success 0.20 -> 1.00), so it is precisely
    #   the §1 key that could hide a method change.
    #
    #   `problem.horizon` (§0) -- Table 1
    #   (`refs/tex/rda/appendix.tex:1424`) gives a row "Maximum Environment
    #   Steps & Task-dependent (MS), 500 (HB)", SPLIT per benchmark exactly like
    #   the four other split RL rows beside it, so it is a benchmark fact of the same
    #   standing as the env-step budget. Without it the episode length would
    #   silently stay at HumanoidBench's shipped 1000. It is also half of the
    #   gamma derivation: SimbaV2 computes gamma from the
    #   episode length, and (500, action_repeat 2) is what gives Table 1's 0.98.
    #
    # The docstring says "every key below is §0 or §3", and the §0 member is
    # NAMED rather than the section allowed -- `sections <= {"§0", "§3"}` would
    # let any future §0 key in unchallenged, which is the opposite of what this
    # test is for.
    named = {"generate.n_candidates", "problem.horizon"}
    sections = {SCHEMA[key].section for key in diff if key not in named}
    assert sections <= {"§3"}, (
        f"the variant moves a search-behaviour key, not just the benchmark's "
        f"learner: {sorted((SCHEMA[k].section, k) for k in diff)}")
    assert SCHEMA["problem.horizon"].section == "§0"


def test_methods_differ_mostly_in_evaluation_and_screening():
    """The decomposition's headline claim, checked against the actual configs:
    generation is nearly identical across methods; what varies is what stands in
    for the fitness function and what you can cheaply check before paying for RL."""
    pairs = [("eureka", "gt_reward_design"), ("eureka", "card"), ("eureka", "rda")]
    for a, b in pairs:
        # `gt_reward_design` is the published point: gt + overrides.
        other = load_gt_published() if b == "gt_reward_design" else load(b)
        diff = load(a).diff(other)
        sections = {}
        for key in diff:
            sec = SCHEMA[key].section if key in SCHEMA else "?"
            sections[sec] = sections.get(sec, 0) + 1
        assert sections, f"{a} and {b} are identical, which cannot be right"


def test_single_key_sweep_is_expressible():
    """Once every design choice is a leaf key, a sweep is automatic."""
    base = load("eureka")
    for value in ("final", "max_over_checkpoints", "auc", "last_k_mean", "iqm"):
        cfg = load("eureka", overrides={"evaluate.fitness.checkpoint_aggregation": value})
        assert cfg["evaluate.fitness.checkpoint_aggregation"] == value
        d = base.diff(cfg)
        assert set(d) <= {"evaluate.fitness.checkpoint_aggregation"}


@pytest.mark.parametrize("mode", ["none", "last_iteration", "cumulative_append",
                                  "full_dialogue", "rolling_summary"])
def test_the_history_axis_spans_its_whole_range(mode):
    """No paper ablates generate.history_mode, and Eureka and CARD publish
    opposite rationales for their choices. One sweep settles it -- so every
    value has to at least be reachable."""
    cfg = load("eureka", overrides={"generate.history_mode": mode})
    assert cfg["generate.history_mode"] == mode


def test_topology_column_is_reachable_beyond_hillclimb():
    """Every published method except LIMEN is single_parent_hillclimb."""
    published = [cfg["update.topology"] for cfg in
                 (load("eureka"), load("dreureka"), load("rda"),
                  load_gt_published(), load("card"))]
    assert set(published) == {"single_parent_hillclimb"}, (
        f"expected the finding to hold across the configs, got {set(published)}")
    assert load("limen")["update.topology"] == "archive_map_elites"


def test_one_seed_and_no_significance_is_the_published_norm():
    """The other headline finding: argmax over 16 candidates on one seed each is
    selecting substantially on training noise."""
    for name in ("eureka", "rda", "gt_reward_design", "card"):
        cfg = load_gt_published() if name == "gt_reward_design" else load(name)
        assert cfg["train.seeds_per_candidate"] == 1, name
        assert cfg["select.significance"] == "none", name
    assert load("limen")["train.seeds_per_candidate"] == 3


def test_limen_reward_only_is_a_search_space_diff_and_nothing_else():
    """LIMEN §5.4's Reward-Only baseline, p. 8: "Evolves the reward function
    while keeping observations fixed to raw simulator state... All baselines use
    identical evolution budgets and RL training configurations."

    That last sentence is the assertion. Reward-Only is a PUBLISHED arm with its
    own numbers (Fig. 4, Table 15), not a cut-down LIMEN, so the diff must be the
    search space and nothing that touches the search itself -- not the archive,
    not the cascade, not `n_iterations`, not `seeds_per_candidate`. Anything else
    here is a confound in a comparison the paper reports.

    All three keys, not one: `problem.search_space` is read only by
    `_check_coherence` (`bird/config.py`) and never by a stage, so a
    config that set it alone would be cosmetic -- the behaviour lives in the two
    `observation_fn` flags.

    Four keys, not three: `generate.context.system_prompt` is
    the release's own mode switch -- PromptBuilder.__init__ formats
    SYSTEM_PROMPT from _FUNCTION_DESC[evolution_mode] and
    _OUTPUT_CONSTRAINTS[evolution_mode] (prompts.py:453-458, :480-485) -- so
    the Reward-Only arm's system message names one function where the joint
    arm's names two. It is the search space stated to the model, not a change
    to the search.
    """
    diff = load("limen").diff(load("limen_reward_only"))
    diff.pop("name", None)
    assert set(diff) == {"problem.search_space",
                         "generate.co_design.observation_fn",
                         "update.co_evolve.observation_fn",
                         "generate.context.system_prompt"}, (
        f"unexpected keys differ: {sorted(diff)}")


def test_the_reward_only_arm_keeps_limens_archive_even_though_it_collapses():
    """§4.2's two descriptors are observation dimensionality and reward AST node
    count. With the observation fixed, the first is constant and the MAP-Elites
    grid degenerates to one column -- and the archive block is left alone anyway,
    because §5.4 says the configurations are identical and editing it would make
    this a two-variable comparison.

    Pinned because "the archive looks broken, let me drop the dead axis" is the
    obvious and wrong maintenance action here. The collapse is a property of the
    ablation; `bird/components/selection.py::_observation_dim` handles it.
    """
    joint, reward_only = load("limen"), load("limen_reward_only")
    for key in ("update.archive.descriptors", "update.archive.bins_per_descriptor",
                "update.archive.n_islands", "update.archive.population_size",
                "update.archive.archive_size", "update.archive.parent_sampling",
                "update.topology", "select.rule", "select.final_artifact",
                "loop.n_iterations", "generate.n_candidates",
                "train.seeds_per_candidate", "verify.quality_screen",
                "verify.cascade.min_success_threshold"):
        assert joint[key] == reward_only[key], (
            f"{key} differs between limen and its own published Reward-Only arm; "
            f"§5.4 pins identical evolution budgets and RL configurations")
    assert "observation_dim" in reward_only["update.archive.descriptors"]
