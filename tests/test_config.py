"""Config loading, inheritance and validation.

The three rules under test: lists replace on extends, unknown keys are fatal,
and the resolved config is a hashable artifact.
"""

import pytest
import yaml

from conftest import (CONFIG_DIRS, PAPER_CONFIGS,
                      _configs, config_dirs, unclassified_config_dirs)
from bird.config import (CONFIG_ROOT, Config, ConfigError, _find_config,
                         deep_merge, load, parse_override, parse_overrides)


@pytest.mark.parametrize("path", PAPER_CONFIGS, ids=lambda p: p.stem)
def test_paper_config_loads_and_validates(path):
    cfg = load(path)
    assert cfg["name"], f"{path.name} has no name"
    assert cfg["loop.n_iterations"] >= 1


def _profile(stem: str) -> dict:
    """A profile's OWN yaml, read raw rather than resolved through `load`.

    Deliberately not `load`: a profile is a fragment that states execution keys
    only, so the question these tests ask is what the FILE pins, not what it
    resolves to once a method config has been merged over it. Reading it raw is
    also what keeps them independent of how `--profile` composes.
    """
    return yaml.safe_load((CONFIG_ROOT / "_profiles" / f"{stem}.yaml").read_text())


def test_tester_profile_is_offline_and_mock():
    """The tester profile is offline: the suite runs it end to end with no API
    key and no wandb run."""
    prof = _profile("tester")
    assert prof["llm"]["generator"]["provider"] == "mock", (
        "the tester profile does not use the mock LLM; it would need an API key")
    assert prof["llm"]["evaluator"]["provider"] == "mock", (
        "the tester profile's evaluator is not mock; it would need an API key")
    assert prof["output"]["tracker"] == "none", (
        "the tester profile would open a wandb run")


# --------------------------------------------------------------------------
# every directory under `configs/` is classified, or this fails
# --------------------------------------------------------------------------


def test_every_config_directory_is_classified():
    """A new config directory must fail LOUDLY until someone says what it holds.

    A directory nobody classified arrives completely unconstrained -- a file
    there could quietly change `select.rule` and no test would notice -- unless
    something requires every new directory to be classified.

    The failure mode is the one that matters most in this repo: not a red test,
    but a green one that never looked. So the classification is mandatory
    rather than derived: every directory is listed WITH A STATED REASON, and
    adding a directory without one lands here."""
    unknown = unclassified_config_dirs(config_dirs())
    assert not unknown, (
        f"unclassified directories under configs/: {sorted(unknown)}.\n"
        "Add each to tests/conftest.py's CONFIG_DIRS with what it holds and the "
        "test that covers it. A directory nothing constrains can change the "
        "method unseen.")


def test_the_classifier_would_catch_a_new_directory():
    """The mechanism, not the current directory listing.

    An assertion that only ever runs over the directories that happen to exist
    cannot demonstrate that it would catch one that does not -- it passes
    identically whether it is strict or vacuous.

    `known=` is passed explicitly, so both branches are exercised against a set
    the test controls: a classifier that reported EVERY name as unclassified
    would satisfy the negative half alone."""
    assert unclassified_config_dirs(["alpha", "beta"],
                                    known={"alpha", "beta"}) == set()
    assert unclassified_config_dirs(["alpha", "brand_new_dir"],
                                    known={"alpha"}) == {"brand_new_dir"}
    # And the wiring: the default `known` is CONFIG_DIRS, which does not hold it.
    assert unclassified_config_dirs(["brand_new_dir"]) == {"brand_new_dir"}


def test_every_exemption_states_a_reason():
    """`CONFIG_DIRS` is a mapping and not a set on purpose: a classification is
    an argument someone has to make, and an empty one is a permission slip."""
    for name, reason in CONFIG_DIRS.items():
        assert reason and len(reason) > 40, (
            f"configs/{name}/ is classified with no real reason recorded; say "
            "what it holds and which test covers it")



def test_dev_profile_is_small_but_real():
    """The dev profile is the one that actually runs -- real model, real learner,
    small budgets -- against paper configs that stay unrunnable on purpose. The
    budget ceiling is the assertion worth keeping: a `dev` run is supposed to
    FINISH, and the failure it catches
    is somebody raising `train.env_steps` here to make one comparison look better
    and silently turning every dev run into an overnight job."""
    prof = _profile("dev")
    assert prof["loop"]["n_iterations"] >= 1
    assert prof["train"]["env_steps"] <= 2_000_000, (
        f"the dev profile asks for {prof['train']['env_steps']} env steps; "
        "these are meant to finish")
    assert prof["train"]["backend"] == "sb3", (
        "the dev profile is the tier that runs a REAL learner; a mock backend "
        "here would leave nothing between `tester` and `full`")


def test_unknown_key_is_fatal(tmp_path):
    """A typo'd key that silently defaults quietly invalidates an ablation."""
    p = tmp_path / "typo.yaml"
    p.write_text("name: typo\ngenerate:\n  n_candidatez: 4\n")
    with pytest.raises(ConfigError) as exc:
        load(p)
    assert "n_candidatez" in str(exc.value)
    assert "n_candidates" in str(exc.value), "should suggest the near-miss key"


def test_out_of_enum_value_is_fatal(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nselect:\n  scope: sideways\n")
    with pytest.raises(ConfigError, match="scope"):
        load(p)


def test_validation_reports_every_problem_at_once(tmp_path):
    p = tmp_path / "many.yaml"
    p.write_text("name: many\nselect:\n  scope: sideways\ngenerate:\n  n_candidatez: 4\n")
    with pytest.raises(ConfigError) as exc:
        load(p)
    msg = str(exc.value)
    assert "scope" in msg and "n_candidatez" in msg, "should not stop at the first problem"


def test_a_legal_null_does_not_escape_validate_as_a_type_error():
    """`select.n_survivors: null` is schema-legal (nullable int) and the runtime
    readers -- `selection._cfg_int`, update.py's `or 1` -- treat it as 1. A
    `_check_coherence` that compares `None > 1` makes validate() escape with a
    raw TypeError: the collected problems never reach the user and the config
    path is lost with them. The second override is the problem
    that must still be reported once the comparison stops raising, which is the
    every-problem-at-once contract `test_validation_reports_every_problem_at_once`
    pins for a different pair of keys."""
    with pytest.raises(ConfigError) as exc:
        load("eureka", profile="tester",
             overrides={"select.n_survivors": None, "generate.n_candidates": "oops"})
    msg = str(exc.value)
    assert "generate.n_candidates" in msg and "oops" in msg
    assert "n_survivors" not in msg, "null n_survivors is legal and must not be reported"
    # Alone, the null is simply the default.
    assert load("eureka", profile="tester",
                overrides={"select.n_survivors": None})["select.n_survivors"] is None
    # And the rule the comparison serves still fires on a real value.
    with pytest.raises(ConfigError, match="n_survivors>1 is meaningless"):
        load("eureka", profile="tester",
             overrides={"select.n_survivors": 2,
                        "update.topology": "single_parent_hillclimb"})


def test_a_non_int_n_candidates_is_reported_not_raised():
    """The sibling: `select.rule: none` compares `generate.n_candidates > 1`, and
    `g(...) or 0` let a string through to the comparison. `n_cand_or` exists for
    exactly this and the R* rules already use it."""
    with pytest.raises(ConfigError) as exc:
        load("eureka", profile="tester",
             overrides={"generate.n_candidates": "oops", "select.rule": "none"})
    assert "generate.n_candidates" in str(exc.value)


def test_a_profile_outranks_the_method_on_its_keys():
    """The merge order, PINNED BY MEASUREMENT: `defaults <- extends chain <- file
    <- profile <- CLI overrides`, measured here so that `load()`'s docstring,
    `PROFILE_KEY_PREFIXES`' header, `_check_profile_scope`'s error text and
    configs/methods/rda_humanoidbench.yaml cannot disagree with the code unnoticed. The reason the
    profile wins is in `PROFILE_KEY_PREFIXES`: eureka.yaml pins its paper budget,
    so a profile underneath it would be inert on exactly the keys it exists to
    set. A CLI override still beats the profile, and the profile NAME is
    recorded, so the layer is visible in the resolved artifact."""
    eureka = yaml.safe_load((CONFIG_ROOT / "methods" / "eureka.yaml").read_text())
    tester = _profile("tester")
    paper, tier = eureka["train"]["env_steps"], tester["train"]["env_steps"]
    assert paper != tier, "the probe needs a key on which the two disagree"
    assert load("eureka")["train.env_steps"] == paper
    resolved = load("eureka", profile="tester")
    assert resolved["train.env_steps"] == tier, (
        f"the tester profile lost to eureka.yaml on train.env_steps ({resolved['train.env_steps']}): "
        "a profile OUTRANKS the method on PROFILE_KEY_PREFIXES, or the tester tier "
        "is not a tester tier")
    assert resolved["loop.n_iterations"] == tester["loop"]["n_iterations"] != eureka["loop"]["n_iterations"]
    assert resolved["profile"] == "tester"
    assert load("eureka", profile="tester",
                overrides={"train.env_steps": 12345})["train.env_steps"] == 12345, (
        "a CLI override sits above the profile")
    # The statement and the measurement, in one place: the docstring must say
    # what this test just measured.
    assert "defaults <- extends chain <- file <- profile <- CLI overrides" in load.__doc__


def test_lists_replace_they_do_not_append():
    """Appending across an extends chain silently changes the method."""
    base = {"verify": {"dynamic_checks": ["execution_smoke", "output_shape"]}}
    override = {"verify": {"dynamic_checks": ["finite_values"]}}
    assert deep_merge(base, override)["verify"]["dynamic_checks"] == ["finite_values"]


def test_mappings_merge_recursively():
    base = {"train": {"env_steps": 100, "seeds_per_candidate": 1}}
    override = {"train": {"seeds_per_candidate": 3}}
    merged = deep_merge(base, override)
    assert merged["train"] == {"env_steps": 100, "seeds_per_candidate": 3}


def test_extends_chain_resolves_oldest_first(tmp_path):
    (tmp_path / "a.yaml").write_text("name: a\nloop:\n  n_iterations: 1\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\nname: b\nloop:\n  n_iterations: 5\n")
    (tmp_path / "c.yaml").write_text("extends: b.yaml\nname: c\n")
    cfg = load(tmp_path / "c.yaml", validate_config=False)
    assert cfg["name"] == "c"
    assert cfg["loop.n_iterations"] == 5
    assert cfg.lineage == ["a.yaml", "b.yaml", "c.yaml"]


def test_circular_extends_is_detected(tmp_path):
    (tmp_path / "x.yaml").write_text("extends: y.yaml\nname: x\n")
    (tmp_path / "y.yaml").write_text("extends: x.yaml\nname: y\n")
    with pytest.raises(ConfigError, match="circular"):
        load(tmp_path / "x.yaml", validate_config=False)


def test_config_hash_is_stable_and_content_sensitive():
    a = load("eureka")
    b = load("eureka")
    assert a.hash() == b.hash()
    c = load("eureka", overrides={"generate.n_candidates": 999})
    assert a.hash() != c.hash()


def test_cli_override_parsing():
    assert parse_override("generate.n_candidates=16") == ("generate.n_candidates", 16)
    assert parse_override("verify.enabled=true") == ("verify.enabled", True)
    assert parse_override("verify.tpe.discount=0.99") == ("verify.tpe.discount", 0.99)
    assert parse_override("name=foo") == ("name", "foo")


def test_a_repeated_set_key_is_refused_rather_than_taken_last():
    """A repeated `-s` key would otherwise let argument order decide an
    experiment silently, so it is an ERROR.

    `dict(parse_override(s) for s in args.set)` takes the last value with
    nothing printed. A command line that sets `loop.carry` twice -- once
    including `policy_checkpoint` and once not -- is then decided by the order
    the two were written in. An ablation whose definition is "the checkpoint
    component is off" would run with it on, and nothing in the run would say
    so: both orders resolve, both exit 0, and `_check_coherence` refuses
    neither combination.

    The message names BOTH values, because "you gave it twice" without them
    sends the reader back to a shell line they have already misread once.
    """
    with pytest.raises(ConfigError) as exc:
        parse_overrides(["loop.carry=[\"best_reward\",\"policy_checkpoint\"]",
                         "loop.n_iterations=5",
                         "loop.carry=[\"best_reward\"]"])
    msg = str(exc.value)
    assert "loop.carry" in msg and "given twice" in msg
    assert "policy_checkpoint" in msg, "the first value is not in the message"
    assert "best_reward" in msg, "the second value is not in the message"


def test_distinct_set_keys_are_unaffected_and_still_parse_to_values():
    """The other direction, so the refusal is not "more than one --set".

    Without this, a rule that rejected every repeated FLAG rather than every
    repeated KEY would pass the test above and break every real command line.
    """
    got = parse_overrides(["generate.n_candidates=16", "verify.enabled=true",
                           "name=foo", "verify.tpe.discount=0.99"])
    assert got == {"generate.n_candidates": 16, "verify.enabled": True,
                   "name": "foo", "verify.tpe.discount": 0.99}
    assert parse_overrides([]) == {}


def test_the_refusal_does_not_depend_on_how_the_flag_is_spelled():
    """`-s`, `--set`, `-sK=V`, `--set=K=V` and any mix of them.

    This is why the guard lives at the mapping and not in a pre-flight shell
    check over the command line: argparse has already normalised all four
    spellings into one list by the time `parse_overrides` sees them, so the
    refusal cannot be walked around by writing the second one differently. A
    `grep`-based check over argv can be: one matching a bare `-s` token goes
    quiet on `--set`, on `-sK=V`, on `--set=K=V`, and on a mixed pair.

    `main()` is where the normalisation happens, so the argv shapes are
    asserted there in `tests/test_cli_duplicate_set.py`; this pins the half
    that does not need a subprocess.
    """
    for pair in (["a.b=1", "a.b=2"], ["a.b=1", "a.b=1"]):
        with pytest.raises(ConfigError):
            parse_overrides(pair)


def test_coherence_rules_fire(tmp_path):
    """Cross-key invariants a per-key schema cannot express."""
    p = tmp_path / "incoherent.yaml"
    p.write_text(yaml.safe_dump({
        "name": "incoherent",
        # per_subtask feedback with nothing defining the subtasks
        "evaluate": {"feedback": {"granularity": "per_subtask"}},
        # secondary replay buffer that is never carried between iterations
        "train": {"init": "secondary_replay_buffer"},
    }))
    with pytest.raises(ConfigError) as exc:
        load(p)
    msg = str(exc.value)
    assert "decomposition.enabled" in msg
    assert "replay_buffer" in msg


def test_a_fasttd3_cell_cannot_freeze_its_actor():
    """`num_updates > 1` with `policy_frequency == 1` never updates the actor.

    The inner loop's condition is `i % policy_frequency == 1`, and no integer
    satisfies that when the frequency is 1 -- so the critic trains, the run
    reports normally, and the policy that ships is the initialisation. A
    silent null result is the worst failure shape this repo has, and it is
    one config key away.

    THE BRANCH IS UPSTREAM'S AND THE REFUSAL IS OURS. FastTD3's own loop has
    it (`refs/code/FastTD3/fast_td3/train.py:669-674`) and upstream cannot
    reach it by default either -- `hyperparams.py:63` declares
    `policy_frequency: int = 2` once and no preset overrides it. Faithful
    transcription and "cannot silently freeze the actor" are different
    requirements; the port keeps the branch, the loader refuses the pair.

    Refusing costs nothing: resolved across every config x profile under both
    fasttd3 routes, every fasttd3-effective cell is (num_updates=2,
    policy_frequency=2) and nothing in configs/ or scripts/ sets either key.
    """
    base = {"train.backend": "fasttd3", "train.algorithm": "fasttd3"}

    with pytest.raises(ConfigError) as exc:
        load("eureka", profile="tester", overrides=dict(
            base, **{"train.hyperparameters": {"num_updates": 2,
                                               "policy_frequency": 1}}))
    msg = str(exc.value)
    assert "never updates the actor" in msg, msg
    assert "train.py:669-674" in msg, "the message must name the upstream branch: " + msg

    # THE DEFAULTS MUST NOT BE REFUSED, which is the half a guard usually
    # gets wrong. (2, 2) is what every shipped config resolves to.
    ok = load("eureka", profile="tester", overrides=dict(
        base, **{"train.hyperparameters": {"num_updates": 2,
                                           "policy_frequency": 2}}))
    assert ok["train.hyperparameters"]["policy_frequency"] == 2

    # policy_frequency=1 is FINE on its own: with num_updates == 1 the loop
    # takes the other branch (`global_it % policy_frequency == 0`), which is
    # true every iteration -- the actor updates every time.
    load("eureka", profile="tester", overrides=dict(
        base, **{"train.hyperparameters": {"num_updates": 1,
                                           "policy_frequency": 1}}))

    # And the pair is only meaningful on this learner: the same keys mean
    # nothing to sb3 and must not be refused there.
    load("eureka", profile="tester", overrides={
        "train.backend": "sb3", "train.algorithm": "sac",
        "train.hyperparameters": {"num_updates": 2, "policy_frequency": 1}})


def test_nl_spec_needs_a_writer_of_candidate_nl_spec():
    """`update.prompt.assistant_content: nl_spec` (GT, App. B) carries
    `Candidate.nl_spec`; with nothing writing that field every carried turn
    would silently fall back to the code, so the load refuses it. Two writers
    exist -- two_stage_nl_then_code (GT) and design_thought: inline_brace
    (RF-Agent) -- and either satisfies the rule."""
    with pytest.raises(ConfigError) as exc:
        load("gt", profile="tester",
             overrides={"generate.output.two_stage_nl_then_code": False})
    msg = str(exc.value)
    assert "needs a writer of Candidate.nl_spec" in msg
    assert "two_stage_nl_then_code" in msg
    # the other writer
    cfg = load("gt", profile="tester",
               overrides={"generate.output.two_stage_nl_then_code": False,
                          "generate.output.design_thought": "inline_brace"})
    assert cfg["update.prompt.assistant_content"] == "nl_spec"


#: The only keys on which two configs in one directory may be each other's sole
#: difference -- they NAME an arm, they do not define it.
_LABEL_ONLY_KEYS = {"name", "output.wandb.tags"}


#: The directories this pairwise check runs over: the ERA recipes at the top level
#: and every directory beside them (`examples/`, `hillclimb/`, `methods/`).
_PAIRWISE_DIRS = [""] + config_dirs()


@pytest.mark.parametrize("subdir", _PAIRWISE_DIRS, ids=lambda d: d or "top")
def test_no_two_configs_in_a_directory_are_the_same_method(subdir):
    """Two arms in one directory must resolve to two different methods.

    An overlay that sets the keys DEFINING an ablation to the values its baseline
    already uses fails nothing: it validates, it runs, and it produces a report row
    whose "A vs B" difference is pure sampling noise while costing a second full
    search. An overlay of `eureka_no_evolution` that writes 5 x 16 on both
    arms does exactly that: it overwrites `loop.n_iterations` and
    `generate.n_candidates`, which ARE the ablation
    (`tests/test_ablation.py::test_a_published_ablation_is_a_two_key_diff`).

    `test_ablation.py` cannot see it: it loads only the published configs.

    A directory holding fewer than two configs yields no pair and passes
    trivially; that is a property of the directory, not of this check."""
    paths = _configs(subdir)
    assert paths, f"configs/{subdir} holds no configs at all"
    resolved = {p.stem: load(p) for p in paths}
    collisions = []
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            differing = set(resolved[a.stem].diff(resolved[b.stem]))
            if differing <= _LABEL_ONLY_KEYS:
                collisions.append(
                    f"{a.stem} == {b.stem} (differ only in {sorted(differing)})")
    assert not collisions, (
        f"configs/{subdir} holds arms that resolve to the same method, so running "
        f"both buys one measurement twice: {collisions}")


# --------------------------------------------------------------------------
# Config references: an exact path under `configs/`, else a unique bare name.
# --------------------------------------------------------------------------


def test_a_config_resolves_by_relative_path_or_by_bare_name():
    """The references the README, `scripts/paper_cell.py` and `--list-configs`
    hand out: a bare name finds the one file of that name wherever it lives, and
    a path relative to `configs/` (with or without `.yaml`) names it exactly."""
    for ref, rel in (("eureka", "methods/eureka.yaml"),
                     ("methods/eureka", "methods/eureka.yaml"),
                     ("eureka.yaml", "methods/eureka.yaml"),
                     ("era_u", "era_u.yaml"),
                     ("v2_verify", "hillclimb/v2_verify.yaml"),
                     ("hillclimb/v2_verify", "hillclimb/v2_verify.yaml"),
                     ("hillclimb/v2_verify.yaml", "hillclimb/v2_verify.yaml"),
                     ("jax_reward", "examples/jax_reward.yaml")):
        assert _find_config(ref) == CONFIG_ROOT / rel, ref


def test_every_shipped_config_is_reachable_by_its_bare_name():
    """No two shipped configs share a name, so every bare name is unambiguous and
    names the file `--list-configs` shows for it. A second `eureka.yaml` in any
    directory would make `-c eureka` an error (or, at the top of `configs/`,
    silently shadow the method), and it fails here first. Underscore-prefixed
    directories (`_profiles/`) hold execution profiles, not configs, and are
    outside the name space."""
    seen = 0
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        rel = path.relative_to(CONFIG_ROOT)
        if path.name.startswith("_") or any(p.startswith("_") for p in rel.parent.parts):
            continue
        assert _find_config(path.stem) == path, rel
        seen += 1
    assert seen >= len(PAPER_CONFIGS) + 24, "the walk stopped matching the tree"


def test_list_configs_prints_every_config_once_as_something_config_accepts():
    """`bird.py --list-configs | grep -v '^#'` is a list of `--config` arguments:
    each line resolves, and together they name every shipped config exactly once
    (the ERA recipes, `methods/`, `hillclimb/`, `examples/`)."""
    import subprocess
    import sys

    from conftest import REPO
    proc = subprocess.run([sys.executable, str(REPO / "bird.py"), "--list-configs"],
                          cwd=REPO, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    refs = [ln for ln in proc.stdout.splitlines() if not ln.startswith("#")]
    assert all(refs), "a blank line would be an empty --config argument"
    listed = [_find_config(r) for r in refs]
    shipped = sorted(p for p in CONFIG_ROOT.rglob("*.yaml")
                     if not any(part.startswith("_") for part in p.relative_to(CONFIG_ROOT).parts)
                     and "sweeps" not in p.relative_to(CONFIG_ROOT).parts)
    assert sorted(listed) == shipped


def test_an_ambiguous_bare_name_is_refused_and_lists_every_match(tmp_path, monkeypatch):
    """Two files with one name: the bare name is an error naming both, never a
    silent pick -- a name that resolved to whichever file a directory walk met
    first would change the method being run when someone added a file elsewhere.
    Each file stays reachable by its path, an exact path under `configs/` wins
    over the name search, and a copy inside an underscore-prefixed directory is
    not a second match.

    THE PROBE LIVES UNDER `tmp_path`, NEVER IN THE CHECKOUT: a file written into
    `configs/` is visible to every other test that walks the tree (a parallel
    shard can list it and then `load()` it after it is gone). So the whole tree
    is copied and `_config_root` -- the single resolver behind `CONFIG_ROOT`,
    `_find_config`, `_resolve_extends` and the defaults read -- is pointed at the
    copy."""
    import shutil
    import bird.config as config_module

    root = tmp_path / "configs"
    shutil.copytree(CONFIG_ROOT, root)
    monkeypatch.setattr(config_module, "_config_root", lambda: root)

    def probe(rel, name):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"extends: {'../' * (len(path.relative_to(root).parts) - 1)}"
                        f"methods/zeroshot.yaml\nname: {name}\n")

    probe("methods/probe_twice.yaml", "in_methods")
    probe("hillclimb/probe_twice.yaml", "in_hillclimb")
    with pytest.raises(ConfigError, match="ambiguous") as exc:
        load("probe_twice")
    assert "configs/methods/probe_twice.yaml" in str(exc.value)
    assert "configs/hillclimb/probe_twice.yaml" in str(exc.value)
    assert load("methods/probe_twice")["name"] == "in_methods"
    assert load("hillclimb/probe_twice.yaml")["name"] == "in_hillclimb"

    probe("probe_top.yaml", "at_top")
    probe("hillclimb/probe_top.yaml", "nested")
    assert load("probe_top")["name"] == "at_top", "an exact relative path comes first"

    probe("_scratch/probe_once.yaml", "hidden")
    probe("examples/probe_once.yaml", "listed")
    assert load("probe_once")["name"] == "listed"

    assert not (CONFIG_ROOT / "methods" / "probe_twice.yaml").exists()


def test_a_config_that_does_not_exist_still_raises():
    """The name search must not turn a typo into a silent miss of a different kind."""
    with pytest.raises(ConfigError, match="config not found"):
        load("eureka_no_evolutionn")
    with pytest.raises(ConfigError, match="config not found"):
        load("hillclimb/eureka")
