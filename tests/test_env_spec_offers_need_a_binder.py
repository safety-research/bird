"""An `env_spec` that OFFERS identifiers must have something that binds them.

`pythonic_class_abstraction` and `state_action_api_stub` render the state as
code -- typed class attributes, `@dataclass` fields, `def helper(s): ...` --
under headings that say the model may use them. The `## OUTPUT CONTRACT` then
pins a free function whose `state` is an ndarray, and
`training._reward_namespace` binds `{np, numpy, math}` and nothing else. With
no `generate.postprocess.symbol_mapping` to rewrite them, every name the prompt
offers is a name the candidate cannot use.

WHAT THIS COSTS, because the refusal is only worth its weight if the reader knows
why it exists: without a binder, every candidate that writes a rendered name
such as `state.x_velocity` fails, and the failure can read as `status: ok` with
every candidate at the failure value. Nothing but a live run would show it.

THE LIMIT OF THIS RULE, stated here rather than discovered later. A non-empty
mapping satisfies it, and a mapping can still miss nearly everything: the four
Text2Reward-lineage points carry a five-entry table that overlaps their eleven
rendered names by exactly ONE. Coverage is a property of (adapter, member) and
`_check_coherence` has no env to ask, so it is checked by
`tests/test_env_spec_rendered_resolves.py` instead. This file covers the
cheap half: the no-mapping-at-all case is refused at load.
"""

import pytest

from bird.config import load

#: Kept in step with `config._OFFERS_IDENTIFIERS` by
#: `test_the_tuple_here_matches_the_one_the_rule_uses` below -- two copies of one fact,
#: which drift when a registry member is added or a config moves onto a new member,
#: and the mismatch otherwise surfaces somewhere unrelated, e.g. as a red
#: `test_a_mapping_satisfies_the_rule` -- a test about one config, naming neither the
#: member nor the tuple.
#:
#: `t2r_class_abstraction` is NOT here, and that is deliberate rather than an omission:
#: it carries a stricter COVERAGE rule instead (see `config._OFFERS_IDENTIFIERS`).
#: These two are the members whose rendered symbols depend on the ADAPTER and so are
#: unknowable at load, which is why presence is all this rule can ask.
OFFERS_IDENTIFIERS = ("pythonic_class_abstraction", "state_action_api_stub")

#: Published points exempt from the rule, pending repair. Enumerated in
#: the config module so a FOURTH cannot join them silently, and asserted here in
#: both directions so the list shrinks visibly rather than rotting.
EXEMPT = ("gt", "limen", "limen_reward_only")


@pytest.mark.parametrize("member", OFFERS_IDENTIFIERS)
def test_an_offering_env_spec_without_a_mapping_is_refused_at_load(member):
    """The exact configuration that wastes a run, refused before it can run."""
    with pytest.raises(Exception) as exc:
        load("revolve", overrides={"generate.context.env_spec": member})
    message = str(exc.value)
    assert "generate.context.env_spec" in message, message
    assert "symbol_mapping" in message, message
    # The message must name a way out, not just say no.
    assert "natural_language_only" in message, message


def test_a_mapping_satisfies_the_rule():
    """`card` carries `per_task`, so it loads -- the rule is about ABSENCE.

    Deliberately a published config rather than an override: if this rule ever
    starts refusing a point that ships a mapping, that is a regression in the
    rule and not in the config.
    """
    # `text2reward_zeroshot`, NOT `card`: CARD takes `t2r_class_abstraction` -- a
    # member this rule deliberately does not own -- so asserting on it would fail
    # for a reason that has nothing to do with what this checks. The point is that a
    # published point taking one of THESE members carries a table, and the T2R pair
    # is where that is true.
    cfg = load("text2reward_zeroshot")
    assert cfg.get("generate.context.env_spec") in OFFERS_IDENTIFIERS
    assert cfg.get("generate.postprocess.symbol_mapping")


@pytest.mark.parametrize("name", EXEMPT)
def test_the_named_legacy_exemptions_still_load(name):
    cfg = load(name)
    assert cfg.get("generate.context.env_spec") in OFFERS_IDENTIFIERS
    assert not cfg.get("generate.postprocess.symbol_mapping")


def test_revolve_ships_a_member_that_offers_no_unusable_identifier():
    """REvolve's own release reads ONE environment text for every arm.

    `refs/code/Revolve/main.py:103-106` loads `prompts/env_input` -- the verbatim
    Gymnasium `HumanoidEnv` docstring, an index table -- before the generation
    loop; `cfg.evolution.baseline` selects the operator prompt at `:177` and
    never the environment text. So REvolve, REvolve Auto, Eureka and Eureka Auto
    all saw the same table, which is why the published rewards index
    `observation[22]` (`refs/tex/revolve/main.tex:2459`, `:2492`).
    """
    assert load("revolve").get("generate.context.env_spec") == "natural_language_only"


def test_the_exemption_list_is_exactly_the_configs_that_need_it():
    """Both directions: a new offender cannot hide, and a fixed one cannot linger."""
    import inspect

    from bird.config import _check_coherence  # noqa: PLC0415 -- the list lives there
    src = inspect.getsource(_check_coherence)
    assert '_UNGUARDED_LEGACY = ("gt", "limen", "limen_reward_only")' in src, (
        "the exemption list in bird/config.py has moved or changed; this file's "
        "EXEMPT must follow it, and every entry must still need the exemption")
    # Collected rather than asserted in the loop: this test's claim is about the
    # LIST as a whole, so a reader needs every stale entry at once, not the
    # first one. The sibling per-config claims
    # are parametrised instead, which is the right shape for a per-config fact.
    stale = []
    for name in EXEMPT:
        cfg = load(name)
        if not (cfg.get("generate.context.env_spec") in OFFERS_IDENTIFIERS
                and not cfg.get("generate.postprocess.symbol_mapping")):
            stale.append(name)
    assert not stale, (
        f"exempted but no longer tripping the rule: {stale} -- remove each from "
        f"_UNGUARDED_LEGACY in bird/config.py and from EXEMPT here")


def test_the_exemption_survives_a_name_override_and_cannot_be_assumed_by_one():
    """Keyed on the config FILE, not on `name`.

    `name` is an ordinary overridable key. The published GT point is `gt.yaml`
    loaded with `name: gt_reward_design`
    (`tests/conftest.py::GT_PUBLISHED_OVERRIDES`), so exempting by `name` would
    refuse a published point, which `tests/test_ablation.py` catches. The same
    looseness runs the other way: any caller could write `name: gt` and walk
    through. `cfg.source` is the
    file the point came from and no override moves it.
    """
    from conftest import load_gt_published  # noqa: PLC0415 -- test-only helper

    # Forward: the real published GT point still loads, under its own name.
    cfg = load_gt_published()
    assert cfg["name"] == "gt_reward_design"

    # Backward: borrowing an exempted NAME does not borrow its exemption.
    with pytest.raises(Exception) as exc:
        load("revolve", overrides={"name": "gt",
                                   "generate.context.env_spec": "pythonic_class_abstraction"})
    assert "symbol_mapping" in str(exc.value)


@pytest.mark.parametrize("name", EXEMPT)
def test_the_exemption_survives_a_source_whose_basename_has_no_yaml_suffix(name):
    """The guarded slice, held to the behaviour rather than to its own spelling.

    `_check_coherence` keys the exemption on the config FILE, and strips the
    extension with

        _src[:-len(".yaml")] if _src.endswith(".yaml") else _src

    rather than `str.removesuffix(".yaml")` -- the same guarded-slice
    spelling `bird/tasks.py` uses for its `-v0` strip, and the one this test
    pins by behaviour.

    THE GUARD IS THE POINT, and the obvious simplification breaks it. Measured
    on the three exempt names with the `endswith` dropped:

        basename             guarded              UNguarded `[:-5]`
        gt                   gt                   ''
        limen                limen                ''
        limen_reward_only    limen_reward_only    'limen_reward'

    So an unguarded slice does not merely truncate to empty -- on
    `limen_reward_only` it yields a DIFFERENT, plausible-looking name, and the
    exemption is dropped for every source whose basename lacks the extension.
    A `--print-config` of a published point would then be refused for a rule it
    is exempt from, and the failure would name the env_spec rather than the
    slice.

    Asserted on BEHAVIOUR (the exemption still holds) rather than by comparing
    the two expressions, because an equivalence test between the guarded slice
    and `removesuffix` is true for every input by construction and so could
    never fail. This one fails on the simplification, which is the only edit
    anyone is likely to make here.
    """
    from bird.config import _check_coherence

    cfg = load(name)
    cfg.source = f"/somewhere/else/{name}"                # no .yaml suffix
    offending = [p for p in _check_coherence(cfg)
                 if "symbol_mapping is empty" in p]
    assert not offending, (
        f"{name}: the exemption was lost when the source basename carried no "
        f"'.yaml' suffix, which is what an unguarded `[:-len('.yaml')]` does -- "
        f"it would strip five characters from {name!r} and compare the wrong "
        f"string against the exemption list. Keep the `endswith` guard.")


def test_the_tuple_here_matches_the_one_the_rule_uses():
    """The two copies of this list must not drift.

    `OFFERS_IDENTIFIERS` above and `config._OFFERS_IDENTIFIERS` are the same fact
    written twice. When a member is added to the registry and a config moves onto
    it without the rule being widened, what goes red otherwise is an unrelated
    test such as `test_a_mapping_satisfies_the_rule` -- naming neither the member
    nor the tuple. This asserts the identity
    directly so the next member says so in one line.
    """
    import bird.config as cfgmod
    import ast, inspect
    src = inspect.getsource(cfgmod._check_coherence)
    for node in ast.walk(ast.parse(src.lstrip())):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_OFFERS_IDENTIFIERS"):
            assert tuple(ast.literal_eval(node.value)) == OFFERS_IDENTIFIERS
            return
    raise AssertionError("_OFFERS_IDENTIFIERS not found in _check_coherence; if it "
                         "moved, move this assertion with it rather than deleting it")


def test_every_registered_env_spec_member_is_classified():
    """A new registry member must be decided about, not defaulted into silence.

    The rule's blind spot is a member nobody thought about: it is simply not in the
    tuple, so it renders identifiers and is never refused. Enumerating the whole
    family here means adding one to the registry fails this until someone puts it on
    one side or the other -- which is the same argument `KINDS` makes for the schema.
    """
    import bird.registry as registry
    registry.load_all()
    #: Members that render real environment SOURCE or no source at all. Their names
    #: are the env's own, so the open question is mapping COVERAGE, not binding.
    RENDERS_SOURCE_OR_NOTHING = ("full_source", "natural_language_only", "none")
    #: Members with a STRICTER rule elsewhere, so this one deliberately skips them.
    #: `t2r_class_abstraction`'s prompt is a published constant, so its symbol list is
    #: knowable at load and the load checks table COVERAGE rather than mere presence.
    GUARDED_BY_THE_COVERAGE_RULE = ("t2r_class_abstraction",)
    known = (set(OFFERS_IDENTIFIERS) | set(RENDERS_SOURCE_OR_NOTHING)
             | set(GUARDED_BY_THE_COVERAGE_RULE))
    registered = set(registry.names("env_spec"))
    unclassified = sorted(registered - known)
    assert not unclassified, (
        f"env_spec member(s) {unclassified} are registered but classified neither as "
        f"offering identifiers (OFFERS_IDENTIFIERS, and add them to "
        f"`config._OFFERS_IDENTIFIERS` too), nor as carrying the stricter coverage "
        f"rule (GUARDED_BY_THE_COVERAGE_RULE + ENV_SPEC_REQUIRED_SYMBOLS), nor as "
        f"rendering source/nothing. A member in none of the three is silently exempt "
        f"from every binding rule -- which is not a wrong classification but an "
        f"ABSENT one, and is the hole this test exists to close.")
    stale = sorted(known - registered)
    assert not stale, f"classified but no longer registered: {stale}"


def test_the_coverage_rule_really_owns_the_member_this_file_defers_to():
    """The deferral must be to a rule that exists, not to a believed one.

    `OFFERS_IDENTIFIERS` leaves `t2r_class_abstraction` out because the coverage
    check (`ENV_SPEC_REQUIRED_SYMBOLS`) owns it. If that check ever stops listing the member, the deferral becomes
    an exemption and the member is guarded by nothing -- silently, and in the
    direction this whole file exists to prevent.
    """
    from bird.components.generation import ENV_SPEC_REQUIRED_SYMBOLS
    assert ENV_SPEC_REQUIRED_SYMBOLS.get("t2r_class_abstraction"), (
        "t2r_class_abstraction is excluded from OFFERS_IDENTIFIERS on the grounds that "
        "ENV_SPEC_REQUIRED_SYMBOLS covers it, and it no longer does: either restore it "
        "there or add the member to OFFERS_IDENTIFIERS and to "
        "`config._OFFERS_IDENTIFIERS`")
