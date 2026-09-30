"""`generate.context.env_spec: t2r_class_abstraction` must send T2R's/CARD's bytes.

WHY A PROMPT-TEXT PIN IS NOT A REFLEX TEST, on the same argument
`tests/test_reward_recipe.py` makes for the sibling key and for the same class of
failure. Configuring CARD's environment representation as
`pythonic_class_abstraction` would be wrong: on Meta-World that member renders the
ADAPTER's tables -- 39 documented observation slots plus the eight inlineable
expressions of `bird/envs/metaworld.py::_HELPERS`, two of which are Meta-World's
own `tolerance()` and `hamacher_product()`, the primitives the shipped v2 reward is
written out of. Every counter would read normal: the key read, the section
rendered, the run completed, and the generator shown the reference reward's
building blocks by name. Nothing but the prompt bytes can witness that -- a
config value that is read, but read into something other than what it claims.

THE PUBLISHED TEXT IS A SECOND COPY HERE, ON PURPOSE. `refs/` is fetched
separately (`scripts/fetch_refs.sh`) and is absent in CI, so it cannot be the
oracle there, and a
disagreement between this copy and the constant is exactly the failure to raise.
One test below does compare against the vendored file and skips when `refs/` is
absent; that is the belt, this is the braces.

Anyone re-deriving the text must use the RELEASE file
(`refs/code/text2reward/code_generation/single_flow/classlike_prompt/MetaworldPrompt.py:14-26`),
keep `gym.env` lower-case exactly as upstream wrote it, keep the `**if any**`
markdown on obj2, and keep the blank line between each class.
"""

import importlib.util
import json
import re

import pytest
from conftest import REPO

from bird.components import generation
from bird.config import ConfigError, load

KEY = "generate.context.env_spec"
TABLE = "generate.postprocess.symbol_mapping"

# refs/code/text2reward/code_generation/single_flow/classlike_prompt/MetaworldPrompt.py:14-26
# = refs/code/CARD/code_generation/self_reflection/benchmark_prompt/metaworld_prompt.py:12-24
# = refs/tex/card/main.tex:734-746 (App. C, tab:metaworld_system_prompt)
T2R_METAWORLD_ABSTRACTION = """class BaseEnv(gym.env):
    self.robot : Robot # the robot in the environment
    self.obj1 : RigidObject # the first object in the environment
    self.obj2 : RigidObject # the second object in the environment, **if any**
    self.goal_position : np.ndarray[(3,)] # indicate the 3D position of the goal

class Robot:
    self.ee_position : np.ndarray[(3,)] # indicate the 3D position of the end-effector
    self.gripper_openness : float # a normalized measurement of how open the gripper is, range in [-1, 1]

class RigidObject:
    self.position : np.ndarray[(3,)] # indicate the 3D position of the rigid object
    self.quaternion : np.ndarray[(4,)] # indicate the quaternion of the rigid object"""

#: T2R's Meta-World general->specific table, `metaworld_exp.py:19-27`. The KEYS
#: are the assertion; the right-hand sides are checked separately (see
#: `test_the_right_hand_sides_are_upstreams_and_only_the_goal_deviates`).
T2R_MAPPING_KEYS = (
    "self.robot.ee_position",
    "self.robot.gripper_openness",
    "self.obj1.position",
    "self.obj1.quaternion",
    "self.obj2.position",
    "self.obj2.quaternion",
    "self.goal_position",
)

#: Two names that occur in the Meta-World adapter's `pythonic_class_abstraction`
#: rendering and in no published prompt. They are Meta-World's own shaping
#: kernels, so their presence in a generation prompt is the specific leak this
#: member exists to close.
KERNELS = ("hamacher_product", "tolerance(")


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def _user_messages(cfg, tmp_path):
    """Every user-role message of every prompt.json the run wrote, first prompt first."""
    _entry().run(cfg, out_root=str(tmp_path))
    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    prompts = sorted(p for p in (run / "candidates").glob("*/prompt.json"))
    assert prompts, "the run wrote no prompt.json"
    out = []
    for p in prompts:
        msgs = json.loads(p.read_text())
        out.append(next(m["content"] for m in reversed(msgs) if m["role"] == "user"))
    return out


# --------------------------------------------------------------------------
# the text
# --------------------------------------------------------------------------


def test_the_constant_is_the_published_text():
    """The bytes, against this file's independent copy.

    Old code fails on import: there is no `_T2R_METAWORLD_CLASS_ABSTRACTION`."""
    assert generation._T2R_METAWORLD_CLASS_ABSTRACTION == T2R_METAWORLD_ABSTRACTION


def test_the_constant_matches_the_vendored_release_file():
    """The other oracle, when it is present.

    This is what makes the hard-coded copy above a CHECK rather than a
    restatement: the two are derived independently and either can refute the
    other. Skips where `refs/` is absent (`scripts/fetch_refs.sh` fetches
    it)."""
    src = (REPO / "refs" / "code" / "text2reward" / "code_generation" / "single_flow"
           / "classlike_prompt" / "MetaworldPrompt.py")
    if not src.exists():
        pytest.skip("refs/ not checked out")
    lines = src.read_text().splitlines()
    # 14-26 ONE-INDEXED, and the range is TIGHT -- both halves asserted, because
    # 14-27 is the natural misreading. 27 is the blank line after the block, so a range ending
    # there is a superset by exactly the separator, which is the same
    # off-by-one `test_reward_recipe.py` records for `main.tex:728-731`.
    # A locator in a comment is prose; this makes it a claim the file refutes.
    assert "\n".join(lines[13:26]) == T2R_METAWORLD_ABSTRACTION
    assert lines[12] == "", "line 13 should be the blank before the block"
    assert lines[26] == "", "line 27 should be the blank after the block"
    assert lines[27].startswith("You are allowed to use any existing python package")


def test_the_two_sibling_locators_name_exactly_the_same_block():
    """CARD's release copy and CARD's appendix, on the same tight-range rule.

    Three citations travel with this constant and all three were checkable, so
    all three are checked: an unverified locator beside two verified ones is the
    one a future reader will trust and the one that will be wrong. Same refs/
    skip as above."""
    card_py = (REPO / "refs" / "code" / "CARD" / "code_generation" / "self_reflection"
               / "benchmark_prompt" / "metaworld_prompt.py")
    tex = REPO / "refs" / "tex" / "card" / "main.tex"
    if not card_py.exists() or not tex.exists():
        pytest.skip("refs/ not checked out")

    # metaworld_prompt.py:12-24 -- CARD's copy of T2R's block, character-identical.
    py_lines = card_py.read_text().splitlines()
    assert "\n".join(py_lines[11:24]) == T2R_METAWORLD_ABSTRACTION
    assert py_lines[10] == "" and py_lines[24] == ""

    # main.tex:734-746 -- App. C, tab:metaworld_system_prompt. The paper prints
    # the block too, so paper and both releases agree character for character
    # and the constant carries no provenance marker.
    tex_lines = tex.read_text().splitlines()
    assert "\n".join(tex_lines[733:746]) == T2R_METAWORLD_ABSTRACTION
    assert tex_lines[732] == "" and tex_lines[746] == ""


def test_the_text_declares_no_helper_methods_and_no_observation_table():
    """The property that distinguishes it, asserted structurally rather than by eye.

    Three classes, eight attribute lines, no `def`, and none of Meta-World's
    kernels. `pythonic_class_abstraction` fails every clause of this on
    Meta-World, which is why it is a separate member and not an edit."""
    text = generation._T2R_METAWORLD_CLASS_ABSTRACTION
    assert text.count("class ") == 3
    assert [ln.split(":")[0].strip() for ln in text.splitlines() if ln.startswith("    self.")] == [
        "self.robot", "self.obj1", "self.obj2", "self.goal_position",
        "self.ee_position", "self.gripper_openness",
        "self.position", "self.quaternion",
    ]
    assert "def " not in text
    for kernel in KERNELS:
        assert kernel not in text
    # Upstream's own two oddities, kept because verbatim means verbatim: the
    # lower-case `gym.env` (not `gym.Env`), and a prompt that tells the model
    # `gripper_openness` is in [-1, 1] when the slot it is mapped to is
    # Meta-World's `gripper_distance_apart`, clipped to [0, 1]. See
    # `generation._SYMBOLS_T2R_METAWORLD` on why the error is reproduced.
    assert "class BaseEnv(gym.env):" in text
    assert "range in [-1, 1]" in text


# --------------------------------------------------------------------------
# what a run actually sends
# --------------------------------------------------------------------------


def test_card_sends_the_published_abstraction_and_neither_kernel(tmp_path):
    """The end-to-end claim: card's prompts carry the block and not the leak.

    Old code fails -- card resolved `pythonic_class_abstraction`."""
    user = _user_messages(load("card", profile="tester"), tmp_path)[0]
    assert "## ENVIRONMENT\n" + T2R_METAWORLD_ABSTRACTION in user
    for kernel in KERNELS:
        assert kernel not in user


def test_the_member_ignores_the_adapter_however_rich_it_is(tmp_path):
    """A constant, not a rendering -- proven against an adapter that offers more.

    The other members probe the env for a matching attribute
    (`_spec_or_stub`), so what they send grows whenever an adapter's tables
    grow. This one must not, or the leak it closes comes back the next time
    someone documents an observation. Asserted by handing the renderer an env
    that exposes exactly the attribute `pythonic_class_abstraction` would use."""
    from bird.components.generation import env_spec_t2r_class_abstraction

    class _Rich:
        pythonic_class_abstraction = "class Leak:\n    self.everything = 1"
        class_abstraction = pythonic_class_abstraction
        t2r_class_abstraction = "class AlsoALeak:\n    self.smuggled = 1"
        observation_fields = [f"slot_{i}" for i in range(39)]
        helper_methods = ["tolerance(x, bounds, margin)", "hamacher_product(a, b)"]

    class _Ctx:
        env = _Rich()
        cfg = load("card", profile="tester")

    out = env_spec_t2r_class_abstraction(_Ctx(), None)
    assert out == T2R_METAWORLD_ABSTRACTION
    assert "Leak" not in out and "smuggled" not in out and "slot_0" not in out


def test_the_other_member_still_renders_the_adapter():
    """The contrast: this member is a NEW value, not an edit of the other one.

    `pythonic_class_abstraction` keeps its meaning -- it is T2R's ManiSkill2
    shape, whose callables are the documented difference from GT's stub -- and
    on the same rich adapter it renders what the adapter offered. If this ever
    starts returning a constant, the two members have collapsed into one and
    `--diff` stops meaning anything between them."""
    from bird.components.generation import env_spec_pythonic_class_abstraction

    class _Rich:
        pythonic_class_abstraction = "class Env:\n    self.cubeA : RigidObject"

    class _Ctx:
        env = _Rich()
        cfg = load("card", profile="tester")

    out = env_spec_pythonic_class_abstraction(_Ctx(), None)
    assert "self.cubeA" in out
    assert out != T2R_METAWORLD_ABSTRACTION


# --------------------------------------------------------------------------
# the companion table
# --------------------------------------------------------------------------


def test_the_table_carries_exactly_the_names_the_prompt_shows():
    """The pair's invariant: every symbol the model is given is rewritable.

    This is the one that keeps the two halves honest. The prompt shows eight
    attribute lines; seven of them are readable quantities and the eighth
    (`self.obj2`, a bare handle) is reachable only through its own fields. Every
    LHS of the table must occur in the abstraction, and every readable name in
    the abstraction must have an entry -- either direction failing means a
    generated program can name something nothing rewrites, which is a candidate
    the harness destroys and records as the generator's failure."""
    table = generation._SYMBOLS_T2R_METAWORLD
    assert tuple(table) == T2R_MAPPING_KEYS
    text = generation._T2R_METAWORLD_CLASS_ABSTRACTION
    for name in table:
        leaf = name.rsplit(".", 1)[-1]
        assert f"self.{leaf}" in text, f"{name} maps a symbol the prompt never shows"
    # The reverse: the prompt's readable leaves, with the two typed handles
    # (`robot`, and the obj1/obj2 pair) excluded because they are namespaces.
    leaves = {ln.split(":")[0].strip().removeprefix("self.")
              for ln in text.splitlines() if ln.startswith("    self.")}
    assert leaves - {"robot", "obj1", "obj2"} == {
        "goal_position", "ee_position", "gripper_openness", "position", "quaternion"}


def test_the_right_hand_sides_are_upstreams_and_only_the_goal_deviates():
    """Six of seven values byte-identical to the release; the seventh is forced.

    Writing them as `s[...]`, on the reasoning that `s` is BIRD's own
    convention, would be wrong in the way this whole seam can be wrong: a
    substituted value has to be GRAMMATICAL IN THE FUNCTION THE CONTRACT ASKS
    FOR. `generate.output.signature` pins a signature whose observation
    parameter is `obs`, so `s[4:7]` would be a NameError on every candidate.

    Only the goal deviates, and it cannot not: upstream calls
    `self.env._get_pos_goal()` on the wrapped env, and a free reward function has
    no env to call -- `_make_binder` passes None for a `self` parameter,
    deliberately. `bird/envs/metaworld.py`'s state table documents those three
    slots as that method's value for the episode."""
    table = generation._SYMBOLS_T2R_METAWORLD
    assert table == {
        "self.robot.ee_position": "obs[:3]",
        "self.robot.gripper_openness": "obs[3]",
        "self.obj1.position": "obs[4:7]",
        "self.obj1.quaternion": "obs[7:11]",
        "self.obj2.position": "obs[11:14]",
        "self.obj2.quaternion": "obs[14:18]",
        "self.goal_position": "obs[36:39]",
    }
    # And the six are the RELEASE's bytes, read out of `refs/` rather than
    # restated -- the deviation is one entry and this says which.
    exp = (REPO / "refs" / "code" / "text2reward" / "code_generation" / "single_flow"
           / "metaworld_exp.py")
    if exp.exists():
        import ast as _ast
        released = None
        for node in _ast.parse(exp.read_text()).body:
            if isinstance(node, _ast.Assign) and any(
                    getattr(t, "id", "") == "mapping_dicts" for t in node.targets):
                released = _ast.literal_eval(node.value)
        assert released is not None, "metaworld_exp.py:19-27 no longer defines mapping_dicts"
        assert set(released) == set(table)
        deviating = [k for k in released if released[k] != table[k]]
        assert deviating == ["self.goal_position"], deviating
        assert released["self.goal_position"] == "self.env._get_pos_goal()"
    # No RHS may reach for a live env: a BIRD reward program is handed `s` and
    # `a`, and `_make_binder` passes None for `self` (training.py), so a
    # `self.env.<anything>` survivor is an AttributeError at training time
    # wearing a reward's name.
    for value in table.values():
        assert "self." not in value and "(" not in value


def test_the_table_is_registered_and_resolves_through_the_registry():
    """A config value is a registry key, including this one.

    Old code fails: the string form resolved by `getattr(ctx.env,
    "symbol_mapping")` and there was no `symbol_table` family, so a table this
    repo authors had nowhere to live but an adapter it does not belong to."""
    from bird import registry

    registry.load_all()
    assert "symbol_table" in registry.KINDS
    assert set(registry.names("symbol_table")) == {"per_task", "t2r_metaworld_global"}

    class _Ctx:
        env = None
        cfg = load("card", profile="tester")

    assert generation._symbol_mapping(_Ctx()) == generation._SYMBOLS_T2R_METAWORLD


def test_an_inline_dict_still_works():
    """The other half of the key's type, unchanged.

    `(dict, S)` is why this is not a `kind=` field, so the dict path has to keep
    working or the schema note is wrong about itself.

    The dict has to be COMPLETE, and that is the coverage rule doing its job on
    this very test: a single-entry dict is (rightly) refused -- one symbol
    covered of seven shown is the partial case that produces a garbled
    program."""
    # `obs[...]`, not `s[...]`: an inline dict is refused just as the named table
    # would be if its values named a variable the pinned signature does not
    # declare. The dict path is the key's other half, not an exemption from the
    # seam.
    table = {
        "self.robot.ee_position": "obs[:3]",
        "self.robot.gripper_openness": "obs[3]",
        "self.obj1.position": "obs[4:7]",
        "self.obj1.quaternion": "obs[7:11]",
        "self.obj2.position": "obs[11:14]",
        "self.obj2.quaternion": "obs[14:18]",
        "self.goal_position": "obs[36:39]",
    }

    class _Ctx:
        env = None
        cfg = load("card", profile="tester", overrides={TABLE: dict(table)})

    assert generation._symbol_mapping(_Ctx()) == table


# --------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------


def test_per_task_is_refused_beside_the_abstraction():
    """The hole a presence check leaves, and the reason the rule is COVERAGE.

    `t2r_class_abstraction` + `symbol_mapping: per_task` passes a presence
    check -- a table exists -- and is exactly the run the check exists to
    stop: `per_task` resolves to the TASK
    SPEC's block, which is BIRD's own vocabulary, so none of the seven symbols
    the prompt shows is covered. The refusal names them, because the question a
    reader hits is "which one did I miss"."""
    with pytest.raises(ConfigError, match="does not cover") as exc:
        load("card", overrides={TABLE: "per_task"})
    msg = str(exc.value)
    for name in T2R_MAPPING_KEYS:
        assert name in msg, f"the refusal does not name {name}"


def test_a_table_missing_some_symbols_is_refused_and_names_exactly_those():
    """Partial coverage is refused, and only the uncovered names are listed.

    An inline dict is the shape where partial coverage is most plausible --
    somebody hand-writing the table and stopping early -- and a message that
    said "does not cover" without saying which would send them back to count."""
    partial = {"self.robot.ee_position": "obs[:3]", "self.goal_position": "obs[36:39]"}
    with pytest.raises(ConfigError, match="does not cover") as exc:
        load("card", overrides={TABLE: partial})
    msg = str(exc.value)
    for name in partial:
        assert name not in msg.split("shows the model:")[1], \
            f"{name} IS covered and must not be listed as missing"
    for name in set(T2R_MAPPING_KEYS) - set(partial):
        assert name in msg


def test_a_suffix_key_rewrites_one_of_the_symbols_PARTIALLY():
    """The regression that makes the rule about garbling and not only absence.

    `_apply_symbol_mapping` substitutes on WORD BOUNDARIES and `.` is not a word
    character, so the task spec's key `goal_position` matches the tail of
    `self.goal_position` and rewrites it to `self.s[36:39]` -- an attribute
    access on the state array, `AttributeError` at stage 2, attributed to the
    model. Measured against the real function; this pins it so that if the
    coverage rule is ever relaxed, the thing it protects against is still on the
    page.

    Asserted on the FUNCTION rather than through a run, because the point is the
    substitution's behaviour and not any one config's resolution."""
    from bird.components.generation import _apply_symbol_mapping

    spec_table = {"goal_position": "s[36:39]", "object_position": "s[4:7]"}
    out = _apply_symbol_mapping(spec_table, "d = norm(self.obj1.position - self.goal_position)")
    assert "self.s[36:39]" in out, (
        "the suffix collision no longer reproduces -- if _apply_symbol_mapping was "
        "made boundary-aware across dots, say so here and in the coherence rule")
    # And the six that are not suffixes of a spec key pass through untouched,
    # which is what makes the garbled program plausible rather than obviously broken.
    assert "self.obj1.position" in out


def test_the_abstraction_without_a_table_is_refused():
    """The pair is the method, so half of it is a config error at load.

    Without this the run reaches the LLM, spends a call, and every candidate
    dies in stage 2 naming an attribute of nothing -- a 100%-invalid run that
    reads as the model's fault."""
    with pytest.raises(ConfigError, match="t2r_class_abstraction"):
        load("card", overrides={TABLE: None})


def test_an_unregistered_table_name_is_refused_with_the_allowed_values():
    """The enum path a `kind=` field would have given, done in _check_coherence.

    Accepting `per_taskk` and warning at run time that the env exposes no
    table would leave every symbol unmapped -- the silent version of the
    failure above."""
    with pytest.raises(ConfigError, match="t2r_metaworld_global"):
        load("card", overrides={TABLE: "per_taskk"})


def test_the_other_members_do_not_require_a_table():
    """The COVERAGE refusal is scoped to the one member whose symbols are knowable.

    `full_source` and the rest render names the adapter defines, so this rule asks
    nothing of them and must go on asking nothing -- eureka pins no table at all.

    THE SECOND CASE KEEPS A TABLE, and the reason is worth reading before changing
    it. `load("card", {KEY: "pythonic_class_abstraction", TABLE: None})` -- a member
    with NO table -- would demonstrate this rule's scoping only by relying on nothing
    else refusing that pair, and a second, independent rule does refuse it:
    `config._OFFERS_IDENTIFIERS`, a PRESENCE check over `pythonic_class_abstraction`
    and `state_action_api_stub`, whose symbols depend on the adapter and so cannot be
    coverage-checked at load. Without a table, candidates write identifiers such as
    `state.x_velocity` against a namespace binding only `{np, numpy, math}`, and
    none of them trains. Keeping the table makes
    the case test what the docstring says: the coverage rule does not demand T2R's
    seven symbols of a DIFFERENT member.

    THE TABLE IS DELIBERATELY A NON-COVERING ONE, and this is the half a later
    reader will want to "simplify" away. `card` INHERITS `t2r_metaworld_global`,
    which supplies exactly the seven symbols this rule checks -- so a case using the
    inherited table is satisfied even by a coverage rule that has LEAKED onto
    `pythonic_class_abstraction`, and cannot fail for the reason named above.
    Measured by mutation: with
    `"pythonic_class_abstraction": tuple(_SYMBOLS_T2R_METAWORLD)` added to
    `ENV_SPEC_REQUIRED_SYMBOLS`, the inherited-table form still passes and this form
    goes red with "does not cover 7 of the 7 symbols". An INLINE dict rather than a
    registered member because both registered tables cover the seven by
    construction, so neither can express "a table that is not this rule's table"."""
    assert load("eureka")[TABLE] is None
    cfg = load("card", overrides={KEY: "pythonic_class_abstraction",
                                  TABLE: {"handle_pos": "obs[4:7]"}})
    assert cfg[KEY] == "pythonic_class_abstraction"
    assert cfg[TABLE] == {"handle_pos": "obs[4:7]"}


# --------------------------------------------------------------------------
# the Meta-World half: what `pythonic_class_abstraction` actually renders
# --------------------------------------------------------------------------


def test_on_metaworld_the_old_member_renders_the_kernels_and_the_new_one_does_not():
    """The measurement behind this member, on the real adapter.

    `importorskip`s, so the default test install skips it and reports green --
    run `tests/test_t2r_class_abstraction.py` by hand with `--extra metaworld`.
    It is the only test here that can witness the SIZE of what CARD's prompt
    carried: the count of observation slots, and `tolerance()` /
    `hamacher_product()` by name."""
    pytest.importorskip("metaworld")
    from bird.components.generation import (env_spec_pythonic_class_abstraction,
                                            env_spec_t2r_class_abstraction)
    from bird import registry

    registry.load_all()

    class _Ctx:
        cfg = load("card", profile="tester",
                   overrides={"problem.env_id": "mt50_door-unlock-v3"})

    ctx = _Ctx()
    ctx.env = registry.get("env", "mt50_door-unlock-v3")(ctx)

    old = env_spec_pythonic_class_abstraction(ctx, None)
    new = env_spec_t2r_class_abstraction(ctx, None)

    assert new == T2R_METAWORLD_ABSTRACTION
    for kernel in KERNELS:
        assert kernel in old, "pythonic_class_abstraction stopped rendering the kernels"
        assert kernel not in new
    # The size gap, stated as numbers so a future reader does not have to trust
    # the adjective. NOT counted on "self." -- the adapter's rendering does not
    # use that spelling (it lists `    <slot>: <doc>` lines, so a count on
    # "self." reads 0 while the kernel checks above pass). Counted on what the
    # rendering actually contains: the
    # adapter's own observation slot names, and the bytes.
    for slot in ("hand_x", "gripper_opening", "object2_qw", "goal_z"):
        assert slot in old, f"the adapter stopped rendering {slot}"
        assert slot not in new
    # The crispest form of the difference: `_render_class_abstraction`
    # (`bird/envs/base.py`) emits `def <sig>: ...` for every `_helpers` row, so
    # `pythonic_class_abstraction` literally declares callable methods on Meta-World where
    # the published prompt declares none.
    assert "def " in old and "def " not in new
    assert len(old) > 3 * len(new)


# --------------------------------------------------------------------------
# generate.output.signature -- the seam between the abstraction and the contract
# --------------------------------------------------------------------------
#
# THE DEFECT THESE THREE TESTS ARE WRITTEN FROM, so a reader knows what they are
# for. The abstraction shows attributes on `self`; the table rewrites exactly
# those into `obs[...]`. Both halves are claims about the FUNCTION'S PARAMETERS,
# so they must be pinned: under the default signature
# `compute_reward(state, action=None, next_state=None)`, which declares neither
# `self` nor `obs`, the prompt asks for a function in which its own symbols do
# not resolve, and the table rewrites them into a variable that does not exist.
# A model that obeys every instruction and uses only the listed attributes has to
# bridge the gap itself, e.g. --
#
#     env = state[0]
#     handle_position = env.obj1.position
#
# -- which matches no `self.`-keyed entry, so nothing is rewritten and the
# program reaches an attribute of a float: every repair attempt fails the same
# way until `VerificationAborted: repair attempts exhausted`, and no training
# step is reached.


def test_the_pinned_signature_is_the_release_line_for_both_configs():
    """(a) The rendered OUTPUT CONTRACT line == the release's own line.

    Two configs, two different published lines, each read out of `refs/` at its
    locator rather than restated here -- the same arrangement the abstraction's
    own provenance tests use, and refs-gated for the same reason.

    The two lines are deliberately NOT normalised into one. T2R's carries no
    annotations, returns `-> float`, and has NO TRAILING COLON because upstream
    quotes it inline in prose; CARD's is a code block with `np.ndarray`
    annotations and `Tuple[float, Dict[str, float]]`. A single "cleaned up"
    signature would be a line neither paper sent."""
    from bird.components.generation import _signature_clause

    t2r_src = (REPO / "refs" / "code" / "text2reward" / "code_generation" / "single_flow"
               / "classlike_prompt" / "MetaworldPrompt.py")
    card_src = (REPO / "refs" / "code" / "CARD" / "code_generation" / "self_reflection"
                / "benchmark_prompt" / "metaworld_prompt.py")
    if not (t2r_src.exists() and card_src.exists()):
        pytest.skip("refs/ not checked out")

    t2r_line = t2r_src.read_text().splitlines()[31]
    assert t2r_line.startswith("2. Then write a function that format as")
    quoted = re.findall(r"`([^`]+)`", t2r_line)
    assert len(quoted) == 1, "MetaworldPrompt.py:32 should quote exactly one signature"
    card_line = card_src.read_text().splitlines()[33]

    class _Ctx:
        def __init__(self, cfg):
            self.cfg = cfg

    want = {
        "metaworld_text2reward_zeroshot": quoted[0],
        "card": card_line,
    }
    for name, published in want.items():
        cfg = load(name, profile="tester")
        assert _signature_clause(_Ctx(cfg)).strip() == published, name


def test_a_program_in_the_prompts_vocabulary_maps_compiles_and_runs():
    """(b), the UNGATED half: the map -> compile -> bind -> execute chain works.

    The program here is three lines written in the abstraction's own vocabulary --
    not a vendored upstream file -- so this test runs in every CI job (no `refs/`,
    no `metaworld`, no simulator) and cannot drift from a copy. The
    sibling below does the real six.

    What it pins is the chain the defect broke, end to end: the table rewrites the
    `self.` symbols into `obs[...]`, the published signature declares `obs`, and
    `CompiledReward._make_binder` binds it -- plan `[-1, 1, 0]`, i.e. `self` ->
    None, `action` -> the action, `obs` -> the state. `self` being None is
    harmless BECAUSE the mapping leaves no `self` use behind, and that is the
    property worth asserting: it is what makes a method-shaped published prompt
    legal in a repo whose rewards are free functions."""
    import numpy as np
    from bird.components.generation import (_SYMBOLS_T2R_METAWORLD,
                                            _apply_symbol_mapping)
    from bird.components.training import CompiledReward

    source = (
        "def compute_dense_reward(self, action, obs) -> float:\n"
        "    reach = np.linalg.norm(self.robot.ee_position - self.obj1.position)\n"
        "    place = np.linalg.norm(self.obj1.position - self.goal_position)\n"
        "    return -reach - place - 0.01 * float(self.robot.gripper_openness)\n"
    )
    mapped = _apply_symbol_mapping(_SYMBOLS_T2R_METAWORLD, source)
    assert "self." not in mapped, mapped
    assert "obs[0:3]" in mapped or "obs[:3]" in mapped

    ns = {"np": np}
    exec(compile(mapped, "<mapped>", "exec"), ns)
    reward = CompiledReward(ns["compute_dense_reward"], "compute_dense_reward")
    assert reward._bind == [-1, 1, 0], reward._bind

    rng = np.random.default_rng(20260918)
    total, comps = reward(rng.normal(size=39), rng.uniform(-1, 1, size=4),
                          rng.normal(size=39))
    assert np.isfinite(total)
    assert comps == {}  # a bare float is `scalar_only`'s shape, and T2R's


#: T2R names them `-v2`; this repo's env ids are `-v3`, the same reward text
#: under Meta-World's own rename.
RELEASED_TASKS = ("door-unlock-v2", "drawer-open-v2", "handle-press-side-v2",
                  "handle-press-v2", "sweep-into-v2", "window-open-v2")


@pytest.mark.parametrize("task", RELEASED_TASKS)
def test_a_released_t2r_program_runs_unmodified_in_this_repo(task):
    """UPSTREAM'S OWN ARTIFACT, executed here, with nothing transcribed.

    T2R ships the programs its pipeline generated for six Meta-World tasks
    (`run_metaworld/reward_code/<task>/general.py`, the pre-conversion
    spelling). Mapped with BIRD's table they compile, bind `[-1, 1, 0]`, and
    return a finite reward -- so "a program written against this prompt is
    runnable here" is a measurement rather than an argument.

    It also checks the mapped code lands on the SAME STATE SLOTS as T2R's own
    `specific.py`, which is the converter's output and what the paper's SAC
    trained on. The goal term is the one allowed difference (upstream calls
    `self.env._get_pos_goal()`; a free function has no env), so it is compared by
    COUNT rather than dropped -- a table that silently rendered nothing for the
    goal would otherwise pass.

    PARAMETRISED, AND THE SKIP IS PER TASK AND PER MODULE, which is the point:
    two of these six released programs import **scipy** -- measured, not guessed:
    `door-unlock-v2` imports `scipy.spatial.distance` and `sweep-into-v2`
    `scipy.spatial`, while the other four import nothing but numpy -- and the
    default test install does not include scipy. A single test over all six
    would fail on an import that has nothing to do with what it checks, and
    gating the whole thing on scipy would throw away the four that do run. So
    each task skips only for the module ITS OWN program imports, named in the
    skip reason.

    That scipy dependency is not only a test fact -- it is a run prerequisite,
    since a model prompted as these programs' author was writes the same
    imports."""
    import numpy as np
    from bird.components.generation import (_SYMBOLS_T2R_METAWORLD,
                                            _apply_symbol_mapping)
    from bird.components.training import CompiledReward

    root = (REPO / "refs" / "code" / "text2reward" / "run_metaworld" / "reward_code")
    if not root.exists():
        pytest.skip("refs/ not checked out")

    general = (root / task / "general.py").read_text()
    specific = (root / task / "specific.py").read_text()

    # Whatever the released program imports, it must be importable to run it.
    for line in general.splitlines():
        mod = re.match(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", line)
        if mod and mod.group(1).split(".")[0] != "numpy":
            pytest.importorskip(
                mod.group(1).split(".")[0],
                reason=f"{task}'s released program imports "
                       f"{mod.group(1)}; not installed here")

    rng = np.random.default_rng(20260918)
    s, a, s2 = rng.normal(size=39), rng.uniform(-1, 1, size=4), rng.normal(size=39)

    def slots(text, var):
        out = []
        for m in re.finditer(rf"(?<!\w){var}\[\s*(-?\d*)\s*(:?)\s*(-?\d*)\s*\]", text):
            lo, colon, hi = m.groups()
            out.append(f"[{lo or '0'}:{hi or 'end'}]" if colon else f"[{lo}]")
        return out

    mapped = _apply_symbol_mapping(_SYMBOLS_T2R_METAWORLD, general)
    assert "self." not in mapped, f"{task}: {mapped}"

    ns = {"np": np}
    exec(compile(mapped, f"<t2r {task}>", "exec"), ns)
    reward = CompiledReward(ns["compute_dense_reward"], "compute_dense_reward")
    assert reward._bind == [-1, 1, 0], f"{task}: {reward._bind}"
    total, _ = reward(s, a, s2)
    assert np.isfinite(total), task

    mine = slots(mapped, "obs")
    assert [x for x in mine if x != "[36:39]"] == slots(specific, "obs"), task
    assert mine.count("[36:39]") == specific.count("_get_pos_goal()"), task


def test_the_default_signature_is_refused_beside_the_abstraction():
    """(c) The invariant that would have refused the broken run at load.

    Both halves of the seam are named, because a reader who hits one has to be
    told which other key to look at -- that is the whole cost of a cross-key
    defect. `_check_coherence` reports both: the missing `self` receiver and the
    missing `obs` the table rewrites into.

    Without this check the config loads, and every candidate exhausts its repair
    attempts without reaching a training step."""
    with pytest.raises(ConfigError) as exc:
        load("card", profile="tester",
             overrides={"generate.output.signature": "compute_reward_state_action_next"})
    msg = str(exc.value)
    assert "generate.output.signature" in msg
    assert "generate.context.env_spec" in msg
    assert "t2r_class_abstraction" in msg
    # the two distinct failures, not one message twice
    assert "first parameter is 'state'" in msg
    assert "NameError" in msg
