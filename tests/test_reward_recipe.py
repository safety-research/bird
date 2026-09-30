"""`generate.context.include_reward_recipe_hints` must send the PUBLISHED text.

Why a prompt-text pin is not a reflex test: citable records -- configs/methods/card.yaml
("verbatim"), configs/methods/text2reward_zeroshot.yaml -- can describe a three-/five-item
published template while every prompt sends BIRD's own four-item paraphrase
(`generation._RECIPE_HINTS`), and every counter reads normal: the key is read,
the section rendered, the run completes. Nothing but the prompt bytes themselves
can witness this class, which is the same shape as a key read only into a string
(the `images=` family).

The published strings are HARD-CODED here as a second copy on purpose. refs/ is
not part of the repository (`scripts/fetch_refs.sh` fetches it), so it cannot be
the oracle in CI, and a disagreement between this copy and the constant is
exactly the failure to raise. Anyone re-deriving them must use the RELEASE files, not
refs/tex/text2reward (the paper drops the hedge and "the"), and must keep the
trailing "..." line.
"""

import importlib.util
import json

import pytest
from conftest import REPO

from bird.components import generation
from bird.config import ConfigError, load
from bird.schema import SCHEMA

KEY = "generate.context.include_reward_recipe_hints"

# refs/code/text2reward/code_generation/single_flow/classlike_prompt/MetaworldPrompt.py:8-12
# = refs/code/CARD/code_generation/self_reflection/benchmark_prompt/metaworld_prompt.py:6-10
# = refs/tex/card/main.tex:728-732
T2R_METAWORLD = (
    "Typically, the reward function of a manipulation task is consisted of these "
    "following parts (some part is optional, so only include it if really necessary):\n"
    "1. the distance between robot's gripper and our target object\n"
    "2. difference between current state of object and its goal state\n"
    "3. regularization of the robot's action\n"
    "..."
)

# refs/code/text2reward/code_generation/single_flow/classlike_prompt/PandaPrompt.py:9-15
# = refs/code/CARD/code_generation/self_reflection/benchmark_prompt/panda_prompt.py:9-15
T2R_PANDA = (
    "Typically, the reward function of a manipulation task is consisted of these "
    "following parts (some part is optional, so only include it if really necessary):\n"
    "1. the distance between robot's gripper and our target object\n"
    "2. difference between current state of object and its goal state\n"
    "3. regularization of the robot's action\n"
    "4. [optional] extra constraint of the target object, which is often implied by "
    "the task instruction\n"
    "5. [optional] extra constraint of the robot, which is often implied by "
    "the task instruction\n"
    "..."
)

# A phrase that occurs in `_RECIPE_HINTS` and in neither published text.
OURS_ONLY = "a bonus on task completion"


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


def test_card_sends_the_published_metaworld_recipe(tmp_path):
    """card resolves `t2r_metaworld`; the section body is the three-item release text.

    Fails if card resolves `true`: the body is then `_RECIPE_HINTS`, and
    'consisted of these following parts' occurs in 0/4 prompts.
    """
    user = _user_messages(load("card", profile="tester"), tmp_path)[0]
    assert "## REWARD RECIPE\n" + T2R_METAWORLD in user
    assert OURS_ONLY not in user


def test_text2reward_sends_the_published_panda_recipe(tmp_path):
    """text2reward_zeroshot resolves `t2r_panda`; the five-item release text, both
    [optional] items included. The paraphrase fails: items 4-5 absent, completion bonus present."""
    user = _user_messages(load("text2reward_zeroshot", profile="tester"), tmp_path)[0]
    assert "## REWARD RECIPE\n" + T2R_PANDA in user
    assert "4. [optional] extra constraint of the target object" in user
    assert "5. [optional] extra constraint of the robot" in user
    assert OURS_ONLY not in user


def test_text2reward_human_inherits_the_panda_recipe():
    """Resolves, does not run. Guards the `extends: text2reward_zeroshot.yaml`
    chain: values replace on extends, so a future pin in the child would silently
    diverge the two T2R points. Resolving `true` fails."""
    assert load("text2reward_human")[KEY] == "t2r_panda"


def test_a_method_that_does_not_pin_the_key_renders_no_recipe(tmp_path):
    """The identity half: `false` renders nothing, in any prompt of the run.

    Passes under the paraphrase too. It is here to guard the `elif recipe:` branch
    against a future `if recipe is not False` slip that would render the
    paraphrase for every method that never asked for a recipe -- Eureka's
    explicit anti-value."""
    for user in _user_messages(load("eureka", profile="tester"), tmp_path):
        assert "## REWARD RECIPE" not in user


def test_true_is_birds_paraphrase_and_neither_published_text(tmp_path):
    """`true` survives as BIRD's own unpublished paraphrase, byte-stable,
    so every existing card/T2R run dir keeps validating and resuming.

    `true` is the value a reader will most plausibly misread as 'the published
    template on'; this test is the record that it is not."""
    cfg = load("card", profile="tester", overrides={KEY: True})
    user = _user_messages(cfg, tmp_path)[0]
    assert "## REWARD RECIPE\n" + generation._RECIPE_HINTS in user
    assert T2R_METAWORLD not in user
    assert T2R_PANDA not in user


def test_recipe_values_and_texts_stay_in_sync():
    """Every string the schema admits has a text, and every text is a published one.

    Old code fails on import (no `_RECIPES`; the field has no enum). Without this
    a value added to the enum but not the dict validates and then KeyErrors at the
    first prompt -- after status.json already says `running`."""
    strings = {v for v in SCHEMA[KEY].enum if isinstance(v, str)}
    assert set(generation._RECIPES) == strings
    assert False in SCHEMA[KEY].enum and True in SCHEMA[KEY].enum
    for text in generation._RECIPES.values():
        assert text.startswith(
            "Typically, the reward function of a manipulation task is consisted of "
            "these following parts")
        assert text.endswith("...")
    assert generation._RECIPES["t2r_metaworld"] == T2R_METAWORLD
    assert generation._RECIPES["t2r_panda"] == T2R_PANDA


def test_the_card_tex_locator_names_exactly_the_lines_the_constant_reproduces():
    """`refs/tex/card/main.tex:728-732` is five lines: header, three items, "...".

    A range of 728-731 would omit exactly the open "..." line the constant
    reproduces, and every check would stay green because a locator in a
    comment is prose. This turns the range into a claim the file can refute.
    Skips where refs/ is absent (fetch it with `scripts/fetch_refs.sh`)."""
    tex = REPO / "refs" / "tex" / "card" / "main.tex"
    if not tex.exists():
        pytest.skip("refs/ not checked out")
    lines = tex.read_text().splitlines()
    assert "\n".join(lines[727:732]) == generation._RECIPES["t2r_metaworld"]
    # The line before is the sentence that introduces the template, and the
    # line after leaves it -- so 728-732 is the tight range, not a superset.
    assert not lines[726].startswith("Typically")
    assert lines[732] != "..."


def test_an_unlisted_string_is_refused_with_the_allowed_values():
    """The enum path through the REAL validator on the first mixed bool|str field.

    Old code raises too, but with 'expected bool'; the match on the value list is
    what fails there."""
    with pytest.raises(ConfigError, match="t2r_metaworld"):
        load("card", overrides={KEY: "eureka"})
