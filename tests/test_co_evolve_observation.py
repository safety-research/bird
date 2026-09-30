"""`update.co_evolve.observation_fn` is READ, and means what its note says.

The failure this guards: a §6 key declared, hashed, diffed --
`--diff limen limen_reward_only` lists it as one of the four keys the two
published points differ on -- and read by nothing. A `limen` tester run with
only this key flipped would validate, get a new hash, and produce a run tree
byte-identical to the unflipped one in every candidate file, record, state,
budget counter and lineage. Observation co-design is driven by
`problem.search_space` (stage 3 installs) and `generate.co_design.observation_fn`
(stage 1 asks for, and lifts, `get_observation`); neither decides whether a
parent's observation carries into its child.

The key is honoured with its stated meaning -- "whether the observation
function persists ... across iterations" -- at the one place a parent's program
reaches a child, `generation._inheritable_program`: with the key false the
parent's `get_observation` and `OBS_DIM` are cut out of everything the child is
handed (the PARENT REWARD and ARCHIVE ELITES blocks, and the program a patching
`edit_mode` inherits), so each candidate carries the observation its own
generation produced. `_check_coherence` ties `true` to
`generate.co_design.observation_fn: true` (persisting something never generated
is the fabricated-pin shape) and refuses the `weights_only` corner where not
persisting leaves no way for an observation to arrive.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from bird import registry
from bird.budget import Budget
from bird.components import generation
from bird.config import ConfigError, load
from bird.context import Context
from bird.types import Candidate, CandidateReport, TrainResult

REPO = pathlib.Path(__file__).resolve().parents[1]

#: MEASURED, not chosen for convenience: under seeds 1, 2 and 5 the mock's
#: iteration-1 LIMEN child is an invalid program under at least one value of the
#: key (no `observation.py` to compare), under 0, 3, 4, 6, 7, 8 and 9 it is valid
#: under BOTH values. The first of those. The mock seeds every sample from its
#: prompt, so any change to prompt text (adapter source included) re-rolls this.
SEED = 0

#: A real model's shape: one code block, a shared helper both functions call
#: (`tests/test_limen_observation.py` explains why that shape matters).
_PROGRAM = '''import math

_OFFSET = 0.045


def _tcp(state):
    return state[0], state[1], state[2] - _OFFSET


def compute_reward(state, action, next_state):
    x, y, z = _tcp(next_state)
    return -math.sqrt((state[4] - x) ** 2 + (state[5] - y) ** 2 + z * z)


OBS_DIM = 3


def get_observation(state):
    x, y, z = _tcp(state)
    return [state[4] - x, state[5] - y, z]
'''

_REWARD_ONLY = "def compute_reward(state, action, next_state):\n    return 0.0\n"


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx(**overrides) -> Context:
    registry.load_all()
    cfg = load("limen", profile="tester", overrides=overrides or None)
    return Context(cfg=cfg, budget=Budget(), env=registry.get("env", "toy_reacher")({}))


def _section(prompt: str, title: str) -> str:
    """One `## TITLE` block of an assembled prompt, up to the next heading."""
    start = prompt.find("## " + title)
    assert start >= 0, f"no {title} section in the prompt"
    end = prompt.find("\n## ", start + 1)
    return prompt[start:end if end > 0 else None]


def _run(tmp_path: pathlib.Path, flag: bool) -> pathlib.Path:
    cfg = load("limen", profile="tester",
               overrides={"seed": SEED, "update.co_evolve.observation_fn": flag})
    _entry().run(cfg, out_root=str(tmp_path / str(flag)))
    return next((tmp_path / str(flag)).iterdir())


# --------------------------------------------------------------------------
# the no-op probe, inverted
# --------------------------------------------------------------------------


def test_the_lone_flip_changes_what_the_child_is_shown_and_writes(tmp_path) -> None:
    """Two `limen` tester runs differing in this key alone.

    The load-bearing assertion is on the PROMPT: with the key true the
    iteration-1 child's PARENT REWARD and ARCHIVE ELITES blocks quote the
    parent's whole program, `get_observation` included; with it false they quote
    the reward and nothing else, while the OBSERVATION CO-DESIGN contract still
    asks the child for its own. The artifact assertions follow from that (the
    mock seeds every sample from its prompt), and they are the no-op probe
    run again: the two trees must differ at the child and -- because the key
    acts only at the iteration boundary -- nowhere before it.
    """
    runs = {flag: _run(tmp_path, flag) for flag in (True, False)}

    for flag, run in runs.items():
        child = run / "candidates" / "iter01_c0001"
        assert json.loads((child / "meta.json").read_text())["valid"], (
            f"premise broken: seed {SEED}'s iteration-1 child is invalid under "
            f"co_evolve={flag}; pick another measured seed")
        prompt = "\n".join(m["content"] for m in json.loads((child / "prompt.json").read_text()))
        parent, elites = _section(prompt, "PARENT REWARD"), _section(prompt, "ARCHIVE ELITES")
        # The reward persists under both values -- that half is not this key's.
        assert "def compute_reward" in parent and "def compute_reward" in elites
        # The observation persists under exactly one.
        assert ("def get_observation" in parent) is flag, (
            f"co_evolve.observation_fn={flag}: PARENT REWARD "
            f"{'hides' if flag else 'shows'} the parent's get_observation")
        assert ("OBS_DIM" in parent) is flag
        assert ("def get_observation" in elites) is flag, (
            f"co_evolve.observation_fn={flag}: ARCHIVE ELITES "
            f"{'hides' if flag else 'shows'} an elite's get_observation")
        # Still ASKED for, either way: the §1 contract is untouched by §6.
        assert "def get_observation(state):" in _section(prompt, "OBSERVATION CO-DESIGN")

    on, off = (runs[True] / "candidates", runs[False] / "candidates")
    assert (on / "iter01_c0001" / "observation.py").read_text() != \
        (off / "iter01_c0001" / "observation.py").read_text(), (
            "the two runs' iteration-1 observations are byte-identical: the flip "
            "did not reach the child's generation")
    for name in ("reward.py", "observation.py", "prompt.json"):
        assert (on / "iter00_c0000" / name).read_text() == \
            (off / "iter00_c0000" / name).read_text(), (
                f"iteration 0 has no parent to inherit from, so {name} must not move")


# --------------------------------------------------------------------------
# the two paths a parent's program takes into a child
# --------------------------------------------------------------------------


def test_a_patching_edit_mode_does_not_inherit_the_parents_observation() -> None:
    """The second path. Under `diff` / `weights_only`, `_apply_edit_mode` patches
    or copies `parent_code`, and `_apply_co_design` then lifts `get_observation`
    out of whatever that produced -- so handing it the raw parent program would
    carry a parent's observation into a child's `reward_code` verbatim, whatever
    §6 said. `_parent_ids` hands over the inheritable program."""
    registry.load_all()
    parent = Candidate(cand_id="c0000", iteration=0, reward_code=_PROGRAM,
                       observation_code=generation._extract_observation(_PROGRAM)[0])
    report = CandidateReport(cand_id="c0000", candidate=parent,
                             result=TrainResult(cand_id="c0000", candidate=parent),
                             fitness=0.5)
    for flag in (True, False):
        ctx = _ctx(**{"generate.output.edit_mode": "diff",
                      "update.co_evolve.observation_fn": flag})
        parent_id, code = generation._parent_ids(ctx, [report])
        assert parent_id == "c0000"
        assert "def compute_reward" in code and "_tcp" in code
        assert ("def get_observation" in code) is flag, (
            f"co_evolve.observation_fn={flag}: the program handed to the edit mode "
            f"{'lacks' if flag else 'carries'} the parent's get_observation")
        assert ("OBS_DIM" in code) is flag
    assert generation._parent_ids(ctx, []) == (None, "")


def test_strip_observation_cuts_the_observation_and_nothing_else() -> None:
    out = generation._strip_observation(_PROGRAM)
    assert "get_observation" not in out and "OBS_DIM" not in out
    assert "def compute_reward" in out and "def _tcp" in out and "_OFFSET = 0.045" in out
    compile(out, "<stripped>", "exec")   # still a program
    # Identity when there is nothing to strip, or nothing to parse.
    assert generation._strip_observation(_REWARD_ONLY) == _REWARD_ONLY
    assert generation._strip_observation("def broken(:") == "def broken(:"


def test_persistence_is_moot_when_nothing_is_co_designed() -> None:
    """With §1's elicitation off a `get_observation` in a program is a helper
    the reward may call, not a co-designed artefact: nothing to persist or not.
    Every non-LIMEN config takes this branch, so the key cannot change it."""
    ctx = _ctx(**{"generate.co_design.observation_fn": False,
                  "update.co_evolve.observation_fn": False})
    assert generation._inheritable_program(ctx, _PROGRAM) == _PROGRAM
    on = _ctx()
    assert on.cfg["update.co_evolve.observation_fn"] is True, "limen's published value"
    assert generation._inheritable_program(on, _PROGRAM) == _PROGRAM


# --------------------------------------------------------------------------
# coherence
# --------------------------------------------------------------------------


def test_persisting_an_observation_never_generated_is_refused() -> None:
    """`co_evolve: true` over `co_design: false` -- limen with §1 switched off
    alone -- is a pin on a mechanism with nothing to act on."""
    with pytest.raises(ConfigError) as exc:
        load("limen", profile="tester", overrides={"generate.co_design.observation_fn": False})
    msg = str(exc.value)
    assert "update.co_evolve.observation_fn" in msg
    assert "generate.co_design.observation_fn" in msg


def test_weights_only_without_persistence_is_refused() -> None:
    """`weights_only` copies the parent's program verbatim, so with the
    observation not persisting no child after iteration 0 could carry one."""
    with pytest.raises(ConfigError) as exc:
        load("limen", profile="tester",
             overrides={"generate.output.edit_mode": "weights_only",
                        "update.co_evolve.observation_fn": False})
    msg = str(exc.value)
    assert "weights_only" in msg and "update.co_evolve.observation_fn" in msg
    # With persistence on, the same edit mode is a legal (if unpublished) point.
    load("limen", profile="tester", overrides={"generate.output.edit_mode": "weights_only"})


def test_the_ablation_point_and_both_published_points_load() -> None:
    """The lone flip is a real arm, and the shipped pair differs in exactly the
    keys the release's ablation differs in.

    Four, not three: LIMEN's release resolves its SYSTEM prompt
    per evolution mode (`limen/prompts.py:453-458` -- the `full` text asks for two
    functions, the `reward_only` text for one), so the faithful reward-only arm
    pins its own mode's `generate.context.system_prompt`;
    `tests/test_ablation.py::test_limen_reward_only_is_a_search_space_diff_and_nothing_else`
    carries the same set for the same reason.
    """
    off = load("limen", profile="tester", overrides={"update.co_evolve.observation_fn": False})
    assert off["generate.co_design.observation_fn"] is True
    diff = load("limen").diff(load("limen_reward_only"))
    assert set(diff) == {"name", "problem.search_space",
                         "generate.co_design.observation_fn",
                         "update.co_evolve.observation_fn",
                         "generate.context.system_prompt"}, sorted(diff)


def test_the_mock_child_evolves_a_quoted_parent_observation_structurally() -> None:
    """The mock's co-evolved child differs from a fresh one BY CONSTRUCTION, not by
    a seeded draw: with a parent `OBS_DIM = k` quoted in the prompt the child's width
    is k moved by one and the docstring names the parent; without it the width is
    the draw. Same seed, same table, one quoted line apart. This is what makes the
    test above a test of the flip rather than of two draws happening to differ
    (two seeded draws can stop differing when an unrelated prompt paragraph moves
    the hash, deterministically, on every run)."""
    import random
    from bird.llm import mock as M
    table = "\n".join(f"    {n}: float  # s[{i}]" for i, n in
                      enumerate(("x", "y", "vx", "vy", "goal_x", "goal_y")))
    fresh = M._observation_program(random.Random(7), 512, table)
    quoted = M._observation_program(random.Random(7), 512, table + "\nOBS_DIM = 5\n")
    assert fresh != quoted
    assert "evolved from the parent's 5-feature observation" in quoted
    assert quoted.splitlines()[0] == "OBS_DIM = 6"
    # prose that merely NAMES the declaration is not a parent
    prose = M._observation_program(random.Random(7), 512,
                                   table + "\nDeclare the count explicitly as OBS_DIM = <n>.\n")
    assert prose == fresh
