"""`legend_lines` on the pure-numpy adapters: `bird/envs/control.py` (`pendulum`,
`pendulum_discrete`, `acrobot`) and `bird/envs/toy.py` (`toy_reacher`, `toy_gridworld`,
`toy_hungry_thirsty`).

`EnvAdapter.legend_lines` is the contract: the lines name what the task ASKS at a state
and never how it is going, because a label on a frame scored by
`evaluate.fitness.source: vlm_score` must not carry the answer. Nothing is grandfathered
and the verdict ban is unconditional.

Simulator-free by construction -- pure numpy, well under a second -- so this file runs in
the default CI selection. Pinned here are the properties whose failure is SILENT:

  * a line `bird/envs/hud.py`'s font cannot draw (a tofu box, invisible to every other
    test) or one past the 28-character budget;
  * a verdict in the label, checked two ways -- a word ban, and the equality that the
    legend on a state `task_metric` scores 1.0 is the legend on one it scores 0.0. The
    equality is the real check; a legend that printed the distance to go as a bare
    number would pass the ban;
  * a per-episode target read from the CLASS instead of the STATE -- right on the episode
    being rendered, wrong on every stored row (`toy_reacher`'s `s[4:6]`, `toy_gridworld`'s
    `s[2:4]`);
  * the filter over `registry.names("env")` going quiet: the id set is an EQUALITY, so a
    module rename cannot leave every test here green over nothing.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from bird import registry
from bird.envs import control, hud, toy

#: The factory's `__module__` is the filter (`tests/conftest.py::env_param` uses the same
#: key for the HumanoidBench marker): a class list would have to be told about a new
#: adapter, whereas anything either module registers is parametrised here automatically.
_MODULES = frozenset({control.__name__, toy.__name__})

#: An EQUALITY, not a floor: a floor only catches shrinkage. A new
#: env in either module must be added here -- and thereby gets every test below.
EXPECTED_IDS = ("acrobot", "pendulum", "pendulum_discrete",
                "toy_gridworld", "toy_hungry_thirsty", "toy_reacher")

#: Words that name the ANSWER to the question a frame poses, including the spelled-out
#: metric. `DIST` also catches `DISTANCE`.
_VERDICT_WORDS = ("SUCCESS", "FAIL", "PASS", "SCORE", "METRIC", "FITNESS", "DIST",
                  "REWARD", "DONE", "TASK METRIC")

#: `EnvAdapter.legend_lines`: a 5x7 glyph at 2x with one pixel of tracking is 12 px a
#: character, so 28 is what a 360-wide frame holds with the panel's own padding.
MAX_CHARS = 28


def _numpy_env_ids():
    registry.load_all()
    return sorted(n for n in registry.names("env")
                  if getattr(registry.get("env", n), "__module__", "") in _MODULES)


ENV_IDS = _numpy_env_ids()


def _make(env_id: str):
    return registry.get("env", env_id)({})


def _assert_legible(env_id: str, lines) -> None:
    """The legibility half of the contract: a non-empty list of non-empty upper-case
    strings, each at most `MAX_CHARS`, drawn only from the characters
    `bird/envs/hud.py`'s sheet has a glyph for. No parentheses, no underscore, no
    asterisk and no lowercase are in that sheet, so the set difference covers them."""
    assert isinstance(lines, list), (env_id, type(lines))
    assert lines, f"{env_id}: an empty legend says nothing about what the task asks"
    ok = set(hud._SHEET_CHARS)
    for line in lines:
        assert isinstance(line, str) and line.strip(), (env_id, line)
        assert line == line.upper(), f"{env_id}: not upper-case: {line!r}"
        assert len(line) <= MAX_CHARS, f"{env_id}: {len(line)} chars: {line!r}"
        stray = sorted(set(line) - ok)
        assert not stray, f"{env_id}: no glyph for {stray} in {line!r}"


def _states(env, seed: int, n: int = 16):
    """The reset state and `n` coverage states. `random_state` deliberately: it spans
    the whole box (fast spins, the far corners), where `reset` is the narrow initial
    distribution, and the legend has to be legible on every row a rollout can store."""
    rng = np.random.default_rng(seed)
    return [env.reset(rng)] + [env.random_state(rng) for _ in range(n)]


#: Per env, `(hit, miss)`: a state `task_metric` scores 1.0 over `[s, s]` and one it
#: scores 0.0, agreeing on every dimension the legend may read (the target block, where
#: there is one). A table keyed on the id rather than an if-chain, and its key set is
#: asserted equal to `ENV_IDS` so a new adapter cannot arrive without a pair.
_VERDICT_PAIRS = {
    "pendulum": lambda env: (env._obs(0.0, 0.0), env._obs(math.pi, 0.0)),
    "pendulum_discrete": lambda env: (env._obs(0.0, 0.0), env._obs(math.pi, 0.0)),
    # th1 = pi folds the arm straight up: tip height 2.0 against a bar at 1.0.
    "acrobot": lambda env: (env._obs(np.array([math.pi, 0.0, 0.0, 0.0])),
                            env._obs(np.zeros(4))),
    "toy_reacher": lambda env: (np.array([0.5, 0.5, 0.0, 0.0, 0.5, 0.5]),
                                np.array([-0.8, -0.8, 0.0, 0.0, 0.5, 0.5])),
    "toy_gridworld": lambda env: (env._obs(5, 5), env._obs(0, 0)),
    # fed vs hungry: `task_metric` is the fraction of steps with s[2] < 0.5.
    "toy_hungry_thirsty": lambda env: (np.array([0.0, 0.0, 0.0, 0.0]),
                                       np.array([0.0, 0.0, 1.0, 1.0])),
}


# --------------------------------------------------------------------------
# the family
# --------------------------------------------------------------------------

def test_the_two_modules_register_exactly_these_envs():
    """The parametrisation below is derived; this is what stops it deriving nothing."""
    assert tuple(ENV_IDS) == EXPECTED_IDS
    assert set(_VERDICT_PAIRS) == set(ENV_IDS)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_every_numpy_env_returns_a_legible_legend_on_reset_and_random_states(env_id):
    env = _make(env_id)
    for s in _states(env, seed=11):
        _assert_legible(env_id, env.legend_lines(s))
    # A stored row may come back as a list (JSON round trip); the legend must not care.
    _assert_legible(env_id, env.legend_lines(list(map(float, env.reset(np.random.default_rng(1))))))


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_the_legend_names_the_ask_and_never_a_verdict_word(env_id):
    """The ban is unconditional -- nothing is grandfathered."""
    env = _make(env_id)
    for s in _states(env, seed=3, n=8):
        text = " ".join(env.legend_lines(s))
        for banned in _VERDICT_WORDS:
            assert banned not in text, f"{env_id}: legend says {banned!r}: {text!r}"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_the_legend_is_identical_on_a_scored_and_an_unscored_state(env_id):
    """The verdict flips and the legend does not. Asserted through `task_metric` on a
    two-row trajectory, so the pair is proven to BE a flip on this tree rather than
    assumed from the constants."""
    env = _make(env_id)
    hit, miss = _VERDICT_PAIRS[env_id](env)
    assert env.task_metric(np.array([hit, hit])) == 1.0, env_id
    assert env.task_metric(np.array([miss, miss])) == 0.0, env_id
    assert env.legend_lines(hit) == env.legend_lines(miss), env_id


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_nothing_but_a_target_moves_the_legend(env_id):
    """`random_state` spans the whole state box and never moves a target (`toy_reacher`
    and `toy_gridworld` write the class goal into every row), so over many draws the
    legend must take exactly ONE value. A legend that quoted an angle, a speed, a
    position or a flag would take many."""
    env = _make(env_id)
    seen = {tuple(env.legend_lines(s)) for s in _states(env, seed=7, n=40)}
    assert len(seen) == 1, (env_id, sorted(seen))


# --------------------------------------------------------------------------
# control.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("env_id", ["pendulum", "pendulum_discrete"])
def test_pendulum_legend_is_the_ask_and_both_tolerances_built_from_the_constants(env_id):
    """Three constant lines: the ask, then the angle AND the speed tolerance --
    `task_metric` is upright-and-still, so an angle alone would state half the ask.
    Built from `upright_angle` / `upright_speed`, and the numbers on the frame are
    checked against those attributes, not against a second copy of the string."""
    env = _make(env_id)
    lines = env.legend_lines(env._obs(0.0, 0.0))
    assert lines == ["SWING UP, HOLD UPRIGHT",
                     "UPRIGHT: ANGLE < 0.2 RAD",
                     "STILL: SPEED < 1 RAD/S"]
    assert f"{env.upright_angle:g}" in lines[1]
    assert f"{env.upright_speed:g}" in lines[2]
    # A rod at 1.234 rad spinning at 5.67 rad/s: neither number, in any rounding, may
    # reach the frame. The legend is the same three lines.
    assert env.legend_lines(env._obs(1.234, 5.67)) == lines


def test_pendulum_discrete_inherits_the_pendulum_legend_unchanged():
    """Same physics, same ask: `pendulum` -> `pendulum_discrete` changes the state
    representation and nothing else, and the legend must say so."""
    s = np.array([math.cos(0.4), math.sin(0.4), -2.0])
    assert _make("pendulum_discrete").legend_lines(s) == _make("pendulum").legend_lines(s)


def test_acrobot_legend_names_the_bar_from_goal_height_and_never_the_tip():
    """The bar is the ask (`tasks/acrobot/shared_spec.yaml`: "a bar one link-length
    above the shoulder"); the tip's height beside it would be the verdict in
    link-lengths. An inverted arm (tip at 2.0) gets the same two lines as a hanging one,
    and `2.0` appears nowhere."""
    env = _make("acrobot")
    hanging = env._obs(np.zeros(4))
    lines = env.legend_lines(hanging)
    assert lines == ["SWING TIP ABOVE BAR, STAY", "BAR: 1.0 LINK ABOVE SHOULDER"]
    assert f"{env.goal_height:.1f}" in lines[1]
    inverted = env._obs(np.array([math.pi, 0.0, 0.0, 0.0]))
    assert env.tip_height(inverted) > env.goal_height
    assert env.legend_lines(inverted) == lines
    assert "2.0" not in " ".join(lines)


def test_acrobot_step_reports_the_per_step_flag_its_legend_docstring_says():
    """`Acrobot.legend_lines` documents that `_step` carries `tip_height` AND a
    `success` key -- the tip above the bar on that step. Two copies of one fact; this
    holds them together, and fails the day the flag goes away so the docstring gets
    revisited with it."""
    env = _make("acrobot")
    s = env.reset(np.random.default_rng(0))
    _s2, _done, info = env.step(s, np.zeros(1))
    assert "tip_height" in info and "success" in info
    assert info["success"] == (info["tip_height"] > env.goal_height)


# --------------------------------------------------------------------------
# toy.py
# --------------------------------------------------------------------------

def test_toy_reacher_legend_reads_the_goal_from_the_state_and_moves_only_with_it():
    """`s[4:6]` is the goal block `_reset` writes and `task_metric` measures against.
    Two states differing ONLY in that block differ ONLY in the goal line; a state
    differing everywhere BUT that block does not move the legend at all. The quoted
    numbers are parsed back and compared to the array, not re-formatted."""
    env = _make("toy_reacher")
    s = env.reset(np.random.default_rng(5))
    assert tuple(s[4:6]) == env.goal
    base = env.legend_lines(s)
    assert base == ["REACH THE GOAL AND STAY", "GOAL X 0.50 Y 0.50", "GOAL RADIUS 0.22"]
    assert f"{env.goal_radius:.2f}" in base[2]

    moved = np.array(s, dtype=float)
    moved[4:6] = (-0.37, 0.81)
    got = env.legend_lines(moved)
    assert got != base
    assert got[1] == "GOAL X -0.37 Y 0.81"
    assert got[0] == base[0] and got[2] == base[2]
    quoted = [float(x) for x in got[1].split()[2::2]]
    assert np.allclose(quoted, moved[4:6], atol=0.005 + 1e-9), (got[1], moved[4:6])

    other = np.array(s, dtype=float)
    other[0:4] = (0.9, -0.9, 1.0, -1.0)          # position and velocity, not the goal
    assert env.legend_lines(other) == base

    # The class constant is NOT what is read: a row whose goal is elsewhere is captioned
    # with its own goal even though `env.goal` never changed.
    assert tuple(env.goal) == (0.5, 0.5)
    assert "0.50" not in got[1]

    # `-0.00` is normalised, and the widest goal the box allows still fits the font.
    zero = np.array(s, dtype=float)
    zero[4:6] = (-0.001, 0.0)
    assert env.legend_lines(zero)[1] == "GOAL X 0.00 Y 0.00"
    wide = np.array(s, dtype=float)
    wide[4:6] = (-1.0, -1.0)
    _assert_legible("toy_reacher", env.legend_lines(wide))


def test_toy_gridworld_legend_reads_the_goal_cell_from_the_state():
    """`s[2:4]` is the goal cell `_obs` writes. The agent's cell and the in-lava flag
    are in the state too and must not move the legend: at the goal, in the lava and
    anywhere else the two lines are the same."""
    env = _make("toy_gridworld")
    s = env._obs(0, 3)
    assert env.legend_lines(s) == ["REACH CELL 5,5", "AVOID THE LAVA"]
    moved = np.array(s, dtype=float)
    moved[2:4] = (2, 0)
    assert env.legend_lines(moved) == ["REACH CELL 2,0", "AVOID THE LAVA"]
    assert env.legend_lines(env._obs(1, 1)) == env.legend_lines(s)      # in the lava
    assert env.legend_lines(env._obs(5, 5)) == env.legend_lines(s)      # at the goal
    assert env._obs(1, 1)[4] == 1.0 and s[4] == 0.0, "the pair above did flip the flag"


def test_toy_hungry_thirsty_legend_omits_hunger_because_hunger_is_the_metric():
    """`task_metric` is the fraction of steps not hungry and `_step`'s per-step flag is
    `not hungry`, so `HUNGRY` / `FED` on a frame would be the verdict verbatim; thirst
    goes with it (`ToyHungryThirsty.legend_lines` has the argument). The legend is two
    constant lines naming the cells from `food_cell` / `water_cell`, identical across
    every combination of the two flags -- while the metric flips."""
    env = _make("toy_hungry_thirsty")
    fed = np.array([0.0, 0.0, 0.0, 0.0])
    lines = env.legend_lines(fed)
    assert lines == ["EAT AT FOOD 0,0 WHEN HUNGRY", "IF THIRSTY, DRINK AT 0,2"]
    assert f"{env.food_cell[0]},{env.food_cell[1]}" in lines[0]
    assert f"{env.water_cell[0]},{env.water_cell[1]}" in lines[1]
    for h in (0.0, 1.0):
        for t in (0.0, 1.0):
            s = np.array([3.0, 1.0, h, t])
            assert env.legend_lines(s) == lines, (h, t)
    hungry = np.array([0.0, 0.0, 1.0, 0.0])
    assert env.task_metric(np.array([fed, fed])) == 1.0
    assert env.task_metric(np.array([hungry, hungry])) == 0.0
