"""`evaluate.checkpoint_eval_episodes` -- how many episodes stand behind one checkpoint.

WHY THE KEY EXISTS, and it is a number that was never a decision. The series
`evaluate.fitness.checkpoint_aggregation` collapses is the checkpoint curve, and
each of its rows is an estimate over some number of episodes. The surrogate path
chose 4, with 8 on the last row, and `EVAL_EPISODES`' docstring argues the choice:
"Two is cheaper and *wrong*: on an env whose ground-truth metric is close to
binary, a two-episode estimate swings the curve between 0 and 1 and
`checkpoint_aggregation: final` then selects on evaluation noise rather than on
the reward." The sb3 path -- the only one that runs a real learner -- uses its own
constant, 3, below that floor.

THE CASE THAT MOTIVATES IT. REvolve's App. B.2
Eq. 3 pays 0 unless the humanoid survives all 1000 steps, so it is exactly the
"close to binary" metric the docstring warns about, and a 5M-step training yields
20 checkpoints. Under `checkpoint_aggregation: final` (REvolve's) the fitness is one
3-episode estimate; under `max_over_checkpoints` (Eureka's, and faithful) it is the
MAX of twenty of them, so the noise does not merely widen -- it biases UP. A policy
surviving half its episodes reads a perfect 3/3 on some checkpoint **93%** of the
time. Two arms compared that way differ by their aggregation's noise behaviour
before they differ by their rewards.

THE DEFAULT IS null ON PURPOSE. It means "each backend's own constant", so the key
moves no config hash and changes no existing curve. Only a config that asks gets
something else.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from bird.components.training import (
    _SB3_EVAL_EPISODES,
    EVAL_EPISODES,
    EVAL_EPISODES_FINAL,
    _checkpoint_episodes,
)

TRAINING = pathlib.Path(__file__).resolve().parents[1] / "bird" / "components" / "training.py"


class _Cfg(dict):
    """`cfg.get` over a plain dict -- the only surface `_checkpoint_episodes` uses."""


def _cfg(value):
    return _Cfg({"evaluate.checkpoint_eval_episodes": value})


def test_null_preserves_each_backends_own_constant():
    """The default must change NOTHING, or every recorded curve becomes incomparable."""
    assert _checkpoint_episodes(_cfg(None), sb3=True) == _SB3_EVAL_EPISODES == 3
    assert _checkpoint_episodes(_cfg(None), last=False) == EVAL_EPISODES == 4
    assert _checkpoint_episodes(_cfg(None), last=True) == EVAL_EPISODES_FINAL == 8


def test_a_missing_key_behaves_as_null():
    """An older config that predates the key must not change behaviour either."""
    assert _checkpoint_episodes(_Cfg(), sb3=True) == _SB3_EVAL_EPISODES
    assert _checkpoint_episodes(_Cfg(), last=True) == EVAL_EPISODES_FINAL
    assert _checkpoint_episodes(None, sb3=True) == _SB3_EVAL_EPISODES


@pytest.mark.parametrize("value", [1, 3, 5, 10, 25])
def test_an_integer_applies_to_every_intermediate_checkpoint_on_both_paths(value):
    assert _checkpoint_episodes(_cfg(value), sb3=True) == value
    assert _checkpoint_episodes(_cfg(value), last=False) == value


@pytest.mark.parametrize(("value", "expected"), [(1, 8), (5, 8), (8, 8), (10, 10), (25, 25)])
def test_the_surrogate_last_row_keeps_its_documented_role(value, expected):
    """`EVAL_EPISODES_FINAL` is not folded away by the key.

    That row is the one `checkpoint_aggregation: final` reads, so it is deliberately
    estimated over more episodes than the intermediate ones. An explicit value may
    RAISE that floor and must never lower it -- otherwise setting the key to 5 would
    quietly make the most load-bearing row *less* precise than it is today.
    """
    assert _checkpoint_episodes(_cfg(value), last=True) == expected


def test_a_value_below_one_cannot_disable_evaluation():
    """Zero episodes would make every checkpoint a fabricated number, not a measured one."""
    assert _checkpoint_episodes(_cfg(0), sb3=True) == 1
    assert _checkpoint_episodes(_cfg(-4), sb3=True) == 1


def test_no_checkpoint_evaluation_passes_a_bare_episode_count():
    """Asserted on CALLS, not on words: every `_evaluate_policy_timed` must route
    its episode count through `_checkpoint_episodes`.

    This is the guard that actually holds the key in place. The defect it exists for
    is a literal argument at a call site -- invisible to any grep for the key's name,
    and invisible to a test that only checks the resolver, because the resolver is
    correct and simply not called.
    """
    tree = ast.parse(TRAINING.read_text(encoding="utf-8"))
    bare = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_evaluate_policy_timed"):
            continue
        # signature: (env, policy, rng, reward, n_episodes, on_error)
        if len(node.args) < 5:
            continue
        n_arg = node.args[4]
        routed = (isinstance(n_arg, ast.Call) and isinstance(n_arg.func, ast.Name)
                  and n_arg.func.id == "_checkpoint_episodes")
        if not routed:
            bare.append((node.lineno, ast.dump(n_arg)[:60]))
    assert not bare, (
        "these checkpoint evaluations do not take their episode count from "
        "`_checkpoint_episodes`, so `evaluate.checkpoint_eval_episodes` does not reach "
        f"them: {bare}"
    )


def test_the_guard_above_can_actually_fire():
    """A detector reporting zero is indistinguishable from a broken one.

    Feeds the same AST shape the guard walks, with a bare literal where the resolver
    call belongs, and asserts it is caught.
    """
    src = ("_evaluate_policy_timed(env, policy, rng, reward, 3, on_error)\n"
           "_evaluate_policy_timed(env, policy, rng, reward, _checkpoint_episodes(cfg), on_error)\n")
    tree = ast.parse(src)
    bare = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_evaluate_policy_timed" and len(node.args) >= 5):
            n_arg = node.args[4]
            if not (isinstance(n_arg, ast.Call) and isinstance(n_arg.func, ast.Name)
                    and n_arg.func.id == "_checkpoint_episodes"):
                bare.append(node.lineno)
    assert bare == [1], f"the guard missed the bare literal it exists to catch: {bare}"
