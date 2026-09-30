"""R*'s module-level crossover reaches the reward PPO trains on, and its critic
author is told what the paper tells it.

Crossover. A `module_insert` that appends the donor's module to the
recipient's returned DICT without touching the recipient's scalar total is
silently wrong: `CompiledReward` reads `out[0]` as the total whenever a reward
returns `(total, dict)`, so for every recipient that spells its total out --
`total = reach; return total, {...}` -- the child would train on exactly the
recipient's reward while the artifact advertised a recombined one. The wrong
answer is a well-formed child that compiles, trains and scores; only the number
is a lie.

Critics. Prompt 1/2 (pp.13-14) and App. A p.12 give the critic the task, the
environment, the success criteria and named raw observation keys. A prompt
listing only `Available keys: rewards, states` (which reading
`env.state_fields`, an attribute no adapter defines -- they carry
`_state_fields` -- produces), with no environment description and no success
conditions, and a `_traj_dict` handing the critic the COLLECTING candidate's
own reward series to rank, would give it none of that.
"""
from __future__ import annotations

import ast
import random
from types import SimpleNamespace

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components import alignment
from bird.components.alignment import (_critic_key_hint, _traj_dict, author_critics,
                                       module_insert, module_insert_detail)
from bird.components.training import compile_reward
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Trajectory

_Z = (np.array([0.0]), np.array([0.0]), np.array([0.0]))


def _total(code: str, s=_Z):
    return compile_reward(code)(*s)


# ------------------------------------------------------------------ crossover

def test_a_module_grafted_into_an_explicit_total_reaches_the_total():
    """Recipient scalar 1, donor module 10: the child must not return
    (1.0, {'reach': 1.0, 'lift': 10.0})."""
    recipient = ('def reward(s, a, s2):\n    reach = 1.0\n    total = reach\n'
                 '    return total, {"reach": reach}')
    donor = 'def reward(s, a, s2):\n    lift = 10.0\n    return lift, {"lift": lift}'
    code, composition = module_insert_detail(recipient, donor, "lift")
    assert composition == "add"
    total, comps = _total(code)
    assert comps == {"reach": 1.0, "lift": 10.0}
    assert total == 11.0, "the policy must see the module it is told about"


def test_an_inline_dict_beside_the_total_is_amended_and_added():
    recipient = 'def reward(s, a, s2):\n    reach = 1.0\n    return reach, {"reach": reach}'
    donor = 'def reward(s, a, s2):\n    lift = 10.0\n    return lift, {"lift": lift}'
    code, composition = module_insert_detail(recipient, donor, "lift")
    assert composition == "add"
    assert _total(code) == (11.0, {"reach": 1.0, "lift": 10.0})


def test_a_total_summed_from_the_dict_is_not_counted_twice():
    """`return sum(parts.values()), parts`: the amended dict already carries the
    module into the total, so adding it again would double it."""
    recipient = ('def reward(s, a, s2):\n    d = s[0]\n    parts = {"reach": d}\n'
                 '    return sum(parts.values()), parts')
    donor = ('def reward(s, a, s2):\n    q = 10.0 * s[0]\n    parts = {"lift": q}\n'
             '    return sum(parts.values()), parts')
    code, composition = module_insert_detail(recipient, donor, "lift")
    assert composition == "dict_sum"
    total, comps = _total(code, (np.array([2.0]), np.array([0.0]), np.array([0.0])))
    assert comps == {"reach": 2.0, "lift": 20.0}
    assert total == 22.0


def test_a_dict_only_reward_needs_no_total_edit():
    recipient = 'def reward(s, a, s2):\n    reach = 1.0\n    return {"reach": reach}'
    donor = 'def reward(s, a, s2):\n    lift = 10.0\n    return {"lift": lift}'
    code, composition = module_insert_detail(recipient, donor, "lift")
    assert composition == "dict_only"
    assert _total(code) == (11.0, {"reach": 1.0, "lift": 10.0}), "`_unpack` sums the dict"


def test_the_donor_value_is_bound_once_and_read_twice():
    """The donor's value expression is bound to a name and READ twice -- in the
    dict and added to the total -- rather than duplicated, so a module whose
    value has a side effect cannot report one number and train on another."""
    recipient = ('def reward(s, a, s2):\n    reach = 1.0\n    total = reach\n'
                 '    return total, {"reach": reach}')
    donor = 'def reward(s, a, s2):\n    lift = 7.0 * s[0] + 3.0\n    return lift, {"lift": lift}'
    code, composition = module_insert_detail(recipient, donor, "lift")
    assert composition == "add"
    # The value expression is grafted exactly once (bound to a fresh name).
    #
    # COMPARED BY AST, NOT BY TEXT. `ast.dump` with default arguments excludes line and
    # column attributes, so this asserts the STRUCTURE the test's docstring is about and
    # is indifferent to how the tree was rendered. A text form such as
    # `code.count("7.0 * s[0] + 3.0") == 1` would silently encode one unparser's
    # parenthesisation as the contract: another unparser (the `astunparse` package,
    # say) renders the same AST as
    # `((7.0 * s[0]) + 3.0)` and would fail a test about grafting for a reason that
    # has nothing to do with grafting.
    donor_value = ast.parse("7.0 * s[0] + 3.0", mode="eval").body
    grafts = sum(1 for n in ast.walk(ast.parse(code))
                 if ast.dump(n) == ast.dump(donor_value))
    assert grafts == 1, f"donor value expression appears {grafts} times in the AST"
    total, comps = _total(code, (np.array([2.0]), np.array([0.0]), np.array([0.0])))
    assert comps["lift"] == 17.0 and total == pytest.approx(1.0 + 17.0)


def test_module_insert_still_returns_only_the_code():
    recipient = 'def reward(s, a, s2):\n    reach = 1.0\n    return reach, {"reach": reach}'
    donor = 'def reward(s, a, s2):\n    lift = 10.0\n    return lift, {"lift": lift}'
    assert module_insert(recipient, donor, "lift") == module_insert_detail(recipient, donor, "lift")[0]


# -------------------------------------------------------------------- critics

def _stub_env():
    return SimpleNamespace(_state_fields=[("SEMANTIC_FIELD_SENTINEL", "distance to target"),
                                          ("other", "another field")],
                           describe=lambda kind: "ENV_DESCRIPTION_SENTINEL",
                           describe_success=lambda: "SUCCESS_CONDITION_SENTINEL")


def test_the_critic_prompt_carries_env_fields_success_and_task_and_not_rewards():
    """A sentinel state field must reach the prompt, which must say more than
    `Available keys: rewards, states`."""
    registry.load_all()
    ctx = Context(cfg=load("rstar"), budget=Budget(), env=_stub_env(), rng=random.Random(0))
    prompts = []
    ctx.generator = lambda prompt, **kw: prompts.append(prompt) or []

    author_critics(ctx, RunState())

    assert len(prompts) == 1
    p = prompts[0]
    assert "SEMANTIC_FIELD_SENTINEL" in p and "s[0]" in p and "distance to target" in p
    assert "ENV_DESCRIPTION_SENTINEL" in p, "the environment, as the generator sees it"
    assert "SUCCESS_CONDITION_SENTINEL" in p, "the success criteria (App. A p.12)"
    assert ctx.cfg["problem.task_description"] in p
    assert "rewards" not in p, "a critic must not rank the reward it is judging"
    assert _critic_key_hint(ctx) == ["states", "SEMANTIC_FIELD_SENTINEL", "other"]


def test_traj_dict_exposes_named_state_columns_and_no_reward_series():
    traj = Trajectory(states=np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
                      rewards=[5.0, 5.0, 5.0], component_values={"reach": [1, 1, 1]}, length=3)
    d = _traj_dict(traj, ["dist", "vel"])
    assert set(d) == {"states", "dist", "vel"}
    assert d["dist"].tolist() == [1.0, 2.0, 3.0] and d["vel"].tolist() == [10.0, 20.0, 30.0]
    assert d["states"].shape == (3, 2)
    # A width mismatch keeps `states` only rather than mislabel a column.
    assert set(_traj_dict(traj, ["a", "b", "c"])) == {"states"}


def test_a_real_adapter_states_its_success_conditions():
    registry.load_all()
    env = registry.get("env", "toy_reacher")({})
    text = env.describe_success()
    assert "0.2" in text, "the bar `success()` applies, since the spec states no prose"
    assert [f for f, _ in env._state_fields][:2] == ["x", "y"]


def test_the_mock_critic_labels_from_states():
    """The tester tier runs R* end to end on the mock LLM; its critic must read
    a key the critic actually receives, or the labelling path silently produces
    zero segments."""
    registry.load_all()
    cfg = load("rstar", profile="tester")
    ctx = Context(cfg=cfg, budget=Budget(), env=registry.get("env", cfg["problem.env_id"])({}),
                  rng=random.Random(0))
    ctx.generator = registry.get("llm", "mock")(ctx, role="generator")
    fns = [alignment._compile_critic(c, str(i))
           for i, c in enumerate(author_critics(ctx, RunState()))]
    a = _traj_dict(Trajectory(states=np.ones((4, 6)), length=4), [f for f, _ in ctx.env._state_fields])
    b = _traj_dict(Trajectory(states=np.zeros((4, 6)), length=4), [f for f, _ in ctx.env._state_fields])
    labels = [fn(a, b) for fn in fns if fn is not None]
    assert labels and all(len(list(l)) == 4 for l in labels)
    assert any(int(v) == 1 for l in labels for v in l)
