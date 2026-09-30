"""LaRes trains on a real reward evolution and does not crash on continuous
actions.

THE ELITE REACHES THE PROMPT. `configs/methods/lares.yaml` extends `zeroshot.yaml`; with
the prompt context channels off, `parent_source: global_best` selects an elite
that never enters the prompt, and every "evolved" reward is a fresh zero-shot
proposal from the same task, byte-identical to round 0. The release appends the elite's
response and a feedback block (win_rate + reward stats) to the next round's
messages (`LaRes_from_scratch.py:203-204`, `:645-658`). Enabled here via
`include_parent_code`, `include_numeric_reflection` and
`evaluate.feedback.numeric_reflection` (the last is LaRes's feedback channel;
`fitness.source: success_rate` makes the reflection's fitness line the elite's
win rate).

CONTINUOUS ACTIONS. `population.scaling_elite_moments` must not index
`env.action_set` with the replay's action, which for a continuous env is an
action VECTOR: that raises IndexError before SAC runs and fails every
next-round candidate setup on MetaWorld. A continuous action is already the value the reward wants; only a
discrete (scalar) index maps through `action_set`.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components.generation import _build_messages
from bird.components.population import apply_reward_scaling, scaling_elite_moments
from bird.components.training import _ReplaySlice, _REPLAY_STORE, compile_reward
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult


def _ctx(profile="tester", env=None):
    registry.load_all()
    cfg = load("lares", profile=profile)
    e = env if env is not None else registry.get("env", cfg["problem.env_id"])({})
    return Context(cfg=cfg, budget=Budget(), env=e, rng=random.Random(0))


def _elite_report():
    elite = Candidate("elite", 0,
                      "def compute_reward(s, a, s2):\n    ELITE_CODE_SENTINEL = 1.0\n"
                      "    return ELITE_CODE_SENTINEL, {}")
    rep = CandidateReport("elite", elite,
                          TrainResult("elite", elite,
                                      component_traces={"FEEDBACK_SENTINEL": [0.1, 0.2, 0.3]}),
                          fitness=0.42, fitness_source="success_rate", per_seed_fitness=[0.42])
    return rep


# ------------------------------------------------------ the elite reaches the prompt

def test_lares_pins_the_elite_and_reflection_channels():
    cfg = load("lares")
    assert cfg["generate.parent_source"] == "global_best"
    assert cfg["generate.context.include_parent_code"] is True
    assert cfg["generate.context.include_numeric_reflection"] is True
    assert cfg["evaluate.feedback.numeric_reflection"] is True, "LaRes numeric reflection must render"
    assert cfg["evaluate.fitness.source"] == "success_rate", "so the fitness line is win_rate"


def test_the_second_round_prompt_carries_the_elite_and_its_feedback():
    """The probe: with the elite selected but never rendered, round 1 == round 0
    and neither sentinel appears."""
    ctx = _ctx()
    rep = _elite_report()
    rep.meta["numeric_reflection"] = "FEEDBACK_SENTINEL: win_rate 0.42, reward stats ..."
    state = RunState(iteration=1)
    state.best = rep

    round0 = _build_messages(ctx, RunState(iteration=0), [])
    round1 = _build_messages(ctx, state, [rep])

    assert round1 != round0, "the evolved prompt must differ from the zero-shot one"
    assert "ELITE_CODE_SENTINEL" in str(round1), "the elite's code (include_parent_code)"
    assert "FEEDBACK_SENTINEL" in str(round1), "the elite feedback (numeric reflection)"


# ------------------------------------------------------------- continuous actions

def _continuous_env():
    return SimpleNamespace(action_set=np.array([[-1.0], [0.0], [1.0]]))


def test_continuous_replay_actions_do_not_crash_reward_scaling():
    """The probe: a continuous `_ReplaySlice` and an env-shaped `action_set`, on
    which indexing by the action raises IndexError before SAC runs."""
    ctx = _ctx(env=_continuous_env())
    _REPLAY_STORE["replay:cont"] = _ReplaySlice(
        np.array([[0.0], [1.0], [2.0]]), np.array([[0.25], [0.50], [0.75]]),
        np.array([[1.0], [2.0], [3.0]]), np.array([False, False, False]))
    state = RunState(iteration=1)
    state.replay_ref = "replay:cont"
    state.best = _elite_report()
    new = Candidate("new", 1, "def reward(s, a, s2): return 2.0 * float(s[0]) + float(a[0]), {}")

    # Stage 3 PLANS in the parent, the backend APPLIES (`train.reward_scaling`
    # is a planner): a decode failure would
    # surface here as a skip with `applied: False`, never as a raise.
    plan = scaling_elite_moments(ctx, state, new)
    assert plan["applied"], f"a usable scaled reward, not a skip: {plan}"
    out = apply_reward_scaling(new, compile_reward(new.reward_code, new))

    assert out is not None, "a usable scaled reward, not a skip"
    # The scaled reward evaluates on a continuous action without raising.
    val = out(np.array([1.0]), np.array([0.5]), np.array([2.0]))
    assert np.isfinite(float(val[0] if isinstance(val, tuple) else val))


def test_discrete_replay_indices_still_map_through_the_action_set():
    """A scalar index must still resolve through `action_set` -- the continuous
    branch covers vectors only, it does not drop the discrete path."""
    ctx = _ctx(env=SimpleNamespace(action_set=np.array([[-1.0], [0.0], [1.0]])))
    _REPLAY_STORE["replay:disc"] = _ReplaySlice(
        np.array([[0.0], [1.0], [2.0]]), np.array([0, 1, 2]),
        np.array([[1.0], [2.0], [3.0]]), np.array([False, False, False]))
    state = RunState(iteration=1)
    state.replay_ref = "replay:disc"
    state.best = _elite_report()
    new = Candidate("new", 1, "def reward(s, a, s2): return 2.0 * float(s[0]) + float(a[0]), {}")

    plan = scaling_elite_moments(ctx, state, new)
    assert plan["applied"], f"a usable scaled reward, not a skip: {plan}"
    out = apply_reward_scaling(new, compile_reward(new.reward_code, new))
    assert out is not None
