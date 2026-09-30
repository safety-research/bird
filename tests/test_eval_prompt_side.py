"""Prompt-side evaluation checks.

Every check here is against text that reaches an LLM (the generator's
reflection, the judge's table, the analyzer's prompt) or a number a fitness is
built from (`_pearson`, `_parse_score`). Each probe reproduces one concrete
failure shape.
"""
from __future__ import annotations

import math
import random
from types import SimpleNamespace

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components import evaluation as E
from bird.components import generation as G
from bird.components import update as U
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult, Trajectory

_CODE = ("def compute_reward(s, a, s2):\n"
         "    return 1.0, {'reach': 1.0}\n")


def _cand(cid="c0001", weights=None):
    c = Candidate(cid, 0, _CODE)
    if weights:
        c.weights = dict(weights)
    return c


def _traj(n=10, *, success=True, obs_dim=3, comps=None, ret=None):
    """One rollout with ndarray states, the shape `training._rollout` stores."""
    return Trajectory(states=np.arange(n * obs_dim, dtype=float).reshape(n, obs_dim) / 10.0,
                      actions=np.zeros((n, 1)), rewards=[0.1] * n,
                      component_values=comps or {}, success=success, length=n,
                      ret=float(n * 0.1) if ret is None else ret)


def _report(cid="c0001", *, pruned=False, timed_out=False, trajs=None,
            weights=None, fitness=0.5, seed_metrics=None, checkpoints=None):
    cand = _cand(cid, weights)
    res = TrainResult(cand_id=cid, candidate=cand, pruned=pruned, timed_out=timed_out,
                      trajectories=list(trajs or []),
                      seed_metrics=list(seed_metrics or []),
                      checkpoints=list(checkpoints or []),
                      component_traces={"reach": [0.1, 0.2, 0.3, 0.4]})
    return CandidateReport(cid, cand, res, fitness=fitness, fitness_source="task_success",
                           per_seed_fitness=[fitness])


def _ns_ctx(cfg, env=None, evaluator=None):
    """A duck-typed context: `ctx.cfg.get`/`[]` over a plain dict, an env with
    the attributes the renderers read, and an evaluator that records its prompt."""
    return SimpleNamespace(cfg=dict(cfg), env=env if env is not None else SimpleNamespace(horizon=10),
                           evaluator=evaluator, human=None, rng=random.Random(0))


class _RecordingClient:
    def __init__(self, answer="The agent wanders."):
        self.prompts = []
        self.answer = answer

    def complete(self, messages, **kw):
        self.prompts.append(messages[0]["content"] if isinstance(messages, list) else messages)
        return self.answer


# ---------------------------------------------- a pruned curve is announced

_PRUNED_CKPTS = [{"round": r, "fitness": 0.05 * r, "reward_return": 0.1 * r} for r in range(1, 9)]


def _pruned_report(**kw):
    return _report(pruned=True, seed_metrics=[{"pruned_at_round": 8}],
                   checkpoints=_PRUNED_CKPTS, trajs=[_traj()], **kw)


@pytest.mark.parametrize("metric, names", [
    ("own_reward", ("own reward", "own_reward")),
    ("task_metric", ("task metric", "task_metric")),
])
def test_a_pruned_winner_is_announced_in_the_numeric_section(metric, names):
    """The NUMERIC REFLECTION rendition (`_section_numeric`) must not show a
    pruned curve under 'Training statistics, sampled at checkpoints:' as if it
    were complete; a caveat that lived only in `_behavioural_analysis`, which
    eureka never renders, would never reach eureka's prompt. The note names the
    metric `train.pruning_metric` actually watched, not 'its own learning
    curve' regardless."""
    ctx = _ns_ctx({"evaluate.feedback.granularity": "per_component",
                   "evaluate.feedback.include_task_metric": True,
                   "evaluate.fitness.metric": "task_success",
                   "train.pruning_metric": metric})
    text = E._section_numeric(ctx, _pruned_report())
    assert "STOPPED EARLY" in text, text
    assert "checkpoint 8" in text
    assert any(n in text for n in names), (metric, text)
    assert text.index("STOPPED EARLY") < text.index("Training statistics"), \
        "the caveat must come BEFORE the curve it qualifies"


def test_the_numeric_caveat_does_not_claim_the_wrong_metric():
    """`own_reward` must not be described as the task metric and vice versa."""
    base = {"evaluate.feedback.granularity": "per_component",
            "evaluate.fitness.metric": "task_success"}
    own = E._section_numeric(_ns_ctx({**base, "train.pruning_metric": "own_reward"}),
                             _pruned_report())
    task = E._section_numeric(_ns_ctx({**base, "train.pruning_metric": "task_metric"}),
                              _pruned_report())
    assert "task metric" not in own.split("Training statistics")[0]
    assert "own reward" not in task.split("Training statistics")[0]


def test_a_timed_out_winner_is_announced_in_the_numeric_section():
    ctx = _ns_ctx({"evaluate.feedback.granularity": "per_component",
                   "evaluate.fitness.metric": "task_success"})
    rep = _report(timed_out=True, checkpoints=_PRUNED_CKPTS, trajs=[_traj()])
    text = E._section_numeric(ctx, rep)
    assert "wall-clock" in text and "truncated" in text, text


def test_a_completed_winner_carries_no_caveat():
    ctx = _ns_ctx({"evaluate.feedback.granularity": "per_component",
                   "evaluate.fitness.metric": "task_success"})
    rep = _report(checkpoints=_PRUNED_CKPTS, trajs=[_traj()])
    text = E._section_numeric(ctx, rep)
    assert text and "STOPPED EARLY" not in text and "wall-clock" not in text


def _real_ctx(name, profile="tester", overrides=None):
    registry.load_all()
    cfg = load(name, profile=profile, overrides=overrides)
    env = registry.get("env", cfg["problem.env_id"])({})
    return Context(cfg=cfg, budget=Budget(), env=env, rng=random.Random(0))


def test_the_caveat_reaches_the_next_prompt_and_the_carried_turn_exactly_once():
    """The probe end to end: `configs/methods/eureka` pruned under the tester
    profile (`train.pruning: median_stop`, `pruning_metric: task_metric`). The
    NEXT iteration's prompt -- the NUMERIC REFLECTION section AND the dialogue
    turn §6 carries -- must say the winner was stopped early, and say it once
    per rendition: a method that renders both the numeric and the behavioural
    channel must not announce it twice."""
    ctx = _real_ctx("eureka", overrides={"train.pruning": "median_stop",
                                         "train.pruning_metric": "task_metric"})
    assert ctx.cfg["train.pruning"] != "none"
    rep = _pruned_report()
    state = RunState(iteration=1)
    prose, channel = E.feedback_default(ctx, state, rep)
    rep.feedback, rep.feedback_channel = prose, channel
    state.best = rep

    assert "STOPPED EARLY" in rep.meta.get("numeric_reflection", ""), \
        "the single source of truth for §1's NUMERIC REFLECTION must carry it"
    section = G._numeric_reflection(ctx, rep)
    assert section.count("STOPPED EARLY") == 1, section
    assert "task metric" in section, "this arm watches task_metric; the note must say so"

    msgs = _build_messages(ctx, state, [rep])
    prompt = "\n".join(m["content"] for m in msgs)
    assert "STOPPED EARLY" in prompt
    # One announcement per rendition in the prompt: the numeric section owns it.
    assert prompt.count("STOPPED EARLY") == 1, prompt

    turns = U._turns(ctx, state, rep)
    user = [t for t in turns if t["role"] == "user"][0]["content"]
    assert user.count("STOPPED EARLY") == 1, user


def _build_messages(ctx, state, parents):
    return G._build_messages(ctx, state, parents)


def test_behavioural_analysis_still_carries_the_caveat_when_there_is_no_numeric_channel():
    """RDA's shape: `numeric_reflection: false`, subtask scores. The caveat's only
    home is the behavioural section, and it must stay there."""
    ctx = _ns_ctx({"evaluate.feedback.numeric_reflection": False,
                   "train.pruning_metric": "own_reward"})
    rep = _pruned_report()
    rep.subtask_scores = {"reach": 0.2}
    rep.subtask_rationales = {"reach": "never got close"}
    text = G._behavioural_analysis(ctx, rep)
    assert text.count("STOPPED EARLY") == 1, text


# ------------------------------------------------ the parent-section lead

def _parent_ctx():
    # `_parent_section` and `_parent_group` hand each parent's program through
    # `_inheritable_program` (which honours `update.co_evolve.observation_fn`),
    # which reads BOTH observation keys with `[]` -- a declared key
    # every resolved config carries. This duck-typed dict has to carry them
    # too, at their `configs/_default.yaml` values (both false): with §1's
    # elicitation off the program is handed on verbatim, so these tests pin
    # the LEAD wording and not the observation-stripping path.
    return _ns_ctx({"generate.context.include_parent_thought": False,
                    "evaluate.feedback.state_selection_scalar": True,
                    "generate.co_design.observation_fn": False,
                    "update.co_evolve.observation_fn": False})


def test_one_parent_with_a_weights_dict_is_introduced_as_one_parent():
    """A parent under `output.format: component_dict_plus_weights` renders TWO
    blocks (code + weights), so a lead picked by `len(blocks) == 1` would make
    every RDA/GT iteration>0 prompt say 'These are the parent rewards to
    combine and improve on' over a single reward -- crossover wording for a
    `single_parent_hillclimb` operator (App. 7.5: 'introducing one
    modification')."""
    rep = _report(weights={"reach": 1.0, "grasp": 0.5})
    text = G._parent_section(_parent_ctx(), [rep])
    assert text.startswith("This is the reward to improve on:"), text
    assert "combine" not in text
    assert "weights:" in text, "the weights block itself must survive"
    assert text.count("Reward c0001") == 1


def test_two_parents_keep_the_group_wording():
    reps = [_report("c0001", weights={"a": 1.0}), _report("c0002", weights={"a": 2.0})]
    text = G._parent_section(_parent_ctx(), reps)
    assert text.startswith("These are the parent rewards to combine and improve on:"), text


# ---------------------------------------------- non-finite values mask pairs

def test_pearson_masks_pairs_not_elements():
    """Dropping each series' non-finite entries independently and then pairing
    by index would let one NaN misalign every later pair. Both probes are r=1
    over the finite PAIRS."""
    nan = float("nan")
    assert E._pearson([1.0, nan, 2.0, 3.0, 4.0], [1.0, 5.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert E._pearson([1, 2, 3, 4, 5], [1, nan, 3, 4, 5]) == pytest.approx(1.0)


def test_pearson_with_a_nan_equals_pearson_of_the_finite_pairs():
    rng = np.random.default_rng(3)
    a = list(rng.normal(size=12))
    b = list(0.6 * np.asarray(a) + rng.normal(scale=0.5, size=12))
    k = 4
    a_nan = list(a)
    a_nan[k] = float("nan")
    keep = [i for i in range(12) if i != k]
    expected = float(np.corrcoef([a[i] for i in keep], [b[i] for i in keep])[0, 1])
    assert E._pearson(a_nan, b) == pytest.approx(expected, abs=1e-12)
    # Symmetric: a NaN on the OTHER side masks the same pair.
    b_nan = list(b)
    b_nan[k] = float("inf")
    assert E._pearson(a, b_nan) == pytest.approx(expected, abs=1e-12)


def test_pearson_still_refuses_too_few_pairs():
    assert E._pearson([1.0, float("nan"), 2.0], [1.0, 2.0, float("nan")]) is None


def test_pseudometric_pairs_on_the_same_transition():
    """The same holds in `_pseudometric`. A candidate whose per-step reward is
    NaN at one step and otherwise IDENTICAL to the reference has EPIC distance
    0 over the finite pairs; positional pairing after a one-sided drop would
    make it positive."""
    nan = float("nan")
    t = Trajectory(states=None, actions=None, rewards=[1.0, nan, 2.0, 3.0, 4.0, 2.5],
                   component_values={"gt_reward": [1.0, 5.0, 2.0, 3.0, 4.0, 2.5]},
                   length=6, ret=12.5)
    rep = _report(trajs=[t])
    ctx = _ns_ctx({})
    d = E._pseudometric(ctx, RunState(iteration=0), rep, "epic")
    assert d == pytest.approx(0.0, abs=1e-9), d


# ----------------------------------------------------------- score parsing

@pytest.mark.parametrize("text, lo, hi, want", [
    # the two probes: a likert fraction, and an incidental step fraction
    ("Score: 4/5", 1.0, 5.0, 4.0),
    ("At step 12/50 the arm moved; score 0.8", 0.0, 1.0, 0.8),
    # the ordinary parses
    ("0.8", 0.0, 1.0, 0.8),
    ("Score: 4/5", 0.0, 1.0, 0.8),
    ("8/10", 0.0, 1.0, 0.8),
    ("Score: 0.5, reason: it stalled", 0.0, 1.0, 0.5),
    ("4", 1.0, 5.0, 4.0),
    ("I would rate this 3 out of the five", 1.0, 5.0, 3.0),
    ("5/5", 1.0, 5.0, 5.0),
    ("1/1", 0.0, 1.0, 1.0),
    ("rating: 2", 1.0, 5.0, 2.0),
    # an explicit score beats an incidental fraction on either scale
    ("Reached step 12/50. Score: 4", 1.0, 5.0, 4.0),
    ("Reached step 12/50, then stopped. Score 0.9", 0.0, 1.0, 0.9),
    # a JSON answer (the mock judge's shape): the key anchors, the earlier field does not
    ('{"rollout": 1, "score": 4, "reason": "ok"}', 1.0, 5.0, 4.0),
])
def test_parse_score_reads_the_stated_score(text, lo, hi, want):
    assert E._parse_score(text, lo, hi) == pytest.approx(want)


@pytest.mark.parametrize("text, lo, hi", [
    ("", 0.0, 1.0),
    ("no number here", 0.0, 1.0),
    ("Score: 7/10", 1.0, 5.0),   # a foreign denominator on likert is not 3.8
    ("Score: 0/5", 1.0, 5.0),    # 0 is off a 1-5 scale, and the '5' must not be read as the score
    ("Score: 12/50", 1.0, 5.0),
])
def test_parse_score_never_invents_a_number(text, lo, hi):
    assert E._parse_score(text, lo, hi) is None


# -------------------------------------- the task-metric gate on the analyzer

def _analyzer_cfg(show):
    # `problem.env_id` is load-bearing here: every renderer that can name the
    # success flag also asks whether the BENCHMARK ships one
    # (`_show_env_success`), and an unnamed env resolves to no spec -> no
    # native success -> suppressed for a reason that has nothing to do with
    # this gate. A Meta-World task (discrete_success.kind: discrete) keeps
    # `include_task_metric` the only gate under test.
    return {"problem.env_id": "mt10_reach-v3",
            "evaluate.feedback.include_task_metric": show,
            "evaluate.feedback.analyzer": "llm",
            "evaluate.rollouts_per_candidate": 3,
            "evaluate.feedback.trajectory_sample_interval": 10,
            "evaluate.feedback.trajectory_examples": "best_and_worst",
            "problem.task_description": "reach the goal"}


def test_include_task_metric_false_hides_success_from_the_analyzer_prompt():
    """The gate must be read by the analyzer too, not only by
    `_section_numeric`: the analyzer's no-caption branch renders
    `render_trajectory(worst, ...)`, whose header is `success=<flag>` and whose
    closing sentence is 'the agent solves the task'. Ungated, GT
    (`fitness_access: none`, `include_task_metric: false`, `analyzer: llm`)
    would hand the evaluator LLM the ground-truth flag on that branch."""
    client = _RecordingClient()
    ctx = _ns_ctx(_analyzer_cfg(False), evaluator=client)
    rep = _report(trajs=[_traj(success=True), _traj(success=True, ret=0.5)])
    out = E._section_analyzer(ctx, rep)
    assert out, "the analyzer must still run"
    assert client.prompts, "the evaluator was not asked"
    prompt = client.prompts[0]
    assert "success=" not in prompt, prompt
    assert "solves the task" not in prompt and "has not solved" not in prompt, prompt
    assert "t=0" in prompt, "the per-step rendering itself must survive the gate"


def test_include_task_metric_true_keeps_the_flag_in_the_analyzer_prompt():
    client = _RecordingClient()
    ctx = _ns_ctx(_analyzer_cfg(True), evaluator=client)
    E._section_analyzer(ctx, _report(trajs=[_traj(success=True)]))
    assert "success=True" in client.prompts[0]
    assert "solves the task" in client.prompts[0]


def test_include_task_metric_false_hides_success_from_rendered_examples_and_digests():
    rep = _report(trajs=[_traj(success=True), _traj(success=False, ret=0.2)])
    hidden = _ns_ctx(_analyzer_cfg(False))
    shown = _ns_ctx(_analyzer_cfg(True))
    block = E._section_trajectories(hidden, rep)
    assert block and "success=" not in block and "solves the task" not in block, block
    assert "success=" in E._section_trajectories(shown, rep)
    assert "success=" not in E._behaviour_digest(hidden, rep.result)
    assert "success=" in E._behaviour_digest(shown, rep.result)


# ------------------------------------------ the judge's component table

def _many_components(k, n=100):
    names = [f"c{i:02d}" for i in range(k - 1)] + ["success_bonus"]
    return names, {name: [float(i)] * n for i, name in enumerate(names)}


def test_the_judge_table_renders_every_component_of_a_twelve_component_reward():
    """`_traj_component_table` must not slice `[:8]` silently: real RDA rewards
    carry 7-20 components and the tail (success bonus, penalties) is exactly
    what App. 7.3's 'one component overshadowing others' diagnosis needs."""
    names, comps = _many_components(12)
    t = Trajectory(states=[], actions=[], rewards=[0.0] * 100, length=100,
                   component_values=comps)
    out = E._traj_component_table(t, [0, 50, 99])
    shown = [k for k in names if f"{k}=" in out]
    assert shown == names, (len(shown), out)
    assert "success_bonus=" in out
    assert "showing" not in out, "nothing omitted, so no omission notice"


def test_the_judge_table_says_n_of_m_when_its_ceiling_binds():
    names, comps = _many_components(40)
    t = Trajectory(states=[], actions=[], rewards=[0.0] * 100, length=100,
                   component_values=comps)
    out = E._traj_component_table(t, [0, 99])
    shown = [k for k in names if f"{k}=" in out]
    assert 8 < len(shown) < 40, len(shown)
    assert f"showing the first {len(shown)} of 40" in out, out


# ----------------------------------------- observation columns (CARD §4.2.2)

def test_an_ndarray_trajectory_renders_named_observation_columns():
    """`_state_vars` must render observation variables for array states, not
    only dict-shaped ones: every rollout stores `np.asarray(states)`, so a
    dict-only renderer leaves CARD's §4.2.2 'observation parameters' channel
    empty on every backend."""
    t = _traj(n=10, obs_dim=3)
    out = E.render_trajectory(t, "best", stride=5, state_fields=("hand_x", "hand_y", "goal_x"))
    assert "hand_x=" in out and "goal_x=" in out, out
    # The value is the stored observation, at the rendered step, in field order.
    row = [ln for ln in out.splitlines() if ln.strip().startswith("t=4 ")][0]
    assert "hand_y=" + E._fmt(float(t.states[4][1])) in row, row
    assert "showing" not in out and " of 3" not in out


def test_observation_columns_are_capped_with_an_n_of_m_legend():
    fields = tuple(f"f{i}" for i in range(12))
    t = _traj(n=6, obs_dim=12)
    out = E.render_trajectory(t, "worst", stride=3, state_fields=fields)
    assert "f0=" in out and "f7=" in out
    assert "f8=" not in out and "f11=" not in out
    assert "8 of 12" in out, out


def test_the_trajectory_section_names_the_env_declared_fields():
    """Through the section builder: the adapter's `_state_fields` legend (what
    `_traj_state_table` already uses for the judge) names the columns."""
    env = SimpleNamespace(horizon=10, _state_fields=(("hand_x", "x of the hand"),
                                                     ("hand_y", "y"), ("goal_x", "x of the goal")))
    ctx = _ns_ctx(_analyzer_cfg(True), env=env)
    rep = _report(trajs=[_traj(success=False), _traj(success=True, ret=2.0)])
    block = E._section_trajectories(ctx, rep)
    assert "hand_x=" in block and "goal_x=" in block, block


def test_a_dict_shaped_state_still_renders_as_before():
    t = Trajectory(states={"x": [0.1, 0.2, 0.3], "y": [1.0, 2.0, 3.0]}, actions=None,
                   rewards=[0.0, 0.0, 0.0], length=3)
    assert E._state_vars(t, 1) == [("x", 0.2), ("y", 2.0)]
    assert E._state_vars(t, 1, ("ignored",)) == [("x", 0.2), ("y", 2.0)]


def test_no_declared_fields_means_no_anonymous_columns():
    """An anonymous array is not printed as s[0]..s[173]."""
    t = _traj(n=4, obs_dim=5)
    assert E._state_vars(t, 0) == []
    assert "s[0]" not in E.render_trajectory(t, "r", stride=2)
