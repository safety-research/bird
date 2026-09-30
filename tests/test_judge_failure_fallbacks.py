"""Failed (V)LM judgments retry, then abstain -- they do not rank candidates by
their own reward or by ground-truth success.

Preference comparisons (GT): a comparison whose caption or verdict does not
parse must not fall through to `_decide_offline`, which ranks by
`(clip.success, clip.mean_per_step_return, id)` -- task success and each
program's OWN reward scale, neither of which is a VLM preference. A large
constant reward would win a failed judgment and its label would enter D_pref.

Subtask scores (RDA): a missing VLM subtask score must not be replaced by the
environment's success flag (`hi if traj.success else lo`) and averaged into
fitness, despite RDA's `problem.fitness_access: none`. A task-successful but
instruction-violating policy would score perfectly whenever judging failed.

Both: one re-query, then ABSTAIN under a real provider (the pair leaves
D_pref / the subtask leaves the mean), with a deterministic synthetic stand-in
kept only for `provider: mock`, where there is no real judge and the offline
suite must still run the path.
"""
from __future__ import annotations

import random

import pytest

import numpy as np

from bird import registry
from bird.budget import Budget
from bird.components.evaluation import _vlm_trajectory_analysis
from bird.components.preferences import ABSTAIN, _fallback_decision
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult, Trajectory


# ------------------------------------------------------ preference comparisons

def _pref_report(cid: str, ret: float) -> CandidateReport:
    traj = Trajectory(states=np.zeros((4, 1)), actions=np.zeros((3, 1)),
                      rewards=[ret / 3.0] * 3, success=True, length=3, ret=ret)
    cand = Candidate(cid, 0, "def reward(s, a, s2): return 0.0, {}")
    return CandidateReport(cid, cand, TrainResult(cid, cand, trajectories=[traj]), fitness=0.0)


def test_a_failed_judgment_abstains_under_a_real_provider():
    """Two reports identical but for return scale: the higher-return side must
    not win a failed judgment; both abstain."""
    registry.load_all()
    ctx = Context(cfg=load("gt", overrides={"llm.evaluator.provider": "anthropic"}),
                  budget=Budget(), rng=random.Random(0))
    events = []
    ctx.event = lambda name, **k: events.append((name, k))

    v_equal = _fallback_decision(ctx, _pref_report("L", 1.0), _pref_report("R", 1.0),
                                 role="pair_vlm", reason="unparseable")
    v_skewed = _fallback_decision(ctx, _pref_report("L", 100.0), _pref_report("R", 1.0),
                                  role="pair_vlm", reason="unparseable")

    assert v_equal == ABSTAIN and v_skewed == ABSTAIN, "reward magnitude must not decide"
    assert all(e[1]["fallback"] == "abstain" for e in events if e[0] == "judge_fallback")


def test_a_failed_judgment_keeps_a_deterministic_stand_in_under_mock():
    """The offline suite has no real judge to retry; the synthetic decision is
    confined to `mock` and is deterministic."""
    registry.load_all()
    ctx = Context(cfg=load("gt", profile="tester"), budget=Budget(), rng=random.Random(0))
    ctx.event = lambda *a, **k: None
    v = _fallback_decision(ctx, _pref_report("L", 1.0), _pref_report("R", 1.0),
                           role="pair_vlm", reason="x")
    assert v in (0, 1), "a real label, not ABSTAIN, under mock"


# ------------------------------------------------------------- subtask scores

def _analysis_ctx(provider: str):
    registry.load_all()
    cfg = load("rda", overrides={"llm.evaluator.provider": provider,
                                 "llm.generator.provider": provider})
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(0))

    class _Garbage:
        def __init__(self, prov):
            self.provider = prov

        def __call__(self, prompt, **kw):
            return ["there is no json here"]

    ctx.generator = _Garbage(provider)
    return ctx


def _traj(success: bool):
    return Trajectory(states=np.zeros((3, 6)), actions=np.zeros((2, 6)),
                      rewards=[0.0, 0.0], success=success, length=2)


def test_a_missing_subtask_score_abstains_under_a_real_provider():
    """Identical unparseable replies: a success-flag fallback would score a
    failed trajectory [0, 0] and a successful one [1, 1]. Both abstain (None),
    so no ground truth reaches fitness."""
    ctx = _analysis_ctx("anthropic")
    cand = Candidate("c", 0, "def reward(s, a, s2): return 0.0, {}")
    subs = ["reach", "grasp"]
    failed = _vlm_trajectory_analysis(ctx, _traj(False), subs, "", True,
                                      frames=None, prov={}, candidate=cand, trace=None)
    ok = _vlm_trajectory_analysis(ctx, _traj(True), subs, "", True,
                                  frames=None, prov={}, candidate=cand, trace=None)
    assert [x[0] for x in failed] == [None, None]
    assert [x[0] for x in ok] == [None, None], "a successful traj must NOT become [1, 1]"


def test_a_missing_subtask_score_keeps_a_synthetic_stand_in_under_mock():
    ctx = _analysis_ctx("mock")
    cand = Candidate("c", 0, "def reward(s, a, s2): return 0.0, {}")
    subs = ["reach", "grasp"]
    failed = _vlm_trajectory_analysis(ctx, _traj(False), subs, "", True,
                                      frames=None, prov={}, candidate=cand, trace=None)
    ok = _vlm_trajectory_analysis(ctx, _traj(True), subs, "", True,
                                  frames=None, prov={}, candidate=cand, trace=None)
    assert [x[0] for x in failed] == [0.0, 0.0]
    assert [x[0] for x in ok] == [1.0, 1.0], "the mock stand-in is the GT flag, recorded synthetic"


# ---------------------------------------------------------- budget accounting

def test_a_self_recording_judge_client_is_charged_once():
    """`_query` reserves a budget slot before the call (cap safety under
    concurrent judging) AND estimates tokens for a duck-typed callable that
    accounts for nothing. A real `LLMClient` records its own exact usage inside
    `_client_call`, so charging both would double-count every GT judge
    round-trip. The reservation is rolled back when the client recorded."""
    from bird.budget import Budget
    from bird.components.preferences import _query
    from bird.llm.base import LLMClient

    registry.load_all()

    class _Recording(LLMClient):
        provider = "rec"

        def _complete(self, messages, n, temperature, images, tag):
            self._record(10, 20)
            return ["LEFT"]

    ctx = Context(cfg=load("gt"), budget=Budget())
    ctx.evaluator = _Recording(ctx, "evaluator")
    _query(ctx, "judge this pair", vlm=True, role="pair_vlm")
    assert ctx.budget.llm_calls == 1, "one round-trip, one call"
    assert ctx.budget.vlm_calls == 1
    assert (ctx.budget.llm_prompt_tokens, ctx.budget.llm_completion_tokens) == (10, 20), \
        "the client's exact counts, not client + estimate"


def test_a_non_recording_judge_callable_is_still_charged_once():
    """The reservation estimate must still fire for a judge that records
    nothing, or a GT run with a bare callable would look free."""
    from bird.budget import Budget
    from bird.components.preferences import _query

    ctx = Context(cfg=load("gt"), budget=Budget())
    ctx.evaluator = lambda prompt, **kw: "RIGHT"
    _query(ctx, "judge me", vlm=True, role="pair_vlm")
    assert ctx.budget.llm_calls == 1
    assert ctx.budget.llm_prompt_tokens == len("judge me") // 4
    assert ctx.budget.llm_completion_tokens == len("RIGHT") // 4


# ------------------------------------------ preference comparisons, the tie door

def _tie_client():
    """A judge that answers TIE even though `allow_ties: false` told it not to."""
    class _Tie:
        modality = "vlm"

        def __init__(self):
            self.calls = 0

        def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
            self.calls += 1
            return "TIE"
    return _Tie()


def _trained_pair():
    reports = []
    for i in (1, 2):
        traj = Trajectory(states=np.zeros((4, 1)), actions=np.zeros((3, 1)),
                          rewards=[0.1 * i] * 3, success=True, length=3, ret=0.3 * i)
        cand = Candidate(f"c{i}", 0, "def compute_reward(state, action=None): return 0.0")
        res = TrainResult(f"c{i}", cand, trajectories=[traj], trained=True)
        reports.append(CandidateReport(f"c{i}", cand, res, fitness=float(i)))
    return reports


@pytest.mark.parametrize("provider,dropped", [("anthropic", True), ("mock", False)])
def test_a_leaked_tie_under_allow_ties_false_abstains_for_a_real_provider(provider, dropped):
    """`gt.yaml` pins `allow_ties: false`; resolving a judge that answers TIE
    anyway with `_decide_offline` would rank by the agents' own reward scale --
    the same fallback, through the tie door. A real provider ABSTAINS (the pair
    leaves D_pref);
    `provider: mock`, with no real judge, keeps the deterministic resolution."""
    from bird.components.preferences import run_preferences

    registry.load_all()
    cfg = load("gt", overrides={
        "llm.evaluator.provider": provider, "llm.generator.provider": provider,
        "output.tracker": "none", "problem.env_id": "pendulum",
        "evaluate.preferences.comparator": "vlm",
        "evaluate.preferences.store_dataset": True,
    }, profile="dev")
    assert cfg["evaluate.preferences.allow_ties"] is False
    ctx = Context(cfg=cfg, budget=Budget(), rundir=None, rng=random.Random(0))
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    ctx.evaluator = _tie_client()

    state = RunState()
    run_preferences(ctx, state, _trained_pair())

    if dropped:
        assert state.preferences == [], "a leaked tie must not enter D_pref under a real provider"
    else:
        assert state.preferences, "mock resolves the leaked tie deterministically, not dropped"
