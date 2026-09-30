"""`select.rule: pareto` honours the module invariant: a report with
`fitness is None` never outranks one that has a fitness.

selection.py states the invariant for every rule and the pareto docstring
promised that a candidate "cannot join the front by declining to produce
evidence". A rule that contests the whole `_contest_pool` and maps a missing
fitness to -inf on that axis ONLY breaks this, because -inf on one axis does not
dominate a report that is best on another -- so under
`objectives: [fitness, -reward_ast_nodes]` an unjudged one-line reward would be
non-dominated, join the front, and with `n_survivors: 1` + `tie_break: first`
win by index, become `state.latest` through `become_parent` and parent the
next round while `_update_global_best` refused it. Mixed None/scored pools are
reachable: `fitness.source: human_score` past `queries_per_iteration` and
`preference_bt` with no recorded strength both leave None.

No shipped config selects `pareto`; this pins a config-space point, not a
shipped one.
"""

from __future__ import annotations

import pytest

from bird import registry
from bird.budget import Budget
from bird.components import selection as S
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult

SHORT = "def compute_reward(x):\n    return 0.0\n"
LONG = ("def compute_reward(x):\n    a = x[0] * 2 + x[1]\n"
        "    b = a ** 2 - 3 * x[2]\n    return a + b + 1.0\n")


def _rep(cid, fitness, code):
    cand = Candidate(cand_id=cid, iteration=0, reward_code=code)
    res = TrainResult(cand_id=cid, candidate=cand, trained=True)
    return CandidateReport(cand_id=cid, candidate=cand, result=res, fitness=fitness,
                           fitness_source="human_score")


def _ctx(objectives, **overrides):
    registry.load_all()
    cfg = load("eureka", profile="tester", overrides={
        "select.rule": "pareto", "select.objectives": list(objectives),
        "select.n_survivors": 1, "select.tie_break": "first", **overrides})
    return Context(cfg=cfg, budget=Budget(), rundir=None)


def test_a_report_with_no_fitness_cannot_join_the_front_by_being_shortest():
    """The failure shape: None fitness, the shortest program, lowest index."""
    ctx = _ctx(["fitness", "-reward_ast_nodes"])
    reports = [_rep("c0000", None, SHORT), _rep("c0001", 0.9, LONG)]
    sel = S.rule_pareto(ctx, RunState(), reports)
    assert [w.cand_id for w in sel.winners] == ["c0001"]
    assert sel.winner.fitness == pytest.approx(0.9)
    assert not sel.tie_broken, "one scored report is a front of one, not a tie"
    assert "1 unscored candidate(s) excluded from the front" in sel.notes
    assert [r.cand_id for r in sel.losers] == ["c0000"]


def test_the_exclusion_holds_whatever_the_index_order():
    ctx = _ctx(["fitness", "-reward_ast_nodes"])
    reports = [_rep("c0000", 0.9, LONG), _rep("c0001", None, SHORT)]
    sel = S.rule_pareto(ctx, RunState(), reports)
    assert sel.winner.cand_id == "c0000" and sel.winner.fitness is not None


def test_scored_reports_still_form_a_genuine_front():
    """Control: with every report scored the rule is unchanged -- two
    non-dominated reports, cut by `tie_break: first`, recorded as a tie."""
    ctx = _ctx(["fitness", "-reward_ast_nodes"])
    reports = [_rep("c0000", 0.8, SHORT), _rep("c0001", 0.9, LONG)]
    # c0000: worse fitness, fewer nodes; c0001: better fitness, more nodes
    sel = S.rule_pareto(ctx, RunState(), reports)
    assert sel.tie_broken and sorted(sel.tied_ids) == ["c0000", "c0001"]
    assert sel.winner.cand_id == "c0000"
    assert "unscored" not in sel.notes


def test_a_pool_with_no_fitness_at_all_selects_nobody():
    """`None` stays distinct from `0.0`: nothing to rank is an empty selection
    with a note, never a winner picked among the unranked."""
    ctx = _ctx(["fitness", "-reward_ast_nodes"])
    reports = [_rep("c0000", None, SHORT), _rep("c0001", None, LONG)]
    sel = S.rule_pareto(ctx, RunState(), reports)
    assert sel.winners == [] and sel.winner is None
    assert "no candidate carried a fitness" in sel.notes
    assert len(sel.losers) == 2


def test_every_fitness_ranking_rule_refuses_the_unscored(monkeypatch):
    """The invariant is the module's, not one rule's: the same pool through
    `argmax_fitness` and `pareto` agrees on who is excluded."""
    reports = [_rep("c0000", None, SHORT), _rep("c0001", 0.9, LONG)]
    for rule in ("argmax_fitness", "pareto"):
        ctx = _ctx(["fitness", "-reward_ast_nodes"], **{"select.rule": rule})
        sel = registry.get("select_rule", rule)(ctx, RunState(), reports)
        assert sel.winner.cand_id == "c0001", rule
        assert "1 unscored candidate(s) excluded" in sel.notes, rule
