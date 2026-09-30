"""`parent_source: puct_leaf` picks the child the docstring's arithmetic says.

Why a test and not a look: the failure is silent. A wrong argmax among children
with EQUAL Q -- the prior deciding, or the lowest index on a full tie -- renders
as a perfectly plausible run with a different parent, and every fitness in the
run dir stays identical (the same reason `select.tie_break: first` is pinned).
The numbers are the hand-checked worked examples in
`tree.py::parent_puct_leaf`; `uct_leaf` on the same tree is pinned beside them
so a change to the shared descent shows up in both.
"""

from __future__ import annotations

import random

import pytest

from bird.budget import Budget
from bird.components import tree
from bird.config import Config
from bird.context import Context
from bird.state import RunState, TreeNode
from bird.types import Candidate, CandidateReport, TrainResult


def _ctx(*, lam: float = 0.4, verify: bool) -> Context:
    cfg = Config({
        "generate": {"tree": {"uct_lambda0": lam, "uct_lambda_final": lam,
                              "horizon_trainings": None, "max_depth": None}},
        "evaluate": {"self_verify": {"enabled": verify, "range": [-1, 1]}},
    })
    return Context(cfg=cfg, budget=Budget(), rundir=None, rng=random.Random(0))


def _state(children) -> RunState:
    """Root with the given `(id, q, self_verify)` children, each visits 1."""
    st = RunState()
    st.tree[tree.ROOT] = TreeNode(node_id=tree.ROOT, parent_id=None, depth=0,
                                  action="root", visits=len(children))
    for i, (cid, q, sv) in enumerate(children, start=1):
        st.tree[cid] = TreeNode(node_id=cid, parent_id=tree.ROOT, depth=1, action="init",
                                iteration=0, score=q, q=q, visits=1, total=q,
                                self_verify=sv, sim_index=i)
        st.tree[tree.ROOT].children.append(cid)
        cand = Candidate(cand_id=cid, iteration=0, reward_code="")
        st.all_reports.append(CandidateReport(
            cand_id=cid, candidate=cand, result=TrainResult(cand_id=cid, candidate=cand),
            fitness=q))
    return st


def test_puct_worked_example_matches_the_docstring():
    st = _state([("A", 0.5, 0.5), ("B", 0.0, -0.5)])
    d = tree._DescentInputs(_ctx(verify=True), st)
    scores, rows = tree._score_puct(st.tree[tree.ROOT], [st.tree["A"], st.tree["B"]], d)
    assert scores == pytest.approx([1.206775, 0.076068], abs=1e-6)
    assert [r["prior"] for r in rows] == pytest.approx([0.731059, 0.268941], abs=1e-6)
    assert [r["explore"] for r in rows] == pytest.approx([0.707107, 0.707107], abs=1e-6)
    # and uct_leaf on the same tree, the docstring's other example
    uct, _ = tree._score_uct(st.tree[tree.ROOT], [st.tree["A"], st.tree["B"]], d)
    assert uct == pytest.approx([1.885345, 0.700498], abs=1e-6)
    assert [r.cand_id for r in tree.parent_puct_leaf(_ctx(verify=True), st)] == ["A"]


def test_equal_q_the_prior_decides_and_without_it_the_lowest_index():
    st = _state([("A", 0.5, -0.5), ("B", 0.5, 0.5)])
    assert [r.cand_id for r in tree.parent_puct_leaf(_ctx(verify=True), st)] == ["B"]
    # self-verify off: the prior is uniform, both score 1.141421, strict argmax keeps A
    scores, rows = tree._score_puct(
        st.tree[tree.ROOT], [st.tree["A"], st.tree["B"]], tree._DescentInputs(_ctx(verify=False), st))
    assert scores == pytest.approx([1.141421, 1.141421], abs=1e-6)
    assert [r["prior"] for r in rows] == [0.5, 0.5]
    assert [r.cand_id for r in tree.parent_puct_leaf(_ctx(verify=False), st)] == ["A"]


def test_empty_tree_is_the_init_round():
    assert tree.parent_puct_leaf(_ctx(verify=True), RunState()) == []
