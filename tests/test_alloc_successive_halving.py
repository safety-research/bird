"""`select.allocation: successive_halving` spends exactly the budget it claims.

Its docstring: "The split preserves the total budget `n_trainable *
seeds_per_candidate`". Computing the top half's share as
`(n*s - n_lo*lo_seeds) // n_hi` drops the remainder of that floor division, so
every odd `n` where the freed seeds do not divide over the top half would train
fewer seeds than `uniform` -- 3 candidates at 2 seeds would run 5 of 6, 7 at 2
would run 11 of 14 -- while `policy_trainings` counted honestly and the
docstring did not. A `uniform` vs `successive_halving` ablation is budget-matched only
if this equality holds, so it is pinned here as an EQUALITY over a grid of
shapes, never a floor. The sibling allocators (`bandit`,
`bayesian_experimental_design`) already sum exactly through
`_largest_remainder`; this test holds all four to the same line.

No shipped config selects this allocator; the point is in the schema.
"""

from __future__ import annotations

import random
import types

import pytest

from bird import registry
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult

_CODE = "def compute_reward(x):\n    return 0.0\n"


def _state(n: int) -> RunState:
    """`n` parent lineages with strictly decreasing prior fitness, so the
    priors discriminate and the uniform early-return is not taken."""
    st = RunState()
    for i in range(n):
        pid = f"p{i}"
        cand = Candidate(cand_id=pid, iteration=0, reward_code=_CODE)
        res = TrainResult(cand_id=pid, candidate=cand, trained=True)
        st.all_reports.append(CandidateReport(cand_id=pid, candidate=cand, result=res,
                                              fitness=float(n - i)))
    return st


def _cands(n: int):
    return [Candidate(cand_id=f"c{i}", iteration=1, reward_code=_CODE, parent_id=f"p{i}")
            for i in range(n)]


def _ctx(s: int):
    registry.load_all()
    return types.SimpleNamespace(cfg={"train.seeds_per_candidate": s}, rng=random.Random(0),
                                 event=lambda *a, **k: None)


@pytest.mark.parametrize("n", range(2, 9))
@pytest.mark.parametrize("s", range(2, 7))
def test_the_plan_spends_exactly_the_uniform_budget(n, s):
    sh = registry.get("allocation", "successive_halving")(_ctx(s), _state(n), _cands(n))
    uni = registry.get("allocation", "uniform")(_ctx(s), _state(n), _cands(n))
    assert sum(uni.values()) == n * s
    assert sum(sh.values()) == n * s, f"n={n} s={s}: {sh}"
    assert min(sh.values()) >= 1


@pytest.mark.parametrize("n,s", [(3, 2), (5, 2), (7, 2), (3, 5), (7, 6)])
def test_the_top_half_gets_more_and_the_remainder_goes_to_the_best_ranked(n, s):
    """The shapes a plain floor division under-spends on. c0 has the best
    lineage prior, so it is ranked first and takes the first spare seed."""
    plan = registry.get("allocation", "successive_halving")(_ctx(s), _state(n), _cands(n))
    seeds = [plan[f"c{i}"] for i in range(n)]
    n_hi = (n + 1) // 2
    assert seeds == sorted(seeds, reverse=True), "better-ranked never gets fewer"
    assert min(seeds[:n_hi]) >= s >= max(seeds[n_hi:])
    assert set(seeds[n_hi:]) == {max(1, s // 2)}
    assert seeds[0] - seeds[n_hi - 1] <= 1, "the remainder is spread one seed each"


def test_every_allocator_is_budget_matched_to_uniform():
    """Same inputs, four allocators, one sum."""
    n, s = 5, 3
    plans = {name: registry.get("allocation", name)(_ctx(s), _state(n), _cands(n))
             for name in ("uniform", "successive_halving", "bandit",
                          "bayesian_experimental_design")}
    assert {name: sum(p.values()) for name, p in plans.items()} == {
        name: n * s for name in plans}


def test_without_a_discriminating_prior_the_split_is_uniform():
    n, s = 5, 2
    plan = registry.get("allocation", "successive_halving")(_ctx(s), RunState(), _cands(n))
    assert set(plan.values()) == {s}
