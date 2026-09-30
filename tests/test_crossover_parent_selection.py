"""`generate.crossover.temperature` at 0 is greedy, not the default.

Reading the key as `float(cfg[...]) or 1.0` in `parents_softmax_fitness` would
silently replace a schema-legal 0 -- greedy parent selection, which the
`max(temp, 1e-9)` clamp a few lines down exists to make runnable -- with the
plain softmax: `-s generate.crossover.temperature=0` would validate, move the
hash, open a new run directory and draw exactly the parents 1.0 draws
(identical pairs at 0 and at 1.0 under a fixed seed). With no `or`, 0 reaches
the clamp; a negative temperature, which would rank the pool upside down, is
refused by name at load.

`generate.crossover.min_segment_len` is deliberately NOT touched: 0 and 1 are
the same value there (`_runs` clamps with `max(1, ...)`), so nothing is hidden.
"""

from __future__ import annotations

import random
from typing import List

import pytest

from bird import registry
from bird.budget import Budget
from bird.config import ConfigError, load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult

#: Five parents, one clear winner at index 1.
FITS = [0.1, 0.9, 0.3, 0.5, 0.2]
TOP = "c0001"
N_PAIRS = 20
SEED = 0


def _reports(fits: List[float]) -> List[CandidateReport]:
    out = []
    for i, f in enumerate(fits):
        c = Candidate(cand_id=f"c{i:04d}", iteration=0,
                      reward_code="def compute_reward(s, a, n):\n    return 0.0\n")
        out.append(CandidateReport(cand_id=c.cand_id, candidate=c,
                                   result=TrainResult(cand_id=c.cand_id, candidate=c),
                                   fitness=f))
    return out


def _pairs(temperature: float) -> List[List[str]]:
    """`N_PAIRS` parent pairs drawn from R*'s tester point at `temperature`,
    from a fixed `random.Random(SEED)` -- the run's one reproducible stream."""
    registry.load_all()
    cfg = load("rstar", profile="tester",
               overrides={"generate.crossover.temperature": temperature})
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(SEED))
    state = RunState()
    state.all_reports = _reports(FITS)
    select = registry.get("crossover_parent_selection", "softmax_fitness")
    flat = [r.cand_id for r in select(ctx, state, N_PAIRS)]
    assert len(flat) == 2 * N_PAIRS
    return [flat[i:i + 2] for i in range(0, len(flat), 2)]


def test_temperature_zero_is_greedy() -> None:
    """Every pair holds the top-fitness parent: at 0 the clamp makes the
    softmax a point mass, and `_sample_two` draws the mass first every time."""
    pairs = _pairs(0)
    missing = [p for p in pairs if TOP not in p]
    assert not missing, (
        f"{len(missing)} of {N_PAIRS} pairs lack the top-fitness parent at "
        f"temperature 0 -- greedy selection is not running: {missing[:3]}")
    assert all(a != b for a, b in pairs), "a parent crossed with itself"


def test_temperature_one_is_not_greedy_under_the_same_seed() -> None:
    """The contrast, from the same stream: the plain softmax lets a lower
    parent pair without the top one. This is what an ignored 0 would draw."""
    pairs = _pairs(1.0)
    assert any(TOP not in p for p in pairs), (
        "the plain softmax never drew a pair without the top parent; the "
        "contrast this file depends on is gone -- re-measure FITS / SEED")


def test_zero_and_one_draw_different_parents() -> None:
    """The two values are two different runs, not one."""
    assert _pairs(0) != _pairs(1.0)


def test_a_negative_temperature_is_refused_by_name() -> None:
    with pytest.raises(ConfigError) as exc:
        load("rstar", profile="tester", overrides={"generate.crossover.temperature": -0.5})
    msg = str(exc.value)
    assert "generate.crossover.temperature" in msg and "-0.5" in msg
    # 0 is legal, and resolves to 0 -- not to the default.
    assert load("rstar", profile="tester",
                overrides={"generate.crossover.temperature": 0})["generate.crossover.temperature"] == 0
