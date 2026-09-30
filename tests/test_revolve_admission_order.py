"""REvolve's island admission gate sees offspring in CANDIDATE order, not in
§5's fitness order.

`above_island_mean` recomputes the deme mean after every admission, so the
order in which a round's offspring are offered decides who gets in. The
release offers them in the order they were generated (`refs/code/Revolve/
main.py:237-245`, `rewards_database.py:102-126`). BIRD's `select.rule:
argmax_fitness` hands `topo_island_lineage` its winners sorted by fitness, and
admitting in that order would let the best offspring go first, lift the mean,
and reject siblings the release admits. The wrong answer is
a smaller, fitter-looking deme, which renders as a plausible population.

The probe: incumbent at 0, same-island offspring in slots [0.25, 1.0].
Release: admit 0.25 against 0, then 1.0 against 0.125 -> [0, 0.25, 1].
Admitting in fitness order gives [1, 0].
"""
from __future__ import annotations

import random

from bird import registry
from bird.budget import Budget
from bird.components import evolution, selection, update
from bird.config import load
from bird.context import Context
from bird.envs.toy import ToyReacher
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult


def _ctx() -> Context:
    registry.load_all()
    return Context(cfg=load("revolve"), budget=Budget(), env=ToyReacher(), rng=random.Random(0))


def _report(cid: str, fitness: float, *, iteration: int, slot: int, island: int = 0):
    cand = Candidate(cid, iteration, "def reward(o, a, o2):\n    return 0.0, {}\n",
                     meta={"island": island, "sample_index": slot})
    return CandidateReport(cid, cand, TrainResult(cid, cand), fitness=fitness)


def _release_admission(incumbents, offered):
    """`rewards_database.py:102-126`: running mean, recomputed per candidate."""
    pop = list(incumbents)
    for x in offered:
        if x >= sum(pop) / len(pop):
            pop.append(x)
    return pop


def test_offspring_are_admitted_in_candidate_order_not_fitness_order():
    ctx = _ctx()
    assert ctx.cfg["select.rule"] == "argmax_fitness", "the published REvolve selector"
    assert ctx.cfg["update.archive.admission"] == "above_island_mean"
    state = RunState(iteration=1)
    incumbent = _report("old", 0.0, iteration=0, slot=0)
    update._set_population(state, 0, [incumbent])
    state.best = incumbent

    lo = _report("slot0", 0.25, iteration=1, slot=0)
    hi = _report("slot1", 1.0, iteration=1, slot=1)
    sel = selection.rule_argmax_fitness(ctx, state, [lo, hi])
    assert [r.cand_id for r in sel.winners] == ["slot1", "slot0"], (
        "§5 sorts by fitness; the gate below must not inherit that order")

    evolution.topo_island_lineage(ctx, state, sel)

    got = sorted(r.fitness for r in update._population(state, 0))
    assert got == sorted(_release_admission([0.0], [0.25, 1.0])) == [0.0, 0.25, 1.0], got
    assert state.best.cand_id == "slot1", "§5 still names the winner"


def test_candidate_order_survives_a_repair_id():
    """A repaired candidate keeps its slot through `meta["sample_index"]`
    (`verification._regenerate` copies meta); its `r####` id would otherwise
    sort after every `c####` sibling."""
    a = _report("r0000", 0.1, iteration=1, slot=0)
    b = _report("c0001", 0.2, iteration=1, slot=1)
    c = _report("c0002", 0.3, iteration=1, slot=2)
    assert [r.cand_id for r in evolution._candidate_order([c, b, a])] == ["r0000", "c0001", "c0002"]
