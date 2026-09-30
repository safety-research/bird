"""REvolve's generation 0 is ZERO-SHOT, and a candidate with no measurement never
seeds a deme (both in `bird/components/evolution.py`).

Generation 0. The release's generation 0 sends an EMPTY user turn: `main.py:159-163`
sets `operator_prompt = ""` and `in_context_samples = (None, None)`, and
`modules.py:69-71` returns `""` from `prepare_in_context_prompt` when
`evolve` is False. The paper's `\\subsection{Initialization}` (main.tex:285-287)
describes the seed population as GPT-4's "zero-shot instruction following".
A sampler that wrote the MUTATION instruction into GUIDANCE unconditionally
would tell every generation-0 candidate to "iterate on the reward function by
mutating a single component" with no PARENT REWARD section to mutate, and stamp
it `operator: mutation` for an operator no code ran.

Unmeasured survivors. Under `select.n_survivors: 16` every report with a numeric fitness is a
§5 winner, INCLUDING a candidate whose training raised or that never compiled
(`fitness = select.failure_value = -10000`, `_blank`). At generation 0 the
admission rule is forced to `truncate`, so without a guard the sentinel would
be filed into its seeded deme, and both roulette wheels (`update._pick_island_avg_weighted`,
`update._fitness_proportional`) shift their weights by the MINIMUM -- one
-10000 member flattens island selection to ~uniform and its own deme's
0.045-vs-0.0 pair to 0.507/0.493. The release has no path on which an
unmeasured individual reaches `seed_islands` / `add_individuals_to_islands`
(main.py:236-253): an invalid function ends `generate_valid_reward` with
`None` (main.py:87-90) and a training exception propagates out of
`future.result()` (modules.py:256).

Both tests drive the two components directly with a stub LLM backend rather
than through `bird.py`: a 3-candidate tester run of this config takes ~40 s,
and the defect is a property of one prompt and one deme, not
of a run.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

import pytest

from bird import registry
from bird.budget import Budget
from bird.components import evolution, selection, update
from bird.config import load
from bird.context import Context
from bird.envs.toy import ToyReacher
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult

#: The operative sentence of each vendored operator prompt, as `evolution.py`
#: lifts it (`_MUTATION` / `_CROSSOVER`); neither may appear on a zero-shot turn.
MUTATION_TELL = "mutating a single component"
CROSSOVER_TELL = "combine high-performing reward components"
GUIDANCE = "Guidance marker: make the at-goal state the global argmax."

CODE = '''def compute_reward(state, action=None, *_extra, **_kw):
    components = {"ctrl": 0.0, "smoothness": 0.0}
    return sum(components.values()), components
'''
RAW = "Penalise control effort.\n\n```python\n" + CODE + "```\n"


class _Backend:
    """A stub LLM: returns `RAW` and keeps every prompt it was shown."""
    needs_prompt = True

    def __init__(self) -> None:
        self.prompts: List[List[Dict[str, str]]] = []

    def __call__(self, ctx, state, messages, n):
        self.prompts.append(messages)
        return [RAW] * int(n)


class _Journal:
    """Stands in for `RunDir` so `ctx.event` is observable without a run dir."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def event(self, stage: str, **fields: Any) -> None:
        self.events.append({"stage": stage, **fields})


def _ctx(**overrides) -> Context:
    registry.load_all()
    cfg = load("revolve", profile="tester", overrides=overrides or None)
    return Context(cfg=cfg, budget=Budget(), env=ToyReacher(),
                   rng=random.Random(0), rundir=_Journal())


def _user_turn(messages: List[Dict[str, str]]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def _report(cid: str, fitness: float, *, iteration: int, slot: int, island: int,
            valid: bool = True, error: str = "") -> CandidateReport:
    cand = Candidate(cid, iteration, CODE, meta={"island": island, "sample_index": slot})
    if not valid:
        cand = cand.failed("signature_parse: cannot parse: SyntaxError", "invalid")
    result = TrainResult(cid, cand, error=error)
    rep = CandidateReport(cid, cand, result, fitness=fitness)
    if not valid or error:
        rep.meta["fitness_note"] = error or cand.failure
    return rep


# ==========================================================================
# generation 0 is zero-shot
# ==========================================================================


def test_generation_zero_carries_no_operator_instruction_and_is_stamped_init():
    ctx = _ctx(**{"generate.n_candidates": 4, "generate.context.guidance": GUIDANCE})
    assert ctx.cfg["generate.sampling_mode"] == "evolutionary_operators"
    backend = _Backend()
    state = RunState(iteration=0)          # empty archive: no cohort can be drawn

    out = evolution.sample_evolutionary_operators(ctx, state, backend)

    assert len(out) == 4 and len(backend.prompts) == 4
    for cand, messages in zip(out, backend.prompts):
        user = _user_turn(messages)
        assert "## PARENT REWARD" not in user, "generation 0 has no parent"
        assert MUTATION_TELL not in user and CROSSOVER_TELL not in user, (
            f"{cand.cand_id}: a zero-shot turn was handed an operator instruction "
            "(release: main.py:159-163 operator_prompt = \"\"; modules.py:69-71)")
        # The CONFIGURED guidance still renders -- dropping the operator text
        # must not take `generate.context.guidance` down with it.
        assert "## GUIDANCE\n" + GUIDANCE in user
        assert cand.meta["operator"] == "init", cand.meta
        assert cand.meta["operator_fallback"] == ""
        assert cand.meta["parent_ids"] == []
        assert cand.meta["island_source"] == "seeded"
        assert "GUIDANCE" not in cand.meta["prompt_sections_overridden"]
        # The Bernoulli draw is still consumed (the stream position of every
        # later draw is unchanged) and recorded, so `init` is not a hole.
        assert cand.meta["operator_drawn"] in ("mutation", "crossover")


def test_generation_one_still_carries_the_operator_instruction():
    """The other half of the gate: with a parent drawn, the operator text is
    present and `operator` names the operator that ran -- so the fix cannot be
    a blanket removal of the instruction."""
    ctx = _ctx(**{"generate.n_candidates": 4})
    backend = _Backend()
    state = RunState(iteration=1)
    for isl, (cid, f) in enumerate([("p0", 0.4), ("p1", 0.3), ("p2", 0.2)]):
        update._set_population(state, isl, [
            _report(cid, f, iteration=0, slot=isl, island=isl),
            _report(cid + "b", f / 2, iteration=0, slot=isl + 8, island=isl)])

    out = evolution.sample_evolutionary_operators(ctx, state, backend)

    assert len(out) == 4
    for cand, messages in zip(out, backend.prompts):
        user = _user_turn(messages)
        assert "## PARENT REWARD" in user
        assert cand.meta["operator"] in ("mutation", "crossover"), cand.meta
        tell = MUTATION_TELL if cand.meta["operator"] == "mutation" else CROSSOVER_TELL
        assert tell in user, (cand.meta, user[-600:])
        assert cand.meta["parent_ids"], "a generation>0 offspring has a lineage"
        assert "GUIDANCE" in cand.meta["prompt_sections_overridden"]
        assert cand.meta["operator_drawn"] in ("mutation", "crossover")


# ==========================================================================
# an unmeasured survivor never enters a deme
# ==========================================================================


def _round(iteration: int, *, with_failures: bool):
    """Four measured offspring across three demes, plus (optionally) an invalid
    one and a training-error one stamped into deme 8 and deme 1."""
    reports = [
        _report("c0000", 0.045, iteration=iteration, slot=0, island=0),
        _report("c0001", 0.0, iteration=iteration, slot=1, island=0),
        _report("c0002", 0.04, iteration=iteration, slot=2, island=1),
        _report("c0003", 0.02, iteration=iteration, slot=3, island=2),
    ]
    if with_failures:
        reports.append(_report("c0004", -10000.0, iteration=iteration, slot=4,
                               island=8, valid=False))
        reports.append(_report("c0005", -10000.0, iteration=iteration, slot=5,
                               island=1, error="ZeroDivisionError: float division"))
    return reports


def _archive(state: RunState):
    return {k: (c.report.cand_id, c.fitness) for k, c in sorted(state.archive.items())}


def _island_draws(ctx: Context, state: RunState, n: int = 2000) -> Dict[int, int]:
    cells = [c for c in state.archive.values() if c.report is not None]
    ctx.rng = random.Random(7)
    counts: Dict[int, int] = {}
    for _ in range(n):
        isl = update._pick_island_avg_weighted(ctx, cells)
        counts[isl] = counts.get(isl, 0) + 1
    return counts


@pytest.mark.parametrize("iteration", [0, 1],
                         ids=["generation-0 (truncate, forced)",
                              "generation-1 (above_island_mean, empty deme admits all)"])
def test_a_sentinel_valued_survivor_is_skipped_not_filed(iteration):
    ctx = _ctx()
    assert ctx.cfg["select.failure_value"] == -10000
    assert ctx.cfg["select.n_survivors"] == 16
    state = RunState(iteration=iteration)
    reports = _round(iteration, with_failures=True)
    sel = selection.rule_argmax_fitness(ctx, state, reports)
    # Precondition, and the reason this is a §6 bug: §5 hands both failures
    # over as winners, because `_scored` drops `None` and nothing else.
    assert {"c0004", "c0005"} <= {r.cand_id for r in sel.winners}

    evolution.topo_island_lineage(ctx, state, sel)

    filed = {c.report.cand_id: c for c in state.archive.values() if c.report is not None}
    assert "c0004" not in filed, "an INVALID candidate was filed into deme 8"
    assert "c0005" not in filed, "a candidate whose training RAISED was filed into deme 1"
    assert all(c.fitness > -10000 for c in filed.values())
    assert update._population(state, 8) == []
    assert [r.cand_id for r in update._population(state, 1)] == ["c0002"]

    skipped = [e for e in ctx.rundir.events if e["stage"] == "update_island_skipped"]
    assert sorted(e["cand_id"] for e in skipped) == ["c0004", "c0005"], (
        "a skipped survivor must be journaled -- a silent skip is indistinguishable "
        "from a candidate that was never a winner")
    by_id = {e["cand_id"]: e for e in skipped}
    assert by_id["c0004"]["island"] == 8 and "invalid" in by_id["c0004"]["reason"]
    assert by_id["c0005"]["island"] == 1 and "ZeroDivisionError" in by_id["c0005"]["reason"]
    assert by_id["c0004"]["fitness"] == -10000.0

    # The whole point: the archive -- and therefore both roulette wheels -- is
    # exactly what a round WITHOUT the failures produces.
    clean_ctx = _ctx()
    clean = RunState(iteration=iteration)
    evolution.topo_island_lineage(
        clean_ctx, clean,
        selection.rule_argmax_fitness(clean_ctx, clean, _round(iteration, with_failures=False)))
    assert _archive(state) == _archive(clean)
    assert _island_draws(ctx, state) == _island_draws(clean_ctx, clean)


def test_the_measured_survivors_of_the_same_round_are_still_filed():
    """Skipping is per report, not per round: the four measured offspring land
    exactly where they would have with no failure beside them."""
    ctx = _ctx()
    state = RunState(iteration=0)
    evolution.topo_island_lineage(
        ctx, state, selection.rule_argmax_fitness(ctx, state, _round(0, with_failures=True)))
    assert sorted(r.cand_id for r in update._population(state, 0)) == ["c0000", "c0001"]
    assert [r.cand_id for r in update._population(state, 1)] == ["c0002"]
    assert [r.cand_id for r in update._population(state, 2)] == ["c0003"]
    touched = [e for e in ctx.rundir.events if e["stage"] == "update_island"]
    assert sorted(e["island"] for e in touched) == [0, 1, 2], (
        "deme 8 received only a skipped report and was never touched")
