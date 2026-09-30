"""One Bradley-Terry fit decides both the parent and the human's subject.

GT Alg. 1 computes ONE set of scores `b_1:N <- bradley({p_ij})` (tex:318) and
takes ONE argmax (tex:320) that is at once the parent `R_best`, the human's
subject `pi_best` (tex:322) and the source of `eta_best` (tex:324). Two fits
would disagree: `preferences.aggregate_bradley_terry` (alpha = 0.5 virtual
comparisons split over every pair) writes `meta["bt_strength"]`, which becomes
`report.fitness` and is what `phases._human_targets` sorts on, while a refit in
`selection.rule_bradley_terry` with `_bt_strengths` (`BT_PRIOR` against a fixed
anchor) would select on a different number. Over the 1024 all-pairs outcomes of
a 5-agent round the two argmaxes differ in 50. In the example round below the
first fit ties c0001 and c0004 exactly (human target c0001, by name) while the
refit splits them by one ulp and hands the parent to c0004 -- the human narrates
one agent, the next prompt carries another's code, `final_retrain` retrains the
second and reports the first's fitness, and the journal says `tie_broken: false`.

The rule reads the strengths the preferences phase already wrote and refits
only when there are none. Each test below fails independently if the rule
always refits, except the fallback pins, which keep the refit path reachable.
"""

from __future__ import annotations

import itertools
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird import registry  # noqa: E402
from bird.budget import Budget  # noqa: E402
from bird.components.evaluation import _resolve_deferred  # noqa: E402
from bird.components.phases import _FINAL_ARTIFACT_RULES, _human_targets  # noqa: E402
from bird.components.preferences import _write_back, aggregate_bradley_terry  # noqa: E402
from bird.components.selection import _bt_strengths, rule_bradley_terry  # noqa: E402
from bird.components.update import win_become_parent  # noqa: E402
from bird.config import load  # noqa: E402
from bird.context import Context  # noqa: E402
from bird.state import RunState  # noqa: E402
from bird.types import Candidate, CandidateReport, Preference, TrainResult  # noqa: E402

IDS = [f"c{i:04d}" for i in range(5)]
PAIRS = list(itertools.combinations(range(5), 2))

#: The example round: labels over `PAIRS` in order (1 = left wins). c0001 and
#: c0004 finish 3-1 with symmetric records; the preferences fit ties them
#: exactly, the selection refit splits them by one ulp toward c0004.
EXAMPLE_ROUND = (0, 0, 0, 0, 0, 1, 1, 0, 0, 0)


def _ctx():
    registry.load_all()
    cfg = load("gt", profile="tester")
    assert cfg["select.rule"] == "bradley_terry"
    assert cfg["evaluate.fitness.source"] == "preference_bt"
    events: list = []

    class _Rundir:
        def event(self, stage, **fields):
            events.append({"stage": stage, **fields})

    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(0))
    ctx.rundir = _Rundir()
    return ctx, events


def _reports():
    out = []
    for cid in IDS:
        cand = Candidate(cand_id=cid, iteration=0, reward_code="def f(): ...")
        res = TrainResult(cand_id=cid, candidate=cand, trained=True)
        out.append(CandidateReport(cand_id=cid, candidate=cand, result=res,
                                   fitness=None, fitness_source="preference_bt"))
    return out


def _prefs(bits):
    return [Preference(left_id=IDS[i], right_id=IDS[j], label=b, source="vlm", iteration=0)
            for (i, j), b in zip(PAIRS, bits)]


def _stage4(ctx, state, reports, bits):
    """What stage 4 leaves behind: the preferences phase fits b_1:N and writes
    it to `meta["bt_strength"]`, then the deferred `preference_bt` fitness
    resolves from that key -- the same sequence `bird.evaluate` runs."""
    prefs = _prefs(bits)
    state.preferences = list(prefs)
    strengths = aggregate_bradley_terry(ctx, IDS, prefs)
    _write_back(ctx, reports, strengths, prefs, "llm_on_vlm_captions", "bradley_terry")
    for r in reports:
        _resolve_deferred(ctx, r)
    return strengths


def _select_event(events):
    got = [e for e in events if e["stage"] == "select_bt"]
    assert len(got) == 1, events
    return got[0]


# --------------------------------------------------------------------------
# the example round, end to end through the real components
# --------------------------------------------------------------------------


def test_the_parent_is_the_agent_the_human_was_asked_about():
    """Alg. 1 l.15-19: one argmax is both `pi_best` (human) and `R_best` (parent)."""
    ctx, events = _ctx()
    state, reports = RunState(), _reports()
    recorded = _stage4(ctx, state, reports, EXAMPLE_ROUND)
    assert recorded["c0001"] == recorded["c0004"], "the round is a structural tie at the top"

    target = _human_targets(ctx, reports)[0]
    sel = rule_bradley_terry(ctx, state, reports)
    assert sel.winners[0].cand_id == target.cand_id, (
        f"human narrated {target.cand_id}, parent is {sel.winners[0].cand_id}")
    # The strengths the round was selected on are the ones the fitness reports.
    for r in reports:
        assert _select_event(events)["strengths"][r.cand_id] == round(r.fitness, 6)


def test_the_refit_alone_would_have_disagreed_on_this_round():
    """Documents the defect's shape rather than the fix: the second fit is
    still in the file (the fallback below needs it) and on this round its
    argmax is not the recorded fit's. If this ever passes trivially the round
    no longer demonstrates anything and needs replacing."""
    prefs = _prefs(EXAMPLE_ROUND)
    recorded = aggregate_bradley_terry(None, IDS, prefs)
    refit = _bt_strengths(IDS, prefs)
    by_name = sorted(IDS, key=lambda c: (-recorded[c], c))[0]
    by_value = sorted(IDS, key=lambda c: -refit[c])[0]
    assert by_name != by_value, (recorded, refit)


def test_a_structural_tie_is_reported_as_a_tie():
    """`tie_broken: false` while a rounding bit decides the parent would be a
    false record. On the recorded fit the symmetric pair is an exact tie, so
    `_rank_and_cut` sees it and the journal says so."""
    ctx, events = _ctx()
    state, reports = RunState(), _reports()
    _stage4(ctx, state, reports, EXAMPLE_ROUND)
    sel = rule_bradley_terry(ctx, state, reports)
    assert sel.tie_broken is True
    assert set(sel.tied_ids) == {"c0001", "c0004"}


def test_chain_end_and_final_retrain_follow_the_human_target():
    """`select.final_artifact: chain_end` returns `state.latest`, which
    `winner_action: become_parent` writes from the selection; `run_final_retrain`
    reads the same rule and reports `report.fitness` as `selected_fitness`. All
    of it must be the agent the human was shown, and the fitness it reports must
    be the strength the round was selected on."""
    ctx, events = _ctx()
    state, reports = RunState(), _reports()
    _stage4(ctx, state, reports, EXAMPLE_ROUND)
    target = _human_targets(ctx, reports)[0]
    sel = rule_bradley_terry(ctx, state, reports)
    win_become_parent(ctx, state, sel.winners[0])
    final = _FINAL_ARTIFACT_RULES["chain_end"](state)
    assert final is not None and final.cand_id == target.cand_id
    assert round(final.fitness, 6) == _select_event(events)["strengths"][final.cand_id]


def test_every_all_pairs_outcome_agrees():
    """Two fits disagree on 50 outcomes of 1024; with one fit there must be none.
    Both paths sort on the same floats: exact ties go to `cand_id` order in
    `_human_targets` and to `tie_break: first` (report order, which is id
    order) in `_rank_and_cut`."""
    ctx, events = _ctx()
    disagreements = []
    for bits in itertools.product((0, 1), repeat=len(PAIRS)):
        state, reports = RunState(), _reports()
        _stage4(ctx, state, reports, bits)
        target = _human_targets(ctx, reports)[0].cand_id
        parent = rule_bradley_terry(ctx, state, reports).winners[0].cand_id
        if target != parent:
            disagreements.append((bits, target, parent))
    assert not disagreements, f"{len(disagreements)} of 1024, e.g. {disagreements[:3]}"


# --------------------------------------------------------------------------
# the mechanism, pinned without numerics
# --------------------------------------------------------------------------


def _hand_built(state):
    """Recorded strengths say c0000; the stored preferences say c0004 won
    everything and c0000 lost everything. Any refit picks c0004."""
    reports = _reports()
    for r in reports:
        r.meta["pref_aggregator"] = "bradley_terry"
        r.meta["bt_strength"] = 0.5 if r.cand_id == "c0000" else 0.125
        r.meta["pref_comparisons"] = 4
        r.fitness = r.meta["bt_strength"]
    prefs = []
    for i, j in PAIRS:
        left, right = IDS[i], IDS[j]
        if left == "c0004":
            label = 1
        elif right == "c0004":
            label = 0
        else:
            label = 0 if left == "c0000" else 1
        prefs.append(Preference(left_id=left, right_id=right, label=label, iteration=0))
    state.preferences = prefs
    return reports


def test_selection_reads_the_recorded_strengths_not_a_refit():
    ctx, events = _ctx()
    state = RunState()
    reports = _hand_built(state)
    sel = rule_bradley_terry(ctx, state, reports)
    assert sel.winners[0].cand_id == "c0000", sel.winners[0].cand_id
    ev = _select_event(events)
    assert ev["strengths"] == {r.cand_id: r.meta["bt_strength"] for r in reports}
    assert ev["fit"] == "preferences"
    assert "refit" not in sel.notes


def test_without_recorded_strengths_the_rule_fits_its_own():
    """The fallback: a config that ranks by BT without running the preferences
    phase's aggregator (or a checkpoint with no recorded strengths) still gets a fit, and
    the journal says which."""
    ctx, events = _ctx()
    state = RunState()
    reports = _hand_built(state)
    for r in reports:
        r.meta.pop("bt_strength")
        r.meta.pop("pref_aggregator")
    sel = rule_bradley_terry(ctx, state, reports)
    assert sel.winners[0].cand_id == "c0004"
    assert _select_event(events)["fit"] == "refit"


def test_another_aggregators_strengths_are_not_adopted_as_bradley_terry():
    """`select.rule: bradley_terry` selects on a Bradley-Terry fit. Strengths an
    `elo`/`borda`/`copeland` aggregator wrote are a different model; the rule
    keeps fitting its own rather than silently becoming argmax-of-whatever."""
    ctx, events = _ctx()
    state = RunState()
    reports = _hand_built(state)
    for r in reports:
        r.meta["pref_aggregator"] = "elo"
    sel = rule_bradley_terry(ctx, state, reports)
    assert sel.winners[0].cand_id == "c0004"
    assert _select_event(events)["fit"] == "refit"


def test_a_cumulative_contest_still_refits_over_the_pooled_preferences():
    """§5: per-round strengths are renormalised per round and cross no round
    boundary, so a `select.scope: cumulative` pool spanning rounds cannot be
    ranked on them. That variant (which GT does not run) keeps the refit."""
    ctx, events = _ctx()
    ctx.cfg._data["select"]["scope"] = "cumulative"
    state = RunState()
    reports = _hand_built(state)
    sel = rule_bradley_terry(ctx, state, reports)
    assert sel.winners[0].cand_id == "c0004"
    assert _select_event(events)["fit"] == "refit"


def test_a_non_finite_recorded_strength_falls_back():
    ctx, events = _ctx()
    state = RunState()
    reports = _hand_built(state)
    reports[2].meta["bt_strength"] = math.nan
    sel = rule_bradley_terry(ctx, state, reports)
    assert sel.winners[0].cand_id == "c0004"
    assert _select_event(events)["fit"] == "refit"
