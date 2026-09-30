"""REvolve's fitness is the Elo RATING, and a tie is ONE f = 0.5 game.

The paper defines sigma as the rating itself (`refs/tex/revolve/main.tex:
993-1007`, K = 32, start 1500) and gates admission `sigma >= sigma^P` on it
(`:314`). The `elo` aggregator replays those games and then maps the ratings
through 10^(R/400) onto a simplex before they become fitness.
`above_island_mean` is affine-invariant but not exp-invariant: by Jensen the
deme mean of 10^(R/400) exceeds 10^(mean R/400), so under `elo` that gate is
strictly stricter than the paper's (deme {1400, 1600}, offspring 1500: paper
admits, `elo` rejects 0.2993 < 0.3503). Recording a tie as two mirrored
decisive games, which `elo` replays as two K=32 updates, moves a pair about
twice the paper's single f = 0.5 step, and order-dependently.

`elo_raw` is the `pref_aggregator` value the published human arm uses: the same
replay, the raw rating returned, and a `tie=True` pair folded into one half
game. `elo` is kept bit-for-bit as a variant.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components import preferences
from bird.context import Context
from bird.envs.toy import ToyReacher
from bird.state import RunState
from bird.types import Candidate, CandidateReport, Preference, TrainResult, Trajectory
from conftest import load_revolve_published

IDS = ["a", "b", "c"]


def _p(left: str, right: str, label: int, **kw) -> Preference:
    return Preference(left, right, label, source="human", iteration=0, **kw)


def _log():
    # a > b, a > c, and (b, c) labelled 0: c beat b.
    return [_p("a", "b", 1), _p("a", "c", 1), _p("b", "c", 0)]


def _agg(name: str):
    registry.load_all()
    return registry.get("pref_aggregator", name)


def _paper_replay(ids, prefs):
    """tex:993-1007 verbatim: K = 32, start 1500, E = 1 / (1 + 10^((R_b - R_a)/400))."""
    rating = {cid: 1500.0 for cid in ids}
    for pref in prefs:
        ra, rb = rating[pref.left_id], rating[pref.right_id]
        expected = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        score = 1.0 if pref.label == 1 else 0.0
        rating[pref.left_id] = ra + 32.0 * (score - expected)
        rating[pref.right_id] = rb + 32.0 * ((1.0 - score) - (1.0 - expected))
    return rating


def test_elo_raw_fitness_is_the_elo_rating():
    """`elo_raw` returns the paper's replayed ratings, not a simplex."""
    got = _agg("elo_raw")(None, IDS, _log())
    # The paper's replay computed on this tree, pinned as an equality.
    assert got == {"a": 1531.263693206478, "b": 1468.0339081301693, "c": 1500.7023986633528}
    assert got == _paper_replay(IDS, _log())
    # No simplex: these are ratings, not a distribution.
    assert sum(got.values()) == pytest.approx(3 * 1500.0)


def test_elo_is_unchanged_bit_for_bit():
    """A regression pin on `elo`'s numbers, which must not move because the
    replay loop is shared with `elo_raw` through `_elo_replay`."""
    got = _agg("elo")(None, IDS, _log())
    assert got == {"a": 0.3946974633536348, "b": 0.27427730441314446, "c": 0.33102523223322083}
    assert sum(got.values()) == pytest.approx(1.0)
    # The equivalence behind replacing `run_preferences`' `len(pool) < 2`
    # literal with an aggregator call: every simplex aggregator answers 1.0 for
    # one id and {} for none ...
    for name in ("bradley_terry", "elo", "borda", "copeland"):
        assert _agg(name)(None, ["a"], []) == {"a": 1.0}, name
        assert _agg(name)(None, [], []) == {}, name
    # ... and `elo_raw` answers the paper's start rating.
    assert _agg("elo_raw")(None, ["a"], []) == {"a": 1500.0}
    assert _agg("elo_raw")(None, [], []) == {}


def test_a_tie_is_one_half_game_under_elo_raw():
    """A `tie=True` pair is one f = 0.5 game, not two decisive ones."""
    tie_pair = [_p("a", "b", 1, tie=True), _p("a", "b", 0, tie=True)]
    # (i) Equal ratings: E = 0.5, f = 0.5, delta 0 -- the paper's tie is a no-op.
    assert _agg("elo_raw")(None, ["a", "b"], tie_pair) == {"a": 1500.0, "b": 1500.0}
    # `elo` never reads the flag: two decisive games, +-1.47 of movement.
    old = _agg("elo")(None, ["a", "b"], tie_pair)
    assert old["a"] != old["b"]
    # (ii) A decisive game then a tie: ONE f = 0.5 update at the post-game ratings.
    got = _agg("elo_raw")(None, ["a", "b"], [_p("a", "b", 1)] + tie_pair)
    a, b = 1516.0, 1484.0  # after a > b from 1500/1500
    e_a = 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))
    a2 = a + 32.0 * (0.5 - e_a)
    b2 = b + 32.0 * ((1.0 - 0.5) - (1.0 - e_a))
    assert got == {"a": a2, "b": b2}
    # The paper: A (higher) moves down by 32 * (0.5 - E_A) = -1.47 here; `elo`'s
    # two-update tie moves the same pair about twice as far.
    assert got["a"] == pytest.approx(1516.0 - 1.4695, abs=1e-3)
    # The label-0 mirror alone contributes nothing (`_pref` always emits the pair).
    assert _agg("elo_raw")(None, ["a", "b"], [_p("a", "b", 0, tie=True)]) == {"a": 1500.0, "b": 1500.0}


class _TyingHuman:
    """The interactive path maps -1 to TIE (preferences.py); this stub says it directly."""

    def compare(self, left, right):
        return preferences.TIE

    def feedback(self, *args, **kwargs):
        return ""


def _trained(cid: str) -> CandidateReport:
    traj = Trajectory(states=np.array([[0.0], [1.0], [2.0]]),
                      actions=np.array([[0.1], [0.2]]),
                      rewards=[0.0, 1.0], success=True, length=2)
    cand = Candidate(cid, 1, "def reward(s, a, s2): return 0.0, {}")
    return CandidateReport(cid, cand, TrainResult(cid, cand, trajectories=[traj]), fitness=None)


def test_run_preferences_marks_a_tie_on_both_records():
    """The published human arm (override set + `elo_raw`) writes a tie as a
    `tie=True` pair and the aggregator folds it: both agents stay at 1500.0."""
    registry.load_all()
    cfg = load_revolve_published(profile="tester")
    assert cfg["evaluate.preferences.aggregator"] == "elo_raw"
    assert cfg["evaluate.preferences.allow_ties"] is True
    ctx = Context(cfg=cfg, budget=Budget(), env=ToyReacher(), rng=random.Random(0))
    ctx.human = _TyingHuman()
    state = RunState(iteration=1)
    reports = [_trained("x"), _trained("y")]

    preferences.run_preferences(ctx, state, reports)

    assert len(state.preferences) == 2
    assert all(p.tie for p in state.preferences)
    assert [p.label for p in state.preferences] == [1, 0]
    for rep in reports:
        assert rep.meta["bt_strength"] == 1500.0
        assert rep.meta["pref_aggregator"] == "elo_raw"


def test_the_prompt_labels_a_rating_as_a_rating():
    """`_section_preference` labels `bt_strength` by the aggregator that wrote
    it: on the human arm that number is an Elo rating, not a Bradley-Terry
    strength. Every other aggregator keeps the Bradley-Terry wording.

    The context is REvolve's own: the block is
    gated by `evaluate.feedback.state_selection_scalar`, which REvolve leaves
    true -- its prompt SHOWS the rating, so the label has to be right."""
    from bird.components.evaluation import _section_preference
    ctx = Context(cfg=load_revolve_published(profile="tester"), budget=Budget())
    assert ctx.cfg.get("evaluate.feedback.state_selection_scalar") is True
    cand = Candidate("z", 0, "def reward(s, a, s2): return 0.0, {}")
    rep = CandidateReport("z", cand, TrainResult("z", cand), fitness=None,
                          meta={"bt_strength": 1531.2637, "pref_aggregator": "elo_raw"})
    text = _section_preference(ctx, rep)
    assert "Elo rating" in text and "Bradley-Terry" not in text
    rep.meta["pref_aggregator"] = "elo"
    rep.meta["bt_strength"] = 0.3947
    assert "Bradley-Terry strength" in _section_preference(ctx, rep)


def _untrained(cid: str) -> CandidateReport:
    """A valid candidate whose training failed: no policy, so it cannot race.
    `evaluation._blank` leaves such a report at `select.failure_value`."""
    cand = Candidate(cid, 1, "def reward(s, a, s2): return 0.0, {}")
    return CandidateReport(cid, cand, TrainResult(cid, cand, trained=False), fitness=-10000.0)


def test_a_never_raced_candidate_gets_no_fabricated_rating():
    """`_run_preferences` writes `bt_strength = 0.0` onto every report as the
    never-raced floor and `_write_back` fills in only the pool, so on the human
    arm a candidate whose training failed carries `bt_strength: 0.0` with no
    `pref_aggregator`. Rendered naively, its feedback would read "Preference
    evidence: Bradley-Terry strength 0.000." beside peers reading "Elo rating
    1672.029". Neither half is true -- the arm's aggregator is `elo_raw`, and
    0.0 is not a rating it produced: an unraced POOL member reads 1500.0
    (`test_elo_is_unchanged_bit_for_bit`). The honest line states the absence
    and carries no number."""
    from bird.components.evaluation import _section_preference
    registry.load_all()
    ctx = Context(cfg=load_revolve_published(profile="tester"), budget=Budget(),
                  env=ToyReacher(), rng=random.Random(0))
    ctx.human = _TyingHuman()
    reports = [_trained("x"), _trained("y"), _untrained("z")]
    preferences.run_preferences(ctx, RunState(iteration=1), reports)
    z = reports[2]
    # The writer's contract this test relies on: the floor, and nothing from `_write_back`.
    assert z.meta["bt_strength"] == 0.0
    assert not any(k.startswith("pref_") for k in z.meta)
    text = _section_preference(ctx, z)
    assert "Bradley-Terry" not in text
    assert text == "Preference evidence: not compared (no trained policy to race this round)."
    # The raced peers read as ratings; 1500 is the aggregator's word, not this
    # test's. (The tally after it is `_write_back`'s, which counts a tie's two
    # mirrored records as one win and one loss each -- not this test's subject.)
    for rep in reports[:2]:
        assert _section_preference(ctx, rep).startswith(
            "Preference evidence: Elo rating 1500.000, ")
    # Prose only: the artifact keeps the floor and stage 5 keeps the sentinel.
    assert z.meta["bt_strength"] == 0.0 and z.fitness == -10000.0


def test_the_label_falls_back_to_the_configured_aggregator():
    """A raced report whose meta carries no `pref_aggregator` -- built by hand,
    or written by a `_write_back` that does not stamp it -- is labelled by
    `evaluate.preferences.aggregator`, never by a hard-wired default: the
    elo_raw arm says "Elo rating", a bradley_terry override says
    "Bradley-Terry strength", and an explicit stamp on the meta still wins.
    A literal "Bradley-Terry strength" fallback fails at the first `assert`."""
    from bird.components.evaluation import _section_preference
    cand = Candidate("z", 0, "def reward(s, a, s2): return 0.0, {}")
    rep = CandidateReport("z", cand, TrainResult("z", cand), fitness=None,
                          meta={"bt_strength": 1531.2637, "pref_comparisons": 2,
                                "pref_wins": 2, "pref_losses": 0})
    elo_ctx = Context(cfg=load_revolve_published(profile="tester"), budget=Budget())
    assert "Elo rating 1531.264" in _section_preference(elo_ctx, rep)
    bt_ctx = Context(cfg=load_revolve_published(
        profile="tester", overrides={"evaluate.preferences.aggregator": "bradley_terry"}),
        budget=Budget())
    assert "Bradley-Terry strength 1531.264" in _section_preference(bt_ctx, rep)
    rep.meta["pref_aggregator"] = "elo"
    assert "Bradley-Terry strength" in _section_preference(elo_ctx, rep)
