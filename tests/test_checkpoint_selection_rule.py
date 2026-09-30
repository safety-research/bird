"""`train.checkpoint_selection`, the half that trains nothing.

The rule itself, stage 4's reading of a restored checkpoint, the coherence
refusals and the check over the shipped configs. None of these runs a learner,
so the module carries no `slow` mark and the default CI selection runs it; the backend-driving tests live in
`tests/test_checkpoint_selection.py`, whose docstring explains the mechanism.
"""

from __future__ import annotations

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components import training
from bird.config import ConfigError, load
from bird.context import Context
from bird.types import Candidate, TrainResult

_REWARD = """
def compute_reward(s, a, s2):
    import numpy as np
    o = np.asarray(s2, dtype=float)
    return float(-np.linalg.norm(o[0:2] - o[2:4]))
"""

#: A designed-reward curve that peaks at the THIRD checkpoint and then decays.
_PEAKED = [1.0, 2.0, 5.0, 3.0, 2.5, 2.0, 1.5, 1.2, 1.1, 1.0]


def _cand(cid="c0000"):
    return Candidate(cand_id=cid, iteration=0, reward_code=_REWARD)


def test_the_rule_restores_only_a_peak_that_beats_the_last_by_the_margin():
    curve = [{"reward_return": v} for v in _PEAKED[:5]]
    assert training._select_checkpoint(curve, "best_by_reward", 0.02) == (2, "restored")
    assert training._select_checkpoint(curve, "final", 0.02) == (None, "final")
    assert training._select_checkpoint([], "best_by_reward", 0.02) == (None, "no_checkpoints")
    rising = [{"reward_return": v} for v in (1.0, 2.0, 3.0)]
    assert training._select_checkpoint(rising, "best_by_reward", 0.02) == (None, "last_is_best")
    # peak 2.01 over a last of 2.0: 0.01 of a 1.01 span is inside 2%, so the
    # LAST ships; with no margin the same curve restores.
    noisy = [{"reward_return": v} for v in (1.0, 2.0, 2.01, 2.0, 2.0)]
    assert training._select_checkpoint(noisy, "best_by_reward", 0.02) == (None, "within_margin")
    assert training._select_checkpoint(noisy, "best_by_reward", 0.0) == (2, "restored")


def test_ties_go_to_the_latest_checkpoint_and_non_finite_never_wins():
    assert training._argmax_latest([1.0, 3.0, 3.0, 2.0]) == 2
    assert training._argmax_latest([1.0, float("nan"), float("inf"), 0.5]) == 0
    # A curve whose LAST evaluation blew up: the best finite row restores, and
    # the margin is measured over the finite rows only.
    blown = [{"reward_return": v} for v in (1.0, 4.0, float("nan"))]
    assert training._select_checkpoint(blown, "best_by_reward", 0.02) == (1, "restored")


def _result(per_seed_rows, shipped, restored):
    """A TrainResult as a backend would leave it: per-seed curves in the seed
    rows, the seed-mean curve on the result, the selection fields as given."""
    cand = _cand()
    seeds = []
    for i, (rows, ship, rest) in enumerate(zip(per_seed_rows, shipped, restored)):
        curve = [{"round": float(r + 1), "step": float(100 * (r + 1)), "fitness": f,
                  "task_success": f, "reward_return": 0.0} for r, f in enumerate(rows)]
        seeds.append({"seed": i, "fitness": rows[(ship or len(rows)) - 1],
                      "final": rows[(ship or len(rows)) - 1], "max": max(rows),
                      "shipped_checkpoint": ship, "restored_checkpoint": rest,
                      "checkpoints": curve})
    res = TrainResult(cand_id=cand.cand_id, candidate=cand, seed_metrics=seeds)
    res.checkpoints = training._mean_curve([m["checkpoints"] for m in seeds])
    return res


def _score(res, **overrides):
    registry.load_all()
    overrides.setdefault("evaluate.fitness.metric", "task_success")
    ctx = Context(cfg=load("eureka", profile="tester", overrides=overrides), budget=Budget(),
                  env=registry.get("env", "toy_reacher")({}))
    (rep,) = registry.get("fitness_source", "ground_truth_metric")(ctx, None, [res])
    return rep


def test_checkpoint_aggregation_final_scores_the_shipped_row_when_one_was_restored():
    rows = [0.1, 0.9, 0.4]
    # restored the second checkpoint -> `final` scores 0.9, not the curve's last
    rep = _score(_result([rows], shipped=[2], restored=[2]),
                 **{"evaluate.fitness.checkpoint_aggregation": "final"})
    assert rep.fitness == pytest.approx(0.9)
    assert rep.meta["fitness_from"] == "shipped_checkpoint"
    assert rep.meta["restored_checkpoints"] == [2]
    assert rep.meta["checkpoint_aggregation_applied"] is True
    # nothing restored -> the ordinary curve path, byte-for-byte as before
    rep = _score(_result([rows], shipped=[3], restored=[None]),
                 **{"evaluate.fitness.checkpoint_aggregation": "final"})
    assert rep.fitness == pytest.approx(0.4)
    assert "fitness_from" not in rep.meta
    # `max_over_checkpoints` describes the curve and is untouched by a restore
    rep = _score(_result([rows], shipped=[2], restored=[2]),
                 **{"evaluate.fitness.checkpoint_aggregation": "max_over_checkpoints"})
    assert rep.fitness == pytest.approx(0.9) and "fitness_from" not in rep.meta


def test_the_shipped_rows_are_read_per_seed_and_handed_to_seed_aggregation_per_seed():
    """Two seeds, two different restored rounds: each seed's OWN shipped row is
    read (0.9 and 0.7; the seed-mean curve has no row describing either
    policy) and the two reach `seed_aggregation` as TWO values -- the same
    shape the ordinary path hands it (one value per seed, off each seed's own
    curve), so a candidate that restored is scored under the same seed
    semantics as a pool-mate that did not. Handing it ONE pooled value would
    make `seed_aggregation: min` equal `mean` on every multi-seed run."""
    a, b = [0.1, 0.9, 0.4], [0.7, 0.2, 0.3]
    rep = _score(_result([a, b], shipped=[2, 1], restored=[2, 1]),
                 **{"evaluate.fitness.checkpoint_aggregation": "final",
                    "evaluate.fitness.seed_aggregation": "min"})
    assert rep.per_seed_fitness == pytest.approx([0.9, 0.7])
    assert rep.fitness == pytest.approx(0.7)          # `min` over the two seeds
    assert rep.meta["shipped_per_seed"] == pytest.approx([0.9, 0.7])
    assert rep.meta["n_seeds"] == 2
    assert rep.meta["n_checkpoints"] == 6             # rows read, summed over seeds
    # the control: nothing restored -> each seed's own last row, still per seed
    rep = _score(_result([a, b], shipped=[3, 3], restored=[None, None]),
                 **{"evaluate.fitness.checkpoint_aggregation": "final",
                    "evaluate.fitness.seed_aggregation": "min"})
    assert rep.per_seed_fitness == pytest.approx([0.4, 0.3]) and rep.meta["n_checkpoints"] == 6
    assert rep.fitness == pytest.approx(0.3)
    # one seed restored, the other shipped its last: both read off their own row
    rep = _score(_result([a, b], shipped=[2, 3], restored=[2, None]),
                 **{"evaluate.fitness.checkpoint_aggregation": "final"})
    assert rep.meta["shipped_per_seed"] == pytest.approx([0.9, 0.3])
    assert rep.per_seed_fitness == pytest.approx([0.9, 0.3])
    assert rep.fitness == pytest.approx(0.6)          # `mean`, the default


def test_best_by_reward_is_refused_on_the_planner_backend():
    with pytest.raises(ConfigError, match="checkpoint_selection"):
        load("eureka", profile="tester",
             overrides={"train.backend": "none", "train.checkpoint_selection": "best_by_reward"})
    # the same pair with `final` is not a coherence problem
    load("eureka", profile="tester", overrides={"train.backend": "none"})


@pytest.mark.parametrize("bad", [-0.01, 1.5])
def test_min_delta_outside_the_unit_interval_is_refused(bad):
    """A fraction of the curve's range: below 0 the guard never fires, above 1
    it always does and `best_by_reward` silently becomes `final`."""
    with pytest.raises(ConfigError, match="min_delta"):
        load("eureka", profile="tester",
             overrides={"train.checkpoint_selection_cfg.min_delta": bad})
    load("eureka", profile="tester", overrides={"train.checkpoint_selection_cfg.min_delta": 0.0})
    load("eureka", profile="tester", overrides={"train.checkpoint_selection_cfg.min_delta": 1.0})


#: The method points that are NOT a published method: this paper's own ERA-U and
#: ERA-S points, which extend `hillclimb/v4_peak_noes40` and so inherit the rule the
#: peak cells were measured under (`test_the_peak_arms_name_best_own_return`).
_OWN_METHODS = {"era_u", "era_s"}


def test_no_config_in_the_corpus_selects_a_checkpoint():
    """Every published method ships its final weights; a config that quietly
    turned this on would no longer be the paper's point."""
    from conftest import PAPER_CONFIGS
    seen = set()
    for path in PAPER_CONFIGS:
        cfg = load(path.stem)
        if path.stem in _OWN_METHODS:
            seen.add(path.stem)
            assert cfg.get("train.checkpoint_selection") == "best_own_return", path.name
            continue
        assert cfg.get("train.checkpoint_selection") == "final", path.name
    assert seen == _OWN_METHODS, "an exempted config is gone: the exemption is stale"


# --------------------------------------------------------------------------
# `best_own_return` IS the hill-climb's rule; `best_by_reward` is not
# --------------------------------------------------------------------------
#
# `hillclimb/v4_peak_noes40` (ERA-U) selects the peak own-reward checkpoint,
# and `best_own_return` is meant to be exactly the hill-climb recipe's rule.
# That is an equivalence to DEMONSTRATE on the checkpoint index, not to state --
# so the reference below writes the rule out as a spec (first finite argmax,
# restored iff it is not the last row), and the implemented rule is held to it
# on curves with ties, NaNs, flat runs and a peak at the end. The two cases where `best_by_reward` picks a DIFFERENT checkpoint
# are then named, which is why the hill-climb rule is its own enum value rather
# than an alias.


def _reference_best_checkpoint_index(curve):
    """Reference rule: the first finite argmax of `reward_return`."""
    import math
    best = None
    for i, p in enumerate(curve):
        r = float(p.get("reward_return", float("nan")))
        if not math.isfinite(r):
            continue
        if best is None or r > float(curve[best]["reward_return"]):
            best = i
    return best


def _reference_restores(curve):
    """Reference rule: hand on the peak iff it is not the last row."""
    idx = _reference_best_checkpoint_index(curve)
    return idx if (idx is not None and idx != len(curve) - 1) else None


def _curve(vals):
    return [{"round": float(i + 1), "reward_return": v} for i, v in enumerate(vals)]


_CURVES = [
    [0.2, 0.9, 0.9, 0.4],          # interior tie
    [0.0, 1.0, 1.0],               # tie WITH the last row
    [0.0, 1.0, 0.99],              # peak just inside a 2% margin
    [0.0, 0.5, 1.0],               # monotone: the last is the best
    [1.0, 1.0, 1.0],               # flat
    [float("nan"), 0.1],           # non-finite first row
    [0.3, float("nan"), 0.2],      # non-finite interior peak-looking row
    [float("nan")],                # nothing finite
    [],                            # empty
]


@pytest.mark.parametrize("vals", _CURVES, ids=[str(v) for v in _CURVES])
def test_best_own_return_matches_the_reference_hillclimb_rule(vals):
    curve = _curve(vals)
    idx, _reason = training._select_checkpoint(curve, "best_own_return", 0.02)
    assert idx == _reference_restores(curve)
    # `min_delta` is not read by this rule: any value, same checkpoint
    assert training._select_checkpoint(curve, "best_own_return", 1.0)[0] == idx


def test_best_own_return_holds_on_random_curves_too():
    rng = np.random.default_rng(0)
    for _ in range(500):
        n = int(rng.integers(1, 12))
        vals = list(np.round(rng.normal(size=n), 1))  # rounding manufactures ties
        if rng.random() < 0.2:
            vals[int(rng.integers(0, n))] = float("nan")
        curve = _curve(vals)
        assert training._select_checkpoint(curve, "best_own_return", 0.02)[0] == _reference_restores(curve), vals


def test_the_snapshot_decision_agrees_with_the_post_loop_rule_for_both_rules():
    """`_running_best` decides WHEN to copy the parameters; the post-loop rule
    decides WHICH row ships. If they disagreed the backend would report
    `no_snapshot` and ship the last row -- plausibly -- so for every prefix of
    every curve the last index at which the snapshot was taken must be the
    index the rule then selects (when it selects one)."""
    for vals in _CURVES:
        curve = _curve(vals)
        for rule in ("best_by_reward", "best_own_return"):
            snap_idx = -1
            for k in range(1, len(curve) + 1):
                if training._running_best(curve[:k], rule):
                    snap_idx = k - 1
            idx, _ = training._select_checkpoint(curve, rule, 0.0)
            if idx is not None:
                assert snap_idx == idx, (vals, rule, snap_idx, idx)
        assert training._running_best(curve, "final") is False


def test_best_by_reward_differs_from_best_own_return_in_exactly_the_tie_and_the_margin():
    """The two cases that made `best_own_return` a value and not an alias."""
    sel = training._select_checkpoint
    # (1) a tie: the hill-climb rule takes the FIRST equal peak, `best_by_reward` the LATEST
    tie = _curve([0.2, 0.9, 0.9, 0.4])
    assert sel(tie, "best_own_return", 0.0)[0] == 1
    assert sel(tie, "best_by_reward", 0.0)[0] == 2
    # ... and a tie with the last row: the hill-climb rule restores the earlier
    # equal peak, `best_by_reward` ships the last row
    tie_last = _curve([0.0, 1.0, 1.0])
    assert sel(tie_last, "best_own_return", 0.0) == (1, "restored")
    assert sel(tie_last, "best_by_reward", 0.0) == (None, "last_is_best")
    # (2) the margin: inside min_delta x range `best_by_reward` ships the last row
    near = _curve([0.0, 1.0, 0.99])
    assert sel(near, "best_own_return", 0.02) == (1, "restored")
    assert sel(near, "best_by_reward", 0.02) == (None, "within_margin")
    # Outside both, on a curve with no ties, they agree
    clear = _curve([0.0, 1.0, 0.5, 0.7])
    assert sel(clear, "best_own_return", 0.02) == sel(clear, "best_by_reward", 0.02) == (1, "restored")


def test_the_peak_arms_name_best_own_return():
    """`hillclimb/v4_peak*` are the hill-climb peak configurations and must
    resolve to the rule those cells were measured under, not `best_by_reward`'s
    guarded one."""
    for name in ("hillclimb/v4_peak", "hillclimb/v4_peak_noes40", "era_u", "era_s"):
        assert load(name)["train.checkpoint_selection"] == "best_own_return", name
