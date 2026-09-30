"""§4's two-step collapse: `checkpoint_aggregation` PER SEED, then
`seed_aggregation` ACROSS seeds -- in that order, on the seeds' own curves.

The schema's §4 declares the order and `_per_seed_values` documents why it is
load-bearing: `max_over_checkpoints` of a seed-MEAN curve is the max of an
average, which hides exactly the per-seed instability the max is biased by.
`result.checkpoints` is `_mean_curve(per_seed_curves)`, whose rows carry an
AVERAGED `seed` field, so grouping it by seed yields one pooled curve however
many seeds trained: `seed_aggregation` would receive a list of length one on
every multi-seed run, `min` would equal `mean`, `per_seed_fitness` would have
length one, `meta.n_seeds` would read 1 on every 3-seed LIMEN candidate, and
`select.significance` / `tie_break: lowest_variance` / the `seed_std` objective
could never see a second seed. The two orders coincide for `final` + `mean` on
equal-length curves -- so a published multi-seed point (`limen`,
`text2reward_zeroshot`) scores the right NUMBER even with the key inert -- so
every case below is one where they differ.

None of this runs a learner: the results are hand-built the way a backend
leaves them (per-seed curves in the seed rows, the seed-mean curve on the
result), and the pipeline is driven through the registered `fitness_source`.
"""

from __future__ import annotations

import pytest

from bird import registry
from bird.budget import Budget
from bird.components import evaluation, training
from bird.config import load
from bird.context import Context
from bird.types import Candidate, TrainResult

_REWARD = "def compute_reward(s, a, s2):\n    return 0.0\n"


def _result(per_seed_rows, *, mean_curve=True):
    """A TrainResult as a backend leaves it: each seed's curve in its seed row,
    the seed-MEAN curve on the result (`_mean_curve` truncates to the shortest
    curve, as the backends do)."""
    cand = Candidate(cand_id="c0000", iteration=0, reward_code=_REWARD)
    seeds = []
    for i, rows in enumerate(per_seed_rows):
        curve = [{"round": float(r + 1), "step": float(100 * (r + 1)), "seed": float(i),
                  "fitness": f, "task_success": f, "reward_return": 0.0}
                 for r, f in enumerate(rows)]
        seeds.append({"seed": i, "fitness": rows[-1], "final": rows[-1], "max": max(rows),
                      "shipped_checkpoint": len(rows), "restored_checkpoint": None,
                      "checkpoints": curve})
    res = TrainResult(cand_id=cand.cand_id, candidate=cand, seed_metrics=seeds)
    if mean_curve:
        res.checkpoints = training._mean_curve([m["checkpoints"] for m in seeds])
    return res


def _score(res, **overrides):
    registry.load_all()
    overrides.setdefault("evaluate.fitness.metric", "task_success")
    # `eureka` is `max_over_checkpoints`; every test here names the rule it
    # means, and the ones about SEED aggregation mean the plain last row.
    overrides.setdefault("evaluate.fitness.checkpoint_aggregation", "final")
    ctx = Context(cfg=load("eureka", profile="tester", overrides=overrides), budget=Budget(),
                  env=registry.get("env", "toy_reacher")({}))
    (rep,) = registry.get("fitness_source", "ground_truth_metric")(ctx, None, [res])
    return rep


#: Two seeds, three checkpoints, chosen so the two orders disagree:
#:   max THEN mean (documented): max(0, 1, 0) = 1, max(1, 0, 0) = 1 -> mean 1.0
#:   mean THEN max (the bug):    mean curve = (0.5, 0.5, 0.0)      -> max 0.5
_ANTI = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


def test_max_over_checkpoints_is_taken_per_seed_and_then_aggregated():
    rep = _score(_result(_ANTI),
                 **{"evaluate.fitness.checkpoint_aggregation": "max_over_checkpoints",
                    "evaluate.fitness.seed_aggregation": "mean"})
    assert rep.per_seed_fitness == pytest.approx([1.0, 1.0]), (
        "per_seed_fitness must hold each seed's own reduced value, not the "
        "pooled curve's")
    assert rep.fitness == pytest.approx(1.0), (
        "max-then-mean is 1.0; 0.5 is the max of the seed-mean curve, the "
        "order §4 forbids")
    assert rep.meta["n_seeds"] == 2
    assert rep.meta["checkpoint_aggregation_applied"] is True
    assert rep.meta["n_checkpoints"] == 6


def test_seed_aggregation_min_and_mean_are_different_numbers_on_two_seeds():
    """The §9 ablation `seed_aggregation: min` must move the scalar. Seeds
    (0.0, 0.0, 1.0)-shaped: A finishes (0.0, 1.0), B finishes (0.4, 0.4)."""
    a = _result([[0.1, 0.0], [0.2, 1.0]])
    b = _result([[0.3, 0.4], [0.3, 0.4]])
    mean_a = _score(a, **{"evaluate.fitness.seed_aggregation": "mean"}).fitness
    min_a = _score(a, **{"evaluate.fitness.seed_aggregation": "min"}).fitness
    mean_b = _score(b, **{"evaluate.fitness.seed_aggregation": "mean"}).fitness
    min_b = _score(b, **{"evaluate.fitness.seed_aggregation": "min"}).fitness
    assert mean_a == pytest.approx(0.5) and min_a == pytest.approx(0.0)
    assert mean_b == pytest.approx(0.4) and min_b == pytest.approx(0.4)
    assert mean_a > mean_b and min_a < min_b, (
        "`min` must be able to reverse `mean`'s ranking; if it cannot, the key "
        "is inert")
    for agg in ("median", "iqm"):
        rep = _score(a, **{"evaluate.fitness.seed_aggregation": agg})
        assert len(rep.per_seed_fitness) == 2 and rep.meta["n_seeds"] == 2


def test_final_reads_each_seeds_own_last_checkpoint_when_one_was_pruned():
    """`_mean_curve` truncates to the shortest curve, so on the pooled path a
    seed pruned at round 2 makes `final` read EVERY seed at round 2. Each
    seed's own last row is the documented reading."""
    rep = _score(_result([[0.1, 0.2, 0.9], [0.1, 0.3]]),
                 **{"evaluate.fitness.checkpoint_aggregation": "final"})
    assert rep.per_seed_fitness == pytest.approx([0.9, 0.3])
    assert rep.fitness == pytest.approx(0.6)


def test_a_single_seed_scores_bit_identically_to_the_pooled_reading():
    """The paper protocol is `seeds_per_candidate: 1` everywhere but LIMEN,
    and per-seed aggregation must not move a single-seed number by one ulp: the
    mean of one row is that row. Compared against the seed-mean curve read the
    pooled way, with `==` and not approx."""
    rows = [0.13, 0.71, 0.42, 0.9, 0.35]
    res = _result([rows])
    for ckpt in ("final", "max_over_checkpoints", "auc", "last_k_mean", "iqm"):
        rep = _score(res, **{"evaluate.fitness.checkpoint_aggregation": ckpt})
        agg = registry.get("checkpoint_aggregation", ckpt)
        pooled = evaluation._series(evaluation._checkpoint_curves(res)[0],
                                    ("task_success", "fitness"))
        assert rep.fitness == float(agg(pooled)), ckpt
        assert rep.per_seed_fitness == [rep.fitness] and rep.meta["n_seeds"] == 1
        assert rep.meta["n_checkpoints"] == len(rows)


def test_a_result_without_seed_curves_still_scores_off_the_pooled_curve():
    """A hand-built result (or an artifact from before the seed rows carried
    curves) has only `result.checkpoints`; it is one pooled curve and says so
    through `n_seeds`, never a failure."""
    res = _result([[0.1, 0.5], [0.3, 0.7]])
    for m in res.seed_metrics:
        del m["checkpoints"]
    rep = _score(res)
    assert rep.fitness == pytest.approx(0.6)   # the mean curve's last row
    assert rep.meta["n_seeds"] == 1 and rep.meta["checkpoint_aggregation_applied"] is True


def test_no_published_multi_seed_point_moves_under_the_fix():
    """`limen` (3 seeds) is `final` + `mean` with no pruner: mean of each
    seed's last row IS the mean curve's last row, so the published number is
    unchanged and only the artifact's per-seed detail grows."""
    cfg = load("limen")
    assert cfg["train.seeds_per_candidate"] == 3
    assert cfg["evaluate.fitness.checkpoint_aggregation"] == "final"
    assert cfg["evaluate.fitness.seed_aggregation"] == "mean"
    assert cfg["train.pruning"] == "none"
    rows = [[0.2, 0.5, 0.8], [0.1, 0.4, 0.6], [0.3, 0.3, 0.7]]
    rep = _score(_result(rows), **{"evaluate.fitness.checkpoint_aggregation": "final",
                                   "evaluate.fitness.seed_aggregation": "mean"})
    assert rep.fitness == pytest.approx(0.7)
    assert rep.per_seed_fitness == pytest.approx([0.8, 0.6, 0.7]) and rep.meta["n_seeds"] == 3
