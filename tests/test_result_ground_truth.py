"""`result.json` must say WHAT its headline number is, not just what it equals.

`returned_fitness` is whatever `evaluate.fitness.source` produced, and across the
published configs that is four different quantities -- `ground_truth_metric`,
`vlm_score`, `preference_bt`, `none`. Two methods whose fitness sources differ can
report numbers on different scales for equally good candidates: a
`preference_bt` score well below a `ground_truth_metric` score of 1.000 reads as
worse even when, on ground truth, both returned a candidate at exactly 1.000.

These tests pin the two halves of the remedy: the number carries its own
provenance, and a common scale is recorded next to it.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _run(tmp_path, config="eureka"):
    import importlib.util
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_result_records_which_quantity_returned_fitness_is(tmp_path):
    """A bare float in a field named `fitness` is not enough: four sources feed it."""
    mod = _run(tmp_path)
    cfg = mod.load("eureka", profile="tester", overrides={"output.tracker": "none",
                                               "output.dir": str(tmp_path)})
    mod.registry.load_all()
    result = mod.run(cfg)
    if cfg["evaluate.fitness.source"] == "native":
        # `native` resolves PER TASK (bird/native_signal.py): the run records the
        # channel that actually produced the number -- `native_success` where the
        # task ships a success, else `native_reward` -- not the umbrella value,
        # so a reader of result.json cannot mistake a return for a success rate.
        # eureka.yaml pins `native`; the tester profile's toy env ships a success.
        assert result["returned_fitness_source"] in ("native_success", "native_reward"), result
    else:
        assert result["returned_fitness_source"] == cfg["evaluate.fitness.source"]
    # The reduction rule too -- it decides whether the scalar is a final or a max.
    assert result["fitness_checkpoint_aggregation"] == \
        cfg["evaluate.fitness.checkpoint_aggregation"]


def test_result_records_ground_truth_on_the_common_scale(tmp_path):
    mod = _run(tmp_path)
    cfg = mod.load("eureka", profile="tester", overrides={"output.tracker": "none",
                                               "output.dir": str(tmp_path)})
    mod.registry.load_all()
    gt = mod.run(cfg)["returned_task_metric"]
    assert gt is not None, "a run that trained something must record ground truth"
    # EVERY statistic is named. A ground-truth number that did not say whether it
    # was a final, a max or an AUC would recreate the trap this field closes.
    for key in ("final", "max", "auc", "n_seeds", "source", "aggregation_over_seeds"):
        assert key in gt, f"{key} missing -- the value must carry its own meaning"
    assert gt["source"] == "env.task_metric"
    assert gt["n_seeds"] >= 1
    assert 0.0 <= gt["final"] <= 1.0 and 0.0 <= gt["max"] <= 1.0
    # `max` is over checkpoints of the same run, so it can never be below `final`.
    assert gt["max"] >= gt["final"] - 1e-9


def test_ground_truth_is_none_not_zero_when_nothing_trained():
    """`l2r` trains nothing (`train.backend: none`). Absent must not read as 0.0."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._ground_truth_of(None) is None

    class _R:
        seed_metrics = []

    class _Rep:
        result = _R()

    assert mod._ground_truth_of(_Rep()) is None

