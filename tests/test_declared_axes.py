"""Declared keys that no stage reads are REFUSED when set to a value the code
cannot honour, instead of validating, moving the hash and running the same
method (a declared key that nothing reads is a fabricated pin).

Two shapes:

* `problem.reward_representation` and `problem.search_space_mode` are §0's
  DESCRIPTIVE axes: the keys that dispatch are `generate.output.format` /
  `generate.generator_backend` and the `pre` phases. `_check_coherence` now
  ties each value to the dispatching value(s), so `-s
  problem.reward_representation=free_form_code` on `rda` -- which would
  otherwise validate and produce a byte-identical run under a new hash -- is
  refused with a message naming both keys.
* `select.retrain_before_select` and `update.meta.prompt_optimizer` are declared
  and implemented by nothing. Any non-default value is refused as "declared but
  not implemented" until an implementation lands; the defaults still validate.
"""

from __future__ import annotations

import pytest

from bird.config import ConfigError, load


# --------------------------------------------------------------------------
# problem.reward_representation <-> generate.output.format / generator_backend
# --------------------------------------------------------------------------


def test_the_rda_free_form_ablation_is_refused_naming_both_keys() -> None:
    """The failure scenario in this module's docstring, verbatim."""
    with pytest.raises(ConfigError) as exc:
        load("rda", profile="tester",
             overrides={"problem.reward_representation": "free_form_code"})
    msg = str(exc.value)
    assert "problem.reward_representation" in msg
    assert "generate.output.format" in msg
    assert "component_dict_plus_weights" in msg


@pytest.mark.parametrize("value", ["weighted_components", "template_dsl", "tabular"])
def test_a_representation_without_its_dispatching_format_is_refused(value: str) -> None:
    """eureka dispatches `component_dict_return` from the `llm` backend, which
    implements only `free_form_code`."""
    with pytest.raises(ConfigError) as exc:
        load("eureka", profile="tester", overrides={"problem.reward_representation": value})
    assert "problem.reward_representation" in str(exc.value)


def test_a_format_that_implies_another_representation_is_refused() -> None:
    """The other direction: moving the dispatching key alone leaves the
    declaration lying about the run."""
    with pytest.raises(ConfigError) as exc:
        load("eureka", profile="tester",
             overrides={"generate.output.format": "component_dict_plus_weights"})
    assert "problem.reward_representation" in str(exc.value)


def test_the_consistent_pair_moves_together() -> None:
    cfg = load("eureka", profile="tester", overrides={
        "problem.reward_representation": "weighted_components",
        "generate.output.format": "component_dict_plus_weights"})
    assert cfg["problem.reward_representation"] == "weighted_components"
    for name in ("gt", "rda"):
        cfg = load(name, profile="tester")
        assert cfg["problem.reward_representation"] == "weighted_components"
        assert cfg["generate.output.format"] == "component_dict_plus_weights"


def test_tabular_is_the_enumeration_backend_and_only_that() -> None:
    """Singh's point is the one config at `tabular`, and its dispatch is the
    generator backend, not the output format."""
    cfg = load("singh_orp")
    assert cfg["problem.reward_representation"] == "tabular"
    assert cfg["generate.generator_backend"] == "exhaustive_enumeration"
    with pytest.raises(ConfigError) as exc:
        load("singh_orp", overrides={"problem.reward_representation": "free_form_code"})
    assert "generate.generator_backend" in str(exc.value)
    with pytest.raises(ConfigError):
        load("eureka", profile="tester",
             overrides={"generate.generator_backend": "exhaustive_enumeration"})


# --------------------------------------------------------------------------
# problem.search_space_mode <-> the pre phases that stage a surface
# --------------------------------------------------------------------------


def test_staged_with_nothing_to_stage_is_refused() -> None:
    with pytest.raises(ConfigError) as exc:
        load("eureka", profile="tester", overrides={"problem.search_space_mode": "staged"})
    msg = str(exc.value)
    assert "problem.search_space_mode" in msg and "nothing to stage" in msg
    # two surfaces but no phase that stages them is the other refusal
    with pytest.raises(ConfigError) as exc:
        load("limen", profile="tester", overrides={"problem.search_space_mode": "staged"})
    msg = str(exc.value)
    assert "problem.search_space_mode" in msg and "dr_generation" in msg and "pre=" in msg


def test_joint_over_dreurekas_staged_phases_is_refused() -> None:
    """DrEureka's staging IS `pre: [rapp, dr_generation]`; declaring `joint`
    beside it would otherwise be a hash-moving no-op."""
    assert load("dreureka")["problem.search_space_mode"] == "staged"
    with pytest.raises(ConfigError) as exc:
        load("dreureka", overrides={"problem.search_space_mode": "joint"})
    msg = str(exc.value)
    assert "problem.search_space_mode" in msg and "dr_generation" in msg


def test_limens_joint_point_still_loads() -> None:
    cfg = load("limen", profile="tester")
    assert cfg["problem.search_space_mode"] == "joint"
    assert cfg["problem.search_space"] == ["reward", "observation"]


# --------------------------------------------------------------------------
# declared, not implemented: refuse anything but the default
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key,value", [
    ("select.retrain_before_select", True),
    ("update.meta.prompt_optimizer", "gepa"),
    ("update.meta.prompt_optimizer", "manual"),
])
def test_a_declared_unimplemented_key_refuses_a_non_default_value(key: str, value) -> None:
    with pytest.raises(ConfigError) as exc:
        load("eureka", profile="tester", overrides={key: value})
    msg = str(exc.value)
    assert key in msg and "not implemented" in msg


def test_the_unimplemented_keys_still_validate_at_their_defaults() -> None:
    cfg = load("eureka", profile="tester")
    assert cfg["select.retrain_before_select"] is False
    assert cfg["update.meta.prompt_optimizer"] == "none"
    # the explicit default is not a "value" -- it is the absence of the mechanism
    load("eureka", profile="tester", overrides={"select.retrain_before_select": False,
                                                "update.meta.prompt_optimizer": "none"})


# --------------------------------------------------------------------------
# the thompson_success guard is keyed on the ACCESS declaration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rda", "gt"])
def test_a_gt_free_method_cannot_take_the_ground_truth_allocator(name: str) -> None:
    """`problem.fitness_access: none` with a VLM / preference source must not
    validate beside `thompson_success` -- the allocator reads the env's
    success flag every slice, so the unsupervised column would read the truth."""
    cfg = load(name, profile="tester")
    assert cfg["problem.fitness_access"] == "none" and cfg["evaluate.fitness.source"] != "none"
    with pytest.raises(ConfigError) as exc:
        load(name, profile="tester", overrides={"train.interaction": "shared_population",
                                                "train.interaction_allocator": "thompson_success"})
    msg = str(exc.value)
    assert "thompson_success" in msg
    assert "problem.fitness_access" in msg and "evaluate.fitness.source" in msg
    # the paper's own control is still allowed on the same GT-free config
    load(name, profile="tester", overrides={"train.interaction": "shared_population",
                                            "train.interaction_allocator": "uniform"})


def test_a_method_that_declares_ground_truth_keeps_the_allocator() -> None:
    """The other direction: lares declares `ground_truth_metric` and publishes
    `thompson_success`; the guard must not take the method's own value away."""
    cfg = load("lares", profile="tester")
    assert cfg["problem.fitness_access"] == "ground_truth_metric"
    assert cfg["train.interaction_allocator"] == "thompson_success"
    load("lares", profile="tester", overrides={"problem.fitness_access": "success_indicator"})
    with pytest.raises(ConfigError):
        load("lares", profile="tester", overrides={"problem.fitness_access": "demonstrations"})
