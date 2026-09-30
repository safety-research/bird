"""A non-native arm is shown no success on a task that ships none.

`_section_numeric`'s native branch reads the native curve keys; its ELSE
branch -- every arm whose `evaluate.fitness.source` is not
native/native_success/native_reward -- would otherwise print
`evaluate.fitness.metric` (default `task_success`) off the BIRD curve, and a
rollout renderer gating `success=` on `include_task_metric` alone would print
the flag. On `assistax_feeding` (`discrete_success.kind: continuous_only`) a
CARD arm would then be told, each iteration,

    task_success: [0.00, 0.00, 0.00, 0.26, 0.80, 0.86, 0.80, 0.86, 0.93, 0.67]

plus a `success=` flag per rollout and "the agent solves the task", for a
criterion assistax does not define -- `EnvAdapter.success()` there is a
threshold this repo chose on a per-step check this repo wrote, which the spec
itself records in `no_discrete_success_because`.

The gate is `has_native_success`, not "has any native signal": assistax ships
`reward.human.kind: published_dense`, so `native_curve_keys` returns
`gt_return` for it and a gate on that would leave the line in place. The two tasks below differ in `discrete_success.kind` and in nothing
else the renderers read -- same config, same result object, same rollouts.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from bird.components import evaluation as E
from bird.native_signal import has_native_success, kinds_for_env
from bird.types import Candidate, CandidateReport, TrainResult, Trajectory

#: continuous_only + published_dense: a native REWARD, no native success.
CONTINUOUS_ONLY = "assistax_feeding"
#: discrete + published_dense: Meta-World's own success check.
NATIVE_SUCCESS = "mt10_reach-v3"

_CODE = "def compute_reward(s, a, s2):\n    return 1.0, {'reach': 1.0}\n"

#: The shape of the leaked reading: a `task_success` column on the pooled
#: checkpoint curve, which is `custom_metric` on both backends.
_CURVE = [{"round": r, "task_success": v, "reward_return": 0.1 * r, "gt_return": 2.0 * r}
          for r, v in enumerate([0.0, 0.0, 0.0, 0.26, 0.80, 0.86, 0.80, 0.86, 0.93, 0.67], 1)]


#: Every way a success-shaped reading reaches the generator through
#: `feedback_default`: the numeric section's metric line and its rollout
#: success mean, the rendered rollouts' header flag, and the terminal
#: sentence. One needle per emission site, so a gate that stops covering one
#: of them fails a NAMED case rather than shrinking a single assertion.
_NEEDLES = ["task_success", "success: mean", "success=", "solves the task"]


def _traj(n=10, *, success=True):
    return Trajectory(states=np.zeros((n, 3)), actions=np.zeros((n, 1)),
                      rewards=[0.1] * n, component_values={}, success=success,
                      length=n, ret=float(n * 0.1))


def _report():
    cand = Candidate("c0001", 0, _CODE)
    res = TrainResult(cand_id="c0001", candidate=cand,
                      trajectories=[_traj(success=True), _traj(success=False)],
                      checkpoints=list(_CURVE), component_traces={})
    # `fitness.source: none` produces exactly this report shape
    # (evaluation.py, the `none` fitness_source: `fitness=None,
    # fitness_source="none"`), which is also why the header names no source.
    return CandidateReport("c0001", cand, res, fitness=None,
                           fitness_source="none", per_seed_fitness=[])


def _ctx(env_id, source="none"):
    """The CARD arm's reading of the keys these renderers consult:
    `fitness.source: none` (so the ELSE branch) and `include_task_metric:
    true` (so nothing else suppresses the flag)."""
    cfg = {"problem.env_id": env_id,
           "problem.task_description": "bring the spoon to the mouth",
           "evaluate.fitness.source": source,
           "evaluate.fitness.metric": "task_success",
           "evaluate.feedback.include_task_metric": True,
           "evaluate.feedback.numeric_reflection": True,
           "evaluate.feedback.granularity": "scalar_only",
           "evaluate.feedback.trajectory_examples": "best_and_worst",
           "evaluate.feedback.trajectory_sample_interval": 10,
           "evaluate.rollouts_per_candidate": 2}
    return SimpleNamespace(cfg=cfg, env=SimpleNamespace(horizon=10), evaluator=None,
                           human=None, rng=random.Random(0))


def _feedback(env_id, source="none"):
    text, _channel = E.feedback_default(_ctx(env_id, source), SimpleNamespace(), _report())
    return text


def test_the_two_tasks_differ_in_exactly_the_fact_under_test():
    """Both ship a reference reward; only one ships a success check. If this
    stops holding, the pair below is no longer a controlled comparison."""
    cont_ds, cont_rw = kinds_for_env(env_id=CONTINUOUS_ONLY)
    nat_ds, nat_rw = kinds_for_env(env_id=NATIVE_SUCCESS)
    assert cont_rw == nat_rw == "published_dense"
    assert (cont_ds, nat_ds) == ("continuous_only", "discrete")
    assert not has_native_success(cont_ds) and has_native_success(nat_ds)


@pytest.mark.parametrize("needle", _NEEDLES)
def test_a_continuous_only_task_is_shown_no_success(needle):
    text = _feedback(CONTINUOUS_ONLY)
    assert text, "the feedback must still be built"
    assert needle not in text, f"{needle!r} in:\n{text}"


@pytest.mark.parametrize("needle", _NEEDLES)
def test_a_native_success_task_still_shows_it(needle):
    """The control. The gate must cost nothing where the benchmark does ship
    the check -- otherwise it is not a fidelity fix, it is a mute button."""
    assert needle in _feedback(NATIVE_SUCCESS), _feedback(NATIVE_SUCCESS)


def test_the_curve_itself_still_reaches_a_native_arm():
    """The suppression is of the success-shaped READING, not of the numeric
    channel: the continuous_only task still gets its reflection section, its
    episode return and its rendered rollouts."""
    text = _feedback(CONTINUOUS_ONLY)
    assert "Training statistics, sampled at checkpoints:" in text
    assert "episode return:" in text
    assert "t=0" in text, "the per-step rendering must survive the gate"


def test_a_native_arm_on_the_same_task_keeps_its_native_curve():
    """The other half of the control, and the one that says this is a gate on
    the SIGNAL rather than on the task: eureka's key set (`fitness.source:
    native`) on the same continuous_only task still gets a metric line -- the
    reference reward's return, which assistax does ship -- while the
    success-shaped reading stays suppressed."""
    text = _feedback(CONTINUOUS_ONLY, source="native")
    assert "reference reward return:" in text, text
    for needle in _NEEDLES:
        assert needle not in text, f"{needle!r} in:\n{text}"


@pytest.mark.parametrize("env_id, want", [(CONTINUOUS_ONLY, False), (NATIVE_SUCCESS, True)])
def test_the_summariser_digest_takes_the_same_gate(env_id, want):
    """`_behaviour_digest` is the third renderer that names the flag (RDA's
    summarisation section and the `human_score` prompt), and it is reached by
    neither of the two paths above -- it prints `success=<flag>`
    per rollout on its own. It gates on the same fact or the leak just moves."""
    digest = E._behaviour_digest(_ctx(env_id), _report().result)
    assert ("success=" in digest) is want, digest
