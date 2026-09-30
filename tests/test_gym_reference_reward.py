"""The reference reward is a per-TASK fact, and its absence is explicit everywhere.

A `GymMujoco.reference_reward` that returned the family row's reward -- the BASE
simulator's `vx - w|a|^2` -- on every task built on that simulator would include
the five whose spec declares `reward.human.kind: none` and says in words that the
built-in reward pays "the exact thing this task penalises". A backward-running
cheetah's `gt_return` would measure how fast it ran FORWARD, `pearson_curve` on
those tasks would correlate candidates against an objective the task penalises,
`phases._reference_score` (the scripted human oracle) would prefer that curve, and
`scripts/train_expert.py` would train "expert" anchors on it: the target-speed
"expert" becomes a max-speed policy, and hop-in-place's comes in below random.

Two halves, on purpose. The first is simulator-free and runs in CI: the rule
itself (`_reference_reward_shape`) over every registered gym spec, and every
consumer of `gt_return` handed an env whose `reference_reward` raises. The second
`importorskip`s the simulator and holds the LIVE adapters to the same rule, so
`has_reference_reward` and `reference_reward` can never disagree on a real task.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from bird import registry
from bird.components import evaluation, phases, training
from bird.envs.base import EnvAdapter
from bird.envs.gym_mujoco import _FAMILIES, _gym_specs, _reference_reward_shape
from bird.types import Candidate, CandidateReport, TrainResult

registry.load_all()

ALL_IDS = sorted(n for n in registry.names("env") if n.startswith("gym_"))


def _human_kind(spec) -> str:
    return str(((spec.reward or {}).get("human") or {}).get("kind"))


# --------------------------------------------------------------------------
# simulator-free: the rule, and the consumers
# --------------------------------------------------------------------------


def test_the_shape_follows_the_specs_reward_kind_not_the_simulator():
    """`ref_reward` is a FAMILY row; whether a task exposes it is the spec's
    `reward.human.kind`. Both populations are asserted non-empty so a catalogue
    that lost one side could not leave this vacuously green."""
    claimed, disowned = [], []
    for task_id, spec in _gym_specs().items():
        shape = _reference_reward_shape(spec, _FAMILIES[spec.env_id])
        if _human_kind(spec) == "none":
            disowned.append(task_id)
            assert shape is None, (
                f"{task_id}: spec disowns the built-in reward (reward.human.kind: none) "
                "but the adapter would still hand it out as `reference_reward`")
        else:
            claimed.append(task_id)
            assert shape == tuple(_FAMILIES[spec.env_id]["ref_reward"]), task_id
    assert sorted(disowned) == ["half_cheetah_backward", "half_cheetah_target_speed",
                                "hopper_hop_in_place", "reacher_hold", "swimmer_heading"]
    assert sorted(claimed) == ["half_cheetah", "hopper_hop", "humanoid_run",
                               "inverted_pendulum_balance", "reacher_reach",
                               "swimmer_forward"]


def test_two_tasks_on_one_simulator_can_differ():
    """The whole defect in one row: `HalfCheetah-v5` backs three specs, and the
    rule must split them -- forward keeps the wheel's reward, backward and
    target-speed do not."""
    specs = _gym_specs()
    fam = _FAMILIES["HalfCheetah-v5"]
    assert _reference_reward_shape(specs["half_cheetah"], fam) == ("velocity", 1.0, 0.1, False)
    assert _reference_reward_shape(specs["half_cheetah_backward"], fam) is None
    assert _reference_reward_shape(specs["half_cheetah_target_speed"], fam) is None


def test_the_base_flag_defaults_to_whether_the_method_is_overridden():
    """`has_reference_reward` is True on every adapter that overrides the method
    (all of them, today) and False on the bare base, whose method raises."""
    toy = registry.get("env", "toy_reacher")({})
    assert toy.has_reference_reward is True
    assert isinstance(toy.reference_reward(toy.reset(np.random.default_rng(0))), float)

    class _Bare(EnvAdapter):
        def _build_action_set(self):
            return np.zeros((1, 1))

        def _bounds(self):
            return np.zeros(1), np.ones(1)

    bare = _Bare()
    assert bare.has_reference_reward is False
    with pytest.raises(NotImplementedError):
        bare.reference_reward(np.zeros(1))


class _NoRef:
    """A toy env whose task disowns its reward: everything forwards to the real
    adapter except `reference_reward`, which raises the adapter contract."""

    has_reference_reward = False

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reference_reward(self, s, a=None):
        raise NotImplementedError("no reference reward on this task")


def _zero_policy(env):
    return lambda s: np.zeros(env.action_dim)


def test_rollout_reports_no_reference_return_rather_than_zero():
    """`gt_return` is None, not 0.0 -- and on an env WITH a reference it is the
    plain float it always was, so the guard changes nothing where it does not
    apply. A `_rollout` that let the NotImplementedError propagate fails
    this."""
    env = registry.get("env", "toy_reacher")({})
    traj, steps, gt = training._rollout(env, _zero_policy(env), np.random.default_rng(0))
    assert steps == traj.length > 0
    assert isinstance(gt, float) and math.isfinite(gt)

    traj2, steps2, gt2 = training._rollout(_NoRef(env), _zero_policy(env),
                                           np.random.default_rng(0))
    assert gt2 is None
    assert steps2 == steps and traj2.length == traj.length, (
        "the absent reference must not shorten or otherwise change the episode")


def test_evaluate_policy_and_the_seed_mean_carry_the_absence_through():
    """The per-checkpoint metric is None and the across-seed mean keeps it None;
    a `float(None)` in `_mean_curve` would be a TypeError."""
    env = registry.get("env", "toy_reacher")({})
    on_error = lambda exc: 0.0  # noqa: E731
    metrics, _comps, _steps, trajs = training._evaluate_policy(
        _NoRef(env), _zero_policy(env), np.random.default_rng(0), None, 2, on_error)
    assert metrics["gt_return"] is None
    assert isinstance(metrics["fitness"], float)
    assert len(trajs) == 2

    with_ref, *_ = training._evaluate_policy(
        env, _zero_policy(env), np.random.default_rng(0), None, 2, on_error)
    assert isinstance(with_ref["gt_return"], float)

    curve_a = [{"step": 1.0, "fitness": 0.1, "gt_return": None}]
    curve_b = [{"step": 1.0, "fitness": 0.3, "gt_return": 2.0}]
    mean = training._mean_curve([curve_a, curve_b])
    assert mean[0]["gt_return"] is None
    assert mean[0]["fitness"] == pytest.approx(0.2)
    assert training._mean_curve([curve_b, curve_b])[0]["gt_return"] == pytest.approx(2.0)
    assert training._opt_float(None) is None and training._opt_float(3) == 3.0


def _report(gt_curve, cand_curve=(1.0, 2.0, 3.0, 4.0)):
    cand = Candidate(cand_id="c0", iteration=0, reward_code="")
    result = TrainResult(cand_id="c0", candidate=cand,
                         checkpoints=[{"step": float(i), "reward_return": float(v),
                                       "return": float(v), "fitness": 0.1 * i,
                                       "gt_return": g}
                                      for i, (v, g) in enumerate(zip(cand_curve, gt_curve))],
                         gt_reward_curve=list(gt_curve))
    return CandidateReport(cand_id="c0", candidate=cand, result=result)


class _Cfg(dict):
    def get(self, key, default=None):  # noqa: D401 -- Config-like `.get`
        return super().get(key, default)


def test_the_similarity_and_the_oracle_see_no_channel_not_a_flat_zero():
    """`pearson_curve` is undefined (None) against an absent reference, and the
    scripted human oracle falls THROUGH a curve of Nones instead of averaging it
    (which would be a TypeError) -- it must not report the reference paid zero.

    Native-signal rule: the oracle does not fall through to
    `env.task_metric` -- that is the BIRD custom_metric, and the human oracle is
    a §4 search signal. So even though `env.task_metric` is available here (0.5),
    the oracle IGNORES it and takes the next admissible fallback
    (`mean_per_step_return`); the shipped-reward return (`gt_reward_curve`,
    native_reward) remains the preferred channel when present."""
    ctx = SimpleNamespace(cfg=_Cfg({"evaluate.similarity.reference": "gt_reward",
                                    "evaluate.similarity.role": "report"}),
                          env=SimpleNamespace(task_metric=lambda t: 0.5))
    absent = _report([None, None, None, None])
    assert evaluation._reference_curve(ctx, None, absent) == []
    assert evaluation.similarity_pearson(ctx, None, absent) is None

    present = _report([1.0, 2.0, 3.0, 4.0])
    assert evaluation._reference_curve(ctx, None, present) == [1.0, 2.0, 3.0, 4.0]
    assert evaluation.similarity_pearson(ctx, None, present) == pytest.approx(1.0)

    score, source = phases._reference_score(ctx, present)
    assert (score, source) == (pytest.approx(2.5), "gt_reward_curve")
    absent.result.trajectories = [SimpleNamespace(mean_per_step_return=9.0)]
    score, source = phases._reference_score(ctx, absent)
    # N: env.task_metric (0.5) is present but must NOT be read -> falls to the
    # next fallback (9.0), never a flat zero and never the custom metric.
    assert source == "mean_per_step_return" and score == pytest.approx(9.0), (
        "a curve of Nones is no channel; the oracle must fall through to a "
        "non-custom signal, not read task_metric and not report zero")


def test_train_expert_refuses_a_task_whose_spec_disowns_the_reward():
    """`scripts/train_expert.py` exits 2 with a reason rather than training an
    "expert" on a reward the task disowns; on an env with a reference it proceeds.
    `reward_trained_on` names the spec's pin, not the bare word."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "train_expert", Path(__file__).resolve().parents[1] / "scripts" / "train_expert.py")
    te = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(te)

    env = registry.get("env", "toy_reacher")({})
    assert te._refusal(env) is None
    why = te._refusal(_NoRef(env))
    assert why and "reward.human.kind: none" in why and "refusing" in why

    fake_gym = SimpleNamespace(name="gym_half_cheetah_backward", task="half_cheetah_backward",
                               has_reference_reward=False)
    assert "tasks/half_cheetah_backward/shared_spec.yaml" in te._refusal(fake_gym)

    on = te._reward_trained_on(SimpleNamespace(name="gym_half_cheetah"))
    assert on["method"] == "reference_reward" and on["spec"] == "half_cheetah"
    assert on["kind"] == "published_dense" and on["symbol"] == "HalfCheetahEnv._get_rew"
    # an env with no task spec at all records the absence, not a guess
    assert te._reward_trained_on(SimpleNamespace(name="no_such_env_anywhere"))["spec"] is None


# --------------------------------------------------------------------------
# with the simulator: the live adapters obey the same rule
# --------------------------------------------------------------------------


class _Short:
    """The adapter with a short horizon, so an end-to-end rollout costs five steps
    rather than a thousand. Nothing else is touched."""

    def __init__(self, env, horizon):
        self._env, self.horizon = env, horizon

    def __getattr__(self, name):
        return getattr(self._env, name)


@pytest.fixture(scope="module")
def adapters():
    pytest.importorskip("gymnasium", reason="needs `uv sync --extra metaworld`")
    pytest.importorskip("mujoco", reason="needs `uv sync --extra metaworld`")
    return {name: registry.get("env", name)({}) for name in ALL_IDS}


def test_reference_reward_is_available_iff_the_spec_claims_a_human_reward(adapters):
    """On every registered gym task: `has_reference_reward` equals
    `reward.human.kind != none`, and the call agrees with the flag -- a float
    where it is True, `NotImplementedError` naming the spec where it is False.
    An adapter with no flag that returned the family reward on all of them
    fails this."""
    specs = _gym_specs()
    seen_absent = seen_present = 0
    for name, env in adapters.items():
        spec = specs[env.task]
        expect = _human_kind(spec) != "none"
        assert env.has_reference_reward is expect, (name, _human_kind(spec))
        s = np.zeros(env.obs_dim)
        s[env.n_q] = 1.5                       # moving forward: a nonzero reward
        a = np.zeros(env.action_dim)
        if expect:
            seen_present += 1
            assert isinstance(env.reference_reward(s, a), float)
        else:
            seen_absent += 1
            with pytest.raises(NotImplementedError, match=rf"tasks/{env.task}/shared_spec.yaml"):
                env.reference_reward(s, a)
    # 11 gym tasks: the five derived tasks disown the family reward; the six
    # base tasks (humanoid_run the eleventh) keep it.
    assert seen_absent == 5 and seen_present == 6


def test_a_derived_task_has_no_gt_return_and_its_base_task_still_does(adapters):
    """End to end through `_rollout`: the backward cheetah yields `gt_return`
    None rather than the FORWARD reward; the forward cheetah on the same
    simulator keeps its reference."""
    back, fwd = adapters["gym_half_cheetah_backward"], adapters["gym_half_cheetah"]
    policy = lambda s: np.zeros(back.action_dim)  # noqa: E731
    traj_b, steps_b, gt_b = training._rollout(_Short(back, 5), policy, np.random.default_rng(0))
    traj_f, steps_f, gt_f = training._rollout(_Short(fwd, 5), policy, np.random.default_rng(0))
    assert steps_b == steps_f == 5 and traj_b.length == traj_f.length == 5
    assert gt_b is None
    assert isinstance(gt_f, float) and math.isfinite(gt_f)
