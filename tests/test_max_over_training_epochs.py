"""`max_over_training_epochs`: Eureka's fitness is the max over the FULL training series.

`eureka.py:253` is `metric_cur_max = max(tensorboard_logs[metric])` over every
logged epoch and `:256` appends exactly that to `successes`, the ranking input;
the `[::epoch_freq]` stride at `:252` is only the ~10 values verbalised in the
reflection. `max_over_checkpoints` maxes the 5-20 held-out checkpoint
evaluations instead (`MIN_CHECKPOINTS`/`MAX_CHECKPOINTS`), which is a different
statistic on a different series.

The tests below pin the thing that actually distinguishes them: a spike BETWEEN
checkpoints. The checkpoint rule cannot see it; the epoch rule must.
"""
from __future__ import annotations

import pytest

from bird import native_signal, registry
from bird.components import evaluation, training
from bird.types import Candidate, TrainResult


def _result(epoch_series=None, curve=None, cid="c0"):
    cand = Candidate(cand_id=cid, iteration=0, reward_code="")
    r = TrainResult(cand_id=cid, candidate=cand)
    row = {}
    if curve is not None:
        row["checkpoints"] = [{"fitness": v} for v in curve]
    if epoch_series is not None:
        row["training_epoch_metrics"] = list(epoch_series)
    r.seed_metrics = [row]
    return r


class _Cfg:
    def __init__(self, rule):
        self._rule = rule

    def get(self, key, default=None):
        if key == "evaluate.fitness.checkpoint_aggregation":
            return self._rule
        return default


class _Ctx:
    def __init__(self, rule):
        self.cfg = _Cfg(rule)


class _Rep:
    def __init__(self):
        self.meta = {}


def _score(rule, result):
    rep = _Rep()
    vals = evaluation._per_seed_values(_Ctx(rule), result, ["fitness"], rep)
    return vals, rep.meta


# --------------------------------------------------------------- the reducer

def test_the_value_is_registered_and_is_a_plain_max():
    fn = registry.get("checkpoint_aggregation", "max_over_training_epochs")
    assert fn([0.1, 0.9, 0.2]) == pytest.approx(0.9)
    assert fn([]) != fn([])  # nan: an empty series has no max, and says so


def test_a_spike_between_checkpoints_is_caught_by_the_new_rule_and_missed_by_the_old():
    """The whole point of the epoch rule, as one assertion.

    The policy touched 0.95 mid-training and was back to 0.2 by the next
    held-out evaluation. Eureka's number is 0.95; the checkpoint max is 0.2."""
    r = _result(epoch_series=[0.10, 0.95, 0.20, 0.18], curve=[0.1, 0.2, 0.2])
    new, meta_new = _score("max_over_training_epochs", r)
    old, meta_old = _score("max_over_checkpoints", r)
    assert new == [pytest.approx(0.95)], "the new rule must see the spike"
    assert old == [pytest.approx(0.20)], "the old rule cannot see it, by construction"
    assert meta_new["fitness_from"] == "training_epoch_series"
    assert meta_new["n_training_epochs"] == 4
    assert meta_old.get("fitness_from") != "training_epoch_series"


def test_a_missing_series_falls_back_and_SAYS_SO():
    """Non-silent, deliberately. A quiet fallback would make this rule
    indistinguishable from `max_over_checkpoints` in the artifact, which is the
    one confusion that would invalidate an epoch-vs-checkpoint comparison -- and it is
    the live case on the fasttd3 BACKEND, which records no series yet."""
    r = _result(epoch_series=None, curve=[0.1, 0.7, 0.3])
    vals, meta = _score("max_over_training_epochs", r)
    assert vals == [pytest.approx(0.7)]
    assert meta["training_epoch_series_missing"] is True


def test_the_two_step_collapse_is_per_seed():
    """Checkpoints first, then seeds -- collapsing seeds first would maximise
    over an average and hide the instability the max is biased by."""
    cand = Candidate(cand_id="c1", iteration=0, reward_code="")
    r = TrainResult(cand_id="c1", candidate=cand)
    r.seed_metrics = [{"training_epoch_metrics": [0.1, 0.9]},
                      {"training_epoch_metrics": [0.2, 0.3]}]
    vals, _ = _score("max_over_training_epochs", r)
    assert vals == [pytest.approx(0.9), pytest.approx(0.3)], "one value per seed"


# ------------------------------------------------- the per-episode scalar

class _FakeEnv:
    """Only what `_episode_native_scalar` reads."""

    def __init__(self, metric=0.0, ref=None):
        self._m = metric
        self._ref = ref
        self.success_threshold = 0.5

    def task_metric(self, traj):
        return self._m

    def success(self, traj):
        return bool(self.task_metric(traj) >= self.success_threshold)

    def reference_reward(self, s, a=None):
        if self._ref is None:
            raise NotImplementedError
        return self._ref


def test_native_success_uses_the_vendor_pinned_check_not_a_custom_score():
    """On a `discrete` task the spec pins a verified reimplementation of the
    VENDOR's own success check, so `env.success` IS the native signal there."""
    hit = training._episode_native_scalar(
        _FakeEnv(metric=0.8), native_signal.NATIVE_SUCCESS, [1, 2], [1])
    miss = training._episode_native_scalar(
        _FakeEnv(metric=0.2), native_signal.NATIVE_SUCCESS, [1, 2], [1])
    assert (hit, miss) == (1.0, 0.0)


def test_native_reward_sums_the_reference_reward_over_the_episode():
    v = training._episode_native_scalar(
        _FakeEnv(ref=2.5), native_signal.NATIVE_REWARD, [0, 1, 2, 3], [0, 0, 0])
    assert v == pytest.approx(7.5)


def test_a_task_with_no_reference_reward_yields_None_not_zero():
    """UNKNOWN is not zero: a zero would be a data point the run never took."""
    assert training._episode_native_scalar(
        _FakeEnv(ref=None), native_signal.NATIVE_REWARD, [0, 1], [0]) is None
    assert training._episode_native_scalar(
        _FakeEnv(), native_signal.REFUSE, [0, 1], [0]) is None


# ------------------------------------------------- the source must be native

def test_the_rule_is_refused_beside_a_non_native_fitness_source():
    """The reducer replaces the SERIES, not just the reduction, and the series
    is the native channel. Pinned beside `ground_truth_metric` the report would
    read `fitness_source: ground_truth_metric` over a number that is the max of
    a reference-reward RETURN on one task and a success RATE on the next --
    nothing downstream can detect that, so it is refused at load."""
    from bird.config import ConfigError, load
    for src in ("ground_truth_metric", "success_rate", "native_success", "native_reward"):
        # `native_success` / `native_reward` are refused too: the series' channel is
        # the one `native` resolves per task, and an explicit sub-channel could
        # disagree with it on a task that ships both a success and a reward.
        with pytest.raises(ConfigError, match="must be `native` exactly"):
            load("eureka", overrides={"evaluate.fitness.source": src})
    assert load("eureka", overrides={"evaluate.fitness.source": "native"})


def test_rstar_does_not_inherit_eurekas_aggregation():
    """`rstar` extends `eureka.yaml`, so eureka's `max_over_training_epochs`
    pin would reach it by inheritance -- an undisclosed fidelity change to a
    PUBLISHED method, and one that would also score R*'s `success_rate` fitness
    off the native series. `rstar` therefore pins `max_over_checkpoints`
    explicitly."""
    from bird.config import load
    r = load("rstar")
    assert r.get("evaluate.fitness.checkpoint_aggregation") == "max_over_checkpoints"
    assert r.get("evaluate.fitness.source") == "success_rate"
    # WHAT THIS TEST IS FOR is the two asserts above: rstar's explicit
    # `max_over_checkpoints` / `success_rate` pins still resolve to the
    # checkpoint-rule values. The hash below is a RECOMPUTED pin, and it moves with
    # any change to `configs/_default.yaml` -- a new key, a removed key, or an
    # edited default VALUE -- because `Config.hash()` covers all of `_data`,
    # including keys a config never mentions. Before updating the literal,
    # check the two asserts first: if THEY moved, the pin has stopped being a
    # no-op and R* runs are invalidated, which is a different and much worse
    # finding than a default set that changed.
    #
    # MEASURED WITH `load("rstar")` AND NO PROFILE, as this test calls it: a
    # hash is a property of (tree, profile), so the same config under
    # `--profile dev` is a different, real hash.
    assert r.hash() == "425ef6f1332e", (
        "rstar's resolved config hash moved. If the two asserts above still "
        "pass, a key or default in configs/_default.yaml changed and the literal "
        "needs re-measuring; if they do not, rstar's explicit pin has stopped "
        "being a no-op and R* runs are invalidated")


def test_every_config_carrying_the_rule_reads_a_native_source():
    """The blast radius, asserted rather than assumed: the epoch rule reaches
    every config that extends `eureka.yaml`, directly or through a chain, and
    every one of them must satisfy the coherence rule."""
    import glob, os
    from bird.config import load
    carrying = []
    for path in sorted(glob.glob("configs/*.yaml") + glob.glob("configs/methods/*.yaml")
                       + glob.glob("configs/hillclimb/*.yaml")
                       + glob.glob("configs/examples/*.yaml")):
        base = os.path.basename(path)[:-5]
        if base == "_default":
            continue
        sub = os.path.basename(os.path.dirname(path))
        name = f"{sub}/{base}" if sub in ("hillclimb", "examples") else base
        try:
            cfg = load(name)
        except Exception:
            continue
        if cfg.get("evaluate.fitness.checkpoint_aggregation") == "max_over_training_epochs":
            carrying.append(name)
            assert cfg.get("evaluate.fitness.source") == "native", name
    assert "eureka" in carrying and "rstar" not in carrying


# ---------------------------------------------------------------------------
# The load-bearing half: the SB3 callback that RECORDS the series. Without these
# an inert hook would leave every eureka run on the checkpoint rule with a green suite
# (the reducer's fallback is designed to be visible, not to fail).
# ---------------------------------------------------------------------------

_SB3_REWARD = "def reward(s, a, s2):\n    return -float(abs(s2[0])), {}\n"


def _sb3_row(aggregation: str):
    """One real PPO training of a valid reward on toy_reacher, 3000 steps, one
    seed, under the given `checkpoint_aggregation`; returns the seed row."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    torch = pytest.importorskip("torch")
    import types as pytypes
    from bird import registry
    from bird.budget import Budget
    from bird.config import load
    from bird.context import Context
    from bird.types import Candidate
    torch.set_num_threads(1)
    registry.load_all()
    cfg = load("eureka", profile="tester", overrides={
        "seed": 0, "train.backend": "sb3", "train.algorithm": "ppo",
        "train.env_steps": 3000, "train.seeds_per_candidate": 1,
        "evaluate.rollouts_per_candidate": 1,
        "evaluate.fitness.checkpoint_aggregation": aggregation})
    ctx = Context(cfg=cfg, budget=Budget(), env=registry.get("env", "toy_reacher")({}))
    res = registry.get("train_backend", "sb3")(
        ctx, pytypes.SimpleNamespace(restart=0, iteration=0),
        Candidate(cand_id="c0000", iteration=0, reward_code=_SB3_REWARD), 1)
    assert res.trained and not res.error, res.error
    (row,) = res.seed_metrics
    return ctx, res, row


def test_sb3_records_the_series_and_the_reducer_scores_it_not_the_fallback():
    """Under `max_over_training_epochs` a real SB3 training leaves a non-empty
    per-update native series on the seed row (toy_reacher ships a native success,
    so every entry is a success fraction in [0, 1]), and the `native` fitness
    source scores THAT series -- `fitness_from: training_epoch_series`, no
    `training_epoch_series_missing` marker."""
    from bird import registry
    ctx, res, row = _sb3_row("max_over_training_epochs")
    series = row.get("training_epoch_metrics")
    assert isinstance(series, list) and len(series) >= 1, row.keys()
    assert all(0.0 <= v <= 1.0 for v in series), series
    (rep,) = registry.get("fitness_source", "native")(ctx, None, [res])
    assert rep.meta.get("fitness_from") == "training_epoch_series", rep.meta
    assert "training_epoch_series_missing" not in rep.meta, rep.meta
    assert rep.fitness == max(series)


def test_sb3_records_no_series_under_any_other_reducer():
    """The recording is gated on the reducer: under `max_over_checkpoints` the
    seed row carries no `training_epoch_metrics` key at all, so a config that
    does not select the epoch rule has a byte-identical artifact and pays no per-step
    native scoring inside training."""
    from bird import registry
    ctx, res, row = _sb3_row("max_over_checkpoints")
    assert "training_epoch_metrics" not in row, sorted(row.keys())
    # And the gate agrees with the reducer: a config that does NOT select the
    # rule is scored off the checkpoint curve with no fallback marker -- the
    # marker means "selected the rule, found no series", never "did not select".
    (rep,) = registry.get("fitness_source", "native")(ctx, None, [res])
    assert "training_epoch_series_missing" not in rep.meta, rep.meta
    assert rep.meta.get("fitness_from") != "training_epoch_series"
