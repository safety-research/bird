"""`train.checkpoint_selection` -- ship the best checkpoint by the designed reward.

Both backends train in chunks and evaluate the policy after each one. Shipping
the LAST weights whatever the curve says makes the policy rolled out for the
judge, recorded to video, scored by §4 and stored under `policy_ref` always
the final one. On HalfCheetah most candidates peak mid-training and finish
5-10 m lower, so Eureka's
`evaluate.fitness.checkpoint_aggregation: max_over_checkpoints` would report one
policy's number and ship another.

`best_by_reward` restores the checkpoint whose greedy evaluation paid the
candidate's OWN reward the most -- inside `run_seed`, into the live model, so
the rollouts, the video, the stored blob and `warm_start_from_best` all get the
same weights with no further plumbing. Three properties, and each has a test
here that fails when its mechanism is removed (verified by mutation):

  * the criterion is `reward_return`, never the task metric -- selecting on
    `fitness` would leak ground truth into training;
  * the restore is real: the stored policy IS the snapshot taken at the best
    checkpoint, and the trajectories §4 sees came from it;
  * it is recorded as EXECUTED: a run whose last checkpoint was already the
    best and a run that restored are the same curve, and only the seed row
    tells them apart.

WHY THE EVALUATION IS SCRIPTED. `_evaluate_policy` is wrapped so that
`reward_return` follows a chosen sequence while every other number -- the
rollouts, the task metric, the step counts -- stays real. A real curve peaks
where the learner happens to peak, which is the one thing a test about
"restore the peak" must not leave to chance. The wrapper keeps the production
call in place, so a backend that stopped consulting the curve is caught by the
assertions, not by the script.

PARAMETRISED OVER THE BACKENDS, as `tests/test_pruning.py` is and for its
reason: the two trainers are separate implementations of the same stage that
grew separate answers to the same key once. The planner is excluded from the
`best_by_reward` cases -- it learns no parameters, and `_check_coherence`
refuses the pair at load (tested below).
"""

from __future__ import annotations

import copy
import io
import json
import types as pytypes

import numpy as np
import pytest

from test_parallelism import scrub_volatile   # the ONE volatile list, recursively
from conftest import backend_param, backend_param_unimplemented

from bird import registry
from bird.budget import Budget
from bird.components import fasttd3, simba_v2
from bird.components import training
from bird.config import ConfigError, load
from bird.context import Context
from bird.types import Candidate, TrainResult

#: Not in the tester-tier smoke suite (heavyweight execution: a real learner run per backend).
#: Deselected by `-m "not slow"`; run it with `-m slow`. See pyproject.toml.
pytestmark = pytest.mark.slow

#: Toy 2-D reacher: 6-D observation, 2-D continuous action, horizon 25.
_REWARD = """
def compute_reward(s, a, s2):
    import numpy as np
    o = np.asarray(s2, dtype=float)
    return float(-np.linalg.norm(o[0:2] - o[2:4]))
"""

#: `_sb3_run`'s chunk arithmetic lands on 5 checkpoints here and the surrogate
#: backends on 6; both are enough rows for a peak at the third to be an
#: unambiguous mid-curve restore (see `_PEAKED`).
_STEPS = 3000

#: A designed-reward curve that peaks at the THIRD checkpoint and then decays.
#: Clamped at its last value for any backend with more checkpoints than rows.
#: Span 4.0, so the peak beats the last row by far more than 2% of the range.
_PEAKED = [1.0, 2.0, 5.0, 3.0, 2.5, 2.0, 1.5, 1.2, 1.1, 1.0]
_PEAK_INDEX = 2


def _backends(*, exclude=()):
    registry.load_all()
    # Every registered backend is a LEARNER: this file parametrises over them all.
    return sorted(n for (k, n) in registry._REGISTRY
                  if k == "train_backend" and n not in exclude)


def _skip_if_unavailable(name: str) -> None:
    if name == "sb3":
        torch = pytest.importorskip("torch")
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("gymnasium")
        torch.set_num_threads(1)
    if name == "fasttd3":
        torch = pytest.importorskip("torch")
        torch.set_num_threads(1)
    # `simba_v2` is a torch learner backend and this arm exists because
    # REGISTERING A BACKEND CREATES CASES IN FILES THE CHANGE NEVER TOUCHES.
    # `_backends()` reads the registry, so registering `train_backend: simba_v2`
    # gives every parametrised body here a `[simba_v2]` case. Without the arm it falls through
    # to the backend, which raises "train.backend: simba_v2 needs torch, which
    # is not installed", and because this file is slow-marked the default
    # `-m "not slow"` selection deselects it and cannot see the failure. `torch` and nothing else: the
    # port is torch, not a wrapper round upstream's JAX, so there is no
    # `jax`/`flax` to name and this is not the never-opening gate
    # `tests/test_no_silent_skip.py` refuses.
    if name == "simba_v2":
        torch = pytest.importorskip("torch")
        torch.set_num_threads(1)

#: `train.backend: fasttd3` runs its upstream defaults otherwise -- 128 envs,
#: batch 32768, 1024-wide distributional critics, a GPU-scale set -- and this
#: file tests the CONTRACT, not learning. Injected only for that backend; the
#: others keep the config as written.
_FASTTD3_TINY = {"num_envs": 2, "batch_size": 32, "buffer_size": 64,
                 "critic_hidden_dim": 8, "actor_hidden_dim": 8, "num_atoms": 5,
                 "v_min": -10.0, "v_max": 10.0, "learning_starts": 1,
                 "compile": False}

#: `train.backend: simba_v2` runs UPSTREAM SimbaV2's defaults otherwise --
#: 512-wide hyperspherical blocks, a 101-bin distributional critic, batch 256 --
#: and this file tests the CONTRACT, not learning, for the same reason the
#: fasttd3 set above exists: the assertions are about checkpoint rows, pruning
#: decisions and budget accounting, none of which reads a layer width.
#:
#: SIZE KEYS ONLY, AND `learning_starts` IS DELIBERATELY NOT HERE.
#: `tests/test_simba_v2.py::TINY` also pins `learning_starts: 8` and
#: `buffer_size: 256`, and copying it verbatim makes these cases SLOWER than no
#: injection at all -- measured on one case
#: (`test_a_tie_at_the_peak_ships_the_later_checkpoint[simba_v2]`, cold, same
#: machine, same load): 31.5 s with upstream defaults, 109.0 s with TINY verbatim,
#: 14.7 s with the size keys alone. `learning_starts` is not a size knob, it is
#: a WORK knob: upstream's default exceeds these files' whole step budget, so
#: lowering it turns a near-zero-update run into thousands of updates and buys
#: nothing an assertion here reads. Copying a constant because its name says
#: "tiny" is the trap; the keys that matter are the ones that shrink a tensor.
#: Over all 21 `[simba_v2]` cases in the three files: 455 s with no injection,
#: 195 s with this one.
_SIMBA_V2_TINY = {"batch_size": 32, "critic_hidden_dim": 16,
                  "actor_hidden_dim": 16, "critic_num_bins": 11,
                  "compile": False, "actor_num_blocks": 1,
                  "critic_num_blocks": 1}


#: sb3 + PPO only: a rollout that divides the chunk, so `_STEPS` holds the five
#: checkpoints the scripted curves below are written for. `_sb3_run` charges
#: measured steps and stops at the budget, and SB3's on-policy `learn()` rounds
#: every request up to a whole `n_steps` (2048 by default) -- unpinned, `_STEPS`
#: would be two rollouts and two checkpoints. See tests/test_pruning.py.
_SB3_PPO_TINY = {"n_steps": 600, "batch_size": 100}


def _ctx(**overrides):
    registry.load_all()
    overrides.setdefault("seed", 0)
    overrides.setdefault("train.env_steps", _STEPS)
    overrides.setdefault("train.seeds_per_candidate", 1)
    overrides.setdefault("evaluate.rollouts_per_candidate", 1)
    if overrides.get("train.backend") == "fasttd3":
        overrides.setdefault("train.hyperparameters", dict(_FASTTD3_TINY))
    if overrides.get("train.backend") == "simba_v2":
        overrides.setdefault("train.hyperparameters", dict(_SIMBA_V2_TINY))
    if overrides.get("train.backend") == "sb3" and \
            overrides.get("train.algorithm", "ppo") == "ppo":
        overrides.setdefault("train.hyperparameters", dict(_SB3_PPO_TINY))
    cfg = load("eureka", overrides=overrides, profile="tester")
    return Context(cfg=cfg, budget=Budget(),
                   env=registry.get("env", "toy_reacher")({}))


def _state():
    return pytypes.SimpleNamespace(restart=0, iteration=0)


def _cand(cid="c0000"):
    return Candidate(cand_id=cid, iteration=0, reward_code=_REWARD)


class _Script:
    """Make `reward_return` (and optionally `fitness`) follow a sequence, per
    evaluation call, leaving the real rollout in place.

    `period` restarts the sequence every N calls so a multi-seed run hands every
    seed the same curve -- and hands it whether the seeds ran in one process (a
    shared counter) or in forked workers (each child inherits the counter at
    zero), which is what lets the parallel-vs-sequential comparison below be
    about the restore rather than about the script."""

    def __init__(self, returns, fits=None, period=None):
        self.returns, self.fits, self.period = list(returns), fits, period
        self.calls = 0

    def install(self, monkeypatch):
        real = training._evaluate_policy

        def fake(env, policy, rng, reward, n_episodes, on_error):
            metrics, comps, steps, trajs = real(env, policy, rng, reward, n_episodes, on_error)
            i = self.calls % self.period if self.period else self.calls
            metrics = dict(metrics)
            # Clamp the two scripts SEPARATELY. Clamping the shared counter to the
            # reward script's length before indexing `fits` hands every row past
            # the tenth the same fitness -- on tabular's 12-checkpoint curve the
            # last row would read fits[9] where the test expects fits[11] -- and
            # fails the test while the production code under test is right.
            metrics["reward_return"] = float(self.returns[min(i, len(self.returns) - 1)])
            if self.fits is not None:
                metrics["fitness"] = float(self.fits[min(i, len(self.fits) - 1)])
            self.calls += 1
            return metrics, comps, steps, trajs

        monkeypatch.setattr(training, "_evaluate_policy", fake)
        return self


def _train(ctx, **kw):
    backend = registry.get("train_backend", ctx.cfg["train.backend"])
    return backend(ctx, _state(), _cand(), int(ctx.cfg.get("train.seeds_per_candidate", 1)),
                   **kw)


def _round_of(m, index):
    return int(m["checkpoints"][index]["round"])


@pytest.fixture(autouse=True)
def _clean_stores():
    training._POLICY_STORE.clear()
    training._REPLAY_STORE.clear()
    yield
    training._POLICY_STORE.clear()
    training._REPLAY_STORE.clear()


# --------------------------------------------------------------------------
# The key reaches every backend, and `final` is the unchanged default
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_every_backend_consults_the_selector_once_per_seed(name, monkeypatch):
    """The `_pruner_for` argument: a backend that stops reading the key turns it
    into a declared-but-unread value on that backend alone (§8.3)."""
    _skip_if_unavailable(name)
    seen = []
    real = training._select_checkpoint

    def spy(curve, rule, min_delta):
        seen.append((len(curve), rule, float(min_delta)))
        return real(curve, rule, min_delta)

    monkeypatch.setattr(training, "_select_checkpoint", spy)
    ctx = _ctx(**{"train.backend": name, "train.seeds_per_candidate": 2})
    res = _train(ctx)
    assert len(seen) == 2, f"{name}: {len(seen)} selector calls for 2 seeds"
    assert all(rule == "final" and md == 0.02 and n > 0 for n, rule, md in seen), seen
    for m in res.seed_metrics:
        assert m["checkpoint_selection"] == "final"
        assert m["checkpoint_selection_reason"] == "final"
        assert m["checkpoint_selection_reason"] in training._SELECTION_REASONS
        assert m["restored_checkpoint"] is None
        assert m["shipped_checkpoint"] == _round_of(m, -1)
        assert m["fitness"] == m["final"] == m["final_checkpoint_fitness"] \
            == m["checkpoints"][-1]["fitness"]


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends(exclude=("none",))])
def test_final_never_snapshots_or_restores_even_on_a_peaked_curve(name, monkeypatch):
    """The default must cost nothing and change nothing: no parameter copy is
    taken and no restore is attempted, whatever the curve does."""
    _skip_if_unavailable(name)
    _Script(_PEAKED).install(monkeypatch)
    for fn in ("_snapshot_learner", "_snapshot_sb3", "_restore_sb3"):
        monkeypatch.setattr(training, fn,
                            lambda *a, _fn=fn, **k: pytest.fail(f"{_fn} called under final"))
    # BOTH torch backends: simba_v2 owns its own `_snapshot_agent` /
    # `_restore_agent` (it snapshots the temperature too, which fasttd3 has no
    # equivalent of), so patching only fasttd3's would leave the `[simba_v2]`
    # case asserting nothing about the backend under test -- a VACUOUS pass
    # rather than a failure, which is why this loop takes the module list
    # rather than one module.
    for mod in (fasttd3, simba_v2):
        for fn in ("_snapshot_agent", "_restore_agent"):
            monkeypatch.setattr(mod, fn,
                                lambda *a, _fn=fn, **k: pytest.fail(f"{_fn} called under final"))
    for cls in (training._Learner, training._QLearner, training._CEMLearner):
        monkeypatch.setattr(cls, "restore",
                            lambda self, snap: pytest.fail("restore called under final"))
    ctx = _ctx(**{"train.backend": name, "train.checkpoint_selection": "final"})
    res = _train(ctx)
    (m,) = res.seed_metrics
    assert m["restored_checkpoint"] is None and m["shipped_checkpoint"] == _round_of(m, -1)
    assert m["fitness"] == m["checkpoints"][-1]["fitness"]


# --------------------------------------------------------------------------
# best_by_reward restores the peak, and what leaves the function IS the peak
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends(exclude=("none",))])
def test_best_by_reward_restores_the_peak_and_records_it_as_executed(name, monkeypatch):
    _skip_if_unavailable(name)
    script = _Script(_PEAKED).install(monkeypatch)
    ctx = _ctx(**{"train.backend": name, "train.checkpoint_selection": "best_by_reward"})
    res = _train(ctx)
    assert res.trained and not res.error, res.error
    (m,) = res.seed_metrics
    n = m["n_checkpoints"]
    assert script.calls == n >= 4
    # the script clamps at its last value, so a backend with more checkpoints
    # than rows (tabular: 12 on toy_reacher) sees a flat tail, not a repeat
    assert [p["reward_return"] for p in m["checkpoints"]] == \
        [_PEAKED[min(i, len(_PEAKED) - 1)] for i in range(n)]
    assert m["checkpoint_selection"] == "best_by_reward"
    assert m["checkpoint_selection_reason"] == "restored"
    assert m["restored_checkpoint"] == m["shipped_checkpoint"] == _round_of(m, _PEAK_INDEX)
    # `fitness`/`final` describe the SHIPPED row; the full curve stays recorded.
    assert m["fitness"] == m["final"] == m["checkpoints"][_PEAK_INDEX]["fitness"]
    assert m["final_checkpoint_fitness"] == m["checkpoints"][-1]["fitness"]
    assert m["max"] == max(p["fitness"] for p in m["checkpoints"])
    # every per-row statistic the seed row carries describes the SHIPPED row
    # (the surrogate rows carry these two; sb3's do not)
    for key in ("success_rate", "gt_return"):
        if key in m:
            assert m[key] == m["checkpoints"][_PEAK_INDEX][key], key


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends(exclude=("none",))])
def test_the_selection_reads_the_designed_reward_not_the_task_metric(name, monkeypatch):
    """`fitness` peaks at the LAST row and `reward_return` at the third. Selecting
    on the task metric would ship the last checkpoint (and leak ground truth
    into training); the designed reward ships the third."""
    _skip_if_unavailable(name)
    fits = [0.02 * (i + 1) for i in range(40)]          # monotone, last is best;
                                                        # longer than any backend's curve
    _Script(_PEAKED, fits=fits).install(monkeypatch)
    ctx = _ctx(**{"train.backend": name, "train.checkpoint_selection": "best_by_reward"})
    (m,) = _train(ctx).seed_metrics
    assert m["restored_checkpoint"] == _round_of(m, _PEAK_INDEX)
    assert m["fitness"] == pytest.approx(fits[_PEAK_INDEX])
    assert m["final_checkpoint_fitness"] == pytest.approx(fits[m["n_checkpoints"] - 1])
    assert m["final_checkpoint_fitness"] > m["fitness"], "the two rows must differ here"


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends(exclude=("none",))])
def test_within_the_margin_the_last_checkpoint_ships_and_says_so(name, monkeypatch):
    _skip_if_unavailable(name)
    _Script([1.0, 2.0, 2.01, 2.0, 2.0, 2.0, 2.0, 2.0]).install(monkeypatch)
    ctx = _ctx(**{"train.backend": name, "train.checkpoint_selection": "best_by_reward"})
    (m,) = _train(ctx).seed_metrics
    assert m["checkpoint_selection"] == "best_by_reward"
    assert m["checkpoint_selection_reason"] == "within_margin"
    assert m["restored_checkpoint"] is None
    assert m["shipped_checkpoint"] == _round_of(m, -1)
    assert m["fitness"] == m["final_checkpoint_fitness"] == m["checkpoints"][-1]["fitness"]

    _Script([1.0, 2.0, 2.01, 2.0, 2.0, 2.0, 2.0, 2.0]).install(monkeypatch)
    ctx = _ctx(**{"train.backend": name, "train.checkpoint_selection": "best_by_reward",
                  "train.checkpoint_selection_cfg.min_delta": 0.0})
    (m,) = _train(ctx).seed_metrics
    assert m["checkpoint_selection_reason"] == "restored"
    assert m["restored_checkpoint"] == _round_of(m, 2)


@pytest.mark.parametrize("name,rebuild", [("mock", "_linear_policy"), ("tabular", "_greedy_policy")])
def test_the_surrogate_ships_exactly_the_snapshot_taken_at_the_peak(name, rebuild, monkeypatch):
    """The stored policy IS the parameters copied at the best row, and the
    trajectory §4 sees was rolled out by those parameters -- compared by
    re-rolling the stored policy on the same rng.

    Over BOTH surrogate learners, and the second is the one with teeth: the CEM
    searcher reassigns its mean every round, so a snapshot that merely aliased
    the live array would happen to survive, while the Q-learner updates its
    table in place, so it would not. Returning the live array from
    `_snapshot_learner` passes this test on `mock` alone (mutation run) and
    fails it on `tabular`."""
    taken = []
    real = training._snapshot_learner

    def spy(learner):
        snap = real(learner)
        taken.append(None if snap is None else np.array(snap, copy=True))
        return snap

    monkeypatch.setattr(training, "_snapshot_learner", spy)
    _Script(_PEAKED).install(monkeypatch)
    ctx = _ctx(**{"train.backend": name, "train.checkpoint_selection": "best_by_reward"})
    res = _train(ctx)
    (m,) = res.seed_metrics
    assert m["checkpoint_selection_reason"] == "restored"
    # one snapshot per running-best row: 1.0, 2.0, 5.0 -- and never for a
    # later row that did not improve
    assert len(taken) == _PEAK_INDEX + 1
    stored = training._POLICY_STORE[res.policy_ref]
    np.testing.assert_array_equal(stored, taken[-1])
    assert not np.array_equal(taken[-1], taken[-2]), "training did not move the parameters"

    reward = training.compile_reward(_REWARD, _cand())
    base = training._seed_base(ctx, _state(), _cand())
    policy = getattr(training, rebuild)(ctx.env, stored)
    traj, _, _ = training._rollout(ctx.env, policy, np.random.default_rng(base + 104729),
                                   reward, training._error_router("exception_soft", io.StringIO()))
    np.testing.assert_array_equal(np.asarray(traj.actions), np.asarray(res.trajectories[0].actions))


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_sb3_ships_exactly_the_snapshot_taken_at_the_peak(algorithm, monkeypatch):
    """The sb3 half: the `_PolicyBlob` in the store, the live model's parameters
    after the restore, and the snapshot at the peak row are one and the same
    policy state dict -- and the recorded trajectory is that model's. Over PPO
    (two state dicts) and SAC (four, plus `log_ent_coef` outside them)."""
    _skip_if_unavailable("sb3")
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file

    taken, models = [], []
    real_snap, real_restore = training._snapshot_sb3, training._restore_sb3

    def snap_spy(model):
        snap = real_snap(model)
        taken.append(copy.deepcopy(snap))
        return snap

    def restore_spy(model, params):
        models.append(model)
        return real_restore(model, params)

    monkeypatch.setattr(training, "_snapshot_sb3", snap_spy)
    monkeypatch.setattr(training, "_restore_sb3", restore_spy)
    _Script(_PEAKED).install(monkeypatch)
    ctx = _ctx(**{"train.backend": "sb3", "train.checkpoint_selection": "best_by_reward",
                  "train.algorithm": algorithm})
    res = _train(ctx)
    (m,) = res.seed_metrics
    # `algorithm_cited`, the paper's algorithm: this test
    # parametrises `train.algorithm` over ppo/sac, so it reads back the
    # citation it set, not the learner -- which is `sb3` on both.
    assert (m["algorithm_cited"] == algorithm
            and m["checkpoint_selection_reason"] == "restored")
    # one snapshot per running-best row (1.0, 2.0, 5.0); `_restore_sb3` itself
    # takes one more copy of the final parameters before it loads the peak
    assert len(taken) == _PEAK_INDEX + 2 and len(models) == 1
    taken.pop()  # the pre-restore copy of the FINAL parameters

    def same(a, b):
        return set(a) == set(b) and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)

    peak = taken[-1]["policy"]
    assert not same(peak, taken[-2]["policy"]), "training did not move the parameters"
    (model,) = models
    assert same(model.get_parameters()["policy"], peak), "the live model was not restored"
    blob = training._POLICY_STORE[res.policy_ref]
    _, params, _ = load_from_zip_file(io.BytesIO(bytes(blob)), device="cpu")
    assert same(params["policy"], peak), "policy_ref does not hold the restored checkpoint"

    reward = training.compile_reward(_REWARD, _cand())
    base = training._seed_base(ctx, _state(), _cand())

    def policy(s):
        act, _ = model.predict(np.asarray(s, dtype=np.float32), deterministic=True)
        return np.asarray(act, dtype=float).ravel()

    traj, _, _ = training._rollout(ctx.env, policy, np.random.default_rng(base + 104729),
                                   reward, training._error_router("exception_soft", io.StringIO()))
    np.testing.assert_array_equal(np.asarray(traj.actions), np.asarray(res.trajectories[0].actions))


def test_selection_costs_no_environment_steps(monkeypatch):
    """Restoring is a parameter copy, never a rollout: the step count and the
    budget are identical between the two rules on the same scripted curve."""
    used = {}
    for rule in ("final", "best_by_reward"):
        _Script(_PEAKED).install(monkeypatch)
        ctx = _ctx(**{"train.backend": "mock", "train.checkpoint_selection": rule})
        res = _train(ctx)
        used[rule] = (res.env_steps_used, ctx.budget.env_steps, res.seed_metrics[0]["env_steps"])
    assert used["final"] == used["best_by_reward"], used


# --------------------------------------------------------------------------
# Schedules: the seed fork must ship the restored parameters too
# --------------------------------------------------------------------------


def test_the_seed_fork_ships_the_same_restored_policy_as_sequential(monkeypatch):
    """`run_seed` restores inside the (possibly forked) worker, so the payload a
    child sends home is the restored snapshot and the parent's rollouts run on
    it. Whole result compared, not a scalar: seed rows, curves, trajectories
    and the stored parameters."""
    if not hasattr(__import__("os"), "fork"):
        pytest.skip("no fork on this platform")
    probe = _Script(_PEAKED).install(monkeypatch)
    period = _train(_ctx(**{"train.backend": "mock"})).seed_metrics[0]["n_checkpoints"]
    assert probe.calls == period

    def run(schedule):
        training._POLICY_STORE.clear()
        _Script(_PEAKED, period=period).install(monkeypatch)
        ctx = _ctx(**{"train.backend": "mock", "train.checkpoint_selection": "best_by_reward",
                      "train.seeds_per_candidate": 2,
                      "train.candidate_parallelism": schedule,
                      "loop.max_parallel_trainings": 2})
        # `(workers, downgrade_reason)`: the reason is what reaches a reader
        # of a run's artefact when the batched tier forces seeds sequential,
        # so it is not a bare int. This assertion is about the WORKER COUNT
        # only.
        assert training._seed_fork_workers(ctx.cfg, 2)[0] == (
            2 if schedule == "parallel" else 1)
        res = _train(ctx)
        # THE SHARED LIST, RECURSIVELY. A hand-rolled `if k != "wallclock_s"`
        # cannot reach `eval_wall_s`, which lives on each point of the curve
        # NESTED inside the row.
        rows = [scrub_volatile(m) for m in res.seed_metrics]
        # AND THE CURVE: compared unstripped beside the stripped rows, it is a
        # second path to the same failure at one site.
        return {"rows": rows, "curve": scrub_volatile(res.checkpoints),
                "actions": [np.asarray(t.actions).tolist() for t in res.trajectories],
                "stored": np.asarray(training._POLICY_STORE[res.policy_ref]).tolist()}

    seq, par = run("sequential"), run("parallel")
    assert all(m["checkpoint_selection_reason"] == "restored" for m in seq["rows"]), \
        [m["checkpoint_selection_reason"] for m in seq["rows"]]
    assert seq == par


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends(exclude=("none",))])
def test_a_tie_at_the_peak_ships_the_later_checkpoint(name, monkeypatch):
    """Same number, more training: of two checkpoints tied on the designed reward
    the LATER one ships. The in-loop snapshot and the post-loop decision must
    agree on that -- a loop that snapshotted the FIRST of the tie would leave the
    rule pointing at a checkpoint it has no parameters for (`no_snapshot`)."""
    _skip_if_unavailable(name)
    _Script([1.0, 5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0]).install(monkeypatch)
    ctx = _ctx(**{"train.backend": name, "train.checkpoint_selection": "best_by_reward"})
    (m,) = _train(ctx).seed_metrics
    assert m["checkpoint_selection_reason"] == "restored"
    assert m["restored_checkpoint"] == _round_of(m, 2)


def test_the_sb3_seed_fork_ships_the_restored_policy(monkeypatch):
    """The exact seam: `seed_child` serialises the model AFTER `run_seed`
    restored it, so the blob that comes home -- and the policy the parent
    rebuilds and rolls out -- is the restored checkpoint, identically to the
    sequential schedule. Whole result compared."""
    _skip_if_unavailable("sb3")
    from stable_baselines3.common.save_util import load_from_zip_file
    import torch

    probe = _Script(_PEAKED).install(monkeypatch)
    period = _train(_ctx(**{"train.backend": "sb3"})).seed_metrics[0]["n_checkpoints"]
    assert probe.calls == period

    def run(schedule):
        training._POLICY_STORE.clear()
        _Script(_PEAKED, period=period).install(monkeypatch)
        ctx = _ctx(**{"train.backend": "sb3", "train.checkpoint_selection": "best_by_reward",
                      "train.seeds_per_candidate": 2,
                      "train.candidate_parallelism": schedule,
                      "loop.max_parallel_trainings": 2})
        # `(workers, downgrade_reason)`: the reason is what reaches a reader
        # of a run's artefact when the batched tier forces seeds sequential,
        # so it is not a bare int. This assertion is about the WORKER COUNT
        # only.
        assert training._seed_fork_workers(ctx.cfg, 2)[0] == (
            2 if schedule == "parallel" else 1)
        res = _train(ctx)
        assert res.trained and not res.error, res.error
        # The second stripping site in this file, through the same shared list:
        # a hand-rolled key filter here would be latent rather than red, not safe.
        rows = [scrub_volatile(m) for m in res.seed_metrics]
        _, params, _ = load_from_zip_file(
            io.BytesIO(bytes(training._POLICY_STORE[res.policy_ref])), device="cpu")
        return rows, [np.asarray(t.actions).tolist() for t in res.trajectories], params["policy"]

    seq_rows, seq_act, seq_pol = run("sequential")
    par_rows, par_act, par_pol = run("parallel")
    assert all(m["n_checkpoints"] == period and m["checkpoint_selection_reason"] == "restored"
               for m in seq_rows), [(m["n_checkpoints"], m["checkpoint_selection_reason"])
                                    for m in seq_rows]
    assert seq_rows == par_rows
    assert seq_act == par_act
    assert set(seq_pol) == set(par_pol) and all(
        torch.equal(seq_pol[k], par_pol[k]) for k in seq_pol)


def _stored_policy(name, ref):
    """What `_POLICY_STORE` holds under `ref`, in a form two runs can be compared on."""
    blob = training._POLICY_STORE[ref]
    if name != "sb3":
        return np.asarray(blob).tolist()
    from stable_baselines3.common.save_util import load_from_zip_file
    _, params, _ = load_from_zip_file(io.BytesIO(bytes(blob)), device="cpu")
    return {k: v.cpu().numpy().tolist() for k, v in params["policy"].items()}


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends(exclude=("none",))])
def test_a_restore_that_does_not_take_ships_what_final_would_have(name):
    """`restore_failed` and `no_snapshot`, driven through a real backend: the
    seed row says the last checkpoint shipped and why, `fitness` is the final
    row's, and the policy in the store is byte-for-byte the one a `final` run
    of the same seed stores -- so the row and the weights agree on the failure
    path too, which nothing else here checks."""
    _skip_if_unavailable(name)

    def run(rule, patches):
        with pytest.MonkeyPatch.context() as mp:
            training._POLICY_STORE.clear()
            _Script(_PEAKED).install(mp)
            for target, attr, value in patches:
                mp.setattr(target, attr, value)
            res = _train(_ctx(**{"train.backend": name, "train.checkpoint_selection": rule}))
            assert res.trained and not res.error, res.error
            (m,) = res.seed_metrics
            return m, _stored_policy(name, res.policy_ref)

    control, control_policy = run("final", [])
    cases = {
        # `simba_v2` needs its own two entries: with no patch for its module the
        # restore SUCCEEDS, the row reads `restored`, and the assertion on
        # `restore_failed` compares a working path against a failure contract.
        # The skip arm is necessary and not sufficient -- this table is a second
        # per-backend enumeration in the same file, and a registered backend has
        # to appear in every one of them.
        "restore_failed": [(training, "_restore_sb3", lambda model, params: False),
                           (training._QLearner, "restore", lambda self, snap: False),
                           (training._CEMLearner, "restore", lambda self, snap: False),
                           (fasttd3, "_restore_agent", lambda agent, snap: False),
                           (simba_v2, "_restore_agent", lambda agent, snap: False)],
        "no_snapshot": [(training, "_snapshot_sb3", lambda model: None),
                        (training, "_snapshot_learner", lambda learner: None),
                        (fasttd3, "_snapshot_agent", lambda agent: None),
                        (simba_v2, "_snapshot_agent", lambda agent: None)],
    }
    for reason, patches in cases.items():
        m, policy = run("best_by_reward", patches)
        assert m["checkpoint_selection"] == "best_by_reward"
        assert m["checkpoint_selection_reason"] == reason
        assert m["restored_checkpoint"] is None
        assert m["shipped_checkpoint"] == _round_of(m, -1) == control["shipped_checkpoint"]
        assert m["fitness"] == m["final_checkpoint_fitness"] == control["fitness"]
        assert policy == control_policy, f"{reason}: the stored policy is not the final one"


def test_a_restore_inside_the_candidate_fork_leaves_the_run_dir_bit_identical(tmp_path):
    """`train.candidate_parallelism: parallel` runs the restore inside a
    candidate worker and ships the TrainResult and the `_POLICY_STORE` delta
    home. Whole run directories compared under `test_parallelism`'s own
    normalisation, with at least one restore asserted to have happened -- the
    cross-section entry there walks the same path but cannot promise a peak;
    `min_delta: 0` here restores on any strictly better checkpoint."""
    from test_parallelism import _run as _run_search

    extra = {"train.checkpoint_selection": "best_by_reward",
             "train.checkpoint_selection_cfg.min_delta": 0.0}
    seq = _run_search("eureka", tmp_path / "seq", extra=extra)
    par = _run_search("eureka", tmp_path / "par", extra=extra,
                      **{"train.candidate_parallelism": "parallel",
                         "loop.max_parallel_trainings": 2})
    (run,) = [p for p in (tmp_path / "seq").iterdir() if p.is_dir()]
    restored = [m["restored_checkpoint"]
                for tr in run.glob("candidates/*/train_result.json")
                for m in json.loads(tr.read_text()).get("seed_metrics", [])
                if isinstance(m, dict) and m.get("restored_checkpoint") is not None]
    assert restored, "no candidate restored a checkpoint, so the comparison would be vacuous"
    assert seq == par


def test_a_failed_sb3_restore_puts_the_final_parameters_back():
    """`set_parameters` loads module by module and can fail part-way; a restore
    that fails must leave the model holding its final parameters, not half of
    each, or the seed row's `shipped the last` would be a lie."""
    _skip_if_unavailable("sb3")
    import torch
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO

    class _E(gym.Env):
        observation_space = spaces.Box(-np.ones(3, np.float32), np.ones(3, np.float32))
        action_space = spaces.Box(-np.ones(1, np.float32), np.ones(1, np.float32))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, a):
            return self.observation_space.sample(), 0.0, False, True, {}

    model = PPO("MlpPolicy", _E(), seed=1, device="cpu", n_steps=64, batch_size=32, verbose=0)
    early = training._snapshot_sb3(model)          # the freshly initialised weights
    model.learn(total_timesteps=128)
    final = training._snapshot_sb3(model)
    assert not all(torch.equal(early["policy"][k], final["policy"][k]) for k in final["policy"]), \
        "training did not move the parameters, so a rollback would be indistinguishable from none"
    # The EARLY weights with one policy tensor mis-shaped. `load_state_dict`
    # copies every matching tensor and raises for the mismatch at the end, so a
    # `_restore_sb3` without its rollback would leave the model holding the
    # early values of everything else -- which is what this detects (a `bad`
    # built from the final weights could not: whatever loaded would equal what
    # was already there).
    bad = copy.deepcopy(early)
    key = next(k for k, v in bad["policy"].items() if torch.is_tensor(v) and v.dim() == 2)
    bad["policy"][key] = torch.zeros(bad["policy"][key].shape[0] + 1, bad["policy"][key].shape[1])
    assert training._restore_sb3(model, bad) is False
    after = training._snapshot_sb3(model)["policy"]
    assert all(torch.equal(after[k], final["policy"][k]) for k in after), \
        "a failed restore left early parameters in the model"


def test_the_seed_row_separates_the_cited_algorithm_from_the_learner():
    """A row can carry `algorithm: 'ppo'` beside `learner: 'fasttd3'`.

    BOTH ARE TRUE, which is why the obvious fix is wrong. `train.algorithm`
    "records the algorithm the method's paper used and is kept as a citation
    even when no backend here can run it" (`bird/schema.py`, and
    `configs/_default.yaml` says it again) -- its enum carries `q_learning`
    and `none` precisely because nothing here executes those. So one name that
    reads as "what ran" would hold two different kinds of fact, and
    overwriting it with the resolved learner would delete a citation from a
    framework whose whole point is that published values stay citable.

    ON THE MOCK BACKEND ON PURPOSE. The other two writers (`fasttd3`, `sb3`)
    are behind `torch`, so their assertions skip in the `--extra test` venv
    and run only with the full extras; this one runs
    everywhere, and it is the same `run_seed` row shape.
    """
    res = _train(_ctx(**{"train.backend": "mock"}))
    m = res.seed_metrics[0]

    # Eureka's paper used PPO; the mock learner is what ran. Two facts, two
    # names, neither pretending to be the other.
    assert m["algorithm_cited"] == "ppo"
    assert m["learner"] != "ppo" and m["backend"] == "mock"
    assert "algorithm" not in m, (
        "the bare name is back, and with it the reading that a row's "
        "`algorithm` says what executed")

    # The citation FOLLOWS THE CONFIG rather than the backend: a method
    # citing SAC still records SAC while the mock learner runs.
    sac = _train(_ctx(**{"train.backend": "mock", "train.algorithm": "sac"}))
    assert sac.seed_metrics[0]["algorithm_cited"] == "sac"
    assert sac.seed_metrics[0]["learner"] == m["learner"]
