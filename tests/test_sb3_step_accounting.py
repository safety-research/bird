"""`_sb3_run` charges the steps the model actually took, and stops at the budget.

SB3's on-policy `learn()` completes whole `n_steps` rollouts -- `while
self.num_timesteps < total_timesteps: collect_rollouts(..., n_rollout_steps=
self.n_steps)` -- so `learn(total_timesteps=chunk)` executes
`ceil(chunk / n_steps) * n_steps` steps, 2048 per rollout by default, and no
shipped config pins `n_steps`. A chunk loop that did `learn(chunk); spent +=
chunk` over a fixed `n_chunks` would, under PPO, make every figure derived from
`spent` -- `seed_metrics[].train_steps` (commented "ACTUAL, not requested"),
`env_steps`, `TrainResult.env_steps_used`, `budget.env_steps`, the curve's
`step`, the pruner's `budget_frac` -- the request while the model trained the
rollouts: the dev profile's 20,000 would run 36,864 and record 19,998 (1.84x);
a 600-step test would run 10,240 and record 600 (17x). SAC/TD3
(`train_freq: 1`) are exact, so PPO methods would over-train their stated
budget and under-report their cost relative to SAC methods "at the same
budget".

`_sb3_learn_chunks` therefore measures `model.num_timesteps` across each `learn()`,
asks only for what is left, and stops once the MEASURED spend reaches the
budget (overshoot strictly under one rollout). The first tests drive it with a
fake model that has SB3's rollout contract and need no SB3; the last runs
real PPO and reads the live model.
"""
from __future__ import annotations

import types as pytypes

import pytest

from bird import registry
from bird.budget import Budget
from bird.components import training
from bird.components.training import _sb3_learn_chunks
from bird.config import load
from bird.context import Context
from bird.types import Candidate


class _RolloutModel:
    """SB3's on-policy contract, and nothing else: `learn(total_timesteps,
    reset_num_timesteps=False)` advances `num_timesteps` in whole rollouts of
    `n_steps` until it is at or past the request. `n_steps=1` is an exact
    (off-policy, `train_freq: 1`) learner."""

    def __init__(self, n_steps: int) -> None:
        self.n_steps = n_steps
        self.num_timesteps = 0
        self.requests: list = []

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = True) -> None:
        assert reset_num_timesteps is False, "a chunked learner must not reset its clock"
        self.requests.append(int(total_timesteps))
        target = self.num_timesteps + int(total_timesteps)
        while self.num_timesteps < target:
            self.num_timesteps += self.n_steps


def _plan(total: int, n: int) -> list:
    """`_sb3_run`'s partition of `total` into `n` chunks."""
    return [total // n + (1 if ck < total % n else 0) for ck in range(n)]


# --------------------------------------------------------------------------
# the accounting, on a fake model (no SB3 needed)
# --------------------------------------------------------------------------

def test_the_charge_is_what_the_model_executed_not_what_was_asked():
    """The dev profile's shape: 20,000 steps over 9 chunks, PPO's 2048 rollout.
    Each 2,222-step request runs two rollouts; the loop must charge 4,096 and
    stop after five chunks, not charge 2,222 nine times."""
    model = _RolloutModel(2048)
    spends = [s for _, s in _sb3_learn_chunks(model, _plan(20000, 9), 20000)]
    assert spends == [4096, 8192, 12288, 16384, 20480], spends
    assert spends[-1] == model.num_timesteps, "the charge and the model's clock disagree"
    assert 20000 <= spends[-1] < 20000 + 2048, "overshoot must stay under one rollout"
    # A fixed-chunk loop would have run all nine chunks -- 36,864 steps -- and
    # recorded 9 * 2222 = 19,998 of them.
    assert model.num_timesteps < 9 * 4096


def test_an_exact_learner_walks_the_whole_plan_and_lands_on_the_budget():
    """SAC/TD3 (`train_freq: 1`) execute exactly what is asked: nine chunks,
    nine checkpoints, and the budget met exactly -- 20,000, not the 19,998 that
    `9 * (20000 // 9)` truncates to."""
    model = _RolloutModel(1)
    out = list(_sb3_learn_chunks(model, _plan(20000, 9), 20000))
    assert [ck for ck, _ in out] == list(range(9))
    assert out[-1][1] == 20000 == model.num_timesteps
    assert model.requests == _plan(20000, 9)


def test_a_budget_inside_one_rollout_is_one_chunk_and_one_rollout():
    """The 600-step test shape. One request of 120 runs one 2,048-step rollout,
    which already exceeds the budget: one checkpoint, 2,048 charged -- not five
    rollouts recorded as 600."""
    model = _RolloutModel(2048)
    out = list(_sb3_learn_chunks(model, _plan(600, 5), 600))
    assert out == [(0, 2048)]
    assert model.requests == [120]


@pytest.mark.parametrize("total,n_chunks,n_steps", [
    (20000, 9, 2048), (1_000_000, 20, 2048), (3000, 5, 2048), (3000, 5, 600),
    (256, 5, 64), (20000, 9, 1), (7, 5, 1), (5000, 5, 1000),
])
def test_no_request_exceeds_what_is_left_and_the_overshoot_is_under_a_rollout(
        total, n_chunks, n_steps):
    model = _RolloutModel(n_steps)
    spent_before = 0
    for _ck, spent in _sb3_learn_chunks(model, _plan(total, n_chunks), total):
        assert model.requests[-1] <= total - spent_before, (
            f"asked for {model.requests[-1]} with {total - spent_before} left")
        assert spent == model.num_timesteps
        spent_before = spent
    assert total <= spent_before < total + n_steps, (total, spent_before, n_steps)
    assert len(model.requests) <= n_chunks


# --------------------------------------------------------------------------
# the real thing
# --------------------------------------------------------------------------

_REWARD = """
def compute_reward(s, a, s2):
    import numpy as np
    o = np.asarray(s2, dtype=float)
    return float(-np.linalg.norm(o[0:2] - o[2:4]))
"""


def test_ppo_seed_row_reports_the_models_own_step_count(monkeypatch):
    """Real PPO at its DEFAULT `n_steps` -- the shipped configs' shape,
    deliberately not pinned smaller -- on a 3,000-step budget. The seed row's
    `train_steps` must be the live model's `num_timesteps`, within one rollout
    of the budget, and the curve's last `step` must be that same number, not the
    requested 3000."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    registry.load_all()
    cfg = load("eureka", profile="tester", overrides={
        "seed": 0, "train.backend": "sb3", "train.algorithm": "ppo",
        "train.env_steps": 3000, "train.seeds_per_candidate": 1,
        "evaluate.rollouts_per_candidate": 1})
    assert "n_steps" not in (cfg.get("train.hyperparameters") or {}), "keep PPO's default rollout"
    ctx = Context(cfg=cfg, budget=Budget(), env=registry.get("env", "toy_reacher")({}))

    # `_sb3_input_dim(model)` is called once per seed, inside `run_seed`, with
    # the live model when its row is written -- the one module-level seam that
    # sees the trained model on the sequential schedule.
    models: list = []
    real_input_dim = training._sb3_input_dim

    def spy(model):
        models.append(model)
        return real_input_dim(model)

    monkeypatch.setattr(training, "_sb3_input_dim", spy)
    res = registry.get("train_backend", "sb3")(
        ctx, pytypes.SimpleNamespace(restart=0, iteration=0),
        Candidate(cand_id="c0000", iteration=0, reward_code=_REWARD), 1)
    assert res.trained and not res.error, res.error
    (model,) = models
    (row,) = res.seed_metrics
    assert row["train_steps"] == model.num_timesteps, (row["train_steps"], model.num_timesteps)
    assert 3000 <= row["train_steps"] < 3000 + model.n_steps, (row["train_steps"], model.n_steps)
    assert row["train_steps_requested"] == 3000
    assert row["checkpoints"][-1]["step"] == row["train_steps"]
    assert row["n_checkpoints"] == len(row["checkpoints"]) == 2, "3,000 steps hold two 2,048 rollouts"
    assert row["env_steps"] >= row["train_steps"], "evaluation steps are charged on top"
    assert ctx.budget.env_steps >= row["env_steps"]
