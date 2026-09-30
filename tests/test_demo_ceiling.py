"""`train.pruning_cfg.ceiling: demo_return` and `train.pruning_metric: demo_fraction`.

Why this is a test. A guard that never fires and a guard that always fires
both render as a plausible run: the first is `plateau` with a
demonstration-access declaration it never used, the second is
`train.pruning: none` under another name. Both leave every counter looking
normal -- the same family as a config key that is read only into a string --
so the guard's DECISION is asserted here on a real backend call,
and the only place a wrong answer can be seen is `ceiling_decisions`.
"""
from __future__ import annotations

import types as pytypes

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components import training as T
from bird.config import ConfigError, load
from bird.context import Context
from bird.types import Candidate

_REWARD = """
def compute_reward(s, a, s2):
    import numpy as np
    o = np.asarray(s2, dtype=float)
    return float(-np.linalg.norm(o[0:2] - o[4:6]))
"""


def _ctx(**overrides):
    registry.load_all()
    overrides.setdefault("seed", 0)
    overrides.setdefault("problem.env_id", "toy_reacher")
    overrides.setdefault("problem.fitness_access", "demonstrations")
    overrides.setdefault("train.env_steps", 3000)
    overrides.setdefault("train.seeds_per_candidate", 1)
    overrides.setdefault("evaluate.rollouts_per_candidate", 1)
    cfg = load("eureka", overrides=overrides, profile="tester")
    return Context(cfg=cfg, budget=Budget(), env=registry.get("env", "toy_reacher")({}))


class _Always:
    """A rule that fires at every checkpoint from the third on."""

    def __call__(self, curve, budget_frac) -> bool:
        return len(curve) >= 3


# --------------------------------------------------------------------------
# the pure helpers
# --------------------------------------------------------------------------


def _cfg(**kv):
    return pytypes.SimpleNamespace(get=lambda k, d=None: kv.get(k, d))


def test_progress_is_the_share_of_the_random_to_expert_gap():
    ceiling = {"informative": True, "expert_return": -10.0, "random_return": -500.0}
    assert T._ceiling_progress(-500.0, ceiling) == pytest.approx(0.0)
    assert T._ceiling_progress(-10.0, ceiling) == pytest.approx(1.0)
    assert T._ceiling_progress(-255.0, ceiling) == pytest.approx(0.5)
    assert T._ceiling_progress(1.0, {"informative": False}) is None
    assert T._ceiling_progress(1.0, None) is None


def test_demo_fraction_falls_back_to_own_range_when_the_ceiling_is_uninformative():
    info = {"informative": True, "expert_return": 10.0, "random_return": 0.0}
    assert T._demo_fraction_point(5.0, info, [0.0, 1.0]) == pytest.approx(0.5)
    none = {"informative": False, "status": "uninformative"}
    assert T._demo_fraction_point(3.0, none, [1.0, 2.0]) == pytest.approx(1.0)  # a new best
    assert T._demo_fraction_point(1.0, none, [1.0, 3.0]) == pytest.approx(0.0)
    assert T._demo_fraction_point(1.0, none, []) == 0.0


def test_the_guard_blocks_below_the_fraction_and_allows_above_it():
    cfg = _cfg(**{"train.pruning_cfg.ceiling": "demo_return",
                  "train.pruning_cfg.ceiling_fraction": 0.8})
    ceiling = {"informative": True, "status": "informative",
               "expert_return": 10.0, "random_return": 0.0}
    decisions = []
    guarded = T._ceiling_guard(cfg, _Always(), "own_reward", ceiling, decisions)
    assert guarded([0.0, 1.0], 0.1) is False and decisions == [], "the rule did not fire"
    assert guarded([0.0, 1.0, 5.0], 0.3) is False, "50% of the gap: blocked"
    assert decisions[-1]["blocked"] is True and decisions[-1]["progress"] == pytest.approx(0.5)
    assert guarded([0.0, 1.0, 5.0, 9.0], 0.4) is True, "90% of the gap: allowed"
    assert decisions[-1]["blocked"] is False and decisions[-1]["progress"] == pytest.approx(0.9)
    # under `demo_fraction` the curve IS the progress
    decisions.clear()
    guarded = T._ceiling_guard(cfg, _Always(), "demo_fraction", ceiling, decisions)
    assert guarded([0.1, 0.2, 0.5], 0.3) is False and decisions[-1]["progress"] == pytest.approx(0.5)


def test_an_uninformative_or_absent_ceiling_cannot_guard_and_says_so():
    cfg = _cfg(**{"train.pruning_cfg.ceiling": "demo_return",
                  "train.pruning_cfg.ceiling_fraction": 0.8})
    for ceiling in ({"informative": False, "status": "uninformative"},
                    {"informative": False, "status": "unavailable"}, None):
        decisions = []
        guarded = T._ceiling_guard(cfg, _Always(), "own_reward", ceiling, decisions)
        assert guarded([0.0, 0.0, 0.0], 0.3) is True, "the plain rule's verdict stands"
        assert decisions[-1]["blocked"] is False
        assert decisions[-1]["reason"].startswith("ceiling_")


def test_the_guard_is_the_identity_when_off():
    rule = _Always()
    assert T._ceiling_guard(_cfg(), rule, "own_reward", None, []) is rule


# --------------------------------------------------------------------------
# on a real backend call
# --------------------------------------------------------------------------


def _train(ctx, monkeypatch, **kw):
    monkeypatch.setitem(T._PRUNERS, "median_stop", _Always())
    state = pytypes.SimpleNamespace(restart=0, iteration=0)
    cand = Candidate(cand_id="c0000", iteration=0, reward_code=_REWARD)
    res = T.mock_backend(ctx, state, cand, n_seeds=1, **kw)
    return res, cand


def test_an_unreached_ceiling_keeps_a_training_alive_that_the_rule_would_have_cut(monkeypatch):
    """Same rule, same reward, same seed; the only difference is the guard.
    `ceiling_fraction: 1.0` cannot be reached by a policy short of the expert,
    so every firing is blocked and the training runs to its budget."""
    plain, _ = _train(_ctx(**{"train.pruning": "median_stop"}), monkeypatch)
    assert plain.pruned is True, "the control must have been cut, or the guard proves nothing"
    guarded, cand = _train(_ctx(**{"train.pruning": "median_stop",
                                   "train.pruning_cfg.ceiling": "demo_return",
                                   "train.pruning_cfg.ceiling_fraction": 1.0}), monkeypatch)
    assert guarded.pruned is False
    assert guarded.env_steps_used > plain.env_steps_used
    ceiling = cand.meta["ceiling"]
    assert ceiling["status"] == "informative", ceiling
    assert ceiling["policies"][0].startswith("expert[env:toy_reacher]")
    decisions = guarded.seed_metrics[0]["ceiling_decisions"]
    assert decisions and all(d["blocked"] for d in decisions), decisions
    assert ceiling["blocked_prunes"] == len(decisions) and ceiling["allowed_prunes"] == 0
    # the `demo_fraction` field is on every checkpoint once a ceiling exists
    assert all("demo_fraction" in p for p in guarded.checkpoints)


def test_the_ceiling_is_computed_once_per_candidate_and_charged_as_rollouts(monkeypatch):
    ctx = _ctx(**{"train.pruning": "median_stop", "train.pruning_cfg.ceiling": "demo_return"})
    before = ctx.budget.env_steps
    res, cand = _train(ctx, monkeypatch)
    ceiling = cand.meta["ceiling"]
    # 2 policies x `verify.demo_screen.episodes` episodes, charged to env_steps
    # and never to `policy_trainings`
    assert ceiling["env_steps"] > 0
    assert ctx.budget.policy_trainings == 1
    # `env_steps_used`, not the seed-metric sum. The sum is
    # `spent + spent_eval` per seed; `env_steps_used` adds the post-training
    # evaluation rollouts, which are charged to the budget as the real env
    # interaction they are. Derived rather than a literal +25, because the
    # term moves with `evaluate.rollouts_per_candidate` and the episode
    # length -- a hardcoded literal goes stale.
    trained = sum(int(sm["env_steps"]) for sm in res.seed_metrics)
    eval_rollouts = int(res.env_steps_used) - trained
    assert eval_rollouts >= 0, (
        "env_steps_used is below the seed-metric sum, so the two are no "
        "longer nested and this decomposition is wrong, not merely stale")
    assert ctx.budget.env_steps - before == ceiling["env_steps"] + trained + eval_rollouts
    # a second call for the same candidate reuses the cached ceiling
    seen = dict(ceiling)
    T._demo_ceiling(ctx, pytypes.SimpleNamespace(iteration=0), cand, None)
    assert {k: cand.meta["ceiling"][k] for k in seen} == seen


def test_nothing_is_rolled_out_when_no_config_asks_for_a_ceiling(monkeypatch):
    ctx = _ctx(**{"train.pruning": "median_stop"})
    res, cand = _train(ctx, monkeypatch)
    assert "ceiling" not in cand.meta
    assert "ceiling_decisions" not in res.seed_metrics[0]
    assert not any("demo_fraction" in p for p in res.checkpoints)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad, needle", [
    ({"train.pruning_cfg.ceiling": "demo_return", "train.pruning": "none",
      "problem.fitness_access": "demonstrations"}, "nothing to guard"),
    ({"train.pruning_cfg.ceiling": "demo_return", "train.pruning": "plateau",
      "problem.fitness_access": "none"}, "fitness_access=demonstrations"),
    ({"train.pruning_cfg.ceiling": "demo_return", "train.pruning": "plateau",
      "train.pruning_metric": "task_metric", "problem.fitness_access": "demonstrations"},
     "two different quantities"),
    ({"train.pruning_cfg.ceiling": "demo_return", "train.pruning": "plateau",
      "train.pruning_cfg.ceiling_fraction": 1.5, "problem.fitness_access": "demonstrations"},
     "ceiling_fraction"),
    ({"train.pruning_metric": "demo_fraction", "train.pruning": "plateau",
      "problem.fitness_access": "none"}, "solution policy"),
    ({"train.pruning_metric": "demo_fraction", "train.pruning": "none",
      "problem.fitness_access": "demonstrations"}, "no rule reads"),
])
def test_the_ceiling_refuses_a_declared_knob_nothing_executes(bad, needle):
    with pytest.raises(ConfigError, match=needle):
        load("eureka", profile="tester", overrides=bad)
