"""`train.elite_constraint.kind: l2_params` -- LaRes Eq. 4, anchored to the ELITE.

WHY THE REFERENCE IS EXPLICIT. `attach_l2_constraint` snapshots whatever the
model holds when it is called. Called after `train.init` has loaded `init_blob`,
without reading the descriptor's `reference_policy_ref`, it would anchor to the
wrong thing: under `train.interaction: shared_population` every slice after an
arm's first passes `resume_ref` = that arm's OWN previous slice, so `init_blob`
is the arm's own parameters and the "elite reference" would be itself -- a
proximal term toward its own starting point, for four of `configs/methods/lares.yaml`'s
five slices, with `elite_constraint_optimisers` still reading 2 on every one.

WHAT THE PAPER SAYS. Eq. 4 pulls each non-elite actor and both its critics
toward theta_elite, and the release holds ONE reference per evolution: each
worker loads `elite_actor` / `elite_q1` / `elite_q2` from `best_*.pth` when
`update_model_flag` is set and keeps them for the whole interval while the
elite's own agent keeps training (`refs/code/LaRes/utils.py:2119-2125`). In
BIRD that snapshot is `state.policy_ref`, the elite checkpoint carried into
the round, which only stage 6 moves -- so every slice of a round shares it.

WHAT THIS FILE PINS:

  1. the gradient term pulls toward the parameters in the ELITE's blob, not
     toward the model's own -- measured on `p.grad` after one patched step --
     and the model's own parameters are intact afterwards;
  2. on a resumed slice the seed row's `elite_constraint_reference` is the
     elite's policy ref while `warm_started_from` is the arm's own -- the two
     differ, and that difference is the point.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from conftest import REPO  # noqa: F401  (path setup)
from bird import registry
from bird.budget import Budget
from bird.components import training as T
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport

pytestmark = pytest.mark.slow

try:
    import stable_baselines3  # noqa: F401

    HAVE_SB3 = True
except ImportError:  # pragma: no cover - depends on the machine
    HAVE_SB3 = False

needs_sb3 = pytest.mark.skipif(not HAVE_SB3, reason="train.backend: sb3 needs sb3+gymnasium")

UPRIGHT = """
def compute_reward(state, action):
    import numpy as np
    th = np.arctan2(state[1], state[0])
    return -(th ** 2) - 0.1 * state[2] ** 2, {"upright": -(th ** 2)}
"""

#: The constrained arm's OWN program. Not UPRIGHT: a member carrying the
#: elite's program unchanged IS the elite continuing and `apply_to:
#: non_elite` exempts it (`population._is_elite`), so an arm that shared the
#: elite's code would be -- correctly -- unconstrained.
COSINE = """
def compute_reward(state, action):
    import numpy as np
    th = np.arctan2(state[1], state[0])
    return 10.0 * np.cos(th) + 3.0, {"cos": float(np.cos(th))}
"""


@pytest.fixture(autouse=True)
def _clean_stores():
    for store in (T._POLICY_STORE, T._REPLAY_STORE, T._ROUND_REPLAY):
        store.clear()
    if HAVE_SB3:
        import torch
        torch.set_num_threads(1)
    yield
    for store in (T._POLICY_STORE, T._REPLAY_STORE, T._ROUND_REPLAY):
        store.clear()


def _ctx(**overrides):
    """`lares` on the dev profile (sb3/SAC), pendulum, Eq. 4 on, Eq. 3 off."""
    base = {"seed": 0, "problem.env_id": "pendulum", "output.tracker": "none",
            "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "evaluate.rollouts_per_candidate": 1,
            "train.env_steps": 400,
            "train.interaction_cfg.slices_per_candidate": 2,
            "train.interaction_allocator": "uniform",
            "train.reward_scaling": "none",
            "train.elite_constraint.kind": "l2_params",
            "loop.carry": ["best_reward", "replay_buffer", "policy_checkpoint", "archive"]}
    base.update(overrides)
    cfg = load("lares", overrides=base, profile="dev")
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", "pendulum")(ctx)
    events = []
    ctx.event = lambda stage, **fields: events.append({"stage": stage, **fields})
    return ctx, events


def _elite_round(ctx, state, steps=100):
    backend = registry.get("train_backend", "sb3")
    elite = Candidate(cand_id="c0000", reward_code=UPRIGHT, iteration=0)
    res = backend(ctx, state, elite, 1, env_steps=steps)
    assert res.trained and res.policy_ref, res.error
    state.best = CandidateReport(cand_id=elite.cand_id, candidate=elite, result=res,
                                 fitness=0.5)
    state.replay_ref = res.replay_ref
    state.policy_ref = res.policy_ref
    return elite


# ==========================================================================
# 1. the term pulls toward the reference, not toward the model
# ==========================================================================


@needs_sb3
def test_the_gradient_pulls_toward_the_reference_blob_not_the_model():
    """`grad += 2w (theta - theta_elite)` with theta_elite read off the blob.

    The model is rebuilt from the elite's blob and then moved by +1 on every
    parameter, which is exactly a slice that has trained away from the elite.
    Through `_sb3_attach_elite_constraint` the patched step adds `2w * 1.0` to
    every grad; `attach_l2_constraint` on the model as it stands would add
    `2w * 0` -- the model would be its own reference.
    """
    import torch
    from stable_baselines3 import SAC

    ctx, _ = _ctx()
    state = RunState()
    _elite_round(ctx, state)
    blob = T._POLICY_STORE[state.policy_ref]
    model = SAC.load(io.BytesIO(np.asarray(blob, dtype=np.uint8).tobytes()), device="cpu")
    with torch.no_grad():
        for p in model.policy.parameters():
            p.add_(1.0)
    moved = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}

    w = 0.5
    n = T._sb3_attach_elite_constraint(model, w, blob, state.policy_ref)
    assert n >= 2, "actor and critic optimisers must both be patched"
    for k, v in model.policy.state_dict().items():
        assert torch.equal(v, moved[k]), f"the model's own parameters were not restored ({k})"

    for p in model.policy.parameters():
        p.grad = torch.zeros_like(p)
    model.actor.optimizer.step()
    model.critic.optimizer.step()
    pulled = [p for p in model.policy.parameters() if p.grad is not None
              and float(p.grad.abs().max()) > 0]
    assert pulled, "no parameter felt the pull toward the elite"
    for group in list(model.actor.optimizer.param_groups) + list(model.critic.optimizer.param_groups):
        for p in group["params"]:
            # grad = 2w (theta - theta_elite) = 2w * 1.0 for every entry, before
            # the optimiser's own step consumed it (Adam does not zero grads).
            assert torch.allclose(p.grad, torch.full_like(p.grad, 2.0 * w), atol=1e-5), (
                "the pull is not toward the reference blob")


# ==========================================================================
# 2. a resumed slice anchors to the elite, not to itself
# ==========================================================================


@needs_sb3
def test_a_resumed_slice_is_anchored_to_the_elite_not_to_its_own_previous_slice():
    """The self-anchoring shape, on the backend that runs it.

    Arm B's first slice warm-starts from the elite (`train.init`), its second
    RESUMES from its own first slice (`resume_ref`) -- and must still be
    constrained toward the elite. The seed row has to say both: what it
    started from and what it is pulled toward. Without the explicit
    reference, the pull would be toward `warm_started_from`.
    """
    ctx, _ = _ctx()
    state = RunState()
    _elite_round(ctx, state)
    backend = registry.get("train_backend", "sb3")
    arm = Candidate(cand_id="c0001", reward_code=COSINE, iteration=1)

    first = backend(ctx, state, arm, 1, env_steps=100,
                    handoff=T.SliceHandoff(index=0, export_ref="new:c0001:0"))
    assert first.trained, first.error
    row0 = first.seed_metrics[0]
    assert row0["warm_started_from"] == state.policy_ref
    assert row0["elite_constraint_optimisers"] >= 1
    assert row0["elite_constraint_reference"] == state.policy_ref

    second = backend(ctx, state, arm, 1, env_steps=100, resume_ref=first.policy_ref,
                     handoff=T.SliceHandoff(index=1, export_ref="new:c0001:1"))
    assert second.trained, second.error
    row1 = second.seed_metrics[0]
    assert row1["warm_started_from"] == first.policy_ref, "slice 2 resumes its own slice 1"
    assert first.policy_ref != state.policy_ref
    assert row1["elite_constraint_optimisers"] >= 1, "the term must still run on slice 2"
    assert row1["elite_constraint_reference"] == state.policy_ref, (
        "slice 2 must be pulled toward the ELITE, not toward "
        f"{row1['warm_started_from']} (its own previous slice)")
