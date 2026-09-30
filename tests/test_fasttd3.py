"""`train.backend: fasttd3` -- the port is held to the vendored upstream.

Two families of check, and they guard different failures:

  * FIDELITY -- the port must compute what `refs/code/FastTD3` computes.
    `_FASTTD3_DEFAULTS` is compared against an AST read of upstream's
    `BaseArgs` (a re-vendor that moves a default fails here instead of
    drifting), and the networks and the categorical projection are compared
    NUMERICALLY against the vendored classes after a state-dict copy -- both
    architectures. These need the vendored tree; a checkout without
    `refs/code/FastTD3` skips them by name.

  * CONTRACT -- the backend must mean the same thing every other backend
    means: a real checkpoint curve with `_sb3_run`'s keys, `train.init` off
    the same state slots, honest `env_steps`, a discrete env refused loudly,
    determinism from the config seed. These run on `pendulum` at tiny widths
    and are `slow`-marked like `test_train_init.py`'s, for the same reason:
    heavyweight execution, deselected by `-m "not slow"`.

torch gates everything below the AST tests: `--extra test` has no torch, so
the default CI job skips these by design (the sb3 precedent); `--extra all`
runs them.
"""

import ast
import io
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO  # noqa: F401  (path setup)
from bird import registry
from bird.budget import Budget
from bird.config import ConfigError, load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate

import bird.components.fasttd3 as FT
from bird.components import training as T

UPSTREAM = Path(REPO) / "refs" / "code" / "FastTD3" / "fast_td3"

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="train.backend: fasttd3 needs torch")
needs_upstream = pytest.mark.skipif(
    not UPSTREAM.is_dir(),
    reason="refs/code/FastTD3 not present; run scripts/fetch_refs.sh")

#: Tiny everywhere: the contract is what is under test, not learning.
TINY = {"num_envs": 2, "batch_size": 64, "buffer_size": 128,
        "critic_hidden_dim": 16, "actor_hidden_dim": 16,
        "num_atoms": 11, "v_min": -20.0, "v_max": 20.0,
        "learning_starts": 3, "compile": False,
        "actor_num_blocks": 1, "critic_num_blocks": 1}

REWARD = """
def compute_reward(state, action):
    import numpy as np
    th = np.arctan2(state[1], state[0])
    return -(th ** 2) - 0.1 * state[2] ** 2, {"upright": -(th ** 2)}
"""


def _ctx(env_id="pendulum", **overrides):
    base = {"seed": 0, "problem.env_id": env_id, "output.tracker": "none",
            "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "evaluate.rollouts_per_candidate": 2,
            "train.backend": "fasttd3", "train.algorithm": "fasttd3",
            "train.architecture": "mlp",
            "train.hyperparameters": dict(TINY), "train.env_steps": 400}
    base.update(overrides)
    cfg = load("rda", overrides=base, profile="dev")
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", env_id)(ctx)
    return ctx


def _train(ctx, cand_id="c0", state=None, **kw):
    backend = registry.get("train_backend", "fasttd3")
    cand = Candidate(cand_id=cand_id, reward_code=REWARD, iteration=0)
    return backend(ctx, state if state is not None else RunState(), cand, 1, **kw)


@pytest.fixture(autouse=True)
def _clean_stores():
    T._POLICY_STORE.clear()
    T._REPLAY_STORE.clear()
    yield
    T._POLICY_STORE.clear()
    T._REPLAY_STORE.clear()


# ==========================================================================
# No torch needed: registration, laziness, defaults-vs-upstream, coherence
# ==========================================================================


def test_the_backend_is_registered_and_the_module_imports_without_torch():
    """`registry.load_all()` imports this module for every config in the repo,
    so an import-time torch dependency would make a torch-less machine unable
    to validate configs it never intended to run -- the sb3 rule. Checked in a
    subprocess so an already-imported torch in THIS process cannot mask it."""
    assert "fasttd3" in registry.names("train_backend")
    code = ("import sys; import bird.components.fasttd3; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        "importing bird.components.fasttd3 imported torch at module scope:\n"
        + proc.stderr)


@needs_upstream
def test_defaults_match_upstreams_base_args_line_for_line():
    """`_FASTTD3_DEFAULTS` claims to be hyperparams.py's BaseArgs verbatim.
    Read BaseArgs off the vendored file with `ast` (no torch, no tyro) and
    hold every shared key to equality -- a re-vendor that moves an upstream
    default fails here instead of leaving the port citing a stale value."""
    tree = ast.parse((UPSTREAM / "hyperparams.py").read_text())
    base = next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "BaseArgs")
    upstream: dict = {}
    for node in base.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None \
                and isinstance(node.target, ast.Name):
            try:
                upstream[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass  # e.g. os.path.basename(...) -- harness-owned anyway
    # Keys the port deliberately owns differently, each argued in the module:
    #   compile  -- None means "cuda only"; upstream's True assumes the GPU it
    #               hard-requires (train.py raises "No GPU available").
    #   device   -- upstream has a `cuda` flag; this repo's key is `device`.
    ours = dict(FT._FASTTD3_DEFAULTS)
    exceptions = {"compile", "device"}
    shared = (set(ours) & set(upstream)) - exceptions
    assert len(shared) >= 25, f"suspiciously few shared keys: {sorted(shared)}"
    diffs = {k: (ours[k], upstream[k]) for k in sorted(shared)
             if ours[k] != upstream[k]}
    assert not diffs, f"port defaults diverge from BaseArgs (ours, upstream): {diffs}"
    # And the exceptions must still EXIST upstream in their upstream form, so
    # this list cannot quietly grow to hide a real divergence.
    assert upstream.get("compile") is True
    assert "cuda" in upstream and "device" not in upstream


def test_fasttd3_on_the_sb3_backend_is_refused_at_load():
    """sb3's algo map falls back to PPO for a name it does not know, so
    `train.algorithm: fasttd3` on `train.backend: sb3` would silently run a
    different algorithm under fasttd3's name -- `_check_coherence` refuses it."""
    with pytest.raises(ConfigError, match="fasttd3"):
        load("rda", overrides={"train.backend": "sb3",
                               "train.algorithm": "fasttd3"}, profile="dev")



# ==========================================================================
# Fidelity: the port against the vendored implementation (torch + refs)
# ==========================================================================


def _load_upstream(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_up_{name}", UPSTREAM / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_torch
@needs_upstream
@pytest.mark.parametrize("arch", ["mlp", "simba_v2"])
def test_networks_and_projection_match_upstream_numerically(arch):
    """Construct the vendored network and the port's, copy the vendored state
    dict across (the attribute names match by construction), and hold forward
    passes AND the categorical projection to exact equality on shared inputs.
    This is the whole fidelity claim in one check: same parameters -> same
    logits -> same projected target distribution."""
    import torch
    ns = FT._torch_classes()
    n_obs, n_act, atoms, hid = 5, 3, 11, 16
    torch.manual_seed(0)
    if arch == "mlp":
        up = _load_upstream("fast_td3")
        u_actor = up.Actor(n_obs=n_obs, n_act=n_act, num_envs=2, init_scale=0.01,
                           hidden_dim=hid, std_min=0.001, std_max=0.4,
                           sim_type="", sim_dimension=64, seq_len=8)
        u_critic = up.Critic(n_obs=n_obs, n_act=n_act, num_atoms=atoms,
                             v_min=-5.0, v_max=5.0, hidden_dim=hid,
                             sim_type="", sim_dimension=64, seq_len=8)
        p_actor = ns["Actor"](n_obs=n_obs, n_act=n_act, num_envs=2,
                              init_scale=0.01, hidden_dim=hid,
                              std_min=0.001, std_max=0.4)
        p_critic = ns["Critic"](n_obs=n_obs, n_act=n_act, num_atoms=atoms,
                                v_min=-5.0, v_max=5.0, hidden_dim=hid)
    else:
        import math
        up = _load_upstream("fast_td3_simbav2")
        kw_a = dict(n_obs=n_obs, n_act=n_act, num_envs=2, hidden_dim=hid,
                    scaler_init=math.sqrt(2.0 / hid), scaler_scale=math.sqrt(2.0 / hid),
                    alpha_init=1.0 / 2, alpha_scale=1.0 / math.sqrt(hid),
                    expansion=4, c_shift=3.0, num_blocks=1,
                    std_min=0.001, std_max=0.4)
        kw_c = dict(n_obs=n_obs, n_act=n_act, num_atoms=atoms, v_min=-5.0,
                    v_max=5.0, hidden_dim=hid,
                    scaler_init=math.sqrt(2.0 / hid), scaler_scale=math.sqrt(2.0 / hid),
                    alpha_init=1.0 / 2, alpha_scale=1.0 / math.sqrt(hid),
                    num_blocks=1, c_shift=3.0, expansion=4)
        u_actor, u_critic = up.Actor(**kw_a), up.Critic(**kw_c)
        p_actor, p_critic = ns["ActorSimba"](**kw_a), ns["CriticSimba"](**kw_c)

    p_actor.load_state_dict(u_actor.state_dict())
    p_critic.load_state_dict(u_critic.state_dict())

    torch.manual_seed(1)
    obs = torch.randn(7, n_obs)
    act = torch.randn(7, n_act).clamp(-1, 1)
    rew = torch.randn(7)
    bootstrap = torch.tensor([1., 1., 0., 1., 0., 1., 1.])
    discount = torch.full((7,), 0.99)

    with torch.no_grad():
        assert torch.equal(u_actor(obs), p_actor(obs))
        uq1, uq2 = u_critic(obs, act)
        pq1, pq2 = p_critic(obs, act)
        assert torch.equal(uq1, pq1) and torch.equal(uq2, pq2)
        up1, up2 = u_critic.projection(obs, act, rew, bootstrap, discount)
        pp1, pp2 = p_critic.projection(obs, act, rew, bootstrap, discount)
        assert torch.allclose(up1, pp1, atol=0, rtol=0)
        assert torch.allclose(up2, pp2, atol=0, rtol=0)
        # The projection must be a distribution: the atom-exact edge case
        # (fast_td3.py:131-137) loses mass if the masks are mis-ordered.
        assert torch.allclose(pp1.sum(dim=1), torch.ones(7), atol=1e-5)


@needs_torch
@needs_upstream
def test_projection_atom_exact_targets_lose_no_mass():
    """rewards that land exactly on an atom (b integer) are the projection's
    edge case; a port that fuses the two boundary masks double-deposits.
    Checked against upstream on inputs built to hit it."""
    import torch
    ns = FT._torch_classes()
    up = _load_upstream("fast_td3")
    torch.manual_seed(2)
    kw = dict(n_obs=4, n_act=2, num_atoms=5, v_min=-2.0, v_max=2.0, hidden_dim=16)
    u = up.Critic(sim_type="", sim_dimension=64, seq_len=8, **kw)
    p = ns["Critic"](**kw)
    p.load_state_dict(u.state_dict())
    obs, act = torch.randn(5, 4), torch.randn(5, 2)
    # delta_z = 1.0, so with bootstrap 0 these rewards project EXACTLY onto
    # atoms 0, 1, 2, 4 and the clamp edge. Atom 1 is the load-bearing case:
    # a port that fuses the two boundary masks re-reads the DECREMENTED floor,
    # and b == 1 is the only input where that changes the answer (low becomes
    # 0, the u_mask fires, and mass double-deposits at atoms 0 and 2) --
    # verified by mutation: without the -1.0 row the fused-mask bug is green.
    rew = torch.tensor([-2.0, -1.0, 0.0, 2.0, 3.5])
    zeros = torch.zeros(5)
    with torch.no_grad():
        up1, _ = u.projection(obs, act, rew, zeros, torch.full((5,), 0.99))
        pp1, _ = p.projection(obs, act, rew, zeros, torch.full((5,), 0.99))
    assert torch.allclose(up1, pp1, atol=0, rtol=0)
    assert torch.allclose(pp1.sum(dim=1), torch.ones(5), atol=1e-6)


# ==========================================================================
# Contract: the backend behaves like a §3 backend (torch; slow)
# ==========================================================================


@needs_torch
@pytest.mark.slow
@pytest.mark.parametrize("arch", ["mlp", "simba_v2"])
def test_the_backend_trains_end_to_end_with_sb3s_contract(arch):
    ctx = _ctx(**{"train.architecture": arch})
    res = _train(ctx, f"e2e_{arch}")
    assert res.trained, res.error
    assert res.error == "" and not res.timed_out
    # the curve is real and carries _sb3_run's keys
    assert len(res.checkpoints) >= T.MIN_CHECKPOINTS
    for key in ("step", "round", "fitness", "success_rate", "gt_return",
                "reward_return", "return"):
        assert key in res.checkpoints[0], key
    assert len(res.gt_reward_curve) == len(res.checkpoints)
    assert len(res.trajectories) == 2
    assert res.trajectories[0].states is not None
    # component traces reach §4 (the reward above declares one component)
    assert "upright" in res.component_traces
    # the seed row records the backend as executed
    sm = res.seed_metrics[0]
    assert sm["backend"] == "fasttd3" and sm["learner"] == "fasttd3"
    assert sm["architecture"] == arch
    # `algorithm_cited` IS THE PAPER'S ALGORITHM, NOT THE ONE THAT RAN, and
    # must not be read as the backend as executed. It reads
    # `fasttd3` here only because THIS config cites fasttd3; the execution
    # facts are `learner` and `backend` on the line above, and a config
    # citing PPO would put `ppo` here while still training on fasttd3.
    assert sm["algorithm_cited"] == ctx.cfg["train.algorithm"] == "fasttd3"
    assert "algorithm" not in sm, (
        "the bare name is back: a reader cannot tell a citation from an "
        "execution fact, which is what the rename removed")
    assert sm["num_envs"] == TINY["num_envs"]
    assert sm["checkpoint_selection"] == "final"
    # honest step accounting: learning steps == iterations * num_envs <= ask
    assert sm["train_steps"] <= sm["train_steps_requested"] == 400
    assert sm["train_steps"] % TINY["num_envs"] == 0
    assert res.env_steps_used > sm["train_steps"]  # + eval + rollouts
    assert ctx.budget.env_steps >= sm["env_steps"]
    # the last seed's agent is published for train.init
    assert res.policy_ref and res.policy_ref in T._POLICY_STORE


@needs_torch
@pytest.mark.slow
def test_env_workers_are_a_schedule_not_a_science_knob():
    """`train.n_parallel_envs` forks slot workers behind `_VecEnvView`. Same
    seed, one worker vs four: every checkpoint row identical -- the slots keep
    their own `(seed, slot, episode)` streams and the reward normaliser runs in
    the parent in slot order, so the worker count must not be able to reach a
    single number. Also pins that the seed row says what ran (`env_workers`)
    and what the buffer allocated (`buffer_rows` <= iterations + 1)."""
    # Eight slots over THREE workers: blocks of 3, 3 and 2, so every reply
    # carries a multi-element raw-reward array (a one-slot-per-worker split
    # would let an `ndarray == "error"` comparison pass unseen).
    hyper = dict(TINY, num_envs=8)
    # Same cand_id on purpose: `training._seed_base` folds it into the seed.
    r1 = _train(_ctx(seed=11, **{"train.hyperparameters": hyper}), "wk")
    r4 = _train(_ctx(seed=11, **{"train.hyperparameters": hyper,
                                 "train.n_parallel_envs": 3}), "wk")
    key = lambda r: [(p["fitness"], p["gt_return"], p["reward_return"])  # noqa: E731
                     for p in r.checkpoints]
    assert key(r1) == key(r4), "the worker count changed the numbers"
    m1, m4 = r1.seed_metrics[0], r4.seed_metrics[0]
    assert (m1["env_workers"], m4["env_workers"]) == (1, 3)
    assert m1["buffer_rows"] == m4["buffer_rows"] <= 400 // 8 + 1


@needs_torch
@pytest.mark.slow
def test_same_seed_same_run_different_seed_different_run():
    r1 = _train(_ctx(seed=7), "det")
    r2 = _train(_ctx(seed=7), "det")
    r3 = _train(_ctx(seed=8), "det")
    key = lambda r: [(p["fitness"], p["gt_return"], p["reward_return"])  # noqa: E731
                     for p in r.checkpoints]
    assert key(r1) == key(r2), "one seed, two different runs"
    assert key(r1) != key(r3), "the seed key does not reach the learner"


@needs_torch
def test_a_discrete_env_is_refused_loudly():
    """A config-level condition must not burn the search failing candidates
    one by one -- the backend raises, naming the alternative."""
    ctx = _ctx(env_id="toy_gridworld")
    with pytest.raises(ValueError, match="discrete"):
        _train(ctx, "disc")


@needs_torch
@pytest.mark.slow
def test_a_citation_architecture_warns_and_runs_the_papers_network(caplog):
    """rda pins `train.architecture: simba` as a citation; the backend runs the
    base mlp for any value it does not implement, says so, and the seed row
    records the EXECUTED value -- the surrogates' warn-and-record shape, so a
    published config stays runnable without editing a citation key."""
    import logging
    ctx = _ctx(**{"train.architecture": "simba", "train.env_steps": 80})
    with caplog.at_level(logging.WARNING, logger="bird.train"):
        res = _train(ctx, "arch")
    assert res.trained
    assert res.seed_metrics[0]["architecture"] == "mlp"
    assert any("train.architecture" in r.message for r in caplog.records)


@needs_torch
@pytest.mark.slow
def test_warm_start_round_trips_and_a_foreign_blob_cold_starts():
    """`train.init: warm_start_from_best` off the same state slots as every
    backend: run one candidate, hand its policy_ref to a second via RunState,
    and the second's seed row records the warm start AS EXECUTED. A garbage
    blob (another backend's format) degrades to a cold start with a warning,
    never a crash."""
    ctx = _ctx()
    first = _train(ctx, "incumbent")
    assert first.policy_ref
    state = RunState()
    state.policy_ref = first.policy_ref
    # No loop.carry override: `_training_init` reads `state.policy_ref`
    # directly, and replacing rda's carry list would break ITS coherence rules.
    ctx2 = _ctx(**{"train.init": "warm_start_from_best"})
    second = _train(ctx2, "challenger", state=state)
    assert second.trained
    assert second.seed_metrics[0]["warm_started_from"] == first.policy_ref

    # a foreign blob under the same ref: cold start, recorded as such
    T._POLICY_STORE[first.policy_ref] = T._PolicyBlob(b"not a torch payload")
    third = _train(ctx2, "challenger2", state=state)
    assert third.trained
    assert third.seed_metrics[0]["warm_started_from"] == ""


@needs_torch
@pytest.mark.slow
def test_the_short_budget_override_is_honoured():
    """`screens._train_backend_short` passes `env_steps=` when the backend
    advertises it; a screen that silently ran the full budget would misreport
    exactly the quantity LIMEN's cascade exists to save."""
    ctx = _ctx()
    res = _train(ctx, "short", env_steps=60)
    assert res.trained
    assert res.seed_metrics[0]["train_steps_requested"] == 60
    assert res.seed_metrics[0]["train_steps"] <= 60


@needs_torch
@pytest.mark.slow
def test_n_step_returns_train_end_to_end():
    hyper = dict(TINY, num_steps=3)
    ctx = _ctx(**{"train.hyperparameters": hyper})
    res = _train(ctx, "nstep")
    assert res.trained, res.error
    assert len(res.checkpoints) >= T.MIN_CHECKPOINTS


@needs_torch
@pytest.mark.slow
def test_the_vec_view_reproduces_the_upstream_env_contract():
    """The `_VecEnvView` data flow against upstream's wrapper semantics:
    `dones` is termination OR truncation, `time_outs` flags the truncation, a
    finished slot auto-resets (the returned next_obs is the new episode's
    first state) while `true_next` carries the terminal state, and slots are
    reproducible from (seed, slot, episode)."""
    from bird.components.training import compile_reward
    ctx = _ctx()
    reward = compile_reward(REWARD, None)
    env = ctx.env
    view = FT._VecEnvView(env, 2, reward,
                          lambda s: np.asarray(s, dtype=np.float32),
                          "none", seed=3, obs_width=int(np.asarray(env.obs_low).size))
    obs = view.reset()
    assert obs.shape == (2, int(np.asarray(env.obs_low).size))
    # the action map is the spec, stated directly: +/-1 in agent space IS the
    # env bound (pendulum: +/-2), not a clamp and not the identity
    assert np.allclose(view.to_env_action(np.ones_like(env.action_high)),
                       env.action_high)
    assert np.allclose(view.to_env_action(-np.ones_like(env.action_low)),
                       env.action_low)
    horizon = int(env.horizon)
    a = np.zeros((2, int(np.asarray(env.action_low).size)), dtype=np.float32)
    for t in range(horizon - 1):
        _, _, dones, touts, _ = view.step(a)
        assert not dones.any(), "pendulum has no terminal before the horizon"
    next_obs, _, dones, touts, true_next = view.step(a)
    assert dones.all() and touts.all(), "the horizon is a truncation"
    # auto-reset: the state the buffer bootstraps from is the terminal one,
    # the state the next action sees is a fresh episode's
    assert not np.allclose(next_obs, true_next)
    # and the fresh episodes are the deterministic (seed, slot, episode=1) draws
    view2 = FT._VecEnvView(env, 2, reward,
                           lambda s: np.asarray(s, dtype=np.float32),
                           "none", seed=3, obs_width=int(np.asarray(env.obs_low).size))
    view2.reset()
    for t in range(horizon):
        n2, _, _, _, _ = view2.step(a)
    assert np.allclose(next_obs, n2)


@needs_torch
@pytest.mark.slow
def test_the_dedicated_tpe_store_pool_is_collected():
    """CARD's shape: `update.memory.trajectory_store != none` +
    `verify.tpe.trajectories_per_iteration > 0` collects a DEDICATED pool into
    `TrainResult.store_trajectories` (update.py reads it with an explicit
    no-fallback rule, so a backend that skips it starves the TPE screen forever
    and CARD silently becomes its own ablation). Charged to the budget's
    rollout counter, and the
    eval rollouts stay byte-identical when collection is toggled (separate rng)."""
    ctx = _ctx(**{"update.memory.trajectory_store": "append_on_pass",
                  "verify.tpe.trajectories_per_iteration": 2})
    res = _train(ctx, "store")
    assert res.trained, res.error
    assert len(res.store_trajectories) == 2
    assert all(t.states is not None and t.length > 0 for t in res.store_trajectories)
    # `record_rollout_steps` charges the shared env_steps counter (see its
    # docstring), so the budget must exceed the training-side
    # charge by at least the collection steps.
    store_steps = sum(t.length for t in res.store_trajectories)
    assert ctx.budget.env_steps >= res.seed_metrics[0]["env_steps"] + store_steps
    # toggling collection must not move the eval rollouts (separate rng)
    ctx0 = _ctx()
    res0 = _train(ctx0, "store")
    assert [t.ret for t in res0.trajectories] == [t.ret for t in res.trajectories]


@needs_torch
@pytest.mark.slow
def test_the_seed_fork_matches_the_sequential_schedule():
    """`train.candidate_parallelism: parallel` widens the SEED loop of a
    backend call running in the parent (the final_retrain shape). The forked
    schedule rebuilds each policy from the exact parameters the child trained,
    so the seed rows and the parent-side rollouts must match sequential's --
    a schedule knob that moved a number would be changing the method."""
    def run(mode):
        ctx = _ctx(**{"train.seeds_per_candidate": 2,
                      "train.candidate_parallelism": mode,
                      "loop.max_parallel_trainings": 2})
        backend = registry.get("train_backend", "fasttd3")
        cand = Candidate(cand_id="fork", reward_code=REWARD, iteration=0)
        return backend(ctx, RunState(), cand, 2)

    seq = run("sequential")
    par = run("parallel")
    key = lambda r: [[(p["fitness"], p["gt_return"], p["reward_return"])  # noqa: E731
                      for p in m["checkpoints"]] for m in r.seed_metrics]
    assert key(seq) == key(par)
    assert [m["seed"] for m in seq.seed_metrics] == [m["seed"] for m in par.seed_metrics]
    assert seq.policy_ref and par.policy_ref
    assert [t.ret for t in seq.trajectories] == [t.ret for t in par.trajectories]


@needs_torch
@pytest.mark.slow
def test_actions_reach_the_env_in_env_space():
    """Pendulum's torque bound is +/-2; the agent acts in [-1, 1] and the
    boundary maps linearly. A rollout's recorded actions are ENV-space."""
    ctx = _ctx()
    res = _train(ctx, "act")
    acts = np.concatenate([t.actions.ravel() for t in res.trajectories])
    hi = float(np.asarray(ctx.env.action_high).ravel()[0])
    assert np.all(np.abs(acts) <= hi + 1e-6)
    # and the seed row says how often TRAINING's sampled action had to be
    # clipped to become the executed one the reward was evaluated on
    assert 0.0 <= res.seed_metrics[0]["action_clip_fraction"] <= 1.0


# ==========================================================================
# LaRes keys on this backend: `train.reward_scaling`, `train.elite_constraint`
# ==========================================================================
#
# `_run_backend` and `_sb3_run` wrap compilation in `_scaled_reward` and
# `_sb3_run` attaches the L2 pull after the warm start. A backend that compiled
# bare and never read `train.elite_constraint.kind` -- with no warning, no
# seed-row field and no coherence rule -- would make
# `-c lares -s train.backend=fasttd3` run LaRes's own Fig. 6 ablation under
# LaRes's name. The scaling test below also guards `elite_moments`: indexing
# `action_set` with the ENV-SPACE action arrays a continuous backend's replay
# export holds would raise on every row, and Eq. 3 would announce "fewer than
# two usable transitions" on every continuous env. fasttd3 refuses discrete
# envs, so without that handling nothing would execute.

#: Exactly `10 * REWARD + 3`, so LaRes Eq. 3's affine map is x -> 10x + 3 to
#: rounding, whatever the buffer holds.
ELITE_REWARD = """
def compute_reward(state, action):
    import numpy as np
    th = np.arctan2(state[1], state[0])
    return 10.0 * (-(th ** 2) - 0.1 * state[2] ** 2) + 3.0, {"upright": -(th ** 2)}
"""


def _elite_state(elite_id="elite", replay=None, policy_ref=None):
    """A `RunState` after one iteration under LaRes: an incumbent, and the
    carried replay/policy refs the two mechanisms read."""
    from bird.types import CandidateReport, TrainResult
    elite = Candidate(cand_id=elite_id, reward_code=ELITE_REWARD, iteration=0)
    st = RunState()
    st.best = CandidateReport(cand_id=elite_id, candidate=elite,
                              result=TrainResult(cand_id=elite_id, candidate=elite),
                              fitness=1.0)
    if replay is not None:
        T._REPLAY_STORE["replay:elite"] = replay
        st.replay_ref = "replay:elite"
    st.policy_ref = policy_ref
    return st


def _continuous_slice(env, n=64, seed=0):
    """What `_fasttd3_export_replay`/`_sb3_export_replay` write: state-space
    observations and ENV-SPACE action ARRAYS, not `action_set` indices."""
    rng = np.random.default_rng(seed)
    obs = np.stack([env.random_state(rng) for _ in range(n)]).astype(np.float32)
    act = rng.uniform(env.action_low, env.action_high,
                      size=(n, int(np.asarray(env.action_low).size))).astype(np.float32)
    nxt = np.stack([np.asarray(env.step(s, a)[0]) for s, a in zip(obs, act)]).astype(np.float32)
    return T._ReplaySlice(obs, act, nxt, np.zeros(n, dtype=bool))


LARES_SCALING = {"train.reward_scaling": "elite_moments",
                 "loop.carry": ["best_reward", "subtask_list", "replay_buffer", "policy_checkpoint"]}


def _plan_scaling(ctx, state, cand):
    """Stage 3's half, as `bird.py::train` runs it in the PARENT once per
    trainable candidate: the planner leaves its plan on
    `candidate.meta["reward_scaling"]`; the backend only APPLIES it."""
    return registry.get("reward_scaling", ctx.cfg["train.reward_scaling"])(ctx, state, cand)


def test_elite_moments_scales_a_slice_with_env_space_actions():
    """The component on a continuous backend's export: the parent's plan
    applies, and the backend's apply step (`_scaled_reward`) honours it.
    Skipping every row is announced, not raised, so only the plan tells."""
    ctx = _ctx(**LARES_SCALING)
    state = _elite_state(replay=_continuous_slice(ctx.env))
    cand = Candidate(cand_id="c_new", reward_code=REWARD, iteration=0)
    plan = _plan_scaling(ctx, state, cand)
    assert plan["applied"] is True, f"every row was skipped: {plan}"
    assert cand.meta["reward_scaling"] is plan
    assert plan["scale"] == pytest.approx(10.0, rel=1e-5)
    raw = T.compile_reward(REWARD, cand)
    scaled = T._scaled_reward(ctx, state, cand, raw)
    assert scaled is not raw
    s, a, s2, _ = T._REPLAY_STORE["replay:elite"][0]
    assert scaled(s, a, s2)[0] == pytest.approx(10.0 * raw(s, a, s2)[0] + 3.0, rel=1e-5)


@needs_torch
@pytest.mark.slow
def test_reward_scaling_reaches_the_fasttd3_learner(monkeypatch):
    """Plan in the parent, apply in the backend: the reward the slots collect
    under is the SCALED one and the seed row carries the plan's record. The
    spy takes the reward off `_VecEnvView`'s constructor by duck type so it
    does not depend on the constructor's shape."""
    ctx = _ctx(**LARES_SCALING)
    state = _elite_state(replay=_continuous_slice(ctx.env))
    cand = Candidate(cand_id="c_new", reward_code=REWARD, iteration=0)
    assert _plan_scaling(ctx, state, cand)["applied"] is True
    seen = []
    real_view = FT._VecEnvView

    class Spy(real_view):
        def __init__(self, *args, **kwargs):
            seen.append(next(a for a in args if hasattr(a, "component_names")))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(FT, "_VecEnvView", Spy)
    res = registry.get("train_backend", "fasttd3")(ctx, state, cand, 1)
    assert res.trained, res.error
    rec = res.seed_metrics[0]["reward_scaling"]
    assert rec["applied"] is True and rec["mode"] == "elite_moments"
    assert rec["scale"] == pytest.approx(10.0, rel=1e-5)
    raw = T.compile_reward(REWARD, None)
    s, a, s2, _ = T._REPLAY_STORE["replay:elite"][0]
    assert seen, "no view was built"
    assert seen[0](s, a, s2)[0] == pytest.approx(10.0 * raw(s, a, s2)[0] + 3.0, rel=1e-5)
    # a planned SKIP is recorded as such, with its reason -- not as nothing
    cand2 = Candidate(cand_id="c_skip", reward_code=REWARD, iteration=0)
    state2 = _elite_state()                      # an elite, but no buffer yet
    assert _plan_scaling(ctx, state2, cand2)["applied"] is False
    res2 = registry.get("train_backend", "fasttd3")(ctx, state2, cand2, 1)
    rec2 = res2.seed_metrics[0]["reward_scaling"]
    assert rec2["applied"] is False and rec2["reason"]
    # and under `none` the row carries no field at all (byte-identical to before)
    res0 = _train(_ctx(), "c_plain")
    assert "reward_scaling" not in res0.seed_metrics[0]


@needs_torch
def test_attach_l2_to_optimisers_adds_the_exact_gradient_of_the_pull():
    """`grad += 2w(theta - theta_0)` before every step, on every optimiser
    given once, matched by parameter IDENTITY: the actor and the critic share
    parameter names (`weight`, `bias` here; `net.0.weight` in the port), and a
    name-keyed reference would let one shadow the other."""
    import torch
    from bird.components.population import attach_l2_to_optimisers
    torch.manual_seed(0)
    actor, critic = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    params = list(actor.parameters()) + list(critic.parameters())
    opts = [torch.optim.SGD(actor.parameters(), lr=1.0),
            torch.optim.SGD(critic.parameters(), lr=1.0)]
    ref = [p.detach().clone() for p in params]
    # the same optimiser passed twice is patched once
    assert attach_l2_to_optimisers(params, opts + [opts[0]], weight=0.5) == 2
    with torch.no_grad():
        for p in params:
            p.add_(1.0)                       # theta = theta_0 + 1, everywhere
            p.grad = torch.zeros_like(p)      # the loss itself says "stay"
    for o in opts:
        o.step()
    # SGD at lr 1 with the injected 2 * 0.5 * (theta - theta_0) = 1 lands on theta_0
    for p, r in zip(params, ref):
        assert torch.allclose(p.detach(), r), "the pull did not reach this parameter"
    # and re-attaching does not double the term
    assert attach_l2_to_optimisers(params, opts, weight=0.5) == 0


@needs_torch
@pytest.mark.slow
def test_the_elite_constraint_is_attached_to_actor_and_critic_optimisers():
    """LaRes Eq. 4 on this backend: after `warm_start_from_best` loads the
    elite, both AdamW optimisers carry the pull and the seed row counts them
    -- 2, as on sb3's SAC (`actor.optimizer`, `critic.optimizer`). The elite
    itself is exempt under `apply_to: non_elite` and records 0."""
    first = _train(_ctx(), "elite")
    assert first.trained and first.policy_ref in T._POLICY_STORE
    ctx = _ctx(**{"train.init": "warm_start_from_best",
                  "train.elite_constraint.kind": "l2_params",
                  "loop.carry": ["best_reward", "subtask_list", "policy_checkpoint"]})
    state = _elite_state(elite_id="elite", policy_ref=first.policy_ref)
    res = _train(ctx, "c_new", state=state)
    assert res.trained, res.error
    sm = res.seed_metrics[0]
    assert sm["warm_started_from"] == first.policy_ref
    assert sm["elite_constraint_optimisers"] == 2
    assert sm["elite_constraint_reference"] == first.policy_ref
    res_e = _train(ctx, "elite", state=state)
    assert res_e.seed_metrics[0]["elite_constraint_optimisers"] == 0
    assert res_e.seed_metrics[0]["elite_constraint_reference"] == ""
    # a plain run carries the count too, at 0 -- absence and "not running" must
    # not look alike in the artifact -- and no reference under `kind: none`
    assert first.seed_metrics[0]["elite_constraint_optimisers"] == 0
    assert "elite_constraint_reference" not in first.seed_metrics[0]


@needs_torch
def test_the_fasttd3_pull_is_toward_the_elite_blob_not_the_live_agent():
    """`_fasttd3_attach_elite_constraint` mirrors `_sb3_attach_elite_constraint`:
    the term is `2w (theta - theta_ELITE)` with theta_elite read off the blob,
    whatever the live agent holds, and the agent's own parameters are intact
    afterwards. A blob that is not ours patches nothing and touches nothing."""
    import torch
    ns = FT._torch_classes()
    dev = torch.device("cpu")

    def build(seed):
        torch.manual_seed(seed)
        actor = ns["Actor"](n_obs=3, n_act=1, num_envs=2, init_scale=0.01, hidden_dim=16,
                            std_min=0.05, std_max=0.4, device=dev)
        kw = dict(n_obs=3, n_act=1, num_atoms=11, v_min=-20.0, v_max=20.0,
                  hidden_dim=16, device=dev)
        qnet, qt = ns["Critic"](**kw), ns["Critic"](**kw)
        qt.load_state_dict(qnet.state_dict())
        return actor, qnet, qt

    e_actor, e_qnet, e_qt = build(1)
    blob = FT._fasttd3_blob(ns, "mlp", e_actor, e_qnet, e_qt, FT._NullState())
    elite = [p.detach().clone() for p in list(e_actor.parameters()) + list(e_qnet.parameters())]
    actor, qnet, _ = build(2)                          # a slice that trained away
    params = list(actor.parameters()) + list(qnet.parameters())
    own = [p.detach().clone() for p in params]
    opts = [torch.optim.SGD(actor.parameters(), lr=1.0),
            torch.optim.SGD(qnet.parameters(), lr=1.0)]
    n = FT._fasttd3_attach_elite_constraint(ns, "mlp", actor, qnet, opts, 0.5, blob, "policy:elite")
    assert n == 2
    for p, o in zip(params, own):
        assert torch.equal(p.detach(), o), "the swap must leave the agent's own parameters"
    for p in params:
        p.grad = torch.zeros_like(p)               # the loss itself says "stay"
    for o in opts:
        o.step()
    for p, o, e in zip(params, own, elite):
        assert torch.allclose(p.grad, 2 * 0.5 * (o - e)), "pull is toward the ELITE, not the agent"
        assert torch.allclose(p.detach(), e, atol=1e-6), "SGD at lr 1 lands on theta_elite"
    # not ours (an sb3 zip, say): 0, unpatched, untouched
    actor2, qnet2, _ = build(3)
    before = [p.detach().clone() for p in list(actor2.parameters()) + list(qnet2.parameters())]
    opts2 = [torch.optim.SGD(actor2.parameters(), lr=1.0)]
    assert FT._fasttd3_attach_elite_constraint(ns, "mlp", actor2, qnet2, opts2, 0.5,
                                               np.zeros(8, dtype=np.uint8), "policy:x") == 0
    assert not getattr(opts2[0], "_bird_l2_elite", False)
    for p, b in zip(list(actor2.parameters()) + list(qnet2.parameters()), before):
        assert torch.equal(p.detach(), b)


@needs_torch
@pytest.mark.slow
def test_a_resumed_fasttd3_slice_is_anchored_to_the_elite_not_to_itself():
    """LaRes's failing shape (tests/test_elite_constraint.py) on this backend:
    an arm's slice 1 warm-starts from the elite, its slice 2 RESUMES from its
    own slice 1 (`resume_ref`) and must still be pulled toward the elite. The
    seed row has to say both: what it started from and what it is pulled
    toward, and on slice 2 the two differ."""
    first = _train(_ctx(), "elite")
    assert first.trained and first.policy_ref in T._POLICY_STORE
    ctx = _ctx(**{"train.init": "warm_start_from_best",
                  "train.elite_constraint.kind": "l2_params",
                  "loop.carry": ["best_reward", "subtask_list", "policy_checkpoint"]})
    state = _elite_state(elite_id="elite", policy_ref=first.policy_ref)
    slice1 = _train(ctx, "arm", state=state)
    assert slice1.trained, slice1.error
    row0 = slice1.seed_metrics[0]
    assert row0["warm_started_from"] == first.policy_ref
    assert row0["elite_constraint_optimisers"] == 2
    assert row0["elite_constraint_reference"] == first.policy_ref
    slice2 = _train(ctx, "arm", state=state, resume_ref=slice1.policy_ref)
    assert slice2.trained, slice2.error
    row1 = slice2.seed_metrics[0]
    assert slice1.policy_ref != first.policy_ref
    assert row1["warm_started_from"] == slice1.policy_ref, "slice 2 resumes its own slice 1"
    assert row1["elite_constraint_optimisers"] == 2, "the term must still run on slice 2"
    assert row1["elite_constraint_reference"] == first.policy_ref, (
        "slice 2 must be pulled toward the ELITE, not toward "
        f"{row1['warm_started_from']} (its own previous slice)")


# ==========================================================================
# One adapter instance per slot: domain randomisation and the episode RNG
# ==========================================================================
#
# `EnvAdapter.reset` caches the DR draw and the episode RNG on the INSTANCE
# (`_dr_now`, `_rng`) and `_step` reads them back, so `num_envs` episodes
# interleaved on one adapter would all run under whichever slot had reset
# last, and the between-chunk evaluation's final reset would leak its draw into
# every in-flight training episode. No test here needs torch: the
# view is numpy over the adapters.

DR_RANGE = {"torque_scale": (0.5, 1.5), "action_noise": (0.05, 0.1)}


def _feat(s):
    return np.asarray(s, dtype=np.float32)


def test_each_slot_owns_its_dr_draw_and_keeps_it_for_the_whole_episode():
    """One adapter, four slots: slot i executes ITS OWN `(seed, i, episode)`
    draw on every one of its steps, while another slot resets mid-episode and
    the same adapter is reset by an evaluation in between -- the shared-adapter
    leak, closed by the per-slot restore rather than by four instances."""
    ctx = _ctx()
    env = ctx.env
    env.set_dr(DR_RANGE)
    width = int(np.asarray(env.obs_low).size)
    view = FT._VecEnvView(env, 4, T.compile_reward(REWARD, None), _feat, "none",
                          seed=3, obs_width=width)
    assert view.env is env, "training runs on ctx.env; nothing else is constructed"
    view.reset()
    draws = [view.episode_state(i)[1]["torque_scale"] for i in range(4)]
    assert len(set(draws)) == 4, f"one draw shared by the fleet: {draws}"
    # slot i holds ITS OWN (seed, i, episode) draw -- the class docstring's
    # 'distinct by construction' promise, stated against a fresh adapter
    for i in range(4):
        probe = registry.get("env", "pendulum")(ctx)
        probe.set_dr(DR_RANGE)
        probe.reset(np.random.default_rng((3, i, 0)))
        assert view.episode_state(i)[1] == probe._dr_now
    # ...and every one of its steps RUNS under that draw: the spy reads the
    # adapter's live `_dr_now` at `_step` time; slots are stepped in order,
    # so call k belongs to slot k % 4
    calls = []
    real_step = env._step

    def spy(s, a):
        calls.append(env._dr_now["torque_scale"])
        return real_step(s, a)
    env._step = spy
    a = np.zeros((4, int(np.asarray(env.action_low).size)), dtype=np.float32)
    for t in range(5):
        view.step(a)
    view._reset_slot(2)
    env.reset(np.random.default_rng(99))          # the between-chunk eval reset
    for t in range(5):
        view.step(a)
    seen = {i: set() for i in range(4)}
    for k, v in enumerate(calls):
        seen[k % 4].add(v)
    for i in (0, 1, 3):
        assert seen[i] == {draws[i]}, f"slot {i} ran under a foreign draw: {seen[i]}"
    assert len(seen[2]) == 2 and draws[2] in seen[2], "slot 2's reset drew afresh"
    assert view.episode_state(2)[1]["torque_scale"] != draws[2]


def test_one_adapter_with_per_slot_restore_matches_a_fleet_of_isolated_adapters():
    """The correctness claim behind not constructing `num_envs` adapters: the
    view on ONE adapter produces, bit for bit, the transitions a fleet of
    separately constructed adapters stepped in lockstep produces -- under DR
    on both axes pendulum randomises, with `action_noise` drawing from the
    episode RNG every step, across an auto-reset, and with `ctx.env` reset by
    an evaluation halfway through."""
    ctx = _ctx()
    env = ctx.env
    env.set_dr(DR_RANGE)
    n, width = 4, int(np.asarray(env.obs_low).size)
    reward = T.compile_reward(REWARD, None)
    horizon = int(env.horizon)
    g = np.random.default_rng(5)
    acts = [g.uniform(-1.2, 1.2, size=(n, int(np.asarray(env.action_low).size)))
            .astype(np.float32) for _ in range(horizon + 3)]

    view = FT._VecEnvView(env, n, reward, _feat, "none", seed=11, obs_width=width)
    view.reset()
    got = []
    for t, a in enumerate(acts):
        if t == horizon // 2:
            env.reset(np.random.default_rng(99))
        obs, rew, dones, _, true_next = view.step(a)
        got.append((obs.copy(), rew.copy(), dones.copy(), true_next.copy()))

    fleet = []
    for i in range(n):
        e = registry.get("env", "pendulum")(ctx)
        e.set_dr(DR_RANGE)
        fleet.append(e)
    ep = [0] * n
    s = [np.asarray(e.reset(np.random.default_rng((11, i, 0))), dtype=float)
         for i, e in enumerate(fleet)]
    tt = [0] * n
    for t, a in enumerate(acts):
        obs_t, rew_t, done_t, true_t = got[t]
        for i, e in enumerate(fleet):
            a_env = view.to_env_action(a[i])
            s2, term, _ = e.step(s[i], a_env)
            s2 = np.asarray(s2, dtype=float)
            r, _ = reward(s[i], view.clip_env_action(a_env), s2)
            assert rew_t[i] == np.float32(r)
            assert np.array_equal(true_t[i], _feat(s2))
            tt[i] += 1
            if term or tt[i] >= horizon:
                assert done_t[i] == 1
                ep[i] += 1
                tt[i] = 0
                s2 = np.asarray(e.reset(np.random.default_rng((11, i, ep[i]))), dtype=float)
            else:
                assert done_t[i] == 0
            s[i] = s2
            assert np.array_equal(obs_t[i], _feat(s2)), f"t={t} slot={i}"


def _count_constructions(monkeypatch):
    import bird.envs.base as B
    built = []
    real_init = B.EnvAdapter.__init__

    def counting(self, *args, **kwargs):
        built.append(type(self).__name__)
        return real_init(self, *args, **kwargs)
    monkeypatch.setattr(B.EnvAdapter, "__init__", counting)
    return built


def test_a_128_slot_view_constructs_no_adapter(monkeypatch):
    """One instance per slot would construct `num_envs` (default 128) adapters
    per seed, and on HumanoidBench a
    construction is a MuJoCo model plus an offscreen GL context. The bound is
    ZERO -- the view runs on `ctx.env` -- and it still gives every slot its own
    draw."""
    ctx = _ctx()
    env = ctx.env
    env.set_dr(DR_RANGE)
    width = int(np.asarray(env.obs_low).size)
    built = _count_constructions(monkeypatch)
    view = FT._VecEnvView(env, 128, T.compile_reward(REWARD, None), _feat, "none",
                          seed=3, obs_width=width)
    view.reset()
    a = np.zeros((128, int(np.asarray(env.action_low).size)), dtype=np.float32)
    for _ in range(3):
        view.step(a)
    assert built == [], f"a 128-slot view constructed {len(built)} adapters: {built[:3]}..."
    assert view.n == 128 and view.env is env
    assert len({view.episode_state(i)[1]["torque_scale"] for i in range(128)}) == 128


@needs_torch
@pytest.mark.slow
def test_training_at_num_envs_128_constructs_no_adapter(monkeypatch):
    """The same bound through the backend entry, at upstream's default width:
    a whole seed -- collection, updates, between-chunk evaluation on `ctx.env`
    -- constructs nothing `bird.py` did not. On the one-instance-per-slot
    head this counted 128."""
    ctx = _ctx(**{"train.hyperparameters": {**TINY, "num_envs": 128, "buffer_size": 1024},
                  "train.domain_randomization.mode": "fixed_human",
                  "train.domain_randomization.params": {k: list(v) for k, v in DR_RANGE.items()}})
    built = _count_constructions(monkeypatch)
    res = _train(ctx, "wide")
    assert res.trained, res.error
    assert built == [], f"training at num_envs=128 constructed {len(built)} adapters"
    assert ctx.env._dr_now == ctx.env._dr_nominal, "ctx.env left under a candidate's DR"


def test_the_fleet_is_reproducible_from_the_seed_under_dr():
    """Two fleets, one seed, DR ranges on both axes pendulum randomises
    (`action_noise` draws from the episode RNG every step): identical states
    after a full episode plus one auto-reset."""
    ctx = _ctx()
    width = int(np.asarray(ctx.env.obs_low).size)
    reward = T.compile_reward(REWARD, None)
    a = np.full((3, int(np.asarray(ctx.env.action_low).size)), 0.3, dtype=np.float32)

    def run():
        ctx.env.set_dr(DR_RANGE)
        view = FT._VecEnvView(ctx.env, 3, reward, _feat, "none",
                              seed=11, obs_width=width)
        view.reset()
        for _ in range(int(ctx.env.horizon) + 3):
            obs, rewards, _, _, _ = view.step(a)
        return obs, rewards
    o1, r1 = run()
    o2, r2 = run()
    assert np.array_equal(o1, o2) and np.array_equal(r1, r2)


@needs_torch
@pytest.mark.slow
def test_training_under_dr_is_reproducible_and_leaves_ctx_env_nominal():
    dr_cfg = {"train.domain_randomization.mode": "fixed_human",
              "train.domain_randomization.params": {k: list(v) for k, v in DR_RANGE.items()}}
    r1 = _train(_ctx(seed=5, **dr_cfg), "dr")
    r2 = _train(_ctx(seed=5, **dr_cfg), "dr")
    assert r1.trained, r1.error
    key = lambda r: [(p["fitness"], p["gt_return"], p["reward_return"])  # noqa: E731
                     for p in r.checkpoints]
    assert key(r1) == key(r2)


# ==========================================================================
# The reward sees the action the dynamics executed
# ==========================================================================


def test_the_reward_is_evaluated_on_the_executed_action_not_the_sampled_one():
    """Exploration noise takes the agent action past tanh's +/-1; the linear
    map takes it past the adapter's bound; the adapter clips it for the
    dynamics. The reward must see what the dynamics saw -- SB3 clips to the
    Box before `_Gym.step`, upstream's env computes its reward on the clamped
    action -- while the env call and the buffer keep the sampled value, as
    upstream steps and stores it (train.py:594). The reward must never get
    the phantom value."""
    ctx = _ctx()
    env = ctx.env
    norm_reward = T.compile_reward(
        "def compute_reward(state, action):\n"
        "    import numpy as np\n"
        "    return float(np.linalg.norm(action)), {}\n", None)
    width = int(np.asarray(env.obs_low).size)
    view = FT._VecEnvView(env, 1, norm_reward,
                          lambda s: np.asarray(s, dtype=np.float32), "none",
                          seed=1, obs_width=width)
    view.reset()
    s = view._s[0].copy()
    a_agent = np.full((1, int(np.asarray(env.action_low).size)), 1.7, dtype=np.float32)
    a_env = view.to_env_action(a_agent[0])
    hi = np.asarray(env.action_high, dtype=float)
    assert np.all(a_env > hi), "the probe must be out of bounds to mean anything"
    _, rewards, _, _, _ = view.step(a_agent)
    assert rewards[0] == pytest.approx(float(np.linalg.norm(hi)))          # 2.0, not 3.4
    assert rewards[0] != pytest.approx(float(np.linalg.norm(a_env)))
    # the executed action IS what the dynamics did: clipped and sampled step alike
    s2_clip, _, _ = env.step(s, np.clip(a_env, env.action_low, hi))
    s2_raw, _, _ = env.step(s, a_env)
    assert np.allclose(s2_clip, s2_raw) and np.allclose(view._s[0], s2_clip)
    assert (view.n_clipped, view.n_steps) == (1, 1)
    # in-bounds actions are untouched and not counted
    view.step(np.zeros_like(a_agent))
    assert (view.n_clipped, view.n_steps) == (1, 2)


@needs_torch
@pytest.mark.slow
def test_the_numpy_view_takes_a_tensor_identically():
    """`_VecEnvView.step` given a torch tensor is bit-identical to being given
    the pre-converted numpy array, including `n_clipped`.

    THE CONVERSION MOVED, AND THIS PINS THAT IT ONLY MOVED. The learner used
    to call `view.step(actions.float().cpu().numpy())`; the contract is now
    "the view takes agent actions as the learner holds them", so the device
    view can take a device tensor and do `jnp.from_dlpack` instead of a
    host round trip.
    The numpy view therefore does those same three ops itself, first thing.

    A relocation is the easiest change to get subtly wrong -- a different
    dtype, a different order, a copy where there was a view -- and the family
    of guarantees this belongs to is the one that says `parallel` must stay
    bit-identical to `sequential`. So this compares OUTPUTS, not the source.
    Two views from the same seed, one fed a tensor and one fed the array that
    tensor converts to, must agree on every returned array and on the clip
    counter.
    """
    import torch
    from bird.components.training import compile_reward

    def _fresh():
        ctx = _ctx()
        env = ctx.env
        return env, FT._VecEnvView(
            env, 2, compile_reward(REWARD, None),
            lambda s: np.asarray(s, dtype=np.float32),
            "none", seed=11,
            obs_width=int(np.asarray(env.obs_low).size))

    env_a, view_a = _fresh()
    env_b, view_b = _fresh()
    view_a.reset(); view_b.reset()

    # Deliberately outside [-1, 1] on one column so `clip_env_action` fires and
    # `n_clipped` is exercised rather than left at zero by a gentle input.
    acts = np.zeros((2, int(np.asarray(env_a.action_high).size)), dtype=np.float32)
    acts[0, 0] = 5.0

    # A **CUDA** TENSOR, AND THAT IS THE ENTIRE POINT OF THE TEST.
    # A cpu tensor proves nothing here: `to_env_action` does
    # `np.asarray(a_agent, dtype=float)`, which accepts a cpu torch tensor, so
    # the relocated `.float().cpu().numpy()` is a no-op on that input and the
    # test passes whether the conversion is present or not -- under mutation,
    # a green test that cannot fail. `np.asarray` on a CUDA tensor raises, so only a device tensor
    # exercises the line the relocation moved.
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU: a cpu tensor cannot exercise the conversion")
    tensor = torch.as_tensor(acts).cuda()

    out_tensor = view_a.step(tensor)
    out_array = view_b.step(acts)

    assert len(out_tensor) == len(out_array) == 5
    for name, x, y in zip(
            ("next_obs", "rewards", "dones", "time_outs", "true_next"),
            out_tensor, out_array):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype, f"{name}: dtype moved, {x.dtype} vs {y.dtype}"
        assert np.array_equal(x, y), f"{name}: values moved"
    assert view_a.n_clipped == view_b.n_clipped > 0, (
        "n_clipped must match AND be non-zero -- a test where nothing clipped "
        "would pass without exercising the counter the relocation touches")
    assert view_a.n_steps == view_b.n_steps


# --------------------------------------------------------------------------
# three seed-row fields that must measure rather than answer
# --------------------------------------------------------------------------


def test_the_compile_clock_times_the_first_call_of_each_region():
    """`jit_compile_s` must not read 0.0 on a run with minutes of compilation.

    MECHANISM: if `self.jit_compile_s = 0.0` in `_JaxVecEnvView.__init__`
    were the only assignment in the tree, the seed row would read it back
    through `getattr(view, "jit_compile_s", 0.0)` and get the initialiser.
    Its two siblings on the same object ARE accumulated (`reward_sync_s +=`,
    `n_clipped_reads += 1`), and a field that can only answer zero is worse
    than an absent one: the zero is quotable.

    STUBBED, NOT jax: the regions are timed through `_timed_first`, so a
    callable that sleeps a known interval exercises the real accumulator
    with no jax, no mujoco and no GPU. The jax-marked tests skip in the
    default CI job (no jax installed), so a test that needs a device to fail
    is a test that never runs in CI.

    ASSERTS >=, NEVER EQUALITY with a fixed figure. The compile time is a
    wall-clock measurement, so a test pinned to one number would fail when the
    number got better.
    """
    from bird.components.fasttd3 import _JaxVecEnvView

    view = object.__new__(_JaxVecEnvView)       # no jax, no adapter, no device
    view.jit_compile_s = 0.0
    view._jit_timed = set()
    view._jax = types.SimpleNamespace(
        block_until_ready=lambda x: x)          # the real one waits; here it is a no-op

    calls = []

    def slow(tag):
        time.sleep(0.05)
        calls.append(tag)
        return tag

    assert view._timed_first("step_batch", slow, "a") == "a"
    first = view.jit_compile_s
    assert first >= 0.05, f"the first call was not timed at all: {first}"

    # SECOND CALL OF THE SAME REGION IS FREE. If it were timed too, the field
    # would be "total wall in the jitted regions" -- a per-step cost wearing
    # a compile-time name, and 1e6 steps of it.
    view._timed_first("step_batch", slow, "b")
    assert view.jit_compile_s == first, "a repeat call was charged as a compile"

    # A DIFFERENT REGION IS ITS OWN COMPILE. A cold run can compile
    # `reset_batch` twice at two shapes, so regions that share a name lose
    # one of them silently.
    view._timed_first("reset_batch", slow, "c")
    assert view.jit_compile_s >= first + 0.05
    assert calls == ["a", "b", "c"]


def test_the_compile_clock_survives_a_failing_block_until_ready():
    """Provenance must not kill a training.

    `block_until_ready` is what makes the measurement real (jax dispatch is
    async, so an unblocked call times the dispatch and reports microseconds
    against a four-minute compile). If it raises, the call still has to
    return the value the caller wanted: the row would rather be timed short
    than have the run die inside its instrument.
    """
    from bird.components.fasttd3 import _JaxVecEnvView

    def boom(_x):
        raise RuntimeError("device went away")

    view = object.__new__(_JaxVecEnvView)
    view.jit_compile_s = 0.0
    view._jit_timed = set()
    view._jax = types.SimpleNamespace(block_until_ready=boom)

    assert view._timed_first("step_batch", lambda: "value") == "value"
    assert view.jit_compile_s >= 0.0
    assert "step_batch" in view._jit_timed


def test_the_utilisation_samples_are_spread_and_summarised():
    """One point sample must not read 0% on a run that used the GPU.

    ONE point sample, taken at the middle chunk and recorded under a name
    every reader takes as a summary, can land inside the compile window on
    a cold JIT backend -- the one interval where 0 is the true answer: a single
    sample inside the compile window reads 0% on a run that then sustains
    real utilisation through stepping.

    The scripted sequence below is that shape: two readings from the compile
    window, two from stepping. The max is the one that says a GPU was used.
    """
    from bird.components.fasttd3 import _UtilSampler, _GPU_UTIL_SAMPLES

    scripted = iter([0, 0, 55, 58])
    s = _UtilSampler(40, enabled=True, sampler=lambda: next(scripted))
    assert len(s.at) == _GPU_UTIL_SAMPLES, s.at
    assert 0 not in s.at, (
        "a sample at chunk 0 is the compile window on a cold backend, which "
        "is the reading that makes a single point sample a zero")
    for ck in range(40):
        s.maybe_sample(ck)

    row = s.row_fields()
    assert row["gpu_utilization_max_pct"] == 58, row
    assert row["gpu_utilization_mean_pct"] == 28.2, row
    assert row["gpu_utilization_n_samples"] == 4, row
    assert "gpu_utilization_point_pct" not in row, (
        "the singular field is back: one sample under a summary name is the "
        "defect, and keeping it beside the new three gives one row two "
        "answers to the same question")


def test_a_failed_utilisation_read_is_not_a_zero():
    """"could not tell" and "the GPU was idle" are different facts.

    `_gpu_utilization_percent` returns None on every failure path -- no
    nvidia-smi, a timeout, unparseable output -- and a None recorded as 0
    would tell a reader of the seed row "this GPU did nothing" about rows
    that never measured anything. The statistics are None and the count is
    0, which no reader can average by accident.
    """
    from bird.components.fasttd3 import _UtilSampler

    s = _UtilSampler(40, enabled=True, sampler=lambda: None)
    for ck in range(40):
        s.maybe_sample(ck)
    assert s.row_fields() == {"gpu_utilization_max_pct": None,
                              "gpu_utilization_mean_pct": None,
                              "gpu_utilization_n_samples": 0}

    # Off cuda nothing is sampled at all, and the row says so the same way.
    off = _UtilSampler(40, enabled=False, sampler=lambda: 99)
    for ck in range(40):
        off.maybe_sample(ck)
    assert off.row_fields()["gpu_utilization_n_samples"] == 0
    assert off.samples == []


def test_one_chunk_still_gets_a_sample_and_never_more_than_it_has():
    """The arithmetic at the edges, where an evenly-spaced rule usually breaks.

    A short training (`n_chunks` below `_GPU_UTIL_SAMPLES`) must not ask for
    more samples than it has chunks, and must not fall through to zero
    samples -- a short run's seed row still reports its utilisation.
    """
    from bird.components.fasttd3 import _UtilSampler

    one = _UtilSampler(1, enabled=True, sampler=lambda: 42)
    assert one.at == {0}, one.at          # the only chunk there is
    one.maybe_sample(0)
    assert one.row_fields()["gpu_utilization_n_samples"] == 1

    two = _UtilSampler(2, enabled=True, sampler=lambda: 42)
    assert len(two.at) <= 2 and two.at, two.at

    none = _UtilSampler(0, enabled=True, sampler=lambda: 42)
    assert none.at == set()
    assert none.row_fields()["gpu_utilization_n_samples"] == 0


def test_every_jitted_region_is_entered_through_the_compile_clock():
    """The wiring, which the unit tests above cannot reach.

    `_timed_first` is exercised directly by `test_the_compile_clock_times_
    the_first_call_of_each_region`, so a change that unwired the CALL SITES
    -- calling `self.env.step_batch(...)` again instead of through the clock
    -- would leave those tests green and put `jit_compile_s` back to 0.0 on
    every real run. That is the shape of the defect being guarded, so it
    needs its own guard.

    SOURCE-LEVEL, AND THAT IS A WEAKER INSTRUMENT, stated rather than
    glossed: the strong version constructs a view and reads the field, which
    needs jax, mujoco and a GPU, and the jax-marked tests skip in the default
    CI job (no jax installed) -- a guard that only runs on a GPU machine is a
    guard that is not running when this breaks. It is AST rather than `grep`:
    a substring check passes on the word appearing in a comment.
    """
    src = (REPO / "bird" / "components" / "fasttd3.py").read_text()
    cls = next(n for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.ClassDef) and n.name == "_JaxVecEnvView")

    #: The jitted regions, by the attribute the call goes through.
    REGIONS = {"step_batch", "reset_batch", "_reward_row", "_tail"}

    def attr_name(node):
        if isinstance(node, ast.Attribute):
            return node.attr
        return getattr(node, "id", None)

    # Every mention of a region as a CALLABLE: either `x.step_batch(...)`
    # (called directly -- the defect) or passed to `_timed_first` (correct).
    timed, direct = [], []
    for node in ast.walk(cls):
        if not isinstance(node, ast.Call):
            continue
        fname = attr_name(node.func)
        if fname == "_timed_first":
            # `_timed_first(name, fn, *args)`: the second argument is the region
            if len(node.args) >= 2:
                got = attr_name(node.args[1])
                if got in REGIONS:
                    timed.append(got)
        elif fname in REGIONS:
            direct.append((fname, node.lineno))

    assert not direct, (
        "these jitted regions are called directly rather than through "
        "`_timed_first`, so their compilation is not counted and "
        "`jit_compile_s` reads 0.0 on a run that compiled for minutes:\n  "
        + "\n  ".join(f"{n} at fasttd3.py:{ln}" for n, ln in direct))
    assert set(timed) == REGIONS, (
        f"only {sorted(set(timed))} go through the compile clock; "
        f"missing {sorted(REGIONS - set(timed))}")


def test_the_seed_row_takes_its_utilisation_fields_from_the_sampler():
    """The other half of the same wiring argument.

    `_UtilSampler` is unit-tested above; this is what stops the row being
    rebuilt inline from a single sample again. Same weakness and same reason
    as the test above -- the strong version needs a CUDA machine, and the
    defect it guards is invisible on every other machine for exactly that
    reason.
    """
    src = (REPO / "bird" / "components" / "fasttd3.py").read_text()
    tree = ast.parse(src)

    # The singular field must be gone from THIS module. AST, NOT A
    # SUBSTRING: the name may appear in this module's PROSE, and a text
    # search cannot tell an explanation from a write. What is forbidden is
    # the name as a dict KEY -- which is what a seed row is.
    keys = [k.value for d in ast.walk(tree) if isinstance(d, ast.Dict)
            for k in d.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    assert "gpu_utilization_point_pct" not in keys, (
        "fasttd3.py still writes the singular point sample as a row key; one "
        "row carrying both it and the max/mean/n triple gives a reader two "
        "answers to the same question")

    # And the three fields reach the row through the sampler, not inline.
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "row_fields"]
    assert calls, "nothing calls `_UtilSampler.row_fields`: the row is built by hand again"
