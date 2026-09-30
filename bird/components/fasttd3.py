"""`train.backend: fasttd3` -- FastTD3 (Seo et al., 2025), ported line-faithfully.

The source of truth is the released implementation vendored at
`refs/code/FastTD3` (younggyoseo/FastTD3 @ 229ed59); every design choice below
is pinned to a file:line in that tree, the way `configs/methods/*.yaml` pin paper
sections. The paper (arXiv:2505.22642)
is context; where paper and code could disagree, this port follows the CODE,
because "FastTD3" in practice means the released hyperparameters and update
loop -- the published HumanoidBench returns this repo's h1hand anchors cite
were produced by exactly that code.

What the algorithm is (fast_td3/train.py): TD3 with
  * a distributional critic -- two categorical Q heads over `num_atoms` atoms
    on [v_min, v_max], trained by cross-entropy against the C51 projection of
    the smoothed target (fast_td3.py:109-155);
  * clipped double Q as a PER-SAMPLE distribution pick: the projected target
    distribution of whichever head has the lower expected value (train.py:446-456,
    `use_cdq`);
  * massively parallel collection -- `num_envs` environments stepped together,
    one transition per env per iteration, `num_updates` gradient steps per
    iteration on batches of `batch_size` transitions (train.py:578-676);
  * per-env exploration noise: each env draws its own noise scale uniformly
    from [std_min, std_max] and re-draws it when its episode ends
    (fast_td3.py:288-315), so one policy explores at many temperatures at once;
  * an empirical observation normalizer updated online -- including on replayed
    batches, which is upstream's behaviour, not an accident here
    (train.py:589-591 and 645-655 call the same `normalize_obs` with
    `update=True`; only evaluation passes `update=False`);
  * n-step returns off a circular per-env buffer with a guard against sampling
    across the write head (fast_td3_utils.py:126-200);
  * AdamW with weight decay 0.1 and cosine LR annealed per ITERATION, not per
    gradient step (train.py:292-313, 745-747); `tau: 0.1` soft target updates
    after every gradient step (train.py:676).

Two network families, selected by `train.architecture` (the key the surrogate
backends record but cannot honour -- here it finally selects a network):
  mlp       fast_td3.py's Actor/Critic -- plain ReLU MLPs, tanh policy head
            initialised at `init_scale` (the paper's architecture).
  simba_v2  fast_td3_simbav2.py -- l2-normalised hyperspherical layers
            (SimbaV2), the variant upstream now recommends; its derived
            constructor constants follow train.py:249-270 exactly.
Any other value (the published configs pin citations like rda's `simba`) warns
and runs `mlp`, with the EXECUTED value recorded in `seed_metrics.architecture`.

Deviations from upstream, each deliberate and none of them algorithmic:
  * The env boundary. Upstream vectorises HumanoidBench with one SubprocVecEnv
    process per env (environments/humanoid_bench_env.py); BIRD adapters expose
    a functional `step(state, action)`, so `_VecEnvView` steps `num_envs`
    episode slots in slot order, in-process, on the ONE adapter `ctx.env`,
    putting each slot's own episode state back before each of its steps
    (`EnvAdapter.episode_state` / `restore_episode_state`). Same data flow
    (auto-reset, `time_outs`, true terminal observation), no process herd
    inside the candidate-parallelism fork, and no adapter constructed that
    `bird.py` did not. The restore is what makes the interleave honest: an
    episode's state is not all in the state array -- `reset()` leaves the
    domain-randomisation draw and the episode RNG on the instance (`_dr_now`,
    `_rng`) and `_step` reads them -- so without it every slot would run under
    whichever slot had reset LAST: a fleet would execute one DR draw per
    fleet-episode instead of `num_envs`, slot i's own `(seed, i, episode)`
    draw would never run, and the between-chunk evaluation's final reset
    would leak its draw into every in-flight training episode. One adapter
    INSTANCE per slot would also be correct, but on HumanoidBench that is 128
    MuJoCo models and 128 offscreen GL contexts per seed (`humanoid.py` opens
    the context at construction); the restore gives the same transitions, bit
    for bit, for zero constructions (`tests/test_fasttd3.py::
    test_one_adapter_with_per_slot_restore_matches_a_fleet_of_isolated_adapters`).
    One second-order caveat: storing the
    true terminal obs for TERMINATIONS too (upstream substitutes it only for
    truncations) is exact for the projection (bootstrap 0 spreads unit mass
    whatever `next_obs` says) but not for the obs normalizer, whose statistics
    see replayed `next_observations`.
  * Actions. Upstream hardcodes action bounds to [-1, 1] because HumanoidBench
    pre-normalises (humanoid_bench_env.py:55); BIRD envs vary, so the agent
    acts in [-1, 1] and `_VecEnvView` maps linearly to [action_low,
    action_high] at the boundary. The env is handed the sampled action
    unclipped, as upstream steps it (train.py:594; the MuJoCo-backed adapters
    clip for their own dynamics -- `humanoid.py`, `humanoid_hand.py`,
    `assistax.py`, `gym_mujoco.py`'s `action_clip`, or
    MuJoCo's `ctrlrange` -- while `EnvAdapter.step` itself passes the action
    through, and `toy.py` / `control.py`'s pendulum saturate the TORQUE rather
    than the command, so on the tester tier an out-of-range command reaches
    `_step` unclipped), and the buffer holds the sampled agent-space action, as
    upstream stores it. The candidate REWARD is called on RAW states and the
    ENV-SPACE action the dynamics EXECUTED -- the sampled action clipped to
    the adapter's bounds (`_VecEnvView.clip_env_action`) -- which is where
    upstream's own env reward is computed (the isaaclab/mtbench wrappers
    clamp before dynamics and reward) and where `_sb3_run._Gym.step` sits,
    since SB3 clips to the Box before it steps. Scoring the unclipped value
    would charge an effort term for torque the env never applied, on this
    backend only (the greedy evaluation policy ends in tanh, so
    `reward_return` and the judge always see in-bounds actions).
  * The reward. The reward this backend trains on replaces the env reward --
    that is the whole point of §3 -- and `train.reward_norm` applies to it at
    collection, as on every other backend.

    WHICH reward that is is a config choice, `train.reward_source`: the
    candidate's compiled reward by default, or under `reward_source: reference`
    the ENVIRONMENT's own shipped reward (`EnvAdapter.reference_reward`), which
    is the oracle arm every searched curve is read against. Every backend takes
    that reward from one dispatch, `training._training_reward`, so the channel
    is the same here as on the others. Upstream's own
    `reward_normalization` (the running-discounted-return scaler,
    fast_td3_utils.py:501-541) is a separate, faithful knob applied at sample
    time, off by default exactly as upstream defaults it.
  * `tensordict` is not a dependency; batches are plain dicts of tensors.
    Container, not math.
  * Not ported: multi-GPU (train_multigpu.py), MTBench multi-task heads,
    asymmetric/privileged observations (no BIRD adapter has them), the
    SimNorm/SEM heads (`sim_type`, a post-paper addition), wandb/rendering/
    checkpoint files (BIRD's artifact machinery owns those), and the
    `actor_detach` `from_module` trick (an inference-overhead optimisation;
    rollouts here run the live actor under `torch.no_grad()`).
  * `compile` covers the two update closures only; upstream also compiles the
    exploration policy and the normalizer under mode=None (train.py:539-541).
  * Checkpoint evaluation runs out-of-band on the raw adapter (the shared
    `_evaluate_policy`, like every backend); upstream's evaluate() reuses the
    TRAINING envs and resets them all afterwards (train.py:695-698), so its
    collection stream restarts at every eval interval and this port's does not.
  * `train.anchor.*` and `train.reference_policy` are sb3-backend features
    (they clone or step SB3 policy objects); a config that enables them on
    this backend trains without them, LOUDLY -- same shape as kl_clone on a
    non-PPO algorithm.
  * `train.reward_scaling` and `train.elite_constraint` (LaRes Eq. 3 and 4)
    are honoured here exactly as on sb3. Eq. 3 is PLANNED by stage 3 in the
    parent, once per candidate per round (`population.scaling_elite_moments`
    draws from `ctx.rng` and journals, which only the parent may do) and
    APPLIED here off `candidate.meta["reward_scaling"]` through
    `training._scaled_reward` before any slot sees the reward. Eq. 4 is
    attached to the actor and critic AdamW steps once `train.init` has loaded
    a policy, with the ROUND-START ELITE's stored parameters as its reference
    (`_fasttd3_attach_elite_constraint`, this blob format's mirror of
    `training._sb3_attach_elite_constraint`: the elite's blob is held in the
    live networks for exactly the duration of
    `population.attach_l2_to_optimisers`'s snapshot, then the seed's own
    parameters go back), so a resumed `shared_population` slice is pulled
    toward the elite and not toward its own previous slice. The seed row
    records both AS EXECUTED: `reward_scaling` (the plan's record, or
    `applied: False` with the reason; only under a non-`none` key),
    `elite_constraint_optimisers` and `elite_constraint_reference`. Without
    both keys honoured, `-c lares -s train.backend=fasttd3` would silently run
    the paper's own ablation under the paper's name.

torch is imported lazily inside the backend call, never at module import:
`registry.load_all()` imports this module for every config in the repo,
including `--dry-run` and `--validate-all` on machines with no torch.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import signal
import subprocess
import time
import traceback
from multiprocessing import connection as _mpc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# MODULE SCOPE AND SAFE THERE: `jax_base` imports no jax
# (`tests/test_jax_base.py` pins that `sys.modules` stays free of it), so this
# costs a machine without the jax extra nothing. The `isinstance` check that
# selects the device view has to be able to run everywhere a config loads.
from ..envs.jax_base import BatchedEnvAdapter

from ..config import reward_language
from ..registry import register
from ..types import Candidate, TrainResult
from . import training as _training
from .training import (
    MAX_CHECKPOINTS,
    MIN_CHECKPOINTS,
    CompiledReward,
    _POLICY_STORE,
    _POLICY_STORE_MAX_BYTES,
    _REPLAY_STORE,
    _REPLAY_STORE_MAX_BYTES,
    _PolicyBlob,
    _ReplaySlice,
    _RewardNorm,
    _ceiling_guard,
    _ceiling_wanted,
    _demo_ceiling,
    _demo_fraction_point,
    _dr_params,
    _error_router,
    _fork_seeds,
    _install_observation,
    _mean_components,
    _mean_curve,
    _n_replayed_rollouts,
    _n_store_rollouts,
    _pruner_for,
    _remember_code,
    _retain_replay_states,
    _rollout,
    _running_best,
    SEED_STRIDE,
    _note_seed_schedule,
    _seed_base,
    _seed_for,
    _seed_fork_workers,
    _stamp_seed_rows,
    _selection_for,
    _selection_reason,
    _single_thread_torch,
    _store,
    _summarise_ceiling,
    _training_init,
    _wants_replay,
    compile_reward,
    log,
)

#: Upstream defaults, verbatim from `refs/code/FastTD3/fast_td3/hyperparams.py`
#: `BaseArgs` (lines 11-136) -- upstream's own comment: "Default hyperparameters
#: -- specifically for HumanoidBench". Per-task overrides (H1HandPushArgs' wider
#: v support, HumanoidBenchArgs' 100k iterations, MuJoCoPlaygroundArgs' 1024
#: envs, ...) live in configs via `train.hyperparameters`, not here: get_args'
#: env-name dispatch is a launcher convenience, and reproducing it would couple
#: this backend to env ids the way no other backend is.
#:
#: Keys upstream owns that BIRD owns elsewhere are ABSENT, not defaulted:
#: `total_timesteps` is `train.env_steps` (in env interactions; iterations =
#: env_steps // num_envs), `seed` is `_seed_base`, `env_name` is
#: `problem.env_id`, `agent` is `train.architecture`, and eval/render/save
#: intervals are the checkpoint schedule every backend shares.
_FASTTD3_DEFAULTS: Dict[str, Any] = {
    "num_envs": 128,                  # hyperparams.py:31
    "critic_learning_rate": 3e-4,     # hyperparams.py:37
    "actor_learning_rate": 3e-4,      # hyperparams.py:39
    "critic_learning_rate_end": 3e-4,  # hyperparams.py:41 (== start: constant LR)
    "actor_learning_rate_end": 3e-4,  # hyperparams.py:43
    "buffer_size": 1024 * 50,         # hyperparams.py:45 -- PER ENV rows; the
                                      # buffer is (num_envs, buffer_size, ...)
    "num_steps": 1,                   # hyperparams.py:47 (n-step return length)
    "gamma": 0.99,                    # hyperparams.py:49
    "tau": 0.1,                       # hyperparams.py:51 -- yes, 0.1; TD3's
                                      # classic 0.005 is what FastTD3 moved OFF
    "batch_size": 32768,              # hyperparams.py:53
    "policy_noise": 0.001,            # hyperparams.py:55 (target smoothing)
    "std_min": 0.001,                 # hyperparams.py:57 (exploration floor)
    "std_max": 0.4,                   # hyperparams.py:59
    "learning_starts": 10,            # hyperparams.py:61 -- ITERATIONS, so
                                      # 10 * num_envs transitions
    "policy_frequency": 2,            # hyperparams.py:63 (delayed policy update)
    "noise_clip": 0.5,                # hyperparams.py:65
    "num_updates": 2,                 # hyperparams.py:67 (grad steps / iteration)
    "init_scale": 0.01,               # hyperparams.py:69 (mlp tanh head init)
    "num_atoms": 101,                 # hyperparams.py:71
    "v_min": -250.0,                  # hyperparams.py:73
    "v_max": 250.0,                   # hyperparams.py:75
    "critic_hidden_dim": 1024,        # hyperparams.py:77
    "actor_hidden_dim": 512,          # hyperparams.py:79
    "critic_num_blocks": 2,           # hyperparams.py:81 (simba_v2 only)
    "actor_num_blocks": 1,            # hyperparams.py:83 (simba_v2 only)
    "use_cdq": True,                  # hyperparams.py:85 (clipped double Q)
    "compile": None,                  # hyperparams.py:93 says True -- but True
                                      # there assumes the CUDA device upstream
                                      # requires ("No GPU available" is fatal,
                                      # train.py:89). None means "device is
                                      # cuda": on the repo-default cpu device an
                                      # inductor warmup per candidate-seed buys
                                      # nothing. An explicit true/false wins.
    "compile_mode": "reduce-overhead",  # hyperparams.py:95
    "obs_normalization": True,        # hyperparams.py:97
    "reward_normalization": False,    # hyperparams.py:99
    "use_grad_norm_clipping": False,  # hyperparams.py:101
    "max_grad_norm": 0.0,             # hyperparams.py:103
    "amp": True,                      # hyperparams.py:105 -- inert off cuda:
                                      # amp_enabled = amp AND cuda (train.py:58)
    "amp_dtype": "bf16",              # hyperparams.py:107
    "disable_bootstrap": False,       # hyperparams.py:109
    "weight_decay": 0.1,              # hyperparams.py:123 (AdamW, both nets)
    "torch_deterministic": True,      # hyperparams.py:17 (cudnn.deterministic)
    "device": "cpu",                  # BIRD's key, not upstream's `cuda` flag,
                                      # for `_sb3_algo_and_hyper`'s reasons:
                                      # cpu runs everywhere and a sweep should
                                      # not serialise behind the free GPUs.
                                      # "auto" = cuda if available.
}

#: Names a config may not supply because the harness owns them; same rule and
#: message shape as `_sb3_algo_and_hyper`'s reserved list.
_RESERVED = ("seed", "env_name", "agent", "total_timesteps", "cuda",
             "device_rank", "exp_name", "project", "use_wandb", "checkpoint_path",
             "num_eval_envs", "eval_interval", "render_interval", "save_interval",
             "measure_burnin")

_ARCHITECTURES = ("mlp", "simba_v2")


def _fasttd3_hyper(cfg: Any) -> Dict[str, Any]:
    """`train.hyperparameters` over `_FASTTD3_DEFAULTS`, upstream's names.

    Same two rules as `_learner_kwargs`: a known name genuinely changes the run,
    an unknown one is WARNED about rather than dropped in silence -- a swept
    name that reaches no learner makes `train.hyperparameter_search` a max over
    nothing. Reserved names are set by the harness and warned about too.
    """
    out = dict(_FASTTD3_DEFAULTS)
    hyper = cfg.get("train.hyperparameters") or {}
    unknown: List[str] = []
    for key, value in hyper.items():
        if key in _RESERVED:
            log.warning("train.hyperparameters: %r is set by the harness on "
                        "train.backend=fasttd3 (seed by train.seeds_per_candidate, "
                        "total_timesteps by train.env_steps, agent by "
                        "train.architecture) and was ignored", key)
        elif key in out:
            out[key] = value
        else:
            unknown.append(key)
    if unknown:
        log.warning("train.hyperparameters: %s is not a parameter of the fasttd3 "
                    "learner and will not affect this run (it accepts %s). A swept "
                    "name that reaches no learner makes train.hyperparameter_search "
                    "a max over nothing.", sorted(unknown), sorted(_FASTTD3_DEFAULTS))
    return out


# ==========================================================================
# The networks, buffer and normalizers -- defined lazily, cached
# ==========================================================================
#
# Same idiom as `training._mixed_buffer_class`: these subclass `torch.nn.Module`,
# so defining them needs torch imported, and this module is imported by
# `registry.load_all()` on machines that have none.

_TORCH_NS: Optional[Dict[str, Any]] = None


def _torch_classes() -> Dict[str, Any]:
    global _TORCH_NS
    if _TORCH_NS is not None:
        return _TORCH_NS

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _categorical_projection(qnet: Any, obs: Any, actions: Any, rewards: Any,
                                bootstrap: Any, discount: Any, q_support: Any) -> Any:
        """The C51 projection, verbatim from fast_td3.py:109-155 (the simba_v2
        copy at fast_td3_simbav2.py:267-313 is character-identical, which is why
        one function serves both)."""
        v_min, v_max, num_atoms = qnet.v_min, qnet.v_max, qnet.num_atoms
        delta_z = (v_max - v_min) / (num_atoms - 1)
        batch_size = rewards.shape[0]
        target_z = (rewards.unsqueeze(1)
                    + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support)
        target_z = target_z.clamp(v_min, v_max)
        b = (target_z - v_min) / delta_z
        low = torch.floor(b).long()
        up = torch.ceil(b).long()
        # When b lands exactly on an atom, floor == ceil and the two index_adds
        # below would each deposit mass scaled by 0 -- losing it. Upstream
        # shifts one bound off the atom (fast_td3.py:131-137). BOTH masks are
        # computed from the ORIGINAL floor index before either bound moves;
        # fusing them reads the decremented value and double-deposits at b == 1.
        is_int = low == up
        l_mask = is_int & (low > 0)
        u_mask = is_int & (low == 0)
        low = torch.where(l_mask, low - 1, low)
        up = torch.where(u_mask, up + 1, up)
        next_dist = F.softmax(qnet(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (torch.linspace(0, (batch_size - 1) * num_atoms, batch_size,
                                 device=q_support.device)
                  .unsqueeze(1).expand(batch_size, num_atoms).long())
        proj_dist.view(-1).index_add_(
            0, (low + offset).view(-1), (next_dist * (up.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(
            0, (up + offset).view(-1), (next_dist * (b - low.float())).view(-1))
        return proj_dist

    # -- base family (fast_td3.py, `agent: fasttd3`) -------------------------

    class DistributionalQNetwork(nn.Module):
        """fast_td3.py:59-108, `sim_type: ""` head (the paper's architecture)."""

        def __init__(self, n_obs: int, n_act: int, num_atoms: int, v_min: float,
                     v_max: float, hidden_dim: int, device: Any = None) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_obs + n_act, hidden_dim, device=device), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2, device=device), nn.ReLU())
            self.fc_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device), nn.ReLU(),
                nn.Linear(hidden_dim // 4, num_atoms, device=device))
            self.v_min, self.v_max, self.num_atoms = v_min, v_max, num_atoms

        def forward(self, obs: Any, actions: Any) -> Any:
            return self.fc_head(self.net(torch.cat([obs, actions], 1)))

        def projection(self, obs, actions, rewards, bootstrap, discount, q_support):
            return _categorical_projection(self, obs, actions, rewards, bootstrap,
                                           discount, q_support)

    class Critic(nn.Module):
        """fast_td3.py:157-236: two Q heads, one support."""

        def __init__(self, n_obs: int, n_act: int, num_atoms: int, v_min: float,
                     v_max: float, hidden_dim: int, device: Any = None) -> None:
            super().__init__()
            kw = dict(n_obs=n_obs, n_act=n_act, num_atoms=num_atoms, v_min=v_min,
                      v_max=v_max, hidden_dim=hidden_dim, device=device)
            self.qnet1 = DistributionalQNetwork(**kw)
            self.qnet2 = DistributionalQNetwork(**kw)
            self.register_buffer("q_support",
                                 torch.linspace(v_min, v_max, num_atoms, device=device))

        def forward(self, obs: Any, actions: Any) -> Any:
            return self.qnet1(obs, actions), self.qnet2(obs, actions)

        def projection(self, obs, actions, rewards, bootstrap, discount):
            return (self.qnet1.projection(obs, actions, rewards, bootstrap,
                                          discount, self.q_support),
                    self.qnet2.projection(obs, actions, rewards, bootstrap,
                                          discount, self.q_support))

        def get_value(self, probs: Any) -> Any:
            return torch.sum(probs * self.q_support, dim=1)

    class Actor(nn.Module):
        """fast_td3.py:238-315. Per-env exploration noise is the part that is
        FastTD3 rather than TD3: `noise_scales` is one scale per env, drawn
        U[std_min, std_max) at construction and re-drawn for an env whenever its
        episode ends, so the fleet explores at `num_envs` temperatures at once."""

        def __init__(self, n_obs: int, n_act: int, num_envs: int, init_scale: float,
                     hidden_dim: int, std_min: float, std_max: float,
                     device: Any = None) -> None:
            super().__init__()
            self.n_act = n_act
            self.net = nn.Sequential(
                nn.Linear(n_obs, hidden_dim, device=device), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2, device=device), nn.ReLU())
            self.fc_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device), nn.ReLU())
            self.fc_mu = nn.Sequential(
                nn.Linear(hidden_dim // 4, n_act, device=device), nn.Tanh())
            nn.init.normal_(self.fc_mu[0].weight, 0.0, init_scale)
            nn.init.constant_(self.fc_mu[0].bias, 0.0)
            noise_scales = (torch.rand(num_envs, 1, device=device)
                            * (std_max - std_min) + std_min)
            self.register_buffer("noise_scales", noise_scales)
            self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
            self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
            self.n_envs = num_envs

        def forward(self, obs: Any) -> Any:
            return self.fc_mu(self.fc_head(self.net(obs)))

        def explore(self, obs: Any, dones: Any = None, deterministic: bool = False) -> Any:
            if dones is not None and dones.sum() > 0:
                new_scales = (torch.rand(self.n_envs, 1, device=obs.device)
                              * (self.std_max - self.std_min) + self.std_min)
                dones_view = dones.view(-1, 1) > 0
                self.noise_scales.copy_(
                    torch.where(dones_view, new_scales, self.noise_scales))
            act = self(obs)
            if deterministic:
                return act
            return act + torch.randn_like(act) * self.noise_scales

    # -- simba_v2 family (fast_td3_simbav2.py, `agent: fasttd3_simbav2`) -----

    def l2normalize(tensor: Any, axis: int = -1, eps: float = 1e-8) -> Any:
        return tensor / (torch.linalg.norm(tensor, ord=2, dim=axis, keepdim=True) + eps)

    class Scaler(nn.Module):
        """fast_td3_simbav2.py:14-31."""

        def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0,
                     device: Any = None) -> None:
            super().__init__()
            self.scaler = nn.Parameter(torch.full((dim,), init * scale, device=device))
            self.forward_scaler = init / scale

        def forward(self, x: Any) -> Any:
            return self.scaler.to(x.dtype) * self.forward_scaler * x

    class HyperDense(nn.Module):
        """fast_td3_simbav2.py:34-45: no bias, orthogonal init."""

        def __init__(self, in_dim: int, hidden_dim: int, device: Any = None) -> None:
            super().__init__()
            self.w = nn.Linear(in_dim, hidden_dim, bias=False, device=device)
            nn.init.orthogonal_(self.w.weight, gain=1.0)

        def forward(self, x: Any) -> Any:
            return self.w(x)

    class HyperMLP(nn.Module):
        """fast_td3_simbav2.py:48-77."""

        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                     scaler_init: float, scaler_scale: float, eps: float = 1e-8,
                     device: Any = None) -> None:
            super().__init__()
            self.w1 = HyperDense(in_dim, hidden_dim, device=device)
            self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
            self.w2 = HyperDense(hidden_dim, out_dim, device=device)
            self.eps = eps

        def forward(self, x: Any) -> Any:
            x = self.w2(F.relu(self.scaler(self.w1(x))) + self.eps)
            return l2normalize(x, axis=-1)

    class HyperEmbedder(nn.Module):
        """fast_td3_simbav2.py:80-109: concat `c_shift`, normalise, embed."""

        def __init__(self, in_dim: int, hidden_dim: int, scaler_init: float,
                     scaler_scale: float, c_shift: float, device: Any = None) -> None:
            super().__init__()
            self.w = HyperDense(in_dim + 1, hidden_dim, device=device)
            self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
            self.c_shift = c_shift

        def forward(self, x: Any) -> Any:
            new_axis = torch.full((*x.shape[:-1], 1), self.c_shift,
                                  device=x.device, dtype=x.dtype)
            x = l2normalize(torch.cat([x, new_axis], dim=-1), axis=-1)
            return l2normalize(self.scaler(self.w(x)), axis=-1)

    class HyperLERPBlock(nn.Module):
        """fast_td3_simbav2.py:112-150: residual by learned interpolation."""

        def __init__(self, hidden_dim: int, scaler_init: float, scaler_scale: float,
                     alpha_init: float, alpha_scale: float, expansion: int = 4,
                     device: Any = None) -> None:
            super().__init__()
            self.mlp = HyperMLP(hidden_dim, hidden_dim * expansion, hidden_dim,
                                scaler_init / math.sqrt(expansion),
                                scaler_scale / math.sqrt(expansion), device=device)
            self.alpha_scaler = Scaler(hidden_dim, alpha_init, alpha_scale,
                                       device=device)

        def forward(self, x: Any) -> Any:
            residual = x
            x = residual + self.alpha_scaler(self.mlp(x) - residual)
            return l2normalize(x, axis=-1)

    class HyperTanhPolicy(nn.Module):
        """fast_td3_simbav2.py:153-177."""

        def __init__(self, hidden_dim: int, action_dim: int, scaler_init: float,
                     scaler_scale: float, device: Any = None) -> None:
            super().__init__()
            self.mean_w1 = HyperDense(hidden_dim, hidden_dim, device=device)
            self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale,
                                      device=device)
            self.mean_w2 = HyperDense(hidden_dim, action_dim, device=device)
            self.mean_bias = nn.Parameter(torch.zeros(action_dim, device=device))

        def forward(self, x: Any) -> Any:
            mean = self.mean_w2(self.mean_scaler(self.mean_w1(x)))
            return torch.tanh(mean + self.mean_bias.to(mean.dtype))

    class HyperCategoricalValue(nn.Module):
        """fast_td3_simbav2.py:180-203."""

        def __init__(self, hidden_dim: int, num_bins: int, scaler_init: float,
                     scaler_scale: float, device: Any = None) -> None:
            super().__init__()
            self.w1 = HyperDense(hidden_dim, hidden_dim, device=device)
            self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
            self.w2 = HyperDense(hidden_dim, num_bins, device=device)
            self.bias = nn.Parameter(torch.zeros(num_bins, device=device))

        def forward(self, x: Any) -> Any:
            logits = self.w2(self.scaler(self.w1(x)))
            return logits + self.bias.to(logits.dtype)

    class DistributionalQNetworkSimba(nn.Module):
        """fast_td3_simbav2.py:206-313."""

        def __init__(self, n_obs: int, n_act: int, num_atoms: int, v_min: float,
                     v_max: float, hidden_dim: int, scaler_init: float,
                     scaler_scale: float, alpha_init: float, alpha_scale: float,
                     num_blocks: int, c_shift: float, expansion: int,
                     device: Any = None) -> None:
            super().__init__()
            self.embedder = HyperEmbedder(n_obs + n_act, hidden_dim, scaler_init,
                                          scaler_scale, c_shift, device=device)
            self.encoder = nn.Sequential(*[
                HyperLERPBlock(hidden_dim, scaler_init, scaler_scale, alpha_init,
                               alpha_scale, expansion, device=device)
                for _ in range(num_blocks)])
            self.predictor = HyperCategoricalValue(hidden_dim, num_atoms, 1.0, 1.0,
                                                   device=device)
            self.v_min, self.v_max, self.num_atoms = v_min, v_max, num_atoms

        def forward(self, obs: Any, actions: Any) -> Any:
            x = torch.cat([obs, actions], 1)
            return self.predictor(self.encoder(self.embedder(x)))

        def projection(self, obs, actions, rewards, bootstrap, discount, q_support):
            return _categorical_projection(self, obs, actions, rewards, bootstrap,
                                           discount, q_support)

    class CriticSimba(nn.Module):
        """fast_td3_simbav2.py:316-408."""

        def __init__(self, n_obs: int, n_act: int, num_atoms: int, v_min: float,
                     v_max: float, hidden_dim: int, scaler_init: float,
                     scaler_scale: float, alpha_init: float, alpha_scale: float,
                     num_blocks: int, c_shift: float, expansion: int,
                     device: Any = None) -> None:
            super().__init__()
            kw = dict(n_obs=n_obs, n_act=n_act, num_atoms=num_atoms, v_min=v_min,
                      v_max=v_max, hidden_dim=hidden_dim, scaler_init=scaler_init,
                      scaler_scale=scaler_scale, alpha_init=alpha_init,
                      alpha_scale=alpha_scale, num_blocks=num_blocks,
                      c_shift=c_shift, expansion=expansion, device=device)
            self.qnet1 = DistributionalQNetworkSimba(**kw)
            self.qnet2 = DistributionalQNetworkSimba(**kw)
            self.register_buffer("q_support",
                                 torch.linspace(v_min, v_max, num_atoms, device=device))

        def forward(self, obs: Any, actions: Any) -> Any:
            return self.qnet1(obs, actions), self.qnet2(obs, actions)

        def projection(self, obs, actions, rewards, bootstrap, discount):
            return (self.qnet1.projection(obs, actions, rewards, bootstrap,
                                          discount, self.q_support),
                    self.qnet2.projection(obs, actions, rewards, bootstrap,
                                          discount, self.q_support))

        def get_value(self, probs: Any) -> Any:
            return torch.sum(probs * self.q_support, dim=1)

    class ActorSimba(nn.Module):
        """fast_td3_simbav2.py:410-479; explore is identical to the base Actor's
        (fast_td3_simbav2.py:481-503)."""

        def __init__(self, n_obs: int, n_act: int, num_envs: int, hidden_dim: int,
                     scaler_init: float, scaler_scale: float, alpha_init: float,
                     alpha_scale: float, expansion: int, c_shift: float,
                     num_blocks: int, std_min: float, std_max: float,
                     device: Any = None) -> None:
            super().__init__()
            self.n_act = n_act
            self.embedder = HyperEmbedder(n_obs, hidden_dim, scaler_init,
                                          scaler_scale, c_shift, device=device)
            self.encoder = nn.Sequential(*[
                HyperLERPBlock(hidden_dim, scaler_init, scaler_scale, alpha_init,
                               alpha_scale, expansion, device=device)
                for _ in range(num_blocks)])
            self.predictor = HyperTanhPolicy(hidden_dim, n_act, 1.0, 1.0,
                                             device=device)
            noise_scales = (torch.rand(num_envs, 1, device=device)
                            * (std_max - std_min) + std_min)
            self.register_buffer("noise_scales", noise_scales)
            self.register_buffer("std_min", torch.as_tensor(std_min, device=device))
            self.register_buffer("std_max", torch.as_tensor(std_max, device=device))
            self.n_envs = num_envs

        def forward(self, obs: Any) -> Any:
            return self.predictor(self.encoder(self.embedder(obs)))

        explore = Actor.explore

    # -- utilities (fast_td3_utils.py) ---------------------------------------

    class EmpiricalNormalization(nn.Module):
        """fast_td3_utils.py:404-498 minus the torch.distributed branch (this
        backend is single-process by construction -- candidate parallelism forks
        around it, never inside it). Chan et al. parallel-variance update,
        verbatim; `forward` UPDATES whenever the module is in training mode and
        `update=True`, which is how replayed batches feed the statistics too."""

        def __init__(self, shape: Any, device: Any, eps: float = 1e-2,
                     until: Optional[int] = None) -> None:
            super().__init__()
            self.eps = eps
            self.until = until
            self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
            self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
            self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
            self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

        @torch.no_grad()
        def forward(self, x: Any, center: bool = True, update: bool = True) -> Any:
            if x.shape[1:] != self._mean.shape[1:]:
                raise ValueError(
                    f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}")
            if self.training and update:
                self.update(x)
            if center:
                return (x - self._mean) / (self._std + self.eps)
            return x / (self._std + self.eps)

        def update(self, x: Any) -> None:
            if self.until is not None and self.count >= self.until:
                return
            batch_size = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
            new_count = self.count + batch_size
            delta = batch_mean - self._mean
            self._mean.copy_(self._mean + delta * (batch_size / new_count))
            delta2 = batch_mean - self._mean
            m_a = self._var * self.count
            m_b = batch_var * batch_size
            m2 = m_a + m_b + delta2.pow(2) * (self.count * batch_size / new_count)
            self._var.copy_(m2 / new_count)
            self._std.copy_(self._var.sqrt())
            self.count.copy_(new_count)

    class RewardNormalizer(nn.Module):
        """fast_td3_utils.py:501-541: scale rewards by the running std of the
        DISCOUNTED RETURN, floored so the largest |G| seen maps to at most
        `g_max`. Applied at sample time; the buffer stores raw rewards."""

        def __init__(self, gamma: float, device: Any, g_max: float = 10.0,
                     epsilon: float = 1e-8) -> None:
            super().__init__()
            self.register_buffer("g_running", torch.zeros(1, device=device))
            self.register_buffer("g_r_max", torch.zeros(1, device=device))
            self.g_rms = EmpiricalNormalization(shape=1, device=device)
            self.gamma = gamma
            self.g_max = g_max
            self.epsilon = epsilon

        def update_stats(self, rewards: Any, dones: Any) -> None:
            self.g_running = self.gamma * (1 - dones) * self.g_running + rewards
            self.g_rms.update(self.g_running.view(-1, 1))
            local_max = torch.max(torch.abs(self.g_running))
            self.g_r_max = torch.maximum(self.g_r_max, local_max)

        def forward(self, rewards: Any) -> Any:
            var_denominator = self.g_rms._std.squeeze(0)[0] + self.epsilon
            min_required_denominator = self.g_r_max / self.g_max
            return rewards / torch.maximum(var_denominator, min_required_denominator)

    class SimpleReplayBuffer(nn.Module):
        """fast_td3_utils.py:12-401, symmetric observations only (no BIRD
        adapter has privileged critic observations, so the asymmetric and
        playground branches are not ported). Circular per-env buffer; `sample`
        draws `batch_size` indices PER ENV and returns `n_env * batch_size`
        flat rows, with the n-step return, its per-sample effective length and
        the cross-episode guard exactly as upstream (including temporarily
        marking the write head truncated while sampling a full buffer)."""

        def __init__(self, n_env: int, buffer_size: int, n_obs: int, n_act: int,
                     n_steps: int = 1, gamma: float = 0.99, device: Any = None) -> None:
            super().__init__()
            self.n_env, self.buffer_size = n_env, buffer_size
            self.n_obs, self.n_act = n_obs, n_act
            self.gamma, self.n_steps, self.device = gamma, n_steps, device
            self.observations = torch.zeros((n_env, buffer_size, n_obs),
                                            device=device, dtype=torch.float)
            self.actions = torch.zeros((n_env, buffer_size, n_act),
                                       device=device, dtype=torch.float)
            self.rewards = torch.zeros((n_env, buffer_size), device=device,
                                       dtype=torch.float)
            self.dones = torch.zeros((n_env, buffer_size), device=device,
                                     dtype=torch.long)
            self.truncations = torch.zeros((n_env, buffer_size), device=device,
                                           dtype=torch.long)
            self.next_observations = torch.zeros((n_env, buffer_size, n_obs),
                                                 device=device, dtype=torch.float)
            self.ptr = 0

        @torch.no_grad()
        def extend(self, observations: Any, actions: Any, rewards: Any,
                   dones: Any, truncations: Any, next_observations: Any) -> None:
            ptr = self.ptr % self.buffer_size
            self.observations[:, ptr] = observations
            self.actions[:, ptr] = actions
            self.rewards[:, ptr] = rewards
            self.dones[:, ptr] = dones
            self.truncations[:, ptr] = truncations
            self.next_observations[:, ptr] = next_observations
            self.ptr += 1

        @torch.no_grad()
        def sample(self, batch_size: int) -> Dict[str, Any]:
            if self.n_steps == 1:
                indices = torch.randint(0, min(self.buffer_size, self.ptr),
                                        (self.n_env, batch_size), device=self.device)
                obs_idx = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
                act_idx = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
                observations = torch.gather(self.observations, 1, obs_idx).reshape(
                    self.n_env * batch_size, self.n_obs)
                next_observations = torch.gather(self.next_observations, 1, obs_idx
                                                 ).reshape(self.n_env * batch_size, self.n_obs)
                actions = torch.gather(self.actions, 1, act_idx).reshape(
                    self.n_env * batch_size, self.n_act)
                rewards = torch.gather(self.rewards, 1, indices).reshape(-1)
                dones = torch.gather(self.dones, 1, indices).reshape(-1)
                truncations = torch.gather(self.truncations, 1, indices).reshape(-1)
                effective_n_steps = torch.ones_like(dones)
            else:
                if self.ptr >= self.buffer_size:
                    # A full circular buffer has no episode boundary at the
                    # write head, so an n-step window can straddle it. Upstream
                    # marks the row before the head truncated for the duration
                    # of the sample and restores it (fast_td3_utils.py:190-200,
                    # 396-399, after SB3's #1622 fix).
                    current_pos = self.ptr % self.buffer_size
                    curr_truncations = self.truncations[:, current_pos - 1].clone()
                    self.truncations[:, current_pos - 1] = torch.logical_not(
                        self.dones[:, current_pos - 1]).long()
                    indices = torch.randint(0, self.buffer_size,
                                            (self.n_env, batch_size), device=self.device)
                else:
                    max_start_idx = max(1, self.ptr - self.n_steps + 1)
                    indices = torch.randint(0, max_start_idx,
                                            (self.n_env, batch_size), device=self.device)
                obs_idx = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
                act_idx = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
                observations = torch.gather(self.observations, 1, obs_idx).reshape(
                    self.n_env * batch_size, self.n_obs)
                actions = torch.gather(self.actions, 1, act_idx).reshape(
                    self.n_env * batch_size, self.n_act)
                seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
                all_indices = (indices.unsqueeze(-1) + seq_offsets) % self.buffer_size
                all_rewards = torch.gather(
                    self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
                all_dones = torch.gather(
                    self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
                all_truncations = torch.gather(
                    self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                    1, all_indices)
                # Zero every reward after the first done; the first row of the
                # window is never masked (fast_td3_utils.py:270-278).
                all_dones_shifted = torch.cat(
                    [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2)
                done_masks = torch.cumprod(1.0 - all_dones_shifted.float(), dim=2)
                effective_n_steps = done_masks.sum(2)
                discounts = torch.pow(self.gamma,
                                      torch.arange(self.n_steps, device=self.device))
                n_step_rewards = (all_rewards * done_masks
                                  * discounts.view(1, 1, -1)).sum(dim=2)
                first_done = torch.argmax((all_dones > 0).float(), dim=2)
                first_trunc = torch.argmax((all_truncations > 0).float(), dim=2)
                no_dones = all_dones.sum(dim=2) == 0
                no_truncs = all_truncations.sum(dim=2) == 0
                first_done = torch.where(no_dones, self.n_steps - 1, first_done)
                first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)
                final_indices = torch.minimum(first_done, first_trunc)
                final_next_obs_idx = torch.gather(
                    all_indices, 2, final_indices.unsqueeze(-1)).squeeze(-1)
                next_observations = self.next_observations.gather(
                    1, final_next_obs_idx.unsqueeze(-1).expand(-1, -1, self.n_obs)
                ).reshape(self.n_env * batch_size, self.n_obs)
                rewards = n_step_rewards.reshape(-1)
                dones = self.dones.gather(1, final_next_obs_idx).reshape(-1)
                truncations = self.truncations.gather(1, final_next_obs_idx).reshape(-1)
                effective_n_steps = effective_n_steps.reshape(-1)
                if self.ptr >= self.buffer_size:
                    self.truncations[:, current_pos - 1] = curr_truncations
            return {"observations": observations, "actions": actions,
                    "rewards": rewards, "dones": dones, "truncations": truncations,
                    "next_observations": next_observations,
                    "effective_n_steps": effective_n_steps}

    ns = {"torch": torch, "nn": nn, "F": F,
          "Actor": Actor, "Critic": Critic,
          "ActorSimba": ActorSimba, "CriticSimba": CriticSimba,
          "EmpiricalNormalization": EmpiricalNormalization,
          "RewardNormalizer": RewardNormalizer,
          "SimpleReplayBuffer": SimpleReplayBuffer}
    _TORCH_NS = ns
    return ns


# ==========================================================================
# The env boundary
# ==========================================================================


def _snapshot_agent(agent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """`train.checkpoint_selection`'s snapshot: the four state dicts that define
    the policy and its continuation (optimizer state deliberately not -- the
    restore happens after the last gradient step, so there is nothing left to
    optimise; `_snapshot_sb3` carries optimizers only because
    `set_parameters(exact_match=True)` demands them). Module-level, like
    `_snapshot_sb3` and `_Learner.restore`, so the checkpoint-selection sweep
    can inject a failure into this backend the way it does into the others.
    """
    out: Dict[str, Any] = {}
    for name in ("actor", "qnet", "qnet_target", "obs_normalizer"):
        mod = agent[name]
        if mod is not None:
            out[name] = {k: v.detach().clone()
                         for k, v in mod.state_dict().items()}
    return out


def _restore_agent(agent: Dict[str, Any], snap: Optional[Dict[str, Any]]) -> bool:
    if snap is None:
        return False
    try:
        for name, sd in snap.items():
            agent[name].load_state_dict(sd)
        return True
    except Exception as exc:  # noqa: BLE001 - a failed restore ships the final weights
        log.warning("fasttd3: could not restore the selected checkpoint "
                    "(%s: %s); shipping the final one", type(exc).__name__, exc)
        return False


def _resolve_arch(cfg: Any) -> str:
    """The architecture this backend EXECUTES for `train.architecture`.

    `train.architecture` is free-form and the published configs pin it as a
    CITATION (rda: `simba`, limen: `rnn_ppo`); this backend implements two
    values and runs the paper's own network for anything else, loudly, with
    the executed value in `seed_metrics.architecture` -- the same
    warn-and-record shape as the surrogates' binning warning. ONE resolver,
    shared by the training loop and by `fasttd3_policy_from_blob`, so a
    policy is rebuilt under the architecture it was trained under and not
    under the citation.
    """
    arch = str(cfg.get("train.architecture", "mlp") or "mlp")
    if arch not in _ARCHITECTURES:
        log.warning("train.architecture=%r is not implemented by "
                    "train.backend=fasttd3 (it implements %s); training the "
                    "base FastTD3 mlp, and the seed rows record that",
                    arch, list(_ARCHITECTURES))
        arch = "mlp"
    return arch


def _action_affine(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    """`(center, half)` of the adapter's action box: agent [-1, 1] maps to
    `center + a * half`, linearly, WITHOUT clamping (`_VecEnvView.to_env_action`).
    One definition, used by the view and by the rebuilt policy, so a policy
    rolled out after training maps its actions exactly as the loop did."""
    low = np.asarray(env.action_low, dtype=float).ravel()
    high = np.asarray(env.action_high, dtype=float).ravel()
    center = (high + low) / 2.0
    half = (high - low) / 2.0
    return center, np.where(half > 0, half, 1.0)


def _build_networks(ns: Dict[str, Any], arch: str, hyper: Dict[str, Any], n_obs: int,
                    n_act: int, num_envs: int, device: Any) -> Tuple[Any, Any, Any, Any]:
    """Actor, critic, target critic and observation normaliser for `arch`, UNSEEDED
    and UNSYNCED: `_build_agent` seeds torch before calling this and copies the
    critic into its target after; `fasttd3_policy_from_blob` calls it and then
    loads a stored state dict over it. One constructor, so the network a blob is
    loaded into is the network it was saved from."""
    if arch == "mlp":
        actor = ns["Actor"](n_obs=n_obs, n_act=n_act, num_envs=num_envs,
                            init_scale=float(hyper["init_scale"]),
                            hidden_dim=int(hyper["actor_hidden_dim"]),
                            std_min=float(hyper["std_min"]),
                            std_max=float(hyper["std_max"]), device=device)
        critic_kw = dict(n_obs=n_obs, n_act=n_act,
                         num_atoms=int(hyper["num_atoms"]),
                         v_min=float(hyper["v_min"]), v_max=float(hyper["v_max"]),
                         hidden_dim=int(hyper["critic_hidden_dim"]), device=device)
        qnet, qnet_target = ns["Critic"](**critic_kw), ns["Critic"](**critic_kw)
    else:
        a_hid, c_hid = int(hyper["actor_hidden_dim"]), int(hyper["critic_hidden_dim"])
        a_blocks, c_blocks = int(hyper["actor_num_blocks"]), int(hyper["critic_num_blocks"])
        actor = ns["ActorSimba"](
            n_obs=n_obs, n_act=n_act, num_envs=num_envs, hidden_dim=a_hid,
            scaler_init=math.sqrt(2.0 / a_hid), scaler_scale=math.sqrt(2.0 / a_hid),
            alpha_init=1.0 / (a_blocks + 1), alpha_scale=1.0 / math.sqrt(a_hid),
            expansion=4, c_shift=3.0, num_blocks=a_blocks,
            std_min=float(hyper["std_min"]), std_max=float(hyper["std_max"]),
            device=device)
        critic_kw = dict(
            n_obs=n_obs, n_act=n_act, num_atoms=int(hyper["num_atoms"]),
            v_min=float(hyper["v_min"]), v_max=float(hyper["v_max"]),
            hidden_dim=c_hid, scaler_init=math.sqrt(2.0 / c_hid),
            scaler_scale=math.sqrt(2.0 / c_hid), alpha_init=1.0 / (c_blocks + 1),
            alpha_scale=1.0 / math.sqrt(c_hid), num_blocks=c_blocks,
            c_shift=3.0, expansion=4, device=device)
        qnet = ns["CriticSimba"](**critic_kw)
        qnet_target = ns["CriticSimba"](**critic_kw)
    if bool(hyper["obs_normalization"]):
        obs_normalizer = ns["EmpiricalNormalization"](shape=n_obs, device=device)
    else:
        obs_normalizer = None
    return actor, qnet, qnet_target, obs_normalizer


def _greedy_policy(ns: Dict[str, Any], actor: Any, obs_normalizer: Any,
                   features: Callable[[Any], np.ndarray],
                   to_env_action: Callable[[np.ndarray], np.ndarray],
                   device: Any) -> Callable[[np.ndarray], np.ndarray]:
    """THE action path: raw state -> feature -> normalise WITHOUT updating
    (train.py:346-349) -> actor -> the env's action box. Driven by the training
    loop through `_policy_fn` and by `fasttd3_policy_from_blob` for a rebuilt
    policy, so the rolled-out policy IS the trained action path and not a second
    implementation that could differ in the normaliser or the squash invisibly.
    `_single_thread_torch` at the call sites keeps the forked schedule's parent
    predicts at the children's thread count."""
    torch = ns["torch"]

    def policy(s: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(features(s), device=device).unsqueeze(0)
            if obs_normalizer is not None:
                x = obs_normalizer(x, update=False)
            a = actor(x).squeeze(0).float().cpu().numpy()
        return to_env_action(a)

    return policy


class _VecEnvView:
    """`num_envs` episode slots on ONE BIRD adapter -- the in-process stand-in
    for upstream's `HumanoidBenchEnv` (environments/humanoid_bench_env.py).

    One adapter, `num_envs` episode states. An adapter's `step(state, action)`
    is state-passing but reads two things `reset()` left on the INSTANCE --
    the episode RNG and the domain-randomisation draw (`_rng`, `_dr_now`; see
    `EnvAdapter.episode_state`) -- so interleaving `num_envs` episodes on one
    adapter with nothing else made every slot run under the most recent
    reset's draw and RNG stream: one DR draw per fleet-episode instead of
    `num_envs`, slot i's own `(seed, i, episode)` draw never executed, and the
    between-chunk `_evaluate_policy` on the same adapter leaking its last draw
    into every in-flight training episode. Each slot therefore keeps the
    `episode_state()` blob its reset produced, and `step` puts it back
    (`restore_episode_state`) before stepping that slot and re-captures it
    after. A slot's episode sees exactly the draw its reset made and exactly
    its own RNG stream, whatever was reset in between -- another slot, or
    `ctx.env` by an evaluation -- and the transitions are bit-identical to a
    fleet of `num_envs` separately constructed adapters stepped in lockstep
    (pinned on pendulum, where `action_noise` draws from the episode RNG every
    step). The alternative -- one adapter INSTANCE per slot -- is the same
    correctness for `num_envs` constructions, and on HumanoidBench a
    construction is a MuJoCo model plus an offscreen GL context (`humanoid.py`
    calls `gym.make(render_mode="rgb_array")` in `__init__`); 128 of those per
    seed, per candidate, per worker is a cost this design does not pay.
    `tests/test_fasttd3.py` pins that building and stepping a 128-slot view
    constructs no adapter at all.

    Slots are stepped in slot order every iteration, each through the adapter's
    functional `step(state, action)`, so the data flow is upstream's exactly:
    one transition per env per iteration, auto-reset on episode end, a
    `time_outs` flag distinct from termination, and the TRUE terminal
    observation in the row a bootstrapped target will read (upstream
    substitutes it only for truncations, humanoid_bench_env.py:107-113;
    substituting it for terminations too is arithmetically identical, because
    a terminal row's bootstrap is 0 and the projection then spreads exactly
    unit mass whatever `next_obs` says).

    The candidate reward is computed on RAW states and the ENV-SPACE action the
    dynamics EXECUTED -- the sampled action clipped to the adapter's bounds
    (`clip_env_action`) -- then `train.reward_norm`ed: the boundary
    `_sb3_run._Gym.step` actually sits at, since SB3 clips to the Box before
    it steps (and like there, a reward that raises aborts the seed; verify's
    dynamic checks are the net for that). The ENV is handed the sampled action
    as upstream steps it (train.py:594; the MuJoCo-backed adapters clip it for
    their dynamics, the toy and classic-control ones saturate the torque -- see
    the module docstring), and the buffer holds the SAMPLED action, as
    upstream stores it: the feature space (`phi(s)` under a co-designed
    observation, the raw state otherwise) and agent-space actions as drawn --
    exploration noise takes them past [-1, 1] and that is upstream's buffer
    too. `n_clipped / n_steps` is the fraction of transitions whose executed
    action differed from the sampled one (`action_clip_fraction` on the seed
    row). One `_RewardNorm` is shared by all slots: under `running_std` the
    statistics are of the merged collection stream, which is the honest
    analogue of upstream's single normaliser over a vector of envs.

    Episode RNG streams are `default_rng((seed, slot, episode))` -- distinct by
    construction, reproducible from the config seed, no stride arithmetic to
    alias.
    """

    def __init__(self, env: Any, num_envs: int, reward: CompiledReward,
                 features: Callable[[Any], np.ndarray], norm_mode: str,
                 seed: int, obs_width: int, workers: int = 1,
                 action_repeat: int = 1) -> None:
        self.env = env
        self.n = max(1, int(num_envs))
        self._reward = reward
        self._feat = features
        self._width = int(obs_width)
        self._norm = _RewardNorm(norm_mode)
        self._seed = int(seed)
        self.horizon = int(env.horizon)
        # ACTION REPEAT -- `train.backend: simba_v2`'s only claim on this view,
        # and 1 (every backend before it) is the identity: the loop in
        # `_step_slot` runs exactly once, the same call in the same order on
        # the same input, so `fasttd3` is bit-identical to before this
        # parameter existed (`tests/test_simba_v2.py::
        # test_action_repeat_1_is_bit_identical_to_no_repeat`).
        #
        # WHOSE KNOB IT IS. SimbaV2 applies it as a gym wrapper OUTSIDE the
        # TimeLimit (`refs/code/SimbaV2/scale_rl/envs/__init__.py:84-88`, whose
        # own comment is "limit max_steps before action_repeat"), so
        # `max_episode_steps` counts SIMULATOR steps and `env.horizon` here
        # means the same thing it always did. It lives on the view rather than
        # in the backend's loop because a repeat must be PER SLOT: upstream
        # breaks out of its repeat on `terminated or truncated`
        # (`envs/wrappers/repeat_action.py:20-21`) and each of its envs is its
        # own process, so a fleet-wide break would be a different function.
        # `_step_slot` is also what a forked slot worker calls, so the workers
        # inherit the repeat with no change to the pipe protocol.
        self.action_repeat = max(1, int(action_repeat))
        low = np.asarray(env.action_low, dtype=float).ravel()
        high = np.asarray(env.action_high, dtype=float).ravel()
        self._low, self._high = low, high
        self.n_clipped = 0
        self.n_steps = 0
        # A zero-width action dimension maps every agent action to its one
        # value; 1.0 keeps the inverse map (used by the replay export) finite.
        # `_action_affine`: one definition, shared with the rebuilt policy.
        self._act_center, self._act_half = _action_affine(env)
        self._episode = [0] * self.n
        self._t = [0] * self.n
        self._s: List[np.ndarray] = [np.zeros(1)] * self.n
        # Slot i's `episode_state()` blob -- what its reset left on the adapter
        # instance -- put back before each of its steps. `None` until reset.
        self._ep: List[Any] = [None] * self.n
        # `train.n_parallel_envs` > 1: the slots are stepped by forked worker
        # processes, `workers` of them, each on ITS OWN copy of the adapter
        # (a fork is a copy; MuJoCo model + data included). A SCHEDULE, not a
        # science knob: every slot keeps its own `(seed, slot, episode)` RNG
        # and DR draw exactly as above, the workers return RAW rewards and the
        # parent applies `_RewardNorm` in slot order, so the transitions and
        # the normaliser's statistics are bit-identical to `workers=1`
        # (tests/test_fasttd3.py pins it). Measured: the sequential loop costs
        # 0.12 s (Assistax) / 0.70 s (h1hand) per 128-slot iteration against
        # 0.057 s for the compiled FastTD3 update on an H200 GPU, so without
        # this the GPU idles 2-12x. Forked HERE, at construction, which `_run_one_seed`
        # places BEFORE the agent is built: a child forked from a process that
        # already touched CUDA must never touch it, and these never do (numpy
        # and the adapter only), but the fork itself is still cleanest before.
        self._pool: Optional[_SlotPool] = None
        if int(workers) > 1 and self.n > 1:
            self._pool = _SlotPool(self, min(int(workers), self.n))

    def close(self) -> None:
        """Stop the slot workers, if any. Idempotent; called at the end of a
        seed and again by `__del__`, and a pipe EOF makes a child exit on its
        own if the parent dies without either."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.close()

    def __del__(self) -> None:  # pragma: no cover - best effort
        with contextlib.suppress(Exception):
            self.close()

    def to_env_action(self, a_agent: np.ndarray) -> np.ndarray:
        """[-1, 1] -> [action_low, action_high], linearly, WITHOUT clamping:
        upstream steps the env with the noisy action as sampled (train.py:594)
        and stores it; only the target-smoothing noise is clamped. This is the
        env's and the buffer's action; the REWARD gets `clip_env_action` of it."""
        return self._act_center + np.asarray(a_agent, dtype=float) * self._act_half

    def clip_env_action(self, a_env: np.ndarray) -> np.ndarray:
        """The action the dynamics EXECUTE: `a_env` clipped to the adapter's
        bounds, which is what every adapter does inside `_step` (a motor cannot
        be commanded past saturation) and what SB3 hands `_Gym.step`. The
        candidate reward is evaluated on this, never on the sampled value."""
        return np.clip(np.asarray(a_env, dtype=float), self._low, self._high)

    def to_agent_action(self, a_env: np.ndarray) -> np.ndarray:
        return (np.asarray(a_env, dtype=float) - self._act_center) / self._act_half

    def _reset_slot(self, i: int) -> None:
        rng = np.random.default_rng((self._seed, i, self._episode[i]))
        self._episode[i] += 1
        self._t[i] = 0
        self._s[i] = np.asarray(self.env.reset(rng), dtype=float)
        self._ep[i] = self.env.episode_state()

    def episode_state(self, i: int) -> Any:
        """Slot i's adapter-side episode state (RNG + DR draw) as of its last
        reset or step. Read by tests; `_reset_slot` and `step` are the writers."""
        return self._ep[i]

    def reset(self) -> np.ndarray:
        if self._pool is not None:
            self._pool.reset()
        else:
            for i in range(self.n):
                self._reset_slot(i)
        return self.observations()

    def observations(self) -> np.ndarray:
        return np.stack([np.asarray(self._feat(s), dtype=np.float32)
                         for s in self._s])

    def step(self, actions_agent: Any
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(next_obs, rewards, dones, time_outs, true_next_obs), all length n.

        TAKES AGENT ACTIONS AS THE LEARNER HOLDS THEM -- a torch tensor on the
        learner's device -- and does its own conversion to numpy here. Each
        view owns its conversion because the device view wants those actions
        on the GPU, and a host round trip at the call site would defeat its
        zero-copy path. Here it is `actions.float().cpu().numpy()`, and
        `test_the_numpy_view_takes_a_tensor_identically` pins that the outputs
        and `n_clipped` are bit-identical to passing the pre-converted array.

        `dones` is termination OR truncation, `time_outs` truncation-and-not-
        termination -- the same pair SB3's VecEnv hands upstream's wrapper, so
        `bootstrap = truncations | ~dones` (train.py:422-425) means here what
        it means there.
        """
        if hasattr(actions_agent, "detach"):   # a torch tensor
            actions_agent = actions_agent.float().cpu().numpy()
        rewards = np.zeros(self.n, dtype=np.float32)
        dones = np.zeros(self.n, dtype=np.int64)
        time_outs = np.zeros(self.n, dtype=np.int64)
        true_next = np.zeros((self.n, self._width), dtype=np.float32)
        if self._pool is not None:
            raw = self._pool.step(np.asarray(actions_agent), dones, time_outs, true_next)
        else:
            raw = np.zeros(self.n, dtype=np.float64)
            for i in range(self.n):
                raw[i], dones[i], time_outs[i] = self._step_slot(
                    i, actions_agent[i], true_next[i])
        # The normaliser runs in the PARENT, in slot order, on raw rewards --
        # the one place the worker count could otherwise reach the numerics.
        for i in range(self.n):
            rewards[i] = self._norm(float(raw[i]))
        return self.observations(), rewards, dones, time_outs, true_next

    def _step_slot(self, i: int, a_agent: np.ndarray, true_next_row: np.ndarray
                   ) -> Tuple[float, int, int]:
        """One slot, ONE AGENT DECISION, on THIS process's adapter. Returns the
        RAW candidate reward (normalised by the caller, in slot order), `done`
        and `time_out`; writes the true next observation into `true_next_row`.

        One decision is `self.action_repeat` simulator steps holding the action
        (default 1, the identity). The candidate reward is called on EVERY
        simulator step and SUMMED over the repeat, which is what upstream's
        wrapper does with the env reward
        (`refs/code/SimbaV2/scale_rl/envs/wrappers/repeat_action.py:10-23 RepeatAction.step`),
        and the repeat BREAKS on termination or truncation, so a slot that
        ends mid-repeat is reset once and its next decision starts from the
        fresh episode rather than part-way through it. `n_steps`/`n_clipped`
        count simulator steps, so `action_clip_fraction` keeps its meaning.
        """
        env = self.env
        a_env = self.to_env_action(a_agent)
        a_exec = self.clip_env_action(a_env)
        clipped = not np.array_equal(a_exec, a_env)
        total = 0.0
        for _rep in range(self.action_repeat):
            s = self._s[i]
            # Slot i's own RNG stream and DR draw, whatever was reset since
            # its last step (another slot, or `ctx.env` by an evaluation).
            env.restore_episode_state(self._ep[i])
            s2, terminated, _info = env.step(s, a_env)
            self._ep[i] = env.episode_state()
            s2 = np.asarray(s2, dtype=float)
            self.n_steps += 1
            if clipped:
                self.n_clipped += 1
            r, _cv = self._reward(s, a_exec, s2)
            total += float(r)
            self._t[i] += 1
            truncated = (not terminated) and self._t[i] >= self.horizon
            true_next_row[:] = np.asarray(self._feat(s2), dtype=np.float32)
            if terminated or truncated:
                self._reset_slot(i)
                return total, 1, int(truncated)
            self._s[i] = s2
        return total, 0, 0


#: Re-exported so the view and the tests have one name for it; the function
#: itself lives in `bird.xla_env`, which is torch-free because `bird.py`'s env
#: construction site imports it on every run of every tier.
from bird.xla_env import (  # noqa: E402
    set_xla_determinism_flags, applied_in_time as _xla_applied_in_time)


class _JaxVecEnvView:
    """`_VecEnvView`'s device twin: `num_envs` rows stepped inside one jitted
    region on a `BatchedEnvAdapter`, with no slot workers and no pipes.

    SELECTED BY THE ADAPTER'S CAPABILITY, NEVER BY A CONFIG FLAG. The
    caller builds this when `isinstance(env, BatchedEnvAdapter)`; the science
    knob is `problem.env_id`, and a config key saying "go fast" would be a
    second way to say something the env already says -- and a way for the two
    to disagree.

    WHAT IT KEEPS IDENTICAL TO `_VecEnvView`, because the tier is an execution
    choice and must not move a number:

      * `to_env_action` is the unclamped affine map and `clip_env_action` is
        what the dynamics execute AND what the reward sees -- the same split,
        for the same reason (upstream steps the env with the noisy action as
        sampled and only the reward is evaluated on the executed one).
      * `n_clipped` counts rows whose executed action differed from the
        sampled one, so `action_clip_fraction` means what it always meant.
      * auto-reset fires on `terminated | truncated`, with `truncated` being
        `(not terminated) and t >= horizon` -- `_step_slot`'s rule, per row.
      * `_RewardNorm` normalises the scalar fed to the learner only, and the
        recorded component values stay raw.

    WHAT IT DOES NOT KEEP: `episode_state(i)` is a JAX key rather than a numpy
    generator plus a DR draw, because `reset_batch` takes a key by design
    and there is no mutable generator to fork. Anything reading the old shape
    off this view gets a key and should say so rather than duck-type it.

    WHAT IS NOT TESTED YET, AND WHY, so a green suite is not read as
    coverage it does not have. `tests/test_jax_vec_view.py` exercises this
    class against a TEST DOUBLE: the action maps, the per-row auto-reset,
    truncation-not-termination, the clip accounting, both DLPack directions
    and the compilation cache across two processes. A double cannot settle:

      * that a training on this view LEARNS the same as the numpy view on the
        same env and seed -- an acceptance run on a real batched environment
        is the only thing that can catch a semantic difference the interface
        hides;
      * whether the compilation cache HITS ACROSS TRAININGS of one search, or
        only across seeds of one candidate -- the reward's cache key is
        candidate-dependent by construction (the two-jit split below), but
        the physics half is only assumed to be candidate-independent until
        measured on a real multi-training search;
      * `clip_env_action` and `_RewardNorm` producing IDENTICAL numbers here
        and on the n=1 bridge the trace path uses. Shared code makes it
        likely, not certain, and if they diverge the claimed-vs-measured
        trace silently measures a different clipping than the training did.

    NOT ON THE TRACE PATH. `_rollout` -- and therefore claimed-vs-measured,
    checkpoint evals, screens and video -- runs on the RAW ADAPTER through
    `BatchedEnvAdapter._step`'s n=1 bridge (the `_rollout(env, ...)` call
    sites pass `env`, never a view). This class is
    only ever the training loop's env. That separation is why the trace keeps
    working here rather than silently producing nothing.
    """

    #: Both directions are DLPack and both are measured at construction rather
    #: than asserted: `dlpack_zero_copy` goes onto the seed row per direction
    #: (`obs`, `actions`), because a field that says "zero copy" on belief
    #: rather than measurement is exactly the kind that misleads.
    def __init__(self, env: Any, num_envs: int, reward: CompiledReward,
                 features: Callable[[Any], Any], norm_mode: str, seed: int,
                 obs_width: int) -> None:
        import torch

        # TWO FRAMEWORKS, ONE GPU. XLA preallocates ~75% of the device by
        # default and torch then cannot allocate; these two settings are read
        # by XLA at BACKEND INITIALISATION, which is the first jax operation
        # in the process and not the import. Set before `import jax` here, so
        # a process that has never touched jax gets them.
        #
        # ONCE PER PROCESS, AND THAT IS THE TRAP. In a process that builds
        # several views, whichever builds the first view fixes the backend for
        # every later one, and a per-view setting would silently apply to the
        # first view only. So the effective values are READ BACK
        # onto the seed row rather than assumed from what we asked for.
        # AUTOTUNING MAKES THIS TIER IRREPRODUCIBLE, measured over six
        # processes: with default flags the BATCHED step (n>=4) returns one of
        # TWO distinct results per launch, spread 2.03e-04, stable within a
        # process; with `--xla_gpu_autotune_level=0` all six launches agree
        # exactly, spread 0.0. `--xla_gpu_deterministic_ops=true` and a
        # `highest` matmul precision were NOT needed. So the flag is what makes
        # an acceptance test an equality against a rerun rather than a
        # tolerance.
        #
        # WHAT THE FLAG DOES NOT BUY: it picks a branch, it does not compute a new value -- the
        # autotune-off result EQUALS one of the two default results, and there
        # is no evidence it is the branch closer to CPU MuJoCo (a separate
        # measurement against the 7.66e-4 CPU divergence baseline). A residual
        # 1.79e-07 between n=1 and n=4 survives the flag and is real: a
        # singleton and a vectorised batch compile to genuinely different
        # fused kernels. The tier is reproducible PER SHAPE, not across
        # shapes, so the n=1 bridge is NOT a bit-oracle for this view.
        #
        # APPENDED, NOT `setdefault`, and that distinction is the bug this
        # avoids. `XLA_FLAGS` is one space-separated string holding every XLA
        # flag: `setdefault` on it is a no-op the moment anything else has set
        # a single unrelated flag, so the process would silently keep
        # autotuning while `xla_env` recorded a value we never applied. An
        # operator who has already pinned an autotune level keeps theirs.
        # SETTING IT HERE IS TOO LATE IN A REAL RUN. A jax adapter calls
        # `mjx.put_model` in its own `__init__`, which initialises the XLA
        # backend BEFORE this view exists -- so an `XLA_FLAGS` append here
        # would change nothing while `xla_env` faithfully reported the flag as
        # present.
        #
        # So: `set_xla_determinism_flags()` is called at PROCESS ENTRY and
        # refuses once jax is imported, and all the view does is READ BACK
        # what the process actually has and record whether it was applied in
        # time. A flag recorded as present but applied late is worse than an
        # absent one, because it is quoted in a reproducibility claim.
        # REPORT THE PROCESS'S VERDICT, NOT THIS CALL'S. By the time a view
        # exists the adapter has imported jax, so this call is always "too
        # late" and a row reporting it would read False on every real run,
        # even on a correctly wired process. `applied_in_time()` is what the
        # entry point recorded.
        set_xla_determinism_flags(strict=False)
        _applied_in_time = _xla_applied_in_time()

        _xla_asked = {
            "XLA_PYTHON_CLIENT_PREALLOCATE":
                os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
            "XLA_PYTHON_CLIENT_MEM_FRACTION":
                os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6"),
            # READ BACK, never the string we just built: another view in this
            # process may have fixed the backend first, in which
            # case our append arrived too late to matter and the seed row must
            # say what the process actually has.
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            # WHETHER THE FLAG COULD STILL BITE. False means jax was already
            # imported when it was set, so the backend may have been
            # initialised without it and the string above is what the
            # environment says rather than what XLA is doing. Any equality
            # claim from a row with this False is void.
            "autotune_flag_applied_before_jax_import": bool(_applied_in_time),
        }

        import jax                       # local: this class only exists on the
        import jax.numpy as jnp          # jax family, whose extra pins it

        self._jax, self._jnp, self._torch = jax, jnp, torch
        self.xla_env = dict(_xla_asked)

        # THE INSTALLED VERSION, NOT THE PIN. The tier pins jax <= 0.8.0 but
        # an environment can hold a newer one, so "which jax produced this
        # number" is a question the seed row must answer for itself -- a pin in `pyproject.toml` says
        # what was asked for, and the whole reason `xla_env` is read back
        # rather than asserted is that those are different facts. Kernel
        # selection and the autotune flag's behaviour both live in XLA, which
        # ships inside jaxlib, so a reproducibility claim that does not name
        # the version is scoped to nothing.
        self.jax_version = str(getattr(jax, "__version__", ""))

        # JAX'S PERSISTENT COMPILATION CACHE, NODE-LOCAL, for the same reason
        # the inductor cache is node-local: the compile is per (model, batch
        # shape) PER PROCESS, and every new process pays it again. A large model
        # can take minutes per shape; a small model compiles in seconds and
        # will NOT show this.
        #
        # `num_envs` is fixed within a search, so the shape is stable and the
        # cache is reusable across that search's trainings.
        #
        # Node-local /tmp, not a shared network filesystem: 44.6 ms vs 1.4 ms
        # per small file was the measured difference for the inductor cache,
        # and a compilation cache is thousands of small files. Keyed by
        # nothing but the content, so two processes on one node share it safely.
        cache_dir = os.environ.get("BIRD_JAX_CACHE_DIR") or os.path.join(
            "/tmp", os.environ.get("USER", "bird"), "jax-compilation-cache")
        # THE MINIMUM COMPILE TIME IS SET EXPLICITLY, NOT INHERITED. jax
        # caches only compiles slower than this threshold, and the default is
        # not ours to rely on -- measured here: with the default, a cheap
        # compile produces NEITHER a hit NOR a miss, so `jax_cache` reads
        # `{hits: 0, misses: 0}` and the cache looks broken when it is merely
        # declining to store something trivial. Stating the value means the
        # row records the policy that was in force rather than whatever the
        # installed jax happened to default to.
        #
        # 1.0 s keeps the disk for compiles worth storing (Assistax's ~170 s
        # shapes) and skips the ones that are cheaper to redo than to load.
        # `{hits: 0, misses: 0}` alongside a real `dir` is therefore a
        # MEANINGFUL reading -- "nothing here was slow enough to cache" --
        # and not a failure.
        min_s = float(os.environ.get("BIRD_JAX_CACHE_MIN_COMPILE_S", "1.0"))
        try:
            os.makedirs(cache_dir, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", min_s)
            self.jax_cache_dir = cache_dir
        except Exception as exc:  # noqa: BLE001 - a cache is an optimisation
            self.jax_cache_dir = None    # and must never fail a training
            log.warning("jax compilation cache unavailable at %s: %s",
                        cache_dir, exc)

        # HIT OR MISS, FROM JAX'S OWN COUNTERS rather than from timing.
        # Measured on jax 0.8.0: a cold process emits
        # `/jax/compilation_cache/cache_misses` and a warm one emits
        # `/jax/compilation_cache/cache_hits`, so the two are distinguishable
        # without inferring anything from how long a compile took -- which is
        # the inference that would quietly become circular the moment anyone
        # used it to justify the cache.
        self.jax_cache = {"dir": self.jax_cache_dir, "min_compile_s": min_s,
                          "hits": 0, "misses": 0}

        def _on_event(name: str, **_kw: Any) -> None:
            if name.endswith("/cache_hits"):
                self.jax_cache["hits"] += 1
            elif name.endswith("/cache_misses"):
                self.jax_cache["misses"] += 1

        try:
            jax.monitoring.register_event_listener(_on_event)
        except Exception:  # noqa: BLE001 - instrumentation is never fatal
            pass

        # REFUSE, DO NOT WARN, IF THE TWO FRAMEWORKS ARE ON DIFFERENT DEVICES.
        # A DLPack hand-off between two
        # devices does not fail loudly -- it either copies or produces a
        # tensor the learner then reads from the wrong GPU.
        #
        # COMPARED ON SOMETHING THAT MEANS THE SAME ON BOTH SIDES. A torch
        # ordinal and a jax `Device.id` are different numbering schemes and
        # need not agree under `CUDA_VISIBLE_DEVICES`; comparing them directly
        # is the "two things that looked alike" mistake. Where both expose a
        # PCI bus id that is the comparison; otherwise both indices are
        # recorded and the check is skipped rather than faked.
        jdev = jax.devices()[0]
        if jdev.platform != "gpu":
            raise RuntimeError(
                f"the jax family needs a GPU backend; jax reports "
                f"platform={jdev.platform!r}. A jax env on cpu is a refusal, "
                f"not a downgrade.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "the jax family needs torch on cuda; torch.cuda.is_available() "
                "is False while jax has a GPU. The learner would train on cpu "
                "while the env stepped on the GPU.")
        t_bus = getattr(torch.cuda.get_device_properties(
            torch.cuda.current_device()), "pci_bus_id", None)
        j_bus = getattr(jdev, "pci_bus_id", None) or getattr(
            getattr(jdev, "client", None), "pci_bus_id", None)
        # AN EXPLICIT `status`, NOT A NULL TO BE INTERPRETED. `jax_pci: null`
        # inside an otherwise populated dict reads at a glance as "checked,
        # and equal" -- the field is there, three of its four values are real,
        # and the absence is one key deep. A null that means "not measured"
        # sitting where a measurement goes is easily read as evidence.
        #
        # So the dict says which of the two it is, in a key whose value
        # cannot be mistaken for a measurement, and carries the reason.
        _agreed = bool(t_bus) and bool(j_bus)
        self.device_check = {"torch_index": int(torch.cuda.current_device()),
                             "jax_id": int(getattr(jdev, "id", -1)),
                             "torch_pci": t_bus, "jax_pci": j_bus,
                             "status": "compared" if _agreed else "skipped",
                             "skipped_because": "" if _agreed else (
                                 "no PCI bus id exposed by "
                                 + (" and ".join(
                                     [n for n, v in (("torch", t_bus),
                                                     ("jax", j_bus)) if not v])
                                    or "either framework")
                                 + " on this device; the two frameworks' "
                                   "indices are recorded but NOT compared, so "
                                   "this row does not assert they agree")}
        if t_bus and j_bus and str(t_bus) != str(j_bus):
            raise RuntimeError(
                f"torch is on {t_bus} and jax is on {j_bus}: a DLPack hand-off "
                f"across devices copies or reads the wrong GPU. Pin both with "
                f"CUDA_VISIBLE_DEVICES.")
        self.env = env
        self.n = max(1, int(num_envs))
        self._reward = reward
        self._feat = features
        self._width = int(obs_width)
        self._norm = _RewardNorm(norm_mode)
        self._seed = int(seed)
        self.horizon = int(env.horizon)
        # NO `self.n_clipped = 0` HERE: it is a property backed by a device
        # counter (see below), and assigning it would raise. `_VecEnvView`
        # has a plain attribute; this one deliberately does not, so that the
        # sync happens where the value is read rather than every step.
        self.n_steps = 0

        # COMPILE TIMING, INITIALISED HERE AND NOT BESIDE ITS SIBLINGS BELOW.
        # `reward_sync_s` and `n_clipped_reads` are declared after the
        # construction `reset_batch` a few dozen lines down, which is fine
        # for them and would be fatal here: THAT RESET IS ITSELF ONE OF THE
        # COMPILES (~120 s for `reset_batch` on a cold Assistax run), so a
        # counter initialised after it would miss the largest single
        # contribution and still read plausibly.
        #
        # The row reads `getattr(view, "jit_compile_s", 0.0)`, so a counter
        # that is declared but never accumulated would report 0.0 for a run
        # with minutes of compilation. A field that exists to record compile
        # time and can only answer zero is worse than an absent one, because
        # the zero is quotable.
        self.jit_compile_s = 0.0
        #: Which jitted regions have been through `_timed_first` already.
        #: Names, not the callables: the adapter's bound methods are
        #: re-looked-up per call and an identity set would time every step.
        self._jit_timed: set = set()

        # THE ROW FUNCTION, NOT THE COMPILED WRAPPER. `CompiledReward.__call__` goes
        # through `_unpack`, which casts to a host float and kills the trace;
        # `row()` is the same contract with the numbers left on device, the
        # TRAINER's argument binding applied and the weighted total computed
        # in jnp. Taking the un-jitted row rather than `traced()` lets this
        # class fuse reward, phi and the auto-reset into ONE jitted region and
        # time the compile it actually pays.
        # VMAPPED HERE, not called with the batch. `row()` is ONE row by
        # contract -- that is what makes it composable -- and handing it the
        # `(n, obs)` batch silently does the wrong thing for any reward that
        # indexes its state, which is most of them: a reward reading
        # `next_state[0]` would get the whole first ROW rather than that row's
        # first coordinate (`tests/test_jax_vec_view.py` pins this).
        # `traced()` exists for callers that want the jit too;
        # this class vmaps and jits for itself so it can fuse the reward with
        # phi and the auto-reset into one region and time its own compile.
        # TWO JITTED FUNCTIONS, NOT ONE FUSED REGION, and the reason is the
        # compilation cache rather than kernel count. Fusing the reward into
        # the same `jit` as the physics makes the CACHE KEY
        # candidate-dependent, so the expensive physics compile -- ~170 s per
        # shape on Assistax -- would be paid once per candidate, eight times
        # an iteration at K=8, instead of once per (model, num_envs) for the
        # life of the cache. Full fusion buys one kernel launch; the split
        # buys the whole compile cache, which is larger by orders of magnitude
        # on the suites that hurt.
        #
        #   candidate-INdependent: the adapter's own jitted `step_batch`,
        #       plus `_tail` below (done/truncation/auto-reset). Cached
        #       across every training of the same model and num_envs.
        #   candidate-DEpendent:   the reward, jitted separately. Seconds to
        #       compile, missed once per candidate, which is correct.
        #
        # Expected hit pattern on a multi-training search, to be checked against
        # `jax_cache` rather than assumed: physics hits from the second
        # training onward, reward misses once per candidate.
        self._reward_row = jax.jit(jax.vmap(reward.row()))
        self._tail = jax.jit(self._tail_impl)

        low = np.asarray(env.action_low, dtype=float)
        high = np.asarray(env.action_high, dtype=float)
        self._low, self._high = low, high
        self._act_center = (high + low) / 2.0
        half = (high - low) / 2.0
        self._act_half = np.where(half > 0, half, 1.0)
        # The same three constants on device, so the action maps do not leave
        # the GPU to be applied.
        self._d_low, self._d_high = jnp.asarray(low), jnp.asarray(high)
        self._d_center = jnp.asarray(self._act_center)
        self._d_half = jnp.asarray(self._act_half)

        self._key = jax.random.key(self._seed)
        self._key, sub = jax.random.split(self._key)
        self._s = self._timed_first("reset_batch", env.reset_batch, sub, self.n)
        # THE PER-ROW PHYSICS PYTREE. `physics_rows` is the adapter's
        # single host-dict-to-pytree place, so the view and the n=1 bridge
        # cannot disagree about what a draw means. `None` for a family that
        # declares no DR axes, which is most of them and is not an error.
        #
        # CALLED DIRECTLY, NOT THROUGH `getattr`. An adapter that does not
        # implement it should raise here at construction, loudly, rather than
        # have the view quietly fall back to no-DR -- that fallback is the
        # bug this whole argument exists to prevent.
        self._physics = env.physics_rows(self.n)
        self._t = jnp.zeros((self.n,), dtype=jnp.int32)
        self._episode = jnp.zeros((self.n,), dtype=jnp.int32)
        # (`jit_compile_s` is initialised at the top of `__init__`, before
        # the construction `reset_batch` that is one of the compiles it
        # counts -- see the comment there.)
        self.reward_sync_s = 0.0
        self.dlpack_zero_copy = {"obs": None, "actions": None}
        self._d_clipped = jnp.zeros((), dtype=jnp.int32)
        #: Rows the adapter reported as nonfinite, accumulated ON DEVICE for
        #: the same reason as the clip count: a per-step host sync to read a
        #: number nobody reads per step is a price paid for nothing.
        self._d_nonfinite = jnp.zeros((), dtype=jnp.int32)
        self.n_clipped_reads = 0

    def _timed_first(self, name: str, fn: Callable[..., Any],
                     *args: Any, **kw: Any) -> Any:
        """Call `fn`, adding the wall of its FIRST call to `jit_compile_s`.

        WHAT THE FIELD MEASURES, written down because the name is shorter
        than the truth: the wall of the first call of each jitted region --
        XLA compilation plus that one call's execution. On the tier this
        exists for the compile is 120-215 s per region and the execution is
        milliseconds (measured on cold Assistax runs), so it is the compile
        to within one step. It is NOT a pure compile clock and must not be
        quoted as one.

        BLOCKING IS THE MEASUREMENT, not a precaution. jax dispatch is
        asynchronous: without `block_until_ready` this would time the
        dispatch and report microseconds against a four-minute compile --
        a field that lies quietly, which is the thing being fixed. The sync
        is paid ONCE per region and never per step, which is why it can live
        here and why `reward_sync_s` next door counts a different cost.

        NOT WIRED TO `jax.monitoring`'s CACHE COUNTERS, which is the obvious
        alternative and is wrong here: the listener is registered during this
        backend's setup, and the verify compile happens before that, so the
        counters see five misses where the cache directory gains six entries
        (measured). A field built on them would under-report by
        exactly one compile and look right doing it.

        The clock is `perf_counter`, and failures are swallowed: a
        `block_until_ready` that raises must not fail a training for the sake
        of provenance. An unblocked call is then timed short rather than not
        at all, which the row cannot distinguish -- accepted, because the
        alternative is a crash in the measuring instrument.
        """
        if name in self._jit_timed:
            return fn(*args, **kw)
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        with contextlib.suppress(Exception):
            self._jax.block_until_ready(out)
        self.jit_compile_s += time.perf_counter() - t0
        self._jit_timed.add(name)
        return out

    def _tail_impl(self, s2: Any, terminated: Any, t1: Any, fresh: Any,
                   adapter_to: Any) -> Tuple[Any, Any, Any, Any]:
        """done / truncation / auto-reset, with NO REFERENCE TO THE REWARD.

        Kept candidate-independent on purpose: this compiles into the same
        cache entry for every candidate of a given model and `num_envs`, so
        the physics-shaped compile is paid once rather than once per
        candidate. Everything here is `jnp` and branchless -- a Python `if`
        cannot see a traced value, and a row-wise reset written as control
        flow is the classic way to make a vmapped env quietly wrong.
        """
        jnp = self._jnp
        horizon_to = t1 >= self.horizon
        truncated = jnp.logical_and(
            jnp.logical_not(terminated),
            jnp.logical_or(horizon_to, adapter_to))
        done = jnp.logical_or(terminated, truncated)
        keep = jnp.reshape(jnp.logical_not(done),
                           (self.n,) + (1,) * (s2.ndim - 1))
        return done, truncated, jnp.where(keep, s2, fresh), jnp.where(
            done, 0, t1).astype(jnp.int32)

    def _redraw_physics(self, done: Any) -> None:
        """Redraw the physics of rows that just auto-reset, behind `any(done)`.

        AN EPISODE'S DRAW BELONGS TO THE EPISODE. A row that auto-resets and
        keeps the previous episode's physics is domain randomisation that
        randomises once and then stops -- which reads as DR in every config,
        every log and every seed row, and is the failure this method exists
        for. `reset()` redraws the whole batch for the same reason.

        THE GATE IS A HOST SYNC AND IS DELIBERATE. `bool(jnp.any(done))`
        blocks until the device answers, so it costs a sync per step. The
        alternative is calling `physics_rows` unconditionally, which is a
        HOST call producing a fresh pytree every step whether or not anything
        reset -- strictly worse, because most steps reset nothing. Rows
        terminate rarely, so the common path pays one `any` and returns.

        THE SELECT IS BRANCHLESS, like the state reset in `_tail_impl`: a
        Python `if` over a traced value cannot see rows, so the redraw is
        `where(done, fresh, old)` per leaf. `done` is `(n,)` and a leaf is
        `(n, *)`, so it is reshaped to broadcast rather than relying on
        numpy's trailing-axis rule, which would align the wrong axis on any
        leaf with more than one dimension.
        """
        if self._physics is None:
            return                     # a family with no DR axes: nothing to draw
        jnp, jax = self._jnp, self._jax
        if not bool(jnp.any(done)):
            return

        # A FRESH DRAW PER RESETTING ROW, which is what "redraw" has to mean.
        # `physics_rows(n)` broadcasts the adapter's CURRENT `_dr_now` to n
        # rows, so calling it again returns the same dict -- domain
        # randomisation that randomises once, at construction, and never
        # again, while every field on the seed row says DR is on. Inert on a
        # family that declares no DR axes, and wrong on every family that
        # does.
        #
        # `_sample_dr(rng)` is the adapter's own per-episode draw -- the same
        # call its `reset()` makes, so a row that resets here gets exactly
        # the draw it would have got there. `physics_rows` stays the single
        # host-dict-to-pytree place: it is called per row with that row's
        # draw and the result written into the row's slot.
        jnp, jax = self._jnp, self._jax
        rows = [int(i) for i in np.nonzero(np.asarray(done))[0]]
        env = self.env
        for i in rows:
            draw = env._sample_dr(env._rng)
            one = env.physics_rows(1, draw)
            if one is None:
                break
            self._physics = jax.tree_util.tree_map(
                lambda cur, new, _i=i: cur.at[_i].set(new[0]), self._physics, one)
        fresh = self.env.physics_rows(self.n)
        if fresh is None:
            # The adapter declared DR axes at construction and none now. That
            # is a state change this view cannot represent -- the pytree's
            # shape is fixed by the compiled step -- so it refuses rather
            # than silently keeping the old draw.
            raise RuntimeError(
                "physics_rows() returned None after returning a pytree at "
                "construction: the DR axes changed mid-episode, which the "
                "compiled step_batch cannot represent")
        # The per-row writes above are the redraw; `fresh` is fetched only to
        # detect the shape change below, not to select with. Selecting from a
        # broadcast `physics_rows(n)` is exactly the bug this replaces.

    def device_peaks(self) -> Dict[str, Any]:
        """Peak device memory for BOTH frameworks, in MiB.

        `torch.cuda.max_memory_allocated` is tensors only, which is the right
        instrument here for the same reason it is right on the seed row:
        `nvidia-smi` would show the CUDA context and XLA's arena and tell you
        nothing about either framework's own demand. The jax side is read from
        its live memory stats where the backend exposes them, and reported as
        None rather than 0 where it does not -- absent and zero are different
        facts and a `.get(k, 0)` would flatten them.
        """
        out: Dict[str, Any] = {"mem_fraction": self.xla_env.get(
            "XLA_PYTHON_CLIENT_MEM_FRACTION")}
        try:
            out["torch"] = round(
                self._torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
        except Exception:  # noqa: BLE001
            out["torch"] = None
        try:
            st = self._jax.devices()[0].memory_stats() or {}
            peak = st.get("peak_bytes_in_use")
            out["jax"] = round(peak / (1024 ** 2), 1) if peak is not None else None
        except Exception:  # noqa: BLE001
            out["jax"] = None
        return out

    @property
    def nonfinite_rows(self) -> int:
        """Rows the adapter flagged nonfinite, synced on read.

        Same device-counter shape as `n_clipped` and for the same reason: it
        is read once per training when the seed row is built, so a per-step
        host sync would be paid on every step to serve one read.
        """
        return int(self._d_nonfinite)

    @property
    def n_clipped(self) -> int:
        """The device counter, synced on read, counting its own reads.

        A property rather than an attribute so the sync happens where the
        value is consumed -- once per rollout chunk at the call site -- and
        `n_clipped_reads` on the seed row says how many syncs that was. An
        attribute updated per step would put the sync back in the hot loop,
        which is the whole thing this avoids.
        """
        self.n_clipped_reads += 1
        return int(self._jax.device_get(self._d_clipped))

    # -- action maps: the same arithmetic as `_VecEnvView`, on device --------

    def to_env_action(self, a_agent: Any) -> Any:
        """[-1, 1] -> [low, high], linearly, WITHOUT clamping -- `_VecEnvView`'s
        rule and upstream's: the env and the buffer get the sampled action."""
        return self._d_center + self._jnp.asarray(a_agent) * self._d_half

    def clip_env_action(self, a_env: Any) -> Any:
        """What the dynamics EXECUTE and what the reward is evaluated on."""
        return self._jnp.clip(self._jnp.asarray(a_env), self._d_low, self._d_high)

    def to_agent_action(self, a_env: Any) -> Any:
        return (self._jnp.asarray(a_env) - self._d_center) / self._d_half

    def episode_state(self, i: int) -> Any:
        """Row i's PRNG key. NOT the numpy generator plus DR draw that
        `_VecEnvView.episode_state` returns -- see the class docstring."""
        return self._key

    def observations(self) -> Any:
        return self._as_torch(self._features_of(self._s))

    def close(self) -> None:
        """Nothing to close: there are no forked workers and no pipes. Defined
        because the caller calls it unconditionally."""
        return None

    # -- the device boundary, measured rather than asserted ------------------

    def _as_torch(self, x: Any) -> Any:
        """A jax device array as a torch tensor, zero copy, WITHOUT leaving the
        GPU. Measured with jax 0.8.0 and torch 2.13.0+cu130:
        `torch.from_dlpack` preserves the pointer, and the learner's later
        `torch.as_tensor(t, device=cuda)` and `.float()` on an already-float32
        tensor are both no-ops on the same pointer -- which is why the learner
        loop needs no change to receive these.
        """
        return self._torch.from_dlpack(x)

    def _as_jax(self, t: Any) -> Any:
        """A torch tensor as a jax array, zero copy. The ACTION direction.

        Handing the view `actions.float().cpu().numpy()` would be a
        device->host->device round trip on every step. The contract is "the
        view takes agent actions as the learner holds them" and each view owns
        its own conversion: this one converts on device, and `_VecEnvView`
        does the `.cpu().numpy()` it needs, inside.
        """
        if hasattr(t, "__dlpack__"):
            return self._jnp.from_dlpack(t)
        return self._jnp.asarray(np.asarray(t, dtype=np.float32))

    def _measure_zero_copy(self, obs_t: Any, jax_obs: Any,
                           act_t: Any, jax_act: Any) -> None:
        """Fill `dlpack_zero_copy` from actual pointers, once, per direction.

        Per direction and never one boolean for the pair, because the two
        conversions are independent and a regression in either is invisible in
        a summary. `unsafe_buffer_pointer` is the jax side's address; equality
        with torch's `data_ptr` is the whole test.
        """
        def same(j: Any, t: Any) -> Optional[bool]:
            try:
                return int(j.unsafe_buffer_pointer()) == int(t.data_ptr())
            except Exception:  # noqa: BLE001 - an unmeasurable direction is
                return None    # reported as unknown, never as True
        if self.dlpack_zero_copy["obs"] is None:
            self.dlpack_zero_copy["obs"] = same(jax_obs, obs_t)
        if self.dlpack_zero_copy["actions"] is None:
            self.dlpack_zero_copy["actions"] = same(jax_act, act_t)

    def _features_of(self, states: Any) -> Any:
        """phi on device, or the state itself when there is none.

        `problem.search_space` phi applies ONLY to policy features: dynamics,
        reward, metric and the trace all stay on the raw state,
        and this method is the only place the distinction is applied here.
        """
        if self._feat is None:
            return states.astype(self._jnp.float32)
        return self._jnp.asarray(self._feat(states), dtype=self._jnp.float32)

    def reset(self) -> Any:
        key, sub = self._jax.random.split(self._key)
        self._key = key
        # SAME REGION NAME AS THE CONSTRUCTOR'S RESET, so this is a
        # passthrough in every normal run -- the constructor already paid
        # that compile. It goes through the clock anyway because a whole-batch
        # reset here is the FIRST one in any caller that builds the view
        # without stepping it, and a region that is timed on one path and not
        # the other is a field whose value depends on the caller.
        self._s = self._timed_first("reset_batch", self.env.reset_batch,
                                    sub, self.n)
        # A whole-batch reset redraws the whole batch, for the same reason the
        # per-row auto-reset redraws one row: an episode's physics is drawn
        # when the episode starts, and a row carrying the previous episode's
        # draw is the silent version of no DR at all.
        self._physics = self.env.physics_rows(self.n)
        self._t = self._jnp.zeros((self.n,), dtype=self._jnp.int32)
        return self.observations()

    def step(self, actions_agent: Any) -> Tuple[Any, Any, Any, Any, Any]:
        """`(next_obs, rewards, dones, time_outs, true_next_obs)`, each `(n,)`
        or `(n, obs)`, as torch CUDA tensors.

        `dones` is termination OR truncation and `time_outs` is truncation-
        and-not-termination, exactly as `_VecEnvView.step` defines them; the
        learner's bootstrap distinction depends on that and is not a detail
        this view may reinterpret.

        THE AUTO-RESET IS `_step_slot`'S RULE, PER ROW AND BRANCHLESS.
        `_step_slot` resets slot i and returns the pre-reset `true_next`, so
        the buffer stores the real terminal observation while the next
        `observations()` is the fresh episode. Here the same thing is a
        `jnp.where` over rows: a Python `if` cannot see a traced value, and a
        row-wise reset written as control flow is the classic way to make a
        vmapped env quietly wrong rather than loudly broken.
        """
        jnp, jax = self._jnp, self._jax
        a_agent = self._as_jax(actions_agent)
        a_env = self.to_env_action(a_agent)
        a_exec = self.clip_env_action(a_env)

        s = self._s
        # POSITIONAL, ALWAYS, AND NEVER OMITTED. `step_batch`'s third
        # parameter has a DEFAULT of None, so a two-argument call compiles,
        # runs, passes every test and silently trains every DR family on the
        # adapter's default physics. Passing it positionally means a contract
        # change that reorders or renames the parameter breaks loudly here
        # instead of resolving to the default.
        s2, terminated, info = self._timed_first(
            "step_batch", self.env.step_batch, s, a_exec, self._physics)
        terminated = jnp.asarray(terminated, dtype=bool)

        # THE REWARD SEES THE EXECUTED ACTION, never the sampled one.
        r_raw, _comps = self._timed_first("reward_row", self._reward_row,
                                          s, a_exec, s2)

        t1 = self._t + 1
        # `time_out` from the adapter if it offers one, else the horizon rule.
        # Both are truncation-AND-NOT-termination, so a row that terminates on
        # its horizon step is a termination, not a truncation -- the learner
        # bootstraps differently on the two and `_step_slot` is explicit.
        adapter_to = info.get("time_out") if isinstance(info, dict) else None
        adapter_to = (jnp.zeros((self.n,), dtype=bool) if adapter_to is None
                      else jnp.asarray(adapter_to, dtype=bool))

        # A NONFINITE ROW IS A TERMINATION. The adapter reports it in
        # `info["nonfinite"]` rather than in `done`, on the promise that the
        # view terminates the row and counts it; without this read, a row
        # whose physics blew up would keep stepping on NaNs for the rest of
        # the episode, poisoning every transition it wrote into the buffer.
        # Nothing would fail: the numbers would be finite-shaped garbage.
        #
        # TERMINATED, NOT TRUNCATED, so the learner does NOT bootstrap from a
        # blown-up state -- that value is meaningless and bootstrapping it
        # propagates the blow-up into the critic.
        #
        # COUNTED, NOT REFUSED. A handful of nonfinite rows over millions of
        # steps is an env quirk to record; refusing would take down a
        # training for something the auto-reset handles. The count goes on
        # the seed row so it can never be zero-by-omission.
        nonfinite = info.get("nonfinite") if isinstance(info, dict) else None
        if nonfinite is None:
            nonfinite = jnp.zeros((self.n,), dtype=bool)
        else:
            nonfinite = jnp.asarray(nonfinite, dtype=bool)
            self._d_nonfinite = self._d_nonfinite + jnp.sum(
                nonfinite.astype(jnp.int32))
        terminated = jnp.logical_or(terminated, nonfinite)

        # The TRUE next observation is the pre-reset one, for every row.
        true_next = self._features_of(s2)

        # WHICH ROWS RESET, ON THE HOST, ONCE. This sync is already paid for
        # the physics redraw; it also gates `reset_batch`, which would
        # otherwise run on EVERY step to build a fresh batch that `_tail`
        # discards whenever nothing terminated -- a host draw plus a
        # host-to-device transfer per step, for rows that reset a few times
        # in a thousand.
        #
        # `_tail`'s select is `where(keep, s2, fresh)`, and when no row is
        # done `keep` is all-True, so `fresh` is never read -- passing `s2`
        # in its place is the same arithmetic without the draw.
        horizon_to = t1 >= self.horizon
        any_done = bool(jnp.any(jnp.logical_or(
            terminated, jnp.logical_or(horizon_to, adapter_to))))
        if any_done:
            key, sub = jax.random.split(self._key)
            self._key = key
            # A SEPARATE NAME FROM THE CONSTRUCTION RESET, deliberately. A
            # cold run compiles `reset_batch` TWICE (measured 120 s and
            # 140 s, two shapes), so sharing one name with the constructor
            # would count the first and silently drop the second.
            fresh = self._timed_first("reset_batch_done", self.env.reset_batch,
                                      sub, self.n)
        else:
            fresh = s2
        done, truncated, self._s, self._t = self._timed_first(
            "tail", self._tail, s2, terminated, t1, fresh, adapter_to)
        self._episode = self._episode + done.astype(jnp.int32)
        if any_done:
            self._redraw_physics(done)

        # CLIP COUNTING, matching `_VecEnvView`: one per ROW whose executed
        # action differed from the sampled one, and `n_steps` by rows too, so
        # `action_clip_fraction` is the same ratio over the same denominator.
        # CLIP COUNT ACCUMULATES ON DEVICE and is read once per rollout chunk,
        # not per step. `n_clipped` feeds only the seed row's
        # `action_clip_fraction`, via chunk deltas at the call site -- nobody
        # reads it per step, and the slot-pool path already reports it per
        # round, so chunk granularity is the existing contract rather than a
        # relaxation of it. `n_clipped` below is a property that syncs on read
        # and counts its own reads, so the artifact says how often it synced.
        clipped_rows = jnp.any(a_exec != a_env, axis=tuple(range(1, a_env.ndim)))
        self._d_clipped = self._d_clipped + jnp.sum(clipped_rows)
        self.n_steps += int(self.n)

        # REWARD NORMALISATION IS THE ONLY PER-STEP HOST SYNC, AND ONLY WHEN
        # ASKED FOR. `_RewardNorm` keeps running statistics in Python floats
        # and is shared with every other tier; matching its arithmetic on
        # device would be a second implementation of a stateful thing, and
        # "parallel must stay bit-identical to sequential" is the rule that
        # says not to. So under `train.reward_norm != none` the rows come to
        # host, the cost is timed into `reward_sync_s`, and the seed row
        # carries it -- the tier's price is stated, not hidden in steps/s.
        #
        # Under `none` -- the default -- there is NO host round trip at all: the rewards go to torch through the
        # same DLPack path as the observations.
        if self._norm.mode == "none":
            rewards_out = self._as_torch(r_raw.astype(jnp.float32))
        else:
            _t0 = time.time()
            r_host = np.asarray(jax.device_get(r_raw), dtype=float).reshape(-1)
            self.reward_sync_s += time.time() - _t0
            r_norm = np.asarray([self._norm(float(v)) for v in r_host],
                                dtype=np.float32)
            rewards_out = None      # built below, on the obs tensor's device

        next_obs = self._features_of(self._s)
        obs_t = self._as_torch(next_obs)
        self._measure_zero_copy(obs_t, next_obs,
                                actions_agent, a_agent)
        if rewards_out is None:
            rewards_out = self._torch.as_tensor(r_norm, device=obs_t.device)
        return (obs_t,
                rewards_out,
                self._as_torch(done),
                self._as_torch(truncated),
                self._as_torch(true_next))


class _SlotPool:
    """`workers` forked children, each stepping a contiguous block of a
    `_VecEnvView`'s slots on its own fork-copy of the adapter.

    Protocol, one duplex pipe per child: the parent sends `("reset",)`,
    `("step", actions_block)` or `("close",)`; the child answers with its
    block's `(raw_rewards, dones, time_outs, true_next, states, episode_blobs,
    t, episode, n_steps, n_clipped)` or `("error", traceback)`. The parent
    mirrors the per-slot state so `observations()`, `episode_state(i)` and the
    clip counters read exactly as they do in-process. Children are waited on by
    NAMED pid (`_fork_seeds`'s rule: the parent may own other children), never
    `waitpid(-1)`; a child whose pipe hits EOF exits on its own.
    """

    def __init__(self, view: "_VecEnvView", workers: int) -> None:
        self.view = view
        n = view.n
        bounds = np.linspace(0, n, workers + 1).astype(int)
        self.blocks: List[Tuple[int, int]] = [(int(bounds[k]), int(bounds[k + 1]))
                                              for k in range(workers) if bounds[k + 1] > bounds[k]]
        self.conns: List[Any] = []
        self.pids: List[int] = []
        for lo, hi in self.blocks:
            parent_end, child_end = _mpc.Pipe(duplex=True)
            _training._flush_all_streams()
            pid = os.fork()
            if pid == 0:  # -- child; never returns --
                parent_end.close()
                for earlier in self.conns:  # the parent ends of siblings forked before us
                    with contextlib.suppress(Exception):
                        earlier.close()
                _slot_worker_main(view, lo, hi, child_end)
            child_end.close()
            self.conns.append(parent_end)
            self.pids.append(pid)

    def _round(self, msg: Tuple[Any, ...]) -> List[Any]:
        for c in self.conns:
            c.send(msg)
        out = []
        for (lo, hi), c in zip(self.blocks, self.conns):
            try:
                rep = c.recv()
            except EOFError:
                self.close()
                raise RuntimeError(f"fasttd3 env worker for slots [{lo}, {hi}) died "
                                   "(pipe closed) -- see the log above for its traceback")
            if _is_worker_error(rep):
                self.close()
                raise RuntimeError(f"fasttd3 env worker for slots [{lo}, {hi}) raised:\n{rep[1]}")
            out.append(rep)
        return out

    def reset(self) -> None:
        v = self.view
        for (lo, hi), rep in zip(self.blocks, self._round(("reset",))):
            states, blobs, t, ep = rep
            v._s[lo:hi], v._ep[lo:hi], v._t[lo:hi], v._episode[lo:hi] = states, blobs, t, ep

    def step(self, actions: np.ndarray, dones: np.ndarray, time_outs: np.ndarray,
             true_next: np.ndarray) -> np.ndarray:
        v = self.view
        raw = np.zeros(v.n, dtype=np.float64)
        for c, (lo, hi) in zip(self.conns, self.blocks):
            c.send(("step", np.ascontiguousarray(actions[lo:hi])))
        for (lo, hi), c in zip(self.blocks, self.conns):
            try:
                rep = c.recv()
            except EOFError:
                self.close()
                raise RuntimeError(f"fasttd3 env worker for slots [{lo}, {hi}) died "
                                   "(pipe closed) -- see the log above for its traceback")
            if _is_worker_error(rep):
                self.close()
                raise RuntimeError(f"fasttd3 env worker for slots [{lo}, {hi}) raised:\n{rep[1]}")
            r, d, to, tn, states, blobs, t, ep, n_steps, n_clipped = rep
            raw[lo:hi] = r
            dones[lo:hi] = d
            time_outs[lo:hi] = to
            true_next[lo:hi] = tn
            v._s[lo:hi], v._ep[lo:hi], v._t[lo:hi], v._episode[lo:hi] = states, blobs, t, ep
            v.n_steps += int(n_steps)
            v.n_clipped += int(n_clipped)
        return raw

    def close(self) -> None:
        conns, self.conns = self.conns, []
        pids, self.pids = self.pids, []
        for c in conns:
            with contextlib.suppress(Exception):
                c.send(("close",))
            with contextlib.suppress(Exception):
                c.close()
        deadline = time.monotonic() + 5.0
        for pid in pids:
            while True:
                try:
                    done, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    break
                if done == pid:
                    break
                if time.monotonic() > deadline:
                    with contextlib.suppress(OSError):
                        os.kill(pid, signal.SIGKILL)
                    with contextlib.suppress(OSError):
                        os.waitpid(pid, 0)
                    break
                time.sleep(0.01)


def _is_worker_error(rep: Any) -> bool:
    """A worker's `("error", traceback)` reply. The type check comes FIRST: a
    normal step reply's first element is the block's raw-reward ndarray, and
    `ndarray == "error"` is an elementwise comparison whose truth value raises
    for any block of two or more slots -- which a one-slot-per-worker test
    cannot see."""
    return (isinstance(rep, tuple) and len(rep) == 2 and isinstance(rep[0], str)
            and rep[0] == "error")


def _slot_worker_main(view: "_VecEnvView", lo: int, hi: int, conn: Any) -> None:
    """Body of one forked slot worker. Steps slots [lo, hi) on this process's
    copy of the adapter; the parent-side `_pool` is dropped so nothing here
    can recurse into the pipes. Exits the process on `("close",)` or EOF."""
    code = 0
    try:
        view._pool = None
        _training._pin_one_thread()
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            kind = msg[0]
            if kind == "close":
                break
            try:
                if kind == "reset":
                    for i in range(lo, hi):
                        view._reset_slot(i)
                    conn.send((list(view._s[lo:hi]), list(view._ep[lo:hi]),
                               list(view._t[lo:hi]), list(view._episode[lo:hi])))
                elif kind == "step":
                    acts = msg[1]
                    k = hi - lo
                    raw = np.zeros(k, dtype=np.float64)
                    dones = np.zeros(k, dtype=np.int64)
                    tos = np.zeros(k, dtype=np.int64)
                    tn = np.zeros((k, view._width), dtype=np.float32)
                    n_steps0, n_clipped0 = view.n_steps, view.n_clipped
                    for j, i in enumerate(range(lo, hi)):
                        raw[j], dones[j], tos[j] = view._step_slot(i, acts[j], tn[j])
                    conn.send((raw, dones, tos, tn, list(view._s[lo:hi]), list(view._ep[lo:hi]),
                               list(view._t[lo:hi]), list(view._episode[lo:hi]),
                               view.n_steps - n_steps0, view.n_clipped - n_clipped0))
                else:
                    conn.send(("error", f"unknown message {kind!r}"))
            except Exception:  # noqa: BLE001 -- reported to the parent, which raises
                conn.send(("error", traceback.format_exc()))
    except BaseException:  # noqa: BLE001 -- a child must never unwind into the parent's code
        code = 1
    finally:
        with contextlib.suppress(Exception):
            conn.close()
        os._exit(code)


# ==========================================================================
# Policy blobs and replay slices -- `train.init`, `loop.carry`
# ==========================================================================

_BLOB_FORMAT = "fasttd3/v1"


def _fasttd3_blob(ns: Dict[str, Any], arch: str, actor: Any, qnet: Any,
                  qnet_target: Any, obs_normalizer: Any) -> Optional[_PolicyBlob]:
    """Serialise the agent for `_POLICY_STORE`. Same role as `_sb3_policy_blob`;
    tensors only plus the format tag, so the reader can load it with
    `weights_only=True` -- run directories may live on a shared, world-writable
    mount, and a pickle-bearing checkpoint is the attack `checkpoint._DECODABLE`
    closes."""
    torch = ns["torch"]
    try:
        buf = io.BytesIO()
        torch.save({"format": _BLOB_FORMAT, "architecture": arch,
                    "actor": actor.state_dict(), "qnet": qnet.state_dict(),
                    "qnet_target": qnet_target.state_dict(),
                    "obs_normalizer": obs_normalizer.state_dict()
                    if hasattr(obs_normalizer, "state_dict") else {}}, buf)
        return _PolicyBlob(buf.getvalue())
    except Exception as exc:  # noqa: BLE001 - a missing checkpoint is not a failed run
        log.warning("fasttd3: could not serialise the policy for train.init "
                    "(%s: %s); this candidate will carry no policy_ref",
                    type(exc).__name__, exc)
        return None


def _fasttd3_apply(ns: Dict[str, Any], arch: str, actor: Any, qnet: Any,
                   qnet_target: Any, obs_normalizer: Any, blob: Any, ref: str) -> bool:
    """Load a stored blob into live networks. Returns whether it took.

    Same leniency contract as `_sb3_apply_policy`: a blob another backend wrote
    (an SB3 zip, the bc_prior clone), the other architecture's, or one whose
    shapes no longer fit is a WARNING and a cold start, never a crash -- a
    checkpoint that no longer fits must not destroy the search that found it.
    The actor's `noise_scales` buffer is per-run exploration state, not policy,
    so the LIVE model's own is kept (it is also how a blob trained at one
    `num_envs` warm-starts a run at another).
    """
    if blob is None:
        return False
    try:
        payload = _fasttd3_payload(ns, arch, blob, ref)
        actor_sd = dict(payload["actor"])
        actor_sd["noise_scales"] = actor.state_dict()["noise_scales"]
        actor.load_state_dict(actor_sd)
        qnet.load_state_dict(payload["qnet"])
        qnet_target.load_state_dict(payload["qnet_target"])
        if payload.get("obs_normalizer") and hasattr(obs_normalizer, "load_state_dict"):
            obs_normalizer.load_state_dict(payload["obs_normalizer"])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("train.init: %s did not load into this model (%s: %s) -- "
                    "training from scratch", ref, type(exc).__name__, exc)
        return False


def _fasttd3_payload(ns: Dict[str, Any], arch: str, blob: Any, ref: str) -> Dict[str, Any]:
    """Decode a `_POLICY_STORE` blob into its tensors-only payload, or raise
    `ValueError` naming why it is not one of ours (another backend's format,
    the other architecture). Shared by the warm start (`_fasttd3_apply`) and
    the elite anchor (`_fasttd3_attach_elite_constraint`), so the two cannot
    disagree about what a fasttd3 checkpoint is."""
    if not isinstance(blob, np.ndarray) or blob.dtype != np.uint8:
        raise ValueError(f"{ref} holds a {type(blob).__name__}, which is not a fasttd3 blob")
    torch = ns["torch"]
    payload = torch.load(io.BytesIO(np.asarray(blob, dtype=np.uint8).tobytes()),
                         map_location="cpu", weights_only=True)
    if not (isinstance(payload, dict) and payload.get("format") == _BLOB_FORMAT):
        raise ValueError("not a fasttd3/v1 payload (an sb3 zip, or another "
                         "backend's checkpoint?)")
    if payload.get("architecture") != arch:
        raise ValueError(f"architecture {payload.get('architecture')!r} does not "
                         f"match train.architecture={arch!r}")
    return payload


def fasttd3_policy_from_blob(cfg: Any, env: Any, phi: Any, blob: Any,
                             ref: str) -> Callable[[np.ndarray], np.ndarray]:
    """The fasttd3 branch of `training.policy_from_ref`: the run's architecture,
    the blob's weights, the loop's own action path.

    Mirrors `_sb3_policy_from_blob`. The payload is decoded and CHECKED by
    `_fasttd3_payload` (the `format: fasttd3/v1` marker and the architecture),
    the actor and observation normaliser are built by `_build_networks` under
    the resolved `train.architecture` and `train.hyperparameters` -- the same
    constructor the training loop used -- the stored state dicts are loaded, and
    the callable returned is `_greedy_policy`, the very function `_policy_fn`
    hands the loop, over `_action_affine(env)`. On the host (`cpu`): every
    caller (`EnvAdapter.dr_probe`) rolls one state at a time, and the payload is
    mapped to cpu on load.

    NEVER DEGRADES, like its caller: anything that cannot produce the policy the
    blob describes raises `ValueError` naming `ref` -- no config to name the
    network, a payload that is not `fasttd3/v1` or is the other architecture, a
    state dict that does not fit the rebuilt actor, a normaliser the config no
    longer declares. The actor's `noise_scales` buffer is per-run exploration
    state, not policy (`_fasttd3_apply`), so the rebuilt actor keeps its own.
    """
    if cfg is None:
        raise ValueError(
            f"{ref}: a fasttd3 policy blob can only be rebuilt with the run's config -- "
            "train.architecture and train.hyperparameters name the network it was saved "
            "from, and no config was given")
    ns = _torch_classes()
    torch = ns["torch"]
    arch = _resolve_arch(cfg)
    try:
        payload = _fasttd3_payload(ns, arch, blob, ref)
    except ValueError as exc:
        raise ValueError(f"{ref}: {exc}") from exc
    hyper = _fasttd3_hyper(cfg)
    device = torch.device("cpu")
    n_obs = int(phi.dim) if phi is not None else int(np.asarray(env.obs_low).size)
    n_act = int(np.asarray(env.action_low).size)
    num_envs = max(1, int(hyper["num_envs"]))
    with _training._single_thread_torch():
        actor, _qnet, _qnet_target, obs_normalizer = _build_networks(
            ns, arch, hyper, n_obs, n_act, num_envs, device)
        actor_sd = dict(payload["actor"])
        actor_sd["noise_scales"] = actor.state_dict()["noise_scales"]
        try:
            actor.load_state_dict(actor_sd)
        except Exception as exc:  # noqa: BLE001 -- a shape that does not fit is the finding
            raise ValueError(
                f"{ref}: the stored fasttd3 actor does not fit the network this config "
                f"builds ({type(exc).__name__}: {exc}) -- a policy trained on a "
                "co-designed observation needs its observation_code, and a run's "
                "train.hyperparameters name its widths") from exc
        stored_norm = payload.get("obs_normalizer") or {}
        if stored_norm and obs_normalizer is None:
            raise ValueError(
                f"{ref}: the blob carries an observation normaliser but this config's "
                "train.hyperparameters.obs_normalization is off; the rebuilt policy would "
                "act on unnormalised features it was never trained on")
        if stored_norm:
            obs_normalizer.load_state_dict(stored_norm)
        actor.eval()
    center, half = _action_affine(env)

    def _features(s: Any) -> np.ndarray:
        return np.asarray(phi(s) if phi is not None else s, dtype=np.float32)

    inner = _greedy_policy(ns, actor, obs_normalizer, _features,
                           lambda a: center + np.asarray(a, dtype=float) * half, device)

    def policy(s: np.ndarray) -> np.ndarray:
        with _training._single_thread_torch():
            return inner(s)

    return policy


def _fasttd3_attach_elite_constraint(ns: Dict[str, Any], arch: str, actor: Any, qnet: Any,
                                     optimisers: Sequence[Any], weight: float,
                                     blob: Any, ref: str) -> int:
    """LaRes Eq. 4 anchored to the ELITE, whatever the networks currently hold
    -- `training._sb3_attach_elite_constraint` for this backend's blob format.

    `population.attach_l2_to_optimisers` snapshots the live parameters as its
    reference, so this loads the elite's stored actor and critic into the live
    modules for exactly the duration of that call and then puts the seed's own
    back: `load_state_dict` copies VALUES into the same `Parameter` objects,
    so the snapshot holds theta_elite while the live parameters are the seed's
    again, and the term computes `2w (theta - theta_elite)` from then on. On an
    unsliced `warm_start_from_best` the networks already hold the elite and the
    swap is a no-op, bit for bit; under `train.interaction: shared_population`
    a resumed slice holds its OWN previous slice, and a snapshot of that would
    be a proximal term toward itself (the same trap as on sb3). The
    reference is the round-start elite -- `state.policy_ref`, which only stage
    6 moves -- fixed for every slice of the round, as the release's workers
    load `elite_actor` / `elite_q1` / `elite_q2` once per evolution and hold
    them (`refs/code/LaRes/utils.py:2119-2125`). The actor's `noise_scales`
    buffer is per-run exploration state and is never swapped. Returns the
    number of optimisers patched; 0 means the term is NOT running and the
    reason has been logged.
    """
    own = {"actor": {k: v.detach().clone() for k, v in actor.state_dict().items()},
           "qnet": {k: v.detach().clone() for k, v in qnet.state_dict().items()}}

    def _restore() -> None:
        actor.load_state_dict(own["actor"])
        qnet.load_state_dict(own["qnet"])

    try:
        payload = _fasttd3_payload(ns, arch, blob, ref)
        actor_sd = dict(payload["actor"])
        actor_sd["noise_scales"] = own["actor"]["noise_scales"]
        actor.load_state_dict(actor_sd)
        qnet.load_state_dict(payload["qnet"])
    except Exception as exc:  # noqa: BLE001
        # torch copies what fits before raising on what does not, so put the
        # seed's own parameters back before saying the term is not running.
        _restore()
        log.warning("train.elite_constraint.kind=l2_params: %s does not hold this agent's "
                    "parameters (%s: %s); the term is NOT running", ref, type(exc).__name__,
                    str(exc).splitlines()[0] if str(exc) else "")
        return 0
    try:
        from .population import attach_l2_to_optimisers
        return attach_l2_to_optimisers(list(actor.parameters()) + list(qnet.parameters()),
                                       optimisers, float(weight))
    finally:
        _restore()


def _data_root_source() -> Optional[str]:
    """Which candidate supplied `tasks/`, for the seed row. Never raises.

    `required=False`, because a provenance field must not be the thing that
    fails a training that has already finished. A run that got this far has
    its catalogue; if the lookup somehow cannot be repeated now, the honest
    record is None rather than a crash at the artifact-writing step.
    """
    try:
        from .. import paths
        return paths.data_dir_with_source("tasks", required=False)[1]
    except Exception:
        return None


def _param_device(agent: Dict[str, Any]) -> str:
    """Every distinct device the LEARNER'S PARAMETERS are on, sorted and joined.

    THE ACTOR'S FIRST PARAMETER IS NOT ENOUGH. `fasttd3` builds the actor and
    the critics in separate calls with their own `device=`, so there are at
    least two independent chances to get it wrong, and a critic on cpu, or
    any single parameter left behind, must show up here.

    So this returns the SET: "cuda:0" when everything agrees, and
    "cpu, cuda:0" when it does not -- a visible result rather than a pass.

    Not the configured device and not a probe tensor: the thing a gradient
    step actually ran on. A probe tensor can report a live cuda device while
    the learner sits on cpu, and a `device` field alone is easy to overlook.

    Returns `"unknown"` rather than raising: provenance may never be the thing
    that fails a training.
    """
    try:
        seen = set()
        for key in ("actor", "qnet", "qnet_target"):
            net = agent.get(key)
            params = getattr(net, "parameters", None)
            if params is None:
                continue
            for prm in params():
                seen.add(str(prm.device))
        return ", ".join(sorted(seen)) if seen else "unknown"
    except Exception:  # noqa: BLE001 - no params, no actor, a stub in a test
        return "unknown"


#: How many utilisation point samples one training takes. Four
#: rather than one because a single sample can land inside a cold backend's
#: compile window and read 0 on a GPU that is 33-58% busy for the rest of the
#: run; four rather than forty because each is an `nvidia-smi` subprocess and
#: this field is provenance, not a profiler.
_GPU_UTIL_SAMPLES = 4


class _UtilSampler:
    """`_GPU_UTIL_SAMPLES` utilisation point samples across one training.

    AN OBJECT RATHER THAN FOUR LOCALS, so the chunk arithmetic and the row
    fields can be tested without a GPU: a test that needs a GPU to fail is a
    test that rarely runs. Everything here is exercised by
    `tests/test_fasttd3.py` on CPU with the sampler stubbed.

    WHERE THE SAMPLES FALL. Evenly spaced across the chunks and NEVER chunk
    0: on a cold JIT backend the first chunk is still compiling, and that is
    precisely the window in which 0 is both the true answer and the least
    representative one.

    None IS DROPPED, NEVER RECORDED AS 0. "could not tell" (no nvidia-smi, a
    timeout) and "the GPU was idle" are the two facts this family exists to
    keep apart; `n_samples` is what says how many readings the max and the
    mean are over.
    """

    def __init__(self, n_chunks: int, enabled: bool,
                 sampler: Callable[[], Optional[int]] = None) -> None:
        self._sampler = sampler or _gpu_utilization_percent
        self.samples: List[int] = []
        self.at: set = set()
        if enabled and n_chunks > 0:
            k = min(_GPU_UTIL_SAMPLES, n_chunks)
            self.at = {max(1, round((i + 1) * n_chunks / (k + 1))) - 1
                       for i in range(k)}

    def maybe_sample(self, ck: int) -> None:
        if ck not in self.at:
            return
        got = self._sampler()
        if got is not None:
            self.samples.append(int(got))

    def row_fields(self) -> Dict[str, Any]:
        """The three seed-row keys. `None` statistics when nothing was read,
        so a row that measured nothing cannot be averaged with one that did."""
        if not self.samples:
            return {"gpu_utilization_max_pct": None,
                    "gpu_utilization_mean_pct": None,
                    "gpu_utilization_n_samples": 0}
        return {
            "gpu_utilization_max_pct": max(self.samples),
            "gpu_utilization_mean_pct": round(
                sum(self.samples) / len(self.samples), 1),
            # THE DENOMINATOR TRAVELS WITH THE STATISTIC: a max over one
            # sample and a max over four are different claims.
            "gpu_utilization_n_samples": len(self.samples),
        }


def _gpu_utilization_percent() -> Optional[int]:
    """One `nvidia-smi` utilisation POINT SAMPLE, or None. NOT evidence of idleness.

    Taken mid-run rather than at the end, because after the loop stops the
    number is the idle GPU -- but mid-run is not enough to make a low reading
    mean anything. Measured: a job read **0% on all eight 20 s samples while
    genuinely training**, because a 6-130 ms update burst inside a ~900 ms
    iteration is invisible at that cadence.

    So this is recorded and never *read to decide anything*. A non-zero
    reading is worth having and costs one subprocess; a zero means "no
    evidence", never "idle". The actual evidence that a job ran on a GPU is
    `learner_device` plus a non-zero
    `torch.cuda.max_memory_allocated` -- both on the same seed row.

    CALLED `_GPU_UTIL_SAMPLES` TIMES PER TRAINING, not once. A single sample
    reads as a summary of the run, and on a cold JIT backend it can land
    inside the compile window, which is the one interval where 0 is the true
    answer and the least representative one. The row carries
    `gpu_utilization_max_pct`, `_mean_pct` and `_n_samples` -- three fields
    that cannot be read as "the GPU utilisation" by accident. Everything
    above still holds for each individual sample.

    THE ALTERNATIVE, AND WHY NOT: `nvidia-smi dmon -s u -d 1` for 30 s around
    the midpoint and report the max would be real evidence. It also blocks
    the training loop for 30 s per seed, inside the hot path, to produce a
    field nothing reads. The honest cheap sample plus an honest label is the
    better trade; if someone later wants utilisation as evidence, dmon behind
    a config key is the shape, not this function with a longer timeout.

    None on every failure path -- no nvidia-smi, a timeout, unparseable output
    -- because "could not tell" and "the GPU was idle" are different facts and
    a cost report must not read the first as the second.
    """
    try:
        # SCOPED TO THIS PROCESS'S GPU, not GPU 0. On a shared node GPU 0 is
        # very often somebody else's job, and reporting their utilisation
        # beside our seed row is worse than reporting none -- it is a number
        # about the wrong machine wearing our run's name. `CUDA_VISIBLE_DEVICES`
        # is what a cluster scheduler sets; `-i` takes its first entry (the
        # device torch calls `cuda:0`). Unset means we are not scoped and GPU 0
        # really is ours.
        visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
               "--format=csv,noheader,nounits"]
        if visible:
            cmd[1:1] = ["-i", visible]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except Exception:  # noqa: BLE001 - not installed, not on PATH, timed out
        return None
    if out.returncode != 0:
        return None
    first = (out.stdout or "").strip().splitlines()
    try:
        return int(first[0].strip())
    except (IndexError, ValueError):
        return None


def _fasttd3_export_replay(view: _VecEnvView, rb: Any, cap: int) -> Optional[_ReplaySlice]:
    """The last `cap` transitions, RAW, oldest first -- `_sb3_export_replay`'s
    contract exactly: no rewards (GT's next round relabels with ITS OWN reward),
    `done` with the timeout mask already applied, actions mapped back to ENV
    space and observations in STATE space, because a slice in the store has to
    mean the same thing whichever backend wrote it (the caller already refuses
    to export under a co-designed observation, as sb3 does)."""
    n_filled = int(min(rb.ptr, rb.buffer_size))
    if n_filled <= 0:
        return None
    order = np.arange(int(rb.ptr) - n_filled, int(rb.ptr)) % int(rb.buffer_size)
    # Time-major flatten: rows [t0/env0, t0/env1, ..., t1/env0, ...] keeps
    # "chronological, then the tail" true across the whole fleet.
    obs = rb.observations[:, order].transpose(0, 1).reshape(-1, rb.n_obs).cpu().numpy()
    act = rb.actions[:, order].transpose(0, 1).reshape(-1, rb.n_act).cpu().numpy()
    nxt = rb.next_observations[:, order].transpose(0, 1).reshape(-1, rb.n_obs).cpu().numpy()
    done = rb.dones[:, order].transpose(0, 1).reshape(-1).cpu().numpy()
    trunc = rb.truncations[:, order].transpose(0, 1).reshape(-1).cpu().numpy()
    keep = slice(-max(1, int(cap)), None)
    term = (done * (1 - trunc)).astype(bool)[keep]
    # Executed actions, not sampled ones: a reader relabels these rows with ITS
    # reward, and sb3's slice holds what SB3 clipped before stepping.
    return _ReplaySlice(np.asarray(obs[keep], dtype=np.float32),
                        np.stack([view.clip_env_action(view.to_env_action(a))
                                  for a in act[keep]]),
                        np.asarray(nxt[keep], dtype=np.float32),
                        term)


def _fasttd3_prefill_plan(num_steps, handoff, phi):
    """What the replay half of a `shared_population` slice will do, decided
    BEFORE `buffer_rows` is widened. Returns `(rows, ref, supported)`.

    Split out of `fasttd3_backend` for one reason: it is the only part of the
    prefill decision that needs no torch, and leaving it inline would make the
    `num_steps` refusal testable only through a `@needs_torch`, `slow`-marked
    end-to-end test -- which skips in the default test configuration. A
    refusal whose only test never executes is the shape this file exists to
    prevent one level up.

    `supported` is FALSE only where this learner configuration CANNOT honour a
    hand-off at all -- the same meaning `replay_prefill_supported` carries on
    sb3, where it is `bool(off_policy)`. It is therefore False on
    `num_steps > 1` and True in the two circumstantial cases (the ref is not in
    the round store; a co-designed observation is in play), which stay
    `supported` with `rows=None` and are read off `replay_prefill_*: 0`.
    Refused and nothing-to-pool both look like `pooled: 0`, so the one that is
    a REFUSAL says so.

    `num_steps > 1` is decided here rather than inside
    `_fasttd3_prefill_replay` because the caller widens `buffer_rows` for the
    pool between the two: refusing late would widen the buffer for rows that
    are never going to be written, and still report `supported: True`.
    """
    ref = getattr(handoff, "prefill_ref", None) if handoff is not None else None
    if not ref:
        return None, None, True
    if int(num_steps) > 1:
        log.warning("shared_population: replay prefill is not supported with "
                    "train.hyperparameters.num_steps=%d on fasttd3 -- an n-step "
                    "window over pooled rows would sum rewards across unrelated "
                    "transitions (the mask is built from `dones` only, and the "
                    "pool is not one episode stream). The buffer is NOT widened "
                    "and every seed row reports replay_prefill_supported=false",
                    int(num_steps))
        return None, None, False
    from .training import _ROUND_REPLAY
    rows = _ROUND_REPLAY.get(ref)
    if rows is None or len(rows) == 0:
        log.warning("shared_population: %s is not in the round replay store -- "
                    "this slice starts from an empty buffer", ref)
        return None, ref, True
    if phi is not None:
        # `use_secondary`'s argument: the stored rows are RAW states and this
        # candidate's networks eat phi(s), which is its own.
        log.warning("shared_population: a co-designed observation and a restored "
                    "replay buffer cannot both be honoured on fasttd3 (the rows "
                    "are states, the buffer holds features); this slice starts "
                    "from an empty buffer")
        return None, ref, True
    return rows, ref, True


def _fasttd3_prefill_replay(ns: Dict[str, Any], view: _VecEnvView, rb: Any,
                            rows: Any, reward: CompiledReward, norm_mode: str,
                            device: Any, num_envs: int, n_steps: int) -> int:
    """LaRes §4.3 on this backend: fill `rb` with `rows`, relabelled, before training.

    `_sb3_prefill_replay`'s semantics on FastTD3's tensors, and the three
    places the two backends genuinely differ are the whole content of this
    function.

    **The lane layout is the export's inverse.** SB3's buffer is one stream;
    this one is `(n_env, rows)` with a single write head, and
    `_fasttd3_export_replay` flattens it TIME-MAJOR (`t0/env0, t0/env1, ...`).
    So a flat pool is laid back as `reshape(k, n_env, dim).permute(1, 0, 2)`,
    which makes a round trip through export-then-prefill the identity. Rows
    are truncated to a multiple of `n_env`, OLDEST first as the release's ring
    does -- a partial final column would otherwise leave padding lanes that
    `sample` cannot distinguish from real transitions.

    **`n_steps > 1` is REFUSED, not degraded.** The n-step window walks
    forward from a sampled index assuming the lane is one contiguous episode
    stream, and its reward mask is built from `dones` alone -- truncations do
    not stop the sum (`SimpleReplayBuffer.sample`). A pooled region is several
    arms' rows in arbitrary order, so a window straddling it would add up
    rewards from unrelated transitions and call the result an n-step return.
    That is a wrong number that looks like a right one, so the pool is refused
    here exactly as `_sb3_prefill_replay` refuses `optimize_memory_usage`, and
    for the same reason.

    **Capacity is the caller's.** `fasttd3_backend` widens `buffer_rows` to
    make room before constructing `rb`; this fills what it is given and drops
    the oldest overflow.

    Returns the number of rows written (0 = nothing, and why is logged).
    """
    if int(n_steps) > 1:
        log.warning("shared_population: replay prefill is not supported with "
                    "train.hyperparameters.num_steps=%d on fasttd3 -- an n-step "
                    "window over pooled rows would sum rewards across unrelated "
                    "transitions (the mask is built from `dones` only, and the "
                    "pool is not one episode stream). This slice starts from an "
                    "empty buffer", int(n_steps))
        return 0
    cols = _fasttd3_secondary(ns, view, rows, reward, norm_mode, device)
    if cols is None:
        return 0
    torch = ns["torch"]
    n = int(cols["n"])
    capacity = int(rb.buffer_size) * int(num_envs)
    keep = min(n - (n % int(num_envs)), capacity)
    if keep <= 0:
        log.warning("shared_population: %d pooled row(s) is fewer than num_envs=%d, "
                    "so there is no whole lane-column to write; this slice starts "
                    "from an empty buffer", n, int(num_envs))
        return 0
    k = keep // int(num_envs)
    sl = slice(n - keep, n)   # the NEWEST `keep` rows; overflow drops oldest

    def lanes(x: Any, width: int) -> Any:
        return x[sl].reshape(k, int(num_envs), width).permute(1, 0, 2)

    rb.observations[:, :k] = lanes(cols["observations"], rb.n_obs)
    rb.next_observations[:, :k] = lanes(cols["next_observations"], rb.n_obs)
    rb.actions[:, :k] = lanes(cols["actions"], rb.n_act)
    rb.rewards[:, :k] = cols["rewards"][sl].reshape(k, int(num_envs)).permute(1, 0)
    rb.dones[:, :k] = cols["dones"][sl].reshape(k, int(num_envs)).permute(1, 0)
    # The pool's `done` already carries the timeout mask (both exporters apply
    # it before writing), so a prefilled row is never a truncation.
    rb.truncations[:, :k] = torch.zeros_like(rb.truncations[:, :k])
    rb.ptr = k
    return int(keep)


def _fasttd3_secondary(ns: Dict[str, Any], view: _VecEnvView, secondary: Any,
                       reward: CompiledReward, norm_mode: str, device: Any
                       ) -> Optional[Dict[str, Any]]:
    """GT's shared buffer, relabelled ONCE under this candidate's reward --
    `_sb3_secondary_buffer`'s semantics on this backend's tensors. Rows arrive
    in state space with env-space actions; the networks train in agent space,
    so actions are inverted at import. Returns tensors ready to splice."""
    from .training import _replay_columns
    cols = _replay_columns(secondary)
    if cols is None:
        return None
    obs, act, nxt, done = cols
    n = int(obs.shape[0])
    if n <= 0:
        return None
    torch = ns["torch"]
    relabel = _RewardNorm(norm_mode)
    on_error = _error_router("exception_soft", io.StringIO())
    rew = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = np.asarray(act[i], dtype=float).ravel()
        try:
            raw, _cv = reward(obs[i], a, nxt[i])
            if not math.isfinite(raw):
                raise FloatingPointError(f"reward returned {raw!r}")
        except Exception as exc:  # noqa: BLE001
            raw = on_error(exc)
        rew[i] = relabel(float(raw))
    to = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)  # noqa: E731
    return {"observations": to(obs),
            "actions": to(np.stack([view.to_agent_action(a) for a in act])),
            "next_observations": to(nxt), "rewards": to(rew),
            "dones": torch.as_tensor(np.asarray(done, dtype=np.int64), device=device),
            "n": n}


# ==========================================================================
# registry: `train.backend: fasttd3`
# ==========================================================================


@register("train_backend", "fasttd3")
def fasttd3_backend(ctx: Any, state: Any, candidate: Candidate, n_seeds: int = 1,
                    env_steps: Optional[int] = None,
                    resume_ref: Optional[str] = None,
                    seed_phase: str = "",
                    handoff: Optional[Any] = None) -> TrainResult:
    """FastTD3 against the candidate reward, on `num_envs` slots of the adapter.

    Same shape as `_sb3_run` deliberately -- chunked training with a real
    checkpoint curve, `_pruner_for`/`_selection_for`/`train.timeout_s` consulted
    between chunks, `train.init` read off the same state slots, the last seed's
    agent published to `_POLICY_STORE` -- so every §3 key means the same thing
    here as on sb3. What differs is the learner: the update loop is the
    line-faithful FastTD3 port documented at the top of this module.

    `train.env_steps` counts ENV INTERACTIONS, as on every backend; FastTD3
    counts iterations of `num_envs` transitions each, so iterations =
    env_steps // num_envs (upstream's HumanoidBench "100000 timesteps" at 128
    envs is `train.env_steps: 12_800_000` here -- the honest unit for a budget
    a cost report compares across backends).

    `handoff` (`train.interaction: shared_population`, a `training.SliceHandoff`)
    is honoured IN FULL: the slice index salts the seed, the round pool
    pre-fills this learner's buffer relabelled under the candidate's own
    reward (`_fasttd3_prefill_replay`), and the rows this slice added are
    exported raw through the same `_ROUND_REPLAY` key the sb3 path writes --
    which is why `population.py` needs no backend branch. The seed row says
    `replay_prefill_supported: true` with the row counts beside it, so a pool
    that did not execute is still visible as `replay_prefill_*: 0` rather than
    inferred from the backend's name -- EXCEPT on `num_steps > 1`, where the
    pool cannot be honoured at all: that is decided before the buffer is
    widened and the row reads `replay_prefill_supported: false`, because
    "refused" and "nothing to pool" both look like `pooled: 0` and need to be
    told apart.

    Two things the sb3 path does not have to think about: the buffer is
    `(n_env, rows)` so a flat pool is laid back time-major, the exact inverse
    of the export (`_fasttd3_prefill_replay`); and `num_steps > 1` is refused
    rather than degraded, because an n-step window over pooled rows would sum
    rewards across unrelated transitions.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError(
            "train.backend: fasttd3 needs torch, which is not installed. Either\n"
            "    uv sync --extra fasttd3\n"
            "or run the same method point under the tester profile (`--profile tester`),\n"
            "which sets\n"
            "    train.backend: mock\n"
            f"(underlying import error: {exc})"
        ) from exc

    cfg, env = ctx.cfg, ctx.env
    t0 = time.monotonic()
    result = TrainResult(cand_id=candidate.cand_id, candidate=candidate)

    # Config-level refusals, not candidate failures: these hold for every
    # candidate of every iteration, so failing candidates one by one would
    # burn the whole search to report a config error.
    if getattr(env, "exact_states", None) is not None:
        raise ValueError(
            "train.backend=fasttd3 needs a continuous-action env (a tanh policy "
            f"has no discrete head); {getattr(env, 'name', '?')} is discrete. "
            "Use train.backend: tabular or mock there.")
    arch = _resolve_arch(cfg)

    try:
        # `train.reward_scaling` between compilation and training, as
        # `_run_backend` and `_sb3_run` do: a pure APPLY of the plan stage 3
        # left on `candidate.meta` in the parent (`population.
        # scaling_elite_moments`, `bird.py::train`); `none` plans nothing and
        # is the identity. What this seed trained under goes on the seed row
        # as `reward_scaling` (`_scaling_record`), under a non-`none` key only.
        reward = _training._scaled_reward(
            ctx, state, candidate,
            _training._training_reward(ctx, state, candidate))
    except Exception as exc:  # noqa: BLE001
        ctx.budget.record_training(env_steps=0, gpu_seconds=time.monotonic() - t0)
        result.trained, result.error = False, f"{type(exc).__name__}: {exc}"
        return result

    # The co-designed observation -- same refusal as `_sb3_run`: no silent fall
    # back to the raw state, or a co-designing method quietly becomes its
    # reward-only ablation.
    try:
        _view, phi = _install_observation(ctx, candidate, env)
    except Exception as exc:  # noqa: BLE001
        ctx.budget.record_training(env_steps=0, gpu_seconds=time.monotonic() - t0)
        result.trained, result.error = False, f"{type(exc).__name__}: {exc}"
        candidate.meta["observation_installed"] = False
        candidate.meta["observation_error"] = result.error
        return result

    def _features(s: Any) -> np.ndarray:
        return np.asarray(phi(s) if phi is not None else s, dtype=np.float32)

    ns = _torch_classes()
    torch = ns["torch"]
    nn = ns["nn"]

    hyper = _fasttd3_hyper(cfg)
    dev_name = str(hyper["device"] or "cpu")
    requested = dev_name
    if dev_name == "auto":
        # `auto` KEEPS its fallback: it means "use a GPU if there is one", so
        # running on cpu is the answer rather than a disappointment.
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_name.startswith("cuda") and not torch.cuda.is_available():
        # AN EXPLICIT `cuda` IS A REFUSAL, not a downgrade with a warning.
        #
        # A warning in a batch job's stdout is not read, and the downgrade
        # takes more than the device with it: `amp_enabled` and `compile_on`
        # are both derived from `device.type == "cuda"` immediately below, so
        # a "GPU" job would quietly become a cpu job WITHOUT autocast and
        # WITHOUT torch.compile, while metadata built from a probe tensor
        # could still report a live cuda device.
        #
        # Asking for cuda and silently getting cpu is a measurement that looks
        # like the one you wanted, which is the failure this repo exists to
        # avoid. `auto` is how you ask for the fallback.
        raise RuntimeError(
            f"train.hyperparameters.device={requested!r} but torch.cuda.is_available() "
            "is False on this machine. Refused rather than run on cpu: the fallback "
            "also disables AMP and torch.compile (both are `device.type == 'cuda'` "
            "below), so the run would be several times slower than the number it "
            "would be compared against, and would say so nowhere a reader looks. "
            "Use device: auto to accept a cpu fallback deliberately, or fix the "
            "allocation -- under a cluster scheduler this usually means the job "
            "was allocated no GPU.")
    device = torch.device(dev_name)
    # amp_enabled exactly as train.py:58 computes it: requested AND on cuda, so
    # the default `amp: true` is inert on the repo-default cpu device.
    amp_enabled = bool(hyper["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(hyper["amp_dtype"]) == "bf16" else torch.float16
    compile_on = (bool(hyper["compile"]) if hyper["compile"] is not None
                  else device.type == "cuda")

    # sb3-backend features a config can enable that this learner cannot honour:
    # train without them and SAY SO -- the kl_clone-on-non-PPO shape, and the
    # runtime echo of what `_check_coherence` cannot know (it has no backend
    # x feature matrix). `anchor: {}` in the seed rows below is the artifact's
    # side of the same statement.
    if str(cfg.get("train.anchor.kind", "none") or "none") != "none":
        log.warning("train.anchor.kind=%s is implemented on the sb3 backend only "
                    "(it clones an SB3 policy); train.backend=fasttd3 trains "
                    "unanchored", cfg.get("train.anchor.kind"))
    if bool(cfg.get("train.reference_policy.enabled", False)):
        log.warning("train.reference_policy is implemented on the sb3 backend only; "
                    "train.backend=fasttd3 trains unregularised")

    num_envs = max(1, int(hyper["num_envs"]))
    n_obs = int(phi.dim) if phi is not None else int(np.asarray(env.obs_low).size)
    n_act = int(np.asarray(env.action_low).size)
    norm_mode = cfg.get("train.reward_norm", "none")

    dr_params = _dr_params(cfg, candidate)
    env.set_dr(dr_params)
    total_steps = int((env_steps if env_steps is not None
                       else cfg.get("train.env_steps", 20000)) or 20000)
    iters = max(1, total_steps // num_envs)
    # `train.n_parallel_envs` (hash-excluded, a schedule): forked slot workers
    # behind `_VecEnvView`. 1 = the in-process loop.
    env_workers = max(1, int(cfg.get("train.n_parallel_envs", 1) or 1))
    # The buffer is `(num_envs, rows)` on the device and `extend` is called once
    # per iteration, so rows beyond `iters + 1` are never written and never
    # sampled (`sample` draws below `min(rows, ptr)`; the wrap branch needs
    # `ptr >= rows`). Capping is therefore bit-neutral and takes the published
    # 51,200-row buffer from 10.8-16.5 GiB per training (measured on h1hand,
    # obs 167) to ~2 GiB at the 1M-step / 128-env budget. The
    # published `buffer_size` is still what the config says; `buffer_rows`
    # on the seed row is what was allocated.
    buffer_rows = min(int(hyper["buffer_size"]), iters + 1)

    # -- the replay half of a shared_population slice (LaRes §4.3) ----------
    # Resolved ONCE here, against the round store the driver filled before this
    # wave, and handed to every seed below -- `_sb3_run`'s shape, and each
    # refusal is a warning plus a cold buffer, never a crash, recorded in the
    # seed row so a slice that declared a shared buffer and got none is visible.
    #
    # THE BUFFER IS WIDENED FOR THE POOL, and the cap above is why that needs
    # saying: `iters + 1` is exactly the rows TRAINING will write, so prefilling
    # into it would make the run wrap and overwrite the pool it was given. The
    # widened figure stays bounded by the published `buffer_size`, which is both
    # the memory argument above and LaRes's own rule that capacity is the
    # learner's (`train.hyperparameters.buffer_size`); a pool bigger than the
    # headroom is truncated OLDEST first inside `_fasttd3_prefill_replay`.
    #
    # THE n-STEP REFUSAL IS DECIDED HERE, NOT PER SEED. `_fasttd3_prefill_replay`
    # still refuses `num_steps > 1` -- it is reachable directly and has its own
    # test -- but deciding it only there would mean the parent had ALREADY
    # widened `buffer_rows` for a pool that would never be written, and the seed
    # row would read `replay_prefill_supported: True` with
    # `replay_prefill_pooled: 0`: a pool that did not execute, left to be
    # INFERRED from a count instead of stated. Refusing up here skips
    # the widening and lets the row say so, with the same meaning
    # `replay_prefill_supported` carries on sb3 (`bool(off_policy)`): a statement
    # about what this learner CONFIGURATION can honour, not about what happened.
    slice_index = int(getattr(handoff, "index", 0) or 0)
    prefill_rows, prefill_ref, prefill_supported = _fasttd3_prefill_plan(
        int(hyper["num_steps"]), handoff, phi)
    if prefill_rows is not None:
        want_cols = -(-len(prefill_rows) // num_envs)     # ceil
        headroom = max(0, int(hyper["buffer_size"]) - buffer_rows)
        buffer_rows += min(want_cols, headroom)
    # Same checkpoint count rule as `_sb3_run` (on the interaction budget, so a
    # config moved between backends keeps its curve length); each chunk is that
    # share of ITERATIONS.
    n_chunks = max(MIN_CHECKPOINTS,
                   min(MAX_CHECKPOINTS, total_steps // 2048 or MIN_CHECKPOINTS))
    # The chunk plan PARTITIONS the iteration budget exactly (never `max(1,
    # iters // n_chunks)` per chunk): on a budget smaller than the checkpoint
    # count that formula trains past `train.env_steps` and steps the cosine
    # schedulers past T_max. A chunk can be empty; its checkpoint
    # still evaluates, so the curve keeps `n_chunks` rows on every budget.
    chunk_plan = [iters // n_chunks + (1 if ck < iters % n_chunks else 0)
                  for ck in range(n_chunks)]
    base_seed = _seed_base(ctx, state, candidate, seed_phase)
    rule, prune_field = _pruner_for(cfg)
    prune_metric = str(cfg.get("train.pruning_metric", "own_reward") or "own_reward")
    select_rule, select_min_delta = _selection_for(cfg)
    # The demonstration ceiling, same call and same place as `_sb3_run`: under
    # this candidate's DR, once per candidate, before any seed forks.
    ceiling = _demo_ceiling(ctx, state, candidate, reward) if _ceiling_wanted(cfg) else None
    timeout_s = float(cfg.get("train.timeout_s", 3600) or 3600)
    if iters < int(hyper["learning_starts"]) + 2:
        # A short screen (`screens._train_backend_short`) or a tiny budget can
        # sit entirely inside the warm-up: the run then measures the untrained
        # init, which is a true statement about FastTD3 under that budget --
        # but only if somebody can see it happened.
        log.warning("fasttd3: %d iteration(s) (%d env steps / %d envs) with "
                    "learning_starts=%s -- no gradient step will run; lower "
                    "train.hyperparameters.num_envs or learning_starts for "
                    "short budgets", iters, total_steps, num_envs,
                    hyper["learning_starts"])

    # -- train.init, off the same state slots as every backend ---------------
    init = _training_init(ctx, state, cfg, resume_ref, candidate)
    init_ref, secondary, sec_ratio = init.ref, init.secondary, init.ratio
    result.init_source, result.init_from_cand_id = init.source, init.from_cand_id
    result.init_similarity = init.similarity
    init_blob = _POLICY_STORE.get(init_ref or "") if init_ref else None
    use_secondary = bool(sec_ratio > 0.0 and len(secondary or ()))
    if phi is not None and use_secondary:
        # Verbatim the `_sb3_run` refusal, for verbatim the reason: the shared
        # rows are states, this candidate's networks eat phi(s), and every
        # candidate's phi is its own.
        log.warning("train.init=secondary_replay_buffer and problem.search_space "
                    "with 'observation' are mutually exclusive on fasttd3: the "
                    "networks train in this candidate's feature space and the "
                    "shared rows are states. The shared buffer is disabled for "
                    "this candidate; the search continues.")
        use_secondary = False
    if init_ref and init_blob is None:
        log.warning("train.init=%s: %s is not in _POLICY_STORE "
                    "(evicted, or written by another process) -- training from "
                    "scratch", cfg.get("train.init", "from_scratch"), init_ref)

    per_seed_curves: List[List[Dict[str, float]]] = []
    per_seed_components: List[List[Dict[str, float]]] = []
    want_components = "reward_component_values" in set(cfg.get("train.log", []) or [])
    policies: List[Callable[[np.ndarray], np.ndarray]] = []
    used = 0
    fatal_error = ""

    gamma = float(hyper["gamma"])

    def _build_agent(seed: int) -> Dict[str, Any]:
        """Networks + normalizers, seeded: `torch.manual_seed` covers every
        torch draw (init, `noise_scales`, exploration, batch indices), which is
        what makes two runs of one seed identical. train.py:76-79 also seeds
        the global `random`/`np.random` streams; this port deliberately does
        not -- nothing here draws from them, and a backend that reseeded the
        process-global numpy stream would perturb every OTHER component's
        draws, which no backend may do."""
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = bool(hyper["torch_deterministic"])
        actor, qnet, qnet_target, obs_normalizer = _build_networks(
            ns, arch, hyper, n_obs, n_act, num_envs, device)
        qnet_target.load_state_dict(qnet.state_dict())
        reward_normalizer = None
        if bool(hyper["reward_normalization"]):
            # g_max off the support exactly as train.py:167-171 sizes it.
            reward_normalizer = ns["RewardNormalizer"](
                gamma=gamma, device=device,
                g_max=min(abs(float(hyper["v_min"])), abs(float(hyper["v_max"]))))
        return {"actor": actor, "qnet": qnet, "qnet_target": qnet_target,
                "obs_normalizer": obs_normalizer,
                "reward_normalizer": reward_normalizer}

    def _norm_obs(agent: Dict[str, Any], x: Any, update: bool = True) -> Any:
        norm = agent["obs_normalizer"]
        return x if norm is None else norm(x, update=update)

    def _policy_fn(agent: Dict[str, Any], view: _VecEnvView
                   ) -> Callable[[np.ndarray], np.ndarray]:
        """Greedy policy over RAW states, for `_evaluate_policy`/`_rollout`:
        feature, normalise WITHOUT updating (train.py:346-349), act, map to env
        space. `_single_thread_torch` at the call sites keeps the forked
        schedule's parent predicts at the children's thread count."""
        # `_greedy_policy`: the one action path, shared with the rebuilt policy
        # (`fasttd3_policy_from_blob`), so what is rolled out after training is
        # what was driven during it.
        return _greedy_policy(ns, agent["actor"], agent["obs_normalizer"], _features,
                              view.to_env_action, device)

    # ---------------------------------------------------------------- seeds --

    def run_seed(seed_i: int) -> Dict[str, Any]:
        """Train ONE seed. Same contract as `_sb3_run.run_seed`: pure of
        `result`, the accumulators and `ctx.budget`, so the body runs inline or
        inside a forked seed worker with the fold in the parent either way. A
        raise out of this body aborts the whole backend call."""
        seed = _seed_for(base_seed, seed_i,
                         int(getattr(handoff, "index", 0) or 0))
        seed_t0 = time.monotonic()
        # ONE torch thread, whichever schedule is running. Upstream pins
        # OMP_NUM_THREADS=1 before importing torch (train.py:4-5), so this is
        # the faithful CPU configuration -- and it is also what makes the seed
        # fork bit-identical to the sequential loop: a forked child trains at
        # one thread (`_pin_one_thread`), and an unpinned sequential parent
        # diverges from it by ~1e-6 per checkpoint return (thread-count
        # reduction order; measured on the parity test).
        with _single_thread_torch():
            return _run_one_seed(seed, seed_t0)

    def _run_one_seed(seed: int, seed_t0: float) -> Dict[str, Any]:
        import torch  # local: the fork path re-enters here in a fresh child
        from torch.amp import GradScaler, autocast

        # Before the agent: `_VecEnvView` forks its slot workers at construction
        # (`train.n_parallel_envs`), and a fork should precede this process's
        # first CUDA allocation.
        #
        # THE ADAPTER'S CAPABILITY CHOOSES THE VIEW, not a config flag:
        # an env that can step a batch on device gets the device view, and the
        # science knob stays `problem.env_id`. A `train.use_jax`-shaped key
        # would be a second way to say what the env already says, and a way
        # for the two to disagree.
        #
        # THIS IS THE ONLY CONSTRUCTION SITE THAT SWAPS. The other one
        # (`view2`, below, when rebuilding a trained policy from a blob) is an
        # action map that is "never reset or stepped" by its own comment, so a
        # device view there would pay a jit and a device round trip for
        # arithmetic that never touches the GPU.
        if isinstance(env, BatchedEnvAdapter):
            if int(env_workers or 0) > 1:
                log.warning(
                    "train.n_parallel_envs=%s is ignored on the jax family: "
                    "%s steps %d rows inside one jitted region, with no slot "
                    "workers and no pipes. The seed row records "
                    "`n_parallel_envs_ignored: true` so this is auditable in "
                    "the artifact and not only here.",
                    env_workers, type(env).__name__, num_envs)
            view = _JaxVecEnvView(env, num_envs, reward, _features, norm_mode,
                                  seed, n_obs)
        else:
            view = _VecEnvView(env, num_envs, reward, _features, norm_mode, seed,
                               n_obs, workers=env_workers)
        agent = _build_agent(seed)
        actor, qnet, qnet_target = agent["actor"], agent["qnet"], agent["qnet_target"]
        rew_norm = agent["reward_normalizer"]

        # AdamW + cosine annealing over the ITERATION count, stepped once per
        # iteration whether or not learning has started -- train.py:292-313 and
        # :745-747. `weight_decay` applies to both nets (upstream passes
        # `args.weight_decay` to both optimizers).
        # Learning rates as TENSORS, as train.py:292-313 passes them -- under
        # `compile` + cudagraphs the optimizer step is captured and a python
        # float would bake the initial LR into the graph.
        q_optimizer = torch.optim.AdamW(
            qnet.parameters(),
            lr=torch.tensor(float(hyper["critic_learning_rate"]), device=device),
            weight_decay=float(hyper["weight_decay"]))
        actor_optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=torch.tensor(float(hyper["actor_learning_rate"]), device=device),
            weight_decay=float(hyper["weight_decay"]))
        amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

        warm_started = bool(init_blob is not None
                            and _fasttd3_apply(ns, arch, actor, qnet, qnet_target,
                                               agent["obs_normalizer"], init_blob,
                                               init_ref or ""))
        # LaRes Eq. 4 (`train.elite_constraint.kind: l2_params`), `_sb3_run`'s
        # block for this backend. The reference is the ELITE's stored
        # parameters (`elite_desc["reference_policy_ref"]` = `state.policy_ref`,
        # fixed for the round), read off `_POLICY_STORE` and held in the live
        # networks while `attach_l2_to_optimisers` snapshots
        # (`_fasttd3_attach_elite_constraint`) -- NOT a snapshot of whatever
        # `train.init` just loaded, which on a resumed `shared_population`
        # slice is this arm's own previous slice. The actor's and the critic's
        # optimisers both: the term is on "the actor and both critics" and
        # `qnet` holds both heads. Still gated on `warm_started`, because LaRes
        # constrains the agents it re-initialised from the elite (Alg. 1 lines
        # 10, 15) and the coherence rule pairs the two keys. `n_constrained`
        # and `constraint_ref` go on the seed row rather than into the journal,
        # because this body may run in a forked seed worker with no
        # `ctx.rundir`; 0 beside a non-`none` key is the declared-but-not-
        # running case and must stay visible.
        n_constrained = 0
        constraint_ref = ""
        if str(cfg.get("train.elite_constraint.kind", "none") or "none") != "none":
            from .. import registry as _reg
            elite_desc = _reg.get("elite_constraint",
                                  cfg["train.elite_constraint.kind"])(ctx, state, candidate)
            if elite_desc is not None and warm_started:
                elite_ref = str(elite_desc.get("reference_policy_ref") or "")
                ref_blob = _POLICY_STORE.get(elite_ref) if elite_ref else None
                if ref_blob is None:
                    log.warning("train.elite_constraint.kind=l2_params: the elite's "
                                "parameters (%s) are not in _POLICY_STORE, so there is "
                                "nothing to constrain toward; training unconstrained",
                                elite_ref)
                else:
                    n_constrained = _fasttd3_attach_elite_constraint(
                        ns, arch, actor, qnet, [actor_optimizer, q_optimizer],
                        float(elite_desc["weight"]), ref_blob, elite_ref)
                    if n_constrained == 0:
                        log.warning("train.elite_constraint.kind=l2_params found no "
                                    "optimiser to constrain on this agent; the term is "
                                    "NOT running")
                    else:
                        constraint_ref = elite_ref
            elif elite_desc is not None:
                log.warning("train.elite_constraint.kind=l2_params: no elite parameters "
                            "were loaded for this candidate, so there is nothing to "
                            "constrain toward; training unconstrained")
        # The schedulers AFTER the constraint: `LRScheduler.__init__` wraps
        # `optimizer.step` to check call order, and a step patched after that
        # wrap makes torch warn on every scheduler tick that the optimiser was
        # "overridden after learning rate scheduler initialization".
        q_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            q_optimizer, T_max=iters,
            eta_min=torch.tensor(float(hyper["critic_learning_rate_end"]), device=device))
        actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            actor_optimizer, T_max=iters,
            eta_min=torch.tensor(float(hyper["actor_learning_rate_end"]), device=device))

        rb = ns["SimpleReplayBuffer"](
            n_env=num_envs, buffer_size=buffer_rows, n_obs=n_obs,
            n_act=n_act, n_steps=int(hyper["num_steps"]), gamma=gamma, device=device)

        # The round pool, relabelled under THIS candidate's reward, before any
        # gradient step -- so the learner samples uniformly over pool + this
        # slice's new rows, which is the release's single `All_buffer`
        # (`_sb3_prefill_replay`'s argument, unchanged here).
        prefilled = 0
        if prefill_rows is not None:
            prefilled = _fasttd3_prefill_replay(
                ns, view, rb, prefill_rows, reward, norm_mode, device,
                num_envs, int(hyper["num_steps"]))

        secondary_rows: Optional[Dict[str, Any]] = None
        sec_gen = None
        if use_secondary:
            secondary_rows = _fasttd3_secondary(ns, view, secondary, reward,
                                                norm_mode, device)
            if secondary_rows is not None:
                sec_gen = torch.Generator(device="cpu")
                sec_gen.manual_seed(seed + 65537)
        secondary_used = int(secondary_rows["n"]) if secondary_rows is not None else 0

        policy_noise = float(hyper["policy_noise"])
        noise_clip = float(hyper["noise_clip"])
        tau = float(hyper["tau"])
        num_updates = int(hyper["num_updates"])
        policy_frequency = int(hyper["policy_frequency"])
        # `learning_starts` is paid ONCE per arm per round, as on sb3 -- but
        # the UNIT here is ITERATIONS, not transitions (`learning_starts: 10`
        # is 10 * num_envs steps, hyperparams.py:61), so the prefill converts
        # at the lane width rather than row-for-row. Without this a resumed
        # slice re-spends the warm-up on a buffer that already holds it, which
        # on a 5-slice round is five warm-ups for one agent.
        learning_starts = int(hyper["learning_starts"])
        learning_starts_effective = max(0, learning_starts - (prefilled // num_envs))
        learning_starts = learning_starts_effective
        use_cdq = bool(hyper["use_cdq"])
        disable_bootstrap = bool(hyper["disable_bootstrap"])
        use_clip = bool(hyper["use_grad_norm_clipping"])
        max_grad_norm = float(hyper["max_grad_norm"])
        batch_per_env = max(1, int(hyper["batch_size"]) // num_envs)

        def _splice_secondary(data: Dict[str, Any]) -> Dict[str, Any]:
            """`train.secondary_buffer.ratio` of EVERY batch from the shared
            rows -- `_mixed_buffer_class`'s rule (a pre-seeded buffer's share
            would decay as the primary fills). Runs BEFORE normalisation, so
            the spliced rows are normalised and reward-scaled exactly like the
            rows they replace. Spliced rows are 1-step by construction."""
            if secondary_rows is None:
                return data
            total = int(data["rewards"].shape[0])
            n2 = max(0, min(total, int(round(float(sec_ratio) * total))))
            if n2 == 0:
                return data
            idx = torch.randint(0, secondary_rows["n"], (n2,), generator=sec_gen)
            idx = idx.to(device)
            data["observations"][:n2] = secondary_rows["observations"][idx]
            data["actions"][:n2] = secondary_rows["actions"][idx]
            data["next_observations"][:n2] = secondary_rows["next_observations"][idx]
            data["rewards"][:n2] = secondary_rows["rewards"][idx]
            data["dones"][:n2] = secondary_rows["dones"][idx]
            data["truncations"][:n2] = 0
            data["effective_n_steps"][:n2] = 1
            return data

        def update_main(data: Dict[str, Any]) -> None:
            """The critic step, train.py:406-490 (symmetric-observation form)."""
            with autocast(device_type=amp_device_type, dtype=amp_dtype,
                          enabled=amp_enabled):
                observations = data["observations"]
                next_observations = data["next_observations"]
                actions = data["actions"]
                rewards = data["rewards"]
                dones = data["dones"].bool()
                truncations = data["truncations"].bool()
                if disable_bootstrap:
                    bootstrap = (~dones).float()
                else:
                    bootstrap = (truncations | ~dones).float()
                clipped_noise = torch.randn_like(actions).mul(policy_noise).clamp(
                    -noise_clip, noise_clip)
                next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                    -1.0, 1.0)
                discount = gamma ** data["effective_n_steps"].float()
                with torch.no_grad():
                    q1_proj, q2_proj = qnet_target.projection(
                        next_observations, next_state_actions, rewards, bootstrap,
                        discount)
                    q1_val = qnet_target.get_value(q1_proj)
                    q2_val = qnet_target.get_value(q2_proj)
                    if use_cdq:
                        target_dist = ns["torch"].where(
                            q1_val.unsqueeze(1) < q2_val.unsqueeze(1), q1_proj, q2_proj)
                        q1_target_dist = q2_target_dist = target_dist
                    else:
                        q1_target_dist, q2_target_dist = q1_proj, q2_proj
                qf1, qf2 = qnet(observations, actions)
                qf1_loss = -torch.sum(
                    q1_target_dist * ns["F"].log_softmax(qf1, dim=1), dim=1).mean()
                qf2_loss = -torch.sum(
                    q2_target_dist * ns["F"].log_softmax(qf2, dim=1), dim=1).mean()
                qf_loss = qf1_loss + qf2_loss
            q_optimizer.zero_grad(set_to_none=True)
            scaler.scale(qf_loss).backward()
            scaler.unscale_(q_optimizer)
            if use_clip:
                torch.nn.utils.clip_grad_norm_(
                    qnet.parameters(),
                    max_norm=max_grad_norm if max_grad_norm > 0 else float("inf"))
            scaler.step(q_optimizer)
            scaler.update()

        def update_pol(data: Dict[str, Any]) -> None:
            """The delayed policy step, train.py:492-525."""
            with autocast(device_type=amp_device_type, dtype=amp_dtype,
                          enabled=amp_enabled):
                qf1, qf2 = qnet(data["observations"], actor(data["observations"]))
                qf1_value = qnet.get_value(ns["F"].softmax(qf1, dim=1))
                qf2_value = qnet.get_value(ns["F"].softmax(qf2, dim=1))
                if use_cdq:
                    qf_value = torch.minimum(qf1_value, qf2_value)
                else:
                    qf_value = (qf1_value + qf2_value) / 2.0
                actor_loss = -qf_value.mean()
            actor_optimizer.zero_grad(set_to_none=True)
            scaler.scale(actor_loss).backward()
            scaler.unscale_(actor_optimizer)
            if use_clip:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(),
                    max_norm=max_grad_norm if max_grad_norm > 0 else float("inf"))
            scaler.step(actor_optimizer)
            scaler.update()

        @torch.no_grad()
        def soft_update(src: Any, tgt: Any) -> None:
            """train.py:527-533, `torch._foreach_*` form."""
            src_ps = [p.data for p in src.parameters()]
            tgt_ps = [p.data for p in tgt.parameters()]
            torch._foreach_mul_(tgt_ps, 1.0 - tau)
            torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

        def normalize_obs(x: Any, update: bool = True) -> Any:
            return _norm_obs(agent, x, update=update)

        # Unconditional per iteration, as upstream's mark_step() is
        # (train.py:579, fast_td3_utils.py:811-813) -- a no-op without
        # cudagraphs in flight.
        mark_step = getattr(getattr(torch, "compiler", None),
                            "cudagraph_mark_step_begin", None)
        u_main, u_pol = update_main, update_pol
        if compile_on:
            # The two update closures under the configured mode (train.py:536-538).
            # Upstream ALSO compiles the exploration policy and the normalizer
            # under mode=None (train.py:539-541); this port leaves those eager --
            # a declared deviation (module docstring): they are a no_grad forward
            # pass per iteration against `num_updates` full backward passes, and
            # compiling them would add warm-up on a path no test here exercises.
            u_main = torch.compile(update_main, mode=str(hyper["compile_mode"]))
            u_pol = torch.compile(update_pol, mode=str(hyper["compile_mode"]))

        curve: List[Dict[str, float]] = []
        comp_curve: List[Dict[str, float]] = []
        spent = 0
        spent_eval = 0
        seed_error = ""
        pruned_at = None
        timed_out = False
        best_snap: Optional[Dict[str, Any]] = None
        best_snap_idx = -1
        ceiling_decisions: List[Dict[str, Any]] = []
        pruner = _ceiling_guard(cfg, rule, prune_metric, ceiling, ceiling_decisions)

        eval_policy = _policy_fn(agent, view)
        obs_t = torch.as_tensor(view.reset(), device=device)
        dones_prev: Any = None
        global_it = 0
        # Utilisation samples are taken DURING the run rather than at the end:
        # after the loop stops the GPU is idle, and an idle reading is
        # reassuring and wrong. None off cuda and on every failure path,
        # because "could not tell" is not "was idle".
        # K SAMPLES, NOT ONE, AND SPREAD OVER THE RUN. A single mid-run sample
        # can read 0 on a run that holds 33-58% through its whole stepping
        # phase, because on a cold JIT backend the midpoint chunk can still be
        # inside compilation -- the one window in which the true answer IS
        # zero (measured with a 30 s poll alongside: 0% on all 25 samples
        # through the compile phase, then 33-58% sustained).
        #
        # Still point samples, and still cheap: one `nvidia-smi` subprocess
        # each, k of them across the run, and NOT the `dmon` window the
        # function's docstring rejects (30 s of blocking per seed, inside the
        # hot path, for a field nothing reads to decide anything).
        util = _UtilSampler(n_chunks, enabled=(device.type == "cuda"))
        if device.type == "cuda":
            # A cold peak for THIS seed: without the reset the figure is
            # whatever the process high-water mark already was, which on a
            # multi-seed call is the previous seed's.
            with contextlib.suppress(Exception):
                torch.cuda.reset_peak_memory_stats(device)

        for ck in range(n_chunks):
            util.maybe_sample(ck)
            for _ in range(chunk_plan[ck]):
                if mark_step is not None:
                    # train.py:579 `mark_step()`: once per iteration, before any
                    # compiled function, for cudagraph replay correctness.
                    mark_step()
                with torch.no_grad(), autocast(device_type=amp_device_type,
                                               dtype=amp_dtype, enabled=amp_enabled):
                    norm_obs = normalize_obs(obs_t)
                    actions = actor.explore(norm_obs, dones=dones_prev)
                next_obs, rewards, dones, time_outs, true_next = view.step(
                    actions.float())
                rewards_t = torch.as_tensor(rewards, device=device)
                dones_t = torch.as_tensor(dones, device=device)
                truncs_t = torch.as_tensor(time_outs, device=device)
                if rew_norm is not None:
                    rew_norm.update_stats(rewards_t, dones_t.float())
                rb.extend(obs_t, actions.float(), rewards_t, dones_t, truncs_t,
                          torch.as_tensor(true_next, device=device))
                obs_t = torch.as_tensor(next_obs, device=device)
                dones_prev = dones_t
                if global_it > learning_starts:
                    for i in range(num_updates):
                        data = rb.sample(batch_per_env)
                        data = _splice_secondary(data)
                        data["observations"] = normalize_obs(data["observations"])
                        data["next_observations"] = normalize_obs(
                            data["next_observations"])
                        if rew_norm is not None:
                            data["rewards"] = rew_norm(data["rewards"])
                        u_main(data)
                        # train.py:669-674: with several updates per iteration
                        # the delay counts UPDATES; with one it counts iterations.
                        if num_updates > 1:
                            if i % policy_frequency == 1:
                                u_pol(data)
                        else:
                            if global_it % policy_frequency == 0:
                                u_pol(data)
                        soft_update(qnet, qnet_target)
                global_it += 1
                # BOTH SCHEDULERS STEP OUTSIDE THE `learning_starts` GUARD,
                # AND THAT IS UPSTREAM'S PLACEMENT, NOT AN OVERSIGHT.
                # `train.py:744-746` puts `global_step += 1` and both
                # `scheduler.step()` calls at the same indentation as its
                # `if global_step > args.learning_starts:` at `:643`, exactly
                # as here. The consequence: with `learning_starts=10` the first
                # update runs
                # at `global_it == 11`, so ELEVEN cosine steps are consumed
                # before any gradient step and the initial LR is never
                # applied -- about 0.13% of the schedule on a 1M-step
                # training. torch says so out loud, twice per training, one
                # warning per scheduler instance: "Detected call of
                # `lr_scheduler.step()` before `optimizer.step()`".
                #
                # LEAVE IT. Moving these inside the guard would train under a
                # schedule the released code does not use, which is a method
                # change wearing a bug fix's clothes; the port matches
                # upstream, and a deliberate departure is a config value with
                # its own before/after, not an indent.
                actor_scheduler.step()
                q_scheduler.step()
            spent = global_it * num_envs

            metrics, comps, ev_steps, _tr, ev_wall = _training._evaluate_policy_timed(
                env, eval_policy, np.random.default_rng(seed + ck), reward, 3,
                _error_router("exception_soft", io.StringIO()))
            spent_eval += ev_steps
            comp_curve.append(comps if want_components else {})
            curve.append({"step": float(spent), "round": float(ck + 1),
                          "seed": float(seed), "fitness": metrics["fitness"],
                          "score": metrics["fitness"], "task_success": metrics["fitness"],
                          "consecutive_successes": metrics["fitness"],
                          "success_rate": metrics["success_rate"],
                          "gt_return": metrics["gt_return"],
                          "return": metrics["reward_return"],
                          "reward_return": metrics["reward_return"],
                          # Evaluation wall clock: on the jax tier this loop is
                          # a large fixed cost. Per checkpoint, because the
                          # first one carries MJX compile and a single total
                          # would hide that.
                          # WALL-CLOCK: listed in `tests/test_parallelism._VOLATILE`. A timing
                          # field that is not listed there breaks bit-identity by construction --
                          # parallel candidates contend, so it differs from sequential on every
                          # run, and the failure surfaces as "parallel changed
                          # candidates/.../train_result.json" three files from the cause.
                          "eval_wall_s": ev_wall})
            if ceiling is not None:
                curve[-1]["demo_fraction"] = _demo_fraction_point(
                    metrics["reward_return"], ceiling, [p["reward_return"] for p in curve[:-1]])
            if _running_best(curve, select_rule):
                best_snap = _snapshot_agent(agent)
                best_snap_idx = len(curve) - 1
            if pruner([p[prune_field] for p in curve], spent / max(1, total_steps)):
                pruned_at = ck + 1
                break
            # `ck + 1 < n_chunks` for `_sb3_run`'s reason: a seed that finished
            # its whole budget must not be marked failed for crossing the wall
            # clock on the very last chunk.
            if ck + 1 < n_chunks and time.monotonic() - seed_t0 > timeout_s:
                seed_error = f"timeout after {timeout_s:g}s"
                timed_out = True
                break

        # `train.checkpoint_selection`, restored into the LIVE agent inside
        # run_seed for `_sb3_run`'s reason: the rollouts, the blob a child
        # serialises and `_POLICY_STORE` all see the same weights for free.
        ship_idx, select_reason = _training._select_checkpoint(
            curve, select_rule, select_min_delta)
        if ship_idx is not None:
            if best_snap is None or best_snap_idx != ship_idx:
                ship_idx, select_reason = None, "no_snapshot"
            elif not _restore_agent(agent, best_snap):
                ship_idx, select_reason = None, "restore_failed"

        wall = float(time.monotonic() - seed_t0)
        # Rows THIS slice added, for the seed row and for the export cap.
        # Read off the buffer rather than derived from the iteration count so
        # a pruned or timed-out seed reports what it actually wrote; once the
        # ring has wrapped the prefill has been partly overwritten, and the
        # subtraction can only under-report, never over-.
        rows_added = max(0, min(int(rb.ptr), int(rb.buffer_size)) * int(num_envs)
                         - int(prefilled))
        gpu_peak_mib = None
        if device.type == "cuda":
            try:
                gpu_peak_mib = round(
                    torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1)
            except Exception:  # noqa: BLE001 - provenance must not fail a run
                gpu_peak_mib = None
        fits = [p["fitness"] for p in curve] or [0.0]
        ship_i = ship_idx if ship_idx is not None else len(fits) - 1
        seed_metric = {
            "seed": int(seed), "fitness": float(fits[ship_i]), "final": float(fits[ship_i]),
            "max": float(np.max(fits)), "auc": float(np.mean(fits)),
            "env_steps": int(spent + spent_eval), "train_steps": int(spent),
            "train_steps_requested": int(total_steps),
            "wallclock_s": wall,
            "n_checkpoints": len(curve), "pruned_at_round": pruned_at,
            "timed_out": bool(timed_out), "learner": "fasttd3",
            # `algorithm_cited`, NOT `algorithm`, AND IT IS NOT THE LEARNER.
            # `train.algorithm` "records the algorithm the method's paper
            # used and is kept as a citation even when no backend here can
            # run it" (schema.py, and _default.yaml says it again) -- its enum
            # includes `q_learning` and `none` precisely because nothing here
            # executes those. So a row reading `ppo` beside `learner: fasttd3`
            # is two TRUE facts, and a bare `algorithm` name would make them
            # ambiguous; overwriting it with the resolved learner would delete
            # a citation from configs whose whole point is that published
            # values stay citable. The name says which kind of fact it is;
            # `learner` and `backend` remain the execution facts.
            #
            # `cfg[...]`, not `cfg.get(..., "ppo")`: the key is always in
            # `_default.yaml`, so the fallback was dead -- and if it ever
            # stopped being dead the row would silently claim PPO for a
            # method that cited something else.
            "algorithm_cited": cfg["train.algorithm"], "backend": "fasttd3",
            "architecture": arch, "num_envs": int(num_envs), "device": str(device),
            # WHERE THE PARAMETERS ACTUALLY LIVE, read off the actor rather
            # than from the config or a probe tensor. `device` above is what
            # this function resolved; this is what the network is on, and the
            # two can only differ if something moved the model -- in which
            # case the artifact should say which one the gradients ran on.
            # (A probe tensor allocated on cuda can report a GPU while the
            # learner trains on cpu.)
            "learner_device": _param_device(agent),
            # WHICH SOURCE SUPPLIED THE TASK CATALOGUE, not whether one did.
            # Same argument as `learner_device` directly above: the value
            # resolved fine, three candidates could have produced it
            # (a checkout, $BIRD_DATA_ROOT, the shipped copy), and which one
            # did was recorded nowhere -- so a run reading a stale or
            # different-but-equivalent catalogue is invisible until the day
            # the sources stop agreeing.
            #
            # `tasks` specifically, because that is the root a job needs and
            # the one whose absence stops it (see `bird/paths.py`).
            "data_root_source": _data_root_source(),
            # THE INSTRUMENT IS IN THE NAME, because the two available ones
            # answer different questions and only one is evidence.
            # `gpu_peak_alloc_torch_mib` is `torch.cuda.max_memory_allocated`
            # -- TENSORS -- which reads exactly 0 on cpu, so NON-ZERO is the
            # whole test and no MiB floor is wanted: a short training on a
            # GPU can peak at 0.51 GiB, under any floor set from a 1M-step
            # figure, because peak scales with steps (the replay buffer caps
            # at `iters + 1` rows). `nvidia-smi` MEMORY would be the wrong
            # instrument entirely: it shows the ~600 MiB CUDA context even
            # when nothing trains, so non-zero there proves nothing.
            #
            # THE UTILISATION FIELDS ARE `_UtilSampler`'s THREE, and each
            # sample behind them is still a POINT SAMPLE of a ~1 s window
            # that is NOT evidence of idleness. A training can read 0% on all
            # eight 20 s samples, because a 6-130 ms update burst per ~900 ms
            # iteration is invisible at that cadence, and a single sample on
            # a cold JIT run can land inside the compile window. Hence several
            # samples and
            # three fields -- max, mean and the count they are over -- rather
            # than one number under a summary's name. A zero still means "no
            # evidence", never "idle", and nothing reads them to decide.
            #
            # Both are None off cuda rather than 0 -- "no GPU" and "a GPU
            # that did nothing" are different rows a cost report must not
            # average together.
            "gpu_peak_alloc_torch_mib": gpu_peak_mib,
            **util.row_fields(),
            "env_workers": int(env_workers), "buffer_rows": int(buffer_rows),
            "observation_dim": (int(phi.dim) if phi is not None else None),
            "policy_input_dim": int(n_obs),
            # THE EPISODE LENGTH THIS SEED ACTUALLY TRAINED UNDER, read off the
            # adapter and not from the config. `problem.horizon` truncates the
            # env's own (`envs.base.apply_horizon`), and a horizon is the one
            # environment fact a method may pin -- RDA's HumanoidBench column
            # reports 500 against HumanoidBench's shipped 1000 -- so a results
            # table that compared two runs without it would be comparing two
            # different tasks. `null` in the config and 1000 here is the
            # ordinary case; 500 here beside a 1000-step spec is the paper's
            # column, and 1000 here beside a config that asked for 500 would be
            # the override not having reached this process.
            "horizon_effective": int(getattr(env, "horizon", 0) or 0),
            # AND WHAT WAS ASKED FOR, beside it. `problem.horizon` is CLAMPED
            # at construction when the chosen env's episode is shorter (a
            # paper's horizon against another tier's env is a citation, the
            # same thing `train.env_steps: 10M` is on the tester tier), so the
            # pair is what makes the cap visible rather than inferable:
            # `requested` 500 with `effective` 25 is a tester run of RDA's
            # column, and `requested` None is "the environment's own" rather
            # than a number nobody chose. `_check_coherence` refuses instead of
            # clamping when the config chose its own env, so on a real tier the
            # two agree or the run never started.
            "horizon_requested": (int(cfg.get("problem.horizon"))
                                  if cfg.get("problem.horizon") is not None else None),
            "init": cfg.get("train.init", "from_scratch"),
            "warm_started_from": (init_ref or "") if warm_started else "",
            # `train.elite_constraint.kind` AS EXECUTED, `_sb3_run`'s two
            # fields: how many optimisers carry the L2 term, and WHICH
            # parameters it pulls toward -- the elite's stored policy ref, or
            # "" when the term is not running -- beside `warm_started_from`,
            # because on a resumed slice the two DIFFER and that difference
            # is the fix. The reference only under a non-`none` kind, so every
            # other seed row is unchanged. A run whose term never installed
            # and an ablation are the same curve; only these tell them apart.
            "elite_constraint_optimisers": int(n_constrained),
            **({"elite_constraint_reference": constraint_ref}
               if str(cfg.get("train.elite_constraint.kind", "none") or "none") != "none"
               else {}),
            # Share of collected transitions whose executed action was the
            # sampled one clipped to the adapter's bounds -- the rows on which
            # the reward's action argument and the buffer's differ.
            "action_clip_fraction": float(view.n_clipped / max(1, view.n_steps)),
            # WHICH VIEW STEPPED THIS, AND WHAT THE DEVICE BOUNDARY ACTUALLY
            # DID. Every field here is read off the view rather than inferred
            # from the config, because the config says what was asked for and
            # a results table needs what happened.
            #
            # `dlpack_zero_copy` is PER DIRECTION and never one boolean for
            # the pair: the observation and action conversions are
            # independent, each was measured by comparing a jax buffer
            # pointer against a torch `data_ptr` on a real step, and a
            # regression in either is invisible in a summary. `None` means
            # the pointers could not be read, which is not the same as False
            # and is recorded as its own value.
            #
            # `n_parallel_envs_ignored` is here because a warning is the
            # weakest instrument we have: a config key that is silently inert
            # is the "declared-but-unread key" class, and a row that carries
            # the fact makes the ignoring auditable after the run.
            **({"view": "jax",
                "physics": getattr(env, "physics", "mjx"),
                "num_envs": int(view.n),
                "n_parallel_envs_ignored": bool(int(env_workers or 0) > 1),
                # FIRST-CALL WALL OF EACH JITTED REGION: XLA compilation
                # PLUS that one call's execution, summed over the regions
                # (`_timed_first`). On this tier the compile is 120-215 s per
                # region and the execution is milliseconds, so it is the
                # compile to within one step -- but it is NOT a pure compile
                # clock and must not be compared against one from elsewhere.
                # Said here as well as on `_timed_first` because THIS is the
                # artifact a reader meets.
                "jit_compile_s": float(getattr(view, "jit_compile_s", 0.0)),
                # The tier's two possible per-step host syncs, stated
                # separately so the view's price is not mistaken for the
                # tier's.
                "reward_sync_s": float(getattr(view, "reward_sync_s", 0.0)),
                "n_clipped_reads": int(getattr(view, "n_clipped_reads", 0)),
                # ROWS THE ADAPTER REPORTED AS NONFINITE, terminated by the
                # view. Recorded rather than refused: a handful over millions
                # of steps is an env quirk, and zero-by-omission is the
                # reading this field exists to prevent.
                "nonfinite_rows": int(getattr(view, "nonfinite_rows", 0)),
                # BOTH FRAMEWORKS' PEAK ON ONE CARD, because
                # XLA_PYTHON_CLIENT_MEM_FRACTION splits it between them and
                # nobody should tune that split from a guess. 0.6 leaves torch
                # 40%; at 4,096 envs the fasttd3 buffer is min(buffer_size,
                # iters+1) rows -- a few hundred MB -- so 0.6 is generous, but
                # "generous" is an argument and these two numbers are
                # evidence. Read the recorded peaks before moving the
                # fraction.
                "device_peak_mib": dict(view.device_peaks()),
                # Compilation-cache hit or miss for THIS training, from jax's
                # own counters. A row that compiled from cold and one that
                # reused a neighbour's work are different runs at the same
                # wall clock, and only this field separates them.
                "jax_cache": dict(getattr(view, "jax_cache", {})),
                "dlpack_zero_copy": dict(getattr(view, "dlpack_zero_copy", {})),
                "xla_env": dict(getattr(view, "xla_env", {})),
                "jax_version": str(getattr(view, "jax_version", "")),
                "jax_torch_device": dict(getattr(view, "device_check", {}))}
               if isinstance(view, _JaxVecEnvView) else {}),
            "secondary_transitions": int(secondary_used),
            "secondary_ratio": float(sec_ratio if use_secondary else 0.0),
            # As executed: this backend never installs one (the module
            # docstring's last deviation), and `{}` is how a seed row says so
            # -- the same value an sb3 row carries when no anchor was hung.
            "anchor": {}, "anchor_released": False,
            "checkpoint_selection": select_rule,
            "checkpoint_selection_reason": _selection_reason(select_reason),
            "restored_checkpoint": (int(curve[ship_idx]["round"])
                                    if ship_idx is not None else None),
            "shipped_checkpoint": int(curve[ship_i]["round"]) if curve else None,
            "final_checkpoint_fitness": float(fits[-1]),
            "error": seed_error, "checkpoints": curve}
        if str(cfg.get("train.reward_scaling", "none") or "none") != "none":
            # LaRes Eq. 3 as executed (`_scaling_record`): the plan's affine
            # parameters and moments this seed's learner actually trained
            # under, or `applied: False` with the plan's reason. Only under a
            # non-`none` key, as on sb3.
            seed_metric["reward_scaling"] = _training._scaling_record(candidate, reward)
        if ceiling is not None:
            seed_metric["ceiling_decisions"] = ceiling_decisions
        if handoff is not None:
            # The slice hand-off AS EXECUTED, the same fields `_sb3_run`
            # records, so a LaRes row means the same thing on either backend.
            # `replay_prefill_supported` is TRUE here whenever this learner
            # configuration CAN honour the hand-off. It is False on
            # `num_steps > 1`, decided in the parent above so the
            # buffer is not widened for a pool that cannot be written; that
            # case is a refusal and says so rather than being left to be
            # inferred from `replay_prefill_pooled: 0`, which is also what a
            # slice with nothing to pool from looks like.
            own = min(int(getattr(handoff, "prefill_own", 0) or 0), int(prefilled))
            seed_metric.update({
                "slice_index": int(slice_index),
                "replay_prefill_supported": bool(prefill_supported),
                "replay_prefill_ref": (prefill_ref or "") if prefilled else "",
                "replay_prefill_own": int(own),
                "replay_prefill_pooled": int(prefilled - own),
                "replay_added": int(rows_added),
                "learning_starts_effective": int(learning_starts_effective),
            })
        view.close()
        return {"curve": curve, "comp_curve": comp_curve, "spent": spent,
                "spent_eval": spent_eval, "pruned_at": pruned_at,
                "timed_out": timed_out, "seed_error": seed_error,
                "wallclock_s": wall, "agent": agent, "rb": rb, "view": view,
                "replay_added": int(rows_added), "seed_metric": seed_metric}

    def fold_seed(out: Dict[str, Any]) -> None:
        """Parent-side bookkeeping for one seed, strictly in seed order."""
        nonlocal used, fatal_error
        used += out["spent"] + out["spent_eval"]
        if out["pruned_at"] is not None:
            result.pruned = True
        if out["timed_out"]:
            result.timed_out = True
        if out["seed_error"] and not fatal_error:
            fatal_error = out["seed_error"]
        per_seed_curves.append(out["curve"])
        per_seed_components.append(out.get("comp_curve") or [])
        result.seed_metrics.append(out["seed_metric"])
        ctx.budget.record_training(env_steps=out["spent"] + out["spent_eval"],
                                   gpu_seconds=out["wallclock_s"])

    n_runs = max(1, int(n_seeds))
    seed_workers, seed_reason = _seed_fork_workers(cfg, n_runs)
    _note_seed_schedule(ctx, candidate, seed_workers, seed_reason)
    final_blob = None
    final_slice = None
    # phi gating for `_sb3_run`'s reason: an exported slice must be in state
    # space or the store is poisoned for every reader.
    want_last_replay = bool(_wants_replay(cfg) and phi is None)
    replay_cap = int(cfg.get("train.secondary_buffer.size", 100000) or 100000)
    # The slice export (`SliceHandoff.export_ref`): the rows THIS call added,
    # raw, for the driver to pool. Same "last seed's learner" rule and the
    # same `phi is None` guard as the store write below, both for `_sb3_run`'s
    # reasons. Written through the SAME `_ROUND_REPLAY` key the sb3 path uses,
    # which is the point: `population.py` needs no backend branch.
    export_ref = getattr(handoff, "export_ref", None) if handoff is not None else None
    want_slice_export = bool(export_ref and phi is None)
    slice_added = 0      # rows the last seed added (its `replay_added`)
    slice_export = None  # and the export of exactly those rows

    try:
        if seed_workers <= 1:
            last_agent = None
            last_rb = None
            last_view = None
            for seed_i in range(n_runs):
                out = run_seed(seed_i)
                slice_added = int(out.get("replay_added", 0) or 0)
                last_agent = out.pop("agent")
                last_rb = out.pop("rb")
                last_view = out.pop("view")
                policies.append(_policy_fn(last_agent,
                                           last_view))
                fold_seed(out)
        else:
            def seed_child(i: int) -> Dict[str, Any]:
                out = run_seed(i)
                agent = out.pop("agent")
                rb = out.pop("rb")
                view = out.pop("view")
                out["blob"] = _fasttd3_blob(ns, arch, agent["actor"], agent["qnet"],
                                            agent["qnet_target"],
                                            agent["obs_normalizer"]
                                            if agent["obs_normalizer"] is not None
                                            else _NullState())
                if i == n_runs - 1 and want_last_replay:
                    out["replay"] = _fasttd3_export_replay(view, rb, replay_cap)
                # Exported in the CHILD, because `rb` lives on the device in
                # this process and does not cross the fork -- the same reason
                # `want_last_replay` is handled here rather than in the parent.
                if (i == n_runs - 1 and want_slice_export
                        and int(out.get("replay_added", 0) or 0) > 0):
                    out["slice_replay"] = _fasttd3_export_replay(
                        view, rb, int(out["replay_added"]))
                return out

            for seed_i, out in enumerate(
                    _fork_seeds(n_runs, seed_child, seed_workers, "fasttd3")):
                if "fatal" in out:
                    raise RuntimeError(f"fasttd3 seed worker {seed_i}/{n_runs}: "
                                       f"{out['fatal']}")
                blob = out.pop("blob", None)
                if "slice_replay" in out:
                    slice_export = out.pop("slice_replay")
                    slice_added = int(out.get("replay_added", 0) or 0)
                # Rebuild the trained policy from the exact parameters the child
                # saved -- `_sb3_run`'s rule, same fatal-not-mismeasured handling
                # when the blob does not load: rollouts on fresh weights would be
                # a corrupted measurement wearing a healthy one's clothes.
                agent2 = _build_agent(int(out["seed_metric"]["seed"]))
                # Action map only (`_policy_fn` reads `to_env_action`): this
                # view is never reset or stepped.
                view2 = _VecEnvView(env, 1, reward, _features, norm_mode,
                                    int(out["seed_metric"]["seed"]), n_obs)
                if blob is None or not _fasttd3_apply(
                        ns, arch, agent2["actor"], agent2["qnet"],
                        agent2["qnet_target"], agent2["obs_normalizer"], blob,
                        f"seed:{out['seed_metric']['seed']}"):
                    fatal_error = fatal_error or (
                        f"fasttd3 seed fork: seed {seed_i} sent no loadable policy "
                        "(serialisation failed in the child, or the blob did not "
                        "load into this architecture); its rollouts would run on "
                        "untrained weights, so the result is marked failed rather "
                        "than mis-measured")
                    blob = None
                policies.append(_policy_fn(agent2, view2))
                if seed_i == n_runs - 1:
                    final_blob = blob if blob is not None else None
                    final_slice = out.pop("replay", None) if blob is not None else None
                fold_seed(out)

        result.checkpoints = _mean_curve(per_seed_curves)
        _stamp_seed_rows(result.seed_metrics, seed_workers, seed_reason)
        _summarise_ceiling(candidate, result.seed_metrics)
        result.component_traces = _mean_components(per_seed_components)
        result.gt_reward_curve = [p["gt_return"] for p in result.checkpoints]

        # Single-threaded on BOTH schedules (not only the forked one, as
        # `_sb3_run` does): every seed above TRAINED at one thread, so the
        # rollouts must predict at one thread too or the sequential path's
        # trajectories would be the one part of the run whose numerics vary
        # with `--cpus-per-task`.
        with _single_thread_torch():
            roll_rng = np.random.default_rng(base_seed + 104729)
            for i in range(max(1, int(cfg.get("evaluate.rollouts_per_candidate", 3) or 1))):
                traj, st, _gt = _rollout(env, policies[i % len(policies)], roll_rng,
                                         reward,
                                         _error_router("exception_soft", io.StringIO()))
                result.trajectories.append(traj)
                if i < _n_replayed_rollouts(cfg):
                    _retain_replay_states(env, traj)
                used += st
                # RECORDED, not only accumulated into `env_steps_used`,
                # exactly as the TPE-store loop below does: otherwise
                # `budget.json` under-reports every run by
                # `evaluate.rollouts_per_candidate` x the horizon, which on a
                # short training can exceed the training itself.
                #
                # Charged per rollout rather than summed after the loop, so a
                # rollout that raises mid-loop still charges what it spent.
                ctx.budget.record_rollout_steps(st)
            # Dedicated TPE-store pool -- `_sb3_run`'s block verbatim, for
            # `_run_backend`'s reasoning: a separate rng so the eval rollouts
            # above stay byte-identical when collection is toggled, no state
            # retention (LRU pressure). Without it `update.memory.trajectory_store`
            # never grows on this backend, the TPE screen fails open forever and
            # `policy_trainings_skipped` stays 0 -- CARD silently becomes its own
            # ablation.
            n_store = _n_store_rollouts(cfg)
            if n_store > 0:
                store_rng = np.random.default_rng(base_seed + 224737)
                store_steps = 0
                for i in range(n_store):
                    traj, st, _gt = _rollout(env, policies[i % len(policies)],
                                             store_rng, reward,
                                             _error_router("exception_soft", io.StringIO()))
                    result.store_trajectories.append(traj)
                    used += st
                    store_steps += st
                # Collection steps must reach budget.json, not only
                # env_steps_used.
                ctx.budget.record_rollout_steps(store_steps)

        if seed_workers <= 1:
            if last_agent is not None:
                final_blob = _fasttd3_blob(
                    ns, arch, last_agent["actor"], last_agent["qnet"],
                    last_agent["qnet_target"],
                    last_agent["obs_normalizer"]
                    if last_agent["obs_normalizer"] is not None else _NullState())
            if want_last_replay and last_rb is not None and last_view is not None:
                final_slice = _fasttd3_export_replay(last_view, last_rb, replay_cap)
            if (want_slice_export and slice_added > 0
                    and last_rb is not None and last_view is not None):
                slice_export = _fasttd3_export_replay(last_view, last_rb, slice_added)
        if final_blob is not None:
            result.policy_ref = _store(_POLICY_STORE, f"policy:{candidate.cand_id}",
                                       final_blob, _POLICY_STORE_MAX_BYTES)
            _remember_code(candidate)
        if final_slice is not None:
            result.replay_ref = _store(_REPLAY_STORE, f"replay:{candidate.cand_id}",
                                       final_slice, _REPLAY_STORE_MAX_BYTES)
        if slice_export is not None and want_slice_export:
            # Plain assignment, no cap, into the SAME store and under the SAME
            # key the sb3 path writes: `_ROUND_REPLAY` is bounded by the driver,
            # which pops this key right after the wave. This line is what lets
            # `population.py` stay backend-free -- it reads one store, not two.
            from .training import _ROUND_REPLAY
            _ROUND_REPLAY[str(export_ref)] = slice_export
            for row in result.seed_metrics:
                row["replay_exported"] = int(slice_added)
    finally:
        env.set_dr(None)

    result.env_steps_used = int(used)
    result.wallclock_s = float(time.monotonic() - t0)
    if fatal_error:
        result.trained = False
        result.error = fatal_error
    return result


class _NullState:
    """Stands in for a disabled observation normalizer at serialisation time:
    `state_dict()` is empty, so the blob records "none" as an absence."""

    @staticmethod
    def state_dict() -> Dict[str, Any]:
        return {}
