"""`train.backend: simba_v2` -- SAC + SimbaV2 (Lee et al., 2025), ported line-faithfully.

The source of truth is the released implementation vendored at
`refs/code/SimbaV2` (dojeon-ai/SimbaV2, Apache-2.0); every design choice below
is pinned to a file:line in that tree, the way `configs/methods/*.yaml` pin paper
sections and `bird/components/fasttd3.py` pins `refs/code/FastTD3`.

WHY THIS BACKEND EXISTS. RDA (Lee et al., RLJ 2026) reports its HumanoidBench
column as "SAC" + "SimbaV2" (`refs/tex/rda/appendix.tex:1403` Table 1, rows
:1416-1426). Neither of the other backends is that: `train.backend: sb3` is SAC
with SB3's two-layer `MlpPolicy` and records `train.architecture` without
honouring it (`training.py`), and `train.backend: fasttd3` has SimbaV2-shaped
layers with a **TD3** update.

THE OFFICIAL RELEASE IS ALREADY SAC, AND IT DERIVES TWO OF RDA'S TABLE-1 ROWS.
`configs/agent/simbaV2.yaml:2` is headed "SAC with Hyper-Simba architecture",
so RDA's learner is one repository rather than an assembly of two papers. Two
of Table 1's numbers then fall out of that repo's own defaults rather than
being independent choices -- which is both the evidence that RDA ran this
release and a check on every pin in `rda_humanoidbench.yaml`:

    configs/online_rl.yaml:19-20
        eff_episode_len = max_episode_steps / action_repeat
        gamma = max(min((eff/5 - 1) / (eff/5), 0.995), 0.95)
    configs/online_rl.yaml:27, :28
        num_interaction_steps      = num_env_steps / (num_train_envs * action_repeat)
        updates_per_interaction_step = action_repeat

    at max_episode_steps 500, action_repeat 2 (HB's default,
    configs/env/hb_locomotion.yaml:13-15), num_train_envs 16, 10M env steps:

        eff = 250       -> gamma   = (50 - 1) / 50 = 0.98      <- Table 1 :1426
        updates         = 10e6 / (16 * 2) * 2 = 625,000        <- Table 1 :1423

`(max_episode_steps 500, action_repeat 2)` is the UNIQUE pair consistent with
both rows: `action_repeat 1` at 500 gives gamma 0.99, and `max_episode_steps
250` gives gamma 0.98 but contradicts Table 1's own 500. So Table 1's
"Max episode steps 500", "Training envs 16", "Env steps 10M", "Update steps
625K" and "Discount gamma 0.98" are ONE decision with four consequences, and
this backend expresses all of them as keys it reads.

What the algorithm is (`scale_rl/agents/simbaV2/`):
  * a HYPERSPHERICAL network (`simbaV2_layer.py`): every linear map is a
    bias-free `HyperDense` whose kernel is L2-normalised, inputs are embedded
    by appending a constant `c_shift` and normalising, each residual block
    interpolates toward its MLP output through a learned per-feature `alpha`
    and re-normalises (`HyperLERPBlock`), and a learnable `Scaler` restores
    the scale the normalisation removed;
  * the WEIGHT PROJECTION the paper is named for: after EVERY actor and critic
    gradient step -- and once at initialisation -- every `hyper_dense` kernel is
    projected back onto the unit sphere (`simbaV2_update.py:37 l2normalize_network`
    `l2normalize_network`, called at `:91`, `:239` and
    `simbaV2_agent.py:193-195`);
  * a SAC actor: a state-dependent tanh-Gaussian (`HyperNormalTanhPolicy`,
    `simbaV2_layer.py:120 HyperNormalTanhPolicy`) with `log_std` squashed into
    [`log_std_min`, `log_std_max`] = [-10, 2] and a learned entropy
    temperature (`SimbaV2Temperature`, `simbaV2_network.py:168 SimbaV2Temperature`) trained
    against `temp_target_entropy_coef * action_dim`;
  * a DISTRIBUTIONAL critic over `critic_num_bins` bins on
    [-normalized_g_max, +normalized_g_max] of the NORMALISED reward, trained by
    cross-entropy against the entropy-regularised categorical target
    `reward + gamma^n * (bin_values - temp * log pi(a'|s')) * (1 - terminated)`
    (`simbaV2_update.py:96 categorical_td_loss` `categorical_td_loss`);
  * clipped double Q as a PER-SAMPLE distribution pick -- the target log-probs
    of whichever head has the lower expected value (`simbaV2_update.py:165-168`),
    enabled on episodic envs (`configs/agent/simbaV2.yaml:31`,
    `critic_use_cdq: ${env.episodic}`, and HumanoidBench is episodic);
  * `optax.adam` on all three networks with a LINEAR learning-rate schedule
    from `learning_rate_init` to `learning_rate_end` over the whole update
    budget (`simbaV2_agent.py:114`), no weight decay and no gradient
    clipping;
  * soft target updates every gradient step at `target_tau` 0.005, no policy
    delay and no target-policy smoothing (`simbaV2_update.py:244 update_target_network`);
  * an online observation normaliser and a running-discounted-return reward
    scaler (`agents/wrappers/normalization.py`), both updated at COLLECTION
    only;
  * `action_repeat` env steps per agent decision, and
    `updates_per_interaction_step` gradient steps per decision
    (`run_online.py:131-138`).

Deviations from the official release, each deliberate and named. The first two
are the port itself; the rest are numbers, and every one of them is a key this
module READS so a config can restore the official value:

  * FRAMEWORK. Upstream is JAX/Flax/optax with `tensorflow_probability`
    (`deps/requirements.txt`); this is PyTorch, for `fasttd3.py`'s reasons and
    one more: `refs/code/` is re-fetched at HEAD by `scripts/fetch_refs.sh`, so
    importing it would mean the code that ran is not in this repo's history.
    `optax.adam` -> `torch.optim.Adam`, `optax.linear_schedule` -> a `LambdaLR`
    stepped once per gradient step (optax indexes its schedule by optimiser
    step, so the two agree step for step), and
    `tfd.TransformedDistribution(MultivariateNormalDiag, tfb.Tanh)` ->
    an explicit tanh-Gaussian whose `log_prob` uses the stable
    `2 * (log 2 - u - softplus(-2u))` form of the tanh log-determinant.
    `nn.initializers.orthogonal(scale=1.0, column_axis=0)` ->
    `nn.init.orthogonal_`; both are immediately followed by the unit-sphere
    projection above, which is what actually sets the kernel's scale.
  * THE ENV BOUNDARY is BIRD's, exactly as on `fasttd3`: upstream vectorises
    HumanoidBench with gymnasium vector envs (`scale_rl/envs/humanoid_bench.py`),
    BIRD adapters expose a functional `step(state, action)`, so this backend
    reuses `fasttd3._VecEnvView` -- `num_envs` episode slots on ONE adapter
    with each slot's episode state put back before each of its steps, the
    reward this backend trains on replacing the env reward at the boundary, and
    the reward called on the ENV-SPACE action the dynamics executed. Read
    `fasttd3.py`'s own deviation list for the whole contract; it is shared code
    and not a second implementation.

    WHICH reward that is is a config choice, `train.reward_source`: the
    candidate's compiled reward by default, or under `reward_source: reference`
    the ENVIRONMENT's own shipped reward (`EnvAdapter.reference_reward`), the
    oracle arm every searched curve is read against. Every backend takes that
    reward from one dispatch, `training._training_reward`, so the channel is
    the same here as on the others. Worded to match `fasttd3.py`'s bullet
    deliberately: two descriptions of one shared seam that drift apart are how
    a reader ends up trusting the wrong one.
  * THE REPLAY BUFFER is `fasttd3`'s `SimpleReplayBuffer` (a per-env circular
    buffer sampled `batch_size // num_envs` rows per env) rather than
    upstream's flat `scale_rl/buffers/numpy_buffer.py`. Uniform sampling and
    n-step both agree; what differs is that a batch is STRATIFIED over the 16
    envs instead of drawn uniformly from the pool. `buffer_size` and
    `batch_size` here are TOTALS, as upstream's `max_length` and
    `sample_batch_size` are, and `buffer_rows` / `batch_per_env` on the seed
    row record the per-env figures actually allocated and drawn.
  * `min_length` (upstream 5,000 transitions before the first update,
    `configs/buffer/numpy_uniform.yaml:7`) is `learning_starts` here, in
    TRANSITIONS, converted to interaction steps by `ceil(x / num_envs)`.
    Upstream also feeds the normalisers with RANDOM actions until the buffer
    can sample (`run_online.py:106-107`); this port does the same.
  * `amp` DEFAULTS OFF (`false`), where `fasttd3`'s defaults on: upstream is
    float32 throughout and has no mixed-precision path at all, so bf16
    autocast would be a numerical departure with no source. It stays a key
    because an operator may want it and the seed row records what ran.
  * `torch.compile` is applied to the update closure only, and only on cuda,
    as on `fasttd3`. Upstream `jax.jit`s the whole update; the closure is the
    same region.
  * NOT PORTED: the offline/BC arm (`actor_bc_alpha`, `simbaV2_bc.yaml`,
    `run_offline.py`), `load_param_key`/`load_only_param` checkpoint surgery
    (BIRD's `train.init` + `_POLICY_STORE` own that), Simba v1
    (`agents/simba/`), the `num_qs != 2` critic, hydra/wandb/video (BIRD's
    artifact machinery owns those) and `scale_rl/envs/` (BIRD's adapters).
  * `train.anchor.*`, `train.reference_policy` and
    `train.interaction_cfg.shared_buffer` are sb3-backend features; a config
    that enables them here trains without them, LOUDLY, exactly as on
    `fasttd3`.

Departures from `fasttd3`'s SimbaV2 layers, listed because the obvious cheaper
route was to add a SAC update to those and it would NOT have been SimbaV2.
`refs/code/FastTD3/fast_td3/fast_td3_simbav2.py` is FastTD3's port of these
layers with FastTD3's own choices baked in, and it differs from the official
source in six ways, each closed here:

    | thing               | official SimbaV2                  | FastTD3's port      |
    |---------------------|-----------------------------------|---------------------|
    | policy head         | tanh-GAUSSIAN, state-dep log_std  | deterministic tanh  |
    | `Scaler` init       | param = `scale`, forward `init`   | param = `init*scale`|
    | weight projection   | after every gradient step         | ABSENT              |
    | critic support      | +-`normalized_g_max` (5) on the    | +-250 on RAW reward |
    |                     | NORMALISED reward                 |                     |
    | target update       | tau 0.005, every step, no delay   | tau 0.1, delay 2,   |
    |                     |                                   | policy_noise 0.001  |
    | optimiser           | Adam, linear 1e-4 -> 5e-5          | AdamW wd 0.1,       |
    |                     |                                   | cosine from 3e-4    |
    | hidden dims         | actor 128/1, critic 512/2         | actor 512, crit 1024|
    | reward scaler g_max | 5.0                               | 10.0 (default)      |

The `Scaler` row is the subtle one: official initialises the parameter to
`scale` and multiplies by `forward_scaler = init / scale`, so the layer scales
by `init` at initialisation; FastTD3 initialises it to `init * scale`, so the
layer scales by `init**2`. At `scaler_init = scaler_scale = sqrt(2/hidden)`
that is `sqrt(2/h)` against `2/h`.

torch is imported lazily inside the backend call, never at module import:
`registry.load_all()` imports this module for every config in the repo,
including `--dry-run` and `--validate-all` on machines with no torch.
"""

from __future__ import annotations

import contextlib
import io
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..envs.jax_base import BatchedEnvAdapter
from ..registry import register
from ..types import Candidate, TrainResult
from . import training as _training
from . import fasttd3 as _fasttd3
from .fasttd3 import (
    _JaxVecEnvView,
    _UtilSampler,
    _VecEnvView,
    _action_affine,
    _data_root_source,
    _param_device,
)
from .training import (
    MAX_CHECKPOINTS,
    MIN_CHECKPOINTS,
    _POLICY_STORE,
    _PolicyBlob,
    _ceiling_guard,
    _ceiling_wanted,
    _demo_ceiling,
    _demo_fraction_point,
    _dr_params,
    _error_router,
    _install_observation,
    _pruner_for,
    _running_best,
    _seed_base,
    _seed_for,
    _selection_for,
    _selection_reason,
    _single_thread_torch,
    _training_init,
    log,
)

#: The blob format tag. Its own value, never `fasttd3/v1`: the two backends
#: write torch containers of the same SHAPE (`training._blob_kind` cannot tell
#: them apart, and until it read this marker a simba_v2 checkpoint would have
#: been handed to `fasttd3_policy_from_blob`), and a rebuild under the wrong
#: learner is a misroute rather than a refusal -- the failure
#: `tests/test_policy_from_ref_knows_fasttd3.py` was written for.
_BLOB_FORMAT = "simba_v2/v1"

#: Upstream defaults, verbatim from `refs/code/SimbaV2`. Three files, because
#: upstream splits the agent from the loop from the buffer, and BIRD's
#: `train.hyperparameters` is one dict:
#:
#:   configs/agent/simbaV2.yaml      the learner
#:   configs/online_rl.yaml          the loop (updates per decision, the LR horizon)
#:   configs/buffer/numpy_uniform.yaml   the replay buffer
#:   configs/env/hb_locomotion.yaml  HumanoidBench's action_repeat
#:
#: Keys upstream owns that BIRD owns elsewhere are ABSENT, not defaulted:
#: `num_env_steps` is `train.env_steps` (in SIMULATOR steps, upstream's own
#: accounting -- `run_online.py:160` logs `interaction_step * action_repeat *
#: num_train_envs`), `num_train_envs` is `train.n_parallel_envs`, `seed` is
#: `_seed_base`, `env_name` is `problem.env_id`, `gamma` is a
#: `train.hyperparameters` key the METHOD pins (RDA: 0.98), and the eval /
#: record / checkpoint intervals are the checkpoint schedule every backend
#: shares.
#:
#: `critic_min_v`/`critic_max_v` and the four derived `scaler`/`alpha`
#: constants are NOT here: upstream computes them from other values
#: (`simbaV2.yaml:25-28`, `:36-41`) and so does `_derived`, once, so the two cannot
#: disagree.
_SIMBAV2_DEFAULTS: Dict[str, Any] = {
    # -- the loop (configs/online_rl.yaml, configs/env/hb_locomotion.yaml) ---
    "action_repeat": 1,                # hb_locomotion.yaml:13 is 2; the default
                                       # is the most degenerate sensible value,
                                       # 1, so a config that wants the paper's
                                       # decision rate states it.
                                       # rda_humanoidbench pins 2.
    "updates_per_interaction_step": None,  # online_rl.yaml:28 -- `${action_repeat}`.
                                       # None resolves to `action_repeat`, which is
                                       # upstream's expression rather than a copy
                                       # of its value; a number overrides it.
    "learning_rate_decay_rate": 1.0,   # online_rl.yaml / simbaV2.yaml:20 -- the
                                       # share of the update budget the linear LR
                                       # schedule spans (1.0 = all of it)
    # -- the learner (configs/agent/simbaV2.yaml) ---------------------------
    "normalize_observation": True,     # simbaV2.yaml:8
    "normalize_reward": True,          # simbaV2.yaml:9
    "normalized_g_max": 5.0,           # simbaV2.yaml:10 -- BOTH the reward
                                       # scaler's g_max AND the critic's support
                                       # (+-this), which is why it is one key
    "learning_rate_init": 1e-4,        # simbaV2.yaml:17
    "learning_rate_end": 5e-5,         # simbaV2.yaml:18
    "actor_num_blocks": 1,             # simbaV2.yaml:22
    "actor_hidden_dim": 128,           # simbaV2.yaml:23
    "actor_c_shift": 3.0,              # simbaV2.yaml:24
    "critic_use_cdq": True,            # simbaV2.yaml:31 is `${env.episodic}`;
                                       # HumanoidBench is episodic
                                       # (env/hb_locomotion.yaml:8) and so is every
                                       # BIRD adapter with a horizon
    "critic_num_blocks": 2,            # simbaV2.yaml:32
    "critic_hidden_dim": 512,          # simbaV2.yaml:33
    "critic_c_shift": 3.0,             # simbaV2.yaml:34
    "critic_num_bins": 101,            # simbaV2.yaml:35
    "expansion": 4,                    # simbaV2_layer.py:96 HyperLERPBlock default
    "log_std_min": -10.0,              # simbaV2_layer.py:121
    "log_std_max": 2.0,                # simbaV2_layer.py:122
    "target_tau": 0.005,               # simbaV2.yaml:43
    "temp_initial_value": 0.01,        # simbaV2.yaml:45
    "temp_target_entropy": None,       # simbaV2.yaml:46 -- null; the agent fills it
                                       # from the coefficient (simbaV2_agent.py:309)
    "temp_target_entropy_coef": -0.5,  # simbaV2.yaml:47
    "gamma": 0.99,                     # NOT upstream's: online_rl.yaml:21 DERIVES
                                       # gamma from the episode length, and BIRD has
                                       # no episode-length key here (the horizon is
                                       # an env fact). So the method config pins it
                                       # -- rda_humanoidbench pins the paper's 0.98
                                       # -- and this is the inert fallback.
    "n_step": 1,                       # online_rl.yaml:21
    # -- the buffer (configs/buffer/numpy_uniform.yaml) ---------------------
    "buffer_size": 1_000_000,          # numpy_uniform.yaml:6 `max_length`, TOTAL
                                       # transitions across every env
    "learning_starts": 5_000,          # numpy_uniform.yaml:7 `min_length`, TOTAL
                                       # transitions before the first update
    "batch_size": 256,                 # numpy_uniform.yaml:8 `sample_batch_size`,
                                       # TOTAL rows per gradient step
    # -- BIRD's own, not upstream's ----------------------------------------
    "device": "cpu",                   # `_FASTTD3_DEFAULTS`' key and its reasons:
                                       # cpu runs everywhere and a sweep should not
                                       # serialise behind the free GPUs. "auto" =
                                       # cuda if available; an explicit `cuda` with
                                       # no cuda is REFUSED, never downgraded
    "env_workers": 1,                  # forked slot-worker processes behind
                                       # `_VecEnvView` -- a SCHEDULE, bit-identical
                                       # to 1. `train.n_parallel_envs` is the
                                       # paper's ENV COUNT on this backend (see
                                       # `simba_v2_backend`), so the schedule needs
                                       # a key of its own and this is it
    "amp": False,                      # upstream is float32 throughout; see the
                                       # module docstring
    "amp_dtype": "bf16",
    "compile": False,                  # NOT `None` ("cuda") as on fasttd3, and the
                                       # difference is a statement about what has been
                                       # tested rather than a preference. `update`
                                       # mutates parameters in place inside a loop over
                                       # named submodules (`project_`) and steps three
                                       # optimisers through a GradScaler; upstream jits
                                       # the equivalent region, but inductor over THIS
                                       # shape is not validated. A default of "compile
                                       # on cuda" would make a GPU run of this backend
                                       # also a test of that, and a graph break or an
                                       # InductorError an hour into a long job is the
                                       # avoidable version of that experiment. Enable
                                       # it deliberately and compare before/after.
    "compile_mode": "reduce-overhead",
    "torch_deterministic": True,
    "use_grad_norm_clipping": False,   # upstream clips nothing
    "max_grad_norm": 0.0,
}

#: Keys that belong to ANOTHER backend's expression of something this one
#: expresses differently -- named so the warning says which, rather than
#: "unknown".
#:
#: WHY THIS LIST EXISTS AT ALL. `train.backend` is a PROFILE key, so one
#: method config's `train.hyperparameters` is read by whichever learner the
#: profile picked, and `configs/methods/rda_humanoidbench.yaml` therefore carries RDA's
#: 625K update budget TWICE: once as `train_freq: 16, gradient_steps: 1` (the
#: only way `train.backend: sb3` can express it, measured there) and once as
#: `action_repeat` + `updates_per_interaction_step` (this backend's, which is
#: upstream SimbaV2's own). Each profile honours one pair and the other is
#: inert -- unavoidable, and the thing to get right is that the inert one SAYS
#: it is inert and says what replaced it. A bare "not a parameter of the
#: simba_v2 learner" would read as a typo.
_FOREIGN: Dict[str, str] = {
    "train_freq": "the sb3 backend's update budget; simba_v2 expresses the same "
                  "ratio as action_repeat x updates_per_interaction_step "
                  "(upstream's own: configs/online_rl.yaml:27, :28)",
    "gradient_steps": "the sb3 backend's update budget; see train_freq -- "
                      "updates_per_interaction_step is this backend's key",
    "num_envs": "the fasttd3 backend's fleet width; on simba_v2 the fleet is "
                "train.n_parallel_envs, which is the paper's `num_train_envs`",
    "buffer_size_per_env": "fasttd3 sizes its buffer per env; buffer_size here "
                           "is upstream's `max_length`, a TOTAL",
}

#: Names a config may not supply because the harness owns them; same rule and
#: message shape as `_fasttd3_hyper`'s.
_RESERVED = ("seed", "env_name", "agent", "agent_type", "num_env_steps",
             "num_train_envs", "num_interaction_steps", "max_episode_steps",
             "episodic", "cuda", "load_path", "save_path", "use_wandb",
             "num_eval_envs", "num_eval_episodes", "actor_bc_alpha",
             "load_only_param", "load_param_key", "load_observation_normalizer",
             "load_reward_normalizer")

#: The one architecture this backend implements. `train.architecture` is
#: free-form and the published configs pin it as a citation (`rda.yaml`:
#: `simba`; `rda_humanoidbench.yaml`: `simba_v2`); anything else warns and runs
#: this, with the EXECUTED value on the seed row -- `_resolve_arch`'s shape.
_ARCHITECTURES = ("simba_v2",)


def _simba_hyper(cfg: Any) -> Dict[str, Any]:
    """`train.hyperparameters` over `_SIMBAV2_DEFAULTS`, upstream's names.

    Same two rules as `_fasttd3_hyper` and `_learner_kwargs`: a known name
    genuinely changes the run, an unknown one is WARNED about rather than
    dropped in silence -- a declared key nothing reads is a fabricated pin,
    and a swept name that reaches no learner makes
    `train.hyperparameter_search` a max over nothing. Reserved names are set
    by the harness and warned about too.
    """
    out = dict(_SIMBAV2_DEFAULTS)
    hyper = cfg.get("train.hyperparameters") or {}
    unknown: List[str] = []
    for key, value in hyper.items():
        if key in _RESERVED:
            log.warning("train.hyperparameters: %r is set by the harness on "
                        "train.backend=simba_v2 (the env by problem.env_id, the "
                        "step budget by train.env_steps, the env count by "
                        "train.n_parallel_envs, the seed by "
                        "train.seeds_per_candidate) and was ignored", key)
        elif key in out:
            out[key] = value
        elif key in _FOREIGN:
            log.warning("train.hyperparameters: %r is %s -- it does not affect "
                        "this run, and that is the two-backends-one-config "
                        "shape rather than a typo (see `_FOREIGN`)",
                        key, _FOREIGN[key])
        else:
            unknown.append(key)
    if unknown:
        log.warning("train.hyperparameters: %s is not a parameter of the simba_v2 "
                    "learner and will not affect this run (it accepts %s). A "
                    "declared key nothing reads is a fabricated pin.",
                    sorted(unknown), sorted(_SIMBAV2_DEFAULTS))
    return out


def _derived(hyper: Dict[str, Any], n_act: int) -> Dict[str, Any]:
    """The constants upstream computes from other constants, computed ONCE.

    `configs/agent/simbaV2.yaml:25-28, :36-41` writes each of these as an
    OmegaConf `${eval:...}` over `actor_hidden_dim` / `critic_hidden_dim` /
    `*_num_blocks` / `normalized_g_max`, and `simbaV2_agent.py:309` fills
    `temp_target_entropy` from `temp_target_entropy_coef * action_dim`. Copying
    the resolved numbers into `_SIMBAV2_DEFAULTS` would let a config change a
    hidden dimension and silently keep the old scaler constants, which is the
    two-copies-of-one-fact shape; so they are derived here and nowhere else.
    """
    a_hid = int(hyper["actor_hidden_dim"])
    c_hid = int(hyper["critic_hidden_dim"])
    a_blocks = int(hyper["actor_num_blocks"])
    c_blocks = int(hyper["critic_num_blocks"])
    g_max = float(hyper["normalized_g_max"])
    target_entropy = hyper["temp_target_entropy"]
    if target_entropy is None:
        target_entropy = float(hyper["temp_target_entropy_coef"]) * int(n_act)
    return {
        # simbaV2.yaml:25-26, :36-37 -- sqrt(2 / hidden_dim) for both
        "actor_scaler_init": math.sqrt(2.0 / a_hid),
        "actor_scaler_scale": math.sqrt(2.0 / a_hid),
        "critic_scaler_init": math.sqrt(2.0 / c_hid),
        "critic_scaler_scale": math.sqrt(2.0 / c_hid),
        # simbaV2.yaml:27-28, :40-41
        "actor_alpha_init": 1.0 / (a_blocks + 1),
        "actor_alpha_scale": 1.0 / math.sqrt(a_hid),
        "critic_alpha_init": 1.0 / (c_blocks + 1),
        "critic_alpha_scale": 1.0 / math.sqrt(c_hid),
        # simbaV2.yaml:38-39 -- the support IS the reward scaler's g_max, on
        # the NORMALISED reward. This is the row FastTD3's port does not have
        # (+-250 on raw reward there), and it is why `normalize_reward: false`
        # with these bounds would put every real return outside the support.
        "critic_min_v": -g_max,
        "critic_max_v": +g_max,
        "temp_target_entropy": float(target_entropy),
    }


# ==========================================================================
# The networks -- defined lazily, cached
# ==========================================================================
#
# Same idiom as `fasttd3._torch_classes`: these subclass `torch.nn.Module`, so
# defining them needs torch imported, and this module is imported by
# `registry.load_all()` on machines that have none.
#
# The normalisers and the replay buffer are NOT redefined here -- they are
# `fasttd3`'s, which are themselves ports of these same upstream constructions
# (`fast_td3_utils.py`'s `RewardNormalizer` is SimbaV2's `RewardNormalizer`
# with a different default g_max, and `EmpiricalNormalization` is its
# `ObservationNormalizer`). One copy, imported, so the two backends cannot
# disagree about what "normalised" means.

_TORCH_NS: Optional[Dict[str, Any]] = None


def _torch_classes() -> Dict[str, Any]:
    global _TORCH_NS
    if _TORCH_NS is not None:
        return _TORCH_NS

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def l2normalize(tensor: Any, axis: int = -1, eps: float = 1e-8) -> Any:
        """`simbaV2_update.py:14 l2normalize`. `maximum(norm, eps)`, NOT `norm + eps`:
        FastTD3's port uses the second (`fast_td3_simbav2.py:8-11`), which
        shrinks every vector slightly and is a different function for a
        near-zero one."""
        norm = torch.linalg.norm(tensor, ord=2, dim=axis, keepdim=True)
        return tensor / torch.clamp(norm, min=eps)

    class Scaler(nn.Module):
        """`simbaV2_layer.py:14 Scaler`.

        THE PARAMETER IS INITIALISED TO `scale`, NOT TO `init * scale`, and the
        difference is the departure named in the module docstring. Upstream:
        `self.param("scaler", nn.initializers.constant(1.0 * self.scale), dim)`
        with `forward_scaler = init / scale`, so the layer multiplies by
        `scale * init / scale == init`. FastTD3's port stores `init * scale`
        and so multiplies by `init ** 2`; at
        `scaler_init = scaler_scale = sqrt(2/h)` that is `sqrt(2/h)` against
        `2/h`.
        """

        def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0,
                     device: Any = None) -> None:
            super().__init__()
            self.scaler = nn.Parameter(torch.full((dim,), float(scale), device=device))
            self.forward_scaler = init / scale

        def forward(self, x: Any) -> Any:
            return self.scaler.to(x.dtype) * self.forward_scaler * x

    class HyperDense(nn.Module):
        """`simbaV2_layer.py:31 HyperDense`: no bias, orthogonal init.

        MARKED, not just shaped: `l2normalize_network` finds these by the
        upstream layer NAME (`regex="hyper_dense"`,
        `simbaV2_update.py:37 l2normalize_network`), so the port needs its own marker for the
        same projection to reach the same set of kernels. `_HYPER = True` is
        it, and `_project_` below is the only reader.
        """

        _HYPER = True

        def __init__(self, in_dim: int, hidden_dim: int, device: Any = None) -> None:
            super().__init__()
            self.w = nn.Linear(in_dim, hidden_dim, bias=False, device=device)
            nn.init.orthogonal_(self.w.weight, gain=1.0)

        def forward(self, x: Any) -> Any:
            return self.w(x)

    @torch.no_grad()
    def project_(module: Any, eps: float = 1e-8) -> int:
        """THE HYPERSPHERICAL PROJECTION -- `simbaV2_update.py:24-46 l2normalize_layer/_network`.

        Every `hyper_dense` kernel back onto the unit sphere, after every
        actor and critic gradient step (`:91`, `:239`) and once at
        initialisation (`simbaV2_agent.py:193-195`). This is the mechanism
        SimbaV2 is named for and it is ABSENT from FastTD3's port, which is
        the single strongest reason this module exists rather than a SAC
        update bolted onto `fast_td3_simbav2.py`.

        THE AXIS. Upstream normalises a flax `Dense` kernel of shape
        `(in, out)` along `axis=0` -- one unit vector per OUTPUT unit, over the
        inputs. `torch.nn.Linear.weight` is `(out, in)`, so the same set of
        vectors is `dim=1`. Getting this wrong is silent: both axes produce
        unit-norm weights and only one of them is upstream's.

        Returns how many kernels it touched, so a caller can assert the
        projection reached the network rather than trust that it did.
        """
        n = 0
        for mod in module.modules():
            if getattr(mod, "_HYPER", False):
                w = mod.w.weight
                w.div_(torch.clamp(
                    torch.linalg.norm(w, ord=2, dim=1, keepdim=True), min=eps))
                n += 1
        return n

    class HyperMLP(nn.Module):
        """`simbaV2_layer.py:46 HyperMLP`."""

        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                     scaler_init: float, scaler_scale: float, eps: float = 1e-8,
                     device: Any = None) -> None:
            super().__init__()
            self.w1 = HyperDense(in_dim, hidden_dim, device=device)
            self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
            self.w2 = HyperDense(hidden_dim, out_dim, device=device)
            self.eps = eps

        def forward(self, x: Any) -> Any:
            # `+ eps` after the relu is upstream's own comment: "required to
            # prevent zero vector", because the l2normalize below would
            # otherwise divide a dead unit's output by eps.
            x = self.w2(F.relu(self.scaler(self.w1(x))) + self.eps)
            return l2normalize(x, axis=-1)

    class HyperEmbedder(nn.Module):
        """`simbaV2_layer.py:68 HyperEmbedder`: append the constant `c_shift`, normalise,
        embed, scale, normalise. The appended constant is what keeps the
        normalised input from losing its magnitude entirely."""

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
        """`simbaV2_layer.py:89 HyperLERPBlock`: residual by learned per-feature
        interpolation, then re-normalise onto the sphere."""

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

    class HyperNormalTanhPolicy(nn.Module):
        """`simbaV2_layer.py:120 HyperNormalTanhPolicy` -- the SAC head, and the one FastTD3's
        port does not have (its `HyperTanhPolicy` emits a mean and nothing
        else).

        Two `HyperDense` towers, one for the mean and one for `log_std`, each
        with its own bias; `log_std` is squashed into
        [`log_std_min`, `log_std_max`] through
        `min + (max - min) * 0.5 * (1 + tanh(log_std))` -- upstream's own
        comment calls it "normalize log-stds for stability" -- and the
        distribution is `tanh(N(mean, exp(log_std) * temperature))`.

        `temperature` here is the SAMPLING temperature of
        `simbaV2_agent.py:329 sample_actions` (1.0 while training, **0.0** for
        evaluation), NOT the entropy temperature. At 0.0 the scale is zero and
        the distribution collapses to `tanh(mean)`, which is what makes the
        greedy policy the same code path as the exploring one.
        """

        def __init__(self, hidden_dim: int, action_dim: int, scaler_init: float,
                     scaler_scale: float, log_std_min: float = -10.0,
                     log_std_max: float = 2.0, device: Any = None) -> None:
            super().__init__()
            self.mean_w1 = HyperDense(hidden_dim, hidden_dim, device=device)
            self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale,
                                      device=device)
            self.mean_w2 = HyperDense(hidden_dim, action_dim, device=device)
            self.mean_bias = nn.Parameter(torch.zeros(action_dim, device=device))
            self.std_w1 = HyperDense(hidden_dim, hidden_dim, device=device)
            self.std_scaler = Scaler(hidden_dim, scaler_init, scaler_scale,
                                     device=device)
            self.std_w2 = HyperDense(hidden_dim, action_dim, device=device)
            self.std_bias = nn.Parameter(torch.zeros(action_dim, device=device))
            self.log_std_min = float(log_std_min)
            self.log_std_max = float(log_std_max)

        def forward(self, x: Any) -> Tuple[Any, Any]:
            mean = self.mean_w2(self.mean_scaler(self.mean_w1(x)))
            mean = mean + self.mean_bias.to(mean.dtype)
            log_std = self.std_w2(self.std_scaler(self.std_w1(x)))
            log_std = log_std + self.std_bias.to(log_std.dtype)
            log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (
                1.0 + torch.tanh(log_std))
            return mean, log_std

    class HyperCategoricalValue(nn.Module):
        """`simbaV2_layer.py:174 HyperCategoricalValue`: bin logits -> `log_softmax` -> the
        expected value against the bin support. Returns BOTH, because the
        update needs the log-probabilities (cross-entropy) and the actor needs
        the value."""

        def __init__(self, hidden_dim: int, num_bins: int, min_v: float, max_v: float,
                     scaler_init: float, scaler_scale: float,
                     device: Any = None) -> None:
            super().__init__()
            self.w1 = HyperDense(hidden_dim, hidden_dim, device=device)
            self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale, device=device)
            self.w2 = HyperDense(hidden_dim, num_bins, device=device)
            self.bias = nn.Parameter(torch.zeros(num_bins, device=device))
            self.register_buffer(
                "bin_values",
                torch.linspace(min_v, max_v, num_bins, device=device).view(1, -1))

        def forward(self, x: Any) -> Tuple[Any, Any]:
            logits = self.w2(self.scaler(self.w1(x)))
            logits = logits + self.bias.to(logits.dtype)
            log_prob = F.log_softmax(logits, dim=1)
            value = torch.sum(torch.exp(log_prob) * self.bin_values.to(log_prob.dtype),
                              dim=1)
            return value, log_prob

    class SimbaV2Actor(nn.Module):
        """`simbaV2_network.py:16 SimbaV2Actor` -- embedder, `num_blocks` LERP blocks,
        the tanh-Gaussian head.

        `sample` is `simbaV2_agent.py:329 sample_actions` plus `update_actor`'s own
        `dist.sample(); dist.log_prob(actions)` pair
        (`simbaV2_update.py:65-66`), in ONE method so the action a log-prob
        belongs to is the action that was returned -- two calls to a
        distribution object would draw twice.

        The log-determinant of the tanh is the numerically stable
        `2 * (log 2 - u - softplus(-2u))` rather than `log(1 - tanh(u)^2)`,
        which underflows to `-inf` for |u| past ~9 and would make an actor
        loss NaN at exactly the saturation SAC drives the policy toward.
        `tfb.Tanh` does the same thing internally.
        """

        def __init__(self, n_obs: int, n_act: int, hidden_dim: int, num_blocks: int,
                     scaler_init: float, scaler_scale: float, alpha_init: float,
                     alpha_scale: float, c_shift: float, expansion: int,
                     log_std_min: float, log_std_max: float,
                     device: Any = None) -> None:
            super().__init__()
            self.n_act = n_act
            self.embedder = HyperEmbedder(n_obs, hidden_dim, scaler_init,
                                          scaler_scale, c_shift, device=device)
            self.encoder = nn.Sequential(*[
                HyperLERPBlock(hidden_dim, scaler_init, scaler_scale, alpha_init,
                               alpha_scale, expansion, device=device)
                for _ in range(num_blocks)])
            self.predictor = HyperNormalTanhPolicy(
                hidden_dim, n_act, 1.0, 1.0, log_std_min, log_std_max, device=device)

        def forward(self, obs: Any) -> Tuple[Any, Any]:
            return self.predictor(self.encoder(self.embedder(obs)))

        def sample(self, obs: Any, temperature: float = 1.0
                   ) -> Tuple[Any, Any]:
            """`(action, log_prob)`; `temperature=0.0` is the deterministic
            `tanh(mean)` with a log-prob of zeros (a point mass has no
            density, and no caller reads it on that path)."""
            mean, log_std = self(obs)
            if temperature == 0.0:
                return torch.tanh(mean), torch.zeros(mean.shape[0],
                                                     device=mean.device,
                                                     dtype=mean.dtype)
            std = torch.exp(log_std) * temperature
            u = mean + std * torch.randn_like(mean)
            action = torch.tanh(u)
            # Normal log-density, summed over the action dims (upstream's
            # MultivariateNormalDiag event shape) ...
            log_prob = (-0.5 * ((u - mean) / std).pow(2)
                        - log_std - 0.5 * math.log(2.0 * math.pi)
                        - math.log(temperature)).sum(-1)
            # ... minus the tanh log-determinant.
            log_prob = log_prob - (2.0 * (math.log(2.0) - u
                                          - F.softplus(-2.0 * u))).sum(-1)
            return action, log_prob

    class SimbaV2Critic(nn.Module):
        """ONE Q head: `simbaV2_network.py:65 SimbaV2Critic`."""

        def __init__(self, n_obs: int, n_act: int, hidden_dim: int, num_blocks: int,
                     scaler_init: float, scaler_scale: float, alpha_init: float,
                     alpha_scale: float, c_shift: float, expansion: int,
                     num_bins: int, min_v: float, max_v: float,
                     device: Any = None) -> None:
            super().__init__()
            self.embedder = HyperEmbedder(n_obs + n_act, hidden_dim, scaler_init,
                                          scaler_scale, c_shift, device=device)
            self.encoder = nn.Sequential(*[
                HyperLERPBlock(hidden_dim, scaler_init, scaler_scale, alpha_init,
                               alpha_scale, expansion, device=device)
                for _ in range(num_blocks)])
            self.predictor = HyperCategoricalValue(hidden_dim, num_bins, min_v, max_v,
                                                   1.0, 1.0, device=device)

        def forward(self, obs: Any, actions: Any) -> Tuple[Any, Any]:
            x = torch.cat([obs, actions], 1)
            return self.predictor(self.encoder(self.embedder(x)))

    class SimbaV2DoubleCritic(nn.Module):
        """`simbaV2_network.py:118 SimbaV2DoubleCritic`: two INDEPENDENT critics for clipped
        double Q. Upstream vmaps one definition over a leading axis with split
        parameter rngs; two modules in a container is the same set of
        parameters, and `num_qs != 2` is not ported (no published config asks
        for it and the update's per-sample pick is written for two)."""

        def __init__(self, **kw: Any) -> None:
            super().__init__()
            self.q1 = SimbaV2Critic(**kw)
            self.q2 = SimbaV2Critic(**kw)
            self.register_buffer(
                "bin_values",
                torch.linspace(float(kw["min_v"]), float(kw["max_v"]),
                               int(kw["num_bins"]), device=kw.get("device")).view(1, -1))
            self.num_bins = int(kw["num_bins"])
            self.min_v = float(kw["min_v"])
            self.max_v = float(kw["max_v"])

        def forward(self, obs: Any, actions: Any) -> Tuple[Any, Any, Any, Any]:
            v1, lp1 = self.q1(obs, actions)
            v2, lp2 = self.q2(obs, actions)
            return v1, lp1, v2, lp2

    class Temperature(nn.Module):
        """`simbaV2_network.py:168 SimbaV2Temperature`: `exp(log_temp)`, one scalar
        parameter, initialised at `log(temp_initial_value)`."""

        def __init__(self, initial_value: float = 0.01, device: Any = None) -> None:
            super().__init__()
            self.log_temp = nn.Parameter(
                torch.tensor(float(math.log(initial_value)), device=device))

        def forward(self) -> Any:
            return torch.exp(self.log_temp)

    def categorical_td_target(log_probs_next: Any, reward: Any, done: Any,
                              actor_entropy: Any, discount: Any, bin_values: Any,
                              num_bins: int, min_v: float, max_v: float) -> Any:
        """`simbaV2_update.py:96 categorical_td_loss` `categorical_td_loss`, the TARGET half.

        `target = reward + gamma^n * (bin_values - temp * log pi(a'|s')) * (1 - done)`,
        clipped to the support and projected back onto the bins by linear
        interpolation. Two things about it are NOT the C51 projection
        `fasttd3._categorical_projection` implements, and both matter:

          * the ENTROPY TERM inside the bracket. This is what makes the critic
            a SOFT value function -- the target is the entropy-regularised
            return, not the return. `fasttd3`'s projection has no such term
            because TD3 has no entropy bonus.
          * `done` is `terminated` ONLY (`simbaV2_update.py:190`,
            `batch["terminated"]`), so a TRUNCATED episode still bootstraps.
            `fasttd3` computes `bootstrap = truncations | ~dones`, which is the
            same statement written the other way round.

        Upstream also does NOT shift a bound when `b` lands exactly on an atom
        (`fast_td3.py:131-137` does), and this port follows upstream: at
        `l == u` its `m_l` factor becomes `u + 1 - b == 1` and `m_u` becomes
        `b - l == 0`, so the whole mass lands on the one atom rather than being
        lost. The two formulations agree; only FastTD3's needs the shift,
        because it does not add the `(l == u)` term.
        """
        reward = reward.view(-1, 1)
        done = done.view(-1, 1)
        actor_entropy = actor_entropy.view(-1, 1)
        discount = discount.view(-1, 1)
        target = reward + discount * (bin_values - actor_entropy) * (1.0 - done)
        target = target.clamp(min_v, max_v)
        b = (target - min_v) / ((max_v - min_v) / (num_bins - 1))
        low = torch.floor(b)
        up = torch.ceil(b)
        probs_next = torch.exp(log_probs_next)            # (B, num_bins)
        m_low = probs_next * (up + (low == up).to(b.dtype) - b)
        m_up = probs_next * (b - low)
        out = torch.zeros_like(probs_next)
        out.scatter_add_(1, low.long().clamp(0, num_bins - 1), m_low)
        out.scatter_add_(1, up.long().clamp(0, num_bins - 1), m_up)
        return out.detach()

    ns = {"torch": torch, "nn": nn, "F": F,
          "l2normalize": l2normalize, "project_": project_,
          "Scaler": Scaler, "HyperDense": HyperDense,
          "SimbaV2Actor": SimbaV2Actor,
          "SimbaV2Critic": SimbaV2Critic,
          "SimbaV2DoubleCritic": SimbaV2DoubleCritic,
          "Temperature": Temperature,
          "categorical_td_target": categorical_td_target}
    # The normalisers and the buffer come from `fasttd3`'s namespace, which is
    # where this repo's single port of each lives (see the note above).
    fns = _fasttd3._torch_classes()
    for shared in ("EmpiricalNormalization", "RewardNormalizer", "SimpleReplayBuffer"):
        ns[shared] = fns[shared]
    _TORCH_NS = ns
    return ns


def _resolve_arch(cfg: Any) -> str:
    """The architecture this backend EXECUTES for `train.architecture`.

    `train.architecture` is free-form and the published configs pin it as a
    CITATION (`rda.yaml`: `simba`, the family name §5.1's prose gives;
    `rda_humanoidbench.yaml`: `simba_v2`, Table 1's value). This backend
    implements exactly one network and says so rather than silently accepting
    any string -- the warn-and-record shape `fasttd3._resolve_arch` uses, with
    the EXECUTED value in `seed_metrics.architecture`.
    """
    arch = str(cfg.get("train.architecture", "simba_v2") or "simba_v2")
    if arch not in _ARCHITECTURES:
        log.warning("train.architecture=%r is not implemented by "
                    "train.backend=simba_v2 (it implements %s -- `simba` is "
                    "rda.yaml's citation for the same family); training "
                    "SimbaV2, and the seed rows record that",
                    arch, list(_ARCHITECTURES))
        arch = "simba_v2"
    return arch


def _build_networks(ns: Dict[str, Any], hyper: Dict[str, Any], der: Dict[str, Any],
                    n_obs: int, n_act: int, device: Any
                    ) -> Tuple[Any, Any, Any, Any, Any]:
    """Actor, critic, target critic, temperature and observation normaliser,
    UNSEEDED and UNSYNCED: `_build_agent` seeds torch before calling this,
    copies the critic into its target after, and projects all three onto the
    sphere. One constructor, so the network a blob is loaded into is the
    network it was saved from (`simba_v2_policy_from_blob` calls this too).
    """
    actor = ns["SimbaV2Actor"](
        n_obs=n_obs, n_act=n_act,
        hidden_dim=int(hyper["actor_hidden_dim"]),
        num_blocks=int(hyper["actor_num_blocks"]),
        scaler_init=der["actor_scaler_init"], scaler_scale=der["actor_scaler_scale"],
        alpha_init=der["actor_alpha_init"], alpha_scale=der["actor_alpha_scale"],
        c_shift=float(hyper["actor_c_shift"]), expansion=int(hyper["expansion"]),
        log_std_min=float(hyper["log_std_min"]),
        log_std_max=float(hyper["log_std_max"]), device=device)
    critic_kw = dict(
        n_obs=n_obs, n_act=n_act,
        hidden_dim=int(hyper["critic_hidden_dim"]),
        num_blocks=int(hyper["critic_num_blocks"]),
        scaler_init=der["critic_scaler_init"], scaler_scale=der["critic_scaler_scale"],
        alpha_init=der["critic_alpha_init"], alpha_scale=der["critic_alpha_scale"],
        c_shift=float(hyper["critic_c_shift"]), expansion=int(hyper["expansion"]),
        num_bins=int(hyper["critic_num_bins"]),
        min_v=der["critic_min_v"], max_v=der["critic_max_v"], device=device)
    # `critic_use_cdq: false` is upstream's non-episodic branch
    # (`simbaV2_agent.py:137-149 (the `else` branch)` builds the SINGLE critic there); both
    # branches are built as the double module here and the SECOND head is
    # simply never consulted, because a second Q head costs one forward pass
    # and a second module class costs a divergence. `critic_use_cdq` on the
    # seed row is what says which was used.
    qnet = ns["SimbaV2DoubleCritic"](**critic_kw)
    qnet_target = ns["SimbaV2DoubleCritic"](**critic_kw)
    temp = ns["Temperature"](float(hyper["temp_initial_value"]), device=device)
    obs_normalizer = None
    if bool(hyper["normalize_observation"]):
        obs_normalizer = ns["EmpiricalNormalization"](shape=n_obs, device=device)
    return actor, qnet, qnet_target, temp, obs_normalizer


def _greedy_policy(ns: Dict[str, Any], actor: Any, obs_normalizer: Any,
                   features: Callable[[Any], np.ndarray],
                   to_env_action: Callable[[np.ndarray], np.ndarray],
                   device: Any) -> Callable[[np.ndarray], np.ndarray]:
    """THE action path: raw state -> feature -> normalise WITHOUT updating ->
    actor at SAMPLING TEMPERATURE 0 -> the env's action box.

    Temperature 0 is upstream's own evaluation path
    (`simbaV2_agent.py:336-338`: `temperature = 1.0` if training else `0.0`),
    which collapses the tanh-Gaussian to `tanh(mean)`. So the greedy policy is
    the exploring policy at a different temperature and not a second
    implementation that could differ in the squash or the normaliser
    invisibly -- `fasttd3._greedy_policy`'s argument, and the reason
    `_policy_fn` and `simba_v2_policy_from_blob` both come here.
    """
    torch = ns["torch"]

    def policy(s: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(features(s), device=device).unsqueeze(0)
            if obs_normalizer is not None:
                x = obs_normalizer(x, update=False)
            a, _lp = actor.sample(x, temperature=0.0)
            a = a.squeeze(0).float().cpu().numpy()
        return to_env_action(a)

    return policy


def _snapshot_agent(agent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """`train.checkpoint_selection`'s snapshot. `fasttd3._snapshot_agent`'s
    four state dicts PLUS the temperature, which is a learned parameter of
    this learner and not of that one -- restoring a checkpoint without it
    would ship a policy beside an entropy coefficient from a later point in
    training, and the pair is what the actor loss is."""
    out: Dict[str, Any] = {}
    for name in ("actor", "qnet", "qnet_target", "temp", "obs_normalizer"):
        mod = agent.get(name)
        if mod is not None:
            out[name] = {k: v.detach().clone() for k, v in mod.state_dict().items()}
    return out


def _restore_agent(agent: Dict[str, Any], snap: Optional[Dict[str, Any]]) -> bool:
    if snap is None:
        return False
    try:
        for name, sd in snap.items():
            agent[name].load_state_dict(sd)
        return True
    except Exception as exc:  # noqa: BLE001 - a failed restore ships the final weights
        log.warning("simba_v2: could not restore the selected checkpoint "
                    "(%s: %s); shipping the final one", type(exc).__name__, exc)
        return False


def _blob(ns: Dict[str, Any], arch: str, agent: Dict[str, Any]) -> Optional[_PolicyBlob]:
    """Serialise the agent for `_POLICY_STORE`. Tensors only plus the format
    tag, so the reader can load it with `weights_only=True` -- run directories
    may live on a shared, world-writable mount, and a pickle-bearing checkpoint
    is the attack `checkpoint._DECODABLE` closes."""
    torch = ns["torch"]
    try:
        buf = io.BytesIO()
        payload = {"format": _BLOB_FORMAT, "architecture": arch}
        for name in ("actor", "qnet", "qnet_target", "temp"):
            payload[name] = agent[name].state_dict()
        payload["obs_normalizer"] = (agent["obs_normalizer"].state_dict()
                                     if agent.get("obs_normalizer") is not None else {})
        torch.save(payload, buf)
        return _PolicyBlob(buf.getvalue())
    except Exception as exc:  # noqa: BLE001 - a missing checkpoint is not a failed run
        log.warning("simba_v2: could not serialise the policy for train.init "
                    "(%s: %s); this candidate will carry no policy_ref",
                    type(exc).__name__, exc)
        return None


def _payload(ns: Dict[str, Any], arch: str, blob: Any, ref: str) -> Dict[str, Any]:
    """Decode a `_POLICY_STORE` blob, or raise `ValueError` naming why it is
    not one of ours. `fasttd3._fasttd3_payload`'s contract, with OUR format
    tag: the two backends write torch containers of the same shape, so the tag
    is the only thing that distinguishes a simba_v2 checkpoint from a fasttd3
    one and a misroute would rebuild a policy under the wrong learner."""
    if not isinstance(blob, np.ndarray) or blob.dtype != np.uint8:
        raise ValueError(f"{ref} holds a {type(blob).__name__}, which is not a "
                         "simba_v2 blob")
    torch = ns["torch"]
    payload = torch.load(io.BytesIO(np.asarray(blob, dtype=np.uint8).tobytes()),
                         map_location="cpu", weights_only=True)
    if not (isinstance(payload, dict) and payload.get("format") == _BLOB_FORMAT):
        raise ValueError(f"not a {_BLOB_FORMAT} payload (an sb3 zip, a fasttd3/v1 "
                         "checkpoint, or another backend's?)")
    if payload.get("architecture") != arch:
        raise ValueError(f"architecture {payload.get('architecture')!r} does not "
                         f"match train.architecture={arch!r}")
    return payload


def _apply(ns: Dict[str, Any], arch: str, agent: Dict[str, Any], blob: Any,
           ref: str) -> bool:
    """Load a stored blob into live networks. Returns whether it took.

    Same leniency contract as `fasttd3._fasttd3_apply`: a blob another backend
    wrote, or one whose shapes no longer fit, is a WARNING and a cold start,
    never a crash -- a checkpoint that no longer fits must not destroy the
    search that found it. There is no per-run exploration buffer to preserve
    here (this actor's stochasticity is its own `log_std` head, not a
    `noise_scales` buffer), so every tensor in the payload is loaded.
    """
    if blob is None:
        return False
    try:
        payload = _payload(ns, arch, blob, ref)
        agent["actor"].load_state_dict(payload["actor"])
        agent["qnet"].load_state_dict(payload["qnet"])
        agent["qnet_target"].load_state_dict(payload["qnet_target"])
        agent["temp"].load_state_dict(payload["temp"])
        if payload.get("obs_normalizer") and agent.get("obs_normalizer") is not None:
            agent["obs_normalizer"].load_state_dict(payload["obs_normalizer"])
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("train.init: %s did not load into this model (%s: %s) -- "
                    "training from scratch", ref, type(exc).__name__, exc)
        return False


def blob_format(blob: Any) -> str:
    """The `format` tag of a torch-container policy blob, or `""`.

    `training._blob_kind` tells an sb3 archive from a torch container by the
    container's members and cannot go further: `fasttd3._fasttd3_blob` and
    `_blob` above write the same container shape. So `policy_from_ref` asks
    THIS before choosing a rebuild, matches the tag against the two learners
    that write one, and refuses anything else AT THE ROUTING SITE rather than
    handing it to whichever branch happens to be the else. With `fasttd3` as
    the fall-through, an unknown tag would reach it and be refused by its
    format check, naming fasttd3's format for a blob that was not fasttd3's.

    torch is imported here and not at module scope for the usual reason. Any
    failure is `""` -- "not one I can name" -- because the caller's next move
    is a refusal either way and a provenance read must not raise.
    """
    try:
        import torch
    except ImportError:
        return ""
    try:
        payload = torch.load(io.BytesIO(np.asarray(blob, dtype=np.uint8).tobytes()),
                             map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        return ""
    return str(payload.get("format", "")) if isinstance(payload, dict) else ""


def simba_v2_policy_from_blob(cfg: Any, env: Any, phi: Any, blob: Any,
                              ref: str) -> Callable[[np.ndarray], np.ndarray]:
    """The simba_v2 branch of `training.policy_from_ref`: the run's
    architecture, the blob's weights, the loop's own action path.

    Mirrors `fasttd3.fasttd3_policy_from_blob` exactly, including the reasons:
    the payload is decoded and CHECKED by `_payload` (the format tag and the
    architecture), the networks are built by `_build_networks` under the
    resolved `train.hyperparameters` -- the same constructor the training loop
    used -- and the callable returned is `_greedy_policy`, the very function
    `_policy_fn` hands the loop. On the host (`cpu`): every caller rolls one
    state at a time, so a device round trip per action would cost more than
    the forward pass.
    """
    if cfg is None or not hasattr(cfg, "get"):
        raise ValueError(
            f"{ref}: a simba_v2 policy blob can only be rebuilt with the run's "
            "config -- the network shape is `train.hyperparameters` "
            "(actor_hidden_dim, actor_num_blocks, critic_num_bins, ...) and a "
            "blob carries weights, not the constructor that fits them")
    ns = _torch_classes()
    arch = _resolve_arch(cfg)
    hyper = _simba_hyper(cfg)
    n_obs = int(phi.dim) if phi is not None else int(np.asarray(env.obs_low).size)
    n_act = int(np.asarray(env.action_low).size)
    der = _derived(hyper, n_act)
    device = ns["torch"].device("cpu")
    actor, qnet, qnet_target, temp, obs_normalizer = _build_networks(
        ns, hyper, der, n_obs, n_act, device)
    agent = {"actor": actor, "qnet": qnet, "qnet_target": qnet_target,
             "temp": temp, "obs_normalizer": obs_normalizer}
    # RAISES, NEVER DEGRADES -- `policy_from_ref`'s contract: there is no
    # training after this load, the callable IS the measurement, and an
    # untrained network wearing a stored policy's ref is a corrupted
    # measurement wearing a healthy one's clothes. The ref is prefixed onto
    # every message here for the same reason `fasttd3_policy_from_blob` does
    # it: `_payload`'s format and architecture branches do not carry it.
    try:
        payload = _payload(ns, arch, blob, ref)
    except ValueError as exc:
        raise ValueError(f"{ref}: {exc}") from exc
    try:
        actor.load_state_dict(payload["actor"])
        temp.load_state_dict(payload["temp"])
        if payload.get("obs_normalizer") and obs_normalizer is not None:
            obs_normalizer.load_state_dict(payload["obs_normalizer"])
    except Exception as exc:  # noqa: BLE001 -- a shape that does not fit is the finding
        raise ValueError(
            f"{ref}: the stored parameters do not fit a simba_v2 network built "
            f"from this config's train.hyperparameters ({type(exc).__name__}: "
            f"{exc}); the blob was trained under different dimensions") from exc
    actor.eval()
    if obs_normalizer is not None:
        obs_normalizer.eval()
    del agent, qnet, qnet_target

    def _features(s: Any) -> np.ndarray:
        return np.asarray(phi(s) if phi is not None else s, dtype=np.float32)

    center, half = _action_affine(env)

    def to_env_action(a_agent: np.ndarray) -> np.ndarray:
        return center + np.asarray(a_agent, dtype=float) * half

    return _greedy_policy(ns, actor, obs_normalizer, _features, to_env_action, device)


@register("train_backend", "simba_v2")
def simba_v2_backend(ctx: Any, state: Any, candidate: Candidate, n_seeds: int = 1,
                     env_steps: Optional[int] = None,
                     resume_ref: Optional[str] = None,
                     seed_phase: str = "",
                     handoff: Optional[Any] = None) -> TrainResult:
    """SAC + SimbaV2 against the candidate reward, on `train.n_parallel_envs`
    slots of the adapter.

    Same shape as `_sb3_run` and `fasttd3_backend` deliberately -- chunked
    training with a real checkpoint curve, `_pruner_for`/`_selection_for`/
    `train.timeout_s` consulted between chunks, `train.init` read off the same
    state slots, the last seed's agent published to `_POLICY_STORE` -- so every
    §3 key means the same thing here as on the other backends. What differs is
    the learner: the update is the line-faithful port documented at the top of
    this module.

    THE FOUR BUDGET KEYS, and this is the one place they are related, so it is
    worth reading once:

        train.env_steps          SIMULATOR steps per seed, upstream's own
                                 accounting (`run_online.py:160` logs
                                 `interaction_step * action_repeat *
                                 num_train_envs`). RDA's HumanoidBench column
                                 is 10,000,000.
        train.n_parallel_envs    the PAPER'S TRAINING ENVS -- upstream's
                                 `num_train_envs`, RDA's 16. On `sb3` this key
                                 is episodes per round and on `fasttd3` it is
                                 forked slot workers; here it is the fleet
                                 width, which is what `rda_humanoidbench.yaml`
                                 documents it as ("§5.1, HumanoidBench").
                                 The SCHEDULE that key means on `fasttd3` is
                                 `train.hyperparameters.env_workers` here, so
                                 the two are separate knobs and neither is
                                 overloaded.
        action_repeat            simulator steps per agent decision. 2 on
                                 HumanoidBench upstream.
        updates_per_interaction_step  gradient steps per decision; `None` ->
                                 `action_repeat`, which is upstream's
                                 expression of it.

    so

        interaction_steps = env_steps // (n_parallel_envs * action_repeat)
        updates           = interaction_steps * updates_per_interaction_step

    and at RDA's column (10M, 16, 2, 2) that is 312,500 decisions and
    **625,000 updates** -- Table 1's number, reached by the paper's own
    arithmetic rather than asserted. Both figures land on the seed row as
    `interaction_steps` and `update_steps`, so a run can be audited against
    this docstring.

    `handoff` (`train.interaction: shared_population`) is honoured for its
    SLICE INDEX only -- the seed salt -- not for the replay half, exactly as on
    `fasttd3`, and `_check_coherence` refuses `interaction_cfg.shared_buffer`
    beside this backend rather than let a declared pool not execute.
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError(
            "train.backend: simba_v2 needs torch, which is not installed. Either\n"
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
            "train.backend=simba_v2 needs a continuous-action env (a tanh-Gaussian "
            f"policy has no discrete head); {getattr(env, 'name', '?')} is discrete. "
            "Use train.backend: tabular or mock there.")
    arch = _resolve_arch(cfg)

    try:
        # TWO SHARED DISPATCHES, NOT A LOCAL `compile_reward`, and this call is
        # the whole of what stops `simba_v2` from being the backend the inner
        # one was written to prevent.
        #
        # `_training_reward` honours `train.reward_source` (`candidate` |
        # `reference`) -- whether the LEARNER trains on the generator's
        # program or on the environment's own shipped reward, which is the ORACLE
        # arm of Text2Reward's Fig. 2 and CARD's Fig. 3. Its docstring says why
        # it is one function in as many words: *"so a fourth backend cannot
        # quietly train on a different channel than the other three."* A backend
        # that compiled `candidate.reward_code` directly would train the
        # candidate's reward under `train.reward_source: reference` while
        # `config.resolved.yaml` recorded `reference` -- a declared-but-unread
        # key ON ONE BACKEND ONLY, the worst-behaved member of that class
        # precisely because every test it could fail is green. The
        # family-enumerating spy (`tests/test_reward_source.py::
        # test_every_backend_routes_its_reward_through_the_dispatch`) keeps it
        # honest: it parametrises over the registry, so this backend is inside
        # its coverage the moment it is registered, and it fails on the spy
        # never being called rather than on a fitness number.
        #
        # `_scaled_reward` then applies `train.reward_scaling` (LaRes Eq. 3)
        # between compilation and training, as the other three sites do: a pure
        # APPLY of the plan stage 3 left on `candidate.meta` in the parent
        # (`population.scaling_elite_moments`, `bird.py::train`), so it reads
        # nothing off `ctx` and is safe inside a forked worker. `none` plans
        # nothing and is the identity, which is what every published config
        # resolves to.
        reward = _training._scaled_reward(
            ctx, state, candidate,
            _training._training_reward(ctx, state, candidate))
    except Exception as exc:  # noqa: BLE001
        ctx.budget.record_training(env_steps=0, gpu_seconds=time.monotonic() - t0)
        result.trained, result.error = False, f"{type(exc).__name__}: {exc}"
        return result

    # The co-designed observation -- same refusal as every other backend: no
    # silent fall back to the raw state, or a co-designing method quietly
    # becomes its reward-only ablation.
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

    hyper = _simba_hyper(cfg)
    dev_name = str(hyper["device"] or "cpu")
    requested = dev_name
    if dev_name == "auto":
        # `auto` KEEPS its fallback: it means "use a GPU if there is one".
        dev_name = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_name.startswith("cuda") and not torch.cuda.is_available():
        # AN EXPLICIT `cuda` IS A REFUSAL, not a downgrade with a warning --
        # `fasttd3_backend`'s rule and its reason: a run can train on cpu
        # while metadata built from a probe tensor reports a live cuda
        # device. Asking for cuda and silently getting cpu is a measurement
        # that looks like the one you wanted.
        raise RuntimeError(
            f"train.hyperparameters.device={requested!r} but torch.cuda.is_available() "
            "is False on this machine. Refused rather than run on cpu: the fallback "
            "also disables torch.compile, and AMP where a config enabled it (both "
            "are `device.type == 'cuda'` below; note `amp` DEFAULTS OFF on this "
            "backend, because upstream SimbaV2 is float32 throughout -- so unlike "
            "fasttd3 the AMP half of this bites only a config that asked for it). "
            "The run would be several times slower than the number it would be "
            "compared against, and would say so nowhere a reader looks. "
            "Use device: auto to accept a cpu fallback deliberately, or fix the "
            "allocation -- under a cluster scheduler this usually means the job "
            "was allocated no GPU.")
    device = torch.device(dev_name)
    amp_enabled = bool(hyper["amp"]) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(hyper["amp_dtype"]) == "bf16" else torch.float16
    compile_on = (bool(hyper["compile"]) if hyper["compile"] is not None
                  else device.type == "cuda")

    # sb3-backend features a config can enable that this learner cannot
    # honour: train without them and SAY SO -- the kl_clone-on-non-PPO shape.
    if str(cfg.get("train.anchor.kind", "none") or "none") != "none":
        log.warning("train.anchor.kind=%s is implemented on the sb3 backend only "
                    "(it clones an SB3 policy); train.backend=simba_v2 trains "
                    "unanchored", cfg.get("train.anchor.kind"))
    if bool(cfg.get("train.reference_policy.enabled", False)):
        log.warning("train.reference_policy is implemented on the sb3 backend only; "
                    "train.backend=simba_v2 trains unregularised")
    if str(cfg.get("train.elite_constraint.kind", "none") or "none") != "none":
        log.warning("train.elite_constraint.kind=%s (LaRes Eq. 4) is implemented on "
                    "the sb3 and fasttd3 backends; train.backend=simba_v2 trains "
                    "unconstrained and the seed row records "
                    "elite_constraint_optimisers: 0",
                    cfg.get("train.elite_constraint.kind"))

    # -- the four budget keys, related exactly as the docstring says ---------
    num_envs = max(1, int(cfg.get("train.n_parallel_envs", 1) or 1))
    action_repeat = max(1, int(hyper["action_repeat"]))
    upd_per_step = hyper["updates_per_interaction_step"]
    # `None` -> `action_repeat`: upstream writes `${action_repeat}` rather than
    # a number (`configs/online_rl.yaml:28`), and copying the number would let
    # a config change the repeat and silently keep the old update rate.
    upd_per_step = action_repeat if upd_per_step is None else max(1, int(upd_per_step))
    n_obs = int(phi.dim) if phi is not None else int(np.asarray(env.obs_low).size)
    n_act = int(np.asarray(env.action_low).size)
    der = _derived(hyper, n_act)
    norm_mode = cfg.get("train.reward_norm", "none")

    dr_params = _dr_params(cfg, candidate)
    env.set_dr(dr_params)
    total_steps = int((env_steps if env_steps is not None
                       else cfg.get("train.env_steps", 20000)) or 20000)
    iters = max(1, total_steps // (num_envs * action_repeat))
    update_steps = iters * upd_per_step
    # `env_workers`: forked slot workers behind `_VecEnvView`, a SCHEDULE
    # (bit-identical to 1). NOT `train.n_parallel_envs`, which on this backend
    # is the paper's env count -- see the docstring.
    env_workers = max(1, int(hyper["env_workers"]))
    # `learning_starts` and `batch_size` are upstream's TOTALS
    # (`configs/buffer/numpy_uniform.yaml`); the buffer is per-env, so both
    # are divided here and BOTH figures go on the seed row.
    batch_per_env = max(1, int(hyper["batch_size"]) // num_envs)
    start_iters = max(1, -(-int(hyper["learning_starts"]) // num_envs))
    # The buffer is `(num_envs, rows)` on the device and `extend` is called
    # once per interaction step, so rows beyond `iters + 1` are never written
    # and never sampled -- capping is bit-neutral and is what keeps a
    # 1M-transition buffer from allocating 1M rows for a 20k-step tester run.
    # The published total is still what the config says; `buffer_rows` on the
    # seed row is what was allocated.
    buffer_rows = min(max(1, -(-int(hyper["buffer_size"]) // num_envs)), iters + 1)
    # Same checkpoint count rule as every backend (on the interaction budget,
    # so a config moved between backends keeps its curve length); each chunk
    # is that share of INTERACTION STEPS, and the plan PARTITIONS the budget
    # exactly rather than `max(1, iters // n_chunks)` per chunk, which trains
    # past `train.env_steps` on a budget smaller than the checkpoint count.
    n_chunks = max(MIN_CHECKPOINTS,
                   min(MAX_CHECKPOINTS, total_steps // 2048 or MIN_CHECKPOINTS))
    chunk_plan = [iters // n_chunks + (1 if ck < iters % n_chunks else 0)
                  for ck in range(n_chunks)]
    base_seed = _seed_base(ctx, state, candidate, seed_phase)
    rule, prune_field = _pruner_for(cfg)
    prune_metric = str(cfg.get("train.pruning_metric", "own_reward") or "own_reward")
    select_rule, select_min_delta = _selection_for(cfg)
    ceiling = _demo_ceiling(ctx, state, candidate, reward) if _ceiling_wanted(cfg) else None
    timeout_s = float(cfg.get("train.timeout_s", 3600) or 3600)
    if iters < start_iters + 1:
        # A short screen or a tiny budget can sit entirely inside the warm-up:
        # the run then measures the untrained init, which is a true statement
        # about the learner under that budget -- but only if somebody can see
        # it happened.
        log.warning("simba_v2: %d interaction step(s) (%d env steps / %d envs / "
                    "action_repeat %d) with learning_starts=%s transitions "
                    "(%d steps) -- no gradient step will run; lower "
                    "train.hyperparameters.learning_starts for short budgets",
                    iters, total_steps, num_envs, action_repeat,
                    hyper["learning_starts"], start_iters)

    # -- train.init, off the same state slots as every backend ---------------
    init = _training_init(ctx, state, cfg, resume_ref, candidate)
    init_ref = init.ref
    result.init_source, result.init_from_cand_id = init.source, init.from_cand_id
    result.init_similarity = init.similarity
    init_blob = _POLICY_STORE.get(init_ref or "") if init_ref else None
    if init_ref and init_blob is None:
        log.warning("train.init=%s: %s is not in _POLICY_STORE "
                    "(evicted, or written by another process) -- training from "
                    "scratch", cfg.get("train.init", "from_scratch"), init_ref)
    if init.ratio > 0.0 and len(init.secondary or ()):
        log.warning("train.init=secondary_replay_buffer is not wired on "
                    "train.backend=simba_v2 (the backend neither pre-fills its "
                    "buffer from the round pool nor exports what it added); this "
                    "candidate trains on its own transitions only, and the seed "
                    "row records secondary_transitions: 0")

    per_seed_curves: List[List[Dict[str, float]]] = []
    per_seed_components: List[List[Dict[str, float]]] = []
    want_components = "reward_component_values" in set(cfg.get("train.log", []) or [])
    policies: List[Callable[[np.ndarray], np.ndarray]] = []
    used = 0
    fatal_error = ""
    gamma = float(hyper["gamma"])

    def _build_agent(seed: int) -> Dict[str, Any]:
        """Networks + normalisers, seeded, target synced, and PROJECTED ONTO
        THE SPHERE -- `simbaV2_agent.py:193-195` l2-normalises the actor, the
        critic and the target critic immediately after initialisation, before
        any gradient step, so the orthogonal init's scale is not what trains.

        `torch.manual_seed` covers every torch draw (init, the actor's own
        sampling, batch indices), which is what makes two runs of one seed
        identical. Upstream also seeds the global `random`/`numpy` streams
        (`run_online.py:39-40`); this port deliberately does not -- nothing
        here draws from them, and a backend that reseeded the process-global
        numpy stream would perturb every OTHER component's draws.
        """
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = bool(hyper["torch_deterministic"])
        actor, qnet, qnet_target, temp, obs_normalizer = _build_networks(
            ns, hyper, der, n_obs, n_act, device)
        qnet_target.load_state_dict(qnet.state_dict())
        # THE INIT PROJECTION: all three networks, matching
        # `simbaV2_agent.py:193-195`. The per-NETWORK counts are kept, not just
        # the sum, because they are what makes the per-STEP counter below an
        # EQUALITY rather than a second non-zero (`n_projection_kernels ==
        # update_steps * (k_actor + k_critic)`), and an equality is the only
        # form that catches a projection that stopped happening.
        k_actor = ns["project_"](actor)
        k_critic = ns["project_"](qnet)
        k_target = ns["project_"](qnet_target)
        reward_normalizer = None
        if bool(hyper["normalize_reward"]):
            # g_max IS `normalized_g_max` -- the same key that sets the
            # critic's support, `simbaV2.yaml:10` and :38-39. FastTD3's port
            # defaults it to 10.0 and sizes the support from the raw-return
            # bounds instead; that pair is the departure named in the module
            # docstring.
            reward_normalizer = ns["RewardNormalizer"](
                gamma=gamma, device=device, g_max=float(hyper["normalized_g_max"]))
        return {"actor": actor, "qnet": qnet, "qnet_target": qnet_target,
                "temp": temp, "obs_normalizer": obs_normalizer,
                "reward_normalizer": reward_normalizer,
                "n_projected": int(k_actor + k_critic + k_target),
                "k_actor": int(k_actor), "k_critic": int(k_critic),
                # THE LIVE COUNTER, mutated by `update`. A dict rather than a
                # closure variable so the update closure can increment it
                # without `nonlocal`, which `torch.compile` handles worse.
                "proj": {"calls": 0, "kernels": 0}}

    def _norm_obs(agent: Dict[str, Any], x: Any, update: bool = True) -> Any:
        norm = agent["obs_normalizer"]
        return x if norm is None else norm(x, update=update)

    def _policy_fn(agent: Dict[str, Any], view: Any) -> Callable[[np.ndarray], np.ndarray]:
        return _greedy_policy(ns, agent["actor"], agent["obs_normalizer"], _features,
                              view.to_env_action, device)

    # ---------------------------------------------------------------- seeds --

    def run_seed(seed_i: int) -> Dict[str, Any]:
        """Train ONE seed. Same contract as `_sb3_run.run_seed`: pure of
        `result`, the accumulators and `ctx.budget`, so the body runs inline or
        inside a forked seed worker with the fold in the parent either way."""
        seed = _seed_for(base_seed, seed_i,
                         int(getattr(handoff, "index", 0) or 0))
        seed_t0 = time.monotonic()
        # ONE torch thread, whichever schedule is running -- `fasttd3`'s
        # reasons: it is what makes the seed fork bit-identical to the
        # sequential loop (thread-count reduction order), and upstream pins
        # OMP_NUM_THREADS=1 before importing torch on the CPU path too.
        with _single_thread_torch():
            return _run_one_seed(seed, seed_t0)

    def _run_one_seed(seed: int, seed_t0: float) -> Dict[str, Any]:
        import torch  # local: the fork path re-enters here in a fresh child
        from torch.amp import GradScaler, autocast

        # Before the agent: `_VecEnvView` forks its slot workers at
        # construction, and a fork should precede this process's first CUDA
        # allocation.
        if isinstance(env, BatchedEnvAdapter):
            if action_repeat > 1:
                raise ValueError(
                    "train.hyperparameters.action_repeat=%d is not implemented on "
                    "the jax family: `_JaxVecEnvView` steps its rows inside one "
                    "jitted region and a per-row repeat with a per-row break is a "
                    "different jitted function, not a loop around this one. "
                    "Refused rather than silently run at repeat 1, which would "
                    "halve the decision rate the config asked for and change the "
                    "effective discount horizon." % action_repeat)
            if env_workers > 1:
                log.warning(
                    "train.hyperparameters.env_workers=%s is ignored on the jax "
                    "family: %s steps %d rows inside one jitted region, with no "
                    "slot workers and no pipes.",
                    env_workers, type(env).__name__, num_envs)
            view = _JaxVecEnvView(env, num_envs, reward, _features, norm_mode,
                                  seed, n_obs)
        else:
            view = _VecEnvView(env, num_envs, reward, _features, norm_mode, seed,
                               n_obs, workers=env_workers,
                               action_repeat=action_repeat)
        agent = _build_agent(seed)
        actor, qnet, qnet_target = agent["actor"], agent["qnet"], agent["qnet_target"]
        temp = agent["temp"]
        rew_norm = agent["reward_normalizer"]
        proj = agent["proj"]

        # `optax.adam` on all three, with ONE linear schedule each
        # (`simbaV2_agent.py:114, :157, :183 (optax.adam)`): no weight decay,
        # no gradient clipping and no cosine -- all three of which `fasttd3`
        # has, and none of which upstream does.
        lr0 = float(hyper["learning_rate_init"])
        lr1 = float(hyper["learning_rate_end"])
        # `learning_rate_decay_step` = decay_rate * num_interaction_steps *
        # updates_per_interaction_step (`simbaV2.yaml:20`) -- i.e. the whole
        # UPDATE budget at the default rate 1.0. Indexed by OPTIMISER step, as
        # `optax.linear_schedule` is, so the schedulers below step once per
        # gradient step and not once per interaction step.
        decay_steps = max(1, int(round(float(hyper["learning_rate_decay_rate"])
                                       * update_steps)))
        opt_actor = torch.optim.Adam(actor.parameters(), lr=lr0)
        opt_critic = torch.optim.Adam(qnet.parameters(), lr=lr0)
        opt_temp = torch.optim.Adam(temp.parameters(), lr=lr0)

        def _lin(step: int) -> float:
            """`optax.linear_schedule(init, end, transition_steps)` as a
            multiplier on `lr0`: linear in the step count, CLAMPED at
            `end_value` past the horizon (optax holds the end value, it does
            not extrapolate)."""
            frac = min(1.0, step / decay_steps)
            return (lr0 + (lr1 - lr0) * frac) / lr0

        scheds = [torch.optim.lr_scheduler.LambdaLR(o, _lin)
                  for o in (opt_actor, opt_critic, opt_temp)]
        # THREE OPTIMISERS THROUGH ONE SCALER, and that is worth a line because
        # it is not the single-optimiser shape torch's own examples show. With
        # `amp: false` (this backend's default, because upstream is float32
        # throughout) and with bf16, `GradScaler(enabled=False)` makes every
        # scale/unscale/step/update below a pass-through, so the shape costs
        # nothing. Under fp16 it means the loss scale can be revised up to
        # three times per gradient step, which is conservative rather than
        # wrong -- the actor's revision reaches the critic's scaling in the
        # same step. `fasttd3` has the same shape with two optimisers.
        scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
        amp_device_type = "cuda" if device.type == "cuda" else "cpu"

        warm_started = bool(init_blob is not None
                            and _apply(ns, arch, agent, init_blob, init_ref or ""))

        rb = ns["SimpleReplayBuffer"](
            n_env=num_envs, buffer_size=buffer_rows, n_obs=n_obs, n_act=n_act,
            n_steps=int(hyper["n_step"]), gamma=gamma, device=device)

        use_cdq = bool(hyper["critic_use_cdq"])
        tau = float(hyper["target_tau"])
        num_bins = int(hyper["critic_num_bins"])
        min_v, max_v = der["critic_min_v"], der["critic_max_v"]
        target_entropy = der["temp_target_entropy"]
        use_clip = bool(hyper["use_grad_norm_clipping"])
        max_grad_norm = float(hyper["max_grad_norm"])

        def _clip(params: Any) -> None:
            if use_clip:
                torch.nn.utils.clip_grad_norm_(
                    params, max_norm=max_grad_norm if max_grad_norm > 0
                    else float("inf"))

        def update(data: Dict[str, Any]) -> None:
            """ONE gradient step -- `simbaV2_agent.py:247, :257, :263, :278`
            `_update_simbav2_networks`, in its order, which is load-bearing:

                actor  (against the LIVE critic and the OLD temperature)
                temp   (against the actor's entropy just measured)
                critic (against the NEW actor and the NEW temperature)
                target (soft, every step)

            Two orderings a reader might expect and upstream does not use: the
            critic first (SAC's usual presentation), and the temperature after
            the critic. Both change what the critic's target is built from.
            """
            observations = data["observations"]
            next_observations = data["next_observations"]
            actions = data["actions"]
            rewards = data["rewards"]
            # `terminated` ONLY: upstream passes `batch["terminated"]`
            # (`simbaV2_update.py:190`), so a truncated episode bootstraps.
            # `_VecEnvView` gives `dones` = terminated OR truncated and
            # `truncations` = truncated-and-not-terminated, so terminated is
            # the difference.
            terminated = (data["dones"].bool() & ~data["truncations"].bool()).float()
            discount = gamma ** data["effective_n_steps"].float()

            # -- actor ------------------------------------------------------
            with autocast(device_type=amp_device_type, dtype=amp_dtype,
                          enabled=amp_enabled):
                act_new, logp = actor.sample(observations)
                v1, _lp1, v2, _lp2 = qnet(observations, act_new)
                q = torch.minimum(v1, v2) if use_cdq else v1
                with torch.no_grad():
                    temp_now = temp()
                actor_loss = (logp * temp_now - q).mean()
            opt_actor.zero_grad(set_to_none=True)
            scaler.scale(actor_loss).backward()
            scaler.unscale_(opt_actor)
            _clip(actor.parameters())
            scaler.step(opt_actor)
            scaler.update()
            # THE PROJECTION, after the actor's step -- `simbaV2_update.py:91`.
            # COUNTED, and that is not bookkeeping: if these two calls
            # discarded their return values, the only projection field on the
            # seed row would be the INIT count -- so deleting both would leave
            # the artifact byte-identical, and a run whose actor and critic
            # never projected after initialisation would report exactly what a
            # correct one does. That run is FastTD3's port, which is the one
            # thing this module exists not to be. `n_projection_kernels` has a
            # PREDICTABLE value (`update_steps * (k_actor + k_critic)`), so it
            # is an equality a test can assert rather than another non-zero.
            proj["calls"] += 1
            proj["kernels"] += ns["project_"](actor)

            # -- temperature ------------------------------------------------
            # `temperature_loss = temp * (entropy - target_entropy)` on the
            # SCALAR entropy the actor step just reported
            # (`simbaV2_update.py:261 update_temperature`, fed `actor_info["actor/entropy"]`
            # = `-log_probs.mean()`). `temp` is `exp(log_temp)`, so the
            # gradient reaches `log_temp` through the exp.
            entropy = (-logp).mean().detach()
            with autocast(device_type=amp_device_type, dtype=amp_dtype,
                          enabled=amp_enabled):
                temp_loss = temp() * (entropy - target_entropy)
            opt_temp.zero_grad(set_to_none=True)
            scaler.scale(temp_loss).backward()
            scaler.unscale_(opt_temp)
            scaler.step(opt_temp)
            scaler.update()

            # -- critic -----------------------------------------------------
            with autocast(device_type=amp_device_type, dtype=amp_dtype,
                          enabled=amp_enabled):
                with torch.no_grad():
                    temp_new = temp()
                    next_act, next_logp = actor.sample(next_observations)
                    # `next_actor_entropy = temperature() * log_probs`
                    # (`simbaV2_update.py:156`) -- note the SIGN: it is
                    # +temp*log_pi, subtracted from the bins below, which is
                    # the entropy BONUS.
                    next_entropy_term = temp_new * next_logp
                    nv1, nlp1, nv2, nlp2 = qnet_target(next_observations, next_act)
                    if use_cdq:
                        # THE PER-SAMPLE PICK: the target log-probs of
                        # whichever head has the lower expected value
                        # (`simbaV2_update.py:165-168`), not the lower value
                        # itself and not an average of the distributions.
                        pick = (nv1 <= nv2).view(-1, 1)
                        next_lp = torch.where(pick, nlp1, nlp2)
                    else:
                        next_lp = nlp1
                    target = ns["categorical_td_target"](
                        next_lp, rewards, terminated, next_entropy_term, discount,
                        qnet.bin_values.to(next_lp.dtype), num_bins, min_v, max_v)
                _v1, lp1, _v2, lp2 = qnet(observations, actions)
                # Cross-entropy of BOTH heads against the SAME target, summed
                # (`simbaV2_update.py:208`: `loss_1 + loss_2`).
                critic_loss = (-(target * lp1).sum(1).mean()
                               - (target * lp2).sum(1).mean())
            opt_critic.zero_grad(set_to_none=True)
            scaler.scale(critic_loss).backward()
            scaler.unscale_(opt_critic)
            _clip(qnet.parameters())
            scaler.step(opt_critic)
            scaler.update()
            proj["calls"] += 1                        # `simbaV2_update.py:239`
            proj["kernels"] += ns["project_"](qnet)

            # -- target critic, every step, no delay ------------------------
            with torch.no_grad():
                src = [p.data for p in qnet.parameters()]
                tgt = [p.data for p in qnet_target.parameters()]
                torch._foreach_mul_(tgt, 1.0 - tau)
                torch._foreach_add_(tgt, src, alpha=tau)
                # The target critic is a convex combination of two projected
                # networks, which is NOT itself on the sphere. Upstream
                # projects it at init (`simbaV2_agent.py:195`) and never again
                # -- `update_target_network` (`simbaV2_update.py:244 update_target_network`) is a
                # bare tree_map. Left exactly as upstream leaves it; noted
                # because "project everything, always" is the obvious wrong
                # improvement here.
                pass

        def normalize_obs(x: Any, update_stats: bool = True) -> Any:
            return _norm_obs(agent, x, update=update_stats)

        upd = update
        if compile_on:
            upd = torch.compile(update, mode=str(hyper["compile_mode"]))

        curve: List[Dict[str, float]] = []
        comp_curve: List[Dict[str, float]] = []
        spent = 0
        spent_eval = 0
        seed_error = ""
        pruned_at = None
        timed_out = False
        best_snap: Optional[Dict[str, Any]] = None
        best_snap_idx = -1
        n_random_steps = 0
        ceiling_decisions: List[Dict[str, Any]] = []
        pruner = _ceiling_guard(cfg, rule, prune_metric, ceiling, ceiling_decisions)

        eval_policy = _policy_fn(agent, view)
        obs_t = torch.as_tensor(view.reset(), device=device)
        global_it = 0
        # `_UtilSampler`: several point samples spread across the run, never
        # one, and never at the end -- `fasttd3`'s reasons verbatim (a single
        # sample can land inside a cold backend's compile window, which is the
        # one interval where 0 is both the true answer and the least
        # representative one). None off cuda, because "could not tell" is not
        # "was idle".
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
                learning = global_it >= start_iters
                with torch.no_grad(), autocast(device_type=amp_device_type,
                                               dtype=amp_dtype, enabled=amp_enabled):
                    # THE NORMALISER IS FED ON EVERY INTERACTION STEP,
                    # INCLUDING THE RANDOM ONES. Upstream's own comment:
                    # "While using random actions until buffer.can_sample(),
                    # we feed data into agent to compute statistics within a
                    # wrapper" (`run_online.py:99-107`). It updates HERE and
                    # nowhere else -- `ObservationNormalizer.update` is called
                    # from `sample_actions` only, so replayed batches are
                    # normalised WITHOUT updating (the `update_stats=False`
                    # below). `fasttd3` updates on replayed batches too,
                    # because FastTD3 does; the two backends therefore
                    # normalise differently on purpose.
                    norm_obs = normalize_obs(obs_t, update_stats=True)
                    if learning:
                        actions, _lp = actor.sample(norm_obs, temperature=1.0)
                    else:
                        # `train_env.action_space.sample()` until the buffer
                        # can sample (`run_online.py:106-107`): uniform on the
                        # agent's own [-1, 1] box, which `_VecEnvView`
                        # maps to the adapter's.
                        actions = (torch.rand((num_envs, n_act), device=device)
                                   * 2.0 - 1.0)
                        n_random_steps += num_envs * action_repeat
                next_obs, rewards, dones, time_outs, true_next = view.step(
                    actions.float())
                rewards_t = torch.as_tensor(rewards, device=device)
                dones_t = torch.as_tensor(dones, device=device)
                truncs_t = torch.as_tensor(time_outs, device=device)
                if rew_norm is not None:
                    # Statistics at COLLECTION, scaling at SAMPLE time, and the
                    # buffer holds RAW rewards -- `RewardNormalizer`'s own
                    # split upstream (`agents/wrappers/normalization.py:143-150`).
                    rew_norm.update_stats(rewards_t, dones_t.float())
                rb.extend(obs_t, actions.float(), rewards_t, dones_t, truncs_t,
                          torch.as_tensor(true_next, device=device))
                obs_t = torch.as_tensor(next_obs, device=device)
                if learning:
                    for _u in range(upd_per_step):
                        data = rb.sample(batch_per_env)
                        data["observations"] = normalize_obs(
                            data["observations"], update_stats=False)
                        data["next_observations"] = normalize_obs(
                            data["next_observations"], update_stats=False)
                        if rew_norm is not None:
                            data["rewards"] = rew_norm(data["rewards"])
                        upd(data)
                        # ONE SCHEDULER STEP PER GRADIENT STEP, inside the
                        # `learning` guard. `optax.linear_schedule` is indexed
                        # by the optimiser's own step count, so a schedule
                        # stepped per interaction step -- or stepped through
                        # the warm-up, as FastTD3's is -- would be a different
                        # schedule. Upstream cannot have that bug: the
                        # schedule is a function of the optimiser state, not a
                        # separate object someone has to step.
                        for sched in scheds:
                            sched.step()
                global_it += 1
            # SIMULATOR steps, upstream's accounting: one interaction step is
            # `num_envs * action_repeat` of them (`run_online.py:160`).
            spent = global_it * num_envs * action_repeat

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
                          # WALL-CLOCK: listed in `tests/test_parallelism._VOLATILE`.
                          "eval_wall_s": ev_wall})
            if ceiling is not None:
                curve[-1]["demo_fraction"] = _demo_fraction_point(
                    metrics["reward_return"], ceiling,
                    [p["reward_return"] for p in curve[:-1]])
            if _running_best(curve, select_rule):
                best_snap = _snapshot_agent(agent)
                best_snap_idx = len(curve) - 1
            if pruner([p[prune_field] for p in curve], spent / max(1, total_steps)):
                pruned_at = ck + 1
                break
            # `ck + 1 < n_chunks`: a seed that finished its whole budget must
            # not be marked failed for crossing the wall clock on the very
            # last chunk.
            if ck + 1 < n_chunks and time.monotonic() - seed_t0 > timeout_s:
                seed_error = f"timeout after {timeout_s:g}s"
                timed_out = True
                break

        ship_idx, select_reason = _training._select_checkpoint(
            curve, select_rule, select_min_delta)
        if ship_idx is not None:
            if best_snap is None or best_snap_idx != ship_idx:
                ship_idx, select_reason = None, "no_snapshot"
            elif not _restore_agent(agent, best_snap):
                ship_idx, select_reason = None, "restore_failed"

        wall = float(time.monotonic() - seed_t0)
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
            "seed": int(seed), "fitness": float(fits[ship_i]),
            "final": float(fits[ship_i]),
            "max": float(np.max(fits)), "auc": float(np.mean(fits)),
            "env_steps": int(spent + spent_eval), "train_steps": int(spent),
            "train_steps_requested": int(total_steps),
            "wallclock_s": wall,
            "n_checkpoints": len(curve), "pruned_at_round": pruned_at,
            "timed_out": bool(timed_out), "learner": "simba_v2",
            # `algorithm_cited`, NOT `algorithm`, and it is NOT the learner
            # -- `fasttd3_backend`'s field and its whole argument: the key
            # records the algorithm the method's PAPER used and stays a
            # citation. Here the two happen to agree (RDA cites `sac` and
            # this backend is SAC), which is exactly why the names must still
            # be kept apart: a row where a citation and an execution fact
            # coincide is the one that teaches a reader to read them as one.
            "algorithm_cited": cfg["train.algorithm"], "backend": "simba_v2",
            "architecture": arch,
            # THE FOUR BUDGET FACTS AS EXECUTED, so the 625K claim is audited
            # from the artifact and not from the config. `num_envs` is
            # `train.n_parallel_envs` -- the paper's training-env count on this
            # backend -- and `env_workers` is the schedule.
            "num_envs": int(num_envs), "action_repeat": int(action_repeat),
            "interaction_steps": int(global_it),
            "updates_per_interaction_step": int(upd_per_step),
            "update_steps_planned": int(update_steps),
            "update_steps": int(max(0, global_it - start_iters) * upd_per_step),
            "lr_decay_steps": int(decay_steps),
            "learning_starts_transitions": int(hyper["learning_starts"]),
            "learning_starts_steps": int(start_iters),
            "random_action_env_steps": int(n_random_steps),
            "device": str(device),
            # WHERE THE PARAMETERS ACTUALLY LIVE, read off the networks rather
            # than from the config or a probe tensor -- `_param_device`
            # returns the SET over actor and both critics so a single
            # parameter left on cpu is a visible result and not a pass.
            "learner_device": _param_device(agent),
            "data_root_source": _data_root_source(),
            # `torch.cuda.max_memory_allocated` -- TENSORS -- which reads
            # exactly 0 on cpu, so NON-ZERO is the whole test and no MiB floor
            # is wanted.
            "gpu_peak_alloc_torch_mib": gpu_peak_mib,
            **util.row_fields(),
            "env_workers": int(env_workers), "buffer_rows": int(buffer_rows),
            "buffer_size_transitions": int(hyper["buffer_size"]),
            "batch_size": int(hyper["batch_size"]),
            "batch_per_env": int(batch_per_env),
            "critic_use_cdq": bool(use_cdq),
            "gamma": float(gamma),
            "target_tau": float(tau),
            "temp_target_entropy": float(target_entropy),
            "critic_support": [float(min_v), float(max_v)],
            "normalize_reward": bool(hyper["normalize_reward"]),
            "normalize_observation": bool(hyper["normalize_observation"]),
            # THE HYPERSPHERICAL PROJECTION, IN FOUR FIELDS, because one was
            # not enough and the way it was not enough is instructive.
            #
            # `n_hyper_kernels_projected` is the INIT count (all three
            # networks, `simbaV2_agent.py:193-195`). It catches the refactor
            # case: the mechanism is a loop over marked submodules, so a rename
            # or a wrapper around `HyperDense` would silently project NOTHING
            # and 0 here is that bug.
            #
            # IT DOES NOT CATCH THE CASE THIS MODULE EXISTS FOR: it is written
            # once, by the constructor, so deleting the two per-step
            # `project_` calls would leave this row byte-identical. A run that
            # projected at init and never again would report what a correct
            # run reports -- and that run is FastTD3's port, whose projection
            # is absent. The field answers "did the constructor project?", not
            # "did the projection reach the network?".
            #
            # So `n_projection_calls` and `n_projection_kernels` are the
            # per-STEP counters, and they are worth having because they have a
            # PREDICTABLE value rather than merely a non-zero one:
            #
            #     n_projection_calls   == 2 * update_steps
            #     n_projection_kernels == update_steps * (k_actor + k_critic)
            #
            # which a test asserts as an EQUALITY (`tests/test_simba_v2.py`).
            # `n_hyper_kernels_actor` / `_critic` are on the row so that
            # equality is checkable from the artifact alone, without knowing
            # the network's shape.
            #
            # ONE CAVEAT, stated because a silent undercount would be worse
            # than no field: these are Python side effects inside `update`, and
            # `update` is wrapped in `torch.compile` when
            # `train.hyperparameters.compile` is on. `compiled` is on this same
            # row, so a reader can tell whether the counters were taken under a
            # compiled closure; the equality test runs with compile off, which
            # is this backend's default.
            "n_hyper_kernels_projected": int(agent["n_projected"]),
            "n_hyper_kernels_actor": int(agent["k_actor"]),
            "n_hyper_kernels_critic": int(agent["k_critic"]),
            "n_projection_calls": int(proj["calls"]),
            "n_projection_kernels": int(proj["kernels"]),
            "amp": bool(amp_enabled), "compiled": bool(compile_on),
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
            # As executed: this backend installs neither (see the warnings at
            # the top of the call), and these are how a seed row says so.
            "elite_constraint_optimisers": 0,
            "anchor": {}, "anchor_released": False,
            "secondary_transitions": 0, "secondary_ratio": 0.0,
            "action_clip_fraction": float(view.n_clipped / max(1, view.n_steps)),
            **({"view": "jax",
                "physics": getattr(env, "physics", "mjx"),
                "num_envs": int(view.n),
                "jit_compile_s": float(getattr(view, "jit_compile_s", 0.0)),
                "reward_sync_s": float(getattr(view, "reward_sync_s", 0.0)),
                "nonfinite_rows": int(getattr(view, "nonfinite_rows", 0)),
                "device_peak_mib": dict(view.device_peaks()),
                "jax_cache": dict(getattr(view, "jax_cache", {})),
                "dlpack_zero_copy": dict(getattr(view, "dlpack_zero_copy", {})),
                "jax_version": str(getattr(view, "jax_version", ""))}
               if isinstance(view, _JaxVecEnvView) else {}),
            "checkpoint_selection": select_rule,
            "checkpoint_selection_reason": _selection_reason(select_reason),
            "restored_checkpoint": (int(curve[ship_idx]["round"])
                                    if ship_idx is not None else None),
            "shipped_checkpoint": int(curve[ship_i]["round"]) if curve else None,
            "final_checkpoint_fitness": float(fits[-1]),
            "error": seed_error, "checkpoints": curve}
        if str(cfg.get("train.reward_scaling", "none") or "none") != "none":
            seed_metric["reward_scaling"] = _training._scaling_record(candidate, reward)
        if ceiling is not None:
            seed_metric["ceiling_decisions"] = ceiling_decisions
        view.close()
        return {"curve": curve, "comp_curve": comp_curve, "spent": spent,
                "spent_eval": spent_eval, "pruned_at": pruned_at,
                "timed_out": timed_out, "seed_error": seed_error,
                "wallclock_s": wall, "agent": agent, "view": view,
                "seed_metric": seed_metric}

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
    seed_workers, seed_reason = _fasttd3._seed_fork_workers(cfg, n_runs)
    _fasttd3._note_seed_schedule(ctx, candidate, seed_workers, seed_reason)
    final_blob = None

    try:
        if seed_workers <= 1:
            last_agent = None
            for seed_i in range(n_runs):
                out = run_seed(seed_i)
                last_agent = out.pop("agent")
                last_view = out.pop("view")
                policies.append(_policy_fn(last_agent, last_view))
                fold_seed(out)
        else:
            def seed_child(i: int) -> Dict[str, Any]:
                out = run_seed(i)
                agent = out.pop("agent")
                out.pop("view")
                out["blob"] = _blob(ns, arch, agent)
                return out

            for seed_i, out in enumerate(
                    _fasttd3._fork_seeds(n_runs, seed_child, seed_workers, "simba_v2")):
                if "fatal" in out:
                    raise RuntimeError(f"simba_v2 seed worker {seed_i}/{n_runs}: "
                                       f"{out['fatal']}")
                blob = out.pop("blob", None)
                # Rebuild the trained policy from the exact parameters the
                # child saved -- `_sb3_run`'s rule, same fatal-not-mismeasured
                # handling when the blob does not load: rollouts on fresh
                # weights would be a corrupted measurement wearing a healthy
                # one's clothes.
                agent2 = _build_agent(int(out["seed_metric"]["seed"]))
                # Action map only (`_policy_fn` reads `to_env_action`): this
                # view is never reset or stepped, so it gets no repeat and no
                # workers.
                view2 = _VecEnvView(env, 1, reward, _features, norm_mode,
                                    int(out["seed_metric"]["seed"]), n_obs)
                if blob is None or not _apply(
                        ns, arch, agent2, blob,
                        f"seed:{out['seed_metric']['seed']}"):
                    fatal_error = fatal_error or (
                        f"simba_v2 seed fork: seed {seed_i} sent no loadable policy "
                        "(serialisation failed in the child, or the blob did not "
                        "load into this architecture); its rollouts would run on "
                        "untrained weights, so the result is marked failed rather "
                        "than mis-measured")
                    blob = None
                policies.append(_policy_fn(agent2, view2))
                if seed_i == n_runs - 1:
                    final_blob = blob if blob is not None else None
                fold_seed(out)

        result.checkpoints = _fasttd3._mean_curve(per_seed_curves)
        _fasttd3._stamp_seed_rows(result.seed_metrics, seed_workers, seed_reason)
        _fasttd3._summarise_ceiling(candidate, result.seed_metrics)
        result.component_traces = _fasttd3._mean_components(per_seed_components)
        result.gt_reward_curve = [p["gt_return"] for p in result.checkpoints]

        # Single-threaded on BOTH schedules: every seed above TRAINED at one
        # thread, so the rollouts must predict at one thread too or the
        # sequential path's trajectories would be the one part of the run
        # whose numerics vary with `--cpus-per-task`.
        with _single_thread_torch():
            roll_rng = np.random.default_rng(base_seed + 104729)
            for i in range(max(1, int(cfg.get("evaluate.rollouts_per_candidate", 3) or 1))):
                traj, st, _gt = _fasttd3._rollout(
                    env, policies[i % len(policies)], roll_rng, reward,
                    _error_router("exception_soft", io.StringIO()))
                result.trajectories.append(traj)
                if i < _fasttd3._n_replayed_rollouts(cfg):
                    _fasttd3._retain_replay_states(env, traj)
                used += st
                # Collection steps must reach budget.json, not only
                # env_steps_used -- charged per rollout so one that raises
                # mid-loop still pays for what it spent.
                ctx.budget.record_rollout_steps(st)
            # The dedicated TPE-store pool (CARD's check pool): a separate rng
            # so the eval rollouts above stay byte-identical when collection
            # is toggled. Without it `update.memory.trajectory_store` never
            # grows on this backend, the TPE screen fails open forever and
            # `policy_trainings_skipped` stays 0 -- CARD silently becomes its
            # own ablation.
            n_store = _fasttd3._n_store_rollouts(cfg)
            if n_store > 0:
                store_rng = np.random.default_rng(base_seed + 224737)
                store_steps = 0
                for i in range(n_store):
                    traj, st, _gt = _fasttd3._rollout(
                        env, policies[i % len(policies)], store_rng, reward,
                        _error_router("exception_soft", io.StringIO()))
                    result.store_trajectories.append(traj)
                    used += st
                    store_steps += st
                ctx.budget.record_rollout_steps(store_steps)

        if seed_workers <= 1 and last_agent is not None:
            final_blob = _blob(ns, arch, last_agent)
        if final_blob is not None:
            result.policy_ref = _fasttd3._store(
                _POLICY_STORE, f"policy:{candidate.cand_id}", final_blob,
                _fasttd3._POLICY_STORE_MAX_BYTES)
            _fasttd3._remember_code(candidate)
        # NO REPLAY EXPORT, and the absence is stated rather than left to be
        # discovered: `train.init: secondary_replay_buffer` and
        # `train.interaction_cfg.shared_buffer` both want a slice in STATE
        # space, and wiring one here without the import half would give the
        # pool a producer and no consumer. `_check_coherence` refuses the
        # shared buffer beside this backend for the same reason it does
        # beside `fasttd3`.
    finally:
        env.set_dr(None)

    result.env_steps_used = int(used)
    result.wallclock_s = float(time.monotonic() - t0)
    if fatal_error:
        result.trained = False
        result.error = fatal_error
    return result
