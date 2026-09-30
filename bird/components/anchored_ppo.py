"""A behaviour-cloned prior and KL-anchored PPO fine-tuning.

`train.init: bc_prior` starts every candidate's policy from a clone of the task's
scripted policy (`policies/`), built once per run; `train.anchor.kind: kl_clone`
keeps the fine-tuned policy close to that clone with a KL penalty whose coefficient
never reaches zero. Both were measured before they were keys, and the two findings
that shaped the defaults are worth restating where the code is:

* A naive PPO fine-tune destroys the clone within a few hundred thousand steps on
  every MT10 task tried, and the obvious protection (a small initial log-std) makes
  it worse: a Gaussian policy's log-prob gradient on the mean scales as 1/sigma^2.
* A KL penalty to the frozen clone holds on every task tried (five MT10, two
  HumanoidBench) as long as its coefficient stays on; annealing it to zero is where
  the drift restarts. Adapting the coefficient toward a per-state KL budget was the
  best arm at 8M steps and improved on the clone on every original task.

CONFINEMENT. The scripted policy enters here, on the training side, and nowhere
else: nothing in this module builds a prompt, and `policies/README.md`'s rule that
no scripted policy reaches the reward-designing model holds because the generate
stage never imports it.

Everything torch- or sb3-shaped is imported lazily: `registry.load_all()` imports
this module on machines with neither.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import math

import numpy as np

log = logging.getLogger("bird")

__all__ = ["actor_parameters", "deterministic_action", "bc_fit", "kl_schedule",
           "make_kl_ppo_class", "freeze_actor", "find_registry_policy", "collect_demos",
           "build_prior", "gaussian_kl", "divergence", "AnchorPenalty", "attach_anchor", "reward_anchor"]


# ---------------------------------------------------------------------------
# the clone

def find_registry_policy(env_id: str, requested: str = "auto") -> str:
    """The registry id of the scripted policy for `env_id`; see `policies.policy_for_env`
    (the same function, which lives there so `config._check_coherence` can call it
    without importing anything torch-shaped)."""
    from ..policies import policy_for_env
    return policy_for_env(env_id, requested)


ACCEPT_RULES = ("success", "any", "metric_floor", "success_else_top", "success_else_best")
#: `success_else_best`'s defaults (`train.bc_prior.fallback_min_metric` / `.fallback_max_episodes`).
FALLBACK_MIN_METRIC = 0.0
FALLBACK_MAX_EPISODES = 40
#: `success_else_top`'s defaults (`train.bc_prior.fallback_oversample` / `.fallback_keep`).
FALLBACK_OVERSAMPLE = 4
FALLBACK_KEEP = 0.25


def collect_demos(env: Any, policy: Any, n_demos: int, success_margin: int,
                  rng: np.random.Generator, accept: str = "success",
                  min_metric: Optional[float] = None,
                  fallback_oversample: int = FALLBACK_OVERSAMPLE,
                  fallback_keep: float = FALLBACK_KEEP,
                  fallback_min_metric: float = FALLBACK_MIN_METRIC,
                  fallback_max_episodes: int = FALLBACK_MAX_EPISODES) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """(observation, action) pairs from `n_demos` episodes of the scripted policy on the
    adapter's own reset distribution. `accept` (`train.bc_prior.accept`) is the rule that
    decides which episodes are demonstrations at all:

    * `success` -- on an env with a success bar (finite `success_threshold`) each episode
      is truncated `success_margin` steps after its first success -- the scripted policies
      park in a `hold` phase after the task is done, and a clone trained on that learns to
      idle -- and episodes that never succeed are dropped: a failed demonstration teaches
      the failure. On an env with NO success bar (`success_threshold` inf, the gym MuJoCo
      adapters) neither applies: the whole episode is the demonstration, nothing is
      dropped, and the returned counts carry `success_gated: False` to say so.
    * `any` -- every episode is a demonstration, kept whole. For a teacher that under-
      performs the goal but is still the behaviour to start from (a scripted policy
      registered as `partial`/`negative`): a learner clones what it does and RL moves on.
    * `metric_floor` -- an episode is a demonstration when the adapter's own `task_metric`
      over its states is at least `min_metric` (`train.bc_prior.accept_min_metric`); a
      kept episode is still truncated after its first success when one occurs. The floor
      is in the metric's own units, so 0.0 keeps everything the way `any` does, and the
      task's `success_threshold` reproduces `env.success()` -- the records' success
      column -- which is not always the per-step success flag `success` keys on (an
      episode with one flagged step can be kept by `success` and dropped here).

    * `success_else_top` -- `success`, and when NOT ONE of the `n_demos` episodes
      succeeds (a teacher registered `negative`, with a success rate near zero),
      roll `fallback_oversample` x `n_demos` fresh episodes and keep the top `fallback_keep`
      fraction by the adapter's `task_metric` over the whole episode (whole episodes; one
      that does succeed is still cut at success + margin). The metric ranks the TEACHER's
      episodes only -- it never touches a candidate -- and the fallback is recorded in the
      counts (`fallback_used`, the pool size, what was kept, the metric floor it implied and
      every episode's metric) so a run says which demonstrations it cloned from. A clone
      of the teacher's best quarter beats a clone of its average when the average is a
      fall.

    * `success_else_best` -- `success`, and when nothing succeeds, KEEP SAMPLING: roll teacher
      episodes until `n_demos` of them have task_metric > `fallback_min_metric`, or until
      `fallback_max_episodes` x `n_demos` episodes have been rolled (the cap), and clone the
      best `n_demos` above the floor. Differs from `success_else_top` in what it promises: a
      fixed-size set of the teacher's better episodes rather than a fixed fraction of a fixed
      pool (which, on a teacher that scores on 1% of episodes, is 96% failures). The rule
      finds what the metric rewards, which need not be what the task intends -- if the only
      episodes above the floor are degenerate ones, those are what gets cloned; the counts
      say so (`fallback_pool`, `fallback_kept_metrics`). If fewer than
      `n_demos` episodes clear the floor by the cap, the ones that did are cloned (>= 1), else
      it raises like `success` does.

    Whatever the rule, an empty demonstration set raises rather than cloning nothing."""
    if accept not in ACCEPT_RULES:
        raise ValueError(f"bc_prior: accept must be one of {ACCEPT_RULES}, got {accept!r}")
    if accept == "metric_floor" and min_metric is None:
        raise ValueError("bc_prior: accept=metric_floor needs min_metric (train.bc_prior.accept_min_metric)")
    xs, ys = [], []
    used = skipped = 0
    horizon = int(getattr(env, "horizon", 500))
    # An env with no success bar (`success_threshold` inf: the gym MuJoCo adapters, whose
    # `success()` is False always because no random->expert span exists) can never
    # report a success, so gating on one would drop every demonstration and raise below
    # for a teacher that is doing the task perfectly well ("none of 25 demonstrations
    # succeeded" on every gym MuJoCo policy, whose metric is continuous by
    # construction). There the whole episode IS the demonstration: nothing is truncated
    # and nothing is dropped.
    bar = float(getattr(env, "success_threshold", float("inf")))
    fallback = accept in ("success_else_top", "success_else_best")
    fallback_rule = accept if fallback else ""
    if fallback:
        accept = "success"           # the first pass IS the success rule, verbatim
    success_gated = bool(np.isfinite(bar)) and accept == "success"
    # Under `success` and `metric_floor` a demonstration that does succeed is still cut at
    # success + margin (the hold-phase argument above); under `any` nothing is cut.
    truncate_at_success = accept in ("success", "metric_floor") and np.isfinite(bar)
    # The adapter's declared bounds, not a hard-coded [-1, 1]: the registry policies
    # emit actions in the adapter's normalised units and the adapter clips to its
    # own range (which need not be +/-1), so the clone must be fit to what the
    # simulator actually received.
    lo = np.asarray(env.action_low, dtype=float).ravel()
    hi = np.asarray(env.action_high, dtype=float).ravel()
    def _roll(seed: int):
        """One teacher episode: (pairs x, pairs y, first-success step, cut index, last state)."""
        s = np.asarray(env.reset(np.random.default_rng(seed)), dtype=float)
        policy.reset(np.random.default_rng(seed))
        ex, ey = [], []
        first = -1
        # Under `metric_floor` the episode runs to its end so the floor is judged on the
        # SAME number the registry records report -- task_metric over the whole episode --
        # and only the pairs kept for cloning are cut at success + margin afterwards.
        # Judging the prefix instead drops late successes (e.g. an episode that succeeds
        # at step 63 with full metric 0.79 scores 0.29 on the prefix, against a 0.3 bar).
        cut = None
        for t in range(horizon):
            a = np.clip(np.asarray(policy.act(s, t=t, env=env), dtype=float).ravel(), lo, hi)
            ex.append(s.astype(np.float32)); ey.append(a.astype(np.float32))
            s2, done, info = env.step(s, a)
            s = np.asarray(s2, dtype=float)
            if truncate_at_success and first < 0 and float(info.get("success", 0.0)) > 0.5:
                first = t
            if first >= 0 and cut is None and t >= first + int(success_margin):
                cut = len(ex)
                if accept == "success":
                    break
            if done:
                break
        return ex, ey, first, cut, s

    for _ in range(int(n_demos)):
        seed = int(rng.integers(2**31 - 1))
        ex, ey, first, cut, s = _roll(seed)
        if accept == "success" and success_gated and first < 0:
            skipped += 1
            continue
        if accept == "metric_floor":
            # the adapter's own metric over every state this episode visited (the final
            # state included); a metric that is not a number (a diverged episode) is not
            # over any floor
            m = float(env.task_metric(np.asarray(ex + [s.astype(np.float32)], dtype=np.float32)))
            if not (m >= float(min_metric)):
                skipped += 1
                continue
            if cut is not None:
                ex, ey = ex[:cut], ey[:cut]
        used += 1
        xs.extend(ex); ys.extend(ey)
    # The fallback keys appear ONLY when a fallback rule was asked for: the other three
    # rules' counts (and their bc_prior.json) stay byte-identical.
    fb: Dict[str, Any] = {"fallback_used": False, "fallback_rule": fallback_rule} if fallback else {}
    if fallback_rule == "success_else_best" and not xs:
        # KEEP SAMPLING, CAPPED: stop as soon as `n_demos` episodes clear the floor.
        cap = max(1, int(fallback_max_episodes) * int(n_demos))
        floor = float(fallback_min_metric)
        pool = []
        rolled = 0
        while rolled < cap and sum(1 for r in pool if r[0] > floor) < int(n_demos):
            seed = int(rng.integers(2**31 - 1))
            ex, ey, first, cut, s = _roll(seed)
            m = float(env.task_metric(np.asarray(ex + [s.astype(np.float32)], dtype=np.float32)))
            pool.append((m if np.isfinite(m) else -np.inf, seed, ex, ey, first, cut))
            rolled += 1
        above = sorted((r for r in pool if r[0] > floor), key=lambda r: -r[0])
        kept = above[:int(n_demos)]
        for m, seed, ex, ey, first, cut in kept:
            if cut is not None:
                ex, ey = ex[:cut], ey[:cut]
            used += 1
            xs.extend(ex); ys.extend(ey)
        skipped += rolled - len(kept)
        fb = {"fallback_used": True, "fallback_rule": "success_else_best",
              "fallback_min_metric": floor, "fallback_max_episodes": int(fallback_max_episodes),
              "fallback_pool": rolled, "fallback_cap": cap, "fallback_above_floor": len(above),
              "fallback_kept": len(kept), "fallback_hit_cap": rolled >= cap and len(above) < int(n_demos),
              "fallback_metric_floor": float(kept[-1][0]) if kept else None,
              "fallback_kept_metrics": [round(float(r[0]), 4) for r in kept],
              "fallback_pool_metrics": [round(float(r[0]), 4) for r in pool],
              "fallback_kept_seeds": [int(r[1]) for r in kept],
              "fallback_kept_lengths": [len(r[2]) if r[5] is None else r[5] for r in kept]}
    elif fallback and not xs:
        # THE FALLBACK. Nothing succeeded, so `success` has no demonstration; instead of
        # raising, rank a larger pool of whole episodes by the adapter's own metric and
        # clone the best quarter. Fresh seeds, drawn from the same rng after the first
        # pass, so a run's demonstrations are reproducible from its seed alone.
        pool_n = max(1, int(round(float(fallback_oversample) * int(n_demos))))
        keep_n = max(1, int(math.ceil(float(fallback_keep) * pool_n)))
        pool = []
        for _ in range(pool_n):
            seed = int(rng.integers(2**31 - 1))
            ex, ey, first, cut, s = _roll(seed)
            m = float(env.task_metric(np.asarray(ex + [s.astype(np.float32)], dtype=np.float32)))
            pool.append((m if np.isfinite(m) else -np.inf, seed, ex, ey, first, cut))
        pool.sort(key=lambda r: -r[0])            # best metric first; ties keep roll order
        kept = pool[:keep_n]
        for m, seed, ex, ey, first, cut in kept:
            if cut is not None:                     # a success in the pool is still cut
                ex, ey = ex[:cut], ey[:cut]
            used += 1
            xs.extend(ex); ys.extend(ey)
        skipped += pool_n - len(kept)
        fb = {"fallback_used": True, "fallback_rule": "success_else_top",
              "fallback_oversample": int(fallback_oversample),
              "fallback_keep": float(fallback_keep), "fallback_pool": pool_n,
              "fallback_kept": len(kept),
              "fallback_metric_floor": float(kept[-1][0]) if kept else None,
              "fallback_kept_metrics": [round(float(r[0]), 4) for r in kept],
              "fallback_pool_metrics": [round(float(r[0]), 4) for r in pool],
              "fallback_kept_seeds": [int(r[1]) for r in kept],
              "fallback_kept_lengths": [len(r[2]) if r[5] is None else r[5] for r in kept]}
    if not xs:
        why = {"success": "succeeded", "any": "ran", "metric_floor": f"reached task_metric >= {min_metric}"}[accept]
        raise RuntimeError(f"bc_prior: none of {n_demos} demonstrations {why}; nothing to clone")
    return (np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
            {"demos_used": used, "demos_skipped": skipped, "pairs": len(xs),
             # whether demonstrations were truncated at success and failures dropped, or
             # kept whole because the env has no success bar to gate on (or the rule is not
             # `success`)
             "success_gated": success_gated,
             "accept": (fallback_rule if fallback else accept),
             "accept_min_metric": (None if min_metric is None else float(min_metric)),
             **fb})


def actor_parameters(policy: Any) -> List[Any]:
    """The parameters that define the policy's ACTION output, whatever SB3 family it is.

    Off-policy policies (SAC, TD3, DDPG) carry an `actor` module and the answer is its
    parameters. An on-policy ActorCriticPolicy (PPO, A2C) has no such module: the
    answer is the pi half of the extractor (and any shared trunk), the mean head, and
    the log-std. Nothing else here assumes an algorithm, so the prior and the
    warm-up freeze work on either family.
    """
    actor = getattr(policy, "actor", None)
    if actor is not None:
        return list(actor.parameters())
    params: List[Any] = []
    ext = policy.mlp_extractor
    for name in ("shared_net", "policy_net"):
        net = getattr(ext, name, None)
        if net is not None:
            params.extend(p for p in net.parameters())
    params.extend(policy.action_net.parameters())
    if getattr(policy, "log_std", None) is not None:
        params.append(policy.log_std)
    return params


def deterministic_action(policy: Any, obs_t: Any) -> Any:
    """The policy's deterministic action for a batch of observations, DIFFERENTIABLE,
    in the policy's own output space: the unsquashed mean for an ActorCriticPolicy
    (PPO clips at the env), the tanh-squashed actor output for SAC/TD3/DDPG."""
    actor = getattr(policy, "actor", None)
    if actor is not None:
        try:
            return actor(obs_t, deterministic=True)   # SAC's actor takes the flag
        except TypeError:
            return actor(obs_t)                       # TD3/DDPG's is deterministic already
    return policy.get_distribution(obs_t).distribution.mean


def _squashed(policy: Any) -> bool:
    """Off-policy actors emit tanh-squashed actions in [-1, 1] and SB3 rescales them to
    the env's bounds on the way out; their BC targets must be scaled the same way."""
    return getattr(policy, "actor", None) is not None


def freeze_actor(policy: Any, frozen: bool) -> None:
    """Value warm-up: with the actor frozen, PPO's first updates fit only the critic,
    so the first policy gradient is taken against a value function that has seen the
    clone's returns rather than against noise."""
    for p in actor_parameters(policy):
        p.requires_grad_(not frozen)


def _device_of(policy: Any) -> Any:
    """Where the policy's parameters live. Every tensor this module hands a policy is
    built here rather than on torch's default device: `train.hyperparameters.device`
    reaches SB3 through `_learner_kwargs`, so a `cuda` learner is one config key away
    and a CPU-built demonstration batch would meet GPU parameters mid-fit."""
    dev = getattr(policy, "device", None)
    if dev is None:
        dev = next(policy.parameters()).device
    return dev


def bc_fit(model: Any, X: np.ndarray, Y: np.ndarray, *, epochs: int = 400, lr: float = 1e-3,
           batch: int = 128, seed: int = 0) -> Dict[str, Any]:
    """Regress the actor's MEAN onto the demonstrations (MSE), leaving the critic and
    the log-std untouched. Returns the loss trace and the per-dimension residual std."""
    import torch as th
    th.manual_seed(seed)
    policy = model.policy
    log_std_param = getattr(policy, "log_std", None)
    params = [p for p in actor_parameters(policy) if p is not log_std_param]
    opt = th.optim.Adam(params, lr=lr)
    dev = _device_of(policy)
    Xt = th.as_tensor(X, dtype=th.float32, device=dev)
    # squashed actors are regressed in their own [-1, 1] space; SB3 unscales on the way out
    Yt = th.as_tensor(policy.scale_action(Y) if _squashed(policy) else Y, dtype=th.float32,
                      device=dev)
    n = len(Xt)
    gen = th.Generator().manual_seed(seed)
    losses: List[float] = []
    policy.set_training_mode(True)
    for _ in range(int(epochs)):
        perm = th.randperm(n, generator=gen)
        tot = 0.0
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            mean = deterministic_action(policy, Xt[idx])
            loss = th.mean((mean - Yt[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        losses.append(tot / n)
    policy.set_training_mode(False)
    with th.no_grad():
        mean = deterministic_action(policy, Xt)
        resid = (Yt - mean).cpu().numpy()
    std = np.clip(resid.std(axis=0), 0.05, 1.0)
    return {"losses": losses, "final_mse": float(np.mean(resid ** 2)),
            "resid_std": std.tolist(),
            # a fitted log-std is only meaningful where the policy HAS a global one
            "log_std_fit": np.log(std).tolist() if log_std_param is not None else None}


def build_prior(env: Any, model: Any, *, policy_id: str, n_demos: int, epochs: int,
                success_margin: int, seed: int, accept: str = "success",
                min_metric: Optional[float] = None,
                fallback_oversample: int = FALLBACK_OVERSAMPLE,
                fallback_keep: float = FALLBACK_KEEP,
                fallback_min_metric: float = FALLBACK_MIN_METRIC,
                fallback_max_episodes: int = FALLBACK_MAX_EPISODES) -> Dict[str, Any]:
    """Fit `model`'s actor to the scripted policy `policy_id` on `env` and return the
    record (`demos`, `final_mse`, the fitted log-std). The caller owns the model and
    serialises it; this function only teaches it."""
    from ..policy_api import load_policy
    policy = load_policy(policy_id)
    X, Y, counts = collect_demos(env, policy, n_demos, success_margin, np.random.default_rng(seed),
                                 accept=accept, min_metric=min_metric,
                                 fallback_oversample=fallback_oversample, fallback_keep=fallback_keep,
                                 fallback_min_metric=fallback_min_metric,
                                 fallback_max_episodes=fallback_max_episodes)
    fit = bc_fit(model, X, Y, epochs=epochs, seed=seed)
    return {"policy_id": policy_id, "demos": counts, "final_mse": fit["final_mse"],
            "log_std_fit": fit["log_std_fit"], "epochs": int(epochs), "seed": int(seed)}


# ---------------------------------------------------------------------------
# the anchor

def kl_schedule(beta0: float, frac: float, const: bool = False) -> Callable[[float], float]:
    """The keep-close weight over training. `const`: `beta0` throughout -- the RLHF
    convention, a fixed coefficient on KL to the reference policy (Ziegler et al. 2019;
    an adaptive controller toward a KL target is the usual refinement). Otherwise a
    linear anneal from `beta0` to 0 over the first `frac` of training, so the policy is
    eventually free of the clone -- measured to be where the drift restarts."""
    def fn(progress_remaining: float) -> float:
        if const:
            return float(beta0)
        done = 1.0 - float(progress_remaining)
        if frac <= 0.0 or done >= frac:
            return 0.0
        return float(beta0) * (1.0 - done / frac)
    return fn


_KLPPO: Optional[type] = None


def make_kl_ppo_class() -> type:
    """PPO with an optional KL-to-clone penalty and an optional demonstration term.
    Built inside a function, and cached, so this module imports without
    stable-baselines3."""
    global _KLPPO
    if _KLPPO is not None:
        return _KLPPO
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3 import PPO
    from stable_baselines3.common.distributions import kl_divergence
    from stable_baselines3.common.utils import explained_variance

    class KLPPO(PPO):
        """`train()` is SB3 2.x's, plus `beta(progress) * KL(clone || current)` on
        every minibatch when a frozen clone policy is attached (the keep-close
        anchor), and optionally a demonstration MSE term. With nothing attached it
        is PPO, and `model.save()` still works: clear the attributes first."""

        clone_policy: Any = None
        kl_beta_fn: Any = None
        #: sum the per-dimension KL (a per-sample KL) instead of averaging it; the
        #: mean under-weights the penalty by the action dimension (4 on MT10).
        kl_sum_dims: bool = False
        #: adaptive coefficient (RLHF's controller, Ziegler et al. 2019): after every
        #: update the coefficient moves by up to `kl_adapt_rate` toward keeping the
        #: measured clone-KL at `kl_target`. `kl_beta_cur` is the live value.
        kl_target: Optional[float] = None
        kl_adapt_rate: float = 0.1
        kl_beta_cur: float = 0.0
        #: floor under the adaptive coefficient (`train.anchor.beta_min`)
        kl_beta_min: float = 1e-3
        #: the last `train()`'s reading, in the shape `AnchorPenalty.adapt` returns,
        #: so `_sb3_run`'s seed row records both anchor forms the same way
        kl_last: Optional[Dict[str, float]] = None
        #: DAPG-style demonstration term (Rajeswaran et al. 2018): an MSE between the
        #: actor's mean and the demonstrated action on a demo minibatch, weighted by
        #: `bc_coef_fn(progress)`, added to every PPO minibatch loss.
        bc_data: Any = None
        bc_coef_fn: Any = None

        def detach_anchor(self) -> None:
            """Drop every unpicklable attachment before `save()`."""
            self.clone_policy = None
            self.kl_beta_fn = None
            self.bc_data = None
            self.bc_coef_fn = None

        def train(self) -> None:  # noqa: C901 - mirrors upstream
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)
            clip_range = self.clip_range(self._current_progress_remaining)
            clip_range_vf = None
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
            beta = 0.0
            if self.clone_policy is not None and self.kl_beta_fn is not None:
                beta = float(self.kl_beta_fn(self._current_progress_remaining))
            if self.kl_target is not None and self.clone_policy is not None:
                if self.kl_beta_cur <= 0.0:
                    self.kl_beta_cur = beta if beta > 0.0 else 1.0
                beta = self.kl_beta_cur
            bc_coef = 0.0
            if self.bc_data is not None and self.bc_coef_fn is not None:
                bc_coef = float(self.bc_coef_fn(self._current_progress_remaining))
            pg_losses, value_losses, entropy_losses, clone_kls, approx_kls, bc_losses = [], [], [], [], [], []
            continue_training = True
            for _epoch in range(self.n_epochs):
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions)
                    values = values.flatten()
                    advantages = rollout_data.advantages
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss = -th.min(
                        advantages * ratio,
                        advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)).mean()
                    if clip_range_vf is None:
                        values_pred = values
                    else:
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                    if beta > 0.0:
                        with th.no_grad():
                            ref_dist = self.clone_policy.get_distribution(rollout_data.observations)
                        cur_dist = self.policy.get_distribution(rollout_data.observations)
                        kl = kl_divergence(ref_dist, cur_dist)
                        kl = kl.sum(-1).mean() if self.kl_sum_dims else kl.mean()
                        loss = loss + beta * kl
                        clone_kls.append(kl.item())
                    if bc_coef > 0.0:
                        Xd, Yd = self.bc_data
                        idx = th.randint(0, len(Xd), (min(self.batch_size, len(Xd)),))
                        mean = self.policy.get_distribution(Xd[idx]).distribution.mean
                        bc_loss = F.mse_loss(mean, Yd[idx])
                        loss = loss + bc_coef * bc_loss
                        bc_losses.append(bc_loss.item())
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl = float(th.mean((th.exp(log_ratio) - 1) - log_ratio))
                    approx_kls.append(approx_kl)
                    if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                        continue_training = False
                        break
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()
                    pg_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())
                self._n_updates += 1
                if not continue_training:
                    break
            if self.kl_target is not None and clone_kls:
                self.kl_beta_cur = adapt_beta(self.kl_beta_cur, float(np.mean(clone_kls)), self.kl_target,
                                              self.kl_adapt_rate, self.kl_beta_min)
            if self.clone_policy is not None:
                self.kl_last = {"mean_divergence": float(np.mean(clone_kls)) if clone_kls else 0.0,
                                "beta": float(self.kl_beta_cur if self.kl_target is not None else beta)}
            ev = explained_variance(self.rollout_buffer.values.flatten(),
                                    self.rollout_buffer.returns.flatten())
            self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)) if pg_losses else 0.0)
            self.logger.record("train/value_loss", float(np.mean(value_losses)) if value_losses else 0.0)
            self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)) if entropy_losses else 0.0)
            self.logger.record("train/approx_kl", float(np.mean(approx_kls)) if approx_kls else 0.0)
            self.logger.record("train/clone_kl", float(np.mean(clone_kls)) if clone_kls else 0.0)
            self.logger.record("train/clone_beta", beta)
            self.logger.record("train/bc_loss", float(np.mean(bc_losses)) if bc_losses else 0.0)
            self.logger.record("train/bc_coef", bc_coef)
            self.logger.record("train/explained_variance", float(ev))
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    _KLPPO = KLPPO
    return KLPPO


def adapt_beta(beta: float, mean_divergence: float, target: float, adapt_rate: float,
               beta_min: float) -> float:
    """ONE controller step, shared by both anchor forms (RLHF's proportional rule, Ziegler
    et al. 2019): move `beta` by at most `adapt_rate` x 20% toward keeping the measured
    divergence at `target`, and never below `beta_min` (`train.anchor.beta_min`)."""
    err = float(np.clip((float(mean_divergence) - float(target)) / float(target), -0.2, 0.2))
    return float(max(float(beta_min), float(beta) * (1.0 + float(adapt_rate) * err)))


def gaussian_kl(mu_c: Any, log_std_c: Any, mu: Any, log_std: Any) -> Any:
    """KL(N(mu_c, std_c) || N(mu, std)) for diagonal Gaussians, summed over dimensions.
    Torch tensors or numpy arrays alike. Exact for SAC's squashed Gaussians too: tanh is
    a bijection and KL is invariant under one."""
    d = log_std - log_std_c
    e = ((2.0 * log_std_c).exp() if hasattr(log_std_c, "exp") else np.exp(2.0 * log_std_c))
    v = ((2.0 * log_std).exp() if hasattr(log_std, "exp") else np.exp(2.0 * log_std))
    kl = d + (e + (mu_c - mu) ** 2) / (2.0 * v) - 0.5
    return kl.sum(-1)


def divergence(clone: Any, policy: Any, obs_t: Any) -> Any:
    """Per-observation distance from the clone to the current policy, no gradient:
    the Gaussian KL for ActorCriticPolicy (PPO) and for SAC (pre-tanh parameters), and
    the squared action distance for a deterministic actor (TD3/DDPG), which has no
    distribution to take a KL of. Shape (batch,)."""
    import torch as th
    with th.no_grad():
        actor = getattr(policy, "actor", None)
        if actor is None:
            from stable_baselines3.common.distributions import kl_divergence
            kl = kl_divergence(clone.get_distribution(obs_t), policy.get_distribution(obs_t))
            return kl.sum(-1) if kl.dim() > 1 else kl
        if hasattr(actor, "get_action_dist_params"):
            mu_c, ls_c, _ = clone.actor.get_action_dist_params(obs_t)
            mu, ls, _ = actor.get_action_dist_params(obs_t)
            return gaussian_kl(mu_c, ls_c, mu, ls)
        return ((clone.actor(obs_t) - actor(obs_t)) ** 2).sum(-1)


class AnchorPenalty:
    """`train.anchor.kind: kl_reward` -- the keep-close term applied to the REWARD, the
    RLHF form (r' = r - beta * D(clone || current)(s)), which no algorithm's update has
    to know about: SAC, TD3 and PPO all just see a shaped reward. `beta` is adapted per
    chunk toward `target` when `schedule` is adaptive, from the divergence the env
    accumulated over that chunk."""

    def __init__(self, policy: Any, clone: Any, *, beta: float, schedule: str, target: float,
                 adapt_rate: float = 0.1, beta_min: float = 1e-3) -> None:
        self.policy, self.clone = policy, clone
        self.beta = float(beta)
        self.beta_min = float(beta_min)
        self.schedule, self.target, self.adapt_rate = schedule, float(target), float(adapt_rate)
        self._sum = 0.0
        self._n = 0

    def penalty(self, obs: np.ndarray) -> float:
        import torch as th
        obs_t = th.as_tensor(np.asarray(obs, dtype=np.float32),
                             device=_device_of(self.policy)).unsqueeze(0)
        d = float(divergence(self.clone, self.policy, obs_t).reshape(-1)[0])
        self._sum += d
        self._n += 1
        return self.beta * d

    def adapt(self) -> Dict[str, float]:
        """End of a chunk: move `beta` toward keeping the mean divergence at `target`."""
        mean = self._sum / self._n if self._n else 0.0
        if self.schedule == "adaptive" and self._n:
            self.beta = adapt_beta(self.beta, mean, self.target, self.adapt_rate, self.beta_min)
        self._sum, self._n = 0.0, 0
        return {"mean_divergence": mean, "beta": self.beta}


def reward_anchor(model: Any, *, beta: float, schedule: str, target: float,
                  log_std_init: Optional[float], beta_min: float = 1e-3) -> "AnchorPenalty":
    """`train.anchor.kind: kl_reward` -- the algorithm-agnostic form. Freezes a copy of
    `model.policy` and returns the penalty object `_Gym.step` subtracts from the
    reward; nothing in the algorithm's update changes."""
    import copy
    import torch as th
    if log_std_init is not None:
        if getattr(model.policy, "log_std", None) is not None:
            with th.no_grad():
                model.policy.log_std.fill_(float(log_std_init))
        else:
            log.info("train.anchor.log_std_init applies to a policy with a global log-std "
                     "(PPO); this one has none, so it is ignored")
    clone = copy.deepcopy(model.policy).eval()
    for p in clone.parameters():
        p.requires_grad_(False)
    return AnchorPenalty(model.policy, clone, beta=beta, schedule=schedule, target=target, beta_min=beta_min)


def attach_anchor(model: Any, *, beta: float, schedule: str, target: float,
                  log_std_init: Optional[float], beta_min: float = 1e-3) -> Dict[str, Any]:
    """`train.anchor.kind: kl_clone` -- freeze a copy of `model.policy` as the clone and
    install the keep-close term IN THE LOSS (the form the experiment measured; PPO
    only, since it lives in `KLPPO.train`). Returns what was installed."""
    import copy
    import torch as th
    if log_std_init is not None:
        if getattr(model.policy, "log_std", None) is not None:
            with th.no_grad():
                model.policy.log_std.fill_(float(log_std_init))
        else:
            log.info("train.anchor.log_std_init applies to a policy with a global log-std "
                     "(PPO); this one has none, so it is ignored")
    clone = copy.deepcopy(model.policy).eval()
    for p in clone.parameters():
        p.requires_grad_(False)
    model.clone_policy = clone
    model.kl_sum_dims = True
    model.kl_beta_fn = kl_schedule(float(beta), 1.0, const=True)
    model.kl_target = float(target) if schedule == "adaptive" else None
    model.kl_beta_cur = 0.0
    model.kl_beta_min = float(beta_min)
    return {"kind": "kl_clone", "schedule": schedule, "beta": float(beta), "beta_min": float(beta_min),
            "target": float(target) if schedule == "adaptive" else None,
            "log_std_init": log_std_init}
