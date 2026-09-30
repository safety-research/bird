#!/usr/bin/env python3
"""Train a policy on the ENVIRONMENT'S OWN reward and measure it as an expert anchor.

    uv run --no-sync python3 scripts/train_expert.py --env pendulum --steps 1000000

WHAT THIS IS FOR. `pendulum` and `acrobot` ship no scripted solution, so their
expert anchors are trained with this script: inventing a ceiling would make every
`human_normalised` number on them a ratio against a number nobody measured.
Meta-World has `ENV_POLICY_MAP`; classic control has nothing. So the ceiling has
to be *earned*: train on the ground-truth reward and measure what that reaches.

WHAT "GROUND TRUTH" MEANS HERE, precisely. `env.reference_reward(s_{t+1}, a_t)` --
the hand-written reward the adapter carries as the answer an LLM is trying to
match. NOT a candidate reward, and NOT `task_metric`. Training on `task_metric`
directly would be optimising the evaluation metric, and the resulting "expert"
would be a ceiling on nothing but its own objective.

WHY THE MEASUREMENT REUSES `measure_anchors.py`. The expert anchor is only
meaningful beside the random one, and the two are comparable only if scored the
same way: same `task_metric`, same both-reductions-from-one-rollout rule, same
interval estimator, same n. Importing rather than reimplementing is what keeps
that true when either side changes.

THE RESULT IS A FLOOR ON THE CEILING, and the artifact says so. A trained policy
is whatever the training run reached; a better one may exist. That is weaker than
Meta-World's scripted policy, which is the benchmark's own answer, and the two
must not be read as the same kind of number -- hence `method: trained_ppo` rather
than `scripted_policy`, and the recorded budget.

IT REFUSES A TASK WITH NO REFERENCE REWARD (`env.has_reference_reward` False, i.e.
the task's spec says `reward.human.kind: none`), exit 2, instead of training on
whatever `reference_reward` would have returned. On `gym_half_cheetah_backward`,
`gym_half_cheetah_target_speed`, `gym_hopper_hop_in_place`, `gym_swimmer_heading`
and `gym_reacher_hold` the BASE simulator's reward is forward velocity, which all
five specs disown in words: an "expert" trained on it is a max-speed policy on
target-speed, and on hop-in-place it comes in below random.
An anchor trained on a reward the task disowns is not a ceiling on the task; there is
no flag to override this, because the recipe has nothing honest to record. The
payload's `reward_trained_on` names the reward precisely (the spec's
`reward.human` pin, when the env has a spec) rather than the bare word
`reference_reward`, which is true of every run and says nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # `scripts/` is not a package

from bird.envs.spec import ANY_STEP, PER_STEP           # noqa: E402
from measure_anchors import interval                    # noqa: E402


def _refusal(env) -> Optional[str]:
    """Why this env cannot be trained on, or None. See the module docstring."""
    if getattr(env, "has_reference_reward", True):
        return None
    task = getattr(env, "task", None)
    where = f"tasks/{task}/shared_spec.yaml" if task else "its task spec"
    return (f"{env.name}: no reference reward to train on -- {where} declares "
            "`reward.human.kind: none`, so the simulator's built-in reward is not this "
            "task's objective and training an \"expert\" on it would produce a policy "
            "for a different task. Nothing to record honestly; refusing.")


def _reward_trained_on(env) -> Dict[str, Any]:
    """What `reference_reward` IS on this env, from the task spec when there is one."""
    out: Dict[str, Any] = {"method": "reference_reward", "env": env.name}
    try:
        from bird import tasks as _tasks
        spec = _tasks.by_env_id(env.name)
    except Exception:  # noqa: BLE001 -- a spec-less env is not an error here
        spec = None
    if spec is None:
        out["spec"] = None
        return out
    human = (spec.reward or {}).get("human") or {}
    ref = human.get("reference") or {}
    out.update({"spec": spec.id, "kind": human.get("kind"),
                "symbol": ref.get("symbol"), "path": ref.get("path"),
                "verified_against": human.get("verified_against")})
    return out


def _gym_view(env, seed: int):
    """A gymnasium view of the adapter whose reward is `reference_reward`.

    Deliberately the same shape as `training.py::_sb3_run`'s `_Gym`, including
    the per-episode seed stride: SB3's `DummyVecEnv.reset` passes a seed exactly
    once and every auto-reset after it arrives with `seed=None`, so a naive
    wrapper draws OS entropy and the run stops being reproducible from `--seed`.
    7907 is the same prime that file uses, so two seeds cannot alias.
    """
    import gymnasium as gym
    from gymnasium import spaces

    continuous = getattr(env, "exact_states", None) is None

    class _Gym(gym.Env):
        def __init__(self) -> None:
            self.observation_space = spaces.Box(
                np.asarray(env.obs_low, dtype=np.float32),
                np.asarray(env.obs_high, dtype=np.float32))
            self.action_space = (
                spaces.Box(np.asarray(env.action_low, dtype=np.float32),
                           np.asarray(env.action_high, dtype=np.float32))
                if continuous else spaces.Discrete(env.n_actions))
            self._s = None
            self._t = 0
            self._episode_seed = seed
            self._episode = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:
                self._episode_seed, self._episode = int(seed), 0
            self._s = env.reset(np.random.default_rng(
                self._episode_seed + self._episode * 7907))
            self._episode += 1
            self._t = 0
            return np.asarray(self._s, dtype=np.float32), {}

        def step(self, action):
            a = (np.asarray(action, dtype=float).ravel() if continuous
                 else env.action_set[int(action)])
            s2, done, info = env.step(self._s, a)
            r = float(env.reference_reward(s2, a))    # <- the ground-truth reward
            self._s = s2
            self._t += 1
            return (np.asarray(s2, dtype=np.float32), r,
                    bool(done), self._t >= env.horizon, info)

    return _Gym


def _rollout(env, model, n: int, seed: int) -> Dict[str, List[float]]:
    """Score the trained policy exactly as `measure_anchors._episode` does.

    Actions are DETERMINISTIC. An anchor is a statement about what the policy
    achieves, and sampling the stochastic policy would fold exploration noise
    into a ceiling.
    """
    continuous = getattr(env, "exact_states", None) is None
    out: Dict[str, List[float]] = {PER_STEP: [], ANY_STEP: []}
    for ep in range(n):
        rng = np.random.default_rng(seed + ep)
        state = env.reset(rng)
        states = [state]
        for _ in range(env.horizon):
            act, _ = model.predict(np.asarray(state, dtype=np.float32),
                                   deterministic=True)
            a = (np.asarray(act, dtype=float).ravel() if continuous
                 else env.action_set[int(act)])
            state, _done, _info = env.step(state, a)
            states.append(state)
        arr = np.asarray(states, dtype=float)
        out[PER_STEP].append(float(env.task_metric(arr)))
        out[ANY_STEP].append(float(env.success(arr)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", required=True)
    #: DEFAULTS ARE A MATCHED BUDGET, NOT A TRAINER'S: a small SAC budget
    #: comparable to one SAC candidate training under the `dev` profile
    #: (`train.backend: sb3`, `train.env_steps: 20000`).
    #: An anchor trained with a different learner or a bigger budget answers
    #: "what can this environment reach", which is a fine question and NOT the
    #: one a reward comparison asks -- there the gap to the ceiling would be part
    #: reward quality and part budget, with no way to tell which.
    ap.add_argument("--algo", default="sac", choices=("sac", "ppo", "td3"))
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--n-envs", type=int, default=1,
                    help="vectorised copies in ONE process; 1 for off-policy")
    ap.add_argument("--train-seeds", type=int, default=3,
                    help="independent policies trained; the anchor pools them, so "
                         "it carries training variance rather than one lucky run")
    ap.add_argument("--n-eval", type=int, default=30,
                    help="TOTAL episodes across seeds; matches the random anchor's n")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0,
                    help="evals without improvement before stopping (0 disables, "
                         "which is the matched protocol -- a candidate got the "
                         "full 20k)")
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--out", default=None, help="write JSON here")
    #: The anchor is a number; a POLICY is a reusable artifact. Without this the
    #: trained expert is scored and then dropped, so anyone who wants to roll it
    #: out, film it, or compare behaviour has to retrain from scratch.
    ap.add_argument("--save-policy", default=None, metavar="DIR",
                    help="write each seed's trained policy as an SB3 .zip here")
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(1)          # one task, one CPU thread
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import (
        EvalCallback, StopTrainingOnNoModelImprovement)
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    algo_cls = {"sac": SAC, "ppo": PPO, "td3": TD3}[args.algo]

    from bird import registry
    registry.load_all()
    if args.env not in registry.names("env"):
        print(f"unknown env {args.env!r}", file=sys.stderr)
        return 2
    env = registry.get("env", args.env)(None)
    why = _refusal(env)
    if why is not None:
        print(why, file=sys.stderr)
        return 2
    GymCls = _gym_view(env, args.seed)

    per_seed_eps = max(1, args.n_eval // max(1, args.train_seeds))
    scores: Dict[str, List[float]] = {PER_STEP: [], ANY_STEP: []}
    per_seed: List[Dict[str, Any]] = []
    t0 = time.monotonic()
    trained_steps = 0

    for k in range(args.train_seeds):
        seed = args.seed + k
        train_vec = DummyVecEnv([lambda: Monitor(GymCls())
                                 for _ in range(args.n_envs)])
        train_vec.seed(seed)
        cb = None
        if args.patience > 0:
            eval_vec = DummyVecEnv([lambda: Monitor(GymCls())])
            eval_vec.seed(seed + 10_000)
            cb = EvalCallback(
                eval_vec, n_eval_episodes=5, deterministic=True,
                eval_freq=max(1, args.eval_every // args.n_envs),
                callback_after_eval=StopTrainingOnNoModelImprovement(
                    max_no_improvement_evals=args.patience, min_evals=3,
                    verbose=1),
                verbose=1)
        model = algo_cls("MlpPolicy", train_vec, seed=seed, device="cpu", verbose=0)
        model.learn(total_timesteps=args.steps, callback=cb)
        trained_steps += int(model.num_timesteps)
        policy_path = None
        if args.save_policy:
            pdir = Path(args.save_policy)
            pdir.mkdir(parents=True, exist_ok=True)
            policy_path = str(pdir / f"{args.env}-{args.algo}-s{seed}.zip")
            model.save(policy_path)
            print(f"saved policy {policy_path}", file=sys.stderr)
        got = _rollout(env, model, per_seed_eps, seed * 1_000)
        for red in (PER_STEP, ANY_STEP):
            scores[red].extend(got[red])
        per_seed.append({"seed": seed, "steps": int(model.num_timesteps),
                         "policy": policy_path,
                         PER_STEP: round(float(np.mean(got[PER_STEP])), 4),
                         ANY_STEP: round(float(np.mean(got[ANY_STEP])), 4)})
        print(f"[seed {seed}] steps={model.num_timesteps} "
              f"per_step={np.mean(got[PER_STEP]):.4f} "
              f"any_step={np.mean(got[ANY_STEP]):.4f}", file=sys.stderr)

    elapsed = time.monotonic() - t0
    payload: Dict[str, Any] = {
        "env": args.env,
        "date": date.today().isoformat(),
        "method": f"trained_{args.algo}",
        "reward_trained_on": _reward_trained_on(env),
        "algorithm": args.algo,
        "backend": "sb3",
        "requested_steps_per_seed": args.steps,
        "trained_steps_total": trained_steps,
        "early_stopped": trained_steps < args.steps * args.train_seeds,
        "wall_clock_s": round(elapsed, 1),
        "seed": args.seed,
        "n_seeds": args.train_seeds,
        "n_eval_episodes": len(scores[PER_STEP]),
        "per_seed": per_seed,
        "by_reduction": {k: interval(v) for k, v in scores.items()},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
