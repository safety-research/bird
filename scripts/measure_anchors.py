#!/usr/bin/env python3
"""Measure `anchors` for the Meta-World tasks, under BOTH reductions.

    uv run python3 scripts/measure_anchors.py --n 30
    uv run python3 scripts/measure_anchors.py --tasks window_open --n 5 --no-write

Why both reductions in one pass: `task_metric` is the fraction of an episode's arriving
states inside the goal region, and `success()` is the same check reduced to "on at
least one step" (`success_threshold = 1/(2*horizon)` makes the second a threshold on the
first). They are different numbers -- 0.15 vs 0.25 for a uniform-random policy on drawer-close (n=100) --
and a spec that records one without saying which cannot normalise anything safely. Taking
them from the same episodes is also the only way the pair is internally consistent.

Cost: 10 tasks x 2 policies x n episodes x 500 steps. At the measured ~2,900 steps/s
through the adapter that is a few minutes on one machine for n=30.

`--n` defaults to 30: "mean task_metric over 2 episodes" is not a measurement, and the
schema's required `n_episodes` makes the weakness of a small n visible rather than
arguable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird import tasks                                    # noqa: E402
from bird.envs.spec import ANY_STEP, PER_STEP             # noqa: E402


def _episode(env, policy, rng) -> Dict[str, float]:
    """One episode, scored under both reductions from the same rollout."""
    state = env.reset(rng)
    states = [state]
    for _ in range(env.horizon):
        if policy is None:
            # EXACTLY a random-policy metric pass, discriminator
            # included. This repo already has a definition of "a random policy" and the
            # anchors must not carry a second one -- a floor measured under one definition
            # and a number measured under another cannot be read together.
            #
            # `exact_states is None` is the right test and `action_high > action_low` is
            # NOT: `EnvAdapter.__init__` derives the bounds from the action set's per-axis
            # min and max, so a discrete env whose four actions are encoded 0..3 reports a
            # box, and a uniform draw hands it 2.37. The tabular envs (`pendulum_discrete`,
            # `toy_gridworld`, `toy_hungry_thirsty`) would all be fed fractional actions.
            action = (rng.uniform(env.action_low, env.action_high)
                      if getattr(env, "exact_states", None) is None
                      else env.action_set[rng.integers(env.n_actions)])
        else:
            # A COPY: Meta-World's scripted policies mutate the observation in place
            # (`sawyer_door_open_v3_policy.py:38` does `pos_door[0] -= 0.05`), which
            # would corrupt the success read on the following step.
            action = policy.get_action(np.array(state, dtype=float, copy=True))
        state, _done, _info = env.step(state, action)
        states.append(state)
    arr = np.asarray(states, dtype=float)
    return {PER_STEP: float(env.task_metric(arr)), ANY_STEP: float(env.success(arr))}


def interval(values: List[float]) -> Dict[str, Any]:
    """A 95% interval for an anchor, and the reason this exists at all.

    An anchor is an estimate, and an anchor recorded without its uncertainty leaves a
    reader unable to tell a solid 0.0 from an under-powered 0.08. Not hypothetical: a
    `drawer_close` random anchor of 0.0833 measured over 12 episodes is literally 1/12,
    and at that n the interval is wider than the estimate.

    **The episode is the independent unit, not the step.** A per-step fraction is a mean
    over 500 highly correlated steps, so n is the episode count either way. Treating
    500 x n as the sample size understates the interval by more than an order of
    magnitude, and it is easy to get wrong precisely because the per-step reduction LOOKS
    better-powered.

    **Wilson when the sample lies entirely in {0, 1}, otherwise a t/normal SEM** -- and
    this is a statement about THIS SAMPLE, not an inference about the metric's support. A
    sample lying entirely in {0, 1} is Bernoulli *as a sample* whatever the metric could
    in principle return, so Wilson is correct for it rather than an approximation to
    something better.

    That branch is not a corner. Measured at n=30 over the six simulator-free envs, THREE
    route a continuous metric through it: `acrobot` and `toy_reacher` return 0.0 for every
    episode and `toy_gridworld` returns only {0.0, 1.0}. It is the default shape for a
    random anchor on any task with a hard floor -- which is most of them. On mixed {0,1}
    samples Wilson and SEM agree to the third decimal, so the choice only moves a number
    where one of them is degenerate.

    AND THAT IS THE GENERAL STATEMENT, which is worth having in preference to either
    special case: **a degenerate sample breaks the normal interval regardless of what the
    metric is.**
    A continuous metric returning 0.0 every episode gives `np.std(ddof=1) == 0`, so SEM
    reports `[0.0, 0.0]`. A proportion at 0/12 gives the normal approximation `+/- 0.0000`.
    Same pathology, opposite starting points, and in both the ZERO WIDTH lands on the
    measurement least entitled to certainty. It is a property of the estimator meeting a
    degenerate sample, not of proportions or of continuous metrics -- so it covers both
    without needing to know which you are holding, which is why Wilson is the default here
    rather than a special case for one metric shape.

    The genuinely wrong case is narrower than "a continuous metric": a metric with rare
    INTERMEDIATE mass that n happens to miss, leaving the binomial model no room for
    values that exist but were not drawn. Not constructible on the envs measured here -- the three
    envs above sit at a floor with nothing intermediate to miss, and both envs that do
    have intermediate values already route to SEM. Conceivable, not present.

    `estimator` is returned because a `half_width` is not comparable across anchors
    without it -- one may be Wilson and the next SEM, and an artifact that does not say
    which invites exactly the cross-anchor comparison it cannot support. Same discipline
    as `failure_kind` elsewhere in this repo.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0, "half_width": 0.0,
                "estimator": "none"}
    mean = float(np.mean(values))
    z = 1.959963984540054
    binary = all(v in (0.0, 1.0) for v in values)
    if binary:
        denom = 1.0 + z * z / n
        centre = (mean + z * z / (2 * n)) / denom
        spread = (z / denom) * math.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n))
        lo, hi = max(0.0, centre - spread), min(1.0, centre + spread)
    else:
        sem = float(np.std(values, ddof=1)) / math.sqrt(n) if n > 1 else 0.0
        lo, hi = max(0.0, mean - z * sem), min(1.0, mean + z * sem)
    return {"n": n, "mean": mean, "lo": round(lo, 4), "hi": round(hi, 4),
            "half_width": round((hi - lo) / 2.0, 4),
            "estimator": "wilson" if binary else "sem"}


def _adapter(spec):
    """Build the adapter for a spec, or None if this repo has no environment for it."""
    from bird import registry

    registry.load_all()
    if spec.bird_env_id is None or spec.bird_env_id not in registry.names("env"):
        return None
    return registry.get("env", spec.bird_env_id)(None)


def _expert_policy(env):
    """The benchmark's own scripted solution, where one exists.

    Meta-World ships `ENV_POLICY_MAP`; nothing else here does. `None` is a real answer
    and the caller records the anchor as `unavailable` WITH a reason -- an expert anchor
    invented for an environment that has no expert would be a fabricated ceiling, and
    every human-normalised number on that env would be scaled against it.
    """
    try:
        from metaworld.policies import ENV_POLICY_MAP
    except ImportError:
        return None
    task = getattr(env, "task", None)
    return ENV_POLICY_MAP[task]() if task in (ENV_POLICY_MAP or {}) else None


def _construct(env_id: str):
    """The adapter for `env_id`, through the registry. An env whose spec does not exist
    yet raises `TaskSpecError` at construction; a generator that measures BEFORE it writes
    the spec constructs its own spec-less adapter (`gen_assistax_specs.construct_without_spec`)
    and calls `_measure_with` on it directly."""
    from bird import registry

    registry.load_all()
    if env_id not in registry.names("env"):
        raise SystemExit(f"unknown env {env_id!r}; registered: {sorted(registry.names('env'))}")
    return registry.get("env", env_id)(None)


def measure_env(env_id: str, n: int, seed: int) -> Dict[str, Dict[str, float]]:
    """Anchors for a REGISTERED ENV, with no spec required.

    The bootstrap path: a new spec cannot carry measured anchors before it exists, and
    authoring it with invented ones is exactly what the schema's "absence is explicit"
    rule is against. Measure first, then write the file.
    """
    return _measure_with(_construct(env_id), n, seed)


def _measure_env_job(args: Tuple[str, int, int]) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """One env in a worker process (`--jobs`): MuJoCo and a GL context are per process."""
    env_id, n, seed = args
    return env_id, measure_env(env_id, n, seed)


def measure(task_id: str, n: int, seed: int) -> Dict[str, Dict[str, float]]:
    spec = tasks.load(task_id)
    env = _adapter(spec)
    if env is None:
        raise SystemExit(f"{task_id}: no registered environment for {spec.bird_env_id!r}")
    return _measure_with(env, n, seed)


def _measure_with(env, n: int, seed: int) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for role in ("random", "expert"):
        policy = None if role == "random" else _expert_policy(env)
        if role == "expert" and policy is None:
            out[role] = {}                     # recorded as unavailable by the caller
            continue
        rng = np.random.default_rng(seed)
        per = [_episode(env, policy, rng) for _ in range(n)]
        out[role] = {
            PER_STEP: float(np.mean([e[PER_STEP] for e in per])),
            ANY_STEP: float(np.mean([e[ANY_STEP] for e in per])),
            "ci": {PER_STEP: interval([e[PER_STEP] for e in per]),
                   ANY_STEP: interval([e[ANY_STEP] for e in per])},
            # The raw per-episode values, so any statistic anyone wants later -- a
            # different interval, a bootstrap, a distributional check -- is derivable
            # without paying for the rollouts again. The estimator branch itself is not
            # recoverable from a mean.
            "episodes": {PER_STEP: [round(e[PER_STEP], 6) for e in per],
                         ANY_STEP: [round(e[ANY_STEP], 6) for e in per]},
        }
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task ids (default: every spec this repo can build)")
    ap.add_argument("--env", nargs="*", default=None,
                    help="registered env ids, measured with no spec required -- the "
                         "bootstrap path when authoring a new spec")
    ap.add_argument("--n", type=int, default=30, help="episodes per policy (default 30)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="raw record (default tasks/_anchors_<date>.json for --tasks, keyed by "
                         "spec id; tasks/_anchors_env_<date>.json for --env, keyed by env id)")
    ap.add_argument("--no-write", action="store_true", help="measure and print; write nothing")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes for --env (each constructs its own adapters)")
    args = ap.parse_args(argv)

    from bird import registry

    registry.load_all()
    envs = set(registry.names("env"))
    ids = args.tasks or sorted(s.id for s in tasks.index().values()
                               if s.bird_env_id in envs)
    if args.env:
        # A record keyed by ENV ID (the `--tasks` record is keyed by spec id): an env
        # measured before its spec exists, and a spec generator may read this file. Written incrementally so a killed run keeps what it
        # measured; `--no-write` prints only.
        if args.n < 30 and not args.no_write:
            print(f"refusing to write anchors from n={args.n}: the record would carry an "
                  "n_episodes the specs stamp and nobody should normalise against. Use "
                  "--no-write to explore.", file=sys.stderr)
            return 2
        out_path = None if args.no_write else Path(
            args.out or f"tasks/_anchors_env_{date.today().isoformat()}.json")
        record: Dict[str, Any] = {"date": date.today().isoformat(), "n_episodes": args.n,
                                  "seed": args.seed, "envs": {}}
        if out_path is not None and out_path.is_file():
            record = json.loads(out_path.read_text())
            record.setdefault("envs", {})
            # One record, one protocol: a resumed run must measure what the file says it
            # holds, or the spec generators stamp the file's n/seed over rows taken
            # under another.
            for key, want in (("n_episodes", args.n), ("seed", args.seed)):
                if record.get(key) != want:
                    print(f"refusing to extend {out_path}: it records {key}={record.get(key)!r} "
                          f"and this run asks for {want!r}. Use a new --out for a new protocol.",
                          file=sys.stderr)
                    return 2
        todo = [e for e in args.env if e not in record["envs"]]
        jobs = [(env_id, args.n, args.seed) for env_id in todo]

        def _report(env_id, got):
            def _f(role, red):
                if not got.get(role):
                    return "    --      "
                ci = got[role]["ci"][red]
                return f"{got[role][red]:.4f}+/-{ci['half_width']:.4f}"
            print(f"{env_id:<70} "
                  f"per_step  random={_f('random', PER_STEP)} expert={_f('expert', PER_STEP)}   "
                  f"any_step  random={_f('random', ANY_STEP)} expert={_f('expert', ANY_STEP)}",
                  flush=True)
            record["envs"][env_id] = got
            if out_path is not None and not args.no_write:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # atomic: this is the resumable record, and a kill mid-write would leave
                # a truncated file the next run's json.loads refuses instead of resuming
                tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                tmp.write_text(json.dumps(record, indent=1) + "\n")
                os.replace(tmp, out_path)

        if args.jobs > 1 and len(jobs) > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(args.jobs) as pool:
                for env_id, got in pool.imap_unordered(_measure_env_job, jobs):
                    _report(env_id, got)
        else:
            for env_id, n, seed in jobs:
                _report(env_id, measure_env(env_id, n, seed))
        return 0

    if args.n < 30 and not args.no_write:
        print(f"refusing to write anchors from n={args.n}: the record would carry an "
              "n_episodes the schema shows and nobody should normalise against. Use "
              "--no-write to explore.", file=sys.stderr)
        return 2

    stamp = date.today().isoformat()
    record = {"date": stamp, "n_episodes": args.n, "seed": args.seed, "tasks": {}}
    for task_id in ids:
        got = measure(task_id, args.n, args.seed)
        record["tasks"][task_id] = got
        def _f(role, red):
            if not got.get(role):
                return "    --      "
            ci = got[role]["ci"][red]
            return f"{got[role][red]:.4f}+/-{ci['half_width']:.4f}"
        print(f"{task_id:<26} "
              f"per_step  random={_f('random', PER_STEP)} expert={_f('expert', PER_STEP)}   "
              f"any_step  random={_f('random', ANY_STEP)} expert={_f('expert', ANY_STEP)}")

    if args.no_write:
        return 0
    out = Path(args.out or f"tasks/_anchors_{stamp}.json")
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, out)
    print(f"\nwrote {out}. Fold it into the specs by re-running the spec generator "
          "that reads it (scripts/gen_mt50_specs.py reads ANCHORS_RECORD), or by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
