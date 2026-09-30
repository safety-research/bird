"""Demonstration policies as reward VERIFIERS -- the evidence behind
`verify.quality_screen: demo_margin` and `evaluate.artifacts: demo_reward_traces`.

The idea: a reward candidate can be checked WITHOUT training a
policy on it. Roll out policies whose quality we already know -- the task's
solution policy from `policies/`, that policy with action noise, a uniform-random
policy -- and score each rollout with the candidate's own reward. A reward worth
training on pays the expert more than random, and pays a monotone amount along
the quality ladder in between. One that does not is thrown away before it costs
a million environment steps. This is CARD's Trajectory Preference Evaluation
with a better trajectory store: CARD orders its OWN past rollouts by the env's
success flag; here the store is a graded policy set and the order is known by
construction.

WHAT THIS IS, AND IS NOT, ALLOWED TO READ. A solution policy is privileged
information -- demonstration access -- and a config that uses it declares
`problem.fitness_access: demonstrations` (`_check_coherence` refuses the screen
otherwise). What is never read is `env.task_metric`, `env.success` or
`env.reference_reward`: nothing here learns how the demonstration DID on the
task, only what the candidate reward PAID it. That is the line between "demos"
and "ground truth", and `policies/README.md`'s rule that a solution script is
never shown to the reward-designing LLM still holds: the LLM and the judge see
the candidate's reward along the rollout as a sequence of numbers, never the
policy or its code.

Every rollout is charged to `Budget.record_rollout_steps`, so the cost of
verifying is visible beside the cost of training it replaces.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bird.types import Trajectory

__all__ = [
    "DemoPolicy",
    "demo_record_for",
    "policy_set",
    "rollouts_under",
    "quality_ladder",
    "strided",
]

#: How many values a per-step series keeps when it is rendered for a reader
#: (the judge, the reflecting LLM, the journal). Strided, never truncated, so
#: the tail of the episode is always present -- the same rule `_even_indices`
#: applies to video frames.
SERIES_POINTS = 12


class DemoPolicy:
    """`policy(s) -> a` for `training._rollout`, with an episode `reset`.

    `training._rollout` resets the ENV and then calls the policy per step; a
    registry policy is a phase machine that needs its own reset per episode
    (`policy_api.Policy.reset`), so the caller resets this wrapper before each
    rollout. `quality` orders the set for the monotonicity check: 0 is the best
    policy, larger is worse.
    """

    def __init__(self, name: str, quality: int, act: Callable[[np.ndarray, int], np.ndarray],
                 reset: Optional[Callable[[np.random.Generator], None]] = None) -> None:
        self.name = name
        self.quality = int(quality)
        self._act = act
        self._reset = reset
        self._t = 0

    def reset(self, rng: np.random.Generator) -> None:
        self._t = 0
        if self._reset is not None:
            self._reset(rng)

    def __call__(self, s: np.ndarray) -> np.ndarray:
        a = self._act(s, self._t)
        self._t += 1
        return np.asarray(a, dtype=float).ravel()


def demo_record_for(env_id: str) -> Optional[Any]:
    """The best `policies/` record for `env_id`, or None when the registry has none.

    "Best" is the manifest's `score.value` among `status: solved` entries. The
    registry is the ONLY source: `policies/` is a committed repo
    artifact at the same trust level as `bird/` (`policy_api`'s trust model), and
    a demonstration that came from anywhere else would be a fabricated pin.
    """
    from bird.policies import index as policy_index  # stdlib+pyyaml only
    best, best_score = None, -math.inf
    for rec in policy_index().values():
        if rec.env_id != env_id or rec.status != "solved":
            continue
        val = (rec.score or {}).get("value")
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if v > best_score:
            best, best_score = rec, v
    return best


def _bundled_metaworld_expert(env: Any, env_id: str) -> Optional[DemoPolicy]:
    """Meta-World's own scripted policy for an `mt10_<task>`/`mt50_<task>` env, or None.

    `metaworld.policies.ENV_POLICY_MAP[task]` is the benchmark's shipped
    solution -- the policy `tasks/<task>/shared_spec.yaml`'s expert anchors
    were measured with. It always constructs, which is why it is the fallback:
    a registry record whose constants were never archived (`params: null`,
    where the committed defaults may score 0.0) must not become a silent
    zero-quality "expert".
    """
    # Both Meta-World prefixes (`bird/tasks.py::METAWORLD_MT10` decides which a task
    # carries); an `mt10_`-only gate here would make every expert-needing screen a
    # silent no-op on the forty MT50 tasks.
    prefix = next((p for p in ("mt10_", "mt50_") if env_id.startswith(p)), None)
    if prefix is None:
        return None
    try:
        from metaworld.policies import ENV_POLICY_MAP  # the metaworld venv only
    except ImportError:
        return None
    task = env_id[len(prefix):]
    cls = ENV_POLICY_MAP.get(task)
    if cls is None:
        return None
    lo, hi = env.action_low, env.action_high
    box: Dict[str, Any] = {"pol": cls()}

    def act(s: np.ndarray, t: int) -> np.ndarray:  # noqa: ARG001
        # The scripted policies mutate the observation in place; the adapter's
        # arrays are frozen (`MetaWorld._emit`), so hand over a writable copy.
        a = box["pol"].get_action(np.array(s, dtype=np.float64, copy=True))
        return np.clip(np.asarray(a, dtype=float).ravel(), lo, hi)

    def reset(rng: np.random.Generator) -> None:  # noqa: ARG001
        box["pol"] = cls()  # a fresh phase machine per episode

    return DemoPolicy(f"expert[metaworld:{task}]", 0, act, reset=reset)


def _expert(env: Any, env_id: str) -> Tuple[Optional[DemoPolicy], str]:
    """The solution policy for this env, wrapped -- or (None, why).

    Order: the `policies/` registry record (a scripted policy measured through
    the BIRD adapter), PROBED by constructing it once -- a registered entry can
    still fail to construct (`params: null` on a factory that needs them) --
    then Meta-World's bundled scripted policy, then the
    adapter's own `expert_policy()` (an analytic law an env of ours ships, e.g.
    toy_reacher's PD controller; `EnvAdapter.expert_policy` states what may
    live there). Which one was used is in the policy's name and therefore in
    every journal event: `expert[<record id>]`, `expert[metaworld:<task>]`,
    `expert[env:<env id>]`.
    """
    lo, hi = env.action_low, env.action_high
    rec = demo_record_for(env_id)
    why = f"no solved policy for {env_id!r} in policies/"
    if rec is not None:
        try:
            from bird.policy_api import load_policy
            pol = load_policy(rec.id)
            pol.reset(np.random.default_rng(0))  # the probe: factories are called lazily here
            if rec.params is None and rec.params_file is None and \
                    not (rec.entry or {}).get("kwargs_from"):
                raise RuntimeError("manifest carries no params for a factory that takes them "
                                   "(the policy's constants were not archived)")

            def act(s: np.ndarray, t: int) -> np.ndarray:
                a = pol.act(np.array(s, dtype=float, copy=True), t=t, env=env)
                return np.clip(np.asarray(a, dtype=float).ravel(), lo, hi)

            return DemoPolicy(f"expert[{rec.id}]", 0, act, reset=pol.reset), ""
        except Exception as exc:  # noqa: BLE001 -- fall through to the bundled policy
            why = f"policies/ record {rec.id} unusable ({type(exc).__name__}: {exc})"
    bundled = _bundled_metaworld_expert(env, env_id)
    if bundled is not None:
        return bundled, ""
    shipped = getattr(env, "expert_policy", None)
    shipped = shipped() if callable(shipped) else None
    if shipped is not None:
        def act_env(s: np.ndarray, t: int) -> np.ndarray:
            a = shipped(np.array(s, dtype=float, copy=True), t)
            return np.clip(np.asarray(a, dtype=float).ravel(), lo, hi)

        return DemoPolicy(f"expert[env:{env_id}]", 0, act_env), ""
    return None, why


def _noised(base: DemoPolicy, sigma: float, quality: int, rng_box: Dict[str, Any],
            env: Any) -> DemoPolicy:
    """`base` plus Gaussian action noise of `sigma` x half the action range."""
    lo, hi = env.action_low, env.action_high
    half = (np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)) / 2.0

    def act(s: np.ndarray, t: int) -> np.ndarray:
        a = base._act(s, t)  # noqa: SLF001 -- the wrapped step, without base's counter
        noise = rng_box["rng"].normal(0.0, sigma, size=np.shape(a)) * half
        return np.clip(np.asarray(a, dtype=float).ravel() + noise, lo, hi)

    def reset(rng: np.random.Generator) -> None:
        base.reset(rng)
        rng_box["rng"] = rng

    return DemoPolicy(f"expert+noise({sigma:g})", quality, act, reset=reset)


def _random(env: Any, quality: int, rng_box: Dict[str, Any]) -> DemoPolicy:
    lo = np.asarray(env.action_low, dtype=float)
    hi = np.asarray(env.action_high, dtype=float)

    def act(s: np.ndarray, t: int) -> np.ndarray:  # noqa: ARG001
        return rng_box["rng"].uniform(lo, hi)

    def reset(rng: np.random.Generator) -> None:
        rng_box["rng"] = rng

    return DemoPolicy("random", quality, act, reset=reset)


def policy_set(env: Any, env_id: str, names: Sequence[str]) -> Tuple[List[DemoPolicy], str]:
    """Build the graded set named by `verify.demo_screen.policies`.

    Names: `expert`, `expert_noise:<sigma>`, `random`. Returned in QUALITY
    order (expert first, random last) whatever order the config lists them,
    because the monotonicity score is a rank correlation against that order.
    An env with no registered expert returns `([], reason)` and the screen
    passes everything with the reason in the journal -- failing OPEN, like
    TPE's cold start, because a missing demonstration is a fact about the
    catalogue and must not read as "every reward is bad".
    """
    parsed: List[Tuple[int, str, float]] = []  # (sort key, kind, sigma)
    for raw in names:
        n = str(raw).strip()
        if n == "expert":
            parsed.append((0, "expert", 0.0))
        elif n.startswith("expert_noise:"):
            try:
                sigma = float(n.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"verify.demo_screen.policies: bad entry {raw!r}") from exc
            parsed.append((1, "noise", sigma))
        elif n == "random":
            parsed.append((2, "random", 0.0))
        else:
            raise ValueError(f"verify.demo_screen.policies: unknown entry {raw!r} "
                             "(expert | expert_noise:<sigma> | random)")
    parsed.sort(key=lambda p: (p[0], p[2]))

    needs_expert = any(k in ("expert", "noise") for _, k, _ in parsed)
    expert, why = (_expert(env, env_id) if needs_expert else (None, ""))
    if needs_expert and expert is None:
        return [], why

    out: List[DemoPolicy] = []
    for q, (_, kind, sigma) in enumerate(parsed):
        if kind == "expert":
            assert expert is not None
            out.append(DemoPolicy(expert.name, q, expert._act, reset=expert.reset))  # noqa: SLF001
        elif kind == "noise":
            assert expert is not None
            out.append(_noised(expert, sigma, q, {"rng": np.random.default_rng(0)}, env))
        else:
            out.append(_random(env, q, {"rng": np.random.default_rng(0)}))
    return out, ""


def rollouts_under(env: Any, reward: Callable[..., Any], pol: DemoPolicy,
                   seeds: Sequence[int],
                   on_error: Callable[[BaseException], float]) -> Tuple[List[Trajectory], int]:
    """`len(seeds)` episodes of `pol`, each scored by the CANDIDATE's reward.

    Deterministic in `seeds`: the env and the policy are both reset from the
    same seed, so the same candidate over the same set on the same env is the
    same number twice -- which `tests/test_parallelism.py`'s bit-identity
    requirement needs and a screen that consumed the run's RNG would break.
    Returns the trajectories and the env steps spent, for `Budget`.
    """
    from bird.components.training import _rollout  # heavy module; import at call
    trajs: List[Trajectory] = []
    steps = 0
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        pol.reset(np.random.default_rng(int(seed) + 1))
        traj, n, _gt = _rollout(env, pol, rng, reward, on_error)
        # `_rollout` computes `gt_return` for its training-side callers; it is
        # discarded here on purpose and never stored -- see the module docstring.
        trajs.append(traj)
        steps += int(n)
    return trajs, steps


def quality_ladder(returns_by_quality: Sequence[Tuple[int, float]]) -> float:
    """Spearman rank correlation between "better policy" and "higher return".

    +1: the candidate reward orders the whole ladder correctly; -1: inverted;
    0 for a two-policy set is impossible (it is +1 or -1), so with only
    `expert` and `random` this is the sign of the margin and nothing more.
    """
    if len(returns_by_quality) < 2:
        return 0.0
    q = np.asarray([-float(a) for a, _ in returns_by_quality])  # better = larger
    r = np.asarray([float(b) for _, b in returns_by_quality])
    if np.ptp(r) == 0.0:
        return 0.0
    qr = np.argsort(np.argsort(q)).astype(float)
    rr = np.argsort(np.argsort(r)).astype(float)
    qr -= qr.mean()
    rr -= rr.mean()
    den = float(np.sqrt((qr ** 2).sum() * (rr ** 2).sum()))
    return float((qr * rr).sum() / den) if den > 0 else 0.0


def strided(seq: Sequence[Optional[float]], n: int = SERIES_POINTS) -> List[Optional[float]]:
    """`n` evenly spaced values of `seq` including both ends; the whole of a short one.

    `None` passes through at its own index: a per-step component series holds
    `None` on a step where the reward claimed nothing, and a strided
    view that dropped it would shift every later value onto the wrong step."""
    vals = [None if v is None else float(v) for v in seq]
    if len(vals) <= n:
        return vals
    idx = np.linspace(0, len(vals) - 1, n)
    return [vals[int(round(i))] for i in idx]
