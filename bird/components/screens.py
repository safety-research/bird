"""Stage 2, screen half: behavioural culling before we pay for a training run.

Quality screens are the single highest-leverage axis of stage 2, and the
literature has only three points on it. The three published screens
share four attributes (score source / keep rule / cost per candidate /
cold-start) and differ on **all four**. The cold-start polarity is the one that
bites, so it is stated in every docstring below and implemented literally:

    tac      (GT)    CUT     -- sigma is undefined on an empty `D_pref`, but the
                               `keep_top_n` cut still applies: GT trains 5 in
                               EVERY iteration, the first included
    cascade  (LIMEN) LIVE    -- needs no store, threshold-based from round 0
    tpe      (CARD)  VACUOUS -- an empty or one-sided store passes everything,
                               so a never-succeeding reward always trains

That last one is a real property of the method (TPE's blind spot, the mirror
of TAC's), not a bug to be quietly fixed here.

Two invariants hold for every screen in this file:

* **A screen never mutates a candidate it passes.** Not `verify_records`, not
  `meta`. A screen's opinion about a candidate it admits is written to the run
  journal via `ctx.event`, never onto the object -- otherwise "passed the
  screen" and "was inspected by the screen" become indistinguishable
  downstream.
* **Screen rejections are the `screened` population**, never `invalid`.
  Validity failures are the producer's fault and live in
  `verification.py`; a screen rejection is a *budget* judgment about a program
  that compiles perfectly well.

EPIC / STARC / `policy_rank_corr` are unpublished: a pseudometric screen needs
no labels at all and attacks the dominant cost term with neither blind spot.
They are implemented honestly and their approximations are named; silent
wrongness would defeat the point of having them.

Provenance markers: **†** paper and released code disagree; **‡** the paper is
silent and the released code (or this repo) supplies the value.
"""

from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..budget import BudgetExceeded
from ..registry import register
from ..types import Candidate, Trajectory
from .verification import (
    Transition,
    compile_reward,
    reward_scalar,
    sample_transitions,
)

log = logging.getLogger("bird")

#: How many draws to average the EPIC/STARC shaping expectation over. More is
#: tighter; 128 keeps a whole-method test in the sub-second range and the
#: estimator's noise well under the differences the screen is asked to rank.
CANON_SAMPLES = 128

#: Actions compared per state by `policy_rank_corr` when building the induced
#: greedy policy. ‡ Unpublished screen, so there is no reference value.
POLICY_ACTIONS_PER_STATE = 8


# ==========================================================================
# Shared plumbing
# ==========================================================================


def _live(candidates: Sequence[Candidate]) -> List[Candidate]:
    """Candidates a screen may have an opinion about: still trainable, non-empty."""
    return [c for c in candidates if c.trainable and (c.reward_code or "").strip()]


def _reject(candidate: Candidate, reason: str, **record: Any) -> None:
    """Move a candidate into the `screened` population. Mutation is allowed
    here and only here: this candidate did not pass."""
    candidate.screened_out = True
    candidate.failure = reason
    candidate.failure_kind = "screened"
    candidate.verify_records.append({"phase": "screen", "ok": False,
                                     "detail": reason, **record})


def _compiled(ctx: Any, candidates: Sequence[Candidate]) -> Dict[str, Callable[..., Any]]:
    """Compile every live candidate. Uncompilable ones are simply absent.

    A screen has no opinion about a program that will not compile -- that is
    the validity half's verdict, and stage 2 runs the screen even when
    `verify.enabled` is false, so the screen must not invent an `invalid`
    rejection of its own.
    """
    out: Dict[str, Callable[..., Any]] = {}
    for c in candidates:
        fn, err = compile_reward(ctx, c)
        if fn is None:
            ctx.event("screen_skip", cand_id=c.cand_id, reason=err)
            continue
        out[c.cand_id] = fn
    return out


def _keep_top_n(ctx: Any) -> int:
    """The rank keep-count, `verify.alignment_filter.keep_top_n` (GT: 5 of 10).

    Shared by every rank-based screen here. §2 gives the pseudometric screens no
    threshold key of their own because nobody has published one, and inventing a
    YAML key would fabricate a pin; reusing the existing rank keep-count gives
    them GT's property instead -- rank-based, so they can never reject a whole
    round.
    """
    return max(int(ctx.cfg.get("verify.alignment_filter.keep_top_n") or 1), 1)


def _rank_keep(ctx: Any, candidates: Sequence[Candidate], scores: Dict[str, float],
               keep_n: int, label: str) -> List[Candidate]:
    """Keep the `keep_n` highest scorers; screen out the rest.

    Ties break on `cand_id`, deterministically -- a non-trivial fraction of
    multi-candidate rounds can be decided by `np.argmax`'s lowest-index
    accident, reason enough not to leave the order to chance here too.
    """
    ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
    keep = set(ranked[:keep_n])
    by_id = {c.cand_id: c for c in candidates}
    for rank, cid in enumerate(ranked):
        if cid in keep:
            # Passing candidates are returned untouched; the journal carries the score.
            ctx.event("screen_pass", screen=label, cand_id=cid,
                      score=scores[cid], rank=rank + 1)
            continue
        _reject(by_id[cid], f"{label}: rank {rank + 1}/{len(ranked)} "
                            f"(score {scores[cid]:.4f}), keep_top_n={keep_n}",
                screen=label, score=scores[cid], rank=rank + 1)
    return list(candidates)


# --------------------------------------------------------------------------
# Re-scoring stored trajectories with a candidate's own reward
# --------------------------------------------------------------------------


def _traj_transitions(traj: Trajectory) -> Optional[List[Transition]]:
    """``(s, a, s')`` triples from a stored rollout, or None if it kept no states.

    `Trajectory.states`/`.actions` are `Any` because the env adapter owns their
    layout; all we require is that they are indexable and the same length. When
    they are absent we return None rather than falling back to the recorded
    `rewards`: those were produced by a *different* reward function, and
    silently ranking candidates by someone else's reward is exactly the kind of
    quiet wrongness a screen must not commit.
    """
    states, actions = traj.states, traj.actions
    if states is None:
        return None
    try:
        n = len(states)
    except TypeError:
        return None
    if n < 2:
        return None
    acts = actions if actions is not None else [None] * n
    try:
        n_a = len(acts)
    except TypeError:
        return None
    out: List[Transition] = []
    for t in range(n - 1):
        out.append(Transition(states[t], acts[t] if t < n_a else None, states[t + 1]))
    return out


def _rescore_transitions(ctx: Any, fn: Callable[..., Any],
                         trs: Sequence[Transition]) -> Tuple[str, List[float]]:
    """Re-run `fn` over ``(s, a, s')`` triples: ``('ok', per_step)`` or
    ``('reward_error', [])``.

    A raise or a non-finite value is a REWARD pathology, not missing data, and
    the two must stay distinguishable: CARD's release
    crashes outright on a raising reward and a NaN success fails its strict
    ``>`` comparison, counting as incorrectly ordered (utils.py:264-276,
    :297-300) -- so a caller must be able to count an errored re-score AGAINST
    the candidate rather than dropping the trajectory from the statistic.
    """
    rewards: List[float] = []
    for tr in trs:
        try:
            rewards.append(reward_scalar(ctx, fn, tr))
        except Exception as exc:
            log.debug("screen: reward raised while re-scoring (%s)", exc)
            return "reward_error", []
    if not all(math.isfinite(r) for r in rewards):
        return "reward_error", []
    return "ok", rewards


def _aggregate_rewards(rewards: Sequence[float], discount: Optional[float]) -> float:
    """`discount=None` is the undiscounted **mean per step** -- the length-bias
    correction both TAC and TPE specify, and (†) what CARD's released code
    actually computes. A float applies gamma^t, which is what CARD's Def. 4.1
    writes. Both average over the T executed steps: the release appends one
    obs per step (utils.py:145, :171), so its len(obs) is T and there is no
    1/n divergence between the two.
    """
    if discount is None:
        return float(np.mean(rewards))
    g = float(discount)
    return float(sum((g ** t) * r for t, r in enumerate(rewards)))


def _score_transitions(ctx: Any, fn: Callable[..., Any], trs: Sequence[Transition],
                       discount: Optional[float]) -> Optional[float]:
    """Return of a transition sequence under `fn`, or None on ANY trouble.

    The None-on-anything compatibility wrapper over `_rescore_transitions`:
    TAC keeps exactly this contract (a pair with an unscorable side is excluded,
    see `_score_traj`'s window note), and only TPE needs the finer-grained
    statuses.
    """
    if not trs:
        return None
    status, rewards = _rescore_transitions(ctx, fn, trs)
    if status != "ok" or not rewards:
        return None
    return _aggregate_rewards(rewards, discount)


def _score_traj(ctx: Any, fn: Callable[..., Any], traj: Optional[Trajectory],
                discount: Optional[float] = None,
                window: Optional[Tuple[int, int]] = None) -> Optional[float]:
    if traj is None:
        return None
    trs = _traj_transitions(traj)
    if trs is None:
        return None
    if window is not None:
        lo, hi = window
        trs = trs[lo:hi]
        # A window that falls off this side of the pair would leave us comparing
        # a slice against a whole rollout. Exclude the pair instead.
        if not trs:
            return None
    return _score_transitions(ctx, fn, trs, discount)


def _rescore_traj(ctx: Any, fn: Callable[..., Any], traj: Optional[Trajectory],
                  discount: Optional[float],
                  window: Optional[Tuple[int, int]] = None) -> Tuple[str, Optional[float], List[float]]:
    """``('ok', aggregate, per_step)`` | ``('reward_error', None, [])`` |
    ``('no_data', None, [])``.

    TPE's scorer: unlike `_score_traj` it separates a
    reward that BROKE on stored states (counted against the candidate) from a
    trajectory that kept no re-scorable states (store quality -- not the
    candidate's fault, excluded from the statistic).
    """
    trs = _traj_transitions(traj) if traj is not None else None
    if trs and window is not None:
        lo, hi = window
        trs = trs[lo:hi]
    if not trs:
        return "no_data", None, []
    status, rewards = _rescore_transitions(ctx, fn, trs)
    if status != "ok":
        return "reward_error", None, []
    return "ok", _aggregate_rewards(rewards, discount), rewards


# --------------------------------------------------------------------------
# Reference rewards (EPIC / STARC / policy_rank_corr)
# --------------------------------------------------------------------------


def reference_reward(ctx: Any, state: Any) -> Tuple[Optional[Callable[..., Any]], str]:
    """Resolve `evaluate.similarity.reference` to a callable.

    §4 declares the reference once, for the similarity *metric*, and promoting
    `evaluate.similarity.role` from `report` to `screen` is one key away.
    Rather than duplicate the declaration, the pseudometric screens read the
    same key -- so `epic` as a screen and `epic` as a report are measuring
    against the same thing by construction.
    """
    ref = ctx.cfg.get("evaluate.similarity.reference", "gt_reward")
    if ref == "none":
        return None, "evaluate.similarity.reference=none"
    if ref == "previous_best":
        best = getattr(state, "best", None) if state is not None else None
        if best is None:
            return None, "no incumbent yet (iteration 1)"
        fn, err = compile_reward(ctx, best.candidate)
        return (fn, "") if fn is not None else (None, f"incumbent will not compile: {err}")
    env = getattr(ctx, "env", None)
    for attr in ("gt_reward", "ground_truth_reward", "true_reward", "reward_fn",
                 "compute_reward"):
        fn = getattr(env, attr, None)
        if callable(fn):
            return fn, ""
    return None, "env adapter exposes no ground-truth reward"


# ==========================================================================
# EPIC / STARC canonicalisation  (also used by dedup:epic_distance)
# ==========================================================================


def _gamma(ctx: Any) -> float:
    """Discount for canonicalisation. `train.hyperparameters` is a schema leaf,
    so a method that publishes gamma (GT: 0.9896, App. E Table 3) supplies it
    from there; 0.99 otherwise."""
    hp = ctx.cfg.get("train.hyperparameters") or {}
    try:
        return float(hp.get("gamma", 0.99))
    except (TypeError, ValueError):
        return 0.99


def epic_canonical_profile(ctx: Any, fn: Callable[..., Any],
                           transitions: Sequence[Transition]) -> Optional[np.ndarray]:
    """EPIC canonicalisation of `fn`, evaluated on `transitions`.

    Gleave et al.'s canonically-shaped reward, with the shaping expectation
    taken over sampled states and actions:

        C(R)(s,a,s') = R(s,a,s')
                       + gamma * E_{A,S'}[R(s', A, S')]
                       -         E_{A,S'}[R(s,  A, S')]
                       - gamma * E_{S,A,S'}[R(S, A, S')]

    Subtracting the expected shaping term is what makes the profile invariant
    to potential shaping, which is the property that lets two rewards inducing
    the same optimal policies compare as identical. The expectations are Monte
    Carlo over :data:`CANON_SAMPLES` draws from the same transition sampler --
    an approximation, and named as one: EPIC's own guarantees are stated for
    exact expectations over a coverage distribution, and ours is the empirical
    distribution the env adapter hands us.

    Returns None if the reward cannot be evaluated; the caller then has no
    opinion about that candidate rather than a wrong one.
    """
    if not transitions:
        return None
    g = _gamma(ctx)
    draws = sample_transitions(ctx, CANON_SAMPLES)
    try:
        # E over (A, S') for a fixed first state, and the fully-marginal term.
        base = np.array([reward_scalar(ctx, fn, d) for d in draws], dtype=float)
        if not np.all(np.isfinite(base)):
            return None
        marginal = float(np.mean(base))

        def shaped(s: Any) -> float:
            vals = [reward_scalar(ctx, fn, Transition(s, d.action, d.next_state))
                    for d in draws]
            return float(np.mean(vals))

        out = np.empty(len(transitions), dtype=float)
        for i, tr in enumerate(transitions):
            r = reward_scalar(ctx, fn, tr)
            out[i] = r + g * shaped(tr.next_state) - shaped(tr.state) - g * marginal
    except Exception as exc:
        log.debug("EPIC canonicalisation failed (%s)", exc)
        return None
    return out if np.all(np.isfinite(out)) else None


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = float(np.std(a)), float(np.std(b))
    if sa <= 0.0 or sb <= 0.0:
        # A constant canonical profile has no correlation with anything. Treat
        # it as maximally dissimilar rather than as a division by zero.
        return 0.0
    return float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (sa * sb))


def epic_distance(a: np.ndarray, b: np.ndarray) -> float:
    """EPIC pseudometric between two canonical profiles: sqrt((1 - rho) / 2).

    In [0, 1]: 0 = same ordering up to positive affine transformation and
    shaping, 1 = perfectly anti-correlated.
    """
    if a is None or b is None or len(a) != len(b) or len(a) == 0:
        return 1.0
    return float(math.sqrt(max(0.0, 1.0 - _pearson(a, b)) / 2.0))


def _starc_normalise(x: np.ndarray) -> np.ndarray:
    """STARC's normalisation step: unit L2 norm after centring."""
    y = x - float(np.mean(x))
    n = float(np.linalg.norm(y))
    return y / n if n > 0 else y


def starc_distance(a: np.ndarray, b: np.ndarray) -> float:
    """STARC pseudometric: canonicalise, normalise, then metrise.

    Skalse et al.'s three-step construction, with EPIC's canonicalisation as
    `c`, unit-L2 as `s`, and half the Euclidean distance as `m` -- which puts
    the result in [0, 1] and, unlike EPIC's correlation form, gives STARC its
    bound on worst-case regret. Same approximation caveat as
    :func:`epic_canonical_profile`.
    """
    if a is None or b is None or len(a) != len(b) or len(a) == 0:
        return 1.0
    return float(0.5 * np.linalg.norm(_starc_normalise(a) - _starc_normalise(b)))


# ==========================================================================
# screen:none
# ==========================================================================


@register("screen", "none")
def screen_none(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """Identity. Every method except GT, LIMEN and CARD (§2).

    Not a placeholder: "train everything you generated" is the majority
    published position, and it is the baseline the other three are measured
    against.
    """
    return candidates


# ==========================================================================
# screen:tac  --  Gran Turismo
# ==========================================================================


@register("screen", "tac")
def screen_tac(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """GT's Trajectory Alignment Coefficient: the only published *pre-training*
    prune over a candidate set (§2).

    Score source: `state.preferences` (`D_pref`), accumulated in §4/§6. Each
    candidate re-scores both trajectories of every stored pair with its OWN
    reward; the induced preference is compared against the stored label.

        sigma = (2P - N) / N          in [-1, 1]

    a Kendall tau-a over the pair set, where P counts agreements and N is the
    dataset size. Returns are **mean per step** (`Trajectory.mean_per_step_return`
    is the same quantity for a stored rollout): successful episodes terminate
    early, so cumulative return is length-biased toward failures, and the paper
    motivates the correction explicitly.

    Sub-trajectories of `verify.alignment_filter.subtraj_len` (GT's L, never
    valued in the paper) -- the same window is applied to both sides of a pair,
    drawn from `ctx.rng` so the screen is reproducible.

    Keep rule: `verify.alignment_filter.keep_top_n` by RANK (GT keeps 5 of the
    10 written). Rank-based, so unlike a threshold it can never reject a whole
    round -- a round always has a top 5.

    COLD START: **the cut still applies**. `D_pref` is empty at iteration 1, so
    sigma is undefined -- but the paper trains the kept 5 in EVERY iteration,
    the first included, and pins that count five independent ways: §4 Training
    ("For the first iteration, each of the N reward functions are trained from
    scratch", N = the kept count), §5.1 (top five trained per iteration), §5.3
    (10 preferences/iteration = C(5,2)), App. G (20 rewards/seed = 4 x 5), and
    Figs 7-8 (5 reward functions at Iteration 0). The paper pins the COUNT, never the
    ORDER of a cut sigma cannot rank, and GT has no release to supply one --
    so the cand_id order is BIRD's OWN choice, stated plainly rather than
    marked ‡ (that marker means "released code supplies the value", and here
    none exists). GT's own §5.3 still stands: TAC only
    beats random past ~100 preferences, and their runs gathered 40 -- the
    screen is under-powered early, but it is never inert.
    """
    prefs = list(getattr(state, "preferences", []) or []) if state is not None else []
    live = _live(candidates)
    if not live:
        return candidates
    if not prefs:
        return _cold_start_cut(ctx, candidates, live, _keep_top_n(ctx),
                               "empty preference dataset")

    L = max(int(ctx.cfg.get("verify.alignment_filter.subtraj_len") or 0), 0)
    rng = getattr(ctx, "rng", None)

    # One window per preference, shared by both sides so the comparison is fair.
    windows: List[Optional[Tuple[int, int]]] = []
    for p in prefs:
        if L <= 0:
            windows.append(None)
            continue
        span = max(getattr(p.left_traj, "length", 0) or 0,
                   getattr(p.right_traj, "length", 0) or 0)
        start = 0
        if rng is not None and span > L:
            start = rng.randrange(0, span - L)
        windows.append((start, start + L))

    fns = _compiled(ctx, live)
    scores: Dict[str, float] = {}
    #: Compiled candidates whose reward raised or went non-finite on EVERY pair
    #: that carried states, with that pair count. Tracked explicitly because a
    #: candidate that is merely skipped is absent from `scores`, never ranked by
    #: `_rank_keep` and never cut by `_cold_start_cut` (which fires only when
    #: `scores` is empty) -- so in a mixed pool it would stay TRAINABLE on top of
    #: the `keep_top_n` ranked survivors with no event at all, and the count GT
    #: pins five ways would be exceeded by exactly the rewards that break on
    #: real states.
    unscorable: List[Tuple[Candidate, int]] = []
    for c in live:
        fn = fns.get(c.cand_id)
        if fn is None:
            continue
        agree = usable = broken = 0
        for p, win in zip(prefs, windows):
            l_status, left, _ = _rescore_traj(ctx, fn, p.left_traj, None, window=win)
            r_status, right, _ = _rescore_traj(ctx, fn, p.right_traj, None, window=win)
            if left is None or right is None:
                # A pair with an unscorable side is excluded from THIS candidate's
                # sigma (TAC's contract); whether it was excluded by the store
                # (no states) or by the reward (raise / non-finite) decides
                # below what a candidate with no usable pair at all means.
                if "reward_error" in (l_status, r_status):
                    broken += 1
                continue
            usable += 1
            induced = 1 if left > right else 0
            agree += int(induced == int(p.label))
        if usable == 0:
            if broken:
                unscorable.append((c, broken))
            continue
        scores[c.cand_id] = (2.0 * agree - usable) / usable

    if not scores:
        # Our extension of the same count argument: the paper's per-iteration
        # accounting never varies with whether sigma could be computed, so a
        # store whose pairs all lost their states -- or a pool whose every
        # reward broke on them -- gets the cold-start cut too.
        reason = ("no stored pair carried re-scorable states" if not unscorable else
                  "every candidate's reward raised or returned a non-finite value "
                  "on every stored pair")
        return _cold_start_cut(ctx, candidates, live, _keep_top_n(ctx), reason)

    # Something scored, so the rank cut is real -- and a reward that could not
    # be scored because it BROKE on stored states ranks below every reward
    # that could (the rule TPE's re-scorer follows: an errored re-score counts
    # against the candidate, it is not missing data).
    # Screened, with the reason journalled, rather than silently trainable.
    keep_n = _keep_top_n(ctx)
    for c, broken in unscorable:
        _reject(c, f"tac: unscorable -- the reward raised or returned a non-finite "
                   f"value on all {broken} stored pair(s) it could have been scored "
                   f"on, so it ranks below every reward that could be scored; "
                   f"keep_top_n={keep_n}",
                screen="tac", unscorable=True, broken_pairs=broken)
        ctx.event("screen_reject", screen="tac", cand_id=c.cand_id,
                  reason="unscorable_tac", broken_pairs=broken)

    return _rank_keep(ctx, candidates, scores, keep_n, "tac")


def _cold_start_cut(ctx: Any, candidates: Sequence[Candidate],
                    live: Sequence[Candidate], keep_n: int,
                    reason: str) -> List[Candidate]:
    """TAC's rank cut when sigma is undefined.

    GT trains `keep_top_n` candidates in every iteration including the first
    (§4 Training; §5.1; §5.3; App. G; Figs 7-8), and an empty `D_pref` gives
    the filter nothing to rank by -- so the cut is deterministic-arbitrary by
    `cand_id`, ascending (the paper pins the count, not the order; the order is
    BIRD's own choice, there being no GT release to take one from). Not
    routed through `_rank_keep` with invented 0.0 scores: a `screen_pass`
    event carrying a sigma nobody computed would be a fabricated record, so
    the journal gets `screen_cold_start` naming the reason and the kept ids.
    """
    ranked = sorted(live, key=lambda c: c.cand_id)
    kept = ranked[:keep_n]
    keep_ids = {c.cand_id for c in kept}
    for rank, c in enumerate(ranked):
        if c.cand_id in keep_ids:
            continue
        _reject(c, f"tac: cold-start cut rank {rank + 1}/{len(ranked)} "
                   f"(sigma undefined: {reason}), keep_top_n={keep_n}",
                screen="tac", rank=rank + 1)
    ctx.event("screen_cold_start", screen="tac", reason=reason,
              n_kept=len(kept), n_screened=len(ranked) - len(kept),
              kept=sorted(keep_ids))
    return list(candidates)


# ==========================================================================
# screen:cascade  --  LIMEN
# ==========================================================================


class _ShortRunCrashed(RuntimeError):
    """A cascade short run raised. Internal: converted into a screen rejection."""


def _train_backend_short(ctx: Any, state: Any, candidate: Candidate,
                         steps: int) -> Tuple[Any, bool]:
    """Run the configured train backend on a reduced step budget.

    The screen is **not free** (§2: it buys a short training run per candidate),
    so the run is charged. The backend normally does its own
    `budget.record_training`; we snapshot the counter and only charge ourselves
    if it did not, which keeps the cost column right without double-counting.

    The backend's signature is `(ctx, state, candidate, n_seeds=...)` per
    `bird.py`; `train.env_steps` is a config key it reads for itself. We pass an
    `env_steps` override only when the backend advertises the parameter, and
    record whether the short budget was honoured -- claiming a 1M-step screen
    that silently ran the full 25M would misreport exactly the quantity LIMEN's
    cascade exists to save.

    HONOURED IS MEASURED, NOT INFERRED FROM THE SIGNATURE. A backend that
    accepts `env_steps` and reads `train.env_steps` anyway would journal
    `honoured: true` at full cost if the flag followed the parameter's
    existence -- the signature is not where the loop bound lives.
    `_measured_short_run` reads the bound the backend's own per-seed rows say
    it ran against.
    """
    from ..registry import get  # local: registry imports this module

    backend = get("train_backend", ctx.cfg["train.backend"])
    kwargs: Dict[str, Any] = {"n_seeds": 1}
    try:
        params = inspect.signature(backend).parameters
    except (TypeError, ValueError):
        params = {}
    for key in ("env_steps", "steps", "max_steps", "budget_steps"):
        if key in params:
            kwargs[key] = steps
            break

    before = ctx.budget.policy_trainings
    try:
        res = backend(ctx, state, candidate, **kwargs)
    except BudgetExceeded:
        raise  # the loop's own control flow; not this candidate's verdict
    except Exception as exc:
        # A candidate that crashes the learner inside its short run is exactly
        # what the cascade exists to drop, so this is a screen verdict and not
        # an error. The step spend still happened, so it is still charged.
        if ctx.budget.policy_trainings == before:
            ctx.budget.record_training(env_steps=steps)
        log.debug("cascade: %s crashed during its short run (%s)", candidate.cand_id, exc)
        raise _ShortRunCrashed(f"{type(exc).__name__}: {exc}") from exc
    if ctx.budget.policy_trainings == before:
        ctx.budget.record_training(env_steps=getattr(res, "env_steps_used", steps) or steps)
    _adopted, _executed, honoured = _measured_short_run(res, steps)
    return res, honoured


def _measured_short_run(res: Any, steps: int) -> Tuple[Optional[int], Optional[int], bool]:
    """``(bound the backend ADOPTED, training steps it EXECUTED, honoured)``,
    read off `res.seed_metrics`.

    `train_steps_requested` is the number the learner loop actually ran against
    (`training._run_backend`'s `steps_budget`, and the same row in `_sb3_run`
    and `fasttd3`): `min(ask, cap)` when the override was read, `train.env_steps`
    when it was not. `train_steps` is what it executed -- a learner may overshoot
    the bound by one round, so the two differ and both are kept. Over several
    seeds the adopted bound is the largest and the executed count the sum.

    Honoured means the backend adopted a bound no larger than the ask. NOT
    `env_steps_used <= steps`: that total includes checkpoint-evaluation and
    feedback-rollout steps and exceeds the ask on every honest run (2050 for a
    400-step ask, measured). A result with no rows measured nothing, and
    unmeasured is not honoured.
    """
    adopted: Optional[int] = None
    executed: Optional[int] = None
    for m in (getattr(res, "seed_metrics", None) or []):
        if not isinstance(m, dict):
            continue
        req, ran = m.get("train_steps_requested"), m.get("train_steps")
        if req is not None:
            adopted = int(req) if adopted is None else max(adopted, int(req))
        if ran is not None:
            executed = int(ran) + (executed or 0)
    honoured = adopted is not None and adopted <= int(steps)
    return adopted, executed, honoured


def _success_rate(res: Any) -> Optional[float]:
    """Success rate out of a TrainResult, however the backend chose to report it.

    The native-signal rule (N): LIMEN's cascade screens on the env's SUCCESS
    FLAG, so this reads only genuine success-flag keys (`NATIVE_SUCCESS_KEYS`),
    never `task_success`/`score`/`fitness` -- those ALIAS the BIRD `task_metric`
    (custom_metric) in the backends' curve rows, so a fall-through to them would
    screen a supervised search on the number we wrote. The trajectory fallback
    below (`t.success`) is the env flag."""
    if res is None:
        return None
    from ..native_signal import NATIVE_SUCCESS_KEYS
    vals: List[float] = []
    for m in (getattr(res, "seed_metrics", None) or []):
        if not isinstance(m, dict):
            continue
        for key in NATIVE_SUCCESS_KEYS:
            if key in m:
                try:
                    vals.append(float(m[key]))
                except (TypeError, ValueError):
                    pass
                break
    if vals:
        return float(np.mean(vals))
    trajs = getattr(res, "trajectories", None) or []
    if trajs:
        return float(np.mean([1.0 if t.success else 0.0 for t in trajs]))
    return None


def _cascade_budget(ctx: Any) -> int:
    """`verify.cascade.short_budget_steps`, or a fraction of the real budget.

    LIMEN's three XLand tasks pin the pair (short, full) at 3M/5M, 1M/2M and
    500K/1M -- **0.60, 0.50, 0.50**. The quantity the paper holds fixed across
    tasks is the RATIO; the absolute is a lossy encoding of it that has to be
    re-derived by hand every time the full budget moves.

    That re-derivation is exactly what does not happen. An execution profile
    (`bird/config.py::PROFILE_KEY_PREFIXES`) may move `train.env_steps` and may
    NOT move `short_budget_steps` -- correctly, since the screen is a §2
    behaviour key and a profile that moved it would be running a different
    method. So a *legal* budget change silently re-tunes the screen: a
    config that keeps `short_budget_steps: 10000` and runs at
    `env_steps: 100000` gives a ratio of **0.10 against the paper's 0.60** --
    a screen six times harsher than the published one, with nobody having
    edited it. In one such run it rejected 15 of 24 candidates, so the
    population reaching `train` was not the one LIMEN selects from.

    `short_budget_fraction` lets a config pin the ratio directly, which is the
    thing that transfers. `null` keeps the absolute, so no existing config's
    hash moves.
    """
    frac = ctx.cfg.get("verify.cascade.short_budget_fraction")
    if frac is None:
        return int(ctx.cfg.get("verify.cascade.short_budget_steps") or 0)
    full = int(ctx.cfg.get("train.env_steps") or 0)
    # Round rather than truncate, and never to zero: a fraction that floored to
    # 0 would turn the screen into `_train_backend_short(steps=0)`, which is a
    # different failure wearing the same config.
    return max(1, int(round(full * float(frac))))


@register("screen", "cascade")
def screen_cascade(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """LIMEN's short-budget filter: train briefly, drop what does not move.

    Score source: an actual (cheap) training run of
    `verify.cascade.short_budget_steps` (LIMEN: 500K / 1M / 3M by task
    difficulty), scored by success rate. Keep rule:
    `verify.cascade.min_success_threshold` (0.01-0.05) -- THRESHOLD-based, so
    unlike TAC it CAN reject every candidate in a round, and a run where nothing
    clears the bar is a legitimate outcome that `loop.on_total_failure` then
    handles.

    COLD START: **live from round 0**. It needs no accumulated store, which is
    the whole trade against TAC: it pays real environment steps per candidate
    instead of waiting for labels.

    † The paper's MuJoCo protocol skips the cascade entirely, so a LIMEN config
    for those tasks sets `verify.quality_screen: none` rather than tuning the
    threshold to zero.

    A candidate whose short run reports no success signal at all is passed
    through untouched: the screen measured nothing, and rejecting on absent
    evidence would be a threshold judgment we did not actually make.
    """
    live = _live(candidates)
    if not live:
        return candidates
    steps = _cascade_budget(ctx)
    thr = float(ctx.cfg.get("verify.cascade.min_success_threshold") or 0.0)

    for c in live:
        try:
            res, honoured = _train_backend_short(ctx, state, c, steps)
        except _ShortRunCrashed as exc:
            _reject(c, f"cascade: short run crashed after {steps} steps: {exc}",
                    screen="cascade", short_budget_steps=steps, crashed=True)
            continue
        rate = _success_rate(res)
        # Report what RAN, not what was asked for. When `honoured` is false the
        # backend ignored `steps` and used `train.env_steps`, so wording the
        # rejection as "after 10000 short-budget steps" would assert the exact
        # opposite of the `short_budget_honoured: false` beside it in the same
        # record -- a 10k screen that actually cost 100k. A reader of
        # `candidates/` or `journal.jsonl` who did not cross-check the flag would
        # conclude the cascade was working, which is how a cost claim survives
        # being false. Both numbers are MEASURED off the backend's rows, never
        # copied from the request: a mock run the cap clamped to 6,000 steps
        # must not record the 3,000,000 it was asked for.
        adopted, ran, _ = _measured_short_run(res, steps)
        if ran is None:
            ran = getattr(res, "env_steps_used", None)
        spend = dict(short_budget_steps=steps, short_budget_adopted=adopted,
                     steps_actually_run=ran, short_budget_honoured=honoured)
        if rate is None:
            ctx.event("screen_pass", screen="cascade", cand_id=c.cand_id,
                      success_rate=None, note="backend reported no success signal",
                      **spend)
            continue
        if rate < thr:
            budget_note = (f"after {ran} short-budget steps" if honoured else
                           f"after {ran} steps -- THE SHORT BUDGET OF {steps} WAS NOT "
                           f"HONOURED (the backend ran against {adopted}), this screen "
                           f"cost a full training")
            _reject(c, f"cascade: success {rate:.4f} < min_success_threshold {thr:.4f} "
                       f"{budget_note}",
                    screen="cascade", success_rate=rate, **spend)
        else:
            ctx.event("screen_pass", screen="cascade", cand_id=c.cand_id,
                      success_rate=rate, **spend)
    return candidates


# ==========================================================================
# screen:tpe  --  CARD
# ==========================================================================


def _tpe_discount(ctx: Any) -> Optional[float]:
    """`verify.tpe.discount`: null = undiscounted mean-per-step, float = gamma^t.

    † Def. 4.1 writes the return discounted; the released code is undiscounted.
    The key is nullable precisely so both readings are runnable.
    """
    d = ctx.cfg.get("verify.tpe.discount")
    if d is None:
        return None
    try:
        return float(d)
    except (TypeError, ValueError):
        return None


def _tpe_verdict(rule: str, thr: float, succ: List[float],
                 fail: List[float]) -> Tuple[bool, float, str]:
    """Evaluate one of the three published readings of "order-preserving".

    All three ask whether the candidate's own reward ranks ground-truth
    successes above ground-truth failures; they disagree about what counts.
    Dispatch is on the config VALUE `verify.tpe.rule`, never on a method name.
    """
    if rule == "min_over_max":
        # Def. 4.1 literally: min over successes must exceed max over failures.
        # The threshold=1.0 special case of frac_above_max_failure.
        score = 1.0 if min(succ) > max(fail) else 0.0
        return score >= 1.0, score, f"min(success)={min(succ):.4f} vs max(failure)={max(fail):.4f}"
    if rule == "frac_above_max_failure":
        # † The released code scores each success against the HARDEST failure.
        worst = max(fail)
        score = float(np.mean([1.0 if s > worst else 0.0 for s in succ]))
        return score >= thr, score, f"{score:.4f} of successes beat max failure {worst:.4f}"
    # pair_accuracy -- §4.4's prose: count success-vs-failure pairs.
    ok = sum(1 for s in succ for f in fail if s > f)
    total = len(succ) * len(fail)
    score = ok / total if total else 0.0
    return score >= thr, score, f"pair accuracy {ok}/{total} = {score:.4f}"


@register("screen", "tpe")
def screen_tpe(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """CARD's Trajectory Preference Evaluation: order-preservation over a
    ground-truth-labelled trajectory store.

    Score source: `state.trajectory_store`, whose `Trajectory.success` flags are
    free ground truth from the environment -- no preference labels, no VLM, no
    human. The candidate's OWN reward re-scores those stored rollouts; it passes
    iff the ordering it induces puts successes above failures, under
    `verify.tpe.rule`:

        pair_accuracy          §4.4 prose: fraction of (success, failure) pairs
                               ordered correctly, vs delta = `verify.tpe.threshold` (0.8)
        frac_above_max_failure † released code: each success vs the HARDEST failure
        min_over_max           Def. 4.1's min > max -- the delta = 1.0 special case

    `verify.tpe.discount`: null = undiscounted mean-per-step (released code), a
    float applies gamma^t († Def. 4.1). `verify.tpe.trajectories_per_iteration`
    (100) is the COLLECTION count -- App. B.2 Table 8's "# of trajectories
    collected per iteration", the release's `--trajectory_num` check pool
    (metaworld_exp_one_step.py:276, :230), collected by stage 3
    (`training._n_store_rollouts`) into `TrainResult.store_trajectories` and
    appended by §6 on a passing iteration. The screen checks the FULL
    accumulated store, unwindowed: the release's `trajectories_for_check_list`
    only ever extends (:428), so at introspection 3 the check sees 200
    trajectories spanning both prior policies. (Reading the key as a
    `store[-N:]` recency cap instead would leave nothing collecting with it:
    the store would grow only by the 3-rollout feedback pool, 1/33 of the
    published evidence, and the two readings coincide only while the store
    total stays <= N.)

    COLD START: **fails OPEN**. An empty store, a store with no successes, or a
    store with no failures passes every candidate. Implemented literally,
    because it is a real property of the method: until the task has been solved
    once there is nothing to order-preserve, so a never-succeeding reward always
    trains. This is TPE's blind spot -- the mirror image of TAC's fail-closed
    cold start -- and a fail-closed TPE variant is an open direction.

    `verify.tpe.on_failure` is THE fork (†) and both arms are real:

        skip_training  §4.4, the paper: the candidate is screened out and never
                       reaches the learner. The abstract's cost claim rests on
                       this, and `budget.policy_trainings_skipped` is where it
                       becomes visible.
        train_anyway   the released drivers: the verdict is recorded and only
                       routes feedback (§6 `update.feedback.routing: tpe_verdict`);
                       training happens regardless.

    Either way the failed candidate keeps chain headship (§6) -- nothing here
    touches `generate.parent_source: latest`.

    The screen only READS the store. Its growth is §6's
    `update.memory.trajectory_store` (`verify.tpe.store: append_on_pass` is the
    self-gating declaration: the screen gates the growth of its own evidence).

    Contract for `update.feedback.routing: tpe_verdict`: a FAILING candidate
    carries `meta["tpe"]` with the verdict AND the screen's own re-scored
    evidence -- per-trajectory scores plus the (worst-success, best-failure)
    example pair with per-step values and store indices, all JSON-native
    (§6 must quote the numbers THIS candidate's reward
    produced over the store the screen judged, never the collection-time
    values). A passing one carries nothing at all, because a screen must not
    mutate what it passes. Absence means pass.
    """
    live = _live(candidates)
    if not live:
        return candidates

    store = list(getattr(state, "trajectory_store", []) or []) if state is not None else []
    # No truncation: the release checks the whole accumulated
    # `trajectories_for_check_list` -- it extends on pass
    # (metaworld_exp_one_step.py:428) and is truncated nowhere. A
    # `store[-window:]` cap would drop every earlier policy's evidence the
    # moment the store first exceeds 100 -- iteration 3 of CARD's own protocol
    # -- and could split unevenly across the success/failure populations,
    # moving max(failure) and hence every success's comparison.
    # Native-signal rule (N): the TPE partition splits the store by the env
    # success flag (`t.success`). That is only admissible where the flag is
    # NATIVE -- a task that ships no native success (the gymnasium locomotion
    # tasks) has `t.success` computed from a BIRD threshold (custom_success), so
    # partitioning on it would screen a supervised search on the number we wrote.
    # There the screen has nothing admissible to split on -> fail open (vacuous
    # pass), journalled, exactly like a one-sided store below. Meta-World
    # (discrete) and the repo-authored toy family (native_authored) DO ship a
    # native success, so CARD's screen runs faithfully there.
    from ..native_signal import channel_for_env, NATIVE_SUCCESS
    if channel_for_env(env_id=ctx.cfg.get("problem.env_id"),
                       task_id=ctx.cfg.get("problem.task_id")) != NATIVE_SUCCESS:
        ctx.event("screen_vacuous_pass", screen="tpe", n_store=len(store),
                  note="native-signal rule (N): task ships no native success, so the "
                       "TPE success/failure partition has nothing admissible to split "
                       "on (t.success would be a BIRD custom_success); fails open")
        return candidates

    successes = [(i, t) for i, t in enumerate(store) if getattr(t, "success", False)]
    failures = [(i, t) for i, t in enumerate(store) if not getattr(t, "success", False)]

    if not successes or not failures:
        # Fail open. Deliberate; see the docstring.
        ctx.event("screen_vacuous_pass", screen="tpe", n_store=len(store),
                  n_success=len(successes), n_failure=len(failures),
                  note="empty or one-sided store: TPE fails open by construction")
        return candidates

    rule = str(ctx.cfg.get("verify.tpe.rule", "pair_accuracy"))
    thr = float(ctx.cfg.get("verify.tpe.threshold") or 0.0)
    discount = _tpe_discount(ctx)
    on_failure = str(ctx.cfg.get("verify.tpe.on_failure", "skip_training"))
    fns = _compiled(ctx, live)

    for c in live:
        fn = fns.get(c.cand_id)
        if fn is None:
            continue
        succ_scored = [(i, *_rescore_traj(ctx, fn, t, discount)) for i, t in successes]
        fail_scored = [(i, *_rescore_traj(ctx, fn, t, discount)) for i, t in failures]
        # `no_data` is a property of the STORE, not of this reward: a trajectory
        # that kept no states says nothing about the candidate, so it is
        # excluded -- and when a whole side is like that, fail open as before:
        # no evidence is not evidence of a bad reward.
        succ_scored = [e for e in succ_scored if e[1] != "no_data"]
        fail_scored = [e for e in fail_scored if e[1] != "no_data"]
        if not succ_scored or not fail_scored:
            ctx.event("screen_vacuous_pass", screen="tpe", cand_id=c.cand_id,
                      note="stored trajectories carry no re-scorable states")
            continue

        n_succ_err = sum(1 for e in succ_scored if e[1] == "reward_error")
        n_fail_err = sum(1 for e in fail_scored if e[1] == "reward_error")
        # A SUCCESS whose re-score raised or went
        # non-finite counts as INCORRECTLY ordered -- scored below every
        # failure, which implements it uniformly under all three rules with no
        # change to `_tpe_verdict`. (Release side: a raise crashes the step and
        # a NaN success fails the strict `>` at utils.py:298-300. One documented
        # divergence, in the other direction: a +inf success passes the
        # release's `>` and is a reward_error here.)
        succ_vals = [e[2] if e[1] == "ok" else float("-inf") for e in succ_scored]
        # An errored FAILURE re-score is excluded from max(fail) but counted in
        # the verdict: the release's behaviour there is a crash or a
        # position-dependent NaN inside max(), so there is no faithful value to
        # fold in.
        fail_vals = [e[2] for e in fail_scored if e[1] == "ok"]

        if not fail_vals:
            # Two-sided store, but reward errors left no usable failure score:
            # a fail-shaped REFUSAL, never a vacuous pass. score=None
            # stays distinct from 0.0 -- no ordering statistic was computed.
            passed, score = False, None
            detail = (f"re-scoring raised or returned non-finite on all "
                      f"{len(fail_scored)} failure trajectories; no ordering "
                      f"statistic is computable")
        else:
            passed, score, detail = _tpe_verdict(rule, thr, succ_vals, fail_vals)
        if passed:
            ctx.event("screen_pass", screen="tpe", cand_id=c.cand_id,
                      rule=rule, score=score, detail=detail,
                      n_success_errors=n_succ_err, n_failure_errors=n_fail_err)
            continue

        # The failing verdict carries the screen's own
        # re-scored evidence, so §6's preference report selects and quotes THIS
        # candidate's numbers over the store the screen judged (utils.py:262-326)
        # -- never `Trajectory.mean_per_step_return`, which is the reward
        # recorded at COLLECTION time, i.e. a previous candidate's. Errors are
        # null, never -inf: the payload must stay JSON-native for meta.json and
        # the strict checkpoint codec.
        ok_succ = [e for e in succ_scored if e[1] == "ok"]
        ok_fail = [e for e in fail_scored if e[1] == "ok"]
        example = None
        if ok_succ and ok_fail:
            worst_s = min(ok_succ, key=lambda e: e[2])   # utils.py:316 success_min
            best_f = max(ok_fail, key=lambda e: e[2])    # utils.py:295 failure_max
            example = {"success": _tpe_example(store[worst_s[0]], worst_s),
                       "failure": _tpe_example(store[best_f[0]], best_f)}
        # Adjacent counts, DIFFERENT populations, on purpose: `n_success`
        # counts every success in the statistic -- an errored one enters it at
        # -inf, i.e. incorrectly ordered -- while `n_failure` counts only
        # usable failures, because an errored failure has no faithful value to
        # fold into max(fail). The two error counters beside them are the
        # disambiguation; read all four before comparing the pair.
        verdict = {"passed": False, "rule": rule, "score": score, "threshold": thr,
                   "detail": detail,
                   "n_success": len(succ_vals), "n_failure": len(fail_vals),
                   "n_success_errors": n_succ_err, "n_failure_errors": n_fail_err,
                   "on_failure": on_failure,
                   "success_scores": [e[2] if e[1] == "ok" else None for e in succ_scored],
                   "failure_scores": [e[2] if e[1] == "ok" else None for e in fail_scored],
                   "example_pair": example}
        c.meta["tpe"] = verdict
        reason = (f"tpe[{rule}]: {detail}" if score is None
                  else f"tpe[{rule}]: {detail} < threshold {thr:.4f}")
        if on_failure == "skip_training":
            _reject(c, reason,
                    screen="tpe", **{k: verdict[k] for k in ("rule", "score", "threshold")})
        else:
            # train_anyway: the verdict is recorded and routes feedback in §6,
            # but the candidate stays trainable. Mutating it is fine -- it did
            # not pass; the invariant only protects candidates the screen admits.
            # (§6's `append_on_pass` store rule keys on THIS record: a trained-
            # but-failing candidate must not extend the store.)
            c.verify_records.append({"phase": "screen", "screen": "tpe", "ok": False,
                                     "detail": detail, "trained_anyway": True})
            ctx.event("screen_verdict", screen="tpe", cand_id=c.cand_id,
                      rule=rule, score=score, passed=False, on_failure=on_failure,
                      n_success_errors=n_succ_err, n_failure_errors=n_fail_err)
    return candidates


def _tpe_example(traj: Trajectory, scored: Tuple[int, str, Optional[float], List[float]]
                 ) -> Dict[str, Any]:
    """One side of `meta['tpe']['example_pair']`: the re-scored per-step values
    plus the three aggregates the release quotes (return, length, average per
    step -- utils.py:323-326), all JSON-native floats/ints. Only status-`ok`
    entries reach this, so every value is finite by construction."""
    idx, _status, agg, per_step = scored
    n = len(per_step)
    return {"store_index": int(idx),
            "score": float(agg) if agg is not None else None,
            "per_step": [float(r) for r in per_step],
            "return": float(sum(per_step)),
            "mean_per_step": float(sum(per_step) / max(n, 1)),
            "length": int(getattr(traj, "length", 0) or n)}


# ==========================================================================
# screen:epic / screen:starc  --  unpublished
# ==========================================================================


def _pseudometric_screen(ctx: Any, state: Any, candidates: List[Candidate],
                         label: str, metric: Callable[[np.ndarray, np.ndarray], float]
                         ) -> List[Candidate]:
    """Shared body: canonicalise every candidate, rank by closeness to the
    reference, keep `verify.alignment_filter.keep_top_n`."""
    live = _live(candidates)
    if not live:
        return candidates

    ref_fn, err = reference_reward(ctx, state)
    if ref_fn is None:
        ctx.event("screen_inert", screen=label, reason=err)
        return candidates

    transitions = sample_transitions(ctx, CANON_SAMPLES)
    ref_profile = epic_canonical_profile(ctx, ref_fn, transitions)
    if ref_profile is None:
        ctx.event("screen_inert", screen=label,
                  reason="reference reward could not be canonicalised")
        return candidates

    fns = _compiled(ctx, live)
    scores: Dict[str, float] = {}
    for c in live:
        fn = fns.get(c.cand_id)
        if fn is None:
            continue
        prof = epic_canonical_profile(ctx, fn, transitions)
        if prof is None:
            ctx.event("screen_skip", screen=label, cand_id=c.cand_id,
                      reason="candidate reward could not be canonicalised")
            continue
        # Rank on similarity, so higher is better and _rank_keep needs no polarity flag.
        scores[c.cand_id] = -metric(ref_profile, prof)

    if not scores:
        ctx.event("screen_inert", screen=label, reason="no candidate could be canonicalised")
        return candidates
    return _rank_keep(ctx, candidates, scores, _keep_top_n(ctx), label)


@register("screen", "epic")
def screen_epic(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """EPIC pseudometric against `evaluate.similarity.reference`.

    ‡ Published by nobody as a screen: a cold-start-free prefilter that needs
    no labels at all, attacking the same cost term as TAC
    and TPE with neither of their blind spots -- no preference store to fill,
    no success flag to wait for, just transitions.

    Canonicalise both rewards (subtract the expected shaping term, which makes
    the score invariant to potential shaping), then Pearson-correlate;
    `sqrt((1 - rho) / 2)` is the distance. Ranked, keeping
    `verify.alignment_filter.keep_top_n` -- see :func:`_keep_top_n` for why the
    keep rule is a rank and not a threshold.

    The approximation, stated plainly: the expectations are Monte Carlo over
    :data:`CANON_SAMPLES` transitions drawn from `ctx.env`, and EPIC's
    guarantees are stated for exact expectations over a coverage distribution.
    The ordering it produces is sound where the sampler covers the state space
    and is worth no more than that sampler is.

    COLD START: **live from round 0** with `reference: gt_reward` -- but note
    that a ground-truth reference is exactly what `generate.context
    .strip_existing_reward` keeps away from the generator. Using it as a screen
    is legitimate (the search never sees it) and is the same posture as Eureka
    computing curve-Pearson against the GT reward for reporting.
    With `reference: previous_best` the screen is inert at iteration 1 instead.
    """
    return _pseudometric_screen(ctx, state, candidates, "epic", epic_distance)


@register("screen", "starc")
def screen_starc(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """STARC pseudometric against `evaluate.similarity.reference`.

    ‡ Unpublished as a screen, like `epic`. Same canonicalisation, then STARC's
    two extra steps: normalise the canonical profile to unit L2 norm, then
    metrise with half the Euclidean distance. The reason to prefer it over EPIC
    is that STARC's distance bounds worst-case regret, where EPIC's correlation
    form does not -- a tighter statement about what "close reward" buys you.

    Same rank keep rule, same sampling approximation, same cold-start
    behaviour as `screen:epic`.
    """
    return _pseudometric_screen(ctx, state, candidates, "starc", starc_distance)


# ==========================================================================
# screen:policy_rank_corr  --  unpublished
# ==========================================================================


@register("screen", "policy_rank_corr")
def screen_policy_rank_corr(ctx: Any, state: Any,
                            candidates: List[Candidate]) -> List[Candidate]:
    """Rank correlation between the policies the candidate and the reference
    reward induce.

    ‡ Unpublished. The motivation is that EPIC/STARC compare *rewards* while
    what actually matters is the *policies* they induce, and two rewards can be
    far apart as functions and induce the same behaviour.

    The honest approximation, named: a full version would train a policy under
    each reward and correlate their action rankings, which costs exactly what
    the screen is supposed to save. Instead we compare the ONE-STEP GREEDY
    policies -- for each sampled state, rank :data:`POLICY_ACTIONS_PER_STATE`
    sampled actions under each reward and take Spearman's rho, averaged over
    states. That is the induced policy at gamma = 0; it captures disagreement
    about immediate action preference and is blind to disagreement that only
    shows up over a horizon. Where the two rewards differ only in shaping, this
    screen will (correctly, for EPIC's purposes; arguably wrongly for this
    one's) call them different -- which is precisely why it is a different
    screen and not a variant of `epic`.

    Ranked keep, `verify.alignment_filter.keep_top_n`. COLD START: live from
    round 0 with `reference: gt_reward`; inert at iteration 1 with
    `previous_best`.
    """
    live = _live(candidates)
    if not live:
        return candidates

    ref_fn, err = reference_reward(ctx, state)
    if ref_fn is None:
        ctx.event("screen_inert", screen="policy_rank_corr", reason=err)
        return candidates

    n_states = max(CANON_SAMPLES // 8, 4)
    anchors = sample_transitions(ctx, n_states)
    action_pool = sample_transitions(ctx, POLICY_ACTIONS_PER_STATE)
    grids = [[Transition(tr.state, p.action, tr.next_state) for p in action_pool]
             for tr in anchors]

    def ranks_for(fn: Callable[..., Any]) -> Optional[List[np.ndarray]]:
        out: List[np.ndarray] = []
        for grid in grids:
            try:
                vals = np.array([reward_scalar(ctx, fn, t) for t in grid], dtype=float)
            except Exception:
                return None
            if not np.all(np.isfinite(vals)):
                return None
            out.append(np.argsort(np.argsort(vals)).astype(float))
        return out

    ref_ranks = ranks_for(ref_fn)
    if ref_ranks is None:
        ctx.event("screen_inert", screen="policy_rank_corr",
                  reason="reference reward could not be ranked")
        return candidates

    fns = _compiled(ctx, live)
    scores: Dict[str, float] = {}
    for c in live:
        fn = fns.get(c.cand_id)
        if fn is None:
            continue
        cand_ranks = ranks_for(fn)
        if cand_ranks is None:
            ctx.event("screen_skip", screen="policy_rank_corr", cand_id=c.cand_id,
                      reason="candidate reward could not be ranked")
            continue
        rhos = [_pearson(a, b) for a, b in zip(ref_ranks, cand_ranks)]
        scores[c.cand_id] = float(np.mean(rhos)) if rhos else 0.0

    if not scores:
        ctx.event("screen_inert", screen="policy_rank_corr",
                  reason="no candidate could be ranked")
        return candidates
    return _rank_keep(ctx, candidates, scores, _keep_top_n(ctx), "policy_rank_corr")


# --------------------------------------------------------------------------
# The demonstration verifier -- `verify.quality_screen: demo_margin`
# --------------------------------------------------------------------------
#
# CARD's TPE with a trajectory store whose ORDER is known by construction: a
# graded set of policies (the task's solution policy from `policies/`, that
# policy under action noise, uniform-random) is rolled out and each rollout is
# scored by the candidate's OWN reward, before any training. `bird/demos.py`
# builds the set and states what it may read (a demonstration's rollouts, never
# task_metric / success / reference_reward). BIRD's own screen; no paper pin.


def _demo_keep_count(ctx: Any, state: Any, passers: Sequence[Candidate],
                     scores: Dict[str, float]) -> Tuple[int, str]:
    """How many sign-test passers go on to training this round, and why.

    `keep: all` is a pure filter. `fraction` / `top_k` are fixed. `adaptive` is
    the bandwidth control: k starts at `cap` and only ever SHRINKS -- when the
    judge's best has failed to beat the incumbent for `patience` completed
    rounds (the `term_fitness_plateau` fold, on `state.fitness_history`), or
    when the passers' scores cluster (spread below `min_spread`, i.e. there is
    nothing left for the judge to discriminate). Monotone across rounds via
    `ctx.counters`, floor `floor`. Every input is deterministic given the run:
    the round's candidates, the judge's history, and the config.
    """
    mode = str(ctx.cfg.get("verify.demo_screen.keep", "all") or "all")
    n = len(passers)
    if mode == "all":
        return n, "all"
    if mode == "fraction":
        frac = float(ctx.cfg.get("verify.demo_screen.keep_fraction", 0.75) or 0.75)
        return max(1, min(n, math.ceil(frac * n))), f"fraction={frac:g}"
    if mode == "top_k":
        return max(1, min(n, int(ctx.cfg.get("verify.demo_screen.keep_top_k", 4) or 4))), "top_k"
    # adaptive
    cap = int(ctx.cfg.get("verify.demo_screen.keep_cfg.cap", 8) or 8)
    floor = int(ctx.cfg.get("verify.demo_screen.keep_cfg.floor", 2) or 2)
    patience = int(ctx.cfg.get("verify.demo_screen.keep_cfg.patience", 2) or 2)
    decay = float(ctx.cfg.get("verify.demo_screen.keep_cfg.decay", 0.5) or 0.5)
    min_spread = float(ctx.cfg.get("verify.demo_screen.keep_cfg.min_spread", 0.05) or 0.0)

    k = int(ctx.counters.get("demo_screen_k", cap))
    reasons: List[str] = []
    # (i) the judge's best has stalled -- the same fold as loop.termination:
    #     fitness_plateau, but the answer is "narrower", never "stop".
    hist = [f for f in (getattr(state, "fitness_history", None) or []) if f is not None]
    best, stall, shrinks = float("-inf"), 0, 0
    for f in hist:
        if f > best:
            best, stall = f, 0
        else:
            stall += 1
            if stall >= patience:
                shrinks, stall = shrinks + 1, 0
    already = int(ctx.counters.get("demo_screen_stall_shrinks", 0))
    if shrinks > already:
        k = max(floor, math.ceil(k * decay))
        reasons.append(f"judge stalled {patience} round(s)")
        ctx.counters["demo_screen_stall_shrinks"] = shrinks
    # (ii) the passers' scores cluster: bandwidth buys nothing to choose between.
    vals = [scores[c.cand_id] for c in passers]
    spread = float(np.std(vals)) if len(vals) > 1 else 0.0
    if len(vals) > 1 and spread < min_spread:
        k = max(floor, math.ceil(k * decay))
        reasons.append(f"score spread {spread:.3f} < {min_spread:g}")
    k = max(floor, min(k, cap))
    ctx.counters["demo_screen_k"] = k  # monotone: the next round starts from here
    return max(1, min(n, k)), ("adaptive: " + "; ".join(reasons)) if reasons else f"adaptive: k={k}"


@register("screen", "demo_margin")
def screen_demo_margin(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """Roll out the graded policy set under each candidate's reward; keep the
    rewards that pay the ladder in the right order.

    Per candidate: `margin` = (expert return - random return) / (|expert| +
    |random|), in [-1, 1] and scale-free; `monotonicity` = Spearman between
    policy quality and return over the whole set (+1 = correct order).
    `score` = margin + monotonicity. Two rejections:

      sign test   `reject_nonpositive_margin`: margin <= 0 -- the reward pays a
                  random policy at least what it pays the solution. The one
                  ABSOLUTE test, because it is scale-free and wrong in every pool.
      rank        `_demo_keep_count` of the passers by `score`, ties on cand_id
                  (`_rank_keep`). Pool-relative by construction.

    Evidence travels on `candidate.meta["demo"]` (JSON-native): per policy the
    returns and a strided per-step reward/component series of episode 0, plus
    the three numbers above. `evaluate.artifacts: demo_reward_traces` renders
    it for the judge and for the reflecting LLM; `fitness_source: demo_margin`
    ranks on `score` when nothing is trained (the verifier-only mode). A
    passing candidate's meta is written too -- unlike TPE the evidence IS the
    artifact here, not a verdict.

    Cold start fails OPEN: an env with no registered solution passes everything
    and says so (`screen_inert`), the same way TPE treats an empty store.
    """
    from bird import demos

    live = _live(candidates)
    if not live:
        return candidates
    env = getattr(ctx, "env", None)
    env_id = str(ctx.cfg.get("problem.env_id") or "")
    names = list(ctx.cfg.get("verify.demo_screen.policies") or [])
    pols, why = demos.policy_set(env, env_id, names)
    if not pols:
        ctx.event("screen_inert", screen="demo_margin", reason=why)
        log.warning("demo_margin: %s -- passing every candidate", why)
        return candidates

    fns = _compiled(ctx, live)
    episodes = max(1, int(ctx.cfg.get("verify.demo_screen.episodes", 2) or 1))
    max_err = float(ctx.cfg.get("verify.demo_screen.max_reward_error_rate", 0.05) or 0.0)
    reject_sign = bool(ctx.cfg.get("verify.demo_screen.reject_nonpositive_margin", True))
    it = int(getattr(state, "iteration", 0) or 0)
    # Seeds are a function of (run seed, iteration, policy, episode) and of
    # nothing else, so the same candidate scores the same under `parallel` and
    # `sequential`; nothing here touches ctx.rng.
    base_seed = int(ctx.cfg.get("seed", 0) or 0) * 1_000_003 + it * 1_009

    scores: Dict[str, float] = {}
    total_steps = 0
    n_sign_rejected = 0
    for c in live:
        fn = fns.get(c.cand_id)
        if fn is None:
            continue
        errors = {"n": 0, "steps": 0}

        def on_error(exc: BaseException) -> float:  # noqa: ARG001
            errors["n"] += 1
            return 0.0

        per_pol: List[Dict[str, Any]] = []
        ladder: List[Tuple[int, float]] = []
        for j, pol in enumerate(pols):
            seeds = [base_seed + 97 * j + e for e in range(episodes)]
            trajs, steps = demos.rollouts_under(env, fn, pol, seeds, on_error)
            total_steps += steps
            errors["steps"] += steps
            rets = [float(t.mean_per_step_return) for t in trajs]
            mean_ret = float(np.mean(rets)) if rets else 0.0
            ladder.append((pol.quality, mean_ret))
            ep0 = trajs[0] if trajs else None
            per_pol.append({
                "name": pol.name, "quality": pol.quality, "episodes": len(trajs),
                "mean_per_step_return": mean_ret,
                "per_step_returns": rets,
                "lengths": [int(t.length) for t in trajs],
                "reward_series": demos.strided(list(ep0.rewards)) if ep0 is not None else [],
                "component_series": ({k: demos.strided(list(v))
                                      for k, v in (ep0.component_values or {}).items()}
                                     if ep0 is not None else {}),
            })
        err_rate = errors["n"] / max(1, errors["steps"])
        if err_rate > max_err:
            _reject(c, f"demo_margin: reward raised on {err_rate:.1%} of steps "
                       f"(> {max_err:.0%})", screen="demo_margin", error_rate=err_rate)
            c.meta["demo"] = {"policies": per_pol, "error_rate": err_rate, "kept": False}
            continue
        by_q = {q: r for q, r in ladder}
        r_best, r_worst = by_q[min(by_q)], by_q[max(by_q)]
        margin = (r_best - r_worst) / (abs(r_best) + abs(r_worst) + 1e-12)
        mono = demos.quality_ladder(ladder)
        score = float(margin + mono)
        c.meta["demo"] = {"policies": per_pol, "margin": float(margin),
                          "monotonicity": float(mono), "score": score,
                          "error_rate": err_rate, "kept": True,
                          # `passed_sign_test` is what `fitness_source: demo_margin`
                          # gates on. It is NOT `kept`: under keep=skip_all every
                          # passer is un-kept and un-trainable by design, so
                          # neither of those can tell a sign-test reject from a
                          # scored-not-trained passer.
                          "passed_sign_test": True}
        ctx.event("screen_demo", cand_id=c.cand_id, margin=float(margin),
                  monotonicity=float(mono), score=score, error_rate=err_rate,
                  returns={p["name"]: p["mean_per_step_return"] for p in per_pol})
        if reject_sign and margin <= 0.0:
            n_sign_rejected += 1
            c.meta["demo"]["kept"] = False
            c.meta["demo"]["passed_sign_test"] = False
            _reject(c, f"demo_margin: expert paid no more than random "
                       f"(margin {margin:+.3f}, monotonicity {mono:+.2f})",
                    screen="demo_margin", margin=float(margin), monotonicity=float(mono))
            continue
        scores[c.cand_id] = score

    if total_steps:
        ctx.budget.record_rollout_steps(total_steps)
    passers = [c for c in live if c.cand_id in scores]
    if not passers:
        ctx.event("screen_demo_round", n_live=len(live), n_sign_rejected=n_sign_rejected,
                  n_kept=0, keep_rule="none passed", env_steps=total_steps)
        return candidates
    if str(ctx.cfg.get("verify.demo_screen.keep", "all")) == "skip_all":
        # Verifier-only mode: the verifier IS the evaluation. Every passer is scored (its
        # meta carries the evidence and `fitness_source: demo_margin` ranks on
        # it) and NONE is trained -- CARD's skip made total, counted in
        # `budget.policy_trainings_skipped` by stage 3's `not trainable` branch.
        for c in passers:
            c.meta["demo"]["kept"] = False
            _reject(c, f"demo_margin: verifier-only round (keep=skip_all); scored "
                       f"{scores[c.cand_id]:+.3f}, not trained",
                    screen="demo_margin", score=scores[c.cand_id])
        ctx.event("screen_demo_round", n_live=len(live), n_sign_rejected=n_sign_rejected,
                  n_passers=len(passers), n_kept=0, keep_rule="skip_all",
                  env_steps=total_steps, policies=[p.name for p in pols])
        return candidates
    k, rule = _demo_keep_count(ctx, state, passers, scores)
    out = _rank_keep(ctx, passers, scores, k, "demo_margin")
    for c in passers:
        c.meta["demo"]["kept"] = not c.screened_out
    ctx.event("screen_demo_round", n_live=len(live), n_sign_rejected=n_sign_rejected,
              n_passers=len(passers), n_kept=sum(1 for c in passers if not c.screened_out),
              keep_rule=rule, k=k, env_steps=total_steps,
              policies=[p.name for p in pols])
    return candidates


# ==========================================================================
# screen:complexity_cap  --  a hard budget on the reward program (ours)
# ==========================================================================


@register("screen", "complexity_cap")
def screen_complexity_cap(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """Screen every candidate whose reward program exceeds a size budget.

    Published by nobody. Rewards from the hill-climb's configurations were
    observed to grow monotonically over iterations (more `if`
    blocks, more returned terms, more thresholds), with nothing in the loop
    pushing back. This is the CONSTRAINT form of that pressure -- a bound
    rather than a penalty -- and it costs no API call and no RL: it reads
    `Candidate.component_names` (the parsed reward-dict entries, §1
    `generate.output.format`) and the AST node count of the reward
    `FunctionDef` (`selection._ast_node_count`, LIMEN's descriptor, so the
    same quantity `select.objectives: [-reward_ast_nodes]` penalises).

    A candidate over either bound is moved into the `screened` population
    (`_reject`): still `valid` -- the code was fine -- but not `trainable`, and
    the verify record says which bound and by how much. A bound of 0 is off;
    `_check_coherence` refuses the screen with both off.
    """
    from .selection import _ast_node_count  # local: selection imports nothing from here

    max_terms = int(ctx.cfg.get("verify.complexity_cap.max_components") or 0)
    max_nodes = int(ctx.cfg.get("verify.complexity_cap.max_ast_nodes") or 0)
    n_screened = 0
    for c in _live(candidates):
        n_terms = len(c.component_names or [])
        n_nodes = int(_ast_node_count(c.reward_code))
        reasons = []
        if max_terms and n_terms > max_terms:
            reasons.append(f"{n_terms} reward components > max_components={max_terms}")
        if max_nodes and n_nodes > max_nodes:
            reasons.append(f"{n_nodes} reward AST nodes > max_ast_nodes={max_nodes}")
        if reasons:
            _reject(c, "complexity_cap: " + "; ".join(reasons),
                    check="complexity_cap", n_components=n_terms, ast_nodes=n_nodes,
                    max_components=max_terms, max_ast_nodes=max_nodes)
            n_screened += 1
    ctx.event("screen_complexity_cap", max_components=max_terms, max_ast_nodes=max_nodes,
              screened=n_screened, pool=len(candidates))
    return candidates
