"""Stage 5 -- Reward Selection (§5).

Four registry families live here, and they are the four questions §5 asks:

    select_rule   how the evidence collapses into a verdict
    tie_break     what happens when the evidence does not discriminate
    significance  whether the verdict survives contact with the noise
    allocation    how much training budget each candidate gets (called from §3)

§5 is where the literature is statistically sloppiest, and this file is written
to make that visible rather than to launder it:

  * `argmax_fitness` + `tie_break: first` reproduces `np.argmax`'s lowest-index
    accident *and records that it happened* (`Selection.tie_broken`,
    `.tied_ids`). A non-trivial fraction of multi-candidate rounds can be
    decided this way -- a number that only exists because the tie is counted,
    so it stays counted.
  * `significance: none` is every published method, and with
    `train.seeds_per_candidate: 1` it is the only *honest* setting: a t-test or
    a bootstrap over one observation per arm cannot separate anything. The
    tests here say so and decline, rather than manufacturing a p-value.
  * `allocation` is called from stage 3 **before** any training, so an adaptive
    allocator can only use prior-iteration evidence plus cheap static signals.
    Every allocator below documents exactly what it can and cannot see.

Two invariants hold across every rule:

  * a report with `fitness is None` never outranks one that has a fitness.
    `select.failure_value` (Eureka's -10000) is stage 4's sentinel and cannot
    provide this guarantee on its own -- a sentinel of -10000 outranks a
    genuine fitness of -50000 -- so the ordering here is structural, not
    numeric.
  * reports are never mutated. The single, deliberate exception is
    `map_elites_insert`, which writes the descriptors and frozen cell coords it
    computed onto the report (and folds them into `state.descriptor_stats`) so
    that stage 6 inserts against exactly the grid the verdict was decided on;
    the contract for that hand-off is documented on the rule.

Scope note (§5, `select.scope`): Eureka gates its *return* on a global best but
its *parent* on the round best. Stage 6 owns the parent; this file owns the
scope of the contest. `within_iteration` (the published default everywhere)
contests only this round's reports; `cumulative` contests every report seen so
far, so a previous round's program can win again and be re-promoted.

That re-promotion reaches the PARENT, not just the return: stage 6 sets
`state.iteration_best = selection.winner` (update.py), so under `cumulative`
that field holds the global best despite its name, and `parent_source:
iteration_best` silently becomes `global_best`. A config that pairs
`cumulative` with `parent_source: iteration_best` is therefore not Eureka.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import registry
from ..config import ConfigError
from ..context import Context
from ..state import RunState
from ..types import Candidate, CandidateReport, Selection
# The SAME predicate `run_preferences` races on -- one copy, so the BT contest
# and the race it aggregates cannot drift about who is a contender (no cycle:
# preferences imports evaluation, and nothing imports this module back).
from .preferences import _eligible

log = logging.getLogger("bird")

# --------------------------------------------------------------------------
# Constants that the schema has no key for.
#
# §5 declares `select.significance` but no alpha, no bootstrap count, no BT
# prior. Rather than invent config keys, the pins live here, named, so that a
# reader can see them and a future schema key knows where to take its default
# from.
# --------------------------------------------------------------------------

#: Two-sided-equivalent level for `t_test` / `bootstrap_ci`. Would be
#: `select.significance_alpha` if §5 had one.
ALPHA = 0.05
#: Bootstrap resamples. Drawn from `ctx.rng`, so runs stay reproducible.
N_BOOTSTRAP = 2000
#: Bradley-Terry MM iterations / convergence tolerance. ‡ -- GT names Bradley-Terry and is silent on the fit, and has no
#: released code. These are ours, and they are pins, not findings.
BT_MM_ITERS = 200
BT_MM_TOL = 1e-10
#: Symmetric pseudo-count against a virtual anchor of strength 1.0. Without it
#: an undefeated (or winless) item has no finite MLE. ‡: GT does not state one.
#: This is the prior of `_bt_strengths`, the FALLBACK fit: when the preferences
#: phase has already fitted this round (`preferences.aggregate_bradley_terry`,
#: alpha virtual comparisons split over every pair -- its own ‡), `rule_bradley_terry`
#: selects on that fit and never reaches this one.
BT_PRIOR = 0.5
#: Monte-Carlo draws for the probability-of-being-best term in `bayesian_
#: experimental_design`.
BED_MC_SAMPLES = 4096


# ==========================================================================
# Shared helpers
# ==========================================================================


def _cfg_int(ctx: Context, key: str, default: int) -> int:
    v = ctx.cfg.get(key, default)
    try:
        return int(v) if v else default
    except (TypeError, ValueError):
        return default


def _contest_pool(ctx: Context, state: RunState,
                  reports: List[CandidateReport]) -> List[CandidateReport]:
    """The population the rule actually contests (`select.scope`).

    `bird.py` extends `state.all_reports` with this round's reports *before*
    calling stage 5, so the cumulative pool is deduplicated by `cand_id` rather
    than concatenated -- otherwise this round would count twice.
    """
    if ctx.cfg.get("select.scope", "within_iteration") != "cumulative":
        return list(reports)
    seen: set = set()
    pool: List[CandidateReport] = []
    for r in list(state.all_reports) + list(reports):
        if r.cand_id in seen:
            continue
        seen.add(r.cand_id)
        pool.append(r)
    return pool


def _losers(reports: List[CandidateReport],
            winners: List[CandidateReport]) -> List[CandidateReport]:
    """Losers are always drawn from THIS round.

    Under `select.scope: cumulative` a winner may be an old report; stage 6
    must not re-apply `update.loser.action` to reports it already processed in
    an earlier iteration, so the loser list never reaches backwards.
    """
    win_ids = {w.cand_id for w in winners}
    return [r for r in reports if r.cand_id not in win_ids]


def _n_survivors(ctx: Context) -> int:
    """`select.n_survivors`. 1 in every published method; >1 is a real
    population and every rule below honours it for real."""
    return max(1, _cfg_int(ctx, "select.n_survivors", 1))


def _rank_and_cut(ctx: Context, state: RunState, items: List[CandidateReport],
                  values: Sequence[float], n: int
                  ) -> Tuple[List[CandidateReport], bool, List[str]]:
    """Order by `values` (higher wins), break exact ties, take the top `n`.

    Two properties are load-bearing:

    1. Python's sort is stable and stays stable under `reverse=True`, so equal
       values keep their original order -- which is precisely `np.argmax`'s
       lowest-index rule. `tie_break: first` therefore *is* numpy's accident,
       not an imitation of it.
    2. Grouping is by exact float equality, again matching `np.argmax`: near
       ties are not ties, they are just a close race that the scalar resolved.

    Returns `(winners, tie_broken, tied_ids)`. `tie_broken` is True only when a
    tie group straddled the survivor cutoff, i.e. when the tie-break rule --
    not the evidence -- decided who got in.
    """
    if not items:
        return [], False, []
    order = sorted(range(len(items)), key=lambda i: values[i], reverse=True)

    groups: List[List[int]] = []
    for i in order:
        if groups and values[groups[-1][0]] == values[i]:
            groups[-1].append(i)
        else:
            groups.append([i])

    tie_fn = registry.get("tie_break", ctx.cfg.get("select.tie_break", "first"))
    ordered: List[CandidateReport] = []
    tie_broken = False
    tied_ids: List[str] = []
    slots = n
    for g in groups:
        members = [items[i] for i in g]
        if len(members) > 1:
            members = list(tie_fn(ctx, state, members))
            if 0 < slots < len(members):
                # The cut falls inside this tie: the tie-break rule decided it.
                tie_broken = True
                tied_ids = [m.cand_id for m in members]
        ordered.extend(members)
        slots -= len(members)
    return ordered[:n], tie_broken, tied_ids


def _scored(pool: List[CandidateReport]) -> Tuple[List[CandidateReport], List[float]]:
    """Split off the reports that carry a fitness.

    Reports with `fitness is None` are dropped from the ranking entirely rather
    than mapped onto a sentinel: `select.failure_value` guarantees a failed
    candidate *loses to a normal one*, but it cannot guarantee it loses to an
    unusually bad one. Structural exclusion can.
    """
    items = [r for r in pool if r.fitness is not None]
    return items, [float(r.fitness) for r in items]


def _minmax(values: List[Optional[float]], missing: float = 0.0) -> List[float]:
    """Scale a mixed/None channel into [0, 1]; missing evidence scores `missing`.

    Used only by `weighted_evidence`, where channels with incomparable units
    (a task metric, a Pearson correlation, a BT strength summing to 1) have to
    be added together. Both the normalisation and the treatment of missing
    evidence are inventions -- no paper has run this rule, so there is nothing
    to copy. Missing scores the WORST end of the channel on the grounds that
    imputing the mean would reward producing no evidence at all: the minimum
    (0.0) under a positive weight, and the MAXIMUM (1.0) under a negative one
    -- the caller passes `missing=1.0` for a minimised channel. Scoring
    missing as 0.0 regardless of sign would, under `{n_components: -0.1}`,
    give a candidate with no recorded components (a scalar reward) zero
    penalty, i.e. reward it as the simplest in the pool.
    """
    present = [v for v in values if v is not None]
    if not present:
        return [missing] * len(values)
    lo, hi = min(present), max(present)
    if hi == lo:
        # one value of evidence: present scores the BEST end, missing the worst
        return [(1.0 - missing) if v is not None else missing for v in values]
    return [missing if v is None else (float(v) - lo) / (hi - lo) for v in values]


# -- cheap static signals ---------------------------------------------------


def _ast_node_count(code: str) -> float:
    """AST nodes of the REWARD FUNCTION (LIMEN's `reward_ast_node_count`).

    The count is of the reward `FunctionDef` alone, matching
    `database.py:106::_count_reward_complexity` ("Count AST nodes in the
    compute_reward function"), probed by the same names stage 3 compiles
    (`training._REWARD_NAMES`). Counting the WHOLE artifact -- imports,
    helpers, the co-designed observation -- measures LIMEN's five released
    winner FILES at 1468-4965 nodes against the release's function-only
    313-1842: a different quantity, not merely differently binned. Two
    deliberate deviations from the release, flagged as such: a program whose
    reward function cannot be found falls back to the
    whole-module count where `_count_reward_complexity` returns 0 (better a
    coarse complexity than a smallest-cell sentinel for code that did parse),
    and the probe matches any of `training._REWARD_NAMES` (8 names) rather
    than only the release's `compute_reward` -- internally consistent with
    what stage 3 actually compiles, which the release's fixed name is not.

    Unparseable code has no AST; it scores 0, which bins it into the smallest
    cell where it will lose to anything that actually ran. That is the intended
    outcome, not a silent failure -- stage 2 already recorded *why* it did not
    parse (the two failure populations stay distinguishable).
    """
    if not code:
        return 0.0
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return 0.0
    from .training import _REWARD_NAMES  # local: training imports no selection
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in _REWARD_NAMES:
            return float(sum(1 for _ in ast.walk(node)))
    return float(sum(1 for _ in ast.walk(tree)))


def _reward_subtree_bag(code: str) -> Dict[str, int]:
    """Multiset of structural subtrees of the reward function, as dump strings.

    The same locator `_ast_node_count` uses (the `FunctionDef` named in
    `training._REWARD_NAMES`, whole module if none), the same docstring strip
    `verification._normalised_ast_hash` applies, and `ast.dump` with attributes
    off so line numbers do not make every subtree unique. Every node
    contributes the dump of the subtree it roots, so two programs that differ
    in one leaf share every subtree except those on the path above it -- a
    change deep in one term costs little, a rewrite costs everything.
    Constants stay in (a weight change IS a different reward). Unparseable
    code has no bag.
    """
    if not code:
        return {}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return {}
    from .training import _REWARD_NAMES  # local: training imports no selection
    root: ast.AST = tree
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in _REWARD_NAMES:
            root = node
            break
    for node in ast.walk(root):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    bag: Dict[str, int] = {}
    for node in ast.walk(root):
        if isinstance(node, ast.expr_context):  # Load/Store: noise, never structure
            continue
        key = ast.dump(node, annotate_fields=True, include_attributes=False)
        bag[key] = bag.get(key, 0) + 1
    return bag


def _structural_similarity(a: str, b: str) -> float:
    """Weighted Jaccard of two programs' subtree bags, in [0, 1].

    1.0 is identical structure (formatting and docstrings aside), 0.0 is
    disjoint or unparseable. Symmetric and deterministic -- no hashing seed,
    no embedding -- which is what lets `train.init: warm_start_from_similar`
    pick the same donor under sequential and parallel scheduling.
    """
    A, B = _reward_subtree_bag(a), _reward_subtree_bag(b)
    if not A or not B:
        return 0.0
    inter = sum(min(n, B.get(k, 0)) for k, n in A.items())
    union = sum(A.values()) + sum(B.values()) - inter
    return float(inter) / float(union) if union else 0.0


def _literal_return_width(code: str) -> Optional[float]:
    """Width of the widest literal list/tuple returned by `code`.

    A deliberately shallow heuristic for `observation_dim` when the observation
    function was co-designed (`generate.co_design.observation_fn`): it reads
    `return [a, b, c]` and `return np.array([...])` and gives up on anything
    computed. `np.concatenate([a, b])` reports 2, not the true width -- a lower
    bound, and documented as such rather than dressed up as a measurement.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    widths: List[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        v: Any = node.value
        if isinstance(v, ast.Call) and v.args:
            v = v.args[0]  # np.array([...]) / np.concatenate([...]) / np.stack
        if isinstance(v, (ast.List, ast.Tuple)):
            widths.append(len(v.elts))
    if widths:
        return float(max(widths))
    # Fallback: distinct attribute/subscript reads, i.e. how many things the
    # observation touches. Correlates with width; is not width.
    reads = {ast.dump(n) for n in ast.walk(tree)
             if isinstance(n, (ast.Attribute, ast.Subscript))}
    return float(len(reads)) if reads else None


def _observation_dim(ctx: Context, report: CandidateReport) -> Optional[float]:
    """LIMEN's first archive descriptor.

    When `problem.search_space` is reward-only there is no co-designed
    observation function, so this is the environment's constant obs dimension
    and the archive collapses to a single row. That is a property of the config
    -- `limen_reward_only.yaml`'s header says so and calls the collapse expected
    -- and not a bug here.

    THE ORDER OF THE PROBES MATTERS. `meta["observation_dim"]` is checked
    FIRST because `training._install_observation` writes the MEASURED width
    there -- phi called on real states. The `_literal_return_width` branch below it is the
    fallback for a candidate that was never trained, and it says in its own
    docstring that it is a lower bound. So on a run where the observation reached
    a learner this axis is a measurement; on one where it did not, it is an
    estimate; and the archive is binned on whichever was available. That is worth
    knowing before comparing two archives.
    """
    for holder in (report.meta, report.candidate.meta):
        if isinstance(holder, dict) and holder.get("observation_dim") is not None:
            return float(holder["observation_dim"])
    if report.candidate.observation_code:
        w = _literal_return_width(report.candidate.observation_code)
        if w is not None:
            return w
    env = ctx.env
    for attr in ("obs_dim", "observation_dim", "n_obs"):
        v = getattr(env, attr, None)
        if isinstance(v, (int, float)):
            return float(v)
    space = getattr(env, "observation_space", None)
    shape = getattr(space, "shape", None)
    if shape:
        return float(int(np.prod(shape)))
    return None


# -- the evidence table (§4 sources, addressable by name) -------------------

#: Canonical metric names. `select.objectives` and `update.archive.descriptors`
#: are free-form lists in the schema, so this table is the only place that says
#: what may go in them. Descriptors are conventionally written
#: `observation_dim` / `reward_ast_node_count` (§6) and objectives
#: `-obs_dim` / `-reward_ast_nodes` (§5), so both spellings resolve here.
_ALIASES = {
    "obs_dim": "observation_dim",
    "observation_dimension": "observation_dim",
    "reward_ast_nodes": "reward_ast_node_count",
    "ast_nodes": "reward_ast_node_count",
    "code_chars": "reward_code_chars",
    "subtask_score_mean": "subtask_mean",
    "safety_mean": "safety",
    "fitness_std": "seed_std",
    "seed_variance": "seed_std",
    "bt": "bt_strength",
}


def _metric(name: str, ctx: Context, state: RunState, report: CandidateReport,
            extra: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[float]:
    """One named piece of §4 evidence, or None when the report does not have it."""
    key = _ALIASES.get(name, name)
    c, res = report.candidate, report.result

    if key == "fitness":
        return None if report.fitness is None else float(report.fitness)
    if key == "similarity":
        return None if report.similarity is None else float(report.similarity)
    if key == "subtask_mean":
        vals = list(report.subtask_scores.values())
        return float(np.mean(vals)) if vals else None
    if key == "safety":
        vals = list(report.safety.values())
        return float(np.mean(vals)) if vals else None
    if key == "bt_strength":
        table = (extra or {}).get("bt") or {}
        v = table.get(report.cand_id)
        return None if v is None else float(v)
    if key == "observation_dim":
        return _observation_dim(ctx, report)
    if key == "reward_ast_node_count":
        return _ast_node_count(c.reward_code)
    if key == "reward_code_chars":
        return float(len(c.reward_code or ""))
    if key == "n_components":
        return float(len(c.component_names)) if c.component_names else None
    if key == "seed_std":
        vals = report.per_seed_fitness
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    if key == "env_steps":
        return float(res.env_steps_used)

    raise ConfigError(
        f"unknown selection metric {name!r}; known names: "
        f"{sorted(set(list(_ALIASES) + _KNOWN_METRICS))}")


_KNOWN_METRICS = [
    "fitness", "similarity", "subtask_mean", "safety", "bt_strength",
    "observation_dim", "reward_ast_node_count", "reward_code_chars",
    "n_components", "seed_std", "env_steps",
]


def _parse_objective(entry: Any) -> Tuple[str, float]:
    """`"fitness"` -> (fitness, +1); `"-obs_dim"` -> (obs_dim, -1).

    A leading minus means minimise, exactly as §5 writes it:
    `objectives: [fitness, -obs_dim, -reward_ast_nodes]`.
    """
    if isinstance(entry, dict):
        if len(entry) != 1:
            raise ConfigError(f"select.objectives: {entry!r} must have exactly one key")
        name, weight = next(iter(entry.items()))
        return str(name).lstrip("-"), float(weight)
    text = str(entry).strip()
    sign = -1.0 if text.startswith("-") else 1.0
    return text.lstrip("+-"), sign


# -- the human hook ---------------------------------------------------------

_MISSING = object()


def _invoke(fn: Any, ctx: Context, state: RunState,
            reports: List[CandidateReport]) -> Any:
    """Call a hook whose signature we do not own.

    `ctx.human` is built by another component family (`phase:human_oracle`), so
    stage 5 cannot assume its arity. We inspect the signature and pass the
    argument tuple that fits, instead of catching `TypeError` -- which would
    silently swallow a genuine `TypeError` raised inside the oracle.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return _MISSING
    params = list(sig.parameters.values())
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return fn(ctx, state, reports)
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = sum(1 for p in positional if p.default is p.empty)
    for args in ((ctx, state, reports), (state, reports), (reports,), ()):
        if required <= len(args) <= len(positional):
            return fn(*args)
    return _MISSING


def _ask_human(ctx: Context, state: RunState,
               reports: List[CandidateReport]) -> Optional[List[CandidateReport]]:
    """Ask `ctx.human` to pick, and translate whatever it says into reports.

    Accepts a `CandidateReport`, a `cand_id`, an index, or a list of any of
    those. Returns None when no oracle is reachable -- the caller decides
    whether that is fatal (`rule: human`) or a no-op (`select.human_override`).
    """
    oracle = ctx.human
    if oracle is None:
        return None
    for meth in ("select", "choose", "pick", "rank", "ask", "query", "__call__"):
        fn = getattr(oracle, meth, None)
        if not callable(fn):
            continue
        try:
            out = _invoke(fn, ctx, state, reports)
        except Exception as exc:  # an offline run must not die on the oracle
            log.warning("human oracle .%s() failed (%s); trying the next hook", meth, exc)
            continue
        if out is _MISSING or out is None:
            continue
        picked = _as_reports(out, reports)
        if picked:
            # Counted only on a usable answer: a hook that returned prose was
            # a text-feedback hook, not a selection query, and inflating
            # `budget.human_queries` would corrupt the run's cost accounting.
            ctx.budget.record_human()
            return picked
    return None


def _as_reports(value: Any, reports: List[CandidateReport]) -> List[CandidateReport]:
    by_id = {r.cand_id: r for r in reports}
    items = value if isinstance(value, (list, tuple)) else [value]
    out: List[CandidateReport] = []
    for v in items:
        if isinstance(v, CandidateReport):
            out.append(v)
        elif isinstance(v, str) and v in by_id:
            out.append(by_id[v])
        elif isinstance(v, bool):
            continue  # a bool is not an index; `True` must not mean reports[1]
        elif isinstance(v, int) and 0 <= v < len(reports):
            out.append(reports[v])
    return out


def _finalise(ctx: Context, state: RunState, reports: List[CandidateReport],
              winners: List[CandidateReport], *, rule: str, contested: bool = True,
              tie_broken: bool = False, tied_ids: Sequence[str] = (),
              notes: str = "", allow_override: bool = True) -> Selection:
    """Build the `Selection` every rule returns.

    Single choke point so that `select.human_override` is honoured identically
    by every rule (§5 lists it as a rule-independent key) and so that the loser
    list is computed the same way everywhere.
    """
    if allow_override and ctx.cfg.get("select.human_override", False):
        picked = _ask_human(ctx, state, reports)
        if picked:
            notes = (notes + " | " if notes else "") + \
                f"human override: {[p.cand_id for p in picked]}"
            winners = picked
        else:
            notes = (notes + " | " if notes else "") + \
                "human override requested but no oracle answered"
    return Selection(
        winners=list(winners),
        losers=_losers(reports, winners),
        rule=rule,
        contested=contested,
        tie_broken=tie_broken,
        tied_ids=list(tied_ids),
        notes=notes,
    )


# ==========================================================================
# tie_break -- what happens when the evidence does not discriminate
#
# Signature: fn(ctx, state, tied) -> list[CandidateReport], preferred first.
# Called only on groups of EXACTLY equal score, so a reordering is the whole
# job. Tie-breaking is live, not hypothetical: exact ties are common.
# ==========================================================================


@registry.register("tie_break", "first")
def tie_first(ctx: Context, state: RunState,
              tied: List[CandidateReport]) -> List[CandidateReport]:
    """Lowest index wins -- i.e. `np.argmax`.

    Method: every published method, none of them on purpose. Keeping identity here is not laziness, it is
    the control condition: `first` must be *exactly* what argmax did, or the
    other three tie-breaks are not measured against the published behaviour.

    The caller sets `Selection.tie_broken` whenever this runs on a group that
    straddles the cutoff, so the artifact records the accident instead of
    presenting it as a decision.
    """
    return list(tied)


@registry.register("tie_break", "random")
def tie_random(ctx: Context, state: RunState,
               tied: List[CandidateReport]) -> List[CandidateReport]:
    """Uniform among the tied. Published nowhere; the unbiased alternative.

    Unbiased in the sense that matters: `first` correlates the winner with
    generation order (and therefore with whatever ordering the sampler
    imposed), which is a systematic bias; this one is only noise.
    """
    out = list(tied)
    ctx.rng.shuffle(out)
    return out


@registry.register("tie_break", "lowest_complexity")
def tie_lowest_complexity(ctx: Context, state: RunState,
                          tied: List[CandidateReport]) -> List[CandidateReport]:
    """Fewest reward-AST nodes wins. Published nowhere.

    An Occam prior on reward programs: among rewards that scored identically,
    prefer the one with less machinery to overfit the fitness. Ties within the
    tie fall back to original order (stable sort), i.e. to `first`.
    """
    return sorted(tied, key=lambda r: _ast_node_count(r.candidate.reward_code))


@registry.register("tie_break", "lowest_variance")
def tie_lowest_variance(ctx: Context, state: RunState,
                        tied: List[CandidateReport]) -> List[CandidateReport]:
    """Lowest across-seed spread wins. Published nowhere.

    Note the dependency: with `train.seeds_per_candidate: 1` -- the setting in
    almost every published method -- no candidate has a variance, every
    key is +inf, and the stable sort degrades this to `first`. The rule is only
    reachable in configs that pay for seeds; that is the honest failure mode
    rather than a fabricated tie-break on a single sample.
    """
    def key(r: CandidateReport) -> float:
        vals = r.per_seed_fitness
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else float("inf")
    return sorted(tied, key=key)


# ==========================================================================
# select_rule -- fn(ctx, state, reports) -> Selection
# ==========================================================================


@registry.register("select_rule", "argmax_fitness")
def rule_argmax_fitness(ctx: Context, state: RunState,
                        reports: List[CandidateReport]) -> Selection:
    """Highest fitness wins.

    Method: Eureka (line 9), DrEureka, RDA, Text2Reward, LIMEN's within-cell
    comparison -- effectively the whole field wherever a scalar fitness exists.

    Three things this implementation refuses to hide:

    * **The tie.** Exact ties are grouped and handed to `select.tie_break`;
      when the tie straddles the survivor cutoff, `Selection.tie_broken` and
      `.tied_ids` record it. With the default `tie_break: first` the outcome is
      bit-identical to `np.argmax`, so switching the key is a clean ablation.
    * **The unscored.** A report with no fitness is excluded from the ranking
      rather than mapped to `select.failure_value`; see `_scored`.
    * **The scope.** `select.scope: cumulative` lets a previous round's report
      win this round. Eureka does not do this for its parent (only for its
      return, which is §6/`select.final_artifact`), so `within_iteration` stays
      the default and the difference is one key.
    """
    pool = _contest_pool(ctx, state, reports)
    items, values = _scored(pool)
    if not items:
        return _finalise(ctx, state, reports, [], rule="argmax_fitness",
                         notes="no candidate carried a fitness; nothing to argmax over")
    from . import update as _update  # no cycle: update imports no selection
    if not any(_update.measured_fitness(r) for r in items):
        # Every scored report is at `select.failure_value`. The sentinel's §5
        # contract is "loses every comparison but stays recorded"; with nothing
        # to lose to, tie-breaking among sentinels would elect a screened,
        # never-trained program as round winner, which
        # `require_improvement_over_incumbent` then has to roll back.
        # `run_iteration`'s all-failed guard cannot catch it:
        # that guard is the EXECUTION predicate (compiled, ran without a
        # traceback), and a screened candidate is a different population that
        # CARD's `rule: none` deliberately hands to §6 "valid or not". So the
        # rule that ranks by value is where "no measurement" means "no
        # argmax". Tested by FLAG (`update.measured_fitness`), never by
        # comparing against the sentinel value. A sentinel BESIDE a measurement
        # still loses by value exactly as before, RF-Agent's 0.0 included.
        return _finalise(ctx, state, reports, [], rule="argmax_fitness",
                         notes=f"no candidate carried a measured fitness ({len(items)} "
                               f"at select.failure_value); nothing to argmax over")
    winners, tie_broken, tied_ids = _rank_and_cut(
        ctx, state, items, values, _n_survivors(ctx))
    n_unscored = len(pool) - len(items)
    notes = f"argmax over {len(items)} scored candidate(s)"
    if n_unscored:
        notes += f"; {n_unscored} unscored candidate(s) excluded from the ranking"
    if tie_broken:
        notes += (f"; {len(tied_ids)}-way tie at the cutoff broken by "
                  f"'{ctx.cfg.get('select.tie_break', 'first')}'")
    return _finalise(ctx, state, reports, winners, rule="argmax_fitness",
                     tie_broken=tie_broken, tied_ids=tied_ids, notes=notes)


@registry.register("select_rule", "bradley_terry")
def rule_bradley_terry(ctx: Context, state: RunState,
                       reports: List[CandidateReport]) -> Selection:
    """Highest Bradley-Terry strength over the pairwise preferences wins.

    Method: Gran Turismo (GT). GT has no ground-truth fitness at all
    (`problem.fitness_access: none`); its ranking signal is VLM/human pairwise
    preferences aggregated by BT. §5 records that GT renormalises strengths per
    round and that they cross no rounds -- so `select.scope: within_iteration`
    filters `state.preferences` to this iteration, and `cumulative` pools every
    preference ever collected (a variant GT does not run).

    ‡ GT names Bradley-Terry and nothing else: the fit (MM iterations, the
    prior that keeps an undefeated item finite) is an implementation choice,
    pinned at the top of this file. An item nobody compared lands exactly on
    the anchor strength, which means it can outrank an item that was compared
    and lost. That is correct Bayesian behaviour and it is also a trap; the
    note on the Selection says how many pool members were unjudged.

    The contest is over the N RACED agents only -- `_eligible`, the same
    predicate `run_preferences` pools on. Alg. 1 l.11-15 has the preferences
    p_ij, the strengths b_1:N and the argmax all over the trained policies, so
    a TAC-screened, never-trained reward is structurally outside the contest
    and can never be R_best. Were the pool every report, the prior anchor
    would hand each screened candidate roughly the pool-average strength: it
    would outrank every raced agent with a below-average record, and in an
    exactly-tied round (a regular tournament maps to identical floats)
    `tie_break: first` could hand the WIN to a screened reward -- the failure
    sentinel that normally guarantees a screened candidate loses never applies
    here, because this rule ranks by strength, not fitness. A round in which nothing trained degrades to argmax-fitness
    over the full pool (the sentinel convention still yields a parent there),
    and the note says so.

    **One fit, not two.** Alg. 1 has ONE
    `b_1:N <- bradley({p_ij})` (tex:318) and ONE argmax (tex:320) that is at
    once the parent `R_best`, the human's subject `pi_best` (tex:322) and the
    source of `eta_best` (tex:324). The preferences phase fits that `b_1:N` and
    writes it to `meta["bt_strength"]`, where `evaluate.fitness.source:
    preference_bt` reads it as the fitness, `phases._human_targets` sorts on it
    and `final_retrain.selected_fitness` reports it. Fitting a SECOND
    Bradley-Terry here -- `_bt_strengths`, `BT_PRIOR` against a fixed anchor,
    against alpha virtual comparisons split over every pair there -- and
    selecting on that is not equivalent: over the 1024 all-pairs outcomes of a
    5-agent round the two argmaxes differ in 50, so the human would narrate
    one agent while another became the parent, and in 44 of them
    `_rank_and_cut` sees no tie because last-ulp noise in the refit splits a
    structurally symmetric pair that the recorded fit ties exactly
    (`tie_broken: false` while a rounding bit decides). `_recorded_strengths`
    says when the recorded fit is adopted; `_bt_strengths` remains the fit for
    a contest the phase did not rank -- another aggregator, a `cumulative`
    pool spanning rounds, a checkpoint without recorded strengths -- and the
    `select_bt` event's `fit` field says which ran.
    """
    pool = _contest_pool(ctx, state, reports)
    contenders = _eligible(pool)
    if not contenders:
        items, values = _scored(pool)
        if items:
            log.warning("bradley_terry: no trained agent this round; "
                        "falling back to fitness over the full pool")
            winners, tb, tied = _rank_and_cut(ctx, state, items, values, _n_survivors(ctx))
            return _finalise(ctx, state, reports, winners, rule="bradley_terry",
                             tie_broken=tb, tied_ids=tied,
                             notes="no trained agent this round; degraded to "
                                   "argmax_fitness over all reports (screened/failed only)")
        log.warning("bradley_terry: no trained agent and no fitness; no winner")
        return _finalise(ctx, state, reports, [], rule="bradley_terry",
                         notes="no trained agent and no fitness: BT has nothing to rank")
    n_excluded = len(pool) - len(contenders)
    prefs = _relevant_preferences(ctx, state, contenders)
    recorded = _recorded_strengths(ctx, contenders)
    if not prefs and recorded is None:
        items, values = _scored(contenders)
        if items:
            log.warning("bradley_terry: no preference data; falling back to fitness")
            winners, tb, tied = _rank_and_cut(ctx, state, items, values, _n_survivors(ctx))
            return _finalise(ctx, state, reports, winners, rule="bradley_terry",
                             tie_broken=tb, tied_ids=tied,
                             notes="no preferences available; degraded to argmax_fitness")
        log.warning("bradley_terry: no preference data and no fitness; no winner")
        return _finalise(ctx, state, reports, [], rule="bradley_terry",
                         notes="no preferences and no fitness: BT has nothing to rank")

    if recorded is not None:
        strengths, fit = recorded, "preferences"
        # The phase's own tally says who raced; `state.preferences` may be
        # empty when `store_dataset` is off, and the ranking stands regardless.
        judged = {r.cand_id for r in contenders
                  if (r.meta.get("pref_comparisons") or 0) > 0}
    else:
        strengths, fit = _bt_strengths([r.cand_id for r in contenders], prefs), "refit"
        judged = {p.left_id for p in prefs} | {p.right_id for p in prefs}
    values = [strengths[r.cand_id] for r in contenders]
    winners, tie_broken, tied_ids = _rank_and_cut(
        ctx, state, contenders, values, _n_survivors(ctx))
    ctx.event("select_bt", n_preferences=len(prefs), fit=fit,
              strengths={r.cand_id: round(strengths[r.cand_id], 6)
                         for r in contenders})
    n_unjudged = sum(1 for r in contenders if r.cand_id not in judged)
    notes = (f"BT over {len(prefs)} preference(s)"
             + (" (the preferences phase's fit, one b_1:N for parent and human alike)"
                if fit == "preferences" else
                " (refit here: no recorded Bradley-Terry strengths on the contenders)"))
    if n_excluded:
        notes += (f"; {n_excluded} untrained report(s) outside the contest "
                  f"(Alg. 1 ranks only the raced agents)")
    if n_unjudged:
        notes += f"; {n_unjudged} contender(s) unjudged and sitting on the prior anchor"
    return _finalise(ctx, state, reports, winners, rule="bradley_terry",
                     tie_broken=tie_broken, tied_ids=tied_ids, notes=notes)


def _relevant_preferences(ctx: Context, state: RunState,
                          pool: List[CandidateReport]) -> List[Any]:
    """Preferences that this round's BT fit is allowed to see."""
    ids = {r.cand_id for r in pool}
    prefs = [p for p in state.preferences if p.left_id in ids and p.right_id in ids]
    if ctx.cfg.get("select.scope", "within_iteration") == "cumulative":
        return prefs
    this_round = [p for p in prefs if p.iteration == state.iteration]
    # `Preference.iteration` defaults to -1; a producer that never stamps it
    # would otherwise silently yield an empty round. Fall back rather than
    # pretend the round had no comparisons.
    return this_round if this_round else prefs


def _recorded_strengths(ctx: Context, contenders: List[CandidateReport]
                        ) -> Optional[Dict[str, float]]:
    """The Bradley-Terry strengths the preferences phase fitted THIS contest,
    or None when the rule has to fit its own.

    Adopted only when it is one fit over exactly this contest: every contender
    carries a finite `meta["bt_strength"]` written by the `bradley_terry`
    aggregator (`preferences._write_back` stamps `pref_aggregator` beside it),
    and `select.scope` is `within_iteration`, where `_eligible(reports)` is the
    very pool the phase ranked. Three cases deliberately return None:

    * another aggregator (`elo`, `elo_raw`, `borda`, `copeland`) -- a different
      model, and this rule's name says which one it selects on;
    * `select.scope: cumulative` -- per-round strengths are renormalised per
      round and cross no round boundary (§5), so a pool spanning rounds cannot
      be ranked on them; the refit over the pooled preferences stays;
    * a missing or non-finite strength on any contender (a checkpoint without
      recorded strengths, a phase that did not run) -- half a recorded fit is
      not a fit.
    """
    if ctx.cfg.get("select.scope", "within_iteration") != "within_iteration":
        return None
    out: Dict[str, float] = {}
    for r in contenders:
        if r.meta.get("pref_aggregator") != "bradley_terry":
            return None
        try:
            v = float(r.meta.get("bt_strength"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v):
            return None
        out[r.cand_id] = v
    return out


def _bt_strengths(ids: List[str], prefs: List[Any]) -> Dict[str, float]:
    """Minorisation-maximisation fit of Bradley-Terry, normalised to sum 1.

    Update (Hunter 2004), with every item additionally given `BT_PRIOR` wins
    and `BT_PRIOR` losses against a virtual anchor of fixed strength 1:

        p_i <- (w_i + a) / ( sum_{j != i} n_ij / (p_i + p_j) + 2a / (p_i + 1) )

    The anchor is what makes the fit finite when an item wins (or loses) all of
    its comparisons -- the unregularised MLE diverges there. GT has no release
    and the paper is silent (‡), so the anchor is ours, and it differs from
    `preferences.aggregate_bradley_terry`'s prior.

    This is the FALLBACK fit (see `_recorded_strengths`): under GT's own config
    the preferences phase has already fitted the round and this function is not
    reached. It is still the `bt_strength` channel of `rule_weighted_evidence`.
    """
    index = {cid: k for k, cid in enumerate(ids)}
    n = len(ids)
    wins = np.zeros(n)
    pairs = np.zeros((n, n))
    for p in prefs:
        i, j = index[p.left_id], index[p.right_id]
        pairs[i, j] += 1.0
        pairs[j, i] += 1.0
        if p.label == 1:
            wins[i] += 1.0
        elif p.label == 0:
            wins[j] += 1.0
        else:
            # `evaluate.preferences.allow_ties`: GT sets it false deliberately,
            # but a tie label splits the win when a config turns it on.
            wins[i] += 0.5
            wins[j] += 0.5

    p = np.ones(n)
    for _ in range(BT_MM_ITERS):
        denom = np.zeros(n)
        for i in range(n):
            mask = pairs[i] > 0
            if mask.any():
                denom[i] = float(np.sum(pairs[i][mask] / (p[i] + p[mask])))
            denom[i] += 2.0 * BT_PRIOR / (p[i] + 1.0)
        new = (wins + BT_PRIOR) / np.maximum(denom, 1e-300)
        new = new / max(float(np.sum(new)), 1e-300) * n  # keep scale ~1 while iterating
        if float(np.max(np.abs(new - p))) < BT_MM_TOL:
            p = new
            break
        p = new
    p = p / max(float(np.sum(p)), 1e-300)  # GT: renormalised per round
    return {cid: float(p[index[cid]]) for cid in ids}


@registry.register("select_rule", "pareto")
def rule_pareto(ctx: Context, state: RunState,
                reports: List[CandidateReport]) -> Selection:
    """The non-dominated set over `select.objectives`.

    Method: nobody, yet. §5 gives the shape -- `[fitness, -obs_dim,
    -reward_ast_nodes]`, leading minus = minimise -- and LIMEN explicitly flags
    penalising observation dimensionality as untried future work. This rule is
    that future work, as a config.

    A report with `fitness is None` is excluded before the front is computed
    (`_scored`, the module invariant): a candidate cannot join the front by
    declining to produce evidence. Mapping the missing fitness to -inf on that
    axis only would not suffice: -inf on one axis does not dominate a report
    that is best on another, so an unjudged one-line reward would be
    non-dominated under `[fitness, -reward_ast_nodes]` and win by index under
    `tie_break: first`. A report missing any OTHER objective is
    worst on that axis and nothing more, which is all Pareto dominance can say
    about it.

    The front is genuinely unordered, so when it is larger than
    `select.n_survivors` the cut is made by `select.tie_break` and recorded as
    a tie (`tie_broken=True`, `tied_ids` = the whole front). With
    `n_survivors: 1` and `tie_break: first` that is, once again, lowest index.
    """
    everyone = _contest_pool(ctx, state, reports)
    pool, _ = _scored(everyone)
    n_unscored = len(everyone) - len(pool)
    if not pool:
        return _finalise(ctx, state, reports, [], rule="pareto",
                         notes=("no candidate carried a fitness; nothing to rank"
                                if everyone else "empty pool"))

    raw = ctx.cfg.get("select.objectives") or []
    if not raw:
        raw = ["fitness"]
        log.warning("select.rule=pareto with empty select.objectives; using [fitness]")
    objectives = [_parse_objective(o) for o in raw]
    labels = [("-" if s < 0 else "+") + n for n, s in objectives]

    # Maximise-transformed matrix; missing evidence -> -inf on that axis.
    cols: List[List[float]] = []
    for name, sign in objectives:
        vals = [_metric(name, ctx, state, r) for r in pool]
        cols.append([float("-inf") if v is None else sign * float(v) for v in vals])
    mat = np.asarray(cols, dtype=float).T  # (n_reports, n_objectives)

    front: List[CandidateReport] = []
    for i, r in enumerate(pool):
        dominated = False
        for j in range(len(pool)):
            if i == j:
                continue
            if np.all(mat[j] >= mat[i]) and np.any(mat[j] > mat[i]):
                dominated = True
                break
        if not dominated:
            front.append(r)

    ctx.event("select_pareto", objectives=labels,
              front=[r.cand_id for r in front], pool=len(pool), unscored=n_unscored)

    n = _n_survivors(ctx)
    tie_broken = len(front) > n
    tied_ids = [r.cand_id for r in front] if tie_broken else []
    if tie_broken:
        tie_fn = registry.get("tie_break", ctx.cfg.get("select.tie_break", "first"))
        front = list(tie_fn(ctx, state, front))
    notes = f"Pareto front {len(front)}/{len(pool)} over {labels}"
    if n_unscored:
        notes += f"; {n_unscored} unscored candidate(s) excluded from the front"
    if tie_broken:
        notes += (f"; front larger than n_survivors={n}, cut by "
                  f"'{ctx.cfg.get('select.tie_break', 'first')}'")
    return _finalise(ctx, state, reports, front[:n], rule="pareto",
                     tie_broken=tie_broken, tied_ids=tied_ids, notes=notes)


@registry.register("select_rule", "map_elites_insert")
def rule_map_elites_insert(ctx: Context, state: RunState,
                           reports: List[CandidateReport]) -> Selection:
    """Per-cell comparison at archive write time.

    Method: LIMEN (the field's one non-hillclimb topology, §6). There is no
    global contest here: a candidate competes only against whatever already
    occupies its behavioural cell IN ITS OWN ISLAND'S grid, so a mediocre
    program in an empty cell wins and a strong program in a crowded cell can
    lose. `select.scope` is therefore irrelevant to this rule -- the archive is
    cumulative by construction.

    Hand-off contract with stage 6 (which owns the archive and does the actual
    insert): this rule writes, per report,

        report.descriptors["<name>"]        the raw descriptor value
        report.descriptors["<name>__bin"]   its bin index, as a float
        report.meta["archive_island"]       int, the deme
        report.meta["archive_coords"]       tuple[int, ...], frozen cell coords
        report.meta["archive_cell_win"]     bool, our verdict for that cell

    so `update.insert_into_archive` CONSUMES exactly these -- one binning,
    computed here, keyed `(island,)+coords`. Binning twice (fixed ranges here,
    an adaptive re-bin in stage 6) would compute the win verdict against one
    grid and insert on another. Binning itself is `update.bin_report`, LIMEN's
    running-min/max scaler with frozen per-program coords -- the † against
    §4.2's fixed grid, why fixed ranges are not used, and the worked example
    all live on that function.

    Two deliberate convention notes. (1) This is the one place in stage 5 that
    writes to a report, and it writes only fields stage 4 does not own. It ALSO
    mutates `state.descriptor_stats`: binning IS the cell verdict, and a stage
    6 that binned instead could never supply the incumbent comparison this
    rule owes `Selection.winners`. (2) Reports are processed IN LIST ORDER
    (candidate index -- parallel training reassembles by index, so the order
    is deterministic), matching the release's sequential `add()`: two
    same-round reports landing in one cell contest each other through the
    provisional occupant, first-in on an exact tie, exactly `database.py`'s
    strict `>`.

    ELIGIBILITY: a report contests a cell only if
    its fitness is a MEASUREMENT -- trained cleanly, or carrying the screen's
    own short-run rate (`meta["fitness_from_screen"]`, `evaluate.
    screened_fitness: screen_measurement`). Tested via flags
    (`update.measured_fitness`), never by comparing the value against
    `select.failure_value`. An ineligible report never touches
    `state.descriptor_stats` (the release's crash-filter failures never reach
    `_bin_features`), never wins, and is marked `meta["archive_excluded"]` --
    which is what stops a -10000 sentinel from seeding an empty cell and
    parenting the rest of the run.
    """
    descriptors = list(ctx.cfg.get("update.archive.descriptors") or [])
    bins = list(ctx.cfg.get("update.archive.bins_per_descriptor") or [])
    if not descriptors:
        log.warning("select.rule=map_elites_insert with no update.archive.descriptors; "
                    "every candidate lands in the single cell ()")

    from . import update as _update  # no cycle: update imports no selection

    winners: List[CandidateReport] = []
    by_cell: Dict[Tuple[int, ...], List[CandidateReport]] = {}
    pending: Dict[Tuple[int, ...], float] = {}  # this round's provisional occupants
    n_excluded = 0
    for r in reports:
        # Raw descriptor values are recorded for every report -- they are
        # artifact material and cost an AST walk -- but only measured reports
        # are BINNED, because binning folds the value into the running stats.
        for name in descriptors:
            value = _metric(name, ctx, state, r)
            r.descriptors[name] = 0.0 if value is None else float(value)

        if not _update.measured_fitness(r) or r.fitness is None:
            r.meta["archive_cell_win"] = False
            r.meta["archive_excluded"] = (
                "no fitness measurement" if r.fitness is None else
                r.candidate.failure_kind or "unmeasured")
            n_excluded += 1
            continue

        island = _update._island_for(ctx, state, r)
        coords = _update.bin_report(ctx, state, r, descriptors, bins)
        for k, name in enumerate(descriptors):
            r.descriptors[f"{name}__bin"] = float(coords[k])
        r.meta["archive_island"] = island
        r.meta["archive_coords"] = coords

        key = (island,) + coords
        by_cell.setdefault(key, []).append(r)
        incumbent = pending.get(key, _archive_incumbent_fitness(state, key))
        won = incumbent is None or float(r.fitness) > incumbent
        r.meta["archive_cell_win"] = bool(won)
        r.meta["archive_cell_incumbent_fitness"] = incumbent
        if won:
            pending[key] = float(r.fitness)
            winners.append(r)

    ctx.event("select_map_elites",
              cells={",".join(map(str, k)): [r.cand_id for r in v]
                     for k, v in by_cell.items()},
              winners=[w.cand_id for w in winners],
              excluded=[r.cand_id for r in reports if r.meta.get("archive_excluded")],
              occupied_before=len(state.archive))
    notes = (f"map-elites: {len(by_cell)} cell(s) touched, {len(winners)} won "
             f"(archive held {len(state.archive)} cell(s) before this round)")
    if n_excluded:
        notes += (f"; {n_excluded} report(s) without a fitness measurement "
                  f"excluded from the archive contest")
    # n_survivors is deliberately ignored: the number of winners is a property
    # of the cell structure, not a knob. tie_broken stays False: the per-cell
    # contest is a strict >, so an exact tie keeps the incumbent -- there is no
    # tie group for `select.tie_break` to order.
    return _finalise(ctx, state, reports, winners, rule="map_elites_insert",
                     notes=notes)


def _archive_incumbent_fitness(state: RunState,
                               key: Tuple[int, ...]) -> Optional[float]:
    """Fitness stored at exactly `key = (island, *coords)`, or None when empty.

    An EXACT lookup, matching `database.py::add` (L263-270): a newcomer
    contests only the occupant of the same cell of the same island's grid.
    Matching any archive key whose coordinate TAIL equals the coords, across
    ALL islands, would either auto-pass (no key happens to tail-match) or
    contest an arbitrary cross-island incumbent.
    """
    cell = state.archive.get(key)
    if cell is None:
        return None
    f = getattr(cell, "fitness", None)
    if f is None or f == float("-inf"):
        return None
    return float(f)


@registry.register("select_rule", "weighted_evidence")
def rule_weighted_evidence(ctx: Context, state: RunState,
                           reports: List[CandidateReport]) -> Selection:
    """Weighted sum over the §4 evidence channels. Published by nobody.

    A weights dict over §4 sources -- hybrids no paper has run: fitness,
    similarity, subtask mean, BT strength, safety. The unpublished
    "preference-based fitness with per-component credit" is the flagship point
    it makes reachable (robust BT *selection*, targeted per-subtask revision).

    **Where the weights live.** §5 declares no `select.weights` key, and the
    schema is a frozen contract, so `select.objectives` doubles as the weight
    list for this rule -- it is the one free-form list §5 has. Accepted forms:

        objectives: [fitness, similarity]              # implicit weight 1.0
        objectives: ["-reward_ast_nodes"]              # leading minus = -1.0
        objectives: [{fitness: 1.0}, {bt_strength: 0.5}]

    Empty objectives degrade to `{fitness: 1.0}`, i.e. to `argmax_fitness`.

    **The invented part, stated plainly.** Channels have incomparable units, so
    each is min-max scaled across the pool before weighting (`_minmax`), and a
    report with no evidence on a channel scores that channel's WORST end (the
    minimum under a positive weight, the maximum under a negative one). Both
    choices change the ranking and neither is published, because the rule is
    not published. Anyone reporting a result from this rule owes the reader
    those two sentences.
    """
    pool = _contest_pool(ctx, state, reports)
    if not pool:
        return _finalise(ctx, state, reports, [], rule="weighted_evidence",
                         notes="empty pool")

    raw = ctx.cfg.get("select.objectives") or []
    weights = [_parse_objective(o) for o in raw] or [("fitness", 1.0)]

    extra: Dict[str, Dict[str, float]] = {}
    if any(_ALIASES.get(n, n) == "bt_strength" for n, _ in weights):
        prefs = _relevant_preferences(ctx, state, pool)
        # `_eligible`, the same predicate `rule_bradley_terry` fits over: a
        # screened, untrained report entering the fit would land at the prior's
        # pool-average anchor (see `rule_bradley_terry`). Ineligible
        # reports simply miss the table, and `_metric` scores a missing channel
        # at the pool minimum, this rule's documented no-evidence convention.
        extra["bt"] = (_bt_strengths([r.cand_id for r in _eligible(pool)], prefs)
                       if prefs else {})

    total = np.zeros(len(pool))
    empty_channels: List[str] = []
    for name, w in weights:
        vals = [_metric(name, ctx, state, r, extra) for r in pool]
        if all(v is None for v in vals):
            empty_channels.append(name)
        # Missing evidence scores the channel's WORST end: 0 when the channel is
        # maximised, 1 when it is minimised (negative weight) -- see `_minmax`.
        scaled = _minmax(vals, missing=1.0 if float(w) < 0 else 0.0)
        total += float(w) * np.asarray(scaled, dtype=float)

    winners, tie_broken, tied_ids = _rank_and_cut(
        ctx, state, pool, list(total), _n_survivors(ctx))
    ctx.event("select_weighted",
              weights={n: w for n, w in weights},
              scores={r.cand_id: round(float(total[i]), 6) for i, r in enumerate(pool)})
    notes = ("weighted evidence over " +
             ", ".join(f"{w:+g}*{n}" for n, w in weights) +
             " (each channel min-max scaled across the pool; missing = minimum)")
    if empty_channels:
        notes += f"; channels with no evidence at all: {empty_channels}"
    if tie_broken:
        notes += f"; tie at the cutoff broken by '{ctx.cfg.get('select.tie_break', 'first')}'"
    return _finalise(ctx, state, reports, winners, rule="weighted_evidence",
                     tie_broken=tie_broken, tied_ids=tied_ids, notes=notes)


@registry.register("select_rule", "random")
def rule_random(ctx: Context, state: RunState,
                reports: List[CandidateReport]) -> Selection:
    """Uniform among the candidates that ran. The control condition.

    Method: nobody, and that is the point -- "does the selection rule beat
    coin-flipping?" is the ablation that anchors every §5 number, and with
    `train.seeds_per_candidate: 1` it is not a rhetorical question.

    `tie_broken` stays False: nothing was tied. The randomness *is* the rule,
    and conflating it with `tie_break: random` would corrupt the tie statistic
    this file exists to keep honest.

    Draws from `ctx.rng`, so a run is reproducible from `seed` alone.
    """
    pool = _contest_pool(ctx, state, reports)
    eligible = [r for r in pool if r.candidate.trainable] or pool
    if not eligible:
        return _finalise(ctx, state, reports, [], rule="random", notes="empty pool")
    n = min(_n_survivors(ctx), len(eligible))
    winners = ctx.rng.sample(eligible, n)
    return _finalise(ctx, state, reports, winners, rule="random",
                     notes=f"uniform draw of {n} from {len(eligible)} eligible candidate(s)")


@registry.register("select_rule", "human")
def rule_human(ctx: Context, state: RunState,
               reports: List[CandidateReport]) -> Selection:
    """The human picks.

    Method: adjacent to Text2Reward-human and GT, neither of which is exactly
    this. T2R-human's human supplies the *evaluation signal* (§4,
    `evaluate.fitness.source: human_score`) and selection is still an argmax;
    GT's human is capped at one query per iteration about the *already
    selected* agent. Putting the human on the selection rule itself is a
    reachable point that nobody has published.

    `ctx.human` is built by `phase:human_oracle`, whose interface stage 5 does
    not own, so the call is signature-inspected (`_invoke`). When no oracle
    answers -- the default `evaluate.human.mode: none`, or a scripted oracle
    with nothing to say -- this degrades to argmax over fitness and says so in
    the notes rather than stalling the loop.
    """
    pool = _contest_pool(ctx, state, reports)
    picked = _ask_human(ctx, state, pool)
    if picked:
        n = _n_survivors(ctx)
        return _finalise(ctx, state, reports, picked[:n], rule="human",
                         notes=f"human selected {[p.cand_id for p in picked[:n]]}",
                         allow_override=False)
    log.warning("select.rule=human but no human oracle answered; using argmax_fitness")
    items, values = _scored(pool)
    if not items:
        return _finalise(ctx, state, reports, [], rule="human", allow_override=False,
                         notes="no human oracle and no fitness; no winner")
    winners, tb, tied = _rank_and_cut(ctx, state, items, values, _n_survivors(ctx))
    return _finalise(ctx, state, reports, winners, rule="human", tie_broken=tb,
                     tied_ids=tied, allow_override=False,
                     notes="no human oracle answered; degraded to argmax_fitness")


@registry.register("select_rule", "none")
def rule_none(ctx: Context, state: RunState,
              reports: List[CandidateReport]) -> Selection:
    """No contest exists.

    Method: CARD and LIMEN's generation step -- K = 1, so there is exactly one
    program and nothing to compare it to. CARD additionally computes no ranking
    scalar at all (§4: its curves feed only the prompt, and TPE's verdict is an
    admission gate, not a fitness).

    So this rule invents nothing. It returns the single report as the winner,
    valid or not, scored or not -- CARD's chain head is the last *generated*
    program, and two routes hand `select.final_artifact: chain_end` one that
    was never trained: a TPE rejection under `verify.tpe.on_failure:
    skip_training` (the head advances past the rejection), and the
    generate-only last iteration under `loop.termination: fixed_generations`
    (CARD's published shape, Alg. 1 l.17 + `Ensure R`),
    whose report has `fitness=None` and which ONLY this rule adopts (`_scored`
    drops None; §5). `TrainResult.skip_reason` and `failure_kind` tell the two
    apart. It also sets `contested=False`, which is what stops `bird.py` from
    running a significance test on a one-item field.

    A pool of more than one under this rule is a config smell: something asked
    for K > 1 candidates and then declined to rank them. We keep the first and
    warn, because inventing a ranking here would be exactly the failure this
    rule exists to avoid.
    """
    if len(reports) > 1:
        log.warning("select.rule=none with %d candidates; keeping the first and "
                    "ranking nothing", len(reports))
    winners = reports[:1]
    notes = "no contest (K=1)" if len(reports) <= 1 else \
        f"no contest declared but {len(reports)} candidates present; kept the first"
    return _finalise(ctx, state, reports, winners, rule="none",
                     contested=False, notes=notes)


# ==========================================================================
# significance -- fn(ctx, state, selection, reports) -> Selection
#
# Runs only when the rule declared a contest and produced a winner. §5: `none`
# in every published method, and with one seed per candidate an argmax over 16
# is selecting substantially on training noise.
# ==========================================================================


@registry.register("significance", "none")
def sig_none(ctx: Context, state: RunState, selection: Selection,
             reports: List[CandidateReport]) -> Selection:
    """No test. Identity.

    Method: every published method (§5). Worth stating why this is not merely
    laziness on their part: with `train.seeds_per_candidate: 1` there is no
    test that *could* run, so `none` is the only setting consistent with the
    data those runs collected. The fix is upstream -- pay for seeds -- and this
    key only becomes meaningful once you have.
    """
    return selection


def _samples(report: Optional[CandidateReport]) -> List[float]:
    """Per-seed fitness for one report; a single point when that is all there is."""
    if report is None:
        return []
    if report.per_seed_fitness:
        return [float(v) for v in report.per_seed_fitness]
    return [] if report.fitness is None else [float(report.fitness)]


def _reference(ctx: Context, state: RunState, selection: Selection,
               reports: List[CandidateReport]) -> Tuple[Optional[CandidateReport], str]:
    """What the winner is tested against: the incumbent, else the runner-up."""
    winner = selection.winner
    if state.best is not None and winner is not None and state.best.cand_id != winner.cand_id:
        return state.best, "incumbent"
    if state.best is not None and winner is not None and state.best.cand_id == winner.cand_id:
        return None, "winner is the incumbent"
    pool = _contest_pool(ctx, state, reports)
    rivals = [r for r in pool
              if winner is None or r.cand_id != winner.cand_id]
    scored, values = _scored(rivals)
    if not scored:
        return None, "no reference available"
    runner_up, _, _ = _rank_and_cut(ctx, state, scored, values, 1)
    return runner_up[0], "runner-up"


def _defer_to_incumbent(ctx: Context, state: RunState, selection: Selection,
                        reports: List[CandidateReport], why: str) -> Selection:
    """"No significant difference" resolves in favour of the incumbent.

    Prefer-the-incumbent is the conservative reading and the one that actually
    guards against selection-on-noise: promoting a challenger you cannot
    distinguish from the current best is how a hill-climb random-walks. When
    there is no incumbent yet (iteration 0) there is nothing to prefer, so the
    winner stands and the notes record that it was untested.
    """
    if state.best is None:
        return replace(selection, notes=_append_note(
            selection.notes, f"{why}; no incumbent to fall back on, winner stands"))
    winners = [state.best]
    return replace(selection, winners=winners, losers=_losers(reports, winners),
                   notes=_append_note(selection.notes,
                                      f"{why}; incumbent {state.best.cand_id} retained"))


def _append_note(existing: str, extra: str) -> str:
    return f"{existing} | {extra}" if existing else extra


@registry.register("significance", "t_test")
def sig_t_test(ctx: Context, state: RunState, selection: Selection,
               reports: List[CandidateReport]) -> Selection:
    """Welch's one-sided t-test: is the winner really better than the incumbent?

    Method: nobody (§5: `significance: none` everywhere). Cheap to fix, and
    probably worth an ablation on its own.

    **With `train.seeds_per_candidate: 1` this test cannot separate anything.**
    One observation per arm gives no variance estimate and no degrees of
    freedom, so there is no p-value to compute. In that case we do not
    manufacture one: we declare no significant winner and defer to the
    incumbent, which means a config that asks for significance while paying for
    a single seed will promote once (iteration 0, no incumbent) and then freeze.
    That is a loud, correct signal about the config, not a bug here.

    `ALPHA` is a module constant because §5 declares no key for it.

    The t-distribution tail is computed from a regularised incomplete beta
    implemented below rather than from scipy, so the p-value is bit-identical
    on any machine and the repo keeps its stdlib+numpy dependency floor.
    """
    winner = selection.winner
    ref, kind = _reference(ctx, state, selection, reports)
    a, b = _samples(winner), _samples(ref)
    if ref is None:
        return replace(selection, notes=_append_note(selection.notes,
                                                     f"t_test skipped: {kind}"))
    if len(a) < 2 or len(b) < 2:
        return _defer_to_incumbent(
            ctx, state, selection, reports,
            f"t_test cannot run on {len(a)} vs {len(b)} seed(s) "
            f"(needs >= 2 per arm; see train.seeds_per_candidate)")

    xa, xb = np.asarray(a), np.asarray(b)
    va, vb = float(np.var(xa, ddof=1)), float(np.var(xb, ddof=1))
    se = math.sqrt(va / len(xa) + vb / len(xb))
    diff = float(np.mean(xa) - np.mean(xb))
    if se == 0.0:
        # Zero observed variance on both arms. Not evidence of precision -- it
        # is what a handful of identical seeds looks like -- so we resolve it
        # by the means alone and say so.
        significant, p = diff > 0.0, 0.0 if diff > 0 else 1.0
        detail = "zero observed variance; decided on means"
    else:
        df = (va / len(xa) + vb / len(xb)) ** 2 / (
            (va / len(xa)) ** 2 / (len(xa) - 1) + (vb / len(xb)) ** 2 / (len(xb) - 1))
        t = diff / se
        p = _t_sf(t, df)
        significant = p < ALPHA
        detail = f"t={t:.3f}, df={df:.1f}, p={p:.4g}, alpha={ALPHA}"

    if significant:
        return replace(selection, notes=_append_note(
            selection.notes,
            f"t_test vs {kind} {ref.cand_id}: significant ({detail})"))
    return _defer_to_incumbent(ctx, state, selection, reports,
                               f"t_test vs {kind} {ref.cand_id}: not significant ({detail})")


@registry.register("significance", "bootstrap_ci")
def sig_bootstrap_ci(ctx: Context, state: RunState, selection: Selection,
                     reports: List[CandidateReport]) -> Selection:
    """Percentile bootstrap CI on the mean difference; significant iff it excludes 0.

    Method: nobody (§5). Distribution-free alternative to `t_test`, which is
    the relevant property here -- per-seed RL returns are routinely bimodal
    (the seed either solved the task or did not), and a t-test on a bimodal
    two-sample comparison is exactly the wrong tool.

    **Same one-seed dead end as `t_test`, for a sharper reason.** With one
    observation per arm every resample is identical, so the CI has width zero
    and would report "significant" for *any* non-zero difference. That would be
    the worst possible failure -- a statistical test that rubber-stamps noise --
    so fewer than two observations per arm is refused outright and defers to the
    incumbent.

    Resamples are drawn from a numpy Generator seeded from `ctx.rng`, so the
    interval is reproducible from the run seed.
    """
    winner = selection.winner
    ref, kind = _reference(ctx, state, selection, reports)
    a, b = _samples(winner), _samples(ref)
    if ref is None:
        return replace(selection, notes=_append_note(selection.notes,
                                                     f"bootstrap_ci skipped: {kind}"))
    if len(a) < 2 or len(b) < 2:
        return _defer_to_incumbent(
            ctx, state, selection, reports,
            f"bootstrap_ci cannot run on {len(a)} vs {len(b)} seed(s) "
            f"(every resample would be identical; see train.seeds_per_candidate)")

    rng = np.random.default_rng(ctx.rng.getrandbits(64))
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    da = rng.choice(xa, size=(N_BOOTSTRAP, len(xa)), replace=True).mean(axis=1)
    db = rng.choice(xb, size=(N_BOOTSTRAP, len(xb)), replace=True).mean(axis=1)
    lo, hi = np.percentile(da - db, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    detail = f"{int((1 - ALPHA) * 100)}% CI on mean diff = [{lo:.4g}, {hi:.4g}]"

    if lo > 0.0:
        return replace(selection, notes=_append_note(
            selection.notes,
            f"bootstrap_ci vs {kind} {ref.cand_id}: significant ({detail})"))
    return _defer_to_incumbent(
        ctx, state, selection, reports,
        f"bootstrap_ci vs {kind} {ref.cand_id}: CI includes 0 ({detail})")


# -- Student's t tail, without scipy ---------------------------------------


def _betacf(a: float, b: float, x: float, itmax: int = 300,
            eps: float = 3e-16, fpmin: float = 1e-300) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_sf(t: float, df: float) -> float:
    """P(T > t) for T ~ Student-t(df). One-sided; H1 is `challenger > reference`.

    NaN in propagates to NaN out, and NaN fails the `p < ALPHA` test -- an
    undefined statistic must not read as a significant one.
    """
    if math.isnan(t) or math.isnan(df) or df <= 0:
        return float("nan")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    half = 0.5 * _betainc(0.5 * df, 0.5, df / (df + t * t))  # = P(T > |t|)
    return half if t > 0 else 1.0 - half


# ==========================================================================
# allocation -- fn(ctx, state, candidates) -> {cand_id: n_seeds}
#
# Called from STAGE 3, before a single environment step has been taken this
# iteration. The honest consequence: an adaptive allocator here can use
# (a) `state.all_reports` -- outcomes from PREVIOUS iterations, and
# (b) cheap static properties of the candidate program -- and nothing else.
# It cannot peek at this round's results, because they do not exist yet.
#
# One structural caveat applies to all three adaptive allocators. Under
# `generate.parent_source: none` + `sampling_mode: iid_parallel` (Eureka's 16
# i.i.d. samples) every candidate in a round shares the same lineage, so every
# prior below is identical and each allocator degenerates to `uniform`. That is
# not a defect in the allocator: it is a faithful report of the information
# available before training. Lineage-carrying configs (parent_source:
# global_best / archive_sample / top_k) are where these have anything to say.
# ==========================================================================


def _lineage_index(state: RunState) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """(fitness by cand_id, fitness by parent_id) over everything seen so far."""
    by_id: Dict[str, List[float]] = {}
    by_parent: Dict[str, List[float]] = {}
    for r in state.all_reports:
        if r.fitness is None:
            continue
        by_id.setdefault(r.cand_id, []).append(float(r.fitness))
        by_parent.setdefault(r.candidate.parent_id or "__root__", []).append(float(r.fitness))
    return by_id, by_parent


def _prior_sample(c: Candidate, by_id: Dict[str, List[float]],
                  by_parent: Dict[str, List[float]]) -> List[float]:
    """Everything already known about this candidate's lineage.

    Two sources: how the parent itself scored, and how the parent's other
    children scored. Neither says anything about *this* program -- it has not
    been trained -- which is the whole difficulty with pre-training allocation.
    """
    out: List[float] = []
    if c.parent_id:
        out.extend(by_id.get(c.parent_id, []))
    out.extend(by_parent.get(c.parent_id or "__root__", []))
    return out


def _largest_remainder(weights: np.ndarray, total: int, floor: int = 1) -> List[int]:
    """Split `total` units by `weights`, giving everyone at least `floor`.

    Largest-remainder (Hare) apportionment, so the allocation sums to exactly
    `total` and ties in the remainder resolve by index -- the same lowest-index
    convention `tie_break: first` makes explicit elsewhere in this file.
    """
    n = len(weights)
    if n == 0:
        return []
    base = [floor] * n
    spare = total - floor * n
    if spare <= 0:
        return base
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    if float(w.sum()) <= 0.0:
        w = np.ones(n)
    exact = w / w.sum() * spare
    whole = np.floor(exact).astype(int)
    rem = exact - whole
    left = spare - int(whole.sum())
    for i in sorted(range(n), key=lambda k: (-rem[k], k))[:max(0, left)]:
        whole[i] += 1
    return [base[i] + int(whole[i]) for i in range(n)]


@registry.register("allocation", "uniform")
def alloc_uniform(ctx: Context, state: RunState,
                  candidates: List[Candidate]) -> Dict[str, int]:
    """Every candidate gets `train.seeds_per_candidate`.

    Method: every published method (§5: "uniform everywhere published"). With
    `seeds_per_candidate: 1` -- also almost everywhere -- this is one
    training run per candidate and the entire selection rests on it.
    """
    s = max(1, _cfg_int(ctx, "train.seeds_per_candidate", 1))
    return {c.cand_id: s for c in candidates}


@registry.register("allocation", "successive_halving")
def alloc_successive_halving(ctx: Context, state: RunState,
                             candidates: List[Candidate]) -> Dict[str, int]:
    """Budget-preserving split: promising lineages get more seeds, the rest fewer.

    Method: nobody (§5 lists adaptive allocation as the open direction).

    **What this is not.** Real successive halving is a *rung* schedule -- train
    everything briefly, kill the bottom half, train the survivors longer. That
    loop lives inside training, and this repo has a key for it:
    `train.pruning: successive_halving` (§3), owned by stage 3. The
    `select.allocation` hook is called once, before any training, so the only
    half of SH expressible here is the pre-allocation: how many seeds each
    candidate starts with. Saying so is better than shipping something that
    claims to halve and does not.

    The split preserves the total budget `n_trainable * seeds_per_candidate`
    EXACTLY: the better-ranked half gets the seeds the worse-ranked half gives
    up, and when those do not divide evenly over the top half the remainder
    goes one seed each to the best-ranked. A floor division would drop that
    remainder, so every odd `n` where `n_lo * (s - s // 2)` is not a multiple
    of `n_hi` would train fewer seeds than `uniform` (3 candidates at 2 seeds:
    5 of 6; 7 at 2: 11 of 14) -- and budget parity is the premise of the
    `uniform` vs `successive_halving` ablation. Degenerate (and honestly so)
    when `seeds_per_candidate < 2` -- there
    is no budget to move -- or when no candidate has a lineage to rank on.
    """
    s = max(1, _cfg_int(ctx, "train.seeds_per_candidate", 1))
    plan = {c.cand_id: s for c in candidates}
    trainable = [c for c in candidates if c.trainable]
    if len(trainable) < 2 or s < 2:
        return plan

    by_id, by_parent = _lineage_index(state)
    priors = [_prior_sample(c, by_id, by_parent) for c in trainable]
    if not any(priors) or len({tuple(p) for p in priors}) == 1:
        log.debug("successive_halving: no discriminating prior; allocating uniformly")
        return plan

    scores = [float(np.mean(p)) if p else float("-inf") for p in priors]
    order = sorted(range(len(trainable)), key=lambda i: scores[i], reverse=True)
    n = len(trainable)
    n_hi = (n + 1) // 2
    n_lo = n - n_hi
    lo_seeds = max(1, s // 2)
    # Everything the bottom half gives up is spent by the top half, remainder
    # included: `hi_total / n_hi >= s >= lo_seeds` always, so no floor is needed.
    hi_total = n * s - n_lo * lo_seeds
    hi_seeds, extra = divmod(hi_total, n_hi)
    for rank, i in enumerate(order):
        if rank < n_hi:
            plan[trainable[i].cand_id] = hi_seeds + (1 if rank < extra else 0)
        else:
            plan[trainable[i].cand_id] = lo_seeds
    return plan


@registry.register("allocation", "bandit")
def alloc_bandit(ctx: Context, state: RunState,
                 candidates: List[Candidate]) -> Dict[str, int]:
    """UCB1 over lineages, spending the round's seed budget on the best arms.

    Method: nobody (§5).

    **What the arm is.** Candidates are freshly generated every iteration, so
    no candidate has ever been pulled and a per-candidate bandit is ill-posed.
    The only identity that survives across iterations is the *lineage* -- the
    parent a candidate was mutated from -- so that is the arm. The reframing is
    ours; no published method allocates adaptively at all, so there is nothing
    to be faithful to.

    Index is standard UCB1 on min-max-normalised prior fitness,
    `mu_a + sqrt(2 ln N / n_a)`, with an unpulled arm getting priority (its
    bonus is infinite). Seeds are apportioned proportional to the index with a
    floor of one seed each, so the round costs exactly what `uniform` costs.
    Degenerates to uniform when `seeds_per_candidate == 1` (no spare budget) or
    when every candidate shares one lineage.
    """
    s = max(1, _cfg_int(ctx, "train.seeds_per_candidate", 1))
    plan = {c.cand_id: s for c in candidates}
    trainable = [c for c in candidates if c.trainable]
    if len(trainable) < 2 or s < 2:
        return plan

    by_id, by_parent = _lineage_index(state)
    all_f = [f for vals in by_parent.values() for f in vals]
    if not all_f:
        return plan
    lo, hi = min(all_f), max(all_f)
    span = (hi - lo) or 1.0
    total_pulls = max(1, len(all_f))

    index: List[float] = []
    for c in trainable:
        sample = _prior_sample(c, by_id, by_parent)
        if not sample:
            index.append(float("inf"))  # UCB1: try the untried arm first
            continue
        mu = (float(np.mean(sample)) - lo) / span
        index.append(mu + math.sqrt(2.0 * math.log(total_pulls) / len(sample)))

    finite = [v for v in index if math.isfinite(v)]
    cap = (max(finite) + 1.0) if finite else 1.0
    weights = np.asarray([cap if not math.isfinite(v) else v for v in index], dtype=float)
    weights = weights - min(0.0, float(weights.min()))  # keep apportionment non-negative
    seeds = _largest_remainder(weights, len(trainable) * s, floor=1)
    for c, k in zip(trainable, seeds):
        plan[c.cand_id] = k
    return plan


@registry.register("allocation", "bayesian_experimental_design")
def alloc_bayesian_experimental_design(ctx: Context, state: RunState,
                                       candidates: List[Candidate]) -> Dict[str, int]:
    """Spend seeds where they most reduce uncertainty about which reward is best.

    Method: nobody, and no shipped config selects it, so no config exercises
    this allocator end to end. It stays because it is a named open research
    direction, paired with `evaluate.preferences.pairs: active`: the schema offers the value
    and the value has an implementation behind it, which is the property the
    registry exists to guarantee.

    **The approximation, stated up front.** Real BED is sequential: allocate,
    observe, update the posterior, allocate again. The stage contract calls
    allocation exactly once before any training, so the sequential half is
    unreachable from this key -- it would need a §3 knob that does not exist.
    What is implementable is the one-shot version: form a cheap prior over each
    candidate's fitness, estimate each candidate's probability of being the
    best, and put seeds where `P(best) x sigma` is largest -- i.e. on candidates
    that are both plausibly optimal and poorly understood. Measuring a
    candidate that is certainly mediocre buys no information about the argmax.

    The prior per candidate is its lineage mean (`_prior_sample`), with the
    spread inflated by a **structural novelty** term: how far the program's AST
    size sits from the round's median, scaled by the median absolute deviation.
    A program unlike anything measured so far is one the lineage prior does not
    actually cover. That term is our invention and it is the part that keeps
    this allocator from collapsing to uniform on the very first iteration,
    where no lineage exists at all.

    Monte-Carlo draws use a numpy Generator seeded from `ctx.rng`.
    """
    s = max(1, _cfg_int(ctx, "train.seeds_per_candidate", 1))
    plan = {c.cand_id: s for c in candidates}
    trainable = [c for c in candidates if c.trainable]
    if len(trainable) < 2 or s < 2:
        return plan

    by_id, by_parent = _lineage_index(state)
    all_f = [f for vals in by_parent.values() for f in vals]
    global_mu = float(np.mean(all_f)) if all_f else 0.0
    global_sd = float(np.std(all_f)) if len(all_f) > 1 else 1.0
    global_sd = global_sd or 1.0

    novelty = _structural_novelty(trainable)
    mus, sds = [], []
    for c, nov in zip(trainable, novelty):
        sample = _prior_sample(c, by_id, by_parent)
        if len(sample) >= 2:
            mu, sd = float(np.mean(sample)), float(np.std(sample, ddof=1))
        elif len(sample) == 1:
            mu, sd = float(sample[0]), global_sd
        else:
            mu, sd = global_mu, global_sd
        mus.append(mu)
        sds.append(max(sd, 1e-9) * (1.0 + nov))
    mu_v, sd_v = np.asarray(mus), np.asarray(sds)

    rng = np.random.default_rng(ctx.rng.getrandbits(64))
    draws = rng.normal(mu_v, sd_v, size=(BED_MC_SAMPLES, len(trainable)))
    wins = np.bincount(np.argmax(draws, axis=1), minlength=len(trainable))
    p_best = wins / float(BED_MC_SAMPLES)

    # Expected information about the argmax: plausibly optimal AND uncertain.
    eig = p_best * (sd_v / max(float(sd_v.max()), 1e-12))
    seeds = _largest_remainder(eig, len(trainable) * s, floor=1)
    for c, k in zip(trainable, seeds):
        plan[c.cand_id] = k
    ctx.event("select_allocation", rule="bayesian_experimental_design",
              plan={c.cand_id: plan[c.cand_id] for c in trainable},
              p_best={c.cand_id: round(float(p), 4) for c, p in zip(trainable, p_best)})
    return plan


def _structural_novelty(candidates: List[Candidate]) -> List[float]:
    """How unusual each program's AST size is within the round, in [0, 1].

    Robust scaling (median / MAD) rather than mean / std, so that one enormous
    generation does not flatten the term for everybody else. MAD collapses to
    zero when a majority of the round is byte-identical in size -- common when
    the generator is cold or the temperature is low -- so the std is the
    documented fallback; only when the whole round is the same size is every
    candidate genuinely equally novel, and the term correctly vanishes.
    """
    sizes = np.asarray([_ast_node_count(c.reward_code) for c in candidates], dtype=float)
    if len(sizes) < 2:
        return [0.0] * len(sizes)
    med = float(np.median(sizes))
    scale = 1.4826 * float(np.median(np.abs(sizes - med)))
    if scale <= 0.0:
        scale = float(np.std(sizes))
    if scale <= 0.0:
        return [0.0] * len(sizes)
    z = np.abs(sizes - med) / scale
    return [float(min(1.0, v / 3.0)) for v in z]
