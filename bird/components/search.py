"""Two search axes that no other stage owns, because no other stage found them.

This module is the residue of the framework's acceptance criterion: *if a
method cannot be expressed as a config, the schema is missing a knob*. Two
methods hit that wall, sixteen years apart, and neither gap fits inside an
existing component family -- so each got its own.

**A. `train.hyperparameter_search` (`hyperparameter_search`).** Singh, Lewis &
Barto (2009) do not score a candidate reward under *the* learner. They score it
under *its own best* learner: "we estimate the mean cumulative fitness of `r_A`
as the **maximum** estimate obtained ... over a coarse discretization of the
space of feasible `(alpha, epsilon)` pairs" (p. 2603). Two aggregations stack
there and BIRD only had the first: mean over sampled environments is
`evaluate.fitness.seed_aggregation`, but *max over a learner grid* had no key
at all. `evaluate.fitness.checkpoint_aggregation: max_over_checkpoints`
maximises over training *time*, which is a different quantity and was the
tempting wrong answer. Every LLM-era method
deliberately pins `train.hyperparameters` across candidates "to isolate reward
effects"; Singh deliberately does the opposite, to remove the confound where a
good reward loses because the shared learning rate happened to suit a different
one. That disagreement is a finding, and a finding you cannot state in the
config space is a finding the repo has thrown away.

**B. `verify.check_order` (`check_order`).** LIMEN's `crash_filter.py` is three
*stages*, not a list: `_check_stage0` (parse + required functions present),
`_check_stage1` (import/instantiate), `_check_stage2` (run on a dummy state),
short-circuiting at the first failure, with per-stage timeouts of 5 / 10 / 30 s
(`hard_rule_chain.yaml`; 5 / 10 / 60 in the Brax configs).
`verify.static_checks` / `verify.dynamic_checks` cover the same ground but are
flat unordered lists run to completion, so "cheapest first, stop early, budget
each stage separately" is not sayable with them. `verify.check_order` says it
in one key.

Why both live in one module: they are the two places where the *search over
candidates* is itself parameterised -- one orders the filter, one widens the
evaluation -- and neither belongs to the stage it decorates. `training.py` owns
what a single RL run is; this owns how many of them a fitness estimate costs.
`verification.py` owns what a check decides; this owns which check runs when.

Provenance markers as elsewhere: **dagger** paper and released code disagree,
**double-dagger** the paper is silent and the released code supplies the value.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from ..registry import register
from ..schema import SCHEMA
from ..types import TrainResult

log = logging.getLogger("bird.search")

__all__ = ["CheckPlan"]


# ==========================================================================
# `verify.check_order` -- the plan, not the execution
# ==========================================================================


@dataclass(frozen=True)
class CheckPlan:
    """What the validity phase should run, in what order, and how long to wait.

    A `check_order` component is a **planner, not an executor**: it never calls
    a check, never touches a candidate, and never reads anything but the config
    and the two declared lists. `bird/components/verification.py` executes the
    plan. The split is what keeps ordering a single testable config key instead
    of control flow smeared through the validity loop -- you can print the plan
    for a config without running anything, and swapping `declared` for `staged`
    is provably a reordering plus two flags rather than a different algorithm.

    Fields:

    steps
        `(kind, name)` pairs in execution order, `kind` in `{"static",
        "dynamic"}`. Names are registry names within the `static_check` /
        `dynamic_check` families, so `registry.get(kind + "_check", name)`
        resolves each one. Duplicates in the declared lists are preserved
        rather than collapsed: a config's list is the config's business.
    short_circuit
        Stop at the first failing step. `False` runs every check and collects
        every failure, which is what a method that reports all validity
        problems at once needs; `True` is LIMEN's staged filter, where the
        point of ordering cheap-first is never paying for stage 2.
    timeouts
        Seconds per **check name** (not per step index -- a name appears at
        most once in practice and per-name is what `verify.stage_timeouts_s`
        is keyed by). This mapping is *total over `steps`*: every name in
        `steps` has an entry, and a check the config did not mention gets
        `verify.timeout_s`. Consumers may therefore write `plan.timeouts[name]`
        with no fallback logic of their own, which is the whole reason it is
        filled in here.
    """

    steps: Tuple[Tuple[str, str], ...]
    short_circuit: bool
    timeouts: Dict[str, float] = field(default_factory=dict)

    def names(self) -> List[str]:
        """Just the check names, in order -- for logging and for artifacts."""
        return [name for _kind, name in self.steps]


#: Relative cost of each check, lowest first. Used only by `staged`.
#:
#: The numbers are ordinal, not measured seconds: what matters is that a check
#: which merely walks an AST runs before one that imports a module, and that
#: anything which *executes generated code* runs last. That is LIMEN's stage
#: boundary (`crash_filter.py`): stage 0 parses, stage 1 imports, stage 2
#: runs -- and the ordering exists so that a program which fails stage 0 never
#: costs a stage-2 timeout.
#:
#: Within the statics the ordering is deliberately the same as `_default.yaml`'s
#: declared list `[signature_parse, ast_syntax]`. That is worth stating: on the
#: default list `staged` reorders *nothing*, so flipping `verify.check_order`
#: alone is a clean single-key ablation of short-circuiting-and-timeouts rather
#: than a confounded change of both order and policy. `signature_parse` also
#: reports parse errors itself (see its docstring), so leading with it loses no
#: diagnostic when the program does not parse.
#:
#: A registered check absent from this table sorts *after* every known one,
#: keeping its declared position among the other unknowns. Guessing a cost for
#: a check this table has never seen would fabricate a value; putting it last is
#: the conservative reading -- an unknown check is assumed expensive.
STATIC_COST: Dict[str, int] = {
    "signature_parse": 0,  # parse + locate the entrypoint
    "ast_syntax": 1,  # parse only
    "import_allowlist": 2,  # parse + walk every node
    "forbidden_symbols": 3,  # parse + walk every node + match a symbol list
}

DYNAMIC_COST: Dict[str, int] = {
    "execution_smoke": 0,  # one call: does it run at all
    "output_shape": 1,  # reads the smoke output
    "output_dtype": 2,  # reads the smoke output
    "finite_values": 3,  # reads the smoke output
    "non_constant_reward": 4,  # needs several transitions
    "bounded_magnitude": 5,  # needs several transitions
    "unit_tests": 6,  # an env-supplied suite of unknown size
}


def _timeouts(ctx: Any, steps: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    """Fill `CheckPlan.timeouts` for every step: config override, else the default.

    `verify.stage_timeouts_s` naming a check that is in neither list is already
    a validation error (`config._check_coherence`), so anything unmatched here
    is a check the plan simply did not schedule and is dropped silently.
    """
    default = float(ctx.cfg.get("verify.timeout_s", 60) or 60)
    declared = ctx.cfg.get("verify.stage_timeouts_s") or {}
    out: Dict[str, float] = {}
    for _kind, name in steps:
        raw = declared.get(name, default)
        try:
            out[name] = float(raw)
        except (TypeError, ValueError):
            log.warning("verify.stage_timeouts_s[%s]=%r is not a number; using "
                        "verify.timeout_s=%s", name, raw, default)
            out[name] = default
    return out


@register("check_order", "declared")
def order_declared(ctx: Any, static_checks: List[str],
                   dynamic_checks: List[str]) -> CheckPlan:
    """Run the lists exactly as written: all statics, then all dynamics, no early exit.

    The default, and the behaviour of every published method except LIMEN.
    A method's list order is part of the method -- each paper gives its checks
    in a stated order -- so reordering them silently
    would be the same class of error as appending to a list down an `extends`
    chain. Statics before dynamics is not an ordering choice: a dynamic check
    is handed a compiled callable, which only exists once the statics have not
    rejected the program.

    `short_circuit=False`: every check runs and the caller sees every failure,
    matching `config.validate()`'s rule that a stage reports *all* problems at
    once rather than the first.

    Per-check timeouts are still honoured if the config sets them. They are
    empty by default, so this is byte-for-byte today's behaviour; but a config
    that sets `verify.stage_timeouts_s` and leaves the order `declared` means
    it, and ignoring the pin would make the key a lie.
    """
    steps = tuple([("static", str(n)) for n in (static_checks or [])]
                  + [("dynamic", str(n)) for n in (dynamic_checks or [])])
    return CheckPlan(steps=steps, short_circuit=False, timeouts=_timeouts(ctx, steps))


@register("check_order", "staged")
def order_staged(ctx: Any, static_checks: List[str],
                 dynamic_checks: List[str]) -> CheckPlan:
    """LIMEN: cheapest first, stop at the first failure, one timeout per stage.

    LIMEN's `crash_filter.py` runs `_check_stage0` (static) ->
    `_check_stage1` (import) -> `_check_stage2` (execute) and returns the
    moment one fails, with `stage0_timeout: 5`, `stage1_timeout: 10`,
    `stage2_timeout: 30` in `hard_rule_chain.yaml` (60 for stage 2 in the Brax
    configs -- so 30 is not a universal constant and belongs in the config, not
    here). BIRD has finer-grained checks than LIMEN has stages, so the mapping
    is by cost rather than one-to-one: `STATIC_COST` and `DYNAMIC_COST` above
    are the table, statics always before dynamics, and ties keep their declared
    order (a stable sort) so the key stays deterministic.

    `short_circuit=True` is the substance, not the ordering. Ordering only
    decides *which* check you stop at; short-circuiting is what makes a syntax
    error cost a parse instead of a timeout, and it is the reason LIMEN can
    afford a large candidate pool. Note the cost this buys: the artifact then
    records one failure per candidate rather than all of them, which is a real
    loss of diagnostic and the honest price of the setting.
    """
    def _sorted(names: Sequence[str], table: Mapping[str, int], kind: str
                ) -> List[Tuple[str, str]]:
        unknown = len(table)
        pairs = list(enumerate(str(n) for n in (names or [])))
        pairs.sort(key=lambda p: (table.get(p[1], unknown), p[0]))
        for _i, n in pairs:
            if n not in table:
                log.debug("check_order=staged: %s check %r has no entry in the cost "
                          "table, scheduling it last", kind, n)
        return [(kind, n) for _i, n in pairs]

    steps = tuple(_sorted(static_checks, STATIC_COST, "static")
                  + _sorted(dynamic_checks, DYNAMIC_COST, "dynamic"))
    return CheckPlan(steps=steps, short_circuit=True, timeouts=_timeouts(ctx, steps))


# ==========================================================================
# `train.hyperparameter_search` -- wrappers around the train backend
# ==========================================================================
#
# The contract, which `bird.py` stage 3 calls and which every component below
# implements:
#
#     hpsearch = registry.get("hyperparameter_search", cfg["train.hyperparameter_search"])
#     result   = hpsearch(ctx, state, candidate, backend, n_seeds=n)
#
# `backend` is whatever `registry.get("train_backend", cfg["train.backend"])`
# returned, with signature `backend(ctx, state, candidate, n_seeds=1, **kw) ->
# TrainResult`; `**kw` is whatever the round driver adds (`env_steps` and
# `resume_ref` for a rung of `train.pruning: successive_halving_pool`,
# `bird/components/rounds.py`) and is passed through untouched -- empty on
# every other config. A search component therefore *wraps* training and never
# reimplements it: swapping `train.backend` and swapping
# `train.hyperparameter_search` stay orthogonal, which is the only way a
# profile or a `-s train.backend=...` override can keep Singh's protocol while
# changing his learner.
#
# Cost. Each backend call already books its own trainings
# (`components/training.py` calls `ctx.budget.record_training()` once per seed),
# so these wrappers record nothing themselves -- and deliberately hide nothing
# either. A 15-point grid at 2 seeds is 30 entries in `budget.policy_trainings`
# and the report says 30, which is the entire point: Singh's protocol is
# `|grid|` times more expensive per candidate than Eureka's and the cost column
# is where that shows up. Hiding it would make `hyperparameter_search: grid`
# look free, which is the mirror image of the failure `record_skip()` exists to
# prevent for CARD.


_MISSING = object()


def _raw(cfg: Any) -> Dict[str, Any]:
    """The mutable dict behind a `Config`, for the overlay below."""
    data = getattr(cfg, "_data", None)
    if isinstance(data, dict):
        return data
    if isinstance(cfg, dict):
        return cfg
    raise TypeError(
        f"train.hyperparameter_search cannot overlay a config of type "
        f"{type(cfg).__name__}: it exposes no mutable mapping. A grid that "
        f"cannot reach the backend is a sweep that silently trains the same "
        f"point N times, so this refuses rather than pretending to search.")


def _get_path(raw: Dict[str, Any], path: str) -> Any:
    node: Any = raw
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _set_path(raw: Dict[str, Any], path: str, value: Any) -> None:
    node = raw
    parts = path.split(".")
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _del_path(raw: Dict[str, Any], path: str) -> None:
    node: Any = raw
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def _split_point(point: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Route each grid parameter to the config key it should overlay.

    Two destinations, distinguished by one rule that is easy to state and
    impossible to get wrong by accident:

    * a **dotted name that exists in the schema** (`train.env_steps`,
      `train.reward_norm`) overlays that key directly;
    * **anything else** (`alpha`, `epsilon`, `gamma`) becomes an entry in the
      `train.hyperparameters` dict, which is the declared channel for
      learner-level settings and is a schema leaf precisely so that arbitrary
      names can live in it.

    A bare name that happens to collide with a top-level key (`seed`) goes to
    `train.hyperparameters`, not to the key: sweeping a config key requires
    naming it in full, so the intent is always explicit in the YAML.

    The second destination is live: `components/training.py::_learner_kwargs`
    filters `train.hyperparameters` to the constructor parameters of whichever
    learner the environment forced, so a bare `alpha` or `epsilon` genuinely
    changes a tabular run and a bare `pop` or `sigma0` genuinely changes a CEM
    one. A name no learner accepts is WARNED about rather than dropped, because
    a swept key that reaches nothing turns `train.hyperparameter_search` into a
    max over a grid that did nothing, reported as if it had searched.
    """
    direct: Dict[str, Any] = {}
    hp: Dict[str, Any] = {}
    for name, value in point.items():
        if "." in name and name in SCHEMA:
            direct[name] = value
        else:
            hp[name] = value
    return direct, hp


@contextlib.contextmanager
def _overlaid(cfg: Any, point: Mapping[str, Any]) -> Iterator[None]:
    """Install one grid point on the live config, then put the config back.

    **This mutates the resolved config for the duration of one backend call.**
    It is the only honest channel available: the train backends read every
    setting they honour off `ctx.cfg` at call time (`cfg.get("train.env_steps")`
    and friends in `components/training.py::_run_backend`) and take no
    per-call overrides, and `train.hyperparameters` is likewise read from the
    config by whoever consumes it (`components/screens.py::_gamma` reads
    `gamma` from it today). Threading a parallel override dict through the
    backend signature would fork the contract that `bird.py` stage 3 pins, and
    copying the `Config` would not help because the backend is handed `ctx`,
    not the copy. So: overlay, call, restore in `finally` -- including when the
    backend raises or `BudgetExceeded` ends the run mid-grid.

    The resolved-config artifact is written once at run start from the loaded
    config and is never re-read from `ctx.cfg`, so a restored overlay leaves it
    untouched; the sweep survives in the artifact through the annotations on
    `TrainResult.seed_metrics` instead. That is the right place for it anyway:
    the grid point is a property of one training run, not of the run's config.

    Not validated. A grid value of the wrong type reaches the backend as
    written and fails there, loudly, rather than being coerced here into
    something the config never said.
    """
    raw = _raw(cfg)
    direct, hp = _split_point(point)
    undo: List[Tuple[str, Any]] = []
    try:
        for key, value in direct.items():
            undo.append((key, _get_path(raw, key)))
            _set_path(raw, key, value)
        if hp:
            previous = _get_path(raw, "train.hyperparameters")
            undo.append(("train.hyperparameters", previous))
            merged = dict(previous) if isinstance(previous, dict) else {}
            merged.update(hp)
            _set_path(raw, "train.hyperparameters", merged)
        yield
    finally:
        for key, old in reversed(undo):
            if old is _MISSING:
                _del_path(raw, key)
            else:
                _set_path(raw, key, old)


def _grid_points(grid: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Cartesian product of `train.hyperparameter_grid`, in declared order.

    Insertion order of the YAML mapping is the axis order and the first value
    of each axis is the first point, so point 0 is the config's own
    "first-listed everything". That matters because the tie-break below keeps
    the earliest winner: with an unresponsive grid the reported winner is a
    stated position in the YAML rather than an arbitrary one.
    """
    names = [str(k) for k in grid]
    axes: List[List[Any]] = []
    for name in names:
        values = grid[name]
        if isinstance(values, (list, tuple)):
            axes.append(list(values))
        else:
            # The schema documents {param: [values]}; a scalar is almost
            # certainly a slip, and silently treating it as a one-point axis
            # without saying so would turn a typo into a smaller search.
            log.warning("train.hyperparameter_grid[%s]=%r is not a list; treating it "
                        "as a single-value axis", name, values)
            axes.append([values])
    if not axes or any(not a for a in axes):
        return []
    return [dict(zip(names, combo)) for combo in itertools.product(*axes)]


def _point_fitness(ctx: Any, result: TrainResult) -> Optional[float]:
    """Score one grid point with **the same collapse stage 4 will apply**.

    Deliberately not a second, private definition of fitness. Singh's max is
    over the same quantity his mean is over -- "the mean cumulative fitness of
    `r_A` ... maximum estimate obtained over a coarse discretization" -- so
    this reuses `evaluate.fitness.checkpoint_aggregation` then
    `evaluate.fitness.seed_aggregation` through `components/evaluation.py`'s
    own two-step collapse, whose order is load-bearing (aggregating seeds first
    would maximise over an average). A grid that maximised `final` while stage
    4 ranked on `iqm` would select points the outer loop then disagrees with,
    and nothing in the artifact would show it.

    The `fitness_source` component is *not* invoked: `vlm_score` and
    `human_score` would spend LLM calls and human queries once per grid point,
    turning a learner sweep into a judging sweep. The ground-truth metric keys
    are read directly instead, which is what `fitness_source:
    ground_truth_metric` does and is the only source that is free to re-read.

    Returns `None` when the run produced no metric at all -- distinct from
    `0.0`, per the repo-wide rule -- so an unscoreable point can never win.
    """
    if not getattr(result, "trained", False):
        return None
    # Lazy: `registry.load_all()` imports both modules, and resolving the
    # aggregation names at call time is what keeps them config values.
    from ..types import CandidateReport
    from .evaluation import _component, _num, _per_seed_values
    from ..native_signal import native_curve_keys

    # Native-signal rule (N): the hyperparameter/Singh grid scores each point on
    # the env's OWN signal (`success_rate` / `gt_return`), never the BIRD
    # `task_metric` (custom_metric) -- it must mirror `fitness_source: native`,
    # not `ground_truth_metric`. A task with no native signal is unscoreable
    # natively -> None (distinct from 0.0), so no point can win on our metric.
    keys, _channel = native_curve_keys(env_id=ctx.cfg.get("problem.env_id"),
                                       task_id=ctx.cfg.get("problem.task_id"))
    if not keys:
        return None
    probe = CandidateReport(cand_id=result.cand_id, candidate=result.candidate,
                            result=result)
    per_seed = _per_seed_values(ctx, result, keys, probe)
    if not per_seed:
        return None
    seed_agg = _component("seed_aggregation",
                          ctx.cfg.get("evaluate.fitness.seed_aggregation", "mean"))
    return _num(seed_agg(per_seed))


def _annotate(result: TrainResult, mode: str, winner: Mapping[str, Any],
              scoreboard: Sequence[Dict[str, Any]], full: bool) -> None:
    """Write the sweep into the artifact, via `seed_metrics`.

    `TrainResult` is serialised with `dataclasses.asdict`, so an attribute
    tacked onto the instance would vanish on the way to disk; `seed_metrics`
    entries are `Dict[str, Any]` and survive. The keys are namespaced away from
    anything `_per_seed_values` looks for, so annotating cannot perturb the
    fitness it just measured.
    """
    rows = [r for r in (result.seed_metrics or []) if isinstance(r, dict)]
    if not rows:
        # A point that never compiled has no seed rows. Inventing one to hang
        # provenance off would put a fake seed in front of stage 4's fallback
        # path, so the annotation is simply lost here and the log carries it.
        log.debug("hyperparameter_search=%s: %s produced no seed rows to annotate",
                  mode, result.cand_id)
        return
    for row in rows:
        row["hyperparameter_search"] = mode
        row["hyperparameter_point"] = dict(winner)
        row["hyperparameter_grid_size"] = len(scoreboard)
    if full:
        rows[0]["hyperparameter_grid_scores"] = [dict(s) for s in scoreboard]


def _search(ctx: Any, state: Any, candidate: Any,
            backend: Callable[..., TrainResult], n_seeds: int,
            *, mode: str, full_record: bool, **backend_kw: Any) -> TrainResult:
    """Train once per grid point; return the best point's `TrainResult`.

    Shared body of `grid` and `per_candidate_best`. The winner is the strict
    maximum, so an exactly-tied grid keeps the earliest point -- deterministic,
    and it means "which point won" is answerable rather than a coin flip.
    """
    grid = ctx.cfg.get("train.hyperparameter_grid") or {}
    points = _grid_points(grid)
    if not points:
        # `config._check_coherence` already rejects an empty grid with a
        # non-`none` search, so this is only reachable from a hand-built ctx.
        log.warning("train.hyperparameter_search=%s with an empty "
                    "train.hyperparameter_grid: running one unmodified training run",
                    mode)
        return backend(ctx, state, candidate, n_seeds=n_seeds, **backend_kw)

    scoreboard: List[Dict[str, Any]] = []
    first: Optional[TrainResult] = None
    best: Optional[TrainResult] = None
    best_fitness: Optional[float] = None
    best_point: Dict[str, Any] = {}

    for point in points:
        with _overlaid(ctx.cfg, point):
            result = backend(ctx, state, candidate, n_seeds=n_seeds, **backend_kw)
        fitness = _point_fitness(ctx, result)
        scoreboard.append({"point": dict(point), "fitness": fitness})
        if first is None:
            first = result
        if fitness is not None and (best_fitness is None or fitness > best_fitness):
            best, best_fitness, best_point = result, fitness, dict(point)

    if best is None:
        # Every point failed to produce a metric. Returning the first keeps the
        # candidate's failure visible to stage 4 (`trained=False`, `error` set)
        # instead of dropping it, which is what `select.failure_value` is for.
        best, best_point = first, dict(points[0])  # type: ignore[assignment]
        log.warning("hyperparameter_search=%s: no point scored for %s; "
                    "keeping the first point's result", mode, candidate.cand_id)

    scored = [s["fitness"] for s in scoreboard if s["fitness"] is not None]
    if len(points) > 1 and len(scored) > 1 and max(scored) == min(scored):
        # Loud on purpose, and carefully worded: an all-tied scoreboard is an
        # *observation*, not a diagnosis. Two things produce it and the wrong
        # one is easy to assume.
        #
        # (a) The candidate is genuinely insensitive to the grid -- a reward
        #     that scores 0 whatever the learner does ties at 0 legitimately,
        #     and the max over the grid is then honestly the first point.
        # (b) The grid never reached the learner. `components/training.py`
        #     reads `train.env_steps`, `train.reward_norm`,
        #     `train.n_parallel_envs` and friends off the config, and
        #     `training.py::_learner_kwargs` routes bare names to the learner
        #     constructor and warns about any it does not accept, so this
        #     reading announces itself separately -- but a name the learner
        #     accepts and is simply flat in can still tie here.
        #
        # Either way the cost was paid, so the cost is what the message leads
        # with; the two readings follow. Silently reporting a max over a grid
        # that did nothing would present a no-op setting as a search result.
        log.warning("hyperparameter_search=%s: all %d grid points scored %.6g for %s, "
                    "at a cost of %d trainings, so the max is the first point by "
                    "tie-break. Either this candidate is genuinely insensitive to the "
                    "grid, or the swept names are not parameters of the learner this "
                    "env forced -- training.py warns separately about the second, so "
                    "if you saw no such warning it is the first.",
                    mode, len(points), scored[0], candidate.cand_id,
                    len(points) * max(1, n_seeds))

    _annotate(best, mode, best_point, scoreboard, full_record)
    log.info("  hyperparameter_search=%s: %d point(s) x %d seed(s) for %s -> "
             "best %s at fitness %s", mode, len(points), max(1, n_seeds),
             candidate.cand_id, best_point,
             "n/a" if best_fitness is None else f"{best_fitness:.4g}")
    return best


@register("hyperparameter_search", "none")
def hpsearch_none(ctx: Any, state: Any, candidate: Any,
                  backend: Callable[..., TrainResult], n_seeds: int = 1,
                  **backend_kw: Any) -> TrainResult:
    """No search: one training run under `train.hyperparameters` as written.

    The default, and every published LLM-era method. Pinning the learner
    across candidates is a deliberate choice -- Eureka,
    DrEureka, Text2Reward, L2R, RDA, GT, LIMEN and CARD all vary the reward and
    nothing else, so that a fitness difference is attributable to the reward.

    Exactly transparent, by construction: this is a bare tail call. No config
    is touched, no result is annotated, and `ctx.budget` sees precisely what it
    would have seen if stage 3 had called the backend directly. Anything more
    would make the default path pay for a feature it does not use, and would
    make "did wrapping change the numbers?" a question worth asking.
    """
    return backend(ctx, state, candidate, n_seeds=n_seeds, **backend_kw)


@register("hyperparameter_search", "grid")
def hpsearch_grid(ctx: Any, state: Any, candidate: Any,
                  backend: Callable[..., TrainResult], n_seeds: int = 1,
                  **backend_kw: Any) -> TrainResult:
    """Exhaustive sweep over `train.hyperparameter_grid`; keep the best point.

    Singh's protocol minus its bookkeeping: the fitness reported for a
    candidate is the maximum over the learner grid, which is the substantive
    change to the estimator, but the artifact records only *that* a grid ran
    and which point won -- not the whole scoreboard.

    Use this when the max is the claim. Use `per_candidate_best` when the
    winning point is itself a result you intend to report, which for Singh it
    is: his figures compare rewards whose learner settings differ, and that
    comparison is unreadable without the settings.
    """
    return _search(ctx, state, candidate, backend, n_seeds,
                   mode="grid", full_record=False, **backend_kw)


@register("hyperparameter_search", "per_candidate_best")
def hpsearch_per_candidate_best(ctx: Any, state: Any, candidate: Any,
                                backend: Callable[..., TrainResult],
                                n_seeds: int = 1, **backend_kw: Any) -> TrainResult:
    """Singh 2009: each reward is scored under **its own** best learner settings.

    Singh et al. (2009): "we estimate the mean cumulative fitness of
    `r_A` as the maximum estimate obtained ... over a coarse discretization of
    the space of feasible `(alpha, epsilon)` pairs" (p. 2603). The grid is the
    coarse discretization; `train.seeds_per_candidate` with
    `evaluate.fitness.seed_aggregation: mean` is the inner "average over N
    sampled environments"; this component is the outer max.

    Selection is identical to `grid` -- same points, same order, same
    tie-break, same cost. The difference is the record: the full per-point
    scoreboard is written to `seed_metrics[0]["hyperparameter_grid_scores"]`
    and the winning point to every seed row, so the artifact answers "which
    (alpha, epsilon) won for *this* reward, and by how much over the others?".
    That question is Singh's protocol rather than a detail of it: the whole
    reason to search per candidate is the claim that different rewards want
    different learners, and a run that cannot show two candidates choosing
    different points has not demonstrated it. Keeping the two values distinct
    means an ablation can turn the bookkeeping off and see the cost of the
    honest version.

    INFERRED, not double-dagger (there is no released code to supply a value):
    the paper says "coarse discretization
    of the space of feasible (alpha, epsilon)" and never lists the values, and
    there is no released code, so `train.hyperparameter_grid` in
    `configs/methods/singh_orp.yaml` is inferred and must be marked as such.
    """
    return _search(ctx, state, candidate, backend, n_seeds,
                   mode="per_candidate_best", full_record=True, **backend_kw)
