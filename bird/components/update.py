"""Stage 6 components -- Reward Update (§6).

What survives an iteration, and in what shape: the population topology, the
variation operator, what happens to the winner and to each loser, the prompt
history that goes back into §1, and which feedback CHANNEL the next prompt
speaks in.

Three slots, three different questions (`bird/state.py`):

    state.best            global incumbent -- moves only on improvement when
                          `update.elitism.keep_global_best` (carry slot
                          `best_reward`; a config that omits it forgets)
    state.iteration_best  this round's winner, whatever it was worth
    state.latest          the CHAIN HEAD -- what `generate.parent_source:
                          latest` reads next round

Keeping `latest` apart from `best` is not tidiness. **Every published parent
adoption is ungated**: Eureka (line 9), RDA (`R^i_{n*}`), GT (lines 15->19) and
CARD's chain all promote the round winner even when the round regressed (§6,
`update.winner.action`). CARD goes further -- §2's TPE fork records that "either
way the failed candidate keeps chain headship", so a candidate the screen
rejected still parents the next one. `single_parent_hillclimb` therefore
promotes UNCONDITIONALLY, and the two regression guards
(`update.rollback_if_worse`, which no published method sets, and §5's
`select.require_improvement_over_incumbent`, ditto) are the only things that
change that. If a run promotes downhill, that is the method, not a bug.

Ownership conventions this module fixes, so §1 does not have to guess:

  * **§6 owns `state.dialogue`.** `update.prompt.mode` is the WRITE policy for
    the carried transcript; `generate.history_mode` is §1's READ policy over
    the same list. Two halves of the one axis no published method ablates.
  * **§6 owns `state.archive`.** MAP-Elites cells are keyed
    `(island, *bin_indices)`; the unpublished `elitist_population` / `island`
    topologies reuse the same dict with `coords = ("elite", rank)`, because
    RunState has exactly one carried population slot and inventing a second
    one outside `CARRY_SLOTS` would make it invisible to `loop.carry`.
    The three knobs that SIZE that one dict are not interchangeable and LIMEN's
    released code sets all three (‡):
    `update.archive.population_size` is the per-island capacity of the elite
    slots, `update.archive.archive_size` caps the SAMPLING LIST the
    exploitation parent branch draws from (`exploitation_pool` -- the grid
    itself is never evicted; LIMEN's `_update_archive` edits only the top-50
    list, database.py:739-764), and
    `update.archive.migration_rate` is the fraction of a deme that crosses
    to its ring neighbour each migration event. Without those keys LIMEN
    -- the one non-hillclimb point in the whole space -- could not be stated as
    a config at all: the missing knob would be the bug, not the method.
  * `sample_archive_parent()` and `sample_archive_parents()` are exported here
    (not in §1) because `update.archive.parent_sampling` is a §6 key; §1's
    `parent_source: archive_sample` calls the first -- one sampler, so §1 and
    §6 cannot drift apart. The plural
    is REvolve's cohort draw, in which one deme supplies every parent of one
    offspring; the two share `_draw_branch` for the same reason.

Nothing here calls `state.apply_carry` -- `bird.py` does that after `update()`
returns, which is what makes "not in `loop.carry`" mean "gone".
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import preference_log, registry
from ..config import ConfigError
from ..context import Context
from ..registry import register
from ..state import ArchiveCell, RunState
from ..types import Candidate, CandidateReport, Preference, Selection, Trajectory

log = logging.getLogger("bird")

NEG_INF = float("-inf")

#: Tag that marks a population slot inside `state.archive`, so the
#: elite-slot topologies and MAP-Elites can share one carried dict without
#: their keys ever colliding (MAP-Elites coords are ints).
_ELITE = "elite"


# ==========================================================================
# scalars, gates, bookkeeping
# ==========================================================================


def _score(report: Optional[CandidateReport]) -> float:
    """Comparable scalar for a report.

    `fitness is None` means "no ranking scalar exists" -- CARD ranks nothing at
    all (§0 `problem.fitness_access: none`). Such a report sorts below
    everything and so can never displace an incumbent; CARD's chain advances
    through `winner_action`, never through elitism. That asymmetry is the
    method, and it is why `select.failure_value` (Eureka's -10000, a candidate
    that ran and lost) is deliberately NOT reused here for the unranked case.
    """
    if report is None or report.fitness is None:
        return NEG_INF
    return float(report.fitness)


def _update_global_best(ctx: Context, state: RunState, winner: CandidateReport) -> bool:
    """Maintain `state.best` and the stagnation counter. Returns "improved?".

    `update.elitism.keep_global_best` is exactly the Eureka `if s_best >
    s_eureka` guard (§5 `select.scope`): a bad iteration cannot destroy
    progress. With elitism off, the slot still tracks the last winner and can
    move downhill -- that is what "no elitism" means, and keeping the slot
    maintained is what keeps `parent_source`/`final_artifact: global_best`
    honest under that setting rather than silently frozen.
    """
    improved = _score(winner) > _score(state.best)
    if ctx.cfg.get("update.elitism.keep_global_best", True):
        if improved or state.best is None:
            state.best = winner
    else:
        state.best = winner
    # Owned here because §6 is the only stage that sees "did this round move".
    # `update.restart.patience` and the `restart` operator read it.
    state.stagnation = 0 if improved else state.stagnation + 1
    _carry_inner_loop_refs(state)
    return improved


def _carry_inner_loop_refs(state: RunState) -> None:
    """Publish the incumbent's policy / replay handles into the carry slots.

    §3's `train.init` reads `state.policy_ref` and `state.replay_ref`
    (`training._training_init`), `loop.carry` names them
    (`CARRY_SLOTS['policy_checkpoint'|'replay_buffer']`), `RunState` declares
    them and `checkpoint.py` persists them -- and **this function is the only
    place either one is assigned**. Without it `_training_init` would return
    `(None, [], 0.0)` on every call of every run, so `warm_start_from_best`
    (rda) and `secondary_replay_buffer` (gt) would be inert on EVERY backend,
    mock included, while the config still carried the method's name.

    THE SOURCE IS `state.best`, and the choice is the papers': RDA §4.2 warm-
    starts "from the best checkpoint of the previous iteration" (main.tex:299)
    and GT §4 hands on "the replay
    buffer from the previous iteration's best agent". Both say *best*, so both
    read the incumbent slot this function is called from -- which under
    `update.elitism.keep_global_best: false` tracks the last winner instead,
    exactly as `parent_source: global_best` already does. Deliberately NOT a
    new config key: `phases._policy_ref` offers `incumbent_best | latest |
    policy_checkpoint` for RAPP, but no published method ablates this axis, and
    a key nobody sets that changes what `train.init` means would be a knob
    invented to avoid writing this comment.

    Assignment is unconditional here; `RunState.apply_carry` -- run by
    `bird.py` AFTER `update()` -- is what makes a config that omits
    `policy_checkpoint` / `replay_buffer` from `loop.carry` truly not have it.
    Writing them and letting carry clear them is the same order every other
    slot follows.
    """
    result = getattr(state.best, "result", None)
    state.policy_ref = getattr(result, "policy_ref", None)
    state.replay_ref = getattr(result, "replay_ref", None)


def _promotion_blocked(ctx: Context, state: RunState, selection: Selection,
                       winner: CandidateReport) -> str:
    """The two regression guards, both published by nobody (§5, §6).

    Returns a reason string, or "" for the default -- promote regardless. The
    references differ on purpose: `rollback_if_worse` compares against the
    CHAIN HEAD it would replace, while §5's
    `select.require_improvement_over_incumbent` (evaluated in `bird.py`, which
    leaves the verdict on `Selection.improved_over_incumbent`) compares against
    the global incumbent. Acting on the §5 flag here is not §5 reaching
    forward: §5 computes the comparison, §6 owns promotion, and a declared key
    that no stage honours would silently do nothing.
    """
    if ctx.cfg.get("update.rollback_if_worse", False) and state.latest is not None:
        if _score(winner) < _score(state.latest):
            return (f"rollback_if_worse: {winner.cand_id} ({winner.fitness}) is worse than "
                    f"chain head {state.latest.cand_id} ({state.latest.fitness})")
    if selection.improved_over_incumbent is False:
        return (f"select.require_improvement_over_incumbent: {winner.cand_id} "
                f"({winner.fitness}) did not beat the incumbent")
    return ""


def _population_size(ctx: Context) -> int:
    """How many individuals one island's elite slots hold.

    ‡ `update.archive.population_size` (§6) is the DECLARED capacity, and the
    LIMEN paper never states it: the value comes from the released code --
    200 on the three XLand tasks, 40 on Brax / Panda / Go1
    (`refs/code/LIMEN/configs/*.yaml`, `limen/config.py` L166). Check that pin
    before leaning on it for a stronger reason than the usual ‡: grep the tree
    and `population_size` appears only in `DatabaseConfig` and the seven task
    YAMLs -- LIMEN's own loop never reads it. So this is a number the code
    supplies but does not itself honour. We honour it as the per-island
    capacity, which is the only reading under which the field does anything at
    all; it is NOT a † (the paper does not disagree, it is simply silent).

    §5's `select.n_survivors` and §1's `generate.n_parents` stay as a FLOOR
    rather than being replaced -- they are individuals the other two stages
    have already committed to keeping, and a deme smaller than either would
    quietly drop parents §1 is about to ask for. So this §6 key can only raise
    the capacity, never lower it; with its degenerate default of 1 the max is
    an identity, so a config that does not set the key is unaffected.
    """
    cfg = ctx.cfg
    floor = max(int(cfg.get("select.n_survivors", 1) or 1),
                int(cfg.get("generate.n_parents", 1) or 1), 1)
    declared = int(cfg.get("update.archive.population_size", 1) or 1)
    if declared < floor:
        log.debug("update.archive.population_size=%d is smaller than the %d slots "
                  "select.n_survivors / generate.n_parents already claim; keeping %d",
                  declared, floor, floor)
    return max(declared, floor)


# --- the trajectory store (`update.memory.trajectory_store`) --------------
#
# Table-driven rather than an if-chain: the config VALUE selects the rule, in
# the same spirit as the registry. This key has no registry family of its own
# because its three values are selectors over one list, not implementations.

def _screen_failed(candidate: Candidate) -> bool:
    """Any screen-phase verify record with `ok: False` -- the VERDICT, not the
    routing. `candidate.trainable` alone cannot see the `train_anyway` arm,
    where a failing candidate trains and stays trainable but must still not
    extend the store: both the paper (Alg. 1 l.9-11,
    T grows only inside the if-preference branch) and the release
    (metaworld_exp_one_step.py:428, extend inside the pass_flag branch; the
    :430-438 else routes feedback and never extends) key store growth on the
    screen's verdict."""
    return any(rec.get("phase") == "screen" and rec.get("ok") is False
               for rec in candidate.verify_records)


_TRAJECTORY_SELECTORS: Dict[str, Callable[[Selection], List[CandidateReport]]] = {
    "none": lambda sel: [],
    # CARD: the screen gates the growth of its own evidence (§2, `verify.tpe.store`).
    # "Passed the screens" is two conditions, and both read the verdict:
    # `trainable` covers the skip_training arm (a screened candidate is not
    # trainable), `not _screen_failed` covers train_anyway's trained-but-failed
    # record.
    "append_on_pass": lambda sel: [r for r in sel.winners
                                   if r.candidate.trainable
                                   and not _screen_failed(r.candidate)],
    # GT's `D_pref` analogue: grows unconditionally.
    "append_always": lambda sel: list(sel.winners) + list(sel.losers),
}


def _maintain_memory(ctx: Context, state: RunState, selection: Selection) -> None:
    """Grow `state.trajectory_store` per `update.memory.trajectory_store`.

    Called by every topology, because the store is a property of the round and
    not of the population shape; hiding it inside one topology would make the
    key mean different things under different configs. Identity-deduped so
    that a §2 screen which already appended the same `Trajectory` objects
    (CARD carries `trajectory_store` for exactly this evidence) cannot
    double-count them.
    """
    mode = ctx.cfg.get("update.memory.trajectory_store", "none")
    try:
        selector = _TRAJECTORY_SELECTORS[mode]
    except KeyError:
        raise ConfigError(
            f"update.memory.trajectory_store: unknown value {mode!r}; "
            f"expected one of {sorted(_TRAJECTORY_SELECTORS)}") from None

    # Which pool grows the store: when
    # `verify.tpe.trajectories_per_iteration` > 0, stage 3 collected a DEDICATED
    # pool of that size (`TrainResult.store_trajectories`, CARD's
    # `--trajectory_num` check pool) and that is what enters the store --
    # decoupled from `result.trajectories`, which `evaluate.rollouts_per_candidate`
    # sizes for feedback. No silent fallback between the two: an empty dedicated
    # pool (e.g. training failed) must read as empty, not as the wrong
    # population.
    dedicated = int(ctx.cfg.get("verify.tpe.trajectories_per_iteration") or 0) > 0
    seen = {id(t) for t in state.trajectory_store}
    for report in selector(selection):
        pool = report.result.store_trajectories if dedicated else report.result.trajectories
        for traj in pool:
            if id(traj) in seen:
                continue
            seen.add(id(traj))
            state.trajectory_store.append(traj)


def _dispatch_winner(ctx: Context, state: RunState, winner: CandidateReport) -> None:
    registry.get("winner_action", ctx.cfg.get("update.winner.action", "become_parent"))(
        ctx, state, winner)


def _dispatch_operator(ctx: Context, state: RunState, selection: Selection) -> None:
    # After the winner is installed: `restart` must be able to wipe a chain
    # that already advanced, and `archive_insert` needs the survivor set.
    registry.get("update_operator", ctx.cfg.get("update.operator", "none"))(
        ctx, state, selection)


# ==========================================================================
# the archive (LIMEN) -- cells, islands, migration
# ==========================================================================


def _descriptor_spec(ctx: Context) -> Tuple[List[str], List[int]]:
    names = list(ctx.cfg.get("update.archive.descriptors") or [])
    bins = [int(b) for b in (ctx.cfg.get("update.archive.bins_per_descriptor") or [])]
    if len(bins) < len(names):
        # config validation already rejects this pairing; pad rather than
        # raise so a hand-built test Config still runs.
        log.debug("archive: %d descriptors but %d bin counts; padding with 1",
                  len(names), len(bins))
        bins = bins + [1] * (len(names) - len(bins))
    return names, bins


def bin_report(ctx: Context, state: RunState, report: CandidateReport,
               names: Sequence[str], bins: Sequence[int]) -> tuple:
    """Bin ONE report onto the descriptor grid, LIMEN's way, exactly once.

    Reproduces `database.py::_bin_features` (L703-732): per axis, fold the raw
    value into `state.descriptor_stats`' running min/max (never reset within a
    run), min-max scale it to [0, 1] (0.5 when no range exists yet or the range
    is degenerate, `_scale_feature` L724-732), then
    `clamp(int(scaled * n_bins), 0, n_bins - 1)`. The coords this returns are
    FROZEN by the caller into `report.meta["archive_coords"]` -- the stats
    moving later re-keys nothing, which is what keeps the grid append-only.

    Worked example, 5 bins on one axis: the first value ever seen (say 313) has
    no range, scales to 0.5, and lands in the middle bin 2; a later 1842 sets
    max=1842, scales to 1.0, clamps into the top bin 4; 313 re-seen by a NEW
    report now scales to 0.0 -> bin 0 -- while the first report's frozen (2,)
    stays where it was, exactly as LIMEN's per-program `feature_coords` do.

    † WHY NOT §4.2's FIXED GRID. The paper describes a fixed uniform grid, and
    fixed ranges (say [0, 400] AST nodes, [0, 512] obs) would keep cells
    comparable across iterations. But a fixed range fails on the release's own
    data: LIMEN's five published winners have compute_reward AST counts
    313/631/978/1787/1842 (`examples/evolved_interfaces/*.py`), so 4/5 would
    saturate a [0, 400] cap and the complexity axis would degenerate. The
    release's adaptive scaler is EXECUTED code in every reported run (a config
    file may be a dev leftover; `_scale_feature` is not), so this follows what
    the code ran; §4.2's grid keeps the † as the paper's reading. Freezing
    per-program coords -- also the release's behaviour, `migrate()` copies
    `feature_coords` verbatim -- answers the comparability concern: a cell
    never changes meaning under an occupant, it only means less about reports
    binned later.

    The raw value is read from `report.descriptors` (stage 5 writes it there
    before calling; a report that never passed through stage 5 falls back to
    0.0, warned about by `insert_into_archive`).
    """
    coords: List[int] = []
    for k, name in enumerate(names):
        value = float(report.descriptors.get(name, 0.0))
        stats = state.descriptor_stats.setdefault(name, {"min": value, "max": value})
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
        span = stats["max"] - stats["min"]
        scaled = 0.5 if span < 1e-8 else (value - stats["min"]) / span
        n_bins = max(int(bins[k] or 1), 1) if k < len(bins) else 1
        coords.append(max(0, min(n_bins - 1, int(scaled * n_bins))))
    return tuple(coords)


def _age_key(report: Optional[CandidateReport]) -> Tuple[int, str]:
    """A deterministic stand-in for insertion time, used only to break ties.

    `ArchiveCell` has no timestamp and `bird/state.py` is not this module's to
    extend, so age is read off the candidate: ids are handed out by
    `Context.next_id` in creation order (`c0000`, `c0001`, ...), which makes
    `(iteration, cand_id)` an ordering by when the individual was made.

    Deliberately NOT the archive dict's own ordering. That order is
    deterministic in CPython, but it tracks insertion churn (a fitter newcomer
    re-writes its cell's entry), so dict position is when a cell last CHANGED
    rather than how old its occupant is, and a rule resting on it would quietly
    change meaning whenever a cell was contested.

    Deliberately not `ctx.rng` either: a coin flip here would consume draws
    from the same stream `sample_archive_parent` reads, so merely capping the
    sampling pool would shift which parents the run goes on to pick. The pool
    cut and migration below therefore use no randomness at all, which is what
    keeps `tests/test_pipeline.py`'s byte-reproducibility claim true when these
    keys are turned on.
    """
    if report is None:
        return (-1, "")
    return (int(report.candidate.iteration), str(report.cand_id))


def exploitation_pool(ctx: Context, cells: Sequence[ArchiveCell]) -> List[ArchiveCell]:
    """The top `update.archive.archive_size` GRID cells by fitness -- the pool
    the global parent branches draw from. `null` (the default) is all of them.

    ‡ The LIMEN paper is silent on the cap; its code sets one -- 50 on the
    XLand tasks, 20 on Brax / Panda / Go1 (`limen/config.py` L167, task YAMLs).
    WHAT IT CAPS IS A SAMPLING LIST, NOT THE GRID:
    `database.py::_update_archive` (L739-764) edits only `self.archive`, the
    top-50 id LIST, which `_sample_exploitation` (L341-346) alone reads; the
    grid and the islands are untouched, and a program refused by the list keeps
    its cell. Applying the same number as a hard cap on occupied descriptor
    CELLS would evict the lowest-fitness niches outright -- up to a third of a
    75-cell LIMEN archive.

    Derived top-K rather than a stored membership list: the release's list is
    path-dependent (first-50-in stay until strictly beaten; a program DISPLACED
    from its grid cell keeps its list membership), which would cost another
    RunState field for a marginal fidelity gain. Deterministic ties go
    oldest-first (`_age_key`, stable sort), so among equally fit cells the
    longest-standing elite makes the cut -- the same keep-the-incumbent
    convention as `insert_into_archive` and `_truncate` -- and no `ctx.rng` is
    consumed (see `_age_key`).

    Grid cells only: the `_ELITE` population slots (module docstring) are the
    release's non-grid island programs, which its archive list never holds --
    a cell LOSER filed there by `lose_store_as_negative_example` is reachable
    through the island branches, exactly as in the release.
    """
    grid = [c for c in cells if c.report is not None and c.coords[:1] != (_ELITE,)]
    cap = ctx.cfg.get("update.archive.archive_size", None)
    if cap is None:
        return grid
    cap = int(cap)
    if cap < 0:
        # Once per run, not once per draw. The flag lives on `ctx.counters` for
        # the same reason the migration cadence does -- a module global would
        # leak across runs and break test isolation.
        if not ctx.counters.get("archive_cap_negative_warned"):
            ctx.counters["archive_cap_negative_warned"] = True
            log.warning("update.archive.archive_size=%d is negative; leaving the "
                        "sampling pool unbounded", cap)
        return grid
    grid.sort(key=lambda c: _age_key(c.report))          # ties: oldest first
    grid.sort(key=lambda c: c.fitness, reverse=True)     # fitness dominates
    return grid[:cap]


def _island_for(ctx: Context, state: RunState, report: CandidateReport) -> int:
    """Which deme an insertion belongs to.

    ‡ `n_islands` is unstated in the LIMEN paper; its code uses 3. Offspring
    inherit their parent's island (that is what makes an island a lineage);
    a candidate with no archived parent is placed round-robin by iteration,
    which keeps island assignment deterministic without consuming `ctx.rng`.

    "Their parent's island" is the deme the parent was SAMPLED FROM:
    `meta["parent_island"]`, stamped by §1 (`generation._build_candidate`)
    off the parent report's `archive_island`. For a migrant that is the
    destination deme -- the release's stored-island rule (`database.py:245-247`,
    `:567-580`). The scan by `parent_id` below is the fallback for a candidate
    built without the stamp; on its own it would find the SOURCE cell first for
    a migrated parent.
    """
    n = max(int(ctx.cfg.get("update.archive.n_islands", 1) or 1), 1)
    if n == 1:
        return 0
    stamped = report.candidate.meta.get("parent_island")
    if stamped is not None:
        return int(stamped) % n
    parent_id = report.candidate.parent_id
    if parent_id:
        for cell in state.archive.values():
            if cell.report is not None and cell.report.cand_id == parent_id:
                return cell.island
    return int(state.iteration) % n


def insert_into_archive(ctx: Context, state: RunState, report: Optional[CandidateReport],
                        island: Optional[int] = None, count: bool = True) -> bool:
    """Insert one report into `state.archive`; the fitter occupant keeps the cell.

    LIMEN (§6, `update.archive.*`) -- the one non-hillclimb point in the
    literature. One cell per `(island, *bin_tuple)`, contested exactly as
    `database.py::add` L263-292: a strict `>` against that one cell's occupant,
    so ties keep the incumbent, and NOTHING ELSE moves -- the grid is
    append-only per cell, never rebuilt and never evicted (the release has
    neither an adaptive re-key, which could merge two occupants and drop one,
    nor a capacity eviction pass).

    COORDS ARE CONSUMED, NOT RECOMPUTED. `report.meta["archive_coords"]` --
    frozen by `selection.rule_map_elites_insert`, whose verdict
    `meta["archive_cell_win"]` was computed against exactly this key -- wins
    when present, which is also what makes `_maybe_migrate`'s dest-island copy
    the release's `migrate()`: coords travel verbatim, only the island prefix
    changes (the explicit `island` parameter outranks `meta["archive_island"]`
    for that reason). A report that never passed through the rule (the other
    select rules' `winner.action`/`loser.action: insert_archive` paths) is
    binned here by `bin_report` and its coords frozen the same way, so no
    caller changes behaviour silently.

    A report with `fitness is None` is REFUSED, not filed at -inf: None means
    no ranking scalar exists (`_score`), and a cell whose occupant cannot lose
    a comparison would be unbeatable garbage. A measured 0.0 inserts -- None
    stays distinct from 0.0.
    """
    if report is None:
        return False
    if report.fitness is None:
        log.debug("      archive: %s carries no fitness; a cell is a fitness "
                  "contest, so it is not inserted", report.cand_id)
        return False
    names, bins = _descriptor_spec(ctx)
    if names and not any(n in report.descriptors for n in names):
        log.warning("archive: %s carries none of the descriptors %s that §5 is "
                    "configured to measure; it lands in the origin cell",
                    report.cand_id, names)
    if island is not None:
        isl = int(island)
    elif report.meta.get("archive_island") is not None:
        isl = int(report.meta["archive_island"])
    else:
        isl = _island_for(ctx, state, report)

    coords = report.meta.get("archive_coords")
    if coords is None:
        coords = bin_report(ctx, state, report, names, bins)
        report.meta["archive_coords"] = coords
    else:
        coords = tuple(int(c) for c in coords)

    key = (isl,) + coords
    fitness = _score(report)
    current = state.archive.get(key)
    if current is None or fitness > current.fitness:
        state.archive[key] = ArchiveCell(coords=coords, island=isl,
                                         report=report, fitness=fitness)
    report.meta.setdefault("archive_island", isl)
    placed = state.archive[key].report is report
    # ONE evaluated result, ONE count. LIMEN's config reaches this function
    # three times per winner -- the topology's own insertion, then
    # `winner.action: insert_archive`, then `operator: archive_insert` -- and
    # counting on each path would count a single win three times and fire
    # `migration_interval` three times too early. The release counts once
    # per evaluated program (`controller.py:503-508`, `:533-540`). The grid
    # write above is idempotent, and so is this bookkeeping.
    if count and not report.meta.get("archive_counted"):
        report.meta["archive_counted"] = True
        n = ctx.counters.get("archive_inserts", 0) + 1
        ctx.counters["archive_inserts"] = n
        ctx.event("update_archive", cand_id=report.cand_id, island=isl, placed=placed,
                  occupied=len(state.archive), inserts=n)
        _maybe_migrate(ctx, state)
    return placed


def _island_members(state: RunState, island: int) -> List[ArchiveCell]:
    """Everything one deme currently holds, fittest first.

    Both cell kinds count. The descriptor cells MAP-Elites fills and the elite
    slots the `island` topology writes are the same deme seen through two
    stores (module docstring), so a migration RATE that looked at only one of
    them would silently mean a different thing under each topology -- and the
    rate is a number a config states once.

    Ties are ordered oldest-first, so the individual that has held its niche
    longest is the one that crosses; the ordering is a pure function of the
    cells, never of dict churn or of `ctx.rng`.
    """
    cells = [c for c in state.archive.values()
             if c.report is not None and c.island == island]
    cells.sort(key=lambda c: _age_key(c.report))
    cells.sort(key=lambda c: c.fitness, reverse=True)
    return cells


def _n_migrants(ctx: Context, population: int) -> int:
    """How many of a deme's `population` individuals cross per event.

    ‡ `update.archive.migration_rate` (§6) is unstated in the LIMEN paper and
    supplied by the code: 0.1 on the XLand tasks, 0.2 on Brax / Panda / Go1
    (`limen/config.py` L177 and the seven task YAMLs).
    Implemented there as
    `n_migrate = max(1, int(len(src_programs) * migration_rate))`
    (`limen/database.py::migrate` L561) -- a floor of one, so any positive rate
    too small to round up to a whole individual still moves the island's best.

    BIRD keeps that floor and adds the one thing LIMEN's expression cannot say:
    `0.0` is a hard OFF switch, not another small rate. It has to be, because
    0.0 is the default every non-archive config inherits, and "the degenerate
    default changes nothing" is the property that lets this key be added to a
    schema whose other points are already pinned.
    """
    if population <= 0:
        return 0
    rate = float(ctx.cfg.get("update.archive.migration_rate", 0.0) or 0.0)
    if rate <= 0.0:
        return 0
    return max(1, min(int(population * rate), population))


def _island_generations(inserts: int, n_islands: int) -> List[int]:
    """The LIMEN release's per-island generation counters after `inserts` iterations.

    A pure function of the iteration count -- and BIRD's `archive_inserts` IS that
    count, one per evaluated result (see `insert_into_archive`) -- so the island clock cannot
    drift from, or double-count against, the global one. Mirrors `controller.py:533-540`
    exactly: increment the CURRENT island's generation each iteration, then rotate the
    current island every `n_islands` iterations. With 3 islands and 30 iterations this
    returns [12, 9, 9] -- island 0 leads because it is current for the first block of
    every super-cycle -- whose max, 12, is below a `migration_interval` of 20, which is
    why the release migrates zero times at the published 30-candidate budget.
    """
    gens = [0] * n_islands
    cur = 0
    for i in range(1, inserts + 1):
        gens[cur] += 1
        if i % n_islands == 0:
            cur = (cur + 1) % n_islands
    return gens


def _maybe_migrate(ctx: Context, state: RunState,
                   capacity: Optional[int] = None) -> None:
    """Ring migration between demes, every `update.archive.migration_interval`
    insertions.

    ‡ Both the interval and the island count are unstated in the LIMEN paper;
    its code migrates on an interval of 20 (5 on the Brax-family tasks) across
    2 or 3 islands by task -- there is no single "the" LIMEN configuration.
    The cadence counter lives in `ctx.counters` (the
    Context's run-scoped counter facility, already used for candidate ids)
    because `RunState` has no slot for it and a module global would leak
    between runs and break test isolation.

    WHO GOES: the top `_n_migrants` of each deme by fitness -- so
    `migration_rate: 0.0`, the default, migrates nobody at all.

    WHERE TO: a RING, `dest = (src + 1) % n_islands`, with every deme sending
    to its successor -- `limen/database.py::migrate` L556-566. The ring is
    worth defending over a random destination: its per-event contact graph is a
    permutation, so no deme is a sink, each both sends and receives exactly
    once, and the number of arrivals is fixed by the rate alone instead of by
    how many senders a random draw happened to aim at the same island. It also
    consumes no randomness, so switching migration on cannot shift the
    `ctx.rng` stream `sample_archive_parent` draws parents from -- an island
    model whose diversity claim was entangled with a re-seeded parent sequence
    would not be measurable.

    Migration COPIES rather than moves: the source deme keeps its elite, which
    is the standard island-model semantics, is what LIMEN's code does (it
    constructs a new `Program` for the destination), and is the only version
    that cannot empty an island.

    `capacity` is the size a RECEIVING deme's elite slots are truncated back
    to, and it is a parameter because `_population_size` is a FLOOR and not a
    cap: it returns `max(population_size, select.n_survivors,
    generate.n_parents, 1)`, so a topology whose demes are deliberately
    smaller than the round's survivor count would have every receiving deme
    silently re-truncated to the larger number while the resolved config still
    reads its own `population_size` -- with a ring, that is every deme, every
    migration event, and nothing in the artifact would say so. `None` is the
    floor.
    """
    n_islands = max(int(ctx.cfg.get("update.archive.n_islands", 1) or 1), 1)
    interval = int(ctx.cfg.get("update.archive.migration_interval", 0) or 0)
    if n_islands < 2 or interval <= 0:
        return
    inserts = ctx.counters.get("archive_inserts", 0)
    if inserts <= 0:
        return
    # WHICH CLOCK counts `interval`. `island_generations` is the LIMEN release
    # (`database.py::should_migrate` :532-535): a per-island generation clock under which
    # 3 islands x 30 iterations never migrate. `global_inserts` (the default) counts every
    # insertion, `inserts % interval`.
    clock = str(ctx.cfg.get("update.archive.migration_clock", "global_inserts") or "global_inserts")
    if clock == "island_generations":
        # A migration fires exactly when the MAX island generation crosses another
        # `interval` boundary -- `database.py::should_migrate` (:532-535) with its
        # `last_migration_gen` folded away. `_maybe_migrate` runs at every consecutive
        # `archive_inserts` (one per evaluated result) and the max rises by
        # at most one per iteration, so comparing this iteration's max against the
        # previous iteration's tests the same crossing with NO second, resumable counter:
        # the whole clock is a pure function of `archive_inserts`.
        now = max(_island_generations(inserts, n_islands))
        prev = max(_island_generations(inserts - 1, n_islands)) if inserts > 1 else 0
        if now // interval == prev // interval:
            return
    else:
        if inserts % interval:
            return

    # Every sender is chosen BEFORE anything moves: an arrival must not be
    # eligible to hop onward within the same event (that would make the
    # effective rate depend on island order), and both stores are rebuilt
    # under us as migrants land. sorted(): order must not depend on dict churn.
    islands = sorted({c.island for c in state.archive.values() if c.report is not None})
    outbound: List[Tuple[int, ArchiveCell, int]] = []
    for isl in islands:
        members = _island_members(state, isl)
        n = _n_migrants(ctx, len(members))
        # `n` travels with each migrant so the journal records how many left
        # THAT deme -- the rate is per-deme, and a run-wide total would not be
        # comparable across configs with different `n_islands`.
        outbound.extend((isl, cell, n) for cell in members[:n])

    for isl, cell, n in outbound:
        dest = (isl + 1) % n_islands
        if cell.coords[:1] == (_ELITE,):
            # elite-slot store (the unpublished topologies): merge into the
            # destination deme's ranking rather than into a descriptor cell.
            cap = _population_size(ctx) if capacity is None else max(int(capacity), 1)
            merged = _truncate(_population(state, dest) + [cell.report], cap)
            _set_population(state, dest, merged)
        else:
            # A DISTINCT report object for the destination, carrying its own
            # island: the release constructs a new `Program` with the
            # destination island stored on it (`database.py:567-580`) and
            # assigns that program's offspring by the stored island
            # (`:245-247`). Filing the SAME object in two demes would lose that:
            # `sample_archive_parent` would return it with no memory of which
            # cell it came from, `_island_for` would find the SOURCE cell first,
            # and a migrant's offspring would rejoin the island it had left.
            # Same `cand_id`: it is the same program.
            migrant = replace(cell.report, meta={**cell.report.meta,
                                                 "archive_island": dest,
                                                 "migrated_from": isl})
            insert_into_archive(ctx, state, migrant, island=dest, count=False)
        ctx.event("update_migration", cand_id=cell.report.cand_id, src=isl, dest=dest,
                  interval=interval, rate=ctx.cfg.get("update.archive.migration_rate", 0.0),
                  n_migrants=n, n_outbound=len(outbound))


# ==========================================================================
# elite-slot population store (shares `state.archive`)
# ==========================================================================


def _population(state: RunState, island: int) -> List[CandidateReport]:
    cells = [c for c in state.archive.values()
             if c.report is not None and c.island == island and c.coords[:1] == (_ELITE,)]
    cells.sort(key=lambda c: c.coords[1])  # by rank, so order is the stored order
    return [c.report for c in cells]


def _set_population(state: RunState, island: int, reports: Sequence[CandidateReport]) -> None:
    kept = {k: c for k, c in state.archive.items()
            if not (c.island == island and c.coords[:1] == (_ELITE,))}
    for rank, rep in enumerate(reports):
        coords = (_ELITE, rank)
        kept[(island,) + coords] = ArchiveCell(
            coords=coords, island=island, report=rep, fitness=_score(rep))
    state.archive = kept


def _truncate(reports: Sequence[CandidateReport], k: int) -> List[CandidateReport]:
    """Truncation selection, deduped by cand_id.

    The sort is stable and incumbents are passed in first, so an exact tie
    keeps the sitting elite rather than churning the population -- the §5
    `tie_break` axis decides contests, not this.
    """
    seen, uniq = set(), []
    for r in reports:
        if r is None or r.cand_id in seen:
            continue
        seen.add(r.cand_id)
        uniq.append(r)
    uniq.sort(key=_score, reverse=True)
    return uniq[:max(int(k), 1)]


# ==========================================================================
# topology  --  topology_fn(ctx, state, selection) -> RunState
# ==========================================================================


@register("topology", "single_parent_hillclimb")
def topo_single_parent_hillclimb(ctx: Context, state: RunState,
                                 selection: Selection) -> RunState:
    """One survivor, one parent: Eureka, DrEureka, Text2Reward, L2R, RDA, GT, CARD.

    Everything published except LIMEN (§6). Eureka calls itself evolutionary
    search, but with one survivor and no crossover it is hill climbing with 16
    proposals per step.

    The promotion is UNCONDITIONAL by default -- no published method gates it.
    Note the ordering: `state.best` is updated (under elitism) BEFORE the
    promotion gate, so even a rolled-back round still records a new global
    incumbent if it found one; and `state.iteration_best` is set before any
    gate at all, because "this iteration's best" is a fact about the round,
    not a decision about it.
    """
    winner = selection.winner
    state.iteration_best = winner
    _maintain_memory(ctx, state, selection)
    if winner is None:
        return state

    _update_global_best(ctx, state, winner)

    blocked = _promotion_blocked(ctx, state, selection, winner)
    if blocked:
        log.info("      parent NOT promoted -- %s", blocked)
        state.last_selection_notes = blocked
        ctx.event("update_rollback", cand_id=winner.cand_id, reason=blocked)
        return state

    _dispatch_winner(ctx, state, winner)
    _dispatch_operator(ctx, state, selection)
    return state


@register("topology", "elitist_population")
def topo_elitist_population(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """A real (mu + lambda) population. Published by LaRes and nobody else.

    `configs/methods/lares.yaml` takes it at `select.n_survivors: 3` and names
    `archive` in `loop.carry`, which is where the survivors live: a config
    that omits `archive` drops the survivors this function maintains at the
    iteration boundary. Selected is not the same as effective.

    The survivors are this round's winners merged with the sitting elites and
    truncated to `select.n_survivors` / `generate.n_parents`, so with
    `n_survivors: 1` this degenerates exactly to hill climbing with global
    elitism -- which is the honest unification: population size is a §5 key
    and §6 only maintains what §5 decided.
    """
    winner = selection.winner
    state.iteration_best = winner
    _maintain_memory(ctx, state, selection)
    if winner is None:
        return state

    _update_global_best(ctx, state, winner)
    blocked = _promotion_blocked(ctx, state, selection, winner)
    if blocked:
        log.info("      population NOT updated -- %s", blocked)
        state.last_selection_notes = blocked
        ctx.event("update_rollback", cand_id=winner.cand_id, reason=blocked)
        return state

    survivors = _truncate(_population(state, 0) + list(selection.winners),
                          _population_size(ctx))
    _set_population(state, 0, survivors)
    log.debug("      population: %s", [r.cand_id for r in survivors])

    _dispatch_winner(ctx, state, winner)
    _dispatch_operator(ctx, state, selection)
    return state


@register("topology", "island")
def topo_island(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """`n_islands` independent demes with periodic migration. Published by
    nobody as a topology -- LIMEN's islands exist, but inside its archive (§6).

    Each round feeds one deme, chosen round-robin by iteration so that the
    assignment is deterministic and every island is fed equally often; each
    deme holds `update.archive.population_size` individuals (floored by §5's
    `select.n_survivors`, see `_population_size`). The demes only ever talk
    through `update.archive.migration_interval` x `migration_rate` -- when to
    migrate, and what fraction crosses -- which is the whole point of the
    topology: it slows the loss of diversity that `elitist_population`
    suffers, and at `migration_rate: 0.0` it degenerates to `n_islands`
    genuinely independent searches sharing one `state.best`.
    """
    winner = selection.winner
    state.iteration_best = winner
    _maintain_memory(ctx, state, selection)
    if winner is None:
        return state

    _update_global_best(ctx, state, winner)
    blocked = _promotion_blocked(ctx, state, selection, winner)
    if blocked:
        log.info("      island NOT updated -- %s", blocked)
        state.last_selection_notes = blocked
        ctx.event("update_rollback", cand_id=winner.cand_id, reason=blocked)
        return state

    n_islands = max(int(ctx.cfg.get("update.archive.n_islands", 1) or 1), 1)
    isl = int(state.iteration) % n_islands
    survivors = _truncate(_population(state, isl) + list(selection.winners),
                          _population_size(ctx))
    _set_population(state, isl, survivors)
    ctx.counters["archive_inserts"] = ctx.counters.get("archive_inserts", 0) + 1
    ctx.event("update_island", island=isl, size=len(survivors),
              members=[r.cand_id for r in survivors])
    _maybe_migrate(ctx, state)

    _dispatch_winner(ctx, state, winner)
    _dispatch_operator(ctx, state, selection)
    return state


@register("topology", "archive_map_elites")
def topo_archive_map_elites(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """LIMEN: MAP-Elites + islands + migration -- the one archive point in the
    literature (§6).

    Deliberately NOT gated by `_promotion_blocked`: there is no chain to roll
    back, and a worse-but-differently-shaped elite occupying an empty niche is
    the entire mechanism. `state.best` is still maintained so that
    `select.final_artifact: global_best` remains answerable under an archive
    config.

    The topology inserts the round's winners itself rather than relying on
    `update.winner.action: insert_archive`, so that a config which leaves the
    winner action at its `become_parent` default cannot silently run
    MAP-Elites with an empty archive. Re-inserting the same report through the
    winner action or the `archive_insert` operator afterwards is idempotent for
    the GRID (same cell, not fitter) AND for the BOOKKEEPING: `insert_into_archive`
    counts a report once however many of the three paths a config selects
    (LIMEN's selects all three), so `archive_inserts` moves once per evaluated
    result and `migration_interval` means what it says.
    """
    winner = selection.winner
    state.iteration_best = winner
    _maintain_memory(ctx, state, selection)
    if winner is None:
        return state

    names, _ = _descriptor_spec(ctx)
    if not names:
        log.warning("update.topology=archive_map_elites with no "
                    "update.archive.descriptors: the archive collapses to one cell")

    _update_global_best(ctx, state, winner)
    for report in selection.winners:
        insert_into_archive(ctx, state, report)

    _dispatch_winner(ctx, state, winner)
    _dispatch_operator(ctx, state, selection)
    return state


# ==========================================================================
# update_operator  --  operator_fn(ctx, state, selection) -> None
#
# Called from the topology, after the survivors are installed. The operator is
# the VARIATION rule; §1 performs the variation (an LLM is the mutation
# operator), so what these do is guarantee the state that §1 needs in order to
# apply it.
# ==========================================================================


@register("update_operator", "reflect_and_mutate")
def op_reflect_and_mutate(ctx: Context, state: RunState, selection: Selection) -> None:
    """Parent + critique -> next candidate. Eureka, DrEureka, RDA, GT, CARD.

    The mutation operator is the LLM itself, so §6's whole job is to guarantee
    that the parent (`state.latest` / `state.best`) and the critique (the
    prose on the winning report, routed into `state.dialogue` by
    `update.prompt.mode`) are both present. The note left here,
    `state.last_selection_notes`, has no reader anywhere in bird/ (only
    writers), so this operator is behaviourally
    `none` for every config that pins it.
    """
    if selection.notes:
        state.last_selection_notes = selection.notes


@register("update_operator", "crossover")
def op_crossover(ctx: Context, state: RunState, selection: Selection) -> None:
    """LLM-merge of two parents. Selected by no config.

    Every method takes `reflect_and_mutate`, `archive_insert` or `none`.

    This is a claim about this ROUND-LEVEL operator and not about
    recombination: REvolve, R* and RF-Agent
    each publish one, and each reaches it from §1 instead
    (`generate.crossover_rate` inside `evolutionary_operators`,
    `generate.crossover.operator: module_insert`, `crossover_elite` in
    `generate.actions`). Choosing one operator for the whole round, which is
    what this family does, is still published by nobody.

    §6's contribution is the parent pool: the round's best `generate.n_parents`
    reports are pinned into the population store so that §1's `n_parents > 1`
    prompt has something to mix. With `n_parents: 1` this is mutation wearing
    a different name, and says so.
    """
    n_parents = int(ctx.cfg.get("generate.n_parents", 1) or 1)
    if n_parents < 2:
        log.warning("update.operator=crossover with generate.n_parents=%d: "
                    "one parent cannot be crossed over (this is mutation)", n_parents)
    pool = _truncate(_population(state, 0) + list(selection.winners) + list(selection.losers),
                     max(n_parents, _population_size(ctx)))
    _set_population(state, 0, pool)


@register("update_operator", "archive_insert")
def op_archive_insert(ctx: Context, state: RunState, selection: Selection) -> None:
    """Insert the survivors into the archive (LIMEN's operator form).

    The same act as `winner.action: insert_archive`, reachable from the
    operator axis so that a hill-climbing config can keep a quality-diversity
    record on the side without changing its topology -- the "archive + rich
    feedback" point, one config edit away and untried.
    """
    for report in selection.winners:
        insert_into_archive(ctx, state, report)


@register("update_operator", "restart")
def op_restart(ctx: Context, state: RunState, selection: Selection) -> None:
    """Drop the chain and start cold. Published by nobody.

    Gated by `update.restart.on_stagnation` + `update.restart.patience` against
    the stagnation counter `_update_global_best` maintains. With
    `on_stagnation: false` the restart is unconditional -- a memoryless search
    that regenerates from the task spec every round, which is the useful limit
    case for measuring what the chain is worth. `state.best` and the archive
    survive: this restarts the SEARCH, it does not discard the results.
    """
    patience = int(ctx.cfg.get("update.restart.patience", 0) or 0)
    if ctx.cfg.get("update.restart.on_stagnation", False) and state.stagnation < patience:
        return
    log.info("      restart: dropping chain head and dialogue (stagnation=%d)", state.stagnation)
    ctx.event("update_restart", stagnation=state.stagnation, patience=patience,
              dropped=state.latest.cand_id if state.latest else None)
    state.latest = None
    state.dialogue = []
    state.stagnation = 0


@register("update_operator", "none")
def op_none(ctx: Context, state: RunState, selection: Selection) -> None:
    """No variation operator: Text2Reward, L2R, Singh -- single-shot methods
    where nothing is carried forward to vary."""
    return None


# ==========================================================================
# winner_action  --  winner_fn(ctx, state, report) -> None
# ==========================================================================


def _winner_extension_steps(ctx: Context) -> int:
    """`train.winner_extension_fraction` x `train.env_steps`, in env steps.

    THE one reader of that key, and a literal `cfg.get` rather than a composed
    name so an AST search for the key finds it
    (`tests/test_declared_keys_are_read.py`). 0 everywhere but ROSKA.
    """
    frac = float(ctx.cfg.get("train.winner_extension_fraction", 0.0) or 0.0)
    full = int(ctx.cfg.get("train.env_steps", 0) or 0)
    return max(1, int(round(frac * full))) if frac > 0 and full > 0 else 0


@register("winner_action", "become_parent")
def win_become_parent(ctx: Context, state: RunState, report: CandidateReport) -> None:
    """The winner becomes the chain head. Eureka, DrEureka, RDA, GT, CARD.

    Writes `state.latest` -- what `generate.parent_source: latest` reads. This
    is the ungated adoption §6 records for every published method, and under
    CARD it is reached even by a candidate the TPE screen rejected (§2: "either
    way the failed candidate keeps chain headship"), which is a real and
    slightly alarming property of the method rather than an oversight here.
    `state.best` is NOT touched: elitism is the topology's business, and
    conflating the two would quietly turn every method into Eureka.
    """
    state.latest = report


@register("winner_action", "extend_then_become_parent")
def win_extend_then_become_parent(ctx: Context, state: RunState,
                                  report: CandidateReport) -> None:
    """ROSKA: the round's winner trains on, then becomes the chain head.

    "The best-performed policy is selected for an extended training of 2500
    epochs" (appendix.tex:42). That is a per-ITERATION step after selection,
    which is why it lives here rather than in a `post:` phase: `pre`/`post`
    run outside the loop and would spend it once for the whole search.

    **Which policy it continues from is `resume_ref`, and that is a ‡.** The
    paper says "the best-performed policy is selected for an extended
    training" and never says from which parameters -- and there is no released
    implementation to settle it (the URL in appendix.tex:49 is 404; the repo
    that exists carries models only). Continuing from the winner's OWN
    returned checkpoint is the only reading under which the sentence describes
    extended training rather than a fresh run, and it is what `resume_ref`
    means everywhere else in this codebase: "how a round driver hands a
    candidate back its own policy from an earlier call in the same round".

    A failed extension is a WARNING and the unextended winner, never a raise:
    the round has already chosen this candidate, and losing the whole search
    to a failure in a top-up would be the wrong trade. The journal records
    whether it took, because an extension that silently did not happen is a
    different method costing 2500 epochs less.
    """
    steps = _winner_extension_steps(ctx)
    ref = getattr(report.result, "policy_ref", None) if report.result is not None else None
    if steps and ref and report.candidate.trainable:
        backend = registry.get("train_backend", ctx.cfg["train.backend"])
        try:
            extended = backend(ctx, state, report.candidate, n_seeds=1,
                               env_steps=steps, resume_ref=ref,
                               seed_phase="winner_extension")
        except Exception as exc:  # noqa: BLE001 - see the docstring
            log.warning("winner extension failed for %s (%s: %s); the round keeps its "
                        "UNEXTENDED winner", report.cand_id, type(exc).__name__, exc)
            ctx.event("winner_extension", cand_id=report.cand_id, applied=False,
                      env_steps=0, error=f"{type(exc).__name__}: {exc}")
        else:
            new_ref = getattr(extended, "policy_ref", None)
            if new_ref:
                report.result.policy_ref = new_ref
            ctx.event("winner_extension", cand_id=report.cand_id, applied=True,
                      env_steps=int(getattr(extended, "env_steps_used", 0) or 0),
                      requested=steps)
    elif steps:
        ctx.event("winner_extension", cand_id=report.cand_id, applied=False,
                  env_steps=0, error="no policy_ref to continue from")
    win_become_parent(ctx, state, report)


@register("winner_action", "insert_archive")
def win_insert_archive(ctx: Context, state: RunState, report: CandidateReport) -> None:
    """The winner is filed in the quality-diversity archive instead of
    becoming a parent (LIMEN: the archive IS the population, and the next
    parent is sampled from it rather than inherited)."""
    insert_into_archive(ctx, state, report)


@register("winner_action", "both")
def win_both(ctx: Context, state: RunState, report: CandidateReport) -> None:
    """Chain head AND archive cell -- an archive kept alongside a live chain.
    Unpublished; a cheap hybrid."""
    win_become_parent(ctx, state, report)
    win_insert_archive(ctx, state, report)


# ==========================================================================
# loser_action  --  loser_fn(ctx, state, loser_report) -> None
# ==========================================================================


@register("loser_action", "discard")
def lose_discard(ctx: Context, state: RunState, report: CandidateReport) -> None:
    """Eureka: the 15 non-winners leave no trace beyond the run artifact.

    Distinct from `none` below, which records that losers cannot exist at all.
    Both are no-ops in code; only one of them is a CHOICE.
    """
    log.debug("      discard loser %s (fitness=%s)", report.cand_id, report.fitness)


def measured_fitness(report: CandidateReport) -> bool:
    """Does this report's fitness come from a MEASUREMENT?

    True for a candidate that trained cleanly, and for one whose fitness §4
    lifted off the screen's own short-run measurement
    (`evaluate.screened_fitness: screen_measurement`, `meta["fitness_from_
    screen"]`). Tested via FLAGS, never by comparing the value against
    `select.failure_value` -- a sentinel is a convention about losing, not an
    encoding a reader may decode.
    """
    c, res = report.candidate, report.result
    if bool(report.meta.get("fitness_from_screen")):
        return True
    return bool(c.valid and getattr(res, "trained", False) and not getattr(res, "error", ""))


@register("loser_action", "store_as_negative_example")
def lose_store_as_negative_example(ctx: Context, state: RunState,
                                   report: CandidateReport) -> None:
    """LIMEN: FAILED programs plus their error traces are kept as negatives --
    and a trained program that merely lost its cell contest is NOT one of them.

    The release's failure populations are exactly: crash-filter failures /
    no-metrics crashes (`database.py::add` L225-233, `add_failure` then
    `return None`) and trained programs whose fitness is EXACTLY 0.0 (L236-243,
    added to `recent_failures` WITHOUT returning -- so they are also still
    grid/island-placed). A cell loser is neither: "Still store it (it may be
    sampled as a non-grid program)" (L294-299) keeps it in `programs` and its
    island, reachable by the exploration and fitness-weighted parent branches
    and by `get_top_programs`. Appending EVERY non-winner to `failure_memory`
    would quote a respectable trained candidate that lost its niche into every
    later prompt under "These programs were generated earlier and FAILED" with
    "- error: (unrecorded)" -- an inverted training signal -- while it
    simultaneously vanished from the parent pool.

    So, dispatch on the measurement flag (`measured_fitness`) and on NOTHING
    ELSE. Whether a scalar exists is a SEPARATE question, asked only after the
    filing, because it decides one thing only: the release's double
    bookkeeping for an exact 0.0.

      * measured, fitness > 0.0 (or < 0.0 on a reward-type fitness): filed into
        its island's population store, never `failure_memory`;
      * measured, fitness is None: filed the same way. A config with no ranking
        scalar at all (`evaluate.fitness.source: none`, `problem.
        fitness_access: none`) still trains candidates cleanly, and a clean
        trained loser is a clean trained loser whether or not the run computes
        a number to rank it by. `_score` sorts it at NEG_INF -- the unranked
        marker §6 uses everywhere -- so it cannot displace a ranked incumbent,
        is skipped by the fitness-proportional branches and stays drawable by
        the uniform ones. Keying the filing on `fitness is not None` would
        reintroduce the inverted signal above for every config without a
        ranking scalar: a whole run's trained population would re-enter every
        prompt as FAILED with "- error: (unrecorded)" while every counter read
        normal.
      * measured, fitness == 0.0: BOTH -- a negative example AND still
        samplable, the release's own double bookkeeping. `is None` is NOT that
        case and must never be coerced into it: `None` and `0.0` stay distinct
        (CARD ranks nothing at all).
      * unmeasured (invalid, or a screen that measured nothing): the negative-
        example path.

    What that buys, stated exactly: no TRAINED loser reaches `failure_memory`
    without a real failure string, so `generation._failure_traces`'
    "(unrecorded)" rendering never applies to trained losers -- the only
    trained loser that arrives here is
    the exact-0.0 case and it is given "trained and scored exactly 0.0". It is
    NOT true that every entry carries one. An UNMEASURED loser's string comes
    from `report.candidate.failure`, which the parse and verify stages fill but
    a training-time crash does not: a candidate that compiled, passed every
    screen and then raised inside the learner keeps `failure == ""` and
    `failure_kind == ""` (the error is on `result.seed_metrics[*].error`), so
    it is filed under the "lost_selection" fallback and still renders as
    "(unrecorded)". Measured on the tester profile: a `wrong_signature` mock
    candidate raising `TypeError: compute_reward() takes 0 positional
    arguments` does exactly this. That is a separate defect in what the
    failure stages RECORD, not in this dispatch, and it is not fixed here.

    One accepted gap, for the record: a grid incumbent DISPLACED by a fitter
    newcomer is dropped from this dict entirely, where the release keeps it in
    `programs` (though discarded from its island set, so it is not
    island-samplable there either -- only `get_top_programs(island=None)` and
    the stale archive list can still see it).

    `failure_memory` is `List[Dict[str, str]]`, so everything is stringified
    here -- the store is prompt material, not a record to compute over. The two
    failure populations stay apart: `failure_kind` says whether this never compiled
    or a screen rejected it, and the two are never collapsed.
    """
    fitness = report.fitness
    scored_zero = False
    if measured_fitness(report):
        isl = _island_for(ctx, state, report)
        merged = _truncate(_population(state, isl) + [report], _population_size(ctx))
        _set_population(state, isl, merged)
        ctx.event("update_loser_kept", cand_id=report.cand_id, island=isl,
                  fitness=None if fitness is None else float(fitness),
                  in_population=any(r.cand_id == report.cand_id for r in merged))
        # The exact-0.0 question is asked ONLY where a scalar exists. No
        # scalar is not a zero, so an unranked loser returns here like any
        # other filed loser rather than falling through to the FAILED framing.
        if fitness is None or float(fitness) != 0.0:
            return
        scored_zero = True
        # fitness == 0.0 falls through: samplable AND a negative example.
    state.failure_memory.append({
        "cand_id": report.cand_id,
        "iteration": str(report.candidate.iteration),
        "failure": report.candidate.failure
        or ("trained and scored exactly 0.0" if scored_zero else ""),
        "failure_kind": report.candidate.failure_kind or "lost_selection",
        "fitness": "" if report.fitness is None else f"{report.fitness:.6g}",
        "reward_code": report.candidate.reward_code or "",
        "feedback": report.feedback,
    })


@register("loser_action", "insert_archive")
def lose_insert_archive(ctx: Context, state: RunState, report: CandidateReport) -> None:
    """A loser still fills an empty niche.

    This is MAP-Elites proper (§6): losing the round says nothing about
    occupying a cell no one else occupies, and refusing low-fitness losers is
    what turns an archive back into a population.
    """
    insert_into_archive(ctx, state, report)


@register("loser_action", "add_to_preference_dataset")
def lose_add_to_preference_dataset(ctx: Context, state: RunState,
                                   report: CandidateReport) -> None:
    """GT: losers become preference data (§6, `update.loser.action`).

    The label comes from this round's verdict, so `source="selection"` rather
    than the `Preference` default `"vlm"` -- §4's comparator writes its own
    pairs and these must stay distinguishable inside `D_pref` (they are the
    supervision `verify.alignment_filter` later screens against, and mixing
    provenance silently would make GT's TAC cold-start unreadable).

    The preferred side is `state.iteration_best`, which the topology set
    before `bird.py` walks the losers -- that ordering is the reason this
    signature does not need the `Selection`.
    """
    winner = state.iteration_best
    if winner is None or winner.cand_id == report.cand_id:
        return
    pref = Preference(
        left_id=winner.cand_id,
        right_id=report.cand_id,
        label=1,  # 1 = left preferred; the winner is always the left side
        left_traj=winner.result.trajectories[0] if winner.result.trajectories else None,
        right_traj=report.result.trajectories[0] if report.result.trajectories else None,
        source="selection",
        iteration=report.candidate.iteration,
    )
    state.preferences.append(pref)
    # `source: selection`, and the dataset must be able to tell it apart. This
    # one is INFERRED from the ranking rather than judged head to head -- and it
    # can contradict the VLM's own verdict on the same pair -- so a pooled
    # analysis that ignores `source` mixes two different
    # kinds of claim. Recording it with its source is what makes that filterable
    # instead of invisible.
    preference_log.record(ctx, pref, state, judge="inferred")


@register("loser_action", "none")
def lose_none(ctx: Context, state: RunState, report: CandidateReport) -> None:
    """CARD: no losers exist. K=1, so `Selection.losers` is empty and this is
    never called; every iterate persists verbatim in the transcript instead
    (§6). Registered so the config can SAY that, rather than borrowing
    `discard` and implying a choice that was never available."""
    return None


# ==========================================================================
# prompt_mode  --  prompt_fn(ctx, state, selection) -> RunState
#
# The history axis, and no paper ablates it. `update.prompt.mode` is the
# write policy for `state.dialogue`; `generate.history_mode` is §1's read
# policy over the same list.
# ==========================================================================


def _diagnostics(report: CandidateReport) -> str:
    """Training-budget line for the carried turn (seeds, env_steps,
    not-trained/error). Ungated: no existing key governs it and no published
    prompt carries it (a known deviation from Eureka's prompt); GT's eta is
    the per-component value curve, which is §4's
    numeric_reflection (it arrives inside `report.feedback`), not this line."""
    res = report.result
    bits = []
    if res.seed_metrics:
        bits.append(f"seeds={len(res.seed_metrics)}")
    if res.env_steps_used:
        bits.append(f"env_steps={res.env_steps_used}")
    # No `train_s={res.wallclock_s:.1f}` here, deliberately: it would put
    # WALL-CLOCK INTO THE PROMPT, which makes a run irreproducible from `seed`
    # -- the exact failure `_seed_base`'s docstring refuses `hash()` over.
    # Measured on the tester profile (l2r, one seed, PYTHONHASHSEED=0):
    # training takes 0.53-0.58 s, so such a line renders `train_s=0.5` or
    # `train_s=0.6` at about 50/50, the mock LLM hashes its prompt to pick a
    # sample, and the two prompts produce two DIFFERENT rewards -- and
    # therefore different fitness, different winners, different everything
    # downstream -- from a one-character difference in the prompt.
    #
    # It is also not evidence about the reward, which is what this string is
    # for: `env_steps` above already carries the cost, deterministically, and a
    # model told "this took 0.6 s" cannot act on it. Timing belongs in
    # `budget.json` and `seed_metrics[*].wallclock_s`.
    # Keeping it out is a precondition for `train.candidate_parallelism: parallel`
    # being comparable to `sequential` at all, since parallelism changes
    # wall-clock by construction.
    if not res.trained:
        bits.append(f"NOT TRAINED ({res.skip_reason or 'skipped'})")
    if res.error:
        bits.append(f"error={res.error}")
    return ", ".join(bits)


def _turns(ctx: Context, state: RunState, report: CandidateReport) -> List[Dict[str, str]]:
    """The (assistant, user) pair one round contributes to a transcript.

    GT's rule (Alg. 1 line 19) is `prompt := prompt : feedback : R_best :
    eta_best`, where eta_best is per-component training diagnostics
    (neurips_2025.tex:336); the Bradley-Terry strength is never appended
    (:318-320, App. B :618-662), so the scalar written here under
    `evaluate.feedback.state_selection_scalar` (default true) is BIRD's
    default, not GT's prompt -- gt pins false. Eureka pins it
    false too: its published feedback is statistics + tips only, never labels
    the scalar or its aggregation rule, and never states a global best
    (eureka.py:250-267). When shown, that is the
    one place §6 touches `fitness`; it DECIDES nothing with it -- stage 5
    already ruled -- and the 4->5->6 contract (only the scalar in §5, only the
    prose in §6) is about who gets to judge, not about what may be quoted to
    the model.

    `update.prompt.assistant_content` picks what the assistant turn carries:
    the extracted reward code, the model's full raw response including its own
    reasoning prose -- upstream carries the best sample's whole message
    content (eureka.py:331,335) -- or the candidate's stage-1 ENGLISH design
    (`Candidate.nl_spec`, written by `generate.output.two_stage_nl_then_code`
    or `design_thought: inline_brace`). `reward_code` and `raw_response` fall
    back to each other when their slot is empty, so the turn is never blank.

    `nl_spec` is GT's App. B (neurips_2025.tex:627-636): the subsequent-round
    prompt accumulates `{all_english_rewards}` -- every round's English design
    -- and shows `{reward_code}`, the latest program, exactly once. Carrying the
    spec here and the program in PARENT REWARD (`include_parent_code`)
    reproduces that shape; carrying the code does not (K programs, zero
    English, and the parent program twice per prompt).
    `raw_response` is no escape hatch: under two-stage generation it is the
    CODER's reply, never the English. When the spec is empty (a candidate whose
    spec was never written, a resumed or legacy record) the turn falls back to
    the code and the fallback is journalled as `assistant_content_fallback`.

    Every turn is stamped with a bookkeeping key `content_kind` naming what it
    actually holds -- `records/iterNN.json` then shows a fallback for what it
    is, and `generation._clean_turns` strips the key before the wire (the same
    convention as CARD's `valid: False`).
    """
    want = str(ctx.cfg.get("update.prompt.assistant_content", "reward_code"))
    cand = report.candidate
    if want == "nl_spec":
        carried, kind = cand.nl_spec or "", "nl_spec"
        if not carried.strip():
            carried = cand.reward_code or cand.raw_response
            kind = "reward_code" if cand.reward_code else "raw_response"
            log.info("candidate %s has no nl_spec; the carried assistant turn falls "
                     "back to its %s", report.cand_id, kind)
            ctx.event("assistant_content_fallback", cand_id=report.cand_id,
                      iteration=cand.iteration, wanted="nl_spec", carried=kind)
    elif want == "raw_response":
        carried = cand.raw_response or cand.reward_code
        kind = "raw_response" if cand.raw_response else "reward_code"
    else:
        carried = cand.reward_code or cand.raw_response
        kind = "reward_code" if cand.reward_code else "raw_response"
    turns = [{"role": "assistant", "content": carried, "content_kind": kind}]
    lines: List[str] = []
    if ctx.cfg.get("evaluate.feedback.state_selection_scalar", True):
        if report.fitness is not None:
            lines.append(f"fitness ({report.fitness_source}): {report.fitness:.6g}")
        if state.best is not None and state.best.fitness is not None:
            lines.append(f"best so far: {state.best.fitness:.6g} ({state.best.cand_id})")
    diag = _diagnostics(report)
    if diag:
        lines.append(f"training: {diag}")
    if report.feedback:
        lines.append(report.feedback)
    turns.append({"role": "user", "content": "\n".join(lines)})
    return turns


def _enforce_token_cap(ctx: Context, state: RunState) -> None:
    """Honour `update.prompt.max_length_tokens` by dropping oldest turns.

    ~4 characters per token: there is no tokenizer offline, and a declared key
    the loop cannot honour would silently do nothing. Oldest-first is
    the only truncation that preserves the recency a reflection prompt needs.
    """
    cap = int(ctx.cfg.get("update.prompt.max_length_tokens", 0) or 0)
    if cap <= 0 or not state.dialogue:
        return
    budget = cap * 4

    def _size() -> int:
        return sum(len(m.get("content", "")) for m in state.dialogue)

    while len(state.dialogue) > 1 and _size() > budget:
        state.dialogue.pop(0)
    if _size() > budget:
        last = state.dialogue[-1]
        keep = max(budget - 64, 0)
        last["content"] = (last.get("content", "")[:keep]
                           + "\n... [truncated to update.prompt.max_length_tokens]")


@register("prompt_mode", "append")
def prompt_append(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """GT / CARD: the transcript accumulates forever.

    GT (Alg. 1 line 19): `prompt := prompt : feedback : R_best : eta_best` --
    every round's turn stays in the prompt. App. B (neurips_2025.tex:627-636)
    accumulates the English designs and shows the latest code once: that is
    `update.prompt.assistant_content: nl_spec` (configs/methods/gt.yaml), under which
    `_turns` carries the stage-1 spec and PARENT REWARD alone carries the
    program.

    CARD is `append` with a twist that lives right here: only VALIDATED turns
    ever enter its transcript -- its compile failures leave no record (§4.3 of
    the paper, and `generate.history_mode: full_dialogue` in §1). That is the
    stated mechanism for catching negative optimization: the model is shown
    the sequence of things that RAN, so a regression is visible as a
    regression instead of being drowned in syntax errors. So the gate is
    `candidate.valid` and NOT `candidate.trainable`: a candidate the quality
    screen rejected compiled fine and stays in the transcript -- which is the
    same asymmetry that lets CARD's chain head advance after a screen failure.
    """
    winner = selection.winner
    if winner is None:
        return state
    if not winner.candidate.valid:
        log.debug("      prompt.append: %s never compiled -- leaving no trace in the "
                  "transcript", winner.cand_id)
        return state
    state.dialogue = list(state.dialogue) + _turns(ctx, state, winner)
    _enforce_token_cap(ctx, state)
    return state


@register("prompt_mode", "replace")
def prompt_replace(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """RDA: Markov-1. The prompt is rebuilt each round and remembers nothing else.

    Rebuilt from instruction, environment, subtask list, best reward and best
    summary (§6). What is written here is the carried MEMORY, not the finished
    prompt -- §1 composes the live message from this plus its own
    `generate.context.*` sections. Keeping it to one turn is the whole point:
    after this call there is exactly one thing to inherit.
    """
    cfg = ctx.cfg
    winner = selection.winner
    best = state.best or winner

    parts = [f"Task: {cfg.get('problem.task_description', '')}",
             f"Environment: {cfg.get('problem.env_id', '')}"]
    if state.subtasks:
        parts.append("Subtasks:\n" + "\n".join(
            f"  {i + 1}. {s}" for i, s in enumerate(state.subtasks)))
    if best is not None:
        if best.fitness is not None:
            parts.append(f"Best reward so far: {best.cand_id} (fitness {best.fitness:.6g})")
        if best.candidate.reward_code:
            parts.append("Best reward function:\n" + best.candidate.reward_code)
        summary = best.feedback or (winner.feedback if winner is not None else "")
        if summary:
            parts.append("Summary of the last round:\n" + summary)

    state.dialogue = [{"role": "user", "content": "\n\n".join(p for p in parts if p)}]
    _enforce_token_cap(ctx, state)
    return state


@register("prompt_mode", "sliding_window")
def prompt_sliding_window(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """Eureka: append, then hard-cap the transcript at `generate.history_max_turns`.

    Eureka's repo caps the dialogue at 4 MESSAGES (§1), and its App. D argues
    for Markov-1 from context limits -- the exact opposite of CARD's published
    rationale for keeping everything. The cap is read from the §1 key on
    purpose: it is the same knob seen from the write side, and a second §6 copy
    of it would be free to disagree with the one §1 reads.

    No validity gate here, unlike `append`: under `loop.on_total_failure:
    retry_iteration` an all-invalid round never reaches §6 at all, so the case
    is unreachable, and leaving it out keeps this a pure last-K rule.
    """
    winner = selection.winner
    if winner is None:
        return state
    messages = list(state.dialogue) + _turns(ctx, state, winner)
    cap = int(ctx.cfg.get("generate.history_max_turns", 0) or 0)
    if cap > 0:
        messages = messages[-cap:]
    state.dialogue = messages
    _enforce_token_cap(ctx, state)
    return state


@register("prompt_mode", "reset")
def prompt_reset(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """No history at all: LIMEN, Text2Reward, L2R, Singh.

    LIMEN is the interesting one -- it is not memoryless, it keeps its state in
    the ARCHIVE rather than in the prompt (`generate.context.include_archive_
    elites` + `include_failure_traces`), which is why `loop.carry` for it is
    `[archive, failure_memory]` and not `[dialogue]`. Clearing explicitly
    rather than leaving the list alone is what makes that true even when
    `dialogue` is (wrongly) named in `loop.carry`.
    """
    state.dialogue = []
    return state


# ==========================================================================
# feedback_routing  --  routing_fn(ctx, state, selection) -> RunState
# ==========================================================================


@register("feedback_routing", "fixed")
def route_fixed(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """Every method except CARD: §4 chose the channel and §6 uses it as-is.

    The channel is still published onto `state.feedback_channel` so §1 can say
    which kind of feedback it is quoting.
    """
    winner = selection.winner
    state.feedback_channel = winner.feedback_channel if winner is not None else "none"
    return state


def _first(record: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in record and record[k] is not None:
            return record[k]
    return default


def _tpe_verdict(candidate: Candidate) -> Tuple[Optional[bool], Dict[str, Any]]:
    """Read the screen's verdict off the candidate. `None` = no verdict exists.

    §2 owns the screen; §6 only reads what it recorded, and probes a few key
    spellings rather than dictating one, since the record's shape is the
    screen's business. `None` matters: with no TPE record and no screen
    rejection there is nothing to route on, and inventing a verdict here would
    turn "the screen was off" into "the candidate passed".
    """
    for rec in candidate.verify_records:
        name = str(_first(rec, "check", "name", "screen", "kind", default="")).lower()
        if "tpe" not in name:
            continue
        for key in ("passed", "pass", "ok", "order_preserving"):
            if key in rec:
                return bool(rec[key]), rec
        return (not candidate.screened_out), rec
    if candidate.screened_out or candidate.failure_kind == "screened":
        return False, {}
    return None, {}


def _preference_report(ctx: Context, state: RunState, report: CandidateReport,
                       record: Dict[str, Any]) -> str:
    """CARD's fail-channel message: order accuracy, the most-violating pair
    quoted under THIS candidate's reward, and both example trajectories in the
    full trajectory-feedback format.

    The evidence is the screen's
    own `meta['tpe']` payload: the release selects the (min-success,
    max-failure) example pair from per-step averages computed by RE-RUNNING the
    failing candidate's reward over the stored trajectories, quotes those
    re-scored aggregates, and appends the full convert_traj_to_str rendering of
    both trajectories (utils.py:262-326; §4.2.3 "if the return of successful
    trajectories exceeds that of unsuccessful ones when computed using the
    current reward function"; §4.4 "provides details of two trajectories in
    the trajectory feedback format"). Selecting and quoting
    `Trajectory.mean_per_step_return` instead would quote the reward recorded
    at COLLECTION time -- a previous candidate's -- while telling the model
    "a SUCCESS your reward scores BELOW a FAILURE".

    The renderings' `reward=` channel carries the candidate's re-scored
    per-step values (`rewards_override`); component/observation channels stay
    the stored ones, as the release renders them. That is one notch MORE
    internally consistent than the release, whose renderings show the stored
    reward channels verbatim -- re-importing the collection-time mislabel at
    the per-step level -- and it matches the paper's "computed using the current
    reward function".
    """
    meta = report.candidate.meta.get("tpe") or {}

    lines = [
        "TPE verdict: FAIL -- this reward does not order successful trajectories above "
        "failed ones.",
        "This report REPLACES the usual process and trajectory feedback: the channel is "
        "exclusive, not additive.",
    ]

    rule = _first(meta, "rule", default=_first(
        record, "rule", default=ctx.cfg.get("verify.tpe.rule", "pair_accuracy")))
    threshold = _first(meta, "threshold", default=_first(
        record, "threshold", default=ctx.cfg.get("verify.tpe.threshold")))
    if meta:
        score = meta.get("score")
        if score is None:
            # The refusal case: no ordering statistic was computable. None is
            # not 0.0 and must not be verbalised as an accuracy of zero.
            lines.append(f"order accuracy: not computable -- "
                         f"{meta.get('detail', 're-scoring failed on the stored trajectories')} "
                         f"(rule={rule}, threshold={threshold})")
        else:
            lines.append(f"order accuracy: {float(score):.1%} "
                         f"(rule={rule}, threshold={threshold})")
        n_err = int(meta.get("n_success_errors") or 0) + int(meta.get("n_failure_errors") or 0)
        if n_err:
            lines.append(f"note: your reward raised an exception or returned a non-finite "
                         f"value on {n_err} stored trajector{'y' if n_err == 1 else 'ies'}; "
                         f"an errored success counts as incorrectly ordered.")
    else:
        accuracy = _first(record, "accuracy", "pair_accuracy", "score", "value")
        if accuracy is not None:
            lines.append(f"order accuracy: {float(accuracy):.1%} "
                         f"(rule={rule}, threshold={threshold})")
        else:
            lines.append("order accuracy: unavailable (the screen recorded no score).")

    example = meta.get("example_pair") if meta else None
    if example:
        s, f = example["success"], example["failure"]
        margin = f["mean_per_step"] - s["mean_per_step"]
        if margin > 0:
            lines.append("most-violating pair -- a SUCCESS your reward scores BELOW a "
                         "FAILURE (both re-scored with YOUR reward function over the "
                         "stored trajectories):")
        else:
            lines.append("closest pair under your reward (no single inversion; the "
                         "aggregate rule is what failed):")
        # The release's three quantities per side (utils.py:323-326): re-scored
        # return, length, average reward per step.
        lines.append(
            f"For example, this is a trajectory where the agent successfully solved the "
            f"task, with a return of {s['return']:.6g}, a length of {s['length']}, and an "
            f"average reward per step of {s['mean_per_step']:.6g}:")
        lines.append(_rendered_store_example(ctx, state, s, "successful trajectory"))
        lines.append(
            f"However, the following shows a trajectory where the agent failed to solve "
            f"the task, with a return of {f['return']:.6g}, a length of {f['length']}, and "
            f"an average reward per step of {f['mean_per_step']:.6g}:")
        lines.append(_rendered_store_example(ctx, state, f, "failed trajectory"))
    elif meta:
        lines.append("no example pair is quotable: re-scoring with your reward left no "
                     "usable (success, failure) pair among the stored trajectories.")
    else:
        # Degraded fallback -- no screen payload at all (a hand-built record, or
        # a screen that wrote no `meta['tpe']`). A selection over stored
        # collection-time rewards, LABELLED HONESTLY: these numbers were
        # produced by the reward in force when each trajectory was collected,
        # not by the candidate under review, so no inversion claim is made.
        trajs: Sequence[Trajectory] = list(state.trajectory_store)
        successes = [t for t in trajs if t.success]
        failures = [t for t in trajs if not t.success]
        if successes and failures:
            worst_success = min(successes, key=lambda t: t.mean_per_step_return)
            best_failure = max(failures, key=lambda t: t.mean_per_step_return)
            lines.append("closest stored pair, by the reward recorded at collection time "
                         "(NOT your reward -- the screen left no re-scored values, so no "
                         "single inversion can be exhibited):")
            lines.append(f"  success rollout: mean per-step reward "
                         f"{worst_success.mean_per_step_return:.6g} over "
                         f"{worst_success.length} steps")
            lines.append(f"  failure rollout: mean per-step reward "
                         f"{best_failure.mean_per_step_return:.6g} over "
                         f"{best_failure.length} steps")
    lines.append("Revise the reward so that every successful trajectory outscores every "
                 "failed one.")
    return "\n".join(lines)


def _rendered_store_example(ctx: Context, state: RunState, side: Dict[str, Any],
                            label: str) -> str:
    """Render one side of the example pair in the trajectory-feedback format
    with the candidate's re-scored per-step rewards on the `reward=`
    channel. Fetch is by `store_index`, bounds-checked: a missing index
    degrades to a stated line, never to a wrong trajectory."""
    # Local import, not module-level: keeps §4's module out of §6's import
    # surface except on CARD's fail channel. No cycle either way (evaluation
    # imports judgments/observability/registry/types only).
    from .evaluation import (_declared_state_fields, _show_task_metric, _traj_stride,
                             render_trajectory)
    idx = side.get("store_index")
    store = state.trajectory_store or []
    if not isinstance(idx, int) or not (0 <= idx < len(store)):
        return (f"  [{label}] (stored trajectory #{idx} is no longer available; "
                f"its re-scored per-step rewards were: "
                f"{', '.join(f'{r:.4g}' for r in side.get('per_step', []))})")
    return render_trajectory(store[idx], label, stride=_traj_stride(ctx),
                             horizon=getattr(ctx.env, "horizon", None),
                             rewards_override=side.get("per_step"),
                             show_success=_show_task_metric(ctx),
                             state_fields=_declared_state_fields(ctx))


@register("feedback_routing", "tpe_verdict")
def route_tpe_verdict(ctx: Context, state: RunState, selection: Selection) -> RunState:
    """CARD: the screen's verdict switches the CHANNEL, exclusively (§6).

    Pass -> the process + trajectory feedback §4 already built. Fail -> a
    preference report INSTEAD: order accuracy plus the most-violating
    success/failure pair. Not additive -- the failing branch overwrites
    `report.feedback` rather than appending to it, which is what makes this a
    routing decision and not another feedback builder.

    Same score-conditioned-prompt family as LIMEN's banded guidance, at a
    coarser condition (one bit rather than a band). Candidates with no verdict
    are left exactly as §4 wrote them.
    """
    for report in list(selection.winners) + list(selection.losers):
        passed, record = _tpe_verdict(report.candidate)
        if passed is None:
            continue
        if passed:
            # The prose is §4's; routing only names the channel it belongs to.
            report.feedback_channel = "process+trajectory"
        else:
            report.feedback = _preference_report(ctx, state, report, record)
            report.feedback_channel = "preference"
        ctx.event("update_routing", cand_id=report.cand_id, verdict="pass" if passed else "fail",
                  channel=report.feedback_channel)

    winner = selection.winner
    state.feedback_channel = winner.feedback_channel if winner is not None else "none"
    return state


# ==========================================================================
# archive parent sampling (`update.archive.parent_sampling`)
#
# Exported for §1's `generate.parent_source: archive_sample`; the key is a §6
# key, so the branches live here.
# ==========================================================================


def _pick_island(ctx: Context, cells: Sequence[ArchiveCell]) -> int:
    return ctx.rng.choice(sorted({c.island for c in cells}))


def _fitness_proportional(ctx: Context, cells: Sequence[ArchiveCell]) -> Optional[ArchiveCell]:
    """Roulette wheel over fitness, shifted so the minimum weighs ~0.

    Unranked (-inf) cells cannot be weighted at all -- they fall back to
    uniform, which is the only defined behaviour when a config combines an
    archive with `fitness_access: none`.
    """
    if not cells:
        return None
    finite = [c for c in cells if c.fitness > NEG_INF and c.fitness == c.fitness]
    if not finite:
        return ctx.rng.choice(list(cells))
    lo = min(c.fitness for c in finite)
    weights = [(c.fitness - lo) + 1e-9 for c in finite]
    total = sum(weights)
    if total <= 0:
        return ctx.rng.choice(finite)
    x = ctx.rng.random() * total
    acc = 0.0
    for cell, w in zip(finite, weights):
        acc += w
        if x <= acc:
            return cell
    return finite[-1]


def _branch_global_uniform(ctx: Context, cells: Sequence[ArchiveCell]) -> Optional[ArchiveCell]:
    return ctx.rng.choice(list(cells)) if cells else None


def _branch_global_fitness_proportional(ctx: Context,
                                        cells: Sequence[ArchiveCell]) -> Optional[ArchiveCell]:
    return _fitness_proportional(ctx, cells)


def _branch_island_uniform(ctx: Context, cells: Sequence[ArchiveCell]) -> Optional[ArchiveCell]:
    if not cells:
        return None
    island = _pick_island(ctx, cells)
    return ctx.rng.choice([c for c in cells if c.island == island])


def _branch_island_fitness_weighted(ctx: Context,
                                    cells: Sequence[ArchiveCell]) -> Optional[ArchiveCell]:
    if not cells:
        return None
    island = _pick_island(ctx, cells)
    return _fitness_proportional(ctx, [c for c in cells if c.island == island])


def _pick_island_avg_weighted(ctx: Context, cells: Sequence[ArchiveCell]) -> int:
    """Roulette over the demes, weighted by each deme's AVERAGE occupant fitness.

    REvolve's first stage: `P ~ WeightedSample(D, {sigma^P_j}_{j=1..I})`,
    Alg. 1 line 11, p.4, "weighted sampling on island average fitness" (§3.2,
    p.5); `refs/code/Revolve/rewards_database.py:235-241` builds exactly those
    weights out of `Island.average_fitness_score`
    (`refs/code/Revolve/evolutionary_utils/entities.py:176`). It is the half of
    the island model that exploits: `_pick_island`'s uniform draw parents a
    dead deme as often as the best one, and a REvolve island is supposed to
    earn its share of the K offspring.

    The shift is `_fitness_proportional`'s -- minimum weighs ~0, plus 1e-9 --
    deliberately, and not a second convention: two roulettes in one module that
    disagreed about how a negative fitness is handled would not be comparable,
    and a config states `parent_sampling` weights over both at once.

    ‡ FIDELITY GAP. REvolve's wheel is a SOFTMAX, `exp(m/T) / sum(exp(m/T))`
    (`refs/code/Revolve/rewards_database.py:12-14`), under a temperature
    schedule that is a shipped no-op: `cfg/generate.yaml:19-20` sets
    `initial_temp: 1` and `final_temp: 1`, so `utils.py:363`'s linear schedule
    returns 1 at every generation. The paper says only "weighted sampling", so
    neither shape is contradicted by it -- but the two do not weigh the same
    demes the same way, and this is not the paper's rule made explicit. The
    measured difference is at the bottom of the wheel: the shift puts the WORST
    deme at 1e-9, i.e. effectively excluded (3000 draws across five demes with
    means 0.05 / 0.15 / 0.0 / 0.15 / 9.25 gave the 0.0 deme none), where a
    softmax at T=1 leaves it a small positive share. `_fitness_proportional`
    has always had that property for individuals; this is the same convention
    applied to demes, and preferring it over a second convention is the trade.
    A second divergence follows from the shift and is likewise recorded: a deme
    with no finite occupant takes the MINIMUM mean here, i.e. weight 1e-9 and
    reachable only when every deme is equal, where the release hands an empty
    island `mean([-sys.maxsize - 1])` (`entities.py:166-168`) and the softmax
    underflows it to exactly 0. -inf cannot be a shift origin -- it poisons
    every weight on the wheel -- which is why the minimum is the floor.
    """
    islands = sorted({c.island for c in cells})
    finite: Dict[int, List[float]] = {}
    for cell in cells:
        # Same test as `_fitness_proportional`: NEG_INF is `_score`'s unranked
        # marker and `f == f` rejects NaN. An unranked occupant does not drag
        # its deme's average down; it simply does not vote.
        if cell.fitness > NEG_INF and cell.fitness == cell.fitness:
            finite.setdefault(cell.island, []).append(cell.fitness)
    means = {isl: sum(v) / len(v) for isl, v in finite.items()}
    if not means:
        return ctx.rng.choice(islands)
    lo = min(means.values())
    weights = [(means.get(isl, lo) - lo) + 1e-9 for isl in islands]
    total = sum(weights)
    if total <= 0:
        return ctx.rng.choice(islands)
    x = ctx.rng.random() * total
    acc = 0.0
    for isl, weight in zip(islands, weights):
        acc += weight
        if x <= acc:
            return isl
    return islands[-1]


def _branch_island_avg_weighted(ctx: Context,
                                cells: Sequence[ArchiveCell]) -> Optional[ArchiveCell]:
    """REvolve's two-stage draw: the deme by its average, then a member by its
    own fitness (Alg. 1 line 11, p.4; `rewards_database.py:235-266`)."""
    if not cells:
        return None
    island = _pick_island_avg_weighted(ctx, cells)
    return _fitness_proportional(ctx, [c for c in cells if c.island == island])


#: Branch name -> sampler. The config's dict KEYS choose which of these run:
#: † LIMEN's paper and code disagree on the branch DEFINITIONS, not only the
#: weights (paper: 70% global fitness-proportional / 30% island uniform; code:
#: 0.7 uniform-over-elites / 0.2 island-uniform / 0.1 island-fitness-weighted),
#: so neither reading may be hardcoded -- a config states which branches exist
#: and this table supplies them. `island_avg_weighted` is REvolve's, and is the
#: only branch either paper weights the DEMES with (Alg. 1 line 11, p.4).
_PARENT_BRANCHES: Dict[str, Callable[[Context, Sequence[ArchiveCell]], Optional[ArchiveCell]]] = {
    "global_uniform": _branch_global_uniform,
    "global_fitness_proportional": _branch_global_fitness_proportional,
    "island_uniform": _branch_island_uniform,
    "island_fitness_weighted": _branch_island_fitness_weighted,
    "island_avg_weighted": _branch_island_avg_weighted,
}

#: Spelling aliases only. These map names for the SAME branch (the code calls
#: the global-uniform branch "uniform over elites"); nothing here papers over
#: the † definitional disagreement above.
_BRANCH_ALIASES = {
    "uniform_over_elites": "global_uniform",
    "elite_uniform": "global_uniform",
    "uniform": "global_uniform",
    "fitness_proportional": "global_fitness_proportional",
    "global_fitness_weighted": "global_fitness_proportional",
    "island_fitness_proportional": "island_fitness_weighted",
    "island_average_weighted": "island_avg_weighted",
}


def _canonical_branch(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    return _BRANCH_ALIASES.get(key, key)


def _branch_pool(ctx: Context, branch: str,
                 cells: Sequence[ArchiveCell]) -> List[ArchiveCell]:
    """What each branch is allowed to draw from.

    The GLOBAL branches draw from `exploitation_pool` -- the release's 0.7
    branch reads `self.archive`, the archive_size-capped top-fitness LIST
    (`database.py::_sample_exploitation` L341-346), never the grid or the
    non-grid programs. The ISLAND branches draw from everything the chosen
    island holds, grid cells AND `_ELITE` population slots -- the release's
    `islands[i]` sets hold every stored program, which is what makes a filed
    cell loser a reachable parent (L294-299, L331-364). Empty exploitation
    pool (no grid cells yet) degrades to all cells rather than to no parent.
    """
    if branch.startswith("island"):
        return list(cells)
    return exploitation_pool(ctx, cells) or list(cells)


def _draw_branch(ctx: Context,
                 cells: Sequence[ArchiveCell]) -> Tuple[str, List[ArchiveCell]]:
    """One spin of the `update.archive.parent_sampling` wheel: which branch,
    and which cells that branch is allowed to draw from.

    ONE copy, read by both `sample_archive_parent` and
    `sample_archive_parents`, so the single-parent and cohort paths cannot come
    to disagree about what a branch name means. Two copies of one rule drift
    apart, which is also why §1 calls this module's sampler instead of
    keeping its own.

    An unknown branch raises before any `ctx.rng` is touched: the two published
    readings disagree about which branches EXIST, so a typo must not quietly
    become the other paper's method, and a validation that happened after the
    draw would make the error depend on the weights.

    The unweighted default -- the `{}` every non-archive config inherits -- is
    uniform over EVERY occupied cell, and deliberately NOT over
    `_branch_pool("global_uniform", ...)`'s exploitation pool. With no branch
    named there is no branch whose pool rule could apply, and narrowing it here
    would silently change what `parent_source: archive_sample` means for every
    config that never stated one. It also consumes no `ctx.rng`, so the
    default's draw stream does not depend on the key.
    """
    weights = dict(ctx.cfg.get("update.archive.parent_sampling") or {})
    branches: List[Tuple[str, float]] = []
    for raw, weight in weights.items():
        key = _canonical_branch(raw)
        if key not in _PARENT_BRANCHES:
            raise ConfigError(
                f"update.archive.parent_sampling: unknown branch {raw!r}. "
                f"Known branches: {sorted(_PARENT_BRANCHES)}; "
                f"accepted aliases: {sorted(_BRANCH_ALIASES)}")
        branches.append((key, float(weight)))

    total = sum(w for _, w in branches if w > 0)
    if total <= 0:
        return "global_uniform", list(cells)

    x = ctx.rng.random() * total
    acc = 0.0
    for key, weight in branches:
        if weight <= 0:
            continue
        acc += weight
        if x <= acc:
            return key, _branch_pool(ctx, key, cells)
    key = branches[-1][0]
    return key, _branch_pool(ctx, key, cells)


def sample_archive_parent(ctx: Context, state: RunState) -> Optional[CandidateReport]:
    """Draw one parent from the archive per `update.archive.parent_sampling`.

    LIMEN (§6). The dict is a weighting over BRANCHES: each key names a
    sampling rule and its value is that rule's share. An unknown key is a hard
    ConfigError rather than a silent fallback -- the whole point of reading the
    keys is that the two published readings disagree about which branches
    exist, so a typo must not quietly become the other paper's method.

    An empty dict (the default) is uniform over occupied cells. Consumes only
    `ctx.rng`, so a run is reproducible from `seed` alone. §1's
    `parent_source: archive_sample` calls this -- it is THE sampler, one copy
    (two samplers over one archive would drift apart just as two binnings over
    one grid would).

    N INDEPENDENT calls give N independent draws -- a fresh branch, and for the
    island branches a fresh DEME, every time. That is what
    `parent_source: archive_sample` does for `generate.n_parents > 1`, and it
    is a different method from `sample_archive_parents` below; see there.
    """
    cells = [c for c in state.archive.values() if c.report is not None]
    if not cells:
        return None
    key, pool = _draw_branch(ctx, cells)
    cell = _PARENT_BRANCHES[key](ctx, pool)
    return cell.report if cell else None


def sample_archive_parents(ctx: Context, state: RunState,
                           n: int) -> List[CandidateReport]:
    """Draw a COHORT of up to `n` distinct parents -- from ONE deme.

    REvolve (Alg. 1 line 11, p.4). The paper samples one island `P` per
    offspring and then indexes `D[P]` for both `Mutate` (1 parent) and
    `Crossover` (2 parents), so a crossover recombines two members of the SAME
    sub-population; the release is unambiguous about it --
    `refs/code/Revolve/rewards_database.py:251-266` is "STEP 1: sample an
    island" once, then `np.random.choice(range(sampled_island.size), ...,
    replace=False)` for `num_in_context_samples` members *of that island*.
    Recombining across demes would make the islands one panmictic population
    and delete the diversity claim the topology exists for.

    This is why it is a separate entry point rather than a loop over
    `sample_archive_parent`: that loop re-spins the branch wheel per parent and
    each island branch re-picks its deme, so with `generate.n_parents: 2` a
    crossover routinely gets parent A out of island 0 and parent B out of
    island 2. Both behaviours are wanted -- LIMEN's per-parent draw and
    REvolve's cohort draw -- so both exist, as two `generate.parent_source`
    values over one branch table, and neither is anybody's silent default.

    HOW THE DEME IS FIXED: by the first draw, not by a second copy of the
    island rule. The branch's own first pick already carries the branch's own
    island law (uniform for `island_uniform`, average-weighted for
    `island_avg_weighted`), so confining the pool to that cell's island and
    then re-running the SAME branch function for the rest is the only version
    with one copy of each rule in it. The re-run spends one redundant `ctx.rng`
    draw picking a deme out of a one-deme pool; that is the price of not
    writing the second copy, and it is deterministic.

    DISTINCTNESS is by `cand_id` and by removal, matching the release's
    `replace=False` -- and, unlike the release, without a retry loop: the pool
    only shrinks, so the draw terminates whether or not the deme is big enough.
    REvolve's own author left `# TODO: getting trapped in the while loop in the
    initial phases` at `rewards_database.py:254` against exactly this case, and
    at 13 islands and 16 offspring a deme with fewer than 2 members is the
    modal case in early generations, not a corner. A short cohort is returned
    short; it is the caller that decides whether a one-parent crossover is
    still a crossover.

    `ctx.counters["last_sampled_island"]` is left holding the deme this cohort
    came from (`None` for the global branches, and cleared on entry so that a
    caller can never read the PREVIOUS cohort's island off an empty draw --
    a stale integer there is indistinguishable from a measured one in
    `records/iterNN.json`). `ctx.counters` is the run-scoped facility the
    migration cadence and `archive_cap_negative_warned` already use.
    """
    ctx.counters["last_sampled_island"] = None
    cells = [c for c in state.archive.values() if c.report is not None]
    n = int(n)
    if not cells or n <= 0:
        return []

    key, pool = _draw_branch(ctx, cells)
    first = _PARENT_BRANCHES[key](ctx, pool)
    if first is None:
        return []
    drawn = [first]
    if key.startswith("island"):
        # `_branch_pool` already hands the island branches every cell; the
        # cohort narrows that to the one deme the first draw landed in.
        ctx.counters["last_sampled_island"] = int(first.island)
        pool = [c for c in pool if c.island == first.island]

    while len(drawn) < n:
        taken = {c.report.cand_id for c in drawn}
        remaining = [c for c in pool if c.report.cand_id not in taken]
        if not remaining:
            break
        cell = _PARENT_BRANCHES[key](ctx, remaining)
        if cell is None:
            break
        drawn.append(cell)
    return [c.report for c in drawn]
