"""LIMEN's archive bookkeeping, migrant lineage, migration clock and crash retry.

One mechanism each, all of whose failures render as plausible output:

* Insert counting -- one archive win must count once. LIMEN's config reaches
  `insert_into_archive` through the topology, `winner.action: insert_archive`
  AND `operator: archive_insert`; if each moved `archive_inserts` and ran the
  migration check, `migration_interval: 20` would fire every ~7 wins.
* Migrant lineage -- a migrant's offspring must stay on the destination
  island. Filing the SAME report object in two demes would let the sampler
  return it with no memory of which cell it came from, and `_island_for` would
  find the source cell first.
* Crash repair -- `max_repair_attempts: 1` must be live. `on_failure:
  teach_next_iteration` settles the slot at the first failure, so pairing it
  with a repair cap would skip the one same-iteration crash retry the release
  makes (`controller.py:423-448`).
* Taught traceback -- the prompt must read the key the writer stores, or it
  renders "(unrecorded)".
* Migration clock -- LIMEN pins the release's `island_generations` clock.
"""
from __future__ import annotations

import random

import pytest

from bird import registry
from bird.budget import Budget
from bird.components import generation, update, verification
from bird.config import ConfigError, load
from bird.context import Context
from bird.envs.toy import ToyReacher
from bird.state import ArchiveCell, RunState
from bird.types import Candidate, CandidateReport, Selection, TrainResult

_GOOD = ("def compute_reward(s, a, s2):\n"
         "    import numpy as np\n"
         "    o = np.asarray(s2, dtype=float)\n"
         "    return float(-np.linalg.norm(o[0:2] - o[2:4]))\n")
_BAD = "def compute_reward(state):\n    invalid =\n"


def _ctx(profile=None, **overrides) -> Context:
    registry.load_all()
    cfg = load("limen", profile=profile, overrides=overrides)
    ctx = Context(cfg=cfg, budget=Budget(), env=ToyReacher(), rng=random.Random(0))
    return ctx


def _report(cid: str, fitness: float, *, iteration: int = 0, parent_id=None, meta=None,
            cand_meta=None) -> CandidateReport:
    cand = Candidate(cid, iteration, _GOOD, parent_id=parent_id, meta=dict(cand_meta or {}))
    return CandidateReport(cid, cand, TrainResult(cid, cand), fitness=fitness,
                           meta=dict(meta or {}))


# ------------------------------------------------------------- insert counting

def test_one_archive_win_is_counted_once_however_many_paths_offer_it(monkeypatch):
    ctx = _ctx()
    assert ctx.cfg["update.topology"] == "archive_map_elites"
    assert ctx.cfg["update.winner.action"] == "insert_archive"
    assert ctx.cfg["update.operator"] == "archive_insert", "all three paths, as published"
    migrations = []
    real = update._maybe_migrate
    monkeypatch.setattr(update, "_maybe_migrate",
                        lambda ctx, state, capacity=None: migrations.append(1) or real(ctx, state, capacity))
    state = RunState(iteration=0)
    winner = _report("w", 0.5)

    update.topo_archive_map_elites(ctx, state, Selection(winners=[winner]))

    occupied = [c for c in state.archive.values() if c.report is winner]
    assert len(occupied) == 1, "the grid write is idempotent"
    assert ctx.counters["archive_inserts"] == 1, "one win, three paths, counted once"
    assert len(migrations) == 1, "and the migration check runs once"


# ------------------------------------------------------------- migrant lineage

def test_a_migrant_carries_its_destination_island_and_so_do_its_offspring():
    # This is about migrant LINEAGE, independent of which clock triggers the event.
    # LIMEN pins the release's `island_generations` clock (tested below), under which a bare
    # `archive_inserts = interval` does NOT fire; force the `global_inserts` variant so
    # this test exercises the ring at a known insertion.
    ctx = _ctx(**{"update.archive.migration_clock": "global_inserts"})
    interval = int(ctx.cfg["update.archive.migration_interval"])
    assert ctx.cfg["update.archive.n_islands"] >= 2 and interval > 0
    state = RunState(iteration=3)
    src = _report("elite", 0.9, meta={"archive_island": 0, "archive_coords": (0, 0)})
    state.archive[(0, 0, 0)] = ArchiveCell(coords=(0, 0), island=0, report=src, fitness=0.9)
    ctx.counters["archive_inserts"] = interval  # the next check fires

    update._maybe_migrate(ctx, state, None)

    dest = state.archive.get((1, 0, 0))
    assert dest is not None and dest.report is not None, "ring: island 0 -> island 1"
    migrant = dest.report
    assert migrant is not src, "a destination identity, not the source object filed twice"
    assert migrant.cand_id == src.cand_id, "the same program"
    assert migrant.meta["archive_island"] == 1 and migrant.meta["migrated_from"] == 0
    assert src.meta["archive_island"] == 0, "the source copy is untouched"

    # §1 builds an offspring from the migrant: the sampled deme rides along ...
    backend = lambda ctx, state, messages, n: ["```python\n" + _GOOD + "```"] * n  # noqa: E731
    child = generation._build_candidate(
        ctx, state, backend, [{"role": "user", "content": "x"}], raw=None,
        parent_id=migrant.cand_id, parent_code=migrant.candidate.reward_code,
        meta={"sampling_mode": "iid_parallel", "sample_index": 0}, parents=[migrant])
    assert child.meta["parent_island"] == 1

    # ... and §6 files it there, not in the island the parent left.
    child_report = CandidateReport(child.cand_id, child, TrainResult(child.cand_id, child), fitness=0.1)
    assert update._island_for(ctx, state, child_report) == 1
    # The unstamped fallback: the scan finds the source first.
    unstamped = CandidateReport("u", Candidate("u", 3, _GOOD, parent_id=src.cand_id),
                                TrainResult("u", Candidate("u", 3, _GOOD)), fitness=0.1)
    assert update._island_for(ctx, state, unstamped) == 0


# ------------------------------------------------ crash repair, taught traceback

def _bad_candidate() -> Candidate:
    return Candidate("bad", 0, _BAD, prompt_messages=[{"role": "user", "content": "write a reward"}])


def test_limen_repairs_a_crash_once_in_the_same_iteration():
    """The release's `retry_on_crash`: one corrective call, re-evaluated in
    place, rather than dropping the slot without calling the generator."""
    ctx = _ctx(profile="tester")
    calls = []
    ctx.generator = lambda messages, **kw: calls.append(messages) or ["```python\n" + _GOOD + "```"]
    state = RunState()

    out = verification.run_validity(ctx, state, [_bad_candidate()])

    assert len(calls) == 1, "exactly one repair call"
    assert "failed verification" in calls[0][-1]["content"], "and it carried the trace"
    assert len(out) == 1 and out[0].valid and out[0].trainable
    assert out[0].meta["repaired_from"] == "bad" and out[0].meta["repair_attempt"] == 1
    assert ctx.budget.verify_resamples == 1
    assert state.failure_memory == [], "a repaired program is not a negative example"


def test_limen_teaches_the_next_prompt_when_the_repair_also_crashes():
    ctx = _ctx(profile="tester")
    calls = []
    ctx.generator = lambda messages, **kw: calls.append(messages) or ["```python\n" + _BAD + "```"]
    state = RunState()

    out = verification.run_validity(ctx, state, [_bad_candidate()])

    assert len(calls) == 1, "one retry (max_repair_attempts: 1), then teaching"
    assert out == [], "dropped: it does not occupy a slot"
    assert len(state.failure_memory) == 1, "taught once, by the exhaustion policy"
    entry = state.failure_memory[0]
    assert entry["error"] and entry["code"], entry
    rendered = generation._failure_traces(state)
    assert "(unrecorded)" not in rendered, "the prompt must show the taught error"
    assert entry["error"].splitlines()[0] in rendered


def test_a_carried_memory_written_under_the_old_key_still_renders():
    state = RunState()
    state.failure_memory.append({"cand_id": "old", "iteration": "0", "code": _BAD,
                                 "traceback": "SyntaxError: invalid syntax"})
    assert "SyntaxError" in generation._failure_traces(state)
    assert "(unrecorded)" not in generation._failure_traces(state)


def test_teach_on_failure_refuses_a_repair_cap_it_cannot_honour():
    with pytest.raises(ConfigError, match="max_repair_attempts"):
        load("limen", overrides={"verify.on_failure": "teach_next_iteration"})
    load("limen", overrides={"verify.on_failure": "teach_next_iteration",
                             "verify.max_repair_attempts": 0})  # the honest pairing


@pytest.mark.parametrize("name", ["limen", "limen_reward_only"])
def test_limen_pins_repair_once_then_teach(name):
    cfg = load(name)
    assert cfg["verify.on_failure"] == "resample_with_trace"
    assert cfg["verify.max_repair_attempts"] == 1
    assert cfg["verify.on_exhaustion"] == "teach_next_iteration"


# ------------------------------------------------------------- migration clock

def _drive(ctx, n, monkeypatch):
    """Run `_maybe_migrate` for archive_inserts = 1..n over three populated islands,
    returning the iteration numbers at which a migration actually fired."""
    state = RunState(iteration=0)
    for isl in range(3):
        rep = _report(f"e{isl}", 0.9 - 0.1 * isl, meta={"archive_island": isl})
        state.archive[(isl, 0, 0)] = ArchiveCell(coords=(0, 0), island=isl,
                                                 report=rep, fitness=0.9 - 0.1 * isl)
    fired = []
    real_event = ctx.event

    def rec(stage, **fields):
        if stage == "update_migration":
            fired.append(ctx.counters.get("archive_inserts"))
        return real_event(stage, **fields)

    monkeypatch.setattr(ctx, "event", rec)
    for t in range(1, n + 1):
        ctx.counters["archive_inserts"] = t
        update._maybe_migrate(ctx, state, None)
    # de-dup to the first firing per iteration (one event per migrant otherwise)
    return sorted(set(fired))


def test_island_generations_reproduce_the_release_rotation():
    """`update._island_generations` is `controller.py:533-540`: current-island increment,
    rotation every n_islands. The published 3-island, 30-iteration protocol ends [12,9,9]
    -- max 12, below interval 20 -- so it NEVER migrates; the first crossing is at 56."""
    assert update._island_generations(1, 3) == [1, 0, 0]
    assert update._island_generations(3, 3) == [3, 0, 0]
    assert update._island_generations(30, 3) == [12, 9, 9]
    assert max(update._island_generations(30, 3)) == 12  # < 20: zero migrations
    first = next(t for t in range(1, 200) if max(update._island_generations(t, 3)) >= 20)
    assert first == 56, "the release's first migration is at result 56, not 20"


def test_limen_migrates_zero_times_in_the_published_thirty_iteration_protocol(monkeypatch):
    """With the release clock LIMEN pins, 30 single-candidate iterations produce no
    cross-island exchange at all -- faithful to the release, where the `global_inserts`
    clock would fire once at the 20th insertion."""
    ctx = _ctx()
    assert ctx.cfg["update.archive.migration_clock"] == "island_generations", \
        "the published LIMEN config must pin the release clock"
    assert ctx.cfg["update.archive.n_islands"] == 3
    assert int(ctx.cfg["update.archive.migration_interval"]) == 20
    fired = _drive(ctx, 30, monkeypatch)
    assert fired == [], f"the release migrates 0 times at 30; fired at {fired}"
    assert "archive_last_migration_gen" not in ctx.counters, \
        "the island clock is a pure function of archive_inserts -- no watermark counter"


def test_the_island_clock_first_migrates_at_iteration_fifty_six(monkeypatch):
    ctx = _ctx()
    fired = _drive(ctx, 60, monkeypatch)
    assert fired and fired[0] == 56, f"first migration at 56 under the release clock; got {fired}"


def test_the_global_inserts_variant_still_fires_every_interval(monkeypatch):
    """The `global_inserts` clock is kept as a labelled variant: it fires on every
    `interval`-th archive insertion, run-wide."""
    ctx = _ctx(**{"update.archive.migration_clock": "global_inserts"})
    fired = _drive(ctx, 40, monkeypatch)
    assert fired == [20, 40], f"global_inserts fires at each interval; got {fired}"
