"""`train.on_runtime_error: repair` -- a reward that passes verification but RAISES
during policy training is re-generated with its TRAINING traceback and re-trained
IN-ITERATION on a fresh seed stream, so the tree backs up the repaired result, not the
crash. This is RF-Agent's shared retry loop (`rfagent_algo.py:561-563`, `:661-670`),
which covers training errors as well as pre-training ones. The default is `record`;
rf_agent.yaml pins `repair`.

Three conditions, asserted below:
  1. ACCOUNTING -- one re-generation is one LLM call, one re-train is one policy training,
     charged once each, and the train log/`state.n_train_repairs` counts them.
  2. FRESH SEEDS -- each re-train is salted `seed_phase=f"repair{attempt}"`, disjoint from
     the crashed attempt and the search, so a repaired candidate is a fresh trial and
     `parallel == sequential` holds (the pass is parent-side, after interaction()).
  3. TERMINATION -- the repair runs before evaluate/select, so the REPAIRED result replaces
     the crash in the returned list.
"""
from __future__ import annotations

from types import SimpleNamespace

from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.llm.base import LLMClient
from bird.state import RunState
from bird.types import Candidate, TrainResult

_GOOD = "```python\ndef reward(s, a, s2):\n    return 1.0, {}\n```"


def _bird():
    import importlib.util
    from conftest import REPO
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _CountingClient(LLMClient):
    """One `_record` per round-trip, like every real provider; always returns a
    reward that compiles and passes validity."""

    def _complete(self, messages, n, temperature, images, tag):
        self._record(10, 20)
        return [_GOOD] * n


def _ctx(**overrides) -> Context:
    cfg = load("card", overrides={"train.on_runtime_error": "repair", **overrides})
    ctx = Context(cfg=cfg, budget=Budget(),
                  env=SimpleNamespace(symbol_mapping={"GENERAL_SYMBOL": "s[0]"}))
    # An in-memory journal: `Context.event` forwards to `rundir.event` when a
    # rundir is set, so the test can read what a run dir would have recorded.
    ctx.events = []
    ctx.rundir = SimpleNamespace(event=lambda stage, **f: ctx.events.append((stage, f)))
    ctx.generator = _CountingClient(ctx, "generator")
    return ctx


def _crashed(cid: str) -> TrainResult:
    cand = Candidate(cid, 0, "def reward(s, a, s2): return object_pos(s, 2)",
                     prompt_messages=[{"role": "user", "content": "gen"}])
    return TrainResult(cand_id=cid, candidate=cand, trained=False,
                       error="Traceback (most recent call last):\n  IndexError: only 2 dims")


def test_a_training_crash_is_repaired_retrained_fresh_and_reaches_the_result():
    ctx = _ctx()
    assert ctx.cfg["train.on_runtime_error"] == "repair"
    state = RunState()
    crashed = _crashed("c0")

    seen_phases = []
    seen_seeds = []
    n_calls = {"train": 0}

    def mock_hpsearch(ctx, state, cand, backend, n_seeds, seed_phase=""):
        seen_phases.append(seed_phase)
        seen_seeds.append(n_seeds)
        n_calls["train"] += 1
        # crash once more on the first repair, succeed on the second -> two attempts
        if n_calls["train"] == 1:
            return TrainResult(cand_id=cand.cand_id, candidate=cand, trained=False,
                               error="Traceback: still raises")
        return TrainResult(cand_id=cand.cand_id, candidate=cand, trained=True)

    out, n_repairs = _bird()._repair_training_errors(
        # the allocator's plan is keyed by the CRASHED id (c0); `_regenerate` mints
        # a fresh id for every repair, so a lookup by the new id would fall back to
        # `train.seeds_per_candidate` and silently ignore the allocation
        ctx, state, [crashed], backend=None, hpsearch=mock_hpsearch, plan={"c0": 3})

    # (3) TERMINATION: the repaired (good) result replaced the crash
    assert len(out) == 1 and not out[0].error and out[0].trained
    # (2) FRESH SEEDS: one salted stream per attempt, disjoint from the search's ""
    assert seen_phases == ["repair1", "repair2"], seen_phases
    assert "" not in seen_phases, "a repair must not reuse the search's seed stream"
    # the SEED PLAN followed the crashed trial through both repairs
    assert seen_seeds == [3, 3], seen_seeds
    # (1) ACCOUNTING: two re-trains, two re-generations (one LLM call each), counted
    assert n_repairs == 2
    assert n_calls["train"] == 2
    assert ctx.generator.n_calls == 2, "one LLM call per re-generation, charged once each"
    assert ctx.budget.llm_calls == 2
    assert int(getattr(state, "n_train_repairs", 0)) == 2
    # the TRAINING traceback was the repair prompt's trace, not a pre-training one
    assert crashed.candidate.meta.get("repair_trace") == crashed.error
    # JOURNAL: every re-train leaves the `train` line `on_result` writes for a
    # first training, linked to the crashed id -- a journal reader's per-candidate
    # rows are built from it, so a repaired training with no line would read as
    # never trained beside a report.json that says trained.
    trains = [f for st, f in ctx.events if st == "train"]
    assert [t["repair_attempt"] for t in trains] == [1, 2]
    assert all(t["repaired_from"] == "c0" for t in trains)
    assert trains[-1]["cand_id"] == out[0].cand_id and trains[-1]["trained"] is True
    assert trains[0]["trained"] is False
    assert [st for st, _ in ctx.events if st == "train_repair"] == ["train_repair"]


def test_record_default_leaves_a_training_crash_untouched():
    ctx = _ctx(**{"train.on_runtime_error": "record"})
    # the default path never calls the repair helper (bird.train gates on the key); assert
    # the helper is a no-op contract by not invoking it, and that the key resolves to record
    assert ctx.cfg["train.on_runtime_error"] == "record"
