"""`train.pruning: successive_halving_pool` -- the across-candidate rule (rounds.py).

Why this file exists: every failure it guards renders as a plausible run. A
rung that charged itself as a new training would trip
`budget.max_policy_trainings` a round early and read as "the cap bit"; a driver whose decisions depended on completion order would make
`parallel` a science knob and read as seed noise; a stitched seed row whose
`restored_checkpoint` kept the rung-local round would name a real checkpoint
of the wrong rung. None of those crash, and none shows in a run's summary.

The mechanism is exercised on the tester tier through `hillclimb/v4` with the
three pool keys set (`_HALVING`), with `train.env_steps` raised from the
profile's 400 so the rungs stay above the mock backend's 200-step floor
(`_run_backend`: `max(200, ...)`; a rung it cannot honour trains 200 anyway and
the artifact's `train_steps` says so).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from bird.components.rounds import _merge_rungs, _rank_key, training_rounds
from bird.config import ConfigError, load
from bird.types import Candidate, TrainResult

REPO = Path(__file__).resolve().parents[1]

#: Heavyweight execution: every `_run` is a full tester-tier search, and the
#: bit-identity test forks workers (pyproject's assign-by-kind rule).
pytestmark = pytest.mark.slow

#: The successive-halving pool on the hill-climb's v4 point: the three keys that
#: select the across-candidate rule and its demonstration-scored metric.
_BASE = "hillclimb/v4"
_HALVING = {
    "problem.fitness_access": "demonstrations",
    "train.pruning": "successive_halving_pool",
    "train.pruning_metric": "demo_fraction",
}

#: K=4, budget 1600, rung_fraction 0.25, eta 2 -> rungs of 400 (all 4), 800
#: (2 survivors), 400 (the last survivor, capped at the budget): 4 trainings,
#: 3 continuations, 4x400 + 2x800 + 1x400 = 3600 steps requested against
#: 4x1600 = 6400 for the plain round.
_OVERRIDES = {
    "seed": 0,
    "train.env_steps": 1600,
    "generate.n_candidates": 4,
    "generate.candidate_schedule": "constant",
    "generate.candidate_schedule_values": [],
    "loop.n_iterations": 1,
    "final_retrain.enabled": False,
    "post": [],
}


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _events(out: Path, *stages: str) -> list:
    (journal,) = sorted(out.rglob("journal.jsonl"))
    rows = [json.loads(l) for l in journal.read_text().splitlines() if l.strip()]
    return [e for e in rows if e.get("stage") in stages]


def _scrub(e: dict) -> dict:
    return {k: v for k, v in e.items() if k not in ("t",)}


def _run(tmp_path: Path, mode: str) -> dict:
    cfg = load(_BASE, profile="tester",
               overrides={**_HALVING, **_OVERRIDES, "train.candidate_parallelism": mode})
    out = tmp_path / mode
    report = _entry().run(cfg, out_root=str(out))
    return {"budget": report["budget"],
            "rungs": [_scrub(e) for e in _events(out, "train_halving_rung")],
            "train": [_scrub(e) for e in _events(out, "train")]}


# --------------------------------------------------------------------------
# the driver, on the hill-climb v4 point
# --------------------------------------------------------------------------


def test_the_round_trains_in_rungs_and_a_rung_is_not_a_new_training(tmp_path):
    """The rung geometry is derived from the driver's own inputs (`n_ok` per
    rung), not hard-coded: the mock generator may hand the round a reward that
    does not compile, and a failed candidate is neither ranked nor continued.
    What IS fixed is the schedule -- 400, then 800, then the 400 left of the
    1600 budget -- and the accounting."""
    got = _run(tmp_path, "sequential")
    rungs = got["rungs"]
    assert len(rungs) == 3 and rungs[-1]["finished"] is True, rungs
    assert [(r["rung"], r["env_steps_each"], r["cumulative_steps"]) for r in rungs] \
        == [(0, 400, 400), (1, 800, 1200), (2, 400, 1600)], rungs
    assert rungs[0]["n_trained"] == 4
    # keep the top 1/eta of the candidates that trained OK, never fewer than one
    expect_keep = max(1, rungs[0]["n_ok"] // 2)
    assert len(rungs[0]["kept"]) == expect_keep and rungs[1]["n_trained"] == expect_keep
    assert len(rungs[1]["kept"]) == 1 and rungs[2]["n_trained"] == 1
    assert rungs[2]["kept"] == []
    # One training per candidate that LAUNCHED (a screened-out candidate is a
    # skip and one that never compiled is `candidates_invalid`, as on every
    # other config); every later rung is a continuation and lands in
    # `training_slices`, the LaRes column, so `max_policy_trainings` keeps
    # counting candidates.
    continuations = rungs[1]["n_trained"] + rungs[2]["n_trained"]
    b = got["budget"]
    assert b["policy_trainings"] == 4 - b["policy_trainings_skipped"] - b.get("candidates_invalid", 0), b
    assert got["budget"]["training_slices"] == continuations, got["budget"]
    # Every OK candidate the pool dropped is `pruned` -- a lower bound, not a
    # finished training -- and the survivor is not.
    pruned = {e["cand_id"]: e["pruned"] for e in got["train"] if e["trained"]}
    survivor = rungs[1]["kept"][0]
    assert pruned[survivor] is False
    assert sum(pruned.values()) == rungs[0]["n_ok"] - 1, (pruned, rungs[0])
    # THE metric reached the ranking, and the basis of every row is decided by
    # the candidate's OWN ceiling, not by the field name. `toy_reacher` ships
    # an expert, so a ceiling is never `unavailable`; but "an expert exists"
    # is not "every candidate's reward pays the expert more than random", and
    # `_demo_ceiling` rules a ceiling UNINFORMATIVE exactly when it does not
    # (`expert paid no more than random`, 2 episodes). Asserting
    # `demo_fraction` on every row would pass only while the mock's draws
    # happen to order the ladder right: the mock hashes the prompt into its
    # RNG, so a prompt edit can hand it rewards that pay random more than the
    # expert (e.g. 45.7 against 43.7) -- a correct
    # `own_range (ceiling uninformative)`, tier 0, below every informative row,
    # which is `_rank_key`'s documented rule. So the claim is the LINKAGE, per
    # row: `demo_fraction` iff the candidate's recorded ceiling is informative,
    # `own_range` with the recorded status otherwise, never `non_finite`; at
    # least one row in every rung ranks on `demo_fraction` (the metric reached
    # the ranking at all); and every informative row precedes every
    # uninformative one. This fails on a ceiling the ranking ignored, in either
    # direction, where a looser assertion admitting the whole codomain could
    # not fail.
    root = next(p for p in Path(tmp_path).rglob("candidates")).parent
    def _ceiling(cid: str) -> dict:
        meta = json.loads((root / "candidates" / f"iter00_{cid}" / "meta.json").read_text())
        return (meta.get("meta", meta)).get("ceiling") or {}
    for r in rungs:
        bases = [row["basis"] for row in r["ranking"]]
        assert "demo_fraction" in bases, r
        assert "non_finite" not in bases, r
        seen_uninformative = False
        for row in r["ranking"]:
            c = _ceiling(row["cand_id"])
            if c.get("informative"):
                assert row["basis"] == "demo_fraction" and row["tier"] == 1, (row, c)
                assert not seen_uninformative, ("an informative row ranked below an uninformative one", r)
            else:
                assert row["basis"] == f"own_range (ceiling {c.get('status', 'unavailable')})" \
                    and row["tier"] == 0, (row, c)
                assert c.get("status") == "uninformative" and c.get("reason"), (
                    "toy_reacher ships an expert: a ceiling here is informative or uninformative "
                    f"with a reason, never unavailable: {c}")
                seen_uninformative = True


def test_parallel_is_bit_identical_to_sequential(tmp_path):
    """The rung decisions and the merged results must not depend on the
    schedule: a survivor's policy crosses the rung boundary through the store
    merge, and the ranking ties break on candidate index, never completion."""
    seq = _run(tmp_path, "sequential")
    par = _run(tmp_path, "parallel")
    assert par["rungs"] == seq["rungs"]
    assert par["train"] == seq["train"]
    assert {k: v for k, v in par["budget"].items() if k not in ("gpu_seconds", "wallclock_s")} \
        == {k: v for k, v in seq["budget"].items() if k not in ("gpu_seconds", "wallclock_s")}


# --------------------------------------------------------------------------
# stitching
# --------------------------------------------------------------------------


def _part(steps: int, rounds: int, *, restored=None, shipped=None) -> TrainResult:
    c = Candidate(cand_id="c0", iteration=0, reward_code="def compute_reward(s,a,s2): return 0.0")
    r = TrainResult(cand_id="c0", candidate=c)
    r.checkpoints = [{"step": float(steps * (i + 1) / rounds), "round": float(i + 1),
                      "reward_return": float(i)} for i in range(rounds)]
    r.env_steps_used = steps
    r.seed_metrics = [{"seed": 1, "env_steps": steps, "train_steps": steps,
                       "train_steps_requested": steps, "wallclock_s": 1.0,
                       "n_checkpoints": rounds, "max": float(rounds), "auc": 1.0,
                       "checkpoints": list(r.checkpoints), "init": "from_scratch",
                       "warm_started_from": "", "restored_checkpoint": restored,
                       "shipped_checkpoint": shipped, "pruned_at_round": None}]
    r.policy_ref = f"policy:c0@{steps}"
    return r


def test_merge_lays_the_rungs_end_to_end_and_moves_the_restored_round():
    """Rungs of 2 and 3 checkpoints; the last rung restored ITS round 1 (its
    peak). On the merged curve that is round 3 of 5 -- unshifted it would name
    the first rung's round 1, a real checkpoint of the wrong rung."""
    a, b = _part(400, 2), _part(800, 3, restored=1, shipped=1)
    m = _merge_rungs([a, b], 1600, dropped=False)
    assert [p["round"] for p in m.checkpoints] == [1.0, 2.0, 3.0, 4.0, 5.0]
    steps = [p["step"] for p in m.checkpoints]
    assert steps == sorted(steps) and steps[-1] == 1200.0
    assert m.env_steps_used == 1200 and m.policy_ref == "policy:c0@800"
    row = m.seed_metrics[0]
    assert (row["train_steps"], row["train_steps_requested"]) == (1200, 1600)
    assert (row["restored_checkpoint"], row["shipped_checkpoint"]) == (3, 3)
    assert (row["rungs"], row["rung_train_steps"], row["halving_dropped"]) == (2, [400, 800], False)
    assert m.pruned is False


def test_a_dropped_candidate_is_pruned_at_its_stitched_checkpoint_count():
    m = _merge_rungs([_part(400, 2)], 1600, dropped=True)
    assert m.pruned is True
    assert m.seed_metrics[0]["pruned_at_round"] == 2
    assert m.seed_metrics[0]["halving_dropped"] is True


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad, needle", [
    ({"train.pruning_metric": "own_reward"}, "own units"),
    ({"train.interaction": "shared_population"}, "one driver"),
    ({"train.seeds_per_candidate": 2}, "seeds_per_candidate=1"),
    ({"train.hyperparameter_search": "grid"}, "hyperparameter_search=none"),
    ({"train.pruning_cfg.eta": 1}, "eta"),
    ({"train.pruning_cfg.rung_fraction": 0.0}, "rung_fraction"),
    ({"select.allocation": "successive_halving"}, "select.allocation"),
    # the ceiling GUARD wraps a within-training rule; the pool has none
    ({"train.pruning_cfg.ceiling": "demo_return"}, "WITHIN-training"),
])
def test_the_pool_rule_refuses_what_it_cannot_execute(bad, needle):
    with pytest.raises(ConfigError, match=needle):
        load(_BASE, profile="tester", overrides={**_HALVING, **bad})


# --------------------------------------------------------------------------
# rung accounting: LaRes's `resume_ref` + `slice_mode` == a continuation counter
# --------------------------------------------------------------------------
#
# The reference accounting charges a later rung as a CONTINUATION:
# `training_continuations += 1`, `env_steps += n`, `gpu_seconds += t`,
# `_check()`, and `policy_trainings` unchanged. This implementation routes the
# same rung through the backend's `resume_ref` under `Budget.slice_mode()`,
# where `record_training` does `training_slices += 1`, `env_steps += n`,
# `gpu_seconds += t`, `_check()`, and `policy_trainings` unchanged. Same
# arithmetic, one column renamed. The test below drives the driver with a stub
# backend that charges exactly what a backend charges and compares the three
# numbers -- and the cap -- to what the reference accounting (`_reference_accounting`)
# produces on the same rungs.


def _reference_accounting(k: int, rungs: list) -> dict:
    """What the reference accounting (a training for rung 0, a continuation for
    every later rung) produces on `rungs`, a list of (n_candidates_trained,
    steps_each) with rung 0 first."""
    out = {"policy_trainings": 0, "continuations": 0, "env_steps": 0}
    for i, (n, steps) in enumerate(rungs):
        for _ in range(n):
            if i == 0:
                out["policy_trainings"] += 1
            else:
                out["continuations"] += 1
            out["env_steps"] += steps
    return out


#: The schedule as the CODE runs it (`slice_ *= eta` from rung 1, the last slice
#: capped at the budget, `keep = max(1, n // eta)`), on the 1600-step budget:
#: rungs of (candidates trained, steps each). In budgets: K=8 -> 4.5 with TWO
#: survivors at the full budget, K=6 -> 3.25, K=4 -> 2.25, K=2 -> 1.25. The
#: tempting reading for K=8 -- 4.25 budgets and one survivor -- is not what the
#: code does, and a cost estimate or a footnote citing it would be wrong by 0.25
#: budgets and one survivor per round, with nothing rendering that as a failure.
_SCHEDULES = {
    8: [(8, 400), (4, 800), (2, 400)],
    6: [(6, 400), (3, 800), (1, 400)],
    4: [(4, 400), (2, 800), (1, 400)],
    2: [(2, 400), (1, 800), (1, 400)],
}


#: The documented cost, in budgets (`_default.yaml`, rounds.py) -- asserted
#: against the DRIVER's own calls below,
#: not against `_SCHEDULES` (a test that compares a table with itself calls no code).
_BUDGETS = {8: 4.5, 6: 3.25, 4: 2.25, 2: 1.25}


@pytest.mark.parametrize("k", sorted(_SCHEDULES))
def test_a_rung_is_charged_as_a_record_continuation(k):
    import types as pytypes
    from bird.budget import Budget, BudgetExceeded
    from bird.components import training as T
    from bird.components.rounds import _successive_halving_pool

    total = 1600
    cfg = pytypes.SimpleNamespace(get=lambda key, d=None: {
        "problem.env_id": "toy_reacher",
        "train.env_steps": total, "train.pruning_cfg.rung_fraction": 0.25,
        "train.pruning_cfg.eta": 2, "train.pruning_metric": "demo_fraction"}.get(key, d))
    # A cap of exactly K: a continuation never moves `policy_trainings`, so the
    # cap must not fire on a rung either.
    budget = Budget(max_policy_trainings=k)
    events = []
    from bird.envs.toy import ToyReacher
    ctx = pytypes.SimpleNamespace(cfg=cfg, budget=budget, env=ToyReacher(),
                                  event=lambda stage, **kw: events.append((stage, kw)))
    cands = [Candidate(cand_id=f"c{i:04d}", iteration=0,
                       reward_code="def compute_reward(s,a,s2): return 0.0")
             for i in range(k)]
    for i, c in enumerate(cands):
        c.meta["ceiling"] = {"informative": True, "status": "informative"}
    calls = []

    def run_one(c, **kw):
        # what a backend charges: one `record_training` per seed, its steps
        steps = int(kw["env_steps"])
        calls.append((c.cand_id, steps, kw.get("resume_ref")))
        ctx.budget.record_training(env_steps=steps, gpu_seconds=1.0)
        r = TrainResult(cand_id=c.cand_id, candidate=c)
        r.env_steps_used = steps
        # rank by index so the survivors are deterministic: higher index wins
        r.checkpoints = [{"step": float(steps), "round": 1.0, "reward_return": 0.0,
                          "demo_fraction": float(int(c.cand_id[1:])) / k}]
        r.seed_metrics = [{"seed": 0, "env_steps": steps, "train_steps": steps,
                           "train_steps_requested": steps, "wallclock_s": 1.0,
                           "n_checkpoints": 1, "max": 0.0, "auc": 0.0,
                           "checkpoints": list(r.checkpoints)}]
        r.policy_ref = f"policy:{c.cand_id}"
        return r

    def round_pass(ctx_, state_, batch, run_fn, on_res):
        return [run_fn(c) for c in batch]

    try:
        results = _successive_halving_pool(ctx, None, cands, run_one, lambda c, r: None, round_pass)
    except BudgetExceeded as exc:  # pragma: no cover - the failure this test exists to catch
        pytest.fail(f"a rung continuation tripped max_policy_trainings={k}: {exc}")

    rungs = _SCHEDULES[k]
    assert [(len([c for c in calls if c[1] == s and (c[2] is None) == (i == 0)]), s)
            for i, (n, s) in enumerate(rungs)] == rungs, calls
    # every continuation resumed the candidate's OWN policy
    assert all(ref == f"policy:{cid}" for cid, _s, ref in calls if ref is not None)
    ref = _reference_accounting(k, rungs)
    assert budget.policy_trainings == ref["policy_trainings"] == k
    n_cont = sum(n for n, _s in rungs[1:])
    assert budget.training_slices == ref["continuations"] == n_cont
    assert budget.env_steps == ref["env_steps"] == sum(n * s for n, s in rungs)
    assert budget.gpu_seconds == pytest.approx(len(calls))
    survivors = rungs[-1][0]
    assert len(results) == k and sum(r.pruned for r in results) == k - survivors
    # the documented cost, read off what the driver actually requested
    assert sum(steps for _c, steps, _r in calls) / total == _BUDGETS[k]
    if k == 8:
        assert survivors == 2, "K=8 ends with two full-budget survivors, not one"
    # the survivors are the highest-index candidates (the stub ranks by index) and
    # each of them trained the whole budget across its rungs
    for r in results[k - survivors:]:
        assert not r.pruned and r.env_steps_used == total, r


# --------------------------------------------------------------------------
# `demo_fraction` on an env with no expert: refuse, do not rank on index
# --------------------------------------------------------------------------
#
# `_demo_fraction_point` falls back to own-range progress when a candidate's
# ceiling is not informative, and `_rank_key` puts every such candidate in tier
# 0 with ties on index. That is the right reading for ONE uninformative reward
# among informative ones. On an env with no demonstration policy it is every
# candidate, and a rung would keep the lowest-index half of the pool while the
# journal read `basis: own_range (ceiling unavailable)` -- index-order culling
# rendered as a ranking. `_check_coherence` cannot see the demo catalogue, so
# the refusal is the driver's, before rung 0 trains anything.


def test_demo_fraction_refuses_before_rung_0_on_an_env_with_no_expert():
    import types as pytypes
    import numpy as np
    # The scenario: a successive_halving_pool config pointed at an env with no
    # demonstration policy. `gym_pendulum` here is a synthetic id with no
    # policies/ record. Every toy env has a policies/ record, so the env is a
    # stub with exactly what `demos._expert` reads -- the action bounds and no
    # `expert_policy` -- under an id with no record and no bundled policy.
    env = pytypes.SimpleNamespace(action_low=np.array([-1.0]), action_high=np.array([1.0]))
    cfg = pytypes.SimpleNamespace(get=lambda key, d=None: {
        "problem.env_id": "gym_pendulum", "train.env_steps": 1600,
        "train.pruning": "successive_halving_pool",
        "train.pruning_cfg.rung_fraction": 0.25, "train.pruning_cfg.eta": 2,
        "train.pruning_metric": "demo_fraction"}.get(key, d))
    ctx = pytypes.SimpleNamespace(cfg=cfg, env=env, budget=None, event=lambda *a, **k: None)
    cands = [Candidate(cand_id=f"c{i:04d}", iteration=0, reward_code="def compute_reward(s,a,s2): return 0.0")
             for i in range(4)]
    trained = []

    def run_one(c, **kw):  # pragma: no cover - the failure this test exists to catch
        trained.append(c.cand_id)
        return TrainResult(cand_id=c.cand_id, candidate=c)

    with pytest.raises(ConfigError, match="demo_fraction.*gym_pendulum.*no demonstration policy"):
        training_rounds(ctx, None, cands, run_one, lambda c, r: None,
                        lambda ctx_, s_, batch, fn, on: [fn(c) for c in batch])
    assert trained == [], "the refusal came after a rung had already been paid for"
    # ... and under a WITHIN-training rule too: `_default.yaml` promises the
    # refusal for the metric, not for one rule.
    cfg2 = pytypes.SimpleNamespace(get=lambda key, d=None: {
        "problem.env_id": "gym_pendulum", "train.pruning": "plateau",
        "train.pruning_metric": "demo_fraction"}.get(key, d))
    with pytest.raises(ConfigError, match="demo_fraction.*gym_pendulum"):
        training_rounds(pytypes.SimpleNamespace(cfg=cfg2, env=env, budget=None, event=lambda *a, **k: None),
                        None, cands, run_one, lambda c, r: None,
                        lambda ctx_, s_, batch, fn, on: [fn(c) for c in batch])
    assert trained == []


def test_demo_fraction_does_not_refuse_where_an_expert_exists():
    """The same lookup on `toy_reacher`, which ships its PD law: no refusal --
    the guard is on the catalogue, not on the metric."""
    import types as pytypes
    from bird.components.rounds import _require_demonstrations
    from bird.envs.toy import ToyReacher
    env = ToyReacher()
    cfg = pytypes.SimpleNamespace(get=lambda key, d=None: {"problem.env_id": "toy_reacher"}.get(key, d))
    _require_demonstrations(pytypes.SimpleNamespace(cfg=cfg, env=env))


def test_a_tie_on_the_metric_keeps_the_lowest_candidate_index():
    """`rounds.py`: "ties break on index". The stub scores above are all
    distinct, so without this test flipping the sort to prefer the HIGHER index
    would pass every test. Three candidates, two of them tied on
    the metric: the tied pair must rank in index order, and with eta 2 over
    three survivors the pool keeps one -- the higher score, then the lower index."""
    def part(cid, score):
        r = TrainResult(cand_id=cid, candidate=Candidate(cand_id=cid, iteration=0, reward_code="x"))
        r.checkpoints = [{"step": 1.0, "round": 1.0, "reward_return": 0.0, "demo_fraction": score}]
        return r
    cands = {cid: Candidate(cand_id=cid, iteration=0, reward_code="x") for cid in ("c0000", "c0001", "c0002")}
    for c in cands.values():
        c.meta["ceiling"] = {"informative": True, "status": "informative"}
    keys = {cid: _rank_key([part(cid, s)], "demo_fraction", cands[cid])
            for cid, s in (("c0000", 0.5), ("c0001", 0.9), ("c0002", 0.9))}
    order = sorted(keys, key=lambda cid: (-keys[cid][0], -keys[cid][1], cid))
    assert order == ["c0001", "c0002", "c0000"], order
    assert keys["c0001"][1] == keys["c0002"][1], "the tie is real"


def test_a_pool_dropped_candidate_is_not_told_its_curve_stopped_improving():
    """`evaluation.training_caveat` must not render a rank-dropped candidate
    like a rule-cut one -- "because the demo_fraction curve stopped improving" --
    for a curve that may have been rising. The caveat names the pool ranking and
    the rung, and says the curve may still have been rising."""
    import types as pytypes
    from bird.components.evaluation import training_caveat
    c = Candidate(cand_id="c0003", iteration=0, reward_code="x")
    rising = TrainResult(cand_id="c0003", candidate=c, trained=True)
    rising.checkpoints = [{"step": float(i), "round": float(i), "reward_return": float(i),
                           "demo_fraction": 0.1 * i} for i in range(1, 4)]
    rising.seed_metrics = [{"seed": 0, "n_checkpoints": 3, "checkpoints": list(rising.checkpoints),
                            "train_steps": 400, "env_steps": 400}]
    merged = _merge_rungs([rising], 1600, dropped=True)
    assert merged.pruned and merged.seed_metrics[0]["halving_dropped"] is True
    ctx = pytypes.SimpleNamespace(cfg={"train.pruning_metric": "demo_fraction"})
    report = pytypes.SimpleNamespace(result=merged)
    text = training_caveat(ctx, report)
    assert "successive-halving pool" in text and "rung 1" in text, text
    assert "may still have been rising" in text, text
    assert "because the `demo_fraction` curve stopped improving" not in text, text
    assert "LOWER BOUND" in text, text
