"""`train.interaction_cfg.shared_buffer` -- LaRes's buffer, honoured.

WHAT GOES WRONG WITHOUT IT. The key is declared (`configs/_default.yaml`,
`bird/schema.py`), defaults `true` and is cited to LaRes §4.3 in
`configs/methods/lares.yaml`. If no code reads it, then under `train.interaction:
shared_population` every slice of an arm hands back only `policy_ref`, so each
of the five slices per candidate builds a fresh SAC on an EMPTY replay buffer,
re-pays `learning_starts`, and -- if `_seed_base` knows nothing about slices --
re-seeds to the same value as the slice before it, replaying its predecessor's
episode stream. The method that runs is "SAC restarted from an empty buffer five
times per candidate", and `shared_buffer: false` is bit-identical to `true`
(measured: only `config_hash` and the two timing fields move). That is a
fabricated pin, and the remedy is to honour the key, never to delete it quietly.

WHAT THIS FILE PINS, one test each, all of which FAIL without the mechanism:

  1. a resumed slice starts from its own previous transitions -- the buffer
     restored equals what the slice before it added;
  2. `true` and `false` produce different numbers: under `true` an arm's
     second slice holds rows another arm collected, under `false` none;
  3. two consecutive slices of one arm do not draw the same seed, and the
     FIRST slice's seed is exactly what it was before (so every unsliced
     config is untouched);
  4. the coherence rule refuses `true` beside a learner that has no buffer to
     fill, and the on-policy arm (LaRes under PPO, `false`) still loads;
  5. on the tester profile the mock backend ignores the buffer and SAYS SO in the
     journal, so a LaRes tester run cannot be read as having pooled anything.

The sb3 tests use pendulum SAC at a few hundred steps: the assertion is on
the buffer accounting the backend writes into the seed row, which is exact,
not on a fitness that a lucky seed could satisfy.
"""

from __future__ import annotations

import json

import pytest

from conftest import REPO  # noqa: F401  (path setup)
from bird import registry
from bird.budget import Budget
from bird.components import training as T
from bird.config import ConfigError, load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate

pytestmark = pytest.mark.slow

try:
    import stable_baselines3  # noqa: F401

    HAVE_SB3 = True
except ImportError:  # pragma: no cover - depends on the machine
    HAVE_SB3 = False

needs_sb3 = pytest.mark.skipif(not HAVE_SB3, reason="train.backend: sb3 needs sb3+gymnasium")

UPRIGHT = """
def compute_reward(state, action):
    import numpy as np
    th = np.arctan2(state[1], state[0])
    return -(th ** 2) - 0.1 * state[2] ** 2, {"upright": -(th ** 2)}
"""


@pytest.fixture(autouse=True)
def _clean_stores():
    for store in (T._POLICY_STORE, T._REPLAY_STORE, T._ROUND_REPLAY):
        store.clear()
    if HAVE_SB3:
        # One torch thread: the assertions are on buffer COUNTS, which no thread
        # count can move, and a 4-thread SAC on a shared 4-core machine measured
        # 15+ minutes for what takes 20 s pinned (the candidate workers already
        # pin one thread each; this is the in-process path).
        import torch
        torch.set_num_threads(1)
    yield
    for store in (T._POLICY_STORE, T._REPLAY_STORE, T._ROUND_REPLAY):
        store.clear()


def _ctx(shared: bool, **overrides):
    """`lares` on the dev profile (sb3/SAC), pendulum, two slices per arm, and
    the two stabilisers off so the buffer is the only coupling under test."""
    base = {"seed": 0, "problem.env_id": "pendulum", "output.tracker": "none",
            "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "evaluate.rollouts_per_candidate": 1,
            "train.env_steps": 400,
            "train.interaction_cfg.slices_per_candidate": 2,
            "train.interaction_allocator": "uniform",
            "train.interaction_cfg.shared_buffer": shared,
            "train.reward_scaling": "none",
            "train.elite_constraint.kind": "none",
            # `archive` because `elitist_population` keeps its population there
            # and the coherence rule refuses a carry that drops it.
            "loop.carry": ["best_reward", "replay_buffer", "policy_checkpoint", "archive"]}
    base.update(overrides)
    cfg = load("lares", overrides=base, profile="dev")
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", "pendulum")(ctx)
    events = []
    ctx.event = lambda stage, **fields: events.append({"stage": stage, **fields})
    return ctx, events


def _round(ctx, n_arms=2):
    """One shared_population round, sequential schedule, `n_arms` arms."""
    cands = [Candidate(cand_id=f"c000{i}", reward_code=UPRIGHT, iteration=0)
             for i in range(n_arms)]
    backend = registry.get("train_backend", ctx.cfg["train.backend"])
    schedule = registry.get("candidate_parallelism", "sequential")
    inter = registry.get("interaction", "shared_population")
    state = RunState()
    return inter(ctx, state, cands, lambda c: backend(ctx, state, c, 1),
                 lambda c, r: None, schedule, backend)


# ==========================================================================
# 1. per-arm continuity
# ==========================================================================


@needs_sb3
def test_a_resumed_slice_starts_from_its_own_previous_transitions():
    """Slice 2 of an arm is pre-filled with exactly the rows slice 1 added.

    Driven through the backend directly with the hand-off the driver builds,
    so the assertion is on the mechanism and not on the allocator's choices.
    `replay_added` is the count `_sb3_run` exports and `replay_prefill_own` the
    count `_sb3_prefill_replay` wrote back; without the mechanism neither
    field exists because no slice ever restores anything.
    """
    ctx, _ = _ctx(shared=False)
    backend = registry.get("train_backend", "sb3")
    cand = Candidate(cand_id="c0000", reward_code=UPRIGHT, iteration=0)
    state = RunState()

    first = backend(ctx, state, cand, 1, env_steps=200,
                    handoff=T.SliceHandoff(index=0, export_ref="new:c0000:0"))
    row0 = first.seed_metrics[0]
    assert row0["replay_prefill_supported"] is True
    assert row0["replay_prefill_own"] == 0 and row0["replay_prefill_pooled"] == 0
    exported = T._ROUND_REPLAY.get("new:c0000:0")
    assert exported is not None and len(exported) == row0["replay_added"] > 0, \
        "slice 1 exported nothing, so slice 2 has nothing to resume from"
    assert len(exported) == row0["train_steps"], \
        "the export must be the rows this slice ADDED -- one per env step"

    # Hand the export back as the arm's own buffer, the way the driver does.
    T._ROUND_REPLAY["own:c0000"] = exported
    second = backend(ctx, state, cand, 1, env_steps=200, resume_ref=first.policy_ref,
                     handoff=T.SliceHandoff(index=1, prefill_ref="own:c0000",
                                            prefill_own=len(exported),
                                            export_ref="new:c0000:1"))
    row1 = second.seed_metrics[0]
    assert row1["replay_prefill_own"] == len(exported), (
        "slice 2 declared a resume and started from an empty buffer -- this is "
        f"the defect: {row1}")
    assert row1["replay_prefill_pooled"] == 0, "no other arm exists to pool from"
    assert row1["replay_prefill_ref"] == "own:c0000"
    assert row1["warm_started_from"] == first.policy_ref, "the policy half must still resume"
    # SB3's warm-up (default learning_starts=100) was paid by slice 1; slice 2
    # holds 200 rows and must not pay it again.
    assert row1["learning_starts_effective"] == 0
    assert row0["learning_starts_effective"] == 100
    # And slice 2 exports only what IT added, not the restored rows too.
    assert len(T._ROUND_REPLAY["new:c0000:1"]) == row1["replay_added"] == row1["train_steps"]


# ==========================================================================
# 2. the ablation moves a number
# ==========================================================================


@needs_sb3
def test_shared_buffer_true_pools_other_arms_rows_and_false_does_not():
    """`true` vs `false` must differ, or the key is still a fabricated pin.

    Two arms, two slices each, uniform allocator. After wave 0 both arms have
    exported; in wave 1 arm 0's prefill holds arm 1's rows under `true` and
    none of them under `false`. That is the exact quantity the paper's
    contribution 2 adds, read off the seed row rather than off a fitness.
    """
    got = {}
    for shared in (True, False):
        for store in (T._POLICY_STORE, T._REPLAY_STORE, T._ROUND_REPLAY):
            store.clear()
        ctx, events = _ctx(shared=shared)
        results = _round(ctx)
        last_rows = {r.cand_id: r.seed_metrics[-1] for r in results}
        slices = [e for e in events if e["stage"] == "interaction_slice"]
        # Read the carried pool NOW: the next run clears the stores.
        pools = {r.replay_ref: len(T._REPLAY_STORE.get(r.replay_ref) or ())
                 for r in results}
        got[shared] = (last_rows, slices, results, pools)

    rows_t, slices_t, res_t, pools_t = got[True]
    rows_f, slices_f, res_f, _pools_f = got[False]
    for cid, row in rows_t.items():
        assert row["replay_prefill_pooled"] > 0, (
            f"shared_buffer=true but {cid}'s second slice saw no other arm's rows: {row}")
        assert row["replay_prefill_own"] > 0
    for cid, row in rows_f.items():
        assert row["replay_prefill_pooled"] == 0, (
            f"shared_buffer=false but {cid}'s second slice saw another arm's rows: {row}")
        assert row["replay_prefill_own"] > 0, "per-arm continuity must survive the ablation"

    # The journal carries the same accounting per slice, so a run dir shows it.
    second_wave_t = [e for e in slices_t if e["wave"] == 1]
    assert second_wave_t and all(e["buffer_prefill_pooled"] > 0 for e in second_wave_t)
    assert all(e["buffer_added"] > 0 for e in slices_t)
    second_wave_f = [e for e in slices_f if e["wave"] == 1]
    assert second_wave_f and all(e["buffer_prefill_pooled"] == 0 for e in second_wave_f)

    # Under `true` the round's pool is what the carry hands on, for every arm.
    pool_refs = {r.replay_ref for r in res_t}
    assert len(pool_refs) == 1 and next(iter(pool_refs)).startswith("replay:pool:"), \
        f"every arm must point at the ONE round pool: {pool_refs}"
    (pool_rows,) = set(pools_t.values())
    assert pool_rows == sum(e["buffer_added"] for e in slices_t) > 0, \
        f"the carried pool holds {pool_rows} rows, not every slice's export"
    # And the learning curves are not the same curves: the buffer changed the
    # work. Compared on the candidate's OWN reward (`reward_return`), which
    # moves at this budget; pendulum's task `fitness` is 0.0 on both sides at
    # a few hundred steps and would compare equal for the wrong reason.
    curves_t = [tuple(round(p["reward_return"], 6) for p in r.checkpoints) for r in res_t]
    curves_f = [tuple(round(p["reward_return"], 6) for p in r.checkpoints) for r in res_f]
    assert curves_t != curves_f, "pooling changed no checkpoint -- did it run?"


# ==========================================================================
# 3. seeds
# ==========================================================================


@needs_sb3
def test_consecutive_slices_of_one_arm_do_not_share_a_seed():
    """Slice k+1 must not replay slice k's `_Gym` episode stream.

    The env's reset seeds derive from the seed SB3 hands the env, which is the
    seed in the row; two slices at one seed reset to the same states, draw the
    same warm-up actions and the same torch stream. The first slice's seed is
    pinned to the unsalted value so nothing outside `shared_population` moved.
    """
    ctx, events = _ctx(shared=False)
    results = _round(ctx)
    slices = [e for e in events if e["stage"] == "interaction_slice"]
    for r in results:
        # `_merge_slices` keeps only the LAST slice's seed row (it describes the
        # policy that leaves the round), so the per-slice seeds are read off
        # the journal, where every slice has one.
        seeds = [e["seed"] for e in slices if e["cand_id"] == r.cand_id]
        assert len(seeds) == 2, slices
        assert seeds[0] != seeds[1], f"{r.cand_id}: both slices seeded {seeds[0]}"
        assert seeds[0] == T._seed_base(ctx, RunState(), r.candidate), \
            "the FIRST slice's seed must be exactly the pre-existing one"
        assert seeds[1] == seeds[0] + T._SLICE_SEED_STRIDE


def test_the_slice_seed_is_the_old_seed_plus_a_prime_stride():
    """Pure function, pinned: slice 0 is unchanged for every backend."""
    assert T._seed_for(1234, 0) == 1234
    assert T._seed_for(1234, 2) == 1234 + 2 * 7919
    assert T._seed_for(1234, 0, 3) == 1234 + 3 * T._SLICE_SEED_STRIDE
    # Co-prime with `_Gym.reset`'s 7907 episode stride and the 7919 seed stride.
    import math
    assert math.gcd(T._SLICE_SEED_STRIDE, 7907) == 1
    assert math.gcd(T._SLICE_SEED_STRIDE, 7919) == 1


# ==========================================================================
# 4. coherence
# ==========================================================================


def test_shared_buffer_true_is_refused_beside_a_learner_with_no_buffer():
    """A key reading `true` while nothing shares anything is the fabricated pin."""
    with pytest.raises(ConfigError) as exc:
        load("lares", profile="tester", overrides={"train.algorithm": "ppo"})
    assert "train.interaction_cfg.shared_buffer" in str(exc.value)
    with pytest.raises(ConfigError) as exc:
        load("lares", profile="tester", overrides={"train.algorithm": "none"})
    assert "train.interaction_cfg.shared_buffer" in str(exc.value)
    # THE fasttd3 CLAUSE IS AN INVERSE, NOT AN OMISSION. `train.backend:
    # fasttd3` beside `shared_buffer: true` LOADS: the backend honours the slice
    # hand-off's replay half (`fasttd3._fasttd3_prefill_plan`,
    # `_fasttd3_prefill_replay`), so refusing it would be refusing a working
    # config. Asserted, because the CLAUSE IS WHAT PINS THE RULE'S SCOPE: this
    # refusal is about a learner that keeps no replay buffer, and fasttd3 keeps
    # one -- an edit that widened it over an off-policy backend would pass every
    # other line in this test.
    # The clause is asserted in BOTH spellings -- the bare backend, and the
    # backend paired with `train.algorithm: fasttd3` -- because a refusal of
    # the paired form would come ENTIRELY from the shared_buffer rule (measured:
    # without that rule the pair loads), and asserting only the bare backend
    # would leave the paired form unchecked.
    for extra in ({}, {"train.algorithm": "fasttd3"}):
        accepted = load("lares", profile="tester",
                        overrides={"train.backend": "fasttd3", **extra})
        assert accepted["train.interaction_cfg.shared_buffer"] is True, extra
    # The on-policy arm says `false` and must keep loading: `false` beside ppo is
    # the honest arm and is not refused, with or without Eq. 3's buffer-moment
    # scaling turned off beside it.
    for extra in ({}, {"train.reward_scaling": "none"}):
        cfg = load("lares", profile="tester",
                   overrides={"train.algorithm": "ppo",
                              "train.interaction_cfg.shared_buffer": False, **extra})
        assert cfg["train.interaction_cfg.shared_buffer"] is False, extra
    # The rule is scoped to `shared_population`: the default `true` beside the
    # default `independent` + `ppo` is every other shipped config.
    load("eureka", profile="tester")


def test_every_shipped_config_still_validates(tmp_path):
    """`--validate-all` as a test: the rule must refuse nothing that ships."""
    import subprocess
    import sys
    proc = subprocess.run([sys.executable, str(REPO / "bird.py"), "--validate-all"],
                          capture_output=True, text=True, cwd=str(REPO))
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ==========================================================================
# 5. the tester profile says what it did not do
# ==========================================================================


def test_the_mock_backend_ignores_the_buffer_and_the_journal_says_so(tmp_path):
    """A LaRes tester run must not read as having pooled anything.

    The mock learner takes the policy and the seed salt and ignores the replay
    hand-off; the seed row records `replay_prefill_supported: false` and the
    driver journals `interaction_buffer_unsupported` once per round.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    cfg = load("lares", profile="tester", overrides={
        "seed": 0, "train.interaction_allocator": "uniform",
        "train.candidate_parallelism": "sequential"})
    entry.run(cfg, out_root=str(tmp_path))
    (journal,) = sorted(tmp_path.rglob("journal.jsonl"))
    events = [json.loads(l) for l in journal.read_text().splitlines() if l.strip()]
    unsupported = [e for e in events if e.get("stage") == "interaction_buffer_unsupported"]
    closes = [e for e in events if e.get("stage") == "interaction_close"]
    assert closes, "no round ran"
    assert len(unsupported) == len(closes), (
        "one interaction_buffer_unsupported per round on the mock backend, got "
        f"{len(unsupported)} for {len(closes)} rounds")
    assert all(e["backend"] == "mock" for e in unsupported)
    slices = [e for e in events if e.get("stage") == "interaction_slice"]
    assert slices and all(e["buffer_added"] == 0 for e in slices)
    # Seeds still move between an arm's slices on the mock: the salt is
    # backend-agnostic.
    # An arm whose program never compiles has `seed: None` on every slice --
    # no seed was drawn, so none can repeat -- and must not fail this by
    # accident; the claim is about SEEDED slices, and at least one arm must
    # have two of them for it to say anything.
    by_arm = {}
    for e in slices:
        if e.get("seed") is not None:
            by_arm.setdefault(e["cand_id"], []).append(e["seed"])
    assert any(len(v) > 1 for v in by_arm.values()), "no arm ran two seeded slices"
    for cid, seeds in by_arm.items():
        assert len(set(seeds)) == len(seeds), f"{cid} re-used a seed across slices: {seeds}"
    results = sorted(tmp_path.rglob("candidates/iter00_*/train_result.json"))
    assert results, "no train_result.json written"
    rows = [r for f in results for r in json.loads(f.read_text())["seed_metrics"]]
    assert rows and all(r.get("replay_prefill_supported") is False for r in rows)
