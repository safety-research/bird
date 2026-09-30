"""`final_retrain` trains on seeds the search never used.

The phase exists to remove post-selection bias: LIMEN retrains its winner over
ten FRESH seeds because the archive ranked it on three noisy trials
(`configs/methods/limen.yaml`, `final_retrain`). A `_seed_base` with no notion of a
phase would retrain a last-round winner on exactly its own search seeds -- the
first three of the ten "fresh" trials would be the three it was selected on,
and the artifact would report a gap computed against reruns of themselves.
That wrong answer renders as a perfectly plausible retrained fitness, which is
why it gets a test rather than a comment.

The worked counterexample: seed 0, restart 0, winner `last_winner` from
iteration 29 -> search `[60881, 68800, 76719]`; an unsalted retrain would begin
`[60881, 68800, 76719, ...]`.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from bird import registry
from bird.artifacts import RunDir
from bird.budget import Budget
from bird.components.phases import run_final_retrain
from bird.components.training import SEED_STRIDE, _phase_salt, _seed_base
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate

# The counterexample's inputs.
_WINNER = "last_winner"
_LAST_ITERATION = 29
_SEARCH_SEEDS_PER_CANDIDATE = 3   # `configs/methods/limen.yaml` train.seeds_per_candidate
_RETRAIN_SEEDS = 10               # `configs/methods/limen.yaml` final_retrain.n_seeds


def _stream(base: int, n: int) -> list:
    return [base + i * SEED_STRIDE for i in range(n)]


def _ctx(seed: int = 0):
    return SimpleNamespace(cfg={"seed": seed})


def _state(iteration: int, restart: int = 0):
    return SimpleNamespace(restart=restart, iteration=iteration)


def test_the_search_seed_arithmetic_is_unchanged():
    """An EQUALITY, not a floor: every recorded search trained under these
    seeds, and phase salting must not move a single one of them -- a phase
    salts its OWN stream, the search's stays byte-identical."""
    base = _seed_base(_ctx(0), _state(_LAST_ITERATION), Candidate(_WINNER, _LAST_ITERATION, ""))
    assert _stream(base, _SEARCH_SEEDS_PER_CANDIDATE) == [60881, 68800, 76719]
    assert _phase_salt("") == 0, "the search is the unsalted stream"


def test_the_worked_counterexample_no_longer_holds():
    """Winner from the final round: an unsalted retrain would START with its
    search seeds. No retrain seed is a search seed."""
    cand = Candidate(_WINNER, _LAST_ITERATION, "")
    search = _stream(_seed_base(_ctx(), _state(_LAST_ITERATION), cand),
                     _SEARCH_SEEDS_PER_CANDIDATE)
    retrain = _stream(_seed_base(_ctx(), _state(_LAST_ITERATION), cand, "final_retrain"),
                      _RETRAIN_SEEDS)
    assert search == [60881, 68800, 76719]
    assert not set(search) & set(retrain), (search, retrain)
    assert len(set(retrain)) == _RETRAIN_SEEDS


@pytest.mark.parametrize("seed", [0, 1, 7, 12345])
@pytest.mark.parametrize("winner_iteration", [0, 3, _LAST_ITERATION])
@pytest.mark.parametrize("n_search,n_retrain", [(1, 5), (3, 10), (5, 3), (10, 10)])
def test_retrain_stream_is_disjoint_from_the_winners_own_search_stream(
        seed, winner_iteration, n_search, n_retrain):
    """For ANY (`train.seeds_per_candidate`, `final_retrain.n_seeds`) and a
    winner from ANY round, the two streams do not meet. The retrain is anchored
    on the candidate's own iteration, not the loop's last one, because the seeds
    it must not repeat are the ones it was selected on."""
    cand = Candidate(_WINNER, winner_iteration, "")
    search = _stream(_seed_base(_ctx(seed), _state(winner_iteration), cand), n_search)
    retrain = _stream(_seed_base(_ctx(seed), _state(_LAST_ITERATION), cand, "final_retrain"),
                      n_retrain)
    assert not set(search) & set(retrain)


@pytest.mark.parametrize("phase", ["final_retrain", "real_world_eval", "x", "a" * 40])
def test_a_phase_salt_is_never_a_stride_multiple(phase):
    """The disjointness proof: `base + salt + j*S == base + i*S` needs
    `salt == (i-j)*S`, so a salt off every multiple of S cannot collide for any
    i, j. Also bounded so a salted seed stays a legal legacy NumPy seed."""
    salt = _phase_salt(phase)
    assert salt > 0
    assert salt % SEED_STRIDE != 0
    assert salt < 2 ** 24


def test_final_retrain_records_its_seeds_and_they_are_fresh(tmp_path):
    """End to end on the mock backend, the phase as `bird.py` runs it: a search
    training of the winner at its own iteration, then `run_final_retrain` on a
    state whose `best` is that report. `phases/final_retrain.json` must name
    both seed sets and an empty overlap -- the recorded fact a reader checks
    instead of re-deriving the arithmetic. An unsalted retrain fails this with a
    three-seed overlap."""
    registry.load_all()
    cfg = load("limen", profile="tester", overrides={
        "seed": 0,
        "train.env_steps": 200,
        "train.seeds_per_candidate": _SEARCH_SEEDS_PER_CANDIDATE,
        "evaluate.rollouts_per_candidate": 1,
        "final_retrain.n_seeds": 4,
        # LIMEN's own rule reads the archive, which a direct backend call never
        # fills; the seed mechanism does not depend on which rule picks the report.
        "select.final_artifact": "global_best",
    })
    ctx = Context(cfg=cfg, budget=Budget(), env=registry.get("env", cfg["problem.env_id"])({}))
    ctx.rundir = RunDir(tmp_path, "t", "0" * 8)
    reward = ("def compute_reward(s, a, s2):\n"
              "    import numpy as np\n"
              "    o = np.asarray(s2, dtype=float)\n"
              "    return float(-np.linalg.norm(o[0:2] - o[2:4]))\n")
    last = cfg["loop.n_iterations"] - 1
    cand = Candidate(_WINNER, last, reward)

    # Stage 3 + 4 for the winner, at the iteration it was generated in.
    state = RunState(iteration=last, restart=0)
    backend = registry.get("train_backend", cfg["train.backend"])
    search_result = backend(ctx, state, cand, n_seeds=_SEARCH_SEEDS_PER_CANDIDATE)
    score = registry.get("fitness_source", cfg["evaluate.fitness.source"])
    report = score(ctx, state, [search_result])[0]
    state.best = report

    run_final_retrain(ctx, state)

    body = json.loads((ctx.rundir.path / "phases" / "final_retrain.json").read_text())
    search_seeds = [int(m["seed"]) for m in search_result.seed_metrics]
    assert len(search_seeds) == _SEARCH_SEEDS_PER_CANDIDATE
    assert body["search_seeds"] == search_seeds
    assert len(body["seeds"]) == 4 == body["n_seeds"]
    assert len(set(body["seeds"])) == 4
    assert body["seed_overlap"] == []
    assert not set(body["seeds"]) & set(search_seeds)
    assert body["seed_phase"] == "final_retrain"
    assert body["error"] == ""


def test_final_retrain_records_how_its_seeds_were_scheduled(tmp_path, monkeypatch):
    """`phases/final_retrain.json` says whether the seeds ran in parallel, and why not.

    The case is specific: on the batched tier seed workers are forced
    sequential (they fork IN THE PARENT, where jax is initialised), so a
    retrain costs up to N times the wall-clock somebody budgeted for. `n_seeds`
    does not say that; it says how many ran, not whether they ran side by side.

    The journal already carried it. This is about the file a person actually
    opens when they are looking at a retrain: without the field they see an
    unexplained cost and have to guess whether it was a regression.

    Read off the seed rows rather than re-derived, so the phase record cannot
    disagree with the rows it summarises (the journal's key name and the seed
    row's must be the same).
    """
    import bird.components.training as T

    # Force the tier predicate rather than needing a jax env: the field under
    # test is written by the phase, and the phase does not care WHICH env made
    # the schedule sequential.
    monkeypatch.setattr(T, "_cfg_declares_batched", lambda cfg: True)
    monkeypatch.setattr(T, "_worker_is_batched", lambda env: True)

    entry_cfg = load("eureka", profile="tester", overrides={
        "seed": 0, "train.candidate_parallelism": "parallel",
        "generate.n_candidates": 2, "loop.max_parallel_trainings": 2,
        "loop.n_iterations": 1, "train.seeds_per_candidate": 3})

    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bird_entry", repo / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.run(entry_cfg, out_root=str(tmp_path))

    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    rec = json.loads((run / "phases" / "final_retrain.json").read_text())

    assert rec.get("seed_workers") == 1, (
        f"the retrain ran its seeds sequentially but the record says "
        f"seed_workers={rec.get('seed_workers')!r}")
    assert rec.get("seed_schedule_reason"), (
        "seed_workers=1 with no reason is indistinguishable from a run that "
        "only ever asked for one worker; the downgrade must explain itself")
    reason = rec["seed_schedule_reason"]
    assert "batched" in reason and "sequential" in reason, reason
