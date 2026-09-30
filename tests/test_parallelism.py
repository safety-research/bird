"""`train.candidate_parallelism` -- the schedule must not change the science.

Two keys, `train.candidate_parallelism` and `loop.max_parallel_trainings`, are
honoured rather than merely declared: a key the algorithm cannot honour -- like
a `temperature` field the LLM client ignores -- is a fabricated pin.

Honouring them is only safe if `parallel` is *provably* the same run as
`sequential`, because `sequential` is the default every published config
inherits: if flipping the key moved a number, the key would be changing the
method rather than the schedule, and every `--diff` between two configs that
disagreed on it would be unreadable. Hence the first test, which compares whole
run directories rather than a summary scalar.

WHAT THE COMPARISON HAS TO NORMALISE, and why each one is not a cop-out:

  * `wallclock_s` / `gpu_seconds` / `eval_wall_s` are wall-clock. Parallel
    candidates contend; these are the numbers that are *supposed* to move.
    The list lives in `_VOLATILE`; a timing field added to a compared
    artifact without an entry there turns every bit-identity case red.
  * `config_hash` and `config.resolved.yaml` differ because the two runs differ
    by exactly the overrides under test.
  * `t` in `journal.jsonl` is a timestamp, and the journal's absolute video
    paths embed the run directory's own name (`<name>-<hash>-<timestamp>/`).
  * JSON key ORDER varies run to run even between two sequential runs, because
    `training.py::_mean_curve` builds each checkpoint from
    `set().union(*(set(c[i]) for c in curves))` and string-set iteration order
    is hash-salted. Compared with `sort_keys=True` rather than as bytes; the
    suite does not control `PYTHONHASHSEED`, and pinning it would hide the
    problem rather than normalise it.

Everything else -- every fitness, every seed metric, every checkpoint, every
budget counter, every reward program, every `meta.json`, every rendered frame
-- is compared exactly.

MEASURED, eureka on pendulum with the real SB3 backend, 8 candidates x 4000
steps: sequential 259.34 s / 257.96 s, parallel
66.65 s => 3.89x end to end and 4.27x on stage 3, with all three run
directories (53 files, mp4s included) byte-identical after the normalisation
above. See `MEASURED_SPEEDUP` in `bird/components/training.py`.
"""

import ast
import importlib.util
import json
import os
import random
import re
import signal
from pathlib import Path

import pytest

from conftest import (GT_PUBLISHED_OVERRIDES, TESTER_SEARCH_CAP,  # noqa: F401
                      apply_search_cap, REPO)
from bird import registry
from bird.budget import Budget, BudgetExceeded
from bird.config import load
from bird.context import Context
from bird.types import Candidate, TrainResult

#: Not in the tester-tier smoke suite (heavyweight execution: forks one process per candidate and diffs whole run dirs).
#: Deselected by `-m "not slow"`. See pyproject.toml.
pytestmark = pytest.mark.slow


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# run-directory fingerprint
# --------------------------------------------------------------------------

#: Timing, and the identity of the run itself. Nothing here is a result.
# `run` is here for the same reason `config_hash` is, and is the same fact: a
# run directory is named `<config>-<confighash>-<stamp>`, and the two runs under
# comparison differ by exactly the override that selects the schedule, so their
# hashes and therefore their names MUST differ. `preferences.jsonl` carries the
# name in every record (it is what makes a pooled dataset joinable back to its
# run), which made it the first artifact to put that name inside a file rather
# than only in the path `_scrub` already rewrites.
#: `eval_wall_s` is a WALL CLOCK ON THE CHECKPOINT POINT, so it lands inside
#: `train_result.json`'s `checkpoints` -- deeper than the other entries here,
#: which sit at the top of a record. Left out of this set it turns every
#: bit-identity case red: a timing field in a compared artifact makes
#: `parallel` differ from `sequential` on every run, which is the
#: scheduling-knob-becomes-a-science-knob failure this file exists to prevent.
_VOLATILE = {"wallclock_s", "gpu_seconds", "t", "pid", "host", "argv", "started",
             "updated", "elapsed_s", "slurm_job_id", "slurm_array_task_id",
             "config_hash", "run", "eval_wall_s",
             # -- SCHEDULE FACTS, a DIFFERENT line from
             # `eval_wall_s`'s. That one is a clock; these differ because
             # the two runs under comparison ARE two different schedules --
             # `seed_workers` is 1 under `sequential` and N under
             # `parallel` by definition, so comparing it asserts the two
             # schedules are the same schedule. The rest is the spawn
             # worker's instrumentation. A field that differs because the
             # SCHEDULE differed is volatile; a field that differs because
             # one run was not verified is evidence, and belongs nowhere
             # near this set.
             "seed_workers", "seed_schedule_reason",
             "worker_start", "worker_startup_s", "startup_s", "xla_env",
             "jax_cache",
             # -- CLOCKS AND SAMPLES ON THE FASTTD3 SEED ROW.
             # `jit_compile_s` is a real clock (`+=` around the first call
             # of each jitted region) that lands on the seed row, and the
             # three `gpu_utilization_*` fields are `nvidia-smi` samples of
             # a busy machine. A varying number inside a compared artifact
             # turns every bit-identity case red, three files from the
             # cause. LATENT here only because no bit-identity case runs
             # fasttd3.
             #
             # `_CLOCK_SHAPED` catches none of them and cannot be made to
             # cheaply: it is a SUBSTRING list, so adding `_s` would match
             # `env_steps`, `train_steps` and every other count, and
             # `_pct`/`_n_samples` are not clock-shaped at all. Hand-listed,
             # like `wallclock_s` always has been.
             #
             # `reward_sync_s` is an accumulated clock on the same object,
             # with the same latency and the same blast radius; leaving it
             # out would put the next fasttd3 comparison red for a different
             # name.
             "jit_compile_s", "reward_sync_s",
             "gpu_utilization_max_pct", "gpu_utilization_mean_pct",
             "gpu_utilization_n_samples"}

#: Journal stages that exist on ONE schedule only, dropped as whole ROWS.
#:
#: `_VOLATILE` strips KEYS and `_CLOCK_SHAPED` catches clock-shaped NAMES;
#: neither can catch this, because there is no offending key -- there is an
#: offending ROW. A spawned wave writes `worker_start` and `worker_wave`
#: and a sequential run writes neither, so the extra lines fail the
#: comparison on their EXISTENCE and the failure reads as a science
#: difference.
_SCHEDULE_ONLY_STAGES = {"worker_start", "worker_wave"}

#: Name shapes that are almost certainly a clock. A field matching one of
#: these and NOT declared volatile is a clock missing from `_VOLATILE`, so
#: `test_a_timing_field_cannot_enter_a_compared_artifact` fails on it rather
#: than every bit-identity case going red for a reason whose cause is three
#: files away.
#: SUFFIXES, checked with `endswith`, and the distinction from the substring
#: list above is the whole point. `_s` is the suffix every duration in this repo
#: carries -- `wallclock_s`, `eval_wall_s`, `jit_compile_s`, `reward_sync_s`
#: -- and it CANNOT join `_CLOCK_SHAPED`, which is matched as a substring:
#: measured on one tester run, `_s` as a substring matches `env_steps`,
#: `env_steps_cap`, `env_steps_per_seed`, `config_train_env_steps`,
#: `n_seeds`, `n_subtasks`, `search_seeds`, `init_source`, `init_similarity`,
#: `per_seed_fitness`, `post_selection_gap` and `seed_schedule_reason` --
#: twelve counts and strings in one small run.
_CLOCK_SUFFIXES = ("_s",)

#: Names that END in a clock suffix and are NOT clocks. One entry, and it
#: earns its exemption: `min_compile_s` (fasttd3.py, inside `jax_cache`) is
#: `jax_persistent_cache_min_compile_time_secs` -- a CONFIG-DERIVED
#: PROVENANCE CONSTANT. Declaring it volatile to satisfy the predicate would
#: be actively wrong: two runs configured with different cache thresholds
#: would then compare EQUAL, which is a real difference hidden to silence a
#: guard. The classification test asserts this name is exempt-not-clock
#: precisely so nobody removes the exemption and papers the failure over with
#: a `_VOLATILE` entry, which would look identical from the outside.
_NOT_A_CLOCK = frozenset({"min_compile_s"})

_CLOCK_SHAPED = ("_wall_s", "_wall", "wall_", "_seconds", "_elapsed", "elapsed_",
                 "_duration", "duration_", "_ms", "_secs")


def scrub_volatile(obj):
    """`_VOLATILE` applied RECURSIVELY. The shared entry point for any test
    that compares seed rows, curves or checkpoint points for equality.

    PUBLIC (no underscore) BECAUSE OTHER FILES IMPORT IT. A file that imports
    `_VOLATILE` from here rather than keeping its own copy inherits every new
    entry for free, where hand-kept copies go red or latent. That import is
    the model; this function is the other half of it, because the list alone
    is not enough:

    RECURSION IS THE POINT. A seed row NESTS its curve (`"checkpoints": curve`
    in fasttd3.py and training.py) and the clock is on each checkpoint point.
    A top-level dict comprehension cannot reach it: adding a nested clock to a
    top-level strip leaves the comparison failing identically.

    Independent hand-kept strip lists are the deny-list blind spot: a name
    guard does not catch a stale copy, and neither does a careful author.
    """
    if isinstance(obj, dict):
        return {k: scrub_volatile(v) for k, v in sorted(obj.items()) if k not in _VOLATILE}
    if isinstance(obj, list):
        return [scrub_volatile(v) for v in obj]
    return obj


def _is_clock_shaped(name):
    """Does this field NAME look like a clock? Substring list, then suffixes.

    Two mechanisms because the two kinds of evidence differ: `_wall`,
    `_elapsed` and friends are fragments that can sit anywhere in a name,
    while `_s` is only a duration at the END (`env_steps` is a count). A
    single list cannot express both, and the one-character version of this
    change -- adding `_s` to `_CLOCK_SHAPED` -- classifies every count in the
    repo as a clock.
    """
    if name in _NOT_A_CLOCK:
        return False
    return (any(m in name for m in _CLOCK_SHAPED)
            or name.endswith(_CLOCK_SUFFIXES))


def _clock_shaped_keys(obj, out=None):
    """Every key anywhere in a JSON payload whose NAME looks like a clock."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_clock_shaped(k):
                out.add(k)
            _clock_shaped_keys(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _clock_shaped_keys(v, out)
    return out


def _scrub(obj, run_dir: str):
    if isinstance(obj, dict):
        return {k: _scrub(v, run_dir) for k, v in sorted(obj.items()) if k not in _VOLATILE}
    if isinstance(obj, list):
        return [_scrub(v, run_dir) for v in obj]
    if isinstance(obj, str):
        return obj.replace(run_dir, "<RUNDIR>")
    return obj


def _fingerprint(root: Path) -> dict:
    """Every artifact a run wrote, canonicalised. `config.resolved.yaml` is
    excluded: the two runs under comparison differ by exactly the override that
    selects the schedule, so it is the one file that MUST differ."""
    (run,) = [p for p in Path(root).iterdir() if p.is_dir()]
    rd = str(run)
    out = {}
    for f in sorted(run.rglob("*")):
        if not f.is_file() or f.name == "config.resolved.yaml":
            continue
        rel = str(f.relative_to(run))
        if f.suffix == ".json":
            out[rel] = json.dumps(_scrub(json.loads(f.read_text()), rd), sort_keys=True)
        elif f.suffix == ".jsonl":
            rows = [json.loads(line) for line in f.read_text().splitlines()
                    if line.strip()]
            out[rel] = "\n".join(
                json.dumps(_scrub(r, rd), sort_keys=True) for r in rows
                if not (isinstance(r, dict)
                        and r.get("stage") in _SCHEDULE_ONLY_STAGES))
        else:
            try:
                out[rel] = f.read_text().replace(rd, "<RUNDIR>")
            except UnicodeDecodeError:  # rendered frames
                out[rel] = f.read_bytes().hex()
    return out


def _run(name, out, profile="tester", extra=None, **overrides):
    # `extra` is a plain dict, not **kwargs, because a config override can be
    # named `name` (GT_PUBLISHED_OVERRIDES is) and would collide with the
    # positional parameter if splatted.
    overrides = {**(extra or {}), **overrides}
    overrides.setdefault("seed", 0)
    # See conftest.apply_search_cap: the profile shrinks the training, not the
    # search, and singh_orp's published width is 3,240 candidates. A CEILING --
    # capping must never raise a config published below it.
    overrides = apply_search_cap(name, overrides, profile)
    _entry().run(load(name, overrides=overrides, profile=profile), out_root=str(out))
    return _fingerprint(out)


#: A deliberately chosen cross-section rather than every tester config: one
#: config per concurrency-hostile mechanism, so the suite stays inside a
#: minute while still exercising every way a worker could corrupt a
#: neighbour. Every tester config was verified by hand at four worker counts
#: (auto/1/2/8); these are the ones that would catch a regression.
#: Each entry is (config, extra overrides, why). The overrides slot exists for
#: exactly one member: the published GT point is not a file of its own (one
#: config per method) but gt + `conftest.GT_PUBLISHED_OVERRIDES`, and its
#: subject here -- the
#: `_REPLAY_STORE` handoff -- is worth exercising at the published point, with
#: the human channel on, rather than only at the closed-loop arm.
_CROSS_SECTION = [
    # live-config mutation: `_overlaid` rewrites Config._data in place, and a
    # thread-shared config let candidate B observe candidate A's grid point.
    ("singh_orp", {}, "train.hyperparameter_search: per_candidate_best"),
    # module-level `_REPLAY_STORE` handoff -- degrades SILENTLY to an empty
    # buffer across a process boundary, i.e. runs the ablation, not the method.
    ("gt", GT_PUBLISHED_OVERRIDES, "train.init: secondary_replay_buffer"),
    # module-level `_POLICY_STORE` handoff.
    ("rda", {}, "train.init: warm_start_from_best"),
    # per-candidate LLM-written DR blob + `env.set_dr(None)` in the finally,
    # plus `failure_detection: stdout_grep` (process-global sys.stdout).
    ("dreureka", {}, "domain_randomization: generated + stdout_grep"),
    # `policy_trainings_skipped` -- the counter CARD's whole claim rests on.
    ("limen", {}, "cascade screen; skips must survive"),
    # evolution + `select.tie_break: first`, which is order-dependent.
    ("eureka", {}, "iteration_best parents; tie_break: first"),
    # `fitness is None` must stay distinct from 0.0.
    ("card", {}, "no ranking scalar at all"),
    # `search_tree` backup runs in candidate-index order and `sim_index` is an
    # insertion counter: reassembly by completion order would move the decay.
    # n_candidates stated so `apply_search_cap` leaves the published width: the
    # action counts sum to 8 and the coherence check refuses a capped pool.
    ("rf_agent", {"generate.n_candidates": 8}, "search_tree: sim_index + backup order"),
    # `warm_start_from_parent` under `sequential_conditioned`: sample i's parent is
    # sample i-1 of the SAME round, whose policy the sequential loop has stored
    # and a forked child has not. Accepting it would make the two schedules
    # train different candidates, so the plan takes only an EARLIER
    # iteration's parent.
    ("hillclimb/v4", {"generate.sampling_mode": "sequential_conditioned",
                      "train.init": "warm_start_from_parent",
                      "generate.n_candidates": 3, "loop.n_iterations": 2,
                      "generate.candidate_schedule": "constant",
                      "generate.candidate_schedule_values": [],
                      "final_retrain.enabled": False, "post": []},
     "warm_start_from_parent + sequential_conditioned: same-round parent"),
    # `train.checkpoint_selection: best_by_reward` restores an earlier
    # checkpoint's parameters INSIDE `run_seed`, i.e. inside a candidate worker
    # and, on the retrain path, inside a forked seed worker; the restored
    # parameters must be what the child serialises and the parent rolls out.
    # Two seeds so the surrogate's seed fork is exercised as well.
    ("eureka", {"name": "eureka_best_ckpt", "train.checkpoint_selection": "best_by_reward",
                "train.seeds_per_candidate": 2},
     "checkpoint_selection: restore inside the worker, ship the restored snapshot"),
    # `train.interaction: shared_population` (LaRes): the WAVES are sequential
    # and each wave's arms go through this schedule; the round-scoped replay
    # hand-off (`_ROUND_REPLAY`) crosses the fork in the child payload, and
    # Eq. 3's plan is made in the parent (`moment_samples: 50` makes its
    # `ctx.rng` draw fire on the tester buffer -- a draw that, made inside the
    # worker, would move the Thompson allocator's waves).
    ("lares", {"train.interaction_cfg.moment_samples": 50},
     "shared_population waves + round replay hand-off + parent-side Eq. 3 plan"),
]


@pytest.mark.parametrize("name,extra,why", _CROSS_SECTION,
                         ids=[ov.get("name", n) for n, ov, _ in _CROSS_SECTION])
@pytest.mark.parametrize("workers", [2, 8])
def test_parallel_is_bit_identical_to_sequential(name, extra, why, workers, tmp_path):
    """The test that matters. Same seed, same everything -- including the
    winner, which no fitness comparison would catch on its own.

    Parametrised over TWO worker counts on purpose. Each worker pins exactly
    one BLAS/OpenMP thread precisely so that the worker count cannot be a
    numerics knob; if a future change divided a thread budget instead
    (`cpus // workers`), `loop.max_parallel_trainings` would silently start
    changing torch's reduction order and therefore the fitness values. One
    worker count would not notice.
    """
    seq = _run(name, tmp_path / "seq", extra=extra, **{
        "train.candidate_parallelism": "sequential"})
    par = _run(name, tmp_path / f"par{workers}", extra=extra, **{
        "train.candidate_parallelism": "parallel",
        "loop.max_parallel_trainings": workers})

    assert set(seq) == set(par), (
        f"{name} ({why}): different artifact file sets\n"
        f"  only sequential: {sorted(set(seq) - set(par))}\n"
        f"  only parallel:   {sorted(set(par) - set(seq))}")
    differing = [k for k in sorted(seq) if seq[k] != par[k]]
    assert not differing, f"{name} ({why}): parallel changed {differing}"


#: (config, extra) per equality row that needs its own control. A control run
#: on a DIFFERENT cell proves the fingerprint works somewhere, not that it
#: works on the cell whose row it guards.
_CONTROL_CELLS = [
    pytest.param("eureka", {}, id="eureka"),
    # NO JAX CELL, and the reason is the refusal one file over. A control for a
    # jax row would have to LOAD `candidate_parallelism: parallel` on a
    # `batched` env where that is not yet exercised by an equality row here.
    # When one is added, its control belongs beside it.
]


# Params go through AS `pytest.param` objects rather than as plain tuples.
# There is only one cell today so nothing carries `marks`, but a rebuild into
# tuples DROPS them, and a later jax cell would then run unmarked -- erroring
# in a venv with no jax instead of skipping, and collected by a CI job that
# cannot import it. Keeping the param form now is what stops that being a live
# defect later.
@pytest.mark.parametrize("cell,cell_extra", _CONTROL_CELLS)
def test_the_equality_above_can_actually_fail(cell, cell_extra, tmp_path, monkeypatch):
    """The negative control for `test_parallel_is_bit_identical_to_sequential`.

    An equality never seen red is one whose subject might not be reachable: if
    `_fingerprint` compared nothing a worker can move, every row above would
    pass on a broken tree and say nothing.

    WHAT IS BROKEN, AND WHY NOT A CONSTANT. `_seed_for` is a pure function of
    (base_seed, seed_i, slice_index), so ADDING A CONSTANT shifts sequential
    and parallel IDENTICALLY and the equality still holds -- a control that
    breaks the science without breaking the comparison, which proves nothing
    while looking like a control. The defect this guards is a seed riding on
    something the fork changes, so the control salts with the PID: one process
    under `sequential`, one per candidate under `parallel`.

    The evidence is a PAIR. This row going red is only meaningful beside the
    unmodified equality going green on the same tree -- otherwise it would
    equally indicate a noisy fingerprint.
    """
    import bird.components.training as _tr

    real = _tr._seed_for
    monkeypatch.setattr(_tr, "_seed_for",
                        # THE FULL PID, NOT `% 7`: a modulus can
                        # collide with the parent's residue, and every child
                        # colliding at once is rare rather than impossible --
                        # which would show up as a SPURIOUS RED here (no
                        # differences found), not as a false pass. Same shape,
                        # no collapse.
                        lambda b, i, s=0: real(b, i, s) + os.getpid())

    ex = {"generate.n_candidates": 2, "final_retrain.enabled": False, "post": [],
          **cell_extra}
    seq = _run(cell, tmp_path / "seq", extra=ex,
               **{"train.candidate_parallelism": "sequential"})
    par = _run(cell, tmp_path / "par", extra=ex,
               **{"train.candidate_parallelism": "parallel",
                  "loop.max_parallel_trainings": 2})

    differing = [k for k in sorted(set(seq) | set(par)) if seq.get(k) != par.get(k)]
    assert differing, (
        "THE INSTRUMENT IS BLIND. A pid-salted seed makes parallel and "
        "sequential genuinely different runs, and `_fingerprint` reported them "
        "identical -- so the equality above would pass on a tree where a worker "
        "had corrupted its candidate's seed. Fix the fingerprint, not this test.")


@pytest.mark.parametrize("name", ["limen", "gt", "eureka"])
def test_budget_counters_survive_the_process_boundary(name, tmp_path):
    """Every counter, not just the total -- a worker's `record_training` lands
    in an address space the parent never sees unless it is merged back.

    `limen` and `gt` are here because they SKIP candidates: without a
    nonzero `policy_trainings_skipped` this test would pass on 0 == 0 and would
    say nothing about the one number CARD's contribution is made of.
    """
    entry = _entry()
    got = {}
    for mode in ("sequential", "parallel"):
        cfg = load(name, profile="tester", overrides={
            "seed": 0, "train.candidate_parallelism": mode,
            "loop.max_parallel_trainings": 3})
        got[mode] = entry.run(cfg, out_root=str(tmp_path / mode))["budget"]

    # `wallclock_s` and `gpu_seconds` are wall-clock: contention is supposed to
    # move them, and only them.
    for key in sorted(set(got["sequential"]) - {"wallclock_s", "gpu_seconds"}):
        assert got["parallel"][key] == got["sequential"][key], (
            f"{name}: budget.{key} {got['parallel'][key]} != {got['sequential'][key]}")

    if name in ("limen", "gt"):
        assert got["parallel"]["policy_trainings_skipped"] > 0, (
            "this config is supposed to skip candidates; if it stopped doing so "
            "the equality above is 0 == 0 and CARD's counter is untested")


@pytest.mark.parametrize("cap", [1, 3, 5])
def test_a_budget_cap_fires_at_the_same_candidate(cap, tmp_path):
    """A capped run must not quietly become an uncapped one.

    The failure this guards is not a slow counter but a silent one: if worker
    deltas were dropped, `budget.max_policy_trainings` would never be reached
    and the run would keep launching RL jobs. Comparing the artifact FILE SET
    as well as the counters is what pins *where* the cap fired -- the same
    counters with a different stopping candidate would mean the merge order
    had drifted off candidate order.
    """
    entry = _entry()
    seen = {}
    for mode in ("sequential", "parallel"):
        out = tmp_path / mode
        cfg = load("eureka", profile="tester", overrides={
            "seed": 0, "budget.max_policy_trainings": cap,
            "train.candidate_parallelism": mode, "loop.max_parallel_trainings": 4})
        try:
            entry.run(cfg, out_root=str(out))
        except BudgetExceeded:
            pass
        (run,) = [p for p in out.iterdir() if p.is_dir()]
        seen[mode] = (
            json.loads((run / "budget.json").read_text()),
            sorted(str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()),
        )

    seq_budget, seq_files = seen["sequential"]
    par_budget, par_files = seen["parallel"]
    assert par_budget["policy_trainings"] == seq_budget["policy_trainings"]
    assert par_budget["env_steps"] == seq_budget["env_steps"]
    assert par_files == seq_files, "the cap fired at a different candidate"


def test_parallelism_parallel_actually_sizes_its_wave(tmp_path, monkeypatch):
    """That `wave_under_cap` is CALLED, not merely correct.

    With the call site deleted, the pure-function tests above still pass. A
    helper can be perfect and unreached.

    Observed through `resolve_workers`, which takes the job count as its
    second argument immediately after the sizing and immediately before the
    fork loop. End-state artefacts cannot serve here -- at cap 1 the
    candidate directories are EMPTY ON BOTH PATHS because the run aborts
    before any is written, so a test on them asserts [] == [] and passes
    against no sizing at all.
    """
    from bird.components import training as T

    seen = []
    real = T.resolve_workers
    monkeypatch.setattr(T, "resolve_workers",
                        lambda cfg, n_jobs: seen.append(n_jobs) or real(cfg, n_jobs))

    entry = _entry()
    cfg = load("eureka", profile="tester", overrides={
        "seed": 0, "budget.max_policy_trainings": 1,
        "train.candidate_parallelism": "parallel", "loop.max_parallel_trainings": 4})
    try:
        entry.run(cfg, out_root=str(tmp_path / "run"))
    except BudgetExceeded:
        pass

    assert seen, "resolve_workers was never called; this probe no longer observes the fork"
    assert seen[0] == 1, (
        f"the first wave was sized {seen[0]}, not 1. `generate.n_candidates` is "
        f"16 here and the cap admits one training, so `wave_under_cap` either "
        f"was not called or did not trim -- a helper that is correct and "
        f"unreached is what this test exists to catch.")


def test_every_cap_binds_before_the_work_counting_attempts_not_completions():
    """All three caps, and COUNTING ATTEMPTS is the whole method.

    A probe that counts successful returns certifies a broken cap as working
    -- because the call that raises is the one already paid for. A cap on a
    paid external service that lets the crossing call happen and then reports
    it is not a cap; counting completions cannot see the difference, and
    counting attempts is the only thing that can.

    A predicate that can ANSWER for a cap is not a cap unless something ASKS
    it before the work: `max_llm_calls: 40` must not permit 41 API calls. A
    capability with no call site is the same defect as a helper that is never
    called.
    """
    from bird.budget import Budget

    # TRAININGS: attempts, driven through the real record path.
    b = Budget(max_policy_trainings=2)
    attempts = 0
    for _ in range(6):
        if b.would_exceed_training(1):
            break
        attempts += 1           # the work would happen HERE
        b.record_training()
    assert attempts == 2, f"cap 2 permitted {attempts} trainings to START"

    # LLM CALLS: the same shape; an unasked predicate would read 3.
    b = Budget(max_llm_calls=2)
    attempts = 0
    for _ in range(6):
        if b.would_exceed(llm_calls=1):
            break
        attempts += 1           # the provider round-trip would happen HERE
        b.record_llm()
    assert attempts == 2, f"cap 2 permitted {attempts} API calls to be MADE"

    # GPU HOURS: a duration is not knowable in advance, so the enforceable
    # rule is "do not START once spent" and the predicate differs in kind.
    # Two half-hour units fit under a 1-hour cap; the third must not start.
    b = Budget(max_gpu_hours=1.0)
    attempts = 0
    for _ in range(6):
        if b.gpu_hours_exhausted():
            break
        attempts += 1
        b.record_training(gpu_seconds=1800)
    assert attempts == 2, f"cap 1 h permitted {attempts} half-hour units to START"


def test_the_llm_client_asks_before_the_round_trip(monkeypatch):
    """That the LLM pre-flight is CALLED, not merely correct.

    The unit test above proves `Budget.would_exceed` answers for llm calls.
    It does NOT prove the client asks -- the same distinction as
    `wave_under_cap` being correct but unreached. Deleting the pre-flight from
    `LLMClient.__call__` leaves the unit test green, so this is the one that
    has to fail.

    Counts the PROVIDER round-trips ATTEMPTED, by stubbing `_complete`: the
    call that raises is the one already paid for, so completions cannot see
    the defect and attempts can.
    """
    from bird.budget import Budget, BudgetExceeded
    from bird.llm.base import LLMClient

    attempts = []

    class _Probe(LLMClient):
        def _complete(self, *a, **kw):
            attempts.append(1)
            self._record(prompt_tokens=1, completion_tokens=1)
            n = a[1] if len(a) > 1 else kw.get("n", 1)
            return [""] * n

    import types as _t
    ctx = _t.SimpleNamespace(
        cfg=load("eureka", profile="tester"), budget=Budget(max_llm_calls=2),
        rng=None, counters={})
    client = _Probe(ctx, role="generator")

    made = 0
    for _ in range(6):
        try:
            client([{"role": "user", "content": "x"}])
            made += 1
        except BudgetExceeded:
            break
    assert len(attempts) == 2, (
        f"the provider was reached {len(attempts)} times under a cap of 2. "
        f"The pre-flight in LLMClient.__call__ is missing or unreached -- a "
        f"predicate nothing asks is not a cap.")


def test_the_wave_is_sized_to_the_cap_before_the_fork():
    """WORK parity, beside the counter parity `test_a_budget_cap_fires_at_the_
    same_candidate` pins. Those are different claims and only the first held.

    A `parallel` that forked every trainable candidate and threw away the
    results past the cap would keep the counters in agreement while the
    COMPUTE differed: the discarded candidates' evaluation rollouts are real
    env interaction no artifact records (at cap=1, sequential would evaluate
    one candidate and parallel sixteen). The rule is bit-identity, so that
    trade is not one to make.

    Asserted on `wave_under_cap` DIRECTLY rather than through a run: comparing
    the two paths' candidate directory listings at cap=1 compares two EMPTY
    lists -- the run aborts before any candidate dir is written -- so it would
    assert [] == [] and pass with the sizing removed entirely. A behavioural
    observable that cannot distinguish the defect is worse than none: it
    reads as coverage.
    """
    from bird.components.training import wave_under_cap

    # No cap: nothing is ever trimmed.
    assert wave_under_cap([0, 1, 2], None, 0, 1) == ([0, 1, 2], [])

    # cap=1, nothing spent, one seed: exactly ONE training on both paths.
    # The candidate that would cross does not run at all -- the cap binds
    # before the work, not after it.
    assert wave_under_cap(list(range(16)), 1, 0, 1) == ([0], list(range(1, 16)))

    # Part-way through: three of five spent leaves room for two more.
    assert wave_under_cap(list(range(8)), 5, 3, 1) == ([0, 1], [2, 3, 4, 5, 6, 7])

    # SEEDS. Three seeds per candidate and a cap of six: exactly two
    # candidates fit. A candidate count would say six here and be wrong by a
    # factor of the seed count.
    assert wave_under_cap(list(range(8)), 6, 0, 3) == ([0, 1], [2, 3, 4, 5, 6, 7])

    # Already at the cap: nothing runs, and the caller raises.
    assert wave_under_cap([0, 1, 2], 4, 4, 1) == ([], [0, 1, 2])


def test_one_worker_crashing_does_not_lose_the_others(tmp_path):
    """A dead worker is ONE failed candidate, never a failed iteration.

    Called against the registered runner directly rather than through a search,
    because the three interesting deaths cannot be provoked from a config: a
    SIGSEGV, a SIGKILL (the OOM killer, or a job scheduler), and an ordinary
    exception.

    This is also the case that rules out `ProcessPoolExecutor`, which answers a
    worker segfault with `BrokenProcessPool` on every pending future and so
    loses the neighbours' results -- and the case that makes `parallel` an
    improvement rather than a wash on the metaworld tier, where a MuJoCo
    SIGSEGV under `sequential` kills the interpreter outright and leaves
    `status.json` reading `running` forever.
    """
    import ctypes

    registry.load_all()
    runner = registry.get("candidate_parallelism", "parallel")
    cfg = load("eureka", profile="tester",
               overrides={"loop.max_parallel_trainings": 3})
    ctx = Context(cfg=cfg, budget=Budget(), env=registry.get("env", "toy_reacher")({}))

    cands = [Candidate(cand_id=f"c{i:04d}", iteration=0, reward_code="") for i in range(6)]
    cands[4] = cands[4].failed("screened by test", kind="screened")
    cands[4].screened_out, cands[4].valid = True, True

    def run_one(c: Candidate) -> TrainResult:
        # Inside the worker. pytest installs a faulthandler that would dump a
        # C-level traceback to stderr for the deliberate SIGSEGV below; the
        # crash is the point of the test, the 40 lines of stack are not. Only
        # the child's copy is affected, so a real segfault elsewhere still
        # reports normally.
        import faulthandler
        faulthandler.disable()
        i = int(c.cand_id[1:])
        ctx.budget.record_training(env_steps=100)
        if i == 1:
            ctypes.string_at(0)                     # SIGSEGV
        if i == 2:
            os.kill(os.getpid(), signal.SIGKILL)    # SIGKILL
        if i == 3:
            raise RuntimeError("candidate 3 exploded")
        return TrainResult(cand_id=c.cand_id, candidate=c, trained=True, env_steps_used=100)

    results = runner(ctx, None, cands, run_one, lambda c, r: None)

    assert [r.cand_id for r in results] == [c.cand_id for c in cands], \
        "results must come back in candidate order, or tie_break: first changes winners"
    assert results[0].trained and results[5].trained, "a crash took its neighbours with it"
    assert not results[0].error and not results[5].error
    for i in (1, 2, 3):
        assert not results[i].trained
        assert results[i].error.startswith("worker: "), results[i].error
    assert results[4].skip_reason and not results[4].trained
    # Two survivors charged, one skip counted, the three dead workers' partial
    # spend discarded -- a worker that never reported cannot report a cost.
    assert ctx.budget.policy_trainings == 2
    assert ctx.budget.policy_trainings_skipped == 1
    assert ctx.budget.env_steps == 200


def _wave(n_candidates, dies, tmp_path=None):
    """Run one wave of `n_candidates` through the registered `parallel` runner,
    with `run_one` dying in every worker whose index is in `dies`."""
    registry.load_all()
    runner = registry.get("candidate_parallelism", "parallel")
    cfg = load("eureka", profile="tester",
               overrides={"loop.max_parallel_trainings": n_candidates})
    ctx = Context(cfg=cfg, budget=Budget(), env=registry.get("env", "toy_reacher")({}))
    cands = [Candidate(cand_id=f"c{i:04d}", iteration=0, reward_code="")
             for i in range(n_candidates)]

    def run_one(c: Candidate) -> TrainResult:
        i = int(c.cand_id[1:])
        if i in dies:
            raise RuntimeError(f"candidate {i} exploded")
        ctx.budget.record_training(env_steps=100)
        return TrainResult(cand_id=c.cand_id, candidate=c, trained=True, env_steps_used=100)

    return runner, ctx, cands, run_one


def test_a_wave_of_all_failed_workers_fails_the_run():
    """The whole point: a wave that trained NOTHING must not read as a result.

    Without it, a run with every worker dead exits **0** with `[3] train ->
    0 trained, 3 failed` in an INFO line. An exit code is what a script and a
    human both read first, so a dead run that reports success
    is a dead run nobody notices.

    The message is asserted, not just the exception type: a raise carrying no
    count and no worker error moves the problem from "no signal" to "a signal
    that says nothing", and the count against the wave's own size is what
    separates this from one unlucky candidate.
    """
    from bird.components.training import WaveFailed

    runner, ctx, cands, run_one = _wave(3, dies={0, 1, 2})
    with pytest.raises(WaveFailed) as excinfo:
        runner(ctx, None, cands, run_one, lambda c, r: None)

    msg = str(excinfo.value)
    assert "[3] train" in msg, msg                      # which stage
    assert "3 of 3" in msg, msg                         # how many of K
    assert "c0000" in msg, msg                          # which candidate first
    assert "candidate 0 exploded" in msg, msg           # the worker's error, verbatim


def test_a_partial_wave_failure_reports_the_count_and_keeps_going(caplog):
    """One dead worker is still ONE failed candidate -- with a count now.

    This is the property `test_one_worker_crashing_does_not_lose_the_others`
    protects, asserted from the other side: the all-failed raise must not fire
    on a wave that still trained something, because a search surviving a dead
    candidate is the reason the fork path exists at all.
    """
    import logging

    runner, ctx, cands, run_one = _wave(3, dies={1})
    with caplog.at_level(logging.ERROR, logger="bird.train"):
        results = runner(ctx, None, cands, run_one, lambda c, r: None)

    assert [r.cand_id for r in results] == [c.cand_id for c in cands]
    assert results[0].trained and results[2].trained
    assert not results[1].trained and results[1].error.startswith("worker: ")
    assert ctx.budget.policy_trainings == 2

    summary = [r.getMessage() for r in caplog.records
               if "candidate-parallelism worker(s) failed" in r.getMessage()]
    assert len(summary) == 1, caplog.text
    assert "1 of 3" in summary[0], summary[0]
    assert "c0001" in summary[0] and "candidate 1 exploded" in summary[0], summary[0]
def test_a_skipped_candidate_cannot_disarm_the_all_failed_guard():
    """The denominator is the workers that LAUNCHED, not the candidates.

    The per-arm-denominator mistake in a different costume. A failing wave
    can look like this:

        c0003  worker: <dead>
        c0004  skip_reason="signature_parse: cannot parse: SyntaxError ..."
        c0005  worker: <dead>

    Nothing trained, and the only reason the wave was not 3-for-3
    infrastructure death is that the mock LLM emitted one unparseable reward
    -- which is ORDINARY method behaviour and happens constantly. A guard
    reading "every candidate in this wave is a worker failure" goes silent
    there, so one bad reward would disarm it for the whole wave.

    It does not, because `jobs` is built from the TRAINABLE candidates before
    anything forks and the count is taken against that -- but "it happens to
    be right" is not a property, so this asserts it. A merged result that
    reports a skip from INSIDE a living worker (`train.interaction:
    shared_population`, a wave whose arm was not selected) is deliberately a
    different case: that worker ran, so it votes, and it votes not-a-failure.

    Both halves follow from one rule -- the guard asks "did every worker
    DIE", never "did anything train". A wave that trains nothing is a RESULT
    here: CARD's entire contribution is RL runs it did not launch
    (`bird/budget.py` counts them), so a guard keyed on an empty wave would
    raise on a correct run of a published method.
    """
    from bird.components.training import WaveFailed

    runner, ctx, cands, run_one = _wave(3, dies={0, 2})
    cands[1] = cands[1].failed("signature_parse: cannot parse", kind="invalid")

    with pytest.raises(WaveFailed) as excinfo:
        runner(ctx, None, cands, run_one, lambda c, r: None)
    msg = str(excinfo.value)
    assert "2 of 2" in msg, (
        "the skipped candidate was counted in the denominator, so a wave in "
        "which every LAUNCHED worker died did not read as one: " + msg)
    assert "c0000" in msg and "candidate 0 exploded" in msg, msg
def test_a_wave_with_nothing_trainable_is_not_a_wave_failure(caplog):
    """Zero launched workers is the "nothing trainable" path, not a dead run.

    The mirror of the test above, and the two must not be conflated:
    `WaveFailed` means at least one worker launched and
    every launched worker died. A wave in which nothing was trainable launched
    nothing, broke nothing, and is already reported by the skip and budget
    counters (`policy_trainings_skipped`, `[3] train -> ... N skipped`).
    Raising here would say "the infrastructure failed" about a round where the
    generator produced no usable code -- two different findings wearing one
    exit code, which is the failure this whole change is undoing rather than
    a second copy of it.

    It also guards the arithmetic: with an empty wave `len(failures)` and
    `n_jobs` are both zero, so a predicate written as `len(failures) ==
    n_jobs` without the emptiness check first would raise on every wave that
    trained nothing on purpose.
    """
    import logging

    runner, ctx, cands, run_one = _wave(3, dies=set())
    # BOTH non-trainable populations, because they are two counters and not
    # one (`_skip_result`): a screen deciding against a candidate is CARD's
    # saving, an unparseable one is a defect. Neither launched a worker, so
    # neither may vote, and a guard that happened to key on only one of them
    # would pass a single-population test.
    cands[0] = cands[0].failed("verify: unparseable", kind="invalid")
    cands[1] = cands[1].failed("screened by test", kind="screened")
    cands[1].screened_out, cands[1].valid = True, True
    cands[2] = cands[2].failed("verify: unparseable", kind="invalid")

    with caplog.at_level(logging.ERROR, logger="bird.train"):
        results = runner(ctx, None, cands, run_one, lambda c, r: None)

    assert [r.cand_id for r in results] == [c.cand_id for c in cands]
    assert all(not r.trained and r.skip_reason for r in results), \
        [(r.cand_id, r.trained, r.skip_reason) for r in results]
    assert ctx.budget.policy_trainings == 0
    assert ctx.budget.policy_trainings_skipped == 1, "the screened one is the saving"
    assert ctx.budget.candidates_invalid == 2, "the unparseable ones are the defect"
    assert not [r for r in caplog.records
                if "candidate-parallelism worker(s) failed" in r.getMessage()], caplog.text


def test_the_all_failed_wave_leaves_a_failed_run_and_a_non_zero_exit(tmp_path):
    """The claim end to end, in a real process, because the exit code IS the finding.

    A unit test can only show the raise. What was actually broken is the chain
    after it -- nothing in `bird.py`'s iteration loop may swallow this (only
    `BudgetExceeded` is caught there), `run()` must mark the directory failed
    on the way past, and `main()` must not return 0. A `try/except Exception`
    added to the loop later would leave both tests above green and restore the
    exact defect, so this one runs the CLI and reads `$?`.

    Every worker is killed by patching `build_child_payload` BEFORE the fork,
    so each child raises inside `_child_main` and writes a `fatal` payload --
    the same shape a CUDA-OOM or a dead adapter produces, without needing one.
    """
    import subprocess
    import sys

    out = tmp_path / "runs"
    code = (
        "import runpy, sys\n"
        "from bird.components import training\n"
        "def _boom(*a, **k):\n"
        "    raise RuntimeError('deliberate worker death')\n"
        "training.build_child_payload = _boom\n"
        f"sys.argv = ['bird.py', '-c', 'eureka', '-p', 'tester', '--out', {str(out)!r},\n"
        "             '-s', 'train.candidate_parallelism=parallel',\n"
        "             '-s', 'generate.n_candidates=3']\n"
        "runpy.run_path('bird.py', run_name='__main__')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                          capture_output=True, text=True, timeout=600)

    assert proc.returncode != 0, (
        "a run whose every worker died exited 0:\n" + proc.stdout[-4000:] + proc.stderr[-4000:])
    both = proc.stdout + proc.stderr
    assert "WaveFailed" in proc.stderr, proc.stderr[-4000:]
    # The count is written against the WAVE's size rather than against
    # `generate.n_candidates`: a screened candidate is not a worker, so
    # asserting a literal 3 here would be asserting the screen's behaviour.
    assert re.search(r"\[3\] train: (\d+) of \1 candidate-parallelism worker\(s\) failed",
                     both), both[-4000:]
    assert "EVERY worker in this wave failed" in both, both[-4000:]

    (run_dir,) = [p for p in out.iterdir() if p.is_dir()]
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "failed", status


def test_every_spawned_payload_passes_the_parents_validator(tmp_path, monkeypatch):
    """A defect only an end-to-end check can catch.

    THE FAILURE SHAPE. If the spawn worker adds keys to its payload
    (`worker_start`, `worker_startup_s`, `xla_env`) that `payload_problems`
    -- a STRICT allowlist -- does not declare, it answers "unexpected key(s)"
    for every payload, `_merge_child` refuses every one, each candidate is
    charged for the env steps it really spent, and the run reports `0
    trained, 16 failed` -- which reads as sixteen bad rewards, not as a
    protocol disagreement.

    WHY UNIT TESTS MISS IT. Every worker-side test is green, because the
    payload is well-formed *in the worker*. The validator is green on its own
    fixtures. The disagreement exists only between the two, and a unit test
    is scoped to one address space.

    So this asserts the ACTUAL payloads the ACTUAL worker produced against
    the ACTUAL validator, by watching the call the fold makes -- not a
    hand-built payload, which would be a fixture agreeing with its author,
    and not a list of expected keys, which would need editing by the same
    person who forgot to edit the allowlist.
    """
    from bird.components import training as T

    seen = []
    real = T.payload_problems
    def watched(payload):
        problems = real(payload)
        seen.append((sorted(payload) if isinstance(payload, dict) else payload,
                     problems))
        return problems
    monkeypatch.setattr(T, "payload_problems", watched)
    monkeypatch.setattr(T, "_worker_is_batched", lambda env: True)

    entry = _entry()
    cfg = load("eureka", profile="tester", overrides={
        "seed": 0, "train.candidate_parallelism": "parallel",
        # SMALL ON PURPOSE. `eureka`'s tester width is 16 candidates and every
        # spawned worker pays a whole interpreter start plus a `bird` import,
        # so the default turns this into ~32 fresh interpreters and minutes of
        # CI. Three candidates over two waves is all these properties need:
        # that workers are spawned, say so, and are reused across a wave.
        "generate.n_candidates": 3, "loop.max_parallel_trainings": 2,
        "loop.n_iterations": 1})
    entry.run(cfg, out_root=str(tmp_path))

    assert seen, (
        "the validator was never called, so this test watched nothing -- "
        "either the wave did not spawn or the fold no longer validates")
    bad = [(keys, probs) for keys, probs in seen if probs]
    assert not bad, (
        f"{len(bad)} of {len(seen)} spawned payloads were REFUSED by the "
        f"parent. First: {bad[0][1]} (keys sent: {bad[0][0]})")


def test_the_allowlist_still_refuses_a_key_nobody_declared(tmp_path):
    """The other direction, and without it the test above is half a guard.

    Widening `PAYLOAD_FIELDS` until everything validates would make the one
    above pass forever. The allowlist exists because a worker and its parent
    can run different code versions, so an undeclared key means the far side
    is running code this side does not know about -- and that must still
    cost one candidate rather than be waved through.
    """
    from bird.components.training import payload_problems, PAYLOAD_FIELDS
    from bird.types import Candidate, TrainResult

    c = Candidate(cand_id="c0000", iteration=0, reward_code="")
    base = {"result": TrainResult(cand_id="c0000", candidate=c), "candidate": c,
            "budget": {}, "policy": [], "replay": [], "round_replay": [],
            "code": [], "env_states": None, "exceeded": None}
    assert payload_problems(dict(base)) == [], "the baseline payload must be clean"

    rogue = dict(base, something_new_from_a_newer_worker=1)
    problems = payload_problems(rogue)
    assert any("unexpected" in p for p in problems), problems
    assert "worker_start" in PAYLOAD_FIELDS, (
        "the spawn keys must be DECLARED, not tolerated by a loosened check")


def test_the_schedule_fields_are_all_covered_by_the_deny_list():
    """Every key a seed-schedule record writes must be scrubbed before the
    bit-identity comparison.

    If the journal called the reason field `reason` while the seed row called
    it `seed_schedule_reason`, only one of the two names would be in
    `_VOLATILE`; the spawn and sequential journals would then differ on that
    one key and the comparison would return DIFFERENT -- a SCIENCE-difference
    verdict from a pure schedule fact, in the test whose only job is to tell
    those apart.

    Reads the names from the constant the code writes them with, so this
    cannot be satisfied by editing a list here to match a list there.
    """
    from bird.components.training import SEED_SCHEDULE_FIELDS

    missing = [f for f in SEED_SCHEDULE_FIELDS if f not in _VOLATILE]
    assert not missing, (
        f"{missing} is written onto seed rows and journal rows but is not in "
        f"_VOLATILE, so the sequential-vs-parallel comparison will read a "
        f"schedule downgrade as a science difference")


def test_a_spawned_wave_says_so_in_the_journal(tmp_path, monkeypatch):
    """Evidence that the worker really was a FRESH INTERPRETER, in an artifact.

    This is the half that cannot be got from the equality test beside it.
    That one proves a spawned wave computes what sequential computes; it
    would pass just as well if the spawn branch had quietly fallen back to
    forking, because a forked wave is also equal to sequential on the numpy
    tier. So the two together are the claim: the results match AND the work
    ran where it was supposed to.

    Reads the run's own journal rather than a return value, because the
    journal is what a reader opens months later, and because a key that
    validates and travels but reaches no artifact is the declared-but-unread
    defect.

    The tester env is not batched, so the routing predicate is forced: what
    is under test is the worker START, which is env-independent. Forcing it
    here is also the only way to exercise the spawn path without the jax
    extra installed.
    """
    from bird.components import training as T

    monkeypatch.setattr(T, "_worker_is_batched", lambda env: True)
    entry = _entry()
    cfg = load("eureka", profile="tester", overrides={
        "seed": 0, "train.candidate_parallelism": "parallel",
        # SMALL ON PURPOSE. `eureka`'s tester width is 16 candidates and every
        # spawned worker pays a whole interpreter start plus a `bird` import,
        # so the default turns this into ~32 fresh interpreters and minutes of
        # CI. Three candidates over two waves is all these properties need:
        # that workers are spawned, say so, and are reused across a wave.
        "generate.n_candidates": 3, "loop.max_parallel_trainings": 2,
        "loop.n_iterations": 1})
    entry.run(cfg, out_root=str(tmp_path))

    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    journal = [json.loads(line) for line in
               (run / "journal.jsonl").read_text().splitlines() if line.strip()]
    rows = [e for e in journal if e.get("stage") == "worker_start"]

    assert rows, (
        "no `worker_start` row in the journal: either the wave did not spawn, "
        "or the payload's start fields are being dropped in the fold -- the "
        "second is invisible in every other assertion in this file")
    assert {r["start"] for r in rows} == {"spawn"}, {r["start"] for r in rows}

    # The four-part split, present and ordered. `total` is the ruled number;
    # a row whose parts are all zero means the timer was never started and
    # the scheduling call this tier owes cannot be made from it.
    for r in rows:
        parts = r["startup_s"]
        assert set(parts) >= {"import", "env_construct", "total"}, parts
        assert parts["total"] > 0, f"worker reported zero startup: {parts}"
        assert parts["total"] >= parts["import"], parts
        # REGRESSION FOR THE SECOND DEFECT: once the allowlist accepted these
        # keys they were read by NOBODY -- validated, travelled, and absent
        # from every artefact. Each one the worker sends has to arrive here,
        # or it is decoration that passes its own type check.
        assert "xla_env" in r and "jax_cache" in r, sorted(r)

    # And the parent's own before/after count around the wave, which is the
    # half that shows the compiles did not happen in the parent.
    waves = [e for e in journal if e.get("stage") == "worker_wave"]
    assert waves, "no `worker_wave` row: the parent recorded no cache reading"
    for w in waves:
        assert set(w) >= {"parent_jax_cache_before", "parent_jax_cache_after",
                          "parent_jax_cache_grew"}, sorted(w)
        # None, not 0, when no persistent cache is configured -- which is the
        # case in the test environment. A 0 here would mean a run that measured nothing
        # was recorded as a run that proved the property.
        assert w["parent_jax_cache_before"] is None or \
            isinstance(w["parent_jax_cache_before"], int), w


def test_the_rebuilt_run_one_is_the_parents_run_one(tmp_path):
    """The pin `_rebuild_run_one`'s docstring promises.

    A spawned worker cannot be sent a closure, so it REBUILDS `run_one` from
    the names in `worker_inputs`. That is two constructions of one thing, in
    two files, and the failure mode if they drift is the worst shape this
    repo has: both sides run, both produce a `TrainResult`, and the numbers
    differ. No exception, no red test -- just a `parallel` run that is no
    longer the `sequential` run it is required by contract to equal.

    Compares the RESULT, not the resolved parts. Asserting that both sides
    picked the same backend object would pass while the two disagreed about
    `n_seeds`, which is the input most likely to drift because it is the one
    that comes from a per-iteration allocation rather than from the config.
    """
    from bird.components.training import (worker_inputs, _rebuild_run_one,
                                          _hydrate_ctx)

    def fresh():
        cfg = load("eureka", profile="tester", overrides={"seed": 0})
        ctx = Context(cfg=cfg, budget=Budget.from_config(cfg),
                      rng=random.Random(cfg["seed"]))
        ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
        return ctx, cfg

    cand = Candidate(cand_id="c0000", iteration=0,
                     reward_code="def compute_reward(state, action):\n    return 0.5\n")

    # -- the PARENT's construction, the same five lines bird.py has ----------
    pctx, cfg = fresh()
    backend = registry.get("train_backend", cfg["train.backend"])
    hpsearch = registry.get("hyperparameter_search", cfg["train.hyperparameter_search"])
    plan = {"c0000": 2}

    def parent_run_one(c, **kw):
        return hpsearch(pctx, None, c, backend,
                        n_seeds=plan.get(c.cand_id, cfg["train.seeds_per_candidate"]), **kw)

    # -- the WORKER's, through the real payload, in a fresh ctx -------------
    wctx_src, _ = fresh()
    given = worker_inputs(wctx_src, 0, cand, str(tmp_path), state=None,
                          backend_name=cfg["train.backend"],
                          hpsearch_name=cfg["train.hyperparameter_search"],
                          seed_plan=plan,
                          default_n_seeds=cfg["train.seeds_per_candidate"])
    import pickle
    given = pickle.loads(pickle.dumps(given))     # it really does travel
    worker_run_one = _rebuild_run_one(_hydrate_ctx(given), given)

    a = parent_run_one(cand)
    b = worker_run_one(cand)

    assert a.trained == b.trained
    assert a.error == b.error
    assert a.env_steps_used == b.env_steps_used, (
        f"rebuilt run_one spent {b.env_steps_used} env steps where the "
        f"parent's spent {a.env_steps_used}")
    assert a.component_traces == b.component_traces
    assert a.gt_reward_curve == b.gt_reward_curve, (
        "the learning curves differ, so the two constructions are not "
        "running the same training")
    # THE SEED COUNT, read off `seed_metrics` rather than off the closure:
    # the allocation is per-iteration data, not config, so it is the input
    # most able to drift without either side looking wrong. Asserted
    # non-empty first -- 0 == 0 would pass against a plan neither side read.
    assert len(a.seed_metrics) == 2, (
        f"the parent ran {len(a.seed_metrics)} seed(s), not the 2 the plan "
        f"asked for; the comparison below would be vacuous")
    assert len(a.seed_metrics) == len(b.seed_metrics), (
        "the two constructions disagree about n_seeds")


def test_worker_inputs_does_not_advance_the_parent_rng():
    """The mechanism behind the `worker_inputs`/`rng` exemption above.

    `worker_inputs` reads `ctx.rng` -- the one name the rule in that test
    exists to police -- and is allowed to only because `getstate()` observes
    the generator without consuming from it. If someone ever changes that
    read to a DRAW (`rng.random()`, `rng.getrandbits()`, a `Random(...)`
    re-seed), the parent's stream shifts by one per candidate per iteration
    and every later candidate in the run gets different numbers. The
    symptom would be a run that no longer reproduces from `seed`, with no
    error anywhere -- so the exemption is only as good as this assertion.

    Asserts the STREAM, not just the state tuple: comparing `getstate()` to
    itself would pass even if the read had been replaced by something that
    consumed and then restored, and it is the next value the run actually
    depends on.
    """
    from bird.components.training import worker_inputs

    cfg = load("eureka", profile="tester", overrides={"seed": 0})
    ctx = Context(cfg=cfg, budget=Budget.from_config(cfg), rng=random.Random(0))
    for _ in range(3):
        ctx.rng.random()            # mid-stream, as a real run would be

    before_state = ctx.rng.getstate()
    cand = Candidate(cand_id="c0000", iteration=0, reward_code="")
    worker_inputs(ctx, 0, cand, "/tmp", state=None,
                  backend_name=cfg["train.backend"],
                  hpsearch_name=cfg["train.hyperparameter_search"])
    assert ctx.rng.getstate() == before_state, (
        "worker_inputs moved the parent's generator state")

    # The stream itself: what the next candidate would draw must be what it
    # would have drawn had worker_inputs never been called.
    expected = random.Random(0)
    for _ in range(3):
        expected.random()
    assert ctx.rng.random() == expected.random(), (
        "worker_inputs consumed from the parent's rng; every candidate after "
        "this one in the run now draws different numbers")


def test_a_child_raised_reward_source_event_reaches_the_parent():
    """Mechanism behind the `_training_reward`/`event` exemption, candidate fork.

    `_training_reward` runs INSIDE the candidate worker under
    `train.candidate_parallelism: parallel`, and `Context.event` drops
    silently when neither `rundir` nor `event_buffer` is set -- there is no
    `else` clause to notice. The exemption claims the buffer makes that safe.
    This asserts the whole round trip for THIS row rather than for events in
    general: the child raises it, the payload carries it, the parent journals
    it. A shared mechanism is still a separate claim about each row that
    leans on it.

    Deliberately not a test that `ctx.event` exists or that buffering works
    -- `test_forked_candidate_events.py` owns those. This one
    fails if the reward_source row in particular stops making the trip,
    which is what the exemption above promises.
    """
    from bird.components import training
    from bird.config import Config

    class _Env:
        has_reference_reward = True

        @staticmethod
        def reference_reward(state, action):
            return 1.0

    cand = Candidate(cand_id="c0000", iteration=0, reward_code="")
    child = Context(cfg=Config({"name": "eureka", "train": {"reward_source": "reference"}}),
                    budget=Budget(), env=_Env(), rundir=None, event_buffer=[])

    training._training_reward(child, None, cand)
    rows = [f for stage, f in child.event_buffer if f.get("event") == "reward_source"]
    assert rows, (
        "the child raised no reward_source row -- with rundir None and the "
        "buffer armed this is the call the exemption is about")
    assert rows[0]["source"] == "reference"

    payload = {
        "result": training.TrainResult(cand_id="c0000", candidate=cand),
        "candidate": cand, "budget": {}, "policy": [], "replay": [],
        "round_replay": [], "code": [], "env_states": None, "exceeded": "",
        "events": list(child.event_buffer),
    }
    assert training.payload_problems(payload) == []

    seen = []

    class _Parent:
        cfg = Config({})
        budget = Budget()

        def event(self, stage, **fields):
            seen.append((stage, fields))

    training._merge_child(_Parent(), cand, payload)
    assert any(f.get("event") == "reward_source" and f.get("source") == "reference"
               for _s, f in seen), (
        "the row did not survive the fold -- buffered but never replayed is "
        "the same artifact as never emitted")


def test_the_reward_source_event_is_raised_above_every_seed_fork():
    """The OTHER half of that exemption, and the half with no mechanism.

    `seed_child` never calls `build_child_payload`, so the seed payload has no
    `events` channel and a row raised inside a seed child is dropped into
    nothing -- the backend's fold over `_fork_seeds` has no such key to replay it. The
    `_training_reward` exemption is therefore safe under the seed fork only
    because the call sits ABOVE `_fork_seeds` in every backend that forks.

    That is an ORDERING argument, which is exactly what this table's first
    entry exists to reject: "provably runs only in the parent" is exactly the
    kind of claim that can be false and cost a real journal. So it is pinned
    structurally rather than asserted in a comment.

    THREE WAYS THE OBVIOUS VERSION OF THIS TEST IS WRONG ON CORRECT CODE:

      1. A per-MODULE `max(reward) < min(fork)` fails on `training.py`, which
         holds two independent backends: `_sb3_run` emits the reward after
         `_run_backend` forks, although each backend emits before its OWN
         fork. The comparison is only meaningful per enclosing function, so
         the calls are grouped by one.
      2. Walking the `("training", "search")` tuple the test above uses pins
         two of the three pairs and silently misses the third, which lives in
         `fasttd3.py` -- and a fixed `("training", "fasttd3")` would in turn
         miss `simba_v2`. The module list is therefore discovered.
      3. `fasttd3.py` calls it as `_training._training_reward(...)` -- an
         `ast.Attribute`, not an `ast.Name` -- so a Name-only matcher finds
         zero reward calls there and trap 2 passes quietly.

    Hence: both call shapes, both modules, grouped by enclosing function, and
    a pair COUNT floor so that a rename cannot make this vacuous by finding
    nothing. The functions are discovered rather than named, because a
    hand-written name (`_fasttd3_run` for `fasttd3_backend`) is easy to get
    wrong.
    """
    # THE MODULES ARE DISCOVERED, NOT LISTED, and that is the fourth trap.
    # `simba_v2` routes through `_training_reward` the way `fasttd3` does, so
    # a fixed `("training", "fasttd3")` tuple would not cover a new site and
    # would say nothing -- a guard scoped to yesterday's file list. Walk
    # everything that could hold a backend instead.
    sources = sorted((REPO / "bird" / "components").glob("*.py")) + [REPO / "bird.py"]
    pairs = {}
    for src in sources:
        rel = src.relative_to(REPO).as_posix()
        tree = ast.parse(src.read_text())
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef):
                continue
            reward, forks = [], []
            for n in ast.walk(fn):
                if not isinstance(n, ast.Call):
                    continue
                f = n.func
                got = (f.id if isinstance(f, ast.Name)
                       else f.attr if isinstance(f, ast.Attribute) else None)
                if got == "_training_reward":
                    reward.append(n.lineno)
                elif got == "_fork_seeds":
                    forks.append(n.lineno)
            if reward and forks:
                pairs[f"{rel}::{fn.name}"] = (reward, forks)

    # A FLOOR, NOT AN EQUALITY, and the asymmetry is deliberate. The real
    # guard is the per-function assertion below, which covers however many
    # sites exist and so covers a new backend without anyone editing this
    # file. The count is here only to stop the matcher going vacuous: rename
    # either callee and every assertion below passes by finding nothing,
    # which is the failure mode a `== 0` cannot announce.
    assert len(pairs) >= 3, (
        "fewer than three backends both resolve the training reward and fork "
        f"seeds; found {sorted(pairs)}. Either a backend stopped forking (fine, "
        "but check the seed half of the _training_reward exemption still needs "
        "to exist) or -- far more likely -- `_training_reward` or `_fork_seeds` "
        "was renamed and this test is now matching nothing.")

    for where, (reward, forks) in sorted(pairs.items()):
        assert max(reward) < min(forks), (
            f"{where} resolves the training reward at {reward} but forks seeds "
            f"at {forks}: a reward_source event raised at or below the fork is "
            "raised in a SEED CHILD, whose payload has no `events` channel "
            "(`seed_child` never calls `build_child_payload`), so it is dropped "
            "silently. Either keep the call above the fork, or give the seed "
            "payload the buffering the candidate payload has and replace the "
            "seed half of this exemption with that mechanism.")


def test_the_train_stage_touches_only_cfg_env_budget():
    """The determinism argument, turned into a test.

    A forked worker holds a PRIVATE copy of everything, so anything it reads
    off `ctx` that the parent later mutates -- `ctx.rng`, `ctx.counters`,
    `ctx.next_id` -- desynchronises the run silently, and anything it WRITES
    (`ctx.rundir`, `ctx.tracker`) is lost or, worse, races. Today neither
    module touches any of them; this keeps it so, because the day one does the
    symptom is a run that no longer reproduces from `seed` and nothing else.
    """
    allowed = {"cfg", "env", "budget"}
    # ONE ATTRIBUTE, IN ONE FUNCTION, and both halves of that are the point.
    #
    # `_merge_child` reads `ctx.event` to replay a worker's buffered events.
    # The claim that makes this safe does not depend on which side runs it:
    # `ctx.event` is safe in this body FROM EITHER SIDE, because a child's
    # event buffer is armed before the fold, so its events are BUFFERED AND
    # SHIPPED rather than written into nothing. That mechanism is why this one
    # read is not the hazard the rule describes.
    #
    # Everything else stays policed inside `_merge_child` on both paths --
    # `rng`, `next_id`, `counters`, `tracker` are exactly the names whose
    # private-copy behaviour has no such mechanism, and a read of any of
    # them there would still be the silent desynchronisation this test
    # exists for. Exempting the whole body would have given all four away
    # for free, which is the same shape as exempting an attribute name
    # globally -- a hole that can hide a `ctx.tracker` read.
    #
    # THE SECOND ENTRY, `worker_inputs`/`rng`, and it needs its own mechanism
    # because "it runs in the parent" is the argument this table already
    # rejects once above. The mechanism is that the read is
    # `ctx.rng.getstate()`, which DOES NOT ADVANCE the generator, and that
    # the value is sent to a SPAWNED worker so that it starts from the same
    # stream position a FORKED worker inherits for free. So the read does
    # not desynchronise the parent (nothing is consumed) and its purpose is
    # to remove a fork/spawn divergence rather than to let a worker draw.
    # That claim is not left as prose: `test_worker_inputs_does_not_advance_
    # the_parent_rng` below asserts the non-advancement directly, which is
    # the half a reader cannot check by eye.
    #
    # THE THIRD ENTRY, `_note_seed_schedule`/`event`, rests on the SAME
    # mechanism as the first and not on a new argument. On the parallel
    # path the backends that call it run INSIDE the candidate worker, so
    # this is a child write -- and it is safe for the reason `_merge_child`
    # is: the child's event buffer is armed before the work
    # (`build_child_payload`), so the row is buffered and shipped home
    # rather than written into nothing. If that arming ever goes away, both
    # exemptions fail together, which is the right coupling.
    # THE FIFTH ENTRY, `_training_reward`/`event` (the reward_source row), and
    # it needs TWO mechanisms because it sits under two different forks.
    #
    # Under the CANDIDATE fork it rests on exactly the mechanism the first and
    # third entries rest on, and deliberately not on a new one: the child's
    # event buffer is armed before the work (`build_child_payload`
    # `ctx.event_buffer = []`), the row is shipped in the payload's `events`
    # and `_merge_child` replays it. So if that arming ever goes away, FOUR
    # exemptions fail together rather than three -- the coupling the third
    # entry names, extended. `test_a_child_raised_reward_source_event_reaches_
    # the_parent` below asserts that round trip for this row specifically,
    # because a mechanism shared with another entry is still a claim about
    # this one.
    #
    # Under the SEED fork there IS no such mechanism, and that half is held by
    # position rather than by machinery: `seed_child` never calls
    # `build_child_payload`, so the seed payload has no `events` channel at all
    # and a row raised inside a seed child is dropped silently. The call is
    # safe only because it runs ABOVE `_fork_seeds` in all three backends. That
    # is an ORDERING argument -- the class this table's first entry exists to
    # reject -- so it is not left as prose either:
    # `test_the_reward_source_event_is_raised_above_every_seed_fork` below
    # pins the position structurally, and fails the day a refactor moves the
    # call below a fork or adds a fourth backend that does.
    exempt_in = {"_merge_child": {"event"}, "worker_inputs": {"rng"},
                 "_note_seed_schedule": {"event"},
                 "_note_worker_wave": {"event"},
                 "_training_reward": {"event"}}
    for module in ("training", "search"):
        src = (REPO / "bird" / "components" / f"{module}.py").read_text()
        tree = ast.parse(src)
        # attribute-node id -> the names exempt where it sits
        exempt_here = {}
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name in exempt_in:
                for n in ast.walk(fn):
                    exempt_here[id(n)] = exempt_in[fn.name]
        used = {node.attr for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name) and node.value.id == "ctx"
                and isinstance(node.ctx, ast.Load)
                and node.attr not in exempt_here.get(id(node), ())}
        assert used <= allowed, (
            f"bird/components/{module}.py reads ctx.{sorted(used - allowed)} -- "
            "a forked training worker sees a private copy of that")

    # The exemption table is pinned: each entry is a claim that THAT
    # attribute is safe in THAT body, and every such claim needs a mechanism
    # behind it rather than an argument about which process runs the code.
    assert exempt_in == {"_merge_child": {"event"},
                         "worker_inputs": {"rng"},
                         "_note_seed_schedule": {"event"},
                         "_note_worker_wave": {"event"},
                         "_training_reward": {"event"}}, (
        "the exemption table changed. `ctx.event` in `_merge_child` is "
        "exempt because the child's event buffer is armed before the fold, "
        "so events are buffered and shipped rather than dropped -- not "
        "because of which process runs the function. A new "
        "entry needs its own mechanism, named here")


@pytest.mark.parametrize("slurm,key,n_jobs,expect", [
    (None, "auto", 1, 1),      # CARD: K=1 is sequential by construction
    (None, 4, 8, 4),
    (None, 4, 2, 2),           # never more workers than jobs
    (None, 1, 8, 1),
    ("2", "auto", 8, 2),       # --cpus-per-task=2: 2 workers x 1 thread
    ("16", "auto", 8, 8),
    ("224", "auto", 8, 8),     # a large node; still capped by the job count
    (None, "auto", 0, 1),      # nothing to run
])
def test_resolve_workers(slurm, key, n_jobs, expect, monkeypatch):
    """`auto` reads the ALLOCATION, not the machine.

    `os.cpu_count()` would report 224 on a large node holding a 2-CPU
    allocation; `SLURM_CPUS_PER_TASK` and, failing that, the affinity mask are
    the two things that know what the job actually got.
    """
    from bird.components.training import resolve_workers

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    if slurm is not None:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", slurm)
    cfg = load("eureka", profile="tester",
               overrides={"loop.max_parallel_trainings": key})
    got = resolve_workers(cfg, n_jobs)
    if slurm is None and key == "auto" and n_jobs > 1:
        assert got == min(len(os.sched_getaffinity(0)), n_jobs)
    else:
        assert got == expect


@pytest.mark.parametrize("bad", ["ato", "AUTO ", "", 0, -3])
def test_a_bad_max_parallel_trainings_is_caught_by_validate(bad):
    """`loop.max_parallel_trainings` is `int | str` in the schema, so only
    `_check_coherence` can say that the one legal string is `auto`. Before
    this check, `ato` would validate cleanly and then die in `int()` deep inside stage 3
    -- after generation had already spent LLM calls."""
    from bird.config import ConfigError

    with pytest.raises(ConfigError, match="max_parallel_trainings"):
        load("eureka", profile="tester",
             overrides={"loop.max_parallel_trainings": bad})


# --------------------------------------------------------------------------
# seed forking -- the same schedule key, one level down
# --------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [2, 4])
def test_final_retrain_seed_fork_is_bit_identical(workers, tmp_path):
    """`final_retrain.n_seeds` trainings fork under `parallel`, and the fork
    must not move a number -- the same bar the candidate fork is held to.

    Why this exists: the retrain runs in the PARENT, after the search, so the
    candidate fork never touched it, and serial retrains can be a large share
    of a run's wall-clock. Whole run dirs are compared, which also proves the retrain
    actually ran both times -- `final_retrain.json` is part of the fingerprint,
    and the explicit asserts below keep the test from passing vacuously if the
    phase is ever skipped."""
    common = {"final_retrain.enabled": True, "final_retrain.n_seeds": 4,
              "generate.n_candidates": 2}
    seq = _run("eureka", tmp_path / "seq", **common,
               **{"train.candidate_parallelism": "sequential"})
    par = _run("eureka", tmp_path / f"par{workers}", **common,
               **{"train.candidate_parallelism": "parallel",
                  "loop.max_parallel_trainings": workers})
    assert "phases/final_retrain.json" in seq, \
        "the retrain never ran; the test is vacuous"
    fr = json.loads(seq["phases/final_retrain.json"])
    assert fr["n_seeds"] == 4 and fr["retrained_fitness"] is not None
    assert set(seq) == set(par), (
        f"different artifact file sets\n  only sequential: {sorted(set(seq) - set(par))}\n"
        f"  only parallel:   {sorted(set(par) - set(seq))}")
    differing = [k for k in sorted(seq) if seq[k] != par[k]]
    assert not differing, f"the seed fork changed {differing}"


def test_seed_fork_gates(monkeypatch):
    """Seed forking is refused with one seed, under `sequential`, and inside a
    candidate worker -- k candidate workers each forking s seed workers would
    oversubscribe the allocation k-fold with no process able to see it."""
    from bird.components import training

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    # An explicit worker cap and LITERAL expectations -- asserting
    # `min(5, affinity)` would restate `resolve_workers`'s arithmetic and agree
    # with itself by construction, never exercising the cap on a narrow runner.
    par_cfg = load("eureka", profile="tester",
                   overrides={"train.candidate_parallelism": "parallel",
                              "loop.max_parallel_trainings": 3})
    # RETURNS (workers, reason). The reason is asserted too, not discarded:
    # it is the only thing that reaches a reader of the run's artefact
    # saying why a retrain took N times as long, so "did it come back empty
    # when nothing was downgraded" is half the contract.
    assert training._seed_fork_workers(par_cfg, 1) == (1, "")
    assert training._seed_fork_workers(par_cfg, 5) == (3, "")  # the knob caps
    assert training._seed_fork_workers(par_cfg, 2) == (2, "")  # never > seeds
    monkeypatch.setattr(training, "_IN_TRAIN_WORKER", True)
    assert training._seed_fork_workers(par_cfg, 5) == (1, "")
    monkeypatch.setattr(training, "_IN_TRAIN_WORKER", False)
    seq_cfg = load("eureka", profile="tester")
    assert training._seed_fork_workers(seq_cfg, 5) == (1, "")

    # THE BATCHED TIER: downgraded to 1, and it must SAY SO. A bare 1 here
    # is indistinguishable from "one seed was asked for".
    jax_cfg = load("examples/jax_reward", profile="tester",
                   overrides={"train.candidate_parallelism": "parallel",
                              "loop.max_parallel_trainings": 3})
    workers, reason = training._seed_fork_workers(jax_cfg, 5)
    assert workers == 1, workers
    assert "batched" in reason and "sequential" in reason, reason
    assert "bit-identical" in reason, (
        "the reason must say results are unchanged, or a reader takes a "
        "schedule downgrade for a science change")


def test_final_retrain_seed_fork_matches_sequential_on_sb3(tmp_path, monkeypatch):
    """The sb3 path of the same claim, at ONE torch thread on both schedules --
    thread count is the one knob allowed to move sb3 numbers (see (d) in
    `parallelism_parallel`), the children pin one thread, and the parent-side
    rollouts run under `_single_thread_torch`. Needs torch, which the default
    CI install lacks.

    `INVALID_FRACTION` is pinned to 0 because the run has ONE candidate: the mock
    draws its invalid archetypes by a hash of the prompt, so any change to the
    prompt text anywhere in the tree re-rolls the draw, and a `nan_values` draw
    leaves nothing valid to retrain -- `final_retrain` then (correctly) writes no
    `phases/final_retrain.json` and this test reads as a parallelism failure. The
    claim under test is the seed fork, not the mock's validity mix, which
    `tests/test_pipeline.py` covers."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    torch.set_num_threads(1)
    from bird.llm import mock as mock_llm
    monkeypatch.setattr(mock_llm, "INVALID_FRACTION", 0.0)
    entry = _entry()
    outs = {}
    for tag, sched in (("seq", "sequential"), ("par", "parallel")):
        cfg = load("eureka", profile="dev", overrides={
            "seed": 0,
            "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "output.tracker": "none",
            "loop.n_iterations": 1, "generate.n_candidates": 1,
            "train.env_steps": 600, "evaluate.rollouts_per_candidate": 1,
            "train.candidate_parallelism": sched,
            "final_retrain.enabled": True, "final_retrain.n_seeds": 2,
            "final_retrain.env_steps": 600,
            "post": ["final_retrain"],
        })
        entry.run(cfg, out_root=str(tmp_path / tag))
        (run,) = [p for p in (tmp_path / tag).iterdir() if p.is_dir()]
        outs[tag] = _scrub(json.loads(
            (run / "phases" / "final_retrain.json").read_text()), str(run))
    assert outs["seq"] == outs["par"]


def test_a_lost_policy_blob_fails_the_result_rather_than_mismeasuring(monkeypatch):
    """The forked schedule's one state `sequential` cannot reach: sequential
    rolls the LIVE trained model, the fork a deserialised copy. A child whose
    `model.save()` failed must not send a successful payload, or the parent
    quietly rolls FRESH weights into the fitness -- a corrupted measurement
    wearing a healthy one's clothes, with a log.warning as the only trace.
    The result must instead be marked failed, same as every other broken
    measurement (`trained=False` + `error`)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    torch.set_num_threads(1)
    from bird.budget import Budget
    from bird.components import training
    from bird.context import Context
    from bird.types import Candidate

    monkeypatch.setattr(training, "_sb3_policy_blob", lambda m: None)
    cfg = load("eureka", profile="dev", overrides={
        "seed": 0,
        "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
        "output.tracker": "none",
        "train.env_steps": 600, "evaluate.rollouts_per_candidate": 1,
        "train.candidate_parallelism": "parallel",
        "loop.max_parallel_trainings": 2,
    })
    registry.load_all()
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    cand = Candidate(cand_id="c0000", iteration=0,
                     reward_code="def reward(s, a, s2):\n    return 0.0\n")
    result = training.sb3_backend(ctx, None, cand, n_seeds=2)
    assert result.trained is False
    assert "no loadable policy" in result.error


def test_sb3_is_reproducible_from_seed(tmp_path):
    """Same config seed, same sb3 numbers -- the precondition for every
    parallel-vs-sequential comparison on a real backend.

    SB3's `DummyVecEnv.reset` hands a real seed exactly once before calling
    `_reset_seeds()`, so a `_Gym.reset` that did
    `env.reset(np.random.default_rng(seed))` would see `seed=None` on every
    auto-reset after the first and draw OS entropy from `default_rng(None)`.
    Measured on eureka/pendulum: two runs at the same config seed returned
    winner fitness 0.0746 and 0.0945. Every real-learner run would be
    affected, independently of `parallel`, and no parallel-vs-sequential
    comparison on a real backend could mean anything.
    """
    pytest.importorskip("stable_baselines3")
    entry = _entry()
    seeds = []
    for i in range(2):
        cfg = load("eureka", profile="dev", overrides={
            "seed": 0,
            "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "output.tracker": "none",
            "loop.n_iterations": 1, "generate.n_candidates": 1,
            "train.env_steps": 600, "evaluate.rollouts_per_candidate": 1,
            "post": [],
        })
        entry.run(cfg, out_root=str(tmp_path / f"r{i}"))
        (run,) = [p for p in (tmp_path / f"r{i}").iterdir() if p.is_dir()]
        payloads = sorted(run.glob("candidates/*/train_result.json"))
        seeds.append([json.loads(p.read_text())["seed_metrics"] for p in payloads])

    def _numbers(blob):
        return [{k: v for k, v in row.items() if k not in ("wallclock_s", "checkpoints")}
                for cand in blob for row in cand]

    assert _numbers(seeds[0]) == _numbers(seeds[1]), \
        "two sb3 runs at the same seed disagreed"


@pytest.mark.parametrize("task", ["mt10_drawer-open-v3"])
def test_export_import_states_round_trips_across_a_fork(task, tmp_path, gl):
    """MetaWorld's obs -> simulator-snapshot cache is per-process.

    Without `export_states`/`import_states` a worker's trajectories come back
    un-renderable and un-scoreable: `task_metric` keeps working (it is a pure
    function of the observation rows) so fitness LOOKS fine, while
    `reference_reward`, `gt_reward_curve`, the rollout videos and therefore
    rda's `vlm_score` and gt's `preference_bt` all die. That is the shape of
    failure `budget.blind_comparisons` exists to catch, so it must not be
    reachable by flipping a scheduling key.
    """
    pytest.importorskip("metaworld")
    import pickle

    import numpy as np

    from bird.envs.metaworld import UnknownStateError

    registry.load_all()
    env = registry.get("env", task)({})
    spool = tmp_path / "x.pkl"

    pid = os.fork()
    if pid == 0:  # -- child: produce a trajectory the parent has never seen --
        code = 0
        try:
            s = env.reset(np.random.default_rng(3))
            rows = [s]
            for _ in range(12):
                s, _done, _info = env.step(s, np.zeros(np.shape(env.action_low)))
                rows.append(s)
            arr = np.asarray(rows)
            spool.write_bytes(pickle.dumps(
                (arr, env.export_states([arr]),
                 [float(env.reference_reward(r)) for r in arr])))
        except BaseException:
            code = 1
        finally:
            os._exit(code)
    _pid, status = os.waitpid(pid, 0)
    assert status == 0, "the child could not produce a trajectory"

    arr, blob, child_ref = pickle.loads(spool.read_bytes())
    with pytest.raises(UnknownStateError):
        env.reference_reward(arr[5])

    env.import_states(blob)
    assert [float(env.reference_reward(r)) for r in arr] == child_ref, \
        "the reference reward must be bit-identical to the worker's"

    # Rendering needs a headless GL context, which `MUJOCO_GL=disable` (how the
    # suite runs on a machine without one) deliberately does not provide. Asked the
    # same way the `gl` fixture in tests/conftest.py declares it: a GL failure is not
    # this test's subject, but on a machine that CAN render, a frame that comes
    # back malformed must not pass silently.
    frame = env.render(arr[5])
    assert frame.ndim == 3 and frame.dtype.name == "uint8"


def test_the_clock_shape_predicate_separates_durations_from_counts():
    """The three cases that decide whether the suffix rule is safe.

    `_CLOCK_SHAPED` is matched as a SUBSTRING, so the tempting one-character
    fix -- adding `_s` to it -- classifies `env_steps`, `n_seeds`,
    `search_seeds` and every other count as a clock -- THIRTEEN of the
    fourteen names asserted below contain `_s` somewhere and would flip
    (`num_envs` is the one that would not), counted rather than estimated.
    The suffix list is checked with `endswith` instead, which is why it is a
    separate mechanism rather than another entry.

    WHAT THE FOURTEEN COUNTS DO NOT TEST, said plainly so a later reader does
    not over-credit them: not one of them ends in `_s` -- they end in a bare
    `s` -- so they do not exercise the `endswith` boundary at all. What they
    guard is regression to SUBSTRING matching, the tempting mutation, which
    thirteen of them catch. The `_s`-ending non-clock class has exactly one
    member in this tree, `min_compile_s`, and it is the exemption below; if a
    second one ever appears, it belongs in that list and in this loop's
    sibling.

    THE THIRD CASE IS THE ONE THAT MATTERS. `min_compile_s` ends in the
    suffix and is not a clock: it is `jax_persistent_cache_min_compile_time_secs`,
    a config-derived constant recorded as provenance, and declaring it
    volatile would make two runs with different cache thresholds compare
    equal -- a real difference hidden to silence a guard. Asserting it is
    exempt-NOT-clock is what stops the exemption being deleted and the
    resulting failure papered over with a `_VOLATILE` entry, which from the
    outside looks exactly the same.

    WHAT THIS TEST COVERS THAT THE RUN-SCANNING ONE CANNOT:
    `test_a_timing_field_cannot_enter_a_compared_artifact` reads a TESTER
    run, which never produces a fasttd3 seed row, so it never sees
    `jit_compile_s`, `reward_sync_s` or `min_compile_s` at all -- a guard
    that cannot see the fields it exists to protect. This half is a pure
    function and sees them.
    """
    # durations: caught
    for name in ("wallclock_s", "eval_wall_s", "jit_compile_s", "reward_sync_s",
                 "elapsed_s", "gpu_seconds", "worker_startup_s"):
        assert _is_clock_shaped(name), f"{name} is a duration and must be caught"

    # counts and strings: NOT caught, and every one of these is a real key
    # from a tester run that the substring version of this rule would have
    # misclassified
    for name in ("env_steps", "env_steps_cap", "env_steps_per_seed",
                 "config_train_env_steps", "n_seeds", "n_subtasks",
                 "search_seeds", "init_source", "init_similarity",
                 "per_seed_fitness", "post_selection_gap",
                 "seed_schedule_reason", "train_steps", "num_envs"):
        assert not _is_clock_shaped(name), (
            f"{name} is a count or a string, not a clock; classifying it "
            f"would force it into _VOLATILE and stop the comparison checking it")

    # the exemption, and the reason it is an exemption rather than an absence
    assert "min_compile_s".endswith(_CLOCK_SUFFIXES), (
        "this test is vacuous unless min_compile_s really ends in the suffix")
    assert not _is_clock_shaped("min_compile_s"), (
        "min_compile_s is a config-derived provenance constant, not a clock")
    assert "min_compile_s" in _NOT_A_CLOCK, (
        "it must be exempt BY NAME: if the exemption is gone and the name is "
        "merely absent from the suffix rule, the next config constant ending "
        "in _s is silently volatile")
    assert "min_compile_s" not in _VOLATILE, (
        "the paper-over: declaring it volatile would silence the guard and "
        "make two differently-configured runs compare equal")


def test_a_timing_field_cannot_enter_a_compared_artifact(tmp_path):
    """A clock in a compared artifact breaks bit-identity. Catch it BY NAME.

    A clock such as `eval_wall_s` on every checkpoint point, not declared
    volatile, turns every case of `test_parallel_is_bit_identical_to_sequential`
    red -- with a message naming `candidates/.../train_result.json`, three
    files away from the `perf_counter()` that caused it. The diagnosis costs
    more than the bug.

    THIS TEST IS THE CHEAP HALF: it reads the artifacts of ONE ordinary
    sequential run and fails on any key whose NAME looks like a clock and
    which `_VOLATILE` does not declare. One run, no parallel comparison, and
    the failure says exactly which field and which file.

    IT SCANS A REAL RUN, not the source. A grep for `perf_counter` would miss
    a duration computed from two timestamps, a field renamed on the way into
    the payload, or one added by a component nobody thought to grep -- and
    the thing that breaks the comparison is the KEY IN THE ARTIFACT, which is
    what this reads.

    Adding a genuine clock is still allowed: declare it in `_VOLATILE`, which
    is the one edit that both silences this and keeps bit-identity honest.
    """
    fp = _run("eureka", tmp_path / "one", **{"train.candidate_parallelism": "sequential"})
    assert fp, "the run produced no artifacts, so this test checked nothing"

    offenders = {}
    for rel, blob in fp.items():
        if not rel.endswith((".json", ".jsonl")):
            continue
        for line in blob.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k in _clock_shaped_keys(payload):
                if k not in _VOLATILE:
                    offenders.setdefault(k, set()).add(rel)

    assert not offenders, (
        "clock-shaped fields reach a COMPARED artifact without being declared in "
        "`_VOLATILE`, so `test_parallel_is_bit_identical_to_sequential` will go red "
        "on every config:\n" + "\n".join(
            f"  {k}  in {sorted(v)}" for k, v in sorted(offenders.items())) +
        "\nIf it is genuinely a clock, add it to `_VOLATILE`; if it is a result, it "
        "must be deterministic across schedules.")
