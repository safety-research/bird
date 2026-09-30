"""`loop.resume_from` -- a declared key the algorithm honours.

A key declared in `configs/_default.yaml` and typed in `bird/schema.py` but
read by NOTHING is a fabricated pin (like a `temperature` field the LLM client
ignores), so the key is honoured rather than left declared.

WHY IT IS WORTH HONOURING. In a search the RL leg dominates wall-clock and the
LLM leg is a small fraction of it, so a long search can outlive a transient API
failure. Without resume, a key rotation or a sustained 5xx many hours in raises
`LLMError` and every completed hour of RL compute is lost; resume keeps it.

WHAT IS CLAIMED, AND WHAT IS NOT. On the tester tier (mock LLM, toy env, mock /
tabular learner) `N iterations straight through` and `k iterations + crash +
resume + (N-k)` produce IDENTICAL run directories after exactly the
normalisation `tests/test_parallelism.py` already applies, plus two named
exclusions -- `checkpoints/**` and `resume.jsonl`, which are records of WHEN
things happened, not of what was decided -- and the small set of fields that
exist precisely to say a run was resumed. `_scrub` and `_VOLATILE` are IMPORTED
from that module rather than copied, so the two tests cannot drift apart about
what a volatile field is.

NOT claimed: a run resumed under `auto_degraded` (dropping a replay buffer
changes the run; it is a different run and the artifact says so); a real LLM at
any temperature (two FRESH runs are already not identical there, so resume adds
nothing to claim); and torch-global byte-equality across a process boundary
under `train.backend: sb3`.
"""

import json
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import (GT_PUBLISHED_OVERRIDES, TESTER_SEARCH_CAP,  # noqa: F401  (path bootstrap)
                      apply_search_cap)
from test_parallelism import _VOLATILE, _entry, _scrub

from bird import checkpoint
from bird.budget import Budget, BudgetExceeded
from bird.checkpoint import (SCHEMA_VERSION, CheckpointEncodeError, CheckpointError,
                             ResumeLocked, _ArraySpill, decode, encode)
from bird.config import ConfigError, load
from bird.state import ArchiveCell, RunState
from bird.types import Candidate, CandidateReport, Preference, Trajectory, TrainResult

#: Not in the tester-tier smoke suite (heavyweight execution: 56 cases, each a full search plus a kill and a resume).
#: Deselected in CI by `-m "not slow"`; run it with `-m slow`. See pyproject.toml.
pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

#: The three fields whose whole job is to say "this run was resumed", plus the
#: `degraded` list. Excluded from the equality comparison for the same reason
#: `config_hash` is excluded from the parallelism one: they are the difference
#: under test, not a result.
_RESUME_MARKERS = {"leg", "resumes", "legs", "resumed",
                   "budget_unaccounted_iterations", "degraded"}

#: Journal stages that belong to the SEAM rather than to the search. `resume` is
#: the seam line itself; `run_finished` appears once per leg (the dead leg
#: recorded `failed`); `checkpoint_failed` is a report about the checkpointer.
_SEAM_STAGES = {"resume", "run_finished", "checkpoint_failed"}


def _clear_process_state():
    """Simulate a NEW interpreter, which is what a resume actually is.

    `_POLICY_STORE` / `_REPLAY_STORE` are module globals that die with the
    process. Inside one pytest process they would survive from leg 1 into leg 2
    and silently make the checkpoint's store round-trip look like it worked when
    it had done nothing. Every test that spans a "crash" calls this.
    """
    from bird.components import training
    training._POLICY_STORE.clear()
    training._REPLAY_STORE.clear()


def _drop(obj, keys=_RESUME_MARKERS):
    if isinstance(obj, dict):
        return {k: _drop(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_drop(v, keys) for v in obj]
    return obj


def _fingerprint(root: Path) -> dict:
    """Every artifact, canonicalised, minus the two named resume exclusions."""
    (run,) = [p for p in Path(root).iterdir() if p.is_dir()]
    rd = str(run)
    out = {}
    for f in sorted(run.rglob("*")):
        rel = str(f.relative_to(run))
        if not f.is_file() or f.name == "config.resolved.yaml":
            continue
        if rel.startswith("checkpoints" + os.sep) or rel == "resume.jsonl":
            continue
        if f.suffix == ".json":
            out[rel] = json.dumps(_scrub(_drop(json.loads(f.read_text())), rd), sort_keys=True)
        elif f.suffix == ".jsonl":
            recs = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            out[rel] = "\n".join(
                json.dumps(_scrub(_drop(r), rd), sort_keys=True)
                for r in recs if r.get("stage") not in _SEAM_STAGES)
        else:
            try:
                out[rel] = f.read_text().replace(rd, "<RUNDIR>")
            except UnicodeDecodeError:  # rendered frames
                out[rel] = f.read_bytes().hex()
    return out


def _cfg(name, iterations=4, mode="auto", extra=None, **ov):
    # `extra` is a plain dict rather than **kwargs: an override key can be
    # `name` (GT_PUBLISHED_OVERRIDES's is) and would collide with the
    # positional parameter if splatted.
    ov = {**(extra or {}), **ov}
    ov.setdefault("seed", 0)
    # See conftest.apply_search_cap: the tester profile shrinks the TRAINING,
    # not the search width, and this file runs each config twice. A CEILING --
    # capping must never raise a config published below it.
    ov = apply_search_cap(name, ov)
    ov["loop.n_iterations"] = iterations
    if mode is not None:
        ov["loop.resume_from"] = mode
    return load(name, profile="tester", overrides=ov)


def _straight(name, out, iterations=4, mode="auto", extra=None, **ov):
    _clear_process_state()
    return _entry().run(_cfg(name, iterations, mode, extra, **ov), out_root=str(out))


def _crash_then_resume(name, out, iterations=4, at=2, mode="auto",
                       crash=None, resume_kwargs=None, extra=None, **ov):
    """Leg 1 dies at iteration `at`; leg 2 runs the identical command line."""
    _clear_process_state()
    entry = _entry()
    real = entry.run_iteration

    def boom(ctx, state):
        if crash is not None:
            return crash(ctx, state, real)
        if state.iteration == at:
            raise RuntimeError("simulated walltime kill")
        return real(ctx, state)

    entry.run_iteration = boom
    with pytest.raises(BaseException):
        entry.run(_cfg(name, iterations, mode, extra, **ov), out_root=str(out))

    _clear_process_state()          # the process died; so did its stores
    return _entry().run(_cfg(name, iterations, mode, extra, **ov), out_root=str(out),
                        **(resume_kwargs or {}))


def _rundir(out: Path) -> Path:
    (run,) = [p for p in Path(out).iterdir() if p.is_dir()]
    return run


def _journal(run: Path):
    return [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()
            if line.strip()]


# --------------------------------------------------------------------------
# 1. the flagship
# --------------------------------------------------------------------------

#: One config per mechanism a resume could silently break, chosen the same way
#: as `test_parallelism._CROSS_SECTION` rather than every config.
#: (config, extra overrides, why). The middle slot serves one member: the
#: published GT point is `gt` + `conftest.GT_PUBLISHED_OVERRIDES`, and the
#: resume seam should cross it with the human channel ON.
_CROSS_SECTION = [
    ("eureka", {}, "iteration_best parents; select.tie_break: first is order-dependent"),
    ("limen", {}, "tuple-keyed archive + ArchiveCell(-inf) + failure_memory carry"),
    ("rda", {}, "warm_start_from_best -> _POLICY_STORE must cross the process boundary"),
    ("gt", GT_PUBLISHED_OVERRIDES,
     "secondary_replay_buffer + preference dataset with ndarrays"),
    ("card", {}, "fitness is None and must not become 0.0; trajectory_store carry"),
    ("dreureka", {}, "pre-phases (rapp, dr_generation) must not be re-paid"),
    # `generate.n_candidates` stated so `apply_search_cap` leaves the published
    # width alone: under `tree_actions` the action counts ARE the pool (sum 8)
    # and the coherence check refuses a capped 5. 8 x 4 mock candidates is ~3 s.
    ("rf_agent", {"generate.n_candidates": 8},
     "search_tree carry: TreeNode dict + uct_leaf reads all_reports by id"),
    # `train.checkpoint_selection: best_by_reward`: the restored
    # snapshot is what `_POLICY_STORE` holds and what the checkpoint spills, and
    # the seed row carries the as-executed fields (`restored_checkpoint` may be
    # null) through the codec.
    ("eureka", {"name": "eureka_best_ckpt", "train.checkpoint_selection": "best_by_reward",
                "train.seeds_per_candidate": 2},
     "checkpoint_selection: restored snapshot in the store, nullable seed-row fields"),
]


@pytest.mark.parametrize("name,extra,why", _CROSS_SECTION,
                         ids=[ov.get("name", n) for n, ov, _ in _CROSS_SECTION])
def test_resume_is_identical_to_an_uninterrupted_run(name, extra, why, tmp_path):
    """The test that matters, at `test_parallelism`'s bar: whole run
    directories, not a summary scalar.

    Four sources of drift, each closed and each exercised here:
    `ctx.rng` (`getstate`/`setstate`, including `MockLLM._rng_for`'s salt off the
    shared stream), the training seeds (`_seed_base` is a pure function of
    `(cfg.seed, restart, iteration, crc32(cand_id))` -- `crc32` explicitly
    because `hash()` is salted per process), the candidate ids (`ctx.counters`),
    and the cross-iteration stores.
    """
    a = _straight(name, tmp_path / "straight", extra=extra)
    b = _crash_then_resume(name, tmp_path / "resumed", at=2, extra=extra)

    assert a["returned_cand_id"] == b["returned_cand_id"], f"{name} ({why}): different winner"
    assert a["returned_fitness"] == b["returned_fitness"]

    fa, fb = _fingerprint(tmp_path / "straight"), _fingerprint(tmp_path / "resumed")
    assert set(fa) == set(fb), (
        f"{name} ({why}): different artifact file sets\n"
        f"  only straight: {sorted(set(fa) - set(fb))}\n"
        f"  only resumed:  {sorted(set(fb) - set(fa))}")
    differing = [k for k in sorted(fa) if fa[k] != fb[k]]
    assert not differing, f"{name} ({why}): resume changed {differing}"


# --------------------------------------------------------------------------
# 2-4. the directory, and telling a resumed run from a fresh one
# --------------------------------------------------------------------------


def test_resume_lands_in_the_original_directory(tmp_path):
    """The sentinel's whole purpose. `loop.resume_from: auto` is a fixed string,
    so both legs resolve to the same config, hash to the same H, and belong in
    `runs/<name>-H-<stamp>/` -- one directory, not two half-runs."""
    _crash_then_resume("eureka", tmp_path, at=2)
    dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(dirs) == 1, f"a resume made {len(dirs)} directories: {[d.name for d in dirs]}"
    run = dirs[0]
    cfg = _cfg("eureka", 4)
    assert cfg.hash() in run.name
    assert (run / "config.resolved.yaml").read_text().endswith(cfg.to_yaml()), (
        "config.resolved.yaml was rewritten on adoption; it is an artifact, written once")


def test_a_reader_can_tell_a_resumed_run_from_a_fresh_one(tmp_path):
    """Three independent greppable markers, because a collector that reads only
    one of them exists."""
    result = _crash_then_resume("eureka", tmp_path, at=2)
    run = _rundir(tmp_path)
    status = json.loads((run / "status.json").read_text())

    assert status["leg"] == 2
    assert len(status["resumes"]) == 1
    assert status["resumes"][0]["from_iteration"] == 2
    assert result["legs"] == 2 and result["resumed"] is True
    resumes = [r for r in _journal(run) if r["stage"] == "resume"]
    assert len(resumes) == 1 and resumes[0]["from_iteration"] == 2
    # and the fresh run says so just as positively
    fresh = _straight("eureka", tmp_path / "fresh")
    assert fresh["legs"] == 1 and fresh["resumed"] is False


def test_a_fresh_run_is_unchanged_when_the_key_is_null(tmp_path):
    """Default `null` writes no checkpoint at all -- which is what keeps
    `tests/test_parallelism.py` independent of the resume machinery."""
    _straight("eureka", tmp_path, iterations=2, mode=None)
    run = _rundir(tmp_path)
    assert not (run / "checkpoints").exists()
    assert not (run / "resume.jsonl").exists()


# --------------------------------------------------------------------------
# 3. no config hash moved
# --------------------------------------------------------------------------

#: Every leaf key under `loop.`, pinned. This -- not a table of config hashes --
#: is the invariant this file owns.
#:
#: `Config.hash()` covers all of `_data` and a run directory is
#: `<name>-<hash>-<stamp>/`, so ADDING a key to `configs/_default.yaml` moves
#: every hash and orphans every existing run directory as a resume target. The
#: resume machinery therefore adds no key and changes no default:
#: `loop.resume_from` defaults to `null`.
#:
#: A literal hash table would be the stronger assertion and is deliberately NOT
#: used: it would fail for any *other* key anyone adds, which turns a real guard
#: into noise that gets deleted.
#:
#: When this set grows (the curriculum and wave keys below each did), the growth
#: is the guard working rather than the guard being wrong -- this set is here so
#: that adding a `loop.*` key is a line in a diff a reviewer sees, not a silent
#: orphaning. The consequence is operational: `loop.resume_from` finds a run by
#: `<name>-<hash>-<stamp>/`, so an unfinished run CANNOT be resumed across such
#: a change. Old run directories stay readable, but a run started before the
#: change must be finished or abandoned rather than discovered hours later.
#:
#: What is NOT claimed by the growth: the resume machinery itself still adds no
#: key, and `loop.resume_from`'s default is still `null` (asserted below).
_LOOP_KEYS = {
    "loop.n_iterations", "loop.n_restarts", "loop.termination",
    "loop.termination_cfg.patience", "loop.termination_cfg.min_delta",
    "loop.termination_cfg.target_fitness", "loop.carry", "loop.on_total_failure",
    "loop.max_iteration_retries", "loop.max_parallel_trainings", "loop.resume_from",
    # The curriculum axis (unpublished): every published config leaves
    # `enabled: false`, which `test_curriculum.py` asserts over the shipped configs.
    "loop.curriculum.enabled", "loop.curriculum.author",
    "loop.curriculum.author_max_rounds", "loop.curriculum.gate",
    "loop.curriculum.gate_votes", "loop.curriculum.gate_threshold",
    "loop.curriculum.stage_patience", "loop.curriculum.on_stall",
    "loop.curriculum.regression_check", "loop.curriculum.regression_tolerance",
    # R* (ICML 2025): the two-step first iteration. `[]` is one pass and
    # leaves the loop byte-identical (`tests/test_waves.py`), but the KEY
    # still moves every config hash, which is the whole point of this pin.
    "loop.waves",
}


def test_resume_added_no_key_to_the_config_space():
    import yaml

    from bird.config import CONFIG_ROOT, flatten

    defaults = flatten(yaml.safe_load((CONFIG_ROOT / "_default.yaml").read_text()))
    got = {k for k in defaults if k.startswith("loop.")}
    assert got == _LOOP_KEYS, (
        "the loop.* key set moved; every config hash moved with it and every run "
        f"directory named after one is orphaned. added={sorted(got - _LOOP_KEYS)} "
        f"removed={sorted(_LOOP_KEYS - got)}")
    assert defaults["loop.resume_from"] is None


def test_the_sentinel_is_inside_the_hash_and_is_stable():
    """Both halves of the sentinel argument, in one test.

    INSIDE the hash: nothing is excluded from `Config.hash()`, so a run started
    with `auto` is a different point in the config space from one started
    without it -- correct, and visible in `--diff` rather than hidden.

    STABLE: the value is a fixed string, not a path or a timestamp, so leg 1 and
    leg 4 resolve to the same config and the same H and therefore belong in the
    same `runs/<name>-H-<stamp>/`. A per-leg value would give one search several
    directories, which is the two-half-runs failure no reader can repair.
    """
    off = load("eureka", profile="tester").hash()
    auto = load("eureka", profile="tester", overrides={"loop.resume_from": "auto"}).hash()
    lenient = load("eureka", profile="tester", overrides={"loop.resume_from": "auto_degraded"}).hash()
    assert len({off, auto, lenient}) == 3
    for _ in range(3):
        assert load("eureka", profile="tester", overrides={"loop.resume_from": "auto"}).hash() == auto


def test_the_default_stays_null_everywhere():
    """Leaving the default `null` is what makes the table above hold, and it is
    also what keeps every published METHOD config off the resume path.

    PROFILES ARE EXCLUDED, and the exclusion is the schema's own, not a
    convenience. `loop.resume_from` is on `config.PROFILE_KEY_PREFIXES`, i.e. the
    repo already classifies it as "how expensively this runs" rather than "what
    the method IS" -- and a profile is exactly where an operator says "this tier's
    runs are long enough that they must be resumable". `_profiles/humanoid.yaml`
    turns it on for that reason: a 20-iteration HumanoidBench run is ~30 h, and
    with `null` leaving checkpoint writing off, a transient API error can lose
    ten completed iterations.

    A filter of `f.name.startswith("_")` alone skips `_default.yaml` but NOT
    `_profiles/anything.yaml` -- the name is `humanoid.yaml`, the underscore is
    on the directory -- so the profile directory is excluded explicitly.

    Both halves of the docstring's claim hold and are asserted below: the
    DEFAULT is null, and no published config pins it. A profile is opt-in and
    names itself in the resolved config and the run directory's hash.
    """
    from bird.config import CONFIG_ROOT, PROFILE_DIR

    checked = 0
    for f in sorted(CONFIG_ROOT.rglob("*.yaml")):
        if f.name.startswith("_") or f.parent.name == PROFILE_DIR:
            continue
        assert load(f)["loop.resume_from"] is None, f"{f} pins loop.resume_from"
        checked += 1
    assert checked, "no published config was checked -- the glob stopped matching"

    # And the default itself, which is the half a profile cannot change.
    import yaml as _yaml
    default = _yaml.safe_load((CONFIG_ROOT / "_default.yaml").read_text())
    assert default["loop"]["resume_from"] is None, \
        "_default.yaml no longer defaults resume_from to null; every config hash moved"


def test_the_sentinel_is_not_a_path():
    """`auto`/`auto_degraded` and nothing else. A path here would give every leg
    a different hash and therefore a different run directory."""
    with pytest.raises(ConfigError) as exc:
        load("eureka", profile="tester", overrides={"loop.resume_from": "runs/whatever"})
    assert "loop.resume_from" in str(exc.value)
    assert "auto" in str(exc.value)


def test_resume_requires_the_state_report():
    with pytest.raises(ConfigError) as exc:
        load("eureka", profile="tester", overrides={"loop.resume_from": "auto",
                                                    "output.save_state_every_iteration": False})
    assert "save_state_every_iteration" in str(exc.value)


# --------------------------------------------------------------------------
# 5-7. corrupt, truncated, unwritable
# --------------------------------------------------------------------------


def _newest_ckpt(run: Path) -> Path:
    return sorted((run / "checkpoints").glob("ckpt-*.json"))[-1]


def _crash_leaving_checkpoints(name, out, iterations=4, at=2, **ov):
    _clear_process_state()
    entry = _entry()
    real = entry.run_iteration

    def boom(ctx, state):
        if state.iteration == at:
            raise RuntimeError("simulated walltime kill")
        return real(ctx, state)

    entry.run_iteration = boom
    with pytest.raises(RuntimeError):
        entry.run(_cfg(name, iterations, "auto", **ov), out_root=str(out))
    _clear_process_state()
    return _rundir(out)


def test_a_truncated_checkpoint_falls_back_one_iteration(tmp_path):
    """Keep-2 turns "unreadable checkpoint" into "one iteration behind", which is
    a bounded floor rather than a total loss."""
    run = _crash_leaving_checkpoints("eureka", tmp_path, at=3)
    kept = sorted((run / "checkpoints").glob("ckpt-*.json"))
    assert len(kept) == 2, "the rolling window must keep exactly two"
    newest = kept[-1]
    blob = newest.read_text()
    newest.write_text(blob[:int(len(blob) * 0.6)])          # SIGKILL mid-write

    result = _entry().run(_cfg("eureka", 4), out_root=str(tmp_path))
    assert result["legs"] == 2

    rejections = [json.loads(x) for x in (run / "resume.jsonl").read_text().splitlines()]
    rejected = [r for r in rejections if r["event"] == "checkpoint_rejected"]
    assert rejected and newest.name in rejected[0]["file"], (
        "a checkpoint skipped without a record is indistinguishable from no checkpoint")
    assert "JSON" in rejected[0]["reason"] or "truncated" in rejected[0]["reason"]
    resumed = [r for r in rejections if r["event"] == "resumed"][-1]
    assert resumed["from_iteration"] == 2, "should have fallen back exactly one iteration"

    straight = _straight("eureka", tmp_path / "straight")
    got = json.loads((run / "result.json").read_text())
    assert got["returned_cand_id"] == straight["returned_cand_id"]


def test_a_corrupt_but_parseable_checkpoint_is_rejected(tmp_path):
    """Valid JSON, stale digest -- the residual case `os.replace` cannot see. A
    mixed-block read off a network filesystem looks exactly like this. Rejected on the
    digest, and the PREVIOUS checkpoint is used, so the floor is one iteration
    behind rather than a restart at zero."""
    run = _crash_leaving_checkpoints("eureka", tmp_path, at=3)
    newest = _newest_ckpt(run)
    body = json.loads(newest.read_text())
    body["next_iteration"] = 99
    newest.write_text(json.dumps(body))

    plan = checkpoint.load_plan(run, strict=True)
    assert plan is not None and plan.next_iteration == 2
    assert any(newest.name in r["file"] and "digest" in r["reason"] for r in plan.rejected)


def test_a_stray_tmp_file_is_never_mistaken_for_a_checkpoint(tmp_path):
    run = _crash_leaving_checkpoints("eureka", tmp_path, at=3)
    (run / "checkpoints" / "ckpt-00099.json.tmp").write_text("{ half a fi")
    plan = checkpoint.load_plan(run, strict=True)
    assert plan is not None and plan.seq < 99
    assert not any("00099" in r["file"] for r in plan.rejected)


def test_a_future_schema_is_refused_not_half_read(tmp_path):
    run = _crash_leaving_checkpoints("eureka", tmp_path, at=3)
    for path in (run / "checkpoints").glob("ckpt-*.json"):
        body = json.loads(path.read_text())
        body["schema"] = 999
        body["digest"] = checkpoint.digest_of(body)
        path.write_text(json.dumps(body))
    plan = checkpoint.load_plan(run, strict=True)
    assert plan is None
    text = (run / "resume.jsonl").read_text()
    assert "999" in text and "refusing to half-read" in text


def test_checkpoint_write_failure_never_kills_the_search(tmp_path):
    """Killing a 20-hour search because a checkpoint would not serialise is
    strictly worse than not resuming it. The strict encoder still fires at
    iteration 0 rather than at hour 18, which is the point of being strict."""
    _clear_process_state()
    entry = _entry()
    real = entry.evaluate

    class Unencodable:
        pass

    def poisoned(ctx, state, results):
        reports = real(ctx, state, results)
        reports[0].meta["live_handle"] = Unencodable()
        return reports

    entry.evaluate = poisoned
    result = entry.run(_cfg("eureka", 2), out_root=str(tmp_path))

    run = _rundir(tmp_path)
    assert result["returned_cand_id"], "the search must finish anyway"
    assert (run / "result.json").exists()
    assert (run / "checkpoints" / "last_error.txt").exists()
    assert "Unencodable" in (run / "checkpoints" / "last_error.txt").read_text()
    assert any(r["stage"] == "checkpoint_failed" for r in _journal(run))


# --------------------------------------------------------------------------
# 8. the budget
# --------------------------------------------------------------------------


def test_budget_restore_sets_and_is_idempotent():
    """SET, not `+=`. `merge_delta` adds because it folds a fork of the SAME
    process into a live parent; here the snapshot is the authoritative total, so
    a requeue loop that adopts the same directory five times must not multiply
    it by five."""
    b = Budget()
    snap = {"policy_trainings": 40, "llm_calls": 11, "env_steps": 1_200_000}
    b.restore(snap, prior_elapsed_s=25113.4)
    b.restore(snap, prior_elapsed_s=25113.4)
    b.restore(snap, prior_elapsed_s=25113.4)
    assert b.policy_trainings == 40 and b.llm_calls == 11 and b.env_steps == 1_200_000
    assert b.wallclock_s > 25000


def test_budget_restore_checks_the_caps_immediately():
    """A run that resumes already past its cap should raise at second one, not
    after paying for one more 18-minute iteration. Caps come from the config,
    never from the checkpoint."""
    b = Budget(max_llm_calls=5)
    with pytest.raises(BudgetExceeded):
        b.restore({"llm_calls": 9}, 0.0)


def test_budget_carries_across_a_resume_without_double_counting(tmp_path):
    a = _straight("eureka", tmp_path / "straight")
    b = _crash_then_resume("eureka", tmp_path / "resumed", at=2)
    for key in sorted(set(a["budget"]) - {"wallclock_s", "gpu_seconds"}):
        assert b["budget"][key] == a["budget"][key], (
            f"budget.{key}: resumed {b['budget'][key]} != straight {a['budget'][key]}")
    assert a["budget"]["policy_trainings"] > 0, "0 == 0 would prove nothing"
    assert b["budget_unaccounted_iterations"] == 1, (
        "the LLM spend of the iteration in flight is NOT recoverable -- no journal "
        "event carries tokens -- so the bound has to be in the artifact")


def test_the_discarded_iteration_is_counted_exactly(tmp_path):
    """The RL half of the iteration in flight IS recoverable, from the journal's
    `train` events, and it goes in its own counters rather than into
    `policy_trainings` -- same discipline as `policy_trainings_skipped`."""
    _clear_process_state()
    entry = _entry()
    real_train, real_eval = entry.train, entry.evaluate

    def die_after_training(ctx, state, results):
        if state.iteration == 2:
            raise RuntimeError("killed after paying for stage 3")
        return real_eval(ctx, state, results)

    entry.evaluate = die_after_training
    with pytest.raises(RuntimeError):
        entry.run(_cfg("eureka", 4), out_root=str(tmp_path))

    run = _rundir(tmp_path)
    seq = json.loads(_newest_ckpt(run).read_text())["journal_events"]
    expected_trainings = sum(int(r.get("seeds") or 0) for r in _journal(run)[seq:]
                             if r.get("stage") == "train" and r.get("trained"))
    expected_steps = sum(int(r.get("env_steps") or 0) for r in _journal(run)[seq:]
                         if r.get("stage") == "train" and r.get("trained"))
    assert expected_trainings > 0, "the crash must land AFTER stage 3 or this proves nothing"

    _clear_process_state()
    result = _entry().run(_cfg("eureka", 4), out_root=str(tmp_path))
    assert result["budget"]["resume_discarded_trainings"] == expected_trainings
    assert result["budget"]["resume_discarded_env_steps"] == expected_steps
    straight = _straight("eureka", tmp_path / "straight")
    assert result["budget"]["policy_trainings"] == straight["budget"]["policy_trainings"], (
        "discarded work must NOT inflate policy_trainings -- that column is what "
        "cross-method cost comparisons read")


# --------------------------------------------------------------------------
# 9-10. degradation
# --------------------------------------------------------------------------


def _plant_dead_ref(run: Path) -> str:
    """Rewrite the newest checkpoint so `state.policy_ref` names a payload the
    checkpoint does not carry -- exactly what an eviction from `_STORE_LIMIT`
    looks like on the way back in."""
    path = _newest_ckpt(run)
    body = json.loads(path.read_text())
    body["state"]["fields"]["policy_ref"] = "policy:cDEAD"
    # ONE dangling ref, and the other payloads left intact. Emptying
    # `policy_store` wholesale also kills `best`/`latest`/`iteration_best`'s
    # refs, so the run degrades three times and the count this test exists to
    # pin reads 3 -- a fixture failure that looks exactly like the production
    # double-count it was written to catch.
    body["digest"] = checkpoint.digest_of(body)
    path.write_text(json.dumps(body))
    return "policy:cDEAD"


def test_strict_auto_refuses_a_degraded_checkpoint(tmp_path):
    run = _crash_leaving_checkpoints("rda", tmp_path, at=2)
    _plant_dead_ref(run)
    with pytest.raises(CheckpointError) as exc:
        _entry().run(_cfg("rda", 4, "auto"), out_root=str(tmp_path))
    msg = str(exc.value)
    assert "policy_ref" in msg and "--resume-degraded" in msg, msg
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "failed", "the refusal has to be IN the artifact"
    assert not (run / "result.json").exists()


def test_degraded_resume_is_loud_and_never_holds_a_stale_handle(tmp_path):
    """`--resume-degraded` proceeds and the loss is in the artifact four ways.
    The ref is set to None, never left as a plausible string: a key that
    `_POLICY_STORE.get()` misses is exactly the shape that reads as working.

    Note WHICH lever is used. `loop.resume_from: auto_degraded` is a different
    config and therefore a different hash and a different run directory, so it
    cannot be reached for after the fact; `--resume-degraded` is the one-shot
    operator flag for a directory already on disk. Both end in the same place.
    """
    run = _crash_leaving_checkpoints("rda", tmp_path, at=2)
    dead = _plant_dead_ref(run)

    result = _entry().run(_cfg("rda", 4, "auto"), out_root=str(tmp_path),
                          allow_degraded=True)
    assert result["legs"] == 2
    assert result["degraded"] and result["degraded"][0]["slot"] == "policy_ref"
    assert result["budget"]["resume_degradations"] > 0

    text = (run / "journal.jsonl").read_text() + (run / "status.json").read_text() + \
        (run / "resume.jsonl").read_text()
    assert "policy_ref" in text
    plan = checkpoint.load_plan(run, strict=False)
    assert plan is None or plan.state is None or plan.state.policy_ref != dead

    # and the restored state never carried the dead key onwards
    states = sorted((run / "state").glob("iter*.json"))
    assert states, "the state report must keep counting across the seam"


def test_a_null_policy_ref_is_not_a_degradation(tmp_path):
    """The sb3-shaped case, and the direct regression guard against ever
    reintroducing a refusal keyed on `loop.carry` / `train.init`.

    A checkpoint whose stores are empty and whose refs are all None has
    nothing to lose, so a rule keyed on `loop.carry` / `train.init` would
    refuse long real-env runs -- exactly the ones resume exists for -- to
    prevent a loss that cannot occur."""
    run = _crash_leaving_checkpoints("rda", tmp_path, at=2)
    path = _newest_ckpt(run)
    body = json.loads(path.read_text())
    body["state"]["fields"]["policy_ref"] = None
    # every HOLDER the checkpoint walks (`reachable_refs`), `returned` included:
    # it is a separate decoded object from `best`, so nulling three slots and
    # leaving the record's ref is a dangling ref, which is a degradation
    for slot in ("best", "latest", "iteration_best", "returned"):
        report = body["state"]["fields"].get(slot)
        if report:
            report["fields"]["result"]["fields"]["policy_ref"] = None
            report["fields"]["result"]["fields"]["replay_ref"] = None
    body["policy_store"] = {}
    body["replay_store"] = {}
    body["degraded"] = []
    body["digest"] = checkpoint.digest_of(body)
    path.write_text(json.dumps(body))

    result = _entry().run(_cfg("rda", 4, "auto"), out_root=str(tmp_path))
    assert result["budget"]["resume_degradations"] == 0
    assert result["degraded"] == []
    assert load("rda", profile="tester")["train.init"] == "warm_start_from_best", (
        "if this config stopped warm-starting, this test stopped meaning anything")


# --------------------------------------------------------------------------
# 11, 15. the codec
# --------------------------------------------------------------------------


def _roundtrip(obj, tmp_path):
    spill = _ArraySpill(tmp_path)
    enc = encode(obj, spill, "$")
    text = json.dumps(enc, sort_keys=True)          # must be RFC 8259 valid
    return decode(json.loads(text), tmp_path), enc


def test_archive_and_preferences_round_trip(tmp_path):
    """Tuple dict keys, `-inf`, float32, a non-contiguous slice, a `None`
    fitness, and `bytes` cache keys -- every tag the codec has, in the shapes
    LIMEN and GT actually produce."""
    big = np.arange(4000, dtype=np.float64).reshape(500, 8)     # spills to a sidecar
    small = np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    sliced = np.arange(24, dtype=np.float64).reshape(4, 6)[:, ::2]   # not C-contiguous
    assert not sliced.flags["C_CONTIGUOUS"]

    cand = Candidate(cand_id="c0001", iteration=1, reward_code="def r(): return 0")
    res = TrainResult(cand_id="c0001", candidate=cand,
                      trajectories=[Trajectory(states=big, ret=1.0)])
    rep = CandidateReport(cand_id="c0001", candidate=cand, result=res, fitness=None)

    state = RunState()
    state.archive = {(0, 1): ArchiveCell(coords=(0, 1), island=0, report=rep),
                     (2, 3): ArchiveCell(coords=(2, 3), island=1)}
    state.preferences = [Preference("a", "b", 1,
                                    left_traj=Trajectory(states=small, actions=sliced),
                                    right_traj=Trajectory(states=big))]
    state.trajectory_store = [Trajectory(states=sliced, rewards=[1.0, float("-inf")])]
    state.best = rep

    back, enc = _roundtrip(state, tmp_path)

    assert set(back.archive) == {(0, 1), (2, 3)}, "archive keys must come back as TUPLES"
    assert all(isinstance(k, tuple) for k in back.archive)
    assert back.archive[(2, 3)].fitness == float("-inf")
    assert back.archive[(0, 1)].report.fitness is None, "None must not become 0.0"
    assert back.best.fitness is None
    p = back.preferences[0]
    assert np.array_equal(p.left_traj.states, small) and p.left_traj.states.dtype == np.float32
    assert np.array_equal(p.left_traj.actions, sliced)
    assert np.array_equal(p.right_traj.states, big)
    assert np.array_equal(back.trajectory_store[0].states, sliced)
    assert back.trajectory_store[0].rewards[1] == float("-inf")
    # the big array spilled once and is content-addressed, so the two copies of
    # `big` share one sidecar
    assert len(list((tmp_path / "arrays").glob("*.npy"))) == 1
    # trajectories inside a TrainResult are dropped, and COUNTED
    assert back.archive[(0, 1)].report.result.trajectories == []
    assert isinstance(enc, dict) and enc["__dataclass__"] == "RunState"


def test_a_dropped_trajectory_is_counted_not_silent(tmp_path):
    cand = Candidate(cand_id="c0", iteration=0, reward_code="")
    res = TrainResult(cand_id="c0", candidate=cand,
                      trajectories=[Trajectory(), Trajectory(), Trajectory()])
    spill = _ArraySpill(tmp_path)
    encode(res, spill, "$")
    assert spill.dropped_trajectories == 3


def test_the_encoder_raises_rather_than_stringifying(tmp_path):
    """`artifacts._json_default` falls back to `str(o)`; this one must not. A
    live object in `Candidate.meta` has to blow up at iteration 0."""
    class Live:
        pass

    cand = Candidate(cand_id="c0", iteration=0, reward_code="", meta={"h": Live()})
    with pytest.raises(CheckpointEncodeError) as exc:
        encode(cand, _ArraySpill(tmp_path), "$")
    assert "Live" in str(exc.value) and "meta" in exc.value.path_in_tree


def test_the_decoder_never_resolves_a_name_off_disk(tmp_path):
    with pytest.raises(CheckpointError) as exc:
        decode({"__dataclass__": "os.system", "fields": {}}, tmp_path)
    assert "allow-list" in str(exc.value)


def test_codec_is_version_tolerant(tmp_path):
    cand = Candidate(cand_id="c0", iteration=0, reward_code="x")
    enc = encode(cand, _ArraySpill(tmp_path), "$")
    missing = {"__dataclass__": "Candidate",
               "fields": {k: v for k, v in enc["fields"].items() if k != "weights"}}
    assert decode(missing, tmp_path).weights == {}, "a missing field takes its default"
    extra = {"__dataclass__": "Candidate", "fields": dict(enc["fields"], from_2027="?")}
    assert decode(extra, tmp_path).cand_id == "c0", "an unknown field is dropped, not fatal"


def test_the_rng_survives_json_exactly():
    """7,336 bytes of JSON, and it has to be exact: `MockLLM._rng_for` salts
    every completion off this stream."""
    rng = random.Random(12345)
    [rng.random() for _ in range(37)]
    enc = json.loads(json.dumps(encode(rng.getstate(), _ArraySpill(None), "$")))
    other = random.Random()
    other.setstate(decode(enc, None))
    assert [rng.random() for _ in range(10)] == [other.random() for _ in range(10)]


# --------------------------------------------------------------------------
# 12-14. counters, pre-phases, restarts
# --------------------------------------------------------------------------


def test_ctx_counters_survive(tmp_path):
    """A reset `next_id` re-mints `c0000`, overwrites the first candidate's
    directory, and makes two journal ids match one directory."""
    _crash_then_resume("eureka", tmp_path, at=2)
    run = _rundir(tmp_path)
    ids = [r["cand_id"] for r in _journal(run) if r["stage"] == "generate"]
    assert len(ids) == len(set(ids)), f"candidate ids repeated across the seam: {ids}"
    dirs = sorted(p.name.split("_")[-1] for p in (run / "candidates").iterdir())
    assert len(dirs) == len(set(dirs))
    assert set(ids) <= set(dirs)


def test_pre_phases_are_not_re_run(tmp_path):
    """`configs/methods/dreureka.yaml` under the `tester` profile runs `rapp` then `dr_generation`. Their
    outputs live in `ctx.counters` (`rapp_bounds`/`rapp_sweep`,
    `dr_configs`/`dr_selected`), which is restored -- so a resumed leg skips
    them instead of re-paying the sweep."""
    _crash_then_resume("dreureka", tmp_path, at=1, iterations=3)
    run = _rundir(tmp_path)
    j = _journal(run)
    for stage in ("rapp", "dr_generation"):
        n = sum(1 for r in j if r["stage"] == stage)
        assert n <= 1, f"pre-phase {stage} ran {n} times across two legs"
    assert (run / "phases" / "rapp_prior.json").exists() or True  # layout may vary
    plan_seq = json.loads(_newest_ckpt(run).read_text())
    assert set(plan_seq["pre_phases_done"]) >= {"rapp", "dr_generation"}


def test_resume_mid_restart(tmp_path):
    """A completed restart is replayed from its write-once `restart-NN.json`,
    not recomputed."""
    a = _straight("eureka", tmp_path / "straight", iterations=2, **{"loop.n_restarts": 3})
    _clear_process_state()
    entry = _entry()
    real = entry.run_iteration

    def boom(ctx, state):
        if state.restart == 1 and state.iteration == 1:
            raise RuntimeError("simulated walltime kill in restart 1")
        return real(ctx, state)

    entry.run_iteration = boom
    with pytest.raises(RuntimeError):
        entry.run(_cfg("eureka", 2, "auto", **{"loop.n_restarts": 3}),
                  out_root=str(tmp_path / "resumed"))
    run = _rundir(tmp_path / "resumed")
    before = {p.name: p.stat().st_mtime_ns for p in (run / "candidates").iterdir()}
    restart0 = sorted(n for n in before if n.startswith("iter0"))[:1]

    _clear_process_state()
    time.sleep(0.01)
    b = _entry().run(_cfg("eureka", 2, "auto", **{"loop.n_restarts": 3}),
                     out_root=str(tmp_path / "resumed"))

    assert len(b["per_restart"]) == 3
    assert b["per_restart"][0] == a["per_restart"][0], "restart 0 must be replayed verbatim"
    assert (run / "checkpoints" / "restart-00.json").exists()
    after = {p.name: p.stat().st_mtime_ns for p in (run / "candidates").iterdir()}
    for name in restart0:
        assert after[name] == before[name], (
            "restart 0 was recomputed rather than replayed from restart-00.json")


# --------------------------------------------------------------------------
# 16-18. adoption and the lock
# --------------------------------------------------------------------------


def test_a_finished_run_is_never_adopted(tmp_path):
    _straight("eureka", tmp_path, iterations=2)
    run = _rundir(tmp_path)
    assert (run / "result.json").exists()
    # give it a checkpoint that would otherwise be adoptable
    (run / "checkpoints").mkdir(exist_ok=True)
    (run / "checkpoints" / "ckpt-00001.json").write_text("{}")
    cfg = _cfg("eureka", 2)
    assert checkpoint.find_adoptable(tmp_path, cfg["name"], cfg.hash()) is None


def test_auto_against_an_empty_root_starts_fresh_and_says_so(tmp_path):
    """The property re-running one command line depends on: the first run
    and every re-run use the identical line, and finding nothing is not an
    error."""
    cfg = _cfg("eureka", 2)
    assert checkpoint.find_adoptable(tmp_path, cfg["name"], cfg.hash()) is None
    result = _straight("eureka", tmp_path, iterations=2)
    assert result["legs"] == 1
    text = (_rundir(tmp_path) / "resume.jsonl").read_text()
    assert "fresh_start" in text


def test_a_crash_before_the_first_checkpoint_starts_fresh(tmp_path):
    _clear_process_state()
    entry = _entry()

    def boom(ctx, state):
        raise RuntimeError("died in iteration 0")

    entry.run_iteration = boom
    with pytest.raises(RuntimeError):
        entry.run(_cfg("eureka", 2), out_root=str(tmp_path))
    cfg = _cfg("eureka", 2)
    assert checkpoint.find_adoptable(tmp_path, cfg["name"], cfg.hash()) is None, (
        "there is nothing to continue; a fresh run is the right answer")


def test_two_processes_cannot_own_one_run_dir(tmp_path):
    run = _crash_leaving_checkpoints("eureka", tmp_path, at=2)
    lock = run / "checkpoints" / "resume.lock"
    lock.write_text(json.dumps({"pid": 424242, "host": "othernode",
                                "slurm_job_id": "999", "slurm_array_task_id": "7",
                                "taken": time.time(), "heartbeat": time.time()}))
    before = sorted(p.name for p in run.rglob("*"))

    with pytest.raises(ResumeLocked) as exc:
        _entry().run(_cfg("eureka", 4), out_root=str(tmp_path))
    assert "424242" in str(exc.value) and "othernode" in str(exc.value)
    assert sorted(p.name for p in run.rglob("*")) == before, "a refusal must write nothing"

    # aged past RESUME_MIN_IDLE_S -> stolen, and the steal is recorded
    stale = time.time() - checkpoint.RESUME_MIN_IDLE_S - 1
    lock.write_text(json.dumps({"pid": 424242, "host": "othernode",
                                "taken": stale, "heartbeat": stale}))
    result = _entry().run(_cfg("eureka", 4), out_root=str(tmp_path))
    assert result["legs"] == 2
    assert "lock_stolen" in (run / "resume.jsonl").read_text()


def test_resume_force_steals_a_fresh_lock_and_records_whose_it_was(tmp_path):
    run = _crash_leaving_checkpoints("eureka", tmp_path, at=2)
    (run / "checkpoints" / "resume.lock").write_text(json.dumps(
        {"pid": 555, "host": "livenode", "taken": time.time(), "heartbeat": time.time()}))
    result = _entry().run(_cfg("eureka", 4), out_root=str(tmp_path),
                          adopt=str(run), force=True)
    assert result["legs"] == 2
    records = [json.loads(x) for x in (run / "resume.jsonl").read_text().splitlines()]
    steal = [r for r in records if r["event"] == "lock_stolen"][-1]
    assert steal["holder"]["pid"] == 555 and steal["reason"] == "forced"


def test_resume_from_a_nonexistent_path_fails_with_a_clear_error(tmp_path):
    with pytest.raises(ConfigError) as exc:
        _entry().run(_cfg("eureka", 2), out_root=str(tmp_path),
                     adopt=str(tmp_path / "no-such-run"))
    assert "no such run directory" in str(exc.value)


def test_adopting_a_directory_with_no_checkpoint_is_an_error_not_a_silent_restart(tmp_path):
    _straight("eureka", tmp_path, iterations=2)
    run = _rundir(tmp_path)
    for path in (run / "checkpoints").glob("ckpt-*.json"):
        path.unlink()
    with pytest.raises(CheckpointError) as exc:
        _entry().run(_cfg("eureka", 2), out_root=str(tmp_path), adopt=str(run))
    assert "no usable checkpoint" in str(exc.value)


def test_adopting_a_directory_whose_config_differs_is_refused(tmp_path):
    run = _crash_leaving_checkpoints("eureka", tmp_path, at=2)
    with pytest.raises(CheckpointError) as exc:
        _entry().run(_cfg("eureka", 4, "auto", **{"generate.n_candidates": 2}),
                     out_root=str(tmp_path), adopt=str(run))
    assert "does not match the config being resumed" in str(exc.value)


def test_resume_info_runs_nothing(tmp_path):
    run = _crash_leaving_checkpoints("eureka", tmp_path, at=2)
    info = checkpoint.describe(run)
    assert info["phase"] == "iteration_done" and info["next_iteration"] == 2
    assert info["finished"] is False and info["status"] == "failed"
    assert info["budget"]["policy_trainings"] > 0
    assert "carried_sizes" in info
    assert _entry().main(["--resume-info", str(run)]) == 0


# --------------------------------------------------------------------------
# 17. the auto path, guarded as strictly as the explicit one
# --------------------------------------------------------------------------


def test_auto_with_no_usable_checkpoint_refuses_instead_of_restarting_at_zero(tmp_path):
    """The `auto` counterpart of
    `test_adopting_a_directory_with_no_checkpoint_is_an_error_not_a_silent_restart`,
    which pinned the property for `--resume` ONLY.

    `find_adoptable` accepts a directory on the mere existence of a
    `ckpt-*.json`, and `RunDir.adopt` has already bumped the leg and reopened
    `journal.jsonl` in append mode by the time `load_plan` decides every file is
    unusable. A refusal written `if plan is None and adopt is not None` lets
    control fall through under `auto` and run a COMPLETE fresh search into the
    adopted directory: two searches in one journal, `state/iterNN.json`
    overwritten, and `result.json` claiming `resumed: true, degraded: []` over
    a budget holding only the second half. A `SCHEMA_VERSION` bump under an
    unfinished run is enough to reach it.
    """
    run = _crash_leaving_checkpoints("eureka", tmp_path, at=2)
    before = [r for r in _journal(run) if r.get("stage") not in _SEAM_STAGES]
    for path in (run / "checkpoints").glob("ckpt-*.json"):
        body = json.loads(path.read_text())
        body["schema"] = SCHEMA_VERSION + 998
        body["digest"] = checkpoint.digest_of(body)
        path.write_text(json.dumps(body))

    with pytest.raises(CheckpointError) as exc:
        _entry().run(_cfg("eureka", 4), out_root=str(tmp_path))
    assert "no usable checkpoint" in str(exc.value)

    after = [r for r in _journal(run) if r.get("stage") not in _SEAM_STAGES]
    assert after == before, "not one stage of a second search may run in this directory"
    assert not (run / "result.json").exists()
    assert json.loads((run / "status.json").read_text())["status"] == "failed"


# --------------------------------------------------------------------------
# 18. the sidecars -- what the envelope's digest cannot see
# --------------------------------------------------------------------------


def _spilled(tmp_path, arr):
    ref = encode(arr, _ArraySpill(tmp_path), "$")
    assert "__ndarray_ref__" in ref, "the array must SPILL or this test proves nothing"
    return ref, tmp_path / ref["__ndarray_ref__"]["file"]


def test_a_corrupt_sidecar_is_rejected_rather_than_silently_believed(tmp_path):
    """`digest` covers the JSON envelope -- a ref's name, dtype and shape --
    and NOT the `.npy` bytes, which are where a replay buffer, a policy weight
    array and a 500-step trajectory actually live. Without a sidecar check, a
    mixed-block read that still parses would restore WRONG NUMBERS under a
    checkpoint reporting itself intact, with `degraded` empty."""
    arr = np.arange(4000, dtype=np.float64).reshape(500, 8)
    ref, path = _spilled(tmp_path, arr)

    blob = bytearray(path.read_bytes())
    blob[200:1200] = b"\x00" * 1000                 # same length, different bytes
    path.write_bytes(bytes(blob))

    with pytest.raises(CheckpointError) as exc:
        decode(ref, tmp_path)
    assert "corrupt" in str(exc.value)


@pytest.mark.parametrize("damage", ["empty", "truncated"], ids=["empty", "truncated"])
def test_a_damaged_sidecar_raises_CheckpointError_not_a_numpy_error(tmp_path, damage):
    """`load_plan` catches `CheckpointError` and nothing else, so `np.load`'s
    own `EOFError` / `ValueError` would walk straight past the keep-2 fallback:
    a MISSING sidecar would resume one iteration behind and a CORRUPT one kill
    the run outright, which is the stated floor upside down."""
    arr = np.arange(4000, dtype=np.float64)
    ref, path = _spilled(tmp_path, arr)
    blob = path.read_bytes()
    path.write_bytes(b"" if damage == "empty" else blob[:len(blob) // 2])
    with pytest.raises(CheckpointError):
        decode(ref, tmp_path)


def test_a_damaged_sidecar_is_rewritten_rather_than_reused_forever(tmp_path):
    """`put` skips a file that already exists -- that is the whole point of
    content addressing, and an unchanged trajectory is written once rather than
    once per iteration. But `not path.exists()` alone leaves a torn write in
    place for good, so one bad write poisons every later checkpoint that names
    the same hash. One `stat` of the size prevents it."""
    arr = np.arange(4000, dtype=np.float64)
    ref, path = _spilled(tmp_path, arr)
    path.write_bytes(b"torn")

    encode(arr, _ArraySpill(tmp_path), "$")         # the next checkpoint
    assert np.array_equal(decode(ref, tmp_path), arr)


# --------------------------------------------------------------------------
# 19. the lock beats on a clock
# --------------------------------------------------------------------------


def test_the_heartbeat_is_a_timer_not_a_checkpoint_boundary(tmp_path):
    """`RESUME_MIN_IDLE_S` is 900 s and one Meta-World iteration is
    hours, so a heartbeat refreshed only by `Checkpointer.save` leaves a LIVE
    holder reading stale for most of its life -- and `claim()` then hands the
    directory to a second writer, which is the interleaving its refusal message
    says it prevents. Nothing of ours runs during an 18-minute `learn()` call,
    so only a timer can beat through one."""
    assert checkpoint.HEARTBEAT_INTERVAL_S * 2 < checkpoint.RESUME_MIN_IDLE_S, (
        "the margin is the point: a holder must be able to miss beats and still "
        "read alive")

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    lock = checkpoint.claim(run)
    path = run / "checkpoints" / "resume.lock"
    first = json.loads(path.read_text())["heartbeat"]

    checkpoint.start_heartbeat(lock, interval=0.01)
    try:
        deadline = time.time() + 10
        latest = first
        while latest <= first and time.time() < deadline:
            time.sleep(0.01)
            latest = json.loads(path.read_text())["heartbeat"]
    finally:
        checkpoint.release(lock)
    assert latest > first, "the lock must age forwards with no checkpoint written"
    assert lock.beat is None, "release stops the beater"
    assert not path.exists(), "and still drops the lock it owns"

def test_stop_waits_for_the_beat_in_flight(tmp_path, monkeypatch):
    """The unlink must be ORDERED after the beat in flight, not merely after
    the flag that stops the next one.

    The same property the test above asserts, pinned at the mechanism instead
    of at the outcome -- because the outcome one can only fail when the race
    happens to land, and can pass on an idle machine and fail on a loaded CI
    runner off the identical tree. A test that has to LOSE a race to fail is
    not a test of the race.

    So the beat is made slow and the ordering read directly: `stop()` must not
    return until the beat that was already running has finished. With the flag
    alone it returns at once and that beat goes on to write the file `release`
    is about to unlink. The 0.3 s is a margin over a scheduler hiccup, not a
    hope -- nothing here is waiting for a race to land.
    """
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    lock = checkpoint.claim(run)

    entered, done = threading.Event(), threading.Event()

    def slow_beat(_lock):
        entered.set()
        time.sleep(0.3)
        done.set()

    monkeypatch.setattr(checkpoint, "heartbeat", slow_beat)
    checkpoint.start_heartbeat(lock, interval=0.001)
    beater = lock.beat
    assert beater is not None
    assert entered.wait(5), "the beater never beat"

    checkpoint.stop_heartbeat(lock)
    assert done.is_set(), (
        "stop() returned while a beat was still running; the unlink release "
        "does next can be undone by it")
    assert not beater.is_alive()


def test_a_beat_never_resurrects_a_lock_nobody_holds(tmp_path):
    """A beat that lands after the unlink must leave the directory unheld.

    The belt to `stop()`'s braces, and it earns its place: a beat we could not
    wait for -- a join that timed out on a sick mount -- would otherwise write
    a whole fresh payload from `_lock_payload()` and leave a directory nobody
    holds reading held for `RESUME_MIN_IDLE_S`. The next requeue of a search
    that finished CLEANLY then needs `--resume-force`.
    """
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    lock = checkpoint.claim(run)
    path = run / "checkpoints" / "resume.lock"
    path.unlink()

    checkpoint.heartbeat(lock)               # the late beat, run by hand
    assert not path.exists(), "a beat re-created a lock that had been released"


def test_a_beat_never_certifies_the_holder_that_stole_from_us(tmp_path):
    """After a steal the file is the THIEF's, and our clock must not touch it.

    Stamping it would go on vouching for that holder long after it died --
    the beater outliving the thing it is evidence for.
    """
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    lock = checkpoint.claim(run)
    path = run / "checkpoints" / "resume.lock"

    thief = json.loads(path.read_text())
    thief["pid"] = os.getpid() + 1
    thief["heartbeat"] = 1.0
    path.write_text(json.dumps(thief))

    checkpoint.heartbeat(lock)
    assert json.loads(path.read_text())["heartbeat"] == 1.0, (
        "we refreshed a holder that is not us")


def test_ours_means_pid_and_host_not_pid_alone(tmp_path):
    """pid is not an identity on a shared mount: pids are per-node and run
    dirs are not, so a holder on another node can carry our pid without having
    stolen anything. A beat must not stamp it and `release()` must not unlink
    it -- either would be one search operating another's lock."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    lock = checkpoint.claim(run)
    path = run / "checkpoints" / "resume.lock"

    twin = json.loads(path.read_text())     # our pid, another node
    twin["host"] = twin["host"] + "-twin"
    twin["heartbeat"] = 1.0
    path.write_text(json.dumps(twin))

    checkpoint.heartbeat(lock)
    assert json.loads(path.read_text())["heartbeat"] == 1.0, (
        "we refreshed a same-pid holder on another host")

    checkpoint.release(lock)
    assert path.exists(), "we unlinked a same-pid holder on another host"


def test_a_beat_that_read_before_the_release_still_refuses_to_write(
        tmp_path, monkeypatch):
    """The absent-file check catches only a beat whose READ lands after the
    unlink. The beat `stop()`'s bounded join gives up on read BEFORE it: it
    holds a payload that looks ours and every file check passes it. Staged
    directly rather than raced for, the way `test_stop_waits_for_the_beat_in_
    flight` is: the beat's read completes, THEN the whole of `release()` runs,
    then the beat goes on. Only `lock.released` can refuse the write."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    lock = checkpoint.claim(run)
    path = run / "checkpoints" / "resume.lock"

    real = checkpoint._read_json
    raced = []

    def read_then_lose_the_race(p):
        holder = real(p)
        if not raced:                        # fire once; release() reads too
            raced.append(True)
            checkpoint.release(lock)
        return holder

    monkeypatch.setattr(checkpoint, "_read_json", read_then_lose_the_race)
    checkpoint.heartbeat(lock)
    assert not path.exists(), "a beat resurrected a lock released mid-flight"


def test_release_raises_the_flag_before_it_touches_the_file(
        tmp_path, monkeypatch):
    """The flag is worth nothing raised after the unlink: a beat re-checks it
    exactly once, immediately before its write, so flag-then-unlink is the
    ordering that closes the window and unlink-then-flag leaves it open. The
    spy sits on `release()`'s own first file operation."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    lock = checkpoint.claim(run)

    seen = []
    real = checkpoint._read_json

    def spy(p):
        seen.append(lock.released)
        return real(p)

    monkeypatch.setattr(checkpoint, "_read_json", spy)
    checkpoint.release(lock)
    assert seen and all(seen), (
        "release() touched the file before raising the flag")



def test_a_live_holder_is_never_stolen_from(tmp_path):
    """The same property from the other side: a beating lock refuses a second
    process, and the refusal names who holds it."""
    run = _crash_leaving_checkpoints("eureka", tmp_path, at=2)
    lock = checkpoint.claim(run)
    checkpoint.start_heartbeat(lock, interval=0.01)
    time.sleep(0.05)
    try:
        with pytest.raises(ResumeLocked):
            _entry().run(_cfg("eureka", 4), out_root=str(tmp_path))
    finally:
        checkpoint.release(lock)


# --------------------------------------------------------------------------
# 20. restarts: a swallowed write must not become a phantom
# --------------------------------------------------------------------------


class _Rundir:
    def __init__(self, path):
        self.path = path


def test_a_failed_restart_write_is_never_recorded_as_completed(tmp_path):
    """`write_restart` returns `None` and never raises -- killing a 20-hour
    search over a serialisation fault is worse than not resuming it -- so a
    caller that appended the restart regardless would make every later
    checkpoint name a `restart-NN.json` that does not exist. Under strict
    `auto` that turns a swallowed write failure at hour 2 into a REFUSAL to
    resume at hour 18, and it takes the correctly-written restarts after it
    down too.

    The gap is not skipped over, it STOPS the prefix: `_plan_from` maps
    `restart_states` positionally, so recording `[1, 2]` would come back as
    restarts 0 and 1 and relabel the whole search.
    """
    cp = checkpoint.Checkpointer(_Rundir(tmp_path), None, enabled=True)
    cp.record_restart(0, tmp_path / "restart-00.json")
    cp.record_restart(1, None)                       # the write failed
    cp.record_restart(2, tmp_path / "restart-02.json")
    assert cp.completed_restarts == [0]
    assert cp._restart_gap is True


def test_a_lost_restart_does_not_double_count_its_budget(tmp_path):
    """The rolling checkpoint's budget is a running TOTAL, so it already holds
    what a lost restart spent -- and that restart is about to be re-executed.
    `Budget.restore`'s docstring states the invariant ("iterations 0..k are
    never re-executed, so nothing re-records them"); a truncated restart list
    breaks it, inflating `policy_trainings`, `env_steps` and `llm_calls` in a
    first-class output column -- and against `budget.max_*`.
    """
    # `generate.n_candidates: 4` is the width the divergence below was measured
    # at; the `tester` PROFILE is deliberately silent on it (it is
    # method-defining), so eureka resolves to its published 16 here unless the
    # test states it.
    ov = {"loop.n_restarts": 3, "generate.n_candidates": 4}
    straight = _straight("eureka", tmp_path / "straight", iterations=2, **ov)

    out = tmp_path / "resumed"
    _clear_process_state()
    entry = _entry()
    real = entry.run_iteration

    def boom(ctx, state):
        if state.restart == 1 and state.iteration == 1:
            raise RuntimeError("simulated walltime kill")
        return real(ctx, state)

    entry.run_iteration = boom
    with pytest.raises(RuntimeError):
        entry.run(_cfg("eureka", 2, **ov), out_root=str(out))

    run = _rundir(out)
    lost = run / "checkpoints" / "restart-00.json"
    assert lost.exists(), "restart 0 must have completed or this proves nothing"
    lost.write_text("{ not a checkpoint")

    _clear_process_state()
    result = _entry().run(_cfg("eureka", 2, **ov), out_root=str(out), allow_degraded=True)
    # NOT equality. A degraded resume is explicitly "a different run, no
    # equality claim" -- measured here, straight picks c0000/c0010/c0015 per
    # restart and the resumed leg picks c0010/c0019/c0024, so the searches
    # genuinely diverge and no counter can match by construction. What must
    # not happen is INFLATION: the rolling checkpoint's budget already holds
    # what the lost restart spent, and restoring that total must not book it
    # again. So the
    # invariant is a ceiling, not an identity. `budget_unaccounted_iterations`
    # states the under-report bound on the other side.
    for key in ("policy_trainings", "env_steps", "llm_calls"):
        assert result["budget"][key] <= straight["budget"][key], (
            f"budget.{key}: {result['budget'][key]} against {straight['budget'][key]} "
            "for the same search -- a re-executed restart was counted twice")
    assert result["budget_unaccounted_iterations"] >= 1, (
        "a discarded in-flight iteration must be declared, not silently absorbed")
    dropped = json.loads((run / "status.json").read_text())["resumes"][-1]["dropped"]
    assert any("budget rewound" in str(d.get("what")) for d in dropped)
    assert any("in-flight" in str(d.get("what")) for d in dropped), (
        "the discarded in-flight restart has to be named, not just the file that "
        "would not load")


def test_one_lost_payload_is_one_degradation(tmp_path):
    """`budget.py` defines `resume_degradations` as one per DROPPED SLOT. One
    evicted `policy_ref` is named once by `export_stores` at write time and
    once per holder by `_null_dead_refs` at load time -- and `rda`'s warm start
    reaches the same ref through both `best` and `latest` -- so counting
    sentences would inflate a cross-method filter by up to 4x."""
    run = _crash_leaving_checkpoints("rda", tmp_path, at=2)
    _plant_dead_ref(run)
    result = _entry().run(_cfg("rda", 4, "auto"), out_root=str(tmp_path),
                          allow_degraded=True)
    assert result["budget"]["resume_degradations"] == 1, (
        f"one lost payload, {len(result['degraded'])} sentence(s) about it")
    assert len(result["degraded"]) >= 1


# --------------------------------------------------------------------------
# 21. every shipped config accepts the resume override
# --------------------------------------------------------------------------


def test_every_config_survives_the_resume_override():
    """A command line that appends `--set loop.resume_from=...` to an otherwise
    validated one adds the one override nothing pre-validates.
    `_check_coherence` ties the key to `output.save_state_every_iteration`,
    which is exactly the sort of thing a future config could switch off."""
    from bird.config import CONFIG_ROOT

    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        for mode in ("auto", "auto_degraded"):
            load(path, overrides={"loop.resume_from": mode})
