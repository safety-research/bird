"""`--full-iterations` -- continuing a search that stopped SHORT of its budget.

`loop.termination: fitness_plateau` ends the outer loop early, and the run is
then finished: `result.json` is written, `status.json` says `ok`, and
`find_adoptable` skips it by design ("that search FINISHED"). So a run stopped
by a plateau rule cannot be extended to `loop.n_iterations` afterwards, and the
only route was re-running it from zero.

This flag is the operator action that closes that gap. It is a FLAG, not a
config key, for the reason `--resume-degraded` is: overriding
`loop.termination` instead would change the resolved config, and then
`_assert_same_config` refuses the adopt -- correctly, because the directory is
named `<name>-H-<stamp>` for a hash its contents would no longer produce.

THE FAILURE THIS FILE EXISTS FOR. `plan.next_iteration` is 0 on a finished
run: retention keeps only the newest two checkpoints, and once a loop ends
those are `restart_done` and `loop_done`, neither of which carries an iteration
cursor. Continuing from it runs a SECOND search into the same directory --
journal iterations come out `[0,1,2,3, 0,1,2,3,4]` -- which is precisely the
"two searches concatenated in one set of artifacts" `bird/checkpoint.py`
refuses elsewhere, and it passes a naive `sorted(set(...))` assertion. Hence
`_iterations_in_order` below keeps duplicates, and the budget is checked too: a
re-run inflates it.
"""
import dataclasses
import json
from pathlib import Path

import pytest

from conftest import TESTER_SEARCH_CAP, REPO  # noqa: F401  (path bootstrap)
from test_parallelism import _entry
from test_resume import _cfg, _rundir

from bird import checkpoint
from bird.checkpoint import CheckpointError

PLATEAU = {"loop.termination": "fitness_plateau",
           "loop.termination_cfg.patience": 1,
           "loop.termination_cfg.min_delta": 0.01}


def _iterations_in_order(run: Path):
    """Every iteration the loop STARTED, duplicates kept -- a re-run shows up
    here as a repeat and would be invisible to a set-based assertion."""
    out = []
    for line in (run / "journal.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("stage") == "stage_begin" and e.get("of") == "generate":
            out.append(e.get("iteration"))
    return out


def _budget_trainings(run: Path) -> int:
    return int(json.loads((run / "budget.json").read_text())["policy_trainings"])


def _plateau_stopped_run(out: Path, iterations: int = 5, seed: int = 0):
    cfg = _cfg("eureka", iterations=iterations, mode="auto", seed=seed, **PLATEAU)
    _entry().run(cfg, out_root=str(out))
    run = _rundir(out)
    ran = _iterations_in_order(run)
    assert len(ran) < iterations, (
        f"the plateau rule did not stop this run short ({ran}); the test needs an "
        "early-stopped run to extend")
    return run, ran


def test_full_iterations_continues_and_does_not_rerun(tmp_path):
    """The flag reaches `loop.n_iterations` by ADDING iterations, not by
    repeating the ones already paid for."""
    run, before = _plateau_stopped_run(tmp_path / "out")
    trainings_before = _budget_trainings(run)

    _entry().run(_cfg("eureka", iterations=5, mode="auto", **PLATEAU),
                 out_root=str(tmp_path / "out"), adopt=str(run),
                 full_iterations=True)

    after = _iterations_in_order(run)
    assert after[:len(before)] == before, (
        f"the iterations already run were not preserved: {before} -> {after}")
    assert after == sorted(set(after)), (
        f"an iteration was re-executed: {after}. Retention leaves only "
        "`restart_done`/`loop_done` checkpoints on a finished run, both with "
        "next_iteration=0; the cursor must come from the restart state instead")
    assert len(after) == 5, f"did not reach loop.n_iterations: {after}"
    # a re-run would roughly double this; one extra iteration is a small delta
    grew = _budget_trainings(run) - trainings_before
    assert 0 < grew <= trainings_before, (
        f"budget grew by {grew} on top of {trainings_before}: that is a re-run, "
        "not a continuation")


def test_full_iterations_records_the_extension(tmp_path):
    """An extension that leaves no trace is indistinguishable from a search that
    simply ran longer under its own rule."""
    run, before = _plateau_stopped_run(tmp_path / "out")
    _entry().run(_cfg("eureka", iterations=5, mode="auto", **PLATEAU),
                 out_root=str(tmp_path / "out"), adopt=str(run),
                 full_iterations=True)

    recs = [json.loads(l) for l in (run / "resume.jsonl").read_text().splitlines()
            if l.strip() and "extended_full_iterations" in l]
    assert recs, "resume.jsonl does not record the extension"
    assert recs[-1]["from_iteration"] == len(before)
    assert recs[-1]["to_iteration"] == 5
    assert recs[-1]["suppressed_termination"] == "fitness_plateau"
    # the superseded outcome is kept, not silently overwritten
    assert (run / "result.pre-extend.json").exists(), \
        "the pre-extension result.json was overwritten instead of preserved"


def test_full_iterations_refuses_a_run_that_already_used_its_budget(tmp_path):
    """Nothing to extend is an error, not a no-op that quietly re-runs."""
    out = tmp_path / "out"
    cfg = _cfg("eureka", iterations=2, mode="auto")      # fixed_iterations: runs both
    _entry().run(cfg, out_root=str(out))
    run = _rundir(out)
    assert len(_iterations_in_order(run)) == 2
    with pytest.raises(CheckpointError, match="nothing to extend"):
        _entry().run(_cfg("eureka", iterations=2, mode="auto"), out_root=str(out),
                     adopt=str(run), full_iterations=True)


def test_full_iterations_is_named_in_the_result_and_the_budget(tmp_path):
    """`resume.jsonl` says the extension happened; `result.json` and
    `budget.json` say THIS NUMBER was produced under it.

    The convention is the repo's own and `--resume-degraded`'s help states it:
    the loss is "counted in budget.resume_degradations and named in
    result.json". Without both, a finished extended run reads as a config that
    says stop at plateau, a journal showing the full budget, and
    `resumed: true` -- which is also exactly what an ordinary resumed run says.
    An extended run and one that reached `loop.n_iterations` under its own rule
    are different treatments; pooling them silently mixes two treatments, and
    `config.resolved.yaml` actively points the wrong way here.
    """
    run, _before = _plateau_stopped_run(tmp_path / "out")
    # The unextended run is the control: the markers must SEPARATE the two, and
    # a key that is always true separates nothing.
    pre = json.loads((run / "result.json").read_text())
    assert pre["extended_full_iterations"] is False
    assert pre["suppressed_termination"] == ""
    assert json.loads((run / "budget.json").read_text())["resume_extensions"] == 0

    _entry().run(_cfg("eureka", iterations=5, mode="auto", **PLATEAU),
                 out_root=str(tmp_path / "out"), adopt=str(run),
                 full_iterations=True)

    result = json.loads((run / "result.json").read_text())
    assert result["extended_full_iterations"] is True, (
        "result.json does not say this outcome was produced under an extension; "
        "the config it sits beside says loop.termination: fitness_plateau")
    assert result["suppressed_termination"] == "fitness_plateau", (
        "the overridden rule is not named, so nothing in the result says which "
        "rule the run was allowed to ignore")
    budget = json.loads((run / "budget.json").read_text())
    assert budget["resume_extensions"] == 1, (
        "the iterations an extension adds are real spend that the config's own "
        "termination rule declined to authorise; the cost column must carry it")


def test_full_iterations_refuses_when_an_earlier_restart_stopped_short(tmp_path):
    """Only the LAST restart can be extended, so a search with more than one
    early-stopped restart is refused rather than partly extended.

    `_execute` seeds `states` from `plan.restart_states` and loops from
    `len(states)`, so restarts before the re-entered one are replayed from their
    stored end states and never re-enter their loops -- they cannot, because a
    completed restart's write-once state carries no rng and retention has long
    since dropped their iteration checkpoints. Extending anyway would put one
    full restart beside short ones in a single `result.json` with nothing saying
    the extension was partial: `configs/methods/eureka.yaml` is `n_restarts: 5`, so the
    operator would ask for the full budget, be told it was delivered, and get a
    fifth of it.

    The refusal must also leave the directory ALONE -- it happens before the
    `result.json` copy, so a refused extension cannot half-rewrite a finished
    run's artifacts.
    """
    out = tmp_path / "out"
    cfg = _cfg("eureka", iterations=5, mode="auto", **{**PLATEAU,
                                                      "loop.n_restarts": 2})
    _entry().run(cfg, out_root=str(out))
    run = _rundir(out)
    per_restart = json.loads((run / "result.json").read_text())["per_restart"]
    assert len(per_restart) == 2, f"expected two restarts, got {per_restart}"
    assert all(r["iteration"] + 1 < 5 for r in per_restart), (
        f"both restarts must stop short for this test to mean anything: "
        f"{[r['iteration'] for r in per_restart]}")

    with pytest.raises(CheckpointError, match="cannot be re-entered"):
        _entry().run(_cfg("eureka", iterations=5, mode="auto",
                          **{**PLATEAU, "loop.n_restarts": 2}),
                     out_root=str(out), adopt=str(run), full_iterations=True)

    assert not (run / "result.pre-extend.json").exists(), (
        "a refused extension moved the prior result aside anyway")
    assert json.loads((run / "result.json").read_text())["per_restart"] == per_restart

    # AND THE CASE THE OTHER GUARD CANNOT SEE. On the cleanly finished run above,
    # `plan.restart != len(states)` would have refused anyway -- `loop_done`
    # records `restart: 0` against two restart states -- so the assertion above
    # passes even with the short-restart check deleted, as long as the message
    # matches. It is exercised properly only where that cursor guard agrees:
    # when the newest checkpoint is `restart_done` rather than `loop_done`, which
    # is a leg killed between the two writes, `plan.restart` IS `len(states)` and
    # the partial extension goes through silently. Constructed directly rather
    # than by racing a checkpoint write, because the point is the guard and not
    # the ordering that reaches it.
    plan = checkpoint.load_plan(run)
    assert plan is not None and len(plan.restart_states) == 2
    faked = dataclasses.replace(plan, restart=1)
    with pytest.raises(CheckpointError, match="cannot be re-entered"):
        _entry()._reenter_for_full_iterations(
            faked, _cfg("eureka", iterations=5, mode="auto",
                        **{**PLATEAU, "loop.n_restarts": 2}))


def test_a_failed_extension_leaves_the_prior_result_in_place(tmp_path):
    """If the extension leg dies, the run must still HAVE a `result.json`.

    The prior outcome is renamed out of the way before `_execute` runs, so
    without a restore a crash in the extension would leave `status.json: failed`
    beside no result at all, with the real one under a name nothing looks for --
    a reader of the run directory would see a run that produced nothing, i.e.
    the failure of the extension would present as the loss of the search it was
    extending.
    """
    run, before = _plateau_stopped_run(tmp_path / "out")
    prior = json.loads((run / "result.json").read_text())

    entry = _entry()
    real = entry.run_iteration

    def boom(ctx, state):
        if state.iteration >= len(before):
            raise RuntimeError("simulated walltime kill during the extension")
        return real(ctx, state)

    entry.run_iteration = boom
    with pytest.raises(BaseException):
        entry.run(_cfg("eureka", iterations=5, mode="auto", **PLATEAU),
                  out_root=str(tmp_path / "out"), adopt=str(run),
                  full_iterations=True)

    # NOT `== "failed"`, deliberately: see
    # `_restore_status_after_failed_extension`. The search this run records
    # COMPLETED; only the attempt to extend it died. Leaving the directory marked
    # `failed` would remove it from every tool that selects `status == "ok"`, so
    # a live result set would silently shrink. The failure is still visible, in `status.json["extension"]`, in
    # `status.extend-failed.json`, and in `resume.jsonl`.
    st = json.loads((run / "status.json").read_text())
    assert st["status"] == "ok"
    assert st["extension"]["outcome"] == "failed"
    assert (run / "result.json").exists(), (
        "the extension died and took the run's only result with it")
    assert json.loads((run / "result.json").read_text()) == prior, (
        "result.json is neither the prior outcome nor a new one")
    assert (run / "result.pre-extend.json").exists()


_EXTENSION_SEED = 4


def test_an_extension_reproduces_the_search_that_never_stopped_early(tmp_path):
    """An extended run is the continuation the search would have had, not merely
    a longer one.

    `_EXTENSION_SEED` is MEASURED: the mock seeds every sample from its prompt,
    and at seed 0 the eureka tester search scores only 0.0 and 1.0, which the
    non-degeneracy guard below rightly refuses as evidence. Seed 4 is the first
    whose search stops short under the plateau rule AND scores at least three
    distinct real fitness values (0.0, 0.125, 0.375, 0.75, 1.0, besides the
    invalid-candidate sentinel). A prompt edit re-rolls this.

    BOUNDED EXACTLY AS THE RESUME GUARANTEE IS BOUNDED: tester tier, mock LLM, mock
    learner, `auto` and not `auto_degraded`, one process. Nothing here claims it
    of a live provider (two FRESH runs already differ there) or across a process
    boundary under `train.backend: sb3`.

    Without this the natural reader assumption is unpinned in either direction:
    the other tests in this file claim only "adds iterations, does not repeat
    them", which a continuation from the wrong rng would satisfy while producing
    a different search.
    """
    seed = _EXTENSION_SEED
    extended, before = _plateau_stopped_run(tmp_path / "plateau", seed=seed)
    _entry().run(_cfg("eureka", iterations=5, mode="auto", seed=seed, **PLATEAU),
                 out_root=str(tmp_path / "plateau"), adopt=str(extended),
                 full_iterations=True)

    # The counterfactual: the same seed and the same budget, never told to stop.
    _entry().run(_cfg("eureka", iterations=5, mode="auto", seed=seed),
                 out_root=str(tmp_path / "straight"))
    straight = _rundir(tmp_path / "straight")

    def scored(run: Path):
        out = []
        for f in sorted(run.glob("candidates/*/report.json")):
            r = json.loads(f.read_text())
            out.append((r["iteration"], r["cand_id"], r["fitness"]))
        return sorted(out)

    a, b = scored(extended), scored(straight)
    assert a == b, (
        "the extension did not reproduce the uninterrupted search; the "
        "continuation is reading the wrong rng or state. "
        f"extended={len(a)} straight={len(b)} first difference="
        f"{next((x for x, y in zip(a, b) if x != y), None)}")
    # Not a degenerate comparison: identical empty lists, or one fitness value
    # repeated, would satisfy the assertion above while proving nothing.
    assert len(a) >= 5 and len({f for _i, _c, f in a}) >= 3, (
        f"too few distinct results to be evidence: {a}")
    assert len(before) < 5, "the plateau run did not stop short"


# --------------------------------------------------------------------------
# a FAILED extension must not downgrade the search it was extending
# --------------------------------------------------------------------------

def test_a_crash_during_the_extension_leaves_the_finished_run_ok(tmp_path, monkeypatch):
    """`--full-iterations` adopts a run that ALREADY FINISHED, and adoption
    rewrites `status.json` to "running". If the extension then raises, `run`'s
    handler writes "failed" -- so an API outage, an OOM or the walltime
    silently converts a successful search into a failed one, though nothing
    about that search changed.
    """
    run, before = _plateau_stopped_run(tmp_path / "out")
    assert json.loads((run / "status.json").read_text())["status"] == "ok"

    entry = _entry()
    boom = RuntimeError("simulated crash inside the extension")

    def explode(*a, **k):
        raise boom
    monkeypatch.setattr(entry, "run_search", explode)

    with pytest.raises(BaseException):
        entry.run(_cfg("eureka", iterations=5, mode="auto", **PLATEAU),
                  out_root=str(tmp_path / "out"), adopt=str(run),
                  full_iterations=True)

    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "ok", (
        "a failed EXTENSION downgraded a search that had already finished; its "
        "terminal status is the record of that search, not of the attempt")
    # the attempt is preserved rather than hidden
    assert (run / "status.extend-failed.json").exists()
    assert json.loads((run / "status.extend-failed.json").read_text())["status"] == "failed"
    recs = [json.loads(l) for l in (run / "resume.jsonl").read_text().splitlines()
            if l.strip() and "extension_failed" in l]
    assert recs and recs[-1]["restored_status"] == "ok"
    # and the run is still recognisably short, so it can be retried
    assert len(_iterations_in_order(run)) == len(before)
