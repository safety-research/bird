"""`final_retrain` retrains the winner on seeds the search never used.

The phase exists to remove post-selection bias (LIMEN retrains the winner
"from scratch over 10 fresh seeds" for exactly that reason; Eureka evaluates
"with distinct seeds"), and it reports `post_selection_gap` -- retrained minus
selected -- as the measurement of that bias. Every backend seeds a training
from `training._seed_base(ctx, state, candidate, phase) + seed_i * SEED_STRIDE`, keyed on
`(cfg.seed, state.restart, state.iteration, cand_id)`, and `run_final_retrain`
hands it the END-OF-SEARCH state, whose `iteration` is still the last one run.
So whenever the winner was trained in the last iteration -- always under
`select.final_artifact: chain_end`, and under `global_best` whenever the best
came from the final round -- the retrain's first `train.seeds_per_candidate`
seeds would be the search's own seeds for that candidate, bit for bit. Measured
on the eureka tester point with `chain_end` and `n_seeds: 1` without the salt
below: identical seed, identical fitness, gap exactly 0.0 -- the number the
phase exists to report, biased by the very effect it exists to remove.

The mechanism: `run_final_retrain` hands the backend
`seed_phase="final_retrain"` and `training._seed_base` salts that stream off
the winner's own search stream (`tests/test_final_retrain_seeds.py` pins the
arithmetic). These tests check the CLAIM on whole runs, through the artifact:
`final_retrain.json` records `seeds`, `search_seeds` and `seed_overlap`, and
the seeds must be disjoint from the winner's search seeds as read off
`candidates/*/train_result.json` -- under `chain_end`, where the collision
would be certain, and under `global_best` with a one-iteration search, where it
would be too.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from test_parallelism import _run, scrub_volatile

from bird.components.phases import _retrain_config
from bird.config import load
from bird.types import Candidate

_TRAIN_RESULT = re.compile(r"^candidates/iter(\d+)_(c\d+)/train_result\.json$")


def _search_seeds(fp: dict) -> dict:
    """`{cand_id: {seed, ...}}` for every candidate the SEARCH trained, read off
    the same artifact a reader of the run would use."""
    out: dict = {}
    for rel, text in fp.items():
        m = _TRAIN_RESULT.match(rel)
        if m is None:
            continue
        for row in json.loads(text).get("seed_metrics") or []:
            if isinstance(row, dict) and "seed" in row:
                out.setdefault(m.group(2), set()).add(int(row["seed"]))
    return out


def _retrain(tmp_path, **overrides):
    fp = _run("eureka", tmp_path, **{"final_retrain.enabled": True,
                                     "final_retrain.n_seeds": 3, **overrides})
    assert "phases/final_retrain.json" in fp, "the retrain never ran; the test is vacuous"
    fr = json.loads(fp["phases/final_retrain.json"])
    assert fr["retrained_fitness"] is not None and not fr["error"], fr
    return fp, fr


@pytest.mark.parametrize("overrides", [
    # The always-colliding case: the chain's end IS the last iteration.
    {"select.final_artifact": "chain_end"},
    # The data-dependent case, made certain: with one iteration the global best
    # was necessarily trained in the last one.
    {"select.final_artifact": "global_best", "loop.n_iterations": 1},
], ids=["chain_end", "global_best_from_the_last_iteration"])
def test_the_retrain_seeds_are_disjoint_from_the_winners_search_seeds(tmp_path, overrides):
    fp, fr = _retrain(tmp_path, **overrides)
    search = _search_seeds(fp)
    winner = fr["cand_id"]
    assert winner in search and search[winner], (
        f"the winner {winner} has no search seeds in the artifact; the test cannot "
        f"say anything about a collision")
    seeds = [int(s) for s in fr["seeds"]]
    assert len(seeds) == fr["n_seeds"] == 3, fr
    assert len(set(seeds)) == 3, f"the retrain reused a seed within itself: {seeds}"
    replayed = set(seeds) & search[winner]
    assert not replayed, (
        f"retrain seed(s) {sorted(replayed)} are the search's own seeds for {winner}: "
        f"the 'fresh' retrain replays the run whose argmax selected it")
    # ... and the artifact says the same about itself, off the same rows.
    assert set(fr["search_seeds"]) == search[winner], (fr["search_seeds"], search[winner])
    assert fr["seed_overlap"] == [] and fr["seed_phase"] == "final_retrain", fr


def test_the_retrain_is_never_pruned_while_the_search_still_is(tmp_path, monkeypatch):
    """A pruner is a SEARCH economy -- it stops a candidate that is losing to its
    cohort -- and the retrain has no cohort: one reward, `n_seeds` times, for
    `env_steps` each, to report a number. A `_retrain_config` that inherited
    `train.pruning` would, under `median_stop`, run the "from scratch for
    `env_steps`" retrain for a fraction of its stated steps while
    `final_retrain.json` said `env_steps_per_seed` at the configured count.

    Both halves are asserted: the retrain's backend call sees `pruning: none`
    and cuts no seed, AND the search under the same config still prunes -- so
    pruning is not disabled globally."""
    from bird.components import phases

    seen: list = []
    real_get = phases.registry_get

    def spying_get(kind, name):
        fn = real_get(kind, name)
        if kind != "train_backend":
            return fn

        def wrapped(ctx, state, candidate, n_seeds=1, **kw):
            res = fn(ctx, state, candidate, n_seeds=n_seeds, **kw)
            seen.append({"pruning": ctx.cfg.get("train.pruning"), "pruned": res.pruned,
                         "rounds": [m.get("pruned_at_round") for m in res.seed_metrics]})
            return res
        return wrapped

    # `phases.registry_get` resolves the RETRAIN's backend; the search's train
    # stage resolves its own through `registry.get`, so `seen` is the retrain.
    monkeypatch.setattr(phases, "registry_get", spying_get)
    fp = _run("eureka", tmp_path, **{"train.pruning": "median_stop", "train.env_steps": 4000,
                                     "final_retrain.enabled": True, "final_retrain.n_seeds": 3})
    search_pruned = [rel for rel, text in fp.items()
                     if _TRAIN_RESULT.match(rel) and json.loads(text).get("pruned")]
    assert search_pruned, ("median_stop never fired in the SEARCH under this config; "
                           "the test cannot show the retrain is exempt from it")
    (call,) = seen
    assert call["pruning"] == "none", f"the retrain inherited train.pruning={call['pruning']!r}"
    assert call["pruned"] is False and all(r is None for r in call["rounds"]), call
    fr = json.loads(fp["phases/final_retrain.json"])
    assert fr["train_pruning"] == "none" and fr["pruned"] is False, fr


# --------------------------------------------------------------------------
# "from scratch" strips the selection's products, not a fixed prior
# --------------------------------------------------------------------------

_CARRY = ["best_reward", "dialogue", "policy_checkpoint", "replay_buffer"]


@pytest.mark.parametrize("init,anchor,elite,expect_init,expect_anchor", [
    ("from_scratch", "none", "none", "from_scratch", "none"),
    ("warm_start_from_best", "kl_reward", "none", "from_scratch", "none"),
    ("warm_start_from_best", "none", "l2_params", "from_scratch", "none"),   # lares' shape
    ("secondary_replay_buffer", "none", "none", "from_scratch", "none"),     # gt's shape
    ("bc_prior", "kl_clone", "none", "bc_prior", "kl_clone"),
    ("bc_prior_then_warm_start", "kl_clone", "none", "bc_prior", "kl_clone"),
], ids=["scratch", "warm+kl_reward", "warm+l2", "replay", "bc_prior+kl_clone", "chain+kl_clone"])
def test_the_retrain_keeps_a_fixed_prior_and_strips_every_product_of_the_selection(
        init, anchor, elite, expect_init, expect_anchor):
    """`warm_start_from_best` / `secondary_replay_buffer` hand the retrain the
    winner's own checkpoint or buffer -- the bias the phase removes -- and go;
    `bc_prior` is the task's scripted policy cloned once per run and is kept
    (`bc_prior_then_warm_start` keeps its round-0 half). The anchor lives
    exactly as long as the prior, the elite constraint never -- and the result
    is a config `validate` accepts, which `anchor + from_scratch` is not."""
    from bird.config import validate

    cfg = load("eureka", profile="tester", overrides={
        "problem.env_id": "pendulum", "loop.carry": _CARRY, "train.init": init,
        "train.anchor.kind": anchor, "train.elite_constraint.kind": elite,
        # a replay buffer needs an off-policy learner to load into (coherence
        # rule); `kl_clone` needs ppo -- so the algorithm follows the case
        "train.algorithm": "sac" if init == "secondary_replay_buffer" else "ppo"})
    rc = _retrain_config(cfg, 400)
    assert rc["train.init"] == expect_init
    assert rc["train.anchor.kind"] == expect_anchor
    assert rc["train.elite_constraint.kind"] == "none"
    assert rc["train.pruning"] == "none" and rc["train.secondary_buffer.ratio"] == 0.0
    validate(rc)  # raises ConfigError on an incoherent pairing
    # the run's own config is untouched
    assert cfg["train.init"] == init and cfg["train.anchor.kind"] == anchor


def test_every_shipped_configs_retrain_config_is_itself_a_valid_config():
    """`_retrain_config` builds `Config(...)` directly, which never validates,
    so the pairings `_check_coherence` refuses at load could be constructed
    here unseen -- for every prior+anchor config. Every config
    `--validate-all` accepts must yield a retrain config it would accept too;
    the published from-scratch protocols (eureka, limen, gt) must still say
    `from_scratch`, and a prior+anchor config must keep its prior.

    No shipped config selects `train.init: bc_prior`, so the prior half is
    exercised on `hillclimb/v1` with the prior+anchor keys set by override
    (PPO, because the anchor is a PPO subclass)."""
    from bird.config import CONFIG_ROOT, validate

    def retrain_of(cfg):
        rc = _retrain_config(cfg, int(cfg.get("final_retrain.env_steps") or cfg["train.env_steps"]))
        validate(rc)
        return rc["train.init"], rc["train.anchor.kind"]

    paths = [p for p in sorted(CONFIG_ROOT.glob("*.yaml")) if not p.name.startswith("_")]
    for sub in ("methods", "hillclimb", "examples"):
        paths += sorted((CONFIG_ROOT / sub).glob("*.yaml"))
    assert paths, "no configs found; the test is vacuous"
    seen = {}
    for path in paths:
        seen[path.stem] = retrain_of(load(path))
    for stem in ("eureka", "limen", "gt"):
        assert seen[stem] == ("from_scratch", "none"), (stem, seen[stem])
    priors = {k: v for k, v in seen.items() if v[0] == "bc_prior"}
    assert all(v == ("bc_prior", "kl_clone") for v in priors.values()), priors
    prior_cfg = load("hillclimb/v1", overrides={
        "train.algorithm": "ppo", "train.init": "bc_prior",
        "train.anchor.kind": "kl_clone", "train.anchor.schedule": "adaptive"})
    assert retrain_of(prior_cfg) == ("bc_prior", "kl_clone"), (
        "a prior+anchor config lost its prior on the way into the retrain")


def test_a_bc_prior_retrain_starts_from_the_clone_and_keeps_its_anchor(tmp_path):
    """Real PPO. The search's learner is an anchored fine-tune of the clone
    (`warm_started_from == BC_PRIOR_REF`, `anchor.kind == kl_clone`); the
    retrain must be the SAME learner, on fresh seeds, and the artifact must
    say so as executed -- not `train_init: from_scratch` over an unprimed,
    unanchored PPO that only a log.warning ever mentioned. The clone is popped
    from the store first: a long search's FIFO store may have evicted it, and
    the retrain re-pins it from the RUN's config and seed, not the retrain's."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from bird import registry
    from bird.artifacts import RunDir
    from bird.budget import Budget
    from bird.components import training as T
    from bird.components.phases import run_final_retrain
    from bird.context import Context
    from bird.state import RunState
    from bird.types import CandidateReport

    registry.load_all()
    cfg = load("eureka", profile="tester", overrides={
        "seed": 0, "problem.env_id": "pendulum", "train.backend": "sb3", "train.algorithm": "ppo",
        "train.env_steps": 256,
        "train.hyperparameters": {"n_steps": 64, "batch_size": 32, "n_epochs": 1},
        "train.init": "bc_prior", "train.anchor.kind": "kl_clone", "train.anchor.warmup_steps": 0,
        "train.bc_prior.n_demos": 4, "train.bc_prior.epochs": 40,
        "train.seeds_per_candidate": 1, "evaluate.rollouts_per_candidate": 1,
        "final_retrain.enabled": True, "final_retrain.n_seeds": 1, "final_retrain.env_steps": 256,
        "post": ["final_retrain"]})
    ctx = Context(cfg=cfg, budget=Budget(), rundir=RunDir(tmp_path, "retrain", "hash0"))
    ctx.env = registry.get("env", "pendulum")(ctx)
    T._POLICY_STORE.clear()
    T.ensure_bc_prior(ctx, None, cfg)
    cand = Candidate(cand_id="c0", iteration=0,
                     reward_code="def compute_reward(state, action):\n"
                                 "    return -float(state[2] ** 2), {'v': 1.0}\n")
    search = registry.get("train_backend", "sb3")(ctx, RunState(), cand, 1)
    (row,) = search.seed_metrics
    assert row["warm_started_from"] == T.BC_PRIOR_REF and row["anchor"].get("kind") == "kl_clone", (
        f"precondition: the SEARCH's learner is the anchored clone fine-tune; got {row}")
    state = RunState()
    state.best = CandidateReport(cand_id="c0", candidate=cand, result=search, fitness=1.0)
    T._POLICY_STORE.pop(T.BC_PRIOR_REF, None)

    run_final_retrain(ctx, state)

    fr = json.loads((ctx.rundir.path / "phases" / "final_retrain.json").read_text())
    assert not fr["error"], fr["error"]
    assert fr["train_init"] == "bc_prior" and fr["train_anchor"] == "kl_clone", fr
    (executed,) = fr["executed"]
    assert executed["init"] == "bc_prior", executed
    assert executed["warm_started_from"] == T.BC_PRIOR_REF, "the retrain did not start from the clone"
    assert executed["anchor"].get("kind") == "kl_clone", "the retrain trained unanchored"
    assert fr["pruned"] is False
    assert fr["seeds"] and fr["seeds"][0] != row["seed"], "the retrain replayed the search's seed"


# --------------------------------------------------------------------------
# a report protocol failing must not replace the search's result
# --------------------------------------------------------------------------

def _events(fp: dict) -> list:
    return [json.loads(line) for line in fp["journal.jsonl"].splitlines() if line.strip()]


def test_a_retrain_that_raises_does_not_take_the_searchs_result_with_it(tmp_path, monkeypatch):
    """The realistic shape: a retrain seed worker dies (`_fork_seeds` fatal
    payload -> `_sb3_run` raises RuntimeError in the PARENT). Handling only
    `BudgetExceeded` would let the error escape `_execute` before `result.json`
    is composed, and `run()` would stamp the whole run `failed` -- a completed
    20-hour search reading as one that never finished."""
    from bird.components import phases

    real_get = phases.registry_get

    def exploding_get(kind, name):
        if kind != "train_backend":
            return real_get(kind, name)

        def boom(ctx, state, candidate, n_seeds=1, **kw):
            raise RuntimeError("final_retrain seed worker 1/3 died: SIGSEGV")
        return boom

    monkeypatch.setattr(phases, "registry_get", exploding_get)
    fp = _run("eureka", tmp_path, **{"final_retrain.enabled": True, "final_retrain.n_seeds": 3})

    res = json.loads(fp["result.json"])
    assert res["returned_cand_id"] and res["returned_fitness"] is not None, (
        "the search's own result was lost")
    assert json.loads(fp["status.json"])["status"] == "ok"
    fr = json.loads(fp["phases/final_retrain.json"])
    assert fr["error_type"] == "RuntimeError" and "SIGSEGV" in fr["error"], fr
    assert fr["retrained_fitness"] is None and fr["post_selection_gap"] is None
    assert fr["cand_id"] == res["returned_cand_id"]
    # visible on result.json, and not silently: the journal names it
    assert res["post_phase_errors"] == {"final_retrain": fr["error"]}, res["post_phase_errors"]
    failed = [e for e in _events(fp) if e.get("stage") == "final_retrain_failed"]
    assert len(failed) == 1 and "SIGSEGV" in failed[0]["error"], failed


def test_a_post_phase_that_raises_outright_is_recorded_on_the_result(tmp_path, monkeypatch):
    """The other half, in `_execute`: a phase that does NOT handle its own
    failure. `registry.load_all` is guarded, so the swapped entry is the one
    `_execute` resolves."""
    from bird import registry

    def boom(ctx, state=None):
        raise ValueError("the report protocol fell over")

    registry.load_all()
    monkeypatch.setitem(registry._REGISTRY, ("phase", "final_retrain"), boom)
    fp = _run("eureka", tmp_path, **{"final_retrain.enabled": True})

    res = json.loads(fp["result.json"])
    assert res["returned_cand_id"] and json.loads(fp["status.json"])["status"] == "ok"
    assert res["post_phase_errors"] == {"final_retrain": "ValueError: the report protocol fell over"}
    assert "phases/final_retrain.json" not in fp, "the phase never ran far enough to write one"
    failed = [e for e in _events(fp) if e.get("stage") == "post_phase_failed"]
    assert len(failed) == 1 and failed[0]["phase"] == "final_retrain", failed


def test_a_healthy_retrain_records_no_error():
    """Guard the guard: the two fields the failure path fills are empty on the
    ordinary path, so a reader can key on them."""
    # (the end-to-end runs above already assert `error == ""`; this pins the
    # result.json side, which they do not read)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        fp = _run("eureka", Path(d), **{"final_retrain.enabled": True})
    res = json.loads(fp["result.json"])
    fr = json.loads(fp["phases/final_retrain.json"])
    assert res["post_phase_errors"] == {} and fr["error"] == "" and fr["error_type"] == ""


# ==========================================================================
# The retrain's CURVE (`checkpoint_series`)
# ==========================================================================
#
# CARD's Fig. 3 (`fig:metaworld_comparison`) compares SUCCESS-RATE TRAINING
# CURVES. Under `loop.termination: fixed_generations` the returned candidate is
# never search-trained, so its `train_result.json` holds zero checkpoints and
# its ONLY training is this phase's. Were endpoints alone persisted, a CARD arm
# would have no returned-candidate curve anywhere in the artifact, and a table
# built from the run dir could reach for an in-loop candidate's curve and print
# the returned candidate's id beside it (a table's label must come from the same
# object as its numbers). These two tests are the two halves:
# the series is written and is the SAME RECORD the in-loop training writes, and
# nothing that was already in `final_retrain.json` moved.

def _series(fp: dict) -> dict:
    """The retrain's curve file, found through the record rather than by
    guessing the path -- so the two cannot drift apart silently."""
    fr = json.loads(fp["phases/final_retrain.json"])
    rel = fr["checkpoint_series"]
    assert rel, f"the record points at no series: {fr['checkpoint_series']!r}"
    key = f"phases/{rel}"
    assert key in fp, f"{key} is named by the record and is not in the run dir"
    return json.loads(fp[key])


def _series_on_disk(out) -> dict:
    """The same file, unscrubbed. `_fingerprint` strips `_VOLATILE` recursively
    (`eval_wall_s` among them), so a field-presence check has to open the real
    artifact -- which also proves the path the record names is a real path."""
    (run,) = [q for q in Path(out).iterdir() if q.is_dir()]
    return json.loads((run / "phases" / "final_retrain" / "checkpoints.json").read_text())


def _inloop_points(fp: dict) -> list:
    """Every checkpoint record the SEARCH wrote, off `train_result.json`."""
    out = []
    for rel, text in fp.items():
        if _TRAIN_RESULT.match(rel) is None:
            continue
        for row in json.loads(text).get("seed_metrics") or []:
            out.extend(row.get("checkpoints") or [])
    return out


def test_the_retrain_writes_the_same_checkpoint_record_the_search_does(tmp_path):
    """The series exists, is per seed, and each point is SHAPE-IDENTICAL to an
    in-loop point -- same keys, so the retrain's curve and a search
    candidate's curve go on one axis with no reshape. A reshape here is the one
    thing that would leave them incomparable, which is the whole reason the
    phase needed a curve.
    """
    fp, fr = _retrain(tmp_path)
    series = _series(fp)

    assert [s["seed"] for s in series["per_seed"]] == [float(s) for s in fr["seeds"]], (
        "the curve's seeds must be the retrain's own seeds, in order")
    assert len(series["per_seed"]) == fr["n_seeds"] == 3, series
    assert series["cand_id"] == fr["cand_id"] and series["seed_phase"] == "final_retrain"
    assert series["n_seeds"] == fr["n_seeds"]
    assert series["env_steps_per_seed"] == fr["env_steps_per_seed"]
    # As CONFIGURED, null included: `_checkpoint_episodes` turns it into a
    # number using the backend's own constant, and this file must not
    # second-guess that rule. Null is the default and is what eureka resolves
    # to, so the honouring half is asserted separately below on a run that
    # pins the key.
    assert series["checkpoint_eval_episodes"] == load(
        "eureka", profile="tester")["evaluate.checkpoint_eval_episodes"]

    inloop = _inloop_points(fp)
    assert inloop, "the search trained nothing; the shape comparison is vacuous"
    shape = {frozenset(p) for p in inloop}
    assert len(shape) == 1, f"the search itself writes >1 checkpoint shape: {shape}"
    for s in series["per_seed"]:
        assert s["checkpoints"], f"seed {s['seed']} has an empty curve"
        for p in s["checkpoints"]:
            assert frozenset(p) == next(iter(shape)), (
                f"retrain point {sorted(p)} is not the in-loop record "
                f"{sorted(next(iter(shape)))}")
    # The four fields a reader of the retrain curve relies on, named so a
    # rename fails here rather than downstream. Read OFF
    # DISK, not off the fingerprint: `_fingerprint` scrubs `_VOLATILE`, so
    # `eval_wall_s` is gone from every dict above and asserting it there would
    # be a check on the scrubber.
    on_disk = _series_on_disk(tmp_path)
    for field in ("step", "success_rate", "task_success", "eval_wall_s"):
        assert field in on_disk["per_seed"][0]["checkpoints"][0], field
    # `mean` is `TrainResult.checkpoints` -- the same across-seed mean
    # `train_result.json` writes at its top level, and the same length.
    assert len(series["mean"]) == len(series["per_seed"][0]["checkpoints"])
    assert frozenset(series["mean"][0]) == next(iter(shape))
    # ...and the endpoint the record already published is ON the seed's own
    # curve, so the scalar and the curve cannot describe different runs.
    assert len(fr["per_seed_task_metric"]) == len(series["per_seed"])
    for scalar, s in zip(fr["per_seed_task_metric"], series["per_seed"]):
        assert scalar in [p["task_success"] for p in s["checkpoints"]], (
            f"the published endpoint {scalar} is on no checkpoint of seed "
            f"{s['seed']}: the record and the curve describe different runs")


def test_the_curve_is_purely_additive_to_the_records_existing_fields(tmp_path, monkeypatch):
    """The endpoints are BYTE-IDENTICAL with the series on and off.

    Two whole runs of the same config at the same seed on the mock backend,
    the second with the series suppressed at its single source
    (`_checkpoint_series`), the only thing the series adds to the phase's
    calls. Both
    `phases/final_retrain.json` bodies are compared after `scrub_volatile`
    (the clocks, which differ between any two runs) and after dropping the one
    key that is meant to differ. Anything else moving -- a reordered field, a
    score recomputed, a selection that went elsewhere -- fails here.
    """
    from bird.components import phases as phases_mod

    with_series = _run("eureka", tmp_path / "with", **{"final_retrain.enabled": True,
                                                       "final_retrain.n_seeds": 3})
    monkeypatch.setattr(phases_mod, "_checkpoint_series", lambda result: None)
    without = _run("eureka", tmp_path / "without", **{"final_retrain.enabled": True,
                                                      "final_retrain.n_seeds": 3})

    a = json.loads(with_series["phases/final_retrain.json"])
    b = json.loads(without["phases/final_retrain.json"])
    assert a.pop("checkpoint_series") == "final_retrain/checkpoints.json"
    assert b.pop("checkpoint_series") is None
    assert scrub_volatile(a) == scrub_volatile(b), (
        "the retrain's record moved when the curve was written")
    # ...and the key set is the series-off one plus exactly one name.
    assert set(json.loads(with_series["phases/final_retrain.json"])) == (
        set(json.loads(without["phases/final_retrain.json"])))
    # The file follows the switch, both ways: no curve, no file.
    assert "phases/final_retrain/checkpoints.json" in with_series
    assert not [k for k in without if k.startswith("phases/final_retrain/")], (
        "a suppressed series still wrote a file")


def test_the_series_records_the_episode_count_the_run_resolved(tmp_path):
    """`evaluate.checkpoint_eval_episodes` is the key that changes the
    checkpoint evaluation episode count, so the curve must say which value
    produced it. Pinned on a
    run that sets it, since the default is null and a null cannot tell a
    recorded key from an absent one."""
    fp, _fr = _retrain(tmp_path, **{"evaluate.checkpoint_eval_episodes": 2})
    assert _series(fp)["checkpoint_eval_episodes"] == 2
