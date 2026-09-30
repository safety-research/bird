"""ROSKA's fusion ratio: that it is applied, on BOTH backends, the right way round.

Three claims, and every one of them fails silently if it breaks — which is the
only reason this file exists. Looking at a run cannot catch any of them: the search prints nothing,
a wrong alpha produces a perfectly plausible number, and a fusion that never
happens is a warm start, which is itself a published method.

The third claim is the structural one. `train.init` is applied by TWO different
mechanisms — the surrogate learners blend in `__init__`, where theta_0 exists,
and sb3 blends inside `_sb3_apply_policy` around `set_parameters`. A split like
that produces silent no-ops and latent NameErrors on whichever side nothing
exercises, and "it compiles and validates" is no evidence either way. A
mechanism added to one backend must never be able to be a no-op on the other
without a test going red.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird.components.training import _blend_sb3_params

# The tester tier plus a real learner: mock LLM, toy env, tiny budgets.
_OVERRIDES = [
    "train.init=fused_warm_start",
    "train.fusion.ratio_search=sc_bo",
    "train.fusion.sc_bo.probe_fraction=0.05",
    "train.fusion.sc_bo.n_evaluations=3",
    "train.env_steps=2000",
    # The published schedule, without which `_check_coherence` refuses the
    # config outright: a `fused_warm_start` run that reduces neither fraction
    # trains every candidate for a FULL budget and pays the probes on top.
    "train.fusion.sc_bo.post_probe_fraction=0.1",
    "train.fusion.first_round_fraction=0.1667",
    "loop.n_iterations=2",
    "generate.n_candidates=2",
]


class _F(float):
    """Duck-type of a floating torch tensor: enough for `_blend_sb3_params`."""

    def is_floating_point(self) -> bool:
        return True


class _I(int):
    def is_floating_point(self) -> bool:
        return False


def test_alpha_one_is_the_inherited_policy_and_alpha_zero_the_fresh_one():
    """The endpoints ARE existing methods, so they must be exact, not close.

    alpha=1.0 is `warm_start_from_best` (RDA) and alpha=0.0 is `from_scratch`
    (Eureka). If the blend were applied to the wrong side these two would swap
    and every ROSKA number would still look reasonable — which is why "alpha
    reaches the parameters" is not enough on its own to pin it.
    """
    inherited = {"policy": {"w": _F(10.0), "n": _I(7)}}
    fresh = {"policy": {"w": _F(0.0), "n": _I(0)}}

    assert _blend_sb3_params(inherited, fresh, 1.0)["policy"]["w"] == 10.0
    assert _blend_sb3_params(inherited, fresh, 0.0)["policy"]["w"] == 0.0
    assert _blend_sb3_params(inherited, fresh, 0.5)["policy"]["w"] == 5.0
    # Non-float entries are counters, not parameters: they keep the inherited
    # value, which is what makes alpha=1.0 BE warm_start_from_best rather than
    # merely resemble it.
    assert _blend_sb3_params(inherited, fresh, 0.0)["policy"]["n"] == 7


def _search_records(run_dir: Path):
    """Every candidate's fusion record, keyed by candidate directory name."""
    out = {}
    for f in sorted(run_dir.glob("candidates/*/train_result.json")):
        rec = (json.loads(f.read_text()) or {}).get("fusion") or {}
        if rec:
            out[f.parent.name] = rec
    return out


def _layer(*words):
    """Later `KEY=VALUE` words override earlier ones, resolved HERE not at the shell.

    These fixtures are built as layers -- a base, `_OVERRIDES`, then a per-test
    `extra` that deliberately changes one of them (the `ratio_search=fixed`
    control against `_OVERRIDES`' `sc_bo` is the clearest case).

    `bird.py` REFUSES a repeated `--set` key, because letting the last one win
    silently lets word order decide an experiment.

    So the layering is resolved here instead, where it is the fixture's own
    stated intent rather than an emergent property of word order. The emitted
    command line contains each key exactly once.
    """
    resolved = {}
    for word in words:
        resolved[word.split("=", 1)[0]] = word
    return list(resolved.values())


def _run(tmp_path, backend, extra=()):
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "bird.py", "-c", "rda", "--profile", "tester",
           "--out", str(tmp_path)]
    for o in _layer(f"train.backend={backend}", *_OVERRIDES, *extra):
        cmd += ["-s", o]
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    dirs = [d for d in Path(tmp_path).rglob("result.json")]
    assert dirs, "no run directory was written"
    return dirs[0].parent


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["mock", "sb3"])
def test_the_fusion_search_runs_on_both_backends(tmp_path, backend):
    """The guard against the two-`run_seed` split.

    The test is slow-marked, and its `sb3` half needs `--extra sb3`
    (`importorskip`ped) -- it runs in an environment that has the learner
    (verified with sb3 2.9.0 / torch 2.13.0).

    **What this does NOT prove, verified by mutation.** It asserts the SEARCH
    ran, not that its result was used. The record is written by the probe, and
    the probe fuses correctly on its own, so deleting `fusion_alpha=` from the
    real seed's `_sb3_apply_policy` call leaves every record here intact and
    every assertion green while the trained policy gets a plain warm start.
    `test_the_chosen_alpha_reaches_the_trained_policy_on_sb3` is the one that
    fails on that mutation; this one is the guard against the search not
    running at all, which is a different and also silent failure.
    """
    if backend == "sb3":
        pytest.importorskip("stable_baselines3")
    run_dir = _run(tmp_path, backend, extra=(["train.algorithm=ppo"] if backend == "sb3" else []))

    records = _search_records(run_dir)
    assert records, f"{backend}: no candidate carried a fusion search record at all"

    # ROUND 1 MUST NOT SEARCH, and that is an ASSERTION rather than a filter: a
    # filter such as
    #     later = {k: v for k, v in records.items() if not k.startswith("iter00_")}
    # justified by "iteration 0 has no search -- the paper's shape too" would
    # hide code that searches at iteration 0 against an inherited policy of
    # None. Dropping a case because you believe it is empty asserts nothing
    # about whether it is.
    first = {k: v for k, v in records.items() if k.startswith("iter00_")}
    assert not first, (
        f"{backend}: round 1 carried a fusion search record ({sorted(first)}), but it has no "
        "previous policy to fuse with -- the paper trains round 1 short and searches "
        "nothing (appendix.tex:7), and searching here spends J x T_BO per candidate that "
        "its TTS does not count")
    later = {k: v for k, v in records.items() if not k.startswith("iter00_")}
    assert later, f"{backend}: no post-first-iteration candidate searched"
    for name, rec in later.items():
        assert rec["search"] == "sc_bo", f"{backend}/{name}: {rec['search']}"
        assert rec["fell_back"] is False, f"{backend}/{name}: silently fell back to a fixed alpha"
        assert rec["probe_train_steps_nominal"] > 0, f"{backend}/{name}: probed nothing"
        assert len(rec["alphas"]) == rec["evaluations"], f"{backend}/{name}: record is inconsistent"
        assert 0.0 <= rec["alpha"] <= 1.0


def test_no_policy_to_fuse_with_means_no_search_on_every_backend():
    """The condition itself, where all three backends share it.

    `_training_init` is the ONE place `_InitPlan.fusion_search` is set, and the
    mock, sb3 and fasttd3 backends all resolve their plan through it
    (`training.py` twice, `fasttd3.py` once -- asserted below rather than
    assumed, because "all three backends" is exactly the claim that goes stale
    when something is added to one of them). So this one condition is
    the whole of round 1's behaviour on every tier, including the two that
    `importorskip` out of CI.

    The run-level test above proves it on the tier that runs; this proves the
    condition, and fails on the exact mutation that matters -- taking
    `ratio_search` from the config unconditionally.
    """
    import types as _t

    from bird.components import training as T

    cfg = {"train.init": "fused_warm_start", "train.fusion.ratio_search": "sc_bo"}

    # Round 1: nothing carried, so nothing to fuse with.
    plan = T._training_init(None, _t.SimpleNamespace(policy_ref=None), cfg)
    assert plan.source == "scratch", plan
    assert plan.fusion_search is None, (
        "round 1 resolved a fusion search with no policy to fuse with: the probes would "
        "rank alphas for a blend that cannot happen, and their cost is not in the paper's TTS")

    # A ref that RESOLVES: rounds 2+, where the search is the method.
    ref = "policy:test-fusion-guard"
    T._POLICY_STORE[ref] = object()
    try:
        plan = T._training_init(None, _t.SimpleNamespace(policy_ref=ref), cfg)
        assert plan.source == "best", plan
        assert plan.fusion_search == "sc_bo", plan
        # An EVICTED ref is round 1 again, not a search against nothing: the
        # condition is `_resolved_source`'s verdict, not `ref is None`.
        del T._POLICY_STORE[ref]
        plan = T._training_init(None, _t.SimpleNamespace(policy_ref=ref), cfg)
        assert plan.source == "scratch", plan
        assert plan.fusion_search is None, plan
    finally:
        T._POLICY_STORE.pop(ref, None)

    # All three backends, by construction rather than by three runs.
    root = Path(__file__).resolve().parents[1]
    for rel, n in (("bird/components/training.py", 2), ("bird/components/fasttd3.py", 1)):
        src = (root / rel).read_text()
        got = src.count("init = _training_init(ctx, state, cfg, resume_ref, candidate)")
        assert got == n, (
            f"{rel} resolves its init plan through _training_init {got} times, expected {n}: "
            "if a backend stopped using it, round 1's no-search condition no longer covers it")


@pytest.mark.slow
def test_the_chosen_alpha_reaches_the_trained_policy_on_sb3(tmp_path):
    """That the search's RESULT is used, not merely recorded.

    Asserting on the fusion record proves the SEARCH ran; it says nothing about
    whether the chosen alpha reached the policy that trained, because the record
    is written by the probe, and the probe fuses correctly on its own. Deleting
    `fusion_alpha=` from the real seed's `_sb3_apply_policy` call leaves every
    record intact and every record-level assertion green while the trained
    policy gets a plain warm start — the artifact claiming a fused policy that
    never existed.

    So this pins the endpoints end to end instead: alpha=0.0 (from scratch) and
    alpha=1.0 (the inherited policy) must produce DIFFERENT results. If fusion
    is a no-op both collapse onto the warm start and the two runs agree.
    """
    pytest.importorskip("stable_baselines3")
    # `seed=1` is MEASURED, not chosen for convenience: the mock seeds every
    # sample from its prompt, and at seed 0 both runs return a round-1
    # candidate (which fuses nothing), so the round-2 difference the fusion
    # makes never reaches result.json. Seed 1 is the first at which both runs
    # return the same round-2 candidate, so the comparison below is on the
    # fused candidate itself. A prompt edit re-rolls this.
    common = ["train.algorithm=ppo", "train.fusion.ratio_search=fixed", "seed=1"]
    scratch = _run(tmp_path / "a", "sb3", extra=common + ["train.fusion.alpha=0.0"])
    inherit = _run(tmp_path / "b", "sb3", extra=common + ["train.fusion.alpha=1.0"])

    def _curve(run_dir):
        vals = []
        for f in sorted(run_dir.glob("candidates/*/train_result.json")):
            d = json.loads(f.read_text())
            for row in (d.get("seed_metrics") or []):
                if row.get("init") == "fused_warm_start":
                    vals.append((f.parent.name, row.get("fusion_alpha")))
        return vals

    a, b = _curve(scratch), _curve(inherit)
    # Round 1 has no incumbent to blend with, so it fuses nothing and records
    # no alpha -- the paper's own shape (`_training_init`, appendix.tex:6-7).
    for label, rows in (("alpha=0.0", a), ("alpha=1.0", b)):
        first = [v for c, v in rows if c.startswith("iter00_")]
        assert first and all(v is None for v in first), (
            f"{label} run recorded a fusion alpha in round 1: {rows}")
    a = [(c, v) for c, v in a if not c.startswith("iter00_")]
    b = [(c, v) for c, v in b if not c.startswith("iter00_")]
    assert a and b, "no later-round seed row recorded a fused init on one of the runs"
    # The alpha is recorded as executed, per seed, on both.
    assert {v for _, v in a} == {0.0}, f"alpha=0.0 run recorded {a}"
    assert {v for _, v in b} == {1.0}, f"alpha=1.0 run recorded {b}"

    # And it CHANGED something: the two runs must not agree everywhere. A
    # no-op fusion makes both of them `warm_start_from_best` and identical.
    ra = json.loads((scratch / "result.json").read_text())
    rb = json.loads((inherit / "result.json").read_text())
    # `returned_fitness` must be PRESENT: comparing a key result.json does not
    # write would read None on both sides, and the check could then only ever
    # pass on the candidate id.
    assert ra.get("returned_fitness") is not None and \
        rb.get("returned_fitness") is not None, (sorted(ra), sorted(rb))
    assert ra["returned_fitness"] != rb["returned_fitness"] or \
        ra.get("returned_cand_id") != rb.get("returned_cand_id"), (
            "alpha=0.0 and alpha=1.0 produced identical runs: the fusion ratio "
            "is not reaching the trained policy")


@pytest.mark.slow
def test_the_artifact_totals_equal_the_configs_fractions(tmp_path):
    """Every term of the published schedule is spent, and spent once.

    The paper's efficiency claim is arithmetic over four budgets
    (`appendix.tex:100-107`), so a term that is declared and not spent makes the
    claim false while every other number still looks right: a key can sit in
    `_default.yaml` and `schema.py`, cited in the config with paper line
    numbers, and be read by nothing.

    Asserted against the ARTIFACT, never against the config: the config is where
    the expectation comes from, so a test that read it on both sides would
    compare a number to itself.
    """
    frac = dict(first=0.1667, post=0.1, ext=0.8333, probe=0.0667)
    extra = ["train.fusion.ratio_search=sc_bo",
             f"train.fusion.sc_bo.probe_fraction={frac['probe']}",
             "train.fusion.sc_bo.n_evaluations=6",
             f"train.fusion.sc_bo.post_probe_fraction={frac['post']}",
             f"train.fusion.first_round_fraction={frac['first']}",
             f"train.winner_extension_fraction={frac['ext']}",
             "update.winner.action=extend_then_become_parent",
             "generate.parent_source=global_best",
             "loop.n_iterations=3"]
    run_dir = _run(tmp_path, "mock", extra=extra)

    import yaml
    rows = [json.loads(l) for l in (run_dir / "journal.jsonl").open()]
    full = int(yaml.safe_load((run_dir / "config.resolved.yaml").read_text())["train"]["env_steps"])

    budgets = [r for r in rows if r.get("stage") == "round_budget"]
    exts = [r for r in rows if r.get("stage") == "winner_extension"]
    assert budgets, "no round_budget was journalled: the schedule is not being applied"
    assert exts, "no winner_extension was journalled: the +2500 term is not being spent"

    # Round 1 is the short training; every later round is the post-probe top-up.
    by_slot = {}
    for b in budgets:
        by_slot.setdefault(b["slot"], set()).add(b["env_steps_per_candidate"])
    assert by_slot["round1_train"] == {round(frac["first"] * full)}, by_slot
    assert by_slot["post_probe"] == {round(frac["post"] * full)}, by_slot
    # Exactly one round-1 budget, and one extension per iteration.
    assert len([b for b in budgets if b["slot"] == "round1_train"]) == 1, budgets
    assert len(exts) == len(budgets), (len(exts), len(budgets))
    for e in exts:
        assert e["applied"] is True, e
        assert e["requested"] == round(frac["ext"] * full), e

    # THE ONE THAT MATTERS, and the reason this block exists. Everything above
    # reads the journal, which the PARENT writes beside the dispatch -- so it
    # says what the budget was MEANT to be and would keep saying it if the
    # number never reached the backend. Verified by mutation:
    # deleting `backend_kw.setdefault("env_steps", round_budget)` from stage 3
    # leaves every assertion above green while every candidate trains for a
    # full `train.env_steps`. So compare against what each candidate ACTUALLY
    # spent, which only the backend can write.
    #
    # `>=` and a ceiling rather than equality: a learner spends whole episodes,
    # so actual is `max(requested, horizon)` -- but it must never be the FULL
    # budget, which is the failure being excluded.
    spends = []
    for f in sorted(run_dir.glob("candidates/*/train_result.json")):
        d = json.loads(f.read_text())
        used = int(d.get("env_steps_used") or 0)
        if used:
            spends.append((f.parent.name, used))
    assert spends, "no candidate recorded env_steps_used"
    biggest_slot = max(round(frac["first"] * full), round(frac["post"] * full))
    for name, used in spends:
        assert used < full, (
            f"{name} spent {used} of a full {full}: the per-round budget is not "
            f"reaching the backend, so the schedule is Eureka-plus-probes")
        assert used >= min(round(frac["first"] * full), round(frac["post"] * full)), (name, used)
    assert max(u for _, u in spends) <= max(biggest_slot, full // 2), (
        f"a candidate spent more than the largest scheduled slot: {spends}")

    # The probes: nominal is the paper's arithmetic and must be exactly
    # evaluations x fraction x full. `probe_train_steps` (actual) may exceed it
    # by the whole-episode floor, which is why both are recorded.
    recs = [json.loads(f.read_text()).get("fusion") or {}
            for f in sorted(run_dir.glob("candidates/*/train_result.json"))]
    searched = [r for r in recs if r.get("evaluations")]
    assert searched, "no candidate searched: the probe budget is not being spent"
    for r in searched:
        assert r["probe_train_steps_nominal"] == round(
            r["evaluations"] * r["probe_fraction"] * full), r
        assert r["probe_train_steps"] >= r["probe_train_steps_nominal"], r


@pytest.mark.slow
def test_the_probe_stream_is_reproducible_across_invocations(tmp_path):
    """One seeded config must give one answer, in two separate processes.

    A fusion probe seeded from `abs(hash((cand_id, alpha)))` would not be:
    Python salts its string hash per process, so the probe scores -- hence the
    GP's chosen alpha, hence every `sc_bo` run -- would differ between two
    invocations of the same seeded config. `_seed_base`'s own docstring forbids
    `hash()` for exactly this.

    **No other determinism test can catch it, and the reason is worth
    keeping**: `tests/test_parallelism.py` compares a forked run against a
    sequential one IN THE SAME PROCESS TREE, and a fork inherits the parent's
    hash salt. So the two agree perfectly while the thing that actually varies
    -- running the config again tomorrow -- does not. A determinism test that
    never crosses a process boundary cannot see a per-process seed.

    Hence two subprocesses under different `PYTHONHASHSEED`s, which is the only
    shape that fails on a hash-seeded probe.
    """
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]

    def alphas(seed_env, out):
        env = dict(os.environ, PYTHONHASHSEED=seed_env)
        cmd = [sys.executable, "bird.py", "-c", "rda", "--profile", "tester",
               "--out", str(out)]
        for o in _layer("train.backend=mock", *_OVERRIDES,
                        "generate.parent_source=global_best"):
            cmd += ["-s", o]
        p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=1800)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        got = []
        for f in sorted(Path(out).rglob("candidates/*/train_result.json")):
            rec = (json.loads(f.read_text()).get("fusion") or {})
            if rec.get("evaluations"):
                got.append((f.parent.name, round(rec["alpha"], 9),
                            [round(v, 9) for v in rec["scores"]]))
        return got

    a = alphas("1", tmp_path / "a")
    b = alphas("2", tmp_path / "b")
    assert a, "no candidate searched, so this proves nothing"
    assert a == b, (
        "the fusion search is not reproducible across invocations: the same seeded "
        f"config gave different alphas under two PYTHONHASHSEEDs.\n  {a}\n  {b}")


@pytest.mark.slow
def test_the_probe_steps_reach_the_budget_and_not_the_train_results(tmp_path):
    """`budget.json` counts the probes; the summed `train_result`s do not.

    ROSKA's probes are its LARGEST cost term -- 12 x 200 against Eureka's 3000
    per candidate -- and uncharged they would be invisible to `budget.json`:
    6,600 of 10,800 env steps on a tester run, 61% of the cost, missing in the
    one direction that flatters a method whose whole claim is efficiency. The
    precedent is `Budget.record_rollout_steps`' own docstring: a collection loop
    that reaches `TrainResult.env_steps_used` and never reaches the budget
    leaves two artifacts disagreeing about one quantity.

    Without this test a refactor could drop the charge with everything green.

    **The two artifacts count different things ON PURPOSE**, and the assertions
    below are what says so: `budget.json` is every env interaction the run paid
    for (training + evaluation + feedback rollouts + probes), while the summed
    `train_result.env_steps_used` is what the CANDIDATES' own trainings cost and
    omits the probes.

    WHY THIS IS A DIFFERENCE OF TWO RUNS AND NOT AN EQUALITY AGAINST ONE.
    `rollout_env_steps == probes` does not hold: the per-training feedback
    rollouts are charged through the same counter in every backend, so
    `rollout_env_steps` is `probes + feedback`.

    So the control run is the decomposition: the same fixture with
    `train.fusion.ratio_search: fixed`, which resolves alpha at no training
    cost and therefore spends NO probe steps, while training exactly as many
    candidates for exactly as many feedback rollouts (the rollout count is
    `evaluate.rollouts_per_candidate`, which does not depend on how long a
    training ran). Its `rollout_env_steps` is the feedback term alone --
    measured 675 against the searched run's 7275 with 6600 of probes -- and
    the difference is exactly the quantity a deleted charge would remove,
    which is the property this test is built for.
    """
    extra = ["train.fusion.ratio_search=sc_bo",
             "train.fusion.sc_bo.probe_fraction=0.0667",
             "train.fusion.sc_bo.n_evaluations=6",
             "generate.parent_source=global_best",
             "loop.n_iterations=2"]
    run_dir = _run(tmp_path / "searched", "mock", extra=extra)
    # The control: identical but for the search, so it pays the same feedback
    # rollouts and no probes. `_layer` resolves this over the `sc_bo` in
    # `_OVERRIDES` and in `extra` above, so the emitted line carries
    # `ratio_search` once -- `bird.py` refuses it twice.
    ctl_dir = _run(tmp_path / "fixed", "mock",
                   extra=list(extra) + ["train.fusion.ratio_search=fixed"])

    def _read(d):
        budget = json.loads((d / "budget.json").read_text())
        results = [json.loads(f.read_text())
                   for f in sorted(d.glob("candidates/*/train_result.json"))]
        return (budget,
                sum(int(r.get("env_steps_used") or 0) for r in results),
                sum(int((r.get("fusion") or {}).get("probe_env_steps") or 0)
                    for r in results))

    budget, trained, probes = _read(run_dir)
    ctl_budget, _ctl_trained, ctl_probes = _read(ctl_dir)

    assert probes > 0, "no probe steps were recorded; this asserts nothing"
    assert ctl_probes == 0, (
        f"the control run spent {ctl_probes} probe steps, so it is not a control: "
        "`train.fusion.ratio_search: fixed` must resolve alpha without training")

    # EQUALITY against the budget's own decomposition, not an inequality against
    # its total. `budget.env_steps > trained` and `>= trained + probes` BOTH
    # hold with the charge deleted -- `env_steps` also counts evaluation and
    # feedback rollouts, so it clears those bars either way (verified by
    # mutation with `record_rollout_steps` removed). Asserting on an aggregate
    # proves nothing about the term inside it; `rollout_env_steps` exists so
    # the claim is checkable.
    charged = int(budget["rollout_env_steps"]) - int(ctl_budget["rollout_env_steps"])
    assert charged == probes, (
        f"budget.json rollout_env_steps moved by {charged} between the searched run "
        f"({budget['rollout_env_steps']}) and the probe-free control "
        f"({ctl_budget['rollout_env_steps']}), but the candidates recorded {probes} "
        "probe steps: the probes are not reaching budget.json, so the run "
        "under-reports its own cost by its largest term")
    # And the two artifacts still count different things, which is the point of
    # recording both: the candidates' own trainings omit the probes.
    assert int(budget["env_steps"]) >= trained + probes, (
        f"budget {budget['env_steps']} < trained {trained} + probes {probes}")


def test_a_probe_that_raises_fails_the_candidate_and_not_the_run(tmp_path, monkeypatch):
    """`_resolve_fusion` runs the CANDIDATE'S reward inside the probe; a program
    that compiles and dies at its first call (the mock's `wrong_signature`
    archetype) therefore dies in the probe, before the seed loop whose own
    `except` records `trained=False`. Unguarded, that takes the whole run down
    whenever the mock happens to draw that archetype. The probe is forced to
    raise here so the case does not depend on which archetype seed 0 draws:
    every candidate must come back `trained=False` naming the probe, the run must
    finish, and the slot must be charged (`policy_trainings` counts launches)."""
    import importlib.util
    from bird.components import training as tr
    from bird.config import load

    def _boom(ctx, cfg, init, probe=None):
        raise RuntimeError("probe boom")
    monkeypatch.setattr(tr, "_resolve_fusion", _boom)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bird_entry_roska", root / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import yaml
    # `load` takes TYPED values; the fixture words are CLI strings, so coerce each
    # the way `bird.py -s` does (YAML scalar: 0.1667 -> float, 2 -> int).
    overrides = {k: yaml.safe_load(v) for k, v in
                 (w.split("=", 1) for w in _layer("train.backend=mock", *_OVERRIDES))}
    cfg = load("rda", profile="tester", overrides={**overrides, "seed": 0})
    out = mod.run(cfg, out_root=str(tmp_path))
    results = sorted(Path(tmp_path).rglob("candidates/iter*_*/train_result.json"))
    assert results, "no candidate was trained"
    for f in results:
        d = json.loads(f.read_text())
        if d.get("skip_reason"):
            continue
        assert d["trained"] is False and "fusion probe: RuntimeError: probe boom" in d["error"], (f, d)
    assert out["budget"]["policy_trainings"] == len([f for f in results
                                                     if not json.loads(f.read_text()).get("skip_reason")])
