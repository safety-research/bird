"""End-to-end: every method actually runs.

This is the framework's acceptance criterion, made executable -- if a
method cannot be expressed as a config, the schema is missing a knob, and the
way to find out is by running all of them through the same six functions with a
mock LLM and a toy env.
"""

import importlib.util
import json

import pytest

from conftest import REPO, TESTER_POINTS, apply_search_cap
from bird.config import load


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: Tester points whose PUBLISHED search width cannot run under the tester profile,
#: and so take `conftest.TESTER_SEARCH_CAP` as a ceiling. `singh_orp`'s published
#: hyperparameter search resolves to 3,240 candidates -- a hang, not a slow test
#: (the note on `TESTER_SEARCH_CAP` has the measurement). Every other point runs
#: at its published width, which is what "every method actually runs" means; a
#: blanket cap would also refuse points whose width is load-bearing (rf_agent's
#: action counts must sum to its `n_candidates`).
_SEARCH_CAPPED = frozenset({"singh_orp"})


def _load_point(path, profile):
    overrides = apply_search_cap(path, {}, profile) if path.stem in _SEARCH_CAPPED else {}
    return load(path, profile=profile, overrides=overrides)


def test_every_search_capped_point_is_a_tester_point():
    """The cap table names points that exist, or it caps nothing."""
    assert _SEARCH_CAPPED <= {p.stem for p, _ in TESTER_POINTS}


#: Tester configs that legitimately produce NO valid candidate, each with the
#: reason it is expected.
#:
#: An entry here is an ARGUMENT, not a permission slip: it is what the failure
#: message prints, and "we do not know why" is a finding rather than an
#: allowlist entry. Empty on purpose -- every shipped method is supposed to
#: produce a reward program on the tier built to prove it can.
EXPECTED_ZERO_VALID: dict = {}


def _valid_candidates(run) -> list:
    """Every candidate the run wrote that survived §2, from the artifact.

    Read off `candidates/*/meta.json` rather than off `result`, because
    `result` is the search's summary and a search that produced nothing
    summarises cleanly -- which is the whole failure being guarded against.
    """
    out = []
    for meta in sorted(run.glob("candidates/*/meta.json")):
        try:
            out.append(json.loads(meta.read_text()))
        except (OSError, ValueError):  # pragma: no cover - unreadable artifact
            continue
    return [m for m in out if m.get("valid")]


@pytest.mark.parametrize("path,profile", TESTER_POINTS, ids=lambda v: getattr(v, "stem", v))
def test_method_runs_end_to_end(path, profile, tmp_path):
    cfg = _load_point(path, profile)
    result = _entry().run(cfg, out_root=str(tmp_path))

    assert result["name"] == cfg["name"]
    assert result["config_hash"] == cfg.hash()
    # A run must always report its cost -- a first-class output.
    assert "budget" in result and "policy_trainings" in result["budget"]

    # A RUN THAT RAISED NOTHING AND PRODUCED NOTHING IS NOT A METHOD THAT RAN.
    #
    # Everything above passes on a search whose every candidate was rejected:
    # `name` and `config_hash` come off the config, and `budget` exists from the
    # moment the run does. A mock that emits no `get_observation`, which
    # `generate.co_design.observation_fn` requires, would make every LIMEN
    # candidate fail §2 -- 2 candidates, 2 invalid, 0 trainings, no state file
    # and `returned_cand_id: None` at every seed -- and still pass the checks
    # above, with the method never exercised end to end at all.
    #
    # "Runs end to end" has to mean something was produced, or it means "did not
    # raise" -- which is the same sentence as a CI floor that only counts
    # whether pytest exited 0.
    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    valid = _valid_candidates(run)
    expected_zero = EXPECTED_ZERO_VALID.get(path.stem)
    n_written = len(sorted(run.glob("candidates/*/meta.json")))
    if expected_zero is not None:
        assert not valid, (
            f"{path.stem} is on EXPECTED_ZERO_VALID ({expected_zero}) but "
            f"produced {len(valid)} valid candidate(s). Remove the entry: an "
            "allowlist that outlives its reason hides the next regression.")
        return
    assert valid, (
        f"{path.stem} completed without raising and produced no valid "
        f"candidate ({n_written} written, all rejected at §2). Either the "
        "method is broken on this tier, or the mock does not satisfy a "
        "contract the prompt asks for -- both are findings. If it is genuinely "
        "expected, add it to EXPECTED_ZERO_VALID with the reason."
        + _first_failures(run))


def _first_failures(run, limit: int = 3) -> str:
    """The rejection reasons, so the failure names the cause and not just the
    count. A floor that says `0 != >0` sends the reader back to the artifact."""
    reasons = []
    for meta in sorted(run.glob("candidates/*/meta.json")):
        try:
            m = json.loads(meta.read_text())
        except (OSError, ValueError):  # pragma: no cover
            continue
        if not m.get("valid") and m.get("failure"):
            reasons.append(f"    {m.get('cand_id')}: {m['failure']}")
    if not reasons:
        return ""
    shown = reasons[:limit]
    more = f"\n    ... and {len(reasons) - limit} more" if len(reasons) > limit else ""
    return "\n  rejections:\n" + "\n".join(shown) + more


@pytest.mark.parametrize("path,profile", TESTER_POINTS, ids=lambda v: getattr(v, "stem", v))
def test_run_writes_a_reproducible_artifact(path, profile, tmp_path):
    cfg = _load_point(path, profile)
    _entry().run(cfg, out_root=str(tmp_path))
    runs = list(tmp_path.iterdir())
    assert len(runs) == 1
    run = runs[0]
    for expected in ("config.resolved.yaml", "journal.jsonl", "budget.json", "result.json"):
        assert (run / expected).exists(), f"{expected} missing from {run.name}"
    # the resolved config alone must be enough to reproduce the run
    assert cfg.hash() in run.name


def test_skipped_trainings_are_counted_not_hidden(tmp_path):
    """CARD's entire contribution is RL runs it did NOT launch. If the counter
    stays at zero the cost claim is invisible and the comparison is dishonest."""
    cfg = load("card", profile="tester", overrides={
        "verify.tpe.on_failure": "skip_training",
        "loop.n_iterations": 3,
    })
    result = _entry().run(cfg, out_root=str(tmp_path))
    b = result["budget"]
    assert "policy_trainings_skipped" in b
    assert b["policy_trainings"] + b["policy_trainings_skipped"] > 0


def test_tpe_skip_versus_train_anyway_changes_the_cost(tmp_path):
    """The headline two-value ablation: CARD-the-paper vs CARD-the-code."""
    entry = _entry()
    costs = {}
    for mode in ("skip_training", "train_anyway"):
        cfg = load("card", profile="tester", overrides={
            "verify.tpe.on_failure": mode,
            "loop.n_iterations": 3,
            "seed": 0,
        })
        costs[mode] = entry.run(cfg, out_root=str(tmp_path / mode))["budget"]
    assert costs["skip_training"]["policy_trainings"] <= costs["train_anyway"]["policy_trainings"], (
        "skip_training must not launch more RL runs than train_anyway")


def test_tpe_store_grows_by_the_collection_count_unwindowed(tmp_path):
    """Looking cannot catch this: a store of the wrong size renders exactly like
    a healthy run -- journal, prompts and result all keep their shape, only the
    evidence population the TPE gate judges on is wrong (e.g. 3 rollouts per
    iteration instead of the published 100, or truncated to a recency window the
    release does not have)."""
    cfg = load("card", profile="tester", overrides={
        "verify.tpe.trajectories_per_iteration": 5,
        "evaluate.rollouts_per_candidate": 3,
        "loop.n_iterations": 3,
        "seed": 0,
        # The published point is `loop.termination:
        # fixed_generations`: the LAST iteration is generate-only, trains nothing
        # and so appends nothing (measured: 5, 10, 10). This test is about the
        # STORE -- three appends, cumulative, unwindowed -- so it pins the loop
        # that trains every iteration; tests/test_fixed_generations.py measures
        # the generate-only end.
        "loop.termination": "fixed_iterations",
    })
    _entry().run(cfg, out_root=str(tmp_path))
    run = next(tmp_path.iterdir())
    states = sorted((run / "state").glob("iter*.json"))
    ns = [json.loads(p.read_text())["n_trajectories"] for p in states]
    # Every trained tester-tier card iteration passes the screen (vacuously: the
    # toy store is one-sided) and trains, so each appends exactly the COLLECTION
    # count -- 5, decoupled from rollouts_per_candidate=3 -- and nothing ever
    # truncates: 5, 10, 15.
    assert ns == [5 * (i + 1) for i in range(len(ns))], (
        f"store sizes per iteration were {ns}; expected the collection count "
        f"(5) per store-appending iteration, cumulative and unwindowed")


def test_gt_cold_start_trains_the_kept_five_not_all_ten(tmp_path):
    """Silent failure: an inert TAC cold start trains all 10 and still renders
    as a plausible run -- every artifact keeps its shape, only the count GT
    pins five independent ways (§4 Training, §5.1, §5.3, App. G, Figs 7-8)
    moves, and with it the C(10,2)=45-vs-C(5,2)=10 seeding of D_pref."""
    cfg = load("gt", profile="tester",
               overrides={"loop.n_iterations": 1, "seed": 0})
    keep = int(cfg["verify.alignment_filter.keep_top_n"])
    n = int(cfg["generate.n_candidates"])
    assert keep < n, "precondition: the cut must have something to cut"
    _entry().run(cfg, out_root=str(tmp_path))
    run = next(p for p in tmp_path.iterdir() if p.is_dir())
    # Count the CANDIDATES the iteration trained, off the artifact -- not
    # `budget.policy_trainings`, which honestly also counts the final_retrain
    # post phase's seeds and would let 10-trained-here hide behind a changed
    # retrain protocol.
    trained = [p for p in run.glob("candidates/iter00_*/train_result.json")
               if json.loads(p.read_text()).get("trained")]
    assert len(trained) == keep, (
        f"iteration 1 trained {len(trained)} candidates; "
        f"GT trains the kept {keep} in EVERY iteration, the first included")
    events = [json.loads(line) for line in
              (run / "journal.jsonl").read_text().splitlines() if line.strip()]
    cold = [e for e in events if e.get("stage") == "screen_cold_start"]
    assert cold, "the sigma-undefined cut must be journalled, never silent"
    assert cold[0]["n_kept"] == keep and cold[0]["n_screened"] == n - keep


def test_failed_tpe_verdict_never_extends_the_store():
    """Silent failure: under `on_failure: train_anyway`
    a TPE-failing candidate stays trainable, so a trainability-only gate lets
    its trajectories pollute the evidence every later verdict is computed from
    -- and a polluted store still yields plausible verdicts. Both sources gate
    store growth on the VERDICT (Alg. 1 l.9-11; driver :428 inside pass_flag)."""
    from bird.components.update import _TRAJECTORY_SELECTORS
    from bird.types import Candidate, CandidateReport, Selection, TrainResult

    def _report(records):
        c = Candidate(cand_id="c", iteration=0, reward_code="def compute_reward(s, a, s2):\n    return 0.0\n")
        c.verify_records = list(records)
        return CandidateReport(cand_id="c", candidate=c,
                               result=TrainResult(cand_id="c", candidate=c))

    failing = _report([{"phase": "screen", "screen": "tpe", "ok": False,
                        "detail": "x", "trained_anyway": True}])
    clean = _report([{"phase": "screen", "screen": "tpe", "ok": True}])
    assert failing.candidate.trainable, "precondition: train_anyway keeps it trainable"
    pick = _TRAJECTORY_SELECTORS["append_on_pass"]
    assert pick(Selection(winners=[failing])) == []
    assert pick(Selection(winners=[clean])) == [clean]


def test_tpe_reward_error_is_a_refusal_not_a_vacuous_pass():
    """Silent failure: a reward that raises (or goes non-finite) on every stored
    trajectory, if excluded from the statistic entirely, leaves an empty
    statistic that reads as a vacuous PASS -- a pathological reward trained
    where the release would fail it or halt. The
    refusal's score must be None, never 0.0 (no ordering statistic exists)."""
    import random

    import numpy as np

    from bird import registry
    from bird.budget import Budget
    from bird.components.screens import screen_tpe
    from bird.context import Context
    from bird.state import RunState
    from bird.types import Candidate, Trajectory

    registry.load_all()
    cfg = load("card", profile="tester")
    ctx = Context(cfg=cfg, budget=Budget(),
                  env=registry.get("env", "toy_reacher")({}))
    ctx.rng = random.Random(0)
    state = RunState()

    def _traj(success):
        return Trajectory(states=np.ones((5, 4)), actions=np.zeros((4, 2)),
                          rewards=[0.0] * 4, length=4, ret=0.0, success=success)

    # Two-sided store with real state arrays: the re-score failure is the
    # REWARD's, not missing data.
    state.trajectory_store = [_traj(True), _traj(False)]
    c = Candidate(cand_id="c", iteration=1,
                  reward_code="def compute_reward(s, a, s2):\n    raise ValueError('boom')\n")
    screen_tpe(ctx, state, [c])
    assert c.screened_out and c.failure_kind == "screened", (
        "an all-errored re-score must screen the candidate out (skip_training), "
        "not pass it vacuously")
    verdict = c.meta["tpe"]
    assert verdict["passed"] is False
    assert verdict["score"] is None, "refusal: None stays distinct from 0.0"
    assert verdict["n_failure_errors"] == 1 and verdict["n_success_errors"] == 1


def test_failures_stay_distinguishable_in_the_artifact(tmp_path):
    """Never collapse 'never compiled' and 'a screen rejected it'."""
    import json
    cfg = load("eureka", profile="tester", overrides={"loop.n_iterations": 2})
    _entry().run(cfg, out_root=str(tmp_path))
    run = next(tmp_path.iterdir())
    kinds = set()
    for meta in (run / "candidates").rglob("meta.json"):
        kinds.add(json.loads(meta.read_text())["failure_kind"])
    assert kinds, "no candidates were recorded at all"
    assert kinds <= {"", "invalid", "screened"}, f"unexpected failure kinds: {kinds}"


def test_a_run_is_deterministic_under_a_fixed_seed(tmp_path):
    entry = _entry()
    a = entry.run(load("eureka", profile="tester", overrides={"seed": 7}), out_root=str(tmp_path / "a"))
    b = entry.run(load("eureka", profile="tester", overrides={"seed": 7}), out_root=str(tmp_path / "b"))
    assert a["returned_fitness"] == b["returned_fitness"]
    assert a["returned_reward_code"] == b["returned_reward_code"]


# --------------------------------------------------------------------------
# what the mock emits vs what the prompt asked for
# --------------------------------------------------------------------------

#: (tester config, format, does the prompt ask for a separate ```json block?)
#:
#: Checked across all four `generate.output.format` values, because the defect
#: has a general shape: A MOCK WHOSE OUTPUT SATISFIES A CONTRACT THE PROMPT DOES
#: NOT REQUEST TURNS EVERY TEST ON THAT PATH INTO A NO-OP.
#:
#: `component_dict_plus_weights` asks for the weights in a separate ```json
#: block; a mock that put them only in a module-level dict, which
#: `_parse_weights` accepts as its SECOND source, would parse the weights and
#: pass every test while the branch the contract actually names is never taken
#: by anything. `template_params` is tracked separately: its
#: prompt forbids a function outright, which no mock can satisfy while the
#: pipeline still requires an entrypoint.
_FORMAT_CONTRACT = [
    ("eureka", "component_dict_return", False),
    ("rda", "component_dict_plus_weights", True),
    ("text2reward_human", "scalar_only", False),
    pytest.param("l2r", "template_params", True, marks=pytest.mark.xfail(
        strict=True,
        reason="template_params is UNSATISFIABLE by any mock as the contract "
               "stands: the prompt says `Do not write a reward function ... "
               "emit a single json code block ... Emit nothing else`, and the "
               "pipeline requires a reward entrypoint. A mock that obeys the "
               "prompt produces an invalid candidate; one that produces a valid "
               "candidate disobeys the prompt, silently. xfail rather than "
               "allowlisted because it is blocked on a DESIGN decision (is the "
               "adapter missing, or is the schema claiming a capability the "
               "algorithm lacks?) and not on a small fix. strict=True so that "
               "resolving it turns this red and someone deletes the mark.")),
]


@pytest.mark.parametrize("name,fmt,wants_json_block", _FORMAT_CONTRACT,
                         ids=["component_dict_return", "component_dict_plus_weights",
                              "scalar_only", "template_params"])
def test_the_mock_answers_the_contract_the_prompt_states(name, fmt, wants_json_block,
                                                         tmp_path):
    """The mock is the only responder this suite ever sees, so a clause of the
    contract it does not exercise is a clause nothing exercises."""
    # FOUR SAMPLES ONLY WHERE THE REASON APPLIES. The docstring below justifies
    # sampling 4 by the fenced-json branch being a dice roll on one draw -- that
    # argument is exactly the `wants_json_block` cases and no others. Forcing 4
    # everywhere also breaks `text2reward_human`, whose `select.rule: none`
    # ranks nothing and returns `reports[:1]`, so `_check_coherence` refuses
    # K > 1 there: three of every four would be generated, trained and thrown
    # away. Its own config says so -- "there is one program, no contest exists".
    # One draw is the right number for a config that only ever has one.
    n_samples = 4 if wants_json_block else 1
    cfg = load(name, profile="tester", overrides={
        "seed": 0, "loop.n_iterations": 1,
        "generate.n_candidates": n_samples, "post": []})
    assert cfg["generate.output.format"] == fmt, "fixture drifted from the config"
    _entry().run(cfg, out_root=str(tmp_path))

    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    cands = sorted(run.glob("candidates/*"))
    assert len(cands) == n_samples
    raws = {c: (c / "response.txt").read_text() for c in cands}

    for raw in raws.values():
        assert "```python" in raw, f"{fmt}: every contract asks for a python block"
    if wants_json_block:
        # `_parse_weights` reads a fenced json block FIRST and a module-level
        # dict second. Emitting only the second leaves the first untested, and
        # the first is the one the prompt names.
        #
        # AT LEAST ONE of the sampled responses, not the single first draw.
        # The mock's per-sample RNG hashes the PROMPT (mock.py::_rng), so any
        # edit that moves prompt-visible bytes re-rolls whether draw #1 lands
        # on an invalid archetype -- and an invalid program has no weights to
        # emit. Even renaming a comment inside an adapter class body that
        # `full_source` renders into the prompt re-rolls it, with the method
        # contract untouched. One draw would make this a dice roll on every
        # prose edit; the docstring's actual claim is that the fenced-json branch
        # is exercised by SOMETHING, so sample 4 (INVALID_FRACTION is 0.15) and
        # require one hit.
        hits = [c for c, raw in raws.items() if "```json" in raw]
        assert hits, (
            f"{fmt}: the prompt asks for the weights in a separate ```json "
            "block and none of the 4 sampled responses carried one, so "
            "`_parse_weights`'s fenced-json branch is exercised by nothing")
        meta = json.loads((hits[0] / "meta.json").read_text())
        assert meta.get("weights"), (
            f"{fmt}: no weights parsed -- this format exists to produce them")
        assert not (meta.get("meta") or {}).get("weights_missing")


def test_a_published_failure_value_of_zero_is_honoured():
    """Regression pin for a silent one: a `_failure_value` that spelt its
    default with `or` would read `select.failure_value: 0.0` as -10000.0, so a
    config pinning RF-Agent's `reward_fail_bound = 0` would run Eureka's
    sentinel while its resolved config and journal both said 0. Nothing renders
    wrong: a -10000 in `report.json` reads as an ordinary failure -- a wrong
    number that renders as a plausible one. `None` (the key absent) must still
    be the sentinel."""
    from types import SimpleNamespace

    from bird.components.evaluation import _failure_value

    def ctx_with(value):
        return SimpleNamespace(cfg=SimpleNamespace(get=lambda key, default=None: value))

    assert _failure_value(ctx_with(0.0)) == 0.0
    assert _failure_value(ctx_with(-1.0)) == -1.0
    assert _failure_value(ctx_with(None)) == -10000.0


def test_a_sign_test_reject_never_carries_its_demo_score_as_fitness():
    """Silent case: a `fitness_demo_margin` gated on `candidate.trainable is not
    None`, which is always true, would let a candidate the demo screen REJECTED
    (margin <= 0) keep its negative score as fitness and win a verifier-only
    round with a number that reads as an ordinary score. The gate is the
    screen's own `passed_sign_test`; a reject sentinels."""
    from types import SimpleNamespace
    from bird.components.evaluation import fitness_demo_margin
    from bird.types import Candidate, TrainResult

    cfg = {"select.failure_value": -10000.0, "evaluate.fitness.seed_aggregation": "mean",
           "evaluate.fitness.checkpoint_aggregation": "final",
           "evaluate.fitness.normalisation": "none", "evaluate.screened_fitness": "sentinel"}
    ctx = SimpleNamespace(cfg=SimpleNamespace(get=lambda k, d=None: cfg.get(k, d),
                                              __getitem__=lambda self, k: cfg[k]),
                          event=lambda *a, **k: None)

    def result(cid, demo, screened):
        c = Candidate(cand_id=cid, iteration=0, reward_code="def r(): return 0",
                      screened_out=screened, failure="demo_margin: x" if screened else "",
                      failure_kind="screened" if screened else "")
        c.meta["demo"] = demo
        return TrainResult(cand_id=cid, candidate=c, trained=False)

    passer = result("c0", {"score": 0.4, "margin": 0.3, "monotonicity": 0.1,
                           "kept": False, "passed_sign_test": True}, screened=True)   # skip_all
    reject = result("c1", {"score": -0.2, "margin": -0.3, "monotonicity": 0.1,
                           "kept": False, "passed_sign_test": False}, screened=True)
    reps = {r.cand_id: r for r in fitness_demo_margin(ctx, None, [passer, reject])}
    assert reps["c0"].fitness == 0.4
    assert reps["c1"].fitness == -10000.0, "a sign-test reject must lose every comparison"
