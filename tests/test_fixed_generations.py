"""`loop.termination: fixed_generations` -- the last iteration ends at §1 (CARD).

The wrong answer that renders as a plausible one: under `evaluate.fitness.source:
none` a TRAINED chain end and an UNTRAINED one both read `returned_fitness: null`
in `result.json`, both are adopted by `select.rule: none`, both are returned by
`select.final_artifact: chain_end`, and both are retrained by `post:
[final_retrain]`. Nothing in the returned program or the reflection prompt
differs. What differs is the accounting -- `budget.policy_trainings`, the
`policy_trainings_skipped` column CARD's cost claim is read from, the task metric
(a 1-seed number vs null), the `skip_reason` on the train result and the
`screen_*` events in the journal -- so those are the only tells, and they are what
this file measures. A plain query-then-train iteration would TPE-screen, train
and evaluate CARD's third generation before returning it, where Alg. 1 l.17 +
`Ensure R` and the release (metaworld_exp_one_step.py:374-381, then
query_llm_metaworld.py:44-53) hand it over straight from the Coder.

Every count below is an EQUALITY measured on this tree against the tester
profile's `loop.n_iterations: 2` (the profile wins over card's 3) and card's
`final_retrain.n_seeds: 5`; a change to either moves them, which is the intended
alarm (a floor would only catch shrinkage). Under tester the generate-only
iteration is iteration 1 and its candidate is `iter01_c0001`. The mock LLM's
samples are keyed to the PROMPT, so card's prompt (including its T2R Meta-World
template) is a third input to the LLM-side counts.

`SEED` is MEASURED, not chosen for convenience: under seeds 0 and 4 the mock's
iteration-1 sample fails validity and is resampled (`iter01_r0000`, `llm_calls:
3`, `verify_resamples: 1`), and under 5 it is resampled once as well; under 1,
2, 3, 6, 7, 8 and 9 the chain end is `iter01_c0001` with no resample. The first
of those. A prompt edit re-rolls this.
"""

from __future__ import annotations

import json

import pytest

from test_parallelism import _run

from bird.components.training import GENERATION_ONLY_SKIP
from bird.config import ConfigError, load

SEED = 1


def _journal(fp: dict) -> list:
    return [json.loads(line) for line in fp["journal.jsonl"].splitlines() if line.strip()]


def _events_between(events: list, iteration: int, start: str, end: str) -> list:
    """The journal slice from `stage_begin(of=start)` to `stage_begin(of=end)`
    within one iteration -- what a stage did, by the lines it wrote."""
    out, inside = [], False
    for e in events:
        if e.get("stage") == "stage_begin" and e.get("iteration") == iteration:
            if e.get("of") == start:
                inside = True
                continue
            if e.get("of") == end:
                break
        if inside:
            out.append(e)
    return out


@pytest.fixture(scope="module")
def card_run(tmp_path_factory) -> dict:
    """One `card` tester run: `configs/methods/card.yaml` resolves to
    `loop.termination: fixed_generations` on its own, so no override here --
    this IS the published point under the execution profile."""
    return _run("card", tmp_path_factory.mktemp("fg"), seed=SEED)


def test_the_final_iteration_generates_and_trains_nothing(card_run):
    """A query-then-train iteration would write `trained: true` to
    `iter01_c0001/train_result.json` and no `generation_only` journal line."""
    last = json.loads(card_run["candidates/iter01_c0001/train_result.json"])
    assert last["trained"] is False
    assert "fixed_generations" in last["skip_reason"]
    assert last["skip_reason"] == GENERATION_ONLY_SKIP
    first = json.loads(card_run["candidates/iter00_c0000/train_result.json"])
    assert first["trained"] is True, "iteration 0 is a full iteration and must still train"

    gen_only = [e for e in _journal(card_run) if e.get("stage") == "generation_only"]
    assert len(gen_only) == 1
    assert gen_only[0]["iteration"] == 1
    assert gen_only[0]["cand_ids"] == ["c0001"]
    assert gen_only[0]["trainable"] == 1
    assert gen_only[0]["termination"] == "fixed_generations"
    # And the per-candidate train line says the same, so the journal alone
    # (without the candidate dir) can tell an untrained chain end from a
    # trained one.
    trains = [e for e in _journal(card_run) if e.get("stage") == "train"]
    assert [t["trained"] for t in trains] == [True, False]
    assert trains[1]["seeds"] == 0 and trains[1]["env_steps"] == 0


def test_the_final_generation_is_validity_checked_but_not_screened(card_run):
    """A screened chain end would emit `screen_vacuous_pass` at iteration 1 too
    (TPE running on the chain end and passing it vacuously)."""
    events = _journal(card_run)
    it1 = _events_between(events, 1, "verify", "train")
    assert not [e for e in it1 if str(e.get("stage", "")).startswith("screen_")], it1
    # Positive control: the SAME slice for the full iteration does carry the
    # screen's event, so an empty slice above means "not screened", not "the
    # slice is empty for some other reason".
    it0 = _events_between(events, 0, "verify", "train")
    assert [e for e in it0 if str(e.get("stage", "")).startswith("screen_")], it0

    meta = json.loads(card_run["candidates/iter01_c0001/meta.json"])
    assert meta["valid"] is True
    assert meta["screened_out"] is False
    assert meta["failure_kind"] == ""
    # The validity half still ran: the candidate carries its verify records.
    assert meta["verify_records"], "validity did not run on the final generation"


def test_the_budget_books_the_unlaunched_training(card_run):
    """Training the chain end would book 7 / 0 (2 in-loop + 5 retrain, nothing
    skipped)."""
    b = json.loads(card_run["budget.json"])
    assert b["policy_trainings"] == 6          # 1 in-loop + 5 final_retrain
    assert b["policy_trainings_skipped"] == 1  # the generate-only third step
    assert b["candidates_invalid"] == 0
    # The paper's 2: one Coder call per generation, and validity still runs in
    # the generate-only iteration. Training the chain end would not change this
    # -- the LLM spend is what this key does NOT change. MEASURED, and
    # prompt-dependent: the mock's per-sample RNG is keyed to a hash of the
    # prompt (bird/llm/mock.py, `_rng_for`), so a different prompt can make the
    # mock forfeit a slot that validity then resamples (`llm_calls: 3`,
    # `verify_resamples: 1`). A future prompt edit to card moves these, which is
    # the intended alarm.
    assert b["llm_calls"] == 2
    assert b["verify_resamples"] == 0


def test_the_run_returns_the_untrained_program_and_says_so(card_run):
    """A trained chain end would give a `returned_task_metric` dict with
    `n_seeds: 1`; `returned_trained` is what tells the two apart."""
    r = json.loads(card_run["result.json"])
    assert r["returned_cand_id"] == "c0001"
    assert r["returned_fitness"] is None
    assert r["returned_fitness_source"] == "none"
    assert r["returned_task_metric"] is None
    assert r["returned_trained"] is False
    assert "fixed_generations" in r["returned_skip_reason"]
    assert r["returned_reward_code"] == card_run["candidates/iter01_c0001/reward.py"]
    assert json.loads(card_run["status.json"])["status"] == "ok"
    # The report §4 wrote for it carries the fixed feedback on the `none`
    # channel -- not a training reflection, because there was no training.
    rep = json.loads(card_run["candidates/iter01_c0001/report.json"])
    assert rep["fitness"] is None
    assert rep["feedback"] == GENERATION_ONLY_SKIP


def test_final_retrain_still_trains_the_returned_program(card_run):
    """The interaction guard: the retrain runs on the program the loop never
    trained, and says so in `selected_trained`."""
    fr = json.loads(card_run["phases/final_retrain.json"])
    assert fr["cand_id"] == "c0001"
    assert fr["n_seeds"] == 5
    assert fr["error"] == ""
    assert len(fr["seeds"]) == 5
    assert fr["selected_fitness"] is None
    assert fr["selected_trained"] is False
    assert fr["post_selection_gap"] is None
    assert fr["search_seeds"] == []  # nothing to overlap with: never trained in-loop
    assert json.loads(card_run["result.json"])["post_phase_errors"] == {}


def test_final_retrain_records_the_native_number_beside_the_arms_own(card_run):
    """CARD searches with `evaluate.fitness.source: none`, so its
    `retrained_fitness` is honestly null -- and if that were ALL the phase
    wrote, the five retrained seeds' native metrics would go nowhere and the
    arm would have no headline number at all. The same retrain is also scored
    on the env's own signal, with the channel named so a reader knows which
    quantity the number is."""
    fr = json.loads(card_run["phases/final_retrain.json"])
    assert fr["retrained_fitness"] is None and fr["per_seed_fitness"] == []
    assert fr["native_channel"] == "native_success", fr
    assert fr["native_error"] == "", fr
    assert isinstance(fr["retrained_native"], float)
    assert len(fr["per_seed_native"]) == fr["n_seeds"] == 5
    assert all(0.0 <= v <= 1.0 for v in fr["per_seed_native"])
    # And on the TASK METRIC, which is a third quantity again. CARD is the
    # sharpest case for this: its fitness source is `none`, so
    # `per_seed_fitness` is EMPTY above and the task metric is the only per-seed
    # task-success number the artifact carries at all. One value per seed that
    # ran -- the length is the half that matters, since a list silently shorter
    # than `n_seeds` hides the gap.
    assert len(fr["per_seed_task_metric"]) == fr["n_seeds"] == 5, fr
    assert all(v is None or 0.0 <= v <= 1.0 for v in fr["per_seed_task_metric"]), fr


def test_the_returned_candidate_finally_has_a_training_curve(card_run):
    """The reason `checkpoint_series` exists.

    CARD's Fig. 3 (`fig:metaworld_comparison`) compares SUCCESS-RATE TRAINING
    CURVES. Under `fixed_generations` the returned candidate is handed over
    from a generate-only iteration, so its own `train_result.json` holds ZERO
    checkpoints -- asserted here as the premise, not assumed -- and the only
    training it ever gets is `final_retrain`'s. A retrain that persisted
    endpoints alone would leave the quantity the paper plots nowhere in a
    finished CARD run, and a table built from the run dir could fall through
    to an IN-LOOP candidate's curve while printing the returned candidate's id
    beside it. The curve is written for the candidate the run actually returns.
    """
    last = json.loads(card_run["candidates/iter01_c0001/train_result.json"])
    assert last["trained"] is False and not (last.get("checkpoints") or []), (
        "the returned candidate was search-trained; this test's premise is gone")
    assert not [row.get("checkpoints") for row in (last.get("seed_metrics") or [])], last

    fr = json.loads(card_run["phases/final_retrain.json"])
    assert fr["cand_id"] == "c0001" and fr["selected_trained"] is False
    assert fr["checkpoint_series"] == "final_retrain/checkpoints.json"
    series = json.loads(card_run["phases/final_retrain/checkpoints.json"])
    assert series["cand_id"] == "c0001", "the curve must belong to the RETURNED candidate"
    assert len(series["per_seed"]) == fr["n_seeds"] == 5
    assert [s["seed"] for s in series["per_seed"]] == [float(x) for x in fr["seeds"]]
    for s in series["per_seed"]:
        assert s["checkpoints"], f"seed {s['seed']} retrained with no curve"
        # The two quantities the figure is drawn from, on every point.
        assert all("success_rate" in p and "task_success" in p for p in s["checkpoints"])
    assert len(series["mean"]) == len(series["per_seed"][0]["checkpoints"])


def test_fixed_iterations_is_the_loop_it_always_was(tmp_path):
    """Regression guard for the default path: the stage bodies gained a branch
    every other config must never take."""
    fp = _run("card", tmp_path, seed=SEED, **{"loop.termination": "fixed_iterations"})
    last = json.loads(fp["candidates/iter01_c0001/train_result.json"])
    assert last["trained"] is True
    b = json.loads(fp["budget.json"])
    assert (b["policy_trainings"], b["policy_trainings_skipped"]) == (7, 0)
    assert not [e for e in _journal(fp) if e.get("stage") == "generation_only"]
    r = json.loads(fp["result.json"])
    assert r["returned_trained"] is True
    assert r["returned_skip_reason"] == ""
    assert r["returned_task_metric"]["n_seeds"] == 1
    assert json.loads(fp["phases/final_retrain.json"])["selected_trained"] is True


def test_fixed_generations_refuses_a_final_artifact_that_cannot_return_it():
    """Neither override can return the generate-only chain end, so both are
    refused at load rather than validating cleanly."""
    load("card")  # positive control: the published point validates
    with pytest.raises(ConfigError, match="fixed_generations"):
        load("card", overrides={"select.final_artifact": "global_best"})
    with pytest.raises(ConfigError, match="fixed_generations"):
        load("card", overrides={"select.rule": "argmax_fitness"})
    # And the rule is over the KEY, not the method: eureka with the value
    # bolted on is refused for the same reason, before its 16 LLM calls.
    with pytest.raises(ConfigError, match="fixed_generations"):
        load("eureka", overrides={"loop.termination": "fixed_generations"})
