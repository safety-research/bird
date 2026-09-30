"""The run returns what the search produced, whatever `loop.carry` drops.

`select.final_artifact` reads a memory slot (`global_best`/`all_survivors` ->
`state.best`, `archive_best` -> `state.archive`) or the chain head, and
`update()` ends every iteration with `apply_carry`, which clears every memory
slot the config does not name. A run that resolves its return AFTER the last
carry fails on `configs/methods/zeroshot.yaml` (`loop.carry: []`, one iteration,
inherited `global_best`): it finishes with `returned_cand_id: None`,
`returned_fitness: None` and no reward code in `result.json` while its
candidate sits in `candidates/` -- a wrong artifact for a published method
point. (`configs/methods/text2reward_zeroshot.yaml` sidesteps the hazard in the config
with `chain_end`, which reads the ungated chain head.) Post phases resolving
through their own copy of the rule table have the same hole, so
`final_retrain` under `loop.carry: []` would find nothing to retrain either.

So the return is recorded on `RunState.returned` in `update()`, BEFORE the
carry, and both `bird.final_artifact` and the post phases read it through the
one resolver `RunState.final_artifact`. `apply_carry`'s contract is untouched
(`tests/test_carry_contract.py`): `best` is still cleared; the RECORD is not a
slot.
"""

from __future__ import annotations

import importlib.util
import json
import random

import pytest

from conftest import REPO
from bird import registry
from bird.budget import Budget
from bird.components import phases
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, Selection, TrainResult


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _valid_candidate_ids(run) -> set:
    out = set()
    for meta in run.glob("candidates/*/meta.json"):
        m = json.loads(meta.read_text())
        if m.get("valid"):
            out.add(m.get("cand_id"))
    return out


@pytest.mark.parametrize("name", ["zeroshot", "text2reward_zeroshot"])
def test_a_search_that_carries_nothing_still_returns_the_reward_it_produced(name, tmp_path):
    """`zeroshot` is the published point (inherited `global_best`);
    `text2reward_zeroshot` is its child, which chose `chain_end` for the
    method's own reasons."""
    cfg = load(name, profile="tester")
    assert cfg["loop.carry"] == [], "the whole point: nothing is carried"
    result = _entry().run(cfg, out_root=str(tmp_path))
    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    valid = _valid_candidate_ids(run)
    assert valid, "the mock must produce a valid candidate for the return to mean anything"
    assert result["returned_cand_id"] in valid, (
        f"{name} returned {result['returned_cand_id']!r}; the run produced {sorted(valid)}")
    assert result["returned_fitness"] is not None
    assert result["returned_reward_code"], "the returned program must be in result.json"
    # The state snapshot after the carry still shows an honestly empty
    # incumbent -- and the return beside it, because they are different things.
    last = sorted(run.glob("state/iter*.json"))[-1]
    snap = json.loads(last.read_text())
    if cfg["select.final_artifact"] == "global_best":
        assert snap["best_id"] is None, "apply_carry([]) must still clear the incumbent"
    assert snap["returned_id"] == result["returned_cand_id"]


def _report(cid: str, fitness: float) -> CandidateReport:
    cand = Candidate(cand_id=cid, iteration=0, reward_code="def compute_reward(): return 0.0")
    return CandidateReport(cand_id=cid, candidate=cand,
                           result=TrainResult(cand_id=cid, candidate=cand), fitness=fitness)


def test_update_records_the_return_before_the_boundary_carry():
    """The mechanism, on the real stage 6 under `loop.carry: []`."""
    registry.load_all()
    driver = _entry()
    cfg = load("eureka", profile="tester", overrides={"loop.carry": []})
    ctx = Context(cfg=cfg, budget=Budget(), rundir=None, rng=random.Random(0))
    state = RunState()
    winner = _report("c0000", 0.8)
    state = driver.update(ctx, state, Selection(winners=[winner], losers=[]))
    assert state.best is None, "the carry contract is untouched: best is a memory slot"
    assert state.returned is winner, "the return was recorded before the carry"
    assert driver.final_artifact(ctx, state) is winner
    # the post phases resolve the same object through the same resolver
    assert phases._FINAL_ARTIFACT_RULES["global_best"](state) is winner
    assert state.summary()["returned_id"] == "c0000" and state.summary()["best_id"] is None


def test_a_later_iteration_that_resolves_nothing_does_not_unknow_the_return():
    registry.load_all()
    driver = _entry()
    cfg = load("eureka", profile="tester", overrides={"loop.carry": []})
    ctx = Context(cfg=cfg, budget=Budget(), rundir=None, rng=random.Random(0))
    state = RunState()
    first = _report("c0000", 0.8)
    state = driver.update(ctx, state, Selection(winners=[first], losers=[]))
    # an iteration whose every candidate failed selects nobody
    state = driver.update(ctx, state, Selection(winners=[], losers=[]))
    assert driver.final_artifact(ctx, state) is first


def test_the_live_slots_still_resolve_when_no_record_exists():
    """A state built by hand (`tests/test_post_phase_target.py`'s shape) or one
    that never completed an iteration: the rules read the slots as before."""
    r = _report("c1", 0.5)
    st = RunState(best=r, latest=r)
    assert st.final_artifact("global_best") is r and st.final_artifact("chain_end") is r
    assert st.final_artifact("all_survivors") is r
    assert RunState().final_artifact("global_best") is None
    assert RunState().final_artifact("archive_best") is None
    assert RunState().resolve_final_artifact("not_a_rule") is None


def test_an_unknown_rule_is_a_config_error_at_the_driver():
    driver = _entry()
    from bird.config import ConfigError
    cfg = load("eureka", profile="tester")
    ctx = Context(cfg=cfg, budget=Budget(), rundir=None)
    ctx.cfg._data["select"]["final_artifact"] = "nope"  # bypass the schema on purpose
    with pytest.raises(ConfigError):
        driver.final_artifact(ctx, RunState())


def test_the_checkpoint_walks_the_return_record_for_store_refs():
    """`returned` is a record and not a slot, so it can outlive the slot it was
    copied from (`chain_end` after `op_restart` dropped `latest`, or a later
    iteration that resolved nothing). `checkpoint.reachable_refs` decides which
    store payloads a checkpoint carries and `_null_dead_refs` which restored
    refs are set to None -- enumerating only `best`/`latest`/`iteration_best`/
    archive there would let a resumed leg's return dangle a `policy_ref` the
    store does not hold, recorded nowhere."""
    from bird import checkpoint
    r = _report("c7", 0.9)
    r.result.policy_ref = "policy:c7"
    r.result.replay_ref = "replay:c7"
    st = RunState(returned=r)  # every slot cleared by the carry; only the record survives
    pol, rep = checkpoint.reachable_refs(st)
    assert pol == {"policy:c7"} and rep == {"replay:c7"}
    dropped = checkpoint._null_dead_refs(st, policy={}, replay={})
    assert r.result.policy_ref is None and r.result.replay_ref is None, (
        "a ref with no payload behind it must not survive as a plausible string")
    assert {(d["slot"], d["where"]) for d in dropped} == {
        ("policy_ref", "returned"), ("replay_ref", "returned")}
