"""`loop.carry` governs the MEMORY slots; the parent-chain slots survive ungated.

`RunState.apply_carry` clears every `CARRY_SLOTS` attribute not named in
`loop.carry`. `iteration_best` (the round's winner) and `latest` (the chain
head) have no carry slot and are never cleared: every hill-climbing method
point reads one of them under a carry list that does not name it, so gating
them would break every declared method. That is the intended contract. This
file pins both lists as EQUALITIES against the code, the schema and
`bird/state.py`'s own docstring, so none of them can drift silently -- a
docstring that says "nothing else persists" would contradict the code.
"""

from __future__ import annotations

from conftest import REPO
from bird.schema import SCHEMA
from bird.state import CARRY_COMPANIONS, CARRY_SLOTS, RunState
from bird.types import Candidate, CandidateReport, TrainResult

#: The memory slots `loop.carry` may name. Copied here on purpose: the point of
#: the test is that this list, `CARRY_SLOTS` and the schema enum agree.
DOCUMENTED_CARRY = {
    "best_reward", "dialogue", "preference_dataset", "trajectory_store", "subtask_list",
    "replay_buffer", "policy_checkpoint", "failure_memory", "archive", "curriculum",
    "search_tree", "critic_population",
}
#: The parent-chain slots: written every round, never cleared by `apply_carry`.
UNGATED = {"iteration_best", "latest"}


def _report(cid: str) -> CandidateReport:
    cand = Candidate(cand_id=cid, iteration=0, reward_code="def compute_reward(): ...")
    return CandidateReport(cand_id=cid, candidate=cand,
                           result=TrainResult(cand_id=cid, candidate=cand), fitness=1.0)


def test_the_carry_slots_are_exactly_the_documented_memory_slots():
    assert set(CARRY_SLOTS) == DOCUMENTED_CARRY
    assert set(SCHEMA["loop.carry"].item_enum) == DOCUMENTED_CARRY


def test_the_parent_chain_slots_have_no_carry_slot_and_are_named_in_the_state_docstring():
    fields = set(RunState.__dataclass_fields__)
    assert UNGATED <= fields
    assert not (UNGATED & set(CARRY_SLOTS.values()))
    companions = {a for names in CARRY_COMPANIONS.values() for a in names}
    assert not (UNGATED & companions)
    text = (REPO / "bird/state.py").read_text()
    for name in UNGATED:
        assert f"`{name}`" in text, f"bird/state.py must name {name} as an ungated survivor"


def test_apply_carry_empty_clears_every_memory_slot_and_keeps_the_parent_chain():
    st = RunState()
    st.best = _report("best")
    st.iteration_best = _report("winner")
    st.latest = _report("chain")
    st.dialogue = [{"role": "user", "content": "x"}]
    st.subtasks = ["a", "b"]
    st.policy_ref = "policy:best"
    st.replay_ref = "replay:best"
    st.failure_memory = ["boom"]
    st.returned = st.best
    st.apply_carry([])
    for attr in CARRY_SLOTS.values():
        assert getattr(st, attr) in (None, [], {}), f"{attr} must be cleared by loop.carry=[]"
    assert st.iteration_best.cand_id == "winner"
    assert st.latest.cand_id == "chain"
    # The run's return is a RECORD, not a slot: `update()` writes it before this
    # carry precisely so the carry cannot take it (see state.py's docstring).
    assert st.returned.cand_id == "best"
    assert st.final_artifact("global_best").cand_id == "best"


def test_a_named_slot_survives_and_the_rest_do_not():
    st = RunState()
    st.best = _report("best")
    st.dialogue = [{"role": "user", "content": "x"}]
    st.apply_carry(["best_reward"])
    assert st.best.cand_id == "best" and st.dialogue == []
