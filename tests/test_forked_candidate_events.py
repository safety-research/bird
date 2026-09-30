"""A forked candidate's journal comes home.

`training._child_main` clears `ctx.rundir` so a worker cannot write into the
run directory. Without a fallback, `Context.event` would then be a no-op in
every forked child, and every event a worker emitted would be discarded: the
journal of a `candidate_parallelism: parallel` run would carry the parent's
rows only, and read as though the candidates had journalled nothing.

The fallback is `Context.event_buffer`: a child with no run dir collects its
events, `build_child_payload` ships them in the payload's `events`, and
`_merge_child` replays them into the parent's one journal, in emission order,
before anything the fold itself emits.
"""

from __future__ import annotations

from bird.budget import Budget
from bird.config import Config
from bird.context import Context
from bird.types import Candidate


def _ctx(**kw):
    return Context(cfg=Config({"name": "eureka"}), budget=Budget(), **kw)


class _Parent:
    """A parent context that records what the fold replays into it."""

    cfg = Config({})
    budget = Budget()

    def __init__(self):
        self.seen = []

    def event(self, stage, **fields):
        self.seen.append((stage, fields))


def _payload(cand, **extra):
    from bird.components import training

    payload = {
        "result": training.TrainResult(cand_id=cand.cand_id, candidate=cand),
        "candidate": cand, "budget": {}, "policy": [], "replay": [],
        "round_replay": [], "code": [], "env_states": None, "exceeded": "",
    }
    payload.update(extra)
    return payload


def test_a_context_with_no_run_dir_and_no_buffer_still_drops():
    """The default is unchanged, which is what keeps every other caller safe.

    A Context built by hand in a test, or one before `run()` installs a
    RunDir, must not accumulate events forever.
    """
    ctx = _ctx()
    ctx.event("train", cand_id="c0000")
    assert ctx.event_buffer is None


def test_a_child_collects_its_events_in_order():
    """Order is the part that is easy to lose and impossible to reconstruct.

    The journal is read as a sequence; a worker's rows arriving sorted by
    anything but emission order would describe a run that did not happen.
    """
    ctx = _ctx(event_buffer=[])
    ctx.event("reward_source", cand_id="c0000", source="native")
    ctx.event("train_ceiling", cand_id="c0000")
    assert [s for s, _ in ctx.event_buffer] == ["reward_source", "train_ceiling"]
    assert ctx.event_buffer[0][1] == {"cand_id": "c0000", "source": "native"}


def test_the_payload_carries_the_events_and_the_parent_replays_them():
    """End to end: what the child emitted is what the parent journals.

    The fold replays into the parent's `ctx.event`, so the rows land in the
    one journal, written by the one process that owns the file.
    """
    from bird.components import training

    child = _ctx(rundir=None, event_buffer=[])
    child.event("reward_source", cand_id="c0000", source="native")
    cand = Candidate(cand_id="c0000", iteration=0, reward_code="")

    payload = _payload(cand, events=list(child.event_buffer))
    assert training.payload_problems(payload) == []

    parent = _Parent()
    training._merge_child(parent, cand, payload)
    assert parent.seen == [("reward_source", {"cand_id": "c0000", "source": "native"})]


def test_a_payload_without_events_is_not_malformed():
    """A payload without the field must cost nothing.

    Treating a missing `events` as malformed would turn "this
    payload has no journal" into one failed candidate, which is the cost
    `payload_problems` exists to avoid rather than to cause.
    """
    from bird.components import training

    cand = Candidate(cand_id="c0000", iteration=0, reward_code="")
    assert "events" in training.PAYLOAD_FIELDS
    assert "events" in training.PAYLOAD_OPTIONAL
    assert training.payload_problems(_payload(cand)) == []


def test_a_malformed_event_row_does_not_fail_the_fold():
    """The training is done and paid for by the time the fold runs.

    A journalling defect must not discard a result, so a bad row is logged
    and skipped rather than raised. The good rows around it still land.
    """
    from bird.components import training

    cand = Candidate(cand_id="c0000", iteration=0, reward_code="")
    payload = _payload(cand, events=["not a pair", ("ok", {"a": 1})])
    parent = _Parent()
    training._merge_child(parent, cand, payload)
    assert parent.seen == [("ok", {"a": 1})], (
        "a malformed row took the good one with it, or the fold raised")
