"""A malformed worker payload is one failed candidate, never a failed run.

`_merge_child` reads `payload["result"]` and `payload["candidate"]` unguarded,
so without validation a worker returning a partial payload would raise
`KeyError` **inside the parent's train stage** and kill the run. That is precisely the class
`_worker_failure` exists for, and its own docstring makes the argument: "a
worker that died is ONE failed candidate, not a failed iteration".

WHY IT IS NOT HYPOTHETICAL. On the fork path only a bug in this repo's own
child could produce it, but a worker that does not inherit its parent's
memory (a spawned worker, which imports the code afresh) makes a payload
that disagrees about the format a protocol mismatch -- and a protocol
mismatch must cost one candidate, never a run.

The eight-of-nine case is the one to keep in mind while reading: it is what a
truncated write looks like, it would fold "successfully" for eight keys
without validation, and the ninth is the one that raises.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from bird.components import training as T
from bird.types import Candidate, TrainResult


def _payload(**overrides):
    """A payload of exactly the keys `build_child_payload` produces.

    The drift assertion below catches an added key -- a fixture that quietly
    kept an old key count would keep passing while the builder and the
    validator moved on, which is the two-copies-of-one-fact shape this suite
    exists to prevent.
    """
    cand = Candidate(cand_id="c0000", iteration=0, reward_code="")
    base = {
        "result": TrainResult(cand_id="c0000", candidate=cand),
        "candidate": cand,
        "budget": {},
        "policy": [],
        "replay": [],
        "round_replay": [],
        "code": [],
        "env_states": None,
        "exceeded": "",
        # Optional -- a payload without journal lines still folds -- but a
        # COMPLETE payload has it, and this fixture stands for a complete one. The
        # missing-key case is asserted by
        # `test_a_payload_missing_an_OPTIONAL_key_is_not_malformed` below.
        "events": [],
        # The spawn worker's four. Optional for a structural reason: a FORKED
        # worker never sends them at all, so they are absent on the majority
        # path by design. Present here for the same reason `events` is --
        # this fixture stands for a COMPLETE payload, and the absent case is
        # asserted by the optional-key tests.
        "worker_start": "spawn",
        "worker_startup_s": {"import": 0.1, "env_construct": 0.2,
                             "first_compile": None, "total": 0.3},
        "xla_env": {"applied_in_time": None, "prepare_returned": False,
                    "preallocate": "false"},
        "jax_cache": {"entries": None},
    }
    assert set(base) == set(T.PAYLOAD_FIELDS), (
        "this fixture has drifted from PAYLOAD_FIELDS; it is derived from the "
        "constant on purpose so the suite cannot agree with itself about a key "
        "set the builder has moved on from")
    base.update(overrides)
    return base


def test_PAYLOAD_FIELDS_is_the_BUILDER_s_key_set_read_by_AST():
    """Tied to `build_child_payload`, not to this file's literal.

    THE FAILURE THIS PREVENTS IS THE DANGEROUS ONE, and it only exists
    because the refusal is strict. Suppose someone adds a
    tenth key to `build_child_payload`. `payload_problems` then rejects EVERY
    real payload as carrying an unexpected key -- every candidate of every
    parallel run fails -- while a test written against a hand-maintained
    literal of nine keys stays green, because the literal and the constant
    still agree with each other and neither agrees with the builder.

    Two copies of one fact keeping each other company while the truth moves
    is the defect this repo names most often. So the assertion reads the
    builder's own `payload = {...}` literal out of its AST: one source, and
    the test fails on the change that would otherwise break production
    silently.
    """
    tree = ast.parse(inspect.getsource(T.build_child_payload))
    built = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(getattr(t, "id", None) == "payload" for t in node.targets)):
            built = [k.value for k in node.value.keys
                     if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            break

    # TWO PRODUCERS, and the tie has to read both or it is a tie to half the
    # truth. `build_child_payload` writes the dict literal above;
    # `_spawn_child_main` then adds the spawn-only instrumentation by
    # subscript (`payload["worker_start"] = ...`) AFTER it returns, because
    # those facts are properties of having been spawned and the builder is
    # shared with workers that are not. So the constant is the UNION of what
    # any producer sends, and reading only the literal would let a fifth spawn
    # key drift into exactly the production failure this test was written
    # for: a spawned worker refused on every payload.
    spawn_tree = ast.parse(inspect.getsource(T._spawn_child_main))
    spawn_added = []
    for node in ast.walk(spawn_tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and getattr(node.targets[0].value, "id", None) == "payload"
                and isinstance(node.targets[0].slice, ast.Constant)
                and isinstance(node.targets[0].slice.value, str)):
            k = node.targets[0].slice.value
            if k not in spawn_added:
                spawn_added.append(k)
    assert spawn_added, (
        "found no `payload[\"...\"] = ` stores in _spawn_child_main -- the "
        "AST shape this half reads has changed. Re-point it rather than "
        "dropping it, or the spawn keys go untied")
    built = (built or []) + spawn_added

    assert built, (
        "could not find `payload = {...}` in build_child_payload -- the AST "
        "shape this test reads has changed, and the tie is only as good as "
        "its ability to find the literal. Re-point it rather than deleting it")
    assert built == list(T.PAYLOAD_FIELDS), (
        f"the two producers write {built} but PAYLOAD_FIELDS says "
        f"{list(T.PAYLOAD_FIELDS)}. With a strict unexpected-key refusal this "
        "makes the parent reject every real payload; order is asserted too, "
        "so the declared list stays the builder's own order")


def test_a_well_formed_payload_has_no_problems():
    assert T.payload_problems(_payload()) == []
    # And the field list is the builder's, not a second opinion about it.
    assert set(T.PAYLOAD_FIELDS) == set(_payload())


@pytest.mark.parametrize(
    "dropped", [k for k in T.PAYLOAD_FIELDS if k not in T.PAYLOAD_OPTIONAL])
def test_a_payload_missing_a_REQUIRED_key_is_refused(dropped):
    """EVERY required key, not just the two read unguarded.

    Parametrised over `PAYLOAD_FIELDS` MINUS `PAYLOAD_OPTIONAL` rather than
    over a literal list: `events` is optional because a payload without its
    journal lines still folds. Deriving the list means adding an optional key
    cannot silently make this test assert the opposite of the validator.

    `result` and `candidate` are the keys that raise today, so a test
    covering only those would pass while a payload missing `budget` folded
    silently and under-reported the cost column -- the quieter version of
    the same defect.
    """
    payload = _payload()
    del payload[dropped]
    problems = T.payload_problems(payload)
    assert problems, f"a payload missing {dropped!r} was accepted"
    assert any(dropped in p for p in problems), (
        f"the refusal does not name {dropped!r}: {problems}")


@pytest.mark.parametrize("dropped", list(T.PAYLOAD_OPTIONAL))
def test_a_payload_missing_an_OPTIONAL_key_is_not_malformed(dropped):
    """A worker that omits an optional key must cost nothing.

    The mirror of the required-key case above: `events` only carries journal
    lines, and a FORKED worker never sends the spawn-only keys. Treating either
    absence as malformed would turn "no journal lines" (or "the worker was
    forked") into one failed candidate, which is the cost `payload_problems`
    exists to avoid rather than to cause.
    """
    assert dropped in T.PAYLOAD_FIELDS
    payload = _payload()
    del payload[dropped]
    assert T.payload_problems(payload) == []


def test_every_problem_is_reported_not_just_the_first():
    """`Config.validate()`'s rule, for its reason: the text reaches a human
    hours later, and "missing candidate" when the truth is "missing five keys
    and result is a str" sends them to the wrong side of the boundary."""
    payload = {"result": "not a TrainResult", "policy": "not a list"}
    problems = T.payload_problems(payload)
    assert len(problems) >= 3, problems
    assert any("missing key" in p for p in problems)
    assert any("result is str" in p for p in problems)
    assert any("policy is str" in p for p in problems)


def test_a_store_row_that_is_not_a_pair_is_refused():
    """`_merge_child` unpacks these as `for key, value in ...`; a bare string
    unpacks into characters rather than raising, so this one is not caught by
    the fold failing -- it is caught by the fold SUCCEEDING and writing
    nonsense into the store."""
    for name in ("policy", "replay", "round_replay", "code"):
        problems = T.payload_problems(_payload(**{name: [("ok", 1), "xy"]}))
        assert any(name in p and "pair" in p or name in p and "(str, value)" in p
                   for p in problems), (name, problems)


def test_a_payload_that_is_not_a_dict_is_refused_without_raising():
    for junk in (None, [], "payload", 7):
        problems = T.payload_problems(junk)
        assert problems and "expected dict" in problems[0]


def test_an_unexpected_key_is_refused_too():
    """Not because `_merge_child` would trip on it -- it ignores what it does
    not read -- but because a payload carrying a key this code has never heard
    of was built by code that disagrees about the format -- a child running
    different code from its parent -- and silently dropping the extra key is
    how the mismatch stays invisible until a number is wrong."""
    problems = T.payload_problems(_payload(surprise=1))
    assert any("unexpected key" in p and "surprise" in p for p in problems)


def test_the_rejection_carries_the_KEY_SET_and_not_only_the_reason():
    """Which side built the payload, not just what is wrong with it.

    Eight of nine keys is a truncated write; a wholly different set is a
    child running different code from its parent. "missing key(s): candidate" cannot distinguish them, and the two have
    different fixes.

    Asserted on `payload_rejection`'s OUTPUT rather than on the call site's
    source: grepping `inspect.getsource` for the string would assert on a
    record rather than on the thing the mechanism produces, so the formatting
    lives in a function to make it an output.
    """
    payload = _payload()
    del payload["candidate"]
    text = T.payload_rejection(payload, T.payload_problems(payload))
    assert "missing key(s): candidate" in text
    assert "keys received:" in text
    for key in ("result", "budget", "policy", "exceeded"):
        assert key in text.split("keys received:")[1], (
            "the surviving keys must be listed, or a truncated write and a "
            "code mismatch read identically")
    assert "candidate" not in text.split("keys received:")[1]

    # Not a dict at all: no keys, and it must not raise while saying so.
    assert "keys received: none" in T.payload_rejection("junk", ["x"])


def test_the_fold_site_does_not_journal_through_ctx():
    """`training.py` may read only `ctx.cfg`, `ctx.env` and `ctx.budget`.

    Journalling the key set from here is not possible, because
    `tests/test_parallelism.py::test_the_train_stage_touches_only_cfg_env_budget`
    AST-checks that restriction -- a forked worker sees a private copy of
    anything else on the Context. The rejection text carries the information
    into `TrainResult.error` instead, which reaches the candidate's record.

    This one IS a source assertion, deliberately: the property is "this module
    does not call ctx.event", which has no runtime output to observe.
    """
    import inspect
    src = inspect.getsource(T)
    i = src.index("bad = payload_problems(payload)")
    assert "ctx.event(" not in src[i:i + 1600]


# ==========================================================================
# shape, not just presence
# ==========================================================================


def test_the_shape_table_covers_PAYLOAD_FIELDS_exactly():
    """One source again. A tenth field added to the protocol must not slip
    through unvalidated just because nobody remembered the second table."""
    assert set(T._PAYLOAD_SHAPE) == set(T.PAYLOAD_FIELDS), (
        "the shape table and the field list disagree: "
        f"{set(T._PAYLOAD_SHAPE) ^ set(T.PAYLOAD_FIELDS)}")


@pytest.mark.parametrize("field", ["result", "candidate"])
def test_a_key_that_is_PRESENT_BUT_EMPTY_is_refused(field):
    """Presence is not a value, and these two are the ones that matter.

    A presence check alone passes `{"result": None, ...}`, which carries every
    key -- then `_merge_child` folds a `None` result and every downstream
    reader sees a candidate with no training. A
    truncated write produces exactly that shape.
    """
    problems = T.payload_problems(_payload(**{field: None}))
    assert any(field in p and "None" in p for p in problems), (
        f"{field}=None passed the gate: presence was checked, value was not")


@pytest.mark.parametrize("field,value", [
    ("policy", []), ("replay", []), ("round_replay", []), ("code", []),
    ("budget", {}), ("exceeded", ""), ("env_states", None),
])
def test_the_fields_that_are_LEGITIMATELY_empty_stay_legal(field, value):
    """The other half of the rule, and the half a blanket non-empty check
    would break: most candidates write no store entry, most workers never
    trip the budget cap, and most adapters export no env states. Refusing
    those would fail every healthy run -- emptiness is per field, and a rule
    that forgot so would be worse than a presence-only check.
    """
    assert T.payload_problems(_payload(**{field: value})) == []


def test_a_result_about_a_DIFFERENT_candidate_is_refused():
    """The cross-field check a per-field pass cannot make.

    No key is missing and no type is wrong: the payload is well-formed and
    about someone else. It folds cleanly and attributes one candidate's
    training to another, so every downstream number is a real measurement of
    the wrong thing.
    """
    other = Candidate(cand_id="c0007", iteration=0, reward_code="")
    payload = _payload(result=TrainResult(cand_id="c0007", candidate=other))
    problems = T.payload_problems(payload)
    assert any("different candidate" in p for p in problems), problems

    # Not tripped when either side is unset -- a blank id is a different
    # defect and is not evidence of mis-routing.
    blank = Candidate(cand_id="", iteration=0, reward_code="")
    assert not any("different candidate" in p for p in
                   T.payload_problems(_payload(candidate=blank)))


def test_a_wrong_type_still_names_the_field_and_both_types():
    problems = T.payload_problems(_payload(budget=["not", "a", "dict"]))
    assert any("budget is list" in p and "dict" in p for p in problems), problems
