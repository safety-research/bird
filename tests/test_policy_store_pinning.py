"""The warm-start no-op: eviction drops the blob the state still points at.

`_store` evicts OLDEST-FIRST, and the oldest entry is the incumbent -- the one
entry the run cannot lose. The 16 GiB cap makes that unreachable for a
~305 MiB policy blob; these tests pin the property itself, so it also holds
for the next blob that outgrows the cap. That is the failure this guards: the ref stays alive in the RunState, the blob does not,
the warm start finds nothing to restore, the candidate cold-starts, and
`train.init: warm_start_from_best` reads as configured on every side.

Under a 512 MiB cap, a K=8 x 5-iteration run should warm-start 32 of its 40
trainings, and eviction of the incumbent makes it fall far short of that.
"""
import numpy as np
import pytest

from bird.components import training


BLOB_MIB = 305  # the measured K=8 FastTD3/SimbaV2 policy blob: 319,265,762 B


def _blob(mib: int = BLOB_MIB):
    """A blob of `mib` MiB; mib=0 gives a 1-byte one for the count-bound test,
    which needs many entries and must not cost 64 x 305 MiB of RSS to run."""
    return training._PolicyBlob(b"\0" * max(1, mib * 1024 * 1024))


@pytest.fixture(autouse=True)
def _clean():
    training.reset_policy_stores()
    yield
    training.reset_policy_stores()


def test_a_pinned_policy_survives_eviction_by_bytes():
    """The incumbent must survive a cap that MUST evict something.

    The cap is set to three blobs, so storing a round of eight forces eviction
    on every write after the third -- otherwise this is a test of nothing. At
    the production 16 GiB cap with nine 305 MiB blobs neither bound is reached
    and an unpinned `_store` keeps the incumbent too, so a test there would
    pass without the pin.
    """
    cap = 3 * BLOB_MIB * 1024 * 1024
    store = {}
    training._store(store, "policy:incumbent", _blob(), cap)
    training._PINNED_REFS.add("policy:incumbent")
    for i in range(8):
        training._store(store, f"policy:c{i:04d}", _blob(), cap)
    assert "policy:incumbent" in store, (
        "eviction dropped the ref the RunState still holds -- this is the "
        "warm-start no-op, and it is silent"
    )
    assert len(store) < 9, "nothing was evicted, so the cap was never binding"


def test_a_pinned_policy_survives_eviction_by_count():
    """The same, for the OTHER bound: `_STORE_LIMIT` entries, not bytes."""
    store = {}
    training._store(store, "policy:incumbent", _blob(0), None)
    training._PINNED_REFS.add("policy:incumbent")
    for i in range(training._STORE_LIMIT + 4):
        training._store(store, f"policy:c{i:04d}", _blob(0), None)
    assert "policy:incumbent" in store
    assert len(store) <= training._STORE_LIMIT


def test_an_unpinned_entry_is_still_evicted_oldest_first():
    """The pin narrows eviction; it must not disable it."""
    store = {}
    cap = 2 * BLOB_MIB * 1024 * 1024
    for name in ("a", "b", "c"):
        training._store(store, f"policy:{name}", _blob(), cap)
    assert "policy:a" not in store
    assert "policy:c" in store


def test_eviction_never_drops_the_entry_just_written():
    store = {}
    training._store(store, "policy:keep", _blob(), 1)  # cap below one blob
    assert "policy:keep" in store


def test_pin_refs_is_bounded_by_an_archive():
    """A big archive must NOT enlarge the pin set.

    `checkpoint.reachable_refs` walks every archive cell, and the archive is
    bounded by config, not by `_STORE_LIMIT`. Pinning it turns both store
    bounds off: measured on a 75-cell LIMEN-shaped state, 77 refs were
    pinned, the store grew past its 64-entry limit, and a byte cap of 1
    evicted nothing. `pin_refs` therefore names only what an initialisation
    loads, never the archive.
    """
    class _R:
        def __init__(self, i):
            self.policy_ref = f"policy:arch{i}"
            self.replay_ref = f"replay:arch{i}"

    class _Cell:
        def __init__(self, i):
            self.report = type("rep", (), {"result": _R(i)})()

    class _State:
        policy_ref = "policy:winner"
        replay_ref = "replay:winner"
        best = latest = iteration_best = returned = None
        archive = {i: _Cell(i) for i in range(75)}

    training.pin_refs(_State())
    assert len(training._PINNED_REFS) <= training._MAX_PINNED, (
        f"{len(training._PINNED_REFS)} refs pinned; an archive-sized pin set "
        f"disables both store bounds"
    )
    assert "policy:arch0" not in training._PINNED_REFS


def test_pin_refs_reads_the_state_not_the_config():
    """`pin_refs` must protect the refs an initialisation LOADS.

    Deliberately not "what `reachable_refs` names" -- that is the superset which
    includes the archive, and pinning it disables both store bounds. See
    `test_pin_refs_is_bounded_by_an_archive`.
    """
    class _Result:
        policy_ref = "policy:winner"
        replay_ref = None

    class _Report:
        result = _Result()

    class _State:
        policy_ref = "policy:winner"
        replay_ref = "replay:winner"
        best = _Report()
        latest = None
        iteration_best = None
        returned = None
        archive = {}

    training.pin_refs(_State())
    assert "policy:winner" in training._PINNED_REFS
    assert "replay:winner" in training._PINNED_REFS
    assert training.BC_PRIOR_REF in training._PINNED_REFS


def test_pin_refs_replaces_rather_than_accumulates():
    """A ref the state has dropped stops being protected."""
    class _State:
        policy_ref = "policy:new"
        replay_ref = None
        best = latest = iteration_best = returned = None
        archive = {}

    training._PINNED_REFS.add("policy:stale")
    training.pin_refs(_State())
    assert "policy:stale" not in training._PINNED_REFS
    assert "policy:new" in training._PINNED_REFS


def test_reset_clears_the_pins_too():
    training._PINNED_REFS.add("policy:x")
    training.reset_policy_stores()
    assert not training._PINNED_REFS


def test_the_cap_holds_an_incumbent_plus_a_full_round():
    """The constant must be sized from the measured blob, not the mt10 one."""
    need = 9 * BLOB_MIB * 1024 * 1024
    assert training._POLICY_STORE_MAX_BYTES >= need, (
        f"{training._POLICY_STORE_MAX_BYTES} bytes cannot hold an incumbent "
        f"plus 8 candidates of {BLOB_MIB} MiB"
    )
