"""The fasttd3 replay-prefill DECISION, on the tier that actually runs.

`tests/test_fasttd3_shared_replay.py` covers the same mechanism end to end, and
every one of its assertions is behind `pytest.mark.slow` plus, for three of the
five, torch gating. Both are correct for what that file does -- it builds real
torch buffers -- but they narrow where it runs: under `--extra test` with
`-m "not slow"` none of its five cases runs, and its three torch cases skip
under any install without torch (`--extra fasttd3` or `--extra all` brings it
in). A skip prints as a dot.

So the `num_steps > 1` refusal's end-to-end test -- `test_an_n_step_pool_is_
refused_rather_than_summed`, torch-gated -- runs only where torch is installed
and `-m slow` is selected.

The part of the mechanism that decides WHETHER to prefill needs no torch at
all, so it is separated into `_fasttd3_prefill_plan` and tested here, unmarked
-- the only tier of this mechanism the default test selection runs. The split
exists because the `num_steps > 1` refusal must be decided before the caller
widens the buffer, and this checks that on every run.

WHAT THIS FILE CANNOT SEE, stated so the gap is not mistaken for coverage: the
lane layout, the `.permute` inverse, the row counts and the device path are all
in the other file and need a GPU. This one covers the decision and the seed
row's claim about it, and nothing else.
"""

from __future__ import annotations

import pytest

from bird.components.fasttd3 import _fasttd3_prefill_plan


class _Handoff:
    """The two attributes the plan reads off a `shared_population` slice."""

    def __init__(self, ref, index=0):
        self.prefill_ref = ref
        self.index = index


@pytest.fixture
def pool(monkeypatch):
    """A round store holding one non-empty pool under the key 'round:c0000'."""
    from bird.components import training
    rows = [object(), object(), object()]
    monkeypatch.setitem(training._ROUND_REPLAY, "round:c0000", rows)
    return rows


def test_no_handoff_asks_for_nothing_and_is_still_supported(pool):
    assert _fasttd3_prefill_plan(1, None, None) == (None, None, True)
    assert _fasttd3_prefill_plan(1, _Handoff(None), None) == (None, None, True)


def test_a_resolvable_ref_returns_the_pool(pool):
    rows, ref, supported = _fasttd3_prefill_plan(1, _Handoff("round:c0000"), None)
    assert rows is pool and ref == "round:c0000" and supported is True


def test_n_step_is_a_REFUSAL_and_says_so_rather_than_returning_an_empty_pool(pool):
    """The defect this test exists for.

    `num_steps > 1` cannot be honoured: the n-step window masks on `dones`
    alone, so a window straddling pooled rows sums rewards from unrelated
    transitions and returns the total as an n-step return. Refusing it inside
    `_fasttd3_prefill_replay`, per seed, would be AFTER the caller has widened
    `buffer_rows` for a pool that will never be written, and with the seed row
    still reading `replay_prefill_supported: True`.

    Two distinct properties are asserted, because only the first is obvious.
    `supported is False` is the row's claim. `rows is None` is what stops the
    widening: the caller widens under `if prefill_rows is not None`, so
    returning None is the mechanism, not a detail. A future edit that returned
    an empty list instead would satisfy "no rows" and still widen the buffer.
    """
    rows, ref, supported = _fasttd3_prefill_plan(3, _Handoff("round:c0000"), None)
    assert supported is False, "an n-step configuration cannot honour a pool"
    assert rows is None, "None is what suppresses the buffer widening upstream"
    assert ref is None, "no ref means the seed row's replay_prefill_ref stays ''"
    # and it is a property of num_steps, not of the store: the pool IS there
    assert _fasttd3_prefill_plan(1, _Handoff("round:c0000"), None)[2] is True


@pytest.mark.parametrize("num_steps", [2, 3, 8])
def test_every_n_step_length_above_one_refuses(num_steps, pool):
    assert _fasttd3_prefill_plan(num_steps, _Handoff("round:c0000"), None)[2] is False


def test_the_two_CIRCUMSTANTIAL_misses_stay_supported(pool):
    """`supported` is a statement about the CONFIGURATION, not about what
    happened -- the meaning it carries on sb3, where it is `bool(off_policy)`.

    A ref that is not in the round store, and a co-designed observation whose
    features the pooled raw states do not match, are both "nothing was pooled
    this time" rather than "this learner cannot pool". They keep
    `supported: True` and are read off `replay_prefill_*: 0`. Collapsing the
    two would make the refusal unreadable again, from the other direction.
    """
    absent = _fasttd3_prefill_plan(1, _Handoff("round:nosuch"), None)
    assert absent == (None, "round:nosuch", True)

    with_phi = _fasttd3_prefill_plan(1, _Handoff("round:c0000"), phi=object())
    assert with_phi == (None, "round:c0000", True)


def test_an_empty_pool_is_not_a_prefill(monkeypatch):
    from bird.components import training
    monkeypatch.setitem(training._ROUND_REPLAY, "round:empty", [])
    assert _fasttd3_prefill_plan(1, _Handoff("round:empty"), None) == (
        None, "round:empty", True)
