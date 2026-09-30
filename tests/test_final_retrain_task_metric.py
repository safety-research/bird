"""`final_retrain.json` must record the per-seed TASK METRIC, not only the
arm's fitness and the native signal.

The artifact carries two other per-seed lists, and neither is `env.task_metric`:

  `per_seed_fitness`  the ARM'S OWN fitness source -- a VLM score for a
                      VLM-judged method such as rda, nothing at all for CARD's
                      `none`.
  `per_seed_native`   the env's native signal, which on Assistax resolves to
                      `native_reward`, i.e. the reference-reward return.

Every search-time figure is on `env.task_metric`, so a retrained table built
from those two lists alone would compare arms on a different quantity than the
one the search reported. The three numbers are not close: the real cell below
scored `per_seed_fitness` 0.387 and `retrained_native` 2769 while its per-seed
task metric was [0.0, 0.058, 0.009].

NOTHING NEW IS MEASURED. `env.task_metric` is already evaluated in the greedy
checkpoint eval and rides home on the TrainResult as
`seed_metrics[*]["fitness"]` -- this is a recording question, not a
measurement one.

THE NAME IS THE TRAP: a seed row's `fitness` key IS the task metric
(`training.py`'s `_PRUNING_FIELDS` maps `task_metric` -> `fitness`), while
`CandidateReport.per_seed_fitness` is the arm's fitness. One word apart.

The fixture is REAL, not invented: the seed rows below are the three from a
real final retrain of the `v4_peak_noes40` hill-climb configuration on
`assistax_feeding`, beside the same cell's real `final_retrain.json` values. A
fixture written by the same hand as the reader agrees with it by construction;
this one does not.
"""
from collections.abc import Mapping
import logging
import math

from bird.components.phases import _per_seed_task_metric
from bird.types import Candidate, TrainResult


# --- the real cell ---------------------------------------------------------
# v4_peak_noes40 on assistax_feeding.
REAL_SEED_ROWS = [
    {"seed": 12922312, "fitness": 0.0, "max": 0.0, "auc": 0.0,
     "shipped_checkpoint": 19, "env_steps": 1059936, "error": ""},
    {"seed": 12930231, "fitness": 0.058, "max": 0.058, "auc": 0.0116,
     "shipped_checkpoint": 19, "env_steps": 1059936, "error": ""},
    {"seed": 12938150, "fitness": 0.009333333333333334, "max": 0.014,
     "auc": 0.0042, "shipped_checkpoint": 19, "env_steps": 1059936, "error": ""},
]
# the SAME cell's phases/final_retrain.json, as written at the time
REAL_PER_SEED_FITNESS = [0.38690476190476186]        # the arm's VLM score
REAL_PER_SEED_NATIVE = [2692.354402116863,           # native_reward returns
                        2287.3766105860354,
                        3327.4183177215837]


def _result(rows):
    cand = Candidate(cand_id="c0005", iteration=0, reward_code="def r(s, a): return 0.0")
    r = TrainResult(cand_id=cand.cand_id, candidate=cand)
    r.seed_metrics = list(rows)
    return r


def test_the_task_metric_is_read_off_the_real_seed_rows():
    """The whole contract, on the real artifact: one value per seed, in order."""
    got = _per_seed_task_metric(_result(REAL_SEED_ROWS))
    assert got == [0.0, 0.058, 0.009333333333333334], got
    assert len(got) == len(REAL_SEED_ROWS) == 3


def test_it_is_a_different_quantity_from_the_two_already_recorded():
    """The reason the field exists. If any of these three coincided, recording
    the task metric separately would be redundant -- they do not, and the
    arm's fitness is not even the same LENGTH as the seed list."""
    task = _per_seed_task_metric(_result(REAL_SEED_ROWS))

    # 1. not the arm's own fitness: different length AND different values
    assert len(task) != len(REAL_PER_SEED_FITNESS), (task, REAL_PER_SEED_FITNESS)
    assert not any(math.isclose(t, REAL_PER_SEED_FITNESS[0], rel_tol=1e-6) for t in task)

    # 2. not the native signal: same length here, wildly different scale --
    #    the task metric is a [0, 1] fraction, native_reward is a return
    assert len(task) == len(REAL_PER_SEED_NATIVE)
    assert all(0.0 <= t <= 1.0 for t in task), task
    assert all(n > 100.0 for n in REAL_PER_SEED_NATIVE), REAL_PER_SEED_NATIVE

    # 3. and the ORDERING disagrees, which is what would silently flip a
    #    per-seed comparison: native ranks seed 2 best, the task metric ranks
    #    seed 1 best.
    assert task.index(max(task)) != REAL_PER_SEED_NATIVE.index(max(REAL_PER_SEED_NATIVE))


def test_a_missing_value_is_none_and_never_a_zero_score():
    """A seed that recorded no value must not read as having scored 0.0 --
    that is the difference between "did not finish" and "failed the task",
    and `.fitness is None` vs `0.0` is a distinction the whole repo keeps."""
    rows = [{"seed": 1, "fitness": 0.5}, {"seed": 2}, {"seed": 3, "fitness": None}]
    assert _per_seed_task_metric(_result(rows)) == [0.5, None, None]


def test_it_never_fails_the_phase_it_decorates():
    """A retrain that produced no result, and a malformed row, both yield a
    value rather than raising: this decorates the report protocol and must not
    be able to break it (same contract as `_score_native`)."""
    assert _per_seed_task_metric(None) == []
    assert _per_seed_task_metric(_result([])) == []
    empty = _result([])
    empty.seed_metrics = None
    assert _per_seed_task_metric(empty) == []
    assert _per_seed_task_metric(_result(["not a dict", 7])) == [None, None]


def test_the_length_is_the_seeds_that_ran():
    """Length is load-bearing: `per_seed_fitness` on the real cell has length
    1 for a 3-seed retrain, so a reader cannot tell a collapsed list from a
    1-seed run."""
    for n in (1, 3, 5):
        rows = [{"seed": i, "fitness": float(i) / 10} for i in range(n)]
        assert len(_per_seed_task_metric(_result(rows))) == n


def test_a_wrapped_payload_is_distinguishable_from_no_result(caplog):
    """`[]` must not mean two things at once.

    A forked training child returns `{"result": TrainResult, "meta": ...}`
    (`training.build_child_payload`). Because
    `getattr(a_dict, "seed_metrics", None)` is None, that envelope would fall
    through `or []` and produce the SAME `[]` that means "the retrain
    produced nothing" -- silently. The phase itself is not affected (it
    holds the unwrapped result), but any collector reusing this helper would
    read a wrapper as an empty retrain with nothing to tell it apart.

    The helper logs rather than raises, because the contract in the docstring is
    that this never raises: it decorates a retrain and must not be able to
    fail it. So the LOG LINE is the whole distinguishing signal, and that is
    what this asserts -- not merely that both return `[]`, which they still
    do and must.
    """
    inner = _result([{"seed": 0, "fitness": 0.5}])

    with caplog.at_level(logging.WARNING, logger="bird"):
        assert _per_seed_task_metric(None) == []
    assert "wrapped payload" not in caplog.text, \
        "a genuine no-result must stay quiet, or the warning means nothing"

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="bird"):
        assert _per_seed_task_metric({"result": inner, "meta": {}}) == []
    assert "wrapped payload, unwrap first" in caplog.text, \
        "the wrapper case must say so; [] alone cannot distinguish the two states"

    # And the real thing still works, which is the case a guard like this is
    # most likely to break.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="bird"):
        assert _per_seed_task_metric(inner) == [0.5]
    assert "wrapped payload" not in caplog.text


def test_no_path_returns_an_empty_list_silently(caplog):
    """Three states share one value; each says which it is.

    `[]` means "no result", "wrapped payload" or "the read itself raised". A
    malformed seed row raising mid-loop returning `[]` without a word would be
    a gap the same size as the one the wrapper guard closes.

    An exotic Mapping is the case that motivated moving the guard inside the
    `try`: `isinstance` cannot raise, but a mapping's `__contains__` and
    `__getitem__` are arbitrary code, and a guard that can fail the retrain is
    the opposite of one that decorates it.
    """
    # NOT a Mapping: any Mapping is caught by the envelope guard before the
    # read is attempted, so a hostile Mapping cannot reach the except path at
    # all. The probe has to be something the function will genuinely try to
    # read -- an object whose `seed_metrics` blows up on access.
    class Hostile:
        @property
        def seed_metrics(self):        raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING, logger="bird"):
        assert _per_seed_task_metric(Hostile()) == []
    assert "the read itself failed" in caplog.text, \
        "a raising input must say so; [] alone cannot distinguish it from no result"
    assert "RuntimeError" in caplog.text

    # A malformed seed row raising mid-loop.
    caplog.clear()
    class BadRow(dict):
        def get(self, *a, **k):        raise ValueError("bad row")
    with caplog.at_level(logging.WARNING, logger="bird"):
        assert _per_seed_task_metric(_result([BadRow()])) == []
    assert "the read itself failed" in caplog.text
    assert "ValueError" in caplog.text

    # And the retrain is never failed: every case above returned rather than raised.


def test_an_envelope_whose_result_is_null_is_still_an_envelope(caplog):
    """`{"result": None, ...}` is what a FAILED cell's envelope looks like.

    A guard that probes a key is wrong: it asks what is INSIDE a thing the
    type already identifies. `"result" in result` can raise on an exotic
    Mapping; `.get("result") is not None` silently misses this case, letting a
    plain dict fall through to the `getattr` below and return `[]` with no log
    -- the silent state the guard exists to abolish.

    `TrainResult` is a dataclass and not a Mapping, so ANY mapping arriving
    here is the envelope and the type alone settles it.
    """
    with caplog.at_level(logging.WARNING, logger="bird"):
        assert _per_seed_task_metric({"result": None, "meta": {}}) == []
    assert "wrapped payload, unwrap first" in caplog.text, \
        "a null-result envelope is still an envelope and must say so"

    # An empty mapping too: nothing to probe, still not a TrainResult.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="bird"):
        assert _per_seed_task_metric({}) == []
    assert "wrapped payload, unwrap first" in caplog.text
