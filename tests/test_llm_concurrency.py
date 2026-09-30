"""`llm.max_concurrent_requests` -- a transport schedule, never a method key.

The Messages API has no `n`, so K i.i.d. samples are K round-trips; this key
sets how many of them one sampling call may have in flight at once. Like
`train.candidate_parallelism`, flipping it must not move a number: samples stay
ordered by index, the budget counts the same calls and tokens, and the mock
provider -- whose per-sample RNG draws consume the run's stream in order --
never fans out at all (`LLMClient.concurrent_samples` is False there whatever
the key says).

What it buys: serial LLM round-trips dominate an iteration's wall clock
while stage-1 generation runs -- 16 serial round-trips per iteration during
which every worker core idles.
"""

import itertools
import threading
import time

import pytest

from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.llm.base import LLMClient, chunk_sizes


KEY = "llm.max_concurrent_requests"


class _Probe(LLMClient):
    """A provider stub: records every `_complete(n)` it receives, answers by
    global sample index so reassembly order is visible in the output."""

    provider = "probe"

    def __init__(self, ctx, role="generator"):
        super().__init__(ctx, role)
        self.calls = []
        self._i = 0
        self._i_lock = threading.Lock()

    def _complete(self, messages, n, temperature, images, tag):
        self.calls.append(n)
        out = []
        for _ in range(n):
            with self._i_lock:
                i = self._i
                self._i += 1
            self._record(prompt_tokens=10, completion_tokens=5)
            out.append(f"sample-{i}")
        return out


def _ctx(**overrides):
    cfg = load("eureka", profile="tester", overrides=overrides)
    return Context(cfg=cfg, budget=Budget())


def test_serial_providers_keep_their_chunk_sequence():
    """The key set high on a provider that cannot fan out (the mock's shape)
    changes NOTHING: same chunk pattern, same order -- and therefore the same
    RNG draw sequence for a stateful sample generator."""
    probe = _Probe(_ctx(**{"llm.max_concurrent_requests": 8,
                           "generate.sampling.chunk_size": 4}))
    assert probe.concurrent_samples is False  # the base default
    out = probe("write a reward", n=10)
    assert probe.calls == chunk_sizes(10, 4)
    assert out == [f"sample-{i}" for i in range(10)]


def test_concurrent_providers_get_the_whole_n_in_one_call():
    """A fanning provider receives one `_complete(n)`: chunking would serialise
    the waves the fan-out exists to overlap, and means nothing to an API where
    every sample is its own round-trip regardless."""
    probe = _Probe(_ctx(**{"llm.max_concurrent_requests": 8,
                           "generate.sampling.chunk_size": 4}))
    probe.concurrent_samples = True
    out = probe("write a reward", n=10)
    assert probe.calls == [10]
    assert out == [f"sample-{i}" for i in range(10)]


def test_concurrency_of_one_is_byte_for_byte_the_serial_loop():
    """Set to 1, a FANNING provider behaves exactly like a serial one: the same
    chunk sequence, the same order, no subtle difference.

    The value is set explicitly rather than taken from the default (8, in
    `configs/_default.yaml`), so this pins what 1 MEANS
    rather than who happens to choose it: a change of default policy must not
    delete a behavioural guarantee that has nothing to do with policy.
    """
    probe = _Probe(_ctx(**{"llm.max_concurrent_requests": 1}))
    probe.concurrent_samples = True
    assert probe.max_concurrent_requests == 1
    probe("write a reward", n=6)
    assert probe.calls == chunk_sizes(6, probe.chunk_size)


def test_the_default_overlaps_and_the_mock_still_does_not():
    """The other half, stated as policy rather than smuggled into the one above.

    8, not 1: the RDA judge block is 108 serial round-trips per iteration on the
    HumanoidBench tier, and serial judge round-trips dominate an iteration's wall
    clock. `configs/_default.yaml` states it as a fixed ceiling -- a stated bound,
    not a self-tuning one.

    Asserted with the MOCK beside it because that is the pairing that matters: a
    higher default is only safe while the provider whose RNG draws must stay in
    order refuses to fan out regardless. If `concurrent_samples` ever became True
    on the mock, every tester-tier determinism claim would move with it, and this
    is where that shows up.
    """
    from bird.llm.mock import MockLLM

    cfg_default = _Probe(_ctx()).max_concurrent_requests
    assert cfg_default == 8, (
        f"the shipped default is {cfg_default}, not 8 -- update this test with it "
        "and the comment in `configs/_default.yaml`")
    assert MockLLM.concurrent_samples is False, (
        "the mock now fans out, so `llm.max_concurrent_requests` reorders its "
        "per-sample RNG draws and the tester tier is no longer deterministic")


class _CountingLock:
    """A lock that records how many times it was ENTERED.

    The point of the indirection: asserting on counter exactness cannot show
    that `_record` locks, because on a GIL build it is exact either way (see
    `test_record_takes_the_lock_on_every_call`). Entry count can, and is
    deterministic -- so the guard is pinned by a fact about the code rather
    than by a race the interpreter may decline to produce.
    """

    def __init__(self) -> None:
        self._inner = threading.Lock()
        self.entries = 0

    def __enter__(self):
        self._inner.acquire()
        self.entries += 1          # under the lock: this count is exact
        return self

    def __exit__(self, *exc) -> bool:
        self._inner.release()
        return False


def test_record_takes_the_lock_on_every_call():
    """`_record` must serialise, and THIS is the assertion that can fail.

    `Budget.record_llm` is a run of unsynchronised `+=`. That is a real race in
    principle -- and a free-threaded build removes the only thing standing in
    its way -- so the lock stays. What cannot be asserted is a lost increment:
    measured on CPython 3.11.15 (GIL, 5 ms switch interval), 16 threads x
    20,000 unlocked `record_llm` calls lost **zero**, because CPython prefers to
    drop the GIL at the `self._check()` call boundary rather than between an
    attribute's LOAD and STORE. A test asserting counter exactness at 16 x 500
    therefore passes with the lock removed -- it proves nothing about the guard.

    So assert the mechanism: every `_record` enters `_record_lock` exactly once.
    Delete the `with` and this goes red at the first call.
    """
    probe = _Probe(_ctx())
    probe._record_lock = _CountingLock()
    n_threads, per = 16, 500

    def worker():
        for _ in range(per):
            probe._record(prompt_tokens=1, completion_tokens=1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert probe._record_lock.entries == n_threads * per, (
        "_record did not take _record_lock on every call")
    # The counters must also be exact. On this build they would be even without
    # the lock, so this is a regression guard on the arithmetic, not on the
    # guard -- the assertion above is the one that pins the lock.
    assert probe.ctx.budget.llm_calls == n_threads * per
    assert probe.ctx.budget.llm_prompt_tokens == n_threads * per
    assert probe.n_calls == n_threads * per


# -- the fan-out's failure path -------------------------------------------
#
# `_complete`'s leader-then-followers block is the only place a sampling call
# can fail PARTWAY.


def _fanout_client(max_concurrent_requests, fail_at=None, delay=0.0):
    """A real `AnthropicClient` with `__init__` bypassed and `_one` replaced.

    Only the fan-out block is under test, so the SDK, the key and the transport
    are all out of scope -- what is hand-set is exactly what that block reads.
    """
    from bird.llm.anthropic_client import AnthropicClient

    obj = object.__new__(AnthropicClient)
    obj.modality, obj.role = "text", "generator"
    obj.max_concurrent_requests = max_concurrent_requests
    obj._record_lock = threading.Lock()
    obj.started = []
    counter, guard = itertools.count(), threading.Lock()

    def one(system, turns, temperature, n_images):
        with guard:
            i = next(counter)
            obj.started.append(i)
        if fail_at is not None and i == fail_at:
            raise RuntimeError(f"follower {i} failed")
        time.sleep(delay)
        return f"sample-{i}"

    obj._one = one
    return obj


_MSG = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("workers", [1, 4, 8])
def test_fanout_returns_samples_in_submission_order(workers, monkeypatch):
    """Reassembly is by index, never completion -- the `as_completed` trap.

    `select.tie_break: first` decides a real share of multi-candidate rounds, so
    a reordered sampling call moves selected reward functions while every fitness
    stays identical. A test comparing
    fitness would not see it; this one would.

    Each sample is labelled by its SUBMISSION index (the pool's `submit` is
    wrapped to hand it over), never by the order its thread happened to start:
    labelling by start order would make the expected list itself racy on a
    loaded machine. Later submissions sleep less, so completion order is the REVERSE of
    submission order and a reassembly by completion would fail here.
    """
    import bird.llm.anthropic_client as AC

    n = 8
    slot = threading.local()
    real = AC.ThreadPoolExecutor

    class _Labelled(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._submitted = itertools.count(1)   # index 0 is the leader

        def submit(self, fn, *args, **kwargs):
            idx = next(self._submitted)

            def labelled():
                slot.idx = idx
                return fn(*args, **kwargs)
            return super().submit(labelled)

    monkeypatch.setattr(AC, "ThreadPoolExecutor", _Labelled)
    client = _fanout_client(workers)

    # Calls on the calling thread (the leader, or every call on the serial
    # path at `workers=1`) are labelled in call order.
    in_caller = itertools.count()

    def one(system, turns, temperature, n_images):
        idx = slot.idx if hasattr(slot, "idx") else next(in_caller)
        time.sleep(0.002 * (n - idx))
        return f"sample-{idx}"

    client._one = one
    out = client._complete(_MSG, n, 0.0, None, "t")
    assert out == [f"sample-{i}" for i in range(n)]


def test_a_failed_follower_cancels_the_ones_still_queued():
    """A raise keeps as much of its serial meaning as a pool allows.

    Serial stops at the first failure and never sends the rest. The fan-out
    cannot un-send what is already in flight, but it CAN decline to start what
    is still queued -- so a failure must not cost `n - 1` requests' worth of
    tokens.

    What this does NOT pin, measured rather than assumed: `pool.map` satisfies
    it too. `Executor.map`'s result iterator has a `finally: for future in fs:
    future.cancel()`, so an early raise already cancels the queue -- 3 of 12
    started under `pool.map` with a failing first item, same as here. Explicit
    futures buy noticing a LATE follower's failure without
    draining earlier results in submission order first, which is a narrower win
    than "queued work is cancelled" and is not what this test measures.

    What it does pin is the property itself, against a refactor that drops
    cancellation entirely -- `[f.result() for f in futures]` with no cancel
    sends all n and turns this red (verified).
    """
    n = 12
    # fail_at=1, NOT 0: call 0 is the LEADER, which runs alone before the pool
    # opens, so failing it would raise before a single follower was submitted
    # and would prove nothing about cancellation.
    client = _fanout_client(2, fail_at=1, delay=0.2)
    with pytest.raises(RuntimeError, match="follower 1 failed"):
        client._complete(_MSG, n, 0.0, None, "t")
    # `started` counts the leader plus every follower that was actually sent, so
    # n means "all of them ran". Under `pool.map` it is exactly n and this fails.
    assert len(client.started) < n, (
        f"every request was sent ({len(client.started)} of {n}); "
        "queued followers were not cancelled")


def test_the_serial_path_is_taken_when_the_key_is_default():
    """`max_concurrent_requests: 1` never opens a pool at all."""
    client = _fanout_client(1)
    out = client._complete(_MSG, 4, 0.0, None, "t")
    assert out == [f"sample-{i}" for i in range(4)]
    assert client.started == [0, 1, 2, 3]


def test_retry_after_is_jittered_and_the_spread_follows_the_fanout(monkeypatch):
    """A `retry-after` is the rate-limit case, so it is the one every in-flight
    follower receives at once -- and without jitter on that branch, K
    followers wake in the same millisecond and re-hit the limit together.
    Serial cannot reach that state.

    Two properties, and the first is the safety one: the delay is never SHORTER
    than the server asked. The second is the spread: it scales with
    `llm.max_concurrent_requests`, so widening the fan-out widens the window it
    is spread over instead of concentrating the herd.
    """
    from bird.llm import anthropic_client as ac

    def delays_for(mcr, n=40, retry_after=5.0):
        client = _fanout_client(mcr)
        seen = []
        monkeypatch.setattr(ac.time, "sleep", seen.append)
        for _ in range(n):
            client._sleep(0, retry_after)
        return seen

    serial = delays_for(1)
    assert all(d >= 5.0 for d in serial), "slept for less than retry-after"
    assert max(serial) <= 6.0, "serial spread is wider than uniform(0, 1)"
    assert max(serial) > 5.0, "retry-after branch is not jittered at all"

    wide = delays_for(16)
    assert all(d >= 5.0 for d in wide)
    assert max(wide) - min(wide) > max(serial) - min(serial), (
        "the spread does not widen with max_concurrent_requests")
