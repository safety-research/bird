"""The Anthropic transport, offline, with the defaults a run actually uses.

`bird/llm/anthropic_client.py` is the only code path in the repo that talks to a
real model, and a transport that is broken for EVERY live config need not turn a
single other test red. Two individually reasonable facts are fatal together:

  * `DEFAULT_MAX_TOKENS` is 32,768 rather than 8,192, because at 8,192 half the
    Meta-World samples are cut mid-program; and
  * `anthropic` 1.0.0's `Messages.create()` refuses a non-streaming request
    whose output cap implies it might run over ten minutes:
    `expected = 3600 * max_tokens / 128_000 > 600`, i.e. any
    `max_tokens > 21_333`, raised as a client-side `ValueError` before a socket
    is opened.

A non-streaming transport with that cap is unusable, and the failure is
invisible to everything else that guards this repo: `--validate-all` never
calls a model, and a probe that calls one with `max_tokens=1` checks the
credential and not the request shape, so it passes on a build whose every job
would die in `generate` of iteration 1. That is what
these tests are for: they exercise the transport with the DEFAULTS A RUN
ACTUALLY USES, offline.
"""

from __future__ import annotations

import sys
import itertools
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird.llm.anthropic_client import (  # noqa: E402
    BACKOFF_BASE_S,
    BACKOFF_MAX_S,
    DEFAULT_MAX_TOKENS,
    MAX_RETRIES,
    AnthropicClient,
)
from bird.llm.base import LLMError  # noqa: E402


class _FakeMessage:
    def __init__(self) -> None:
        self.content = [types.SimpleNamespace(type="text", text="ok")]
        self.usage = types.SimpleNamespace(input_tokens=10, output_tokens=3)
        self.stop_reason = "end_turn"


class _FakeStreamManager:
    def __init__(self, message: _FakeMessage) -> None:
        self._message = message

    def __enter__(self):
        return types.SimpleNamespace(get_final_message=lambda: self._message)

    def __exit__(self, *exc):
        return False


class _FakeMessages:
    """A `client.messages` that models the SDK's non-streaming precondition."""

    def __init__(self, reject: tuple[str, ...] = ()) -> None:
        self.reject = reject
        self.create_calls = 0
        self.stream_calls: list[dict] = []

    def create(self, **params):
        self.create_calls += 1
        # The real precondition, copied from anthropic/_base_client.py so the
        # fake fails the same way the SDK does rather than in a way we invented.
        if 3600 * int(params["max_tokens"]) / 128_000 > 600:
            raise ValueError(
                "Streaming is required for operations that may take longer "
                "than 10 minutes.")
        return _FakeMessage()

    def stream(self, **params):
        for key in self.reject:
            if key in params:
                raise TypeError(
                    f"Messages.stream() got an unexpected keyword argument '{key}'")
        self.stream_calls.append(dict(params))
        return _FakeStreamManager(_FakeMessage())


def _fake_sdk() -> types.SimpleNamespace:
    class _Err(Exception):
        pass

    return types.SimpleNamespace(
        BadRequestError=type("BadRequestError", (_Err,), {}),
        RateLimitError=type("RateLimitError", (_Err,), {}),
        APIStatusError=type("APIStatusError", (_Err,), {}),
        APIConnectionError=type("APIConnectionError", (_Err,), {}),
        APITimeoutError=type("APITimeoutError", (_Err,), {}),
    )


def _client(reject: tuple[str, ...] = ()) -> tuple[AnthropicClient, _FakeMessages]:
    """An `AnthropicClient` with the SDK injected and `__init__` bypassed.

    `__init__` reads a `Context` and connects; neither is needed to exercise
    `_call`, and requiring them would make this test need a config and a key.
    """
    obj = object.__new__(AnthropicClient)
    messages = _FakeMessages(reject)
    # `__init__` is bypassed, so hand-set what `_call` now touches: `_dropped`
    # snapshots and `_degrade` mutations are serialised under `_record_lock`
    # since the fan-out path exists.
    obj._record_lock = threading.Lock()
    obj._sdk = _fake_sdk()
    obj._client = types.SimpleNamespace(messages=messages)
    obj._dropped = set()
    obj.model = "claude-opus-5"
    obj.max_tokens = DEFAULT_MAX_TOKENS
    return obj, messages


def _params(**extra) -> dict:
    p = {"model": "claude-opus-5", "max_tokens": DEFAULT_MAX_TOKENS,
         "messages": [{"role": "user", "content": "hi"}]}
    p.update(extra)
    return p


def test_the_default_output_cap_exceeds_the_nonstreaming_ceiling():
    """The reason streaming is mandatory, asserted rather than asserted-in-prose.

    If `DEFAULT_MAX_TOKENS` is lowered under the ceiling this test fails and
    points at the truncation measurement that set it; if the ceiling is raised
    this test fails and points here. Either way the two numbers cannot drift
    apart silently.
    """
    ceiling = 600 * 128_000 // 3600          # 21,333 output tokens
    assert DEFAULT_MAX_TOKENS > ceiling, (
        f"DEFAULT_MAX_TOKENS={DEFAULT_MAX_TOKENS} now fits under the SDK's "
        f"non-streaming ceiling of {ceiling}. That is not wrong, but the "
        "docstring in anthropic_client.py::_call explains the cap as the reason "
        "for streaming -- update it, or restore the cap that the Meta-World "
        "truncation measurement asked for.")


def test_the_transport_streams_and_never_calls_create():
    """`messages.create` at the shipped cap is a `ValueError`, not a request."""
    client, messages = _client()
    out = client._call(_params())
    assert out.stop_reason == "end_turn"
    assert messages.create_calls == 0, (
        "the transport used messages.create, which the SDK refuses outright at "
        "DEFAULT_MAX_TOKENS -- every live run dies in generate of iteration 1")
    assert len(messages.stream_calls) == 1
    assert messages.stream_calls[0]["max_tokens"] == DEFAULT_MAX_TOKENS


def test_a_pin_the_stream_signature_rejects_is_still_dropped_once():
    """`stream()` has no `temperature` either, so `_degrade` must still fire.

    This is the interaction that hid the bug: attempt 1 died on `temperature`,
    so the `max_tokens` `ValueError` only surfaced on attempt 2 and the
    traceback named neither `max_tokens` nor the config key behind it.
    """
    client, messages = _client(reject=("temperature",))
    out = client._call(_params(temperature=1.0))
    assert out.stop_reason == "end_turn"
    assert "temperature" in client._dropped
    assert len(messages.stream_calls) == 1
    assert "temperature" not in messages.stream_calls[0]


def test_the_installed_sdk_really_does_refuse_the_shipped_cap():
    """The fake above copies a precondition; this checks it against the SDK.

    Offline -- `_calculate_nonstreaming_timeout` is pure arithmetic and opens no
    socket. Skipped where the optional extra is absent, which is the same
    machine on which nothing calls a real model anyway.
    """
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(api_key="not-a-real-key")
    with pytest.raises(ValueError, match="Streaming is required"):
        client._calculate_nonstreaming_timeout(DEFAULT_MAX_TOKENS, None)


# ==========================================================================
# Transport errors that escape the SDK's own wrapping
#
# `_call` streams. The SDK maps httpx failures onto `APIConnectionError` at the
# REQUEST layer, but a stream that dies while `get_final_message()` iterates it
# raises the RAW transport exception out of the generator. Observed on real
# runs: searches dying exactly this way, with `ReadTimeout` and
# `RemoteProtocolError`, each throwing away hours of RL that was already paid
# for. Raising MAX_RETRIES does nothing for them, because the retry COUNT is
# not the problem -- the exception type is outside every `except`.
# ==========================================================================


class _DyingStreamManager:
    """A stream that opens fine and fails while being read.

    The distinction is the whole point: an exception from `stream(...)` itself
    is a request-layer failure the SDK already wraps. This one is raised from
    `get_final_message()`, which is where the real ones came from.
    """

    def __init__(self, exc: BaseException | None) -> None:
        self._exc = exc

    def __enter__(self):
        def _final():
            if self._exc is not None:
                raise self._exc
            return _FakeMessage()
        return types.SimpleNamespace(get_final_message=_final)

    def __exit__(self, *exc):
        return False


def _transport_sdk():
    """A fake SDK carrying a `_base_client.httpx2`, as `anthropic` 1.0.0 does."""
    sdk = _fake_sdk()

    class TransportError(Exception):
        pass

    class ReadTimeout(TransportError):
        pass

    sdk._base_client = types.SimpleNamespace(
        httpx2=types.SimpleNamespace(TransportError=TransportError))
    return sdk, ReadTimeout


def test_a_stream_that_dies_mid_read_is_retried_not_fatal():
    obj, _ = _client()
    sdk, ReadTimeout = _transport_sdk()
    obj._sdk = sdk
    calls = {"n": 0}

    def stream(**params):
        calls["n"] += 1
        # Fail the first two reads, then succeed -- a transient blip, which is
        # what these are: the SAME request succeeds moments later.
        return _DyingStreamManager(ReadTimeout("read timed out") if calls["n"] <= 2 else None)

    obj._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    obj._sleep = lambda *a, **k: None  # no real backoff in a unit test

    msg = obj._call(_params())
    assert msg.content[0].text == "ok"
    assert calls["n"] == 3, "the two dead streams should have been retried, not raised"


def test_a_stream_that_never_recovers_still_ends_as_LLMError():
    """Retrying must not turn a permanent transport fault into a hang or a
    raw httpx traceback -- it stays a bounded, named failure."""
    obj, _ = _client()
    sdk, ReadTimeout = _transport_sdk()
    obj._sdk = sdk

    def stream(**params):
        return _DyingStreamManager(ReadTimeout("read timed out"))

    obj._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    obj._sleep = lambda *a, **k: None

    with pytest.raises(LLMError, match="after .* retries"):
        obj._call(_params())


def test_the_transport_module_is_discovered_not_hardcoded():
    """`anthropic` 1.0.0 imports `httpx2`, not `httpx`, and a bare
    `import httpx` raises ImportError in this venv. A hardcoded name would
    break silently the day the SDK moves, so the lookup tries both and an
    unrecognised SDK degrades to an empty tuple -- a legal `except` target
    that never matches, i.e. exactly the old behaviour."""
    err2 = type("TransportError", (Exception,), {})
    sdk2 = types.SimpleNamespace(
        _base_client=types.SimpleNamespace(
            httpx2=types.SimpleNamespace(TransportError=err2)))
    assert AnthropicClient._transport_errors(sdk2) == (err2,)

    err1 = type("TransportError", (Exception,), {})
    sdk1 = types.SimpleNamespace(
        _base_client=types.SimpleNamespace(
            httpx=types.SimpleNamespace(TransportError=err1)))
    assert AnthropicClient._transport_errors(sdk1) == (err1,)

    assert AnthropicClient._transport_errors(types.SimpleNamespace()) == ()
    empty = types.SimpleNamespace(_base_client=types.SimpleNamespace())
    assert AnthropicClient._transport_errors(empty) == ()


def test_the_installed_sdk_really_does_leak_transport_errors():
    """The premise, checked against the SDK rather than assumed.

    If a future SDK starts wrapping mid-stream failures into
    `APIConnectionError`, this fails and the extra clause becomes dead code
    that should be removed -- which is the point of pinning it.

    Offline: only class objects are inspected, no socket is opened.
    """
    anthropic = pytest.importorskip("anthropic")
    found = AnthropicClient._transport_errors(anthropic)
    assert found, "no TransportError located on the installed SDK"
    transport = found[0]
    assert not issubclass(transport, anthropic.APIConnectionError), (
        "the SDK now wraps transport errors; the dedicated except clause in "
        "_call is redundant and should be deleted")
    http = anthropic._base_client.httpx2
    for name in ("ReadTimeout", "RemoteProtocolError"):
        assert issubclass(getattr(http, name), transport), (
            f"{name} -- observed killing real runs -- is no longer caught")


# ==========================================================================
# A mid-stream `error` event (`_retryable_status`)
#
# A plain `overloaded_error` can end a search with ZERO retries, because the
# exception a mid-stream SSE error builds carries the STREAMING response -- 200
# OK -- so a `status_code >= 500` test reads the transport's status and not the
# error's. `MAX_RETRIES` is then never consulted, which is why raising it does
# nothing (the same shape as the transport-error case above).
# ==========================================================================


def _status_error(cls, *, status: int, err_type: str | None,
                  retry_after: str | None = None):
    """An `APIStatusError`-alike carrying what the real one carries.

    `status_code` and `type` are set exactly as `APIStatusError.__init__` sets
    them -- from the response, and from `body["error"]["type"]` respectively --
    which is the pairing `_retryable_status` turns on.
    """
    exc = cls("boom")
    exc.status_code = status
    exc.type = err_type
    exc.response = types.SimpleNamespace(
        status_code=status,
        headers={"retry-after": retry_after} if retry_after else {})
    return exc


def test_a_mid_stream_overload_is_retried_not_fatal():
    """The regression. Status 200, type `overloaded_error` -- transient."""
    obj, _ = _client()
    calls = {"n": 0}

    def stream(**params):
        calls["n"] += 1
        if calls["n"] <= 2:
            return _DyingStreamManager(_status_error(
                obj._sdk.APIStatusError, status=200,
                err_type="overloaded_error"))
        return _DyingStreamManager(None)

    obj._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    obj._sleep = lambda *a, **k: None

    msg = obj._call(_params())
    assert msg.content[0].text == "ok"
    assert calls["n"] == 3, (
        "a mid-stream overload reads as status 200; it must be retried on its "
        "error TYPE, not raised because the transport succeeded")


def test_the_retry_ladder_rides_out_a_sustained_overload():
    """The budget, not just the classification.

    `_retryable_status` made a mid-stream `overloaded_error` retryable; this is
    the second half. A ladder of 5 retries -- ~2 minutes of tolerance -- does
    not cover a provider outage, and runs die `failed after 5 retries` late in
    a search. The ladder must span the length of outage actually observed
    (~10 min), so this asserts on the SUM rather than on
    MAX_RETRIES alone: raising the count while leaving BACKOFF_MAX_S at 60 s, or
    vice versa, both fail here.
    """
    total, delay = 0.0, BACKOFF_BASE_S
    for _ in range(MAX_RETRIES):
        total += min(delay, BACKOFF_MAX_S)
        delay *= 2
    assert total >= 600, (
        f"the ladder spans only {total:.0f}s; a sustained overload outlasts it "
        f"and ends a search holding hours of RL")
    # Not unbounded either: a stuck provider must still fail in bounded time --
    # under an hour, so what reaches the operator is LLMError carrying the
    # provider's last word, not a run that reads as hung. Deliberately not tied
    # to any config key: the ladder bounds an LLM call, and no training-stage
    # timeout governs those.
    assert total <= 3600, f"{total:.0f}s is long enough to look like a hang"


def test_the_backoff_is_capped_so_one_wait_cannot_dominate():
    """`BACKOFF_MAX_S` is what stops doubling running away: without a cap, 12
    retries would reach 2048 s on the last wait alone."""
    delay, waits = BACKOFF_BASE_S, []
    for _ in range(MAX_RETRIES):
        waits.append(min(delay, BACKOFF_MAX_S))
        delay *= 2
    assert max(waits) == BACKOFF_MAX_S, "the cap never binds; doubling is unbounded"
    assert max(waits) <= 300, "a single wait longer than 5 min reads as a hang"


def test_the_final_attempts_failure_does_not_sleep_first(monkeypatch):
    """`_call` makes MAX_RETRIES + 1 attempts, and a retryable failure on the
    LAST one has no retry left to wait for: sleeping there spent up to
    BACKOFF_MAX_S (180 s) delaying the terminal LLMError -- a wait that buys
    nothing -- and logged a `retry 13/12` that was never going to be made. The guard lives
    inside `_sleep`, so this drives the REAL `_sleep` and stubs only
    `time.sleep`, counting what would actually have been slept."""
    from bird.llm import anthropic_client as mod

    obj, _ = _client()
    calls = {"n": 0}

    def stream(**params):
        calls["n"] += 1
        return _DyingStreamManager(_status_error(
            obj._sdk.APIStatusError, status=200, err_type="overloaded_error"))

    obj._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    slept: list = []
    monkeypatch.setattr(mod.time, "sleep", lambda s: slept.append(s))

    with pytest.raises(LLMError, match="after .* retries"):
        obj._call(_params())
    assert calls["n"] == MAX_RETRIES + 1, "every attempt is still made"
    assert len(slept) == MAX_RETRIES, (
        f"{len(slept)} sleeps for {MAX_RETRIES} retries: the final attempt's "
        "failure slept for a retry that never happens")


def test_a_mid_stream_overload_that_never_clears_ends_as_LLMError():
    """Retrying must stay bounded and named -- not become a hang, and not
    resurface as a raw SDK traceback from inside a worker thread."""
    obj, _ = _client()

    def stream(**params):
        return _DyingStreamManager(_status_error(
            obj._sdk.APIStatusError, status=200, err_type="overloaded_error"))

    obj._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    obj._sleep = lambda *a, **k: None

    with pytest.raises(LLMError, match="after .* retries"):
        obj._call(_params())


def test_a_genuine_client_error_is_still_immediate():
    """The other half. Widening the predicate must not turn a 404 -- or any
    permanent 4xx -- into six pointless round-trips followed by a wrong
    diagnosis: the run should stop and say what the provider said.
    """
    obj, _ = _client()
    calls = {"n": 0}
    sentinel = _status_error(obj._sdk.APIStatusError, status=404, err_type=None)

    def stream(**params):
        calls["n"] += 1
        return _DyingStreamManager(sentinel)

    obj._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    obj._sleep = lambda *a, **k: None

    with pytest.raises(type(sentinel)):
        obj._call(_params())
    assert calls["n"] == 1, "a permanent 4xx must not be retried"


def test_the_predicate_reads_both_halves():
    """`_retryable_status` is an OR and both arms are load-bearing: a real 5xx
    has no SSE body so `.type` is None, and a mid-stream error has a 200 that
    says nothing about the failure."""
    cls = _fake_sdk().APIStatusError
    r = AnthropicClient._retryable_status
    assert r(_status_error(cls, status=529, err_type=None)), "a real 5xx"
    assert r(_status_error(cls, status=200, err_type="overloaded_error"))
    assert r(_status_error(cls, status=200, err_type="api_error"))
    assert not r(_status_error(cls, status=404, err_type=None))
    assert not r(_status_error(cls, status=200, err_type="invalid_request_error"))
    # An exception carrying neither is not evidence of a transient fault.
    assert not r(Exception("bare"))


def test_a_mid_stream_rate_limit_still_honours_retry_after():
    """The clause now passes `_retry_after(exc)` rather than None, because a
    `rate_limit_error` can arrive mid-stream too and the provider's own number
    is better than our exponential guess."""
    obj, _ = _client()
    seen: list = []
    obj._sleep = lambda attempt, retry_after: seen.append(retry_after)

    def stream(**params):
        if len(seen) < 1:
            return _DyingStreamManager(_status_error(
                obj._sdk.APIStatusError, status=200,
                err_type="rate_limit_error", retry_after="7"))
        return _DyingStreamManager(None)

    obj._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    obj._call(_params())
    assert seen == [7.0], f"retry-after should reach _sleep, got {seen}"


def test_the_installed_sdk_really_does_report_200_for_a_mid_stream_error():
    """The premise, checked against the SDK rather than assumed.

    If a future SDK starts stamping the SSE error's own status onto the
    exception, the `.type` arm becomes redundant and this fails so somebody
    reads it rather than inheriting a workaround for a fixed bug.

    Offline: an `httpx2.Response` is constructed in memory, no socket opened.
    """
    anthropic = pytest.importorskip("anthropic")
    httpx2 = anthropic._base_client.httpx2
    body = {"type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"}}
    response = httpx2.Response(
        200, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    exc = anthropic.APIStatusError(f"{body}", response=response, body=body)

    assert exc.status_code == 200, (
        "the SDK now reports the SSE error's own status; re-read "
        "_retryable_status -- the `.type` arm may be dead code")
    assert exc.type == "overloaded_error", (
        "APIStatusError no longer lifts `error.type` out of the body, so "
        "_retryable_status has nothing to read and needs a new source")
    assert AnthropicClient._retryable_status(exc)


# ==========================================================================
# Prompt caching (`llm.prompt_caching`)
#
# Not a method key: `cache_control` is metadata telling the provider where it
# may reuse computation, so the model reads exactly the same bytes with it and
# without it. It is here because it is most of the bill. Eureka issues
# `generate.n_candidates` round-trips off ONE prompt per iteration (the Messages
# API has no `n`), and MEASURED on `claude-sonnet-4-5` with Eureka's real
# ~5,330-token prompt on pendulum: call 0 wrote 5,327 cache
# tokens and calls 1-3 read 5,327 each, taking the billed-equivalent input from
# 21,320 tokens to 8,269 over four calls (-61%; at Eureka's eight it is -76%).
#
# The two things that silently undo it are a breakpoint that moves with the
# request and an accounting bug, so both get a test.
# ==========================================================================


def _cache_client(caching: bool = True) -> AnthropicClient:
    obj, messages = _client()
    obj.prompt_caching = caching
    obj.messages_fake = messages  # type: ignore[attr-defined]
    obj.modality = "vlm"
    obj.reasoning_effort = "none"
    obj.role = "generator"
    return obj


def _marked(blocks) -> int:
    return sum(1 for b in blocks
               if isinstance(b, dict) and "cache_control" in b)


def test_the_system_prompt_is_a_cache_breakpoint():
    """The only prefix that survives a changed user turn. `generation.
    _build_messages` emits an identical system block on every call a role
    makes."""
    client = _cache_client()
    system, turns = client._cache_breakpoints("you are a reward engineer", [
        {"role": "user", "content": "write one"}])
    assert isinstance(system, list) and _marked(system) == 1
    assert system[0]["text"] == "you are a reward engineer", "the TEXT must not change"


def test_the_end_of_the_final_turn_is_a_cache_breakpoint():
    """Eureka's shape: n identical round-trips, so the whole request is prefix."""
    client = _cache_client()
    _system, turns = client._cache_breakpoints("", [
        {"role": "user", "content": "the prompt"}])
    assert _marked(turns[-1]["content"]) == 1
    assert turns[-1]["content"][-1]["text"] == "the prompt"


def test_the_last_image_is_its_own_breakpoint():
    """RDA's §4 scorer sends the same 20 rollout frames to `len(subtasks)`
    consecutive queries that differ only in the sentence after them. Measured
    on three real subtask queries over one Pendulum rollout: 3,035 of
    3,038 input tokens were cache reads on queries 2 and 3."""
    client = _cache_client()
    content = [{"type": "image", "source": {"type": "base64", "data": "a"}},
               {"type": "image", "source": {"type": "base64", "data": "b"}},
               {"type": "text", "text": "score subtask 1"}]
    _system, turns = client._cache_breakpoints("sys", [{"role": "user", "content": content}])
    blocks = turns[-1]["content"]
    images = [b for b in blocks if b.get("type") == "image"]
    assert "cache_control" in images[-1], "the image prefix is not reusable across subtasks"
    assert "cache_control" not in images[0], "one breakpoint per prefix, not one per block"
    assert _marked(blocks) == 2  # last image + end of turn


def test_the_breakpoint_count_stays_under_the_api_limit():
    """Four is the API's cap. Three leaves a caller room for one of its own."""
    client = _cache_client()
    content = [{"type": "image", "source": {}}, {"type": "text", "text": "q"}]
    system, turns = client._cache_breakpoints("sys", [{"role": "user", "content": content}])
    assert _marked(system) + _marked(turns[-1]["content"]) <= 4


def test_caching_off_touches_nothing():
    """`llm.prompt_caching: false` must produce byte-identical requests to the
    pre-caching client -- it is how the saving is measured."""
    client = _cache_client(caching=False)
    turns_in = [{"role": "user", "content": "the prompt"}]
    system, turns = client._cache_breakpoints("sys", turns_in)
    assert system == "sys" and turns is turns_in


def test_caching_does_not_alter_a_single_byte_the_model_reads():
    """The transparency claim, asserted. If this ever fails, `llm.prompt_caching`
    is a method key and belongs in a different section of the schema."""
    client = _cache_client()
    content = [{"type": "image", "source": {"type": "base64", "data": "a"}},
               {"type": "text", "text": "score it"}]
    turns_in = [{"role": "user", "content": list(content)}]
    system, turns = client._cache_breakpoints("sys prompt", turns_in)
    assert "".join(b["text"] for b in system) == "sys prompt"
    stripped = [{k: v for k, v in b.items() if k != "cache_control"}
                for b in turns[-1]["content"]]
    assert stripped == content
    assert turns_in[0]["content"] == content, "the caller's messages were mutated"


def test_cached_input_is_counted_once_and_split_not_dropped():
    """The API reports `input_tokens`, `cache_read_input_tokens` and
    `cache_creation_input_tokens` as DISJOINT buckets.

    Reading `input_tokens` alone under-reports a cached run by ~4x and makes
    caching look like a modelling change; adding the cache figures on top of an
    already-inclusive total would double-count. `llm_prompt_tokens` must keep
    meaning "input tokens this call consumed", the same thing it meant before
    caching existed.
    """
    import random as _random

    from bird.budget import Budget
    from bird.llm.base import LLMClient

    class _Ctx:
        cfg = {"llm.max_context_tokens": 128000}

        def __init__(self):
            self.budget = Budget()
            self.rng = _random.Random(0)

    obj = object.__new__(AnthropicClient)
    ctx = _Ctx()
    obj.ctx, obj.modality, obj.n_calls = ctx, "text", 0
    obj.max_tokens, obj.model, obj.reasoning_effort = DEFAULT_MAX_TOKENS, "m", "none"
    obj.prompt_caching, obj._dropped, obj._truncated = True, set(), 0
    # `__init__` is bypassed, so the attributes it would set for the
    # fan-out path must be supplied by hand: `_record` serialises under
    # `_record_lock` since `llm.max_concurrent_requests` exists.
    obj.max_concurrent_requests, obj._record_lock = 1, threading.Lock()
    obj._sdk = _fake_sdk()

    message = _FakeMessage()
    message.usage = types.SimpleNamespace(
        input_tokens=12, output_tokens=3,
        cache_read_input_tokens=5000, cache_creation_input_tokens=0)
    obj._client = types.SimpleNamespace(
        messages=types.SimpleNamespace(stream=lambda **p: _FakeStreamManager(message)))

    obj._one("sys", [{"role": "user", "content": "hi"}], 0.0, 0)
    b = ctx.budget
    assert b.llm_prompt_tokens == 5012, "input_tokens excludes the cached buckets"
    assert b.llm_cache_read_tokens == 5000
    assert b.llm_cache_write_tokens == 0
    assert b.llm_cache_read_tokens <= b.llm_prompt_tokens, "the split is not an addition"
    # `LLMClient._record` is the only path that may write these.
    assert "cache_read_tokens" in LLMClient._record.__code__.co_varnames


def test_a_400_that_names_only_a_subfield_still_degrades():
    """MEASURED on `claude-sonnet-4-5` with
    `llm.generator.reasoning_effort: medium`:

        400 "This model does not support the effort parameter."

    `effort` is the field INSIDE `output_config`, so nothing the request sent by
    NAME appears in the message; a `_degrade` that matched only field names
    would return False and let the exception propagate -- the run dying in
    `generate` of iteration 1 rather than dropping the pin and continuing.
    Same shape as the `temperature` TypeError this file
    already covers: a pin the far side refuses has to be droppable however the
    far side phrases it."""
    client, _unused = _client()
    sent: list[dict] = []

    def stream(**params):
        sent.append(dict(params))
        if "output_config" in params:
            raise client._sdk.BadRequestError(
                "This model does not support the effort parameter.")
        if "thinking" in params:
            raise client._sdk.BadRequestError(
                "This model does not support extended thinking.")
        return _FakeStreamManager(_FakeMessage())

    client._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    out = client._call(_params(output_config={"effort": "medium"},
                               thinking={"type": "adaptive"}))
    assert out.stop_reason == "end_turn"
    assert client._dropped == {"output_config", "thinking"}, (
        "dropping output_config may leave thinking for the NEXT attempt to "
        "reject; both must be learned rather than one being re-learned per call")
    assert "output_config" not in sent[-1] and "thinking" not in sent[-1]
    assert len(sent) == 3, "one attempt per rejected pin, then the real request"


# ==========================================================================
# The retry envelope and the request timeout, as OPERATOR levers
#
# `BIRD_LLM_MAX_RETRIES`, `BIRD_LLM_BACKOFF_BASE_S`, `BIRD_LLM_BACKOFF_MAX_S`
# and `BIRD_LLM_TIMEOUT_S` are environment variables and not config keys, for
# the reason `MAX_RETRIES`' comment gives: they are transport, and a config key
# would move every run's config hash to express something that cannot change
# what the model reads.
#
# The default envelope is ~2-3 minutes of patience, which is sized for a
# transient 429 and NOT for the failure a multi-day HumanoidBench search meets:
# a provider incident at hour 18, where dying costs a queue wait plus a re-paid
# iteration and waiting half an hour costs half an hour. So the levers exist --
# and each of the three properties below is one that would make them useless if
# it were quietly lost.
# ==========================================================================


#: THE MODULE IS NEVER RELOADED HERE, and that is not squeamishness.
#: `importlib.reload(bird.llm.anthropic_client)` re-runs its module-level
#: `@register("llm", "anthropic")` with a NEW function object, and
#: `registry.register` raises `duplicate registration` on exactly that
#: (`_REGISTRY[key] is not fn`). So the obvious way to observe an import-time
#: read takes the test file down instead. What is tested instead is the pair the
#: constants are made of: `_env_num`, which owns every parsing and fallback
#: rule, and the SOURCE binding, which owns "this constant is still read from
#: that variable" -- the half a pure-function test cannot see.


def test_the_retry_envelope_defaults_are_pinned():
    """No behaviour change for anyone who sets nothing.

    The whole argument for env vars over config keys is that an unset
    environment must be indistinguishable from the code before them -- if this
    drifts, every existing run silently changes its retry behaviour without a
    config diff to show it.

    THE NUMBERS HAVE A REASON: sustained provider 5xx bursts outlast 5
    retries, so the ladder is 12 retries capped at 180 s. Making the
    ladder tunable must not become an occasion to re-pick its rungs -- an
    operator who exports a SMALLER value shortens the window a long search can
    ride out, which is exactly what this assertion is here to make visible.
    """
    from bird.llm import anthropic_client as mod

    assert (mod.MAX_RETRIES, mod.BACKOFF_BASE_S, mod.BACKOFF_MAX_S) == (12, 1.0, 180.0)
    assert mod.LLM_TIMEOUT_S is None, (
        "an unset BIRD_LLM_TIMEOUT_S must leave the SDK's own default alone "
        "rather than pin one nobody asked for")


def test_a_generous_envelope_is_read_from_the_environment(monkeypatch):
    """Documented operator values parse to what they are documented to mean."""
    from bird.llm.anthropic_client import _env_num

    monkeypatch.setenv("BIRD_LLM_MAX_RETRIES", "20")
    monkeypatch.setenv("BIRD_LLM_BACKOFF_MAX_S", "300")
    monkeypatch.setenv("BIRD_LLM_TIMEOUT_S", "1800")

    retries = _env_num("BIRD_LLM_MAX_RETRIES", 12, int, 0)
    assert retries == 20 and isinstance(retries, int), (
        "MAX_RETRIES feeds `range(MAX_RETRIES + 1)`; a float would raise there")
    assert _env_num("BIRD_LLM_BACKOFF_MAX_S", 180.0, float, 0.0) == 300.0
    assert _env_num("BIRD_LLM_TIMEOUT_S", 0.0, float, 0.0) == 1800.0
    # An unset lever keeps its own default rather than following its neighbours.
    monkeypatch.delenv("BIRD_LLM_BACKOFF_BASE_S", raising=False)
    assert _env_num("BIRD_LLM_BACKOFF_BASE_S", 1.0, float, 0.0) == 1.0


@pytest.mark.parametrize("raw", ["lots", "", "   ", "-5", "1e", "8,000"])
def test_a_malformed_lever_keeps_the_default(monkeypatch, raw):
    """A typo must not take out every job of a batch before the search starts.

    These are read AT IMPORT, so raising here kills the process before it can
    say anything useful about what it was going to do -- and it kills all N of
    a batch's jobs, simultaneously, at the moment they start.

    `-5` is in the list because a NEGATIVE retry count is the one malformed
    value that parses: `range(-5 + 1)` is empty, so the transport would make no
    attempt at all and every call would raise `LLMError` having sent nothing.
    """
    from bird.llm.anthropic_client import _env_num

    monkeypatch.setenv("BIRD_LLM_MAX_RETRIES", raw)
    assert _env_num("BIRD_LLM_MAX_RETRIES", 12, int, 0) == 12


def test_an_empty_timeout_is_unset_rather_than_zero():
    """`0` and "not set" must not collide.

    `LLM_TIMEOUT_S` is `_env_num(...) or None`, so a zero -- which is what an
    empty or malformed value falls back to -- has to mean "leave the SDK alone".
    A literal `BIRD_LLM_TIMEOUT_S=0` therefore means the same thing, which is
    the reading an operator would expect and the only one that is safe: a
    genuine zero-second timeout would fail every request.
    """
    from bird.llm.anthropic_client import LLM_TIMEOUT_S, _env_num

    assert (_env_num("BIRD_LLM_TIMEOUT_S", 0.0, float, 0.0) or None) is None
    assert LLM_TIMEOUT_S is None or LLM_TIMEOUT_S > 0


def test_each_constant_is_still_bound_to_its_environment_variable():
    """The half `_env_num`'s own tests cannot see.

    `_env_num` can be perfect while `MAX_RETRIES = 5` sits above it, and the
    levers would then be documented, tested and dead -- which is precisely the
    declared-but-unread failure: a key that sits in the schema, is read by
    nothing, and passes every test. Source-level, because the module cannot be
    re-imported under a different environment (see the note above), and
    grep-style.
    """
    import re
    from pathlib import Path as _Path

    source = (_Path(__file__).resolve().parents[1]
              / "bird" / "llm" / "anthropic_client.py").read_text()
    for const, var in (("MAX_RETRIES", "BIRD_LLM_MAX_RETRIES"),
                       ("BACKOFF_BASE_S", "BIRD_LLM_BACKOFF_BASE_S"),
                       ("BACKOFF_MAX_S", "BIRD_LLM_BACKOFF_MAX_S"),
                       ("LLM_TIMEOUT_S", "BIRD_LLM_TIMEOUT_S")):
        assert re.search(rf'^{const}[^\n]*_env_num\("{var}"', source, re.M), (
            f"{const} is no longer read from {var}: the operator lever is dead, "
            "and a dead lever is worse than none because the documentation still "
            "tells people to set it")


def test_the_timeout_reaches_the_sdk_constructor(monkeypatch):
    """`_connect` must PASS the timeout, not merely read it.

    This is the joint that makes the lever real rather than decorative: the
    value is read at module scope and used one function away, so a refactor can
    drop it with nothing else going red.

    It also pins the INVERSE, which is the more dangerous direction: an unset
    lever must pass NO `timeout` keyword at all rather than `timeout=None`.
    Passing None reads as "no timeout" to the SDK, which would silently remove
    the default read deadline -- and a request with no deadline does not fail,
    it HANGS, holding an allocation to the walltime. That is strictly worse than
    the mid-stream `ReadTimeout` the top half of this file exists for, because
    at least a timeout raises something the retry loop can see.
    """
    import types as _types

    from bird.llm import anthropic_client as mod

    seen: list[dict] = []

    class _Fake:
        def __init__(self, **kwargs):
            seen.append(kwargs)

    monkeypatch.setitem(sys.modules, "anthropic",
                        _types.SimpleNamespace(Anthropic=_Fake))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    def _connect_fresh():
        obj = object.__new__(mod.AnthropicClient)
        obj._client = None
        obj._sdk = None
        obj._connect()

    monkeypatch.setattr(mod, "LLM_TIMEOUT_S", None)
    _connect_fresh()
    assert "timeout" not in seen[-1], (
        "an unset BIRD_LLM_TIMEOUT_S passed a timeout to the SDK anyway; "
        "timeout=None means NO deadline, i.e. a hang instead of a retry")
    assert seen[-1].get("api_key") == "k"

    monkeypatch.setattr(mod, "LLM_TIMEOUT_S", 1800.0)
    _connect_fresh()
    assert seen[-1].get("timeout") == 1800.0
    assert seen[-1].get("api_key") == "k", "the key must survive the new kwargs path"


# --------------------------------------------------------------------------
# Refusals and truncations: retried where free, counted where not, and named
# to the generate stage instead of arriving as "no code block matched".
#
# A provider can return `stop_reason: refusal` for a large share of generator
# calls, across whole iterations, while the run still finishes `status: ok`;
# without these counters the artifact would say the model wrote unparseable
# code. Such refusals are transient: the identical prompt can answer normally
# later.
# --------------------------------------------------------------------------

def _outcome_client(stop_reasons: list[str]):
    """A serial client whose successive round-trips stop for the given reasons."""
    import random as _random
    import threading
    from bird.budget import Budget

    class _Ctx:
        cfg = {"llm.max_context_tokens": 128000}

        def __init__(self):
            self.budget = Budget()
            self.rng = _random.Random(0)

    obj = object.__new__(AnthropicClient)
    obj.ctx = _Ctx()
    obj.modality, obj.n_calls, obj.role = "text", 0, "generator"
    obj.max_tokens, obj.model, obj.reasoning_effort = DEFAULT_MAX_TOKENS, "m", "none"
    obj.prompt_caching, obj._dropped, obj._truncated, obj._refused = True, set(), 0, 0
    obj.max_concurrent_requests, obj._record_lock = 1, threading.Lock()
    obj._sdk = _fake_sdk()
    obj._sleep = lambda *a, **k: None      # no transport backoff in a unit test
    import bird.llm.anthropic_client as _ac
    _ac.REFUSAL_PAUSE_S = 0.0              # and no refusal pause either
    queue = list(stop_reasons)

    def stream(**p):
        m = _FakeMessage()
        m.stop_reason = queue.pop(0)
        if m.stop_reason == "refusal":
            m.content = []
            m.usage = types.SimpleNamespace(input_tokens=10, output_tokens=0)
            m.stop_details = types.SimpleNamespace(category="frontier_llm")
        return _FakeStreamManager(m)

    obj._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    return obj, queue


def test_a_refusal_is_retried_and_a_later_answer_is_kept():
    obj, queue = _outcome_client(["refusal", "refusal", "end_turn"])
    out = obj._one("sys", [{"role": "user", "content": "hi"}], 0.0, 0)
    assert out == "ok"
    assert queue == [], "all three round-trips were made"
    b = obj.ctx.budget
    assert b.llm_calls == 3, "a refused round-trip still bills its input"
    assert b.llm_refusals == 2 and b.llm_truncations == 0
    assert obj._refused == 0, "a refusal that a retry recovered is not a forfeited sample"


def test_a_refusal_that_survives_every_retry_is_empty_counted_and_named():
    from bird.llm.anthropic_client import REFUSAL_RETRIES
    obj, _ = _outcome_client(["refusal"] * (REFUSAL_RETRIES + 1))
    out = obj._one("sys", [{"role": "user", "content": "hi"}], 0.0, 0)
    assert out == ""
    assert obj._refused == 1
    assert obj.ctx.budget.llm_refusals == REFUSAL_RETRIES + 1
    assert obj.ctx.budget.llm_calls == REFUSAL_RETRIES + 1
    reason = obj._take_empty_reason()
    assert "refused" in reason and "frontier_llm" in reason
    assert obj._take_empty_reason() == "", "taking the reason clears it"


def test_a_truncated_sample_is_counted_and_named():
    obj, _ = _outcome_client(["max_tokens"])
    out = obj._one("sys", [{"role": "user", "content": "hi"}], 0.0, 0)
    assert out == ""
    assert obj._truncated == 1
    assert obj.ctx.budget.llm_truncations == 1 and obj.ctx.budget.llm_refusals == 0
    assert "max_tokens" in obj._take_empty_reason()


def test_the_budget_carries_both_counters_into_its_summary():
    from bird.budget import Budget
    b = Budget()
    b.record_llm(prompt_tokens=1, completion_tokens=0, refused=True)
    b.record_llm(prompt_tokens=1, completion_tokens=0, truncated=True)
    d = b.report()
    assert d["llm_refusals"] == 1 and d["llm_truncations"] == 1


# -- the reason must land on the SAME slot as the empty sample -----------------
#
# One list per `_complete`, appended in completion order, would let a fan-out
# or a chunked `__call__` hand a refused sample's reason to a neighbour.
# Alignment is by index, at every layer.

def _outcome_client_by_slot(stop_by_slot: list[str], concurrency: int = 1):
    """Round-trip k answers with `stop_by_slot[k]` (in REQUEST order)."""
    import itertools
    obj, _ = _outcome_client([])
    obj.max_concurrent_requests = concurrency
    obj.concurrent_samples = True
    counter = itertools.count()
    lock = threading.Lock()

    def stream(**p):
        with lock:
            k = next(counter)
        m = _FakeMessage()
        m.stop_reason = stop_by_slot[k]
        m.content = [types.SimpleNamespace(type="text", text=f"sample-{k}")]
        if m.stop_reason == "refusal":
            m.content = []
            m.stop_details = types.SimpleNamespace(category="frontier_llm")
        return _FakeStreamManager(m)

    obj._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    return obj


def _outcome_client_by_thread(leader: list[str], followers: list[list[str]]):
    """Round-trip answers keyed on the REQUESTING THREAD, not on request order.

    `_outcome_client_by_slot` hands out `stop_by_slot[k]` to the k-th request, which is
    only the k-th SLOT when requests are sequential. A fanned-out `_complete` runs its
    followers concurrently, and a follower whose answer is `refusal` retries, so two
    followers interleave their draws from one shared list: both take refusals meant for
    one of them, the retries exhaust the list early, and the next request indexes past
    its end (`IndexError`, a flake). Here the leader -- the calling thread --
    draws from `leader`; each pool thread is assigned the next `followers` queue the
    first time it asks and draws from that queue on every retry; an exhausted queue
    answers `end_turn`, so a pool thread that happens to serve two futures (the executor
    reuses an idle worker) still produces exactly the refusals its queue held."""
    obj, _ = _outcome_client([])
    obj.max_concurrent_requests = max(1, len(followers))
    obj.concurrent_samples = True
    leader_q = list(leader)
    queues = [list(q) for q in followers]
    by_thread: dict[int, list[str]] = {}
    caller = threading.get_ident()
    lock = threading.Lock()
    counter = itertools.count()

    def stream(**p):
        with lock:
            k = next(counter)
            me = threading.get_ident()
            if me == caller:
                q = leader_q
            else:
                if me not in by_thread:
                    by_thread[me] = queues[len(by_thread)] if len(by_thread) < len(queues) else []
                q = by_thread[me]
            stop = q.pop(0) if q else "end_turn"
        m = _FakeMessage()
        m.stop_reason = stop
        m.content = [types.SimpleNamespace(type="text", text=f"sample-{k}")]
        if stop == "refusal":
            m.content = []
            m.stop_details = types.SimpleNamespace(category="frontier_llm")
        return _FakeStreamManager(m)

    obj._client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=stream))
    return obj


def test_a_sequential_complete_aligns_reasons_with_samples():
    from bird.llm.anthropic_client import REFUSAL_RETRIES
    # slot 1 refused on every attempt, slots 0 and 2 fine
    stops = ["end_turn"] + ["refusal"] * (REFUSAL_RETRIES + 1) + ["end_turn"]
    obj = _outcome_client_by_slot(stops, concurrency=1)
    out = obj._complete([{"role": "user", "content": "hi"}], 3, 0.0, None, "t")
    assert out[0] == "sample-0" and out[1] == "" and out[2].startswith("sample-")
    reasons = obj._chunk_empty_reasons
    assert len(reasons) == 3
    assert reasons[0] == "" and "refused" in reasons[1] and reasons[2] == ""


def test_a_fanned_out_complete_keeps_the_reason_on_its_own_slot():
    """Followers finish in any order; the (text, reason) pair is read in the
    thread that produced it, so a refusal cannot drift to a neighbour."""
    from bird.llm.anthropic_client import REFUSAL_RETRIES
    # leader fine; follower A refused on every attempt; follower B fine. The stub keys
    # the answers on the requesting thread: with a shared request-ordered list the two
    # followers interleaved their draws and the retries ran it off the end (a flake).
    obj = _outcome_client_by_thread(["end_turn"], [["refusal"] * (REFUSAL_RETRIES + 1), ["end_turn"]])
    out = obj._complete([{"role": "user", "content": "hi"}], 3, 0.0, None, "t")
    reasons = obj._chunk_empty_reasons
    assert len(out) == len(reasons) == 3
    for text, reason in zip(out, reasons):
        assert (text == "") == bool(reason), (text, reason)
    assert sum(1 for r in reasons if r) == 1 and "refused" in next(r for r in reasons if r)


def test_call_stitches_reasons_across_chunks_and_pads_a_short_return():
    """`LLMClient.__call__` is the one place chunks are reassembled; it must
    carry each chunk's reasons along and name the slots a short provider
    never filled."""
    from bird.llm.base import LLMClient

    class _Chunky(LLMClient):
        provider = "chunky"

        def __init__(self):        # bypass config binding
            self.chunk_size, self.max_concurrent_requests = 2, 1
            self.concurrent_samples, self.n_calls, self.temperature = False, 0, 0.0
            self._record_lock = threading.Lock()
            self.calls = 0

        def _complete(self, messages, n, temperature, images, tag):
            self.calls += 1
            if self.calls == 1:            # chunk of 2: second sample empty
                self._chunk_empty_reasons = ["", "refused by the provider (category=x)"]
                return ["a", ""]
            self._chunk_empty_reasons = []  # chunk of 2 requested, ONE returned
            return ["c"]

    c = _Chunky()
    out = c([{"role": "user", "content": "hi"}], n=4)
    assert out == ["a", "", "c", ""]
    assert c.last_empty_reasons[0] == "" and "refused" in c.last_empty_reasons[1]
    assert c.last_empty_reasons[2] == ""
    assert "3/4 samples" in c.last_empty_reasons[3]
