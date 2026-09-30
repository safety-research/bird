"""`OpenAIClient._one` honours the empty-sample contract the anthropic client
sets: a completion the model cut at its output ceiling, or refused, is an EMPTY
sample -- counted, named, and never parsed as a program.

Returning `choice.message.content` whatever `finish_reason` says would be
wrong. The client deliberately sends no `max_tokens`, so a
`finish_reason == "length"` completion is the model's own ceiling; the cut
program has no closing fence, `_extract_code` falls through the default
`generate.parse.patterns` to the docstring pattern, and verification records a
SyntaxError as if the model wrote bad code -- while `budget.llm_truncations`
stays 0 and LIMEN's failure memory teaches the next prompt a "syntax error"
that never happened. `message.refusal` must likewise be read.

The `openai` SDK is not installed in the test environment and `_one` imports it
for its exception classes, so a stub module is injected; the response objects
are the SDK's attribute shapes, as the anthropic transport tests do.
"""

from __future__ import annotations

import random
import sys
import threading
import types

import pytest

from bird.budget import Budget


class _Ctx:
    cfg = {"llm.max_context_tokens": 128000}

    def __init__(self):
        self.budget = Budget()
        self.rng = random.Random(0)


@pytest.fixture
def fake_openai(monkeypatch):
    """`_one` does `import openai` for the exception ladder; give it one."""
    class _Err(Exception):
        pass
    mod = types.ModuleType("openai")
    for name in ("BadRequestError", "RateLimitError", "APIConnectionError",
                 "APITimeoutError", "APIStatusError"):
        setattr(mod, name, type(name, (_Err,), {}))
    monkeypatch.setitem(sys.modules, "openai", mod)
    return mod


def _resp(content, finish_reason: str, refusal=None, completion_tokens: int = 5):
    message = types.SimpleNamespace(content=content, refusal=refusal)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=completion_tokens,
                                  prompt_tokens_details=None)
    return types.SimpleNamespace(choices=[choice], usage=usage)


def _client(responses):
    """An `OpenAIClient` with `__init__` bypassed (it would connect) and the
    transport replaced by a queue of canned responses."""
    from bird.llm.openai_client import OpenAIClient
    obj = object.__new__(OpenAIClient)
    obj.ctx = _Ctx()
    obj.modality, obj.n_calls, obj.role = "text", 0, "generator"
    obj.model, obj.reasoning_effort, obj.temperature = "qwen/qwen3-max", "none", 1.0
    obj._dropped, obj._record_lock, obj.max_concurrent_requests = set(), threading.Lock(), 1
    obj.chunk_size = 8
    queue = list(responses)
    create = lambda **params: queue.pop(0)  # noqa: E731
    obj._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return obj


_TURNS = [{"role": "user", "content": "write compute_reward"}]


def test_a_length_cut_completion_is_an_empty_truncated_sample(fake_openai) -> None:
    obj = _client([_resp("```python\ndef compute_reward(state, action", "length",
                         completion_tokens=4096)])
    out = obj._one(_TURNS, 0.0, 0)
    assert out == "", "a program cut mid-token was returned as if complete"
    b = obj.ctx.budget
    assert b.llm_calls == 1 and b.llm_truncations == 1 and b.llm_refusals == 0
    assert b.llm_completion_tokens == 4096, "the cut output was still billed"
    reason = obj._take_empty_reason()
    assert "length" in reason and "4096" in reason
    assert obj._truncated == 1


def test_a_stop_completion_is_returned_unchanged(fake_openai) -> None:
    obj = _client([_resp("```python\ndef compute_reward(s, a, s2):\n    return 0.0\n```", "stop")])
    out = obj._one(_TURNS, 0.0, 0)
    assert out.startswith("```python")
    b = obj.ctx.budget
    assert (b.llm_calls, b.llm_truncations, b.llm_refusals) == (1, 0, 0)
    assert obj._take_empty_reason() == ""


def test_a_refusal_is_an_empty_refused_sample(fake_openai) -> None:
    obj = _client([_resp(None, "stop", refusal="I can't help with that.")])
    out = obj._one(_TURNS, 0.0, 0)
    assert out == ""
    b = obj.ctx.budget
    assert (b.llm_calls, b.llm_truncations, b.llm_refusals) == (1, 0, 1)
    assert "refus" in obj._take_empty_reason()
    assert obj._refused == 1


def test_a_content_filter_finish_is_a_refusal(fake_openai) -> None:
    obj = _client([_resp("", "content_filter")])
    assert obj._one(_TURNS, 0.0, 0) == ""
    assert obj.ctx.budget.llm_refusals == 1
    assert "content_filter" in obj._take_empty_reason()


def test_the_reason_lands_on_the_sample_slot_through_complete(fake_openai) -> None:
    """`_complete` must fill `_chunk_empty_reasons` index-aligned with its
    output, as the anthropic client does, or `LLMClient.__call__` cannot hand
    the generate stage the reason for the exact forfeited slot."""
    obj = _client([_resp("ok one", "stop"), _resp("cut", "length", completion_tokens=99),
                   _resp("ok three", "stop")])
    out = obj._complete(_TURNS, 3, 0.0, None, "generate")
    assert out == ["ok one", "", "ok three"]
    reasons = list(obj._chunk_empty_reasons)
    assert reasons[0] == "" and reasons[2] == ""
    assert "length" in reasons[1]


def test_last_empty_reasons_reach_the_caller_through___call__(fake_openai) -> None:
    obj = _client([_resp("ok", "stop"), _resp("cut", "length", completion_tokens=7)])
    out = obj(_TURNS, n=2, tag="generate")
    assert out == ["ok", ""]
    assert obj.last_empty_reasons[0] == "" and "length" in obj.last_empty_reasons[1]
    assert obj.ctx.budget.llm_truncations == 1
