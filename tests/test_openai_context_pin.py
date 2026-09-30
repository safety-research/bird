"""`_pin_max_tokens`: repair a router-invented completion allowance, and refuse
everything else.

The 400 this handles arrives with a prompt that FITS its window -- 5,871 against 8,192 --
and a 3,410-token completion allowance the client never sent, so without the repair a
search fails on an otherwise valid request.  The tests below use that
provider text verbatim,
because the whole mechanism is reading numbers out of prose and the only honest fixture
is the prose that actually arrived.

The refusal cases carry as much weight as the repair: a prompt that genuinely does not
fit must raise, not come back as a truncated reward function that scores like a bad idea.
"""
from __future__ import annotations

import random
import sys
import threading
import types

import pytest

from bird.budget import Budget
from bird.llm.openai_client import OpenAIClient, _int_after, _int_before

#: Verbatim from an OpenRouter 400, lowercased the way `_degrade` lowercases it.
REAL = (
    "error code: 400 - {'error': {'message': 'provider returned error', "
    "'code': 400, 'metadata': {'raw': '{\\n  \"error\": {\\n    \"message\": "
    "\"this model's maximum context length is 8192 tokens. however, you "
    "requested 9281 tokens (5871 in the messages, 3410 in the completion). "
    "please reduce the length of the messages or completion.\",\\n    "
    "\"type\": \"invalid_request_error\",\\n    \"code\": "
    "\"context_length_exceeded\"\\n  }\\n}'}}}"
)


class _Bare(OpenAIClient):
    """`_pin_max_tokens` touches only `self._dropped`, `self.model` and its two
    class constants, so the test drives it without a network client or an API
    key -- constructing a real one would test the constructor, not the repair."""

    def __init__(self):
        self._dropped = set()
        self.model = "openai/gpt-4"


def test_the_numbers_come_out_of_the_real_400():
    assert _int_after(REAL, r"maximum context length is") == 8192
    assert _int_before(REAL, r"in the messages") == 5871


def test_the_real_400_is_repaired_to_the_real_headroom():
    c, params = _Bare(), {"model": "openai/gpt-4", "messages": []}
    assert c._pin_max_tokens(params, REAL) is True
    # 8192 - 5871 - 64 margin.  Asserted as the arithmetic, not as a constant,
    # so a change to the margin shows up here as an intent change.
    assert params["max_tokens"] == 8192 - 5871 - OpenAIClient._HEADROOM_MARGIN_TOKENS
    assert params["max_tokens"] == 2257


def test_the_pinned_room_comfortably_exceeds_a_typical_completion():
    """The repaired allowance leaves room for a typical reward-program completion
    (a few hundred tokens) several times over."""
    c, params = _Bare(), {}
    c._pin_max_tokens(params, REAL)
    assert params["max_tokens"] > 4 * 500


def test_it_cannot_loop_within_one_call():
    """A retry inside one `_one` call reuses one `params`, so a second 400 on the
    SAME request must not be repaired again -- it must raise."""
    c, params = _Bare(), {}
    assert c._pin_max_tokens(params, REAL) is True
    assert c._pin_max_tokens(params, REAL) is False


def test_it_repairs_EVERY_call_not_just_the_first():
    """THE REGRESSION TEST.

    `_one` rebuilds `params` per call, so latching on anything run-scoped would make
    the repair fire ONCE per client and every later 400 raise. Eureka sends 16
    candidates an iteration against a growing prompt, so the second occurrence is
    near-certain and the run would die a few calls after the first repair.
    """
    c = _Bare()
    for call in range(1, 4):
        params = {"model": "openai/gpt-4", "messages": []}   # fresh, as `_one` builds it
        assert c._pin_max_tokens(params, REAL) is True, f"call {call} was not repaired"
        assert params["max_tokens"] == 2257


def test_the_headroom_is_rederived_per_call_from_that_calls_prompt():
    """Not a constant: a later call with a bigger prompt gets a smaller pin."""
    c = _Bare()
    p1, p2 = {}, {}
    c._pin_max_tokens(p1, REAL)
    bigger = ("this model's maximum context length is 8192 tokens. however, you "
              "requested 9999 tokens (6500 in the messages, 3499 in the completion).")
    c._pin_max_tokens(p2, bigger)
    assert p1["max_tokens"] == 8192 - 5871 - OpenAIClient._HEADROOM_MARGIN_TOKENS
    assert p2["max_tokens"] == 8192 - 6500 - OpenAIClient._HEADROOM_MARGIN_TOKENS
    assert p2["max_tokens"] < p1["max_tokens"]


def test_it_refuses_when_the_parsed_prompt_size_contradicts_our_own_estimate():
    """A reworded 400 claiming a tiny prompt must not pin the whole window.

    The parse is positional over provider prose. If a message read "(0 in the
    messages, N in the completion)" on a request whose prompt really does fill the
    window, pinning `max_tokens` to ~8,128 would turn a clean 400 into a second
    doomed call. The cross-check against `estimate_tokens` refuses instead.
    """
    c = _Bare()
    big = "x" * 40000                      # ~10,000 tokens at 4 chars/token
    params = {"model": "openai/gpt-4",
              "messages": [{"role": "user", "content": big}]}
    liar = ("this model's maximum context length is 8192 tokens. however, you "
            "requested 9000 tokens (12 in the messages, 8988 in the completion).")
    assert c._pin_max_tokens(params, liar) is False
    assert "max_tokens" not in params


def test_the_cross_check_does_not_fire_on_the_real_400():
    """...and the sanity floor must not reject the case the repair exists for."""
    c = _Bare()
    # A prompt whose own estimate is in the right ballpark for 5,871 tokens.
    params = {"model": "openai/gpt-4",
              "messages": [{"role": "user", "content": "y" * (5871 * 4)}]}
    assert c._pin_max_tokens(params, REAL) is True
    assert params["max_tokens"] == 2257


def test_it_refuses_when_the_prompt_genuinely_does_not_fit():
    """No repair is possible here, and a truncated answer would be worse than
    the error: the 400 must reach the caller."""
    text = ("this model's maximum context length is 8192 tokens. however, you "
            "requested 8600 tokens (8100 in the messages, 500 in the completion).")
    c, params = _Bare(), {}
    assert c._pin_max_tokens(params, text) is False
    assert "max_tokens" not in params


def test_it_refuses_when_the_prompt_alone_exceeds_the_window():
    text = ("maximum context length is 8192 tokens. you requested 9000 tokens "
            "(9000 in the messages, 0 in the completion).")
    c, params = _Bare(), {}
    assert c._pin_max_tokens(params, text) is False


def test_it_refuses_a_400_that_is_not_about_context():
    c, params = _Bare(), {}
    assert c._pin_max_tokens(params, "unsupported parameter: 'temperature'") is False
    assert params == {}


def test_it_refuses_when_the_numbers_are_not_there():
    """A provider that rewords its message must not be guessed at."""
    text = "this model's context window was exceeded by your request."
    c, params = _Bare(), {}
    assert c._pin_max_tokens(params, text) is False


def test_degrade_routes_the_context_400_here_and_leaves_the_others_alone():
    """The repair is reached through the existing ladder, in order, and does not
    shadow the temperature and reasoning rungs."""
    c = _Bare()
    p = {"temperature": 1.0}
    assert c._degrade(p, "unsupported value: 'temperature' does not support 1.0") is True
    assert "temperature" not in p and "max_tokens" not in p

    c2 = _Bare()
    p2 = {"model": "openai/gpt-4", "messages": []}
    assert c2._degrade(p2, REAL) is True
    assert p2["max_tokens"] == 2257


@pytest.mark.parametrize("margin_free_room,expected", [
    (OpenAIClient._MIN_USEFUL_COMPLETION_TOKENS - 1, False),
    (OpenAIClient._MIN_USEFUL_COMPLETION_TOKENS, True),
])
def test_the_minimum_useful_completion_boundary(margin_free_room, expected):
    """Exactly at the floor it repairs; one token below it refuses."""
    window = 10000
    used = window - margin_free_room - OpenAIClient._HEADROOM_MARGIN_TOKENS
    text = ("maximum context length is %d tokens. however, you requested many "
            "tokens (%d in the messages, 3410 in the completion)." % (window, used))
    c, params = _Bare(), {}
    assert c._pin_max_tokens(params, text) is expected


# ---------------------------------------------------------------------------
# End to end through `_one`'s retry loop.
#
# Everything above tests the decision.  These test that the decision REACHES THE
# WIRE -- that the retried request actually carries the pin, and that a caller
# who never sees a 400 is unaffected.  Without these, a `_pin_max_tokens` that
# mutated a copy of `params` would pass every test above and still let the run
# die.
# ---------------------------------------------------------------------------

class _Ctx:
    cfg = {"llm.max_context_tokens": 8192}

    def __init__(self):
        self.budget = Budget()
        self.rng = random.Random(0)


@pytest.fixture
def fake_openai(monkeypatch):
    """`_one` does `import openai` for its exception ladder; give it one.
    Same shape as `tests/test_openai_outcomes.py`, which the SDK's absence from
    the test environment forces on both files."""
    class _Err(Exception):
        pass
    mod = types.ModuleType("openai")
    for name in ("BadRequestError", "RateLimitError", "APIConnectionError",
                 "APITimeoutError", "APIStatusError"):
        setattr(mod, name, type(name, (_Err,), {}))
    monkeypatch.setitem(sys.modules, "openai", mod)
    return mod


def _resp(content: str):
    message = types.SimpleNamespace(content=content, refusal=None)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    usage = types.SimpleNamespace(prompt_tokens=5871, completion_tokens=461,
                                  prompt_tokens_details=None)
    return types.SimpleNamespace(choices=[choice], usage=usage)


def _client(create):
    """An `OpenAIClient` with `__init__` bypassed -- it would open a connection
    and demand a key -- and the transport replaced."""
    obj = object.__new__(OpenAIClient)
    obj.ctx = _Ctx()
    obj.modality, obj.n_calls, obj.role = "text", 0, "generator"
    obj.model, obj.reasoning_effort, obj.temperature = "openai/gpt-4", "none", 1.0
    obj._dropped, obj._record_lock, obj.max_concurrent_requests = set(), threading.Lock(), 1
    obj.chunk_size = 8
    obj._truncated = obj._refused = 0
    obj._chunk_empty_reasons = []
    obj._tl = threading.local()
    obj._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return obj


_TURNS = [{"role": "user", "content": "write compute_reward"}]


def test_the_retried_request_actually_carries_the_pin(fake_openai):
    """THE TEST THAT CAN FAIL FOR THE REASON THAT MATTERS.

    First call raises the real 400; the second must reach the transport with
    `max_tokens` equal to the window minus the prompt, less the margin. Asserted
    on the params the STUB RECEIVED, not on the client's internal state.
    """
    seen = []

    def create(**params):
        seen.append(dict(params))
        if len(seen) == 1:
            raise fake_openai.BadRequestError(REAL)
        return _resp("def compute_reward(...): ...")

    c = _client(create)
    out = c._one(list(_TURNS), temperature=1.0, n_img=0)

    assert len(seen) == 2, "the 400 must be retried exactly once, not swallowed"
    assert "max_tokens" not in seen[0], "the first request must send no max_tokens"
    assert seen[1]["max_tokens"] == 8192 - 5871 - OpenAIClient._HEADROOM_MARGIN_TOKENS
    assert seen[1]["max_tokens"] == 2257
    assert seen[1]["messages"] == _TURNS, "the prompt must be unchanged by the repair"
    assert out.startswith("def compute_reward")


def test_a_call_that_never_400s_is_untouched(fake_openai):
    """The default -- no `max_tokens` -- must hold for every other request."""
    seen = []

    def create(**params):
        seen.append(dict(params))
        return _resp("ok")

    c = _client(create)
    c._one(list(_TURNS), temperature=1.0, n_img=0)
    assert len(seen) == 1
    assert "max_tokens" not in seen[0]


def test_an_unrepairable_context_400_still_raises(fake_openai):
    """When the prompt genuinely does not fit, the caller sees the error rather
    than a quietly shrunken answer -- and the transport is not retried."""
    from bird.llm.base import LLMError

    seen = []
    text = ("this model's maximum context length is 8192 tokens. however, you "
            "requested 8600 tokens (8100 in the messages, 500 in the completion).")

    def create(**params):
        seen.append(dict(params))
        raise fake_openai.BadRequestError(text)

    c = _client(create)
    with pytest.raises(LLMError):
        c._one(list(_TURNS), temperature=1.0, n_img=0)
    assert len(seen) == 1
