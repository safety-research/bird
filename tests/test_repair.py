"""Stage 2's repair path (`verify.on_failure: resample_with_trace` / `resample_blind`):
what a repair CHARGES and what it SENDS.

Both halves can go wrong silently:

* `_regenerate` calls the generator through `_llm_text`, whose client records
  its own round-trip, per the `bird/llm/base.py` contract. Charging a second
  chars/4 estimate on top whenever `verify.repairs_count_against_budget` is
  true (the default in every config) would make one repair cost `llm_calls` +2
  and ~2x the tokens, while with the key false the client's record would stand,
  so repairs would never be excluded either -- the key would do the opposite of
  what it declares, on exactly the method points (gt, rf_agent, card, revolve)
  whose argument is cost.
* A repair request that carries the original prompt and the traceback but not
  the program that failed leaves `line 14` in the trace pointing into code the
  model cannot see. `configs/methods/rf_agent.yaml` cites the release's repair prompt,
  whose first line is the failed program.

Everything here runs the real `MockLLM` under the `tester` profile, offline.
"""

from __future__ import annotations

import pytest

from bird import registry
from bird.budget import Budget
from bird.components import verification
from bird.config import load
from bird.context import Context
from bird.llm.base import estimate_tokens, messages_text
from bird.types import Candidate

#: A program that parses but cannot run: the shape `execution_smoke` catches.
_BROKEN = "def compute_reward(state, action, next_state):\n    return dist\n"
#: What `_probe` forwards: the tail of the traceback, line numbers and no source.
_TRACE = ('File "<candidate c0000>", line 2, in compute_reward\n'
          "NameError: name 'dist' is not defined")


class _Recording:
    """Wraps a client so a test can see the request that actually went out."""

    def __init__(self, inner):
        self.inner = inner
        self.requests = []

    def __call__(self, messages):
        self.requests.append([dict(m) for m in messages])
        return self.inner(messages, tag="repair")


def _ctx(name: str = "gt", **overrides) -> Context:
    registry.load_all()
    cfg = load(name, profile="tester", overrides=overrides or None)
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.generator = registry.get("llm", "mock")(ctx, role="generator")
    return ctx


def _failed(trace: str = _TRACE) -> Candidate:
    c = Candidate(
        cand_id="c0000", iteration=0, reward_code=_BROKEN,
        raw_response="Here is the reward.\n```python\n" + _BROKEN + "```",
        prompt_messages=[{"role": "system", "content": "You write reward functions."},
                         {"role": "user", "content": "Write compute_reward for the reacher."}])
    if trace:
        c.meta[verification._REPAIR_TRACE] = trace
    return c


# --------------------------------------------------------------------------
# what a repair charges
# --------------------------------------------------------------------------


def test_a_repair_is_charged_exactly_once_when_repairs_count() -> None:
    """One provider round-trip, one `record_llm`: the client's own, not the
    client's plus an estimate on top."""
    ctx = _ctx(**{"verify.repairs_count_against_budget": True})
    rec = ctx.generator = _Recording(ctx.generator)

    fresh, err = verification._regenerate(ctx, None, _failed(), attempt=1)

    assert fresh is not None, err
    assert rec.inner.n_calls == 1
    assert ctx.budget.llm_calls == 1, (
        f"llm_calls={ctx.budget.llm_calls} for one repair: the client's record "
        f"and _regenerate's estimate were both charged")
    assert ctx.budget.verify_resamples == 1
    # The tokens are the CLIENT'S figure for the request it sent -- not that
    # figure plus a second chars/4 pass over the same messages.
    assert ctx.budget.llm_prompt_tokens == estimate_tokens(messages_text(rec.requests[0]))
    assert ctx.budget.llm_completion_tokens == estimate_tokens(fresh.raw_response)


def test_n_repairs_are_n_calls_when_repairs_count() -> None:
    ctx = _ctx(**{"verify.repairs_count_against_budget": True})
    current = _failed()
    for attempt in range(1, 4):
        current.meta[verification._REPAIR_TRACE] = _TRACE
        current, err = verification._regenerate(ctx, None, current, attempt=attempt)
        assert current is not None, err
    assert ctx.budget.llm_calls == 3 == ctx.budget.verify_resamples


def test_a_repair_is_not_charged_at_all_when_repairs_do_not_count() -> None:
    """`false` means EXCLUDED: the call happens (the client counts it), the
    repair is counted as a repair, and the LLM cost column does not move."""
    ctx = _ctx(**{"verify.repairs_count_against_budget": False})

    fresh, err = verification._regenerate(ctx, None, _failed(), attempt=1)

    assert fresh is not None, err
    assert ctx.generator.n_calls == 1, "the repair was not made at all"
    assert ctx.budget.verify_resamples == 1
    assert (ctx.budget.llm_calls, ctx.budget.llm_prompt_tokens,
            ctx.budget.llm_completion_tokens) == (0, 0, 0), (
        "verify.repairs_count_against_budget=false still charged the repair")

    # The latch is scoped to the repair: the next ordinary call counts again.
    ctx.generator([{"role": "user", "content": "write a reward"}], tag="generate")
    assert ctx.budget.llm_calls == 1


def test_a_client_that_records_nothing_is_still_charged_once() -> None:
    """The estimate is a fallback for a client outside the `bird.llm` contract,
    not a surcharge on one inside it."""
    class Bare:
        def __call__(self, messages):
            return "```python\n" + _BROKEN + "```"

    ctx = _ctx(**{"verify.repairs_count_against_budget": True})
    ctx.generator = Bare()
    fresh, err = verification._regenerate(ctx, None, _failed(), attempt=1)
    assert fresh is not None, err
    assert ctx.budget.llm_calls == 1
    assert ctx.budget.llm_prompt_tokens > 0 and ctx.budget.llm_completion_tokens > 0


@pytest.mark.parametrize("counted", [True, False])
def test_the_whole_validity_phase_keeps_calls_equal_to_repairs(counted: bool) -> None:
    """Through the public stage, on the method point that repairs uncapped (GT):
    every repair is exactly one call when counted and none when not, whatever
    the mock decides to do with each attempt."""
    ctx = _ctx(**{"verify.repairs_count_against_budget": counted})
    ctx.env = registry.get("env", ctx.cfg["problem.env_id"])(ctx)
    from bird.state import RunState
    state = RunState()

    out = verification.run_validity(ctx, state, [_failed(trace="")])

    assert len(out) == 1
    assert ctx.budget.verify_resamples >= 1, "the broken program was never repaired"
    expect = ctx.budget.verify_resamples if counted else 0
    assert ctx.budget.llm_calls == expect, (
        f"counted={counted}: {ctx.budget.verify_resamples} repairs, "
        f"{ctx.budget.llm_calls} llm_calls")


def test_uncounted_llm_is_a_scoped_latch() -> None:
    b = Budget()
    with b.uncounted_llm():
        b.record_llm(prompt_tokens=10, completion_tokens=5)
        with b.uncounted_llm():
            b.record_llm(prompt_tokens=10, completion_tokens=5)
        b.record_llm(prompt_tokens=10, completion_tokens=5)
    assert (b.llm_calls, b.llm_prompt_tokens, b.llm_completion_tokens) == (0, 0, 0)
    b.record_llm(prompt_tokens=10, completion_tokens=5)
    assert (b.llm_calls, b.llm_prompt_tokens, b.llm_completion_tokens) == (1, 10, 5)


# --------------------------------------------------------------------------
# what a repair sends
# --------------------------------------------------------------------------


def test_a_traced_repair_shows_the_model_the_program_that_failed() -> None:
    """`resample_with_trace`: original request, then the failed program as the
    assistant turn it was, then the trace -- so `line N` in the trace refers to
    code in the conversation. Through the real MockLLM."""
    ctx = _ctx()
    rec = ctx.generator = _Recording(ctx.generator)
    cand = _failed()

    fresh, err = verification._regenerate(ctx, None, cand, attempt=1)

    assert fresh is not None, err
    sent = rec.requests[0]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"], (
        "the repair request never showed the model the program that failed")
    assert sent[:2] == cand.prompt_messages
    assert sent[2]["content"] == cand.raw_response
    assert _BROKEN in sent[2]["content"]
    assert _TRACE in sent[3]["content"]
    assert fresh.meta["repair_trace_sent"] is True
    assert fresh.meta["repaired_from"] == cand.cand_id


def test_a_traced_repair_falls_back_to_the_fenced_code_without_a_raw_response() -> None:
    """A candidate that reached stage 2 without its raw turn (a resumed run's
    record, a test-built candidate) still shows the program, fenced."""
    ctx = _ctx()
    rec = ctx.generator = _Recording(ctx.generator)
    cand = _failed()
    cand.raw_response = ""

    fresh, err = verification._regenerate(ctx, None, cand, attempt=1)

    assert fresh is not None, err
    shown = rec.requests[0][2]
    assert shown["role"] == "assistant"
    assert shown["content"].startswith("```python\n") and _BROKEN in shown["content"]


def test_a_blind_resample_shows_the_model_nothing() -> None:
    """`resample_blind` (CARD) is the identical request and no more: no failed
    draft, no trace -- the one-field contrast with `resample_with_trace`."""
    ctx = _ctx("card")
    rec = ctx.generator = _Recording(ctx.generator)
    cand = _failed(trace="")

    fresh, err = verification._regenerate(ctx, None, cand, attempt=1)

    assert fresh is not None, err
    assert rec.requests[0] == cand.prompt_messages
    assert fresh.meta["repair_trace_sent"] is False


# --------------------------------------------------------------------------
# one extractor for both stages
# --------------------------------------------------------------------------


def _patterns(ctx):
    return ctx.cfg["generate.parse.patterns"]


def test_both_stages_strip_a_language_tag_identically() -> None:
    """Under the default pattern list a ```py fence is caught by the bare
    ```(.*?)``` pattern with its tag; a separate repair-path extractor would
    keep the tag as the program's first line."""
    from bird.components import generation
    ctx = _ctx("card")
    raw = "Here you go.\n```py\ndef compute_reward(state, action, next_state):\n    return 0.0\n```\n"
    code1, _ = generation._extract_code(ctx, raw)
    code2 = verification._extract_code(ctx, raw)
    assert code1 == code2, (code1, code2)
    assert code1.startswith("def compute_reward"), code1
    for tag in ("python", "json"):
        assert verification._extract_code(ctx, raw.replace("```py", f"```{tag}")) == code1


def test_an_indented_fence_body_and_a_bare_program_parse_the_same_both_ways() -> None:
    from bird.components import generation
    ctx = _ctx("card")
    indented = "```python\n    def compute_reward(s, a, s2):\n        return 0.0\n```"
    bare = "def compute_reward(s, a, s2):\n    return 1.0\n"
    for raw in (indented, bare):
        code1, _ = generation._extract_code(ctx, raw)
        assert code1 is not None and code1 == verification._extract_code(ctx, raw)


def test_a_prose_only_reply_is_no_code_for_both_stages() -> None:
    from bird.components import generation
    ctx = _ctx("card")
    prose = "I cannot write that reward; the task is underspecified."
    assert generation._extract_code(ctx, prose) == (None, "")
    assert verification._extract_code(ctx, prose) == ""


def test_a_repaired_py_fence_is_a_runnable_program() -> None:
    class PyFence:
        def __call__(self, messages):
            return "```py\ndef compute_reward(state, action, next_state):\n    return 0.5\n```"
    ctx = _ctx()
    ctx.generator = PyFence()
    fresh, err = verification._regenerate(ctx, None, _failed(), attempt=1)
    assert fresh is not None, err
    assert fresh.reward_code.startswith("def compute_reward"), fresh.reward_code
    assert "repair_no_code" not in fresh.meta


def test_a_prose_only_repair_reply_keeps_retrying_under_the_cap_with_an_honest_label() -> None:
    """The retry semantics upstream CARD has: a reply with no program is a
    failed attempt that is resampled under `verify.max_repair_attempts`, not
    an exhaustion -- and it is labelled as no code, not as a SyntaxError."""
    class Prose:
        def __call__(self, messages, **kw):
            return "I would rather describe the reward than write it."
    ctx = _ctx(**{"verify.max_repair_attempts": 2})
    ctx.generator = Prose()
    ctx.env = registry.get("env", ctx.cfg["problem.env_id"])(ctx)
    from bird.state import RunState

    (settled,) = verification.run_validity(ctx, RunState(), [_failed(trace="")])

    assert not settled.valid and settled.failure_kind == "invalid"
    assert ctx.budget.verify_resamples == 2, "the prose reply must be resampled under the cap"
    assert settled.meta.get("repair_no_code") is True
    assert settled.meta.get("repair_attempt") == 2
    assert "cannot parse" not in settled.failure, settled.failure
    assert settled.reward_code == ""
