"""A verification repair is ONE LLM call and is charged once.

`llm/base.py`'s contract is that a client records its own usage -- one
`record_llm` per provider round-trip, with the provider's exact token counts.
A second row recorded on top of it by `verification._regenerate`, built from a
characters-over-four estimate whenever `verify.repairs_count_against_budget` is
on (every published config: the default is `true`), would make one repair read
as two calls and exact-plus-estimated tokens, make `budget.max_llm_calls` bite
twice as early on the repair path, and make CARD -- whose headline result is
the token cost of its repair loop -- pay double in the cost column with every
counter reading normal. The repair path therefore reconciles the way stage 1
does (`generation._call_llm`): snapshot `llm_calls`, record the estimate only
when the client recorded nothing.

The first test below is the direct probe: one synthetic completion that
records (10, 20) must come back as one call with exactly those counts, not as
`budget_calls=2, prompt_tokens=13, completion_tokens=34`.
"""
from __future__ import annotations

from types import SimpleNamespace

from bird.budget import Budget
from bird.components.verification import _regenerate
from bird.config import load
from bird.context import Context
from bird.llm.base import LLMClient
from bird.state import RunState
from bird.types import Candidate

_REPLY = "```python\ndef reward(s, a, s2): return GENERAL_SYMBOL, {}\n```"
_PROMPT = [{"role": "user", "content": "repair prompt"}]


class _CountingClient(LLMClient):
    """What every real provider does: one `_record` per round-trip, with the
    provider's own token counts."""

    def _complete(self, messages, n, temperature, images, tag):
        self._record(10, 20)
        return [_REPLY] * n


def _ctx(**overrides) -> Context:
    cfg = load("card", overrides=overrides)
    return Context(cfg=cfg, budget=Budget(),
                   env=SimpleNamespace(symbol_mapping={"GENERAL_SYMBOL": "s[0]"}))


def _candidate() -> Candidate:
    return Candidate("bad", 0, "def reward(s, a, s2): return 1/0", prompt_messages=list(_PROMPT))


def test_a_repair_through_a_recording_client_is_charged_exactly_once():
    ctx = _ctx()
    assert ctx.cfg.get("verify.repairs_count_against_budget") is True, "CARD's published value"
    ctx.generator = _CountingClient(ctx, "generator")

    fresh, error = _regenerate(ctx, RunState(), _candidate(), 1)

    assert error == "" and fresh is not None
    assert ctx.generator.n_calls == 1, "the provider was asked once"
    assert ctx.budget.llm_calls == 1, "so the budget saw one call, not two"
    # The client's exact counts, not the client's plus an estimate of the same.
    assert ctx.budget.llm_prompt_tokens == 10
    assert ctx.budget.llm_completion_tokens == 20
    assert ctx.budget.verify_resamples == 1


def test_a_repair_through_a_bare_callable_is_still_charged_once():
    """The estimate exists for a client that records nothing (a callable
    standing in for one). It must fire then -- silently charging zero would
    make a method look free -- and only then."""
    ctx = _ctx()
    ctx.generator = lambda messages, **kw: [_REPLY]

    fresh, error = _regenerate(ctx, RunState(), _candidate(), 1)

    assert error == "" and fresh is not None
    assert ctx.budget.llm_calls == 1
    assert ctx.budget.llm_prompt_tokens == len(_PROMPT[0]["content"]) // 4
    assert ctx.budget.llm_completion_tokens == len(_REPLY) // 4


def test_a_recording_client_is_not_charged_twice_across_repeated_repairs():
    """A double charge compounds: `budget.max_llm_calls` would reach its cap in
    half the repairs. Three repairs are three calls."""
    ctx = _ctx()
    ctx.generator = _CountingClient(ctx, "generator")
    cand = _candidate()
    for attempt in (1, 2, 3):
        cand, error = _regenerate(ctx, RunState(), cand, attempt)
        assert error == ""
    assert ctx.generator.n_calls == 3
    assert ctx.budget.llm_calls == 3
    assert (ctx.budget.llm_prompt_tokens, ctx.budget.llm_completion_tokens) == (30, 60)
