"""Two-stage generation (`generate.output.two_stage_nl_then_code`, GT and L2R)
never sends an empty assistant turn, and records which stage came back empty.

Appending the thinker's prose as `{"role": "assistant", "content": prose}`
unconditionally would be fatal. An empty prose -- the client's designed output
for a refusal that survives its retries, for a max_tokens truncation, or a short
return padded to "" -- would go to the coder as an empty NON-final assistant
message, which the Messages API rejects with a 400 that `_degrade` cannot match
to any droppable pin; `_call` re-raises and nothing up to `run()` catches it.
One refused thinker sample among GT's ten would kill the whole search
mid-iteration, where every single-stage method records the same transient as one
forfeited slot and finishes. The offline suite cannot see this without help: the
mock accepts empty content.

The thinker is silenced here by a wrapper around the real mock that answers ""
to the spec request (the one whose tail says "Write NO code in this turn") for
chosen sample indices and delegates everything else; the wrapper also records
every request so the assertion is on what was SENT.
"""

from __future__ import annotations

import importlib.util
import random

import pytest

from conftest import REPO
from bird import registry
from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.state import RunState

_THINKER_TAIL = "Write NO code in this turn"
_CODER_HEAD = "Now implement exactly that specification"


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _SilentThinker:
    """The real MockLLM, except that thinker requests for the sample indices in
    `silent` (None = all of them) come back empty with a parked reason."""

    def __init__(self, inner, silent=None):
        self.inner = inner
        self.silent = silent
        self.requests = []
        self.last_empty_reasons = []
        self.n_thinker = 0

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        msgs = [dict(m) for m in messages]
        self.requests.append(msgs)
        if _THINKER_TAIL in str(msgs[-1].get("content", "")):
            out, reasons = [], []
            for _ in range(int(n)):
                idx, self.n_thinker = self.n_thinker, self.n_thinker + 1
                if self.silent is None or idx in self.silent:
                    out.append("")
                    reasons.append("refused by the provider (test stub)")
                else:
                    out.append(self.inner(messages, n=1, temperature=temperature,
                                          images=images, tag=tag)[0])
                    reasons.append("")
            self.last_empty_reasons = reasons
            return out
        got = self.inner(messages, n=n, temperature=temperature, images=images, tag=tag)
        self.last_empty_reasons = list(getattr(self.inner, "last_empty_reasons", []) or [])
        return got

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _ctx(n: int, **overrides):
    registry.load_all()
    # keep_top_n must not exceed the pool (a coherence rule); the screen is not under test
    cfg = load("gt", profile="tester", overrides={"generate.n_candidates": n,
                                                  "verify.alignment_filter.keep_top_n": 1,
                                                  **overrides})
    assert cfg["generate.output.two_stage_nl_then_code"] is True, "precondition: gt is two-stage"
    # Seeded as `_execute` seeds it: the mock salts every sample's RNG from
    # `ctx.rng`, so an unseeded Context would make the run non-reproducible
    # here and only here.
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(cfg["seed"]))
    ctx.generator = registry.get("llm", "mock")(ctx, role="generator")
    ctx.evaluator = registry.get("llm", "mock")(ctx, role="evaluator")
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    ctx.human = registry.get("phase", "human_oracle")(ctx)
    return ctx


def _empty_assistant_turns(requests):
    return [m for req in requests for m in req
            if m.get("role") == "assistant" and not str(m.get("content", "")).strip()]


def _coder_requests(requests):
    return [req for req in requests if _CODER_HEAD in str(req[-1].get("content", ""))]


@pytest.mark.parametrize("mode", ["iid_parallel", "distinct_prompts", "personas",
                                  "sequential_conditioned"])
def test_an_empty_thinker_reply_is_a_recorded_empty_candidate_not_a_request(mode) -> None:
    entry = _entry()
    ctx = _ctx(3, **{"generate.sampling_mode": mode})
    client = ctx.generator = _SilentThinker(ctx.generator, silent=None)

    cands = entry.generate(ctx, RunState())

    assert len(cands) == 3
    assert not _empty_assistant_turns(client.requests), (
        "an empty assistant turn was sent to the coder -- the Messages API rejects it")
    assert not _coder_requests(client.requests), "the coder stage ran on an empty spec"
    for c in cands:
        assert not c.valid and c.failure_kind == "invalid"
        assert c.meta.get("empty_response") is True
        assert c.meta.get("empty_stage") == "thinker"
        assert "thinker stage returned nothing" in c.failure and "test stub" in c.failure
        assert c.prompt_messages and _THINKER_TAIL in c.prompt_messages[-1]["content"], (
            "the artifact should carry the request that produced the empty reply")


def test_one_silent_thinker_forfeits_one_slot_and_the_rest_proceed() -> None:
    entry = _entry()
    ctx = _ctx(3, **{"generate.sampling_mode": "iid_parallel"})
    client = ctx.generator = _SilentThinker(ctx.generator, silent={1})

    cands = entry.generate(ctx, RunState())

    assert len(cands) == 3
    assert not _empty_assistant_turns(client.requests)
    coder = _coder_requests(client.requests)
    assert len(coder) == 2, "exactly the two non-empty specs reached the coder"
    for req in coder:
        assert req[-2]["role"] == "assistant" and req[-2]["content"].strip()
    empty = [c for c in cands if c.meta.get("empty_response")]
    assert [c.meta["sample_index"] for c in empty] == [1]
    assert empty[0].meta["empty_stage"] == "thinker"
    assert all(c.raw_response.strip() for c in cands if c is not empty[0])


def test_an_empty_coder_reply_is_labelled_as_such() -> None:
    """The other stage: the spec arrived, the code did not."""
    from bird.components import generation

    class _SilentCoder(_SilentThinker):
        def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
            msgs = [dict(m) for m in messages]
            if _CODER_HEAD in str(msgs[-1].get("content", "")):
                self.requests.append(msgs)
                self.last_empty_reasons = ["truncated (test stub)"] * int(n)
                return [""] * int(n)
            return super().__call__(messages, n=n, temperature=temperature, images=images, tag=tag)

    entry = _entry()
    ctx = _ctx(1)
    ctx.generator = _SilentCoder(ctx.generator, silent=set())
    (cand,) = entry.generate(ctx, RunState())
    assert cand.meta.get("empty_response") is True
    assert cand.meta.get("empty_stage") == "coder"
    assert "test stub" in cand.failure
    assert generation._two_stage_meta("code", "") == {}


# --------------------------------------------------------------------------
# under the tester profile the thinker turn is answered with PROSE
# --------------------------------------------------------------------------


def test_the_mock_routes_the_real_thinker_prompt_to_prose_and_the_coder_to_code() -> None:
    from bird.components import generation
    from bird.llm.base import messages_text
    ctx = _ctx(1)
    mock = ctx.generator
    thinker = generation._build_messages(ctx, RunState(), [], tail=generation._SPEC_TAIL,
                                         call_index=generation._CALL_THINKER)
    assert mock._route("generate", messages_text(thinker)) == "nl_spec", (
        "the thinker turn is tagged like every stage-1 call; the mock must read its contract")
    coder = generation._build_messages(ctx, RunState(), [], call_index=generation._CALL_CODER)
    coder.append({"role": "assistant", "content": "A dense distance term and a small effort penalty."})
    coder.append({"role": "user", "content": _CODER_HEAD + ", changing nothing about it."})
    assert mock._route("generate", messages_text(coder)) == "program"


def test_a_tester_tier_gt_candidate_carries_prose_in_nl_spec_and_code_in_reward_code() -> None:
    """A GT tester candidate must record prose, not a fenced program, as its
    'specification', and hand that prose to the coder as the spec."""
    entry = _entry()
    ctx = _ctx(3, **{"generate.sampling_mode": "iid_parallel"})
    cands = entry.generate(ctx, RunState())
    assert len(cands) == 3
    for c in cands:
        assert c.nl_spec.strip(), "the thinker answered nothing"
        assert "```" not in c.nl_spec and "def " not in c.nl_spec, c.nl_spec[:200]
        assert c.meta.get("two_stage") is True
        assert c.raw_response.strip(), "the coder answered nothing"
        # the coder was handed the PROSE as its specification
        assert c.prompt_messages[-2]["role"] == "assistant"
        assert c.prompt_messages[-2]["content"] == c.nl_spec


def test_the_prose_stage_is_deterministic_under_seed() -> None:
    entry = _entry()

    def run():
        ctx = _ctx(3, **{"seed": 11})
        return [(c.nl_spec, c.reward_code, c.valid) for c in entry.generate(ctx, RunState())]
    assert run() == run()
