"""A VLM judge whose sighted call fails is NOT re-asked without its frames.

`evaluation._client_text` is the one call helper behind RDA's in-loop subtask
scorer (`_vlm_trajectory_analysis`), the curriculum gate (`_score_stage`) and
GT's post-hoc `alignment_rate`. A ladder of [with images, without images] per
call shape, where any exception OR an empty answer on the sighted call falls
through to the blind one, is wrong: those are exactly the shapes
`AnthropicClient` produces on a `refusal` or `max_tokens` stop (`_one` returns
""), on a non-retryable 4xx such as an oversized image payload, and on an
exhausted retry ladder (`LLMError`). A refused 20-frame request would be
silently re-asked from the text tables alone; the caller has already written
`n_images=20` to its query record, before the call, so the blind verdict would
be filed as an image-graded one and `budget.vlm_calls` would count two calls.
A VLM-graded method would run partly text-graded with every counter reading
normal -- a variant of the "images declared, not attached" failure.

The contract: frames supplied -> every attempt carries them -> a failure
returns `""`, which every caller already treats as "no verdict" (RDA falls
back to the env's success flag and records `fallback`; the gate scores None;
`alignment_rate` counts `n_unparsed`), and `judgments.note_response` files it
as `no_answer` against a query that says how many frames it carried.
"""
from __future__ import annotations

import numpy as np
import pytest

from test_judgments import _VLM, _ctx, _recs, _score, _traj

from bird.components.evaluation import _client_text, client_sees_images
from bird.state import CurriculumState, RunState
from bird.types import Candidate, CandidateReport, TrainResult


class _RefusesFrames(_VLM):
    """`[""]` whenever frames are attached -- what `LLMClient.__call__` returns
    when `_one` hits a `refusal` or `max_tokens` stop -- and a confident verdict
    when asked blind. The blind verdict is the bait: a blind fallback takes it."""

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        super().__call__(messages, n=n, temperature=temperature, images=images, tag=tag)
        return [""] * n if images else [self.reply] * n


class _ErrorsOnFrames(_VLM):
    """Raises on the sighted call -- a 4xx the client does not retry, or an
    exhausted retry ladder -- and answers blind."""

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        super().__call__(messages, n=n, temperature=temperature, images=images, tag=tag)
        if images:
            raise RuntimeError("request_too_large")
        return [self.reply] * n


# --------------------------------------------------------------------------
# the helper
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [_RefusesFrames, _ErrorsOnFrames],
                         ids=["empty_answer", "raises"])
def test_a_failed_sighted_call_is_not_retried_without_its_frames(cls):
    client = cls("Score: 0.9\nReason: looks done.")
    assert client_sees_images(client), "precondition: the callers would attach frames"
    assert _client_text(client, "Rate this rollout", images=[b"png-1", b"png-2"]) == "", (
        "the blind answer came back as the verdict")
    assert len(client.calls) == 1, f"expected exactly one (sighted) call, got {client.calls}"
    assert client.calls[0]["n_images"] == 2, "the one call did not carry the frames"


def test_a_client_handed_no_frames_is_asked_once_and_answers():
    """The text path is unchanged: no images -> one call -> its text. This is
    every `modality: text` client, whose callers hand it no frames because
    `client_sees_images` said not to."""
    client = _RefusesFrames("Score: 0.9")
    assert _client_text(client, "Rate this rollout") == "Score: 0.9"
    assert len(client.calls) == 1 and client.calls[0]["n_images"] == 0


def test_the_frames_ride_along_on_every_call_shape():
    """Duck typing across call SHAPES survives -- but never at the cost of the
    frames. A client whose only shape is `complete(prompt, images=...)` is
    reached through the TypeError on the message-list attempt and gets its
    images on the attempt that works."""
    class _PromptOnly:
        modality = "vlm"

        def __init__(self):
            self.carried = []

        def complete(self, prompt, images=None):
            if not isinstance(prompt, str):
                raise TypeError("prompt must be a string")
            self.carried.append(len(images or ()))
            return "Score: 0.5"

    client = _PromptOnly()
    assert _client_text(client, "Rate this", images=[b"png"]) == "Score: 0.5"
    assert client.carried == [1]


# --------------------------------------------------------------------------
# the callers, through the judgment record
# --------------------------------------------------------------------------

def test_a_refused_sighted_judgment_is_filed_as_no_answer_not_as_a_verdict(tmp_path):
    """RDA's in-loop scorer. The query record states the intent (`n_images`,
    written before the call); the response must then say NO verdict was
    reached -- not carry a score the judge produced blind."""
    ctx = _ctx(tmp_path, images=4)
    ctx.generator = ctx.evaluator = _RefusesFrames(
        '```json\n{"subtasks": [{"number": 1, "score": 1.0}]}\n```')
    _score(ctx, n_rollouts=1)

    calls = ctx.generator.calls
    queries = _recs(ctx, "query", role="vlm_subtask_score")
    assert queries and all(q["n_images"] == 4 for q in queries)
    assert calls and all(c["n_images"] == 4 for c in calls), (
        "a call was made without the frames the query record claims")
    assert len(calls) == len(queries), (
        f"{len(calls)} calls for {len(queries)} queries: the judge was re-asked")
    answers = _recs(ctx, "response", role="vlm_subtask_score")
    assert answers
    for a in answers:
        assert a["failure_kind"] == "no_answer" and a["parsed"] is None, a
        # No GT-success fallback: a judge that
        # produced nothing ABSTAINS under a real provider (`fallback: abstain`,
        # `used: None`) and only `provider: mock` keeps a synthetic stand-in.
        assert a["fallback"] == "abstain" and a["used"] is None, a
        assert not (a.get("raw") or "").strip(), "a blind verdict was recorded as raw"


def test_a_refused_gate_holds_the_stage_and_says_nobody_answered(tmp_path):
    """The curriculum gate, same client. `score: None` with `n_answered: 0` is
    the documented "dead provider" state -- distinct from a stage that failed
    -- and the response records must agree with it."""
    # The gate reads only its own `gate_votes` / `gate_threshold`; enabling the
    # whole curriculum would drag in carry-slot coherence rules this test is
    # not about.
    ctx = _ctx(tmp_path, images=4, **{"loop.curriculum.gate_votes": 2})
    from bird.components.curriculum import gate_vlm_ensemble

    ctx.generator = ctx.evaluator = _RefusesFrames("Score: 1.0\nReason: done.")
    cand = Candidate(cand_id="c0", iteration=0,
                     reward_code="def compute_reward(state, action=None): return 0.0")
    res = TrainResult(cand_id="c0", candidate=cand, trajectories=[_traj()])
    report = CandidateReport(cand_id="c0", candidate=cand, result=res, fitness=1.0)
    state = RunState()
    state.curriculum = CurriculumState(stages=["a", "b"], index=0)

    out = gate_vlm_ensemble(ctx, state, report, "a", 0)
    assert out["n_images"] == 4, out  # the frames WERE cut and sent
    assert out["score"] is None and out["n_answered"] == 0 and out["passed"] is False, out
    assert len(ctx.evaluator.calls) == 2, ctx.evaluator.calls
    assert all(c["n_images"] == 4 for c in ctx.evaluator.calls)
    answers = _recs(ctx, "response", role="curriculum_gate")
    assert len(answers) == 2
    assert all(a["failure_kind"] == "no_answer" and a["parsed"] is None for a in answers)
