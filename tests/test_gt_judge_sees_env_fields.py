"""GT's pairwise judge is shown the ENVIRONMENT's fields, never the candidate's
own reward.

App. C's caption prompt hands the VLM "frame-by-frame data coming directly from
the environment's API" and, once, ahead of both clips, "Documentation about each
frame's data" (refs/tex/gt_reward_design/neurips_2025.tex:682-688). Two things
must hold of what `preferences._render` puts in that slot:

  1. no trajectory row ends in `r=<the candidate's OWN per-step reward>`. A
     reward is not an environment field, GT's judge never sees one, and a reward
     can inflate it -- in stored real runs, 4 of 4 captions for a round-1 winner
     cited the reward trace as the evidence of task success, so the preference
     labels (GT's stand-in for the inaccessible F) become partly a function of
     the candidate's self-report;
  2. the rows carry documentation: `s=[-0.78 -0.375 0 ...]` with no names is
     unreadable, and every adapter declares `_state_fields` / `_action_fields`
     and the generator's `env_spec_state_action_api_stub` already renders them.

Each claim below fails independently. The suite stays OFFLINE: the client is a
stub that records what it was handed.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird import config as cfgmod  # noqa: E402
from bird import registry  # noqa: E402
from bird.budget import Budget  # noqa: E402
from bird.components.preferences import (  # noqa: E402
    _render, comparator_llm_on_vlm_captions, comparator_vlm)
from bird.context import Context  # noqa: E402
from bird.state import RunState  # noqa: E402
from bird.types import Candidate, CandidateReport, TrainResult, Trajectory  # noqa: E402

#: A per-step reward whose rendering (`f"{x:.4f}"` -> "12.3457") no other number
#: in a pendulum prompt can produce by accident.
REWARD = 12.3457

#: Pendulum's declared fields (`tasks/pendulum/shared_spec.yaml`), as the legend
#: names them: `s[i] <name>` / `a[i] <name>`. Keyed on the LEGEND ROW and not the
#: bare name, because "torque" is also a word in pendulum's task description and a
#: bare-name check would pass with no documentation at all.
STATE_FIELDS = ("cos_theta", "sin_theta", "theta_dot")
ACTION_FIELDS = ("torque",)
LEGEND_ROWS = tuple(f"s[{i}] {n}" for i, n in enumerate(STATE_FIELDS)) + \
    tuple(f"a[{i}] {n}" for i, n in enumerate(ACTION_FIELDS))
#: The adapter's doc string for `cos_theta`, so names alone cannot satisfy this.
DOC_TEXT = "cosine of the angle"

#: The `r=` token as `_render` wrote it: `t=... s=[...] a=[...] r=<float>`.
REWARD_COLUMN = re.compile(r"\br=")


class _SpyVLM:
    """A client honouring `bird/llm/base.py`, which records what it was sent."""

    modality = "vlm"

    def __init__(self, reply: str = "LEFT") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        text = messages if isinstance(messages, str) else "\n".join(
            str(m.get("content", "")) for m in messages)
        self.calls.append({"prompt": text, "n_images": len(images or ()), "tag": tag})
        return [self.reply] * n


def _ctx(**overrides):
    registry.load_all()
    base = {"llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "output.tracker": "none", "problem.env_id": "pendulum"}
    base.update(overrides)
    cfg = cfgmod.load("gt", overrides=base, profile="dev")
    ctx = Context(cfg=cfg, budget=Budget(), rundir=None, rng=random.Random(0))
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    ctx.evaluator = _SpyVLM()
    return ctx


def _report(i: int, n: int = 400) -> CandidateReport:
    """One agent's rollout, held at an angle that differs per `i`, paying itself
    `REWARD` at every step."""
    arr = np.array([[np.cos(0.3 * i), np.sin(0.3 * i), 0.0]] * n, dtype=float)
    acts = np.array([[0.5 * i]] * n, dtype=float)
    traj = Trajectory(states=arr, actions=acts, rewards=[REWARD] * n,
                      success=bool(i % 2), length=n, ret=REWARD * n)
    cand = Candidate(cand_id=f"c{i}", iteration=0,
                     reward_code="def compute_reward(state, action=None): return 0.0")
    res = TrainResult(cand_id=cand.cand_id, candidate=cand, trajectories=[traj])
    return CandidateReport(cand_id=cand.cand_id, candidate=cand, result=res,
                           fitness=float(i))


def _caption_prompt(ctx) -> str:
    comparator_llm_on_vlm_captions(ctx, RunState(), _report(1), _report(2))
    calls = ctx.evaluator.calls
    assert len(calls) == 2, "caption step then decide step"
    return calls[0]["prompt"]


# --------------------------------------------------------------------------
# 1 -- the candidate's reward is not an environment field
# --------------------------------------------------------------------------


def test_the_caption_prompt_carries_no_reward_column():
    """tex:682: the VLM receives environment-API data. The candidate's own
    per-step reward is not that, and the judge must not be able to read task
    success off it."""
    prompt = _caption_prompt(_ctx())
    assert "t=" in prompt, "the trajectory rows themselves must still be shown"
    assert not REWARD_COLUMN.search(prompt), (
        "a trajectory row still carries the candidate's own reward:\n"
        + "\n".join(l for l in prompt.splitlines() if REWARD_COLUMN.search(l))[:400])
    assert f"{REWARD:.4f}" not in prompt, "the reward's value leaked into the prompt"


def test_render_alone_writes_no_reward_column():
    """The rows are rendered in `_render`, so pin it there too: whoever composes
    a prompt out of `_render`'s text -- the one-step `vlm`, the two-step
    captioner, the human comparator -- inherits the property."""
    text = _render(_ctx(), _report(3), "LEFT", attach=False)
    assert "t=" in text
    assert not REWARD_COLUMN.search(text), text


# --------------------------------------------------------------------------
# 2 -- the environment's documentation, once, ahead of both clips
# --------------------------------------------------------------------------


def test_the_caption_prompt_documents_the_environment_fields():
    """tex:682: 'Documentation about each frame's data is given here:
    {variables_documentation}'. Names AND their meaning, from the adapter's own
    declaration -- the legend `env_spec_state_action_api_stub` already renders
    for the generator."""
    prompt = _caption_prompt(_ctx())
    for row in LEGEND_ROWS:
        assert row in prompt, f"legend row {row!r} is not in the caption prompt"
    assert DOC_TEXT in prompt, "field documentation (not just names) must reach the judge"


def test_the_documentation_appears_once_ahead_of_both_clips():
    """App. C places the block once, before `{frames_and_observations_1}`; a
    copy per side would be the same tokens twice."""
    prompt = _caption_prompt(_ctx())
    first_field = prompt.index(LEGEND_ROWS[0])
    assert prompt.count(DOC_TEXT) == 1, "documentation rendered more than once"
    assert first_field < prompt.index("[LEFT]") < prompt.index("[RIGHT]"), (
        "the documentation must precede both clips, as the paper's template does")


def test_the_one_step_comparator_gets_the_same_documentation():
    """`comparator_vlm` is the ablation against the captioner; the two must
    differ in the caption bottleneck and nothing else."""
    ctx = _ctx()
    comparator_vlm(ctx, RunState(), _report(1), _report(2))
    prompt = ctx.evaluator.calls[0]["prompt"]
    for row in LEGEND_ROWS:
        assert row in prompt
    assert DOC_TEXT in prompt
    assert not REWARD_COLUMN.search(prompt)


def test_the_decider_never_sees_the_documentation():
    """Step 2 reads the caption and the task and never the clips (§4); frame
    documentation without frames would be noise in its context."""
    ctx = _ctx()
    comparator_llm_on_vlm_captions(ctx, RunState(), _report(1), _report(2))
    decide = ctx.evaluator.calls[1]["prompt"]
    for row in LEGEND_ROWS:
        assert row not in decide
    assert DOC_TEXT not in decide


def test_no_documentation_without_state_trajectories():
    """Table 2's ablation: `videos` alone is the WEAKER judge, on purpose. The
    documentation belongs to the `state_trajectories` channel and must not
    leak into the video-only arm."""
    prompt = _caption_prompt(_ctx(**{"evaluate.artifacts": ["videos"]}))
    for row in LEGEND_ROWS:
        assert row not in prompt
    assert DOC_TEXT not in prompt
    assert "t=" not in prompt
    assert "DOCUMENTATION" not in prompt
