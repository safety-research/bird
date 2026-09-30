"""RDA's VLM judge must be sent PIXELS, and must not be told the answer.

Three separate failures can live in one function
(`evaluation._vlm_trajectory_analysis`) and each of them produces a complete,
plausible, wrong number:

  1. **No images.** A prompt containing the sentence
     `f"{n} frames sampled from {traj.video_path}"` -- a FILENAME -- and a
     client helper that never passes `images=` send every judge call with
     `n_images = 0`. The judge's `modality: vlm` and
     `evaluate.vlm.images_per_query: 20` are then read only to interpolate a
     string, and RDA's central mechanism -- §4.2 "Visual Analysis", App. §7.3's
     "first describe the behavior from images" -- runs text-only. Its whole
     delta from Eureka is not running.
  2. **The ground-truth success flag in the prompt.** `success=<bool>` is
     the environment's own success check, the one `verify.forbidden_symbols` keeps
     out of the reward code on the Meta-World configs. Handing it to the judge
     returns it through the other door.
  3. **The candidate's own TOTAL RETURN in the prompt.** `return=<float>`
     is computed by the reward being graded, so scores track it and a candidate
     wins by scaling its reward up. The withholding covers the total return and
     the env success flag ONLY -- per-step reward COMPONENT values and the
     reward function code are deliberately IN the prompt, because App. 7.3
     hands the judge both ('Support your evaluation with both image and reward
     evidence') and RDA's mis-specification diagnoses are impossible without
     them.

The judge is `ctx.generator` (the Agent VLM) and scores ALL subtasks in one
structured call per trajectory; the three failure modes above apply to that
call unchanged.

None of the three raises, none changes the shape of the artifact, and all three
survive every other test in this suite, which is exactly why they get one of
their own. The suite stays OFFLINE: the client here is a stub that records what
it was handed.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bird import config as cfgmod  # noqa: E402
from bird import registry  # noqa: E402
from bird.budget import Budget  # noqa: E402
from bird.components.evaluation import (  # noqa: E402
    _vlm_frames, _vlm_trajectory_analysis, client_sees_images, judge_digest)
from bird.context import Context  # noqa: E402
from bird.types import Trajectory  # noqa: E402

#: Not in the tester-tier smoke suite (rendering: builds the frames a VLM judge replays).
#: Deselected by `-m "not slow"`. See pyproject.toml.
pytestmark = pytest.mark.slow


class _SpyVLM:
    """A client honouring `bird/llm/base.py`, which records what it was sent."""

    modality = "vlm"

    def __init__(self, reply: str = "Score: 0.4\nReason: partial.") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        text = messages if isinstance(messages, str) else "\n".join(
            str(m.get("content", "")) for m in messages)
        self.calls.append({"prompt": text, "n_images": len(images or ())})
        return [self.reply] * n


class _TextOnlyClient(_SpyVLM):
    """Same contract, but declares `modality: text` -- the provider would drop
    any image handed to it (`AnthropicClient._complete` warns and continues)."""

    modality = "text"


def _ctx(**overrides):
    """The scoring client is `ctx.generator` -- RDA's Agent VLM does the in-loop
    judging -- and `ctx.evaluator` here is a spy that must receive NOTHING: an
    in-loop call landing on it is a misrouting."""
    registry.load_all()
    base = {"llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "output.tracker": "none", "problem.env_id": "pendulum"}
    base.update(overrides)
    cfg = cfgmod.load("rda", overrides=base, profile="dev")
    ctx = Context(cfg=cfg, budget=Budget(), rundir=None, rng=random.Random(0))
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    ctx.generator = _SpyVLM()
    ctx.evaluator = _SpyVLM()
    return ctx


_SUBTASKS = ["swing the rod toward upright", "balance the rod upright"]
_REWARD_CODE = "def compute_reward(state, action=None):\n    return 0.0\n"


def _traj(theta: float = 0.0, n: int = 40, success: bool = True,
          ret: float = 12345.0) -> Trajectory:
    """A Pendulum rollout held at one angle. `states` is what `render` replays."""
    states = np.array([[np.cos(theta), np.sin(theta), 0.0]] * n, dtype=float)
    return Trajectory(states=states, actions=None, rewards=[0.0] * n,
                      success=success, length=n, ret=ret)


# --------------------------------------------------------------------------
# 1 -- the pixels
# --------------------------------------------------------------------------

def test_the_judge_is_handed_real_frames_and_not_a_filename():
    """The headline regression. `images=` must carry
    `evaluate.vlm.images_per_query` PNG payloads, and the prompt must not name a
    path -- there is no path: `record_rollouts` runs AFTER stage 4.

    One call per trajectory covering the whole subtask list, judged by
    `ctx.generator`, with the reward code in the prompt (App. 7.3's evidence)."""
    ctx = _ctx()
    traj = _traj()
    assert traj.video_path is None, "stage 4 runs before anything is recorded"

    frames, note = _vlm_frames(ctx, traj, client=ctx.generator)
    assert note == "", note
    assert len(frames) == ctx.cfg["evaluate.vlm.images_per_query"] == 20
    assert all(f[:8] == b"\x89PNG\r\n\x1a\n" for f in frames), "not PNG bytes"

    _vlm_trajectory_analysis(ctx, traj, _SUBTASKS, _REWARD_CODE, True, frames=frames)
    call = ctx.generator.calls[-1]
    assert call["n_images"] == 20, "the VLM was called with no images"
    assert "video_path" not in call["prompt"] and ".mp4" not in call["prompt"]
    # The paper's reward evidence is IN the prompt -- the candidate's own code,
    # and every subtask, so the one call really covers the list.
    assert _REWARD_CODE.strip() in call["prompt"]
    for sub in _SUBTASKS:
        assert sub in call["prompt"]
    # Nothing in-loop reaches the evaluator (the post-hoc role).
    assert ctx.evaluator.calls == [], (
        "an in-loop scoring call reached ctx.evaluator; GPT-4.1 serves only "
        "the post-hoc alignment metric (§10.1)")


def test_frames_do_not_depend_on_which_candidate_was_recorded():
    """`observability.sample_frames` renders from `traj.states`, never from what
    `record_rollouts` happened to write. Narrowing `output.video.record` to
    `best_and_worst` to save rendering cost is safe only because no fitness
    reads it -- which is only true while this holds."""
    for mode in ("none", "best", "best_and_worst", "all"):
        # `output.wandb.video_record` is coherence-checked against `record`, so
        # it moves with it; the point of the test is the JUDGE'S input.
        ctx = _ctx(**{"output.video.record": mode,
                      "output.wandb.video_record": "none"})
        frames, note = _vlm_frames(ctx, _traj(), client=ctx.generator)
        assert len(frames) == 20, f"output.video.record={mode} changed the judge's input ({note})"


def test_a_blind_judgment_is_counted_and_never_silent():
    """Three ways to end up with no pixels, and each must be legible in the
    artifact rather than only in a log line -- a text-only score and a
    frame-grounded one are the same float."""
    # (a) the config does not declare rollout footage as evidence
    ctx = _ctx(**{"evaluate.artifacts": ["scalar_metrics"]})
    ctx.evaluator = _SpyVLM()
    assert _vlm_frames(ctx, _traj()) == ([], "evaluate.artifacts does not list videos")

    # (b) a client the provider would strip the images from anyway
    ctx = _ctx()
    ctx.evaluator = _TextOnlyClient()
    frames, note = _vlm_frames(ctx, _traj())
    assert frames == [] and "modality" in note

    # (c) a rollout with no states to render
    ctx = _ctx()
    ctx.evaluator = _SpyVLM()
    empty = Trajectory(states=None, rewards=[], success=False, length=0)
    frames, note = _vlm_frames(ctx, empty)
    assert frames == [] and "no frames" in note


def test_every_scored_candidate_journals_its_frame_count():
    """Recorded on the GOOD path too. "No warning in the log" is not evidence
    that the judge had eyes, so the frame count is journalled on the good path
    as well."""
    from bird.state import RunState
    from bird.types import Candidate, TrainResult

    ctx = _ctx()
    ctx.evaluator = _SpyVLM()
    events = []
    ctx.event = lambda name, **kw: events.append((name, kw))
    cand = Candidate(cand_id="c0", iteration=0,
                     reward_code="def compute_reward(state, action=None): return 0.0")
    res = TrainResult(cand_id="c0", candidate=cand, trajectories=[_traj(), _traj()])
    registry.get("fitness_source", "vlm_score")(ctx, RunState(), [res])
    seen = [kw for name, kw in events if name == "vlm_frames"]
    assert seen and seen[0]["images_per_rollout"] == [20, 20]
    assert seen[0]["n_blind_rollouts"] == 0
    assert ctx.budget.blind_judgments == 0


def test_the_blind_count_reaches_the_budget():
    """`budget.blind_judgments` is the sibling of `blind_comparisons`: nonzero
    means the reported ranking is partly text-only, whatever the config says."""
    ctx = _ctx(**{"evaluate.artifacts": ["scalar_metrics"]})
    ctx.evaluator = _SpyVLM()
    score = registry.get("fitness_source", "vlm_score")
    from bird.state import RunState
    from bird.types import Candidate, TrainResult
    cand = Candidate(cand_id="c0", iteration=0,
                     reward_code="def compute_reward(state, action=None): return 0.0")
    res = TrainResult(cand_id="c0", candidate=cand, trajectories=[_traj(), _traj()])
    score(ctx, RunState(), [res])
    assert ctx.budget.blind_judgments > 0, (
        "a VLM-graded method ran blind and the cost column could not say so")


def test_frames_are_sampled_once_per_rollout_not_once_per_subtask():
    """A Meta-World frame is a MuJoCo render. `fitness_vlm_score` samples per
    trajectory and reuses across subtasks and `evaluate.vlm.repeats`; doing it
    per query would multiply the render bill by `len(subtasks) * repeats` for
    byte-identical pixels -- and, on Anthropic, would also destroy the image
    prefix that `llm.prompt_caching` reuses across those queries."""
    src = Path("bird/components/evaluation.py").read_text()
    body = src[src.index('@register("fitness_source", "vlm_score")'):]
    body = body[:body.index('@register("fitness_source", "preference_bt")')]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    # Pinned as a PREFIX, because the call also passes `_vlm_frames`'
    # provenance out-parameter, plus a count -- which is the stronger form of the
    # same guarantee the whole-line pin was reaching for: one call per
    # trajectory means exactly one call site in this body, and a second one
    # anywhere inside the subtask or repeat loops would fail this whatever it
    # was spelled.
    assert "frames_by_traj = [_vlm_frames(ctx, t" in code
    assert code.count("_vlm_frames(") == 1, (
        "the frames are sampled somewhere other than once per trajectory")
    assert code.index("frames_by_traj =") < code.index("for repeat in range(repeats)")


# --------------------------------------------------------------------------
# 2 and 3 -- what the judge must NOT be told
# --------------------------------------------------------------------------

def test_the_judge_is_never_told_the_ground_truth_success_flag():
    """The withholding covers the success flag and the total return, and this
    is one of the two."""
    ctx = _ctx()
    traj = _traj(success=True)
    _vlm_trajectory_analysis(ctx, traj, _SUBTASKS, _REWARD_CODE, True,
                             frames=_vlm_frames(ctx, traj, client=ctx.generator)[0])
    prompt = ctx.generator.calls[-1]["prompt"]
    assert "success=" not in prompt, (
        "the environment's own success check is in the judge's prompt; "
        "verify.forbidden_symbols keeps it out of the reward code for the same "
        "reason it must stay out of here")


def test_the_judge_is_never_told_the_candidates_own_return():
    """The other half of the withholding: per-step COMPONENT values are sent on
    purpose (App. 7.3), the TOTAL return and per-step mean never are."""
    ctx = _ctx()
    traj = _traj(ret=12345.0)
    _vlm_trajectory_analysis(ctx, traj, _SUBTASKS, _REWARD_CODE, True,
                             frames=_vlm_frames(ctx, traj, client=ctx.generator)[0])
    prompt = ctx.generator.calls[-1]["prompt"]
    for leak in ("return=", "mean_per_step=", "12345"):
        assert leak not in prompt, (
            f"{leak!r} is the reward being graded; a judge shown it scores "
            "monotonically in it and a candidate wins by scaling up")


def test_the_component_table_is_sent_and_never_the_reference_reward():
    """The positive half, and its guard. The per-step table carries the
    candidate's own component values at the sampled frames' episode steps --
    the paper's reward evidence -- and `_REFERENCE_KEYS` stays filtered: the
    hidden reference reward leaking to the judge would be a §4 breach."""
    ctx = _ctx()
    n = 40
    traj = _traj(n=n)
    traj.component_values = {"reward_dist": [0.1] * n,
                             "gt_reward": [9.9] * n}
    prov: dict = {}
    frames, _ = _vlm_frames(ctx, traj, prov, client=ctx.generator)
    _vlm_trajectory_analysis(ctx, traj, _SUBTASKS, _REWARD_CODE, True,
                             frames=frames, prov=prov)
    prompt = ctx.generator.calls[-1]["prompt"]
    assert "reward_dist=" in prompt, "the component table never reached the judge"
    assert "gt_reward" not in prompt and "9.9" not in prompt, (
        "_REFERENCE_KEYS leaked into the judge's component table")
    # Recorded alignment: the table's rows are the frames' own episode steps.
    for step in prov["steps"]:
        assert f"step={step}" in prompt


def test_judge_digest_carries_length_and_nothing_else():
    """Pinned as a whole string: a future field added to `_traj_digest` must not
    silently become something the judge is told."""
    assert judge_digest(_traj(success=True, ret=999.0)) == "length=40 steps"


def test_client_sees_images_refuses_a_text_modality_client():
    assert client_sees_images(_SpyVLM()) is True
    assert client_sees_images(_TextOnlyClient()) is False
    assert client_sees_images(None) is False


# --------------------------------------------------------------------------
# the reported metric takes the same two rules
# --------------------------------------------------------------------------

def test_alignment_rate_sends_frames_and_no_ground_truth():
    src = Path("bird/components/phases.py").read_text()
    body = src[src.index('@register("phase", "alignment_rate")'):]
    body = body[:body.index('@register("phase", "real_world_eval")')]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    # A prefix, for the reason above: the call also passes `prov=prov` so the
    # rating records which pixels it was made from. What the pin is for: the
    # frames come from `sample_frames` over the stored states at the configured
    # count, not from a filename in a sentence.
    assert "sample_frames(ctx, traj, n_images" in code
    assert "images=images" in code
    assert "_traj_digest" not in code, "App. §10.2 makes the success flag the OTHER metric"
    assert "judge_digest(traj)" in code
