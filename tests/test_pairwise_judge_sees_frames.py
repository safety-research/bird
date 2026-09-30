"""GT's pairwise judge must be sent PIXELS, not the name of a file.

The sibling of `tests/test_vlm_sees_frames.py`, for the other judge. The failure
both files pin: a prompt that names the frames instead of carrying them
(`f"video: {n} frames @ {fps}fps, file={where}"`) and a client call
`messages=[{"role": "user", "content": prompt}]` with no `images=` on any path.
Then `llm.evaluator.modality: vlm`, `evaluate.artifacts: [videos]` and
`evaluate.vlm.images_per_query: 20` are three keys read only to build a string --
a fabricated pin, declared and never honoured.

Nothing raises in that state and no counter reads wrong: `gt` /
`gt_reward_design`, whose contribution is a VLM comparing agents by watching
them, runs without a single image leaving the process. A
`budget.blind_comparisons` keyed to `clip.video_path` reports on the RECORDER
instead: under `output.video.record: best_and_worst` it fires for eight of ten
candidates while all ten are blind.

Four claims are pinned here and each fails independently:

  1. the frames reach the client, both sides, on both frames-attaching members;
  2. step 2 of `llm_on_vlm_captions` still gets NONE -- that is GT's design (the
     clips do not fit in the decider's context), and a decider that could see
     them would collapse the two-step comparator into the one-step one and make
     the ablation between them measure nothing;
  3. the judge's window does not depend on what the recorder happened to keep,
     nor on the episode behind the clip;
  4. a blind comparison is counted and legible, and a config that never asked
     for pixels is not called blind.

The suite stays OFFLINE: the client is a stub that records what it was handed.
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
from bird.components.preferences import (  # noqa: E402
    _client_call, _pair_frames, run_preferences)
from bird.context import Context  # noqa: E402
from bird.state import RunState  # noqa: E402
from bird.types import Candidate, CandidateReport, TrainResult, Trajectory  # noqa: E402

#: NO `slow` mark, deliberately, and unlike its sibling
#: `tests/test_vlm_sees_frames.py`. The whole file is 9.0 s, it is numpy and the
#: standard library, and the pendulum adapter rasterises its frames in pure
#: numpy (`bird/envs/control.py`) -- so it needs no GL, no `--extra`, and no
#: cluster. The default CI job runs `-m "not slow"`, so slow-marking this to
#: match its neighbours would take every check in it out of CI -- and what it
#: checks is a defect that produces a complete, plausible, wrong number. A
#: refactor that dropped `images=` would then go green in CI, which is
#: precisely what this file exists to prevent.


class _SpyVLM:
    """A client honouring `bird/llm/base.py`, which records what it was sent."""

    modality = "vlm"

    def __init__(self, reply: str = "LEFT") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        text = messages if isinstance(messages, str) else "\n".join(
            str(m.get("content", "")) for m in messages)
        self.calls.append({"prompt": text, "n_images": len(images or ())})
        return [self.reply] * n


class _TextOnlyClient(_SpyVLM):
    """Same contract, `modality: text`. `AnthropicClient._complete` DROPS images
    handed to such a client with a warning, so sending them would be a silent
    waste of a render and a judgment that reads as sighted."""

    modality = "text"


class _NoImagesClient:
    """An older provider: no `images=` keyword at all. Still asked, still
    answers, and the artifact -- not silence -- is what says it went blind."""

    modality = "vlm"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, messages):
        self.calls.append(messages if isinstance(messages, str) else str(messages))
        return "LEFT"


def _ctx(client=None, **overrides):
    registry.load_all()
    base = {"llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "output.tracker": "none", "problem.env_id": "pendulum"}
    base.update(overrides)
    cfg = cfgmod.load("gt", overrides=base, profile="dev")
    ctx = Context(cfg=cfg, budget=Budget(), rundir=None, rng=random.Random(0))
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    ctx.evaluator = _SpyVLM() if client is None else client
    return ctx


def _report(i: int, n: int = 400, states: bool = True) -> CandidateReport:
    """One agent's rollout, held at an angle that differs per `i` so two sides
    cannot render byte-identical frames -- the frame-set id is a digest of the
    pixels, and a shared id would make the per-side assertions tautological."""
    arr = (np.array([[np.cos(0.3 * i), np.sin(0.3 * i), 0.0]] * n, dtype=float)
           if states else None)
    traj = Trajectory(states=arr, actions=None, rewards=[0.1 * i] * n,
                      success=bool(i % 2), length=n, ret=0.1 * i * n)
    cand = Candidate(cand_id=f"c{i}", iteration=0,
                     reward_code="def compute_reward(state, action=None): return 0.0")
    res = TrainResult(cand_id=cand.cand_id, candidate=cand, trajectories=[traj])
    return CandidateReport(cand_id=cand.cand_id, candidate=cand, result=res,
                           fitness=float(i))


# --------------------------------------------------------------------------
# 1 -- the pixels
# --------------------------------------------------------------------------

def test_the_one_step_comparator_is_handed_both_clips_as_images():
    """The headline property. `images=` must carry
    `evaluate.vlm.images_per_query` PNG payloads PER SIDE, and the prompt must
    not name a file -- the judge cannot open one, and under
    `output.video.record: best_and_worst` it does not exist for most candidates.
    """
    ctx = _ctx()
    per_side = ctx.cfg["evaluate.vlm.images_per_query"]
    assert per_side == 20

    registry.get("comparator", "vlm")(ctx, RunState(), _report(1), _report(2))
    call = ctx.evaluator.calls[-1]
    assert call["n_images"] == 2 * per_side, (
        "the pairwise VLM was called with %d images; in the n_images=0 state "
        "a VLM-graded method never sees a frame" % call["n_images"])
    assert "file=" not in call["prompt"] and ".mp4" not in call["prompt"]
    assert ctx.budget.blind_comparisons == 0


def test_the_prompt_says_whose_images_are_whose():
    """Images arrive as one flat, ordered run of blocks ahead of the text
    (`AnthropicClient._attach_images`), so ORDER is the only thing that says
    which agent is which. Without the key, forty unlabelled frames of two agents
    doing nearly the same thing is not a harder question than the pairwise one
    -- it is a different question, whose answer is noise."""
    ctx = _ctx()
    registry.get("comparator", "vlm")(ctx, RunState(), _report(1), _report(2))
    prompt = ctx.evaluator.calls[-1]["prompt"]
    assert "the first 20 are LEFT's clip, the next 20 are RIGHT's" in prompt


def test_the_frames_are_bytes_the_client_could_actually_send():
    """PNG bytes, not paths and not arrays: `AnthropicClient._image_block`
    base64s `bytes` straight into an image block."""
    ctx = _ctx()
    png, prov = _pair_frames(ctx, _report(1))
    assert len(png) == 20 and all(f[:8] == b"\x89PNG\r\n\x1a\n" for f in png)
    assert prov["blind_reason"] == "" and prov["id"]


# --------------------------------------------------------------------------
# 2 -- the two-step comparator: only step 1 may see
# --------------------------------------------------------------------------

def test_the_captioner_sees_and_the_decider_does_not():
    """GT's actual design (§4) and a context-length workaround: the two clips do
    not fit in the decider's context, so a VLM captions them contrastively and a
    separate TEXT model reads only the caption. Give step 2 the frames and this
    component becomes `vlm` under another name."""
    ctx = _ctx(_SpyVLM(reply="LEFT and RIGHT both approached; LEFT was closer."))
    registry.get("comparator", "llm_on_vlm_captions")(
        ctx, RunState(), _report(1), _report(2))

    assert len(ctx.evaluator.calls) == 2
    caption, decide = ctx.evaluator.calls
    assert caption["n_images"] == 40, "step 1 is the VLM; it must see the clips"
    assert decide["n_images"] == 0, (
        "step 2 saw the clips; that is `comparator: vlm` wearing this "
        "component's name, and the ablation between them measures nothing")
    assert "You cannot see the clips" in decide["prompt"]


# --------------------------------------------------------------------------
# 3 -- what the judge sees must not depend on the recorder
# --------------------------------------------------------------------------

def test_frames_do_not_depend_on_which_candidate_was_recorded():
    """Narrowing `output.video.record` to save rendering cost is safe only if no
    fitness reads it. That is only true while this holds -- and here it decides
    who wins a tournament."""
    for mode in ("none", "best", "best_and_worst", "all"):
        ctx = _ctx(**{"output.video.record": mode,
                      "output.wandb.video_record": "none"})
        png, prov = _pair_frames(ctx, _report(1))
        assert len(png) == 20, (
            "output.video.record=%s changed the judge's input (%s)"
            % (mode, prov.get("blind_reason")))


def test_the_frames_come_from_the_clip_window_and_not_the_episode(tmp_path):
    """`_clip_for` cuts GT's 15-second window and carries the SOURCE episode's
    `video_path` with it. Handing that path to `sample_frames` would sample
    across the whole episode, make `evaluate.preferences.clip_length_s` inert,
    and -- worse -- do it for only the two candidates that happen to have a file
    under `record: best_and_worst`, so two sides of a tournament would be judged
    on the episode and eight on the clip with nothing saying so.

    Pinned through the recorded provenance rather than the pixels: the steps are
    EPISODE indices (remapped back out of the clip), so the window is checkable
    without re-deriving a stride that has since moved.
    """
    from bird.observability import encode_png

    # A REAL recording on disk, and that is the whole point: `sample_frames`
    # tries `_frames_from_disk` FIRST, so a nonexistent path proves nothing --
    # it falls through to the renderer and the test passes with the clip window
    # thrown away. Verified by mutation: with `video_path` left on the clip, a
    # missing file left every assertion below green.
    episode = tmp_path / "rollout_frames"
    episode.mkdir()
    for i in range(400):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, 0] = i % 256
        (episode / ("f%04d.png" % i)).write_bytes(encode_png(frame))

    ctx = _ctx()
    rep = _report(1, n=400)
    rep.result.trajectories[0].video_path = str(episode)

    png, prov = _pair_frames(ctx, rep)
    assert prov["source"] == "render", (
        "the on-disk EPISODE (%s) was preferred over the clip's own states, so "
        "evaluate.preferences.clip_length_s no longer selects anything"
        % prov["source"])
    assert len(png) == 20

    steps = prov["steps"]
    assert steps is not None and len(steps) == 20
    assert steps == sorted(steps) and len(set(steps)) == 20
    fps = ctx.cfg["evaluate.preferences.fps"]
    clip_s = ctx.cfg["evaluate.preferences.clip_length_s"]
    # 400 episode steps, a 15s@10fps clip = 150 of them, then 20 images across
    # those. The clip's own indices run 0..149; unremapped, `max` would be 149.
    assert max(steps) > clip_s * fps, (
        "steps look clip-relative (max=%d); a consumer reading them as episode "
        "instants would be wrong by exactly the amount that looks like a "
        "reward diverging late" % max(steps))
    assert max(steps) < 400


# --------------------------------------------------------------------------
# 4 -- going blind must be counted, and only when pixels were wanted
# --------------------------------------------------------------------------

@pytest.mark.parametrize("client,fragment", [
    (_TextOnlyClient(), "modality"),
    (_NoImagesClient(), "modality"),
])
def test_a_client_that_cannot_see_is_named_not_guessed_at(client, fragment):
    """Two silent failures wearing one face: a text-modality client (the
    provider drops the images and warns into a 20-hour log) and one with no
    `images=` keyword at all. Both answer; neither looked."""
    ctx = _ctx(client)
    png, prov = _pair_frames(ctx, _report(1))
    assert png == [] and fragment in prov["blind_reason"]


class _UnreadableSignature:
    """A client whose signature `inspect` cannot read, and which then rejects
    `images=`. Not exotic: `client_sees_images` returns True for an unreadable
    signature on purpose ("let the call itself decide"), and a C-implemented or
    heavily wrapped `__call__` is unreadable. It answers, so the comparison
    completes and looks sighted."""

    modality = "vlm"
    __signature__ = None  # makes inspect.signature raise

    def __init__(self) -> None:
        self.n_images: list[int] = []

    def __call__(self, *args, **kw):
        self.n_images.append(len(kw.get("images") or ()))
        if "images" in kw:
            raise TypeError("this provider takes no images")
        return ["LEFT"]


def test_a_dropped_attachment_is_journalled_and_not_claimed_as_sent():
    """The narrow hole, and the one whose cost is highest: the client ANSWERS,
    just not the call that carried the frames. `_query` writes its record before
    the call (so a run that dies mid-judgment still leaves one), so the record
    states the intent -- and if nothing reported the shortfall, that record
    would claim forty images against a comparison decided on text. Which is,
    exactly, the artifact this whole change exists to stop producing."""
    client = _UnreadableSignature()
    ctx = _ctx(client)
    from bird.components.evaluation import client_sees_images
    assert client_sees_images(client), (
        "precondition: an unreadable signature is treated as sighted, which is "
        "what makes this path reachable at all")

    events: list[tuple] = []
    ctx.event = lambda name, **kw: events.append((name, kw))
    registry.get("comparator", "vlm")(ctx, RunState(), _report(1), _report(2))

    assert client.n_images and client.n_images[0] == 40, "it was offered the frames"
    assert client.n_images[-1] == 0, "and answered the call without them"
    dropped = [kw for name, kw in events if name == "judge_images_dropped"]
    assert dropped and dropped[0] == {"role": "pair_vlm", "intended": 40, "sent": 0}, (
        "the comparison was decided on text and the artifact says it was "
        "decided on forty frames: %r" % (events,))


class _SilentUnreadableSignature(_UnreadableSignature):
    """`_UnreadableSignature`, except that the fallback call answers NOTHING.

    The provider is down (or replied with an empty body): the images-bearing
    call is refused for the shape of its signature and the call that does go
    out returns nothing to parse, so no verdict was reached on any input at all
    and the comparator falls through to `_decide_offline`."""

    def __call__(self, *args, **kw):
        self.n_images.append(len(kw.get("images") or ()))
        if "images" in kw:
            raise TypeError("this provider takes no images")
        return []  # nothing to parse: `_as_text` reads None off this


def test_a_provider_that_answered_nothing_is_not_filed_as_a_dropped_attachment():
    """The sibling hole of the test above, and the same rule read the other way.

    `judge_images_dropped` makes ONE claim -- a verdict was reached, and it was
    reached without the pixels -- and it is the count of blind-but-answered
    comparisons that makes it worth having. A provider that returned nothing
    reached no verdict, so filing it here would put a provider outage and a
    client that cannot take `images=` into one stream under the name of the
    rarer of the two, with different remedies behind each. The outage has its
    own evidence: `_client_call` logs it and `_answer` files a response with no
    text against the query's own `query_id`."""
    client = _SilentUnreadableSignature()
    ctx = _ctx(client)
    events: list[tuple] = []
    ctx.event = lambda name, **kw: events.append((name, kw))
    out = registry.get("comparator", "vlm")(ctx, RunState(), _report(1), _report(2))

    assert client.n_images == [40, 0], (
        "precondition: it was offered the frames, refused them, and was asked "
        "again without them: %r" % (client.n_images,))
    assert out is not None, "the comparator still had to reach a decision offline"
    assert [kw for name, kw in events if name == "judge_images_dropped"] == [], (
        "a provider that answered nothing was filed as a comparison decided on "
        "text, which is a different degradation with a different fix: %r"
        % (events,))


def test_a_call_that_raised_reports_no_frames_as_sent():
    """`_client_call`'s second return value is what the ANSWERING call carried.

    A raising call is not an answering one -- the frames left this process and
    no judgment came back on them, the comparator going offline instead -- so
    counting them here would report a provider outage as a sighted judgment.
    That is the same false artifact as `len(images)`-on-intent one layer down,
    and it is invisible in exactly the same way."""

    class _Boom:
        modality = "vlm"

        def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
            raise RuntimeError("the provider is down")

    text, sent = _client_call(_Boom(), "which agent is better?", [b"png"] * 40)
    assert text is None
    assert sent == 0, (
        "the call raised, so nothing was judged on those frames; a nonzero "
        "count makes the record claim a sighted verdict (sent=%r)" % (sent,))


def test_a_blind_comparison_reaches_the_budget():
    """`budget.blind_comparisons` is the sibling of `blind_judgments`: nonzero
    means the reported ranking is partly text-only, whatever the config says."""
    ctx = _ctx(_TextOnlyClient())
    registry.get("comparator", "vlm")(ctx, RunState(), _report(1), _report(2))
    assert ctx.budget.blind_comparisons == 2, (
        "a method whose whole mechanism is watching rollouts ran blind and the "
        "cost column could not say so")


def test_a_scalars_only_config_is_not_called_blind():
    """The condition is "pixels were wanted and did not arrive", never "there
    are no pixels". A config that lists no videos asked for none, and calling it
    blind is the false alarm a `clip.video_path` condition would raise for
    eight of ten candidates on every `record: best_and_worst` run."""
    ctx = _ctx(**{"evaluate.artifacts": ["scalar_metrics"]})
    registry.get("comparator", "vlm")(ctx, RunState(), _report(1), _report(2))
    assert ctx.evaluator.calls[-1]["n_images"] == 0
    assert ctx.budget.blind_comparisons == 0


def test_a_rollout_with_no_states_is_blind_with_a_specific_reason():
    """`sample_frames` knows WHICH of its sources failed; this must not overwrite
    that with a summary, because the remedies differ."""
    ctx = _ctx()
    png, prov = _pair_frames(ctx, _report(1, states=False))
    assert png == [] and "states" in prov["blind_reason"]


# --------------------------------------------------------------------------
# 5 -- the render happens once, and before any thread exists
# --------------------------------------------------------------------------

def test_frames_are_rendered_once_per_candidate_not_once_per_comparison():
    """On `all_pairs` over ten agents each side is judged nine times and
    `evaluate.vlm.repeats` multiplies that again. On Meta-World a frame is a
    MuJoCo render, so re-cutting per comparison would multiply the render bill
    by the pair count for identical pixels."""
    ctx = _ctx()
    renders = {"n": 0}
    inner = ctx.env.render

    def counting(state):
        renders["n"] += 1
        return inner(state)

    ctx.env.render = counting
    left, right = _report(1), _report(2)
    for _ in range(3):
        registry.get("comparator", "vlm")(ctx, RunState(), left, right)
    assert len(ctx.evaluator.calls) == 3
    assert renders["n"] == 40, (
        "%d renders for 3 comparisons of the same two agents; the frames are "
        "not cached on the report" % renders["n"])


def test_the_phase_renders_every_candidate_before_it_opens_a_pool():
    """`sample_frames` drives the shared env renderer and two threads on one
    MuJoCo adapter segfault 3/3 (measured). So every
    render must be finished before `judge_concurrency` can open a pool -- and
    the frames must be on record before the first comparison is asked, so a run
    killed mid-round still says what its judge was looking at."""
    ctx = _ctx()
    events: list[tuple] = []
    ctx.event = lambda name, **kw: events.append((name, kw))
    reports = [_report(i) for i in range(1, 5)]

    order: list[str] = []
    inner = ctx.env.render
    ctx.env.render = lambda s: (order.append("render"), inner(s))[1]
    spy = ctx.evaluator
    call = spy.__call__

    def watched(*a, **kw):
        order.append("judge")
        return call(*a, **kw)

    spy.__call__ = watched  # type: ignore[method-assign]
    run_preferences(ctx, RunState(), reports)

    assert "judge" in order and "render" in order
    assert order.index("judge") > max(i for i, k in enumerate(order) if k == "render"), (
        "a render happened after the first judge call; under "
        "`judge_concurrency > 1` that render is inside a thread pool")

    seen = [kw for name, kw in events if name == "pair_frames"]
    assert seen and seen[0]["images_per_candidate"] == [20, 20, 20, 20]
    assert seen[0]["n_blind"] == 0


def test_the_frames_never_ride_out_on_the_report():
    """Twenty PNGs per candidate is megabytes, `CandidateReport.meta` is carried
    into `RunState` and serialised, and `checkpoint.encode()` raises on anything
    it has no tag for -- so a frame list left behind turns a resume at hour 18
    into a crash rather than into a large file."""
    ctx = _ctx()
    reports = [_report(i) for i in range(1, 4)]
    run_preferences(ctx, RunState(), reports)
    for rep in reports:
        leftover = [k for k in rep.meta if k.startswith("_pref_")]
        assert not leftover, leftover


def test_the_frames_are_dropped_even_when_the_round_raises():
    """The exit that matters is the one nothing would report.

    `BudgetExceeded` out of `_query` is how a round ends when the LLM budget
    runs out mid-comparison -- normal operation, not a crash -- and it skips
    any drop placed only at the bottom of `run_preferences`. `checkpoint.encode`
    has a `bytes` tag, so leaked PNGs would be base64ed into the checkpoint
    JSON rather than raising: a silently fat artifact whose digest
    covers megabytes of pixels no resume reads.
    """
    from bird.budget import BudgetExceeded

    ctx = _ctx()
    ctx.budget.max_llm_calls = 1  # the second comparison cannot be paid for
    reports = [_report(i) for i in range(1, 5)]
    with pytest.raises(BudgetExceeded):
        run_preferences(ctx, RunState(), reports)
    for rep in reports:
        assert not [k for k in rep.meta if k.startswith("_pref_")], (
            "%s carried its cache out through the raise" % rep.cand_id)


# --------------------------------------------------------------------------
# 6 -- the human comparator is not an API call
# --------------------------------------------------------------------------

def test_the_human_comparator_still_gets_the_file_path():
    """Its judge is a person at a terminal, and the path is the one channel
    through which they can actually watch the rollout. Rendering base64 frames
    for an API call nobody is making would be pure cost.

    `human_queries == 0`, not 1: with `ctx.human = None` nobody is asked and
    the verdict falls to `_decide_offline`. A call-site charge would
    double-count every answered comparison; the oracle is the one place that
    charges, and an unanswered comparison is not a human query --
    `test_a_human_comparison_is_charged_once` and
    `test_a_comparison_no_person_answered_is_not_a_human_query` pin both
    halves.
    """
    ctx = _ctx()
    ctx.human = None
    left = _report(1)
    left.result.trajectories[0].video_path = "runs/x/videos/c1/rollout.mp4"
    registry.get("comparator", "human")(ctx, RunState(), left, _report(2))
    assert ctx.budget.human_queries == 0
    assert ctx.evaluator.calls == [], "the human comparator called the evaluator"
    assert not getattr(registry.get("comparator", "human"), "attaches_frames", False)
