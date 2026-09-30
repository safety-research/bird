"""`output.trajectory_trace`: what the reward claimed, beside what it paid.

`RunDir.save_train_result` drops `TrainResult.trajectories` on the stated
grounds that they are large and "already summarised by the curves". That is
true of every question except one. A reward that inflates its own component
values agrees with itself perfectly in `component_traces`, because that summary
is a mean over the very numbers being inflated -- the disagreement exists only
per step, between what the reward CLAIMED it was paying and what the harness
MEASURED it paid. This artifact is that pair, and nothing else in the run
directory carries it.

The test that matters here is the alignment one. Frames are strided across an
episode by `_even_indices`, never truncated, so frame 150 of 300 is step 250 of
a 500-step episode. A trace sampled at 0..N instead would put every claimed
value one stride away from the measurement it is supposed to be compared
against -- and the artefact of that mistake looks exactly like the finding the
artifact exists to produce: two curves agreeing early and diverging late. A
wrong frame index and a real divergence are indistinguishable in the data, so
the mapping is pinned here rather than trusted.
"""

from __future__ import annotations

import json

import numpy as np

from bird.artifacts import RunDir
from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.observability import _even_indices, record_rollouts
from bird.types import Candidate, CandidateReport, Trajectory, TrainResult

import pytest
#: Not in the tester-tier smoke suite (rendering: strided per-step traces alongside a recorded episode).
#: Deselected by `-m "not slow"`. See pyproject.toml.
pytestmark = pytest.mark.slow


class RenderingEnv:
    """An env that renders a stored state, which is the contract recording needs."""

    def render(self, state):
        return np.zeros((4, 4, 3), dtype=np.uint8)


def _traj(n_steps: int, *, components=None, rewards=None) -> Trajectory:
    """One episode whose per-step numbers are a function of the step index.

    Every series is `f(i)` for a distinct `f`, so a value read back out of the
    artifact names the step it came from. That is what lets the alignment test
    assert the mapping rather than merely assert that some numbers arrived.
    """
    return Trajectory(
        states=[[float(i)] for i in range(n_steps)],
        rewards=rewards if rewards is not None else [float(i) * 10.0 for i in range(n_steps)],
        component_values=(components if components is not None
                          else {"proximity": [float(i) * 100.0 for i in range(n_steps)]}),
        length=n_steps,
        ret=float(sum(range(n_steps))),
        success=False,
    )


def _report(traj: Trajectory, cand_id: str = "c0") -> CandidateReport:
    cand = Candidate(cand_id=cand_id, iteration=0, reward_code="def compute_reward(): ...")
    result = TrainResult(cand_id=cand_id, candidate=cand, trained=True, trajectories=[traj])
    return CandidateReport(cand_id=cand_id, candidate=cand, result=result, fitness=1.0)


def _ctx(tmp_path, *, record="components", max_frames=300, max_steps=300) -> Context:
    cfg = load("eureka", profile="tester", overrides={
        "output.video.enabled": True,
        "output.video.record": "all",
        "output.video.format": "frames",
        "output.video.max_frames": max_frames,
        "output.video.width": 4,
        "output.video.timeout_s": None,
        "output.trajectory_trace.record": record,
        "output.trajectory_trace.max_steps": max_steps,
    })
    return Context(cfg=cfg, budget=Budget(), rundir=RunDir(tmp_path, "trace", "hash0"),
                   env=RenderingEnv())


def _trace(ctx, cand_id: str = "c0") -> dict:
    return json.loads((ctx.rundir.path / "traces" / f"{cand_id}.json").read_text())


# --------------------------------------------------------------------------
# The default
# --------------------------------------------------------------------------


def test_the_default_writes_nothing_at_all(tmp_path):
    """Every published config inherits `none`, so no run's artifacts change.

    The key exists to be turned on for a run someone intends to read. If the
    default wrote anything, every large sweep would start paying for an
    artifact nobody asked for -- and per-step traces are large.
    """
    ctx = _ctx(tmp_path, record="none")
    written = record_rollouts(ctx, [_report(_traj(10))])

    assert written, "the video itself should still be recorded"
    assert not (ctx.rundir.path / "traces").exists(), (
        "`none` wrote a traces/ directory; the default must cost nothing")


def test_the_default_in_the_shipped_config_is_none():
    """Read off the config rather than asserted about it."""
    assert load("eureka", profile="tester").get("output.trajectory_trace.record") == "none"


# --------------------------------------------------------------------------
# Alignment -- the reason this file exists
# --------------------------------------------------------------------------


def test_each_sample_carries_the_episode_step_its_frame_was_rendered_from(tmp_path):
    """The stride, pinned end to end.

    500 states into 10 frames is a stride of ~55. A trace sampled 0..9 would
    hold the first ten steps of the episode while the frames show ten moments
    spread across all of it, so each sample's frame index would be wrong by up
    to 490 steps -- and wrong *increasingly*, which reads as a reward diverging late.
    """
    n, frames = 500, 10
    ctx = _ctx(tmp_path, max_frames=frames, max_steps=frames)
    record_rollouts(ctx, [_report(_traj(n))])
    tr = _trace(ctx)

    expected = _even_indices(n, frames)
    assert tr["samples"]["step"] == expected, (
        "the trace did not sample where the frames were rendered from")
    assert tr["samples"]["step"] != list(range(len(expected))), (
        "sampling collapsed to 0..N -- the stride was lost, which is the whole bug")

    # And the values prove it independently of the index list: the fixture makes
    # every series a known function of the step, so a mis-strided read shows up
    # as numbers, not just as indices.
    assert tr["measured"]["reward"] == [i * 10.0 for i in expected]
    assert tr["claimed"]["proximity"] == [i * 100.0 for i in expected]


def test_every_sample_has_a_frame_behind_it(tmp_path):
    """`frame` indexes the frames on disk, so every sample maps to a frame on disk."""
    ctx = _ctx(tmp_path, max_frames=8, max_steps=8)
    record_rollouts(ctx, [_report(_traj(100))])
    tr = _trace(ctx)

    n_frames = tr["n_frames"]
    assert tr["samples"]["frame"] == list(range(n_frames)), (
        "samples must map onto the rendered frames one-for-one at this cap")
    assert len(tr["samples"]["step"]) == len(tr["samples"]["frame"]), (
        "a step with no frame index has no frame to align to")
    assert all(0 <= f < n_frames for f in tr["samples"]["frame"])


def test_max_steps_below_the_frame_count_thins_evenly_rather_than_clipping(tmp_path):
    """A thinner trace is still spread over the whole episode.

    Clipping the tail would describe a different episode than the one recorded,
    and would do it silently -- the same argument `_even_indices` makes for
    frames, applied one level down.
    """
    ctx = _ctx(tmp_path, max_frames=20, max_steps=5)
    record_rollouts(ctx, [_report(_traj(200))])
    tr = _trace(ctx)

    steps = tr["samples"]["step"]
    assert len(steps) <= 5
    frame_steps = _even_indices(200, 20)
    assert steps == [frame_steps[k] for k in _even_indices(len(frame_steps), 5)]
    assert steps[-1] >= frame_steps[-1] - 1, (
        "the last sample is nowhere near the end of the episode -- the tail was clipped")


# --------------------------------------------------------------------------
# Not fabricating numbers
# --------------------------------------------------------------------------


def test_a_series_shorter_than_the_episode_stays_absent_at_those_steps(tmp_path):
    """A component that stopped being emitted reads as null, never as a value.

    Padding it -- with zero, or with its own last value -- would draw a claimed
    line that agrees with the measurement exactly where there was no claim at
    all. Agreement is the finding this artifact reports, so a fabricated one is
    worse than a gap.
    """
    n = 40
    traj = _traj(n, components={"stops_early": [1.0] * 10})
    ctx = _ctx(tmp_path, max_frames=8, max_steps=8)
    record_rollouts(ctx, [_report(traj)])
    tr = _trace(ctx)

    steps = tr["samples"]["step"]
    got = tr["claimed"]["stops_early"]
    for step, value in zip(steps, got):
        if step < 10:
            assert value == 1.0
        else:
            assert value is None, f"step {step} had no claim and was given {value!r}"


def test_states_are_written_only_when_asked_for(tmp_path):
    """`components` is the disagreement check; the raw observation is opt-in.

    A 39-dimensional Meta-World state is 39x the cost of the numbers that
    answer the question, so the two are separate values of one key rather than
    one value that always pays for both.
    """
    lean = _ctx(tmp_path / "lean", record="components", max_frames=4, max_steps=4)
    record_rollouts(lean, [_report(_traj(20))])
    assert "states" not in _trace(lean)

    full = _ctx(tmp_path / "full", record="components+states", max_frames=4, max_steps=4)
    record_rollouts(full, [_report(_traj(20))])
    tr = _trace(full)
    assert "states" in tr
    assert len(tr["states"]) == len(tr["samples"]["step"])
    # The fixture's state at step i is [i], so the states prove their own stride.
    assert [row[0] for row in tr["states"]] == [float(s) for s in tr["samples"]["step"]]


# --------------------------------------------------------------------------
# It describes the episode the video shows
# --------------------------------------------------------------------------


def test_states_stay_index_aligned_with_the_steps_they_describe(tmp_path):
    """`states[j]` must describe `samples.step[j]`, one for one.

    Filtering out-of-range indices would shift every entry after the gap: a
    reader would then draw one step's observation against
    another step's frame with nothing in the payload saying so. That is the
    mirror image of the padding `_sample_series` refuses, and does the same
    damage in the other direction, so the two lengths are pinned rather than
    assumed.
    """
    ctx = _ctx(tmp_path, record="components+states", max_frames=6, max_steps=6)
    record_rollouts(ctx, [_report(_traj(60))])
    tr = _trace(ctx)

    assert len(tr["states"]) == len(tr["samples"]["step"]), (
        "states and steps must stay one-for-one, or the alignment lies")
    # The fixture's state at step i is [i], so alignment is checkable by value
    # and not merely by length.
    for step, row in zip(tr["samples"]["step"], tr["states"]):
        assert row is not None and row[0] == float(step)


def test_a_trace_that_cannot_be_written_does_not_end_the_run(tmp_path):
    """Everything else in `record_rollouts` degrades; so must this.

    The write may land on a shared network mount that can hang, for an
    artifact that is off by default, and it happens AFTER the training is paid
    for. An OSError here must not propagate into `bird.py`'s loop and end a
    search that has already bought its result.
    """
    import bird.observability as obs

    ctx = _ctx(tmp_path, max_frames=4, max_steps=4)
    boom = lambda *a, **k: (_ for _ in ()).throw(OSError("shared mount is not answering"))
    original, obs.save_trajectory_trace = obs.save_trajectory_trace, boom
    try:
        written = record_rollouts(ctx, [_report(_traj(20))])
    finally:
        obs.save_trajectory_trace = original

    assert written, "the video was still recorded and the search continued"
    assert not (ctx.rundir.path / "traces").exists()


def test_the_trace_describes_the_same_episode_the_frames_do(tmp_path):
    """Rollout 0, the one `record_rollouts` renders.

    Tracing a different rollout than the one recorded would pair a sample with
    footage of another episode -- the same class of error as the stride, and
    less visible, because both halves would be internally consistent.
    """
    shown = _traj(30, components={"c": [1.0] * 30})
    other = _traj(30, components={"c": [99.0] * 30})
    report = _report(shown)
    report.result.trajectories.append(other)

    ctx = _ctx(tmp_path, max_frames=6, max_steps=6)
    record_rollouts(ctx, [report])
    tr = _trace(ctx)

    assert set(tr["claimed"]["c"]) == {1.0}, "the trace came from a rollout nobody watched"


def test_the_journal_names_the_trace_it_wrote(tmp_path):
    """Discoverable from the event stream, like the video beside it.

    A trace nothing points at is one a reader has to guess the path of, and a
    run where it was skipped looks identical to one where it was never asked
    for.
    """
    ctx = _ctx(tmp_path, max_frames=4, max_steps=4)
    record_rollouts(ctx, [_report(_traj(20))])

    events = [json.loads(l) for l in
              (ctx.rundir.path / "journal.jsonl").read_text().splitlines() if l.strip()]
    rec = [e for e in events if e.get("stage") == "record"]
    assert rec and rec[0]["trace"].endswith("traces/c0.json"), rec


def test_an_env_that_cannot_render_writes_no_trace(tmp_path):
    """No frames means no frame index, so a trace would have nothing to align to.

    The three toy envs take this path, which is why the whole suite does.
    """
    ctx = _ctx(tmp_path, max_frames=4, max_steps=4)
    ctx.env = object()  # no render()
    assert record_rollouts(ctx, [_report(_traj(20))]) == []
    assert not (ctx.rundir.path / "traces").exists()
