"""`render_stats.json`: a clip's health, measured at write time.

A renderer can silently produce near-black frames: the run finishes
`status: ok`, the journal says `n_frames: 300`, the files are on disk, the
render watchdog never fires -- and the candidates have already been VLM-scored
off frames that show nothing. Two runs of identical config can then differ many
times over in mean brightness, and without these statistics nothing in the
artifact tells a good clip from a black one.

What these tests pin is the DESIGN, not a detector. A fixed detector cannot
work (near-black compressed video still yields ~1,480 distinct colours; a
reacher clip is legitimately darker than a half-cheetah one), so `clip_stats`
records measurements and draws no verdicts -- the within-run comparison
belongs to the reader, for whom `output.video.record: all` makes every
candidate a same-env sibling. The tests therefore check that the numbers are
RIGHT and that the failure's signature (a large brightness gap between
siblings) is visible in the artifact, never that anything was flagged.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bird.artifacts import RunDir
from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.observability import clip_stats, record_rollouts
from bird.types import Candidate, CandidateReport, TrainResult, Trajectory

#: Not in the tester-tier smoke suite (rendering: writes and re-reads frames).
#: Deselected by `-m "not slow"`. See pyproject.toml.
pytestmark = pytest.mark.slow


class StateBrightnessEnv:
    """`render(state)` -> a flat grey frame whose brightness IS the state.

    The contract recording needs (render a STORED state), shaped so that a
    trajectory's stored rows decide how bright its clip is -- which is what
    lets ONE run hold a bright candidate and a dark sibling, the exact shape
    of the failure `clip_stats` exists to make visible. Grey on purpose:
    R = G = B = v makes the Rec. 601 luma exactly v, so every expected value
    below is readable without arithmetic.
    """

    def render(self, state):
        v = float(np.asarray(state, dtype=float).ravel()[0])
        return np.full((4, 4, 3), int(np.clip(v, 0, 255)), dtype=np.uint8)


def _traj(brightness: float, n_steps: int = 6) -> Trajectory:
    return Trajectory(states=[[brightness]] * n_steps,
                      rewards=[0.0] * n_steps, length=n_steps, ret=0.0)


def _report(traj: Trajectory, cand_id: str) -> CandidateReport:
    cand = Candidate(cand_id=cand_id, iteration=0, reward_code="def r(): ...")
    result = TrainResult(cand_id=cand_id, candidate=cand, trained=True,
                         trajectories=[traj])
    return CandidateReport(cand_id=cand_id, candidate=cand, result=result, fitness=1.0)


def _ctx(tmp_path) -> Context:
    cfg = load("eureka", profile="tester", overrides={
        "output.video.enabled": True,
        "output.video.record": "all",
        "output.video.format": "frames",
        "output.video.max_frames": 8,
        "output.video.width": 4,
        "output.video.timeout_s": None,
    })
    return Context(cfg=cfg, budget=Budget(), rundir=RunDir(tmp_path, "vs", "hash0"),
                   env=StateBrightnessEnv())


def _events(ctx) -> list:
    return [json.loads(line) for line in
            (ctx.rundir.path / "journal.jsonl").read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# The measurement itself
# --------------------------------------------------------------------------


def test_clip_stats_measures_what_it_claims():
    black = [np.zeros((4, 6, 3), dtype=np.uint8)] * 5
    grey = [np.full((4, 6, 3), 200, dtype=np.uint8)] * 5

    dark, bright = clip_stats(black), clip_stats(grey)
    assert dark["luma_mean"] == 0.0
    assert bright["luma_mean"] == 200.0, "R=G=B=v must give luma exactly v"
    assert dark["n_frames"] == bright["n_frames"] == 5
    assert (bright["width"], bright["height"]) == (6, 4)
    assert len(bright["luma_per_frame"]) == 5, (
        "the per-frame series is the diagnosis: a clip that goes black halfway "
        "has the same mean as one dim throughout")
    assert dark["distinct_colours"] == 1 and bright["distinct_colours"] == 1

    two_tone = np.zeros((4, 6, 3), dtype=np.uint8)
    two_tone[:2] = 200
    assert clip_stats([two_tone])["distinct_colours"] == 2


def test_clip_stats_is_a_measurement_not_a_verdict():
    """No boolean, no threshold, no 'healthy' field.

    Any fixed verdict lies: a near-black compressed clip still carries ~1,480
    distinct colours, and no absolute brightness floor holds across
    environments. A verdict key added
    here later should trip this and force that argument to be re-had.
    """
    out = clip_stats([np.zeros((2, 2, 3), dtype=np.uint8)])
    verdictish = {k for k in out
                  if isinstance(out[k], bool) or "ok" in k or "healthy" in k
                  or "black" in k or "bad" in k}
    assert not verdictish, f"clip_stats grew verdict-shaped fields: {verdictish}"


# --------------------------------------------------------------------------
# The artifact: beside the clip, and in the journal
# --------------------------------------------------------------------------


def test_stats_land_beside_the_clip_and_on_the_record_event(tmp_path):
    ctx = _ctx(tmp_path)
    written = record_rollouts(ctx, [_report(_traj(200.0), "c0")])
    assert written, "the recording itself must still happen"

    stats = json.loads((ctx.rundir.path / "videos" / "c0" / "render_stats.json").read_text())
    assert stats["cand_id"] == "c0"
    assert stats["format"] == "frames", "off the FILE the encoder produced, not the config"
    assert stats["luma_mean"] == 200.0
    assert stats["n_frames"] == len(list(
        (ctx.rundir.path / "videos" / "c0").glob("frame_*.png")))

    (ev,) = [e for e in _events(ctx) if e.get("stage") == "record"]
    assert ev["luma_mean"] == 200.0
    assert ev["distinct_colours"] == 1
    assert ev["stats"].endswith("render_stats.json"), (
        "`record` is the line a reader greps when footage looks wrong; the "
        "scalars must be on it, not only in a file 300 PNGs away")


def test_a_brightness_gap_between_siblings_is_visible(tmp_path):
    """Two same-config candidates, 40x apart in brightness.

    Without this artifact the two clips are indistinguishable without decoding
    their frames; the run reads healthy and the dark one's candidates have
    already been VLM-scored. The assertion is on the RATIO between
    same-run siblings, because that is the only comparison the design
    endorses -- no absolute threshold generalises across environments.
    """
    ctx = _ctx(tmp_path)
    record_rollouts(ctx, [_report(_traj(200.0), "bright"),
                          _report(_traj(5.0), "dark")])

    read = lambda cid: json.loads(
        (ctx.rundir.path / "videos" / cid / "render_stats.json").read_text())
    bright, dark = read("bright"), read("dark")
    assert bright["luma_mean"] / dark["luma_mean"] == 40.0
    by_cand = {e["cand_id"]: e for e in _events(ctx) if e.get("stage") == "record"}
    assert by_cand["bright"]["luma_mean"] > by_cand["dark"]["luma_mean"]


def test_a_stats_failure_does_not_end_the_run(tmp_path, monkeypatch):
    """Everything else in `record_rollouts` degrades; so must the measurement.

    And the journal must say nothing rather than null: absent-when-never-
    computed is the `subtask_scores` convention, and a null luma on a healthy
    clip would read as a measured zero -- i.e. as a black frame.
    """
    import bird.observability as obs

    monkeypatch.setattr(obs, "clip_stats",
                        lambda frames: (_ for _ in ()).throw(OSError("boom")))
    ctx = _ctx(tmp_path)
    written = record_rollouts(ctx, [_report(_traj(200.0), "c0")])

    assert written, "a lost statistic must not be a lost recording"
    assert (ctx.rundir.path / "videos" / "c0" / "frame_0000.png").exists()
    assert not (ctx.rundir.path / "videos" / "c0" / "render_stats.json").exists()
    (ev,) = [e for e in _events(ctx) if e.get("stage") == "record"]
    assert "luma_mean" not in ev and "stats" not in ev


def test_clip_stats_accepts_every_frame_shape_the_recorder_does():
    """The recording path hands `clip_stats` frames it has already normalised,
    but grayscale, RGBA and float frames are valid render outputs everywhere
    else in the pipeline -- so the measurement coerces with `_as_rgb8`, the
    encoders' own rule, rather than assuming (H, W, 3) uint8.
    """
    from bird.observability import clip_stats as _cs

    grey = np.full((4, 4), 200, dtype=np.uint8)                # (H, W)
    rgba = np.full((4, 4, 4), 200, dtype=np.uint8)             # alpha dropped
    unit = np.ones((4, 4, 3), dtype=np.float64)                # [0, 1] scaled

    out = _cs([grey, rgba, unit])
    assert out["n_frames"] == 3
    assert out["luma_per_frame"] == [200.0, 200.0, 255.0], (
        "R=G=B frames must keep luma exactly v under every accepted shape")
    assert out["distinct_colours"] == 2


def test_distinct_colours_counts_triples_not_lossy_codes():
    """The count is over RGB TRIPLES, and the packing must stay injective.

    `clip_stats` packs each pixel into one uint32 rather than uniquing an
    (N, 3) array. These four colours are
    chosen to collide under the lossier packings that invite themselves --
    summing the channels, or dropping a channel's low bits -- so a count of
    anything but 4 says the code lost information rather than overhead.
    """
    from bird.observability import clip_stats as _cs

    # The first three share a channel sum (1); all four collapse to black if
    # a channel's low bits are dropped.
    colours = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    frame = np.zeros((4, 1, 3), dtype=np.uint8)
    for i, c in enumerate(colours):
        frame[i, 0] = c
    assert _cs([frame])["distinct_colours"] == 4


def test_the_stats_write_is_atomic_and_leaves_no_tmp(tmp_path):
    """`_read_json` reports a torn file as NO STATS, so a half-written
    render_stats.json would read exactly like a clip measured before the file
    existed -- the same conflation records/ writes atomically to avoid.
    """
    ctx = _ctx(tmp_path)
    record_rollouts(ctx, [_report(_traj(200.0), "c0")])

    d = ctx.rundir.path / "videos" / "c0"
    assert not list(d.glob("*.tmp")), "a completed write must leave no tmp sibling"
    assert json.loads((d / "render_stats.json").read_text())["luma_mean"] == 200.0


def test_a_failed_stats_write_leaves_no_tmp_behind(tmp_path, monkeypatch):
    """The stats write is the one atomic writer whose failure is SWALLOWED.

    `save_records` and `_write_status` fail into paths where the run is
    ending; this one degrades and the loop carries on to the next candidate
    and the next iteration, so a failing `os.replace` would strand one
    `.tmp` per attempt under `videos/<cand>/`. Failing at the replace
    specifically, because that is the step that
    leaves a fully-written temp file on disk.
    """
    import bird.observability as obs

    # NARROW on purpose: `obs.os` IS the os module, so a blanket patch also
    # breaks the journal and status writes and the OSError escapes from
    # somewhere this test is not about.
    real_replace = obs.os.replace

    def _boom(src, dst):
        if str(dst).endswith("render_stats.json"):
            raise OSError("replace failed")
        return real_replace(src, dst)

    monkeypatch.setattr(obs.os, "replace", _boom)
    ctx = _ctx(tmp_path)
    written = record_rollouts(ctx, [_report(_traj(200.0), "c0")])

    d = ctx.rundir.path / "videos" / "c0"
    assert written, "a lost statistic must not be a lost recording"
    assert not (d / "render_stats.json").exists()
    assert not list(d.glob("*.tmp")), (
        f"a failed write stranded {[p.name for p in d.glob('*.tmp')]}")
