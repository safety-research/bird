"""`output.video.timeout_s` in the JUDGE path: a deadline is not a dead renderer.

`sample_frames` renders the frames a VLM judge is shown (`evaluate` runs before
`record_rollouts`, so the judge never finds a recording to reuse) under the same
`_RenderClock` `record_rollouts` uses. Mapping every `None` from
`_render_frames` -- a spent clock, a `render()` that takes no state, an
unrenderable frame alike -- to `blind_reason: "the renderer is unusable for
every trajectory"` with no journal event, while `_record_rollouts` branches on
`clock.expired` and journals `record_timeout` for the identical clock, would
file every deadline-blinded judgment in `judgments/*.jsonl` as a broken adapter
on a slow node, with nothing in the journal saying the guard fired during
scoring. The fitness is blind either way; the CAUSE is what these tests pin,
because the remedies differ.

tests/test_render_timeout.py covers the recorder's half; this is the judge's.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from bird.artifacts import RunDir
from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.observability import sample_frames
from bird.types import Trajectory

FRAME_S = 0.02


class SlowEnv:
    """Renders -- always -- just too slowly to finish before the deadline."""

    def __init__(self, per_frame_s: float = FRAME_S):
        self.per_frame_s = per_frame_s
        self.calls = 0

    def render(self, state):
        self.calls += 1
        time.sleep(self.per_frame_s)
        return np.zeros((8, 8, 3), dtype=np.uint8)


class NoStateEnv:
    """A renderer that cannot replay a stored state: genuinely unusable here."""

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)


def _ctx(tmp_path, timeout_s, env) -> Context:
    cfg = load("eureka", profile="tester", overrides={
        "output.video.enabled": True, "output.video.width": 8,
        "output.video.timeout_s": timeout_s})
    return Context(cfg=cfg, budget=Budget(), rundir=RunDir(tmp_path, "vid", "hash0"), env=env)


def _traj(n: int = 40) -> Trajectory:
    return Trajectory(states=[[float(i)] for i in range(n)], rewards=[0.0] * n, length=n)


def _events(ctx, stage: str) -> list:
    path = ctx.rundir.path / "journal.jsonl"
    if not path.exists():
        return []
    return [e for e in (json.loads(l) for l in path.read_text().splitlines())
            if e.get("stage") == stage]


def test_a_deadline_hit_is_filed_as_a_timeout_and_journalled(tmp_path):
    ctx = _ctx(tmp_path, timeout_s=FRAME_S, env=SlowEnv())
    prov: dict = {}
    frames = sample_frames(ctx, _traj(), 8, prov=prov)
    assert frames == [] and prov.get("n_frames", 0) == 0, "blind, as before"
    reason = prov.get("blind_reason", "")
    assert "output.video.timeout_s" in reason and "elapsed after" in reason, reason
    assert "unusable for every trajectory" not in reason, (
        "a slow renderer is not a broken one; the record must not say it is")
    events = _events(ctx, "sample_timeout")
    assert len(events) == 1, "one event per blinded sampling call"
    ev = events[0]
    assert ev["timeout_s"] == pytest.approx(FRAME_S)
    assert ev["elapsed_s"] >= FRAME_S
    assert ev["n_frames"] >= 1, "how far the renderer got before the deadline"
    assert ev["n_requested"] == 8 and ev["traj_length"] == 40


def test_a_genuinely_unusable_renderer_is_still_called_unusable(tmp_path):
    ctx = _ctx(tmp_path, timeout_s=60.0, env=NoStateEnv())
    prov: dict = {}
    assert sample_frames(ctx, _traj(), 8, prov=prov) == []
    assert "unusable" in prov.get("blind_reason", "")
    assert "timeout" not in prov.get("blind_reason", "")
    assert _events(ctx, "sample_timeout") == []


def test_no_deadline_samples_normally_and_journals_nothing(tmp_path):
    ctx = _ctx(tmp_path, timeout_s=None, env=SlowEnv(per_frame_s=0.0))
    prov: dict = {}
    frames = sample_frames(ctx, _traj(), 8, prov=prov)
    assert len(frames) == 8 and prov.get("blind_reason", "") == ""
    assert _events(ctx, "sample_timeout") == []


def test_a_generous_deadline_changes_nothing(tmp_path):
    ctx = _ctx(tmp_path, timeout_s=60.0, env=SlowEnv(per_frame_s=0.0))
    prov: dict = {}
    assert len(sample_frames(ctx, _traj(), 8, prov=prov)) == 8
    assert _events(ctx, "sample_timeout") == []
