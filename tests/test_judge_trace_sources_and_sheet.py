"""`output.judge_trace.record: inputs+frames` stores BOTH the source frames and
the sheet a composing `frame_policy` sent.

The mode's own rationale (`configs/_default.yaml`: "plus the PNGs and any
contact sheet"; `note_composed`'s docstring: "Store the COMPOSED image a policy
actually sent, beside its source frames"). Every in-loop
caller -- `evaluation._vlm_frames`, `preferences._pair_frames`,
`curriculum._stage_frames`, `phases.run_alignment_rate` -- runs
`sample_frames(prov=prov)` and THEN `note_frames(..., png=<delivered>)`. Under
`contact_sheet`, `_deliver` (inside `sample_frames`) calls `note_composed`, which
creates `frames/<id>/` and writes `sheet.png`. A `_write_pixels` that returns
because the directory exists would then store the sheet alone -- and the
caller's `note_frames` is in any case handed the DELIVERED list, the sheet, not
the four sources the record's `n_frames` and `sha256` describe. The record could
then not show the instant behind a sheet cell against its source frame, the
question the mode exists to answer.

These tests replay the callers' order against the real `sample_frames`,
`note_frames`, `_write_pixels` and `note_composed`, with no model involved.
"""

from __future__ import annotations

import hashlib

import numpy as np

from bird import judgments
from bird.artifacts import RunDir
from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.observability import sample_frames
from bird.types import Trajectory


class RenderingEnv:
    def render(self, state):
        v = int(float(np.asarray(state).ravel()[0])) % 256
        return np.full((8, 8, 3), (v, 40, 90), dtype=np.uint8)


def _ctx(tmp_path, *, policy: str, record: str = "inputs+frames") -> Context:
    cfg = load("eureka", profile="tester", overrides={
        "output.video.enabled": True, "output.video.width": 8,
        "output.judge_trace.record": record,
        "evaluate.vlm.frame_policy": policy})
    return Context(cfg=cfg, budget=Budget(), rundir=RunDir(tmp_path, "jt", "hash0"),
                   env=RenderingEnv())


def _traj(n: int = 40) -> Trajectory:
    return Trajectory(states=[[float(i)] for i in range(n)], rewards=[0.0] * n, length=n)


def _replay_caller(ctx, n_images: int = 4):
    """`sample_frames` then `note_frames(png=<delivered>)`: every caller's order."""
    prov: dict = {}
    delivered = list(sample_frames(ctx, _traj(), n_images, prov=prov))
    set_id = judgments.note_frames(ctx, prov, iteration=0, cand_id="c0", rollout=0,
                                   png=delivered)
    return prov, delivered, ctx.rundir.path / "judgments" / "frames" / set_id


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_composing_policy_stores_the_sources_and_the_sheet(tmp_path):
    prov, delivered, d = _replay_caller(_ctx(tmp_path, policy="contact_sheet"))
    assert len(delivered) == 1, "a contact sheet is one image"
    assert prov["n_frames"] == 4 and len(prov["sha256"]) == 4
    assert sorted(p.name for p in d.iterdir()) == [
        "frame_0000.png", "frame_0001.png", "frame_0002.png", "frame_0003.png", "sheet.png"]
    for i, sha in enumerate(prov["sha256"]):
        assert _sha(d / f"frame_{i:04d}.png").startswith(sha), (
            f"frame_{i:04d}.png must be the source frame the record hashed")
    assert _sha(d / "sheet.png").startswith(prov["composed"]["sheet_sha256"]), (
        "sheet.png must be the bytes that crossed the wire")
    assert _sha(d / "sheet.png") != _sha(d / "frame_0000.png")


def test_even_stores_the_frames_exactly_as_before(tmp_path):
    prov, delivered, d = _replay_caller(_ctx(tmp_path, policy="even"))
    assert len(delivered) == 4 and "composed" not in prov
    assert sorted(p.name for p in d.iterdir()) == [f"frame_{i:04d}.png" for i in range(4)]
    for i, sha in enumerate(prov["sha256"]):
        assert _sha(d / f"frame_{i:04d}.png").startswith(sha)


def test_inputs_mode_stores_no_pixels_under_either_policy(tmp_path):
    for policy in ("even", "contact_sheet"):
        ctx = _ctx(tmp_path / policy, policy=policy, record="inputs")
        prov: dict = {}
        delivered = list(sample_frames(ctx, _traj(), 4, prov=prov))
        judgments.note_frames(ctx, prov, iteration=0, cand_id="c0", rollout=0, png=delivered)
        assert not (ctx.rundir.path / "judgments" / "frames").exists(), policy


def test_write_pixels_judges_presence_per_file_not_per_directory(tmp_path):
    """The mechanism behind that loss: a directory `note_composed` created for
    the sheet must not read as "the frames are already here"."""
    ctx = _ctx(tmp_path, policy="even")
    judgments.note_composed(ctx, "setid", b"\x89PNG-sheet")
    d = ctx.rundir.path / "judgments" / "frames" / "setid"
    assert sorted(p.name for p in d.iterdir()) == ["sheet.png"]
    judgments.note_source_frames(ctx, "setid", [b"\x89PNG-a", b"\x89PNG-b"])
    assert sorted(p.name for p in d.iterdir()) == ["frame_0000.png", "frame_0001.png", "sheet.png"]
    assert (d / "frame_0001.png").read_bytes() == b"\x89PNG-b"
    # idempotent once the frames are present: a second set with the same id
    # holds the same bytes, and the sheet must not be mistaken for a frame
    judgments.note_source_frames(ctx, "setid", [b"\x89PNG-other"])
    assert (d / "frame_0000.png").read_bytes() == b"\x89PNG-a"
