"""A pairwise judge shown a contact sheet is told what it is, and the labels it
sees, the manifest it reads and the record's `steps` are ONE index space.

frames.py's header: "the label and the manifest carry the same numbers ... the
index is the EPISODE STEP", because a judge naming "frame 137" and a record
resolving "frame 137" must mean one instant. The pairwise path
(`preferences._pair_frames`) can break that: running `sample_frames` on the CLIP
`_clip_for` cut (indices 0..149 of a 450-step episode) and remapping
`prov["steps"]` to episode indices only afterwards is too late, because
`_deliver` composes the sheet inside `sample_frames` -- the burned cell labels,
`composed.steps` and the manifest would stay clip-local while the record says
0, 90, 181, .... Independently, the pairwise prompt (`_render`, `_sides`) and
`alignment_rate`'s must say what the one attached image is: "1 evenly-spaced
frames of this clip are attached", over a 4x5 grid with unexplained numbers, does
not.

No shipped config selects `contact_sheet`; these pin a config-space point with
no model involved (a spy client that would take images stands for the VLM).
"""

from __future__ import annotations

import random

import numpy as np

from bird import registry
from bird.budget import Budget
from bird.components import preferences as P
from bird.components.frames import sheet_cells, sheet_manifest
from bird.config import load
from bird.context import Context
from bird.types import Candidate, CandidateReport, Trajectory, TrainResult


class _SpyVLM:
    modality = "vlm"

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        return ["LEFT"] * n


class RenderingEnv:
    def render(self, state):
        v = int(float(np.asarray(state).ravel()[0])) % 256
        return np.full((8, 8, 3), (v, 40, 90), dtype=np.uint8)


EPISODE = 450   # > gt's 15 s x 10 fps = 150-frame clip budget, so the clip is STRIDED


def _ctx(policy: str) -> Context:
    registry.load_all()
    cfg = load("gt", profile="tester", overrides={
        "output.video.enabled": True, "output.video.width": 8,
        "evaluate.vlm.frame_policy": policy, "evaluate.vlm.images_per_query": 6})
    ctx = Context(cfg=cfg, budget=Budget(), rundir=None, rng=random.Random(0), env=RenderingEnv())
    ctx.evaluator = _SpyVLM()
    return ctx


def _report(cid: str) -> CandidateReport:
    traj = Trajectory(states=[[float(i)] for i in range(EPISODE)], rewards=[0.0] * EPISODE,
                      length=EPISODE, ret=0.0)
    cand = Candidate(cand_id=cid, iteration=0, reward_code="def compute_reward(): ...")
    return CandidateReport(cand_id=cid, candidate=cand, fitness=1.0,
                           result=TrainResult(cand_id=cid, candidate=cand, trajectories=[traj]))


def test_the_sheets_labels_its_manifest_and_the_record_share_the_episode_index_space():
    ctx = _ctx("contact_sheet")
    rep = _report("c0")
    png, prov = P._pair_frames(ctx, rep)
    assert len(png) == 1, "one sheet"
    window = rep.meta[P._CLIP_STEPS_KEY]
    assert len(window) == 150 and window[-1] == EPISODE - 1, "a strided clip, as gt cuts it"
    steps = prov["steps"]
    assert len(steps) == 6 and max(steps) > 150, "episode indices, not clip-local 0..149"
    assert all(s in window for s in steps), "every recorded step is a step the clip holds"
    comp = prov["composed"]
    assert comp["steps"] == steps, "the burned labels are the record's steps"
    for s in steps:
        assert str(s) in comp["manifest"], f"the manifest must list episode step {s}"
    assert "Steps shown, in reading order: " + ", ".join(str(s) for s in steps) in comp["manifest"]


def test_the_pairwise_prompt_says_it_is_a_sheet_and_carries_the_manifest():
    ctx = _ctx("contact_sheet")
    rep = _report("c0")
    text = P._render(ctx, rep, "LEFT", attach=True)
    _png, prov = P._pair_frames(ctx, rep)
    assert "one contact sheet tiling 6 evenly-spaced frames of this clip is attached" in text
    assert "1 evenly-spaced frames" not in text
    for s in prov["steps"]:
        assert str(s) in text, "the labels the judge sees are named in the prompt"
    assert "Steps shown, in reading order" in text


def test_the_attached_images_sentence_describes_sheets_as_sheets():
    ctx = _ctx("contact_sheet")
    body, images = P._sides(ctx, _report("c0"), _report("c1"), attach=True)
    assert len(images) == 2
    assert ("ATTACHED IMAGES, in order: the first 1 (one contact sheet tiling 6 frames) "
            "are LEFT's clip, the next 1 (one contact sheet tiling 6 frames) are RIGHT's") in body
    assert "labelled with the EPISODE step" in body


def test_even_keeps_the_old_wording_byte_for_byte():
    ctx = _ctx("even")
    rep = _report("c0")
    text = P._render(ctx, rep, "LEFT", attach=True)
    assert "6 evenly-spaced frames of this clip are attached" in text
    assert "contact sheet" not in text and "Steps shown" not in text
    body, images = P._sides(ctx, _report("c0"), _report("c1"), attach=True)
    assert len(images) == 12
    assert ("ATTACHED IMAGES, in order: the first 6 are LEFT's clip, the next 6 are RIGHT's. "
            "Both are evenly spaced over the same window of each agent's episode.") in body
    _png, prov = P._pair_frames(ctx, rep)
    assert max(prov["steps"]) > 150 and "composed" not in prov


def test_the_sheet_readers_are_none_and_empty_when_nothing_was_composed():
    assert sheet_cells(None) is None and sheet_manifest(None) == ""
    assert sheet_cells({"composed": {"frame_policy": "contact_sheet", "degraded": "x"}}) is None
    assert sheet_cells({"composed": {"n_cells": 4, "sheet_sha256": "ab"}}) == 4
    assert sheet_manifest({"composed": {"manifest": " m \n"}}) == "m"
