"""A missing WANDB_API_KEY degrades the tracker IN PROCESS, never in the config.

`Config.hash()` covers every key by design, and `loop.resume_from: auto` finds the
leg it continues only through that hash (`runs/<name>-<hash>-*`). Degrading in the
config -- `output.tracker=none` whenever WANDB_API_KEY is absent -- would move that
hash, so a resumed leg run from a shell that differed from leg 1's only in that
variable would find no run directory to adopt and start a fresh search beside the
unfinished one. The rule -- key absent, no wandb --
lives in `bird/observability.py::make_tracker`, which swaps the tracker after the
config is final and writes down that it did. These tests hold the two parts of
that: the swap happens and touches neither the config nor its hash; and the swap is
recorded where the artifact is read. None of them needs `wandb` installed -- the
degraded path never imports it.
"""

from __future__ import annotations

import json
import logging

import pytest

from bird.artifacts import RunDir
from bird.config import load
from bird.context import Context
from bird.observability import NullTracker, WandbTracker, make_tracker


def _cfg(**over):
    return load("eureka", overrides={"output.tracker": "wandb", **over})


def _journal(rd: RunDir) -> list:
    return [json.loads(l) for l in (rd.path / "journal.jsonl").read_text().splitlines() if l.strip()]


def test_a_missing_key_degrades_to_none_and_the_artifact_says_so(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = _cfg()
    rd = RunDir(tmp_path, cfg["name"], cfg.hash())
    ctx = Context(cfg=cfg, budget=None, rundir=rd)
    with caplog.at_level(logging.WARNING, logger="bird.observability"):
        tracker = make_tracker(ctx)
    try:
        assert isinstance(tracker, NullTracker) and not isinstance(tracker, WandbTracker)
        expected = {"from": "wandb", "to": "none", "reason": "WANDB_API_KEY unset"}
        # The config is NOT what changed: the resolved value still says wandb.
        assert cfg["output.tracker"] == "wandb"
        # ... and it is loud in the three places the artifact is read from.
        assert any("wandb -> none" in r.getMessage() for r in caplog.records), caplog.text
        assert [e for e in _journal(rd) if e["stage"] == "tracker_degraded"][0] == {
            **expected, "t": pytest.approx(_journal(rd)[-1]["t"], abs=60), "stage": "tracker_degraded"}
        assert json.loads((rd.path / "status.json").read_text())["tracker_degraded"] == expected
    finally:
        rd.finish("ok")
        rd.close()
    # Survives the terminal status write: a reader of status.json on a finished run still sees it.
    status = json.loads((rd.path / "status.json").read_text())
    assert status["status"] == "ok" and status["tracker_degraded"] == expected


def test_the_config_hash_and_run_directory_do_not_depend_on_the_key(monkeypatch):
    """The whole point: two legs of one submit line, one with the key in its
    shell and one without, must land in the SAME `runs/<name>-<hash>-*`."""
    monkeypatch.setenv("WANDB_API_KEY", "k")
    with_key = _cfg()
    monkeypatch.delenv("WANDB_API_KEY")
    without = _cfg()
    assert with_key.hash() == without.hash()
    assert WandbTracker.degradation(Context(cfg=without, budget=None)) == {
        "from": "wandb", "to": "none", "reason": "WANDB_API_KEY unset"}
    # And degrading does not reach back into the config.
    make_tracker(Context(cfg=without, budget=None))
    assert without.hash() == with_key.hash() and without["output.tracker"] == "wandb"


def test_the_key_present_or_offline_mode_means_no_degradation(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    assert WandbTracker.degradation(Context(cfg=_cfg(), budget=None)) is None
    monkeypatch.delenv("WANDB_API_KEY")
    assert WandbTracker.degradation(
        Context(cfg=_cfg(**{"output.wandb.mode": "offline"}), budget=None)) is None, \
        "offline mode writes to the run dir and needs no credential"


def test_a_none_tracker_records_no_degradation(tmp_path, monkeypatch):
    """The key is absent from status.json unless a swap happened: a run under
    `none` has no dashboard to have lost."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = load("eureka", overrides={"output.tracker": "none"})
    rd = RunDir(tmp_path, cfg["name"], cfg.hash())
    try:
        assert isinstance(make_tracker(Context(cfg=cfg, budget=None, rundir=rd)), NullTracker)
        assert "tracker_degraded" not in json.loads((rd.path / "status.json").read_text())
        assert not [e for e in _journal(rd) if e["stage"] == "tracker_degraded"]
    finally:
        rd.close()
