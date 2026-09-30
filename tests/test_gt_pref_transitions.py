"""GT stores real `(s, a, s')` transitions for TAC, and shows the judge an
even-strided view separately.

`_clip_for` even-strides the source window down to the frames the judge sees.
Storing THAT clip in `D_pref` would be wrong: `screens.screen_tac` reconstructs
`(states[t], actions[t], states[t+1])` from it, so a candidate's own reward
would be re-scored on states up to 13 real steps apart paired with a single
earlier action, with the terminal next-state dropped. TAC could then reject a
correctly aligned terminal/progress reward or prefer a distorted one -- a
plausible sigma computed on transitions that never occurred.

`_pref` therefore stores `_pref_traj_for`: the same source window at full
resolution, so its transitions are real. The view clip stays the subsample.
"""
from __future__ import annotations

import random

import numpy as np

from bird import registry
from bird.budget import Budget
from bird.components.preferences import _clip_for, _pref_traj_for
from bird.components.screens import _traj_transitions
from bird.config import load
from bird.context import Context
from bird.types import Candidate, CandidateReport, TrainResult, Trajectory


def _ctx():
    registry.load_all()
    return Context(cfg=load("gt", profile="tester"), budget=Budget(), rng=random.Random(0))


def _report(traj: Trajectory) -> CandidateReport:
    cand = Candidate("c", 0, "def reward(s, a, s2): return 0.0, {}")
    return CandidateReport("c", cand, TrainResult("c", cand, trajectories=[traj]), fitness=0.0)


def test_the_stored_trajectory_keeps_every_real_transition():
    """Three transitions, reward only at the terminal state. Storing the view
    clip would drop to two transitions and lose the terminal next-state."""
    traj = Trajectory(states=np.array([[0.0], [1.0], [2.0], [3.0]]),
                      actions=np.array([[0.1], [0.2], [0.3]]),
                      rewards=[0.0, 0.0, 1.0], success=True, length=3)
    sub = _pref_traj_for(_ctx(), _report(traj))
    trans = _traj_transitions(sub)
    assert trans is not None and len(trans) == 3, "all three transitions, not two"
    assert [float(t.next_state[0]) for t in trans] == [1.0, 2.0, 3.0], "terminal state kept"
    assert [float(t.state[0]) for t in trans] == [0.0, 1.0, 2.0]


def test_a_long_rollout_keeps_adjacent_transitions():
    """A 1800-step rollout. Storing the view clip would give 150 states whose
    adjacent pairs are up to 13 real steps apart. TAC re-scores adjacency, so
    the jumps would corrupt the score."""
    n = 1800
    traj = Trajectory(states=np.arange(n + 1, dtype=float).reshape(-1, 1),
                      actions=np.arange(n, dtype=float).reshape(-1, 1),
                      rewards=[0.0] * n, success=False, length=n)
    sub = _pref_traj_for(_ctx(), _report(traj))
    trans = _traj_transitions(sub)
    jumps = [abs(float(t.next_state[0]) - float(t.state[0])) for t in trans]
    assert max(jumps) == 1.0, "every stored transition is a real adjacent step"
    assert len(trans) == n


def test_the_window_bounds_the_stored_trajectory():
    """The stored trajectory is the first `max_video_s * fps` transitions, the
    same window the view is drawn from -- not the whole episode past it, and not
    the subsample."""
    ctx = _ctx()
    fps = int(ctx.cfg.get("evaluate.preferences.fps") or 10)
    max_s = int(ctx.cfg.get("evaluate.preferences.max_video_s") or 180)
    window = max_s * fps
    n = window + 500
    traj = Trajectory(states=np.arange(n + 1, dtype=float).reshape(-1, 1),
                      actions=np.arange(n, dtype=float).reshape(-1, 1),
                      rewards=[0.0] * n, success=False, length=n)
    sub = _pref_traj_for(ctx, _report(traj))
    assert sub.length == window
    assert _traj_transitions(sub) is not None and len(_traj_transitions(sub)) == window


def test_the_view_clip_is_unchanged_and_subsampled():
    """The judge's frames stay an even-stride subsample; only what TAC re-scores
    is full resolution. A 1800-step episode still shows a `clip_length_s * fps` view."""
    ctx = _ctx()
    fps = int(ctx.cfg.get("evaluate.preferences.fps") or 10)
    clip_s = int(ctx.cfg.get("evaluate.preferences.clip_length_s") or 15)
    n = 1800
    traj = Trajectory(states=np.arange(n + 1, dtype=float).reshape(-1, 1),
                      actions=np.arange(n, dtype=float).reshape(-1, 1),
                      rewards=[0.0] * n, success=False, length=n)
    view = _clip_for(ctx, _report(traj))
    assert view.length == clip_s * fps, "the view is still the subsampled clip"
    assert view.length < n


def test_the_view_and_the_tac_trajectory_are_distinct_objects():
    ctx = _ctx()
    traj = Trajectory(states=np.array([[0.0], [1.0], [2.0], [3.0]]),
                      actions=np.array([[0.1], [0.2], [0.3]]),
                      rewards=[0.0, 0.0, 1.0], success=True, length=3)
    rep = _report(traj)
    assert _pref_traj_for(ctx, rep) is not _clip_for(ctx, rep)
