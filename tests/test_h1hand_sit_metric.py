"""`H1HandSitSimple.task_metric` divides by the HORIZON, not by the rows visited.

A metric that reads `np.mean(_seated(states[1:]))` -- a fraction of whatever rows the
trajectory happens to hold -- is exploitable, because both consumers stop the rollout
at `done`:
`training._rollout` breaks on the terminal and `policy_api.run_episode` loops
`while not done`. `Sit.get_terminated` fires at pelvis z < 0.5 m, and a robot toppling
BACKWARDS off the spawn passes through the seat band (pelvis over the footprint, z in
the seated band) on its way down, so the fall itself becomes the scored trajectory:
under that mean, 9 of 30 uniform-random episodes clear `success_threshold` 0.25 (mean
0.147, max 0.604), every "seated" step landing before the terminal. `_H1HandBalance` and
`H1HandWindow` in the same module divide by the horizon for exactly this reason --
"leaving the scoring posture is what ends the episode" -- and the spec declares
`unit: fraction of horizon`.

Sim-free on purpose: synthetic trajectories through the metric, called unbound the way
`tests/test_humanoid_hand.py::_metric` does, so this runs in the default selection
rather than behind the `humanoid` marker. The number that must NOT come out is 0.5.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from bird.envs import humanoid_hand as HH

#: A pelvis seated over the chair: the footprint's centre, 0.225 m above the seat surface
#: (the shipped reward's own (0.68, 0.72) kernel puts it there; the metric's band is wider).
SEATED = (-0.25, 0.0, 0.70)
#: Where a backward topple ends up: over the footprint, below the 0.5 m terminal.
FLOOR = (-0.25, 0.0, 0.30)


#: Read off the CLASS rather than the module constant, so that a visited-rows metric
#: fails these tests on the number (0.5 against 0.015) and not on an AttributeError.
HORIZON = HH.H1HandSitSimple.horizon


def _rows(n):
    """`n` resting states: pelvis standing at 0.98 m, identity quaternion, rest zero."""
    s = np.zeros((n, HH.H1HandSitSimple.obs_dim))
    s[:, 2] = 0.98
    s[:, 3] = 1.0
    return s


def _metric(states):
    """Unbound, `self` None -- the sim-free convention `test_humanoid_hand.py` argues for."""
    return HH.H1HandSitSimple.task_metric(None, states)


def _env():
    """An adapter with `__init__` skipped (`test_humanoid_hand.py::_metric_only`'s idiom),
    for `success()`, which is a threshold on the bound metric."""
    return object.__new__(HH.H1HandSitSimple)


def test_a_topple_through_the_seat_band_forfeits_the_steps_it_did_not_run():
    """The exploit's shape: 30 arrivals, the first 15 inside the band over the footprint,
    the next 15 on the floor, then the terminal. Fifteen seated steps of a 1000-step
    episode are 0.015 -- not the 0.5 a visited-rows mean reads, which is double
    `success_threshold`."""
    topple = _rows(31)
    topple[1:16, 0:3] = SEATED
    topple[16:, 0:3] = FLOOR
    value = _metric(topple)
    assert value == pytest.approx(15 / HORIZON)
    assert value != pytest.approx(0.5), "a fraction of the rows visited is the old bug"
    assert not _env().success(SimpleNamespace(states=topple)), (
        "a backward fall through the band is not a successful sit")


def test_a_full_horizon_seated_trajectory_still_reads_one():
    """A trajectory that runs the whole horizon is unchanged by the denominator: a
    full-horizon seated trajectory keeps its score."""
    seated = _rows(HORIZON + 1)
    seated[:, 0:3] = SEATED
    assert _metric(seated) == pytest.approx(1.0)
    assert _env().success(SimpleNamespace(states=seated))


@pytest.mark.parametrize("n_arrivals", [10, 100, 999])
def test_the_denominator_is_the_horizon_whatever_the_episode_length(n_arrivals):
    """Every arrival seated, then termination: the score is arrivals / horizon, so the
    same posture held for a tenth of the episode is worth a tenth, not full marks."""
    seated = _rows(n_arrivals + 1)
    seated[:, 0:3] = SEATED
    assert _metric(seated) == pytest.approx(n_arrivals / HORIZON)


def test_the_floor_and_the_spawn_still_read_zero():
    """The near-misses stay at 0.0 under the horizon denominator; the denominator
    changes only what a PARTIAL trajectory is divided by."""
    assert _metric(_rows(20)) == 0.0                       # standing at the spawn
    floor = _rows(20)
    floor[:, 0:3] = FLOOR
    assert _metric(floor) == 0.0                           # collapsed over the footprint
    assert _metric(_rows(1)) == 0.0                        # no arrivals at all


def test_the_horizon_constant_is_the_class_horizon_and_the_specs():
    """`_SIT_HORIZON` is module-level only so the metric can be called unbound; the class
    binds `horizon` to it and the spec declares the same 1000, so a change to any one of
    the three without the others fails here rather than moving the score silently."""
    from bird import tasks

    assert HH.H1HandSitSimple.horizon == HH._SIT_HORIZON == 1000
    assert int(tasks.load("h1hand_sit_simple").env["horizon"]) == HH._SIT_HORIZON
