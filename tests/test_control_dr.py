"""Acrobot honours the `EnvAdapter` domain-randomisation contract (`train.domain_randomization`).

An adapter that overrides `set_dr`, `_sample_dr` and `dr_config` (bird/envs/control.py)
can break the contract `bird/envs/base.py` documents three ways at once: a `set_dr` doing
`float(v)` raises TypeError at `env.set_dr(...)` in stage 3 on the `[lo, hi]` blob
`generation._parse_dr_config` emits for DrEureka (`mode: generated`, the only mode any
config selects); a `_sample_dr` drawing into `_dr_now` while the dynamics (`_p`) read a
separate `_dr` randomises no episode, the draws only shifting the start state; and a
`dr_config` reporting the full table as installed at nominal misstates what ran. The suite
cannot see any of it otherwise: the mock LLM emits no DR block, so every tester run calls
`set_dr(None)`.

The reference for what must NOT change is the equations of motion from a given state. That
rollout is pinned here.

A second hole follows once the draw reaches `_p`: a `tip_height` that read the link lengths
through it would move `task_metric`, `success()` (a threshold on `task_metric`) and
`reference_reward` with the DR blob -- and `training._evaluate_policy` scores `task_metric`
INSIDE the `set_dr` window, on a blob that is LLM-authored under `mode: generated`.
Measured: 1.4142 at nominal, 1.6263 with both lengths drawn at 1.15, so
`link_length_*: [1.15, 1.15]` lifts a nominal 0.87 over the 1.0 bar. A candidate choosing
its DR would be choosing its ground truth. `tip_height` therefore reads the class
attributes (nominal geometry) and only `_dsdt` reads `_p`. Separately, `link_length_2`
enters no equation of motion at all (Sutton-Barto; gymnasium's l2 is render-only), so as
an axis it would randomise the metric and the reference reward and nothing physical; it is
not in the spec, and the last two tests here hold both halves.
"""
from __future__ import annotations

import numpy as np
import pytest

from bird import registry

#: The spec's `domain_randomization` axes. `link_length_2` is deliberately NOT one: it enters
#: no equation of motion (see `test_link_length_2_is_not_an_axis_because_it_moves_no_dynamics`).
AXES = ("link_mass_1", "link_mass_2", "link_length_1")
NOMINAL = {k: 1.0 for k in AXES}

#: Final state of 300 steps from `_reset(default_rng(3))` under `default_rng(11).uniform(-1, 1,
#: size=(300, 1))` actions with no DR installed -- MEASURED with per-adapter DR overrides
#: and bit-identical without them. The DR contract moves the reset stream, not the physics;
#: this is what says so.
PRE_FIX_FINAL_STATE = [0.9983685655117842, -0.05709822587385883, 0.870792556479864, 0.49165061128738863, 0.04173125709424261, 0.8959529980712051]


@pytest.fixture
def acrobot():
    registry.load_all()
    return registry.get("env", "acrobot")(None)


def test_the_dr_surface_is_the_spec_s(acrobot):
    # `SpecEnvAdapter._apply_spec` installs `domain_randomization` from the task spec; the
    # adapter carries no second copy of the table (two homes for one fact is the bug class).
    assert set(acrobot.dr_parameters) == set(AXES)
    assert acrobot._dr_nominal == NOMINAL
    assert not hasattr(acrobot, "_DR") and not hasattr(acrobot, "_dr")


@pytest.mark.parametrize("blob", [
    {"link_mass_1": [0.8, 1.2]},                 # what `_parse_dr_config` always emits
    {"link_mass_1": (0.8, 1.2)},
    {"link_mass_1": {"low": 0.8, "high": 1.2}},
], ids=["list", "tuple", "mapping"])
def test_every_documented_dr_shape_installs(acrobot, blob):
    acrobot.set_dr(blob)                         # a `float(v)` set_dr raises TypeError here
    assert acrobot.dr_config == {"link_mass_1": (0.8, 1.2)}


def test_a_range_is_drawn_once_per_episode_and_reaches_the_dynamics(acrobot):
    acrobot.set_dr({"link_mass_1": (0.7, 1.3)})
    acrobot.reset(np.random.default_rng(1))
    p1 = acrobot._p("link_mass_1")
    acrobot.reset(np.random.default_rng(2))
    p2 = acrobot._p("link_mass_1")
    assert p1 != p2, "two seeded resets under a range drew the same mass: nothing is randomised"
    assert all(0.7 <= p <= 1.3 for p in (p1, p2))
    # The draw is what the equations of motion integrate: same state, same action, a
    # different draw -> a different next state. With the draw not reaching `_p` this
    # difference is 0.0.
    s = acrobot._reset(np.random.default_rng(3))
    a = np.array([0.7])
    acrobot._dr_now["link_mass_1"] = 0.7
    lo = acrobot.step(s, a)[0]
    acrobot._dr_now["link_mass_1"] = 1.3
    hi = acrobot.step(s, a)[0]
    assert not np.array_equal(lo, hi)


def test_nominal_means_nominal(acrobot):
    acrobot.set_dr({"link_mass_1": (0.7, 1.3)})
    acrobot.set_dr(None)
    assert acrobot.dr_config == {}, "dr_config must be empty at nominal, not the allowed table"
    acrobot.reset(np.random.default_rng(5))
    assert {k: acrobot._p(k) for k in AXES} == NOMINAL
    # With nothing to randomise, `_sample_dr` draws nothing: the start state is the FIRST
    # four uniforms of the episode stream (the spec's `rng_note`), not draws 5-8.
    got = acrobot.reset(np.random.default_rng(3))
    want = acrobot._obs(np.random.default_rng(3).uniform(-0.1, 0.1, size=4))
    assert np.array_equal(got, want)


def test_nominal_dynamics_are_bit_identical_to_the_pre_fix_code(acrobot):
    acrobot.set_dr(None)
    s = acrobot._reset(np.random.default_rng(3))
    for a in np.random.default_rng(11).uniform(-1.0, 1.0, size=(300, 1)):
        s, _done, _info = acrobot.step(s, a)
    assert s.tolist() == PRE_FIX_FINAL_STATE


def _obs(acrobot, th1, th2, d1=0.0, d2=0.0):
    return acrobot._obs(np.array([th1, th2, d1, d2]))


def _every_dr_reading(acrobot):
    """Ground-truth numbers on fixed inputs, to be compared across DR installs."""
    s = _obs(acrobot, 3 * np.pi / 4, 0.0)            # tip at sqrt(2) nominal link-lengths
    # th2 = 0, th1 over [0, pi]: heights -2..2, and a run of states in (1/1.15, 1.0) -- below
    # the bar on nominal geometry, above it at lengths 1.15 -- so `task_metric` would move
    # under a DR-dependent reading, not merely `tip_height`.
    traj = np.array([_obs(acrobot, th, 0.0) for th in np.linspace(0.0, np.pi, 300)])
    return (acrobot.tip_height(s), acrobot.task_metric(traj),
            acrobot.reference_reward(s, np.array([0.5])))


def test_the_ground_truth_reads_nominal_geometry_under_any_draw(acrobot):
    acrobot.set_dr(None)
    nominal = _every_dr_reading(acrobot)
    assert nominal[0] == pytest.approx(np.sqrt(2.0))
    # Gymnasium's own dimensionless check: fraction of th1 in [0, pi] with -2 cos(th1) > 1.
    assert nominal[1] == pytest.approx(1.0 / 3.0, abs=0.01)
    # A degenerate range is a fixed draw, so this is exactly the blob a candidate would emit
    # to lift its own bar; `set_dr` puts it into `_dr_now` at once and every reset re-draws it.
    acrobot.set_dr({"link_length_1": [1.15, 1.15], "link_mass_1": [1.3, 1.3]})
    acrobot.reset(np.random.default_rng(0))
    assert acrobot._p("link_length_1") == 1.15, "the draw did not install: this test is not probing"
    assert _every_dr_reading(acrobot) == nominal
    # Belt and braces: whatever the spec's table holds, nothing in `_dr_now` may reach it.
    # Geometry read through `_p` would make this pair 1.6263 (+15%) against 1.4142.
    acrobot._dr_now.update({"link_length_1": 1.15, "link_length_2": 1.15})
    assert _every_dr_reading(acrobot) == nominal
    acrobot._dr_now.update({"link_length_1": 0.85, "link_length_2": 0.85})
    assert _every_dr_reading(acrobot) == nominal


@pytest.mark.parametrize("axis", AXES)
def test_every_spec_dr_axis_changes_the_dynamics(acrobot, axis):
    # An axis in the spec's `domain_randomization` that `_dsdt` never reads randomises
    # nothing physical -- which is exactly what `link_length_2` would be.
    lo, hi = acrobot.dr_parameters[axis]
    s = acrobot._reset(np.random.default_rng(3))
    a = np.array([0.7])
    acrobot.set_dr(None)
    acrobot._dr_now[axis] = lo
    at_lo = acrobot.step(s, a)[0]
    acrobot._dr_now[axis] = hi
    at_hi = acrobot.step(s, a)[0]
    assert not np.array_equal(at_lo, at_hi), f"{axis} in [{lo}, {hi}] leaves step() unchanged"


def test_link_length_2_is_not_an_axis_because_it_moves_no_dynamics(acrobot):
    assert "link_length_2" not in acrobot.dr_parameters
    assert "link_length_2" not in acrobot._dr_nominal
    # The measurement behind dropping it: from one (s, a), 0.85 and 1.15 give the same next
    # state bit for bit. (Sutton & Barto's equations carry l1 and lc2, never l2.)
    s = acrobot._reset(np.random.default_rng(3))
    a = np.array([0.7])
    acrobot.set_dr(None)
    acrobot._dr_now["link_length_2"] = 0.85
    at_lo = acrobot.step(s, a)[0]
    acrobot._dr_now["link_length_2"] = 1.15
    at_hi = acrobot.step(s, a)[0]
    assert np.array_equal(at_lo, at_hi)
