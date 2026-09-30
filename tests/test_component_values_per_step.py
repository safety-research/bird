"""`Trajectory.component_values` is packed PER STEP: index `t` is step `t`.

The disagreement between what a reward claimed to pay (`component_values`) and
what the harness measured it paid (`rewards`) exists only per step, and the
alignment is recorded, never recomputed. A `_rollout` in which `rewards` grows
every step but each component list grows only on the steps its key is emitted
would break that at the source: a reward that sets a key conditionally (`if
dist < 0.05: components["success_bonus"] = 10.0`, ordinary Eureka/RDA style) or
any step a swallowing `on_error` handled -- which is `exception_soft`, the
router EVERY config's final rollouts use -- would leave `component_values[k][i]`
describing a later step than `rewards[i]`. Every reader that pairs the two by
index (the trajectory trace, the RDA judge's per-step table, CARD's reflection
rows, the EPIC/STARC pairing) would then show a claim at the wrong instant: a
bonus paid in the last twenty steps listed on the first rows, beside frames
where nothing had happened. The mock LLM never emits a conditional key, so
nothing else in the suite would see it.

Each test below drives the real producer (`compile_reward` + `_rollout` on the
pure-numpy `toy_reacher` adapter) or a real reader, with per-step values that
are a function of the step index so a misplaced value names the step it came
from.
"""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest

from bird import demos, registry
from bird.components import evaluation as ev
from bird.components.training import _error_router, _evaluate_policy, _rollout, compile_reward
from bird.observability import _sample_series, save_trajectory_trace
from bird.types import Candidate, CandidateReport, Trajectory, TrainResult

#: `bonus` on odd steps only; `dist` every step; total = t.
ODD_BONUS = """
_t = [0]
def compute_reward(state, action, next_state):
    t = _t[0]; _t[0] += 1
    comps = {"dist": float(t)}
    if t % 2 == 1:
        comps["bonus"] = 100.0 + t
    return float(t), comps
"""

#: raises on step 2, otherwise total = t with one dense component.
RAISES_AT_2 = """
_t = [0]
def compute_reward(state, action, next_state):
    t = _t[0]; _t[0] += 1
    if t == 2:
        raise ValueError("boom")
    return float(t), {"dist": float(t)}
"""

#: a non-finite TOTAL on step 2, with the components computed fine.
NAN_AT_2 = """
_t = [0]
def compute_reward(state, action, next_state):
    t = _t[0]; _t[0] += 1
    return (float("nan") if t == 2 else float(t)), {"dist": float(t)}
"""

DENSE = """
_t = [0]
def compute_reward(state, action, next_state):
    t = _t[0]; _t[0] += 1
    return float(t), {"dist": float(t), "ctrl": -float(t)}
"""


def _env():
    registry.load_all()
    return registry.get("env", "toy_reacher")({})


def _policy(env):
    a0 = np.asarray(env.action_set[0], dtype=float)
    return lambda s: a0


def _roll(code, on_error=None):
    env = _env()
    traj, _steps, _gt = _rollout(env, _policy(env), np.random.default_rng(0),
                                 compile_reward(code), on_error)
    return traj


def _swallow():
    return _error_router("stdout_grep", io.StringIO())


def test_a_key_claimed_on_some_steps_reads_none_on_the_others():
    traj = _roll(ODD_BONUS)
    n = traj.length
    assert n >= 4 and len(traj.rewards) == n
    bonus = list(traj.component_values["bonus"])
    dist = list(traj.component_values["dist"])
    assert len(bonus) == n == len(dist), "one entry per step, claimed or not"
    for t in range(n):
        assert dist[t] == float(t)
        if t % 2:
            assert bonus[t] == 100.0 + t, f"step {t}: the claim made at step {t}"
        else:
            assert bonus[t] is None, f"step {t}: nothing was claimed here"
    assert list(traj.component_values) == ["dist", "bonus"], "first-emission key order kept"


def test_a_swallowed_raise_leaves_a_gap_at_that_step_and_shifts_nothing():
    traj = _roll(RAISES_AT_2, _swallow())
    n = traj.length
    dist = list(traj.component_values["dist"])
    assert len(dist) == n == len(traj.rewards)
    assert dist[2] is None, "the reward raised at step 2 and claimed nothing"
    assert dist[1] == 1.0 and dist[3] == 3.0 and dist[n - 1] == float(n - 1)
    assert traj.rewards[2] == 0.0 and traj.rewards[3] == 3.0


def test_a_non_finite_total_keeps_the_components_the_reward_did_compute():
    """The step whose claim the trace most needs: the total was NaN, the
    router paid 0.0, and the components say what the reward thought."""
    traj = _roll(NAN_AT_2, _swallow())
    dist = list(traj.component_values["dist"])
    assert dist[2] == 2.0 and traj.rewards[2] == 0.0
    assert dist[3] == 3.0 and len(dist) == traj.length


def test_the_default_router_still_raises():
    with pytest.raises(ValueError, match="boom"):
        _roll(RAISES_AT_2, _error_router("exception", io.StringIO()))


def test_a_dense_reward_is_packed_exactly_as_before():
    traj = _roll(DENSE)
    n = traj.length
    assert list(traj.component_values["dist"]) == [float(t) for t in range(n)]
    assert list(traj.component_values["ctrl"]) == [-float(t) for t in range(n)]
    assert all(v is not None for v in traj.component_values["dist"])


def test_the_trace_shows_no_claim_where_none_was_made(tmp_path):
    traj = _roll(ODD_BONUS)
    n = traj.length
    path = save_trajectory_trace(tmp_path, "c0", traj, list(range(n)),
                                 mode="components", max_steps=n)
    payload = json.loads(path.read_text())  # strict JSON: no NaN tokens
    steps = payload["samples"]["step"]
    for j, step in enumerate(steps):
        assert payload["measured"]["reward"][j] == float(step)
        claim = payload["claimed"]["bonus"][j]
        assert claim == (100.0 + step if step % 2 else None), (step, claim)


def test_sample_series_reads_none_and_non_finite_as_absent():
    assert _sample_series([1.0, None, float("nan"), float("inf"), 2.0],
                          [0, 1, 2, 3, 4, 5]) == [1.0, None, None, None, 2.0, None]


def test_the_judge_table_and_the_reflection_rows_omit_an_unclaimed_cell():
    traj = _roll(ODD_BONUS)
    table = ev._traj_component_table(traj, [0, 1, 2, 3])
    rows = {int(ln.split()[0].split("=")[1]): ln for ln in table.splitlines() if "step=" in ln}
    assert "bonus" not in rows[0] and "dist=0.000" in rows[0]
    assert "bonus=101.000" in rows[1]
    assert "bonus" not in rows[2] and "bonus=103.000" in rows[3]
    text = ev.render_trajectory(traj, "x", stride=1)
    lines = {int(ln.split()[0].split("=")[1]): ln for ln in text.splitlines() if "t=" in ln}
    assert "bonus" not in lines[0] and "dist=0.000" in lines[0]
    assert "bonus=101.000" in lines[1] and "n/a" not in text


def test_episode_component_means_are_over_the_claims_that_were_made():
    env = _env()
    _m, comps, _steps, trajs = _evaluate_policy(env, _policy(env), np.random.default_rng(0),
                                                compile_reward(ODD_BONUS), 1, None)
    claimed = [v for v in trajs[0].component_values["bonus"] if v is not None]
    assert comps["bonus"] == pytest.approx(float(np.mean(claimed)))
    assert math.isfinite(comps["bonus"]) and math.isfinite(comps["dist"])


def test_strided_keeps_an_unclaimed_step_at_its_own_index():
    assert demos.strided([1.0, None, 3.0], 10) == [1.0, None, 3.0]
    out = demos.strided([None] + [float(t) for t in range(1, 40)], 4)
    assert out[0] is None and out[-1] == 39.0


def test_epic_pairs_only_the_steps_where_the_reference_was_claimed():
    traj = Trajectory(rewards=[10.0, 20.0, 30.0],
                      component_values={"gt_reward": [1.0, None, 3.0]}, length=3, ret=60.0)
    cand = Candidate(cand_id="c0", iteration=0, reward_code="def compute_reward(): ...")
    rep = CandidateReport(cand_id="c0", candidate=cand,
                          result=TrainResult(cand_id="c0", candidate=cand, trajectories=[traj]),
                          fitness=1.0)
    assert ev._paired_rewards(None, None, rep) == ([10.0, 30.0], [1.0, 3.0])


def test_the_demo_trace_table_renders_an_unclaimed_step_in_place():
    """`screens.py`'s `demo_margin` writes `component_series` as
    `strided(ep0.component_values)`, which carries None on every step the reward
    did not claim the key -- so a `_demo_trace_table` that did `float(x)` would
    raise `TypeError` in stage 4 for any config pairing `demo_margin` with
    `demo_reward_traces` (`configs/hillclimb/v2_verify_noes.yaml`).
    The gap is rendered `n/a` AT ITS OWN INDEX: dropping it would misalign the
    component against `reward_series`, and 0.0 would assert a claim that was
    never made."""
    class _Ctx:
        cfg = {"evaluate.artifacts": ["demo_reward_traces"]}

    cand = Candidate(cand_id="c0", iteration=0, reward_code="def compute_reward(): ...")
    cand.meta["demo"] = {"policies": [{
        "name": "solution", "mean_per_step_return": 5.0, "episodes": 1,
        "reward_series": [0.0, 0.0, 10.0],
        "component_series": {"dist": [0.0, 1.0, 2.0], "success_bonus": [None, None, 10.0]},
    }], "margin": 0.5, "monotonicity": 1.0}
    text = ev._demo_trace_table(_Ctx(), cand)  # a float(x) cast would raise TypeError here
    rows = {ln.strip().split(":")[0]: ln for ln in text.splitlines() if ": [" in ln}
    assert rows["dist"].endswith("[0.000, 1.000, 2.000]")
    assert rows["success_bonus"].endswith("[n/a, n/a, 10.000]"), rows["success_bonus"]
    assert "per-step reward, episode 0 (strided): [0.000, 0.000, 10.000]" in text
    assert "best-vs-worst margin +0.500" in text
