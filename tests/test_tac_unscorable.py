"""GT's TAC screen screens a candidate it could not score; it does not pass it.

A `screen_tac` that `continue`s past a live candidate whose reward raised or
returned a non-finite value on every stored pair's window (`usable == 0`)
leaves it absent from `scores`; `_rank_keep` ranks only `scores`, and
`_cold_start_cut` fires only when `scores` is empty -- so in a mixed pool the
unscorable candidates would stay TRAINABLE on top of the `keep_top_n` ranked
survivors, with no `screen_pass`, `screen_skip` or rejection at all. More than
`keep_top_n` would train (GT pins five, five ways), and the one that escaped
would be the reward pathology `_rescore_transitions` exists to count AGAINST a
candidate (the rule the TPE screen already applies). Reachable under
`configs/methods/gt.yaml`: its `dynamic_checks: [execution_smoke]` catches a raise on
sampled transitions but not a NaN, so a NaN-everywhere reward passes validity
and would pass the screen with zero events.

So: unscorable in a pool where something scored -> screened (`failure_kind
"screened"`, a `screen_reject` event with reason `unscorable_tac`); a pool
where NOTHING scored -> the documented cold-start cut; `keep_top_n` is an upper
bound on the trainable count after the screen either way.
"""

from __future__ import annotations

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components import screens
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, Preference, Trajectory

_GOOD = ("def compute_reward(state, action, next_state):\n"
         "    return -float(abs(next_state[0]))\n")
_GOOD2 = ("def compute_reward(state, action, next_state):\n"
          "    return float(next_state[1])\n")
_RAISING = ("def compute_reward(state, action, next_state):\n"
            "    return 1.0 / 0.0\n")
_NAN = ("def compute_reward(state, action, next_state):\n"
        "    return float('nan')\n")


def _traj(seed: int, n: int = 8) -> Trajectory:
    rng = np.random.default_rng(seed)
    return Trajectory(states=rng.normal(size=(n, 6)), actions=rng.normal(size=(n, 2)),
                      rewards=[0.0] * n, length=n, ret=0.0)


def _state(n_pairs: int = 4) -> RunState:
    st = RunState()
    st.preferences = [Preference(left_id=f"l{i}", right_id=f"r{i}", label=i % 2,
                                 left_traj=_traj(2 * i), right_traj=_traj(2 * i + 1))
                      for i in range(n_pairs)]
    return st


def _ctx(keep: int):
    registry.load_all()
    cfg = load("gt", profile="tester", overrides={
        "verify.alignment_filter.keep_top_n": keep,
        "verify.alignment_filter.subtraj_len": 0})
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    events = []
    ctx.event = lambda stage, **fields: events.append({"stage": stage, **fields})
    return ctx, events


def _pool(*codes: str):
    return [Candidate(cand_id=f"c{i:04d}", iteration=0, reward_code=code)
            for i, code in enumerate(codes)]


def test_a_raising_candidate_is_screened_and_keep_top_n_holds() -> None:
    ctx, events = _ctx(keep=1)
    cands = _pool(_GOOD, _RAISING, _GOOD2)

    screens.screen_tac(ctx, _state(), cands)

    trainable = [c for c in cands if c.trainable]
    assert len(trainable) == 1, [(c.cand_id, c.trainable) for c in cands]
    raising = cands[1]
    assert not raising.trainable and raising.failure_kind == "screened"
    assert "unscorable" in raising.failure
    rejects = [e for e in events if e["stage"] == "screen_reject"]
    assert [e["cand_id"] for e in rejects] == ["c0001"]
    assert rejects[0]["reason"] == "unscorable_tac" and rejects[0]["screen"] == "tac"
    # the scored ones went through the rank cut as before
    assert [e["cand_id"] for e in events if e["stage"] == "screen_pass"] == [trainable[0].cand_id]


def test_a_nan_candidate_is_unscorable_too() -> None:
    """The shape that passes gt.yaml's validity checks: no raise, no finite value."""
    ctx, events = _ctx(keep=2)
    cands = _pool(_GOOD, _NAN, _GOOD2)

    screens.screen_tac(ctx, _state(), cands)

    assert [c.cand_id for c in cands if c.trainable] == ["c0000", "c0002"]
    assert cands[1].failure_kind == "screened"
    assert [e["cand_id"] for e in events if e["stage"] == "screen_reject"] == ["c0001"]


def test_an_all_unscorable_pool_still_takes_the_cold_start_cut() -> None:
    """Nothing scored, so nothing ranks: the documented count cut applies and
    it is journalled as such -- not as `keep_top_n` fabricated rejections."""
    ctx, events = _ctx(keep=1)
    cands = _pool(_RAISING, _NAN, _RAISING)

    screens.screen_tac(ctx, _state(), cands)

    assert sum(c.trainable for c in cands) == 1
    cold = [e for e in events if e["stage"] == "screen_cold_start"]
    assert len(cold) == 1 and cold[0]["n_kept"] == 1 and cold[0]["n_screened"] == 2
    assert not [e for e in events if e["stage"] == "screen_reject"]


def test_keep_top_n_is_an_upper_bound_whatever_the_mix() -> None:
    for keep in (1, 2, 3, 5):
        ctx, _events = _ctx(keep=keep)
        cands = _pool(_GOOD, _RAISING, _GOOD2, _NAN, _GOOD)
        screens.screen_tac(ctx, _state(), cands)
        assert sum(c.trainable for c in cands) <= keep
        assert sum(c.trainable for c in cands) == min(keep, 3), (
            f"keep={keep}: the {3} scorable candidates are what the cut ranks")
