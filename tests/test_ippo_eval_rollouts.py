"""The JAX tier returns its evaluation rollouts as `Trajectory` objects.

WHY. A worker that returns scalars only -- `gt_return`, `r_pref`, the eval payload --
leaves `TrainResult.trajectories` empty, so every fitness source that reads a rollout
(`vlm_score` renders frames off states; `ground_truth_metric` and `success_rate` read
the banked rows) has nothing to read: an arm scored that way selects on
`fitness = -10000.0` every iteration, while arms whose fitness is the native scalar run
normally. Three pieces, each tested here without a simulator:

  1. `row_logging_class` -- the eval wrapper puts the adapter's row in `infos`
     (never in `state.metrics`: upstream's auto-reset select needs both branches to
     share a pytree, and only the stepped branch would carry the key).
  2. `trajectories_from_eval` -- the pure host-side fold from upstream's stacked
     eval logs to `Trajectory` objects: task reward is `reward - pref` (R_pref
     stays held out), the episode is cut at its first `done`, success is the
     adapter's own predicate.
  3. `build_train_result` -- ships the LAST seed's trajectories, the same rule as
     `policy_ref`.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bird.components import _ippo_backend as B
from bird.components import _ippo_reward_env as RE
from bird.components import _ippo_train as J
from bird.types import Candidate, Trajectory


# ---------------------------------------------------------------- 1. the wrapper


class _FakeBase:
    """A stand-in for `LoadAgentWrapper`: `step` returns the five-tuple."""

    def __init__(self):
        self.calls = 0

    def step(self, key, state, actions, reset_state=None):
        self.calls += 1
        return ({"robot": np.zeros(3)}, {"t": state["t"] + 1},
                {"robot": 1.0, "__all__": 1.0}, {"__all__": False},
                {"upstream_key": 7})


def test_the_row_logger_adds_the_row_to_infos_and_nothing_else():
    seen = []

    def to_row(nxt, prev):
        seen.append((nxt["t"], prev["t"]))
        return np.full(4, float(nxt["t"]))

    cls = RE.row_logging_class(_FakeBase, to_row)
    assert cls.bird_row_logged is True and cls.bird_row_key == "bird_row"
    assert cls.__name__ == "RowLogged_FakeBase"
    env = cls()
    obs, nxt, rew, done, infos = env.step(None, {"t": 3}, {"robot": np.zeros(7)})
    assert nxt == {"t": 4} and rew["robot"] == 1.0 and done == {"__all__": False}
    # the upstream key survives, the row is added, and it is the NEXT state's row
    # differenced against the PREVIOUS state (tool velocity is a finite difference)
    assert infos["upstream_key"] == 7
    assert np.array_equal(infos["bird_row"], np.full(4, 4.0))
    assert seen == [(4, 3)]
    assert env.calls == 1


def test_the_row_logger_does_not_touch_the_state():
    """The reason it is `infos`: a key in `state.metrics` would be present on the
    stepped branch and absent on upstream's inner-reset branch of the same
    `jax.tree.map` select."""
    cls = RE.row_logging_class(_FakeBase, lambda n, p: np.zeros(2), key="rows")
    _obs, nxt, _r, _d, infos = cls().step(None, {"t": 0}, {})
    assert nxt == {"t": 1}
    assert set(infos) == {"upstream_key", "rows"}


# ---------------------------------------------------------------- 2. the fold


class _FakeEnv:
    EGO = "robot"

    def __init__(self, succeed_when_rows_gt=None):
        self.thr = succeed_when_rows_gt
        self.asked = []

    def success(self, traj):
        traj = np.asarray(traj)
        self.asked.append(traj.shape)
        return bool(self.thr is not None and traj.max() > self.thr)


def _logs(T=6, n_eps=4, W=5, A=2):
    rows = np.arange(T * n_eps * W, dtype=float).reshape(T, n_eps, W) / 10.0
    rewards = np.ones((T, n_eps)) * 2.0
    pref = np.ones((T, n_eps)) * 0.5
    done = np.zeros((T, n_eps), dtype=bool)
    done[3, 1] = True                       # episode 1 ends at step index 3
    actions = np.zeros((T, n_eps, A)) + np.arange(T)[:, None, None]
    return rows, rewards, pref, done, actions


def test_the_fold_makes_k_task_reward_trajectories_cut_at_done():
    rows, rewards, pref, done, actions = _logs()
    env = _FakeEnv(succeed_when_rows_gt=10.0)
    out = J.trajectories_from_eval(rows=rows, rewards=rewards, pref=pref, done=done,
                                   actions=actions, env=env, k=3)
    assert len(out) == 3 and all(isinstance(t, Trajectory) for t in out)
    # episode 0: full horizon, task reward = 2.0 - 0.5 per step
    t0 = out[0]
    assert t0.length == 6 and np.asarray(t0.states).shape == (6, 5)
    assert t0.rewards == [1.5] * 6 and t0.ret == 9.0
    assert np.array_equal(np.asarray(t0.actions), actions[:, 0, :])
    # episode 1: cut at its first done, inclusive
    t1 = out[1]
    assert t1.length == 4 and np.asarray(t1.states).shape == (4, 5) and t1.ret == 6.0
    # states are that episode's rows, time-major
    assert np.array_equal(np.asarray(t1.states), rows[:4, 1, :])
    # success is the adapter's predicate on the banked rows, asked once per episode
    assert env.asked == [(6, 5), (4, 5), (6, 5)]
    # the predicate sees the CUT episode (episode 1 ends at step 4), not the full log
    assert [t.success for t in out] == [bool(rows[:n, e, :].max() > 10.0) for e, n in ((0, 6), (1, 4), (2, 6))]


def test_the_fold_takes_rewards_as_time_by_episode_and_no_actions():
    """`reward`/`done` are (T, n_eps) out of the eval scan (measured (1000, 32)); the fold treats that as the authority for T and n_eps and never
    re-orients the scalars. A 1-D reward series is one episode."""
    rows, rewards, pref, done, _ = _logs()
    out = J.trajectories_from_eval(rows=rows, rewards=rewards, pref=pref, done=done,
                                   actions=None, env=_FakeEnv(), k=2)
    assert [t.length for t in out] == [6, 4]
    assert all(t.actions is None for t in out)
    assert all(t.success is False for t in out)
    one = J.trajectories_from_eval(rows=rows[:, :1, :], rewards=rewards[:, 0], pref=pref[:, 0],
                                   done=done[:, 0], actions=None, env=_FakeEnv(), k=3)
    assert len(one) == 1 and one[0].length == 6


def test_the_fold_is_absent_not_an_error_when_no_rows_were_logged():
    rows, rewards, pref, done, actions = _logs()
    assert J.trajectories_from_eval(rows=None, rewards=rewards, pref=pref, done=done,
                                    actions=actions, env=_FakeEnv(), k=3) == []
    # a 2-D "rows" is not a stacked eval log
    assert J.trajectories_from_eval(rows=rows[:, 0, :], rewards=rewards, pref=pref, done=done,
                                    actions=actions, env=_FakeEnv(), k=3) == []


def test_k_never_exceeds_the_episodes_logged_and_never_drops_below_one():
    rows, rewards, pref, done, actions = _logs(n_eps=2)
    assert len(J.trajectories_from_eval(rows=rows, rewards=rewards, pref=pref, done=done,
                                        actions=actions, env=_FakeEnv(), k=9)) == 2
    assert J._rollouts_k(SimpleNamespace()) == 3
    assert J._rollouts_k(SimpleNamespace(_cfg={"evaluate.rollouts_per_candidate": 5})) == 5
    assert J._rollouts_k(SimpleNamespace(_cfg={"evaluate.rollouts_per_candidate": 0})) == 3


def test_a_failing_success_predicate_records_false_rather_than_raising():
    class _Raises(_FakeEnv):
        def success(self, traj):
            raise RuntimeError("no threshold on this task")
    rows, rewards, pref, done, actions = _logs()
    out = J.trajectories_from_eval(rows=rows, rewards=rewards, pref=pref, done=done,
                                   actions=actions, env=_Raises(), k=1)
    assert len(out) == 1 and out[0].success is False


# ---------------------------------------------------------------- 3. the ship


def test_build_train_result_ships_the_last_seeds_trajectories():
    cand = Candidate(cand_id="c0000", iteration=0, reward_code="def compute_reward(s, a, s2):\n    return 0.0\n")
    t_a = Trajectory(states=np.zeros((3, 2)), rewards=[0.0] * 3, length=3, ret=0.0)
    t_b = Trajectory(states=np.ones((3, 2)), rewards=[1.0] * 3, length=3, ret=3.0)
    o0 = B.SeedOutcome(seed=0, trajectories=[t_a])
    o1 = B.SeedOutcome(seed=1, trajectories=[t_b])
    res = B.build_train_result(cand, [o0, o1])
    assert res.trajectories == [t_b], "the LAST seed's rollouts ship, as policy_ref does"
    assert res.trained is True
    assert B.build_train_result(cand, [o0]).trajectories == [t_a]
    assert B.build_train_result(cand, []).trajectories == []


def test_a_seed_outcome_defaults_to_no_trajectories():
    o = B.SeedOutcome(seed=0)
    assert o.trajectories == []
    res = B.build_train_result(Candidate(cand_id="c0", iteration=0, reward_code="x = 1\n"), [o])
    assert res.trajectories == []


# ---------------------------------------------------------------- 1b. the log wrapper


def test_the_log_wrapper_binding_preserves_the_row_key_and_restores():
    """Upstream's LogWrapper(replace_info=True) keeps only PRESERVE_KEYS; the
    binding widens that set by the row key on the TRAINER MODULE's name and puts
    the original back, on the happy path and on an exception."""
    import sys
    import types

    made = []

    class _FakeLogWrapper:
        def __init__(self, env, replace_info=False, crossplay_info=False, preserve_keys=None):
            made.append(dict(env=env, replace_info=replace_info, preserve_keys=set(preserve_keys or ())))

    modname = "bird_test_fake_baselines"
    fake_mod = types.ModuleType(modname); fake_mod.PRESERVE_KEYS = {"preference_metrics", "preference_tracking"}
    _FakeLogWrapper.__module__ = modname
    sys.modules[modname] = fake_mod
    try:
        trainer = types.SimpleNamespace(LogWrapper=_FakeLogWrapper)
        with RE.log_wrapper_binding(trainer, {"bird_row"}) as cls:
            assert trainer.LogWrapper is cls and cls is not _FakeLogWrapper
            assert cls.bird_preserved_keys == frozenset({"bird_row"})
            trainer.LogWrapper("env", replace_info=True)
        assert trainer.LogWrapper is _FakeLogWrapper, "the original must be restored"
        assert made == [dict(env="env", replace_info=True,
                             preserve_keys={"preference_metrics", "preference_tracking", "bird_row"})]
        # an explicit preserve_keys is widened, not replaced
        with RE.log_wrapper_binding(trainer, {"bird_row"}):
            trainer.LogWrapper("env2", preserve_keys={"only_this"})
        assert made[-1]["preserve_keys"] == {"only_this", "bird_row"}
        # restored on an exception too
        try:
            with RE.log_wrapper_binding(trainer, {"bird_row"}):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert trainer.LogWrapper is _FakeLogWrapper
    finally:
        sys.modules.pop(modname, None)


def test_the_fold_reads_upstreams_swapped_axes_layout():
    """`_env_step` swaps axes 0 and 1 of every per-step info array, so the logged row
    arrives stacked as (T, W, n_eps). Measured: without this, each "rollout" is one row
    component across the 32 eval episodes.
    Width is disambiguated by the adapter's obs_dim (here 5, distinct from T=6 and
    n_eps=4); the result must equal the (T, n_eps, W) read."""
    rows, rewards, pref, done, actions = _logs()            # rows (T=6, n_eps=4, W=5)
    env = _FakeEnv(); env.obs_dim = 5
    ref = J.trajectories_from_eval(rows=rows, rewards=rewards, pref=pref, done=done,
                                   actions=actions, env=env, k=2)
    # the MEASURED layout: (T, W, n_eps) -- reward (1000, 32), row (1000, 88, 32), measured
    swapped = np.transpose(rows, (0, 2, 1))
    got = J.trajectories_from_eval(rows=swapped, rewards=rewards, pref=pref, done=done,
                                   actions=actions, env=env, k=2)
    assert [t.length for t in got] == [t.length for t in ref] == [6, 4]
    for a, b in zip(got, ref):
        assert np.array_equal(np.asarray(a.states), np.asarray(b.states))
        assert a.rewards == b.rewards
    # episodes-first (n_eps, T, W), as a robust caller might hand it, reads the same
    eps_first = np.transpose(rows, (1, 0, 2))
    got2 = J.trajectories_from_eval(rows=eps_first, rewards=rewards, pref=pref, done=done,
                                    actions=actions, env=env, k=2)
    assert np.array_equal(np.asarray(got2[1].states), np.asarray(ref[1].states))


def test_the_fold_refuses_a_layout_it_cannot_tell_apart():
    """T == n_eps == W leaves the axes ambiguous; absent beats a scrambled rollout."""
    T = n = W = 4
    rows = np.arange(T * n * W, dtype=float).reshape(T, n, W)
    rewards = np.ones((T, n)); pref = np.zeros((T, n)); done = np.zeros((T, n), dtype=bool)
    env = _FakeEnv(); env.obs_dim = W
    # square in every axis: no axis can be told apart -> ABSENT, never a guess
    assert J.trajectories_from_eval(rows=rows, rewards=rewards, pref=pref, done=done,
                                    actions=None, env=env, k=1) == []
    # a width the adapter does not declare is refused too
    env.obs_dim = 7
    assert J.trajectories_from_eval(rows=rows, rewards=rewards, pref=pref, done=done,
                                    actions=None, env=env, k=1) == []


def test_an_absent_fold_names_its_reason_on_the_log(caplog):
    import logging
    import numpy as np

    def _env5():
        env = _FakeEnv(); env.obs_dim = 5
        return env
    from bird.components._ippo_train import trajectories_from_eval
    with caplog.at_level(logging.WARNING, logger="bird.components._ippo_train"):
        out = trajectories_from_eval(rows=np.zeros((5, 5, 5)), rewards=np.zeros((5, 5)),
                                     pref=None, done=None, actions=None, env=_env5(), k=1)
    assert out == []
    assert any("eval rollouts absent" in r.getMessage() and "not exactly one" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]
