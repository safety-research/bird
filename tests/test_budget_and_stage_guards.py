"""Guards across stage 2, stage 3, stage 5 and the budget.

Five defects, one file, because each is a two-line mistake whose regression
test needs the same fixtures (a pendulum context, a mock-LLM tester config):

* **No silent PPO** -- a `training._sb3_algo_and_hyper` that did
  `{...}.get(name, PPO)` would train PPO for `train.algorithm: none` (l2r) or
  `q_learning` (singh_orp) under a profile that pins `train.backend: sb3`,
  under a seed row that says otherwise. The map is strict and the coherence
  rule covers every algorithm the map lacks.
* **Env-space actions in the exported slice** -- for SB3's off-policy
  collector `rb.actions` are the [-1, 1]-SCALED actions
  (`off_policy_algorithm._sample_action`: `buffer_action = scaled_action`),
  while the candidate reward is written for env-space actions. Copying them
  verbatim is identity on MT10 (+/-1) and wrong on pendulum (+/-2) and every
  Humanoid/Pusher-class task. The slice holds env-space actions -- the same
  thing `_QLearner.seen` means -- and the relabel scales them back for the
  learner through ONE helper (`_sb3_stored_actions`) that LaRes's prefill into
  SAC's PRIMARY buffer shares, so the two fills cannot drift.
* **The DR clip reads the attribute the adapters define** --
  `verification.fail_degrade` must read `dr_parameters` (the name RAPP's
  `phases._dr_ranges` reads), not `env.dr_bounds`, which no adapter here
  defines; otherwise the documented clip-into-bounds never runs and
  `clipped_dr_keys` is always `[]`.
* **One human query per comparison** -- if both
  `preferences.comparator_human` and `phases._ScriptedHumanOracle.compare`
  charged `budget.record_human()`, one comparison would count twice (a tester
  run would read 512 for 240 comparisons + 32 feedback queries). One charge,
  at the oracle -- the place that knows a person was asked.
* **No sentinel round winners** -- a round in which every candidate was
  screened out keeps the candidates `valid` with no train error, so
  `run_iteration`'s all-failed guard passes, every report carries
  `select.failure_value`, and an `argmax_fitness` that tie-broke among the
  sentinels would elect a never-trained program as round winner. The rule
  elects nobody when no fitness in the pool is a measurement.

No test here needs metaworld, mujoco or an API key; the sb3 tests skip where
stable_baselines3 is absent.
"""
from __future__ import annotations

import importlib.util
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import REPO  # noqa: F401  (path setup)
from bird import config as C
from bird import schema as SCH
from bird import registry
from bird.budget import Budget
from bird.components import phases as P
from bird.components import preferences as PR
from bird.components import selection as S
from bird.components import training as T
from bird.components import verification as V
from bird.config import ConfigError, load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult

try:
    import stable_baselines3  # noqa: F401

    HAVE_SB3 = True
except ImportError:  # pragma: no cover - depends on the machine
    HAVE_SB3 = False

needs_sb3 = pytest.mark.skipif(not HAVE_SB3, reason="needs stable_baselines3 + gymnasium")

ZERO_REWARD = "def compute_reward(state, action):\n    return 0.0, {}\n"


def _ctx(config: str, profile: str, **overrides) -> Context:
    registry.load_all()
    base = {"output.tracker": "none", "llm.generator.provider": "mock",
            "llm.evaluator.provider": "mock"}
    base.update(overrides)
    cfg = load(config, profile=profile, overrides=base)
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(0))
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    ctx.tracker = SimpleNamespace(log_iteration=lambda *a, **k: None,
                                  log_media=lambda *a, **k: None)
    return ctx


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry_stage_guards", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==========================================================================
# no silent PPO
# ==========================================================================


@pytest.mark.parametrize("config,algorithm,own_backend", [
    ("l2r", "none", "none"),
    ("singh_orp", "q_learning", "tabular"),
])
@pytest.mark.parametrize("profile", ["dev", "full"])
def test_sb3_refuses_an_algorithm_it_has_no_learner_for(config, algorithm, own_backend, profile):
    """`dev` and `full` pin `train.backend: sb3` and a profile outranks the
    method, so l2r and singh_orp resolve to sb3 and would train PPO under a
    seed row citing `none` / `q_learning`. The pairing is refused at load,
    naming both keys, and the method's own backend is the way out."""
    with pytest.raises(ConfigError) as exc:
        load(config, profile=profile)
    msg = str(exc.value)
    assert f"train.algorithm={algorithm}" in msg and "sb3" in msg, msg
    assert f"train.backend: {own_backend}" in msg, msg
    # The way out the message names has to work.
    cfg = load(config, profile=profile, overrides={"train.backend": own_backend})
    assert cfg["train.algorithm"] == algorithm and cfg["train.backend"] == own_backend


def test_the_coherence_rule_names_every_algorithm_the_sb3_map_lacks():
    """The rule is the complement of the map, not a list of two names."""
    allowed = set(C.SB3_ALGORITHMS)
    every = set(SCH.SCHEMA["train.algorithm"].enum)
    assert allowed < every
    for alg in sorted(every - allowed):
        with pytest.raises(ConfigError, match=f"train.algorithm={alg}"):
            load("eureka", profile="dev", overrides={"train.algorithm": alg})
    for alg in sorted(allowed):
        load("eureka", profile="dev", overrides={"train.algorithm": alg})


@needs_sb3
def test_the_sb3_algo_map_is_strict():
    """Belt to the coherence rule's braces: a config that reached the backend
    through some path validate() did not see must still not train PPO."""
    class _Cfg(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    for alg in C.SB3_ALGORITHMS:
        algo, _hyper, _anchor = T._sb3_algo_and_hyper(_Cfg({"train.algorithm": alg}))
        assert algo is not None
    with pytest.raises(ValueError) as exc:
        T._sb3_algo_and_hyper(_Cfg({"train.algorithm": "q_learning"}))
    assert "q_learning" in str(exc.value) and "sb3" in str(exc.value)


# ==========================================================================
# env-space actions in the exported slice
# ==========================================================================


def _logging_gym(env, taken: list):
    """gym.Env over the adapter's spaces that records every action it is
    handed -- the ground truth an exported transition has to match."""
    import gymnasium as gym
    from gymnasium import spaces

    class _Shim(gym.Env):
        observation_space = spaces.Box(np.asarray(env.obs_low, np.float32),
                                       np.asarray(env.obs_high, np.float32))
        action_space = spaces.Box(np.asarray(env.action_low, np.float32),
                                  np.asarray(env.action_high, np.float32))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, a):
            taken.append(np.asarray(a, dtype=np.float32).copy())
            return self.observation_space.sample(), 0.0, False, False, {}

    return _Shim()


@needs_sb3
def test_the_exported_replay_slice_holds_the_actions_the_env_was_given():
    """pendulum's torque is [-2, 2]. SB3 stores the [-1, 1]-scaled action and
    hands the env the unscaled one; the slice has to hold the latter, because
    that is what the candidate reward is a function of."""
    from stable_baselines3 import SAC

    ctx = _ctx("eureka", "dev", **{"problem.env_id": "pendulum", "train.algorithm": "sac"})
    assert float(ctx.env.action_high[0]) == 2.0, "the test needs a non-identity scaling"
    taken: list = []
    model = SAC("MlpPolicy", _logging_gym(ctx.env, taken), seed=0, device="cpu",
                buffer_size=512, learning_starts=64, train_freq=8, gradient_steps=1,
                policy_kwargs={"net_arch": [8, 8]}, verbose=0)
    model.learn(160)

    sl = T._sb3_export_replay(model, 1000)
    assert sl is not None and len(sl) == len(taken)
    np.testing.assert_allclose(sl.act, np.stack(taken), rtol=1e-5, atol=1e-5)
    # Not merely "in [low, high]": warm-up samples the space uniformly, so a
    # slice that was still scaled would sit inside [-1, 1] and fail here.
    assert float(np.abs(sl.act).max()) > 1.0 + 1e-3


@needs_sb3
def test_the_secondary_buffer_relabels_on_the_env_space_action_and_trains_on_the_scaled_one():
    """Two halves, both required: the reward sees the action the env took,
    and the learner's buffer holds what SAC's critic expects (scaled)."""
    from stable_baselines3 import SAC

    ctx = _ctx("eureka", "dev", **{"problem.env_id": "pendulum", "train.algorithm": "sac"})
    taken: list = []
    model = SAC("MlpPolicy", _logging_gym(ctx.env, taken), seed=0, device="cpu",
                buffer_size=512, learning_starts=64, train_freq=8, gradient_steps=1,
                policy_kwargs={"net_arch": [8, 8]}, verbose=0)
    model.learn(160)
    sl = T._sb3_export_replay(model, 1000)

    reward = T.compile_reward("def compute_reward(state, action):\n"
                              "    a = float(action[0])\n"
                              "    return a * a, {'torque_sq': a * a}\n")
    mix = T._sb3_secondary_buffer(sl, 0.2, model, ctx.env, reward, "none", True)
    assert mix is not None and mix.buffer_size == len(sl)
    env_actions = np.stack(taken)[:, 0]
    np.testing.assert_allclose(mix.rewards[:, 0], env_actions ** 2, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(mix.actions[:, 0, 0],
                               model.policy.scale_action(np.stack(taken))[:, 0],
                               rtol=1e-5, atol=1e-5)
    assert float(np.abs(mix.actions).max()) <= 1.0 + 1e-5


@needs_sb3
def test_the_prefilled_primary_buffer_relabels_on_the_env_space_action_and_stores_the_scaled_one():
    """LaRes's prefill writes the pool into SAC's PRIMARY buffer, and the
    pool is `_sb3_export_replay` slices, which are env-space. Both halves
    of the secondary-buffer contract hold here too: the reward sees the action
    the env took, and `rb.actions` holds what SAC's critic expects. Red when
    the prefill copies `act` verbatim: on pendulum (+/-2) the buffer then holds
    torques up to 2 under a [-1, 1] convention."""
    from stable_baselines3 import SAC

    ctx = _ctx("eureka", "dev", **{"problem.env_id": "pendulum", "train.algorithm": "sac"})
    taken: list = []
    model = SAC("MlpPolicy", _logging_gym(ctx.env, taken), seed=0, device="cpu",
                buffer_size=512, learning_starts=64, train_freq=8, gradient_steps=1,
                policy_kwargs={"net_arch": [8, 8]}, verbose=0)
    model.learn(160)
    pool = T._sb3_export_replay(model, 1000)
    assert pool is not None and float(np.abs(pool.act).max()) > 1.0 + 1e-3

    fresh = SAC("MlpPolicy", _logging_gym(ctx.env, []), seed=1, device="cpu",
                buffer_size=512, learning_starts=64, policy_kwargs={"net_arch": [8, 8]},
                verbose=0)
    reward = T.compile_reward("def compute_reward(state, action):\n"
                              "    a = float(action[0])\n"
                              "    return a * a, {'torque_sq': a * a}\n")
    n = T._sb3_prefill_replay(fresh, pool, ctx.env, reward, "none", True)
    assert n == len(pool) == len(taken)
    rb = fresh.replay_buffer
    env_actions = np.stack(taken)
    np.testing.assert_allclose(rb.rewards[:n, 0], env_actions[:, 0] ** 2, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(rb.actions[:n, 0, 0],
                               fresh.policy.scale_action(env_actions)[:, 0],
                               rtol=1e-5, atol=1e-5)
    assert float(np.abs(rb.actions[:n]).max()) <= 1.0 + 1e-5
    # The two fills agree row for row -- the point of one helper.
    mix = T._sb3_secondary_buffer(pool, 0.2, fresh, ctx.env, reward, "none", True)
    np.testing.assert_allclose(mix.actions[:, 0, 0], rb.actions[:n, 0, 0], rtol=1e-6, atol=1e-6)


@needs_sb3
def test_the_sb3_learner_map_and_the_config_constant_cannot_disagree_silently(monkeypatch):
    """The coherence rule reads `config.SB3_ALGORITHMS`; the backend's map is
    what dispatches. A raise, not an `assert`: `python -O` strips asserts, and
    a map that grew an algorithm the rule still refuses (or the reverse) would
    otherwise run under whichever copy happened to be read."""
    class _Cfg(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    monkeypatch.setattr(T, "SB3_ALGORITHMS", ("ppo", "sac", "td3"))
    with pytest.raises(RuntimeError) as exc:
        T._sb3_algo_and_hyper(_Cfg({"train.algorithm": "ppo"}))
    assert "SB3_ALGORITHMS" in str(exc.value) and "qr_sac" in str(exc.value)


# ==========================================================================
# the DR clip reads the attribute the adapters define
# ==========================================================================


def test_degrade_clips_a_generated_dr_range_into_the_envs_declared_bounds():
    ctx = _ctx("dreureka", "tester", **{"problem.env_id": "pendulum"})
    bounds = ctx.env.dr_parameters
    assert "mass" in bounds and not hasattr(ctx.env, "dr_bounds")
    lo, hi = bounds["mass"]
    cand = Candidate("c0000", 0, ZERO_REWARD,
                     dr_config={"mass": [lo - 1.0, hi + 5.0], "gravity": list(bounds["gravity"])})

    out = V.fail_degrade(ctx, RunState(), cand, "dr range out of bounds", 1)

    assert out is not None and out.valid
    rec = [r for r in out.verify_records if r.get("policy") == "degrade"][-1]
    assert rec["clipped_dr_keys"] == ["mass"], rec
    assert out.dr_config["mass"] == [lo, hi]
    assert out.dr_config["gravity"] == list(bounds["gravity"])


def test_degrade_still_reads_dr_bounds_on_an_external_adapter():
    """The documented fallback, for an adapter this repo does not own."""
    ctx = _ctx("dreureka", "tester", **{"problem.env_id": "pendulum"})
    ctx.env = SimpleNamespace(dr_bounds={"friction": (0.0, 1.0)})
    cand = Candidate("c0000", 0, ZERO_REWARD, dr_config={"friction": [-1.0, 0.5]})
    out = V.fail_degrade(ctx, RunState(), cand, "reason", 1)
    rec = [r for r in out.verify_records if r.get("policy") == "degrade"][-1]
    assert rec["clipped_dr_keys"] == ["friction"] and out.dr_config["friction"] == [0.0, 0.5]


# ==========================================================================
# one human query per comparison
# ==========================================================================


def _report(cid: str, gt: float) -> CandidateReport:
    cand = Candidate(cid, 0, ZERO_REWARD)
    res = TrainResult(cid, cand, gt_reward_curve=[gt])
    return CandidateReport(cid, cand, res, fitness=gt, fitness_source="test")


def test_a_human_comparison_is_charged_once():
    ctx = _ctx("gt", "tester", **{"evaluate.human.mode": "preference_labels"})
    ctx.human = P.build_human_oracle(ctx)
    assert getattr(ctx.human, "scripted", False), "the scripted oracle charges itself"
    left, right = _report("c0000", 1.0), _report("c0001", 0.5)

    assert PR.comparator_human(ctx, RunState(), left, right) == 1
    assert ctx.budget.human_queries == 1, ctx.budget.human_queries
    PR.comparator_human(ctx, RunState(), right, left)
    assert ctx.budget.human_queries == 2

    # The other queries the oracle answers keep their one charge each, so the
    # counter reads comparisons + feedback queries and nothing else.
    ctx.human.feedback(left)
    assert ctx.budget.human_queries == 3


def test_a_comparison_no_person_answered_is_not_a_human_query():
    """`evaluate.human.mode: none` yields an oracle that refuses every query
    and the comparator degrades to the offline rule -- nobody was asked."""
    ctx = _ctx("gt", "tester")
    ctx.human = P.build_human_oracle(ctx)
    left, right = _report("c0000", 1.0), _report("c0001", 0.5)
    PR.comparator_human(ctx, RunState(), left, right)
    assert ctx.budget.human_queries == 0


# ==========================================================================
# no -10000 round winners
# ==========================================================================


def _screened(cid: str, sentinel: float) -> CandidateReport:
    """What §2-§4 hand on for a candidate a screen rejected: still `valid`,
    never trained, fitness at `select.failure_value`."""
    cand = Candidate(cid, 0, ZERO_REWARD)
    cand.screened_out, cand.failure, cand.failure_kind = True, "demo_margin: expert paid no more than random", "screened"
    cand.verify_records.append({"phase": "screen", "ok": False, "detail": cand.failure})
    res = TrainResult(cid, cand, trained=False, skip_reason=cand.failure)
    rep = CandidateReport(cid, cand, res, fitness=sentinel, fitness_source="ground_truth_metric")
    rep.meta["fitness_note"] = cand.failure
    return rep


def _trained(cid: str, fitness: float) -> CandidateReport:
    cand = Candidate(cid, 0, ZERO_REWARD)
    res = TrainResult(cid, cand, trained=True)
    return CandidateReport(cid, cand, res, fitness=fitness, fitness_source="ground_truth_metric")


def test_argmax_elects_nobody_when_no_fitness_is_a_measurement():
    ctx = _ctx("eureka", "tester")
    sentinel = float(ctx.cfg["select.failure_value"])
    reports = [_screened("c0018", sentinel), _screened("c0019", sentinel)]
    sel = S.rule_argmax_fitness(ctx, RunState(), reports)
    assert sel.winners == [] and not sel.tie_broken, sel
    assert "measured" in sel.notes, sel.notes
    # The sentinel's §5 contract -- loses every comparison, stays recorded --
    # is intact: the reports are still in the selection, as losers.
    assert {r.cand_id for r in sel.losers} == {"c0018", "c0019"}


def test_argmax_is_unchanged_whenever_one_fitness_is_a_measurement():
    """Scope: a sentinel beside a measurement still loses BY VALUE, the way it
    always does -- including RF-Agent's published `failure_value: 0.0`, where
    a crash ties a zero-success training and the tie breaks by the usual rule
    (§5). Only the all-sentinel round is special."""
    ctx = _ctx("eureka", "tester")
    sentinel = float(ctx.cfg["select.failure_value"])
    sel = S.rule_argmax_fitness(ctx, RunState(),
                                [_screened("c0000", sentinel), _trained("c0001", 0.3)])
    assert [w.cand_id for w in sel.winners] == ["c0001"]
    zero = S.rule_argmax_fitness(ctx, RunState(), [_screened("c0000", 0.0), _trained("c0001", 0.0)])
    assert [w.cand_id for w in zero.winners] == ["c0000"] and zero.tie_broken


def test_an_all_screened_round_writes_no_select_event_with_a_sentinel_winner(tmp_path, monkeypatch):
    """End to end on the tester tier: every candidate screened out, one round.
    The `select` journal event must not name a winner, and stage 6 must not
    have adopted a never-trained program as parent or best."""
    def reject_all(ctx, state, candidates):
        for c in candidates:
            if c.trainable:
                c.screened_out, c.failure_kind = True, "screened"
                c.failure = "test screen: rejects everything"
                c.verify_records.append({"phase": "screen", "ok": False, "detail": c.failure})
        return candidates

    registry.load_all()
    monkeypatch.setitem(registry._REGISTRY, ("screen", "none"), reject_all)
    # `verify.enabled: true`: with stage 2 off and `quality_screen: none`,
    # `bird.verify` returns before consulting the screen registry at all.
    cfg = load("eureka", profile="tester", overrides={"loop.n_iterations": 1,
                                                       "generate.n_candidates": 3,
                                                       "verify.enabled": True})
    _entry().run(cfg, out_root=str(tmp_path))
    run = next(p for p in tmp_path.iterdir() if p.is_dir())
    events = [json.loads(l) for l in (run / "journal.jsonl").read_text().splitlines() if l.strip()]

    screened = [e for e in events if e.get("stage") == "verify" and e.get("kind") == "screened"]
    assert len(screened) == 3, "the screen did not reject every candidate"
    selects = [e for e in events if e.get("stage") == "select"]
    assert selects, "stage 5 ran"
    assert all(e["winners"] == [] for e in selects), selects
    result = json.loads((run / "result.json").read_text())
    assert result.get("returned_cand_id") is None, result
