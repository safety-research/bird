"""`train.init` -- two keys that are easy to declare everywhere and honour nowhere.

WHAT CAN GO WRONG, and why a large suite stays green while it does. `train.init`
has three values (`configs/_default.yaml`, `bird/schema.py` §3) and several
configs set a non-default one: `warm_start_from_best` is RDA's +Ckpt (§4.2,
Fig. 7a) and `secondary_replay_buffer` is Gran Turismo's shared buffer (§4,
App. E Table 3). Either goes inert if one of two independent places breaks:

  1. `training._training_init` reads `state.policy_ref` / `state.replay_ref`.
     `RunState` declares them, `CARRY_SLOTS` names them, `loop.carry` gates
     them, `checkpoint.py` persists them -- and if **nothing assigns
     either**, every call returns `(None, [], 0.0)`, on every backend.
  2. A `_sb3_run` that writes neither store and reads neither key leaves
     nothing to carry on `train.backend: sb3` even with (1) in place -- and
     that is every config run under the `dev` or `full` profile, i.e. every
     expensive run.

Two failure points, one symptom, and the symptom is silence: the search
completes, the dashboard fills, the artifact records `init:
warm_start_from_best`, and what runs is `from_scratch`. No exception, no
warning, no missing file. That is the same class as `_merge_child`'s
dropped-store note, and the reason this file exists is that otherwise the suite
cannot tell the method from its own ablation.

THE SHAPE OF THE TESTS. Three layers, because the failure has three layers:

  * `test_every_backend_under_contract_populates_both_stores` -- the WRITE
    side, parametrised over BACKEND so a fifth backend cannot reintroduce the
    hole by simply not implementing it.
  * `test_no_config_declares_train_init_on_an_untested_backend` -- keeps that
    parametrisation honest. A new tier on a new backend fails here rather than
    silently escaping coverage.
  * `test_the_incumbents_refs_reach_the_next_iteration` and the end-to-end
    pair -- the READ side, which is where the assignment lives.
"""

import glob
import io
import os

import numpy as np
import pytest

from conftest import REPO, load_gt_published  # noqa: F401  (path setup)
from bird import config as C
from bird import registry
from bird.budget import Budget
from bird.checkpoint import _ArraySpill, _encode_replay, decode
from bird.components import training as T
from bird.components.update import _carry_inner_loop_refs
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult

#: Not in the tester-tier smoke suite (heavyweight execution: runs a search per init policy).
#: Deselected in CI by `-m "not slow"`; run it with `-m slow`. See pyproject.toml.
pytestmark = pytest.mark.slow

try:  # the sb3 rows are the point of this file, but the dep is optional
    import stable_baselines3  # noqa: F401

    HAVE_SB3 = True
except ImportError:  # pragma: no cover - depends on the machine
    HAVE_SB3 = False

needs_sb3 = pytest.mark.skipif(not HAVE_SB3, reason="train.backend: sb3 needs sb3+gymnasium")


# A reward good enough that a warm start is visibly worth something. Pendulum
# with SAC reaches the upright in ~15-20k steps under this and plainly does not
# in 1,500 -- which is the gap the warm-start test measures.
UPRIGHT = """
def compute_reward(state, action):
    import numpy as np
    th = np.arctan2(state[1], state[0])
    return -(th ** 2) - 0.1 * state[2] ** 2, {"upright": -(th ** 2)}
"""

#: On a tabular env the reward has to be a function of the discrete state, and
#: `toy_gridworld`'s ground truth is "reach the goal".
ANY_REWARD = """
def compute_reward(state, action):
    import numpy as np
    return -float(np.sum(np.abs(np.asarray(state, dtype=float)))), {"c": 1.0}
"""


def _ctx(env_id, **overrides):
    base = {"seed": 0, "problem.env_id": env_id, "output.tracker": "none",
            "llm.generator.provider": "mock", "llm.evaluator.provider": "mock",
            "evaluate.rollouts_per_candidate": 1}
    base.update(overrides)
    cfg = load("rda", overrides=base, profile="dev")
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", env_id)(ctx)
    return ctx


def _train(ctx, cand_id, state=None, code=None):
    backend = registry.get("train_backend", ctx.cfg["train.backend"])
    cand = Candidate(cand_id=cand_id, reward_code=code or ANY_REWARD, iteration=0)
    return backend(ctx, state if state is not None else RunState(), cand, 1)


@pytest.fixture(autouse=True)
def _clean_stores():
    """The stores are module globals; a leak between tests would let one test
    satisfy another's assertion, which is the failure mode they exist to catch.

    `reset_policy_stores` rather than clearing two of them by hand: it also
    empties `_PINNED_REFS`, and a pin left behind by an earlier test's run
    (a `replay:c0000` its state carried) exempts that ref from eviction and
    fails the byte-cap tests below depending on test order."""
    T.reset_policy_stores()
    yield
    T.reset_policy_stores()


# ==========================================================================
# The write side, parametrised over backend
# ==========================================================================

#: (backend, env_id, train.env_steps, why this env).
#:
#: The env is chosen per backend so the learner under it actually HAS the
#: mechanism: `mock` on a continuous env selects `_CEMLearner`, which keeps no
#: transitions at all, so a replay assertion there would pass on an empty list
#: and say nothing. `toy_gridworld` declares `exact_states`, which is what
#: routes `mock` to `_QLearner` -- the tabular learner that both snapshots a
#: Q-table and fills `seen`.
_STORE_CONTRACT = [
    ("mock", "toy_gridworld", 2000),
    ("tabular", "toy_hungry_thirsty", 2000),
    pytest.param(("sb3", "pendulum", 1200), marks=needs_sb3),
]


def _backend_of(case):
    return (case.values[0] if isinstance(case, type(pytest.param(0))) else case)[0]


@pytest.mark.parametrize("case", _STORE_CONTRACT, ids=[_backend_of(c) for c in _STORE_CONTRACT])
def test_every_backend_under_contract_populates_both_stores(case):
    """A backend that does not write the stores runs the ablation, silently.

    A backend that leaves `policy_ref` and `replay_ref` `None` on every
    candidate gives `train.init` nothing to dereference. Parametrised over
    backend because the hole is per-backend -- `_run_backend` serves three of
    the four, and the fourth (sb3) is the one every real config uses.
    """
    backend, env_id, steps = case
    ctx = _ctx(env_id, **{"train.backend": backend, "train.env_steps": steps,
                          "train.algorithm": "sac" if backend == "sb3" else "q_learning",
                          # `_wants_replay`: a config that does not carry the buffer
                          # can never dereference it, and pays 31 MB per candidate on
                          # sb3 for the privilege. Declaring the carry here is what
                          # puts this test inside the contract rather than outside it.
                          "loop.carry": ["best_reward", "subtask_list",
                                         "policy_checkpoint", "replay_buffer"],
                          "train.init": "secondary_replay_buffer"})
    res = _train(ctx, "c0000", code=UPRIGHT if env_id == "pendulum" else ANY_REWARD)

    assert res.policy_ref, (
        f"{backend}: train.init=warm_start_from_best has nothing to warm start from -- "
        "this backend never writes _POLICY_STORE")
    assert T._POLICY_STORE.get(res.policy_ref) is not None, \
        f"{backend}: policy_ref names a key that is not in the store"

    assert res.replay_ref, (
        f"{backend}: train.init=secondary_replay_buffer has no buffer to mix -- "
        "this backend never writes _REPLAY_STORE")
    buf = T._REPLAY_STORE.get(res.replay_ref)
    assert buf is not None and len(buf) > 0, (
        f"{backend}: replay_ref resolves to an EMPTY buffer, which is the ablation "
        "wearing the method's name")

    # The two representations must be interchangeable at the read side, which
    # is the only reason one store can hold both.
    obs, act, nxt, done = T._replay_columns(buf)
    assert obs.shape[0] == len(buf) and nxt.shape[0] == len(buf)
    assert done.dtype == bool
    tail = T._replay_tail(buf, 5)
    assert len(tail) == min(5, len(buf))
    assert len(tail[0]) == 4, "a transition is (obs, action, next_obs, done) on every backend"


def test_a_config_that_cannot_carry_a_buffer_does_not_pay_for_one():
    """`_wants_replay`. The mirror image of the failure: storing what nothing reads.

    `RunState.apply_carry` nulls `state.replay_ref` unless `loop.carry` names
    `replay_buffer`, and `_training_init` reads nowhere else -- so on a config
    without the carry the stored buffer is unreachable BY CONSTRUCTION. On sb3
    at a Meta-World shape that is 31 MB per candidate (measured, mt10 obs 39,
    `secondary_buffer.size: 100000`) for every sb3 config that does not carry
    the buffer. The policy is deliberately NOT gated the same way: DrEureka's RAPP
    sweep reads `state.best.result.policy_ref` directly (`phases._policy_ref`,
    `rapp.policy: incumbent_best`) without `policy_checkpoint` in the carry.
    """
    # `train.init` is pinned to `from_scratch` and that is not incidental. This
    # test needs `policy_checkpoint` ABSENT from the carry -- that absence is the
    # whole of its second claim -- and `rda` ships `train.init:
    # warm_start_from_best`, which `_check_coherence` refuses without that
    # carry slot (rightly: warm-starting off a policy the boundary nulls
    # silently trains from scratch). The two are only in tension
    # because this test borrows `rda` as a vehicle for a question about what gets
    # STORED, which has nothing to do with where training starts. Pinning the
    # init mode states that independence instead of relying on it.
    ctx = _ctx("toy_gridworld", **{"train.backend": "mock", "train.env_steps": 2000,
                                   "train.algorithm": "q_learning",
                                   "train.init": "from_scratch",
                                   "loop.carry": ["best_reward", "subtask_list"]})
    res = _train(ctx, "c0000")
    assert res.replay_ref is None, "stored a buffer no config key can ever reach"
    assert res.policy_ref, "the POLICY is still stored -- rapp.policy reads it uncarried"
    assert not T._REPLAY_STORE


def test_no_config_declares_train_init_on_an_untested_backend():
    """Keeps the parametrisation above honest.

    Coverage that is a hand-written list rots the moment a tier is added: a new
    tier that inherits `train.init` from published points would escape it
    unnoticed. If a config ever declares `train.init` on a backend
    `_STORE_CONTRACT` does not cover, this fails HERE, where the fix is one
    line, rather than in a long run hours in.
    """
    covered = {_backend_of(c) for c in _STORE_CONTRACT}
    offenders = []
    for path in sorted(glob.glob(str(C.CONFIG_ROOT / "**" / "*.yaml"), recursive=True)):
        if os.path.basename(path).startswith("_"):
            continue
        cfg = C.load(path)
        if cfg.get("train.init", "from_scratch") == "from_scratch":
            continue
        if cfg.get("train.backend") not in covered:
            offenders.append(f"{path}: train.init={cfg.get('train.init')} on "
                             f"train.backend={cfg.get('train.backend')}")
    assert not offenders, (
        "these configs declare train.init on a backend no test proves populates the "
        "stores; add the backend to _STORE_CONTRACT:\n  " + "\n  ".join(offenders))


def test_at_least_one_config_in_the_corpus_reaches_each_train_init_value():
    """If the shipped configs stopped setting these, every test above would pass
    vacuously.

    The shipped configs are every file under `configs/`, the hill-climb
    configurations included; the two values checked here come from published
    configs.
    """
    seen = set()
    for path in sorted(glob.glob(str(C.CONFIG_ROOT / "**" / "*.yaml"), recursive=True)):
        if os.path.basename(path).startswith("_"):
            continue
        seen.add(C.load(path).get("train.init", "from_scratch"))
    assert {"warm_start_from_best", "secondary_replay_buffer"} <= seen


# ==========================================================================
# The read side -- the assignment that carries the incumbent
# ==========================================================================


def test_the_incumbents_refs_reach_the_next_iteration():
    """`state.policy_ref` must be written when the incumbent changes.

    Not an end-to-end test on purpose: it pins the ONE assignment, so a future
    refactor that moves `_update_global_best` cannot quietly drop it
    while every integration test keeps passing on a slower learning curve.
    """
    state = RunState()
    cand = Candidate(cand_id="c0001", reward_code=ANY_REWARD, iteration=0)
    result = TrainResult(cand_id="c0001", candidate=cand, policy_ref="policy:c0001",
                         replay_ref="replay:c0001")
    state.best = CandidateReport(cand_id="c0001", candidate=cand, result=result,
                                 fitness=1.0)

    _carry_inner_loop_refs(state)
    assert state.policy_ref == "policy:c0001"
    assert state.replay_ref == "replay:c0001"

    # And `loop.carry` still decides whether they survive the boundary --
    # writing them here must not smuggle them past a config that omits them.
    state.apply_carry(["best_reward"])
    assert state.policy_ref is None and state.replay_ref is None


def test_training_init_dereferences_what_the_incumbent_stored():
    """`_training_init` -> the store, on both keys, with the cap applied."""
    T._store(T._POLICY_STORE, "policy:cW", np.arange(6.0).reshape(2, 3))
    T._store(T._REPLAY_STORE, "replay:cW",
             [(np.zeros(2), i, np.ones(2), False) for i in range(50)])

    state = RunState()
    state.policy_ref, state.replay_ref = "policy:cW", "replay:cW"

    cfg = load("rda", overrides={"train.init": "warm_start_from_best"},
               profile="dev")
    plan = T._training_init(None, state, cfg)
    ref, sec, ratio = plan.ref, plan.secondary, plan.ratio
    assert ref == "policy:cW" and sec == [] and ratio == 0.0
    assert (plan.source, plan.from_cand_id) == ("best", "cW")

    # `warm_start_from_parent`: the lineage parent's policy when it is stored,
    # the incumbent's when it is not -- and `source` names which. A fallback
    # that ran as the other arm while recording `parent` would be a plausible
    # value in every artifact, which is why this is asserted here.
    T._store(T._POLICY_STORE, "policy:cP", np.arange(6.0).reshape(2, 3) + 1)
    T._POLICY_CODE["cP"] = (0, ANY_REWARD)  # trained in round 0
    cfg = load("rda", overrides={"train.init": "warm_start_from_parent"},
               profile="dev")
    child = Candidate(cand_id="cC", reward_code=ANY_REWARD, iteration=1, parent_id="cP")
    plan = T._training_init(None, state, cfg, None, child)
    assert (plan.ref, plan.source, plan.from_cand_id) == ("policy:cP", "parent", "cP")
    # A SAME-ROUND parent (`sequential_conditioned`: sample i's parent is sample
    # i-1 of this round) is not a donor -- a forked worker cannot see its policy,
    # so taking it would make `parallel != sequential`. Falls to `best`.
    sibling_child = Candidate(cand_id="cD", reward_code=ANY_REWARD, iteration=0, parent_id="cP")
    plan = T._training_init(None, state, cfg, None, sibling_child)
    assert (plan.ref, plan.source, plan.from_cand_id) == ("policy:cW", "best", "cW")
    orphan = Candidate(cand_id="cO", reward_code=ANY_REWARD, iteration=1, parent_id="cGone")
    plan = T._training_init(None, state, cfg, None, orphan)
    assert (plan.ref, plan.source, plan.from_cand_id) == ("policy:cW", "best", "cW")
    root = Candidate(cand_id="cR", reward_code=ANY_REWARD, iteration=0)
    state.policy_ref = None
    plan = T._training_init(None, state, cfg, None, root)
    assert (plan.ref, plan.source, plan.from_cand_id) == (None, "scratch", None)
    state.policy_ref = "policy:cW"

    # `warm_start_from_similar`: the donor is the EARLIER-iteration stored
    # policy whose reward is structurally closest, not the incumbent, and a
    # tie goes to the lowest candidate index. Round 0 falls back to scratch.
    T._POLICY_CODE.clear()
    far = "def compute_reward(obs, act):\n    return float((obs ** 2).sum()) * -3.0\n"
    T._POLICY_CODE["cW"] = (0, far)
    T._POLICY_CODE["cP"] = (0, ANY_REWARD)
    cfg = load("rda", overrides={"train.init": "warm_start_from_similar"},
               profile="dev")
    child = Candidate(cand_id="cC", reward_code=ANY_REWARD, iteration=1)
    plan = T._training_init(None, state, cfg, None, child)
    assert (plan.ref, plan.source, plan.from_cand_id) == ("policy:cP", "similar", "cP")
    assert plan.similarity == 1.0
    T._store(T._POLICY_STORE, "policy:cA", np.arange(6.0).reshape(2, 3) + 2)
    T._POLICY_CODE["cA"] = (0, ANY_REWARD)  # an identical, lower-index donor wins the tie
    plan = T._training_init(None, state, cfg, None, child)
    assert (plan.from_cand_id, plan.similarity) == ("cA", 1.0)
    same_round = Candidate(cand_id="cS", reward_code=ANY_REWARD, iteration=0)
    state.policy_ref = None
    plan = T._training_init(None, state, cfg, None, same_round)
    assert (plan.source, plan.similarity) == ("scratch", None), \
        "same-iteration policies are not donors (sequential/parallel identity)"
    state.policy_ref = "policy:cW"
    T._POLICY_CODE.clear()

    # A round driver's `resume_ref` (a LaRes slice, a halving rung) is the
    # candidate's OWN policy and is recorded as such, whatever `train.init` says.
    plan = T._training_init(None, state, cfg, "policy:cW", child)
    assert (plan.ref, plan.source, plan.from_cand_id) == ("policy:cW", "continuation", "cW")

    cfg = load_gt_published(profile="dev", overrides={
        "train.init": "secondary_replay_buffer", "train.secondary_buffer.size": 20})
    plan = T._training_init(None, state, cfg)
    ref, sec, ratio = plan.ref, plan.secondary, plan.ratio
    assert ref is None, "GT shares a BUFFER, never a policy"
    assert len(sec) == 20, "train.secondary_buffer.size must cap what is handed over"
    assert ratio == pytest.approx(0.2)
    assert sec[0][1] == 30, "the cap keeps the TAIL -- the incumbent's latest experience"


@needs_sb3
def test_train_init_is_live_end_to_end_on_sb3():
    """The whole chain: candidate 1 stores, `update` carries, candidate 2 loads.

    `warm_started_from` is the artifact field that distinguishes a warm start
    that took from one that was configured and dropped, so it is what this
    asserts -- a fitness comparison would also pass on a lucky seed.
    """
    ctx = _ctx("pendulum", **{"train.backend": "sb3", "train.algorithm": "sac",
                              "train.env_steps": 1200,
                              "train.init": "warm_start_from_best"})
    first = _train(ctx, "c0000", code=UPRIGHT)
    assert first.policy_ref, "iteration 0 must leave a checkpoint behind"
    assert first.seed_metrics[0]["warm_started_from"] == "", \
        "iteration 0 has no incumbent; it must report a cold start"

    state = RunState()
    state.best = CandidateReport(cand_id="c0000", candidate=first.candidate,
                                 result=first, fitness=1.0)
    _carry_inner_loop_refs(state)

    second = _train(ctx, "c0001", state=state, code=UPRIGHT)
    assert second.seed_metrics[0]["warm_started_from"] == first.policy_ref, (
        "iteration 1 declared warm_start_from_best and started cold -- "
        "this is RDA silently running as RDA-minus-Ckpt")


def test_train_init_is_live_end_to_end_on_the_surrogate_backend():
    """The same chain on `mock`, because failure (1) is backend-independent.

    A test that only covered sb3 would let a missing `state.policy_ref`
    assignment survive under the `tester` profile, which is the tier the whole
    suite runs.
    """
    ctx = _ctx("toy_gridworld", **{"train.backend": "mock", "train.env_steps": 2000,
                                   "train.algorithm": "q_learning",
                                   "train.init": "warm_start_from_best"})
    first = _train(ctx, "c0000")
    assert first.seed_metrics[0]["warm_started_from"] == ""

    state = RunState()
    state.best = CandidateReport(cand_id="c0000", candidate=first.candidate,
                                 result=first, fitness=1.0)
    _carry_inner_loop_refs(state)

    second = _train(ctx, "c0001", state=state)
    assert second.seed_metrics[0]["warm_started_from"] == first.policy_ref


# ==========================================================================
# sb3: the warm start and the buffer have to CHANGE the run
# ==========================================================================


@needs_sb3
def test_a_warm_started_sb3_policy_is_the_stored_one_exactly():
    """Deterministic half of "the warm start works".

    Loading weights and then not using them would still set
    `warm_started_from`, so this compares actions rather than bookkeeping.
    Exact equality is available here and an RL curve comparison is not, which
    is why both tests exist.
    """
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC

    class _E(gym.Env):
        observation_space = spaces.Box(-np.ones(3, np.float32), np.ones(3, np.float32))
        action_space = spaces.Box(-np.ones(1, np.float32), np.ones(1, np.float32))

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, a):
            return self.observation_space.sample(), 0.0, False, True, {}

    teacher = SAC("MlpPolicy", _E(), seed=1, device="cpu", buffer_size=200, verbose=0)
    teacher.learn(total_timesteps=300)
    blob = T._sb3_policy_blob(teacher)
    assert blob is not None and blob.dtype == np.uint8 and blob.nbytes > 10_000

    student = SAC("MlpPolicy", _E(), seed=2, device="cpu", buffer_size=200, verbose=0)
    probe = np.array([[0.2, -0.4, 0.6]], dtype=np.float32)
    before, _ = student.predict(probe, deterministic=True)
    assert T._sb3_apply_policy(student, blob, "policy:cT") is True
    after, _ = student.predict(probe, deterministic=True)
    want, _ = teacher.predict(probe, deterministic=True)

    assert not np.allclose(before, want), "the probe cannot distinguish the two models"
    np.testing.assert_allclose(after, want, atol=1e-6)

    # A blob from another backend must be ignored, not fatal: a checkpoint that
    # no longer fits must never destroy the search that produced it.
    assert T._sb3_apply_policy(student, np.zeros((4, 4)), "policy:cQ") is False


@needs_sb3
def test_a_warm_start_measurably_changes_what_sb3_learns():
    """The measured half. 1,500 SAC steps on pendulum is far too few to reach
    the upright from scratch and plenty to keep it from a trained checkpoint,
    so the gap is a step change (measured 0.0 -> 1.0) rather than a margin."""
    teacher_ctx = _ctx("pendulum", **{"train.backend": "sb3", "train.algorithm": "sac",
                                      "train.env_steps": 8000})
    teacher = _train(teacher_ctx, "cTEACH", code=UPRIGHT)
    assert max(p["fitness"] for p in teacher.checkpoints) > 0.5, \
        "the teacher never learned, so there is nothing to warm start FROM"

    warm_state = RunState()
    warm_state.policy_ref = teacher.policy_ref

    scores = {}
    for label, state, init in (("cold", RunState(), "from_scratch"),
                               ("warm", warm_state, "warm_start_from_best")):
        ctx = _ctx("pendulum", **{"train.backend": "sb3", "train.algorithm": "sac",
                                  "train.env_steps": 1500, "train.init": init})
        res = _train(ctx, "cSTUDENT", state=state, code=UPRIGHT)
        scores[label] = res.checkpoints[0]["fitness"]

    assert scores["warm"] > scores["cold"], (
        f"warm start did not change the run: first checkpoint {scores} -- "
        "train.init is a fabricated pin again")


@needs_sb3
def test_the_secondary_buffer_supplies_exactly_its_ratio_of_every_batch():
    """`train.secondary_buffer.ratio` is GT App. E Table 3's SAMPLING ratio.

    Pre-seeding the primary buffer instead would make the share decay as the
    primary fills -- 0.2 for the first few thousand steps of a 1M-step run and
    ~0.003 by the end. Marking the two sources with disjoint rewards is what
    turns "the buffer is wired up" into an exact count.
    """
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.buffers import ReplayBuffer

    obs_space = spaces.Box(-np.ones(2, np.float32), np.ones(2, np.float32))
    act_space = spaces.Box(-np.ones(1, np.float32), np.ones(1, np.float32))
    cls = T._mixed_buffer_class()

    primary = cls(64, obs_space, act_space, device="cpu", n_envs=1,
                  handle_timeout_termination=False)
    for i in range(64):
        primary.add(np.zeros((1, 2), np.float32), np.zeros((1, 2), np.float32),
                    np.zeros((1, 1), np.float32), np.array([1.0]), np.array([False]), [{}])
    secondary = ReplayBuffer(32, obs_space, act_space, device="cpu", n_envs=1,
                             handle_timeout_termination=False)
    secondary.rewards[:] = -1.0
    secondary.pos, secondary.full = 0, True

    primary.secondary_buffer = secondary
    primary.secondary_ratio = 0.25
    batch = primary.sample(100)
    from_secondary = int((batch.rewards.numpy() < 0).sum())
    assert from_secondary == 25, f"expected 25 of 100 from the secondary, got {from_secondary}"
    assert batch.observations.shape[0] == 100

    # ratio 0 must be byte-for-byte the stock buffer, because `from_scratch` is
    # what every other published config inherits.
    primary.secondary_ratio = 0.0
    assert int((primary.sample(100).rewards.numpy() < 0).sum()) == 0


@needs_sb3
def test_the_secondary_buffer_is_relabelled_with_the_new_candidates_reward():
    """The one thing a shared buffer must not do is leak the incumbent's reward.

    `_QLearner._replay_secondary` says so in prose; on sb3 the relabelling is
    precomputed, so it is checkable exactly.
    """
    ctx = _ctx("pendulum", **{"train.backend": "sb3", "train.algorithm": "sac",
                              "train.env_steps": 1200,
                              "loop.carry": ["best_reward", "subtask_list", "replay_buffer"],
                              "train.init": "secondary_replay_buffer"})
    donor = _train(ctx, "cDONOR", code=UPRIGHT)
    stored = T._REPLAY_STORE[donor.replay_ref]
    assert isinstance(stored, T._ReplaySlice)
    assert not hasattr(stored, "rew") and len(stored[0]) == 4, \
        "the store must hold RAW transitions -- a stored reward is the leak"

    from stable_baselines3 import SAC

    reward = T.compile_reward("def compute_reward(state, action):\n"
                              "    return 7.0, {'k': 7.0}\n")
    model = SAC("MlpPolicy", _gym_shim(ctx.env), seed=0, device="cpu",
                buffer_size=64, verbose=0)
    mix = T._sb3_secondary_buffer(stored, 0.2, model, ctx.env, reward, "none", True)
    assert mix is not None and mix.buffer_size == len(stored)
    np.testing.assert_allclose(mix.rewards[:, 0], 7.0, atol=1e-5)


def _gym_shim(env):
    """Minimal gym.Env with the adapter's spaces -- `_sb3_secondary_buffer`
    only ever reads `model.observation_space` / `model.action_space`."""
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
            return self.observation_space.sample(), 0.0, False, True, {}

    return _Shim()


# ==========================================================================
# The two things the stores have to survive: the fork, and the checkpoint
# ==========================================================================


@needs_sb3
def test_the_sb3_stores_cross_the_candidate_parallelism_fork():
    """`parallelism_parallel` forks; a worker's store writes die with it.

    `_merge_child` already replays them and `tests/test_parallelism.py` proves
    the mock backend's do -- but the payload is only as good as what the child
    put in the store, and an sb3 child that stores nothing leaves nothing to
    replay. This fails for a different reason than the sequential test would,
    which is why it is separate.
    """
    ctx = _ctx("pendulum", **{"train.backend": "sb3", "train.algorithm": "sac",
                              "train.env_steps": 1000,
                              "loop.carry": ["best_reward", "subtask_list",
                                             "policy_checkpoint", "replay_buffer"],
                              "train.init": "secondary_replay_buffer",
                              "train.candidate_parallelism": "parallel",
                              "loop.max_parallel_trainings": 2})
    cands = [Candidate(cand_id=f"c000{i}", reward_code=UPRIGHT, iteration=0)
             for i in range(2)]
    backend = registry.get("train_backend", "sb3")
    runner = registry.get("candidate_parallelism", "parallel")
    results = runner(ctx, RunState(), cands,
                     lambda c: backend(ctx, RunState(), c, 1), lambda c, r: None)

    assert [r.cand_id for r in results] == ["c0000", "c0001"], \
        "reassembly must be by candidate index, never completion order"
    for r in results:
        assert r.policy_ref and T._POLICY_STORE.get(r.policy_ref) is not None, (
            f"{r.cand_id}: the worker's _POLICY_STORE write did not come home -- "
            "warm_start_from_best silently becomes from_scratch")
        assert r.replay_ref and len(T._REPLAY_STORE.get(r.replay_ref) or ()) > 0, (
            f"{r.cand_id}: the worker's _REPLAY_STORE write did not come home")


def test_a_replay_slice_round_trips_through_the_checkpoint_codec(tmp_path):
    """A resumed 1M-step run must restore the buffer it had, not an empty one."""
    n = 40
    original = T._ReplaySlice(
        np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        np.linspace(-1, 1, n, dtype=np.float32).reshape(n, 1),
        np.arange(n * 3, dtype=np.float32).reshape(n, 3) + 0.5,
        np.arange(n) % 3 == 0)
    spill = _ArraySpill(tmp_path)
    back = decode(_encode_replay(original, spill, "$.replay"), tmp_path)

    assert isinstance(back, T._ReplaySlice), \
        "decoding columnar into 100,000 tuples defeats the representation"
    np.testing.assert_allclose(back.obs, original.obs)
    np.testing.assert_allclose(back.act, original.act)
    np.testing.assert_allclose(back.next_obs, original.next_obs)
    np.testing.assert_array_equal(back.done, original.done)


def test_a_continuous_action_survives_the_legacy_tuple_encoding(tmp_path):
    """An unconditional `int(r[1])`, inside a `try` whose `except` never fires,
    is a silent truncation.

    On a continuous action space it turns 0.37 into 0 for every row -- a
    total, silent loss of the action column, and the buffer still decodes.
    """
    rows = [(np.zeros(2, np.float32), np.float32(0.37), np.ones(2, np.float32), False)]
    back = decode(_encode_replay(rows, _ArraySpill(tmp_path), "$.replay"), tmp_path)
    assert float(np.asarray(back[0][1]).ravel()[0]) == pytest.approx(0.37, abs=1e-6), \
        "a continuous action was truncated to an int by the replay codec"


@needs_sb3
def test_a_policy_blob_round_trips_through_the_checkpoint_codec(tmp_path):
    """`_PolicyBlob` is an ndarray precisely so the spill path verifies it."""
    from bird.checkpoint import encode

    blob = T._PolicyBlob(b"\x00\x01\x02" * 100_000)
    spill = _ArraySpill(tmp_path)
    tagged = encode(blob, spill, "$.policy")
    assert "__ndarray_ref__" in tagged, \
        "a 300 KB policy must spill to a sha256-verified sidecar, not inline as base64"
    back = decode(tagged, tmp_path)
    assert back.dtype == np.uint8 and back.tobytes() == blob.tobytes()


# ==========================================================================
# Bounds
# ==========================================================================


def test_the_replay_store_is_bounded_in_bytes_as_well_as_in_entries():
    """`train.secondary_buffer.size: 100000` is ~33 MB per candidate on
    Meta-World, and `_STORE_LIMIT` alone would hold 64 of them. At most three
    are ever dereferenceable (`checkpoint.reachable_refs`), so the rest are
    1.3 GB of resident garbage in the parent of an 8-worker sweep."""
    big = T._ReplaySlice(np.zeros((4000, 64), np.float32), np.zeros((4000, 2), np.float32),
                         np.zeros((4000, 64), np.float32), np.zeros(4000, bool))
    limit = 8 * big.nbytes
    for i in range(40):
        T._store(T._REPLAY_STORE, f"replay:c{i:04d}", big, limit)
    assert len(T._REPLAY_STORE) <= 8
    assert "replay:c0039" in T._REPLAY_STORE, "eviction must be oldest-first"
    assert "replay:c0000" not in T._REPLAY_STORE

    # The count bound still applies, and on the surrogate backends it is the
    # only one that ever fires.
    T._REPLAY_STORE.clear()
    for i in range(T._STORE_LIMIT + 5):
        T._store(T._REPLAY_STORE, f"replay:c{i:04d}", [(np.zeros(2), 0, np.zeros(2), False)],
                 512 * 1024 * 1024)
    assert len(T._REPLAY_STORE) == T._STORE_LIMIT


def test_the_byte_cap_also_holds_on_the_merge_path(monkeypatch):
    """A cap is only as good as the path the PARENT actually accumulates on.

    Every in-worker `_store` call passes its cap, and `_merge_child` -- the
    parent-side replay of a forked child's writes, i.e. the ONLY path on which
    the parent grows under `train.candidate_parallelism: parallel` -- must too.
    `_store` skips its byte-eviction loop entirely when `max_bytes` is falsy, so
    without it the parent is bounded by `_STORE_LIMIT` alone: 40 x 31.38 MB
    `_ReplaySlice`s (8 candidates x 5 iterations at a Meta-World shape) is
    ~1.25 GB against a 512 MB cap, and 40 < 64 so nothing evicts. It would also
    diverge from `sequential`, which is the property `tests/test_parallelism.py`
    exists to protect.

    The test above cannot see any of that: it calls `_store` directly.
    """
    slice_ = T._ReplaySlice(np.zeros((500, 64), np.float32), np.zeros((500, 2), np.float32),
                            np.zeros((500, 64), np.float32), np.zeros(500, bool))
    blob = np.zeros(4096, np.uint8)
    monkeypatch.setattr(T, "_REPLAY_STORE_MAX_BYTES", 8 * slice_.nbytes)
    monkeypatch.setattr(T, "_POLICY_STORE_MAX_BYTES", 8 * int(blob.nbytes))
    T._REPLAY_STORE.clear()
    T._POLICY_STORE.clear()

    ctx = Context(cfg=load("eureka", profile="tester"), budget=Budget())
    for i in range(40):
        cand = Candidate(cand_id=f"c{i:04d}", iteration=i // 8, reward_code="")
        T._merge_child(ctx, cand, {
            "result": TrainResult(cand_id=cand.cand_id, candidate=cand, trained=True),
            "candidate": cand,
            "policy": [(f"policy:c{i:04d}", blob)],
            "replay": [(f"replay:c{i:04d}", slice_)],
            "env_states": None,
            "budget": {},
        })

    assert len(T._REPLAY_STORE) <= 8, (
        f"{len(T._REPLAY_STORE)} replay entries survived 40 merges under an "
        f"8-entry byte cap; _merge_child is not passing it")
    assert len(T._POLICY_STORE) <= 8
    assert "replay:c0039" in T._REPLAY_STORE and "replay:c0000" not in T._REPLAY_STORE, \
        "eviction on the merge path must be oldest-first, as it is sequentially"
    T._REPLAY_STORE.clear()
    T._POLICY_STORE.clear()


# ==========================================================================
# The chain: `bc_prior_then_warm_start`
# ==========================================================================

def test_the_chain_starts_from_the_clone_and_then_from_the_selected_policy():
    """Round 0 has nothing selected yet, so the chain must start from the clone's
    fixed slot; every later round starts from `state.policy_ref`, the policy the
    judge selected in the round before. `warm_start_from_best` alone cold-starts
    round 0, which is why the mode exists."""
    from types import SimpleNamespace
    from bird.components.training import BC_PRIOR_REF, _training_init
    cfg = {"train.init": "bc_prior_then_warm_start"}
    cfg = SimpleNamespace(get=lambda k, d=None: cfg.get(k, d)) if False else _Cfg(cfg)
    ref, extra, ratio = _training_init(None, SimpleNamespace(policy_ref=None), cfg)[:3]
    assert (ref, extra, ratio) == (BC_PRIOR_REF, [], 0.0)
    ref = _training_init(None, SimpleNamespace(policy_ref="policy:c0007"), cfg).ref
    assert ref == "policy:c0007"
    # and the plain warm start still cold-starts round 0 -- the two modes differ
    # exactly there, which is the whole reason for the second one
    ref = _training_init(None, SimpleNamespace(policy_ref=None),
                         _Cfg({"train.init": "warm_start_from_best"})).ref
    assert ref is None


class _Cfg(dict):
    def get(self, key, default=None):  # the `cfg.get(key, default)` surface training.py reads
        return dict.get(self, key, default)


def test_configs_copy_of_the_policy_store_limit_is_the_stores():
    """`bird/config.py` may not import `bird.components` (pyyaml + stdlib only), so
    it carries `_POLICY_STORE_LIMIT` as a literal for the warm-start coherence
    check; this holds it equal to the store's real bound."""
    from bird import config as C
    from bird.components import training as T
    assert C._POLICY_STORE_LIMIT == T._STORE_LIMIT
