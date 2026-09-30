"""The co-designed observation, checked on a real MuJoCo task rather than a toy.

`tests/test_limen_observation.py` pins the install on `toy_reacher`: six dims,
pure numpy, analytic dynamics. Everything it asserts is necessary and none of it
is sufficient, because three of the properties the install depends on are true of
that adapter for reasons that have nothing to do with the install:

  * **A `toy_reacher` state can be inverted arithmetically.** MetaWorld's cannot
    -- a 39-D observation does not determine `(qpos, qvel)`, so that adapter
    keeps an obs -> simulator-snapshot cache and raises `UnknownStateError` on a
    miss (`EnvAdapter.export_states` documents this at length). An install that
    let features leak into `env.step`, `task_metric`, `reference_reward` or the
    recorded trajectory would fail HERE and pass there.
  * **`toy_reacher` has no ignorable dimensions.** On MT10, 18 of 39 are always
    exactly zero (`object2_*` and their `prev_` copies) and another 18 are
    `prev_*` duplicates; the goal sits at 36:39. That is the situation LIMEN's
    §4.1 claim is actually about, and it is what makes the co-designed width
    (3-6) differ from the environment's (39) by an order of magnitude rather than
    by half.
  * **`toy_reacher`'s state fields are the ones the mock generator was written
    against** (`x, y, vx, vy, goal_x, goal_y`). On any other env the mock's
    hardcoded slot indices address the wrong slots -- see
    `test_the_mock_addresses_this_environments_slots`, which guards a defect
    that makes every candidate here untrainable for a reason that is the
    double's and not the method's.

`importorskip`, so this file is invisible without `--extra metaworld`. That is
the same bargain `tests/test_metaworld.py` makes and it has the same cost: a CI
job that installs `--extra test` only **reports these tests green by skipping**,
and only an install with `--extra metaworld` runs them. Run them by hand when touching the install:

    uv sync --frozen --extra metaworld --extra test
    uv run --frozen python3 -m pytest tests/test_limen_observation_mujoco.py
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("metaworld")
pytest.importorskip("mujoco")

from bird import registry                                          # noqa: E402
from bird.budget import Budget                                     # noqa: E402
from bird.components import training                               # noqa: E402
from bird.config import load                                       # noqa: E402
from bird.context import Context                                   # noqa: E402
from bird.state import RunState                                    # noqa: E402
from bird.types import Candidate                                   # noqa: E402

#: Dependency-gated AND heavyweight (it constructs MuJoCo and runs training
#: calls), so `slow` by two of the marker's own criteria. See pyproject.
pytestmark = pytest.mark.slow

ENV_ID = "mt10_reach-v3"

#: The interface a competent model writes for `reach`: the goal-relative
#: displacement of the hand, and nothing else. 3 features against the
#: environment's 39.
PHI = """
OBS_DIM = 3

def get_observation(state):
    import numpy as np
    hand = np.asarray(state[0:3], dtype=float)
    goal = np.asarray(state[36:39], dtype=float)
    d = goal - hand
    return [d[0], d[1], d[2]]
"""

#: Written against the STATE, exactly as LIMEN's prompt asks for -- the same
#: `state` `get_observation` receives. If the install ever re-based the reward
#: onto the features, `next_state[36:39]` would be out of bounds on a 3-vector
#: and this would raise rather than quietly score something else.
REWARD = """
def compute_reward(state, action, next_state):
    import numpy as np
    hand = np.asarray(next_state[0:3], dtype=float)
    goal = np.asarray(next_state[36:39], dtype=float)
    return -float(np.linalg.norm(goal - hand))
"""


@pytest.fixture(scope="module")
def env():
    registry.load_all()
    return registry.get("env", ENV_ID)({})


def _ctx(name: str = "limen", *, env=None, **over) -> Context:
    """`env` is a PARAMETER because the snapshot cache is per-adapter instance.

    MetaWorld keeps an obs -> (qpos, qvel) cache and raises `UnknownStateError`
    on a miss, so a test that trains on one instance and then scores the
    trajectory on another gets `cache 0/8192 entries, 1 misses` -- which reads
    exactly like the leak these tests are looking for and is not one. Sharing one
    instance across contexts produces exactly that false positive.
    """
    registry.load_all()
    base = {"problem.env_id": ENV_ID,
            "output.video.enabled": False, "output.wandb.video_record": "none"}
    base.update(over)
    cfg = load(name, profile="tester", overrides=base)
    return Context(cfg=cfg, budget=Budget(),
                   env=env if env is not None else registry.get("env", ENV_ID)({}))


def _candidate(obs_code=None, cand_id="c0000") -> Candidate:
    return Candidate(cand_id=cand_id, iteration=0, reward_code=REWARD,
                     observation_code=obs_code)


# --------------------------------------------------------------------------
# the config axis has to resolve at all
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["limen", "limen_reward_only", "card", "rda"])
def test_a_verifying_method_resolves_on_a_mujoco_env(name: str) -> None:
    """`-s problem.env_id=<a MuJoCo env>` must not be a validation error.

    It can be, for every published method that actually verifies. Every published
    config resolves `verify.static_checks: [signature_parse, ast_syntax]`; every
    MuJoCo task spec supplies a non-empty `verify.forbidden_symbols`; and
    `_check_coherence` refuses a gate in force whose reader is not in the list --
    correctly, it calls that its one dangerous check. The env is an override and
    there is no file to pin both keys together in, so unless the spec-supplied
    path adds the reader, `card`, `rda`, `limen` and `limen_reward_only` are all
    refused outright:

        verify.forbidden_symbols is in force but 'forbidden_symbols' is not in
        verify.static_checks, so nothing would ever check it

    `eureka` is the exception and not the counter-example -- it passes only
    because `verify.enabled: false` puts its gate out of force, which is why a
    one-config spot check would miss this.
    """
    # The ENV override alone, and nothing else. Also passing
    # `output.video.enabled: false` (which the training tests below need, to stay
    # off a GL context) makes `rda` refuse it -- correctly: `evaluate.artifacts`
    # includes `videos` because that method scores candidates by LOOKING at
    # frames. Adding an unrelated override to a config-resolution test is how a
    # test starts measuring the override.
    cfg = load(name, profile="tester", overrides={"problem.env_id": ENV_ID})

    gate = cfg["verify.forbidden_symbols"]
    assert gate, "precondition: this env's spec supplies a leak gate"
    if cfg["verify.enabled"]:
        assert "forbidden_symbols" in cfg["verify.static_checks"], (
            "the gate is in force and nothing reads it")


def test_the_derived_check_is_not_added_where_it_was_not_derived() -> None:
    """Only the spec-supplied path may edit an author's `verify.static_checks`.

    An AUTHORED `verify.forbidden_symbols` -- including the deliberate `[]` that
    `singh_orp` pins -- must reach `_check_coherence` untouched, so a config whose
    §2 is wrong stays a validation error with a file to fix it in. Asserted on
    the native env, where the spec supplies no gate and nothing should be added.
    """
    cfg = load("limen", profile="tester")
    assert cfg["verify.forbidden_symbols"] == []
    assert cfg["verify.static_checks"] == ["signature_parse", "ast_syntax"], (
        "a check was appended on an env whose spec supplies no gate")


# --------------------------------------------------------------------------
# the split, on an adapter that cannot survive getting it wrong
# --------------------------------------------------------------------------


def test_the_observation_reaches_the_policy_and_nothing_else(env) -> None:
    """The whole install, on 39 real MuJoCo dimensions.

    Four claims in one training call, because they are one property:
    `policy_input_dim` is 3 (the learner was built on phi), `observation_dim`
    agrees (the install is what put it there), the recorded trajectory is 39 wide
    (the raw state survived to the recorder), and the run is `trained` at all
    (the reward was called on states, so `next_state[36:39]` was in bounds).
    """
    ctx = _ctx(env=env)
    cand = _candidate(PHI)
    res = registry.get("train_backend", "mock")(ctx, RunState(), cand, n_seeds=1)

    assert res.trained, res.error
    sm = res.seed_metrics[0]
    assert sm["policy_input_dim"] == 3, (
        f"the learner was built on {sm['policy_input_dim']} inputs, not phi's 3")
    assert sm["observation_dim"] == 3
    assert cand.meta["observation_dim"] == 3
    assert env.obs_dim == 39, "precondition: this env is the 39-D one"

    assert res.trajectories
    for traj in res.trajectories:
        assert np.asarray(traj.states).shape[1] == 39, (
            "the recorded trajectory is in FEATURE space. On this adapter that is "
            "not merely wrong for the readers -- `task_metric`, the screens, the "
            "recorder -- it is unrecoverable: a 39-D MetaWorld observation does "
            "not determine (qpos, qvel), so a featured row can never be replayed.")


def test_the_snapshot_cache_still_resolves_every_recorded_state(env) -> None:
    """`reference_reward` on a recorded row is the check that phi did not leak.

    MetaWorld's obs -> snapshot cache raises `UnknownStateError` on a miss, so
    this is the one adapter in the repo where a featured trajectory fails LOUDLY
    instead of silently scoring the wrong thing. `_run_backend` computes
    `gt_return` from `env.reference_reward` during the rollout, so a leak would
    already have raised -- this asserts it from the artifact side, where a future
    refactor that moved recording after training could reintroduce it.
    """
    ctx = _ctx(env=env)
    res = registry.get("train_backend", "mock")(ctx, RunState(), _candidate(PHI),
                                                n_seeds=1)
    assert res.trained, res.error
    traj = res.trajectories[0]
    rows = np.asarray(traj.states)
    # Every row, not a sample: the cache is an LRU and a partial hit is the
    # failure mode `retain_states` exists for.
    vals = [float(env.reference_reward(r)) for r in rows]
    assert len(vals) == rows.shape[0]
    assert all(np.isfinite(v) for v in vals)
    assert 0.0 <= float(env.task_metric(traj)) <= 1.0


def test_the_two_limen_points_differ_on_this_env() -> None:
    """The §5.4 ablation, on the tier where it is meant to be run.

    One adapter per arm here, deliberately: the two are independent runs and
    sharing a cache between them would let one arm's trajectories satisfy the
    other's lookups.
    """
    backend = registry.get("train_backend", "mock")
    joint = backend(_ctx("limen"), RunState(), _candidate(PHI, "j"), n_seeds=1)
    solo = backend(_ctx("limen_reward_only"), RunState(), _candidate(PHI, "s"),
                   n_seeds=1)
    assert joint.trained and solo.trained, (joint.error, solo.error)
    assert joint.seed_metrics[0]["policy_input_dim"] == 3
    assert solo.seed_metrics[0]["policy_input_dim"] == 39
    assert solo.seed_metrics[0]["observation_dim"] is None


# --------------------------------------------------------------------------
# the generator double has to address THIS environment
# --------------------------------------------------------------------------


def test_the_mock_addresses_this_environments_slots(env) -> None:
    """The mock's features must read the slots this env actually has.

    THE DEFECT THIS PINS. `_OBS_FEATURES` names slots
    through `_f(names, index)`: the accessor tries the names (for a dict or
    attribute state) and falls back to the index for an ndarray -- and every
    adapter here hands out an ndarray, so the INDEX is what is used. Indices
    written for `toy_reacher`, where `goal_x` is `s[4]`, are wrong here: on this
    env `s[4]` is `object_x` and the goal is at 36:39, so
    `_f(("goal_x", "target_x"), 4)` would read the OBJECT.

    Once the policy trains on the observation, that is the difference between an
    interface with a goal in it and one without: the cascade screen rejects every
    candidate at `success 0.0000 < 0.0100` (measured), and the tester profile on
    this env exercises the archive and the screen while never producing a
    trainable interface. A failure that looks like the method's and is the
    double's.
    """
    from bird.llm import mock as M

    prompt = env.describe("state_action_api_stub")
    table = M._state_field_index(prompt)
    assert table.get("hand_x") == 0 and table.get("goal_x") == 36, (
        "the prompt's own state table did not parse; the mock has nothing to "
        "resolve against and would fall back to toy_reacher's indices")

    import random
    for seed in range(8):
        prog = M._observation_program(random.Random(seed), 512, prompt)
        assert "), 4)" not in prog, (
            "a feature still reads s[4] (object_x) where it means the goal:\n" + prog)
        assert '"goal_x", "target_x"), 36)' in prog or "goal_x" not in prog

        # And it has to RUN, and run at its declared width.
        phi = training.compile_observation(prog, env)
        assert phi.dim >= 3
        declared = [int(l.split("=")[1]) for l in prog.splitlines()
                    if l.startswith("OBS_DIM")]
        assert declared and declared[0] == phi.dim, (
            f"OBS_DIM says {declared} and phi measures {phi.dim}")


def test_the_mock_drops_features_this_env_cannot_supply(env) -> None:
    """No velocity is observable on MT10, so no velocity feature may be emitted.

    The adapter says so in as many words (`prev_hand_x` is provided *instead*).
    Emitting `_f(("vx", ...), 2)` would hand the policy `hand_z` labelled `vx` --
    a feature that reads as deliberate and is noise. A shorter feature vector is
    a true statement about what the environment exposes, which is why
    `_resolve_feature` returns None rather than keeping the fallback index.
    """
    from bird.llm import mock as M
    import random

    names = {f for f, _ in env._state_fields}
    assert not {"vx", "xdot", "vel_x"} & names, "precondition: no velocity here"
    for seed in range(8):
        prog = M._observation_program(random.Random(seed), 512,
                                      env.describe("state_action_api_stub"))
        assert "# vx" not in prog and "# speed" not in prog, prog


def test_a_narrow_phi_makes_a_narrow_network(env) -> None:
    """The install has to reach the POLICY CLASS on this env too.

    39 -> 3 is a 13x reduction in the linear searcher's weight matrix, and
    `_CEMLearner` sizes it `(action_dim, obs_dim + 1)`. This is the check that
    fails if `_ObsView` is built, recorded, and never handed to `_make_learner`.
    """
    phi = training.compile_observation(PHI, env)
    view = training._ObsView(env, phi)
    reward = training.compile_reward(REWARD)
    rng = np.random.default_rng(0)

    def make(on):
        return training._make_learner("auto", on, reward, rng,
                                      training._RewardNorm("none"),
                                      lambda exc: 0.0, 600, None, (), 0.0, 1)

    assert make(view).dim == (env.action_dim, 3 + 1)
    assert make(env).dim == (env.action_dim, 39 + 1)
