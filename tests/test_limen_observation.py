"""LIMEN's co-designed observation has to reach the policy, or `limen` IS `limen_reward_only`.

THE DEFECT THIS FILE EXISTS TO PREVENT is not a crash and not a wrong number --
it is two config points that differ in every artifact and not at all in
behaviour. The shape it takes:

  * `generation.py` prompts for `get_observation(state)` in the same block as
    the reward, `_apply_co_design` lifts it out and rejects it over
    `co_design.observation_max_dim`, `artifacts.py` writes it to
    `candidates/<id>/observation.py`, and `selection.py::_observation_dim` uses
    its width as one of the two MAP-Elites descriptor axes;
  * but if no env adapter and no backend reads `Candidate.observation_code`,
    none of that reaches the learner.

The policy would then train on the environment's own observation while the
archive binned candidates by the dimensionality of a function no learner had
seen, and `limen` / `limen_reward_only` -- the paper's central §5.4 ablation
(obs-only 100% on Easy against 0% on Panda; reward-only 19% Medium / 71%
Panda) -- would produce two archives, two hashes, two run directories and ONE
method. Nothing in any artifact would say so, which is what makes this a test
rather than a comment: every observable a reader would check is already
different.

WHAT EACH CHECK BELOW IS FOR. They are not variations on one assertion; the four
ways this regresses are independent, and three of them are silent:

  1. `test_the_two_limen_points_differ_in_what_the_policy_sees` -- the headline.
     Fails if the install is removed.
  2. `test_search_space_is_the_gate` / `..._warns_when_ignored` -- fails if the
     gate moves back to `generate.co_design.observation_fn`, which would make
     `problem.search_space` declared-and-unread (a key that validates, moves
     the hash, and changes nothing).
  3. `test_the_reward_and_the_ground_truth_stay_on_the_raw_state` -- the one
     that produces a complete, plausible, WRONG result if it regresses. Both
     functions are handed the same `state` by LIMEN's prompt, so a reward
     re-based onto the features scores a different program from the one the
     model wrote, and a `task_metric` re-based onto them would let a candidate
     move the ground truth by choosing what it observes.
  4. `test_the_width_is_measured_and_not_estimated` -- fails if `dim` goes back
     to being read off the source. `_literal_return_width`'s own docstring says
     its AST read is a LOWER BOUND, and an archive axis and a `max_dim` cap
     enforced against a lower bound are a different axis and a cap that does
     not hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components import training
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate

# DELIBERATELY NOT `pytest.mark.slow`, and the reason is the marker's own rule.
# pyproject's note assigns `slow` BY KIND, not by stopwatch, and the kinds are
# heavyweight execution, dependency-gated tiers, rendering and sweeps over the
# task catalogue. This file is none of them: mock LLM, mock learner, toy env, six
# single-candidate training calls at 400 env steps, 3.2 s for the file. It is a
# tester-tier smoke test by that rule and it runs on every PR.
#
# Which is the point. The defect here is silent -- a run completes, an archive
# fills, two run directories appear, and the method is the other method -- so it
# is exactly the class a PR must not be able to reintroduce and see green. The
# question this file asks is the one `-m "not slow"` is defined to ask: did this
# change break the algorithm on the tester tier?


#: A phi that is narrower than `toy_reacher`'s 6-D state AND sufficient for the
#: task: goal-relative displacement plus its norm. Both halves matter -- narrower
#: so a width difference is observable, sufficient so a screen rejecting the
#: candidate cannot be mistaken for the install failing.
_OBS_3 = """
def get_observation(state):
    dx = state[4] - state[0]
    dy = state[5] - state[1]
    return [dx, dy, (dx * dx + dy * dy) ** 0.5]
"""

#: Same features, one more. Two widths in one test keep "the archive can bin
#: this" honest: a fixed width would pass against an implementation that
#: hard-coded any number at all.
_OBS_4 = """
def get_observation(state):
    dx = state[4] - state[0]
    dy = state[5] - state[1]
    return [dx, dy, (dx * dx + dy * dy) ** 0.5, state[2]]
"""

#: Width the AST heuristic CANNOT read. `selection._literal_return_width` gives
#: 2 for `np.concatenate([a, b])` (it takes `len` of the first argument's
#: elements) and `generation._estimate_obs_dim` gives 2 as well, against a true
#: width of 4. That gap is the whole argument for measuring.
_OBS_CONCAT = """
import numpy as np

def get_observation(state):
    near = np.asarray([state[4] - state[0], state[5] - state[1]])
    vel = np.asarray([state[2], state[3]])
    return np.concatenate([near, vel])
"""

_REWARD = """
def compute_reward(state, action, next_state):
    dx = next_state[4] - next_state[0]
    dy = next_state[5] - next_state[1]
    return -(dx * dx + dy * dy) ** 0.5
"""


def _env():
    registry.load_all()
    return registry.get("env", "toy_reacher")({})


def _ctx(name: str = "limen", **overrides) -> Context:
    registry.load_all()
    cfg = load(name, profile="tester", overrides=overrides or None)
    return Context(cfg=cfg, budget=Budget(), env=_env())


def _candidate(obs_code=None, cand_id: str = "c0000") -> Candidate:
    return Candidate(cand_id=cand_id, iteration=0, reward_code=_REWARD,
                     observation_code=obs_code)


# --------------------------------------------------------------------------
# 1. the headline: the two published points must differ behaviourally
# --------------------------------------------------------------------------


def test_the_two_limen_points_differ_in_what_the_policy_sees() -> None:
    """§5.4's ablation has to be an ablation, not two names for one MDP.

    Asserted through `train()` and its ARTIFACT rather than by calling
    `_install_observation` directly: what can break is the wiring between the
    install and the learner, and a unit test of the installer would pass
    throughout.
    """
    joint = _ctx("limen")
    solo = _ctx("limen_reward_only")

    backend = registry.get("train_backend", "mock")
    r_joint = backend(joint, RunState(), _candidate(_OBS_3), n_seeds=1)
    r_solo = backend(solo, RunState(), _candidate(_OBS_3), n_seeds=1)

    assert r_joint.trained and r_solo.trained, (r_joint.error, r_solo.error)
    joint_m, solo_m = r_joint.seed_metrics[0], r_solo.seed_metrics[0]

    # THE ASSERTION IS ON `policy_input_dim`, NOT ON `observation_dim`, AND THAT
    # IS THE WHOLE DESIGN OF THIS TEST. `observation_dim` is derived from phi,
    # which is compiled before the learner is built -- verified by mutation:
    # handing `_make_learner` the raw adapter instead of the view leaves every
    # phi-derived assertion in this file green while the policy trains on the
    # environment's own observation, i.e. it reproduces the exact defect and the
    # test does not see it. `policy_input_dim` is read off the env object the
    # LEARNER is holding, so it is the only field here that can tell them apart.
    assert joint_m["policy_input_dim"] == 3, (
        "the co-designed observation did not reach the policy: `limen` built its "
        f"learner on {joint_m['policy_input_dim']!r} inputs against a 3-feature "
        "phi. That is the state this file exists to prevent -- "
        "the search space says the MDP interface and the learner sees the "
        "environment's own observation.")
    assert joint_m["observation_dim"] == 3
    assert solo_m["policy_input_dim"] == joint.env.obs_dim == 6
    assert solo_m["observation_dim"] is None, (
        f"`limen_reward_only` installed an observation "
        f"(dim={solo_m['observation_dim']!r}). §5.4 is 'keeping observations "
        "fixed to raw simulator state'; installing one here would make the "
        "ablation arm the joint point.")
    # The one comparison a reader would make, stated once.
    assert joint_m["policy_input_dim"] != solo_m["policy_input_dim"]


def test_the_archive_axis_becomes_a_measurement() -> None:
    """LIMEN's first MAP-Elites descriptor stops being an AST guess.

    `selection._observation_dim` prefers `meta['observation_dim']` over its own
    heuristic, so installing the observation also fixes the axis -- and the
    DECLARED width is kept beside it rather than reconciled, so a program that
    lies about its own `OBS_DIM` is visible instead of silently redefining the
    x-axis of the archive.
    """
    ctx = _ctx("limen")
    cand = _candidate(_OBS_3)
    cand.meta["obs_dim"] = 99                     # what the model claimed
    registry.get("train_backend", "mock")(ctx, RunState(), cand, n_seeds=1)

    assert cand.meta["observation_dim"] == 3
    assert cand.meta["observation_dim_declared"] == 99, (
        "the declared width was dropped or overwritten -- a model that lies "
        "about OBS_DIM must stay visible in meta.json")
    assert cand.meta["observation_installed"] is True


def test_two_candidates_can_occupy_two_archive_columns() -> None:
    """Widths differ per candidate, which is what the descriptor axis is for."""
    ctx = _ctx("limen")
    backend = registry.get("train_backend", "mock")
    a = _candidate(_OBS_3, "c0000")
    b = _candidate(_OBS_4, "c0001")
    backend(ctx, RunState(), a, n_seeds=1)
    backend(ctx, RunState(), b, n_seeds=1)
    assert (a.meta["observation_dim"], b.meta["observation_dim"]) == (3, 4)


# --------------------------------------------------------------------------
# 2. `problem.search_space` is the gate
# --------------------------------------------------------------------------


def test_search_space_is_the_gate() -> None:
    """Stage 3 dispatches on §0's key, so the key is honoured rather than declared.

    `limen_reward_only` reaches the same state through the published route
    (`search_space: [reward]` plus `co_design.observation_fn: false`); this
    drives the gate directly, so removing the dispatch and relying on
    `co_design.observation_fn` alone fails here even though the published pair
    would still look right.
    """
    ctx = _ctx("limen", **{"problem.search_space": ["reward"],
                           "generate.co_design.observation_fn": False,
                           "update.co_evolve.observation_fn": False})
    view, phi = training._install_observation(ctx, _candidate(_OBS_3), ctx.env)
    assert phi is None
    assert view is ctx.env, "a view was built for a search space that excludes observation"


def test_the_gate_is_section_0_and_not_section_1() -> None:
    """§0 says what the searched object IS; §1 says how it is ELICITED.

    This is the check that separates the two, and it needs a config where they
    disagree. `_check_coherence` forbids `co_design.observation_fn: true` without
    `observation` in the space, so the disagreement can only run the other way --
    a space that includes the observation with the §1 elicitation off, which is
    what a staged search space looks like and which validates cleanly. A
    candidate arriving with an observation there has one because §0 says the
    observation is part of the searched object, and stage 3 installs it.

    Verified by mutation: gating on `generate.co_design.observation_fn` instead
    passes every other check in this file, because the coherence rule keeps the
    two keys moving together in every published config. Without this test,
    "`problem.search_space` is honoured" would be unfalsifiable -- as it would
    be if only `_check_coherence` read it.

    §6's `update.co_evolve.observation_fn` is switched off beside §1's: it
    requires §1's key (persisting an observation nothing generates is
    refused), and it is not what this test is about.
    """
    ctx = _ctx("limen", **{"generate.co_design.observation_fn": False,
                           "update.co_evolve.observation_fn": False})
    assert ctx.cfg["problem.search_space"] == ["reward", "observation"]
    _view, phi = training._install_observation(ctx, _candidate(_OBS_3), ctx.env)
    assert phi is not None and phi.dim == 3, (
        "stage 3 declined to install an observation the search space includes, "
        "which means the gate is reading §1's elicitation key rather than §0's "
        "declaration of the space")


def test_an_ignored_observation_is_a_warning_and_not_a_silence(caplog) -> None:
    """A candidate carrying an observation nothing will install must say so.

    The alternative is the exact invisibility this file is about: code on disk
    in `candidates/<id>/observation.py`, an archive axis binned on it, and a
    policy that never saw it.
    """
    ctx = _ctx("limen", **{"problem.search_space": ["reward"],
                           "generate.co_design.observation_fn": False,
                           "update.co_evolve.observation_fn": False})
    with caplog.at_level("WARNING"):
        training._install_observation(ctx, _candidate(_OBS_3), ctx.env)
    assert any("search_space" in r.getMessage() for r in caplog.records), caplog.text


# --------------------------------------------------------------------------
# 3. the split: only the POLICY sees phi
# --------------------------------------------------------------------------


def test_the_reward_and_the_ground_truth_stay_on_the_raw_state() -> None:
    """The failure mode here is a plausible wrong number, so it is asserted directly.

    A reward re-based onto phi would be scoring a different program from the one
    the model wrote (LIMEN's prompt hands both functions the same `state`), and
    a `task_metric` re-based onto phi would let a candidate move the ground
    truth by choosing what it observes. Both are checked by WIDTH, because a
    6-column trajectory cannot be a 3-feature one.
    """
    ctx = _ctx("limen")
    cand = _candidate(_OBS_3)
    result = registry.get("train_backend", "mock")(ctx, RunState(), cand, n_seeds=1)

    assert result.trained, result.error
    assert cand.meta["observation_dim"] == 3, "precondition: phi was installed"
    assert result.trajectories, "no rollout to inspect"
    for traj in result.trajectories:
        states = np.asarray(traj.states)
        assert states.shape[1] == ctx.env.obs_dim == 6, (
            f"the recorded trajectory is {states.shape[1]} wide, i.e. in FEATURE "
            "space. Every downstream reader -- task_metric, reference_reward, the "
            "EPIC/STARC screens, the preference comparator, the rollout video -- "
            "is written against the state.")


def test_the_view_delegates_dynamics_and_overrides_only_the_feature_space() -> None:
    """`_ObsView` is a policy-feature view, not an environment.

    Stated as a test because the tempting implementation -- a wrapper whose
    `reset`/`step` return phi(s) -- is unimplementable here rather than merely
    wrong: `EnvAdapter.step(state, action)` is state-PASSING, so a caller
    holding phi(s) has nothing to step with.
    """
    env = _env()
    phi = training.compile_observation(_OBS_3, env)
    view = training._ObsView(env, phi)

    rng = np.random.default_rng(0)
    s = view.reset(rng)
    assert np.asarray(s).shape == (env.obs_dim,), (
        "`reset` returned features; the reward and the dynamics both need the state")
    s2, _done, _info = view.step(s, view.action_set[0])
    assert np.asarray(s2).shape == (env.obs_dim,)

    assert view.policy_features(s).shape == (3,)
    assert view.obs_dim == 3 and view.obs_low.shape == view.obs_high.shape == (3,)
    assert view.horizon == env.horizon and view.n_actions == env.n_actions


def test_the_view_refuses_to_claim_an_exact_state_space() -> None:
    """`exact_states is not None` promises `discretise` is a BIJECTION.

    phi is a model-authored map with no injectivity guarantee -- Singh's 64
    hungry-thirsty states can collapse to three features -- so a view that
    inherited `exact_states` would make `train.backend: tabular`'s one honest
    exactness claim a false one.
    """
    registry.load_all()
    env = registry.get("env", "toy_hungry_thirsty")({})
    assert env.exact_states is not None, "precondition: this env enumerates"
    view = training._ObsView(env, training.compile_observation(
        "def get_observation(state):\n    return [state[0], state[1]]\n", env))
    assert view.exact_states is None
    assert view.n_disc_states <= env.n_disc_states, (
        "the binning grew the table: `n_disc_states` is a memory claim "
        "(`_QLearner` allocates n_disc_states x n_actions)")


def test_a_narrower_observation_gives_the_linear_searcher_a_narrower_weight_matrix() -> None:
    """The install has to reach the POLICY CLASS, not only a recorded number.

    `_CEMLearner` sizes its weights `(action_dim, obs_dim + 1)`, so this is the
    one check that would fail if `_ObsView` were built, recorded, and then never
    handed to `_make_learner`.
    """
    env = _env()
    phi = training.compile_observation(_OBS_3, env)
    view = training._ObsView(env, phi)
    reward = training.compile_reward(_REWARD)
    rng = np.random.default_rng(0)

    def make(on):
        return training._make_learner(
            "auto", on, reward, rng, training._RewardNorm("none"),
            lambda exc: 0.0, 400, None, (), 0.0, 1)

    assert make(view).dim == (env.action_dim, 3 + 1)
    assert make(env).dim == (env.action_dim, env.obs_dim + 1)
    assert make(view).policy_input_dim() == 3
    assert make(env).policy_input_dim() == env.obs_dim


#: Every function inside `_run_backend` that turns stored parameters into a
#: policy, or builds a learner. All of them must be handed the FEATURE-SPACE env.
_POLICY_BUILDERS = ("_make_learner", "_greedy_policy", "_linear_policy")


def test_every_policy_builder_in_run_backend_gets_the_feature_space_env() -> None:
    """A source invariant, because the runtime path that breaks it needs a fork.

    THE FAILURE IS REAL AND EASY TO WRITE. Under
    `train.candidate_parallelism: parallel`, `final_retrain` forks one process
    per seed and the PARENT rebuilds each child's policy from the snapshot it
    sent home -- `_greedy_policy` / `_linear_policy` on a closed kind map. If
    either line passes the raw adapter, a snapshot shaped for a 3-feature phi
    meets a 7-wide `[s; 1]` and the final rollouts die on a matmul shape error.
    `tests/test_parallelism.py[limen]` catches it, but that file is `slow`: it
    does not run on a pull request.

    Checked on the SOURCE and not by running it, deliberately. Reproducing it
    needs a forked seed schedule and a real `final_retrain`, which is
    heavyweight execution and belongs in the file that already has it. What is
    cheap is the shape of the call, and the shape is the whole bug: there are
    only three functions in this module that build something the env's feature
    width has to match, and every one of them must be handed `learn_env`. This
    is the `test_no_method_branching.py` idiom -- an AST grep where the runtime
    check is expensive and the syntactic property is exactly equivalent.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(training._run_backend))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in _POLICY_BUILDERS:
            continue
        # EVERY argument, positional AND keyword, matched on the UNPARSED
        # expression rather than on `isinstance(arg, ast.Name)`. Checking
        # positional `ast.Name` args only for the literal id `env` is too weak
        # in two directions at once: `_make_learner(kind, env=env, ...)` is a
        # keyword and `_make_learner(kind, ctx.env, ...)` is an Attribute, so
        # either regression would reappear against a green test. An AST check
        # standing in for an expensive runtime one has to be at least as strict
        # as the property it stands for, or it is worse than no check -- it is a
        # green light over the exact defect.
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            expr = ast.unparse(arg)
            if expr == "env" or expr.endswith(".env"):
                bad.append(f"{node.func.id}(... {expr} ...) at line {node.lineno} "
                           "of the function")

    # THE PROBE IS CHECKED BEFORE ITS VERDICT IS BELIEVED. A green AST grep and a
    # broken AST grep look identical: a version that checked positional
    # `ast.Name` args only would let `env=env` and `ctx.env` both sail
    # through. `_flag` is the same code path the loop above
    # runs, so a future weakening fails here rather than going quiet.
    def _flag(src: str) -> list:
        found = []
        for n in ast.walk(ast.parse(src)):
            if not isinstance(n, ast.Call) or not isinstance(n.func, ast.Name):
                continue
            if n.func.id not in _POLICY_BUILDERS:
                continue
            for a in list(n.args) + [k.value for k in n.keywords]:
                e = ast.unparse(a)
                if e == "env" or e.endswith(".env"):
                    found.append(e)
        return found

    assert _flag("_make_learner(kind, learn_env, r)") == []
    assert _flag("_make_learner(kind, env, r)") == ["env"]
    assert _flag("_make_learner(kind, env=env, reward=r)") == ["env"]
    assert _flag("_linear_policy(ctx.env, w)") == ["ctx.env"]
    assert _flag("_greedy_policy(self.env, q)") == ["self.env"]
    assert _flag("some_other_call(env)") == [], "the check must be scoped to builders"

    assert not bad, (
        "a policy builder inside `_run_backend` was handed the raw adapter:\n  "
        + "\n  ".join(bad)
        + "\nUnder `problem.search_space` with 'observation' the learner and every "
          "policy rebuilt from its snapshot live in phi's feature space, so these "
          "take `learn_env`. `env` stays correct for `_evaluate_policy`, `_rollout` "
          "and `set_dr`, which is why the two names exist.")


def test_the_mock_never_falls_back_to_toy_indices_when_the_prompt_named_slots() -> None:
    """A populated field table must be USED, even when no named feature resolves.

    `_OBS_FEATURES` recognises `x`/`vx`/`goal_x` and their aliases. An env whose
    stub names `cart_position`/`pole_angle` matches none of them -- and falling
    back to the unresolved list there, i.e. to `toy_reacher`'s indices, would
    reintroduce the exact wrong-slot problem the resolver exists to remove, in
    the one case where the prompt DID carry enough to build a correct
    observation.

    Two directions, because the fallback is right in one of them: a prompt with
    NO table (`env_spec: none`, or prose only) has nothing to resolve against and
    keeps the unresolved list.
    """
    from bird.llm import mock as M
    import random

    stub = ("@dataclass\nclass State:\n"
            "    cart_position: float   # s[0] -- cart x\n"
            "    cart_velocity: float   # s[1] -- cart xdot\n"
            "    pole_angle: float   # s[2] -- pole theta\n"
            "    pole_velocity: float   # s[3] -- pole thetadot\n")
    assert M._state_field_index(stub), "precondition: the table parses"

    for seed in range(6):
        prog = M._observation_program(random.Random(seed), 512, stub)
        assert "goal_x" not in prog and "tip_x" not in prog, (
            "the mock fell back to toy_reacher's feature list on an env whose "
            "state it could read:\n" + prog)
        for name in ("cart_position", "cart_velocity", "pole_angle", "pole_velocity"):
            if name in prog:
                break
        else:
            raise AssertionError("no field from this env's own table:\n" + prog)

    # No table at all -> the unresolved list, which is the only thing available.
    bare = M._observation_program(random.Random(0), 512, "prose, no field table")
    assert "goal_x" in bare


@pytest.mark.parametrize("dim", [1, 2, 3, 6, 12])
@pytest.mark.parametrize("budget", [2, 4, 64, 324, 3600])
def test_the_view_never_grows_the_table_it_inherited(dim: int, budget: int) -> None:
    """`n_disc_states` is a memory claim, so the view's must not exceed the env's.

    `_QLearner` allocates `n_disc_states x n_actions` up front. Deriving
    `bins = max(2, budget ** (1/dim))` and then declaring `bins ** dim` would
    OVERSHOOT whenever the budget is small, because the floor of 2 exists (one
    bin per axis is not a binning). An env advertising 2 (or none, the
    `getattr` default) with a 3-feature phi would ask for 8 cells against 2;
    with a 30-feature phi, 2**30.

    Parametrised over both axes rather than spot-checked, because the failure
    lives at the SMALL end of the budget and the LARGE end of the width, and a
    single realistic pair (324, 3) misses it in both directions.
    """
    class _Env:
        name = "fake"
        n_disc_states = budget
        n_actions = 2
        obs_dim = 16
        obs_low = np.full(16, -1.0)
        obs_high = np.full(16, 1.0)

    body = "def get_observation(state):\n    return [" + \
           ", ".join(f"state[{i}]" for i in range(dim)) + "]\n"
    env = _Env()
    view = training._ObsView(env, training.compile_observation(body, env))

    assert view.obs_dim == dim, "the feature width is phi's, whatever the table does"
    assert view.n_disc_states <= max(2, budget), (
        f"a {dim}-feature phi on a {budget}-cell env declared "
        f"{view.n_disc_states} cells")
    assert view.n_disc_states >= 1
    # And the index it produces must land inside what it declared, or `_QLearner`
    # indexes out of its own array.
    rng = np.random.default_rng(0)
    for _ in range(25):
        s = rng.uniform(-2.0, 2.0, 16)
        assert 0 <= view.discretise(s) < view.n_disc_states


def test_the_view_and_the_raw_env_build_differently_shaped_policies() -> None:
    """The runtime half of the check above, for the part that costs nothing."""
    env = _env()
    view = training._ObsView(env, training.compile_observation(_OBS_3, env))
    raw_state = np.zeros(env.obs_dim)             # what `_rollout` hands a policy

    w = np.ones((int(env.action_dim), 3 + 1))     # a child's `cem_linear` snapshot
    assert np.shape(training._linear_policy(view, w)(raw_state)) == (env.action_dim,)
    with pytest.raises(ValueError):
        training._linear_policy(env, w)(raw_state)


# --------------------------------------------------------------------------
# 4. the width is measured, never inferred
# --------------------------------------------------------------------------


def test_the_width_is_measured_and_not_estimated() -> None:
    """A width the AST cannot read must still come out right.

    Both existing readers give 2 for `np.concatenate([near, vel])` against a
    true width of 4 -- `selection._literal_return_width` says so in its own
    docstring and `generation._estimate_obs_dim` takes `len` of the first
    literal it finds. Measuring is what makes the archive axis and the
    `observation_max_dim` cap mean what they say.
    """
    from bird.components import generation, selection

    assert selection._literal_return_width(_OBS_CONCAT) == 2.0, (
        "the heuristic changed; this test's premise is that it under-reads")
    _src, estimated = generation._extract_observation(_OBS_CONCAT)
    assert estimated == 2, estimated

    phi = training.compile_observation(_OBS_CONCAT, _env())
    assert phi.dim == 4


#: What a real model actually writes, and what the mock cannot. Both functions
#: come out of ONE code block (LIMEN's prompt says so in as many words), so a
#: shared module-level helper is the normal shape of an answer -- and
#: `_apply_co_design` lifts only the `def get_observation` out of it.
_SHARED_HELPER_PROGRAM = """
import math

_OFFSET = 0.045


def _tcp(state):
    return state[0], state[1], state[2] - _OFFSET


def compute_reward(state, action, next_state):
    x, y, z = _tcp(next_state)
    return -math.sqrt((state[4] - x) ** 2 + (state[5] - y) ** 2 + z * z)


def get_observation(state):
    x, y, z = _tcp(state)
    return [state[4] - x, state[5] - y, z]
"""

#: Exactly what `_apply_co_design` records in `candidates/<id>/observation.py`:
#: `ast.get_source_segment` of the `get_observation` node, and nothing else.
_LIFTED_SEGMENT = """def get_observation(state):
    x, y, z = _tcp(state)
    return [state[4] - x, state[5] - y, z]"""


def test_a_lifted_observation_can_reach_the_programs_helpers() -> None:
    """The lift is a VIEW onto the program, so compiling it needs the program.

    A REAL GENERATOR PRODUCES THIS SHAPE ROUTINELY: in a `limen` run with a real
    LLM on `mt10_reach-v3`, every candidate wrote a `_tcp(state)` helper -- the
    tool-centre-point offset the adapter's own field docs point at -- and called
    it from both functions. `_apply_co_design` lifts only the `def
    get_observation`, so without the program's namespace the helper is left
    behind and every candidate dies with `NameError: name '_tcp' is not
    defined` on every probe state; the install refuses them correctly given
    what it is handed, and the search produces nothing.

    WHY NO MOCK-DRIVEN TEST CAN CATCH IT. The mock inlines its `_f` accessor
    into `get_observation` on purpose -- `_OBS_ACCESSOR` exists so the lifted
    segment is self-contained -- so the double satisfies a contract more strictly
    than any real model does, and every test that goes through the mock is blind
    to this. Hence the literals above rather than a generated program: the case
    has to be supplied because the double cannot produce it.
    """
    env = _env()
    cand = Candidate(cand_id="helper", iteration=0,
                     reward_code=_SHARED_HELPER_PROGRAM,
                     observation_code=_LIFTED_SEGMENT)

    # The segment ALONE cannot compile, and asserting it keeps this test honest
    # about what it is testing.
    with pytest.raises(Exception, match="_tcp"):
        training.compile_observation(_LIFTED_SEGMENT, env)

    phi = training.compile_observation(_LIFTED_SEGMENT, env, cand)
    assert phi.dim == 3
    assert np.shape(phi(np.zeros(env.obs_dim))) == (3,)


def test_the_lifted_segment_wins_over_the_programs_copy() -> None:
    """The recorded `observation.py` is what defines the function, not the program.

    Order matters because `_apply_co_design` calls `_apply_symbol_mapping` on
    `observation_code` alone (Text2Reward's general->specific table), so the
    segment can legitimately differ from the text still sitting in
    `reward_code`. Executing the program second would silently install the
    UNMAPPED version while `candidates/<id>/observation.py` recorded the mapped
    one -- an artifact that disagrees with the run.
    """
    env = _env()
    program = _SHARED_HELPER_PROGRAM  # its get_observation returns 3 features
    mapped = """def get_observation(state):
    x, y, z = _tcp(state)
    return [state[4] - x, state[5] - y, z, _OFFSET]"""
    cand = Candidate(cand_id="mapped", iteration=0, reward_code=program,
                     observation_code=mapped)
    phi = training.compile_observation(mapped, env, cand)
    assert phi.dim == 4, "the program's own definition won over the recorded segment"


def test_a_program_that_will_not_execute_is_not_fatal_to_the_segment() -> None:
    """A self-contained segment must still install when the program is broken.

    Stage 3 refuses the candidate before this if the REWARD would not compile, so
    reaching here with a broken program means something odder -- a module-level
    statement that raises on import, say. The segment is what the policy needs,
    so a program that cannot build the namespace is a debug line and the fallback
    is the segment alone.
    """
    env = _env()
    cand = Candidate(cand_id="broken", iteration=0,
                     reward_code="raise RuntimeError('boom')\n" + _OBS_3,
                     observation_code=_OBS_3)
    phi = training.compile_observation(_OBS_3, env, cand)
    assert phi.dim == 3


def test_a_variable_width_observation_is_refused() -> None:
    """A policy's input layer is fixed before the first step.

    So this is a refusal and not a truncation: silently keeping the first `k`
    features would train a network on a different function of the state
    depending on where in the state space an episode started.
    """
    code = """
def get_observation(state):
    out = [state[0], state[1]]
    if state[2] > 0.0:
        out.append(state[3])
    return out
"""
    with pytest.raises(ValueError, match="cannot vary per step"):
        training.compile_observation(code, _env())


def test_an_observation_that_will_not_run_fails_the_candidate() -> None:
    """Same treatment a reward that will not compile gets, and for the same reason.

    Not a degrade-to-raw-state path: falling back would convert this candidate
    into a `limen_reward_only` candidate inside a `limen` run, which is exactly
    the confusion the install exists to end. The RL job is counted as spent
    (`execute_rate` and the budget stay honest) and the reason is recorded.
    """
    ctx = _ctx("limen")
    cand = _candidate("def get_observation(state):\n    return state[99]\n")
    result = registry.get("train_backend", "mock")(ctx, RunState(), cand, n_seeds=1)

    assert not result.trained
    assert "observation" in result.error.lower() or "index" in result.error.lower(), result.error
    assert cand.meta["observation_installed"] is False
    assert cand.meta["observation_error"]
    assert ctx.budget.snapshot()["policy_trainings"] == 1, (
        "a launched-and-died RL job must still count against the budget")


def test_the_cap_holds_against_the_measurement_not_the_estimate() -> None:
    """`observation_max_dim` enforced against a lower bound is not a cap.

    §1 checks the model's declared `OBS_DIM` or an AST estimate, and the estimate
    under-reads by construction. This is the same constraint applied to the width
    the policy actually gets -- which is where LIMEN applies it too
    (`interface.validate`, called from the crash filter).
    """
    ctx = _ctx("limen", **{"generate.co_design.observation_max_dim": 3})
    cand = _candidate(_OBS_CONCAT)          # AST says 2, truth is 4
    result = registry.get("train_backend", "mock")(ctx, RunState(), cand, n_seeds=1)

    assert not result.trained
    assert "observation_max_dim" in result.error, result.error
    assert cand.meta["observation_installed"] is False
    # And the cap does not fire on a width that is under it.
    ok = _candidate(_OBS_3)
    assert registry.get("train_backend", "mock")(ctx, RunState(), ok, n_seeds=1).trained


def test_the_planner_backend_refuses_an_observation_rather_than_recording_one() -> None:
    """`train.algorithm: none` (L2R) has no policy, so it has no feature space.

    A shooting planner re-plans over raw states every step; there is nothing for
    phi to be the input of. Recording `observation_dim` there would put a
    co-design in the artifact that no consumer exists for -- the fabricated pin
    in miniature, and the harder kind to notice because the number would be
    right about phi and wrong about the run.
    """
    ctx = _ctx("limen")
    cand = _candidate(_OBS_3)
    result = registry.get("train_backend", "none")(ctx, RunState(), cand, n_seeds=1)

    assert result.trained, result.error
    assert result.seed_metrics[0]["observation_dim"] is None
    assert cand.meta["observation_installed"] is False
    assert "no policy" in cand.meta["observation_error"]
    assert "observation_dim" not in cand.meta, (
        "a width was recorded for a backend that consumed no observation")


def test_the_probe_does_not_disturb_the_run(monkeypatch) -> None:
    """Measuring phi must not advance the env's RNG or resample its DR.

    Otherwise installing an observation would change the reward search around
    it, and `limen` vs `limen_reward_only` would differ for a second reason
    nobody chose.
    """
    env = _env()
    calls = []
    monkeypatch.setattr(type(env), "reset",
                        lambda self, rng=None: calls.append("reset") or np.zeros(self.obs_dim))
    monkeypatch.setattr(type(env), "random_state",
                        lambda self, rng: calls.append("random_state") or np.zeros(self.obs_dim))
    training.compile_observation(_OBS_3, env)
    assert calls == [], f"the probe touched the adapter: {calls}"


# --------------------------------------------------------------------------
# 5. archive semantics
#
# Every regression here renders as a PLAUSIBLE archive -- wrong membership,
# wrong occupants, an inverted prompt -- which is what justifies a test:
# nothing on screen distinguishes an archive built by two disagreeing binnings
# from one built by LIMEN's.
# --------------------------------------------------------------------------

_REWARD_BIG = """
def compute_reward(state, action, next_state):
    dx = next_state[4] - next_state[0]
    dy = next_state[5] - next_state[1]
    d = (dx * dx + dy * dy) ** 0.5
    bonus = 0.5 if d < 0.1 else 0.0
    shaping = -0.01 * (action[0] * action[0] + action[1] * action[1])
    return -d + bonus + shaping
"""


def _trained_report(ctx, cand_id: str, fitness: float, reward=_REWARD):
    from bird.types import CandidateReport, TrainResult
    cand = Candidate(cand_id=cand_id, iteration=0, reward_code=reward)
    res = TrainResult(cand_id=cand_id, candidate=cand, trained=True)
    rep = CandidateReport(cand_id=cand_id, candidate=cand, result=res,
                          fitness_source="success_rate")
    rep.fitness = fitness
    return rep


def test_stage_5_coords_are_exactly_the_key_stage_6_stores() -> None:
    """ONE binning. The cell the win verdict was computed against must be the
    dict key the report ends up under. If stage 5 binned on fixed ranges and
    stage 6 re-binned the whole archive adaptively, the verdict and the insert
    would happen on two unrelated grids.
    """
    ctx = _ctx("limen")
    state = RunState()
    reports = [_trained_report(ctx, "c0000", 0.4),
               _trained_report(ctx, "c0001", 0.7, reward=_REWARD_BIG)]

    selection = registry.get("select_rule", "map_elites_insert")(ctx, state, reports)
    state = registry.get("topology", "archive_map_elites")(ctx, state, selection)

    assert selection.winners, "two measured reports and an empty archive: someone wins"
    for w in selection.winners:
        key = (w.meta["archive_island"],) + tuple(w.meta["archive_coords"])
        cell = state.archive.get(key)
        assert cell is not None, (
            f"{w.cand_id} won cell {key} in stage 5 and stage 6 stored it "
            f"somewhere else: keys={sorted(state.archive)}")
    # The two rewards differ in compute_reward AST size, so once binned they
    # must not share a complexity coordinate under the adaptive stats.
    a, b = reports
    assert a.meta["archive_coords"] != b.meta["archive_coords"], (
        "two structurally different rewards collapsed into one cell: the "
        "adaptive stats did not separate what they measured apart")
    # And the stats are RUNNING state, carried with the archive.
    assert state.descriptor_stats, "binning left no running stats behind"


def test_a_sentinel_never_occupies_a_cell_and_a_measured_reject_does() -> None:
    """Both halves. An invalid candidate's -10000 sentinel must not seed an
    empty niche (it would win any empty cell and parent the rest of the run);
    a cascade reject whose screen MEASURED a rate enters at exactly
    that rate, which is the release's SHORT_TRAIN_REJECTED population.
    """
    from bird.types import TrainResult
    ctx = _ctx("limen")
    state = RunState()

    # Half 1: never compiled -> sentinel -> excluded from the contest.
    bad = Candidate(cand_id="c0000", iteration=0, reward_code="def broken(").failed(
        "syntax error")
    bad_res = TrainResult(cand_id="c0000", candidate=bad, trained=False,
                          skip_reason="invalid")
    [bad_rep] = registry.get("fitness_source", "success_rate")(ctx, RunState(), [bad_res])
    assert bad_rep.fitness == ctx.cfg["select.failure_value"], "precondition: sentinel"

    # Half 2: cascade-screened WITH a measured short-run rate.
    screened = Candidate(cand_id="c0001", iteration=0, reward_code=_REWARD)
    screened.screened_out = True
    screened.failure_kind = "screened"
    screened.failure = "cascade: success 0.0050 < min_success_threshold 0.01"
    screened.verify_records.append({"phase": "screen", "ok": False,
                                    "screen": "cascade", "success_rate": 0.005})
    scr_res = TrainResult(cand_id="c0001", candidate=screened, trained=False,
                          skip_reason="screened")
    [scr_rep] = registry.get("fitness_source", "success_rate")(ctx, RunState(), [scr_res])
    assert scr_rep.fitness == 0.005, (
        f"evaluate.screened_fitness: screen_measurement should carry the "
        f"screen's own rate; got {scr_rep.fitness!r}")
    assert scr_rep.meta.get("fitness_from_screen") is True

    selection = registry.get("select_rule", "map_elites_insert")(
        ctx, state, [bad_rep, scr_rep])
    state = registry.get("topology", "archive_map_elites")(ctx, state, selection)

    assert bad_rep.meta["archive_cell_win"] is False
    assert bad_rep.meta.get("archive_excluded"), "the exclusion must be named"
    occupants = {c.report.cand_id: c.fitness for c in state.archive.values()
                 if c.report is not None}
    assert "c0000" not in occupants, "a -10000 junk elite occupies a niche"
    assert occupants.get("c0001") == 0.005, (
        f"the measured reject should hold its cell at its measured rate: "
        f"{occupants}")
    # The excluded report must not have moved the running stats either
    # (the release's crash-filter failures never reach _bin_features).
    seen = state.descriptor_stats.get("reward_ast_node_count", {})
    assert seen.get("min") == seen.get("max"), (
        "two different programs' AST counts entered the stats but only one "
        "report was eligible to be binned")


def test_a_trained_cell_loser_is_a_parent_not_a_failure() -> None:
    """A trained program that lost its cell contest stays out of
    `failure_memory` (the FAILED framing would invert the training signal) and is
    filed into its island's population, where the island parent branches can
    still draw it -- database.py:294-299 'Still store it'.
    """
    ctx = _ctx("limen")
    state = RunState()
    winner = _trained_report(ctx, "c0000", 0.9)
    loser = _trained_report(ctx, "c0001", 0.4)  # same code -> same cell

    selection = registry.get("select_rule", "map_elites_insert")(
        ctx, state, [winner, loser])
    assert [w.cand_id for w in selection.winners] == ["c0000"]
    assert [l.cand_id for l in selection.losers] == ["c0001"]
    state = registry.get("topology", "archive_map_elites")(ctx, state, selection)
    for l in selection.losers:
        registry.get("loser_action", "store_as_negative_example")(ctx, state, l)

    assert not any(e["cand_id"] == "c0001" for e in state.failure_memory), (
        "a trained candidate that merely lost its niche was taught to the "
        "LLM as FAILED")
    from bird.components import update as U
    filed = [r.cand_id for isl in range(3) for r in U._population(state, isl)]
    assert "c0001" in filed, "the loser vanished from every island population"

    # Reachable: the island branches sample over the island's whole holding.
    from bird.components.generation import parent_archive_sample
    ctx2 = _ctx("limen", **{"update.archive.parent_sampling": {"island_uniform": 1.0},
                            "generate.n_parents": 1})
    drawn = set()
    for seed in range(60):
        ctx2.rng.seed(seed)
        drawn.update(r.cand_id for r in parent_archive_sample(ctx2, state))
    assert "c0001" in drawn, (
        f"60 island-uniform draws never produced the filed loser: {drawn}")

    # And a 0.0-fitness loser is BOTH samplable and a negative example
    # (database.py:236-243 adds to recent_failures without returning).
    zero = _trained_report(ctx, "c0002", 0.0)
    registry.get("loser_action", "store_as_negative_example")(ctx, state, zero)
    assert any(e["cand_id"] == "c0002" and e["failure"] for e in state.failure_memory)
    filed = [r.cand_id for isl in range(3) for r in U._population(state, isl)]
    assert "c0002" in filed


def test_a_trained_cell_loser_with_no_fitness_scalar_is_still_a_parent() -> None:
    """The unranked half, and the reason it is its own test.

    A guard written `if measured_fitness(report) and fitness is not None:`
    breaks here, because `.fitness` is `Optional[float]`: under
    `evaluate.fitness.source: none` (or `problem.fitness_access: none`) a
    candidate trains perfectly cleanly and has NO ranking scalar, so it would
    fail the second conjunct and fall through to `failure_memory` with
    `failure: ""` -- the whole inverted training signal back, restricted to the
    GT-free configs. `test_a_trained_cell_loser_is_a_parent_not_a_failure`
    above only ever builds reports with a float, and no shipped config pairs
    the two (`l2r`, `text2reward_human` and `card` run on fitness `none` and
    none of them takes this loser action; `limen`/`limen_reward_only` do and
    both are `success_rate`). A preference-driven method point would pair them
    the day it is added, which is why this is pinned rather than commented.

    Asserted on the SAME three observables as the float case -- absent from
    `failure_memory`, present in an island population, drawable as a parent --
    because the defect is not that the call raises. It does not raise: it
    produces a complete, plausible run in which a whole trained population is
    taught to the LLM as FAILED while every counter reads normal.
    """
    from bird.components import update as U

    ctx = _ctx("limen")
    state = RunState()
    unranked = _trained_report(ctx, "c0001", 0.0)
    unranked.fitness = None          # what `fitness_source: none` produces
    unranked.fitness_source = "none"

    assert U.measured_fitness(unranked), (
        "a cleanly trained candidate is a MEASUREMENT whether or not the run "
        "computes a scalar; if this flips, the test below stops testing the "
        "dispatch and starts testing the flag")

    registry.get("loser_action", "store_as_negative_example")(ctx, state, unranked)

    assert not any(e["cand_id"] == "c0001" for e in state.failure_memory), (
        "a trained candidate with no ranking scalar was taught to the LLM as "
        "FAILED; `generation._failure_traces` renders it under 'These programs "
        "were generated earlier and FAILED' with '- error: (unrecorded)' "
        f"because its failure string is empty: {state.failure_memory}")

    filed = [r.cand_id for isl in range(3) for r in U._population(state, isl)]
    assert "c0001" in filed, (
        f"the unranked loser vanished from every island population: {filed}")

    from bird.components.generation import parent_archive_sample
    ctx2 = _ctx("limen", **{"update.archive.parent_sampling": {"island_uniform": 1.0},
                            "generate.n_parents": 1})
    drawn = set()
    for seed in range(60):
        ctx2.rng.seed(seed)
        drawn.update(r.cand_id for r in parent_archive_sample(ctx2, state))
    assert "c0001" in drawn, (
        f"60 island-uniform draws never produced the filed unranked loser: "
        f"{drawn}")

    # `None` is not `0.0`. The 0.0 case is the release's double bookkeeping
    # (database.py:236-243) and the unranked case is not it -- if a fix ever
    # coerces the scalar, this entry appears and CARD's "ranks nothing at all"
    # stops being expressible.
    assert not any(e["cand_id"] == "c0001" for e in state.failure_memory)
    cell = next(c for c in state.archive.values()
                if c.report is not None and c.report.cand_id == "c0001")
    assert cell.fitness == U.NEG_INF, (
        f"the unranked loser was filed at {cell.fitness!r}; it must sort at "
        "the unranked marker so it cannot displace a ranked incumbent")
    assert cell.report.fitness is None, "the report's own None was overwritten"


def test_archive_size_caps_the_sampling_pool_and_never_the_grid() -> None:
    """Filling more cells than `archive_size` must evict nothing --
    the released cap bounds the exploitation SAMPLING list (database.py:739-764
    edits only the top-50 list; the grid is untouched). The regression renders
    as a plausible, smaller archive, so it is asserted on the numbers.
    """
    from bird.components import update as U
    ctx = _ctx("limen", **{"update.archive.archive_size": 3,
                           "update.archive.n_islands": 1,
                           "update.archive.migration_rate": 0.0})
    state = RunState()
    # Extremes first, so the running min/max spreads the later values across
    # bins instead of clamping each new maximum into the top one.
    for i, ast_nodes in enumerate([0.0, 500.0, 100.0, 200.0, 300.0, 400.0]):
        rep = _trained_report(ctx, f"c{i:04d}", 0.1 * i)
        rep.descriptors["reward_ast_node_count"] = ast_nodes
        rep.descriptors["observation_dim"] = 6.0
        U.insert_into_archive(ctx, state, rep)

    grid = [c for c in state.archive.values() if c.report is not None]
    assert len(grid) > 3, (
        f"only {len(grid)} occupied cells: the grid was evicted to the cap, "
        "which destroys the niching MAP-Elites exists to maintain")
    pool = U.exploitation_pool(ctx, grid)
    assert len(pool) == 3
    assert [c.report.cand_id for c in pool] == ["c0005", "c0004", "c0003"], (
        "the pool is not the top-archive_size by fitness")
