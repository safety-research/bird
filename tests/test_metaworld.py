"""The Meta-World MT10 adapter (`bird/envs/metaworld.py`).

Two halves, split by whether the simulator is installed, and the split is the
point rather than a convenience.

**Without metaworld** (the machine most of CI runs on) the module must still
import, register its ten ids, satisfy `problem.env_id`'s registry-backed schema,
and -- when someone does try to build one -- fail with a message that names the
extra to install. `registry.load_all()` imports every module in `_MODULES` and
forgives an ImportError only for `anthropic_client`, so a module-scope
`import mujoco` here would take down `--validate-all`, `--list-configs` and the
whole test suite on every machine without a simulator.

**With metaworld** the thing worth testing is not "does it step" but the one
property this adapter invents and the rest of the repo silently depends on:
`_step(s, a)` is a function *of the state passed in*. MuJoCo is stateful and a
39-D Meta-World observation cannot be inverted to `(qpos, qvel)`, so the adapter
carries an observation -> simulator-snapshot cache and restores before it steps.
Three harness call sites (`_PlannerLearner`, the `_sb3_run` evaluation
interleave, `sample_transitions`) hand it a state the simulator is not in. If
the restore is *approximately* right, nothing raises: the EPIC/STARC screens
just quietly compare reward functions on physics that never happened, and the
number lands in the artifact attributed to the candidate. So the restore tests
here assert **bitwise** equality, never `np.allclose` -- an approximate restore
is exactly the failure that has no other symptom.
"""

import inspect
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from conftest import REPO
from bird import registry
from bird.config import ConfigError, load
from bird.envs.metaworld import (_EXPECTED_METAWORLD, _EXPECTED_MT10,
                                 _EXPECTED_MT50_ONLY, MetaWorld, UnknownStateError,
                                 _env_id)

#: Not in the tester-tier smoke suite (dependency-gated: importorskip('metaworld'), which the default CI job does not install).
#: Deselected by `-m "not slow"`. See pyproject.toml.
pytestmark = pytest.mark.slow

#: One task carries the expensive proofs. `drawer-open-v3` rather than the more
#: obvious `reach-v3`: reach is the ONE task whose success check is an
#: approximation (`tcp_center` is not in the 39-D observation, so it is
#: reconstructed as hand + (0, 0, -0.045), which disagrees with the simulator on
#: 1-8 of every 300 expert steps, seed-dependent, typically within ~2 mm of the
#: 0.05 m boundary), and a contract test should be exact. drawer-open has an
#: exact check, a scripted expert that solves it, and a moving fixture body
#: -- so it exercises the mjMODEL half of the snapshot, which a free-body task
#: like push would not.
TASK = "drawer-open-v3"

#: The ten ids, derived the same way the module derives them, so the test cannot
#: drift from the registration loop without one of them disappearing.
ALL_IDS = [_env_id(t) for t in _EXPECTED_MT10]
#: The forty MT50-only tasks, registered as `mt50_<key>`.
MT50_IDS = [_env_id(t) for t in _EXPECTED_MT50_ONLY]


# --------------------------------------------------------------------------
# no simulator installed
# --------------------------------------------------------------------------

#: Run in a subprocess with `metaworld`/`mujoco` poisoned in `sys.modules`, which
#: is CPython's documented way to make `import x` raise. A subprocess and not
#: `monkeypatch.setitem` because two of the three claims are about *import time*
#: -- by the time this test runs, conftest has already imported the module for
#: real -- and because `MUJOCO_GL` is process-global state that a validation path
#: must not touch.
_NO_SIMULATOR_PROBE = r"""
import json, os, sys

sys.modules["metaworld"] = None
sys.modules["mujoco"] = None
os.environ.pop("MUJOCO_GL", None)

import bird.envs.metaworld as mw
from bird import registry

out = {
    "gl_after_import": os.environ.get("MUJOCO_GL"),
    "imported": sorted(m for m in ("metaworld", "mujoco", "gymnasium")
                       if sys.modules.get(m) is not None),
    "ids": sorted(n for n in registry.names("env") if n.startswith("mt10_")),
}
try:
    mw.MetaWorld("reach-v3")
except Exception as exc:
    out["error_type"] = type(exc).__name__
    out["error"] = str(exc)
else:
    out["error_type"] = None
    out["error"] = ""
print("PROBE" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def without_simulator() -> dict:
    from conftest import run_probe

    return run_probe(_NO_SIMULATOR_PROBE)


def test_the_module_imports_with_no_simulator_installed(without_simulator):
    """`registry.load_all()` tolerates ImportError for `anthropic_client` and
    nothing else, so an eager `import mujoco` here would break every CLI path and
    the whole suite on a machine with no simulator."""
    assert without_simulator["imported"] == [], (
        "importing the adapter module pulled in "
        f"{without_simulator['imported']}; the simulator imports belong inside "
        "MetaWorld.__init__, not at module scope")


def test_importing_the_module_does_not_touch_the_process_environment(without_simulator):
    """`MUJOCO_GL` is set in `__init__`, never at import: `--validate-all`,
    `--list-configs` and every pytest run import this module, and a validation
    path that mutates the environment changes how a later run renders."""
    assert without_simulator["gl_after_import"] is None


def test_all_ten_mt10_ids_register_without_the_simulator(without_simulator):
    assert without_simulator["ids"] == sorted(ALL_IDS)


def test_constructing_without_the_dependency_names_the_extra(without_simulator):
    """A bare `ModuleNotFoundError: No module named 'metaworld'` from four frames
    down tells an operator nothing about which config key asked for it."""
    assert without_simulator["error_type"] == "ImportError"
    message = without_simulator["error"]
    for expected in ("problem.env_id", "mt10_reach-v3", "metaworld", "mujoco",
                     "--extra metaworld"):
        assert expected in message, f"the install hint omits {expected!r}:\n{message}"


def test_the_ten_ids_are_registered_as_envs():
    """Ten ids, mechanically derived from the task strings by `_env_id` -- ten
    hand-written decorators would be ten chances for an id to disagree with the
    task it builds."""
    names = set(registry.names("env"))
    assert len(ALL_IDS) == 10
    missing = [n for n in ALL_IDS if n not in names]
    assert not missing, f"registered env ids are missing {missing}"


def test_the_forty_mt50_ids_are_registered_under_their_own_prefix():
    """The other forty of MT50 register as `mt50_<key>`, the ten keep
    `mt10_`, and no key appears under both: MT10 is a subset of MT50, and one task
    under two ids would be two config points for one environment -- the worst bug
    shape a unified config space can have."""
    names = set(registry.names("env"))
    assert len(MT50_IDS) == 40 and all(n.startswith("mt50_") for n in MT50_IDS)
    assert len(set(_EXPECTED_METAWORLD)) == 50
    missing = [n for n in MT50_IDS if n not in names]
    assert not missing, f"registered env ids are missing {missing}"
    assert not {n for n in names if n.startswith("mt50_")} & set(ALL_IDS)
    for key in _EXPECTED_MT10:
        assert "mt50_" + key not in names, f"{key} registered under both prefixes"


def test_the_prefix_rule_is_the_id_rule_in_tasks_py():
    """`_env_id` here and `TaskSpec.bird_env_id` in bird/tasks.py must derive the same
    id for every one of the fifty, or a spec would back an env the registry names
    differently -- which the coverage partition test would report as a spec with no
    adapter AND an adapter with no spec."""
    from bird import tasks

    for key in _EXPECTED_METAWORLD:
        spec = next(s for s in tasks.index().values()
                    if s.library == "metaworld" and s.env_id == key)
        assert spec.bird_env_id == _env_id(key), key


@pytest.mark.parametrize("env_id", ALL_IDS + MT50_IDS)
def test_problem_env_id_accepts_every_metaworld_id(env_id):
    """`problem.env_id` is `F("§0", (S,), kind="env")`, so registering the
    adapter IS the schema edit. This asserts the two agree in the direction that
    matters: a config naming any of the fifty Meta-World tasks validates."""
    cfg = load("eureka", overrides={"problem.env_id": env_id}, profile="tester")
    assert cfg["problem.env_id"] == env_id


def test_a_mistyped_mt10_id_is_still_fatal():
    """The other direction: registering ten names must not have opened the key
    up to anything that merely looks like one."""
    with pytest.raises(ConfigError):
        load("eureka", overrides={"problem.env_id": "mt10_drawer_open"},
             profile="tester")


def test_an_unknown_task_lists_the_ten():
    """Raised before the simulator import, so this holds everywhere."""
    with pytest.raises(KeyError) as excinfo:
        MetaWorld("open-the-pod-bay-doors-v3")
    assert "reach-v3" in str(excinfo.value)


# --------------------------------------------------------------------------
# with the simulator: fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simulator():
    """The gate for everything below.

    A fixture and NOT a module-level `pytest.importorskip`: at module scope the
    skip fires during collection and takes the no-simulator half of this file --
    which is the half that must run on a machine without metaworld -- with it.
    """
    return pytest.importorskip("metaworld", reason="needs `uv sync --extra metaworld`")


@pytest.fixture(scope="module")
def env(simulator) -> MetaWorld:
    """One adapter for the module. Construction is ~1.3 s (`MT1` pre-generates 50
    goals and briefly reseeds the global numpy RNG), and the cache it carries is
    never cleared, so sharing it is also closer to how the harness uses one."""
    return MetaWorld(TASK)


def _rollout(env: MetaWorld, seed: int, n: int, expert: bool = False):
    """`(states, actions, successes)` for one episode.

    The scripted expert is `metaworld.policies.ENV_POLICY_MAP[task]`, the
    benchmark's own solution: it is the only cheap way to reach states where
    `info["success"]` is 1, and a success check tested only on failures is
    tested on the constant it defaults to.
    """
    from metaworld.policies import ENV_POLICY_MAP

    rng = np.random.default_rng(seed)
    policy = ENV_POLICY_MAP[env.task]() if expert else None
    state = env.reset(rng)
    states, actions, successes = [state], [], []
    for _ in range(n):
        action = (policy.get_action(np.asarray(state)) if policy is not None
                  else env.action_set[int(rng.integers(env.n_actions))])
        state, done, info = env.step(state, action)
        assert done is False, "Meta-World does not terminate; the harness truncates"
        states.append(state)
        actions.append(np.asarray(action, dtype=float))
        successes.append(float(info["success"]))
    return np.asarray(states), actions, successes


def _sample_transitions(env: MetaWorld, rng, n: int):
    """`EnvAdapter.sample_transitions`, called by parameter NAME.

    Its positional order is `(rng, n)`; `verification._call_flexible` now binds
    by name, so either order works from the harness. Binding by name here means
    this test proves the adapter's behaviour rather than the base class's
    current argument order.
    """
    params = list(inspect.signature(env.sample_transitions).parameters)
    return (env.sample_transitions(rng, n) if params[0] == "rng"
            else env.sample_transitions(n, rng))


def test_the_screens_reach_this_adapters_sampler(env):
    """The EPIC/STARC screens must see REACHABLE Meta-World states.

    `verification.sample_transitions` degrades to 39-D Gaussian draws when it
    cannot call the adapter, and on this env that is not a weaker screen but no
    screen: a Gaussian is not a state the simulator ever emitted, so
    `reference_reward` raises `UnknownStateError` on every one of them, the
    screen reports itself inert, and `verify.alignment_filter.keep_top_n` never
    applies. Every rung of that ladder swallows its exception, so nothing else
    in the suite would notice.
    """
    import random

    from bird.components import verification

    ctx = types.SimpleNamespace(env=env, rng=random.Random(0))
    transitions = verification.sample_transitions(ctx, 6)
    assert len(transitions) == 6
    for tr in transitions:
        # The proof that these came from the adapter and not from the fallback:
        # `reference_reward` is cache-backed, so it can only score a state the
        # adapter itself emitted.
        env.reference_reward(tr.state, tr.action)


# --------------------------------------------------------------------------
# the GL stack and Triton
# --------------------------------------------------------------------------

#: Build an adapter (which sets `MUJOCO_GL` and imports mujoco), then do what
#: `train.backend: sb3` does a moment later: reach Triton's C extension through
#: `torch.optim.Adam`. In a subprocess because the failure mode is SIGSEGV --
#: there is no exception to catch and an in-process check would take the whole
#: suite with it.
_GL_TRITON_PROBE = r"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
from bird.envs.metaworld import MetaWorld
MetaWorld("reach-v3")
import torch
opt = torch.optim.Adam(torch.nn.Linear(4, 4).parameters())
opt.step()
print("SURVIVED")
"""


def test_the_adapter_and_triton_can_coexist_in_one_process(simulator):
    """Without the preload, the whole Meta-World tier dies at stage [3] train with exit 139.

    Not intermittent and not a rendering bug: the system `libOSMesa.so.8` is
    llvmpipe and pulls in `libLLVM.so.20.1` with global visibility, while
    `triton/_C/libtriton.so` embeds its own LLVM, so whichever loads second gets
    the other's symbols interposed. `train.backend: sb3` reaches Triton lazily
    from `torch.optim.Adam`'s constructor inside SAC's policy, which is AFTER the
    env adapter has been built -- exactly the losing order.
    `MetaWorld.__init__` therefore calls `_preload_llvm_before_mujoco()`.

    Pinned to `osmesa` on purpose: it is the adapter's default, and an operator
    or `scripts/setup_gl.sh` can set it explicitly, so "the machine happens to
    have a GPU EGL" must not be what keeps the tier alive.
    """
    pytest.importorskip("torch", reason="needs `uv sync --extra sb3`")
    pytest.importorskip("triton", reason="no triton in this install; "
                                         "the collision cannot happen")
    proc = subprocess.run([sys.executable, "-c", _GL_TRITON_PROBE],
                          cwd=str(REPO), env=dict(os.environ, PYTHONPATH=str(REPO)),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"exit {proc.returncode} (-11/139 is SIGSEGV): loading the software GL "
        "stack before Triton is a dynamic-linker LLVM collision, and it kills "
        "every Meta-World run at stage [3] train.\n"
        + proc.stdout + proc.stderr)
    assert "SURVIVED" in proc.stdout


# --------------------------------------------------------------------------
# what the adapter declares vs what it emits
# --------------------------------------------------------------------------


def test_the_declared_shapes_are_the_emitted_shapes(env):
    assert (env.obs_dim, env.action_dim, env.horizon) == (39, 4, 500)
    state = env.reset(0)
    assert state.shape == (env.obs_dim,)
    assert state.dtype == np.float64


def test_emitted_states_are_read_only(env):
    """A repo-wide convention, and it earns its place: Meta-World's own
    scripted door policy does `pos_door[0] -= 0.05` on a view into the
    observation, which poisons a cache keyed on that observation. Read-only turns
    that into an error at the mutation site instead of a miss three frames on."""
    state = env.reset(0)
    assert state.flags.writeable is False
    nxt, _done, _info = env.step(state, env.action_set[0])
    assert nxt.flags.writeable is False
    with pytest.raises(ValueError):
        nxt[0] = 1.0


def test_observations_stay_inside_the_declared_bounds(env):
    """`_bounds` reads `sawyer_observation_space`, not `observation_space`:
    metaworld 3.1.1 caches the latter while `_partially_observable` is still
    True, so it reports the three goal dims as low == high == 0 and every
    emitted state is out of bounds on them."""
    assert env.obs_low.shape == env.obs_high.shape == (env.obs_dim,)
    degenerate = np.flatnonzero(env.obs_high <= env.obs_low)
    assert degenerate.size == 0, f"degenerate observation dims: {degenerate.tolist()}"

    states, _actions, _successes = _rollout(env, seed=2, n=40, expert=True)
    below = np.flatnonzero((states < env.obs_low - 1e-9).any(axis=0))
    above = np.flatnonzero((states > env.obs_high + 1e-9).any(axis=0))
    assert below.size == above.size == 0, (
        f"out-of-bounds dims: below {below.tolist()}, above {above.tolist()}")


def test_the_action_set_spans_the_declared_action_box(env):
    """`action_low`/`action_high` are DERIVED from the set's per-axis extremes,
    so a set that never commands +-1 on an axis quietly shrinks the continuous
    action space SAC is given."""
    assert env.action_set.ndim == 2 and env.action_set.shape[1] == env.action_dim
    assert np.all(np.abs(env.action_set) <= 1.0)
    assert np.allclose(env.action_low, -1.0)
    assert np.allclose(env.action_high, 1.0)
    assert np.allclose(env.action_set[0], 0.0), (
        "the zero action must come first: it is `_run_backend`'s no-policy "
        "fallback and the planner's `best_first`")


def test_exact_states_is_none_so_sb3_gets_a_continuous_action_space(env):
    """`_sb3_run` derives `continuous` solely from `exact_states is None`. Any
    integer here silently hands SAC a `Discrete` space over the 15-row proxy set
    and turns "real SAC on Meta-World" into a 15-armed bandit."""
    assert env.exact_states is None


def test_discretise_lands_in_range_for_reachable_states(env):
    assert env.n_disc_states == 3600
    states, _actions, _successes = _rollout(env, seed=3, n=60, expert=True)
    indices = [env.discretise(s) for s in states]
    assert all(isinstance(i, int) for i in indices)
    assert all(0 <= i < env.n_disc_states for i in indices)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_two_adapters_with_the_same_seed_produce_identical_episodes(simulator):
    """Bitwise, not close. The construction route exists for this: `gym.make(
    'Meta-World/MT1')` wraps the env in a `RandomTaskSelectWrapper` that
    resamples the goal every reset and ignores `reset(seed=)`, so two candidates
    would not even be compared on the same task."""
    a, b = MetaWorld(TASK), MetaWorld(TASK)
    sa, sb = a.reset(np.random.default_rng(11)), b.reset(np.random.default_rng(11))
    assert np.array_equal(sa, sb)
    for i in range(15):
        action = a.action_set[i % a.n_actions]
        sa, _da, _ia = a.step(sa, action)
        sb, _db, _ib = b.step(sb, action)
        assert np.array_equal(sa, sb), f"diverged at step {i}"


def test_different_seeds_give_different_initial_conditions(env):
    """`_reset` picks one of the 50 pinned task instances with the episode RNG,
    so `cfg.seed` varies the initial conditions while the benchmark stays fixed.
    If it did not, `loop.n_restarts` would be n copies of one episode."""
    firsts = {env.reset(np.random.default_rng(s)).tobytes() for s in range(6)}
    assert len(firsts) > 1


# --------------------------------------------------------------------------
# the adapter against the live simulator
# --------------------------------------------------------------------------


def _raw_env(env: MetaWorld, seed: int):
    """A second, plain Meta-World env reset onto the same task instance.

    The instance is the one `_reset` drew, found by reproducing the draw and
    *verifying* it bitwise, with a scan over all 50 as the fallback -- so this
    test pins the adapter's observations against the simulator's, not against
    the adapter's own RNG bookkeeping.
    """
    import metaworld

    bench = metaworld.MT1(env.task, seed=MetaWorld.benchmark_seed)
    raw = bench.train_classes[env.task](
        render_mode="rgb_array", camera_name=MetaWorld.camera_name,
        width=MetaWorld.render_width, height=MetaWorld.render_width)
    reference = env.reset(np.random.default_rng(seed))
    guess = int(np.random.default_rng(seed).integers(len(bench.train_tasks)))
    for index in [guess] + list(range(len(bench.train_tasks))):
        raw.set_task(bench.train_tasks[index])
        obs, _info = raw.reset()
        if np.array_equal(np.asarray(obs, dtype=float), reference):
            return raw, reference
    raise AssertionError(
        "no pinned Meta-World task instance reproduces the adapter's reset "
        "observation; the adapter is not resetting onto the benchmark's goals")


def test_stepping_the_adapter_reproduces_the_live_env_exactly(env):
    """The adapter restores a snapshot before every step. That machinery must be
    invisible: sequentially, it has to return exactly what a plain Meta-World env
    driven with the same actions returns -- observation, success flag and (via
    `reference_reward`) the shaped reward, all bitwise."""
    raw, state = _raw_env(env, seed=5)
    rng = np.random.default_rng(7)
    for i in range(12):
        action = env.action_set[int(rng.integers(env.n_actions))]
        state, _done, info = env.step(state, action)
        obs, reward, _term, _trunc, raw_info = raw.step(
            np.clip(np.asarray(action, dtype=float), -1.0, 1.0))
        assert np.array_equal(state, np.asarray(obs, dtype=float)), f"obs differs at {i}"
        assert info["success"] == float(raw_info["success"])
        # `reference_reward` re-enters the UNTOUCHED `evaluate_state` from the
        # restored snapshot, so it must equal the reward `step` itself returned.
        assert env.reference_reward(state, action) == float(reward), f"reward differs at {i}"


def test_gt_reward_is_the_alias_the_screens_actually_probe(env):
    """`screens.reference_reward` probes
    `gt_reward|ground_truth_reward|true_reward|reward_fn|compute_reward` and NOT
    `reference_reward`. Without the alias EPIC, STARC and policy_rank_corr are
    inert on this env -- they log "no ground-truth reward" and return nothing,
    which reads as "the screen passed"."""
    states, actions, _successes = _rollout(env, seed=8, n=4)
    s, s2, a = states[0], states[1], actions[0]
    # BIRD's convention is that a reward scores the state it ARRIVES in.
    assert env.gt_reward(s, a, s2) == env.reference_reward(s2, a)
    assert env.gt_reward(s2, a) == env.reference_reward(s2, a)


# --------------------------------------------------------------------------
# arbitrary-state stepping: the property everything else rests on
# --------------------------------------------------------------------------


def test_restoring_a_state_reproduces_the_rollout_bitwise(env):
    """Save, step away, come back, replay: the hard one.

    The interleave in the middle is not decoration -- it is `_sb3_run`, where
    `_evaluate_policy` RESETS the shared adapter between `model.learn` chunks and
    SB3 then resumes from a state emitted before that reset, onto a different
    task instance. That is why the snapshot has to carry mjMODEL fields
    (`reset_model` rewrites `body_pos` and the goal site every episode) and not
    just `qpos`/`qvel`.
    """
    states, actions, successes = _rollout(env, seed=13, n=20, expert=True)

    # Drive the simulator somewhere else entirely: a full reset onto another
    # instance, then some steps of its own.
    other = env.reset(np.random.default_rng(99))
    for _ in range(5):
        other, _done, _info = env.step(other, env.action_set[1])

    resumed = states[10]
    for i, action in enumerate(actions[10:]):
        resumed, _done, info = env.step(resumed, action)
        expected = states[11 + i]
        assert np.array_equal(resumed, expected), (
            f"replay diverged at step {10 + i}: max|d| "
            f"{np.abs(resumed - expected).max():.3e} -- an approximate restore is "
            "the failure with no other symptom")
        assert info["success"] == successes[10 + i]


def test_sample_transitions_returns_reproducible_transitions(env):
    """`sample_transitions` is what feeds the EPIC/STARC screens, and it is the
    call site that hands `_step` a state with no relationship to the previous
    one. Each successor must be reproducible from its own predecessor *after* the
    simulator has been driven elsewhere -- otherwise a screen is comparing reward
    functions at states the simulator was never actually in."""
    transitions = _sample_transitions(env, np.random.default_rng(0), 12)
    assert len(transitions) == 12
    assert len({s.tobytes() for s, _a, _s2 in transitions}) == 12, (
        "random_state repeated itself; the coverage pool is not being refilled")

    for s, _a, s2 in transitions:
        assert s.shape == s2.shape == (env.obs_dim,)
        assert np.all(s >= env.obs_low - 1e-9) and np.all(s <= env.obs_high + 1e-9), (
            "sampled a state outside the observation box -- a uniform draw over "
            "39 dims would give quaternions of norm 3 and negative gripper openings")

    # Move the simulator far away, then replay every transition out of order.
    away = env.reset(np.random.default_rng(4))
    for _ in range(5):
        away, _done, _info = env.step(away, env.action_set[2])

    for i, (s, a, s2) in enumerate(reversed(transitions)):
        again, _done, _info = env.step(s, a)
        assert np.array_equal(again, s2), (
            f"transition {len(transitions) - 1 - i} did not reproduce: max|d| "
            f"{np.abs(again - s2).max():.3e}")


def test_a_state_the_adapter_never_emitted_is_an_error_not_a_guess(env):
    """Returning 0.0, re-resetting or "step from wherever the sim is" would each
    produce a plausible number computed on physics that never happened -- and it
    would be attributed to the candidate reward. The message has to name the
    three legal origins, because the reader's next question is "where was I
    supposed to get this state from?"."""
    invented = np.zeros(env.obs_dim)
    for call in (lambda: env.step(invented, env.action_set[0]),
                 lambda: env.reference_reward(invented),
                 lambda: env.render(invented)):
        with pytest.raises(UnknownStateError) as excinfo:
            call()
        message = str(excinfo.value)
        for expected in ("reset()", "step()", "random_state()", env.name):
            assert expected in message, f"the diagnostic omits {expected!r}:\n{message}"


def test_a_perturbed_state_does_not_silently_hit_the_cache(env):
    """The cache is keyed on float32 bytes, so a state that has been nudged --
    the in-place mutation case -- must miss rather than land on its neighbour."""
    state = env.reset(np.random.default_rng(21))
    perturbed = np.array(state, dtype=float)
    perturbed[0] += 1e-3
    with pytest.raises(UnknownStateError):
        env.step(perturbed, env.action_set[0])


# --------------------------------------------------------------------------
# the ground truth, and its independence from the reward
# --------------------------------------------------------------------------


def test_task_metric_is_the_simulators_own_success_flag(env):
    """The reason this tier exists. `task_metric` re-derives the success
    check from the OBSERVATION -- it cannot read `info`, because it is called
    on trajectories the simulator has long forgotten -- so the re-derivation has
    to be tested for EQUALITY against the flag Meta-World itself published, not
    merely for resemblance."""
    states, _actions, successes = _rollout(env, seed=31, n=200, expert=True)
    assert max(successes) == 1.0, "the scripted expert did not solve the task"

    mine = [env._success_of(np.asarray(row, dtype=float)) for row in states[1:]]
    mismatches = [i for i, (x, y) in enumerate(zip(mine, successes)) if x != y]
    assert not mismatches, f"success check disagrees with info['success'] at {mismatches[:8]}"

    metric = env.task_metric(states)
    assert 0.0 <= metric <= 1.0
    assert metric == pytest.approx(float(np.mean(successes)))


#: The two transcriptions that approximate a quantity the observation does not carry,
#: with the disagreement rate measured over 900 steps each. Everything
#: else is exact and asserted so below. `reach` itself is documented on `_success_reach`.
_APPROXIMATE = {
    "reach-wall-v3": "tcp_center (hand body - 0.045 m in z), as on reach",
    "stick-push-v3": "touching_main_object (a contact query) as tcp_to_stick <= 0.04",
}


@pytest.mark.parametrize("task", _EXPECTED_MT50_ONLY)
def test_every_mt50_success_check_is_the_simulators_own_flag(simulator, task):
    """The MT10 proof above, for each of the forty MT50-only tasks: the observation-only
    re-derivation must EQUAL `info["success"]` step for step, on a scripted-expert
    episode (which reaches success) and a random one (which mostly does not). Two tasks
    are documented approximations -- `_APPROXIMATE` -- and are held to the mismatch
    ceiling their docstrings state rather than to zero; a third would be a new
    finding, and this test is where it would surface. ~3 s per task."""
    from metaworld.policies import ENV_POLICY_MAP

    adapter = MetaWorld(task)
    policy = ENV_POLICY_MAP[task]()
    rng = np.random.default_rng(7)
    mine, theirs = [], []
    # 400 expert steps: bin-picking, disassemble and push-back take more than 150 on
    # some instances (measured), and the assertion below needs the flag to fire.
    for expert, n_steps in ((True, 400), (False, 150)):
        state = adapter.reset(rng)
        for _ in range(n_steps):
            action = (policy.get_action(np.array(state, dtype=float, copy=True)) if expert
                      else rng.uniform(adapter.action_low, adapter.action_high))
            state, _done, info = adapter.step(state, action)
            mine.append(adapter._success_of(np.asarray(state, dtype=float)))
            theirs.append(float(info["success"]))
    assert max(theirs) == 1.0, f"{task}: the scripted expert did not reach success in 400 steps"
    mismatches = [i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b]
    if task in _APPROXIMATE:
        assert len(mismatches) <= 0.02 * len(mine), (
            f"{task}: {len(mismatches)} of {len(mine)} steps disagree; the documented "
            f"approximation ({_APPROXIMATE[task]}) was measured at 1% or under")
    else:
        assert not mismatches, (
            f"{task}: the transcription disagrees with info['success'] at steps "
            f"{mismatches[:8]} ({len(mismatches)} of {len(mine)})")


@pytest.mark.parametrize("task", _EXPECTED_METAWORLD)
def test_every_metaworld_task_restores_bitwise(simulator, task):
    """The snapshot proof, for each of the FIFTY: after 30 scripted-expert steps, restore
    the state from step 10 and re-step it -- the observation must come back bitwise and
    Meta-World's own reward for that step must match the live one. The per-task
    snapshot attributes (`_assigned_attrs`) are what this tests, in both directions: a
    reward that reads an attribute `reset_model` set and the snapshot forgot disagrees
    here, and so does one the snapshot must NOT carry -- pick-place's `init_left_pad`
    (a live view into `data.xpos`, `_PY_NEVER`) reads max|dreward| = 5.4e-2 when the ast
    scan collects it. The MT10 ten are in this
    parametrisation for exactly that reason: the module fixture above proves
    drawer-open only."""
    from metaworld.policies import ENV_POLICY_MAP

    adapter = MetaWorld(task)
    policy = ENV_POLICY_MAP[task]()
    rng = np.random.default_rng(3)
    state = adapter.reset(rng)
    states, actions, rewards = [state], [], []
    for _ in range(30):
        action = np.clip(policy.get_action(np.array(state, dtype=float, copy=True)), -1, 1)
        nxt, _done, _info = adapter.step(state, action)
        rewards.append(adapter.reference_reward(nxt, action))
        states.append(nxt)
        actions.append(action)
        state = nxt
    adapter.retain_states([np.asarray(states)])
    replay, _d, _i = adapter.step(states[10], actions[10])
    assert np.asarray(replay).tobytes() == np.asarray(states[11]).tobytes(), (
        f"{task}: restoring step 10 and re-stepping is not bitwise equal to the rollout")
    assert adapter.reference_reward(states[11], actions[10]) == rewards[10], (
        f"{task}: the shipped reward after a restore differs from the live one")


@pytest.mark.parametrize("task", ["hammer-v3", "pick-place-v3"])
def test_the_bundled_expert_is_found_under_either_prefix(simulator, task):
    """A `bird/demos.py::_bundled_metaworld_expert` gated on `mt10_` would make every
    `expert`-needing quality screen silently a no-op on an `mt50_*` env while the same
    config on an `mt10_*` env ran it -- a method running differently by env tier with
    every counter reading normal. Both prefixes must resolve."""
    from bird import demos

    adapter = MetaWorld(task)
    pol = demos._bundled_metaworld_expert(adapter, _env_id(task))
    assert pol is not None and task in pol.name, (task, pol)


def test_success_is_metaworlds_any_step_criterion(env):
    """`success()` is inherited as `task_metric >= success_threshold`, and the
    threshold is `1 / (2 * horizon)`: one successful step out of <= 501 clears it,
    zero does not. So the binary is exactly Meta-World's published `max_t
    success_t` while the metric stays dense, and the two cannot disagree because
    one is a threshold on the other."""
    assert env.success_threshold == 1.0 / (2.0 * env.horizon)

    states, _actions, successes = _rollout(env, seed=31, n=200, expert=True)
    assert env.success(states) is True
    assert env.success(states) == (max(successes) > 0.0)

    hit = states[1 + successes.index(1.0)]
    miss = states[0]
    # One success among a full-length episode of failures still counts.
    one_in_many = np.asarray([miss] * env.horizon + [hit])
    assert 0.0 < env.task_metric(one_in_many) <= 2.0 / env.horizon
    assert env.success(one_in_many) is True
    assert env.task_metric(np.asarray([miss, miss, miss])) == 0.0
    assert env.success(np.asarray([miss, miss, miss])) is False


def test_task_metric_survives_the_simulator_forgetting_the_episode(env):
    """A pure function of the stored states, with no cache lookup in it: that is
    what lets `phases._reference_score` and §4 re-scoring work on records read
    back from `journal.jsonl` after the LRU has evicted them."""
    states, _actions, _successes = _rollout(env, seed=31, n=120, expert=True)
    before = env.task_metric(states)

    for seed in range(3):                       # drive the sim well away
        _rollout(env, seed=200 + seed, n=30)

    assert env.task_metric(states) == before
    # ...and through a JSON round trip, which is how §4 actually receives it.
    assert env.task_metric(json.loads(json.dumps(states.tolist()))) == before
    assert env.task_metric(types.SimpleNamespace(states=states)) == before


def test_task_metric_does_not_move_when_the_reward_does(env):
    """The no-leak property, tested the only way that is not circular: replace
    Meta-World's shaped reward and check the ground truth is unmoved.

    This is also the trap in the obvious design. `info["success"]` is computed
    inside `evaluate_state`, which CALLS `compute_reward` and reads geometric
    intermediates out of its return tuple -- so an adapter that monkey-patched
    the reward would destroy its own ground truth. This one re-derives the
    check from the observation instead, so the metric is invariant while
    `reference_reward` follows the swap.
    """
    states, actions, successes = _rollout(env, seed=31, n=200, expert=True)
    before = env.task_metric(states)
    assert before > 0.0, "a metric pinned at zero would pass this test vacuously"

    original = type(env._env).compute_reward

    def constant_reward(*args, **kwargs):
        """Keep the shape (`evaluate_state` unpacks the tuple and reads the
        geometric terms for `success`), change only the scalar."""
        out = original(env._env, *args, **kwargs)
        return (-999.0,) + tuple(out[1:]) if isinstance(out, tuple) else -999.0

    env._env.compute_reward = constant_reward
    try:
        assert env.reference_reward(states[1], actions[0]) == -999.0, (
            "the reward swap did not take effect, so this test proves nothing")
        assert env.task_metric(states) == before
        assert [env._success_of(np.asarray(r)) for r in states[1:]] == successes
    finally:
        del env._env.compute_reward
    assert env.reference_reward(states[1], actions[0]) != -999.0


def test_step_info_carries_the_flag_and_nothing_else(env):
    """`_Gym` forwards `info` straight to SB3. `unscaled_reward` is byte-identical
    to the shaped reward the candidate is competing to replace, and
    `in_place_reward`/`grasp_reward` are its two factors, so anything left in
    here is a leak with a wide blast radius."""
    state = env.reset(np.random.default_rng(1))
    _next, _done, info = env.step(state, env.action_set[3])
    assert set(info) == {"success"}
    assert isinstance(info["success"], float)


def test_the_prompt_gate_removes_the_reward_and_the_metric(env):
    """`full_source` is the real Meta-World source, so it contains both
    `compute_reward` (the artifact under search) and `evaluate_state` (the
    METRIC, the worse leak). `reward_source` is the authoritative gate
    `_strip_reward` prefers over its regex, and it must be an exact substring of
    what `describe` hands over -- `inspect.getsource(cls.evaluate_state)` returns
    the decorator's `inner` and would silently fail that test."""
    from bird.components.generation import _strip_reward

    source = env.describe("full_source")
    assert "def compute_reward" in source and "def evaluate_state" in source
    assert env.reward_source.strip() in source

    stripped = _strip_reward(types.SimpleNamespace(env=env), source)
    assert "def compute_reward" not in stripped
    assert "def evaluate_state" not in stripped
    assert "withheld" in stripped
    # What must survive: the observation is not described anywhere else.
    assert "def _get_obs" in stripped
    assert "def tolerance" in stripped


@pytest.mark.parametrize("kind", ["natural_language_only", "state_action_api_stub",
                                  "pythonic_class_abstraction"])
def test_describe_documents_the_39_dimensions(env, kind):
    """An empty `_state_fields` makes `generate.context.env_spec` a no-op with no
    error, and on a 39-D vector the field docs are the only thing telling an LLM
    what `s[36:39]` is."""
    assert len(env._state_fields) == env.obs_dim
    assert len(env._action_fields) == env.action_dim
    text = env.describe(kind)
    assert text
    for field in ("hand_x", "goal_x", "prev_object2_qw", "gripper"):
        assert field in text, f"{kind} does not document {field}"
    assert env.symbol_mapping["goal_position"] == "s[36:39]"
    assert env.describe("none") == ""


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def test_render_returns_a_frame_of_the_state_it_was_given(env, gl):
    states, _actions, _successes = _rollout(env, seed=51, n=30, expert=True)

    # Render out of order and after the simulator has moved on -- which is what
    # `observability.record_rollouts` does: it replays a trajectory stored during
    # training, long after training finished.
    away = env.reset(np.random.default_rng(52))
    env.step(away, env.action_set[0])

    frames = [env.render(states[i]) for i in (25, 3, 14)]
    for frame in frames:
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert frame.std() > 1.0, "a uniform frame means the scene never rendered"
    assert not np.array_equal(frames[0], frames[1]), (
        "two different states rendered identically: render is showing the "
        "simulator's current pose rather than the state it was handed")


def test_render_takes_exactly_one_positional_state(env):
    """`observability._accepts_a_state` gates on `signature.bind(object())`; a
    renderer that fails it is skipped with a warning and the run records no
    video at all."""
    from bird.observability import _accepts_a_state

    assert _accepts_a_state(env.render)


def test_a_renderer_that_cannot_run_degrades_to_no_video(env, monkeypatch):
    """The documented degradation, and the reason `render` may raise instead of
    rasterising a fallback: a hand-drawn Sawyer would be plausible-looking and
    wrong, and `rda`/`gt_reward_design` do not log frames, they GRADE on them.

    An EMPTY frame list and not `None`, and the difference is per-candidate
    versus per-iteration. (`_render_frames` returns `(frames, steps)` on any
    path that produced a frame list at all, so "no frames" is `([], [])` --
    the stride travels with the frames it maps, and an empty render has an
    empty stride. `None` remains the one answer that is about the RENDERER.) `_render_frames` returns `None` to say "stop, this renderer
    is unusable for everything"; a raise out of `render(state)` does not license
    that claim, because on this adapter the commonest cause is LRU eviction of
    an OLDER trajectory and the candidate that trained last renders fine.
    Returning `None` here would let one evicted rollout cost the whole iteration's
    video -- measured at `generate.n_candidates: 2`, where `videos/` is not
    created at all. Either way no frames come back, which is the part that
    protects the VLM-graded methods from a wrong frame.
    """
    from bird.observability import _render_frames
    from bird.types import Trajectory

    states, _actions, _successes = _rollout(env, seed=61, n=6)
    traj = Trajectory(states=states, length=len(states))

    def no_gl(state, width=480):
        raise RuntimeError("Could not create a GL context")

    monkeypatch.setattr(env, "render", no_gl)
    assert _render_frames(env.render, traj, max_frames=4, width=64) == ([], [])

    # Same path for a state that fell out of the LRU: one warning, no frames,
    # and emphatically not a wrong frame.
    def evicted(state, width=480):
        raise UnknownStateError("evicted")

    monkeypatch.setattr(env, "render", evicted)
    assert _render_frames(env.render, traj, max_frames=4, width=64) == ([], [])

    # The one condition that IS a claim about the renderer rather than about a
    # trajectory: a `render` the harness cannot even hand a state to. That still
    # stops the iteration, because it will fail identically for every candidate.
    def stateless():
        raise AssertionError("never called")

    monkeypatch.setattr(env, "render", stateless)
    assert _render_frames(env.render, traj, max_frames=4, width=64) is None


# --------------------------------------------------------------------------
# extra viewpoints (`output.video.n_views`, spec `judge.extra_views`)
# --------------------------------------------------------------------------

def test_the_spec_names_the_camera_this_adapter_renders(env):
    """The judge is told the primary panel is `judge.camera.name`, so the spec and the
    class must agree; `__init__` refuses a spec that names another camera."""
    assert env.primary_view.name == env.camera_name == "corner"
    assert [v.name for v in env.extra_views] == ["topview", "behindGripper"]


def test_the_corner_cameras_are_upside_down_and_the_others_are_not(env):
    """The orientation rule `envs.cameras.MujocoViews.inverted` applies: measured
    over the seven Meta-World cameras, every `corner*` camera's image-up
    axis points into the floor and `topview`/`behindGripper`/`gripperPOV` are upright.
    No GL context is needed to read a camera's live rotation."""
    env.reset(0)
    views = env._views()
    inverted = {name: views.inverted(views.camera_id(name)) for name in views.camera_names()}
    assert inverted == {"topview": False, "corner": True, "corner2": True, "corner3": True,
                        "corner4": True, "behindGripper": False, "gripperPOV": False}
    assert views._renderer is None, "reading the rotation must not open a GL context"


def test_render_view_refuses_a_camera_the_model_lacks(env):
    from bird.envs.base import View

    with pytest.raises(KeyError, match="no camera of that name"):
        env.render_view(env.reset(0), View(name="corner9"))


def test_every_spec_view_renders_the_state_it_was_given(env, gl):
    """Each extra view is a real frame of the STORED state -- out of order, after the
    simulator has moved on -- and differs from the primary and from the other views:
    a `render_view` that quietly returned the primary camera would put one camera in
    two panels the judge is told are two viewpoints."""
    states, _actions, _successes = _rollout(env, seed=61, n=30, expert=True)
    env.step(env.reset(np.random.default_rng(62)), env.action_set[0])
    primary = env.render(states[20])
    seen = [primary]
    for view in env.extra_views:
        frame = env.render_view(states[20], view)
        assert frame.shape == primary.shape and frame.dtype == np.uint8
        assert frame.std() > 1.0, f"{view.name}: a uniform frame means the scene never rendered"
        for other in seen:
            assert not np.array_equal(frame, other), f"{view.name} duplicates another panel"
        # a pure function of the state, like `render`
        assert np.array_equal(env.render_view(states[20], view), frame)
        assert not np.array_equal(env.render_view(states[3], view), frame), (
            f"{view.name}: two different states rendered identically")
        seen.append(frame)


def test_the_composite_keeps_the_single_view_frame_as_its_first_panel(env, gl):
    """The ablation's premise: `n_views: 3` adds two panels to the right of the frame
    every published config records and moves nothing in it."""
    from bird import multiview
    from bird.config import load
    from bird.observability import _as_rgb8, _resize_nearest, _view_plan

    width = 96
    cfg = load("eureka", profile="tester", overrides={
        "problem.env_id": env.name, "output.video.n_views": 3, "output.video.width": width})
    ctx = types.SimpleNamespace(cfg=cfg, env=env)
    plan = _view_plan(ctx)
    assert [p["name"] for p in plan.layout["panels"]] == ["corner", "topview", "behindGripper"]
    state = env.reset(7)
    composite = plan.render(state)
    single = _resize_nearest(_as_rgb8(env.render(state)), width)
    assert composite.shape == (single.shape[0], 3 * width + 2 * multiview.GUTTER_PX, 3)
    assert np.array_equal(composite[:, :width], single)
    gap = composite[:, width:width + multiview.GUTTER_PX]
    assert not gap.any()


# --------------------------------------------------------------------------
# the goal slot
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", _EXPECTED_MT10)
def test_the_goal_slot_is_the_simulators_target_on_every_task(simulator, task):
    """Ten constructions (~1.3 s each) rather than the module `env`, because the goal
    block has to be checked against each task's own `_target_pos`. `s[36:39]` being
    the goal (`_GOAL`) is a claim about `SawyerXYZEnv._get_obs` that `_success_of` and
    `discretise` both rely on; this is the one place it is measured."""
    adapter = MetaWorld(task)
    state = adapter.reset(np.random.default_rng(7))
    target = np.asarray(adapter._env._target_pos, dtype=float)
    assert target.shape == (3,)
    assert np.allclose(state[36:39], target, atol=1e-9), (
        f"{task}: s[36:39] is not the simulator's _target_pos")


# --------------------------------------------------------------------------
# domain randomisation
# --------------------------------------------------------------------------
#
# The DR contract -- "scale x nominal, never compounding", through `set_dr`,
# `_sample_dr` and `_apply_dr` -- pinned from both sides: a nominal reset is bitwise
# the compiled model, and a fixed draw reaches exactly the arrays its axis writes and
# survives a snapshot restore mid-episode.


def _dr_arrays(env):
    """The model arrays `_apply_dr` writes, copied."""
    model = env._env.model
    out = {f: np.copy(getattr(model, f)) for f in env._DR_MODEL_FIELDS}
    out["gravity"] = np.copy(model.opt.gravity)
    return out


def _n_changed(env, before):
    """How many model entries differ from `before` (a `_dr_arrays` copy)."""
    now = _dr_arrays(env)
    return sum(int(np.count_nonzero(now[f] != before[f])) for f in before)


def test_a_nominal_reset_touches_nothing(simulator):
    """`_apply_dr` runs on every reset writing `1.0 x nominal`; at nominal that must be
    a no-op on the compiled model, or every nominal run would change physics."""
    env = MetaWorld(TASK)
    compiled = _dr_arrays(env)
    env.reset(np.random.default_rng(0))
    assert _n_changed(env, compiled) == 0


def test_a_fixed_draw_reaches_the_model_and_survives_a_restore(simulator):
    """A bare scalar is a fixed DR value (`EnvAdapter.set_dr`): the arrays the axis
    writes move by exactly the factor on the next reset, a snapshot restore mid-episode
    keeps the draw rather than putting stock physics back, and `set_dr(None)` -- the
    training backends' `finally` -- returns the next episode to the compiled model."""
    env = MetaWorld(TASK)                                        # drawer-open: a fixture task
    compiled = _dr_arrays(env)
    env.set_dr({"fixture_damping_scale": 4.0, "arm_damping_scale": 2.0})
    s = env.reset(np.random.default_rng(3))
    model = env._env.model
    dof = env._fixture_dofs[0]
    assert model.dof_damping[dof] == 4.0 * env._nominal["dof_damping"][dof]
    assert np.array_equal(model.dof_damping[0:7], 2.0 * env._nominal["dof_damping"][0:7])
    assert _n_changed(env, compiled) == 8                        # 7 arm dofs + the drawer slide
    env.step(s, env.action_set[3])
    env.step(s, env.action_set[4])                               # restore from `s` mid-episode
    assert model.dof_damping[dof] == 4.0 * env._nominal["dof_damping"][dof], \
        "a snapshot restore put stock physics back under a DR draw"
    env.set_dr(None)
    env.reset(np.random.default_rng(4))
    assert _n_changed(env, compiled) == 0


def test_an_axis_that_matches_nothing_writes_nothing(simulator):
    """`object_mass_scale` matches no body on the five fixture tasks and
    `fixture_damping_scale` no joint on the free-object tasks (measured; the comment on
    `MetaWorld.dr_parameters` says so): a draw on either leaves the model untouched."""
    door = MetaWorld("door-open-v3")
    compiled = _dr_arrays(door)
    door.set_dr({"object_mass_scale": 2.0})
    door.reset(np.random.default_rng(0))
    assert _n_changed(door, compiled) == 0
    push = MetaWorld("push-v3")
    compiled = _dr_arrays(push)
    push.set_dr({"fixture_damping_scale": 4.0})
    push.reset(np.random.default_rng(0))
    assert _n_changed(push, compiled) == 0


def test_contact_friction_scales_the_table_where_the_legacy_axis_cannot(simulator):
    """MuJoCo max-mixes the two sides of a contact, and every table/fixture collider sits
    at 1.0 with no geom priority -- so the published `friction_scale` (object + pads) can
    never lower the friction the puck slides on. `contact_friction_scale` scales every
    non-robot collider, the table included."""
    env = MetaWorld("push-v3")
    compiled = _dr_arrays(env)
    env.set_dr({"contact_friction_scale": 0.5})
    env.reset(np.random.default_rng(0))
    model, mj = env._env.model, env._mj
    table = next(g for g in env._dr_colliders
                 if mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) == "tablelink")
    assert model.geom_friction[table, 0] == 0.5
    assert _n_changed(env, compiled) == len(env._dr_colliders) > 5
    legacy = MetaWorld("push-v3")
    legacy.set_dr({"friction_scale": 0.5})
    legacy.reset(np.random.default_rng(0))
    assert legacy._env.model.geom_friction[table, 0] == 1.0, "the legacy axis leaves the table alone"
