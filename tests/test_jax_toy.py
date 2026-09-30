"""`jax_toy` -- `ToyReacher` in jax.numpy, and the `jax_reward` tester config that runs
the jax reward contract end to end.

The offline half runs with no jax: the module imports without it, the factory refuses
without it and names the extra, the config loads and resolves the native channel off
the spec, and the mock LLM's jnp rewrite is a text property. The `-m jax` half holds
the port to `ToyReacher` within float32 under nominal AND moved physics, runs the
mock's programs through the traced verifier, and runs `examples/jax_reward`
under `--profile tester` end to end.
"""

from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bird import registry, tasks
from bird.config import load
from bird.envs.suites import env_suite
from bird.components import generation as G
from bird.components import verification as V
from bird.llm import mock as M

REPO = Path(__file__).resolve().parents[1]

registry.load_all()

FORMATS = ("component_dict_return", "component_dict_plus_weights", "scalar_only",
           "template_params")


# ==========================================================================
# offline
# ==========================================================================


def test_importing_the_module_does_not_import_jax():
    """`registry.load_all()` imports this module on every run, on boxes with no jax."""
    code = ("import sys; import bird.envs.jax_toy; "
            "assert 'jax' not in sys.modules, 'jax imported at module scope'; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_the_factory_refuses_without_jax_and_names_the_extra(monkeypatch):
    """ImportError FIRST, before `EnvAdapter.__init__` -- which is what lets every
    catalogue test skip or deselect this id on a machine without the extra."""
    import builtins

    real = builtins.__import__

    def no_jax(name, *args, **kwargs):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("no jax here")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_jax)
    with pytest.raises(ImportError, match=r"jax_toy needs the `jax` extra"):
        registry.get("env", "jax_toy")(None)


def test_the_config_loads_offline_and_the_spec_gives_it_a_native_channel():
    """A config load constructs no env, so `--validate-all` passes with no jax; the
    coherence pair (jax_* env, reward_language jax) holds; and the spec -- not the
    adapter -- is what makes `evaluate.fitness.source: native` resolvable, which is
    why `jax_toy` carries one at all (module docstring)."""
    from bird.native_signal import NATIVE_SUCCESS, channel_for_env

    cfg = load("examples/jax_reward", profile="tester")
    assert cfg["problem.env_id"] == "jax_toy"
    assert cfg["generate.reward_language"] == "jax"
    assert env_suite("jax_toy") == "jax"
    # Both inherited from the spec, not authored in the config.
    assert cfg["verify.forbidden_symbols"] == []
    assert cfg["problem.task_description"] == tasks.by_env_id("jax_toy").instruction
    assert cfg["llm.generator.provider"] == "mock"
    assert channel_for_env("jax_toy") == NATIVE_SUCCESS
    spec = tasks.by_env_id("jax_toy")
    assert spec is not None and spec.id == "jax_toy"
    assert set(spec.domain_randomization["parameters"]) == {"mass", "damping", "force_scale"}, (
        "step_batch is pure and takes no key, so there is no action_noise axis")


@pytest.mark.parametrize("fmt", FORMATS)
def test_the_mock_rewrites_the_one_branch_on_a_value_under_jax(fmt):
    """The numpy program is untouched; the jax program has the jnp preamble, no
    `float()` cast, and `jnp.where` where the numpy terms had `1.0 if ... else 0.0`."""
    tier = 4  # goal_shaped_success: the tier whose terms carry the branch
    np_code = M._valid_program(tier, random.Random(0), fmt, "numpy")
    jx_code = M._valid_program(tier, random.Random(0), fmt, "jax")
    assert "def _f(names" in np_code and "jnp" not in np_code
    assert "jnp.asarray(state" in jx_code and "def _f(names" not in jx_code
    if fmt != "template_params":
        assert " if dist <" in np_code
    assert " if dist <" not in jx_code and "jnp.where(dist <" in jx_code
    assert "float(total)" not in jx_code
    if fmt == "scalar_only":
        assert "return float(total)" in np_code and "return total" in jx_code
    # the weights the emitted source declares are read back the same way under both
    assert M._weights_of(np_code).keys() == M._weights_of(jx_code).keys()


# ==========================================================================
# under jax
# ==========================================================================


@pytest.fixture
def jax():
    return pytest.importorskip("jax")


def _pair():
    from bird.envs.toy import ToyReacher
    return registry.get("env", "jax_toy")(None), ToyReacher()


@pytest.mark.jax
@pytest.mark.parametrize("physics", [None, {"mass": 2.0, "damping": 0.3, "force_scale": 1.1}])
def test_step_matches_toy_reacher_to_float32(jax, physics):
    """The port is line for line: 64 random transitions agree with `ToyReacher._step`
    to float32 under nominal physics and under a moved draw, success flag included."""
    env, ref = _pair()
    if physics:
        env._dr_now = dict(physics)
        ref._dr_now = {**physics, "action_noise": 0.0}
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(64):
        s = env.random_state(rng)
        a = env.action_set[rng.integers(env.n_actions)]
        s2, done, info = env._step(s, a)
        s2_ref, done_ref, info_ref = ref._step(s, a)
        worst = max(worst, float(np.max(np.abs(s2 - s2_ref))))
        assert done is False and done_ref is False
        assert bool(info["success"]) == bool(info_ref["success"])
        assert isinstance(info["success"], (bool, np.bool_)), type(info["success"])
    assert worst < 1e-6, worst
    assert s2.dtype == np.float64  # the bridge hands float64 back to numpy consumers


@pytest.mark.jax
def test_reset_is_keyed_deterministic_and_in_the_far_quadrant(jax):
    env, _ = _pair()
    a = env.reset(np.random.default_rng(3))
    b = env.reset(np.random.default_rng(3))
    assert np.array_equal(a, b) and a.shape == (6,)
    rows = np.stack([env.reset(np.random.default_rng(i)) for i in range(24)])
    assert np.all((-0.85 <= rows[:, 0:2]) & (rows[:, 0:2] <= -0.25))
    assert np.all(rows[:, 2:4] == 0.0) and np.all(rows[:, 4:6] == env.goal)
    assert len({tuple(r[0:2]) for r in rows}) == 24, "24 seeds drew 24 starts"
    batch = np.asarray(env.reset_batch(jax.random.PRNGKey(0), 16))
    assert batch.shape == (16, 6) and batch.dtype == np.float32


@pytest.mark.jax
def test_episode_state_carries_the_dr_draw_across_interleaved_slots(jax):
    """`step_batch` reads `_dr_now` per call and
    `_VecEnvView` restores `episode_state()` per slot, so the blob must carry the
    draw or two interleaved slots both step under whichever reset last. Two
    episodes under two `set_dr` draws, interleaved on ONE adapter, must match two
    adapters each holding one draw."""
    env, _ = _pair()
    twins = [_pair()[0], _pair()[0]]
    draws = [{"mass": (0.5, 0.5), "damping": (0.9, 0.9), "force_scale": (1.2, 1.2)},
             {"mass": (3.0, 3.0), "damping": (0.2, 0.2), "force_scale": (0.3, 0.3)}]
    blobs, states, ref_states = [], [], []
    for i, dr in enumerate(draws):
        env.set_dr(dr)
        states.append(env.reset(np.random.default_rng(i)))
        blobs.append(env.episode_state())
        twins[i].set_dr(dr)
        ref_states.append(twins[i].reset(np.random.default_rng(i)))
        assert np.array_equal(states[i], ref_states[i])
    assert blobs[0][2] != blobs[1][2], "the two blobs carry two different draws"
    a = env.action_set[8]
    for t in range(6):
        for i in (0, 1):
            env.restore_episode_state(blobs[i])
            states[i], _d, _info = env.step(states[i], a)
            blobs[i] = env.episode_state()
            ref_states[i], _d, _info = twins[i].step(ref_states[i], a)
            assert np.allclose(states[i], ref_states[i], atol=1e-6), (t, i)
    assert not np.allclose(states[0], states[1]), "the two draws move the mass differently"


@pytest.mark.jax
def test_step_batch_takes_the_draw_as_an_argument_and_never_reads_self(jax):
    """Two rows step under two draws in ONE call, and what is on `self._dr_now` is
    irrelevant to `step_batch` -- the bridge is what threads the blob's draw
    through `physics_rows`, so the per-row pytree and the n=1 path are the same
    call."""
    env, ref = _pair()
    s = np.stack([env.random_state(np.random.default_rng(i)) for i in range(2)])
    a = np.stack([env.action_set[8], env.action_set[8]])
    draws = [{"mass": 0.5, "damping": 0.9, "force_scale": 1.2},
             {"mass": 3.0, "damping": 0.2, "force_scale": 0.3}]
    physics = {k: np.array([draws[0][k], draws[1][k]], dtype=np.float32) for k in draws[0]}
    env._dr_now = {"mass": 99.0, "damping": 0.0, "force_scale": 0.0}  # must be ignored
    s2, done, info = env.step_batch(s, a, physics)
    for i in (0, 1):
        ref._dr_now = {**draws[i], "action_noise": 0.0}
        s2_ref, _d, _i = ref._step(s[i], a[i])
        assert np.allclose(np.asarray(s2[i]), s2_ref, atol=1e-6), i
    assert not np.allclose(np.asarray(s2[0]), np.asarray(s2[1]))
    # None = nominal, whatever self._dr_now says
    s2n, _d, _i = env.step_batch(s, a, None)
    ref._dr_now = {**ref._dr_nominal}
    assert np.allclose(np.asarray(s2n[0]), ref._step(s[0], a[0])[0], atol=1e-6)
    assert env.physics_rows(3)["mass"].shape == (3,)


@pytest.mark.jax
def test_the_metric_discriminates_the_expert_from_random(jax):
    """The one property a toy must have (bird/envs/toy.py's module docstring): the PD
    expert written for `toy_reacher` clears `success_threshold` here, random does not."""
    from bird.envs.toy import toy_reacher_expert

    env, _ = _pair()

    def episode(policy, seed):
        rng = np.random.default_rng(seed)
        states = [env.reset(rng)]
        for t in range(env.horizon):
            s2, _done, _info = env.step(states[-1], policy(states[-1], t, rng))
            states.append(s2)
        return env.task_metric(np.stack(states))

    expert = np.mean([episode(lambda s, t, r: toy_reacher_expert(s, t), i) for i in range(5)])
    rand = np.mean([episode(lambda s, t, r: env.action_set[r.integers(env.n_actions)], i)
                    for i in range(5)])
    assert expert > env.success_threshold > rand, (expert, rand)


@pytest.mark.jax
@pytest.mark.parametrize("fmt", FORMATS)
def test_every_mock_program_passes_the_traced_verifier(jax, fmt):
    """What the tester run relies on: each tier's jnp program, in each format, compiles
    in the restricted namespace and passes every probing check under jit(vmap)."""
    from bird.budget import Budget
    from bird.components import verification as V
    from bird.context import Context
    from bird.types import Candidate

    cfg = load("examples/jax_reward", profile="tester")
    # After load, not through overrides: `generate.output.format` is coupled to
    # `problem.reward_representation` by coherence, and this test is about the
    # verifier's reading of each format's program, not about a legal config pair.
    cfg._data["generate"]["output"]["format"] = fmt
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(0))
    ctx.env = registry.get("env", "jax_toy")(ctx)
    r = random.Random(1)
    for tier in range(len(M.TIERS)):
        cand = Candidate(cand_id=f"{fmt}-{tier}", iteration=0,
                         reward_code=M._valid_program(tier, r, fmt, "jax"))
        fn, err = V.compile_reward(ctx, cand)
        assert fn is not None, (fmt, tier, err)
        for check in ("execution_smoke", "output_shape", "output_dtype", "finite_values",
                      "non_constant_reward"):
            verdict = registry.get("dynamic_check", check)(ctx, cand, fn)
            assert verdict == "", (fmt, tier, verdict)


@pytest.mark.jax
def test_jax_reward_runs_end_to_end_under_the_tester_profile(jax, tmp_path):
    """The whole point of the env: the tier's plumbing, offline, in seconds. Mock LLM
    writing jnp, the traced verifier, the mock learner stepping a jitted env through
    the single-env bridge, and rule N scoring on the spec's native success."""
    spec = importlib.util.spec_from_file_location("bird_entry_jax_toy", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    cfg = load("examples/jax_reward", profile="tester",
               overrides={"seed": 0, "generate.n_candidates": 3})
    entry.run(cfg, out_root=str(tmp_path))

    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert json.loads((run / "status.json").read_text())["status"] == "ok"
    resolved = (run / "config.resolved.yaml").read_text()
    assert "reward_language: jax" in resolved and "env_id: jax_toy" in resolved
    dirs = sorted(p for p in (run / "candidates").iterdir() if p.is_dir())
    assert len(dirs) == 6, "3 candidates x 2 iterations"
    metas = {d: json.loads((d / "meta.json").read_text()) for d in dirs}
    kinds = {m.get("failure_kind") or "" for m in metas.values()}
    assert kinds <= {"", "invalid"}, kinds
    valid = [d for d, m in metas.items() if not m.get("failure")]
    assert valid, "the mock's jnp programs were all refused"
    for d in valid:
        code = (d / "reward.py").read_text()
        # A valid program is the mock's jnp program -- or one of its designed
        # "invalid" kinds (`tier=-1`) that this config's three dynamic checks do
        # not catch (constant_reward), exactly as under numpy.
        assert ("jnp." in code) or ("# bird-mock: tier=-1" in code), code[:120]
        assert "def _f(names" not in code
        assert (d / "train_result.json").exists()


@pytest.mark.jax
def test_the_committed_spec_is_what_the_generator_produces(jax):
    """`tasks/jax_toy/shared_spec.yaml` is generated, never hand-edited: `--check`
    regenerates it (anchors and reset ranges re-measured through the adapter) and diffs.
    The same `--check` drift gate as the other generated specs, for the same reason."""
    r = subprocess.run([sys.executable, "scripts/derive_jax_spec.py", "--source", "toy_reacher",
                        "--target", "jax_toy", "--adapter", "bird/envs/jax_toy.py",
                        "--class-name", "JaxToyReacher", "--drop-dr-axis", "action_noise", "--check"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ==========================================================================
# the generated spec against its parent (offline: `derive()` with nothing measured)
# ==========================================================================


def _derived_doc():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import derive_jax_spec as D

    warnings = []
    doc = D.derive("toy_reacher", "jax_toy", Path("bird/envs/jax_toy.py"), "JaxToyReacher",
                   "_reset_pure", "_step_pure", ["action_noise"], None,
                   "test", "2026-09-15", False, None, warnings)
    return D, doc, warnings


def test_the_derived_contract_is_the_verifier_and_the_prompt_not_a_retyping():
    """For a jax-suite target the generator derives
    `reward.contract` -- framework jax, allowed_modules EQUAL to the verifier's jax import
    allowlist, signature and idiom_rule imported from generation -- and derives the
    natural_language sentence from that contract. Three equalities so record, verifier
    and prompt cannot drift; a numpy-suite target keeps its parent's contract."""
    import yaml

    D, doc, _ = _derived_doc()
    contract = doc["reward"]["contract"]
    assert contract["framework"] == "jax"
    assert contract["allowed_modules"] == sorted(V.import_allowlist("jax"))
    assert contract["signature"] == G._SIGNATURE
    assert contract["idiom_rule"] == G._JAX_CLAUSE
    assert contract["returns"] == "(scalar, components)"
    for key in ("env_prose", "natural_language"):
        assert doc["description"][key].endswith(D.JAX_REWARD_SENTENCE), key
    # the committed file says the same (it is generated; `--check` holds it there)
    committed = yaml.safe_load((REPO / "tasks/jax_toy/shared_spec.yaml").read_text())
    assert committed["reward"]["contract"] == contract
    assert committed["description"]["natural_language"] == doc["description"]["natural_language"]
    # a parent that is not a jax target is left alone
    parent = yaml.safe_load((REPO / "tasks/toy_reacher/shared_spec.yaml").read_text())
    untouched = yaml.safe_load(yaml.safe_dump(parent))
    assert D._derive_contract(untouched, "toy_reacher") == "numpy"
    assert untouched["reward"]["contract"] == parent["reward"]["contract"]


def test_the_shared_blocks_are_the_parents_byte_for_byte_and_full_source_repoints():
    """What a derived spec COPIES is the parent's verbatim, and what it re-derives is
    named: the task's surface, budget, success blocks, judge and exploits are
    byte-identical to toy_reacher's; `description` differs by exactly the derived
    sentence and the re-pointed `full_source` (asserted positively, at the adapter);
    `symbol_mapping` differs by exactly the norm rewrite; DR by the dropped axis."""
    import yaml

    D, doc, _ = _derived_doc()
    parent = yaml.safe_load((REPO / "tasks/toy_reacher/shared_spec.yaml").read_text())
    for block in ("state_surface", "budget", "discrete_success", "continuous_success",
                  "judge", "exploits"):
        assert yaml.safe_dump(doc[block]) == yaml.safe_dump(parent[block]), block
    d, p = doc["description"], parent["description"]
    assert d["full_source"] == {"path": "bird/envs/jax_toy.py", "lines": None}
    assert p["full_source"]["path"] == "bird/envs/toy.py"
    for key in ("env_prose", "natural_language"):
        assert d[key] == p[key].rstrip() + " " + D.JAX_REWARD_SENTENCE, key
    for key in set(p) - {"full_source", "env_prose", "natural_language"}:
        assert d[key] == p[key], key
    sm, psm = doc["symbol_mapping"], parent["symbol_mapping"]
    assert set(sm) == set(psm)
    assert psm["distance_to_goal"].startswith("np.linalg.norm(")
    assert sm["distance_to_goal"] == "(((s[0:2] - s[4:6]) ** 2).sum() ** 0.5)"
    assert {k: v for k, v in sm.items() if k != "distance_to_goal"} == \
        {k: v for k, v in psm.items() if k != "distance_to_goal"}
    assert set(parent["domain_randomization"]["parameters"]) - set(doc["domain_randomization"]["parameters"]) == {"action_noise"}


def test_parallel_on_the_batched_tier_loads_and_routes_away_from_fork():
    """The batched tier RUNS `parallel`, without forking.

    TWO HAZARDS are why this tier may not fork:

      * a child forked after the parent initialised CUDA through jax cannot
        use CUDA itself -- its first CUDA call fails;
      * a fork from a parent whose jax runtime has started its thread pools
        can deadlock, because the child inherits locks held by threads that
        do not exist in it.

    WHY THE GUARD IS AT THE FORK SITES. A load-time REFUSAL of `parallel` on
    this tier cannot be measured -- the config will not even load, so the
    tier's own end-to-end run is impossible to collect -- so the guard sits at
    each fork site, where it can be. There are three, and a test that checked
    one would leave the other two to be found in production:

      1. the CANDIDATE worker: spawned, not forked, when the live env is a
         `BatchedEnvAdapter`;
      2. the SEED workers: not forked at all on this tier (sequential seeds
         are bit-identical by contract);
      3. the numpy slot worker in `fasttd3._VecEnvView`: never reached here,
         per `_JaxVecEnvView` -- "no slot workers and no pipes".

    Keyed on the DECLARED capability, not on the `jax_` prefix: the prefix is
    a name and this is a capability, and rules keyed on the name go wrong
    (the device rule and `prepare_for_env` are both keyed on capability for
    the same reason).
    """
    from bird.config import load
    from bird.components.training import _worker_is_batched, _cfg_declares_batched

    # -- it LOADS, on both routes to the pin -----------------------------
    load("examples/jax_reward", profile="tester")           # shipped default
    cfg = load("examples/jax_reward", profile="tester",
               overrides={"train.candidate_parallelism": "parallel"})
    load("examples/jax_reward", profile="full")   # the profile sets parallel

    # -- 1. the candidate worker is routed by CAPABILITY, on the instance --
    from bird import registry
    registry.load_all()
    from bird.envs.jax_base import BatchedEnvAdapter

    class _Batched(BatchedEnvAdapter):
        # The abstract methods, stubbed: the routing question is about the
        # TYPE, and a real adapter here would drag in the whole tier.
        def reset_batch(self, *a, **k): raise NotImplementedError
        def step_batch(self, *a, **k): raise NotImplementedError
        def source_parents(self, *a, **k): raise NotImplementedError
        # `__init_subclass__` requires a CONCRETE `random_state` -- the
        # screens call it and the fallback is silent, so the base class
        # refuses a subclass without one. Stubbed for the same reason as
        # the rest: this test asks a routing question about the type.
        def random_state(self, *a, **k): raise NotImplementedError

    assert _worker_is_batched(_Batched.__new__(_Batched)) is True
    assert _worker_is_batched(object()) is False, (
        "a non-batched env must still FORK -- `parallel` is the normal "
        "route for every numpy-tier config and spawning them all would be a "
        "silent cost on every run")

    # -- 2. the seed fork is refused on this tier, and only on this tier ---
    from bird.components.training import _seed_fork_workers
    assert _cfg_declares_batched(cfg) is True
    assert _seed_fork_workers(cfg, 4)[0] == 1, (
        "seeds must not fork on the batched tier: they fork IN THE PARENT, "
        "where jax is initialised, which is the same hazard one site over")

    numpy_cfg = load("eureka", profile="tester", overrides={
        "train.candidate_parallelism": "parallel", "loop.max_parallel_trainings": 4})
    assert _cfg_declares_batched(numpy_cfg) is False
    assert _seed_fork_workers(numpy_cfg, 4)[0] > 1, (
        "the numpy tier must still fork its seeds; a guard that stopped it "
        "would slow every numpy-tier run with nothing red")
    assert _seed_fork_workers(numpy_cfg, 4)[1] == "", (
        "the numpy tier was not downgraded, so it must carry no reason -- a "
        "reason on every row is a reason nobody reads")


def test_an_unresolvable_or_undeclared_env_is_treated_as_batched():
    """The safe direction of `_cfg_declares_batched`, asserted as behaviour.

    The two errors are not symmetric and the code says so in prose; this is
    the assertion. Guessing False lets a new batched adapter fork and hang a
    paid allocation with nothing red; guessing True costs wall-clock on a
    path that is bit-identical either way. So an id that will not resolve,
    and a jax-suite adapter that declares nothing, both answer True.
    """
    from bird.components.training import _cfg_declares_batched

    class _Cfg(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    assert _cfg_declares_batched(_Cfg({"problem.env_id": "jax_not_a_real_env"})) is True
    assert _cfg_declares_batched(_Cfg({"problem.env_id": "definitely_not_registered"})) is False, (
        "a NON-jax id that does not resolve is not this guard's business -- "
        "returning True there would route unrelated broken configs away from "
        "forking and hide the real error")


def test_the_recorded_stack_follows_the_library_not_the_interpreter():
    """`--measure` reports every version the interpreter has; the spec records only
    what its library uses. Otherwise the same commit's `--check` is red in a
    mujoco-equipped venv and green in a mujoco-less one, because the toy's spec
    acquires a `mujoco:` line from whichever interpreter happens to run it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "derive_jax_spec", REPO / "scripts" / "derive_jax_spec.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    measured = {"jax": "0.8.0", "python": "3.11.15", "mujoco": "3.13.0"}
    toy = mod._recorded_stack("bird", measured, "2026-09-16", ["cpu"])
    sim = mod._recorded_stack("mujoco", measured, "2026-09-16", ["cpu", "cuda"])
    assert set(toy) == {"jax", "python"}, toy
    assert set(sim) == {"jax", "python", "mujoco"}, sim
    assert sim["mujoco"] == "3.13.0 (measured 2026-09-16, cpu/cuda)"


def test_every_jax_suite_env_declares_whether_it_needs_cuda(monkeypatch):
    """No default, so the declaration is mandatory and its absence fails HERE.

    `_check_coherence` refuses a jax-suite env configured without
    `train.hyperparameters.device=cuda`. Keying that refusal on
    `env_suite(env_id) == "jax"`, which maps by PREFIX, would let the NAME do
    the work of CAPABILITY: it would catch `jax_toy`, a tester-tier offline env
    in `jnp` with no device hand-off, and make
    `configs/examples/jax_reward.yaml` unloadable; it would equally MISS an
    MJX adapter not named `jax_*`.

    So the refusal reads `requires_cuda` off the registered object, and the
    rule supplies NO DEFAULT. Both defaults are wrong in a way nothing would
    report:

      default False -- a future MJX adapter that forgets to declare trains
                       on cpu beside an idle GPU with every counter reading
                       normal;
      default True  -- the next CPU-capable jax env is refused, the same
                       mistake as keying on the prefix.

    An omission therefore refuses nothing, silently, and THIS test is what
    turns it into a failure at registration time -- it is most of the rule's
    value.

    READ OFF THE REGISTERED OBJECT, INCLUDING FACTORIES. `registry.get` holds
    a FACTORY FUNCTION for `jax_toy` and for `upstream_assistax_*`, and a
    function inherits nothing from the class it returns -- so a check that
    assumed classes would pass vacuously on exactly the env this rule was
    getting wrong.
    """
    from bird import registry
    from bird.envs.suites import env_suite

    registry.load_all()
    jax_ids = [n for n in registry.names("env") if env_suite(n) == "jax"]
    assert jax_ids, "no jax-suite envs registered -- this test would check nothing"

    undeclared = []
    missing_batched = []
    for env_id in jax_ids:
        obj = registry.get("env", env_id)
        # `batched` is asserted BESIDE `requires_cuda`: the parallelism
        # routing reads that one off the registered object the same way, so a
        # new adapter must not be able to forget either. TWO attributes and
        # not one because `jax_toy` is batched and does NOT require cuda --
        # collapsing the two questions would answer one of them wrongly.
        if not isinstance(getattr(obj, "batched", None), bool):
            missing_batched.append(f"{env_id} ({getattr(obj, '__name__', '?')})")
        if not isinstance(getattr(obj, "requires_cuda", None), bool):
            undeclared.append(f"{env_id} ({type(obj).__name__} {getattr(obj, '__name__', '?')})")
    assert not undeclared, (
        "these jax-suite envs declare no `requires_cuda` on the object the registry "
        "holds, so `_check_coherence` cannot tell whether they need a CUDA learner "
        "device and refuses nothing for them:\n  " + "\n  ".join(undeclared) +
        "\nDeclare it on the adapter class, and on the factory too where a factory "
        "is what `@register` decorates -- a function inherits nothing from the class "
        "it returns."
    )
    assert not missing_batched, (
        "these jax-suite envs declare no `batched` on the object the registry holds, "
        "so the parallelism routing cannot tell whether "
        "forking their candidate worker would deadlock:\n  " +
        "\n  ".join(missing_batched) +
        "\nDeclare it on the adapter and carry it onto the factory with `_declare`, "
        "as `requires_cuda` is."
    )

    # -- AND EACH ONE MUST SET THE XLA FLAGS WHEN IT IS CONSTRUCTED ---------
    #
    # `prepare_for_env` is called from `bird.py::run` and from the spawn
    # child, and from nowhere else -- so any THIRD path that builds a jax env
    # would get no `XLA_FLAGS` and no `XLA_PYTHON_CLIENT_PREALLOCATE` either:
    # an env constructed directly, the variable unset, and a parent that
    # reserves the whole card behind its own workers. Each registered factory
    # therefore makes the call itself; this is the guard that the next adapter
    # cannot forget it, and it belongs HERE because "walk every registered
    # jax-suite env" is what this test already does.
    #
    # THE GATE IS PATCHED TO RAISE, so construction stops at it: the check
    # then needs no jax, no mujoco and no GPU, and runs in every venv rather
    # than skipping in the ones CI has. An env that does NOT call it runs on
    # into its own lazy import and raises something else -- caught below and
    # reported as a miss, which is the answer either way.
    import bird.xla_env as xla_env

    class _GateReached(Exception):
        pass

    calls: list = []

    def _record(env_id_arg, strict=True):
        calls.append((env_id_arg, strict))
        raise _GateReached

    monkeypatch.setattr(xla_env, "prepare_for_env", _record)

    never_called = []
    wrong_id = []
    strict_call = []
    for env_id in jax_ids:
        calls.clear()
        try:
            registry.get("env", env_id)(None)
        except _GateReached:
            pass
        except BaseException:  # noqa: BLE001 - "it raised elsewhere" is a miss
            pass
        if not calls:
            never_called.append(env_id)
            continue
        got_id, got_strict = calls[0]
        if got_id != env_id:
            wrong_id.append(f"{env_id} called prepare_for_env({got_id!r})")
        if got_strict is not False:
            strict_call.append(f"{env_id} (strict={got_strict!r})")

    assert not never_called, (
        "these jax-suite envs construct without calling `prepare_for_env`, so a "
        "process that reaches them by any path other than `bird.py::run` or the "
        "spawn child gets no XLA_FLAGS and no XLA_PYTHON_CLIENT_PREALLOCATE:\n  "
        + "\n  ".join(never_called) +
        "\nCall it at the top of the registered factory (or of `__init__` where a "
        "class is what `@register` decorates), before the lazy jax import.")
    assert not wrong_id, (
        "the gate decides by env id -- the suite, and `requires_cuda` off the "
        "registered object -- so an id that is not this env's asks the wrong "
        "question:\n  " + "\n  ".join(wrong_id))
    assert not strict_call, (
        "the construction-site call must pass `strict=False`: a strict call from "
        "inside `__init__` turns a late flag into a dead candidate, and reds every "
        "jax-tier test whose fixture imports jax before constructing an adapter. "
        "The strict calls are the two entry points:\n  " + "\n  ".join(strict_call))


@pytest.mark.jax
def test_the_batched_sampler_threads_the_nominal_draw_as_a_pytree(jax):
    """`jax_toy` is the only batched adapter, so this is the one place
    `BatchedEnvAdapter.sample_transitions`'s DR path threads the
    `physics_rows(n, _dr_nominal)` pytree through `step_batch`. n triples,
    float64, each equal to `ToyReacher._step` under nominal
    physics from the same state and action; `len(out) == n` on the assembled
    list, not the batch's leading dimension; the episode state put back."""
    from bird.envs.toy import ToyReacher

    env = registry.get("env", "jax_toy")(None)
    ref = ToyReacher()
    env.set_dr({"mass": (2.0, 2.0)})                    # a live draw the screen must NOT see
    env.reset(np.random.default_rng(9))
    before = env.episode_state()
    out = env.sample_transitions(np.random.default_rng(0), 7)
    assert len(out) == 7
    for s, a, s2 in out:
        assert s.dtype == np.float64 and s2.dtype == np.float64 and s.shape == (6,)
        ref._dr_now = {**ref._dr_nominal}
        s2_ref, _d, _i = ref._step(s, a)
        assert np.allclose(s2, s2_ref, atol=1e-6), "sampled under NOMINAL physics, not the live draw"
    after = env.episode_state()
    assert after[2] == before[2] and after[2]["mass"] == 2.0, "the episode's own draw is put back after the sample"
    assert bool(np.array_equal(np.asarray(after[0]), np.asarray(before[0]))), "the key is restored"
