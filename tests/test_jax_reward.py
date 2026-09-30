"""`generate.reward_language: jax` -- the reward contract.

Two halves, split by what they need installed, and the split is the point:

* The STATIC half runs with nothing but numpy: the prompt clause, the
  per-language import allowlist, the shape of the two exec namespaces under
  `numpy`, and the two coherence refusals. These are the parts a CPU runner can
  hold to, and they are called DIRECTLY with a config -- not through a run --
  so a machine without jax still proves the contract is worded and gated right.

* The TRACED half carries `@pytest.mark.jax` and `importorskip("jax")`: a pure
  `jnp` body passes every dynamic check under `jax.jit(jax.vmap(...))`, the
  three ways a correct numpy reward fails to trace are each refused and NAMED,
  the probe is one traced call and not eight, and the training namespace and
  the observation lattice agree with the verifier. It skips in the default CI
  job (no jax installed); a developer with the extra runs it locally.

Every jax-language context here is a NUMPY env (`toy_reacher`) with the key
flipped after load. `config._check_coherence` refuses that pairing at load,
correctly -- nothing would run it -- and that refusal is itself under test
below; the checks are then exercised on the resulting context because what
they read is the language and a batch of transitions, and a toy reacher's
transitions are as good a batch as any. `jax_toy`, the env that makes the
pairing legal end to end, is a separate package.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components import generation as G
from bird.components import training as T
from bird.components import verification as V
from bird.config import ConfigError, _check_coherence, load, reward_language
from bird.context import Context
from bird.types import Candidate

REPO = Path(__file__).resolve().parents[1]
EUREKA = REPO / "configs" / "methods" / "eureka.yaml"

registry.load_all()

#: A reward the contract admits: array ops only, `jnp.where` for the branch,
#: every returned value a scalar.
PURE_JNP = '''
import jax.numpy as jnp
def compute_reward(state, action=None, next_state=None):
    d = jnp.linalg.norm(state[0:2] - state[4:6])
    bonus = jnp.where(d < 0.22, 1.0, 0.0)
    eff = jnp.sum(action * action)
    comps = {"reach": -d, "bonus": bonus, "ctrl": -0.01 * eff}
    return comps["reach"] + comps["bonus"] + comps["ctrl"], comps
'''

#: Correct numpy; refused by a trace: a Python `if` on a traced value.
BRANCH_ON_VALUE = '''
def compute_reward(state, action=None, next_state=None):
    d = jnp.linalg.norm(state[0:2] - state[4:6])
    if d < 0.22:
        return 1.0, {"bonus": 1.0}
    return -d, {"bonus": 0.0}
'''

#: Correct numpy; refused by a trace: `float()` on a traced value.
FLOAT_OF_TRACER = '''
def compute_reward(state, action=None, next_state=None):
    d = float(jnp.linalg.norm(state[0:2] - state[4:6]))
    return -d, {"reach": -d}
'''

#: Traces fine and is wrong: log of a negative is NaN on every row.
NAN_BODY = '''
def compute_reward(state, action=None, next_state=None):
    d = jnp.linalg.norm(state[0:2] - state[4:6])
    return jnp.log(-d), {"reach": jnp.log(-d)}
'''

OS_IMPORT = "import os\ndef compute_reward(state, action=None, next_state=None):\n    return 0.0, {}\n"


def _cand(cid: str, code: str) -> Candidate:
    return Candidate(cand_id=cid, iteration=0, reward_code=code)


def _ctx(language: str, **overrides) -> Context:
    """A tester-tier context on `toy_reacher` with the language set AFTER load.

    After, not through `overrides`: `jax` on a numpy env is refused at load
    (`test_coherence_refuses_jax_on_a_numpy_env`), and these tests want the
    context that refusal protects a run from, so the key is flipped on the
    loaded object. `Config.get` walks the live data, so every reader sees it.
    """
    base = {"output.tracker": "none", "llm.generator.provider": "mock",
            "llm.evaluator.provider": "mock", "verify.smoke_test_steps": 3}
    base.update(overrides)
    cfg = load(EUREKA, profile="tester", overrides=base)
    cfg._data["generate"]["reward_language"] = language
    assert cfg.get("generate.reward_language") == language
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(0))
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    return ctx


def _checks(ctx: Context, code: str, names=("execution_smoke", "output_shape", "output_dtype",
                                             "finite_values", "non_constant_reward")):
    cand = _cand("c", code)
    fn, err = V.compile_reward(ctx, cand)
    assert fn is not None, err
    return {n: registry.get("dynamic_check", n)(ctx, cand, fn) for n in names}


# ==========================================================================
# the static half: no jax needed
# ==========================================================================


def test_reward_language_reads_numpy_off_anything_without_the_key():
    """The one reader defaults on a bare object, a `.get` stub, and a loaded config
    (whose default IS numpy), so a frozen pre-key config reads the same everywhere."""
    assert reward_language(SimpleNamespace()) == "numpy"
    assert reward_language(SimpleNamespace(get=lambda k, d=None: None)) == "numpy"
    assert reward_language(load(EUREKA, profile="tester")) == "numpy"
    assert reward_language(_ctx("jax").cfg) == "jax"


@pytest.mark.parametrize("fmt", ["component_dict_return", "component_dict_plus_weights",
                                 "scalar_only"])
def test_the_prompt_carries_the_tracing_clause_only_under_jax(fmt):
    """Every code-writing output format shows the signature; under `jax` it also
    names the trace and the three idioms, and under `numpy` not a word of it."""
    prompt_np = registry.get("output_format", fmt)(_ctx("numpy"))
    prompt_jx = registry.get("output_format", fmt)(_ctx("jax"))
    for prompt in (prompt_np, prompt_jx):
        assert G._SIGNATURE in prompt
    for token in ("jax.jit", "jax.vmap", "jnp.where", "`.item()`", ".at[...].set(...)"):
        assert token in prompt_jx, token
        assert token not in prompt_np, token
    # The clause sits directly under the signature, before the format's own
    # return-shape sentence, so the two are read as one contract.
    assert prompt_jx.index(G._SIGNATURE) < prompt_jx.index("jax.jit") < prompt_jx.index("Put the function" if fmt != "component_dict_plus_weights" else "```json")


def test_import_allowlist_admits_jax_only_under_jax():
    """`import jax.numpy` is a contract breach under numpy and the contract under
    jax; `os` is refused under both. The static check and the exec-time guard
    read the same per-language list."""
    jx, npy = _ctx("jax"), _ctx("numpy")
    assert V.static_import_allowlist(jx, _cand("a", PURE_JNP)) == ""
    verdict = V.static_import_allowlist(npy, _cand("a", PURE_JNP))
    assert "jax.numpy" in verdict and "import_allowlist" in verdict
    for ctx in (jx, npy):
        assert "os" in V.static_import_allowlist(ctx, _cand("b", OS_IMPORT))
    assert V.import_allowlist("numpy") == V.IMPORT_ALLOWLIST
    assert V.import_allowlist("jax") == V.IMPORT_ALLOWLIST | {"jax"}
    # exec-time: the guard in the numpy namespace refuses what the static check refused
    fn, err = V.compile_reward(npy, _cand("a", PURE_JNP))
    assert fn is None and "jax.numpy" in err and "not permitted" in err


def test_the_numpy_namespaces_are_unchanged_by_the_tier_existing():
    """Neither exec namespace grows a name under `numpy`: verifier and trainer
    both grant exactly np/numpy/math (plus the verifier's fence and torch when
    present). A numpy run's prompt promises no `jnp`, so its namespace has none."""
    ver = set(V._restricted_globals("numpy")) - {"__builtins__", "__name__", "torch"}
    assert ver == {"np", "numpy", "math"}
    assert set(T._reward_namespace("numpy")) == {"np", "numpy", "math"}
    assert V.JAX_PROBE_BATCH == 8  # the fixed probe batch; not `verify.smoke_test_steps`


def test_coherence_refuses_jax_on_a_numpy_env():
    """Nothing would run a jnp reward against a numpy env; refused at load with
    the env named, never downgraded."""
    with pytest.raises(ConfigError) as exc:
        load(EUREKA, profile="tester", overrides={"generate.reward_language": "jax"})
    msg = str(exc.value)
    assert "reward_language=jax" in msg and "non-jax env" in msg and "toy_reacher" in msg


def test_coherence_refuses_numpy_on_a_jax_env():
    """The other direction, through the suite table and not a prefix test here:
    a `jax_*` id under `numpy` is refused, because the batched path would call
    the reward row by row on the host and no counter would show it."""
    cfg = load(EUREKA, profile="tester")
    cfg._data["problem"]["env_id"] = "jax_toy"
    problems = [p for p in _check_coherence(cfg) if "reward_language" in p]
    assert problems and "needs generate.reward_language=jax" in problems[0]
    assert "jax_toy" in problems[0]
    cfg._data["generate"]["reward_language"] = "jax"
    assert not [p for p in _check_coherence(cfg) if "reward_language" in p]


# ==========================================================================
# the traced half: needs jax
# ==========================================================================


@pytest.fixture
def jax():
    return pytest.importorskip("jax")


@pytest.mark.jax
def test_a_pure_jnp_reward_passes_every_traced_check(jax):
    verdicts = _checks(_ctx("jax"), PURE_JNP)
    assert verdicts == {k: "" for k in verdicts}, verdicts


@pytest.mark.jax
def test_the_probe_is_one_traced_call_over_eight_rows(jax):
    """Batched, not looped: under `jax` the reward's Python body runs ONCE (the
    trace) for eight transitions; under `numpy` it runs once per transition.
    Counted on the candidate function itself, which both binders reach."""
    import functools

    calls = []

    def counting(fn):
        @functools.wraps(fn)  # `inspect.signature` follows `__wrapped__`, so both binders see the real plan
        def wrapped(*a, **k):
            calls.append(1)
            return fn(*a, **k)
        return wrapped

    ctx = _ctx("jax")
    fn, _ = V.compile_reward(ctx, _cand("c", PURE_JNP))
    rows, err = V._probe(ctx, counting(fn), 3)
    assert err == "" and len(rows) == V.JAX_PROBE_BATCH == 8
    assert len(calls) == 1, "a trace runs the body once, whatever the batch"
    total, comps = V.split_reward(rows[0])
    assert np.ndim(total) == 0 and set(comps) == {"reach", "bonus", "ctrl"}
    assert all(np.ndim(v) == 0 for v in comps.values())

    calls.clear()
    npy = _ctx("numpy")
    fn_np, _ = V.compile_reward(npy, _cand("n", "import numpy as np\ndef compute_reward(state, action=None, next_state=None):\n    return -float(np.linalg.norm(state[0:2] - state[4:6])), {}\n"))
    rows, err = V._probe(npy, counting(fn_np), 3)
    assert err == "" and len(rows) == 3 == len(calls)


@pytest.mark.jax
def test_python_control_flow_on_a_value_fails_to_trace_and_is_named(jax):
    """A correct numpy reward with `if d < 0.22:` is refused -- and the verdict
    names the jax error class and the candidate's own line, which is what a
    repair prompt has to quote. Eager `jnp` would have passed this program."""
    verdicts = _checks(_ctx("jax"), BRANCH_ON_VALUE)
    for name, v in verdicts.items():
        assert v.startswith(f"{name}: trace under jax.jit(jax.vmap(...)) failed:"), v
        assert "TracerBoolConversionError" in v and "<candidate c> line 4" in v, v


@pytest.mark.jax
def test_float_of_a_traced_value_fails_to_trace_and_is_named(jax):
    v = _checks(_ctx("jax"), FLOAT_OF_TRACER, names=("execution_smoke",))["execution_smoke"]
    assert "ConcretizationTypeError" in v and "<candidate c> line 3" in v, v


@pytest.mark.jax
def test_a_nan_is_still_caught_after_the_trace(jax):
    """Tracing is not the only verdict: a body that traces and returns NaN on
    every row is refused by `finite_values` exactly as under numpy."""
    verdicts = _checks(_ctx("jax"), NAN_BODY)
    assert verdicts["execution_smoke"] == "" and verdicts["output_shape"] == ""
    assert verdicts["finite_values"] == "finite_values: total is nan"


@pytest.mark.jax
def test_the_two_exec_namespaces_grant_the_same_names_under_jax(jax):
    """Verifier and trainer must agree on what a program may reach for, or a
    candidate passes §2 and dies in §3 on a NameError -- the failure
    `compile_reward`'s shared-entry-point docstring exists to rule out."""
    ver = set(V._restricted_globals("jax")) - {"__builtins__", "__name__", "torch"}
    assert ver == set(T._reward_namespace("jax")) == {"np", "numpy", "math", "jnp", "jax"}
    compiled = T.compile_reward(PURE_JNP, language="jax")
    assert compiled.name == "compute_reward"
    # The trainer has no fence, so the difference shows at CALL time: the same
    # body compiled under numpy has no `jnp` bound and dies on its first row.
    bare = PURE_JNP.replace("import jax.numpy as jnp\n", "")
    assert T.compile_reward(bare, language="jax").fn(np.zeros(6), np.zeros(2))
    with pytest.raises(NameError, match="jnp"):
        T.compile_reward(bare, language="numpy").fn(np.zeros(6), np.zeros(2))


@pytest.mark.jax
def test_the_observation_lattice_is_traced_once_under_jax(jax):
    """LIMEN under `jax`: phi is measured through `jax.jit(jax.vmap(...))` over the
    whole probe lattice in one call; a phi that will not trace has no width and
    is refused, as a phi that raises on every probe is under numpy."""
    env = _ctx("jax").env
    phi = T.compile_observation(
        "def get_observation(state):\n"
        "    return jnp.concatenate([state[0:2] - state[4:6], state[2:4]])\n", env, language="jax")
    assert phi.dim == 4 and phi.language == "jax" and phi.n_probe_failures == 0
    assert phi(np.zeros(6)).shape == (4,)
    with pytest.raises(ValueError, match="raised on every one of"):
        T.compile_observation("def get_observation(state):\n    return [float(state[0])]\n",
                              env, language="jax")


@pytest.mark.jax
def test_row_applies_the_binder_and_the_weights_on_device(jax):
    """`CompiledReward.traced()` is the seam the jax training backend traces: the candidate's argument
    plan and the LLM's weights are applied inside it, in jnp, so a training
    backend holds neither convention. Checked against `__call__` row by row."""
    import jax.numpy as jnp

    reward = T.compile_reward(PURE_JNP, language="jax")
    reward.weights = {"reach": 2.0, "bonus": 0.5, "ctrl": 1.0}
    env = _ctx("jax").env
    trs = V.sample_transitions(_ctx("jax"), 8)
    S = jnp.asarray(np.stack([t.state for t in trs]))
    A = jnp.asarray(np.stack([t.action for t in trs]))
    S2 = jnp.asarray(np.stack([t.next_state for t in trs]))
    total, comps = reward.traced()(S, A, S2)
    assert total.shape == (8,) and set(comps) == {"reach", "bonus", "ctrl"}
    row_total, _ = jax.jit(jax.vmap(reward.row()))(S, A, S2)  # the backend's own jit over the primitive
    assert np.allclose(np.asarray(row_total), np.asarray(total))
    assert reward.component_names == ["reach", "bonus", "ctrl"]
    for i, t in enumerate(trs):
        host_total, host_comps = reward(t.state, t.action, t.next_state)
        assert float(total[i]) == pytest.approx(host_total, rel=1e-5)
        for k, v in host_comps.items():
            assert float(comps[k][i]) == pytest.approx(v, rel=1e-5)


@pytest.mark.jax
def test_traced_binds_a_one_argument_reward_and_a_bare_scalar(jax):
    import jax.numpy as jnp

    reward = T.compile_reward("def compute_reward(s):\n    return -jnp.sum(s[0:2] ** 2)\n",
                              language="jax")
    S = jnp.ones((4, 6))
    total, comps = reward.traced()(S, jnp.zeros((4, 2)), S)
    assert comps == {} and np.allclose(np.asarray(total), -2.0)


@pytest.mark.jax
def test_traced_refuses_a_vector_component_at_trace_time(jax):
    """`_scalar` averages a vector on the host; the traced path refuses it and
    names the component, because a mean is a different reward."""
    import jax.numpy as jnp

    reward = T.compile_reward(
        "def compute_reward(state, action=None, next_state=None):\n"
        "    return jnp.sum(state), {'vec': state[0:2]}\n", language="jax")
    with pytest.raises(TypeError, match="component 'vec' has shape \\(2,\\)"):
        reward.traced(with_action=False, with_next=False)(jnp.ones((3, 6)), None, None)
    # ... and the verifier's probe files the same refusal, named, as a trace failure.
    v = _checks(_ctx("jax"), "def compute_reward(state, action=None, next_state=None):\n"
                "    return jnp.sum(state), {'vec': state[0:2]}\n", names=("output_shape",))
    assert "trace under jax.jit(jax.vmap(...)) failed: TypeError: reward component 'vec'" in v["output_shape"]
