"""Stage 2, validity half: static checks, dynamic checks, failure routing, dedup.

Two populations share stage 2 and must never be collapsed: **validity** checks
are near-infallible w.r.t. their contract and address the *producer* -- a
semantically excellent reward cannot fail them, so their rejections are
recorded with ``failure_kind="invalid"``. **Quality screens**
(``bird/components/screens.py``) are fallible judgments against accumulated
evidence and address the *budget*; they record ``failure_kind="screened"``. The
run artifact keeps them apart because they answer different questions about a
run.

Method points implemented here:

* Eureka        -- ``verify.enabled: false`` in the paper; errors surface only
                   when the RL launch dies, so the slot is forfeited at the
                   failure sentinel. That is ``on_failure: record_neg_inf``.
* GT            -- ``resample_with_trace``, uncapped ("until they pass", §4).
* CARD          -- ``resample_blind`` capped at 10, then ``abort_run``.
* LIMEN         -- ``teach_next_iteration``: drop now, quote the traceback into
                   the next prompt; plus ``non_constant_reward`` /
                   ``output_shape`` dynamic filters.
* (repo-authored) -- ``degrade``: salvage what parses, clip into bounds, train
                   anyway. Not DrEureka's: its release has no ``parse_dr`` salvage
                   step and pastes the block verbatim (dr_eureka.py:109-128).
* LIMEN (again) -- ``check_order: staged`` + ``stage_timeouts_s``: its
                   ``crash_filter.py`` is three stages (static / import /
                   execute) with 5 / 10 / 30 s deadlines and a short circuit,
                   which a flat unordered list cannot express.
* L2R           -- ``human_confirm_before_execute``: its
                   ``confirmation_safe_executor.py`` shows generated code to a
                   person and waits for "yes" before running it. A human
                   *inside* the validity gate, distinct from
                   ``select.human_override`` (after training) and
                   ``evaluate.human.mode`` (grading behaviour).

The ``phase:validity`` registry name is NOT registered here -- ``phases.py``
owns the ``phase`` family and registers it by importing :func:`run_validity`
from this module. Everything else in stage 2's validity half is registered
below.

Provenance markers: **†** paper and released code
disagree; **‡** the paper is silent and the released code (or, where noted,
this repo) supplies the value.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import logging
import math
import re
import os
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import lru_cache
from typing import (Any, Callable, Dict, FrozenSet, Iterator, List, NamedTuple,
                    Optional, Sequence, Tuple)

import numpy as np

from ..config import reward_language
from ..parsing import extract_code
from ..registry import register
from ..types import Candidate, CandidateReport, TrainResult

log = logging.getLogger("bird")


class SamplerFailed(RuntimeError):
    """The env's OWN `sample_transitions` raised or returned nothing.

    Raised, never swallowed, because the alternative is a silent failure: a
    sampler exception logged at DEBUG would let the harness fall through to a
    gym-shaped rollout -- which on every `EnvAdapter` dies on `step(state,
    action)` called without a state -- and then to Gaussian draws, so the
    EPIC/STARC/`policy_rank_corr` screens would score noise while every counter
    read normal (`tests/test_verification_sampling.py` exists to catch exactly
    this). A harness fault, not a candidate's:
    it propagates out of the check and the stage rather than becoming a verdict.
    """


class VerificationAborted(RuntimeError):
    """Raised by ``verify.on_exhaustion: abort_run``.

    † CARD's released code calls ``exit()`` when the repair cap is hit; the
    honest translation into a library is an exception that leaves ``run()``.
    ``bird.py`` catches only ``BudgetExceeded``, so this propagates -- which is
    the point.
    """


# ==========================================================================
# Knobs the schema deliberately does not expose
# ==========================================================================
#
# The schema has `verify.forbidden_symbols` but no import allowlist and no
# magnitude bound: no published method has either, so inventing YAML keys for
# them would fabricate pins. They live here as named constants instead, where a
# reader can see they are this repo's choice and not a paper's.

#: Roots a generated reward program may import. Everything else -- `os`,
#: `sys`, `subprocess`, `pickle`, `importlib`, ... -- is refused, statically by
#: `import_allowlist` and again at exec time by the guarded `__import__` in the
#: restricted namespace, so a dynamic import cannot walk around the static one.
IMPORT_ALLOWLIST = frozenset({
    "math", "numpy", "typing", "collections", "itertools", "functools",
    "operator", "dataclasses", "enum", "abc", "statistics", "random",
    "torch", "gym", "gymnasium",
})

#: The roots `generate.reward_language: jax` ADDS: `jax`
#: covers `jax.numpy`, `jax.lax`, `jax.nn` -- `_guarded_import` and the static
#: check both test the root. Conditional on the language rather than folded
#: into `IMPORT_ALLOWLIST`, because the numpy contract is unchanged by the tier
#: existing: a numpy-language candidate that imports jax is a candidate nothing
#: would run (`config._check_coherence`, the reward_language block), and the
#: static check saying so at §2 is cheaper than a training slot saying it.
JAX_IMPORT_ROOTS = frozenset({"jax"})

#: How many sampled transitions the batched probe traces the candidate over
#: under `reward_language: jax`. A BATCH, not a loop: the contract is "this
#: function runs under `jax.jit(jax.vmap(...))`", which one traced call over 8
#: rows tests and 8 eager calls do not -- eager `jnp` accepts Python control
#: flow on values that a trace refuses. The verify budget
#: `verify.smoke_test_steps` is the NUMPY count and is not consulted here, so
#: a method's published smoke budget is not silently re-read as a batch width.
JAX_PROBE_BATCH = 8

#: How many sampled transitions the batched probe calls the candidate over
#: under `reward_language: torch`. A BATCH, not a loop, for the reason
#: `_probe_batched_torch` argues: the contract is "this function is called once
#: per env step with `(num_envs, *)` tensors", which one batched call tests and
#: `n` single-row calls do not -- a body written for one transition broadcasts
#: silently on a batch rather than raising. 8 for parity with the jax probe
#: and for the same reason it is not `verify.smoke_test_steps`: a method's
#: published smoke budget is a COUNT OF TRANSITIONS and must not be silently
#: re-read as a batch WIDTH. It is deliberately not the fleet size either --
#: 4096 rows would make §2 pay a fraction of a training to check a shape.
TORCH_PROBE_BATCH = 8


def import_allowlist(language: str = "numpy") -> frozenset:
    """The import roots a candidate may use under `generate.reward_language`.

    `torch` needs no addition: it is already in `IMPORT_ALLOWLIST` (many
    published programs are written against it, whatever the language key says)
    and already injected into `_restricted_globals` when installed. So
    `reward_language: torch` widens nothing here. The namespace that has to
    match is the TRAINING one (`training._reward_namespace`), which carries
    torch as well, so a torch candidate that passes §2 does not fail at §3.
    """
    return IMPORT_ALLOWLIST | JAX_IMPORT_ROOTS if language == "jax" else IMPORT_ALLOWLIST

#: `bounded_magnitude`'s bound. Large enough that a sane dense reward never
#: trips it; small enough that a runaway term (the usual cause of a policy
#: collapsing to one degenerate behaviour) is caught before we buy a training.
MAX_REWARD_MAGNITUDE = 1e6

#: Entrypoint names a generated program may use. Eureka/DrEureka/LIMEN/CARD all
#: emit `compute_reward`; Text2Reward emits `compute_dense_reward`.
REWARD_FN_NAMES = ("compute_reward", "compute_dense_reward", "reward",
                   "get_reward", "reward_fn")

#: Backstop for `verify.max_repair_attempts: 0` ("uncapped", GT §4). A truly
#: unbounded loop plus a deterministic mock LLM is a hang, and the test suite
#: runs whole methods end to end. Hitting the backstop routes to
#: `verify.on_exhaustion` exactly as hitting a configured cap would, so the
#: observable semantics of "uncapped" only change after 100 wasted samples.
UNCAPPED_SAFETY_LIMIT = 100

#: Characters per token, for charging repairs when the LLM client reports no
#: usage of its own -- a FALLBACK, never an addition: `_regenerate` snapshots
#: `budget.llm_calls` around the call and estimates only if the client recorded
#: nothing, exactly as `generation._call_llm` does. Crude, but
#: `verify.repairs_count_against_budget` exists to make token-cost comparisons
#: fair, and silently charging zero is worse.
CHARS_PER_TOKEN = 4

#: `generate.output.format` values that promise a named-component dict. Read as
#: a config VALUE (never a method name): the shape check has to know what the
#: producer was told to emit, or it cannot tell a contract breach from a
#: legitimately scalar reward.
_COMPONENT_FORMATS = frozenset({"component_dict_return", "component_dict_plus_weights"})


# ==========================================================================
# Compiling a candidate into a callable
# ==========================================================================


class Transition(NamedTuple):
    """One ``(s, a, s')`` triple. The unit every check and screen re-scores."""

    state: Any
    action: Any
    next_state: Any


class RewardShapeError(TypeError):
    """The reward returned something that is not a total (+ optional dict)."""


_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "callable", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "getattr", "hasattr", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "object", "pow", "print", "range", "repr", "reversed", "round",
    "set", "setattr", "slice", "sorted", "str", "sum", "tuple", "type", "zip",
    "classmethod", "staticmethod", "property", "super",
    # `class Foo:` compiles to a __build_class__ call; the Eureka lineage emits
    # its reward as a method, so refusing this would refuse a published shape.
    "__build_class__",
    # exceptions a reward body may legitimately raise or catch
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "ZeroDivisionError", "RuntimeError", "NotImplementedError",
)


def _make_guarded_import(allowed: frozenset) -> Callable[..., Any]:
    """`__import__` for the restricted namespace: allowlist or nothing.

    (`fromlist` arrives as None for a plain `import x`, which is why it is not
    defaulted to an empty tuple.)
    """
    def _guarded_import(name: str, globals_: Any = None, locals_: Any = None,
                        fromlist: Optional[Sequence[str]] = None, level: int = 0) -> Any:
        root = name.split(".")[0]
        if root not in allowed:
            raise ImportError(f"import of {name!r} is not permitted in a reward program")
        return builtins.__import__(name, globals_, locals_, list(fromlist or ()), level)
    return _guarded_import


#: The numpy-contract guard, under the module-level name its callers bind.
_guarded_import = _make_guarded_import(IMPORT_ALLOWLIST)


def _jax_namespace() -> Dict[str, Any]:
    """`jnp` and `jax` for a `reward_language: jax` namespace.

    Imported HERE and nowhere at module scope: `registry.load_all()` imports
    this module on every run, on boxes with no jax and no GPU. Raises
    `ImportError` naming the extra when jax is absent -- a harness fault, not a
    candidate's, so it is NOT caught into a `failure_kind="invalid"` row: the
    coherence check has already tied `jax` to a `jax_*` env whose factory
    imports jax first, so reaching this line without it is a broken install.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover -- depends on the machine
        raise ImportError("generate.reward_language=jax needs the `jax` extra "
                          "(`uv sync --extra jax`, or scripts/setup_jax.sh); "
                          f"import failed: {exc}") from exc
    return {"jnp": jnp, "jax": jax}


def _restricted_globals(language: str = "numpy") -> Dict[str, Any]:
    """The namespace a candidate program is exec'd in.

    Not a security sandbox -- Python has no such thing at this level, and the
    threat model here is a confused LLM, not an adversary. It is a *contract*
    boundary: the program gets numeric tooling and nothing that touches the
    filesystem, the network, or the harness's own state.

    `language` is `generate.reward_language`. Under
    `jax` the program also sees `jnp` and `jax`, and may import them; `np`
    stays, for constants (`np.pi`, a shape) -- the prompt says so in the same
    words, and a namespace narrower than the prompt would fail programs the
    model was told to write.
    """
    safe = {n: getattr(builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(builtins, n)}
    safe["__import__"] = _make_guarded_import(import_allowlist(language))
    ns: Dict[str, Any] = {
        "__builtins__": safe,
        "__name__": "bird_reward_candidate",
        "np": np,
        "numpy": np,
        "math": math,
    }
    if language == "jax":
        ns.update(_jax_namespace())
    try:  # optional: many published programs are written against torch
        import torch  # noqa: F401  (lazy: an optional dependency)

        ns["torch"] = torch
    except Exception:  # pragma: no cover -- torch is not a dependency
        pass
    return ns


class _SelfProxy:
    """Stand-in for the ``self`` of a generated ``compute_reward(self, ...)``.

    Eureka-lineage programs are emitted as *methods* of the environment class
    and read task state off ``self``. We have no environment class here, so we
    forward attribute reads to the env adapter; a missing attribute raises
    ``AttributeError``, which is exactly the smoke failure we want to record.
    """

    __slots__ = ("_env",)

    def __init__(self, env: Any) -> None:
        object.__setattr__(self, "_env", env)

    def __getattr__(self, item: str) -> Any:
        env = object.__getattribute__(self, "_env")
        if env is None:
            raise AttributeError(f"reward program read self.{item} but no env is attached")
        return getattr(env, item)


def _find_reward_node(tree: ast.Module) -> Optional[ast.FunctionDef]:
    """The entrypoint: a named reward function, else the sole top-level def,
    else a named method on a top-level class (the Eureka emission shape)."""
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for n in funcs:
        if n.name in REWARD_FN_NAMES:
            return n  # type: ignore[return-value]
    if len(funcs) == 1:
        return funcs[0]  # type: ignore[return-value]
    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for n in cls.body:
            if isinstance(n, ast.FunctionDef) and n.name in REWARD_FN_NAMES:
                return n
    return None


def compile_reward(ctx: Any, candidate: Candidate) -> Tuple[Optional[Callable[..., Any]], str]:
    """Exec ``candidate.reward_code`` in a restricted namespace.

    Returns ``(fn, "")`` or ``(None, reason)``. Never raises: a candidate that
    explodes at import time is a *result* of stage 2, not an error in it.
    """
    code = candidate.reward_code or ""
    if not code.strip():
        return None, "empty reward program"
    ns = _restricted_globals(reward_language(getattr(ctx, "cfg", None)))
    try:
        exec(compile(code, f"<candidate {candidate.cand_id}>", "exec"), ns)
    except SyntaxError as exc:
        return None, f"compile: {type(exc).__name__}: {exc.msg} (line {exc.lineno})"
    except Exception as exc:
        return None, f"compile: {type(exc).__name__}: {exc}"

    fn = None
    for name in REWARD_FN_NAMES:
        obj = ns.get(name)
        if callable(obj):
            fn = obj
            break
    if fn is None:
        # A class-shaped emission: bind the method, `self` is supplied at call
        # time by _SelfProxy.
        for obj in ns.values():
            if isinstance(obj, type):
                for name in REWARD_FN_NAMES:
                    m = getattr(obj, name, None)
                    if callable(m):
                        fn = m
                        break
            if fn is not None:
                break
    if fn is None:
        defined = sorted(k for k, v in ns.items() if callable(v) and not k.startswith("_"))
        return None, (f"no reward entrypoint: expected one of {list(REWARD_FN_NAMES)}, "
                      f"defined {defined}")
    return fn, ""


@lru_cache(maxsize=2048)
def _positional_params_cached(fn: Callable[..., Any]) -> Tuple[Tuple[str, ...], bool]:
    """`_positional_params`, memoised on the function OBJECT.

    A reward program's signature cannot change between calls, but `call_reward`
    asks for it on every single evaluation, and the pseudometric screens
    evaluate one candidate tens of thousands of times: `epic_canonical_profile`
    is `n_transitions * (2 * CANON_SAMPLES + 1) + CANON_SAMPLES` calls, which at
    the reported n=1024 is ~263,000 `inspect.signature` invocations per
    candidate, all returning the same answer.

    Measured on one `gym_reacher_reach` candidate (n=256, 65,920
    evaluations): `inspect.signature`
    and its callees were **3.04 s of 6.92 s -- 44% of the whole computation**,
    against 3.12 s for the reward program itself. Caching is not a micro-tune
    here; the introspection cost was comparable to the arithmetic it was
    introspecting.

    Cached as a TUPLE and copied to a list by the caller. An `lru_cache` handing
    out one shared mutable list would turn any caller's in-place edit into a
    permanent wrong answer for that function -- no caller does that today, and
    the point is that none can start.

    Bounded rather than unbounded: the key holds a strong reference to the
    function, which keeps its whole `exec` namespace alive, and a sweep compiles
    thousands of candidates.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return (), True
    names: List[str] = []
    star = False
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            names.append(p.name)
        elif p.kind is p.VAR_POSITIONAL:
            star = True
    return tuple(names), star


def _positional_params(fn: Callable[..., Any]) -> Tuple[List[str], bool]:
    """Positional parameter names (``self`` included) and whether ``*args``."""
    try:
        names, star = _positional_params_cached(fn)
    except TypeError:  # an unhashable callable (a functools.partial on a dict, ...)
        return _positional_params_uncached(fn)
    return list(names), star


def _positional_params_uncached(fn: Callable[..., Any]) -> Tuple[List[str], bool]:
    """The unmemoised path, for a callable that cannot be a cache key."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return [], True
    names, star = [], False
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            names.append(p.name)
        elif p.kind is p.VAR_POSITIONAL:
            star = True
    return names, star


def _defaulted_positional_names(fn: Callable[..., Any]) -> FrozenSet[str]:
    """Positional parameter names of ``fn`` that carry a default.

    A sibling of `_positional_params` rather than a change to it: that one is
    memoised and has three callers, and widening its return shape to serve one
    of them is how a helper grows a second job. Failure is silence (an empty
    set), because a signature this cannot read is one `call_reward` falls back
    to positional binding for anyway.
    """
    try:
        return frozenset(
            prm.name for prm in inspect.signature(fn).parameters.values()
            if prm.kind in (prm.POSITIONAL_ONLY, prm.POSITIONAL_OR_KEYWORD)
            and prm.default is not inspect.Parameter.empty)
    except (TypeError, ValueError):
        return frozenset()


def call_reward(ctx: Any, fn: Callable[..., Any], tr: Transition) -> Any:
    """Call a compiled reward on one transition, adapting to its arity.

    The schema pins the reward *representation* but never the signature -- no
    paper does either, and the four lineages emit three different ones
    (``(self, ...)`` methods, ``(state, action)``, ``(s, a, s')``). Adapting
    here keeps that variation out of every call site.

    Binding is BY PARAMETER NAME, through ``training._ARG_BY_NAME``, so this
    stage and the trainer cannot hand one program different arguments; see the
    comment in the body for the measurement that forced it. Positional binding
    survives only for names the table does not know.
    """
    names, star = _positional_params(fn)
    args: List[Any] = []
    if names and names[0] in ("self", "cls"):
        args.append(_SelfProxy(getattr(ctx, "env", None)))
        names = names[1:]
    full = [tr.state, tr.action, tr.next_state]
    if star:
        args.extend(full)
        return fn(*args)

    # BY NAME FIRST, FROM THE TRAINER'S OWN TABLE. Positional binding with a
    # single name test -- whether the SECOND parameter contains "next" -- hands
    # a signature whose order is not `(state, action, ...)` its arguments
    # backwards. The published Meta-World line is exactly such a signature:
    #
    #     def compute_dense_reward(self, action, obs)
    #
    # and positional binding gives it `action=state`, `obs=action`. Measured,
    # one program through both stages with a 39-wide state of 0..38 and a
    # 4-wide action of -1, with this stage binding positionally:
    #
    #     VERIFY  call_reward     -> action_len=39  obs_len=4   obs[0]=-1.0
    #     TRAIN   CompiledReward  -> action_len=4   obs_len=39  obs[0]=0.0
    #
    # so §2 would score a program on swapped inputs while §3 trains it
    # correctly. On Meta-World `obs` is then the 4-wide ACTION, so every
    # candidate dies `ValueError: operands could not be broadcast together with
    # shapes (3,) (0,)` at `obs[4:7]` -- deterministically, which marks it as a
    # harness defect and not a bad generation.
    #
    # `_ARG_BY_NAME` is IMPORTED rather than restated: the failure is two
    # stages disagreeing about what a program was handed, and a second copy of
    # the table would be the same failure with extra steps. Local import for the
    # reason the jax branch below gives -- `training` is the heavy module.
    #
    # POSITIONAL REMAINS THE FALLBACK, deliberately. `(s, a)`, `(state, action)`
    # and `(s, a, s2)` are all in the table and bind by name to the same
    # arguments positional binding gives them; a program with genuinely
    # unrecognised parameter names (`(x, y)`) is bound positionally rather than
    # refused, because stage 2 exists to run programs, not to grade their
    # naming.
    from .training import _ARG_BY_NAME

    # In declaration order: fill every name the table knows, and STOP at the
    # first name it does not know THAT HAS A DEFAULT -- leaving it and anything
    # after it to their defaults. That last clause keeps name binding from
    # being one parameter wide: `(self, action, obs, extra=0)` is a line away
    # from the pinned signature, and without it that shape would fall through
    # to the positional path and be swapped.
    defaulted = _defaulted_positional_names(fn)
    bound: List[Any] = []
    named_ok = bool(names)
    for nm in names:
        slot = _ARG_BY_NAME.get(nm.lower())
        if slot is not None:
            bound.append(full[slot])
            continue
        if nm in defaulted:
            break  # this one and the rest keep their defaults
        named_ok = False  # an unnamed, non-defaulted parameter: cannot bind by name
        break
    if named_ok and bound:
        args.extend(bound)
        return fn(*args)

    n = len(names)
    if n >= 3:
        args.extend(full)
    elif n == 2:
        # `(state, next_state)` is a real published shape; disambiguate by name.
        args.extend([tr.state, tr.next_state] if "next" in names[1].lower()
                    else [tr.state, tr.action])
    elif n == 1:
        args.append(tr.state)
    return fn(*args)


def split_reward(out: Any) -> Tuple[Any, Dict[str, Any]]:
    """Normalise a reward return into ``(total, components)``.

    ``component_dict_return`` promises ``(total, {name: value})``;
    ``scalar_only`` promises a bare number. Both, plus the bare-dict variant
    some programs emit, land here so no check has to re-derive the convention.
    """
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        return out[0], dict(out[1])
    if isinstance(out, dict):
        if "total" in out:
            comps = {k: v for k, v in out.items() if k != "total"}
            return out["total"], comps
        return sum(_as_float(v) for v in out.values()), dict(out)
    if isinstance(out, (tuple, list)):
        raise RewardShapeError(
            f"expected a scalar or (total, dict), got a {type(out).__name__} of len {len(out)}")
    return out, {}


def _as_float(v: Any) -> float:
    """Coerce a reward-ish value to a Python float, or raise RewardShapeError."""
    if isinstance(v, bool):
        raise RewardShapeError("reward value is a bool, not a number")
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    arr = np.asarray(v)
    if arr.dtype.kind not in "fiu":
        raise RewardShapeError(f"reward value has non-numeric dtype {arr.dtype}")
    if arr.size != 1:
        raise RewardShapeError(f"reward value has shape {arr.shape}, expected a scalar")
    return float(arr.reshape(()).item())


def reward_scalar(ctx: Any, fn: Callable[..., Any], tr: Transition) -> float:
    """Total reward for one transition, as a float. The screens' workhorse."""
    total, _ = split_reward(call_reward(ctx, fn, tr))
    return _as_float(total)


# ==========================================================================
# Sampling transitions to check / screen against
# ==========================================================================


def sample_transitions(ctx: Any, n: int) -> List[Transition]:
    """``n`` transitions to exercise a reward on, deterministic given ``ctx.rng``.

    The env adapter family is owned elsewhere, so this probes for the richest
    interface it offers: a dedicated sampler, else a reset/step rollout, else
    synthetic draws shaped by whatever dimensions the adapter advertises. The
    last rung matters -- GT's paper checks validity on *random* states and
    actions, so synthetic draws are a faithful fallback for an env
    that HAS no sampler, not only a convenience.

    THE LADDER IS FOR ENVS WITHOUT A SAMPLER, NEVER FOR A SAMPLER THAT FAILED. An
    env that offers `sample_transitions` and raises, or returns nothing, gets
    `SamplerFailed` -- loud -- rather than the next rung: every `EnvAdapter`
    offers one, the rollout rung cannot even call an `EnvAdapter` (`step(state,
    action)` takes a state the gym-shaped loop does not have), and the synthetic
    rung would let the screens score Gaussian noise silently -- the failure
    `_call_flexible`'s docstring describes.
    """
    n = max(int(n), 1)
    env = getattr(ctx, "env", None)
    rng = getattr(ctx, "rng", None)
    seed = rng.getrandbits(63) if rng is not None else 0
    nprng = np.random.default_rng(seed)

    if env is not None:
        sampler = getattr(env, "sample_transitions", None)
        if callable(sampler):
            try:
                raw = _call_flexible(sampler, n, nprng)
                out = [_coerce_transition(t) for t in raw]
            except Exception as exc:
                raise SamplerFailed(
                    f"{type(env).__name__}.sample_transitions failed ({type(exc).__name__}: "
                    f"{exc}); refusing to fall back to a rollout or to synthetic draws -- "
                    "the screens would score noise while every counter read normal") from exc
            if not out:
                raise SamplerFailed(
                    f"{type(env).__name__}.sample_transitions returned no transitions for "
                    f"n={n}; refusing to fall back to a rollout or to synthetic draws")
            return out[:n]

        out = _rollout_transitions(env, n, nprng)
        if out:
            return out

    dim_s = _first_int(env, ("observation_dim", "obs_dim", "state_dim", "n_obs"), 4)
    dim_a = _first_int(env, ("action_dim", "n_actions", "act_dim"), 2)
    return [Transition(nprng.normal(size=dim_s), nprng.normal(size=dim_a),
                       nprng.normal(size=dim_s)) for _ in range(n)]


#: Probe transitions already drawn in THIS process, keyed by (env, n).
_PROBE_TRANSITIONS: Dict[Tuple[int, str, int], List[Transition]] = {}


def reset_probe_transitions() -> None:
    """Drop the cache. For tests, and for a caller that swaps the env."""
    _PROBE_TRANSITIONS.clear()


def probe_transitions(ctx: Any, n: int) -> List[Transition]:
    """`sample_transitions`, drawn ONCE PER PROCESS, for the traced probe.

    OTHERWISE THE ENV'S COMPILE IS CHARGED TO EACH CANDIDATE'S BUDGET. Against
    an adapter whose first sampling pays an XLA compile of several minutes,
    drawing per candidate made two of eight candidates fail `execution_smoke`
    with "timed out after 60s" on ONE traced call over eight transitions, and
    verify of eight took 31 minutes. The candidates were not slow. The env's
    first `sample_transitions` was, and every candidate paid it again from its
    own 60 seconds.

    Drawn once, the compile lands on the first candidate of the process and
    outside every later one's clock, and each smoke becomes what it was meant
    to be: one jit of the reward over rows that already exist.

    EVERY CANDIDATE SEES THE SAME ROWS. For a smoke -- "does this run at all"
    -- that is an improvement: eight candidates judged on identical inputs are
    comparable, where eight independent draws are not.

    SCOPED TO THE TRACED PROBE, NOT PUT INSIDE `sample_transitions`, and the
    scope is the point. The EPIC/STARC screens draw through that function too
    and re-draw per call by design -- a pseudometric over reward functions
    wants its own distribution -- so a cache there would silently change what
    the screens measure. The numpy and torch probes are left alone as well:
    they have no multi-minute compile to amortise, so caching would buy them
    nothing and change their tiers' behaviour for it.

    `BIRD_NO_PROBE_CACHE=1` restores per-call drawing without a code edit.
    """
    if os.environ.get("BIRD_NO_PROBE_CACHE") == "1":
        return sample_transitions(ctx, n)
    env = getattr(ctx, "env", None)
    key = (id(env), type(env).__name__, int(n))
    hit = _PROBE_TRANSITIONS.get(key)
    if hit is None:
        hit = sample_transitions(ctx, n)
        _PROBE_TRANSITIONS[key] = hit
    return hit


#: Parameter names that mean "how many" and "the rng", for `_call_flexible`.
_COUNT_PARAMS = ("n", "n_samples", "num", "num_samples", "count", "size", "k",
                 "batch", "batch_size")
_RNG_PARAMS = ("rng", "np_random", "random_state", "generator", "nprng", "prng")


def _call_flexible(fn: Callable[..., Any], n: int, nprng: Any) -> Any:
    """Call ``fn`` with a count and, if it asks for one, an rng -- binding by
    parameter NAME first, position only as a fallback.

    Name first because the one sampler in this repo puts them the other way
    round: ``EnvAdapter.sample_transitions(self, rng, n)``. Assuming ``n`` is
    positional-first and passing the rng by keyword emits
    ``fn(n, rng=nprng)`` -- ``TypeError: got multiple values for argument
    'rng'`` for *every* adapter in ``bird/envs/``. With that TypeError
    swallowed, `sample_transitions` would never run, the gym-shaped rollout
    rung does not fit ``step(state, action)`` either, and the screens would
    silently score Gaussian draws instead of reachable transitions. On Meta-World that
    is not a degraded screen but no screen at all -- `reference_reward` raises
    `UnknownStateError` on a state the simulator never emitted, so EPIC/STARC/
    `policy_rank_corr` report themselves inert and `keep_top_n` never applies.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return fn(n)
    params = sig.parameters

    kw: Dict[str, Any] = {}
    bound_count = False
    for name, p in params.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD, p.POSITIONAL_ONLY):
            continue
        low = name.lower()
        if not bound_count and low in _COUNT_PARAMS:
            kw[name], bound_count = n, True
        elif low in _RNG_PARAMS:
            kw[name] = nprng
    if bound_count:
        try:
            sig.bind(**kw)
        except TypeError:
            pass  # some other parameter is required; fall through to position
        else:
            return fn(**kw)

    names, star = _positional_params(fn)
    if len(names) >= 2 or star:
        return fn(n, nprng)
    return fn(n)


def _coerce_transition(t: Any) -> Transition:
    if isinstance(t, Transition):
        return t
    if isinstance(t, dict):
        return Transition(t.get("state", t.get("obs")), t.get("action"),
                          t.get("next_state", t.get("next_obs")))
    seq = list(t)
    while len(seq) < 3:
        seq.append(None)
    return Transition(seq[0], seq[1], seq[2])


def _rollout_transitions(env: Any, n: int, nprng: Any) -> List[Transition]:
    """Best-effort gym-shaped rollout. Returns [] if the adapter isn't shaped
    like that; the caller falls through to synthetic draws."""
    reset, step = getattr(env, "reset", None), getattr(env, "step", None)
    if not (callable(reset) and callable(step)):
        return []
    try:
        obs = reset()
        if isinstance(obs, tuple) and len(obs) == 2:  # gymnasium (obs, info)
            obs = obs[0]
        out: List[Transition] = []
        for _ in range(n):
            act = _sample_action(env, nprng)
            res = step(act)
            nxt = res[0] if isinstance(res, tuple) and res else res
            out.append(Transition(obs, act, nxt))
            done = bool(res[2]) if isinstance(res, tuple) and len(res) > 2 else False
            if done:
                obs = reset()
                if isinstance(obs, tuple) and len(obs) == 2:
                    obs = obs[0]
            else:
                obs = nxt
        return out
    except Exception as exc:
        log.debug("env rollout failed (%s); falling back to synthetic draws", exc)
        return []


def _sample_action(env: Any, nprng: Any) -> Any:
    for attr in ("sample_action", "random_action"):
        fn = getattr(env, attr, None)
        if callable(fn):
            return fn()
    space = getattr(env, "action_space", None)
    if space is not None and callable(getattr(space, "sample", None)):
        return space.sample()
    return nprng.normal(size=_first_int(env, ("action_dim", "n_actions", "act_dim"), 2))


def _first_int(obj: Any, attrs: Sequence[str], default: int) -> int:
    for a in attrs:
        v = getattr(obj, a, None)
        if isinstance(v, (int, np.integer)) and int(v) > 0:
            return int(v)
    return default


# ==========================================================================
# Static checks  --  static_check_fn(ctx, candidate) -> "" | reason
# ==========================================================================


def _parse(candidate: Candidate) -> Tuple[Optional[ast.Module], str]:
    try:
        return ast.parse(candidate.reward_code or ""), ""
    except SyntaxError as exc:
        return None, f"{type(exc).__name__}: {exc.msg} (line {exc.lineno}, offset {exc.offset})"


@register("static_check", "signature_parse")
def static_signature_parse(ctx: Any, candidate: Candidate) -> str:
    """A reward entrypoint exists and takes at least one argument.

    In `verify.static_checks` by default, and first in the default list, so it
    also has to survive an unparseable program: a syntax error is reported here
    rather than raised, and `ast_syntax` reports it again if configured. Every
    method with `verify.enabled: true` runs this (§2 Validity).
    """
    tree, err = _parse(candidate)
    if tree is None:
        return f"signature_parse: cannot parse: {err}"
    node = _find_reward_node(tree)
    if node is None:
        defined = [n.name for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        return (f"signature_parse: no reward entrypoint; expected one of "
                f"{list(REWARD_FN_NAMES)}, found {defined or 'nothing'}")
    args = node.args
    n_pos = len(args.posonlyargs) + len(args.args)
    if args.args and args.args[0].arg in ("self", "cls"):
        n_pos -= 1
    if n_pos < 1 and not args.vararg:
        return (f"signature_parse: {node.name}() takes no state argument; a reward "
                f"must be a function of the transition")
    return ""


@register("static_check", "ast_syntax")
def static_ast_syntax(ctx: Any, candidate: Candidate) -> str:
    """The program parses. The cheapest possible validity check, and in every
    published method's implicit list (§2 Validity)."""
    tree, err = _parse(candidate)
    return "" if tree is not None else f"ast_syntax: {err}"


@register("static_check", "import_allowlist")
def static_import_allowlist(ctx: Any, candidate: Candidate) -> str:
    """No imports outside :data:`IMPORT_ALLOWLIST`.

    ‡ No published method has an allowlist; the schema therefore exposes no key
    for it and the list is a repo constant. Enforced again at exec time by the
    guarded ``__import__``, so `importlib`-style evasion fails there too.
    """
    tree, err = _parse(candidate)
    if tree is None:
        return f"import_allowlist: cannot parse: {err}"
    allowed = import_allowlist(reward_language(getattr(ctx, "cfg", None)))
    bad: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] not in allowed]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or mod.split(".")[0] not in allowed:
                bad.append(mod or ".")
    if bad:
        return (f"import_allowlist: disallowed import(s) {sorted(set(bad))}; "
                f"allowed roots {sorted(allowed)}")
    return ""


def _referenced_symbols(tree: ast.Module) -> set:
    """Every identifier a program REACHES FOR, as bare names AND dotted chains.

    A real AST walk, not a substring search: `verify.forbidden_symbols` is an
    anti-reward-hacking gate, and a substring test both misses
    `getattr(env, "success")` and false-positives on the word appearing inside
    a comment or an unrelated identifier. Attribute chains are reassembled so
    an entry like `env.success` matches the access, not merely the leaf.

    A name the program BINDS ITSELF is not a reach: a parameter, an assignment
    or loop target, a nested `def`, a comprehension variable. `success = (z > 0.1
    and up > 0.95)` is the candidate's own sparse bonus computed from the state
    it was handed, and the only way it could read the harness's `success` is
    through an attribute (`env.success`), a `getattr` string, or a FREE name --
    all of which are still collected. MEASURED on one baseline run: every one
    of the gate's 11 rejections was that local variable, in
    a reward whose signature receives arrays and no env at all (a slot forfeited
    each time, since `generate.parse.max_retries: 0` and verify never resamples).

    Binding is resolved PER SCOPE, the way Python resolves it:
    a comprehension's target lives in the comprehension, a nested function's
    locals live in that function, and a class body does not enclose the
    functions defined in it. So `[success for success in xs]` followed by
    `return success` in the enclosing function is a free-name reach and is
    flagged; the flat "any binding anywhere excuses every use" reading would
    have let it through, which for an anti-leak gate is the wrong direction to
    fail in. `global x` makes `x` a reach wherever it is used; `nonlocal x`
    binds to an enclosing function and is not. An imported name is BOTH a
    binding and a reach: it excuses no later free read of itself, because it
    is itself reported.
    """
    found: set = set()

    class _Scope:
        __slots__ = ("node", "parent", "bound", "is_class", "globals_")

        def __init__(self, node: Any, parent: Optional["_Scope"]) -> None:
            self.node, self.parent = node, parent
            self.bound: set = set()
            self.is_class = isinstance(node, ast.ClassDef)
            self.globals_: set = set()

    _COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    _FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
    scope_of: Dict[int, _Scope] = {}      # id(node) -> the scope that node is IN

    # Pass 1: build scopes and what each binds.
    def build(node: ast.AST, scope: _Scope) -> None:
        scope_of[id(node)] = scope
        inner = scope
        if isinstance(node, _FUNCTIONS + (ast.ClassDef,) + _COMPREHENSIONS):
            if not isinstance(node, _COMPREHENSIONS + (ast.Lambda,)):
                scope.bound.add(node.name)          # the def/class name binds OUTSIDE
            inner = _Scope(node, scope)
            if isinstance(node, _FUNCTIONS):
                a = node.args
                for arg in (a.posonlyargs + a.args + a.kwonlyargs
                            + [x for x in (a.vararg, a.kwarg) if x is not None]):
                    inner.bound.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            scope.bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            scope.bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                name = a.asname or a.name.split(".")[0]
                scope.bound.add(name)
                # An import binds the name AND is a reach -- the canonical
                # reach for something outside the program, which is the set
                # this function collects. Binding alone would let
                # `from helpers import success; success(state)` through the
                # gate with only the exec-time allowlist left to catch it.
                found.add(name)
        elif isinstance(node, ast.Global):
            scope.globals_.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            scope.bound.update(node.names)          # bound in an enclosing function
        for child in ast.iter_child_nodes(node):
            build(child, inner)

    root = _Scope(tree, None)
    build(tree, root)

    def is_bound(name: str, scope: _Scope) -> bool:
        """Python's rule, near enough: own scope, then enclosing FUNCTION
        scopes (a class body is skipped unless it is the scope itself), then
        module. A `global` declaration in the reading scope wins over all."""
        if name in scope.globals_:
            return False
        s: Optional[_Scope] = scope
        first = True
        while s is not None:
            if (first or not s.is_class) and name in s.bound:
                return True
            first = False
            s = s.parent
        return False

    def chain(node: ast.AST) -> Optional[str]:
        parts: List[str] = []
        cur: Any = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
        return None

    # Pass 2: reaches.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and not is_bound(node.id, scope_of[id(node)]):
                found.add(node.id)
        elif isinstance(node, ast.Global):
            found.update(node.names)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
            dotted = chain(node)
            if dotted:
                found.add(dotted)
        elif isinstance(node, ast.Call):
            # getattr(x, "success") hides the symbol in a string literal.
            fname = node.func.id if isinstance(node.func, ast.Name) else None
            if fname in ("getattr", "hasattr", "setattr"):
                for a in node.args[1:2]:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        found.add(a.value)
                        base = chain(node.args[0]) if node.args else None
                        if base:
                            found.add(f"{base}.{a.value}")
    return found


@register("static_check", "forbidden_symbols")
def static_forbidden_symbols(ctx: Any, candidate: Candidate) -> str:
    """Reject a program that reaches for a symbol in `verify.forbidden_symbols`.

    The anti-reward-hacking gate (§2 Validity): the canonical entries are the
    ground-truth reward and `env.success` -- the quantities the search is
    *benchmarked* against. A candidate that reads them is not a reward, it is a
    leak, and its fitness would be meaningless.

    "Reads" is the operative word: a candidate's OWN local variable that happens
    to share a name with an entry is not a read of anything (see
    `_referenced_symbols`), and rejecting it forfeits a training slot for a
    reward that could not have leaked.

    ‡ No published method has this gate, and nothing enforces it at run time:
    `_SelfProxy.__getattr__` forwards every attribute read to the env adapter
    unchecked, so a name the AST cannot see -- `getattr(self, "succ" + "ess")`,
    `getattr(self, k)`, `vars(self._env)["success"]` -- passes this static gate
    where a literal `self.success` or `getattr(self, "success")` is rejected
    (`_referenced_symbols` collects literal strings). Acceptable under the
    threat model `_restricted_globals` states -- a confused LLM, not an
    adversary -- and the reward the search is benchmarked against is withheld
    from the prompt upstream of this gate anyway (`generation._strip_reward`).
    """
    forbidden = [str(s) for s in (ctx.cfg.get("verify.forbidden_symbols") or [])]
    if not forbidden:
        return ""
    tree, err = _parse(candidate)
    if tree is None:
        return f"forbidden_symbols: cannot parse: {err}"
    seen = _referenced_symbols(tree)
    hits = sorted({f for f in forbidden if f in seen or f.split(".")[-1] in seen})
    if hits:
        return (f"forbidden_symbols: program references {hits}; these are the "
                f"quantities the search is scored against")
    return ""


# ==========================================================================
# Dynamic checks  --  dynamic_check_fn(ctx, candidate, fn) -> "" | reason
# ==========================================================================
#
# PER-CHECK DEADLINES (`verify.stage_timeouts_s`, via `verify.check_order`).
#
# LIMEN gives each of its three validity stages its own deadline -- 5 / 10 /
# 30 s for static / import / execute in `easy_pickup.yaml` -- because they are not equally
# expensive: one number for all three is either too tight for the last or
# useless for the first. The `check_order` plan carries those deadlines as
# `plan.timeouts[check_name]`.
#
# A dynamic check's registered signature is `(ctx, candidate, fn)` and stage 2's
# contract does not include a deadline argument, so widening it would rewrite
# every implementation to thread through a value only `_probe` reads. The
# deadline therefore travels out of band: `_first_failure` publishes it around
# each step and `_probe` reads it. A ContextVar and not a module global so that
# a harness which ever checks two candidates concurrently cannot leak one
# step's deadline into another's, and so the value is always restored.
#
# WHAT THE DEADLINE CAN AND CANNOT BOUND, stated because a pin the code ignores
# is a fabricated pin. It bounds exactly the checks that loop over transitions through
# `_probe` -- which is LIMEN's expensive stage 2, the one the key exists for.
# It does NOT bound a static check (a single AST walk with no loop to check the
# clock between, so a `stage_timeouts_s` entry for one is inert), and it does
# not bound the `exec` in `_compile_gate` (LIMEN's stage 1): interrupting a
# single call needs a subprocess, which `_probe` below declines for the same
# reason. A config that prices `signature_parse` at 5 s is buying
# documentation, not enforcement.

_ACTIVE_TIMEOUT_S: ContextVar[Optional[float]] = ContextVar(
    "bird_verify_active_timeout_s", default=None)


@contextmanager
def _deadline(seconds: Optional[float]) -> Iterator[None]:
    """Make `seconds` the deadline every `_probe` in this block honours.

    ``None`` means "no per-check deadline was priced", which falls back to
    `verify.timeout_s`, the one deadline every check shares.
    """
    token = _ACTIVE_TIMEOUT_S.set(None if seconds is None else float(seconds))
    try:
        yield
    finally:
        _ACTIVE_TIMEOUT_S.reset(token)


def _timeout_s(ctx: Any) -> float:
    """The deadline in force: the running check's, else `verify.timeout_s`.

    Two fallbacks in one line, both deliberate. A check run outside any plan
    (a screen re-using `_probe`, a test calling a check directly) gets the
    global timeout, and a plan that priced only some of its steps leaves the
    rest at the global one rather than unbounded.
    """
    active = _ACTIVE_TIMEOUT_S.get()
    if active is not None:
        return float(active)
    return float(ctx.cfg.get("verify.timeout_s") or 60)


def _probe(ctx: Any, fn: Callable[..., Any], n: int,
           candidate: Optional[Candidate] = None) -> Tuple[List[Any], str]:
    """Call ``fn`` on ``n`` sampled transitions, honouring the deadline in force.

    The timeout is *cooperative*: we check the clock between calls rather than
    killing a thread. A hard kill needs a subprocess, and stage 2 exists to be
    cheap; a program that hangs inside one call is caught by `train.timeout_s`
    when the training backend actually launches it.

    Under `generate.reward_language: jax` this is ONE traced call over a batch
    (`_probe_traced`), and every dynamic check that probes -- `execution_smoke`,
    `output_shape`, `output_dtype`, `finite_values`, `non_constant_reward`,
    `bounded_magnitude` -- inherits that through this one seam. The trace
    contract matters most for the first, second and fourth; the others come
    along because the alternative is two checks in one stage testing two
    different contracts of one program.
    """
    _lang = reward_language(getattr(ctx, "cfg", None))
    if _lang == "jax":
        return _probe_traced(ctx, fn, JAX_PROBE_BATCH, candidate)
    if _lang == "torch":
        return _probe_batched_torch(ctx, fn, TORCH_PROBE_BATCH, candidate)
    budget_s = _timeout_s(ctx)
    deadline = time.time() + budget_s
    outs: List[Any] = []
    for tr in sample_transitions(ctx, n):
        if time.time() > deadline:
            return outs, (f"timed out after {budget_s:g}s "
                          f"({len(outs)}/{n} transitions)")
        try:
            outs.append(call_reward(ctx, fn, tr))
        except Exception:
            tb = traceback.format_exc(limit=6).strip().splitlines()
            return outs, "\n".join(tb[-4:])
    return outs, ""


def _probe_batched_torch(ctx: Any, fn: Callable[..., Any], n: int,
                         candidate: Optional[Candidate] = None
                         ) -> Tuple[List[Any], str]:
    """`fn` on ONE BATCH of `n` sampled transitions as torch tensors.

    THE CONTRACT BEING TESTED IS THE BATCH, and it is a different contract from
    both the numpy probe's and the jax one's. A batched torch candidate is
    called ONCE per env step with `(num_envs, obs_dim)` tensors and must return
    one value per environment -- that is the shape Eureka's prompt asks for
    (`refs/code/Eureka/eureka/utils/prompts/reward_signature.txt`:
    `@torch.jit.script def compute_reward(...) -> Tuple[torch.Tensor,
    Dict[str, torch.Tensor]]`). A body written for one transition broadcasts
    silently into a wrong number rather than raising: `dist[:, 0] - goal[0]`
    is legal on both shapes and means something different on each, so calling
    it `n` times on single rows PROVES NOTHING ABOUT THE SHAPE IT WILL RUN AT.
    One batched call is the only probe that does.

    NOT A TRACE, unlike the jax path, and that is deliberate rather than a
    shortcut. `torch.jit.script` is what Eureka's decorator applies, but the
    candidate arrives here WITHOUT the decorator (the harness execs a plain
    function, and `_restricted_globals` is not a TorchScript environment), and
    scripting it here would refuse programs that train perfectly well
    un-scripted -- a stricter §2 than the method's own. Eager on a batch tests
    the property that actually breaks; TorchScript compatibility is a property
    of the PROMPT, and `verify.forbidden_symbols` plus the static checks are
    where a prompt-contract violation belongs.

    A SCALAR RETURN IS ADMITTED, not refused, and recorded by §3 instead
    (`CompiledReward.torch_batch` broadcasts it). A candidate whose penalty
    term is genuinely global -- a fleet-wide `.mean()` of an action norm -- is
    a real reward and not a defect, and refusing it here would fail a
    population §3 can run. What is refused is a value of any OTHER length,
    which is the mis-broadcast this probe exists to catch.

    The deadline is read after the call, for `_probe_traced`'s reason: one
    Python call cannot be interrupted cooperatively.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the machine
        # A HARNESS fault, named as one, and NOT filed as a candidate failure:
        # `generate.reward_language: torch` with no torch installed is a broken
        # install, and recording it as `failure_kind="invalid"` would blame a
        # program nothing ran. Same handling as `_jax_namespace`'s ImportError.
        raise ImportError(
            "generate.reward_language=torch needs torch (uv sync --extra fasttd3); "
            f"import failed: {exc}") from exc

    budget_s = _timeout_s(ctx)
    t0 = time.time()
    trs = sample_transitions(ctx, n)
    if not trs:
        return [], ""
    try:
        states = torch.as_tensor(
            np.stack([np.asarray(t.state, dtype=np.float32) for t in trs]))
        has_a = all(t.action is not None for t in trs)
        has_s2 = all(t.next_state is not None for t in trs)
        actions = (torch.as_tensor(
            np.stack([np.asarray(t.action, dtype=np.float32) for t in trs]))
            if has_a else None)
        nexts = (torch.as_tensor(
            np.stack([np.asarray(t.next_state, dtype=np.float32) for t in trs]))
            if has_s2 else None)
    except Exception as exc:  # noqa: BLE001 -- a sampler fault, named as one
        return [], (f"could not stack {len(trs)} sampled transitions into a "
                    f"(batch, *) torch tensor ({type(exc).__name__}: {exc})")

    # THE TRAINER'S BINDER AND THE CANDIDATE'S WEIGHTS, for `_probe_traced`'s
    # reason: §2 must call a candidate the way the tier that TRAINS it does, or
    # it proves a different program. `torch_batch()` is `CompiledReward`'s one
    # batched torch path.
    from .training import CompiledReward

    reward = CompiledReward(fn, getattr(fn, "__name__", "compute_reward"),
                            getattr(candidate, "weights", None))
    try:
        total, comps = reward.torch_batch()(states, actions, nexts)
    except Exception as exc:  # noqa: BLE001 -- the batched call failing IS the verdict
        return [], ("call on a (batch, *) torch tensor failed: "
                    + _trace_error(exc))
    elapsed = time.time() - t0
    if elapsed > budget_s:
        return [], (f"timed out after {budget_s:g}s (one batched call over "
                    f"{n} transitions)")

    # Split back into rows so every check downstream reads this exactly as it
    # reads the numpy probe -- `_probe_traced`'s rule, and the reason the four
    # probing checks need no torch branch of their own.
    host_total = total.detach().cpu().numpy()
    host_comps = {k: v.detach().cpu().numpy() for k, v in comps.items()}
    rows: List[Any] = []
    for i in range(len(trs)):
        rows.append((float(host_total[i] if host_total.ndim else host_total),
                     {k: float(v[i] if v.ndim else v) for k, v in host_comps.items()}))
    return rows, ""


def _probe_traced(ctx: Any, fn: Callable[..., Any], n: int,
                  candidate: Optional[Candidate] = None) -> Tuple[List[Any], str]:
    """`fn` under `jax.jit(jax.vmap(...))` on a batch of `n` sampled transitions.

    THE CONTRACT BEING TESTED IS THE TRACE, not the arithmetic. A `jnp` body
    called eagerly on numpy rows will happily run `if dist < 0.1:` and
    `float(total)`; the same body under `_JaxVecEnvView` is traced once and
    either of those is a `ConcretizationTypeError` on the first training step
    -- a slot spent on what §2 exists to catch. So the probe traces -- through
    `training.CompiledReward.traced()`, the very function `_JaxVecEnvView`
    builds its step from, with the trainer's argument binding -- and a program
    that fails to trace is recorded exactly as one that failed to compile: the
    check returns the reason and the failure policy files it
    `failure_kind="invalid"`. A value that is not a scalar is refused there too
    (`CompiledReward.row`), by name.

    BATCHED, NOT LOOPED, and the returned list is the batch split back into
    rows so the checks downstream read it as they read the numpy probe. Rows
    are stacked on the host and handed over once; a sampler that yields ragged
    shapes or a `None` action is reported as a probe failure, not a candidate
    failure, in the message.

    The deadline is honoured after the fact: a trace is one Python call and
    cannot be interrupted cooperatively, so the clock is read once it returns
    and an overrun is reported as the numpy probe reports one. DIAGNOSTIC
    CONSEQUENCE: a verify stage that appears stuck under `reward_language:
    jax` is a trace that has not returned (a candidate whose unrolled body is
    enormous, or a first-call XLA compile on a cold device), never the sampler
    -- `sample_transitions` has already returned by the time the trace starts.
    """
    import jax  # local: this branch runs only under reward_language=jax
    import jax.numpy as jnp

    budget_s = _timeout_s(ctx)
    # DRAWN BEFORE THE CLOCK STARTS, so the env's own sampling -- possibly an
    # XLA compile of minutes on the first call -- is not charged to the
    # candidate's budget and reported as the candidate timing out. The
    # budget times the TRACE.
    trs = probe_transitions(ctx, n)
    t0 = time.time()
    if not trs:
        return [], ""
    try:
        states = jnp.asarray(np.stack([np.asarray(t.state, dtype=float) for t in trs]))
        has_a = all(t.action is not None for t in trs)
        has_s2 = all(t.next_state is not None for t in trs)
        actions = (jnp.asarray(np.stack([np.asarray(t.action, dtype=float) for t in trs]))
                   if has_a else None)
        nexts = (jnp.asarray(np.stack([np.asarray(t.next_state, dtype=float) for t in trs]))
                 if has_s2 else None)
    except Exception as exc:  # noqa: BLE001 -- a sampler fault, named as one
        return [], (f"could not stack {len(trs)} sampled transitions into a batch "
                    f"for jax.vmap ({type(exc).__name__}: {exc})")

    # THE TRAINER'S BINDER, NOT `call_reward`'s. The two bind a method-style
    # candidate differently (`_make_binder` passes `self` as None;
    # `call_reward` a `_SelfProxy`), and a jax tier must call a candidate the
    # way the tier that TRAINS it will, or §2 proves a different program from
    # the one §3 runs. Under `jax` a method-style candidate is therefore bound
    # the trainer's way at §2 too, so a program the numpy verifier would admit
    # via `_SelfProxy` and then break in training is refused up front -- an
    # intended difference between the two languages' §2.
    # `CompiledReward.row()` is that one truth; imported here rather than at
    # module scope because `training` is the heavier module and this branch
    # runs only under `jax`.
    from .training import CompiledReward

    # WITH the candidate's weights when the check handed the candidate over:
    # under `component_dict_plus_weights` §3 trains the WEIGHTED total, so
    # `bounded_magnitude` and `finite_values` should read that total and not
    # the body's own sum. The numpy probe reads the body's own sum.
    reward = CompiledReward(fn, getattr(fn, "__name__", "compute_reward"),
                            getattr(candidate, "weights", None))
    try:
        out = reward.traced(with_action=has_a, with_next=has_s2)(states, actions, nexts)
        out = jax.block_until_ready(out)
    except Exception as exc:  # noqa: BLE001 -- the trace failing IS the verdict
        return [], "trace under jax.jit(jax.vmap(...)) failed: " + _trace_error(exc)
    elapsed = time.time() - t0
    if elapsed > budget_s:
        return [], f"timed out after {budget_s:g}s (one traced call over {n} transitions)"

    host = jax.tree_util.tree_map(np.asarray, out)
    rows: List[Any] = []
    for i in range(n):
        rows.append(jax.tree_util.tree_map(
            lambda x, i=i: x[i] if np.ndim(x) >= 1 and x.shape[0] == n else x, host))
    return rows, ""


def _trace_error(exc: BaseException) -> str:
    """One line naming a trace failure: the exception's first non-blank line and
    the candidate's own frame. jax's messages run to a screen of prose with
    blank lines in it, so the numpy probe's "last four traceback lines" comes
    out empty here; the first line is the one that says `ConcretizationTypeError`
    or `TracerBoolConversionError`, which is what the repair prompt needs."""
    head = next((ln.strip() for ln in str(exc).splitlines() if ln.strip()), "")
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if str(f.filename).startswith("<candidate")]
    loc = f" (at {frames[-1].filename} line {frames[-1].lineno})" if frames else ""
    return f"{type(exc).__name__}: {head[:300]}{loc}"


@register("dynamic_check", "execution_smoke")
def dyn_execution_smoke(ctx: Any, candidate: Candidate, fn: Callable[..., Any]) -> str:
    """The reward runs at all, over `verify.smoke_test_steps` transitions.

    † CARD's smoke is 1 reset + 1 random step in the train *and* eval env plus
    return-type asserts; GT's paper describes a call on random states and
    actions. Both collapse to "call it and see", parameterised by
    `verify.smoke_test_steps` -- the env pair is the env adapter's business.
    """
    n = max(int(ctx.cfg.get("verify.smoke_test_steps") or 1), 1)
    outs, err = _probe(ctx, fn, n, candidate)
    if err:
        return f"execution_smoke: {err}"
    if not outs:
        return "execution_smoke: no transitions were produced to test against"
    return ""


@register("dynamic_check", "output_shape")
def dyn_output_shape(ctx: Any, candidate: Candidate, fn: Callable[..., Any]) -> str:
    """The return matches the shape `generate.output.format` asked for.

    LIMEN filters shape-invalid candidates alongside constant-reward ones (§2).
    Reading `generate.output.format` here is a config *value* lookup, not a
    method branch: the check cannot tell a breached contract from a
    legitimately scalar reward without knowing what the producer was told.
    """
    n = max(int(ctx.cfg.get("verify.smoke_test_steps") or 1), 1)
    outs, err = _probe(ctx, fn, n, candidate)
    if err:
        return f"output_shape: {err}"
    fmt = ctx.cfg.get("generate.output.format", "")
    for out in outs:
        try:
            total, comps = split_reward(out)
            _as_float(total)
        except RewardShapeError as exc:
            return f"output_shape: {exc}"
        if fmt in _COMPONENT_FORMATS and not comps:
            return (f"output_shape: generate.output.format={fmt!r} requires a named-component "
                    f"dict alongside the total; got {type(out).__name__}")
        for name, value in comps.items():
            try:
                _as_float(value)
            except RewardShapeError as exc:
                return f"output_shape: component {name!r}: {exc}"
    return ""


@register("dynamic_check", "output_dtype")
def dyn_output_dtype(ctx: Any, candidate: Candidate, fn: Callable[..., Any]) -> str:
    """The total is a real number -- not a bool, complex, string or None.

    Separate from `output_shape` on purpose: a `True` reward has the right
    shape and the wrong type, and the two failures want different fixes in the
    next prompt.
    """
    n = max(int(ctx.cfg.get("verify.smoke_test_steps") or 1), 1)
    outs, err = _probe(ctx, fn, n, candidate)
    if err:
        return f"output_dtype: {err}"
    for out in outs:
        try:
            total, comps = split_reward(out)
        except RewardShapeError as exc:
            return f"output_dtype: {exc}"
        for label, value in [("total", total)] + sorted(comps.items()):
            if isinstance(value, bool):
                return f"output_dtype: {label} is a bool; a reward must be a number"
            if isinstance(value, complex):
                return f"output_dtype: {label} is complex"
            if value is None:
                return f"output_dtype: {label} is None"
            try:
                _as_float(value)
            except RewardShapeError as exc:
                return f"output_dtype: {label}: {exc}"
    return ""


@register("dynamic_check", "finite_values")
def dyn_finite_values(ctx: Any, candidate: Candidate, fn: Callable[..., Any]) -> str:
    """No NaN and no infinity, in the total or any component.

    In the default `verify.dynamic_checks`. A NaN reward does not crash the
    learner, it silently destroys it -- one of the few failures that is cheaper
    to catch here than anywhere downstream.
    """
    n = max(int(ctx.cfg.get("verify.smoke_test_steps") or 1), 1)
    outs, err = _probe(ctx, fn, n, candidate)
    if err:
        return f"finite_values: {err}"
    for out in outs:
        try:
            total, comps = split_reward(out)
            for label, value in [("total", total)] + sorted(comps.items()):
                v = _as_float(value)
                if not math.isfinite(v):
                    return f"finite_values: {label} is {v}"
        except RewardShapeError as exc:
            return f"finite_values: {exc}"
    return ""


@register("dynamic_check", "non_constant_reward")
def dyn_non_constant_reward(ctx: Any, candidate: Candidate, fn: Callable[..., Any]) -> str:
    """The reward varies across transitions (§2; ours -- LIMEN's paper attributes
    constant-reward culling to its cascade, main.tex:300, and its release has no
    such check).

    A constant reward is perfectly valid Python and perfectly useless: every
    policy is optimal under it, so the training run is guaranteed waste. Uses
    at least 8 transitions regardless of `verify.smoke_test_steps`, since the
    check is meaningless on one sample.
    """
    n = max(int(ctx.cfg.get("verify.smoke_test_steps") or 1), 8)
    outs, err = _probe(ctx, fn, n, candidate)
    if err:
        return f"non_constant_reward: {err}"
    try:
        totals = [_as_float(split_reward(o)[0]) for o in outs]
    except RewardShapeError as exc:
        return f"non_constant_reward: {exc}"
    if len(totals) < 2:
        return ""  # nothing to compare; not this check's failure to report
    if max(totals) - min(totals) <= 0.0:
        return (f"non_constant_reward: reward is constant at {totals[0]!r} over "
                f"{len(totals)} sampled transitions; every policy is optimal under it")
    return ""


@register("dynamic_check", "bounded_magnitude")
def dyn_bounded_magnitude(ctx: Any, candidate: Candidate, fn: Callable[..., Any]) -> str:
    """No term exceeds :data:`MAX_REWARD_MAGNITUDE`.

    ‡ No paper publishes a bound, so the schema exposes no key and the constant
    lives in this module. A runaway term is the usual cause of a policy
    collapsing onto one degenerate component, and it is detectable before we
    pay for the training rather than after.
    """
    n = max(int(ctx.cfg.get("verify.smoke_test_steps") or 1), 4)
    outs, err = _probe(ctx, fn, n, candidate)
    if err:
        return f"bounded_magnitude: {err}"
    for out in outs:
        try:
            total, comps = split_reward(out)
            for label, value in [("total", total)] + sorted(comps.items()):
                v = abs(_as_float(value))
                if v > MAX_REWARD_MAGNITUDE:
                    return (f"bounded_magnitude: |{label}| = {v:.3g} exceeds "
                            f"{MAX_REWARD_MAGNITUDE:.3g}")
        except RewardShapeError as exc:
            return f"bounded_magnitude: {exc}"
    return ""


@register("dynamic_check", "unit_tests")
def dyn_unit_tests(ctx: Any, candidate: Candidate, fn: Callable[..., Any]) -> str:
    """Run env-supplied reward unit tests, if the adapter offers any.

    ‡ `unit_tests` is in the dynamic-check vocabulary but no method
    supplies tests, and tests are necessarily per-environment. So this defers
    entirely to the env adapter: `env.reward_unit_tests` may be a callable
    taking the compiled reward and returning a failure string (``""`` = pass),
    or an iterable of such callables. An adapter with no tests passes
    vacuously -- silently, because a missing suite is not a candidate's fault.
    """
    tests = getattr(getattr(ctx, "env", None), "reward_unit_tests", None)
    if tests is None:
        return ""
    suite = [tests] if callable(tests) else list(tests)
    for i, test in enumerate(suite):
        try:
            verdict = test(fn)
        except Exception:
            tb = traceback.format_exc(limit=4).strip().splitlines()
            return f"unit_tests: test {i} raised: {tb[-1] if tb else 'unknown'}"
        if verdict is False:
            return f"unit_tests: test {i} failed"
        if isinstance(verdict, str) and verdict:
            return f"unit_tests: test {i}: {verdict}"
    return ""


# ==========================================================================
# Failure routing  --  failure_fn(ctx, state, candidate, reason, attempt)
#                      -> Candidate | None   (None = "regenerate me")
# ==========================================================================
#
# This is the axis on which the methods genuinely differ (§2). Each function
# below is one published policy; none of them knows which method it belongs to.
#
# `candidate.meta` is used as the channel back to `run_validity`:
#   meta["verify_disposition"] = "drop"  -> recorded on disk, removed from the pool
#   meta["repair_trace"]                 -> attached to the regeneration prompt
# Absence of `repair_trace` is what makes `resample_blind` blind.

_DISPOSITION = "verify_disposition"
_REPAIR_TRACE = "repair_trace"


def _record(candidate: Candidate, **fields: Any) -> None:
    """Append a stage-2 audit row. Mutates in place -- the caller owns the object."""
    candidate.verify_records.append(dict(fields))


@register("verify_failure", "discard")
def fail_discard(ctx: Any, state: Any, candidate: Candidate,
                 reason: str, attempt: int) -> Optional[Candidate]:
    """Drop the candidate entirely; no slot, no regeneration.

    The degenerate policy: unlike `record_neg_inf` the failure does not occupy
    a slot in §5's comparison, it simply stops existing for the rest of the
    iteration. `run_validity` still writes the program and the reason to the
    run directory first -- the artifact keeps failures, and
    "discard" is about the *pool*, not about the record.
    """
    out = candidate.failed(f"discard: {reason}", kind="invalid")
    out.meta[_DISPOSITION] = "drop"
    _record(out, phase="on_failure", policy="discard", reason=reason, attempt=attempt)
    return out


@register("verify_failure", "record_neg_inf")
def fail_record_neg_inf(ctx: Any, state: Any, candidate: Candidate,
                        reason: str, attempt: int) -> Optional[Candidate]:
    """Eureka. The failure keeps its slot and scores `select.failure_value`.

    §2: Eureka's `verify.enabled` is effectively false -- the error surfaces
    only when the RL launch dies, and the slot is forfeited at the failure
    sentinel (-10000). No regeneration: the i.i.d. pool is the redundancy, which
    is Eureka's whole argument for sampling 16 at once. The sentinel itself is
    §5's `select.failure_value`; all that is decided here is that the candidate
    stays in the population as an untrainable record.
    """
    out = candidate.failed(reason, kind="invalid")
    out.meta["fitness_sentinel"] = True
    _record(out, phase="on_failure", policy="record_neg_inf", reason=reason, attempt=attempt)
    return out


@register("verify_failure", "resample_with_trace")
def fail_resample_with_trace(ctx: Any, state: Any, candidate: Candidate,
                             reason: str, attempt: int) -> Optional[Candidate]:
    """GT. Regenerate with the error trace attached, until it passes.

    §4 of the paper: uncapped ("until they pass"), so the natural config is
    `verify.max_repair_attempts: 0`. The trace rides on `meta["repair_trace"]`
    and `_regenerate` appends the failed program and that trace to the
    regeneration prompt -- the contrast with `resample_blind`, which shows the
    model neither, is exactly this one field.
    """
    candidate.meta[_REPAIR_TRACE] = reason
    _record(candidate, phase="on_failure", policy="resample_with_trace",
            reason=reason, attempt=attempt, trace_sent=True)
    return None


@register("verify_failure", "resample_blind")
def fail_resample_blind(ctx: Any, state: Any, candidate: Candidate,
                        reason: str, attempt: int) -> Optional[Candidate]:
    """CARD. The identical prompt is re-sent; the model is never told why.

    † The paper (§4.1) says "re-initiate a query"; the released code sends the
    same messages with no error information, and the failed draft leaves **no
    record in the dialogue** -- which is why CARD's `full_dialogue` history is
    "validated turns only" (§1). A `regenerate_with_trace` reading of CARD is
    contradicted by the release; we implement the code.

    So: no `repair_trace`, and nothing is appended to `state.dialogue`. Capped
    at `verify.max_repair_attempts` (CARD: 10), after which
    `verify.on_exhaustion` applies -- `abort_run` for CARD, whose released code
    exits the process.
    """
    candidate.meta.pop(_REPAIR_TRACE, None)
    _record(candidate, phase="on_failure", policy="resample_blind",
            reason=reason, attempt=attempt, trace_sent=False, dialogue_record=False)
    return None


def _teach(state: Any, candidate: Candidate, reason: str) -> None:
    """File a failed program and its error as a negative example for the NEXT
    prompt (`state.failure_memory`, rendered by `generation._failure_traces`
    under `generate.context.include_failure_traces`).

    ONE writer for both the on-failure and the on-exhaustion teach policies, so
    the key the prompt reads (`error`) is the key that gets written; a
    mismatch would render "(unrecorded)" under every failure the prompt is
    meant to learn from.
    `failure_memory` must be in `loop.carry` or it is cleared at the iteration
    boundary and teaching silently becomes `discard`.
    """
    if state is None:
        return
    state.failure_memory.append({
        "cand_id": candidate.cand_id,
        "iteration": str(candidate.iteration),
        "code": candidate.reward_code or "",
        "error": reason,
        "failure_kind": "invalid",
    })


@register("verify_failure", "teach_next_iteration")
def fail_teach_next_iteration(ctx: Any, state: Any, candidate: Candidate,
                              reason: str, attempt: int) -> Optional[Candidate]:
    """Discard now; quote the code and error into the NEXT prompt. No repair.

    The failure does not occupy a slot: it becomes a negative example via
    `_teach`. Because this policy SETTLES the slot at the first failure,
    `verify.max_repair_attempts` is unreachable under it and `_check_coherence`
    refuses any value but 0. LIMEN's release makes one same-iteration crash
    retry (`controller.py:423-448`, `retry_on_crash`), so LIMEN's point is
    `on_failure: resample_with_trace` + `max_repair_attempts: 1` +
    `on_exhaustion: teach_next_iteration`: repair once in the same iteration,
    teach if that fails too.
    """
    _teach(state, candidate, reason)
    out = candidate.failed(reason, kind="invalid")
    out.meta[_DISPOSITION] = "drop"
    out.meta["taught"] = True
    _record(out, phase="on_failure", policy="teach_next_iteration",
            reason=reason, attempt=attempt, remembered=True)
    return out


def _salvage_source(code: str, max_drops: int = 25) -> Tuple[str, List[int]]:
    """Drop unparseable lines until the rest parses. Repo-authored (no DrEureka analogue)."""
    lines = (code or "").splitlines()
    dropped: List[int] = []
    for _ in range(max_drops):
        try:
            ast.parse("\n".join(lines))
            return "\n".join(lines), dropped
        except SyntaxError as exc:
            i = (exc.lineno or 1) - 1
            if not (0 <= i < len(lines)):
                break
            dropped.append(i + 1)
            lines[i] = ""
    return "\n".join(lines), dropped


def _clip_dr_config(dr: Dict[str, Any], bounds: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """Clip generated DR ranges into bounds. Returns (clipped, changed keys)."""
    if not isinstance(dr, dict):
        return dr, []
    out, changed = dict(dr), []
    for key, value in dr.items():
        lohi = (bounds or {}).get(key)
        if not (isinstance(value, (list, tuple)) and len(value) == 2 and lohi):
            continue
        lo, hi = float(lohi[0]), float(lohi[1])
        new = [min(max(float(value[0]), lo), hi), min(max(float(value[1]), lo), hi)]
        if new[0] > new[1]:
            new = [new[1], new[0]]
        if new != [float(value[0]), float(value[1])]:
            out[key] = new
            changed.append(key)
    return out, changed


def _env_dr_bounds(env: Any) -> Optional[Dict[str, Tuple[float, float]]]:
    """The env's declared DR bounds as ``{name: (lo, hi)}``, or None.

    `dr_parameters` first -- the adapter interface (`envs/base.py`), populated
    per task by `spec.dr_of` and hard-coded on MetaWorld -- then `dr_bounds`
    for an adapter this repo does not own. Accepts the same shapes
    `phases._dr_ranges` does (a 2-sequence or a ``{min,max}``/``{lo,hi}``
    dict per axis) and skips an axis it cannot read rather than guessing.
    """
    for attr in ("dr_parameters", "dr_bounds"):
        raw = getattr(env, attr, None)
        if callable(raw):
            try:
                raw = raw()
            except Exception:  # noqa: BLE001 -- an env hook we do not own
                raw = None
        if not isinstance(raw, dict) or not raw:
            continue
        out: Dict[str, Tuple[float, float]] = {}
        for name, spec in raw.items():
            lo = hi = None
            if isinstance(spec, dict):
                lo, hi = spec.get("min", spec.get("lo")), spec.get("max", spec.get("hi"))
            elif isinstance(spec, (list, tuple)) and len(spec) >= 2:
                lo, hi = spec[0], spec[1]
            if lo is None or hi is None:
                continue
            try:
                out[str(name)] = (float(lo), float(hi))
            except (TypeError, ValueError):
                continue
        if out:
            return out
    return None


@register("verify_failure", "degrade")
def fail_degrade(ctx: Any, state: Any, candidate: Candidate,
                 reason: str, attempt: int) -> Optional[Candidate]:
    """Repo-authored `degrade`: salvage what parses, clip into bounds, train anyway.

    Three moves (no released analogue -- dr_eureka.py:109-128 pastes the LLM's
    block verbatim and scores a broken one DUMMY_FAILURE):

    1. drop unparseable lines until the remainder parses;
    2. clip any generated DR range into the bounds the env adapter advertises
       (‡ the paper takes those bounds from the RAPP sweep; we read
       `env.dr_parameters` -- the attribute every adapter here declares
       (`envs/base.py`) and RAPP's `phases._dr_ranges` reads -- falling back to
       `dr_bounds` for an external adapter, and leave the config alone when
       neither exists, rather than inventing a schema key; probing
       `dr_bounds` alone would never clip, since no adapter here defines it,
       and `set_dr` would clip silently at training time instead);
    3. admit the candidate regardless.

    Step 3 is the whole point and is why `degrade` never regenerates: when the
    failure is behavioural rather than syntactic there is nothing to salvage,
    and "train anyway" is still the policy. The candidate stays `valid=True`
    with the degradation recorded, so nothing downstream mistakes it for clean.
    """
    salvaged, dropped = _salvage_source(candidate.reward_code or "")
    bounds = _env_dr_bounds(getattr(ctx, "env", None))
    dr, clipped = _clip_dr_config(candidate.dr_config or {}, bounds)

    out = replace(candidate, reward_code=salvaged,
                  dr_config=dr if candidate.dr_config is not None else None)
    parses = _parse(out)[0] is not None
    out.meta["degraded"] = True
    out.meta["degrade_reason"] = reason
    _record(out, phase="on_failure", policy="degrade", reason=reason, attempt=attempt,
            dropped_lines=dropped, clipped_dr_keys=clipped, parses_after_salvage=parses)
    if not parses:
        # Nothing survived. Admitting an unparseable program would hand stage 3
        # something it cannot import, which is a lie about what degrade did.
        failed = out.failed(f"degrade: nothing salvageable ({reason})", kind="invalid")
        failed.meta.update(out.meta)
        return failed
    return out


# ==========================================================================
# Dedup  --  dedup_fn(ctx, candidates) -> list[Candidate]
# ==========================================================================
#
# Dedup prevents paying twice for the same candidate; no published method has one.
# A duplicate is not a validity failure -- it compiles, it is a perfectly good
# reward. It is a *budget* rejection, so it joins the screened population
# (`failure_kind="screened"`), not the invalid one. Duplicates are MARKED, never
# dropped: an artifact that quietly loses candidates cannot be audited, and the
# execute rate would be computed over a shrunken pool.


def _mark_duplicate(dup: Candidate, original: Candidate, detail: str) -> None:
    dup.screened_out = True
    dup.failure = f"duplicate of {original.cand_id} ({detail})"
    dup.failure_kind = "screened"
    dup.meta["duplicate_of"] = original.cand_id
    _record(dup, phase="dedup", duplicate_of=original.cand_id, detail=detail)


def _dedup_candidates(candidates: List[Candidate]) -> List[Candidate]:
    """The subset a dedup method should compare: still-live programs only."""
    return [c for c in candidates if c.trainable and (c.reward_code or "").strip()]


@register("dedup", "none")
def dedup_none(ctx: Any, candidates: List[Candidate]) -> List[Candidate]:
    """Identity. The published setting everywhere -- no method deduplicates."""
    return candidates


def _normalised_ast_hash(code: str) -> Optional[str]:
    """Structure-only digest: docstrings and formatting out, structure in.

    Two programs that differ by a comment, a blank line, or a docstring are the
    same candidate and must not be trained twice. Two that differ by a constant
    are NOT -- a weight change is the commonest useful mutation, so constants
    stay in the digest.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, annotate_fields=True,
                                   include_attributes=False).encode()).hexdigest()


@register("dedup", "ast_hash")
def dedup_ast_hash(ctx: Any, candidates: List[Candidate]) -> List[Candidate]:
    """Exact structural duplicates, by normalised AST digest.

    The cheap, exact end of the axis: no threshold, no embeddings, no false
    positives. `verify.dedup.threshold` is ignored -- an AST hash either
    collides or does not.
    """
    seen: Dict[str, Candidate] = {}
    for c in _dedup_candidates(candidates):
        digest = _normalised_ast_hash(c.reward_code)
        if digest is None:
            continue  # unparseable: the validity checks own that verdict
        first = seen.get(digest)
        if first is None:
            seen[digest] = c
        else:
            _mark_duplicate(c, first, "identical normalised AST")
    return candidates


def _token_vector(code: str) -> Dict[str, float]:
    """Identifier/operator frequency vector. The offline stand-in for an embedding."""
    try:
        tree: Optional[ast.Module] = ast.parse(code or "")
    except SyntaxError:
        tree = None
    if tree is None:
        toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", code or "")
    else:
        toks = []
        for node in ast.walk(tree):
            toks.append(type(node).__name__)
            if isinstance(node, ast.Name):
                toks.append(f"n:{node.id}")
            elif isinstance(node, ast.Attribute):
                toks.append(f"a:{node.attr}")
    vec: Dict[str, float] = {}
    for t in toks:
        vec[t] = vec.get(t, 0.0) + 1.0
    return vec


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


@register("dedup", "embedding")
def dedup_embedding(ctx: Any, candidates: List[Candidate]) -> List[Candidate]:
    """Near-duplicates by embedding cosine >= `verify.dedup.threshold`.

    ‡ Nobody published this, so there is no reference embedding model to be
    faithful to. A hosted embedding API would break the repo's offline and
    determinism guarantees, so the "embedding" here is a deterministic
    bag-of-AST-tokens vector computed locally. That is an honest approximation
    of the *axis* (fuzzy similarity with a tunable threshold) and is documented
    as such rather than pretending to be sentence embeddings: swap
    `_token_vector` for a real encoder and the rest of the function is unchanged.

    `verify.dedup.threshold <= 0` is read as *exact* match, matching what a zero
    distance means for `ast_hash` and `epic_distance`. Taken literally, "cosine
    >= 0" would mark every candidate after the first a duplicate, and 0.0 is the
    schema default.
    """
    thr = float(ctx.cfg.get("verify.dedup.threshold") or 0.0)
    if thr <= 0.0:
        thr = 1.0 - 1e-12
    live = _dedup_candidates(candidates)
    vecs = {c.cand_id: _token_vector(c.reward_code) for c in live}
    kept: List[Candidate] = []
    for c in live:
        match = next((k for k in kept if _cosine(vecs[c.cand_id], vecs[k.cand_id]) >= thr), None)
        if match is None:
            kept.append(c)
        else:
            sim = _cosine(vecs[c.cand_id], vecs[match.cand_id])
            _mark_duplicate(c, match, f"token-vector cosine {sim:.3f} >= {thr:.3f}")
    return candidates


@register("dedup", "epic_distance")
def dedup_epic_distance(ctx: Any, candidates: List[Candidate]) -> List[Candidate]:
    """Near-duplicates by EPIC pseudometric <= `verify.dedup.threshold`.

    The behaviourally honest member of the family: two programs can share no
    tokens and induce the same ordering over transitions, in which case
    training both is pure waste. Uses the same canonicalisation as
    `screen:epic`, imported lazily so `screens.py` can depend on this module at
    import time without a cycle.

    ‡ Unpublished, like the rest of the dedup axis.
    """
    from .screens import epic_canonical_profile, epic_distance  # local: avoids a cycle

    thr = float(ctx.cfg.get("verify.dedup.threshold") or 0.0)
    transitions = sample_transitions(ctx, 64)
    profiles: Dict[str, Any] = {}
    for c in _dedup_candidates(candidates):
        fn, err = compile_reward(ctx, c)
        if fn is None:
            continue  # validity owns that verdict
        prof = epic_canonical_profile(ctx, fn, transitions)
        if prof is not None:
            profiles[c.cand_id] = prof

    kept: List[Candidate] = []
    by_id = {c.cand_id: c for c in candidates}
    for cid, prof in profiles.items():
        match = next((k for k in kept if epic_distance(profiles[k], prof) <= thr), None)
        if match is None:
            kept.append(cid)
        else:
            d = epic_distance(profiles[match], prof)
            _mark_duplicate(by_id[cid], by_id[match], f"EPIC distance {d:.4f} <= {thr:.4f}")
    return candidates


# ==========================================================================
# Regeneration (used only by the resample_* policies)
# ==========================================================================


def _accepts_one_positional(fn: Callable[..., Any]) -> bool:
    # `inspect.signature` already drops the receiver of a bound method, so only
    # an explicitly-named `self`/`cls` (an unbound function) needs stripping.
    names, star = _positional_params(fn)
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return star or len(names) >= 1


def _llm_text(ctx: Any, messages: List[Dict[str, str]]) -> Tuple[Optional[str], str, int]:
    """Send `messages` to the generator client. Returns (text, error, tokens_out).

    The LLM client family is owned elsewhere and its calling convention is not
    part of the frozen contract, so we probe the usual shapes rather than
    guessing one. A client that matches none of them is not an error in stage 2
    -- it means this config cannot repair, which the caller records.
    """
    client = getattr(ctx, "generator", None)
    if client is None:
        return None, "no generator client attached to the context", 0
    call = None
    for attr in ("complete", "chat", "generate", "__call__"):
        fn = getattr(client, attr, None)
        if callable(fn) and _accepts_one_positional(fn):
            call = fn
            break
    if call is None:
        return None, f"generator client {type(client).__name__} exposes no usable call", 0
    try:
        out = call(messages)
    except Exception as exc:
        return None, f"generator raised {type(exc).__name__}: {exc}", 0
    text = _as_text(out)
    if not text:
        return None, "generator returned no text", 0
    return text, "", len(text) // CHARS_PER_TOKEN


def _as_text(out: Any) -> str:
    if isinstance(out, str):
        return out
    for attr in ("text", "content", "completion", "message"):
        v = getattr(out, attr, None)
        if isinstance(v, str):
            return v
    if isinstance(out, dict):
        for k in ("text", "content", "completion"):
            if isinstance(out.get(k), str):
                return out[k]
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], str):
        return out[0]
    return ""


def _extract_code(ctx: Any, text: str) -> str:
    """Pull a code block out with `generate.parse.patterns`; "" when there is none.

    THE SAME RULE AS STAGE 1, by import from `bird/parsing.py` -- not a copy.
    A private copy of stage 1's extractor drifts, and every drift is a harness
    defect charged to the model: no language-tag strip (a ```py fence repaired
    into a program whose first line is `py`), `strip("\n")` for `strip()` (an
    indented body into an IndentationError), or the WHOLE reply returned when
    nothing matches (a prose reply into a "program" failing with a
    SyntaxError). The pattern list is config; the code around it is shared
    too, through a module neither family owns.
    """
    code, _pattern = extract_code(ctx.cfg.get("generate.parse.patterns") or [], text)
    return code or ""


#: `generate.output.design_thought: inline_brace`, read the way stage 1 reads it
#: (generation._THOUGHT_RE; rfagent_algo.py:426). A copy of that one-line
#: pattern rather than an import.
_DESIGN_THOUGHT_RE = re.compile(r"\{(.*?)\}", re.DOTALL)


def _regenerate(ctx: Any, state: Any, candidate: Candidate,
                attempt: int) -> Tuple[Optional[Candidate], str]:
    """Re-sample this slot. The trace is attached iff the policy left one.

    Charging is explicit here because the cost column is a first-class
    output: every repair is a `record_resample`, and a repair also counts as an
    LLM call when `verify.repairs_count_against_budget` -- which is exactly the
    knob that makes CARD-vs-Eureka token comparisons fair.

    EXACTLY ONCE under `true`, and NOT AT ALL under `false`. The client we call
    through `_llm_text` already records its own round-trip (the `bird/llm/base.py`
    contract: no stage calls `record_llm`, with the provider's exact token
    counts), so charging here as well would count every repair twice with the
    key on and once with it off -- the inverse of what the key declares, on the
    path whose cost is CARD's headline result, with every counter reading
    normal (on `gt` at the tester tier: `llm_calls` +2 per repair, tokens
    ~2x). So: snapshot `llm_calls`, call, and estimate only if the client
    recorded nothing -- a bare callable standing in for a client, which is
    what the estimate is for; the same reconciliation as stage 1's
    `generation._call_llm`. Under `false`, latch the counters
    (`Budget.uncounted_llm`) so the client's own record is what gets excluded.
    """
    ctx.budget.record_resample()

    messages = [dict(m) for m in (candidate.prompt_messages or [])]
    if not messages:
        return None, "no prompt to resend (candidate carries no prompt_messages)"
    trace = candidate.meta.get(_REPAIR_TRACE)
    if trace:
        # The program that failed goes back as the assistant turn it was, so
        # the trace's `line N` points into code the model can see. The trace
        # alone is not enough: `prompt_messages` is the REQUEST that produced
        # the candidate (system + user, no reply), and the traceback carries no
        # source (`compile_reward` names the module `<candidate cN>`, which
        # linecache does not know) -- so "the program you produced" would refer
        # to nothing in the conversation and every repair would be a rewrite
        # from scratch guided by an error it could not localise. The release's
        # own repair prompt opens with the failed program (RF-Agent
        # initial_failed_feedback.txt:1). `raw_response` when there is one (the
        # model's actual turn), else the code fenced. A request always ends
        # with a user turn, so alternation holds.
        failed = candidate.raw_response or f"```python\n{candidate.reward_code}\n```"
        messages.append({"role": "assistant", "content": failed})
        messages.append({
            "role": "user",
            "content": ("The program you produced failed verification:\n\n"
                        f"{trace}\n\nRewrite it so it passes. Return the full program."),
        })

    if ctx.cfg.get("verify.repairs_count_against_budget"):
        before = ctx.budget.llm_calls
        text, err, out_tokens = _llm_text(ctx, messages)
        if ctx.budget.llm_calls == before:
            # A client outside the `bird.llm` contract (a bare callable) records
            # nothing; charge the chars/4 estimate so the repair is not free.
            in_tokens = sum(len(m.get("content", "")) for m in messages) // CHARS_PER_TOKEN
            ctx.budget.record_llm(prompt_tokens=in_tokens, completion_tokens=out_tokens)
    else:
        with ctx.budget.uncounted_llm():
            text, err, out_tokens = _llm_text(ctx, messages)
    if text is None:
        return None, err

    code = _extract_code(ctx, text)
    no_code = not code.strip()
    # A reply with no program is NOT an exhaustion. Returning `None` here
    # would make `run_validity` treat it as "cannot resample" and settle at
    # once (`on_exhaustion`, i.e. `abort_run` for CARD). Upstream re-queries
    # until it compiles or the cap is hit, so the slot is built with EMPTY
    # code: `_first_failure` fails it for having no
    # entrypoint, `verify.on_failure` decides, and the label says what
    # happened (`meta["repair_no_code"]`) instead of blaming a SyntaxError.

    # The repaired program is a NEW program, so its design thought is re-read
    # from the repaired response rather than inherited: the release's repair
    # prompt asks for the brace again (initial_failed_feedback.txt:4-5) and
    # parses it afresh (rfagent_algo.py:426). Absent -> "" and recorded, the
    # same outcome `generation._build_candidate` writes.
    nl_spec = candidate.nl_spec
    thought_parsed: Optional[bool] = None
    if ctx.cfg.get("generate.output.design_thought") == "inline_brace":
        # Prose before the first fence only, as `generation._parse_design_thought`
        # does: a dict literal inside the repaired code is not a design idea.
        fence = text.find("```")
        m = _DESIGN_THOUGHT_RE.search(text if fence < 0 else text[:fence])
        nl_spec = m.group(1).strip() if m else ""
        thought_parsed = bool(nl_spec)

    # THE SAME POSTPROCESS `_build_candidate` APPLIES. A repaired program is a
    # NEW program and needs `generate.postprocess.*` run over it exactly as a
    # generated one does; without it, every repair under a non-empty
    # `symbol_mapping` reaches the harness with the prompt's own symbols
    # unrewritten and cannot pass a dynamic check however many attempts the
    # cap allows (`card` on Meta-World: all 10 attempts per candidate failing
    # `AttributeError: 'MetaWorld' object has no attribute 'robot'`, the
    # table's exact keys still in the persisted program). Imported from the
    # module that owns the chain rather than re-applied here, so a second
    # postprocess step cannot land in one path and not the other.
    from .generation import postprocess_code

    code, post_meta = postprocess_code(ctx, code)

    # META IS RECOMPUTED, NOT INHERITED WHOLESALE, and the distinction is not
    # pedantry: inheriting the parent's `symbol_mapping_applied` would make the
    # artifact assert that the mapping had been applied to a program it had
    # never seen. An artifact that lies is worse than the bug it hides, because
    # the bug is at least findable. The keys DERIVED FROM THE PARENT'S OWN
    # PROGRAM TEXT are dropped and re-supplied from this program's postprocess;
    # the ones describing how the parent was SAMPLED (`sampling_mode`,
    # `sample_index`, `history_modes`) are genuinely inherited, because the
    # repair is a re-query of that same sample.
    #
    # `parse_pattern` / `parse_attempts` are dropped and NOT re-supplied:
    # `_extract_code` above does its own extraction and does not report which
    # pattern matched, so the honest record is their absence rather than the
    # parent's values wearing this program's name.
    _TEXT_DERIVED = ("symbol_mapping_applied", "parse_pattern", "parse_attempts")
    meta = {k: v for k, v in candidate.meta.items()
            if k != _REPAIR_TRACE and k not in _TEXT_DERIVED}
    meta.update(post_meta)

    fresh = replace(
        candidate,
        cand_id=ctx.next_id("r"),
        reward_code=code,
        raw_response=text,
        nl_spec=nl_spec,
        valid=True, screened_out=False, failure="", failure_kind="",
        verify_records=list(candidate.verify_records),
        meta=meta,
    )
    fresh.meta["repaired_from"] = candidate.cand_id
    fresh.meta["repair_attempt"] = attempt
    fresh.meta["repair_trace_sent"] = bool(trace)
    if no_code:
        fresh.meta["repair_no_code"] = True
    if thought_parsed is not None:
        fresh.meta["design_thought_parsed"] = thought_parsed
    return fresh, ""


# ==========================================================================
# The validity phase itself
# ==========================================================================
#
# Exported, not registered: `phases.py` owns the `phase` family and registers
# this as `phase:validity`. `bird.verify()` reaches it through that name.


# --------------------------------------------------------------------------
# The human in the validity gate  (`verify.human_confirm_before_execute`)
# --------------------------------------------------------------------------
#
# L2R's `confirmation_safe_executor.py` prints the generated program, warns
# that it is untrusted, and blocks on `input()` until a person types "yes"
# That is a human *inside* the validity
# gate and neither `select.human_override` (which acts on already-trained
# candidates) nor `evaluate.human.mode` (which grades behaviour) reaches it.
#
# The gate fires once per candidate per attempt, immediately before the compile
# step -- `exec`ing the module body already runs the candidate's code, so
# confirming after it would confirm nothing. It is also the cheapest honest
# placement: statics run first, so a person is never shown a program that does
# not parse, and human attention is the scarcest cost column.

#: Marks a candidate the human refused. `run_validity` reads it to settle the
#: candidate directly instead of routing it through `verify.on_failure`: no
#: repair policy can fix "a person said no", and `degrade` must not train past
#: it. The prefix below is the greppable half of the same fact, in the artifact.
_HUMAN_REFUSED = "human_refused_before_execute"
_HUMAN_REFUSAL_PREFIX = "human_confirm_before_execute: refused: "

#: "this hook had nothing to say", distinct from a hook that answered `False`.
_NO_ANSWER = object()


def _invoke_human(fn: Callable[..., Any], ctx: Any, state: Any, payload: Any) -> Any:
    """Call an oracle hook whose arity we do not own.

    `ctx.human` is built by another component family (`phase:human_oracle`) and
    may be the scripted oracle, the null oracle, or an interactive one a config
    asked for, so stage 2 inspects the signature and passes the argument tuple
    that fits rather than catching `TypeError` -- which would swallow a genuine
    `TypeError` raised *inside* the oracle. `selection._invoke` does the same
    thing for stage 5's hooks; the two are not shared because they pass
    different payloads and neither family owns the other's call convention.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return _NO_ANSWER
    params = list(sig.parameters.values())
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return fn(ctx, state, payload)
    positional = [p for p in params
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    required = sum(1 for p in positional if p.default is p.empty)
    for args in ((ctx, state, payload), (state, payload), (payload,), ()):
        if required <= len(args) <= len(positional):
            return fn(*args)
    return _NO_ANSWER


def _pre_execution_report(candidate: Candidate) -> CandidateReport:
    """Wrap a candidate in the report shape the oracle's documented hooks take.

    `phase:human_oracle`'s query surface is defined over `CandidateReport`
    (`veto(report) -> bool`), because every other human query in the repo
    happens after training. Here nothing has been trained yet, so the report
    carries an explicitly untrained `TrainResult`: the scripted oracle then
    observes no rollouts, no success rate and no reference score, and answers
    "no reason to refuse" -- deterministic, offline, and visibly not a claim
    about behaviour it never saw.
    """
    empty = TrainResult(
        cand_id=candidate.cand_id, candidate=candidate, trained=False,
        skip_reason="not trained: asked before stage 2 executed the program")
    return CandidateReport(cand_id=candidate.cand_id, candidate=candidate, result=empty)


def _confirm_before_execute(ctx: Any, state: Any, candidate: Candidate) -> str:
    """Ask the human to approve this program. Returns "" or the refusal reason.

    Off by default, and when off this function touches nothing -- no oracle
    call, no budget row, no audit row.

    Hook order is specific-to-general: `confirm`/`approve` are what an
    interactive oracle registered under `phase:human_oracle` would offer for
    exactly this question, and `veto` is the refusal channel the shipped
    oracles already document. Every answered query charges
    `budget.record_human()` at the call site, as `selection._ask_human` does
    (`preferences.comparator_human` does not: there the oracle charges, and a
    call-site charge would double every comparison). The caveat in
    `phases.py` applies here: a scripted `veto` also charges itself, so those
    queries are counted twice.

    When nothing answers -- `evaluate.human.mode: none`, whose oracle raises on
    every query -- this degrades with a warning rather than stalling the loop,
    matching `select.rule: human`. The alternative, failing every candidate
    closed, would turn a missing oracle into a run in which nothing is ever
    verified, and the warning plus the ``asked=False`` audit row is the visible
    failure `phases.py` asks callers to produce.
    """
    if not ctx.cfg.get("verify.human_confirm_before_execute", False):
        return ""

    oracle = getattr(ctx, "human", None)
    report = _pre_execution_report(candidate)
    # (hook name, payload, does a True answer mean "refuse")
    hooks = (("confirm", candidate, False),
             ("approve", candidate, False),
             ("veto", report, True))
    for meth, payload, inverted in hooks:
        hook = getattr(oracle, meth, None)
        if not callable(hook):
            continue
        try:
            answer = _invoke_human(hook, ctx, state, payload)
        except Exception as exc:  # an oracle that refuses to answer is not an error
            log.warning("human oracle .%s() failed (%s); trying the next hook", meth, exc)
            continue
        if answer is _NO_ANSWER or answer is None:
            continue
        ctx.budget.record_human()
        refused = bool(answer) if inverted else not bool(answer)
        _record(candidate, phase="human_confirm", ok=not refused, asked=True,
                hook=meth, verdict=repr(answer))
        if refused:
            detail = (f"the human declined to run {candidate.cand_id} "
                      f"(oracle.{meth} -> {answer!r})")
            candidate.meta[_HUMAN_REFUSED] = True
            ctx.event("verify_human_refused", cand_id=candidate.cand_id, hook=meth)
            return _HUMAN_REFUSAL_PREFIX + detail
        return ""

    log.warning("verify.human_confirm_before_execute is set but no human oracle "
                "answered (evaluate.human.mode=%r); the program runs unconfirmed",
                ctx.cfg.get("evaluate.human.mode"))
    _record(candidate, phase="human_confirm", ok=True, asked=False,
            detail="no human oracle answered; execution was not confirmed")
    return ""


# --------------------------------------------------------------------------
# Running the checks
# --------------------------------------------------------------------------


def _check_plan(ctx: Any) -> Any:
    """Order the two check lists into a plan (`verify.check_order`).

    The `check_order` family owns what "in order" means: `declared` is the two
    lists exactly as the config wrote them -- a method's list order is part of
    its point in the space -- and `staged` is LIMEN's cheapest-first staging.
    Looked up through the registry rather than imported so the choice stays a
    config *value* and stage 2 keeps no opinion about which method wants which.
    """
    build = _get("check_order", ctx.cfg.get("verify.check_order", "declared"))
    return build(ctx,
                 list(ctx.cfg.get("verify.static_checks") or []),
                 list(ctx.cfg.get("verify.dynamic_checks") or []))


def _compile_gate(ctx: Any, state: Any,
                  candidate: Candidate) -> Tuple[Optional[Callable[..., Any]], str]:
    """Confirm, then compile. The price of admission to executing a program.

    Not a step in the plan, because it is neither optional nor orderable:
    `exec`ing the module body is already running the candidate's code, so the
    human gate and the compile belong to the same moment, and every dynamic
    check needs the callable it produces. Run exactly once per candidate per
    repair attempt, however many dynamic checks follow -- and only through
    :func:`_open_gate`, which is where the precondition lives.
    """
    refusal = _confirm_before_execute(ctx, state, candidate)
    if refusal:
        return None, refusal
    fn, err = compile_reward(ctx, candidate)
    if fn is None:
        _record(candidate, phase="compile", ok=False, detail=err)
        return None, err
    _record(candidate, phase="compile", ok=True)
    return fn, ""


def _open_gate(ctx: Any, state: Any, candidate: Candidate,
               blocked: str) -> Tuple[Optional[Callable[..., Any]], str]:
    """Open the compile gate, or record why it stayed shut. `blocked` closes it.

    A check that has already rejected the program withdraws permission to run
    it, whatever `plan.short_circuit` says. Short-circuiting is a *cost* policy
    -- how much of the artifact a rejected candidate is worth filling in -- and
    it must not be readable as permission to execute code a static check just
    refused. `verify.forbidden_symbols` is the sharp case: it exists to catch a
    program that reads the ground-truth reward it is being scored against, and
    "we rejected it and then ran its module body anyway" is not a rejection.
    `search.order_declared` says the same thing from the planner's side -- "a
    dynamic check is handed a compiled callable, which only exists once the
    statics have not rejected the program".

    So this is the one ordering constraint stage 2 keeps for itself, and it is
    a precondition rather than a step: no `check_order` component can schedule
    around it, because it is about what has already failed, not about order.

    Returns `(None, "")` when shut: no new failure (the verdict is already
    `blocked`), no callable, and one audit row explaining the absence of the
    compile and dynamic rows a reader would otherwise expect.
    """
    if blocked:
        _record(candidate, phase="gate", ok=False, executed=False,
                detail=f"not executed: rejected by an earlier check ({blocked})")
        return None, ""
    return _compile_gate(ctx, state, candidate)


def _first_failure(ctx: Any, state: Any,
                   candidate: Candidate) -> Tuple[str, Optional[Callable[..., Any]]]:
    """Walk the check plan and return the first failure, plus the callable.

    Three knobs meet here and only one of them decides the verdict:

    * `verify.check_order` supplies `plan.steps`, the ``(kind, name)`` sequence
      to walk. Order is never this function's choice.
    * `plan.timeouts[name]` is the deadline for that step, `verify.timeout_s`
      when the plan priced no deadline for it (`verify.stage_timeouts_s`).
    * `plan.short_circuit` decides whether a failing step ends the walk. It
      does NOT decide the verdict: the reason returned is always the *first*
      failure, so a non-short-circuiting plan settles the candidate exactly as
      a short-circuiting one would and differs only in how much of the artifact
      gets filled in. `check_order: declared` -- no short circuit -- therefore
      returns the same verdict and pays for the extra rows; `staged` (LIMEN) stops at
      the first failing stage and never pays for the expensive one.

    What short-circuiting does NOT buy either plan is permission to execute a
    program some check already rejected: see :func:`_open_gate`, which is why
    `declared`'s extra rows are diagnostics rather than a wider blast radius.
    Compilation is not a step; see :func:`_compile_gate`.
    """
    plan = _check_plan(ctx)
    fn: Optional[Callable[..., Any]] = None
    gated = False  # the compile gate has been decided, open or shut
    first_reason = ""

    for kind, name in plan.steps:
        if kind == "dynamic":
            if not gated:
                gated = True
                fn, gate_err = _open_gate(ctx, state, candidate, first_reason)
                if gate_err:
                    first_reason = first_reason or gate_err
                    if candidate.meta.get(_HUMAN_REFUSED) or plan.short_circuit:
                        # A refusal is terminal whatever the plan says: the
                        # person withdrew permission to run this program, so
                        # no later step may run it either.
                        return first_reason, None
            if fn is None:
                continue  # nothing to call; the gate row is already the record

        check = _get("static_check" if kind == "static" else "dynamic_check", name)
        with _deadline(plan.timeouts.get(name)):
            reason = check(ctx, candidate) if kind == "static" else check(ctx, candidate, fn)
        if reason:
            _record(candidate, phase=kind, check=name, ok=False, detail=reason)
            first_reason = first_reason or reason
            if plan.short_circuit:
                return first_reason, fn
        else:
            _record(candidate, phase=kind, check=name, ok=True)

    if not gated:
        # A plan with no dynamic steps still has to establish that the program
        # imports: a candidate that cannot be exec'd is not trainable, and
        # leaving stage 3 to discover that would forfeit a training slot. Same
        # precondition as above -- if a static already rejected it, the gate
        # stays shut and `fn` stays None.
        fn, gate_err = _open_gate(ctx, state, candidate, first_reason)
        if gate_err:
            first_reason = first_reason or gate_err
    return first_reason, fn


def _get(kind: str, name: str) -> Callable[..., Any]:
    from ..registry import get  # local import: registry imports this module

    return get(kind, name)


def _exhausted(ctx: Any, state: Any, candidate: Candidate, reason: str,
               attempt: int) -> Candidate:
    """Apply `verify.on_exhaustion` once the repair cap is spent.

    `teach_next_iteration` is the exhaustion-side twin of the `on_failure`
    policy of that name: the program that still fails after its repairs is
    dropped and filed as a negative example for the next prompt (LIMEN,
    `controller.py:428-433` -> `prompts.py:665-670`; `main.tex:298`).
    """
    policy = ctx.cfg.get("verify.on_exhaustion", "record_and_continue")
    detail = f"repair attempts exhausted after {attempt} (last failure: {reason})"
    if policy == "abort_run":
        # † CARD's released code exit()s here. The exception leaves run().
        raise VerificationAborted(f"{candidate.cand_id}: {detail}")
    out = candidate.failed(detail, kind="invalid")
    if policy == "teach_next_iteration":
        _teach(state, candidate, reason)
        out.meta[_DISPOSITION] = "drop"
        out.meta["taught"] = True
    _record(out, phase="exhaustion", policy=policy, reason=reason, attempts=attempt)
    return out


def run_validity(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """Stage 2's validity half: check, route failures, repair, dedup.

    Registered as `phase:validity` by `phases.py`; `bird.verify()` calls it when
    `verify.enabled`, then applies `verify.quality_screen` separately. Keeping
    the two apart in code is what keeps the two populations apart in the
    artifact: everything rejected here carries
    ``failure_kind="invalid"``, everything the screen rejects carries
    ``"screened"``.

    The repair loop is `verify.on_failure` x `verify.max_repair_attempts` x
    `verify.on_exhaustion`. A policy that returns a Candidate settles the slot;
    a policy that returns None asks for a re-sample, and the cap (0 = uncapped,
    GT) plus :data:`UNCAPPED_SAFETY_LIMIT` decides when we stop trying.

    What the checks are and what order they run in belongs to
    :func:`_first_failure` (`verify.check_order`, `verify.stage_timeouts_s`);
    the one verdict this loop takes back from it is a human refusal
    (`verify.human_confirm_before_execute`), which bypasses the repair loop
    entirely because no repair policy can answer it.
    """
    cfg = ctx.cfg
    on_failure = _get("verify_failure", cfg["verify.on_failure"])
    cap = int(cfg.get("verify.max_repair_attempts") or 0)
    limit = cap if cap > 0 else UNCAPPED_SAFETY_LIMIT

    out: List[Candidate] = []
    for candidate in candidates:
        if not candidate.valid:
            out.append(candidate)  # already failed upstream (e.g. unparseable response)
            continue

        current, attempt, settled = candidate, 0, None
        while True:
            reason, _fn = _first_failure(ctx, state, current)
            if not reason:
                settled = current
                break

            if current.meta.get(_HUMAN_REFUSED):
                # `verify.human_confirm_before_execute`: a refusal is not a
                # defect in the program, so no `verify.on_failure` policy
                # applies -- there is nothing to repair, a resample would ask
                # the same person the same question, and `degrade` would train
                # past a person who said no. It is `invalid` and not
                # `screened`: the code was never established to be fine.
                settled = current.failed(reason, kind="invalid")
                _record(settled, phase="human_confirm", policy="refused",
                        reason=reason, attempt=attempt)
                break

            verdict = on_failure(ctx, state, current, reason, attempt)
            if verdict is not None:
                settled = verdict
                break

            attempt += 1
            if attempt > limit:
                settled = _exhausted(ctx, state, current, reason, attempt - 1)
                break
            fresh, err = _regenerate(ctx, state, current, attempt)
            if fresh is None:
                settled = _exhausted(ctx, state, current, f"{reason} | cannot resample: {err}", attempt)
                break
            current = fresh

        if settled.meta.get(_DISPOSITION) == "drop":
            # Removed from the pool, never from the record.
            if getattr(ctx, "rundir", None) is not None:
                ctx.rundir.save_candidate(settled)
            ctx.event("verify_drop", cand_id=settled.cand_id, kind=settled.failure_kind,
                      failure=settled.failure)
            continue
        out.append(settled)

    dedup = _get("dedup", cfg.get("verify.dedup.method", "none"))
    return dedup(ctx, out)
