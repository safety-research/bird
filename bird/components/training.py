"""Stage 3 -- Policy Training. The inner loop `pi = A_M(R)`.

Four `train.backend` values live here, and the split
between `train.backend` and `train.algorithm` is the point of the key pair:
`algorithm` is the **paper's** claim (PPO / SAC / QR-SAC / TD3 / Q-learning /
none) and is recorded verbatim in every artifact; `backend` is **what actually
executes on this machine**. The published configs keep their paper values and
the execution profiles in `configs/_profiles/` swap only the backend, which is
what lets a paper point stay citable while still running on a laptop with no
GPU and no API key.

  mock      A surrogate learner. It compiles the candidate reward, improves a
            policy against *that* reward, and then measures ground-truth
            `env.task_metric` / `env.success`. That indirection is the whole
            reason the harness produces a usable fitness signal: a misaligned
            reward trains a policy that scores badly on the real metric, so
            reward quality propagates into fitness instead of being asserted.
  tabular   Q-learning over `env.discretise`. Exact on an env that declares
            `exact_states` (Singh's hungry-thirsty; `configs/methods/singh_orp.yaml`),
            a binning with a warning anywhere else.
  sb3       stable-baselines3, imported lazily. Never at module import time --
            `registry.load_all()` imports this module unconditionally, so an
            import-time dependency would make every config in the repo fail on
            a machine without torch.
  none      L2R: no policy learning at all. A short receding-horizon shooting
            planner solves the reward online, which still produces trajectories
            and still produces a curve -- a flat one, honestly, because nothing
            is learning.

What a `TrainResult` has to carry, and who reads it (§4 consumes all of it):

  seed_metrics      one entry per seed. `train.seeds_per_candidate` is 1 in
                    every published search loop except LIMEN's, and >1 has to
                    genuinely work or `select.significance` is untestable.
  checkpoints       a real curve at `train.checkpoint_interval`, because
                    `evaluate.fitness.checkpoint_aggregation` offers `final`,
                    `max_over_checkpoints`, `auc`, `last_k_mean` and `iqm` and
                    four of those are meaningless on two endpoints. Eureka's
                    published `max` is optimistically biased *relative to the
                    others*, which you cannot see without the others.
  component_traces  per-named-component values over training. Precondition for
                    `evaluate.feedback.granularity: per_component`; without it
                    that entire branch of §4 is dead code.
  trajectories      `evaluate.rollouts_per_candidate` real rollouts. Screens
                    (TAC/TPE/EPIC) and the preference comparator re-score these,
                    so `states`/`actions` are real arrays, never placeholders.
  gt_reward_curve   the reference reward's curve over the same checkpoints, for
                    Eureka's `evaluate.similarity.metric: pearson_curve`.

Failure is data, not an exception: a
reward that blows up mid-training comes back as `trained=False` with `error`
set, so the candidate still occupies a slot at `select.failure_value` and the
execute-rate statistic stays honest.
"""

from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import inspect
import io
import logging
import math
import os
import pickle
import shutil
import signal
import sys
import tempfile
import json
import time
from pathlib import Path
import traceback
import zlib
from types import CodeType
from typing import Any, Callable, Dict, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from ..budget import BudgetExceeded
from ..config import SB3_ALGORITHMS, ConfigError, reward_language
from ..context import _NoTracker
from ..envs.base import _bin
from ..registry import get as _registry_get, register
from .. import native_signal
from ..types import Candidate, TrainResult, Trajectory

log = logging.getLogger("bird.train")

__all__ = ["compile_reward", "CompiledReward",
           "compile_observation", "CompiledObservation",
           "policy_from_ref"]

#: Hard ceilings on env steps per seed. The paper points ask for 1M-50M steps
#: (`train.env_steps`); honouring that literally would take hours per candidate
#: and the surrogate would learn nothing more than it does at 6k. The cap is
#: applied silently but the *executed* count is what reaches `ctx.budget`, so
#: the budget report says surrogate steps and never pretends to be the paper's.
MOCK_STEP_CAP = 6000
TABULAR_STEP_CAP = 12000
PLANNER_STEP_CAP = 6000

#: Keep the checkpoint curve inside a range where every aggregation in §4 is
#: meaningful: `last_k_mean`/`iqm` need several points, `auc` needs an even
#: sampling, and nobody wants a 2000-row curve from `checkpoint_interval: 10`.
MIN_CHECKPOINTS = 5
MAX_CHECKPOINTS = 20

#: Episodes behind each checkpoint. Two is cheaper and *wrong*: on an env whose
#: ground-truth metric is close to binary, a two-episode estimate swings the
#: curve between 0 and 1 and `checkpoint_aggregation: final` then selects on
#: evaluation noise rather than on the reward. Evaluation steps are counted in
#: the cost report but not against `train.env_steps`, which budgets learning.
EVAL_EPISODES = 4
EVAL_EPISODES_FINAL = 8

#: Episodes behind one sb3 checkpoint when `evaluate.checkpoint_eval_episodes` is
#: null. Named so `_checkpoint_episodes` has one truth to fall back to and so the
#: value is greppable.
_SB3_EVAL_EPISODES = 3


def _checkpoint_episodes(cfg: Any, *, last: bool = False, sb3: bool = False) -> int:
    """`evaluate.checkpoint_eval_episodes` -> episodes behind one checkpoint.

    null (the default) means each backend's own constant, so a CURVE is unchanged
    unless a config asks otherwise. The HASH is a separate claim: `Config.hash()`
    covers all of `_data`, so this key's PRESENCE in `configs/_default.yaml` is
    part of every config's hash even while its value is null and nothing
    evaluates differently. `rstar`'s literal in
    `tests/test_max_over_training_epochs.py` records the arithmetic.

    `EVAL_EPISODES_FINAL`'s role is kept EXPLICIT rather than folded away: the
    surrogate path's last row is the row `checkpoint_aggregation: final` reads, so
    it takes more episodes than the intermediate ones, and an explicit value raises
    the floor without lowering that.

    The sb3 default, `_SB3_EVAL_EPISODES` = 3, sits below the 4 that
    `EVAL_EPISODES`' own comment argues is the minimum defensible; set the key to
    raise it.
    """
    raw = cfg.get("evaluate.checkpoint_eval_episodes") if cfg is not None else None
    if raw is None:
        return (_SB3_EVAL_EPISODES if sb3
                else (EVAL_EPISODES_FINAL if last else EVAL_EPISODES))
    n = max(1, int(raw))
    return max(n, EVAL_EPISODES_FINAL) if (last and not sb3) else n


_REWARD_CLIP = 10.0  # `train.reward_norm: clip`
_DIVERGENCE = 1e12  # |Q| beyond this means the reward blew the learner up
_MAX_LOGGED_TRACEBACKS = 4  # stdout_grep only ever reads the first one


# ==========================================================================
# Reward compilation
# ==========================================================================


def _scalar(x: Any) -> float:
    """Coerce whatever a reward returned into one float, or raise."""
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    arr = np.asarray(x, dtype=float)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    if arr.size == 0:
        raise ValueError("reward returned an empty array")
    return float(arr.mean())  # a vectorised reward over a batch of one env


#: Parameter-name -> which of (s, a, s_next) to bind. LLM-written rewards are
#: not consistent about signature: Eureka-style bodies take `(obs, action)`,
#: GT's take a `State` and an `Action`, T2R's take `(self, ...)` lifted off a
#: class. Bind by name where the name is recognisable and by position otherwise
#: -- a mis-binding shows up as a *silent* wrong fitness, which is worse than a
#: crash, so this table is deliberately generous.
_ARG_BY_NAME = {
    "s": 0, "state": 0, "states": 0, "obs": 0, "observation": 0, "observations": 0,
    "cur_obs": 0, "current_state": 0,
    "a": 1, "act": 1, "action": 1, "actions": 1, "u": 1,
    "s2": 2, "s_next": 2, "next_state": 2, "next_obs": 2, "new_state": 2,
    "next_observation": 2, "sp": 2, "s_prime": 2,
}
_REWARD_NAMES = ("compute_reward", "reward", "reward_fn", "compute_dense_reward",
                 "get_reward", "dense_reward", "r", "_reward")


class CompiledReward:
    """A candidate reward program, callable as `R(s, a, s_next) -> (total, {})`.

    Returns the named-component dict alongside the scalar because component
    *visibility* is a precondition for per-component reflection in §4 -- the
    `generate.output.format: component_dict_return` contract exists for exactly
    this, and a backend that threw the dict away would silently disable it.
    """

    def __init__(self, fn: Callable[..., Any], name: str,
                 weights: Optional[Dict[str, float]] = None) -> None:
        self.fn = fn
        self.name = name
        self.weights = dict(weights or {})
        self.component_names: List[str] = []
        self._bind = self._make_binder(fn)

    @staticmethod
    def _make_binder(fn: Callable[..., Any]) -> List[int]:
        try:
            params = list(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError):  # builtins / C callables
            return [0, 1, 2]
        plan: List[int] = []
        nxt = 0
        for p in params:
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if p.kind == inspect.Parameter.KEYWORD_ONLY and p.default is not inspect.Parameter.empty:
                continue
            if p.name in ("self", "env", "task", "cfg", "config"):
                plan.append(-1)  # a method lifted off a class: pass None and let it fail loudly
                continue
            slot = _ARG_BY_NAME.get(p.name.lower())
            if slot is None:
                slot = nxt
            plan.append(slot)
            nxt = max(nxt, slot) + 1
            if len(plan) >= 4:
                break
        # `or [0]` IS A GUARD AND NOT AN OVERSIGHT -- do not "fix" it to `[]`.
        # An empty plan means no NAMED parameter was bound, which happens two
        # ways: `*args` (which can accept the state, and must be given it)
        # and a signature with no positional slot at all, `def compute_reward():`
        # or `**kw` only. Handing the second one an argument it declared nowhere
        # to put is what makes the call RAISE. Binding it "correctly" with zero
        # arguments would instead succeed and return whatever constant the
        # program computes from nothing -- scoring every transition identically,
        # which is a plausible number rather than a crash, and so invisible.
        #
        # The loud failure is the whole value here, because it is the LAST line
        # of defence, not the first. `signature_parse` rejects both shapes
        # before execution (`verification.py::static_signature_parse`, §2
        # Validity) and costs no training slot -- but only where
        # `verify.enabled` is true. On a config where it is false, this line is
        # the only thing standing between a zero-arity program and a silent
        # constant reward; `roska` under the tester profile is such a config,
        # and the mock LLM's `wrong_signature` archetype reaches it whenever a
        # prompt change re-rolls the mock's draws. The invariant is pinned by
        # `tests/test_unbindable_reward.py`, which asserts that no value comes
        # back rather than pinning this mechanism.
        return plan or [0]

    @staticmethod
    def _apply_bind(bind: List[int], s: Any, a: Any, s_next: Any) -> List[Any]:
        """The plan from `_make_binder`, applied: one copy, used by the host call
        and the device row alike, so a change to the plan's shape
        -- a fourth slot, another sentinel -- cannot land in one and not the other."""
        pool = (s, a, s_next)
        return [None if i < 0 else (pool[i] if i < 3 else None) for i in bind]

    def __call__(self, s: np.ndarray, a: np.ndarray,
                 s_next: np.ndarray) -> Tuple[float, Dict[str, float]]:
        out = self.fn(*self._apply_bind(self._bind, s, a, s_next))
        return self._unpack(out)

    @staticmethod
    def _split(out: Any) -> Tuple[Any, Dict[str, Any]]:
        """The STRUCTURE of a reward's return, values untouched: `(total | None, comps)`.

        One place for the four accepted shapes -- a bare scalar, `(total, dict)`,
        a bare dict, a 1-tuple of either -- so the host path (`_unpack`) and the
        traced path (`traced`) cannot disagree about which is which. Structure
        only: a jax trace fixes the container shape at trace time and this
        function reads containers, never values, so it is the same code under
        both.
        """
        if isinstance(out, dict):
            return None, {str(k): v for k, v in out.items()}
        if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], dict):
            return out[0], {str(k): v for k, v in out[1].items()}
        if isinstance(out, (tuple, list)) and len(out) >= 1 and not isinstance(out[0], dict):
            return out[0], {}
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            return None, {str(k): v for k, v in out[0].items()}
        return out, {}

    def _unpack(self, out: Any) -> Tuple[float, Dict[str, float]]:
        raw_total, raw_comps = self._split(out)
        comps: Dict[str, float] = {k: _scalar(v) for k, v in raw_comps.items()}
        total: Optional[float] = None if raw_total is None else _scalar(raw_total)

        if comps and self.weights:
            # `generate.output.format: component_dict_plus_weights` (RDA, GT):
            # the LLM writes the weights separately, so they -- not whatever the
            # body summed -- define the total. Silently ignoring them would make
            # the whole `weights_only` edit mode a no-op.
            total = sum(float(self.weights.get(k, 1.0)) * v for k, v in comps.items())
        elif total is None:
            total = sum(comps.values())
        if comps and not self.component_names:
            self.component_names = list(comps)
        return float(total), comps

    def _device_row(self, asarray: Callable[[Any], Any], ndim: Callable[[Any], int],
                    size: Callable[[Any], int], shape: Callable[[Any], Any],
                    reshape_scalar: Callable[[Any], Any], zero: Callable[[], Any],
                    language: str) -> Callable[[Any, Any, Any], Tuple[Any, Dict[str, Any]]]:
        """`row()`'s body, with the array library passed in.

        TWO LANGUAGES, ONE ROW. `row()` (jax) and `torch_row()` (torch) are the
        same six operations -- bind the candidate's arguments, split the return
        structure, refuse a non-scalar value by name, apply the LLM's weights,
        fill `component_names` -- and the only difference is which library's
        `asarray` and zero they are written in. Keeping one body is the same
        argument `_apply_bind` and `_split` already make one level down: a
        change to the weighting convention or to the non-scalar refusal that
        landed in one language and not the other would be a DIFFERENT REWARD
        under one `generate.reward_language`, which is precisely the class of
        defect `tests/test_no_method_branching.py` exists to prevent one level
        up. `language` appears in the refusal message only.
        """
        fn, bind, weights = self.fn, list(self._bind), dict(self.weights)

        def _dev(label: str, v: Any) -> Any:
            arr = asarray(v)
            if ndim(arr) == 0:
                return arr
            if size(arr) == 1:
                return reshape_scalar(arr)
            raise TypeError(
                f"reward {label} has shape {tuple(shape(arr))}; a {language} reward "
                "row returns a scalar for the total and for each component")

        def row(s: Any, a: Any, s_next: Any) -> Tuple[Any, Dict[str, Any]]:
            raw_total, raw_comps = self._split(fn(*self._apply_bind(bind, s, a, s_next)))
            comps = {k: _dev(f"component {k!r}", v) for k, v in raw_comps.items()}
            if comps and weights:
                total = sum((asarray(float(weights.get(k, 1.0))) * v
                             for k, v in comps.items()), zero())
            elif raw_total is None:
                total = sum(comps.values(), zero())
            else:
                total = _dev("total", raw_total)
            if comps and not self.component_names:
                self.component_names = list(comps)
            return total, comps

        return row

    def torch_row(self) -> Callable[[Any, Any, Any], Tuple[Any, Dict[str, Any]]]:
        """The reward as one DEVICE row in torch: `generate.reward_language: torch`.

        WHY THIS EXISTS AND WHAT IT IS FOR. A GPU-batched learner hands the
        reward a `(num_envs, *)` tensor, and `__call__` is a per-transition host
        contract whose `_unpack` casts every value to a Python float -- thousands
        of Python calls per env step. Eureka's own rewards are already in this
        batched shape and its prompt asks for it verbatim -- `@torch.jit.script def compute_reward(...) ->
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]`
        (`refs/code/Eureka/eureka/utils/prompts/reward_signature.txt`), with
        "make sure that the code is compatible with TorchScript (e.g., use
        torch tensor instead of numpy array)" in the system prompt
        (`initial_system.txt`).

        SO THE BATCH IS THE CANDIDATE'S OWN, NOT A vmap. Unlike `traced()`
        below, there is no wrapping: a torch candidate is written against
        batched tensors and broadcasts over the leading axis itself, which is
        how upstream's `compute_reward` works and the reason `torch.jit.script`
        can compile it at all. This row therefore returns `(total, comps)` with
        whatever leading shape the candidate produced, and the ONE thing it
        refuses is the case `row()` refuses for jax -- a value that is neither a
        scalar nor one-per-env, which would be a reward averaged across
        environments wearing a per-environment reward's clothes.
        """
        import torch  # local: this path runs only under reward_language=torch

        return self._device_row(
            asarray=lambda v: (v if isinstance(v, torch.Tensor)
                               else torch.as_tensor(v)),
            ndim=lambda a: int(a.dim()),
            size=lambda a: int(a.numel()),
            shape=lambda a: a.shape,
            reshape_scalar=lambda a: a.reshape(()),
            zero=lambda: torch.zeros((), dtype=torch.float32),
            language="torch")

    def torch_batch(self) -> Callable[[Any, Any, Any], Tuple[Any, Dict[str, Any]]]:
        """`torch_row()` over a BATCH, with the per-env axis kept.

        The batched contract: `(S(n,*), A(n,*), S2(n,*)) -> (total(n,),
        comps{name: (n,)})`, which §2's batched torch probe calls. The candidate broadcasts, so this is `torch_row()`
        with the scalar refusal WIDENED to admit a length-`n` vector -- and
        nothing else, because a value of any other length is the mis-broadcast
        this function exists to catch rather than to reshape.
        """
        import torch  # local: this path runs only under reward_language=torch

        fn, bind, weights = self.fn, list(self._bind), dict(self.weights)

        def _dev(label: str, v: Any, n: int) -> Any:
            arr = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            if arr.dim() == 0:
                # A scalar from a batched reward is a reward that collapsed the
                # fleet -- `.mean()` somewhere in the candidate. Broadcast it,
                # and let the caller record that it happened: refusing would
                # fail candidates whose penalty terms are genuinely global.
                return arr.expand(n) if n > 1 else arr.reshape(1)
            flat = arr.reshape(-1) if arr.dim() > 1 and arr.numel() == n else arr
            if flat.dim() == 1 and flat.numel() == n:
                return flat
            raise TypeError(
                f"reward {label} has shape {tuple(arr.shape)} on a batch of {n} "
                "environments; a batched torch reward returns one value per "
                "environment (or a scalar, which is broadcast)")

        def batch(S: Any, A: Any, S2: Any) -> Tuple[Any, Dict[str, Any]]:
            n = int(S.shape[0])
            raw_total, raw_comps = self._split(fn(*self._apply_bind(bind, S, A, S2)))
            comps = {k: _dev(f"component {k!r}", v, n) for k, v in raw_comps.items()}
            if comps and weights:
                total = sum((float(weights.get(k, 1.0)) * v for k, v in comps.items()),
                            torch.zeros(n, dtype=torch.float32,
                                        device=comps[next(iter(comps))].device))
            elif raw_total is None:
                total = sum(comps.values(),
                            torch.zeros(n, dtype=torch.float32,
                                        device=comps[next(iter(comps))].device)
                            if comps else torch.zeros(n, dtype=torch.float32))
            else:
                total = _dev("total", raw_total, n)
            if comps and not self.component_names:
                self.component_names = list(comps)
            return total, comps

        return batch

    def row(self) -> Callable[[Any, Any, Any], Tuple[Any, Dict[str, Any]]]:
        """The reward as ONE un-jitted device row: `(s, a, s_next) -> (total, comps)`.

        THE SHARED PRIMITIVE of `generate.reward_language: jax`.
        `__call__` is the host contract -- `_unpack` casts every value to a
        Python float -- and a trace dies on that cast. This is the same contract
        with the numbers left on device: the candidate's ARGUMENT BINDING
        (`self._bind`, the plan `_make_binder` derived from its signature -- the
        TRAINER'S binding, so the jax tier calls a candidate the way the tier
        that trains it does) and the WEIGHTED TOTAL (`component_dict_plus_
        weights`: the LLM's weights, not whatever the body summed, define the
        total) are applied here, in `jnp`, so neither convention has a second
        copy anywhere. Two callers, one truth: `traced()` below wraps this in
        `jit(vmap(...))` for the verifier's probe; `fasttd3._JaxVecEnvView` takes
        this row into ITS OWN jit, fused with phi and the auto-reset, so it can
        time the compile it actually pays.

        `component_names` is filled at trace time, as `_unpack` fills it on the
        first host call, so the attribute means the same thing in both tiers.

        WHAT IS REFUSED, and where. The return STRUCTURE is Python containers,
        read by `_split` -- static under a trace, since a container shape cannot
        depend on a traced value. Every VALUE must be a scalar: `_scalar`'s
        host-side tolerance (a batch-of-one array, a vector averaged down) has
        no traced counterpart, because averaging a vector is a different reward
        and a trace is where that stops being silent. A non-scalar raises
        `TypeError` at trace, naming the component -- the same candidate the
        verifier's `output_shape` refuses under the traced probe, so a program
        reaching here has already been shown to return scalars.
        """
        import jax.numpy as jnp  # local: this path runs only under reward_language=jax

        # `_device_row` is the shared body (see there): `traced()` wraps what
        # this returns, and `fasttd3._JaxVecEnvView` takes the same row into its
        # own jit.
        return self._device_row(
            asarray=jnp.asarray,
            ndim=lambda a: int(a.ndim),
            size=lambda a: int(a.size),
            shape=lambda a: a.shape,
            reshape_scalar=lambda a: a.reshape(()),
            zero=lambda: jnp.asarray(0.0),
            language="jax")

    def traced(self, *, with_action: bool = True,
               with_next: bool = True) -> Callable[[Any, Any, Any], Tuple[Any, Dict[str, Any]]]:
        """`jax.jit(jax.vmap(self.row()))` over a batch: `(S, A, S2) -> (total(n,), comps)`.

        `with_action=False` / `with_next=False` mark that slot unbatched (a
        `None` passed through), for a sampler that yielded no actions. Built
        FROM `row()` and nothing else, so what the verifier proves under this
        function is what `fasttd3._JaxVecEnvView` traces under its own.
        """
        import jax  # local: this path runs only under reward_language=jax

        in_axes = (0, 0 if with_action else None, 0 if with_next else None)
        return jax.jit(jax.vmap(self.row(), in_axes=in_axes))


#: Names the exec namespace owns, so "last function defined wins" never picks
#: a module we injected. `jnp`/`jax` are present only under `reward_language:
#: jax`, `torch` and the typing names only under `torch`, and all of them are
#: listed unconditionally: a name that is absent is not picked.
_NAMESPACE_KEYS = ("np", "numpy", "math", "jnp", "jax", "torch",
                   "Tuple", "Dict", "List", "Optional", "Any", "Union")


def _reward_namespace(language: str = "numpy") -> Dict[str, Any]:
    """The globals a candidate's program (reward or observation) executes in.

    `{"np", "numpy", "math"}`; under `generate.reward_language: jax` also
    `jnp` and `jax`, imported HERE so the module stays
    importable with no jax installed -- `registry.load_all()` imports this file
    on every run. The verifier's `_restricted_globals` grants the same names
    behind its builtins fence; the two lists agree by construction only if
    both are edited together, and `tests/test_jax_reward.py` holds them to it.
    """
    ns: Dict[str, Any] = {"np": np, "numpy": np, "math": math}
    if language == "jax":
        import jax  # local: see the docstring
        import jax.numpy as jnp

        ns["jnp"], ns["jax"] = jnp, jax
    if language == "torch":
        # `torch`, and `Tuple`/`Dict`/`List`/`Optional` beside it. The typing
        # names are not decoration: Eureka's prompt asks for
        # `@torch.jit.script def compute_reward(...) -> Tuple[torch.Tensor,
        # Dict[str, torch.Tensor]]` (reward_signature.txt), TorchScript
        # requires the annotation to be resolvable, and a program the prompt
        # asked for that raises `NameError: Tuple` at exec is a candidate the
        # harness broke, not one the model got wrong.
        import typing

        import torch  # local: see the docstring

        ns["torch"] = torch
        for _n in ("Tuple", "Dict", "List", "Optional", "Any", "Union"):
            ns[_n] = getattr(typing, _n)
    return ns


def compile_reward(code: Union[str, CodeType], candidate: Optional[Candidate] = None,
                   bindings: Optional[Dict[str, Any]] = None,
                   language: str = "numpy") -> CompiledReward:
    """Execute a candidate's source and return the reward callable it defines.

    Shared entry point on purpose: §2's smoke checks and §3's training must
    agree on *which* function a candidate defines and how it is called, or a
    candidate can pass verification and then fail training for a reason
    verification could never have caught.

    Globals and locals are the same dict so that helper functions the program
    defines can see each other (`generate.output.forbid_helper_functions:
    false` permits them). No sandbox: `verify.forbidden_symbols` is the gate
    for anti-hacking, and it runs statically in §2 where it belongs.

    `code` may also be a module code object already produced by `compile`, and
    `bindings` names the program may read, bound before it runs. Both exist for
    one caller: R*'s parameter alignment (`alignment.py::_Parametrised`)
    compiles a candidate ONCE with its float literals rewritten to reads of a
    parameter vector and re-executes that code object per parameter step. It
    goes through here rather than `exec`-ing for itself so that the function it
    scores is found by the same rule training will use -- the whole point of a
    single entry point -- and because re-executing the module (a few `def`s)
    rather than rebinding a global keeps module-level statements evaluated per
    step, exactly as the re-parsed source was.
    """
    if isinstance(code, CodeType):
        code_obj = code
    else:
        if not code or not str(code).strip():
            raise ValueError("candidate has no reward code")
        code_obj = compile(str(code), f"<candidate:{getattr(candidate, 'cand_id', '?')}>", "exec")
    ns = _reward_namespace(language)
    if bindings:
        ns.update(bindings)
    exec(code_obj, ns, ns)

    fn = None
    name = ""
    for probe in _REWARD_NAMES:
        cand = ns.get(probe)
        if callable(cand):
            fn, name = cand, probe
            break
    if fn is None:  # last function defined wins -- dicts preserve insertion order
        for key, value in reversed(list(ns.items())):
            if key.startswith("__") or key in _NAMESPACE_KEYS:
                continue
            if inspect.isfunction(value):
                fn, name = value, key
                break
    if fn is None:
        raise ValueError("candidate defines no callable reward function "
                         f"(expected one of {_REWARD_NAMES[:3]})")
    return CompiledReward(fn, name, getattr(candidate, "weights", None))


# ==========================================================================
# The co-designed observation -- `problem.search_space: [... observation]`
# ==========================================================================
#
# LIMEN (Jaswal, Baghel & Chopra, RLC 2026, §4.1) is the one published method
# whose search space is the MDP INTERFACE rather than the reward: one completion
# emits `get_observation(state)` and `compute_reward(state, action, next_state)`
# together, and the policy is trained on the observation the model chose.
#
# THIS HALF MUST BE INSTALLED, NOT MERELY PARSED. `generation.py` prompts for
# it, `_apply_co_design` lifts it out and dimension-checks it against
# `co_design.observation_max_dim`, `artifacts.py` writes it to
# `candidates/<id>/observation.py`, and `selection.py::_observation_dim` bins
# the MAP-Elites archive on its width. If no backend read
# `candidate.observation_code`, the policy would train on the environment's own
# observation, the archive's first descriptor axis would measure something no
# learner had seen, and `limen` and `limen_reward_only` -- the paper's central
# §5.4 ablation -- would induce the SAME MDP while differing in config, in hash
# and in run directory, with nothing in any artifact saying so. This section
# is the install.
#
# WHERE IT MAY BE APPLIED, and the reason the answer is "one place". See
# `EnvAdapter.policy_features`: the observation is what the POLICY consumes and
# nothing else. Re-basing the candidate reward onto it would score a different
# program from the one the model wrote (the prompt hands both functions the same
# `state`); re-basing `task_metric` onto it would let a candidate move the ground
# truth by choosing what it observes.

#: Probed by name first, positionally never: an observation function is not the
#: reward and must not be found by `compile_reward`'s "last function defined
#: wins" fallback. `generation.py::_extract_observation` only ever extracts a
#: `def get_observation`, so the aliases below are for a hand-written or
#: externally-supplied program, not for anything this repo generates.
_OBSERVATION_NAMES = ("get_observation", "observation_fn", "get_obs", "observe", "phi")

#: How many synthetic states `CompiledObservation` calls phi on to MEASURE its
#: width and range. Small on purpose: this runs once per candidate per training
#: call (and LIMEN's cascade screen adds one more), phi is pure Python, and the
#: quantity that has to be exact -- the feature COUNT -- is settled by the first
#: probe that returns.
_OBS_PROBES = 9

#: A finite window for the PROBE DOMAIN of an env whose declared state box is
#: infinite on some axis. An unbounded `Box` is legal for gymnasium and for
#: SB3, and an adapter whose per-column ranges are unmeasured may publish one
#: on purpose rather than a guessed number. Without this clip `_measure`'s lattice is
#: unusable on such an axis: the `lo`/`hi` corners are -inf/+inf, and the
#: midpoint `0.5 * (lo + hi)` and every draw `lo + (hi - lo) * u` are NaN
#: (`inf - inf`). `gym_mujoco` and `metaworld` never reach it: both publish
#: finite bounds of their own (`vel_lo`/`vel_hi` on each family row; metaworld
#: clips its raw space to +/-10, the same figure as this constant, so nothing
#: real is clipped there either). Only the probe domain is clipped; phi itself
#: is never clipped and the measured feature range is padded below.
_OBS_PROBE_CLIP = 10.0


class CompiledObservation:
    """A candidate's co-designed `get_observation(state)`, as a callable phi.

    `dim`, `low` and `high` are MEASURED by calling phi, never read off the
    source, and that distinction is load-bearing rather than tidy:
    `selection._literal_return_width` -- the AST estimate of the width -- says
    in its own docstring that its AST read is a LOWER
    BOUND (`np.concatenate([a, b])` reports 2, not the true width), and
    `generation._estimate_obs_dim` returns `None` on anything non-literal. An
    archive binned on a lower bound is a different archive from one binned on
    the truth, and `co_design.observation_max_dim` enforced against a lower
    bound is a cap that does not hold.

    Measuring it also makes the declared `OBS_DIM` checkable rather than
    trusted: `observation_dim_declared` is recorded beside the measured width so
    a program that lies about its own width is visible in `meta.json` instead of
    silently redefining the archive's x-axis.
    """

    def __init__(self, fn: Callable[..., Any], name: str, env: Any,
                 language: str = "numpy") -> None:
        self.fn = fn
        self.name = name
        self.language = language
        self.dim = 0
        self.low = np.zeros(0)
        self.high = np.zeros(0)
        self.n_probe_failures = 0
        self._measure(env)

    # -- measurement -------------------------------------------------------

    def _measure(self, env: Any) -> None:
        """Call phi on synthetic states drawn from the env's DECLARED bounds.

        Synthetic and not `env.reset()`/`env.random_state()` for one reason that
        is not performance: both of those advance or read adapter state
        (`_rng`, `_dr_now`, and on MetaWorld a MuJoCo snapshot cache), and a
        measurement that perturbs the run's RNG streams would make the install
        of an observation change the reward search around it. The probe is a
        fixed, seeded lattice over `[obs_low, obs_high]`, so it costs no env
        steps, touches nothing, and returns the same numbers on every process.
        """
        dim_env = int(getattr(env, "obs_dim", 0) or 0)
        lo = np.clip(np.asarray(getattr(env, "obs_low", np.zeros(dim_env)), dtype=float),
                     -_OBS_PROBE_CLIP, _OBS_PROBE_CLIP)
        hi = np.clip(np.asarray(getattr(env, "obs_high", np.ones(dim_env)), dtype=float),
                     -_OBS_PROBE_CLIP, _OBS_PROBE_CLIP)
        if lo.shape != hi.shape or lo.size == 0:
            lo, hi = np.zeros(max(1, dim_env)), np.ones(max(1, dim_env))
        rng = np.random.default_rng(0)
        probes = [lo, hi, 0.5 * (lo + hi)]
        probes += [lo + (hi - lo) * rng.random(lo.shape)
                   for _ in range(max(0, _OBS_PROBES - len(probes)))]

        rows: List[np.ndarray] = []
        first: Optional[BaseException] = None
        if self.language == "jax":
            # ONE trace over the whole lattice: under this language phi is
            # applied on device by `fasttd3._JaxVecEnvView`, so the
            # question is whether it TRACES, and a trace is all-or-nothing --
            # there is no per-probe failure count to keep, only a phi that
            # compiles under `jit(vmap)` and one that does not.
            import jax  # local: this branch runs only under reward_language=jax
            import jax.numpy as jnp

            try:
                batch = jax.jit(jax.vmap(self.fn))(jnp.asarray(np.stack(probes)))
                batch = np.asarray(jax.block_until_ready(batch), dtype=float)
                rows = [self._as_features(batch[i]) for i in range(batch.shape[0])]
            except Exception as exc:  # noqa: BLE001 -- the trace failing IS the verdict
                self.n_probe_failures = len(probes)
                first = exc
                rows = []
        else:
            for s in probes:
                try:
                    rows.append(self._as_features(self.fn(np.asarray(s, dtype=float))))
                except Exception as exc:  # noqa: BLE001 -- counted, then raised if total
                    self.n_probe_failures += 1
                    first = first or exc
                    continue
        if not rows:
            raise ValueError(
                f"co-designed observation {self.name}() raised on every one of "
                f"{len(probes)} probe states -- there is no feature space to train a "
                f"policy on ({type(first).__name__}: {first})")
        # A phi whose width varies with the state is not an observation
        # function: the policy's input layer is fixed before the first step, so
        # this has to be a refusal and not a truncation.
        widths = {int(r.size) for r in rows}
        if len(widths) > 1:
            raise ValueError(
                f"co-designed observation {self.name}() returned "
                f"{sorted(widths)} features on different states; a policy's input "
                "width cannot vary per step")
        stack = np.stack(rows)
        self.dim = int(stack.shape[1])
        # Padded, and honest about why: the probe covers the declared state box,
        # not the reachable set, so a measured extreme is a lower bound on the
        # true one. Nothing CLIPS to these -- they populate SB3's `Box` (which
        # does not clip without `VecNormalize`) and `_ObsView.discretise`'s bin
        # edges -- so a generous pad costs a slightly coarser tabular binning
        # and a narrow one would fold distinct features into one bin.
        span = np.maximum(stack.max(axis=0) - stack.min(axis=0), 1.0)
        self.low = stack.min(axis=0) - span
        self.high = stack.max(axis=0) + span

    @staticmethod
    def _as_features(out: Any) -> np.ndarray:
        arr = np.asarray(out, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError("observation returned no features")
        if not np.all(np.isfinite(arr)):
            raise FloatingPointError(f"observation returned {out!r}")
        return arr

    # -- the call the policy makes ------------------------------------------

    def __call__(self, state: Any) -> np.ndarray:
        """phi(s). Raises on a non-finite or wrong-width return.

        Raising rather than substituting zeros: a feature vector that silently
        becomes zeros is a policy trained blind, and stage 3 already has a place
        to put that -- `train.failure_detection` routes the exception and the
        candidate comes back `trained=False` with the reason, which is the same
        treatment a reward that blows up mid-training gets.
        """
        arr = self._as_features(self.fn(np.asarray(state, dtype=float)))
        if arr.size != self.dim:
            raise ValueError(
                f"co-designed observation {self.name}() returned {arr.size} features "
                f"here and {self.dim} when it was measured")
        return arr


def compile_observation(code: str, env: Any,
                        candidate: Optional[Candidate] = None,
                        language: str = "numpy") -> CompiledObservation:
    """Execute a candidate's observation source and measure the phi it defines.

    Same namespace policy as `compile_reward` and for the same reason -- globals
    and locals are one dict so helpers the program defines can see each other,
    and there is no sandbox because `verify.forbidden_symbols` is the gate for
    that and it runs statically in §2.

    THE WHOLE PROGRAM IS EXECUTED FIRST, AND THAT IS NOT AN OPTIMISATION.
    `generation._apply_co_design` lifts `get_observation` out with
    `ast.get_source_segment`, so what reaches here is a bare `def` -- and
    `_apply_co_design`'s own docstring already says what that means:
    "`observation_code` is a view onto the program, not a partition of it".
    LIMEN's prompt asks for both functions IN THE SAME CODE BLOCK, so a shared
    module-level helper is the normal shape of a real answer and the lift leaves
    it behind.

    On `limen` against a real generator (`mt10_reach-v3`), candidates write a
    `_tcp(state)` helper -- the tool-centre-point offset this adapter's own
    field docs point at, 4.5 cm below `hand_*` -- and call it from both
    functions. Executed alone, the lifted segment dies at

        NameError: name '_tcp' is not defined

    on every probe state, and the search produces nothing.

    So the two functions share one namespace. They were written in one block
    and they DO resolve each other's names -- that is the contract, not an
    accident, and isolating them is not fidelity but breakage. The MOCK does not
    exercise this: it inlines its `_f` accessor into `get_observation`
    (`_OBS_ACCESSOR` exists precisely so the lifted segment is self-contained),
    so an isolated namespace would pass against the double and fail on a real
    generator. A test double that satisfies a contract more strictly than the
    real thing does is the same class of blind spot as one that satisfies a
    different clause of it.

    The recorded segment is still what DEFINES the function -- it is exec'd
    second, into the program's namespace, so a symbol mapping applied to
    `observation_code` alone (`_apply_co_design` calls `_apply_symbol_mapping`
    on it) wins over whatever the unmapped program said. A program that will not
    execute is a debug line and not a failure here: the segment alone may still
    be self-contained, and stage 3 has already refused the candidate if the
    reward would not compile.
    """
    if not code or not str(code).strip():
        raise ValueError("candidate has no observation code")
    cid = getattr(candidate, "cand_id", "?")
    ns = _reward_namespace(language)
    program = str(getattr(candidate, "reward_code", "") or "")
    if program.strip() and program.strip() != str(code).strip():
        try:
            exec(compile(program, f"<candidate:{cid}>", "exec"), ns, ns)
        except Exception as exc:  # noqa: BLE001 - the segment may stand alone
            log.debug("  %s: the candidate program did not execute while building the "
                      "observation namespace (%s: %s); falling back to the lifted "
                      "segment alone", cid, type(exc).__name__, exc)
            ns = _reward_namespace(language)
    exec(compile(str(code), f"<observation:{cid}>", "exec"), ns, ns)
    for probe in _OBSERVATION_NAMES:
        fn = ns.get(probe)
        if callable(fn):
            return CompiledObservation(fn, probe, env, language)
    # The WHOLE set, not a prefix. `compile_reward` truncates its own list to
    # three because `_REWARD_NAMES` has eight and the first three are the ones
    # any real candidate uses; there are five here and a candidate that defined
    # `observe` or `phi` would read a message saying its function is not among
    # the accepted names, which is the opposite of true.
    raise ValueError("candidate defines no callable observation function "
                     f"(expected one of {_OBSERVATION_NAMES})")


class _ObsView:
    """An env view whose POLICY FEATURE SPACE is the co-designed observation.

    Delegates everything to the adapter it wraps -- `reset`, `step`,
    `reference_reward`, `task_metric`, `success`, DR, the lot -- and overrides
    only the members that describe what a POLICY consumes. Handed to
    `_make_learner` and to nothing else: `_run_backend` keeps the raw adapter for
    `_evaluate_policy`, `_rollout` and `set_dr`, so the trajectory that is
    recorded, scored and re-scored by the screens is the one the environment
    actually produced.

    `exact_states` is `None` here whatever the inner adapter says, and that is a
    fact rather than a conservative choice: `exact_states is not None` promises
    `discretise` is a BIJECTION over the state space, and phi is a
    model-authored map with no injectivity guarantee at all -- `toy_hungry_
    thirsty`'s 64 states can collapse to three features. So the table below is a
    binning, `_make_learner`'s `auto` path picks the linear-policy searcher, and
    `train.backend: tabular` gets the binning warning it already prints. Calling
    it exact would make Singh's one honest tabular claim a false one.
    """

    def __init__(self, env: Any, phi: CompiledObservation) -> None:
        self._env = env
        self._phi = phi
        self.obs_dim = int(phi.dim)
        self.obs_low = np.asarray(phi.low, dtype=float)
        self.obs_high = np.asarray(phi.high, dtype=float)
        self.exact_states = None
        # Bins per feature, chosen so the table is no bigger than the one the
        # inner adapter already sized for itself. `n_disc_states` is a memory
        # claim (`_QLearner` allocates `n_disc_states x n_actions`), so a
        # 3-feature phi must not turn a 324-cell table into 10^3 x n_actions
        # just because the feature count is small enough to make it look cheap.
        budget = max(2, int(getattr(env, "n_disc_states", 2) or 2))
        self._bins = max(2, int(budget ** (1.0 / max(1, self.obs_dim))))
        # AND THEN THE PRODUCT IS CAPPED, because the per-feature count alone does
        # not bound it. `_bins` has a floor of 2 -- one bin per axis is not a
        # binning -- so `bins ** obs_dim` OVERSHOOTS the budget whenever the
        # budget is small: an env advertising 2 (or none at all, the getattr
        # default) with a 3-feature phi gives 2**3 = 8, and a 30-feature phi on
        # the same env gives 2**30. `_QLearner` allocates `n_disc_states x
        # n_actions` up front, so that is a real allocation and not a bookkeeping
        # number.
        #
        # The budget is spent on a PREFIX of the features rather than by dropping
        # to one bin each. Fewer features at a usable resolution beats every
        # feature at a resolution that cannot separate anything -- and the axes
        # that fall off are named in the warning, because a tabular learner blind
        # to feature 3 onward is a fact about the run.
        keep = self.obs_dim
        while keep > 0 and self._bins ** keep > budget:
            keep -= 1
        self._keep = keep
        self.n_disc_states = int(self._bins ** keep) if keep else 1
        if keep < self.obs_dim:
            log.warning(
                "train.backend=tabular over a %d-feature co-designed observation on "
                "%s: %d bins each would need %d table cells against a budget of %d, "
                "so only the first %d feature(s) index the table and the rest are "
                "invisible to it. The linear-policy searcher (the `auto` path) sees "
                "all %d.", self.obs_dim, getattr(env, "name", "?"), self._bins,
                self._bins ** self.obs_dim, budget, keep, self.obs_dim)

    # -- the overrides ------------------------------------------------------

    def policy_features(self, state: Any) -> np.ndarray:
        return self._phi(state)

    def discretise(self, state: Any) -> int:
        """Uniform binning of phi(s) -- `_bin` in `bird.envs.base`, per feature."""
        f = self._phi(state)
        idx = 0
        for i in range(self._keep):
            idx = idx * self._bins + _bin(float(f[i]), float(self.obs_low[i]),
                                          float(self.obs_high[i]), self._bins)
        return int(idx)

    # -- everything else is the adapter's ----------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only reached for names this class does not define, so the four
        # overrides above can never be shadowed by the inner adapter.
        return getattr(self._env, name)


def _observation_for(ctx: Any, candidate: Candidate) -> Optional[CompiledObservation]:
    """Compile the candidate's observation, or `None` if none is being searched.

    `problem.search_space` IS THE GATE, and this function is what makes that key
    honoured rather than declared. Were it read only by
    `config._check_coherence`, nothing would dispatch on it, and a config
    extending it with `[observation]` would validate clean and change nothing --
    a declared-but-unread key.

    Reading `search_space` rather than `generate.co_design.observation_fn` is
    deliberate even though `_check_coherence` keeps the two consistent (obs_fn
    requires `observation` in the space). §0's key says what the searched object
    IS; §1's says how it is elicited. Stage 3 installs what is being searched,
    so it asks §0 -- which is also what makes `limen_reward_only`'s
    `search_space: [reward]` a behavioural statement instead of a comment, and
    what makes an observation arriving on a reward-only config a warning rather
    than a silent install.
    """
    space = ctx.cfg.get("problem.search_space") or []
    if "observation" not in space:
        if candidate.observation_code:
            log.warning("%s carries observation code but problem.search_space is %s: "
                        "the policy trains on the environment's own observation and "
                        "the co-designed one is not installed",
                        candidate.cand_id, list(space))
        return None
    if not candidate.observation_code:
        # Not an error here. `_apply_co_design` already fails a candidate that
        # was ASKED for an observation and did not write one, so reaching this
        # line means the config declares the space and `co_design.observation_fn`
        # is off -- a staged search space, or a partially-configured one.
        return None
    return compile_observation(candidate.observation_code, ctx.env, candidate,
                               language=reward_language(ctx.cfg))


def _install_observation(ctx: Any, candidate: Candidate, env: Any,
                         kind: str = "") -> Tuple[Any, Optional[CompiledObservation]]:
    """`(what the learner sees, phi)`. Records the MEASURED width on the candidate.

    `meta["observation_dim"]` is the key `selection._observation_dim` prefers
    over its own AST heuristic, so installing the observation also converts
    LIMEN's first MAP-Elites descriptor from an estimate into a measurement.
    `_apply_co_design` writes `meta["obs_dim"]` -- the model's DECLARED count --
    under a different name on purpose: the two are kept side by side so a
    program that lies about its width is visible rather than reconciled.

    The measured width is also what a forked candidate worker has to ship home;
    it rides on `Candidate.meta`, which `_merge_child` replays field by field.
    """
    phi = _observation_for(ctx, candidate)
    if phi is None:
        return env, None
    if kind == "planner":
        # `train.algorithm: none` (L2R). A shooting planner has no policy and no
        # feature space: it re-plans over raw states every step, scoring action
        # sequences under the model. So there is nothing here for phi to be the
        # input of, and installing it would leave `observation_dim` in the
        # artifact asserting a co-design that no consumer exists for -- the
        # fabricated pin in miniature. Refused and named, rather than recorded.
        #
        # No config pairs them (L2R does not co-design an observation), so this
        # is a guard against a future one.
        log.warning("%s: problem.search_space includes 'observation' but "
                    "train.algorithm resolves to the planner backend, which has no "
                    "policy and no feature space -- the co-designed observation is "
                    "not installed", candidate.cand_id)
        candidate.meta["observation_installed"] = False
        candidate.meta["observation_error"] = "no policy: train.algorithm=none"
        return env, None
    # THE CAP, RE-CHECKED AGAINST THE MEASUREMENT. `_apply_co_design` already
    # enforces `observation_max_dim` in §1, against the model's declared
    # `OBS_DIM` or `_estimate_obs_dim`'s AST read -- and that read is a LOWER
    # BOUND (it returns `None` on anything non-literal, and 2 for
    # `np.concatenate([a, b])`), so a program that declares nothing and computes
    # its width passes a cap it may be far over. Enforcing it again here is the
    # same constraint applied to a number that is actually the width.
    #
    # WHERE LIMEN PUTS IT, and the gap that leaves. `interface.validate`
    # (interface.py:135) raises `Observation dim N exceeds max_obs_dim` from
    # inside the CRASH FILTER -- §2 in BIRD's terms -- so an over-cap candidate
    # is `invalid` there and costs no training slot, where here it is a stage-3
    # refusal that spends one. Moving it to §2 would need a new `dynamic_check`
    # registry entry (plus the schema and config-doc edits that go with one);
    # enforcing it here at least makes the cap hold, because a cap enforced
    # against a lower bound is not a cap.
    cap = int(ctx.cfg.get("generate.co_design.observation_max_dim", 512) or 512)
    if phi.dim > cap:
        raise ValueError(
            f"co_design.observation_max_dim: {phi.name}() measures {phi.dim} features "
            f"> cap {cap} (§1 saw {candidate.meta.get('obs_dim')!r})")
    candidate.meta["observation_dim"] = int(phi.dim)
    candidate.meta["observation_dim_declared"] = candidate.meta.get("obs_dim")
    candidate.meta["observation_installed"] = True
    if phi.n_probe_failures:
        candidate.meta["observation_probe_failures"] = int(phi.n_probe_failures)
    log.debug("  %s: co-designed observation %s() installed, %d feature(s) "
              "(env obs_dim %s)", candidate.cand_id, phi.name, phi.dim,
              getattr(env, "obs_dim", "?"))
    return _ObsView(env, phi), phi


# ==========================================================================
# Reward normalisation -- `train.reward_norm`
# ==========================================================================


class _RewardNorm:
    """`train.reward_norm`: none | running_std | clip.

    **`running_std` silently undoes an LLM-written weights dict.** Dividing by
    a running standard deviation destroys the *scale* of the reward, and scale
    is the only thing a weights dict controls: `{"reach": 10.0, "effort": 0.1}`
    and `{"reach": 1.0, "effort": 0.01}` become the same reward after
    normalisation. It is a real confound in any config that pairs
    `component_dict_plus_weights` with normalisation, and this class does not
    paper over it -- it just does what the config asked and says so here.

    Normalisation applies to the scalar fed to the learner only. The recorded
    `component_traces` stay raw, because §4 verbalises those numbers back to
    the LLM and normalised component values would be a lie about the program it
    wrote.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def __call__(self, r: float) -> float:
        if self.mode == "clip":
            return max(-_REWARD_CLIP, min(_REWARD_CLIP, r))
        if self.mode != "running_std":
            return r
        self.n += 1
        delta = r - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (r - self.mean)
        if self.n < 2:
            return r
        std = math.sqrt(self.m2 / (self.n - 1))
        return r / (std + 1e-8)


# ==========================================================================
# Cross-iteration policy / replay stores
# ==========================================================================

#: `RunState.policy_ref` and `.replay_ref` are typed `Optional[str]` -- the
#: state stays JSON-serialisable and checkpointable, so the arrays
#: themselves live here and the state carries only a handle. Bounded because a
#: 30-iteration LIMEN run would otherwise pin every policy it ever trained.
#:
#: WHAT A VALUE IS, and why it is not one type. The store holds "whatever this
#: backend needs to resume from", which is backend-specific by nature:
#:
#:   backend        _POLICY_STORE value            _REPLAY_STORE value
#:   mock/tabular   the learner's `snapshot()`     list[(obs, act, next_obs, done)]
#:                  (a Q-table or a linear w)      -- `_QLearner.seen`, <=4096 rows
#:   sb3            `_PolicyBlob`, a uint8 view    `_ReplaySlice`, four columnar
#:                  of `model.save()`'s zip        arrays
#:
#: Both `_PolicyBlob` and `_ReplaySlice` ARE ndarray-shaped (the first IS an
#: ndarray subclass; the second is four of them) so `checkpoint.encode` spills
#: them to sha256-verified `.npy` sidecars exactly like a Q-table, and both
#: cross the `parallelism_parallel` fork by pickle at numpy speed.
#:
#: A ref written by one backend and read by another is IGNORED, not fatal:
#: `_make_learner` shape-guards `init_q`/`init_w`, and `_sb3_apply_policy`
#: warns and trains from scratch. Nothing in the repo can mix backends inside
#: one run (`screens._train_backend_short` re-reads `train.backend`), so this
#: is a resume-from-a-foreign-checkpoint guard rather than a live path.
_POLICY_STORE: Dict[str, Any] = {}
_REPLAY_STORE: Dict[str, Any] = {}
#: cand_id -> (iteration, reward_code) for every `policy:<cand_id>` in
#: `_POLICY_STORE`, so `train.init: warm_start_from_similar` can ask "which
#: stored policy was trained under the reward most like this one?" without a
#: run-dir read (a forked worker has `ctx.rundir = None`). Written by
#: `_remember_code` beside every policy write, pruned to the store's live keys
#: there and in `_merge_child`, and shipped home from a fork like the stores.
#: NOT checkpointed: `checkpoint.export_stores` keeps only the refs reachable
#: from `state`, so after a resume the store holds the incumbent alone and
#: `similar` falls back to `best` -- recorded as such in `init_source`.
_POLICY_CODE: Dict[str, Tuple[int, str]] = {}
#: `train.init: bc_prior`'s slot in `_POLICY_STORE`. One per process, refreshed by
#: `ensure_bc_prior` at every stage-3 entry so the FIFO eviction never reaches it.
BC_PRIOR_REF = "policy:bc_prior"
_STORE_LIMIT = 64

#: A SECOND bound, on bytes. `train.secondary_buffer.size` is a paper pin (GT
#: App. E Table 3: 100,000) that the surrogate learners never reach --
#: `_QLearner.seen` self-caps at 4096 rows, so `_STORE_LIMIT` alone bounds them
#: at 64 x ~0.3 MB. On `train.backend: sb3` the same key is honoured literally:
#: 100,000 Meta-World transitions is 39 floats in and 39 out, i.e. ~33 MB per
#: candidate, and an 8-candidate x 5-iteration search holds 40 of them = 1.3 GB
#: of buffers in the PARENT, of which at most three are ever dereferenceable
#: (`checkpoint.reachable_refs`). Oldest-first, so it is the same eviction order
#: `_STORE_LIMIT` already uses and `_merge_child`'s write-order replay still
#: reproduces it exactly. Far above anything the surrogate backends can
#: produce, so this is dead code on mock/tabular and the two backends keep one
#: eviction rule.
#:
#: MEASURED on Meta-World MT10 (obs 39, act 4, SAC, 8 candidates,
#: `secondary_buffer.size: 100000`):
#:
#:   _PolicyBlob    3.30 MB/candidate   -> 64 entries = 211 MB, so `_STORE_LIMIT`
#:                                         is the binding bound for policies
#:   _ReplaySlice  31.38 MB/candidate   -> 512 MB would hold 16, i.e. two
#:                                         iterations of history at 8 a round
#:
#: The policy blob depends on the learner: a larger network measures 304.48 MiB
#: (92x the MT10 figure; measured by subtraction, a serialised message carrying
#: one policy minus the same message carrying none), and at that size a 512 MiB cap holds
#: ONE, every store past the first evicts, and oldest-first eviction drops the
#: INCUMBENT -- the one entry the run cannot lose. When the incumbent's replay
#: buffer is evicted GT falls back to an empty secondary, and that is LOUD:
#: `_sb3_run` warns and `seed_metrics[*].secondary_transitions` reads 0.
#: DO NOT SIZE THIS CAP OFF A NUMBER IN A COMMENT, including these.
#:
#: WHY 16 GiB RATHER THAN A TIGHTER FIT: worst case is 40 candidate blobs plus
#: the incumbent = 41, and 41 x 304.48 MiB = 12.19 GiB, so 16 GiB is never
#: reached by a full run and the cap stays NON-BINDING for its whole length --
#: which is the intent. `_STORE_LIMIT` = 64 > 41, so the entry cap does not bind
#: either. The true ceiling with both caps bound is 2 x 16 GiB = 32 GiB per run;
#: the EXPECTED footprint under the 41-blob bound is 12.19 GiB. Both numbers are
#: stated because the expected case is not the ceiling the cap permits.
_POLICY_STORE_MAX_BYTES = 17179869184
_REPLAY_STORE_MAX_BYTES = 17179869184

#: Refs eviction may never drop. Set once per train stage by `pin_refs`, the
#: only place a `state` is in scope, and capped at `_MAX_PINNED` = 3.
#:
#: NOT empty on the surrogate paths: `bird.train` pins unconditionally, so a
#: mock/tabular run pins `policy:bc_prior` plus whatever the state points at.
#: Harmless: a surrogate blob is three orders of magnitude under the BYTE cap
#: and never reaches it, and while the COUNT bound is reachable -- a
#: 6-iteration mock run ends at exactly `_STORE_LIMIT` entries -- three pinned
#: entries out of 64 leave eviction working.
#:
#: WHY A PIN AS WELL AS THE CAP ABOVE. The cap is sized from a measurement of
#: the blob, and a blob that outgrows it does so with a silent wrong answer as
#: the result: the incumbent is evicted, the next training cold starts, and
#: `train.init: warm_start_from_best` reads as configured on every side. A cap
#: cannot stop that recurring, because the next blob to outgrow it will do so
#: quietly; a pin is sized from the thing that READS the store, so it holds at
#: any cap and for any blob. Measured without the pin (512 MiB cap, 304 MiB
#: blob): a K=8 x 5-iteration search that should warm-start 32 of its 40
#: trainings managed at most 9, and about half of such searches managed 0 or 1.
_PINNED_REFS: set = set()


def _nbytes(value: Any) -> int:
    """Best-effort resident size of one store entry. Never raises."""
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, _ReplaySlice):
        return value.nbytes
    if isinstance(value, (list, tuple)):
        # A list of (obs, act, next_obs, done). 64 B/row is the CPython object
        # overhead the arrays themselves do not account for.
        total = 64 * len(value)
        for row in value[:1] + value[-1:]:
            per = sum(int(x.nbytes) for x in row if isinstance(x, np.ndarray))
            total += per * len(value) // 2
        return int(total)
    return 0


def _evictable(store: Dict[str, Any], key: str) -> Optional[str]:
    """The oldest entry eviction is allowed to take, or None.

    Oldest-first, skipping (a) the entry just written and (b) anything in
    `_PINNED_REFS` -- which is the THREE refs an initialisation loads, not
    everything the `RunState` reaches: pinning the archive too would switch
    both store bounds off (`pin_refs` carries the argument). Dropping a
    pinned entry leaves the ref alive in the state and the blob gone, which is
    a silent cold start rather than an error.
    """
    for candidate_key in store:
        if candidate_key == key or candidate_key in _PINNED_REFS:
            continue
        return candidate_key
    return None


def _store(store: Dict[str, Any], key: str, value: Any,
           max_bytes: Optional[int] = None) -> str:
    store[key] = value
    while len(store) > _STORE_LIMIT:
        victim = _evictable(store, key)
        if victim is None:
            break
        store.pop(victim)
    if max_bytes:
        used = sum(_nbytes(v) for v in store.values())
        while used > max_bytes and len(store) > 1:
            victim = _evictable(store, key)
            if victim is None:
                # Everything left is reachable or is the write itself. Going
                # over the cap is the lesser evil: the alternative is to drop a
                # blob the state still points at, which reads as a cold start
                # rather than as memory pressure. LOUD, because at 16 GiB this
                # should be unreachable -- if it fires, the blob outgrew the cap.
                log.warning("store: %d entries over the %d MB cap are all pinned "
                            "or current; keeping them", len(store), max_bytes // 2**20)
                break
            used -= _nbytes(store.pop(victim))
            log.debug("store: evicted %s to stay under %d MB", victim, max_bytes // 2**20)
    return key


#: The most entries `pin_refs` may ever protect. Three, and the number is the
#: whole argument: incumbent policy, incumbent replay slice, BC prior. See the
#: docstring for why this is NOT `checkpoint.reachable_refs`.
_MAX_PINNED = 3


def pin_refs(state: Any) -> None:
    """Pin the refs THIS stage's initialisation will dereference. At most three.

    Called from `bird.train` before any candidate of the round is stored.
    REPLACES the set rather than adding to it, so a ref the state has dropped
    stops being protected the moment it is unreachable.

    NOT `checkpoint.reachable_refs`, and that distinction is the whole safety
    argument. `reachable_refs` walks the ARCHIVE as well, which is bounded by
    config and not by `_STORE_LIMIT`: on a LIMEN- or REvolve-shaped state a
    75-cell archive yields 77 refs, the store grows past its 64-entry limit,
    and a byte cap of any size evicts nothing because every entry is pinned.
    That would turn this guard into the memory leak the store's own header
    warns about ("a 30-iteration LIMEN run would otherwise pin every policy it
    ever trained").

    So the pin names only what an initialisation actually loads:
    `warm_start_from_best` reads `state.policy_ref`, `secondary_replay_buffer`
    reads `state.replay_ref`, and `bc_prior` reads `BC_PRIOR_REF`. The two
    per-candidate modes (`warm_start_from_parent`, `warm_start_from_similar`)
    are deliberately NOT pinned: their pool is every stored policy, so pinning
    it is the unbounded case again, and both already declare a fallback
    (`best`, then scratch) that `init_source` records.
    """
    global _PINNED_REFS
    pins = {BC_PRIOR_REF}
    try:
        for attr in ("policy_ref", "replay_ref"):
            ref = getattr(state, attr, None)
            if ref:
                pins.add(ref)
    except Exception:  # bookkeeping never fails a run
        log.warning("pin_refs: could not read the state's refs; "
                    "eviction is unguarded for this stage")
    assert len(pins) <= _MAX_PINNED, pins
    _PINNED_REFS = pins


def reset_policy_stores() -> None:
    """Empty the three module-level stores. Called by `bird.py` at the start of
    every restart (`loop.n_restarts`) and of every fresh `run()`.

    `_POLICY_STORE`, `_POLICY_CODE` and `_REPLAY_STORE` are process globals. Left
    uncleared, restart r+1's `warm_start_from_similar` would find restart r's
    policies in the pool (keyed on iteration alone) and warm-start from another
    run of the loop, so `loop.n_restarts`' "independent runs of the whole loop"
    would not be independent. A fresh `RunState` already drops every ref INTO the
    stores (`policy_ref`, `replay_ref`), so nothing a restart may legitimately
    read is lost; `ensure_bc_prior` rebuilds `BC_PRIOR_REF` at every stage-3
    entry. `_ROUND_REPLAY` is round-scoped and popped by its driver; not touched.
    """
    _POLICY_STORE.clear()
    _POLICY_CODE.clear()
    _REPLAY_STORE.clear()
    _PINNED_REFS.clear()


def _remember_code(candidate: Candidate) -> None:
    """Record the reward behind `policy:<cand_id>`; forget the evicted ones."""
    _POLICY_CODE[candidate.cand_id] = (int(candidate.iteration), candidate.reward_code or "")
    for cid in [c for c in _POLICY_CODE if f"policy:{c}" not in _POLICY_STORE]:
        del _POLICY_CODE[cid]


class _PolicyBlob(np.ndarray):
    """A serialised SB3 model, worn as a 1-D uint8 array.

    Not cleverness for its own sake: it is what makes ONE store hold both a
    Q-table and a torch policy without a type switch at every reader.
    `checkpoint.encode` spills any ndarray over `INLINE_ARRAY_MAX` to
    `arrays/<sha256>.npy` and verifies dtype, shape and hash on the way back
    in; a bare `bytes` would have gone inline as ~1.9 MB of base64 inside the
    checkpoint JSON with no integrity check on the part that matters.
    """

    def __new__(cls, blob: bytes) -> "_PolicyBlob":
        return np.frombuffer(bytes(blob), dtype=np.uint8).view(cls)

    def tobytes(self, order: str = "C") -> bytes:  # noqa: D102 - ndarray API
        return np.asarray(self, dtype=np.uint8).tobytes(order)


@dataclasses.dataclass
class _ReplaySlice:
    """Columnar `(obs, action, next_obs, done)`, sequence-compatible.

    Indexing yields the same 4-tuple `_QLearner.seen` holds, so
    `_QLearner._replay_secondary` and `checkpoint._encode_replay` read it
    without knowing which backend wrote it. Stored columnar because the sb3
    path holds 100,000 rows: as a Python list of tuples that is ~80 MB and
    ~1.5 s to pickle across the fork, and as four arrays it is ~33 MB and
    ~10 ms.
    """

    obs: np.ndarray        # (n, obs_dim)   float32
    act: np.ndarray        # (n,) int64  |  (n, act_dim) float32
    next_obs: np.ndarray   # (n, obs_dim)   float32
    done: np.ndarray       # (n,)           bool

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return _ReplaySlice(self.obs[i], self.act[i], self.next_obs[i], self.done[i])
        act = self.act[i]
        return (self.obs[i], int(act) if np.ndim(act) == 0 else act,
                self.next_obs[i], bool(self.done[i]))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def nbytes(self) -> int:
        return int(self.obs.nbytes + self.act.nbytes + self.next_obs.nbytes + self.done.nbytes)


#: ROUND-SCOPED replay hand-off for `train.interaction: shared_population`
#: (LaRes, NeurIPS 2025 §4.3). A THIRD store, and deliberately not a third
#: kind of entry in `_REPLAY_STORE`:
#:
#:   * `_REPLAY_STORE` is the CROSS-ITERATION carry (`loop.carry: replay_buffer`,
#:     GT's secondary buffer, LaRes Eq. 3's moments) and is byte-capped,
#:     oldest-first. The buffers LaRes hands from one slice of a round to the
#:     next are the mechanism itself, and a cap that silently evicted an arm's
#:     buffer mid-round would turn `shared_buffer: true` back into the "SAC
#:     restarted from an empty buffer five times per candidate" this store
#:     exists to end -- with every counter reading normal.
#:   * It is bounded by construction instead: the driver
#:     (`population.interaction_shared_population`) writes at most one pool (or
#:     one per-arm buffer) of `_replay_capacity(cfg)` rows before a wave, pops
#:     each slice's export after it, and clears the dict at round close.
#:
#: Keys: `pool` / `own:<cand_id>` are PREFILL entries the parent writes before
#: a wave and a slice reads (a forked worker inherits them by copy-on-write);
#: `new:<cand_id>:<wave>` are EXPORT entries a slice writes with the RAW
#: transitions it added -- no rewards, for the reason `_sb3_export_replay`
#: gives -- which cross the fork in the child payload (`_CHILD_PAYLOAD`
#: "round_replay") and the parent folds into its groups. Nothing here is
#: checkpointed: a resume is per iteration and a round is redone whole.
_ROUND_REPLAY: Dict[str, Any] = {}

#: Seed stride between consecutive SLICES of one arm in a round. `_seed_base`
#: is per (run, restart, iteration, candidate), and a candidate can be trained
#: several times per iteration, so without a stride every slice of an arm under
#: `shared_population` would re-seed SB3 identically: the same `_Gym` episode
#: stream from episode 0, the same warm-up actions, the same torch draws --
#: a resumed policy replaying its predecessor's environment. Prime, and
#: co-prime with `_Gym.reset`'s 7907 episode stride and the 7919 seed stride,
#: so slice k's episode e and slice k+1's episode e' cannot coincide for any
#: e < 1299709. Slice 0 adds nothing, so a config that never slices seeds
#: exactly as `_seed_base` alone would.
_SLICE_SEED_STRIDE = 1_299_709


@dataclasses.dataclass(frozen=True)
class SliceHandoff:
    """What `train.interaction: shared_population` hands a backend for ONE slice.

    Not a config key and never in a config, like `resume_ref`, which carries
    the policy half of the same hand-off and stays a separate argument because
    `_training_init` already reads it. This is the replay half plus the slice
    index:

      index        0 for the arm's first slice of the round; salts the seed
      prefill_ref  `_ROUND_REPLAY` key whose RAW rows the learner's replay
                   buffer is pre-filled with, relabelled under this
                   candidate's own reward, before it learns; None = nothing
                   to restore (first slice, or nothing was exported yet)
      prefill_own  how many of those rows this arm collected itself -- the
                   parent knows (it built the pool), the backend records it
                   beside the pooled remainder so the artifact says whether
                   the buffer restored the arm's own history or others' too
      export_ref   `_ROUND_REPLAY` key to write the slice's NEW raw
                   transitions to; None = do not export
    """

    index: int
    prefill_ref: Optional[str] = None
    prefill_own: int = 0
    export_ref: Optional[str] = None


def _replay_capacity(cfg: Any) -> int:
    """Rows the round pool (and a resumed learner's buffer) can hold.

    The learner's OWN replay capacity, read off the same key that sizes it:
    `train.hyperparameters.buffer_size` when set, else SB3's SAC/TD3 default of
    1,000,000 -- which is also LaRes's release default (`arguments.py:19`,
    `--buffer-size 1e6`) ‡ the paper never states it. Not
    `train.secondary_buffer.size`: that is GT's pin for how much of the previous
    winner's buffer the next round may READ, and a LaRes pool truncated to it
    would silently be a 100k ring wearing the paper's 1M.
    """
    hyper = cfg.get("train.hyperparameters") or {}
    try:
        cap = int(hyper.get("buffer_size") or 0)
    except (TypeError, ValueError):
        cap = 0
    return cap if cap > 0 else 1_000_000


def _wants_replay(cfg: Any) -> bool:
    """Will anything ever dereference a `replay_ref` under this config?

    `_training_init` reads exactly one place -- `state.replay_ref` -- and
    `RunState.apply_carry` nulls that slot after every iteration unless
    `replay_buffer` is in `loop.carry`. So without the carry the stored buffer
    is unreachable by construction, and on `sb3` it costs 31 MB per candidate
    (measured on MT10, obs 39, `secondary_buffer.size: 100000`): a config
    without the carry would pay 31 MB x 8 candidates x 5 iterations for
    something nothing can read.

    Applied to BOTH backends rather than only the expensive one, because two
    backends that disagree about when a ref exists would be a difference
    between backends that no config asked for. It is unobservable elsewhere: the only
    other reader of `result.replay_ref` is `checkpoint.reachable_refs`, which
    exists to persist what a resumed run could dereference -- i.e. the same
    set.
    """
    return "replay_buffer" in (cfg.get("loop.carry") or [])


def _replay_tail(buf: Any, cap: int) -> Any:
    """The last `cap` transitions of a buffer of either representation.

    A plain `list(buf)[-cap:]` would turn 33 MB of arrays into 80 MB of tuples
    on the read path of every candidate.
    """
    if buf is None:
        return []
    if isinstance(buf, _ReplaySlice):
        return buf[max(0, len(buf) - cap):]
    return list(buf)[-cap:]


def _replay_columns(buf: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """(obs, act, next_obs, done) as arrays, from either representation."""
    if isinstance(buf, _ReplaySlice):
        return buf.obs, buf.act, buf.next_obs, buf.done
    rows = list(buf or ())
    if not rows:
        return None
    try:
        obs = np.asarray([np.asarray(r[0], dtype=np.float32) for r in rows], dtype=np.float32)
        nxt = np.asarray([np.asarray(r[2], dtype=np.float32) for r in rows], dtype=np.float32)
        act = np.asarray([r[1] for r in rows])
        done = np.asarray([bool(r[3]) for r in rows], dtype=bool)
    except Exception:  # noqa: BLE001 - a ragged buffer is unusable, not fatal
        return None
    if obs.dtype == object or nxt.dtype == object:
        return None
    return obs, act, nxt, done


# ==========================================================================
# Rollouts
# ==========================================================================


def _n_replayed_rollouts(cfg: Any) -> int:
    """How many of a candidate's rollouts something downstream will RE-RENDER.

    Rollout 0 always: `observability.record_rollouts` renders it and
    `preferences._clip_for` cuts it. Every scored rollout as well when the
    fitness source is the VLM one, because `evaluation._vlm_frames` samples
    frames from each of the `evaluate.rollouts_per_candidate` trajectories it
    scores -- and stage 4 runs after every candidate in the iteration has
    trained, by which point on `MetaWorld` only the retained rows survive
    (rollouts 1..K-1 of candidate 0 are long evicted by candidate 7's training).
    An evicted row does not raise a visible error: `_render_frames` returns
    `[]`, the judgment is made from a text digest, and the fitness looks
    identical. That is the failure `budget.blind_judgments` counts, and this is
    the cheap way to not cause it.

    Read off a config VALUE and not a method name. It is deliberately not
    "always all of them": `configs/methods/limen.yaml` pins
    `evaluate.rollouts_per_candidate: 50`, which on a 500-step Meta-World
    horizon is 25,050 rows per
    candidate against a 32,768-entry pinned store, so retaining all of them for
    a method that never re-renders one would FIFO-drop the rollout-0 rows that
    are the only thing limen's videos need.
    """
    if cfg.get("evaluate.fitness.source") != "vlm_score":
        return 1
    if "videos" not in (cfg.get("evaluate.artifacts") or []):
        return 1
    return max(1, int(cfg.get("evaluate.rollouts_per_candidate", 3) or 1))


def _retain_replay_states(env: Any, traj: Any) -> None:
    """Ask the adapter to keep a rollout something downstream will replay.

    Which ones is `_n_replayed_rollouts`' decision; this function is only the
    handoff to the adapter.

    Called the moment the trajectory exists rather than after the rollout loop,
    because on `MetaWorld` the rollouts THEMSELVES are enough to evict it:
    LIMEN's `evaluate.rollouts_per_candidate: 50` x a 500-step horizon = 25,050
    rows into an 8,192-entry LRU, so without this it records zero videos on
    every iteration even with one candidate and nothing else running.

    Silent no-op on every adapter but `MetaWorld` (see
    `EnvAdapter.retain_states`), and it never raises: a video is not worth a
    training run.
    """
    retain = getattr(env, "retain_states", None)
    if retain is None or traj is None or traj.states is None:
        return
    with contextlib.suppress(Exception):
        retain([traj.states])


def _n_store_rollouts(cfg: Any) -> int:
    """How many labelled rollouts to collect into the TPE trajectory store.

    `verify.tpe.trajectories_per_iteration` is the COLLECTION count -- CARD's
    App. B.2 Table 8 "# of trajectories collected per iteration = 100", the
    release's `--trajectory_num` check pool, collected in a DEDICATED env
    (metaworld_exp_one_step.py:230) distinct from the `--n_eval_episodes`
    feedback pool (:232). One BIRD key cannot size both, which is why this is
    decoupled from `evaluate.rollouts_per_candidate`: a store grown by the
    3-rollout feedback pool would hold 1/33 of the published evidence.

    Read off config VALUES, never a method name: collection only happens when
    something will store it (`update.memory.trajectory_store != none`), so the
    default 100 is inert on every config without a trajectory store and no
    hash-neutral run changes. 0 is the explicit opt-out: the store then grows
    from the evaluation rollouts.
    """
    if cfg.get("update.memory.trajectory_store", "none") in (None, "none"):
        return 0
    return max(0, int(cfg.get("verify.tpe.trajectories_per_iteration", 0) or 0))


def _rollout(env: Any, policy: Callable[[np.ndarray], np.ndarray], rng: np.random.Generator,
             reward: Optional[CompiledReward] = None,
             on_error: Optional[Callable[[BaseException], float]] = None
             ) -> Tuple[Trajectory, int, Optional[float]]:
    """One episode. Returns `(trajectory, steps, gt_return)`.

    `states` carries T+1 rows and `actions` T, per the convention in
    `bird.envs.toy` -- screens re-score these as (s, a, s') triples and the
    terminal state is where a goal bonus lives.

    `gt_return` is the REFERENCE reward summed over the episode, or **None when
    the env has none** -- `env.reference_reward` raising `NotImplementedError`
    is the adapter contract for "this task disowns the built-in reward"
    (`gym_half_cheetah_backward` and its four siblings, `EnvAdapter.
    has_reference_reward`). None, never 0.0: the sum feeds `gt_reward_curve`,
    and from there `pearson_curve` and the scripted human oracle
    (`phases._reference_score`), both of which treat an absent channel and a
    flat zero differently -- a zero curve would correlate with nothing and be
    reported as r=None anyway, but a zero would read as "the reference paid
    nothing", which is a claim about the policy, not about the task. An adapter
    that returned the base simulator's forward reward on a derived task instead
    of raising would silently report the wrong reference, which is why the
    disowning is explicit.
    """
    s = env.reset(rng)
    states: List[np.ndarray] = [np.asarray(s, dtype=float)]
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    #: One dict per step -- `{}` on a step where the reward claimed nothing --
    #: so `component_values` below can be packed BY STEP. Appending each
    #: component only when emitted, while `rewards` grows every step, would let
    #: a key the reward sets on some steps only (`if dist < 0.05:
    #: components["success_bonus"] = 10.0`), or any step a swallowing
    #: `on_error` handled, leave `component_values[k][t]` describing a LATER
    #: step than `rewards[t]` -- and every reader that pairs the two by index
    #: (the trajectory trace, the RDA judge table, CARD's reflection rows) would
    #: show a claim at the wrong instant, which is the one shape those artifacts
    #: exist to detect.
    per_step: List[Dict[str, float]] = []
    gt_return: Optional[float] = 0.0
    steps = 0

    for _ in range(int(env.horizon)):
        a = np.asarray(policy(s), dtype=float).ravel()
        s2, done, _info = env.step(s, a)
        steps += 1
        if reward is not None:
            cv: Dict[str, float] = {}
            try:
                r, cv = reward(s, a, s2)
                if not math.isfinite(r):
                    raise FloatingPointError(f"reward returned {r!r}")
            except Exception as exc:  # noqa: BLE001 -- routed by failure_detection
                if on_error is None:
                    raise
                # `cv` keeps whatever the reward computed before its total came
                # out non-finite -- the one step whose claim the trace most needs
                # -- and stays `{}` for a reward that raised and claimed nothing.
                r = on_error(exc)
            rewards.append(r)
            per_step.append(cv)
        if gt_return is not None:
            try:
                gt_return += float(env.reference_reward(s2, a))
            except NotImplementedError:
                gt_return = None      # no reference on this task; stays None
        states.append(np.asarray(s2, dtype=float))
        actions.append(a)
        s = s2
        if done:
            break

    # Packed per step: `component_values[k][t]` is what the reward claimed for
    # `k` at step `t`, `None` where it claimed nothing there. Keys in first-
    # emission order; a key claimed on every step is a dense list.
    names: List[str] = []
    for cv in per_step:
        for k in cv:
            if k not in names:
                names.append(k)
    traj = Trajectory(
        states=np.asarray(states, dtype=float),
        actions=np.asarray(actions, dtype=float),
        rewards=rewards,
        component_values={k: [cv.get(k) for cv in per_step] for k in names},
        length=len(actions),
        ret=float(sum(rewards)),
    )
    traj.success = bool(env.success(traj))
    return traj, steps, gt_return


def _greedy_policy(env: Any, q: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    action_set = env.action_set

    def policy(s: np.ndarray) -> np.ndarray:
        return action_set[int(np.argmax(q[env.discretise(s)]))]

    return policy


def _linear_policy(env: Any, w: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """`a = clip(W [phi(s); 1])`, where phi is `env.policy_features`.

    Identity on every adapter (`EnvAdapter.policy_features`), so this is
    bit-for-bit the raw-state map it has always been -- and the ONE line that
    installs LIMEN's co-designed observation into the linear-policy searcher,
    because `_ObsView` overrides that method. The closure takes a RAW state and
    features it internally, which is what lets `_evaluate_policy` and `_rollout`
    go on driving the unwrapped adapter.
    """
    lo, hi = env.action_low, env.action_high
    features = getattr(env, "policy_features", None)
    if features is None:  # an env object predating `policy_features`
        def features(s: Any) -> np.ndarray:
            return np.asarray(s, dtype=float)

    def policy(s: np.ndarray) -> np.ndarray:
        return np.clip(w @ np.append(features(s), 1.0), lo, hi)

    return policy


# ==========================================================================
# The surrogate learners
# ==========================================================================
#
# Two policy classes, chosen by what the *environment* can support, never by
# what method is running: an exact table where the state space enumerates, a
# linear feedback policy where it does not. `train.architecture` (`mlp`,
# `simba_v2`, ...) is recorded but not honoured -- there is no network here,
# and a declared key the algorithm silently ignores is a fabricated pin. That
# is why the surrogate says so in `seed_metrics`.


class _Learner:
    """Common shape: `.improve(n_steps)` then `.policy()` and `.snapshot()`.

    `steps_per_round()` exists so the driver can size the checkpoint schedule
    in *rounds* the learner actually costs. Without it a config's
    `checkpoint_interval` would mean something different per learner, and the
    curve §4 aggregates over would silently change length with the backend.
    """

    kind = "abstract"

    def policy_input_dim(self) -> Optional[int]:
        """The width this learner's policy ACTUALLY consumes, as constructed.

        Read off the env object the learner was HANDED, which is the only place
        that can answer it: under `problem.search_space` with `observation` that
        object is an `_ObsView` and this is phi's width, and otherwise it is the
        adapter's own `obs_dim`.

        It exists because the obvious way to record the install is to record
        phi's `dim`, and phi's `dim` is measured before the learner is built --
        so it goes on reading 3 after a refactor hands `_make_learner` the raw
        adapter, and the field then describes an install that reached nothing.
        Verified by mutation: passing `env` instead of `learn_env` to
        `_make_learner` leaves every phi-derived assertion green.
        """
        env = getattr(self, "env", None)
        dim = getattr(env, "obs_dim", None)
        return int(dim) if isinstance(dim, (int, float)) else None

    def steps_per_round(self) -> int:
        raise NotImplementedError

    def improve(self, budget: int) -> int:
        raise NotImplementedError

    def policy(self) -> Callable[[np.ndarray], np.ndarray]:
        raise NotImplementedError

    def snapshot(self) -> Optional[np.ndarray]:
        return None

    def restore(self, snapshot: Optional[np.ndarray]) -> bool:
        """`train.checkpoint_selection`: put an earlier `snapshot()` back, IN PLACE.

        In place, so a policy closure already built over the parameter array
        (`_greedy_policy`, `_linear_policy`) sees the restored values rather
        than the array it was handed. False when this learner has nothing to
        restore (the planner keeps no parameters) or the shapes disagree; the
        caller records that in the seed row and ships the last checkpoint.
        """
        return False

    def diverged(self) -> bool:
        return False


class _QLearner(_Learner):
    """Episodic Q-learning with backward sweeps, replay, and a decaying step size.

    All three pieces earn their place, and the third is not optional:

    * **Backward sweeps** replay the finished episode in reverse, so a terminal
      value propagates the whole way back in one pass instead of one step per
      episode. That is what makes a few thousand steps enough.
    * **Replay** samples decorrelated past transitions. Sweeping the same
      episode repeatedly at a high step size overfits one sampled path through a
      *stochastic* env, and hungry-thirsty is stochastic.
    * **A decaying step size.** Without it the greedy policy thrashes between
      checkpoints -- the curve reads 0.66, 0.13, 0.50, 0.00 -- and every §4
      aggregation then measures learner instability rather than reward quality.
      That is not a cosmetic problem: it is the difference between a fitness
      signal and a random number generator.
    """

    kind = "tabular_q"

    def __init__(self, env: Any, reward: CompiledReward, rng: np.random.Generator,
                 norm: _RewardNorm, on_error: Callable[[BaseException], float],
                 alpha: float = 0.4, alpha_min: float = 0.02, gamma: float = 0.97,
                 sweeps: int = 2, replay_k: int = 80,
                 eps_decay: float = 0.5, epsilon: Optional[float] = None,
                 q_init_low: float = -1e-3, q_init_high: float = 1e-3,
                 total_budget: int = 6000,
                 init_q: Optional[np.ndarray] = None, fusion_alpha: Optional[float] = None,
                 secondary: Optional[Sequence[Any]] = None, secondary_ratio: float = 0.0,
                 n_parallel: int = 1) -> None:
        self.env, self.reward, self.rng, self.norm = env, reward, rng, norm
        self.on_error = on_error
        self.alpha0, self.alpha_min = alpha, alpha_min
        self.alpha = alpha
        self.gamma, self.sweeps, self.replay_k = gamma, sweeps, replay_k
        self.eps_decay, self.total = eps_decay, max(1, total_budget)
        #: `None` = anneal 1.0 -> 0.05 over `eps_decay` of the budget, which is
        #: what every LLM-era point wants. A float PINS epsilon for the whole run,
        #: which is Singh 2009's rule: his learner is flatly epsilon-greedy and
        #: `epsilon` is one of the two axes he maximises fitness over (p. 2603).
        #: Annealing would make that axis meaningless, since every setting would
        #: converge to the same 0.05 tail.
        self.epsilon = None if epsilon is None else float(epsilon)
        self.n_parallel = max(1, int(n_parallel))
        shape = (int(env.n_disc_states), int(env.n_actions))
        #: Singh 2009 p. 2603 initialises Q uniformly from [-0.001, 0.001], which
        #: is this default. Settable because it is a pin his config cites, and a
        #: cited value the learner ignores is a fabricated pin.
        self.q = rng.uniform(float(q_init_low), float(q_init_high), shape)
        #: Did `train.init: warm_start_from_best` actually take? A mismatched
        #: snapshot is dropped in silence below, which reads exactly like the
        #: ablation; `seed_metrics[*].warm_started_from` reports this flag.
        self.init_applied = init_q is not None and np.shape(init_q) == shape
        if self.init_applied:
            # `fusion_alpha is None` is EVERY config but ROSKA's, and it takes the
            # copy verbatim rather than `1.0*init + 0.0*theta0` -- the arithmetic
            # would agree to the bit for finite values and not for -0.0 or a
            # non-finite theta_0, and `tests/test_parallelism.py` compares whole
            # run directories. ROSKA fuses instead (methodology.tex:69):
            #   theta_f = alpha*theta_best + (1-alpha)*theta_0
            # where theta_0 is the fresh draw already in self.q.
            if fusion_alpha is None:
                self.q = np.array(init_q, dtype=float, copy=True)
            else:
                a = float(fusion_alpha)
                self.q = a * np.asarray(init_q, dtype=float) + (1.0 - a) * self.q
                self.fusion_alpha = a
        # `list()` on a `_ReplaySlice` would materialise 100,000 tuples for a
        # buffer that is only ever indexed; keep the columnar form, which
        # indexes to the same 4-tuple.
        self.secondary = secondary if isinstance(secondary, _ReplaySlice) \
            else list(secondary or ())
        self.secondary_ratio = float(secondary_ratio)
        #: raw transitions, for GT's *shared* secondary buffer -- no reward is
        #: stored, because the next round's candidates must relabel with theirs.
        self.seen: List[Tuple[np.ndarray, int, np.ndarray, bool]] = []
        #: this run's own replay, already discretised and already rewarded.
        self.buf: List[Tuple[int, int, float, int, bool]] = []
        self.done_steps = 0
        self._diverged = False

    def steps_per_round(self) -> int:
        return self.n_parallel * int(self.env.horizon)

    def _update(self, si: int, ai: int, r: float, s2i: int, done: bool) -> None:
        target = r if done else r + self.gamma * float(self.q[s2i].max())
        self.q[si, ai] += self.alpha * (target - self.q[si, ai])

    def _replay(self) -> None:
        if not self.buf:
            return
        for _ in range(self.replay_k):
            self._update(*self.buf[int(self.rng.integers(len(self.buf)))])

    def improve(self, budget: int) -> int:
        env, rng = self.env, self.rng
        spent = 0
        for _ in range(self.n_parallel):
            frac = min(1.0, self.done_steps / self.total)
            eps = (self.epsilon if self.epsilon is not None
                   else 1.0 + (0.05 - 1.0) * min(1.0, frac / self.eps_decay))
            self.alpha = self.alpha0 + (self.alpha_min - self.alpha0) * frac
            s = env.reset(rng)
            si = env.discretise(s)
            tape: List[Tuple[int, int, float, int, bool]] = []
            for _t in range(int(env.horizon)):
                ai = (int(rng.integers(env.n_actions)) if rng.random() < eps
                      else int(np.argmax(self.q[si])))
                a = env.action_set[ai]
                s2, done, _info = env.step(s, a)
                try:
                    raw, _cv = self.reward(s, a, s2)
                    if not math.isfinite(raw):
                        raise FloatingPointError(f"reward returned {raw!r}")
                except Exception as exc:  # noqa: BLE001
                    raw = self.on_error(exc)
                r = self.norm(raw)
                s2i = env.discretise(s2)
                self._update(si, ai, r, s2i, done)
                tape.append((si, ai, r, s2i, done))
                if len(self.buf) < 4096:
                    self.buf.append((si, ai, r, s2i, done))
                if len(self.seen) < 4096:
                    self.seen.append((np.asarray(s, dtype=float), ai,
                                      np.asarray(s2, dtype=float), bool(done)))
                si, s = s2i, s2
                spent += 1
                self.done_steps += 1
                if done or spent >= budget:
                    break
            for _ in range(self.sweeps):
                for (si_, ai_, r_, s2i_, d_) in reversed(tape):
                    self._update(si_, ai_, r_, s2i_, d_)
            self._replay()
            self._replay_secondary()
            if not np.all(np.isfinite(self.q)) or float(np.abs(self.q).max()) > _DIVERGENCE:
                self._diverged = True
                return spent
            if spent >= budget:
                break
        return spent

    def _replay_secondary(self) -> None:
        """GT's shared secondary buffer (`train.init: secondary_replay_buffer`).

        The buffer stores raw transitions, never rewards: the previous winner's
        experience is re-labelled with *this* candidate's reward before it
        updates anything. Storing rewards would leak the old candidate's reward
        into the new one's fitness, which is the one thing the shared buffer
        must not do.
        """
        if not self.secondary or self.secondary_ratio <= 0.0:
            return
        n = max(1, int(round(self.secondary_ratio * self.env.horizon)))
        for _ in range(n):
            s, ai, s2, done = self.secondary[int(self.rng.integers(len(self.secondary)))]
            a = self.env.action_set[ai]
            try:
                raw, _cv = self.reward(s, a, s2)
                if not math.isfinite(raw):
                    raise FloatingPointError(f"reward returned {raw!r}")
            except Exception as exc:  # noqa: BLE001
                raw = self.on_error(exc)
            self._update(self.env.discretise(s), ai, self.norm(raw),
                         self.env.discretise(s2), done)

    def policy(self) -> Callable[[np.ndarray], np.ndarray]:
        return _greedy_policy(self.env, self.q)

    def snapshot(self) -> Optional[np.ndarray]:
        return self.q

    def restore(self, snapshot: Optional[np.ndarray]) -> bool:
        if snapshot is None or np.shape(snapshot) != np.shape(self.q):
            return False
        np.copyto(self.q, np.asarray(snapshot, dtype=self.q.dtype))
        return True

    def diverged(self) -> bool:
        return self._diverged


class _CEMLearner(_Learner):
    """Cross-entropy method over a linear feedback policy `a = clip(W [s; 1])`.

    Used where the state space does not enumerate. A linear map over the raw
    observation already contains the PD controller a continuous reaching task
    needs, so the search is over ~14 parameters and converges in a few dozen
    iterations -- which is what a per-candidate budget of a few thousand env
    steps actually buys. A tabular learner on the same env needs an order of
    magnitude more steps to fill its table, and returns noise instead of a
    fitness if it does not get them.
    """

    kind = "cem_linear"

    def __init__(self, env: Any, reward: CompiledReward, rng: np.random.Generator,
                 norm: _RewardNorm, on_error: Callable[[BaseException], float],
                 pop: int = 10, elite: int = 3, sigma0: float = 1.5, floor: float = 0.5,
                 total_budget: int = 6000, init_w: Optional[np.ndarray] = None,
                 fusion_alpha: Optional[float] = None,
                 n_parallel: int = 1) -> None:
        self.env, self.reward, self.rng, self.norm = env, reward, rng, norm
        self.on_error = on_error
        self.pop, self.elite = max(4, int(pop)), max(2, int(elite))
        self.floor = floor
        self.n_parallel = max(1, int(n_parallel))
        self.dim = (int(env.action_dim), int(env.obs_dim) + 1)
        self.mu = np.zeros(self.dim)
        #: See `_QLearner.init_applied`.
        self.init_applied = init_w is not None and np.shape(init_w) == self.dim
        if self.init_applied:
            # See `_QLearner`: verbatim copy unless a fusion ratio was asked for.
            if fusion_alpha is None:
                self.mu = np.array(init_w, dtype=float, copy=True)
            else:
                a = float(fusion_alpha)
                self.mu = a * np.asarray(init_w, dtype=float) + (1.0 - a) * self.mu
                self.fusion_alpha = a
        self.sigma = np.full(self.dim, float(sigma0))
        self.total_iters = max(4, max(1, total_budget) // (self.pop * max(1, int(env.horizon))))
        self.it = 0
        self.seen: List[Tuple[np.ndarray, int, np.ndarray, bool]] = []
        self._diverged = False

    def steps_per_round(self) -> int:
        return self.n_parallel * self.pop * int(self.env.horizon)

    def _score(self, w: np.ndarray, seed: int) -> Tuple[float, int]:
        env = self.env
        rng = np.random.default_rng(seed)
        s = env.reset(rng)
        policy = _linear_policy(env, w)
        total = 0.0
        steps = 0
        for _t in range(int(env.horizon)):
            a = policy(s)
            s2, done, _info = env.step(s, a)
            try:
                raw, _cv = self.reward(s, a, s2)
                if not math.isfinite(raw):
                    raise FloatingPointError(f"reward returned {raw!r}")
            except Exception as exc:  # noqa: BLE001
                raw = self.on_error(exc)
            total += self.norm(raw)
            if len(self.seen) < 4096:
                self.seen.append((np.asarray(s, dtype=float), 0,
                                  np.asarray(s2, dtype=float), bool(done)))
            s = s2
            steps += 1
            if done:
                break
        return total, steps

    def improve(self, budget: int) -> int:
        spent = 0
        for _ in range(self.n_parallel):
            seed = int(self.rng.integers(1 << 30))
            cands = [self.mu + self.sigma * self.rng.standard_normal(self.dim)
                     for _ in range(self.pop)]
            scores = []
            for w in cands:
                sc, st = self._score(w, seed)
                scores.append(sc)
                spent += st
            order = np.argsort(np.asarray(scores, dtype=float))[-self.elite:]
            elites = np.stack([cands[int(i)] for i in order])
            self.mu = elites.mean(axis=0)
            # Extra exploration noise, annealed: the elite spread alone
            # collapses long before the budget is spent.
            decay = max(0.0, 1.0 - self.it / max(self.total_iters - 1, 1))
            self.sigma = elites.std(axis=0) + self.floor * decay
            self.it += 1
            if not np.all(np.isfinite(self.mu)):
                self._diverged = True
                return spent
            if spent >= budget:
                break
        return spent

    def policy(self) -> Callable[[np.ndarray], np.ndarray]:
        return _linear_policy(self.env, self.mu)

    def snapshot(self) -> Optional[np.ndarray]:
        return self.mu

    def restore(self, snapshot: Optional[np.ndarray]) -> bool:
        if snapshot is None or np.shape(snapshot) != np.shape(self.mu):
            return False
        np.copyto(self.mu, np.asarray(snapshot, dtype=self.mu.dtype))
        return True

    def diverged(self) -> bool:
        return self._diverged


class _PlannerLearner(_Learner):
    """L2R's `train.algorithm: none`: a shooting planner, no policy learning.

    Random-shooting MPC -- sample action sequences, score them with the
    candidate reward under the model, commit the first action, re-plan. This is
    the honest analogue of L2R's MuJoCo MPC: it *solves* the reward online, so
    a bad reward produces bad behaviour immediately with no learning dynamics in
    between. `improve()` is a no-op by construction; the checkpoint curve that
    comes out is flat except for episode noise, which is the correct picture of
    a method that never learns.
    """

    kind = "mpc_shooting"

    def __init__(self, env: Any, reward: CompiledReward, rng: np.random.Generator,
                 norm: _RewardNorm, on_error: Callable[[BaseException], float],
                 n_samples: int = 8, lookahead: int = 3) -> None:
        self.env, self.reward, self.rng, self.norm = env, reward, rng, norm
        self.on_error = on_error
        self.n_samples, self.lookahead = int(n_samples), int(lookahead)
        self.plan_rng = np.random.default_rng(int(rng.integers(1 << 30)))
        self.sim_steps = 0
        self.seen: List[Tuple[np.ndarray, int, np.ndarray, bool]] = []

    def steps_per_round(self) -> int:
        # One planned episode: every real step costs a lookahead tree of model
        # queries, and those are env interactions too -- L2R's cost is not zero
        # just because nothing is learning.
        return int(self.env.horizon) * (1 + self.n_samples * self.lookahead)

    def improve(self, budget: int) -> int:
        """No policy improvement -- `train.algorithm: none` means what it says.

        One planned episode is still executed per round so the run consumes a
        real budget and terminates on it; nothing is retained between rounds,
        which is why the checkpoint curve comes out flat.
        """
        before = self.sim_steps
        _traj, real, _gt = _rollout(self.env, self.policy(), self.rng, self.reward,
                                    self.on_error)
        return real + (self.sim_steps - before)

    def policy(self) -> Callable[[np.ndarray], np.ndarray]:
        env = self.env

        def plan(s: np.ndarray) -> np.ndarray:
            best_score, best_first = -math.inf, env.action_set[0]
            # Model queries must not consume the *episode's* randomness, or the
            # executed trajectory stops being reproducible from its own seed.
            # Swapping the generator object leaves the episode one untouched.
            saved = env._rng  # noqa: SLF001 -- documented adapter contract
            env._rng = self.plan_rng  # noqa: SLF001
            try:
                for _ in range(self.n_samples):
                    idx = self.plan_rng.integers(env.n_actions, size=self.lookahead)
                    cur = np.asarray(s, dtype=float)
                    score = 0.0
                    for k in range(self.lookahead):
                        a = env.action_set[int(idx[k])]
                        nxt, done, _info = env.step(cur, a)
                        self.sim_steps += 1
                        try:
                            raw, _cv = self.reward(cur, a, nxt)
                            if not math.isfinite(raw):
                                raise FloatingPointError(f"reward returned {raw!r}")
                        except Exception as exc:  # noqa: BLE001
                            raw = self.on_error(exc)
                        score += self.norm(raw)
                        cur = nxt
                        if done:
                            break
                    if score > best_score:
                        best_score, best_first = score, env.action_set[int(idx[0])]
            finally:
                env._rng = saved  # noqa: SLF001
            return best_first

        return plan


# ==========================================================================
# Inner-loop early stopping -- `train.pruning` (the RULE) x `train.pruning_metric`
# ==========================================================================
#
# Nothing published prunes an ongoing run (LIMEN's cascade is the nearest
# relative and it lives in §2, gating admission rather than an ongoing run).
# These are within-run adaptations of the standard rules, so the axis is at
# least reachable; they read the checkpoint curve and nothing else.
#
# ONE mechanism, and the METRIC is the knob. A rule decides
# WHEN a training has stopped paying; `train.pruning_metric` decides WHAT it
# watches:
#
#   own_reward   the candidate's own return under greedy evaluation -- "has the
#                policy finished learning this reward". Legal for every method:
#                it reads the number being optimised and nothing about the task.
#                THE DEFAULT.
#   task_metric  `env.task_metric` -- ground truth. A validation-loss stop,
#                meaningful only for a method that already reads GT (Eureka);
#                `_check_coherence` refuses it under `problem.fitness_access:
#                none`, so a pruning override cannot hand GT to a method (the
#                RDA family) whose whole claim is that it has none.


def _prune_none(curve: Sequence[float], budget_frac: float) -> bool:
    return False


def _prune_median_stop(curve: Sequence[float], budget_frac: float) -> bool:
    """Vizier's median stopping rule, applied to one run's own history: stop if
    the latest checkpoint is worse than the median of the ones before it."""
    if len(curve) < 4:
        return False
    return float(curve[-1]) < float(np.median(np.asarray(curve[:-1], dtype=float)))


def _prune_plateau(curve: Sequence[float], budget_frac: float, *,
                   patience: int = 4, min_delta: float = 0.02,
                   min_checkpoints: int = 5) -> bool:
    """Stop when the last `patience` checkpoints beat nothing before them.

    "Improvement" is `min_delta` of the curve's own range so far, so the rule
    is scale-free: a reward paying in thousands and one paying in hundredths
    plateau on the same shape. `min_checkpoints` is the warm-up -- under
    `train.init: warm_start_from_best` the first checkpoint is already
    competent and the curve can be flat from step one, which is exactly when
    this fires earliest (that IS the saving, and `pruned_at_round` records it).
    Non-finite values count as no improvement rather than as infinity.

    Worked example, 20 checkpoints, patience 4, min_delta 0.02: returns
    0.10 0.30 0.45 0.52 0.55 | 0.55 0.56 0.55 0.56 -> range 0.46, tolerance
    0.0092, best-before 0.55, window max 0.56 > 0.559 -> keep going;
    0.55 0.55 0.54 0.55 -> window max 0.55 <= 0.559 -> stop at checkpoint 13.
    """
    n = len(curve)
    if n < max(int(min_checkpoints), int(patience) + 1):
        return False
    arr = np.nan_to_num(np.asarray(curve, dtype=float), nan=-np.inf,
                        posinf=-np.inf, neginf=-np.inf)
    finite = arr[np.isfinite(arr)]
    rng = float(finite.max() - finite.min()) if finite.size else 0.0
    before, window = arr[:-int(patience)], arr[-int(patience):]
    return float(window.max()) <= float(before.max()) + float(min_delta) * rng


def _prune_successive_halving(curve: Sequence[float], budget_frac: float) -> bool:
    """Successive halving is normally *across* candidates; within one run the
    only available comparison is the run against itself, so each rung demands
    strict improvement over the best score seen before that rung."""
    rungs = (0.125, 0.25, 0.5)
    if len(curve) < 3:
        return False
    if not any(abs(budget_frac - r) < 0.06 for r in rungs):
        return False
    best_before = float(np.max(np.asarray(curve[:-1], dtype=float)))
    return float(curve[-1]) <= best_before


_PRUNERS: Dict[str, Callable[[Sequence[float], float], bool]] = {
    "none": _prune_none,
    "median_stop": _prune_median_stop,
    "successive_halving": _prune_successive_halving,
    # Hyperband's brackets need a *population*; within one run it degenerates to
    # successive halving on a denser rung schedule. Present so the schema's
    # fourth value is not a dead config.
    "hyperband": _prune_successive_halving,
    "plateau": _prune_plateau,
    # ACROSS candidates, not within one: the rule lives in `rounds.py`, which
    # trains the round in rungs and drops the bottom 1/eta of the pool at each.
    # Inside a single training there is nothing for it to decide, so the
    # per-checkpoint hook is a no-op -- the entry exists so `_pruner_for`
    # resolves the name and a typo cannot fall through to ground truth.
    "successive_halving_pool": _prune_none,
}

#: Rules that read `train.pruning_cfg.*`. An attribute set, not a name test, so a
#: rule added later declares itself rather than being special-cased at the
#: lookup (the `backend_exhaustive_enumeration.needs_prompt` idiom).
_PRUNERS_WITH_CFG = frozenset({"plateau"})

#: `train.pruning_metric` -> the checkpoint-curve field the rule is handed.
#: Both fields are written at every checkpoint by both backends (`reward_return`
#: at the `point` dicts below); the table is closed so a typo cannot fall back
#: to ground truth by accident. `demo_fraction` is written only when
#: `_ceiling_wanted` (it is that metric, or the ceiling guard, that asks for it).
_PRUNING_FIELDS: Dict[str, str] = {
    "own_reward": "reward_return",
    "task_metric": "fitness",
    "demo_fraction": "demo_fraction",
}


# --------------------------------------------------------------------------
# The demonstration ceiling -- `train.pruning_cfg.ceiling: demo_return` and
# `train.pruning_metric: demo_fraction`. NOT PUBLISHED; a BIRD extension for
# training efficiency.
# --------------------------------------------------------------------------
#
# The measured problem (`plateau` on own reward, patience 4, Meta-World
# pick-place): trainings were cut at ~0.47M of 1.03M steps, and in every
# failed seed NO trained candidate reached task success -- the plateau the rule
# sees on pick-place is the reach-and-hover attractor, not a finished policy.
# Own reward alone cannot tell "finished learning" from "stuck on a plateau
# below the task": both are flat curves. What CAN, without ground truth, is
# what the reward would pay a policy that solves the task -- and the
# demonstration machinery already rolls
# exactly that out under each candidate's reward (`bird.demos`,
# `screens.screen_demo_margin`). The expert's return under a candidate is that
# candidate's own ceiling estimate; the random policy's return is its floor.
#
#   progress = (own_return - random_return) / (expert_return - random_return)
#
# is the share of the random->expert gap the training has closed under ITS OWN
# reward -- scale-free AND offset-free, which a plain own/expert ratio is not
# (a cost-shaped reward pays the expert -10 and random -500). It is comparable
# across candidates whose rewards pay in different units, which is what lets
# `successive_halving_pool` rank a pool on it. Same access class as the screen:
# `problem.fitness_access: demonstrations`, and nothing here reads
# `env.task_metric`, `success` or `reference_reward` (`_rollout`'s gt_return is
# discarded by `demos.rollouts_under`).
#
# The ceiling is UNINFORMATIVE when the expert earns no more than random --
# the reward orders the ladder wrong (the sign test the screen rejects on). No
# guard can be built on it, so the plain rule applies and the journal says so;
# for ranking, such a candidate falls below every informative one. It is
# UNAVAILABLE when the env has no expert at all (`demos._expert`'s three
# sources), which is a fact about the catalogue and is named, never invented.
#
# ONE SET OF HELPERS, THREE CALLERS (`_run_backend`, `_sb3_run`,
# `fasttd3.fasttd3_backend`), for `_pruner_for`'s reason: a guard that meant
# one thing on `mock` and another on `sb3` would make every tester run test a
# different method from the one a full-scale run executes.


def _ceiling_wanted(cfg: Any) -> bool:
    """Whether this config needs the expert/random returns under each candidate."""
    if str(cfg.get("train.pruning", "none") or "none") == "none":
        return False
    guard = str(cfg.get("train.pruning_cfg.ceiling", "none") or "none") == "demo_return"
    metric = str(cfg.get("train.pruning_metric", "own_reward") or "own_reward")
    return guard or metric == "demo_fraction"


def _demo_ceiling(ctx: Any, state: Any, candidate: Candidate, reward: Any) -> Dict[str, Any]:
    """Expert and random returns under `reward`, cached on `candidate.meta["ceiling"]`.

    Computed once per candidate (a rung continuation, a second seed and a grid
    point all reuse it; the meta travels home from a forked worker with the
    candidate). Seeds are the demo screen's own -- a function of (run seed,
    iteration, policy, episode) and of nothing else -- so a screened candidate's
    ceiling agrees with its screen numbers and `parallel` equals `sequential`.
    Episodes: `verify.demo_screen.episodes`, the same rollouts the verifier
    uses. Returns are episode totals (`Trajectory.ret`), the unit the curve's
    `reward_return` is in. Env steps are charged as rollouts, not trainings.
    """
    cached = candidate.meta.get("ceiling")
    if isinstance(cached, dict) and "status" in cached:
        return cached
    from .. import demos
    cfg = ctx.cfg
    env = ctx.env
    env_id = str(cfg.get("problem.env_id") or "")
    out: Dict[str, Any] = {"status": "unavailable", "informative": False,
                           "expert_return": None, "random_return": None,
                           "policies": [], "episodes": 0, "env_steps": 0,
                           "reward_errors": 0, "reason": ""}
    pols, why = demos.policy_set(env, env_id, ["expert", "random"])
    if not pols:
        out["reason"] = why
        candidate.meta["ceiling"] = out
        return out
    episodes = max(1, int(cfg.get("verify.demo_screen.episodes", 2) or 1))
    it = int(getattr(state, "iteration", 0) or 0)
    base_seed = int(cfg.get("seed", 0) or 0) * 1_000_003 + it * 1_009
    errors = {"n": 0}

    def on_error(exc: BaseException) -> float:  # noqa: ARG001
        errors["n"] += 1
        return 0.0

    by_quality: Dict[int, float] = {}
    steps = 0
    for j, pol in enumerate(pols):
        seeds = [base_seed + 97 * j + e for e in range(episodes)]
        trajs, n = demos.rollouts_under(env, reward, pol, seeds, on_error)
        steps += int(n)
        by_quality[pol.quality] = float(np.mean([float(t.ret) for t in trajs])) if trajs else 0.0
    if steps:
        ctx.budget.record_rollout_steps(steps)
    expert = by_quality[min(by_quality)]
    random = by_quality[max(by_quality)]
    gap = expert - random
    informative = bool(np.isfinite(gap) and gap > 1e-9 * max(1.0, abs(expert) + abs(random)))
    out.update({"status": "informative" if informative else "uninformative",
                "informative": informative, "expert_return": expert, "random_return": random,
                "policies": [p.name for p in pols], "episodes": episodes, "env_steps": steps,
                "reward_errors": int(errors["n"]),
                "reason": "" if informative else "expert paid no more than random"})
    candidate.meta["ceiling"] = out
    return out


def _ceiling_progress(own: float, ceiling: Optional[Dict[str, Any]]) -> Optional[float]:
    """Share of the random->expert gap closed by an own return; None if no ceiling."""
    if not ceiling or not ceiling.get("informative"):
        return None
    e, r = float(ceiling["expert_return"]), float(ceiling["random_return"])
    return float((float(own) - r) / (e - r))


def _demo_fraction_point(own: float, ceiling: Optional[Dict[str, Any]],
                         own_curve: Sequence[float]) -> float:
    """The `demo_fraction` curve field at one checkpoint.

    `_ceiling_progress` when the ceiling is informative. Otherwise -- the
    ceiling is UNINFORMATIVE, the expert paid no more than random under this
    reward -- the candidate's own-range progress: (own - min) / (max - min)
    over its own returns so far, 1.0 at a new best, the only scale-free reading
    left; `candidate.meta["ceiling"]["status"]` says which of the two a curve
    holds, and `rounds.py` ranks the two populations in separate tiers. An
    UNAVAILABLE ceiling (no demonstration policy for the env) never reaches
    here under `pruning_metric: demo_fraction`: `rounds.training_rounds`
    refuses before the first training, whatever the rule.
    """
    p = _ceiling_progress(own, ceiling)
    if p is not None:
        return p
    arr = np.asarray(list(own_curve) + [float(own)], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or float(arr.max() - arr.min()) <= 0.0:
        return 0.0
    return float((float(own) - arr.min()) / (arr.max() - arr.min()))


def _ceiling_guard(cfg: Any, rule: Callable[[Sequence[float], float], bool], metric: str,
                   ceiling: Optional[Dict[str, Any]], decisions: List[Dict[str, Any]],
                   ) -> Callable[[Sequence[float], float], bool]:
    """`train.pruning_cfg.ceiling: demo_return` -- the rule may fire only once
    the training has closed `ceiling_fraction` of its own random->expert gap.

    Identity when the guard is off. When on, every firing of the wrapped rule
    is appended to `decisions` (the seed row carries them): the best own return
    so far, the progress it amounts to, and whether the guard BLOCKED the stop.
    Below the fraction the training runs on to its budget even though the curve
    is flat -- that is the whole point, the plateau being the one the rule
    mistakes for a finished policy. An uninformative or unavailable ceiling
    cannot guard, so the plain rule's verdict stands and the decision says why.
    """
    if str(cfg.get("train.pruning_cfg.ceiling", "none") or "none") != "demo_return":
        return rule
    fraction = float(cfg.get("train.pruning_cfg.ceiling_fraction", 0.8) or 0.0)
    informative = bool(ceiling and ceiling.get("informative"))
    status = str((ceiling or {}).get("status", "unavailable"))

    def guarded(curve: Sequence[float], budget_frac: float) -> bool:
        if not rule(curve, budget_frac):
            return False
        arr = np.asarray(curve, dtype=float)
        finite = arr[np.isfinite(arr)]
        best = float(finite.max()) if finite.size else float("nan")
        if metric == "demo_fraction":
            progress: Optional[float] = best if informative else None
        elif metric == "own_reward":
            progress = _ceiling_progress(best, ceiling) if informative else None
        else:  # task_metric is not a reward return; `_check_coherence` refuses the pairing
            progress = None
        if progress is None:
            decisions.append({"checkpoint": len(curve), "budget_frac": float(budget_frac),
                              "rule_fired": True, "blocked": False, "progress": None,
                              "reason": f"ceiling_{status}"})
            return True
        blocked = bool(progress < fraction)
        decisions.append({"checkpoint": len(curve), "budget_frac": float(budget_frac),
                          "rule_fired": True, "blocked": blocked, "best_own_return": best,
                          "progress": float(progress), "ceiling_fraction": fraction})
        return not blocked

    return guarded


def _summarise_ceiling(candidate: Candidate, seed_metrics: Sequence[Dict[str, Any]]) -> None:
    """Fold the per-seed guard decisions into `candidate.meta["ceiling"]`."""
    ceiling = candidate.meta.get("ceiling")
    if not isinstance(ceiling, dict):
        return
    decisions = [d for sm in seed_metrics for d in (sm.get("ceiling_decisions") or [])]
    ceiling["blocked_prunes"] = sum(1 for d in decisions if d.get("blocked"))
    ceiling["allowed_prunes"] = sum(1 for d in decisions if d.get("rule_fired") and not d.get("blocked"))


def _pruner_for(cfg: Any) -> Tuple[Callable[[Sequence[float], float], bool], str]:
    """`(rule, curve field)` for this config -- the ONE place rule meets metric.

    Both backends call this rather than indexing `_PRUNERS` themselves, because
    a rule that meant one thing on `mock` and another on `sb3` would make every
    tester run test a different method from the one a full-scale run executes.
    """
    name = str(cfg.get("train.pruning", "none") or "none")
    fn = _PRUNERS.get(name, _prune_none)
    if name in _PRUNERS_WITH_CFG:
        fn = functools.partial(
            fn,
            patience=int(cfg.get("train.pruning_cfg.patience", 4) or 4),
            min_delta=float(cfg.get("train.pruning_cfg.min_delta", 0.02) or 0.0),
            min_checkpoints=int(cfg.get("train.pruning_cfg.min_checkpoints", 5) or 5))
    metric = str(cfg.get("train.pruning_metric", "own_reward") or "own_reward")
    # The schema enum already refuses an unknown metric at load; this is the
    # developer-side guard against the table and the enum drifting apart, and it
    # fails loud instead of quietly reading `fitness` (ground truth).
    assert metric in _PRUNING_FIELDS, f"train.pruning_metric={metric!r} has no curve field"
    if metric == "task_metric":
        # Native-signal rule (`bird.native_signal`): `task_metric` means "prune
        # on the task's ground truth", and that ground truth is the env's OWN signal,
        # never the BIRD `task_metric`/`fitness` curve (custom_metric). Route to
        # the native curve field -- `success_rate` (native_success) or
        # `gt_return` (native_reward). A task with neither is refused at load
        # (`_check_coherence`); reaching here without a native field is a bug.
        from ..native_signal import native_curve_keys
        keys, channel = native_curve_keys(env_id=cfg.get("problem.env_id"),
                                          task_id=cfg.get("problem.task_id"))
        if not keys:
            raise ValueError(
                "train.pruning_metric=task_metric on "
                f"problem.env_id={cfg.get('problem.env_id')!r} has no native signal "
                "to prune on (no shipped success or reward); coherence should have "
                "refused this config")
        return fn, keys[0]
    return fn, _PRUNING_FIELDS[metric]


# ==========================================================================
# Checkpoint selection -- `train.checkpoint_selection`
# ==========================================================================
#
# Both backends train in chunks and evaluate after each one. Under `final`
# they ship the LAST weights whatever the curve says: the policy rolled out for
# the judge, recorded to video, scored by §4 and stored under `policy_ref` is
# the final one. On HalfCheetah most candidates peak mid-training and finish
# 5-10 m lower, so Eureka's `checkpoint_aggregation: max_over_checkpoints`
# reports one policy's number and ships another. `best_by_reward` closes that
# gap from the training side: the checkpoint whose greedy evaluation paid the
# candidate's OWN reward the most is restored into the live model before
# anything downstream sees it, so
# the rollouts, the video, the stored blob and `warm_start_from_best` all get
# the same weights with no further plumbing.
#
# THE CRITERION IS THE DESIGNED REWARD, NEVER THE TASK METRIC. `reward_return`
# is what `_evaluate_policy` measured the candidate's raw reward paying the
# evaluation episodes -- raw, because `_rollout` never goes through `_Gym`, so
# `train.reward_norm`, the reference-policy pull and the anchor penalty shape
# learning and not this number. Selecting on `fitness` would leak ground truth
# into training. The honest limitation follows: a reward that inflates while
# the task metric falls picks the wrong checkpoint, and that is exactly what
# `output.trajectory_trace` exists to show.
#
# THE NOISE GUARD. A checkpoint is a handful of evaluation episodes (3 on sb3,
# `EVAL_EPISODES` = 4 on the surrogates and 8 on their last row), so a bare
# argmax over ~20 of them selects partly on luck -- the E[max] bias `ckpt_max`
# documents. The rule therefore restores only when the best row beats the LAST
# row by more than `min_delta` x the curve's own range (scale-free,
# `_prune_plateau`'s idiom); inside the margin the last checkpoint ships and the
# seed row says so. Ties go to the LATEST checkpoint: same number, more
# training. What "range" is, stated plainly: max minus min over the WHOLE
# finite curve, and the min is usually the untrained first row, so the margin
# scales with how far training moved the reward rather than with evaluation
# noise as such -- conservative in proportion to the climb, and a knob, not an
# estimate of the noise.
#
# WHAT IS NOT RESTORED. Only parameters move. The replay slice `_sb3_export_replay`
# ships and the surrogate's `seen` buffer stay end-of-training -- they are data
# for the next round's candidates to relabel, not a policy -- so under
# `best_by_reward` `policy_ref` and `replay_ref` describe two points in one
# training run. And the cascade screen's short run (`screens._train_backend_short`)
# is a training run through the same backend, so it follows the same rule: a
# LIMEN cascade under `best_by_reward` measures the short run's best checkpoint.
#
# ONE HELPER, TWO CALLERS. `_run_backend` and `_sb3_run` both call
# `_select_checkpoint` on the same `(curve, rule, min_delta)` and snapshot
# through `_running_best(curve, rule)` on the same series, for the reason `_pruner_for`
# gives: a key that meant one thing on `mock` and another on `sb3` would make
# every tester run test a different method from the one a full-scale run
# executes.

#
# TWO PEAK RULES, NOT ONE. `best_own_return` is the rule the hill-climb's
# `configs/hillclimb/v4_peak*.yaml` select. It is a BARE argmax over
# `reward_return`: the FIRST checkpoint on a tie, non-finite rows skipped,
# restored whenever the peak is not the last row, no margin. `best_by_reward`
# differs in exactly two cases and both are pinned by
# `tests/test_checkpoint_selection_rule.py`: a tie goes to the LATEST row, and
# a peak inside `min_delta` x range of the last row is not restored. The two
# are kept as two enum values rather than merged because the hill-climb
# results were measured under `best_own_return`'s exact semantics.
_SELECTION_RULES = ("final", "best_by_reward", "best_own_return")

#: `restored_checkpoint` names the checkpoint that was put back; these are the
#: reasons a seed row can give for NOT restoring, as executed. `final` is the
#: configured rule; the rest are `best_by_reward` outcomes.
_SELECTION_REASONS = ("final", "no_checkpoints", "last_is_best", "within_margin",
                      "no_snapshot", "restore_failed", "restored")


def _selection_reason(reason: str) -> str:
    """The reason a seed row records, checked against the closed list above at
    the moment it is written -- so a misspelt inline string in either backend
    fails the run that writes it rather than shipping an unlistable value."""
    assert reason in _SELECTION_REASONS, f"unknown checkpoint_selection_reason {reason!r}"
    return reason


def _selection_for(cfg: Any) -> Tuple[str, float]:
    """`(rule, min_delta)` for this config -- the one place the key is read."""
    rule = str(cfg.get("train.checkpoint_selection", "final") or "final")
    # The schema enum refuses anything else at load; this guards the tuple
    # and the enum drifting apart, and fails loud rather than reading `final`.
    assert rule in _SELECTION_RULES, f"train.checkpoint_selection={rule!r} is not a rule"
    return rule, float(cfg.get("train.checkpoint_selection_cfg.min_delta", 0.02) or 0.0)


def _best_checkpoint_index(curve: Sequence[Mapping[str, float]]) -> Optional[int]:
    """`best_own_return`: index of the curve's peak `reward_return`, FIRST on ties.

    Strict `>`, so the earliest of equal peaks wins; non-finite returns never
    win; `None` when nothing finite exists.
    Worked example: [0.2, 0.9, 0.9, 0.4] -> 1; [nan, 0.1] -> 1; [] -> None.
    """
    best: Optional[int] = None
    for i, p in enumerate(curve):
        r = float(p.get("reward_return", float("nan")))
        if not math.isfinite(r):
            continue
        if best is None or r > float(curve[best]["reward_return"]):
            best = i
    return best


def _running_best(curve: Sequence[Mapping[str, float]], rule: str) -> bool:
    """Whether the checkpoint just appended is the one whose parameters the
    rule would ship if the curve ended here -- i.e. whether to snapshot NOW.

    The one place the snapshot decision meets the post-loop decision, for the
    reason `_pruner_for` gives: the two rules break ties in opposite directions
    (`best_by_reward` latest, `best_own_return` first), so a backend that
    snapshotted on `_argmax_latest` under `best_own_return` would hold the wrong
    parameters at an interior tie and report `no_snapshot`. `final` never
    snapshots.
    """
    if rule == "final" or not curve:
        return False
    if rule == "best_own_return":
        return _best_checkpoint_index(curve) == len(curve) - 1
    return _argmax_latest([p["reward_return"] for p in curve]) == len(curve) - 1


def _argmax_latest(values: Sequence[float]) -> int:
    """Index of the largest value, ties to the LATEST. Non-finite reads as -inf,
    so a checkpoint whose evaluation blew up can never be the one shipped."""
    arr = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf,
                        posinf=-np.inf, neginf=-np.inf)
    return int(len(arr) - 1 - int(np.argmax(arr[::-1])))


def _select_checkpoint(curve: Sequence[Mapping[str, float]], rule: str,
                       min_delta: float) -> Tuple[Optional[int], str]:
    """Which recorded checkpoint to ship: `(index into curve, reason)`.

    `None` means the last one -- the `final` behaviour -- and the reason says why:
    the rule was `final`, the curve is empty, the last checkpoint IS the best,
    or the best beat it by no more than `min_delta` x the curve's range. Only
    `restored` carries an index, and the callers turn a snapshot they could not
    take or could not load into `no_snapshot` / `restore_failed` themselves.
    """
    if rule == "final":
        return None, "final"
    vals = [float(p.get("reward_return", np.nan)) for p in curve]
    if not vals:
        return None, "no_checkpoints"
    if rule == "best_own_return":
        # A bare argmax: first on ties, no margin (`min_delta` is not read).
        peak = _best_checkpoint_index(curve)
        if peak is None:
            return None, "no_checkpoints"
        if peak == len(vals) - 1:
            return None, "last_is_best"
        return peak, "restored"
    best = _argmax_latest(vals)
    if best == len(vals) - 1:
        return None, "last_is_best"
    arr = np.nan_to_num(np.asarray(vals, dtype=float), nan=-np.inf,
                        posinf=-np.inf, neginf=-np.inf)
    finite = arr[np.isfinite(arr)]
    span = float(finite.max() - finite.min()) if finite.size else 0.0
    if float(arr[best]) - float(arr[-1]) <= float(min_delta) * span:
        return None, "within_margin"
    return best, "restored"


def _snapshot_learner(learner: Any) -> Optional[np.ndarray]:
    """A COPY of the surrogate learner's parameters at this checkpoint.
    `snapshot()` returns the live array (`_QLearner.q`, `_CEMLearner.mu`), which
    the next round overwrites; a snapshot that aliases it would restore nothing."""
    snap = learner.snapshot() if learner is not None else None
    return None if snap is None else np.array(snap, copy=True)


def _snapshot_sb3(model: Any) -> Optional[Dict[str, Any]]:
    """A deep copy of `model.get_parameters()`: every network's state dict and
    every optimizer's -- exactly what `set_parameters(exact_match=True)` puts back.

    Deliberately NOT `_sb3_policy_blob`. That is `model.save`, which serialises
    the whole algorithm object -- hyperparameters, schedules, and mid-training
    whatever an attached anchor hung off it (`anchored_ppo.attach_anchor`'s
    frozen clone, which `run_seed` detaches only after the loop) -- into a zip,
    once per improving checkpoint. The parameters are the whole checkpoint here,
    and `get_parameters()` is exactly the inverse of the `set_parameters` that
    puts them back. The deep copy is load-bearing: `state_dict()` tensors alias
    the live parameters, so a shallow copy would silently track training and
    "restore" the final weights.

    What this does not carry: SB3's `saved_pytorch_variables` (SAC's
    `log_ent_coef`). They are not parameters of the shipped policy -- `predict`
    never reads them -- so the rollouts, the video and §4 are unaffected, and
    nothing downstream loses anything it had: `train.init: warm_start_from_best`
    goes through `set_parameters` too (`_sb3_apply_policy`) and has never
    carried the entropy coefficient across iterations. The only reader would be
    the entropy tuner during CONTINUED training of this same object, which does
    not happen after the loop.
    """
    try:
        import copy
        return copy.deepcopy(model.get_parameters())
    except Exception as exc:  # noqa: BLE001 - a missed snapshot ships the final policy
        log.warning("train.checkpoint_selection: could not snapshot the sb3 parameters "
                    "(%s: %s); this seed will ship its final checkpoint",
                    type(exc).__name__, exc)
        return None


def _restore_sb3(model: Any, params: Optional[Dict[str, Any]]) -> bool:
    """Load a `_snapshot_sb3` back into the live model.

    False when it did not take, and then the model holds its FINAL parameters
    again: `set_parameters` mutates module by module and can fail part-way (a
    shape mismatch in the second state dict leaves the first loaded), so the
    final parameters are copied first and put back on failure. Without that
    the seed row would say the final checkpoint shipped while the weights were
    half one policy and half another. A rollback that itself fails raises --
    the model is then in no state anyone can name, and `run_seed`'s contract is
    that a raise aborts this `_sb3_run` call rather than mis-measuring it.
    """
    if params is None:
        return False
    before = _snapshot_sb3(model)
    try:
        model.set_parameters(params, exact_match=True, device=model.device)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("train.checkpoint_selection: the best checkpoint did not load back "
                    "into the model (%s: %s); shipping the final checkpoint",
                    type(exc).__name__, exc)
        if before is None:
            raise RuntimeError("checkpoint restore failed with no copy of the final "
                               "parameters to fall back to") from exc
        try:
            model.set_parameters(before, exact_match=True, device=model.device)
        except Exception as exc2:  # noqa: BLE001
            raise RuntimeError("checkpoint restore failed and the final parameters could "
                               f"not be put back ({type(exc2).__name__}: {exc2}); the model "
                               "is in an undefined state") from exc2
        return False


# ==========================================================================
# The shared driver
# ==========================================================================


#: Stride between the seeds of ONE training call: seed `i` of a candidate is
#: `_seed_base(...) + i * SEED_STRIDE`. Prime, which is what lets `_phase_salt`
#: put a second stream provably off this one.
SEED_STRIDE = 7919


def _phase_salt(phase: str) -> int:
    """Offset that puts a report phase's seed stream off the search's.

    A `post:` phase retrains a reward the search already trained, and the seeds
    it draws must be FRESH: LIMEN retrains its winner over ten seeds precisely
    to remove the optimism of having selected on the search's own trials
    (`configs/methods/limen.yaml`, `final_retrain`), and RF-Agent's protocol asks the
    same. The salt is derived from the phase name and forced off every multiple
    of `SEED_STRIDE`, which makes the two streams disjoint by construction
    rather than by luck: `base + salt + j*STRIDE == base + i*STRIDE` would need
    `salt == (i - j) * STRIDE`. 24 bits, so a salted seed stays a valid legacy
    NumPy / SB3 seed (< 2**32) for any sane `seed`.

    `""` is the search itself and salts nothing, so a search seed is exactly
    `_seed_base`'s unsalted value.
    """
    if not phase:
        return 0
    salt = zlib.crc32(str(phase).encode()) & 0xFFFFFF
    if salt % SEED_STRIDE == 0:
        salt += 1
    return salt


def _seed_base(ctx: Any, state: Any, candidate: Candidate, phase: str = "") -> int:
    """Deterministic per-(run, restart, iteration, candidate[, phase]) seed.

    `zlib.crc32`, not `hash()`: Python's string hash is salted per process, so
    `hash()` here would make a run unreproducible across invocations -- exactly
    the failure the `seed` key exists to prevent.

    `phase` names a report protocol re-training a search candidate
    (`final_retrain`). That stream is anchored on the CANDIDATE's own iteration
    rather than the loop's current one -- the winner may come from any round,
    and the seeds it must not repeat are the ones it was selected on -- and
    offset by `_phase_salt`, so the two streams cannot intersect for any
    `train.seeds_per_candidate` and `final_retrain.n_seeds`. Without both, a
    last-round winner would be retrained on exactly its own search seeds: the
    same three trials the archive ranked it on would come back as the first
    three of LIMEN's "ten fresh seeds".
    """
    cid = zlib.crc32(str(candidate.cand_id).encode()) & 0xFFFF
    iteration = int(getattr(state, "iteration", 0))
    if phase:
        iteration = int(getattr(candidate, "iteration", iteration))
    return (int(ctx.cfg.get("seed", 0)) * 100003
            + int(getattr(state, "restart", 0)) * 9973
            + iteration * 97
            + cid
            + _phase_salt(phase))


def _seed_for(base_seed: int, seed_i: int, slice_index: int = 0) -> int:
    """The seed of one (seed replicate, slice) of a candidate's training.

    `base_seed + seed_i * SEED_STRIDE` is the line both backends use; the
    third term is `_SLICE_SEED_STRIDE` x the slice index under
    `train.interaction: shared_population`, zero everywhere else.

    HOW THE TWO SALTS COMPOSE. `base_seed` is `_seed_base(ctx, state,
    candidate, seed_phase)`, so a `post:` phase's `_phase_salt` is already
    inside it before this function adds the slice term: `seed = base +
    phase_salt + i * SEED_STRIDE + k * _SLICE_SEED_STRIDE`. The two salts
    never both apply in the shipped configs -- a report phase (`final_retrain`)
    retrains the winner with no `handoff`, so `k = 0`; a search slice runs
    under the search phase `""`, so `phase_salt = 0` -- and each keeps its own
    disjointness argument: the phase salt is forced off every multiple of
    `SEED_STRIDE` (`_phase_salt`), the slice stride is prime and co-prime with
    `SEED_STRIDE` and the episode stride (`_SLICE_SEED_STRIDE`).
    """
    return int(base_seed) + int(seed_i) * SEED_STRIDE + int(slice_index) * _SLICE_SEED_STRIDE


def _dr_params(cfg: Any, candidate: Candidate) -> Optional[Dict[str, Any]]:
    """`train.domain_randomization.mode`: none | fixed_human | generated."""
    mode = cfg.get("train.domain_randomization.mode", "none")
    if mode == "fixed_human":
        return dict(cfg.get("train.domain_randomization.params", {}) or {})
    if mode == "generated":
        # Stage-1 co-design (`generate.co_design.dr_config`) -- BIRD's, not
        # DrEureka's, whose release trains every reward candidate `--dr-config
        # off` (eureka.py:162) and trains under a generated DR only in stage 2,
        # on the one reward already chosen (dr_eureka.py:146). An empty block
        # means the LLM declined or its block failed to parse: train on nominal
        # rather than fail.
        return dict(candidate.dr_config or {}) or None
    return None


def _checkpoint_stride(rounds: int, interval: int) -> int:
    """`train.checkpoint_interval` counts *collection rounds*, not env steps.

    A steps reading would give 2000 checkpoints from the default (`10`) against
    `env_steps: 20000`. Clamped so §4 always has enough points for `iqm` and
    `last_k_mean` and never has thousands.
    """
    stride = max(1, int(interval or 1))
    if rounds // stride > MAX_CHECKPOINTS:
        stride = max(1, rounds // MAX_CHECKPOINTS)
    if rounds // stride < MIN_CHECKPOINTS:
        stride = max(1, rounds // MIN_CHECKPOINTS)
    return stride


def _evaluate_policy(env: Any, policy: Callable[[np.ndarray], np.ndarray],
                     rng: np.random.Generator, reward: CompiledReward,
                     n_episodes: int, on_error: Callable[[BaseException], float]
                     ) -> Tuple[Dict[str, float], Dict[str, float], int, List[Trajectory]]:
    """Greedy evaluation for one checkpoint: ground-truth metrics + components.

    `gt_return` is None -- not 0.0 -- when the env has no reference reward
    (see `_rollout`); every other metric is a float.
    """
    metrics: Dict[str, Any] = {"fitness": 0.0, "success_rate": 0.0, "gt_return": 0.0,
                               "reward_return": 0.0, "episode_length": 0.0}
    comps: Dict[str, List[float]] = {}
    trajs: List[Trajectory] = []
    steps = 0
    for _ in range(max(1, n_episodes)):
        traj, st, gt = _rollout(env, policy, rng, reward, on_error)
        steps += st
        trajs.append(traj)
        metrics["fitness"] += float(env.task_metric(traj))
        metrics["success_rate"] += 1.0 if traj.success else 0.0
        if gt is None or metrics["gt_return"] is None:
            metrics["gt_return"] = None
        else:
            metrics["gt_return"] += gt
        metrics["reward_return"] += float(traj.ret)
        metrics["episode_length"] += float(traj.length)
        for k, values in traj.component_values.items():
            # Per-step packing puts `None` where a component was not claimed;
            # the episode mean is over the claims that were made.
            claimed = [float(v) for v in values if v is not None and math.isfinite(float(v))]
            comps.setdefault(k, []).append(float(np.mean(claimed)) if claimed else 0.0)
    n = float(max(1, n_episodes))
    metrics = {k: (None if v is None else v / n) for k, v in metrics.items()}
    return metrics, {k: float(np.mean(v)) for k, v in comps.items()}, steps, trajs


def _evaluate_policy_timed(*args: Any, **kwargs: Any) -> Tuple[
        Dict[str, float], Dict[str, float], int, List[Trajectory], float]:
    """`_evaluate_policy` plus the WALL CLOCK it took. Same call, one more return.

    On a batched device tier the checkpoint eval can be most of the run. The
    evaluation rollouts step one env at a time through the n=1 bridge
    (`bird/envs/jax_base.py`), so their cost is a FIXED term -- checkpoints x
    rollouts x horizon x per-step latency -- that does not shrink as the
    training batch grows: a run with thousands of envs can train for a minute
    and evaluate for many. Without a clock that split can only be FIT from
    the total wall of several runs; this makes it a measurement.

    A SEPARATE NAME rather than a fifth element on `_evaluate_policy`, whose
    4-tuple return is unpacked by callers and tests
    (`test_gym_reference_reward.py`, `test_component_values_per_step.py`).

    ONE DEFINITION, THREE CALL SITES. A `perf_counter()` pair at each of the
    three callers would be fewer lines and three copies of one fact. The wall
    belongs to the evaluation, so it is measured where the evaluation is.

    WHAT IT DOES NOT MEASURE, so nobody reads it as more than it is: this is
    wall clock around the whole call -- the rollouts, the reward, the
    task_metric and any device sync inside them. It is not a breakdown, and
    on a GPU tier it includes compile time on the first checkpoint, which is
    why the per-checkpoint series matters and a single total would mislead.
    """
    t0 = time.perf_counter()
    metrics, comps, steps, trajs = _evaluate_policy(*args, **kwargs)
    return metrics, comps, steps, trajs, float(time.perf_counter() - t0)


def _learner_kwargs(cls: type, hyper: Mapping[str, Any], label: str) -> Dict[str, Any]:
    """The subset of `train.hyperparameters` this learner class actually accepts.

    `train.hyperparameters` is a free dict, so it is the one config key that can
    silently mean nothing. Two rules keep it honest: a name the learner accepts
    is passed through and genuinely changes the run, and a name it does not is
    WARNED about rather than dropped in silence. The warning matters more than
    the pass-through -- a swept hyperparameter that reaches no learner produces a
    grid of identical results, and `train.hyperparameter_search` would then
    report a max over a search that did nothing.
    """
    if not hyper:
        return {}
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    out: Dict[str, Any] = {}
    unknown: List[str] = []
    for key, value in hyper.items():
        if key in accepted:
            out[key] = value
        else:
            unknown.append(key)
    if unknown:
        log.warning("train.hyperparameters: %s is not a parameter of the %s learner and "
                    "will not affect this run (it accepts %s). A swept name that reaches "
                    "no learner makes train.hyperparameter_search a max over nothing.",
                    sorted(unknown), label, sorted(accepted - _LEARNER_PLUMBING))
    return out


#: Constructor parameters that are harness plumbing rather than tunable knobs;
#: excluded from the "it accepts ..." hint so the message names real choices.
_LEARNER_PLUMBING = frozenset({
    "env", "reward", "rng", "norm", "on_error", "total_budget", "init_q", "init_w",
    "secondary", "secondary_ratio", "n_parallel"})


def _make_learner(kind: str, env: Any, reward: CompiledReward, rng: np.random.Generator,
                  norm: _RewardNorm, on_error: Callable[[BaseException], float],
                  budget: int, init: Optional[np.ndarray],
                  secondary: Sequence[Any], ratio: float, n_parallel: int,
                  hyper: Optional[Mapping[str, Any]] = None,
                  fusion_alpha: Optional[float] = None) -> _Learner:
    """Pick a policy class from what the *environment* supports.

    `force_tabular` is what `train.backend: tabular` means; everything else uses
    an exact table where the state space enumerates and a linear policy where it
    does not. No method name reaches this function and none ever should.

    `hyper` is `train.hyperparameters`, filtered to what the chosen class accepts
    -- which is what makes `train.hyperparameter_search` a real sweep rather than
    a bookkeeping exercise.
    """
    hyper = hyper or {}
    if kind == "planner":
        return _PlannerLearner(env, reward, rng, norm, on_error,
                               **_learner_kwargs(_PlannerLearner, hyper, "planner"))
    exact = getattr(env, "exact_states", None) is not None
    if kind == "force_tabular" or exact:
        if kind == "force_tabular" and not exact:
            log.warning("train.backend=tabular on %s: the state space does not "
                        "enumerate, so the table is a binning of %d cells, not an "
                        "exact representation", getattr(env, "name", "?"),
                        int(getattr(env, "n_disc_states", 0)))
        return _QLearner(env, reward, rng, norm, on_error, total_budget=budget,
                         init_q=init, fusion_alpha=fusion_alpha,
                         secondary=secondary, secondary_ratio=ratio,
                         n_parallel=n_parallel,
                         **_learner_kwargs(_QLearner, hyper, "tabular_q"))
    if secondary:
        log.debug("train.init=secondary_replay_buffer is a no-op under the "
                  "linear-policy searcher (nothing replays a buffer)")
    return _CEMLearner(env, reward, rng, norm, on_error, total_budget=budget,
                       init_w=init, fusion_alpha=fusion_alpha, n_parallel=n_parallel,
                       **_learner_kwargs(_CEMLearner, hyper, "cem"))


# ==========================================================================
# reward_source  --  reward_source_fn(ctx, state, candidate) -> CompiledReward
# ==========================================================================
#
# WHAT REWARD THE LEARNER SEES. Every published method trains on the reward
# the generator wrote, so `candidate` is the default and is what every
# published config resolves to.
#
# WHY THE OTHER MEMBER EXISTS. An ORACLE arm trains the same learner on the
# environment's own shipped reward, and it is a published baseline rather than
# an ablation: it is the "Oracle" curve of Text2Reward's Fig. 2
# (`refs/tex/text2reward/text/experiments.tex:55`, "the expert-written reward
# function provided by the environment") and of CARD's Fig. 3, the bar every
# searched number on those tasks is read against.
#
# WHY IT IS NOT A PROGRAM, and this is the part worth reading before anyone
# proposes replacing it with one. The obvious shape is
# `llm.generator.provider: fixed` plus an authored transcription of the shipped
# reward, and it works on gym HalfCheetah-v5 because that reward is a function
# of the observation alone (it reads `s[9]`). It does NOT generalise, and
# Meta-World is where it breaks:
#
#   * A reward program is handed `(s, a)` and nothing else. `CompiledReward.
#     _make_binder` maps a parameter named `self`/`env`/`task`/`cfg`/`config` to
#     plan entry -1 and `_apply_bind` passes None for it, deliberately ("let it
#     fail loudly"). So a program cannot reach the adapter at training time --
#     though it CAN during verification, where `verification.call_reward` passes
#     `_SelfProxy(ctx.env)`. A `self.reference_reward(...)` program therefore
#     passes stage 2 and dies every step of stage 3, which is exactly the trap
#     `bird/envs/metaworld.py`'s `_HELPERS` comment exists to warn about: "a
#     wasted 500-step training run misattributed to the reward".
#   * And the transcription could not be faithful anyway. On the v2 branch of
#     all six of CARD's Meta-World task files, every one reads simulator state
#     the 39-D observation does not carry.
#     `tcp_center` is the midpoint of the two finger SITES
#     (`metaworld/sawyer_xyz_env.py:72-81`) and is explicitly not `s[0:3]`,
#     which is the `hand` BODY's xpos; `init_tcp`, `obj_init_pos`,
#     `_handle_init_pos` and `window_handle_pos_init` are per-episode constants
#     set in `reset_model`; and `sweep-into` calls `_gripper_caging_reward`,
#     which reads pad site positions and per-task geometry straight out of the
#     simulator. Six approximate transcriptions would produce a curve labelled
#     "Oracle" that is not the oracle -- plausible, and invisible.
#
# So the faithful route is to hand the learner the adapter's own callable, which
# restores the snapshot for the state and calls the UNTOUCHED `evaluate_state`
# (`bird/envs/metaworld.py::reference_reward`). Nothing is transcribed, so
# nothing can drift.


@register("reward_source", "candidate",
          doc="Train on the reward the generator wrote -- every published method.")
def reward_source_candidate(ctx: Any, state: Any, candidate: Candidate) -> Any:
    """The default, and the only value any published config resolves to.

    Exactly `compile_reward` on the candidate's program, the call every
    training site makes."""
    return compile_reward(candidate.reward_code, candidate,
                          language=reward_language(ctx.cfg))


@register("reward_source", "reference",
          doc="Train on the environment's own shipped reward (the ORACLE arm); "
              "the candidate's program is compiled and recorded but not trained on.")
def reward_source_reference(ctx: Any, state: Any, candidate: Candidate) -> Any:
    """`env.reference_reward`, wrapped to the `CompiledReward` interface.

    SCORES THE STATE IT ARRIVES IN, which is the BIRD convention and the same
    choice `EnvAdapter.gt_reward` makes (`s2 or s`).

    The equality is EXACT rather than conventional, and it is worth stating
    because it is the property a reader of an oracle run will want.
    `training._rollout` computes `gt_return += float(
    env.reference_reward(s2, a))` -- `s2` directly, not through `gt_reward`. So
    under `reference` the per-step training reward and the recorded `gt_return`
    increment are LITERALLY THE SAME CALL on the same two arguments: not two
    consistent conventions, one number reported twice.
    `tests/test_reward_source.py` asserts that over a rollout, so a future edit
    to either side breaks it rather than quietly separating them.

    No components, deliberately. The dict is the channel §4's per-component
    reflection reads, and a shipped reward has no BIRD-visible decomposition:
    Meta-World's own `evaluate_state` returns a tuple of named shaping terms,
    but naming them here would be this repo asserting a decomposition upstream
    does not publish as one. An oracle arm has nothing to reflect on anyway --
    `generate.n_candidates: 1` and one iteration, which `_check_coherence`
    requires.

    `has_reference_reward` is checked at LOAD by `_check_coherence` and again
    here, because the load check reads `problem.env_id` and this reads the
    adapter that was actually constructed; an env that lost the method between
    the two would otherwise raise `NotImplementedError` once per step.
    """
    env = ctx.env
    if not getattr(env, "has_reference_reward", False):
        raise ValueError(
            f"train.reward_source=reference needs an adapter with a reference "
            f"reward, and {type(env).__name__} "
            f"(problem.env_id={ctx.cfg.get('problem.env_id')!r}) reports none. "
            "On a multi-task adapter this is a per-TASK fact: "
            "tasks/<id>/shared_spec.yaml's reward.human.kind: none disowns the "
            "simulator's built-in reward even where it is computable "
            "(bird/envs/base.py::has_reference_reward)")

    def compute_reward(state: Any, action: Any, next_state: Any) -> Any:
        arrived = state if next_state is None else next_state
        return float(env.reference_reward(arrived, action)), {}

    return CompiledReward(compute_reward, "reference_reward")


def _training_reward(ctx: Any, state: Any, candidate: Candidate) -> Any:
    """`train.reward_source`: the reward the LEARNER sees.

    One dispatch, used by all four training sites (`_run_backend`, `_sb3_run`,
    `fasttd3._fasttd3_run`, `simba_v2.simba_v2_backend`), so a fifth backend
    cannot quietly train on a different channel than the other four.
    `tests/test_reward_source.py` checks every site by parametrising over the
    backend registry rather than over a literal list, so a new backend that
    bypasses this dispatch fails it. Compilation failures stay the
    caller's to catch: each site records a spent slot and a `TrainResult` with
    the error, and this function deliberately does not swallow that.
    """
    name = str(ctx.cfg.get("train.reward_source") or "candidate")
    if name != "candidate":
        # JOURNALLED, NOT STAMPED ON THE SEED ROWS, and the choice is deliberate.
        # A run whose learner did not see the candidate's program must say so in
        # the artifact -- otherwise an oracle run and a searched run are
        # distinguishable only by their resolved config, and `report.json` would
        # attribute the environment's reward to the generator. The journal is the
        # right home rather than `seed_metrics`: those rows are built in three
        # backends, and a new artefact field there is compared by default by `tests/test_parallelism.py`'s
        # whole-run-dir equality (see `_stamp_seed_rows`).
        #
        # BEFORE THE MEMBER CALL, not after, and the ordering is the whole point.
        # `reward_source_reference` RAISES when the
        # adapter reports no reference reward, and all three call sites turn that
        # into `TrainResult(trained=False)` plus a charged budget slot -- so
        # emitting afterwards means the one failure that most needs this record
        # is the only one without it: a reader would see a failed candidate and a
        # spent slot with nothing but `config.resolved.yaml` to say the learner
        # was never going to see the generator's program. `reward_name` was the
        # only field that needed the constructed object, and it is derivable from
        # the member name for both members, so nothing is lost by emitting first.
        #
        # SAFE UNDER BOTH FORKS, and the reason is stated rather than asserted,
        # because an event journalled from inside a backend call can be lost
        # under a fork (the parallel journal then carries no such event at
        # all). Under `candidate_parallelism: parallel` this is safe because
        # `Context.event` falls back to `self.event_buffer` when `rundir is
        # None`, which is every worker, `build_child_payload` arms that buffer
        # before the work and `_merge_child` replays it. Under the SEED fork it
        # is safe for a different reason: the seed payload carries no `events`
        # channel at all (the backend's fold over `_fork_seeds` replays no such
        # key), so an event emitted inside a seed child WOULD be lost -- and this one runs at the
        # top of the backend call, before `_fork_seeds`.
        ctx.event("train", event="reward_source", source=name,
                  cand_id=getattr(candidate, "cand_id", ""),
                  reward_name="reference_reward" if name == "reference" else name)
    return _registry_get("reward_source", name)(ctx, state, candidate)


def _scaled_reward(ctx: Any, state: Any, candidate: Candidate, reward: Any) -> Any:
    """`train.reward_scaling`, applied between compilation and training.

    LaRes Eq. 3 only; `none` leaves no plan and is what every other config
    resolves to, so this call cannot have moved a number anywhere else. The
    PLAN is stage 3's, made in the parent before any fork
    (`population.scaling_elite_moments`, `bird.py::train`); this only applies
    it, reading nothing off `ctx` -- which is what lets a backend run this
    inside a forked worker without desynchronising `ctx.rng` or losing the
    journal record (`tests/test_parallelism.py`'s invariant).
    """
    from .population import apply_reward_scaling  # local: no import cycle
    return apply_reward_scaling(candidate, reward)


def _scaling_record(candidate: Candidate, reward: Any) -> Dict[str, Any]:
    """`train.reward_scaling` as EXECUTED, for the seed row.

    `population._AffineReward` carries its own `record` (the affine parameters,
    the moments and the sample they were taken over); a reward trained unscaled
    reports the plan's reason (`applied: False, reason: ...`) rather than
    nothing -- a skipped stabiliser and one that ran must stay distinguishable
    in the artifact, which is the whole reason LaRes's Eq. 3 announces every
    skip. Written only when the key is not `none` (see the callers), so every
    other config's seed row carries no such field.
    """
    rec = getattr(reward, "record", None)
    if isinstance(rec, dict) and rec.get("applied"):
        return dict(rec)
    plan = (candidate.meta or {}).get("reward_scaling") if candidate is not None else None
    if isinstance(plan, dict):
        return {**plan, "applied": False}
    return {"applied": False}


def _run_backend(ctx: Any, state: Any, candidate: Candidate, n_seeds: int,
                 *, kind: str, step_cap: int, label: str,
                 env_steps: Optional[int] = None,
                 resume_ref: Optional[str] = None,
                 seed_phase: str = "",
                 handoff: Optional[SliceHandoff] = None) -> TrainResult:
    """Everything the four backends share. One RL run per (candidate, seed).

    Seed i is `_seed_base(...)` plus `_seed_for`'s fixed stride, so it is the
    same integer whichever way the call's seeds are scheduled. That is what
    keeps `train.candidate_parallelism` a schedule key rather than a science
    one.

    `handoff` (`train.interaction: shared_population`) is honoured HERE only
    for its slice index, which salts the seed so a resumed slice does not
    replay its predecessor's episode stream. The replay half -- pre-filling
    the learner from the round pool and exporting what it added -- is not
    wired on the surrogate learners: they are the tester tier, run for
    seconds, and their `seen` buffers self-cap at 4,096 rows. The seed row
    records `replay_prefill_supported: false` so a tester-profile LaRes run
    says in its artifact that the buffer mechanism did not execute there,
    and the driver journals it once per round (`interaction_buffer_unsupported`).
    """
    cfg = ctx.cfg
    env = ctx.env
    t0 = time.monotonic()
    result = TrainResult(cand_id=candidate.cand_id, candidate=candidate)

    # -- compile -----------------------------------------------------------
    try:
        reward = _scaled_reward(ctx, state, candidate,
                                _training_reward(ctx, state, candidate))
    except Exception as exc:  # noqa: BLE001
        # Eureka's model: the RL job launches and dies, so the slot is spent.
        # Counting it keeps `execute_rate` and the budget honest.
        ctx.budget.record_training(env_steps=0, gpu_seconds=time.monotonic() - t0)
        result.trained = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.wallclock_s = time.monotonic() - t0
        log.debug("  %s: reward would not compile -- %s", candidate.cand_id, result.error)
        return result

    # -- install the co-designed observation -------------------------------
    #
    # Failure here is the SAME failure a reward that will not compile is, and
    # for the same reason: the RL job launches, finds it has no input layer, and
    # dies -- so the slot is spent and counting it keeps `execute_rate` and the
    # budget honest. It is not a degrade-to-raw-state path. Falling back to the
    # environment's own observation would silently convert this candidate into a
    # `limen_reward_only` candidate inside a `limen` run, which is precisely the
    # confusion between those two points that installing the observation exists
    # to end.
    try:
        learn_env, _phi = _install_observation(ctx, candidate, env, kind)
    except Exception as exc:  # noqa: BLE001
        ctx.budget.record_training(env_steps=0, gpu_seconds=time.monotonic() - t0)
        result.trained = False
        result.error = f"{type(exc).__name__}: {exc}"
        result.wallclock_s = time.monotonic() - t0
        candidate.meta["observation_installed"] = False
        candidate.meta["observation_error"] = result.error
        log.debug("  %s: co-designed observation would not install -- %s",
                  candidate.cand_id, result.error)
        return result

    # -- config ------------------------------------------------------------
    n_seeds = max(1, int(n_seeds or cfg.get("train.seeds_per_candidate", 1) or 1))
    # `env_steps` is the caller's override and beats the config. It exists for
    # LIMEN's cascade screen, which asks for a SHORT training to reject a
    # candidate cheaply -- see `screens._train_backend_short`. If a backend
    # read `train.env_steps` for itself the screen's request would be silently
    # discarded: a LIMEN run on Meta-World would journal
    # `short_budget_honoured: false` on every candidate while paying 100,000
    # steps for a screen configured at 10,000. Ten times the intended cost, on
    # the one mechanism the method exists to demonstrate.
    #
    # `step_cap` still applies on top: an override may shorten a run, never
    # lengthen it past what the backend can honestly execute.
    requested = env_steps if env_steps is not None else cfg.get("train.env_steps", 20000)
    steps_budget = max(200, min(int(requested or 20000), step_cap))
    n_parallel = max(1, int(cfg.get("train.n_parallel_envs", 1) or 1))
    interval = int(cfg.get("train.checkpoint_interval", 10) or 10)
    timeout_s = float(cfg.get("train.timeout_s", 3600) or 3600)
    norm_mode = cfg.get("train.reward_norm", "none")
    detection = cfg.get("train.failure_detection", "exception")
    rule, prune_field = _pruner_for(cfg)
    prune_metric = str(cfg.get("train.pruning_metric", "own_reward") or "own_reward")
    select_rule, select_min_delta = _selection_for(cfg)
    rollouts = max(1, int(cfg.get("evaluate.rollouts_per_candidate", 3) or 1))
    n_replayed = _n_replayed_rollouts(cfg)
    log_keys = set(cfg.get("train.log", []) or [])
    want_components = "reward_component_values" in log_keys
    want_episode_stats = "episode_stats" in log_keys
    if not want_components and cfg.get("evaluate.feedback.granularity") == "per_component":
        log.warning("evaluate.feedback.granularity=per_component but 'reward_component_values' "
                    "is not in train.log -- per-component feedback will have nothing to read")

    init = _training_init(ctx, state, cfg, resume_ref, candidate)
    init_ref, secondary, ratio = init.ref, init.secondary, init.ratio
    #: ROSKA only. `_training_init` leaves alpha unresolved because `sc_bo`
    #: scores it by training probes, which needs the learner machinery below.
    #: `None` for every other mode, and the learners take their verbatim-copy
    #: path on `None`, so no other method's numbers move.
    #:
    #: Resolved HERE, in the parent, once per candidate -- not inside
    #: `run_seed`. The paper searches per REWARD FUNCTION ("J=12 iterations per
    #: reward function", appendix.tex:42) and its TTS arithmetic counts one
    #: search per reward, so a per-seed search would multiply the probe budget
    #: by `train.seeds_per_candidate` and break the 80300-vs-90000 ratio --
    #: invisibly, because that key is 1 in the default protocol. A plain float
    #: also crosses the seed-fork boundary, which a per-seed cache would not.
    probe_steps_spent = 0
    probe_eval_spent = 0

    def _fusion_probe(alpha: float) -> float:
        """Train `probe_fraction` of a full training from theta_f(alpha), and score it.

        The score is `reward_return` -- the return under the CANDIDATE'S OWN
        reward, the same field `train.checkpoint_selection` selects on. Never
        the task metric: choosing a warm start by ground truth would leak it
        into training, which is the rule `_select_checkpoint` already states.

        Its RNG is its own stream, derived from the candidate and the alpha and
        never drawn from a seed's `rng`, so running the search cannot perturb
        the seed streams beside it -- a probe that shifted them would make
        every ROSKA number depend on how many alphas the GP happened to try.
        """
        nonlocal probe_steps_spent, probe_eval_spent
        # T_BO as a fraction of the FULL training this candidate would get, which
        # is the paper's quantity (appendix.tex:98) and is horizon-independent.
        pr_steps = max(1, int(round(float(cfg.get("train.fusion.sc_bo.probe_fraction", 0.0) or 0.0)
                                    * int(cfg.get("train.env_steps", 0) or 0))))
        # `_seed_base` + crc32, never `hash()`: Python's string hash is salted
        # per process, so the probe scores -- hence the GP's chosen alpha, hence
        # every sc_bo run -- would differ between two invocations of one seeded
        # config. `_seed_base`'s own docstring forbids it for exactly this, and
        # no determinism test here catches it: a fork INHERITS the salt, so
        # `parallel` and `sequential` agree while two separate runs do not.
        prng = np.random.default_rng(
            (_seed_base(ctx, state, candidate, "fusion_probe")
             + zlib.crc32(f"{float(alpha):.6f}".encode())) % (2 ** 32))
        pnorm = _RewardNorm(norm_mode)
        perr = _error_router(detection, io.StringIO())
        learner = _make_learner(kind, learn_env, reward, prng, pnorm, perr, pr_steps,
                                _POLICY_STORE.get(init_ref or ""), secondary, ratio,
                                n_parallel, hyper=cfg.get("train.hyperparameters") or {},
                                fusion_alpha=float(alpha))
        spent = 0
        while spent < pr_steps:
            got = learner.improve(pr_steps - spent)
            if got <= 0:  # a learner that cannot spend its budget must not spin
                break
            spent += got
        metrics, _c, ev_steps, _t = _evaluate_policy(learn_env, learner.policy(), prng,
                                                     reward, 1, perr)
        # Charged whether or not this alpha wins. TRAINING and EVALUATION steps
        # are counted separately and deliberately: the paper's TTS arithmetic
        # (6x200x12 + 300x6 + 2500 per round, 80300 against Eureka's 90000,
        # appendix.tex:100-107) counts training epochs ONLY, so a single total
        # that folded in the greedy scoring rollout could not be checked against
        # it. `probe_train_steps` is the number that reproduces the paper.
        probe_steps_spent += spent
        probe_eval_spent += int(ev_steps)
        # AND charged to the run's budget, not only to the fusion record. The
        # precedent is in `record_rollout_steps`' own docstring: card's TPE
        # collection reached `TrainResult.env_steps_used` and never reached
        # here, so `budget.json` under-reported its env interaction ~4x and two
        # artifacts silently disagreed about one quantity. The probes are the
        # LARGEST term of ROSKA's cost (12 x 200 of Eureka's 3000 per
        # candidate), so leaving them out would understate the method by more
        # than card was understated. Steps only -- a probe is not a policy
        # training, so it must not move `policy_trainings` or any cap.
        #
        # The two artifacts therefore count different things, deliberately:
        # `budget.json` is every env interaction the run paid for (training +
        # evaluation + feedback rollouts + PROBES), while the summed
        # `train_result.env_steps_used` is the candidates' own trainings and
        # omits the probes. Nobody should have to rediscover that subtraction --
        # `tests/test_roska_fusion.py::test_the_probe_steps_reach_the_budget_
        # and_not_the_train_results` pins both halves.
        ctx.budget.record_rollout_steps(spent + int(ev_steps))
        return float(metrics.get("reward_return", 0.0))

    # A probe that RAISES fails the candidate the way a reward that will not
    # compile does, and for the same reason: the probe runs the candidate's own
    # reward (`_fusion_probe` -> `learner.improve` -> `reward(s, a, s2)`), so a
    # program that compiles and dies at its first call -- the mock's
    # `wrong_signature` archetype, `def compute_reward():` -- dies HERE on a
    # ROSKA config, before the seed loop below whose own `except` would have
    # recorded it as `trained=False`. Uncaught, the exception would take the
    # whole run down, and the mock LLM reaches that archetype at a probe
    # whenever a prompt change re-rolls its draws. The slot is spent (the RL
    # job launched and died), so it is charged, exactly as the compile branch
    # above charges it.
    try:
        fusion_alpha, fusion_record = _resolve_fusion(ctx, cfg, init, _fusion_probe)
    except Exception as exc:  # noqa: BLE001 -- the probe ran the candidate's reward
        ctx.budget.record_training(env_steps=0, gpu_seconds=time.monotonic() - t0)
        result.trained = False
        result.error = f"fusion probe: {type(exc).__name__}: {exc}"
        result.wallclock_s = time.monotonic() - t0
        log.debug("  %s: the fusion probe failed -- %s", candidate.cand_id, result.error)
        return result
    if fusion_record is not None:
        # Always present, zero under `fixed`, so the field is never absent and a
        # reader never has to infer whether a search happened.
        # NOMINAL against ACTUAL, both recorded, because they differ and the gap
        # is a property of the environment rather than of the search. A learner
        # spends whole EPISODES (`improve` loops `range(env.horizon)`), so it
        # cannot stop mid-episode: T_BO=200 against a 250-step horizon spends
        # 250, a 1.25x overshoot. The paper's TTS arithmetic is the nominal
        # figure (evaluations x T_BO); what the run actually paid is the other.
        # Reporting only the nominal would understate the cost of every ROSKA
        # run on a long-horizon task; reporting only the actual would make the
        # 80300-vs-90000 claim uncheckable.
        fusion_record["probe_train_steps_nominal"] = int(round(
            int(fusion_record.get("evaluations", 0))
            * float(fusion_record.get("probe_fraction", 0.0))
            * int(cfg.get("train.env_steps", 0) or 0)))
        fusion_record["probe_train_steps"] = probe_steps_spent
        fusion_record["probe_eval_steps"] = probe_eval_spent
        fusion_record["probe_env_steps"] = probe_steps_spent + probe_eval_spent
    result.init_source, result.init_from_cand_id = init.source, init.from_cand_id
    result.init_similarity = init.similarity
    result.fusion = dict(fusion_record) if fusion_record else {}

    env.set_dr(_dr_params(cfg, candidate))
    base_seed = _seed_base(ctx, state, candidate, seed_phase)
    # Under the DR the training itself will see, before any seed forks, so every
    # seed's guard reads one number and a forked seed child inherits it.
    ceiling = _demo_ceiling(ctx, state, candidate, reward) if _ceiling_wanted(cfg) else None

    per_seed_curves: List[List[Dict[str, float]]] = []
    per_seed_components: List[List[Dict[str, float]]] = []
    per_seed_policies: List[Callable[[np.ndarray], np.ndarray]] = []
    total_steps = 0
    fatal_error = ""

    def run_seed(seed_i: int) -> Dict[str, Any]:
        """Train ONE seed and return everything the fold below reads.

        Pure of shared state on purpose: this body runs either inline (the
        sequential schedule) or inside a forked seed worker
        (`_seed_fork_workers` > 1, i.e. `train.candidate_parallelism: parallel`
        in the parent process -- the
        `final_retrain` path, whose `n_seeds` trainings are independent by the
        same argument stage 3's candidates are). It therefore must not touch
        `result`, the accumulators above, or `ctx.budget` -- the fold does, in
        seed order, so counters and `BudgetExceeded` fire exactly where the
        sequential schedule fires them.
        """
        seed = _seed_for(base_seed, seed_i,
                         handoff.index if handoff is not None else 0)
        rng = np.random.default_rng(seed)
        norm = _RewardNorm(norm_mode)
        stdout = io.StringIO()
        on_error = _error_router(detection, stdout)

        curve: List[Dict[str, float]] = []
        comp_curve: List[Dict[str, float]] = []
        spent = 0  # env steps spent *learning* -- what `train.env_steps` budgets
        spent_eval = 0  # checkpoint evaluation, counted in the cost report only
        seed_t0 = time.monotonic()
        seed_error = ""
        pruned_at = None
        timed_out = False
        learner: Optional[_Learner] = None
        best_snap: Optional[np.ndarray] = None  # `train.checkpoint_selection`: the
        best_snap_idx = -1                      # running-best checkpoint's parameters
        ceiling_decisions: List[Dict[str, Any]] = []
        pruner = _ceiling_guard(cfg, rule, prune_metric, ceiling, ceiling_decisions)

        try:
            # `learn_env` and not `env`: the learner's POLICY FEATURE SPACE is
            # the co-designed observation when one is installed, and the raw
            # adapter everywhere else in this function -- `_evaluate_policy`
            # below, `_rollout`, `set_dr`. `_ObsView` delegates `reset`/`step`,
            # so the learner's own training rollouts and the candidate reward
            # still see the state the environment produced.
            learner = _make_learner(kind, learn_env, reward, rng, norm, on_error, steps_budget,
                                    _POLICY_STORE.get(init_ref or ""), secondary, ratio,
                                    n_parallel,
                                    hyper=cfg.get("train.hyperparameters") or {},
                                    fusion_alpha=fusion_alpha)
            # Sized from what this learner costs per round, so
            # `checkpoint_interval` means the same number of curve points
            # whichever policy class the env forced.
            rounds_cap = max(MIN_CHECKPOINTS,
                             steps_budget // max(1, learner.steps_per_round()))
            stride = _checkpoint_stride(rounds_cap, interval)
            with contextlib.redirect_stdout(stdout):
                for rnd in range(1, rounds_cap + 1):
                        spent += learner.improve(steps_budget - spent)
                        if learner.diverged():
                            raise FloatingPointError(
                                "policy values diverged; the reward is unbounded "
                                "or produced non-finite values")
                        last = rnd == rounds_cap or spent >= steps_budget
                        if rnd % stride and not last:
                            continue
                        metrics, comps, ev_steps, _tr, ev_wall = _evaluate_policy_timed(
                            env, learner.policy(), rng, reward,
                            _checkpoint_episodes(cfg, last=last), on_error)
                        spent_eval += ev_steps
                        point: Dict[str, float] = {
                            "step": float(spent), "round": float(rnd), "seed": float(seed),
                            # Aliases: §4 selects the fitness key from
                            # `evaluate.fitness.metric`, whose enum names
                            # `consecutive_successes` and `task_success`.
                            "fitness": metrics["fitness"],
                            "score": metrics["fitness"],
                            "task_success": metrics["fitness"],
                            "consecutive_successes": metrics["fitness"],
                            "success_rate": metrics["success_rate"],
                            "gt_return": metrics["gt_return"],
                            "return": metrics["reward_return"],
                            "reward_return": metrics["reward_return"],
                            # Measured, not fitted. See
                            # `_evaluate_policy_timed`.
                            # WALL-CLOCK: listed in `tests/test_parallelism._VOLATILE`. A timing
                            # field that is not listed there breaks bit-identity by construction --
                            # parallel candidates contend, so it differs from sequential on every
                            # run, and the failure surfaces as "parallel changed
                            # candidates/.../train_result.json" three files from the cause.
                            "eval_wall_s": ev_wall,
                        }
                        if want_episode_stats:
                            point["episode_length"] = metrics["episode_length"]
                        if ceiling is not None:
                            point["demo_fraction"] = _demo_fraction_point(
                                metrics["reward_return"], ceiling,
                                [p["reward_return"] for p in curve])
                        curve.append(point)
                        comp_curve.append(comps if want_components else {})
                        # `train.checkpoint_selection`: keep the parameters of the
                        # running best by the candidate's OWN reward, under the
                        # rule's own tie-break (`_running_best`), so the snapshot
                        # and the post-loop decision cannot disagree.
                        if _running_best(curve, select_rule):
                            best_snap = _snapshot_learner(learner)
                            best_snap_idx = len(curve) - 1
                        if pruner([p[prune_field] for p in curve], spent / max(1, steps_budget)):
                            pruned_at = rnd
                            break
                        if spent >= steps_budget:
                            break
                        # `not last` for the same reason `_sb3_run` needs
                        # `ck + 1 < n_chunks`: a seed that has already finished
                        # its rounds has nothing left to abandon, and failing it
                        # for crossing a deadline it no longer needs discards a
                        # complete training. `spent >= steps_budget` covers the
                        # usual exit; this covers the one where `improve()`
                        # under-delivers and `rounds_cap` is what ends the loop.
                        if not last and time.monotonic() - seed_t0 > timeout_s:
                            seed_error = f"timeout after {timeout_s:g}s"
                            timed_out = True
                            break
        except Exception as exc:  # noqa: BLE001 -- recorded, never raised
            seed_error = f"{type(exc).__name__}: {exc}"

        grepped = _grep_stdout(detection, stdout.getvalue())
        seed_error = seed_error or grepped

        # `train.checkpoint_selection`: ship the best checkpoint by the designed
        # reward. Restored IN PLACE into the learner, before `learner.policy()`
        # is taken below and before `snapshot()` is stored, so the rollouts, the
        # store and a fork's payload all carry the restored parameters.
        ship_idx, select_reason = _select_checkpoint(curve, select_rule, select_min_delta)
        if ship_idx is not None:
            if best_snap is None or best_snap_idx != ship_idx:
                ship_idx, select_reason = None, "no_snapshot"
            elif learner is None or not learner.restore(best_snap):
                ship_idx, select_reason = None, "restore_failed"
        if select_rule != "final" and ship_idx is None:
            log.debug("  %s seed %d: train.checkpoint_selection=%s shipped the last "
                      "checkpoint (%s)", label, seed, select_rule, select_reason)

        wall = float(time.monotonic() - seed_t0)
        fitnesses = [p["fitness"] for p in curve] or [0.0]
        # The SHIPPED checkpoint's row -- the last one unless a restore happened.
        # `fitness`/`final`/`success_rate`/`gt_return` describe the policy that
        # left this function; the full curve stays in `checkpoints` as recorded.
        ship_i = ship_idx if ship_idx is not None else len(fitnesses) - 1
        seed_metric = {
            "seed": int(seed),
            "fitness": float(fitnesses[ship_i]),
            "final": float(fitnesses[ship_i]),
            "max": float(np.max(fitnesses)),
            "auc": float(np.mean(fitnesses)),
            "success_rate": float(curve[ship_i]["success_rate"]) if curve else 0.0,
            # None when the env has no reference reward (`_rollout`), never 0.0.
            "gt_return": _opt_float(curve[ship_i]["gt_return"]) if curve else 0.0,
            "env_steps": int(spent + spent_eval),
            "train_steps": int(spent),
            "wallclock_s": wall,
            "n_checkpoints": len(curve),
            "pruned_at_round": pruned_at,
            "timed_out": bool(timed_out),
            "train_steps_requested": int(steps_budget),
            "learner": learner.kind if learner is not None else "none",
            # `algorithm_cited`: the PAPER'S algorithm, never the learner that
            # ran (the long note is at the fasttd3 writer). All three
            # seed-row writers carry the same name or two backends disagree
            # about what the field means. `cfg[...]` rather than a `.get` with
            # a "ppo" fallback: the key is always in `_default.yaml`, so the
            # fallback would be dead, and a live one would silently claim PPO
            # for a method that cited something else.
            "algorithm_cited": cfg["train.algorithm"],
            "architecture": cfg.get("train.architecture", "mlp"),
            "backend": label,
            # WHICH MDP THIS SEED TRAINED ON, as executed. `None` on
            # `observation_dim` means the raw adapter -- `problem.search_space`
            # does not include `observation`, or the candidate carried none.
            # Recorded per seed for the same reason `warm_started_from` is: an
            # install that did not happen is indistinguishable from the ablation
            # without it.
            #
            # THE PAIR IS THE POINT, and the two halves are derived from
            # different objects on purpose. `observation_dim` is what was
            # INSTALLED (phi, measured before the learner exists);
            # `policy_input_dim` is what the learner was BUILT ON (the env object
            # it is holding). They agree iff the view reached `_make_learner` --
            # precisely the wiring that, if it were missing, a field derived
            # from phi alone would go on reporting as fine.
            "observation_dim": (int(_phi.dim) if _phi is not None else None),
            "policy_input_dim": (learner.policy_input_dim()
                                 if learner is not None else None),
            # `problem.horizon` as executed, read off the adapter -- the SAME
            # expression the other three seed-row writers use, because a field
            # two backends compute differently is worse than one backend
            # missing it.
            #
            # A test that asserts this field by SOURCE INSPECTION, per module,
            # cannot see this writer: `_sb3_run` lives in the same module, so
            # `training.py` would satisfy it on `_sb3_run`'s line while
            # `_run_backend`'s rows carried no `horizon_effective` at all. A
            # test whose unit is the file cannot see a writer inside it
            # (`tests/test_problem_horizon.py` asserts on the rows instead).
            #
            # And the field matters here: `mock` is the tester tier's backend,
            # so every offline run and every `--profile tester` point produces
            # these rows -- precisely the rows obtainable without a GPU.
            # `none` and `tabular` are real method points too (L2R's planner,
            # Singh 2009), and nothing stops either pinning a horizon.
            # The EPISODE length as run; the long note is on the same field
            # in `bird/components/simba_v2.py`.
            "horizon_effective": int(getattr(env, "horizon", 0) or 0),
            # What was ASKED for, beside what ran: the key clamps at
            # construction when the chosen env's episode is shorter, and the
            # long note is on the same pair in `bird/components/simba_v2.py`.
            "horizon_requested": (int(cfg.get("problem.horizon"))
                                  if cfg.get("problem.horizon") is not None else None),
            # `train.init` as EXECUTED. The same four keys `_sb3_run`
            # records, so one reader (and one test) can ask both backends
            # the same question. `warm_started_from` is the ref the learner
            # ACTUALLY adopted: `_make_learner` shape-guards `init_q` /
            # `init_w` and drops a mismatch in silence, and a warm start
            # that did not take is indistinguishable from the ablation
            # without this field.
            "init": cfg.get("train.init", "from_scratch"),
            "warm_started_from": (init_ref or "") if getattr(
                learner, "init_applied", False) else "",
            "fusion_alpha": fusion_alpha,
            "secondary_transitions": int(len(secondary or ())),
            "secondary_ratio": float(ratio),
            # `train.checkpoint_selection` as EXECUTED, the same rule `init` /
            # `warm_started_from` follow: a run whose last checkpoint was already
            # the best and a run that restored are the same curve, and only these
            # fields tell them apart. `restored_checkpoint` is the `round` of the
            # checkpoint put back (null = the last one shipped, and
            # `checkpoint_selection_reason` says why); `shipped_checkpoint` is the
            # round whose row `fitness` above describes; `final_checkpoint_fitness`
            # is what `fitness` would have been without the restore.
            "checkpoint_selection": select_rule,
            "checkpoint_selection_reason": _selection_reason(select_reason),
            "restored_checkpoint": (int(curve[ship_idx]["round"])
                                    if ship_idx is not None else None),
            "shipped_checkpoint": int(curve[ship_i]["round"]) if curve else None,
            "final_checkpoint_fitness": float(fitnesses[-1]),
            "error": seed_error,
            "checkpoints": curve,
        }
        if cfg.get("train.reward_scaling", "none") != "none":
            # LaRes Eq. 3 as executed -- see `_scaling_record`. Only under a
            # non-`none` key, for the same byte-identity reason as `slice_index`.
            seed_metric["reward_scaling"] = _scaling_record(candidate, reward)
        if handoff is not None:
            # Only under a slice hand-off, so an unsliced run's seed row
            # carries no slice fields.
            seed_metric["slice_index"] = int(handoff.index)
            seed_metric["replay_prefill_supported"] = False
        if ceiling is not None:
            seed_metric["ceiling_decisions"] = ceiling_decisions
        return {"curve": curve, "comp_curve": comp_curve, "spent": spent,
                "spent_eval": spent_eval, "pruned_at": pruned_at,
                "timed_out": timed_out, "seed_error": seed_error,
                "wallclock_s": wall, "learner": learner, "seed_metric": seed_metric}

    def fold_seed(out: Dict[str, Any]) -> None:
        """Parent-side bookkeeping for one seed, strictly in seed order."""
        nonlocal total_steps, fatal_error
        if out["seed_error"] and not fatal_error:
            fatal_error = out["seed_error"]
        if out["pruned_at"] is not None:
            result.pruned = True
        if out["timed_out"]:
            result.timed_out = True
        per_seed_curves.append(out["curve"])
        per_seed_components.append(out["comp_curve"])
        total_steps += out["spent"] + out["spent_eval"]
        result.seed_metrics.append(out["seed_metric"])
        # Each seed is one RL run; the budget must count it as one, or
        # `seeds_per_candidate` looks free and the cost column lies.
        ctx.budget.record_training(env_steps=out["spent"] + out["spent_eval"],
                                   gpu_seconds=out["wallclock_s"])

    # The planner's policy closes over live mutable state (`plan_rng`,
    # `sim_steps`) that no snapshot can rebuild in the parent, so its seeds
    # never fork; Q/CEM policies are pure functions of (env, snapshot array).
    if kind == "planner":
        seed_workers, seed_reason = 1, ""
    else:
        seed_workers, seed_reason = _seed_fork_workers(cfg, n_seeds)
    _note_seed_schedule(ctx, candidate, seed_workers, seed_reason)
    learner: Optional[_Learner] = None  # the LAST seed's learner (sequential path)
    final_snapshot = None               # the same two facts, sent home by a fork
    final_seen: Optional[List[Any]] = None
    final_existed = False

    try:
        if seed_workers <= 1:
            for seed_i in range(n_seeds):
                # PER SEED, not per candidate. One seed is one training, so a
                # candidate-level pre-flight would refuse a whole candidate where
                # the cap still has room for some of its seeds -- cap 4 with three
                # seeds must run four trainings, not three.
                # Trainings AND gpu hours, because one seed is a unit of
                # both. The gpu form asks "is the cap already spent" rather
                # than "would this cross it": a training's duration is not
                # knowable in advance, so the enforceable rule is not to
                # START a unit once the budget is gone.
                _why = (ctx.budget.would_exceed_training(1)
                        or ctx.budget.gpu_hours_exhausted())
                if _why:
                    raise BudgetExceeded(_why)
                out = run_seed(seed_i)
                learner = out.pop("learner")
                if learner is not None:
                    per_seed_policies.append(learner.policy())
                fold_seed(out)
            final_existed = learner is not None
            if learner is not None:
                final_snapshot = learner.snapshot()
                final_seen = list(getattr(learner, "seen", []))
        else:
            def seed_child(i: int) -> Dict[str, Any]:
                out = run_seed(i)
                lrn = out.pop("learner")
                out["learner_existed"] = lrn is not None
                out["learner_kind"] = getattr(lrn, "kind", "")
                out["snapshot"] = lrn.snapshot() if lrn is not None else None
                if i == n_seeds - 1 and lrn is not None:
                    out["seen"] = list(getattr(lrn, "seen", []))
                return out

            for seed_i, out in enumerate(
                    _fork_seeds(n_seeds, seed_child, seed_workers, label)):
                if "fatal" in out:
                    # `run_seed` already records every exception it can catch in
                    # `seed_error`; a payload marked fatal is something it could
                    # not have (a SIGSEGV, an unwritable spool), so it aborts
                    # this backend call the way an uncaught raise would.
                    raise RuntimeError(f"{label} seed worker {seed_i}/{n_seeds}: "
                                       f"{out['fatal']}")
                snapshot = out.pop("snapshot", None)
                lkind = out.pop("learner_kind", "")
                existed = bool(out.pop("learner_existed", False))
                if snapshot is not None:
                    # A CLOSED kind map: a future _Learner kind reaching this
                    # fold unmapped must fail its first forked run loudly, not
                    # have its snapshot reinterpreted as linear weights and its
                    # rollout share run on a garbage policy -- the same
                    # corrupted-measurement shape as the sb3 lost-blob case,
                    # refused at the same place.
                    # `learn_env`, NOT `env`. The child trained against the
                    # POLICY FEATURE SPACE -- the co-designed observation where
                    # one is installed -- so `snapshot` is shaped for phi's
                    # width, and rebuilding it against the raw adapter is a
                    # shape mismatch on the first rollout (`test_parallelism.py`
                    # exercises this on `limen`); on every other config the two
                    # objects are the same env.
                    if lkind == "tabular_q":
                        per_seed_policies.append(_greedy_policy(learn_env, snapshot))
                    elif lkind == "cem_linear":
                        per_seed_policies.append(_linear_policy(learn_env, snapshot))
                    else:
                        fatal_error = fatal_error or (
                            f"{label} seed fork: seed {seed_i} returned a "
                            f"snapshot for unmapped learner kind {lkind!r}; "
                            "its rollout share cannot be rebuilt, so the "
                            "result is marked failed rather than mis-measured")
                if seed_i == n_seeds - 1:
                    final_snapshot = snapshot
                    final_seen = out.pop("seen", None)
                    final_existed = existed
                fold_seed(out)

        # -- aggregate across seeds ---------------------------------------
        result.checkpoints = _mean_curve(per_seed_curves)
        result.gt_reward_curve = [p["gt_return"] for p in result.checkpoints]
        _stamp_seed_rows(result.seed_metrics, seed_workers, seed_reason)
        _summarise_ceiling(candidate, result.seed_metrics)
        if want_components:
            result.component_traces = _mean_components(per_seed_components)
        if reward.component_names:
            candidate.component_names = candidate.component_names or reward.component_names

        # -- final rollouts for §4 / the screens --------------------------
        roll_rng = np.random.default_rng(base_seed + 104729)
        for i in range(rollouts):
            policy = per_seed_policies[i % len(per_seed_policies)] if per_seed_policies \
                else (lambda s: env.action_set[0])
            traj, st, _gt = _rollout(env, policy, roll_rng, reward,
                                     _error_router("exception_soft", io.StringIO()))
            result.trajectories.append(traj)
            if i < n_replayed:
                _retain_replay_states(env, traj)
            total_steps += st
            # Charged to the run budget, like the store loop below:
            # `budget.json` and `train_result.json` must not disagree about
            # the same quantity.
            ctx.budget.record_rollout_steps(st)

        # -- dedicated rollouts for the TPE trajectory store ---------------
        # A SEPARATE rng, so toggling collection leaves the eval trajectories
        # above byte-identical (and parallel == sequential holds trivially).
        # Seed nuance, inert on CARD (`train.seeds_per_candidate: 1`) but real
        # at seeds > 1: the release collects all
        # 100 check rollouts from the FIRST seed's trained model
        # (train_and_eval's extra seeds are eval-only), where this loop
        # round-robins across the per-seed policies like the eval rollouts do.
        # Deliberately NO `_retain_replay_states` and NO snapshot export for
        # these: 100 x a MetaWorld horizon would FIFO-flush the adapter's LRU
        # and kill videos (see `_retain_replay_states`). Re-scoring (§2) and
        # rendering (§6) read Trajectory.states/rewards directly and never
        # replay the env.
        n_store = _n_store_rollouts(cfg)
        if n_store > 0:
            store_rng = np.random.default_rng(base_seed + 224737)
            store_steps = 0
            for i in range(n_store):
                policy = per_seed_policies[i % len(per_seed_policies)] if per_seed_policies \
                    else (lambda s: env.action_set[0])
                traj, st, _gt = _rollout(env, policy, store_rng, reward,
                                         _error_router("exception_soft", io.StringIO()))
                result.store_trajectories.append(traj)
                total_steps += st
                store_steps += st
            # Charged to the run budget as well as to `env_steps_used`: 100 x a
            # Meta-World horizon per candidate-iteration is ~33x the feedback
            # rollouts, and CARD's headline claim is about cost --
            # `budget.json` and `train_result.json` must not silently disagree
            # about the same quantity. Travels home from a fork like every
            # other counter, via `Budget.delta_since`/`merge_delta`.
            ctx.budget.record_rollout_steps(store_steps)

        # `RunState` keeps only the handle (state must stay checkpointable),
        # so the arrays live in the module-level stores.
        # "The last seed's learner is the one that is kept", whichever schedule
        # trained it -- a fork sends the snapshot and the raw `seen` buffer home.
        if final_existed and final_snapshot is not None:
            result.policy_ref = _store(_POLICY_STORE, f"policy:{candidate.cand_id}",
                                       final_snapshot, _POLICY_STORE_MAX_BYTES)
            _remember_code(candidate)
        if final_existed and _wants_replay(cfg):
            cap = int(cfg.get("train.secondary_buffer.size", 100000) or 100000)
            result.replay_ref = _store(_REPLAY_STORE, f"replay:{candidate.cand_id}",
                                       _replay_tail(final_seen or [], cap),
                                       _REPLAY_STORE_MAX_BYTES)
    finally:
        # `ctx.env` is one shared adapter: leaving this candidate's DR installed
        # would silently randomise the next candidate's training too.
        env.set_dr(None)

    result.env_steps_used = int(total_steps)
    result.wallclock_s = float(time.monotonic() - t0)
    if fatal_error:
        result.trained = False
        result.error = fatal_error
    return result


class _InitPlan(NamedTuple):
    """What `train.init` resolved to for ONE candidate, before any seed trains.

    `ref` / `secondary` / `ratio` are what the learners consume. `source` and
    `from_cand_id` are the artifact's answer to "where did this policy start?"
    (`TrainResult.init_source` / `.init_from_cand_id`, journalled by `bird.py`).
    `source` names the ref that RESOLVED in `_POLICY_STORE` at plan time --
    `parent`, `similar`, `best`, `bc_prior`, `continuation` -- or `scratch` when
    nothing did, which includes a `warm_start_*` whose ref was evicted: the
    learner then cold-starts, and recording `best` for it would be the
    plausible wrong value. `seed_metrics[*].warm_started_from` is the per-seed
    complement (did the blob actually LOAD into this learner).
    """

    ref: Optional[str]
    secondary: List[Any]
    ratio: float
    source: str
    from_cand_id: Optional[str]
    #: `warm_start_from_similar` only: the donor's structural similarity to
    #: this candidate (`selection._structural_similarity`), else None.
    similarity: Optional[float] = None
    #: `fused_warm_start` only (ROSKA): which `fusion_ratio_search` entry picks
    #: alpha. Resolved to a NUMBER later, in the PARENT and once per candidate
    #: (never per seed: the paper searches per reward function, appendix.tex:42,
    #: and its TTS counts one search per reward), because `sc_bo` scores
    #: candidate alphas by training short probes and there is no learner here.
    #: None everywhere else, so no other mode's plan is touched.
    fusion_search: Optional[str] = None


def _resolve_fusion(ctx: Any, cfg: Any, init: "_InitPlan",
                    probe: Optional[Callable[[float], float]] = None
                    ) -> Tuple[Optional[float], Optional[Dict[str, Any]]]:
    """Resolve ROSKA's fusion ratio, or `(None, None)` for every other mode.

    Returns `None` -- not 1.0 -- when `train.init` is not `fused_warm_start`,
    because the learners branch on `is None` to take their verbatim-copy path.
    A 1.0 here would be arithmetically equal for finite parameters and not for
    -0.0 or a non-finite fresh draw, and `tests/test_parallelism.py` compares
    whole run directories.

    `probe` trains a fused policy for T_BO steps at a candidate alpha and
    returns its return under the candidate's own reward. `sc_bo` REFUSES
    without one rather than degrading to `train.fusion.alpha`: a fixed ratio
    delivered under a config that says the ratio was searched is
    indistinguishable from a searched one in every artifact.
    """
    if init.fusion_search is None:
        return None, None
    search = _registry_get("fusion_ratio_search", init.fusion_search)
    alpha, record = search(ctx, cfg, probe)
    return float(alpha), record


def _resolved_source(ref: Optional[str], label: str) -> Tuple[str, Optional[str]]:
    """`(source, from_cand_id)` for a `policy:<cand_id>` ref, `scratch` if unresolvable."""
    if ref and ref in _POLICY_STORE:
        return label, (ref.split(":", 1)[1] if ":" in ref else ref)
    return "scratch", None


def _training_init(ctx: Any, state: Any, cfg: Any,
                   resume_ref: Optional[str] = None,
                   candidate: Optional[Candidate] = None,
                   quiet: bool = False) -> _InitPlan:
    """`train.init`: from_scratch | warm_start_from_best | warm_start_from_parent
    | warm_start_from_similar | secondary_replay_buffer | bc_prior
    | bc_prior_then_warm_start.

    `resume_ref` is NOT a config key and never reaches a config. It is how a
    round driver hands a candidate back its OWN policy from an earlier call in
    the same round: `train.interaction: shared_population` spends one pooled
    budget in waves, and `train.pruning: successive_halving_pool` trains in
    rungs (`rounds.py`); in both a candidate is trained several times per
    iteration and each later call must continue the last rather than
    cold-start. It overrides only the policy half; the secondary-buffer half of
    the mode is resolved as usual, so a config combining either driver with
    `train.init: secondary_replay_buffer` still gets its buffer. Recorded as
    `source: continuation` -- or `scratch` if that blob is gone, like every ref.

    `candidate` is read by the two per-candidate modes only (`parent`,
    `similar`); every other mode resolves off `state` alone.

    The REPLAY half of a slice hand-off -- the arm's own buffer, or the
    round's pool under `interaction_cfg.shared_buffer: true` -- does not come
    through here at all: it is the `SliceHandoff` argument, resolved by
    `_sb3_run` against `_ROUND_REPLAY`.
    """
    mode = cfg.get("train.init", "from_scratch")
    if resume_ref is not None:
        # `quiet`: the base plan is read for its secondary-buffer half only;
        # its parent/similar fallback did not happen to this call.
        base = _training_init(ctx, state, cfg, None, candidate, quiet=True)
        return _InitPlan(resume_ref, base.secondary, base.ratio,
                         *_resolved_source(resume_ref, "continuation"))
    if mode == "bc_prior_then_warm_start":
        # The CHAIN: round 0 starts from the clone, every later round from the
        # policy the judge selected in the round before (`state.policy_ref`,
        # carried by `loop.carry: policy_checkpoint`). `warm_start_from_best`
        # alone cold-starts round 0 because nothing has been selected yet, and
        # a chain that begins from random weights is not the experiment: it
        # should warm-start all the way through.
        # The anchor, when on, copies whatever policy the candidate starts from,
        # so it is a trust region around the clone in round 0 and around the
        # previous winner after -- no anchor change is needed for that.
        ref = getattr(state, "policy_ref", None)
        if ref is not None:
            return _InitPlan(ref, [], 0.0, *_resolved_source(ref, "best"))
        source = "bc_prior" if BC_PRIOR_REF in _POLICY_STORE else "scratch"
        return _InitPlan(BC_PRIOR_REF, [], 0.0, source, None)
    if mode == "bc_prior":
        # A FIXED ref, never a per-iteration one: every candidate of every iteration
        # starts from the same clone of the task's scripted policy, which
        # `ensure_bc_prior` builds in the parent before any fork. Read
        # exactly like the other modes -- a missing blob is a warning and a cold
        # start, never a crash -- so a backend that cannot build one (mock,
        # tabular: no torch) degrades the way `warm_start_from_best` does there.
        source = "bc_prior" if BC_PRIOR_REF in _POLICY_STORE else "scratch"
        return _InitPlan(BC_PRIOR_REF, [], 0.0, source, None)
    if mode == "warm_start_from_best":
        # RDA warm-starts from the previous best checkpoint ("may be
        # warm-started"; +Ckpt ablation Fig. 7a). The ref lives in RunState and
        # only survives if 'policy_checkpoint' is in `loop.carry`. Returned
        # whether or not it still resolves -- `_sb3_run` warns on a stale one.
        ref = getattr(state, "policy_ref", None)
        return _InitPlan(ref, [], 0.0, *_resolved_source(ref, "best"))
    if mode == "fused_warm_start":
        # ROSKA (AAAI 2025). Same ref as `warm_start_from_best` -- the previous
        # DP-round's best policy, Eq. 8 -- but BLENDED with a fresh draw rather
        # than copied, so the inherited policy keeps plasticity under a reward
        # the LLM may have changed a lot (methodology.tex:61-71). alpha is not
        # resolved here: `sc_bo` needs to train probes to score it.
        ref = getattr(state, "policy_ref", None)
        src, from_id = _resolved_source(ref, "best")
        # NO SEARCH WHEN THERE IS NOTHING TO FUSE WITH, which is round 1 and is
        # the paper's own shape: "In the first round ... These reward functions
        # are then trained using the PPO reinforcement learning algorithm for
        # 500 epochs. Then, a selection process is conducted, and the reward
        # function with the best-performed undergoes an additional 2500 epochs
        # of training" (appendix.tex:6-7) -- no fusion and
        # no SC-BO until there is a theta_best to blend with, which is why the
        # paper's TTS spends the search in rounds 2-5 only (5500 + 4 x (6x200x12
        # + 300x6 + 2500) = 80300, appendix.tex:100-107).
        #
        # Searching in round 1 would run the full J-evaluation search against
        # an inherited policy of None: the GP would rank alphas for a blend
        # that does not happen, and the alpha it returned could reach nothing.
        # On the published config that is 6 x 12 x 200 = 14,400 epochs against
        # a round-1 cost of 5,500, which takes the TTS to 94,700/90,000 = 1.05
        # and inverts the method's headline claim -- the failure
        # `_check_coherence`'s Eureka-plus-probes rule and the round-budget
        # comment in `bird.py` both exist to prevent.
        #
        # The condition is `_resolved_source`'s own verdict rather than
        # `ref is None`, so an evicted blob counts too: a candidate that
        # cold-starts has nothing to fuse with whatever the ref said. The
        # `resume_ref` branch above is unaffected and must stay so -- a winner
        # extension and a halving rung continue a policy, they do not fuse.
        search = None if src == "scratch" else \
            str(cfg.get("train.fusion.ratio_search", "fixed") or "fixed")
        return _InitPlan(ref, [], 0.0, src, from_id, fusion_search=search)
    if mode == "warm_start_from_parent":
        # NOT PUBLISHED (a BIRD extension for training efficiency).
        # Start from the policy trained under the reward THIS candidate was
        # mutated from -- `Candidate.parent_id`, the same id the journal's
        # `generate` event carries -- rather than from the incumbent's. The
        # two differ exactly when the pool holds several parents
        # (`generate.parent_source` / `update.topology`) or when the round
        # regressed and `update.elitism` kept the old incumbent while the
        # generator reflected on the round's own best. Under aggressive
        # `train.pruning` a policy that starts in another reward's basin
        # spends its few checkpoints climbing out; the hypothesis is that
        # the lineage parent's basin is the right one.
        #
        # Fallback chain, each step recorded in `source`: the parent's stored
        # policy -> whatever `warm_start_from_best` would load -> scratch.
        # Round 0 has no parent and falls through silently; a parent with no
        # policy behind it (untrained, screened, or evicted) is a warning,
        # because the search continues as the OTHER arm and only the artifact
        # says so.
        #
        # THE PARENT MUST BE FROM AN EARLIER ITERATION, like `similar`'s pool,
        # and for the same reason: under `generate.sampling_mode:
        # sequential_conditioned` sample i's parent is sample i-1 of the SAME
        # round, whose policy the sequential loop has already stored and a
        # forked child has not -- accepting it would make
        # `parallel != sequential`. A same-round parent falls through to
        # `best`, recorded, with the warning below.
        #
        # Eviction: `_store` is FIFO over `_STORE_LIMIT` (64) entries and every
        # trained candidate writes `policy:<cand_id>`; `_check_coherence`
        # refuses this mode when the run can train more than that, so within a
        # run nothing is ever evicted and the two schedules see the same store.
        parent = getattr(candidate, "parent_id", None) if candidate is not None else None
        iteration = int(getattr(candidate, "iteration", 0) or 0) if candidate is not None else 0
        ref = f"policy:{parent}" if parent else None
        earlier = parent in _POLICY_CODE and _POLICY_CODE[parent][0] < iteration
        if ref is not None and ref in _POLICY_STORE and earlier:
            return _InitPlan(ref, [], 0.0, "parent", parent)
        best = getattr(state, "policy_ref", None)
        if parent and not quiet:
            why = ("is a candidate of this same round (sequential_conditioned), which a "
                   "forked worker cannot see" if parent in _POLICY_CODE and not earlier
                   else "has no stored policy (untrained, evicted, or written by another process)")
            log.warning("train.init=warm_start_from_parent: %s's parent %s %s -- falling back to %s",
                        getattr(candidate, "cand_id", "?"), parent, why, best or "from_scratch")
        return _InitPlan(best, [], 0.0, *_resolved_source(best, "best"))
    if mode == "warm_start_from_similar":
        # NOT PUBLISHED (a BIRD extension for training efficiency).
        # Start from the stored policy whose REWARD is structurally most like
        # this candidate's (`selection._structural_similarity`, weighted Jaccard
        # over reward-function subtrees), whatever its lineage. The more
        # informative of the two: under a single-parent hill-climb
        # `warm_start_from_parent` IS `warm_start_from_best`, whereas the most
        # similar stored reward can be a sibling of the incumbent, or a
        # candidate from three rounds back that this one re-derived.
        #
        # THE POOL IS STRICTLY EARLIER ITERATIONS, and that is what keeps
        # `sequential` and `parallel` bit-identical: a forked child holds the
        # store as it was before any of this round trained, while the
        # sequential loop has already written this round's earlier siblings.
        # The identity holds only while nothing is EVICTED mid-round -- the
        # sequential loop's same-round writes would otherwise push the oldest
        # donors out of the 64-entry FIFO before candidate k plans, which a
        # forked child never sees (measured: identical for 66 trainings, then
        # six donors differ). `_check_coherence` therefore refuses this mode
        # when the run can train more than `_STORE_LIMIT` policies. The pool is also per
        # RESTART: `reset_policy_stores` empties the stores at each restart.
        # `cand_id` is `c%04d` (`Context.next_id`), so `sorted()` is candidate
        # index order and the strict `>` keeps the FIRST of a tie.
        from .selection import _structural_similarity  # local: selection imports training
        iteration = int(getattr(candidate, "iteration", 0) or 0) if candidate is not None else 0
        code = (getattr(candidate, "reward_code", "") or "") if candidate is not None else ""
        own = getattr(candidate, "cand_id", None)
        pool = sorted(cid for cid, (c_it, _) in _POLICY_CODE.items()
                      if c_it < iteration and cid != own
                      and f"policy:{cid}" in _POLICY_STORE)
        donor, donor_sim = None, -1.0
        for cid in pool:
            sim = _structural_similarity(code, _POLICY_CODE[cid][1])
            if sim > donor_sim:
                donor, donor_sim = cid, sim
        if donor is not None:
            return _InitPlan(f"policy:{donor}", [], 0.0, "similar", donor,
                             round(float(donor_sim), 6))
        best = getattr(state, "policy_ref", None)
        if iteration > 0 and not quiet:
            log.warning("train.init=warm_start_from_similar: %s finds no stored policy "
                        "from an earlier iteration (all evicted, or a resumed run) -- "
                        "falling back to %s", own or "?", best or "from_scratch")
        return _InitPlan(best, [], 0.0, *_resolved_source(best, "best"))
    if mode == "secondary_replay_buffer":
        ref = getattr(state, "replay_ref", None)
        buf = _REPLAY_STORE.get(ref or "", [])
        cap = int(cfg.get("train.secondary_buffer.size", 100000) or 100000)
        # GT: one buffer from the previous winner, shared by ALL candidates this
        # round -- so it is read from `state`, never from this candidate.
        return _InitPlan(None, _replay_tail(buf, cap),
                         float(cfg.get("train.secondary_buffer.ratio", 0.2) or 0.0),
                         "scratch", None)
    return _InitPlan(None, [], 0.0, "scratch", None)


def _error_router(detection: str, sink: io.StringIO) -> Callable[[BaseException], float]:
    """`train.failure_detection`: exception | stdout_grep.

    The two are not cosmetic and the cost difference is the point. Under
    `exception` a bad reward aborts the run at the first bad value, and the
    budget spent is whatever it got to. Under `stdout_grep` -- DrEureka's
    actual mechanism, which greps the RL job's log for `Traceback` / `running`
    -- the job keeps going with the bad value swallowed, so the failure is only
    discovered *after* the full training budget has been burned. Reproducing
    that is the only way an ablation on this key measures anything.
    """
    if detection == "stdout_grep":
        seen = [0]

        def swallow(exc: BaseException) -> float:
            # A reward that fails on every step would otherwise write one
            # traceback per env step; the grep only ever reads the first.
            seen[0] += 1
            if seen[0] <= _MAX_LOGGED_TRACEBACKS:
                traceback.print_exception(type(exc), exc, exc.__traceback__,
                                          limit=1, file=sink)
            return 0.0
        return swallow
    if detection == "exception_soft":  # internal: final rollouts, post-mortem
        def soft(exc: BaseException) -> float:
            return 0.0
        return soft

    def raise_it(exc: BaseException) -> float:
        raise exc
    return raise_it


def _grep_stdout(detection: str, text: str) -> str:
    """DrEureka's stdout protocol: a `Traceback` means dead, `running` means alive."""
    if detection != "stdout_grep" or not text:
        return ""
    if "Traceback" not in text and "Error" not in text:
        return ""
    first = next((ln for ln in text.splitlines() if "Error" in ln), "Traceback in training log")
    return f"stdout_grep: {first.strip()}"


def _opt_float(v: Any) -> Optional[float]:
    """`float(v)`, with None passed through -- for the one checkpoint column
    (`gt_return`) that is legitimately absent on a task with no reference reward."""
    return None if v is None else float(v)


def _mean_curve(curves: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Elementwise mean over seeds, truncated to the shortest (pruning cuts
    seeds at different rounds). Per-seed curves stay intact in `seed_metrics`,
    so §4 can aggregate checkpoints *then* seeds rather than the reverse.

    A column that is None on any seed (`gt_return` on a task with no reference
    reward) is None in the mean: an absent channel averaged with itself is
    still absent, and `float(None)` would have been a TypeError at the end of
    every seed's training rather than a number."""
    curves = [c for c in curves if c]
    if not curves:
        return []
    n = min(len(c) for c in curves)
    out: List[Dict[str, Any]] = []
    for i in range(n):
        keys = set().union(*(set(c[i]) for c in curves))
        point: Dict[str, Any] = {}
        for k in keys:
            vals = [c[i].get(k, 0.0) for c in curves]
            point[k] = (None if any(v is None for v in vals)
                        else float(np.mean([float(v) for v in vals])))
        out.append(point)
    return out


def _mean_components(per_seed: Sequence[Sequence[Dict[str, float]]]) -> Dict[str, List[float]]:
    per_seed = [c for c in per_seed if c]
    if not per_seed:
        return {}
    n = min(len(c) for c in per_seed)
    names: List[str] = []
    for c in per_seed:
        for point in c[:n]:
            for k in point:
                if k not in names:
                    names.append(k)
    return {name: [float(np.mean([float(c[i].get(name, 0.0)) for c in per_seed]))
                   for i in range(n)]
            for name in names}


# ==========================================================================
# registry: `train.backend`
# ==========================================================================


@register("train_backend", "mock")
def mock_backend(ctx: Any, state: Any, candidate: Candidate, n_seeds: int = 1,
                 env_steps: Optional[int] = None,
                 resume_ref: Optional[str] = None,
                 seed_phase: str = "",
                 handoff: Optional[SliceHandoff] = None) -> TrainResult:
    """Surrogate inner loop -- the default, and what the tester profile uses.

    Trains against the *candidate* reward and scores against
    the *ground truth*, so reward quality shows up as fitness rather than being
    stipulated. `train.env_steps` is capped hard (the paper points ask for
    1M-50M); the executed count, not the configured one, is what reaches the
    budget report, so a tester-profile run never claims paper-scale cost.

    Under `train.interaction: shared_population` the replay hand-off
    (`handoff.prefill_ref` / `.export_ref`) is IGNORED on this backend -- the
    slice resumes its policy and salts its seed, and the buffer is neither
    restored nor pooled. Stated in the seed row (`replay_prefill_supported:
    false`) and in the journal rather than refused: this is the tester tier,
    and refusing would make every LaRes tester run unloadable.
    """
    return _run_backend(ctx, state, candidate, n_seeds,
                        kind="auto", step_cap=MOCK_STEP_CAP, label="mock",
                        env_steps=env_steps, resume_ref=resume_ref, seed_phase=seed_phase,
                        handoff=handoff)


@register("train_backend", "tabular")
def tabular_backend(ctx: Any, state: Any, candidate: Candidate, n_seeds: int = 1,
                    env_steps: Optional[int] = None,
                    resume_ref: Optional[str] = None,
                    seed_phase: str = "",
                    handoff: Optional[SliceHandoff] = None) -> TrainResult:
    """Exact tabular Q-learning -- `configs/methods/singh_orp.yaml`.

    Singh, Lewis & Barto (2009) predate LLM reward design and searched a
    *tabular* reward space with an exact learner; that is the limit case this
    backend exists to make runnable. Exact only where the env declares
    `exact_states`; on a continuous env the table is a binning and the backend
    logs a warning rather than pretending otherwise.
    """
    return _run_backend(ctx, state, candidate, n_seeds,
                        kind="force_tabular", step_cap=TABULAR_STEP_CAP, label="tabular",
                        env_steps=env_steps, resume_ref=resume_ref, seed_phase=seed_phase,
                        handoff=handoff)


@register("train_backend", "sb3")
def sb3_backend(ctx: Any, state: Any, candidate: Candidate, n_seeds: int = 1,
                env_steps: Optional[int] = None,
                resume_ref: Optional[str] = None,
                seed_phase: str = "",
                handoff: Optional[SliceHandoff] = None) -> TrainResult:
    """stable-baselines3, imported lazily inside the call.

    `registry.load_all()` imports this module for *every* config in the repo,
    including `--dry-run` and `--validate-all`, so a module-level `import
    stable_baselines3` would make a machine without torch unable to validate a
    config it never intended to run. The import therefore happens here, on the
    one path that actually needs it.
    """
    try:
        from stable_baselines3 import PPO, SAC, TD3  # noqa: F401
        import gymnasium  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError(
            "train.backend: sb3 needs stable-baselines3 and gymnasium, which are not "
            "installed. Either\n"
            "    pip install 'stable-baselines3[extra]' gymnasium\n"
            "or run the same method point under the tester profile "
            "(--profile tester), which sets\n"
            "    train.backend: mock\n"
            f"(underlying import error: {exc})"
        ) from exc
    return _sb3_run(ctx, state, candidate, n_seeds, env_steps, resume_ref,
                    seed_phase=seed_phase, handoff=handoff)


@register("train_backend", "none")
def no_training_backend(ctx: Any, state: Any, candidate: Candidate, n_seeds: int = 1,
                        env_steps: Optional[int] = None,
                        resume_ref: Optional[str] = None,
                        seed_phase: str = "",
                        handoff: Optional[SliceHandoff] = None) -> TrainResult:
    """L2R: no policy learning. The reward is solved online by a planner.

    `train.algorithm: none` -- "MuJoCo MPC solves the reward online". Still
    produces trajectories, because §4 has to have something to look at and
    L2R's evaluation is entirely behavioural. The checkpoint curve
    is flat by construction; that is a true statement about a method that never
    learns, not a stub.
    """
    return _run_backend(ctx, state, candidate, n_seeds,
                        kind="planner", step_cap=PLANNER_STEP_CAP, label="none",
                        env_steps=env_steps, seed_phase=seed_phase,
                        handoff=handoff)


# ==========================================================================
# sb3 (only reached when the optional dependency is present)
# ==========================================================================
#
# `train.init` ON THIS BACKEND. `_sb3_run` writes `_POLICY_STORE` /
# `_REPLAY_STORE` and reads `_training_init` exactly as the surrogate driver
# does; without that, `warm_start_from_best` and `secondary_replay_buffer`
# would be inert on every backend a real config uses, and RDA and GT would
# silently run their own ablation. The four functions below carry it. The
# rule they follow is that a value in the
# store has to mean the same thing whichever backend put it there, so the
# ndarray-shaped wrappers above (`_PolicyBlob`, `_ReplaySlice`) are the format
# and there is no sb3-only store.


#: Algorithms with a replay buffer. `train.algorithm: ppo` is on-policy and has
#: none, so `secondary_replay_buffer` cannot mean anything there -- the same
#: situation `_make_learner` already logs for the linear-policy searcher.
_OFF_POLICY = frozenset({"sac", "qr_sac", "td3", "fasttd3"})


def _sb3_input_dim(model: Any) -> Optional[int]:
    """The width of the network's input layer, off the model's own obs space."""
    shape = getattr(getattr(model, "observation_space", None), "shape", None)
    if not shape:
        return None
    return int(np.prod(shape))


def _sb3_policy_blob(model: Any) -> Optional[_PolicyBlob]:
    """`model.save()` into memory. Excludes the replay buffer (SB3's default)."""
    try:
        buf = io.BytesIO()
        model.save(buf)
        return _PolicyBlob(buf.getvalue())
    except Exception as exc:  # noqa: BLE001 - a missing checkpoint is not a failed run
        log.warning("sb3: could not serialise the policy for train.init "
                    "(%s: %s); this candidate will carry no policy_ref",
                    type(exc).__name__, exc)
        return None


def _blend_sb3_params(inherited: Dict[str, Any], fresh: Dict[str, Any],
                      alpha: float) -> Dict[str, Any]:
    """`alpha * inherited + (1 - alpha) * fresh`, per floating-point tensor.

    Non-float entries keep the INHERITED value rather than being averaged:
    they are counters and integer buffers, not policy parameters, and an
    averaged step count is not a number anything should read. Keeping the
    inherited one is also what makes `alpha = 1.0` exactly `warm_start_from_best`
    -- the property the whole axis rests on, since that endpoint is supposed to
    BE RDA's mode. (SB3's MlpPolicy carries no such buffers today; this is a
    guard against a policy class that does, not a live path.)
    """
    out: Dict[str, Any] = {}
    for module, state in inherited.items():
        other = fresh.get(module)
        if not isinstance(state, dict) or not isinstance(other, dict):
            out[module] = state
            continue
        merged = {}
        for key, tensor in state.items():
            peer = other.get(key)
            if peer is None or not getattr(tensor, "is_floating_point", lambda: False)():
                merged[key] = tensor
            else:
                merged[key] = alpha * tensor + (1.0 - alpha) * peer
        out[module] = merged
    return out


def _sb3_apply_policy(model: Any, blob: Any, ref: str,
                      fusion_alpha: Optional[float] = None) -> bool:
    """`train.init: warm_start_from_best` -- load stored weights into `model`.

    Returns whether it took. A blob written by another backend, or by a run
    whose network shape differs, is a WARNING and a cold start, never a crash:
    a checkpoint that no longer fits must not destroy the search that found it.

    `fusion_alpha` is ROSKA's `fused_warm_start` (`methodology.tex:69`): the
    model's OWN freshly-initialised parameters are captured before the blob is
    loaded, and the two are blended afterwards. It must be threaded here and
    not only into the surrogate learners -- those build theta_0 in `__init__`
    and sb3 does not go near them, so a fusion that lived only there would make
    `fused_warm_start` bit-identical to `warm_start_from_best` on PPO, which is
    the backend ROSKA actually runs on. Two config points, one method.
    """
    if blob is None:
        return False
    if not isinstance(blob, np.ndarray) or blob.dtype != np.uint8:
        log.warning("train.init=warm_start_from_best: %s holds a %s, which is not an "
                    "sb3 policy (a surrogate-backend checkpoint?) -- training from "
                    "scratch", ref, type(blob).__name__)
        return False
    # theta_0, captured BEFORE the blob overwrites it. Deep-copied for
    # `_snapshot_sb3`'s reason: `get_parameters()` hands back tensors that
    # alias the live module, so a shallow copy would be overwritten by the
    # very `set_parameters` below and fuse the inherited policy with itself.
    import copy  # local, matching `_snapshot_sb3` -- this module has no top-level copy
    fresh = copy.deepcopy(model.get_parameters()) if fusion_alpha is not None else None
    try:
        model.set_parameters(io.BytesIO(np.asarray(blob, dtype=np.uint8).tobytes()),
                             exact_match=True, device=model.device)
        if fresh is not None:
            model.set_parameters(_blend_sb3_params(model.get_parameters(), fresh,
                                                   float(fusion_alpha)),
                                 exact_match=True, device=model.device)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("train.init=warm_start_from_best: %s did not load into this model "
                    "(%s: %s) -- training from scratch", ref, type(exc).__name__, exc)
        return False


def _blob_kind(blob: np.ndarray) -> str:
    """Which SERIALISER wrote a 1-D uint8 policy blob: `sb3`, `torch`, or `unknown`.

    The names are container formats and not learners, and the third return value
    is `torch` rather than a learner's name: TWO learners write the torch container
    (`fasttd3._fasttd3_blob` and `simba_v2._blob` both `torch.save` a dict of
    state dicts), so this function cannot name one of them without being wrong
    half the time. It answers "which rebuild path"; WHICH LEARNER is then read
    off the payload's own format tag by `policy_from_ref`, which branches on
    `simba_v2/` and `fasttd3/` and RAISES on anything else rather than falling
    through to a guess.

    Read off the CONTAINER'S OWN MEMBERS, never off shape and dtype: an sb3
    `_PolicyBlob` and a fasttd3 one are both 1-D uint8 and both zips (`torch.save`
    writes a zip, so even the `PK` magic agrees), and a shape test would route
    every fasttd3 agent into the sb3 rebuild -- a misroute, worse than a refusal
    (`tests/test_policy_from_ref_knows_fasttd3.py`). torch's container keeps
    everything under one top-level directory with `data.pkl` and a
    `.format_version` beside it (measured on a real fasttd3 blob); sb3's has
    top-level `data` and `policy.pth`. Neither needs torch to tell apart, which
    is why this lives here and the payload's own `format:` marker is checked
    next, in the `torch` branch of `policy_from_ref`.
    """
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(np.asarray(blob, dtype=np.uint8).tobytes())) as z:
            names = set(z.namelist())
    except (zipfile.BadZipFile, OSError, ValueError):
        return "unknown"
    if any(n.endswith("/data.pkl") for n in names) and \
            any(n.endswith("/.format_version") or n.endswith("/byteorder") for n in names):
        # `"torch"`, NOT `"fasttd3"`: `simba_v2._blob` writes the same
        # container, so this function's honest answer is the SERIALISER and the
        # learner comes from the payload's own `format` tag (`policy_from_ref`).
        # Naming it after one of its two producers would invite the sb3/fasttd3
        # misroute again one level up.
        return "torch"
    if "data" in names and any(n.endswith("policy.pth") for n in names):
        return "sb3"
    return "unknown"


def policy_from_ref(cfg: Any, env: Any, blob: Any,
                    observation_code: Optional[str] = None,
                    ref: str = "policy") -> Callable[[np.ndarray], np.ndarray]:
    """Rebuild a stored policy as `raw state -> action in the adapter's units`.

    `_POLICY_STORE` holds several kinds of blob under one key shape
    (`policy:<cand_id>`), and a reader that re-derived the kind for itself could
    know two and return NaN for the rest -- which would make DrEureka's RAPP
    sweep (`MetaWorld.dr_probe`) silently the uninformative-prior arm on the one
    backend that trains a real network. This is the ONE place that knows all
    FOUR, so a new reader cannot drift from it.

    Detected by the CONTAINER for the serialised learners and by shape for the
    surrogates -- never by shape and dtype alone for a 1-D uint8 blob: a fasttd3
    blob is also 1-D uint8 and also a zip, so a shape test would MATCH the sb3
    branch rather than refuse it (`_blob_kind`;
    `tests/test_policy_from_ref_knows_fasttd3.py`).

      1-D uint8, an sb3 archive          `_sb3_run`'s serialised model. Rebuilt with the
                                          algorithm and hyperparameters the run used
                                          (`_sb3_algo_and_hyper(cfg)` -- the same resolver
                                          `_sb3_run` and `ensure_bc_prior` share, for the
                                          reason its docstring gives: a model built under
                                          one `policy_kwargs` cannot load parameters saved
                                          under another), loaded with `_sb3_apply_policy`,
                                          predicted `deterministic=True`.
      1-D uint8, a torch container       `fasttd3._fasttd3_blob`'s agent. Rebuilt by
                                          `fasttd3.fasttd3_policy_from_blob`: the payload's
                                          `format: fasttd3/v1` marker and architecture
                                          checked by `_fasttd3_payload`, the actor and
                                          normaliser built by the loop's own constructor,
                                          the callable the loop's own `_greedy_policy`.
      `(n_disc_states, n_actions)`        `_QLearner`'s table -> `_greedy_policy`.
      `(action_dim, obs_dim + 1)`         `_CEMLearner`'s weights -> `_linear_policy`.

    `observation_code` is LIMEN's co-designed observation (`Candidate.observation_code`).
    When given, the shapes above are checked against an `_ObsView` -- phi's width and
    binning, exactly what `_install_observation` handed the learner -- and the SB3 network
    is fed `phi(s)`, mirroring `_sb3_run._features`. The returned callable ALWAYS takes a
    raw state and features it internally, so the caller's `reset`/`step`/`success` go on
    driving the unwrapped adapter (the split `EnvAdapter.policy_features` documents). A
    phi-width blob handed in WITHOUT its observation code therefore fails the shape check,
    which is the correct answer: nobody knows what that policy consumes.

    NEVER DEGRADES. `_sb3_apply_policy` returns `False` for a blob that does not fit
    because its caller, `train.init: warm_start_from_best`, is right to fall back to a cold
    start. Here there is no training after the load: the callable IS the measurement, and
    an untrained network wearing a stored policy's ref is the seed-fork fold's
    "corrupted measurement wearing a healthy one's clothes" (`_sb3_run`). So every path
    that cannot produce the policy the blob describes raises `ValueError` naming `ref`.

    torch and stable-baselines3 are imported inside the SB3 and fasttd3 branches only:
    `registry.load_all()` imports this module for every config, on machines that have
    neither; `_blob_kind` reads the zip's member list and needs nothing.
    """
    if blob is None:
        raise ValueError(f"{ref}: no policy blob to rebuild")

    phi: Optional[CompiledObservation] = None
    view = env
    if observation_code and str(observation_code).strip():
        phi = compile_observation(observation_code, env, language=reward_language(cfg))
        view = _ObsView(env, phi)

    if isinstance(blob, np.ndarray) and blob.dtype == np.uint8 and blob.ndim == 1:
        kind = _blob_kind(blob)
        if kind == "sb3":
            return _sb3_policy_from_blob(cfg, env, phi, blob, ref)
        if kind == "torch":
            # TWO LEARNERS WRITE THIS CONTAINER, and the container cannot tell
            # them apart. `fasttd3._fasttd3_blob` and `simba_v2._blob` both
            # `torch.save` a dict of state dicts, so `_blob_kind` sees the same
            # `*/data.pkl` + `*/.format_version` members for both -- the
            # distinction is the payload's OWN `format` tag, which each
            # backend's `_payload` already checks and would raise on. Reading
            # it HERE turns "handed to the wrong learner, which raises three
            # frames down naming an architecture mismatch" into a routing
            # decision, which is the same argument `_blob_kind` makes against a
            # shape test that would send every fasttd3 blob into the sb3
            # branch (`tests/test_policy_from_ref_knows_fasttd3.py`).
            from .simba_v2 import blob_format  # lazy: torch lives behind it

            # TWO BRANCHES AND A REFUSAL, never a fall-through. With `fasttd3`
            # as the else, an UNKNOWN tag -- or `""`, which `blob_format`
            # returns for any unreadable payload -- would be handed to fasttd3
            # and refused three frames down by ITS format check, naming
            # fasttd3's format and architecture for a blob that is neither
            # backend's. `blob_format`'s own docstring says "a blob whose tag it
            # does not know is refused rather than handed to the wrong learner",
            # and this is where that is kept: a comment contradicting the code
            # is worse than no comment.
            fmt = blob_format(blob)
            if fmt.startswith("simba_v2/"):
                from .simba_v2 import simba_v2_policy_from_blob

                return simba_v2_policy_from_blob(cfg, env, phi, blob, ref)
            if fmt.startswith("fasttd3/"):
                from .fasttd3 import fasttd3_policy_from_blob

                return fasttd3_policy_from_blob(cfg, env, phi, blob, ref)
            raise ValueError(
                f"{ref}: a torch container whose payload is tagged "
                f"{fmt or '<unreadable, or not a dict>'!r} -- no learner in this "
                "repo writes that tag (fasttd3/v1 and simba_v2/v1 are the two), "
                "so there is nothing here that can rebuild it. A blob from a "
                "future backend, or a corrupted one.")
        raise ValueError(
            f"{ref}: a 1-D uint8 blob of {int(blob.size)} bytes that is neither an sb3 "
            "archive (top-level `data` + `policy.pth`) nor a torch container "
            "(`*/data.pkl` + `*/.format_version`, the fasttd3/v1 shape) -- no learner "
            "this repo has can rebuild it")

    try:
        weights = np.asarray(blob, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ref}: holds a {type(blob).__name__}, which is neither an sb3 "
                         f"policy blob nor a weight matrix ({exc})") from exc
    n_disc = int(getattr(view, "n_disc_states", 0) or 0)
    q_shape = (n_disc, int(view.n_actions))
    w_shape = (int(view.action_dim), int(view.obs_dim) + 1)
    if weights.ndim == 2 and weights.shape == q_shape:
        return _greedy_policy(view, weights)
    if weights.ndim == 2 and weights.shape == w_shape:
        return _linear_policy(view, weights)
    what = "the co-designed observation" if phi is not None else "the raw state"
    raise ValueError(
        f"{ref}: a {weights.shape} array fits neither a Q-table {q_shape} nor a linear "
        f"policy {w_shape} over {what} on {getattr(env, 'name', type(env).__name__)}"
        + ("" if phi is not None else
           " (a policy trained on a co-designed observation needs its observation_code)"))


def _sb3_policy_from_blob(cfg: Any, env: Any, phi: Optional[CompiledObservation],
                          blob: Any, ref: str) -> Callable[[np.ndarray], np.ndarray]:
    """The SB3 branch of `policy_from_ref`: the run's architecture, the blob's weights.

    The model is built over a spaces-only gym env, the shape `ensure_bc_prior` uses: SB3
    sizes the network from `observation_space`/`action_space` and touches the env again
    only in `learn()`, which nothing here calls. The observation space is phi's under
    co-design -- `_sb3_run._Gym` does the same -- because that IS the network's input
    width, and `set_parameters(exact_match=True)` refuses a first layer of any other size.

    Built and predicted under `_single_thread_torch`, for the reason that context manager
    states: the children that trained this policy ran at one thread, so a rebuilt policy
    that predicted at the machine's full thread count would be the one part of the
    measurement whose numerics varied with the hardware.
    """
    if cfg is None:
        raise ValueError(
            f"{ref}: an sb3 policy blob can only be rebuilt with the run's config -- "
            "train.algorithm and train.hyperparameters name the network it was saved from, "
            "and no config was given")
    import gymnasium as gym
    from gymnasium import spaces

    algo, sb3_hyper, _anchor = _sb3_algo_and_hyper(cfg)
    # No learning happens on this model, so a replay buffer sized for learning is a
    # pure allocation: SAC's default `buffer_size` is 1M transitions, ~350 MB on a
    # Meta-World observation, paid on EVERY probe of a RAPP sweep. The buffer is not
    # part of `set_parameters` (torch state dicts only) and `predict` never reads it,
    # so shrinking it changes nothing the callable computes. Guarded on the
    # constructor's signature because on-policy classes do not take the argument.
    if "buffer_size" in inspect.signature(algo.__init__).parameters:
        sb3_hyper = {**sb3_hyper, "buffer_size": 1}
    continuous = getattr(env, "exact_states", None) is None
    obs_lo, obs_hi = ((phi.low, phi.high) if phi is not None
                      else (env.obs_low, env.obs_high))

    class _Spaces(gym.Env):  # pragma: no cover - needs the optional dependency
        """Only the spaces; SB3 builds the network from them and resets nothing until
        `learn()`, which a rebuilt policy never calls."""
        metadata: Dict[str, Any] = {}

        def __init__(self) -> None:
            self.observation_space = spaces.Box(np.asarray(obs_lo, dtype=np.float32),
                                                np.asarray(obs_hi, dtype=np.float32))
            self.action_space = (spaces.Box(np.asarray(env.action_low, dtype=np.float32),
                                            np.asarray(env.action_high, dtype=np.float32))
                                 if continuous else spaces.Discrete(env.n_actions))

        def reset(self, *, seed=None, options=None):
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):
            raise RuntimeError("a rebuilt policy's env is never stepped")

    def _features(s: Any) -> np.ndarray:
        # `_sb3_run._features`, verbatim: phi(s) under co-design, the raw state otherwise.
        return np.asarray(phi(s) if phi is not None else s, dtype=np.float32)

    with _single_thread_torch():
        model = algo("MlpPolicy", _Spaces(), seed=int(cfg.get("seed", 0) or 0),
                     **sb3_hyper)
        if not _sb3_apply_policy(model, blob, ref):
            # `_sb3_apply_policy` has already logged WHY at warning level (a blob from
            # another backend, or a network of another shape). The False is a cold
            # start for a warm start's caller; here it would be an untrained network
            # reported as the stored policy, so it is a refusal.
            raise ValueError(
                f"{ref}: the stored sb3 policy did not load into a fresh "
                f"{algo.__name__} built from this config (see the warning above for "
                "the loader's reason); refusing to hand back untrained weights")

    def policy(s: np.ndarray) -> np.ndarray:
        with _single_thread_torch():
            act, _ = model.predict(_features(s), deterministic=True)
        return (np.asarray(act, dtype=float).ravel() if continuous
                else np.asarray(env.action_set[int(act)], dtype=float))

    return policy


def _sb3_box_policy(model: Any) -> Optional[Any]:
    """`model.policy` when the model acts in a Box space, else None.

    The one condition under which SB3 scales actions into its replay buffer
    (`off_policy_algorithm._sample_action`); a Discrete space is stored as is.
    """
    try:
        from gymnasium import spaces
    except ImportError:  # pragma: no cover - sb3 brings gymnasium
        return None
    policy = getattr(model, "policy", None)
    if policy is None or not isinstance(getattr(model, "action_space", None), spaces.Box):
        return None
    return policy


def _sb3_policy_params(blob: Any, ref: str, device: Any = "cpu") -> Optional[Dict[str, Any]]:
    """The `policy` state dict stored in an sb3 blob, read WITHOUT loading it anywhere.

    `_sb3_apply_policy` is `train.init`'s: it loads a blob INTO the learner.
    This only reads one, with the same loader `set_parameters` uses, so a
    caller can hold the elite's parameters beside a model that is not the
    elite. `None` (with a warning) for a blob this loader cannot read -- a
    surrogate checkpoint, or no policy in it.
    """
    if blob is None:
        return None
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        raw = io.BytesIO(np.asarray(blob, dtype=np.uint8).tobytes())
        _data, params, _vars = load_from_zip_file(raw, load_data=False, device=device)
    except Exception as exc:  # noqa: BLE001
        log.warning("train.elite_constraint.kind=l2_params: could not read the elite's "
                    "parameters from %s (%s: %s)", ref, type(exc).__name__, exc)
        return None
    sd = params.get("policy") if isinstance(params, dict) else None
    if not isinstance(sd, dict) or not sd:
        log.warning("train.elite_constraint.kind=l2_params: %s holds no policy parameters", ref)
        return None
    return sd


def _sb3_attach_elite_constraint(model: Any, weight: float, blob: Any, ref: str) -> int:
    """LaRes Eq. 4 anchored to the ELITE, whatever `model` currently holds.

    `population.attach_l2_constraint` snapshots the model's parameters as its
    reference, so this loads the elite's stored policy parameters into the
    model for exactly the duration of that call and then puts the model's own
    back -- `load_state_dict` copies VALUES into the same `Parameter` objects,
    so the snapshot holds theta_elite while the live parameters are the
    slice's again, and the term computes `2w (theta - theta_elite)` from then
    on. On an unsliced warm start the model already holds the elite and the
    swap is a no-op, bit for bit. Under `train.interaction:
    shared_population` a resumed slice holds its OWN previous slice, so a
    reference snapshotted from the model as it stands would pull Eq. 4 toward
    that slice rather than toward the elite.

    The reference is the round-start elite -- `state.policy_ref`, which only
    stage 6 moves -- fixed for every slice of the round, as the release's
    workers load `elite_actor` / `elite_q1` / `elite_q2` from `best_*.pth`
    once per evolution and hold them (`refs/code/LaRes/utils.py:2119-2125`).

    Returns the number of optimisers patched; 0 means the term is NOT running
    and the reason has been logged (unreadable blob, or a blob that does not
    hold this policy's parameters in its shape -- refused rather than pulled
    toward the parameters that happen to match).
    """
    ref_sd = _sb3_policy_params(blob, ref, getattr(model, "device", "cpu"))
    if ref_sd is None:
        return 0
    policy = getattr(model, "policy", None)
    if policy is None:
        return 0
    own = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    try:
        policy.load_state_dict(ref_sd)
    except Exception as exc:  # noqa: BLE001
        # torch copies what fits before raising on what does not, so put the
        # model's own parameters back before saying the term is not running.
        policy.load_state_dict(own)
        log.warning("train.elite_constraint.kind=l2_params: %s does not hold this policy's "
                    "parameters (%s: %s); the term is NOT running", ref, type(exc).__name__,
                    str(exc).splitlines()[0] if str(exc) else "")
        return 0
    try:
        from .population import attach_l2_constraint
        return attach_l2_constraint(model, weight)
    finally:
        policy.load_state_dict(own)


def _sb3_stored_actions(model: Any, act: Any, n: int, action_dim: int, dtype: Any) -> np.ndarray:
    """ENV-space actions (`_sb3_export_replay`'s convention) -> what the learner's buffer holds.

    SB3's own [-1, 1] convention for a Box space, which is what its critic was
    trained on (`off_policy_algorithm._sample_action` stores `scaled_action`);
    verbatim for a Discrete space. The ONE place a stored row's action is put
    back into SB3's convention, shared by GT's secondary buffer
    (`_sb3_secondary_buffer`) and LaRes's prefill (`_sb3_prefill_replay`) for
    `_relabel_rows`' reason: two copies drift. Had the secondary buffer scaled
    back and the prefill not, the LaRes pool (env-space since the export
    unscales) would be written verbatim into SAC's PRIMARY buffer on every
    non-+/-1 env (pendulum +/-2, `gym_inverted_pendulum_balance` +/-3).
    """
    stored = np.asarray(act, dtype=np.float32).reshape(n, action_dim)
    if _sb3_box_policy(model) is not None:
        stored = np.asarray(model.policy.scale_action(stored), dtype=np.float32)
    return stored.astype(dtype, copy=False)


def _sb3_export_replay(model: Any, cap: int) -> Optional[_ReplaySlice]:
    """The last `cap` transitions of `model.replay_buffer`, RAW.

    No rewards, for the reason `_QLearner._replay_secondary` states: GT hands
    the previous winner's experience to the next round's candidates, and each
    of them must relabel it with ITS OWN reward. Storing the reward would leak
    the incumbent's reward function into every challenger's fitness, which is
    the one thing a shared buffer must not do.

    `done` is the true terminal flag with SB3's timeout mask already applied
    (`dones * (1 - timeouts)`) -- the same expression `ReplayBuffer._get_samples`
    uses, so a transition means here what it means when SB3 trains on it.

    `act` is in ENV space. SB3's off-policy collector stores the [-1, 1]-SCALED
    action for a Box space (`off_policy_algorithm._sample_action`:
    `buffer_action = scaled_action`; the env is handed
    `policy.unscale_action(scaled_action)`), so `rb.actions` copied verbatim
    would hold actions the candidate reward was never written for. That is
    the identity on MT10 and acrobot (both +/-1), which hides it; on pendulum
    (+/-2) the relabelled reward for a torque-squared candidate would be a
    quarter of the right one. Unscaling here is what makes the slice
    mean the same thing `_QLearner.seen` means -- the action the env took --
    whichever backend wrote it; `_sb3_secondary_buffer` scales back for the
    learner, whose critic expects SB3's convention.
    """
    rb = getattr(model, "replay_buffer", None)
    if rb is None or not hasattr(rb, "observations"):
        return None
    size = int(rb.buffer_size)
    n_filled = size if getattr(rb, "full", False) else int(rb.pos)
    if n_filled <= 0:
        return None
    # Chronological, oldest first, then keep the tail.
    order = np.arange(int(rb.pos) - n_filled, int(rb.pos)) % size
    order = order[-max(1, int(cap)):]
    obs = np.array(rb.observations[order, 0], dtype=np.float32, copy=True)
    if getattr(rb, "optimize_memory_usage", False):
        nxt = np.array(rb.observations[(order + 1) % size, 0], dtype=np.float32, copy=True)
    else:
        nxt = np.array(rb.next_observations[order, 0], dtype=np.float32, copy=True)
    act = np.array(rb.actions[order, 0], copy=True)
    if _sb3_box_policy(model) is not None:
        act = np.asarray(model.policy.unscale_action(act), dtype=np.float32)
    timeouts = getattr(rb, "timeouts", None)
    term = rb.dones[order, 0]
    if timeouts is not None:
        term = term * (1.0 - timeouts[order, 0])
    return _ReplaySlice(obs, act, nxt, np.asarray(term, dtype=bool))


def _sb3_secondary_buffer(secondary: Any, ratio: float, model: Any, env: Any,
                          reward: CompiledReward, norm_mode: str,
                          continuous: bool) -> Optional[Any]:
    """Build the relabelled secondary buffer GT's §4 describes, or None.

    Relabelling is done ONCE here rather than at sample time, because a reward
    recomputed per batch would make the number of gradient steps change the
    reward each transition carries. `_RewardNorm` is a FRESH instance over this
    stream: every sb3-tier config sets `train.reward_norm: none` so it is the
    identity today, and under `clip` it is stateless; under `running_std` the
    secondary transitions form their own normalisation stream, which is stated
    here because it is a choice and not a derivation.
    """
    from stable_baselines3.common.buffers import ReplayBuffer

    cols = _replay_columns(secondary)
    if cols is None or ratio <= 0.0:
        return None
    obs, act, nxt, done = cols
    n = int(obs.shape[0])
    if n <= 0:
        return None

    rew = _relabel_rows(obs, act, nxt, env, reward, norm_mode, continuous)

    buf = ReplayBuffer(n, model.observation_space, model.action_space,
                       device=model.device, n_envs=1,
                       handle_timeout_termination=False)
    # Filled by assignment rather than `n` calls to `add()`: same arrays, and
    # `add()` on 100,000 rows is ~4 s of Python per candidate per seed.
    buf.observations[:, 0] = obs.astype(buf.observations.dtype, copy=False)
    buf.next_observations[:, 0] = nxt.astype(buf.next_observations.dtype, copy=False)
    # The slice holds ENV-space actions (`_sb3_export_replay`); the reward
    # above was computed on them, and the learner's buffer gets them back in
    # SB3's own [-1, 1] convention, which is what its critic was trained on.
    buf.actions[:, 0] = _sb3_stored_actions(model, act, n, buf.action_dim, buf.actions.dtype)
    buf.rewards[:, 0] = rew
    buf.dones[:, 0] = np.asarray(done, dtype=np.float32)
    buf.pos, buf.full = 0, True
    return buf


def _relabel_rows(obs: np.ndarray, act: np.ndarray, nxt: np.ndarray, env: Any,
                  reward: CompiledReward, norm_mode: str, continuous: bool) -> np.ndarray:
    """The ONE relabeller: raw `(s, a, s')` rows -> this candidate's reward.

    Shared by GT's secondary buffer (`_sb3_secondary_buffer`) and LaRes's slice
    prefill (`_sb3_prefill_replay`) so the two mechanisms cannot drift apart on
    how a stored transition is re-scored. Errors route through the same soft
    router the live rollouts use, and `_RewardNorm` is a FRESH instance over
    this stream -- identity under `none`, stateless under `clip`, its own
    stream under `running_std` (a choice, stated, not a derivation).
    """
    n = int(obs.shape[0])
    sink = io.StringIO()
    on_error = _error_router("exception_soft", sink)
    relabel = _RewardNorm(norm_mode)
    rew = np.zeros(n, dtype=np.float32)
    for i in range(n):
        a = np.asarray(act[i], dtype=float).ravel() if continuous \
            else env.action_set[int(np.asarray(act[i]).ravel()[0])]
        try:
            raw, _cv = reward(obs[i], a, nxt[i])
            if not math.isfinite(raw):
                raise FloatingPointError(f"reward returned {raw!r}")
        except Exception as exc:  # noqa: BLE001
            raw = on_error(exc)
        rew[i] = relabel(float(raw))
    return rew


def _sb3_prefill_replay(model: Any, rows: Any, env: Any, reward: CompiledReward,
                        norm_mode: str, continuous: bool) -> int:
    """LaRes §4.3: fill `model.replay_buffer` with `rows`, relabelled, BEFORE learning.

    Into the PRIMARY buffer, not `_mixed_buffer_class`'s secondary, and the
    difference is the paper's semantics. The release keeps one `All_buffer`
    that every agent's transitions go into and every agent samples UNIFORMLY
    from, reading its own reward column (`refs/code/LaRes/replay_buffer.py:39-42`
    `sample(batch_size, index)` -> `reward_list[index]`; `sac.py:421`). There is
    no mixing ratio to pin, so a ratio here would be a fabricated one; pre-filling
    makes SB3's own uniform sampling run over pool ∪ this slice's new rows, which
    is the release exactly. (GT's `secondary_buffer.ratio` IS a paper pin, which
    is why that path mixes instead -- see `_mixed_buffer_class`.)

    `rows` are RAW `(s, a, s', done)` -- the reward is recomputed here under THIS
    candidate's reward, which is the relabelling the paper describes ("each
    experience needs to be relabeled by the reward function population", §4.3)
    and the no-leak property `_sb3_export_replay` states. Rows beyond the
    buffer's capacity are dropped OLDEST first, as the release's ring does.
    `done` already carries SB3's timeout mask (see `_sb3_export_replay`), so
    `timeouts` is zeroed for the prefilled rows. Their actions are ENV space
    (that function unscales SB3's [-1, 1] buffer convention on export), so
    `_relabel_rows` scores the action the env took and
    `_sb3_stored_actions` puts SB3's scaled form back into `rb.actions`.

    `learning_starts` is paid ONCE per arm per round: SB3 gates gradient
    updates and random warm-up actions on `num_timesteps < learning_starts`
    (`off_policy_algorithm.py`, `_sample_action` / `learn`), and a fresh model
    starts at 0, so without this a resumed slice would re-spend the warm-up on
    a buffer that already holds it. The remainder, if any, is what a slice with
    a tiny prefill still owes. Recorded as `learning_starts_effective`.

    Returns the number of rows written (0 = nothing prefilled, and why is
    logged).
    """
    rb = getattr(model, "replay_buffer", None)
    if rb is None or not hasattr(rb, "observations"):
        return 0
    if getattr(rb, "optimize_memory_usage", False):
        # Under `optimize_memory_usage` SB3 stores s' as the NEXT row's s, which
        # only holds for one contiguous stream; a pool of several arms' rows is
        # not one. Refuse loudly rather than write next-states that are wrong.
        log.warning("shared_population: replay prefill is not supported with "
                    "train.hyperparameters.optimize_memory_usage=true; this slice "
                    "starts from an empty buffer")
        return 0
    cols = _replay_columns(rows)
    if cols is None:
        return 0
    obs, act, nxt, done = cols
    size = int(rb.buffer_size)
    n = int(obs.shape[0])
    if n <= 0:
        return 0
    if n > size:
        obs, act, nxt, done = obs[-size:], act[-size:], nxt[-size:], done[-size:]
        n = size
    rew = _relabel_rows(obs, act, nxt, env, reward, norm_mode, continuous)
    rb.observations[:n, 0] = obs.astype(rb.observations.dtype, copy=False)
    rb.next_observations[:n, 0] = nxt.astype(rb.next_observations.dtype, copy=False)
    # `act` is ENV space (the pool is `_sb3_export_replay` slices); `_relabel_rows`
    # above scored it as such, and the PRIMARY buffer gets SB3's scaled form.
    rb.actions[:n, 0] = _sb3_stored_actions(model, act, n, rb.action_dim, rb.actions.dtype)
    rb.rewards[:n, 0] = rew
    rb.dones[:n, 0] = np.asarray(done, dtype=np.float32)
    timeouts = getattr(rb, "timeouts", None)
    if timeouts is not None:
        timeouts[:n, 0] = 0.0
    rb.pos = n % size
    rb.full = bool(n >= size)
    owed = max(0, int(getattr(model, "learning_starts", 0) or 0) - n)
    model.learning_starts = owed
    return n


_MIXED_BUFFER_CLASS: Optional[type] = None


def _mixed_buffer_class() -> type:
    """`ReplayBuffer` that draws `secondary_ratio` of every batch elsewhere.

    Defined lazily, and cached, because subclassing it needs sb3 imported and
    this module is imported by `registry.load_all()` on machines that have no
    torch.

    This mirrors `_QLearner._replay_secondary` rather than pre-seeding the
    primary buffer, and the difference is not cosmetic: seeding makes the
    secondary's share decay as the primary fills, so `secondary_buffer.ratio`
    -- GT App. E Table 3's "secondary replay buffer sampling ratio" -- would
    only be the ratio for the first few thousand steps of a 1M-step run.
    """
    global _MIXED_BUFFER_CLASS
    if _MIXED_BUFFER_CLASS is not None:
        return _MIXED_BUFFER_CLASS

    import torch as th
    from stable_baselines3.common.buffers import ReplayBuffer

    class _SecondaryMixReplayBuffer(ReplayBuffer):
        #: Set after construction: SB3 builds the buffer itself, from
        #: `replay_buffer_kwargs`, and those have to survive `model.save()`.
        secondary_buffer: Any = None
        secondary_ratio: float = 0.0

        def sample(self, batch_size: int, env: Any = None):  # noqa: D102
            sec = self.secondary_buffer
            n2 = int(round(self.secondary_ratio * batch_size)) if sec is not None else 0
            n2 = max(0, min(batch_size, n2))
            if n2 == 0:
                return super().sample(batch_size, env=env)
            b = sec.sample(n2, env=env)
            if n2 >= batch_size:
                return b
            a = super().sample(batch_size - n2, env=env)
            # Field-count agnostic: SB3 2.9 added `discounts`, which is None
            # unless an n-step buffer filled it, and a positional rebuild that
            # assumed five fields would have silently dropped it.
            merged = [None if x is None or y is None else th.cat([x, y], dim=0)
                      for x, y in zip(a, b)]
            return type(b)(*merged)

    # `model.save()` cloudpickles `self.replay_buffer_class` as part of the data
    # dict. A class defined inside a function is pickled BY VALUE, which both
    # bloats every `_PolicyBlob` and makes the blob depend on this function's
    # closure; naming it here makes it resolvable by reference instead.
    # MEASURED: with the name set, a blob from a mixed-buffer model is 3009472 B
    # against 3010801 B for a plain one -- by reference, not by value. Resolving
    # that reference needs `_mixed_buffer_class()` to have run in the loading
    # process, which `_sb3_apply_policy` never requires: `set_parameters` reads
    # the zip with `load_data=False` and never unpickles the data dict at all.
    _SecondaryMixReplayBuffer.__module__ = __name__
    _SecondaryMixReplayBuffer.__qualname__ = "_SecondaryMixReplayBuffer"
    globals()["_SecondaryMixReplayBuffer"] = _SecondaryMixReplayBuffer
    _MIXED_BUFFER_CLASS = _SecondaryMixReplayBuffer
    return _MIXED_BUFFER_CLASS


def _sb3_algo_and_hyper(cfg: Any) -> Tuple[type, Dict[str, Any], str]:
    """`train.algorithm` -> the SB3 class, `train.hyperparameters` filtered to what it
    accepts, and the anchor kind that will actually be applied.

    Shared by `_sb3_run` and `ensure_bc_prior`: a clone built under one
    `policy_kwargs` and loaded into a model built under another fails
    `set_parameters(exact_match=True)`, so both must resolve the same way.

    `device` defaults to "cpu". Measured on an H200 node, 20,000 Pendulum steps
    with SAC: 99 steps/s on CPU against 137 on CUDA. So the GPU IS faster --
    about 1.4x -- but only 1.4x, because MlpPolicy is two small dense layers
    and the per-step host<->device copies eat most of what the kernels win.

    CPU is still the right default, for two reasons that are not speed. It runs
    everywhere, including the machines where `--validate-all` and the test suite
    run and no CUDA exists. And on a cluster a 1.4x per-run gain does not repay
    serialising a sweep behind the handful of free GPUs when CPU cores are
    plentiful: 32 concurrent CPU tasks at 99 steps/s beat 14 GPU tasks at 137.

    Pin `train.hyperparameters: {device: cuda}` when a single run's latency is
    what you care about, or when a policy grows enough to repay the transfer.

    `train.anchor.kind: kl_clone` swaps PPO for `anchored_ppo`'s subclass, which
    is PPO with the penalty term and nothing else; on any other algorithm the
    anchor is not implemented, which `_check_coherence` refuses at load and this
    function repeats as a warning for a config that reached it anyway.
    """
    from stable_baselines3 import PPO, SAC, TD3
    name = str(cfg.get("train.algorithm", "ppo"))
    algos = {"ppo": PPO, "sac": SAC, "td3": TD3, "qr_sac": SAC}
    if set(algos) != set(SB3_ALGORITHMS):
        # Not an `assert`: `python -O` strips those, and this guards a refusal.
        raise RuntimeError(
            "training._sb3_algo_and_hyper's learner map and config.SB3_ALGORITHMS "
            f"disagree ({sorted(algos)} vs {sorted(SB3_ALGORITHMS)}); the config "
            "constant is the one copy of the list the coherence rule reads")
    if name not in algos:
        # STRICT. A `.get(name, PPO)` fallback would train PPO for
        # `train.env_steps` under `train.algorithm: none` (l2r) or `q_learning`
        # (singh_orp) on a profile pinning `train.backend: sb3`, while the seed
        # row and the resolved config cited an algorithm that trains nothing.
        # `_check_coherence` refuses the pairing at
        # load; this is the backstop for a config that reached the backend by
        # some path validate() did not see.
        raise ConfigError(
            f"train.algorithm={name!r} has no learner on train.backend=sb3 "
            f"(sb3 implements {', '.join(SB3_ALGORITHMS)}); refusing rather than "
            f"silently training PPO under that citation")
    algo = algos[name]
    anchor = str(cfg.get("train.anchor.kind", "none") or "none")
    if anchor == "kl_clone":
        if algo is PPO:
            from .anchored_ppo import make_kl_ppo_class
            algo = make_kl_ppo_class()
        else:
            log.warning("train.anchor.kind=kl_clone is implemented for train.algorithm=ppo "
                        "only; %s trains without the anchor (kl_reward is the "
                        "algorithm-agnostic form)", name)
            anchor = "none"
    # `kl_reward` is applied to the reward inside `_Gym.step` and needs no subclass:
    # every algorithm sees a shaped reward and nothing else changes.
    # `train.hyperparameters` reaches SB3 too, through the same filter the
    # surrogate learners use, so the key means the same thing on every backend
    # and a name no algorithm accepts is warned about rather than dropped.
    # `verbose` is a DEFAULT here, not a literal at the construction sites.
    # Passed as `verbose=0` ahead of `**sb3_hyper` at the `algo(...)` calls it
    # would carry exactly the duplicate-argument hazard the `reserved` loop
    # below exists to prevent -- without being on the list:
    # `-s train.hyperparameters.verbose=1` would not turn SB3's progress on but
    # kill the run at construction with `TypeError: got multiple values for
    # keyword argument 'verbose'`. Seeding the default BEFORE the update lets a
    # configured value win, and with nothing set it is 0.
    sb3_hyper: Dict[str, Any] = {"device": "cpu", "verbose": 0}
    sb3_hyper.update(_learner_kwargs(algo, cfg.get("train.hyperparameters") or {}, f"sb3/{name}"))
    if anchor == "kl_clone" and "target_kl" not in sb3_hyper and algo.__name__ == "KLPPO":
        # PPO's own per-update early stop, part of the measured bundle: it bounds
        # how far one update can move the policy whatever the advantages say.
        sb3_hyper["target_kl"] = cfg.get("train.anchor.target_kl", 0.02)
    # `policy`, `env` and `seed` are passed positionally below; letting a config
    # also supply them would raise TypeError for a duplicate argument, and
    # `seed` in particular is owned by `_seed_base` so that a run is
    # reproducible. Dropping them here keeps the failure a warning, not a crash.
    for reserved in ("policy", "env", "seed", "_init_setup_model"):
        if sb3_hyper.pop(reserved, None) is not None:
            log.warning("train.hyperparameters: %r is set by the harness (policy/env "
                        "positionally, seed by train.seeds_per_candidate) and was ignored",
                        reserved)
    return algo, sb3_hyper, anchor


def _or_default(value: Any, default: Any, cast: Callable[[Any], Any]) -> Any:
    """`cast(value)`, or `default` when the config key is null. NOT `value or default`:
    that turns an explicit `0` / `0.0` into the default silently."""
    return default if value is None else cast(value)


def ensure_bc_prior(ctx: Any, state: Any, cfg: Any) -> None:
    """Build `train.init: bc_prior`'s clone once per process, in the PARENT.

    Called at the top of both `candidate_parallelism` schedules, before any fork,
    so every worker inherits one blob rather than each rebuilding it (the fork
    copies `_POLICY_STORE`; a child's own write never comes home). Re-called every
    iteration: the blob is re-inserted so the store's FIFO eviction never reaches
    it, and rebuilt -- deterministically, from the same seed -- if it ever did.

    Only the sb3 backend can build one (torch). Anywhere else the ref resolves to
    nothing and `_training_init`'s reader warns and trains from scratch, exactly
    the degradation `warm_start_from_best` has on those backends.
    """
    if cfg.get("train.init", "from_scratch") not in ("bc_prior", "bc_prior_then_warm_start"):
        return
    if cfg.get("train.backend") != "sb3":
        if not getattr(ctx, "_bc_prior_warned", False):
            log.warning("train.init=bc_prior needs train.backend=sb3 to build the clone; "
                        "%s trains from scratch", cfg.get("train.backend"))
            try:
                ctx._bc_prior_warned = True
            except Exception:  # noqa: BLE001 - a frozen ctx just warns each time
                pass
        return
    blob = _POLICY_STORE.pop(BC_PRIOR_REF, None)
    if blob is not None:
        _store(_POLICY_STORE, BC_PRIOR_REF, blob, _POLICY_STORE_MAX_BYTES)
        return
    import gymnasium as gym
    from gymnasium import spaces
    from .anchored_ppo import (FALLBACK_KEEP, FALLBACK_MAX_EPISODES, FALLBACK_MIN_METRIC,
                               FALLBACK_OVERSAMPLE, build_prior, find_registry_policy)

    env = ctx.env
    if getattr(env, "exact_states", None) is not None:
        raise ValueError("train.init=bc_prior needs a continuous-action env (the clone is "
                         "a Gaussian actor); this one is discrete")
    policy_id = find_registry_policy(cfg["problem.env_id"], cfg.get("train.bc_prior.policy", "auto"))
    algo, sb3_hyper, _anchor = _sb3_algo_and_hyper(cfg)

    class _Spaces(gym.Env):  # pragma: no cover - needs the optional dependency
        """Only the spaces: SB3 builds the network from them and resets nothing
        until `learn()`, which the prior never calls."""
        metadata: Dict[str, Any] = {}

        def __init__(self) -> None:
            self.observation_space = spaces.Box(np.asarray(env.obs_low, dtype=np.float32),
                                                np.asarray(env.obs_high, dtype=np.float32))
            self.action_space = spaces.Box(np.asarray(env.action_low, dtype=np.float32),
                                           np.asarray(env.action_high, dtype=np.float32))

        def reset(self, *, seed=None, options=None):
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        def step(self, action):
            raise RuntimeError("the prior's env is never stepped")

    seed = int(cfg.get("seed", 0) or 0)
    t0 = time.monotonic()
    model = algo("MlpPolicy", _Spaces(), seed=seed, **sb3_hyper)
    record = build_prior(env, model, policy_id=policy_id,
                         n_demos=int(cfg.get("train.bc_prior.n_demos", 25)),
                         epochs=int(cfg.get("train.bc_prior.epochs", 400)),
                         success_margin=int(cfg.get("train.bc_prior.success_margin", 25)),
                         seed=seed,
                         accept=cfg.get("train.bc_prior.accept", "success"),
                         min_metric=cfg.get("train.bc_prior.accept_min_metric"),
                         # null = the module constant (the key is null under every rule but
                         # its own, see _check_coherence); `is None`, not `or`, so an explicit
                         # 0 / 0.0 under the rule that reads it is 0 and not the default
                         fallback_oversample=_or_default(cfg.get("train.bc_prior.fallback_oversample"),
                                                         FALLBACK_OVERSAMPLE, int),
                         fallback_keep=_or_default(cfg.get("train.bc_prior.fallback_keep"),
                                                   FALLBACK_KEEP, float),
                         fallback_min_metric=_or_default(cfg.get("train.bc_prior.fallback_min_metric"),
                                                         FALLBACK_MIN_METRIC, float),
                         fallback_max_episodes=_or_default(cfg.get("train.bc_prior.fallback_max_episodes"),
                                                           FALLBACK_MAX_EPISODES, int))
    # HOW STRONG IS THE CLONE? `clone_eval_episodes` greedy episodes of the clone
    # itself, before any RL, scored by the adapter's task_metric and success --
    # ground truth used as a MEASUREMENT of the start policy, recorded in
    # bc_prior.json and nowhere else (no prompt, no judge, no selection reads it).
    # It is what lets a BC-strength setting be described by a number rather than by
    # the name of its demo filter.
    n_eval = int(cfg.get("train.bc_prior.clone_eval_episodes", 0) or 0)
    if n_eval > 0:
        try:
            eval_rng = np.random.default_rng(int(cfg.get("seed", 0) or 0) + 424242)
            def _clone_policy(s: np.ndarray, _m=model) -> np.ndarray:
                act, _ = _m.predict(np.asarray(s, dtype=np.float32), deterministic=True)
                return np.asarray(act, dtype=float).ravel()
            mets, succ, lens = [], [], []
            for _ in range(n_eval):
                traj, _steps, _gt = _rollout(env, _clone_policy, eval_rng, None,
                                             _error_router("exception_soft", io.StringIO()))
                mets.append(float(env.task_metric(traj))); succ.append(bool(env.success(traj)))
                lens.append(int(traj.length))
            record["clone_eval"] = {"episodes": n_eval, "task_metric_mean": float(np.mean(mets)),
                                    "task_metric_median": float(np.median(mets)),
                                    "task_metric_max": float(np.max(mets)),
                                    "success_rate": float(np.mean(succ)),
                                    "task_metrics": [round(m, 4) for m in mets],
                                    "episode_lengths": lens}
            log.info("bc_prior: clone scores task_metric %.3f mean / %.3f max, success %.2f over %d "
                     "episode(s) (measurement only)", record["clone_eval"]["task_metric_mean"],
                     record["clone_eval"]["task_metric_max"], record["clone_eval"]["success_rate"], n_eval)
        except Exception as exc:  # a broken eval must not cost the run its clone
            log.warning("bc_prior: clone_eval failed (%s: %s); recorded without it",
                        type(exc).__name__, exc)
            record["clone_eval"] = {"error": f"{type(exc).__name__}: {exc}"}
    if hasattr(model, "detach_anchor"):
        model.detach_anchor()
    blob = _sb3_policy_blob(model)
    if blob is None:
        raise RuntimeError("train.init=bc_prior: the clone could not be serialised")
    _store(_POLICY_STORE, BC_PRIOR_REF, blob, _POLICY_STORE_MAX_BYTES)
    record.update({"ref": BC_PRIOR_REF, "wall_s": time.monotonic() - t0,
                   "nbytes": int(_nbytes(blob))})
    log.info("bc_prior: cloned %s from %d demonstrations (%d pairs, mse %.2e) in %.1f s",
             policy_id, record["demos"]["demos_used"], record["demos"]["pairs"],
             record["final_mse"], record["wall_s"])
    rundir = getattr(ctx, "rundir", None)
    if rundir is not None and getattr(rundir, "path", None) is not None:
        try:
            (Path(rundir.path) / "bc_prior.json").write_text(json.dumps(record, indent=2, default=str))
        except Exception as exc:  # noqa: BLE001 - the artifact is a courtesy, the blob is the mechanism
            log.warning("bc_prior: could not write bc_prior.json (%s: %s)", type(exc).__name__, exc)


def _sb3_learn_chunks(model: Any, plan: Sequence[int], total_steps: int,
                      callback: Any = None,
                      ) -> Iterator[Tuple[int, int]]:
    """Drive `model.learn` one chunk at a time; yield `(chunk_index, spent)`.

    `spent` is MEASURED off `model.num_timesteps`, never assumed from the
    request, and the loop stops the moment the measured spend reaches
    `total_steps`. Both halves exist because SB3's on-policy `learn()` completes
    whole `n_steps` rollouts (`while self.num_timesteps < total_timesteps:
    collect_rollouts(..., n_rollout_steps=self.n_steps)`), so a request of
    `chunk` steps executes `ceil(chunk / n_steps) * n_steps` of them -- 2048 per
    rollout by default, and no shipped config pins `n_steps`. A loop doing
    `learn(chunk); spent += chunk` over a fixed `n_chunks` would make every
    figure derived from `spent` under PPO -- the seed row's `train_steps`
    ("ACTUAL, not requested"), `env_steps`, `TrainResult.env_steps_used`,
    `budget.env_steps`, the curve's `step` and the pruner's `budget_frac` --
    the request, while the model trained ceil(chunk/2048)*2048 per chunk: the
    dev profile's 20,000 would run 36,864 steps and record 19,998 (1.84x), a
    600-step test 10,240 (17x), the full profile's 1M 1,024,000. SAC/TD3
    (`train_freq: 1`) are exact, so the error would be one-directional and
    PPO-specific: a PPO method and a SAC method "at the same budget" would
    train at different real budgets, `budget.json` would under-report the cost
    column, and successive-halving rungs would fire at the wrong real
    fraction.

    What is asked of the learner is `min(plan[ck], total_steps - spent)`, so
    the final request never exceeds what is left and the overshoot is bounded
    by one rollout (strictly less than `n_steps`). The published algorithm
    hyperparameters are left alone -- `n_steps` is PPO's batch geometry, not a
    scheduling knob, and clamping it to the chunk would change the algorithm on
    small budgets. The consequence is that an on-policy learner can write FEWER
    checkpoints than `len(plan)`: a budget that holds two rollouts holds two
    checkpoints, whatever `MIN_CHECKPOINTS` says, and the seed row's
    `n_checkpoints` records how many there were. An exact learner walks the
    plan to the end and lands on `total_steps` exactly.

    A generator rather than a loop body so the accounting is testable with a
    fake model on a machine with no SB3 installed; `run_seed` owns everything that
    happens between chunks (evaluation, snapshot, pruning, the deadline).
    """
    spent = 0
    for ck, requested in enumerate(plan):
        if spent >= total_steps:
            return
        before = int(model.num_timesteps)
        # `callback` is the training-epoch series' window closer
        # (`max_over_training_epochs`). Passed on EVERY chunk, and
        # it keeps its own cursor into the episode list, so the series is
        # continuous across the chunk boundary rather than restarting with it --
        # chunks are checkpoint boundaries and must not shape this series.
        model.learn(total_timesteps=max(1, min(int(requested), total_steps - spent)),
                    reset_num_timesteps=False,
                    **({"callback": callback} if callback is not None else {}))
        spent += int(model.num_timesteps) - before
        yield ck, spent


def _episode_native_scalar(env: Any, channel: str,
                           states: Sequence[Any],
                           actions: Sequence[Any]) -> Optional[float]:
    """One finished training episode -> its NATIVE-channel scalar, or None.

    For the training-epoch series (`max_over_training_epochs`). The channel is
    `bird.native_signal`'s, resolved once per run, and the
    rule it enforces is about PROVENANCE rather than about a code symbol:

    * `native_success` holds only on a task whose spec pins `discrete_success.
      kind: discrete` -- i.e. the spec pins a verified reimplementation of the
      VENDOR's own success check (Meta-World's geometric check, recomputed from
      the observation). There `env.success(traj)` -- equivalently `task_metric >=
      success_threshold` -- IS the benchmark's own success signal, so reading it
      is correct, not a leak. What the native-signal rule forbids is the CUSTOM
      metric: a BIRD-authored score on a `continuous_only` task.
    * `native_reward` is the task's shipped reference reward, summed over the
      episode -- the same `gt_return` `_rollout` records. It needs the
      ACTIONS as well as the states, which is why the wrapper buffers both.
    * `refuse` is a task with neither (`reward.human.kind: none`), and yields
      None so nothing is recorded; `_check_coherence` refuses the pin at load
      time, so this is belt-and-braces.

    Returns None rather than 0.0 on any failure: a missing episode is UNKNOWN,
    and a zero would be a data point the run never measured.
    """
    if channel == native_signal.NATIVE_SUCCESS:
        try:
            return 1.0 if env.success(list(states)) else 0.0
        except Exception:  # noqa: BLE001 - an adapter that cannot score is not a zero
            return None
    if channel == native_signal.NATIVE_REWARD:
        total = 0.0
        # states holds T+1 rows and actions T, `_rollout`'s convention: the
        # reference reward is credited for ARRIVING in states[i+1] via actions[i].
        for i, a in enumerate(actions):
            try:
                total += float(env.reference_reward(states[i + 1], a))
            except (NotImplementedError, IndexError):
                return None
            except Exception:  # noqa: BLE001
                return None
        return total if actions else None
    return None


def _sb3_run(ctx: Any, state: Any, candidate: Candidate, n_seeds: int,
             env_steps: Optional[int] = None,
             resume_ref: Optional[str] = None,
             seed_phase: str = "",
             handoff: Optional[SliceHandoff] = None) -> TrainResult:
    """Real SB3 training against the candidate reward, on a gym view of the adapter.

    Trained in chunks so the checkpoint curve is real rather than two endpoints
    -- the same requirement §4's `checkpoint_aggregation` places on every other
    backend. `train.algorithm` finally means what it says here: it selects the
    SB3 class. Steps are charged as MEASURED and the chunk loop stops at the
    budget (`_sb3_learn_chunks`): an on-policy learner rounds each request up to
    a whole rollout, so its curve can hold fewer rows than the chunk plan.

    `resume_ref` + `handoff` are one slice of `train.interaction:
    shared_population` (LaRes): the policy is restored by `_training_init`, the
    replay buffer is pre-filled from `_ROUND_REPLAY[handoff.prefill_ref]`
    relabelled under this candidate's reward (`_sb3_prefill_replay`), the seed
    is salted by the slice index, and the slice's NEW raw transitions are
    exported to `_ROUND_REPLAY[handoff.export_ref]` for the driver to pool.
    With only the first of those four, every slice would rebuild SAC on an
    empty buffer at the previous slice's seed.
    """
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback

    cfg, env = ctx.cfg, ctx.env
    t0 = time.monotonic()
    result = TrainResult(cand_id=candidate.cand_id, candidate=candidate)
    # The per-update native series is recorded ONLY when the selected reducer
    # reads it (`reads_training_epochs`, i.e. `max_over_training_epochs`). Under
    # every other `checkpoint_aggregation` none of it runs -- no per-step
    # episode buffering, no native scoring inside training, no extra seed-row
    # key -- so a config that does not select the value pays nothing and its
    # artifact does not change.
    _record_series = bool(getattr(
        _registry_get("checkpoint_aggregation", cfg["evaluate.fitness.checkpoint_aggregation"]),
        "reads_training_epochs", False))
    # Resolved ONCE per run, from the spec, so every episode of every seed
    # scores the same channel and the series cannot change meaning mid-run.
    # `refuse` (a task with neither a native success nor a reference reward)
    # yields no series at all, and `_per_seed_values` marks the report.
    _native_channel = (native_signal.channel_for_env(
        env_id=str(cfg.get("problem.env_id") or ""),
        task_id=str(cfg.get("problem.task_id") or "") or None)
        if _record_series else native_signal.REFUSE)
    try:
        reward = _scaled_reward(ctx, state, candidate,
                                _training_reward(ctx, state, candidate))
    except Exception as exc:  # noqa: BLE001
        ctx.budget.record_training(env_steps=0, gpu_seconds=time.monotonic() - t0)
        result.trained, result.error = False, f"{type(exc).__name__}: {exc}"
        return result

    # The co-designed observation, on the one backend that runs a real network.
    # Same refusal as `_run_backend`: no silent fall back to the raw state, or a
    # `limen` candidate quietly becomes a `limen_reward_only` candidate.
    #
    # The `_ObsView` is discarded here on purpose -- it exists to give
    # `_make_learner` a feature space, and this backend has no `_make_learner`.
    # `_Gym` below is SB3's feature boundary and applies phi itself.
    try:
        _view, phi = _install_observation(ctx, candidate, env)
    except Exception as exc:  # noqa: BLE001
        ctx.budget.record_training(env_steps=0, gpu_seconds=time.monotonic() - t0)
        result.trained, result.error = False, f"{type(exc).__name__}: {exc}"
        candidate.meta["observation_installed"] = False
        candidate.meta["observation_error"] = result.error
        return result

    def _features(s: Any) -> np.ndarray:
        """What SB3 sees. phi(s) under co-design, the raw state otherwise.

        `_Gym` already keeps its true state in `self._s` and returns an
        observation separately, which is exactly the split the co-designed
        observation needs: the reward is still called on `(self._s, a, s2)` --
        raw states -- and only what crosses into the model is featured.
        """
        return np.asarray(phi(s) if phi is not None else s, dtype=np.float32)

    algo, sb3_hyper, anchor_kind = _sb3_algo_and_hyper(cfg)
    continuous = getattr(env, "exact_states", None) is None
    norm_mode = cfg.get("train.reward_norm", "none")

    class _Gym(gym.Env):  # pragma: no cover - requires the optional dependency
        def __init__(self) -> None:
            obs_lo, obs_hi = ((phi.low, phi.high) if phi is not None
                              else (env.obs_low, env.obs_high))
            self.observation_space = spaces.Box(np.asarray(obs_lo, dtype=np.float32),
                                                np.asarray(obs_hi, dtype=np.float32))
            self.action_space = (spaces.Box(np.asarray(env.action_low, dtype=np.float32),
                                            np.asarray(env.action_high, dtype=np.float32))
                                 if continuous else spaces.Discrete(env.n_actions))
            self._s = None
            self._t = 0
            self._norm = _RewardNorm(norm_mode)
            # Training-epoch series. `_ep_states`/`_ep_actions` buffer the
            # CURRENT training episode; `_ep_scalars` collects one
            # native-channel scalar per FINISHED episode; `_epoch_metrics` is
            # the per-update series the
            # `max_over_training_epochs` reducer maxes -- closed by
            # `_EpochWindow._on_rollout_end`, which is the only thing that knows
            # where an update boundary falls. Buffers hold raw states, never
            # features, because the native scalar is a function of the env's own
            # state (`_states_of` takes a plain list).
            self._ep_states: List[Any] = []
            self._ep_actions: List[Any] = []
            self._ep_scalars: List[float] = []
            self._epoch_metrics: List[float] = []
            self._episode_seed = 0
            self._episode = 0
            #: `train.anchor.kind: kl_reward` -- an `anchored_ppo.AnchorPenalty` set by
            #: `run_seed` after the warm start; subtracted from the NORMALISED reward
            #: so `train.anchor.beta` means the same thing under every reward_norm.
            self.anchor: Any = None
            # `train.reference_policy`: a KL-style pull toward the task's
            # demonstration policy, applied to the TRAINING reward only. The
            # reference is the same solution policy the `demo_margin` screen
            # rolls out (bird/demos.py) -- a phase machine, so it is reset per
            # episode and stepped in lockstep with the learner on the learner's
            # own states. The judge and the checkpoint curve see the candidate's
            # raw reward (`_evaluate_policy` and `_rollout` never go through
            # this wrapper), so the penalty shapes learning and nothing else.
            self._ref = None
            self._ref_beta = 0.0
            self._ref_inv2s2 = 0.0
            if bool(cfg.get("train.reference_policy.enabled", False)):
                from bird import demos
                pols, why = demos.policy_set(env, str(cfg.get("problem.env_id") or ""),
                                             ["expert"])
                if pols:
                    self._ref = pols[0]
                    self._ref_beta = float(cfg.get("train.reference_policy.beta", 0.1) or 0.0)
                    sigma = float(cfg.get("train.reference_policy.sigma", 0.5) or 0.5)
                    self._ref_inv2s2 = 1.0 / (2.0 * sigma * sigma)
                else:
                    log.warning("train.reference_policy: %s -- training unregularised", why)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            # Resetting with `env.reset(np.random.default_rng(seed))` would make
            # `train.backend: sb3` IRREPRODUCIBLE FROM `seed`: SB3's
            # `DummyVecEnv.reset` passes `self._seeds[env_idx]` exactly once and
            # then calls `_reset_seeds()`, so every AUTO-reset after the first
            # arrives with `seed=None` -- and `default_rng(None)` draws OS
            # entropy. Measured with Eureka on pendulum, 8 candidates, one seed,
            # two runs at the same config seed: winner fitness 0.0746 vs 0.0945,
            # i.e. a different reward selected.
            #
            # A per-episode counter off the seed SB3 does hand over fixes it:
            # episode 0 draws the seed's own stream, and every later episode
            # draws a distinct, reproducible one. 7907 is prime
            # and matches the 7919 stride `_run_backend` uses between seeds, so
            # two seeds' episode streams cannot alias.
            if seed is not None:
                self._episode_seed = int(seed)
                self._episode = 0
            self._s = env.reset(np.random.default_rng(
                self._episode_seed + self._episode * 7907))
            if self._ref is not None:
                self._ref.reset(np.random.default_rng(
                    self._episode_seed + self._episode * 7907 + 1))
            self._episode += 1
            self._t = 0
            # Row 0 is the initial state: `task_metric`/`success` score
            # `states[1:]`, matching `_rollout`'s (T+1 states, T actions) shape.
            self._ep_states = [self._s]
            self._ep_actions = []
            return _features(self._s), {}

        def step(self, action):
            a = np.asarray(action, dtype=float).ravel() if continuous \
                else env.action_set[int(action)]
            # KL(N(a, s^2) || N(a_ref, s^2)) = ||a - a_ref||^2 / (2 s^2): the
            # Gaussian-policy KL to a deterministic reference, evaluated on the
            # learner's state BEFORE the step so the reference's phase machine
            # advances on the same trajectory the learner is producing.
            kl_pen = 0.0
            if self._ref is not None:
                a_ref = self._ref(self._s)
                kl_pen = self._ref_beta * float(np.sum((a - a_ref) ** 2)) * self._ref_inv2s2
            s2, done, info = env.step(self._s, a)
            r, _cv = reward(self._s, a, s2)
            # `train.reference_policy`'s pull lands on the RAW reward, as its author
            # wrote it (it is sized against the reward's own scale, beta / 2 sigma^2);
            # the anchor below lands on the NORMALISED one so `train.anchor.beta`
            # means the same thing under every reward_norm. Both zero when off.
            r = float(r) - kl_pen
            shaped = self._norm(float(r))
            if self.anchor is not None:
                # r'(s, a) = r - beta * D(s): the divergence at the state the action
                # was chosen in, which is what the loss-side form penalises too
                shaped -= self.anchor.penalty(_features(self._s))
            self._s = s2
            self._t += 1
            terminated = bool(done)
            truncated = self._t >= env.horizon
            if _record_series:
                # Buffer the raw transition, then close the episode at
                # the SAME boundary SB3 is told about, so the series can never
                # disagree with the learner about how many episodes ran.
                self._ep_states.append(s2)
                self._ep_actions.append(a)
                if terminated or truncated:
                    v = _episode_native_scalar(env, _native_channel,
                                               self._ep_states, self._ep_actions)
                    if v is not None:
                        self._ep_scalars.append(float(v))
                    self._ep_states = []
                    self._ep_actions = []
            return (_features(s2), shaped, terminated, truncated, info)

    env.set_dr(_dr_params(cfg, candidate))
    # Caller override beats config -- see the note in `_run_backend`. This is
    # where it matters most: sb3 runs a real learner, so an unhonoured short
    # budget here would cost real SAC wall-clock rather than surrogate steps.
    total_steps = int((env_steps if env_steps is not None
                       else cfg.get("train.env_steps", 20000)) or 20000)
    n_chunks = max(MIN_CHECKPOINTS, min(MAX_CHECKPOINTS, total_steps // 2048 or MIN_CHECKPOINTS))
    # The plan PARTITIONS the budget exactly, as fasttd3's `chunk_plan` does
    # (`total_steps // n_chunks` per chunk would truncate: 20,000 over 9 chunks
    # is 19,998). `_sb3_learn_chunks` walks it, charging what the model actually
    # executed and stopping at the budget.
    chunk_plan = [total_steps // n_chunks + (1 if ck < total_steps % n_chunks else 0)
                  for ck in range(n_chunks)]
    base_seed = _seed_base(ctx, state, candidate, seed_phase)
    # `train.pruning` and `train.timeout_s` are read HERE as well as in
    # `_run_backend`, which serves mock/tabular/none. Otherwise
    # `-s train.pruning=median_stop` on a Meta-World run would resolve, go into
    # `config.resolved.yaml`, contribute its bytes to the run id, and change
    # nothing: a fabricated pin (a declared key the algorithm does not honour).
    # Both are resolved below through the SAME `_pruner_for` and handed the same
    # `(curve, budget_frac)` argument the other backend passes, because a
    # pruning rule that means one thing on `mock` and another on `sb3` is worse
    # than no pruning at all -- every tester run would then be testing a
    # different method from the one a full-scale run executes.
    rule, prune_field = _pruner_for(cfg)
    prune_metric = str(cfg.get("train.pruning_metric", "own_reward") or "own_reward")
    select_rule, select_min_delta = _selection_for(cfg)
    # The demonstration ceiling, same call and same place as `_run_backend`:
    # under this candidate's DR, once per candidate, before any seed forks.
    ceiling = _demo_ceiling(ctx, state, candidate, reward) if _ceiling_wanted(cfg) else None
    timeout_s = float(cfg.get("train.timeout_s", 3600) or 3600)

    # -- train.init ---------------------------------------------------------
    # Read the SAME way `_run_backend` reads it, off the same `state` slots, so
    # a method point means one thing on both backends. Applied per SEED, since
    # `_run_backend` hands its init to every seed's learner too.
    init = _training_init(ctx, state, cfg, resume_ref, candidate)
    init_ref, secondary, sec_ratio = init.ref, init.secondary, init.ratio
    result.init_source, result.init_from_cand_id = init.source, init.from_cand_id
    result.init_similarity = init.similarity
    init_blob = _POLICY_STORE.get(init_ref or "") if init_ref else None

    off_policy = cfg.get("train.algorithm", "ppo") in _OFF_POLICY
    if cfg.get("train.init") == "secondary_replay_buffer" and not off_policy:
        # Keyed on the CONFIG, not on whether a buffer happened to arrive: with
        # an on-policy algorithm nothing ever writes one either, so a
        # data-driven condition would stay quiet in exactly the case that needs
        # saying. The same shape of statement `_make_learner` makes for the
        # linear-policy searcher, at warning level because here it is a
        # published method point (GT) losing its contribution rather than a
        # surrogate approximating one.
        log.warning("train.init=secondary_replay_buffer is a no-op with "
                    "train.algorithm=%s on sb3: an on-policy algorithm has no replay "
                    "buffer to mix into", cfg.get("train.algorithm", "ppo"))
    use_secondary = bool(sec_ratio > 0.0 and len(secondary or ()) and off_policy)
    # A CO-DESIGNED OBSERVATION AND A SHARED REPLAY BUFFER CANNOT BOTH BE
    # HONOURED, and the failure is silent in both directions, so this refuses
    # rather than picks. SB3's buffer holds whatever `_Gym` returned, which under
    # co-design is phi(s) and not s. So:
    #
    #   * relabelling it (`_sb3_secondary_buffer`) would call the candidate
    #     reward on FEATURE vectors, and the reward was written against the state
    #     -- a wrong number, computed without error, on every transition;
    #   * and the buffer is shared ACROSS candidates, each of which has its own
    #     phi, so the rows are not even in one another's feature space (the
    #     widths differ, and equal widths would be worse: it would run).
    #
    # Storing raw states beside the features would fix it, and is deliberately
    # not done here: no shipped config pairs the two -- LIMEN is
    # `train.algorithm: ppo`, i.e. on-policy, with no replay buffer at all -- so
    # this is a guard against a future config, and the guard a future config
    # wants is a refusal it can see, not a mechanism nobody measured. Exporting
    # is disabled too (`want_last_replay` below), for the same reason on the
    # read side: a slice of one candidate's features sitting in a store the next
    # round reads is a stale-value trap.
    if phi is not None and use_secondary:
        log.warning("train.init=secondary_replay_buffer and problem.search_space with "
                    "'observation' are mutually exclusive on sb3: the buffer holds the "
                    "co-designed features, not states, so relabelling it would call "
                    "the candidate reward on the wrong space. The shared buffer is "
                    "disabled for this candidate; the search continues.")
        use_secondary = False
    if init_ref and init_blob is None:
        # An evicted or never-written ref is exactly the silent degradation the
        # journal has to record: the run continues as `from_scratch` and says so.
        log.warning("train.init=%s: %s is not in _POLICY_STORE "
                    "(evicted, or written by another process) -- training from scratch",
                    cfg.get("train.init", "from_scratch"), init_ref)

    # -- the replay half of a shared_population slice (LaRes §4.3) ----------
    # Resolved ONCE here, against the round store the driver filled before
    # this wave, and handed to every seed's learner below -- the same shape as
    # `init_blob` above. Each refusal is a warning and a cold buffer, never a
    # crash, and each is recorded in the seed row (`replay_prefill_*` = 0) so
    # a slice that declared a shared buffer and got none stays visible.
    slice_index = int(handoff.index) if handoff is not None else 0
    prefill_rows: Any = None
    if handoff is not None and handoff.prefill_ref:
        prefill_rows = _ROUND_REPLAY.get(handoff.prefill_ref)
        if prefill_rows is None or len(prefill_rows) == 0:
            log.warning("shared_population: %s is not in the round replay store -- "
                        "this slice starts from an empty buffer", handoff.prefill_ref)
            prefill_rows = None
        elif not off_policy:
            # Refused by `_check_coherence` for `shared_buffer: true`; reachable
            # under `false` only if something exported rows on an on-policy
            # learner, which nothing does. Kept as a guard, not a path.
            log.warning("shared_population: train.algorithm=%s keeps no replay buffer, "
                        "so there is nothing to restore into", cfg.get("train.algorithm"))
            prefill_rows = None
        elif phi is not None:
            # Same argument as `use_secondary` above: the stored rows are RAW
            # states and the buffer under co-design holds phi(s).
            log.warning("shared_population: a co-designed observation and a restored "
                        "replay buffer cannot both be honoured on sb3 (the rows are "
                        "states, the buffer holds features); this slice starts from "
                        "an empty buffer")
            prefill_rows = None

    per_seed_curves: List[List[Dict[str, float]]] = []
    per_seed_components: List[List[Dict[str, float]]] = []
    want_components = "reward_component_values" in set(cfg.get("train.log", []) or [])
    policies: List[Callable[[np.ndarray], np.ndarray]] = []
    used = 0
    fatal_error = ""
    model = None

    def _predict_policy(m: Any) -> Callable[[np.ndarray], np.ndarray]:
        # Takes a RAW state and features it here, so `_evaluate_policy` and
        # `_rollout` go on driving the unwrapped adapter and the recorded
        # trajectory stays in state space.
        def policy(s: np.ndarray, _m=m) -> np.ndarray:
            act, _ = _m.predict(_features(s), deterministic=True)
            return np.asarray(act, dtype=float).ravel() if continuous \
                else env.action_set[int(act)]
        return policy

    # ROSKA's fusion search on this backend. `_Gym`, `_predict_policy`, `algo`
    # and `sb3_hyper` are at this scope, and the seed fold below builds a model
    # here the same way (the `m2` rebuild), so a parent-side probe follows the
    # existing pattern. Resolved once per candidate, for `_run_backend`'s reason: the
    # paper searches per reward function, not per seed.
    probe_train, probe_eval = 0, 0

    def _sb3_fusion_probe(alpha: float) -> float:
        """Train `probe_fraction` of a full training from theta_f(alpha), score it.

        Scored by `reward_return` -- the candidate's OWN reward, never the task
        metric, which would leak ground truth into the choice of warm start.
        Its own RNG and its own throwaway model, so the search cannot perturb
        the seed streams or leave state in the model the seeds will train.
        """
        nonlocal probe_train, probe_eval
        steps = max(1, int(round(float(cfg.get("train.fusion.sc_bo.probe_fraction", 0.0) or 0.0)
                                 * int(cfg.get("train.env_steps", 0) or 0))))
        # `_seed_base` + crc32, never `hash()`: Python's string hash is salted
        # per process, so the probe scores -- hence the GP's chosen alpha, hence
        # every sc_bo run -- would differ between two invocations of one seeded
        # config. `_seed_base`'s own docstring forbids it for exactly this, and
        # no determinism test here catches it: a fork INHERITS the salt, so
        # `parallel` and `sequential` agree while two separate runs do not.
        prng = np.random.default_rng(
            (_seed_base(ctx, state, candidate, "fusion_probe")
             + zlib.crc32(f"{float(alpha):.6f}".encode())) % (2 ** 32))
        m = algo("MlpPolicy", _Gym(), seed=int(cfg.get("seed", 0) or 0), **sb3_hyper)
        if init_blob is not None:
            _sb3_apply_policy(m, init_blob, init_ref or "", fusion_alpha=float(alpha))
        m.learn(total_timesteps=steps)
        metrics, _c, ev, _t = _evaluate_policy(
            env, _predict_policy(m), prng, reward, 1,
            _error_router(cfg.get("train.failure_detection", "exception"), io.StringIO()))
        probe_train += steps
        probe_eval += int(ev)
        ctx.budget.record_rollout_steps(steps + int(ev))  # see the surrogate probe
        return float(metrics.get("reward_return", 0.0))

    fusion_alpha, fusion_record = _resolve_fusion(ctx, cfg, init, _sb3_fusion_probe)
    if fusion_record is not None:
        fusion_record["probe_train_steps_nominal"] = int(round(
            int(fusion_record.get("evaluations", 0))
            * float(fusion_record.get("probe_fraction", 0.0))
            * int(cfg.get("train.env_steps", 0) or 0)))
        fusion_record["probe_train_steps"] = probe_train
        fusion_record["probe_eval_steps"] = probe_eval
        fusion_record["probe_env_steps"] = probe_train + probe_eval
    result.fusion = dict(fusion_record) if fusion_record else {}

    def run_seed(seed_i: int) -> Dict[str, Any]:
        """Train ONE seed. Same contract as `_run_backend.run_seed`: pure of
        `result`, the accumulators and `ctx.budget`, so the body runs inline
        (the sequential schedule) or inside a forked seed worker, with the fold
        below in the parent either way. A raise out of this body aborts the
        whole `_sb3_run` call."""
        seed = _seed_for(base_seed, seed_i, slice_index)
        seed_t0 = time.monotonic()
        seed_hyper = dict(sb3_hyper)
        if use_secondary:
            seed_hyper["replay_buffer_class"] = _mixed_buffer_class()
        # The training-epoch series' window closer. It exists because the
        # update boundary is not visible anywhere else: `chunk_plan` walks the
        # 5-20 CHECKPOINT boundaries, which is precisely the granularity this
        # feature exists to escape. `_on_rollout_end` fires once per PPO rollout
        # and once per off-policy `collect_rollouts` -- one entry per update
        # window on both learner families, which is Eureka's per-epoch unit.
        class _EpochWindow(BaseCallback):  # covered: tests/test_max_over_training_epochs.py (torch tier)
            def __init__(self, gym_ref: Any) -> None:
                super().__init__()
                self._g = gym_ref
                self._seen = 0

            def _on_step(self) -> bool:  # required by the ABC
                return True

            def _on_rollout_end(self) -> None:
                # The window's value is the MEAN over episodes that FINISHED in
                # it -- Eureka's per-epoch log is the same aggregate over its
                # parallel envs. A window in which no episode finished
                # contributes nothing rather than a zero: a long episode
                # spanning several updates has not scored badly, it has not
                # scored yet.
                done = self._g._ep_scalars[self._seen:]
                self._seen = len(self._g._ep_scalars)
                if done:
                    self._g._epoch_metrics.append(float(sum(done) / len(done)))

        gym_env = _Gym()
        model = algo("MlpPolicy", gym_env, seed=seed, **seed_hyper)
        # Per-seed truth, deliberately: a cumulative `|=` would let a later
        # seed's row report a ref only an earlier seed applied. Identical in
        # practice -- `_sb3_apply_policy` is deterministic per (blob,
        # architecture), so every seed agrees -- but the row records what THIS
        # seed did.
        warm_started = bool(init_blob is not None
                            and _sb3_apply_policy(model, init_blob, init_ref or "",
                                                  fusion_alpha=fusion_alpha))
        # LaRes Eq. 4 (`train.elite_constraint.kind: l2_params`). The reference
        # is the ELITE's stored parameters (`elite_desc["reference_policy_ref"]`
        # = `state.policy_ref`, fixed for the round), read off `_POLICY_STORE`
        # and held in the model while `attach_l2_constraint` snapshots
        # (`_sb3_attach_elite_constraint`) -- NOT a snapshot of what
        # `train.init` just loaded, which under `shared_population` is this
        # arm's own previous slice for every slice after its first (a proximal
        # term toward itself, with `elite_constraint_optimisers` looking
        # healthy throughout). Still gated on
        # `warm_started`, because LaRes constrains the agents it re-initialised
        # from the elite (Alg. 1 lines 10, 15) and the coherence rule pairs the
        # two keys. `n_constrained == 0` is recorded rather than swallowed: a
        # declared constraint that patched no optimiser is exactly the
        # "mechanism that does not execute" shape, and `constraint_ref` says
        # WHICH parameters the running term pulls toward.
        n_constrained = 0
        constraint_ref = ""
        elite_desc = None
        if cfg.get("train.elite_constraint.kind", "none") != "none":
            from .. import registry as _reg
            elite_desc = _reg.get("elite_constraint",
                                  cfg["train.elite_constraint.kind"])(ctx, state, candidate)
            if elite_desc is not None and warm_started:
                elite_ref = str(elite_desc.get("reference_policy_ref") or "")
                ref_blob = _POLICY_STORE.get(elite_ref) if elite_ref else None
                if ref_blob is None:
                    log.warning("train.elite_constraint.kind=l2_params: the elite's "
                                "parameters (%s) are not in _POLICY_STORE, so there is "
                                "nothing to constrain toward; training unconstrained",
                                elite_ref)
                else:
                    n_constrained = _sb3_attach_elite_constraint(
                        model, elite_desc["weight"], ref_blob, elite_ref)
                    if n_constrained == 0:
                        log.warning("train.elite_constraint.kind=l2_params found no "
                                    "optimiser to constrain on this policy; the term is "
                                    "NOT running")
                    else:
                        constraint_ref = elite_ref
            elif elite_desc is not None and not warm_started:
                log.warning("train.elite_constraint.kind=l2_params: no elite parameters "
                            "were loaded for this candidate, so there is nothing to "
                            "constrain toward; training unconstrained")
        anchored: Dict[str, Any] = {}
        actor_frozen = False
        # `train.anchor.until_iteration`: anchor only the iterations below it. Past it
        # the candidate still STARTS from the clone (`train.init` is untouched) but
        # trains free of the KL term -- hold in the first iteration, explore after.
        # Recorded in the seed row as `anchor_released` so the two regimes stay
        # distinguishable in the artifact.
        until = cfg.get("train.anchor.until_iteration")
        anchor_released = (anchor_kind in ("kl_clone", "kl_reward") and until is not None
                           and int(getattr(state, "iteration", 0) or 0) >= int(until))
        if anchor_released:
            log.info("train.anchor.until_iteration=%s: iteration %s trains unanchored",
                     until, getattr(state, "iteration", 0))
        if anchor_kind in ("kl_clone", "kl_reward") and warm_started and not anchor_released:
            from .anchored_ppo import attach_anchor, freeze_actor, reward_anchor
            kw = dict(beta=float(cfg.get("train.anchor.beta", 5.0)),
                      schedule=str(cfg.get("train.anchor.schedule", "adaptive")),
                      target=float(cfg.get("train.anchor.target", 0.05)),
                      log_std_init=cfg.get("train.anchor.log_std_init"),
                      beta_min=(1e-3 if cfg.get("train.anchor.beta_min") is None
                                else float(cfg.get("train.anchor.beta_min"))))
            if anchor_kind == "kl_clone":
                anchored = attach_anchor(model, **kw)
            else:
                gym_env.anchor = reward_anchor(model, **kw)
                anchored = {"kind": "kl_reward", "schedule": kw["schedule"], "beta": kw["beta"],
                            "target": kw["target"] if kw["schedule"] == "adaptive" else None,
                            "log_std_init": kw["log_std_init"], "beta_min": kw["beta_min"]}
            if int(cfg.get("train.anchor.warmup_steps", 0) or 0) > 0:
                freeze_actor(model.policy, True)
                actor_frozen = True
        elif anchor_kind in ("kl_clone", "kl_reward") and not anchor_released:
            log.warning("train.anchor.kind=%s: no initial policy was loaded for this "
                        "candidate, so there is nothing to anchor to; training unanchored",
                        anchor_kind)
        secondary_used = 0
        if use_secondary:
            mix = _sb3_secondary_buffer(secondary, sec_ratio, model, env, reward,
                                        norm_mode, continuous)
            if mix is not None:
                model.replay_buffer.secondary_buffer = mix
                model.replay_buffer.secondary_ratio = float(sec_ratio)
                secondary_used = int(mix.buffer_size)
        # LaRes §4.3, the replay half of the slice hand-off. AFTER the warm
        # start and the constraint (neither reads the buffer) and BEFORE the
        # first `learn`, so SB3's first batch already samples the restored
        # history. `learning_starts` is adjusted inside.
        prefilled = 0
        if prefill_rows is not None:
            prefilled = _sb3_prefill_replay(model, prefill_rows, env, reward,
                                            norm_mode, continuous)
        curve: List[Dict[str, float]] = []
        comp_curve: List[Dict[str, float]] = []
        spent = 0  # env steps spent *learning* -- what `train.env_steps` budgets
        spent_eval = 0  # checkpoint evaluation, counted in the cost report only
        seed_error = ""
        pruned_at = None
        timed_out = False
        policy = _predict_policy(model)
        best_snap: Optional[Dict[str, Any]] = None  # `train.checkpoint_selection`: the
        best_snap_idx = -1                          # running-best checkpoint's parameters
        ceiling_decisions: List[Dict[str, Any]] = []
        pruner = _ceiling_guard(cfg, rule, prune_metric, ceiling, ceiling_decisions)

        # `spent` is bound by the generator: MEASURED off `model.num_timesteps`,
        # never `+= chunk` (see `_sb3_learn_chunks` for the PPO under-count that
        # avoids). It keeps its last value after the loop, whichever exit is taken.
        for ck, spent in _sb3_learn_chunks(model, chunk_plan, total_steps,
                                           callback=(_EpochWindow(gym_env)
                                                     if _record_series else None)):
            if actor_frozen and model.num_timesteps >= int(cfg.get("train.anchor.warmup_steps", 0) or 0):
                from .anchored_ppo import freeze_actor
                freeze_actor(model.policy, False)
                actor_frozen = False
            if gym_env.anchor is not None:
                anchored.update(gym_env.anchor.adapt())   # per-chunk controller + the live beta
            elif anchored and getattr(model, "kl_last", None):
                anchored.update(model.kl_last)            # the loss-side form's last reading
            metrics, comps, ev_steps, _tr, ev_wall = _evaluate_policy_timed(
                env, policy, np.random.default_rng(seed + ck), reward,
                _checkpoint_episodes(cfg, sb3=True),
                _error_router("exception_soft", io.StringIO()))
            spent_eval += ev_steps
            # The per-component means `_evaluate_policy` computes are kept:
            # without them `result.component_traces` stays `{}` under
            # `train.backend: sb3` and §4's per-component reflection --
            # Eureka's whole feedback channel, RF-Agent's `l_feedback` --
            # renders only the task_score curve. Same plumbing as
            # `_run_backend`'s `comp_curve`.
            comp_curve.append(comps if want_components else {})
            curve.append({"step": float(spent), "round": float(ck + 1),
                          "seed": float(seed), "fitness": metrics["fitness"],
                          "score": metrics["fitness"], "task_success": metrics["fitness"],
                          "consecutive_successes": metrics["fitness"],
                          "success_rate": metrics["success_rate"],
                          "gt_return": metrics["gt_return"],
                          "return": metrics["reward_return"],
                          "reward_return": metrics["reward_return"],
                          # WALL-CLOCK: listed in `tests/test_parallelism._VOLATILE`. A timing
                          # field that is not listed there breaks bit-identity by construction --
                          # parallel candidates contend, so it differs from sequential on every
                          # run, and the failure surfaces as "parallel changed
                          # candidates/.../train_result.json" three files from the cause.
                          "eval_wall_s": ev_wall})
            if ceiling is not None:
                curve[-1]["demo_fraction"] = _demo_fraction_point(
                    metrics["reward_return"], ceiling, [p["reward_return"] for p in curve[:-1]])
            # `train.checkpoint_selection`: same snapshot rule as `_run_backend`
            # (running best by the candidate's OWN reward, under the rule's own
            # tie-break -- `_running_best`, which the post-loop rule agrees
            # with). One `get_parameters()` deep copy per improving checkpoint,
            # ~1 MB, against a chunk of training between them.
            if _running_best(curve, select_rule):
                best_snap = _snapshot_sb3(model)
                best_snap_idx = len(curve) - 1
            # Same order as `_run_backend`: prune on the curve first, then
            # the wall-clock guard. `budget_frac` is LEARNING steps over the
            # step budget -- evaluation steps are a cost, not progress, and
            # including them would move the successive-halving rungs.
            if pruner([p[prune_field] for p in curve], spent / max(1, total_steps)):
                pruned_at = ck + 1
                break
            # `spent < total_steps` IS LOAD-BEARING. Without it a seed that
            # spent its whole step budget and then crossed the deadline on
            # the very last chunk would be marked FAILED -- a complete 1M
            # training thrown away, and its candidate reported as broken,
            # for overrunning a wall clock it no longer needed. There is
            # nothing left to abandon at that point. `_run_backend` gets
            # this for free because its `spent >= steps_budget` break comes
            # first; here the generator's own stop is checked at the TOP of
            # the next chunk, so it has to be said. On the MEASURED spend,
            # not `ck + 1 < n_chunks`: an on-policy learner reaches the
            # budget before the plan runs out. NOTE the granularity: the
            # deadline is only visible between chunks, so a fire can
            # overshoot by one chunk (~1/20th of the budget).
            if spent < total_steps and time.monotonic() - seed_t0 > timeout_s:
                seed_error = f"timeout after {timeout_s:g}s"
                timed_out = True
                break
        # `train.checkpoint_selection`: restore the best checkpoint by the
        # designed reward into the LIVE model, here inside `run_seed`, so every
        # reader downstream gets it for free -- the sequential schedule rolls
        # this model out, `seed_child` serialises it, the parent rebuilds from
        # that blob, and `_POLICY_STORE` receives the same weights the judge saw.
        ship_idx, select_reason = _select_checkpoint(curve, select_rule, select_min_delta)
        if ship_idx is not None:
            if best_snap is None or best_snap_idx != ship_idx:
                ship_idx, select_reason = None, "no_snapshot"
            elif not _restore_sb3(model, best_snap):
                ship_idx, select_reason = None, "restore_failed"
        if select_rule != "final" and ship_idx is None:
            log.debug("  sb3 seed %d: train.checkpoint_selection=%s shipped the last "
                      "checkpoint (%s)", seed, select_rule, select_reason)

        wall = float(time.monotonic() - seed_t0)
        fits = [p["fitness"] for p in curve] or [0.0]
        # The SHIPPED checkpoint's row, as in `_run_backend`.
        ship_i = ship_idx if ship_idx is not None else len(fits) - 1
        seed_metric = {
            "seed": int(seed), "fitness": float(fits[ship_i]), "final": float(fits[ship_i]),
            "max": float(np.max(fits)), "auc": float(np.mean(fits)),
            # ACTUAL, not requested, and measured, not assumed: `spent` is
            # `model.num_timesteps` moved, per chunk. Reporting `total_steps`
            # would claim the full budget for a run a pruner stopped at 12% --
            # so the one number that proves pruning saved anything would show
            # no saving -- and `spent += chunk` would be the REQUEST under PPO,
            # which executes whole 2048-step rollouts (`_sb3_learn_chunks`):
            # 1.84x under-counted on the dev profile. `train_steps_requested`
            # keeps the ask visible beside it, so the two are comparable in the
            # artifact.
            "env_steps": int(spent + spent_eval), "train_steps": int(spent),
            "train_steps_requested": int(total_steps),
            "wallclock_s": wall,
            "n_checkpoints": len(curve), "pruned_at_round": pruned_at,
            "timed_out": bool(timed_out), "learner": "sb3",
            "algorithm_cited": cfg["train.algorithm"], "backend": "sb3",
            # The same two fields `_run_backend` records, so one reader can ask
            # both backends which MDP a seed trained on. `None` on the first =
            # the raw adapter. `policy_input_dim` comes off the MODEL's own
            # observation space -- the network's input layer -- and not off phi,
            # for the reason `_Learner.policy_input_dim` gives.
            "observation_dim": (int(phi.dim) if phi is not None else None),
            "policy_input_dim": _sb3_input_dim(model),
            # `problem.horizon` as executed, read off the adapter. The long
            # form of the argument is on the same field in
            # `bird/components/simba_v2.py`; the short one is that a horizon is
            # the one environment fact a method may pin, so two runs that
            # differ in it are two tasks and a table without this field cannot
            # tell.
            # The EPISODE length as run; the long note is on the same field
            # in `bird/components/simba_v2.py`.
            "horizon_effective": int(getattr(env, "horizon", 0) or 0),
            # What was ASKED for, beside what ran: the key clamps at
            # construction when the chosen env's episode is shorter, and the
            # long note is on the same pair in `bird/components/simba_v2.py`.
            "horizon_requested": (int(cfg.get("problem.horizon"))
                                  if cfg.get("problem.horizon") is not None else None),
            # `train.init` as EXECUTED, not as configured. A warm start that
            # did not take and an ablation are the same curve; only these two
            # numbers tell them apart after the fact.
            "init": cfg.get("train.init", "from_scratch"),
            "warm_started_from": (init_ref or "") if warm_started else "",
            "fusion_alpha": fusion_alpha,
            # `train.elite_constraint.kind: l2_params` -- how many optimisers the
            # term was actually installed on. Recorded here rather than
            # journalled because a seed body can run in a FORKED worker, where
            # `ctx.rundir` is None and `ctx.event` would be lost
            # (tests/test_parallelism.py::test_the_train_stage_touches_only_cfg_env_budget
            # is what keeps that honest). 0 beside a non-`none` key is the
            # declared-but-not-running case and must stay visible in the artifact.
            "elite_constraint_optimisers": n_constrained,
            # ...and WHICH parameters it pulls toward: the elite's stored policy
            # ref, or "" when the term is not running. Beside `warm_started_from`
            # on purpose -- on a resumed slice the two DIFFER (own previous
            # slice vs the elite), and that difference is the point. Only under
            # a non-`none` kind, so no other seed row carries it.
            **({"elite_constraint_reference": constraint_ref}
               if cfg.get("train.elite_constraint.kind", "none") != "none" else {}),
            "secondary_transitions": int(secondary_used),
            "secondary_ratio": float(sec_ratio if use_secondary else 0.0),
            # `train.anchor` as EXECUTED, for the reason `warm_started_from` is
            # here: an anchor with no incumbent to hang off trains unanchored
            # (the warning above), and that run and `kind: none` are the same
            # curve. `{}` says no anchor was installed; otherwise the kind, the
            # schedule and the controller's LAST reading (live beta, mean
            # divergence over the final chunk).
            "anchor": dict(anchored),
            "anchor_released": bool(anchor_released),
            # `train.checkpoint_selection` as EXECUTED -- the same five fields
            # `_run_backend` writes, documented there.
            "checkpoint_selection": select_rule,
            "checkpoint_selection_reason": _selection_reason(select_reason),
            "restored_checkpoint": (int(curve[ship_idx]["round"])
                                    if ship_idx is not None else None),
            "shipped_checkpoint": int(curve[ship_i]["round"]) if curve else None,
            "final_checkpoint_fitness": float(fits[-1]),
            "error": seed_error, "checkpoints": curve}
        if _record_series:
            # The per-update training series for THIS seed. Rides on the
            # seed row exactly as `checkpoints` does, so it pickles home over
            # the seed fork with no extra plumbing. Present ONLY when the
            # selected reducer reads it (`_record_series`); empty on a `refuse`
            # task -- `_per_seed_values` then falls back to the checkpoint curve
            # and sets `training_epoch_series_missing`, never silently.
            seed_metric["training_epoch_metrics"] = list(gym_env._epoch_metrics)
        if cfg.get("train.reward_scaling", "none") != "none":
            # LaRes Eq. 3 as executed (`_scaling_record`): the affine parameters
            # and the moments this seed's learner actually trained under, or
            # `applied: False`. Only under a non-`none` key.
            seed_metric["reward_scaling"] = _scaling_record(candidate, reward)
        # The slice hand-off as EXECUTED (`train.interaction:
        # shared_population`, LaRes §4.3). Only under a hand-off, so an
        # unsliced run's seed row carries none of these fields.
        # `replay_added` is what this slice put into the buffer -- one row per
        # env step at `n_envs=1`, bounded by the ring -- and is also the count
        # the export below takes, so the artifact and the pool agree.
        replay_added = 0
        if handoff is not None:
            rb = getattr(model, "replay_buffer", None) if off_policy else None
            if rb is not None and hasattr(rb, "buffer_size"):
                replay_added = min(int(model.num_timesteps), int(rb.buffer_size))
            own = min(int(handoff.prefill_own), int(prefilled))
            seed_metric.update({
                "slice_index": int(slice_index),
                "replay_prefill_supported": bool(off_policy),
                "replay_prefill_ref": (handoff.prefill_ref or "") if prefilled else "",
                "replay_prefill_own": int(own),
                "replay_prefill_pooled": int(prefilled - own),
                "replay_added": int(replay_added),
                "learning_starts_effective": int(getattr(model, "learning_starts", 0) or 0),
            })
        if ceiling is not None:
            seed_metric["ceiling_decisions"] = ceiling_decisions
        if hasattr(model, "detach_anchor"):
            # the clone and the schedule closures are not picklable; drop them
            # before `_sb3_policy_blob` serialises this model for the store
            model.detach_anchor()
        gym_env.anchor = None
        return {"curve": curve, "comp_curve": comp_curve, "spent": spent,
                "spent_eval": spent_eval, "pruned_at": pruned_at, "timed_out": timed_out,
                "seed_error": seed_error, "wallclock_s": wall,
                "model": model, "seed_metric": seed_metric, "replay_added": replay_added}

    def fold_seed(out: Dict[str, Any]) -> None:
        """Parent-side bookkeeping for one seed, strictly in seed order."""
        nonlocal used, fatal_error
        used += out["spent"] + out["spent_eval"]
        if out["pruned_at"] is not None:
            result.pruned = True
        if out["timed_out"]:
            result.timed_out = True
        if out["seed_error"] and not fatal_error:
            fatal_error = out["seed_error"]
        per_seed_curves.append(out["curve"])
        per_seed_components.append(out.get("comp_curve") or [])
        result.seed_metrics.append(out["seed_metric"])
        ctx.budget.record_training(env_steps=out["spent"] + out["spent_eval"],
                                   gpu_seconds=out["wallclock_s"])

    n_runs = max(1, int(n_seeds))
    seed_workers, seed_reason = _seed_fork_workers(cfg, n_runs)
    _note_seed_schedule(ctx, candidate, seed_workers, seed_reason)
    final_blob = None   # the LAST seed's serialised model (forked schedule)
    final_slice = None  # and its replay tail, exported in that child
    # `phi is None` for the reason argued at `use_secondary` above: what this
    # would export is one candidate's FEATURES, and the next round's candidates
    # have their own phi to relabel against. Nothing is written, so nothing
    # downstream has to guess whether a stored slice is in state space.
    want_last_replay = bool(off_policy and _wants_replay(cfg) and phi is None)
    replay_cap = int(cfg.get("train.secondary_buffer.size", 100000) or 100000)
    # The slice export (`SliceHandoff.export_ref`): the rows THIS call added,
    # raw, for the driver to pool. Same "last seed's learner" rule as the two
    # store writes below, and the same `phi is None` guard for the same reason.
    want_slice_export = bool(handoff is not None and handoff.export_ref
                             and off_policy and phi is None)
    slice_added = 0     # rows the last seed added (its `replay_added`)
    slice_export = None  # and the export of exactly those rows

    try:
        if seed_workers <= 1:
            for seed_i in range(n_runs):
                # PER SEED, not per candidate. One seed is one training, so a
                # candidate-level pre-flight would refuse a whole candidate where
                # the cap still has room for some of its seeds -- cap 4 with three
                # seeds must run four trainings, not three.
                # Trainings AND gpu hours, because one seed is a unit of
                # both. The gpu form asks "is the cap already spent" rather
                # than "would this cross it": a training's duration is not
                # knowable in advance, so the enforceable rule is not to
                # START a unit once the budget is gone.
                _why = (ctx.budget.would_exceed_training(1)
                        or ctx.budget.gpu_hours_exhausted())
                if _why:
                    raise BudgetExceeded(_why)
                out = run_seed(seed_i)
                model = out.pop("model")
                slice_added = int(out.get("replay_added", 0) or 0)
                policies.append(_predict_policy(model))
                fold_seed(out)
        else:
            def seed_child(i: int) -> Dict[str, Any]:
                out = run_seed(i)
                m = out.pop("model")
                out["blob"] = _sb3_policy_blob(m)
                if i == n_runs - 1 and want_last_replay:
                    out["replay"] = _sb3_export_replay(m, replay_cap)
                if i == n_runs - 1 and want_slice_export and out.get("replay_added"):
                    out["slice_export"] = _sb3_export_replay(m, int(out["replay_added"]))
                return out

            for seed_i, out in enumerate(
                    _fork_seeds(n_runs, seed_child, seed_workers, "sb3")):
                if "fatal" in out:
                    # `_sb3_run` lets a seed's exception escape to its caller;
                    # a fatal payload is that raise, relayed across the fork.
                    raise RuntimeError(f"sb3 seed worker {seed_i}/{n_runs}: "
                                       f"{out['fatal']}")
                blob = out.pop("blob", None)
                # Rebuild the trained policy from the exact parameters the
                # child saved: `predict(deterministic=True)` is a pure function
                # of (weights, obs, thread count), so the parent-side rollouts
                # below reproduce the sequential schedule's byte for byte.
                m2 = algo("MlpPolicy", _Gym(), seed=int(out["seed_metric"]["seed"]),
                          **sb3_hyper)
                if blob is None or not _sb3_apply_policy(
                        m2, blob, f"seed:{out['seed_metric']['seed']}"):
                    # Sequential has no analogue of this state -- it rolls the
                    # LIVE trained model and never serialises -- so a lost blob
                    # must not degrade into a fitness computed from freshly
                    # initialised weights and reported as real. Same idiom as
                    # every other broken measurement: keep the artifact, mark
                    # the number untrustworthy (`trained=False` + `error` at
                    # the tail). `_sb3_apply_policy`'s own leniency is for
                    # `train.init: warm_start_from_best`, where a cold start is
                    # correct behaviour; here it would be a corrupted
                    # measurement wearing a healthy one's clothes.
                    fatal_error = fatal_error or (
                        f"sb3 seed fork: seed {seed_i} sent no loadable policy "
                        "(model.save failed in the child, or the blob did not "
                        "load into this architecture); its rollouts would run "
                        "on untrained weights, so the result is marked failed "
                        "rather than mis-measured")
                    # And never PUBLISH it: a blob that exists but did not load
                    # written to `_POLICY_STORE` becomes a `state.policy_ref`
                    # a later iteration's `warm_start_from_best` points at --
                    # a known-bad snapshot wearing a valid ref.
                    blob = None
                policies.append(_predict_policy(m2))
                if seed_i == n_runs - 1:
                    final_blob = blob
                    final_slice = out.pop("replay", None) if blob is not None else None
                    slice_export = out.pop("slice_export", None) if blob is not None else None
                fold_seed(out)

        result.checkpoints = _mean_curve(per_seed_curves)
        _stamp_seed_rows(result.seed_metrics, seed_workers, seed_reason)
        _summarise_ceiling(candidate, result.seed_metrics)
        if want_components:
            result.component_traces = _mean_components(per_seed_components)
        result.gt_reward_curve = [p["gt_return"] for p in result.checkpoints]
        # Rollouts stay in the parent, on ONE shared roll_rng, in rollout order
        # -- that is what keeps a forked seed schedule's trajectories identical
        # to sequential's. The children trained at one torch thread, so the
        # parent's predicts run at one thread too (see `_single_thread_torch`).
        with (_single_thread_torch() if seed_workers > 1
              else contextlib.nullcontext()):
            roll_rng = np.random.default_rng(base_seed + 104729)
            for i in range(max(1, int(cfg.get("evaluate.rollouts_per_candidate", 3) or 1))):
                traj, st, _gt = _rollout(env, policies[i % len(policies)], roll_rng, reward,
                                         _error_router("exception_soft", io.StringIO()))
                result.trajectories.append(traj)
                if i < _n_replayed_rollouts(cfg):
                    _retain_replay_states(env, traj)
                used += st
                # RECORDED, not only accumulated into `env_steps_used`, like
                # the TPE-store loop below: otherwise `budget.json`
                # under-reports every run by `evaluate.rollouts_per_candidate`
                # x the horizon -- 3000 of 19920 env steps on a small run
                # where the TRAINING itself was 1920.
                #
                # Charged per rollout rather than summed after the loop, so a
                # rollout that raises mid-loop still charges what it spent.
                ctx.budget.record_rollout_steps(st)
            # Dedicated TPE-store pool -- same reasoning as `_run_backend`'s
            # copy: separate rng so the eval rollouts above stay byte-identical
            # when collection is toggled, no state retention (LRU pressure).
            n_store = _n_store_rollouts(cfg)
            if n_store > 0:
                store_rng = np.random.default_rng(base_seed + 224737)
                store_steps = 0
                for i in range(n_store):
                    traj, st, _gt = _rollout(env, policies[i % len(policies)], store_rng,
                                             reward,
                                             _error_router("exception_soft", io.StringIO()))
                    result.store_trajectories.append(traj)
                    used += st
                    store_steps += st
                # Same budget charge as `_run_backend`'s copy: collection
                # steps must reach `budget.json`, not only `env_steps_used`.
                ctx.budget.record_rollout_steps(store_steps)

        # Same two writes `_run_backend` makes, same keys, same eviction, same
        # "last seed's learner is the one that is kept" rule. Without them
        # `state.policy_ref` / `state.replay_ref` stay null forever and
        # `train.init` is a fabricated pin on the only backend that runs a real
        # learner. The SECOND write is skipped for on-policy algorithms, which
        # have no buffer: a `replay_ref` naming nothing would degrade at read
        # time instead of being absent at write time, and `checkpoint.py`
        # reports the first as corruption and the second as normal.
        if seed_workers <= 1:
            final_blob = _sb3_policy_blob(model) if model is not None else None
            if want_last_replay and model is not None:
                final_slice = _sb3_export_replay(model, replay_cap)
            if want_slice_export and model is not None and slice_added > 0:
                slice_export = _sb3_export_replay(model, slice_added)
        if final_blob is not None:
            result.policy_ref = _store(_POLICY_STORE, f"policy:{candidate.cand_id}",
                                       final_blob, _POLICY_STORE_MAX_BYTES)
            _remember_code(candidate)
        if final_slice is not None:
            result.replay_ref = _store(_REPLAY_STORE, f"replay:{candidate.cand_id}",
                                       final_slice, _REPLAY_STORE_MAX_BYTES)
        if slice_export is not None and want_slice_export:
            # Plain assignment, no cap: `_ROUND_REPLAY` is bounded by the
            # driver, which pops this key right after the wave.
            _ROUND_REPLAY[str(handoff.export_ref)] = slice_export
    finally:
        env.set_dr(None)

    result.env_steps_used = int(used)
    result.wallclock_s = float(time.monotonic() - t0)
    # A timeout is a FAILURE (`trained=False`, the reason in `error`); a prune is
    # not (`trained=True`, `pruned=True`). They are different populations for the
    # same reason `failure_kind` splits "never compiled" from "a screen rejected
    # it": one says the reward could not be measured, the other says it was
    # measured and found not worth finishing.
    if fatal_error:
        result.trained = False
        result.error = fatal_error
    return result


# ==========================================================================
# candidate_parallelism  --  train.candidate_parallelism
# ==========================================================================
#
# Iterations MUST stay sequential: reflect-and-mutate is the method, not an
# implementation detail. Candidates *within* one iteration are independent --
# stage 3 hands each of them the same `state` and reads nothing back until
# stage 4 -- so they are the only axis that can be widened without changing
# what is being measured.
#
# Why this is a registry family and not an `if mode == "parallel"` in
# `bird.py`: the schema binds `train.candidate_parallelism` to this family
# (`kind="candidate_parallelism"`), so a value with no implementation cannot be
# offered and an implementation the schema rejects cannot be registered. It
# also keeps `train()` free of a branch, which `tests/test_no_method_branching`
# is there to protect.
#
# Both runners take the same five arguments:
#
#     runner(ctx, state, candidates, run_one, on_result) -> list[TrainResult]
#
# `run_one(candidate) -> TrainResult` is the whole per-candidate call
# (`hpsearch` wrapping the backend, at this candidate's seed allocation) and
# `on_result(candidate, result) -> None` is the parent-side bookkeeping the
# journal needs. Splitting it that way keeps `sequential` a plain loop over
# `run_one`, so what the default schedule does is a reading of the code and
# not a claim about it.
#
# ---------------------------------------------------------------------------
# Why processes, forked, one per candidate, and no pool
# ---------------------------------------------------------------------------
#
# THREADS ARE NOT AVAILABLE, and the reason is not the GIL (though it does not
# help -- 8 candidates on the `mock` backend measured 1.13 s serial -> 1.33 s
# on 8 threads, a 0.87x *slowdown*, since the work is numpy). The reason is
# that six separate pieces of this stage are process-global mutable state, and
# every one of them silently changes results rather than raising:
#
#   * `ctx.env` is ONE shared adapter for the whole run (`bird.py` builds
#     exactly one; `bird/envs/toy.py`'s docstring already says it is not
#     reentrant). Measured on `toy_gridworld` with 6 threads: 2 of 6 candidates
#     changed fitness (0.250 -> 0.125 and 0.250 -> 0.000), because `reset()`
#     reassigns `self._rng` mid-episode under the other candidate.
#   * `env.set_dr(None)` in `_run_backend`'s `finally` disarms domain
#     randomisation for every candidate still training -- DrEureka's entire
#     contribution, partially applied, with nothing logged.
#   * `contextlib.redirect_stdout` (above, in the seed loop) assigns
#     process-global `sys.stdout`. Measured with 3 interleaved threads: sink A
#     received C's traceback and sink B received nothing, so under
#     `train.failure_detection: stdout_grep` the healthy candidate is recorded
#     as failed. One trial also left `sys.stdout` pointing at a dead StringIO
#     for the rest of the process.
#   * `search.py::_overlaid` mutates the LIVE resolved config, so under
#     `train.hyperparameter_search: per_candidate_best` candidate B was
#     measured observing candidate A's grid point mid-training.
#   * `Budget.record_*` is an unsynchronised `+=`. 16 threads x 20000 calls
#     lost 72-78% of `env_steps` increments (89623 of an expected 320000).
#   * `_POLICY_STORE` / `_REPLAY_STORE` are module-level dicts whose eviction
#     loop is a read-modify-write.
#
# All six are one bug -- shared mutable state in one address space -- and a
# separate address space per candidate fixes all six with no code. What is left
# is only the inverse problem, "what the worker computed must come back", which
# is the four explicit return values in `_CHILD_PAYLOAD` below.
#
# And on the metaworld tier threads are not merely wrong, they are fatal:
# two threads stepping one MetaWorld adapter killed the interpreter with
# SIGSEGV 3 times out of 3. A SIGSEGV bypasses `bird.py::run`'s
# `except BaseException -> rundir.finish("failed")`, so `status.json` is left
# reading `running` forever -- per `artifacts.py`'s own convention, that is
# indistinguishable from a job still in the queue.
#
# `spawn` IS NOT AVAILABLE either: the MetaWorld adapter cannot be pickled
# (`self._mj = mujoco` -> "cannot pickle 'module' object"), and hand-removing
# that attribute yields a 42.4 MB blob whose first cache-restoring step raises
# `AttributeError: no attribute '_mj'` -- intermittently, since a step that
# hits `key == self._cur` short-circuits the restore. `fork` never unpickles an
# adapter: the child *inherits* a fully-constructed one by copy-on-write.
# Measured: 4 children x 120 MuJoCo steps = 0.09 s wall, parent RSS unchanged
# at 320 MB. Under `spawn` each worker would pay the full ~786 MB (torch + sb3)
# independently.
#
# A POOL IS NOT AVAILABLE: `ProcessPoolExecutor` answers a worker SIGSEGV with
# `BrokenProcessPool` on *every* pending future, which loses the other
# candidates' results -- the exact outcome this stage's "failure is data, not
# an exception" contract exists to prevent. Fork-per-candidate turns the same
# segfault into one `TrainResult.error` and leaves its neighbours untouched.
#
# ---------------------------------------------------------------------------
# Measured speedup
# ---------------------------------------------------------------------------
# See MEASURED_SPEEDUP below.

#: What a worker sends home. A file rather than a pipe on purpose: a pipe's
#: 64 KiB buffer deadlocks a child writing a multi-MB payload while the parent
#: blocks in `waitpid`, and avoiding that needs a `selectors` loop over every
#: child fd. A file has no capacity limit and no deadlock class. Measured on a
#: real MetaWorld payload: 1.6 MB, 16 ms to pickle, 6 ms to write, 10 ms to
#: read -- against candidate trainings measured in minutes.
_CHILD_PAYLOAD = ("result", "candidate", "budget", "policy", "replay",
                  "round_replay", "env_states", "exceeded")

#: `PR_SET_PDEATHSIG`, from <linux/prctl.h>. A worker whose parent is gone is a
#: MuJoCo process burning shared cores with nobody to reap it; under a batch
#: scheduler that kills the job step and not necessarily its grandchildren,
#: that is somebody else's job that does not start.
_PR_SET_PDEATHSIG = 1

#: MEASURED on a 32-core machine. Eureka on pendulum -- the REAL SB3 (SAC)
#: backend, mock LLM, `generate.n_candidates=8`
#: (7 trainable, 1 rejected by parsing), `train.env_steps=4000`, 1 iteration,
#: `loop.max_parallel_trainings: auto` -> 7 workers:
#:
#:     sequential   259.34 s   (run A)      stage 3 alone: 252 s
#:     sequential   257.96 s   (run B)
#:     parallel      66.65 s                stage 3 alone:  59 s
#:                             --------------------------------------
#:                             3.89x end to end, 4.27x on stage 3
#:
#: Amdahl and not the mechanism is the limit: generate, verify, evaluate,
#: select and the video encode all still run serially in the parent, and they
#: are ~8 s of the 66.65 s. On a Meta-World search, where training is
#: >95% of wall-clock, the collapse approaches the worker count.
#:
#: All three run dirs were compared file by file (53 files each, mp4s
#: included): seqA == seqB == parallel, byte for byte apart from the run-dir
#: name embedded in the journal's absolute video paths.
MEASURED_SPEEDUP = 3.89


def resolve_workers(cfg: Any, n_jobs: int) -> int:
    """`loop.max_parallel_trainings` -> a worker count.

        auto -> min(SLURM_CPUS_PER_TASK or len(sched_getaffinity(0)), n_jobs)
        int  -> min(max(1, int(v)), n_jobs)

    `os.sched_getaffinity`, not `os.cpu_count()`, so a cgroup or a `taskset`
    is respected -- on a batch-scheduled node `cpu_count()` can report all 224
    cores of the machine and none of the 2 the job was given. `SLURM_CPUS_PER_TASK` is
    preferred over the affinity mask because the scheduler sets it exactly and the mask
    is only as narrow as the site's cgroup configuration makes it.

    Capped by `n_jobs` so `parallel` at `generate.n_candidates: 1` resolves to
    one worker and is a no-op -- CARD is sequential by construction (K=1), so
    the value has to be harmless there rather than an error. The single fork
    still happens, which buys crash isolation free.

    Measured on a 32-core machine:

        SLURM_CPUS_PER_TASK  key    n_jobs  workers
        unset                auto   8       8
        unset                auto   1       1
        unset                4      8 / 2   4 / 2
        2                    auto   8       2
        16                   auto   8       8

    At `SLURM_CPUS_PER_TASK=2`, `auto` gives 2 workers x 1 thread rather than
    1 worker x 2 torch threads, which on a two-layer MLP is the better use of
    the allocation.
    """
    if n_jobs <= 0:
        return 1
    raw = cfg.get("loop.max_parallel_trainings", "auto")
    if isinstance(raw, str) and raw.strip().lower() == "auto":
        slurm = os.environ.get("SLURM_CPUS_PER_TASK", "").strip()
        if slurm.isdigit() and int(slurm) > 0:
            avail = int(slurm)
        else:
            try:
                avail = len(os.sched_getaffinity(0))
            except AttributeError:  # pragma: no cover - not Linux
                avail = os.cpu_count() or 1
    else:
        avail = int(raw)
    return max(1, min(avail, n_jobs))


#: `TrainResult.skip_reason` for a candidate the generate-only last iteration
#: of a `loop.termination: fixed_generations` run never launched. A constant,
#: not a format string, so a reader can grep the artifact for one string and a
#: TPE rejection under `verify.tpe.on_failure: skip_training` -- the other route
#: to an untrained chain end -- can never be mistaken for it.
GENERATION_ONLY_SKIP = ("never trained: the final iteration is generation only "
                        "(loop.termination: fixed_generations)")


def generation_only_result(ctx: Any, candidate: Candidate) -> TrainResult:
    """Stage 3 for the generate-only last iteration (`RunState.generation_only`).

    A TRAINABLE candidate here is a training the method DECIDED not to launch
    -- CARD's Alg. 1 returns R_N straight from the Coder (l.17, `Ensure R`) --
    so it is booked in `policy_trainings_skipped`, the same column a screen's
    saving goes to, with `skip_reason` naming this route. An untrainable one
    keeps its own failure and its own counter through `_skip_result`: the
    validity half of §2 still runs in this iteration, and a program that never
    compiled is a defect whichever iteration it fell in.

    Parent-side only and never forked, for `_skip_result`'s reason: the budget
    call must land in the process that writes `budget.json`.
    """
    if not candidate.trainable:
        return _skip_result(ctx, candidate)
    ctx.budget.record_skip()
    return TrainResult(cand_id=candidate.cand_id, candidate=candidate, trained=False,
                       skip_reason=GENERATION_ONLY_SKIP)


def _skip_result(ctx: Any, candidate: Candidate) -> TrainResult:
    """The `not c.trainable` branch, shared by both runners and NEVER forked.

    `record_skip` is the only budget call that must stay in the parent, and
    keeping the branch here rather than inside a worker is what preserves
    `policy_trainings_skipped` -- CARD's entire contribution is RL runs it did
    not launch, and if that counter stays at zero the cost claim is invisible.

    TWO POPULATIONS, TWO COUNTERS. `trainable = valid and not screened_out`, so
    a candidate lands here either because a screen decided against it
    (`failure_kind: "screened"` -- the saving) or because it never compiled or
    forfeited its slot (`failure_kind: "invalid"` -- a defect). Counting both
    in `policy_trainings_skipped` would make an Eureka run with three
    forfeited slots read as three saved trainings in the column CARD is
    compared on; the artifact keeps them apart (`failure_kind`), and so does
    the summary.
    """
    if candidate.valid:
        ctx.budget.record_skip()
    else:
        ctx.budget.record_invalid()
    return TrainResult(cand_id=candidate.cand_id, candidate=candidate, trained=False,
                       skip_reason=candidate.failure or "not trainable")


@register("candidate_parallelism", "sequential")
def parallelism_sequential(ctx: Any, state: Any, candidates: Sequence[Candidate],
                           run_one: Callable[[Candidate], TrainResult],
                           on_result: Callable[[Candidate, TrainResult], None],
                           ) -> List[TrainResult]:
    """One candidate at a time, in list order. The default, and every paper.

    A plain loop: list order, one budget call and one journal event per
    candidate, exceptions propagated (a `BudgetExceeded` from inside a
    candidate aborts stage 3 with the earlier candidates' costs already
    recorded).
    """
    ensure_bc_prior(ctx, state, ctx.cfg)
    results: List[TrainResult] = []
    for c in candidates:
        if not c.trainable:
            results.append(_skip_result(ctx, c))
            continue
        # ONE torch thread around the backend call, the pin every forked worker
        # applies (`_pin_one_thread`, `_seed_child_main`). Otherwise the
        # sequential parent trains at whatever thread count the process
        # inherited (a batch job commonly exports OMP/MKL/TORCH_NUM_THREADS),
        # so on sb3 `parallel` (1 thread) and `sequential` (8 or 16) would run
        # different reduction orders and not be bit-identical, and the same
        # config's numbers would move with the allocation ((d) below measures
        # it). fasttd3 also pins inside its own seed loop; this pins the
        # schedule, so every backend gets the same arithmetic on both.
        with _single_thread_torch():
            res = run_one(c)
        # The SAME device check the parallel fold makes. Both schedules or
        # neither: `parallel` must be bit-identical to `sequential`, and a
        # refusal that fires on one of them turns a scheduling key into a
        # science key -- the exact thing that invariant exists to forbid.
        wrong_device = learner_device_mismatch(ctx.cfg, res)
        if wrong_device:
            log.error("candidate %s: %s", getattr(c, "cand_id", "?"), wrong_device)
            res = _worker_failure(c, 0, {"fatal": wrong_device})
        results.append(res)
        on_result(c, res)
    return results

def _charge_discarded_env_steps(ctx: Any, payload: Any, cand_id: str) -> None:
    """Account a discarded child's env interaction. It happened; the machine paid.

    A worker whose payload is fatal or malformed still stepped the env before
    it died. The result is thrown away, and without this the budget would never
    hear about the work: real interaction, invisible in `budget.json`, so a
    cost column under-reports and nothing says so.

    STEPS ONLY, via `record_rollout_steps`. `policy_trainings` is deliberately
    NOT incremented: the cap counts trainings THIS SEARCH'S RESULT DEPENDS ON
    (`budget.py`), and a discarded candidate's does not -- charging it would
    let a crashing worker consume the cap and change which candidates run,
    which is a scheduling knob becoming a science knob. `record_rollout_steps`
    moves no cap check, which is exactly the behaviour wanted here.
    """
    steps = 0
    try:
        steps = int(((payload or {}).get("budget") or {}).get("env_steps") or 0)
    except Exception:  # noqa: BLE001 - a malformed payload is the normal case here
        steps = 0
    if steps > 0:
        ctx.budget.record_rollout_steps(steps)
        log.info("worker %s: discarded, but charging %d env steps it really spent",
                 cand_id, steps)


def wave_under_cap(jobs: List[int], cap: Optional[int], done: int,
                   seeds_per_candidate: Any) -> Tuple[List[int], List[int]]:
    """Split a wave into the candidates the cap admits and the rest.

    PURE, and extracted from `parallelism_parallel` so it can be tested
    without running a search: the two arithmetic traps below are invisible
    on the default config and cost minutes to exercise end to end.

    IN TRAININGS, NOT CANDIDATES. `policy_trainings` increments per SEED, so
    one candidate is `train.seeds_per_candidate` trainings. With the default
    1 the two coincide, which is why an off-by-seeds error here would show on
    no published config except the GT points.

    THE CROSSING CANDIDATE DOES NOT RUN, on either path. The key caps the
    TOTAL number of trainings (and `budget.merge_delta`'s own docstring
    presumes sequential is exact), so checking the cap AFTER recording --
    which would execute `cap + 1` trainings -- is an off-by-one, not a
    semantics. Both seed loops ask `budget.would_exceed_training(1)` before
    each seed -- PER SEED, since one seed is one training, so the check
    belongs inside the seed loops rather than at the candidate boundary --
    and this sizes to `remaining` with no `+1`. Ceiling division, so a
    candidate whose seeds do not all fit does not run at all.
    """
    if cap is None:
        return list(jobs), []
    per = max(1, int(seeds_per_candidate or 1))
    remaining = max(0, int(cap) - int(done))
    keep = -(-remaining // per)
    return list(jobs[:keep]), list(jobs[keep:])


class WaveFailed(RuntimeError):
    """Every worker of one candidate-parallelism wave failed. The run stops.

    A single dead worker is ONE failed candidate (`_worker_failure`) and that
    is the right answer: the neighbours' results are real, stage 4 scores the
    corpse at `select.failure_value`, and the search goes on. A wave in which
    EVERY worker died is a different event wearing the same clothes -- the
    iteration trained nothing at all, and the cause is never the reward code
    (a reward that raises during training comes back as a folded payload with
    `TrainResult.error` set, from a worker that lived).

    Treated as one case, a wave in which every worker died exits **0** with
    `[3] train -> 0 trained, 3 failed` as the only trace -- a banner in the
    middle of a log nobody reads when the exit code says the run was fine.

    THE ASYMMETRY IS THE ARGUMENT. `parallelism_sequential` has no such class:
    an exception out of `run_one` propagates through the stage, through
    `bird.py::run`'s `except BaseException -> rundir.finish("failed")`, and
    out of `main` -- non-zero exit, `status.json: failed`. Only the fork/spawn
    path turns a dead process into a value, because `_child_main` catches
    `BaseException` and writes it into a payload. So this does not make
    `parallel` stricter than `sequential`; it stops it being quieter.
    """


def _report_wave_failures(n_jobs: int, failures: Sequence[Tuple[str, str, str]]) -> None:
    """Log what the wave lost, and raise `WaveFailed` if it lost everything.

    Separate from the fold so the message exists once and a test can reach it
    without a fork. `failures` is (cand_id, `TrainResult.error` verbatim,
    child traceback or "") in candidate-index order, so "the first failure" is
    the first candidate that failed and not the first process that happened to
    die.

    The line names the STAGE, the COUNT against the wave's own size and the
    first error VERBATIM -- the three things a human reads a log for and none
    of which "0 trained, 3 failed" supplies: it says nothing about which
    stage's schedule broke, nothing about whether one worker or all of them
    died, and nothing whatever about why.

    THE PREDICATE IS "EVERY WORKER DIED", NEVER "NOTHING TRAINED", AND THAT IS
    NOT A NICETY -- a wave that trains nothing is a RESULT in this repo.
    `bird/budget.py` counts policy trainings skipped as well as run because
    CARD's entire contribution is RL runs it did not launch, so a guard keyed
    on an empty wave would fire on a correct run of a published method. That
    is a worse failure than the one this exists to fix: a silent all-dead wave
    lets a broken run look fine, and the inverse would make a fine run look
    broken. A screened candidate never enters `jobs` and so cannot vote at
    all; a LIVING worker that returned a well-formed payload deciding to
    train nothing (`train.interaction: shared_population`, an arm not
    selected this wave) votes not-a-failure, because a payload that folded is
    evidence the infrastructure worked, whatever the worker did with its turn.
    """
    if not failures:
        return
    first_id, first_error, first_tb = failures[0]
    summary = ("[3] train: %d of %d candidate-parallelism worker(s) failed; "
               "first failure (candidate %s): %s"
               % (len(failures), n_jobs, first_id, first_error))
    log.error("%s", summary)
    if len(failures) < n_jobs:
        # A partial failure is the published behaviour and stays it: the
        # survivors are real results and the wave is not a lost cause.
        return
    if first_tb:
        # The child's own traceback, which the parent has always discarded --
        # `_worker_failure` reads `fatal` and nothing reads `traceback`. On a
        # wave that killed the run it is the only stack anybody gets.
        log.error("first worker's traceback:\n%s", first_tb)
    raise WaveFailed(
        summary + " -- EVERY worker in this wave failed, so the iteration "
        "trained nothing. The run stops here rather than scoring an empty "
        "wave and exiting 0.")


@register("candidate_parallelism", "parallel")
def parallelism_parallel(ctx: Any, state: Any, candidates: Sequence[Candidate],
                         run_one: Callable[[Candidate], TrainResult],
                         on_result: Callable[[Candidate, TrainResult], None],
                         ) -> List[TrainResult]:
    """One forked process per trainable candidate, `loop.max_parallel_trainings` at a time.

    Determinism -- the reason this is allowed to exist at all -- rests on three
    facts, each checked rather than argued:

    (a) *A worker cannot observe anything order-dependent.* `training.py` and
        `search.py` between them reference none of `ctx.rng`, `ctx.next_id`,
        `ctx.counters`, `ctx.generator`, `ctx.evaluator`, `ctx.human`,
        `ctx.rundir` or `ctx.tracker` -- only `ctx.cfg`, `ctx.env` and
        `ctx.budget`, all three of which the child holds privately after the
        fork. `tests/test_parallelism.py::test_the_train_stage_touches_only_cfg_env_budget`
        AST-walks both modules to keep it that way. Seeds come from
        `_seed_base`, which reads no index and consumes no shared stream.

    (b) *Adapter residue does not matter.* `EnvAdapter.reset(rng)` assigns
        `self._rng` and re-draws `self._dr_now`, so a forked-fresh adapter and
        a used one produce the same episode from the same `rng`.

    (c) *Reassembly is by candidate index*, never by completion. `select.tie_break:
        first` is the published behaviour of every method here, and tie-breaks
        decide real rounds, so an `as_completed` implementation would change
        iteration winners while every fitness number stayed bit-identical -- a
        diff no numeric comparison catches.

    (d) *Worker count is not a numerics knob.* Each child pins exactly ONE
        thread -- not `cpus // workers`. If per-worker threads varied with the
        worker count, changing `loop.max_parallel_trainings` would change
        torch's CPU reduction order and therefore the fitness values, turning a
        scheduling knob into a science knob. That this is a live risk was
        measured: sequential SB3 at 16 threads and at 8 threads produce
        different `gt_return` -- which is why `parallelism_sequential` pins the
        same one thread around each backend call: the two
        schedules execute the same arithmetic at any allocation.
    """
    ensure_bc_prior(ctx, state, ctx.cfg)   # in the parent: the fork copies the store
    results: List[Optional[TrainResult]] = [None] * len(candidates)
    jobs: List[int] = []
    for i, c in enumerate(candidates):
        if not c.trainable:
            results[i] = _skip_result(ctx, c)
        else:
            jobs.append(i)

    # SIZE THE WAVE TO WHAT THE CAP ALLOWS, before forking anything.
    #
    # Without this the parallel path forks every trainable candidate, and a
    # `budget.max_policy_trainings` that trips mid-iteration is observed only
    # when the results are folded -- so the cores are spent on candidates the
    # sequential path never reaches, and their evaluation rollouts are real
    # env interaction that no artifact counts. Measured at cap=1: sequential
    # evaluated one candidate (75 env steps), parallel evaluated sixteen
    # (1200), and the fold's discard hid the difference.
    #
    # IN TRAININGS, NOT CANDIDATES. `policy_trainings` increments per SEED, so
    # one candidate is `train.seeds_per_candidate` trainings; with the default
    # 1 the two coincide and with the GT points they do not.
    # `tests/test_parallelism.py` names that case.
    jobs, _dropped = wave_under_cap(
        jobs, ctx.budget.max_policy_trainings, ctx.budget.policy_trainings,
        ctx.cfg.get("train.seeds_per_candidate"))

    workers = resolve_workers(ctx.cfg, len(jobs))
    # `tempfile.mkdtemp` honours TMPDIR, i.e. node-local disk on a cluster, and
    # deliberately NOT the run dir: on a network filesystem (NFS, for example) an
    # O_APPEND from a dozen processes is client-emulated and not guaranteed
    # atomic, and none of this belongs in the artifact anyway.
    spool = tempfile.mkdtemp(prefix="bird-train-")
    log.debug("  [3] parallel: %d job(s) over %d worker(s), spool=%s",
              len(jobs), workers, spool)

    inflight: Dict[int, int] = {}  # pid -> candidate index
    exits: Dict[int, int] = {}  # candidate index -> raw waitpid status
    try:
        # EVERY worker is launched and reaped before ANY result is merged, and
        # that ordering is the determinism guarantee rather than an accident of
        # structure: the parent does nothing at all between forks, so every
        # child inherits byte-identical parent state -- same `_POLICY_STORE`,
        # same adapter cache, same live config. Merging eagerly (as results
        # arrive, to keep the budget hot) would mean a candidate forked late
        # inherited its siblings' writes, i.e. its inputs would depend on how
        # fast its neighbours ran.
        #
        # UNDER A CAP THIS COSTS NOTHING, because the wave is sized to the cap
        # BEFORE the fork (see `jobs` above), so the two paths evaluate the
        # same candidates. Forking past the cap and discarding the results
        # would keep counter parity but not work parity: the discarded
        # candidates' evaluation rollouts are real env interaction no artifact
        # records (measured at cap=1: sequential evaluates one candidate, an
        # unsized parallel wave sixteen). Both properties hold together: every
        # worker is still launched and reaped before any result merges, so
        # every child inherits byte-identical parent state, and eager merging
        # is rejected for the reason above -- sizing the wave forks FEWER
        # children, it does not merge earlier.
        # -- fork or spawn: decided ONCE for the wave, not per candidate ----
        # Per wave and not per candidate because every worker in a wave must
        # inherit byte-identical parent state; two start mechanisms in one
        # wave would be two different inheritances and the equality with
        # `sequential` would be comparing a mixture.
        _spawn = _worker_is_batched(ctx.env)
        _spec = getattr(run_one, "worker_spec", None)
        if _spawn and _spec is None:
            # NO FALLBACK TO FORK. Forking this env is precisely the failure
            # being avoided -- `CUDA error: initialization error` on a GPU,
            # and on CPU a deadlock that appears only once the parent has
            # exercised jax, so a fallback would look fine in every cheap
            # test and hang the real run. A caller that builds `run_one`
            # without declaring its inputs is a bug in the caller.
            raise RuntimeError(
                "candidate_parallelism=parallel on a batched env needs spawned "
                "workers, but `run_one` carries no `worker_spec`; the caller "
                "that built it must declare it (see bird.py's train stage)")

        # THE PARENT'S CACHE COUNT, BEFORE THE WAVE. Paired with the same
        # reading after it (journalled as `worker_wave`), it shows whether the
        # parent compiled anything: if the work really runs in the spawned
        # interpreters, the PARENT compiles nothing across a wave and this
        # pair is equal.
        # `None` when no persistent cache is configured, and it stays None
        # rather than becoming 0 -- a run that measured nothing must not
        # look like a run that proved the property.
        from bird.xla_env import compilation_cache_entries as _cache_n
        _cache_before = _cache_n() if _spawn else None

        pending = list(jobs)
        while pending or inflight:
            while pending and len(inflight) < workers:
                idx = pending.pop(0)
                # Flush BEFORE forking. A block-buffered log file is inherited
                # whole by every child and every child flushes it on the way
                # out, so an unflushed line would be duplicated once per
                # worker.
                _flush_all_streams()
                if _spawn:
                    # ONE PRIMITIVE DIFFERS. The pid goes into the same
                    # `inflight`, is reaped by the same `os.waitpid(-1, 0)`
                    # below, and its payload is read by the same
                    # `_read_payload` -- so the wave sizing, the cap trip and
                    # the index-ordered reassembly are not merely "also
                    # correct" for spawned workers, they are the same code.
                    pid = _spawn_worker(
                        ctx, idx, candidates[idx], spool,
                        worker_inputs(ctx, idx, candidates[idx], spool,
                                      state=state, **_spec))
                else:
                    pid = os.fork()
                    if pid == 0:  # -- child; never returns --
                        _child_main(ctx, idx, candidates[idx], run_one, spool)
                inflight[pid] = idx
            pid, status_code = os.waitpid(-1, 0)
            if pid in inflight:
                exits[inflight.pop(pid)] = status_code

        if _spawn:
            _note_worker_wave(ctx, len(jobs), _cache_before, _cache_n())

        # -- reassembly, strictly in candidate index order --
        #
        # Every worker-level failure is COUNTED as well as recorded, in
        # candidate order, and `_report_wave_failures` reads the tally: one
        # dead worker is one failed candidate, a wave of them is a dead run.
        wave_failures: List[Tuple[str, str, str]] = []
        for idx in jobs:
            c = candidates[idx]
            payload = _read_payload(spool, idx)
            if payload is None or "fatal" in (payload or {}):
                # A fatal worker still stepped the env before it died. `payload
                # is None` carries nothing to charge and charges nothing; a
                # `fatal` payload usually carries its budget delta and does.
                _charge_discarded_env_steps(ctx, payload, getattr(c, "cand_id", "?"))
                results[idx] = _worker_failure(c, exits.get(idx, 0), payload)
                # LOGGED, like the malformed and wrong-device branches below:
                # recording the reason for a process that died (a payload never
                # written) only on the `TrainResult` would let a run whose every
                # worker segfaulted say nothing at all in its own log.
                log.error("worker %s: %s", getattr(c, "cand_id", "?"), results[idx].error)
                wave_failures.append((getattr(c, "cand_id", "?"), results[idx].error,
                                      str((payload or {}).get("traceback") or "")))
                on_result(c, results[idx])
                continue
            bad = payload_problems(payload)
            if bad:
                # A malformed payload is ONE failed candidate, not a KeyError
                # out of the train stage.
                #
                # THE RAW KEY SET TRAVELS IN THE REASON, not in a `ctx.event`,
                # because this module may not journal:
                # `tests/test_parallelism.py::test_the_train_stage_touches_
                # only_cfg_env_budget` AST-checks that `training.py` reads only
                # `ctx.cfg`, `ctx.env` and `ctx.budget`, because a forked
                # worker sees a private copy of anything else. The check is
                # module-wide and this particular line runs in the parent, so
                # it would work -- but weakening a guard that protects
                # every other line in the file to save a journal call is a bad
                # trade. The reason lands in `TrainResult.error`, which is
                # written to the candidate's record, so the information is in
                # the artifact either way.
                #
                # The key set matters as much as the reason: eight of nine
                # keys is a truncated write, a wholly different set is a
                # child running different code from its parent, and a reader hours
                # later cannot tell those apart from "missing key(s): candidate".
                reason = payload_rejection(payload, bad)
                log.error("worker %s: %s", getattr(c, "cand_id", "?"), reason)
                _charge_discarded_env_steps(ctx, payload, getattr(c, "cand_id", "?"))
                results[idx] = _worker_failure(c, exits.get(idx, 0), {"fatal": reason})
                wave_failures.append((getattr(c, "cand_id", "?"), results[idx].error, ""))
                on_result(c, results[idx])
                continue
            res = _merge_child(ctx, c, payload)
            wrong_device = learner_device_mismatch(ctx.cfg, res)
            if wrong_device:
                # Folded first, THEN refused: the stores and the budget delta
                # this worker produced are real and the parent's accounting
                # must not silently lose them, but the RESULT is not one the
                # report may contain. The candidate fails with the reason, the
                # run continues, and the cost column still says what was spent
                # -- which is the honest shape for "we paid for this and it
                # measured the wrong thing".
                log.error("worker %s: %s", getattr(c, "cand_id", "?"), wrong_device)
                res = _worker_failure(c, exits.get(idx, 0), {"fatal": wrong_device})
                # A refusal, not a death -- but the wave still yielded nothing
                # foldable from this candidate, and a wave of refusals reaches
                # stage 4 exactly as empty as a wave of corpses.
                wave_failures.append((getattr(c, "cand_id", "?"), res.error, ""))
            results[idx] = res
            on_result(c, res)
            if payload.get("exceeded"):
                # The child tripped its own (fork-time, therefore pessimistic)
                # copy of the cap. Its local count is always <= the parent's
                # merged count at the same index, so a local trip implies a
                # merged trip at or before here; `_merge_child` already
                # re-checked, and if it did not fire the cap is genuinely not
                # reached yet, so the child's partial work stands.
                raise BudgetExceeded(str(payload["exceeded"]))

        # BEFORE the cap's raise below: a wave that trained nothing is a
        # worse answer than a wave that stopped early, and the worker's own
        # error is what a reader needs first.
        _report_wave_failures(len(jobs), wave_failures)

        # THE CAP MUST STILL FIRE WHEN THE WAVE WAS TRIMMED. The crossing
        # candidate does not run, so no child trips its own fork-time copy of
        # the cap: the wave simply ends, `merge_delta`'s post-hoc `>` never
        # fires at exactly the cap, and without this raise the run would
        # continue into the next iteration while sequential stopped (measured
        # without it: parallel 2 trainings against sequential's 1 at cap=1).
        if _dropped:
            raise BudgetExceeded(
                "policy trainings %d would exceed cap %s: %d candidate(s) of "
                "this iteration were not run"
                % (ctx.budget.policy_trainings, ctx.budget.max_policy_trainings,
                   len(_dropped)))
    finally:
        # Never leave workers behind: a parent that raises here (BudgetExceeded
        # is the expected case) would otherwise orphan whatever is still
        # running. PDEATHSIG covers the parent dying; this covers it living.
        for pid in list(inflight):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        for pid in list(inflight):
            with contextlib.suppress(OSError):
                os.waitpid(pid, 0)
        shutil.rmtree(spool, ignore_errors=True)

    return [r for r in results if r is not None]


# -- the child ------------------------------------------------------------


def _flush_all_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    for handler in list(logging.getLogger().handlers):
        with contextlib.suppress(Exception):
            handler.flush()


def _pin_one_thread() -> None:
    """One BLAS/OpenMP thread per worker, always -- see (d) in `parallelism_parallel`.

    The env vars are read by libgomp/BLAS at their lazy first use, which is why
    `torch.set_num_threads(1)` is also needed for an already-initialised
    runtime. `torch` is looked up in `sys.modules` and NEVER imported: a mock
    LLM + toy env run must not pay 750 MB of RSS for a knob it does not use.
    Measured `torch.get_num_threads() == 16` on a 32-core machine with no env
    set, so N unpinned workers on a 224-core node would ask for N x 112
    threads.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    torch = sys.modules.get("torch")
    if torch is not None:
        with contextlib.suppress(Exception):
            torch.set_num_threads(1)


def _set_pdeathsig() -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, ctypes.c_ulong(signal.SIGKILL), 0, 0, 0)
    except Exception:  # pragma: no cover - non-Linux, or no libc
        pass


def build_child_payload(ctx: Any, candidate: Candidate,
                        run_one: Callable[[Candidate], TrainResult]) -> Dict[str, Any]:
    """Run one candidate and build the payload a worker sends home.

    SEPARATE FROM `_child_main`, and the separation is the point: the forked
    and the spawned worker paths must agree about
    what a payload IS. Two copies would drift the moment a key is added, and
    the symptom would be a parent folding a key the worker never sent --
    silently, because `_merge_child` reads keys it finds and ignores what it
    does not.

    Takes its own before-snapshots so a caller cannot get the ordering wrong:
    every delta here is "what changed while `run_one` ran", and a snapshot taken
    a line too late reports a store write as pre-existing.

    WHAT THIS DOES **NOT** DO, and any other caller must do for itself. The
    child-only setup stays in `_child_main` because it is about being a forked
    process, not about building a payload: `_set_pdeathsig`, `_pin_one_thread`,
    `_IN_TRAIN_WORKER = True` (which stops a worker forking seeds of its own --
    any other worker wants this too and must set it), `ctx.rundir = None` and the
    `_NoTracker`. A caller that skips those gets a valid payload from a process
    that may also be writing into the run dir.
    """
    # ARM THE EVENT BUFFER BEFORE THE WORK. `Context.event` is a no-op when
    # `rundir is None`, which is every worker -- so without a buffer a
    # candidate's journal lines would be written into nothing: a run would
    # show one row where K+1 were emitted, missing the K per-candidate ones.
    # Collect here, replay in `_merge_child`, so the parent's journal carries
    # what the worker saw.
    #
    # Not a tracker and not a second journal: a list of (stage, fields) in the
    # payload, replayed verbatim by the one process that owns the file.
    if getattr(ctx, "event_buffer", None) is None:
        ctx.event_buffer = []
    before = ctx.budget.snapshot()
    pol_before = dict(_POLICY_STORE)
    rep_before = dict(_REPLAY_STORE)
    rr_before = dict(_ROUND_REPLAY)
    code_before = dict(_POLICY_CODE)
    exceeded = ""
    try:
        result = run_one(candidate)
    except BudgetExceeded as exc:
        # Still send the partial delta home: the trainings this worker DID
        # launch were paid for, and dropping them would under-report the
        # cost column at exactly the moment a cap is being enforced.
        exceeded = str(exc)
        result = TrainResult(cand_id=candidate.cand_id, candidate=candidate,
                             trained=False, error=f"budget: {exc}")

    export = getattr(ctx.env, "export_states", None)
    env_states = None
    if export is not None:
        # `result.trajectories` only, never `store_trajectories`: exporting
        # 100 x horizon store rows would FIFO-flush the MetaWorld LRU in the
        # parent, evicting the rollout-0 rows the videos need. Nothing
        # replays a store rollout through the adapter -- §2 re-scores and §6
        # renders straight off the Trajectory arrays.
        rows = [t.states for t in result.trajectories if t.states is not None]
        with contextlib.suppress(Exception):
            env_states = export(rows)

    payload = {
        "result": result,
        # The worker mutates the Candidate (`_run_backend` fills
        # `component_names` from the compiled reward). Across a process
        # boundary that mutation is lost, which would leave
        # `candidates/*/meta.json` recording `component_names: []` while
        # `TrainResult.candidate` -- the same candidate -- has them, and
        # `selection.py` scores on `len(component_names)`. Sent home and
        # replayed onto the parent's object in `_merge_child`.
        "candidate": candidate,
        "budget": ctx.budget.delta_since(before),
        "policy": [(k, v) for k, v in _POLICY_STORE.items()
                   if k not in pol_before or pol_before[k] is not v],
        "replay": [(k, v) for k, v in _REPLAY_STORE.items()
                   if k not in rep_before or rep_before[k] is not v],
        # A shared_population slice's export (`_ROUND_REPLAY["new:..."]`).
        # Empty on every unsliced run.
        "round_replay": [(k, v) for k, v in _ROUND_REPLAY.items()
                         if k not in rr_before or rr_before[k] is not v],
        "code": [(k, v) for k, v in _POLICY_CODE.items()
                 if k not in code_before or code_before[k] != v],
        "env_states": env_states,
        "exceeded": exceeded,
        # Every `ctx.event` the worker emitted, in order, for the parent to
        # replay. A list of (stage, fields) rather than formatted lines: the
        # journal's writer belongs to the parent and a worker that formatted
        # its own rows would be a second implementation of the format.
        "events": list(getattr(ctx, "event_buffer", None) or ()),
    }
    return payload


# ---------------------------------------------------------------------------
# The worker payload: what a candidate worker needs when it does NOT inherit
# ---------------------------------------------------------------------------
#
# A FORKED child inherits the parent's whole process image, so `_child_main`
# takes live objects. A SPAWNED child inherits nothing and must be handed the
# same information as data. On the batched tier the fork is the bug
# (measured on an A100 under `--profile full`: four of four fasttd3 children
# died at their first CUDA call after forking a parent that had taken CUDA
# through jax), so that tier gets a fresh interpreter instead.
#
# ONE FUNCTION NAMES THE INPUTS, AND BOTH PATHS READ IT. If the parent built
# the spawn payload from one list of fields and the fork child read another,
# the two would drift and the drift would show up only as a bit-identity
# failure with no indication of which field moved. `tests/` pins that they
# agree by constructing `run_one` both ways from one ctx and comparing.
#
# BY VALUE, NOT RE-DERIVED. Whatever a forked child inherits by value travels
# by value here -- `ctx.rng`'s STATE included. The working assumption is that
# every randomness source in a candidate derives from its explicit per-
# candidate seed, but that is the hypothesis the bit-identity instrument
# exists to test, and a worker built on it would rest its correctness on the
# thing being measured. Carrying the state makes spawn bit-identical BY
# CONSTRUCTION; whether the state is actually needed is a separate experiment
# (drop the restore, see whether the equality still holds).
def worker_inputs(ctx: Any, idx: int, candidate: Candidate, spool: str,
                  state: Any = None, backend_name: str = "",
                  hpsearch_name: str = "",
                  seed_plan: Optional[Dict[str, int]] = None,
                  default_n_seeds: int = 1) -> Dict[str, Any]:
    """Everything a candidate worker needs, named ONCE, for either path.

    NOT `build_child_payload`, which is its mirror and lives below: that one
    is what a worker sends HOME (the TrainResult and its bookkeeping), this
    one is what a worker is GIVEN. Both are "the payload" in conversation and
    they travel in opposite directions; the names are worth keeping apart.

    Returns plain data: the spawn path serialises this, and the fork path
    reads the same keys off the same ctx so the two cannot disagree about
    what a worker is entitled to.

    NOT INCLUDED, deliberately: `rundir` and `tracker`, which `_child_main`
    already nulls (a worker must not write into the run dir, and a child
    holding the tracker handle can mark the PARENT's run finished); and
    `generator`/`evaluator`, because stage 3 makes no LLM call.
    """
    return {
        "idx": int(idx),
        "spool": str(spool),
        "candidate": candidate,
        # NO `hasattr` FALLBACK TO None. `Config.to_dict` (config.py) and
        # `Budget.snapshot` (budget.py) both exist; guarding them would turn a
        # missing accessor into a silent None that the child would then
        # hydrate from, producing a worker running a config nobody wrote. A
        # renamed accessor should raise here, loudly, in the parent.
        "config_resolved": ctx.cfg.to_dict(),
        "config_source": ctx.cfg.source,
        "rng_state": ctx.rng.getstate(),
        "budget_snapshot": ctx.budget.snapshot(),
        # THE WALLCLOCK ORIGIN TRAVELS, even though no cap reads it.
        # `Budget._check` tests policy_trainings, llm_calls and gpu_hours and
        # not time, and `counter_names()` excludes `wallclock_start`, so this
        # value reaches no delta and no cap -- today. It travels anyway
        # because a FORKED child inherits it by value, so anything in a
        # component that reads `ctx.budget.wallclock_s` reads the whole
        # search's elapsed there; a spawned worker rebuilding a fresh Budget
        # would read ~0 and the two paths would quietly disagree. Sending it
        # costs one float and removes the disagreement at the source rather
        # than relying on nothing ever reading the clock.
        "budget_elapsed_s": ctx.budget.wallclock_s,
        "counters": dict(getattr(ctx, "counters", {}) or {}),
        # -- what `run_one` CLOSES OVER, and it is only these ---------------
        # `bird.py`'s `run_one` is
        #     hpsearch(ctx, state, c, backend, n_seeds=plan.get(c.cand_id, ...))
        # so its whole closure is ctx, state, the backend, and the seed plan.
        # Naming them here rather than having the child re-derive them is the
        # point: a second derivation is a second implementation of one
        # construction, inside the thing that must stay bit-identical, and it
        # would surface only as an equality failure with no clue which input
        # moved. The child resolves `backend` from the registry BY NAME, which
        # is data; the callable itself is not sendable and must not be.
        "state": state,
        "backend_name": backend_name,
        "hpsearch_name": hpsearch_name,
        "seed_plan": dict(seed_plan or {}),
        "default_n_seeds": int(default_n_seeds),
    }


def _child_main(ctx: Any, idx: int, candidate: Candidate,
                run_one: Callable[[Candidate], TrainResult], spool: str) -> None:
    """Runs in the forked child. Writes one pickle and calls `os._exit`.

    `os._exit`, not `sys.exit`: a normal interpreter shutdown would run atexit
    hooks the child inherited but does not own. `wandb.init` (which
    `output.tracker: wandb` runs long before stage 3) installs one that would
    mark the parent's run finished from a child.
    """
    code = 0
    try:
        _set_pdeathsig()
        _pin_one_thread()
        # Seeds never fork inside a candidate worker: k candidate workers each
        # forking s seed workers would oversubscribe the allocation k-fold.
        global _IN_TRAIN_WORKER
        _IN_TRAIN_WORKER = True
        # A worker must not write into the run dir or the tracker. Nothing in
        # this module does today -- every artifact write is in `bird.py`'s
        # parent-side `run_iteration` -- and this makes that a property of the
        # child rather than an ongoing convention.
        ctx.rundir = None
        # The same no-op tracker a Context carries before `bird.run()` installs
        # a real one, reused rather than re-declared. `wandb.init` has already
        # run by stage 3 under `output.tracker: wandb`, and a child holding a
        # copy of the run handle can mark the PARENT's run finished.
        ctx.tracker = _NoTracker()

        payload = build_child_payload(ctx, candidate, run_one)
        _write_payload(spool, idx, payload)
    except BaseException as exc:  # noqa: BLE001 - the child owns every failure
        try:
            _write_payload(spool, idx, {"fatal": f"{type(exc).__name__}: {exc}",
                                        "traceback": traceback.format_exc()})
        except BaseException:  # pragma: no cover - spool unwritable
            code = 70
    finally:
        os._exit(code)


def _worker_is_batched(env: Any) -> bool:
    """Does this run's env need a SPAWNED worker rather than a forked one?

    CAPABILITY, NOT PREFIX. The `jax_` prefix is how `env_suite` groups ids;
    it is not a fact about the object, and a batched adapter registered
    without the prefix would fork and deadlock. The question this asks is the
    one that matters: is this env the batched kind whose backend cannot
    survive a fork.

    Asked HERE and not at config load, because at config load it cannot be
    asked at all -- a jax id may register as a factory FUNCTION (`jax_toy`
    does), so there is no instance to test until `run()` has built one. That is why
    `config.py` states its jax-env rules by id (through `env_suite`) and this
    states the routing by type: the same fact, checked at the only two places
    each is knowable.

    Importing `jax_base` costs nothing on a non-jax run: the module imports
    numpy and `.base` and deliberately NOT jax, so that a parent choosing a
    worker start does not make jax a hard dependency of forking. That is a
    property `tests/test_jax_base.py` already pins, not an assumption.
    """
    from bird.envs.jax_base import BatchedEnvAdapter
    return isinstance(env, BatchedEnvAdapter)


def _spawn_worker(ctx: Any, idx: int, candidate: Candidate, spool: str,
                  given: Dict[str, Any]) -> int:
    """Start one candidate worker as a FRESH INTERPRETER. Returns its pid.

    `posix_spawn`, not `subprocess.Popen`, and the reason is the reaping loop
    around the call site: it is `os.waitpid(-1, 0)` over an `inflight` dict,
    and a `Popen` object reaps its own child, so the two would race and the
    loop would block forever on a pid Popen had already collected. A
    posix_spawned child is an ordinary child of this process, so EVERY line
    of the launch/reap/reassemble machinery below stays byte-identical
    between the two paths and only the start primitive differs. That is the
    whole design: one fork/spawn branch, not two schedulers.

    THE CHILD MUST IMPORT THE SAME `bird` THIS PARENT IS RUNNING. If the
    package is installed editable from a different checkout, a child left to
    resolve `bird` for itself would silently run DIFFERENT CODE from its parent -- in the one path required to stay bit-identical,
    and invisibly, because both trees import and both produce a payload. The
    parent's own package directory is therefore put at the FRONT of the
    child's PYTHONPATH, and the child asserts it got it.
    """
    import bird as _bird_pkg
    inputs_path = os.path.join(spool, f"inputs-{idx}.pkl")
    tmp = inputs_path + ".part"
    with open(tmp, "wb") as fh:
        pickle.dump(given, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, inputs_path)          # the worker never sees a half-file

    tree = os.path.dirname(os.path.dirname(os.path.abspath(_bird_pkg.__file__)))
    env = dict(os.environ)
    env["PYTHONPATH"] = tree + (os.pathsep + env["PYTHONPATH"]
                                if env.get("PYTHONPATH") else "")
    # A REAL PROGRAM, not a semicolon one-liner. One `-c` expression with a
    # conditional `sys.exit` inside it would be unreviewable: the guard below
    # is the thing standing between a two-tree run and a silent bit-identity
    # failure, and a guard nobody can read by eye is not a guard.
    code = "\n".join([
        "import os, sys",
        "import bird",
        f"_expected = {tree!r}",
        "_actual = os.path.dirname(os.path.dirname(os.path.abspath(bird.__file__)))",
        "if _actual != _expected:",
        "    sys.stderr.write('bird worker: imported bird from %s but the "
        "parent runs %s; refusing rather than running different code in a "
        "path required to be bit-identical\\n' % (_actual, _expected))",
        "    os._exit(71)",
        "from bird.components.training import _spawn_child_main",
        f"_spawn_child_main({inputs_path!r})",
    ])
    return os.posix_spawn(sys.executable, [sys.executable, "-c", code], env)


def _cfg_from(given: Dict[str, Any]) -> Any:
    """The worker's Config, built ONCE from the payload.

    A function because two callers need it at two different moments --
    `_spawn_child_main` before it sets the XLA flags, `_hydrate_ctx` when it
    builds the Context -- and a second `Config(...)` spelled out in each
    would be two constructions of the run's identity.
    """
    from bird.config import Config
    return Config(given["config_resolved"], source=given["config_source"])


def _hydrate_ctx(given: Dict[str, Any], cfg: Any = None) -> Any:
    """Rebuild the worker's `Context` from `worker_inputs`, and construct its env.

    A DICT RATHER THAN THE RESOLVED-CONFIG FILE. `Config(data, source)` takes a
    plain dict and `Config(cfg.to_dict())` round-trips with the HASH PRESERVED
    -- measured, and the hash is the run id, so a worker whose config hashed differently
    would be a different run. Carrying the dict then beats reading the file
    on three counts: the worker never touches the run dir, which
    `_child_main` nulls precisely so a worker cannot; it does not re-run
    `validate()`, which the parent has already run on the same dict; and it
    cannot read a file some other process has since rewritten.

    `ctx.rng` IS RESTORED FROM THE PARENT'S STATE, not re-seeded. A forked
    child inherits the generator mid-stream; re-seeding from `cfg["seed"]`
    would hand the worker a different stream and break bit-identity in a way
    that looks like a science change. Whether any candidate actually draws
    from it is the separate experiment.
    """
    import random as _random
    from bird.budget import Budget
    from bird.context import Context
    from bird import registry

    cfg = _cfg_from(given) if cfg is None else cfg
    # `restore` SETS rather than adds (budget.py), which is what a worker
    # wants: it starts from a fresh Budget and the snapshot is the total.
    # No `hasattr` guard -- `Budget.restore` exists, so a guard could only
    # convert a rename into a worker silently running with zeroed counters
    # and therefore a cap it cannot hit.
    budget = Budget.from_config(cfg)
    budget.restore(given["budget_snapshot"], given["budget_elapsed_s"])

    rng = _random.Random()
    rng.setstate(given["rng_state"])

    ctx = Context(cfg=cfg, budget=budget, rng=rng)
    # The same two nullings `_child_main` does, for the same two reasons: a
    # worker must not write into the run dir, and a child holding the real
    # tracker can mark the PARENT's run finished from an atexit hook.
    ctx.rundir = None
    ctx.tracker = _NoTracker()
    ctx.counters = dict(given.get("counters") or {})

    # env_construct's boundary is THIS call returning: for an adapter that
    # builds and compiles device state in its constructor, that compile lands
    # here, and a stated boundary beats a clean-looking split.
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    # `problem.horizon`: truncate the episode to the method's own length, or
    # keep the environment's. `apply_horizon` is the ONE place the rule lives
    # (bird/envs/base.py) and it runs immediately after construction, before
    # anything reads `env.horizon`.
    from ..envs.base import apply_horizon as _apply_horizon
    _apply_horizon(ctx.env, cfg)  # (effective, requested); the seed rows read the adapter
    return ctx


def _rebuild_run_one(ctx: Any, given: Dict[str, Any]) -> Callable[..., TrainResult]:
    """`bird.py`'s `run_one`, reconstructed in a spawned worker.

    HYDRATION, NOT A SECOND DERIVATION. Every input comes from
    `worker_inputs`; nothing is recomputed from the config here. The backend
    is resolved from the registry BY NAME because a callable cannot be sent
    and must not be pickled into a worker -- the name is the data, the
    resolution is the registry's job, and both paths therefore get whatever
    `@register` currently binds rather than two snapshots of it.

    `tests/` pins that this and the parent's closure agree, by building both
    from one ctx and comparing the seed allocation and the hyperparameter
    dict. That test is the only thing standing between a future edit to one
    of the two and a silent divergence.
    """
    from bird import registry

    # BOTH are registry components, resolved BY NAME. `hpsearch` is not a
    # function but a family -- `hyperparameter_search` in KINDS, bound to
    # `hpsearch_none` / `hpsearch_grid` / `hpsearch_per_candidate_best` in
    # bird/components/search.py. `bird.py`'s train stage resolves the pair
    # exactly this way, so these stay the same construction rather than two
    # that merely agree.
    backend = registry.get("train_backend", given["backend_name"])
    hpsearch = registry.get("hyperparameter_search", given["hpsearch_name"])
    plan = given["seed_plan"]
    default_n = given["default_n_seeds"]
    state = given["state"]

    def run_one(c: Candidate, **backend_kw: Any) -> TrainResult:
        return hpsearch(ctx, state, c, backend,
                        n_seeds=plan.get(c.cand_id, default_n), **backend_kw)

    return run_one


def _spawn_child_main(inputs_path: str) -> None:
    """Entry point of a SPAWNED candidate worker: a fresh interpreter.

    Mirrors `_child_main` step for step -- the two must not drift, because
    `parallel` has to stay bit-identical to `sequential` on both paths -- with
    one addition at the front: a forked child ALREADY HAS the parent's ctx,
    and this one has to hydrate it from `worker_inputs`.

    ORDER MATTERS AT THE TOP OF THIS FUNCTION. The determinism flags must be
    set BEFORE jax is imported, because XLA reads XLA_FLAGS at backend
    initialisation; in a forked child that is impossible (jax is already in
    `sys.modules`, inherited), which is why the autotune guarantee cannot
    hold inside a forked worker. A fresh interpreter is the first place it
    can, so the call goes above every import that might pull jax in.
    """
    import json as _json
    import pickle as _pickle

    # -- 0. TRITON BEFORE ANYTHING ELSE IN THIS PROCESS --------------------
    # NOT cosmetic ordering: `bird/envs/mujoco_control.py::
    # _preload_llvm_before_mujoco` documents that the system `libOSMesa.so.8`
    # (llvmpipe) and `triton/_C/libtriton.so` EACH EMBED AN LLVM, that
    # whichever loads second gets the other's symbols interposed, and that
    # the symptom is "the crash lands at stage [3] with exit 139 and no
    # traceback". 139 is SIGSEGV.
    #
    # A FORKED child inherits a process where that race is already won,
    # because the parent built its env long ago. A SPAWNED child starts
    # clean and has to win it again -- and it loses, because by the time
    # env construction reaches the preload something in the child's startup
    # has already pulled the GL stack in. MEASURED on an A100 with a
    # MuJoCo-backed batched env: without this call every spawned worker died
    # with SIGSEGV inside `triton/knobs.py` reached from
    # `_preload_llvm_before_mujoco`, and calling that same function FIRST
    # here takes it to zero.
    #
    # Above the flag step deliberately: `set_xla_determinism_flags` only has
    # to precede JAX's import, and importing triton does not import jax, so
    # this cannot cost `applied_in_time`.
    from bird.envs.mujoco_control import _preload_llvm_before_mujoco
    _preload_llvm_before_mujoco()

    t0 = time.time()
    with open(inputs_path, "rb") as fh:
        given = _pickle.load(fh)

    # -- 1. the flags, before anything imports jax ---------------------------
    # THE CONFIG IS BUILT BEFORE THE FLAGS, and reading the env id off the
    # Config rather than off the raw dict is the point. `to_dict()` is
    # NESTED, so the dict spelling is `["problem"]["env_id"]` -- and a
    # `.get("problem", {}).get("env_id")` chain returns None the day that
    # shape changes, which would mean `prepare_for_env(None)`: no flags, no
    # error, and a tier whose determinism guarantee had quietly lapsed.
    # `cfg["problem.env_id"]` is the accessor every other reader in the repo
    # uses and it raises on a key that is not there.
    #
    # Safe to do first: `bird.config` imports yaml and numpy and no jax, so
    # building a Config cannot initialise the backend this is about to
    # configure.
    from bird.xla_env import prepare_for_env
    cfg = _cfg_from(given)
    _flags_in_time = prepare_for_env(cfg["problem.env_id"])

    t_import = time.time()
    _set_pdeathsig()
    _pin_one_thread()
    global _IN_TRAIN_WORKER
    _IN_TRAIN_WORKER = True

    code = 0
    try:
        ctx = _hydrate_ctx(given, cfg)     # env constructed here
        t_env = time.time()
        run_one = _rebuild_run_one(ctx, given)
        payload = build_child_payload(ctx, given["candidate"], run_one)
        # WORKER_STARTUP_S AS FOUR FIELDS, not one number and not a comment:
        # the scheduling question on this tier is exactly whether the
        # per-worker env construction or the first compile dominates, and a
        # comment cannot be plotted across runs. `total` is the headline
        # number; the parts are why. The env_construct/first_compile boundary
        # is the code's own call boundary -- env_construct ends when the
        # adapter's constructor returns -- and for an adapter that compiles
        # device state in its constructor, that span MAY INCLUDE the compile.
        # A stated boundary beats a clean-looking split.
        payload["worker_start"] = "spawn"
        payload["worker_startup_s"] = {
            "import": round(t_import - t0, 3),
            "env_construct": round(t_env - t_import, 3),
            "first_compile": None,         # filled by the view's first step_batch
            "total": round(t_env - t0, 3),
        }
        # `applied_in_time` is the THREE-STATE reader, not a bool: None means
        # this tier never asked, False means someone asked too late. Coercing
        # to bool merges those, and "the flag was not applied" then reads as
        # "the flag failed" on a run that simply was not a jax run.
        from bird.xla_env import (applied_in_time, compilation_cache_entries,
                                  preallocate_setting)
        payload["xla_env"] = {"applied_in_time": applied_in_time(),
                              "prepare_returned": bool(_flags_in_time),
                              # Recorded from the WORKER's own process so a
                              # run shows whether parent and workers agree;
                              # otherwise a disagreement (a 30 GiB parent and
                              # a worker that needed 8) is invisible.
                              "preallocate": preallocate_setting()}
        # This worker's own view of the persistent cache. Paired with the
        # PARENT's before/after count around the wave, the two answer the
        # question the spawn worker exists for: the compiles happened here,
        # in a fresh interpreter, and not in the parent.
        payload["jax_cache"] = {"entries": compilation_cache_entries()}
        _write_payload(given["spool"], given["idx"], payload)
    except BaseException as exc:  # noqa: BLE001 - the worker owns every failure
        try:
            _write_payload(given["spool"], given["idx"],
                           {"fatal": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "worker_start": "spawn"})
        except BaseException:  # pragma: no cover - spool unwritable
            code = 70
        else:
            code = 1
    finally:
        # IN A `finally`, mirroring `_child_main`, and for a WEAKER reason
        # that is worth stating rather than copying silently. There, `os._exit`
        # avoids running atexit hooks the child inherited but does not own --
        # chiefly wandb's, which can mark the PARENT's run finished. A SPAWNED
        # worker inherits no hooks: it never called `bird.run()`, so
        # `wandb.init` has not run in it. What survives is the other half: an
        # unwritable spool must still produce a distinct exit code rather than
        # a traceback on stderr that the parent reads as a mystery death, and
        # a worker must never exit through a path that could block.
        os._exit(code)


def _write_payload(spool: str, idx: int, payload: Dict[str, Any]) -> None:
    tmp = os.path.join(spool, f"{idx}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    # Rename last so the parent can never read a half-written pickle from a
    # child that died mid-dump.
    os.replace(tmp, os.path.join(spool, f"{idx}.pkl"))


def _read_payload(spool: str, idx: int) -> Optional[Dict[str, Any]]:
    path = os.path.join(spool, f"{idx}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as exc:  # noqa: BLE001
        return {"fatal": f"unreadable worker payload: {type(exc).__name__}: {exc}"}


def _describe_exit(status_code: int) -> str:
    if os.WIFSIGNALED(status_code):
        sig = os.WTERMSIG(status_code)
        try:
            name = signal.Signals(sig).name
        except ValueError:  # pragma: no cover
            name = str(sig)
        return f"killed by signal {sig} ({name})"
    return f"exited with status {os.WEXITSTATUS(status_code)}"


def _worker_failure(candidate: Candidate, status_code: int,
                    payload: Optional[Dict[str, Any]]) -> TrainResult:
    """A worker that died is ONE failed candidate, not a failed iteration.

    This is strictly better than sequential and the improvement is the point: a
    MetaWorld SIGSEGV under `sequential` kills the interpreter, bypasses
    `bird.py::run`'s `except BaseException -> rundir.finish("failed")` and
    leaves `status.json` reading `running` forever. Here it is a `TrainResult`
    with a greppable `worker:` prefix, the run continues, and stage 4 scores it
    at `select.failure_value` like any other failed candidate.
    """
    if payload and payload.get("fatal"):
        reason = payload["fatal"]
    else:
        reason = _describe_exit(status_code)
    return TrainResult(cand_id=candidate.cand_id, candidate=candidate, trained=False,
                       error=f"worker: {reason}")


#: Keys a payload may be MISSING without being malformed.
#:
#: `events`, because it only carries journal lines for the parent to replay: a
#: payload without them is still foldable, and treating their absence as malformed
#: would cost a whole candidate for a missing log line, which is the cost this
#: validator exists to avoid, not to cause.
#: The spawn-only keys join it for a structural reason: a FORKED worker never
#: sends them at all, so they are absent on the majority path by design. Adding them as required
#: would fail every forked candidate; omitting them from `PAYLOAD_FIELDS`
#: entirely would fail every SPAWNED one with "unexpected key(s)" -- each
#: charged for the env steps it had really spent, and the run reporting
#: "0 trained, 16 failed" as though the rewards were bad. A unit test of the
#: worker alone cannot see it: the payload is well-formed in the worker and
#: the disagreement lives in the parent.
PAYLOAD_OPTIONAL: Tuple[str, ...] = (
    "events", "worker_start", "worker_startup_s", "xla_env", "jax_cache",
)

#: The keys `build_child_payload` produces and `_merge_child` folds.
#: Named once, so the builder and the validator cannot drift into two different
#: answers about what a payload IS.
PAYLOAD_FIELDS: Tuple[str, ...] = (
    "result", "candidate", "budget", "policy", "replay", "round_replay",
    "code", "env_states", "exceeded", "events",
    # -- instrumentation a SPAWNED worker adds; never sent by a forked one --
    # `worker_start` ("fork"/"spawn"), `worker_startup_s` (the four-part
    # import/env_construct/first_compile/total split) and `xla_env` (whether
    # the determinism flags were set before the backend initialised, which
    # is only answerable in a fresh interpreter).
    "worker_start", "worker_startup_s", "xla_env", "jax_cache",
)


def payload_problems(payload: Any) -> List[str]:
    """Every reason this payload may not be folded. Empty list = clean.

    WHY THIS EXISTS. `_merge_child` reads `payload["result"]` and
    `payload["candidate"]` unguarded, so without this gate a worker that
    returned a malformed or partial payload would raise `KeyError` **inside
    the parent's train stage** and take the whole run with it. That is exactly
    the class `_worker_failure` exists for: a worker that died is ONE failed
    candidate, not a failed iteration.

    A format mismatch must cost one candidate, never one run.

    Reports EVERY problem rather than the first, like `Config.validate()`: the
    failure text goes into a `TrainResult` a human reads hours later, and
    "missing `candidate`" when the truth is "missing five keys and `result` is
    a str" sends them to the wrong side of the boundary.

    Deliberately shallow. It checks the SHAPE `_merge_child` dereferences --
    the nine keys, the two it reads unguarded, the four it iterates as
    (key, value) pairs -- and does not validate the contents of the stores.
    A store entry with wrong bytes in it is a different failure that this
    cannot catch and should not pretend to.
    """
    if not isinstance(payload, dict):
        return [f"payload is {type(payload).__name__}, expected dict"]

    problems: List[str] = []
    missing = [k for k in PAYLOAD_FIELDS
               if k not in payload and k not in PAYLOAD_OPTIONAL]
    extra = sorted(set(payload) - set(PAYLOAD_FIELDS))
    if missing:
        problems.append("missing key(s): " + ", ".join(missing))
    if extra:
        # Not fatal in itself -- `_merge_child` ignores what it does not read
        # -- but a payload with an unexpected key was built by code that
        # disagrees with this one about the format, which is worth the same
        # refusal as a missing key rather than a silent drop.
        problems.append("unexpected key(s): " + ", ".join(extra))

    for name in PAYLOAD_FIELDS:
        if name not in payload:
            continue        # already reported above; do not report it twice
        problems.extend(_field_problems(name, payload[name]))

    # THE CROSS-FIELD CHECK, and it is the one a per-field pass cannot make:
    # the result must be ABOUT the candidate it travels with. A payload whose
    # `result.cand_id` names a different candidate folds cleanly and attributes
    # one candidate's training to another -- no key is missing, no type is
    # wrong, and every downstream number is a real measurement of the wrong
    # thing.
    result, cand = payload.get("result"), payload.get("candidate")
    if isinstance(result, TrainResult) and isinstance(cand, Candidate):
        if result.cand_id and cand.cand_id and result.cand_id != cand.cand_id:
            problems.append(
                f"result.cand_id={result.cand_id!r} but candidate.cand_id="
                f"{cand.cand_id!r} -- this payload is about a different candidate")
    return problems


#: Per-field shape: the types a value may take, and whether EMPTY is a legal
#: value for it. Keyed from `PAYLOAD_FIELDS` and asserted to cover it exactly
#: (`tests/test_worker_payload_validation.py`), so a tenth field cannot be
#: added to the protocol and silently go unvalidated.
#:
#: EMPTINESS IS PER FIELD, NOT A BLANKET RULE, and getting that wrong in either
#: direction is a defect. `policy: []` is the normal case (most candidates
#: write no store entry), `exceeded: ""` is the normal case (most workers do
#: not trip the cap), and refusing those would fail every healthy run. But
#: `result: None` and `candidate: None` are presence without a value -- the
#: shape `_merge_child` dereferences and the shape a truncated write produces
#: -- so for those two, empty is exactly what must be refused.
_PAYLOAD_SHAPE: Dict[str, Tuple[Tuple[type, ...], bool]] = {
    "result": ((TrainResult,), False),
    "candidate": ((Candidate,), False),
    "budget": ((dict,), True),
    "policy": ((list, tuple), True),
    "replay": ((list, tuple), True),
    "round_replay": ((list, tuple), True),
    "code": ((list, tuple), True),
    # `env_states` is whatever the adapter's `export_states` returned, which is
    # adapter-private; None is the normal case (most adapters export nothing).
    # Typed as "anything" deliberately rather than guessed at.
    "env_states": ((object,), True),
    "exceeded": ((str,), True),
    # EMPTY IS LEGAL AND IS THE COMMON CASE. A worker that journalled nothing
    # ships `events: []`, which is most of them -- the field exists so a
    # candidate's own rows are not lost, not because every candidate has
    # rows. Refusing empty here would fail a healthy run for having been
    # quiet, which is the mistake this table's docstring names in the other
    # direction for `result` and `candidate`.
    #
    # `list, tuple` and NOT a row-shape check, deliberately: `_merge_child`
    # replays each row inside its own try/except and logs a malformed one
    # rather than raising, because the training it describes is already done
    # and paid for. Validating row shape here would turn a journalling
    # defect into a refused payload, which is the opposite of that trade.
    # It is also why `events` is in `PAYLOAD_OPTIONAL`: a worker predating
    # the field must cost nothing rather than one candidate.
    "events": ((list, tuple), True),
    # -- the spawn-tier instrumentation ------------------------------------
    # Shapes declared rather than left out. `_field_problems` returns [] for
    # any key it has no entry for, so a field in `PAYLOAD_FIELDS` and absent
    # here is accepted with NO type check at all -- which is how a
    # `worker_startup_s` that arrived as the string "fast" would reach a
    # timing analysis and be plotted.
    "worker_start": ((str,), False),
    "worker_startup_s": ((dict,), False),
    "xla_env": ((dict,), False),
    "jax_cache": ((dict,), False),
}

#: The four that `_merge_child` iterates as `for key, value in ...`.
_PAYLOAD_PAIR_FIELDS = ("policy", "replay", "round_replay", "code")


def _field_problems(name: str, value: Any) -> List[str]:
    """Type, emptiness and row shape for one payload field."""
    spec = _PAYLOAD_SHAPE.get(name)
    if spec is None:            # unknown key; already reported as unexpected
        return []
    types, may_be_empty = spec
    if value is None:
        # None is "present but absent". Legal only where empty is legal --
        # `budget: None` and `exceeded: None` fold as nothing, which is what a
        # worker that did neither would mean.
        return [] if may_be_empty else [
            f"{name} is None, expected {' or '.join(t.__name__ for t in types)}"]
    if types != (object,) and not isinstance(value, types):
        return [f"{name} is {type(value).__name__}, expected "
                f"{' or '.join(t.__name__ for t in types)}"]
    if not may_be_empty and hasattr(value, "__len__") and len(value) == 0:
        return [f"{name} is empty"]
    if name in _PAYLOAD_PAIR_FIELDS:
        for row in value:
            if not (isinstance(row, (list, tuple)) and len(row) == 2
                    and isinstance(row[0], str)):
                # `_merge_child` unpacks these as `for key, value in ...`, and
                # a bare 2-character string unpacks into characters rather
                # than raising -- so this is caught by the fold SUCCEEDING and
                # writing nonsense into the store, not by it failing.
                return [f"{name} holds a row that is not (str, value)"]
    return []


def payload_rejection(payload: Any, problems: List[str]) -> str:
    """The text a rejected payload leaves in `TrainResult.error`.

    A function rather than an f-string at the call site so that the KEY SET
    half is testable as an OUTPUT. It is the half a reader needs and the half
    an assertion on `payload_problems` alone cannot see: eight of nine keys is
    a truncated write, a wholly different set is a child running different code
    from its parent, and "missing key(s): candidate" does not distinguish them.
    """
    keys = sorted(payload) if isinstance(payload, dict) else []
    return ("malformed payload: " + "; ".join(problems)
            + " (keys received: " + (", ".join(keys) or "none") + ")")

def learner_device_mismatch(cfg: Any, result: Any) -> str:
    """`""` when the learner ran where the config asked; the reason otherwise.

    WHAT THIS CATCHES IS NOT HYPOTHETICAL. `train.hyperparameters.device`
    defaults to `cpu`, so a GPU job whose overrides never set it trains on the
    CPU with the GPU idle beside it -- while a provenance record describing a
    probe tensor rather than the learner reports `cuda:0`. The seed row carries
    `device: cpu` throughout. **The fact is in the artifact, and without this
    nothing compares it against intent** -- so this compares it.

    Reads the EXISTING `device` field rather than adding one: it is written
    from the `torch.device` the networks are constructed on, so it is what a
    gradient step ran on, and a second field for one fact is how two fields
    start disagreeing.

    Only an EXPLICIT cuda request is checked. `auto` means "a GPU if there is
    one", so cpu is its correct answer rather than a failure -- the same split
    the backend itself makes.
    """
    try:
        want = str((cfg.get("train.hyperparameters") or {}).get("device") or "")
    except Exception:  # noqa: BLE001 - a stub cfg in a test
        return ""
    if not want.startswith("cuda"):
        return ""
    # `unknown` is NEITHER pass nor fail: it means the provenance could not be
    # read, not that the learner was on cpu. Refusing on it would fail a
    # healthy run whose reporting broke -- the mirror image of the defect this
    # function exists for, and a worse trade, since the failure it is guarding
    # against is at least visible in the artifact.
    rows = [r for r in (getattr(result, "seed_metrics", None) or ())
            if isinstance(r, dict) and r.get("device")
            and str(r["device"]) != "unknown"]
    if not rows:
        # No seed row names a device: the backend that ran is one that does
        # not record it (the surrogates). Silence is not evidence of a
        # mismatch, and refusing on it would fail every mock run.
        return ""
    ran = sorted({str(r["device"]) for r in rows})
    if any(d.startswith("cuda") for d in ran):
        return ""
    return (f"train.hyperparameters.device={want!r} but the learner ran on "
            f"{', '.join(ran)} -- the seed rows say so. A GPU job that silently "
            "became a CPU job is several times slower than the number it will be "
            "compared against and also loses AMP and torch.compile; refused as one "
            "failed candidate rather than folded into a report that cannot show it")


def _merge_child(ctx: Any, candidate: Candidate, payload: Dict[str, Any]) -> TrainResult:
    """Fold one worker's return values back into the parent, in candidate order.

    Callers must clear `payload_problems()` first: this function reads
    `payload["result"]` and `payload["candidate"]` unguarded, deliberately, so
    that a payload which got past the gate and is still malformed fails loudly
    here rather than folding half of itself.
    """
    result: TrainResult = payload["result"]

    # REPLAY THE WORKER'S EVENTS FIRST, before anything this fold emits, so
    # the journal reads in the order the work happened rather than in the
    # order the parent learned about it. A worker's `ctx.event` writes into
    # nothing (`Context.event` is a no-op with no run dir), so without this
    # every per-candidate row a backend journals is lost.
    #
    # Optional (`PAYLOAD_OPTIONAL`): a payload without events still folds.
    #
    # Never raises out of the fold: a malformed event row is a journalling
    # defect and the training it describes is already done and paid for.
    for row in payload.get("events") or ():
        try:
            stage, fields = row
            ctx.event(str(stage), **dict(fields))
        except Exception as exc:  # noqa: BLE001 - a worker's bad row
            log.warning("could not replay a worker event for %s: %s: %s",
                        candidate.cand_id, type(exc).__name__, exc)

    # THE WORKER'S OWN START, JOURNALLED -- and it has to be journalled by
    # SOMETHING or the three keys are declared-but-unread: they validate, they
    # travel, they reach no artifact, and the scheduling question they exist
    # to answer -- does per-worker env construction or the first compile
    # dominate this tier? -- stays unanswerable while every counter reads
    # normal.
    #
    # One row per worker, through the same `ctx.event` the replay above
    # uses, so an analysis reads it out of the run's journal beside
    # everything else rather than out of a second channel. Only a SPAWNED
    # worker sends these, so a forked run writes no row and nothing changes
    # for the numpy tier.
    if payload.get("worker_start"):
        try:
            ctx.event("worker_start",
                      cand_id=candidate.cand_id,
                      start=payload["worker_start"],
                      startup_s=payload.get("worker_startup_s") or {},
                      xla_env=payload.get("xla_env") or {},
                      jax_cache=payload.get("jax_cache") or {})
        except Exception as exc:  # noqa: BLE001 - journalling never fails a fold
            log.warning("could not journal worker start for %s: %s: %s",
                        candidate.cand_id, type(exc).__name__, exc)

    # Identity, not replacement: the parent already holds this Candidate and
    # `bird.py::run_iteration` writes `candidates/*/meta.json` from it, so the
    # artifact and `TrainResult.candidate` must be the same object or they can
    # disagree about the same candidate.
    child_c: Candidate = payload["candidate"]
    for f in dataclasses.fields(Candidate):
        setattr(candidate, f.name, getattr(child_c, f.name))
    result.candidate = candidate

    # Write order preserved, so `_store`'s 64-entry eviction sees the same
    # sequence it would have seen sequentially. Without this replay,
    # `train.init: warm_start_from_best` silently degrades to `from_scratch`
    # and `secondary_replay_buffer` to an empty buffer -- with a plausible
    # `policy_ref` string still in the artifact: RDA and GT would quietly run
    # the ablation instead of the method.
    #
    # BOTH CAPS ARE PASSED, and this is the only path on which the PARENT
    # accumulates. `_store` skips its byte-eviction loop entirely when
    # `max_bytes` is falsy, so omitting them here would leave the parent
    # bounded by `_STORE_LIMIT` alone: 40 `_ReplaySlice`s of 31.38 MB each on
    # an 8-candidate x 5-iteration GT search (`parallel`,
    # `secondary_buffer.size: 100000`) is ~1.25 GB, and 40 < 64 so nothing
    # would ever evict. It would also diverge from `sequential`, where the
    # byte cap governs eviction -- i.e. the schedule key would change which
    # buffers a later iteration can still reach, which is precisely what
    # `tests/test_parallelism.py` exists to forbid.
    for key, value in payload.get("policy") or ():
        _store(_POLICY_STORE, key, value, _POLICY_STORE_MAX_BYTES)
    for key, value in payload.get("replay") or ():
        _store(_REPLAY_STORE, key, value, _REPLAY_STORE_MAX_BYTES)
    # Round-scoped and uncapped by design (see `_ROUND_REPLAY`): the driver
    # pops each export after the wave, so nothing accumulates here.
    for key, value in payload.get("round_replay") or ():
        _ROUND_REPLAY[key] = value
    # After the policy replay, so the prune below sees the store as it now is.
    for key, value in payload.get("code") or ():
        _POLICY_CODE[key] = tuple(value)
    for cid in [c for c in _POLICY_CODE if f"policy:{c}" not in _POLICY_STORE]:
        del _POLICY_CODE[cid]

    # MetaWorld's observation -> simulator-snapshot cache is per-process, so
    # without this the parent raises `UnknownStateError` on every trajectory a
    # worker produced: `reference_reward`, `gt_reward_curve`, `render` and
    # therefore rda's `vlm_score` and gt's `preference_bt` all die, while
    # MetaWorld's `task_metric` keeps working -- i.e. fitness looks fine and the
    # video-grounded methods silently become blind comparators.
    blob = payload.get("env_states")
    if blob is not None:
        importer = getattr(ctx.env, "import_states", None)
        if importer is not None:
            with contextlib.suppress(Exception):
                importer(blob)

    ctx.budget.merge_delta(payload.get("budget") or {})
    return result


# ==========================================================================
# seed forking -- the SAME schedule key, applied to a single call's seeds
# ==========================================================================
#
# `train.candidate_parallelism: parallel` widens the candidates of one
# iteration; a single backend call that trains SEVERAL SEEDS is the same shape
# of independence one level down, and serial seeds are expensive in one place
# above all: `final_retrain` (`phases.run_final_retrain`) runs in the PARENT,
# after the search, and its `final_retrain.n_seeds` trainings would otherwise
# run one after another while every worker core sat idle -- on a long gym
# run the retrain can be more than half the wall-clock, spent on serial
# trainings that are independent by construction (LIMEN retrains from scratch
# precisely so that nothing couples them).
#
# The determinism argument is `parallelism_parallel`'s, one level down:
# seeds come from `_seed_base` + a fixed stride (no shared stream), each child
# pins ONE thread (worker count is never a numerics knob), the fold runs in
# seed index order in the parent (budget counters and `BudgetExceeded` fire at
# the same seed they fire at sequentially), and the final rollouts stay in the
# parent, drawn from the SAME shared `roll_rng` in the same order, against
# policies rebuilt from the exact trained parameters each child sent home.
#
# Nested forking is refused, not managed: inside a candidate worker
# (`_IN_TRAIN_WORKER`) seeds stay sequential, because k candidate workers each
# forking s seed workers would oversubscribe the allocation k-fold with no one
# process able to see it.

#: True inside any forked training worker (candidate- or seed-level).
_IN_TRAIN_WORKER = False


def _cfg_declares_batched(cfg: Any) -> bool:
    """Does this config's env declare `batched`, asked from the CONFIG alone?

    The instance-based `_worker_is_batched` is the better question and is
    used wherever a live env exists. Here there is none: `_seed_fork_workers`
    is handed a cfg by three call sites, one of them in another module, and
    widening that signature to carry a ctx is a larger change than the guard.

    So this asks the registered object -- the `batched` attribute carried onto
    the factories by `_declare` (`bird/envs/jax_toy.py`), not the `jax_`
    prefix, which is a naming convention and not a fact about the env.

    UNKNOWN COUNTS AS BATCHED. An env id that will not resolve, or a jax-suite
    adapter that declares nothing, returns True and the seeds run
    sequentially. The two errors are not symmetric: guessing False hangs a
    paid allocation with nothing red, guessing True costs wall-clock on a
    path that is bit-identical either way.
    """
    env_id = cfg.get("problem.env_id", "") or ""
    try:
        from bird.registry import get as _reg_get
        obj = _reg_get("env", env_id)
    except Exception:  # noqa: BLE001 - an unresolvable id is not this guard's business
        return str(env_id).startswith("jax_")
    declared = getattr(obj, "batched", None)
    if declared is None:
        return str(env_id).startswith("jax_")
    return bool(declared)


#: The two field names a seed-schedule record carries, defined ONCE.
#:
#: They appear in two places -- the journal row and every seed row -- and two
#: names for one fact would silently take the journal outside the
#: bit-identity deny-list that covers the row. A constant rather than a
#: convention, because the two sites are thousands of lines apart and the
#: symptom of a mismatch is a SCIENCE-difference verdict.
SEED_SCHEDULE_FIELDS: Tuple[str, ...] = ("seed_workers", "seed_schedule_reason")


def _note_worker_wave(ctx: Any, n_jobs: int, before: Any, after: Any) -> None:
    """Journal the parent's jax-cache reading across one spawned wave.

    A NAMED FUNCTION rather than a `ctx.event` inline in
    `parallelism_parallel`, because that body is covered by
    `test_the_train_stage_touches_only_cfg_env_budget` and the exemption
    there is per-function. Exempting the whole scheduler to journal one row
    would hand away `rng`, `next_id`, `counters` and `tracker` in the
    largest body in the file; exempting this three-line function gives away
    nothing else. Same mechanism as the other two entries: a child's event
    buffer is armed before the work, so `ctx.event` is never a write into
    nothing wherever this runs.

    `grew` is computed here rather than by the reader so that the None case
    is decided ONCE. Either endpoint being None means no persistent cache
    was configured, and the difference is then unknown -- not zero, which is
    what a reader subtracting two Nones-as-0 would record, and it is the
    reading that would look like proof.
    """
    try:
        # `n_jobs`, NOT `n_workers`: this is how many candidates the wave
        # launched, which is bounded by `loop.max_parallel_trainings` but is
        # not equal to it. Named n_workers it would print 3 for a 2-worker
        # cap -- a mislabelled number in the journal, which is worse than an
        # absent one.
        ctx.event("worker_wave", start="spawn", n_jobs=int(n_jobs),
                  parent_jax_cache_before=before,
                  parent_jax_cache_after=after,
                  parent_jax_cache_grew=(None if before is None or after is None
                                         else after - before))
    except Exception as exc:  # noqa: BLE001 - journalling never fails a wave
        log.warning("could not journal the worker wave: %s: %s",
                    type(exc).__name__, exc)


def _note_seed_schedule(ctx: Any, candidate: Any, workers: int, reason: str) -> None:
    """Put the seed schedule where a reader of the ARTEFACT will find it.

    Two places, deliberately, because they answer two different questions.
    The journal row answers "what happened during this run", read in order
    beside everything else. The per-seed field answers "why is THIS seed's
    wall-clock what it is", which is the question someone asks months later
    holding one row of a cost table and no journal.

    Only the downgrade carries a `reason`; the ordinary case records the
    worker count and nothing else, so a reader can still tell 4 workers from
    1 without every row carrying prose.
    """
    try:
        # `seed_schedule_reason`, THE SAME NAME THE SEED ROW USES. Two names
        # for one fact would let the deny-list in `tests/test_parallelism.py`
        # cover the row and miss the journal, and a spawn/sequential journal
        # difference on this key would read as a science difference in a test
        # whose whole job is to detect one.
        _workers_key, _reason_key = SEED_SCHEDULE_FIELDS
        fields = {"cand_id": getattr(candidate, "cand_id", "?"),
                  _workers_key: int(workers)}
        if reason:
            fields[_reason_key] = reason
        ctx.event("seed_schedule", **fields)
    except Exception as exc:  # noqa: BLE001 - recording never fails a training
        log.warning("could not journal the seed schedule: %s: %s",
                    type(exc).__name__, exc)
    if reason:
        log.info("%s", reason)


def _stamp_seed_rows(rows: Any, workers: int, reason: str) -> None:
    """The same two facts onto each seed row, after the seeds have run.

    A SCHEDULE FACT ON A SCIENCE RECORD, which is why both keys are in
    `tests/test_parallelism.py`'s `_VOLATILE`: `seed_workers` legitimately
    DIFFERS between the sequential and parallel runs the bit-identity test
    compares, and a new artefact field is compared by default.
    """
    workers_key, reason_key = SEED_SCHEDULE_FIELDS
    for row in rows or ():
        if isinstance(row, dict):
            row[workers_key] = int(workers)
            if reason:
                row[reason_key] = reason


def _seed_fork_workers(cfg: Any, n_seeds: int) -> Tuple[int, str]:
    """How many processes one backend call may fork over its seeds.

    RETURNS A PAIR, `(workers, downgrade_reason)`, and the pair is the point.
    A downgrade logged with `log.info` behind a bare int would be silent where
    it matters: a reader of the run's ARTEFACT would see a retrain that took
    three times as long and nothing saying why, because the log line is not
    in the run directory. Returning
    the reason forces every call site to carry it somewhere a reader will
    find it -- an unpack that ignores it is visible in review, a log line
    that nobody emitted is not.

    1 -- i.e. the sequential loop -- unless every condition holds: more than
    one seed, `train.candidate_parallelism: parallel` (the SAME key that widens
    candidates: both are schedules, neither may move a number), running in the
    parent process, and an OS with fork. Capped by `loop.max_parallel_trainings`
    via `resolve_workers`, so one knob bounds this process's workers wherever
    they come from.
    """
    if n_seeds <= 1 or _IN_TRAIN_WORKER or not hasattr(os, "fork"):
        return 1, ""
    if cfg.get("train.candidate_parallelism", "sequential") != "parallel":
        return 1, ""
    # THE BATCHED TIER NEVER FORKS SEEDS, and this is the half of the fork
    # hazard the spawned candidate worker does NOT fix. `parallel` is safe
    # for candidates -- but `final_retrain`'s seeds fork IN THE PARENT, where
    # jax is long since initialised and `_IN_TRAIN_WORKER` is False, so
    # without this the hazard would surface as a silent hang inside a post
    # phase at the very end of a run. Same key, same tier, different fork
    # site.
    #
    # INSIDE THIS FUNCTION rather than at its three call sites (two here,
    # one in fasttd3.py) precisely so no caller can omit it -- a guard on a
    # fork is worth nothing if one of three forks does not ask.
    #
    # Sequential seeds are bit-identical by contract, so only wall-clock
    # moves: the behaviour is slower, not broken, and it says so rather than
    # quietly halving the schedule. A spawned seed worker would remove the
    # downgrade.
    if _cfg_declares_batched(cfg):
        return 1, (
            "problem.env_id=%s is a batched (jax) env. Seed workers "
            "fork IN THE PARENT, where jax is already initialised, so they "
            "are run sequentially instead. Results are unchanged -- "
            "sequential is bit-identical to parallel by contract -- but this "
            "retrain took up to %d times as long as the schedule asked for. "
            "A spawned seed worker would remove the downgrade."
            % (cfg.get("problem.env_id", "?"), resolve_workers(cfg, n_seeds)))
    return resolve_workers(cfg, n_seeds), ""


def _seed_child_main(spool: str, idx: int,
                     child_fn: Callable[[int], Dict[str, Any]]) -> None:
    """Runs in the forked seed worker. Writes one pickle and `os._exit`s.

    `os._exit` for `_child_main`'s reason: atexit hooks inherited from the
    parent (wandb's run handle above all) are the parent's to run, not ours.
    """
    code = 0
    try:
        _set_pdeathsig()
        _pin_one_thread()
        global _IN_TRAIN_WORKER
        _IN_TRAIN_WORKER = True
        _write_payload(spool, idx, child_fn(idx))
    except BaseException as exc:  # noqa: BLE001 - the child owns every failure
        try:
            _write_payload(spool, idx, {"fatal": f"{type(exc).__name__}: {exc}",
                                        "traceback": traceback.format_exc()})
        except BaseException:  # pragma: no cover - spool unwritable
            code = 70
    finally:
        os._exit(code)


def _fork_seeds(n_seeds: int, child_fn: Callable[[int], Dict[str, Any]],
                workers: int, label: str) -> List[Dict[str, Any]]:
    """Run `child_fn(i)` for i in [0, n_seeds) in forked children, `workers` at
    a time, and return the payloads strictly in seed order.

    Waits on NAMED pids (FIFO), never `waitpid(-1)`: the parent here may own
    other children (an ffmpeg encode, a wandb service), and reaping a stranger
    steals the exit status its own waiter is blocked on. FIFO is within noise
    of optimal for this workload -- retrain seeds train the same step budget.
    A child that died without writing its payload comes back as a `fatal`
    entry; the caller decides whether that aborts the call.
    """
    spool = tempfile.mkdtemp(prefix="bird-seeds-")
    active: List[int] = []
    log.debug("  seed fork: %d seed(s) over %d worker(s) [%s], spool=%s",
              n_seeds, workers, label, spool)
    try:
        idx_next = 0
        while idx_next < n_seeds or active:
            while idx_next < n_seeds and len(active) < max(1, workers):
                # Flush BEFORE forking, for `parallelism_parallel`'s reason: a
                # block-buffered log file would be flushed once per child.
                _flush_all_streams()
                pid = os.fork()
                if pid == 0:  # -- child; never returns --
                    _seed_child_main(spool, idx_next, child_fn)
                active.append(pid)
                idx_next += 1
            pid = active.pop(0)
            while True:
                try:
                    os.waitpid(pid, 0)
                    break
                except InterruptedError:
                    continue  # EINTR: the wait must land or the child zombies
                except OSError:
                    break     # ECHILD: already reaped; nothing left to wait on
        return [_read_payload(spool, i)
                or {"fatal": f"{label} seed worker {i} died without a payload"}
                for i in range(n_seeds)]
    finally:
        for pid in active:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        for pid in active:
            with contextlib.suppress(OSError):
                os.waitpid(pid, 0)
        shutil.rmtree(spool, ignore_errors=True)


@contextlib.contextmanager
def _single_thread_torch():
    """Pin torch to one intra-op thread for the body, restoring after.

    The parallel seed schedule's rollouts run in the PARENT against rebuilt
    policies; every child trained at one thread (`_pin_one_thread`), so the
    parent's predicts must run at one thread too or the rollouts would be the
    one part of the run whose numerics vary with the machine's thread count.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        # Not imported yet -- and it never is on the FIRST candidate of a
        # fresh process, since every backend imports torch lazily. Pin the
        # way the parallel workers do (`_pin_one_thread`: env before import),
        # so an import that happens inside the body starts at one thread;
        # otherwise candidate 1 would train at the machine's default thread
        # count and be the one candidate on which `sequential` and `parallel`
        # disagree (measured: 16 threads, then 1). Restored after, though
        # an import inside keeps the count it started with, which is the pin.
        keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        saved = {k: os.environ.get(k) for k in keys}
        for k in keys:
            os.environ[k] = "1"
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        return
    before = None
    try:
        before = int(torch.get_num_threads())
        torch.set_num_threads(1)
    except Exception:  # pragma: no cover - torch present but uninitialised
        pass
    try:
        yield
    finally:
        if before is not None:
            with contextlib.suppress(Exception):
                torch.set_num_threads(before)


# --------------------------------------------------------------------------
# ROSKA's fusion-ratio search (`train.fusion.ratio_search`)
#
# ROSKA inherits the previous round's best policy by FUSING it with a freshly
# initialised one rather than copying it (methodology.tex:69):
#
#     theta_f(alpha) = alpha * theta_best_prev + (1 - alpha) * theta_0
#
# The two endpoints are existing methods -- alpha=1 is `warm_start_from_best`
# (RDA), alpha=0 is `from_scratch` (Eureka) -- which is what makes this one
# continuous axis rather than a third incomparable init mode.
#
# Both entries take a `probe` callable, `probe(alpha) -> float`, that trains a
# fused policy for `train.fusion.sc_bo.probe_fraction` of a full training and returns its
# return under the CANDIDATE'S OWN reward. `fixed` never calls it. Injecting
# the probe rather than building one here keeps the search independent of the
# backend and lets a test drive it with a known function.
# --------------------------------------------------------------------------


@register("fusion_ratio_search", "fixed")
def fusion_ratio_fixed(ctx: Any, cfg: Any, probe: Optional[Callable[[float], float]] = None
                       ) -> Tuple[float, Dict[str, Any]]:
    """`train.fusion.alpha`, verbatim, at no training cost.

    ROSKA's own ablation sweeps alpha in {0, 0.5, 1} and reports all three
    losing to the searched ratio (experiment.tex:113), so this is the ablation
    arm rather than the method -- and it is also how `fused_warm_start` spells
    Eureka (0.0) and RDA's warm start (1.0).
    """
    alpha = float(cfg.get("train.fusion.alpha", 1.0) or 0.0)
    alpha = min(1.0, max(0.0, alpha))
    return alpha, {"search": "fixed", "alpha": alpha, "evaluations": 0,
                   "probe_fraction": 0.0}


def _gp_posterior(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray,
                  length: float = 0.15, noise: float = 1e-6
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-mean GP with an RBF kernel, solved by `np.linalg.solve`.

    Written out rather than pulled from sklearn/scipy because this repo's hard
    deps are pyyaml and numpy, and a BO that needs an optional package would
    make ROSKA unrunnable wherever that package is absent -- the failure mode
    would be a method silently degrading to its `fixed` arm.
    """
    ys = np.asarray(ys, dtype=float)
    centre = float(ys.mean()) if ys.size else 0.0
    k = lambda a, b: np.exp(-0.5 * ((a[:, None] - b[None, :]) / length) ** 2)
    kxx = k(xs, xs) + noise * np.eye(xs.size)
    kgx = k(grid, xs)
    alpha_vec = np.linalg.solve(kxx, ys - centre)
    mean = kgx @ alpha_vec + centre
    v = np.linalg.solve(kxx, kgx.T)
    var = np.maximum(1.0 - np.einsum("ij,ji->i", kgx, v), 1e-12)
    return mean, np.sqrt(var)


@register("fusion_ratio_search", "sc_bo")
def fusion_ratio_sc_bo(ctx: Any, cfg: Any, probe: Optional[Callable[[float], float]] = None
                       ) -> Tuple[float, Dict[str, Any]]:
    """ROSKA's Short-Cut Bayesian Optimization over the fusion ratio.

    A GP with an Expected Improvement acquisition, scored by a SHORT probe
    rather than a full training run. The "short cut" is the paper's own
    observation that policy performance diverges early in training, so a
    truncated probe ranks alphas without paying for convergence
    (methodology.tex:88-94). `alpha_initial = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`,
    `J = 12` evaluations including those, `T_BO = 200` epochs
    (experiment.tex:39).

    With no probe, or a non-positive `probe_fraction`/`n_evaluations`, this
    RAISES rather than falling back to `train.fusion.alpha`. `_check_coherence`
    already refuses a config that pins `sc_bo` without a probe budget, so
    reaching here with nothing to search is a caller error; and a fallback
    would deliver a fixed ratio under a config that says the ratio was
    searched, which is indistinguishable from a real search in every artifact.
    """
    init_points = [min(1.0, max(0.0, float(a)))
                   for a in (cfg.get("train.fusion.sc_bo.init_points") or [])]
    budget = int(cfg.get("train.fusion.sc_bo.n_evaluations", 0) or 0)
    probe_fraction = float(cfg.get("train.fusion.sc_bo.probe_fraction", 0.0) or 0.0)

    # No silent fallback. `_check_coherence` already refuses a config that pins
    # `sc_bo` without a probe budget, so reaching here with nothing to search
    # means the CALLER failed to supply a probe -- a programming error, not a
    # configuration one. Degrading to `train.fusion.alpha` would deliver a fixed
    # ratio under a config that says the ratio was optimised, which is the
    # two-config-points-one-method shape this repo treats as its worst bug.
    if probe is None:
        raise RuntimeError(
            "train.fusion.ratio_search=sc_bo reached the search with no probe: the "
            "backend did not supply one. A fixed ratio here would be indistinguishable "
            "from a searched one in every artifact.")
    if probe_fraction <= 0 or budget <= 0 or not init_points:
        raise ConfigError(
            "train.fusion.ratio_search=sc_bo needs probe_fraction > 0, n_evaluations > 0 "
            f"and a non-empty init_points; got {probe_fraction}, {budget}, {init_points!r}")

    # The initial design is evaluated first and counts against J, so J < len(init)
    # truncates the design rather than overrunning the budget the paper states.
    xs: List[float] = []
    ys: List[float] = []
    for a in init_points[:budget]:
        xs.append(a)
        ys.append(float(probe(a)))

    grid = np.linspace(0.0, 1.0, 201)
    while len(xs) < budget:
        ax, ay = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        mean, sd = _gp_posterior(ax, ay, grid)
        best = float(ay.max())
        # Expected Improvement, standard-normal CDF/PDF written out (no scipy).
        z = (mean - best) / sd
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        ei = (mean - best) * cdf + sd * pdf
        # Never re-evaluate a point already measured: the GP's variance there is
        # ~0 so EI is ~0 anyway, but ties at the floor would otherwise let argmax
        # spend the whole remaining budget on one alpha.
        for seen in xs:
            ei[np.abs(grid - seen) < (grid[1] - grid[0])] = -np.inf
        nxt = float(grid[int(np.argmax(ei))])
        xs.append(nxt)
        ys.append(float(probe(nxt)))

    best_i = int(np.argmax(np.asarray(ys, dtype=float)))
    return xs[best_i], {"search": "sc_bo", "alpha": xs[best_i], "evaluations": len(xs),
                        "probe_fraction": probe_fraction, "fell_back": False,
                        "alphas": [round(a, 4) for a in xs],
                        "scores": [float(v) for v in ys]}
