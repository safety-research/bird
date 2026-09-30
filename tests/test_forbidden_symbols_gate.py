"""`verify.forbidden_symbols` rejects a program that REACHES for a scored
quantity -- not one that happens to bind the same word itself.

The common false positive is the candidate's own local `success = (...)` -- a
sparse bonus computed from the state arrays it was handed. The reward signature
receives no env, so nothing in such a program can read `env.success`; yet each
rejection forfeits a training slot (`generate.parse.max_retries: 0`, verify
never resamples). The collector
therefore distinguishes a name the program binds from a name it reaches for.
Every genuine leak shape below must still be caught.
"""
from __future__ import annotations

import ast
import types

import pytest

from bird.components.verification import _referenced_symbols, static_forbidden_symbols
from bird.state import Candidate

FORBIDDEN = ["task_metric", "success", "reference_reward", "_model", "_data", "_forward"]


def _ctx(forbidden=FORBIDDEN):
    return types.SimpleNamespace(cfg={"verify.forbidden_symbols": list(forbidden)})


def _cand(code: str) -> Candidate:
    return Candidate(cand_id="c", iteration=0, reward_code=code)


def _gate(code: str) -> str:
    return static_forbidden_symbols(_ctx(), _cand(code))


# -- the program's own bindings are not reaches --------------------------------

LOCAL_SUCCESS = '''
def compute_reward(state, action=None, next_state=None):
    z, up = float(state[2]), float(state[3])
    # sparse success bonus, computed from the state it was handed
    success = (abs(z - 0.5) < 0.05 and up > 0.9)
    r_success = 1.0 if success else 0.0
    return r_success, {"success": r_success}
'''


def test_a_local_variable_named_success_is_the_programs_own():
    assert _gate(LOCAL_SUCCESS) == ""


def test_a_locally_computed_sparse_bonus_passes():
    # a weighted sparse bonus, bound locally and reported as a component
    code = '''
def compute_reward(state, action=None, next_state=None):
    h, up_z = float(state[2]), float(state[3])
    w_success = 2.0
    success = 1.0 if (h > 0.5 and up_z > 0.9) else 0.0
    total = w_success * success
    return total, {"success_bonus": w_success * success}
'''
    assert _gate(code) == ""


@pytest.mark.parametrize("code", [
    "def compute_reward(state, success=0.0, next_state=None):\n    return success, {}\n",
    "def compute_reward(state, action=None, next_state=None):\n"
    "    for success in (0.0, 1.0):\n        pass\n    return success, {}\n",
    "def compute_reward(state, action=None, next_state=None):\n"
    "    vals = [success for success in (0.0, 1.0)]\n    return vals[0], {}\n",
    "def compute_reward(state, action=None, next_state=None):\n"
    "    def success(x):\n        return x\n    return success(1.0), {}\n",
    "def compute_reward(state, action=None, next_state=None):\n"
    "    try:\n        pass\n    except Exception as success:\n        pass\n    return 0.0, {}\n",
])
def test_every_binding_form_is_local(code):
    assert _gate(code) == ""


def test_a_string_key_is_not_a_symbol():
    code = 'def compute_reward(s, a=None, n=None):\n    return 1.0, {"success": 1.0, "_data": 0.0}\n'
    assert _gate(code) == ""


# -- every real leak shape is still caught --------------------------------------

@pytest.mark.parametrize("code, hit", [
    ("def compute_reward(state, action=None, next_state=None):\n"
     "    return float(env.success()), {}\n", "success"),
    ("def compute_reward(state, action=None, next_state=None):\n"
     "    return float(success(state)), {}\n", "success"),              # free name
    ("def compute_reward(state, action=None, next_state=None):\n"
     "    return getattr(env, 'success')(), {}\n", "success"),
    ("def compute_reward(state, action=None, next_state=None):\n"
     "    return state._data.qpos[2], {}\n", "_data"),
    ("def compute_reward(state, action=None, next_state=None):\n"
     "    return task_metric(state), {}\n", "task_metric"),
    ("def compute_reward(state, action=None, next_state=None):\n"
     "    global success\n    success = 1.0\n    return success, {}\n", "success"),
    # an import is a reach, not a local binding that excuses the read (a
    # per-scope collector can silently stop reporting these)
    ("from helpers import success\n"
     "def compute_reward(state, action=None, next_state=None):\n"
     "    return float(success(state)), {}\n", "success"),
    ("import numpy as task_metric\n"
     "def compute_reward(state, action=None, next_state=None):\n"
     "    return task_metric(state), {}\n", "task_metric"),
])
def test_reaching_for_a_forbidden_symbol_is_still_rejected(code, hit):
    out = _gate(code)
    assert out.startswith("forbidden_symbols:"), out
    assert hit in out


@pytest.mark.parametrize("code", [
    # a comprehension variable does not excuse a later free read in the enclosing scope
    "def compute_reward(state, action=None, next_state=None):\n"
    "    vals = [success for success in (0.0, 1.0)]\n    return success, {}\n",
    # a nested function's local does not excuse the outer function's free read
    "def compute_reward(state, action=None, next_state=None):\n"
    "    def inner():\n        success = 1.0\n        return success\n"
    "    return success, {}\n",
    # a class-body binding is not visible inside its methods
    "class R:\n    success = 1.0\n    def compute_reward(self, s, a=None, n=None):\n"
    "        return success, {}\n",
])
def test_a_binding_in_another_scope_does_not_mask_a_free_reach(code):
    """One flat `bound` set for the whole tree would let
    `[success for success in xs]` excuse `return success` after it -- a false
    NEGATIVE on an anti-leak gate. Binding is resolved per scope."""
    out = _gate(code)
    assert out.startswith("forbidden_symbols:") and "success" in out, out


def test_nonlocal_binds_to_the_enclosing_function():   # the global case is in the rejected list above
    ok = ("def compute_reward(s, a=None, n=None):\n    success = 0.0\n"
          "    def bump():\n        nonlocal success\n        success = 1.0\n"
          "    bump()\n    return success, {}\n")
    assert _gate(ok) == ""


def test_an_attribute_on_a_local_still_reports_the_attribute():
    """`success` bound locally does not launder `success.foo`: the leaf attr
    and the dotted chain are still collected."""
    code = ("def compute_reward(s, a=None, n=None):\n"
            "    success = s\n    return float(success._model), {}\n")
    seen = _referenced_symbols(ast.parse(code))
    assert "_model" in seen and "success._model" in seen
    assert "success" not in seen, "the local binding itself is not a reach"


def test_dotted_entries_match_the_access_not_the_local():
    ctx = _ctx(["env.success"])
    ok = "def compute_reward(s, a=None, n=None):\n    success = 1.0\n    return success, {}\n"
    bad = "def compute_reward(s, a=None, n=None):\n    return env.success(), {}\n"
    assert static_forbidden_symbols(ctx, _cand(ok)) == ""
    assert "env.success" in static_forbidden_symbols(ctx, _cand(bad))


def test_an_empty_gate_is_a_no_op():
    assert static_forbidden_symbols(_ctx([]), _cand("def compute_reward(s):\n    return success, {}\n")) == ""
