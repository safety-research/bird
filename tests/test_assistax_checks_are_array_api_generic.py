"""The five Assistax success checks must stay array-API-generic AND keep their
thresholds.

WHY THIS IS A TEST AND NOT A ONE-OFF SCRIPT. A check written
`bool(D.dist < 0.04 and D.tip_force_mag < 5.0)` cannot trace: Python's `and`
short-circuits and `bool()` demands a truth value, so under `jax.jit` it raises
TracerBoolConversionError -- the MJX tier could then not emit
`info["success"]` at all, and `base.py` makes that key mandatory. The checks are
written on `D.xp` so ONE definition serves both tiers and the thresholds exist
once. Nothing else in the suite would notice them reverting:
the CPU callers wrap every result in `bool()`, so a `bool(... and ...)` form
passes every CPU test and breaks only the jax tier, at trace time, in a venv
CI does not run.

THE BOUNDARY CASES ARE THE POINT. A 5000-draw random equivalence screen
against the `bool(... and ...)` reference passes -- and it also passes a
deliberately mutated `>=` -> `>` on teethbrushing, because on a CONTINUOUS
field the probability of drawing exactly the threshold is zero. Only the
bedbathing mutant is caught, and only because `n_wiped` is integer-valued so
`n_wiped / 52` lands exactly on 0.5 at 26. Every threshold below is therefore
pinned as an EXACT value on both sides of the comparison; with them, all five
intended mutants fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from bird.envs import assistax as A


class _D:
    """A `_Derived` stand-in carrying only what the checks read."""

    def __init__(self, xp, **kw):
        self.xp = xp
        for k, v in kw.items():
            setattr(self, k, v)


class _Env:
    success_threshold = 0.5


#: (task, fields, expected). Each row sits EXACTLY on a threshold or one ulp
#: off it, so a `<` / `<=` slip or a moved constant fails here by name.
CASES = [
    ("feeding", dict(dist=0.04, tip_force_mag=1.0), False),          # dist < 0.04 is strict
    ("feeding", dict(dist=0.039, tip_force_mag=5.0), False),         # tip_force_mag < 5.0 strict
    ("feeding", dict(dist=0.039, tip_force_mag=4.999), True),
    ("armmanipulation", dict(dist_forearm_waist=0.08), False),
    ("armmanipulation", dict(dist_forearm_waist=0.0799), True),
    ("scratchitch", dict(dist=0.05, tool_force_mag=1.0, tool_speed=1.0), False),
    ("scratchitch", dict(dist=0.049, tool_force_mag=0.5, tool_speed=0.02), True),   # both >=
    ("scratchitch", dict(dist=0.049, tool_force_mag=0.499, tool_speed=0.02), False),
    ("scratchitch", dict(dist=0.049, tool_force_mag=0.5, tool_speed=0.0199), False),
    ("teethbrushing", dict(dist=0.03, tip_force_mag=1.0, tangential_speed=1.0), False),
    ("teethbrushing", dict(dist=0.029, tip_force_mag=0.1, tangential_speed=0.01), True),
    ("teethbrushing", dict(dist=0.029, tip_force_mag=0.0999, tangential_speed=0.01), False),
    ("teethbrushing", dict(dist=0.029, tip_force_mag=0.1, tangential_speed=0.0099), False),
    # n_wiped is integer-valued, so 26/52 hits the 0.5 threshold EXACTLY
    ("bedbathing", dict(n_wiped=25), False),
    ("bedbathing", dict(n_wiped=26), True),
    ("bedbathing", dict(n_wiped=27), True),
]


@pytest.mark.parametrize("task,fields,expected", CASES)
def test_the_threshold_sits_exactly_where_the_row_says(task, fields, expected):
    chk = A._TASKS[task]["check"]
    assert bool(chk(_Env(), _D(np, **fields))) is expected


@pytest.mark.parametrize("task", sorted(A._TASKS))
def test_no_check_uses_bool_or_python_and(task):
    """The source-level guard, because the behavioural one cannot run in CI.

    `tests` installs no jax, so `test_every_check_traces_under_jit` skips in
    every CI job -- a revert to `bool(... and ...)` would pass unnoticed.
    This reads the source instead, which needs nothing installed.

    IT INSPECTS AST NODES, NOT TEXT. A grep of the source for `bool(` and
    ` and ` fails on bedbathing -- whose DOCSTRING contains the word "and". A
    substring check over source is a check of the comments
    as much as the code; the parse tree is the only reading that is about the
    code alone.
    """
    import ast
    import inspect
    import textwrap

    fn = A._TASKS[task]["check"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            raise AssertionError(
                f"{task}: Python `and` short-circuits and does not trace; "
                f"use D.xp.logical_and (line {node.lineno} of the function)")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "bool"):
            raise AssertionError(
                f"{task}: bool() demands a truth value and does not trace "
                f"under jax.jit (line {node.lineno} of the function)")


@pytest.mark.jax
@pytest.mark.parametrize("task", sorted(A._TASKS))
def test_every_check_traces_under_jit(task):
    # `importorskip("jax")` and then a PLAIN import of the submodule.
    # `importorskip("jax.numpy")` is a gate that can never open:
    # `tests/test_no_silent_skip.py` refuses it because `uv.lock` resolves no
    # package of that name and none ever will -- it is a submodule, not a
    # distribution -- so the case would skip in every configuration and cover
    # nothing, which is worse than absent because it prints as a dot.
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    chk = A._TASKS[task]["check"]
    fields = ("dist", "tip_force_mag", "tool_force_mag", "tool_speed",
              "tangential_speed", "dist_forearm_waist", "n_wiped")

    # EVERY attribute is a slice of the traced argument. Concrete floats would
    # trace trivially and exonerate functions that in fact cannot trace.
    def f(v):
        return jnp.asarray(chk(_Env(), _D(jnp, **{n: v[i] for i, n in enumerate(fields)})))

    jax.jit(f)(jnp.arange(float(len(fields))))
