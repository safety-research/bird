"""Every adapter that brings up a MuJoCo GL stack must preload Triton's LLVM first.

WHY THIS FILE EXISTS. `triton/_C/libtriton.so` and the GL stack each embed an LLVM;
whichever loads second gets the other's symbols interposed, and the result is a
SIGSEGV with no traceback at stage [3] train. The fix -- `import triton` for its side
effect on load order, before the simulator imports -- is
`bird/envs/metaworld.py::_preload_llvm_before_mujoco`, and every adapter family that
brings up GL has to call it: a family that does not works until something imports
Triton, and then crashes.

`train.backend: simba_v2` is such a trigger: it `torch.compile`s its update closure on
cuda, which imports Triton, on a tier that otherwise never imports it
(`scripts/setup_humanoid.sh` installs CPU torch on purpose -- "nothing in this tier runs
torch at all"). Measured on an A100 host (mujoco 3.1.6 / torch 2.14.0+cu130 / triton
3.8.0): `python -c "import mujoco; import triton"` exits 139 while `import triton`
alone, `import torch; import triton` and `import jax; import triton` all exit 0, and
the same fault reproduces through the real path (construct `h1hand_package`, then
`torch.compile` on cuda).

So "every family preloads" is a true exhaustiveness claim about the code as written,
protected by nothing that would notice a new family. This test is that protection.

WHAT IT CHECKS, AND WHAT IT DOES NOT. The seam is "the function that decides the GL
backend is about to load it", so the invariant is over the WRITE to
`os.environ["MUJOCO_GL"]`: any function that performs that write must also call
`_preload_llvm_before_mujoco` somewhere in its own AST span. That catches both
spellings in the tree -- `_default_mujoco_gl()` and metaworld's hardcoded
`"osmesa"`.

It does NOT check ORDER within the function, and it does not execute anything: a
family that called the preload after `import mujoco` would pass here and crash at
run time. Order is checked by the one instrument that can see it, which is a run.
Stated rather than implied, because a check narrower than its name is the failure this
file exists to stop.

WHAT THE WRITE MAY LOOK LIKE. A detector matching one shape -- `ast.Assign` with a
`Subscript` target on an `Attribute` named `environ`, the only spelling in the tree --
catches ONE of six synthetic new families that differ ONLY in how the write is
spelled. `AnnAssign` and tuple targets are two of the five it misses, and both are
known failure modes of AST instruments (an `ast.Assign`-only walk missing
`AnnAssign`; an AST scan not descending into tuple targets). A third, `setdefault`, is
a spelling `metaworld.py` carries a comment about REJECTING, so it is live in the
author population this guards.

The detector covers an enumerated set -- `Assign` / `AnnAssign` / `AugAssign`,
targets nested in `Tuple` or `List`, a subscript base that is `os.environ` OR a bare
`environ` from `from os import environ`, and `environ.setdefault` / `environ.update` /
`os.putenv` calls carrying the constant -- and `test_the_detector_fires_on_every_write_
spelling` parametrises over all six and asserts each one is seen. That test is the
point: a detector shown to fire on one input has been shown to fire on one input.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ENVS = pathlib.Path(__file__).resolve().parent.parent / "bird" / "envs"
PRELOAD = "_preload_llvm_before_mujoco"


#: Every way the tree could name the process environment: `os.environ[...]` (an
#: `Attribute`) and `environ[...]` after `from os import environ` (a bare `Name`).
def _is_environ(node: ast.AST) -> bool:
    return ((isinstance(node, ast.Attribute) and node.attr == "environ")
            or (isinstance(node, ast.Name) and node.id == "environ"))


def _is_gl_subscript(node: ast.AST) -> bool:
    """`<environ>["MUJOCO_GL"]`, however `environ` is spelled."""
    return (isinstance(node, ast.Subscript)
            and _is_environ(node.value)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "MUJOCO_GL")


def _flatten_targets(target: ast.AST):
    """Assignment targets, descending into tuple and list unpacking.

    `os.environ["MUJOCO_GL"], other = a, b` is one `Assign` whose single target is a
    `Tuple`; a loop over `node.targets` alone never sees the subscript inside it.
    """
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _flatten_targets(element)
    else:
        yield target


def _names_gl_first(node: ast.Call) -> bool:
    first = node.args[0] if node.args else None
    return isinstance(first, ast.Constant) and first.value == "MUJOCO_GL"


def _writes_gl_by_call(node: ast.AST) -> bool:
    """`environ.setdefault(...)`, `environ.update({...})`, `putenv(...)`.

    `putenv` is accepted BARE as well as as `os.putenv`, for the same reason
    `_is_environ` accepts a bare `environ`: `from os import putenv` is the exact
    parallel of `from os import environ`, and covering one name and not the
    other would be an asymmetry inside this file's own decision rather than a
    limit of it.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "putenv":
        return _names_gl_first(node)
    if isinstance(func, ast.Attribute) and func.attr in ("setdefault", "putenv"):
        # setdefault is on environ; putenv is on os -- both take the name first.
        if func.attr == "putenv" or _is_environ(func.value):
            return _names_gl_first(node)
    if (isinstance(func, ast.Attribute) and func.attr == "update"
            and _is_environ(func.value)):
        for arg in node.args:
            if isinstance(arg, ast.Dict):
                if any(isinstance(k, ast.Constant) and k.value == "MUJOCO_GL"
                       for k in arg.keys):
                    return True
        for kw in node.keywords:
            if kw.arg == "MUJOCO_GL":
                return True
    return False


def _sets_mujoco_gl(node: ast.AST) -> bool:
    """True if this node's span WRITES `MUJOCO_GL` into the environment, any spelling.

    The enumerated set is in the module docstring, and
    `test_the_detector_fires_on_every_write_spelling` holds this function to it.
    """
    for n in ast.walk(node):
        if isinstance(n, ast.AugAssign) and _is_environ(n.target):
            # `os.environ |= {"MUJOCO_GL": ...}` writes THROUGH the mapping, so the
            # target is `os.environ` itself and no subscript appears anywhere. The
            # AugAssign branch below looks like it covers this and cannot: it asks
            # for a subscript target.
            if isinstance(n.value, ast.Dict) and any(
                    isinstance(k, ast.Constant) and k.value == "MUJOCO_GL"
                    for k in n.value.keys):
                return True
            continue
        if isinstance(n, ast.Assign):
            targets = [t for target in n.targets for t in _flatten_targets(target)]
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            targets = list(_flatten_targets(n.target))
        else:
            if _writes_gl_by_call(n):
                return True
            continue
        if any(_is_gl_subscript(t) for t in targets):
            return True
    return False


#: Modules whose import brings up the GL stack. Importing any of these is the
#: event the preload must precede -- NOT the `MUJOCO_GL` write, which sets an
#: environment variable and loads nothing. Measured: at all 4 sites in the tree
#: the preload sits AFTER the write and BEFORE the import, so a rule written
#: against the write would fail every correct site (4 of 4).
_SIM_IMPORTS = ("mujoco", "gymnasium", "gym", "humanoid_bench")


def _is_preload_call(stmt: ast.AST) -> bool:
    """A bare `_preload_llvm_before_mujoco()` as a statement in its own right.

    Bare only. Every real call site spells it bare (`humanoid_hand.py` imports the
    symbol), and also accepting `mod._preload_llvm_before_mujoco()` by attribute
    name would let `some_other_object._preload_llvm_before_mujoco()` satisfy the
    guard. An uncovered widening that weakens the check is worse than the false
    failure it avoids.
    """
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name)
            and stmt.value.func.id == PRELOAD)


def _imports_a_simulator(stmt: ast.AST) -> bool:
    if isinstance(stmt, ast.Import):
        return any((a.name or "").split(".")[0] in _SIM_IMPORTS for a in stmt.names)
    if isinstance(stmt, ast.ImportFrom):
        return (stmt.module or "").split(".")[0] in _SIM_IMPORTS
    return False


def _straight_line(body):
    """The statements that certainly run, in order, flattened.

    Descends into `with` bodies and `try` bodies -- both are straight-line -- and
    deliberately NOT into `if`/`for`/`while` bodies, `except`/`else`/`finally`
    handlers, or nested `def`s. A preload in any of those is not guaranteed to run
    before the import, and a presence-only check accepts all of them: measured, a
    call inside `if False:`, inside a never-called nested `def`, and inside an
    `except` branch each satisfy it.
    """
    for stmt in body:
        if isinstance(stmt, ast.With):
            yield from _straight_line(stmt.body)
        elif isinstance(stmt, ast.Try):
            yield from _straight_line(stmt.body)
        else:
            yield stmt


def _gl_bringup_sites():
    """(module, function) for every function that writes os.environ['MUJOCO_GL']."""
    out = []
    for path in sorted(ENVS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _sets_mujoco_gl(node):
                out.append((path, node))
    return out


_SITES = _gl_bringup_sites()


#: One synthetic new family per way of writing the environment, each a constructor
#: that brings up the GL stack and preloads NOTHING. The detector must see every one.
#: The first entry is the only spelling in `bird/envs/`; a detector matching that one
#: shape lets the other five walk past unflagged.
_WRITE_SPELLINGS = {
    "os_environ_subscript": 'os.environ["MUJOCO_GL"] = chosen',
    "bare_environ_subscript": 'environ["MUJOCO_GL"] = chosen',
    "setdefault": 'os.environ.setdefault("MUJOCO_GL", chosen)',
    "annassign": 'os.environ["MUJOCO_GL"]: str = chosen',
    "tuple_target": 'os.environ["MUJOCO_GL"], _other = chosen, 1',
    "update_dict": 'os.environ.update({"MUJOCO_GL": chosen})',
    "ior_through_mapping": 'os.environ |= {"MUJOCO_GL": chosen}',
    "bare_putenv": 'putenv("MUJOCO_GL", chosen)',
}

#: Spellings this detector KNOWINGLY does not see, kept as a list rather than as
#: silence. Each is a real way to write the environment and none appears in the
#: tree; covering them costs more than it buys, and a reader who adds one should
#: find it named here rather than discover the guard was quiet:
#:
#:   os.environ.update([("MUJOCO_GL", v)])   an iterable of pairs, not a Dict
#:   os.environ.__setitem__("MUJOCO_GL", v)  the dunder spelled out
#:   K = "MUJOCO_GL"; os.environ[K] = v      the key bound to a name first
#:
#: The third is the one that would need real work (constant folding across the
#: function), and it is also the least likely to be written by someone bringing
#: up a simulator.
_KNOWN_UNCOVERED_SPELLINGS = 3

_SYNTHETIC_FAMILY = """
import os
from os import environ


class NewFamily:
    def __init__(self):
        chosen = os.environ.get("MUJOCO_GL") or "egl"
        {write}
        import mujoco
"""


@pytest.mark.parametrize("spelling", sorted(_WRITE_SPELLINGS), ids=sorted(_WRITE_SPELLINGS))
def test_the_detector_fires_on_every_write_spelling(spelling):
    """The detector is held to the enumerated set, not to the tree's current habits.

    THIS IS THE TEST THAT EARNS THE FILE. Every assertion below is of the form "this
    function does not write MUJOCO_GL", and a detector that cannot SEE a write makes
    all of them vacuous for exactly the population the file exists to protect: code
    nobody has written yet. A detector matching one shape lets five of these six
    straight through, two of them (`annassign`, `tuple_target`) in the precise ways
    AST instruments are known to fail.

    Adding a seventh spelling means adding it here, in the same change that teaches
    `_sets_mujoco_gl` to see it.
    """
    source = _SYNTHETIC_FAMILY.format(write=_WRITE_SPELLINGS[spelling])
    tree = ast.parse(source)
    ctors = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    assert len(ctors) == 1, "the fixture should have exactly one constructor"
    assert _sets_mujoco_gl(ctors[0]), (
        f"_sets_mujoco_gl does not see `{_WRITE_SPELLINGS[spelling]}` as a write to "
        "MUJOCO_GL, so a family spelled that way is never enumerated and never "
        "checked -- it would bring up the GL stack with no LLVM preload and this "
        "file would stay green. Teach the detector this shape."
    )
    assert not any(_is_preload_call(st) for st in _straight_line(ctors[0].body)), (
        "the fixture must preload NOTHING, or it cannot show the detector fires"
    )


#: Where the preload can sit and be useless. Each of these satisfies a
#: presence-only check.
_DEAD_POSITIONS = {
    "after_the_import": "        import mujoco\n        _preload_llvm_before_mujoco()",
    "inside_if_false": "        if False:\n            _preload_llvm_before_mujoco()\n        import mujoco",
    "in_a_nested_def": ("        def _later():\n            _preload_llvm_before_mujoco()\n"
                        "        import mujoco"),
    "in_an_except_branch": ("        try:\n            pass\n        except Exception:\n"
                            "            _preload_llvm_before_mujoco()\n        import mujoco"),
    "on_another_object": "        other._preload_llvm_before_mujoco()\n        import mujoco",
}

_POSITION_FAMILY = """
import os


class NewFamily:
    def __init__(self):
        os.environ["MUJOCO_GL"] = "egl"
{body}
"""


@pytest.mark.parametrize("position", sorted(_DEAD_POSITIONS), ids=sorted(_DEAD_POSITIONS))
def test_a_preload_that_cannot_run_first_is_not_a_preload(position):
    """ORDER is the defect, so a call in the wrong place must fail like an absent one.

    All five of these pass a presence-only check, and all five crash at run time:
    the preload's ONLY effect is on load order, so after `import mujoco` it does
    nothing, and in a branch that does not execute it does nothing.
    `on_another_object` is here because accepting any attribute call ending in the
    right name would pass it.
    """
    source = _POSITION_FAMILY.format(body=_DEAD_POSITIONS[position])
    tree = ast.parse(source)
    ctor = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    stmts = list(_straight_line(ctor.body))
    preload_at = next((i for i, st in enumerate(stmts) if _is_preload_call(st)), None)
    import_at = next((i for i, st in enumerate(stmts) if _imports_a_simulator(st)), None)
    assert import_at is not None, "the fixture must import a simulator"
    assert preload_at is None or preload_at > import_at, (
        f"{position}: the checks accept a preload that cannot run before the "
        f"simulator import, which is the whole defect this file exists for"
    )


def test_the_preload_is_seen_when_it_is_in_the_right_place():
    """The positive control, so the position rule is not simply "never satisfied"."""
    source = _POSITION_FAMILY.format(
        body="        _preload_llvm_before_mujoco()\n        import mujoco")
    ctor = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    stmts = list(_straight_line(ctor.body))
    preload_at = next((i for i, st in enumerate(stmts) if _is_preload_call(st)), None)
    import_at = next((i for i, st in enumerate(stmts) if _imports_a_simulator(st)), None)
    assert preload_at is not None and import_at is not None
    assert preload_at < import_at


def test_the_detector_does_not_fire_on_a_read():
    """The other direction, so `_sets_mujoco_gl` is not simply "mentions MUJOCO_GL".

    `cameras.py` reads `os.environ.get("MUJOCO_GL")` to pick a context and writes
    nothing; if that counted, the enumeration would grow a site that has no business
    preloading anything and the fix would be to add a pointless call.
    """
    tree = ast.parse(
        "import os\n"
        "def pick():\n"
        "    return (os.environ.get('MUJOCO_GL') or '').lower()\n"
    )
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    assert not _sets_mujoco_gl(fn)


def test_the_detector_finds_the_sites_it_is_supposed_to_protect():
    """A detector reporting zero is indistinguishable from a broken one.

    Anchored on the two families that motivate the file: `metaworld` (hardcoded
    "osmesa") and `humanoid_hand` (the family the missing-preload crash was
    measured on). If the enumeration ever stops seeing these, every assertion below
    is vacuous and this fails first.
    """
    found = {p.name for p, _ in _SITES}
    assert "metaworld.py" in found, found
    assert "humanoid_hand.py" in found, found
    # 4 measured on the release tree: assistax, gym_mujoco, humanoid_hand and metaworld.
    assert len(_SITES) >= 4, f"only {len(_SITES)} GL bring-up sites found: {sorted(found)}"


@pytest.mark.parametrize(
    "path,func",
    _SITES,
    ids=[f"{p.name}::{f.name}:{f.lineno}" for p, f in _SITES],
)
def test_every_gl_bringup_preloads_llvm_before_importing_a_simulator(path, func):
    """PRESENCE IS NOT THE PROPERTY -- ORDER IS, and this asserts order.

    The defect is a load order: the GL stack and `triton/_C/libtriton.so` each
    embed an LLVM and whichever loads SECOND gets the other's symbols interposed.
    A test asserting only that the preload is called somewhere in the function
    contradicts its own failure message ("add the call immediately before the
    simulator imports"): measured, such a check passes a preload placed AFTER
    `import mujoco`, inside `if False:`, inside a nested `def` that is never
    called, and inside an `except` branch -- every one of which crashes at run
    time.

    So: the call must be a statement in the function's straight-line body, and it
    must come before the first statement there that imports a simulator. Where a
    span imports no simulator (a constructor that builds nothing yet) the ordering
    is vacuous and presence in the straight line is what is left to assert.
    """
    stmts = list(_straight_line(func.body))
    preload_at = next((i for i, st in enumerate(stmts) if _is_preload_call(st)), None)
    assert preload_at is not None, (
        f"{path.name}::{func.name} (line {func.lineno}) writes os.environ['MUJOCO_GL'] "
        f"without calling {PRELOAD}() in its straight-line body. The GL stack and "
        f"triton/_C/libtriton.so each embed an LLVM; whichever loads second gets the "
        f"other's symbols interposed and the process dies with SIGSEGV and no "
        f"traceback at stage [3] train. A call inside an `if`, a loop, an `except` "
        f"or a nested def does not count -- it is not guaranteed to run first. "
        f"See bird/envs/metaworld.py::{PRELOAD} for the long form."
    )
    import_at = next((i for i, st in enumerate(stmts) if _imports_a_simulator(st)), None)
    assert import_at is None or preload_at < import_at, (
        f"{path.name}::{func.name} (line {func.lineno}) calls {PRELOAD}() at "
        f"statement {preload_at} but imports a simulator at statement {import_at}, "
        f"so the GL stack loads FIRST and takes the LLVM symbols. The preload's "
        f"only effect is on load order; after the import it does nothing at all "
        f"and the process dies at run time while this file stays green."
    )
