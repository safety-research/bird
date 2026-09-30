"""The central invariant, enforced automatically.

The rule: no `if config.name == "eureka"` anywhere -- that is the failure mode
this whole exercise exists to prevent.

A rule nobody checks is a rule that decays, so this test walks the source's AST
for comparisons against a method name. If you are here because this test
failed: the fix is never to add the branch and silence the test. It is to add a
config key and a registry entry, because a method that needs bespoke Python is
a method the schema cannot express -- which is the one result this repo is set
up to detect.

Two things about the guard's REACH, both of which must be as wide as the
invariant:

  * `METHOD_NAMES` is every config stem under `configs/`, recursively --
    `configs/methods/`, the ERA recipes at the top level, `configs/hillclimb/`
    and `configs/examples/` (all enumerated by `--list-configs`). Limiting it
    to the published methods would make a branch on `"era_u"` or
    `"v3_thought"` invisible.
  * `_Finder` looks at BOTH sides of a comparison, at tuple/set/list
    comparators (`name in {"eureka", "rda"}`), at `str.startswith` /
    `str.endswith` with a method-name constant, and at `match` / `case`
    patterns. Reading `node.comparators` alone would let `"eureka" in
    name`, `name.startswith("eureka")` and `case "eureka":` all pass.
    `test_the_finder_catches_every_form_a_branch_can_take` is the positive
    control that keeps that reach measured rather than assumed.

The second test is the same smell from the other end: a config-name READ
(`cfg["name"]`, `cfg.get("name")`) used to decide anything -- compared,
membership-tested, prefix-matched or matched, directly or through the
identifier it was bound to. Labelling with it (a wandb group, a run-dir name)
is fine and is not flagged.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import pytest

from conftest import REPO  # noqa: F401  (path setup)
from bird.config import CONFIG_ROOT


def _method_names() -> Set[str]:
    """Every config stem under `configs/`, recursively, minus the two kinds of
    file that are not methods: `_default.yaml` (the key set) and the
    `_profiles/` overlays (`tester`, `dev`, `full`, `humanoid` -- execution
    tiers, and ordinary words besides)."""
    names: Set[str] = set()
    for p in CONFIG_ROOT.rglob("*.yaml"):
        rel = p.relative_to(CONFIG_ROOT)
        if p.name.startswith("_") or "_profiles" in rel.parts:
            continue
        names.add(p.stem)
    return names


METHOD_NAMES = _method_names()
#: bare words that also name methods and could appear innocently
ALSO_INNOCENT = {"zeroshot"}


def _sources(repo_root: Path):
    yield repo_root / "bird.py"
    yield from sorted((repo_root / "bird").rglob("*.py"))
    yield from sorted((repo_root / "scripts").rglob("*.py"))


class _Finder(ast.NodeVisitor):
    """Every place a method-name string constant takes part in a decision."""

    def __init__(self, names: Iterable[str]):
        self.names = set(names) - ALSO_INNOCENT
        self.hits: List[Tuple[int, str]] = []

    def _constants(self, node: ast.AST) -> List[str]:
        """Method-name constants in `node`: a bare string, or the elements of a
        tuple / set / list of them (`name in {"eureka", "rda"}`)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value] if node.value in self.names else []
        if isinstance(node, (ast.Tuple, ast.Set, ast.List)):
            out: List[str] = []
            for e in node.elts:
                out += self._constants(e)
            return out
        return []

    def visit_Compare(self, node: ast.Compare) -> None:
        for side in [node.left, *node.comparators]:
            for v in self._constants(side):
                self.hits.append((node.lineno, v))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("startswith", "endswith"):
            for a in node.args:
                for v in self._constants(a):
                    self.hits.append((node.lineno, v))
        self.generic_visit(node)

    def _pattern(self, pat: ast.AST) -> List[str]:
        if isinstance(pat, ast.MatchValue):
            return self._constants(pat.value)
        if isinstance(pat, ast.MatchOr):
            out: List[str] = []
            for p in pat.patterns:
                out += self._pattern(p)
            return out
        if isinstance(pat, ast.MatchAs) and pat.pattern is not None:
            return self._pattern(pat.pattern)
        return []

    def visit_match_case(self, node: ast.match_case) -> None:
        for v in self._pattern(node.pattern):
            self.hits.append((node.pattern.lineno, v))
        self.generic_visit(node)


def test_the_method_names_cover_hillclimb_and_example_configs():
    """The reach of the first test, measured: a stem in any of the places a
    config can live is a name the source must not branch on."""
    assert {"eureka", "rda", "card"} <= METHOD_NAMES
    assert {"dreureka", "l2r", "text2reward_zeroshot", "eureka_no_evolution",
            "era_u", "era_s"} <= METHOD_NAMES
    for sub in ("methods", "hillclimb", "examples"):
        assert any(p.parent.name == sub and p.stem in METHOD_NAMES
                   for p in CONFIG_ROOT.rglob("*.yaml")), f"configs/{sub}/ stems must be guarded"
    assert "_default" not in METHOD_NAMES
    assert not ({"tester", "dev", "full", "humanoid"} & METHOD_NAMES), (
        "execution profiles are not methods (and are ordinary words)")


def test_no_source_file_branches_on_a_method_name(repo_root):
    offenders = []
    for path in _sources(repo_root):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError as exc:  # pragma: no cover
            pytest.fail(f"{path} does not parse: {exc}")
        f = _Finder(METHOD_NAMES)
        f.visit(tree)
        offenders += [f"{path.relative_to(repo_root)}:{ln} compares against {v!r}"
                      for ln, v in f.hits]
    assert not offenders, (
        "method-name branching found -- add a config key and a registry entry instead:\n  "
        + "\n  ".join(offenders))


_EVASIONS = {
    "rhs equality": 'if name == "eureka": pass',
    "lhs equality": 'if "eureka" == name: pass',
    "membership in the name": 'if "eureka" in name: pass',
    "name in a set": 'if name in {"eureka", "rda"}: pass',
    "name in a tuple": 'if name in ("dreureka",): pass',
    "startswith": 'if name.startswith("eureka"): pass',
    "endswith": 'if name.endswith("l2r"): pass',
    "startswith a tuple": 'if name.startswith(("rda", "card")): pass',
    "match/case": 'match name:\n    case "eureka":\n        pass\n    case _:\n        pass',
    "match/case or-pattern": 'match name:\n    case "rda" | "card":\n        pass',
    "corpus stem": 'if name == "text2reward_zeroshot": pass',
    "hillclimb stem": 'if name == "v3_thought": pass',
    "example stem": 'if name == "jax_reward": pass',
}


@pytest.mark.parametrize("form", sorted(_EVASIONS))
def test_the_finder_catches_every_form_a_branch_can_take(form):
    """Positive control. Each snippet is a way to branch on a method name;
    the finder must see all of them or its green means nothing."""
    f = _Finder(METHOD_NAMES)
    f.visit(ast.parse(_EVASIONS[form]))
    assert f.hits, form


def test_the_finder_leaves_innocent_code_alone():
    f = _Finder(METHOD_NAMES)
    f.visit(ast.parse('label = "eureka"\nlog.info("%s", "eureka")\n'
                      'if mode == "zeroshot": pass\nx = {"eureka": 1}["eureka"]'))
    assert f.hits == []


# --------------------------------------------------------------------------
# Reading the config name to decide
# --------------------------------------------------------------------------

_CFG_WORDS = ("cfg", "config", "conf")


def _is_cfg(node: ast.AST) -> bool:
    """A config object by name: `cfg`, `config`, `base_cfg`, `self.cfg`, `ctx.cfg`."""
    ident = node.id if isinstance(node, ast.Name) else \
        node.attr if isinstance(node, ast.Attribute) else ""
    return bool(ident) and (ident in _CFG_WORDS or ident.endswith(("_cfg", "_config")))


def _reads_name(node: ast.AST) -> bool:
    """`cfg["name"]`, `cfg.get("name")`, `cfg.get("name", default)`."""
    if isinstance(node, ast.Subscript) and _is_cfg(node.value):
        return isinstance(node.slice, ast.Constant) and node.slice.value == "name"
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and _is_cfg(node.func.value)):
        return bool(node.args) and isinstance(node.args[0], ast.Constant) \
            and node.args[0].value == "name"
    return False


class _NameReader(ast.NodeVisitor):
    """A config-name read used as a control-flow signal: compared, membership-
    tested, prefix/suffix-matched or matched -- directly, or through an
    identifier it was bound to in the same scope."""

    def __init__(self):
        self.bound: Set[str] = set()
        self.hits: List[int] = []

    def _scoped(self, node: ast.AST) -> None:
        outer = set(self.bound)
        self.generic_visit(node)
        self.bound = outer

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def _bind(self, target: ast.AST, value: ast.AST) -> None:
        if value is not None and _reads_name(value) and isinstance(target, ast.Name):
            self.bound.add(target.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._bind(t, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._bind(node.target, node.value)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._bind(node.target, node.value)
        self.generic_visit(node)

    def _is_read(self, node: ast.AST) -> bool:
        if isinstance(node, ast.NamedExpr):  # `(name := cfg["name"]) == ...`
            node = node.value
        return _reads_name(node) or (isinstance(node, ast.Name) and node.id in self.bound)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(self._is_read(x) for x in [node.left, *node.comparators]):
            self.hits.append(node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("startswith", "endswith") \
                and self._is_read(f.value):
            self.hits.append(node.lineno)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        if self._is_read(node.subject):
            self.hits.append(node.lineno)
        self.generic_visit(node)


def test_no_source_file_reads_the_config_name_to_decide(repo_root):
    """Reading cfg["name"] for anything but labelling is the same smell."""
    offenders = []
    for path in _sources(repo_root):
        tree = ast.parse(path.read_text())
        r = _NameReader()
        r.visit(tree)
        lines = path.read_text().splitlines()
        offenders += [f"{path.relative_to(repo_root)}:{ln} {lines[ln - 1].strip()}"
                      for ln in r.hits]
    assert not offenders, (
        "config name used as a control-flow signal:\n  " + "\n  ".join(offenders))


_NAME_READS = {
    "one-line compare": 'if cfg["name"] == "x": pass',
    "get compare": 'if cfg.get("name") == "x": pass',
    "startswith on the read": 'if cfg["name"].startswith("dre"): pass',
    "bind then compare": 'name = cfg["name"]\nif name == "x": pass',
    "bind then membership": 'name = cfg.get("name")\nif "l2r" in name: pass',
    "bind then endswith": 'n = ctx.cfg["name"]\nif n.endswith("_pruned"): pass',
    "bind then match": 'name = self.cfg["name"]\nmatch name:\n    case "x":\n        pass',
    "walrus": 'if (name := cfg["name"]) == "x": pass',
    "other config-shaped base": 'base_name = base_cfg["name"]\nif base_name != "x": pass',
}


@pytest.mark.parametrize("form", sorted(_NAME_READS))
def test_the_name_reader_catches_a_read_used_to_decide(form):
    r = _NameReader()
    r.visit(ast.parse(_NAME_READS[form]))
    assert r.hits, form


def test_the_name_reader_leaves_labelling_and_other_names_alone():
    r = _NameReader()
    r.visit(ast.parse(
        'group = cfg.get("wandb.group") or cfg.get("name") or "bird"\n'
        'rundir = RunDir(root, cfg["name"], cfg.hash())\n'
        'log.info("BIRD %s", cfg["name"])\n'
        'want = spec["name"]\nif want == "x": pass\n'          # a task spec, not the config
        'def f():\n    name = cfg["name"]\n    return name\n'
        'def g(name):\n    if name == "x": pass\n'))           # a different `name`, other scope
    assert r.hits == []
