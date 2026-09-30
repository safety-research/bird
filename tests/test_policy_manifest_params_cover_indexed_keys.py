"""Every constant a registry policy's factory INDEXES out of its params must be
supplied by the manifest.

A manifest with `params: null` against a `make_policy(p)` whose closure reads
`p["z_hover"]` on its first step makes `gather_params` return `{}`: the factory
happily builds a policy, and the first CALL dies with `KeyError: 'z_hover'` --
after the LLM has been paid, in every loader of the entry (`load_policy`, the
BC prior), before any training.

The check is static, so it needs neither the campaign runtime nor an
environment: the factory named by `entry.symbol` is found with `ast`, its LAST
positional parameter is taken to be the params dict (`factory(params)` for a
`ClosurePolicy`, `factory(rig, params)` for a `RigPolicy` -- see
`policy_api.load_policy`), and every `params["<literal>"]` subscript anywhere
inside it (nested closures included) is a key the merged manifest params must
carry. `.get(...)` reads and keys computed at runtime are out of scope, as is
a class symbol. A factory that merges `DEFAULTS` itself indexes the merged
dict under another name and is naturally exempt.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from bird import policies as P


def _indexed_keys(path: Path, symbol: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol), None)
    if fn is None or not fn.args.args:
        return None, set()
    arg = fn.args.args[-1].arg
    keys = {n.slice.value for n in ast.walk(fn)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == arg
            and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}
    return arg, keys


def _entries():
    out = []
    for rec in P.index().values():
        entry = rec.entry or {}
        f, sym = entry.get("file"), entry.get("symbol")
        if not f or not sym:
            continue
        mod = Path(rec.root) / f
        if not mod.exists():
            continue
        arg, keys = _indexed_keys(mod, str(sym))
        if keys:
            out.append(pytest.param(rec, mod, arg, keys, id=rec.id, marks=_KNOWN_BROKEN.get(rec.id, ())))
    return out


#: Entries this scan finds broken that are deliberately NOT fixed in the same change,
#: as `pytest.mark.xfail(strict=True, ...)` marks keyed by policy id: the day a manifest
#: is fixed the entry XPASSes and fails, so an exemption cannot outlive its defect.
_KNOWN_BROKEN: dict = {}


@pytest.mark.parametrize("rec,mod,arg,keys", _entries())
def test_manifest_params_cover_every_key_the_factory_indexes(rec, mod, arg, keys):
    from bird.policy_api import gather_params
    merged = gather_params(rec)
    missing = sorted(keys - set(merged))
    assert not missing, (
        f"{rec.id}: {mod.name}::{rec.entry['symbol']} indexes {arg}[...] for {missing}, which the "
        f"manifest's merged params (kwargs_from + params_file + params = {len(merged)} keys) do not "
        f"supply. The factory builds, then its first call raises KeyError. Write the full set into "
        f"the manifest.")


def test_the_motivating_entry_is_scanned():
    """Without it in the parametrisation the test above passes vacuously."""
    ids = {p.id for p in _entries()}
    assert "mt10_peg_insert/waypoint" in ids, sorted(ids)
