#!/usr/bin/env python3
"""Turn one config plus one key into a sweep: `base × [values] -> N configs`.

    python3 scripts/ablate.py eureka generate.history_mode \\
        none last_iteration cumulative_append full_dialogue rolling_summary

    python3 scripts/ablate.py eureka verify.tpe.on_failure skip_training train_anyway --run
    python3 scripts/ablate.py --list-values eureka generate.history_mode

Once every design choice is a leaf key, a sweep over any one
of them is automatic. This script is that sentence, executable. It writes one
config per value into `configs/sweeps/<base>__<key>/`, each `extends:` the base
with exactly one key set -- so every generated config differs from its siblings
in one key and nothing else, which is the only condition under which the result
attributes anything.

The key and the values are checked against `bird/schema.py` BEFORE anything is
written, and a value backed by a registry family is checked against the family.
A sweep whose values are silently rejected at load time is a sweep that produced
five copies of the default, and you would not find out until the plots were flat.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, List

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bird import config as _config  # noqa: E402
from bird.config import ConfigError, load, unflatten  # noqa: E402
# CONFIG_ROOT is resolved ON USE, not bound at import. `bird/config.py` serves
# it through a PEP 562 module `__getattr__` precisely so that a non-editable
# install fails where the value is WANTED rather than where the module is
# loaded -- and `from bird.config import CONFIG_ROOT` at module scope throws
# that away, running the resolver at import time for every caller including
# those that never read a config. Import the MODULE and reach through it.

from bird.schema import SCHEMA  # noqa: E402

def _sweep_root():
    return _config.CONFIG_ROOT / "sweeps"


def allowed_values(key: str) -> List[Any] | None:
    """The schema's allowed set for `key`, resolving a registry family if named."""
    field = SCHEMA.get(key)
    if field is None:
        return None
    if field.enum is not None:
        return list(field.enum)
    if field.kind is not None:
        from bird.registry import names
        return list(names(field.kind))
    return None


def coerce(raw: str) -> Any:
    """Parse a CLI word the same way `--set key=value` does, so the two agree."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def slug(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "p") or "empty"


def check(base: str, key: str, values: List[Any]) -> List[str]:
    problems: List[str] = []
    if key not in SCHEMA:
        near = [k for k in SCHEMA if k.rsplit(".", 1)[-1] == key.rsplit(".", 1)[-1]]
        hint = f" (did you mean {near[0]!r}?)" if near else ""
        problems.append(f"{key!r} is not a config key{hint}")
        return problems

    allowed = allowed_values(key)
    if allowed is not None:
        for v in values:
            if v not in allowed:
                problems.append(f"{key}={v!r} is not one of {sorted(map(str, allowed))}")

    # Cheapest real check available: actually resolve each value against the
    # base config, so cross-key coherence rules fire here rather than at run
    # time on a cluster. A value that is legal per-key but incoherent with the
    # base is the failure this catches -- e.g. sweeping `verify.tpe.rule` on a
    # config whose quality_screen is not `tpe`.
    for v in values:
        try:
            load(base, overrides={key: v})
        except ConfigError as exc:
            first = str(exc).strip().splitlines()[-1].strip(" -")
            problems.append(f"{key}={v!r} does not resolve against {base!r}: {first}")
    return problems


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ablate.py",
        description="Generate one config per value of a single key.")
    p.add_argument("base", help="config path, or a bare name under configs/")
    p.add_argument("key", help="dotted config key, e.g. generate.history_mode")
    p.add_argument("values", nargs="*", help="values to sweep; omit with --all")
    p.add_argument("--all", action="store_true",
                   help="sweep every value the schema allows for the key")
    p.add_argument("--list-values", action="store_true",
                   help="print the allowed values for the key and exit")
    p.add_argument("--out", default=None, help="output dir (default configs/sweeps/<base>__<key>)")
    p.add_argument("--run", action="store_true", help="run each generated config after writing")
    p.add_argument("--dry-run", action="store_true", help="print what would be written")
    args = p.parse_args(argv)

    if args.list_values:
        allowed = allowed_values(args.key)
        if allowed is None:
            print(f"{args.key}: no enumerable value set (free int/float/str/list/dict)")
            return 0 if args.key in SCHEMA else 2
        print(f"{args.key}: " + " ".join(map(str, allowed)))
        return 0

    values = [coerce(v) for v in args.values]
    if args.all:
        allowed = allowed_values(args.key)
        if allowed is None:
            print(f"error: --all needs an enumerable key; {args.key} has no fixed value set",
                  file=sys.stderr)
            return 2
        values = allowed
    if len(values) < 2:
        print("error: a sweep needs at least two values (or --all)", file=sys.stderr)
        return 2

    problems = check(args.base, args.key, values)
    if problems:
        print("error: refusing to write a sweep that would not load:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    base_cfg = load(args.base)
    base_name = base_cfg["name"]
    base_file = Path(base_cfg.source)
    out = Path(args.out) if args.out else _sweep_root() / f"{base_name}__{args.key}"

    written: List[Path] = []
    for value in values:
        name = f"{base_name}__{args.key.rsplit('.', 1)[-1]}__{slug(value)}"
        body = {
            "extends": str(Path("..") / ".." / base_file.relative_to(_config.CONFIG_ROOT)),
            "name": name,
        }
        body.update(unflatten({args.key: value}))
        text = (
            f"# GENERATED by scripts/ablate.py -- do not edit by hand.\n"
            f"# One arm of a single-key sweep over `{args.key}` off `{base_name}`.\n"
            f"# Every sibling differs from this file in that one key and nothing else,\n"
            f"# which is the only condition under which the comparison attributes anything.\n"
            + yaml.safe_dump(body, sort_keys=False, default_flow_style=False)
        )
        path = out / f"{name}.yaml"
        if args.dry_run:
            print(f"--- {path} ---\n{text}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        load(path)  # a generated config that does not validate is a bug here, not later
        written.append(path)
        print(f"  wrote  {path.relative_to(REPO)}")

    if args.dry_run:
        return 0
    print(f"\n{len(written)} arm(s) in {out.relative_to(REPO)}")

    if args.run:
        for path in written:
            print(f"\n=== running {path.stem} ===")
            rc = subprocess.call([sys.executable, str(REPO / "bird.py"), "--config", str(path)])
            if rc != 0:
                print(f"  arm {path.stem} exited {rc}", file=sys.stderr)
    else:
        print("\nrun them with:")
        print(f"  for f in {out.relative_to(REPO)}/*.yaml; do python3 bird.py -c \"$f\"; done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
