#!/usr/bin/env python3
"""Diff two configs, grouped by design-space section (the `§` of each key in bird/schema.py).

    python3 scripts/config_diff.py eureka rda
    python3 scripts/config_diff.py eureka card --only 2,4
    python3 scripts/config_diff.py eureka gt --format md

This is the "what is this paper's contribution?" tool. `bird.py --diff` prints the
same key set flat; this one groups it by `bird.schema.SCHEMA[key].section`, which
is what turns a list of forty keys into a sentence. A claim such as "these two
methods differ mostly in what stands in for the fitness function (§4) and what can
be checked before paying for RL (§2)" can only be read off a section-grouped diff,
and the section counts printed at the bottom state it, per pair, as a number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bird.config import ConfigError, load  # noqa: E402
from bird.schema import SCHEMA  # noqa: E402

SECTION_TITLES = {
    "§0": "Loop-invariant  (env, LLM, iteration count, budget, carry)",
    "§1": "Reward Generation",
    "§2": "Reward Verification  (validity + quality screens)",
    "§3": "Policy Training  (the inner loop)",
    "§4": "Reward Evaluation  (the scalar and the prose)",
    "§5": "Reward Selection",
    "§6": "Reward Update",
    "?": "Unknown section  (key not in the schema)",
}
ORDER = ["§0", "§1", "§2", "§3", "§4", "§5", "§6", "?"]


def _fmt(value: Any, width: int = 34) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 1] + "…"


def group(diff: Dict[str, Tuple[Any, Any]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key in diff:
        field = SCHEMA.get(key)
        out.setdefault(field.section if field else "?", []).append(key)
    return out


def render_text(a_name: str, b_name: str, diff: Dict[str, Tuple[Any, Any]],
                grouped: Dict[str, List[str]], only: List[str]) -> str:
    lines = [
        "",
        f"  {a_name}  ->  {b_name}",
        f"  {len(diff)} key(s) differ across {len(grouped)} section(s)",
        "",
    ]
    width = max((len(k) for k in diff), default=0)
    for section in ORDER:
        keys = grouped.get(section)
        if not keys or (only and section.lstrip("§") not in only):
            continue
        lines.append(f"  {section}  {SECTION_TITLES[section]}")
        lines.append("  " + "-" * 74)
        for key in sorted(keys):
            va, vb = diff[key]
            lines.append(f"    {key:<{width}}  {_fmt(va):>34}  ->  {_fmt(vb)}")
        lines.append("")

    # The per-section tally is the actual output. A pair whose diff is 80% §4
    # and §2 is two methods that disagree about what a reward is worth, not
    # about how to write one.
    lines.append("  by section:")
    for section in ORDER:
        n = len(grouped.get(section, ()))
        if n:
            bar = "#" * min(n, 50)
            lines.append(f"    {section}  {n:>3}  {bar}")
    lines.append("")
    return "\n".join(lines)


def render_md(a_name: str, b_name: str, diff: Dict[str, Tuple[Any, Any]],
              grouped: Dict[str, List[str]], only: List[str]) -> str:
    lines = [f"### `{a_name}` → `{b_name}`", "",
             f"{len(diff)} key(s) differ.", ""]
    for section in ORDER:
        keys = grouped.get(section)
        if not keys or (only and section.lstrip("§") not in only):
            continue
        lines += [f"**{section} {SECTION_TITLES[section]}**", "",
                  "| key | " + a_name + " | " + b_name + " |", "|---|---|---|"]
        for key in sorted(keys):
            va, vb = diff[key]
            lines.append(f"| `{key}` | `{va!r}` | `{vb!r}` |")
        lines.append("")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="config_diff.py",
        description="Diff two BIRD configs, grouped by design-space section (bird/schema.py).")
    p.add_argument("a", help="config path, or a bare name under configs/")
    p.add_argument("b")
    p.add_argument("--only", default="",
                   help="comma-separated section numbers to show, e.g. 2,4")
    p.add_argument("--format", choices=("text", "md"), default="text")
    p.add_argument("--ignore-name", action="store_true", default=True,
                   help="drop the `name` key, which always differs (default: on)")
    args = p.parse_args(argv)

    try:
        a, b = load(args.a), load(args.b)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    diff = a.diff(b)
    if args.ignore_name:
        diff.pop("name", None)
    if not diff:
        print(f"\n  {a['name']} and {b['name']} are identical, which cannot be right.\n")
        return 0

    only = [s.strip().lstrip("§") for s in args.only.split(",") if s.strip()]
    grouped = group(diff)
    render = render_md if args.format == "md" else render_text
    print(render(a["name"], b["name"], diff, grouped, only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
