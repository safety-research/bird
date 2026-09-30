#!/usr/bin/env python3
"""Re-anchor the source-line citations in `tasks/*/shared_spec.yaml` after a code edit.

A spec cites where its adapter does things -- `env.reset.entry` names the `_reset`
that runs and the LINES inside it that draw from the RNG, `reward` and
`*_success` blocks name the lines that compute them -- and
`tests/test_task_specs.py::test_the_reset_entry_belongs_to_this_specs_adapter`
holds every cited range inside the span of the symbol it names. That is a
provenance test and it is right to be strict: a range that has drifted off its
def is indistinguishable, on the page, from a range that was pasted from another
task. The cost is that ANY insertion above a cited def in an adapter file ages
every citation below it, in every spec that cites that file -- one method added
to `Pendulum` moved `Acrobot._reset` by 30 lines and failed `acrobot`, and a
method added to every adapter can fail most of the catalogue at once.

This script moves the citations with the code. For every `(path, lines)` pair
in a spec whose `path` is a file of this repo, the cited lines are located in
the BASELINE version of that file (`--base`, default `HEAD`) and
mapped to their positions in the working tree through a line diff
(`difflib.SequenceMatcher`, equal blocks only): a cited line that survived the
edit lands exactly where it went, and a citation is left alone -- and reported
-- when any of its lines was itself changed or deleted, because then no shift
is the truth and a person has to look.

What it does NOT touch, and why:

  * families whose specs are GENERATED (`scripts/gen_assistax_specs.py` for
    `assistax_*`, `scripts/derive_jax_spec.py` for `upstream_assistax_*` and `jax_*`):
    their generators embed the live spans and compare whole files, so the generator
    is the writer and this script skips them (`--include-generated` to override).
  * anything but the digits of a citation. The edit is textual and line-local --
    the specs are hand-written YAML with comments a dump would flatten -- so the
    only bytes that change are the digits of the citations that moved. That
    covers two shapes: the `lines:` value beside a `path:`, and a PROSE mention
    such as `metaworld.py:1302` inside a note, where the basename names exactly
    one of the repo files the specs cite (`env.py:88` names a library file and
    is left alone; a basename two repo files share is reported, never guessed),
    plus a bare `:1303` that follows such a mention IN THE SAME YAML FIELD and
    means the same file (twelve `rng_note`s once cited a `_reset` that had
    moved). The field boundary is load-bearing: a draw's `text:`
    cites its own `source.path` -- a Meta-World wheel file -- with bare numbers,
    and the `rng_note:` a few lines above it names `base.py`; attributed across
    that boundary, 59 wheel-file numbers in ten specs moved by base.py's +17.
  * citations into library paths (`humanoid_bench/...`, `metaworld/...`): the
    library did not change with the repo.

USAGE

    uv run python3 scripts/refresh_spec_lines.py                 # rewrite, report
    uv run python3 scripts/refresh_spec_lines.py --check         # report only; exit 1 if stale
    uv run python3 scripts/refresh_spec_lines.py --base HEAD~1   # a different baseline (default HEAD)
    uv run python3 scripts/refresh_spec_lines.py --against <sha>  # `--base` under its other name
"""
from __future__ import annotations

import argparse
import bisect
import difflib
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent

#: Spec families whose YAML is written by a generator; the generator moves their
#: citations and its `--check` test holds them, so this script leaves them alone.
GENERATED_PREFIXES = ("assistax_", "upstream_assistax_", "jax_")
#: The generator that writes a prefix's specs, where it is not `gen_<prefix>_specs.py`.
GENERATOR_BY_PREFIX = {"upstream_assistax_": "derive_jax_spec.py", "jax_": "derive_jax_spec.py"}


def generator_of(prefix: str) -> str:
    """The script under `scripts/` that writes the specs of `prefix`."""
    return GENERATOR_BY_PREFIX.get(prefix, f"gen_{prefix.rstrip('_')}_specs.py")

#: `path:` values that name a file of THIS repo. Anything else is a library path.
_REPO_PATH = re.compile(r"^(bird|scripts|tasks)/")

#: `metaworld.py:1302` / `toy.py:120-131` inside prose. The basename must resolve
#: to exactly one cited repo path (`_basename_map`), which is what keeps
#: `sawyer_xyz_env.py:79` -- a library file -- out of reach.
_PROSE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*\.py):(\d+)(?:-(\d+))?\b")

_LINES_RE = re.compile(r"^(?P<indent>\s*)lines:\s*(?P<q>['\"]?)(?P<lo>\d+)(?:-(?P<hi>\d+))?(?P=q)"
                       r"(?P<gap>\s*)(?P<rest>#.*)?$", re.M)
#: A `path:` value, as a mapping key OR as the first key of a list element (`- path:`, the shape of
#: `provenance.authored_from` rows). The list form was missed once, and rows drifted with every
#: module edit, invisibly, because the citation test anchors on backticked names and a `what:`
#: such as 'the reward, ...' has none.
_PATH_RE = re.compile(r"^\s*(?:-\s+)?path:\s*['\"]?(?P<path>[^'\"\s]+)['\"]?\s*(#.*)?$", re.M)


def _git_show(base: str, rel: str) -> Optional[str]:
    try:
        return subprocess.run(["git", "show", f"{base}:{rel}"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return None


class LineMap:
    """Old line number -> new line number for one file, via equal diff blocks.

    `None` for a line that was changed or removed: there is no honest place for a
    citation of a line that no longer exists as written."""

    def __init__(self, old: str, new: str) -> None:
        a, b = old.split("\n"), new.split("\n")
        self.map: Dict[int, int] = {}
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    self.map[i1 + k + 1] = j1 + k + 1   # 1-indexed, like the citations
        self.identical = old == new

    def shift(self, lo: int, hi: int) -> Optional[Tuple[int, int]]:
        new_lo, new_hi = self.map.get(lo), self.map.get(hi)
        if new_lo is None or new_hi is None:
            return None
        # Every interior line must have moved by the same amount, or the range
        # straddles an edit and the citation needs a person.
        if any(self.map.get(k) != new_lo + (k - lo) for k in range(lo, hi + 1)):
            return None
        return new_lo, new_hi


def _basename_map(maps: Dict[str, LineMap]) -> Dict[str, Optional[str]]:
    """basename -> the one cited repo path with that name, or None when two share it."""
    out: Dict[str, Optional[str]] = {}
    for rel in maps:
        base = rel.rsplit("/", 1)[-1]
        out[base] = None if base in out else rel
    return out


#: A BARE `:NNN` after a `file.py:` mention -- `` metaworld.py:794 ... at :605 `` -- which
#: the notes use for a second line of the file they just named. Preceded by a space or
#: an opening bracket so a time (`12:30`), a ratio and a YAML key never match.
_BARE_RE = re.compile(r"(?<=[\s(\[]):(\d+)(?:-(\d+))?\b")

#: The start of a YAML field -- `rng_note: '...`, `  - role: goal`, `text: '...` -- at
#: any indentation. A bare `:NNN` is attributed to a named file only when no field
#: starts between the two: the notes name a file and cite a second line of it inside
#: ONE field, while a draw's `text:` cites the file its own `source.path` names (a
#: Meta-World wheel file, never named with `.py:` in the text) and must not inherit
#: whatever the previous field happened to name. A prose line that begins `word:`
#: inside a multi-line scalar reads as a boundary too, which errs toward leaving a
#: number where it is rather than moving it into another file. Implemented as
#: `_crosses_field` over the field starts, rather than a reset event in the match stream.
_FIELD_RE = re.compile(r"^[ \t]*(?:-[ \t]+)?[A-Za-z_][A-Za-z0-9_.-]*:(?=[ \t]|$)", re.M)


def _crosses_field(bounds: List[int], a: int, b: int) -> bool:
    """True when some YAML field starts in the text between offsets `a` and `b`."""
    return bisect.bisect_right(bounds, a) < bisect.bisect_right(bounds, b)


def refresh_prose(text: str, maps: Dict[str, LineMap], report: List[str], spec_name: str
                  ) -> str:
    """`file.py:NNN` mentions in prose re-anchored through the same maps, and the bare
    `:NNN` mentions that follow one WITHIN THE SAME YAML FIELD, attributed to the LAST
    basename named before them in that field (any `.py:` mention resets it, so a bare
    number after a library file's name stays put; a field boundary resets it to
    nothing, so a bare number in a field that names no file is never moved). Without
    the bare-number rule, ten `rng_note`s saying `(metaworld.py:1302). ... UNSEEDED
    (:1303)` moved only the first. Without the field boundary, a draw's
    `text: 'obj_low/obj_high :40-41 ...'` -- a wheel-file citation, its file named
    only by the entry's `source.path` -- inherited the `rng_note:`'s `base.py` from
    eleven lines above and moved by base.py's insertion count."""
    by_base = _basename_map(maps)
    last_base: List[Optional[str]] = [None]
    last_named_at = -1
    bounds = [m.start() for m in _FIELD_RE.finditer(text)]

    def move(base: str, shown: str, lo_s: str, hi_s: Optional[str], prefix: str) -> str:
        rel = by_base.get(base)
        if rel is None:
            if base in by_base:
                report.append(f"{spec_name}: prose `{shown}` names a basename two cited "
                              "files share; left as written, look by hand")
            return shown
        lm = maps[rel]
        if lm.identical:
            return shown
        lo = int(lo_s)
        hi = int(hi_s) if hi_s else lo
        moved = lm.shift(lo, hi)
        if moved is None:
            report.append(f"{spec_name}: prose `{shown}` -- the cited lines themselves "
                          "changed; left as written, look by hand")
            return shown
        new_lo, new_hi = moved
        if (new_lo, new_hi) == (lo, hi):
            return shown
        value = f"{new_lo}-{new_hi}" if hi_s else f"{new_lo}"
        tag = "" if prefix else f" (bare, after {base})"
        report.append(f"{spec_name}: prose `{shown}` -> `{prefix}:{value}`{tag}")
        return f"{prefix}:{value}"

    # Matches of both shapes, in TEXT order, so a bare mention takes the basename most
    # recently named before it. A bare match that falls inside a basename match (the
    # `:1302` of `metaworld.py:1302`) is skipped by position.
    events = sorted([(m.start(), 0, m) for m in _PROSE_RE.finditer(text)]
                    + [(m.start(), 1, m) for m in _BARE_RE.finditer(text)],
                    key=lambda e: (e[0], e[1]))
    out: List[str] = []
    pos = 0
    for start, kind, m in events:
        if start < pos:
            continue
        out.append(text[pos:start])
        if kind == 0:
            base = m.group(1)
            last_base[0] = base
            last_named_at = start
            out.append(move(base, m.group(0), m.group(2), m.group(3), base))
        else:
            base = last_base[0]
            if (base is None or base not in by_base
                    or _crosses_field(bounds, last_named_at, start)):
                out.append(m.group(0))
            else:
                out.append(move(base, m.group(0), m.group(1), m.group(2), ""))
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)

def refresh_spec(text: str, maps: Dict[str, LineMap], report: List[str], spec_name: str,
                 *, fields: bool = True, prose: bool = True) -> str:
    """One spec's text with its repo citations re-anchored; appends to `report`.

    `fields` is the `lines:` pass, `prose` the `file.py:NNN` pass; both by default,
    one at a time when a tree has already had the other applied (`--prose-only`)."""
    if not fields:
        return refresh_prose(text, maps, report, spec_name) if prose else text
    lines = text.split("\n")
    current_path: Optional[str] = None
    path_line = -1
    out = list(lines)
    for i, ln in enumerate(lines):
        m = _PATH_RE.match(ln)
        if m:
            current_path, path_line = m.group("path"), i
            continue
        m = _LINES_RE.match(ln)
        if not m or current_path is None or i - path_line > 6:
            continue
        if not _REPO_PATH.match(current_path) or current_path not in maps:
            continue
        lm = maps[current_path]
        if lm.identical:
            continue
        lo = int(m.group("lo"))
        hi = int(m.group("hi")) if m.group("hi") else lo
        moved = lm.shift(lo, hi)
        if moved is None:
            report.append(f"{spec_name}: {current_path} lines {lo}-{hi} -- the cited lines "
                          "themselves changed; left as written, look by hand")
            continue
        new_lo, new_hi = moved
        if (new_lo, new_hi) == (lo, hi):
            continue
        value = f"{new_lo}-{new_hi}" if m.group("hi") else f"{new_lo}"
        q = m.group("q")
        rest = (m.group("gap") + m.group("rest")) if m.group("rest") else ""
        out[i] = f"{m.group('indent')}lines: {q}{value}{q}{rest}"
        report.append(f"{spec_name}: {current_path} lines {lo}-{hi} -> {value}")
    joined = "\n".join(out)
    return refresh_prose(joined, maps, report, spec_name) if prose else joined


def cited_repo_paths(spec_texts: Dict[str, str], root: Path = ROOT) -> List[str]:
    """Every repo file the specs cite: by `path:`, and by a prose `file.py:NNN` whose
    basename names exactly one file under `bird/` or `scripts/` (so `base.py:410` in a
    note reaches `bird/envs/base.py` even where no `path:` cites that file, and a
    library basename such as `env.py` resolves to nothing here and stays prose)."""
    paths = set()
    bases = set()
    for text in spec_texts.values():
        for m in _PATH_RE.finditer(text):
            p = m.group("path")
            if _REPO_PATH.match(p):
                paths.add(p)
        for m in _PROSE_RE.finditer(text):
            bases.add(m.group(1))
    known = {p.rsplit("/", 1)[-1] for p in paths}
    for base in sorted(bases - known):
        hits = [q for d in ("bird", "scripts") for q in (root / d).rglob(base)
                if q.is_file()]
        if len(hits) > 1:
            # The specs describe environments, so a basename an adapter file shares
            # with something else in the tree (`base.py`: bird/envs against bird/llm)
            # means the adapter one; anything still ambiguous is left as prose.
            envs = [q for q in hits if q.parent == root / "bird" / "envs"]
            hits = envs if len(envs) == 1 else []
        if len(hits) == 1:
            paths.add(str(hits[0].relative_to(root)))
    return sorted(paths)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", "--against", dest="base", default="HEAD",
                    help="the git ref the citations are correct against (default HEAD, i.e. "
                         "refresh the working tree's uncommitted edits). `--against` is the same "
                         "flag under another name")
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if anything is stale")
    ap.add_argument("--include-generated", action="store_true",
                    help="also touch the generated families (normally their generator does)")
    ap.add_argument("--force", action="store_true",
                    help="also touch specs that already differ from --base (see the skip note)")
    ap.add_argument("--prose-only", action="store_true",
                    help="skip the `lines:` pass; re-anchor only `file.py:NNN` prose mentions "
                         "(for a tree whose `lines:` fields were already refreshed)")
    ap.add_argument("--no-prose", action="store_true", help="skip the prose pass")
    args = ap.parse_args(argv)

    specs = {p.parent.name: p for p in sorted((ROOT / "tasks").glob("*/shared_spec.yaml"))}
    if not args.include_generated:
        specs = {k: v for k, v in specs.items() if not k.startswith(GENERATED_PREFIXES)}
    # A spec that already differs from the baseline may already carry moved citations,
    # and mapping those AGAIN from the baseline would shift them twice. The tool is
    # one-shot per baseline: such specs are skipped and named, and the way to run it
    # again is to commit and use that commit as `--base`.
    already = set(subprocess.run(["git", "diff", "--name-only", args.base, "--", "tasks"],
                                 cwd=ROOT, capture_output=True, text=True).stdout.split())
    # A spec the baseline does not have at all (new and still untracked) carries citations
    # made against the CURRENT tree by its author or its generator, never the baseline's: mapping
    # them from the baseline would move correct citations. Once it
    # is committed it differs from the baseline and the rule above skips it; until then, this one.
    untracked = set(subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "--", "tasks"],
                                   cwd=ROOT, capture_output=True, text=True).stdout.split())
    new_specs = sorted(k for k, v in specs.items() if str(v.relative_to(ROOT)) in untracked)
    if new_specs:
        print(f"{len(new_specs)} spec(s) are not in {args.base} (untracked) and are skipped -- their "
              f"citations are the current tree's: {', '.join(new_specs)}", file=sys.stderr)
        specs = {k: v for k, v in specs.items() if k not in new_specs}
    skipped = sorted(k for k, v in specs.items()
                     if str(v.relative_to(ROOT)) in already)
    if skipped and not args.force:
        print(f"{len(skipped)} spec(s) already differ from {args.base} and are skipped "
              f"(their citations may already be current; commit and re-base to refresh "
              f"them): {', '.join(skipped)}", file=sys.stderr)
        specs = {k: v for k, v in specs.items() if k not in skipped}
    elif skipped:
        # `--force` maps citations as if they were still the baseline's. Run it TWICE on
        # the same tree and every citation the first run moved is moved again -- by
        # content the map cannot tell a moved citation from an unmoved one (measured:
        # `metaworld.py:1302 -> 1350` became `-> 1399`). It is a one-shot: to
        # re-run, restore the specs to the state the baseline's code was correct for
        # (`git checkout <that commit> -- tasks/`) and run once.
        print(f"--force: {len(skipped)} spec(s) already differ from {args.base}; mapping their "
              "citations as the baseline's. ONE SHOT -- do not run this again on the same "
              "tree without restoring the specs first.", file=sys.stderr)
    texts = {k: v.read_text() for k, v in specs.items()}
    maps: Dict[str, LineMap] = {}
    for rel in cited_repo_paths(texts):
        old = _git_show(args.base, rel)
        new_path = ROOT / rel
        if old is None or not new_path.is_file():
            continue
        maps[rel] = LineMap(old, new_path.read_text())
    report: List[str] = []
    changed = 0
    for name, path in specs.items():
        new = refresh_spec(texts[name], maps, report, name,
                           fields=not args.prose_only, prose=not args.no_prose)
        if new != texts[name]:
            changed += 1
            if not args.check:
                path.write_text(new)
    for line in report:
        print(line)
    moved_files = sorted(k for k, m in maps.items() if not m.identical)
    print(f"{len(moved_files)} cited file(s) changed vs {args.base}: {', '.join(moved_files) or '-'}",
          file=sys.stderr)
    print(f"{changed} spec(s) {'would be ' if args.check else ''}rewritten, "
          f"{sum(1 for r in report if 'look by hand' in r)} citation(s) need a person",
          file=sys.stderr)
    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
