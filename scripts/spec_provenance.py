"""The mujoco-wheel provenance rule, in ONE place for every spec generator.

A spec's measured blocks -- reset draws, their spans, the constants -- are properties of
the SOLVER as much as of the scene: the same scene, tree and seeds under two mujoco wheels
can move a column across the 1e-6 line that separates a mover from a constant, so a
guard's verdict can flip on the wheel and a generator can call a correct spec stale.

This module exists because the alternative is one copy of the rule per generator, and
two copies drift: a rule test that imports one copy cannot catch the other.
One rule, one copy, every generator calling it.
"""
from __future__ import annotations

import platform
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


def measured_mujoco() -> str:
    """The wheel of the interpreter TAKING the measurements.

    Every generator calls `build()` for `--check` exactly as for a write, and `build`
    runs the adapter in that same process, so a version read here is by construction the
    one that produced the numbers beside it -- measurement and emission are not
    separable, which is what makes this a record rather than an assertion.
    """
    import mujoco

    return str(mujoco.__version__)


def recorded_mujoco(path: Path) -> Optional[str]:
    """The wheel a COMMITTED spec says its measured blocks were taken under.

    Read from the file, not from a rebuilt doc: the comparison has to happen before a
    rebuild, because its whole purpose is deciding whether rebuilding means anything.
    Absent returns None -- an older spec predating the field has nothing to disagree with.
    """
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None
    stack = (((doc.get("env") or {}).get("library") or {}).get("stack") or {})
    v = stack.get("mujoco")
    return str(v) if v else None


def unjudgeable_under(recorded: Optional[str], installed: str) -> bool:
    """Can a spec recording `recorded` be judged by an interpreter carrying `installed`?

    Two properties this must keep:

    * it compares a spec against the INTERPRETER CHECKING IT, never a derived spec
      against its source. `upstream_assistax_*` is derived from the CPU assistax specs, and
      a child measured under the jax tier's wheel recording 3.13.0 beside a parent recording
      3.3.0 is two correct measurements of two solvers, not a disagreement.
    * absent is not a mismatch, so a spec predating the field stays checkable.
    """
    return bool(recorded) and recorded != installed


# ------------------------------------------------- the WRITE side (foreign wheels)

#: The key a spec carries when it was regenerated under a wheel the tier does not pin,
#: deliberately, with `--allow-foreign-wheel`. OPTIONAL in the schema: an honest spec is
#: never asked to carry it and a spec carrying it still VALIDATES, because the refuser is
#: `tests/test_spec_mujoco_provenance.py`, not the schema -- so the override shows up red
#: on the diff that introduced it, where it can be explained.
OVERRIDE_KEY = "mujoco_regenerated_under_override"

def pinned_mujoco(extra: str, repo: Optional[Path] = None) -> Tuple[str, str]:
    """(version, provenance) of the wheel a TIER INTENDS. Thin wrapper on `pinned_version`.

    Read at run time rather than stored, and read HERE rather than copied into each
    generator, for the reason this module exists: a version written down twice is a
    version that can disagree with itself.

    DELEGATES rather than reading `pyproject.toml` itself. A second mujoco-only reader
    beside `pinned_version` would be the same defect one level up -- two functions
    answering "what version does this tier intend", able to diverge the first time either
    learned something. A tier whose extra holds a range is NOT refused: `uv.lock` resolves
    a range to one exact version, and that resolved version is precisely what
    `uv sync --extra <tier>` installs, so it IS the tier's intent and not a guess. Every
    mujoco tier pins exactly, so this matters only if someone loosens an extra.

    The provenance string travels with the version so every refusal can say where its
    number came from -- a pin quoted with no source is the thing that makes two readers
    disagree silently.
    """
    return pinned_version(extra, "mujoco", repo)


def refuse_foreign_write(extra: str, installed: str, allow: bool,
                         repo: Optional[Path] = None) -> Optional[str]:
    """A refusal message when writing under a wheel the tier does not pin, else None.

    COMPARED AGAINST THE TIER'S PIN, NEVER THE SPEC'S RECORDED VALUE, and the difference
    is the whole design. The pin is the declaration of intent; the recorded value is
    evidence of what happened last time. Asking "are you working under the wheel this
    tier says it uses" makes a legitimate pin move ONE change -- move the pin, regenerate
    under the newly pinned wheel with no flag, and the diff shows every `stack.mujoco`
    moving beside its re-measured numbers. Comparing against the recorded value instead
    would make every spec in that change demand the override and turn a pin move into a
    two-step operation with a tier-wide red in between, which reads as a defect in this
    guard.

    `allow` is `--allow-foreign-wheel`: named for what it does, never a neutral
    `--force`, so the flag in a shell history says which rule was set aside.
    """
    pin, whence = pinned_mujoco(extra, repo)
    if installed == pin or allow:
        return None
    return (f"refusing to write: mujoco {installed} is installed but the `{extra}` tier "
            f"pins {pin} ({whence}), so every measured block this run would write is that other "
            f"solver's. Install the pinned wheel (`uv sync --extra {extra}`) and re-run. "
            f"If crossing wheels is deliberate, pass --allow-foreign-wheel -- the spec "
            f"then records it in `{OVERRIDE_KEY}` and the provenance test goes red until "
            "someone accounts for it.")


def override_note(extra: str, installed: str, repo: Optional[Path] = None) -> Optional[str]:
    """The `OVERRIDE_KEY` value for a run under a foreign wheel, or None when honest.

    Carries BOTH wheels and nothing else: `"3.3.0 -> 3.13.0"`.

    NO TIMESTAMP, DELIBERATELY. A generator's `--check` rebuilds the doc and compares
    bytes, so any field that differs between the write run and the check run makes the
    file STALE AGAINST ITSELF FOREVER -- the check would regenerate a new timestamp and
    report drift on a spec nobody had touched, on every run, and the only repair that
    silenced it would be another write. `derive_jax_spec.py`'s recorded command obeys the
    same constraint (no mode flags, no `--date`, no interpreter path). Anything a
    `--check` cannot reproduce byte-for-byte does not belong in a generated file.

    WHEN is not lost: the commit that introduced the key carries it, with the diff.
    DIRECTION is the part a reader needs inline, because it says which wheel the numbers
    beside it actually came from.
    """
    pin, _ = pinned_mujoco(extra, repo)
    if installed == pin:
        return None
    return f"{pin} -> {installed}"


def check_under_foreign_wheel(extra: str, installed: str,
                             repo: Optional[Path] = None) -> Optional[str]:
    """A note for a `--check` run under a wheel the tier does not pin, else None.

    FOUND BY RUNNING THE GUARD, not by reading it. With a tier's pin moved and the old
    wheel still installed, `--check` rebuilds each spec, the rebuild adds `OVERRIDE_KEY`
    (because installed != pinned is exactly the override condition), the bytes differ,
    and every spec is reported as plain `stale:`. The byte comparison is self-consistent
    -- a spec written under those same conditions round-trips -- but the WORD is wrong,
    and wrong in the expensive direction: "stale" names a defect whose obvious repair is
    to regenerate, and regenerating here stamps the override key into every spec in the
    tier. That is the same "reporting stale invites the one repair that destroys the
    record" failure the read side already guards, arriving through the write side.

    So a check under a foreign wheel says so BEFORE any stale list, and says what the
    list does and does not mean. It is a note rather than a refusal because the check
    itself is still worth running: the specs recorded under this wheel are still checked
    against it, and that half of the answer is sound.
    """
    pin, whence = pinned_mujoco(extra, repo)
    if installed == pin:
        return None
    return (f"note: mujoco {installed} is installed and the `{extra}` tier pins {pin} "
            f"({whence}). Any `stale` below means the committed spec differs from what "
            f"THIS wheel would write, which is expected and is not a defect in the spec. "
            f"Do NOT regenerate to clear it: under this wheel every write records "
            f"`{OVERRIDE_KEY}`, so the repair would stamp a deliberate-crossing marker "
            f"into the whole tier. Install the pinned wheel (`uv sync --extra {extra}`) "
            "and check again.")


def override_entries(extra: str, installed: str, repo: Optional[Path] = None) -> Dict[str, str]:
    """`{}` on an honest run, `{OVERRIDE_KEY: note}` on a deliberate crossing.

    Spread into a generator's `stack` dict (`{..., **override_entries(EXTRA, installed)}`)
    so the key is present exactly when the crossing is, in both `--check` and write, with
    no per-generator conditional to get wrong.
    """
    note = override_note(extra, installed, repo)
    return {OVERRIDE_KEY: note} if note else {}


# --------------------------------------------- the tier's intended version (any library)

def _read_toml(path: Path):
    """(parsed-or-None, reader-name). NAMED because a regex read is not a parsed one.

    `tomllib` is stdlib only from 3.11 and this project declares
    `requires-python = ">=3.10"`, so the chain is tomllib -> tomli -> an anchored regex
    over `[[package]]` blocks. Every message that quotes a version says which reader
    produced it, so nobody mistakes the fallback's answer for a parse.
    """
    for mod_name in ("tomllib", "tomli"):
        try:
            mod = __import__(mod_name)
        except ImportError:
            continue
        with path.open("rb") as fh:
            return mod.load(fh), mod_name
    return None, "regex"


def _lock_candidates(lock: Path, library: str):
    """[(version, [marker strings])] for every `[[package]]` entry naming `library`."""
    parsed, reader = _read_toml(lock)
    out = []
    if parsed is not None:
        for pkg in parsed.get("package", []):
            if pkg.get("name") == library:
                out.append((str(pkg.get("version")), [str(m) for m in
                                                      pkg.get("resolution-markers", [])]))
        return out, reader
    text = lock.read_text()
    for block in re.split(r'^\[\[package\]\]$', text, flags=re.M)[1:]:
        m = re.search(r'^name = "([^"]+)"', block, re.M)
        if not m or m.group(1) != library:
            continue
        v = re.search(r'^version = "([^"]+)"', block, re.M)
        markers = re.findall(r'"((?:[^"\\]|\\.)*python_full_version[^"]*)"', block)
        out.append((v.group(1) if v else "?", markers))
    return out, reader


def _vtuple(s):
    return tuple(int(x) for x in re.findall(r"\d+", s)[:3])


def _marker_admits(marker: str, pyver: str) -> bool:
    """Does `marker`'s python_full_version clauses admit `pyver`? Clauses only."""
    clauses = re.findall(r"python_full_version\s*(==|!=|>=|<=|>|<)\s*'([^']+)'", marker)
    if not clauses:
        return True                      # says nothing about python: admits every version
    want = _vtuple(pyver)
    for op, raw in clauses:
        if raw.endswith(".*"):           # `== '3.11.*'` -- prefix match on the given parts
            pre = _vtuple(raw)
            ok = want[:len(pre)] == pre
            ok = ok if op == "==" else not ok
        else:
            got, ref = want[:len(_vtuple(raw))], _vtuple(raw)
            ok = {"==": got == ref, "!=": got != ref, ">=": got >= ref,
                  "<=": got <= ref, ">": got > ref, "<": got < ref}[op]
        if not ok:
            return False                 # clauses within one marker are ANDed
    return True


def _extra_requirements(pyproject: Path, extra: str) -> Tuple[List[str], str]:
    """(requirement strings of `[project.optional-dependencies].<extra>`, reader name).

    Empty list when the extra does not exist. The regex branch exists only for a
    tomllib-less, tomli-less interpreter and says so, because an answer from a
    hand-rolled parse should never be mistaken for a parsed one.
    """
    parsed, reader = _read_toml(pyproject)
    if parsed is not None:
        opt = (parsed.get("project") or {}).get("optional-dependencies") or {}
        return [str(r) for r in opt.get(extra, [])], reader
    text = pyproject.read_text()
    m = re.search(rf'^{re.escape(extra)} = \[(.*?)^\]', text, re.M | re.S)
    if not m:
        m = re.search(rf'^{re.escape(extra)} = \[([^\n]*)\]', text, re.M)
    if not m:
        return [], reader
    return re.findall(r'"([^"]+)"', m.group(1)), reader


def pinned_version(extra: str, library: str, repo: Optional[Path] = None,
                   pyver: Optional[str] = None) -> Tuple[str, str]:
    """(version, provenance) that a TIER INTENDS for `library`. Never a range.

    Two sources, in order, because a tier expresses intent in whichever it has:

    * an EXACT `==` pin in the extra (`assistax = ["mujoco==3.3.0"]`) -- the tier said the
      version outright;
    * else `uv.lock`'s resolved version for that library, which IS exact even when the
      extra is a range (`jax[cuda13]>=0.4.37,<=0.8.0` locks to 0.8.0). The lock is what
      makes "every jax venv is aligned" a fact rather than an intention.

    A RANGE IS NEVER RETURNED. Comparing an interpreter against a range answers "are you
    allowed here", which is not the question a byte-for-byte `--check` asks.

    SELECTION IS ON `python_full_version` ALONE. uv forks a locked
    resolution, so one library can appear at several versions -- jax is 0.6.2 below 3.11
    and 0.8.0 from 3.11. Anything surviving that filter is a fork on another axis
    (platform), which is refused loudly and by name rather than guessed at, because a
    lock that genuinely forks per platform is a thing a person should look at.

    The provenance string names the extra, the reader, and the python this was evaluated
    against, so a number quoted from here carries how it was obtained.
    """
    root = repo or Path(__file__).resolve().parents[1]
    py = pyver or platform.python_version()

    # PARSED, NOT REGEXED, and the regex it replaces was silently wrong. The old
    # `^<extra> = \[([^\]]*)\]` stops at the FIRST `]` in the block -- including one
    # inside a requirement string. `jax = ["jax[cuda13]>=0.4.37,...", "mujoco>=3.3.7",
    # ...]` therefore read as just `"jax[cuda13` and the reader saw none of the eight
    # requirements after it. It fails SILENTLY in the worst direction: an extra that
    # pins exactly, after any bracketed requirement, looks unpinned and the answer comes
    # from uv.lock instead -- a different source, quietly, with the provenance string
    # confidently naming the wrong one. `metaworld` only escapes because its
    # `mujoco==3.3.0` happens to sit before its `gymnasium[mujoco]`; reordering two
    # requirements would have broken it with nothing going red.
    #
    # A requirement list is TOML, so read it as TOML. The regex fallback below is for
    # the no-tomllib/no-tomli case only, and it names itself when it answers.
    reqs, why = _extra_requirements(root / "pyproject.toml", extra)
    for req in reqs:
        exact = re.match(rf'{re.escape(library)}\s*==\s*([0-9][^\s,;]*)$', req.strip())
        if exact:
            return exact.group(1), f"pyproject `{extra}` extra, exact pin ({why})"

    lock = root / "uv.lock"
    cands, reader = _lock_candidates(lock, library)
    if not cands:
        raise LookupError(
            f"no `{library}` in the `{extra}` extra as an exact pin and none in uv.lock "
            f"(read by {reader}). A tier whose intended version cannot be named has "
            "nothing for a measured record to be checked against.")
    live = [(v, mk) for v, mk in cands if not mk or any(_marker_admits(x, py) for x in mk)]
    if len(live) == 1:
        return live[0][0], (f"uv.lock (read by {reader}) for extra `{extra}`, "
                            f"python_full_version {py}")
    if not live:
        raise LookupError(
            f"uv.lock (read by {reader}) has {len(cands)} `{library}` entries and none "
            f"admits python_full_version {py}: {[v for v, _ in cands]}.")
    raise LookupError(
        f"uv.lock (read by {reader}) leaves {len(live)} `{library}` versions after "
        f"selecting on python_full_version {py}: {[v for v, _ in live]}. The remaining "
        "fork is on another axis -- markers: "
        + " | ".join(mk for _, mks in live for mk in mks[:2])
        + ". Selecting one would be a guess; name the version explicitly instead.")
