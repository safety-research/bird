"""Every mujoco-tier spec records the wheel its measured blocks were taken under.

UNMARKED ON PURPOSE. `tests/test_task_specs.py` is `pytestmark = pytest.mark.slow`, so
the guards that actually measure a reset block are deselected by `-m "not slow"`, and
the default CI job installs `--extra test`, which carries no mujoco at all. This file
reads YAML and TOML and nothing else, so it is the one check on this property that runs
in every CI selection.

WHAT IT IS FOR. A settled reset is solver-dependent: the same scene and the same seeds
can put a column on either side of the 1e-6 line that separates a mover from a constant
under two mujoco wheels, so the reset-draw guard's verdict can FLIP on the wheel.
`pyproject.toml`'s `[tool.uv] conflicts` already declares `assistax` and `jax` mutually
exclusive extras -- the packaging knows these tiers cannot share an interpreter. The
tests must know it too: run the suite in the jax tier's own venv and the assistax specs
would be judged under MJX's mujoco.

THE TIERS ARE LISTED EXPLICITLY, NOT A GLOB. `_TIERS` names each generator-owned mujoco
tier (today only assistax) with the extra that pins it. A glob over `tasks/*/` would silently absorb a new tier
that forgets to record a version -- the omission would look like coverage. A literal list
makes it a visible edit in one place: a new mujoco tier either appears here or is
conspicuously missing. The jax tiers are deliberately ABSENT: `jax_*` and
`upstream_assistax_*` specs are DERIVED by `scripts/derive_jax_spec.py` rather than owned
by a `gen_*_specs.py` generator, and folding measurements taken under another wheel in
would force this test to accept both -- which is the check dissolving.

The failure this prevents is not the red. It is the REPAIR: a generator's `--check` run
under the wrong wheel calls correct specs stale, and a regeneration there would rewrite
measurements taken under the pinned wheel with the other wheel's -- after which every
test goes green, because the generator and the guard would agree with each other about a
wheel the tier does not run. A wrong spec no test can distinguish from a right one is
worse than a red.

WHY THE RECORDED VALUE MUST BE MEASURED. A constant is an assertion, not a record:
regenerated under another wheel a spec that writes the string `"3.3.0"` as a literal keeps
claiming 3.3.0 while carrying that wheel's numbers, and this test would compare 3.3.0
against 3.3.0 and pass on a fully laundered spec set. The generator reads
`mujoco.__version__` in the process that takes the measurements, so a regeneration under
the wrong wheel records that wheel and fails HERE, in CI, with no mujoco installed and
no env constructed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
#: (spec-id prefix, the pyproject extra that pins that tier's mujoco). EXPLICIT, not a
#: glob -- see the header. Each extra pins mujoco exactly because metaworld does
#: (`pyproject.toml`'s own note at the pin), and the pin is read from the file rather than
#: repeated here: a second literal is the thing this whole test exists to prevent.
_TIERS = (("assistax", "assistax"),)
def _shared():
    """`scripts/spec_provenance.py`, the one copy of every rule this file checks."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "spec_provenance", ROOT / "scripts" / "spec_provenance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PIN = re.compile(r'"mujoco==([0-9][^"]*)"')


def _pinned_mujoco(extra: str) -> str:
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(rf'^{re.escape(extra)} = \[([^\]]*)\]', text, re.M)
    assert m, (f"pyproject.toml has no single-line `{extra} = [...]` extra any more -- "
               "this test reads the pin from there and must be taught the new shape "
               "rather than given a literal")
    pin = _PIN.search(m.group(1))
    assert pin, (f"the `{extra}` extra no longer pins mujoco exactly: {m.group(1)!r}. If "
                 "the tier moved to a range, a spec can no longer record 'the' wheel and "
                 "this test's premise needs revisiting")
    return pin.group(1)


def _specs_of(prefix: str):
    return sorted((ROOT / "tasks").glob(f"{prefix}_*/shared_spec.yaml"))


def _all_cases():
    out = []
    for prefix, extra in _TIERS:
        for path in _specs_of(prefix):
            out.append((path, extra))
    return out


@pytest.mark.parametrize("prefix,extra", _TIERS, ids=[p for p, _ in _TIERS])
def test_every_named_tier_has_specs_to_check(prefix, extra):
    """The parametrisation below is silent on an empty glob, so an entire tier could
    vanish from the check without a word. One case per NAMED tier is what fails then --
    and it fails naming the tier, rather than the suite quietly checking one fewer."""
    assert _specs_of(prefix), (
        f"`{prefix}` is named in _TIERS but no tasks/{prefix}_*/shared_spec.yaml exists. "
        "Either the tier was renamed (fix _TIERS) or its specs are gone (fix that).")
    _pinned_mujoco(extra)   # the pin must also still be readable, per tier


@pytest.mark.parametrize("path,extra", _all_cases(),
                         ids=lambda v: v.parent.name if hasattr(v, "parent") else v)
def test_a_spec_records_the_mujoco_it_was_measured_under(path, extra):
    doc = yaml.safe_load(path.read_text()) or {}
    stack = (((doc.get("env") or {}).get("library") or {}).get("stack") or {})
    recorded = stack.get("mujoco")
    assert recorded, (
        f"{path.parent.name}: env.library.stack.mujoco is absent. Every measured block in "
        "this spec -- the reset draws, their spans, the constants -- is a property of the "
        "solver as much as of the scene, so a spec that does not say which wheel measured "
        "it cannot be checked by anything.")
    # THE OVERRIDE KEY FIRST, so its specific message wins over the generic one --
    # pytest stops at the first failure, and the two failures say different things.
    #
    # NOT REDUNDANT WITH THE PIN CHECK BELOW, even though both go red today: the day a
    # tier legitimately moves its pin to 3.13.0, the pin check goes QUIET for a spec
    # regenerated under the OLD wheel, and this key is the only thing that still speaks.
    override = stack.get(_shared().OVERRIDE_KEY)
    assert not override, (
        f"{path.parent.name} carries `{_shared().OVERRIDE_KEY}: {override!r}`. Its "
        "measured blocks were taken under a wheel this tier does not pin, with "
        "--allow-foreign-wheel. The key records that the crossing was DELIBERATE; it "
        "does not make the spec landable. Either re-measure under the pinned wheel and "
        "drop the key, or move the tier's pin deliberately and say why.")

    pinned = _pinned_mujoco(extra)
    assert str(recorded) == pinned, (
        f"{path.parent.name}: recorded mujoco {recorded!r}, but the `{extra}` extra pins "
        f"{pinned!r}. Either this spec was regenerated under the wrong wheel -- in which "
        "case its measured numbers are that wheel's and must NOT be kept -- or the tier "
        "moved its pin and every spec of the tier needs re-measuring under the new one, "
        "deliberately. Do not edit the recorded string to match.")


# ---- the derived-spec rule ----------------------------------------------------------
# A spec DERIVED by `scripts/derive_jax_spec.py` may be measured under the MJX wheel
# (`mujoco>=3.3.7`, resolving 3.13.0) while its CPU parent is measured under
# `mujoco==3.3.0`. The two disagreeing is CORRECT -- they are different measurements of
# different solvers -- so the refusal must compare a spec against the INTERPRETER
# CHECKING IT and never a child against its parent.
#
# Derived specs are not walked by the cases above, deliberately: `_TIERS` names the tier
# that a `gen_*_specs.py` generator owns. Folding them in would force this test to accept
# both 3.3.0 and 3.13.0, which is the check dissolving (the header's argument).
#
# So this stays a direct exercise of the comparison rather than a read of a committed
# file. It is the case that would otherwise be discovered the day a derived spec is
# checked in the parent's interpreter, by a generator refusing a spec that was right.


def _refuses(recorded: str, installed: str) -> bool:
    """THE SHARED RULE ITSELF, imported rather than restated.

    `scripts/spec_provenance.py` is the one copy every generator calls. Importing a
    GENERATOR here would be weaker in a way that matters: a second generator could keep
    its own inline `recorded != installed`, and these cases would say nothing about it
    while claiming to cover it. Import the rule, not a caller.

    The module is yaml + stdlib at import time -- `mujoco` is imported inside
    `measured_mujoco()` only -- so this stays runnable in the `--extra test` job.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "spec_provenance", ROOT / "scripts" / "spec_provenance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.unjudgeable_under(recorded, installed)


def test_every_generator_calls_the_shared_rule_rather_than_its_own_copy():
    """One copy of the rule, checked at the source.

    A per-generator copy is invisible to the cases above: they would exercise whichever
    copy they imported and pass while another generator drifted. Assert on the SOURCE --
    each generator imports the shared rule and defines no `unjudgeable_under`/
    `_recorded_mujoco`/`_measured_mujoco` of its own.
    """
    # DERIVED FROM `_TIERS`, not hand-written: a list that grows by hand can end with a
    # generator calling the shared rule and UNLISTED here, reached by two changes that
    # were each correct alone. `_TIERS` already names every tier the cases above walk,
    # so deriving from it makes "converted" and "listed" the same fact rather than two
    # copies of one, which is this file's own rule applied to itself.
    for name in [f"gen_{prefix}_specs.py" for prefix, _ in _TIERS]:
        src = (ROOT / "scripts" / name).read_text()
        assert "from spec_provenance import" in src, f"{name} does not import the shared rule"
        for local in ("def unjudgeable_under", "def _recorded_mujoco", "def _measured_mujoco"):
            assert local not in src, (
                f"{name} defines its own `{local.split()[-1]}`. One rule, one copy: a "
                "second definition is invisible to the cases above, which exercise "
                "scripts/spec_provenance.py.")
        assert "recorded != installed" not in src, (
            f"{name} still compares versions inline rather than through "
            "`unjudgeable_under`")


def test_the_tiers_pin_is_read_from_the_repo_and_never_from_the_interpreter():
    """Record-what-the-library-uses, applied to the PIN side rather than the recorded side.

    On the recorded side, `--measure` writing `mujoco: 3.13.0` into `jax_toy`'s spec
    because the interpreter happens to carry mujoco, on a tier whose env never imports
    it, would record a version that belongs to the venv as if it belonged to the task;
    the recorded version is therefore what the LIBRARY uses.

    The write-refusal reintroduces exactly that temptation one step over. It needs "what
    wheel does this tier intend", and the cheapest wrong answer is `import mujoco;
    mujoco.__version__` -- which would make the refusal compare the interpreter against
    itself and pass unconditionally, the failure being silent and total.

    So this asserts INDEPENDENCE, not a value: a fake `mujoco` at a different version is
    installed into `sys.modules`, and the tier's pin does not move. Asserting `== 3.3.0`
    alone would pass just as happily against an implementation that imported the module,
    on any machine where the two agree -- which is every machine this normally runs on,
    and is how such a defect survives until someone runs a second venv.
    """
    import sys
    import types

    SENTINEL = "0.0.0+never-a-published-wheel"
    shared = _shared()
    before = shared.pinned_version("assistax", "mujoco")
    # A SENTINEL THAT CANNOT COLLIDE WITH A REAL PIN. A merely unlikely value such as
    # "9.9.9-not-a-real-wheel" makes the substring guard below fire spuriously the moment
    # a tier's pin moves to 9.9.9 -- the real answer would contain the sentinel, and the
    # case would fail for a reason that has nothing to do with what it tests. A fixture
    # value has to be impossible, not merely unlikely.
    fake = types.ModuleType("mujoco")
    fake.__version__ = SENTINEL
    saved = sys.modules.get("mujoco")
    sys.modules["mujoco"] = fake
    try:
        after = shared.pinned_version("assistax", "mujoco")
        assert after == before, (
            f"the `assistax` tier's pin moved from {before} to {after} when a fake mujoco "
            f"{SENTINEL} was importable. The pin must come from pyproject.toml or "
            "uv.lock -- "
            "reading it from the installed module makes the write-refusal compare the "
            "interpreter against itself, which can never refuse anything.")
        assert SENTINEL not in after[0] and SENTINEL not in after[1], (
            f"the fake module's version {SENTINEL} reached the answer {after}: "
            "the pin was read from the interpreter, not from the repo.")
    finally:
        if saved is None:
            sys.modules.pop("mujoco", None)
        else:
            sys.modules["mujoco"] = saved


def test_an_exact_pin_after_a_bracketed_requirement_is_still_read(tmp_path):
    """A `]` inside a requirement string must not end the extra.

    A reader of the form `^<extra> = \\[([^\\]]*)\\]` stops at the FIRST `]` in the
    block -- including the one in `jax[cuda13]` or `gymnasium[mujoco]`. Any exact pin
    after such a requirement would be invisible, and the failure is silent in the worst
    direction: the extra looks unpinned, the answer comes from `uv.lock` instead, and
    the provenance string confidently names a source that was never consulted.

    `metaworld` would escape only by luck of ordering -- its `mujoco==3.3.0` sits before
    its `gymnasium[mujoco]>=1.1`. Swapping two requirements would break it with nothing
    going red, which is why this case fixes the order that fails rather than asserting
    on the tier that happens to pass. The rest of the suite is green against such a
    reader.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0"\n\n'
        '[project.optional-dependencies]\n'
        'demo = ["gymnasium[mujoco]>=1.1", "mujoco==3.3.0"]\n')
    (tmp_path / "uv.lock").write_text("")      # must NOT be reached

    shared = _shared()
    version, whence = shared.pinned_version("demo", "mujoco", repo=tmp_path)
    assert version == "3.3.0", (
        f"read {version!r} for an extra whose `mujoco==3.3.0` follows a bracketed "
        "requirement. The `]` in `gymnasium[mujoco]` truncated the extra.")
    assert "exact pin" in whence, (
        f"answered {whence!r}: the exact pin was in pyproject and was not used, so the "
        "version came from somewhere else while the extra did pin it.")


def test_a_derived_jax_spec_is_not_refused_because_its_cpu_parent_says_3_3_0():
    # the child, measured under MJX, checked by an MJX interpreter: judgeable
    assert not _refuses(recorded="3.13.0", installed="3.13.0")
    # the parent's 3.3.0 is not an input to that decision at all -- stated as a test
    # because the tempting implementation is "compare the derived spec to its source"
    assert not _refuses(recorded="3.13.0", installed="3.13.0")


def test_the_cpu_parent_is_still_refused_under_the_mjx_wheel():
    # and the converse still holds: the 3.3.0-stamped parent cannot be judged by an
    # interpreter carrying 3.13.0
    assert _refuses(recorded="3.3.0", installed="3.13.0")
    assert not _refuses(recorded="3.3.0", installed="3.3.0")


def test_a_spec_with_no_recorded_version_is_not_refused():
    """Absent is not a mismatch: there is nothing to disagree with, and an older spec
    must not become uncheckable merely because it predates the field."""
    assert not _refuses(recorded="", installed="3.13.0")
