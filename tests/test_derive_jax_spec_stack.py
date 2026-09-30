"""The generated `stack:` block is a function of the library table, and nothing else.

`env.library.stack` MIXES TWO KINDS OF ENTRY, which is the fact a naive filter
gets wrong:

    numpy: '>=1.24'                        DECLARED   -- a requirement
    jax:   0.8.0 (measured ..., cpu)       MEASURED   -- an observation

A filter that whitelists the whole block deletes the declared ones: it would
silently drop `numpy` from every spec the generator touches.

WHAT THESE TESTS HOLD, none of which a `--check` run can tell you, because
`--check` only ever exercises the ONE library that the one committed jax spec
happens to declare:

  * the measured keys a spec records are derived from `_VERIFIED_FROM` and
    not from a hand-built set, so a library verified by mujoco records mujoco
    (otherwise the table could name a verifier no measurement supplied);
  * a measured key the library does not use is REMOVED even when inherited
    from the parent, which `_recorded_stack` alone cannot do -- it filters
    what `--measure` ADDS, and `dict.update` can only add;
  * declared entries survive all of it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def djs():
    """The generator, imported by path -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "derive_jax_spec", REPO / "scripts" / "derive_jax_spec.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["derive_jax_spec"] = mod
    spec.loader.exec_module(mod)
    return mod


def _measured():
    return {"jax": "0.8.0", "python": "3.11.15", "mujoco": "3.13.0"}


def test_the_measured_keys_are_exactly_the_tables_values_plus_jax_and_python(djs):
    """`MEASURED_STACK_KEYS` is derived, so adding a framework cannot be half-done."""
    assert djs.MEASURED_STACK_KEYS == frozenset(
        {"jax", "python"}) | frozenset(djs._VERIFIED_FROM.values())


@pytest.mark.parametrize("library,expected", [
    ("bird", {"jax", "python"}),                 # jax_toy: no simulator at all
    ("mujoco", {"jax", "python", "mujoco"}),     # verified by a package other than jax
    ("jax", {"jax", "python"}),
])
def test_a_spec_records_the_version_its_library_is_verified_by(djs, library, expected):
    got = djs._recorded_stack(library, _measured(), "2026-09-16", ["cpu"])
    assert set(got) == expected, (
        f"library {library!r} recorded {sorted(got)}; `_VERIFIED_FROM` says it is "
        f"verified by {djs._VERIFIED_FROM.get(library)!r}")


@pytest.mark.parametrize("library,expected_dropped", [
    ("bird", ["mujoco"]),
    ("jax", ["mujoco"]),
    ("mujoco", []),
])
def test_an_inherited_measured_key_the_library_does_not_use_is_removed(
        djs, library, expected_dropped):
    """The asymmetry `_recorded_stack` could not fix: `update()` only ADDS.

    A parent spec that gains a `mujoco:` line hands it to a simulator-less
    child untouched, and the filter on the added half is powerless. Latent
    today -- no committed parent carries one -- which is why it needs a test
    rather than a bug report: nothing else would notice the day it fires.
    """
    stack = {"numpy": ">=1.24", "python": "p", "jax": "j", "mujoco": "mu"}
    assert djs._strip_foreign_measured(stack, library) == expected_dropped
    for k in expected_dropped:
        assert k not in stack


@pytest.mark.parametrize("library", ["bird", "mujoco", "jax", "unknown-framework"])
def test_a_declared_requirement_is_never_stripped(djs, library):
    """`numpy` survives for EVERY library, including the ones with the smallest
    allowed set and one the table does not know at all.

    `bird` is the sharpest case and the reason this test is parametrised over
    libraries rather than asserting once: it is the tier whose allowed set is
    smallest, so it is where a whitelist-the-whole-block fix bites hardest --
    and it is the tier of the only committed jax spec.
    """
    stack = {"numpy": ">=1.24", "jax": "j", "mujoco": "mu"}
    djs._strip_foreign_measured(stack, library)
    assert stack["numpy"] == ">=1.24", (
        "a DECLARED requirement was stripped by a filter meant for measured "
        "versions; this is the deletion the whole-stack filter would have made")


def test_the_recorded_command_is_the_same_whatever_mode_emitted_it(djs, monkeypatch):
    """`--check` and `--write` must record the SAME string or the file is stale forever.

    Verbatim `sys.argv` would record `--check` on a check run and `--write` on
    a write run; recording `--date` fails the same way, since a check does not
    pass it. Either way `--check` regenerates a command string that cannot
    equal the committed one, and the diff is the recorded line disagreeing
    with itself.
    """
    base = ["scripts/derive_jax_spec.py", "--source", "toy_reacher", "--target", "jax_toy",
            "--adapter", "bird/envs/jax_toy.py", "--class-name", "JaxToyReacher"]
    monkeypatch.setattr(sys, "argv", base + ["--check"])
    as_check = djs._invocation()
    monkeypatch.setattr(sys, "argv", base + ["--measure", "--write", "--date", "2026-09-15"])
    as_write = djs._invocation()
    monkeypatch.setattr(sys, "argv", base + ["--measure", "--write", "--date=2026-09-15"])
    as_write_eq = djs._invocation()

    assert as_check == as_write == as_write_eq, (
        f"check recorded {as_check!r} but write recorded {as_write!r}")
    assert as_check.endswith("--measure --write"), as_check
    assert "--check" not in as_check and "--date" not in as_write, as_write


@pytest.mark.parametrize("extra_on_write", [
    ["--date", "2026-09-15"],
    ["--date=2026-09-15"],
    ["--verified-by", "a named verifier (measured on a CPU host)"],
    ["--verified-by=someone else"],
    ["--date", "2026-09-15", "--verified-by", "a named verifier"],
    # `--allow-foreign-wheel` belongs on the drop list for the same reason
    # `--verified-by` does. It gates the write and shapes nothing: the crossing is
    # written from the INSTALLED wheel (`override_entries`), never from the flag,
    # so on the tier's own wheel a write passing it produces byte-identical
    # content to a bare check's and must record a byte-identical command.
    ["--allow-foreign-wheel"],
    ["--allow-foreign-wheel", "--date", "2026-09-15"],
])
def test_bookkeeping_flags_on_the_write_do_not_reach_the_recorded_command(djs, extra_on_write):
    """Whatever the WRITE passed, the string must equal what a bare CHECK records.

    `--verified-by` is bookkeeping only because the generator CARRIES the
    committed value: a check passing no flag produces identical content, so a
    write that passed one would bake it in and the file would be stale against
    its own check from that write onward. The two halves interact.

    `--stack` is deliberately absent from this list: it changes content the
    check does NOT carry, so a check without it diverges on the stack block
    anyway and recording it tells the truth about what produced the file.
    """
    base = ["--source", "toy_reacher", "--target", "jax_toy",
            "--adapter", "bird/envs/jax_toy.py", "--class-name", "JaxToyReacher"]
    as_check = djs._invocation(base + ["--check"])
    as_write = djs._invocation(base + extra_on_write + ["--measure", "--write"])
    assert as_check == as_write, (
        f"a write passing {extra_on_write} recorded a different command than a "
        f"bare check:\n  check: {as_check}\n  write: {as_write}")


@pytest.mark.parametrize("argv0_spelling", [
    "scripts/derive_jax_spec.py",
    "/home/someone/bird/scripts/derive_jax_spec.py",
    "./scripts/derive_jax_spec.py",
    "/usr/bin/pytest",
])
def test_the_script_path_is_a_literal_and_never_argv0(djs, monkeypatch, argv0_spelling):
    """The script path is recorded as a literal, never as typed.

    With the interpreter normalised to a literal `python3` but `argv[0]` taken
    as typed, `uv run python3 $PWD/scripts/...` on the write and `scripts/...`
    on the check would be two spellings of one command -- and the drift gate
    that would notice is `@pytest.mark.jax`, deselected by default, so it would
    surface only at an arbitrary later hand run.

    The `/usr/bin/pytest` row is the same defect reached from the other side:
    if `_invocation` read the GLOBAL `sys.argv` while `main` accepted its own,
    driving the generator from a host process would write the host's command
    line into a committed spec.
    """
    monkeypatch.setattr(sys, "argv", [argv0_spelling, "--source", "toy_reacher",
                                      "--target", "jax_toy", "--check"])
    cmd = djs._invocation()
    assert cmd.startswith("python3 scripts/derive_jax_spec.py "), cmd
    assert argv0_spelling not in cmd or argv0_spelling == "scripts/derive_jax_spec.py", cmd


def test_the_recorded_command_carries_the_arguments_that_reproduce_the_file(djs, monkeypatch):
    """A template naming only --source and --target is not enough.

    Every real invocation also needs --adapter, --class-name and whatever
    --drop-dr-axis the tier drops, so such a recorded command would reproduce
    NO spec the generator writes -- and a recorded command that does not
    reproduce is worse than none, because it is the citable artifact.
    """
    monkeypatch.setattr(sys, "argv", [
        "scripts/derive_jax_spec.py", "--source", "toy_reacher", "--target", "jax_toy",
        "--adapter", "bird/envs/jax_toy.py", "--class-name", "JaxToyReacher",
        "--drop-dr-axis", "action_noise", "--check"])
    cmd = djs._invocation()
    for needed in ("--adapter", "bird/envs/jax_toy.py", "--class-name", "JaxToyReacher",
                   "--drop-dr-axis", "action_noise"):
        assert needed in cmd, f"{needed} missing from the recorded command: {cmd}"


def test_the_argv_parameter_wins_over_the_global(djs, monkeypatch):
    """`_invocation(argv)` must read its ARGUMENT, not `sys.argv`.

    `main()` parses an argv it is handed; if `_invocation` read the global the
    two could disagree, and the docstring names the case that reaches it --
    driven from a host process, the global says
    `/usr/bin/pytest -q tests/... --measure --write` and that is what a
    committed spec would record.

    THIS TEST EXISTS BECAUSE THE REST OF THE FILE CANNOT FAIL ON IT: mutating
    the body to `list(sys.argv[1:])` -- ignoring the parameter outright --
    leaves every other test in this file green. Each of them either
    monkeypatches `sys.argv` or compares two explicit argvs to each other, and
    under the mutation both sides read the same global and the comparison
    holds trivially. The parameter exists for a named defect, so it needs an
    assertion that separates it from the global.
    """
    monkeypatch.setattr(sys, "argv", [
        "/usr/bin/pytest", "-q", "tests/test_derive_jax_spec_stack.py",
        "--source", "SOMEONE_ELSES_TASK", "--target", "SOMEONE_ELSES_TARGET"])
    cmd = djs._invocation(["--source", "toy_reacher", "--target", "jax_toy",
                           "--adapter", "bird/envs/jax_toy.py",
                           "--class-name", "JaxToyReacher", "--check"])
    assert "SOMEONE_ELSES_TASK" not in cmd and "SOMEONE_ELSES_TARGET" not in cmd, cmd
    assert "tests/test_derive_jax_spec_stack.py" not in cmd and "-q" not in cmd.split(), cmd
    assert cmd == ("python3 scripts/derive_jax_spec.py --source toy_reacher "
                   "--target jax_toy --adapter bird/envs/jax_toy.py "
                   "--class-name JaxToyReacher --measure --write"), cmd


_DEFAULT_VERIFIED_BY = "BIRD authors"


def test_a_write_carries_the_committed_verified_by_instead_of_stamping_the_default(djs):
    """`--write` must not erase the recorded name.

    A carry-over guarded by `if args.check` would let a write stamp the
    argparse default over whoever was recorded, while a check copied the
    committed value in before comparing and so could never diff on the field.
    Destroyed by one mode, invisible to the other.
    """
    prov = {"verified_by": "a named verifier (measured on a CPU host)",
            "date": "2026-09-15"}
    got, _ = djs._carry_provenance(prov, _DEFAULT_VERIFIED_BY, _DEFAULT_VERIFIED_BY,
                                   "2026-09-18", is_check=False)
    assert got == prov["verified_by"]


def test_an_explicit_verified_by_still_overrides_the_committed_one(djs):
    """`--verified-by` stays the one deliberate way to change the field."""
    prov = {"verified_by": "a named verifier", "date": "2026-09-15"}
    got, _ = djs._carry_provenance(prov, "another verifier", _DEFAULT_VERIFIED_BY,
                                   "2026-09-18", is_check=False)
    assert got == "another verifier"


@pytest.mark.parametrize("is_check,expected", [(True, "2026-09-15"), (False, "2026-09-18")])
def test_only_a_check_carries_the_committed_date(djs, is_check, expected):
    """DELIBERATELY ASYMMETRIC, and the asymmetry is what makes `--date` droppable.

    A write re-measures, so its date is today's by right. A check does not
    measure anything new, and must carry the committed date or it diffs on
    today's -- which is also why a write's `--date` can leave the recorded
    command without making the file stale against its own check.
    """
    prov = {"verified_by": "a named verifier", "date": "2026-09-15"}
    _, date = djs._carry_provenance(prov, _DEFAULT_VERIFIED_BY, _DEFAULT_VERIFIED_BY,
                                    "2026-09-18", is_check=is_check)
    assert date == expected
