"""A whole test file may not gate itself out on a missing tool.

The defect this repo keeps rediscovering is a check that reports success having
checked nothing, and a module-level skip is its purest form: every assertion in
the file becomes a passing dot and nothing anywhere goes red.

**A skip count is easy to miss.** A skipped test prints as a dot in `-q`
output, and a skip count that silently grows says nothing about which files
stopped running.

The rule here is deliberately narrower than "no skips". Per-test skips are fine
and this repo has legitimate ones -- `test_metaworld.py` importorskips an
optional extra, which a full-deps install (`--extra all`) runs, so they are
covered somewhere. What has no safe form is a `pytestmark` that removes an
ENTIRE file on a tool being absent, because then no other test covers what it
covered and no job anywhere is configured to notice.

**"Covered somewhere" is a check, not a premise** -- see
`test_every_importorskip_names_a_package_something_installs`. Stating an
exemption's justification and not verifying it is the same shape as the defect
this file guards, one level in. The counterexample is concrete: a
`pytest.importorskip("jsonschema")` on the only test that validates the task
specs against their schema, with nothing in any extra installing jsonschema,
means the schema this repo ships is validated by nothing in any configuration.
That test makes the catch automatically.

If you are here because this test failed: do not add your file to an exemption
list. Gate the one test that needs the tool so the absence is a FAILURE by
default and a skip only when someone opts in (an environment variable, say).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

TESTS = Path(__file__).parent

#: Ways a module-level marker can name "this tool is missing". Matching the
#: CALL rather than the string means `which("node")` and `which(BINARY)` are
#: both caught, and a `skipif` on a genuine runtime condition -- a platform, an
#: env var the suite sets itself -- is not.
_ABSENCE_PROBES = {("shutil", "which"), ("importlib.util", "find_spec")}


def _dotted(node: ast.AST) -> str:
    """`shutil.which` from the attribute chain, or "" for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _probes_for_absence(tree: ast.AST) -> bool:
    for sub in ast.walk(tree):
        if not isinstance(sub, ast.Call):
            continue
        dotted = _dotted(sub.func)
        if not dotted:
            continue
        mod, _, fn = dotted.rpartition(".")
        if (mod, fn) in _ABSENCE_PROBES:
            return True
    return False


def _module_level_skips(path: Path) -> list:
    """`pytestmark = ...` assignments whose value probes for a missing tool.

    Module level only: `pytestmark` inside a class scopes to that class, which
    is a per-test gate wearing the same name and is not what this guards.
    """
    tree = ast.parse(path.read_text())
    out = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "pytestmark" not in names:
            continue
        if _probes_for_absence(node.value):
            out.append(node.lineno)
    return out


@pytest.mark.parametrize("path", sorted(TESTS.glob("test_*.py")), ids=lambda p: p.name)
def test_no_test_file_skips_itself_entirely_on_a_missing_tool(path):
    lines = _module_level_skips(path)
    assert not lines, (
        f"{path.name}:{lines} sets `pytestmark` to skip the WHOLE file when a "
        "tool is absent.\n"
        "Every assertion in it then reports as a passing dot, and a skip count "
        "that silently grows is easy to miss.\n"
        "Gate the test that actually needs the tool instead: absent -> FAIL, "
        "and a skip only when someone sets an opt-out deliberately."
    )


# --------------------------------------------------------------------------
# ...and the exemption above has to verify its own justification
# --------------------------------------------------------------------------


def _lock_packages(repo_root: Path) -> set:
    """Every package name `uv.lock` resolves, normalised.

    The lock is the resolved set across ALL extras, which is exactly the
    question being asked: not "is this named in an extra" -- `torch` and
    `triton` are transitive through `stable-baselines3[extra]` and would fail
    that test wrongly -- but "does any install this repo can perform bring it
    in at all". `uv lock --check` already gates the file in CI, so it cannot
    drift from `pyproject.toml` without going red first.
    """
    text = (repo_root / "uv.lock").read_text()
    return {_norm(m) for m in re.findall(r'^name = "([^"]+)"', text, re.M)}


def _norm(name: str) -> str:
    """`stable_baselines3` (the import) and `stable-baselines3` (the package)."""
    return name.strip().lower().replace("_", "-")


def _importorskip_names(path: Path) -> list:
    """`(lineno, package)` for every `pytest.importorskip("pkg")` in a file."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        if _dotted(node.func).rpartition(".")[2] != "importorskip":
            continue
        if node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            out.append((node.lineno, node.args[0].value))
    return out


def _gates_that_can_never_open(path, known, exempt) -> list:
    """The file's `importorskip` gates that `uv.lock` can never open.

    A PACKAGE-SCOPED EXEMPTION, WHERE THE FILE-SCOPED ONE IS TOO BLUNT.
    `exempt` is that file's `_UNRESOLVABLE` value: a plain string exempts
    the whole FILE, which also exempts the next mistyped package name
    someone adds to it. A `(package, reason)` tuple
    exempts that one import and leaves every other gate in the file
    checked.

    This is a FUNCTION so that the test below can drive the real decision
    instead of re-implementing it: a test carrying its own copy of this
    logic could pass while the exemption covered every package in the file
    -- a replica cannot fail the way its original does.
    """
    if isinstance(exempt, str):
        return []
    only = exempt[0] if exempt else None
    out = []
    for lineno, pkg in _importorskip_names(path):
        if only is not None and _norm(pkg) == _norm(only):
            continue
        if _norm(pkg) not in known:
            out.append(f"{path.name}:{lineno} importorskip({pkg!r})")
    return out


#: Files whose `importorskip` names a package `uv.lock` CANNOT resolve, with the reason.
#: An allow-list and not a pattern on purpose: adding one is an edit visible in the diff,
#: whereas a rule like "any file carrying a marker is exempt" would let the next
#: unlockable dependency in as a side effect of adding a marker.
#:
#: Each entry costs real coverage -- the test runs in no configuration CI has -- so the
#: reason must say why the package CANNOT be locked, not merely that it is awkward.
#:
#: A value may be either the reason STRING, which exempts every `importorskip` in that
#: file, or a `(package, reason)` TUPLE, which exempts only that one package and leaves
#: the file's other gates checked. Prefer the tuple: the string form also waves through
#: the next mistyped package name added to the same file, which is the failure this
#: test exists to catch.
_UNRESOLVABLE = {
    "test_humanoid_hand.py":
        "humanoid_bench is not on PyPI (git checkout, installed --no-deps) AND needs "
        "mujoco==3.1.6, while metaworld 3.1.1 pins mujoco==3.3.0 exactly. One `uv.lock` "
        "cannot hold both, so no `--extra` can bring it in without evicting the MT10 "
        "tier; see pyproject's [tool.pytest.ini_options].markers. The exemption costs "
        "less coverage than it might: only the "
        "CONSTRUCTION-dependent tests there are marked `humanoid` and gated behind the "
        "importorskip, so the id derivation, the spec/class shape agreement, the leak "
        "gate and every ground-truth metric checked against its own task's documented "
        "exploit are unmarked, need no simulator, and DO run in a full-deps install. "
        "An exemption here buys the simulator contract only -- step purity, the "
        "terminal, the render, and the spec's observation table against the live model.",
    "test_policy_load_humanoid.py":
        "Loads every runnable humanoidbench-family registry policy in a fresh process: the same package and "
        "the same conflict as test_humanoid_hand.py above (humanoid_bench is a git checkout that needs "
        "mujoco==3.1.6). The per-campaign load tests are marked `humanoid` and gated behind the "
        "importorskip; the catalogue check beside them runs everywhere.",
    "test_ippo_reward_env.py": ("assistax",
        "`assistax` is a GIT-ONLY fork at a pinned commit with no `uv.lock` entry -- "
        "installed only by `scripts/setup_jax.sh` at `ASSISTAX_REF` -- so "
        "no `--extra` brings it in and these gates skip in every CI job. Package-scoped "
        "rather than file-scoped ON PURPOSE: every other `importorskip` in this file "
        "stays checked. IT COSTS LESS THAN ANY ENTRY ABOVE and the split is why this "
        "file's shape is the one to copy: 43 of the 45 tests need neither upstream nor a "
        "device and RUN everywhere -- the reduction's guards against a per-agent dict and "
        "a 0-d object array, the payload validator including both partner counts and "
        "their axes, the pre-step contact-force shift, the producer-to-consumer seed "
        "rows, and the bytecode check that `run` loads no global the module never binds. "
        "Only the 2 that construct the real wrapper are gated. Measured with assistax "
        "absent: 43 passed, 2 skipped."),
    "test_upstream_assistax_tasks.py": ("assistax",
        "The same git-only `assistax` as test_upstream_assistax.py below, for the "
        "per-task contract file that runs every registered upstream_assistax_* id "
        "(scratchitch, feeding, teethbrushing, armmanipulation) through one row "
        "contract. Package-scoped; its offline half (the "
        "class table, the specs, the gate superset, the per-subclass hook attributes) "
        "runs everywhere, the `jax`-marked half only in the tier's own venv."),
    "test_upstream_assistax.py": ("assistax",
        "The same git-only `assistax` at a pinned commit as test_ippo_reward_env.py above: "
        "no `uv.lock` entry, installed only by `scripts/setup_jax.sh` at `ASSISTAX_REF`, "
        "and no `--extra` can bring it in. Package-scoped, so the file's other gates stay "
        "checked. THIS IS THE EXPENSIVE ONE OF THE PAIR, and the fact worth stating "
        "plainly is larger than the exemption: all 8 of these cases are also "
        "`@pytest.mark.jax`, which the default CI job cannot run -- they skip on the "
        "missing package -- so this tier's coverage is entirely execution-gated. 8 of the "
        "16 tests -- HALF THE FILE -- are the half that checks the adapter against the "
        "real upstream env, the only place the multi-agent contract, the homogenised "
        "widths and the row are checked against something this repo did not write. What "
        "runs everywhere is the sim-free half: the task table, the id derivation, the "
        "consumer gate's symbol set and the spec agreement. A green CI run therefore says "
        "NOTHING about whether this adapter still matches upstream; the evidence for that "
        "is a run by hand in the tier's venv. Measured with assistax absent: "
        "8 passed, 8 skipped."),
}


def test_every_importorskip_names_a_package_something_installs(repo_root):
    """A gate on a package no install brings in can never open.

    `importorskip` is the exemption this file grants, and it is only safe while
    the package is one SOME configuration installs -- then an `--extra all`
    install runs the test. On a package
    in no extra and reachable transitively from none, the test skips in every
    configuration that exists and its subject is covered by nothing, anywhere,
    for ever. That is not a gated test; it is a deleted one that still prints a
    dot.

    Checked against `uv.lock` rather than against `pyproject.toml`'s extras
    because the honest question is whether anything installs it: `torch` and
    `triton` are named in no extra and arrive through `stable-baselines3[extra]`,
    so a pyproject-only rule would fail both for being correct.

    NOTE this is a weaker claim than "some job runs it", deliberately: CI
    installs only `--extra test`, so a package that IS in an extra is proven
    installable here, not proven exercised. This test closes the part that no
    CI configuration can fix, because no configuration installs a package
    nothing declares.
    """
    known = _lock_packages(repo_root)
    missing = []
    for path in sorted(TESTS.glob("test_*.py")):
        missing += _gates_that_can_never_open(
            path, known, _UNRESOLVABLE.get(path.name))
    assert not missing, (
        "these gates can never open -- `uv.lock` resolves no such package, so no "
        "extra installs it in any configuration:\n  "
        + "\n  ".join(missing)
        + "\n\nThe test is skipped everywhere and whatever it covers is covered by "
          "nothing. Declare the dependency in `pyproject.toml` (the `test` extra if "
          "it is pure Python and cheap, its own extra if it is not), re-run `uv lock`, "
          "and the gate becomes a gate.\n\n"
          "If the package genuinely CANNOT be locked -- not on PyPI, or its pins "
          "conflict with another extra's -- add the file to `_UNRESOLVABLE` above with "
          "the reason. That is the exception, and it costs exactly the coverage this "
          "test exists to protect, so give the reason rather than the fact."
    )


def test_a_package_scoped_exemption_leaves_the_files_other_gates_checked(tmp_path):
    """The narrow form must actually be narrow, or it is the blunt one renamed.

    A string entry exempts a whole FILE, which also waves through the next
    mistyped package name added to it, and the assistax pair would inherit
    it.

    THIS DRIVES THE REAL DECISION, `_gates_that_can_never_open`. A test that
    re-implemented the loop could pass while the exemption covered every
    package in the file: a replica cannot fail the way its original does.
    """
    assert isinstance(_UNRESOLVABLE["test_upstream_assistax.py"], tuple), (
        "the assistax entries are package-scoped; a string here re-blunts the check")
    assert isinstance(_UNRESOLVABLE["test_ippo_reward_env.py"], tuple)

    src = tmp_path / "test_fixture.py"
    src.write_text(
        "import pytest\n"
        "def test_a():\n    pytest.importorskip('assistax')\n"
        "def test_b():\n    pytest.importorskip('assitsax')\n"
        "def test_c():\n    pytest.importorskip('numpy')\n"
        "def test_d():\n    pytest.importorskip('not_a_real_package')\n")
    known = {"numpy"}

    got = _gates_that_can_never_open(src, known, ("assistax", "reason"))
    assert [g.rpartition("importorskip(")[2] for g in got] == [
        "'assitsax')", "'not_a_real_package')"], got

    # the blunt form, for contrast: it hides both
    assert _gates_that_can_never_open(src, known, "a reason") == []
    # and with no exemption at all, even the named package is reported
    assert len(_gates_that_can_never_open(src, known, None)) == 3


def test_an_unimplemented_backend_reason_names_every_contract_it_marks():
    """A strict-xfail reason must name each contract family it now covers.

    One entry in
    `_BACKEND_UNIMPLEMENTED` carries the reason for EVERY case marked with
    that backend, so extending the mark to a new contract without extending
    the prose hands the reader of a training-curve xfail an explanation about
    pruners -- a message that is about something else, which is worse than no
    message because it reads as authoritative.

    The check is derived, not a literal list: it reads which test FILES the
    marked cases live in and requires the reason to name each one's contract
    family. Adding a fifth family to the mark therefore fails here until the
    reason is extended, which is the whole point.
    """
    import re

    import conftest

    # the contract family each parametrizing file speaks for, by the word its
    # reason must contain; keyed on the file so a new catalogue is visible.
    # DISTINCTIVE PHRASES, NOT BARE WORDS. A bare "curve" is satisfied
    # incidentally by `train_curve`, so a reason that never mentioned the
    # contract would still pass. Each value is a phrase that can only be there
    # on purpose.
    FAMILY_WORD = {
        "test_pruning.py": "pruner is consulted",
        "test_checkpoint_selection.py": "per-checkpoint restore",
        "test_training_curve_reward.py": "training-curve",
    }
    # BY AST, NOT BY SUBSTRING. A substring match would match this file, which
    # only mentions the name in prose -- a probe that fails on itself is a
    # probe defect, not a finding.
    def _calls_it(path):
        import ast as _ast
        try:
            tree = _ast.parse(path.read_text())
        except SyntaxError:
            return False
        return any(isinstance(n, _ast.Call)
                   and getattr(n.func, "id", None) == "backend_param_unimplemented"
                   for n in _ast.walk(tree))

    marked = sorted(p.name for p in TESTS.glob("test_*.py") if _calls_it(p))
    assert marked, "nothing uses the strict-xfail variant; this check is vacuous"

    reason = conftest._BACKEND_UNIMPLEMENTED["assistax_ppo"].lower()
    missing = [f for f in marked
               if f in FAMILY_WORD and FAMILY_WORD[f] not in reason]
    assert not missing, (
        f"the assistax_ppo xfail reason does not name the contract of {missing}. "
        f"One reason serves every marked case, so a file marked without its "
        f"contract named gives that file's reader the wrong explanation.")

    unmapped = [f for f in marked if f not in FAMILY_WORD]
    assert not unmapped, (
        f"{unmapped} uses backend_param_unimplemented but names no contract "
        f"family here, so nothing checks that the reason covers it -- add it "
        f"to FAMILY_WORD with the word its reason must contain")
