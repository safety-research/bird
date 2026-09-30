"""What bird does when it is NOT running from a checkout.

WHY NO OTHER TEST CAN SEE THIS. Every other test in this suite runs from the
checkout, where `Path(__file__).parent.parent` is the repo root and every
data-root site is correct. Under a non-editable install that expression is
`site-packages` -- a directory that exists, belongs to other projects, and is
not ours -- so a resolution built on it returns a plausible wrong path with no
error, and no amount of testing from a checkout can notice.

THE PROPERTY UNDER TEST IS NOT "IT WORKS". It is **never a plausible wrong
path**: from outside a checkout, a caller either gets the real directory or
an exception that names what is missing and how to supply it.

WHAT SHIPS AND WHAT DOES NOT: `configs/` is small and is package data;
`tasks/` and `policies/` are data trees that grow with every task (per-seed
records) and can reach hundreds of MB, so they are NOT shipped. On a
machine with no checkout they come from `$BIRD_DATA_ROOT`.

A FIXTURE THAT WRITES TO DISK IS SUBJECT TO EVERY REPO-WIDE INVARIANT: running
`uv build` in the checkout leaves `build/lib/bird/policies.py` behind --
gitignored, so `git status` stays clean -- and any test that walks the whole
tree for a pattern fails on the second copy. The wheel is therefore built from
`git archive HEAD` in a temp directory, and the fixture asserts the checkout is
unpolluted afterwards.

THIS FILE BUILDS A WHEEL AND INSTALLS IT, which is slow (tens of seconds) and
needs `uv`. Marked slow and skipped when `uv` is absent, because the
alternative -- faking an install by manipulating `sys.path` -- would test the
fake. The whole defect was that the real layout differs from the one every
test assumed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: Slow only. THE `uv` GATE IS NOT A FILE-LEVEL SKIPIF, deliberately:
#: `tests/test_no_silent_skip.py` refuses a module-level
#: marker whose condition is a missing tool, because every assertion in the
#: file then reports as a passing dot while running none of them -- and a
#: count guard cannot see it, since a skipped test leaves the count rather
#: than shrinking it. The absence is a FAILURE unless someone opts in; see
#: `_uv_or_fail`.
pytestmark = [pytest.mark.slow]


def _uv_or_fail() -> str:
    """`uv`, or a FAILURE -- never a silent skip. Returns its path.

    This file is the ONLY thing in the suite that sees the installed layout; a quiet
    skip here restores exactly the blindness it exists to remove, and it
    would do so on the box where somebody is most likely to be debugging a
    packaging problem.

    The skip stays available because it is legitimate -- bird itself needs
    no `uv` -- but it has to be asked for.
    """
    found = shutil.which("uv")
    if found:
        return found
    if os.environ.get("BIRD_ALLOW_NO_UV"):
        pytest.skip("no uv on PATH, and BIRD_ALLOW_NO_UV says that is fine")
    pytest.fail(
        "uv is not on PATH, so no wheel was built and nothing in this file "
        "proved anything about a non-editable install -- the one layout no "
        "other test in this suite can see.\n"
        "Install uv, or set BIRD_ALLOW_NO_UV=1 to accept the gap "
        "deliberately.")


@pytest.fixture(scope="module")
def installed(tmp_path_factory):
    """A wheel built from this tree, installed into a throwaway venv.

    Module-scoped: the build and install cost dominates, and every case here
    asks a different question of the same artifact.
    """
    _uv_or_fail()

    # THIS FILE TESTS `HEAD`, NOT THE WORKING TREE, because the wheel is
    # built from `git archive HEAD` (see below). An uncommitted change to a
    # packaged file is therefore invisible here, and a green run would be
    # answering a question nobody asked. Refusing a dirty tree is right for a
    # deploy and hostile in a test, so this skips instead -- with the reason
    # named, not as a bare "unavailable".
    #
    # It cannot fire on a clean clone: `build/` and `*.egg-info/` are
    # gitignored, so `git status --porcelain` stays empty even with a previous
    # failure's pollution present.
    # SCOPED TO WHAT CAN CHANGE THE WHEEL. An unscoped porcelain check fires
    # on any dirt, and working trees are routinely dirty -- scratch files,
    # local edits -- so it would skip almost always locally and quietly become
    # a CI-only test. Scoped, it fires exactly when someone is iterating on
    # packaging, which is where a pass would mislead.
    #
    # THE MEMBERSHIP CRITERION IS "can change THIS TEST'S OUTCOME", not "can
    # change the wheel", and the difference is load-bearing in one
    # direction. `pyproject.toml` packages `README.md` into the wheel's
    # METADATA, so a README edit changes the wheel's bytes -- and cannot
    # change whether `configs/` resolves from an installed package, which is
    # what these cases ask. It is correctly absent. Under the looser reading
    # someone adds it and the skip starts firing on prose edits, which is how
    # a scoped guard drifts back into the unscoped one it replaced.
    #
    # `configs` is in the list and is NOT obvious: `bird/_data/configs` is a
    # SYMLINK to it, and the package-data glob resolves through the symlink
    # at build time -- the wheel carries every YAML file that lives in
    # `configs/`. Editing one changes the wheel without touching anything
    # under `bird/`.
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--",
         "pyproject.toml", "bird", "configs"],
        cwd=REPO, capture_output=True, text=True, timeout=120)
    if dirty.returncode == 0 and dirty.stdout.strip():
        pytest.skip(
            f"uncommitted changes to files the wheel is built from "
            f"({', '.join(sorted({l[3:].split('/')[0] for l in dirty.stdout.splitlines() if l[3:]}))}): "
            f"this file builds from `git archive HEAD`, so it would test the "
            f"committed bytes rather than the ones you are editing. Commit "
            f"first. (Dirt elsewhere in the tree is ignored -- only "
            f"pyproject.toml, bird/ and configs/ change the wheel.)")

    before = _artefact_state()
    work = tmp_path_factory.mktemp("wheel")

    # BUILD FROM `git archive HEAD`, NEVER IN THE CHECKOUT.
    # `uv build` with `cwd=REPO` makes setuptools write `build/lib` and
    # `bird.egg-info` into the working tree. Both are gitignored, so
    # `git status` stays clean and nothing looks wrong -- but the tree then
    # holds a SECOND complete copy of the package, and anything that walks
    # it finds it: a test that walks the whole tree for a pattern would
    # match `build/lib/bird/policies.py` and fail the suite. Adding "build"
    # to one skip list would hide it from one walker and leave it for the
    # next.
    #
    # The side benefit is the better reason to keep it: the wheel is a
    # function of COMMITTED bytes, which is what actually ships. The cost is
    # that an uncommitted fix to any packaged file does not appear here
    # until it is committed -- worth knowing when this file fails and the
    # working tree looks right.
    src = work / "src"
    src.mkdir()
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                            cwd=REPO, capture_output=True, timeout=60)
    if inside.returncode != 0:
        pytest.skip("the wheel is built from `git archive HEAD`; this copy is not a git checkout")
    archive = subprocess.run(["git", "archive", "--format=tar", "HEAD"],
                             cwd=REPO, capture_output=True, timeout=900)
    assert archive.returncode == 0, (
        f"git archive failed:\n{archive.stderr.decode()[-2000:]}")
    untar = subprocess.run(["tar", "xf", "-", "-C", str(src)],
                           input=archive.stdout, capture_output=True, timeout=900)
    assert untar.returncode == 0, untar.stderr.decode()[-2000:]

    out = work / "dist"
    build = subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out)],
                           cwd=src, capture_output=True, text=True, timeout=900)
    assert build.returncode == 0, f"wheel build failed:\n{build.stderr[-2000:]}"

    # UNCHANGED, not absent. `bird.egg-info/` exists in EVERY synced
    # checkout -- `uv sync` installs bird editable through
    # setuptools.build_meta, which writes it into the project root, and CI
    # runs `uv sync --frozen --extra test` before pytest. An absence
    # assertion would therefore fail on every developer checkout and every
    # CI run: it would assert that nobody had ever installed the package
    # they were testing.
    #
    # So the claim is the one actually wanted -- THIS BUILD added nothing
    # and touched nothing. `build/` still has to be absent if it was
    # absent, which the same comparison gives for free.
    after = _artefact_state()
    for name in _ARTEFACTS:
        assert after[name] == before[name], (
            f"the wheel build changed {name}/ in the checkout"
            f"{' (it did not exist before)' if before[name] is None else ''}."
            f" Both artefacts are gitignored, so git status will not show "
            f"it, and the next tree walk finds a second copy of the package "
            f"-- and fails any test that walks the tree.")
    wheels = sorted(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {[w.name for w in wheels]}"

    venv = work / "venv"
    mk = subprocess.run(["uv", "venv", str(venv)], capture_output=True,
                        text=True, timeout=300)
    assert mk.returncode == 0, mk.stderr
    python = venv / "bin" / "python"
    inst = subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheels[0])],
        capture_output=True, text=True, timeout=900)
    assert inst.returncode == 0, f"install failed:\n{inst.stderr[-2000:]}"

    # OUTSIDE THE CHECKOUT. Running from `REPO` would find the repo's own
    # `tasks/` by relative path and pass for the wrong reason.
    elsewhere = work / "elsewhere"
    elsewhere.mkdir()
    return python, elsewhere, wheels[0]


#: The two things a setuptools build writes into a project root.
_ARTEFACTS = ("build", "bird.egg-info")


def _artefact_state():
    """A fingerprint of each artefact: None if absent, else its contents.

    Files and mtimes rather than the directory's own mtime, because
    setuptools rewriting a file in place leaves the directory's mtime alone
    -- and "the build rewrote bird.egg-info" is exactly the change worth
    catching in a checkout somebody else is also using.
    """
    state = {}
    for name in _ARTEFACTS:
        root = REPO / name
        if not root.exists():
            state[name] = None
            continue
        state[name] = sorted(
            (str(f.relative_to(root)), f.stat().st_mtime_ns, f.stat().st_size)
            for f in root.rglob("*") if f.is_file())
    return state


def _run(installed, code, env=None):
    python, cwd, _ = installed
    e = dict(os.environ)
    e.pop("BIRD_DATA_ROOT", None)
    e.update(env or {})
    return subprocess.run([str(python), "-c", code], cwd=cwd, env=e,
                          capture_output=True, text=True, timeout=300)


def test_the_wheel_does_not_ship_the_two_large_directories(installed):
    """The size decision, asserted so it cannot be undone by accident.

    A later `include` that swept in `tasks/` would add the whole data tree to
    every install and nothing else would complain.
    """
    import zipfile

    _, _, wheel = installed
    names = zipfile.ZipFile(wheel).namelist()
    assert not [n for n in names if n.startswith(("tasks/", "policies/"))], (
        "the wheel now ships tasks/ or policies/, which are data trees, not package data")
    # 40 MB, against ~19 MB measured (code and configs plus in-package
    # simulator assets, compressed). The ceiling is here to catch a data tree
    # arriving, not to police drift, so it is set well clear of the current
    # figure rather than just above it.
    assert wheel.stat().st_size < 40 * 1024 * 1024, (
        f"wheel is {wheel.stat().st_size / 1e6:.0f} MB, which means something "
        f"large started shipping")


def test_configs_resolve_from_the_installed_package(installed):
    """`configs/` IS shipped, so this must work with no override at all."""
    out = _run(installed, "import bird.paths as P; print(P.data_dir('configs'))")
    assert out.returncode == 0, out.stderr
    assert "site-packages" in out.stdout, out.stdout
    assert out.stdout.strip().endswith("bird/_data/configs"), out.stdout


@pytest.mark.parametrize("name", ["tasks", "policies"])
def test_the_unshipped_directories_raise_a_NAMED_error(installed, name):
    """The heart of it: an exception, not a path into site-packages.

    The message has to carry the override's name, because the person reading
    it is on a compute node with no checkout and no way to guess.
    """
    out = _run(installed, f"""
import bird.paths as P
try:
    p = P.data_dir({name!r})
except P.DataRootMissing as e:
    print("RAISED", e)
else:
    print("RETURNED", p)
""")
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("RAISED"), (
        f"data_dir({name!r}) returned a path from a non-editable install "
        f"instead of raising: {out.stdout.strip()}")
    assert "BIRD_DATA_ROOT" in out.stdout, (
        "the error does not name the override, so it tells a reader what is "
        "wrong and not what to do")


def test_a_task_load_fails_loudly_rather_than_finding_nothing(installed):
    """"No task catalogue" and "an empty task catalogue" are different facts.

    A resolution to a non-existent directory inside site-packages would make a
    glob over it yield zero specs, so a caller could read that as "this build
    has no tasks" rather than "this build cannot see the tasks".
    """
    out = _run(installed, """
from bird import tasks
try:
    tasks.load("h1hand_basketball")
except Exception as e:
    print(type(e).__name__, " ".join(str(e).split()))
""")
    assert out.returncode == 0, out.stderr
    assert "TaskSpecError" in out.stdout
    assert "BIRD_DATA_ROOT" in out.stdout, out.stdout


def test_the_override_makes_an_installed_package_work(installed):
    """The supported mechanism, end to end, from outside the checkout.

    No copying of data directories into site-packages is needed: point the
    override at a data tree instead.
    """
    out = _run(installed, """
from bird import tasks
print("LOADED", tasks.load("h1hand_basketball").id)
""", env={"BIRD_DATA_ROOT": str(REPO)})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "LOADED h1hand_basketball", out.stdout


def test_an_override_pointing_nowhere_does_not_win_over_a_real_directory(installed):
    """A wrong override must not shadow package data that is actually there.

    `configs/` ships, so it resolves whatever the override says; an override
    that silently took precedence and then failed would be the same class of
    defect this whole file is about.
    """
    out = _run(installed, "import bird.paths as P; print(P.data_dir('configs'))",
               env={"BIRD_DATA_ROOT": "/nonexistent-bird-data-root"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("bird/_data/configs"), out.stdout


def test_repo_root_refuses_site_packages(installed):
    """The marker-file check, which is what makes the refusal possible.

    Existence could never have caught this: site-packages exists. Only
    `pyproject.toml` + `bird.py` distinguishes a checkout, and
    `wandb.log_code` is the caller that most needs the distinction -- it
    would otherwise upload site-packages to an external service.
    """
    out = _run(installed, """
import bird.paths as P
print("NONE" if P.repo_root(required=False) is None else "SOMETHING")
try:
    P.repo_root()
except P.DataRootMissing as e:
    print("RAISED", "BIRD_DATA_ROOT" in str(e))
""")
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[0] == "NONE", out.stdout
    assert "RAISED True" in out.stdout, out.stdout


def test_a_data_root_that_is_not_a_checkout(installed, tmp_path):
    """`$BIRD_DATA_ROOT` pointing at a data tree that is not a checkout, pinned.

    A data tree carries `tasks/` (and `configs/`, `policies/`) but not both
    checkout markers, so it is not a checkout by `repo_root`'s marker rule.
    Both halves of that have to be true at once and they are easy to get
    wrong together:

    * `data_dir` must resolve from the override on EXISTENCE alone, or a run
      cannot load a task spec;
    * `repo_root` must still say None, or `wandb.log_code` would upload the
      data tree -- the site-packages upload, moved rather than prevented.
    """
    python, cwd, _ = installed
    staged = tmp_path / "staged"
    (staged / "tasks").mkdir(parents=True)
    spec_dir = staged / "tasks" / "toy_reach"
    spec_dir.mkdir()
    (spec_dir / "shared_spec.yaml").write_text("id: toy_reach\n")
    # It carries **pyproject.toml** and NOT bird.py, and that pairing is the
    # whole reason `repo_root` takes two markers: a one-marker rule keyed on
    # pyproject.toml would call this tree a checkout and `wandb.log_code`
    # would upload it -- so the fixture has to carry pyproject.toml or this
    # test passes for a weaker reason than the case it stands for.
    (staged / "pyproject.toml").write_text("[project]\nname = \"bird\"\n")
    assert not (staged / "bird.py").exists(), "premise: not a checkout"

    out = _run(installed, """
import bird.paths as P
print("ROOT", P.repo_root(required=False))
print("TASKS", P.data_dir("tasks"))
""", env={"BIRD_DATA_ROOT": str(staged)})
    assert out.returncode == 0, out.stderr
    lines = dict(l.split(" ", 1) for l in out.stdout.strip().splitlines())
    assert lines["ROOT"] == "None", (
        f"a data tree was accepted as a checkout ({lines['ROOT']}); "
        f"wandb.log_code would upload it")
    assert lines["TASKS"] == str(staged / "tasks"), (
        f"the override did not resolve on a non-checkout data tree, so a "
        f"run cannot load a task spec: {lines['TASKS']}")


def test_a_real_config_LOADS_from_the_installed_package(installed):
    """A directory check passes on an empty directory.

    `test_configs_resolve_from_the_installed_package` proves the path
    resolves; it would pass just as well if `package-data` shipped the
    directory and none of the YAML in it, or if the symlink resolved to
    nothing. This loads a real config THROUGH the installed package.

    IT NEEDS THE OVERRIDE, AND THAT IS A FINDING RATHER THAN A CONCESSION.
    Shipping `configs/` is not sufficient to `load()` a config: validation
    resolves the config's env through `tasks/`, which is deliberately not
    shipped. So an installed package can READ its configs (the case below)
    but cannot fully load one without the data root -- which is worth
    knowing before someone concludes the wheel is self-sufficient because
    `configs/` is in it.

    LINUX ONLY, and stated rather than discovered: `bird/_data/configs` is a
    symlink into the repo, which is how the directory keeps one canonical
    location. Windows would need a copy at build time.
    """
    out = _run(installed, """
from bird.config import load
c = load("eureka")
print("LOADED", c.get("name"), len(c.hash()))
""", env={"BIRD_DATA_ROOT": str(REPO)})
    assert out.returncode == 0, (
        f"a real config would not load from the installed package:\n"
        f"{out.stderr[-1500:]}")
    _, name, digest = out.stdout.split()
    assert name == "eureka", out.stdout
    # 12, not 64: `Config.hash()` is the sha256 PREFIX used as the run id.
    # Asserted on the length so an empty or default-only config -- which is
    # what a broken configs/ would produce -- cannot pass this.
    assert int(digest) == 12, f"unexpected config hash length: {digest}"


def test_the_shipped_config_yaml_is_really_there_not_just_the_directory(installed):
    """The half that needs NO override: are the bytes in the wheel?

    This is the emptiness check proper. `_default.yaml` is the one file the
    whole schema is read from, so if package-data shipped a directory and no
    contents, this fails and the fuller test above cannot tell you why --
    its failure would look like a missing `tasks/`.
    """
    out = _run(installed, """
import yaml, bird.paths as P
root = P.data_dir("configs")
doc = yaml.safe_load((root / "_default.yaml").read_text())
print("KEYS", len(doc), "HAS_TRAIN", "train" in doc)
""")
    assert out.returncode == 0, out.stderr
    keys = int(out.stdout.split()[1])
    assert keys > 10, f"_default.yaml parsed to {keys} top-level keys"
    assert "HAS_TRAIN True" in out.stdout, out.stdout


def test_the_profiles_subdirectory_ships_too(installed):
    """The glob's depth, which a single top-level file would not exercise."""
    out = _run(installed, """
import bird.paths as P
root = P.data_dir("configs")
prof = sorted(p.name for p in (root / "_profiles").glob("*.yaml"))
print("PROFILES", ",".join(prof))
""")
    assert out.returncode == 0, out.stderr
    assert "tester" in out.stdout and "full" in out.stdout, out.stdout


def test_using_root_redirects_and_never_writes_the_public_name():
    """The redirect API, and the freeze it exists to avoid.

    Writing `tasks.TASKS_ROOT` is the obvious way to redirect the catalogue
    and it is a trap. The name is served by `__getattr__`, so:

    * `monkeypatch.setattr` reads the old value in its SAVE step, which runs
      the resolver and raises where the root does not resolve;
    * `undo()` does not delete the name -- it writes the computed value back
      as a real `__dict__` entry. The first caller to patch the public name
      freezes it into a snapshot for the rest of the process, taken once,
      early, possibly before an override was set.

    The dict-entry assertion is the half that sees the freeze. A test that
    only checked the VALUE would pass against a frozen snapshot that
    happened to be right, which is this module's whole subject.

    Nesting and the restore-then-invalidate order are asserted because
    neither shows in a green run: a clear-on-exit breaks nesting silently,
    and a rebuild-on-exit that raised would leave the index populated from
    the temporary root behind a correct-looking override.
    """
    from bird import tasks as tasks_mod

    real = tasks_mod._tasks_root()
    with tasks_mod.using_root("/nonexistent/outer") as outer:
        assert tasks_mod._tasks_root() == outer
        assert tasks_mod._INDEX is None, "entry must invalidate the index"
        with tasks_mod.using_root("/nonexistent/inner"):
            assert tasks_mod._tasks_root() == Path("/nonexistent/inner")
        assert tasks_mod._tasks_root() == outer, (
            "exiting a nested block cleared the override instead of "
            "restoring the previous one")
    assert tasks_mod._tasks_root() == real
    assert tasks_mod._INDEX is None, "exit must invalidate, not rebuild"

    assert "TASKS_ROOT" not in tasks_mod.__dict__, (
        "using_root wrote the public name into the module dict; from here "
        "on __getattr__ is dead and every reader sees a frozen snapshot")


def test_the_public_name_is_never_frozen_into_the_module_dict():
    """After whatever else ran in this process.

    A writer site is an attribute assignment to `tasks.TASKS_ROOT` or a
    `monkeypatch.setattr` of it (a grep for one spelling misses the other).
    This is the standing check that none appears: one `TASKS_ROOT` entry
    anywhere in the process is permanent.
    """
    from bird import tasks as tasks_mod

    assert "TASKS_ROOT" not in tasks_mod.__dict__, (
        "the public name is frozen in the module dict; "
        "__getattr__ will never run again in this process")


# ---------------------------------------------------------------------------
# The lazy names' visibility and precedence.
# ---------------------------------------------------------------------------


def test_the_lazy_root_is_visible_to_dir():
    """PEP 562's `__getattr__` runs only after the module dict MISSES.

    So `dir(module)`, which reads the dict, omits the name entirely -- and so
    do tab-completion, `inspect.getmembers`, and anything that enumerates a
    module's surface to decide what it exports. The name works and reads as
    absent, which is this file's own failure shape one level up: a
    plausible wrong answer rather than an error.
    """
    from bird import config as config_mod
    from bird import tasks as tasks_mod

    assert "CONFIG_ROOT" in dir(config_mod)
    assert "TASKS_ROOT" in dir(tasks_mod)
    # `__dir__` must ADD to the dict rather than replace it, or every other
    # public name in the module disappears from introspection instead.
    assert "load" in dir(config_mod) and "validate" in dir(config_mod)


def test_an_unknown_name_is_still_an_AttributeError():
    """The pair. A `__getattr__` that answered everything would pass the test
    above and break `hasattr` for every name the module does not have."""
    from bird import config as config_mod
    from bird import tasks as tasks_mod

    for mod in (config_mod, tasks_mod):
        with pytest.raises(AttributeError):
            getattr(mod, "NO_SUCH_NAME_HERE")
        assert not hasattr(mod, "NO_SUCH_NAME_HERE")


def test_hasattr_propagates_when_the_root_cannot_resolve(monkeypatch):
    """`hasattr` is NOT a safe probe for these names, and that is deliberate.

    `hasattr` swallows only `AttributeError`. The resolvers raise
    `TaskSpecError` / `DataRootMissing` when the directory genuinely cannot
    be found, so on the one machine where the question matters -- a
    non-editable install with no `$BIRD_DATA_ROOT` -- `hasattr` RAISES rather
    than returning False.

    Asserted rather than left as a comment because it is the opposite of what
    `hasattr` is usually reached for, and a caller who guards with it would
    convert a clear "set BIRD_DATA_ROOT" error into a crash at an unrelated
    line. The non-raising question is
    `paths.data_dir(name, required=False) is not None`.
    """
    from bird import paths
    from bird import tasks as tasks_mod

    monkeypatch.setattr(paths, "data_dir", lambda *a, **k: None)
    monkeypatch.setattr(tasks_mod, "_ROOT_OVERRIDE", None, raising=False)

    with pytest.raises(tasks_mod.TaskSpecError):
        hasattr(tasks_mod, "TASKS_ROOT")

    assert paths.data_dir("tasks", required=False) is None, (
        "the documented non-raising probe must actually not raise")


def test_using_root_outranks_whatever_the_resolver_would_return(monkeypatch, tmp_path):
    """Precedence, asserted because it is load-bearing and only implicit.

    `_tasks_root()` consults `_ROOT_OVERRIDE` before it asks
    `paths.data_dir`, so a `using_root` block wins over every source the
    resolver knows -- the checkout, `$BIRD_DATA_ROOT`, the shipped copy.
    That is the order a reader wants: the resolver says where this MACHINE
    keeps the data, the block says which catalogue this PIECE OF CODE is
    about, and a test that set up a temporary catalogue would otherwise read
    the operator's real one.

    The resolver is stubbed rather than driven through the environment:
    setting `$BIRD_DATA_ROOT` and asserting the bare call returned it would
    fail in a checkout, because `data_dir` prefers the directory beside the
    package and never reaches the variable. That is correct behaviour and a DIFFERENT
    precedence question -- pinned separately below. Stubbing keeps this case
    about the one ordering `_tasks_root` actually decides.
    """
    from bird import paths
    from bird import tasks as tasks_mod

    resolver_root = tmp_path / "from_resolver"
    resolver_root.mkdir()
    block_root = tmp_path / "from_block"
    block_root.mkdir()

    monkeypatch.setattr(paths, "data_dir", lambda *a, **k: resolver_root)
    assert tasks_mod._tasks_root() == resolver_root, (
        "precondition: with no block active the resolver is what answers")

    with tasks_mod.using_root(block_root):
        assert tasks_mod._tasks_root() == block_root, (
            "the resolver outranked using_root; a temporary catalogue would "
            "be ignored wherever the machine has a real one")

    assert tasks_mod._tasks_root() == resolver_root, (
        "the block did not restore the previous state")


def test_a_checkout_beside_the_package_outranks_the_environment_override(monkeypatch, tmp_path):
    """The resolver's own order, and the half that is easy to get wrong.

    `data_dir` prefers the directory beside the package whenever that
    directory is a CHECKOUT, and only then consults `$BIRD_DATA_ROOT`. So on
    a developer box the variable is inert -- setting it does not redirect
    anything, which is worth knowing before debugging why it "did not take".
    The override exists for the machine with no checkout, which is exactly
    the case it is documented for.
    """
    from bird import paths

    real = paths.data_dir("tasks")
    decoy = tmp_path / "decoy"
    (decoy / "tasks").mkdir(parents=True)

    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(decoy))
    assert paths.data_dir("tasks") == real, (
        "$BIRD_DATA_ROOT overrode a real checkout; the variable is for "
        "machines that HAVE no checkout, and honouring it here would let a "
        "stale variable silently redirect a developer's whole catalogue")

    # And the pair: with no checkout beside the package, the variable IS
    # what answers -- otherwise the test above would pass for a resolver
    # that ignored the variable entirely.
    monkeypatch.setattr(paths, "_is_checkout", lambda p: False)
    assert paths.data_dir("tasks") == decoy / "tasks"


# ---------------------------------------------------------------------------
# `data_root_source`: which candidate answered, recorded rather than guessed.
# ---------------------------------------------------------------------------


def test_the_source_label_matches_the_path_for_each_candidate(monkeypatch, tmp_path):
    """All three branches, because a label is only worth having if it can be
    wrong. A resolver that returned `SOURCE_CHECKOUT` unconditionally would
    pass any single-branch test and be useless exactly when the sources stop
    agreeing, which is the day the field exists for.
    """
    from bird import paths

    # 1. checkout beside the package -- what a developer box does
    path, source = paths.data_dir_with_source("tasks")
    assert source == paths.SOURCE_CHECKOUT
    assert path == paths.data_dir("tasks")

    # 2. no checkout, variable set -- what a cluster worker does
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "tasks").mkdir(parents=True)
    monkeypatch.setattr(paths, "_is_checkout", lambda p: False)
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(elsewhere))
    assert paths.data_dir_with_source("tasks") == (elsewhere / "tasks",
                                                   paths.SOURCE_ENV)

    # 3. no checkout, no variable -- only the shipped copy is left, and only
    #    `configs` is shipped
    monkeypatch.delenv(paths.DATA_ROOT_ENV)
    assert paths.data_dir_with_source("configs")[1] == paths.SOURCE_SHIPPED
    assert paths.data_dir_with_source("tasks", required=False) == (None, None)


def test_data_dir_and_the_source_form_share_one_resolution_order(monkeypatch, tmp_path):
    """`data_dir` must be a WRAPPER, not a second copy of the order.

    Two copies is how a provenance field comes to describe a lookup nobody
    performed -- confidently, and only wrong in the configurations nobody
    tests.

    THE CONFIGURATION MATTERS. Stubbing `_is_checkout` to False and setting
    the variable is a configuration where every plausible ordering agrees, so
    such a test passes against a deliberately re-ordered `data_dir`. A test
    that cannot fail on the defect it names is decoration. The orders only
    diverge where BOTH a checkout and the variable are available, so that is
    what this sets up.
    """
    from bird import paths

    decoy = tmp_path / "decoy"
    (decoy / "policies").mkdir(parents=True)
    monkeypatch.setenv(paths.DATA_ROOT_ENV, str(decoy))
    # `_is_checkout` left alone: this repo IS a checkout, so both candidates
    # are live and an order that preferred the variable would show here.

    path_only = paths.data_dir("policies")
    path_with, source = paths.data_dir_with_source("policies")
    assert path_only == path_with, (
        "the two entry points disagree about where `policies/` is, so one of "
        "them is a second copy of the resolution order")
    assert source == paths.SOURCE_CHECKOUT
    assert path_only != decoy / "policies"


def test_the_seed_row_helper_never_raises(monkeypatch):
    """A provenance field must not fail a training that already finished.

    The value is read at artifact-writing time, after the GPU work is done;
    a lookup that raised there would throw away a completed run to report on
    its own bookkeeping.
    """
    from bird import paths
    from bird.components import fasttd3

    def _boom(*a, **k):
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(paths, "data_dir_with_source", _boom)
    assert fasttd3._data_root_source() is None

