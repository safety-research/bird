import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pytest  # noqa: E402

from bird.config import CONFIG_ROOT  # noqa: E402


def _choose_gl_backend_like_an_adapter() -> None:
    """Do, once and before any test module is collected, what every MuJoCo
    adapter's `__init__` does before it imports its simulator.

    Each MuJoCo adapter (`bird/envs/{metaworld,assistax,gym_mujoco,
    humanoid_hand}.py`) fills an unset `MUJOCO_GL` and preloads Triton
    STRICTLY BEFORE importing `mujoco`: MuJoCo binds its GL backend when it is
    first imported, and Triton's LLVM must be mapped before the software GL
    stack's (`bird/envs/metaworld.py::_preload_llvm_before_mujoco`). A real run
    constructs one adapter, so that order always holds.

    A pytest process constructs dozens, and a test that imports `mujoco` or
    `metaworld` before the first adapter has run -- an `importorskip` ahead of a
    registry lookup, a module-level import at collection -- binds the backend
    while `MUJOCO_GL` is still unset. Every later `mujoco.Renderer` in the
    process then fails with `gladLoadGL error`, and WHICH tests fail depends
    only on which files happened to be collected or run first. Measured on the
    release tree: 10 render tests red in one whole-suite run, each green alone.

    So the session establishes the adapters' order itself, with the adapters'
    own two helpers, before anything can import a simulator. An operator's
    `MUJOCO_GL` still wins, exactly as it does in every adapter: this only fills
    an unset or empty value. Probes started through `run_probe` below still get
    a child environment WITHOUT `MUJOCO_GL`, so the adapters' own
    fill-when-unset path stays under test there.
    """
    if os.environ.get("MUJOCO_GL"):
        return
    from bird.envs.mujoco_control import _default_mujoco_gl, _preload_llvm_before_mujoco
    _preload_llvm_before_mujoco()
    chosen = _default_mujoco_gl()
    if chosen:
        os.environ["MUJOCO_GL"] = chosen


_choose_gl_backend_like_an_adapter()


def run_probe(source: str, timeout: int = 300) -> dict:
    """Run a `python -c` probe in a clean subprocess and return its PROBE payload.

    ONE implementation: independent copies of this scaffolding (the subprocess
    call, the PYTHONPATH injection, the returncode assert carrying both streams,
    and the scrape-the-LAST-`PROBE`-line trick, which is what survives stdout
    warning noise) drift, each fix landing in one copy and not the others. Each
    test file owns only its probe SOURCE string.

    Two properties every caller relies on:

      * `PYTHONPATH` is PREPENDED, never replaced. `dict(os.environ,
        PYTHONPATH=str(REPO))` -- the obvious shape -- drops a
        pre-existing PYTHONPATH, so on a machine whose dependencies arrive that
        way the probe's failure would be misreported as the poisoned import
        failing.
      * `MUJOCO_GL` is stripped from the child environment, so a probe can
        assert that importing (or failing to construct) an adapter leaves it
        unset -- process-global state a validation path must not touch.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("MUJOCO_GL", None)
    proc = subprocess.run([sys.executable, "-c", source], cwd=str(REPO), env=env,
                          capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, (
        "probe subprocess failed. NOTE: a probe that poisons simulator modules in "
        "sys.modules asserts, transitively, that NO registry module imports them "
        "eagerly -- so a failure here can be an unrelated module gaining an eager "
        "import; read the traceback, not just this test's name.\n"
        + proc.stdout + proc.stderr)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE")]
    assert lines, f"probe printed nothing parseable:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(lines[-1][len("PROBE"):])


def _configs(subdir: str = "") -> list:
    root = CONFIG_ROOT / subdir if subdir else CONFIG_ROOT
    return sorted(p for p in root.glob("*.yaml") if not p.name.startswith("_"))


# --------------------------------------------------------------------------
# CONFIG DIRECTORIES, and the fixtures read off them.
# --------------------------------------------------------------------------
#
# Two things that are not the method are deliberately not config directories:
#
#   HOW EXPENSIVELY it runs   -> `configs/_profiles/*.yaml`, selected with
#                                `--profile`
#   WHICH ENVIRONMENT         -> a `-s problem.env_id=...` override
#
# A directory per combination would be the cross product written out by hand
# (methods x environments x profiles, each file restating what one profile and
# one override flag say once). A `--profile` and a `-s` override are not files,
# so no directory-diffing test can check them; the guard is that a profile
# states execution keys ONLY (`bird/config.py::PROFILE_KEY_PREFIXES`), and that
# is a check on `configs/_profiles/`, not here. Every directory that DOES exist
# under `configs/` must be classified in `CONFIG_DIRS` below, or
# `test_every_config_directory_is_classified` fails.

#: The published methods and their published variants, plus `zeroshot`, the
#: root they all extend: `configs/methods/*.yaml`.
METHOD_CONFIGS = _configs("methods")

#: The paper's own recipes, ERA-U and ERA-S: the configs at the top of `configs/`.
RECIPE_CONFIGS = _configs()

#: Every method point: the method corpus and the ERA recipes, ordered by name
#: (the order, and the `p.stem` test ids, the flat `configs/*.yaml` layout had).
PAPER_CONFIGS = sorted(METHOD_CONFIGS + RECIPE_CONFIGS, key=lambda p: p.stem)

#: The execution profiles, at `configs/_profiles/*.yaml`.
PROFILE_CONFIGS = sorted((CONFIG_ROOT / "_profiles").glob("*.yaml"))


#: Every runnable method point, paired with the `tester` profile.
#:
#: The tester tier is ONE profile rather than a file per method, so the cross
#: has to be built here instead of read off the filesystem. It holds EVERY
#: method point in `PAPER_CONFIGS`, including `singh_orp`, whose published search is
#: 3,240 candidates: a test that runs every tester point end to end must pass
#: its overrides through `apply_search_cap` below or it will hang on that one.
#:
#: NOT a list of paths, unlike every fixture above it -- a tester point is a
#: (config, profile) PAIR and there is no single file that denotes one. Callers
#: must `load(path, profile=profile)`.
TESTER_POINTS = [(p, "tester") for p in PAPER_CONFIGS]

#: What a test must ALSO pass to make a tester point run in about a second.
#:
#: The `tester` profile makes a TRAINING cheap (mock learner, 2,000 steps) but it
#: cannot make the SEARCH small: `generate.n_candidates` is a method key, and
#: `PROFILE_KEY_PREFIXES` deliberately refuses it, because 16 candidates IS
#: Eureka's contribution and a profile that could rewrite it could rewrite the
#: method.
#:
#: This is not a rounding difference: `singh_orp` under the tester profile
#: resolves to 3,240 candidates (its published hyperparameter search), each
#: forked and run twice by `tests/test_parallelism.py` at two worker counts.
#: That is a hang, not a slow test.
#:
#: FIVE, not three: `_check_coherence` refuses `keep_top_n(5) > n_candidates`, so
#: five is the floor GT's published value imposes rather than a number chosen
#: for speed.
TESTER_SEARCH_CAP = {"generate.n_candidates": 5}


def apply_search_cap(name, overrides, profile="tester"):
    """Apply `TESTER_SEARCH_CAP` as a CEILING, never as an assignment.

    A `setdefault` would silently RAISE any config published below the cap --
    `card` pins `generate.n_candidates: 1`, and run at 5 it is not a smaller
    card but a broken one: card's `select.rule` is `none`, which ranks nothing
    and returns `reports[:1]`, so four of the five would be generated, trained
    and discarded. `_check_coherence` refuses that combination outright.

    Resolving twice costs ~0.4 s and is the only way to know the published width
    before deciding whether to cap it. Cheap against the 3,240-candidate hang
    the cap exists to prevent.
    """
    from bird.config import load
    out = dict(overrides)
    published = load(name, profile=profile)
    for key, ceiling in TESTER_SEARCH_CAP.items():
        if key in out:
            continue
        if (published.get(key) or 0) > ceiling:
            out[key] = ceiling
    return out


# A HARD ASSERT, not a comment, and deliberately at module level in `conftest`.
#
# When a fixture's subject is deleted, the expression keeps evaluating and `[]`
# is a legal value for a list of paths. `empty_parameter_set_mark =
# "fail_at_collect"` does eventually catch an empty parametrize, but it reports
# it at whichever test file happened to consume the list, naming the parametrize
# argnames -- so the message points at a test that is fine and says nothing
# about `configs/`. The assert fires at the source instead and names the cause.
#
# Blast radius is the objection, and it is the right trade here: an assert in
# `conftest` takes the WHOLE suite down. But an empty `TESTER_POINTS` means
# `configs/methods/` holds no method point at all, and at that point there is
# no method set left for any other test to be meaningfully green about.
assert METHOD_CONFIGS and TESTER_POINTS, (
    f"no method configs at {CONFIG_ROOT}/methods/*.yaml, so every tester point is "
    "gone. Either the method configs moved (check configs/methods/ and "
    "configs/_profiles/) or `_configs()`'s glob no longer matches them. Do NOT "
    "'fix' this by making a consumer tolerate an empty list -- a silently empty "
    "fixture list is the failure this assert exists to catch."
)

#: Every directory under `configs/`, each with what it holds and which test
#: covers it. A value here is an argument, not a permission slip: it is what
#: `test_every_config_directory_is_classified` prints when it fails.
#: `_default.yaml` and `_profiles/` are skipped by the leading underscore.
#: `sweeps` is not committed: `scripts/ablate.py` writes it on demand, and it
#: would otherwise fail the classifier the first time someone runs an ablation.
CONFIG_DIRS = {
    "methods": "the published methods and their published variants (plus the "
               "`zeroshot` root), each holding its paper's values. Every file is "
               "loaded and validated by `test_paper_config_loads_and_validates`, "
               "and `tests/test_ablation.py` holds the published ablations to "
               "their key diffs against their parents.",
    "sweeps": "generated by scripts/ablate.py, one directory per single-key "
              "ablation; contents are derived, not authored.",
    "hillclimb": "OUR hill-climb points (v1 ... v4_peak_noes40) and their chain "
                 "parents: not published points -- they vary §1/§5 keys and "
                 "inherit the environment. What holds them is that each "
                 "`extends:` a published config (or a point that transitively "
                 "does), so `--diff` against that parent is the whole statement "
                 "of what the point changes.",
    "examples": "small illustrative configs (the jax reward path on jax_toy), not "
                "methods; each is loaded by the pairwise same-method check.",
}


def unclassified_config_dirs(names, known=None) -> set:
    """Which of `names` has no stated classification.

    Factored out of the test so the mechanism itself is testable: an assertion
    that only ever runs over the directories that happen to exist today cannot
    show that it would catch one that does not.

    `known` overrides `CONFIG_DIRS`. Without it the mechanism test could only
    run against the directories that happen to be classified today; passing an
    explicit `known` lets the test exercise both branches -- a known name is
    not reported, an unknown one is -- against a set it controls.
    """
    if known is None:
        known = set(CONFIG_DIRS)
    return {n for n in names if n not in known}


#: The PUBLISHED Gran Turismo point (human in the loop), as overrides on the one
#: GT config this repo ships. `configs/methods/gt.yaml` records the App-D
#: automated variant as the runnable point (one config per method); the full
#: method of arXiv 2511.02094 is that file plus EXACTLY this set, which its
#: header points to.
#: `tests/test_ablation.py::test_gt_is_a_minimal_diff` holds the file and this
#: set to that relation, so neither can drift from the other silently.
#: This is the ONE place the set exists in tests -- construct the published
#: point with `load_gt_published(...)` below, never with a local copy.
GT_PUBLISHED_OVERRIDES = {
    "name": "gt_reward_design",
    "generate.context.include_human_feedback": True,
    "evaluate.feedback.analyzer": "none",
    "evaluate.human.mode": "feedback_text",
    "evaluate.human.queries_per_iteration": 1,
}


#: The PUBLISHED REvolve point (human in the loop), as overrides on the one
#: REvolve config this repo ships. `configs/methods/revolve.yaml` records the paper's
#: OWN no-human ablation -- REvolve Auto, §5.2 tex:441-450 (fitness equations in
#: Appendix B.2) -- as the runnable point
#: (one config per method), for the same reason gt.yaml
#: does: the human channel is 10 evaluators x 20 video pairs per generation
#: (1400 comparisons over seven generations) and this repo has no human.
#: The full method of arXiv 2406.01309 is that file plus EXACTLY this set,
#: which its header points to.
#: `tests/test_ablation.py::test_revolve_is_a_minimal_diff` holds the file and
#: this set to that relation, so neither can drift from the other silently.
#:
#: LONGER THAN GT'S FIVE, and the length is the finding rather than a smell:
#: GT's human supplies prose only, so removing it is a feedback-channel swap,
#: while REvolve's human supplies BOTH the prose AND the scalar the whole
#: search is ordered by (an Elo rating over pairwise preferences, tex:317).
#: Deleting that scalar necessarily moves the fitness source, the preference
#: block that computes it, the artifacts the judge is shown, and what §0 is
#: allowed to read -- so the arms differ in a stage the GT arms share.
REVOLVE_PUBLISHED_OVERRIDES = {
    "name": "revolve_human",
    "problem.fitness_access": "none",
    "evaluate.fitness.source": "preference_bt",
    "evaluate.preferences.enabled": True,
    "evaluate.preferences.comparator": "human",
    "evaluate.preferences.aggregator": "elo_raw",
    "evaluate.preferences.allow_ties": True,
    "evaluate.preferences.scope": "cumulative",
    "evaluate.preferences.store_dataset": True,
    "evaluate.preferences.clip_length_s": 35,
    "evaluate.artifacts": ["scalar_metrics", "videos"],
    "evaluate.human.mode": "feedback_text",
    "evaluate.human.applies_to": "all_candidates",
    "evaluate.human.queries_per_iteration": 16,
    "generate.context.include_human_feedback": True,
    "loop.carry": ["archive", "preference_dataset", "best_reward"],
}


def load_revolve_published(profile=None, overrides=None):
    """`load()` for the published REvolve point: revolve + the override set.

    Caller overrides layer ON TOP, same as `load_gt_published`.
    """
    from bird.config import load
    ov = dict(REVOLVE_PUBLISHED_OVERRIDES)
    ov.update(overrides or {})
    return load("revolve", overrides=ov, profile=profile)


def load_gt_published(profile=None, overrides=None):
    """`load()` for the published GT point: gt + GT_PUBLISHED_OVERRIDES.

    Caller overrides layer ON TOP, same as they would on any config, so a test
    can still move a training key without restating the human ones.
    """
    from bird.config import load
    ov = dict(GT_PUBLISHED_OVERRIDES)
    ov.update(overrides or {})
    return load("gt", overrides=ov, profile=profile)


def config_dirs() -> list:
    """Top-level directories under `configs/`, in sorted order.

    Top level only, and leading-underscore names are skipped:
    `configs/_profiles/` (the execution profiles) is underscore-prefixed, so it
    is never mistaken for a config directory. On this tree it returns the
    `CONFIG_DIRS` entries that exist (`examples`, `hillclimb`, `methods`).
    """
    return sorted(p.name for p in CONFIG_ROOT.iterdir()
                  if p.is_dir() and not p.name.startswith("_"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


#: The modules whose adapters need HumanoidBench. Derived per ADAPTER MODULE, not
#: listed per env id: every hand-maintained list of HumanoidBench env ids is one
#: more list to go stale when an env is added.
_HUMANOID_MODULES = ("bird.envs.humanoid_hand",)

#: Simulator-gated adapter module -> the marker its env ids carry in parametrised
#: tests. Read off the FACTORY'S MODULE rather than a name list, so registering a
#: second env from either family marks it automatically. `humanoid` cannot share an
#: interpreter with metaworld, so it is deselected rather than left to skip.
_SIM_MARKERS = {
    **{mod: "humanoid" for mod in _HUMANOID_MODULES},
    # The jax tier's offline env: its factory imports jax first and raises
    # ImportError without it, and the `jax` extra is held out of `all`
    # (pyproject), so a skip here would be permanent -- deselect instead.
    "bird.envs.jax_toy": "jax",
}


def env_param(name: str):
    """One `parametrize` entry for an env id, MARKED if it needs HumanoidBench.

    HumanoidBench cannot share an interpreter with metaworld -- it needs
    `mujoco==3.1.6` and metaworld pins `3.3.0` -- so no install runs both tiers.
    A parametrisation over the env registry therefore has two options for a
    HumanoidBench id, and they are not equivalent:

      skip     adds a permanent skip to every metaworld run, and a skip hides a
               test that silently stopped running among ones that could never run.
      deselect keeps the skip tally meaningful.

    `-m "not humanoid"` in a metaworld venv deselects. That is `pyproject.toml`'s
    stated design for the HumanoidBench tests; this applies it to the
    parametrised cases, which would otherwise skip.

    The marker is read off the FACTORY'S MODULE rather than a name list, so
    registering a second HumanoidBench env marks it automatically. Nothing is
    constructed -- `registry.get` returns the factory, and constructing is the
    thing that needs the simulator.
    """
    from bird import registry
    registry.load_all()
    factory = registry.get("env", name)
    marker = _SIM_MARKERS.get(getattr(factory, "__module__", ""))
    if marker:
        return pytest.param(name, marks=getattr(pytest.mark, marker))
    return name


#: `train_backend` implementation module -> marker, the `_SIM_MARKERS` argument
#: applied to LEARNERS instead of environments. EMPTY today: every learner this
#: release ships is installable from the lock (or gated by its own extra), so none
#: needs a deselect marker. A learner whose runtime no `uv.lock` install provides
#: would take a row here.
_BACKEND_MARKERS: dict = {}


#: Backends that do not implement a given CONTRACT, with the reason. A strict
#: xfail rather than a skip, and the difference matters: a skip keyed on
#: availability would PASS BY ABSENCE on a CPU box and fail on a GPU one,
#: reporting the machine instead of the code. A strict xfail states what is
#: true -- the contract is not implemented -- and turns the case RED the day
#: someone implements it, which is when the mark should come off.
_BACKEND_UNIMPLEMENTED = {
    "assistax_ppo": (
        "assistax_ppo: pruner/timeout/checkpoint-selection AND TRAINING-CURVE "
        "contracts not implemented on the single-program JAX path -- upstream's "
        "IPPO is one jitted program per training, so there is no mid-training "
        "point at which a pruner is consulted, no timeout stop, and no "
        "per-checkpoint restore. THE CURVE CONTRACT IS A SEPARATE CAUSE UNDER "
        "THE SAME ROOT and is named because a reason that omitted it would hand "
        "the reader of a training-curve xfail an explanation about pruners: the "
        "path emits ONE checkpoint row per seed at the final step -- which is "
        "what the native-fitness reader needs -- while upstream's per-update "
        "series (episode returns, losses, entropy, KL) live in the seed row's "
        "`train_curve`, not in the per-checkpoint `reward_return` shape the "
        "training-curve reader consumes. So there is no per-update row for it "
        "to read. Producing per-update checkpoint rows carrying the candidate's "
        "reward changes what the jitted scan emits and moves counts, so it is a "
        "separate change. This mark IS the record -- strict, so it turns red "
        "the day the contract is implemented, which is the only tracking "
        "that cannot go stale."),
}


def backend_param_unimplemented(name: str):
    """`backend_param`, plus a STRICT xfail where the contract is unimplemented.

    `strict=True` is passed EXPLICITLY on every mark this produces, because
    `xfail_strict` is not set in `pyproject.toml` -- a bare `xfail` would
    pass whether the case failed or not, so it could never retire and would
    hide the contract being implemented.
    """
    base = backend_param(name)
    reason = _BACKEND_UNIMPLEMENTED.get(name)
    if reason is None:
        return base
    existing = list(getattr(base, "marks", ()))
    return pytest.param(name, marks=(*existing,
                                     pytest.mark.xfail(strict=True, reason=reason)))


def backend_param(name: str):
    """One `parametrize` entry for a `train_backend`, MARKED if its learner needs a
    runtime no `uv.lock` install provides (none today).

    THE SAME ARGUMENT AS `env_param`, and it is not interchangeable with an
    `importorskip`: a backend catalogue parametrised over the registry would
    otherwise SKIP a learner whose runtime no install can provide, and a skip
    that no install can remove hides the skips that mean something; a deselect
    keeps the tally meaningful. `_BACKEND_MARKERS` is empty today.

    The marker is read off the BACKEND FACTORY'S MODULE rather than a name list,
    so a second learner registered from the same module marks itself. Nothing is
    constructed -- `registry.get` returns the function, and importing the learner
    is the step that would need the runtime.
    """
    from bird import registry
    registry.load_all()
    fn = registry.get("train_backend", name)
    marker = _BACKEND_MARKERS.get(getattr(fn, "__module__", ""))
    if marker:
        return pytest.param(name, marks=getattr(pytest.mark, marker))
    return name
# ==========================================================================
# THE GL DEPENDENCY, DECLARED RATHER THAN DETECTED
# ==========================================================================

@pytest.fixture(scope="session")
def gl() -> None:
    """Declares "this test renders and therefore needs a GL context".

    IT DOES NOTHING, deliberately. Its whole job is to put the name `gl` into
    the test's `fixturenames` so `pytest_collection_modifyitems` below can
    mark it, and so `-m "not gl"` can DESELECT it. On a machine that HAS a GL
    context the behaviour of every test using it is byte-for-byte what it was
    before this fixture existed, because a no-op fixture cannot change it.

    WHY A FIXTURE AND NOT A MESSAGE MATCH. The per-site GL skips this replaces
    carried FIVE different spellings of one condition, two of them seen only
    once each -- and a singleton spelling is exactly the one an enumeration
    drops. A filter written against any one spelling misses the others, and
    nothing tells you. A declared dependency has no spelling.

    THREE UNITS, NEVER INTERCHANGEABLE. CALL SITES are places in the source.
    ITEMS are collected tests, which is what a conversion acts on. REPORTS are
    lines in a pytest skip tally. One
    parametrised function is one site and many items; a module-level
    `importorskip` is a report with NO item at all, which is why a skip it
    causes cannot be fixed by any per-test marker.

    WHY NOT A MODULE ATTRIBUTE. In a file that fails without a GL context,
    only a minority of tests render. A module-level marker would deselect the
    whole file to fix those few and take its PASSING tests out of the run.
    Per-test is the only granularity that deselects exactly what needs GL.

    WHY IT FAILS LOUDLY RATHER THAN SKIPPING. A test that renders and forgets
    to request this fixture is not deselected, so it FAILS on a GL-less
    runner. That is the intended default: a forgotten declaration should be
    visible, and the alternative -- a `try/except -> pytest.skip` per call
    site, which is what this replaces -- hides the same omission behind a
    skip. It also narrows nothing silently: a per-site `except Exception`
    reports any real defect inside the render as an absent GL context.

    A CENSUS MUST KEY ON THE CALL, NOT THE MESSAGE. A grep over tests/ for a
    skip message counts documentation (including this docstring, if it quoted
    one) as a call site. An AST census keyed on the `pytest.skip` CALL node is
    structurally immune, because prose cannot forge a call node: this file
    holds no skip call at all. Note also that a module-level `skipif` marks every collected
    ITEM, so its reports carry ids and are attributable; only `importorskip`
    yields the report with no item, and of those none here is GL-shaped.
    """
    return None


def pytest_collection_modifyitems(config, items):
    """Mark every test that requested `gl`, so `-m "not gl"` can deselect it.

    Derived from `fixturenames`, which pytest has populated by collection
    time -- that is what makes deselection possible at all. A marker applied
    inside the fixture body would come too late: the test would already have
    been selected, and by the time the fixture ran we would be back to
    skipping.

    Follows the same principle as `_SIM_MARKERS` above -- the marker is read
    off a declaration the test itself makes, not off a list of names kept in
    sync by hand.
    """
    for item in items:
        if "gl" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.gl)
