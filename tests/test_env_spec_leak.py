"""`generate.context.env_spec` must not hand the generator the ground truth.

`describe("full_source")` is the DEFAULT (`configs/_default.yaml`), and among the
shipped configs it is what `eureka`, `eureka_no_evolution` and `rda` receive. It renders
`inspect.getsource(type(self))` -- the adapter CLASS -- so anything written inside a
class body reaches the model that is being asked to invent a reward for that class.

Three guards here, and they are deliberately of different strength. Read the last
docstring for the check that is NOT here and why.
"""

import inspect

import pytest

from bird import registry
from bird.components.generation import _apply_symbol_mapping, _strip_reward


class _Ctx:
    def __init__(self, env):
        self.env = env


def _adapters():
    registry.load_all()
    # `mt10_*` are ten parametrisations of one adapter; one is enough and the rest
    # cost a MuJoCo construction each.
    from conftest import env_param

    # `env_param` marks the HumanoidBench ids for `-m "not humanoid"`, so they
    # are DESELECTED rather than reaching `_build`'s skip. A skip there would be
    # permanent on every install.
    # One Meta-World representative for all fifty (`mt10_`/`mt50_`, one adapter class):
    # each would cost a MuJoCo construction and prove nothing the first did not.
    names = [n for n in registry.names("env")
             if not n.startswith(("mt10_", "mt50_"))]
    return [env_param(n) for n in names] + ["mt10_reach-v3"]


def _adapter_names():
    """`_adapters()` as bare strings, for guards that iterate rather than parametrise.

    `_adapters()` returns `pytest.param` objects so `env_param` can mark the
    HumanoidBench ids for `-m "not humanoid"` (a deselect, where a skip would be
    permanent on every install). `parametrize` unwraps a `ParameterSet`; a plain
    `for` loop does not, and hands `registry.get` an unhashable `MarkDecorator`.
    """
    return [getattr(p, "values", (p,))[0] for p in _adapters()]


def _buildable():
    """`(name, env)` for every adapter this interpreter can construct.

    For tests that LOOP over the whole env family rather than being parametrised over
    it. Calling `_build` inside such a loop is wrong: `_build` calls `pytest.skip` --
    which is correct for a parametrised case, where exactly that env's case is skipped,
    and wrong here in two ways:

      * it abandons the WHOLE test because one env's simulator is absent, so the other
        adapters' mappings go unchecked on any install missing an extra;
      * it adds a skip that no single install can remove, which is the exact cost
        `_adapters`' `env_param` deselection exists to avoid.
        Routing these loops through `_build` to avoid the ParameterSet collision would
        therefore re-introduce the problem that deselection solves.

    Raises rather than yielding nothing. A loop over an empty set passes having asserted
    nothing, and that silence is the entire subject of this file.
    """
    out = []
    for name in _adapter_names():
        try:
            out.append((name, registry.get("env", name)({})))
        except ImportError:
            continue
    if not out:
        raise RuntimeError(
            "no env adapter could be constructed, so every loop over the env family "
            "would pass having checked nothing. Install an extra, or fix the registry.")
    return out


#: Both loops below inspect a `symbol_mapping` and SKIP an adapter that has none --
#: legitimately, since not every env declares one. That `continue` is a second way for
#: the loop to go quiet, and `_buildable`'s floor cannot see it: the floor asks whether
#: any adapter was CONSTRUCTED, and these tests assert over adapters that carry a
#: MAPPING. A buildable set in which nothing declares one passes both guards having
#: checked nothing, with the floor above satisfied.
#:
#: Measured under `--extra test`, the weakest supported install: 6 adapters build
#: and 6 carry a mapping, 34 keys total. So this is not vacuous as shipped -- it is
#: pinned so that it cannot become vacuous silently, which is the same argument
#: `test_no_ground_truth_def_survives_full_source` makes with its `assert present`
#: one function up, and the same argument `_buildable` makes one level below.
_NOTHING_CHECKED = (
    "no buildable adapter declared a `symbol_mapping`, so this loop passed having "
    "asserted nothing. `_buildable`'s floor cannot catch this -- it fires when nothing "
    "could be CONSTRUCTED, and here everything was, with nothing to check. Either an "
    "adapter lost its mapping or this install builds only adapters that never had one.")


def _build(name):
    """Construct, or skip. CI installs `--extra test` (pyyaml + numpy + pytest), so the
    simulator-backed adapters cannot be built there and the parametrisation -- which
    keys on NAMES -- cannot know that."""
    try:
        return registry.get("env", name)({})
    except ImportError as exc:
        pytest.skip(f"{name}: adapter needs an optional simulator ({exc})")


#: Every `def` that carries ground truth, across both shapes of render.
#:
#: The first three are `EnvAdapter`'s own and are exactly what `_strip_reward`'s
#: `_GROUND_TRUTH_METHODS` names. The rest are UPSTREAM Meta-World names, and they
#: are why this list exists rather than a loop over `type(env)`'s methods:
#: `MetaWorld._render_full_source` overrides the base and assembles the Sawyer task
#: class instead of the adapter class, so `full_source` on that tier contains no
#: `def task_metric` at all -- measured, 19,781 B of upstream source with neither
#: `class MetaWorld` nor `def task_metric` anywhere in it. A check keyed on the
#: adapter's own method names would therefore pass on the one tier whose render is
#: someone else's code, which is the tier with the most to leak.
#: ...and `evaluate` / `compute_dense_reward` are the task-class names of the
#: Text2Reward / CARD convention (`compute_dense_reward` is their published reward
#: signature, `evaluate` the success check beside it), listed as defence in depth
#: so that a render carrying them is held to the bar from its first run. A
#: `def evaluate` is cut only when the
#: adapter's `reward_source` hands `_strip_reward` its verbatim span, and a
#: `def compute_dense_reward` by the `def \w*reward\w*` regex. Note
#: `"evaluate"` also substring-matches `def evaluate_state` in the Meta-World
#: render -- harmless, since that span is stripped there for its own reason.
#: ...and `_success_flag` is the HumanoidBench tier's per-step predicate, the metric's
#: own per-step term on every h1hand class, which a stripper regex matching `success`
#: only as a whole name would let through.
#: `tests/test_strip_success_flag.py` covers that sim-free over the class TEXT; this
#: list holds it against the real render on an install that can build the tier.
_GROUND_TRUTH_DEFS = ("task_metric", "success", "reference_reward",
                      "compute_reward", "evaluate_state", "_gripper_caging_reward",
                      "evaluate", "compute_dense_reward", "_success_flag")


@pytest.mark.parametrize("env_name", _adapters())
def test_no_ground_truth_def_survives_full_source(env_name):
    """Whatever ground-truth `def` the render CONTAINS, the strip removes.

    Keyed on the rendered text rather than on `type(env)`, so an adapter that
    overrides `_render_full_source` is held to the same bar as one that does not --
    and non-vacuously, because the list of defs actually present is asserted
    non-empty before anything is checked against it. Measured across the shipped adapters:
    every `bird.envs.toy` / `bird.envs.control` adapter renders `task_metric` +
    `reference_reward`, and `mt10_reach-v3` renders `compute_reward` +
    `evaluate_state` + `_gripper_caging_reward`.

    The two mechanisms behind those removals are different and both are covered
    here: `_REWARD_DEF` matches `def \\w*reward\\w*`, `def \\w*success\\w*` and the
    EnvAdapter names, while `evaluate_state` -- the Sawyer success check, which that
    regex does NOT match -- goes only because `MetaWorld.reward_source` hands
    `_strip_reward` its verbatim AST span."""
    env = _build(env_name)
    text = env.describe("full_source")
    stripped = _strip_reward(_Ctx(env), text)

    present = [d for d in _GROUND_TRUTH_DEFS if f"def {d}" in text]
    assert present, (
        f"{env_name}: `full_source` contains none of {_GROUND_TRUTH_DEFS}, so this "
        "check compared nothing. Either the adapter renders no ground truth at all "
        "or its render names it something this list has not been told about -- add "
        "the name rather than leaving the tier unguarded.")
    survived = [d for d in present if f"def {d}" in stripped]
    assert not survived, (
        f"{env_name}: {survived} survived `strip_existing_reward`, so the generator "
        "is being shown the ground truth it is scored on.")


@pytest.mark.parametrize("env_name", _adapters())
def test_the_ground_truth_methods_do_not_survive_full_source(env_name):
    """`strip_existing_reward` removes the reward AND the metric, on every adapter.

    `_GROUND_TRUTH_METHODS` widens the strip from the reward to the *fitness*, because
    on an env whose success check is ours rather than the simulator's the metric is
    the more dangerous leak of the two: a candidate that can read `task_metric` can
    restate it, and then every §4 number measures the model's ability to copy rather
    than to design.

    Asserted only on body lines that are UNIQUE in the unstripped render. Matching
    every body line anywhere in the stripped text fails on four adapters -- on lines
    like `return float(reward)` and `s = np.asarray(s, dtype=float).ravel()`, which
    are boilerplate shared with other methods that are supposed to survive. That is a
    check over-sensitive to the wrong thing: it reports a leak where there is none,
    the mirror of a check that cannot fail. Uniqueness is what makes a surviving line actually mean "this body survived".
    """
    env = _build(env_name)
    text = env.describe("full_source")
    stripped = _strip_reward(_Ctx(env), text)

    try:
        # THE SAME SET THE RENDERER USES, not `type(env)` alone. This gates
        # which methods are checked -- `if f"def {method}" not in source:
        # continue` below -- and a DELEGATING adapter defines none of them on
        # its subclass, so with the subclass alone every method `continue`s
        # and check (b) never runs on the jax tier. That is the "passes while
        # guarding nothing" shape this file exists to catch, one level up.
        source = "\n".join(
            [inspect.getsource(p) for p in (getattr(env, "source_parents", lambda: ())() or ())]
            + [inspect.getsource(type(env))])
    except (OSError, TypeError):  # pragma: no cover - zipimport
        pytest.skip(f"{env_name}: source unavailable")

    for method in ("task_metric", "reference_reward", "_success_flag"):
        fn = getattr(type(env), method, None)
        if fn is None or f"def {method}" not in source:
            continue  # inherited from a base class, so not in this class's source
        body = inspect.getsource(fn)
        # The signature line survives as the `def` that got replaced; the BODY must not.
        payload = [ln.strip() for ln in body.splitlines()
                   if ln.strip() and not ln.strip().startswith(("def ", "#", '"', "'"))]
        # (a) The `def` line itself. Exact, and vacuous on exactly one adapter:
        #     `mt10_*` renders upstream Sawyer source, in which `def task_metric`
        #     never appears, so this can only pass there. That gap is what
        #     `test_no_ground_truth_def_survives_full_source` above covers; this stays
        #     because on the other eight it means "the strip did not fire at all".
        assert f"def {method}" not in stripped, (
            f"{env_name}: `def {method}` survived `strip_existing_reward` -- the strip "
            "did not fire, and the generator is being shown the ground truth it is "
            "scored on.")

        # (b) The body, on lines UNIQUE in the render. Catches a partial strip, which
        #     (a) cannot. `continue`, NOT `pytest.skip`, when no line is unique: skip
        #     raises, so it ends the whole function -- on `mt10_reach-v3` (0 unique
        #     lines in `task_metric`) it would cancel the `reference_reward` pass and
        #     the `withheld` assertion below as well, and add a permanent skip. A
        #     method with nothing checkable must cost the other methods nothing.
        unique = [ln for ln in payload if text.count(ln) == 1]
        # Membership over WHOLE STRIPPED LINES, not substrings. `mt10_reach-v3`'s
        # `return float(reward)` is a substring of upstream's surviving
        # `return float(reward), float(reachDist), float(0.0)`, so the substring form
        # reports a leak that is not there the moment (b) stops being skipped.
        stripped_lines = {ln.strip() for ln in stripped.splitlines()}
        leaked = [ln for ln in unique if ln in stripped_lines]
        assert not leaked, (
            f"{env_name}: `{method}`'s body survived `strip_existing_reward`, so the "
            f"generator is being shown the ground truth it is scored on:\n"
            + "\n".join(leaked[:5]))

    assert "withheld" in stripped, (
        f"{env_name}: nothing was withheld from `full_source`. Either the adapter "
        "defines neither ground-truth method (in which case this env has no ground "
        "truth) or the stripper silently matched nothing.")


@pytest.mark.parametrize("env_name", _adapters())
def test_the_module_docstring_is_not_rendered_into_full_source(env_name):
    """Measured numbers belong in the MODULE docstring, and this is what makes that
    a location rule rather than a discipline.

    `_render_full_source` uses `inspect.getsource(type(self))`, which is the class and
    not the module, so a module docstring is provably not shown to the generator while
    a class-body comment provably is. (On `mt10_*` the override renders upstream Sawyer
    source and BIRD's module docstring cannot appear either way, so this passes there
    without proving anything -- the location rule it enforces is about the files this
    repo writes, which is where the hazard below lives.) That asymmetry is the
    only safe place to put a table of which rewards score well on an environment --
    and `_strip_reward` cannot help, because it removes `def` suites and has no way
    to remove a comment.

    The hazard is concrete: a table of measured numbers, or a description of the gaits
    a search found, written into a class body goes into the prompt for `eureka`,
    `eureka_no_evolution` and `rda`. Re-running the stripper by hand catches it; this
    test catches it in CI.

    THE CHECK THAT IS DELIBERATELY NOT HERE, and the absence is the point. The obvious
    stronger test is "assert no measured figure appears in `full_source`", and it
    cannot be written honestly. A regex over decimals fires on `dt = 0.05` and every
    other legitimate physical constant. A keyword list (`crawl`, `gait`, reward
    names) catches exactly the instances it was written from and not the next,
    which will be phrased differently -- a check whose pass carries no information.
    So this test enforces the *location*, which is mechanical and exact, and
    leaves the content judgement to the reviewer, where it belongs. Do not "strengthen"
    this with a keyword list; that would make it look stronger and be weaker.
    """
    env = _build(env_name)
    module = inspect.getmodule(type(env))
    doc = (getattr(module, "__doc__", None) or "").strip()
    if not doc:
        pytest.skip(f"{env_name}: module has no docstring")

    rendered = env.describe("full_source")
    # A short docstring could coincide with real source; compare on a long slice.
    probe = "\n".join(doc.splitlines()[:5]).strip()
    assert probe not in rendered, (
        f"{env_name}: the module docstring is being rendered into `full_source`. "
        "That breaks the one place measured numbers can safely live -- see this "
        "test's docstring. `_render_full_source` must stay `getsource(type(self))`.")


def test_symbol_mapping_matches_identifiers_and_not_substrings():
    """`generate.postprocess.symbol_mapping` rewrites the candidate's PROGRAM, so a
    boundary-free replace corrupts any word that happens to contain a key.

    The damage is not where it looks. Under a boundary-free replace, a mapping
    `"z" -> "s[1]"` turns `zero` into `s[1]ero`, and a mapping `"x" -> "s[0]"` turns
    `max(` into `mas[0](` -- a SyntaxError.

    **The corrupted programs are mostly not on disk, and that is the point.** A
    SyntaxError trips `verify.on_failure: repair`, which spends another LLM call and
    saves the REPAIRED program. So the artifact looks clean while the repair counters
    climb. A defect that reliably triggers a recovery mechanism leaves no trace in
    what the recovery produces.

    Asserted on the LIVE adapter tables rather than a synthetic mapping, because the
    defect is half mechanism and half table -- a guard over a made-up dict would pass
    with keys like `"z"` and `"x"` still in place.
    """
    registry.load_all()
    hazards = ("zero", "np.zeros(3)", "dead zone", "max(a, b)", "np.exp(v)",
               "index", "for i in range(3)", "joint_velocity", "next_state")
    checked = 0
    for env_name, env in _buildable():
        mapping = getattr(env, "symbol_mapping", None)
        if not mapping:
            continue
        checked += 1
        for src in hazards:
            got = _apply_symbol_mapping(mapping, src)
            assert got == src, (
                f"{env_name}: symbol_mapping rewrote {src!r} into {got!r}. A key is "
                "matching inside a longer word, which corrupts the candidate's program.")
        key = max(mapping, key=len)
        assert _apply_symbol_mapping(mapping, key) == mapping[key], (
            f"{env_name}: symbol_mapping no longer rewrites {key!r} at all")

    assert checked, _NOTHING_CHECKED


def test_no_symbol_mapping_key_is_a_single_letter():
    """A one-character key is a hazard even with word boundaries.

    `x`, `z` and `i` are the most common throwaway identifiers in Python, so a mapping
    that claims one turns `for x in ...` into `for s[0] in ...` -- a SyntaxError -- and
    an assignment `z = ...` into a write straight into the state array. The value of
    mapping a single letter is low; the blast radius is not.
    """
    registry.load_all()
    checked = 0
    for env_name, env in _buildable():
        mapping = getattr(env, "symbol_mapping", None) or {}
        checked += bool(mapping)
        short = sorted(k for k in mapping if len(k) < 2)
        assert not short, (
            f"{env_name}: single-letter symbol_mapping keys {short}. Spell them out "
            "(`x_position`, `height`) -- a one-letter key claims every throwaway "
            "variable a candidate might use.")

    assert checked, _NOTHING_CHECKED
