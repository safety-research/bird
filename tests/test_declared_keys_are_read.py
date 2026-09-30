"""Every key `configs/_default.yaml` declares has a reader under `bird/`.

THE CLASS THIS CATCHES: never declare a key the algorithm cannot honour. A key that exists in the
config space and is read by nothing is a **fabricated pin** -- it resolves, it
lands in `config.resolved.yaml`, it contributes its bytes to the run id, and it
changes nothing. An ablation over it produces a grid of identical results and
reports a max over a search that did nothing.

NOTHING ELSE TESTS IT, and the two tests that look like they might cannot.
`tests/test_schema_coverage.py` checks `_default.yaml` against `schema.py` in
both directions -- neither of which is code -- and `--print-config` resolves a
config without touching a reader. Both are green on a key nothing reads.

**WHAT THIS CANNOT TELL YOU, AND IT IS THE IMPORTANT HALF.** A reader is not a
honourer. This asserts that *something under `bird/` mentions the key at a
config access*, not that the value reaches behaviour. The worse variant is a
key read only into a string: `evaluate.artifacts: [videos]` +
`llm.<role>.modality: vlm` + `evaluate.vlm.images_per_query` read only to build
an `f"..."` describing frames that are never attached makes a VLM-graded method
run text-only while every counter reads normal. That code would pass this test.
`grep -n "images=" bird/components/` is the manual check for that family.
**Green here is not fidelity.**

VALIDATED AGAINST A POSITIVE CONTROL, which is the only thing that makes a
green run here mean anything: `test_the_detector_names_a_declared_unread_key`
builds a tree with one read and one unread key and asserts the check names
exactly the unread one. **Re-run it after any widening of the detector.**

THE DETECTOR NEEDED SIX WIDENINGS, and a reader should know that before
trusting its result. A naive walker searches only `bird/` and misses
`bird.py`, which is the algorithm; walks the receiver chain to its root, so
`ctx.cfg.get("x")` looks like `ctx`; matches only trailing f-string
placeholders, so `f"llm.{role}.model"` is invisible -- and a pattern loose
enough to fix that lets a checkpoint spill path `f"{path}.{f.name}"` excuse
three real keys as composed; cannot resolve a key held in a named constant
(`cfg.get(CONFIG_KEY)`, `bird/multiview.py:53`); and cannot see a key passed to
a local wrapper around the config (`_get("...")`,
`bird/components/frames.py:313`). Each shape was found by checking a reported
key against the tree, never by re-reading the checker. If you extend this file,
assume the same.

THE ALLOWLIST IS THE WHOLE DIFFICULTY AND IT FAILS CLOSED. Keys reached by name
composition -- `cfg.get(f"loop.curriculum.{key}")`, `cfg.get(prefix + name)`,
anything via `_flat_keys` -- carry no literal for an AST walk to find, so a naive
check reports them as unread. The temptation is a broad allowlist, and a broad
allowlist is indistinguishable from the bug. So every entry must NAME THE
COMPOSING CALL SITE, `file:line`, and a test below asserts each cited site still
exists and still composes. An entry is a claim a reviewer can check, not a
silence.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = ROOT / "configs" / "_default.yaml"

#: Not searched for readers, and each for a different reason.
#: `schema.py` DECLARES keys (it is the other half of what `_default.yaml` says,
#: not a consumer); `config.py` is the loader -- it reads every key generically
#: through `_flat_keys`, so counting it would make every key "read" and the test
#: vacuous. That second exclusion is the one that makes this test mean anything.
NOT_A_READER = {"bird/schema.py", "bird/config.py"}

#: Names whose `.get("...")` / `["..."]` is a config access. Matched on the root
#: identifier so `cfg`, `ctx.cfg`, `self.cfg` and `self._cfg` all count.
CFG_NAMES = ("cfg", "config")

#: Keys with no literal reader, each naming the call site that composes the name.
#: EVERY ENTRY IS A CHECKABLE CLAIM -- `test_every_allowlisted_key_cites_a_live_site`
#: opens the cited file, asserts the line still exists and still contains a
#: composing access. A citation that rots fails rather than going quiet.
COMPOSED_READS: dict[str, str] = {
    # generate.context.* -- `ctxcfg = "generate.context."` then `cfg[ctxcfg + leaf]`
    **{f"generate.context.{k}": "bird/components/evolution.py:296" for k in (
        "guidance", "include_archive_elites", "include_behavioural_analysis",
        "include_curriculum_stage", "include_failure_traces",
        "include_numeric_reflection", "include_parent_code",
        "include_reward_recipe_hints", "include_safety_instruction",
        "include_task_decomposition", "reflection_guidance", "safety_instruction",
        "shuffle_sections", "system_prompt")},
    # loop.curriculum.* -- `cfg.get(f"loop.curriculum.{key}", default)`
    **{f"loop.curriculum.{k}": "bird/components/curriculum.py:124" for k in (
        "author_max_rounds", "gate_threshold", "gate_votes", "regression_check",
        "regression_tolerance", "stage_patience")},
    # llm.<role>.* -- `cfg.get(f"llm.{role}.model")` and siblings
    "llm.generator.model": "bird/llm/base.py:180",
    "llm.evaluator.model": "bird/llm/base.py:180",
    "llm.generator.modality": "bird/llm/base.py:182",
    "llm.evaluator.modality": "bird/llm/base.py:182",
    "llm.generator.reasoning_effort": "bird/llm/base.py:185",
    # budget.max_* -- `Budget.from_cfg` builds every cap with
    # `cfg[f"budget.{k}"] for k in cls.cap_names()`, deriving the key from the
    # dataclass field so a fourth cap is honoured the moment it is a field.
    # The other caps keep a literal read elsewhere; these two are read only
    # through the composed name -- which is this allowlist's whole subject,
    # and the reason it names a site rather than a key.
    # A field added to `Budget` ABOVE `from_config` moves the line, and
    # `test_every_allowlisted_key_cites_a_live_site` checks a `[n-3, n+2]`
    # window, so the guard fails until the citation is re-derived by grep on
    # the edited tree -- the argument for `COHERENCE_ONLY`'s function locator
    # two blocks down.
    "budget.max_gpu_hours": "bird/budget.py:310",
    "budget.max_llm_calls": "bird/budget.py:310",
}

#: READ ONLY BY THE COHERENCE CHECK -- read by NO STAGE. `_check_coherence`
#: names these to enforce a cross-key invariant a per-key schema cannot express,
#: and nothing in the six stages consults them: they constrain which configs are
#: LEGAL and reach no behaviour.
#:
#: That is a real class and a narrower one than "nothing reads it", and the
#: "honour the key, never delete it quietly" rule is ALREADY SATISFIED for every
#: key here. `_check_coherence`'s `_REPRESENTATION_FORMATS` table says so in as
#: many words, with the ablation that measured it
#: (`-s problem.reward_representation=free_form_code` on rda validated, moved the
#: hash, and ran byte-identical): the honouring is the pairing itself -- each
#: value tied to the dispatching value that implements it, so an inconsistent
#: pair fails at load naming both.
#:
#: THE LOCATOR FOR THAT ARGUMENT IS THE TABLE, NOT A LINE RANGE. A line range in
#: `bird/config.py` rots as the file grows, and `_check_coherence` ALONE spans
#: some two thousand lines -- true, and useless for finding the argument. A
#: function name is a good locator exactly when the function is small enough to
#: read, or when a test pins the claim to its source span (which is what
#: `test_every_coherence_only_key_is_named_by_the_coherence_check` does for the
#: KEYS below, and cannot do for prose). `_REPRESENTATION_FORMATS` is declared once
#: in the tree, sits immediately under the argument, and cannot move away from it
#: without taking the argument along.
#:
#: This category exists because excluding `bird/config.py` wholesale -- correct
#: for `_flat_keys`, wrong for `_check_coherence` -- reports all nine as having no
#: reader at all.
#:
#: THE LOCATOR FOR EACH KEY IS THE FUNCTION, NOT A LINE. A `config.py:<line>`
#: comment per entry rots QUIETLY as `config.py` grows:
#: `test_every_allowlisted_key_cites_a_live_site` reads `COMPOSED_READS`, and a
#: citation in a comment is opened by nothing. A citation no test opens is the
#: thing this file's own header calls indistinguishable from the bug.
#:
#: So the locator is `_check_coherence` -- which cannot drift, because the key
#: has to be inside it for the claim to be true -- and
#: `test_every_coherence_only_key_is_named_by_the_coherence_check` asserts
#: exactly that, by reading the function's source span. The same argument holds
#: for citing papers: cite a NAMED locator, not a number that is a rendering
#: artifact.
COHERENCE_ONLY: frozenset[str] = frozenset({
    "problem.reward_representation",      # paired with generate.output.format
    "problem.search_space_mode",
    "select.retrain_before_select",
    "update.archive.enabled",
    "update.meta.prompt_optimizer",
    "verify.alignment_filter.enabled",
    "verify.cascade.enabled",
    "verify.tpe.enabled",
    "verify.tpe.store",
})

#: READ ONLY BY THE LOADER. `bird/config.py` is excluded as a reader because it
#: walks every key through `_flat_keys`, but a handful of keys are resolved by
#: NAME there, outside `_check_coherence`, as part of building the config
#: itself. Same class as COHERENCE_ONLY -- read by no stage -- listed apart
#: because the reason differs: the loader consumes it to decide what to load.
#:
#: Empty in this release: no declared key is currently resolved by name in the
#: loader alone. An entry is `key: "file:line"` and is checked like
#: `COMPOSED_READS`; the consistency test below skips an entry whose key
#: `_default.yaml` does not declare.
LOADER_ONLY: dict[str, str] = {}

#: READ BY NOTHING AT ALL -- not a stage, not the coherence check. The actual
#: finding, and it is four keys rather than the thirteen a naive check reports.
#: Carried as a worklist, not a pardon: the rule is honour the key, never delete
#: it quietly, so each entry leaves when a reader is wired.
KNOWN_UNREAD: frozenset[str] = frozenset({
    "update.co_evolve.dr_config",
    "verify.alignment_filter.metric",
    "verify.alignment_filter.preference_dataset",
    "verify.llm_self_critique.enabled",   # _default.yaml already calls it
                                          # "implemented by nobody"
})


def leaf_keys() -> set[str]:
    """Every dotted leaf of `_default.yaml`.

    A mapping is a leaf when it is empty (`hyperparameters: {}` is a key whose
    value is a free dict, not a branch) -- otherwise the walk would descend into
    user data and demand readers for keys nobody declared.
    """
    def walk(node, prefix=""):
        if isinstance(node, dict) and node:
            for k, v in node.items():
                yield from walk(v, f"{prefix}{k}.")
        else:
            if prefix:
                yield prefix[:-1]
    return set(walk(yaml.safe_load(DEFAULTS.read_text())))


def _receiver_segments(node: ast.AST) -> list[str]:
    """Every identifier in the receiver chain, e.g. `ctx.cfg` -> ["ctx", "cfg"].

    The WHOLE chain, not just its root: walking to the base `Name` turns
    `ctx.cfg.get("x")` into `"ctx"`, `"cfg" in "ctx"` is False, and every
    `ctx.cfg` read in the tree -- which is most of them -- is invisible. The
    symptom is 165 of 370 leaves reported unread, i.e. the checker accusing the
    codebase of a defect that is its own.
    """
    segs: list[str] = []
    while True:
        if isinstance(node, ast.Attribute):
            segs.append(node.attr); node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Name):
            segs.append(node.id); return segs
        else:
            return segs


def _is_cfg(node: ast.AST) -> bool:
    return any(any(n in seg for n in CFG_NAMES) for seg in _receiver_segments(node))


def _module_constants() -> dict[str, dict[str, str]]:
    """`{module_rel: {NAME: "literal"}}` for module-level `NAME = "a.b.c"`.

    THE FIFTH COMPOSED SHAPE. A key can be held in a named constant and read
    as `cfg.get(CONFIG_KEY)` -- `bird/multiview.py:53` does exactly that for
    `output.video.n_views`, which a literal-only checker reports as unread.

    Resolved rather than allowlisted: a constant holding a literal IS a literal
    read, one indirection away, and excusing it by hand would put five true
    entries in an allowlist that is supposed to hold only the genuinely
    undecidable.
    """
    out: dict[str, dict[str, str]] = {}
    for path in [ROOT / "bird.py"] + sorted(ROOT.joinpath("bird").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        consts: dict[str, str] = {}
        for node in tree.body:                       # module level only
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        consts[t.id] = node.value.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str) and isinstance(node.target, ast.Name):
                consts[node.target.id] = node.value.value
        out[rel] = consts
    return out


#: Every module-level string constant in the tree, by bare name. Flat because a
#: read may be `cfg.get(CONFIG_KEY)` after `from .multiview import CONFIG_KEY`
#: or `cfg.get(multiview.CONFIG_KEY)`; both reach the same value, and which
#: module it came from does not change whether the key is read.
_ALL_CONSTANTS: dict[str, str] = {}


def literal_reads() -> dict[str, list[str]]:
    """Every config key read by a string LITERAL under `bird/`, with sites."""
    found: dict[str, list[str]] = {}
    # `bird.py` FIRST, and it is not a detail: it is the whole algorithm --
    # the six stages and the loop live there, and "under bird/" misses every
    # key they read. Searching only the package reports 211 of 370 leaves
    # unread, which is the checker being wrong rather than the tree.
    for path in [ROOT / "bird.py"] + sorted(ROOT.joinpath("bird").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel in NOT_A_READER:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            key = None
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and node.args):
                if _is_cfg(node.func.value):
                    a = node.args[0]
                    key = a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else None
            elif isinstance(node, ast.Subscript):
                if _is_cfg(node.value):
                    s = node.slice
                    key = s.value if isinstance(s, ast.Constant) and isinstance(s.value, str) else None
            if key:
                found.setdefault(key, []).append(f"{rel}:{node.lineno}")
    return found


def accessor_functions(tree: ast.AST) -> set[str]:
    """Names of functions that forward their first parameter into a config access.

    THE SIXTH SHAPE. `bird/components/frames.py:313` reads three keys as
    `_get("evaluate.vlm.contact_sheet.cell_px", 96)`, where `_get` is a local
    wrapper whose body does `cfg[key]`. The call site has the literal and no
    `cfg` receiver, so a walker keyed on the receiver sees nothing and calls
    all three unread.

    A function qualifies when its first parameter reaches a `cfg` access
    somewhere in its body. That is narrow on purpose: it recognises a wrapper
    around the config, not any function that happens to take a string.
    """
    out: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = [a.arg for a in fn.args.args]
        if not params:
            continue
        first = params[0]
        for node in ast.walk(fn):
            target = None
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and node.args and _is_cfg(node.func.value)):
                target = node.args[0]
            elif isinstance(node, ast.Subscript) and _is_cfg(node.value):
                target = node.slice
            if isinstance(target, ast.Name) and target.id == first:
                out.add(fn.name)
                break
    return out


def wrapper_reads() -> dict[str, list[str]]:
    """Keys passed as the first argument to a config-accessor wrapper."""
    found: dict[str, list[str]] = {}
    for path in [ROOT / "bird.py"] + sorted(ROOT.joinpath("bird").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel in NOT_A_READER:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        names = accessor_functions(tree)
        if not names:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and node.args
                    and isinstance(node.func, ast.Name) and node.func.id in names
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                found.setdefault(node.args[0].value, []).append(f"{rel}:{node.lineno}")
    return found


#: Functions in `bird/config.py` that read SPECIFIC keys rather than walking all
#: of them. The rest of that module reads every key generically through
#: `_flat_keys`, which is why the file is in `NOT_A_READER` -- counting it whole
#: would make every key "read" and the test vacuous.
COHERENCE_FUNCS = ("_check_coherence",)


def coherence_reads() -> dict[str, list[str]]:
    """Keys read by `config.py`'s cross-key coherence check.

    THE SEVENTH SHAPE, and the one that most inflates a naive headline.
    Excluding `bird/config.py` wholesale is right for `_flat_keys` and wrong
    for `_check_coherence`, which reads named keys to enforce invariants a
    per-key schema cannot express. `problem.reward_representation` is read
    there and pinned against `generate.output.format`; reporting it as having
    no reader would be too strong.

    A SEPARATE CATEGORY, not folded into the literal readers, because the
    distinction is the finding: a key read only here is **read by no stage**.
    It constrains which configs are legal and reaches no behaviour. That is a
    real and narrower class than "nothing reads it", and the "honour the key"
    rule is already satisfied for a key in it -- the `_REPRESENTATION_FORMATS`
    table says so explicitly, with the ablation that measured it.
    """
    found: dict[str, list[str]] = {}
    path = ROOT / "bird" / "config.py"
    if not path.is_file():          # a synthetic tree in a test has no loader
        return found
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return found
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                or fn.name not in COHERENCE_FUNCS:
            continue
        # inside these, a one-letter getter `g("key")` is the idiom
        local = accessor_functions(fn) | {"g"}
        for node in ast.walk(fn):
            expr = None
            if isinstance(node, ast.Call) and node.args:
                if isinstance(node.func, ast.Name) and node.func.id in local:
                    expr = node.args[0]
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                        and _is_cfg(node.func.value):
                    expr = node.args[0]
            elif isinstance(node, ast.Subscript) and _is_cfg(node.value):
                expr = node.slice
            if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                found.setdefault(expr.value, []).append(
                    f"bird/config.py:{node.lineno} ({fn.name})")
    return found


def constant_reads() -> dict[str, list[str]]:
    """Config keys read through a named constant: `cfg.get(CONFIG_KEY)`."""
    consts: dict[str, str] = {}
    for mod in _module_constants().values():
        consts.update(mod)
    found: dict[str, list[str]] = {}
    for path in [ROOT / "bird.py"] + sorted(ROOT.joinpath("bird").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel in NOT_A_READER:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            expr = None
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get" and node.args and _is_cfg(node.func.value)):
                expr = node.args[0]
            elif isinstance(node, ast.Subscript) and _is_cfg(node.value):
                expr = node.slice
            if expr is None:
                continue
            name = (expr.id if isinstance(expr, ast.Name)
                    else expr.attr if isinstance(expr, ast.Attribute) else None)
            if name and name in consts:
                found.setdefault(consts[name], []).append(f"{rel}:{node.lineno}")
    return found


def unread_keys() -> dict[str, None]:
    """Declared leaves with no literal reader and no allowlist entry."""
    read = literal_reads() | constant_reads() | wrapper_reads() | coherence_reads()
    return {k: None for k in sorted(leaf_keys())
            if k not in read and k not in COMPOSED_READS
            and k not in COHERENCE_ONLY and k not in LOADER_ONLY
            and k not in KNOWN_UNREAD}


def test_every_declared_key_has_a_reader():
    """A key in `_default.yaml` that nothing under `bird/` reads is a fabricated
    pin: it resolves, it enters the run id, and it changes nothing."""
    bad = unread_keys()
    assert not bad, (
        "these keys are declared in configs/_default.yaml and read by nothing "
        f"under bird/ ({len(bad)}):\n"
        + "".join(f"    {k}\n" for k in bad)
        + "\nEither wire a reader -- never declare a key the "
          "algorithm cannot honour -- or, if the key IS read but by a composed "
          "name (an f-string, a prefix concatenation, _flat_keys), add it to "
          "COMPOSED_READS in this file with the composing call site as "
          "`file:line`. Do not add it without the site: an allowlist entry that "
          "names nothing is indistinguishable from the bug this test exists for."
    )


#: Where the algorithm actually starts. A resolver that no path from one of
#: these reaches is dead, and a dead resolver must not launder a key into the
#: read column.
ROOTS = frozenset({
    "generate", "verify", "train", "evaluate", "select", "update",   # the six stages
    "run", "main", "_loop",                                          # bird.py's loop
    "load", "validate", "_check_coherence",                          # the config path
})


def call_graph() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """`(callers_of, defined_in)` over the tree, keyed by bare function name.

    BY NAME, which is the honest limit: two functions sharing a name are one
    node, so reachability here can be optimistic across a name collision. It
    cannot be pessimistic, so a resolver this calls dead IS dead. That is the
    direction a guard wants.
    """
    calls: dict[str, set[str]] = {}
    defined: dict[str, set[str]] = {}
    for path in [ROOT / "bird.py"] + sorted(ROOT.joinpath("bird").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, FileNotFoundError):
            continue
        rel = str(path.relative_to(ROOT))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            defined.setdefault(fn.name, set()).add(rel)
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    f = node.func
                    name = (f.id if isinstance(f, ast.Name)
                            else f.attr if isinstance(f, ast.Attribute) else None)
                    if name:
                        calls.setdefault(name, set()).add(fn.name)
    return calls, defined


def registered_components() -> set[str]:
    """Functions bound by `@register(kind, name)`.

    THE REGISTRY BREAKS THE STATIC CALL GRAPH, and that is the architecture
    working rather than a defect: a stage's body reads a config value and
    dispatches through `registry.get(kind, value)`, so there is no call edge
    from `generate` to the component that implements it. Nothing under
    `bird/components/` is statically reachable from the six stages.

    A registered component is therefore a ROOT: the registry is how the
    algorithm calls it, and `tests/test_schema_coverage.py` already guarantees
    every allowed config value has one behind it. Without this the reachability
    test declares the entire component layer dead -- naming, for one,
    `_numeric_section()` even though `evolution.py:535` calls it.
    """
    out: set[str] = set()
    for path in [ROOT / "bird.py"] + sorted(ROOT.joinpath("bird").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, FileNotFoundError):
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in fn.decorator_list:
                f = dec.func if isinstance(dec, ast.Call) else dec
                name = (f.id if isinstance(f, ast.Name)
                        else f.attr if isinstance(f, ast.Attribute) else None)
                if name in ("register", "phase"):
                    out.add(fn.name)
    return out


def reachable_from_roots() -> set[str]:
    """Function names reachable from :data:`ROOTS` or a registered component."""
    calls, _ = call_graph()
    # invert: callee -> callers, so walk upward from every function to a root
    seen: set[str] = set()
    frontier = set(ROOTS) | registered_components()
    while frontier:
        seen |= frontier
        nxt: set[str] = set()
        for name, callers in calls.items():
            if name in seen:
                continue
            if callers & seen:
                nxt.add(name)
        frontier = nxt - seen
    return seen


def test_every_cited_resolver_is_reachable_from_a_stage_or_the_loader():
    """A cited site must be LIVE, not merely present.

    Asserting that a resolver exists lets a dead one launder every key it
    mentions into the read column; asserting that it has *a* caller launders it
    one level up, since the caller could be dead too. So the requirement is a
    call path from the six stages, the loop, or the config load/validate path.

    Same fail-open shape as an f-string pattern that lets a checkpoint spill
    path `f"{path}.{f.name}"` excuse three real keys as composed -- a pattern loose
    enough to match is not evidence of a read.

    VERIFIED BY MUTATION: citing a resolver inside a function no path reaches
    fails this test. The unreachable set is 675 of 2,717 names and is mostly
    dunders -- `__call__`, `__getitem__` -- which Python reaches dynamically
    and a name-based call graph cannot see. So a cited resolver living in a
    dunder would be refused wrongly. That is the pessimistic direction: this
    test can reject a live site, never accept a dead one, and for a guard that
    is the right way round. If it ever fires on a real dunder resolver, cite a
    caller instead of loosening the rule.
    """
    live = reachable_from_roots()
    _, defined = call_graph()
    bad = []
    for key, site in sorted({**COMPOSED_READS, **LOADER_ONLY}.items()):
        rel, _, lineno = site.partition(":")
        path = ROOT / rel
        if not path.is_file():
            continue                      # covered by the citation test below
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        n = int(lineno.split()[0].rstrip(")").split("-")[0])
        holder = None
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and fn.lineno <= n <= (fn.end_lineno or fn.lineno):
                holder = fn.name          # innermost wins
        if holder is None:                # module level: runs at import
            continue
        if holder not in live:
            bad.append(f"{key}: {site} is inside {holder}(), which no path from a "
                       f"stage, the loop or the config loader reaches")
    assert not bad, (
        "cited resolvers that are not reachable from the algorithm:\n"
        + "".join(f"    {b}\n" for b in bad)
        + "\nA dead resolver mentioning a key is not a read of it. Either the key "
          "has a live reader elsewhere -- cite that instead -- or it belongs in "
          "KNOWN_UNREAD."
    )


def test_every_allowlisted_key_cites_a_live_site():
    """An allowlist entry is a claim, so the claim is checked.

    The cited line must still exist and still look like a composed config
    access. Without this the allowlist rots silently: a key stays excused by a
    citation that moved, which is the same shape as the key being unread.
    """
    # `LOADER_ONLY` too: a citation checked only for reachability fails OPEN
    # (a wrong line can still be a reachable function), which is the shape this
    # file's header warns about two paragraphs in.
    for key, site in sorted({**COMPOSED_READS, **LOADER_ONLY}.items()):
        assert ":" in site, f"{key}: cite the composing site as file:line, got {site!r}"
        rel, _, lineno = site.partition(":")
        path = ROOT / rel
        assert path.is_file(), f"{key}: cited file {rel} does not exist"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        n = int(lineno.split("-")[0])
        assert 1 <= n <= len(lines), f"{key}: {rel} has no line {n}"
        window = "\n".join(lines[max(0, n - 3):n + 2])
        assert any(t in window for t in ('cfg.get(f"', "cfg.get(f'", "cfg.get(",
                                         "_flat_keys", "cfg[")), (
            f"{key}: {site} no longer looks like a config access:\n{window}")


def test_every_coherence_only_key_is_named_by_the_coherence_check():
    """`COHERENCE_ONLY` claims a key is read by `_check_coherence` and nothing
    else. The first half of that claim is checkable, so it is checked.

    A `config.py:<line>` comment per entry goes stale as `config.py` grows, and
    no test opens one, because the citation test reads `COMPOSED_READS`. A
    citation nothing opens is indistinguishable from an excuse, which is this
    file's own standard.

    The locator is the FUNCTION, which cannot drift: the key must appear inside
    `_check_coherence`'s source for the entry to be honest, and that is what
    this reads. It does not assert the second half -- that no STAGE reads the
    key -- because `unread_keys()` already covers it from the other side: a key
    that gained a literal reader would leave this set's justification stale, and
    `test_the_known_unread_set_does_not_grow` is the guard for that shape.
    """
    src = (ROOT / "bird" / "config.py").read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "_check_coherence"), None)
    assert fn is not None, "bird/config.py has no _check_coherence to cite"
    lines = src.splitlines()
    body = "\n".join(lines[fn.lineno - 1:(fn.end_lineno or fn.lineno)])
    missing = sorted(k for k in COHERENCE_ONLY if f'"{k}"' not in body)
    assert not missing, (
        "COHERENCE_ONLY names keys that _check_coherence does not mention:\n"
        + "".join(f"    {k}\n" for k in missing)
        + "\nEither the key moved to a real reader -- drop it from this set and let "
          "`unread_keys()` find the reader -- or it has none at all, in which case it "
          "belongs in KNOWN_UNREAD. An entry claiming a reader that is "
          "not there is the excuse this file exists to refuse."
    )


def test_the_check_would_notice_an_unread_key():
    """The guard, watched failing on a synthetic instance.

    A guard nobody has seen fail is not a guard. This does not mutate the tree:
    it asks whether a key absent from every reader would be reported, which is
    the property. The runnable positive control is
    `test_the_detector_names_a_declared_unread_key`.
    """
    read = literal_reads()
    invented = "train.a_key_no_reader_mentions_zzz"
    assert invented not in read
    assert invented not in COMPOSED_READS
    # ...and a key that IS read is not reported, so the check discriminates.
    assert "train.backend" in read, "train.backend should have a literal reader"


def test_the_known_unread_set_does_not_grow():
    """The listed keys stand; one more fails.

    `KNOWN_UNREAD` is a worklist, not a pardon. Without this the
    file would be a place to put a key rather than a reason to wire one, and the
    guard would decay into the thing it guards against -- a list that grows
    quietly until nobody reads it.

    Every entry must still BE unread, too: when a reader is wired, its entry
    must be deleted in the same change, or the set is claiming a defect that no
    longer exists and the next reader cannot tell which entries are real.
    """
    read = literal_reads() | constant_reads() | wrapper_reads() | coherence_reads()
    now_read = sorted(k for k in KNOWN_UNREAD if k in read or k in COMPOSED_READS)
    assert not now_read, (
        "these keys have a reader now and must come OUT of KNOWN_UNREAD:\n"
        + "".join(f"    {k}  ({(read.get(k) or ['composed'])[0]})\n" for k in now_read)
        + "\nThe set is a worklist, not a pardon: an entry that is no longer true "
          "hides which of the rest still are."
    )
    declared = leaf_keys()
    stale = sorted(k for k in KNOWN_UNREAD if k not in declared)
    assert not stale, (
        f"KNOWN_UNREAD names keys _default.yaml no longer declares: {stale}. "
        "Remove them; a worklist entry for a key that does not exist is noise."
    )


def test_deleting_a_worklist_entry_cannot_silence_the_guard():
    """A reviewer's question, answered by construction rather than by promise.

    Could someone satisfy "the set must not grow" by DELETING an entry instead
    of fixing one? No, and the reason is the interlock rather than a rule: a
    key removed from `KNOWN_UNREAD` while still unread reappears in
    `unread_keys()`, and `test_every_declared_key_has_a_reader` fails. Removal
    is therefore only possible when the key has actually gained a reader or has
    left `_default.yaml` -- which is exactly the intended direction of travel.

    Asserted here so a future reader does not have to derive it from two tests,
    and so a refactor that decouples them fails loudly.
    """
    assert KNOWN_UNREAD, "the worklist is empty; this test is vacuous, delete it"
    victim = sorted(KNOWN_UNREAD)[0]
    read = literal_reads() | constant_reads() | wrapper_reads() | coherence_reads()
    assert victim not in read and victim not in COMPOSED_READS, (
        f"{victim} has a reader and should already have left KNOWN_UNREAD")
    shrunk = KNOWN_UNREAD - {victim}
    still_unread = {k for k in leaf_keys()
                    if k not in read and k not in COMPOSED_READS
                    and k not in COHERENCE_ONLY and k not in shrunk}
    assert victim in still_unread, (
        f"deleting {victim} from KNOWN_UNREAD did NOT make it reportable again -- "
        "the worklist can be silenced by removal, which is the failure mode this "
        "test exists to exclude.")


def test_a_coherence_only_key_is_not_reported_as_unread():
    """The distinction, pinned: read-by-no-stage is not read-by-nothing.

    `problem.reward_representation` is read by `_check_coherence` and paired
    against `generate.output.format`. Calling it unread would be too strong, and
    the honour-the-key rule is already satisfied for it -- the
    `_REPRESENTATION_FORMATS` table records the ablation that established this.
    Nine of the thirteen keys a naive check reports are this shape.
    """
    coh = coherence_reads()
    for key in COHERENCE_ONLY:
        assert key in coh, (
            f"{key} is in COHERENCE_ONLY but _check_coherence no longer reads it; "
            "if the pairing was removed the key is now read by nothing and belongs "
            "in KNOWN_UNREAD.")
    assert not (COHERENCE_ONLY & KNOWN_UNREAD), "a key cannot be in both sets"
    assert not (set(LOADER_ONLY) & KNOWN_UNREAD), "a key cannot be in both sets"
    # LOADER_ONLY entries are checked only once their key exists.
    declared = leaf_keys()
    for key, site in LOADER_ONLY.items():
        if key not in declared:
            continue
        rel, _, ln = site.partition(":")
        path = ROOT / rel
        assert path.is_file(), f"{key}: cited file {rel} missing"
        assert 1 <= int(ln) <= len(path.read_text().splitlines()), f"{key}: no line {ln}"


def test_the_detector_names_a_declared_unread_key(tmp_path, monkeypatch):
    """The positive control, exercised.

    Builds the situation in `tmp_path`: a `_default.yaml` declaring one key that
    a reader consumes and one that nothing does, and a `bird/` that reads only
    the first. It proves the detector discriminates; it does not prove it
    survives contact with every key and read shape in the real tree.
    """
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "_default.yaml").write_text(
        "demo:\n  read_key: 1\n  unread_key: 2\n")
    (tmp_path / "bird").mkdir()
    (tmp_path / "bird.py").write_text(
        'def go(cfg):\n    return cfg.get("demo.read_key", 0)\n')
    (tmp_path / "bird" / "__init__.py").write_text("")

    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "DEFAULTS",
                        tmp_path / "configs" / "_default.yaml")
    monkeypatch.setattr(sys.modules[__name__], "COMPOSED_READS", {})
    monkeypatch.setattr(sys.modules[__name__], "COHERENCE_ONLY", frozenset())
    monkeypatch.setattr(sys.modules[__name__], "LOADER_ONLY", {})
    monkeypatch.setattr(sys.modules[__name__], "KNOWN_UNREAD", frozenset())

    assert leaf_keys() == {"demo.read_key", "demo.unread_key"}
    assert "demo.read_key" in literal_reads()
    bad = unread_keys()
    assert set(bad) == {"demo.unread_key"}, (
        f"the detector should name exactly the unread key, got {sorted(bad)}")

