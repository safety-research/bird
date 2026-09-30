#!/usr/bin/env python3
"""Derive a `jax_*` task spec from its CPU parent's.

    uv run python3 scripts/derive_jax_spec.py --source toy_reacher --target jax_toy \\
        --adapter bird/envs/jax_toy.py --class-name JaxToyReacher \\
        --reset-symbol _reset_pure --step-symbol _step_pure \\
        --drop-dr-axis action_noise --measure --write
    uv run python3 scripts/derive_jax_spec.py ... --check     # the drift gate

WHAT A `jax_*` SPEC IS. The same task under a different solver: the surface, the prose,
the helper vocabulary, the budget and the success reduction are the CPU spec's, and are
COPIED; everything that names a file, a line or a measured number is the jnp adapter's,
and is RE-DERIVED here -- citations by `ast` against the adapter, the random anchor and
the seeded reset ranges by running the target adapter (`--measure`), and, for a jax-suite
target, `reward.contract` (framework jax, the verifier's allowlist, generation's clause)
with the one sentence `natural_language` gains DERIVED from that contract, never typed. A `jax_*` spec is
therefore never hand-edited: change the adapter or the parent and regenerate, and
`--check` is what a test holds the committed file to (two copies of one spec drift the
first time one moves).

WHAT IS GENERIC AND WHAT IS NOT. Generic: ids, the jax stack line, every `reset.*.source`
and `reset.entry` re-pointed at the reset symbol's span (a superset of the parent's
line-level cites, which keeps `tests/test_task_specs.py`'s anchor check satisfied and is
honest about granularity -- the jnp reset is one function), step-cited notes re-pointed
at the step symbol's span, `full_source` and `reward.human` at the adapter when it
defines `reference_reward`, `symbol_mapping`'s and the helpers' `np.linalg.norm(X)`,
`np.square(X)` and `np.sum(X)` rewritten to forms that evaluate under numpy AND trace
under jnp, `env.library.commit` refused rather than inherited when `--library` renames
the library (`--stack commit=`), DR axes dropped by flag, the scripted
anchor removed (no `policies/` record has run through a new adapter), the expert anchor
recorded absent with ONE phrasing (`EXPERT_ABSENT_REASON`), provenance generated. NOT
generic, and printed as a warning rather than guessed: any other `np.<fn>` left in a
helper or symbol expression (it will not trace), and prose that names the parent's file.

WHAT A NEW SPEC OWES THE CATALOGUE BESIDES ITSELF. The generator cannot write these for
you, because each is a judgement or a measurement of its own:
  * `tests/test_task_specs.py::RESET_NAME_CHECK_ONLY` -- if the parent is SLICE-LESS (no
    `fields[].slice`), add the target: a derived spec inherits the parent's
    `spaces.obs_dim` and the adapter's raw state width, so it inherits the parent's
    unmappability, and the guard fails loudly by name otherwise.
  * `tests/conftest.py::_SIM_MARKERS` and `tests/test_task_specs.py::_SIM_PACKAGES` -- the
    adapter MODULE's rows (once per module, not per id).

Two families use it: `jax_toy` (this script's first caller, measured) and the four
`upstream_assistax_*` ids (upstream Assistax's own MJX solve, generated WITHOUT
`--measure`: their anchors are null with the reason written in), whose parents state a
`commit` and whose `symbol_mapping` carries an `np.` call that is not a norm -- two places
a parent-with-no-provenance never reaches. Importable: `derive(...)` returns the document.

WHAT `--measure` / `--check` NEED IN THE INTERPRETER: the target adapter's stack (jax at
the tier's pin, mujoco-mjx for an MJX adapter) AND `jsonschema`, which the validation gate
requires and does not skip. On a hand-built jax venv install it with pip into that
interpreter; `uv sync` there prunes the jax stack. And budget time: the anchor is measured
through the n=1 bridge, one jit dispatch per step, so `--check` on a GPU adapter is minutes
(~4 min for the anchor alone), not a pre-commit hook.

REPRODUCIBILITY OF THE MEASURED ANCHOR, because `--check` regenerates with `--measure`.
On a GPU, XLA autotuning can pick one of two kernels per compile and the VECTORISED step
of an MJX model then takes one of two values ~2e-4 apart across processes (measured on
an NVIDIA L4; `XLA_FLAGS=--xla_gpu_autotune_level=0` collapses it, 6/6 identical). A
random anchor sensitive at that decimal would make `--check` diff against its own
committed value for no reason anyone could find. Run `--measure` and `--check` under
that flag on a GPU adapter, and say so in `--measured`; `XLA_FLAGS` is process-global,
so this script sets nothing -- the caller's shell does. The toy's anchor is CPU-only and
stable across repeated checks.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

#: The one phrasing for an expert anchor a jax adapter does not have: every
#: derived spec reads the same, and a reader greps one sentence.
EXPERT_ABSENT_REASON = (
    "No scripted expert has been run through this adapter. Recorded as absent so "
    "`baselines_of` returns None and `_normalise_pool` leaves fitness raw.")

#: WHICH TIER'S WHEEL A MEASUREMENT HERE MUST HAVE BEEN TAKEN UNDER. `jax`, and it is a
#: DECLARATION rather than a description of the target: `upstream_assistax_*` derives from
#: CPU specs whose own extra is `assistax` (mujoco==3.3.0), and naming the parent's extra
#: would refuse every honest write on this tier while passing a dishonest one. The suite
#: NAME standing in for the capability is the trap -- the tier is defined by the wheel it
#: RUNS on, which is the `jax` extra's `mujoco==3.13.0` (pyproject.toml).
#:
#: Do NOT reach for `pinned_version("jax", "mujoco-mjx")`, which also answers 3.13.0
#: cleanly. A field named `mujoco` carrying mjx's version is the exact
#: field-name-is-not-its-definition defect `spec_provenance` exists to prevent, and it
#: would pass every test while being fabricated provenance.
EXTRA = "jax"

#: The argument of one `np.<fn>(...)` call: balanced parentheses to TWO levels of
#: nesting. One level was enough while `np.linalg.norm(s[a:b])` was the only shape in
#: the catalogue; the Assistax parents carry `np.sum(np.square(a))`, and rewriting the
#: inner call first leaves `np.sum(((a) ** 2))`, whose argument is itself a group
#: containing a group. Anything deeper is left and warned about rather than guessed.
_ARG = r"(?:[^()]|\((?:[^()]|\([^()]*\))*\))*"

#: `np.<fn>(X)` -> a form that gives the same number on a numpy array and traces on a
#: tracer. APPLIED INNERMOST-FIRST (square, then sum, then norm), because each rewrite
#: adds a parenthesis level to its argument and the outer pattern has to be able to
#: swallow it. All three are method/operator forms that `jax.numpy` arrays have, which
#: is the whole requirement: `symbol_mapping` and the helper expressions are RENDERED TO
#: THE MODEL (T2R's table) and evaluated under numpy by
#: `tests/test_task_specs.py::test_every_advertised_expression_evaluates`, so a surviving
#: `np.` call on a tier whose reward contract is `jnp` is a record that disagrees with
#: the run -- the render-to-the-model class, in the one field a candidate writes against.
_TRACEABLE_REWRITES = (
    (re.compile(rf"np\.square\(({_ARG})\)"), r"((\1) ** 2)"),
    (re.compile(rf"np\.sum\(({_ARG})\)"), r"((\1).sum())"),
    (re.compile(rf"np\.linalg\.norm\(({_ARG})\)"), r"(((\1) ** 2).sum() ** 0.5)"),
)
_NP_CALL_RE = re.compile(r"\bnp\.[A-Za-z_][A-Za-z_.0-9]*\(")
_RESET_SAMPLES = 24
_ANCHOR_N = 30
_ANCHOR_SEED = 0


def _span(path: Path, class_name: str, symbol: str) -> Tuple[int, int]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == symbol:
                    return m.lineno, m.end_lineno or m.lineno
    raise LookupError(f"{path}: no `{class_name}.{symbol}`")


def _defines(path: Path, class_name: str, symbol: str) -> bool:
    try:
        _span(path, class_name, symbol)
        return True
    except LookupError:
        return False


def _traceable(expr: str, where: str, warnings: List[str]) -> str:
    out = str(expr)
    for pattern, repl in _TRACEABLE_REWRITES:
        out = pattern.sub(repl, out)
    if _NP_CALL_RE.search(out):
        warnings.append(f"{where}: {out!r} still calls np.<fn>, which will not trace under "
                        "jax; rewrite it by hand in the parent or extend this script")
    return out


def _repoint_prose(text: Any, parent_files: Sequence[str], adapter_file: str,
                   span: Tuple[int, int]) -> Any:
    """`toy.py:137` / `gym_mujoco.py:997-1000` in a note -> the adapter file and the span.

    `parent_files` is the parent's `full_source` file PLUS every `--repoint-file`: a
    gym-backed parent cites the wheel and the CPU adapter, and a CPU-adapter cite
    that survives verbatim in a jax spec is a false statement about the tier."""
    if not isinstance(text, str):
        return text
    for parent_file in parent_files:
        pat = re.compile(re.escape(parent_file) + r":\d+(?:-\d+)?")
        text = pat.sub(f"{adapter_file}:{span[0]}-{span[1]}", text)
    return text


def _measure(target: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(anchors, reset ranges) through the TARGET adapter. Imports the adapter's stack."""
    import numpy as np

    from bird import registry
    from measure_anchors import ANY_STEP, PER_STEP, measure_env  # noqa: E402  (scripts/ on path)

    registry.load_all()
    anchors = measure_env(target, _ANCHOR_N, _ANCHOR_SEED)
    env = registry.get("env", target)(None)
    rows = np.stack([env.reset(np.random.default_rng(i)) for i in range(_RESET_SAMPLES)])
    twice = (env.reset(np.random.default_rng(3)), env.reset(np.random.default_rng(3)))
    import platform

    import jax

    stack = {"jax": jax.__version__, "python": platform.python_version()}
    # PROBED FROM `_VERIFIED_FROM`, not from a literal `import mujoco`. The
    # table already names which library verifies which framework, and hard-
    # coding one of its values here would let the two drift: a row naming a
    # verifier no probe measured would leave `version_verified` on the "no
    # mapping for library" warning while the table said there was one.
    # Probing every distinct verifier keeps the two in step by construction.
    #
    # `jax` is already above and `python` is not importable-by-name, so the
    # loop skips both rather than re-deriving them.
    for _lib in sorted(set(_VERIFIED_FROM.values()) - {"jax", "python"}):
        try:
            stack[_lib] = __import__(_lib).__version__
        except Exception:  # noqa: BLE001 -- absent is the normal case per tier
            pass
    devices = sorted({d.platform for d in jax.devices()})
    return ({"per_step": anchors["random"][PER_STEP], "any_step": anchors["random"][ANY_STEP],
             "ci": anchors["random"]["ci"]},
            {"rows": rows, "bitwise": bool(np.array_equal(*twice)), "stack": stack,
             "devices": devices})


#: `--drop-dr-axis`'s default reason: true of an adapter whose jitted step takes its
#: physics as arguments (jax_toy). An MJX adapter drops axes for a DIFFERENT reason --
#: `mjx.put_model` uploads the model once, so a host-side `_apply_dr` would move nothing
#: (not implemented) -- and says so with `--drop-dr-reason`.
DROP_DR_REASON = ("Derived for the jnp adapter: `{class_name}.{step}` is a pure jitted function of "
                  "(state, action, physics) and takes no key, so the {axes} axis of the parent "
                  "does not exist here.")


#: The sentence `description.natural_language` / `env_prose` gain under a jax contract --
#: DERIVED from `reward.contract.framework`, never defaulted or hand-passed:
#: `TaskSpec.instruction` falls through to `natural_language`, so this text reaches the
#: model, and a sentence there that disagreed with the contract would tell the model
#: something false. It speaks of the REWARD only; the dynamics description is the
#: parent's, because the task is the parent's (an MJX solver is not a jnp
#: reimplementation).
JAX_REWARD_SENTENCE = ("The reward is written in `jax.numpy` and traced with `jax.jit` and "
                       "`jax.vmap` (generate.reward_language: jax).")


#: Which measured package verifies a spec's `env.library.name`: a `mujoco` library is
#: verified by its mujoco, a repo-authored env (`bird`) by nothing measurable. Read twice
#: -- for `version_verified` and for which measured versions the stack RECORDS at all.
_VERIFIED_FROM = {"mujoco": "mujoco", "jax": "jax"}


def _carry_provenance(prov: Dict[str, Any], verified_by: str, default_verified_by: str,
                      date: str, is_check: bool) -> Tuple[str, str]:
    """Carry `verified_by` (both modes) and `date` (check only) from the committed file.

    CARRIED IN BOTH MODES, because guarding the carry-over with `if args.check`
    would make the field destroyed by one mode and invisible to the other:

      --write  would carry nothing, so it would stamp the argparse default over
               whoever was recorded.
      --check  copies the committed value INTO the regenerated doc before
               comparing, so it could never diff on that field.

    One mode would erase it and the other could not see it, so no run of either
    could report it -- an instrument that cannot fail, reporting success.

    Carrying on BOTH modes makes `--check`'s silence mean "unchanged" rather
    than "not looked at". An explicit `--verified-by` stays the one deliberate
    way to change it.

    THE DATE IS DELIBERATELY NOT SYMMETRIC: a `--write` re-measures, so its
    date is today's by right, and only a check needs the committed date to
    avoid diffing on today's.
    """
    if not verified_by or verified_by == default_verified_by:
        verified_by = str(prov.get("verified_by") or verified_by)
    if is_check:
        date = str(prov.get("date") or date)
    return verified_by, date


#: The command's fixed head. EMITTED AS A LITERAL, like `python3` itself, and
#: not taken from `argv[0]`: the interpreter was already normalised while the
#: script path was still the transcript, so `uv run python3 $PWD/scripts/...`
#: on the write and `scripts/...` on the check would produce two spellings of one
#: command and the file would be stale on that field forever. tests/test_jax_toy.py
#: runs the relative spelling with `cwd=REPO`, and that gate is `@pytest.mark.jax`
#: -- in NO default test selection -- so nothing else would catch the absolute form.
_INVOCATION_HEAD = "python3 scripts/derive_jax_spec.py"

#: Flags that do not survive into the recorded command, with whether each
#: takes a value. A flag belongs here when the CHECK run does not pass it and
#: the check still produces identical content -- because then the two runs
#: would record different strings for the same bytes.
#:
#: `--verified-by` IS bookkeeping, and only because of `_carry_provenance`: a
#: check that passes no flag carries the committed value and the content
#: matches, so a write that DID pass one would bake it in and go stale against
#: its own check from that write onward.
#:
#: `--stack k=v` deliberately STAYS: it changes content that is NOT carried on
#: check, so a check without it diverges on the stack block anyway and the
#: recorded command is then telling the truth about what produced the file.
#: The line is "is the content carried on check", not "is it a flag".
#: `--allow-foreign-wheel` is bookkeeping by the same rule: it gates the write, it does
#: not shape the file. What the crossing writes is `override_entries(EXTRA, installed)`,
#: which reads the INSTALLED wheel and never the flag, so a write that passes it on the
#: tier's own wheel produces content byte-identical to a bare check's -- and would then
#: have recorded a command the check cannot reproduce, stale against itself forever.
#: The crossing is not lost by dropping it: it is recorded IN THE CONTENT, as the
#: `mujoco_regenerated_under_override` stack entry.
_DROPPED_FLAGS = {"--check": False, "--write": False, "--measure": False,
                  "--date": True, "--verified-by": True,
                  "--allow-foreign-wheel": False}


def _invocation(argv: Optional[Sequence[str]] = None) -> str:
    """The command that REPRODUCES this file, canonicalised -- not a transcript.

    A hard-coded template naming only `--source` and `--target` would reproduce
    no spec the generator writes: every real invocation also needs `--adapter`,
    `--class-name` and the tier's `--drop-dr-axis`. A recorded command that does
    not reproduce is worse than none, because it is the citable artifact and a
    reader who runs it concludes the spec drifted.

    THE INVARIANT: ANYTHING that differs between the write run and the check run makes the
    file disagree with itself forever, because `--check` regenerates the
    string and diffs it. Mode flags (`--check`/`--write`/`--measure`) and
    bookkeeping whose effect the check carries anyway (`--date`,
    `--verified-by`) therefore leave; content arguments stay; and the script
    path is a literal rather than `argv[0]`.

    `argv` DEFAULTS TO `sys.argv[1:]` BUT IS A PARAMETER, because `main()`
    already accepts an argv to parse and the two must not disagree. Driven from
    a host process the global read produces
    `python3 /usr/bin/pytest -q tests/... --measure --write`, which would be
    written into a committed spec.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    out: List[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a in _DROPPED_FLAGS:
            skip_next = _DROPPED_FLAGS[a]
            continue
        if any(a.startswith(f + "=") for f in _DROPPED_FLAGS):
            continue
        out.append(a)
    out += ["--measure", "--write"]
    return _INVOCATION_HEAD + " " + " ".join(shlex.quote(a) for a in out)


def _recorded_stack(library: str, measured: Dict[str, str], date: str,
                    devices: List[str]) -> Dict[str, str]:
    """The measured versions a spec RECORDS: jax and python always, mujoco only when
    `library` is verified by it. `_measure` reports what the interpreter has, and the
    interpreter is not the adapter -- a mujoco-equipped venv running the toy must not
    write `mujoco: <ver>` into a spec whose env never imports it."""
    wanted = {"jax", "python"}
    verifier = _VERIFIED_FROM.get(library)
    if verifier:
        wanted.add(verifier)
    return {k: f"{v} (measured {date}, {'/'.join(devices)})"
            for k, v in measured.items() if k in wanted}


#: Every key a `stack:` entry can hold as a MEASURED version. Anything else in
#: that block is DECLARED -- `numpy: '>=1.24'` is a requirement, not a
#: measurement, and no filter here may touch it.
MEASURED_STACK_KEYS = frozenset({"jax", "python"}) | frozenset(_VERIFIED_FROM.values())


def _strip_foreign_measured(stack: Dict[str, Any], library: str) -> List[str]:
    """Drop measured-version keys this library does not use. Returns what it dropped.

    `_recorded_stack` filters what `--measure` ADDS. It cannot remove what is
    already in the inherited `stack:` block, so a parent that gained a
    `mujoco:` line would hand it to a simulator-less child untouched and the
    filter would be powerless -- the function is named for what a spec RECORDS
    but only ever whitelisted additions.

    WHY THIS IS NOT "FILTER THE WHOLE STACK": the block MIXES declared
    requirements with measured versions. Every committed jax spec carries
    `numpy: '>=1.24'`, which is in no `wanted` set, so whitelisting the base
    dict would silently delete a provenance line from every spec the
    generator touches -- to repair a defect that has never fired. The filter
    therefore keys on "is this a MEASURED-version key" and leaves everything
    else alone.
    """
    verifier = _VERIFIED_FROM.get(library)
    allowed = {"jax", "python"} | ({verifier} if verifier else set())
    dropped = sorted(k for k in list(stack)
                     if k in MEASURED_STACK_KEYS and k not in allowed)
    for k in dropped:
        stack.pop(k, None)
    return dropped


def _derive_contract(doc: Dict[str, Any], target: str) -> str:
    """Set `reward.contract` for a jax-suite target and return the framework.

    The env suite decides (`envs.suites.env_suite`, the rule `config._check_coherence`
    uses to FORCE `generate.reward_language: jax` on every `jax_*` env), so the record
    says what the run does. `allowed_modules` IS the verifier's jax import allowlist
    (`verification.import_allowlist("jax")`, roots), `signature` and `idiom_rule` are
    `generation`'s own clause imported, not retyped; `tests/test_jax_toy.py` holds the
    three equal so the record cannot drift from the code. A non-jax target keeps the
    parent's contract untouched.
    """
    from bird.components import generation as _gen
    from bird.components.verification import import_allowlist
    from bird.envs.suites import env_suite

    contract = ((doc.get("reward") or {}).get("contract") or {})
    if env_suite(target) != "jax":
        return str(contract.get("framework") or "")
    if not contract:
        doc.setdefault("reward", {})["contract"] = contract
    contract["signature"] = _gen._SIGNATURE
    contract["framework"] = "jax"
    contract["batched"] = False   # the candidate is one row; the harness vmaps it
    contract["allowed_modules"] = sorted(import_allowlist("jax"))
    contract["returns"] = "(scalar, components)"
    contract["idiom_rule"] = _gen._JAX_CLAUSE
    return "jax"


def _union_adapter_gate(doc: Dict[str, Any], env_name: str,
                        warnings: List[str]) -> int:
    """Union the TARGET adapter's own `consumer_forbidden_symbols` into the spec's gate.

    WITHOUT THIS THE ADAPTER'S TUPLE IS DEAD CODE, and it fails silently in the
    direction that leaks. `config.py::_inherit_from_task_spec` resolves the gate as
    `forbidden_symbols_of(spec)` and consults the factory attribute only
    `if not gate:` (line 399) -- so THE SPEC WINS whenever it supplies one, and a
    derived spec supplies its PARENT'S. A tier that inherits the CPU spec's list
    and separately declares its own bans on the factory therefore ships a gate
    that bans none of the new ones, with every test green: the factory attribute
    is set, the spec is present, and nothing compares them.

    Read off the registry rather than passed on the command line, because a
    hand-passed list is a second copy of the tuple and the two drift the moment a
    symbol is added to one. The union is also what lets the tier's gate test
    ENUMERATE the adapter's table and assert each entry is banned rather than
    assert a pasted subset -- a subset assertion cannot fail on the symbol
    somebody forgot to add.

    The import is the project venv's, not the jax venv's: an adapter that needs
    jax at MODULE level cannot be derived from here, and refusing is correct --
    writing the parent's gate instead would be the silent leak this guards.
    """
    import importlib
    try:
        registry = importlib.import_module("bird.registry")
        registry.load_all()
        factory = registry.get("env", env_name)
    except Exception as exc:                                  # noqa: BLE001
        raise SystemExit(f"--union-adapter-gate {env_name}: cannot load the adapter "
                         f"({type(exc).__name__}: {exc}). The spec would silently ship "
                         f"the parent's gate, so refusing rather than writing one.")
    extra = list(getattr(factory, "consumer_forbidden_symbols", None) or [])
    if not extra:
        raise SystemExit(f"--union-adapter-gate {env_name}: the factory declares no "
                         f"`consumer_forbidden_symbols`; nothing to union. Drop the flag "
                         f"or fix the adapter.")
    reward = doc.setdefault("reward", {})
    by_consumer = reward.setdefault("forbidden_symbols_by_consumer", {})
    before = list(by_consumer.get("bird") or [])
    seen = set(before)
    added = [sym for sym in extra if not (sym in seen or seen.add(sym))]
    by_consumer["bird"] = before + added
    prior = str(reward.get("forbidden_symbols_note") or "").rstrip()
    reward["forbidden_symbols_note"] = (
        (prior + " " if prior else "")
        + f"The `bird` list is the parent's {len(before)} symbols UNIONED with the "
          f"{len(extra)} on `registry.get('env', '{env_name}').consumer_forbidden_symbols` "
          f"({len(added)} of them new here), because the spec's gate PREEMPTS the factory "
          f"attribute (bird/config.py::_inherit_from_task_spec) and the tier's own bans would otherwise "
          f"never be applied. Regenerate rather than hand-editing: the adapter's table is "
          f"the source of truth and `--check` is the drift gate.")
    return len(added)


def derive(source: str, target: str, adapter: Path, class_name: str, reset_symbol: str,
           step_symbol: Optional[str], drop_dr: Sequence[str], library: Optional[str],
           verified_by: str, date: str, keep_scripted: bool,
           measured: Optional[Tuple[Dict[str, Any], Dict[str, Any]]],
           warnings: List[str], env_id: Optional[str] = None,
           repoint_files: Sequence[str] = (), rng_note: Optional[str] = None,
           stack_overrides: Optional[Dict[str, str]] = None,
           invocation_argv: Optional[Sequence[str]] = None,
           union_adapter_gate: Optional[str] = None,
           drop_dr_reason: Optional[str] = None,
           measured_lines: Sequence[str] = (),
           override_stack: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    src_path = REPO / "tasks" / source / "shared_spec.yaml"
    doc = yaml.safe_load(src_path.read_text())
    adapter_rel = str(adapter.relative_to(REPO)) if adapter.is_absolute() else str(adapter)
    adapter_abs = REPO / adapter_rel
    parent_file = Path(str(((doc.get("description") or {}).get("full_source") or {}).get("path")
                           or "")).name or "<parent>"
    parent_files = [parent_file] + [Path(f).name for f in repoint_files]
    reset_span = _span(adapter_abs, class_name, reset_symbol)
    step_span = _span(adapter_abs, class_name, step_symbol) if step_symbol else reset_span

    doc["id"] = target
    env = doc["env"]
    # The BENCHMARK's env key, "not unique across the catalogue" (`TaskSpec.env_id`):
    # a parent with an upstream registration keeps it (pass `--env-id <upstream id>`); a
    # repo-authored parent has none, so the target id is the default.
    env["env_id"] = env_id or target
    if library:
        env["library"]["name"] = library
    stack = dict(env["library"].get("stack") or {})
    if measured is not None and measured[1].get("stack"):
        # MEASURED off the interpreter that ran the adapter, not asserted; the device
        # list says whether "CPU" was the measurement or only the fallback.
        #
        # RECORDED ONLY FOR WHAT THE LIBRARY USES. `_measure` reports every version
        # the interpreter happens to have, and the interpreter is not the adapter:
        # a mujoco-equipped venv running the toy would otherwise write
        # `mujoco: <ver>` into a spec whose env never imports it, and `--check`
        # would then disagree between two venvs on one tree (red in a mujoco-equipped
        # venv, green in a mujoco-less one, same commit). jax and
        # python are the tier's; mujoco is recorded iff the library is verified
        # by it (`_recorded_stack`), the same table `version_verified` reads.
        _lib = str(env["library"].get("name") or "")
        # STRIP BEFORE UPDATE: the inherited block may already carry a
        # measured-version key this tier does not use (a parent gaining
        # `mujoco:` hands it to a simulator-less child), and `update()` can
        # only add. Declared entries such as `numpy: '>=1.24'` are untouched.
        for _gone in _strip_foreign_measured(stack, _lib):
            warnings.append(
                f"env.library.stack.{_gone} was inherited from the parent but "
                f"library {_lib!r} is not verified by it; dropped. A measured "
                "version belongs only to the tier that measures it.")
        stack.update(_recorded_stack(_lib, measured[1]["stack"], date, measured[1]["devices"]))
    else:
        stack.setdefault("jax", ">=0.4.37 (not yet measured: rerun with --measure)")
    # `version_verified` is the version claim a reader checks first,
    # and the parent's is the CPU library's. Under `--measure` it follows the library
    # the spec names: a `mujoco` library is verified by its mujoco; a repo-authored env stays
    # `in-repo`. A library this table does not know is warned about, not guessed.
    if measured is not None and measured[1].get("stack"):
        lib = str(env["library"].get("name") or "")
        key = _VERIFIED_FROM.get(lib)
        if key and key in measured[1]["stack"]:
            env["library"]["version_verified"] = measured[1]["stack"][key]
        elif lib not in ("bird",) and "version_verified" not in (stack_overrides or {}):
            warnings.append(f"env.library.version_verified={env['library'].get('version_verified')!r} "
                            f"is inherited from the parent and --measure has no mapping for "
                            f"library {lib!r}; pass --stack version_verified=<measured> or extend "
                            "the table")
    # `env.library.commit` IS THE LIBRARY'S, so it cannot survive a `--library` that
    # renames the library. The parent's is the CPU tier's checkout -- for an Assistax
    # parent, assistax at a7d94f4e2063 -- and carrying it across under a renamed library
    # writes that sha beside the new name, which reads as the new library's commit and
    # is simply false.
    #
    # REFUSED RATHER THAN NULLED OR WARNED, and narrowly: only when the library is
    # being renamed AND the parent states a commit AND the caller has not said what the
    # target's is. Nulling would be a silent loss of provenance; a warning in a
    # generator is read by someone already committed to writing. Passing
    # `--stack commit=null` is the deliberate statement that the target's library is a
    # wheel with no checkout -- the ASSET provenance stays where it belongs, on the
    # parent spec this one cites in `provenance.authored_from`.
    if (library and env["library"].get("commit") is not None
            and "commit" not in (stack_overrides or {})):
        raise SystemExit(
            f"refusing to write {target}: --library {library!r} renames the library but "
            f"env.library.commit is the parent's ({env['library']['commit']!r}) and nothing "
            "repoints it, so the spec would state that commit as this library's. Pass "
            "`--stack commit=<sha>` for the target's own checkout, or `--stack commit=null` "
            "when the target's library is a wheel with no checkout.")
    for k, v in (stack_overrides or {}).items():
        if k == "commit":
            env["library"]["commit"] = None if v in ("null", "none", "") else v
            continue
        if k == "version_verified":
            env["library"]["version_verified"] = v
            continue
        if measured is not None and k in (measured[1].get("stack") or {}):
            # Asserted over measured, one level up from the bug this flag fixed.
            warnings.append(f"--stack {k}={v} overrides a value --measure read off the "
                            f"interpreter ({measured[1]['stack'][k]}); dropping the flag is "
                            "the honest choice unless the interpreter is not the one that ran")
        stack[k] = v
    # `{}` on an honest run; `{OVERRIDE_KEY: "3.13.0 -> 3.3.0"}` when the caller crossed
    # wheels deliberately with --allow-foreign-wheel. Spread rather than conditioned on
    # here, so the key is present exactly when the crossing is, in both `--check` and
    # write, with no per-generator conditional to get wrong (`spec_provenance`'s own
    # rule). The note carries no timestamp, deliberately: anything a `--check` cannot
    # reproduce byte-for-byte makes the file stale against itself forever.
    stack.update(override_stack or {})
    env["library"]["stack"] = stack

    reset = env.get("reset") or {}
    reset["entry"] = {"path": adapter_rel, "symbol": f"{class_name}.{reset_symbol}",
                      "lines": f"{reset_span[0]}-{reset_span[1]}"}
    # Every draw and constant is cited at the reset symbol's span, and its prose is
    # made to NAME that symbol: `tests/test_task_specs.py`'s citation check requires
    # something the prose names (a backticked identifier, a decimal) to occur in the
    # cited lines, and a parent's `vx = vy = 0.0` anchors `0.0`, which a jnp reset
    # spells `jnp.zeros`. Naming the function the lines define is the one anchor
    # that is true of every entry, and it is the honest one -- the whole span IS
    # the citation.
    anchor = f" Written by `{reset_symbol}` in the jnp adapter."
    for key in ("draws", "constants"):
        for entry in reset.get(key) or ():
            if entry.get("source"):
                entry["source"] = {"path": adapter_rel, "lines": f"{reset_span[0]}-{reset_span[1]}"}
            for field in ("note", "value"):
                if entry.get(field):
                    entry[field] = _repoint_prose(entry[field], parent_files, Path(adapter_rel).name,
                                                  reset_span)
            dist = entry.get("distribution")
            if isinstance(dist, dict) and dist.get("text"):
                dist["text"] = _repoint_prose(dist["text"], parent_files, Path(adapter_rel).name,
                                              reset_span)
            if entry.get("source"):
                slot = "value" if key == "constants" else "note"
                text = str(entry.get(slot) or "").rstrip()
                if anchor.strip() not in text:
                    entry[slot] = (text + anchor).strip()
    if reset.get("note"):
        reset["note"] = _repoint_prose(reset["note"], parent_files, Path(adapter_rel).name, step_span)
    if rng_note is not None:
        # The mechanism CHANGED rather than moved (a jitted solver cannot call the
        # parent's reset; the draw is host-side and shipped), so the block is replaced,
        # not re-pointed.
        reset["rng_note"] = rng_note
    elif reset.get("rng_note"):
        reset["rng_note"] = _repoint_prose(reset["rng_note"], parent_files, Path(adapter_rel).name,
                                           reset_span)
        if any(f in str(reset["rng_note"]) for f in parent_files):
            warnings.append("env.reset.rng_note still names a parent file; if the reset "
                            "MECHANISM changed (host draw shipped to device), pass --rng-note")
    if drop_dr:
        reason = (drop_dr_reason or DROP_DR_REASON).format(
            class_name=class_name, step=step_symbol or reset_symbol,
            axes=", ".join(f"`{a}`" for a in drop_dr))
        reset["note"] = ((reset.get("note") or "").rstrip() + " " if reset.get("note") else "") + reason

    if measured is not None:
        rows, bitwise = measured[1]["rows"], measured[1]["bitwise"]
        flat = {f["name"]: int(f["index"]) for f in (doc["state_surface"].get("flat_fields") or ())}
        for entry in reset.get("draws") or ():
            idx = [flat[f] for f in entry.get("fields") or () if f in flat]
            if not idx:
                continue
            parts = [f"{f} in [{rows[:, flat[f]].min():.2f}, {rows[:, flat[f]].max():.2f}]"
                     for f in entry["fields"] if f in flat]
            still = [n for n, i in flat.items() if i not in idx and (rows[:, i] == rows[0, i]).all()]
            entry["note"] = (f"{_RESET_SAMPLES} seeded resets through this adapter ({date}): "
                             + "; ".join(parts) + (f"; {', '.join(still)} never move" if still else "")
                             + f"; reset(default_rng(3)) twice is {'bitwise-equal' if bitwise else 'NOT bitwise-equal'}.")
            entry["basis"] = "measured"

    desc = doc.get("description") or {}
    if desc.get("full_source"):
        desc["full_source"] = {"path": adapter_rel, "lines": None}
    framework = _derive_contract(doc, target)
    if framework == "jax":
        for key in ("env_prose", "natural_language"):
            if desc.get(key) and JAX_REWARD_SENTENCE not in desc[key]:
                desc[key] = desc[key].rstrip() + " " + JAX_REWARD_SENTENCE

    for name, expr in list((doc.get("symbol_mapping") or {}).items()):
        doc["symbol_mapping"][name] = _traceable(expr, f"symbol_mapping.{name}", warnings)
    for helper in (doc["state_surface"].get("helpers") or ()):
        if helper.get("expression"):
            helper["expression"] = _traceable(helper["expression"], f"helper {helper.get('name')}", warnings)

    anc = doc.get("anchors") or {}
    if not keep_scripted:
        anc.pop("scripted", None)
    red = anc.get("reduction")
    randoms = [b for b in (anc.get("random"),
                           ((anc.get("by_reduction") or {}).get(red) or {}).get("random")) if b]
    experts = [b for b in (anc.get("expert"),
                           ((anc.get("by_reduction") or {}).get(red) or {}).get("expert")) if b]
    for blk in experts:
        blk.update({"value": None, "method": "unavailable", "source": None, "n_seeds": None,
                    "n_episodes": None, "reason": EXPERT_ABSENT_REASON, "date": None})
    if measured is not None:
        # WHICH NUMBER IS THE ANCHOR is the spec's statement, never a default (every
        # gym spec omits `anchors.reduction`, and a default of ANY_STEP = mean
        # `success()` is identically 0.0 on a `continuous_only` family -- a plausible,
        # schema-legal, `--check`-stable WRONG number in the field `baselines_of`
        # reads). So: a family with no discrete
        # success gets the continuous metric; a declared reduction is honoured; a
        # parent that declares neither stops the run with both numbers in front of
        # the caller, who declares `anchors.reduction` on the parent.
        kind = str((doc.get("discrete_success") or {}).get("kind") or "")
        per_step, any_step = measured[0]["per_step"], measured[0]["any_step"]
        if kind == "continuous_only":
            value = per_step
        elif red == "per_step_fraction":
            value = per_step
        elif red:
            value = any_step
        else:
            raise SystemExit(
                f"refusing to write {target}: tasks/{source}/shared_spec.yaml declares no "
                f"`anchors.reduction` and discrete_success.kind={kind!r}, so the random anchor "
                f"is ambiguous -- per_step_fraction (task_metric) measured {per_step:.4f}, "
                f"any_step (success rate) measured {any_step:.4f}. Declare `anchors.reduction` "
                "on the parent; this script does not pick.")
        for blk in randoms:
            blk.update({"value": round(float(value), 4), "method": "random_policy",
                        "n_seeds": _ANCHOR_N, "n_episodes": _ANCHOR_N, "reason": None, "date": date,
                        "source": (f"MEASURED {date}: mean `task_metric` over n={_ANCHOR_N} uniform-random "
                                   f"episodes through THIS adapter, `{_invocation(invocation_argv)}` "
                                   f"(measure_anchors.measure_env, "
                                   f"seed {_ANCHOR_SEED}) -- re-measured, not copied from {source}.")})
    else:
        for blk in randoms:
            blk.update({"value": None, "method": "unavailable", "source": None, "n_seeds": None,
                        "n_episodes": None, "date": None,
                        "reason": "Not yet measured through this adapter: rerun with --measure."})

    human = (doc.get("reward") or {}).get("human") or {}
    if human and _defines(adapter_abs, class_name, "reference_reward"):
        human["reference"] = {"path": adapter_rel, "lines": None,
                              "symbol": f"{class_name}.reference_reward", "commit": None, "sha256": None}
        human["import_path"] = f"{adapter_rel[:-3].replace('/', '.')}:{class_name}.reference_reward"
    elif human:
        warnings.append(f"{class_name} defines no reference_reward; reward.human left pointing at the parent")

    if union_adapter_gate:
        _n_added = _union_adapter_gate(doc, union_adapter_gate, warnings)
        warnings.append(f"consumer gate: unioned {_n_added} tier-specific symbol(s) from "
                        f"`{union_adapter_gate}`.consumer_forbidden_symbols into "
                        f"reward.forbidden_symbols_by_consumer.bird")

    dr = doc.get("domain_randomization") or {}
    for axis in drop_dr:
        (dr.get("parameters") or {}).pop(axis, None)
        (dr.get("nominal") or {}).pop(axis, None)

    doc["provenance"] = {
        "authored_from": [
            {"path": adapter_rel, "lines": None,
             "what": f"`{class_name}` -- the observation and action tables, the symbol map, the "
                     "randomisation axes, the horizon and the success threshold, all read off the class"},
            {"path": f"tasks/{source}/shared_spec.yaml", "lines": None,
             "what": "the prose, the helper vocabulary, the budget and the success reduction, carried "
                     "over because the task is the same under a different solver"},
        ],
        # A schema ENUM (`tasks/shared_spec.schema.json`): "where did these numbers come
        # from", not "how is this file maintained" -- the generated/never-hand-edit
        # statement lives in the header lines, where a reader meets it. Folding it into
        # this string would violate the enum, and tests/test_task_specs.py (which
        # validates the enum) is slow-marked and deselected by default.
        "rule": "source, not docstrings",
        # The adapter's OWN measurements first (`--measured`, repeatable): on a tier
        # whose purpose is a solver comparison, the solver divergence against the CPU
        # parent, the reset oracle and the reproducibility switch are the facts a
        # reader is looking for; the two the script ran are boilerplate.
        "measured": (list(measured_lines)
                     + ([f"Random anchor: n={_ANCHOR_N} uniform-random episodes through the adapter, {date}.",
                         f"Reset draws: {_RESET_SAMPLES} seeded resets through the adapter, {date}."]
                        if measured is not None else [])),
        "verified_by": verified_by,
        "date": date,
    }
    return doc


def unregenerable_reason(doc: Dict[str, Any]) -> Optional[str]:
    """Why a byte-for-byte `--check` on this spec would be meaningless, or None.

    A spec whose RANDOM anchor is recorded absent has never been measured through its
    adapter. `--check` regenerates WITH `--measure`, so on a machine that can measure it
    the regenerated file necessarily carries a number where the committed one carries
    `null` -- a guaranteed diff that is not drift, and whose repair is `--measure
    --write` rather than anything a reader of the diff would guess. Reporting that as
    "STALE" is honest for a human running the script; asserting `exit 0` in a TEST is a
    permanent red on a spec that is exactly as its author intended.

    A PREDICATE OVER THE SPEC, NEVER A LIST OF IDS, and that is the whole design.
    A skip list is the deny-list shape: it says nothing
    about WHY, it has to be edited in a second place the day an id's anchors are filled,
    and the exemption outlives the condition -- the spec would go on being skipped after
    it became checkable, which is the state a byte gate exists to prevent. This reads
    the condition off the file, so it stops being true the moment `--measure --write`
    runs and no one has to remember to remove anything.

    THE EXPERT ANCHOR IS NOT CONSULTED. Every `jax_*` spec records it absent by design
    (`EXPERT_ABSENT_REASON`: no `policies/` record has run through a new adapter), so
    keying on it would make every spec in the tier permanently unregenerable -- an
    exemption that could never expire, which is the failure above with the sign
    flipped. The random anchor is the one `--measure` fills.
    """
    anchors = doc.get("anchors") or {}
    red = anchors.get("reduction")
    blocks = [b for b in (anchors.get("random"),
                          ((anchors.get("by_reduction") or {}).get(red) or {}).get("random"))
              if isinstance(b, dict)]
    if not blocks or any(b.get("value") is not None for b in blocks):
        return None
    reason = next((str(b.get("reason") or "") for b in blocks if b.get("reason")), "")
    return ("the random anchor is recorded absent" + (f" ({reason})" if reason else "")
            + ", so this spec has never been measured through its adapter and "
              "regenerating it with --measure produces a different file BY DESIGN. "
              "Byte-compare it only after `--measure --write` has filled the anchor.")


def validate_generated(doc: Dict[str, Any], out_path: Path) -> None:
    """The CONSUMER's validation, not a byte comparison: `bird.tasks._check` (what
    `tasks.load` runs) and full 2020-12 jsonschema against `tasks/shared_spec.schema.json`
    (what `tests/test_task_specs.py::test_every_spec_validates` runs).

    A generator plus a regenerating drift gate is a closed loop that agrees with
    itself through any defect it writes (a provenance.rule outside the enum, a 0.0
    anchor) -- green under `--check`, and invisible to the default test selection
    because the spec tests are slow-marked. So `--write` and
    `--check` both run THIS first, and jsonschema is REQUIRED here rather than
    skipped: a gate that skips when its validator is absent is the same loop.
    """
    import json

    from bird import tasks as _tasks

    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        # A pip into the ACTIVE interpreter, never `uv sync`: `--measure` runs in the
        # hand-built jax venvs (scripts/setup_jax.sh, .venv-jax), where `uv sync`
        # prunes the jax stack that is not in the lock. The gate itself stays strict:
        # never importorskip'd here.
        raise SystemExit("derive_jax_spec.py validates the generated spec against the schema and "
                         "needs jsonschema in THIS interpreter: "
                         f"`{sys.executable} -m pip install jsonschema` (or `uv pip install "
                         f"--python {sys.executable} jsonschema`): {exc}") from exc
    problems: List[str] = []
    try:
        _tasks._check(doc, out_path)
    except Exception as exc:  # noqa: BLE001 -- the loader's own message is the finding
        problems.append(f"tasks._check: {exc}")
    schema = json.loads((REPO / "tasks" / "shared_spec.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
        problems.append(f"schema: {'/'.join(map(str, err.path)) or '<root>'}: {err.message}")
    if problems:
        raise SystemExit("refusing to write: the generated spec does not validate:\n  - "
                         + "\n  - ".join(problems))


def render(doc: Dict[str, Any], source: str, target: str, adapter: str, drop_dr: Sequence[str]) -> str:
    header = (f"# GENERATED by scripts/derive_jax_spec.py from tasks/{source}/shared_spec.yaml -- do not\n"
              f"# hand-edit; regenerate (`--write`) and `--check` is the drift gate. `{target}` is the\n"
              f"# same task under a jax solver ({adapter}): surface, prose,\n"
              f"# budget and reduction are the parent's; citations, the random anchor and the reset\n"
              f"# ranges are RE-DERIVED through the jnp adapter"
              + (f"; DR axes dropped: {', '.join(drop_dr)}" if drop_dr else "") + ".\n"
              "# No scripted anchor: no `policies/` record has run through this id. Absence is explicit.\n")
    return header + yaml.safe_dump(doc, sort_keys=False, width=100, allow_unicode=True)


def _assert_id_round_trips(doc: Dict[str, Any], target: str, out_path: Path) -> None:
    """Refuse to write a spec whose id is not what `bird/tasks.py` would derive from it.

    `tests/test_task_specs.py` forces `tasks/<dir>/shared_spec.yaml` to carry `id: <dir>`,
    and `_BIRD_ID_RULES[library]` derives `problem.env_id` from the spec -- so a prefix
    mistake (`jax_` applied twice by a non-idempotent rule) fails HERE, at generation,
    with the three names side by side, not
    at suite time in a test that cannot say which of them is wrong.
    """
    from types import SimpleNamespace

    from bird.tasks import _BIRD_ID_RULES

    library = str(doc["env"]["library"]["name"])
    rule = _BIRD_ID_RULES.get(library)
    derived = rule(SimpleNamespace(id=doc["id"], env_id=doc["env"]["env_id"],
                                   library=library)) if rule else None
    problems = []
    if doc["id"] != target or out_path.parent.name != target:
        problems.append(f"spec id {doc['id']!r} / directory {out_path.parent.name!r} != target {target!r}")
    if derived != target:
        problems.append(f"_BIRD_ID_RULES[{library!r}] derives {derived!r} from this spec, not {target!r}"
                        + ("" if rule else " (no rule for this library: nothing would resolve the env id)"))
    if problems:
        raise SystemExit("refusing to write: " + "; ".join(problems))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="parent task id, e.g. toy_reacher")
    ap.add_argument("--target", required=True, help="jax env id / spec id, e.g. jax_toy")
    ap.add_argument("--adapter", required=True, help="adapter file, e.g. bird/envs/jax_toy.py")
    ap.add_argument("--class-name", required=True)
    ap.add_argument("--reset-symbol", default="_reset_pure")
    ap.add_argument("--step-symbol", default="_step_pure")
    ap.add_argument("--drop-dr-axis", action="append", default=[])
    ap.add_argument("--library", default=None, help="override env.library.name (default: keep)")
    ap.add_argument("--union-adapter-gate", default=None, metavar="ENV_NAME",
                    help="union registry.get('env', ENV_NAME).consumer_forbidden_symbols "
                         "into reward.forbidden_symbols_by_consumer.bird. REQUIRED for any "
                         "tier declaring bans the parent spec does not: the spec's gate "
                         "preempts the factory attribute (config.py::_inherit_from_task_spec), so without "
                         "this the adapter's tuple is a no-op")
    ap.add_argument("--env-id", default=None,
                    help="env.env_id, the BENCHMARK's key (default: the target; pass the upstream "
                         "registration, e.g. HalfCheetah-v5, when the parent has one)")
    ap.add_argument("--repoint-file", action="append", default=[],
                    help="another parent-side file whose `name.py:N` cites are re-pointed at the "
                         "adapter (repeatable; the parent's full_source file always is)")
    ap.add_argument("--rng-note", default=None,
                    help="replace env.reset.rng_note outright (the reset mechanism changed)")
    ap.add_argument("--stack", action="append", default=[], metavar="K=V",
                    help="override an env.library.stack entry (repeatable); --measure reads jax, "
                         "python and mujoco off the interpreter first. Two keys are NOT stack "
                         "entries and are special-cased onto env.library itself: "
                         "`version_verified`, and `commit` (`commit=null` for a library that is a "
                         "wheel with no checkout -- required whenever --library renames a library "
                         "whose parent states a commit)")
    ap.add_argument("--measured", action="append", default=[],
                    help="a measurement the adapter's author made that the script cannot "
                         "(solver divergence vs the CPU parent, reset oracle, ...); repeatable, "
                         "PREPENDED to provenance.measured")
    ap.add_argument("--drop-dr-reason", default=None,
                    help="why the dropped axes do not exist here; {class_name} {step} {axes} "
                         "are filled in (default: the pure-jitted-step sentence)")
    ap.add_argument("--verified-by", default="BIRD authors")
    ap.add_argument("--date", default=None, help="ISO date stamped on measurements (default: today)")
    ap.add_argument("--keep-scripted", action="store_true")
    ap.add_argument("--allow-foreign-wheel", action="store_true",
                    help="write even though the installed mujoco is not the `jax` tier's pin; "
                         "the spec then records the crossing in `mujoco_regenerated_under_override` "
                         "and tests/test_spec_mujoco_provenance.py goes red until someone "
                         "accounts for it")
    ap.add_argument("--measure", action="store_true", help="run the target adapter for anchors and resets")
    ap.add_argument("--write", action="store_true", help="write tasks/<target>/shared_spec.yaml")
    ap.add_argument("--check", action="store_true", help="regenerate (with --measure) and diff")
    # The argv ACTUALLY PARSED, handed to `_invocation` so the recorded
    # command cannot disagree with the run that produced the file. `main`
    # takes an argv, and `_invocation` must not read the global instead.
    argv_in = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv)

    import datetime as _dt
    date = args.date or _dt.date.today().isoformat()
    out_path = REPO / "tasks" / args.target / "shared_spec.yaml"
    if out_path.exists():
        prov = (yaml.safe_load(out_path.read_text()) or {}).get("provenance") or {}
        args.verified_by, date = _carry_provenance(
            prov, args.verified_by, ap.get_default("verified_by"), date, args.check)
    warnings: List[str] = []
    measured = _measure(args.target) if (args.measure or args.check) else None

    # THE WRITE GATE, AND IT IS SCOPED TO MEASUREMENT RATHER THAN TO WRITING.
    #
    # `spec_provenance` exists to stop a spec carrying numbers taken under a wheel the
    # tier does not run. Without `--measure` this script takes no numbers: the anchors
    # are written absent with a reason, the reset ranges are omitted, and every other
    # field is `ast` and prose. There is nothing for a wheel to be wrong ABOUT, and
    # `measured_mujoco()` imports mujoco, which a tree generating unmeasured specs need
    # not have -- the four `upstream_assistax_*` specs were generated without measuring. Gating that path would refuse the one case that is
    # unambiguously honest while catching nothing.
    #
    # So the gate keys on `measured is not None`, which is exactly the condition under
    # which this process ran the adapter. That is `measured_mujoco`'s own argument --
    # "measurement and emission are not separable, which is what makes this a record
    # rather than an assertion" -- applied from the other side: where they ARE separable,
    # because no measurement happened, there is no record to attribute.
    #
    # The extra is `jax` (see EXTRA above), never the CPU parent's `assistax`.
    installed: Optional[str] = None
    override_stack: Dict[str, str] = {}
    if measured is not None:
        from spec_provenance import (measured_mujoco, override_entries,
                                     refuse_foreign_write)
        installed = measured_mujoco()
        refusal = refuse_foreign_write(EXTRA, installed, args.allow_foreign_wheel)
        if refusal:
            raise SystemExit(refusal)
        override_stack = override_entries(EXTRA, installed)

    stack_overrides = dict(kv.split("=", 1) for kv in args.stack)
    doc = derive(args.source, args.target, Path(args.adapter), args.class_name, args.reset_symbol,
                 args.step_symbol, args.drop_dr_axis, args.library,
                 args.verified_by, date, args.keep_scripted, measured, warnings,
                 env_id=args.env_id, repoint_files=args.repoint_file, rng_note=args.rng_note,
                 stack_overrides=stack_overrides, drop_dr_reason=args.drop_dr_reason,
                 measured_lines=args.measured, invocation_argv=argv_in,
                 override_stack=override_stack,
                 union_adapter_gate=args.union_adapter_gate)
    text = render(doc, args.source, args.target, args.adapter, args.drop_dr_axis)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    _assert_id_round_trips(doc, args.target, out_path)
    validate_generated(yaml.safe_load(text), out_path)

    if args.check:
        have = out_path.read_text() if out_path.exists() else ""
        if have == text:
            print(f"{out_path.relative_to(REPO)}: up to date")
            return 0
        # A CHECK UNDER A FOREIGN WHEEL SAYS SO BEFORE ANY STALE LIST. The rebuild
        # legitimately differs there (every write under that wheel records the override
        # key), and "stale" names a defect whose obvious repair is to regenerate --
        # which would stamp a deliberate-crossing marker across the tier.
        #
        # Reachable only under `--allow-foreign-wheel`: a plain `--check` on a foreign
        # wheel has already been refused above. That is a DEPARTURE from the
        # `gen_*_specs.py` generators, which note and continue, and it is right for this script
        # rather than for them -- they walk a whole tier and must carry on past a spec
        # they cannot judge, where this takes one target per invocation and has nothing
        # to carry on to.
        note = None
        if installed is not None:
            from spec_provenance import check_under_foreign_wheel as _cufw
            note = _cufw(EXTRA, installed)
            if note:
                print(note, file=sys.stderr)
        # SAY WHY BEFORE SHOWING THE DIFF. A spec whose random anchor is recorded
        # absent regenerates to a different file by design, and the diff alone does
        # not say so -- a reader sees `value: null` against a number and has no way
        # to tell "never measured" from "drifted".
        why = unregenerable_reason(yaml.safe_load(have) or {}) if have else None
        if why:
            print(f"{out_path.relative_to(REPO)}: NOT BYTE-COMPARABLE -- {why}",
                  file=sys.stderr)
        sys.stdout.writelines(difflib.unified_diff(have.splitlines(True), text.splitlines(True),
                                                   str(out_path.relative_to(REPO)), "regenerated"))
        # THE LAST LINE MUST NOT INSTRUCT WHAT THE NOTE FORBADE. Under a foreign wheel
        # `check_under_foreign_wheel` says "Do NOT regenerate to clear it", because every
        # write under that wheel stamps the override marker; ending with "rerun with
        # --measure --write" would tell the reader to do exactly that, and the last line
        # is the one a reader acts on. Conditional on the NOTE rather than on a flag: the note is
        # present exactly when regenerating is the harmful repair, which is the condition,
        # where `--allow-foreign-wheel` is only how one gets there.
        if note:
            print(f"\n{out_path.relative_to(REPO)} differs from what THIS wheel would "
                  "write. That is expected under a foreign wheel and is not a defect in "
                  "the spec -- see the note above, and do not regenerate to clear it.",
                  file=sys.stderr)
        else:
            print(f"\n{out_path.relative_to(REPO)} is STALE; rerun with --measure --write",
                  file=sys.stderr)
        return 1
    if args.write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"wrote {out_path.relative_to(REPO)}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
