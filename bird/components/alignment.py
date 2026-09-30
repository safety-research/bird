"""Modular reward surgery and preference-fitted parameters (R*, ICML 2025).

Four registry families live here, and they are the three things R* has that
no other reproduced method has:

  crossover_operator          module-level crossover that costs NO API call
  crossover_parent_selection  softmax over archive fitness
  segment_labeller            a population of LLM-WRITTEN critic programs, voting
  param_alignment             Bradley-Terry fit of a reward's own numeric parameters

Why they are here and not in `evolution.py`: every crossover already in this
repo is a PROMPT. REvolve's picks two parents and asks the model to recombine
them; RF-Agent's `crossover_elite` shows the model an elite set. R*'s crossover
is an AST edit -- "No extra API calls are required in this process"
(App. A, p.12) -- and that sentence is the whole reason 4 of its 16 individuals
are free. A prompt-shaped module cannot hold a mechanism whose defining property
is that it never prompts.

WHAT THE PAPER DOES NOT SAY, AND WHAT THIS FILE THEREFORE DECIDES. Each is
marked in the config and repeated here, because an undocumented decision in an
optimiser reads exactly like a published one:

  * WHICH NUMBERS ARE `psi`. The paper says "the parameters within the reward
    module and the weights between modules" (S2, p.2) and never says how a
    tunable coefficient is told apart from an array index. `tunable:
    float_literals` is our rule: every FLOAT literal in the program is a
    parameter, every int literal is not. That separates `2.0 * dist` (a
    coefficient) from `obs[3]` (an index) with no heuristics and no config per
    task, and it is stated rather than inferred.
  * WHICH OPTIMISER. The paper gives the objective (Eq. 2) and a budget ("1000
    iterations of optimization on all training data", App. A) and never names a
    method. Finite-difference descent costs one objective evaluation per
    parameter per step; SPSA costs two regardless of dimension, so at the 13-18
    parameters a mock candidate exposes it is 7-9x fewer re-executions of the
    reward. SPSA it is, and it is a `+-` perturbation of the same loss, not a
    different loss. An objective evaluation is one re-execution of the program
    over every labelled segment and nothing else that scales with the step
    count (`_Parametrised`): measured on the tester tier, ~9 s per candidate
    at 1000 steps and 20 pairs, two thirds of it inside the candidate's own
    Python.
  * HOW THE 70/30 SPLIT IS DRAWN. The paper says "70% as the training set and
    30% as the validation set" (App. A) and nothing about the draw. The
    labelled buffer is not in a neutral order -- `bird.py` extends it once per
    iteration and the labeller emits each batch rung-major, strictest
    consensus first -- so a positional cut would validate on the newest and
    least-agreed segments only. The
    pairs are shuffled with `ctx.rng` once per alignment call before the cut
    (`_split_preferences`): a random 70/30 of the whole buffer, reproducible
    from `seed`, and `meta['alignment']['split']` says so.
"""

from __future__ import annotations

import ast
import builtins
import math
import random
import re
from types import CodeType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import reward_language
from ..registry import register, get as _get
from ..types import Candidate, CandidateReport, Preference, Trajectory


def _unparse(node: Any) -> str:
    """`ast.unparse`: the one place this module renders an AST back to source.

    Every consumer re-parses or compiles the result -- `CompiledReward` compiles
    from the unparsed text -- so the crossover (R*'s module insert) and the
    `generate.alignment.tunable: float_literals` rewrite both go through here.
    """
    return ast.unparse(node)


# ==========================================================================
# AST: the reward program as a dict of modules
# ==========================================================================
#
# "For the modularization of code, we use the Abstract Syntax Tree (AST) to
# decompose the reward functions based on the returned reward dictionary. This
# process returns the code blocks corresponding to each reward component"
# (App. A, p.12). A module is therefore a returned dict ENTRY plus the
# statements that exist only to compute it -- not the whole function body, or a
# crossover would carry the recipient's own components in with it.


def _reward_fn_node(tree: ast.Module) -> Optional[ast.FunctionDef]:
    """The function a reward program defines, by the same rule `compile_reward`
    uses: a known name first, else the last one defined."""
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fns:
        return None
    for probe in ("compute_reward", "reward", "reward_fn", "get_reward",
                  "dense_reward"):
        for fn in fns:
            if fn.name == probe:
                return fn
    return fns[-1]


def _own_nodes(fn: ast.FunctionDef) -> List[ast.AST]:
    """Every node of `fn` EXCEPT those inside a nested function.

    `ast.walk` descends into nested `def`s, and a generated reward routinely
    carries a helper -- the mock's `_f`, a real model's `_norm` -- whose own
    `return` is then the last one `ast.walk` yields. Taking that as the reward's
    return finds no component dict, `split_modules` returns no modules, and
    crossover refuses every insert while reporting only `crossover_short`: a
    mechanism that is configured, logged, and silently contributes nothing
    (0 offspring in every round on the tester tier).
    """
    out: List[ast.AST] = []
    stack: List[ast.AST] = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop(0)
        out.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return out


def _returned_dict(fn: ast.FunctionDef) -> Optional[ast.Dict]:
    """The dict the function returns, whether returned inline or via a name.

    Both shapes appear in generated code: `return {"a": x}` and
    `rew = {"a": x}` / `return total, rew`. Missing one of them would silently
    make a candidate un-modular -- i.e. quietly drop it from crossover.
    """
    ret = None
    for node in _own_nodes(fn):
        if isinstance(node, ast.Return):
            ret = node
    if ret is None or ret.value is None:
        return None
    targets: List[ast.expr] = []
    if isinstance(ret.value, ast.Dict):
        return ret.value
    if isinstance(ret.value, (ast.Tuple, ast.List)):
        targets = list(ret.value.elts)
    else:
        targets = [ret.value]
    names = [t.id for t in targets if isinstance(t, ast.Name)]
    for node in reversed(list(_own_nodes(fn))):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names:
                    return node.value
    for t in targets:
        if isinstance(t, ast.Dict):
            return t
    return None


def _names_read(node: ast.AST) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)}


def _assign_targets(stmt: ast.stmt) -> set:
    out: set = set()
    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            out |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)) and stmt.target is not None:
        out |= {n.id for n in ast.walk(stmt.target) if isinstance(n, ast.Name)}
    return out


class Module:
    """One reward component: its dict key, its value expression, and the
    statements that feed it and nothing else."""

    def __init__(self, key: str, value: ast.expr, stmts: List[ast.stmt]) -> None:
        self.key = key
        self.value = value
        self.stmts = stmts

    def source(self) -> str:
        body = "\n".join(_unparse(s) for s in self.stmts)
        return (body + "\n" if body else "") + f"{self.key!r}: {_unparse(self.value)}"


def split_modules(code: str) -> Tuple[Optional[ast.Module], List[Module]]:
    """Decompose a reward program into modules keyed by the returned dict.

    A statement is attributed to a module when that module's expression reads a
    name the statement binds AND no other module's expression reads it,
    transitively. An assignment two modules share stays shared -- moving it
    would change the donor's meaning and the recipient's both.

    Worked example. For

        d = norm(hand - obj)                # read by both -> shared, never moved
        near = exp(-2.0 * d)                # read by "reach" only  -> reach's
        lifted = obj_z - table_z            # read by "lift" only   -> lift's
        return {"reach": near, "lift": 0.5 * lifted}

    `split_modules` returns two Modules: reach = [`near = ...`] + `near`, and
    lift = [`lifted = ...`] + `0.5 * lifted`. Inserting lift into another
    program carries `lifted = ...` with it and leaves `d = ...` behind, which is
    why the recipient must already define `d` or the insert is rejected below.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, []
    fn = _reward_fn_node(tree)
    if fn is None:
        return None, []
    rd = _returned_dict(fn)
    if rd is None:
        return tree, []

    keys: List[str] = []
    values: List[ast.expr] = []
    for k, v in zip(rd.keys, rd.values):
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            keys.append(k.value)
            values.append(v)
    if not keys:
        return tree, []

    body = [s for s in fn.body if not isinstance(s, ast.Return)]
    # transitive closure of the names each module's expression depends on
    deps: List[set] = []
    for v in values:
        want = _names_read(v)
        changed = True
        while changed:
            changed = False
            for stmt in body:
                tgts = _assign_targets(stmt)
                if tgts & want:
                    new = _names_read(stmt) - want
                    if new:
                        want |= new
                        changed = True
        deps.append(want)

    mods: List[Module] = []
    for i, (k, v) in enumerate(zip(keys, values)):
        mine: List[ast.stmt] = []
        for stmt in body:
            tgts = _assign_targets(stmt)
            if not tgts:
                continue
            if not (tgts & deps[i]):
                continue
            if any(tgts & deps[j] for j in range(len(keys)) if j != i):
                continue  # shared: belongs to no single module
            mine.append(stmt)
        mods.append(Module(k, v, mine))
    return tree, mods


def _return_stmt(fn: ast.FunctionDef) -> Optional[ast.Return]:
    ret = None
    for node in _own_nodes(fn):
        if isinstance(node, ast.Return):
            ret = node
    return ret


def _total_composition(fn: ast.FunctionDef, ret: ast.Return, rd: ast.Dict) -> str:
    """How the recipient's SCALAR total relates to its returned dict, i.e. what
    inserting a module into the dict does to the reward PPO actually trains on.

    `CompiledReward._unpack` reads `out[0]` as the total whenever the reward
    returns `(total, dict)`, and sums the dict only when there is no total. So
    appending a module to the dict is the whole edit ONLY for a recipient whose
    total is computed FROM the dict; for one that spells its total out --
    `total = reach + grasp; return total, {...}` -- the dict advertises a module
    the policy never sees. Four shapes:

      dict_only   `return {...}`: the total IS the dict sum (`_unpack`).
      dict_sum    `return sum(parts.values()), parts` or any total whose
                  transitive read-set reaches a name bound to the dict: the
                  amended dict already carries the module into the total.
      add         an explicit total that never reads the dict (a name, an
                  arithmetic expression, an inline dict beside it): the donor's
                  value has to be ADDED to it, which is Eq. 3's `+ M_j,m`.
      unsupported anything else -- a bare name returned that is not a 2-tuple,
                  a starred total. Refused rather than guessed.
    """
    v = ret.value
    if isinstance(v, ast.Dict):
        return "dict_only"
    if not isinstance(v, (ast.Tuple, ast.List)) or len(v.elts) != 2:
        return "unsupported"
    total_expr = v.elts[0]
    if isinstance(total_expr, ast.Starred):
        return "unsupported"
    if v.elts[1] is rd:
        return "add"  # an inline dict: no earlier statement can have read it
    dict_names = {t.id for st in _own_nodes(fn)
                  if isinstance(st, ast.Assign) and st.value is rd
                  for t in st.targets if isinstance(t, ast.Name)}
    body = [s for s in fn.body if not isinstance(s, ast.Return)]
    want = _names_read(total_expr)
    changed = True
    while changed:
        changed = False
        for stmt in body:
            if _assign_targets(stmt) & want:
                new = _names_read(stmt) - want
                if new:
                    want |= new
                    changed = True
    return "dict_sum" if want & dict_names else "add"


def _fresh_name(key: str, taken: set) -> str:
    base = re.sub(r"\W", "_", str(key)).strip("_") or "module"
    name = f"{base}_module"
    while name in taken or name in dir(builtins):
        name += "_"
    return name


def module_insert(recipient: str, donor: str, donor_key: str) -> Optional[str]:
    """R*'s Eq. 3: `F_new = {M_i,1, ..., M_i,m, M_j,m}`. The code, or None.

    `module_insert_detail` is the same operation reporting HOW the donor's value
    reached the recipient's total (`_total_composition`); this wrapper exists
    for callers that want only the program.
    """
    out = module_insert_detail(recipient, donor, donor_key)
    return out[0] if out else None


def module_insert_detail(recipient: str, donor: str,
                         donor_key: str) -> Optional[Tuple[str, str]]:
    """R*'s Eq. 3: `F_new = {M_i,1, ..., M_i,m, M_j,m}`.

    One module of the donor is added to the recipient -- to its returned dict
    AND to the scalar it returns, which `_total_composition` decides how to do.
    Returns `(code, composition)`, or None -- a REFUSAL, not a silent partial
    edit -- when the donor module reads a name the recipient does not define
    (a program that compiles and raises NameError at step 1 of training is a
    wasted RL run reported as a bad reward), or when the recipient's total is a
    shape this function cannot amend (`unsupported`). Amending only the dict
    would leave a recipient with an explicit total training on exactly its own
    reward while the artifact advertised a recombined one.
    """
    r_tree, r_mods = split_modules(recipient)
    d_tree, d_mods = split_modules(donor)
    if r_tree is None or d_tree is None or not r_mods or not d_mods:
        return None
    donor_mod = next((m for m in d_mods if m.key == donor_key), None)
    if donor_mod is None:
        return None
    if donor_mod.key in {m.key for m in r_mods}:
        return None  # already present: an insert that changes nothing is not one

    r_fn = _reward_fn_node(r_tree)
    if r_fn is None:
        return None
    defined = set(a.arg for a in r_fn.args.args)
    for stmt in r_fn.body:
        defined |= _assign_targets(stmt)
        # A generated reward usually carries a nested helper (`_f`, `_norm`),
        # and a `def` is not an Assign -- so without this every donor module
        # that calls one is refused, which is most of them.
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(stmt.name)
    defined |= {"np", "numpy", "math"}
    needed = _names_read(donor_mod.value) | set().union(
        *[_names_read(s) for s in donor_mod.stmts]) if donor_mod.stmts \
        else _names_read(donor_mod.value)
    bound_by_donor = set().union(*[_assign_targets(s) for s in donor_mod.stmts]) \
        if donor_mod.stmts else set()
    # `dir(builtins)`, NOT `dir(__builtins__)`: inside an imported module
    # `__builtins__` is the builtins DICT, so `dir()` of it returns dict methods
    # (`keys`, `items`, ...) and every real builtin a donor uses -- `min`, `abs`,
    # `sum` -- reads as missing. Every insert is then refused and crossover
    # silently contributes nothing while reporting `crossover_short`.
    missing = needed - defined - bound_by_donor - set(dir(builtins))
    if missing:
        return None

    rd = _returned_dict(r_fn)
    if rd is None:
        return None
    # INSERT BEFORE THE STATEMENT THAT BUILDS THE DICT, not before the return.
    # A reward that assigns `components = {...}` and then `return total,
    # components` has two statements after the dict; inserting there puts the
    # donor's own setup AFTER the expression that reads it, so the module works
    # only when the recipient happens to bind the same name -- and silently
    # computes the recipient's value rather than the donor's when it does.
    # For example, the donor's `effort = 0.0` would land after
    # `total = sum(...)` as dead code while `'ctrl': -33.9 * effort` read the
    # RECIPIENT's effort.
    anchor = next((i for i, st in enumerate(r_fn.body)
                   if any(n is rd for n in ast.walk(st))), len(r_fn.body))
    # A GRAFTED MODULE IS EVALUATED IN THE RECIPIENT'S CONTEXT. Where the donor
    # binds a name the recipient already binds -- `speed`, `dist`, whatever both
    # happened to call it -- the donor's statement is DROPPED and the
    # recipient's value is used. Inserting it instead would overwrite a name the
    # recipient's OWN modules read, so a crossover meant to add one component
    # would silently change the others. That is a real semantic choice, not a
    # detail: R* states only "a reward module from one parent is incorporated
    # into another function" (App. A, p.12), and this is the reading that leaves
    # the recipient's existing modules computing what they computed before.
    kept = [st for st in donor_mod.stmts if not (_assign_targets(st) & defined)]
    ret = _return_stmt(r_fn)
    if ret is None or ret.value is None:
        return None
    composition = _total_composition(r_fn, ret, rd)
    if composition == "unsupported":
        return None
    for off, stmt in enumerate(kept):
        r_fn.body.insert(anchor + off, stmt)
    rd.keys.append(ast.Constant(value=donor_mod.key))
    if composition == "add":
        # Bind the donor's value ONCE and read it twice -- in the dict, and
        # added to the returned total -- so a module with side effects or a
        # random draw cannot report one number and train on another.
        name = _fresh_name(donor_mod.key, defined | bound_by_donor | needed)
        r_fn.body.insert(anchor + len(kept),
                         ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                                    value=donor_mod.value))
        rd.values.append(ast.Name(id=name, ctx=ast.Load()))
        ret.value.elts[0] = ast.BinOp(left=ret.value.elts[0], op=ast.Add(),
                                      right=ast.Name(id=name, ctx=ast.Load()))
    else:
        rd.values.append(donor_mod.value)
    ast.fix_missing_locations(r_tree)
    return _unparse(r_tree), composition


# ==========================================================================
# crossover_parent_selection  --  fn(ctx, state, n) -> list[CandidateReport]
# ==========================================================================


@register("crossover_parent_selection", "none")
def parents_none(ctx: Any, state: Any, n: int) -> List[CandidateReport]:
    """No crossover parents. The default, and inert."""
    return []


@register("crossover_parent_selection", "softmax_fitness")
def parents_softmax_fitness(ctx: Any, state: Any, n: int) -> List[CandidateReport]:
    """"For the crossover operator, we use a softmax function based on the
    success rates to select parent individuals" (App. A, p.12).

    Draws WITHOUT replacement within one pair and WITH replacement across pairs:
    a crossover of a reward with itself is not one, but two pairs may legitimately
    share a parent -- REvolve's `archive_sample` makes the same distinction for
    the same reason.

    A report whose fitness is None (CARD computes no ranking scalar) is excluded
    rather than treated as 0.0: `.fitness is None` must stay distinct from 0.0
    everywhere, and a softmax over a fabricated zero would quietly make the
    unranked the least likely parent rather than an ineligible one.
    """
    pool = [r for r in _archive_reports(state) if r.fitness is not None]
    if len(pool) < 2 or n <= 0:
        return []
    # No `or 1.0`: a legal 0 is GREEDY -- the clamp below is what makes it run
    # -- and `or` would replace it with the plain softmax, so `-s
    # generate.crossover.temperature=0` would move the hash and still draw the
    # default's parents. A negative value is refused at load.
    temp = float(ctx.cfg["generate.crossover.temperature"])
    fits = np.array([float(r.fitness) for r in pool], dtype=float)
    # shift before exp: fitness is a success rate in [0, 1] here, but nothing in
    # the schema says so, and exp() of a large positive fitness overflows to inf
    # -- which selects that parent with probability 1 and reads as a bug in the
    # search rather than in this line.
    z = (fits - fits.max()) / max(temp, 1e-9)
    w = np.exp(z)
    w = w / w.sum()
    out: List[CandidateReport] = []
    for _ in range(n):
        i, j = _sample_two(ctx, w)
        out.extend([pool[i], pool[j]])
    return out


def _sample_two(ctx: Any, w: np.ndarray) -> Tuple[int, int]:
    """Two DISTINCT indices, drawn with probability proportional to `w`.

    `ctx.rng` is a `random.Random` -- the run's one reproducible stream -- not a
    numpy Generator, so there is no `choice(..., replace=False, p=...)` to call
    here. Efraimidis-Spirakis instead: draw a key u**(1/w) per item and take the
    two largest, which is exactly weighted sampling without replacement in one
    pass and consumes a fixed number of draws from the stream whatever the pool
    size -- a rejection loop would not, and the tester tier's determinism test
    compares whole run directories.
    """
    keys = []
    for idx, weight in enumerate(w):
        u = ctx.rng.random()
        wt = float(weight)
        keys.append((u ** (1.0 / wt) if wt > 0 else -1.0, idx))
    keys.sort(reverse=True)
    return int(keys[0][1]), int(keys[1][1])


def _archive_reports(state: Any) -> List[CandidateReport]:
    """Everything trained so far, whatever slot is carrying it.

    R*'s `D_F` is "the reward function archive" and holds every evaluated
    individual (S4.2) -- which is `state.all_reports`, not `state.archive`.
    """
    seen: Dict[str, CandidateReport] = {}
    # `all_reports` IS D_F: every candidate ever evaluated in this run, in
    # order, kept whatever `loop.carry` says. The MAP-Elites `archive` is a
    # different object and would be the wrong one -- it holds at most one
    # individual per descriptor cell, so reading it as the archive would make
    # R*'s parent pool silently a function of LIMEN's binning.
    for rep in getattr(state, "all_reports", []) or []:
        if isinstance(rep, CandidateReport):
            seen.setdefault(rep.cand_id, rep)
    arch = getattr(state, "archive", None)
    if arch:
        for cell in (arch.values() if isinstance(arch, dict) else arch):
            rep = getattr(cell, "report", cell)
            if isinstance(rep, CandidateReport):
                seen.setdefault(rep.cand_id, rep)
    return list(seen.values())


# ==========================================================================
# crossover_operator  --  fn(ctx, state, candidates) -> list[Candidate]
# ==========================================================================


@register("crossover_operator", "none")
def crossover_none(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """No crossover. The default: `generate.crossover.n` is 0 everywhere else."""
    return list(candidates)


@register("crossover_operator", "module_insert")
def crossover_module_insert(ctx: Any, state: Any,
                            candidates: List[Candidate]) -> List[Candidate]:
    """Top the pool back up to `n_candidates` with programmatic crossovers.

    Called AFTER the sampler, which was asked for the wave's `llm` count
    (`generation._this_wave`), or `n_candidates - crossover.n` when `loop.waves`
    is empty (`generation._n_llm_this_iteration`). Two counts are produced here
    and they are different:

      * the SHARE -- `generate.crossover.n`, R*'s `n_c` = 4 of 16 (App. A);
      * the BACK-FILL -- "If the reward function generation by the LLM
        encounters runtime failures ... the failed individuals are supplemented
        with those generated via crossover ... No extra API calls are required"
        (App. A, p.12), under `generate.crossover.backfill_failures`.

    The back-fill counts candidates already marked invalid. At this point in the
    stage only a generation-time failure is visible (an empty or unparseable
    completion); a candidate that fails S2's checks is screened later and is NOT
    re-filled here, because S2 has not run yet. That is a deviation from a paper
    whose "compilation error" is caught in one place, and it is recorded in
    `configs/methods/rstar.yaml` rather than papered over.

    Every offspring costs zero LLM calls, so `ctx.budget` records none -- which
    is the point of the mechanism and is checked by `budget.json` carrying more
    trainings than generation calls.
    """
    from .generation import _this_wave
    wave = _this_wave(ctx, state)
    want = (int(wave["crossover"]) if wave["crossover"] is not None
            else int(ctx.cfg["generate.crossover.n"] or 0))
    if ctx.cfg["generate.crossover.backfill_failures"]:
        want += sum(1 for c in candidates if not c.valid or not c.reward_code.strip())
    if want <= 0:
        return list(candidates)

    select = _get("crossover_parent_selection",
                  ctx.cfg["generate.crossover.parent_selection"])
    # "This process is repeated until the required number of individuals is
    # generated" (App. A, p.12). An insert can legitimately refuse -- the donor
    # module's key is already present, or it reads a name the recipient does not
    # define -- so one draw per offspring silently under-fills the population.
    # `_XOVER_ATTEMPTS` bounds the retry so a pool where EVERY pair refuses
    # (two archetypes sharing all their component names, which is the tester
    # tier) terminates and reports `crossover_short` instead of spinning.
    parents = select(ctx, state, want * _XOVER_ATTEMPTS)
    if not parents:
        # Iteration 1 has no archive; the paper's two-step answer runs here as
        # the `no_archive` wave of `loop.waves` (configs/methods/rstar.yaml), which
        # supplies this iteration's crossover share after the LLM individuals
        # are evaluated, so a short pool here is not an error. The back-fill
        # count above is per wave: failures counted here are NOT carried into
        # the `no_archive` wave.
        ctx.event("crossover_skipped", reason="empty_archive", wanted=want)
        return list(candidates)

    out = list(candidates)
    made = 0
    for i in range(0, len(parents) - 1, 2):
        if made >= want:
            break
        recipient, donor = parents[i], parents[i + 1]
        r_code = recipient.candidate.reward_code
        d_code = donor.candidate.reward_code
        _, d_mods = split_modules(d_code)
        if not d_mods:
            continue
        key = str(ctx.rng.choice([m.key for m in d_mods]))
        detail = module_insert_detail(r_code, d_code, key)
        if detail is None:
            # Visible, not silent: a refused graft is a pair the operator could
            # not recombine, and the shortfall it causes is only legible if the
            # journal says which pair and which module.
            ctx.event("crossover_refused", recipient=recipient.cand_id,
                      donor=donor.cand_id, module=key)
            continue
        merged, composition = detail
        cand = Candidate(
            # `x` prefix: a crossover offspring is not a sample, and the id is
            # what the run directory, the journal and every plot key on. An
            # offspring sharing the sampler's `cNNNN` scheme would be
            # indistinguishable from an LLM call that was never made.
            cand_id=f"x{state.iteration:02d}-{made:02d}",
            iteration=int(state.iteration),
            reward_code=merged,
            parent_id=recipient.cand_id,
        )
        # `produced_by` says how the CANDIDATE was built -- here by module
        # insertion from a donor, not by an LLM sample.
        cand.meta["produced_by"] = "module_insert"
        cand.meta["crossover_donor"] = donor.cand_id
        cand.meta["crossover_module"] = key
        # How the donor's value reached the TOTAL, not just the dict
        # (`_total_composition`): `add`, `dict_sum` or `dict_only`.
        cand.meta["crossover_total"] = composition
        out.append(cand)
        made += 1
        ctx.event("crossover", cand_id=cand.cand_id, recipient=recipient.cand_id,
                  donor=donor.cand_id, module=key, total=composition)
    if made < want:
        ctx.event("crossover_short", wanted=want, made=made)
    return out


# ==========================================================================
# segment_labeller  --  fn(ctx, state) -> list[Preference]
# ==========================================================================
#
# "we propose the critic-based comparison voting mechanism, which constructs a
# population of rule-based critic functions for step-wise comparison" (S4.3,
# p.5). The LLM writes the judge ONCE; the judge then runs on every step of
# every pair for free. That is the difference from every comparator in
# `preferences.py`, where the model IS the judge and each comparison is a call.
#
# THE CRITIC CONTRACT IS OURS, not the paper's. Prompt 2 (p.14, item 2) shows
# `traj_a[key][:,3]` indexing and Prompt 1 (p.13) asks for a `label_list`, which fixes the
# shape -- a dict of named arrays in, one label per step out -- but the paper's
# own signature is bound to Isaac Gym tensors. Ours:
#
#     def compare(traj_a, traj_b) -> list[int]
#         traj_a, traj_b : dict[str, np.ndarray], each array (T, ...)
#         returns        : T labels in {1, 0, -1}; 1 = a better at that step
#
# Keys are the env's state fields plus every reward component recorded for that
# rollout, so a critic can compare either. `_traj_dict` builds them and is the
# only place that mapping exists.


#: Parent PAIRS drawn per offspring wanted, before giving up. Not a config
#: key: the paper says "repeated until", i.e. unbounded, and a bound nobody
#: published is not a pin -- it is a guard against a pool that cannot
#: recombine at all.
_XOVER_ATTEMPTS = 8


_CRITIC_NAMES = ("compare", "critic", "compare_trajectories", "label")


def _compile_critic(code: str, tag: str) -> Optional[Any]:
    """A critic is arbitrary generated code, compiled exactly the way a reward
    is (`training.compile_reward`): same namespace, no sandbox, because
    `verify.forbidden_symbols` is this repo's anti-hacking gate and it runs
    statically. A critic that will not compile is DROPPED, not fatal, and never
    replaced: the ladder's rungs stay as configured over the survivors. The
    paper's ladder relaxes the threshold over a FULL population of five and says
    nothing about population shrinkage (configs/methods/rstar.yaml)."""
    ns: Dict[str, Any] = {"np": np, "numpy": np, "math": math}
    try:
        exec(compile(str(code), f"<critic:{tag}>", "exec"), ns, ns)
    except Exception:  # noqa: BLE001 - a bad critic is data, not a crash
        return None
    for probe in _CRITIC_NAMES:
        fn = ns.get(probe)
        if callable(fn):
            return fn
    for key, value in reversed(list(ns.items())):
        if key.startswith("__") or key in ("np", "numpy", "math"):
            continue
        if callable(value):
            return value
    return None


def _state_field_names(ctx: Any) -> List[str]:
    """The environment's state fields, in `s[i]` order -- `EnvAdapter._state_fields`,
    the `(name, doc)` pairs a task spec fills (`envs/spec.py`). `[]` for an env
    that declares none."""
    env = getattr(ctx, "env", None)
    fields = getattr(env, "_state_fields", None) if env is not None else None
    out: List[str] = []
    for item in list(fields or []):
        name = item[0] if isinstance(item, (tuple, list)) and item else item
        if isinstance(name, str) and name:
            out.append(name)
    return out


def _traj_dict(traj: Trajectory, fields: Sequence[str] = ()) -> Dict[str, np.ndarray]:
    """The named arrays a critic sees: `states` (T x D) and one T-vector per
    state field, named as the environment names it (`fields`, in `s[i]` order).

    RAW STATE ONLY. Prompt 1 hands the critic "raw, unprocessed data" and asks
    for geometric comparisons on named observation keys (pp.13-14); App. A
    p.12 lists "the task description, available variables, and success
    criteria" as its inputs. The dict deliberately omits `rewards` and the
    component series of the candidate that COLLECTED the rollout: with those,
    under an opaque `states` matrix with no field mapping, a critic could only
    rank the very numbers it was meant to judge, or guess an index layout. A
    field vector is emitted only
    when the state width matches the declared fields; a mismatch keeps `states`
    alone rather than mislabel a column."""
    out: Dict[str, np.ndarray] = {}
    states = getattr(traj, "states", None)
    if states is None:
        return out
    arr = np.asarray(states, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out["states"] = arr
    names = list(fields or [])
    if names and arr.ndim == 2 and arr.shape[1] == len(names):
        for i, name in enumerate(names):
            out.setdefault(str(name), arr[:, i])
    return out


@register("phase", "author_critics")
def author_critics(ctx: Any, state: Any) -> List[str]:
    """Stage I, once per search: `PCritic = LLM(L, C, Pcritic)` (Alg. 1 line 3).

    Authored once and carried, never regenerated per iteration -- Alg. 1
    initialises the population before the loop and Stage II never touches it.
    App. A (p.12) announces "an additional round of reflective optimization"
    for the critics that Alg. 1 does not, and Prompt 3 (p.14, "Critic Function
    Reflection Tips") specifies it: a second turn with four checks and an
    output contract; only its call count is unstated. That round is NOT
    implemented -- a known gap recorded in configs/methods/rstar.yaml, not a paper
    silence.

    These are LLM calls and they are charged: R*'s cost against Eureka is wrong
    in R*'s favour by at least this many if they are not (the paper's critics
    also get a reflective round, not run here). budget.json charges them to the
    aggregate llm_calls; the `critics_authored` event is the per-role record.
    """
    n = int(ctx.cfg["generate.alignment.critics"] or 0)
    if n <= 0:
        return []
    prompt = _critic_prompt(ctx, state)
    # ONE call for n samples, the way the reward population is drawn: the
    # Messages API has no `n`, so the client turns this into n round-trips
    # either way, and going through it keeps the budget accounting, the
    # concurrency and the empty-sample reasons identical to every other
    # sampling site.
    texts = ctx.generator(prompt, n=n, tag="critic")
    out: List[str] = []
    for text in texts:
        code = _code_block(text)
        if code:
            out.append(code)
    ctx.event("critics_authored", requested=n, kept=len(out))
    return out


def _critic_key_hint(ctx: Any) -> List[str]:
    """The keys `_traj_dict` will actually populate for this env: `states` plus
    one per declared state field. Adapters carry these as `_state_fields`
    (there is no public `state_fields` attribute), which is what
    `_state_field_names` reads."""
    return ["states"] + _state_field_names(ctx)


def _critic_prompt(ctx: Any, state: Any = None) -> str:
    """R*'s Prompt 1 + Prompt 2 (App. A pp.13-14), on BIRD's own surfaces.

    The paper gives the critic author the task description, the environment's
    observation code, the success conditions and the list of dict keys, then the
    labelling tips. Here: `problem.task_description`; the environment rendered
    THROUGH THE SAME `env_spec` component the reward author gets -- so the
    critic is told no more and no less about the env than the reward author,
    and `strip_existing_reward` withholds the reference
    reward from the critic exactly as it does from the generator (routing
    around it would leak the human reward the whole key exists to hide); the
    success conditions from the task spec (`EnvAdapter.describe_success`); and
    the keys with the `s[i]` field each one is. `rewards` is deliberately
    absent -- see `_traj_dict`. Tip (5) is the paper's manipulation example
    generalised to "agent, object, target"; (1)-(4) are the paper's.
    """
    from ..state import RunState
    cfg = ctx.cfg
    env = getattr(ctx, "env", None)
    env_text = ""
    spec = cfg.get("generate.context.env_spec") or "none"
    if env is not None and spec != "none":
        try:
            fn = _get("env_spec", spec)
            env_text = str(fn(ctx, state if state is not None else RunState()) or "")
        except Exception:  # noqa: BLE001 - a stub env or an unknown spec
            env_text = ""
    success_fn = getattr(env, "describe_success", None)
    success = str(success_fn() if callable(success_fn) else "") or "(not stated)"
    names = _state_field_names(ctx)
    docs: Dict[str, str] = {}
    for item in list(getattr(env, "_state_fields", None) or []):
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            docs[str(item[0])] = str(item[1])
    key_lines = ["  - states: array of shape (T, D), the raw state at every step; "
                 "column i is s[i] below"]
    for i, name in enumerate(names):
        key_lines.append(f"  - {name}: array of shape (T,), s[{i}]"
                         + (f" -- {docs[name]}" if docs.get(name) else ""))
    parts = [
        "You are a trajectory annotator, tasked with distinguishing the quality "
        "of two robot control trajectories.",
        "Your goal is to write a data extraction function that extracts fully "
        "correct sub-trajectories from two given trajectories and assigns "
        "quality labels to them.",
        "Write a Python function `compare(traj_a, traj_b)` where traj_a and "
        "traj_b are dicts of numpy arrays indexed by step.",
    ]
    if env_text.strip():
        parts += ["", "The Python environment is:", env_text.strip()]
    parts += [
        "",
        f"Write a comparison function for the following task: "
        f"{cfg['problem.task_description']}",
        f"The conditions for determining task success are as follows: {success}",
        "",
        "Here is the list of keys for dict:",
        *key_lines,
        "!!! Please ensure that all the keys used exist in the list mentioned above. "
        "If the key does not exist in the list, do not create the variable.",
        "",
        "The output of the function should consist of label list: label list "
        "contains the quality labels for the states at the corresponding indices "
        "in the trajectory. If the state in a is better than the state in b at a "
        "given index, the label is 1; if the state in b is better than the state "
        "in a, the label is -1; otherwise, the label is 0.",
        "Some helpful tips for writing the function code:",
        "(1) Comparison Criteria for States at the Same Index: state a can only be "
        "considered better than state b if it is superior to b in all metrics; if "
        "a is better on one metric and worse on another, the label is 0.",
        "(2) Task Decomposition into Phases for More Accurate Evaluation: the task "
        "may need to be divided into phases, comparing the metric that matters in "
        "each phase.",
        "(3) Do not set thresholds; directly perform numerical comparisons on the "
        "information, for example, if a < b for all distance, a better.",
        "(4) The provided information consists of raw, unprocessed data. Distance "
        "information and other metrics need to be calculated from the state fields.",
        "(5) Typically, you need to consider the distances between the agent (or "
        "manipulator) and the object to be manipulated, and between the object and "
        "its target; or other task-relevant distances and angles.",
        "Return only a python code block.",
    ]
    return "\n".join(parts)


def _code_block(text: str) -> str:
    """The ```python ... ``` block, or the whole answer when the model returned
    bare code. Same tolerance `generation.py` applies to reward completions."""
    if not text:
        return ""
    body = str(text)
    if "```" in body:
        parts = body.split("```")
        for part in parts[1:]:
            chunk = part[6:] if part.lower().startswith("python") else part
            if "def " in chunk:
                return chunk.strip()
    return body.strip() if "def " in body else ""


@register("segment_labeller", "none")
def labeller_none(ctx: Any, state: Any) -> List[Preference]:
    """No labelling. The default."""
    return []


@register("segment_labeller", "critic_population_vote")
def labeller_critic_population(ctx: Any, state: Any) -> List[Preference]:
    """Vote a critic population into labelled segments (S4.3 + App. A, p.12).

    The ladder, verbatim from App. A: "at the beginning, a state pair is labeled
    only if all five critics reach a consensus. We collect 20 trajectory
    segments, each with a length of at least 5. If the required 20 segments are
    not collected, the criteria are relaxed -- labeling occurs if 4 out of 5
    critics agree, and if necessary, further relaxed to 3 out of 5 critics until
    20 labeled samples are obtained."

    `>= min_segment_len`, not `>`: S4.3 says "greater than 5" and App. A says
    "at least 5". The appendix is the specific statement, so it wins, and the
    key is in the config where the choice is visible rather than buried here.

    Each rung is re-run over the SAME pairs rather than continuing from the
    last: relaxing the threshold can only add labels, and `seen` de-duplicates
    an EXACT-span rediscovery only; a relaxed rung normally rediscovers a 5/5
    run as a WIDER run with different bounds, which is appended as a second,
    nested segment (the paper is silent on inter-rung dedupe).
    """
    critics = list(getattr(state, "critics", []) or [])
    if not critics:
        return []
    trajs = [t for t in (getattr(state, "trajectory_store", []) or [])
             if t is not None and (t.length or 0) >= 2]
    if len(trajs) < 2:
        return []

    fns = [f for f in (_compile_critic(c, str(i)) for i, c in enumerate(critics))
           if f is not None]
    if not fns:
        ctx.event("critic_vote", pairs=0, segments=0, reason="no_critic_compiled")
        return []

    ladder = [int(v) for v in (ctx.cfg["generate.alignment.vote_ladder"] or [])]
    ladder = [v for v in ladder if v > 0] or [len(fns)]
    min_len = int(ctx.cfg["generate.alignment.min_segment_len"] or 1)
    target = int(ctx.cfg["generate.alignment.target_segments"] or 0)
    max_pairs = int(ctx.cfg["generate.alignment.max_pairs"] or 0)

    pairs: List[Tuple[int, int]] = [(i, j) for i in range(len(trajs))
                                    for j in range(i + 1, len(trajs))]
    # SAMPLED, not truncated. The store grows every iteration by n_candidates x
    # verify.tpe.trajectories_per_iteration (16 x 100 = 1,600 rollouts under
    # rstar; ~6,400 rollouts and ~20M pairs by iteration 5) -- and taking the
    # first `max_pairs` in index order would
    # compare only the earliest candidates with each other, so the labels would
    # describe iteration 1 forever while the population moved on. The paper says
    # only "Sample trajectories" (Alg. 1 line 10) and never how many, which is
    # why the cap is a config key marked OURS rather than a constant.
    if max_pairs > 0 and len(pairs) > max_pairs:
        ctx.rng.shuffle(pairs)
        pairs = pairs[:max_pairs]

    # one critic pass per pair, reused by every rung of the ladder
    votes: Dict[Tuple[int, int], np.ndarray] = {}
    fields = _state_field_names(ctx)
    for (i, j) in pairs:
        a, b = _traj_dict(trajs[i], fields), _traj_dict(trajs[j], fields)
        T = min(int(trajs[i].length or 0), int(trajs[j].length or 0))
        if T < min_len:
            continue
        rows = []
        for fn in fns:
            try:
                labels = fn(a, b)
            except Exception:  # noqa: BLE001 - a critic that raises abstains
                continue
            arr = np.zeros(T, dtype=int)
            seq = list(labels)[:T]
            arr[:len(seq)] = [int(np.sign(_num(v))) for v in seq]
            rows.append(arr)
        if rows:
            votes[(i, j)] = np.vstack(rows)

    out: List[Preference] = []
    seen: set = set()
    for rung in sorted(ladder, reverse=True):
        if target and len(out) >= target:
            break
        for (i, j), tally in votes.items():
            if target and len(out) >= target:
                break
            agree = np.where((tally == 1).sum(axis=0) >= rung, 1,
                             np.where((tally == -1).sum(axis=0) >= rung, -1, 0))
            for lo, hi, lab in _runs(agree, min_len):
                key = (i, j, lo, hi)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Preference(
                    left_id=f"traj{i}", right_id=f"traj{j}",
                    label=1 if lab == 1 else 0,
                    left_traj=trajs[i], right_traj=trajs[j],
                    left_span=(lo, hi), right_span=(lo, hi),
                    source=f"critic_vote_{rung}of{len(fns)}",
                    iteration=int(getattr(state, "iteration", -1)),
                ))
                if target and len(out) >= target:
                    break
    ctx.event("critic_vote", critics=len(fns), pairs=len(votes),
              segments=len(out), ladder=ladder, target=target)
    return out


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _runs(labels: np.ndarray, min_len: int) -> List[Tuple[int, int, int]]:
    """Maximal runs of a constant non-zero label, as `(start, end_exclusive,
    label)`. "we group consecutive steps with the same label (i.e. 1 or -1) into
    segments" (S4.3) -- 0 is not a label, it is the absence of one, so a run of
    zeros never becomes a segment."""
    out: List[Tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] != 0 and (i - start) >= max(1, min_len):
                out.append((start, i, int(labels[start])))
            start = i
    return out


# ==========================================================================
# param_alignment  --  fn(ctx, state, candidates) -> list[Candidate]
# ==========================================================================
#
# "we optimize the function parameters with preference loss in Equation 2"
# (S4.3). Equation 1 is Bradley-Terry over the SUMMED reward of a segment and
# Equation 2 its cross-entropy. `psi` is "the parameters within the reward
# module and the weights between modules" (S2, p.2).


def _is_tunable(node: ast.AST) -> bool:
    """`tunable: float_literals`: a float `Constant` and nothing else. `bool` is
    an int subclass, never a float, so the isinstance is the whole rule."""
    return isinstance(node, ast.Constant) and isinstance(node.value, float)


class _Literals(ast.NodeTransformer):
    """ONE traversal for everything done to a program's tunable literals.

    Enumerating them (`float_literals`), rewriting them to new values
    (`with_float_literals`) and rewriting them to reads of a parameter vector
    (`_Parametrised`) all go through this class, so the k-th literal is the same
    node under all three by construction. The order is `NodeTransformer`'s:
    depth-first, fields in source order.

    THAT ORDER IS NOT `ast.walk`'s, which is breadth-first. The two index
    spaces agree on a flat expression and disagree the moment a literal sits
    deeper in an earlier sibling than a shallower one in a later sibling:
    enumerated breadth-first, `(1.0 + 2.0 * s[0]) + 3.0` gives
    `[3.0, 1.0, 2.0]`, and written back depth-first that becomes
    `3.0 + 1.0 * s[0] + 2.0`. Every SPSA start point, every "before"
    validation accuracy and every recorded `params_before` would then be of a
    SCRAMBLED program, not the one the model wrote (a `** 0.5` can come back
    as a negative power). `tests/test_alignment_literals.py` pins the round
    trip.

    `replace(i, node)` returns the replacement for the i-th literal, or `None`
    to keep it. Locations are copied so `ast.unparse` and `compile` see a
    well-formed tree; callers still run `ast.fix_missing_locations` for any
    children the replacement introduced.
    """

    def __init__(self, replace: Optional[Callable[[int, ast.Constant],
                                                   Optional[ast.expr]]] = None) -> None:
        self.seen: List[float] = []
        self._replace = replace

    def visit_Constant(self, node: ast.Constant) -> ast.expr:  # noqa: N802
        if not _is_tunable(node):
            return node
        i = len(self.seen)
        self.seen.append(node.value)
        if self._replace is None:
            return node
        new = self._replace(i, node)
        return node if new is None else ast.copy_location(new, node)


def float_literals(code: str) -> List[float]:
    """Every float literal in the program, in the order `_Literals` visits them
    (depth-first, source order) -- our `psi`, and the index space every other
    function here shares.

    FLOATS ONLY, and that is the whole rule. `2.0 * dist` is a coefficient;
    `obs[3]` is an index; `range(10)` is a loop bound. Tuning an int would
    change what a program INDEXES rather than how much it weighs, which is not
    a parameter search, it is corruption. The paper never states the rule (it
    says only "the parameters within the reward module and the weights between
    modules"), so this is ours and is marked so in the config.

    Consequence worth knowing before reading a result: a module written with
    `2 * dist` exposes no parameter and is silently unaligned, while the same
    module written `2.0 * dist` is tuned. Nothing can distinguish those two
    intents from source, so the count of tunable parameters is recorded per
    candidate in `meta['alignment']` rather than assumed.

    Invariant, pinned by `tests/test_alignment_literals.py`:
    `with_float_literals(code, float_literals(code))` is `code` up to
    `ast.unparse` formatting.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    lits = _Literals()
    lits.visit(tree)
    return lits.seen


def with_float_literals(code: str, values: Sequence[float]) -> Optional[str]:
    """The same program with its float literals replaced, k-th for k-th in
    `float_literals` order; literals past the end of `values` are kept.

    `ast.unparse` REWRITES THE SOURCE: comments go, formatting normalises. The
    original is kept at `meta['alignment']['code_before']` so the artifact still
    shows what the model actually wrote -- an aligned candidate whose stored
    code is the unparsed one would make every prompt-vs-artifact diff look like
    the model had changed its style.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    vals = list(values)

    def replace(i: int, node: ast.Constant) -> Optional[ast.expr]:
        return ast.Constant(value=float(vals[i])) if i < len(vals) else None

    tree = _Literals(replace).visit(tree)
    ast.fix_missing_locations(tree)
    try:
        return _unparse(tree)
    except Exception:  # noqa: BLE001
        return None


#: The module-level name a `_Parametrised` program reads its literals from.
#: Dunder-shaped so `compile_reward`'s "last function defined wins" fallback
#: skips it, and closed with two underscores so a class body cannot name-mangle
#: it into something else.
_THETA = "__bird_theta__"


class _Parametrised:
    """One candidate, compiled ONCE, every tunable literal rewritten to a read of
    `__bird_theta__[i]` -- a `Subscript` on a module-level name.

    WHY. Rebuilding the program per evaluation (unparse the AST with new
    constants, re-parse, recompile) is too slow: SPSA makes three evaluations
    per step, so a 1000-step fit would re-parse the SAME source 3000 times per
    candidate. Profiled on the tester tier, that is 98% of a ~215 s run where
    every other method's tester run takes 1-2 s (47,016 `_loss_and_acc`
    calls; parse+unparse 271 s and recompile 30 s of 667 s profiled). Here the
    parse, the rewrite and the bytecode compile happen once
    per candidate per alignment call, and a parameter step costs one `exec` of
    the code object -- a few `def`s -- with a fresh binding of the vector.

    WHY RE-EXECUTE RATHER THAN REBIND A GLOBAL. A literal in a function body is
    read when the function runs, so rebinding the vector in the function's
    globals would suffice for it. A literal in a module-level statement, a
    default argument or a decorator is evaluated when the MODULE runs, and a
    rebinding would leave it frozen at the first vector, where re-parsed
    source would not be. Re-executing the code object per step gives the
    re-parse semantics for both cases, and going through `compile_reward` keeps
    the function scored here the one training will call.

    The rewritten program is compiled from its `ast.unparse` -- the form
    `with_float_literals` emits into the artifact -- so alignment evaluates
    exactly the program that will train.
    `literals` is the vector's starting point, in `float_literals` order.
    """

    def __init__(self, cand: Candidate, language: str = "numpy") -> None:
        self.cand = cand
        self.language = language
        self.literals: List[float] = []
        self.code: Optional[CodeType] = None
        try:
            tree = ast.parse(cand.reward_code)
        except SyntaxError:
            return
        sw = _Literals(lambda i, node: ast.Subscript(
            value=ast.Name(id=_THETA, ctx=ast.Load()), slice=ast.Constant(value=i),
            ctx=ast.Load()))
        tree = sw.visit(tree)
        ast.fix_missing_locations(tree)
        self.literals = sw.seen
        try:
            self.code = compile(_unparse(tree),
                                f"<candidate:{cand.cand_id}:aligned>", "exec")
        except Exception:  # noqa: BLE001 -- accepted by `ast.parse`, rejected by `compile`
            self.code = None

    def bind(self, theta: Sequence[float]) -> Optional[Any]:
        """The reward callable with `theta` in place of the literals, or None
        when the program cannot run -- the same two outcomes one
        `with_float_literals` + `compile_reward` round had per step."""
        from .training import compile_reward  # local: torch-free path stays torch-free
        if self.code is None:
            return None
        try:
            # Python floats, not the ndarray's float64 scalars: the literals they
            # stand in for were Python floats, and the two are not the same
            # arithmetic at the edges (`np.float64 ** -0.5` warns where
            # `float ** -0.5` raises).
            return compile_reward(self.code, self.cand,
                                  bindings={_THETA: [float(v) for v in theta]},
                                  language=self.language)
        except Exception:  # noqa: BLE001
            return None


#: One side of a labelled pair: `(states, actions, lo, hi)`, arrays already
#: converted. `states` is None when the trajectory carries none, and
#: `_segment_sum` then skips the pair.
_Segment = Tuple[Optional[np.ndarray], Optional[np.ndarray], int, int]
_Pair = Tuple[_Segment, _Segment, bool]  # (left, right, label == 1)


def _prepare_pairs(prefs: Sequence[Preference]) -> List[_Pair]:
    """Each preference as two ready segments, the arrays converted ONCE per
    trajectory rather than by every `_segment_sum` call -- ~2 x 3000 x |pairs|
    `np.asarray` calls per candidate on arrays that never change during a fit."""
    cache: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}

    def arrays(traj: Optional[Trajectory]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        key = id(traj)  # every traj is held by a preference for the whole call
        if key not in cache:
            states = getattr(traj, "states", None)
            if states is None:
                cache[key] = (None, None)
            else:
                cache[key] = (np.asarray(states, dtype=float),
                              np.asarray(traj.actions, dtype=float)
                              if traj.actions is not None else None)
        return cache[key]

    out: List[_Pair] = []
    for p in prefs:
        Sl, Al = arrays(p.left_traj)
        Sr, Ar = arrays(p.right_traj)
        out.append(((Sl, Al, int(p.left_span[0]), int(p.left_span[1])),
                    (Sr, Ar, int(p.right_span[0]), int(p.right_span[1])),
                    p.label == 1))
    return out


def _split_preferences(rng: random.Random, prefs: Sequence[Preference],
                       train_fraction: float) -> Tuple[List[Preference], List[Preference]]:
    """The train/validation cut, over a SEEDED SHUFFLE of the labelled pairs.

    ‡ The paper says "70% as the training set and 30% as the validation set"
    (App. A) and never how the split is drawn. `state.preferences` is not in a
    neutral order: `bird.py` extends it once per iteration, and the labeller
    emits each iteration's batch rung-major (5/5 consensus first, then 4/5,
    then 3/5; within a rung, in its own shuffled pair order). A positional
    cut -- `prefs[:n_train]` / `prefs[n_train:]` -- would make the validation
    set that selects the fitted parameters, by construction, the tail of the
    NEWEST batch, and the relaxed-consensus rungs of it whenever the ladder
    had relaxed. At iteration 3 of a tester run, for example, it gives train =
    all 20 iteration-1 labels + the first 8 of iteration 2, valid = the last
    12 of iteration 2; a batch of 17 x 5/5 + 3 x 4/5 puts all three 4/5
    labels in validation. Older batches would never validate, and
    `val_acc_before`/`val_acc_after` would be accuracies on the least-agreed
    labels available.

    Shuffling a COPY with the run's `random.Random` makes the cut a random
    70/30 of the whole buffer, reproducible from `seed`; the caller's list is
    left in order. One shuffle per alignment call, not per candidate, so every
    candidate is fitted and selected on the same split. `n_train` is at least
    1, and when the cut leaves no validation pair the training pairs validate.
    """
    order = list(prefs)
    rng.shuffle(order)
    n_train = max(1, int(round(len(order) * train_fraction)))
    return order[:n_train], order[n_train:] or order[:n_train]


@register("param_alignment", "none")
def alignment_none(ctx: Any, state: Any, candidates: List[Candidate]) -> List[Candidate]:
    """No parameter alignment. The default: every other reproduced method
    takes the LLM's numbers as written, which is exactly the "Suboptimal
    Parameter Assignment" limitation R* names in its introduction."""
    return list(candidates)


@register("param_alignment", "bradley_terry_segments")
def alignment_bradley_terry(ctx: Any, state: Any,
                            candidates: List[Candidate]) -> List[Candidate]:
    """Fit each candidate's float literals to the labelled segments (Eq. 1-2).

    Runs on the pool BEFORE training, on segments labelled from the PREVIOUS
    iteration's rollouts -- Alg. 1 puts lines 10-12 before line 14, and there is
    nothing to sample at iteration 1, so iteration 1 is unaligned. That is the
    paper's own ordering, not a simplification.

    Per call: the labelled segment pairs are shuffled with `ctx.rng` and cut
    `train_fraction` / the rest into train and validation (`_split_preferences`;
    ‡ the paper gives the ratio, 70/30, and not the draw). Per candidate: SPSA
    on the training loss for `iterations` steps, and the parameters with the
    best VALIDATION ACCURACY kept -- "we select the parameters that achieve the
    highest accuracy on the validation set as the final optimized parameters"
    (App. A). Accuracy, not loss: those disagree, and the paper says accuracy.

    A candidate is returned UNCHANGED, with a reason in `meta['alignment']`,
    when it exposes no float literal, when the fit never beat the starting
    validation accuracy, or when the trajectories carry no states to re-execute
    it on. Silently returning the original in those cases would make an
    unaligned run indistinguishable from an aligned one.

    Cost shape: the segment arrays are converted once per call and each
    candidate is compiled once (`_Parametrised`), so a fit costs
    `3 * iterations` re-executions of the reward over the segments and nothing
    else that scales with `iterations`.
    """
    if ctx.cfg["generate.alignment.tunable"] == "none":
        # The method is selected but nothing is declared tunable. Refuse rather
        # than fit an empty parameter vector and report a successful alignment:
        # `_check_coherence` rejects this pairing at load, so reaching here means
        # a component called the aligner directly.
        return list(candidates)
    prefs = [p for p in (getattr(state, "preferences", []) or [])
             if p.left_span is not None and p.right_span is not None]
    if not prefs:
        return list(candidates)
    iters = int(ctx.cfg["generate.alignment.iterations"] or 0)
    lr = float(ctx.cfg["generate.alignment.learning_rate"])
    if iters <= 0:
        return list(candidates)
    # The run's one RNG stream, in this order: the labeller's pair shuffle
    # (already drawn), ONE shuffle of the buffer here, then SPSA's per-step
    # signs per candidate. The shuffle sits after the `iters` guard so an
    # alignment that does nothing also draws nothing.
    train, valid = _split_preferences(
        ctx.rng, prefs, float(ctx.cfg["generate.alignment.train_fraction"]))
    train_pairs, valid_pairs = _prepare_pairs(train), _prepare_pairs(valid)

    out: List[Candidate] = []
    for cand in candidates:
        theta0 = float_literals(cand.reward_code)
        info: Dict[str, Any] = {"n_params": len(theta0), "n_train": len(train),
                                "n_valid": len(valid), "split": "shuffled"}
        if not theta0:
            info["skipped"] = "no_float_literals"
            cand.meta["alignment"] = info
            out.append(cand)
            continue
        prog = _Parametrised(cand, language=reward_language(ctx.cfg))
        if prog.code is None:
            # `ast.parse` took the program but `compile` refused it with the
            # literals replaced -- a float inside a `match ... case 1.0:` pattern
            # is the known shape (a MatchValue must stay a literal). That is a
            # program this optimiser cannot parametrise, not a fit that lost:
            # say so, rather than run SPSA on a `bind()` that returns None and
            # record `val_acc_before: 0.0` for a program never evaluated.
            info["skipped"] = "not_parametrisable"
            cand.meta["alignment"] = info
            out.append(cand)
            continue
        fitted, acc0, acc1 = _spsa(ctx, prog, np.array(theta0, dtype=float),
                                   train_pairs, valid_pairs, iters, lr)
        info.update({"val_acc_before": acc0, "val_acc_after": acc1})
        if fitted is None or acc1 <= acc0:
            info["skipped"] = "no_validation_gain"
            cand.meta["alignment"] = info
            out.append(cand)
            continue
        new_code = with_float_literals(cand.reward_code, fitted)
        if new_code is None:
            info["skipped"] = "unparse_failed"
            cand.meta["alignment"] = info
            out.append(cand)
            continue
        info["code_before"] = cand.reward_code
        info["params_before"] = theta0
        info["params_after"] = [float(v) for v in fitted]
        cand.reward_code = new_code
        cand.meta["alignment"] = info
        ctx.event("param_alignment", cand_id=cand.cand_id, n_params=len(theta0),
                  val_acc_before=acc0, val_acc_after=acc1)
        out.append(cand)
    return out


def _segment_sum(reward: Any, S: Optional[np.ndarray], A: Optional[np.ndarray],
                 lo: int, hi: int) -> Optional[float]:
    """`sum_t F_psi(s_t)` over one segment -- the quantity Eq. 1 exponentiates.
    `S`/`A` are the trajectory's arrays as `_prepare_pairs` converted them."""
    if S is None:
        return None
    total = 0.0
    for t in range(lo, min(hi, len(S) - 1)):
        a = A[t] if A is not None and t < len(A) else np.zeros(1)
        try:
            r, _ = reward(S[t], a, S[t + 1])
        except Exception:  # noqa: BLE001 - a program that raises scores nothing
            return None
        total += float(r)
    return total


def _loss_and_acc(prog: _Parametrised, theta: np.ndarray,
                  pairs: Sequence[_Pair]) -> Tuple[float, float]:
    """Eq. 2 over `pairs`, and the accuracy of the induced ranking.

    Both come out of one pass because they read the same two sums: the loss is
    what SPSA descends and the accuracy is what selects the answer, and
    computing them separately would double the cost of the inner loop for
    nothing. One evaluation is one `prog.bind` plus the segment sums.
    """
    reward = prog.bind(theta)
    if reward is None:
        return float("inf"), 0.0
    loss = 0.0
    n = 0
    correct = 0
    for left, right, want_left in pairs:
        sa = _segment_sum(reward, *left)
        sb = _segment_sum(reward, *right)
        if sa is None or sb is None:
            continue
        # log-sum-exp, not exp/sum: a segment sum of a few hundred overflows
        # float64 in exp() and turns the whole loss into nan, which SPSA reads
        # as "every direction is equally bad" and reports as a converged fit.
        m = max(sa, sb)
        logZ = m + math.log(math.exp(sa - m) + math.exp(sb - m))
        loss += -( (sa if want_left else sb) - logZ )
        correct += int((sa > sb) == want_left)
        n += 1
    if n == 0:
        return float("inf"), 0.0
    return loss / n, correct / n


def _spsa(ctx: Any, prog: _Parametrised, theta0: np.ndarray,
          train: Sequence[_Pair], valid: Sequence[_Pair],
          iters: int, lr: float) -> Tuple[Optional[np.ndarray], float, float]:
    """Simultaneous-perturbation descent on Eq. 2, keeping the best validation
    accuracy.

    TWO objective evaluations per step regardless of how many parameters the
    program exposes -- see this module's docstring for why that, and not a
    finite-difference gradient, at the paper's 1000 iterations.

    The perturbation scale decays as `a/(k+1)**0.101` and the step as
    `lr/(k+1)**0.602`, the standard Spall exponents. They are not in the paper
    (it names no optimiser) and are not exposed as config keys, because a knob
    nobody has a published value for is a fabricated pin.
    """
    theta = theta0.copy()
    best = theta0.copy()
    _, acc0 = _loss_and_acc(prog, theta0, valid)
    best_acc = acc0
    scale = 0.1 * (np.abs(theta0) + 1.0)
    for k in range(iters):
        ck = 1.0 / ((k + 1) ** 0.101)
        ak = lr / ((k + 1) ** 0.602)
        # stdlib `random.Random`, one draw per parameter -- see `_sample_two`
        delta = np.array([-1.0 if ctx.rng.random() < 0.5 else 1.0
                          for _ in range(theta.size)], dtype=float)
        step = ck * scale * delta
        lp, _ = _loss_and_acc(prog, theta + step, train)
        lm, _ = _loss_and_acc(prog, theta - step, train)
        if not (math.isfinite(lp) and math.isfinite(lm)):
            continue
        grad = (lp - lm) / (2.0 * step)
        theta = theta - ak * grad
        _, acc = _loss_and_acc(prog, theta, valid)
        if acc > best_acc:
            best_acc, best = acc, theta.copy()
    return (best if best_acc > acc0 else None), acc0, best_acc
