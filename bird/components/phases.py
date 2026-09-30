"""Termination rules and the pre/post phase plugins.

Two registry families live here.

**`termination`** (`loop.termination`, §0). Every published
method is a fixed bound: `fixed_iterations` for all but CARD, whose
`fixed_generations` is the same bound with the last iteration ending at §1
(the paper's loop body ends on a generation, Alg. 1 l.17). Adaptive stopping is
unclaimed territory; the other four rules exist so the axis has more than
one point to sweep.

**`phase`** (`pre:` / `post:`). Some published stages do
not fit the six-stage loop: DrEureka's RAPP sweep and DR generation run *around*
it, RDA's decomposition runs once per search rather than once per iteration, the
human-in-the-loop query is a §4 side-channel, and the final retrain is a report
protocol rather than a search step. Bending a stage definition around any of
them would corrupt the six-stage contract, so they are named plugins instead.

`bird.py` calls every phase positionally and `state` may be `None` for a
pre-phase, so each takes `(ctx, state)` with the extra arguments the loop
supplies for the in-loop phases. `pre:`/`post:` entries may be a bare string or
a `{name: ...}` mapping; `bird.py` passes no per-entry arguments either way, so
a mapping's extra keys are documentation, not configuration.

Interfaces this module consumes but does not own:

  * `ctx.generator` / `ctx.evaluator` -- `bird/llm/base.py`'s contract,
    ``client(messages, n=1, tag="") -> list[str]`` plus `chat_json`. A client
    records its own token usage, so no phase calls `budget.record_llm`.
  * `ctx.env.dr_parameters` -- ``{name: (lo, hi)}``, the declared simulator
    range for each randomisable physics parameter.
  * `ctx.env.dr_probe(policy_ref, overrides, n_rollouts, cfg=) -> float` -- success
    rate of `policy_ref` with those DR overrides applied. RAPP is inert without
    it, and says so rather than inventing numbers. Duck-typed, because the env
    adapter's contract is `bird/envs/`'s to pin; `EnvAdapter` implements it.
"""

from __future__ import annotations

import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .. import preference_log
from ..budget import BudgetExceeded
from ..config import Config, deep_merge
from ..context import Context
from ..parsing import extract_json
from ..registry import get as registry_get
from ..registry import register
from ..state import FINAL_ARTIFACT_RULES, RunState
from ..types import CandidateReport, Preference, TrainResult

# `phase` is one family with one owner, so the two phases whose bodies live in
# sibling modules are registered here rather than there. Imported, never
# reimplemented -- a second copy of a validity check is a second thing to keep
# in sync with §2's two-population rule.
from .. import judgments as judge_trace
from .. import multiview
from ..observability import sample_frames
from .frames import sheet_cells, sheet_manifest
from .evaluation import (_client_text, _parse_reason, _parse_score,
                         _resolve_deferred, client_sees_images,
                         judge_concurrency, judge_digest)
from .generation import _images_for
from .preferences import run_preferences
from .verification import run_validity

log = logging.getLogger("bird")


# --------------------------------------------------------------------------
# Constants that the schema has no key for.
#
# The schema's rule -- never declare a key the algorithm cannot honor -- has a
# converse: never bury a decision the schema *should* own inside a literal.
# These three are named so the missing knobs stay visible.
# --------------------------------------------------------------------------

#: Grid resolution of the RAPP sweep. DrEureka's released sweep is denser; there
#: is no `rapp.sweep_points` key, so this is the gap.
RAPP_SWEEP_POINTS = 9

#: The RAPP keep rule -- see `run_rapp`'s docstring: a recorded deviation from
#: the published one-rollout boolean, not a disputed reading.
RAPP_SUCCESS_RATE = 0.5

#: `generate.decomposition.n_subtasks: auto` with no usable model reply. RDA
#: reports ~5 subtasks for simple tasks and ~10 for long-horizon ones (§1).
DEFAULT_AUTO_SUBTASKS = 5
MAX_AUTO_SUBTASKS = 12


# ==========================================================================
# Cross-module shims
# ==========================================================================


def _ask(client: Any, prompt: str, tag: str,
         images: Optional[List[Any]] = None) -> str:
    """One text completion (`bird/llm/base.py`: `client(messages, n, tag)`).

    `tag` names the *purpose* of the call. It is advisory for a real provider
    but the mock routes on it, so tagging is what lets the offline suite
    exercise a phase instead of handing it a reward program.

    `images` rides along as the `images=` keyword the client contract declares
    -- `run_decompose` attaches the env image there (App. 7.1) -- and is simply
    omitted when empty, so a text-only client never sees the keyword.

    A client failure degrades to "" and the caller's offline fallback runs: a
    phase that dies because a provider hiccuped would take a finished search
    with it, and the offline path must stay deterministic.
    """
    if client is None:
        return ""
    kwargs: Dict[str, Any] = {"n": 1, "tag": tag}
    if images:
        kwargs["images"] = list(images)
    try:
        out = client(prompt, **kwargs)
    except Exception as exc:  # noqa: BLE001 -- any client failure is a degrade
        log.debug("phase %s: LLM query failed (%s); falling back offline", tag, exc)
        return ""
    if isinstance(out, str):
        return out
    return out[0] if out else ""


def _ask_json(client: Any, prompt: str, schema_hint: Any, tag: str) -> Dict[str, Any]:
    """One structured completion. `{}` when the client has nothing to say."""
    if client is None:
        return {}
    try:
        out = client.chat_json(prompt, schema_hint, tag=tag)
    except Exception as exc:  # noqa: BLE001
        log.debug("phase %s: structured query failed (%s)", tag, exc)
        return {}
    return out if isinstance(out, dict) else {}


def _dr_ranges(ctx: Context) -> Dict[str, Tuple[float, float]]:
    """`ctx.env.dr_parameters` normalised to ``{name: (lo, hi)}``."""
    raw = getattr(ctx.env, "dr_parameters", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:  # noqa: BLE001 -- an env hook we do not own
            raw = None
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    for name, spec in raw.items():
        pair: Optional[Sequence[Any]] = None
        if isinstance(spec, dict):
            lo, hi = spec.get("min", spec.get("lo")), spec.get("max", spec.get("hi"))
            if lo is not None and hi is not None:
                pair = (lo, hi)
        elif isinstance(spec, (list, tuple)) and len(spec) >= 2:
            pair = (spec[0], spec[1])
        if pair is None:
            continue
        try:
            lo_f, hi_f = float(pair[0]), float(pair[1])
        except (TypeError, ValueError):
            continue
        out[str(name)] = (min(lo_f, hi_f), max(lo_f, hi_f))
    return out


def _dr_probe(ctx: Context, policy_ref: Optional[str], name: str, value: float,
              n_rollouts: int) -> float:
    """Success rate of the incumbent policy at one swept parameter value.

    `EnvAdapter.dr_probe` is the implementation on every adapter; it rebuilds
    the stored policy through
    `training.policy_from_ref`, which needs `ctx.cfg` to name an sb3 blob's
    algorithm -- hence the keyword. A blob that cannot be rebuilt raises there and
    is logged and turned into NaN here, never rolled out untrained.

    Returns NaN when the environment adapter exposes no probe. RAPP then
    declines to narrow anything instead of inventing a sweep -- an invented
    prior is exactly what DrEureka's ablations show fails badly (§1
    `generate.co_design.dr_prior`).
    """
    env = ctx.env
    overrides = {name: value}
    attempts = (
        lambda: env.dr_probe(policy_ref, overrides, n_rollouts, cfg=ctx.cfg),
    )
    for attempt in attempts:
        try:
            return float(attempt())
        except (AttributeError, TypeError):
            continue
        except Exception as exc:
            log.debug("rapp: probe of %s=%s failed (%s)", name, value, exc)
            return float("nan")
    return float("nan")


def _write_artifact(ctx: Context, filename: str, payload: Any) -> Optional[str]:
    """Phase outputs go under `<rundir>/phases/`; a dry run writes nothing.

    `filename` may name a subdirectory (`final_retrain/checkpoints.json`), so a
    phase whose output is a record PLUS a bulky series can keep the two apart
    without a second writer. Only the leading directories are created; nothing
    else about the call changes, and a plain name behaves exactly as before.
    """
    if ctx.rundir is None:
        return None
    d = ctx.rundir.path / "phases"
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return str(path)


def _rank_key(report: CandidateReport) -> Tuple[float, str]:
    """Descending fitness, then `cand_id` -- a total, deterministic order.

    §5's tie-break axis is a real one, but it belongs to `select`. A phase
    that has to break a tie must not do it by argmax accident, so it breaks it
    by name and says so.
    """
    f = report.fitness if report.fitness is not None else float("-inf")
    return (-f, report.cand_id)


# ==========================================================================
# Termination  (`loop.termination`, §0)
# ==========================================================================


@register("termination", "fixed_iterations")
def term_fixed_iterations(ctx: Context, state: RunState) -> bool:
    """Run exactly `loop.n_iterations` iterations. Every published method (§0).

    Returning False unconditionally is the whole implementation, not a stub:
    `run_search` already bounds the loop, so "fixed iterations" means precisely
    "never stop before that bound".
    """
    return False


@register("termination", "fixed_generations")
def term_fixed_generations(ctx: Context, state: RunState) -> bool:
    """Run `loop.n_iterations` GENERATIONS: N-1 in-loop trainings, and the last
    iteration ends at §1. CARD (§0).

    Same stopping bound as `fixed_iterations` -- this body returns False for the
    same reason -- plus one declared property, `ends_on_generation`, which
    `run_search` reads off the configured rule and turns into the per-iteration
    transient `RunState.generation_only` for the iteration the declared budget
    ends on. In that iteration §1 generates and §2 runs the VALIDITY half only:
    the candidates are not quality-screened, not trained (`Budget.record_skip`
    books each un-launched training, `TrainResult.skip_reason` says why) and
    not evaluated (`fitness=None, fitness_source="none"`, a fixed "never
    trained" feedback); §5/§6 run unchanged, so `select.rule: none` +
    `update.winner.action: become_parent` adopt the program as chain head and
    `select.final_artifact: chain_end` returns it. `config._check_coherence`
    refuses any other §5 pair.

    WHY A TERMINATION VALUE. CARD's Alg. 1 (refs/tex/card/main.tex:428-447) is
    train-then-query: l.3 generates R0, l.4 trains it, each of N iterations
    screens R (l.8), trains it if TPE passes (l.9-11), appends feedback
    (l.12/l.14) and generates the next R as the body's LAST statement (l.17), so
    `Ensure R` returns a program no stage screened or trained -- N+1
    generations, at most N in-loop trainings. The release has the same phase by
    hand: step k trains code_{k-1} "no matter pass_flag"
    (metaworld_exp_one_step.py:374-381) and one LLM call writes code_k
    (query_llm_metaworld.py:44-53), then the run ends. The paper's loop body
    literally ends on a different statement than BIRD's iteration does, and
    "where the loop stops" is this key's meaning; the paper's reported number
    for the returned program is a separate five-seed training
    (`post: [final_retrain]`), never an in-loop one. Validity still runs
    because the release validates every generation (`max_try_num` retries).

    Not or-able into `any_of` for the same reason `fixed_iterations` is not
    (see `term_any_of`); an adaptive rule cannot express "then generate once
    more", and no published method needs it to.
    """
    return False


term_fixed_generations.ends_on_generation = True  # read by `run_search`


@register("termination", "fitness_plateau")
def term_fitness_plateau(ctx: Context, state: RunState) -> bool:
    """Stop after `patience` iterations with no gain > `min_delta`. Unpublished.

    `state.fitness_history` may hold None -- CARD computes no ranking scalar at
    all (§4 `evaluate.fitness.source: none`). A None carries no evidence about
    the trend, so it is skipped rather than counted as a stall; a run with no
    scalars anywhere therefore never plateaus, which is the honest answer for a
    search that has no fitness to plateau on.

    Improvement is measured against the running best, not the previous value:
    against the previous value a single lucky iteration would reset the counter
    forever.
    """
    patience = ctx.cfg.get("loop.termination_cfg.patience")
    if not patience:
        return False
    min_delta = ctx.cfg.get("loop.termination_cfg.min_delta") or 0.0
    scalars = [f for f in state.fitness_history if f is not None]
    if not scalars:
        return False

    best, stall = float("-inf"), 0
    for f in scalars:
        if f > best + min_delta:
            best, stall = f, 0
        else:
            stall += 1
    return stall >= int(patience)


@register("termination", "budget_exhausted")
def term_budget_exhausted(ctx: Context, state: RunState) -> bool:
    """Stop when a `budget.*` cap is spent (§0).

    `Budget` also raises `BudgetExceeded` from inside whichever stage crosses
    the cap, which `run_search` catches and breaks on. This rule is the polite
    version: it lets the current iteration finish and reports the stop as a
    termination rather than as an exception, so the run still has a winner.
    """
    return bool(ctx.budget.exhausted())


@register("termination", "success_threshold")
def term_success_threshold(ctx: Context, state: RunState) -> bool:
    """Stop once `loop.termination_cfg.target_fitness` has been reached (§0).

    Published by nobody -- adaptive stopping is an open axis.
    Best-so-far, not last-iteration: the run returns `select.final_artifact`
    (`global_best` by default), so once a target-hitting reward is *recorded*
    the search has nothing left to prove. Gating on the last winner instead
    would keep paying for iterations after the answer was already found.
    """
    target = ctx.cfg.get("loop.termination_cfg.target_fitness")
    if target is None:
        return False
    scalars = [f for f in state.fitness_history if f is not None]
    return bool(scalars) and max(scalars) >= float(target)


@register("termination", "any_of")
def term_any_of(ctx: Context, state: RunState) -> bool:
    """Disjunction of the rules that can actually fire.

    `fixed_iterations` and `fixed_generations` are excluded deliberately: both
    *are* the `while it < loop.n_iterations` bound in `run_search`, always in
    force, so or-ing either in would be a no-op that reads like a rule (and
    `fixed_generations`' second property, the generate-only last iteration, is
    read off the configured rule by `run_search`, not fired). Config coherence
    (`config._check_coherence`) already requires `patience` here, since a
    plateau term with no patience would silently reduce this to two rules.
    """
    return (term_fitness_plateau(ctx, state)
            or term_success_threshold(ctx, state)
            or term_budget_exhausted(ctx, state))


# ==========================================================================
# Phases owned by sibling modules
# ==========================================================================

register(
    "phase", "validity",
    doc="Stage-2 validity checks as a named phase (body in verification.py).",
)(run_validity)

register(
    "phase", "preferences",
    doc="Stage-4 preference block as a named phase (body in preferences.py).",
)(run_preferences)


# ==========================================================================
# Subtask decomposition and reflection  (RDA)
# ==========================================================================

_BULLET = re.compile(r"^\s*(?:[-*•]|\(?\d{1,2}[.)])\s*(.*\S)\s*$")
_SPLIT_HINTS = re.compile(r"\s*(?:,|;|\bthen\b|\band\b)\s*", re.IGNORECASE)


def _clean_item(line: str) -> str:
    """One list item, stripped of bullets, markdown emphasis and quotes."""
    m = _BULLET.match(line)
    text = m.group(1) if m else line.strip()
    text = text.strip().strip("`").strip()
    text = re.sub(r"^\*\*(.*)\*\*$", r"\1", text).strip()
    text = text.strip('"').strip("'").strip()
    return re.sub(r"\s+", " ", text)


def _parse_list(text: str, key: str = "") -> List[str]:
    """Parse a model reply into an ordered, de-duplicated list of items.

    A fenced JSON payload wins when there is one -- it is the shape the model
    was actually asked for -- and the numbered prose is the fallback.
    """
    text = (text or "").strip()
    if not text:
        return []
    blob = extract_json(text)
    if isinstance(blob, dict) and key:
        blob = blob.get(key)
    if isinstance(blob, list) and all(isinstance(x, str) for x in blob):
        parsed = _dedup(_clean_item(x) for x in blob)
        if parsed:
            return parsed
    items = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        bulleted = _BULLET.match(raw) is not None
        item = _clean_item(raw)
        # Prose preamble ("Here are the subtasks:") is not a subtask. Unbulleted
        # lines are kept only when nothing is bulleted, so a plain newline list
        # still parses.
        if len(item) > 1 and (bulleted or not item.endswith(":")):
            items.append((bulleted, item))
    if any(b for b, _ in items):
        items = [(b, i) for b, i in items if b]
    return _dedup(i for _, i in items)


def _dedup(items: Any) -> List[str]:
    seen, out = set(), []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _fallback_subtasks(task: str, n: int) -> List[str]:
    """Deterministic offline stand-in when the model returns nothing usable.

    Split on the clause markers the instruction already contains, then pad. It
    is a poor decomposition on purpose -- it keeps a config runnable offline
    without pretending an LLM said something it did not.
    """
    parts = _dedup(p.strip(" .") for p in _SPLIT_HINTS.split(task or ""))
    parts = [p for p in parts if len(p) > 2][:n]
    while len(parts) < n:
        parts.append(f"part {len(parts) + 1} of {n}: {task}".strip())
    return parts[:n]


def _decompose_prompt(task: str, n: Optional[int], env_code: str,
                      has_image: bool, guidance: str = "") -> str:
    """App. 7.1's Subtask Generation prompt, on its three inputs.

    Task instruction, environment code ('includes the agent's observation and
    **success condition**' -- the prompt tells the model to take thresholds
    from it and nowhere else), and the image when one is attached. No
    constraint the paper does not state is added (e.g. no 'independently
    observable from a rollout video' requirement). The count sentence keeps
    the paper's ~5/~10 guidance as the `auto` branch and honours a pinned
    `n_subtasks` exactly.
    """
    count = ("- Use the fewest subtasks needed to describe essential behavior: "
             "about 5 for a simple task, about 10 for a complex or long-horizon one."
             if n is None else f"- Return exactly {n} subtasks.")
    image_input = ("\n- Image: visualizes the agent and environment (attached)."
                   if has_image else "")
    return (
        "You are a task decomposition expert assisting reinforcement learning "
        "agents. Break the given task into concise, high-level subtasks that "
        "describe the main stages required for success.\n\n"
        "You are given the following input:\n"
        "- Task instruction: outlines the goal and intended agent behavior.\n"
        "- Environment code: includes the agent's observation and **success "
        "condition**."
        f"{image_input}\n\n"
        "Create a sequence of goal-oriented subtasks that describe progress "
        "toward completion:\n"
        "- Each subtask should represent a distinct, measurable phase of progress.\n"
        "- Do not include perception or reasoning steps (e.g., \"identify goal\", "
        "\"plan path\").\n"
        "- Avoid numeric thresholds unless they are explicitly defined in the "
        "success condition.\n"
        "- Keep the subtasks concise.\n"
        f"{count}\n"
        "- The final subtask describes the agent achieving the success as an "
        "accomplished goal: do not phrase it as stop/halt; express the agent "
        "actively bringing about the success within the threshold.\n"
        # `generate.decomposition.guidance` (ours, no paper pin): extra
        # instruction appended inside the constraints block; "" when unset.
        f"{guidance}\n"
        f"## Task Instruction\n{task}\n\n"
        f"## Environment Code\n{env_code}\n\n"
        "## Output\nReturn a numbered list (1., 2., ...) of subtasks in order, "
        "and nothing else."
    )


@register("phase", "decompose")
def run_decompose(ctx: Context, state: Optional[RunState] = None) -> List[str]:
    """RDA: split the instruction into subtasks, ONCE per search (§1).

    From RDA (Lee et al. 2026). `generate.decomposition.n_subtasks` may be an
    int or `auto`; `auto` means the model chooses, and RDA reports ~5 subtasks
    for simple tasks and ~10 for long-horizon ones. The query goes to
    `ctx.generator`: the Agent VLM does ALL of RDA's in-loop work -- §5.1 names
    subtask generation among what GPT-5 does -- and GPT-4.1 exists only for the
    post-hoc alignment metric, 'to mitigate potential bias between the
    generator and evaluator' (§10.1) -- not a separate, cheaper model for
    decomposition.

    The prompt carries App. 7.1's three inputs: the same env
    presentation generation uses (`generate.context.env_spec` -- the paper's
    prompt reads thresholds off the success condition in the env code, so a
    decomposition without it invents or omits them), and `env.task_images()`
    when `problem.instruction_modality` attaches one. The image belongs HERE
    and only here: App. 7.2 (reward generation) and 7.5 (reward reflection)
    take no image, so `generation.backend_llm` attaches none. The paper's own
    prompt includes the image ('## Image {image}'), and there is no released
    code to disagree with it.

    Once-per-search is enforced by the caller: `generate()` only dispatches here
    while `state.subtasks` is empty, so this function does the work
    unconditionally rather than re-deriving the guard.
    """
    cfg = ctx.cfg
    want = cfg["generate.decomposition.n_subtasks"]
    task = cfg["problem.task_description"]
    modality = cfg.get("problem.instruction_modality", "text")
    n_req = None if isinstance(want, str) else int(want)

    # The env presentation is the one generation itself uses, so the two
    # prompts cannot disagree about what the model knows of `E` (§4.1's
    # decomposition conditions on the same (I, E) the candidates do).
    env_code = registry_get("env_spec", cfg["generate.context.env_spec"])(ctx, state)
    # `_images_for` gates on `problem.instruction_modality` and wraps the
    # render in the stack-dump watchdog -- on a Meta-World RDA run this is the
    # run's FIRST render, since generation attaches no images.
    images = _images_for(ctx)
    extra = cfg.get("generate.decomposition.guidance")
    guidance = ("\n" + str(extra).strip() + "\n"
                if extra and str(extra).strip() else "")
    reply = _ask(ctx.generator,
                 _decompose_prompt(task, n_req, env_code, bool(images),
                                   guidance=guidance),
                 tag="decompose", images=images)
    subtasks = _parse_list(reply, key="subtasks")

    if n_req is not None:
        # The count is a config pin, so it is honoured exactly: a short reply is
        # topped up rather than silently shrinking J for the rest of the search.
        subtasks = subtasks[:n_req]
        if len(subtasks) < n_req:
            log.warning("    decompose: model returned %d of %d subtasks; padding",
                        len(subtasks), n_req)
            filler = _fallback_subtasks(task, n_req)
            subtasks += [s for s in filler if s not in subtasks][:n_req - len(subtasks)]
    else:
        subtasks = subtasks[:MAX_AUTO_SUBTASKS]
        if not subtasks:
            log.warning("    decompose: no usable reply; falling back to %d offline "
                        "subtasks", DEFAULT_AUTO_SUBTASKS)
            subtasks = _fallback_subtasks(task, DEFAULT_AUTO_SUBTASKS)

    log.info("    decompose -> %d subtask(s) (n_subtasks=%s, modality=%s)",
             len(subtasks), want, modality)
    return subtasks


def _subtask_evidence(subtasks: List[str],
                      report: CandidateReport) -> Dict[int, float]:
    """Map §4's per-subtask scores onto subtask *indices*.

    `CandidateReport.subtask_scores` is keyed by whatever §4 chose to key it by,
    so both the text and the stringified index are accepted.
    """
    by_text = {s: i for i, s in enumerate(subtasks)}
    out: Dict[int, float] = {}
    for key, score in (report.subtask_scores or {}).items():
        idx: Optional[int] = by_text.get(key)
        if idx is None:
            try:
                probe = int(key)
            except (TypeError, ValueError):
                continue
            idx = probe if 0 <= probe < len(subtasks) else None
        if idx is None:
            continue
        try:
            out[idx] = float(score)
        except (TypeError, ValueError):
            continue
    return out


def _reflection_prompt(task: str, subtasks: List[str],
                       report: CandidateReport) -> str:
    """App. 7.4's Subtask Reflection prompt: full per-subtask analysis in, the
    MODEL's choice of which subtask (if any) most contributed to unintended
    behavior out.

    The trajectory-analysis table carries behavior / score / analysis per
    subtask -- the structured output App. 7.4 names as its input -- from
    `winner_report.subtask_behaviors` / `.subtask_scores` /
    `.subtask_rationales`. When scores exist but no prose does, the
    lowest-scoring subtask is named as a SUGGESTED focus only: an argmin rule
    demoted to a hint, kept because without any analysis text there
    is nothing else to point at, and phrased as a suggestion because under the
    paper's own cascade rule the lowest score is typically a downstream
    auto-zeroed subtask, not the root cause (§8.1 revises subtask 4 at 0.5 over
    subtask 5 at 0.0).
    """
    listing = "\n".join(f"{i}. {s}" for i, s in enumerate(subtasks, start=1))
    evidence = _subtask_evidence(subtasks, report)
    blocks: List[str] = []
    for i, sub in enumerate(subtasks):
        bits = [f"Subtask {i + 1}: {sub}"]
        if i in evidence:
            bits.append(f"  score: {evidence[i]:.3f}")
        behavior = (report.subtask_behaviors or {}).get(sub, "")
        if behavior:
            bits.append(f"  behavior: {behavior}")
        analysis = (report.subtask_rationales or {}).get(sub, "")
        if analysis:
            bits.append(f"  analysis: {analysis}")
        blocks.append("\n".join(bits))
    has_prose = any((report.subtask_behaviors or {}).get(s)
                    or (report.subtask_rationales or {}).get(s) for s in subtasks)
    hint = ""
    if evidence and not has_prose:
        worst = min(sorted(evidence), key=lambda i: (evidence[i], i))
        hint = (f"\nNo per-subtask analysis text was recorded this round; the "
                f"lowest-scoring subtask is {worst + 1}, which may suggest a focus "
                f"-- but the choice is yours, and the lowest score may be a "
                f"downstream consequence rather than the cause.")
    table = "\n".join(blocks) if blocks else "(no trajectory analysis was recorded)"
    return (
        "You are a robotics task refinement analyst. Use the trajectory analysis "
        "to decide whether the subtask definitions need refinement; refinement "
        "should help the agent exhibit more natural and goal-aligned behavior.\n\n"
        f"## Task Instruction\n{task}\n\n"
        f"## Subtask List\n{listing}\n\n"
        f"## Trajectory Analysis\n{table}\n{hint}\n"
        "## Instructions\n"
        "- Determine whether refinement is needed.\n"
        "- If no refinement is necessary, keep the original subtask list.\n"
        "- If refinement is needed, select ONE subtask that most contributed to "
        "unintended behavior, and revise it to encourage natural and intended "
        "behavior.\n"
        "- Avoid numeric thresholds. Keep the subtasks concise.\n"
        "- Keep the same total number of subtasks, and keep every other subtask "
        "unchanged.\n\n"
        "## Output\n"
        "Return a JSON object containing:\n"
        "- \"decision\": a short paragraph explaining which subtask (if any) was "
        "refined, why or why not, and how the change supports better reward "
        "function design.\n"
        "- \"subtasks\": the final subtask list, unchanged if no refinement is "
        "needed, or with one updated definition if refinement was applied."
    )


#: The reflection's answer shape -- App. 7.4's own output contract: a decision
#: paragraph plus the FULL final list (not an index; the revised index is
#: derived by diffing against the current list, and the hold-J /
#: at-most-one-changed / may-decline constraints are enforced by validation).
_REFLECT_SCHEMA = {"decision": "str", "subtasks": ["str"]}


@register("phase", "subtask_reflection")
def run_subtask_reflection(ctx: Context, state: RunState,
                           winner_report: CandidateReport) -> List[str]:
    """RDA subtask co-evolution: revise at most one subtask (§6; RDA App. 7.4).

    Three constraints, and they *are* the method (§6
    `update.co_evolve.subtasks`): revise AT MOST ONE subtask per iteration,
    hold J = len(subtasks) constant, and be allowed to decline. All three are
    load-bearing -- a reflection that rewrites the whole list makes the
    per-subtask scores incomparable across iterations, and one that cannot
    decline is forced to churn a list that is already fine. They are enforced
    by VALIDATING the returned list (App. 7.4's output is the full final list),
    never by pre-selecting the target.

    WHICH subtask is the model's choice, on App. 7.4's criterion -- the one
    'that most contributed to unintended behavior', decided from the full
    per-subtask trajectory analysis the prompt carries. Computing the target
    here as the minimum-score subtask would be wrong: under the paper's own
    cascade rule the lowest score is typically a LATE, auto-zeroed subtask,
    while the paper's worked examples revise the mid-list root cause (§8.1:
    subtask 4 at 0.5 revised over subtask 5 at 0.0; Fig. 5(e): subtask 3 at
    0.5). Argmin survives only as a SUGGESTED focus in the prompt when scores
    exist with no analysis text, and as nothing at all otherwise.

    The query goes to `ctx.generator` -- the Agent VLM does all of RDA's
    in-loop reflection (§5.1).
    """
    subtasks = list(state.subtasks or [])
    if not subtasks:
        return subtasks

    evidence = _subtask_evidence(subtasks, winner_report)
    if not evidence:
        # WARNING, not debug. With no per-subtask scores the analysis table is
        # prose-only at best and empty at worst, so the model decides from
        # weaker evidence than §4 normally supplies -- and a mechanism that
        # silently degrades to its weaker form is indistinguishable in a normal
        # log from one working as designed.
        log.warning("    subtask_reflection: no per-subtask scores for the winner; "
                    "the model is choosing from the analysis prose alone. This is "
                    "the weaker path -- the trajectory analysis normally carries "
                    "evaluate.feedback.granularity: per_subtask scores.")

    data = _ask_json(ctx.generator,
                     _reflection_prompt(ctx.cfg.get("problem.task_description", ""),
                                        subtasks, winner_report),
                     _REFLECT_SCHEMA, tag="subtask_reflection")
    declined_reason = ""
    decision = " ".join(str(data.get("decision") or "").split())[:600]
    returned = data.get("subtasks")

    if not data:
        declined_reason = "no reply"
    elif not isinstance(returned, (list, tuple)):
        declined_reason = "no subtask list in the reply"
    else:
        final = [_clean_item(str(s)) for s in returned]
        if any(not s for s in final):
            declined_reason = "reply contained an empty subtask"
        elif len(final) != len(subtasks):
            # Hold J: a list of a different length is a decline, never a
            # truncation or a pad -- either would silently rewrite the method.
            declined_reason = (f"changed the number of subtasks "
                               f"({len(final)} != {len(subtasks)})")
        else:
            changed = [i for i, (a, b) in enumerate(zip(subtasks, final)) if a != b]
            if not changed:
                declined_reason = "model declined"
            elif len(changed) > 1:
                # At most one: applying the first of several would fabricate a
                # single-subtask reflection the model did not perform.
                declined_reason = f"revised {len(changed)} subtasks, not one"

    if declined_reason:
        log.info("    subtask_reflection -> declined (%s); %d subtask(s) unchanged",
                 declined_reason, len(subtasks))
        ctx.event("subtask_reflection", declined=True, reason=declined_reason,
                  decision=decision, n_subtasks=len(subtasks))
        return subtasks

    target = changed[0]
    before = subtasks[target]
    subtasks[target] = final[target]
    log.info("    subtask_reflection -> revised subtask %d of %d",
             target, len(subtasks))
    ctx.event("subtask_reflection", declined=False, index=target,
              before=before, after=final[target], decision=decision,
              n_subtasks=len(subtasks))
    return subtasks  # length unchanged by construction: one slot assigned


# ==========================================================================
# The human in the loop  (§4 `evaluate.human.*`)
# ==========================================================================


def _success_rate(report: CandidateReport) -> Optional[float]:
    """Observed success rate, or None when nothing observed it.

    Native-signal rule (N): the fallback keys are `NATIVE_SUCCESS_KEYS` (genuine
    env success-flag names), never `task_success` -- that ALIASES the BIRD
    task_metric (custom_metric) in the curve rows, and the human oracle is a §4
    search signal that must read the env's own flag."""
    from ..native_signal import NATIVE_SUCCESS_KEYS
    trajs = report.result.trajectories or []
    if trajs:
        return sum(1.0 for t in trajs if t.success) / len(trajs)
    for m in report.result.seed_metrics or []:
        for key in NATIVE_SUCCESS_KEYS:
            if key in m:
                try:
                    return float(m[key])
                except (TypeError, ValueError):
                    continue
    return None


def _reference_score(ctx: Context, report: CandidateReport) -> Tuple[float, str]:
    """The oracle's own view of a candidate, and where it came from.

    Deliberately NOT `report.fitness` when anything better exists: a "human"
    that reads the search's own objective is circular, and GT's whole point is
    that its human supplies a signal the fitness does not contain. The
    ground-truth reward channel (`TrainResult.gt_reward_curve`, §3 `train.log:
    gt_reward`) -- the shipped reward's return, i.e. native_reward -- is
    preferred, then raw returns. `report.fitness` is the last resort and is
    reported as such so the circularity is visible in the artifact rather than
    hidden.

    Native-signal rule (N): the env's own trajectory metric (`env.task_metric`)
    is NOT a fallback -- it is the BIRD custom_metric, and the human oracle
    is a §4 search signal that must never read it on a bird-origin task. The only
    admissible env-signal here is the shipped reward return above (native_reward).
    """
    # None entries are the "no reference reward on this task" marker
    # (`training._rollout`), so a curve of Nones is NO channel and falls through
    # to raw returns -- it is not a reference that paid zero.
    curve = [float(v) for v in (report.result.gt_reward_curve or [])
             if v is not None and math.isfinite(float(v))]
    if curve:
        return float(sum(curve) / len(curve)), "gt_reward_curve"

    trajs = report.result.trajectories or []
    for hook in ("reference_score",):
        scorer = getattr(ctx.env, hook, None)
        if not callable(scorer) or not trajs:
            continue
        try:
            vals = [float(scorer(t)) for t in trajs]
        except Exception as exc:  # noqa: BLE001 -- an env hook we do not own
            log.debug("human oracle: env.%s failed (%s)", hook, exc)
            continue
        if vals:
            return sum(vals) / len(vals), f"env.{hook}"

    if trajs:
        return sum(float(t.mean_per_step_return) for t in trajs) / len(trajs), \
            "mean_per_step_return"

    if report.fitness is not None:
        return float(report.fitness), "fitness (circular)"
    return 0.0, "none"


#: Tie sentinel, shared with `preferences.TIE`. The repo's comparison
#: convention is `types.Preference.label` -- 1 = left preferred, 0 = right --
#: extended with -1 for a tie. Stated here because it is NOT C's cmp
#: convention, and an oracle returning -1 for "right wins" would be read as a
#: tie by every caller.
TIE = -1


class _NullHumanOracle:
    """`evaluate.human.mode: none`. No human exists, so every query is a bug.

    Raising beats returning a neutral value: a silently-empty human channel
    turns GT into its own no-human ablation without the config saying so, and
    §5.1.2 measures that as a real degradation. Callers that duck-type this
    oracle (§4's `human_score`, §5's `rule: human`) catch the refusal and
    degrade with a warning, which is the visible failure we want.
    """

    #: `preference_log`'s `judge` category: nothing is asked, so nothing is
    #: judged.
    judge_kind = "none"

    mode = "none"
    scripted = False

    def _refuse(self, what: str) -> None:
        raise RuntimeError(
            f"human oracle queried for {what} but evaluate.human.mode is 'none'; "
            "set evaluate.human.mode to feedback_text | preference_labels | veto")

    def feedback(self, report: CandidateReport) -> str:
        self._refuse("feedback")
        return ""  # unreachable; keeps the signature honest

    def score(self, report: CandidateReport) -> float:
        self._refuse("a score")
        return 0.0

    def compare(self, a: CandidateReport, b: CandidateReport) -> int:
        self._refuse("a comparison")
        return TIE

    def label(self, a: CandidateReport, b: CandidateReport) -> int:
        self._refuse("a preference label")
        return 0

    def veto(self, report: CandidateReport) -> bool:
        self._refuse("a veto")
        return False


class _ScriptedHumanOracle:
    """A deterministic stand-in for GT's human-in-the-loop (§4).

    GT is explicit that "all human feedback is collected from one of the
    authors", which no test can reproduce. This oracle reads the environment's
    reference reward (see `_reference_score`) and answers from it, so every
    config with `evaluate.human.mode != none` runs unattended and identically on
    every machine. Its prose says it is scripted, in the text itself, so a
    transcript can never be mistaken for a human's.

    The query surface is what the rest of the repo already duck-types:
      * `feedback(report) -> str` -- GT's free-text channel (§4).
      * `score(report) -> float` -- Text2Reward-human's entire evaluation
        signal, on the `evaluate.feedback.score_scale` range.
      * `compare(a, b) -> int` / `label(a, b) -> int` -- the repo convention
        (see `TIE`): 1 = left preferred, 0 = right preferred, -1 = tie.
        `label` never returns a tie; `compare` only does when
        `evaluate.preferences.allow_ties` -- GT sets it false, deliberately.
      * `select(reports) -> list` -- for `select.rule: human` (§5).
      * `veto(report) -> bool` -- `evaluate.human.mode: veto`.

    Every query charges `budget.record_human()`: this is the only place that
    knows a person was actually asked, and human attention is the scarcest
    column in the budget. `preferences.comparator_human` charges nothing of
    its own (charging there too would count every comparison twice). Known
    limitation: `selection._ask_human` and
    `verification._confirm_before_execute` also charge at their call sites,
    so a `select` or `veto` query routed through them is counted twice.
    """

    #: `preference_log`'s `judge` category, and it is NOT "human". This is a
    #: stand-in; letting it pass as a person's judgement would corrupt exactly
    #: the VLM-vs-human agreement number the preference log exists to make
    #: computable. A scripted answer is not a person's, and the artifact has
    #: to say so rather than imply otherwise.
    judge_kind = "scripted"

    scripted = True

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.mode = ctx.cfg.get("evaluate.human.mode", "none")
        self.allow_ties = bool(ctx.cfg.get("evaluate.preferences.allow_ties", False))

    # -- queries --

    def feedback(self, report: CandidateReport) -> str:
        self.ctx.budget.record_human()
        score, source = _reference_score(self.ctx, report)
        rate = _success_rate(report)
        n = len(report.result.trajectories or [])
        verdict = self._verdict(rate)
        seen = f"{n} rollout(s)" if n else "the training curve"
        return (f"[scripted human oracle] {report.cand_id}: watched {seen}. "
                f"Reference return {score:.4f} (source: {source}); "
                f"{'success unobserved' if rate is None else f'success {rate:.0%}'}. "
                f"{verdict}")

    def score(self, report: CandidateReport) -> float:
        """A rating on the configured scale, from what was observed.

        Success rate when there is one -- a person grading a video is grading
        whether the task got done -- and a squashed reference return otherwise.
        `binary` quantises, which is what the scale means.
        """
        self.ctx.budget.record_human()
        lo, hi = (1.0, 5.0) if self.ctx.cfg.get(
            "evaluate.feedback.score_scale") == "likert" else (0.0, 1.0)
        rate = _success_rate(report)
        if rate is None:
            ref, _ = _reference_score(self.ctx, report)
            rate = 1.0 / (1.0 + math.exp(-ref)) if math.isfinite(ref) else 0.0
        if self.ctx.cfg.get("evaluate.feedback.score_scale") == "binary":
            rate = 1.0 if rate >= 0.5 else 0.0
        return lo + (hi - lo) * max(0.0, min(1.0, rate))

    def compare(self, a: CandidateReport, b: CandidateReport) -> int:
        self.ctx.budget.record_human()
        sa, _ = _reference_score(self.ctx, a)
        sb, _ = _reference_score(self.ctx, b)
        if sa > sb:
            return 1
        if sb > sa:
            return 0
        if self.allow_ties:
            return TIE
        # A forced choice still has to be reproducible, so it is made by name,
        # not by whichever happened to be listed first.
        return 1 if a.cand_id <= b.cand_id else 0

    def label(self, a: CandidateReport, b: CandidateReport) -> int:
        verdict = self.compare(a, b)
        if verdict == TIE:  # `Preference.label` has no tie value
            return 1 if a.cand_id <= b.cand_id else 0
        return verdict

    def select(self, reports: List[CandidateReport]) -> List[CandidateReport]:
        """The human picks a winner (§5 `select.rule: human`)."""
        if not reports:
            return []
        self.ctx.budget.record_human()
        # Ties by name, not by list position -- see `_rank_key`.
        return [min(reports, key=lambda r: (-_reference_score(self.ctx, r)[0],
                                            r.cand_id))]

    def veto(self, report: CandidateReport) -> bool:
        """True when the scripted human sees no task behaviour at all.

        The narrowest defensible veto: it fires on "this agent never does the
        task", never on "I would have preferred something else" -- the latter is
        a preference, and there is a mode for that.
        """
        self.ctx.budget.record_human()
        rate = _success_rate(report)
        if rate is not None:
            return rate <= 0.0
        score, _ = _reference_score(self.ctx, report)
        return not math.isfinite(score)

    # -- prose --

    @staticmethod
    def _verdict(rate: Optional[float]) -> str:
        if rate is None:
            return "I could not tell whether it completed the task."
        if rate <= 0.0:
            return "It never completed the task; the reward is not driving the goal."
        if rate < 0.5:
            return ("It completes the task sometimes but not reliably; the failures "
                    "are what to fix, not the successes.")
        if rate < 1.0:
            return ("It usually completes the task; spend the next revision on the "
                    "remaining failures.")
        return ("It completes the task every time; keep this behaviour and stop "
                "adding shaping terms.")


@register("phase", "human_oracle")
def build_human_oracle(ctx: Context) -> Any:
    """Construct `ctx.human` once per run (§4 `evaluate.human.mode`).

    Called from `run()` before the loop, so the oracle is a run-level resource
    like the LLM clients and the env adapter -- not something a stage rebuilds.
    `none` yields an oracle that raises on every query; anything else yields the
    scripted oracle. An interactive oracle would be a third class registered
    under this same name in a config that asks for it.
    """
    mode = ctx.cfg.get("evaluate.human.mode", "none")
    if mode == "none":
        return _NullHumanOracle()
    log.info("human oracle: scripted (mode=%s, cap=%s/iteration, applies_to=%s)",
             mode, ctx.cfg.get("evaluate.human.queries_per_iteration"),
             ctx.cfg.get("evaluate.human.applies_to"))
    return _ScriptedHumanOracle(ctx)


def _human_targets(ctx: Context,
                   reports: List[CandidateReport]) -> List[CandidateReport]:
    """The candidates the human is allowed to be asked about.

    ORDERING CAVEAT, and it is a real one, though narrow.
    `evaluate.human.applies_to: selected_only` is GT's rule -- the human is
    queried about the *selected* agent (Alg. 1 l.15-17: best <- argmax b_1:N,
    feedback <- human(pi_best)). §4 runs before §5, so at this point nothing is
    selected yet; the stand-in is the best-scoring report, which is what the
    fitness argmax would choose. DEFERRED fitness sources (`preference_bt`) are
    resolved here first, so under `select.rule: bradley_terry` that stand-in IS
    the BT argmax -- by construction rather than by agreement:
    `selection.rule_bradley_terry` selects on the very strengths this sort
    reads (`meta["bt_strength"]`, the preferences phase's one `b_1:N`) instead
    of refitting with a second prior. Resolving first matters: with the
    deferred fitness still None, `_rank_key` would map every trained candidate
    to -inf, the sort would collapse to cand_id order (and a
    valid-but-training-failed report at `select.failure_value` = -10000 would
    outrank every trained one), so the one human query per iteration -- GT's
    defining mechanism -- would narrate a lexicographically-chosen agent,
    injected next round as "ground truth" about the actual winner. The
    residual approximation is only for select rules
    whose winner is not the fitness argmax (pareto, random, human); moving the
    query after §5 would put a §4 concern inside §5, which the stage contract
    forbids -- so that residue is recorded here rather than hidden.
    """
    # Idempotent, keyed on `report.fitness_source` through `evaluation._DEFERRED`
    # -- a registry value, never a method name. The preferences phase has
    # already written `meta["bt_strength"]` by the time this phase runs
    # (bird.py stage-4 order: fitness source, preferences, similarity, human).
    for r in reports:
        _resolve_deferred(ctx, r)
    live = [r for r in reports if r.candidate.valid and not r.candidate.screened_out]
    if not live:
        live = [r for r in reports if r.candidate.valid]
    if not live:
        return []
    live = sorted(live, key=_rank_key)
    if ctx.cfg.get("evaluate.human.applies_to", "selected_only") == "selected_only":
        return live[:1]
    return live


def _human_text(ctx: Context, state: RunState, targets: List[CandidateReport],
                cap: int) -> List[str]:
    """`feedback_text` -- GT's free-text critique, injected as ground truth."""
    lines = []
    for report in targets[:cap]:
        text = ctx.human.feedback(report)
        report.meta["human_feedback"] = text
        lines.append(text)
    return lines


def _human_prefs(ctx: Context, state: RunState, targets: List[CandidateReport],
                 cap: int) -> List[str]:
    """`preference_labels` -- the human labels pairs into `D_pref`."""
    if len(targets) < 2:
        log.warning("    human_feedback: preference_labels needs two candidates to "
                    "compare; evaluate.human.applies_to=%s yielded %d",
                    ctx.cfg.get("evaluate.human.applies_to"), len(targets))
        return []
    lines = []
    for i in range(0, min(cap, len(targets) // 2) * 2, 2):
        a, b = targets[i], targets[i + 1]
        label = ctx.human.label(a, b)  # charges the query
        pref = Preference(
            left_id=a.cand_id, right_id=b.cand_id, label=label,
            left_traj=(a.result.trajectories or [None])[0],
            right_traj=(b.result.trajectories or [None])[0],
            source="human", iteration=state.iteration)
        state.preferences.append(pref)
        # `annotator` names the ORACLE, not a person: this is the scripted
        # stand-in `evaluate.human.oracle` selects, and a dataset that let it
        # pass as a human judgement would corrupt exactly the agreement number
        # the log exists to make computable.
        oracle = getattr(ctx, "human", None)
        preference_log.record(
            ctx, pref, state, judge=preference_log.judge_of(oracle),
            annotator=f"oracle:{type(oracle).__name__}")
        winner = a.cand_id if label == 1 else b.cand_id
        lines.append(f"[scripted human oracle] preferred {winner} over "
                     f"{b.cand_id if label == 1 else a.cand_id}.")
    return lines


def _human_veto(ctx: Context, state: RunState, targets: List[CandidateReport],
                cap: int) -> List[str]:
    """`veto` -- a vetoed candidate must lose §5 but stay in the record.

    It is scored at `select.failure_value`, the same sentinel a failed candidate
    gets (§5): "must lose every comparison but stay recorded". Deleting it
    instead would make the veto invisible in the run artifact.
    """
    sentinel = ctx.cfg.get("select.failure_value", -10000.0)
    lines = []
    for report in targets[:cap]:
        if not ctx.human.veto(report):
            continue
        report.meta["human_veto"] = True
        report.meta["fitness_before_veto"] = report.fitness
        report.fitness = float(sentinel)
        lines.append(f"[scripted human oracle] vetoed {report.cand_id}: it does not "
                     f"perform the task at all.")
    return lines


#: Handlers keyed by `evaluate.human.mode`. A dict, not a chain of `if`s: the
#: config value picks the behaviour, which is the same discipline the registry
#: enforces one level up.
_HUMAN_MODES: Dict[str, Callable[..., List[str]]] = {
    "feedback_text": _human_text,
    "preference_labels": _human_prefs,
    "veto": _human_veto,
}


@register("phase", "human_feedback")
def run_human_feedback(ctx: Context, state: RunState,
                       reports: List[CandidateReport]) -> List[CandidateReport]:
    """Query the human and route the answer into next round's prompt (§4).

    From GT (`evaluate.human.mode: feedback_text`, one query per iteration about
    the selected agent, injected into the next prompt as "ground truth" --
    removing it measurably degrades, §5.1.2) and from Text2Reward-human, whose
    entire evaluation signal is the human.

    `evaluate.human.queries_per_iteration` is a hard cap and GT sets it to 1
    deliberately; human attention is the scarcest budget line, so the cap is
    enforced here rather than trusted to the oracle.

    The text lands on `state.human_feedback`, which is where §1's
    `generate.context.include_human_feedback` reads it next round. It is
    replaced, not appended: "one query per iteration" means the prompt carries
    this iteration's answer, and accumulation across iterations is
    `generate.history_mode`'s job, not this phase's.

    See `_human_targets` for the ordering caveat on `applies_to: selected_only`
    (deferred fitness is resolved there first, so the target is the BT argmax
    under `select.rule: bradley_terry`).
    """
    mode = ctx.cfg.get("evaluate.human.mode", "none")
    handler = _HUMAN_MODES.get(mode)
    if handler is None or state is None:  # `none` never reaches here via bird.py
        return reports

    cap = int(ctx.cfg.get("evaluate.human.queries_per_iteration") or 0)
    if cap <= 0:
        log.warning("    human_feedback: mode=%s but queries_per_iteration=%s", mode, cap)
        return reports

    targets = _human_targets(ctx, reports)
    if not targets:
        log.info("    human_feedback: nothing valid to ask about")
        return reports

    before = ctx.budget.human_queries
    lines = handler(ctx, state, targets, cap)
    spent = ctx.budget.human_queries - before

    # Assigned even when empty. A round where the human said nothing must not
    # leave last round's text in the prompt: §1 injects this as "ground truth",
    # and stale ground truth is worse than none.
    state.human_feedback = "\n".join(lines)
    log.info("    human_feedback -> %d quer(y|ies) (mode=%s, applies_to=%s)",
             spent, mode, ctx.cfg.get("evaluate.human.applies_to"))
    ctx.event("human_feedback", mode=mode, queries=spent,
              applies_to=ctx.cfg.get("evaluate.human.applies_to"),
              targets=[r.cand_id for r in targets[:cap]],
              chars=len(state.human_feedback))
    return reports


# ==========================================================================
# DrEureka: RAPP, then DR generation
# ==========================================================================


def _policy_ref(ctx: Context, state: Optional[RunState], which: str) -> Optional[str]:
    """Which trained policy the RAPP sweep drives. None = the env's default.

    `rapp.policy` has no registry family (it is a free string in the schema), so
    the lookup is a dict rather than a branch chain. As a `pre:` phase RAPP
    receives `state=None`, so there IS no incumbent -- DrEureka runs its sweep
    after stage 1 has trained one (Alg. 2 "Require: policy pi_initial";
    rapp.py:100 plays `--run {cfg.run_path}`). A `post:` phase would receive the
    final RunState, but no post-loop phase trains under the generated DR, so the
    published order is unreachable either way. Falling back to the env default
    is the honest reading and is logged.
    """
    sources: Dict[str, Callable[[RunState], Optional[str]]] = {
        "incumbent_best": lambda s: s.best.result.policy_ref if s.best else None,
        "latest": lambda s: s.latest.result.policy_ref if s.latest else None,
        "policy_checkpoint": lambda s: s.policy_ref,
        "none": lambda s: None,
    }
    if state is None:
        log.info("    rapp: no RunState (pre-phase); sweeping the env's default policy")
        return None
    ref = sources.get(which, sources["incumbent_best"])(state)
    if ref is None:
        log.info("    rapp: rapp.policy=%s resolved to no checkpoint; using the "
                 "env's default policy", which)
    return ref


@register("phase", "rapp")
def run_rapp(ctx: Context, state: Optional[RunState] = None) -> None:
    """DrEureka's Reward-Aware Physics Prior: sweep, keep what still works (§0).

    For each parameter in `rapp.parameters`, run the incumbent policy across the
    environment's declared range for that parameter and keep the min and max
    value at which the task is still solved. Those bounds -- not the simulator's
    full range -- are what `dr_generation` is allowed to randomise over.
    "Reward-aware" is the point: the feasible physics range depends on the
    reward that trained the policy, so the prior cannot be written once per
    environment.

    The keep rule is a RECORDED DEVIATION, not a disputed reading. Paper and
    released code agree: each swept value gets ONE rollout of the initial
    policy and is kept iff a boolean criterion holds. The prose: "roll out
    pi_Eureka in this modified simulation. If the policy's performance
    satisfies a pre-defined success criterion, we deem this value as feasible"
    (refs/tex/dreureka/arxiv.tex:322; Alg. 2 applies `F` once per value). The
    code: `rapp.py:100-112` launches one `play.py` subprocess per value (a
    single episode -- play.py L58 sets `Cfg.env.num_envs = 1`) and keeps the value iff
    `success(...)` is true, where `forward_locomotion_success`
    (`rapp.py:20-35`) averages the per-step velocity error over that one run
    and returns `average_success >= -1.0`. No rate, no repeat. THIS
    IMPLEMENTATION instead keeps a value when the probe's success rate over
    `rapp.rollouts_per_value` rollouts is >= RAPP_SUCCESS_RATE (0.5). With
    `rollouts_per_value: 1` a 0/1 rate thresholded at 0.5 IS the published
    boolean, so the deviation lives in the key's value (`_default.yaml` and
    dreureka.yaml both say 100), not in this function; `rapp.success_criterion`
    names the per-rollout metric. The released sweep does NOT threshold a
    success *rate* over repeated rollouts.

    When the environment exposes no DR probe, the sweep cannot run and the
    declared range is recorded unchanged, marked `degenerate`. That is
    DrEureka's own "uninformative prior" ablation arm, which fails badly -- so
    it is logged loudly rather than passed off as a prior.
    """
    cfg = ctx.cfg
    ctx.counters.setdefault("rapp_bounds", {})
    if not cfg.get("rapp.enabled", False):
        log.info("    rapp: rapp.enabled=false -> no prior computed")
        return

    params = list(cfg.get("rapp.parameters") or [])
    ranges = _dr_ranges(ctx)
    rollouts = int(cfg.get("rapp.rollouts_per_value") or 1)
    criterion = cfg.get("rapp.success_criterion") or "task_success"
    policy = _policy_ref(ctx, state, cfg.get("rapp.policy") or "incumbent_best")

    if not params:
        log.warning("    rapp: rapp.parameters is empty; nothing to sweep")
    bounds: Dict[str, List[float]] = {}
    sweeps: Dict[str, List[Dict[str, float]]] = {}
    degenerate: List[str] = []

    for name in params:
        declared = ranges.get(name)
        if declared is None:
            log.warning("    rapp: env declares no range for %r; skipping", name)
            continue
        lo, hi = declared
        span = hi - lo
        values = [lo + span * i / (RAPP_SWEEP_POINTS - 1)
                  for i in range(RAPP_SWEEP_POINTS)]

        rows, feasible = [], []
        for v in values:
            rate = _dr_probe(ctx, policy, name, v, rollouts)
            rows.append({"value": v, "success_rate": rate})
            if math.isfinite(rate) and rate >= RAPP_SUCCESS_RATE:
                feasible.append(v)
        sweeps[name] = rows

        if feasible:
            bounds[name] = [min(feasible), max(feasible)]
        else:
            # Either the probe is missing or the policy solved nothing anywhere.
            # Both mean "no prior"; the declared range is kept so stage 2 still
            # has something to randomise, and both are flagged.
            bounds[name] = [lo, hi]
            degenerate.append(name)
            log.warning("    rapp: %r never met %s >= %.2f; keeping the full declared "
                        "range (this is the uninformative-prior arm)",
                        name, criterion, RAPP_SUCCESS_RATE)

    ctx.counters["rapp_bounds"] = bounds
    ctx.counters["rapp_sweep"] = sweeps
    _write_artifact(ctx, "rapp_prior.json", {
        "policy": cfg.get("rapp.policy"),
        "policy_ref": policy,
        "success_criterion": criterion,
        "keep_rule": f"mean success rate over {rollouts} rollout(s) >= {RAPP_SUCCESS_RATE}",
        "keep_rule_note": "recorded deviation: paper (arxiv.tex:322, Alg. 2) and "
                          "rapp.py:100-112 keep a value on ONE rollout's boolean "
                          "criterion; see run_rapp.__doc__",
        "sweep_points": RAPP_SWEEP_POINTS,
        "declared_ranges": {k: list(v) for k, v in ranges.items()},
        "bounds": bounds,
        "degenerate": degenerate,
        "sweep": sweeps,
    })
    ctx.event("rapp", parameters=list(bounds), bounds=bounds,
              rollouts_per_value=rollouts, degenerate=degenerate)
    log.info("    rapp -> bounds for %d parameter(s)%s", len(bounds),
             f" ({len(degenerate)} degenerate)" if degenerate else "")


_DR_RANGE_LINE = re.compile(
    r"^\s*([A-Za-z_][\w.\-]*)\s*[:=]\s*\[?\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]?\s*$")
_DR_BLOCK = re.compile(r"^\s*(?:#+\s*)?config\b", re.IGNORECASE)


def _dr_prompt(task: str, bounds: Dict[str, List[float]], n: int) -> str:
    """DrEureka stage 2's prompt. Note what is NOT in it: the environment."""
    if bounds:
        lines = "\n".join(f"{k}: [{v[0]:g}, {v[1]:g}]" for k, v in sorted(bounds.items()))
        prior = f"The policy is known to work within these ranges:\n{lines}\n"
    else:
        prior = "No physics prior is available.\n"
    return (
        "Write domain randomisation configurations for a simulated robot.\n\n"
        f"Task: {task}\n\n"
        f"{prior}\n"
        f"Produce {n} distinct configurations. Each is a block beginning with the "
        "word CONFIG, then one parameter per line as `name: [low, high]`.\n"
        "Vary how aggressive the ranges are across the configurations."
    )


def _parse_dr_configs(text: str, bounds: Dict[str, List[float]],
                      n: int) -> List[Dict[str, Any]]:
    """Lenient parse: clip ranges into the prior, drop unparseable lines.

    Repo-authored. DrEureka's release has no `parse_dr`: dr_eureka.py:109-128
    extracts the first code block and pastes it verbatim into
    `class domain_rand_eureka` (legged_robot_config.py:232); nothing clips into
    the RAPP bounds, an unknown name is an inert attribute, and a block that
    breaks the config is scored DUMMY_FAILURE (dr_eureka.py:214-217).
    """
    configs: List[Dict[str, Any]] = []
    current: Optional[Dict[str, List[float]]] = None
    for raw in (text or "").splitlines():
        if _DR_BLOCK.match(raw):
            if current:
                configs.append({"params": current, "source": "llm"})
            current = {}
            continue
        m = _DR_RANGE_LINE.match(raw)
        if not m or current is None:
            continue
        name = m.group(1)
        try:
            lo, hi = float(m.group(2)), float(m.group(3))
        except ValueError:
            continue
        if lo > hi:
            lo, hi = hi, lo
        if bounds:
            if name not in bounds:
                continue  # unknown parameter: dropped, per `degrade`
            blo, bhi = bounds[name][0], bounds[name][1]
            lo, hi = max(lo, blo), min(hi, bhi)
            if lo > hi:
                continue
        current[name] = [lo, hi]
    if current:
        configs.append({"params": current, "source": "llm"})
    return [c for c in configs if c["params"]][:n]


@register("phase", "dr_generation")
def run_dr_generation(ctx: Context, state: Optional[RunState] = None) -> None:
    """DrEureka stage 2: write DR configs, then select none of them (§1, §5).

    Two properties, both surprising, both real:

    1. **The prompt shows no environment at all.** Task text plus the RAPP
       bounds, nothing else -- no source, no observation spec, no reward. Stage
       1 gets `generate.context.env_spec: full_source`; stage 2 gets none of it.
    2. **Stage 2 selects nothing.** Policies trained under different DR
       distributions are not comparable in simulation, so DrEureka declines to
       rank them and sends every configuration to real-world evaluation (§5
       `select.rule`). That is implemented literally here: the configs are
       stored unordered, `dr_selected` is set to None, and nothing sorts them.
       Silently ranking them would fabricate the one judgement the method
       explicitly refuses to make.

    `generate.co_design.dr_prior` chooses where the bounds come from -- `rapp`
    reads what `run_rapp` left on `ctx.counters`, `default_sim_ranges` reads the
    simulator's declared ranges, `none` supplies no bounds at all. DrEureka's
    ablations show the last two fail badly; the config can still ask for them
    because that ablation is the point.
    """
    cfg = ctx.cfg
    n = int(cfg.get("generate.co_design.dr_n_configs") or 0)
    prior_name = cfg.get("generate.co_design.dr_prior", "none")

    priors: Dict[str, Callable[[], Dict[str, List[float]]]] = {
        "none": lambda: {},
        "default_sim_ranges": lambda: {k: list(v) for k, v in _dr_ranges(ctx).items()},
        "rapp": lambda: {k: list(v) for k, v in
                         (ctx.counters.get("rapp_bounds") or {}).items()},
    }
    bounds = priors.get(prior_name, priors["none"])()
    if prior_name == "rapp" and not bounds:
        log.warning("    dr_generation: dr_prior=rapp but no bounds on ctx.counters; "
                    "run the rapp pre-phase first")

    prompt = _dr_prompt(cfg["problem.task_description"], bounds, n)
    configs = _parse_dr_configs(_ask(ctx.generator, prompt, tag="dr_config"), bounds, n)
    n_llm = len(configs)

    # Offline top-up so the phase produces the configured count without an API
    # key. Tagged, never mixed into the LLM's own output in the artifact.
    while len(configs) < n and bounds:
        params = {}
        for name, (lo, hi) in sorted((k, (v[0], v[1])) for k, v in bounds.items()):
            a, b = sorted((ctx.rng.uniform(lo, hi), ctx.rng.uniform(lo, hi)))
            params[name] = [a, b]
        configs.append({"params": params, "source": "fallback_sampled"})
    if len(configs) < n:
        log.warning("    dr_generation: produced %d of %d configs (no prior to sample "
                    "from and no usable model reply)", len(configs), n)

    ctx.counters["dr_configs"] = configs
    ctx.counters["dr_selected"] = None  # literally: stage 2 selects nothing
    _write_artifact(ctx, "dr_configs.json", {
        "n_requested": n,
        "n_from_llm": n_llm,
        "dr_prior": prior_name,
        "bounds": bounds,
        "prompt_shows_environment": False,
        "selected": None,
        "selection_rule": "none -- DR policies are sim-incomparable; every config "
                          "goes to real-world evaluation (§5)",
        "configs": configs,
    })
    ctx.event("dr_generation", n_configs=len(configs), n_from_llm=n_llm,
              dr_prior=prior_name, selected=None)
    log.info("    dr_generation -> %d config(s) (prior=%s); selecting none",
             len(configs), prior_name)


# ==========================================================================
# Post-phases
# ==========================================================================


#: The same resolver `bird.final_artifact` uses -- `RunState.final_artifact`,
#: which reads the return `update()` recorded before the last boundary carry
#: and falls back to the live slots. A copy that read the live slots only would,
#: under `loop.carry: []`, look for `state.best` after the carry had nulled it
#: and retrain nothing. Retraining something other than what the run returns
#: would make the number meaningless, so there is one resolver and it lives on
#: the state both callers hold (the repo-root `bird.py` cannot be imported from
#: the package).
_FINAL_ARTIFACT_RULES: Dict[str, Callable[[RunState], Optional[CandidateReport]]] = {
    name: (lambda s, _rule=name: s.final_artifact(_rule)) for name in FINAL_ARTIFACT_RULES
}


def _report_rollouts(cfg: Config, n_seeds: int) -> int:
    """How many rollouts the retrain must produce for the REPORT to be computable.

    `evaluate.rollouts_per_candidate` is a SEARCH key: it sizes the evidence
    stage 4 ranks candidates on (RDA's K=3, Table 1). The reporting phases have
    their own, differently-pinned counts, and RDA's is App. §10.1: "for each
    trained policy, we collect 5 trajectory videos ... four times per video,
    yielding 20 evaluations per policy ... each task is trained with 3 random
    seeds, this results in 60 total evaluations per task".

    Both numbers are right, and a retrain that rolled out
    `evaluate.rollouts_per_candidate` episodes would reach only the first: with
    `alignment_rate.n_videos: 5` against `rollouts_per_candidate: 3`,
    `alignment_rate` would take `[:5]` of 3 and report the headline metric over
    12 judgments instead of 20. This function is the third count that
    reconciles them.

    `n_videos * n_seeds`, not `n_videos`, and that is the paper's arithmetic
    rather than a safety margin. `_sb3_run` draws its final rollouts from
    `per_seed_policies[i % n_seeds]`, so 5 videos over 3 seed policies is 2/2/1
    per policy; 15 is 5 EACH, which is what "5 videos per trained policy, 3
    seeds, 60 evaluations" means.

    Never smaller than the config's own value: the retrain is also SCORED, by
    `evaluate.fitness.source`, and shrinking its rollout count would change the
    number `final_retrain` exists to report.
    """
    k = int(cfg.get("evaluate.rollouts_per_candidate", 3) or 3)
    if not cfg.get("alignment_rate.enabled", False):
        return k
    if "alignment_rate" not in (cfg.get("post") or []):
        return k
    return max(k, int(cfg.get("alignment_rate.n_videos") or 1) * max(1, int(n_seeds)))


def _retrain_config(cfg: Config, env_steps: int, n_rollouts: Optional[int] = None) -> Config:
    """A config for the retrain only: from scratch, unpruned, no secondary buffer.

    Both published protocols force this. LIMEN retrains the winner from scratch
    over 10 seeds *specifically* to remove post-selection bias -- warm-starting
    from the checkpoint the selection produced would carry that bias straight
    into the reported number. GT retrains from scratch with no secondary buffer
    for the same reason (§3 `final_retrain.*`).

    `train.pruning` is forced to `none` because a pruner is a SEARCH economy:
    it stops a candidate that is losing to its cohort so the budget goes to the
    ones that are not. The retrain has no cohort -- it is one reward, trained
    `n_seeds` times for `env_steps` each, to REPORT a number. Inheriting the
    search's rule would let, e.g., `train.pruning: median_stop` fire inside the
    retrain on the retrain's own history, so the "from scratch for `env_steps`"
    training would run a fraction of its stated steps while
    `final_retrain.json` recorded `env_steps_per_seed` at the configured count.
    `pruned` on the artifact records that no seed was cut, so the claim is
    checkable.

    `n_rollouts` raises `evaluate.rollouts_per_candidate` for the retrain only,
    to whatever the `post:` phases downstream will read (see `_report_rollouts`).
    It is applied around the BACKEND CALL and not around the scoring call, so
    the retrained fitness is still computed over the search's own K and stays
    comparable with the selected fitness it is reported against.

    The seeds are the other half of "from scratch" and are NOT patched here:
    `run_final_retrain` hands the backend `seed_phase=_SEED_PHASE`, and
    `training._seed_base` salts that phase's stream off the winner's own search
    stream, so the run's `seed` stays what it is.

    "From scratch" strips what is a PRODUCT OF THE SELECTION and nothing else.
    `warm_start_from_best` and `secondary_replay_buffer` hand
    the retrain the winner's own checkpoint or buffer -- the bias LIMEN and GT
    retrain to remove -- so both become `from_scratch`. `bc_prior` is a FIXED
    behaviour-cloned prior of the task's scripted policy, built once per run
    from the run seed and identical for every candidate of every iteration; it
    is not a product of the selection, and the retrain keeps it. Dropping it
    would make the search compare anchored fine-tunes of the clone while the
    report measured a from-scratch PPO of the same reward -- a different
    learner.
    `bc_prior_then_warm_start` is pinned to `bc_prior`: the chain's round-0
    shape (the clone) without its later rounds (the selected winner).
    `train.anchor.kind` hangs off whatever the candidate STARTS from and
    `train.elite_constraint.kind: l2_params` off the elite the warm start
    loaded, so the anchor survives exactly when the prior does and the
    constraint never does. Those are the pairings `_check_coherence` refuses at
    load, and `Config(...)` does not validate, so `tests/test_final_retrain.py`
    validates the retrain config of every shipped config.

    The run's own config object is left untouched: it is the hashed artifact
    that identifies the run, so a phase must not edit it.
    """
    init = str(cfg.get("train.init", "from_scratch") or "from_scratch")
    retrain_init = ("bc_prior" if init in ("bc_prior", "bc_prior_then_warm_start")
                    else "from_scratch")
    train_patch: Dict[str, Any] = {
        "env_steps": int(env_steps),
        "init": retrain_init,
        "pruning": "none",
        # ... and with it the two knobs only a rule reads: the demonstration
        # ceiling guards a rule and `demo_fraction` is a rule's metric
        # (in a config that enables pruning), so left as written they are
        # exactly the declared-but-inert pairing `_check_coherence` refuses.
        # The retrain runs the whole budget; nothing here is lost.
        "pruning_metric": "own_reward",
        "pruning_cfg": {"ceiling": "none"},
        "secondary_buffer": {"ratio": 0.0},
        # No elite: the elite is the selection's product, whatever the init.
        "elite_constraint": {"kind": "none"},
        # ROSKA's fusion block goes with the init it hangs off. `fused_warm_start`
        # blends the selection's own winner into the starting weights, so it is a
        # product of the selection and becomes `from_scratch` above -- and a
        # `ratio_search` left behind then searches a blend ratio for a blend that
        # does not happen. `_check_coherence` refuses exactly that pairing
        # ("train.fusion.ratio_search=sc_bo has no effect unless
        # train.init=fused_warm_start"); `configs/methods/roska.yaml` exercises it
        # (`test_every_shipped_configs_retrain_config_is_itself_a_valid_config`).
        # Reset to the inert default rather than tolerated by the rule, because a
        # retrain that reported a searched alpha it never searched would record a
        # setting that did nothing -- and the probes are real env steps, which a
        # report protocol must not spend.
        "fusion": {"ratio_search": "fixed"},
    }
    if retrain_init == "from_scratch":
        # Nothing to anchor to -- the pairing `_check_coherence` refuses.
        train_patch["anchor"] = {"kind": "none"}
    patch: Dict[str, Any] = {"train": train_patch}
    if n_rollouts is not None:
        patch["evaluate"] = {"rollouts_per_candidate": int(n_rollouts)}
    patched = deep_merge(cfg.to_dict(), patch)
    return Config(patched, source=cfg.source, lineage=cfg.lineage)


#: The `seed_phase` the retrain hands the backend (`training._phase_salt`).
_SEED_PHASE = "final_retrain"


def _recorded_seeds(result: Optional[TrainResult]) -> List[int]:
    """The seeds a `TrainResult` actually trained, off `seed_metrics[*].seed`
    (every backend writes it). `[]` for no result or a result without rows."""
    if result is None:
        return []
    out: List[int] = []
    for m in getattr(result, "seed_metrics", None) or []:
        if isinstance(m, dict) and m.get("seed") is not None:
            out.append(int(m["seed"]))
    return out


#: `final_retrain.max_env_steps` sentinel: cap at the launch-resolved `train.env_steps`.
SAME_AS_TRAIN = "same_as_train"


def _retrain_budget(cfg: Config) -> Tuple[int, Optional[int], int]:
    """(pinned, cap, effective) env steps per retrain seed.

    `pinned` is the method's number: `final_retrain.env_steps`, or
    `train.env_steps` when it is null (the default). `cap` is the profile's
    `final_retrain.max_env_steps`: null = no
    cap (the published point), an int, or `same_as_train` = the resolved
    `train.env_steps` (what every profile sets). `effective` = min(pinned, cap)
    is what trains. Kept as one function so a wall-time estimate and the
    phase cannot disagree about what a run will spend.
    """
    pinned = int(cfg.get("final_retrain.env_steps") or cfg["train.env_steps"])
    cap = cfg.get("final_retrain.max_env_steps")
    if cap == SAME_AS_TRAIN:
        # The cap follows the RESOLVED search budget, so `--set train.env_steps=N`
        # on the submit line moves it too. A literal in the profile file would be
        # the profile's budget at authoring time and could cut a 2 M
        # HumanoidBench retrain to the dev profile's 20 k.
        cap = int(cfg["train.env_steps"])
    elif cap is not None:
        cap = int(cap)
    effective = min(pinned, cap) if cap is not None else pinned
    return pinned, cap, effective


def _score_native(ctx: Context, state: Optional[RunState], result: TrainResult
                  ) -> Tuple[Optional[CandidateReport], str, str]:
    """Score a retrain result on the native signal, beside the method's own fitness.

    Returns `(report, channel, error)`: the report and the resolved channel
    (`native_success` / `native_reward`) when the task ships one; `(None, "",
    reason)` when it does not (the five unsupervised-only gym tasks) or the
    scorer failed. Never raises -- this is a second measurement on a result the
    phase already has, and it must not be able to fail the retrain it decorates.
    """
    try:
        reports = registry_get("fitness_source", "native")(ctx, state, [result])
    except Exception as exc:  # noqa: BLE001 -- recorded, never re-raised
        return None, "", f"{type(exc).__name__}: {exc}"
    if not reports:
        return None, "", "native scorer returned no report"
    rep = reports[0]
    return rep, str(rep.fitness_source or ""), ""


def _per_seed_task_metric(result: Optional[TrainResult]) -> List[Optional[float]]:
    """Each retrain seed's `env.task_metric` at the checkpoint it shipped.

    Read straight off `TrainResult.seed_metrics`, where the backend already
    put it: a seed row's `fitness` IS the task metric (`training.py`'s
    `_PRUNING_FIELDS` maps `task_metric` -> `fitness`), NOT the method's fitness
    source -- that is `CandidateReport.per_seed_fitness`, a different quantity
    one word away, and the two are easy to confuse.

    THE VALUE IS THE SHIPPED POINT, NOT A MAX OVER THE CURVE. A seed row's
    `fitness` is `env.task_metric` at the checkpoint that seed actually
    shipped, so a run whose curve peaked and then regressed reports the
    regression: measured on a real retrain, 0.0 in all three slots at
    checkpoint 20 against curve maxima of 0.155 / 0.060 / 0.067. A table that
    puts this beside a max-aggregated column is comparing two different
    statistics, and three true zeros will read as agreement with nothing.

    Returns `[]` when the retrain produced no result (budget, error), and
    `None` in a slot whose seed row carries no value, so the list length is
    always the number of seeds that ran and a missing value never reads as a
    zero score. Never raises: this decorates a retrain and must not fail it.
    """
    # A WRAPPED WORKER PAYLOAD IS NOT A MISSING RESULT, and without this check
    # the two would be the same []. A training worker's payload is a dict
    # wrapping the result ({"result": TrainResult, ...}, as
    # `training.build_child_payload` builds it); `getattr(a_dict,
    # "seed_metrics", None)` is None, so the wrapper would fall through
    # `or []` and return the empty list that means "the retrain produced
    # nothing". The phase itself is unaffected
    # -- it holds the unwrapped result -- but a caller passing a worker payload
    # would otherwise read it as an empty retrain and never know.
    #
    # It LOGS AND RETURNS [] rather than raising, deliberately: the contract
    # three lines up is that this never raises because it decorates a retrain
    # and must not fail it, and a guard that can kill a retrain is a worse
    # trade than a loud line. The log is the whole point -- it makes the two
    # states distinguishable to anyone reading the output.
    try:
        # ANY MAPPING IS WRONG HERE, so the type IS the whole guard and there
        # is no key probe at all. `TrainResult` is a dataclass and not a
        # Mapping (bird/types.py), so a mapping arriving here is a worker's
        # envelope by construction -- there is nothing to look inside for.
        #
        # Probing a key instead would identify by key a thing the TYPE already
        # identifies, and both obvious probes are wrong. `"result" in result`
        # calls an arbitrary Mapping's __contains__ and can raise, which is why
        # the guard sits inside the try. `.get("result") is not None` SILENTLY
        # MISSES `{"result": None, ...}` -- a plain dict, and exactly what a
        # FAILED job's envelope looks like -- which would fall through to the
        # getattr below and return [] with no log. `isinstance` alone cannot
        # raise, cannot be fooled by a null payload, and needs no test for its
        # key handling because it has none.
        if isinstance(result, Mapping):
            log.warning(
                "per_seed_task_metric: wrapped payload, unwrap first -- got a %s "
                "(a worker's envelope) rather than a TrainResult; returning [] , "
                "which otherwise means the retrain produced nothing",
                type(result).__name__
            )
            return []
        rows = getattr(result, "seed_metrics", None) or []
        out: List[Optional[float]] = []
        for row in rows:
            v = row.get("fitness") if isinstance(row, dict) else None
            out.append(None if v is None else float(v))
        return out
    except Exception as exc:  # noqa: BLE001 -- a decoration must not fail the phase
        # NO PATH RETURNS [] SILENTLY. Without this line the function has three
        # states wearing one value: [] for no result, [] for a logged wrapper,
        # and [] for something that raised (e.g. a malformed seed row) -- and
        # only two of them would say anything.
        log.warning("per_seed_task_metric: returning [] after %s: %s -- this is NOT "
                    "'the retrain produced no result'; the read itself failed",
                    type(exc).__name__, exc)
        return []


#: Where `run_final_retrain` puts the retrain's per-checkpoint curves,
#: relative to `<rundir>/phases/`. A FILE IN A DIRECTORY BESIDE
#: `final_retrain.json`, never more keys inside it: the record is the thing a
#: reader opens and greps, and `n_seeds` x `train.checkpoint_interval` curve
#: points would bury the twenty scalars it exists to show. The phase file
#: carries the path (`checkpoint_series`) so the two are joined by the
#: artifact rather than by a convention someone has to know.
CHECKPOINT_SERIES = "final_retrain/checkpoints.json"


def _checkpoint_series(result: Optional[TrainResult]) -> Optional[Dict[str, Any]]:
    """The retrain's per-checkpoint curves, per seed, as the backend wrote them.

    WHAT THIS IS FOR. CARD's Fig. 3 (`fig:metaworld_comparison`) compares
    SUCCESS-RATE TRAINING CURVES, not a ranking scalar -- and under
    `loop.termination: fixed_generations` the returned candidate is never
    search-trained (the final iteration is generation-only, `returned_trained:
    false`, 0 checkpoints in its `train_result.json`). Its ONLY training is
    this phase's, and endpoints alone (`per_seed_task_metric`,
    `per_seed_native`, `retrained_native`) would leave the returned reward with
    no curve anywhere in the artifact, so the one quantity the paper plots
    would not be recoverable from a finished run. Every other candidate has
    one, written by `RunDir.save_train_result`; without this series the
    returned one would not, and a comparison would be tempted to reach for an
    in-loop candidate's curve under the returned candidate's id.

    NOTHING NEW IS COMPUTED OR EVALUATED, exactly as
    `per_seed_task_metric` computes nothing: `seed_metrics[i]["checkpoints"]`
    is the curve the backend already built for seed i, at
    `evaluate.checkpoint_eval_episodes` episodes per point (the retrain runs
    through the same `train_backend` as the search and `_retrain_config` does
    not touch that key), and `TrainResult.checkpoints` is its across-seed mean.
    They are COPIED VERBATIM -- same records, same keys, same order -- so the
    retrain's curve and an in-loop candidate's curve are the same object in two
    files and a reader can put them on one axis without a reshape. Reshaping
    here is the one thing that would make them incomparable, which is the whole
    reason the phase needed a curve in the first place.

    `None` when there is no curve to write -- no result (budget, error), or a
    backend that fills neither field -- so an absent file means "no series",
    never "a series of nothing", and `checkpoint_series` on the record stays
    null rather than pointing at an empty list.
    """
    if result is None:
        return None
    per_seed: List[Dict[str, Any]] = []
    for row in getattr(result, "seed_metrics", None) or []:
        if not isinstance(row, dict):
            continue
        curve = row.get("checkpoints")
        if curve is None:
            continue
        per_seed.append({"seed": row.get("seed"), "checkpoints": list(curve)})
    mean = list(getattr(result, "checkpoints", None) or [])
    if not per_seed and not mean:
        return None
    # `mean` is `TrainResult.checkpoints`, which is what `train_result.json`
    # writes at its top level -- the same field under the same name, so the two
    # files do not disagree about which curve is which.
    return {"per_seed": per_seed, "mean": mean}


@register("phase", "final_retrain")
def run_final_retrain(ctx: Context, state: Optional[RunState] = None) -> None:
    """Retrain the returned reward from scratch, `n_seeds` times (§3).

    A report protocol, not a search step -- which is why it is a `post:` phase
    and why `final_retrain.n_seeds` is a third, distinct seed count from
    `loop.n_restarts` and `train.seeds_per_candidate` (the three are never to
    be conflated).

    LIMEN's reason is the sharp one: the winner was chosen by argmax over noisy
    single-run fitnesses, so its selected fitness is optimistically biased, and
    the only way to report an unbiased number is to train it again from scratch
    on fresh seeds. The artifact therefore records BOTH numbers and the gap
    between them -- the gap is the quantity LIMEN is buying with this phase, and
    dropping it would leave the bias unmeasured all over again.

    "Fresh" is enforced, not assumed: the backend is handed
    `seed_phase=_SEED_PHASE`, which `training._seed_base` salts off the
    winner's own search stream, and the artifact records `seeds`,
    `search_seeds` and `seed_overlap` so the claim is checked rather than
    re-derived.

    `final_retrain.env_steps: null` reuses `train.env_steps`. The profile-owned
    `final_retrain.max_env_steps` caps what is actually trained:
    `min(pinned, cap)` per seed, with `env_steps_pinned`, `env_steps_cap` and
    `env_steps_per_seed` all written to `phases/final_retrain.json`, so a config
    that pins a published 2.62 B retrain (eureka) stays faithful as a file and
    affordable under `dev`/`full` (`_retrain_budget`).
    """
    cfg = ctx.cfg
    if not cfg.get("final_retrain.enabled", False):
        log.info("    final_retrain: final_retrain.enabled=false -> skipped")
        return
    if state is None:
        log.warning("    final_retrain: no RunState to retrain from")
        return

    rule = cfg.get("select.final_artifact", "global_best")
    report = _FINAL_ARTIFACT_RULES.get(rule, _FINAL_ARTIFACT_RULES["global_best"])(state)
    if report is None:
        log.warning("    final_retrain: %s selected nothing to retrain", rule)
        return

    n_seeds = int(cfg.get("final_retrain.n_seeds") or 1)
    pinned_steps, cap_steps, env_steps = _retrain_budget(cfg)
    if cap_steps is not None and env_steps < pinned_steps:
        log.info("    final_retrain: env_steps %d capped to %d per seed by "
                 "final_retrain.max_env_steps (profile); the pinned budget is recorded",
                 pinned_steps, env_steps)
    backend = registry_get("train_backend", cfg["train.backend"])

    before_trainings = ctx.budget.policy_trainings
    before_steps = ctx.budget.env_steps

    result: Optional[TrainResult] = None
    retrained: Optional[CandidateReport] = None
    native_rep: Optional[CandidateReport] = None
    native_channel = ""
    native_error = ""
    error = ""
    error_type = ""
    n_rollouts = _report_rollouts(cfg, n_seeds)
    retrain_cfg = _retrain_config(cfg, env_steps, n_rollouts)
    # Pin the incumbent HERE TOO, not only in `bird.train`. A post phase can be
    # the ONLY thing a leg runs -- `loop.resume_from` on a finished search goes
    # straight here -- and then nothing has called `pin_refs` in this process,
    # so the store's eviction is unguarded for exactly the trainings whose
    # numbers get reported. `warm_start_from_best` and `secondary_replay_buffer`
    # both reach the store from a retrain seed.
    from . import training as _training
    _training.pin_refs(state)
    prev_cfg = ctx.cfg
    try:
        # THE WHOLE RETRAIN IS ONE UNIT UNDER THE CAP, asked here rather than
        # inside a seed loop, and that placement is the point.
        #
        # In the search the caps bind BEFORE the work, per seed, in both
        # sequential seed loops: one seed is one training. That is right for
        # the search, where seeds are replicates and k of n is a noisier
        # measurement. It is wrong here. A retrain is a REPORT PROTOCOL --
        # `final_retrain.n_seeds` is the third, distinct seed count precisely
        # because it is not the other two -- so k seeds of a requested n is
        # not a partial result, and the run would record k paid trainings on
        # one schedule against zero on another for the same config.
        #
        # ABOVE THE BACKEND, so every path answers the same way: sequential
        # and the per-seed fan-out. Placing it in a seed loop would also
        # change the inner loop's per-candidate check.
        #
        # `gpu_hours_exhausted` beside it because a training's duration is not
        # knowable in advance, so that cap's enforceable rule is "do not START
        # once it is spent".
        #
        # The raise is caught by this function's own `except BudgetExceeded`
        # below, which records the error on the artifact and leaves
        # `retrained_fitness` null -- a measurement that did not happen reads
        # as one that did not happen, which is already this phase's rule for
        # a spent budget.
        _why = (ctx.budget.would_exceed_training(n_seeds)
                or ctx.budget.gpu_hours_exhausted())
        if _why:
            raise BudgetExceeded(
                f"final_retrain needs {n_seeds} trainings and {_why}; the "
                "retrain is refused whole rather than run with fewer seeds "
                "than the report protocol asks for")
        if retrain_cfg["train.init"] == "bc_prior":
            # Re-pin the clone before the backend call: a long search's FIFO
            # store may have evicted it, and `ensure_bc_prior` re-inserts the
            # blob or rebuilds it deterministically from the run's config and
            # seed -- so the retrain starts from the same clone every search
            # candidate did, rather than cold-starting behind a log.warning.
            from .training import ensure_bc_prior
            ensure_bc_prior(ctx, state, cfg)
        # Rollouts under the widened config; scoring under the run's own, so the
        # reported fitness is computed over the same K the search used.
        ctx.cfg = retrain_cfg
        try:
            # `seed_phase` puts the retrain on a seed stream disjoint from the
            # one the search trained this candidate on (`training._seed_base`):
            # "fresh seeds" is the whole argument for the phase, and without
            # the salt a last-round winner would get exactly its search seeds
            # back.
            result = backend(ctx, state, report.candidate, n_seeds=n_seeds,
                             seed_phase=_SEED_PHASE)
        finally:
            ctx.cfg = prev_cfg
        score = registry_get("fitness_source", cfg["evaluate.fitness.source"])
        scored = score(ctx, state, [result])
        retrained = scored[0] if scored else None
        native_rep, native_channel, native_error = _score_native(ctx, state, result)
    except BudgetExceeded as exc:
        # The search is already over; a spent budget must not turn the run's
        # reported result into a crash.
        error = f"budget exhausted during final retrain: {exc}"
        error_type = "budget"
        log.warning("    final_retrain: %s", error)
    except Exception as exc:  # noqa: BLE001 -- see below: recorded, never re-raised
        # Same argument, every other failure. This is a REPORT PROTOCOL over a
        # search that has already finished. A retrain seed worker dying
        # (`_fork_seeds` fatal payload -> `_sb3_run` raises RuntimeError, which
        # under `candidate_parallelism: parallel` happens in the parent) would
        # otherwise escape through `_execute` BEFORE `result.json` is composed:
        # `run()` would stamp the whole run `failed` and a long search would
        # read as one that never finished, with no returned candidate anywhere.
        # The search loop contains the same worker death as one failed
        # candidate. So: the error is recorded on this artifact, journalled
        # under its own event name, and handed to `_execute` for `result.json`
        # (`post_phase_errors`), and `retrained_fitness` stays null -- the
        # measurement that did not happen reads as one that did not happen.
        error = f"{type(exc).__name__}: {exc}"
        error_type = type(exc).__name__
        log.exception("    final_retrain failed (%s); the search's own result is kept",
                      error)
        ctx.event("final_retrain_failed", cand_id=report.cand_id, error=error,
                  error_type=error_type)
    finally:
        ctx.cfg = prev_cfg
    if error:
        errors = ctx.counters.setdefault("post_phase_errors", {})
        errors["final_retrain"] = error

    selected_fitness = report.fitness
    retrained_fitness = retrained.fitness if retrained else None
    gap = (retrained_fitness - selected_fitness
           if retrained_fitness is not None and selected_fitness is not None else None)

    # The ACTUAL seeds, both sides, so "fresh" is a recorded fact and not a
    # property of the seed arithmetic someone has to re-derive. A backend that
    # ignores `seed_phase` shows up here as a non-empty overlap.
    # THE TWO SEED-SCHEDULE NAMES, FROM THE CONSTANT, not spelled here. This
    # record is a THIRD writer of these fields; hardcoded names would survive
    # until a rename, when the two writers that follow the constant would move
    # and this one would not, and `final_retrain.json`'s field would silently
    # stop being populated -- the same drift that can open between the journal
    # and the seed row.
    from bird.components.training import SEED_SCHEDULE_FIELDS
    _sched_workers_key, _sched_reason_key = SEED_SCHEDULE_FIELDS

    seeds = _recorded_seeds(result)
    search_seeds = _recorded_seeds(report.result)
    seed_overlap = sorted(set(seeds) & set(search_seeds))
    if seed_overlap:
        log.warning("    final_retrain: %d of %d retrain seed(s) repeat the search's "
                    "(%s); the reported number is not free of post-selection bias",
                    len(seed_overlap), len(seeds), seed_overlap)

    payload = {
        "final_artifact_rule": rule,
        "cand_id": report.cand_id,
        # The restart this phase was handed -- the one that OWNS the returned
        # artifact (`_execute` picks it; `result.json: returned_restart` names
        # the same one). Recorded so the join is checkable: the phase must
        # retrain the returned artifact's restart, not simply the last one.
        "restart": getattr(state, "restart", None),
        "n_seeds": n_seeds,
        # HOW THE SEEDS WERE SCHEDULED, and why if it was downgraded.
        # `n_seeds` above says how many ran; it does not say whether they ran
        # side by side or one after another, and on the batched backends they are
        # forced sequential (seed workers fork IN THE PARENT, where jax is
        # initialised). Without this, a reader of THIS FILE sees a
        # retrain that cost three times the wall-clock they expected and has
        # to go to the journal to find out it was not a regression. The
        # journal already carries it; this is the artefact someone actually
        # opens when they are looking at the retrain.
        #
        # Read off the seed rows the backend wrote rather than re-derived, so
        # this cannot disagree with them. Same two names everywhere
        # (`training.SEED_SCHEDULE_FIELDS`).
        _sched_workers_key: next(
            (m.get(_sched_workers_key) for m in (result.seed_metrics if result else [])
             if m.get(_sched_workers_key) is not None), None),
        _sched_reason_key: next(
            (m.get(_sched_reason_key) for m in (result.seed_metrics if result else [])
             if m.get(_sched_reason_key)), ""),
        "env_steps_per_seed": env_steps,
        "env_steps_pinned": pinned_steps,
        "env_steps_cap": cap_steps,
        # As CONFIGURED for the retrain (`_retrain_config`): `from_scratch`, or
        # `bc_prior` when the search started every candidate from the fixed clone.
        "train_init": retrain_cfg["train.init"],
        "train_anchor": retrain_cfg.get("train.anchor.kind", "none"),
        # As EXECUTED, per seed, off the seed rows the backend wrote: whether the
        # prior actually loaded and whether the anchor actually attached. A prior
        # that did not load and a `from_scratch` retrain are the same curve;
        # only these fields tell them apart after the fact.
        "executed": [{"seed": m.get("seed"), "init": m.get("init"),
                      "warm_started_from": m.get("warm_started_from", ""),
                      "anchor": m.get("anchor", {})}
                     for m in (result.seed_metrics if result else [])
                     if isinstance(m, dict)],
        "train_pruning": "none",
        # Whether ANY seed was cut short by a pruner -- must be False since the
        # retrain config forces `pruning: none`; recorded rather than assumed so
        # `env_steps_per_seed` above cannot silently overstate what was trained.
        "pruned": (bool(result.pruned) if result is not None else None),
        "secondary_buffer": "disabled",
        "seed_phase": _SEED_PHASE,
        "seeds": seeds,
        "search_seeds": search_seeds,
        "seed_overlap": seed_overlap,
        "selected_fitness": selected_fitness,
        # Was the returned program trained IN THE SEARCH at all? Under
        # `loop.termination: fixed_generations` (CARD) the chain end is returned
        # from a generate-only iteration, so this retrain is the program's FIRST
        # training and the reported number cannot be a post-selection gap --
        # `post_selection_gap` stays null, and CARD's five-seed figure captions
        # are `retrained_fitness` / the seed rows, not a correction to anything.
        # A TPE rejection under `verify.tpe.on_failure: skip_training` also
        # lands here as False; `report.result.skip_reason` tells the two apart.
        "selected_trained": (bool(report.result.trained)
                             if report.result is not None else None),
        "retrained_fitness": retrained_fitness,
        "post_selection_gap": gap,
        "per_seed_fitness": list(retrained.per_seed_fitness) if retrained else [],
        # THE SAME RETRAIN, SCORED ON THE ENV'S OWN SIGNAL (rule N) -- recorded
        # whatever `evaluate.fitness.source` the method searches with.
        # `per_seed_fitness` is in the METHOD's units (a VLM score for rda,
        # nothing at all for CARD's `none`), so without this the headline
        # number for a VLM-judged or unscored method would be unrecoverable
        # from the artifact. `native_channel` names which quantity this IS
        # (native_success / native_reward); empty with `native_error` set on a
        # task that ships neither, where the number honestly does not exist.
        "native_channel": native_channel,
        "retrained_native": (float(native_rep.fitness)
                             if native_rep is not None and native_rep.fitness is not None
                             else None),
        "per_seed_native": (list(native_rep.per_seed_fitness) if native_rep is not None
                            else []),
        "native_error": native_error,
        # THE SAME RETRAIN, SCORED ON THE TASK METRIC -- `env.task_metric`, the
        # ground-truth quantity search-time figures use. NEITHER of the two
        # quantities above is the task metric. `per_seed_fitness` is the
        # METHOD'S OWN fitness source (a VLM score for rda, nothing at all for
        # CARD's `none`), and `per_seed_native` is the env's native signal,
        # which on Assistax resolves to `native_reward` -- the reference-reward
        # return. A retrained table built from those alone would compare
        # methods on the shipped/reference reward while the search-time figures
        # compare them on task success.
        #
        # NOTHING NEW IS COMPUTED HERE. `env.task_metric` is already evaluated
        # in the greedy checkpoint eval (`training.py`, `metrics["fitness"] +=
        # float(env.task_metric(traj))`) and rides home on the TrainResult as
        # `seed_metrics[*]["fitness"]` -- the shipped checkpoint's row. This
        # reads it off the result the phase is already holding.
        #
        # THE NAME IS THE TRAP: a seed row's
        # `fitness` key IS the task metric (`_PRUNING_FIELDS` maps
        # `task_metric` -> `fitness`), while `CandidateReport.per_seed_fitness`
        # is the method's fitness. Two different quantities, one word apart.
        "per_seed_task_metric": _per_seed_task_metric(result),
        "cost": {
            "policy_trainings": ctx.budget.policy_trainings - before_trainings,
            "env_steps": ctx.budget.env_steps - before_steps,
            "wallclock_s": round(result.wallclock_s, 3) if result else 0.0,
        },
        "error": error,
        # `""` (none), `"budget"`, or the exception class that stopped the retrain.
        "error_type": error_type,
    }
    # Stash the LIVE TrainResult for a downstream metric phase. It is handed
    # over in memory, never through an artifact: `RunDir.save_train_result`
    # drops trajectories on purpose, and `bird/state.py`'s opening note forbids
    # reading rollouts off a CARRIED report because a resumed run would not
    # have them -- which is exactly the divergence `alignment_rate` hit when it
    # first tried. A result produced in this process, this run, is neither.
    if result is not None:
        ctx.counters["final_retrain_result"] = result
    # THE CURVE, BESIDE THE ENDPOINTS -- written FIRST so the record's
    # `checkpoint_series` never names a file that is not there. Additive in
    # both directions: nothing above is read, nothing above moves, and a run
    # whose retrain produced no curve writes no file and records null, which
    # is what every reader of this record already does with a null field.
    series = _checkpoint_series(result)
    wrote_series = series is not None and _write_artifact(ctx, CHECKPOINT_SERIES, {
        "cand_id": report.cand_id,
        "seed_phase": _SEED_PHASE,
        "n_seeds": n_seeds,
        "env_steps_per_seed": env_steps,
        # THE KEY AS CONFIGURED, null included, and not an episode count
        # derived here. null means "each backend's own constant"
        # (`training._checkpoint_episodes`), and which constant depends on the
        # backend and on whether the point is the last one -- so a number
        # spelled here would be this function's guess at another function's
        # rule, and would go stale the first time that rule moved. The
        # retrain honours the key by running through the same backend the
        # search did; `_retrain_config` leaves `evaluate.*` alone except for
        # the rollout widening `_report_rollouts` asks for.
        "checkpoint_eval_episodes": cfg.get("evaluate.checkpoint_eval_episodes"),
        **series,
    }) is not None
    # RELATIVE TO `phases/`, never the absolute path `_write_artifact` returns:
    # run dirs are archived, moved and re-hosted, and an absolute path recorded
    # on one machine is a dangling reference everywhere else.
    payload["checkpoint_series"] = CHECKPOINT_SERIES if wrote_series else None
    _write_artifact(ctx, "final_retrain.json", payload)
    ctx.event("final_retrain", **{k: v for k, v in payload.items() if k != "cost"})
    log.info("    final_retrain -> %s: selected %s, retrained %s over %d seed(s)"
             " (gap %s)", report.cand_id,
             "n/a" if selected_fitness is None else f"{selected_fitness:.4f}",
             "n/a" if retrained_fitness is None else f"{retrained_fitness:.4f}",
             n_seeds, "n/a" if gap is None else f"{gap:+.4f}")


@register("phase", "alignment_rate")
def run_alignment_rate(ctx: Context, state: Optional[RunState] = None) -> None:
    """RDA's reported metric (§5.1, App. C): how well the FINAL policy's
    behaviour matches the task instruction, judged by a VLM from video.

    A REPORT PROTOCOL, not a search step -- the same reason `final_retrain` is a
    `post:` phase. Nothing here can change which reward was selected; it
    measures the one that was.

    Why it is not `evaluate.vlm.*` with different numbers. RDA runs TWO VLM
    scorers and pins them differently: the in-loop one (§4.2) scores each
    (trajectory, subtask) pair on `s in [0,1]`, 20 images, once; the reported
    one (§5.1) rates whole-video *instruction alignment* on a 5-point Likert
    scale, 50 images, four times per video, over 5 videos and 3 seeds -- 60
    judgments per task. One key cannot hold both, and pinning the reporting
    numbers on the shared key would quadruple the search's VLM bill for a
    measurement the search never reads. So this block is separate, and
    `configs/methods/rda.yaml` is the config where that separation is visible.

    NORMALISATION IS A PIN, NOT A DETAIL. App. C says the mean is "normalized to
    the range [0,1] by dividing by 5", so a Likert 1 maps to **0.2**, not 0.
    Min-max would map it to 0 and shift every reported number; the two are
    offered as `divide_by_max` (the paper) and `min_max`, and the default is the
    paper's. Getting this wrong makes our alignment rates quietly
    incomparable with the published ones, which is the only thing they are for.

    The rating is whole-video against the instruction -- deliberately NOT
    per-subtask. The in-loop scorer decomposes because it is generating credit
    for revision; the metric does not, because it is answering "did this policy
    do what was asked".
    """
    cfg = ctx.cfg
    if not cfg.get("alignment_rate.enabled", False):
        log.info("    alignment_rate: alignment_rate.enabled=false -> skipped")
        return

    evaluator = getattr(ctx, "evaluator", None)
    if evaluator is None:
        # No VLM client is a legitimate configuration (a mock run, or a tester
        # overlay). Record that the metric was not computed rather than
        # emitting a zero, which would read as perfect misalignment.
        _write_artifact(ctx, "alignment_rate.json",
                        {"status": "not_run", "reason": "no evaluator client",
                         "alignment_rate": None})
        log.info("    alignment_rate: no evaluator client -> not run")
        return

    rule = cfg.get("select.final_artifact", "global_best")
    report = _FINAL_ARTIFACT_RULES.get(rule, _FINAL_ARTIFACT_RULES["global_best"])(state) \
        if state is not None else None
    if report is None:
        _write_artifact(ctx, "alignment_rate.json",
                        {"status": "not_run", "reason": "no final artifact selected",
                         "alignment_rate": None})
        log.warning("    alignment_rate: %s selected nothing to evaluate", rule)
        return

    scale = str(cfg.get("alignment_rate.scale", "[0,1]"))
    lo, hi = (1.0, 5.0) if scale == "likert" else (0.0, 1.0)
    n_videos = max(1, int(cfg.get("alignment_rate.n_videos") or 1))
    repeats = max(1, int(cfg.get("alignment_rate.repeats") or 1))
    n_images = int(cfg.get("alignment_rate.images_per_query") or 20)
    instruction = cfg.get("problem.task_description", "")

    # WHERE THE VIDEOS COME FROM, and why not the obvious place. The obvious
    # place is `report.result.trajectories` -- and it is wrong. `bird/state.py`
    # opens with the invariant that nothing downstream may read a CARRIED
    # report's rollouts: they are dropped from the checkpoint and from the
    # artifact, so a resumed run has none. Reading them would make this phase
    # compute a rate after a straight run and `not_run` after a resume -- the
    # same metric, two answers, decided by whether the job was interrupted.
    # That is precisely the "silently divergent" failure that note predicts,
    # and `tests/test_resume.py` guards it by comparing whole run directories.
    #
    # So the rollouts must be FRESH, produced in this process. `final_retrain`
    # makes exactly that and hands its live TrainResult over in memory.
    # This also happens to be what the paper does: App. C evaluates each
    # trained policy over 3 seeds, which is `final_retrain.n_seeds: 3`.
    # `n_videos` IS PER TRAINED POLICY, and there are `final_retrain.n_seeds` of
    # them: App. §10.1 is "5 trajectory videos [per policy] ... 4 queries per
    # video, 20 evaluations per policy ... 3 random seeds ... 60 total
    # evaluations per task". `_sb3_run` cycles its final rollouts over the seed
    # policies, so `n_videos * n_seeds` rollouts is `n_videos` of each -- which
    # is what `phases._report_rollouts` asks the retrain to produce (RDA's
    # SEARCH-side K=3 alone would give `[:5]` of 3).
    n_seeds = max(1, int(cfg.get("final_retrain.n_seeds") or 1))
    wanted = n_videos * n_seeds
    fresh = ctx.counters.get("final_retrain_result")
    available = list(getattr(fresh, "trajectories", None) or [])
    trajectories = available[:wanted]
    # A shortfall is still possible -- `final_retrain` may have been skipped, or
    # a backend may ignore the count -- so it stays logged and both numbers stay
    # in the artifact rather than being silently absorbed into `n_videos`.
    if len(available) < wanted:
        log.warning("    alignment_rate: wanted %d video(s) (%d per policy x %d "
                    "seed(s)), the retrained policy produced %d",
                    wanted, n_videos, n_seeds, len(available))
    if not trajectories:
        reason = ("final_retrain produced no rollouts"
                  if fresh is not None else
                  "no fresh policy to rate: put `final_retrain` before "
                  "`alignment_rate` in post: and enable it")
        _write_artifact(ctx, "alignment_rate.json",
                        {"status": "not_run", "reason": reason,
                         "alignment_rate": None})
        log.warning("    alignment_rate: %s", reason)
        return

    # THE JUDGE IS SENT PIXELS. App. §10.1's user prompt is literally
    # "## Video\n{video}" -- the metric is defined as a VLM watching a clip --
    # so interpolating a FILENAME into a sentence and attaching nothing would
    # not be the metric. `sample_frames` renders the rollout the
    # same way `record_rollouts` would; `final_retrain`'s trajectories are never
    # recorded to disk, so there is no file to read and the states are the only
    # source there has ever been.
    #
    # `_traj_digest` is deliberately NOT used here either: it prints the
    # environment's success flag and the candidate's own return, and this metric
    # exists precisely to say something the success flag does not (§10.2 makes
    # them the two complementary metrics). See `evaluation.judge_digest`.
    sees = client_sees_images(evaluator)
    n_blind = 0
    # Whose reward produced the policy being rated. `report` is the selected
    # candidate and is not None by here (`trajectories` is non-empty, which the
    # `not_run` branch above already required), but this metric must not be able
    # to fail over a label: `""` says "unattributed", which is what an artifact
    # with no selection would honestly be.
    judged_id = str(getattr(report, "cand_id", "") or "")
    # Rendering stays sequential -- `sample_frames` drives the shared env
    # renderer, which is stateful and (through MuJoCo) not thread-safe. Only
    # the judge ROUND-TRIPS overlap, one per (video, repeat), and the fold
    # below keeps the original loop order, so the artifact is judgment-for-
    # judgment what the serial loop wrote. The frame RECORD is written here
    # too, for the same reason: it describes the pixels, and the pixels are
    # produced before anything overlaps.
    per_video: List[Tuple[List[Any], str, str, str]] = []
    for vi, traj in enumerate(trajectories):
        # `prov` is `sample_frames`' out-parameter (see its docstring): the
        # provenance of the pixels this rating was made from. `-1` as the
        # iteration files the records under `judgments/post.jsonl` -- this is a
        # REPORT PROTOCOL over the final retrained policy and belongs to no
        # iteration of the search, and filing it under the last one would claim
        # it was part of that iteration.
        prov: Dict[str, Any] = {}
        images = sample_frames(ctx, traj, n_images, prov=prov) if sees else []
        if not images:
            n_blind += 1
            if not sees:
                prov = judge_trace.frame_set(
                    (), None, blind_reason="the evaluator client takes no images "
                                           "or is not modality: vlm")
        frame_set = judge_trace.note_frames(ctx, prov, iteration=-1,
                                            cand_id=judged_id, rollout=vi,
                                            png=images)
        cells = sheet_cells(prov) if images else None
        if cells:
            # A contact sheet is ONE image tiling many frames; "1 frames ... are
            # attached" would describe it as a frame and leave the burned numbers
            # unexplained. The `even` sentence is byte-identical.
            seen = (f"\nOne contact sheet tiling {cells} frames sampled evenly from "
                    f"the video is attached.\n{sheet_manifest(prov)}")
        else:
            seen = (f"\n{len(images)} frames sampled evenly from the video are attached."
                    if images else "")
        # Multi-view frames (`output.video.n_views > 1`) are described in the
        # spec's own words; `""` for a single-view frame keeps this prompt as it was.
        panels = multiview.describe(prov.get("views"))
        seen = f"{seen}\n{panels}" if panels else seen
        prompt = (
            f"Task instruction: {instruction}\n"
            f"Rollout: {judge_digest(traj)}{seen}\n"
            f"Rate how well the behaviour in this video follows the instruction "
            f"on a {lo}-{hi} scale "
            f"({lo}: complete misalignment, {hi}: perfect alignment).")
        per_video.append((images, prompt, frame_set,
                          str(prov.get("blind_reason") or "")))

    def _ask(vi: int, ri: int) -> Tuple[str, str]:
        images, prompt, frame_set, blind_reason = per_video[vi]
        # Written per QUERY, not per video: `repeats` asks the same question of
        # the same pixels, and a record per ask is what makes the repeats
        # countable in the artifact. Best-effort and file-locked inside
        # `judgments`, so it is safe to call from the pool.
        query_id = judge_trace.note_query(
            ctx, iteration=-1, role="alignment_rate",
            cand_id=judged_id, rollout=vi, repeat=ri,
            frame_set=frame_set, n_images=len(images),
            blind_reason=blind_reason, prompt=prompt)
        # The id travels with the text rather than being recomputed in the fold:
        # it is a digest of the call's identity, and re-deriving it there would
        # be a second copy of which fields identify a query -- the drift this
        # module already avoids by not re-implementing `judgments.read`.
        return _client_text(evaluator, prompt, images=images), query_id

    jobs = [(vi, ri) for vi in range(len(trajectories)) for ri in range(repeats)]
    workers = judge_concurrency(ctx, len(jobs))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="bird-align") as pool:
            texts = list(pool.map(lambda j: _ask(j[0], j[1]), jobs))
    else:
        # A generator: serial judgments happen as the fold consumes them,
        # matching the serial loop's interleaving with no materialised list.
        texts = (_ask(vi, ri) for vi, ri in jobs)
    judgments: List[Dict[str, Any]] = []
    for (vi, ri), (text, query_id) in zip(jobs, texts):
        score = _parse_score(text, lo, hi)
        reason = _parse_reason(text)
        # Here rather than inside `_ask`, because the answer is only half
        # recorded until it has been parsed: `raw` and `parsed` differ exactly
        # when the parse is lossy, and this metric DROPS an unparseable
        # judgment from its mean (`n_unparsed`). A response record whose
        # `parsed` is null is what makes that drop auditable per judgment
        # instead of only as a count.
        judge_trace.note_response(ctx, iteration=-1, query_id=query_id,
                                  raw=text, parsed=score, role="alignment_rate",
                                  cand_id=judged_id, rollout=vi, repeat=ri,
                                  reason=reason)
        judgments.append({"video": vi, "repeat": ri, "score": score,
                          "n_images": len(per_video[vi][0]),
                          "parsed": score is not None,
                          # The join back to `judgments/post.jsonl`: this file
                          # keeps the score, the trace keeps the prompt, the
                          # frames and the raw text, and nothing else connected
                          # a row here to the call that produced it.
                          "query_id": query_id,
                          "reason": reason})
    if n_blind:
        ctx.budget.record_blind_judgment(n_blind * repeats)
        log.warning("    alignment_rate: %d of %d video(s) were rated with NO frames "
                    "attached; those judgments are text-only",
                    n_blind, len(trajectories))

    scored = [j["score"] for j in judgments if j["score"] is not None]
    mean = sum(scored) / len(scored) if scored else None

    # See the docstring: `divide_by_max` is RDA's own wording and is NOT
    # min-max. They differ by a whole Likert step at the bottom of the scale.
    norm = str(cfg.get("alignment_rate.normalisation", "divide_by_max"))
    if mean is None:
        rate = None
    elif norm == "divide_by_max":
        rate = mean / hi if hi else None
    elif norm == "min_max":
        rate = (mean - lo) / (hi - lo) if hi > lo else None
    else:
        rate = mean

    payload = {
        "status": "ok",
        "cand_id": report.cand_id,
        "rated_policy": "final_retrain",
        "final_artifact_rule": rule,
        "instruction": instruction,
        "scale": scale,
        "scale_bounds": [lo, hi],
        "normalisation": norm,
        "n_videos": len(trajectories),
        "n_videos_requested": wanted,
        "n_videos_per_policy": n_videos,
        "n_seeds": n_seeds,
        "repeats": repeats,
        "images_per_query": n_images,
        "n_judgments": len(judgments),
        # A rate computed from frames and one computed from a text digest are
        # the same float and are not the same measurement.
        "n_blind_videos": n_blind,
        # Unparseable judgments are DROPPED from the mean and counted here, not
        # coerced to a bound. An invented score is a fabricated metric, and the
        # count is how a reader knows whether to trust the rate at all.
        "n_unparsed": sum(1 for j in judgments if not j["parsed"]),
        "mean_raw": mean,
        "alignment_rate": rate,
        "judgments": judgments,
    }
    _write_artifact(ctx, "alignment_rate.json", payload)
    ctx.event("alignment_rate", cand_id=report.cand_id, alignment_rate=rate,
              mean_raw=mean, n_judgments=len(judgments),
              n_unparsed=payload["n_unparsed"])
    log.info("    alignment_rate -> %s: %s over %d judgment(s)%s", report.cand_id,
             "n/a" if rate is None else f"{rate:.4f}", len(judgments),
             f" ({payload['n_unparsed']} unparsed)" if payload["n_unparsed"] else "")


@register("phase", "real_world_eval")
def run_real_world_eval(ctx: Context, state: Optional[RunState] = None) -> None:
    """DrEureka stage 3 -- a DELIBERATE no-op. Nothing here runs on hardware.

    This phase exists so that the sim-to-real methods are honest configs rather
    than truncated ones: DrEureka's deliverable is a set of DR configurations
    that stage 2 explicitly refuses to rank, precisely because the ranking can
    only be settled on a real robot. Dropping the phase would make the config
    look like it produced a winner in simulation; faking a number here would be
    worse.

    So it writes a manifest of what a real deployment would need, records that
    no evaluation was run, and returns. `status` stays `"not_run"` and
    `results` stays null in every artifact this produces.
    """
    configs = ctx.counters.get("dr_configs") or []
    policy_ref = None
    if state is not None and state.best is not None:
        policy_ref = state.best.result.policy_ref

    manifest = {
        "status": "not_run",
        "results": None,
        "why": "BIRD runs in simulation only; this phase is a documented stub.",
        "would_deploy": {
            "n_dr_configs": len(configs),
            "dr_configs_are_unranked": True,
            "reward_cand_id": state.best.cand_id
            if state is not None and state.best else None,
            "policy_ref": policy_ref,
        },
        "would_require": [
            "the physical robot and its safety envelope",
            f"one hardware trial per DR config ({len(configs)} policies to deploy) -- "
            "stage 2 selects nothing, so the real world is the selector",
            "a real-world success metric that does not exist in "
            f"problem.fitness_access={ctx.cfg.get('problem.fitness_access')!r}",
            "a human operator to reset the task between trials",
        ],
    }
    _write_artifact(ctx, "real_world_eval.json", manifest)
    ctx.event("real_world_eval", status="not_run", n_dr_configs=len(configs))
    log.info("    real_world_eval: NOT RUN (documented stub). %d unranked DR config(s) "
             "would go to hardware; no numbers are produced here.", len(configs))
