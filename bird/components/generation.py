"""Stage 1 -- Reward Generation (§1).

Six registry families live here, and between them they are the whole of stage 1:

    sampling_mode      how many prompts, and how they relate to each other
    parent_source      which previous candidate (if any) this one is conditioned on
    env_spec           how much of the environment the model is shown
    history_mode       how much of the conversation survives the iteration boundary
    output_format      the contract the response must satisfy
    generator_backend  what actually produces the raw text (an LLM, or enumeration)

*Generation is nearly identical across every published method* -- the methods
differ in §2/§4. So the value of this module is not
cleverness, it is that every published prompt difference is a leaf key rather
than a branch on a method name. Everything below reads `generate.*` and nothing
reads `cfg["name"]`.

**Prompt section order.** Sections are assembled in one stable order
(`_SECTION_ORDER`), with the task statement pinned first and the output
contract pinned last. `generate.context.shuffle_sections` (LIMEN, randomised
section order for decode diversity) shuffles only the *middle* block via
`ctx.rng`: the two anchors stay put because they are structural (the model must
know what it is doing before it reads evidence, and the format instruction is
most reliable adjacent to the generation).

**Environment adapter duck-type.** `ctx.env` is written by another component
module, so every probe here is optional and degrades to a generated stub. In
probe order, an env may supply:

    env.describe(spec_name) -> str    single hook that answers any env_spec
    env.full_source / .source / .source_code
    env.state_action_api_stub / .api_stub / .state_action_api
    env.pythonic_class_abstraction / .class_abstraction / .pythonic_api
    env.natural_language_only / .nl_description / .description
    env.observation_fields / .action_fields   (list[str], used to build stubs)
    env.symbol_mapping                (dict; `postprocess.symbol_mapping: per_task`)
    env.reward_source                 (removed when strip_existing_reward)
    env.default_dr_ranges / .rapp_bounds       (co_design.dr_prior)
    env.task_images                   (instruction_modality: text+image)

**Per-call history (‡).** `generate.history_mode` answers "how much of the
conversation survives" once, for the whole run, which is one answer too few for
any method that makes more than one LLM call per candidate. L2R makes two --
a Thinker that writes the motion description and a Coder that turns it into
code -- and its released code, on a point the paper is silent about, pins one
flag per LLM rather than one per run:
`platforms/barkour/prompts/prompt_thinker_coder.py:160` sets
`keep_message_history = [True, False]`, and `conversation.py:84` honours it by
appending the turn to a queue only for the LLM whose flag is True. The Thinker
keeps the dialogue; the Coder is re-prompted fresh every turn, which one
run-wide history mode cannot express. `generate.stage_history_modes` closes
that gap:
history_mode names in LLM-call order, `[]` meaning "every call uses
generate.history_mode", and L2R therefore reads `[full_dialogue, none]` ‡.
Which mode each call actually resolved to is written to the journal and onto
the candidate, because a pin that leaves no trace in the run record is a pin
nobody can check.

**LLM budget.** The client contract says clients record their own usage, while
the stage contract says every call must be charged. `_call_llm` reconciles the
two by snapshotting `ctx.budget.llm_calls` and only recording when the client
did not -- so a call is charged exactly once either way.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..config import reward_language
from ..context import Context
from ..observability import render_watchdog
from ..parsing import extract_code
from ..registry import register
from ..state import RunState
from ..types import Candidate, CandidateReport

log = logging.getLogger("bird")

# --------------------------------------------------------------------------
# Module constants. These are runtime guards and rendering caps, deliberately
# NOT config keys: a knob nobody in the literature turns is noise in the space
# (LIMEN's release does expose them as PromptConfig, 3 top / 2 failures,
# hard_rule_chain.yaml:73-75; BIRD renders 5/5).
# --------------------------------------------------------------------------

#: `parse.max_retries: inf` (Text2Reward loops until a block appears) still has
#: to terminate on a model that never emits one. This is the runaway guard.
_INF_RETRY_CEILING = 100
_MAX_FAILURE_TRACES = 5  # LIMEN shows "recent" failures; recency, not a corpus
_MAX_ARCHIVE_ELITES = 5  # LIMEN shows "top-performing" cells
_MAX_FEWSHOT = 8
_CHARS_PER_TOKEN = 4  # only used for the budget fallback and history trimming


# ==========================================================================
# Small shared helpers
# ==========================================================================


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _messages_tokens(messages: Sequence[Dict[str, str]]) -> int:
    return sum(_approx_tokens(str(m.get("content", ""))) for m in messages)


def _env_probe(env: Any, names: Sequence[str], spec: str = "") -> Optional[str]:
    """Ask the env adapter for a piece of text, tolerating any of its spellings.

    `env.describe(spec)` is the preferred hook; the attribute names are the
    fallback so an env written before this module still works.
    """
    if env is None:
        return None
    describe = getattr(env, "describe", None)
    if spec and callable(describe):
        try:
            got = describe(spec)
        except Exception:  # an env that does not know this spec is not an error
            got = None
        if isinstance(got, str) and got.strip():
            return got
    for name in names:
        value = getattr(env, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, str) and value.strip():
            return value
    return None


def _env_fields(env: Any, *names: str) -> List[str]:
    for name in names:
        value = getattr(env, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, (list, tuple)) and value:
            return [str(v) for v in value]
    return []


# --- leakage control (`generate.context.strip_existing_reward`) ------------

#: The EnvAdapter methods that ARE the ground truth. `reference_reward` is the
#: hand-written reward the candidate is compared against; `task_metric` and
#: `success` are the fitness the candidate is *scored* by. All three must be
#: withheld, and the second pair is the one that is easy to forget: a name
#: match on "reward" catches `reference_reward` and sails straight past
#: `def task_metric`. Leaking the metric is worse than leaking the reward --
#: a candidate that returns the success check verbatim does not merely
#: imitate a good reward, it collapses the whole experiment into a tautology.
#: Named from the adapter contract in bird/envs/toy.py, not from any method.
_GROUND_TRUTH_METHODS = ("task_metric", "success", "reference_reward")

#: Name STEMS, matched as `def \w*<stem>\w*(`: a method with one of these words
#: ANYWHERE in its name is withheld, whatever surrounds it (`compute_reward`,
#: `_gripper_caging_reward`, `reference_reward`). `success` is a stem too: a
#: whole-name match would withhold `def success(` and let `def _success_flag(`
#: through -- the per-step ground-truth predicate `step()` writes to
#: `info["success"]`, and on every HumanoidBench class the metric's own
#: per-step term (`s[78] > s[2] and s[2] >= _POWERLIFT_STANDING_Z` on
#: powerlift) -- verbatim under `env_spec: full_source`, the default. A
#: per-step success method is how an adapter spells its ground truth whenever
#: the episode check is a reduction over steps, so a stripper keyed on names
#: has to match the stem. Over-matching is cheap here -- a `describe_success`
#: override would lose its prose from the source render and nothing else -- and
#: under-matching is the tautology described above. `expert`:
#: `EnvAdapter.expert_policy()` is where an adapter ships its analytic solution
#: for `bird.demos` (the demo screen, the demo ceiling), and a solution policy
#: in the generator's prompt is the same leak as the metric. The convention is
#: to bind it OUTSIDE the class body (`bird/envs/toy.py`, foot of the file) so
#: it is never in the render at all; this stem is the second line, and
#: `tests/test_expert_leak.py` holds both.
_GROUND_TRUTH_STEMS = ("reward", "success", "expert")

_REWARD_DEF = re.compile(
    r"^(\s*)def\s+(?:"
    + "|".join(rf"\w*{stem}\w*" for stem in _GROUND_TRUTH_STEMS)
    + "|" + "|".join(_GROUND_TRUTH_METHODS)
    + r")\s*\(",
    re.IGNORECASE)
_WITHHELD = "# [reference reward withheld -- generate.context.strip_existing_reward]"


def _strip_reward(ctx: Context, text: str) -> str:
    """Remove the reference reward AND the ground-truth metric from env text.

    Best-effort by construction: the authoritative gate is the env adapter not
    handing over `reward_source` in the first place. This is the second line of
    defence, because a leaked ground-truth reward invalidates every number the
    run produces -- the LLM must not see the reward it is benchmarked against.

    `_GROUND_TRUTH_METHODS` widens that from the reward to the *fitness*. On an
    env whose success check is hand-written rather than supplied by the
    simulator -- `pendulum`, which has no terminal condition of its own -- the
    metric is the more dangerous leak of the two, and `full_source` renders the
    whole adapter class including it.

    What is cut, exactly. Every `def` whose name contains `reward` or `success`
    (`_GROUND_TRUTH_STEMS`) and every `def` named in `_GROUND_TRUTH_METHODS`,
    each with its whole suite -- signature by paren depth, body by indent --
    replaced by one `_WITHHELD` marker; plus the env's `reward_source` span
    verbatim, when it offers one. Nothing else: a class-body comment, a class
    attribute or a helper with a neutral name survives, which is why adapters
    keep scene geometry at module scope (`humanoid_hand.py`) and measured
    tables in the module docstring (`mujoco_control.py`). `success` is a stem,
    as `reward` is, so `def _success_flag(` -- the per-step predicate `step()`
    writes to `info["success"]`, and on the HumanoidBench tier the metric's own
    per-step term -- cannot survive the strip and reach the generator under the
    default render; `tests/test_strip_success_flag.py` holds it there over every
    adapter class.
    """
    if not text:
        return text
    reference = _env_probe(ctx.env, ("reward_source", "reward_code", "reference_reward_code"))
    if reference and reference.strip() and reference.strip() in text:
        text = text.replace(reference.strip(), _WITHHELD)

    lines = text.splitlines()
    out: List[str] = []
    i, dropped = 0, 0
    while i < len(lines):
        match = _REWARD_DEF.match(lines[i])
        if match is None:
            out.append(lines[i])
            i += 1
            continue
        indent = len(match.group(1))
        # STEP 1: consume the SIGNATURE, by paren depth rather than by indent.
        #
        # A multi-line signature closes on a line indented to the SAME level as
        # the `def`, e.g.
        #
        #     def compute_reward(
        #         self,
        #         obs,
        #     ) -> tuple[float, ...]:      <-- indent == the def's
        #
        # so an indent-only scan breaks HERE and emits the entire body. On
        # `mt10_window-open-v3` that writes the withheld marker and then
        # `TARGET_RADIUS = 0.05` and `reward_utils.hamacher_product` verbatim
        # after it -- a false assurance, which is worse than no stripping at
        # all, because the marker is what a reader checks for. Pendulum and
        # Acrobot write single-line signatures, so they never exercise this.
        depth = 0
        while i < len(lines):
            depth += lines[i].count("(") - lines[i].count(")")
            i += 1
            if depth <= 0:
                break
        # STEP 2: now consume the suite: everything more-indented than the `def`.
        while i < len(lines):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent:
                break
            i += 1
        dropped += 1
        out.append(" " * indent + _WITHHELD)
    if dropped:
        log.debug("strip_existing_reward: removed %d ground-truth definition(s)", dropped)
    return "\n".join(out)


# ==========================================================================
# generator_backend  --  backend(ctx, state, prompt_messages, n) -> list[str]
# ==========================================================================


def _call_llm(ctx: Context, client: Any, messages: List[Dict[str, str]], n: int,
              temperature: Optional[float], tag: str,
              images: Optional[List[Any]] = None) -> List[str]:
    """One call to an LLM client, charged exactly once.

    The client records its own budget usage (llm client contract); we detect
    that and only record ourselves when it did not, so neither a client that
    accounts nor one that does not can leave a call uncounted or double-counted.
    """
    if client is None:
        raise RuntimeError(
            "generate: no LLM client on ctx.generator; "
            "set llm.generator.provider to a registered client (e.g. 'mock')")
    before = ctx.budget.llm_calls
    kwargs: Dict[str, Any] = {"n": n, "temperature": temperature, "tag": tag}
    if images:
        kwargs["images"] = images
    out = client(messages, **kwargs)
    if isinstance(out, str):  # a client that forgot it returns a list
        out = [out]
    out = [str(x) for x in (out or [])]
    if ctx.budget.llm_calls == before:
        ctx.budget.record_llm(
            prompt_tokens=_messages_tokens(messages),
            completion_tokens=sum(_approx_tokens(o) for o in out),
            vlm=bool(images))
    return out


def _images_for(ctx: Context) -> Optional[List[Any]]:
    """`problem.instruction_modality` decides whether the env's task images ride
    along. Its caller is the DECOMPOSE phase (`phases.run_decompose`), per RDA's
    App. 7.1, whose user prompt includes '## Image {image}' -- and only that
    phase: App. 7.2 (reward generation) and 7.5 (reward reflection, which in
    BIRD is the next iteration's generate call) take no image, so `backend_llm`
    below attaches nothing. Per-config, not per-method: the gate is the
    modality key.

    `task_images()` RENDERS, and on `rda` with `--profile` on a Meta-World env
    this is the first render of the run: gymnasium builds `MujocoRenderer`
    lazily, so GL context creation (`envs/metaworld.py::render`, 345 ms when it
    works) happens here in stage 1, not in `record_rollouts`. The `except`
    below catches a raise and cannot catch a hang -- which is how a wedged
    renderer fails -- so `output.video.timeout_s`'s stack-dump watchdog is
    armed around the call. Its raw value, not `+ _WATCHDOG_GRACE_S`: the grace
    pays for a between-frames deadline that has already had its chance to give
    up, and there is no such deadline for one frame.
    """
    if ctx.cfg.get("problem.instruction_modality") not in ("text+image", "video"):
        return None
    images = getattr(ctx.env, "task_images", None)
    if callable(images):
        with render_watchdog(ctx.cfg.get("output.video.timeout_s")):
            try:
                images = images()
            except Exception:
                images = None
    return list(images) if images else None


@register("generator_backend", "llm")
def backend_llm(ctx: Context, state: RunState, prompt_messages: List[Dict[str, str]],
                n: int) -> List[str]:
    """Sample `n` raw completions from `ctx.generator` (every LLM method, §1).

    NO images are attached, whatever `problem.instruction_modality` says: RDA's
    reward-generation and reward-reflection prompts (App. 7.2 / 7.5) take no
    image -- the env image belongs to the decompose phase alone (App. 7.1), and
    attaching it here would hand every generate call pixels the published
    method never sends.

    Loops until it has `n` strings because clients differ in whether they honour
    `n`; a call that returns nothing breaks the loop rather than spinning.
    """
    temperature = ctx.cfg.get("llm.generator.temperature")
    out: List[str] = []
    guard = 0
    while len(out) < n and guard < n + 2:
        guard += 1
        batch = _call_llm(ctx, ctx.generator, prompt_messages, n - len(out),
                          temperature, tag="generate")
        if not batch:
            break
        out.extend(batch)
    return out[:n]


backend_llm.needs_prompt = True  # type: ignore[attr-defined]


# --- exhaustive enumeration (Singh et al. 2009, the pre-LLM limit case) ----


def _grid_values(spec: Any) -> List[float]:
    """`values` may be a list, a linspace dict, or a `linspace(a, b, k)` string."""
    if isinstance(spec, (list, tuple)):
        return [float(v) for v in spec]
    if isinstance(spec, dict):
        start = float(spec.get("start", spec.get("min", 0.0)))
        stop = float(spec.get("stop", spec.get("max", 1.0)))
        num = int(spec.get("num", spec.get("n", 2)))
        if num <= 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + step * i for i in range(num)]
    if isinstance(spec, str):
        m = re.match(r"\s*linspace\s*\(([^)]*)\)\s*$", spec)
        if m:
            parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
            if len(parts) >= 3:
                return _grid_values({"start": float(parts[0]), "stop": float(parts[1]),
                                     "num": int(float(parts[2]))})
        raise ValueError(f"generate.search_grid: cannot read values spec {spec!r}")
    raise ValueError(f"generate.search_grid: values must be a list, dict or "
                     f"linspace string, got {type(spec).__name__}")


def _enumerate_grid(cfg: Any) -> Tuple[List[str], List[Tuple[float, ...]]]:
    """Full cartesian product of the declared grid, in a stable order."""
    grid = cfg.get("generate.search_grid") or {}
    features = [str(f) for f in (grid.get("state_features") or [])]
    if not features:
        raise ValueError(
            "generator_backend: exhaustive_enumeration requires "
            "generate.search_grid.state_features (Singh 2009 enumerates a tabular "
            "reward space; with no features there is no space to enumerate)")
    raw_values = grid.get("values", [0.0, 1.0])
    if isinstance(raw_values, dict) and set(raw_values) >= set(features):
        per_feature = [_grid_values(raw_values[f]) for f in features]
    else:
        shared = _grid_values(raw_values)
        per_feature = [list(shared) for _ in features]

    points: List[Tuple[float, ...]] = [()]
    for column in per_feature:
        points = [prefix + (v,) for prefix in points for v in column]
    return features, points


_TABULAR_TEMPLATE = '''# Singh et al. (2009), exhaustive enumeration: grid point {index}/{total}.
# A tabular reward -- one scalar per state feature. No LLM was involved.
TABLE = {{
{rows}}}


def _feature(state, name, index):
    """Read one named feature out of whatever shape the env hands us."""
    if isinstance(state, dict):
        return state.get(name, 0.0)
    value = getattr(state, name, None)
    if value is not None:
        return value
    try:
        return state[index]
    except Exception:
        return 0.0


def compute_reward(state, action=None, next_state=None):
    components = {{}}
    for i, (name, weight) in enumerate(TABLE.items()):
        components[name] = float(weight) * float(_feature(state, name, i))
    total = float(sum(components.values()))
    return total, components
'''


def _tabular_program(features: Sequence[str], values: Sequence[float],
                     index: int, total: int) -> str:
    rows = "".join(f"    {name!r}: {float(v)!r},\n" for name, v in zip(features, values))
    body = _TABULAR_TEMPLATE.format(index=index + 1, total=total, rows=rows)
    # Fenced, so the configured `parse.patterns` extract it exactly as they
    # would extract an LLM's answer -- the parse path stays single.
    return "```python\n" + body + "```"


@register("generator_backend", "exhaustive_enumeration")
def backend_exhaustive_enumeration(ctx: Context, state: RunState,
                                   prompt_messages: List[Dict[str, str]],
                                   n: int) -> List[str]:
    """Singh et al. (2009): enumerate the tabular reward space, no LLM (§1).

    This is the schema's sanity check that the core loop has not quietly assumed
    an LLM anywhere: the same six stages must run when stage 1 is a for-loop over
    a grid. `prompt_messages` is ignored by construction (see `needs_prompt`),
    nothing is charged to the LLM budget, and the emitted count is capped at
    `generate.n_candidates`.

    The sampler hands this backend the full remainder in one call (chunking
    lives in `LLMClient.__call__`, which this backend never enters), so the
    `ctx.counters` cursor advances once per iteration; it is kept so a backend
    asked twice within one iteration still emits distinct points instead of
    re-emitting the first ones. When the grid runs out it returns fewer than `n` (and then
    nothing), which is the honest answer: a 4-point grid asked for 16 candidates
    has 4 candidates, not 4 candidates and 12 duplicates. The cursor is keyed by
    (restart, iteration) so each iteration re-enumerates from the start.
    """
    features, points = _enumerate_grid(ctx.cfg)
    cap = int(ctx.cfg["generate.n_candidates"])
    total = min(len(points), cap) if cap > 0 else len(points)

    key = f"enum_cursor:{state.restart}:{state.iteration}"
    cursor = int(ctx.counters.get(key, 0))
    take = max(0, min(int(n), total - cursor))
    out = [_tabular_program(features, points[cursor + i], cursor + i, total)
           for i in range(take)]
    ctx.counters[key] = cursor + take
    return out


backend_exhaustive_enumeration.needs_prompt = False  # type: ignore[attr-defined]


# ==========================================================================
# candidate_schedule  --  schedule_fn(ctx, state) -> int
# ---------------------------------------------------------------------------
# How many candidates THIS iteration samples. The only key in the config space
# whose value is a function of the iteration index, so it is the one place a
# "count" read out of a resolved config can disagree with what a run executed.
# Anything that SIZES a run (a CPU allocation, worker resolution) must use
# `generate.n_candidates`, which stays the maximum.


@register("candidate_schedule", "constant")
def schedule_constant(ctx: Context, state: RunState) -> int:
    """`generate.n_candidates` every iteration -- every published method."""
    return max(1, int(ctx.cfg["generate.n_candidates"]))


@register("candidate_schedule", "explicit")
def schedule_explicit(ctx: Context, state: RunState) -> int:
    """One count per iteration, verbatim from `candidate_schedule_values`.

    Verbatim rather than derived from a taper rate: the schedule is part of the
    claim being made, so it belongs in the config where it is citable and
    diffable, not recomputed from two other keys. `_check_coherence` pins the
    length at >= `loop.n_iterations` and every entry > 0; the clamp below is a
    backstop for a restart that runs past the list, never the normal path.
    """
    vals = list(ctx.cfg["generate.candidate_schedule_values"] or [])
    if not vals:
        return max(1, int(ctx.cfg["generate.n_candidates"]))
    return max(1, int(vals[min(int(state.iteration), len(vals) - 1)]))


def _n_this_iteration(ctx: Context, state: RunState) -> int:
    """The candidate count for the current iteration (§1). THE TOTAL, including
    any share a non-LLM operator will contribute -- see `_n_llm_this_iteration`,
    which is what a sampler wants."""
    return _get("candidate_schedule", ctx.cfg["generate.candidate_schedule"])(ctx, state)


def active_waves(ctx: Context, state: RunState) -> List[Dict[str, Any]]:
    """The `loop.waves` passes that run in THIS iteration, in order.

    `[]` in the config means one pass whose counts come from
    `generate.n_candidates` and `generate.crossover.n`, which is what every
    method but R* does. It is returned here as a single synthesised entry so
    the loop has exactly one code path -- a `if waves:` fork in
    `run_iteration` would be a second sequencing rule that only R* exercises,
    and a path only one method exercises is the one that breaks unnoticed.

    `when: no_archive` is R*'s own stated condition -- "Since there is no
    archive during the first iteration, crossover cannot be applied in the
    first iteration" (App. A, p.12) -- evaluated against the archive AS THE
    ITERATION BEGAN, never mid-iteration: wave 1 fills the archive, so a
    predicate re-read later would drop the very pass it is there to enable.
    `state.wave_bootstrap` is set once by the loop, before wave 0 runs.
    """
    spec = list(ctx.cfg["loop.waves"] or [])
    if not spec:
        return [{"llm": None, "crossover": None, "when": "always"}]
    boot = bool(getattr(state, "wave_bootstrap", False))
    out: List[Dict[str, Any]] = []
    for w in spec:
        when = str(w.get("when", "always"))
        if when == "no_archive" and not boot:
            continue
        out.append({"llm": int(w.get("llm", 0) or 0),
                    "crossover": int(w.get("crossover", 0) or 0),
                    "when": when})
    return out or [{"llm": 0, "crossover": 0, "when": "always"}]


def _this_wave(ctx: Context, state: RunState) -> Dict[str, Any]:
    waves = active_waves(ctx, state)
    idx = int(getattr(state, "wave", 0) or 0)
    return waves[idx] if 0 <= idx < len(waves) else waves[-1]


def _n_llm_this_iteration(ctx: Context, state: RunState) -> int:
    """How many candidates the SAMPLER must produce this iteration.

    `generate.crossover.n` of the population is built by an operator that issues
    no LLM call (R* App. A, p.12), so the sampler is asked for the remainder.
    Every sampler goes through here rather than through `_n_this_iteration`, so
    the split holds for any sampling mode without one of them knowing about it.

    Where `loop.waves` is configured the wave's `llm` count is authoritative
    and R*'s two-step first iteration (evaluate the LLM individuals, then cross
    them over and evaluate those, then aggregate -- App. A, p.12) runs as its
    `no_archive` pass (`active_waves`). The archive-empty fallback below -- ask
    the LLM for the full population -- applies only to a config with
    `generate.crossover.n > 0` and no waves, which no published config has.
    """
    wave = _this_wave(ctx, state)
    if wave["llm"] is not None:
        # Waves are AUTHORITATIVE where they are configured: the pass says how
        # many the sampler owes, and the archive-empty fallback below is exactly
        # the case waves exist to handle properly instead.
        return int(wave["llm"])
    total = _n_this_iteration(ctx, state)
    share = int(ctx.cfg.get("generate.crossover.n", 0) or 0)
    if share <= 0 or ctx.cfg["generate.crossover.operator"] == "none":
        return total
    from .alignment import _archive_reports  # local: registry import order
    if len(_archive_reports(state)) < 2:
        return total
    return max(1, total - share)


# parent_source  --  parent_fn(ctx, state) -> list[CandidateReport]
# ==========================================================================


def _n_parents(ctx: Context) -> int:
    return max(1, int(ctx.cfg.get("generate.n_parents", 1) or 1))


@register("parent_source", "none")
def parent_none(ctx: Context, state: RunState) -> List[CandidateReport]:
    """No parent: every candidate is generated from the task alone (zeroshot,
    Text2Reward, L2R, and iteration 0 of everything else). §1."""
    return []


@register("parent_source", "global_best")
def parent_global_best(ctx: Context, state: RunState) -> List[CandidateReport]:
    """The incumbent across all iterations. Selected by LaRes and by ROSKA.

    `configs/methods/lares.yaml` takes it (`generate.parent_source`), and so does
    `configs/methods/roska.yaml`, for a different reason worth keeping apart from LaRes's:
    it is ROSKA's DYNAMIC POPULATION, its whole first contribution. The prompt's
    `R_best^{m-1}` is the best-so-far reward, which advances only when a
    candidate beats it -- "Only those combinations that outshine the previous
    round's top performer survive to the next round" (methodology.tex:108). So
    this family carries two published selectors that mean different things: a
    population's incumbent parent, and a sieve.

    Eureka reads like the obvious selector and is NOT one: it keeps a global
    best for what the run RETURNS (`max_success_overall`, eureka.py L292-300)
    and parents the next round on `best_sample_idx` (L330-335), this round's
    winner, ungated (`select.scope: cumulative` would collapse the two the same
    way -- see the note in `configs/methods/eureka.yaml`). The two bests are a real
    distinction in the paper, and this family is the one that collapses them."""
    return [state.best] if state.best is not None else []


@register("parent_source", "iteration_best")
def parent_iteration_best(ctx: Context, state: RunState) -> List[CandidateReport]:
    """The winner of the previous iteration only -- a genuinely different search
    from `global_best` when a round regresses, since every published parent
    adoption is UNGATED (§6). Empty before the first winner exists."""
    return [state.iteration_best] if state.iteration_best is not None else []


@register("parent_source", "latest")
def parent_latest(ctx: Context, state: RunState) -> List[CandidateReport]:
    """CARD's chain head: the newest reward parents the next one *even when it
    failed the screen* (§1, §6). That is the whole point of the chain -- the
    transcript, not the fitness, is what carries forward."""
    return [state.latest] if state.latest is not None else []


@register("parent_source", "archive_sample")
def parent_archive_sample(ctx: Context, state: RunState) -> List[CandidateReport]:
    """LIMEN: draw parents from the MAP-Elites archive maintained in §6.

    One draw per `generate.n_parents`, each through
    `update.sample_archive_parent` -- THE sampler, one copy. (A private branch
    roulette and fitness wheel here would drift from §6's -- accepting unknown
    branch names silently where §6 raises ConfigError, say, or pools that
    ignore `update.archive.archive_size`.) The branch weights are
    `update.archive.parent_sampling`; the GLOBAL branches draw from the
    archive_size-capped top-fitness pool (`update.exploitation_pool`, the
    release's `self.archive` list) and the ISLAND branches from everything the
    chosen island holds -- grid elites AND cell losers filed by
    `update.loser.action: store_as_negative_example`.

    DAGGER: the paper's text says 70% global *fitness-proportional* / 30%
    island-uniform, while the released code is 0.7 uniform-over-elites / 0.2
    island-uniform / 0.1 island-fitness-weighted -- the branch *definitions*
    differ, not just the numbers. Both readings are
    expressible; the config picks one instead of this file picking for it.
    """
    from .update import sample_archive_parent  # no cycle: update imports no generation
    out: List[CandidateReport] = []
    for _ in range(_n_parents(ctx)):
        report = sample_archive_parent(ctx, state)
        if report is None:
            break
        out.append(report)
    return out


@register("parent_source", "archive_island_cohort")
def parent_archive_island_cohort(ctx: Context, state: RunState,
                                 n: Optional[int] = None) -> List[CandidateReport]:
    """REvolve: every parent of one offspring out of ONE deme (§1, §6).

    The difference from `archive_sample` is the difference between LIMEN and
    REvolve, and it is one draw against `generate.n_parents` of them.
    `archive_sample` loops independent calls into `update.sample_archive_parent`,
    each of which re-spins the branch wheel and, on an island branch, re-picks
    the deme -- so with `n_parents: 2` a crossover routinely recombines island 0
    with island 2, which makes the islands one panmictic population and deletes
    the diversity claim the topology exists for. REvolve is unambiguous the
    other way: paper Alg. 1 line 11 (p.4) samples one island `P` and indexes
    `D[P]` for both operators, and `refs/code/Revolve/rewards_database.py:251-266`
    is "STEP 1: sample an island" once, then `np.random.choice(..., size=
    num_in_context_samples, replace=False)` within it.

    The whole draw is `update.sample_archive_parents` -- ONE copy, sharing
    `_draw_branch` with the single-parent sampler, because a second copy of the
    branch table in this file would drift from §6's. `n` is how
    `sampling_mode: evolutionary_operators` asks for a crossover's two parents
    rather than a mutation's one; every other caller uses the
    `generate.n_parents` default and never passes it.
    """
    from .update import sample_archive_parents  # no cycle: update imports no generation
    return sample_archive_parents(ctx, state, _n_parents(ctx) if n is None else int(n))


@register("parent_source", "top_k")
def parent_top_k(ctx: Context, state: RunState) -> List[CandidateReport]:
    """The top `generate.n_parents` seen so far, selected by no config.

    The crossover / elitist-population hook; being unselected is precisely why
    it exists as a config value. That is a claim about THIS key and not about
    either mechanism it hooks: LaRes takes `update.topology: elitist_population`
    (`configs/methods/lares.yaml`), and REvolve, R* and RF-Agent each publish a
    crossover (`generate.crossover_rate` 0.5,
    `generate.crossover.operator: module_insert`, and `crossover_elite` in
    `generate.actions`). All four reach their parents by another route --
    LaRes `global_best`, REvolve `archive_island_cohort`, R* a
    `softmax_fitness` draw over the archive, RF-Agent `uct_leaf` -- so what
    stays unpublished is taking the top k of everything seen, the naive thing
    an evolutionary loop is assumed to do and nobody does."""
    pool = [r for r in state.all_reports
            if r.candidate.valid and r.fitness is not None]
    pool.sort(key=lambda r: (-(r.fitness or 0.0), r.cand_id))
    return pool[:_n_parents(ctx)]


# ==========================================================================
# env_spec  --  env_spec_fn(ctx, state) -> str
# ==========================================================================


def _flat_row_interface(ctx: Context) -> str:
    """How the candidate RECEIVES the state: a flat array, indexed, no attributes.

    WHY THIS EXISTS. Under `generate.context.env_spec: full_source` a
    candidate is shown the adapter's Python source, which is full of named
    attribute access on the adapter's OWN objects, and nothing about what a
    reward receives: `EnvAdapter._render_full_source` is the ONE rendering of
    the four that never touches `_state_fields` (the other three already emit
    `s[i]` beside every name). Candidates then imitate what they were shown and
    write `state.tool_pos`-style accesses (or `state.dist`), which fail
    `execution_smoke` on the traced tier with
    `AttributeError: BatchTracer has no attribute tool_pos`, whatever the
    method, on every configuration that renders `full_source`.

    NEITHER `tool_pos` NOR `dist` IS A SPEC NAME, which is the other half of
    it: the row calls them `tool_x/y/z` and `dist_tool_target`. A rule alone
    ("index it") would leave the model guessing the names; the map is what
    stops that, so the table is rendered in full and never abbreviated.

    SCOPED TO `state` AND `next_state`, AND SILENT ABOUT `self`. The flat-array
    contract is universal -- `EnvAdapter.sample_transitions` is typed
    `List[Tuple[np.ndarray, np.ndarray, np.ndarray]]`, `call_reward` passes
    `tr.state` through, and the CPU trainer calls `fn(np.asarray(s))` -- so
    this is true on every tier and is emitted on every tier. But `self` is a
    different object: `call_reward` binds a `_SelfProxy`, and
    `pythonic_class_abstraction` advertises callable helpers on purpose
    (`spec.helpers_of`). Saying "there are no attributes" without the scope
    would deny T2R and CARD an interface they are deliberately given, and
    would manufacture on those configurations the exact failure this fixes.

    NOT ADDED TO `env_spec: none`, which routes around this seam. That
    rendering withholds env internals BY DESIGN, and a field map there would
    be the failure `describe`'s docstring records: env internals reaching a
    method whose rendering was chosen to exclude them.
    """
    fields = list(getattr(ctx.env, "_state_fields", ()) or ())
    if not fields:
        return ""
    groups = list(getattr(ctx.env, "_state_groups", ()) or [])
    lines = [
        "",
        "# --- the reward interface ---",
        f"# `state` and `next_state` are flat arrays of width {len(fields)}.",
        # THE EXAMPLE USES A REAL NAME FROM THIS ROW, not a placeholder: the
        # failing candidates guessed `state.tool_pos` and `state.dist`, which
        # are not spec names at all, so the correction has to show the actual
        # spelling being indexed rather than a generic `state.foo`.
        f"# Index them: `state[{0}]` is `{fields[0][0]}`, "
        f"never `state.{fields[0][0]}`. They carry no attributes.",
        "#",
        "# index map (name: index):",
    ]
    lines += [f"#   {name}: s[{i}]" for i, (name, _doc) in enumerate(fields)]
    if groups:
        lines += ["#", "# contiguous groups of the same row (name: slice):"]
        lines += [f"#   {name}: s[{a}:{b}]" for name, a, b in groups]
    return "\n".join(lines)


def _spec_or_stub(ctx: Context, spec: str, attrs: Sequence[str], stub: str) -> str:
    text = _env_probe(ctx.env, attrs, spec=spec) or stub
    if ctx.cfg["generate.context.strip_existing_reward"]:
        text = _strip_reward(ctx, text)
    # APPENDED AFTER THE STRIP, deliberately: `_strip_reward` cuts on marker
    # comments in the adapter's source, and a paragraph added before it would
    # be inside the region a `strip_existing_reward` config removes -- and the
    # configs that strip are among those this exists for.
    return (text.strip() + _flat_row_interface(ctx)).strip()


def _field_lines(fields: Sequence[str], annotation: str = "float") -> str:
    return "\n".join(f"    {f}: {annotation}" for f in fields) or "    # (unspecified)"


@register("env_spec", "full_source")
def env_spec_full_source(ctx: Context, state: RunState) -> str:
    """Eureka/DrEureka's `{env}_obs.py` slot (§1). Upstream hands the model the
    task class cut down to `compute_observations()` (eureka.py:41, :58; App. D,
    appendix:202, "just the observation portion"). BIRD does NOT reproduce that
    trim: the text is whatever the adapter's `_render_full_source` returns --
    the whole adapter class by default (envs/base.py); on Meta-World the task
    class, six Sawyer-base helpers and `reward_utils.tolerance` /
    `hamacher_product` (envs/metaworld.py); on gym the gymnasium environment's
    own source plus this adapter's state table (envs/gym_mujoco.py) -- with only
    the `def *reward*` / task_metric / success spans removed by `_strip_reward`
    under `generate.context.strip_existing_reward`. Recorded gap, not keyed.
    GT's objection -- full source rarely
    exists for a real env -- is the reason the other three values exist."""
    fields = _env_fields(ctx.env, "observation_fields", "obs_fields", "state_fields")
    stub = (
        "The environment source was not supplied by the adapter. What is known:\n"
        f"  task: {ctx.cfg['problem.task_description']}\n"
        + ("  observation fields: " + ", ".join(fields) + "\n" if fields else "")
    )
    body = _spec_or_stub(ctx, "full_source", ("full_source", "source", "source_code"), stub)
    return "Environment source (trimmed to what the reward may read):\n\n" + body


@register("env_spec", "state_action_api_stub")
def env_spec_state_action_api_stub(ctx: Context, state: RunState) -> str:
    """GT (§4 fn.2): only the state/action dataclass API, no implementation and
    -- deliberately -- none of the callable helpers the T2R/CARD abstraction
    provides. The narrower spec is the method's claim, not an accident."""
    obs = _env_fields(ctx.env, "observation_fields", "obs_fields", "state_fields")
    act = _env_fields(ctx.env, "action_fields", "act_fields")
    stub = (
        "@dataclass\nclass State:\n" + _field_lines(obs) + "\n\n"
        "@dataclass\nclass Action:\n" + _field_lines(act) + "\n"
    )
    body = _spec_or_stub(ctx, "state_action_api_stub",
                         ("state_action_api_stub", "api_stub", "state_action_api"), stub)
    return ("State/action API (this is the whole interface -- there are no helper "
            "methods and no source):\n\n" + body)


@register("env_spec", "pythonic_class_abstraction")
def env_spec_pythonic_class_abstraction(ctx: Context, state: RunState) -> str:
    """Text2Reward / CARD: typed attributes *plus callable helper methods*
    (`check_grasp`, sdf getters). The helpers are the difference from GT's stub
    and the reason T2R's symbol mapping is needed downstream (§1)."""
    obs = _env_fields(ctx.env, "observation_fields", "obs_fields", "state_fields")
    helpers = _env_fields(ctx.env, "helper_methods", "helpers")
    stub = (
        "class Env:\n" + _field_lines(obs) + "\n\n"
        + ("\n".join(f"    def {h}(self, *args): ..." for h in helpers)
           if helpers else "    # (no helper methods declared by the adapter)")
        + "\n"
    )
    body = _spec_or_stub(ctx, "pythonic_class_abstraction",
                         ("pythonic_class_abstraction", "class_abstraction", "pythonic_api"),
                         stub)
    return ("Environment abstraction -- typed attributes and callable helpers you "
            "may use:\n\n" + body)


@register("env_spec", "t2r_class_abstraction")
def env_spec_t2r_class_abstraction(ctx: Context, state: RunState) -> str:
    """Text2Reward's / CARD's Meta-World class abstraction, byte for byte
    (`MetaworldPrompt.py:14-26`; CARD's copy is character-identical, and
    `card/main.tex:734-746` prints the same block).

    THE ADAPTER IS NOT CONSULTED, and that is the member's entire point. Every
    other `env_spec` value renders whatever the environment offers, so what the
    generator sees grows whenever an adapter's tables grow. This one is a
    CONSTANT: three classes, eight attribute lines, no helper methods, no
    observation table. On Meta-World `pythonic_class_abstraction` renders 39
    documented slots plus `_HELPERS`' eight expressions -- including
    Meta-World's own `tolerance()` and `hamacher_product()`, the two kernels
    the shipped v2 reward is built from -- which is strictly more than the
    published prompt carried. Letting an adapter override here would reintroduce
    exactly that, so it cannot.

    `generate.context.strip_existing_reward` is therefore also a no-op here:
    there is no reward text in a constant to strip. Recorded rather than
    special-cased -- the key stays legal and simply has nothing to do.

    Its companion is `generate.postprocess.symbol_mapping:
    t2r_metaworld_global`, which carries the same seven symbols and is what makes
    a program written against them executable. `_check_coherence` refuses this
    member without a symbol table, because the pair is the method.
    """
    return _T2R_METAWORLD_CLASS_ABSTRACTION


@register("env_spec", "natural_language_only")
def env_spec_natural_language_only(ctx: Context, state: RunState) -> str:
    """No code at all: prose, then each state and action variable as a LABEL
    beside the index it stands for (`- s[0] x_position: <doc>`).

    TWO published sources share this member rather than duplicating it: L2R
    (§1) and REvolve.

    REvolve's release reads ONE environment text --
    `refs/code/Revolve/prompts/env_input`, the verbatim Gymnasium `HumanoidEnv`
    docstring, whose observation section is a `| Num | Observation | Min | Max |`
    INDEX TABLE -- at `main.py:103-106`, before the generation loop, and passes
    it to `RewardFunctionGeneration(system_prompt=..., env_input=...)`.
    `cfg.evolution.baseline` selects the OPERATOR prompt (`mutation` vs
    `mutation_auto`, `main.py:177`) and never the environment text, so every arm
    that paper reports -- REvolve, REvolve Auto, Eureka, Eureka Auto -- saw the
    same index table. That is why its published rewards index `observation[22]`
    (`refs/tex/revolve/main.tex:2459`, `:2492`) instead of reaching for an
    attribute, and it is why `configs/methods/revolve.yaml` selects this member.

    THE PROPERTY THAT MATTERS, and it is measured rather than asserted: this
    member offers the model NO identifier it cannot use. A name appears here
    only as a label beside its index, never in a position that invites
    `state.<name>` or `<name>(s)`. Across the 15 adapter classes
    `tests/test_env_spec_rendered_resolves.py` sweeps, this member
    and `none`
    are the only two that offend on ZERO; `pythonic_class_abstraction` and
    `state_action_api_stub` offend on 15/15 and `full_source` on 10/15, because
    the reward execution namespace binds `{np, numpy, math}` and nothing else.

    ‡ The release's table describes Humanoid-v4's 376-D observation; this
    adapter presents the 47-entry `concat(qpos, qvel)` (`bird/envs/gym_mujoco.py`),
    so the table is REGENERATED against the state a candidate is actually handed
    rather than copied. Rendering the vendored 376-D table verbatim would tell a
    model that `observation[22]` is the forward velocity when on this state
    `s[22]` is a hinge angle -- a silent wrong-index defect, strictly worse than
    a loud one. The form is the release's; the indices are this adapter's.
    """
    stub = str(ctx.cfg["problem.task_description"])
    body = _spec_or_stub(ctx, "natural_language_only",
                         ("natural_language_only", "nl_description", "description"), stub)
    return "Environment (described in words; no code is available):\n\n" + body


@register("env_spec", "none")
def env_spec_none(ctx: Context, state: RunState) -> str:
    """No environment shown. DrEureka's DR stage is the published instance: its
    stage-2 prompt shows NO environment at all -- task text plus RAPP bounds
    only (§1)."""
    return ""


# ==========================================================================
# history_mode  --  history_fn(ctx, state) -> list[dict]
# ==========================================================================


def _clean_turns(turns: Sequence[Any]) -> List[Dict[str, str]]:
    """`state.dialogue` entries may carry bookkeeping keys; the wire wants two."""
    out: List[Dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "user"))
        content = str(turn.get("content", ""))
        if content.strip():
            out.append({"role": role, "content": content})
    return out


def _validated_only(turns: Sequence[Any]) -> List[Any]:
    """CARD §4.3: compile failures never enter the transcript. §6 is what writes
    the dialogue, so all this can do is honour an explicit `valid: False` mark."""
    return [t for t in turns if not (isinstance(t, dict) and t.get("valid") is False)]


def _fit_tokens(turns: List[Dict[str, str]], max_tokens: int) -> List[Dict[str, str]]:
    """Drop from the oldest end until the transcript fits. Truncation is a
    context-limit guard, not a method choice -- `full_dialogue` opts out of it."""
    if max_tokens is None or max_tokens <= 0:
        return turns
    while turns and _messages_tokens(turns) > max_tokens:
        turns = turns[1:]
    return turns


@register("history_mode", "none")
def history_none(ctx: Context, state: RunState) -> List[Dict[str, str]]:
    """No conversation carried at all (LIMEN, zeroshot, Singh). Note this is
    enforced twice over: `loop.carry` without `dialogue` empties the slot at the
    iteration boundary anyway (state.apply_carry)."""
    return []


@register("history_mode", "last_iteration")
def history_last_iteration(ctx: Context, state: RunState) -> List[Dict[str, str]]:
    """Eureka and RDA: Markov-1. Eureka's repo hard-caps the dialogue at 4
    messages and App. D argues *for* forgetting from context limits -- the exact
    opposite of CARD's published rationale for `full_dialogue`. No paper ablates
    the axis; this is the knob that would."""
    turns = _clean_turns(state.dialogue)
    keep = max(1, int(ctx.cfg.get("generate.history_max_turns", 4) or 4))
    return _fit_tokens(turns[-keep:], ctx.cfg.get("generate.history_max_tokens", 0))


@register("history_mode", "cumulative_append")
def history_cumulative_append(ctx: Context, state: RunState) -> List[Dict[str, str]]:
    """GT: the transcript accumulates every round (Alg. 1 line 19,
    neurips_2025.tex:324). App. B (:627-636) accumulates the ENGLISH designs and
    shows the latest code once; what each turn holds is
    `update.prompt.assistant_content` (gt pins `nl_spec`, so the carried turn is
    the stage-1 spec and PARENT REWARD carries the program). Unbounded by turn
    count, bounded only by the token cap."""
    turns = _clean_turns(state.dialogue)
    return _fit_tokens(turns, ctx.cfg.get("generate.history_max_tokens", 0))


@register("history_mode", "full_dialogue")
def history_full_dialogue(ctx: Context, state: RunState) -> List[Dict[str, str]]:
    """CARD: the complete transcript, validated turns only (§4.3) -- its stated
    mechanism for catching negative optimization. Unlike `cumulative_append` it
    does not drop old turns to fit: CARD's transcripts are short by design
    (~14.2k tokens for a whole run vs Eureka's ~663k), so dropping the head
    would silently convert the method into a sliding window."""
    return _clean_turns(_validated_only(state.dialogue))


@register("history_mode", "rolling_summary")
def history_rolling_summary(ctx: Context, state: RunState) -> List[Dict[str, str]]:
    """The untested middle ground between Eureka's window and CARD's transcript
    (§1): keep the last `history_max_turns` turns verbatim and compress
    everything older into one summary turn.

    Published by nobody, so there is no reference implementation to be faithful
    to. Uses `ctx.evaluator` (the cheaper model) when one exists and falls back
    to a deterministic extractive summary otherwise, so the component still
    works with no client at all.
    """
    turns = _clean_turns(state.dialogue)
    keep = max(1, int(ctx.cfg.get("generate.history_max_turns", 4) or 4))
    if len(turns) <= keep:
        return turns
    older, recent = turns[:-keep], turns[-keep:]
    return [{"role": "user", "content": _summarise(ctx, older)}] + recent


def _summarise(ctx: Context, turns: List[Dict[str, str]]) -> str:
    head = "Summary of the earlier conversation:\n"
    if ctx.evaluator is not None:
        ask = [
            {"role": "system", "content": "You compress transcripts. Be terse and factual."},
            {"role": "user", "content":
                "Summarise the reward-design conversation below in at most 10 bullet "
                "points: what was tried, what the measured outcome was, and what was "
                "concluded. Keep every number.\n\n"
                + "\n\n".join(f"[{t['role']}] {t['content']}" for t in turns)},
        ]
        out = _call_llm(ctx, ctx.evaluator, ask, 1,
                        ctx.cfg.get("llm.evaluator.temperature"), tag="history_summary")
        if out and out[0].strip():
            return head + out[0].strip()
    # Extractive fallback: first line of each dropped turn, oldest first.
    bullets = []
    for turn in turns:
        first = turn["content"].strip().splitlines()[0][:200]
        bullets.append(f"  - [{turn['role']}] {first}")
    return head + "\n".join(bullets)


# --- which history_mode governs which call (L2R) --------------------------
#
# Without `generate.stage_history_modes`, one run picks exactly one of the five
# components above and every LLM call gets it. The list makes the choice
# per-call, and the two constants below are what gives an index into it its
# meaning -- the call order of the two-stage sampler -- so they live here, next
# to the family they index into, rather than beside the sampler that walks them.

#: LLM-call order under `generate.output.two_stage_nl_then_code`, matching the
#: order of `keep_message_history` in L2R's `prompt_thinker_coder.py`.
_CALL_THINKER = 0
_CALL_CODER = 1
_CALL_ROLES = ("thinker", "coder")


def _call_role(call_index: int) -> str:
    """A name for the journal. Unnamed indices still get recorded, as `call<i>`,
    because an unlabelled trace beats a missing one."""
    if 0 <= call_index < len(_CALL_ROLES):
        return _CALL_ROLES[call_index]
    return f"call{call_index}"


def _history_mode_for_call(ctx: Context, call_index: Optional[int]) -> str:
    """The `history_mode` that governs LLM call `call_index` of one candidate.

    `call_index=None` means "this candidate is one call", which is every method
    but L2R, and resolves to the global `generate.history_mode`.

    `config._check_coherence` already guarantees the list is either empty or
    exactly the two entries the two-stage sampler consumes, so this does not
    re-validate it. It does raise on an index the list cannot answer: that would
    be a bug in this module rather than in the config, and falling back to the
    global value would silently report L2R's Coder as `full_dialogue` -- a wrong
    answer that the journal would then faithfully agree with.
    """
    stage_modes = list(ctx.cfg.get("generate.stage_history_modes") or [])
    if not stage_modes or call_index is None:
        return str(ctx.cfg["generate.history_mode"])
    if not 0 <= call_index < len(stage_modes):
        raise IndexError(
            f"generate.stage_history_modes has {len(stage_modes)} entries "
            f"({stage_modes}) but generation asked for call index {call_index}: "
            "the list and the sampler disagree about how many LLM calls one "
            "candidate costs")
    return str(stage_modes[call_index])


def _history_messages(ctx: Context, state: RunState,
                      call_index: Optional[int]) -> List[Dict[str, str]]:
    """Resolve the mode for this call, run it, and record which was used.

    The event is emitted only when `stage_history_modes` is actually pinned: in
    the global case the resolved config already says what every call saw, and a
    per-prompt journal line repeating it would be noise. When the modes DO
    differ per call, the config alone no longer tells you which call saw what,
    so the journal has to.
    """
    mode = _history_mode_for_call(ctx, call_index)
    turns = _get("history_mode", mode)(ctx, state)
    if call_index is not None and (ctx.cfg.get("generate.stage_history_modes") or []):
        ctx.event("generate", event="stage_history_mode",
                  iteration=state.iteration, call_index=call_index,
                  call_role=_call_role(call_index), history_mode=mode,
                  n_turns=len(turns))
    return turns


def _history_modes_used(ctx: Context, backend: Callable[..., List[str]]) -> List[str]:
    """The per-call modes this candidate's prompts were built under, in call
    order -- one entry per LLM call the sampler makes, so `["full_dialogue",
    "none"]` on an L2R candidate and `["none"]` on a single-call one.

    Written onto every candidate, not only the per-stage ones, so that reading a
    candidate never requires also resolving the config that produced it.
    """
    if not _two_stage_on(ctx, backend):
        return [_history_mode_for_call(ctx, None)]
    return [_history_mode_for_call(ctx, i) for i in (_CALL_THINKER, _CALL_CODER)]


# ==========================================================================
# output_format  --  output_fmt_fn(ctx) -> str
# ==========================================================================
#
# Each of these carries two attributes read by the assembly code below:
#   .wants_components  -> parse a named-component dict out of the code
#   .wants_weights     -> parse an explicit weights/params dict out of the response
# Reading a registry attribute is how the parser learns what to extract without
# any `if format == ...` -- the same rule the whole repo runs on.

_SIGNATURE = "def compute_reward(state, action=None, next_state=None):"

#: `generate.output.signature`: the signature line the OUTPUT CONTRACT pins, one
#: value per PUBLISHED line, each rendered verbatim from its release.
#:
#: WHY A KEY. BIRD's own line is not a neutral default -- it names the receiver a
#: reward program reads its quantities off. `env_spec: t2r_class_abstraction`
#: shows the model `self.robot.ee_position` and
#: `postprocess.symbol_mapping: t2r_metaworld_global` rewrites exactly those
#: `self.`-prefixed symbols, so a contract that pins a signature WITHOUT `self`
#: asks for a program in a language the mapping cannot read. Measured:
#: gpt-4-turbo, given the abstraction and `compute_reward(state,
#: action=None, next_state=None)`, bridges the two itself --
#:
#:     env = state[0]
#:     handle_position = env.obj1.position
#:
#: -- which obeys every instruction it was given, uses only the six attributes
#: the abstraction lists, matches no `self.`-keyed table entry, and dies in
#: `execution_smoke`, through all ten repair attempts. `_check_coherence`
#: refuses that pairing outright.
#:
#: The published lines are NOT normalised into one. T2R's carries no type
#: annotations, returns `-> float` and has no trailing colon (it is quoted inline
#: in prose); CARD's is a code block with annotations and
#: `Tuple[float, Dict[str, float]]`. Those differ because the two methods differ
#: -- CARD's §4 reads a per-component dict and T2R has no reflection stage at all
#: -- and flattening them would invent a third line neither paper sent. The
#: RETURN SHAPE stays `generate.output.format`'s business
#: (`component_dict_return` vs `scalar_only`); this key carries only the line.
#:
#: `training._ARG_BY_NAME` already binds both: `obs` -> state, `action` ->
#: action, and `self` -> -1 (None) via `_make_binder`. `compute_dense_reward` is
#: already in `_REWARD_NAMES`. Measured: T2R's six released Meta-World programs,
#: mapped with `_SYMBOLS_T2R_METAWORLD`, compile and execute through
#: `CompiledReward` with binder plan `[-1, 1, 0]`
#: (`tests/test_t2r_class_abstraction.py`).
_SIGNATURES: Dict[str, str] = {
    # BIRD's own; every config that does not name another one keeps it.
    "compute_reward_state_action_next": _SIGNATURE,
    # refs/code/text2reward/code_generation/single_flow/classlike_prompt/MetaworldPrompt.py:32
    # (quoted inline in prose, hence no trailing colon -- kept as sent).
    "t2r_compute_dense_reward": "def compute_dense_reward(self, action, obs) -> float",
    # refs/code/CARD/code_generation/self_reflection/benchmark_prompt/metaworld_prompt.py:34
    # (= main.tex:752-761's code block).
    "card_compute_dense_reward": (
        "def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray)"
        " -> Tuple[float, Dict[str, float]]:"),
}

#: What `generate.reward_language: jax` adds under the signature. The three
#: constraints are the three ways a correct numpy reward fails to TRACE, each
#: named with the jnp idiom that replaces it, because a model told only "no
#: Python control flow" writes `float(dist) < 0.1` next. The return shape stays
#: the output format's business: this clause says every returned value is a
#: scalar, the format clause says how they are packaged.
_JAX_CLAUSE = (
    "Write the body in `jax.numpy`, which is in scope as `jnp` (`jax` and `math` are\n"
    "too; `numpy` as `np` is available for constants only). The function is traced\n"
    "with `jax.jit` and `jax.vmap`, so it must be a pure array program: no Python\n"
    "`if`/`while` on values (use `jnp.where`), no `.item()`, `float()` or `bool()`\n"
    "on a traced value, and no in-place mutation (use `.at[...].set(...)`). Every\n"
    "value you return -- the total and each component -- must be a scalar."
)


#: What `generate.reward_language: torch` adds under the signature.
#:
#: PARAPHRASED FROM EUREKA'S OWN PROMPT, not invented, and the two sources are
#: `refs/code/Eureka/eureka/utils/prompts/initial_system.txt` ("Since the
#: reward function will be decorated with @torch.jit.script, please make sure
#: that the code is compatible with TorchScript (e.g., use torch tensor instead
#: of numpy array). Make sure any new tensor or variable you introduce is on
#: the same device as the input tensors.") and `reward_signature.txt` (the
#: `Tuple[torch.Tensor, Dict[str, torch.Tensor]]` return).
#:
#: NOT VERBATIM, and the difference is stated rather than smoothed over.
#: Eureka's sentence arrives inside its SYSTEM message beside a task-specific
#: signature string, and BIRD renders the system message and the signature
#: through different keys (`generate.context.system_prompt`,
#: `generate.output.format`) -- a method that wants Eureka's system message
#: byte-for-byte sets that key and replaces all of it, which is the mechanism
#: that already exists for exactly this. What this clause carries is the part
#: belonging to the LANGUAGE rather than to the method: the batch shape, which
#: Eureka's prompt conveys only implicitly through its signature example and
#: which is the property a candidate silently violates.
#:
#: THE BATCH SENTENCE IS THE LOAD-BEARING ONE. A model told only "use torch"
#: writes a per-transition body that broadcasts into a plausible wrong number
#: on a `(num_envs, *)` input -- the failure
#: `verification._probe_batched_torch` exists to catch -- so the leading axis
#: is said explicitly.
_TORCH_CLAUSE = (
    "Write the body in `torch`, which is in scope (`numpy` as `np` and `math` are\n"
    "too, for constants only). THE INPUTS ARE BATCHED: each argument is a tensor\n"
    "whose FIRST axis is the environment, so `state` has shape `(num_envs, obs_dim)`\n"
    "and your reward must return one value per environment -- shape `(num_envs,)` --\n"
    "for the total and for each component. Index features as `state[:, i]`, never\n"
    "`state[i]`. Use `torch` operations throughout rather than numpy, keep any new\n"
    "tensor on the same device as the inputs (`device=state.device`), and do not\n"
    "call `.item()`, `float()` or `bool()` on a batched value."
)


def _signature_clause(ctx: Context) -> str:
    """The chosen signature line, plus the language contract under `reward_language`.

    `generate.output.signature` selects which published line is pinned; the
    default is BIRD's own. Not a branch on method: one dict lookup, the same
    shape the rest of this module uses.

    Three languages, three contracts, and each is a real difference in what a
    candidate may BE rather than a stylistic preference: `numpy` is one
    transition at a time on the host, `jax` is a traced scalar row, `torch` is
    an eager batched program over the fleet. A model given the wrong one writes
    a program the tier cannot run -- or worse, one it CAN run and that means
    something else, which is the batch case.

    The signature line and the language contract compose rather than compete:
    the line comes from `generate.output.signature`, the contract from
    `reward_language`, and both are rendered.
    """
    sig = f"    {_SIGNATURES[ctx.cfg['generate.output.signature']]}"
    language = reward_language(ctx.cfg)
    if language == "jax":
        return f"{sig}\n\n{_JAX_CLAUSE}"
    if language == "torch":
        return f"{sig}\n\n{_TORCH_CLAUSE}"
    return sig


def _edit_clause(ctx: Context) -> str:
    mode = ctx.cfg["generate.output.edit_mode"]
    if mode == "diff":
        return ("Emit only the changes, as one or more search/replace blocks:\n"
                "    <<<<<<< SEARCH\n    <exact existing lines>\n    =======\n"
                "    <replacement lines>\n    >>>>>>> REPLACE")
    if mode == "weights_only":
        return ("Do not rewrite the code. Change only the weights: emit the new "
                "weights dict and nothing else.")
    return "Emit the complete function, not a patch."


def _helpers_clause(ctx: Context) -> str:
    if ctx.cfg["generate.output.forbid_helper_functions"]:
        # GT: keeps generated code parseable. CARD's analogue is "do not invent
        # any variable or attribute".
        return ("Define no helper functions and no module-level code beyond the "
                "required definitions. Do not invent variables or attributes that "
                "the environment interface above does not declare.")
    return ""


def _tips_clause(ctx: Context) -> str:
    """Eureka appends code_output_tip to EVERY feedback message, error path
    included (eureka.py:218-221, 276). Every prompt here ends in an OUTPUT
    CONTRACT built through `_wrap_contract`, so appending the writing tips as
    its last part reproduces the per-prompt placement."""
    if ctx.cfg["generate.context.include_reward_engineering_tips"]:
        return _TIPS_WRITING
    return ""


def _thought_clause(ctx: Context) -> str:
    """RF-Agent: a one-sentence design idea in braces BEFORE the code, in the
    same LLM turn (`generate.output.design_thought: inline_brace`). It leads the
    OUTPUT CONTRACT because upstream's file states the brace first and the code
    second -- the order the model is asked to produce them in
    (thought_code_output.txt:1-3; lines 4-6 are the return shape the contract
    body already states, and line 7, "Do not give additional explanations.", is
    not carried). `_build_candidate` reads the brace back into
    `Candidate.nl_spec`, the same slot GT's two-stage prose lands in, which is
    why the config validator refuses the two together."""
    if ctx.cfg["generate.output.design_thought"] == "inline_brace":
        return _THOUGHT_CLAUSE
    return ""


def _forbidden_clause(ctx: Context) -> str:
    """Name the identifiers `verify.forbidden_symbols` will reject, when the
    gate is armed. The gate exists to stop a reward reading its own score
    (`verification.static_forbidden_symbols`); a model that is never told the
    list rediscovers the ban one forfeited slot at a time (e.g. a reward
    rejected for a local variable named `success`). Empty when nothing is
    forbidden or the check is not scheduled, so a config without a gate sees
    no such clause."""
    names = [str(n) for n in (ctx.cfg.get("verify.forbidden_symbols") or [])]
    if not names or "forbidden_symbols" not in (ctx.cfg.get("verify.static_checks") or []):
        return ""
    return ("Do not reference these names anywhere in the code -- they belong to the "
            "scoring harness, and a program that reaches for one is rejected before "
            "training: " + ", ".join(f"`{n}`" for n in names) + ".")


def _wrap_contract(ctx: Context, body: str) -> str:
    parts = [_thought_clause(ctx), body, _forbidden_clause(ctx), _edit_clause(ctx),
             _helpers_clause(ctx), _tips_clause(ctx)]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _parse_design_thought(raw: str) -> str:
    """The brace the thought clause asks for, read off the prose BEFORE the first
    code fence. The release searches the whole response (rfagent_algo.py:426),
    so a model that skipped the brace hands it the first dict literal inside the
    program -- `{"reach": ...}` -- as its "design idea", and every later prompt
    and the self-verify judge then reads code as the thought. Stopping at the
    fence keeps the failure visible: no brace, no thought, and
    `meta["design_thought_parsed"]` says so."""
    text = raw or ""
    fence = text.find("```")
    head = text if fence < 0 else text[:fence]
    m = _THOUGHT_RE.search(head)
    return m.group(1).strip() if m else ""


@register("output_format", "component_dict_return")
def output_component_dict_return(ctx: Context) -> str:
    """Eureka / DrEureka / CARD / LIMEN: a free-form body that must return
    `(total, {name: value})` (§1).

    Component *visibility* is a precondition for per-component reflection in §4
    -- the config validator enforces the other direction of that dependency."""
    body = (
        "Write one Python function with exactly this signature:\n\n"
        f"{_signature_clause(ctx)}\n\n"
        "It must return a 2-tuple `(total, components)`: `total` is a float, and\n"
        "`components` is a dict mapping a short name to that term's scalar value\n"
        "for this transition. Every term that contributes to `total` must appear\n"
        "in `components` -- the reflection stage reads that dict and nothing else.\n"
        "Put the function in a single ```python code block."
    )
    return _wrap_contract(ctx, body)


output_component_dict_return.wants_components = True  # type: ignore[attr-defined]
output_component_dict_return.wants_weights = False  # type: ignore[attr-defined]


@register("output_format", "component_dict_plus_weights")
def output_component_dict_plus_weights(ctx: Context) -> str:
    """RDA / GT: the component dict plus an explicit, LLM-written weights dict
    (§1). Separating the weights from the code is what makes
    `edit_mode: weights_only` and per-component credit assignment mechanical
    rather than a re-parse of arbitrary arithmetic."""
    body = (
        "Write one Python function with exactly this signature:\n\n"
        f"{_signature_clause(ctx)}\n\n"
        "It must return a 2-tuple `(total, components)` as described, in a single\n"
        "```python code block.\n\n"
        "Then, in a separate ```json code block, emit a flat dict mapping every\n"
        "component name to the scalar weight your code applies to it, e.g.\n\n"
        '    {"reach": 1.0, "action_penalty": -0.01}\n\n'
        "The weights must be the ones the code actually uses -- they are read\n"
        "back and reported."
    )
    return _wrap_contract(ctx, body)


output_component_dict_plus_weights.wants_components = True  # type: ignore[attr-defined]
output_component_dict_plus_weights.wants_weights = True  # type: ignore[attr-defined]


@register("output_format", "scalar_only")
def output_scalar_only(ctx: Context) -> str:
    """A single float and no decomposition. The degenerate contract, and the one
    that makes `evaluate.feedback.granularity: per_component` incoherent -- the
    config validator rejects that pair rather than silently producing empty
    per-component feedback (§1, config._check_coherence)."""
    body = (
        "Write one Python function with exactly this signature:\n\n"
        f"{_signature_clause(ctx)}\n\n"
        "It must return a single float. Do not return a component breakdown.\n"
        "Put the function in a single ```python code block."
    )
    return _wrap_contract(ctx, body)


output_scalar_only.wants_components = False  # type: ignore[attr-defined]
output_scalar_only.wants_weights = False  # type: ignore[attr-defined]


@register("output_format", "template_params")
def output_template_params(ctx: Context) -> str:
    """L2R: the only `template_dsl` point -- the model fills the parameters of a
    fixed reward API rather than writing code (§1). The parsed parameter dict
    lands on `Candidate.weights`, which is the record's one scalar-dict slot."""
    template = _env_probe(ctx.env, ("reward_template", "template", "reward_api"),
                          spec="reward_template")
    body = (
        "Do not write a reward function. Fill in the fixed reward API below by\n"
        "emitting a single ```json code block: a flat dict mapping each template\n"
        "parameter name to its value. Emit nothing else.\n"
    )
    if template:
        body += "\nReward API template:\n\n" + template.strip() + "\n"
    return _wrap_contract(ctx, body)


output_template_params.wants_components = False  # type: ignore[attr-defined]
output_template_params.wants_weights = True  # type: ignore[attr-defined]


# ==========================================================================
# Prompt assembly
# ==========================================================================

_DEFAULT_SAFETY = (
    "Safety requirement: the behaviour this reward induces must stay within the "
    "platform's safe operating envelope -- no joint-limit violations, no "
    "self-collisions, no torques or velocities that would damage hardware. "
    "Prefer a behaviour that is slower and safe over one that is fast and unsafe."
)

# BIRD's OWN environment-neutral paraphrase of a reward recipe (uncited),
# selected by `include_reward_recipe_hints: true`. NOT a
# published text: it merges the published items 1 and 2 into one progress term,
# adds a sub-goal term and a completion bonus no release lists, and closes the
# open "..." list. It is kept byte-identical because it is the only recipe with
# a referent on a non-manipulator env (pendulum, acrobot, toy_reacher), and
# because a run dir resolved with `true` must keep validating and resuming. The
# published texts are `_RECIPE_T2R_METAWORLD` / `_RECIPE_T2R_PANDA` below.
_RECIPE_HINTS = (
    "A task reward typically consists of these parts (some are optional):\n"
    "  - a progress term on the distance to the goal;\n"
    "  - a term for the sub-goal that must be achieved first (e.g. a grasp);\n"
    "  - a regularisation term on action magnitude or joint velocity;\n"
    "  - a bonus on task completion.\n"
    "Use only the parts this task needs."
)

# The published recipe is keyed by BENCHMARK, not by method: Text2Reward and
# CARD send the same two texts, and neither sends one on MuJoCo locomotion
# (T2R's HopperPrompt.py / AntPrompt.py carry no recipe at all).
#
# Meta-World, three items + open "...". Byte-identical in
#   refs/code/text2reward/code_generation/single_flow/classlike_prompt/MetaworldPrompt.py:8-12
#   refs/code/CARD/code_generation/self_reflection/benchmark_prompt/metaworld_prompt.py:6-10
#   refs/tex/card/main.tex:728-732 (App. C, the Meta-World system prompt)
# Paper and both releases agree character for character -- no marker.
_RECIPE_T2R_METAWORLD = (
    "Typically, the reward function of a manipulation task is consisted of these "
    "following parts (some part is optional, so only include it if really necessary):\n"
    "1. the distance between robot's gripper and our target object\n"
    "2. difference between current state of object and its goal state\n"
    "3. regularization of the robot's action\n"
    "..."
)

# ManiSkill2 / Panda, five items + open "...". Byte-identical in
#   refs/code/text2reward/.../classlike_prompt/PandaPrompt.py:9-15 (also
#   MobilePandaPrompt.py:8, MobileDualArmPrompt.py:9)
#   refs/code/CARD/.../benchmark_prompt/panda_prompt.py:9-15 (also
#   mobile_panda_prompt.py:6-12, mobile_dual_arm_prompt.py:7-13)
# dagger vs refs/tex/text2reward/text/appendix.tex:189-195: the PAPER drops the
# parenthetical hedge on :189 and prints "implied by task instruction" (no
# "the") on :193-194. The release text is what was sent, so it wins.
_RECIPE_T2R_PANDA = (
    "Typically, the reward function of a manipulation task is consisted of these "
    "following parts (some part is optional, so only include it if really necessary):\n"
    "1. the distance between robot's gripper and our target object\n"
    "2. difference between current state of object and its goal state\n"
    "3. regularization of the robot's action\n"
    "4. [optional] extra constraint of the target object, which is often implied by "
    "the task instruction\n"
    "5. [optional] extra constraint of the robot, which is often implied by "
    "the task instruction\n"
    "..."
)

# Text2Reward's Meta-World class abstraction, VERBATIM. The bytes are
#   refs/code/text2reward/code_generation/single_flow/classlike_prompt/MetaworldPrompt.py:14-26
# lifted with no edit of any kind -- three classes, eight attribute lines, NO
# helper methods and NO observation table. CARD's Meta-World prompt is a
# character-identical copy of the same block (its README says the code is based
# on Text2Reward), and refs/tex/card/main.tex:734-746 prints it in App. C
# tab:metaworld_system_prompt; paper and both releases agree character for
# character, so there is no marker on it.
#
# WHY THIS IS A SEPARATE env_spec MEMBER, and not `pythonic_class_abstraction`.
# The two render different amounts of the environment, and the difference is
# not cosmetic -- it is the whole "what did the paper's LLM actually see"
# question. On Meta-World `pythonic_class_abstraction` renders the adapter's
# own tables: 39 documented observation slots plus the 8 inlineable helper
# expressions of `bird/envs/metaworld.py::_HELPERS`, two of which are
# Meta-World's OWN shaping kernels, `tolerance()` and `hamacher_product()`.
# Handing those to a generator is strictly more than T2R or CARD handed theirs,
# and the two kernels in particular are the primitives the shipped v2 reward is
# written out of -- so a run under that member is a generator shown the
# reference reward's building blocks by name. That is a different method, which
# is why T2R and CARD are not credited to that member.
#
# WHAT THE MODEL CAN DO WITH IT. The seven readable names here are not BIRD
# symbols (eight attribute LINES, but `self.robot` and `self.obj1`/`self.obj2`
# are class-typed handles rather than quantities a reward reads):
# `self.obj1.position` is not anything the harness defines. They become
# executable through `generate.postprocess.symbol_mapping`, which is T2R's own
# mechanism (`post_process/post_process.py`,
# `RewardFunctionConverter.general_to_specific`) -- see
# `symbol_table_t2r_metaworld_global` below, which is this text's companion and
# carries the same seven symbols on its left-hand side. Pinning this member
# WITHOUT that table leaves every generated program referring to attributes
# that do not exist, so `_check_coherence` refuses the pair.
#
# `obj1` / `obj2` are deliberately anonymous: T2R never tells the model which
# object a task's obj1 IS. The task identity arrives only through
# `problem.task_description` (T2R's `{instruction}` slot,
# `metaworld_exp.py:6-17`), which is exactly how the released driver sends it.
_T2R_METAWORLD_CLASS_ABSTRACTION = """class BaseEnv(gym.env):
    self.robot : Robot # the robot in the environment
    self.obj1 : RigidObject # the first object in the environment
    self.obj2 : RigidObject # the second object in the environment, **if any**
    self.goal_position : np.ndarray[(3,)] # indicate the 3D position of the goal

class Robot:
    self.ee_position : np.ndarray[(3,)] # indicate the 3D position of the end-effector
    self.gripper_openness : float # a normalized measurement of how open the gripper is, range in [-1, 1]

class RigidObject:
    self.position : np.ndarray[(3,)] # indicate the 3D position of the rigid object
    self.quaternion : np.ndarray[(4,)] # indicate the quaternion of the rigid object"""

#: The string values of `generate.context.include_reward_recipe_hints`, each
#: naming the release FILE both methods carry it in. `true` is `_RECIPE_HINTS`
#: and `false` renders nothing; the schema enum and this dict are held in sync
#: by tests/test_reward_recipe.py.
_RECIPES: Dict[str, str] = {
    "t2r_metaworld": _RECIPE_T2R_METAWORLD,
    "t2r_panda": _RECIPE_T2R_PANDA,
}

_DIVERSITY_AXES = (
    "use a different set of reward components from the other variants",
    "use a different shaping schedule -- dense vs sparse, staged vs monolithic",
    "put the components on a different order of magnitude relative to each other",
    "penalise a different failure mode",
    "express the same objective in a different mathematical form "
    "(e.g. exponential vs quadratic vs piecewise)",
    "decompose the task at a different level of granularity",
)

#: Untried in the literature (`personas`, like `distinct_prompts`, is an
#: unexplored diversity mechanism). Implemented for real so the value is a
#: real point in the space, not a stub.
_PERSONAS = (
    ("control theorist",
     "You are a control theorist. You prefer smooth, well-conditioned potential "
     "functions and you say explicitly what the equilibrium of your reward is."),
    ("RL practitioner",
     "You are a reinforcement-learning practitioner. You care about gradient "
     "signal density and about whether the agent can discover the first reward "
     "by random exploration."),
    ("robotics engineer",
     "You are a robotics engineer. You think in contacts, joint limits and "
     "hardware-safe behaviour, and you distrust terms that are unbounded."),
    ("red-teamer",
     "You are a reward-hacking red-teamer. You write rewards that are hard to "
     "game, and you name the degenerate policy each term is there to rule out."),
    ("minimalist",
     "You are a minimalist. You use the fewest components that could possibly "
     "work, and you justify every one you keep."),
    ("curriculum designer",
     "You are a curriculum designer. You build staged rewards where later terms "
     "only become available once their prerequisites are satisfied."),
)

_SYSTEM_DEFAULT = (
    "You are a reward engineer. You write reward functions for a reinforcement "
    "learning agent. You answer with code, not with commentary."
)

#: The system default MINUS "You answer with code, not with commentary."
#: Used when `generate.context.include_reward_engineering_tips` is on: Eureka's
#: code_feedback explicitly elicits analyse-first-then-code ("Please analyze each
#: existing reward component in the suggested manner above first, and then write
#: the reward function code"), which the code-only sentence works directly
#: against.
_SYSTEM_TIPS = (
    "You are a reward engineer. You write reward functions for a reinforcement "
    "learning agent."
)

#: Eureka's code_feedback.txt, essentially verbatim
#: (refs/code/Eureka/eureka/utils/prompts/code_feedback.txt; appended after every
#: successful round's policy statistics, eureka.py:268-269). Here it follows the
#: NUMERIC REFLECTION body -- the same stats-then-instructions order -- and only
#: when that section is non-empty, matching upstream's successful-round gating.
_TIPS_ANALYSIS = (
    "Please carefully analyze the policy feedback and provide a new, improved "
    "reward function that can better solve the task. Some helpful tips for "
    "analyzing the policy feedback:\n"
    "    (1) If the success rates are always near zero, then you must rewrite "
    "the entire reward function\n"
    "    (2) If the values for a certain reward component are near identical "
    "throughout, then this means RL is not able to optimize this component as "
    "it is written. You may consider\n"
    "        (a) Changing its scale or the value of its temperature parameter\n"
    "        (b) Re-writing the reward component\n"
    "        (c) Discarding the reward component\n"
    "    (3) If some reward components' magnitude is significantly larger, then "
    "you must re-scale its value to a proper range\n"
    "Please analyze each existing reward component in the suggested manner above "
    "first, and then write the reward function code."
)

#: Eureka's code_output_tip.txt, the method-bearing tips
#: (refs/code/Eureka/eureka/utils/prompts/code_output_tip.txt; appended to the
#: system prompt at eureka.py:57 and to every feedback message at 218-221/276).
#: Two deliberate adaptations, and two deliberate omissions:
#:   * upstream tip (1) says "torch.exp"; rendered library-neutral
#:     ("an exponential") because BIRD rewards are numpy and a torch instruction
#:     would crash at runtime under verify.enabled: false;
#:   * upstream tip (4) drops its Isaac-specific "attributes of the provided
#:     environment class / self.-prefix" wording -- the surviving sentence is the
#:     method content ("under no circumstance can you introduce new input
#:     variables");
#:   * upstream tip (3) (TorchScript float-vs-Tensor input typing) is omitted:
#:     `_SIGNATURE` fixes the input types here;
#:   * upstream's return-shape and code-block-format sentences are already the
#:     OUTPUT CONTRACT and are not duplicated.
_TIPS_WRITING = (
    "Some helpful tips for writing the reward function code:\n"
    "    (1) You may find it helpful to normalize the reward to a fixed range "
    "by applying transformations like an exponential to the overall reward or "
    "its components\n"
    "    (2) If you choose to transform a reward component, then you must also "
    "introduce a temperature parameter inside the transformation function; this "
    "parameter must be a named variable in the reward function and it must not "
    "be an input variable. Each transformed reward component should have its "
    "own temperature variable\n"
    "    (3) The reward code must use only the variables the environment "
    "interface above declares; under no circumstance can you introduce new "
    "input variables."
)

#: RF-Agent's thought_code_output.txt:1-3, verbatim
#: (refs/code/RF-Agent/RF_Agent/utils/prompts_rfagent/thought_code_output.txt;
#: appended to every action prompt as `initial_action`, rfagent_algo.py:264-265).
#: Lines 3-7 of that file are the return-shape and code-block sentences, which
#: are already the OUTPUT CONTRACT body and are not duplicated.
_THOUGHT_CLAUSE = (
    "First, describe the design idea and main steps of your reward function in "
    "one sentence.\n"
    "The description must be inside a brace outside the code implementation.\n"
    "Next, write a reward function based on this idea."
)

#: The brace the clause above asks for, read back the way the release reads it:
#: the FIRST `{...}` of the whole response (rfagent_algo.py:426, `re.DOTALL`).
_THOUGHT_RE = re.compile(r"\{(.*?)\}", re.DOTALL)

#: The stable assembly order. `_HEAD` is pinned first and `_TAIL` last;
#: `generate.context.shuffle_sections` shuffles only what lies between.
_SECTION_ORDER = (
    "SAFETY REQUIREMENT",
    "ENVIRONMENT",
    "SUBTASKS",
    "CURRICULUM STAGE",
    "REWARD RECIPE",
    "GUIDANCE",
    "EXAMPLES",
    "PARENT REWARD",
    "NUMERIC REFLECTION",
    "BEHAVIOURAL ANALYSIS",
    "SEARCH ACTION",  # RF-Agent's per-action instruction; after the evidence it reasons over
    "HUMAN FEEDBACK",
    "PAST FAILURES",
    "ARCHIVE ELITES",
    "OBSERVATION CO-DESIGN",
    "DOMAIN RANDOMISATION",
    "PREVIOUS PROPOSAL",
    "DIVERSITY REQUIREMENT",
)


def _fmt_section(title: str, body: str) -> str:
    return f"## {title}\n{body.strip()}"


def _numeric_reflection(ctx: Context, report: CandidateReport) -> str:
    """Eureka: per-component scalar traces at checkpoints, verbalised. CARD's
    'process feedback' is the same family (§4).

    Prefers whatever §4 chose to put on `meta` -- §4's `feedback_default`
    writes `meta["numeric_reflection"]`
    (the `_section_numeric` text), so on every `numeric_reflection: true` method
    this section and the carried dialogue turn are one rendition, not two
    disagreeing ones. The fallback below survives for reports that lack the
    meta (a §4 builder that never renders the numeric section), and it strides
    the trace evenly across the whole run rather than truncating to its head --
    upstream verbalises `tensorboard_logs[metric][::epoch_freq]`, start to end
    (eureka.py:238,250), never the first ten points.
    """
    supplied = report.meta.get("numeric_reflection")
    if isinstance(supplied, str) and supplied.strip():
        return supplied
    # Shared with §4's rendition rather than copied, so the two cannot drift
    # about what "sampled at checkpoints" means. Local import mirrors `_get`:
    # evaluation.py does not import this module, so there is no cycle.
    from .evaluation import _REFLECT_POINTS, _sample_evenly

    show_scalar = bool(ctx.cfg.get("evaluate.feedback.state_selection_scalar", True))
    lines: List[str] = []
    traces = getattr(report.result, "component_traces", {}) or {}
    for name, values in sorted(traces.items()):
        nums = [float(v) for v in values]
        if not nums:
            continue
        # `_REFLECT_POINTS`, never a literal: a hardcoded 10 here would silently
        # disagree with §4's rendition the day that constant moves, leaving two
        # renditions of one curve that disagree.
        shown = ", ".join(f"{v:.3f}" for v in _sample_evenly(nums, _REFLECT_POINTS))
        lines.append(f"  {name}: [{shown}]  (max {max(nums):.3f}, "
                     f"mean {sum(nums) / len(nums):.3f}, min {min(nums):.3f})")
    if show_scalar and report.per_seed_fitness:
        lines.append("  fitness per seed: "
                     + ", ".join(f"{v:.4f}" for v in report.per_seed_fitness))
    if show_scalar and report.fitness is not None:
        lines.append(f"  fitness: {report.fitness:.4f} (source: {report.fitness_source})")
    if not lines:
        return ""
    return ("Component values at training checkpoints for the reward above:\n"
            + "\n".join(lines))


def _behavioural_analysis(ctx: Context, report: CandidateReport) -> str:
    """RDA's per-subtask scores + rationales; GT's behaviour summary (§4)."""
    supplied = report.meta.get("behavioural_analysis")
    if isinstance(supplied, str) and supplied.strip():
        return supplied
    lines: List[str] = []
    for name in sorted(report.subtask_scores):
        score = report.subtask_scores[name]
        why = report.subtask_rationales.get(name, "")
        did = (getattr(report, "subtask_behaviors", None) or {}).get(name, "")
        lines.append(f"  {name}: {score:.3f}"
                     + (f" -- behavior: {did}" if did else "")
                     + (f" -- {why}" if why else ""))
    if lines:
        return (_training_caveat(ctx, report)
                + "What the trained policy actually did, per subtask:\n" + "\n".join(lines))
    # Last resort: the prose channel §4 built, if it is behavioural at all.
    if report.feedback and report.feedback_channel != "numeric":
        return _training_caveat(ctx, report) + report.feedback
    return ""


def _training_caveat(ctx: Context, report: Any) -> str:
    """The STOPPED EARLY / wall-clock note for a pruned or timed-out parent, on
    the behavioural channel ONLY when there is no numeric one.

    The sentence itself is `evaluation.training_caveat` -- one implementation,
    worded on `train.pruning_metric`. `_section_numeric` prepends it wherever a
    curve is shown (§1's NUMERIC REFLECTION via `meta["numeric_reflection"]`,
    the carried turn via `report.feedback`), so on a method with
    `evaluate.feedback.numeric_reflection: true` repeating it here would
    announce one pruned winner twice in one prompt. It cannot be the ONLY
    emitter either: eureka never renders this section, so its pruned winners
    would be reflected on as complete curves.
    """
    if ctx.cfg.get("evaluate.feedback.numeric_reflection", False):
        return ""
    from .evaluation import training_caveat
    return training_caveat(ctx, report)


def _failure_traces(state: RunState) -> str:
    """LIMEN: recent failed programs + error traces as negative examples (§1)."""
    entries = list(state.failure_memory)[-_MAX_FAILURE_TRACES:]
    blocks: List[str] = []
    for entry in entries:
        # `error` is what the teach path writes; `failure` the loser path;
        # `traceback` is a legacy key, read so a memory carried out of an older
        # run renders its error rather than "(unrecorded)".
        err = str(entry.get("error") or entry.get("failure")
                  or entry.get("traceback") or "").strip()
        code = str(entry.get("code") or entry.get("reward_code") or "").strip()
        block = "- error: " + (err or "(unrecorded)")
        if code:
            block += "\n  code:\n```python\n" + code + "\n```"
        blocks.append(block)
    if not blocks:
        return ""
    return ("These programs were generated earlier and FAILED. Do not repeat "
            "their mistakes:\n" + "\n".join(blocks))


def _archive_elites(ctx: Context, state: RunState) -> str:
    """LIMEN: top-performing archive entries as positive examples (§1).

    Sorted over EVERY stored program -- grid elites and the island population
    slots that hold filed cell losers -- matching the release's
    `get_top_programs`, which ranks `self.programs` wholesale rather than one
    per niche (`database.py` L382-403). Each program is quoted through
    `_inheritable_program`: the archive is the cross-iteration store, so an
    elite's observation shown here is an observation persisting.
    """
    cells = [c for c in state.archive.values() if getattr(c, "report", None) is not None]
    cells.sort(key=lambda c: -(float(getattr(c, "fitness", 0.0) or 0.0)))
    blocks: List[str] = []
    for cell in cells[:_MAX_ARCHIVE_ELITES]:
        code = _inheritable_program(ctx, cell.report.candidate.reward_code or "").strip()
        blocks.append(
            f"- cell {tuple(cell.coords)} (island {getattr(cell, 'island', 0)}), "
            f"fitness {float(getattr(cell, 'fitness', 0.0) or 0.0):.4f}:\n"
            "```python\n" + code + "\n```")
    if not blocks:
        return ""
    return ("High-performing rewards already found. "
            "Produce something that occupies a DIFFERENT niche:\n" + "\n".join(blocks))


def _subtasks_section(ctx: Context, state: RunState) -> str:
    """RDA's subtask list `T^i`; L2R's motion descriptors are the same move made
    with a second LLM (§1)."""
    if not state.subtasks:
        return ""
    listing = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(state.subtasks))
    text = "The task has been decomposed into these subtasks:\n" + listing
    if ctx.cfg["generate.decomposition.reward_must_map_to_subtasks"]:
        text += ("\n\nYour reward must be a weighted sum of per-subtask components, "
                 "one component per subtask above, named after it.")
    return text


def _curriculum_section(ctx: Context, state: RunState) -> str:
    """The ONE stage this reward is for, and the stages either side of it.

    The whole hypothesis of the curriculum axis is that the generator is being
    asked for too much at once, so this section has to do two things and they
    pull against each other: name one stage as the target, and give enough of
    the ordering that the model does not write a reward which destroys the
    behaviour the previous stages established. Hence what is included --

      * the stages already PASSED, marked as such, because they are the
        behaviour the warm-started policy already has and must keep;
      * the current stage, marked as the only one to write a reward for;
      * the stages still ahead, named but explicitly out of scope, because a
        model that cannot see where the curriculum is going writes stage
        rewards that dead-end;
      * `note`, which is where a regression warning or a re-split lands. A
        rollback the prompt cannot see is a silent rewrite of what the next
        reward is being written against.

    Empty when there is no curriculum, so the section simply does not render --
    `_build_messages` drops empty bodies, and a heading over nothing would be a
    claim that a curriculum exists.
    """
    cur = getattr(state, "curriculum", None)
    if cur is None or not cur.stages:
        return ""
    lines = []
    for i, stage in enumerate(cur.stages):
        if i < cur.index:
            lines.append(f"  {i + 1}. [DONE] {stage}")
        elif i == cur.index:
            lines.append(f"  {i + 1}. [WRITE THE REWARD FOR THIS ONE] {stage}")
        else:
            lines.append(f"  {i + 1}. [later, not yet] {stage}")
    text = ("The task is being learned as an ordered curriculum. The policy is "
            "carried from one stage to the next, so it already has whatever the "
            "stages marked DONE established.\n" + "\n".join(lines))
    if cur.complete:
        text += ("\n\nEvery stage is done. Write a reward for the whole task, "
                 "keeping all of the behaviour above.")
    else:
        text += ("\n\nYour reward is for the current stage ONLY. It must not "
                 "reward a later stage, and it must not remove any incentive to "
                 "keep doing the stages already marked DONE.")
    if cur.note:
        text += "\n\n" + cur.note
    return text


def _parent_section(ctx: Context, parents: Sequence[CandidateReport],
                    lead: Optional[str] = None) -> str:
    """PARENT REWARD. One parent renders as a single block; a GROUP -- more than
    one parent, or a caller-supplied `lead` -- renders RF-Agent's
    `base_thought_code.txt` blocks ("No.{i} reward function:", the design idea,
    the code, then that parent's OWN trained result), because a group prompt
    (crossover, path reasoning, different thought) reasons over several trained
    rewards at once and the single NUMERIC REFLECTION section can carry only
    one (rfagent_algo.py:296-303). `lead=""` drops the lead sentence: the tree
    sampler's SEARCH ACTION section says what the group is for.
    """
    if not parents:
        return ""
    # Same gate as `_section_header` / `_numeric_reflection`: with
    # `evaluate.feedback.state_selection_scalar: false` no prompt-visible line
    # may label the selection scalar. With `n_survivors: 1` the parent IS the
    # previous round's argmax, so an ungated "(fitness <v>)" here would state
    # the previous best's labeled value on every eureka prompt (upstream never
    # labels it, eureka.py:250-277).
    show_scalar = bool(ctx.cfg.get("evaluate.feedback.state_selection_scalar", True))
    # `generate.context.include_parent_thought`: the "Design Idea:" line of
    # base_thought_code.txt:2, above the code.
    show_thought = bool(ctx.cfg["generate.context.include_parent_thought"])
    if len(parents) > 1 or lead is not None:
        return _parent_group(ctx, parents, lead, show_thought)
    blocks = []
    for report in parents:
        code = _inheritable_program(ctx, report.candidate.reward_code or "").strip()
        head = f"Reward {report.cand_id}"
        if show_scalar and report.fitness is not None:
            head += f" (fitness {report.fitness:.4f})"
        idea = f"Design Idea: {_design_thought(report)}\n" if show_thought else ""
        blocks.append(head + ":\n" + idea + "```python\n" + code + "\n```")
        if report.candidate.weights:
            blocks.append("weights: " + json.dumps(report.candidate.weights))
    # By the number of PARENTS, not of rendered blocks: a parent under
    # `output.format: component_dict_plus_weights` renders two blocks (code +
    # weights), so `len(blocks) == 1` would introduce every RDA/GT single
    # parent with the crossover sentence -- "combine" is an instruction for a
    # different operator than `single_parent_hillclimb`'s "one modification"
    # (App. 7.5).
    lead = ("This is the reward to improve on:" if len(parents) == 1 else
            "These are the parent rewards to combine and improve on:")
    return lead + "\n" + "\n\n".join(blocks)


def _parent_group(ctx: Context, parents: Sequence[CandidateReport],
                  lead: Optional[str], show_thought: bool) -> str:
    """RF-Agent's `base_thought_code.txt`, one block per parent, numbered from
    1 in the order given (rfagent_algo.py:299-303: `i=i + 1`). The release's
    "Code: {reward_function}" slot is a fenced block here so the parent code is
    quoted the way every other section quotes code."""
    blocks = []
    for i, report in enumerate(parents, start=1):
        code = _inheritable_program(ctx, report.candidate.reward_code or "").strip()
        lines = [f"No.{i} reward function:"]
        if show_thought:
            lines.append(f"Design Idea: {_design_thought(report)}")
        lines.append("```python\n" + code + "\n```")
        if report.candidate.weights:
            lines.append("weights: " + json.dumps(report.candidate.weights))
        lines.append("Trained result:\n" + _trained_result(ctx, report))
        blocks.append("\n".join(lines))
    if lead is None:
        lead = "These are the parent rewards to combine and improve on:"
    body = "\n\n".join(blocks)
    return lead + "\n" + body if lead else body


def _design_thought(report: CandidateReport) -> str:
    """The idea shown for a parent. §4's thought alignment rewrites the idea
    from the compiled code and the release shows THAT rewrite thereafter
    (rfagent_algo.py:655, `node.design_thought = design_thought` after the
    align call), so the aligned `meta["design_thought"]` wins over the brace
    the generator wrote (`nl_spec`). "(none)" is a fact about the record, not a
    blank the model could mistake for an idea."""
    thought = report.meta.get("design_thought") or report.candidate.nl_spec or ""
    return " ".join(str(thought).split()) or "(none)"


def _trained_result(ctx: Context, report: CandidateReport) -> str:
    """The "Trained result:" slot of base_thought_code.txt:4. For a trained
    parent it is the numeric reflection (§4's rendition when `meta` carries
    it, `_numeric_reflection`'s fallback otherwise). For one that never ran, it
    is what the release puts in the same slot -- the failure text
    (`node.history_exec_state = traceback_msg`, rfagent_algo.py:665) -- so an
    elite that failed is shown AS a failure and not as a reward with no data."""
    error = getattr(report.result, "error", "") or ""
    if report.candidate.valid and not error:
        text = _numeric_reflection(ctx, report)
        if text.strip():
            return text
    return (report.feedback or error or report.candidate.failure
            or "(no training result recorded)")


# --- few-shot retrieval ---------------------------------------------------


def _fewshot_section(ctx: Context, state: RunState) -> str:
    """Text2Reward's few-shot examples: k=3, OpenAI embeddings + Chroma (§1).

    Retrieval degrades on purpose: with no embedding client available offline,
    `semantic_similarity` falls back to deterministic lexical overlap against the
    task description. That is a weaker retriever, not a different design choice,
    and it is recorded on the candidate's `meta` so a run never silently claims
    an embedding retriever it did not have.
    """
    if not ctx.cfg["generate.fewshot.enabled"]:
        return ""
    corpus = _fewshot_corpus(ctx, state)
    if not corpus:
        return ""
    k = max(1, min(int(ctx.cfg.get("generate.fewshot.k", 3) or 3), _MAX_FEWSHOT))
    retriever = ctx.cfg["generate.fewshot.retriever"]
    if retriever == "random":
        chosen = [corpus[i] for i in _sample_indices(ctx, len(corpus), k)]
    elif retriever == "fixed":
        chosen = corpus[:k]
    else:  # semantic_similarity, lexical fallback offline
        query = str(ctx.cfg["problem.task_description"])
        scored = sorted(corpus, key=lambda ex: (-_overlap(query, ex), ex))
        chosen = scored[:k]
    blocks = ["```python\n" + ex.strip() + "\n```" for ex in chosen]
    return "Examples of reward functions for other tasks:\n" + "\n\n".join(blocks)


def _sample_indices(ctx: Context, n: int, k: int) -> List[int]:
    idx = list(range(n))
    ctx.rng.shuffle(idx)
    return sorted(idx[:k])


def _overlap(query: str, example: str) -> float:
    a = set(re.findall(r"[a-z_]{3,}", query.lower()))
    b = set(re.findall(r"[a-z_]{3,}", example.lower()))
    return len(a & b) / max(1, len(a | b))


def _fewshot_corpus(ctx: Context, state: RunState) -> List[str]:
    """`corpus: self` = this run's past winners; otherwise a path to a .py file,
    a directory of them, or a JSON list of code strings."""
    spec = ctx.cfg.get("generate.fewshot.corpus")
    if not spec:
        return []
    if str(spec) == "self":
        pool = [r for r in state.all_reports if r.candidate.valid and r.fitness is not None]
        pool.sort(key=lambda r: (-(r.fitness or 0.0), r.cand_id))
        return [r.candidate.reward_code for r in pool[:_MAX_FEWSHOT]
                if r.candidate.reward_code]
    from pathlib import Path

    path = Path(str(spec))
    try:
        if path.is_dir():
            return [p.read_text() for p in sorted(path.glob("*.py"))[:_MAX_FEWSHOT]]
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            return [str(x) for x in data][:_MAX_FEWSHOT]
        if path.exists():
            return [path.read_text()]
    except OSError as exc:
        log.warning("generate.fewshot.corpus %s unreadable: %s", spec, exc)
    return []


# --- co-design sections ---------------------------------------------------


def _observation_section(ctx: Context) -> str:
    """LIMEN: emit `get_observation` alongside `compute_reward` (§1).

    The dimension cap is real: unconstrained LLMs emit ~174-feature vectors and
    LIMEN caps at 512. Stating the cap here and enforcing it at parse time is the
    same constraint applied twice, which is what makes the cap load-bearing
    rather than decorative.
    """
    if not ctx.cfg["generate.co_design.observation_fn"]:
        return ""
    cap = int(ctx.cfg["generate.co_design.observation_max_dim"])
    return (
        "In the SAME code block, also emit an observation function:\n\n"
        "    def get_observation(state):\n\n"
        "returning a flat sequence of floats -- the features the policy sees. "
        f"It must return at most {cap} features. Declare the count explicitly as a "
        f"module-level `OBS_DIM = <int>` so the constraint can be checked.\n"
        "The reward and the observation are co-designed: choose features that make "
        "your reward learnable."
    )


def _dr_section(ctx: Context, state: RunState) -> str:
    """In-prompt DR co-design (§1). Repo-authored: DrEureka's release requests no
    DR block in its reward prompt (initial_user.txt:1) and trains stage 1
    `--dr-config off` (eureka.py:162); its DR is written once, after the search.

    `dr_prior` matters more than anything else in this section -- the ablations
    show `none`/uninformative priors fail badly. `rapp` reads the bounds the RAPP
    pre-phase produced; the probe order below is the contract with that phase.
    """
    if not ctx.cfg["generate.co_design.dr_config"]:
        return ""
    prior_mode = ctx.cfg["generate.co_design.dr_prior"]
    n_configs = int(ctx.cfg.get("generate.co_design.dr_n_configs", 1) or 1)
    text = (
        "Also emit a domain-randomisation configuration, in a separate ```json "
        "code block: a flat dict mapping each randomised physics parameter to a "
        "two-element `[low, high]` range.\n"
        f"The search will consider {n_configs} configurations in total; yours must "
        "be a distinct member of that set."
    )
    prior = _dr_prior_text(ctx, prior_mode)
    if prior:
        text += "\n\n" + prior
    elif prior_mode == "none":
        # Recorded, not silently absent: DrEureka's own ablation is that this
        # setting fails badly, so a run using it should look deliberate.
        text += "\n\n(No physics prior is supplied; choose ranges from the task alone.)"
    return text


def _dr_prior_text(ctx: Context, mode: str) -> str:
    if mode == "none":
        return ""
    sources = {
        "rapp": ("rapp_bounds", "rapp", "physics_prior"),
        "default_sim_ranges": ("default_dr_ranges", "dr_ranges", "sim_ranges"),
    }.get(mode, ())
    bounds: Any = None
    for holder in (ctx, getattr(ctx, "counters", {}), ctx.env):
        for name in sources:
            got = (holder.get(name) if isinstance(holder, dict)
                   else getattr(holder, name, None))
            if got:
                bounds = got
                break
        if bounds:
            break
    if not bounds:
        return ""
    label = ("Reward-Aware Physics Prior (RAPP) -- the range over which the "
             "incumbent policy still succeeds" if mode == "rapp"
             else "Default simulator ranges")
    return f"{label}:\n```json\n{json.dumps(bounds, indent=2, default=str)}\n```"


# --- the assembler --------------------------------------------------------


def _build_messages(ctx: Context, state: RunState, parents: Sequence[CandidateReport],
                    *, persona: Optional[Tuple[str, str]] = None,
                    extra: Optional[Dict[str, str]] = None,
                    tail: Optional[str] = None,
                    call_index: Optional[int] = None,
                    parent_lead: Optional[str] = None) -> List[Dict[str, str]]:
    """Assemble one prompt. `extra` injects sampler-specific sections by title
    (they must be members of `_SECTION_ORDER` so ordering stays declared in one
    place); `tail` overrides the pinned output contract (two-stage NL->code).

    `call_index` says which of a candidate's LLM calls this prompt is for, and
    is read by nothing but the history block: `None` (the default, and every
    single-call method) means the global `generate.history_mode`.

    `parent_lead` is handed to `_parent_section` unchanged: `None` (the
    default) keeps its own lead sentence, a string replaces it, `""` removes it
    -- the tree sampler's group prompts carry their instruction in SEARCH
    ACTION instead.
    """
    cfg = ctx.cfg
    ctxcfg = "generate.context."
    extra = extra or {}

    bodies: Dict[str, str] = {}

    if cfg[ctxcfg + "include_safety_instruction"]:
        # DrEureka's `l_safety` (Eq. 6), appended to `l_task` -- stage 1's ENTIRE
        # delta from Eureka. A null instruction with the flag on would make that
        # delta vanish, so it falls back to a real clause.
        bodies["SAFETY REQUIREMENT"] = str(
            cfg.get(ctxcfg + "safety_instruction") or _DEFAULT_SAFETY)

    env_spec_fn = _get("env_spec", cfg[ctxcfg + "env_spec"])
    bodies["ENVIRONMENT"] = env_spec_fn(ctx, state)

    if cfg[ctxcfg + "include_task_decomposition"]:
        bodies["SUBTASKS"] = _subtasks_section(ctx, state)
    if cfg[ctxcfg + "include_curriculum_stage"]:
        bodies["CURRICULUM STAGE"] = _curriculum_section(ctx, state)
    recipe = cfg[ctxcfg + "include_reward_recipe_hints"]
    if isinstance(recipe, str):
        # Text2Reward / CARD in-prompt template, the published benchmark text
        # byte for byte (`t2r_metaworld` | `t2r_panda`) -- Eureka's explicit
        # anti-value ("completely free of ... reward templates").
        bodies["REWARD RECIPE"] = _RECIPES[recipe]
    elif recipe:
        # `true`: BIRD's own environment-neutral paraphrase, not a published text.
        bodies["REWARD RECIPE"] = _RECIPE_HINTS
    guidance = cfg.get(ctxcfg + "guidance")
    if guidance and str(guidance).strip():
        bodies["GUIDANCE"] = str(guidance)
    bodies["EXAMPLES"] = _fewshot_section(ctx, state)
    if cfg[ctxcfg + "include_parent_code"]:
        bodies["PARENT REWARD"] = _parent_section(ctx, parents, lead=parent_lead)
    tips_on = bool(cfg[ctxcfg + "include_reward_engineering_tips"])
    # `generate.context.reflection_guidance`: ours, no paper pin. Where the tips
    # path would render _TIPS_ANALYSIS it substitutes for it (same position, same
    # gating); otherwise it is appended once after whichever feedback section
    # renders, so a method with only behavioural analysis (rda) can carry an
    # analyse-the-feedback instruction too.
    reflection_guidance = cfg.get(ctxcfg + "reflection_guidance")
    reflection_guidance = (str(reflection_guidance)
                           if reflection_guidance and str(reflection_guidance).strip()
                           else None)
    grouped = len(parents) > 1 or parent_lead is not None
    if cfg[ctxcfg + "include_numeric_reflection"] and parents:
        if not grouped:
            bodies["NUMERIC REFLECTION"] = _numeric_reflection(ctx, parents[0])
            if tips_on and bodies["NUMERIC REFLECTION"].strip():
                # Eureka's stats-then-code_feedback order, on successful rounds
                # only (eureka.py:268-269): an empty section is a round with no
                # stats, and upstream sends no analysis tips there either.
                bodies["NUMERIC REFLECTION"] += "\n\n" + (reflection_guidance
                                                          or _TIPS_ANALYSIS)
                reflection_guidance = None  # consumed: substituted for _TIPS_ANALYSIS
        elif tips_on and bodies.get("PARENT REWARD", "").strip():
            # A group prompt carries each parent's reflection inside its own
            # PARENT REWARD block (`_parent_group`), so a NUMERIC REFLECTION of
            # parents[0] alone would restate one of them. The analysis tips then
            # follow the group, which is where the release puts them
            # (action_3_crossover_elite.txt:5-6, "Analysis tips for trained
            # results:" straight after `{reward_func_group}`).
            bodies["PARENT REWARD"] += "\n\n" + (reflection_guidance or _TIPS_ANALYSIS)
            reflection_guidance = None
    if cfg[ctxcfg + "include_behavioural_analysis"] and parents:
        bodies["BEHAVIOURAL ANALYSIS"] = _behavioural_analysis(ctx, parents[0])
    if reflection_guidance and parents:
        # PARENT REWARD is a home for it only on a group prompt -- see above.
        homes = ("NUMERIC REFLECTION", "BEHAVIOURAL ANALYSIS") + \
            (("PARENT REWARD",) if grouped else ())
        for _feedback_title in homes:
            if bodies.get(_feedback_title, "").strip():
                bodies[_feedback_title] += "\n\n" + reflection_guidance
                break
    if cfg[ctxcfg + "include_human_feedback"]:
        # GT: free-text feedback on the selected agent, one per iteration,
        # injected as "ground truth"; removing it measurably degrades (§5.1.2).
        if state.human_feedback:
            bodies["HUMAN FEEDBACK"] = (
                "A human watched the current agent and said, verbatim -- treat this "
                "as ground truth:\n\n" + state.human_feedback)
    if cfg[ctxcfg + "include_failure_traces"]:
        bodies["PAST FAILURES"] = _failure_traces(state)
    if cfg[ctxcfg + "include_archive_elites"]:
        bodies["ARCHIVE ELITES"] = _archive_elites(ctx, state)

    bodies["OBSERVATION CO-DESIGN"] = _observation_section(ctx)
    bodies["DOMAIN RANDOMISATION"] = _dr_section(ctx, state)
    bodies.update(extra)

    middle = [title for title in _SECTION_ORDER if bodies.get(title, "").strip()]
    if cfg[ctxcfg + "shuffle_sections"]:
        # LIMEN: randomised section order for decode diversity. The task head and
        # the format tail stay pinned -- they are structural, not evidence.
        ctx.rng.shuffle(middle)

    head = "## TASK\n" + str(cfg["problem.task_description"]).strip()
    if tail is None:
        tail = _fmt_section("OUTPUT CONTRACT", _get(
            "output_format", cfg["generate.output.format"])(ctx))
    user = "\n\n".join([head] + [_fmt_section(t, bodies[t]) for t in middle] + [tail])

    # Eureka's system prompt is initial_system + code_output_tip (eureka.py:57),
    # and its code_feedback elicits analysis prose FIRST -- so with the tips on,
    # the "code, not commentary" sentence comes out of the default system message.
    # `generate.context.system_prompt` (ours, no paper pin) replaces the whole
    # computed message -- nothing is appended to a replacement, so what the
    # config author wrote is exactly what the model reads. The personas clash is
    # refused at config load (_check_coherence), not resolved here.
    system_override = cfg.get(ctxcfg + "system_prompt")
    if system_override and str(system_override).strip():
        system = str(system_override)
    else:
        system = persona[1] if persona else (_SYSTEM_TIPS if tips_on else _SYSTEM_DEFAULT)
        if tips_on:
            system = system + "\n\n" + _TIPS_WRITING
    messages = [{"role": "system", "content": system}]
    messages.extend(_history_messages(ctx, state, call_index))
    messages.append({"role": "user", "content": user})
    return messages


def _get(kind: str, name: str) -> Callable[..., Any]:
    from ..registry import get  # local import: registry imports this module

    return get(kind, name)


# ==========================================================================
# Response parsing
# ==========================================================================


def _extract_code(ctx: Context, raw: str) -> Tuple[Optional[str], str]:
    """Apply `generate.parse.patterns` in order; returns (code, pattern_used).

    Syntax is NOT checked here: an extracted block that does not compile is a
    §2 `ast_syntax` failure, and stealing it into §1 would move a candidate
    between the two populations that must stay apart.

    The one exception is the leniency floor at the end -- a response that IS the
    program, with no fence at all. It is admitted only if it parses as Python and
    defines something, which is exactly the evidence that no fence was needed.

    The body lives in `bird/parsing.py`, shared with stage 2's repair path
    (`verification._extract_code`) so a repaired sample is parsed by the same
    rule as the original -- a private copy would drift.
    """
    return extract_code(ctx.cfg["generate.parse.patterns"] or [], raw)


def _max_retries(ctx: Context) -> int:
    value = ctx.cfg["generate.parse.max_retries"]
    if isinstance(value, str):
        if value.strip().lower() in ("inf", "infinity"):
            return _INF_RETRY_CEILING  # Text2Reward loops until a block appears
        raise ValueError(
            f"generate.parse.max_retries: {value!r} is not an int or 'inf'")
    return int(value or 0)


_COMPONENT_HOLDERS = frozenset({
    "components", "component", "reward_components", "rew_dict", "reward_dict",
    "rewards", "info", "component_dict",
})


def _parse_component_names(code: str) -> List[str]:
    """Recover the named components the program returns.

    Reads the AST rather than the text, because the component dict is the record
    §4's per-component feedback is built from -- a regex that mistakes a config
    dict for the component dict silently mislabels every reflection downstream.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    def keys_of(node: ast.AST) -> List[str]:
        if not isinstance(node, ast.Dict):
            return []
        return [k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)]

    # 1. `return total, {...}` -- the declared contract.
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) \
                and len(node.value.elts) == 2:
            names = keys_of(node.value.elts[1])
            if names:
                return names
    # 2. a dict literal assigned to a components-ish name.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in _COMPONENT_HOLDERS:
                    names = keys_of(node.value)
                    if names:
                        return names
    # 3. built up by subscript assignment: `components["reach"] = ...`
    found: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) \
                        and isinstance(target.value, ast.Name) \
                        and target.value.id in _COMPONENT_HOLDERS \
                        and isinstance(target.slice, ast.Constant) \
                        and isinstance(target.slice.value, str) \
                        and target.slice.value not in found:
                    found.append(target.slice.value)
    return found


#: Module-level names `_parse_weights` will read a weights/params dict from.
#: `reward_params` is the name `output_template_params` induces; without it
#: `template_dsl`'s only output would parse as empty on every run.
_WEIGHT_NAMES = ("WEIGHTS", "weights", "PARAMS", "params", "TEMPLATE_PARAMS",
                 "reward_params", "REWARD_PARAMS")


def _flat_scalar_dict(obj: Any) -> Dict[str, float]:
    if not isinstance(obj, dict):
        return {}
    out: Dict[str, float] = {}
    for key, value in obj.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _parse_weights(raw: str, code: str) -> Dict[str, float]:
    """Pull the explicit weights/params dict out of the response.

    Three sources in order: a fenced ```json block (what the contract asks for),
    a module-level `WEIGHTS = {...}` in the code, then any bare JSON object in
    the response. Only flat scalar dicts are accepted -- a nested object is not
    a weights dict and quietly flattening one would fabricate weights.
    """
    for match in re.finditer(r"```(?:json)\s*(.*?)```", raw, re.DOTALL):
        try:
            found = _flat_scalar_dict(json.loads(match.group(1)))
        except (ValueError, TypeError):
            continue
        if found:
            return found
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in _WEIGHT_NAMES:
                        try:
                            found = _flat_scalar_dict(ast.literal_eval(node.value))
                        except (ValueError, SyntaxError):
                            continue
                        if found:
                            return found
    for match in re.finditer(r"\{[^{}]*\}", raw, re.DOTALL):
        try:
            found = _flat_scalar_dict(json.loads(match.group(0)))
        except (ValueError, TypeError):
            continue
        if found:
            return found
    return {}


def _parse_dr_config(raw: str, code: str) -> Optional[Dict[str, Any]]:
    """DrEureka: recover `{param: [low, high]}` ranges from the response."""
    def ranges(obj: Any) -> Dict[str, Any]:
        if not isinstance(obj, dict):
            return {}
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    out[str(key)] = [float(value[0]), float(value[1])]
                except (TypeError, ValueError):
                    continue
        return out

    for match in re.finditer(r"```(?:json|yaml)?\s*(.*?)```", raw, re.DOTALL):
        try:
            found = ranges(json.loads(match.group(1)))
        except (ValueError, TypeError):
            found = {}
        if found:
            return found
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                            "DR_CONFIG", "dr_config", "DR_RANGES", "dr_ranges"):
                        try:
                            found = ranges(ast.literal_eval(node.value))
                        except (ValueError, SyntaxError):
                            continue
                        if found:
                            return found
    # Last resort: `param: [lo, hi]` lines. (Repo-authored; DrEureka's release
    # has no `parse_dr` -- dr_eureka.py:109-128 pastes the code block verbatim.)
    found = {}
    for name, lo, hi in re.findall(
            r"^\s*([A-Za-z_][\w.]*)\s*:\s*\[\s*(-?[\d.eE+]+)\s*,\s*(-?[\d.eE+]+)\s*\]",
            raw, re.MULTILINE):
        try:
            found[name] = [float(lo), float(hi)]
        except ValueError:
            continue
    return found or None


def _extract_observation(code: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (get_observation source, declared/estimated feature count)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, None
    declared: Optional[int] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("OBS_DIM", "obs_dim"):
                    if isinstance(node.value, ast.Constant) and \
                            isinstance(node.value.value, int):
                        declared = int(node.value.value)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_observation":
            src = ast.get_source_segment(code, node)
            return src, declared if declared is not None else _estimate_obs_dim(node)
    return None, declared


def _estimate_obs_dim(fn: ast.FunctionDef) -> Optional[int]:
    """Count features when the model did not declare `OBS_DIM`.

    Deliberately conservative: returns None when the shape is not literal, and
    the caller then admits the candidate with an `obs_dim_unknown` note rather
    than failing it. Refusing a candidate on a guess would make LIMEN's cap look
    stricter than it is.
    """
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, (ast.List, ast.Tuple)):
            if all(not isinstance(e, ast.Starred) for e in node.value.elts):
                return len(node.value.elts)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            for arg in node.value.args:
                if isinstance(arg, (ast.List, ast.Tuple)):
                    return len(arg.elts)
    appends = sum(1 for node in ast.walk(fn)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "append")
    return appends or None


def _strip_observation(code: str) -> str:
    """`code` with its module-level `get_observation` and `OBS_DIM` cut out.

    The complement of `_extract_observation`: that lifts the observation OUT of
    a program and leaves the program intact (the view), this returns the program
    WITHOUT it. Module-level definitions only -- the shape `_observation_section`
    asks for, and the only shape the mock and every real model so far have
    written; a nested one is left where it is. A program that does not parse, or
    defines no observation, comes back unchanged, so this is the identity on
    every program that has nothing to strip.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    gone: set = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_observation":
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            gone.update(range(start, (node.end_lineno or node.lineno) + 1))
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("OBS_DIM", "obs_dim")
                for t in node.targets):
            gone.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    if not gone:
        return code
    kept = [ln for i, ln in enumerate(code.splitlines(), start=1) if i not in gone]
    # At most two blank lines in a row (what the cut leaves between top-level
    # defs), and no trailing blank lines: the parent block is quoted verbatim.
    return re.sub(r"\n{4,}", "\n\n\n", "\n".join(kept)).rstrip("\n")


def _inheritable_program(ctx: Context, code: str) -> str:
    """What of a PARENT's program the child is handed (§6 `update.co_evolve`).

    `update.co_evolve.observation_fn` is the §6 half of §1's
    `generate.co_design.observation_fn`: whether the co-designed observation
    PERSISTS across iterations. LIMEN (true) evolves reward and observation as
    one program, so a child is shown its parent's whole program -- `OBS_DIM`
    and `get_observation` included -- and, under a non-rewriting
    `generate.output.edit_mode`, inherits it. With it false the observation is
    designed afresh every iteration: the parent's observation is cut out of
    everything the child is handed (PARENT REWARD, ARCHIVE ELITES, the program
    an edit mode patches), so each candidate carries the observation its OWN
    generation produced and only the reward rides across the boundary.

    Gated on `co_design.observation_fn` too: with §1's elicitation off, a
    `get_observation` in a program is a helper the reward may call, not a
    co-designed artefact, and there is nothing to persist or not persist.

    `--diff limen limen_reward_only` lists this key as a contribution, so
    flipping it alone must change what a child is handed.
    """
    cfg = ctx.cfg
    if not cfg["generate.co_design.observation_fn"] or cfg["update.co_evolve.observation_fn"]:
        return code
    return _strip_observation(code)


# --- post-processing ------------------------------------------------------


# ==========================================================================
# symbol_table  --  symbol_table_fn(ctx) -> dict[str, str]
# ==========================================================================
#
# The string values of `generate.postprocess.symbol_mapping`. They are a
# registry family for the reason `bird/registry.py` gives: a config value is a
# registry key, and `t2r_metaworld_global` is a table this repo authors rather
# than one an adapter happens to expose, so an adapter attribute is not an
# honest home for it.
# The key still accepts an inline dict, so it is NOT a `kind=` field -- an
# unregistered NAME is caught by `_check_coherence` instead.


@register("symbol_table", "per_task")
def symbol_table_per_task(ctx: Context) -> Dict[str, str]:
    """The table the ENVIRONMENT supplies for the task in play -- T2R's
    ManiSkill2 shape, where `query_llm_maniskill.py:56-62` maps each task id to
    its own prompt/mapping pair, and the value CARD inherits.

    On this repo the table is the task spec's `symbol_mapping:` block
    (`bird/tasks.py:342`, reaching the adapter through `bird/envs/spec.py:319`),
    so the vocabulary is whatever that spec author wrote. Note what that means
    on Meta-World: the six paper tasks' blocks are BIRD's own names
    (`gripper_to_object`, `object_position`, ...), which no published prompt
    shows a model -- so `per_task` beside `env_spec: t2r_class_abstraction` maps
    nothing, and `t2r_metaworld_global` is the value that pairs with it.

    An adapter may expose a callable or a dict-of-named-tables; both shapes
    are accepted."""
    table = getattr(ctx.env, "symbol_mapping", None)
    if callable(table):
        try:
            table = table("per_task")
        except TypeError:
            table = table()
    if isinstance(table, dict) and table and all(
            isinstance(v, dict) for v in table.values()):
        table = table.get("per_task", {})  # a dict of named tables, indexed by name
    if isinstance(table, dict):
        return {str(k): str(v) for k, v in table.items()}
    log.warning("postprocess.symbol_mapping=per_task but ctx.env exposes no "
                "symbol_mapping; leaving the code unmapped")
    return {}


#: Text2Reward's Meta-World general->specific table, `metaworld_exp.py:19-27`.
#:
#: It is ONE table for all ten of T2R's Meta-World tasks -- not a per-task one.
#: Per-task mapping is the ManiSkill2 half of the release only. CARD's release
#: is missing its copy of the module that holds the Meta-World table
#: (`metaworld_exp.mapping_dicts`); T2R's release has it, and this is it.
#:
#: THE LEFT-HAND SIDES ARE VERBATIM. They are exactly the seven readable names
#: `_T2R_METAWORLD_CLASS_ABSTRACTION` shows the model, which is the property
#: that makes the pair work.
#:
#: THE RIGHT-HAND SIDES ARE UPSTREAM'S TOO, for six of the seven, and the one
#: exception is forced. They are not BIRD's own `s[...]` convention, because
#: the substituted code has to be GRAMMATICAL IN THE FUNCTION THE CONTRACT ASKS
#: FOR. Under `generate.output.signature: t2r_compute_dense_reward` the
#: observation parameter is named `obs`, so `obs[4:7]` resolves and `s[4:7]`
#: would be a bare NameError -- a configuration that could not produce one
#: runnable candidate whatever the model wrote. `_check_coherence` ties the
#: two keys together so the pairing cannot come apart.
#:
#: The slices themselves never needed changing: Meta-World's own 18-slot current
#: frame is what both sides index. Only the goal moves, from upstream's live
#: `self.env._get_pos_goal()` to `obs[36:39]`.
#:
#: MEASURED, and the measurement is the whole argument for this spelling: T2R's
#: six released Meta-World programs, mapped with this table, COMPILE AND EXECUTE
#: through `CompiledReward` with binder plan `[-1, 1, 0]` (`self` -> None,
#: `action` -> a, `obs` -> s) and return finite rewards -- upstream's own
#: artifacts, run in this repo, with nothing transcribed
#: (`tests/test_t2r_class_abstraction.py`).
#:
#: ONE UPSTREAM ERROR IS REPRODUCED ON PURPOSE. The prompt tells the model
#: `gripper_openness` is "range in [-1, 1]"; the slot it is mapped to is
#: Meta-World's `gripper_distance_apart`, `np.clip(||right - left|| / 0.1, 0,
#: 1)` (`metaworld/sawyer_xyz_env.py`), i.e. [0, 1]. T2R's prompt is wrong
#: about its own benchmark, and BIRD reproduces the error because reproducing
#: the prompt is the point of the member. Recorded here -- never silently fixed.
_SYMBOLS_T2R_METAWORLD: Dict[str, str] = {
    # SIX OF SEVEN ARE BYTE-IDENTICAL to `metaworld_exp.py:19-27`.
    "self.robot.ee_position": "obs[:3]",
    "self.robot.gripper_openness": "obs[3]",
    "self.obj1.position": "obs[4:7]",
    "self.obj1.quaternion": "obs[7:11]",
    "self.obj2.position": "obs[11:14]",
    "self.obj2.quaternion": "obs[14:18]",
    # The seventh cannot be: upstream is `self.env._get_pos_goal()`, a live call
    # on the wrapped env, and a BIRD reward program is a free function with no
    # env to call (`CompiledReward._make_binder` passes None for a `self`
    # parameter, deliberately). `bird/envs/metaworld.py`'s state table documents
    # these three slots as that method's value for the episode.
    "self.goal_position": "obs[36:39]",
}


@register("symbol_table", "t2r_metaworld_global")
def symbol_table_t2r_metaworld_global(ctx: Context) -> Dict[str, str]:
    """Text2Reward's single Meta-World table, `metaworld_exp.py:19-27`, with
    upstream's own `obs[...]` right-hand sides. The companion of `env_spec:
    t2r_class_abstraction`; see `_SYMBOLS_T2R_METAWORLD` for the verbatim
    left-hand sides, the substituted right-hand sides and the one upstream
    error kept on purpose.

    It does NOT read `ctx.env`, deliberately: the table is a property of T2R's
    prompt and Meta-World's observation layout, both fixed, and an env that
    resolved it would make the pin depend on which adapter happened to be
    loaded."""
    return dict(_SYMBOLS_T2R_METAWORLD)


#: WHICH SYMBOLS an `env_spec` member shows the model that a reward program
#: cannot otherwise name, keyed by member. Read by `config._check_coherence` to
#: check that the resolved `symbol_mapping` COVERS them -- not merely that a
#: table exists.
#:
#: SEVEN, not eight. The abstraction has eight attribute LINES, but `self.robot`
#: and `self.obj1`/`self.obj2` are class-typed handles rather than readable
#: quantities: a reward reads `self.obj1.position`, never `self.obj1`. Seven is
#: the number in T2R's own table (`metaworld_exp.py:19-27`), in
#: `_SYMBOLS_T2R_METAWORLD`, and in every message that means symbols.
#:
#: Empty for every other member, and that is a statement rather than a default:
#: `full_source`, `state_action_api_stub` and `pythonic_class_abstraction`
#: render names the ADAPTER defines, which a program can already read off `s`
#: and `a`, so there is nothing for a table to cover.
ENV_SPEC_REQUIRED_SYMBOLS: Dict[str, Tuple[str, ...]] = {
    "t2r_class_abstraction": tuple(_SYMBOLS_T2R_METAWORLD),
}


#: The KEYS each registered `symbol_table` member rewrites, for a caller that
#: needs them WITHOUT a `ctx` -- which is `config._check_coherence`, running at
#: load with no env and no run. `per_task` is absent on purpose: its keys are the
#: task spec's, so they are per-run rather than per-member, and `config.py`
#: resolves that one through `bird/tasks.py` instead. Held beside the members so
#: a third table cannot be added without a row here; the coverage rule reads it,
#: and a member missing from it is treated as covering nothing.
_SYMBOL_TABLE_KEYS: Dict[str, Tuple[str, ...]] = {
    "t2r_metaworld_global": tuple(_SYMBOLS_T2R_METAWORLD),
}

#: The VALUES' side of the same tables, read by `_check_coherence` to learn which
#: variable a rewritten program will index -- `obs` for `t2r_metaworld_global`.
#: A table is only usable if that name is a parameter of the signature
#: `generate.output.signature` pins, and this is how the rule finds out without
#: importing the table itself.
_SYMBOL_TABLE_VALUES: Dict[str, Tuple[str, ...]] = {
    "t2r_metaworld_global": tuple(_SYMBOLS_T2R_METAWORLD.values()),
}


def _symbol_mapping(ctx: Context) -> Dict[str, str]:
    """`generate.postprocess.symbol_mapping`: an inline dict, or the name of a
    `symbol_table` registry member (Text2Reward's general->specific converter;
    CARD inherits it)."""
    spec = ctx.cfg.get("generate.postprocess.symbol_mapping")
    if isinstance(spec, dict):
        return {str(k): str(v) for k, v in spec.items()}
    if isinstance(spec, str) and spec.strip():
        return _get("symbol_table", spec)(ctx)
    return {}


def postprocess_code(ctx: Context, code: str) -> Tuple[str, Dict[str, Any]]:
    """`generate.postprocess.*` applied to a program's text, plus what it did.

    ONE TRUTH FOR TWO CALLERS: `_build_candidate` and the REPAIR path
    (`verification._regenerate`). A repair that built its candidate straight
    from the extractor and applied nothing would, under a non-empty table,
    hand the harness every repaired program with the prompt's own symbols
    unrewritten; each dies on `self.<symbol>`, and the repair loop cannot
    converge however many attempts it is given (`card` on Meta-World: the full
    `verify.max_repair_attempts` (10) per candidate, every attempt failing
    `AttributeError: 'MetaWorld' object has no attribute 'robot'`, with the
    table's exact keys still present in the persisted program).

    Inheriting `meta` would compound it: the artifact would record
    `symbol_mapping_applied: 7` against a program the mapping never touched.
    Returning the meta alongside the code makes that impossible to get wrong:
    a caller cannot record the claim without having made the call.
    """
    meta: Dict[str, Any] = {}
    mapping = _symbol_mapping(ctx)
    if mapping:
        code = _apply_symbol_mapping(mapping, code)
        meta["symbol_mapping_applied"] = len(mapping)
    return code, meta


def _apply_symbol_mapping(mapping: Dict[str, str], text: str) -> str:
    """Longest key first, so a general symbol that is a prefix of a longer one
    cannot rewrite it out from under the longer match (T2R's rule).

    ON WORD BOUNDARIES.

    A substring replace over a whole program does not distinguish an identifier
    from a run of letters inside another word. With a table mapping
    `"z" -> "s[1]"`, a bare `str.replace` rewrites `zero` to `s[1]ero` and
    `dead zone` to `dead s[1]one`, and `np.zeros` becomes `np.s[1]eros` -- a
    SyntaxError, i.e. a candidate destroyed by the harness and recorded as the
    generator's failure; `"x" -> "s[0]"` would do the same to `max` or `np.exp`.

    A lambda supplies the replacement rather than a template, because the values
    are index expressions like `s[9]` and `\\g<0>`/backslash sequences in a
    regex template would be interpreted rather than inserted.

    THIS DOES NOT FIX THE OTHER HALF, and the two are worth keeping separate. A
    boundary-correct rename still rewrites an ASSIGNMENT TARGET: with
    `"pitch" -> "s[2]"`, `pitch = ns[2]` becomes `s[2] = ns[2]`, which writes
    into the state array the reward was handed. Fixing that needs an AST
    transform that rewrites Load contexts only, which this function does not
    attempt.
    """
    if not mapping or not text:
        return text
    for key in sorted(mapping, key=len, reverse=True):
        value = mapping[key]
        text = re.sub(rf"(?<!\w){re.escape(key)}(?!\w)", lambda _m, v=value: v, text)
    return text


_SEARCH_REPLACE = re.compile(
    r"<{5,}\s*SEARCH\s*\n(.*?)\n={5,}\s*\n(.*?)\n>{5,}\s*REPLACE", re.DOTALL)


def _apply_edit_mode(ctx: Context, code: str, raw: str, parent_code: str) -> str:
    """`generate.output.edit_mode`. `full_rewrite` in every published method
    (§1); GT constrains the rewrite to a 4-op grammar in prose, which is a prompt
    nuance rather than a separate key. `diff` and `weights_only` are the
    unpublished members, implemented so the axis is real."""
    mode = ctx.cfg["generate.output.edit_mode"]
    if mode == "full_rewrite" or not parent_code:
        return code
    if mode == "weights_only":
        # The code is inherited verbatim; only `Candidate.weights` changes.
        return parent_code
    if mode == "diff":
        blocks = _SEARCH_REPLACE.findall(raw)
        if not blocks:
            return code  # the model rewrote anyway; take what it gave
        patched = parent_code
        for search, replace in blocks:
            if search in patched:
                patched = patched.replace(search, replace, 1)
        return patched
    return code


def _subtask_index(component_names: Sequence[str], subtasks: Sequence[str]) -> Dict[str, int]:
    """Best-effort component -> subtask attribution, for §4's per-subtask credit.

    Lexical, and therefore approximate; §4 should treat a missing entry as "no
    attribution" rather than as subtask 0."""
    out: Dict[str, int] = {}
    for name in component_names:
        scores = [(_overlap(name.replace("_", " "), task), i)
                  for i, task in enumerate(subtasks)]
        best_score, best_i = max(scores) if scores else (0.0, -1)
        if best_score > 0:
            out[name] = best_i
    return out


# ==========================================================================
# Candidate construction (parse -> retry -> Candidate)
# ==========================================================================


def _build_candidate(ctx: Context, state: RunState, backend: Callable[..., List[str]],
                     messages: List[Dict[str, str]], *, raw: Optional[str],
                     parent_id: Optional[str], parent_code: str,
                     meta: Dict[str, Any], nl_spec: str = "",
                     empty_reason: Optional[str] = None,
                     parents: Sequence[CandidateReport] = ()) -> Candidate:
    """One raw response -> one Candidate, honouring `generate.parse.max_retries`.

    `parents` are the reports the prompt was built from. When the first one
    came out of the MAP-Elites archive it carries `meta["archive_island"]`, the
    deme it was SAMPLED from, and the offspring is stamped `parent_island`
    with it so §6's `_island_for` files the child in that deme -- for a migrant
    the destination, not the source (`update._island_for`).

    `empty_reason` is why a SUPPLIED `raw` is empty, from the client's
    index-aligned `last_empty_reasons` (see `LLMClient.__call__`); a caller that
    batched several samples passes the one for this slot. Left `None`, a raw
    fetched here (the parse-retry loop, one sample per call) or a single-sample
    caller reads slot 0 of the client's last call, which is the sample it got.

    A sample that never parses is returned as a FAILED candidate, never dropped:
    Eureka's execute-rate accounting and its -10000 sentinel both depend
    on the failure occupying its slot in the pool. Silently returning K-1
    candidates would inflate the execute rate of every method that has one.
    """
    cfg = ctx.cfg
    out_fmt = _get("output_format", cfg["generate.output.format"])
    retries_left = _max_retries(ctx)
    attempts = 0
    code, pattern = (None, "")

    while True:
        if raw is None:
            batch = backend(ctx, state, messages, 1)
            raw = batch[0] if batch else ""
            empty_reason = None            # this sample is slot 0 of the call just made
        code, pattern = _extract_code(ctx, raw)
        attempts += 1
        if code is not None or retries_left <= 0:
            break
        retries_left -= 1
        raw = None

    cand_id = ctx.next_id("c")
    meta = dict(meta)
    meta.update({"parse_attempts": attempts, "parse_pattern": pattern,
                 "history_modes": _history_modes_used(ctx, backend)})
    if parents:
        sampled_island = parents[0].meta.get("archive_island")
        if sampled_island is not None:
            meta.setdefault("parent_island", int(sampled_island))
    if not nl_spec and cfg["generate.output.design_thought"] == "inline_brace":
        # Read off the FINAL raw, after the parse-retry loop, so the thought
        # describes the program that was kept. The release takes the first
        # `{...}` of the whole response (rfagent_algo.py:426) -- which is a dict
        # literal out of the code when the model skipped the brace -- so the
        # outcome is recorded as a fact about the response rather than trusted.
        nl_spec = _parse_design_thought(raw or "")
        meta["design_thought_parsed"] = bool(nl_spec)

    if code is None:
        # Slot forfeited (Eureka's `max_retries: 0`), but recorded -- and
        # recorded for what it was. An EMPTY response is not a model that wrote
        # no code block: the provider returned nothing, and the client knows why
        # (refused, or cut at `max_tokens`). Recording both as "no code block
        # matched" would make a run whose iterations were refused wholesale read
        # as a model writing unparseable programs.
        failure = ("no code block matched generate.parse.patterns "
                   f"after {attempts} attempt(s)")
        if not (raw or "").strip():
            reason = empty_reason
            if reason is None:
                reasons = list(getattr(getattr(ctx, "generator", None),
                                       "last_empty_reasons", None) or [])
                reason = reasons[0] if reasons else ""
            failure = ("provider returned an empty sample"
                       + (f" ({reason})" if reason else "")
                       + f"; no code to parse after {attempts} attempt(s)")
            meta["empty_response"] = True
        return Candidate(
            cand_id=cand_id, iteration=state.iteration, reward_code="",
            parent_id=parent_id, raw_response=raw or "", nl_spec=nl_spec,
            prompt_messages=messages, meta=meta,
            valid=False, failure_kind="invalid", failure=failure)

    code = _apply_edit_mode(ctx, code, raw or "", parent_code)
    code, _post_meta = postprocess_code(ctx, code)
    meta.update(_post_meta)

    candidate = Candidate(
        cand_id=cand_id, iteration=state.iteration, reward_code=code,
        parent_id=parent_id, raw_response=raw or "", nl_spec=nl_spec,
        prompt_messages=messages, meta=meta)

    if getattr(out_fmt, "wants_components", False):
        candidate.component_names = _parse_component_names(code)
        if state.subtasks and cfg["generate.decomposition.reward_must_map_to_subtasks"]:
            candidate.subtask_index = _subtask_index(candidate.component_names,
                                                     state.subtasks)
    # `weights_only` makes the weights dict the entire output, whatever the
    # format asked for -- otherwise that edit mode would silently be a no-op.
    if getattr(out_fmt, "wants_weights", False) or \
            cfg["generate.output.edit_mode"] == "weights_only":
        candidate.weights = _parse_weights(raw or "", code)
        if not candidate.weights:
            meta["weights_missing"] = True

    return _apply_co_design(ctx, candidate)


def _apply_co_design(ctx: Context, candidate: Candidate) -> Candidate:
    """LIMEN's observation function and DrEureka's DR ranges (§1 co_design).

    The reward code is left intact when `get_observation` is lifted out: the two
    are co-designed and the learner needs both, so `observation_code` is a view
    onto the program, not a partition of it.
    """
    cfg = ctx.cfg
    if cfg["generate.co_design.observation_fn"]:
        obs_code, obs_dim = _extract_observation(candidate.reward_code)
        if obs_code is None:
            return candidate.failed(
                "co_design.observation_fn: response defines no get_observation")
        # Through the same helper the reward code goes through, so the two
        # texts of one candidate cannot be postprocessed differently; passing
        # the mapping in as an argument would be one more place a future
        # postprocess step could be forgotten.
        candidate.observation_code, _ = postprocess_code(ctx, obs_code)
        cap = int(cfg["generate.co_design.observation_max_dim"])
        if obs_dim is None:
            candidate.meta["obs_dim_unknown"] = True
        else:
            candidate.meta["obs_dim"] = obs_dim
            if obs_dim > cap:
                return candidate.failed(
                    f"co_design.observation_max_dim: {obs_dim} features > cap {cap}")
    if cfg["generate.co_design.dr_config"]:
        candidate.dr_config = _parse_dr_config(candidate.raw_response,
                                               candidate.reward_code)
        if candidate.dr_config is None:
            candidate.meta["dr_config_missing"] = True
    return candidate


# --- two-stage NL -> code -------------------------------------------------

_SPEC_TAIL = (
    "## OUTPUT CONTRACT\n"
    "Write the reward as a specification in PROSE. Name each term, say what "
    "behaviour it is there to produce, say roughly how strong it should be "
    "relative to the others, and say which degenerate policy it rules out. "
    "Write NO code in this turn."
)


def _two_stage_sample(ctx: Context, state: RunState, backend: Callable[..., List[str]],
                      parents: Sequence[CandidateReport],
                      persona: Optional[Tuple[str, str]],
                      extra: Optional[Dict[str, str]],
                      prose: Optional[str] = None,
                      prose_reason: str = "") -> Tuple[str, str, List[Dict[str, str]], str]:
    """GT: a prose reward spec first, then code written from that spec (§1).

    Two LLM turns per candidate -- that cost is the method's, not an
    implementation artefact. CARD's chain-of-thought is the contrasting design:
    inline reasoning in ONE response. `prose` may be supplied when stage 1 was
    batched across candidates, with `prose_reason` saying why it is empty if
    it is (the client's index-aligned `last_empty_reasons` entry).

    The two turns are also the two calls `generate.stage_history_modes` indexes,
    so each is built under its own index: the prose turn is L2R's Thinker
    (index 0), the code turn its Coder (index 1). With the default `[]` both
    resolve to `generate.history_mode`.

    Returns `(raw, prose, request, empty_reason)`. AN EMPTY SPEC ENDS THE SAMPLE
    HERE: `raw` is "" and `empty_reason` names the thinker stage and the
    client's reason, so `_build_candidate` records a forfeited slot
    (`empty_response`) the way single-stage methods do. An empty prose -- a
    refused or truncated thinker reply, or a short return padded to "" --
    must not be appended as `{"role": "assistant", "content": ""}` and sent to
    the coder: the Messages API rejects an empty non-final turn with a 400 that
    `_degrade` cannot match to any pin, so one refused thinker sample among
    GT's ten would kill the whole search mid-iteration. The request handed
    back for the artifact is the thinker's (the one that produced the empty
    reply); with `generate.parse.max_retries > 0` a parse retry re-asks it.
    """
    if prose is None:
        spec_messages = _build_messages(ctx, state, parents, persona=persona,
                                        extra=extra, tail=_SPEC_TAIL,
                                        call_index=_CALL_THINKER)
        got = backend(ctx, state, spec_messages, 1)
        prose = got[0].strip() if got else ""
        if not prose:
            reasons = list(getattr(getattr(ctx, "generator", None),
                                   "last_empty_reasons", None) or [])
            prose_reason = reasons[0] if reasons else ""
    else:
        spec_messages = None
    if not prose:
        request = spec_messages if spec_messages is not None else _build_messages(
            ctx, state, parents, persona=persona, extra=extra, tail=_SPEC_TAIL,
            call_index=_CALL_THINKER)
        reason = ("thinker stage returned nothing"
                  + (f" ({prose_reason})" if prose_reason else ""))
        return "", "", request, reason
    messages = _build_messages(ctx, state, parents, persona=persona, extra=extra,
                               call_index=_CALL_CODER)
    messages.append({"role": "assistant", "content": prose})
    messages.append({"role": "user", "content":
                     "Now implement exactly that specification, changing nothing "
                     "about it.\n\n" + _fmt_section("OUTPUT CONTRACT", _get(
                         "output_format", ctx.cfg["generate.output.format"])(ctx))})
    got = backend(ctx, state, messages, 1)
    return (got[0] if got else ""), prose, messages, ""


def _two_stage_meta(raw: Optional[str], thinker_reason: str) -> Dict[str, Any]:
    """Which of the two turns came back empty, for the artifact (`meta`)."""
    if thinker_reason:
        return {"empty_stage": "thinker"}
    if not (raw or "").strip():
        return {"empty_stage": "coder"}
    return {}


# ==========================================================================
# sampling_mode  --  sampler(ctx, state, backend) -> list[Candidate]
# ==========================================================================


def _parents_for(ctx: Context, state: RunState) -> List[CandidateReport]:
    return list(_get("parent_source", ctx.cfg["generate.parent_source"])(ctx, state) or [])


def _parent_ids(ctx: Context, parents: Sequence[CandidateReport]) -> Tuple[Optional[str], str]:
    """`(parent_id, parent_code)` for `_build_candidate`. The code is the
    INHERITABLE program (`_inheritable_program`): it is what `_apply_edit_mode`
    patches or copies under `diff` / `weights_only`, so a parent's observation
    can ride into a child's `reward_code` -- and be lifted back out by
    `_apply_co_design` -- only when `update.co_evolve.observation_fn` says the
    observation persists."""
    if not parents:
        return None, ""
    return parents[0].cand_id, _inheritable_program(
        ctx, parents[0].candidate.reward_code or "")


def _two_stage_on(ctx: Context, backend: Callable[..., List[str]]) -> bool:
    """Two-stage needs a prompt to answer; the enumeration backend has none."""
    return bool(ctx.cfg["generate.output.two_stage_nl_then_code"]) and \
        getattr(backend, "needs_prompt", True)


@register("sampling_mode", "iid_parallel")
def sample_iid_parallel(ctx: Context, state: RunState,
                        backend: Callable[..., List[str]]) -> List[Candidate]:
    """Eureka/DrEureka/RDA/GT: `n_candidates` i.i.d. samples from ONE prompt.

    This sampler asks the backend for the full remainder; how the samples are
    actually fetched belongs to `LLMClient.__call__`, which reads the same
    keys: a serial provider batches `generate.sampling.chunk_size` per
    `_complete` call (Eureka: 4), while a provider that fans out
    (`concurrent_samples` + `llm.max_concurrent_requests` > 1) takes the whole
    remainder in one call so the round-trips can overlap.

    i.i.d. sampling is what makes at least one executable candidate near-certain
    -- Eureka's central argument for K=16, and the thing CARD cites and rejects
    on cost (§1). RDA's ablation puts numbers on it: 1 -> 8 candidates lifts
    success 0.20 -> 1.00, so this is a first-class scaling knob, not a detail.

    Under `two_stage_nl_then_code` the prose turn is still batched by chunk size
    (the prompt is identical across samples) while the code turn is necessarily
    per-candidate, so the cost is between 1x and 2x calls per candidate.
    """
    cfg = ctx.cfg
    n = _n_llm_this_iteration(ctx, state)
    parents = _parents_for(ctx, state)
    parent_id, parent_code = _parent_ids(ctx, parents)
    needs_prompt = getattr(backend, "needs_prompt", True)

    # No call_index: this is the one-call prompt. Under two-stage it goes unused
    # -- that branch builds both turns itself, each under its own index.
    messages = _build_messages(ctx, state, parents) if needs_prompt else []

    if _two_stage_on(ctx, backend):
        spec_messages = _build_messages(ctx, state, parents, tail=_SPEC_TAIL,
                                        call_index=_CALL_THINKER)
        proses: List[str] = []
        # Why each thinker slot is empty, if it is: the client's index-aligned
        # reasons for each call, taken before the coder calls overwrite them.
        why_empty: List[str] = []
        while len(proses) < n:
            got = backend(ctx, state, spec_messages, n - len(proses))
            if not got:
                break
            proses.extend(g.strip() for g in got)
            reasons = list(getattr(getattr(ctx, "generator", None),
                                   "last_empty_reasons", None) or [])
            why_empty.extend((reasons + [""] * len(got))[:len(got)])
        short = n - len(proses)
        proses = (proses + [""] * n)[:n]
        why_empty = (why_empty + [f"provider returned {n - short}/{n} samples"] * n)[:n]
        out: List[Candidate] = []
        for i, prose in enumerate(proses):
            raw, prose, msgs, empty_reason = _two_stage_sample(
                ctx, state, backend, parents, None, None,
                prose=prose, prose_reason=why_empty[i])
            out.append(_build_candidate(
                ctx, state, backend, msgs, raw=raw, parent_id=parent_id,
                parent_code=parent_code, nl_spec=prose,
                empty_reason=empty_reason or None,
                meta={"sampling_mode": "iid_parallel", "sample_index": i,
                      "two_stage": True, **_two_stage_meta(raw, empty_reason)},
                parents=parents))
        return out

    raws: List[Optional[str]] = []
    while len(raws) < n:
        got = backend(ctx, state, messages, n - len(raws))
        if not got:
            break
        raws.extend(got)
    if needs_prompt:
        # A prompted backend that came up short still owes the pool its slots:
        # each `None` is retried once and then recorded as a failure, because
        # execute rate is (executable / requested), not (executable / returned).
        # An enumeration backend owes nothing -- its grid is simply smaller.
        raws = (raws + [None] * n)[:n]

    # `raws` is the concatenation of the backend's calls in order, and so is
    # the client's `last_empty_reasons` for its LAST call -- which covered the
    # whole remainder unless the provider came back short. Align by index and
    # give up (no reason, never a wrong one) past the end.
    reasons = list(getattr(getattr(ctx, "generator", None), "last_empty_reasons", None) or [])
    offset = len(raws) - len(reasons)
    return [_build_candidate(ctx, state, backend, messages, raw=raw,
                             parent_id=parent_id, parent_code=parent_code,
                             meta={"sampling_mode": "iid_parallel", "sample_index": i},
                             empty_reason=(reasons[i - offset] if 0 <= i - offset < len(reasons) else ""),
                             parents=parents)
            for i, raw in enumerate(raws)]


@register("sampling_mode", "sequential_conditioned")
def sample_sequential_conditioned(ctx: Context, state: RunState,
                                  backend: Callable[..., List[str]]) -> List[Candidate]:
    """CARD's chain: each sample is conditioned on the previous one (§1).

    Sequential by construction, so it cannot be batched and `chunk_size` does not
    apply. Note the provenance: sample i's parent is sample i-1 *whatever
    happened to it* -- a failed sample still fathers the next, which is the same
    rule the chain head follows across iterations (`parent_source: latest`).
    """
    n = _n_llm_this_iteration(ctx, state)
    parents = _parents_for(ctx, state)
    parent_id, parent_code = _parent_ids(ctx, parents)
    out: List[Candidate] = []
    previous: Optional[Candidate] = None

    for i in range(n):
        extra = None
        if previous is not None:
            extra = {"PREVIOUS PROPOSAL":
                     "You already proposed the reward below in this chain. Produce "
                     "the next one: keep what is justified, change what is not, and "
                     "say nothing about the change outside the code.\n"
                     "```python\n" + (previous.reward_code or "(unparseable)") + "\n```"}
        stage_meta: Dict[str, Any] = {}
        empty_reason = ""
        if _two_stage_on(ctx, backend):
            raw, prose, messages, empty_reason = _two_stage_sample(
                ctx, state, backend, parents, None, extra)
            stage_meta = _two_stage_meta(raw, empty_reason)
        else:
            messages = _build_messages(ctx, state, parents, extra=extra)
            got = backend(ctx, state, messages, 1)
            raw, prose = (got[0] if got else ""), ""
        candidate = _build_candidate(
            ctx, state, backend, messages, raw=raw, empty_reason=empty_reason or None,
            parent_id=(previous.cand_id if previous is not None else parent_id),
            parent_code=(previous.reward_code if previous is not None else parent_code),
            nl_spec=prose,
            meta={"sampling_mode": "sequential_conditioned", "chain_position": i,
                  **stage_meta},
            parents=parents)
        out.append(candidate)
        previous = candidate
    return out


@register("sampling_mode", "distinct_prompts")
def sample_distinct_prompts(ctx: Context, state: RunState,
                            backend: Callable[..., List[str]]) -> List[Candidate]:
    """`n_candidates` prompts that differ by an explicit diversity instruction.

    Untried in the literature (an unexplored diversity mechanism, like
    personas): every published multi-candidate method gets its spread
    from decode temperature alone. Axes are assigned by index so the pool covers
    them deterministically; `shuffle_sections` still applies within each prompt.
    """
    n = _n_llm_this_iteration(ctx, state)
    parents = _parents_for(ctx, state)
    parent_id, parent_code = _parent_ids(ctx, parents)
    out: List[Candidate] = []
    for i in range(n):
        axis = _DIVERSITY_AXES[i % len(_DIVERSITY_AXES)]
        extra = {"DIVERSITY REQUIREMENT":
                 f"This is variant {i + 1} of {n}, and the variants are compared "
                 f"against each other. Yours must {axis}."}
        stage_meta: Dict[str, Any] = {}
        empty_reason = ""
        if _two_stage_on(ctx, backend):
            raw, prose, messages, empty_reason = _two_stage_sample(
                ctx, state, backend, parents, None, extra)
            stage_meta = _two_stage_meta(raw, empty_reason)
        else:
            messages = _build_messages(ctx, state, parents, extra=extra)
            got = backend(ctx, state, messages, 1)
            raw, prose = (got[0] if got else ""), ""
        out.append(_build_candidate(
            ctx, state, backend, messages, raw=raw, parent_id=parent_id,
            parent_code=parent_code, nl_spec=prose, empty_reason=empty_reason or None,
            meta={"sampling_mode": "distinct_prompts", "sample_index": i, **stage_meta,
                  "diversity_axis": axis}, parents=parents))
    return out


@register("sampling_mode", "personas")
def sample_personas(ctx: Context, state: RunState,
                    backend: Callable[..., List[str]]) -> List[Candidate]:
    """`n_candidates` samples under distinct system personas.

    Untried in the literature (§1), and the only sampler that varies the SYSTEM
    message rather than the user turn -- which is why it is worth having as a
    separate value rather than folding it into `distinct_prompts`: the two
    hypotheses about where decode diversity comes from are different.

    The roster rotates with the iteration so a multi-iteration run does not keep
    handing the same persona the same slot; the rotation is deterministic, not
    sampled, so runs stay reproducible.
    """
    n = _n_llm_this_iteration(ctx, state)
    parents = _parents_for(ctx, state)
    parent_id, parent_code = _parent_ids(ctx, parents)
    start = state.iteration % len(_PERSONAS)
    out: List[Candidate] = []
    for i in range(n):
        persona = _PERSONAS[(start + i) % len(_PERSONAS)]
        stage_meta: Dict[str, Any] = {}
        empty_reason = ""
        if _two_stage_on(ctx, backend):
            raw, prose, messages, empty_reason = _two_stage_sample(
                ctx, state, backend, parents, persona, None)
            stage_meta = _two_stage_meta(raw, empty_reason)
        else:
            messages = _build_messages(ctx, state, parents, persona=persona)
            got = backend(ctx, state, messages, 1)
            raw, prose = (got[0] if got else ""), ""
        out.append(_build_candidate(
            ctx, state, backend, messages, raw=raw, parent_id=parent_id,
            parent_code=parent_code, nl_spec=prose, empty_reason=empty_reason or None,
            meta={"sampling_mode": "personas", "sample_index": i,
                  "persona": persona[0], **stage_meta}, parents=parents))
    return out
