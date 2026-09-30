"""The offline LLM simulator. Registers `llm:mock`.

This is the component that makes the repo runnable and testable without a
network, so its honesty is the honesty of every offline test. It is used by
the `tester` profile (`configs/_profiles/tester.yaml`) and by the end-to-end
tests that run whole methods.

What it simulates, and how
-------------------------

**Programs that vary in quality, on purpose.** Real reward search only works
because some samples are better than others; a mock that emits one canned
function would make every end-to-end test a test of nothing. So the mock draws
from five archetypes ordered by how well they actually solve a
reach-and-stay task (`TIERS`, worst first):

  0 `action_only`      ignores the goal entirely -- pure control/velocity cost
  1 `control_dominated` right idea, weights absurd; optimal policy is to freeze
  2 `proximity_velocity` plausible but MISALIGNED: rewards being near the goal
                        *and* moving fast, so the learner overshoots
  3 `goal_shaped`      dense proximity kernel, mild control cost -- works
  4 `goal_shaped_success` dense shaping + a stay-in-tolerance bonus -- works well

These are real programs: they read the state, compute real numbers, and a
learner optimising tier 4 genuinely ends up somewhere different from one
optimising tier 2. Nothing is faked downstream of the code text.

**Bias toward quality when the prompt carries good news.** Every emitted
program carries a machine-readable provenance comment,

    # bird-mock: tier=4 archetype=goal_shaped_success q=0.92

so when a later prompt quotes a parent program (`generate.context.
include_parent_code`) the mock can read the parent's quality straight out of
the prompt and tilt its own tier distribution upward by that amount
(`_quality_bias` -> `_sample_tier`). That is the whole mechanism by which
iterating improves under the mock; it is scale-free (no dependence on the toy
env's fitness units), deterministic, and it means an end-to-end test of "does
Eureka's reflection loop climb?" is testing a real climb rather than noise. A
prompt with no parent gets `bias = 0` and the flat base distribution.

**A controllable fraction of invalid programs.** `INVALID_FRACTION` (module
level, monkeypatchable) of samples are drawn from `INVALID_KINDS`: a syntax
error, a wrong signature, a NaN-producing body, and a constant reward -- one
per validity check family in §2, so `verify.static_checks`,
`verify.dynamic_checks` and every `verify.on_failure` routing branch get
exercised. The rate falls with prompt quality and halves when the prompt looks
like a repair prompt (contains a traceback), because a model told exactly what
broke really does break less often.

**Non-program answers.** Subtask decompositions, JSON scores, preference labels
and free-text feedback are routed by the call's `tag` (and, failing that, by
prompt content) and returned in well-formed shapes. Preference labels are
derived from a hash of the two candidate ids plus each side's quality signal,
with a fixed hash-driven noise rate -- never constant, so Bradley-Terry in §4
has real structure to fit rather than a degenerate likelihood.

**Determinism.** Every sample's randomness comes from a seed mixing the shared
`ctx.rng` stream with a stable hash of (model, role, tag, prompt, sample
index). Same config seed, same outputs; different prompts, different outputs.
`hashlib` is used rather than `hash()` because Python randomises string hashing
per process, which would silently break run-to-run reproducibility.

One deliberate unrealism: every emitted program opens with a tolerant field
accessor (`_f`) instead of writing `state.tip_pos` directly. A real LLM sees
the env source and names fields exactly; the mock cannot, so the accessor tries
aliases and then positional indices. It keeps the same synthetic program
executable against whatever shape the toy env's state happens to be, which is
the point of a fixture.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..registry import register
from .base import LLMClient, Message, estimate_tokens, messages_text

# --------------------------------------------------------------------------
# knobs the tests monkeypatch
# --------------------------------------------------------------------------

#: Fraction of generated programs that are invalid on purpose. Set to 0.0 to
#: make a test deterministic in "everything compiles" mode, or to 1.0 to drive
#: `loop.on_total_failure`.
INVALID_FRACTION = 0.15

#: One per validity-check family of the verify stage, so a config that turns
#: on `verify.dynamic_checks` sees each of them fire.
INVALID_KINDS = ("syntax_error", "wrong_signature", "nan_values", "constant_reward")

#: How hard a good parent tilts the tier distribution. 0 = ignore the parent.
QUALITY_TILT = 1.6

#: Probability that a preference label disagrees with the quality ordering.
#: Non-zero so Bradley-Terry fits a real (noisy) model rather than a separable
#: one; fixed and hash-driven so it stays deterministic.
PREFERENCE_NOISE = 0.18

MARKER_RE = re.compile(r"bird-mock:\s*tier=(-?\d+)\s+archetype=(\w+)\s+q=([0-9.]+)")
_ID_RE = re.compile(r"\b([a-z]{1,4}\d{3,5})\b")
_NUM_RE = re.compile(r"-?\d+\.\d+|-?\d+")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+(.{3,80})$", re.MULTILINE)


# --------------------------------------------------------------------------
# deterministic hashing
# --------------------------------------------------------------------------


def _hash_int(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "big")


def _hash01(*parts: Any) -> float:
    return (_hash_int(*parts) % 1_000_003) / 1_000_003.0


# --------------------------------------------------------------------------
# the archetypes
# --------------------------------------------------------------------------
#
# Each tier supplies:
#   terms(r)  -> [(component_name, coefficient, unweighted_expression), ...]
#   params(r) -> the same behaviour expressed as template slots, for
#                `generate.output.format: template_params` (L2R's point: the
#                API is fixed, only the numbers are written by the model)
# Coefficients are jittered per sample so a pool of K candidates is not K
# identical programs -- otherwise `verify.dedup` would collapse every pool and
# `select.tie_break` would never be exercised.


def _round(r: random.Random, lo: float, hi: float, digits: int = 3) -> float:
    return round(r.uniform(lo, hi), digits)


def _terms_action_only(r: random.Random) -> List[Tuple[str, float, str]]:
    return [
        ("ctrl", -_round(r, 0.05, 0.4), "effort"),
        ("smoothness", -_round(r, 0.05, 0.5), "speed"),
    ]


def _terms_control_dominated(r: random.Random) -> List[Tuple[str, float, str]]:
    return [
        ("reach", -_round(r, 0.001, 0.02, 4), "dist"),
        ("ctrl", -_round(r, 20.0, 120.0, 1), "effort"),
        ("stillness", -_round(r, 5.0, 40.0, 1), "speed"),
    ]


def _terms_proximity_velocity(r: random.Random) -> List[Tuple[str, float, str]]:
    k = _round(r, 0.5, 3.0)
    return [
        ("proximity", _round(r, 0.8, 2.0), f"1.0 / (1.0 + {k} * dist)"),
        ("momentum", _round(r, 0.5, 2.5), "speed"),
    ]


def _terms_goal_shaped(r: random.Random) -> List[Tuple[str, float, str]]:
    k = _round(r, 1.0, 4.0)
    return [
        ("proximity", _round(r, 1.0, 2.5), f"1.0 / (1.0 + {k} * dist)"),
        ("ctrl", -_round(r, 0.005, 0.05, 4), "effort"),
    ]


def _terms_goal_shaped_success(r: random.Random) -> List[Tuple[str, float, str]]:
    tol = _round(r, 0.03, 0.12)
    return [
        ("reach", -_round(r, 0.8, 1.6), "dist"),
        ("stay", _round(r, 1.5, 4.0), f"(1.0 if dist < {tol} else 0.0)"),
        ("ctrl", -_round(r, 0.005, 0.05, 4), "effort"),
    ]


def _params(reach: float, bonus: float, ctrl: float, tol: float) -> Dict[str, float]:
    return {"reach_weight": reach, "success_bonus": bonus,
            "control_penalty": ctrl, "success_tolerance": tol}


TIERS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "action_only",
        "q": 0.08,
        "doc": "Penalise control effort and jerky motion.",
        "terms": _terms_action_only,
        "params": lambda r: _params(0.0, 0.0, _round(r, 0.5, 2.0), 0.05),
    },
    {
        "name": "control_dominated",
        "q": 0.25,
        "doc": "Reach the goal while strongly discouraging motion.",
        "terms": _terms_control_dominated,
        "params": lambda r: _params(_round(r, 0.001, 0.02, 4), 0.0, _round(r, 20.0, 120.0, 1), 0.05),
    },
    {
        "name": "proximity_velocity",
        "q": 0.45,
        "doc": "Reward being close to the goal and moving briskly toward it.",
        "terms": _terms_proximity_velocity,
        "params": lambda r: _params(_round(r, 0.15, 0.4), 0.0, -_round(r, 0.2, 0.8), 0.05),
    },
    {
        "name": "goal_shaped",
        "q": 0.75,
        "doc": "Dense proximity shaping with a small control cost.",
        "terms": _terms_goal_shaped,
        "params": lambda r: _params(_round(r, 0.8, 1.4), 0.0, _round(r, 0.005, 0.05, 4), _round(r, 0.03, 0.12)),
    },
    {
        "name": "goal_shaped_success",
        "q": 0.92,
        "doc": "Dense shaping toward the goal plus a bonus for staying inside tolerance.",
        "terms": _terms_goal_shaped_success,
        "params": lambda r: _params(_round(r, 0.9, 1.5), _round(r, 1.5, 4.0),
                                    _round(r, 0.005, 0.05, 4), _round(r, 0.03, 0.12)),
    },
)

#: Flat prior over tiers, tilted upward by `_sample_tier` when the prompt
#: carries a good parent. Mildly bottom-heavy so that iteration 1 of an
#: end-to-end test has somewhere to climb from.
TIER_BASE_WEIGHTS = (0.14, 0.20, 0.26, 0.24, 0.16)


# --------------------------------------------------------------------------
# program text
# --------------------------------------------------------------------------

_PREAMBLE = '''\
def compute_reward(state, action=None, *_extra, **_kw):
    """__DOC__"""
    def _f(names, index=None, default=0.0):
        v = None
        for name in names:
            if isinstance(state, dict):
                if name in state:
                    v = state[name]
                    break
            else:
                got = getattr(state, name, None)
                if got is not None:
                    v = got
                    break
        if v is None and index is not None and not isinstance(state, dict):
            try:
                v = state[index]
            except Exception:
                v = None
        if v is None:
            return float(default)
        try:
            return float(v)
        except (TypeError, ValueError):
            try:
                return float(sum(float(x) * float(x) for x in v)) ** 0.5
            except Exception:
                return float(default)

    px = _f(("x", "tip_x", "pos_x", "hand_x"), 0)
    py = _f(("y", "tip_y", "pos_y", "hand_y"), 1)
    vx = _f(("vx", "xdot", "vel_x"), 2)
    vy = _f(("vy", "ydot", "vel_y"), 3)
    gx = _f(("goal_x", "target_x"), 4)
    gy = _f(("goal_y", "target_y"), 5)
    dist = _f(("distance", "dist", "dist_to_goal"), None, -1.0)
    if dist < 0.0:
        dist = ((gx - px) ** 2 + (gy - py) ** 2) ** 0.5
    speed = (vx * vx + vy * vy) ** 0.5
    effort = 0.0
    if action is not None:
        try:
            effort = float(sum(float(a) * float(a) for a in action)) ** 0.5
        except TypeError:
            effort = abs(float(action))
'''


#: The same program under `generate.reward_language: jax`:
#: a pure array body over the TOY STATE LAYOUT (`x, y, vx, vy, goal_x, goal_y`),
#: read positionally because a traced state is an array and `_f`'s
#: name-then-index accessor has nothing to try names on. No `float()`, no
#: `if` on a value, no `.item()` -- the three things the tracing clause forbids
#: and `verification._probe_traced` refuses. `jax_toy` is the one env this runs
#: on today, and it IS the toy layout; a second jax env with another layout
#: needs `_state_field_index` applied here as `_observation_program` applies it.
_PREAMBLE_JAX = '''\
def compute_reward(state, action=None, *_extra, **_kw):
    """__DOC__"""
    s = jnp.asarray(state, dtype=jnp.float32).reshape(-1)
    px, py, vx, vy, gx, gy = s[0], s[1], s[2], s[3], s[4], s[5]
    dist = jnp.sqrt((gx - px) ** 2 + (gy - py) ** 2)
    speed = jnp.sqrt(vx * vx + vy * vy)
    if action is not None:
        effort = jnp.sqrt(jnp.sum(jnp.asarray(action, dtype=jnp.float32) ** 2))
    else:
        effort = jnp.float32(0.0)
'''

#: `(1.0 if <cond> else 0.0)` -> `jnp.where(<cond>, 1.0, 0.0)`: the one Python
#: branch on a value the numpy terms contain, rewritten for the traced contract.
_PY_BRANCH_RE = re.compile(r"\(1\.0 if (.+?) else 0\.0\)")


def _for_language(expr: str, language: str) -> str:
    """A reward term for `generate.reward_language`; the identity under numpy."""
    if language != "jax":
        return expr
    return _PY_BRANCH_RE.sub(lambda m: f"jnp.where({m.group(1)}, 1.0, 0.0)", expr)


def _marker(tier_index: int, archetype: str, quality: float) -> str:
    return f"# bird-mock: tier={tier_index} archetype={archetype} q={quality:.2f}"


def _weights_of(code: str) -> dict:
    """The `weights = {...}` the assembled program declares, or {}.

    Re-read from the emitted source rather than passed alongside it: the JSON
    block the prompt asks for must be the weights the code actually applies,
    and deriving both from one place is the only way that stays true.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "weights":
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return {}
                    if isinstance(value, dict):
                        return {str(k): v for k, v in value.items()}
    return {}


def _valid_program(tier_index: int, r: random.Random, fmt: str,
                   language: str = "numpy") -> str:
    """Assemble one executable program in the format `generate.output.format`
    asks for. The archetype fixes the behaviour; the format fixes the shape;
    `language` (`generate.reward_language`) fixes the preamble and the one
    branch-on-a-value idiom (`_for_language`)."""
    tier = TIERS[tier_index]
    head = _marker(tier_index, tier["name"], tier["q"])
    preamble = _PREAMBLE_JAX if language == "jax" else _PREAMBLE
    body = preamble.replace("__DOC__", tier["doc"])

    if fmt == "template_params":
        # L2R: a fixed reward API, filled in. The archetype is expressed only
        # through the parameter values -- which is exactly the claim of the
        # template representation, and exactly its limit.
        p = tier["params"](r)
        lines = [head, "reward_params = {"]
        lines += [f'    "{k}": {v},' for k, v in p.items()]
        lines += ["}", "", "", body.rstrip("\n"), "",
                  "    components = {",
                  '        "reach": -reward_params["reach_weight"] * dist,',
                  '        "stay": reward_params["success_bonus"] * '
                  + _for_language('(1.0 if dist < reward_params["success_tolerance"] else 0.0)',
                                  language) + ",",
                  '        "ctrl": -reward_params["control_penalty"] * effort,',
                  "    }",
                  "    total = sum(components.values())",
                  "    return total, components"]
        return "\n".join(lines)

    terms = [(n, c, _for_language(e, language)) for n, c, e in tier["terms"](r)]

    if fmt == "scalar_only":
        expr = " + ".join(f"({c} * ({e}))" for _, c, e in terms)
        # `float(total)` is the numpy contract's cast and a trace's first failure.
        return "\n".join([head, body.rstrip("\n"), "",
                          f"    total = {expr}",
                          "    return total" if language == "jax" else "    return float(total)"])

    if fmt == "component_dict_plus_weights":
        # RDA/GT: the weights live in their own dict so §6 can rewrite them
        # without touching the component implementations.
        lines = [head, "weights = {"]
        lines += [f'    "{name}": {coef},' for name, coef, _ in terms]
        lines += ["}", "", "", body.rstrip("\n"), "", "    components = {"]
        lines += [f'        "{name}": {expr},' for name, _, expr in terms]
        lines += ["    }",
                  "    total = sum(weights.get(k, 1.0) * v for k, v in components.items())",
                  "    return total, components"]
        return "\n".join(lines)

    # component_dict_return (Eureka / DrEureka / CARD): free-form body that
    # returns (total, named components). The default.
    lines = [head, body.rstrip("\n"), "", "    components = {"]
    lines += [f'        "{name}": {coef} * ({expr}),' for name, coef, expr in terms]
    lines += ["    }",
              "    total = sum(components.values())",
              "    return total, components"]
    return "\n".join(lines)


#: Candidate observation features, in the order `_observation_program` emits
#: them. Every entry reads the raw state through the same `_f` accessor the
#: reward preamble defines, so an observation program is executable against the
#: same toy state a reward is.
#:
#: THE ORDER IS LOAD-BEARING. `components.training._install_observation` makes
#: the observation a real MDP interface: the policy trains on this feature
#: vector and nothing else. `_observation_program` emits a PREFIX of this tuple,
#: so a prefix that omits the goal is an environment in which the task is not
#: merely hard but unobservable -- an order beginning `px, py, vx` would make
#: every draw at width 3-5 exactly that, and on `limen` at the tester profile
#: the cascade screen would reject every candidate (`success 0.0000 < 0.0100`)
#: because of the MOCK, not the method.
#:
#: So the goal-relative features come first and every prefix is learnable, which
#: is also what the prompt asks for in as many words ("choose features that make
#: your reward learnable", `generation._observation_section`). Width still varies
#: 3..10 -- see `_observation_program` on why the archive needs that -- but it
#: varies in REDUNDANCY rather than in whether the task is solvable at all.
_OBS_FEATURES = (
    ("dx", '_f(("goal_x", "target_x"), 4) - _f(("x", "tip_x", "pos_x", "hand_x"), 0)'),
    ("dy", '_f(("goal_y", "target_y"), 5) - _f(("y", "tip_y", "pos_y", "hand_y"), 1)'),
    ("dist", '((_f(("goal_x", "target_x"), 4) - '
             '_f(("x", "tip_x", "pos_x", "hand_x"), 0)) ** 2 + '
             '(_f(("goal_y", "target_y"), 5) - '
             '_f(("y", "tip_y", "pos_y", "hand_y"), 1)) ** 2) ** 0.5'),
    ("vx", '_f(("vx", "xdot", "vel_x"), 2)'),
    ("vy", '_f(("vy", "ydot", "vel_y"), 3)'),
    ("speed", '(_f(("vx", "xdot", "vel_x"), 2) ** 2 + '
              '_f(("vy", "ydot", "vel_y"), 3) ** 2) ** 0.5'),
    ("px", '_f(("x", "tip_x", "pos_x", "hand_x"), 0)'),
    ("py", '_f(("y", "tip_y", "pos_y", "hand_y"), 1)'),
    ("gx", '_f(("goal_x", "target_x"), 4)'),
    ("gy", '_f(("goal_y", "target_y"), 5)'),
)


#: `hand_x: float   # s[0] -- gripper body x` in the API stub, and
#: `hand_x: float          # s[0]: gripper body x` in the class abstraction. One
#: regex covers both because the only part that matters is `# s[<i>]`, which every
#: `EnvAdapter.describe()` rendering that names indices at all emits identically.
_FIELD_INDEX_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*float\b[^#\n]*#\s*s\[(\d+)\]")


def _state_field_index(prompt: str) -> Dict[str, int]:
    """name -> index, read out of the prompt's own state table.

    WHY THE MOCK HAS TO DO THIS AT ALL. `_OBS_FEATURES` and the reward tiers name
    state slots through `_f(names, index)`: the accessor tries the NAMES (for a
    dict or attribute state) and falls back to the INDEX for an ndarray. Every
    `EnvAdapter` in this repo hands out a plain ndarray, so on any env the index
    is what is actually used -- and those indices were written for `toy_reacher`
    (`x, y, vx, vy, goal_x, goal_y`).

    On `mt10_reach-v3`, whose 39-D observation puts the goal at 36:39 and the
    object at 4:7, `_f(("goal_x", "target_x"), 4)` would silently read OBJECT x.
    Because `training._install_observation` makes the observation the policy's
    real input, that is not a cosmetic wrongness -- the co-designed interface
    would hand the learner a feature set with no goal in it, the cascade screen
    would reject every candidate at `success 0.0000 < 0.0100`, and the tester
    profile on the Meta-World env would exercise the archive and the screen while
    never once producing a trainable interface. The failure would look like the
    method's and be the double's.

    The prompt is the right source and not a special case: it is what the real
    model reads, `generate.context.env_spec` decides how much of it there is, and
    a mock that resolves slots from anywhere else would be reading a channel the
    thing it stands in for cannot see.
    """
    out: Dict[str, int] = {}
    for line in (prompt or "").splitlines():
        m = _FIELD_INDEX_RE.match(line)
        if m:
            out.setdefault(m.group(1), int(m.group(2)))
    return out


def _resolve_feature(expr: str, table: Dict[str, int]) -> Optional[str]:
    """Rewrite every `_f((names...), <default>)` in `expr` to this env's indices.

    Returns `None` when a name-tuple resolves to NOTHING, and dropping the whole
    feature is the point: `mt10_*` has no velocity observable at all (the adapter
    says so in as many words -- `prev_hand_x` is provided instead), so
    `_f(("vx", "xdot", "vel_x"), 2)` has no honest answer on it. Emitting the
    fallback index would hand the policy `hand_z` under the name `vx`, which is a
    feature that reads as deliberate and is noise. A shorter feature vector is a
    true statement about what the environment exposes.

    An env whose prompt names none of the slots (`env_spec: none`, or prose only)
    yields an empty table and every feature keeps its default index, which is
    correct for `toy_reacher` and the only safe answer when the prompt says
    nothing.
    """
    if not table:
        return expr
    missing: List[str] = []

    def _sub(m: "re.Match[str]") -> str:
        names = [n.strip().strip("\"'") for n in (m.group(1) or "").split(",") if n.strip()]
        hit = next((table[n] for n in names if n in table), None)
        if hit is None:
            missing.append(names[0] if names else "?")
            return m.group(0)
        return f"_f(({m.group(1)}), {hit})"

    # `re.sub` with a callback, not `findall` + `str.replace`. `findall` on a
    # two-group pattern yields TUPLES, and passing one to `re.search` raises a
    # TypeError that a caller's `except TypeError` compatibility shim would
    # swallow, so the resolver would appear to run and return the UNRESOLVED
    # expression.
    out = _F_CALL_RE.sub(_sub, expr)
    return None if missing else out


#: `_f(("goal_x", "target_x"), 4)` -> the name list and the fallback index.
_F_CALL_RE = re.compile(r"_f\(\(([^)]*)\),\s*(\d+)\)")

#: Just the index, for "which slots does this feature set already read".
_F_INDEX_RE = re.compile(r"_f\(\([^)]*\),\s*(\d+)\)")

#: Below this many resolved named features, top up from the raw field table --
#: see `_observation_program`. Three because `r.randint(3, ...)` is the draw and a
#: feature set narrower than the draw's floor cannot vary at all.
_MIN_OBS_FEATURES = 3
#: A PARENT observation quoted in the prompt: `OBS_DIM = k` as a program line
#: (line start), which only a quoted `get_observation` program carries.
_PARENT_OBS_DIM_RE = re.compile(r"^OBS_DIM\s*=\s*(\d+)\s*$", re.MULTILINE)

#: And no further than this, so the topped-up case keeps a width DRAW rather than
#: becoming a fixed projection of the whole state. Matches `len(_OBS_FEATURES)` in
#: spirit: enough spread for the archive's x-axis to have something to bin.
_MAX_TOPPED_UP = 6


def _observation_program(r: random.Random, cap: int, prompt: str = "") -> str:
    """LIMEN's co-designed `get_observation`, for `generate.co_design.observation_fn`.

    The prompt (`generation._observation_section`) asks for this function in
    the same code block as the reward; a mock that never emitted it would make
    every LIMEN candidate die at `_apply_co_design`'s gate with
    `co_design.observation_fn: response defines no get_observation`, on every
    seed, while the suite stayed green. This closes the contract from the mock
    side; the gate itself is correct.

    **The feature count varies on purpose.** `observation_dim` is one of LIMEN's
    two MAP-Elites descriptors (`update.archive.descriptors`), so a mock that
    always returned the same width would put every candidate in one archive
    column and the tester profile would exercise the *shape* of MAP-Elites without
    ever exercising its diversity pressure. Drawing 3..len(_OBS_FEATURES) gives
    the archive something to bin.

    `OBS_DIM` is declared and always equals the real returned width:
    `_extract_observation` prefers the declaration over its AST estimate, so a
    declaration that disagreed with the body would make the mock lie to the very
    check it exists to feed.
    """
    # RESOLVED AGAINST THIS ENV FIRST, then sliced. Sliced-then-resolved would
    # make the emitted width depend on how many of the first `n` features this
    # env happens to expose, so `OBS_DIM` and the archive's x-axis would move
    # with the environment for a reason no one chose. Resolving first means the
    # draw is over what is actually available.
    table = _state_field_index(prompt)
    usable = [(name, expr) for name, expr in
              ((nm, _resolve_feature(ex, table)) for nm, ex in _OBS_FEATURES)
              if expr is not None]
    if len(usable) < _MIN_OBS_FEATURES and table:
        # TOPPED UP FROM THE ENV'S OWN FIELD NAMES, and only when the named
        # features came up short. `gym_reacher_reach` resolves TWO of the ten --
        # its state names neither `x` nor any alias of it -- and a phi that is the
        # goal position with no end-effector in it is not a hard interface, it is
        # an impossible one. Reaching for the unresolved list instead would put
        # `toy_reacher`'s indices on a different env, which is the bug this whole
        # function exists to fix.
        #
        # Gated on the shortfall rather than applied always, because appending to
        # `usable` moves `r.randint(3, len(usable))` and would change the emitted
        # width on every env -- including `toy_reacher`, where all ten named
        # features already resolve and the output must stay byte-identical.
        used = {i for _n, e in usable for i in _F_INDEX_RE.findall(e)}
        for name, idx in sorted(table.items(), key=lambda kv: kv[1]):
            if str(idx) in used:
                continue
            usable.append((name, f'_f(("{name}",), {idx})'))
            if len(usable) >= _MAX_TOPPED_UP:
                break
    if not usable:
        # ONLY when the prompt named no slots at all -- `env_spec: none`, or a
        # prose-only rendering. Then there is nothing to resolve against and the
        # unresolved list is correct for `toy_reacher` and the only answer
        # available anywhere else.
        #
        # Reaching this with a POPULATED table would be a bug: an env whose stub
        # names `cart_position`/`pole_angle` matches none of `_OBS_FEATURES`'
        # aliases, and falling back here would hand it `toy_reacher`'s indices --
        # reintroducing the exact wrong-slot problem the resolver exists to
        # remove, in the one case where the prompt DID carry enough to build a
        # correct observation. So the top-up above runs whenever the table is
        # non-empty, including from zero resolved features.
        usable = list(_OBS_FEATURES)
    n = min(r.randint(3, len(usable)), max(1, cap)) if len(usable) > 3 else len(usable)
    # CO-EVOLUTION IS STRUCTURAL, NOT A DICE ROLL. When the prompt QUOTES a parent
    # observation (`update.co_evolve.observation_fn: true` puts the parent's whole
    # program, `OBS_DIM = k` included, in PARENT REWARD), a co-evolving child is
    # asked to build on it, and this mock does what a model would: it takes the
    # parent's width and moves it by one, and SAYS SO in the docstring. Were the
    # child's width an independent draw from `r`, the flip between "parent
    # observation shown" and "hidden" would reach the child only when two seeded
    # draws happened to differ -- `test_the_lone_flip_changes_what_the_child_is_
    # shown_and_writes` would pass on a 7-in-8 coincidence and break whenever an
    # unrelated prompt change re-rolled the hash. A quoted `OBS_DIM = k` is matched
    # at LINE START only: the co-design contract's prose mentions the name
    # ("Declare the count ... OBS_DIM") and is not a parent.
    parent = _PARENT_OBS_DIM_RE.search(prompt or "")
    evolved_from = None
    if parent is not None and len(usable) >= 3:
        k = int(parent.group(1))
        evolved_from = k
        grown = k + 1 if k + 1 <= min(len(usable), max(1, cap)) else max(3, k - 1)
        n = min(max(3, grown), len(usable), max(1, cap))
    feats = usable[:n]
    n = len(feats)
    doc = ('    """Co-designed observation: %d features off the raw state."""' % n
           if evolved_from is None else
           '    """Co-designed observation: %d features off the raw state, evolved from '
           'the parent\'s %d-feature observation."""' % (n, evolved_from))
    lines = [f"OBS_DIM = {n}", "", "",
             "def get_observation(state):",
             doc]
    # The accessor is defined inside `compute_reward`, so it cannot be reached
    # from here -- inline a local copy rather than reaching across functions.
    lines += ["    " + ln for ln in _OBS_ACCESSOR.strip("\n").splitlines()]
    lines += ["    return ["]
    lines += [f"        {expr},  # {name}" for name, expr in feats]
    lines += ["    ]"]
    return "\n".join(lines)


#: The `_f` accessor again, standalone. `_PREAMBLE` nests it inside
#: `compute_reward`, and `_extract_observation` lifts `get_observation` out as a
#: free-standing source segment -- so the observation function has to carry its
#: own copy or it is unrunnable the moment anything executes it in isolation.
_OBS_ACCESSOR = '''
def _f(names, index=None, default=0.0):
    v = None
    for name in names:
        if isinstance(state, dict):
            if name in state:
                v = state[name]
                break
        else:
            got = getattr(state, name, None)
            if got is not None:
                v = got
                break
    if v is None and index is not None and not isinstance(state, dict):
        try:
            v = state[index]
        except Exception:
            v = None
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)
'''


def _invalid_program(kind: str) -> str:
    """One broken program per validity-check family (§2).

    Each carries `tier=-1 q=0.00` so a prompt that quotes a failed parent does
    not accidentally bias the next generation upward.
    """
    head = _marker(-1, kind, 0.0)
    if kind == "syntax_error":
        return "\n".join([
            head,
            "def compute_reward(state, action=None, *_extra, **_kw):",
            '    dist = float(getattr(state, "distance", 1.0)',
            "    total = -dist",
            '    return total, {"reach": total}',
        ])
    if kind == "wrong_signature":
        return "\n".join([
            head,
            "def compute_reward():",
            '    """Forgot the (state, action) signature entirely."""',
            "    return -1.0",
        ])
    if kind == "nan_values":
        return "\n".join([
            head,
            "def compute_reward(state, action=None, *_extra, **_kw):",
            '    dist = float(getattr(state, "distance", 1.0))',
            '    scale = float("inf")',
            "    reach = -dist * scale",
            "    drift = scale - scale",
            '    components = {"reach": reach, "drift": drift}',
            "    return reach + drift, components",
        ])
    # constant_reward
    return "\n".join([
        head,
        "def compute_reward(state, action=None, *_extra, **_kw):",
        '    """Always the same number, whatever the agent does."""',
        '    components = {"const": 1.0}',
        "    return 1.0, components",
    ])


# --------------------------------------------------------------------------
# quality signal read back out of a prompt
# --------------------------------------------------------------------------


def _markers(text: str) -> List[Tuple[int, str, float]]:
    return [(int(m.group(1)), m.group(2), float(m.group(3)))
            for m in MARKER_RE.finditer(text or "")]


def _quality_bias(prompt: str) -> float:
    """How much good news the prompt carries, in [0, 1].

    The dominant term is the best parent program quoted in the prompt (read
    from its provenance marker). A small bonus for the presence of evaluation
    feedback stands in for "the model was told how the last attempt behaved" --
    which is the entire premise of §4's prose output.
    """
    qs = [q for tier, _, q in _markers(prompt) if tier >= 0]
    parent = max(qs) if qs else 0.0
    low = (prompt or "").lower()
    informed = any(k in low for k in ("fitness", "success rate", "component value",
                                      "feedback", "reflection"))
    return min(1.0, parent + (0.10 if informed else 0.0))


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", (text or "").strip())[0]


def _is_spec_request(prompt: str) -> bool:
    """Is this the two-stage thinker turn -- the prompt that ends in stage 1's
    prose-only OUTPUT CONTRACT? Imported lazily: this module is loaded by
    `registry.load_all()` and must not import a components module at load."""
    from ..components.generation import _SPEC_TAIL
    return bool(prompt) and _SPEC_TAIL in prompt


def _looks_like_repair(prompt: str) -> bool:
    low = (prompt or "").lower()
    return any(k in low for k in ("traceback", "syntaxerror", "typeerror", "nameerror",
                                 "the previous reward failed", "error:"))


def _tier_weights(bias: float) -> List[float]:
    """Exponential tilt of `TIER_BASE_WEIGHTS` toward the good end.

    `w_i ∝ base_i * exp(QUALITY_TILT * bias * (i - 2) / 2)`: at `bias = 0` this
    is the base prior exactly; at `bias = 1` tier 4 is ~5x more likely than
    tier 0 relative to the prior. Monotone in `bias`, so "better parent ⇒ better
    children" holds by construction and not by luck.
    """
    return [b * math.exp(QUALITY_TILT * bias * (i - 2) / 2.0)
            for i, b in enumerate(TIER_BASE_WEIGHTS)]


def _modal_tier(bias: float) -> int:
    """The greedy-decode answer: the most likely tier, not the best one. A
    temperature-0 sample should be typical, not optimal."""
    weights = _tier_weights(bias)
    return max(range(len(weights)), key=lambda i: weights[i])


def _sample_tier(r: random.Random, bias: float) -> int:
    weights = _tier_weights(bias)
    total = sum(weights)
    u = r.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if u <= acc:
            return i
    return len(weights) - 1


# --------------------------------------------------------------------------
# non-program answers
# --------------------------------------------------------------------------

_SUBTASK_POOL = (
    "move the end effector toward the goal region",
    "reduce the distance to the goal monotonically",
    "arrive with low velocity so the tip does not overshoot",
    "stay inside the goal tolerance once reached",
    "keep control effort small throughout the episode",
    "avoid oscillating around the goal",
    "recover quickly if the tip is pushed away from the goal",
    "hold the final pose until the episode ends",
)

_FAILURE_MODES = (
    "the tip overshoots the goal and oscillates",
    "the agent stalls short of the goal region",
    "control effort dominates the return, so the arm barely moves",
    "the goal is reached but not held for the rest of the episode",
    "progress is fast early and then plateaus",
)

_SUGGESTIONS = (
    "increase the weight on the distance term relative to the control cost",
    "add an explicit bonus for remaining inside the goal tolerance",
    "penalise velocity near the goal so the tip settles",
    "rescale the components so no single term exceeds the others by 100x",
    "remove the term that rewards speed; it is the likely cause of overshoot",
)


def _pick(seq: Sequence[Any], r: random.Random, k: int) -> List[Any]:
    k = max(1, min(int(k), len(seq)))
    return r.sample(list(seq), k)


#: "on a 1.0-5.0 scale" / "on a 0-1 scale" as a caller states it in the prompt,
#: or RF-Agent's "limited to [-1,1]" (self_node_value_verify_single.txt:3, with
#: the range `evaluate.self_verify.range` substitutes; spaces optional). Two
#: alternatives, so the bounds are whichever pair matched -- `_prompt_bounds`.
_PROMPT_SCALE_RE = re.compile(
    r"on an?\s+(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s+scale"
    r"|limited to\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def _prompt_bounds(m: "re.Match[str]") -> Tuple[float, float]:
    lo, hi = (float(g) for g in m.groups() if g is not None)
    return lo, hi


def _score_value(scale: str, r: random.Random, prompt: str = "") -> Any:
    """Honour `evaluate.feedback.score_scale` -- a declared key an evaluator
    that ignored it would turn into a fabricated pin.

    The PROMPT wins where it states bounds. A real LLM reads the instruction it
    was given, and BIRD has two VLM scorers on different scales in one run:
    RDA's in-loop scorer is 3-point on [0,1] (App. 7.3) while its reported
    alignment rate is a 5-point Likert (§5.1). A mock that answered only to the
    in-loop key would return 0.43 to a request for a 1-5 rating, every judgment
    would fail to parse, and the metric would report `null` -- a mock that
    cannot exercise a code path is a code path with no test.
    """
    # The 3-point rubric FIRST, before the range regex: a prompt carrying RDA's
    # App. 7.3 rubric ("3-point success scale ... 1.0 = success") must not be
    # read as a continuous range by an incidental "on a X-Y scale" elsewhere in
    # it -- the whole point of `three_point` is that the answer set is
    # {0.0, 0.5, 1.0}.
    if "3-point" in (prompt or ""):
        return r.choice([0.0, 0.5, 1.0])
    m = _PROMPT_SCALE_RE.search(prompt or "")
    if m:
        lo, hi = _prompt_bounds(m)
        if hi > lo:
            # Integers where the caller asked for an integer-looking range,
            # which is what "5-point Likert" means.
            if float(lo).is_integer() and float(hi).is_integer() and hi - lo >= 1:
                return r.randint(int(lo), int(hi))
            return round(lo + r.random() * (hi - lo), 3)
    if scale == "likert":
        return r.randint(1, 5)
    if scale == "binary":
        return r.randint(0, 1)
    if scale == "three_point":
        return r.choice([0.0, 0.5, 1.0])
    return round(r.random(), 3)


def _numbered_items(prompt: str) -> List[str]:
    """Numbered list items in a prompt, minus anything that is obviously code."""
    return [m.strip() for m in _NUMBERED_RE.findall(prompt or "")
            if not m.strip().startswith(("import", "def ", "#"))]


def _section(prompt: str, header: str) -> str:
    """The body of one `## Header` section, up to the next `## ` or the end.

    The paper-shaped prompts (RDA App. 7.1/7.3/7.4) carry several sections, and
    more than one of them can contain numbered lines -- reward code, env code,
    an example JSON. A mock that ran `_numbered_items` over the WHOLE prompt
    would echo code fragments back as subtasks, so anything that needs 'the
    subtask list the prompt states' parses THAT section and nothing else.
    """
    start = (prompt or "").find(header)
    if start < 0:
        return ""
    body = prompt[start + len(header):]
    nxt = body.find("\n## ")
    return body[:nxt] if nxt >= 0 else body


def _section_items(prompt: str, header: str = "## Subtask List") -> List[str]:
    """Numbered items inside one section only (see `_section`)."""
    return _numbered_items(_section(prompt, header))


def _subtasks_in(prompt: str, r: random.Random, n: int) -> List[str]:
    """Reuse the subtask list already in the prompt when there is one, so
    per-subtask scores line up with `state.subtasks`."""
    found = _numbered_items(prompt)
    if len(found) >= 2:
        return found[:max(2, n)]
    return _pick(_SUBTASK_POOL, r, n)


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------


class MockLLM(LLMClient):
    """Deterministic offline stand-in for a provider.

    Not registered as a method: it is the `llm.*.provider: mock` value, which
    the `tester` profile selects so the paper configs stay citable while still
    executing on a laptop with no API key.
    """

    provider = "mock"

    def __init__(self, ctx: Any, role: str):
        super().__init__(ctx, role)
        self._schema_hint: Any = None

    # -- routing ---------------------------------------------------------

    def _route(self, tag: str, prompt: str) -> str:
        """Decide what kind of answer this call wants.

        `tag` first (callers that name their intent get exactly what they
        asked for), prompt keywords second (so a component that forgets the
        tag still gets a usable answer instead of a reward program).
        """
        p = (prompt or "").lower()
        if self._schema_hint:
            return "json"
        # The two-stage THINKER turn (GT, L2R: `generate.output.two_stage_nl_then_code`)
        # arrives tagged "generate" like every stage-1 call, so by tag it is a
        # program request while its OUTPUT CONTRACT says "Write NO code in
        # this turn". If the tag won, `_nl_spec` would be unreachable from any
        # production caller, every tester-profile GT candidate would carry a
        # fenced program in `nl_spec`, the coder would be handed code as its
        # "specification", and the prose path would have no offline coverage at
        # all -- the opposite of what the tag exists for (base.py: "exercise
        # every branch of the loop"). The contract text is generation's own
        # constant, so the two cannot drift apart silently.
        if _is_spec_request(prompt):
            return "nl_spec"
        # Token-prefix matching, not substring: "generate" contains "rate",
        # and routing the main generation call to the scorer would be a very
        # quiet way to break every end-to-end test.
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", (tag or "").lower()) if tok]

        def has(*keys: str) -> bool:
            return any(tok.startswith(k) for tok in tokens for k in keys)

        # `subtask_reflection` lands here too, and should: RDA's reflection
        # returns a revised subtask LIST, not prose (§6, update.co_evolve).
        # BEFORE the "compar"/"pref" rung below, which "critic" would otherwise
        # never reach anyway -- but a future tag like `critic_compare` would,
        # and a critic answered with a preference verdict is a program that
        # will not compile rather than a wrong number, i.e. a silent zero-vote.
        if has("critic"):
            return "critic"
        if has("decompos", "subtask"):
            return "decompose"
        if has("pref", "compar", "pairwise", "duel"):
            return "preference"
        if has("score", "scoring", "rate", "rating", "judge", "vlm"):
            return "score"
        # `thought` / `align`: RF-Agent's thought alignment asks for a
        # re-description of the reward in prose (initial_thought_alignment.txt),
        # never a program.
        if has("feedback", "reflect", "analy", "critique", "summar", "describ", "eval",
               "thought", "align"):
            return "feedback"
        if has("nl", "spec", "prose"):
            return "nl_spec"
        if has("json", "structured"):
            return "json"
        if not tokens:
            if "which" in p and ("prefer" in p or "better" in p):
                return "preference"
            if "subtask" in p and ("decompose" in p or "break the task" in p):
                return "decompose"
            # A rating request with no tag. `evaluation._client_text` calls
            # `complete(prompt)` positionally -- the generic client contract has
            # no tag slot -- so every VLM judgment arrives here untagged, and
            # falling through to "program" would make the scorer parse a number
            # out of a generated REWARD FUNCTION and call it a judgment: it
            # would look like it worked on `[0,1]`, because code is full of
            # numbers in that range, and return nothing at all on a 1-5 Likert,
            # because code is not full of those. Route by what the prompt asks
            # for. RDA's trajectory analysis (App. 7.3, one structured call
            # scoring every subtask) is also untagged and asks
            # for a `"subtasks"` JSON array against a `## Subtask List` section.
            if "## subtask list" in p and '"subtasks"' in p:
                return "score"
            if _PROMPT_SCALE_RE.search(p) or "rate how well" in p or "score how well" in p:
                return "score"
        return "program"

    # -- the provider entry point ---------------------------------------

    def _complete(self, messages: List[Message], n: int, temperature: float,
                  images: Optional[Sequence[Any]], tag: str) -> List[str]:
        prompt = messages_text(messages)
        route = self._route(tag, prompt)
        out: List[str] = []
        for i in range(n):
            r = self._rng_for(tag, prompt, i)
            out.append(self._answer(route, r, prompt, temperature))
        # One record per provider round-trip, not per sample: the mock batches
        # a chunk the way a real API `n` would (base.chunk_sizes).
        self._record(prompt_tokens=estimate_tokens(prompt),
                     completion_tokens=sum(estimate_tokens(t) for t in out),
                     n_images=len(images or ()))
        return out

    def _rng_for(self, tag: str, prompt: str, index: int) -> random.Random:
        """Per-sample RNG: the shared run stream mixed with a stable hash of
        the request. Drawing from `ctx.rng` keeps the whole run reproducible
        from `seed`; mixing in the prompt keeps two different prompts from
        producing the same program at the same point in the stream."""
        salt = self.rng.random()
        return random.Random(_hash_int(self.model, self.role, tag, prompt, index, salt))

    def _answer(self, route: str, r: random.Random, prompt: str, temperature: float) -> str:
        if route == "decompose":
            return self._decomposition(r, prompt)
        if route == "preference":
            return self._preference(r, prompt)
        if route == "score":
            return self._scores(r, prompt)
        if route == "feedback":
            return self._feedback(r, prompt)
        if route == "critic":
            return self._critic(r, prompt)
        if route == "nl_spec":
            return self._nl_spec(r, prompt)
        if route == "json":
            payload = self._json_payload(r, prompt)
            return "Here is the structured result.\n\n```json\n" + \
                json.dumps(payload, indent=2) + "\n```"
        return self._program_response(r, prompt, temperature)

    # -- reward programs -------------------------------------------------

    def _critic(self, r: random.Random, prompt: str) -> str:
        """A rule-based trajectory comparator, the shape R* asks the model for.

        The tester profile has to be able to run every registered component end
        to end, and `segment_labeller: critic_population_vote` compiles whatever
        comes back and RUNS it -- so a mock that returned prose here would make
        the whole labelling path silently produce zero segments, a declared
        mechanism that does not execute. The body must therefore be real code.

        It compares the mean state value per step -- `states`, the one key
        `_traj_dict` always populates; `rewards` is not in the critic's view
        -- and returns a per-step sign, with a
        per-critic threshold drawn from `r` so a population of five disagrees
        the way a real one does -- a population that always agrees would make
        the 5/4/3 vote ladder untestable, since the first rung would always
        fill the quota.
        """
        eps = round(0.01 + 0.08 * r.random(), 4)
        return (
            "```python\n"
            "def compare(traj_a, traj_b):\n"
            "    import numpy as np\n"
            "    a = np.asarray(traj_a.get('states', []), dtype=float)\n"
            "    b = np.asarray(traj_b.get('states', []), dtype=float)\n"
            "    a = a.reshape(len(a), -1).mean(axis=1) if len(a) else a\n"
            "    b = b.reshape(len(b), -1).mean(axis=1) if len(b) else b\n"
            "    n = int(min(len(a), len(b)))\n"
            "    out = []\n"
            "    for i in range(n):\n"
            f"        d = float(a[i]) - float(b[i])\n"
            f"        out.append(1 if d > {eps} else (-1 if d < -{eps} else 0))\n"
            "    return out\n"
            "```"
        )

    def _program_response(self, r: random.Random, prompt: str, temperature: float) -> str:
        cfg = self.ctx.cfg
        fmt = cfg.get("generate.output.format", "component_dict_return")
        language = str(cfg.get("generate.reward_language") or "numpy")
        bias = _quality_bias(prompt)

        # Failure rate falls with prompt quality, and halves again when the
        # prompt is a repair prompt carrying the traceback (`resample_with_trace`).
        p_invalid = INVALID_FRACTION * (1.0 - 0.5 * bias)
        if _looks_like_repair(prompt):
            p_invalid *= 0.5
        # Temperature is a declared config key, so it must do something
        # (a temperature the client ignores is a fabricated pin). At
        # temperature 0 the mock returns the MODAL tier -- typical, not best --
        # and errs half as often, which is what greedy decoding buys you. Only
        # the numeric coefficients still vary between samples, so a K-candidate
        # pool at temperature 0 is a pool of near-duplicates, as it should be.
        greedy = temperature <= 0.0
        if greedy:
            p_invalid *= 0.5

        if r.random() < p_invalid:
            code = _invalid_program(INVALID_KINDS[r.randrange(len(INVALID_KINDS))])
            preamble = "Here is a revised reward function."
            tier = _modal_tier(bias)  # names only the design thought below; no RNG
        else:
            tier = _modal_tier(bias) if greedy else _sample_tier(r, bias)
            code = _valid_program(tier, r, fmt, language)
            preamble = f"{TIERS[tier]['doc']} Reward components are returned separately " \
                       "so their contributions can be inspected."
            # LIMEN co-design: the prompt asks for `get_observation` in the SAME
            # block, and `_apply_co_design` lifts it back out of `reward_code`.
            # Appended only to a VALID program -- an invalid one is meant to fail
            # `verify`, and handing it a well-formed observation function would
            # let it fail for the wrong reason.
            if cfg.get("generate.co_design.observation_fn"):
                cap = int(cfg.get("generate.co_design.observation_max_dim") or 512)
                # `prompt`, so the emitted features address THIS env's state
                # layout -- see `_state_field_index`.
                code = code + "\n\n\n" + _observation_program(r, cap, prompt)
        out = f"{preamble}\n\n```python\n{code}\n```"
        if "inside a brace" in (prompt or "").lower():
            # RF-Agent's `generate.output.design_thought: inline_brace`
            # (thought_code_output.txt:2, "inside a brace outside the code").
            # Invalid programs carry one too: the release's repair prompt asks
            # for the brace again (initial_failed_feedback.txt:4-5), and a
            # response without it would make `_build_candidate`'s brace parse
            # read the first dict literal in the code instead.
            out = "{" + _first_sentence(TIERS[tier]["doc"]) + "}\n\n" + out
        # The contract for `component_dict_plus_weights` asks for the weights in
        # a SEPARATE ```json block. `_parse_weights` accepts three sources and a
        # module-level dict is the second, so a mock that only put them there
        # would parse and pass the tests -- while the branch the prompt actually
        # requests (`generation.py`'s fenced-json regex) would never be taken.
        # A mock that satisfies a different clause of the contract than the
        # prompt asks for leaves the requested clause untested.
        #
        # ORDER IS LOAD-BEARING: `_weights_of` re-reads the weights out of the
        # EMITTED source, so it
        # must run after the co-design block has finished mutating `code`. The
        # two never co-occur today -- `component_dict_plus_weights` is RDA/GT and
        # `co_design.observation_fn` is LIMEN, which is `scalar_only` -- but
        # deriving the JSON from a string the next line then appends to is the
        # kind of thing that is only correct by accident.
        weights = _weights_of(code)
        if fmt == "component_dict_plus_weights" and weights:
            out += ("\n\nWeights applied by the code above:\n\n```json\n"
                    + json.dumps(weights, indent=2) + "\n```")
        return out

    def _nl_spec(self, r: random.Random, prompt: str) -> str:
        """`generate.output.two_stage_nl_then_code` stage 1 (GT): prose only,
        no code -- the code call comes back through the program route."""
        bias = _quality_bias(prompt)
        tier = _sample_tier(r, bias)
        sug = _pick(_SUGGESTIONS, r, 2)
        return (f"Reward specification.\n\n{TIERS[tier]['doc']}\n"
                f"1. A dense term in the distance between the tip and the goal.\n"
                f"2. A term that keeps the tip inside the goal tolerance once reached.\n"
                f"3. A small penalty on control effort.\n\n"
                f"Relative to the previous design I would {sug[0]}, and {sug[1]}.")

    # -- structured answers ----------------------------------------------

    def _decomposition(self, r: random.Random, prompt: str) -> str:
        """Fresh decomposition, or a revision of the list already in the prompt.

        The second case is RDA's subtask reflection (§6): revise at most one
        entry, hold the count constant, and be allowed to decline -- so a
        co-evolution test can distinguish "the list changed" from "the list
        churns every iteration", which is the failure mode that would make
        `update.co_evolve.subtasks` unmeasurable.
        """
        cfg = self.ctx.cfg
        want = cfg.get("generate.decomposition.n_subtasks", "auto")
        n = max(1, want) if isinstance(want, int) else r.randint(3, 5)
        # A structured decompose prompt (App. 7.1) carries the ENVIRONMENT CODE
        # in a `## Environment Code` section, which is free to contain numbered
        # lines; reading those as "the current subtask list" would echo code
        # fragments back as subtasks. A current list, when one exists, lives in
        # a `## Subtask List` section; a fresh App. 7.1 prompt has none.
        if "## Subtask List" in (prompt or ""):
            current = _section_items(prompt)
        elif "## Environment Code" in (prompt or ""):
            current = []
        else:
            current = _numbered_items(prompt)

        if len(current) >= 2:
            subtasks = current[:n] if isinstance(want, int) else list(current)
            unused = [s for s in _SUBTASK_POOL if s not in subtasks]
            if unused and r.random() < 0.5:
                subtasks = list(subtasks)
                subtasks[r.randrange(len(subtasks))] = unused[r.randrange(len(unused))]
                note = "One subtask was ambiguous and has been rewritten; the rest stand."
            else:
                note = "The current subtasks are unambiguous; declining to revise them."
        else:
            subtasks = _pick(_SUBTASK_POOL, r, n)
            note = f"The task decomposes into {len(subtasks)} subtasks."

        listing = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(subtasks))
        payload = json.dumps({"subtasks": subtasks}, indent=2)
        return f"{note}\n\n{listing}\n\n```json\n{payload}\n```"

    def _reflection_payload(self, r: random.Random, prompt: str) -> Dict[str, Any]:
        """App. 7.4's `{"decision", "subtasks"}` shape (see `_fill`)."""
        current = _section_items(prompt) or _numbered_items(prompt) \
            or _pick(_SUBTASK_POOL, r, 3)
        subtasks = list(current)
        unused = [s for s in _SUBTASK_POOL if s not in subtasks]
        if unused and r.random() < 0.5:
            i = r.randrange(len(subtasks))
            before = subtasks[i]
            subtasks[i] = unused[r.randrange(len(unused))]
            decision = (f"Subtask {i + 1} ('{before}') most contributed to "
                        f"unintended behavior ({_pick(_FAILURE_MODES, r, 1)[0]}); "
                        f"it was refined to encourage natural, goal-aligned "
                        f"behavior.")
        else:
            decision = ("No refinement is needed; the subtask list is kept "
                        "unchanged.")
        return {"decision": decision, "subtasks": subtasks}

    def _preference(self, r: random.Random, prompt: str) -> str:
        left, right = self._pair_ids(prompt, r)
        label, conf = self._preference_label(left, right, prompt)
        winner, loser = (left, right) if label == 1 else (right, left)
        payload = {"left_id": left, "right_id": right,
                   "label": label, "preferred": "A" if label == 1 else "B",
                   "confidence": conf}
        return (f"Clip A ({left}) versus clip B ({right}): "
                f"{winner} holds position closer to the goal for longer, while {loser} "
                f"{_pick(_FAILURE_MODES, r, 1)[0]}.\n\n"
                f"```json\n{json.dumps(payload, indent=2)}\n```")

    def _scores(self, r: random.Random, prompt: str) -> str:
        payload = self._json_payload(r, prompt)
        return ("Scored the rollouts against the task description.\n\n"
                f"```json\n{json.dumps(payload, indent=2)}\n```")

    def _feedback(self, r: random.Random, prompt: str) -> str:
        """Free-text critique (§4 prose output / GT's human channel).

        Echoes numbers that are actually in the prompt so a test can assert the
        feedback is conditioned on the evaluation, not invented.
        """
        nums = _NUM_RE.findall(prompt or "")[:3]
        modes = _pick(_FAILURE_MODES, r, 2)
        sug = _pick(_SUGGESTIONS, r, 2)
        seen = f" Observed values: {', '.join(nums)}." if nums else ""
        return ("Behaviour summary.{seen}\n"
                "- What works: the agent makes early progress toward the goal.\n"
                f"- What fails: {modes[0]}; secondarily, {modes[1]}.\n"
                f"- Next revision: {sug[0]}. Also consider: {sug[1]}."
                ).replace("{seen}", seen)

    # -- JSON shaping ----------------------------------------------------

    def chat_json(self, messages: Any, schema_hint: Any = "", **kw: Any) -> Dict[str, Any]:
        """Same contract as the base class, but the mock guarantees a
        well-formed answer by shaping it from `schema_hint` directly instead of
        hoping a text completion parses.

        The hint reaches `_complete` through an instance attribute rather than
        the call signature, because the signature is the frozen contract every
        provider shares. Single call at a time -- the stage loop is sequential.
        """
        self._schema_hint = schema_hint
        try:
            return super().chat_json(messages, schema_hint, **kw)
        finally:
            self._schema_hint = None

    def _json_payload(self, r: random.Random, prompt: str) -> Dict[str, Any]:
        hint = self._schema_hint
        if hint:
            filled = self._fill(hint, r, prompt, key="")
            if isinstance(filled, dict):
                return filled
            return {"items": filled}
        return self._default_payload(r, prompt)

    def _default_payload(self, r: random.Random, prompt: str) -> Dict[str, Any]:
        """What to answer when the caller asked for JSON without a schema."""
        scale = self.ctx.cfg.get("evaluate.feedback.score_scale", "[0,1]")
        if "## Subtask List" in (prompt or "") and '"subtasks"' in (prompt or ""):
            return self._trajectory_analysis_payload(r, prompt, scale)
        payload: Dict[str, Any] = {
            "score": _score_value(scale, r, prompt),
            "reason": _pick(_FAILURE_MODES, r, 1)[0].capitalize() + ".",
        }
        if "subtask" in (prompt or "").lower():
            subtasks = _subtasks_in(prompt, r, r.randint(3, 5))
            payload["scores"] = {s: _score_value(scale, r, prompt) for s in subtasks}
            payload["rationales"] = {s: _pick(_FAILURE_MODES, r, 1)[0] for s in subtasks}
        return payload

    def _trajectory_analysis_payload(self, r: random.Random, prompt: str,
                                     scale: str) -> Dict[str, Any]:
        """RDA's App. 7.3 output: one `subtasks` array covering EVERY listed
        subtask, in order.

        The names come verbatim from the prompt's `## Subtask List` section --
        that section specifically, because the same prompt also carries the
        reward function and a per-step table, either of which can contain
        numbered lines -- so `subtask_scores` stays keyed by the exact texts
        `decompose` journalled and the artifact join tests hold offline.
        """
        subs = _section_items(prompt) or _pick(_SUBTASK_POOL, r, 3)
        entries = []
        for i, s in enumerate(subs, start=1):
            entries.append({
                "number": i,
                "name": s,
                "behavior": ("The agent " + _pick(_FAILURE_MODES, r, 1)[0]
                             + " during this subtask."),
                "score": _score_value(scale, r, prompt),
                "analysis": _pick(_FAILURE_MODES, r, 1)[0].capitalize() + ".",
            })
        return {"subtasks": entries}

    def _fill(self, hint: Any, r: random.Random, prompt: str, key: str,
              parent: str = "") -> Any:
        """Build a value that matches the caller's schema hint.

        The hint is treated as an example/shape rather than a JSON-Schema
        document: dicts recurse, lists give a short list of their element
        shape, and leaf hints are read as either a type name or an example
        value. Key names drive the content, which is what makes
        `{"score": "float", "reason": "str"}` come back sensibly filled. The
        containing key travels down as `parent` so that the leaves of
        `{"scores": {"reach": "float"}}` are still scores.
        """
        scale = self.ctx.cfg.get("evaluate.feedback.score_scale", "[0,1]")
        if isinstance(hint, dict):
            if set(hint) == {"decision", "subtasks"}:
                # RDA's subtask reflection (App. 7.4):
                # the answer is the FULL final list, so a generic fill -- one
                # invented item under "subtasks" -- would read as "changed J"
                # and every tester-profile reflection would decline. Echo the
                # prompt's own list, sometimes rewriting exactly one item, so
                # the apply path and the decline path are both exercised
                # offline.
                return self._reflection_payload(r, prompt)
            return {k: self._fill(v, r, prompt, key=str(k), parent=key)
                    for k, v in hint.items()}
        if isinstance(hint, (list, tuple)):
            if not hint:
                return _subtasks_in(prompt, r, 3) if "subtask" in key.lower() else []
            return [self._fill(hint[0], r, prompt, key, parent)
                    for _ in range(min(3, max(1, len(hint))))]
        if isinstance(hint, str):
            parsed = None
            try:
                parsed = json.loads(hint)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                return self._fill(parsed, r, prompt, key, parent)
            return self._leaf(hint.strip().lower(), r, prompt, key, parent, scale)
        if isinstance(hint, bool):
            return bool(r.getrandbits(1))
        if isinstance(hint, float):
            return round(r.random(), 3)
        if isinstance(hint, int):
            return int(_score_value(scale, r, prompt)) if "score" in key.lower() else r.randint(0, 5)
        return self._leaf("", r, prompt, key, parent, scale)

    def _leaf(self, type_hint: str, r: random.Random, prompt: str, key: str,
              parent: str, scale: str) -> Any:
        k = f"{key or ''} {parent or ''}".lower()
        if type_hint in ("float", "number", "double"):
            return round(r.random(), 3) if "score" not in k else _score_value(scale, r, prompt)
        if type_hint in ("int", "integer"):
            return r.randint(0, 5)
        if type_hint in ("bool", "boolean"):
            return bool(r.getrandbits(1))
        if "subtask" in k:
            return _subtasks_in(prompt, r, 3)
        if any(t in k for t in ("score", "rating", "value", "confidence", "coefficient")):
            return _score_value(scale, r, prompt)
        if any(t in k for t in ("reason", "rationale", "explanation", "analysis",
                                "critique", "feedback", "summary", "comment")):
            return _pick(_FAILURE_MODES, r, 1)[0].capitalize() + "."
        if any(t in k for t in ("prefer", "winner", "choice", "label")):
            left, right = self._pair_ids(prompt, r)
            label, _ = self._preference_label(left, right, prompt)
            return "A" if label == 1 else "B"
        if k.endswith("_id") or k == "id":
            return self._pair_ids(prompt, r)[0]
        if "name" in k:
            return _pick(("reach", "stay", "ctrl", "proximity"), r, 1)[0]
        return _pick(_SUGGESTIONS, r, 1)[0].capitalize() + "."

    # -- preference machinery --------------------------------------------

    def _pair_ids(self, prompt: str, r: random.Random) -> Tuple[str, str]:
        ids: List[str] = []
        for cid in _ID_RE.findall(prompt or ""):
            if cid not in ids:
                ids.append(cid)
        while len(ids) < 2:
            ids.append(f"x{r.randrange(1000):04d}")
        return ids[0], ids[1]

    def _preference_label(self, left: str, right: str, prompt: str) -> Tuple[int, float]:
        """1 = left preferred, matching `types.Preference.label`.

        Signal = each side's quality (its program's provenance marker when the
        prompt quotes the code, otherwise a stable hash of the id, so labels
        are consistent across calls for the same pair). A fixed
        `PREFERENCE_NOISE` fraction of pairs is flipped by an independent hash
        of the pair, so the Bradley-Terry likelihood in §4 is fitting noisy
        comparisons rather than a perfectly separable ordering.
        """
        ql, qr = self._id_quality(left, prompt), self._id_quality(right, prompt)
        label = 1 if ql >= qr else 0
        if _hash01("flip", left, right) < PREFERENCE_NOISE:
            label = 1 - label
        margin = abs(ql - qr)
        return label, round(0.5 + 0.5 * min(1.0, margin), 3)

    @staticmethod
    def _id_quality(cand_id: str, prompt: str) -> float:
        """Quality of a candidate as visible from this prompt."""
        text = prompt or ""
        at = text.find(cand_id)
        if at >= 0:
            window = text[at:at + 6000]
            marks = _markers(window)
            if marks:
                return marks[0][2]
        return _hash01("quality", cand_id)


@register("llm", "mock", doc="Deterministic offline LLM simulator (no network).")
def mock_factory(ctx: Any, role: str) -> MockLLM:
    """Factory for `llm.generator.provider: mock` / `llm.evaluator.provider: mock`."""
    return MockLLM(ctx, role)
