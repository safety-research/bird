"""Component registry.

The rule this file exists to enforce: there is no
`if config.name == "eureka"` anywhere in the codebase. A config *value* is a
registry *key*, and the registry hands back the implementation.

Usage:

    @register("screen", "tpe")
    def tpe_screen(ctx, candidates): ...

    fn = get("screen", cfg.verify.quality_screen)
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Tuple

_REGISTRY: Dict[Tuple[str, str], Callable[..., Any]] = {}
_DOCS: Dict[Tuple[str, str], str] = {}

#: Component families. Each maps 1:1 onto a config key; the schema's `enum`
#: for that key must equal the set of names registered under the family.
KINDS = (
    "sampling_mode",  # generate.sampling_mode
    "crossover_operator",  # generate.crossover.operator
    "crossover_parent_selection",  # generate.crossover.parent_selection
    "segment_labeller",  # generate.alignment.labeller
    "param_alignment",  # generate.alignment.method
    "candidate_schedule",  # generate.candidate_schedule
    "parent_source",  # generate.parent_source
    "env_spec",  # generate.context.env_spec
    "history_mode",  # generate.history_mode
    "output_format",  # generate.output.format
    "generator_backend",  # generate.generator_backend
    "symbol_table",  # generate.postprocess.symbol_mapping (string form)
    "static_check",  # verify.static_checks[]
    "dynamic_check",  # verify.dynamic_checks[]
    "verify_failure",  # verify.on_failure
    "screen",  # verify.quality_screen
    "dedup",  # verify.dedup.method
    "check_order",  # verify.check_order
    "train_backend",  # train.backend
    "candidate_parallelism",  # train.candidate_parallelism
    "interaction",  # train.interaction
    "interaction_allocator",  # train.interaction_allocator
    "reward_scaling",  # train.reward_scaling
    "reward_source",  # train.reward_source
    "elite_constraint",  # train.elite_constraint.kind
    "hyperparameter_search",  # train.hyperparameter_search
    "fusion_ratio_search",  # train.fusion.ratio_search
    "checkpoint_aggregation",  # evaluate.fitness.checkpoint_aggregation
    "seed_aggregation",  # evaluate.fitness.seed_aggregation
    "fitness_source",  # evaluate.fitness.source
    "frame_policy",  # evaluate.vlm.frame_policy
    "feedback_builder",  # composed from evaluate.feedback.*
    "comparator",  # evaluate.preferences.comparator
    "pair_strategy",  # evaluate.preferences.pairs
    "pref_aggregator",  # evaluate.preferences.aggregator
    "similarity",  # evaluate.similarity.metric
    "select_rule",  # select.rule
    "tie_break",  # select.tie_break
    "significance",  # select.significance
    "allocation",  # select.allocation
    "topology",  # update.topology
    "update_operator",  # update.operator
    "winner_action",  # update.winner.action
    "loser_action",  # update.loser.action
    "prompt_mode",  # update.prompt.mode
    "feedback_routing",  # update.feedback.routing
    "curriculum_author",  # loop.curriculum.author
    "stage_gate",  # loop.curriculum.gate
    "stall_action",  # loop.curriculum.on_stall
    "termination",  # loop.termination
    "phase",  # pre[] / post[] plugins
    "env",  # problem.env_id
    "llm",  # llm.generator.provider
    "tracker",  # output.tracker
    "video_format",  # output.video.format
)


class RegistryError(KeyError):
    pass


def register(kind: str, name: str, doc: str = "") -> Callable:
    """Decorator: bind `name` within family `kind` to the decorated callable."""
    if kind not in KINDS:
        raise RegistryError(f"unknown component kind {kind!r}; expected one of {KINDS}")

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        key = (kind, name)
        if key in _REGISTRY and _REGISTRY[key] is not fn:
            raise RegistryError(f"duplicate registration for {kind}:{name}")
        _REGISTRY[key] = fn
        _DOCS[key] = doc or (fn.__doc__ or "").strip().split("\n")[0]
        return fn

    return _wrap


def get(kind: str, name: str) -> Callable[..., Any]:
    load_all()
    try:
        return _REGISTRY[(kind, name)]
    except KeyError:
        raise RegistryError(
            f"no implementation registered for {kind}:{name!r}. "
            f"Available: {sorted(names(kind))}"
        ) from None


def names(kind: str) -> List[str]:
    load_all()
    return sorted(n for (k, n) in _REGISTRY if k == kind)


def describe(kind: str) -> Dict[str, str]:
    load_all()
    return {n: _DOCS[(kind, n)] for (k, n) in _REGISTRY if k == kind}


_LOADED = False
_MODULES = (
    "bird.components.generation",
    "bird.components.verification",
    "bird.components.screens",
    "bird.components.training",
    # Optional learners import torch / jax INSIDE the backend call, never at module
    # scope: `load_all()` runs for every config, including `--validate-all` on a
    # machine with neither installed.
    "bird.components.fasttd3",
    "bird.components.simba_v2",
    "bird.components.assistax_ppo",
    "bird.components.population",
    "bird.components.evaluation",
    "bird.components.preferences",
    "bird.components.selection",
    "bird.components.update",
    "bird.components.evolution",
    "bird.components.alignment",
    "bird.components.phases",
    "bird.components.curriculum",
    "bird.components.frames",
    "bird.components.search",
    "bird.components.tree",
    "bird.envs.toy",
    "bird.envs.spec",
    "bird.envs.control",
    "bird.envs.humanoid_hand",
    "bird.envs.gym_mujoco",
    "bird.envs.assistax",
    "bird.envs.upstream_assistax",
    "bird.envs.metaworld",
    "bird.envs.jax_toy",
    "bird.observability",
)

#: The `llm` family: `llm:mock`, `llm:fixed`, `llm:anthropic`, `llm:openai`.
#: Ordinary registry members, imported by `load_all()` after `_MODULES`. The
#: component modules never import them at module scope
#: (`tests/test_load_all_does_not_import_llm.py`), so the dependency points
#: one way: `bird/llm/` may import the components, never the reverse.
_LLM_MODULES = (
    "bird.llm.mock",
    "bird.llm.fixed",
    "bird.llm.anthropic_client",
    "bird.llm.openai_client",
)


def load_all() -> None:
    """Import every component module so its @register decorators run."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True  # set first: modules may import each other
    mods = _MODULES + _LLM_MODULES
    for mod in mods:
        try:
            importlib.import_module(mod)
        except ImportError as exc:  # optional deps (e.g. anthropic, sb3)
            if mod.rsplit(".", 1)[-1] in ("anthropic_client",):
                continue
            raise RegistryError(f"failed to import component module {mod}: {exc}") from exc
