"""The candidate's reward, substituted into upstream's own IPPO trainer.

WHY A BINDING AND NOT A PARAMETER: upstream's `ippo_ff_nps.make_train` builds
its environment itself --

    env = assistax.make(config["ENV_NAME"], **env_kwargs)        # :258/261
    env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo)     # :259
    env = LogWrapper(env, replace_info=True)                     # :271

and the rollout reads the reward straight off it at `:410`. There is no `env=`
argument and no hook, so training a BIRD candidate on its own reward -- the
entire point of this repo -- requires replacing something upstream reaches by
name. `load_from_zoo` is a classmethod called on the module-level name
`LoadAgentWrapper`, which `ippo_ff_nps` imported with `from ... import`, so
rebinding THAT name for the duration of one call makes `:259` return our
subclass. Upstream's trainer runs byte-identical; the only new code is one
subclass we own.

THE WRAPPER MUST BE THE OUTER ONE, AND IT MUST OVERWRITE, NOT ADD. Two facts
decide this and both are in upstream's source:

  * `LoadAgentWrapper.step` calls `self._env.STEP_ENV(...)` (`aht.py:872`),
    not `step`. A reward wrapper placed INSIDE it, overriding `step`, is never
    called at all -- a silent no-op that looks wired.
  * `aht.py:881` then does
    `rewards = {agent: rewards[agent] + pref_reward for agent in rewards}`.
    So an inner wrapper that overrode `step_env` instead would have its
    candidate reward arrive at the training call site as `candidate + R_pref`
    -- the held-out metric inside the gradient, where nothing downstream can
    remove it.

Overriding `step` on the OUTER wrapper and overwriting the ego's reward after
`super().step` returns is therefore the only placement that trains the robot
on the candidate alone. The partner does not learn, so what the partner's
reward says is irrelevant to the gradient; `gt_return` is still reported as
`state.reward - total_pref_reward`.
"""
from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Optional, Tuple

#: The agent BIRD trains. The partner is frozen and does not learn, so its
#: reward is left exactly as upstream computed it.
EGO = "robot"


def candidate_reward_wrapper_class(base_cls: type,
                                   row_fn: Callable[..., Tuple[Any, Dict[str, Any]]],
                                   to_row: Callable[[Any], Any],
                                   ego: str = EGO) -> type:
    """A `LoadAgentWrapper` subclass whose ego reward is the candidate's.

    Built as a subclass of whatever `base_cls` is passed rather than of
    `LoadAgentWrapper` imported here, so the caller can stack this on top of
    the preference-observation subclass without a second inheritance path
    that the two would have to agree about.
    """

    class _CandidateRewarded(base_cls):                       # type: ignore[misc,valid-type]
        def step(self, key: Any, state: Any, actions: Dict[str, Any],
                 reset_state: Optional[Any] = None) -> Any:
            """Upstream's transition, the ego's reward OVERWRITTEN.

            The candidate is paid on `(row(s), a_ego, row(s'))` -- the same
            88-wide row the prompt documents and `reference_reward` is keyed
            on, so the number the learner maximises and the number the
            artifact records are the same function of the same inputs.

            `super().step` has already added the partner's preference reward
            to every agent. Assigning (not adding) to the ego's key removes it
            from the ego's signal; the partner's entry keeps it, and the
            partner is frozen, so the AHT protocol is unchanged.
            """
            prev_row = to_row(state)
            out = super().step(key, state, actions, reset_state)
            obs, next_state, rewards, dones, infos = out
            # `to_row(next, prev)` -- the row's tool velocity is a finite
            # difference and is NOT a function of one state. Passing only
            # `next_state` silently zeroes it.
            total, _comps = row_fn(prev_row, actions[ego],
                                   to_row(next_state, state))
            rewards = dict(rewards)
            rewards[ego] = total
            # `__all__` is the team scalar some wrappers reduce on. It must
            # follow the ego rather than keep upstream's sum, or a logger
            # reading it would report a reward nobody was trained on.
            if "__all__" in rewards:
                rewards["__all__"] = total
            return obs, next_state, rewards, dones, infos

    # THE MARKER THE GUARD READS. A class ATTRIBUTE, not the class name:
    # matching `__name__.startswith("CandidateRewarded")` against a name
    # assigned elsewhere would be two copies of one fact with nothing holding
    # them together -- rename the class for readability and the guard would
    # silently stop recognising it.
    _CandidateRewarded.bird_candidate_rewarded = True
    _CandidateRewarded.__name__ = f"CandidateRewarded{base_cls.__name__}"
    return _CandidateRewarded


def row_logging_class(base_cls: type, to_row: Callable[..., Any],
                      key: str = "bird_row") -> type:
    """A `LoadAgentWrapper` subclass whose `step` puts the ADAPTER'S ROW in `infos`.

    WHY `infos` AND NOT `state.metrics`. Upstream's `step` builds the auto-reset
    branch from the INNER env's `reset` (`aht.py:882-899`), then `jax.tree.map`s
    it against the stepped branch -- so a key added to `metrics` by this wrapper
    would be present on one side of that select and absent on the other, and
    the scan would refuse the pytree. `infos` is rebuilt every step and only
    ever stacked by the evaluation scan (`EvalInfo.info`), so one extra key
    there is structurally free. The training loop never binds this class.

    WHAT IT IS FOR. Without it this tier returns scalars only (`gt_return`,
    `r_pref`, the eval payload) and no rollout: every fitness source that reads
    a trajectory -- `vlm_score` (frames off states), `ground_truth_metric`,
    `success_rate` -- has nothing to read, and selection runs blind. The row is the same 88-wide vector the prompt
    documents and `reference_reward` is keyed on, so a trajectory of rows is
    exactly what `env.render(state)` and `env.task_metric(traj)` consume.
    """

    class _RowLogged(base_cls):                              # type: ignore[misc,valid-type]
        def step(self, key_: Any, state: Any, actions: Dict[str, Any],
                 reset_state: Optional[Any] = None) -> Any:
            obs, next_state, rewards, dones, infos = super().step(key_, state, actions, reset_state)
            infos = {**infos, key: to_row(next_state, state)}
            return obs, next_state, rewards, dones, infos

    _RowLogged.bird_row_logged = True
    _RowLogged.bird_row_key = key
    _RowLogged.__name__ = f"RowLogged{base_cls.__name__}"
    return _RowLogged


@contextlib.contextmanager
def log_wrapper_binding(trainer_module: Any, extra_keys: Any):
    """Bind a `LogWrapper` subclass that PRESERVES `extra_keys` in `infos`, then restore.

    WHY THIS IS NEEDED BESIDE `row_logging_class`. `make_evaluation` wraps the
    env in `LogWrapper(env, replace_info=True)` (`ippo_ff_nps.py:746`), and that
    wrapper rebuilds `info` from a fixed key set (`PRESERVE_KEYS` = the two
    preference dicts, `baselines.py:55`; `_transform_preference_metrics` drops
    every other key even before `replace_info` does). So a row put into `infos`
    one wrapper down never reaches `EvalInfo.info`: training succeeds and the
    evaluation returns zero rollouts. The subclass passes `preserve_keys = upstream's | extra_keys`; a non-scalar
    value is kept as is by the transform. Bound on the TRAINER MODULE's name,
    exactly as `candidate_reward_binding` does and for the same reason.
    """
    original = trainer_module.LogWrapper
    extra = set(extra_keys)

    class _RowPreservingLogWrapper(original):                 # type: ignore[misc,valid-type]
        def __init__(self, env: Any, replace_info: bool = False, crossplay_info: bool = False,
                     preserve_keys: Any = None) -> None:
            base = set(preserve_keys) if preserve_keys is not None else set(_upstream_preserve_keys(original))
            super().__init__(env, replace_info=replace_info, crossplay_info=crossplay_info,
                             preserve_keys=base | extra)

    _RowPreservingLogWrapper.bird_preserved_keys = frozenset(extra)
    _RowPreservingLogWrapper.__name__ = f"RowPreserving{original.__name__}"
    trainer_module.LogWrapper = _RowPreservingLogWrapper
    try:
        yield _RowPreservingLogWrapper
    finally:
        trainer_module.LogWrapper = original


def _upstream_preserve_keys(log_wrapper_cls: type) -> Any:
    """Upstream's default `PRESERVE_KEYS`, read off the module that defines the
    wrapper class (so a fake base in a test supplies its own)."""
    import sys                                                # noqa: WPS433
    mod = sys.modules.get(log_wrapper_cls.__module__)
    keys = getattr(mod, "PRESERVE_KEYS", None)
    return set(keys) if keys is not None else set()


@contextlib.contextmanager
def candidate_reward_binding(trainer_module: Any, wrapper_cls: type):
    """Bind `wrapper_cls` as the trainer module's `LoadAgentWrapper`, then restore.

    THE MODULE ATTRIBUTE, NOT `assistax.wrappers.aht`'s. `ippo_ff_nps` does
    `from assistax.wrappers.aht import ... LoadAgentWrapper`, so the name it
    calls is its own; patching the defining module would rebind a name nobody
    reads and look like it worked.

    Restored in a `finally`, including on an exception: a trainer module left
    pointing at a candidate-rewarded wrapper would silently change every later
    training AND the evaluation env in the same process, and the symptom would
    be an eval that agrees with training for the wrong reason.
    """
    original = trainer_module.LoadAgentWrapper
    trainer_module.LoadAgentWrapper = wrapper_cls
    if trainer_module.LoadAgentWrapper is not wrapper_cls:      # pragma: no cover
        trainer_module.LoadAgentWrapper = original
        raise RuntimeError(
            "assistax_ppo: the candidate-reward wrapper did not bind onto "
            f"{trainer_module.__name__}.LoadAgentWrapper. Training would run "
            "on UPSTREAM's task reward plus the partner's preference reward "
            "-- the held-out metric inside the gradient -- and would not "
            "raise; refusing instead.")
    try:
        yield wrapper_cls
    finally:
        trainer_module.LoadAgentWrapper = original


def assert_candidate_rewarded(env: Any) -> None:
    """Refuse an env the candidate's reward is not actually on.

    THE FAILURE THIS CATCHES IS SILENT. If the binding does not reach the name
    `make_train` resolves -- wrong module, an import that re-bound it, a
    future upstream that stops calling `LoadAgentWrapper.load_from_zoo` -- then
    the trainer builds upstream's BARE wrapper, the rollout reads upstream's
    task reward plus the partner's preference reward, and the run completes
    normally with a plausible curve. Nothing downstream can tell that apart
    from a candidate that happened to score like upstream's reward.

    A guard that refuses to step a wrapper with no candidate bound does not
    port here -- this design closes over the reward at class construction, so
    there is no unbound state to check -- but the risk it protects against is
    real in a different place, and the guard belongs at the point where this
    design can still get it wrong.
    """
    chain, found, probe = [], False, env
    for _ in range(8):
        chain.append(type(probe).__name__)
        # `in type(probe).__dict__` OR inherited is fine -- a subclass of a
        # candidate-rewarded class is still candidate-rewarded -- but it must
        # be read off the object, not matched on its name.
        found = found or bool(getattr(probe, "bird_candidate_rewarded", False))
        probe = getattr(probe, "_env", None)
        if probe is None:
            break
    if not found:
        raise RuntimeError(
            f"assistax_ppo: the env the trainer built carries no "
            f"candidate-reward wrapper (chain: {' -> '.join(chain)}). It would "
            f"train on upstream's reward plus the partner's preference reward, "
            f"silently and without error.")


def assert_human_only_pref_obs(env: Any) -> None:
    """Refuse an env whose preference-observation override is not in place.

    THE EVAL-SIDE COUNTERPART of `assert_candidate_rewarded`, and deliberately
    a DIFFERENT marker. The evaluation env must carry the observation
    override (so the robot is not shown the 7 held-out preference values) and
    must NOT carry the reward override (so `gt_return` is upstream's reward
    and not the candidate scoring itself). One marker cannot express both:
    asserting `bird_candidate_rewarded` on the eval env would refuse the
    correct configuration.

    Why this rather than the width equality alone: a width check catches a
    mismatched class and passes for any other class with a coincidentally
    equal width. A loud width mismatch is the LUCKY outcome -- matched widths
    would let the held-out dims into the robot's eval observation silently.
    """
    chain, found, probe = [], False, env
    for _ in range(8):
        chain.append(type(probe).__name__)
        found = found or bool(getattr(probe, "bird_human_only_pref_obs", False))
        probe = getattr(probe, "_env", None)
        if probe is None:
            break
    if not found:
        raise RuntimeError(
            f"assistax_ppo: the evaluation env carries no human-only "
            f"preference-observation override (chain: {' -> '.join(chain)}). "
            f"Upstream's bare LoadAgentWrapper appends the 7 held-out "
            f"preference values to EVERY agent, so the robot would be "
            f"evaluated on an observation containing the metric it is scored "
            f"against.")
