"""`train.backend: assistax_ppo` -- upstream Assistax IPPO, our reward.

The robot learns; the human is a FROZEN partner sampled per episode from the HF
zoo. One jitted program per training: MJX env step + zoo partner forward + PPO
update + the LLM reward.

WHAT THIS WRAPS. Upstream's PPO_AHT entry point, `baselines/ZSC/ppo_aht.py`,
has no `make_train`: it is a `@hydra.main` DRIVER that builds zoo partner sets
and calls the trainable, which lives at `baselines/IPPO/ippo_ff_nps.py:235`,
`make_train(config, save_train_state, load_zoo, dynamic_preferences)`. We wrap
that and re-express the driver's partner selection (zoo index query per
algorithm, train/held-out split by a recorded split seed) in our own config;
hydra never enters this process.

TWO DIFFERENT ARRAYS, BY DESIGN, AND CONFUSING THEM IS THE FAILURE THIS
DOCSTRING EXISTS TO PREVENT:

  * the ROBOT POLICY observes UPSTREAM's per-agent robot observation. Their
    tuned PPO hyperparameters were set on that vector and nothing else.
  * the REWARD ROW is evaluated on BIRD's full-state layout,
    `concat(qpos, qvel, slots, tail)`, with the CPU tail lambdas IMPORTED from
    `bird.envs.assistax` so the field layout is byte-identical to what the
    prompt documents to the LLM.

The candidate arrives as `CompiledReward.row()` (`training.py`): the
un-jitted device row `(s, a, s_next) -> (total, comps)` with the trainer's
argument binding and the LLM's weights already applied in `jnp`. We `vmap` and
`jit` it into the update, never call it on the host.


================================ WHAT IS FAITHFUL AND WHAT IS NOT ==============

FAITHFUL to upstream Assistax PPO_AHT:
  * the learner: `ippo_ff_nps.make_train`, their network, their PPO, their
    tuned hyperparameters.
  * the partner: frozen zoo params, sampled per episode, fed its OWN
    preference vector so it acts in-distribution -- it was trained with those
    7 values in its input, and `LoadAgentWrapper` decides that from the zoo
    index (`wrappers/aht.py:665`, `_zoo_agents_expect_pref_obs`).
  * bedbathing's wipe latch is UPSTREAM's `state.info["contact_vector"]`
    (`envs/bedbathing.py:380-390`), read straight into `slots[:52]`, and its
    TERMINATION is upstream's -- the episode ends when the last point is wiped,
    where the CPU tier has no such termination. `task_metric` on this tier is
    upstream's wiped fraction. The reason: the policy lives in upstream's
    environment, that environment already uses this latch for its own
    dynamics, and the zoo partner was trained under it. Recomputing the latch
    from BIRD's own tool-human force would put TWO DISAGREEING WIPE COUNTS in
    one episode -- the env's, which ends it, and the reward row's, which the
    candidate reads -- a worse defect than the parity caveat it would buy.
    This tier's numbers are therefore not a step-for-step comparison with the
    CPU tier's, by design.

NOT FAITHFUL, deliberately, both consequences of holding R_pref out:
  1. UPSTREAM'S PREFERENCE MODE IS NOT RUN FOR THE ROBOT. We pass
     `dynamic_preferences=False`. R_pref is computed OUTSIDE the training graph,
     on eval rollouts, and written to `seed_metrics`/trace and nowhere else --
     never the robot's observation, never its training reward, never a prompt,
     never fitness: it is a held-out validation metric that neither supervised
     nor unsupervised methods may access.
  2. THE ROBOT OBSERVES 7 FEWER DIMENSIONS than PPO_AHT under dynamic
     preferences, so upstream's tuned robot hyperparameters were set on a wider
     input than ours. This is forced. Upstream's
     `LoadAgentWrapper._append_pref_to_obs` (`wrappers/aht.py:782-787`)
     concatenates the partner's 7 preference values onto EVERY agent's
     observation -- `{agent: jnp.concatenate([o, pref_vec]) for agent, o in
     obs.items()}` -- in both `reset` (827-829) and `step` (911-913), and
     widens `observation_spaces` for all agents (590-596); and the same
     `pref_configs` argument that makes the partner act in-distribution is what
     switches that on. No upstream flag separates the two. `_HumanOnlyPrefObs`
     below overrides both so only the human is augmented.

     The override is used instead of stripping the dims after the fact because
     a strip leaves the values inside an array the robot's pipeline has already
     touched, and the next refactor drops the strip while every test still
     passes.

Locators for the record: `aht.py:782-787, 827-829, 911-913, 590-596`;
`ippo_ff_nps.py:268-269`; `wrappers/training.py:525-551`;
`envs/bedbathing.py:380-390`.

================================================================================

NOTHING JAX IS IMPORTED AT MODULE SCOPE: a module-scope import of a missing GPU
dependency would break every config load, for every config, on a machine that
will never run this backend. `jax` lives in its own extra (`pyproject.toml`,
`scripts/setup_jax.sh`) and is held out of `all`. Every import below happens
inside a function.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from ..registry import register
from ..types import Candidate, TrainResult

BACKEND_NAME = "assistax_ppo"

#: The five Assistax tasks, matching `bird/envs/assistax.py`'s. `handover` is a
#: real upstream env (`envs/handover.py`) but is not in this family.
TASKS: Tuple[str, ...] = (
    "feeding", "teethbrushing", "scratchitch", "armmanipulation", "bedbathing",
)

#: Upstream appends exactly these, in this order, to an agent's observation
#: under preference mode (`wrappers/aht.py:772-780`). Named here so the
#: held-out check has one list to assert against rather than a magic 7.
PREF_OBS_FIELDS: Tuple[str, ...] = (
    "w_speed", "w_force", "w_touch",
    "speed_range_min", "speed_range_max",
    "force_range_min", "force_range_max",
)
N_PREF_OBS = len(PREF_OBS_FIELDS)

#: bedbathing's wipe latch, upstream's `contact_vector`. 0 == WIPED, matching
#: both `envs/bedbathing.py:386-389` and our CPU `_adv_bedbathing`
#: (`bird/envs/assistax.py`), whose docstring cites upstream's function. The two
#: conventions agree; this constant exists so a reader does not have to
#: rediscover that they do.
N_WIPE_POINTS = 52
WIPED = 0.0


# --------------------------------------------------------------------------
# The partner wrapper: the human gets its preference vector, the robot does not
# --------------------------------------------------------------------------

#: THE TWO METRIC SLOTS THIS TIER ADDS, named once because three places
#: must agree about them: the seeding wrapper below, the step that fills
#: them, and the key-set test. `bird_` prefixed because they are OURS on
#: upstream's state and an unprefixed name could be read as a future
#: upstream metric.
BIRD_METRIC_SLOTS = ("bird_prev_contact_force", "bird_ag_idx")

#: Seeded at RESET with these, and the dtypes matter as much as the keys: a
#: `lax.scan` carry must match in structure AND leaf dtype/shape. -1 for the
#: partner index is a sentinel meaning "no partner drawn yet", which is only
#: ever seen on a reset state.
BIRD_METRIC_SEEDS = {"bird_prev_contact_force": 0.0, "bird_ag_idx": -1}


def _metric_slot_env(base_env: Any) -> Any:
    """Wrap the BASE env so its reset seeds the two slots.

    WHY RESET MUST SEED THEM. Adding the slots in `step` alone would leave
    `reset` without them, so `lax.scan`'s carry would change structure between
    input and output: "scan body function carry input and carry output must
    have the same pytree structure" at `ippo_ff_nps.py:462`, at TRACE, before
    any training.

    IT HAS TO BE THE INNER ENV, not our wrapper. `LoadAgentWrapper.step`
    builds its auto-reset branch from `self._env.reset(key_reset)`
    (`aht.py:888`) -- the INNER env -- and then selects between that and the
    stepped state (`:900-902`). Seeding only on our outer `reset` would fix
    the initial carry and still mismatch inside that select on the first
    episode boundary. Seeding on the inner env fixes both, because
    `ScratchItch.step` does `state.metrics.update(...)`
    (`scratchitch.py:247`), so the slots survive a step untouched and our
    outer `step` then writes the real values over them.
    """
    class _SeedsBirdMetricSlots(type(base_env)):           # type: ignore[misc]
        pass

    import copy

    wrapped = copy.copy(base_env)
    original_reset = base_env.reset

    def reset(key: Any) -> Any:
        import jax.numpy as jnp                            # noqa: WPS433

        obs, state = original_reset(key)
        seeds = {"bird_prev_contact_force": jnp.float32(BIRD_METRIC_SEEDS[
                     "bird_prev_contact_force"]),
                 "bird_ag_idx": jnp.int32(BIRD_METRIC_SEEDS["bird_ag_idx"])}
        return obs, state.replace(metrics={**state.metrics, **seeds})

    wrapped.reset = reset
    return wrapped


def _human_only_pref_obs_class(base: Optional[type] = None) -> type:
    """Build the `LoadAgentWrapper` subclass. Imported lazily; see module doc.

    `base` EXISTS FOR THE TEST. Without it this function imports assistax, so
    every test of it must `importorskip` and therefore SKIPS in the offline
    suite, and marker mutants (dropping the obs marker, setting the reward
    marker on the eval class) would survive a green run because the only test
    asserting those cells never executes. A guard whose test is deselected
    everywhere is a guard nothing holds.

    Two overrides, and BOTH are needed:

    `_append_pref_to_obs` stops the concatenation reaching the robot at run
    time. `observation_spaces` stops the DECLARED width claiming otherwise: a
    data structure nobody reads today is read by the first refactor tomorrow,
    and a claim in a field is still a claim. Upstream widens the
    space for every agent at `aht.py:590-596`; leaving that would have the
    robot's declared width disagree with the vector it actually receives, which
    is the shape of bug that surfaces as a silent broadcast much later.
    """
    if base is None:
        from assistax.wrappers.aht import LoadAgentWrapper   # noqa: WPS433

        base = LoadAgentWrapper

    class _HumanOnlyPrefObs(base):                       # type: ignore[misc,valid-type]
        """Augment only the loaded (frozen) agents' observations."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # SEED THE SLOTS ON THE INNER ENV BEFORE super() TAKES IT. The
            # auto-reset branch inside `LoadAgentWrapper.step` resets THIS
            # object (`aht.py:888`), so the seeding has to be in place before
            # the wrapper captures it.
            if args:
                args = (_metric_slot_env(args[0]),) + tuple(args[1:])
            elif "env" in kwargs:
                kwargs = {**kwargs, "env": _metric_slot_env(kwargs["env"])}
            super().__init__(*args, **kwargs)
            # Undo upstream's all-agent widening for everyone we did not load.
            # Done here rather than by not calling super(): the widening is
            # entangled with `_num_pref_obs`, which the parent's reset/step
            # branch on, and we want that branch LIVE for the human.
            if self._num_pref_obs > 0:
                import copy
                spaces_now = dict(self._env.observation_spaces)
                for agent, space in spaces_now.items():
                    if agent in self.loaded_agents:
                        continue
                    narrowed = copy.copy(space)
                    narrowed.shape = (space.shape[0] - self._num_pref_obs,)
                    spaces_now[agent] = narrowed
                self._env.observation_spaces = spaces_now

        def step(self, key: Any, state: Any, actions: Dict[str, Any],
                 reset_state: Any = None) -> Any:
            """Upstream's step, plus two SCALARS on the state's metrics.

            WHY HERE AND NOT OFF `EvalInfo.env_state`: reading them off the
            logged env state costs 18 GB. `env_state=True` logs the whole
            `LoadAgentState` -- the entire MJX pipeline state -- for every
            step of every eval episode, and the scan carries it: XLA reported
            18.2 GB of program I/O and could not rematerialize below 16.9
            GiB, and the eval failed allocating 4.14 GiB on a 23 GB L4 GPU.
            "Logged by default" does not mean "free".

            `env_metrics` is the right channel: `ippo_ff_nps.py:866` takes it
            from the env state directly rather than from the LOGGED one, so
            it survives `env_state=False`, and it is a dict of scalars --
            which is how `total_pref_reward` already travels. `info` would
            not do: `:853` does `swapaxes(0, 1)` on every leaf, so a
            per-env scalar of shape (n_envs,) would raise.

            Prefixed `bird_` because these are OURS on upstream's state, and
            an unprefixed name could collide with a future upstream metric
            and be silently read as one.
            """
            obs, states, rewards, dones, infos = super().step(
                key, state, actions, reset_state)
            import jax.numpy as jnp                      # noqa: WPS433

            # WRITES INTO SLOTS RESET ALREADY SEEDED -- never adds a key. The
            # dtypes match `BIRD_METRIC_SEEDS`' because a scan carry compares
            # leaves, not just structure.
            inner = states._state
            cf = states.prev_contact_force
            drawn = (states.ag_idx or {}).get(self.loaded_agents[0]) \
                if isinstance(states.ag_idx, dict) else None
            extra = {
                "bird_prev_contact_force": jnp.float32(
                    0.0 if cf is None else cf),
                "bird_ag_idx": jnp.int32(
                    BIRD_METRIC_SEEDS["bird_ag_idx"] if drawn is None else drawn),
            }
            states = states.replace(
                _state=inner.replace(metrics={**inner.metrics, **extra}))
            return obs, states, rewards, dones, infos

        def _append_pref_to_obs(self, obs: Dict[str, Any],
                                ag_idx: Dict[str, Any]) -> Dict[str, Any]:
            # Imported HERE, not at class-build time: building the class must
            # work without jax so the marker asymmetry can be tested offline
            # through the `base` seam. An eager import would put jax on the
            # path of merely DESCRIBING the class, and that test would skip.
            import jax.numpy as jnp                      # noqa: WPS433

            pref_vec = self._get_pref_obs_vector(ag_idx)
            return {
                agent: (jnp.concatenate([o, pref_vec])
                        if agent in self.loaded_agents else o)
                for agent, o in obs.items()
            }

    # THE MARKER THE EVAL-SIDE GUARD READS. Separate from
    # `bird_candidate_rewarded`, and that separation is the point: the
    # evaluation env must carry the observation override and must NOT carry
    # the reward one, so one marker cannot serve both checks. Asserting
    # `bird_candidate_rewarded` on the eval env would refuse exactly the
    # correct configuration.
    _HumanOnlyPrefObs.bird_human_only_pref_obs = True
    return _HumanOnlyPrefObs


def robot_obs_width_is_unaugmented(env: Any, robot_agent: str) -> bool:
    """The held-out-observation check, as one assertion.

    True iff the robot's DECLARED observation width carries none of the seven
    preference dimensions. Exported rather than inlined so the preference-metric
    consumer asserts the same thing this backend promises, from the same source
    of truth -- two copies of one rule drift, and this one is the whole
    held-out guarantee.
    """
    human_w = env.observation_space(env.loaded_agents[0]).shape[0]
    robot_w = env.observation_space(robot_agent).shape[0]
    return (human_w - robot_w) == N_PREF_OBS


# --------------------------------------------------------------------------
# The BIRD state row the reward sees
# --------------------------------------------------------------------------

@register("train_backend", BACKEND_NAME)
def assistax_ppo_backend(ctx: Any, state: Any, candidate: Candidate,
                         n_seeds: int = 1,
                         env_steps: Optional[int] = None,
                         resume_ref: Optional[str] = None,
                         seed_phase: str = "",
                         handoff: Optional[Any] = None,
                         **backend_kw: Any) -> TrainResult:
    """Upstream Assistax IPPO with our reward. The robot learns; the human is frozen.

    `**backend_kw` is accepted, so a caller that forwards extra backend kwargs
    does not raise a `TypeError` naming this module for someone else's key.

    The call itself lives in `._ippo_train`, imported lazily: `registry.load_all()`
    imports this module for every config in the repo, including `--dry-run` and
    `--validate-all`, and `_ippo_train` reaches upstream's trainer, which imports
    jax. A module-level import would make a machine without the jax venv unable
    to validate a config it never intended to run -- the argument
    `training.sb3_backend` makes for `sb3`.
    """
    from . import _ippo_train                               # noqa: WPS433

    return _ippo_train.run(ctx, state, candidate, n_seeds=n_seeds,
                         env_steps=env_steps, resume_ref=resume_ref,
                         seed_phase=seed_phase,
                         handoff=handoff, **backend_kw)
