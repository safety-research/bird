"""Upstream Assistax, wrapped as BIRD EnvAdapters.

`upstream_assistax_{scratchitch,armmanipulation,feeding,teethbrushing}` (bed bathing: CPU port only).

WHAT THIS IS, AND WHAT IT IS NOT. This family runs UPSTREAM's assistax package
(pinned `a7d94f4e`, `scripts/setup_jax.sh`) -- upstream's solver, upstream's
force definitions, upstream's reward, upstream's termination. It does NOT
carry the CPU tier's (`bird/envs/assistax.py`) own tool-human force, so it
cannot answer a CPU-vs-MJX solver-parity question, and this family is excluded
from such comparisons by design.

================================ WHAT IS FAITHFUL AND WHAT IS NOT ==============

FAITHFUL, and the reason the family exists:
  * `reference_reward` is upstream's own -- `state.reward`, the scalar the env
    paid. NOT re-derived and NOT summed from `state.metrics`. THE METRICS DO
    NOT SUM TO THE REWARD except on bedbathing: `bedbathing.py:252-256` stores
    WEIGHTED components, but `feeding.py:289-295` stores UNWEIGHTED ones while
    the reward applies the weights separately (`:282-286`), teethbrushing logs
    a `reward_force` that is not in its reward at all, and armmanipulation
    carries both raw and `weighted_*` keys. Summing would have been right on
    one task of four and silently wrong on three.
  * the wipe latch and termination on bedbathing are upstream's
    (`state.info["contact_vector"]`, `bedbathing.py:380-390`; episode ends on
    the last point, `:250`). The other four never terminate -- `done = 0.0`
    unconditionally -- so their horizon is the wrapper's.
  * the tail's formulas and field names are the CPU module's, IMPORTED from
    `bird.envs.assistax`, so the prompt's field table is unchanged and no
    formula is transcribed twice.

NOT FAITHFUL, deliberately (see `bird/components/assistax_ppo.py` for the
held-out preference argument in full):
  1. upstream's preference mode is not run for the robot; R_pref is computed
     outside the training graph as a held-out metric.
  2. the robot observes 7 fewer dimensions than PPO_AHT under dynamic
     preferences. MEASURED: robot 29 unaugmented, 36 under upstream's
     `LoadAgentWrapper` with `pref_configs`, 29 with this adapter's override; human
     40/47/47.

INHERITED UPSTREAM DEFECTS -- recorded, not fixed, because faithfulness is the
point. A reader must know these before trusting a number from this family:
  * `armmanipulation.py:254` adds `+ r_rot * self._rot_scale` where `r_rot` is
    the Frobenius norm of a rotation-matrix DIFFERENCE (`:248`) -- it rewards
    angular MISALIGNMENT. Upstream flags it: `# TEMPORARY: NEED TO FIX
    ROTATION!` (`:311`). That task's `era_s` ground truth is questionable.
  * `armmanipulation.py:124-125` swaps the joint limits
    (`upper = jnt_range[:, 0]`), inverting its observation normalisation.
  * `teethbrushing.py:101` -- `TOOTHBRUSH_HEAD_CONTACT_ID = 19 #TODO: update
    these`. Its force channel may read the wrong contacts.
An upstream fix would be a new pin, never an edit of vendored code.

REFS/ IS FOR READING, NEVER FOR RUNNING: `scripts/fetch_refs.sh` prunes
the 56 Franka Panda mesh files (33 MB, third-party) by glob, so
`refs/code/assistax` cannot be instantiated. This adapter imports the
pip-installed package at the same pinned commit.
"""

from __future__ import annotations

from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import os

import numpy as np

from ..registry import register
from .base import EnvAdapter
from .spec import SpecEnvAdapter

#: One list, so the registrations and every consumer cannot disagree about the
#: family's members.
UPSTREAM_TASKS: Tuple[str, ...] = (
    "scratchitch", "bedbathing", "armmanipulation", "feeding", "teethbrushing",
)

#: Only bedbathing ends an episode on its own terms; the other four hardcode
#: `done = 0.0` and the horizon is the episode wrapper's. Stated because the
#: CPU tier (`bird/envs/assistax.py`) runs full 1000-step episodes on every
#: task, so this is a change on bedbathing and a faithfulness gain on the other four.
TERMINATES: Tuple[str, ...] = ("bedbathing",)


class UpstreamReferenceUnavailable(RuntimeError):
    """`reference_reward` asked for a state this adapter never stepped into.

    DELIBERATELY NOT A SUBCLASS OF `NotImplementedError`. Every consumer
    catches that specifically -- the rollouts in `training.py` and
    `policy_api.run_episode` -- and reads it as "this task has no reference
    reward", setting `gt_return = None`. On this family `has_reference_reward`
    is True and `era_s`'s `evaluate.fitness.source: native` reads `gt_return`,
    so being caught would silently remove the fitness signal with nothing red.
    A foreign state is a programming error and must crash.
    """


class _ReferenceCache:
    """Upstream's paid reward, keyed by the transition that produced it.

    WHY A CACHE AND NOT A FUNCTION. Upstream computes its reward INLINE inside
    `step` (`bedbathing.py:248`, `feeding.py:282-286`, ...); there is no
    `_reward()` to call on an arbitrary state. So the only faithful reference
    is the scalar the env actually paid, recorded as it is paid.

    WHY CONTENT-KEYED AND NOT IDENTITY-KEYED. Consumers replay STORED
    trajectories rather than asking about the last step: `policy_api.py` walks
    `traj[k+1]`, `training.py` walks `states[i+1]`, and an offline probe may
    zip stored states. An identity-matched cache refuses all of them.

    KEY AMBIGUITY, STATED RATHER THAN CODED AROUND. Upstream's reward is not a
    pure function of `(s2, a)` -- bedbathing's `new_contacts` is a DELTA
    against the previous contact_vector (`:244-246`) and the preference touch
    penalty is a rising edge on `prev_contact_force`. Where the caller gives us
    the predecessor we key on `(prev, a, s2)`; the two-argument
    `reference_reward(s2, a)` signature cannot, so it keys on `(s2, a)` and, if
    one episode visited the same `(s2, a)` twice from different predecessors,
    returns the first visit's value. Two bit-identical float64 (qpos, qvel,
    slots, tail) rows from different predecessors inside one episode of a
    contact-rich sim is not a case worth code, but it is worth this sentence.

    LIFETIME. The replay consumers walk trajectories AFTER the episode, so the
    cache outlives it: every stepped transition is kept until an explicit
    `clear()`, bounded by an LRU. An eviction-caused miss says so, distinctly
    from never-stepped, because the two call for different fixes.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive; an unbounded cache on a "
                             "long rollout is a leak, and a zero one refuses "
                             "every legitimate replay")
        self._cap = int(capacity)
        self._d: "OrderedDict[bytes, float]" = OrderedDict()
        self._evicted = 0

    @staticmethod
    def _key(s2: np.ndarray, a: Optional[np.ndarray],
             prev: Optional[np.ndarray] = None) -> bytes:
        # float64 C-contiguous bytes. `.tobytes()` on a non-contiguous view
        # would key on a different layout for the same values, so ascontiguous
        # first -- a silent miss is the failure mode this avoids.
        parts = [np.ascontiguousarray(s2, dtype=np.float64).tobytes()]
        parts.append(b"" if a is None
                     else np.ascontiguousarray(a, dtype=np.float64).tobytes())
        if prev is not None:
            parts.append(np.ascontiguousarray(prev, dtype=np.float64).tobytes())
        return b"|".join(parts)

    def put(self, s2: np.ndarray, a: Optional[np.ndarray], reward: float,
            prev: Optional[np.ndarray] = None) -> None:
        # Stored under BOTH keys: the caller that has the predecessor gets the
        # precise answer, the two-argument signature still resolves.
        for k in ({self._key(s2, a), self._key(s2, a, prev)} if prev is not None
                  else {self._key(s2, a)}):
            self._d[k] = float(reward)
            self._d.move_to_end(k)
        while len(self._d) > self._cap:
            self._d.popitem(last=False)
            self._evicted += 1

    def get(self, s2: np.ndarray, a: Optional[np.ndarray],
            prev: Optional[np.ndarray] = None) -> float:
        for k in ([self._key(s2, a, prev)] if prev is not None else []) + \
                 [self._key(s2, a)]:
            if k in self._d:
                self._d.move_to_end(k)
                return self._d[k]
        raise UpstreamReferenceUnavailable(
            "no recorded upstream reward for this (state, action)"
            + (f" -- {self._evicted} transition(s) have been EVICTED from a cache "
               f"of {self._cap}, so this may be a capacity problem rather than a "
               f"foreign state; raise the capacity to (eval episodes x horizon) "
               f"with headroom" if self._evicted else
               " -- and nothing has been evicted, so this state was NEVER STEPPED "
               "by this adapter. reference_reward on this family answers only for "
               "transitions it recorded; it never re-derives.")
        )

    def clear(self) -> None:
        self._d.clear()
        self._evicted = 0

    def __len__(self) -> int:
        return len(self._d)


class UpstreamAssistax(SpecEnvAdapter):
    """Base adapter over one upstream assistax task. Subclasses name the task.

    NOTHING JAX OR ASSISTAX IS IMPORTED AT MODULE SCOPE: a module-scope GPU
    import breaks every config load on every machine, including machines that
    will never run this family. Construction imports; import of this file does
    not.
    """

    #: Declared, and it must be. `xla_env.prepare_for_env` keeps the strict call
    #: for an UNDECLARED adapter on purpose (an MJX adapter whose author forgot
    #: the attribute would otherwise run non-reproducibly with nothing red), and
    #: `tests/test_jax_toy.py` makes `None` unreachable for jax-suite envs.
    #: Declared True: this family steps MJX on the device.
    requires_cuda = True

    #: FALSE, and this is not an oversight on a jax-suite env. `batched`
    #: describes THE ADAPTER the registry holds, and this one steps upstream's
    #: env object one state at a time through `_step` -- there is no
    #: `step_batch`, and it is not a `BatchedEnvAdapter`. The batching in this
    #: family lives in the BACKEND, which vmaps upstream's env inside the
    #: jitted train step; that is the backend's property, not the adapter's.
    #:
    #: `jax_base.BatchedEnvAdapter.batched` is explicit that these are two
    #: different questions -- `jax_toy` IS batched and does NOT require cuda;
    #: this family is the opposite pairing, requires cuda and is NOT batched.
    #: Declaring True here to "match the suite" would be inferring the answer
    #: from the id, which is exactly the mistake that comment warns against.
    batched = False

    task: str = ""
    #: Frame geometry and camera for `render`. Upstream's scenes ship two
    #: cameras (`sys.ncam == 2`); -1 is the free camera, which is the only
    #: choice that cannot fail on a task whose camera list differs.
    render_height: int = 256
    render_width: int = 256
    render_camera: int = -1
    #: THE INNER ENV'S ATTRIBUTE NAMING THE TOOL REFERENCE SITE INDEX -- the site
    #: whose finite difference is `tool_vel` (`_step`, `row_jax`). Upstream names
    #: it per task (`panda_scratcher_tip_idx` on scratchitch, `panda_spoon_centre`
    #: on feeding, ...), so the base class holds the NAME and each subclass sets
    #: it; hardcoding scratchitch's here would make every other task's tool
    #: velocity read the scratcher-tip site that does not exist in its model.
    tool_site_attr: str = "panda_scratcher_tip_idx"
    #: `_get_robo_obs(pipeline_state, info)` on scratchitch/bedbathing (the target
    #: depends on `info`), `_get_robo_obs(pipeline_state)` on feeding/teethbrushing/
    #: armmanipulation (measured off upstream a7d94f4e). One name, two arities,
    #: so the subclass says which rather than a `try/except TypeError` that would
    #: also swallow a real TypeError inside upstream's function.
    robo_obs_takes_info: bool = True
    #: HOMOGENISED, because the zoo is. Every upstream zoo config carries
    #: `ENV_KWARGS.homogenisation_method: max`, so the partner policies were
    #: trained on observations and actions PADDED TO THE TEAM MAXIMUM, and
    #: `LoadAgentWrapper` can only load one into an env built the same way --
    #: a bare env would hand a 3-wide human slot to a 7-wide policy and
    #: mis-index rather than raise (read off the zoo configs' headers).
    #:
    #: Measured here, scratchitch: bare gives robot 29 / human 40 / act 7 and 3;
    #: homogenised gives robot 42 / human 42 / act 7 and 7. The zoo headers read
    #: 49-in, and 49 - 42 = 7 -- exactly the preference dims `LoadAgentWrapper`
    #: appends. So the three numbers reconcile as base 42 + pref 7, and neither
    #: 49 nor 40 was ever the same quantity as the other.
    #:
    #: THIS DOES NOT CHANGE WHAT THE CANDIDATE SEES. The row this adapter emits
    #: is `concat(qpos, qvel, slots, tail)` computed from the physics, not from
    #: upstream's per-agent observation dict, so homogenisation moves the agent
    #: spaces and leaves the reward's observation byte-identical -- asserted by
    #: `test_homogenisation_does_not_move_the_row`, because "it shouldn't
    #: matter" is exactly the kind of claim that turns out to matter.
    homogenisation_method: str = "max"

    #: AUTO-RESET OFF, because BIRD owns the episode boundary. With upstream's
    #: default (`True`) the `AutoResetWrapper` swaps in the NEXT episode's
    #: state on the step that terminates, so the row, `done` and `success`
    #: this adapter banks for the terminal step describe a fresh episode
    #: rather than the one that just ended -- the last transition of every
    #: episode silently belongs to the next. `EnvAdapter.reset` is the reset
    #: here, so upstream's is redundant as well as wrong for this adapter. Turning it
    #: off also makes the inner env an `EpisodeWrapper` rather than an
    #: `AutoResetWrapper`; both proxy the per-task API `_inner` reads.
    auto_reset: bool = False

    #: UPSTREAM'S OWN ENV_KWARGS for the tuned baseline, cited rather than
    #: guessed: `assistax/baselines/IPPO/config/ippo.yaml` at a7d94f4e. The
    #: zoo partners were trained under these, so constructing with anything
    #: else puts the ego robot in a different environment from the one its
    #: partner learned in.
    #:
    #: `ctrl_cost_weight: 0` IS UPSTREAM'S VALUE and it is not the
    #: constructor's default (1e-6). Taking the constructor default would
    #: quietly add a control penalty upstream's own tuned runs did not pay,
    #: to the reference reward this tier exists to reproduce.
    #:
    #: `disability` and `preference_rewards` are COMMENTED OUT in that file,
    #: so upstream's baseline runs with neither and so does this adapter; the preference
    #: reward reaches us through `LoadAgentWrapper` instead, which is where
    #: the AHT protocol puts it.
    #: IMMUTABLE, and not defensively. This is a plain class attribute shared
    #: by all five task subclasses, so the natural future edit -- "feeding
    #: needs a different ctrl_cost" -- would mutate it in place and silently
    #: change the other four. A cross-task fidelity divergence that nothing
    #: reports is the worst bug shape this tier has; a subclass that needs
    #: different kwargs must REBIND the name, which is visible in a diff.
    upstream_env_kwargs: Mapping[str, Any] = MappingProxyType({
        "ctrl_cost_weight": 0,
        "backend": "mjx",
        "het_reward": False,
        "episode_length": 1000,
    })

    #: THE ZOO PARTNER. When both are set the env is wrapped in upstream's
    #: `LoadAgentWrapper`, which is the AHT protocol: it REMOVES `human` from
    #: `env.agents` and supplies the partner action itself, resampling the
    #: population index per episode (`reset_agent_index`). Left None, the
    #: partner is `_partner_action`'s zero -- a PASSIVE human, which is a
    #: different experiment and is why an AHT run must set these.
    #: `zoo_partners` is `PartnerSplit.train` from
    #: `bird.envs.assistax_zoo.split_partners`, shaped
    #: `{algorithm: {"human": [uuid, ...]}}`; the held-out uuids must never
    #: reach it, since they are the R^pref metric's population.
    zoo_path: Optional[str] = None
    zoo_partners: Optional[Dict[str, Any]] = None
    #: THE FULL `PartnerSplit` the adapter derived (seed, train, heldout,
    #: pool_size, index_sha256), when it derived one. `zoo_partners` is its
    #: `.train`. Carried so the trainer's seed row records the split's own
    #: facts (seed, sizes, index hash) rather than reading them off a view
    #: that does not have them -- a view without `seed` fails in the seed-row
    #: writer. None when the caller set `zoo_partners` by hand.
    zoo_split: Any = None
    #: Default capacity: 64 episodes x 1000 steps, upstream's episode_length
    #: (`envs/__init__.py:43`), with headroom. Overridable per run; an eviction
    #: says so in the error rather than looking like a foreign state.
    reference_cache_capacity: int = 64 * 1000

    def __init__(self, ctx: Any = None) -> None:
        # SUPER LAST, and that ordering is load-bearing rather than stylistic:
        # `EnvAdapter.__init__` calls `self._bounds()`, which reads `self._env`.
        # Calling super() first therefore raises `AttributeError: no attribute
        # '_env'` on EVERY construction. `scripts/derive_jax_spec.py --check`
        # constructs the env for its anchor measurement and so exercises this;
        # `--write` never constructs, and a generator that does not exercise
        # what it describes is not a check on it.
        if self.task not in UPSTREAM_TASKS:
            raise ValueError(
                f"{type(self).__name__}.task is {self.task!r}; the family is "
                f"{UPSTREAM_TASKS}. A subclass that names no task would wrap "
                f"whatever assistax.make defaulted to.")
        # BEFORE the import, because it is a configuration error and those
        # should not need a jax venv to surface. It also keeps the check
        # testable without the package installed.
        # THE ZOO COMES FROM THE ENVIRONMENT, NOT THE CONFIG, and the
        # asymmetry is the point rather than an omission. `zoo_path` is an
        # EXECUTION fact -- a per-machine path -- so a config key holding it
        # would fork ONE EXPERIMENT'S HASH ACROSS MACHINES. `BIRD_ZOO_PATH`
        # keeps it out of the hash, which is where a per-machine path belongs.
        #
        # The PROTOCOL half stays in the config and is hashed: `seed` (with
        # `assistax_zoo.DEFAULT_N_TRAIN`) is what `split_partners` is
        # deterministic in, so the 315 uuids are DERIVED here rather than
        # written down. A literal `zoo_partners` in a config would be 315 uuids
        # in the hash and a second copy of a fact the zoo already holds.
        #
        # Only fills a None, so a caller that sets the attribute directly
        # still wins.
        if self.zoo_path is None:
            self.zoo_path = os.environ.get("BIRD_ZOO_PATH") or None
        if self.zoo_path is not None and self.zoo_partners is None:
            from .assistax_zoo import DEFAULT_N_TRAIN, split_partners  # noqa: WPS433
            _cfg = getattr(ctx, "cfg", None)
            def _get(key, default):
                try:
                    return _cfg.get(key, default) if _cfg is not None else default
                except Exception:                          # noqa: BLE001
                    return default
            self.zoo_split = split_partners(
                str(self.zoo_path), str(self.task),
                seed=int(_get("seed", 0) or 0),
                n_train=DEFAULT_N_TRAIN)
            self.zoo_partners = self.zoo_split.train
        if (self.zoo_path is None) != (self.zoo_partners is None):
            raise ValueError(
                "zoo_path and zoo_partners must be set together: one without "
                "the other would silently fall back to the zero partner, which "
                "is a different experiment reported under the same name.")
        import assistax                                   # noqa: WPS433
        # `auto_reset` is this adapter's, not upstream's: it is absent from
        # upstream's ENV_KWARGS (so upstream runs with True) and is turned off
        # because `EnvAdapter` owns the episode boundary. Recorded as a deviation
        # rather than folded in with the cited values.
        self._env = assistax.make(self.task, auto_reset=self.auto_reset,
                                  homogenisation_method=self.homogenisation_method,
                                  **self.upstream_env_kwargs)
        if self.zoo_path is not None:
            from assistax.wrappers.aht import LoadAgentWrapper   # noqa: WPS433
            self._env = LoadAgentWrapper.load_from_zoo(
                self._env, self.zoo_path, self.zoo_partners)
        self._ref = _ReferenceCache(self.reference_cache_capacity)
        # The CPU module's tail spec, imported rather than restated, so this
        # tier and the CPU tier cannot disagree about which field sits where.
        from . import assistax as _cpu                    # noqa: WPS433
        self._cpu_mod = _cpu
        self._tail = self._tail_spec(_cpu)
        self._last_state: Any = None
        self._obs: Any = None
        self._key: Any = None
        # THE INNER ENV. `assistax.make` returns a multi-agent wrapper
        # (`assistax.envs.base_env.ScratchItch`) around an `AutoResetWrapper`
        # around the real task env; the wrapper proxies by attribute, and the
        # per-task API this adapter reads -- `_get_robo_obs`,
        # `panda_scratcher_tip_idx`, `dt` -- lives on the INNER one only.
        self._inner = self._env.env
        # `dt` IS 0.008, and it is read rather than computed. The computed
        # form `action_repeat * sys.opt.timestep` gives 1 x 0.002 = 0.002 --
        # FOUR TIMES TOO FAST, because the env takes four physics substeps per
        # control step. That error does not raise: it silently quadruples
        # `tool_speed`, which is an input to the held-out preference reward,
        # so every r_pref would be wrong by a clean factor and nothing would
        # flag it.
        self._dt = float(self._inner.dt)
        # DESCRIPTIVE ATTRIBUTES, set before `super().__init__()` reads them.
        # The base defaults are `obs_dim = 0` and `horizon = 1`
        # (`EnvAdapter`), and an unset `horizon` makes `describe()` say
        # "an episode lasts at most 1 steps" and a host rollout stop after one
        # step -- a wrong number that looks like a working adapter.
        self.name = f"upstream_assistax_{self.task}"
        self.horizon = int(self._inner.episode_length)
        sys_ = self._env.sys
        # The EGO's width. `_apply_spec` cross-checks it against the spec's
        # `action_dim`; the class default is 1, and nothing else on this tier
        # would read it.
        self.action_dim = int(self._env.action_space(self.EGO).shape[0])
        self.obs_dim = (int(sys_.q_size()) + int(sys_.qd_size())
                        + _N_SLOTS.get(self.task, 0)
                        + len(self._tail))
        # THE SPEC, APPLIED. Without `tasks/<id>/shared_spec.yaml`, `describe()`
        # renders "State variables, in order:" followed by NOTHING, and
        # `generation.py` builds the prompt from exactly that: every LLM
        # candidate on this tier would be written blind, against a 357-symbol
        # gate protecting an observation the model was never shown.
        # `state_surface.flat_fields` carries all 88 entries. The CPU tier does
        # the same (`bird/envs/assistax.py` subclasses `SpecEnvAdapter` and
        # applies it).
        #
        # AFTER the descriptive attributes and BEFORE `super().__init__()`.
        # `_apply_spec` CROSS-CHECKS the spec against the class (horizon must
        # agree), so it has to run once `horizon` is real -- placed earlier it
        # fires on the class default of 1, which is the guard working.
        # `super().__init__()` then reads `_bounds()`.
        from ..tasks import index as _task_index          # noqa: WPS433
        self._apply_spec(_task_index()[f"upstream_assistax_{self.task}"])

        super().__init__()


    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        import jax                                        # noqa: WPS433
        key = jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))
        # `reset` returns `(obs_dict, State)`, not a State: upstream is a
        # multi-agent env and the per-agent observation dict comes back beside
        # the state. Binding the tuple to `st` would feed a 2-tuple to `_row`.
        self._obs, st = self._env.reset(key)
        self._last_state = st
        self._key = key
        self._tool_vel = np.zeros(3, dtype=np.float64)
        # THE CACHE IS NOT CLEARED HERE. `reference_reward` is called by
        # REPLAY callers -- evaluation walks a stored trajectory after the
        # episode is over -- so clearing on reset would empty the cache
        # exactly when the rows that need it are about to be looked up, and
        # every lookup would raise `UpstreamReferenceUnavailable` for a state
        # that really was stepped. The cache is content-keyed and LRU-bounded,
        # so entries from a previous episode are addressed by their own
        # (s, a) and cannot be mistaken for this one's.
        return self._row(st)

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        import jax                                        # noqa: WPS433
        prev = self._last_state
        if prev is None:
            raise RuntimeError(
                "_step before _reset: this adapter steps upstream's env object, "
                "which carries the episode, so there is no state to step FROM. "
                "The base contract passes `s` but upstream's step takes its own "
                "State; they are reconciled by the adapter holding the episode, "
                "and a caller that skipped reset would otherwise step a stale one.")
        # rng FIRST -- assistax's signature is `step(rng, state, action)`
        # (`scratchitch.py:197`), NOT brax's `step(state, action)`. It is
        # accepted and unused in every one of the five envs, but the position
        # matters and getting it wrong passes an action as a state.
        self._key, key = jax.random.split(self._key)
        # A DICT KEYED BY AGENT, not a bare array: upstream's `step` takes
        # `actions: Dict[str, Array]` over `env.agents`. Passing the ego array
        # alone does not raise -- it indexes as a mapping would -- so this is
        # the silent-corruption shape, not the loud one.
        # Only the agents the env still exposes. Under `LoadAgentWrapper`
        # that is the ego alone -- the wrapper removes `human` from `agents`
        # and injects the partner's action itself -- so passing a PARTNER key
        # here would be overwritten by `{**state.load_agent_actions, **actions}`
        # in one direction and collide in the other.
        actions = {self.EGO: a}
        if self.PARTNER in self._env.agents:
            actions[self.PARTNER] = self._partner_action(self._obs, key)
        # Five returns, not a State: `(obs, state, reward, done, info)`.
        self._obs, st, rew, done_d, _info = self._env.step(key, prev, actions)

        # `panda_scratcher_tip_idx` is on the INNER env (0 here), not on the
        # wrapper. Looking on the wrapper (`getattr(self._env, ..., None)`)
        # finds nothing and falls through to a zero tool velocity on every
        # step -- a plausible number, reported forever, feeding the preference
        # reward's speed term.
        idx = int(getattr(self._inner, self.tool_site_attr))
        prev_tip = np.asarray(self._base(prev).pipeline_state.site_xpos, dtype=np.float64)
        now_tip = np.asarray(self._base(st).pipeline_state.site_xpos, dtype=np.float64)
        self._tool_vel = (now_tip[idx] - prev_tip[idx]) / self._dt

        row = self._row(st)
        # UPSTREAM'S PAID REWARD, recorded as it is paid. Never summed from
        # `state.metrics` -- see the module docstring; the metrics are
        # unweighted on three of the five tasks.
        # `rew`/`done` are DICTS over agents plus `__all__`. The reward is
        # shared on every one of the five tasks (measured: `rew["robot"] ==
        # rew["human"] == rew["__all__"]`), so `__all__` is the paid scalar and
        # not a reduction this adapter chose.
        #
        # R_PREF IS SUBTRACTED BACK OUT, and this is the held-out design, not
        # bookkeeping. Under `LoadAgentWrapper` upstream adds the partner's
        # preference reward to EVERY agent's reward --
        # `rewards = {agent: rewards[agent] + pref_reward for agent in rewards}`,
        # `assistax/wrappers/aht.py:881` -- so `rew["__all__"]` is
        # `task + R_pref` the moment a zoo partner is loaded, and is the plain
        # task reward when one is not. Banking that as `reference_reward` would
        # make the reference silently change meaning with the partner setting,
        # and would fold the HELD-OUT metric into the number the candidate is
        # compared against.
        #
        # The components are on `state.metrics` (the wrapper puts them there,
        # `aht.py:882`), so the split is upstream's own arithmetic reversed
        # with upstream's own number rather than a re-derivation of R_pref.
        # REFUSE TO BANK A CANDIDATE-REWARDED ENV. `reference_reward` means
        # "what upstream paid"; under `_ippo_reward_env`'s wrapper the ego's and
        # `__all__`'s entries are the CANDIDATE's reward, so banking either
        # would record the candidate as the reference -- the one number that
        # must never be the thing it is compared against. The measured
        # invariant (`rew["robot"] == rew["human"] == rew["__all__"]`) is
        # broken BY CONSTRUCTION when that wrapper is active, so the invariant
        # cannot be the guard.
        #
        # The training env and this adapter are separate instances and
        # the wrapper is bound only around `make_train`, so this should never
        # fire; it is here because "should never" is what the invariant said.
        if type(self._env).__name__.startswith("CandidateRewarded"):
            raise RuntimeError(
                "reference_reward would bank a CANDIDATE reward: this adapter's "
                "env is wrapped by _ippo_reward_env's candidate-reward wrapper, "
                "under which rew['robot'] and rew['__all__'] are the "
                "candidate's, not upstream's. Evaluate on an UNBOUND env -- the "
                "binding belongs around make_train only.")
        paid = float(np.asarray(rew["__all__"]).reshape(()))
        metrics = getattr(self._base(st), "metrics", None) or {}
        r_pref = float(np.asarray(metrics["total_pref_reward"]).reshape(())) \
            if "total_pref_reward" in metrics else 0.0
        self._ref.put(row, np.asarray(a, dtype=np.float64), paid - r_pref, prev=s)
        self._last_state = st
        done = bool(np.asarray(done_d["__all__"]).reshape(()) > 0.5)
        # The held-out channel, reported beside the step and never in the row:
        # the candidate cannot read `info`, and every symbol that names these
        # quantities is in the consumer gate. `r_pref` is ABSENT rather than
        # 0.0 when no partner is loaded -- a zero would be indistinguishable
        # from a partner who earned nothing, and those are different facts.
        info: Dict[str, Any] = {"success": self.success_from_state(st)}
        if "total_pref_reward" in metrics:
            info["r_pref"] = r_pref
            info["r_task"] = paid - r_pref
        return row, done, info

    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """Upstream's own reward for the transition that arrived in `s`.

        Raises `UpstreamReferenceUnavailable` -- deliberately NOT a
        `NotImplementedError` -- for a state this adapter never stepped into.
        """
        return self._ref.get(np.asarray(s, dtype=np.float64),
                             None if a is None else np.asarray(a, dtype=np.float64))

    def success_from_state(self, st: Any) -> bool:
        """Upstream's own notion, where it has one.

        Only bedbathing terminates on achievement (`done = all wiped`,
        `bedbathing.py:250`), so on the other four `done` is always 0.0 and
        success is NOT readable from it. Reporting `done` as success on those
        would make every episode a failure by construction and the number
        would look like a hard task rather than a missing signal.
        """
        if self.task in TERMINATES:
            return bool(np.asarray(st.done).reshape(()) > 0.5)
        return False

    def task_metric(self, traj: Any) -> float:
        """Ground truth F in [0,1]. bedbathing: upstream's WIPED FRACTION.

        `contact_vector` is 0 == wiped (`bedbathing.py:386-389`), so the
        fraction is `mean(cv == 0)` on the final row -- upstream's own count
        (`n_contacts = count_nonzero(new_contact_vector == 0)`, `:244`),
        not a reimplementation.

        The other four have no upstream success predicate at all. Their metric
        is the CPU module's check evaluated on the same fields, and it is
        PROVISIONAL exactly as on the CPU tier -- flagged here so a reader does
        not take a provisional number for upstream's.
        """
        rows = np.asarray(traj, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[None, :]
        if self.task == "bedbathing":
            nq = int(self._env.sys.q_size()); nv = int(self._env.sys.qd_size())
            cv = rows[-1, nq + nv:nq + nv + N_WIPE_POINTS]
            return float(np.mean(cv == 0.0))
        chk = self._cpu_mod._TASKS[self.task]["check"]
        hits = [int(bool(chk(self, self._row_derived(r)))) for r in rows]
        return float(np.mean(hits)) if hits else 0.0

    def _row_derived(self, row: np.ndarray) -> Any:
        """A `_Derived`-alike rebuilt from ONE STORED ROW, for the checks.

        NO RE-SIMULATION, and none is needed: every field the five checks read
        is already a TAIL COLUMN of the row, because the tail lambdas are the
        things that computed them in the first place. The CPU tier's
        `forward=True` path restores `(qpos, qvel)` and runs `mj_forward`
        because its `_Derived` is built from an `MjData`; here the derived
        quantities were banked at the step that produced them, so reading them
        back is both cheaper and CLOSER TO WHAT WAS PAID -- a restored solve
        cannot reproduce contact forces, which is the caveat the CPU
        `_Derived` docstring states about its own restore.

        Missing names raise rather than default. A check silently reading a
        zero for `tool_force_mag` scores every step a failure and looks like a
        bad policy, not a bad adapter.
        """
        names = [t[0] for t in self._tail]
        base = len(row) - len(names)

        def col(name: str) -> float:
            if name not in names:
                raise KeyError(
                    f"{self.task}: the check needs tail column {name!r}, which "
                    f"_TAIL_{self.task.upper()} does not carry. The tail is the "
                    f"CPU module's and is imported, never restated here, so this "
                    f"is a real gap rather than a transcription slip.")
            return float(row[base + names.index(name)])

        def mag(*cs: str) -> float:
            return float(np.linalg.norm([col(c) for c in cs]))

        # The distance column is `dist_tool_target` on the arm tasks and
        # `dist_tool_mouth` on the head tasks (the CPU tails name what the
        # distance is TO); the checks read it as `D.dist` on both.
        dist_cols = [nm for nm in names if nm.startswith("dist_tool_")]
        if len(dist_cols) != 1:
            raise KeyError(
                f"{self.task}: expected exactly one dist_tool_* tail column, "
                f"found {dist_cols}")
        fields: Dict[str, Any] = {nm: col(nm) for nm in names}
        fields.update(
            slots=row[base - _N_SLOTS.get(self.task, 0):base],
            dist=col(dist_cols[0]),
            tool_speed=mag("tool_vx", "tool_vy", "tool_vz"),
            tool_force_mag=mag("tool_fx", "tool_fy", "tool_fz"),
        )
        # `tip_force_mag` is a tail column on the head tasks and the check reads
        # it; on the arm tasks it is absent from both, so no default is planted.
        # armmanipulation's check reads `dist_forearm_waist`, banked under the
        # tail name `dist_forearm_waist_target`; the CPU `_row_derived` makes the
        # same alias (the column IS the quantity, under the tail's longer name).
        if "dist_forearm_waist_target" in names:
            fields["dist_forearm_waist"] = col("dist_forearm_waist_target")
        # teethbrushing's check reads `tangential_speed`, banked under the tail
        # name `brush_tangential_speed` (without the alias `task_metric` raises
        # AttributeError; `test_the_checks_inputs_are_all_banked_tail_columns`
        # holds every task's check to its `_row_derived`).
        if "brush_tangential_speed" in names:
            fields["tangential_speed"] = col("brush_tangential_speed")
        return _UpstreamDerived(**fields)

    def success(self, traj: Any) -> bool:
        """Any-step success. Upstream's on bedbathing, PROVISIONAL elsewhere."""
        if self.task in TERMINATES:
            return self.task_metric(traj) >= 1.0
        # UPSTREAM HAS NO success predicate on these four -- `done` is the
        # literal 0.0 in every step (e.g. `scratchitch.py:246`) -- so success
        # is the CPU tier's THRESHOLD ON THE METRIC, which is a different
        # claim and is labelled as one. It is the CPU row's own `threshold`
        # (0.3 on scratchitch), read rather than restated, so the two tiers
        # cannot disagree about where the line is.
        #
        # This does not raise. Refusing is right when the alternative is
        # inventing a signal, but it is wrong here: the CPU tier already
        # defines this number, `evaluate` calls `success()` on every episode,
        # and raising makes the tier unrunnable rather than honest. The
        # honesty belongs in the label -- PROVISIONAL, ours, not upstream's --
        # which is the same status it has on the CPU tier.
        thr = self._cpu_mod._TASKS[self.task].get("threshold")
        if thr is None:
            raise NotImplementedError(
                f"{self.task}: upstream defines no success predicate and the "
                f"CPU row carries no `threshold` to stand in for one.")
        return bool(self.task_metric(traj) >= float(thr))

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        """A COVERAGE draw, not an on-policy one (`EnvAdapter.random_state`).

        Without it `verification.sample_transitions` raises and every
        candidate fails the traced probe -- the whole population marked
        invalid for an adapter gap, which reads as a bad generator.

        Upstream gives no way to set an arbitrary state, so this resets and
        walks a random number of steps under random actions: the states are
        genuinely reachable, which a uniform draw over the box would not be,
        and the spread comes from the walk length. It is NOT the whole state
        space, and that is stated rather than implied -- EPIC/STARC on this
        tier are measured over reachable states only.
        """
        s = self._reset(rng)
        for _ in range(int(rng.integers(0, 50))):
            lo, hi = self.action_low, self.action_high
            s, done, _info = self._step(s, rng.uniform(lo, hi))
            if done:
                s = self._reset(rng)
        return s

    def _build_action_set(self) -> Any:
        """The ego action box as a two-row set, NOT None.

        `EnvAdapter.__init__` does `np.asarray(self._build_action_set())` then
        `.min(axis=0)` / `.max(axis=0)` to fill `action_low`/`action_high`.
        `None` becomes a 0-d object array and both reductions give NaN, so
        every candidate would be clipped against NaN bounds -- which does not
        raise, it silently passes everything through.

        Two rows, -1 and +1, are upstream's normalised box
        (`action_space(agent)` is `Box(-1, 1, (7,))` for the robot on all five
        tasks, measured), so min/max over them reproduce it exactly.
        """
        n = int(self._env.action_space(self.EGO).shape[0])
        return np.stack([-np.ones(n, dtype=np.float64),
                         np.ones(n, dtype=np.float64)])

    #: The agent BIRD controls. Upstream is a two-agent env
    #: (`env.agents == ["robot", "human"]`); the candidate's reward drives the
    #: ROBOT and the human is a frozen zoo partner (the AHT protocol). Named
    #: once here because four methods index by it and a literal in each is
    #: four places to get the split wrong.
    EGO = "robot"
    PARTNER = "human"

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """OBSERVATION bounds over the whole row, matching the CPU tier.

        `EnvAdapter.__init__` assigns this to `obs_low`/`obs_high`, so it covers
        `concat(qpos, qvel, slots, tail)` -- NOT the action box. Joint ranges
        come off `sys` exactly as the CPU tier reads them off `mj_model`, and
        the tail's bounds are the tail spec's own `(lo, hi)` rather than a
        restated copy.

        The ego ACTION width is a separate question, answered by
        `action_space(agent)`, NOT `action_size`: this env has no
        `action_size` at all -- that is brax's single-agent attribute and
        reading it raises `AttributeError`. Measured widths (assistax @
        a7d94f4e): robot 7 on all five tasks; human 3 on scratchitch/
        bedbathing/armmanipulation and 2 on feeding/teethbrushing. Taking the
        ego width from the space rather than a constant is what keeps the
        3-vs-2 split from ever needing to be known here.
        """
        sys_ = self._env.sys
        nq, nv = int(sys_.q_size()), int(sys_.qd_size())
        lo = np.full(nq + nv, -np.pi, dtype=np.float64)
        hi = np.full(nq + nv, np.pi, dtype=np.float64)
        for j in range(int(sys_.njnt)):
            adr = int(sys_.jnt_qposadr[j])
            if int(sys_.jnt_type[j]) == 0:                       # free joint
                lo[adr:adr + 3], hi[adr:adr + 3] = (-3.0, -3.0, -1.0), (3.0, 3.0, 3.0)
                lo[adr + 3:adr + 7], hi[adr + 3:adr + 7] = -1.0, 1.0
            elif int(sys_.jnt_limited[j]) and int(sys_.jnt_type[j]) == 3:
                rng = np.asarray(sys_.jnt_range[j], dtype=np.float64)
                lo[adr], hi[adr] = float(rng[0]) - 0.5, float(rng[1]) + 0.5
        lo[nq:nq + 3], hi[nq:nq + 3] = -10.0, 10.0               # root linear
        lo[nq + 3:nq + 6], hi[nq + 3:nq + 6] = -30.0, 30.0       # root angular
        n_slots = _N_SLOTS.get(self.task, 0)
        tail_lo = np.asarray([t[3] for t in self._tail], dtype=np.float64)
        tail_hi = np.asarray([t[4] for t in self._tail], dtype=np.float64)
        return (np.concatenate([lo, np.zeros(n_slots), tail_lo]),
                np.concatenate([hi, np.ones(n_slots), tail_hi]))

    def _partner_action(self, obs: Any, key: Any) -> Any:
        """The frozen partner's action for this step when no zoo partner is loaded.

        ZERO. Under `LoadAgentWrapper` (`zoo_path`/`zoo_partners` set, see
        `bird.envs.assistax_zoo.split_partners`) the wrapper supplies the
        partner action and this is not called. The zero is a declared fallback
        and not a silent default: a zero human is a PASSIVE human, which is a
        different experiment from upstream's AHT protocol, and any number
        produced under it is comparable only to itself.
        """
        import jax.numpy as jnp                           # noqa: WPS433
        return jnp.zeros(self._env.action_space(self.PARTNER).shape)

    def render(self, state: np.ndarray) -> np.ndarray:
        """CPU `mujoco.Renderer` from the row's qpos/qvel -- same MJCF.

        A FAILURE HERE MUST JOURNAL A SHORTFALL, NEVER FAIL THE RUN, and that
        is the caller's contract (`output.video.timeout_s` and `n_views` both
        behave this way). Under `output.video.record: all` this path runs for
        every candidate, not just the selected one, so a raising renderer would
        take a whole run down rather than one frame. This method therefore
        does the draw and lets the recorder catch; it does not raise on its
        own behalf.

        RUNS ON THE HOST, not the device: MJX has no renderer, VLM-judged
        configs need frames, and the scene is the same XML upstream compiled. The
        device holds the rollout; rendering restores one row into a CPU
        `MjData` and draws it, which keeps a frame comparable across tiers.
        """
        import mujoco                                     # noqa: WPS433
        model = self._env.sys.mj_model
        data = mujoco.MjData(model)
        row = np.asarray(state, dtype=np.float64)
        nq, nv = int(model.nq), int(model.nv)
        data.qpos[:] = row[:nq]
        data.qvel[:] = row[nq:nq + nv]
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, self.render_height, self.render_width) as r:
            r.update_scene(data, camera=self.render_camera)
            return np.asarray(r.render(), dtype=np.uint8)

    # -- the tail ----------------------------------------------------------

    def _tail_spec(self, cpu_mod: Any) -> Any:
        """`(name, doc, lambda D: value, lo, hi)` tuples, from the CPU module.

        Looked up by the CPU module's own naming convention rather than
        hardcoded here: re-listing the fields would make every future gap
        ambiguous between "upstream differs" and "someone transcribed a field
        twice", and is the reason the prompt's field table is stable.
        """
        name = f"_TAIL_{self.task.upper()}"
        spec = getattr(cpu_mod, name, None)
        if spec is None:
            raise AttributeError(
                f"{name} not found in bird.envs.assistax. The tail spec is "
                f"imported, never restated; if the CPU module renamed it, this "
                f"adapter must follow rather than grow its own copy.")
        return spec

    # -- state -------------------------------------------------------------

    @staticmethod
    def _base(st: Any) -> Any:
        """The real `State`, through `LoadAgentWrapper`'s `LoadAgentState`.

        The wrapper nests upstream's State under `_state` and carries the
        partner's action, hidden state and population index beside it. Every
        reader of `pipeline_state` / `metrics` / `info` goes through here, so
        wrapping the env cannot silently change what the row is built from --
        which is the one thing that must not differ between the zero-partner
        baseline and a sampled cell.
        """
        return getattr(st, "_state", st)

    def _slots(self, st: Any) -> np.ndarray:
        """The task's latching memory. Empty except on bedbathing.

        bedbathing's 52 wipe flags are UPSTREAM's `info["contact_vector"]`,
        read straight across: upstream's env already uses
        this latch for its own termination and the zoo partner was trained
        under it, so recomputing it from BIRD's force definition would put two
        disagreeing wipe counts inside one episode -- the env's, which ends it,
        and the reward row's, which the candidate reads. Polarity 0 == wiped,
        matching both `bedbathing.py:386-389` and the CPU module's
        `_adv_bedbathing`.
        """
        if self.task == "scratchitch":
            # THE FOUR EPISODE-CONSTANT TARGET SLOTS the spec documents at
            # s[61..64], read from upstream's own `info["scratch"]` rather
            # than recomputed: `{"arm": bool, "arm_geom_idx": int,
            # "pos": (3,)}`, where `pos` IS the itch in the target segment's
            # local frame. Without these the row is 84 wide while the spec and
            # the rendered prompt describe 88 with slots at 61..64, so every
            # index a candidate writes past 60 is off by four -- it reads a
            # joint velocity where it was told the target arm is. That is the
            # worst shape available: no error, a plausible number, and a
            # candidate blamed for it.
            scratch = self._base(st).info.get("scratch")
            if scratch is None:
                raise KeyError(
                    "scratchitch state carries no info['scratch']: the target "
                    "slots are upstream's and this adapter does not synthesise "
                    "them.")
            pos = np.asarray(scratch["pos"], dtype=np.float64).reshape(3)
            return np.concatenate([[float(np.asarray(scratch["arm"]).reshape(()))], pos])
        if self.task != "bedbathing":
            return np.zeros(0, dtype=np.float64)
        cv = self._base(st).info.get("contact_vector")
        if cv is None:
            raise KeyError(
                "bedbathing state carries no info['contact_vector']: the wipe "
                "latch is upstream's and this adapter does not synthesise one.")
        return np.asarray(cv, dtype=np.float64)

    def row_jax(self, st: Any, prev_st: Any = None) -> Any:
        """The same row, built in `jax.numpy`, TRACEABLE under `jit`.

        THE TRAINING LOOP NEEDS THIS AND `_row` CANNOT SERVE IT. `_row` calls
        `np.asarray`, which pulls a tracer to the host and raises inside
        `jit`; upstream's IPPO rollout is jitted and vmapped over 1024 envs,
        so the candidate's reward has to be computed on device from device
        values. Running the host row there would also be a per-step
        device-to-host round trip on every one of those envs.

        ONE FORMULA SET, TWO ARRAY MODULES -- not a second layout. The tail
        lambdas already dispatch through `D.xp` for exactly this reason (the
        CPU `_Derived` documents it: the MJX tier builds an equivalent with
        `xp = jax.numpy` and evaluates THE SAME lambdas), so this builds the
        same `_UpstreamDerived` with `xp` swapped and reads the same
        `_tail_spec`. If the two rows could disagree, the reward the learner
        maximises and the reward the artifact records would be different
        functions, which is the defect this tier is least able to detect.
        `test_the_jax_row_matches_the_host_row` pins them equal.
        """
        import jax.numpy as jnp                             # noqa: WPS433

        base = self._base(st)
        ps = base.pipeline_state
        # TOOL VELOCITY IS A FINITE DIFFERENCE and therefore needs the
        # PREVIOUS state; it is not a function of `st` alone. Hardcoded zeros
        # would make `tool_speed` -- an input to the checks and available to a
        # candidate through the tail -- permanently 0 on the learner's row
        # while the host row carried the real value: the learner would
        # maximise a different function from the one every artifact records,
        # invisibly. `test_the_jax_row_matches_the_host_row` holds this over
        # stepped rows; a comparison of reset rows alone would not, since both
        # are zero there.
        #
        # `prev_st is None` reproduces the host's reset state, where
        # `_tool_vel` is zeroed before any step has happened.
        idx = int(getattr(self._inner, self.tool_site_attr))
        if prev_st is None:
            tool_vel = jnp.zeros(3)
        else:
            prev_ps = self._base(prev_st).pipeline_state
            tool_vel = (jnp.asarray(ps.site_xpos)[idx]
                        - jnp.asarray(prev_ps.site_xpos)[idx]) / self._dt
        derived = self._derived_xp(base, self._slots_jax(base), tool_vel, jnp)
        tail = jnp.stack([jnp.asarray(t[2](derived)).reshape(()) for t in self._tail])
        return jnp.concatenate([jnp.asarray(ps.qpos).reshape(-1),
                                jnp.asarray(ps.qvel).reshape(-1),
                                self._slots_jax(base).reshape(-1),
                                tail])

    def _slots_jax(self, base: Any) -> Any:
        """`_slots` in jnp. Same sources, same polarity, no host conversion."""
        import jax.numpy as jnp                             # noqa: WPS433

        if self.task == "scratchitch":
            scratch = base.info["scratch"]
            return jnp.concatenate([
                jnp.asarray(scratch["arm"], dtype=jnp.float32).reshape(1),
                jnp.asarray(scratch["pos"]).reshape(3)])
        if self.task != "bedbathing":
            return jnp.zeros(0)
        return jnp.asarray(base.info["contact_vector"])

    def _row(self, st: Any) -> np.ndarray:
        """`concat(qpos, qvel, slots, tail)` -- the layout the prompt documents."""
        ps = self._base(st).pipeline_state
        qpos = np.asarray(ps.qpos, dtype=np.float64)
        qvel = np.asarray(ps.qvel, dtype=np.float64)
        slots = self._slots(st)
        D = self._derived(st, slots)
        tail = np.array([float(fn(D)) for (_n, _d, fn, _lo, _hi) in self._tail],
                        dtype=np.float64)
        return np.concatenate([qpos, qvel, slots, tail])

    def _derived(self, st: Any, slots: np.ndarray) -> Any:
        """A `_Derived` the CPU tail lambdas can read -- the host row's."""
        return self._derived_xp(self._base(st), np.asarray(slots, dtype=np.float64),
                                np.asarray(self._tool_vel, dtype=np.float64), np)

    def _robo_obs(self, base: Any) -> Dict[str, Any]:
        """Upstream's `_get_robo_obs`, with the arity the task's env has."""
        if self.robo_obs_takes_info:
            return self._inner._get_robo_obs(base.pipeline_state, base.info)
        return self._inner._get_robo_obs(base.pipeline_state)

    def _derived_xp(self, base: Any, slots: Any, tool_vel: Any, xp: Any) -> Any:
        """ONE FORMULA SET, TWO ARRAY MODULES: the `_UpstreamDerived` the task's
        tail lambdas read, built from upstream's own obs dicts with `xp` numpy
        (the host row, `_row`) or `jax.numpy` (the learner's row, `row_jax`).
        A subclass implements exactly this and nothing else per task, so the
        host and device rows cannot be built from two different mappings --
        `test_the_jax_row_matches_the_host_row_column_by_column` pins them equal
        and this is what makes that pin a test of physics rather than of
        transcription."""
        raise NotImplementedError(
            f"{type(self).__name__} must build the _UpstreamDerived its tail reads")


class _UpstreamDerived:
    """What the CPU tail lambdas read, sourced from UPSTREAM's own obs dicts.

    The CPU `_Derived` (`bird.envs.assistax._Derived`) computes these from a
    restored `mujoco.MjData`. Here every quantity already exists in upstream's
    `_get_robo_obs` / `_get_human_obs` output, so this tier MAPS rather than
    recomputes -- recomputing would reintroduce the CPU tier's force definition
    on a tier whose whole point is upstream's, and would make any disagreement
    ambiguous between "upstream differs" and "recomputed differently".

    `xp = np` matches the CPU `_Derived`'s class attribute, so a lambda that
    calls through `D.xp` gets numpy on this tier exactly as it does on the CPU
    one.
    """

    xp = np

    def __init__(self, **fields: Any) -> None:
        # `xp` arrives as a field on the jnp path (`row_jax`) and is left at
        # the class default on the host path, so ONE class serves both and the
        # tail lambdas cannot be evaluated against the wrong array module by
        # accident.
        for k, v in fields.items():
            setattr(self, k, v)


class UpstreamAssistaxScratchItch(UpstreamAssistax):
    """`upstream_assistax_scratchitch`.

    NO LATCHING, NO TERMINATION. `info["scratch"]` is episode-constant
    randomisation set at reset and never updated (`scratchitch.py:157-162`),
    and `done = 0.0` is a literal (`:246`) -- the horizon is the episode
    wrapper's, `episode_length=1000` by default.

    THE TAIL'S EIGHT INPUTS ARE ALL IN UPSTREAM'S OBS, bar the tool velocity.
    Verified against `_TAIL_SCRATCHITCH`, whose lambdas read exactly
    `dist, larm_pos, target_pos, tool_force, tool_pos, tool_quat, tool_vel,
    uarm_pos`. Seven map straight off `_get_robo_obs` (`scratchitch.py:266-286`);
    `tool_vel` is upstream's own finite difference of the scratcher-tip site
    over `self.dt` (`:233-236`), which is not in the obs dict and so is
    computed here from the two pipeline states -- the same expression, not a
    different one.
    """

    task = "scratchitch"
    tool_site_attr = "panda_scratcher_tip_idx"
    robo_obs_takes_info = True

    def _derived_xp(self, base: Any, slots: Any, tool_vel: Any, xp: Any) -> Any:
        ro = self._robo_obs(base)
        f6 = xp.asarray(ro["force_on_tool"]).reshape(-1)
        return _UpstreamDerived(
            xp=xp,
            slots=slots,
            tool_pos=xp.asarray(ro["tool_position"]),
            tool_quat=xp.asarray(ro["tool_orientation"]),
            target_pos=xp.asarray(ro["target_position"]),
            uarm_pos=xp.asarray(ro["human_uarm_pos"]),
            larm_pos=xp.asarray(ro["human_larm_pos"]),
            # `contact_force(..., to_world_frame=False)` returns a 6-vector
            # force:torque; the CPU `tool_force` is the 3-vector force, so the
            # torque half is dropped rather than silently normed into it.
            tool_force=f6[:3],
            tool_force_mag=xp.linalg.norm(f6[:3]),
            tool_vel=tool_vel,
            tool_speed=xp.linalg.norm(tool_vel),
            dist=xp.asarray(ro["distance_to_target"]).reshape(()),
        )


class UpstreamAssistaxFeeding(UpstreamAssistax):
    """`upstream_assistax_feeding`.

    NO SLOTS, NO LATCHING, NO TERMINATION: `done = 0.0` is a literal
    (`feeding.py:114`) and the state carries no episode memory, so the row is
    `concat(qpos, qvel, tail)` and `_N_SLOTS` has no entry for it.

    THE TAIL'S INPUTS (`_TAIL_FEEDING` = `_COMMON_TAIL` + mouth, distance,
    spoon level, tip force) map off upstream's `_get_robo_obs(pipeline_state)`
    (`feeding.py:310-326`; NO `info` argument on this task): `tool_position` is
    the `spoon_center` site, `tool_orientation` the `spoon` body quaternion,
    `target_position` the `mouth` site, `force_on_tool` the SUM of the two
    spoon contacts (bowl + right side, ids 186/187). Three are computed here
    with the CPU module's own formulas, never a second copy:
      * `dist` = |mouth - tool| (`_derive_feeding`, `assistax.py`); upstream's
        obs dict carries no distance on this task.
      * `spoon_up_dot_world_up` = `_quat_rotate(tool_quat, (-1, 0, 0))[2]`,
        the CPU definition ("the spoon's up is its local -x").
      * `tip_force_mag` = |force on upstream's `spoon_right_side` contact|
        (`_get_force_on_tool(ps, SPOON_RSIDE_CONTACT_ID)`), which is the CPU
        row's `tip_geoms=("spoon_right_side",)` against the head -- upstream's
        own contact, not the CPU tier's pair census.
    `tool_vel` is the finite difference of the `spoon_center` site over `dt`,
    the same expression as scratchitch's on its tip site.
    """

    task = "feeding"
    tool_site_attr = "panda_spoon_centre"
    robo_obs_takes_info = False

    def _derived_xp(self, base: Any, slots: Any, tool_vel: Any, xp: Any) -> Any:
        ro = self._robo_obs(base)
        ps = base.pipeline_state
        f6 = xp.asarray(ro["force_on_tool"]).reshape(-1)
        tip6 = xp.asarray(self._inner._get_force_on_tool(
            ps, self._inner.SPOON_RSIDE_CONTACT_ID)).reshape(-1)
        tool_pos = xp.asarray(ro["tool_position"]).reshape(3)
        tool_quat = xp.asarray(ro["tool_orientation"]).reshape(4)
        mouth_pos = xp.asarray(ro["target_position"]).reshape(3)
        spoon_up = self._cpu_mod._quat_rotate(tool_quat, xp.asarray([-1.0, 0.0, 0.0]), xp=xp)
        return _UpstreamDerived(
            xp=xp,
            slots=slots,
            tool_pos=tool_pos,
            tool_quat=tool_quat,
            mouth_pos=mouth_pos,
            target_pos=mouth_pos,
            tool_force=f6[:3],
            tool_force_mag=xp.linalg.norm(f6[:3]),
            tool_vel=tool_vel,
            tool_speed=xp.linalg.norm(tool_vel),
            dist=xp.linalg.norm(mouth_pos - tool_pos),
            spoon_up=spoon_up,
            spoon_up_dot_world_up=spoon_up[2],
            tip_force=tip6[:3],
            tip_force_mag=xp.linalg.norm(tip6[:3]),
        )


# -- the lifecycle, shared -------------------------------------------------


#: bedbathing's wipe-flag count; 0 == wiped. `n_targets` is a constructor
#: kwarg defaulting to 52 (`bedbathing.py:40`), so this is the default and an
#: adapter constructed with another value must read it off the env.
N_WIPE_POINTS = 52

#: Slot width per task -- ONE definition, because `_slots`, `_bounds`,
#: `obs_dim` and `_row_derived` all need it and four copies is four chances to
#: describe a row the adapter does not emit. scratchitch's four are the
#: episode-constant target descriptors the spec documents at s[61..64].
_N_SLOTS = {"bedbathing": N_WIPE_POINTS, "scratchitch": 4}


# -- the consumer gate ------------------------------------------------------
#
# READ OFF THE REGISTERED OBJECT by `config._inherit_from_task_spec`, so it must
# be attached to whatever is registered -- a factory inherits nothing from the
# class it constructs. A registered class that never set it would carry zero
# forbidden symbols where the CPU `assistax_feeding` carries 248.
#
# A SUPERSET OF THE CPU TIER'S, NOT A NARROWER LIST. A narrower gate tailored to
# this family is rejected: a jax tier's gate should equal or contain its CPU
# parent's, and a tier whose whole point is a held-out metric must not carry a
# weaker gate than the tier without one. The prompt cost (~4.5k -> ~7k chars;
# `generation.py` pastes it verbatim) is the lesser risk, and pruning the
# names that do not apply to this family would need its own measurement.
def _consumer_forbidden() -> Tuple[str, ...]:
    from . import assistax as _cpu                        # noqa: WPS433
    extra = (
        # Upstream's paid reward and its components -- `reference_reward` is
        # upstream's own here, so these are the reference itself.
        #
        # MEASURED, not read off the source and not pasted. This is the union
        # of `state.metrics` over the five tasks, collected in the jax venv
        # (assistax @ a7d94f4e) with:
        #
        #   for t in TASKS: sorted(assistax.make(t).reset(key)[1].metrics)
        #
        # A hand-written list is the failure mode: one held fifteen of the
        # sixteen, missing `reward_scratching` -- the component of SCRATCHITCH
        # -- while every test stayed green, because the gate was present,
        # non-empty, and never compared against the env it gates.
        # `tests/test_upstream_assistax.py::test_gate_covers_the_measured_metric_and_info_keys`
        # is the comparison, marked `jax` because it constructs.
        "reward_align", "reward_brushing", "reward_ctrl", "reward_dist",
        "reward_force", "reward_hook_dist", "reward_orientation",
        "reward_rot", "reward_scratching", "reward_velocity",
        "reward_waist_dist", "reward_wiping", "weighted_reward_ctrl",
        "weighted_reward_hook_dist", "weighted_reward_rot",
        "weighted_reward_waist_dist",
        # `state.info`, same measurement, same union. `scratch` is the
        # scratchitch ACHIEVEMENT channel and `truncation`/`steps` are the
        # horizon -- a candidate reading any of them reads the reduction it is
        # being scored on rather than the physics.
        "action_magnitude", "scratch", "steps", "truncation",
        # The HELD-OUT preference metric: these seven ARE the arguments of
        # `compute_preference_reward`, so a candidate reading them could
        # reconstruct R_pref exactly.
        "w_speed", "w_force", "w_touch", "speed_range_min", "speed_range_max",
        "force_range_min", "force_range_max",
        "reward_budget", "overall_weight", "touch_threshold",
        "compute_preference_reward", "preference_tracking", "pref_configs",
        "speed_pref_reward", "force_pref_reward", "touch_penalty_reward",
        "total_pref_reward", "pref_raw_speed", "pref_raw_force",
        "last_contact_force", "prev_contact_force",
        # Upstream's own channels for the latch and the raw telemetry the
        # preference reward is computed from. The physics is legitimately the
        # candidate's via the tail (`tool_speed`, `tool_force_mag`); these ban
        # the CHANNEL, not the quantity.
        "contact_vector", "_update_contact_vector", "ee_speed", "ee_force",
        "force_on_human", "force_on_tool",
        # The reference reward's SOURCE TEXT. Banned here even though the CPU
        # tier's own gate does not list it.
        "reward_source",
    )
    # THE TIER'S OWN ANALOGUES -- the accessors by which a candidate would reach
    # the paid reward on THIS adapter rather than upstream's. Banning upstream's
    # vocabulary and leaving ours open would gate the copy and not the original.
    ours = (
        # the row/table object and the task key that selects it
        "_row", "_slots", "_tail_spec", "_derived", "_task", "_CLASSES",
        "UPSTREAM_TASKS", "_ReferenceCache", "_UpstreamDerived",
        # the reference-reward accessor and the two reductions
        "reference_reward", "success_from_state", "task_metric", "success",
        # the source-text withholder. This adapter does not define one; the
        # name is banned so that adding one later cannot quietly become
        # reachable, and so the ban does not have to be remembered at that moment.
        "_render_full_source",
    )
    # EVERY callable named in the CPU tier's task table, enumerated rather than
    # pasted. A pasted subset cannot fail on the row somebody adds next -- a
    # test that asserts a handful of names are banned would stay green if a new
    # task shipped an unbanned `_rs_foo`. Reading `_TASKS` means a new row is
    # gated the day it is added, with no edit here. The walk covers all five
    # rows, not only the task this adapter runs: the ban costs a candidate
    # nothing (these are private module symbols no reward legitimately writes)
    # and the leak it closes is a candidate reaching a SIBLING task's reference
    # reward, which is the same leak one name removed.
    def _callables(obj: Any, depth: int = 0) -> List[str]:
        """Every callable name reachable from a row value, AT ANY DEPTH.

        RECURSES INTO COLLECTIONS, and that is the whole point. Testing each
        row value for `callable()` and stopping would silently skip any helper
        a row holds inside a tuple, list or dict rather than as a bare value --
        and such a helper is the same category as `_ref_*`, reachable by a
        candidate through a gate that looked comprehensive.

        The names are re-derived here by walking, never pasted: an independent
        AST pass over the same table agreeing with this walk is what makes it
        a guarantee rather than an assertion, and a paste would make the two
        walks one walk with two copies of its blind spot.
        """
        if depth > 6:
            return []
        if callable(obj):
            # `<lambda>` is not a referenceable symbol -- the tail's lambdas
            # report it and a candidate cannot write it, so banning it would
            # add a name that gates nothing and inflate the count by one.
            # `isidentifier()` is the filter rather than a `!= "<lambda>"`
            # special case, because it is the actual property required.
            name = getattr(obj, "__name__", "")
            return [name] if name.isidentifier() else []
        if isinstance(obj, dict):
            obj = list(obj.values())
        if isinstance(obj, (list, tuple, set, frozenset)):
            return [n for item in obj for n in _callables(item, depth + 1)]
        return []

    rows = tuple(sorted({
        name for row in _cpu._TASKS.values()
        for value in row.values() for name in _callables(value)
    }))
    seen: List[str] = []
    for name in (*_cpu._CONSUMER_FORBIDDEN, *extra, *ours, *rows):
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def _make_factory(cls: type) -> Any:
    def make(ctx: Any = None) -> Any:
        # BEFORE the lazy jax import inside __init__, never after. A process
        # that reaches this adapter by any path other than `bird.py::run` or
        # the spawn child otherwise gets no XLA_FLAGS and no
        # XLA_PYTHON_CLIENT_PREALLOCATE -- and jax reads both AT IMPORT, so
        # calling it later is calling it never. `xla_env.prepare_for_env` keeps
        # the strict call for an UNDECLARED adapter for the same reason.
        from ..xla_env import prepare_for_env             # noqa: WPS433
        # strict=False: this is the CONSTRUCTION-SITE call. A strict one from
        # inside __init__ turns a late flag into a dead candidate and reds
        # every jax-tier test whose fixture imports jax before constructing.
        # The two strict calls are the entry points (see `xla_env.prepare_for_env`).
        prepare_for_env(f"upstream_assistax_{cls.task}", strict=False)
        return cls(ctx)
    make.consumer_forbidden_symbols = _consumer_forbidden()
    # The attribute lives on the REGISTERED object; a factory inherits nothing
    # from the class it constructs. `config._check_coherence` getattrs
    # `requires_cuda` off this too.
    make.requires_cuda = cls.requires_cuda
    make.batched = cls.batched
    make.adapter_cls = cls
    return make


class UpstreamAssistaxTeethBrushing(UpstreamAssistax):
    """`upstream_assistax_teethbrushing`.

    Same shape as feeding (no slots, `done = 0.0` literal, `_get_robo_obs(ps)`
    with no `info`): `tool_position` is the `toothbrush_center` site,
    `tool_orientation` the `toothbrush` body quaternion, `target_position` the
    `mouth` site, `force_on_tool` the sum of the head and right-side contacts
    (upstream's ids 19/20, its own "TODO: update these" comment kept as
    upstream's). The CPU tail (`_TAIL_TEETHBRUSHING`) reads `mouth_pos, dist,
    bristle_dot_to_mouth, tip_force_mag, tangential_speed`, computed here with
    `_derive_teethbrushing`'s formulas through the CPU module's `_quat_rotate`
    and `_unit` (both take `xp`): bristles are the brush's local -x, the
    tangential speed is `tool_vel` minus its component along the mouth->tool
    normal, `tip_force_mag` the right-side contact's force norm (the CPU row's
    `tip_geoms=("toothbrush_right_side",)`).
    """

    task = "teethbrushing"
    tool_site_attr = "panda_toothbrush_centre"
    robo_obs_takes_info = False

    def _derived_xp(self, base: Any, slots: Any, tool_vel: Any, xp: Any) -> Any:
        cpu = self._cpu_mod
        ro = self._robo_obs(base)
        ps = base.pipeline_state
        f6 = xp.asarray(ro["force_on_tool"]).reshape(-1)
        tip6 = xp.asarray(self._inner._get_force_on_tool(
            ps, self._inner.TOOTHBRUSH_RSIDE_CONTACT_ID)).reshape(-1)
        tool_pos = xp.asarray(ro["tool_position"]).reshape(3)
        tool_quat = xp.asarray(ro["tool_orientation"]).reshape(4)
        mouth_pos = xp.asarray(ro["target_position"]).reshape(3)
        tool_vel = xp.asarray(tool_vel).reshape(3)
        to_mouth = cpu._unit(mouth_pos - tool_pos, xp=xp)
        bristle = cpu._quat_rotate(tool_quat, xp.asarray([-1.0, 0.0, 0.0]), xp=xp)
        tilt_axis = cpu._quat_rotate(tool_quat, xp.asarray([0.0, -1.0, 0.0]), xp=xp)
        normal = cpu._unit(tool_pos - mouth_pos, xp=xp)
        v_tan = tool_vel - xp.dot(tool_vel, normal) * normal
        return _UpstreamDerived(
            xp=xp,
            slots=slots,
            tool_pos=tool_pos,
            tool_quat=tool_quat,
            mouth_pos=mouth_pos,
            target_pos=mouth_pos,
            tool_force=f6[:3],
            tool_force_mag=xp.linalg.norm(f6[:3]),
            tool_vel=tool_vel,
            tool_speed=xp.linalg.norm(tool_vel),
            dist=xp.linalg.norm(mouth_pos - tool_pos),
            bristle=bristle,
            tilt_axis=tilt_axis,
            bristle_dot_to_mouth=xp.dot(to_mouth, bristle),
            tilt_dot_to_mouth=xp.dot(to_mouth, tilt_axis),
            tangential_speed=xp.linalg.norm(v_tan),
            tip_force=tip6[:3],
            tip_force_mag=xp.linalg.norm(tip6[:3]),
        )


class UpstreamAssistaxArmManipulation(UpstreamAssistax):
    """`upstream_assistax_armmanipulation`.

    UPSTREAM'S `_get_robo_obs` RETURNS NO FORCE AND NO QUATERNION on this task
    (`armmanipulation.py:300-330`: `force_on_tool` is commented out, the
    orientation is the hook site's ROTATION MATRIX), so the tail's inputs are
    read off the pipeline state and upstream's own index attributes rather
    than off the obs dict, which is why `_robo_obs` is not called here:
      * `tool_pos` = `platform_center` site (`panda_hook_center_idx`),
        `tool_quat` = the `hook` body quaternion (`panda_hook_body_idx`) --
        the CPU row's `tool_body="hook"`;
      * `hook_target_pos` / `waist_target_pos` = the `hook_target` /
        `arm_target` sites; `dist` = |hook_target - tool|,
        `dist_forearm_waist` = |waist_target - hook_target|;
      * `rot_err` = Frobenius norm of `site_xmat[hook_target] -
        site_xmat[platform_center]`, upstream's `tool_target_dist_angular`
        (`:307`) normed as the CPU `_derive_armmanipulation` does;
      * `tool_force` = upstream's own four-contact sum
        (`_get_force_on_tool(ps, UARM_HPLATFORM, LARM_HPLATFORM, UARM_HEND,
        LARM_HEND)`, `:388-394`) -- the hook platform and hook end against the
        upper and lower arm, the CPU row's `tool_geoms` x `human_geoms`;
      * `uarm_pos` / `larm_pos` = `xpos` of the right upper/lower arm bodies,
        exactly what scratchitch's obs dict carries under `human_uarm_pos`.
    No slots, `done = 0.0` literal (`:256`).
    """

    task = "armmanipulation"
    tool_site_attr = "panda_hook_center_idx"
    robo_obs_takes_info = False

    def _derived_xp(self, base: Any, slots: Any, tool_vel: Any, xp: Any) -> Any:
        inner = self._inner
        ps = base.pipeline_state
        tool_pos = xp.asarray(ps.site_xpos)[inner.panda_hook_center_idx].reshape(3)
        tool_quat = xp.asarray(ps.xquat)[inner.panda_hook_body_idx].reshape(4)
        hook_target = xp.asarray(ps.site_xpos)[inner.hook_target_site].reshape(3)
        waist_target = xp.asarray(ps.site_xpos)[inner.arm_target_site].reshape(3)
        diff = (xp.asarray(ps.site_xmat)[inner.hook_target_site]
                - xp.asarray(ps.site_xmat)[inner.panda_hook_center_idx])
        f6 = xp.asarray(inner._get_force_on_tool(
            ps, inner.UARM_HPLATFORM_CONTACT_ID, inner.LARM_HPLATFORM_CONTACT_ID,
            inner.UARM_HEND_CONTACT_ID, inner.LARM_HEND_CONTACT_ID)).reshape(-1)
        tool_vel = xp.asarray(tool_vel).reshape(3)
        return _UpstreamDerived(
            xp=xp,
            slots=slots,
            tool_pos=tool_pos,
            tool_quat=tool_quat,
            hook_target_pos=hook_target,
            waist_target_pos=waist_target,
            target_pos=hook_target,
            uarm_pos=xp.asarray(ps.xpos)[inner.human_tuarm_idx].reshape(3),
            larm_pos=xp.asarray(ps.xpos)[inner.human_tlarm_idx].reshape(3),
            tool_force=f6[:3],
            tool_force_mag=xp.linalg.norm(f6[:3]),
            tool_vel=tool_vel,
            tool_speed=xp.linalg.norm(tool_vel),
            dist=xp.linalg.norm(hook_target - tool_pos),
            dist_forearm_waist=xp.linalg.norm(waist_target - hook_target),
            rot_err=xp.sqrt(xp.sum(diff ** 2)),
        )


_CLASSES = {"scratchitch": UpstreamAssistaxScratchItch,
            "feeding": UpstreamAssistaxFeeding,
            "teethbrushing": UpstreamAssistaxTeethBrushing,
            "armmanipulation": UpstreamAssistaxArmManipulation}


for _task, _cls in _CLASSES.items():
    register("env", f"upstream_assistax_{_task}")(_make_factory(_cls))

#: The tasks not yet adapted. Registering nothing for them is deliberate: an id
#: that resolves to a stub which refuses at construction is harder to
#: distinguish from a broken adapter than an id that does not resolve at all,
#: and `_check_coherence` already refuses a config naming an unknown env.
UNADAPTED: Tuple[str, ...] = tuple(t for t in UPSTREAM_TASKS if t not in _CLASSES)
