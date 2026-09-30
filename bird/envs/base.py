"""`EnvAdapter` -- the single environment interface the rest of BIRD is written against.

Its own module, and not `toy.py`, for one reason: `bird/envs/spec.py` subclasses it and
`toy.py` needs `SpecEnvAdapter` back, which is a cycle. A separate base module breaks it
permanently, rather than papering over it with a deferred import that works only while the
registry happens to load these modules in one particular order.

`toy.py` re-exports `EnvAdapter`, `as_np_rng`, `_bin` and `_states_of`. `EnvAdapter` is in
`toy.py`'s `__all__` and that is a public contract; the three private helpers should be
imported from here.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import logging
import math

import numpy as np

_log = logging.getLogger("bird.env")


class UnknownStateError(RuntimeError):
    """An adapter was handed an observation it never emitted (or has since forgotten),
    and cannot restore, render, reward or SCORE it: the simulator is stateful and the
    flat observation does not invert to it.

    The base of `metaworld.UnknownStateError` (raised by `_step`/`render`/
    `reference_reward` on a snapshot-cache miss; `task_metric` there is a pure function
    of the observation). An adapter whose success flag is recorded when the state was
    emitted, never re-derived, raises it from `task_metric` too. It lives here so a
    caller that scores states it did not produce -- e.g. a ground-truth judge that
    builds a fresh adapter and hands it a trajectory read off disk -- can recognise
    "this adapter cannot score these states" by ONE name, without importing any
    family's module: catching a family-specific subclass in such a judge would be
    method branching on an environment. A `RuntimeError`, for the reason
    `metaworld.py` gives at its subclass.
    """


def as_np_rng(rng: Any) -> np.random.Generator:
    """Accept whatever the caller has and hand back a numpy Generator.

    `ctx.rng` is a `random.Random` (bird.py seeds it from `cfg.seed`), the
    training backends hold `np.random.Generator`s, and phases sometimes pass a
    bare int. All three must produce a *reproducible* stream, so a `Random` is
    consumed rather than ignored: it advances, which keeps repeated resets from
    replaying one identical episode.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(0)
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    getrandbits = getattr(rng, "getrandbits", None)
    if callable(getrandbits):  # random.Random
        return np.random.default_rng(int(getrandbits(63)))
    randint = getattr(rng, "randint", None)
    if callable(randint):  # np.random.RandomState
        return np.random.default_rng(int(randint(0, 2 ** 31 - 1)))
    return np.random.default_rng(0)


def _bin(x: float, lo: float, hi: float, n: int) -> int:
    """Uniform binning, clamped. Plain floats -- this runs once per env step."""
    i = int((x - lo) / (hi - lo) * n)
    if i < 0:
        return 0
    return n - 1 if i >= n else i


def _states_of(traj: Any) -> np.ndarray:
    """`traj` may be a Trajectory, a (T+1, obs_dim) array, or a list of states.

    Ground-truth scoring is called from screens and from §4 on records that
    have been round-tripped through JSON, so be liberal about the container and
    strict about the shape.
    """
    states = getattr(traj, "states", traj)
    if states is None:
        return np.zeros((0, 1))
    arr = np.asarray(states, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# --------------------------------------------------------------------------
# viewpoints
# --------------------------------------------------------------------------


def _freeze(value: Any) -> Any:
    """Lists become tuples, all the way down, so a `View` stays hashable."""
    if isinstance(value, dict):
        return tuple((str(k), _freeze(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class View:
    """One viewpoint a task spec offers the recorder (`judge.camera`, `judge.extra_views[]`).

    `name` is a camera the simulator model defines, unless `pose` is given, in which
    case it is only a label and the camera is built from `pose` at render time
    (`envs.cameras.MujocoViews.camera_for`). `mode` and `note` are DESCRIPTIVE: they
    are what the judge is told about the panel (`multiview.describe`), so a note that
    misdescribes the camera is a false statement handed to the judge, not a comment.
    """

    name: str
    mode: str = "fixed"           # fixed | tracking, the spec's own vocabulary
    note: str = ""
    pose: Optional[Tuple[Tuple[str, Any], ...]] = None

    @staticmethod
    def from_mapping(block: Mapping[str, Any]) -> "View":
        pose = block.get("pose")
        return View(name=str(block["name"]), mode=str(block.get("mode") or "fixed"),
                    note=str(block.get("note") or "").strip(),
                    pose=_freeze(dict(pose)) if pose else None)

    @property
    def pose_dict(self) -> Optional[Dict[str, Any]]:
        if self.pose is None:
            return None
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.pose}

    def as_record(self) -> Dict[str, Any]:
        """JSON-safe: what a journal line or a judgment record carries about a panel."""
        out: Dict[str, Any] = {"name": self.name, "mode": self.mode, "note": self.note}
        if self.pose is not None:
            out["pose"] = self.pose_dict
        return out


# --------------------------------------------------------------------------
# `problem.horizon`
# --------------------------------------------------------------------------


def apply_horizon(env: Any, cfg: Any) -> int:
    """Install `problem.horizon` on a freshly constructed adapter; return the
    horizon in force.

    THE ONE SEAM, and it is deliberately not inside any adapter. `horizon` is a
    plain attribute read by every backend, by `_VecEnvView`, `_rollout`,
    `_evaluate_policy`, the trajectory trace and the prompt renderers, so
    setting it once on the instance is what makes the override reach all of them
    without a second copy of the rule -- and what makes a horizon this key
    shortened visible in `describe()` and therefore in the prompt, which is the
    half a wrapper env would have missed.

    Callers are the sites that construct an adapter WITH a config --
    `bird.py`'s run and `training._hydrate_ctx`'s spawned worker -- and the
    count is not maintained by this docstring: `tests/test_problem_horizon.py::
    test_every_config_bearing_adapter_construction_applies_the_horizon` walks
    the source for `registry.get("env", ...)(ctx)` and requires each such file
    to apply it. A hand-maintained list goes stale, and a missed site is
    silent: since `final_retrain` is a `post:` phase, a site that skipped
    this would retrain at the environment's horizon beside a search at the
    pinned one. The test is what catches a new site.

    A tool that builds an adapter with `(None)` has no config and correctly
    gets the environment's own horizon (`scripts/eval_policy.py`,
    `measure_anchors.py`, `policy_api`). That is why the anchors a
    task records stay comparable across configs: they are measured at the
    environment's horizon, and a run that shortened it says so on its own seed
    rows rather than silently re-basing theirs.

    TRUNCATION ONLY, AND THE TWO SITES DO DIFFERENT THINGS ABOUT IT --
    deliberately.

      at LOAD (`_check_coherence`), when the config CHOSE an env:  REFUSE.
      at CONSTRUCTION (here), whatever the env turned out to be:    CLAMP,
                                                                    warn, and
                                                                    record both.

    The refusal is where it can be acted on: a command line naming
    `h1hand_package` with `problem.horizon: 2000` is a mistake, the spec's 1000
    is readable without constructing a MuJoCo model, and the run dies in the
    second it takes to load. Every run resolves through `config.load` first, so
    that is the path a wrong horizon actually arrives by.

    THE CLAMP IS NOT THE COMPROMISE IT LOOKS LIKE. The obvious objection -- "a
    clamp would let a config state 2000, run 1000, and record the 2000
    nowhere" -- is answered by the caller recording BOTH, as
    `horizon_requested` and `horizon_effective` on every seed row, so nothing
    is unrecorded and a reader sees the cap rather than inferring it.

    What forces it is the configs' own idiom. A paper config here names no env
    -- `configs/methods/rda.yaml`'s header says so outright -- so
    `configs/methods/rda_humanoidbench.yaml`, which pins RDA's Table-1 horizon of 500,
    resolves to `_default.yaml`'s `toy_reacher` and its horizon of 25 under
    every profile. A refusal at construction would therefore make that config
    UNRUNNABLE under the tester profile, and `tests/test_pipeline.py` runs every
    method point there. The same config pins `train.env_steps: 10000000`, which
    is equally impossible for a toy env, and the profile handles it by OVERRIDING:
    `train.env_steps` is a `PROFILE_KEY_PREFIXES` member and
    `_profiles/tester.yaml` pins 400. `problem.horizon` cannot be handled that
    way and must not be -- it is §0, it is RDA's method identity, and a profile
    key is a key a profile may silently change the science with, which is the
    argument `_profiles/humanoid.yaml`'s own header makes about
    `train.algorithm`. So the pin stays a CITATION until an env is chosen, the
    chosen env caps it, and the artifact carries both numbers. That is exactly
    what the profile does to `train.env_steps`, by a different mechanism.

    AN ADAPTER THAT REPORTS NO HORIZON (`own == 0`) takes any positive value,
    and that is vacuous rather than a hole: there is nothing to truncate FROM,
    so "truncation only" has no content there. Said explicitly because a reader
    tracing the `if own and want > own` guard will wonder whether the `own`
    conjunct is a bug.

    Returns `(effective, requested)` so a caller records what ran and what was
    asked for without re-deriving either. `requested` is `None` when the config
    pinned nothing, which is how a row says "the environment's own" rather than
    claiming a number nobody chose.
    """
    own = int(getattr(env, "horizon", 0) or 0)
    get = getattr(cfg, "get", None)
    want = get("problem.horizon") if callable(get) else None
    if want is None:
        return own, None
    want = int(want)
    if want < 1:
        raise ValueError(
            f"problem.horizon={want} is not an episode length; leave it null for "
            f"{getattr(env, 'name', '?')}'s own ({own})")
    if own and want > own:
        # CLAMPED AND SAID SO, at WARNING, because under the tester profile this
        # is the normal case (a paper's horizon against a toy env) and on a real
        # env it is a mistake `_check_coherence` has already refused. A log
        # line is not the record -- `horizon_requested` on the seed row is --
        # but a run whose horizon was capped should say so where an operator
        # watching stdout can see it too.
        _log.warning(
            "problem.horizon=%d is longer than %s's own horizon (%d), so the "
            "episode is the environment's: this key TRUNCATES and never "
            "extends, and a paper's horizon against another tier's env is a "
            "CITATION rather than a pin (the same thing `train.env_steps: 10M` "
            "is on the tester tier). Both numbers are on every seed row as "
            "horizon_requested / horizon_effective. `_check_coherence` refuses "
            "this instead of clamping when the config chose its own env.",
            want, getattr(env, "name", "?"), own)
        return own, want
    env.horizon = want
    return want, want


# --------------------------------------------------------------------------
# the interface
# --------------------------------------------------------------------------


class EnvAdapter:
    """The single environment interface the rest of BIRD is written against.

    Subclasses supply dynamics (`_step`), an initial-state distribution
    (`_reset`), a ground-truth task metric, a reference reward, and a
    discretisation. Everything else -- the four `describe()` renderings, DR
    plumbing, transition sampling, success -- is shared, because those are
    *config axes* (`generate.context.env_spec`,
    `train.domain_randomization.mode`) and a config axis that behaves
    differently per environment is not an axis.

    Beyond the members named in the repo contract, three extras exist for the
    training backends in §3 and are part of the interface:

      `action_set`     (n_actions, action_dim) float array. Every env exposes a
                       finite action set; continuous envs additionally accept
                       any action inside [`action_low`, `action_high`].
      `discretise(s)`  state -> int in [0, n_disc_states). Exact (a bijection)
                       when `exact_states` is not None -- that is what makes
                       `train.backend: tabular` honest on Singh's domain and a
                       binning everywhere else.
      `task_metric(t)` ground-truth fitness F in [0, 1]. `success()` is a
                       threshold on it, so the two can never disagree.
    """

    # -- static description; subclasses override --
    name: str = "abstract"
    obs_dim: int = 0
    action_dim: int = 1
    horizon: int = 1
    n_disc_states: int = 1
    exact_states: Optional[int] = None  # not None => discretise is a bijection
    success_threshold: float = 0.0
    #: Which values of `evaluate.fitness.reduction` (§4) this adapter honours.
    #:
    #: Declared rather than assumed, because a key an adapter silently ignores is a
    #: fabricated pin -- and here it would be the dangerous
    #: kind: the run would complete, the dashboard would fill, and the number would be
    #: the other reduction's. `_check_coherence` refuses a value the selected env does
    #: not list. Only environments whose `task_metric` is a reduction of a per-step
    #: success check can offer more than one.
    supported_reductions: Tuple[str, ...] = ("per_step_fraction",)
    #: `generate.postprocess.symbol_mapping` -- T2R's general->specific table.
    symbol_mapping: Dict[str, str] = {}
    #: `train.domain_randomization` / DrEureka RAPP: name -> (lo, hi) bounds.
    dr_parameters: Dict[str, Tuple[float, float]] = {}
    _dr_nominal: Dict[str, float] = {}

    #: The viewpoint `render` shows, and the extra viewpoints `render_view` can show.
    #: Both come from the task spec's `judge` block (`SpecEnvAdapter._apply_spec`); an
    #: adapter with no spec renders one unnamed view and offers no extras. The
    #: recorder reads these through `multiview.select`, gated by
    #: `output.video.n_views` -- a spec may OFFER three viewpoints while every
    #: published config records one, which keeps every published config's output
    #: unchanged.
    primary_view: View = View(name="primary")
    extra_views: Tuple[View, ...] = ()

    # -- describe() source material --
    _prose: str = ""
    _success_prose: str = ""
    _state_fields: Sequence[Tuple[str, str]] = ()
    #: `(name, start, stop)` groups over the flat row, when the spec declares
    #: them. Empty on every adapter that is not spec-backed, which is why the
    #: prompt renderer treats it as optional rather than required.
    _state_groups: Sequence[Tuple[str, int, int]] = ()
    _action_fields: Sequence[Tuple[str, str]] = ()
    _helpers: Sequence[Tuple[str, str]] = ()

    def __init__(self) -> None:
        self.action_set = np.asarray(self._build_action_set(), dtype=float)
        self.action_low = np.asarray(self.action_set.min(axis=0), dtype=float)
        self.action_high = np.asarray(self.action_set.max(axis=0), dtype=float)
        self.obs_low, self.obs_high = self._bounds()
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._dr_ranges: Dict[str, Tuple[float, float]] = {}
        self._dr_now: Dict[str, float] = dict(self._dr_nominal)

    # -- subclass hooks -----------------------------------------------------

    def _build_action_set(self) -> Any:
        raise NotImplementedError

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        raise NotImplementedError

    def discretise(self, s: np.ndarray) -> int:
        raise NotImplementedError

    def task_metric(self, traj: Any) -> float:
        """Ground-truth fitness F in [0, 1]. The quantity a method is *judged*
        on, never one a candidate reward is allowed to see."""
        raise NotImplementedError

    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """The hand-written reward a candidate is benchmarked against, for arriving
        in `s` having taken `a` -- or `NotImplementedError` where the task has none.

        Raising is the contract for absence, and every consumer handles it
        EXPLICITLY rather than reading 0.0: `components.training._rollout` records
        `gt_return = None` (so `gt_reward_curve`, the `pearson_curve` similarity and
        `phases._reference_score` see no channel rather than a flat zero),
        `policy_api.run_episode` records `reference_return: None`, and
        `scripts/train_expert.py` refuses to train. `has_reference_reward` below is
        the same fact as a flag, for callers that want to ask before calling.
        """
        raise NotImplementedError

    @property
    def has_reference_reward(self) -> bool:
        """Whether `reference_reward` is available on this adapter -- a per-TASK
        fact, not a per-simulator one.

        Default: True exactly when a subclass overrides `reference_reward`, which
        is every adapter in the tree today. An adapter that backs several tasks on
        one simulator (`gym_mujoco.GymMujoco`) overrides THIS too, because the
        simulator's built-in reward being computable says nothing about whether
        the task claims it: `tasks/<id>/shared_spec.yaml::reward.human.kind: none`
        disowns it. Returning the base environment's forward-velocity reward on
        such a task would make `gt_return` score a backward-running or in-place
        task by how fast it ran FORWARD, and `scripts/train_expert.py` would train
        "expert" anchors on it. When this is False, `reference_reward`
        raises `NotImplementedError`; the two must agree, and
        `tests/test_gym_reference_reward.py` holds them together.
        """
        return type(self).reference_reward is not EnvAdapter.reference_reward

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        """A state drawn for *coverage*, not from the initial distribution --
        EPIC/STARC need the whole space, not the on-policy slice."""
        raise NotImplementedError

    def expert_policy(self) -> Optional[Callable[[np.ndarray, int], np.ndarray]]:
        """The environment's OWN shipped solution, `act(s, t) -> a`, or None.

        Third source for `bird.demos._expert`, after the `policies/` registry
        and Meta-World's bundled scripted policies, and of the same kind as the
        latter: a solution the *environment* ships, not one a run produced. Only
        an analytic controller for a toy of ours belongs here (toy_reacher's PD
        law); anything measured or tuned is a `policies/` record with a score
        and a caveat list. Stateless by contract -- a phase machine needs a
        per-episode reset and therefore a `policies/` entry. Reading it is
        demonstration access (`problem.fitness_access: demonstrations`) exactly
        as the other two sources are.

        BIND IT OUTSIDE THE SUBCLASS BODY (see the foot of `bird/envs/toy.py`):
        `env_spec: full_source` ships `inspect.getsource(type(self))` to the
        generator verbatim, so an override written inside the class is a
        solution policy in every prompt.
        """
        return None

    # -- the interface ------------------------------------------------------

    @property
    def n_actions(self) -> int:
        return int(self.action_set.shape[0])

    def reset(self, rng: Any = None) -> np.ndarray:
        """Start an episode. The RNG passed here drives the whole episode."""
        self._rng = as_np_rng(rng)
        self._dr_now = self._sample_dr(self._rng)
        return self._reset(self._rng)

    def step(self, state: np.ndarray, action: Any) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """`(next_state, done, info)`; `info["success"]` is the ground-truth
        per-step success flag. Never returns a reward -- reward is the thing
        under search, so the env does not get to supply one.

        THE FLAG IS AN EVENT, NOT A PROXY FOR PROGRESS: True on a step whose arriving
        state satisfies the task's own success condition -- the comparison
        `task_metric` reduces (tip above the bar, pole upright, inside the tunnel past
        its mouth, running at the bar's speed), never "moving in the right direction".
        Every adapter emits the key, False where the task has no success bar
        (`success_threshold` inf). `train.bc_prior.accept: success` reads the FIRST
        flagged step as the demonstration's success and cuts `success_margin` steps
        after it, so a flag that fires on the first forward lean turns a 1000-step
        demonstration into 26 pairs while every counter reads normal, and an adapter
        that omits the key drops every demonstration as a failure
        (`tests/test_env_success_flag.py` holds the contract)."""
        return self._step(np.asarray(state, dtype=float),
                          np.asarray(action, dtype=float).ravel())

    # -- per-episode INSTANCE state --------------------------------------------

    def episode_state(self) -> Any:
        """Everything `reset()` leaves on the INSTANCE that a later `step()` reads
        and the state array does not carry: the episode RNG and the DR draw.

        `step(state, action)` is state-PASSING, but not quite a pure function of
        its arguments -- `_step` reads `self._rng` (stochastic dynamics,
        `action_noise`) and `self._dr_now` (the episode's domain-randomisation
        draw), both written by `reset()`. Interleaving N episodes on ONE adapter
        therefore needs each episode's pair put back before each of its steps;
        `restore_episode_state(self.episode_state())` is that, and it is what
        `components.fasttd3._VecEnvView` does to run `num_envs` slots on one
        adapter instead of constructing `num_envs` of them (128 MuJoCo models
        and 128 GL contexts, on the envs that backend exists for).
        `sample_transitions` is the older precedent: it saves the same pair,
        steps under nominal dynamics, and puts it back.

        The blob is opaque to callers and may hold live references: a
        `Generator` is mutable, so a step that advances the RNG advances the
        blob's, which is exactly what "the same episode continues" means.
        An adapter whose `_reset` derives MORE instance state from the draw
        would override BOTH methods, `super()` first -- none does: derived
        quantities are read off `_dr_now` at step time instead, so the base pair
        already carries them. The MuJoCo tiers need nothing here either,
        for two different reasons, and the difference is the thing to keep
        straight: `metaworld`'s `_step` restores its simulator from the state's
        own episode snapshot, which holds every model field its DR writes, so
        there the draw travels with the state; HumanoidBench (`humanoid.py`,
        `humanoid_hand.py`) has no draw to restore at all -- every
        h1hand/h1strong spec says `domain_randomization: null` and neither
        adapter reads `_dr_now` -- so once `_step` has written `qpos/qvel`
        back from the state, the base pair is the whole of its instance state.
        `tests/test_env_episode_state.py` interleaves two episodes on one
        adapter against two adapters for every cheaply constructible env.
        """
        return (self._rng, self._dr_now)

    def restore_episode_state(self, blob: Any) -> None:
        """Put back what `episode_state` captured; see it for the contract."""
        self._rng, self._dr_now = blob

    def policy_features(self, state: np.ndarray) -> np.ndarray:
        """What the POLICY consumes. Identity here, and identity is the truth.

        `problem.search_space: [reward]` -- every shipped method but LIMEN
        -- searches the reward over a FIXED observation, and for those the
        policy's input is the state array itself. LIMEN searches the observation
        too (`generate.co_design.observation_fn`), and its co-designed
        `get_observation(state)` is installed by wrapping an adapter in
        `components.training._ObsView`, which overrides this method and the three
        members that describe its output space (`obs_dim`, `obs_low`,
        `obs_high`).

        THE SPLIT IS THE WHOLE CORRECTNESS STORY, so it is stated on both sides.
        This hook is the ONLY place the co-designed observation may be applied:
        dynamics (`step`), the candidate reward, `task_metric`,
        `reference_reward`, the screens and the recorded trajectory all stay on
        the raw state, because LIMEN's prompt hands `get_observation(state)` and
        `compute_reward(state, action, next_state)` the SAME `state`. A reward
        re-based onto the features would be scoring a different program from the
        one the model wrote, and a `task_metric` re-based onto them would let a
        candidate move the ground truth by choosing what it observes.
        """
        return np.asarray(state, dtype=float)

    def success(self, traj: Any) -> bool:
        """Ground-truth episode success: a threshold on `task_metric`, so the
        binary and the continuous ground truth can never disagree."""
        return bool(self.task_metric(traj) >= self.success_threshold)

    def describe_success(self) -> str:
        """The success conditions in words, for a prompt that asks for them
        (R*'s critic author, Prompt 2 `{task_goal}`). The task spec's
        `description.success_criterion_prose` when it states one, else the bar
        `success()` actually applies, so the sentence can never contradict the
        check."""
        prose = str(self._success_prose or "").strip()
        if prose:
            return prose
        return ("An episode counts as a success when the environment's ground-truth "
                f"task metric (a score in [0, 1]) reaches at least "
                f"{float(self.success_threshold):g}.")

    # -- extra viewpoints (`output.video.n_views`) ----------------------------

    def render_view(self, state: np.ndarray, view: View) -> np.ndarray:
        """One `(H, W, 3)` uint8 frame of `state` from `view`, an entry of `extra_views`.

        The sibling of the duck-typed `render(state)` (`observability._render_frames`
        documents that contract), and only ever called with a `View` this adapter
        itself advertised in `extra_views` -- so the base implementation is a refusal,
        not a fallback: an adapter that offers no extra views has nothing to render,
        and `multiview.select` never asks it. Returning the primary frame here instead
        would put two copies of one camera in a frame the judge is told shows two
        viewpoints.

        The MuJoCo tiers implement it through `envs.cameras.MujocoViews`, which is
        where a model camera's orientation is corrected and a `pose` camera is built.
        """
        raise NotImplementedError(
            f"{type(self).__name__} offers no extra views (extra_views is empty)")

    # -- what the task asks, as a short label -----------------------------------

    def legend_lines(self, state: np.ndarray) -> List[str]:
        """Short upper-case lines naming what the task ASKS at `state`, or `[]`.

        Optional: a hook for labelling rendered frames (drawn with `bird/envs/hud.py`);
        the search loop does not call it. The command, the target, the
        phase -- whatever the policy is being asked to do at this instant, read from
        the state (a per-episode goal rides in the state's extra dims; a phase clock is a
        command slot) or from the task's own constants (a fixed target speed, the maze's
        checkpoints).

        Never a verdict: not `task_metric`, not `success`, not the distance still to go.
        Under `evaluate.fitness.source: vlm_score` frames are scored by a VLM, and a
        label on a frame that showed the answer would make that score a copy of the
        ground truth it is supposed to be independent of.

        One stated exception: a STAGED task's current command. Basketball's "catch"
        then "throw", cabinet's "now slide the door open" then "now pull the drawer
        out" -- the stage is what the policy is being asked to do, and it is also,
        unavoidably, how far it has got (cabinet's four asks are its metric in
        quarters). The adapters that do this say so in their docstrings.

        Length: `bird/envs/hud.py` draws a 5x7 glyph at 2x, six scaled pixels per
        character, so a line of about 28 characters fits a 360-wide frame. The base
        implementation says nothing rather than something generic.
        """
        return []

    # -- crossing a process boundary (`train.candidate_parallelism: parallel`) --

    def export_states(self, state_arrays: Sequence[Any]) -> Any:
        """Anything this adapter must ship home so the PARENT can act on states
        a forked worker produced. `None` means "nothing to carry".

        Default is nothing, and for every adapter in `toy.py` and `control.py`
        that is the truth rather than a stub: they invert an observation
        arithmetically, so `step`, `render` and `reference_reward` work on any
        well-formed array regardless of which process produced it.

        `MetaWorld` is the exception and the reason this exists. MuJoCo is a
        stateful simulator and a 39-D Meta-World observation cannot be inverted
        to `(qpos, qvel)`, so that adapter keeps an obs -> snapshot cache and
        raises `UnknownStateError` on a miss. The cache is per-process: without
        this hook, a worker's trajectories are un-renderable and un-scoreable
        in the parent -- `task_metric` (a pure function of the rows) keeps
        working, so fitness looks fine while `reference_reward`,
        `gt_reward_curve`, the rollout videos and therefore rda's `vlm_score`
        and gt's `preference_bt` all die. That failure is invisible in the
        artifact, which is exactly what `budget.blind_comparisons` exists to
        catch.
        """
        return None

    def import_states(self, blob: Any) -> None:
        """Inverse of `export_states`, run in the parent. Default: nothing."""
        return None

    # -- surviving until the recorder replays them ----------------------------

    def retain_states(self, state_arrays: Sequence[Any]) -> None:
        """Declare that these states will be REPLAYED later, so keep them.

        Default is a no-op, and for `toy.py`/`control.py` that is the truth:
        they invert an observation arithmetically, so no state can go stale.

        `MetaWorld` is the exception and the reason this exists. Its obs ->
        snapshot cache is an LRU, and `observability.record_rollouts` replays a
        trajectory that was produced BEFORE everything the rest of the stage
        emitted -- measured on `mt10_window-open-v3`, one candidate
        emits 11,126 states (2,108 training + 7,515 checkpoint evaluation +
        1,503 rollout) against a 8,192-entry cache, so at two candidates the
        first candidate's rollout is 0/501 present by the time the recorder
        asks for it. No cache size fixes that, because the flood scales with
        `train.env_steps` while the cache does not.
        The adapter cannot tell an executed trajectory from a planner's
        lookahead by inspection, so the caller -- which knows -- says so here.

        Called by `components.training` on the rollout the recorder replays
        (rollout 0, the convention `observability.record_rollouts` and
        `preferences._clip_for` share), and released by
        `observability.record_rollouts` when the iteration's videos are on
        disk. It is a HINT, never a guarantee: an adapter may cap how much it
        retains, and a state that was never emitted cannot be retained at all.
        """
        return None

    def release_states(self) -> None:
        """Drop every `retain_states` claim. Default: nothing."""
        return None

    def sample_transitions(self, rng: Any, n: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """`n` (s, a, s_next) triples with coverage over the state-action space.

        The EPIC/STARC screens (`verify.quality_screen`) are pseudometrics
        between reward *functions*, so they need a distribution over transitions
        that does not depend on any policy. Sampled under *nominal* dynamics --
        a screen must not see this candidate's domain randomisation.
        """
        g = as_np_rng(rng)
        saved = self.episode_state()
        self._rng, self._dr_now = g, dict(self._dr_nominal)
        try:
            out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for _ in range(int(n)):
                s = self.random_state(g)
                a = self.action_set[int(g.integers(self.n_actions))].copy()
                s2, _done, _info = self._step(s, a)
                out.append((s, a, s2))
            return out
        finally:
            self.restore_episode_state(saved)

    # -- domain randomisation (`train.domain_randomization.*`) --------------

    def set_dr(self, params: Optional[Dict[str, Any]]) -> None:
        """Install a DR distribution; `None`/`{}` restores nominal dynamics.

        Values may be a `(lo, hi)` pair, a `{"low":, "high":}` mapping, or a
        bare scalar (fixed value) -- all three shapes come out of LLM-written
        `dr_config` blobs. Unknown keys are dropped and ranges are clipped into
        `dr_parameters` bounds rather than rejected; that leniency is
        repo-authored (DrEureka's release has no `parse_dr` -- it pastes the
        block verbatim, dr_eureka.py:109-128).
        """
        if not params:
            self._dr_ranges = {}
            self._dr_now = dict(self._dr_nominal)
            return
        ranges: Dict[str, Tuple[float, float]] = {}
        for key, value in dict(params).items():
            bounds = self.dr_parameters.get(str(key))
            if bounds is None:
                continue
            lo_b, hi_b = float(bounds[0]), float(bounds[1])
            try:
                if isinstance(value, dict):
                    pair = (value.get("low", value.get("min")), value.get("high", value.get("max")))
                    if pair[0] is None or pair[1] is None:
                        pair = tuple(value.get("range", (None, None)))[:2]
                    lo, hi = float(pair[0]), float(pair[1])
                elif isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
                    lo, hi = float(value[0]), float(value[1])
                else:
                    lo = hi = float(value)
            except (TypeError, ValueError):
                continue
            lo, hi = (lo, hi) if lo <= hi else (hi, lo)
            ranges[str(key)] = (min(max(lo, lo_b), hi_b), min(max(hi, lo_b), hi_b))
        self._dr_ranges = ranges
        self._dr_now = {**self._dr_nominal,
                        **{k: 0.5 * (lo + hi) for k, (lo, hi) in ranges.items()}}

    def _sample_dr(self, rng: np.random.Generator) -> Dict[str, float]:
        """One draw per episode -- the standard DR protocol, and the only one
        under which DrEureka's RAPP sweep means anything."""
        if not self._dr_ranges:
            return dict(self._dr_nominal)
        out = dict(self._dr_nominal)
        for key, (lo, hi) in self._dr_ranges.items():
            out[key] = float(lo) if hi <= lo else float(rng.uniform(lo, hi))
        return out

    @property
    def dr_config(self) -> Dict[str, Tuple[float, float]]:
        """The DR distribution currently installed (empty => nominal)."""
        return dict(self._dr_ranges)

    def dr_probe(self, policy_ref: Optional[str], overrides: Dict[str, Any],
                 n_rollouts: int = 3, *, cfg: Any = None) -> float:
        """Any-step success rate of the stored policy `policy_ref` under one DR override.

        DrEureka's RAPP pre-phase (`phases.run_rapp`, `pre: [rapp]`) sweeps each axis of
        `dr_parameters` and keeps the bracket where this rate clears
        `RAPP_SUCCESS_RATE`. `phases._dr_probe` tries this name first, so what this
        method returns IS the prior; a NaN makes RAPP decline to narrow that axis and
        keep the declared range, marked `degenerate` -- which is DrEureka's own
        uninformative-prior ablation arm, the one its paper shows fails badly.

        Implemented here on the base, so RAPP is live on every adapter (a probe that
        existed only on one adapter would leave RAPP inert -- every axis `degenerate`
        -- everywhere else). It builds on `components.training.policy_from_ref`, which
        knows all four blob kinds -- the two surrogates' arrays, sb3's archive and
        fasttd3's torch container, told apart by the container's own members (a
        fasttd3 blob is also 1-D uint8, so the shape alone would misroute it into the
        sb3 rebuild) -- and takes the run's `cfg` to answer the "which algorithm class
        to reconstruct" question, so RAPP is live on `train.backend: sb3` too.

        Semantics:

          * NaN when `_POLICY_STORE` holds nothing under `policy_ref` (no policy to roll
            out is not a zero success rate).
          * A blob that cannot be rebuilt RAISES (`policy_from_ref` never degrades to an
            untrained policy -- and never rebuilds one kind as another: the fasttd3
            branch checks the payload's own `fasttd3/v1` marker); `phases._dr_probe`
            turns the raise into NaN with the reason in its log, rather than a
            silent NaN.
          * `set_dr(dict(overrides))` for the rollouts, the previous `dr_config`
            restored in a `finally` -- `ctx.env` is one shared adapter and a probe that
            left its override installed would randomise the next stage's training.
          * `n_rollouts` full-horizon episodes on a fixed `default_rng(0)`, no early
            stop on `done`, scored by `self.success` on the state sequence -- the
            any-step check on Meta-World, the `task_metric` threshold everywhere else.

        `cfg` is the resolved run config, needed only for an SB3 blob (its algorithm and
        hyperparameters name the network the weights fit); a Q-table or linear policy
        rebuilds without it. Keyword-only, so the positional contract the phase's
        duck-typed probe has always used is unchanged.
        """
        # Lazy: `components.training` imports this module (`_bin`), so a module-level
        # import here would be a cycle.
        from ..components.training import _POLICY_STORE, policy_from_ref

        blob = _POLICY_STORE.get(policy_ref or "")
        if blob is None:
            return float("nan")
        # An adapter with no success criterion has no rate to report. `GymMujoco` overrides
        # `success()` to False unconditionally (its `success_threshold` is `inf`: no usable
        # random -> expert span), so a probe there would count a finite 0.0 at every value,
        # mark every axis `degenerate` and keep the declared range -- an uninformative prior
        # wearing a measurement's clothes. NaN says "unsupported", which
        # RAPP already treats as "decline to narrow", and the log names why.
        bar = getattr(self, "success_threshold", None)
        try:
            bar_ok = bar is not None and math.isfinite(float(bar))
        except (TypeError, ValueError):
            bar_ok = False
        if not bar_ok:
            _log.info("dr_probe: %s defines no finite success_threshold (%r); no success "
                      "rate can be measured, RAPP declines to narrow", type(self).__name__, bar)
            return float("nan")
        # Built BEFORE the override is installed, so a blob that cannot be rebuilt
        # raises with the adapter's DR exactly as the caller left it.
        policy = policy_from_ref(cfg, self, blob, None, ref=policy_ref or "policy")

        saved = self.dr_config
        self.set_dr(dict(overrides))
        try:
            rng = np.random.default_rng(0)
            n = max(1, int(n_rollouts))
            hits = 0.0
            for _ in range(n):
                s = self.reset(rng)
                states = [s]
                for _ in range(self.horizon):
                    s, _done, _info = self.step(s, policy(s))
                    states.append(s)
                hits += float(self.success(np.asarray(states, dtype=float)))
            return hits / n
        finally:
            self.set_dr(saved or None)

    # -- describe(): `generate.context.env_spec` ---------------------------

    def describe(self, kind: str = "natural_language_only") -> str:
        """Env text for one value of `generate.context.env_spec` (§1).

        The four renderings are not cosmetic. GT argues full source rarely
        exists for a real robot and passes only the state/action dataclass API
        (§4 fn.2); T2R and CARD pass a Pythonic class abstraction whose
        *callable helpers* the GT stub deliberately lacks; L2R passes prose
        only. Whether the LLM can name `check_grasp()` is the difference
        between those methods, so the difference lives in the text.

        Note that `full_source` includes `reference_reward` -- that is what
        "full source" means. Stripping it is `generate.context.strip_existing_
        reward`'s job in §1, and the marker comments around it are there so the
        stripper has something unambiguous to cut on.

        WHAT MUST NOT BE SAID SOMEWHERE ELSE INSTEAD. Everything here is gated on
        `env_spec`, and `problem.task_description` is not -- it reaches every method.
        So env internals written into the l_task are handed equally to the methods
        whose `env_spec` deliberately withholds them, and a per-method comparison
        becomes partly a comparison of what each method was told. The rule -- the
        goal goes in `task_description`, the env internals do not:

            not how much is in `task_description`, but whether the GOAL is fully
            there and the ENV INTERNALS are not.

        Both directions fail. Saying too little degrades methods *unequally*,
        correlated with which rendering each receives; saying too much puts env
        internals (a state-layout sentence, say) into `l_task`, the one channel no
        `env_spec` gates, so the methods whose rendering is chosen to withhold env
        internals get them anyway.

        `l2r` is not harmed by keeping internals out of `l_task`: it reads
        `natural_language_only`, which renders `_prose` and `_state_fields` -- so on a
        cart-pole it receives the angle convention BY CONSTRUCTION, before any
        `l_task` says a word.
        """
        key = (kind or "none").strip()
        if key == "none":
            return ""
        if key == "full_source":
            return self._render_full_source()
        if key == "state_action_api_stub":
            return self._render_api_stub()
        if key == "pythonic_class_abstraction":
            return self._render_class_abstraction()
        return self._render_natural_language()

    def _render_full_source(self) -> str:
        """The adapter's source, preceded by the source of any class it
        DELEGATES its dynamics to.

        UNSTRIPPED, and deliberately: `full_source` includes
        `reference_reward` because that is what "full source" means (see the
        `describe` docstring above); `strip_existing_reward` cuts it in §1
        and the marker comments exist so it has something to cut on.

        THE PARENT SEGMENTS EXIST FOR THE DELEGATING TIERS. A batched jax
        adapter holds a CPU adapter rather than subclassing it, so
        `inspect.getsource(type(self))` is the subclass alone -- which
        contains no ground-truth method, which means the strip has nothing to
        remove, which means `tests/test_env_spec_leak.py` PASSES WHILE
        GUARDING NOTHING. Rendering the parent first also makes the dynamics a
        model is shown identical across the two tiers of a family, which is
        the point of having two tiers.

        A non-jax rendering is unaffected: only a `BatchedEnvAdapter` defines
        `source_parents`, plain `EnvAdapter` subclasses have no such method,
        and an empty tuple prepends nothing.
        """
        try:
            src = inspect.getsource(type(self))
        except (OSError, TypeError):  # zipimport / exec'd module
            return self._render_class_abstraction()
        parts = []
        for parent in (getattr(self, "source_parents", lambda: ())() or ()):
            try:
                parts.append(inspect.getsource(parent))
            except (OSError, TypeError):
                # A parent whose source is unavailable is REPORTED in the
                # render, not skipped: a silently missing segment is the
                # unguarded state this method exists to prevent.
                parts.append(f"# parent `{parent.__name__}` -- source unavailable\n")
        parts.append(src)
        return f"# environment `{self.name}` -- full source\n\n" + "\n".join(parts)

    def _render_api_stub(self) -> str:
        lines = [f"# environment `{self.name}` -- state/action API only.",
                 "# No implementation is provided: infer dynamics from the field docs.",
                 "", "@dataclass", "class State:"]
        for i, (field, doc) in enumerate(self._state_fields):
            lines.append(f"    {field}: float   # s[{i}] -- {doc}")
        lines += ["", "@dataclass", "class Action:"]
        for i, (field, doc) in enumerate(self._action_fields):
            lines.append(f"    {field}: float   # a[{i}] -- {doc}")
        lines += ["", f"HORIZON = {self.horizon}"]
        return "\n".join(lines)

    def _render_class_abstraction(self) -> str:
        lines = [f"class {type(self).__name__}:",
                 f'    """{self._prose.strip()}"""', "",
                 "    # --- observed attributes (a state array `s`) ---"]
        for i, (field, doc) in enumerate(self._state_fields):
            lines.append(f"    {field}: float          # s[{i}]: {doc}")
        lines.append("")
        lines.append("    # --- action components (an action array `a`) ---")
        for i, (field, doc) in enumerate(self._action_fields):
            lines.append(f"    {field}: float          # a[{i}]: {doc}")
        if self._helpers:
            lines.append("")
            lines.append("    # --- helper methods you may call ---")
            for sig, doc in self._helpers:
                lines.append(f"    def {sig}: ...      # {doc}")
        lines += ["", f"    HORIZON = {self.horizon}",
                  f"    N_ACTIONS = {self.n_actions}"]
        return "\n".join(lines)

    def _render_natural_language(self) -> str:
        parts = [self._prose.strip(), "", "State variables, in order:"]
        parts += [f"  - s[{i}] {field}: {doc}"
                  for i, (field, doc) in enumerate(self._state_fields)]
        parts += ["", "Action variables, in order:"]
        parts += [f"  - a[{i}] {field}: {doc}"
                  for i, (field, doc) in enumerate(self._action_fields)]
        parts += ["", f"An episode lasts at most {self.horizon} steps."]
        return "\n".join(parts)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<{type(self).__name__} name={self.name} obs_dim={self.obs_dim} "
                f"actions={self.n_actions} horizon={self.horizon}>")


