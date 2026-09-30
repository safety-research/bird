"""Toy world models `M`, and the `EnvAdapter` interface every stage talks to.

`problem.env_id` (§0) and the inner loop `pi = A_M(R)` (§3).
These are not benchmarks. They exist so that a whole method -- generate,
verify, train, evaluate, select, update -- runs end to end offline, in under a
second, deterministically, with a *real* fitness signal. The one property that
matters is discrimination: a reward aligned with the task must train a policy
that scores well on the ground-truth metric, and a misaligned one must not.
Without that, every downstream number (fitness, TAC, TPE, Pearson curve,
Bradley-Terry strength) is noise dressed as evidence and the test suite proves
nothing.

Three envs, chosen to cover the three shapes the literature actually uses:

  toy_reacher         continuous 2-D point mass; the `_default.yaml` env, and
                      the shape Eureka/DrEureka/RDA/T2R all train on.
  toy_gridworld       discrete, with a lava ridge on the direct path. Reward
                      hacking is *expressible* here: `-distance_to_goal` alone
                      is optimised by walking into the lava, because dying ends
                      an all-negative episode early. A good reward must price
                      the terminal states.
  toy_hungry_thirsty  Singh, Lewis & Barto (2009), the pre-LLM limit case and
                      the reason `generator_backend: exhaustive_enumeration`
                      and `train.backend: tabular` exist. Fitness is
                      "fraction of steps not hungry"; the reward that also
                      prices *thirst* is the one that wins -- which is the
                      paper's whole point, and it is reproduced here.

Conventions every consumer depends on (screens, preferences and §4 all
re-score stored rollouts, so these are load-bearing):

  * A state is a 1-D float `np.ndarray` of length `obs_dim`. There is no hidden
    per-step state: `step()` is a function of `(state, action)` plus the
    adapter's RNG and its current domain-randomisation draw.
  * A `Trajectory` stores `states` with **T+1 rows** (s_0 .. s_T, terminal
    included) and `actions` with **T rows**, so
    `zip(states, actions, states[1:])` yields the transitions and
    `len(states) - 1 == len(actions) == length`.
  * Per-step reward is `R(s_t, a_t, s_{t+1})`; candidate rewards may take one,
    two or three of those (see `bird.components.training.compile_reward`).
  * `reference_reward(s, a)` scores the transition that *arrives* in `s` having
    taken `a`, i.e. the ground-truth return of a rollout is
    `sum_t reference_reward(s_{t+1}, a_t)`. `a` is optional so the reference can
    be evaluated on a state alone.
  * The adapter is stateful in exactly two places -- its RNG (set by `reset`)
    and its DR draw -- and it is **not** reentrant: do not interleave two
    episodes on one adapter. `ctx.env` is a single shared instance, so a
    training backend that calls `set_dr` must restore it (see §3).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..registry import register

__all__ = ["EnvAdapter", "ToyReacher", "ToyGridworld", "ToyHungryThirsty"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


#: `EnvAdapter` and the two helpers live in `base.py` -- `spec.py` subclasses
#: the adapter and this module needs `SpecEnvAdapter` back, which is a cycle
#: unless the base has its own home. Re-exported so existing imports still work.
# Re-exported: `phases.py`, the training backends and the tests all import these
# from here, and moving the base class is not a reason to move their imports.
from .base import EnvAdapter, as_np_rng, _bin, _states_of  # noqa: F401
from .spec import SpecEnvAdapter


# ==========================================================================
# toy_reacher -- continuous 2-D point mass
# ==========================================================================


class ToyReacher(SpecEnvAdapter):
    """2-D point mass that must reach a fixed goal *and stay there*.

    The default env of `configs/_default.yaml`, and the shape the continuous-
    control methods (Eureka, DrEureka, RDA, Text2Reward) all train on. Momentum
    is real: the force is damped, so a policy that only sprints at the goal
    overshoots and the ground-truth metric -- fraction of steps spent inside the
    goal radius -- punishes it. That is what makes a velocity/effort term in a
    candidate reward worth something rather than decoration.

    The episode never terminates early. "Stay there" is only measurable over a
    fixed horizon, and early termination would make the metric length-dependent
    (the same length bias `Trajectory.mean_per_step_return` exists to correct).
    """

    name = "toy_reacher"
    obs_dim = 6
    action_dim = 2
    horizon = 25
    #: A fifth of the episode inside the radius. Calibrated, not guessed: a
    #: well-shaped reward trains to ~0.5 here and every degenerate reward tried
    #: (constant, speed-seeking, sign-flipped distance) trains to <0.05.
    success_threshold = 0.20

    goal = (0.5, 0.5)
    goal_radius = 0.22
    dt = 0.18
    #: Error-centred bins: the middle position bin is exactly +/-`goal_radius`
    #: wide, so "in the centre cell" and "at the goal" are the same success check.
    #: Aligning them matters -- with bins laid over raw position the goal
    #: straddles a boundary and no tabular policy can hold station.
    _pos_bins = 9
    _vel_bins = 2
    n_disc_states = 9 * 9 * 2 * 2
    exact_states = None  # continuous: `discretise` is a binning, not a bijection




    def _build_action_set(self) -> Any:
        # 3x3 grid of forces: enough to brake on either axis, small enough that
        # a discretised learner covers it in a few thousand steps.
        return [[fx, fy] for fx in (-1.0, 0.0, 1.0) for fy in (-1.0, 0.0, 1.0)]

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array([-1, -1, -1, -1, -1, -1], dtype=float),
                np.array([1, 1, 1, 1, 1, 1], dtype=float))

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        # Start in the far quadrant so every episode requires real travel; the
        # spread is what gives the checkpoint curve something to average over.
        s = np.empty(6)
        s[0] = rng.uniform(-0.85, -0.25)
        s[1] = rng.uniform(-0.85, -0.25)
        s[2] = 0.0
        s[3] = 0.0
        s[4], s[5] = self.goal
        return s

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        dr = self._dr_now
        ax = float(a[0]) if a.size > 0 else 0.0
        ay = float(a[1]) if a.size > 1 else 0.0
        noise = dr["action_noise"]
        if noise > 0.0:
            ax += float(self._rng.normal(0.0, noise))
            ay += float(self._rng.normal(0.0, noise))
        ax = -1.0 if ax < -1.0 else (1.0 if ax > 1.0 else ax)
        ay = -1.0 if ay < -1.0 else (1.0 if ay > 1.0 else ay)

        gain = dr["force_scale"] / max(dr["mass"], 1e-3)
        vx = dr["damping"] * float(s[2]) + gain * ax
        vy = dr["damping"] * float(s[3]) + gain * ay
        vx = -1.0 if vx < -1.0 else (1.0 if vx > 1.0 else vx)
        vy = -1.0 if vy < -1.0 else (1.0 if vy > 1.0 else vy)
        px = float(s[0]) + self.dt * vx
        py = float(s[1]) + self.dt * vy
        px = -1.0 if px < -1.0 else (1.0 if px > 1.0 else px)
        py = -1.0 if py < -1.0 else (1.0 if py > 1.0 else py)

        s2 = np.array([px, py, vx, vy, s[4], s[5]], dtype=float)
        dist = float(np.hypot(px - float(s[4]), py - float(s[5])))
        return s2, False, {"success": dist <= self.goal_radius, "distance": dist}

    def discretise(self, s: np.ndarray) -> int:
        # Bin the goal-relative *error*, not raw position, and bin velocity by
        # sign only. Coarse on purpose: a table this learner has to fill inside
        # `train.env_steps` on a laptop cannot be larger than a few hundred
        # cells, and sign-of-velocity is all a braking decision needs given the
        # damping above. `train.backend: tabular` on a continuous env is a
        # binning, not an enumeration -- hence `exact_states is None`.
        p = self._pos_bins
        v = self._vel_bins
        bx = _bin(float(s[4]) - float(s[0]), -2.0, 2.0, p)
        by = _bin(float(s[5]) - float(s[1]), -2.0, 2.0, p)
        bvx = _bin(float(s[2]), -1.0, 1.0, v)
        bvy = _bin(float(s[3]), -1.0, 1.0, v)
        return ((bx * p + by) * v + bvx) * v + bvy

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        s = np.empty(6)
        s[0] = rng.uniform(-1.0, 1.0)
        s[1] = rng.uniform(-1.0, 1.0)
        s[2] = rng.uniform(-1.0, 1.0)
        s[3] = rng.uniform(-1.0, 1.0)
        s[4], s[5] = self.goal
        return s

    def task_metric(self, traj: Any) -> float:
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        p = states[1:, 0:2]
        g = states[1:, 4:6]
        dist = np.linalg.norm(p - g, axis=1)
        return float(np.mean(dist <= self.goal_radius))

    # -- what the task asks, for a frame legend (`EnvAdapter.legend_lines`) ----

    def legend_lines(self, state: np.ndarray) -> List[str]:
        """The ask, this episode's goal read out of the state, and the goal radius.

        `s[4:6]` is the goal -- the block `_reset` writes and `task_metric` measures
        against -- and it is read from the ARRAY, never from `self.goal`: the legend
        is a pure function of the row it is handed (the rule `MetaWorld.legend_lines`
        follows), so a stored row whose goal differs from the class constant is
        captioned with ITS goal. Position and velocity are in the state too and are
        deliberately absent: beside the goal they are the distance still to go, which
        is the verdict in numbers (`EnvAdapter.legend_lines`). The radius is the
        ask's own tolerance, a constant.

        Two decimals, and `-0.00` normalised to `0.00` -- a sign on a zero reads as a
        direction the goal does not have. A row too short to carry a goal block gets
        the ask alone: saying less rather than something wrong. `legend_on_frame` and
        `step_success_is_a_check` keep their defaults -- this class has no `render`
        (its frames are rasterised outside the adapter), and `_step`'s per-step flag
        is the real inside-the-radius check.
        """
        s = np.asarray(state, dtype=float).ravel()
        lines = ["REACH THE GOAL AND STAY"]
        if s.size >= 6:
            gx, gy = (np.round(s[4:6], 2) + 0.0).tolist()
            lines.append(f"GOAL X {gx:.2f} Y {gy:.2f}")
        lines.append(f"GOAL RADIUS {self.goal_radius:.2f}")
        return lines

    # --- BEGIN reference reward (ground truth; strip before showing an LLM) ---
    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """The human-written dense reward: get close, arrive slowly, stay.

        Scores the transition that *arrives* in `s`. The three terms are the
        canonical hand-tuned recipe -- distance, an at-goal bonus that makes
        holding strictly better than orbiting, and small velocity/effort costs
        that price the overshoot the dynamics create.
        """
        s = np.asarray(s, dtype=float)
        dist = float(np.hypot(s[0] - s[4], s[1] - s[5]))
        r = -dist - 0.05 * float(np.hypot(s[2], s[3]))
        if dist <= self.goal_radius:
            r += 1.0
        if a is not None:
            aa = np.asarray(a, dtype=float).ravel()
            r -= 0.01 * float(np.dot(aa, aa))
        return r
    # --- END reference reward ---


# ==========================================================================
# toy_gridworld -- discrete, with reward hacking expressible
# ==========================================================================


class ToyGridworld(SpecEnvAdapter):
    """6x6 grid with a lava ridge on the diagonal. Reach the goal; do not die.

    This env exists so that misalignment is *reachable by optimisation*, not
    just by writing nonsense. Entering lava ends the episode. A candidate
    reward of `-distance_to_goal` -- the single most natural thing an LLM
    writes -- is therefore maximised by walking into the lava, because ending
    an all-negative episode early beats surviving it. The ground-truth metric
    (did the agent reach the goal) collapses, exactly as it should. Pricing the
    terminal states, as `reference_reward` does, fixes it.

    A monotone path from any start to (5, 5) that avoids the diagonal always
    exists (go all the way along one axis, then the other), so the task is
    solvable without the detour being contrived.
    """

    name = "toy_gridworld"
    obs_dim = 5
    action_dim = 1
    horizon = 30
    size = 6
    goal_cell = (5, 5)
    hazards = ((1, 1), (2, 2), (3, 3), (4, 4))
    success_threshold = 0.5  # task_metric is {0, 1}; the threshold just binarises
    n_disc_states = 36
    exact_states = 36




    _MOVES = ((0, 1), (0, -1), (1, 0), (-1, 0))

    def _build_action_set(self) -> Any:
        return [[0.0], [1.0], [2.0], [3.0]]

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        n = float(self.size - 1)
        return (np.zeros(5), np.array([n, n, n, n, 1.0]))

    def _obs(self, x: int, y: int) -> np.ndarray:
        return np.array([float(x), float(y), float(self.goal_cell[0]),
                         float(self.goal_cell[1]),
                         1.0 if (x, y) in self.hazards else 0.0], dtype=float)

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        # Random safe, non-goal start: a fixed start would make every episode
        # identical under a deterministic transition and the learning curve a
        # step function rather than a curve.
        while True:
            x = int(rng.integers(self.size))
            y = int(rng.integers(self.size))
            if (x, y) not in self.hazards and (x, y) != self.goal_cell:
                return self._obs(x, y)

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        dr = self._dr_now
        idx = int(round(float(a[0]))) if a.size else 0
        idx = 0 if idx < 0 else (len(self._MOVES) - 1 if idx >= len(self._MOVES) else idx)
        if dr["sticky_prob"] > 0.0 and self._rng.random() < dr["sticky_prob"]:
            dx, dy = 0, 0
        else:
            if dr["slip_prob"] > 0.0 and self._rng.random() < dr["slip_prob"]:
                idx = int(self._rng.integers(len(self._MOVES)))
            dx, dy = self._MOVES[idx]
        x = min(max(int(s[0]) + dx, 0), self.size - 1)
        y = min(max(int(s[1]) + dy, 0), self.size - 1)

        at_goal = (x, y) == self.goal_cell
        in_lava = (x, y) in self.hazards
        return self._obs(x, y), bool(at_goal or in_lava), \
            {"success": at_goal, "hazard": in_lava}

    def discretise(self, s: np.ndarray) -> int:
        return int(s[0]) * self.size + int(s[1])

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        return self._obs(int(rng.integers(self.size)), int(rng.integers(self.size)))

    def task_metric(self, traj: Any) -> float:
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        gx, gy = self.goal_cell
        last = states[-1]
        return 1.0 if (int(last[0]), int(last[1])) == (gx, gy) else 0.0

    # -- what the task asks, for a frame legend (`EnvAdapter.legend_lines`) ----

    def legend_lines(self, state: np.ndarray) -> List[str]:
        """Two lines: the goal cell, read out of the state, and the standing order.

        `s[2:4]` is the goal cell `_obs` writes, read from the array for the same
        reason `ToyReacher.legend_lines` reads its block: the legend is a function of
        the row, not of the class. The agent's cell (`s[0:2]`) and the in-lava flag
        (`s[4]`) are deliberately absent -- the first is the distance to go and the
        second is the episode's death, both verdicts. The lava's position is drawn on
        the frame by the rasteriser, so the line says only what to do about it.
        """
        s = np.asarray(state, dtype=float).ravel()
        if s.size >= 4:
            ask = f"REACH CELL {int(round(s[2]))},{int(round(s[3]))}"
        else:
            ask = "REACH THE GOAL CELL"
        return [ask, "AVOID THE LAVA"]

    # --- BEGIN reference reward (ground truth; strip before showing an LLM) ---
    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """Human-written reward: big terminal values, small shaping.

        The lava penalty has to exceed the worst surviving episode (30 steps of
        the step cost), otherwise dying is still the cheaper way out and the
        reward reproduces the very hack this env exists to expose.
        """
        s = np.asarray(s, dtype=float)
        x, y = int(s[0]), int(s[1])
        if (x, y) == self.goal_cell:
            return 5.0
        if (x, y) in self.hazards:
            return -5.0
        dist = abs(x - self.goal_cell[0]) + abs(y - self.goal_cell[1])
        return -0.02 - 0.05 * (dist / (2.0 * (self.size - 1)))
    # --- END reference reward ---


# ==========================================================================
# toy_hungry_thirsty -- Singh, Lewis & Barto (2009)
# ==========================================================================


class ToyHungryThirsty(SpecEnvAdapter):
    """The hungry-thirsty domain: the tabular / enumeration limit case.

    Singh, Lewis & Barto (2009), "Where do rewards come from?". A 4x4 open grid
    with food at (0, 0) and water at (0, 2) -- an edge cell, not a corner. Eating
    only works when the agent is
    not thirsty, and hunger returns on every step that the agent does not eat.
    **Fitness is the fraction of steps spent not hungry** -- and the reward that
    maximises fitness is *not* the reward that mirrors it. A reward paying only
    for "not hungry" gives an agent no reason to value water, which is the
    paper's finding and the reason `reference_reward` here carries a thirst
    term the fitness function does not have.

    This is the one env where `exact_states` is a real enumeration (64 states),
    which is what makes `train.backend: tabular` and
    `generate.generator_backend: exhaustive_enumeration` (`configs/singh_orp`)
    honest rather than decorative.

    Simplified from the paper in five respects, recorded so nobody mistakes it
    for a reproduction: no interior walls (theirs lengthen the food-water trip);
    a 4x4 grid, 64 states, where the paper's is 6x6 in four 3x3 subspaces
    (144); food and water fixed (the paper draws two corners per environment);
    deterministic moves (the paper's succeed probabilistically, value unstated);
    and 60-step episodes with a reset (the paper's lifetime is continuous). The
    open grid and short trip are there because
    the surrogate learner has to find the food-water shuttle inside a few
    thousand env steps and a longer trip is not discoverable by e-greedy
    exploration at that budget.
    """

    name = "toy_hungry_thirsty"
    obs_dim = 4
    action_dim = 1
    horizon = 60
    size = 4
    food_cell = (0, 0)
    water_cell = (0, 2)
    #: Calibrated: the reference reward trains to ~0.33 here, a reward that
    #: merely copies the fitness function ("+1 when not hungry") to ~0.10, and
    #: rewards that ignore food to ~0.00. The threshold sits between the first
    #: two, which is what makes Singh's result visible as a success rate.
    success_threshold = 0.15
    n_disc_states = 4 * 4 * 2 * 2
    exact_states = 4 * 4 * 2 * 2




    _MOVES = ((0, 1), (0, -1), (1, 0), (-1, 0))
    EAT = 4
    DRINK = 5

    def _build_action_set(self) -> Any:
        return [[float(i)] for i in range(6)]

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        n = float(self.size - 1)
        return (np.zeros(4), np.array([n, n, 1.0, 1.0]))

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([float(rng.integers(self.size)), float(rng.integers(self.size)),
                         1.0, 0.0], dtype=float)

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        dr = self._dr_now
        act = int(round(float(a[0]))) if a.size else 0
        act = 0 if act < 0 else (5 if act > 5 else act)
        x, y = int(s[0]), int(s[1])
        thirsty = bool(s[3] >= 0.5)
        ate = False

        if act < 4:
            if dr["slip_prob"] > 0.0 and self._rng.random() < dr["slip_prob"]:
                act = int(self._rng.integers(4))
            dx, dy = self._MOVES[act]
            x = min(max(x + dx, 0), self.size - 1)
            y = min(max(y + dy, 0), self.size - 1)
        elif act == self.EAT:
            if (x, y) == self.food_cell and not thirsty:
                if dr["eat_fail_prob"] <= 0.0 or self._rng.random() >= dr["eat_fail_prob"]:
                    ate = True
        elif act == self.DRINK:
            if (x, y) == self.water_cell:
                thirsty = False

        # Hunger returns unless the agent just ate; thirst arrives on its own.
        hungry = not ate
        if not thirsty and self._rng.random() < dr["thirst_prob"]:
            thirsty = True

        s2 = np.array([float(x), float(y), 1.0 if hungry else 0.0,
                       1.0 if thirsty else 0.0], dtype=float)
        return s2, False, {"success": not hungry, "ate": ate, "thirsty": thirsty}

    def discretise(self, s: np.ndarray) -> int:
        h = 1 if s[2] >= 0.5 else 0
        t = 1 if s[3] >= 0.5 else 0
        return ((int(s[0]) * self.size + int(s[1])) * 2 + h) * 2 + t

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([float(rng.integers(self.size)), float(rng.integers(self.size)),
                         float(rng.integers(2)), float(rng.integers(2))], dtype=float)

    def task_metric(self, traj: Any) -> float:
        """Singh's fitness, verbatim: the fraction of steps spent not hungry.

        Note what is *absent*: thirst. The whole result of the paper is that the
        reward maximising this does not look like this.
        """
        states = _states_of(traj)
        if states.shape[0] < 2:
            return 0.0
        return float(np.mean(states[1:, 2] < 0.5))

    # -- what the task asks, for a frame legend (`EnvAdapter.legend_lines`) ----

    def legend_lines(self, state: np.ndarray) -> List[str]:
        """Two constant lines: where the food is and what to do there, and the same
        for the water. Nothing from `state`, and the omission is the point.

        Hunger (`s[2]`) is the one thing a reader would want on the frame, and it is
        exactly what must not be there: `_step`'s per-step flag is "not hungry" and
        `task_metric` is the fraction of steps not hungry, so `HUNGRY` / `FED` on a
        frame IS the per-step verdict, verbatim -- the contamination
        `EnvAdapter.legend_lines` forbids, however much it is also the condition the
        ask turns on. Thirst (`s[3]`) goes with it: it is not the metric, but it is
        the state `reference_reward` prices and the paper's whole finding, and a
        legend that printed one flag and withheld the other would read as if the
        withheld one did not exist. A demo clip that WANTS the answer beside the
        question draws it in its own rasteriser, never here.

        Cells from `food_cell` / `water_cell` so the frame cannot name a corner the
        dynamics do not use.
        """
        fx, fy = self.food_cell
        wx, wy = self.water_cell
        return [f"EAT AT FOOD {fx},{fy} WHEN HUNGRY",
                f"IF THIRSTY, DRINK AT {wx},{wy}"]

    # --- BEGIN reference reward (ground truth; strip before showing an LLM) ---
    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """Singh's *better-than-fitness* reward: pay for satiety, price thirst.

        The thirst term is the paper's discovery. It is deliberately not in
        `task_metric`, so a candidate reward that merely copies the fitness
        function scores measurably worse than this one -- which is the entire
        experiment `configs/singh_orp.yaml` runs.
        """
        s = np.asarray(s, dtype=float)
        return (1.0 if s[2] < 0.5 else 0.0) - (0.05 if s[3] >= 0.5 else 0.0)
    # --- END reference reward ---


# ==========================================================================
# registry: `problem.env_id`
# ==========================================================================


@register("env", "toy_reacher")
def toy_reacher(ctx: Any) -> EnvAdapter:
    """Continuous 2-D reaching; the `_default.yaml` env (§0)."""
    return ToyReacher()


@register("env", "toy_gridworld")
def toy_gridworld(ctx: Any) -> EnvAdapter:
    """Discrete grid with a lava ridge -- reward hacking is expressible (§0)."""
    return ToyGridworld()


@register("env", "toy_hungry_thirsty")
def toy_hungry_thirsty(ctx: Any) -> EnvAdapter:
    """Singh et al. (2009); the tabular / enumeration limit case (§0, §3)."""
    return ToyHungryThirsty()


# --------------------------------------------------------------------------
# ToyReacher's shipped expert -- OUTSIDE the class body, on purpose
# --------------------------------------------------------------------------
#
# `EnvAdapter.expert_policy()` is the third source `bird.demos._expert` reads
# (after the `policies/` registry and Meta-World's bundled scripted policies),
# and `ToyReacher` is the one env of ours that ships an analytic solution. It is
# bound here, after the class, and not written as a method inside it, because
# `generate.context.env_spec: full_source` hands the generator
# `inspect.getsource(type(self))` -- the CLASS BODY, verbatim -- and a solution
# policy inside that body is a demonstration handed to the LLM on every
# `full_source` run (and, by changing the prompt, it would make the mock
# generator draw a different reward). A method assigned after the class is not
# in the class's source and never reaches a prompt.


def toy_reacher_expert(s: np.ndarray, t: int) -> np.ndarray:  # noqa: ARG001 -- stateless
    """PD law on the goal error: `a = clip(3 (g - p) - v)`.

    The damped point mass is linear, so a proportional pull with a velocity
    brake IS the solution, not an approximation of one. Measured over 20 reset
    seeds under nominal DR: task_metric mean 0.766, min 0.72
    (random policy 0.002; `success_threshold` 0.20) -- the ~6 of 25 steps spent
    travelling from the far quadrant are the whole gap to 1. Gains (3, 1) were
    the best of seven pairs tried; (8, 3) already overshoots to 0.64. This is
    the demonstration `bird.demos` rolls out under a candidate reward for
    `train.pruning_cfg.ceiling: demo_return` and `verify.quality_screen:
    demo_margin` under the tester profile.
    """
    err = np.asarray(s[4:6], dtype=float) - np.asarray(s[0:2], dtype=float)
    return np.clip(3.0 * err - 1.0 * np.asarray(s[2:4], dtype=float), -1.0, 1.0)


def _toy_reacher_expert_policy(self: ToyReacher) -> Callable[[np.ndarray, int], np.ndarray]:
    return toy_reacher_expert


ToyReacher.expert_policy = _toy_reacher_expert_policy  # type: ignore[method-assign]
