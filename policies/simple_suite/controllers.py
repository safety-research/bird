"""Closed-loop scripted controllers for BIRD's own pure-numpy tier (`bird_control`).

Each `make_<task>(env, rng) -> act(s, t)` is
a closure over the adapter's own constants (the pendulum's g/l/max_torque, the grid's
hazards, the food and water cells) and maps a state to an action every step -- closed
loop, nothing taped. Scores are the adapter's `task_metric` (per-step fraction) and are
recorded by scripts/eval_policy.py; the acrobot controller is honest about not balancing
(it pumps to the goal line and whips through it), which is why its status is negative.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict

import numpy as np


def make_pendulum_energy(env: Any, rng: np.random.Generator) -> Callable:
    """Energy-shaping swing-up, then a PD hold inside the upright window.

    `Pendulum._dynamics` is Pendulum-v1's: theta_ddot = 3g/(2l) sin(theta) +
    3/(m l^2) u with theta = 0 upright, so the potential is (3g/2l) cos(theta)
    and the upright-at-rest energy is E* = 3g/(2l). Below E* the torque pushes
    WITH the velocity (adds energy), above it against; near the top a PD law
    takes over. `max_torque = 2` cannot lift the rod from horizontal, which is
    exactly why the pumping phase is needed.
    """
    g, length, mt = float(env.g), float(env.l), float(env.max_torque)
    e_star = 3.0 * g / (2.0 * length)
    action_set = getattr(env, "action_set", None)
    discrete = action_set is not None and action_set.shape[0] <= 9 \
        and getattr(env, "exact_states", None) is not None

    def act(s: np.ndarray, t: int) -> np.ndarray:
        theta = math.atan2(float(s[1]), float(s[0]))
        thdot = float(s[2])
        if abs(theta) < 0.6 and abs(thdot) < 2.5:
            u = -12.0 * theta - 2.5 * thdot
        else:
            energy = 0.5 * thdot * thdot + e_star * math.cos(theta)
            u = 3.0 * (e_star - energy) * (1.0 if thdot >= 0 else -1.0)
            if abs(thdot) < 1e-3:
                u = mt * (1.0 if theta >= 0 else -1.0)
        u = float(np.clip(u, -mt, mt))
        if discrete:
            torques = action_set[:, 0]
            u = float(torques[int(np.argmin(np.abs(torques - u)))])
        return np.array([u], dtype=float)

    return act


def make_acrobot_pump(env: Any, rng: np.random.Generator) -> Callable:
    """Bang-bang energy pumping at the elbow; brake once the tip is over the bar.

    Below the goal line the elbow torque runs AGAINST the first link's angular
    velocity, which in this adapter's sign convention (theta_1 = 0 hanging down,
    torque positive on theta_2) feeds energy into the coupled swing -- measured,
    not assumed: `-sign(theta_dot_1)` lifts the tip to ~2.0 link-lengths in ~75
    steps on every seed tried, `+sign` never gets it above -0.15. Over the bar
    the sign flips to take energy out. It does not balance -- the tip whips
    through the goal region repeatedly -- so the fraction-of-episode metric it
    scores (0.08-0.21 on seeds 0-3, under the 0.3 threshold) is honest about that;
    the registry record (records/eval_acrobot_pump_*.json) has the number.
    """
    mt = float(env.max_torque)
    goal = float(env.goal_height)

    def act(s: np.ndarray, t: int) -> np.ndarray:
        d1 = float(s[4])
        pump = -mt if d1 >= 0.0 else mt
        return np.array([pump if env.tip_height(s) < goal else -pump], dtype=float)

    return act


def make_toy_reacher_pd(env: Any, rng: np.random.Generator) -> Callable:
    """PD toward the goal with velocity damping, clipped to the action box."""
    def act(s: np.ndarray, t: int) -> np.ndarray:
        ex, ey = float(s[4]) - float(s[0]), float(s[5]) - float(s[1])
        ax = 6.0 * ex - 3.0 * float(s[2])
        ay = 6.0 * ey - 3.0 * float(s[3])
        return np.clip(np.array([ax, ay], dtype=float), -1.0, 1.0)

    return act


def make_toy_gridworld_bfs(env: Any, rng: np.random.Generator) -> Callable:
    """Shortest path to the goal around the lava ridge, recomputed per step."""
    size = int(env.size)
    hazards = set(tuple(h) for h in env.hazards)
    goal = tuple(env.goal_cell)
    moves = list(env._MOVES)

    def first_move(x: int, y: int) -> int:
        from collections import deque  # noqa: PLC0415
        start = (x, y)
        prev: Dict[tuple, tuple] = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            if cur == goal:
                break
            for i, (dx, dy) in enumerate(moves):
                nxt = (min(max(cur[0] + dx, 0), size - 1), min(max(cur[1] + dy, 0), size - 1))
                if nxt in hazards or nxt in prev:
                    continue
                prev[nxt] = (cur, i)
                q.append(nxt)
        if goal not in prev:
            return 0
        node = goal
        while prev[node][0] != start:
            node = prev[node][0]
        return prev[node][1]

    def act(s: np.ndarray, t: int) -> np.ndarray:
        return np.array([float(first_move(int(s[0]), int(s[1])))], dtype=float)

    return act


def make_toy_hungry_thirsty_shuttle(env: Any, rng: np.random.Generator) -> Callable:
    """Drink when thirsty, otherwise eat: the food-water shuttle Singh's reward finds."""
    food, water = tuple(env.food_cell), tuple(env.water_cell)
    moves = list(env._MOVES)

    def step_toward(x: int, y: int, tx: int, ty: int) -> int:
        if x != tx:
            return moves.index((1 if tx > x else -1, 0))
        return moves.index((0, 1 if ty > y else -1))

    def act(s: np.ndarray, t: int) -> np.ndarray:
        x, y = int(s[0]), int(s[1])
        thirsty = float(s[3]) >= 0.5
        if thirsty:
            a = env.DRINK if (x, y) == water else step_toward(x, y, *water)
        else:
            a = env.EAT if (x, y) == food else step_toward(x, y, *food)
        return np.array([float(a)], dtype=float)

    return act
