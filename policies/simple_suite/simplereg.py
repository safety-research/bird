"""Registry factories for `policies/simple_suite` (closure pattern).

Each factory builds the adapter for its env (the controllers read constants off it) and
returns `pol(obs, st)`; `st["t"]` carries the step index the controllers take. `params`
is accepted for the loader's contract and unused: these controllers have no tuned
constants beyond the ones in their code, which is what `params: {}` in the manifest says.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# Load the sibling module by path under a campaign-specific name: putting this
# directory on sys.path and binding plain `controllers` in sys.modules would
# collide with any other campaign that ships a controllers.py.
_spec = importlib.util.spec_from_file_location(
    "simple_suite_controllers", Path(__file__).resolve().parent / "controllers.py")
controllers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(controllers)


def _env(env_id: str):
    from bird import registry
    registry.load_all()
    return registry.get("env", env_id)(None)


def _wrap(make, env_id: str):
    env = _env(env_id)
    act = make(env, np.random.default_rng(0))

    def pol(obs, st):
        t = st.get("t", 0)
        a = act(np.asarray(obs, dtype=float), t)
        st["t"] = t + 1
        return a
    return pol


def pendulum_energy(params):
    """`simple_suite/pendulum_energy`: energy-shaping swing-up, PD hold at the top."""
    return _wrap(controllers.make_pendulum_energy, "pendulum")


def pendulum_discrete_energy(params):
    """`simple_suite/pendulum_discrete_energy`: the same law, snapped to the 5-torque set."""
    return _wrap(controllers.make_pendulum_energy, "pendulum_discrete")


def acrobot_pump(params):
    """`simple_suite/acrobot_pump`: bang-bang elbow pumping; brakes over the goal line."""
    return _wrap(controllers.make_acrobot_pump, "acrobot")


def toy_reacher_pd(params):
    """`simple_suite/toy_reacher_pd`: PD toward the goal with velocity damping."""
    return _wrap(controllers.make_toy_reacher_pd, "toy_reacher")


def toy_gridworld_bfs(params):
    """`simple_suite/toy_gridworld_bfs`: shortest path around the lava ridge, per step."""
    return _wrap(controllers.make_toy_gridworld_bfs, "toy_gridworld")


def toy_hungry_thirsty_shuttle(params):
    """`simple_suite/toy_hungry_thirsty_shuttle`: drink when thirsty, otherwise eat."""
    return _wrap(controllers.make_toy_hungry_thirsty_shuttle, "toy_hungry_thirsty")
