#!/usr/bin/env python3
"""Registry wiring for `h1hand_hb_hacks/hop` and `h1hand_hb_hacks/moonhop`.

The controllers were written against `hk.Rig`, a harness that builds its OWN
`gym.make("h1hand-walk-v0")`, resets it with the gym seed and steps it with `env.step` --
the upstream instrument HumanoidBench's `success_bar` thresholds (`hk.py`'s docstring). A
registry policy must run over the BIRD adapter instead (`policies/README.md`: the adapter
is the instrument of record), so this module adapts and the controller files beside it
stay as they were written:

- `Shim` is `hk.Rig` built over a simulator that already exists -- the adapter's own
  `_env` (the unwrapped `HumanoidEnv`) -- instead of over a gym env of its own. Its
  constructor sets exactly the attributes `hk.Rig.__init__` sets (`m`, `d`, `u`, `names`,
  `idx`, `lo`, `hi`, `default`, `pelvis`), computed by the same expressions, and never
  resets or steps anything: the adapter has already reset, and a second draw would move
  the episode off the adapter's initial state. The sensing methods the controllers call
  (`com`, `comvel`, `q`, `site`, `pitch`) are `hk.Rig`'s own, inherited unchanged.
  `step` is refused -- `bird.policy_api.RigPolicy` hands the ctrl to the adapter, which
  integrates.
- `make_hop` / `make_moonhop` wrap `hop4.Hop` / `moon.Moon` (the controllers,
  called exactly as their own `run()` calls them: `ctl(rig, k)` once per control step with
  `k` from 0) into the registry's `fn(rig, t, ctrl) -> ctrl`. The controllers return a dict
  of joint-name -> position target; `_raw` lays it over `rig.default` and clips to
  `ctrlrange`, which is `hk.Rig.action` WITHOUT its last line (the normalisation), because
  `RigPolicy.act` normalises with the same expression `2*(c-lo)/(hi-lo)-1`. The incoming
  `ctrl` is ignored, as `hk.Rig.action` ignores any previous command.

Where the adapter and `hk.Rig` differ, and the numbers are allowed to: the
adapter restores the state vector and zeroes `qacc_warmstart` before every step (the gym
env does neither), and `scripts/eval_policy.py` sums the shipped reward recomputed at each
arrived state after the rollout rather than reading `env.step`'s. The reset is NOT a
difference: adapter seed `s` and `gym.make("h1hand-walk-v0").reset(seed=s)` give the same
initial qpos/qvel (max |diff| 0.0 on seeds 0, 1, 38 and 1000, measured).
The records in `records/eval_*` are what `scripts/eval_policy.py` measured through this
module; the gym-env numbers from development are in `NOTES_hop.md` / `NOTES_moonhop.md`
and are not retyped into any score.

`hop4.py` and `snap_hop3.py` each prepend their parent directory (`policies/`) to
`sys.path` at import, to find `hk` one level up, where it lived during development; here
`hk.py` sits beside them and `import_beside` imports it first, so that insert is inert.
"""
from pathlib import Path

import numpy as np

from bird.policy_api import import_beside

HERE = Path(__file__).resolve().parent

# hk first, so hop4's / snap_hop3's `from hk import Rig` find THIS copy; moon imports
# snap_hop3 as `H`.
hk, hop4, snap_hop3, moon = import_beside(HERE, "hk", "hop4", "snap_hop3", "moon")

import mujoco  # noqa: E402 -- after hk, which sets MUJOCO_GL before mujoco is loaded


class Shim(hk.Rig):
    """`hk.Rig` over an existing HumanoidBench simulator (the adapter's `_env`).

    `RigPolicy._build_rig` offers the adapter first and walks its `_env` chain on
    failure; the BIRD adapter has no `model`, so the first attribute read below raises and
    the unwrapped `HumanoidEnv` underneath is what this is built on."""

    def __init__(self, env):
        m, d = env.model, env.data
        env.named  # noqa: B018 -- hk.Rig.comvel reads u.named.data.sensordata; fail here, not mid-episode
        self.env = None          # hk.Rig's gym handle: there is none, the adapter steps
        self.u = env
        self.obs = None
        self.m, self.d = m, d
        self.names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.lo = m.actuator_ctrlrange[:, 0].copy()
        self.hi = m.actuator_ctrlrange[:, 1].copy()
        key_q = m.key_qpos[0]
        self.default = np.zeros(m.nu)
        for i in range(m.nu):
            j = int(m.actuator_trnid[i][0])
            self.default[i] = np.clip(key_q[int(m.jnt_qposadr[j])], self.lo[i], self.hi[i])
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.t = 0
        self.ret = 0.0
        self.rows = []
        self.done = False

    def step(self, targets, frames=None):  # noqa: ARG002 - hk.Rig's signature
        raise RuntimeError(
            "Shim.step: the BIRD adapter integrates the simulator; a registry rig only "
            "senses. Use scripts/eval_policy.py (or bird.policy_api.run_episode).")


def _raw(rig, targets):
    """`hk.Rig.action` minus the normalisation: targets over the keyframe default, clipped
    to ctrlrange, in RAW actuator units (RigPolicy normalises)."""
    c = rig.default.copy()
    for n, v in targets.items():
        c[rig.idx[n]] = v
    return np.clip(c, rig.lo, rig.hi)


def _as_fn(ctl):
    def fn(rig, t, ctrl):  # noqa: ARG001 - ctrl: the targets are absolute, as in hk.Rig.action
        return _raw(rig, ctl(rig, int(t)))
    fn.controller = ctl
    return fn


def make_hop(p=None, module=None, provenance=None):  # noqa: ARG001 - final_params.json's other keys
    """`hop4.Hop(p)` -- the two-footed kangaroo hop, `final_params.json`'s `p` block."""
    if p is None:
        raise ValueError("make_hop: no `p` -- the manifest's kwargs_from is final_params.json")
    return _as_fn(hop4.Hop(dict(p)))


def make_moonhop(p=None, note=None):  # noqa: ARG001 - moon_t15.json's prose key
    """`moon.Moon(p)` -- turn round hopping, then hop backwards; `moon_t15.json`'s `p`."""
    if p is None:
        raise ValueError("make_moonhop: no `p` -- the manifest's kwargs_from is moon_t15.json")
    return _as_fn(moon.Moon(dict(p)))
