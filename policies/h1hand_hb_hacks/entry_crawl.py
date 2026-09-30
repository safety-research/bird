#!/usr/bin/env python3
"""Registry wiring for the crawl hack of `h1hand_hb_hacks`.

`h1hand_hb_hacks/crawl_rock` (`rock.controller`) was written against `hk.Rig` over its own
`gym.make("h1hand-crawl-v0")`, exactly as the two walk hacks were, so this module adapts it
the same way `entry.py` adapts `hop4.Hop` / `moon.Moon`, and is kept apart from `entry.py` so
the walk entries' wiring and records stay untouched:

- The rig is `entry.py::Shim` (the manifest's `rig:` block names it): `hk.Rig` seated on the
  adapter's own simulator, never resetting or stepping. It is task-agnostic -- it reads the
  actuator table and keyframe off whichever HumanoidBench model it is handed -- and every
  sensing call the controller makes (`rig.pitch()`, `rig.d.qpos`, `rig.d.site("head")`,
  and `rock._floor_touch(rig)`'s `rig.m` / `rig.d` contact scan) is `hk.Rig`'s own attribute.
- The factory calls `rock.controller(p)` and wraps the returned `ctl(rig, k)` -- called once
  per control step with `k` from 0, as the module's `run()` calls it -- into the
  registry's `fn(rig, t, ctrl) -> ctrl`. `_raw` is `entry.py::_raw` restated (targets over
  `rig.default`, clipped to ctrlrange: `hk.Rig.action` without the normalisation, which
  `bird.policy_api.RigPolicy.act` applies with the same expression). The incoming `ctrl` is
  ignored, as `hk.Rig.action` ignores any previous command. `RigPolicy.reset` rebuilds the
  factory per episode, so the controller's closure state starts fresh on every seed.
- `rock.py` imports `Rig` from `hk.py`; `hk` is bound to THIS directory's copy by the
  `import_beside` call below (`import_beside` checks every module resolved to the file
  beside this one).

The constants file (`robust3_rock_final.json`) is a CEM output whose top-level keys are the
search's bookkeeping (`f`, `vals`, `seeds`, `obj`, `gen`) beside the constants `p`; the
factory takes all of them because `kwargs_from` passes every key, and uses only `p`.
`records/eval_crawl_rock_*` is what `scripts/eval_policy.py` measured through this module;
`records/adapter_positions_crawl_rock_*` is the position pass through the same adapter, from
`poscheck_crawl.py` (every body frame and site).
"""
from pathlib import Path

import numpy as np

from bird.policy_api import import_beside

HERE = Path(__file__).resolve().parent

# hk first, so rock's `from hk import Rig` finds THIS copy.
hk, rock = import_beside(HERE, "hk", "rock")


def _raw(rig, targets):
    """`hk.Rig.action` minus the normalisation (as entry.py::_raw): targets over the keyframe
    default, clipped to ctrlrange, in RAW actuator units (RigPolicy normalises)."""
    c = rig.default.copy()
    for n, v in targets.items():
        c[rig.idx[n]] = v
    return np.clip(c, rig.lo, rig.hi)


def _wrap(ctl):
    def fn(rig, t, ctrl):  # noqa: ARG001 - ctrl: the targets are absolute, as in hk.Rig.action
        return _raw(rig, ctl(rig, int(t)))
    fn.controller = ctl
    return fn


def make_rock(p=None, f=None, vals=None, seeds=None, obj=None, gen=None):  # noqa: ARG001 - CEM bookkeeping keys
    """`rock.controller(p)` -- the kneel-and-rock crawl: on all fours short of the
    tunnel, rocking the trunk fore and aft over planted hands and knees, with no gait at all;
    `robust3_rock_final.json`'s `p` block."""
    if p is None:
        raise ValueError("make_rock: no `p` -- the manifest's kwargs_from is robust3_rock_final.json")
    return _wrap(rock.controller(dict(p)))
