"""An adapter's advertised action box should be the one the simulator applies.

`fasttd3.clip_env_action` (`:1021-1026`) is documented as producing "the action
the dynamics EXECUTE … the candidate reward is evaluated on this, never on the
sampled value". It clips to the ADAPTER's bounds. `_step` then writes the result
into `d.ctrl` and **MuJoCo clips again** to each actuator's own `ctrlrange`.
Where the two differ, a candidate reward keyed on the action reads a command the
simulator discarded, and the docstring says otherwise -- the
`evaluate.artifacts`/`images=` family again: a value that IS read, read as
though it were something it is not, with nothing that can fail.

Measured on Assistax: the adapter advertises `[-1,1]^7` and no actuator's
range is `[-1,1]`; `a[3]` is `[-0.9778,-0.0222]`, entirely negative, so every
positive command on that joint is discarded.

XFAIL, DELIBERATELY. The Assistax action box is kept as the results reported
with this code were produced under; narrowing it would make those results
incomparable with any new run. So this records the census and does not gate.

**AN XPASS HERE IS A SIGNAL, NOT NOISE.** It means every adapter this census
can read now advertises the box its simulator applies, and the right response
is to delete the `xfail` marker and let the test gate, not to re-mark it or
ignore it. `strict=False` is what allows that to be visible: strict would turn
the good news into a failure, and no marker at all would turn a known gap into
a permanently red build.

IT REPORTS WHAT IT COULD NOT INSPECT. An adapter this cannot read is listed as
`unknown`, never as clean: the census's value is the list, and a list that
quietly omits what it could not open is the input-list defect this repo keeps
finding. Only the adapters it actually opened are claimed either way.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

pytestmark = pytest.mark.slow

#: One task per MuJoCo-backed family, not every id: the families share an
#: adapter class, so a per-family probe answers the same question in seconds
#: rather than minutes.
FAMILY_SAMPLE = [
    "assistax_feeding", "assistax_scratchitch", "gym_half_cheetah",
    "mt10_push-v3", "h1hand_walk",
]


def _ctrl_ranges(env: Any) -> Optional[np.ndarray]:
    """The `ctrlrange` of the actuators this adapter's action drives, or None.

    None means "could not read", never "no limits": the caller reports it as
    unknown. Adapters differ in how they name the model and the actuator
    subset, so this duck-types rather than assuming one layout, and gives up
    loudly instead of guessing a mapping that might be wrong.
    """
    model = getattr(env, "_model", None) or getattr(env, "model", None)
    if model is None or not hasattr(model, "actuator_ctrlrange"):
        return None
    cr = np.asarray(model.actuator_ctrlrange, dtype=float)
    idx = getattr(env, "_robot_act", None)
    if idx is not None:
        cr = cr[np.asarray(idx)]
    elif cr.shape[0] != int(getattr(env, "action_dim", -1)):
        return None                      # cannot map action -> actuator; unknown
    return cr


def _advertised(env: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    lo, hi = getattr(env, "action_low", None), getattr(env, "action_high", None)
    if lo is None or hi is None:
        return None
    return np.asarray(lo, float), np.asarray(hi, float)


def census() -> Dict[str, Dict[str, Any]]:
    from bird import registry
    registry.load_all()
    out: Dict[str, Dict[str, Any]] = {}
    for env_id in FAMILY_SAMPLE:
        try:
            env = registry.get("env", env_id)(None)
        except Exception as exc:         # noqa: BLE001 -- a missing extra is unknown
            out[env_id] = {"status": "unknown", "why": f"{type(exc).__name__}: {exc}"[:120]}
            continue
        cr, adv = _ctrl_ranges(env), _advertised(env)
        if cr is None or adv is None:
            out[env_id] = {"status": "unknown",
                           "why": "no readable ctrlrange" if cr is None else "no advertised bounds"}
            continue
        lo, hi = adv
        wider = [i for i in range(cr.shape[0])
                 if lo[i] < cr[i, 0] - 1e-9 or hi[i] > cr[i, 1] + 1e-9]
        out[env_id] = {
            "status": "wider" if wider else "matches",
            "n_actuators": int(cr.shape[0]),
            "n_wider": len(wider),
            "example": ([float(lo[wider[0]]), float(hi[wider[0]]),
                         float(cr[wider[0], 0]), float(cr[wider[0], 1])] if wider else None),
        }
    return out


@pytest.mark.xfail(reason="the executed action is not the advertised one on at least "
                          "Assistax; the action box is held fixed for comparability",
                   strict=False)
def test_every_mujoco_adapter_advertises_the_box_the_simulator_applies() -> None:
    rows = census()
    wider = {k: v for k, v in rows.items() if v["status"] == "wider"}
    unknown = {k: v["why"] for k, v in rows.items() if v["status"] == "unknown"}
    assert not wider, (
        "these adapters advertise an action box wider than their actuators' "
        f"ctrlrange, so `clip_env_action` is not the executed action: {wider}\n"
        f"(not inspected, reported rather than counted clean: {unknown})\n"
        "The Assistax box is held fixed for comparability with reported results.")
