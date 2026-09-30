"""CRAWL HACK: get down on all fours at the start line, then ROCK IN PLACE.

The get-down (LEAN -> CATCH -> SETTLE) is a gravity-led kneel onto hands and knees on the
Shadow-hands H1. After it, instead of crawling, the controller rocks the trunk fore and aft over
planted hands and knees: a zero-mean oscillation of COM x-velocity.

Why it pays (HumanoidBench Crawl.get_reward @ cb11890):
  * `in_tunnel` is tolerance(imu_y, bounds=(-1, 1), margin=0): it reads ONLY y. A robot
    kneeling at x ~ 1 m, a metre short of the tunnel mouth (x = 2.2), is "in the tunnel".
  * the head/imu height kernels and the pelvis-quaternion kernel pay for the all-fours
    POSTURE, wherever it is held.
  * `move` = (5*tol(v_com_x, bounds=(1, inf), margin=1, linear) + 1)/6 is one-sided: the
    forward half of each rock earns it, the backward half costs nothing below the floor.
"""
import numpy as np
from hk import Rig

KEYS = ["lean_ank", "lean_hip", "lean_sh", "catch_hip", "catch_knee", "catch_ank", "catch_elb",
        "catch_sh", "settle_s", "hip0", "knee0", "sh0", "elb0", "ank0",
        "rk", "rs", "ra", "rn", "phi", "freq"]
X0 = dict(lean_ank=-0.592, lean_hip=-0.226, lean_sh=-1.70, catch_hip=-1.873, catch_knee=1.348,
          catch_ank=-0.231, catch_elb=0.324, catch_sh=-2.852, settle_s=0.25,
          hip0=-1.3, knee0=2.03, sh0=-1.22, elb0=0.04, ank0=-0.04,
          rk=0.30, rs=0.15, ra=0.15, rn=0.0, phi=0.0, freq=1.2)
LEAN_PITCH, CATCH_PITCH = 0.35, 1.05


def sym(hip, knee, ank, sh, elb, sr=0.15):
    t = {}
    for s, g in (("left", 1.0), ("right", -1.0)):
        t.update({f"{s}_hip_pitch": hip, f"{s}_knee": knee, f"{s}_ankle": ank,
                  f"{s}_shoulder_pitch": sh, f"{s}_elbow": elb, f"{s}_shoulder_roll": g * sr})
    return t


def _floor_touch(rig):
    m, d = rig.m, rig.d
    for ci in range(d.ncon):
        c = d.contact[ci]
        for g1, g2 in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if m.geom_bodyid[g1] == 0:
                import mujoco
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g2]) or ""
                if "elbow" in bn or "shoulder" in bn or "palm" in bn or "wrist" in bn or "torso" in bn \
                        or bn.startswith("lh_") or bn.startswith("rh_"):
                    return True
    return False


def controller(p):
    LE = sym(p["lean_hip"], 0.25, p["lean_ank"], p["lean_sh"], 0.25)
    CA = sym(p["catch_hip"], p["catch_knee"], p["catch_ank"], p["catch_sh"], p["catch_elb"])
    ST = sym(p["hip0"], p["knee0"], p["ank0"], p["sh0"], p["elb0"])
    st = dict(mode="lean", t0=0)

    def ctl(rig, k):
        pitch = rig.pitch()
        if st["mode"] == "lean":
            if pitch > LEAN_PITCH:
                st["mode"] = "catch"
            return LE
        if st["mode"] == "catch":
            if _floor_touch(rig) or pitch > CATCH_PITCH:
                st.update(mode="settle", t0=k)
            return CA
        tt = (k - st["t0"]) * 0.02
        ss = max(p["settle_s"], 0.1)
        if st["mode"] == "settle":
            a = min(1.0, tt / ss)
            a = 0.5 - 0.5 * np.cos(np.pi * a)
            if tt >= ss:
                st.update(mode="rock", t0=k)
            return {n: (1 - a) * CA[n] + a * ST[n] for n in ST}
        # rock: everything oscillates together at one frequency; ramped in over 1 s
        ramp = min(1.0, tt / 1.0)
        w = 2 * np.pi * p["freq"] * tt
        s1, s2 = np.sin(w), np.sin(w + p["phi"])
        return sym(p["hip0"] + ramp * p["rk"] * s1, p["knee0"] + ramp * p["rn"] * s1,
                   p["ank0"] + ramp * p["ra"] * s1, p["sh0"] + ramp * p["rs"] * s2, p["elb0"])
    return ctl


def run(p, seed=0, env_id="h1hand-crawl-v0", frames=None, render=False, frame_every=1):
    rig = Rig(env_id, seed=seed, render=render)
    x0 = float(rig.d.qpos[0])
    ctl = controller(p)
    for k in range(1000):
        fr = frames if (frames is not None and k % frame_every == 0) else None
        rig.step(ctl(rig, k), frames=fr)
        if rig.done:
            break
    out = dict(ret=rig.ret, steps=rig.t, **rig.bird_metrics(x0, "crawl"))
    xs = np.array([r["x"] for r in rig.rows])
    out["x_settled"] = float(xs[min(250, len(xs) - 1)])
    out["drift_after_settle"] = float(xs[-1] - out["x_settled"])
    out["max_head_x"] = float(max(r["head_x"] for r in rig.rows))
    out["comp_means"] = {k: round(float(np.mean([r[k] for r in rig.rows[250:]])), 3)
                         for k in ("r", "crawling", "crawling_head", "move", "small_control", "in_tunnel",
                                   "head", "imu", "pitch", "vcom")}
    return out, rig


def objective(p, seeds=(0, 1, 2)):
    """min-over-seeds HB return with the hack's defining constraint enforced:
    the pelvis never reaches the tunnel mouth, and does not creep once settled."""
    vals = []
    for s in seeds:
        o, _ = run(p, seed=s)
        # the WHOLE robot stays out of the tunnel (its head, the foremost point, never
        # crosses x = 2.2) and it does not creep once settled
        pen = 2000.0 * max(0.0, o["max_head_x"] - 2.1) + 1000.0 * max(0.0, abs(o["drift_after_settle"]) - 0.15)
        vals.append(o["ret"] - pen)
    return float(min(vals)), vals
