"""BACKWARD HOP ("moonhop"): turn round in place (hopping), then hop BACKWARD the rest of the
episode -- facing away from the direction of travel, never looking where it goes.

HB Walk's move term reads the WORLD x COM velocity only; it never asks which way the robot
faces.  Plant controller: the Raibert hopper of snap_hop3.py (an earlier version of hop4.py)
with the re-tuned knobs of gallop_best_d0.json (feet together).  This file only (a) biases the
hopper's yaw servo toward a heading reference that slews 0 -> pi, hopping in place, and then
(b) commands v_des = -V_b in the (now reversed) heading frame, i.e. +x in the world."""
import json, os
import numpy as np
import snap_hop3 as H

HERE = os.path.dirname(os.path.abspath(__file__))
P_BASE = dict(json.load(open(os.path.join(HERE, "snap_hop3_cem_a.json")))["p"])
P_BASE.update({k: v for k, v in json.load(open(os.path.join(HERE, "gallop_best_d0.json")))["p"].items()
               if k in P_BASE})
KEYS = ["t_turn0", "yaw_rate", "V_b", "phi_b", "k_yaw", "kv", "k_ns", "vz_des", "kp_phi", "ky", "slew"]
X0 = dict(t_turn0=1.0, yaw_rate=0.8, V_b=1.5, phi_b=0.0, k_yaw=P_BASE["k_yaw"], kv=P_BASE["kv"],
          k_ns=P_BASE["k_ns"], vz_des=P_BASE["vz_des"], kp_phi=P_BASE["kp_phi"], ky=P_BASE["ky"], slew=1.0)
SIGF = dict(phi_b=0.05, ky=0.05, t_turn0=0.3)
BOUNDS = dict(t_turn0=(0.5, 4.0), yaw_rate=(0.1, 3.0), V_b=(0.3, 2.5), phi_b=(-0.3, 0.3),
              k_yaw=(0.1, 2.0), kv=(0.0, 0.4), k_ns=(0.5, 2.5), vz_des=(0.4, 1.6), kp_phi=(100, 2500),
              slew=(0.1, 5.0))


def clip(p):
    q = dict(p)
    for k, (a, b) in BOUNDS.items():
        q[k] = float(np.clip(q[k], a, b))
    return [q[k] for k in KEYS]


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


class Moon(H.Hop):
    def __init__(self, p):
        base = dict(P_BASE)
        for k in ("k_yaw", "kv", "k_ns", "vz_des", "kp_phi", "ky"):
            base[k] = p[k]
        base["v_des"] = 0.0
        base["spin_up"] = 0.1
        super().__init__(base)
        self.m = p
        self.yref = 0.0
        self.vcmd = 0.0
        self.k = 0
        self.turned = None

    def sense(self, rig):
        s = super().sense(rig)
        s.yaw_true = s.yaw
        s.yaw = wrap(s.yaw - self.yref)      # the hopper's yaw servo now holds yref
        return s

    def __call__(self, rig, k):
        t = k * 0.02
        m = self.m
        if t > m["t_turn0"]:
            self.yref = min(np.pi, self.yref + m["yaw_rate"] * 0.02)
        R = rig.d.xmat[rig.pelvis].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        facing_back = self.yref >= np.pi - 1e-6 and abs(wrap(yaw - np.pi)) < 0.35
        if facing_back and self.turned is None:
            self.turned = k
        tgt = -m["V_b"] if self.turned is not None else 0.0
        self.vcmd += float(np.clip(tgt - self.vcmd, -m["slew"] * 0.02, m["slew"] * 0.02))
        self.p["v_des"] = self.vcmd
        self.p["phi_des"] = m["phi_b"] if self.turned is not None else P_BASE["phi_des"] * 0.3
        return super().__call__(rig, k)


def run(p, seed=0, frames=None, frame_every=1, steps=1000):
    rig = H.Rig("h1hand-walk-v0", seed=seed, render=frames is not None)
    x0 = float(rig.d.qpos[0])
    h = Moon(p)
    yaws = []
    for k in range(steps):
        fr = frames if (frames is not None and k % frame_every == 0) else None
        rig.step(h(rig, k), frames=fr)
        R = rig.d.xmat[rig.pelvis].reshape(3, 3)
        yaws.append(float(np.arctan2(R[1, 0], R[0, 0])))
        if rig.done:
            break
    R_ = rig.rows
    xs = np.array([r["x"] for r in R_])
    yw = np.array(yaws)
    k0 = h.turned if h.turned is not None else len(yw)
    back = np.abs(np.arctan2(np.sin(yw[k0:] - np.pi), np.cos(yw[k0:] - np.pi))) if k0 < len(yw) else np.array([])
    o = dict(ret=round(rig.ret, 2), steps=rig.t, dx=round(float(xs[-1] - x0), 3), k_turned=h.turned,
             n_hops=h.n_hops, speed_after_turn=round(float((xs[-1] - xs[min(k0, len(xs) - 1)]) / max((len(xs) - k0) * 0.02, 0.02)), 3),
             facing_back_frac_after_turn=round(float(np.mean(back < 0.5)), 3) if len(back) else 0.0,
             y_end=round(float(R_[-1]["y"]), 2))
    for kk in ("move", "standing", "upright", "small_control", "head"):
        o["m_" + kk] = round(float(np.mean([r[kk] for r in R_])), 4)
    return o, rig


def objective(p, seeds=(0, 1)):
    vals = []
    for s in seeds:
        o, _ = run(p, s)
        # must actually be facing backward: penalise any time after the turn not within 0.5 rad of pi
        pen = 2000.0 * max(0.0, 0.95 - o["facing_back_frac_after_turn"]) + (500.0 if o["k_turned"] is None else 0.0)
        vals.append(o["ret"] - pen)
    return float(min(vals)), [round(v, 1) for v in vals]


if __name__ == "__main__":
    import sys
    p = dict(X0)
    if len(sys.argv) > 1:
        p.update(json.loads(sys.argv[1]))
    print(json.dumps(run(p, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)[0]))
