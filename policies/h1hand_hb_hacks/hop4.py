"""WALK HACK: two-footed kangaroo hop (both legs one virtual leg), torque-mode stance.
hop4 = hop3 + an event-gated speed ramp (ramp_mode=1): the speed command rises per healthy hop.

Plant facts used (measured on the model, humanoid_bench @ cb11890):
  thigh = shank = 0.40 m exactly; hip/knee kp 200/300 (kd 5/6), ankle kp 40 (kd 2);
  pitch axes all +y, q_hip > 0 swings the thigh BACK; foot world pitch = phi+q1+q2+q3.
Position servos are used as torque sources in stance: target = q_meas + tau/kp.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import mujoco
from hk import Rig

L = 0.4
W = 53.24 * 9.81
SIDES = ("left", "right")

X0 = dict(
    v_des=0.0, spin_up=3.0,
    r_comp=0.72, r_ext=0.77, r_td=0.74, r_air=0.70,
    ts=0.16, kv=0.10, kp_phi=800.0, kd_phi=0.0, phi_des=0.0, k_ank=0.5, ank_push=0.0,
    ky=0.10, k_yaw=0.5, arm_sp=0.0, t_init=0.5, r_init=0.66, f_td=0.25, thrust_t=0.08,
    tau_max=150.0, kL=50.0, kd_comp=1.0, k_ff=0.0, lead=0.01, ts_adapt=1.0, foot_off=0.0, kp_roll=150.0, kd_roll=10.0, k_h=0.0, vz_des=0.85, boost=2.0, boost_d=0.0, k_ns=1.0, kboost=1.0, k_arm=0.0, arm_ret=0.2, L_des=0.0, ramp_mode=0.0, n_warm=2, vz_gate=0.6, dv_hop=0.1, dv_down=0.1, fo_mode=0.0, fo_v=1.0, r_ext0=0.0,
)
KEYS = list(X0)
# CEM search space: key -> (sigma, lo, hi)
SPACE = dict(
    v_des=(0.2, 0.3, 2.0), spin_up=(1.0, 0.5, 10.0), phi_des=(0.04, -0.1, 0.35), kv=(0.03, 0.0, 0.4),
    k_ns=(0.15, 0.5, 2.5), kp_phi=(150, 100, 2500), kL=(15, 0, 200), r_comp=(0.015, 0.6, 0.78),
    r_ext=(0.01, 0.72, 0.797), r_td=(0.01, 0.68, 0.79), r_air=(0.02, 0.55, 0.78), thrust_t=(0.02, 0.02, 0.2),
    k_h=(0.01, 0.0, 0.1), vz_des=(0.1, 0.4, 1.6), boost=(0.3, 1.0, 3.5), k_ank=(0.15, -0.5, 2.0),
    ank_push=(0.05, -0.2, 0.4), kp_roll=(40, 0, 500), kd_roll=(4, 0, 50), ky=(0.04, -0.1, 0.5),
    k_arm=(0.5, -3, 4), foot_off=(0.02, -0.1, 0.1), t_init=(0.1, 0.2, 1.2), r_init=(0.02, 0.58, 0.74),
    f_td=(0.05, 0.05, 0.6), arm_sp=(0.2, -1.5, 1.5),
    n_warm=(1.0, 0, 8), vz_gate=(0.1, 0.2, 1.2), dv_hop=(0.03, 0.01, 0.4), dv_down=(0.03, 0.0, 0.4),
)


def knee_from_r(r):
    r = float(np.clip(r, 0.3, 2 * L - 1e-3))
    return float(np.arccos(np.clip((r * r - 2 * L * L) / (2 * L * L), -1, 1)))


def ik(xf, zf, phi):
    """ankle target (xf fwd, zf up) rel. hip pitch joint in the heading frame -> q1,q2,q3, sole flat."""
    r = float(np.clip(np.hypot(xf, zf), 0.3, 2 * L - 1e-3))
    q2 = knee_from_r(r)
    # thigh world angle th (0 = straight down, +ve = backward, same sense as q1)
    th_leg = np.arctan2(-xf, -zf)  # direction of the virtual leg, +ve = foot behind
    th1 = th_leg - q2 / 2.0  # knee bends backward: thigh is q2/2 forward of the leg line
    q1 = th1 - phi
    q3 = -phi - q1 - q2
    return float(q1), q2, float(q3)


class Sense:
    pass


class Hop:
    def __init__(self, p):
        self.p = dict(X0); self.p.update(p)
        self.mode = "init"
        self.k_mode = 0
        self.tel = []
        self.n_hops = 0
        self.ids = None
        self.ts_meas = self.p["ts"]
        self.xf_tgt = 0.0
        self.dr = 0.0
        self.arm = self.p["arm_sp"]
        self.q1_lo = -0.4
        self.vcmd = 0.0
        self.k_td = 0

    def _ids(self, rig):
        m = rig.m
        self.ank = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_link") for s in SIDES]
        self.hipb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hip_pitch_link") for s in SIDES]
        self.ids = True

    def sense(self, rig):
        if self.ids is None:
            self._ids(rig)
        d, m = rig.d, rig.m
        s = Sense()
        R = d.xmat[rig.pelvis].reshape(3, 3)
        s.yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        c, sn = np.cos(s.yaw), np.sin(s.yaw)
        Rh = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]])
        Rb = Rh.T @ R
        s.phi = float(np.arctan2(Rb[0, 2], Rb[2, 2]))
        s.roll = float(np.arctan2(-Rb[1, 2], Rb[2, 2]))
        w = Rh.T @ (R @ d.qvel[3:6])
        s.phid, s.rolld, s.yawd = float(w[1]), float(w[0]), float(w[2])
        s.com = rig.com(); s.v = rig.comvel()
        vh = Rh.T @ s.v
        s.vx, s.vy, s.vz = float(vh[0]), float(vh[1]), float(vh[2])
        F = np.zeros(2); f6 = np.zeros(6)
        for ci in range(d.ncon):
            cc = d.contact[ci]
            b1, b2 = m.geom_bodyid[cc.geom1], m.geom_bodyid[cc.geom2]
            for i, a in enumerate(self.ank):
                if (b1 == 0 and b2 == a) or (b2 == 0 and b1 == a):
                    mujoco.mj_contactForce(m, d, ci, f6)
                    F[i] += abs(f6[0])
        s.F = F; s.Ft = float(F.sum())
        # centre of pressure over both feet (normal-force weighted contact positions)
        cop = np.zeros(3); fn = 0.0
        for ci in range(d.ncon):
            cc = d.contact[ci]
            b1, b2 = m.geom_bodyid[cc.geom1], m.geom_bodyid[cc.geom2]
            if (b1 == 0 and b2 in self.ank) or (b2 == 0 and b1 in self.ank):
                mujoco.mj_contactForce(m, d, ci, f6)
                cop += abs(f6[0]) * cc.pos; fn += abs(f6[0])
        s.cop = cop / fn if fn > 1e-6 else None
        mujoco.mj_subtreeVel(m, d)
        s.L = Rh.T @ d.subtree_angmom[rig.pelvis]
        hips = [d.xpos[b] for b in self.hipb]
        anks = [d.xpos[b] for b in self.ank]
        s.hip = 0.5 * (hips[0] + hips[1])
        s.rel = [Rh.T @ (a - h) for a, h in zip(anks, hips)]
        s.relm = 0.5 * (s.rel[0] + s.rel[1])
        s.r = float(np.hypot(s.relm[0], s.relm[2]))
        s.com_hip = Rh.T @ (s.com - s.hip)
        s.Rh = Rh
        s.q = {f"{sd}_{j}": rig.q(f"{sd}_{j}") for sd in SIDES for j in ("hip_pitch", "knee", "ankle", "hip_roll", "hip_yaw")}
        return s

    def __call__(self, rig, k):
        p = self.p
        s = self.sense(rig)
        tt = (k - self.k_mode) * 0.02
        if p["ramp_mode"] > 0.5:
            v_des = self.vcmd  # event-gated: raised per healthy hop, lowered per weak one
        else:
            v_des = p["v_des"] * min(1.0, max(0.0, k * 0.02 - 1.0) / max(p["spin_up"], 0.1))
        tgt = {}
        tau = 0.0
        y_f = s.vy * p["ts"] / 2 + p["ky"] * s.vy
        roll_cmd = float(np.clip(y_f / 0.75 + s.roll, -0.35, 0.35))
        # ---- mode transitions (event-gated) ----
        if self.mode == "init":
            if tt >= p["t_init"]:
                self.mode, self.k_mode = "thrust", k
        elif self.mode == "flight":
            if s.Ft > p["f_td"] * W and tt > 0.06:
                self.mode, self.k_mode = "comp", k
                self.k_td = k
        elif self.mode == "comp":
            if (s.vz > 0 and tt > 0.04) or tt > 0.2:
                self.mode, self.k_mode = "thrust", k
        elif self.mode == "thrust":
            if s.Ft < 0.05 * W and tt > 0.04:
                self.ts_meas = 0.7 * self.ts_meas + 0.3 * (k - self.k_td) * 0.02
                self.dr = float(np.clip(self.dr + p["k_h"] * (p["vz_des"] - s.vz), -0.08, 0.05))
                self.mode, self.k_mode = "flight", k
                self.n_hops += 1
                self.q1_lo = s.q["left_hip_pitch"]
                if self.n_hops >= p["n_warm"]:
                    if s.vz >= p["vz_gate"] and abs(s.phi - p["phi_des"]) < 0.25:
                        self.vcmd = min(p["v_des"], self.vcmd + p["dv_hop"])
                    else:
                        self.vcmd = max(0.0, self.vcmd - p["dv_down"])
        tt = (k - self.k_mode) * 0.02
        stance = self.mode != "flight"
        if self.mode == "init":
            r_cmd = 0.736 + (p["r_init"] - 0.736) * min(1.0, tt / max(p["t_init"] * 0.7, 0.05))
        elif self.mode == "comp":
            r_cmd = p["r_comp"]
        elif self.mode == "thrust":
            r_e = min(p["r_ext"] + self.dr, 0.797)
            if self.n_hops == 0 and p["r_ext0"] > 0:
                r_e = p["r_ext0"]  # the first push, out of the init crouch
            r_cmd = p["r_comp"] + (r_e - p["r_comp"]) * min(1.0, (tt + 0.02) / max(p["thrust_t"], 0.02))
        else:
            r_cmd = p["r_ext"]
        Mdes = Mff = 0.0
        if stance and s.cop is not None:
            # hip torque that makes the ground force produce the desired moment about the COM
            # (massless-leg model: hip torque sets the GRF component perpendicular to the leg).
            e = s.phi - p["phi_des"]
            Mdes = -p["kp_phi"] * e - p["kd_phi"] * s.phid - p["kL"] * (s.L[1] - p["L_des"])
            c = s.Rh.T @ s.cop; h = s.Rh.T @ s.hip; com = s.Rh.T @ s.com
            av = np.array([h[0] - c[0], h[2] - c[2]]); r = float(np.linalg.norm(av)); av = av / r
            D = np.array([c[0] - com[0], c[2] - com[2]])
            fa = max(s.Ft, 0.0)
            Ma = D[1] * av[0] - D[0] * av[1]
            Da = float(D @ av)
            Mff = fa * Ma * p["k_ff"]
            fp = (Mdes - Mff) / Da
            tau = float(np.clip(r * fp, -2 * p["tau_max"], 2 * p["tau_max"]))
        if stance:
            q2 = knee_from_r(r_cmd)
            push = p["ank_push"] if self.mode == "thrust" else 0.0
            e = s.phi - p["phi_des"]
            for sd in SIDES:
                q1m, q2m, q3m = s.q[f"{sd}_hip_pitch"], s.q[f"{sd}_knee"], s.q[f"{sd}_ankle"]
                q1d = float(rig.d.joint(f"{sd}_hip_pitch").qvel[0])
                # torque via the position servo, with the joint's motion over the 20 ms hold predicted
                tgt[f"{sd}_hip_pitch"] = q1m + q1d * p["lead"] + (tau / 2 + 5.0 * q1d * p["kd_comp"]) / 200.0
                tgt[f"{sd}_knee"] = q2m + p["kboost"] * (q2 - q2m)
                # sole flat to the ground (measured foot pitch), plus a pitch-error CoP shift
                foot = s.phi + q1m + q2m + q3m
                tgt[f"{sd}_ankle"] = q3m - foot + p["k_ank"] * e + push
                qr = s.q[f"{sd}_hip_roll"]
                qrd = float(rig.d.joint(f"{sd}_hip_roll").qvel[0])
                tau_r = p["kp_roll"] * s.roll + p["kd_roll"] * s.rolld
                tgt[f"{sd}_hip_roll"] = qr + qrd * p["lead"] + (tau_r + 5.0 * qrd) / 200.0
                tgt[f"{sd}_hip_yaw"] = 0.0
        else:
            ts = self.ts_meas if p["ts_adapt"] > 0 else p["ts"]
            fo = p["foot_off"] * (float(np.clip(s.vx / p["fo_v"], 0.0, 1.0)) if p["fo_mode"] > 0.5 else 1.0)
            xf = s.com_hip[0] + p["k_ns"] * s.vx * ts / 2 + p["kv"] * (s.vx - v_des) - fo
            zf = -(p["r_td"] if s.vz < 0 else p["r_air"])
            self.xf_tgt = xf - s.com_hip[0]
            q1, q2, q3 = ik(xf, zf, s.phi)
            for sd in SIDES:
                q1m = s.q[f"{sd}_hip_pitch"]; q1d = float(rig.d.joint(f"{sd}_hip_pitch").qvel[0])
                tgt[f"{sd}_hip_pitch"] = q1m + p["boost"] * (q1 - q1m) - p["boost_d"] * q1d / 200.0
                tgt[f"{sd}_knee"] = q2; tgt[f"{sd}_ankle"] = q3
                tgt[f"{sd}_hip_roll"] = roll_cmd
                tgt[f"{sd}_hip_yaw"] = float(np.clip(-p["k_yaw"] * s.yaw, -0.4, 0.4))
        # arms counter-swing the legs (reaction cancels part of the leg-swing pitch torque)
        if not stance:
            self.arm = float(np.clip(p["arm_sp"] - p["k_arm"] * (s.q["left_hip_pitch"] - self.q1_lo), -2.8, 2.8))
        else:
            self.arm = self.arm + (p["arm_sp"] - self.arm) * p["arm_ret"]
        for sd in SIDES:
            tgt[f"{sd}_shoulder_pitch"] = self.arm
        dxLR = float((s.rel[0] - s.rel[1])[0])
        self.last = dict(dxLR=dxLR, xf_tgt=(self.xf_tgt if not stance else 0.0), vdes=v_des, mode=self.mode, phi=s.phi, phid=s.phid, Ly=float(s.L[1]), yaw=s.yaw, roll=s.roll,
                         vx=s.vx, vy=s.vy, vz=s.vz, comz=float(s.com[2]), F=s.Ft, FL=float(s.F[0]),
                         FR=float(s.F[1]), r=s.r, xrel=float(s.relm[0] - s.com_hip[0]), tau=tau, Mdes=Mdes,
                         Mff=Mff, copx=(float((s.Rh.T @ (s.cop - s.com))[0]) if s.cop is not None else 0.0),
                         hipF=float(rig.d.actuator_force[rig.idx["left_hip_pitch"]] + rig.d.actuator_force[rig.idx["right_hip_pitch"]]),
                         q1=s.q["left_hip_pitch"], q2=s.q["left_knee"], q3=s.q["left_ankle"])
        return tgt


def run(p, seed=0, env_id="h1hand-walk-v0", frames=None, render=False, frame_every=1, steps=1000):
    rig = Rig(env_id, seed=seed, render=render)
    x0 = float(rig.d.qpos[0])
    h = Hop(p)
    for k in range(steps):
        fr = frames if (frames is not None and k % frame_every == 0) else None
        tgt = h(rig, k)
        row = rig.step(tgt, frames=fr)
        row.update(h.last)
        if rig.done:
            break
    out = dict(ret=rig.ret, steps=rig.t, n_hops=h.n_hops, **rig.bird_metrics(x0, "walk"))
    R = rig.rows
    out["mean_speed"] = float((R[-1]["x"] - x0) / (rig.t * 0.02))
    out["comp_means"] = {k: round(float(np.mean([r[k] for r in R])), 3)
                         for k in ("r", "standing", "upright", "move", "small_control", "head", "vcom")}
    return out, rig


def objective(p, seeds=(0, 1, 2)):
    vals = [run(p, seed=s)[0]["ret"] for s in seeds]
    return float(min(vals)), vals
