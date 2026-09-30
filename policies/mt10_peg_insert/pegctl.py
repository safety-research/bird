#!/usr/bin/env python3
"""A scripted policy for MT10 peg-insert-side-v3, evaluated THROUGH the
BIRD adapter (mt10_peg-insert-side-v3). NOT Meta-World's bundled scripted
policy — written from the spec + obs layout; the bundled expert exists only
as the spec's anchor (any-step 0.8333, per-step 0.5347), which is the bar.

Control: Meta-World actions are [dx, dy, dz, grip], mocap velocity at
0.01 m per unit-step. Waypoint P-control: a[:3] = clip(k * (wp - hand), -1, 1)
with k in 1/cm. After the grasp the controller drives the PEG HEAD, not the
hand: head = obs[4:7] + R(obs[7:11]) @ (-0.13, 0, -0.01) — the exact
transcription the adapter's success check uses — so residual peg tilt in the
grip is closed-loop-corrected instead of assumed away. The success region is
a slot (lateral error x2), so insertion approaches along +x with lateral
servoing before any axial advance.

Phases: approach(above peg) -> descend -> close -> lift (with grasp check &
regrasp) -> align (head to pre-hole point) -> insert (head to goal) -> hold
(grip closed, zero motion — the per-step metric pays holding, and holding is
the honest reading of "keep it inserted"; the shipped check's fling
unsoundness is documented in the spec and deliberately not exploited).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import os
os.environ.setdefault("MUJOCO_GL", "egl")

HEAD_LOCAL = np.array([-0.13, 0.0, -0.01])

# THE QUATERNION TRAP: Meta-World obs quats are scipy scalar-LAST (x,y,z,w),
# not MuJoCo scalar-first. A w-first rotate puts the reconstructed peg head
# ~0.2 m off while the adapter scores the true head INSIDE the hole — the
# policy half-works while its own gates misfire.
# Import the adapter's verified transcription so policy and metric share one
# geometry; never reimplement it.
from bird.envs.metaworld import _rotate  # noqa: E402


def head_of(s):
    return s[4:7] + _rotate(s[7:11], HEAD_LOCAL)


def slot_dist(s):
    return float(np.linalg.norm((head_of(s) - s[36:39])
                                * np.array([1.0, 2.0, 2.0])))


DEFAULTS = dict(k=6.0, z_hover=0.10, z_grasp=0.005, xy_tol=0.010,
                d_tol=0.012, n_close=15, z_carry=0.19, z_lift_gate=0.17,
                grasp_z=0.07, pre_gap=0.10, align_tol=0.012, ins_gain=2.0,
                ins_depth=0.005, hold_gd=0.045, drop_z=0.05)


def make_policy(p):
    st = dict(phase="approach", n=0)

    def act(s):
        hand = s[:3]
        peg = s[4:7]
        goal = s[36:39]
        head = head_of(s)
        a = np.zeros(4)
        st["n"] += 1

        def goto(wp, gain=None, grip=-1.0):
            a[:3] = np.clip((gain or p["k"]) * (wp - hand), -1, 1)
            a[3] = grip

        # dropped-peg detector: anywhere past close, peg back at table level
        # while the hand is high means the grasp failed -> start over
        if st["phase"] in ("lift", "align", "insert") and \
                peg[2] < p["drop_z"] and hand[2] > 0.12:
            st["phase"] = "approach"

        if st["phase"] == "approach":
            wp = np.array([peg[0], peg[1], peg[2] + p["z_hover"]])
            goto(wp, grip=-1.0)
            if np.linalg.norm((hand - wp)[:2]) < p["xy_tol"] and \
                    abs(hand[2] - wp[2]) < 0.02:
                st["phase"] = "descend"
        elif st["phase"] == "descend":
            wp = np.array([peg[0], peg[1], peg[2] + p["z_grasp"]])
            goto(wp, grip=-1.0)
            if np.linalg.norm(hand - wp) < p["d_tol"]:
                st["phase"] = "close"
                st["n_close"] = 0
        elif st["phase"] == "close":
            a[:] = 0.0
            a[3] = 1.0
            st["n_close"] = st.get("n_close", 0) + 1
            if st["n_close"] >= p["n_close"]:
                st["phase"] = "lift"
        elif st["phase"] == "lift":
            wp = np.array([hand[0], hand[1], p["z_carry"]])
            goto(wp, grip=1.0)
            if hand[2] > p["z_lift_gate"]:
                if peg[2] > p["grasp_z"]:
                    st["phase"] = "align"
                else:
                    st["phase"] = "approach"   # lifted an empty gripper
        elif st["phase"] == "align":
            # drive the HEAD to a point pre_gap in front of the hole (+x),
            # exactly on the hole's y/z line
            wp_head = goal + np.array([p["pre_gap"], 0.0, 0.0])
            a[:3] = np.clip(p["k"] * (wp_head - head), -1, 1)
            a[3] = 1.0
            if np.linalg.norm(head - wp_head) < p["align_tol"]:
                st["phase"] = "insert"
        elif st["phase"] == "insert":
            wp_head = goal + np.array([-p["ins_depth"], 0.0, 0.0])
            # slow axial advance, full-rate lateral correction
            e = wp_head - head
            a[0] = np.clip(p["ins_gain"] * e[0], -1, 1)
            a[1] = np.clip(p["k"] * e[1], -1, 1)
            a[2] = np.clip(p["k"] * e[2], -1, 1)
            a[3] = 1.0
            if slot_dist(s) < p["hold_gd"]:
                st["phase"] = "hold"
        elif st["phase"] == "hold":
            a[:] = 0.0
            a[3] = 1.0
        return a, dict(phase=st["phase"], sd=slot_dist(s))
    return act


_rcache = {}


def run_episode(env, seed, p, frames_every=0):
    s = env.reset(np.random.default_rng(seed))
    pol = make_policy(p)
    tr = [s.copy()]
    frames = []
    done = False
    t = 0
    ci = {}
    n_inside = 0
    while not done and t < 500:
        a, ci = pol(s)
        s, done, info = env.step(s, a)
        tr.append(s.copy())
        if slot_dist(s) <= 0.07:
            n_inside += 1
        if frames_every and t % frames_every == 0 and t <= 170:
            # the gymnasium default camera looks up from UNDER the table
            # (measured: 4 near-identical useless frames) -- render through a
            # mujoco.Renderer on the model's own "corner2" camera instead
            if "ren" not in _rcache:
                import mujoco as _mj
                mdl = env._env.unwrapped.model
                _rcache["mj"] = _mj
                _rcache["mdl"] = mdl
                _rcache["dat"] = env._env.unwrapped.data
                _rcache["ren"] = _mj.Renderer(mdl, 360, 480)
            _rcache["ren"].update_scene(_rcache["dat"], camera="corner2")
            # this EGL stack returns the image vertically flipped (measured:
            # table hanging from the sky); flip rows back
            frames.append(_rcache["ren"].render()[::-1].copy())
        t += 1
    S = np.array(tr)
    return dict(seed=seed, steps=t,
                metric=float(env.task_metric(S)),
                success_any=bool(n_inside > 0),
                inside_frac=round(n_inside / max(t, 1), 4),
                final_slot_dist=round(slot_dist(s), 4),
                last_phase=ci.get("phase", "")), S, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--render", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="out_peg")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    from bird import registry
    registry.load_all()
    env = registry.get("env", "mt10_peg-insert-side-v3")(None)

    p = dict(DEFAULTS)
    for kv in args.set:
        k, v = kv.split("=")
        p[k] = float(v)

    rows = []
    for seed in args.seeds:
        fe = 3 if seed in args.render else 0
        r, S, F = run_episode(env, seed, p, frames_every=fe)
        rows.append(r)
        print(json.dumps(r), flush=True)
        if fe and F:
            import imageio
            od = Path(args.out) / ("s%d" % seed)
            od.mkdir(parents=True, exist_ok=True)
            idx = np.linspace(0, len(F) - 1, min(8, len(F))).astype(int)
            for k, i in enumerate(idx):
                imageio.imwrite(od / ("frame_%d.png" % k), F[i])
            imageio.mimsave(od / "clip.gif", F, fps=15, loop=0)
            np.savez_compressed(od / "states.npz", states=S)
    ms = [r["metric"] for r in rows]
    print(json.dumps(dict(
        n=len(rows), mean_metric=round(float(np.mean(ms)), 4),
        min_metric=round(float(np.min(ms)), 4),
        any_step_success_frac=round(
            float(np.mean([r["success_any"] for r in rows])), 4),
        anchors=dict(random=0.0, expert_any=0.8333, expert_per_step=0.5347),
        params=p)))


if __name__ == "__main__":
    main()
