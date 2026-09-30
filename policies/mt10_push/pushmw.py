#!/usr/bin/env python3
"""A scripted policy for MT10 push-v3 through the BIRD adapter
(mt10_push-v3). Success (shipped): ||obs[4:7] - target|| <= 0.05, 3D, over a
500-step horizon with no early termination. Anchors (spec, n=30 bundled
expert): any-step 1.0, per-step 0.5803; random 0.0 — the expert closes every
episode, so the bar is match-1.0-then-beat-the-dwell.

Approach: grasp-and-drag. The puck is graspable, and dragging it AT TABLE
HEIGHT to the goal keeps the manner honest to the task's name while making
the terminal condition trivial to hold (grip closed, freeze — per-step
metric pays the dwell). Phases: approach/descend/close/drag/hold, with a
dropped-puck regrasp. All geometry from the same obs slots the adapter's
success check reads (obj = obs[4:7], goal = obs[36:39]); grasp height gated
on the peg campaign's measured TCP z-floor, not asset arithmetic.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import os
os.environ.setdefault("MUJOCO_GL", "egl")

DEFAULTS = dict(k=14.0, z_hover=0.06, z_grasp=0.02, xy_tol=0.010,
                d_tol=0.016, n_close=8, z_drag=0.035, hold_d=0.035,
                drop_dist=0.12)


def make_policy(p):
    st = dict(phase="approach")

    def act(s):
        hand = s[:3]
        obj = s[4:7]
        goal = s[36:39]
        a = np.zeros(4)

        def goto(wp, grip):
            a[:3] = np.clip(p["k"] * (wp - hand), -1, 1)
            a[3] = grip

        if st["phase"] in ("drag", "hold") and \
                np.linalg.norm(hand[:2] - obj[:2]) > p["drop_dist"]:
            st["phase"] = "approach"   # lost the puck

        if st["phase"] == "approach":
            wp = np.array([obj[0], obj[1], obj[2] + p["z_hover"]])
            goto(wp, -1.0)
            if np.linalg.norm((hand - wp)[:2]) < p["xy_tol"] and \
                    abs(hand[2] - wp[2]) < 0.02:
                st["phase"] = "descend"
        elif st["phase"] == "descend":
            wp = np.array([obj[0], obj[1], obj[2] + p["z_grasp"]])
            goto(wp, -1.0)
            if np.linalg.norm(hand - wp) < p["d_tol"]:
                st["phase"] = "close"
                st["nc"] = 0
        elif st["phase"] == "close":
            a[:] = 0.0
            a[3] = 1.0
            st["nc"] = st.get("nc", 0) + 1
            if st["nc"] >= p["n_close"]:
                st["phase"] = "drag"
        elif st["phase"] == "drag":
            wp = np.array([goal[0], goal[1], p["z_drag"]])
            goto(wp, 1.0)
            if np.linalg.norm(obj - goal) < p["hold_d"]:
                st["phase"] = "hold"
        elif st["phase"] == "hold":
            a[:] = 0.0
            a[3] = 1.0
        return a, dict(phase=st["phase"],
                       gd=float(np.linalg.norm(obj - goal)))
    return act


_rc = {}


def run_episode(env, seed, p, frames_every=0):
    s = env.reset(np.random.default_rng(seed))
    pol = make_policy(p)
    tr = [s.copy()]
    frames = []
    done = False
    t = 0
    n_inside = 0
    ci = {}
    while not done and t < 500:
        a, ci = pol(s)
        s, done, info = env.step(s, a)
        tr.append(s.copy())
        if np.linalg.norm(s[4:7] - s[36:39]) <= 0.05:
            n_inside += 1
        if frames_every and t % frames_every == 0 and t <= 220:
            if "ren" not in _rc:
                import mujoco as _mj
                _rc["ren"] = _mj.Renderer(env._env.unwrapped.model, 360, 480)
                _rc["dat"] = env._env.unwrapped.data
            _rc["ren"].update_scene(_rc["dat"], camera="corner2")
            frames.append(_rc["ren"].render()[::-1].copy())
        t += 1
    S = np.array(tr)
    return dict(seed=seed, steps=t, metric=float(env.task_metric(S)),
                success_any=bool(n_inside > 0),
                inside_frac=round(n_inside / max(t, 1), 4),
                final_gd=round(float(np.linalg.norm(s[4:7] - s[36:39])), 4),
                last_phase=ci.get("phase", "")), S, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--render", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="out_pushmw")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    from bird import registry
    registry.load_all()
    env = registry.get("env", "mt10_push-v3")(None)

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
    ms = [r["metric"] for r in rows]
    print(json.dumps(dict(
        n=len(rows), mean_metric=round(float(np.mean(ms)), 4),
        min_metric=round(float(np.min(ms)), 4),
        any_step=round(float(np.mean([r["success_any"] for r in rows])), 4),
        anchors=dict(random=0.0, expert_any=1.0, expert_per_step=0.5803),
        params=p)))


if __name__ == "__main__":
    main()
