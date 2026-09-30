#!/usr/bin/env python3
"""Own scripted policies for the remaining eight MT10 tasks, evaluated
THROUGH the BIRD adapters (mt10_<task>-v3). Same rules as the peg/push
campaigns: Meta-World's bundled experts are the spec anchors, never the
policy; geometry/success checks are IMPORTED from the adapter module,
never reimplemented; every claim is the adapter's task_metric.

Shared machinery: mocap waypoint P-control (a[:3] = clip(k*(wp-hand)),
gain k in 1/cm, 0.01 m per unit-step), obs slots hand=s[:3], obj=s[4:7],
goal=s[36:39]. The grasp height floor (TCP cannot go below ~0.046 over a
tabletop object) and the corner2/row-flip renderer carry over from peg.

Policy shapes:
  reach          goto goal+(0,0,0.045) (the tcp offset the adapter's own
                 check documents), freeze.
  pick_place     approach/descend/close/carry(3D)/hold.
  button_press   hover over the button, press straight down, keep pressing.
  drawer_close   stand-off on the far side from the goal, push through, hold.
  window_open/   same shape as drawer_close: the handle slides in x; push
  window_close   from the away side through the goal.
  drawer_open    hover/descend onto the handle bar, close, PULL with a live
                 hand-offset (target = goal + (hand-obj)), hold at 0.025.
  door_open      hook the handle from the away side and drag with the same
                 live offset; the door constrains the arc, mocap follows.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from bird.envs import metaworld as MWA  # noqa: E402  (success transcriptions)

TASKS = {
    "reach": dict(env="mt10_reach-v3", check="_success_reach"),
    "pick_place": dict(env="mt10_pick-place-v3", check="_success_pick_place"),
    "door_open": dict(env="mt10_door-open-v3", check="_success_door_open"),
    "drawer_open": dict(env="mt10_drawer-open-v3",
                        check="_success_drawer_open"),
    "drawer_close": dict(env="mt10_drawer-close-v3",
                         check="_success_drawer_close"),
    "button_press_topdown": dict(env="mt10_button-press-topdown-v3",
                                 check="_success_button_press"),
    "window_open": dict(env="mt10_window-open-v3", check="_success_window"),
    "window_close": dict(env="mt10_window-close-v3", check="_success_window"),
}

P0 = dict(k=14.0, z_hover=0.06, z_grasp=0.02, xy_tol=0.010, d_tol=0.016,
          n_close=8, standoff=0.06, press_z=0.05, hold_d=0.05, pull_tol=0.022)


def _goto(a, p, hand, wp, grip):
    a[:3] = np.clip(p["k"] * (wp - hand), -1, 1)
    a[3] = grip


def pol_reach(p):
    def act(s, st):
        a = np.zeros(4)
        _goto(a, p, s[:3], s[36:39] + np.array([0, 0, 0.045]), 1.0)
        return a
    return act


def pol_pick_place(p):
    def act(s, st):
        hand, obj, goal = s[:3], s[4:7], s[36:39]
        a = np.zeros(4)
        ph = st.setdefault("phase", "approach")
        if ph == "approach":
            wp = np.array([obj[0], obj[1], obj[2] + p["z_hover"]])
            _goto(a, p, hand, wp, -1.0)
            if np.linalg.norm((hand - wp)[:2]) < p["xy_tol"] and \
                    abs(hand[2] - wp[2]) < 0.02:
                st["phase"] = "descend"
        elif ph == "descend":
            wp = np.array([obj[0], obj[1], obj[2] + p["z_grasp"]])
            _goto(a, p, hand, wp, -1.0)
            if np.linalg.norm(hand - wp) < p["d_tol"]:
                st["phase"] = "close"
                st["nc"] = 0
        elif ph == "close":
            a[3] = 1.0
            st["nc"] = st.get("nc", 0) + 1
            if st["nc"] >= p["n_close"]:
                st["phase"] = "carry"
        elif ph == "carry":
            _goto(a, p, hand, goal, 1.0)
            if np.linalg.norm(obj - goal) < p["hold_d"]:
                st["phase"] = "hold"
        else:
            a[3] = 1.0
        return a
    return act


def pol_button(p):
    def act(s, st):
        hand, obj = s[:3], s[4:7]
        a = np.zeros(4)
        ph = st.setdefault("phase", "hover")
        if ph == "hover":
            wp = np.array([obj[0], obj[1], obj[2] + p["z_hover"]])
            _goto(a, p, hand, wp, 1.0)
            if np.linalg.norm((hand - wp)[:2]) < p["xy_tol"]:
                st["phase"] = "press"
        else:
            wp = np.array([obj[0], obj[1], obj[2] - p["press_z"]])
            _goto(a, p, hand, wp, 1.0)
        return a
    return act


def _push_through(p, z_of=None):
    """drawer_close / window_*: stand off on the side away from the goal,
    then drive through the goal along the push line. Contact does the work."""
    def act(s, st):
        hand, obj, goal = s[:3], s[4:7], s[36:39]
        a = np.zeros(4)
        u = obj[:2] - goal[:2]
        n = np.linalg.norm(u)
        u = u / n if n > 1e-6 else np.array([1.0, 0.0])
        zc = obj[2] if z_of is None else z_of(obj)
        ph = st.setdefault("phase", "standoff")
        if ph == "standoff":
            wp = np.array([obj[0] + p["standoff"] * u[0],
                           obj[1] + p["standoff"] * u[1], zc + 0.02])
            _goto(a, p, hand, wp, 1.0)
            if np.linalg.norm(hand - wp) < p["d_tol"]:
                st["phase"] = "push"
        else:
            wp = np.array([goal[0] - 0.02 * u[0], goal[1] - 0.02 * u[1], zc])
            _goto(a, p, hand, wp, 1.0)
        return a
    return act


def pol_drawer_open(p):
    def act(s, st):
        hand, obj, goal = s[:3], s[4:7], s[36:39]
        a = np.zeros(4)
        ph = st.setdefault("phase", "hover")
        if ph == "hover":
            wp = np.array([obj[0], obj[1], obj[2] + p["z_hover"]])
            _goto(a, p, hand, wp, -1.0)
            if np.linalg.norm((hand - wp)[:2]) < p["xy_tol"]:
                st["phase"] = "descend"
        elif ph == "descend":
            wp = obj.copy()
            _goto(a, p, hand, wp, -1.0)
            if np.linalg.norm(hand - wp) < p["d_tol"]:
                st["phase"] = "close"
                st["nc"] = 0
        elif ph == "close":
            a[3] = 1.0
            st["nc"] = st.get("nc", 0) + 1
            if st["nc"] >= p["n_close"]:
                st["phase"] = "pull"
        elif ph == "pull":
            wp = goal + (hand - obj)     # live offset: carry the grip frame
            _goto(a, p, hand, wp, 1.0)
            if np.linalg.norm(obj - goal) < p["pull_tol"]:
                st["phase"] = "hold"
        else:
            a[3] = 1.0
        return a
    return act


def pol_door_open(p):
    def act(s, st):
        hand, obj, goal = s[:3], s[4:7], s[36:39]
        a = np.zeros(4)
        u = obj[:2] - goal[:2]
        n = np.linalg.norm(u)
        u = u / n if n > 1e-6 else np.array([1.0, 0.0])
        ph = st.setdefault("phase", "hover")
        if ph == "hover":
            wp = np.array([obj[0] + p["standoff"] * u[0],
                           obj[1] + p["standoff"] * u[1],
                           obj[2] + p["z_hover"]])
            _goto(a, p, hand, wp, -1.0)
            if np.linalg.norm((hand - wp)[:2]) < p["xy_tol"]:
                st["phase"] = "descend"
        elif ph == "descend":
            wp = np.array([obj[0] + 0.02 * u[0], obj[1] + 0.02 * u[1],
                           obj[2] - 0.02])
            _goto(a, p, hand, wp, -1.0)
            if np.linalg.norm(hand - wp) < p["d_tol"]:
                st["phase"] = "drag"
        elif ph == "drag":
            wp = goal + (hand - obj)
            _goto(a, p, hand, wp, 1.0)
            if abs(float(obj[0]) - float(goal[0])) <= 0.06:
                st["phase"] = "hold"
        else:
            a[3] = 1.0
        return a
    return act


POLICIES = {
    "reach": pol_reach,
    "pick_place": pol_pick_place,
    "button_press_topdown": pol_button,
    "drawer_close": _push_through,
    "window_open": _push_through,
    "window_close": _push_through,
    "drawer_open": pol_drawer_open,
    "door_open": pol_door_open,
}

_rc = {}


def run_episode(env, task, seed, p, frames_every=0):
    check = getattr(MWA, TASKS[task]["check"])
    s = env.reset(np.random.default_rng(seed))
    pol = POLICIES[task](p)
    st = {}
    tr = [s.copy()]
    frames = []
    done = False
    t = 0
    n_inside = 0
    while not done and t < 500:
        a = pol(s, st)
        s, done, info = env.step(s, a)
        tr.append(s.copy())
        n_inside += int(check(s, s[4:7], s[36:39]))
        if frames_every and t % frames_every == 0 and t <= 240:
            if "ren" not in _rc:
                import mujoco as _mj
                _rc["ren"] = _mj.Renderer(env._env.unwrapped.model, 360, 480)
            _rc["ren"].update_scene(env._env.unwrapped.data, camera="corner2")
            frames.append(_rc["ren"].render()[::-1].copy())
        t += 1
    S = np.array(tr)
    return dict(task=task, seed=seed, steps=t,
                metric=float(env.task_metric(S)),
                success_any=bool(n_inside > 0),
                inside_frac=round(n_inside / max(t, 1), 4),
                last_phase=st.get("phase", "-")), S, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--render", type=int, nargs="*", default=[])
    ap.add_argument("--out", default="out_suite")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    from bird import registry
    registry.load_all()
    env = registry.get("env", TASKS[args.task]["env"])(None)

    p = dict(P0)
    for kv in args.set:
        k, v = kv.split("=")
        p[k] = float(v)

    rows = []
    for seed in args.seeds:
        fe = 3 if seed in args.render else 0
        r, S, F = run_episode(env, args.task, seed, p, frames_every=fe)
        rows.append(r)
        print(json.dumps(r), flush=True)
        if fe and F:
            import imageio
            od = Path(args.out) / args.task / ("s%d" % seed)
            od.mkdir(parents=True, exist_ok=True)
            idx = np.linspace(0, len(F) - 1, min(8, len(F))).astype(int)
            for k, i in enumerate(idx):
                imageio.imwrite(od / ("frame_%d.png" % k), F[i])
            imageio.mimsave(od / "clip.gif", F, fps=15, loop=0)
    ms = [r["metric"] for r in rows]
    out = dict(task=args.task, n=len(rows),
               mean_metric=round(float(np.mean(ms)), 4),
               min_metric=round(float(np.min(ms)), 4),
               any_step=round(float(np.mean([r["success_any"]
                                             for r in rows])), 4),
               params={k: p[k] for k in sorted(p)})
    print(json.dumps(out))
    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / ("final_%s.json" % args.task), "w") as f:
        json.dump(dict(summary=out, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
