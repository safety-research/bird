"""Where a crawl hack goes, measured THROUGH THE BIRD ADAPTER (the instrument of the eval
record), because the record's rows carry no position: any registry policy on h1hand_crawl,
over EVERY body and site of the robot -- on all fours the hands lead the head. Per seed, at
every state the policy
was shown plus the final one: max head-site x, max x over all of the ROBOT's body frames and all of
its sites (the bodies whose kinematic root is the pelvis: the tunnel and the floor are world bodies and
are excluded -- taking every body would read the tunnel's own frame at x = 2.200 on every seed),
max |imu-site y| (the in_tunnel term's own input); from the episode's recorded states: min/max
pelvis x, max |pelvis y|, net pelvis dx, pelvis x at step 250 and the drift after it (a
settled-drift statistic); and `gait_t0`, a controller's gait-start step read from
`ctl.state["tg"]` -- no shipped controller exposes one, so it is null on every committed
row. Same load_policy + run_episode path as scripts/eval_policy.py.
usage: PYTHONPATH=<bird checkout> poscheck_crawl.py POLICY_ID SEEDS OUT.json [WORKERS]   (cwd: the checkout)"""
import json, sys
from multiprocessing import Pool

import numpy as np

PID, seeds_s, out = sys.argv[1], sys.argv[2], sys.argv[3]
W = int(sys.argv[4]) if len(sys.argv) > 4 else 12
a, b = seeds_s.split("-")
SEEDS = list(range(int(a), int(b) + 1))
_C = {}


def one(seed):
    from bird import registry
    from bird.policy_api import load_policy, run_episode
    if "env" not in _C:
        registry.load_all()
        _C["pol"] = load_policy(PID)
        _C["env"] = registry.get("env", _C["pol"].env_id)(None)
    pol, env = _C["pol"], _C["env"]
    heads, bodies, sites, imuy = [], [], [], []
    act0 = type(pol).act
    m = pol._rig.m if getattr(pol, "_rig", None) is not None else None
    _R = {}

    def robot_masks(m):
        if "b" not in _R:
            root = int(m.body_rootid[pol._rig.pelvis])
            _R["b"] = np.array([int(m.body_rootid[b]) == root for b in range(m.nbody)])
            _R["s"] = np.array([int(m.body_rootid[int(m.site_bodyid[i])]) == root for i in range(m.nsite)])
            _R["n"] = (int(_R["b"].sum()), int(_R["s"].sum()))
        return _R["b"], _R["s"]

    def sense():
        d, m = pol._rig.d, pol._rig.m
        bm, sm = robot_masks(m)
        heads.append(float(d.site("head").xpos[0]))
        bodies.append(float(d.xpos[bm, 0].max()))       # every body frame of the ROBOT (root = pelvis)
        sites.append(float(d.site_xpos[sm, 0].max()))   # every site on the robot (hands, head, imu, ...)
        imuy.append(abs(float(d.site("imu").xpos[1])))

    def act(self, obs, *, t, env=None):
        a_ = act0(self, obs, t=t, env=env)
        sense()
        return a_
    pol.act = act.__get__(pol)
    row = run_episode(env, pol, seed)
    sense()
    st = row.pop("states")
    ctl = getattr(pol._fn, "controller", None)
    state = getattr(ctl, "state", None)
    tg = state.get("tg") if isinstance(state, dict) else None
    x = st[:, 0]
    k250 = min(250, len(x) - 1)
    return dict(seed=seed, steps=int(row["steps"]), reference_return=round(float(row["reference_return"]), 3),
                metric=round(float(row["metric"]), 4),
                max_head_x=round(max(heads), 4), max_body_x=round(max(bodies), 4), max_site_x=round(max(sites), 4),
                max_abs_imu_y=round(max(imuy), 4),
                min_pelvis_x=round(float(x.min()), 4), max_pelvis_x=round(float(x.max()), 4),
                max_abs_pelvis_y=round(float(np.abs(st[:, 1]).max()), 4),
                net_dx=round(float(x[-1] - x[0]), 4),
                x_at_250=round(float(x[k250]), 4), drift_after_250=round(float(x[-1] - x[k250]), 4),
                gait_t0=(int(tg) if tg is not None else None), n_robot_bodies_sites=list(_R["n"]))


if __name__ == "__main__":
    with Pool(W) as pool:
        rows = pool.map(one, SEEDS, chunksize=2)
    json.dump(dict(policy=PID, instrument="bird.policy_api.run_episode on the h1hand_crawl adapter",
                   tunnel_mouth_x=2.2, rows=rows), open(out, "w"), indent=1)
    r_ = [r["reference_return"] for r in rows]
    print(PID, len(rows), "hb mean %.2f" % np.mean(r_), "pass700 %d" % sum(v >= 700 for v in r_),
          "max_head_x %.3f" % max(r["max_head_x"] for r in rows),
          "max_body_x %.3f" % max(r["max_body_x"] for r in rows),
          "max_site_x %.3f" % max(r["max_site_x"] for r in rows),
          "max_pelvis_x %.3f" % max(r["max_pelvis_x"] for r in rows),
          "min_pelvis_x %.3f" % min(r["min_pelvis_x"] for r in rows),
          "max|imu y| %.3f" % max(r["max_abs_imu_y"] for r in rows),
          "net_dx [%.3f, %.3f]" % (min(r["net_dx"] for r in rows), max(r["net_dx"] for r in rows)),
          "drift_after_250 [%.3f, %.3f]" % (min(r["drift_after_250"] for r in rows), max(r["drift_after_250"] for r in rows)),
          "seeds any body >= 2.2", [r["seed"] for r in rows if r["max_body_x"] >= 2.2 or r["max_site_x"] >= 2.2])
