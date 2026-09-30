"""Harness for scripted HumanoidBench reward-hack policies.

The instrument of record is the UPSTREAM gym env (humanoid_bench @ cb11890):
episode return = sum of env.step()'s reward, i.e. exactly the number HumanoidBench's
success_bar (700 for walk and crawl) is a threshold on.  Nothing here recomputes it.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import time
import numpy as np
import gymnasium as gym
import mujoco
import humanoid_bench  # noqa: F401  (registers the envs)

TUNNEL_X0 = 2.2
_CACHE = {}


def _env(env_id, render):
    # One env per (process, id, render): constructing a fresh one per episode leaks file
    # handles until MuJoCo can no longer open the hand meshes. reset(seed) is the
    # benchmark's own reset, so reuse changes nothing about the episode.
    k = (env_id, render)
    if k not in _CACHE:
        _CACHE[k] = gym.make(env_id, render_mode="rgb_array" if render else None)
    return _CACHE[k]


class Rig:
    def __init__(self, env_id, seed=0, render=False):
        self.env = _env(env_id, render)
        self.u = self.env.unwrapped
        self.obs, _ = self.env.reset(seed=seed)
        self.m, self.d = self.u.model, self.u.data
        m = self.m
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

    # -- sensing ---------------------------------------------------------------
    def site(self, n):
        return self.d.site(n).xpos.copy()

    def com(self):
        return self.d.subtree_com[self.pelvis].copy()

    def comvel(self):
        return self.u.named.data.sensordata["pelvis_subtreelinvel"].copy()

    def pitch(self):
        """Pelvis forward pitch in radians (0 upright, +pi/2 face-down)."""
        R = self.d.xmat[self.pelvis].reshape(3, 3)
        return float(np.arctan2(R[0, 2], R[2, 2]))

    def q(self, jn):
        return float(self.d.joint(jn).qpos[0])

    # -- acting ----------------------------------------------------------------
    def action(self, targets):
        c = self.default.copy()
        for n, v in targets.items():
            c[self.idx[n]] = v
        c = np.clip(c, self.lo, self.hi)
        return 2 * (c - self.lo) / (self.hi - self.lo) - 1

    def step(self, targets, frames=None):
        a = self.action(targets)
        self.obs, r, term, trunc, info = self.env.step(a)
        self.t += 1
        self.ret += float(r)
        row = dict(t=self.t, r=float(r), x=float(self.d.qpos[0]), y=float(self.d.qpos[1]),
                   z=float(self.d.qpos[2]), vcom=float(self.comvel()[0]),
                   head=float(self.site("head")[2]), head_x=float(self.site("head")[0]), imu=float(self.site("imu")[2]),
                   pitch=self.pitch())
        for k, v in info.items():
            if np.isscalar(v) and k != "per_timestep_reward":
                row[k] = float(v)
        self.rows.append(row)
        if frames is not None:
            frames.append(self.env.render())
        if term or trunc:
            self.done = True
        return row

    # BIRD's own state metrics, for the side-by-side
    def bird_metrics(self, x0, task):
        xs = np.array([r["x"] for r in self.rows]); ys = np.array([r["y"] for r in self.rows])
        mean_v = (xs[-1] - x0) / (1000 * 0.02)  # BIRD normalises over the elapsed steps; a
        # full-length episode has 1000 of them, which every policy here reaches.
        prog = min(max(mean_v, 0.0), 1.0)
        if task == "crawl":
            inc = float(np.mean(np.abs(ys) <= 1.0))
            return dict(bird_task_metric=prog * inc,
                        bird_success_any=bool(np.any((xs > TUNNEL_X0) & (np.abs(ys) <= 1.0))),
                        net_dx=float(xs[-1] - x0), max_x=float(xs.max()))
        return dict(bird_task_metric=prog, bird_success_any=bool(np.any(xs >= 1.0)),
                    net_dx=float(xs[-1] - x0), max_x=float(xs.max()))


def ease(a, b, s):
    s = min(max(s, 0.0), 1.0)
    s = s * s * (3 - 2 * s)
    return a + (b - a) * s


def blend(p0, p1, s):
    keys = set(p0) | set(p1)
    return {k: ease(p0.get(k, 0.0), p1.get(k, 0.0), s) for k in keys}
