"""Classic-control pendulum: the first *real* reward-design target (§0, §3).

`bird.envs.toy` exists to make the loop *run*; this module exists to make it
*mean something*. The three toy worlds are measured so that a good reward is
distinguishable from a bad one in under a second offline, which is what a test
suite needs and what a result does not. Pendulum swing-up is the smallest
environment on which the published claims can actually be re-run, and it is the
right first target for one specific reason:

    **Its standard reward is already a hand-tuned shaped cost.**
    `-(theta**2 + 0.1*theta_dot**2 + 0.001*torque**2)` is three terms with two
    tuned coefficients, written by a human, shipped in the benchmark, and cited
    for fifteen years. That artifact -- a human-weighted sum of an error term,
    a regulariser and an effort penalty -- is *precisely* the object every
    LLM reward-design method claims to automate. Eureka claims to beat it,
    RDA/GT claim to reach it without a ground-truth metric, Singh (2009) claims
    the fitness-mirroring version of it is not the best one. Here the human's
    answer is available as `reference_reward`, so "did the LLM match the
    hand-tuned cost" is a measurement rather than an anecdote.

Two entries in the `env` family (`problem.env_id`):

  pendulum            continuous state and torque. The target for the
                      neural-network backends (`train.backend: sb3`) and the
                      env the continuous-control methods -- Eureka, DrEureka,
                      Text2Reward, RDA -- are configured against.
  pendulum_discrete   the *same physics* on a 31 x 31 state grid with 5
                      torques, so `discretise` is a bijection and
                      `train.backend: tabular` is an exact solve rather than an
                      implicit binning. This is what `configs/methods/singh_orp.yaml`
                      needs: exhaustive enumeration over tabular rewards
                      (`generate.generator_backend: exhaustive_enumeration`) is
                      only defined when there is a finite state space to write
                      a table over. Keeping the physics identical is the point:
                      "pre-LLM enumeration vs. LLM search" then differs in the
                      method, not in the problem.

What is deliberately *not* here:

  * **No gymnasium import.** The dynamics are forty lines of numpy. The sb3
    backend already wraps an `EnvAdapter` into a `gym.Env` itself
    (`training._sb3_run`), so importing gymnasium here would buy nothing and
    would make `bird.py --validate-all` -- which must load every config on any
    machine -- fail wherever the optional dependency is absent. `load_all()`
    only forgives an ImportError from `anthropic_client`, so a hard third-party
    import in this module is a hard break of the whole registry.
  * **No ground truth in the observation.** `task_metric` is upright-and-still
    (below), and nothing a candidate reward can read tells it what fraction of
    the final third of the episode was spent there. Fitness has to be earned
    through the dynamics, not copied out of the state vector; that leakage is
    what would make every number downstream of §4 meaningless.

Provenance: the constants are Gymnasium's `Pendulum-v1`
(`gymnasium/envs/classic_control/pendulum.py`: g=10.0, m=1.0, l=1.0, dt=0.05,
max_speed=8.0, max_torque=2.0, 200-step time limit), so returns here are
comparable to the numbers the RL literature quotes. Nothing in this file is a
pin taken from one of the reward-design papers -- it is the shared substrate
they get run on -- so it carries no †/‡ markers.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..registry import register
# `_bin` and `_states_of` are imported rather than re-implemented on purpose:
# binning semantics (half-open, clamped, `n` uniform cells) are load-bearing for
# `train.backend: tabular`, and a second definition that rounds differently at a
# cell edge would silently change which state a transition lands in.
from .spec import SpecEnvAdapter
from .base import EnvAdapter, _bin, _states_of

__all__ = ["Pendulum", "PendulumDiscrete"]


TWO_PI = 2.0 * math.pi


def _wrap(theta: float) -> float:
    """Angle to [-pi, pi). `theta = 0` is upright, matching Pendulum-v1."""
    return float(((theta + math.pi) % TWO_PI) - math.pi)


def _clamp(i: int, n: int) -> int:
    return 0 if i < 0 else (n - 1 if i >= n else i)


def _centre(i: int, lo: float, hi: float, n: int) -> float:
    """The representative value of bin `i` -- its centre, not its edge.

    Centres matter: with an *odd* `n` on a symmetric interval one cell centre
    lands exactly on 0, so "upright" and "at rest" are states of the discretised
    system rather than boundaries between two of them. `ToyReacher` aligns its
    bins to the goal for the same reason.
    """
    return lo + (i + 0.5) * (hi - lo) / n


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
#
# `output.video.*` records rollouts as PNG frames, and §4's VLM comparators
# score the same artifact a human watches -- so a frame that a person cannot
# grade at a glance is a frame that silently degrades `evaluate.preferences`
# into noise. Hence: pure numpy rasterisation (stdlib + numpy is the entire
# runtime dependency set; matplotlib/PIL/cv2 would be a new one for a picture of
# a stick), and hence the target is drawn explicitly rather than left implicit
# in "the rod happens to be pointing up".

_BG = (247, 247, 250)
_ORBIT = (214, 216, 223)
_TARGET = (46, 160, 67)
_ROD = (32, 38, 52)
_BOB = (206, 62, 44)
_PIVOT = (16, 18, 24)
#: A second link shade, so a folded arm is visibly different from an extended
#: one. Posture is the information `acrobot` carries and `pendulum` cannot.
_LINK2 = (92, 104, 132)


def _stamp(img: np.ndarray, x: float, y: float, half: int, colour: Tuple[int, int, int]) -> None:
    """Paint a (2*half+1)^2 square centred on (x, y); clipped at the border."""
    h, w = img.shape[0], img.shape[1]
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(cx - half, 0), min(cx + half + 1, w)
    y0, y1 = max(cy - half, 0), min(cy + half + 1, h)
    if x0 < x1 and y0 < y1:
        img[y0:y1, x0:x1] = colour


def _segment(img: np.ndarray, p0: Tuple[float, float], p1: Tuple[float, float],
             half: int, colour: Tuple[int, int, int]) -> None:
    """A thick line, drawn by stamping along it at sub-pixel spacing.

    Sampling at one point per pixel of the longer axis is exactly what makes the
    stroke gap-free at any angle without a Bresenham special case.
    """
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    for t in np.linspace(0.0, 1.0, max(n, 2)):
        _stamp(img, p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]), half, colour)


def _disc(img: np.ndarray, x: float, y: float, r: float, colour: Tuple[int, int, int]) -> None:
    """A filled circle, via a boolean mask on its bounding box."""
    h, w = img.shape[0], img.shape[1]
    x0, x1 = max(int(x - r) - 1, 0), min(int(x + r) + 2, w)
    y0, y1 = max(int(y - r) - 1, 0), min(int(y + r) + 2, h)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1)[None, :] - x
    ys = np.arange(y0, y1)[:, None] - y
    img[y0:y1, x0:x1][(xs * xs + ys * ys) <= r * r] = colour


def _ring(img: np.ndarray, x: float, y: float, r: float, half: int,
          colour: Tuple[int, int, int]) -> None:
    """An unfilled circle. Used for the swing path and the target marker."""
    for a in np.linspace(0.0, TWO_PI, max(24, int(8 * r))):
        _stamp(img, x + r * math.sin(a), y - r * math.cos(a), half, colour)


# ==========================================================================
# pendulum -- continuous swing-up
# ==========================================================================


class Pendulum(SpecEnvAdapter):
    """Under-actuated pendulum swing-up; Gymnasium `Pendulum-v1` in numpy.

    A rod hangs from a pivot. `max_torque = 2.0` is *not enough* to lift it
    against gravity from horizontal, so the only way up is to pump energy by
    swinging back and forth first -- which is the property that makes this a
    reward-design problem instead of a regression. Discrimination is real and
    was measured, not assumed (numbers under `success_threshold`): rewards that
    price only motion or only effort (`-theta_dot**2`, `-torque**2`) leave the
    rod hanging at 0.00-0.13 on the metric below, while rewards that pay for
    being up score 0.60-1.00. Both kinds are things an LLM writes.

    One result is worth stating because it is
    the repo's whole thesis in miniature: a bare `cos(theta)` reward *beats* the
    hand-tuned three-term cost under the tabular learner, 1.00 vs 0.93 on the
    continuous env and 0.98 vs 0.82 on the discrete one. The published cost's
    velocity and effort terms buy smoothness the ground-truth metric does not
    ask for. The human's answer being suboptimal is exactly the gap Eureka et
    al. claim to close -- and on this env it is visible in a two-minute run.

    Stated as a mean over 5 training seeds, and it must be: the ordering
    *inverts on individual seeds* (the reference reward wins outright on the
    discrete env at seed 0, 1.00 vs 0.96). Anyone re-deriving this from one run
    will get the opposite answer half the time. Treat the single-seed number as
    uninformative here -- which is itself the reason `train.seeds_per_candidate`
    and `final_retrain.n_seeds` are separate knobs (§3/§6).

    The episode never terminates early (`done` is always False). "Balance for
    the rest of the episode" is only definable over a fixed horizon, and early
    termination would make the metric length-dependent -- the same reasoning as
    `ToyReacher`, and the reason `Pendulum-v1` itself has no termination
    condition, only a 200-step limit.
    """

    name = "pendulum"
    obs_dim = 3
    action_dim = 1
    horizon = 200

    # -- Pendulum-v1 constants (gymnasium/envs/classic_control/pendulum.py) --
    g = 10.0
    m = 1.0
    l = 1.0
    dt = 0.05
    max_speed = 8.0
    max_torque = 2.0

    #: Ground-truth tolerance, and *only* ground truth: nothing in the state
    #: vector announces these numbers, so a candidate reward cannot read the
    #: metric off the observation. ~11.5 degrees and 1 rad/s is the band inside
    #: which a human calls the rod "balanced", and it is wide enough that the
    #: 31-bin grid of `PendulumDiscrete` can express staying inside it.
    upright_angle = 0.2
    upright_speed = 1.0

    #: Measured, not guessed. Measured with this repo's own tabular surrogate
    #: (`training._QLearner`, gamma 0.99, 200k env steps, then the greedy policy
    #: over 30 evaluation episodes), mean `task_metric` over **5 training
    #: seeds**, given as `pendulum` / `pendulum_discrete` with the per-seed
    #: spread where it is wide:
    #:
    #:     +cos(th)                          1.00           / 0.98 (.96-1.00)
    #:     -(th^2 + 0.1 thd^2 + 0.001 u^2)   0.93 (.72-1.0) / 0.82 (.60-1.00)
    #:     -thd^2       (stillness only)     0.13           / 0.03 (.00-0.07)
    #:     -u^2         (effort only)        0.05           / 0.00
    #:     0            (constant)           0.04 (.00-.13) / 0.01 (.00-0.05)
    #:     +th^2        (sign-flipped)       0.00           / 0.00
    #:     uniform random policy             0.00           / 0.00
    #:
    #: Quoted over seeds because a single seed is not reproducible here: the
    #: reference reward alone spans 0.60-1.00 across seeds on the discrete env,
    #: so any conclusion drawn from one run of it is noise.
    #:
    #: The per-episode metric is sharply bimodal -- the rod either settles
    #: (~1.0) or never gets up (~0.0). Over the 1080 evaluation episodes behind
    #: the table above, 1.9% land strictly inside (0.2, 0.8) and 3.1% inside
    #: (0.1, 0.9), so the threshold's exact value reclassifies almost nothing.
    #: 0.5 reads as "balanced for at least half of the final third", which is
    #: also the weakest claim a human watching the clip would call a success.
    success_threshold = 0.5

    #: 31 x 31. Odd on both axes so that one cell centre is exactly
    #: (theta, theta_dot) = (0, 0) -- see `_centre`. 31 gives a 0.203 rad
    #: angular cell, just under the 0.2 rad success tolerance, so "in the top
    #: cell" implies "upright" and the discretisation cannot manufacture
    #: success. 961 states x 5 torques = 4805 Q-entries, which a laptop fills
    #: inside `train.env_steps` and which is also the size of the tabular reward
    #: space `configs/singh_orp.yaml` enumerates over.
    _theta_bins = 31
    _thetadot_bins = 31
    n_disc_states = 31 * 31
    #: Continuous: `discretise` is a binning, not a bijection. `training.py`
    #: keys the sb3 observation/action spaces off exactly this
    #: (`getattr(env, "exact_states", None) is None` => Box action space).
    exact_states = None

    #: 5 torques, odd so that zero torque exists -- without it the rod cannot be
    #: *held* at the top and the task metric would be unreachable by
    #: construction. This is only a finite proxy for the box action space
    #: [-2, 2]: `train.backend: sb3` uses the box, while the tabular learner,
    #: the random-shooting planner (`train.backend: none`) and
    #: `sample_transitions` (the EPIC/STARC screens) draw from this set. Both
    #: envs share it, so `pendulum` -> `pendulum_discrete` changes the *state*
    #: representation and nothing else.
    n_torques = 5




    # -- construction -------------------------------------------------------

    def _build_action_set(self) -> Any:
        return [[t] for t in np.linspace(-self.max_torque, self.max_torque, self.n_torques)]

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array([-1.0, -1.0, -self.max_speed]),
                np.array([1.0, 1.0, self.max_speed]))

    # -- state <-> observation ---------------------------------------------

    def _obs(self, theta: float, theta_dot: float) -> np.ndarray:
        return np.array([math.cos(theta), math.sin(theta), theta_dot], dtype=float)

    @staticmethod
    def _angle(s: np.ndarray) -> float:
        """Recover theta from (cos, sin). Robust to the un-normalised pairs that
        come back out of a JSON round trip, which `atan2` handles and
        `arccos(s[0])` would not (it also loses the sign)."""
        return float(math.atan2(float(s[1]), float(s[0])))

    # -- dynamics -----------------------------------------------------------

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        # Pendulum-v1's own reset: theta ~ U(-pi, pi), theta_dot ~ U(-1, 1).
        # The wide angular spread is deliberate and must not be narrowed: some
        # variants and tutorials start the rod near the top, which quietly turns
        # swing-up into balancing -- a task an LQR-shaped reward solves and for
        # which no reward *design* is needed. Every episode here starts, on
        # average, half a swing away from upright, so the energy-pumping
        # behaviour is required and a reward that cannot elicit it scores zero.
        return self._obs(float(rng.uniform(-math.pi, math.pi)),
                         float(rng.uniform(-1.0, 1.0)))

    def _torque(self, a: np.ndarray) -> float:
        """Commanded torque -> torque actually applied, under the DR draw.

        Noise is added to the *command* and the actuator limit is applied after
        it (you cannot command your way past a saturated motor), then
        `torque_scale` models motor strength -- so a weak motor stays weak no
        matter what the policy asks for. Same ordering as `ToyReacher`.
        """
        dr = self._dr_now
        u = float(a[0]) if a.size else 0.0
        if dr["action_noise"] > 0.0:
            u += float(self._rng.normal(0.0, dr["action_noise"]))
        u = min(max(u, -self.max_torque), self.max_torque)
        return u * dr["torque_scale"]

    def _dynamics(self, s: np.ndarray, a: np.ndarray) -> Tuple[float, float, float]:
        """One Euler step of the continuous system: `(theta, theta_dot, u)`.

        Verbatim Pendulum-v1:
            newthdot = clip(thdot + (3g/(2l) sin(th) + 3/(m l^2) u) dt, +/-8)
            newth    = th + newthdot dt
        Kept as a separate method so `PendulumDiscrete` discretises the *same*
        physics instead of owning a second copy that can drift from this one.
        """
        dr = self._dr_now
        mass, length = max(dr["mass"], 1e-3), max(dr["length"], 1e-3)
        u = self._torque(a)
        theta, theta_dot = self._angle(s), float(s[2])
        acc = (3.0 * dr["gravity"] / (2.0 * length)) * math.sin(theta) \
            + (3.0 / (mass * length * length)) * u
        theta_dot = min(max(theta_dot + acc * self.dt, -self.max_speed), self.max_speed)
        return _wrap(theta + theta_dot * self.dt), theta_dot, u

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        theta, theta_dot, u = self._dynamics(s, a)
        return self._obs(theta, theta_dot), False, {
            "success": abs(theta) < self.upright_angle and abs(theta_dot) < self.upright_speed,
            "theta": theta, "theta_dot": theta_dot, "torque": u,
        }

    # -- discretisation ------------------------------------------------------

    def discretise(self, s: np.ndarray) -> int:
        """(theta, theta_dot) -> cell index. A *binning* here: many continuous
        states share a cell, which is why `exact_states is None` and why the
        tabular backend reports itself as approximate on this env."""
        ti = _bin(self._angle(s), -math.pi, math.pi, self._theta_bins)
        di = _bin(float(s[2]), -self.max_speed, self.max_speed, self._thetadot_bins)
        return ti * self._thetadot_bins + di

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        # Coverage, not the initial distribution: the EPIC/STARC pseudometrics
        # compare reward functions over the whole space, including the fast
        # spinning states no sane policy visits.
        return self._obs(float(rng.uniform(-math.pi, math.pi)),
                         float(rng.uniform(-self.max_speed, self.max_speed)))

    # -- ground truth --------------------------------------------------------

    def task_metric(self, traj: Any) -> float:
        """Fraction of the *final third* of the episode spent upright and still.

        Three properties this definition is chosen for:

        * **It is a settling criterion, not a return.** Scoring the whole
          episode would reward a lucky swing that passes through the top at
          speed; scoring only the tail asks the question the task actually poses
          ("is it balanced at the end"), and the swing-up is then rewarded only
          instrumentally.
        * **It is in [0, 1] by construction**, as `EnvAdapter` requires, and
          `success()` is a threshold on it, so binary and continuous ground
          truth cannot disagree.
        * **It is not derivable from a candidate's reward.** No term here is a
          per-step quantity a reward function is given; a candidate can only
          move this number by changing the *policy*. That is the leakage this
          repo exists to prevent -- if fitness were readable from the reward,
          every §4 number (fitness, TAC/TPE, Pearson curve, BT strength) would
          measure the LLM's ability to restate the metric.

        Scored over the arriving states `s_1..s_T` (row 0 is the initial state,
        which no action produced), consistent with `reference_reward`.
        """
        states = _states_of(traj)
        if states.shape[0] < 2 or states.shape[1] < 3:
            return 0.0
        arrived = states[1:]
        tail = arrived[-max(1, int(math.ceil(arrived.shape[0] / 3.0))):]
        theta = np.arctan2(tail[:, 1], tail[:, 0])
        return float(np.mean((np.abs(theta) < self.upright_angle)
                             & (np.abs(tail[:, 2]) < self.upright_speed)))

    # --- BEGIN reference reward (ground truth; strip before showing an LLM) ---
    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """Pendulum-v1's shaped cost, negated so that higher is better.

            r = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * torque**2)

        This *is* the artifact under study: an error term, a hand-weighted
        velocity regulariser at 0.1 and a hand-weighted effort penalty at 0.001,
        all chosen by a person. Gymnasium returns it as a cost (<= 0, maximised
        at 0); BIRD's convention throughout is that a reward is maximised, and
        the negation is only a sign -- the optimal policy is unchanged.

        One deliberate off-by-one against the original: Gymnasium charges the
        cost of the state *before* the step, while BIRD's convention
        (`bird.envs.toy`) is that `reference_reward(s_{t+1}, a_t)` scores the
        transition that arrives in `s_{t+1}`. Over a fixed 200-step horizon the
        two differ by one boundary term (the initial state versus the final
        one), which shifts every return by an amount no policy controls and
        therefore changes no ranking. Recorded because gt_return values here are
        meant to be comparable with published Pendulum-v1 numbers.
        """
        s = np.asarray(s, dtype=float)
        theta = self._angle(s)
        cost = theta * theta + 0.1 * float(s[2]) ** 2
        if a is not None:
            u = float(np.asarray(a, dtype=float).ravel()[0])
            cost += 0.001 * u * u
        return -cost
    # --- END reference reward ---

    # -- what the task asks, for a frame legend (`EnvAdapter.legend_lines`) ----

    def legend_lines(self, state: np.ndarray) -> List[str]:
        """Three lines: the ask, and the two tolerances that define "balanced".

        THE ASK, NEVER THE ANSWER -- `EnvAdapter.legend_lines` states the rule.
        Every line here is a
        CONSTANT of the task: the same three strings on the first frame of a swing-up
        as on the last frame of a rod that never left the bottom. The angle and the
        speed are in `state` and are deliberately not printed: beside the tolerances
        they would be the per-step verdict in numbers, on the very frames
        `evaluate.preferences`' VLM comparators grade (`render`'s docstring). The
        tolerances themselves leak nothing a frame does not already show -- `render`
        draws them as the ticks at +/-`upright_angle` -- and they are the whole
        content of "balanced": `task_metric` is upright AND still, so a legend naming
        only the angle would state half the ask.

        Built from `upright_angle` / `upright_speed` rather than typed, so the frame
        cannot say 0.2 rad when the tolerance is something else. `PendulumDiscrete`
        inherits it unchanged: same physics, same ask.

        `legend_on_frame` stays False -- `render` returns the bare frame, and a
        caller that composes the legend over it draws it once.
        `step_success_is_a_check` stays True: `_step`'s per-step flag
        is the real upright-and-still check, not a constant.
        """
        return ["SWING UP, HOLD UPRIGHT",
                f"UPRIGHT: ANGLE < {self.upright_angle:g} RAD",
                f"STILL: SPEED < {self.upright_speed:g} RAD/S"]

    # -- rendering -----------------------------------------------------------

    def render(self, state: np.ndarray, width: int = 320) -> np.ndarray:
        """One RGB frame, `(width, width, 3)` uint8. `output.video.width` = 320.

        Not decorative: this frame is the input to `evaluate.preferences`'
        VLM comparators and to a human grading the same clip, and those two must
        be looking at the same picture or their agreement rate (§4, RDA's core
        measurement) means nothing. So the *target* is drawn, not implied -- a
        green ring at the upright tip position plus the tolerance ticks at
        +/-0.2 rad -- because "is this rod near the target" is a question a VLM
        answers reliably from a marked frame and unreliably from an unmarked
        one, and a single still frame carries no motion to compare against.
        """
        w = max(48, int(width))
        img = np.full((w, w, 3), _BG, dtype=np.uint8)
        cx = cy = (w - 1) / 2.0
        radius = 0.34 * w          # leaves room for the tolerance ticks outside it
        rod_half = max(1, int(round(0.012 * w)))
        tick_half = max(1, int(round(0.005 * w)))
        bob_r = max(2.0, 0.045 * w)

        s = np.asarray(state, dtype=float).ravel()
        theta = self._angle(s) if s.size >= 2 else 0.0

        # Swing path, then the target, then the rod: painter's order, so the
        # rod is never hidden by the marks that describe where it should be.
        _ring(img, cx, cy, radius, 0, _ORBIT)
        # Ticks *outside* the swing path, so they read as a tolerance gate the
        # bob passes between rather than as more of the target marker.
        for edge in (-self.upright_angle, self.upright_angle):
            _segment(img,
                     (cx + 1.14 * radius * math.sin(edge), cy - 1.14 * radius * math.cos(edge)),
                     (cx + 1.30 * radius * math.sin(edge), cy - 1.30 * radius * math.cos(edge)),
                     tick_half, _TARGET)
        _ring(img, cx, cy - radius, bob_r + 2.0, 1, _TARGET)

        tip = (cx + radius * math.sin(theta), cy - radius * math.cos(theta))
        _segment(img, (cx, cy), tip, rod_half, _ROD)
        _disc(img, tip[0], tip[1], bob_r, _BOB)
        _disc(img, cx, cy, max(2.0, 0.018 * w), _PIVOT)
        return img


# ==========================================================================
# pendulum_discrete -- the same physics on an enumerable grid
# ==========================================================================


class PendulumDiscrete(Pendulum):
    """Pendulum on a 31 x 31 state grid with 5 torques: 961 exact states.

    Why it exists: `configs/singh_orp.yaml` is pre-LLM exhaustive enumeration
    over *tabular* rewards, and a table needs an index set. `exact_states` is
    BIRD's declaration that `discretise` is a bijection, which is what makes
    `train.backend: tabular` an exact solve of a real MDP rather than a
    function approximator pretending. `ToyHungryThirsty` is the only other env
    that can say this, and it is not a control problem -- so without this class
    the enumeration baseline and the continuous-control methods can never be
    compared on the same task, and any comparison between them would be
    confounded by the environment.

    **The randomised rounding, and why it is not a hack.** Snapping the state to
    the nearest grid cell after each Euler step -- the obvious implementation --
    produces a *dead* pendulum. One step at the slowest non-zero speed bin moves
    the angle by 0.516 * 0.05 = 0.026 rad, an eighth of the 0.203 rad angular
    cell, so the deterministic snap returns the rod to the cell it started in,
    for ever, at every speed below 4 rad/s. The same argument kills the
    acceleration: a torque of 2 changes theta_dot by 0.3, well under the
    0.516 speed cell. A deterministic grid would therefore have to be so coarse
    that the physics is gone, or so fine that it is not enumerable.

    Instead the state advances to the neighbouring cell with probability equal
    to the fractional part of its continuous displacement, so
    `E[next grid state] == next continuous state` exactly. The result is a
    genuine finite stochastic MDP whose mean dynamics are Pendulum-v1's, which
    is both the standard construction for discretising a continuous system and
    the honest one: the discretisation error is now visible as transition noise
    rather than hidden as a systematic bias toward standing still. Singh's own
    domain is stochastic too, so the enumeration baseline is not being handed an
    easier world than the paper's.
    """

    name = "pendulum_discrete"
    #: 31 * 31, and `discretise` below is a bijection onto [0, 961).
    exact_states = 31 * 31

    # `_prose`, `_state_fields` and `_action_fields` come from
    # `tasks/pendulum_discrete/shared_spec.yaml` and are NOT declared here.
    # `_render_full_source` below concatenates `Pendulum`'s source with THIS class's, so
    # its body reaches the model in full; a declaration here would be overwritten at
    # runtime by `_apply_spec` and still shipped as text, which is two answers to one
    # question in one prompt.

    # -- the grid ------------------------------------------------------------

    def _grid_obs(self, ti: int, di: int) -> np.ndarray:
        """Cell indices -> the observation at that cell's centre."""
        return self._obs(_centre(ti, -math.pi, math.pi, self._theta_bins),
                         _centre(di, -self.max_speed, self.max_speed, self._thetadot_bins))

    def _round(self, v: float, lo: float, hi: float, n: int, wrap: bool) -> int:
        """Randomised rounding of `v` onto the cell grid; see the class docstring.

        `p` is `v` in units of cells, measured from the first cell's *centre*, so
        `floor(p)` and `floor(p) + 1` are the two neighbouring centres and
        `p - floor(p)` is the exact probability the upper one must get for the
        expectation to be preserved. Angles wrap (index -1 is index n-1, which
        is what makes the top of the swing a continuous piece of the state
        space); velocities clamp, matching the `+/-max_speed` clip.
        """
        p = (v - lo) / ((hi - lo) / n) - 0.5
        i = int(math.floor(p))
        if float(self._rng.random()) < (p - i):
            i += 1
        return i % n if wrap else _clamp(i, n)

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        # The same initial distribution as the continuous env, then projected
        # onto the grid -- projected with the same randomised rounding, so the
        # initial distribution's mean is preserved too and the two envs are not
        # quietly starting from different places.
        s = super()._reset(rng)
        return self._grid_obs(
            self._round(self._angle(s), -math.pi, math.pi, self._theta_bins, True),
            self._round(float(s[2]), -self.max_speed, self.max_speed, self._thetadot_bins, False))

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        # Identical physics (`_dynamics` is inherited, not copied), then one
        # projection back onto the grid. The state space is closed under this:
        # every reachable state is a cell centre, which is what `exact_states`
        # promises the tabular backend.
        theta, theta_dot, u = self._dynamics(s, a)
        ti = self._round(theta, -math.pi, math.pi, self._theta_bins, True)
        di = self._round(theta_dot, -self.max_speed, self.max_speed, self._thetadot_bins, False)
        s2 = self._grid_obs(ti, di)
        return s2, False, {
            "success": abs(self._angle(s2)) < self.upright_angle
            and abs(float(s2[2])) < self.upright_speed,
            "theta": float(self._angle(s2)), "theta_dot": float(s2[2]), "torque": u,
        }

    def discretise(self, s: np.ndarray) -> int:
        """Cell centre -> its index: a bijection on the reachable state set.

        Floor binning, like `_bin`, but with one difference that `_bin` cannot
        express and that the bijection depends on: **theta wraps, it does not
        clamp**. `_angle` returns `atan2(...) in [-pi, pi]`, and a rod at
        exactly +pi is the *same physical state* as one at -pi -- cell 0.
        Clamping (what `_bin` does at the top of its range) would send it to
        cell 30 instead, so the top of the swing would be two cells that never
        merge and the state space would not be closed under `_step`.

        Floor is safe here despite the float round trip through
        `cos`/`sin`/`atan2`: a cell centre sits *half a cell* -- 0.1 rad, some
        14 orders of magnitude above the round-trip error -- inside its own bin,
        so no reachable state is anywhere near an edge. That is the point of
        `_centre`, and it is why this stays a bijection rather than merely
        usually agreeing with one.
        """
        n_t, n_d = self._theta_bins, self._thetadot_bins
        ti = int(math.floor((self._angle(s) + math.pi) / (TWO_PI / n_t))) % n_t
        di = _clamp(int(math.floor((float(s[2]) + self.max_speed)
                                   / (2.0 * self.max_speed / n_d))), n_d)
        return ti * n_d + di

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        # Uniform over the 961 cells: on a discrete env, coverage for the
        # EPIC/STARC screens means the uniform distribution over states, not a
        # uniform draw in a continuous box that then lands off-grid.
        return self._grid_obs(int(rng.integers(self._theta_bins)),
                              int(rng.integers(self._thetadot_bins)))

    # -- describe() ----------------------------------------------------------

    def _render_full_source(self) -> str:
        """Base class source *then* this one -- inheritance is not a hiding place.

        `EnvAdapter._render_full_source` is `inspect.getsource(type(self))`,
        which for a subclass is the subclass body alone. That default is right
        for every env in `bird.envs.toy` (they all derive straight from
        `EnvAdapter`) and quietly wrong here, because everything that makes this
        an environment -- `_dynamics`, `_torque`, `_bounds`,
        `_build_action_set`, `task_metric` and `reference_reward` -- lives on
        `Pendulum`. Two things break if that is left alone, and both are silent:

        * **`generate.context.strip_existing_reward` becomes a no-op.** The
          stripper (`generation._strip_reward`) cuts on `def *reward*(`, and
          there is no such def in this class body. So the ablation's two arms
          produce byte-identical prompts and the key reads as having no effect
          -- on the one env where the pre-LLM enumeration baseline is run. That
          is precisely the "a typo'd key quietly invalidates an ablation"
          failure, arrived at by inheritance instead of by typo.
        * **`full_source` would not contain the source.** Eureka and DrEureka
          are *defined* by showing the LLM the environment; handing them a
          40-line subclass with no dynamics in it would make `env_spec:
          full_source` differ between `pendulum` and `pendulum_discrete` for a
          reason that has nothing to do with either method.

        Concatenating text rather than walking `__mro__` generically: the join
        only has to be right for this one two-deep hierarchy, and the ordering
        (base first, so the override reads as an override) is the thing that
        makes the result legible to a model.
        """
        try:
            base = inspect.getsource(Pendulum)
            own = inspect.getsource(type(self))
        except (OSError, TypeError):  # zipimport / exec'd module
            return self._render_class_abstraction()
        return (f"# environment `{self.name}` -- full source, base class first\n\n"
                f"{base}\n\n{own}")


# ==========================================================================
# registry: `problem.env_id`
# ==========================================================================


@register("env", "pendulum")
def pendulum(ctx: Any) -> EnvAdapter:
    """Continuous swing-up; the hand-tuned-cost target (§0)."""
    return Pendulum()


@register("env", "pendulum_discrete")
def pendulum_discrete(ctx: Any) -> EnvAdapter:
    """Swing-up on a 961-state grid: exact tabular / enumeration (§0, §3)."""
    return PendulumDiscrete()


# ==========================================================================
# acrobot -- two links, and the first env in this repo a VLM can actually read
# ==========================================================================


class Acrobot(SpecEnvAdapter):
    """Two-link underactuated arm; swing the tip above the bar and keep it there.

    WHY THIS EXISTS, and it is not "pendulum but bigger". Two reasons, and the
    second is the one that forced it:

    1. **Pendulum saturates.** A competent reward solves swing-up in ~15-20k SAC
       steps, so on a fixed budget most methods pile up against the ceiling and
       stop being distinguishable. Acrobot is underactuated at the *shoulder* --
       the only motor is the elbow -- so the arm cannot be lifted directly at
       all. Energy has to be pumped through a second, coupled link, and a reward
       that merely says "be up" gives a learner almost nothing to climb. That is
       the regime where reward *design* is load-bearing rather than decorative.

    2. **`rda` and `gt_reward_design` score behaviour with a VLM watching
       rollouts**, and Pendulum gives a VLM nothing to look at: one rod on a
       plain field, where every frame differs from every other frame by an
       angle. Two jointed links produce visibly distinct *postures* -- folded,
       extended, swinging, inverted -- and a goal line the tip is either above
       or below. Those methods have not been meaningfully tested until they run
       somewhere a frame carries information, which is the whole content of
       Table 2 in the GT paper (trajectory+video 74.1% human agreement vs 63.0%
       for video alone).

    Dynamics are the standard Acrobot (Sutton & Barto; gymnasium's
    `acrobot.py`), integrated with RK4, with ONE deliberate change: the torque
    is **continuous** on [-1, 1] rather than gymnasium's three-way discrete
    choice. That keeps `train.algorithm: sac` -- the same learner the other
    continuous-control envs use -- so a fitness difference between methods is
    still attributable to the reward and not to a change of `A_M`.

    The second deliberate change: **no terminal state.** Gymnasium ends the
    episode the instant the tip crosses the line, which makes "got there" the
    only measurable thing. Here the episode always runs the full horizon and the
    metric is the fraction of it spent above the line, so *reaching* and
    *staying* are both scored. A reward that flails the tip through the goal
    once looks very different from one that balances above it, and on a
    two-link arm those are genuinely different behaviours.
    """

    name = "acrobot"
    obs_dim = 6
    action_dim = 1
    horizon = 300

    # -- constants, gymnasium/envs/classic_control/acrobot.py -----------------
    dt = 0.2
    link_length_1 = 1.0
    link_length_2 = 1.0
    link_mass_1 = 1.0
    link_mass_2 = 1.0
    link_com_1 = 0.5
    link_com_2 = 0.5
    link_moi = 1.0
    max_vel_1 = 4 * math.pi
    max_vel_2 = 9 * math.pi
    max_torque = 1.0
    g = 9.8

    #: Ground truth, and only ground truth. The tip is "up" when it is one full
    #: link above the shoulder -- gymnasium's own success check,
    #: `-cos(th1) - cos(th2 + th1) > 1.0`. Nothing in the observation announces
    #: this number, so a candidate cannot read the metric off the state, and
    #: `verify.forbidden_symbols` blocks it from reaching for the method.
    goal_height = 1.0

    #: Measured the same way `Pendulum.success_threshold` was, with this repo's own
    #: surrogate (`training._CEMLearner`, 60k env steps, greedy policy, 30
    #: evaluation episodes, mean over 3 training seeds). `task_metric` is the
    #: fraction of the episode with the tip above the bar.
    success_threshold = 0.3

    exact_states = None
    n_disc_states = 15 * 15 * 9 * 9

    _obs_low = np.array([-1.0, -1.0, -1.0, -1.0, -max_vel_1, -max_vel_2])
    _obs_high = np.array([1.0, 1.0, 1.0, 1.0, max_vel_1, max_vel_2])

    # The domain-randomisation surface -- link masses and lengths, the parameters a
    # sim-to-real story would actually be uncertain about -- is the spec's
    # `domain_randomization` block (`tasks/acrobot/shared_spec.yaml`), installed as
    # `dr_parameters` / `_dr_nominal` by `SpecEnvAdapter._apply_spec`. `EnvAdapter.set_dr`
    # and `_sample_dr` are the protocol and `_p` reads the per-episode draw, so the
    # dynamics actually see what DR samples (DrEureka's `[lo, hi]` ranges included).

    # The prompt-facing description is likewise the spec's `state_surface` and
    # `description.env_prose`, which `SpecEnvAdapter` applies at construction and
    # `tests/test_task_specs.py` holds to the observation layout. Nothing is declared
    # here beside the spec: two homes for one fact would put two answers in one prompt.

    # -- adapter surface -----------------------------------------------------

    def _build_action_set(self) -> Any:
        # A finite proxy for the box, used by the tabular/CEM learners only; the
        # sb3 path uses the Box directly. Odd count so zero torque exists -- the
        # arm cannot be *held* still without it.
        return np.array([[t] for t in np.linspace(-self.max_torque, self.max_torque, 5)])

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._obs_low.copy(), self._obs_high.copy()

    def _reset(self, rng: np.random.Generator) -> np.ndarray:
        # gymnasium seeds all four state variables in [-0.1, 0.1]: the arm
        # starts hanging almost straight down and almost still, so every
        # episode requires the full pump-up. Not randomised widely on purpose --
        # a start that is already halfway up would let a bad reward look good.
        st = rng.uniform(-0.1, 0.1, size=4)
        return self._obs(st)

    def _obs(self, st: np.ndarray) -> np.ndarray:
        th1, th2, d1, d2 = st
        return np.array([math.cos(th1), math.sin(th1),
                         math.cos(th2), math.sin(th2), d1, d2], dtype=float)

    @staticmethod
    def _state_of(obs: np.ndarray) -> np.ndarray:
        o = np.asarray(obs, dtype=float).ravel()
        return np.array([math.atan2(o[1], o[0]), math.atan2(o[3], o[2]), o[4], o[5]])

    def _p(self, key: str) -> float:
        """A physics constant under THIS episode's domain-randomisation draw.

        `_dr_now` is what `EnvAdapter.reset` sampled from the ranges `set_dr` installed
        (the spec's nominal values when none are); the class attribute is the fallback
        for an axis the spec does not randomise.
        """
        return float(self._dr_now.get(key, getattr(self, key)))

    def _dsdt(self, st: np.ndarray, torque: float) -> np.ndarray:
        """Acrobot equations of motion (Sutton & Barto's book version)."""
        m1, m2 = self._p("link_mass_1"), self._p("link_mass_2")
        l1 = self._p("link_length_1")
        lc1, lc2 = self.link_com_1, self.link_com_2
        i1 = i2 = self.link_moi
        g = self.g
        th1, th2, d1, d2 = st

        d_1 = (m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * math.cos(th2))
               + i1 + i2)
        d_2 = m2 * (lc2 ** 2 + l1 * lc2 * math.cos(th2)) + i2
        phi2 = m2 * lc2 * g * math.cos(th1 + th2 - math.pi / 2.0)
        phi1 = (-m2 * l1 * lc2 * d2 ** 2 * math.sin(th2)
                - 2 * m2 * l1 * lc2 * d2 * d1 * math.sin(th2)
                + (m1 * lc1 + m2 * l1) * g * math.cos(th1 - math.pi / 2.0) + phi2)
        dd2 = ((torque + d_2 / d_1 * phi1 - m2 * l1 * lc2 * d1 ** 2 * math.sin(th2) - phi2)
               / (m2 * lc2 ** 2 + i2 - d_2 ** 2 / d_1))
        dd1 = -(d_2 * dd2 + phi1) / d_1
        return np.array([d1, d2, dd1, dd2])

    def _step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        st = self._state_of(s)
        torque = float(np.clip(np.asarray(a, dtype=float).ravel()[0],
                               -self.max_torque, self.max_torque))
        # RK4 over one dt, as gymnasium does: Euler at dt=0.2 is unstable here.
        k1 = self._dsdt(st, torque)
        k2 = self._dsdt(st + 0.5 * self.dt * k1, torque)
        k3 = self._dsdt(st + 0.5 * self.dt * k2, torque)
        k4 = self._dsdt(st + self.dt * k3, torque)
        st = st + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        st[0] = _wrap(st[0])
        st[1] = _wrap(st[1])
        st[2] = float(np.clip(st[2], -self.max_vel_1, self.max_vel_1))
        st[3] = float(np.clip(st[3], -self.max_vel_2, self.max_vel_2))
        # No terminal state, deliberately -- see the class docstring. `done` is
        # always False and the horizon ends the episode, so `task_metric` can
        # score *staying* up rather than only *reaching* up.
        #
        # `success` is the per-step comparison `task_metric` averages -- the tip
        # above the bar in the arriving state -- so the flag and the metric cannot
        # disagree about a step. It is the event `train.bc_prior.accept: success`
        # keys a demonstration on; without it that rule would drop every acrobot
        # episode as a failure.
        obs = self._obs(st)
        tip = self.tip_height(obs)
        return obs, False, {"tip_height": tip, "success": bool(tip > self.goal_height)}

    def tip_height(self, obs: np.ndarray) -> float:
        """Height of the tip above the shoulder, in NOMINAL link-lengths. Ground truth.

        Reads the class attributes, never `_p`. This is what `task_metric`, `success()`
        and `reference_reward` score, and `training._evaluate_policy` calls it inside
        stage 3's `set_dr` window -- on a blob that under
        `train.domain_randomization.mode: generated` is LLM-authored. Read through the
        draw, `link_length_1: [1.15, 1.15]` lifts a nominal 0.87 over the 1.0 bar
        (measured +15% at both lengths 1.15): a candidate moving the ground truth by
        choosing its DR, which is the rule the co-designed observation is held to. At
        nominal this is gymnasium's dimensionless check `-cos(th1) - cos(th1 + th2)`;
        only `_dsdt` sees the drawn `link_length_1`.
        """
        o = np.asarray(obs, dtype=float).ravel()
        cos1, sin1, cos2, sin2 = o[0], o[1], o[2], o[3]
        # cos(th1 + th2) = cos1*cos2 - sin1*sin2
        return float(-self.link_length_1 * cos1
                     - self.link_length_2 * (cos1 * cos2 - sin1 * sin2))

    def discretise(self, s: np.ndarray) -> int:
        st = self._state_of(s)
        i = _bin(st[0], -math.pi, math.pi, 15)
        j = _bin(st[1], -math.pi, math.pi, 15)
        k = _bin(st[2], -self.max_vel_1, self.max_vel_1, 9)
        m = _bin(st[3], -self.max_vel_2, self.max_vel_2, 9)
        return ((i * 15 + j) * 9 + k) * 9 + m

    def random_state(self, rng: np.random.Generator) -> np.ndarray:
        st = np.array([rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi),
                       rng.uniform(-self.max_vel_1, self.max_vel_1),
                       rng.uniform(-self.max_vel_2, self.max_vel_2)])
        return self._obs(st)

    def task_metric(self, traj: Any) -> float:
        """Fraction of the episode with the tip above the bar. Ground truth.

        Not "did it ever reach the goal": on a two-link arm a reward that
        whips the tip through the goal region once and loses control looks
        identical to one that balances there, under a reach-once metric, and
        they are not the same behaviour. Fraction-of-time separates them and is
        already in [0, 1], so no normalisation is needed downstream.
        """
        states = _states_of(traj)
        if states is None or len(states) == 0:
            return 0.0
        above = [1.0 if self.tip_height(s) > self.goal_height else 0.0 for s in states]
        return float(np.mean(above))

    def reference_reward(self, s: np.ndarray, a: Optional[np.ndarray] = None) -> float:
        """The hand-written baseline a candidate is measured against.

        Deliberately NOT gymnasium's own reward, which is -1 per step until
        termination: that is a pure sparse signal, it is what makes Acrobot hard
        for RL rather than interesting for reward design, and a candidate that
        merely reproduced it would learn nothing on this budget. This is instead
        what a competent human would write -- pay for tip height, damp the
        thrashing that height alone encourages, and charge a little for torque.
        """
        obs = np.asarray(s, dtype=float).ravel()
        st = self._state_of(obs)
        torque = 0.0 if a is None else float(np.asarray(a, dtype=float).ravel()[0])
        height = self.tip_height(obs)
        spin = 0.02 * (st[2] ** 2 + st[3] ** 2)
        return float(height - spin - 0.01 * torque ** 2)

    # -- what the task asks, for a frame legend (`EnvAdapter.legend_lines`) ----

    def legend_lines(self, state: np.ndarray) -> List[str]:
        """Two lines: the ask, and where the bar is. Both constant across an episode.

        The tip's height is `tip_height(state)` and is deliberately absent: beside
        `goal_height` it is the per-step verdict in link-lengths, on the frames `rda`
        and `gt_reward_design` grade with a VLM -- the class docstring's second reason
        for this env existing at all (`EnvAdapter.legend_lines` has the rule). The bar
        is the ask, `render` already draws it, and the line is built from
        `goal_height` so the number on the frame is the number `task_metric` measures
        to. "One link-length above the shoulder" is `tasks/acrobot/shared_spec.yaml`'s
        own wording of it.

        `legend_on_frame` stays False: `render` draws no text. `step_success_is_a_check`
        stays at its default: `_step` emits `success` -- the tip
        above the bar on that step, the comparison `task_metric` averages -- so a
        composer colouring a timeline by that flag reads a real per-step check here.
        """
        return ["SWING TIP ABOVE BAR, STAY",
                f"BAR: {self.goal_height:.1f} LINK ABOVE SHOULDER"]

    # -- rendering -----------------------------------------------------------

    def render(self, state: np.ndarray, width: int = 320) -> np.ndarray:
        """One RGB frame. The goal line is DRAWN, not implied.

        This env exists partly so a VLM has something to read (see the class
        docstring), and "is the tip above the bar" is a question a VLM answers
        reliably from a frame with the bar in it and unreliably from one
        without. The two links are drawn in different shades so a folded arm is
        distinguishable from an extended one, which is the posture information
        Pendulum cannot carry.
        """
        w = max(48, int(width))
        img = np.full((w, w, 3), _BG, dtype=np.uint8)
        cx = (w - 1) / 2.0
        cy = 0.62 * w                      # shoulder low in frame: the arm swings UP
        scale = 0.21 * w                   # one link-length in pixels
        rod_half = max(1, int(round(0.013 * w)))

        st = self._state_of(state)
        th1, th2 = st[0], st[1]
        # Nominal geometry, as `tip_height` reads it: the frame is the judge's view of the
        # ground truth, and a link drawn at its DR length against a bar at `goal_height`
        # would show a tip above the bar that the metric scores below it.
        l1, l2 = self.link_length_1, self.link_length_2

        # Reachable envelope, then the goal line, then the arm: painter's order.
        _ring(img, cx, cy, (l1 + l2) * scale, 0, _ORBIT)
        goal_y = cy - self.goal_height * scale
        _segment(img, (cx - 0.42 * w, goal_y), (cx + 0.42 * w, goal_y),
                 max(1, int(round(0.006 * w))), _TARGET)

        elbow = (cx + l1 * scale * math.sin(th1), cy + l1 * scale * math.cos(th1))
        tip = (elbow[0] + l2 * scale * math.sin(th1 + th2),
               elbow[1] + l2 * scale * math.cos(th1 + th2))
        _segment(img, (cx, cy), elbow, rod_half, _ROD)
        _segment(img, elbow, tip, rod_half, _LINK2)
        _disc(img, elbow[0], elbow[1], max(2.0, 0.026 * w), _PIVOT)
        _disc(img, tip[0], tip[1], max(2.0, 0.040 * w), _BOB)
        _disc(img, cx, cy, max(2.0, 0.020 * w), _PIVOT)
        return img


@register("env", "acrobot")
def acrobot(ctx: Any) -> EnvAdapter:
    """Two-link underactuated swing-up: the env a VLM can actually read (§0)."""
    return Acrobot()
