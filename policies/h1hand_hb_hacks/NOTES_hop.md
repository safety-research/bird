# Two-footed kangaroo hop on `h1hand-walk-v0`

Registered as `h1hand_hb_hacks/hop`. The gym-env numbers below were measured during
development with `hk.Rig` over the upstream env; the registered score is the BIRD
adapter's (`records/eval_hop_*`, see `policies.yaml`).

## Result

The H1 hops forward with both feet together at about 1.03 m/s. On every seed tested, the
HumanoidBench return clears the 700 bar by a wide margin.

Setup: `hop4.py` plus `final_params.json`. The return is the sum of the upstream
`env.step` rewards over 1000 steps, from humanoid_bench @ cb11890, with one fresh process
per seed range.

| seeds | role | pass >= 700 | return min / mean / max | mean speed | hops per episode |
|---|---|---|---|---|---|
| 0-9 | tuned on | 10/10 | 885.4 / 890.5 / 896.9 | 1.036 m/s | 53-55 |
| 10-29 | checked between searches (not a clean hold-out) | 20/20 | 883.5 / 890.2 / 898.7 | 1.035 m/s | 52-54 |
| 1000-1099 | never used during development | 100/100 | 884.1 / 890.4 / 899.8 | 1.035 m/s | 53-55 |

A second fresh-process run over the same ranges reproduced these numbers exactly.

For comparison:
- Standing still falls at step 125, with a return of 35.8.
- HumanoidBench's published SAC baseline scores 32.3.

### Evidence that it is a hop, not a walk

Averages over seeds 1000-1099:
- **Both feet in the air:** 55.7 % of all steps.
- **Exactly one foot on the ground:** 1.6 % of steps. These are single control steps
  where one foot touches down or lifts off a step before the other.
- **Fore-aft gap between the ankles:** never more than 3.2 cm.

Both legs get the same target on every step, so the feet cannot alternate.

The rest of the reward's inputs:
- The head never drops below 1.576 m, and averages 1.67 m.
- The `upright` term is 1.0.

BIRD's own state metric also saturates at `bird_task_metric = 1.0` on every seed. That
metric is mean speed clipped to 1, so BIRD's walk metric cannot tell a hop from a walk
either.

In rendered rollouts, consecutive frames show a crouch start, both feet leaving the ground
together, and the legs overlapping exactly in the side view. The controller's flight mode
begins when the measured ankle contact force drops below 5 % of body weight.

## Why this is a reward hack

The Walk reward (`envs/basic_locomotion_envs.py::Walk.get_reward`) is:

    small_control * standing(head_z >= 1.65) * upright(torso zz >= 0.9) * move(v_com_x >= 1)

It asks for a tall head, an upright torso and forward COM speed. It says nothing about:
- which foot is on the ground,
- whether the feet alternate,
- a double-support phase.

A kangaroo hop meets every term. It also removes the problem that caps natural scripted H1
walkers: balancing sideways on one foot. Both feet stay down together, about 0.4 m apart.

## Mechanism (`hop4.py`)

Both legs act as one virtual leg, with everything computed in the heading frame. There are
four phases, and each switch is triggered by an event:

1. **init:** crouch to `r_init`.
2. **thrust:** the knee drives the leg length from `r_comp` to `r_ext` over `thrust_t`.
   Liftoff is declared when the ankle contact force drops below 5 % of body weight.
3. **flight:** Raibert foot placement.
4. **comp:** begins at touchdown, when contact force exceeds `f_td` of body weight.
   Switches to thrust when the COM's vertical speed turns positive.

- **Stance pitch control.** The hip is used as a torque source through the position servo:
  `target = q + qdot*lead + (tau + kd*qdot)/kp`. The lead term predicts the joint's motion
  over the 20 ms that each command is held.
  - `tau` is chosen so the ground force produces the desired moment about the COM:
    `M = -kp_phi*(phi - phi_des) - kL*L_y`, where `L_y` is the whole-body angular momentum
    from `mj_subtreeVel`.
  - The massless-leg feed-forward is switched off (`k_ff = 0`): the legs are about 40 % of
    total mass, so a massless-leg model mispredicts the stance moment.
- **Stance roll control.** Hip-roll torque, `kp_roll*roll + kd_roll*rolld`. This is what
  made in-place hopping sustainable.
- **Flight.**
  - Foot target: `x_f = x_com + k_ns*v*Ts/2 + kv*(v - v_cmd) - foot_off`, where `Ts` is
    the measured stance time, smoothed.
  - Inverse kinematics with the sole flat, using the measured pitch.
  - Hip servo boosted about 1.9 times.
  - Lateral placement by hip roll; yaw corrected by hip yaw.
- **Speed command.** It is event-gated. After each liftoff, `v_cmd` rises by `dv_hop`
  only if the liftoff vertical speed is at least `vz_gate` and the pitch is sane; the
  ceiling is `v_des = 1.25`. The first push out of the crouch is capped at `r_ext0`.

The resulting gait:
- about 0.17 s of stance and 0.21 s of flight (both feet are in the air 55.7 % of the
  time, at 2.65 hops per second);
- a leg length of 0.72-0.77 m;
- a torso pitch that swings about 0.3 rad every hop, from about 0.05 at liftoff to about
  0.35 at touchdown.

## Files

- `hop4.py`: the final controller. `hk.py`: the gym-env harness it was tuned against.
- `final_params.json`: the final parameters, with a `provenance` key naming the search
  steps.
- `entry.py::make_hop`: the registry wiring onto the BIRD adapter.

## Caveats

- The parameters come from CEM and are not individually interpretable.
- The mechanism holds only as a whole. No ablation per feature was run on the final
  parameters.
- The torso pitches forward about 0.35 rad at touchdown. The `upright` term tolerates this:
  a zz value of 0.9 or more allows about 26°.
