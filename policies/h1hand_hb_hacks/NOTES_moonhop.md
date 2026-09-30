# Moonhop: hopping backwards to HB >= 700 on `h1hand-walk-v0`

Registered as `h1hand_hb_hacks/moonhop`. The gym-env numbers below were measured during
development; the registered score is the BIRD adapter's (`records/eval_moonhop_*`, see
`policies.yaml`).

**Instrument, everywhere below:** the sum of the UPSTREAM `env.step` reward over 1000 steps.
That is humanoid_bench @ cb11890 via `gym.make("h1hand-walk-v0")`, with no set_state and no
per-step warmstart zeroing, driven through `hk.Rig`. "Held-out" means seeds the row's search
never saw, run in fresh processes.

## Result

**The moonhop passes, and it is more absurd than a hop.** The robot hops round 180 deg on the
spot, then hops BACKWARD for the rest of the episode, so it never faces where it is going. HB's
move term reads the WORLD-frame COM x velocity and never asks which way the robot faces.

* **`moon.py` + `moon_t15.json`.** On fresh held-out seeds 90-149 it averaged **790.8** and
  passed **59/60**. Its one failure is seed 106, which fell at step 264.
  * On the 59 passing seeds, the returns run 799-810.
  * The turn completes at step ~158.
  * After the turn it moves +x at **1.23 m/s**, facing within 0.5 rad of -x on >= 98.6 % of
    steps, and covers ~20.8 m.
  * The reward components are standing 0.988, upright 1.0 and move 0.831.
* In a rendered rollout (seed 90, return 801.9) the arms point the other way from about step
  200 on, while x keeps increasing.
* **How `moon_t15` was chosen.** An 11-knob CEM on seeds 0-1 gave constants that turned at
  0.79 s; `moon_t15` is those constants with the turn start moved to 1.5 s. That setting came
  from a turn-start grid run on 12 seeds the CEM constants fell on plus 8 they passed. For
  `moon_t15`, then, 90-149 is the only clean held-out set.
* **The CEM constants themselves** (turn at 0.79 s) scored 711.0 on 10-29 (17/20) and 711.0 on
  30-89 (51/60), then 671.4 on 90-149 (48/60). All their failures were falls during or just
  after the turn, at steps 124-264, because the turn began while the hopper was still spinning
  up.

## Loophole and artifact checks on the walk reward (all negative, measured)

| check | result | why it cannot pay |
|---|---|---|
| sensor `pelvis_subtreelinvel` vs momentum/mass (`subtree_linvel`) | identical | pelvis is the root body, so internal motion (flailing arms or fingers, joint-limit slams) cannot change COM velocity; only floor friction can |
| per-step SAMPLE vs COM motion over that step (substep replay) | abs diff 0.0047 m/s mean, 0.021 max; mean clip 0.41466 sampled vs 0.41450 finite-difference | the 50 Hz sample aliases nothing, and a 50 Hz COM swing is friction-capped (sum of abs F_x <= mu * sum of N, N averages mg) at ~0.1 m/s peak-to-peak |
| 25 Hz joint chatter, ankle/knee/hip +-0.1/0.3/0.6 rad | no drift, falls in 97-147 steps like the bare keyframe | |
| vibration scoot (bristlebot): vertical bounce (grid 3-8 Hz, CEM 1-12 Hz) (N swings 250-870 N) + in-phase fore-aft foot stroke, feet never lifted | 48-point grid: dx <= 0.058 m in 10 s at every setting; 16-gen CEM: no candidate kept the feet down (>= 95 % of steps) AND scored above standing | at mu = 1 a loaded sole does not slip; unloading it enough to slide means leaving the floor |
| double-support scissor shuffle | 0.12 m commanded stroke moves the ankles < 5 mm | same friction lock |
| rock in place, feet planted | ~183 (standing still ~164) | ZMP: a 30 cm sole caps in-place COM speed amplitude at ~0.2 m/s |
| head height | `head` is a site 0.70 m up the torso z axis; torso yaw does not move it; standing is already 0.99 on every gait | |
| termination | only pelvis z < 0.2; a fallen robot earns ~0 | |
| small_control | 0.97-0.98 on every gait (42/61 actuators are ~zero-force hand joints) | so per-step reward ~= 0.97 * move, and move is the only lever |
| one-sided move, net displacement 0 (hop 6 m out, hop back) | 427.6 on its 13/20 surviving held-out episodes, abs(dx) <= 0.52 m; 372.6 mean over 20 | capped near 0.5/step by (backward speed)/(forward+backward) with measured hop speeds; the predicted bound holds |
| jitterbug (hop +-0.8 m back and forth) | 354 (train seeds) | reversals cost too much time |

The only exploitable properties of the reward are its indifference to gait and heading, and
the one-sidedness of `move`.

## Caveats

* **The moonhop is built on the hopper.** `snap_hop3.py` is a snapshot of `hop3.py`, the
  predecessor of the hop's `hop4.py`, and the moonhop runs it with re-tuned knobs
  (`snap_hop3_cem_a.json` overlaid by `gallop_best_d0.json`, feet together).
  * **The moonhop's own code** only biases the hopper's yaw servo toward a slewed heading
    reference 0 -> pi and then commands v_des < 0 in the reversed heading frame. The CEM tuned
    11 knobs on seeds 0-1, with a penalty unless the robot faces within 0.5 rad of backward on
    >= 95 % of post-turn steps.
* **Floor:** the rendered floor ends at x = 20 m, and the moonhop passes it (~21 m). A MuJoCo
  plane collides everywhere; only its drawing is finite, so a render's last frames show the
  robot past the drawn edge.
* **Moonhop robustness came from a grid, not a CEM.** A min-objective CEM on seeds 0, 1, 13
  and 38 diverged: seed 38 fell at step ~131 in every candidate it tried, so the min carried
  no gradient. It was stopped at generation 5. The fix that worked was the turn-start grid
  above.
