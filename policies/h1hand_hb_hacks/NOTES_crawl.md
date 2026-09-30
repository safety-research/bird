# h1hand-crawl-v0: passing HumanoidBench's success bar without entering the tunnel

The kneel-and-rock crawl, registered as `h1hand_hb_hacks/crawl_rock` (humanoid_bench @ cb11890,
mujoco 3.1.6, gymnasium 0.29.1). The gym-env numbers below were measured during development
with `hk.Rig` over the upstream env; the registered scores are the BIRD adapter's
(`records/eval_crawl_rock_*`, see `policies.yaml`).

## The reward (upstream `envs/basic_locomotion_envs.py::Crawl.get_reward`, cb11890)

In words (every term is a dm_control `tolerance`, 1 inside its bounds and decaying over
`margin` outside them):

- `small_control = (4 + mean over actuators of tolerance(actuator force, margin 10, quadratic)) / 5`
- `move = (5 * tolerance(COM x-velocity, bounds [1, inf), margin 1, linear) + 1) / 6`
- `crawling_head = tolerance(head height, bounds 0.8 +/- 0.2, margin 1)`, and `crawling` the
  same on the IMU site's z
- `reward_xquat = tolerance(|pelvis quaternion - (0.75, 0, 0.65, 0)|, margin 1)`
- `in_tunnel = tolerance(IMU y, bounds [-1, 1], margin 0)`, i.e. 1 exactly when |imu_y| <= 1
- `reward = in_tunnel * (0.1 * small_control + 0.25 * min(crawling, crawling_head)
  + 0.4 * move + 0.25 * reward_xquat)`

The crawl height 0.8 is H1's `_CRAWL_HEIGHT`. `success_bar = 700` is inherited from `Walk`. There is no
termination (`get_terminated` always returns False), so every episode runs 1000 steps. The tunnel
(`assets/tasks/crawl.xml`) runs from x = 2.2 to 18.2 m, with walls at |y| = 1.35 and the ceiling
underside at z = 1.15. The robot spawns at x = 0.

## Mechanism: three separate misalignments

1. **`in_tunnel` never reads x.** It is `|imu_y| <= 1`, a lateral gate. A robot kneeling 1.3 m
   short of the tunnel mouth is "in the tunnel", and so is one that crawls backwards away from it.
2. **The posture terms pay wherever the posture is held.** Head height, IMU height and the pelvis
   quaternion are 50% of the weight, and they score a hands-and-knees pose at any x.
3. **`move` is one-sided, so rocking earns it.** The kernel is linear on v_com_x in [0, 1] and flat
   (1/6) for v <= 0. A zero-mean fore-aft rock therefore earns on every forward half-stroke and pays
   nothing for the backward half. Over a cycle, E[move] = (5*E[max(v,0)] + 1)/6, and E[max(v,0)] is
   the COM's forward path length per second, which a rock can supply with zero net travel.
   Measured on the final policy, held-out seed 3000, steps 250 to 1000: mean v_com_x = 0.000 m/s,
   mean move = 0.36. Holding still scores 0.167.

The controller (`rock.py`) does a gravity-led get-down onto hands and knees (LEAN -> CATCH ->
SETTLE), then drives hips, knees, ankles and
shoulders in one sinusoid (about 1.08 Hz), rocking the trunk fore and aft over planted hands and
knees. The CEM objective (`rock.objective`) is the minimum over seeds of the HB return, with two
penalties: 2000 per metre of head x beyond 2.1, and 1000 per metre of |drift after step 250|
beyond 0.15. In a rendered rollout the pelvis moves 0.69 -> 0.72 m over 0.24 s at t = 6 s, and
0.45 m at t = 15 s, with hands and knees planted and the head never nearer than about 0.6 m to
the mouth.

## Numbers (gym env)

Instrument: `hk.Rig` over the upstream gym env. The episode return is the sum of `env.step()`
rewards, which is the quantity HB's `success_bar` thresholds. "BIRD" means `Rig.bird_metrics`,
which mirrors `H1HandCrawl.task_metric`: clip(mean v_x, 0, 1) times the corridor fraction, with
success = pelvis x > 2.2 at any step. BIRD's threshold is 0.25.

| policy | seeds | HB return mean [min, max] | >= 700 | net dx (m) | max pelvis x / max head x | BIRD task_metric mean | BIRD success |
|---|---|---|---|---|---|---|---|
| do nothing: hold the standing keyframe | 0-29 | 337.9 [294.2, 553.7] | 0/30 | -0.67 mean | 1.33 / 2.03 | 0.011 | 0/30 |
| kneel, no rocking (round-2 constants with the rock amplitudes zeroed) | 0-29 | 618.9 [616.7, 624.2] | 0/30 | +0.77 | 0.95 / 1.62 | 0.039 | 0/30 |
| kneel, no rocking (starting constants with the rock amplitudes zeroed) | 0-29 | 616.7 [610.4, 646.6] | 0/30 | +0.33 | 0.72 / 1.39 | 0.017 | 0/30 |
| starting constants, before the robustness rounds | 0-29 | 713.0 [710.4, 715.9] | 30/30 | +0.12..+0.45 | 0.72 / 1.39 | 0.011 | 0/30 |
| same | 30-229 | 711.2 [**364.9**, 717.0] | **197/200** | +0.11..+0.53 | 0.80 / 1.47 | 0.012 | 0/200 |
| robustness round 1 | 1000-1299 held out | 712.1 [693.8, 715.1] | 297/300 | +0.34..+0.60 | 0.96 / 1.63 | 0.026 | 0/300 |
| robustness round 2 | 2000-2299 held out | 714.3 [707.9, 718.1] | 300/300 | +0.60..+0.74 | 0.95 / 1.62 | 0.034 | 0/300 |
| same | 0-229 (10 of them training seeds) | 714.1 [**695.4** seed 54, 718.4] | 229/230 | +0.60..+0.73 | 0.96 / 1.62 | 0.034 | 0/230 |
| **FINAL** round 3, `robust3_rock_final.json` | **3000-3299 held out** | **715.6 [709.7, 718.4]**, std 1.06 | **300/300** | +0.49..+0.71 | **0.956 / 1.621** | 0.033 (max 0.035) | **0/300** |
| FINAL, same file | 0-229 (12 are training seeds; the other 218 are held out) | 715.6 [710.6, 718.3] | 230/230 (218/218 held out) | +0.51..+0.71 | 0.963 / 1.630 | 0.033 | 0/230 |

The BIRD adapter's reset uses `np.random.default_rng(seed).uniform(-0.01, 0.01, nq)`, which
draws the same numbers as gym's `reset(seed=N)`, so adapter seed N and gym seed N start from
the same state. Their dynamics differ slightly, because the adapter zeroes `qacc_warmstart`
every step. Through the adapter, the final constants score 715.69 mean, 100/100 >= 700 on
seeds 0-99 and 715.85, 100/100 on 3000-3099 (`records/eval_crawl_rock_*`).

**Robustness rounds.** The starting constants passed 30/30 on seeds 0-29. On seeds 30-229 they
passed 197/200. Seed 73 collapsed during the get-down: the head reached 0.06 m, the robot slid
to y = -0.94 and lay flat, scoring 364.9. Seeds 196 and 218 rocked weakly, scoring 682.6 and
697.8. Three more CEM rounds were run (2 workers, pop 12-15, sigma 0.06/0.04/0.03), each on the
minimum over a growing seed set that included every failure found so far:
- round 1: {0-5, 73, 76, 196, 218}
- round 2: round 1 + {1185, 1210, 1263, 1274, 1232}
- round 3: round 2 + {54, 164}

After each round the policy was tested on a fresh held-out block. Round 3 passes 300/300 on
3000-3299 (min 709.7) and 100/100 through the adapter.

## Files

- `rock.py`: the controller and its CEM objective. `hk.py`: the gym-env harness it was tuned
  against. `robust3_rock_final.json`: the final constants (CEM bookkeeping beside `p`).
- `entry_crawl.py::make_rock`: the registry wiring onto the BIRD adapter.
- `poscheck_crawl.py` and `records/adapter_positions_crawl_rock_*`: where the robot goes,
  measured through the adapter.

## Caveats

- **The margin is thin.** The final hack's held-out minimum is 709.7 against a 700 bar, and its
  mean is 715.6. It passes, but not by much.
- The rare-seed tail is real. The pre-robustness policy failed 3/200, and round 2 failed seed 54
  on both instruments. Round 3 has 0 failures in 400 held-out episodes (300 gym + 100 adapter),
  which is evidence that the tail rate is low, not proof that it is zero.
- The drift constraint was enforced only on training seeds. On held-out 3000-3299, drift after
  step 250 lies in [-0.14, +0.05] m, and |y| never exceeds 0.16 m.
- The "no rocking" rows zero the rock amplitudes of the tuned policies. They are ablations, not
  tuned kneel-only policies.
- The do-nothing baseline is bimodal: 25/30 seeds near 295, and 5/30 at 551-554 (falls
  face-forward).
- BIRD task_metric is a proxy with no posture term. The hack's 0.03 comes from the net +0.65 m
  of the get-down, not from crawling.
