# Where these files come from

`tasks/` is **owned by this repository**. Every spec here, and the schema
(`shared_spec.schema.json`), is this repo's to edit: there is no upstream catalogue to defer to
and no drift to check for. `SCHEMA_DELTA.md` explains the fields this repository added to the
schema and why.

## Provenance paths

`provenance.authored_from[].path` is relative to **whatever was read**, and most of what was
read is not in this repo: `humanoid_bench/`, `metaworld/` and `gymnasium/` are installed
packages (their versions are pinned per spec in `env.library`), and `refs/` holds the papers
and released code a spec cites.

**The rule: qualify a path whose bare form collides with a real path in THIS repo.** Upstream
package paths are left bare because no such collision exists.

## Both reductions of an anchor come from the same rollouts

`scripts/measure_anchors.py --n 30` (seed 0) measures **both reductions from the same
rollouts**, and both pairs in every Meta-World spec come from that one run. That is not a
convenience. One success check under two reductions can only be guaranteed consistent if both
come from the same rollouts: a random anchor measured at 0.151 per-step on one set of goal
draws and 0.083 any-step on another would claim a policy inside the goal region for 15% of an
episode's steps and yet inside it on none of them.

n=2 is not a measurement either: re-measuring a two-episode figure moved `peg_insert_side`'s
per-step expert anchor from 0.823 to 0.535 (−35%) and `drawer_close`'s random anchor from 0.239
to 0.151 (−37%).

## Known defects

**Two `saturates_at` fields, one level apart, whose nulls mean OPPOSITE things.**
`continuous_success.raw.saturates_at` null means *the metric is genuinely unbounded above*;
`budget.saturates_at` null means *nobody measured where it saturates*. No code here reads
either — both are recorded evidence, not consumed fields — so this is a trap for a future
consumer rather than a live bug.

**The ten gym `anchors.random` were measured under two different mujoco versions** while
every one of those specs states 3.3.0 uniformly in `env.library.stack`. The split is 4/6,
verified by reproduction — `half_cheetah`'s committed −4.5772 reproduces to every recorded
digit under 3.3.0 and is not close under 3.11.0:

| engine | specs |
|---|---|
| 3.3.0 | `half_cheetah`, `reacher_hold`, `hopper_hop_in_place`, `swimmer_heading` |
| 3.11.0 | `half_cheetah_backward`, `half_cheetah_target_speed`, `hopper_hop`, `inverted_pendulum_balance`, `reacher_reach`, `swimmer_forward` |

And the split leaves **two specs that cannot both be true**: `half_cheetah_backward`'s metric
is the exact negation of `half_cheetah`'s, so their random anchors must negate. Committed they
are **−4.5772 and +4.46** — and that +4.46 is 3.11.0's −4.4554 negated. Neither number is
individually wrong and the pair is still contradictory. Re-measure both under one engine before
normalising against either.

All ten of those specs have an adapter (`bird/envs/gym_mujoco.py`), so the defects are LIVE;
that module's docstring repeats them for whoever reads the adapter.

### Why the per-step anchors exercise the normalisation arithmetic

Under the any-step reduction, seven of the ten computable Meta-World tasks have
`random = 0.0` and `expert = 1.0`, so `clip((raw − random)/(expert − random), 0, 1)` is the
**identity** and a bug in the subtraction or the divisor would pass unnoticed. The
`per_step_fraction` pair does not have that problem (n=30, seed 0):

| | random | expert | exercises |
|---|---|---|---|
| `window_open` | 0.0 | 0.8313 | scale |
| `peg_insert_side` | 0.0 | 0.5347 | scale |
| `door_open` | 0.0 | 0.0598 | scale, and a small divisor |
| `drawer_close` | 0.1511 | 0.8454 | shift **and** scale |
