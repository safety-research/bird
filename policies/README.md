# policies/ — the scripted-policy registry

One directory per campaign (a group of policies that share code), holding the policy code, its
tuned constants, its per-seed evaluation records, and a `policies.yaml` manifest validated by
`policies.schema.json`. `bird/policies.py` loads the manifests; `bird/policy_api.py` runs the
policies.

| directory | what it holds | used by |
|---|---|---|
| `mt10_suite`, `mt10_push`, `mt10_peg_insert` | hand-written waypoint controllers for the ten Meta-World MT10 tasks | the demonstration verifier (`bird/demos.py`), which rolls out a task's solved policy as its expert |
| `simple_suite` | closed-loop controllers for BIRD's pure-numpy tasks (pendulum, acrobot, toy_*) | offline tests and demonstrations |
| `h1hand_hb_hacks` | deliberate HumanoidBench reward hacks: scripted policies that clear the benchmark's walk and crawl success bars without walking or crawling. `moonhop` and `crawl_rock` are the two in Section 4.4 of the paper; `hop` is a supplementary example | the paper's reference-reward analysis |

## The call pattern

`bird/policy_api.py` is the uniform surface:

```python
from bird.policy_api import load_policy
pol = load_policy("mt10_suite/reach")        # id from a policies.yaml
env = registry.get("env", pol.env_id)(None)  # the adapter is the instrument of record
s = env.reset(rng); pol.reset(rng)
a = pol.act(s, t=t, env=env)                 # ALWAYS the adapter's normalised units
```

Two wrappers adapt the two code shapes onto that one signature, without touching the
original code:

| pattern | code shape | wrapper does |
|---|---|---|
| `closure` | `make_policy(p) -> act(s) -> a` (MT10) | passes `obs` through; `t`/`env` unused |
| `rig` | `make_x(**c) -> fn(rig, t, ctrl) -> ctrl` (humanoid) | builds the Rig over `env._env`, threads `ctrl`, normalises raw actuator units against `actuator_ctrlrange` once, in one place |

A campaign's `entry.py` beside the controller files is where rig wiring lives.

`scripts/eval_policy.py --policy <id> --seeds 0-24` is the single evaluation entry point;
it writes the standardized per-seed record shape, so a score enters a manifest by being
measured, never by being typed. That shape is `bird-policy-eval-v2`
(`policies/records.schema.json`), a superset of the v1 records already in the tree: every
v1 key keeps its name and meaning, and it adds the seed-set digest, the engine version,
the record's provenance (`measured`), and the `horizon` every row ran to with
the `--max-steps` cap when one was set (null otherwise) -- so a shortened measurement is
never mistaken for a full one, and a manifest may not cite one. Those two keys are
**optional in the schema and always present from the writer**: older v2 records predate
them, a committed record is never rewritten to fit a schema, and their absence means
exactly "uncapped as far as the record knows" -- `tests/test_policy_records_corpus.py`
validates every committed v2 record against `records.schema.json` so the schema cannot
refuse the records the manifests rest on. **Two records written the same day are two
objects** — the filename carries `HHMMSS` and a digest of the record itself — because a
date-only name overwrites in place. `--root <dir>` measures a policies directory other than
this one (the tests use it for a synthetic campaign in a temporary directory); git cannot
describe a tree outside the checkout, so such a record carries `tree_dirty: null` — never
`false`, which would claim a tree nobody looked at was clean.

`commit` in a record and `verified_at_commit` in a manifest name commits of the development
repository in which the records were measured; they are provenance, not commits of this
release.

## Rules

- **Code files are kept as they were measured.** The wrappers adapt; do not "clean up" a
  controller — the numbers in the manifests were measured on these bytes.
- **Separate runtime families, no shared interpreter.** `metaworld` pins `mujoco==3.3.0`,
  HumanoidBench needs `3.1.6` (`pyproject.toml` §metaworld, `scripts/setup_humanoid.sh`).
  `family:` in the manifest is which venv a policy runs in. `bird/policies.py` (the loader)
  imports neither. `bird_control` is this repo's own pure-numpy tier
  (`bird/envs/control.py`, `bird/envs/toy.py`: pendulum, acrobot, toy_*): no external
  runtime, so it runs in every venv (`policies/simple_suite`).
- **The adapter is the instrument of record.** The BIRD adapters zero `qacc_warmstart`
  every step; a rig that zeroes it at reset only measures something slightly different, so
  a rig is always built over the adapter's own env.
- **`score.value` is the standard env under the standard reset.** Any other number goes
  in `score.secondary`, named.
- **One unit.** Every `score.value` is `bird_task_metric`, the task spec's own metric as the
  adapter scores it; the benchmark's own episode return, where one exists, is a different
  quantity and is recorded beside it (`per_seed_hb_return`, `secondary.hb_return_mean`).
- **These scores are NOT the task anchors.** `tasks/<id>/shared_spec.yaml → anchors` feeds
  fitness normalisation through `baselines_of()` (`bird/envs/spec.py`); this directory
  deliberately does not. Overwriting MT10's bundled-expert anchors with these (higher)
  numbers would rescale every run on the tier.
- **Do not show these to the reward-designing LLM.** `description.l_task` in a task spec
  is the treatment; a solution script in the prompt contaminates the experiment. Nothing
  in `bird/` may feed `policies/` content into a generate-stage prompt.
- **Trust model.** `entry.file`/`entry.symbol` are resolved by dynamic import, which is
  safe here for one reason: `policies/` is a committed repo artifact, the same trust level
  as `bird/` itself. The loader refuses any path that escapes `policies/` and never reads
  a manifest out of a run directory (run directories on a shared mount may be
  world-writable — the `checkpoint._DECODABLE` reasoning).

## Adding a policy

1. Copy the policy code, its constants and its per-seed records into a new directory.
2. Write the `policies.yaml` entry (schema: `policies.schema.json`; the score must be
   re-derivable from the committed record).
3. Run `uv run python3 -m pytest tests/test_policy_records_corpus.py tests/test_eval_policy_record.py`.
4. **HumanoidBench policies: record the benchmark's return too.** `eval_policy.py` writes
   `reference_return` per row -- the env's own reward summed over the episode, computed AFTER
   the rollout from the recorded states and actions (calling `reference_reward` between two
   steps perturbs the next one) -- and `summary.reference_return_mean`. Copy the mean into the
   entry's `score.secondary.hb_return_mean` and name the record in `hb_return_record`
   (`"<path> -- what was summed"`); that key is where a reader finds the task's own metric
   when the headline is `bird_task_metric`.

**An entry is a claim.** Registering a policy asserts that
`load_policy(id)` works today and that `scripts/eval_policy.py` has reproduced
the recorded score through it on a machine with the family runtime.

**`hb_return_mean` comes from the score record.** On a HumanoidBench entry,
`score.secondary.hb_return_mean` is the score record's OWN `summary.reference_return_mean`
-- the shipped HumanoidBench reward summed per episode and averaged over the same seeds,
which `scripts/eval_policy.py` writes into every record -- and `hb_return_record` names that
record. It is never taken from a separate run: a second rollout describes different
trajectories the moment anything moves a row (a reset fix, a re-measure, a constant), and
the pair would then be two numbers about two policies wearing one entry.
