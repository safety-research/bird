# `tasks/` — the base task definition

One task, defined once, in a file. Without it a BIRD task would be two config strings
(`problem.env_id`, `problem.task_description`) plus whatever an `EnvAdapter` subclass happened
to declare as Python class attributes, with the same facts free to disagree between the
adapter, the config and the prose.

The schema is `shared_spec.schema.json` (JSON Schema 2020-12, `additionalProperties: false`
throughout), owned here. `SOURCE.md` records where the specs' facts and anchors come from and
their known defects; `SCHEMA_DELTA.md` lists the fields BIRD added to the schema and why.

```
tasks/
  shared_spec.schema.json     the contract
  <task_id>/shared_spec.yaml  one task
  _no_adapter.json            specs BIRD cannot run, each with a reason
  _anchors_<date>*.json       raw anchor measurements (`scripts/measure_anchors.py`)
  SOURCE.md                   provenance rules and known anchor defects
  SCHEMA_DELTA.md             the fields BIRD added to the schema
```

## What BIRD reads it for

`bird/tasks.py` loads and indexes. Three things read what it loaded, and they read different
halves of the file:

- **`bird/envs/spec.py::SpecEnvAdapter._apply_spec`** fills the *descriptive* half of an
  `EnvAdapter`. Dynamics do not come from a file and cannot: `_step`, `_reset`, `discretise`,
  `reference_reward` and the success callable stay in Python. That split is the schema's own —
  its human reward is **referenced, never vendored**, and the same argument applies to a
  success check whose value depends on constants resolved through a base class.
- **`bird/config.py::_inherit_from_task_spec`** materialises the config keys that are
  environment *facts* rather than method choices (`TASK_SPEC_KEYS`), each one on `null`.
- **`bird/envs/metaworld.py::MetaWorld.__init__`** reads two more fields no other adapter does.

| spec field | BIRD sink | read by |
|---|---|---|
| `env.env_id` + `env.library.name` | `bird_env_id` — the `problem.env_id` this spec backs (`mt10_`/`mt50_` + the key for `metaworld`, by `METAWORLD_MT10`; `gym_` + the spec id for `mujoco`; the gym id with `-v0` dropped and `-` turned into `_` for `humanoid_bench`; `assistax_` / `upstream_assistax_` + upstream's task key for `assistax` / `assistax_upstream`; verbatim for `bird`; the full table is `bird/tasks.py::_BIRD_ID_RULES`) | `tasks.by_env_id` |
| `description.env_prose` | `_prose` → the four `describe()` renderings (`generate.context.env_spec`); its first sentence is also the registered MT10 factory's docstring | `_apply_spec` |
| `state_surface.flat_fields` | `_state_fields` — indices must be 0..n-1 and the count must equal `obs_dim` | `_apply_spec` |
| `env.spaces.action_fields` | `_action_fields` | `_apply_spec` |
| `state_surface.helpers` | `_helpers` (`pythonic_class_abstraction` only; every entry must be `kind: inline_expression`) | `_apply_spec` |
| `env.horizon`, `env.spaces.obs_dim`, `env.spaces.action_dim` | `horizon`, `obs_dim`, `action_dim` | `_apply_spec` |
| `symbol_mapping` | `symbol_mapping` (`generate.postprocess.symbol_mapping: per_task`) | `_apply_spec` |
| `domain_randomization.parameters` + `.nominal` | `dr_parameters` / `_dr_nominal` | `_apply_spec` |
| `description.success_criterion_prose` | `_success_prose` → `describe_success()`, the success conditions in words for a prompt that asks for them (R*'s critic author); without it the sentence is derived from `success_threshold` | `_apply_spec` |
| `judge.camera` + `judge.extra_views` | `primary_view` / `extra_views` — the camera `render` shows and the extra viewpoints `output.video.n_views` composes for the judge | `_apply_spec` (`spec.py::views_of`) |
| `description.l_task` | `problem.task_description`, when the config leaves it `null` | `_inherit_from_task_spec` |
| `reward.forbidden_symbols` | `verify.forbidden_symbols`, when the config leaves it `null` | `_inherit_from_task_spec` |
| `domain_randomization.parameters` (the axis NAMES) | `rapp.parameters`, when the config leaves it `null` | `_inherit_from_task_spec` |
| `discrete_success.shipped.reimplementation.symbol` | the key into `metaworld._SUCCESS_CHECKS` | `MetaWorld.__init__` |
| `anchors.by_reduction[evaluate.fitness.reduction]` | `baselines` → `evaluate.fitness.normalisation: human_normalised` | `MetaWorld.__init__` |

Which spec a config gets is `problem.task_id`; `null` derives it from `problem.env_id`.

**`anchors` reaches `baselines` on Meta-World, and nowhere else.** `_apply_spec` does not
touch `baselines`; `MetaWorld.__init__` calls `baselines_of` itself. Every other spec-backed
adapter leaves it unset, so `evaluation._env_baselines` returns None and `_normalise_pool`
leaves fitness **raw**. The BIRD-authored toy specs would land there anyway: their expert
anchor is `null` (no scripted policy exists), and `baselines_of` returns None on a pair with
a null half rather than inventing a ceiling.

**`continuous_success.normalized.success_threshold` is NOT the adapter's `success_threshold`.**
`EnvAdapter.success()` is `task_metric(traj) >= self.success_threshold` — a threshold on the
**raw** metric — and `_apply_spec` never sets it; every adapter declares it in Python. The
spec's field is a threshold on the **normalised** metric. The two coincide only where the
normalisation is the identity:

- the BIRD-authored toy and classic-control specs are `normalized.method: explicit` with an
  identity formula, so raw and normalised are one number and the two values agree
  (`pendulum` 0.5, `acrobot` 0.3, `toy_reacher` 0.2, `toy_hungry_thirsty` 0.15, …);
- the fifty Meta-World specs are `normalized.method: anchors`, i.e. `(raw − random)/(expert −
  random)`. Their `0.9` is a policy about *that* scale, while `MetaWorld.success_threshold` is
  `1/(2·horizon) = 0.001` about the raw one. On `door_open` under the per-step reduction, 0.9
  normalised is raw 0.054 — 54× the adapter's threshold, and a different question.

The schema's own description of the field ("BIRD's success() is this threshold on the
normalized metric, so the two can never disagree") asserts the same identity, and it holds only
in the identity case. The discrepancy is recorded here.

### Consumed, and merely recorded

Everything in the table above is an **input**. Most of the rest of the file is **evidence**,
and the distinction is worth keeping sharp: a value that looks like a knob and moves nothing
misleads whoever reads it as one.

`budget` is the in-between case: `bird.py::_write_task_spec` copies `budget.train_steps` and
`budget.learning_verified` into the run's `tasks.resolved.json` beside the config's own
`train.env_steps`, and stops there. **Recorded, never inherited** — the spec's budget is a
claim about the *task*, `train.env_steps` is a decision about *this experiment*, and a run far
below the task's stated budget is a fact to write down rather than a refusal.

`judge` (beyond the two camera fields in the table), `exploits`, `provenance`, the whole
`continuous_success` group (including `normalized.success_threshold`), `reward.human`,
`reward.contract`, the top-level `anchors.random` / `anchors.expert` pair,
`state_surface.fields` (the named namespace — `flat_fields` is the table a reward is actually
written against) and `description.natural_language` / `scene` are read by **no code path
at all**. `tasks.py::_check` *validates* some of them — a null anchor needs a reason, `explicit`
needs a formula, `anchors` and a formula together are refused as two scales for one number —
without any of them reaching the algorithm. They record what was measured and where it came
from. Do not read a value there as wired.

**`env.reset` is evidence too, and the one block here that is checked against code rather
than against itself.** It records what ONE call to the adapter's `_reset(rng)` draws from the
episode seed — `draws` and `constants`, each with the lines that do it and a `basis` saying
how the claim was established — and what the rendered prompt says about the start or the
goal (`prompt_states`, a verbatim quote; `prompt_omits`, the drawn roles it never mentions).
`bird/tasks.py` exposes it as `TaskSpec.reset` (None when absent) and carries the closed
role vocabulary as `RESET_ROLES` / `RESET_ROLE_LABELS`; the reader is
`tests/test_task_specs.py`, and **no code path under `bird/`** — the adapter's `_reset` *is*
the reset, and this block describes it. Changing `distribution.low` changes a record, not an
episode. What keeps it honest is the test rather than the loader:
`entry` is resolved with `ast` to the class the registry binds for `problem.env_id`, and
wherever the simulator is installed 24 seeded resets are stacked and the union of
`draws[].fields` must equal the set of observation columns that moved, in both directions.
Optional in the schema so a core-shaped spec stays valid; required on every spec in
this catalogue by that test (`SCHEMA_DELTA.md` #14).

## Terminology, because two vocabularies meet here

| spec | is | BIRD |
|---|---|---|
| `discrete_success` | the boolean pass/fail check, per step | `success()` |
| `continuous_success` | the scalar the search is scored on | `task_metric()` |

Every task has a continuous metric — a search needs a total order and a boolean cannot rank
candidates. On Meta-World the two are one success check under two **reductions**, and one is a
threshold on the other (`MetaWorld.success_threshold = 1/(2·horizon)`), so they cannot
disagree. Which reduction scores a run is `evaluate.fitness.reduction`, and `anchors` carries
a measured pair for each.

## The rules the schema encodes

1. **Absence is explicit.** A value that was not obtained is `null` *with a `reason`*, never
   omitted and never zero-filled. Validation rejects a null anchor with no reason. The holes
   in these files are the real state of the work.
2. **`hardened <= shipped`, by construction.** Where a hardened oracle exists it returns
   `shipped AND (added conjuncts)`, so the two columns are directly comparable, and leaving a
   sound check untightened is as much a decision as tightening a loose one.
3. **Parity of inputs.** Anywhere the human baseline reads something, the model must be able
   to read it too, or the headline comparison measures the asymmetry instead of the method.
4. **Provenance from source, not docstrings.** Every claim carries `path` and `lines` at a
   pinned commit, and `provenance.measured` separates what was *run* from what was *read*.

## A spec is not a claim that BIRD can run it

A spec may be authored ahead of its adapter. Such specs are inert data, listed in
`_no_adapter.json` with a reason each (empty today -- every spec in the catalogue backs a
registered env). The **registry** is the
`problem.env_id` enum — not this directory — so a spec here can never become a config value
with no implementation behind it. `tests/test_task_specs.py` asserts that partition in both
directions.

## Adding a task

1. Write `tasks/<id>/shared_spec.yaml` against the schema. `id` must equal the directory name.
   Include `env.reset`, authored from the adapter's `_reset` (the block's `entry`), not from
   the prose: open the hook, list one draw per sampled quantity with the exact observation
   entries it moves, and put what a reader might assume varies and does not under
   `constants`. `agent_pose` and `goal` must appear in one or the other. For a generated
   spec the block goes in its generator (`scripts/gen_assistax_specs.py`,
   `scripts/gen_mt50_specs.py`, `scripts/derive_jax_spec.py`), never only in the output.
2. Register an `EnvAdapter` for it, or add it to `_no_adapter.json` with a reason.
3. `uv run python3 -m pytest tests/test_task_specs.py -q`. For the reset block alone,
   `uv run python3 -m pytest tests/test_task_specs.py -k reset` — on a machine with the
   simulator it names the column you missed, and the one you claimed that did not move.

Every spec here is BIRD's to edit.
