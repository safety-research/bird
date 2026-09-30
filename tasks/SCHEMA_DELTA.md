# The fields BIRD added to the schema

`shared_spec.schema.json` is owned here. It has a CORE — the fields every spec in the
catalogue carries — and a set of BIRD EXTENSIONS, each listed below with the reason BIRD needs
it, so a reader can tell at a glance which fields are core and which were added (the schema's
own descriptions say `BIRD EXTENSION` on each).

The additions are authored on specs whose surface this repo's adapter is the measured source
of truth for — the Meta-World specs and the BIRD-native environments. The gym specs keep the
core shape today — that is inertia, not a rule; extending them is catalogue work whenever an
adapter needs the fields.

`schema_version` stays `1`: a **required** `continuous_success.raw.saturates_at` was added
without bumping it. The version field is not a compatibility signal for this schema; the
repository version is.

## Status

Authored on the Meta-World specs and the natives. The gym specs keep the core shape exactly,
with one exception: row 14 (`env.reset`) is authored on every family, the gym specs included —
it describes the adapter's reset, which every spec has. Numbers 6 and 8 are unused (see *Known gaps* below).

| # | field | status | why BIRD needs it |
|---|---|---|---|
| 1 | `state_surface.flat_fields[]`, `fields[].slice` | **authored** | BIRD's 39 positional scalars and the core 7-field namespace are two true descriptions of one observation, at different granularities. `flat_fields` is the surface a reward written as `compute_reward(state, action, next_state)` actually sees; `slice` is the checkable bridge, and a test asserts each slice's width is its declared shape. Without it only one consumer can be served. |
| 2 | `discrete_success.shipped.reimplementation` | **authored** | Symmetric with `hardened`'s existing `{module, symbol}` pin and with "the human reward is referenced, never vendored". Without it the ten success checks have no home in the file and it is a task *description*, not a task *definition*. Resolved through a closed allow-list, never `importlib` on the string: run directories may sit on a world-writable shared filesystem. |
| 3 | anchors keyed by reduction | **authored** (`anchors.by_reduction`, `anchors.reduction`) | Without it, measured per-step baselines could only live in a **prose note**. A measured number in a prose field of an honesty-first schema is the exact failure the schema exists to prevent. |
| 4 | `state_surface.helpers[].kind`, `.expression` | **authored** | BIRD's helpers are advertised as inlineable expressions, not methods, because a real method lets a `self.hand_pos(s)` candidate pass verification (which binds a proxy `self`) and then die in training (which binds `self=None`) — a wasted policy run charged to the reward that did not cause it. A measured distinction, previously inexpressible, which is why the MT10 specs shipped `helpers: []` and lost all eight. |
| 5 | `description.env_prose`, `description.l_task` | **authored** | The core `natural_language` field is written at ENVIRONMENT granularity on nine of the MT10 specs and at INSTRUCTION granularity on the hand-written `door_open` (~120 words, authored to its planned hardened oracle). One field, two jobs, and a consumer cannot tell which it is holding — feeding it to `describe()` would put an instruction where a class docstring belongs, silently. BIRD adds two fields of its own rather than rewriting the core one, so `door_open`'s text is preserved exactly where its author put it. |
| 7 | `reward.forbidden_symbols_by_consumer`, `forbidden_symbols_note` | **authored** | Four of BIRD's six forbidden symbols name *adapter internals*, which are not properties of the task. And BIRD's deliberate omission of `success` is backed by a measurement, which the schema's own `not_tightened_because` idiom says should be written down rather than inferred. |
| 9 | `env.spaces.action_fields[]` | **authored** | `_render_api_stub` needs per-dimension `(name, doc)`; the core schema has only `action_semantics` as one prose blob, which cannot be rendered as a typed stub — and a state/action API stub is one of the published prompt variants, so the granularity is load-bearing. |
| 10 | `anchors.*.method: trained_policy` | **authored** | The core schema's five sources are all things you can *obtain*: a random rollout, a shipped script, a demo, a published figure. BIRD's classic-control envs have none of them -- `pendulum` and `acrobot` ship no scripted solution, so their expert anchors sat at `unavailable`. The remaining way to get a reference is to TRAIN one on the environment's own `reference_reward`, which is a different provenance and must not be filed as `scripted_policy`: a trained policy is whatever its run reached, where a scripted one is the benchmark's own answer. Recording the distinction is what stops a matched-budget baseline being read as a ceiling. |
| 11 | `provenance.rule` widened from `const` to `enum` | **authored** | The core schema pinned one string. Four HumanoidBench specs read task geometry out of MJCF assets no Python source contains -- `h1hand_hurdle`'s hurdle spacing is in `generated_xml_hurdles.xml`, `h1hand_room`'s obstacle layout in `room.xml` -- so `source, not docstrings` would be a false claim about where those numbers came from, and an honesty-first schema must not force one. The added value keeps the operative clause verbatim (*a docstring is not evidence*) and widens only what precedes it. This is a WIDENING, so a document valid against the narrower rule stays valid here; the reverse does not hold, which is the direction every row in this table already goes. |
| 12 | `anchors.published` | **authored** | A per-task result published OUTSIDE this repo (FastTD3's HumanoidBench returns, arXiv:2505.22642) needed a home with the anchors' own discipline, and `anchors.expert` is the wrong one twice over: a paper's number is a baseline somebody else measured, not an in-repo expert -- and it arrives in the benchmark's own unit (an HB episode return), not the spec's task_metric, so filing it as `expert` would put a number in a field whose unit it does not share. A keyed block beside `random`/`expert` records value, unit, spread, budget and success bar verbatim, each with its source down to figure/page and commit, and the normalisation machinery -- which reads only `by_reduction` -- structurally cannot pick it up. The anchor `method` enum's original value `published` stays what it always was: for a published figure IN the spec's own metric, which no HumanoidBench source provides. |
| 13 | `judge.extra_views[]` | **authored** | `judge.camera` names the ONE camera the recorder shows and the judge is sent, and its `note` records what that camera hides -- a tracking camera hides absolute progress, a corner camera hides whatever the arm is in front of. Nothing in the core schema can name a second camera, so `output.video.n_views` (BIRD's recorder key) had nowhere to read from. Each entry is a camera the simulator model defines by `name`, or a `pose` a camera is built from (a tracking camera on a body, or a free camera at a point), with the `mode` and `note` the judge is told about the panel. Ordered, so `n_views: N` takes the first `N-1`. Chosen per task by measurement: on MT10 the top view carries planar progress and moves the most pixels of any fixed camera, the behind-gripper camera is the only one that resolves grasp and contact. Optional, so every spec in the core shape stays valid. |
| 14 | `env.reset` | **authored** (every family; the first extension the ten gym specs carry) | What one call to the adapter's `_reset(rng)` draws from the episode seed, and what the model is told about it. The facts lived in SIX prose homes before this row — `l_task`, `env_prose`, `scene`, a `flat_fields[].description`, `provenance.measured`, an adapter docstring — and none was checkable, so a consumer could not say "the seed changes: goal" without parsing a sentence. Two measured consequences of that silence: `pendulum`'s instruction never says the rod starts anywhere in U(-pi, pi), so a reader assumes it hangs; on MT10 a seed does not draw a goal at all — it picks one of fifty instances pinned at construction from `benchmark_seed` 0, so two seeds collide with probability 1/50 per episode and every run and candidate shares the set (`mechanism: pinned_instances`). The block takes `termination`'s shape one line up (a fact about the code beside a verbatim `prompt_states` about the prompt) and adds a CLOSED role vocabulary (`$defs.reset_role`, nine values, restated as `bird/tasks.py::RESET_ROLES` and pinned equal), so "the seed changes: …" is the set of `role`s in `draws` and "fixed: …" the roles in `constants`, with no summary list to drift; `agent_pose` and `goal` must appear in one or the other, so the two questions the block exists to answer are never answered by silence. `fields` is exact in BOTH directions and held against the adapter: wherever the simulator is installed, `tests/test_task_specs.py` stacks 24 seeded resets and requires the union of `draws[].fields` to equal the columns that moved, and `constants[].fields` not to; `entry` is resolved with `ast` to the class the registry binds, so a block pasted from a sibling task fails on its symbol rather than in a reader's head. Optional in the schema (a spec in the core shape stays valid); required on every spec in this catalogue by the test, the `_no_adapter.json` partition precedent. On the HumanoidBench specs it was authored per task; on the generated Assistax and `jax_toy` specs it comes from their generators (`scripts/gen_assistax_specs.py`, `scripts/derive_jax_spec.py`), whose `--check` refuses a hand edit. `domain_randomization` is deliberately NOT repeated here: DR is a config-gated method knob drawn from the same Generator before `_reset`, and a consumer renders "under DR only: …" from that block. |
| 15 | `anchors.scripted` (+ `policy_id`, `unit`, `reduction`, `status` on `$defs/anchor`) | **authored** | Is the scripted policy good in comparison to a specialist trained from scratch? `expert` is the honest ceiling whatever produced it (Meta-World's bundled script on MT10, a PPO-1M specialist on the gym suite, `unavailable` on HumanoidBench) and `published` is a paper's number in the benchmark's unit; neither is THIS repo's own scripted policy, which lives in a `policies/*/policies.yaml` manifest with its own unit and reduction. `scripted` records that entry's score beside the other anchors -- value, `policy_id`, `status`, `unit`, `reduction`, n -- so the comparison is readable off the spec. The manifest stays the primary record: every spec's copy is the registry's best entry for the env (solved > partial > reference > negative, then score), except where the only entries are deliberate metric hacks (`h1hand_walk`, `h1hand_crawl`: see their `exploits`), which never stand as the task's scripted anchor. A task with no qualifying entry carries an explicit `null`. Not in `by_reduction`, so normalisation never reads it. The generated Assistax specs read it off the registry at generation time (`scripts/gen_assistax_specs.py::_scripted`). |

## Known gaps

Two things the schema does not yet express are recorded here rather than as table rows:
who authored a task's success check ("the environment's, not ours" is prose only today),
and the four `describe()` renderings -- `description` carries prose, scene, criterion and
a source pointer, none of which is an API stub or a class abstraction; BIRD renders all
four from the adapter.

## The two branches of `normalized.method`

`continuous_success.normalized.method` is `anchors` on the Meta-World and gym specs, and the
schema asserts `formula` STAYS null on them — filling it would write one copy per spec of the
one expression the enum's own description already carries.

The native specs here use the other branch, `explicit`, with a formula: those environments ship **no scripted expert**, so an anchors-based
scale has no upper anchor to compute from, and the raw metric is already a fraction of steps
in [0, 1]. `method: explicit` with an identity formula is what the schema requires in that
case, and `tests/test_task_specs.py::test_the_normalisation_has_exactly_one_source` enforces
the pairing in both directions here.

`explicit` **with** a formula satisfies both rules.

## Deliberately **not** added

**An executable sampler for `env.reset`.** The block describes the distribution and cites
the lines that draw it; it never carries code that draws. Dynamics do not come from a file
(`bird/envs/spec.py`), the human reward is *referenced, never vendored*, and the success
check resolves through a closed allow-list for a reason that applies here verbatim: run
directories may sit on a world-writable shared filesystem and `tasks/` is ordinary tracked YAML, so
a data field that could name or carry arbitrary code would be an execution path into a
training run. Reference, never carry.

**A per-spec `reproducible_from_seed` boolean.** True on every spec by construction —
`EnvAdapter.reset` hands `_reset` the Generator it was given (`bird/envs/base.py::EnvAdapter.reset`)
— and a field that never varies is a fabricated pin: it looks like a knob and moves nothing. The claim is made ONCE instead, as a universal test:
`reset(default_rng(3))` twice must be bitwise equal for every constructible env
(`tests/test_task_specs.py`, test F). A per-`flat_fields[]` "varies at reset" boolean was
rejected for the same reason in the other direction: 151 to 328 booleans per HumanoidBench
spec that `draws[].fields` already states as a set, and a second copy to drift.

**Any repetition of `domain_randomization` inside `env.reset`.** DR is a config-gated
method knob (`rapp.*`, `train.domain_randomization`), drawn from the same Generator BEFORE
`_reset` runs, so turning it on shifts every draw the block lists. Repeating the axes here
would put a method choice inside a task fact and give a consumer two tables to reconcile;
"under DR only: …" is rendered from the `domain_randomization` block that already exists.

**`env.registry_ids` (a second id per catalogue).** Two spellings of one id is exactly what BIRD's `mt10_`
prefix rationale argues against — the prefix is the `MT10_V3` key verbatim so that an id which
resolves is an id the benchmark accepts. One derivation rule per adapter family, plus a test
that every registered env id resolves to exactly one spec, is correct and cannot drift.

**The `compute_dense_reward(s, action)` reward contract.** BIRD's signature is
`compute_reward(state, action=None, next_state=None)` over 39 flat scalars, and that flat
surface is what `symbol_mapping` values, `discretise`, `training.compile_reward`'s arity
adaptation and `verification.call_reward` are all written against. Adopting the spec's contract
changes §1 for every method and every shipped config. The divergence is recorded per run in
`tasks.resolved.json` instead.

- `anchors.scripted` may be `null`: a spec whose task has no qualifying registry entry carries an explicit null, so a generated spec cannot inherit another task's score on regeneration.
