"""`train.interaction: shared_population` spends what `independent` spends.

**Why this file exists at all**: the failure it guards is silent and produces a
plausible number. LaRes is compared against other methods at a matched env-step
budget; if
`shared_population` pooled or divided that budget even slightly differently,
every arm would still run, still finish, still report a success rate, and the
comparison would quietly be at two different budgets. Nothing else in the tree
measures the round's total spend against the independent path, and looking at a
run cannot catch it -- a 10% budget difference and a 10% method difference
render identically.

The invariant, in one line: **`shared_population` + `uniform` gives every
candidate exactly `train.env_steps`, the same as `independent`.** That is also
what makes the pair LaRes's own Fig. 6a ablation ("removing Thompson sampling
leads to a performance decline") rather than a comparison at two budgets.

`thompson_success` deliberately does NOT get an equal-shares assertion -- the
whole mechanism is unequal shares -- so what is pinned there is the pooled
TOTAL, which is the quantity the budget match rests on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bird.components.population import _UniformAllocator, _merge_slices, _slice_success
from bird.config import ConfigError, load
from bird.types import Candidate, TrainResult

#: `lares` under an on-policy learner: PPO, no shared replay buffer, no reward scaling.
_LARES_UNDER_PPO = {"train.algorithm": "ppo", "train.interaction_cfg.shared_buffer": False,
                    "train.reward_scaling": "none"}


def _res(cand_id: str, steps: int, *, checkpoints=None) -> TrainResult:
    c = Candidate(cand_id=cand_id, iteration=0,
                  reward_code="def r(s,a,s2): return 0.0, {}")
    r = TrainResult(cand_id=cand_id, candidate=c)
    r.env_steps_used = steps
    r.checkpoints = list(checkpoints or [])
    return r


def test_uniform_gives_every_arm_one_turn_per_wave():
    """The control's defining property: round robin, nobody starved."""
    a = _UniformAllocator(4)
    assert a.select() == [0, 1, 2, 3]
    a.update(0, 1.0)
    assert a.select() == [0, 1, 2, 3], "uniform must not react to outcomes"


def test_thompson_concentrates_on_the_arm_that_succeeds():
    """The mechanism, at its cheapest: an arm that keeps succeeding is preferred.

    Not a distributional test -- the draw is random and a flaky assertion here
    would be worse than none. It asserts only the direction the paper claims:
    after a long run of successes for one arm and failures for the rest, that
    arm leads the sampled posterior far more often than chance.
    """
    from bird.components.population import _ThompsonSuccessAllocator

    cfg = load("lares", profile="tester")

    class _Ctx:
        pass

    import random
    ctx = _Ctx()
    ctx.cfg = cfg
    ctx.rng = random.Random(0)
    a = _ThompsonSuccessAllocator(ctx, 3)
    for _ in range(30):
        a.update(0, 1.0)
        a.update(1, 0.0)
        a.update(2, 0.0)
    firsts = [a.select()[0] for _ in range(50)]
    assert firsts.count(0) > 40, (
        f"arm 0 succeeded 30/30 and the others 0/30, but led only "
        f"{firsts.count(0)}/50 draws")


def test_the_window_forgets():
    """`interaction_cfg.window` truncates history -- LaRes sets it per task."""
    from bird.components.population import _ThompsonSuccessAllocator

    cfg = load("lares", profile="tester", overrides={"train.interaction_cfg.window": 5})

    class _Ctx:
        pass

    import random
    ctx = _Ctx()
    ctx.cfg = cfg
    ctx.rng = random.Random(0)
    a = _ThompsonSuccessAllocator(ctx, 2)
    for _ in range(20):
        a.update(0, 1.0)
    assert len(a.history[0]) == 5


def test_merge_sums_the_steps_and_takes_the_last_policy():
    """A merged result must report the WHOLE spend and the FINAL policy.

    Getting either wrong is silent: summing the curve instead of concatenating
    it, or reporting an intermediate policy, both yield a run that looks fine
    and describes a policy that never existed.
    """
    a, b, c = _res("x", 100), _res("x", 100), _res("x", 100)
    a.checkpoints = [{"fitness": 0.1}]
    b.checkpoints = [{"fitness": 0.2}]
    c.checkpoints = [{"fitness": 0.3}]
    c.policy_ref = "policy:final"
    merged = _merge_slices([a, b, c])
    assert merged.env_steps_used == 300
    assert [ck["fitness"] for ck in merged.checkpoints] == [0.1, 0.2, 0.3]
    assert merged.policy_ref == "policy:final"


def test_slice_success_reads_the_environment_flag_not_the_reward():
    """LaRes counts the env's success (`:827`), never the designed reward.

    The native-signal rule: `_slice_success` reads only genuine env
    success-flag keys (`NATIVE_SUCCESS_KEYS`). `fitness`/`task_success` --
    ALIASES of the BIRD `task_metric` (custom_metric) in the curve rows -- are
    NOT a fallback, so `thompson_success` cannot steer the inner loop by a
    number this repo wrote. Both backends always emit `success_rate`."""
    assert _slice_success(_res("x", 1, checkpoints=[{"success_rate": 0.75}])) == 0.75
    # `fitness`/`task_success` are custom_metric aliases -> NOT read as success.
    assert _slice_success(_res("x", 1, checkpoints=[{"fitness": 0.4}])) == 0.0
    assert _slice_success(_res("x", 1, checkpoints=[{"task_success": 0.9}])) == 0.0
    assert _slice_success(_res("x", 1)) == 0.0


def _entry():
    import importlib.util
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bird_entry", repo / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: `train.env_steps` for the spend tests. NOT the tester profile's 400: that
#: makes an 80-step slice, and the mock learner trains in whole CEM rounds of
#: `pop x horizon` = 10 x 25 = 250 steps under a `max(200, ...)` floor
#: (`training._run_backend`), so an 80-step slice LEARNS 250 -- a 3.1x
#: over-spend the driver warns about (`interaction_slice_below_horizon`) and
#: this test must not read as the method's. 2500 gives 500-step slices = two
#: whole rounds, so the mock honours the allocation exactly in both modes and
#: the training spend can be pinned as an EQUALITY. MEASURED: with the
#: profile's 400, `independent` learns 500 per candidate and
#: `shared_population` 250 per 80-step slice (6250 per round against 2000
#: allocated); at 2500 both learn exactly what was allocated.
_HONOURABLE_ENV_STEPS = 2500
#: Per-backend-call overhead on this env, derived rather than measured: each
#: call ends with `evaluate.rollouts_per_candidate` (3) rollouts of the
#: 25-step horizon.
_ROLLOUT_STEPS_PER_CALL = 3 * 25
#: Checkpoint evaluations per call at this shape: a 500-step slice is 2 CEM
#: rounds -> 2 checkpoints (`EVAL_EPISODES` 4 episodes, then the last at
#: `EVAL_EPISODES_FINAL` 8), a 2500-step run is 10 rounds -> 5 checkpoints
#: (`_checkpoint_stride`). MEASURED and pinned as equalities below.
_EVAL_STEPS_PER_SLICE = (4 + 8) * 25
_EVAL_STEPS_PER_RUN = (4 * 4 + 8) * 25


def _journal(root):
    import json
    (j,) = sorted(Path(root).rglob("journal.jsonl"))
    return [json.loads(line) for line in j.read_text().splitlines() if line.strip()]


def _train_results(root):
    import json
    return [json.loads(f.read_text())
            for f in sorted(Path(root).rglob("candidates/iter*_*/train_result.json"))]


def _dead_arms_by_iteration(root) -> dict:
    """`{iteration: [cand_id, ...]}` of the arms that were LAUNCHED and learned
    nothing: `trained` false with an EMPTY `skip_reason` and a recorded `error`.
    A skipped arm (`skip_reason` set; `training.py`, TWO POPULATIONS, TWO
    COUNTERS) was never launched and is not in `interaction_open.n_arms`, so it
    is not a dead arm here. Every iteration that wrote a result has a key, so a
    round with no dead arm reads as `[]` rather than as absent."""
    import json
    import re
    out: dict = {}
    for f in sorted(Path(root).rglob("candidates/iter*_*/train_result.json")):
        tr = json.loads(f.read_text())
        it = int(re.search(r"iter(\d+)_", f.parent.name).group(1))
        out.setdefault(it, [])
        if not tr["trained"] and not tr["skip_reason"] and tr.get("error"):
            out[it].append(tr["cand_id"])
    return out


def test_uniform_shared_population_spends_what_independent_spends(tmp_path):
    """THE budget-match invariant, measured on real runs of both paths.

    Both run end to end on the tester profile (mock LLM, mock learner) at
    `_HONOURABLE_ENV_STEPS`, and the TRAINING env steps are compared:
    `shared_population` + `uniform` must LEARN exactly what `independent`
    learns, per round and in total, or a budget-matched comparison against
    other methods is at two different budgets and nothing says so.

    Three things are pinned, each an equality:

      1. `policy_trainings` agrees and the slice count sits in
         `training_slices` -- a candidate trained in five slices is ONE
         training, and an inflated counter would make LaRes look several
         times more expensive in the one column a cost comparison reads;
      2. the LEARNING spend agrees: `interaction_close.env_steps_train_spent`
         equals the allocation, which equals `n_arms x train.env_steps`, and
         the sum over rounds equals what `independent`'s seed rows learned,
         which equals `n_trainable x train.env_steps x n_iterations`;
      3. the OVERHEAD does not agree, and the artifact says by how much:
         `budget.env_steps` is learning + checkpoint evaluation, and every
         backend call also runs the §4 rollouts. `shared_population` pays
         both per SLICE where `independent` pays them per training, so on the
         same learning budget it bills `slices_per_candidate` x the rollout
         steps and (checkpoints per slice / checkpoints per run) x the
         evaluation steps -- 5x and 2x here, 5x and 5x at a full-profile
         shape (20 checkpoints per 200k slice and per 1M run). Pinned so a reader
         of `budget.env_steps` cannot mistake the overhead for a method
         difference, which the file header says is exactly the confusion
         this file exists to prevent.

    Asserting only (1) would not be enough: at the profile's own 400 steps
    the tester-tier gap (13125 spent against 2000 allocated per round) is
    6250 learning from the mock's 250-step rounds, 5000 evaluation and 1875
    rollouts, and nothing else in the suite reads any of the three.

    An arm that CANNOT learn is accounted for, by name. A program valid at
    compile time that raises at its first reward call -- the mock's
    `wrong_signature` archetype, `def compute_reward():` -- is trainable as
    far as anything before the learner can tell, so it is one of
    `interaction_open.n_arms`, is allocated `train.env_steps` at open, is
    drawn `slices_per_candidate` times, and learns none of it in EITHER mode
    (`trained: false`, `train_steps: 0`, the same `TypeError` in both
    `train_result.json`s). The equalities are therefore over the arms that
    LEARNED, with the dead arms' allocation asserted as the exact shortfall
    -- not dropped from the pool, which is decided before any learner runs,
    and not absorbed into a floor. Any edit to the `toy.py` class body can
    produce one: `generate.context.env_spec: full_source` puts that class body
    in the prompt and the mock mixes the prompt into every draw's RNG
    (`mock.py::_rng_for`). Whether an arm can learn is a property
    of its program, so the per-iteration dead counts must agree across modes
    -- a disagreement means the two runs generated different programs, which
    is upstream of anything this test compares.

    The seed is MEASURED so that the dead-arm path is exercised: at seed 4 the
    mock draws a runtime-dead arm in both iterations of both modes (a NaN reward
    and the `wrong_signature` archetype). Seeds 1-3 draw no dead arm at all, and
    seed 0 draws a program that does not COMPILE -- it fails before the learner
    runs any rollout, so its share is 0, a different case from the one the
    per-dead-arm assertion below describes. A prompt edit re-rolls this.
    """
    entry = _entry()
    got, roots = {}, {}
    for mode in ("independent", "shared_population"):
        cfg = load("lares", profile="tester", overrides={
            "seed": 4,
            "train.interaction": mode,
            "train.env_steps": _HONOURABLE_ENV_STEPS,
            "train.interaction_allocator": "uniform",
            "train.candidate_parallelism": "sequential",
        })
        roots[mode] = tmp_path / mode
        got[mode] = entry.run(cfg, out_root=str(roots[mode]))["budget"]
    n_iterations = int(cfg["loop.n_iterations"])
    slices_per = int(cfg["train.interaction_cfg.slices_per_candidate"])
    env_steps = _HONOURABLE_ENV_STEPS

    # (0) the arms that could not learn -- see the docstring. Counted per
    # iteration on each side. The counts must agree at ITERATION 0, where the
    # two modes' prompts are byte-identical, so a disagreement there means the
    # two runs generated different programs from the same prompt. From
    # iteration 1 on they may legitimately differ: the prompt carries the
    # previous round's training statistics, and `shared_population` trains in
    # `slices_per_candidate` slices with (checkpoints per slice / checkpoints
    # per run) x the checkpoints -- measured on this fixture, 10
    # `task_success` samples against `independent`'s 5 in the iteration-1
    # prompt -- and the mock mixes the prompt into every draw's RNG
    # (`mock.py::_rng_for`), so the iteration-1 programs are different draws.
    # Asserting equality at EVERY iteration would pass only by coincidence
    # (both modes happening to draw only live arms after iteration 0); a
    # prompt edit can make `independent` draw a dead arm at iteration 1 that
    # `shared_population` does not. The equalities below are therefore per mode, each against its
    # OWN dead set, and the cross-mode claim is the per-live-arm one the file
    # header states: every arm that learned learned exactly `train.env_steps`.
    dead = {mode: _dead_arms_by_iteration(roots[mode]) for mode in roots}
    assert len(dead["independent"].get(0, [])) == len(dead["shared_population"].get(0, [])), (
        "which arms cannot learn is a property of the program, not of the "
        f"interaction mode -- at iteration 0 the prompts are identical, so the "
        f"two runs generated different programs from one prompt: {dead}")
    dead_shared = dead["shared_population"]
    n_dead = sum(len(v) for v in dead_shared.values())
    n_dead_indep = sum(len(v) for v in dead["independent"].values())

    # (1) trainings and slices
    assert (got["shared_population"]["policy_trainings"]
            == got["independent"]["policy_trainings"]), (
        "a sliced training is still ONE training; the slice count belongs in "
        f"training_slices -- got {got['shared_population']}")
    assert got["shared_population"]["training_slices"] > 0, (
        "shared_population ran no slices, so this comparison proved nothing")
    assert got["independent"]["training_slices"] == 0

    # (2) learning spend, per round and in total
    closes = [e for e in _journal(roots["shared_population"])
              if e.get("stage") == "interaction_close"]
    opens = [e for e in _journal(roots["shared_population"])
             if e.get("stage") == "interaction_open"]
    assert len(closes) == len(opens) == n_iterations
    for it, (o, e) in enumerate(zip(opens, closes)):
        assert e["pooled_env_steps"] == o["n_arms"] * env_steps, (
            "the pool must be n_arms x train.env_steps, not a number derived from itself")
        assert e["env_steps_allocated"] == e["pooled_env_steps"]
        dead_here = dead_shared.get(it, [])
        assert e["env_steps_train_spent"] == e["env_steps_allocated"] - len(dead_here) * env_steps, (
            "the learners LEARNED more or less than the pool allocated to the arms "
            f"that could learn -- the budget match is off: {e} (dead arms: {dead_here})")
        for cid in dead_here:
            # A dead arm is drawn `slices_per_candidate` times regardless and
            # pays the §4 rollouts on every draw while learning nothing, so
            # its whole share is rollouts. An allocator that retired a dead
            # arm would move this number, and should do so deliberately.
            assert e["shares"][cid] == slices_per * _ROLLOUT_STEPS_PER_CALL, (cid, e["shares"])
        assert e["env_steps_spent"] == (e["env_steps_train_spent"] + e["env_steps_eval_spent"]
                                        + e["env_steps_rollout_spent"]), \
            "spend must decompose exactly into learning + evaluation + rollouts"
    shared_train = sum(e["env_steps_train_spent"] for e in closes)

    # Every arm `independent` LAUNCHED (skipped ones never were), then the
    # ones that learned.
    indep_launched = [tr for tr in _train_results(roots["independent"]) if not tr["skip_reason"]]
    indep = [tr for tr in indep_launched if tr["trained"]]
    indep_train = sum(m["train_steps"] for tr in indep for m in tr["seed_metrics"])
    indep_eval = sum(m["env_steps"] - m["train_steps"] for tr in indep for m in tr["seed_metrics"])
    # Rollouts are paid per backend call whether or not the call learned, so
    # the dead arms are in this sum -- as they are in `env_steps_rollout_spent`.
    indep_roll = sum(tr["env_steps_used"] - sum(m["env_steps"] for m in tr["seed_metrics"])
                     for tr in indep_launched)
    n_arms = sum(o["n_arms"] for o in opens)
    n_live = n_arms - n_dead
    n_live_indep = n_arms - n_dead_indep
    # `policy_trainings` counts LAUNCHES: a dead arm's failed training is still
    # a training this search's result depended on (`budget.py`), in both modes.
    assert len(indep_launched) == n_arms == got["independent"]["policy_trainings"]
    assert len(indep) == n_live_indep
    assert indep_train == n_live_indep * env_steps, (
        f"independent learned {indep_train}, not n_live x train.env_steps")
    # THE invariant, per live arm: `shared_population` + `uniform` learned exactly
    # `train.env_steps` for every arm that could learn, the same as `independent`.
    assert shared_train == n_live * env_steps, (
        f"shared_population learned {shared_train}, not n_live x train.env_steps")
    assert shared_train // max(1, n_live) == indep_train // max(1, n_live_indep) == env_steps, (
        f"per live arm: shared_population {shared_train}/{n_live} against "
        f"independent {indep_train}/{n_live_indep}")
    # `budget.env_steps` is learning + checkpoint evaluation + ROLLOUTS on
    # both paths. The post-training evaluation rollouts are charged because
    # they are real env interaction the machine paid for; leaving them out
    # would under-report `budget.json`. The decomposition asserted at
    # `env_steps_spent` above already names all three terms, so this is the
    # same identity one level up rather than a new claim.
    #
    # Taken from the close events rather than written as a literal: the
    # rollout term moves with `evaluate.rollouts_per_candidate` and with the
    # episode length, and a hardcoded figure goes stale.
    # `indep_roll`, computed twenty lines up, is exactly this term already:
    # `env_steps_used` minus the seed-metric sum, over the LAUNCHED calls.
    # Recomputing it under a second name would put two bindings for one
    # quantity in one function -- the thing that goes stale when only one of
    # them is updated.
    assert got["independent"]["env_steps"] == indep_train + indep_eval + indep_roll
    assert got["shared_population"]["env_steps"] == sum(
        e["env_steps_train_spent"] + e["env_steps_eval_spent"]
        + e["env_steps_rollout_spent"] for e in closes)

    # (3) the overhead, as equalities at this shape. A call that raised at its
    # first reward call ran its rollouts and evaluated no checkpoint, so the
    # rollout equalities are over every call and the evaluation ones over the
    # calls that learned.
    n_calls = sum(e["backend_calls"] for e in closes)
    live_calls = n_live * slices_per
    assert n_calls == n_arms * slices_per
    assert sum(e["env_steps_rollout_spent"] for e in closes) == n_calls * _ROLLOUT_STEPS_PER_CALL
    assert indep_roll == n_arms * _ROLLOUT_STEPS_PER_CALL
    assert sum(e["env_steps_eval_spent"] for e in closes) == live_calls * _EVAL_STEPS_PER_SLICE
    assert indep_eval == n_live_indep * _EVAL_STEPS_PER_RUN
    assert sum(e["checkpoint_evaluations"] for e in closes) == live_calls * 2
    assert sum(m["n_checkpoints"] for tr in indep for m in tr["seed_metrics"]) == n_live_indep * 5
    # So on this env, at equal learning, shared_population bills 5x the
    # rollout steps and 2x the evaluation steps of independent.
    assert (sum(e["env_steps_rollout_spent"] for e in closes)
            == slices_per * indep_roll)
    # Per LIVE arm on each side, because the two modes' dead sets may differ
    # from iteration 1 on (see (0)); each side's total is pinned exactly above.
    assert n_live > 0 and n_live_indep > 0
    assert ((sum(e["env_steps_eval_spent"] for e in closes) // n_live) * _EVAL_STEPS_PER_RUN
            == (indep_eval // n_live_indep) * slices_per * _EVAL_STEPS_PER_SLICE)

def test_uniform_allocates_the_pool_exactly(tmp_path):
    """Every arm is ALLOCATED exactly `train.env_steps`, in `slices_per_candidate`
    slices of `train.env_steps / slices_per_candidate`, and the pool is spent.

    Read off the journal's `interaction_open` / `interaction_slice` /
    `interaction_close` events and pinned to the CONFIG, not to the driver's
    own arithmetic: `env_steps_allocated == pooled_env_steps` alone is the
    loop's exit condition restated (a pool computed over the wrong number of
    arms, or doubled, satisfies it -- checked by mutation), so
    the pool is asserted equal to `n_arms x train.env_steps`, every arm's
    slice count to `slices_per_candidate`, and every slice's allocation to
    the quotient.

    ALLOCATED, not spent, and on the profile's own 400 steps on purpose: what
    a backend SPENDS depends on whether it can honour a slice, and here it
    cannot -- 80-step slices against a learner that trains in 250-step rounds
    -- which is what `test_uniform_shared_population_spends_what_independent_
    spends` measures at a slice the mock can honour. Bounding the loop on
    ALLOCATED rather than spent steps is what makes this hold at all: bounded
    on spent steps, one arm's overshoot eats a later arm's share and `uniform`
    delivers 56% of the pool.
    """
    entry = _entry()
    cfg = load("lares", profile="tester", overrides={
        "seed": 0,
        "train.interaction": "shared_population",
        "train.interaction_allocator": "uniform",
        "train.candidate_parallelism": "sequential",
    })
    entry.run(cfg, out_root=str(tmp_path / "sp"))
    journals = sorted((tmp_path / "sp").rglob("journal.jsonl"))
    assert len(journals) == 1, f"expected one run dir, found {journals}"
    events = _journal(tmp_path / "sp")
    opens = [e for e in events if e.get("stage") == "interaction_open"]
    closes = [e for e in events if e.get("stage") == "interaction_close"]
    slices = [e for e in events if e.get("stage") == "interaction_slice"]
    assert closes, "shared_population wrote no interaction_close event"
    assert len(opens) == len(closes) == int(cfg["loop.n_iterations"])
    per_cand = int(cfg["train.env_steps"])
    n_slices = int(cfg["train.interaction_cfg.slices_per_candidate"])
    for o, e in zip(opens, closes):
        assert e["pooled_env_steps"] == o["n_arms"] * per_cand, (
            f"the pool must be n_arms x train.env_steps: {e['pooled_env_steps']} "
            f"against {o['n_arms']} x {per_cand}")
        assert e["env_steps_allocated"] == e["pooled_env_steps"], (
            "uniform must allocate the whole pool and no more -- "
            f"{e['env_steps_allocated']} of {e['pooled_env_steps']}")
        assert e["backend_calls"] == o["n_arms"] * n_slices
    by_arm = {}
    for s in slices:
        by_arm.setdefault(s["cand_id"], []).append(s["allocated"])
    assert len(by_arm) == sum(o["n_arms"] for o in opens)
    for cid, allocs in by_arm.items():
        assert len(allocs) == n_slices, f"{cid} was allocated {len(allocs)} slices"
        assert all(a == per_cand // n_slices for a in allocs), f"{cid}: {allocs}"
        assert sum(allocs) == per_cand


def test_thompson_success_is_refused_by_a_gt_free_config():
    """The allocator is a ground-truth channel, and the rule that says so must fire.

    `thompson_success` updates its posterior from the ENVIRONMENT's success
    flag once per slice (`LaRes_from_scratch.py:827`), not from the designed
    reward. That is GT access *inside the inner loop* -- earlier and cheaper
    than the access at the selection boundary -- so a config whose fitness
    deliberately never touches the task metric would be leaking it here.

    This is tested rather than left to the rule's existence because the failure
    is silent in the direction that matters: if the rule were dropped or its
    condition inverted, a GT-free arm would keep loading, keep running, and
    keep reporting a plausible success rate while quietly steering its
    interaction budget with the ground truth. Nothing downstream would say so,
    and a GT-free method is compared against methods like this one on exactly
    that distinction.

    `uniform` is asserted to still load, so the test pins the rule's SCOPE and
    not merely that some error is raised -- a rule that refused both would pass
    a raises-only check while removing the paper's own Fig. 6a control.
    """
    with pytest.raises(ConfigError) as exc:
        load("lares", profile="tester",
             overrides={"evaluate.fitness.source": "none"})
    assert "thompson_success" in str(exc.value)

    load("lares", profile="tester", overrides={
        "evaluate.fitness.source": "none",
        "train.interaction_allocator": "uniform",
    })


def test_an_allocator_without_shared_population_is_refused_as_inert():
    """`interaction_allocator` is read only under `shared_population`.

    Same class of failure as above and the reason `_check_coherence` carries
    the rule: under `independent` nothing allocates, so a config naming an
    allocator would declare a mechanism that never runs -- a fabricated pin:
    a key that validates, moves the config hash, and changes nothing.
    """
    with pytest.raises(ConfigError) as exc:
        load("lares", profile="tester",
             overrides={"train.interaction": "independent"})
    assert "interaction_allocator" in str(exc.value)


# ==========================================================================
# elite retention: the population `elitist_population` keeps must be CARRIED
# ==========================================================================


def test_the_elite_set_of_round_r_is_carried_into_round_r_plus_1(tmp_path):
    """`update.topology: elitist_population` keeps its (mu + lambda) population
    in `state.archive`, so `configs/methods/lares.yaml` must carry `archive`: without
    it `RunState.apply_carry` empties the elite set at every boundary,
    `select.n_survivors: 3` (App. B's elite size) sizes a set nothing reads,
    and `archive_occupied` is 0 in every state snapshot of every LaRes run.

    Pinned on a real tester run: the elites round 0 selected are the elites
    round 1 starts with, named on round 1's `interaction_close` beside the
    round's `elite_id`, and the state snapshot written after round 0's carry
    holds them.
    """
    import json
    entry = _entry()
    cfg = load("lares", profile="tester", overrides={"seed": 0})
    assert cfg["update.topology"] == "elitist_population"
    assert "archive" in cfg["loop.carry"], "the published carry must name the elite set's slot"
    entry.run(cfg, out_root=str(tmp_path))
    journal = _journal(tmp_path)
    selects = [e for e in journal if e.get("stage") == "select"]
    closes = [e for e in journal if e.get("stage") == "interaction_close"]
    assert len(selects) >= 2 and len(closes) == len(selects)

    winners_0 = selects[0]["winners"]
    assert len(winners_0) == cfg["select.n_survivors"] == 3
    # Round 1 opens with round 0's survivors as its retained elites, ranked,
    # and its elite is round 0's winner.
    assert set(closes[1]["retained_elites"]) == set(winners_0), closes[1]
    assert closes[1]["elite_id"] == winners_0[0]
    assert closes[1]["retained_elites"][0] == winners_0[0]
    # Round 0 had nothing to retain and no elite yet -- the honest zero.
    assert closes[0]["retained_elites"] == [] and closes[0]["elite_id"] == ""
    # And the snapshot written AFTER round 0's carry still holds the set.
    (state0,) = sorted(Path(tmp_path).rglob("state/iter00.json"))
    assert json.loads(state0.read_text())["archive_occupied"] == 3


def test_a_population_topology_without_the_archive_carry_is_refused():
    """The coherence rule behind the test above, in both directions.

    A carry that drops `archive` under `elitist_population` is the declared
    mechanism that does not execute; the shipped `lares` (which carries it)
    still loads.
    """
    with pytest.raises(ConfigError, match="state.archive"):
        load("lares", profile="tester",
             overrides={"loop.carry": ["best_reward", "replay_buffer", "policy_checkpoint"]})
    assert "archive" in load("lares", profile="tester")["loop.carry"]
    assert "archive" in load("lares", profile="tester", overrides=_LARES_UNDER_PPO)["loop.carry"]


# ==========================================================================
# skip_duplicates: false -- a repeated draw is a CONSECUTIVE slice, not a twin
# ==========================================================================


def test_a_member_drawn_twice_in_one_wave_trains_consecutive_slices(monkeypatch):
    """`interaction_cfg.skip_duplicates: false` is the ablation of App. B's
    rule, and done naively it is a double charge for one slice: both copies
    of a repeated arm scheduled in the same wave from the same `resume[arm]`,
    `by_arm` getting both, `issued` charging both and `resume[arm]` keeping
    only the last -- one slice's learning discarded and billed.

    The allocator is forced to draw `[0, 0, 1]` every wave. Arm 0's second
    slice of the wave must RESUME the first's policy and carry slice index 1,
    in a later pass; arm 1 sits in the first pass beside arm 0's first slice.
    """
    from bird import registry
    from bird.budget import Budget
    from bird.components import population as P
    from bird.context import Context
    from bird.state import RunState

    cfg = load("lares", profile="tester", overrides={
        "seed": 0, "output.tracker": "none",
        "train.interaction_allocator": "thompson_success",
        "train.interaction_cfg.skip_duplicates": False,
        "train.interaction_cfg.slices_per_candidate": 2,
        "train.reward_scaling": "none", "train.elite_constraint.kind": "none"})
    monkeypatch.setattr(P._ThompsonSuccessAllocator, "select", lambda self: [0, 0, 1])
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    events = []
    ctx.event = lambda stage, **fields: events.append({"stage": stage, **fields})
    backend = registry.get("train_backend", cfg["train.backend"])
    calls = []

    def spy(ctx_, state_, c, n_seeds, **kw):
        res = backend(ctx_, state_, c, n_seeds, **kw)
        calls.append((c.cand_id, kw.get("resume_ref"), kw["handoff"].index, res.policy_ref))
        return res

    code = "def compute_reward(state, action):\n    return float(state[0]), {}\n"
    cands = [Candidate(cand_id=f"c000{i}", reward_code=code, iteration=0) for i in range(2)]
    state = RunState()
    inter = registry.get("interaction", "shared_population")
    inter(ctx, state, cands, lambda c: backend(ctx, state, c, 1), lambda c, r: None,
          registry.get("candidate_parallelism", "sequential"), spy)

    first_wave = [e for e in events if e["stage"] == "interaction_slice" and e["wave"] == 0]
    arm0 = [e for e in first_wave if e["arm"] == 0]
    assert [e["slice_index"] for e in arm0] == [0, 1], arm0
    assert [e["wave_pass"] for e in arm0] == [0, 1], "the repeat runs in a later pass"
    assert [e["wave_pass"] for e in first_wave if e["arm"] == 1] == [0]
    c0 = [c for c in calls if c[0] == "c0000"]
    assert c0[0][1] is None and c0[0][2] == 0
    assert c0[1][1] == c0[0][3] and c0[0][3] is not None, (
        "the second slice must resume the first's policy, not the same resume point")
    assert c0[1][2] == 1
    # And nothing was double-charged: every slice that ran is in `issued`.
    close = [e for e in events if e["stage"] == "interaction_close"][0]
    assert close["env_steps_allocated"] == len(calls) * (cfg["train.env_steps"] // 2)


# ==========================================================================
# one agent per member: seeds_per_candidate / select.allocation are refused
# ==========================================================================


def test_shared_population_refuses_seeds_and_allocation_it_cannot_honour():
    """A slice resumes ONE policy, so the driver launches every slice with
    `n_seeds=1` and never reads the `select.allocation` plan. A config
    declaring `train.seeds_per_candidate: 3` beside `shared_population` would
    otherwise resolve, hash and print three replicates while one seed trained
    -- a seeds ablation on LaRes that measures nothing. Refused at load,
    naming both keys; the shipped configs and an `independent` seeds ablation
    still load.
    """
    with pytest.raises(ConfigError) as exc:
        load("lares", profile="tester", overrides={"train.seeds_per_candidate": 3})
    assert "seeds_per_candidate" in str(exc.value) and "shared_population" in str(exc.value)
    with pytest.raises(ConfigError) as exc:
        load("lares", profile="tester", overrides={"select.allocation": "bandit"})
    assert "select.allocation" in str(exc.value) and "shared_population" in str(exc.value)
    # Both directions: what the rule must NOT refuse.
    assert load("lares", profile="tester")["train.seeds_per_candidate"] == 1
    assert load("lares", profile="tester", overrides=_LARES_UNDER_PPO)["select.allocation"] == "uniform"
    assert load("lares", profile="tester", overrides={
        "train.interaction": "independent", "train.interaction_allocator": "uniform",
        "train.seeds_per_candidate": 3})["train.seeds_per_candidate"] == 3


# ==========================================================================
# the elite exemption is by program, so apply_to changes who is constrained
# ==========================================================================


def test_apply_to_changes_who_is_constrained_when_a_member_carries_the_elites_program():
    """An exemption that tested `cand_id` alone would never fire: under
    `lares.yaml` every member is a fresh id each round while the elite is never
    re-admitted, so `apply_to: non_elite` would equal `all` and
    `reward_scaling_exempt` could never fire. The release exempts by SLOT, and an elite
    slot keeps its program verbatim; a member carrying the elite's program
    unchanged is that slot under a new id.
    """
    import numpy as np

    from bird import registry
    from bird.budget import Budget
    from bird.components import training as T
    from bird.components.population import constraint_l2_params, scaling_elite_moments
    from bird.context import Context
    from bird.state import RunState
    from bird.types import CandidateReport

    code_e = "def compute_reward(state, action):\n    return float(state[0]), {}\n"
    code_o = "def compute_reward(state, action):\n    return 2.0 * float(state[1]), {}\n"
    elite = Candidate(cand_id="c0000", reward_code=code_e, iteration=0)
    same = Candidate(cand_id="c0007", reward_code=code_e, iteration=1)   # new id, same program
    other = Candidate(cand_id="c0008", reward_code=code_o, iteration=1)

    def _ctx(apply_to):
        cfg = load("lares", profile="tester",
                   overrides={"train.elite_constraint.apply_to": apply_to})
        ctx = Context(cfg=cfg, budget=Budget())
        ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
        events = []
        ctx.event = lambda stage, **fields: events.append({"stage": stage, **fields})
        state = RunState()
        state.best = CandidateReport(cand_id=elite.cand_id, candidate=elite,
                                     result=TrainResult(cand_id=elite.cand_id, candidate=elite),
                                     fitness=1.0)
        state.policy_ref = "policy:c0000"
        state.replay_ref = "replay:test"
        return ctx, state, events

    try:
        # Eq. 4: the toggle decides whether the elite's program is constrained.
        ctx, state, _ = _ctx("non_elite")
        assert constraint_l2_params(ctx, state, same) is None, (
            "a member carrying the elite's program unchanged IS the elite; non_elite exempts it")
        assert constraint_l2_params(ctx, state, other) is not None
        ctx, state, _ = _ctx("all")
        assert constraint_l2_params(ctx, state, same) is not None, "`all` constrains it too"
        assert constraint_l2_params(ctx, state, other) is not None

        # Eq. 3: the same identity, announced with its reason.
        ctx, state, events = _ctx("non_elite")
        s0 = np.zeros(ctx.env.obs_dim)
        T._REPLAY_STORE["replay:test"] = [(s0, 0, s0, False)] * 4
        plan = scaling_elite_moments(ctx, state, same)
        assert plan["applied"] is False and "elite's reward program" in plan["reason"]
        exempt = [e for e in events if e["stage"] == "reward_scaling_exempt"]
        assert exempt and exempt[0]["cand_id"] == "c0007" and exempt[0]["elite_id"] == "c0000"
        plan = scaling_elite_moments(ctx, state, other)
        assert plan.get("reason") != exempt[0]["reason"], "a different program is not exempt"
    finally:
        T._REPLAY_STORE.pop("replay:test", None)


# ==========================================================================
# an arm is ONE seed lineage: K slices are not K seeds
# ==========================================================================


@pytest.mark.parametrize("aggregation", ["final", "max_over_checkpoints"])
def test_an_arm_is_one_seed_lineage_whose_curve_is_its_slices_concatenated(aggregation, tmp_path):
    """Each slice stamps its checkpoints with its own salted seed. A
    `_merge_slices` that pooled K slices' checkpoints as they were, beside the
    LAST slice's one-checkpoint seed row, would let stage 4 -- which groups a
    pooled curve by `seed` -- read the slices as SEEDS: `n_seeds: 5`, a
    fitness aggregated over slices most of which scored 0 (on lares tester,
    seed 0, uniform: c0000 `fitness 0.0`, `per_seed_fitness` five zeros) --
    while a reader of the row saw one checkpoint.

    An arm has one row, one `seed`, one concatenated curve on a cumulative
    axis, and the pooled curve is the same entries: `final` scores the LAST
    slice's final checkpoint, `max_over_checkpoints` the max over the whole
    arm.
    """
    import json
    entry = _entry()
    cfg = load("lares", profile="tester", overrides={
        "seed": 0, "train.interaction_allocator": "uniform",
        "train.candidate_parallelism": "sequential",
        "evaluate.fitness.checkpoint_aggregation": aggregation})
    entry.run(cfg, out_root=str(tmp_path))
    slices_per = int(cfg["train.interaction_cfg.slices_per_candidate"])
    checked = 0
    for tr_path in sorted(Path(tmp_path).rglob("candidates/iter*_*/train_result.json")):
        tr = json.loads(tr_path.read_text())
        rep = json.loads((tr_path.parent / "report.json").read_text())
        if not tr["trained"] or not tr["seed_metrics"]:
            continue
        (row,) = tr["seed_metrics"]                       # ONE lineage, one row
        curve = row["checkpoints"]
        assert row["n_slices"] == slices_per and len(row["slice_seeds"]) == slices_per
        assert row["seed"] == row["slice_seeds"][0], "the lineage seed is the first slice's"
        assert len({c["seed"] for c in curve}) == 1 and curve[0]["seed"] == row["seed"]
        assert [c["slice_index"] for c in curve] == sorted(c["slice_index"] for c in curve)
        assert {c["slice_index"] for c in curve} == set(range(slices_per))
        assert all(c["slice_seed"] == row["slice_seeds"][c["slice_index"]] for c in curve)
        rounds = [c["round"] for c in curve]
        assert rounds == sorted(rounds) and len(set(rounds)) == len(rounds), (
            "one cumulative axis, not K restarting ones")
        steps = [c["step"] for c in curve]
        assert steps == sorted(steps) and len(set(steps)) == len(steps)
        # the pooled curve is the same entries under the same single seed
        pooled = tr["checkpoints"]
        assert len(pooled) == len(curve) and {c["seed"] for c in pooled} == {row["seed"]}
        assert [c["round"] for c in pooled] == rounds
        # and stage 4 reads ONE seed whose value is the arm's, not the slices'
        assert rep["meta"]["n_seeds"] == 1, rep["meta"]
        assert len(rep["per_seed_fitness"]) == 1
        values = [c["success_rate"] for c in curve]
        expect = values[-1] if aggregation == "final" else max(values)
        assert rep["fitness"] == pytest.approx(expect), (aggregation, values, rep["fitness"])
        checked += 1
    assert checked >= 2, "too few trained arms to have checked anything"
