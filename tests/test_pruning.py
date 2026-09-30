"""`train.pruning` and `train.timeout_s` must be real on EVERY backend.

The bug this file exists to prevent is a backend that ignores both keys:

    _run_backend   `_PRUNERS` consulted, `train.timeout_s` honoured
    _sb3_run       must do the same -- `registry.get("train_backend", "sb3")`
                   reaches `_sb3_run` and never `_run_backend`

A real learner backend is what the dev and full profiles run. If only `mock`,
`tabular` and `none` honoured the keys, both would be dead on every path that
costs money and live only on the tester profile, where nothing they could save is
worth saving.

`-s train.pruning=median_stop` would then be a FABRICATED PIN: it validates, it
resolves, it is written into `config.resolved.yaml`, its bytes go into the sha256
that names the run directory -- and no code reads it. Two runs differing only in
that key would produce different run ids and identical behaviour, which is the
worst possible outcome for a framework whose one claim is that a config value IS
the method.

WHY THE TESTS BELOW ARE PARAMETRISED OVER `registry.available("train_backend")`
rather than testing `sb3` directly: the defect is not that someone writes
`_sb3_run` wrongly, it is that a second implementation of the same stage grows a
second answer to the same config key. A future backend copying `_sb3_run` would
inherit the bug; parametrising means it inherits the test instead.

WHAT "PRUNED" MEANS IN THE ARTIFACT, and why it is a third population:

    completed   trained=True,  error="",  pruned=False
    pruned      trained=True,  error="",  pruned=True   + seed_metrics[i]["pruned_at_round"]
    broken      trained=False, error=<reason>

A pruned candidate's final fitness is a LOWER BOUND -- training stopped early on
purpose -- so ranking it against a completed candidate as though the two numbers
were the same quantity is precisely the confusion `failure` / `failure_kind` exists
to prevent one level up (`bird/artifacts.py`). A timeout is deliberately on the
`broken` side of that line: the run did not finish and nobody chose to stop it.
"""

from __future__ import annotations

import inspect
import types as pytypes

import numpy as np
import pytest

from conftest import backend_param, backend_param_unimplemented

from bird import registry
from bird.budget import Budget
from bird.components import training
from bird.config import load
from bird.context import Context
from bird.types import Candidate

#: Not in the tester-tier smoke suite (heavyweight execution: a search per pruning rule across every sb3 config).
#: Deselected by `-m "not slow"`. See pyproject.toml.
pytestmark = pytest.mark.slow

#: Toy 2-D reacher: 6-D observation, 2-D continuous action, horizon 25. Small
#: enough that a real SB3 run of a few thousand steps is a couple of seconds.
_REWARD = """
def compute_reward(s, a, s2):
    import numpy as np
    o = np.asarray(s2, dtype=float)
    return float(-np.linalg.norm(o[0:2] - o[2:4]))
"""

#: `train.env_steps` for every run here. Chosen so `_sb3_run`'s chunk arithmetic
#: (`max(MIN_CHECKPOINTS, min(MAX_CHECKPOINTS, steps // 2048 or MIN_CHECKPOINTS))`)
#: lands on 5 checkpoints -- two more than the spy needs to fire at -- and so the
#: surrogate backends get enough learner rounds that a prune on checkpoint 3
#: actually skips some. At 1500 the planner backend fires on its LAST checkpoint
#: and saves nothing, which makes the budget assertion below vacuous rather than
#: false; do not shrink this to buy suite seconds.
_STEPS = 3000

#: `train.hyperparameters` for the sb3 backend under PPO. SB3's on-policy
#: `learn()` completes whole `n_steps` rollouts (2048 by default), and
#: `_sb3_run` now charges the steps the model actually took and stops when the
#: MEASURED spend reaches the budget -- so at `_STEPS` an unpinned PPO would run
#: 2048 + 2048 and write two checkpoints, not five. Before that fix the same run
#: wrote five checkpoints because it silently trained 10,240 steps and recorded
#: 3000. A rollout that divides the chunk (3000 / 5 = 600) makes `_STEPS` mean
#: 3000 here too. Injected for sb3 + ppo only; this file tests the CONTRACT,
#: not learning.
_SB3_PPO_TINY = {"n_steps": 600, "batch_size": 100}


def _backends():
    registry.load_all()
    # Every registered learner backend, read off the registry rather than a
    # literal, so a newly registered backend inherits these cases.
    return sorted(n for (k, n) in registry._REGISTRY if k == "train_backend")


def _skip_if_unavailable(name: str) -> None:
    if name == "sb3":
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("gymnasium")
    if name == "fasttd3":
        pytest.importorskip("torch")
    # `simba_v2` is the FOURTH learner backend and this arm exists because
    # REGISTERING A BACKEND CREATES CASES IN FILES THE CHANGE NEVER TOUCHES.
    # `_backends()` reads the registry, so registering `train_backend: simba_v2`
    # gives every parametrised body here a `[simba_v2]` case. Without the arm it
    # falls through to the backend, which raises "train.backend: simba_v2 needs
    # torch, which is not installed", and because this file is slow-marked the
    # default selection deselects it and cannot see the failure. `torch` and nothing else: the
    # port is torch, not a wrapper round upstream's JAX, so there is no
    # `jax`/`flax` to name and this is not the never-opening gate
    # `tests/test_no_silent_skip.py` refuses.
    if name == "simba_v2":
        pytest.importorskip("torch")

#: `train.backend: fasttd3` runs its upstream defaults otherwise -- 128 envs,
#: batch 32768, 1024-wide distributional critics, a GPU-scale set -- and this
#: file tests the CONTRACT, not learning. Injected only for that backend; the
#: others keep the config as written.
_FASTTD3_TINY = {"num_envs": 2, "batch_size": 32, "buffer_size": 64,
                 "critic_hidden_dim": 8, "actor_hidden_dim": 8, "num_atoms": 5,
                 "v_min": -10.0, "v_max": 10.0, "learning_starts": 1,
                 "compile": False}

#: `train.backend: simba_v2` runs UPSTREAM SimbaV2's defaults otherwise --
#: 512-wide hyperspherical blocks, a 101-bin distributional critic, batch 256 --
#: and this file tests the CONTRACT, not learning, for the same reason the
#: fasttd3 set above exists: the assertions are about checkpoint rows, pruning
#: decisions and budget accounting, none of which reads a layer width.
#:
#: SIZE KEYS ONLY, AND `learning_starts` IS DELIBERATELY NOT HERE. This set
#: started as `tests/test_simba_v2.py::TINY` verbatim, which also pins
#: `learning_starts: 8` and `buffer_size: 256`, and that made these cases
#: SLOWER than no injection at all -- measured on one case
#: (`test_a_tie_at_the_peak_ships_the_later_checkpoint[simba_v2]`, cold, same
#: machine, same load): 31.5 s with upstream defaults, 109.0 s with TINY verbatim,
#: 14.7 s with the size keys alone. `learning_starts` is not a size knob, it is
#: a WORK knob: upstream's default exceeds these files' whole step budget, so
#: lowering it turns a near-zero-update run into thousands of updates and buys
#: nothing an assertion here reads. Copying a constant because its name says
#: "tiny" is the trap; the keys that matter are the ones that shrink a tensor.
#: Over all 21 `[simba_v2]` cases in the three files: 455 s with no injection,
#: 195 s with this one.
_SIMBA_V2_TINY = {"batch_size": 32, "critic_hidden_dim": 16,
                  "actor_hidden_dim": 16, "critic_num_bins": 11,
                  "compile": False, "actor_num_blocks": 1,
                  "critic_num_blocks": 1}


def _ctx(**overrides):
    registry.load_all()
    overrides.setdefault("seed", 0)
    overrides.setdefault("train.env_steps", _STEPS)
    overrides.setdefault("train.seeds_per_candidate", 1)
    overrides.setdefault("evaluate.rollouts_per_candidate", 1)
    if overrides.get("train.backend") == "fasttd3":
        overrides.setdefault("train.hyperparameters", dict(_FASTTD3_TINY))
    if overrides.get("train.backend") == "simba_v2":
        overrides.setdefault("train.hyperparameters", dict(_SIMBA_V2_TINY))
    if overrides.get("train.backend") == "sb3" and \
            overrides.get("train.algorithm", "ppo") == "ppo":
        overrides.setdefault("train.hyperparameters", dict(_SB3_PPO_TINY))
    cfg = load("eureka", overrides=overrides, profile="tester")
    return Context(cfg=cfg, budget=Budget(),
                   env=registry.get("env", "toy_reacher")({}))


def _state():
    return pytypes.SimpleNamespace(restart=0, iteration=0)


def _cand(cid="c0000"):
    return Candidate(cand_id=cid, iteration=0, reward_code=_REWARD)


class _Spy:
    """A pruner that fires on the Nth checkpoint and records every call.

    Installed INTO `_PRUNERS` under a real config value, so the test exercises
    the production lookup (`_PRUNERS.get(cfg.get("train.pruning"), _prune_none)`)
    rather than a parallel path invented for the test. If a backend stops
    consulting the table, the spy is never called and the assertion fails on
    `calls`, not on a fitness number that could have moved for other reasons.
    """

    def __init__(self, fire_on: int = 3) -> None:
        self.fire_on = fire_on
        self.calls: list = []

    def __call__(self, curve, budget_frac) -> bool:
        self.calls.append((list(curve), float(budget_frac)))
        return len(self.calls) >= self.fire_on


@pytest.fixture
def spy(monkeypatch):
    s = _Spy()
    monkeypatch.setitem(training._PRUNERS, "median_stop", s)
    return s


# --------------------------------------------------------------------------
# The pruner has to be consulted at all
# --------------------------------------------------------------------------


def test_there_are_backends_to_check() -> None:
    """Guard the guard: a parametrised test over an empty list passes vacuously."""
    assert _backends()


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_every_backend_consults_the_pruner(name, spy):
    """`train.pruning` must reach the checkpoint loop of every backend."""
    _skip_if_unavailable(name)
    ctx = _ctx(**{"train.backend": name, "train.pruning": "median_stop"})
    registry.get("train_backend", name)(ctx, _state(), _cand(), 1)
    assert spy.calls, (
        f"train_backend {name!r} never consulted _PRUNERS. `train.pruning` is "
        f"therefore a declared-but-unread key on this backend: it validates, it "
        f"is written into config.resolved.yaml, it changes the run id, and it "
        f"changes nothing about the run.")


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_the_pruner_is_called_with_the_curve_and_a_budget_fraction(name, spy):
    """Both backends must pass the SAME two arguments, or `median_stop` means one
    thing on the tester profile and another on a real run.

    `budget_frac` is learning steps over the step budget and must be in (0, 1] and
    non-decreasing -- `_prune_successive_halving` gates on rung membership
    (`abs(frac - r) < 0.06`), so a fraction computed against the wrong denominator
    silently disables it rather than misfiring.
    """
    _skip_if_unavailable(name)
    ctx = _ctx(**{"train.backend": name, "train.pruning": "median_stop"})
    registry.get("train_backend", name)(ctx, _state(), _cand(), 1)

    fracs = [f for _c, f in spy.calls]
    # Upper bound is loose on purpose: `spent += learner.improve(...)` returns a
    # WHOLE round, so the last checkpoint of a surrogate backend can overshoot
    # `steps_budget` slightly. What must not happen is a fraction computed
    # against the wrong denominator, which is off by orders of magnitude.
    assert all(0.0 < f < 2.0 for f in fracs), f"{name}: budget_frac={fracs}"
    assert fracs == sorted(fracs), f"{name}: budget_frac not monotone: {fracs}"
    for i, (curve, _f) in enumerate(spy.calls):
        assert len(curve) == i + 1, (
            f"{name}: call {i} saw a curve of {len(curve)} points; the pruner is "
            f"documented to read the run's own checkpoint history so far")
        assert all(isinstance(v, float) for v in curve), f"{name}: curve is not floats"


# --------------------------------------------------------------------------
# Firing has to actually stop the training, and say so
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_pruning_shortens_the_run_and_is_recorded(name, spy):
    """The three things a fired pruner must leave behind, all of which a caller
    downstream reads for a different reason."""
    _skip_if_unavailable(name)
    ctx = _ctx(**{"train.backend": name, "train.pruning": "median_stop"})
    res = registry.get("train_backend", name)(ctx, _state(), _cand(), 1)

    assert len(spy.calls) == spy.fire_on, (
        f"{name}: pruner fired on call {spy.fire_on} but was called "
        f"{len(spy.calls)} times -- the backend ignored its return value")
    assert res.pruned is True, f"{name}: TrainResult.pruned not set"
    assert res.trained is True and not res.error, (
        f"{name}: a prune is not a failure; trained={res.trained} error={res.error!r}")
    (sm,) = res.seed_metrics
    assert sm["n_checkpoints"] == spy.fire_on, (
        f"{name}: {sm['n_checkpoints']} checkpoints written after a prune on "
        f"checkpoint {spy.fire_on} -- training continued past the decision")
    # `pruned_at_round` is the LEARNER round, not the checkpoint index: the two
    # coincide on sb3 (every chunk is a checkpoint) and do not on `_run_backend`,
    # which evaluates every `_checkpoint_stride` rounds. Pinned against the curve
    # so the field keeps meaning the same thing on both.
    assert sm["pruned_at_round"] == sm["checkpoints"][-1]["round"], (
        f"{name}: pruned_at_round={sm['pruned_at_round']!r} does not name the "
        f"last checkpoint's round {sm['checkpoints'][-1]['round']!r}")


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_a_completed_run_is_distinguishable_from_a_pruned_one(name):
    """The other half of the contract: `pruned` must be FALSE by default, or the
    flag carries no information."""
    _skip_if_unavailable(name)
    ctx = _ctx(**{"train.backend": name, "train.pruning": "none"})
    res = registry.get("train_backend", name)(ctx, _state(), _cand(), 1)
    assert res.pruned is False
    assert all(s["pruned_at_round"] is None for s in res.seed_metrics)


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_the_budget_records_steps_spent_not_steps_requested(name, spy):
    """`_sb3_run` recorded `train.env_steps` unconditionally, so a run stopped at
    12% of budget still reported 100% spent -- the one number that would show
    pruning saved anything showed no saving. Cost is a first-class output here
    (`bird/budget.py`), so an over-report is a wrong result, not a wrong log line.
    """
    _skip_if_unavailable(name)
    full = _ctx(**{"train.backend": name, "train.pruning": "none"})
    r_full = registry.get("train_backend", name)(full, _state(), _cand(), 1)
    cut = _ctx(**{"train.backend": name, "train.pruning": "median_stop"})
    r_cut = registry.get("train_backend", name)(cut, _state(), _cand(), 1)

    assert cut.budget.env_steps < full.budget.env_steps, (
        f"{name}: pruned run charged {cut.budget.env_steps} env steps against "
        f"{full.budget.env_steps} for the full run -- the budget is reporting the "
        f"request, not the spend")
    # And the budget agrees with the artifact, so the two cannot drift.
    #
    # AGAINST `env_steps_used`, NOT the seed-metric sum, and the difference is
    # the third category of env interaction. `seed_metrics[i]["env_steps"]` is
    # `spent + spent_eval` -- the training and its in-training checkpoint eval.
    # `TrainResult.env_steps_used` adds the POST-TRAINING evaluation rollouts,
    # which the budget charges because they are real interaction the machine
    # paid for. Summing the seed metrics therefore under-states what the
    # budget holds, by exactly those rollouts.
    #
    # Derived, never a literal: the term moves with
    # `evaluate.rollouts_per_candidate` and with the episode length, and a
    # hardcoded arithmetic would go stale.
    assert cut.budget.env_steps == r_cut.env_steps_used
    assert full.budget.env_steps == r_full.env_steps_used
    assert all(s["train_steps"] <= s["train_steps_requested"] for s in r_cut.seed_metrics)


# --------------------------------------------------------------------------
# train.timeout_s
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_timeout_stops_the_run_and_is_a_failure(name):
    """A wall-clock timeout is on the FAILURE side of the line: nobody chose to
    stop this run, so its fitness must not be selected on as if it were a result.

    0.001 s rather than 0: `float(cfg.get(...) or 3600)` turns a falsy 0 into the
    default, which is deliberate (an unset key must not mean "no time at all") and
    is exactly why the test cannot use it.
    """
    _skip_if_unavailable(name)
    ctx = _ctx(**{"train.backend": name, "train.timeout_s": 0.001})
    res = registry.get("train_backend", name)(ctx, _state(), _cand(), 1)
    assert res.trained is False, f"{name}: a timed-out run reported trained=True"
    assert "timeout" in res.error, f"{name}: error={res.error!r}"
    assert res.pruned is False, f"{name}: a timeout is not a prune"


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_a_generous_timeout_does_not_fire(name):
    _skip_if_unavailable(name)
    ctx = _ctx(**{"train.backend": name, "train.timeout_s": 3600.0})
    res = registry.get("train_backend", name)(ctx, _state(), _cand(), 1)
    assert res.trained is True and not res.error


# --------------------------------------------------------------------------
# The two implementations must not drift apart again
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [training._run_backend, training._sb3_run],
                         ids=["_run_backend", "_sb3_run"])
def test_both_drivers_read_both_keys_from_the_shared_table(fn):
    """Source-level, on purpose. The behavioural tests above need SB3 installed;
    this one runs on a bare `pyyaml + numpy` install, which is the configuration
    `--validate-all` and the suite are promised to work in. A regression that
    only shows up where the optional dependency exists is a regression nobody
    sees until a paid run hits it.
    """
    src = inspect.getsource(fn)
    # `_pruner_for` is the one resolver over `_PRUNERS` x `_PRUNING_FIELDS`
    # (rule x metric); indexing `_PRUNERS` directly would let a driver read the
    # rule and forget the metric, and every rule would then silently read
    # ground truth.
    assert "_pruner_for(" in src, f"{fn.__name__} does not resolve through _pruner_for"
    assert "train.timeout_s" in src, f"{fn.__name__} does not read train.timeout_s"
    assert "_prune_" not in src.replace("_prune_none", ""), (
        f"{fn.__name__} names an individual pruner; the rule must come from "
        f"_pruner_for so the two drivers cannot mean different things by the same "
        f"config value")
    # The seed summaries legitimately read `p["fitness"]` (they REPORT ground
    # truth); only the pruner call may not name the field.
    assert 'pruner([p["fitness"]' not in src, (
        f"{fn.__name__} hands the pruner the ground-truth field by name; the field "
        f"must come from train.pruning_metric via _pruner_for")
    assert "pruner([p[prune_field]" in src, (
        f"{fn.__name__} does not hand the pruner the metric-selected field")


def test_the_default_is_still_none():
    """No published config may change behaviour because of any of this."""
    cfg = load("eureka", profile="tester")
    assert cfg["train.pruning"] == "none"
    assert training._PRUNERS["none"]([0.0, 1.0, 0.0, 1.0], 0.5) is False


def test_pruning_is_deterministic_and_consumes_no_randomness():
    """`tests/test_parallelism.py` pins `parallel` bit-identical to `sequential`.
    That only holds if the prune decision is a pure function of the curve -- a
    pruner that drew from an RNG would desynchronise a forked worker from the
    parent's stream and change every fitness downstream of it.
    """
    for name, fn in training._PRUNERS.items():
        curve = [0.4, 0.9, 0.1, 0.2, 0.15]
        first = fn(curve, 0.25)
        assert all(fn(curve, 0.25) is first for _ in range(5)), (
            f"pruner {name!r} is not a pure function of (curve, budget_frac)")
        assert isinstance(first, (bool, np.bool_))


# --------------------------------------------------------------------------
# A timeout must never fire on a training that FINISHED
# --------------------------------------------------------------------------
#
# A 1M-step seed pinned at `train.timeout_s: 21600` costs ~18,300 s at a
# measured 8-way-parallel rate of 54.78 steps/s per worker -- an 18% margin,
# before any slowdown on shared hardware. Honouring the key in `_sb3_run`
# therefore turns such a pin into a live kill switch with almost no
# headroom, and the FIRST thing to get right is that crossing the deadline on the
# LAST chunk must not fail a run that has already spent its whole budget. There
# is nothing left to abandon at that point; failing it discards a finished 5-hour
# training and reports the candidate as broken.
#
# A timeout that fires on a healthy training is worse than no timeout, because
# the run completes, reports a fitness off a truncated curve, and looks normal.


def _checkpoint_clock(monkeypatch, spy_slot: str = "median_stop"):
    """A clock that advances ONE unit per checkpoint, and nothing else.

    Wall-clock tests of a wall-clock guard are the flaky kind, and the obvious
    monkeypatch is worse than flaky -- it is VACUOUS. Replacing
    `time.monotonic` with `lambda: real() + K` shifts `seed_t0` by exactly K
    too, so `time.monotonic() - seed_t0` is unchanged and the deadline is never
    crossed. That version of this test passed with the guard removed.

    So the offset is advanced by the PRUNER, which both drivers call once per
    checkpoint immediately before they consult the deadline. `seed_t0` is taken
    at offset 0; the check after checkpoint `i` therefore sees an elapsed time
    of exactly `i` units, whatever the machine is doing.
    """
    offset = [0.0]

    def clock() -> float:
        # Strictly increasing, but by a microsecond -- REAL elapsed time must
        # not leak in. `real() + offset` fails here: sb3 genuinely takes ~5 s
        # for this run, which on a 5-checkpoint deadline is another whole unit
        # and lands the crossing a checkpoint early, i.e. it tests the opposite
        # property from the one named on the tin.
        offset[0] += 1e-6
        return offset[0]

    monkeypatch.setattr(training.time, "monotonic", clock)

    def tick(curve, budget_frac) -> bool:
        offset[0] += 1.0
        return False

    monkeypatch.setitem(training._PRUNERS, spy_slot, tick)
    return offset


def _n_checkpoints(name: str) -> int:
    """How many checkpoints this backend writes for `_STEPS`, measured rather
    than derived -- the three surrogate learners disagree on rounds per
    checkpoint and `_sb3_run` chunks differently again."""
    ctx = _ctx(**{"train.backend": name, "train.pruning": "none"})
    res = registry.get("train_backend", name)(ctx, _state(), _cand(), 1)
    return len(res.checkpoints)


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_a_deadline_crossed_on_the_last_checkpoint_does_not_fail_the_run(name, monkeypatch):
    """The whole budget is spent, and only THEN is the deadline past.

    A 1M-step seed pinned at `train.timeout_s: 21600` costs ~18,300 s at a
    measured 8-way-parallel rate -- an 18% margin before any slowdown on shared
    hardware. The likeliest way that bites is the deadline
    landing on the final chunk, where there is nothing left to abandon: failing
    there discards a COMPLETE five-hour training and reports its candidate as
    broken. `_sb3_run`'s chunk loop is bounded by `range`, so unlike
    `_run_backend` it does not get this for free.
    """
    _skip_if_unavailable(name)
    n = _n_checkpoints(name)
    assert n >= 3, f"{name}: only {n} checkpoints; the test cannot place a deadline"
    _checkpoint_clock(monkeypatch)
    ctx = _ctx(**{"train.backend": name, "train.pruning": "median_stop",
                  "train.timeout_s": n - 0.5})
    res = registry.get("train_backend", name)(ctx, _state(), _cand(), 1)
    assert res.trained is True and res.timed_out is False and not res.error, (
        f"{name}: the deadline fired after the last of {n} checkpoints, with the "
        f"step budget already spent. trained={res.trained} error={res.error!r}")
    assert len(res.checkpoints) == n, f"{name}: the run was cut short anyway"


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_the_same_clock_one_checkpoint_earlier_DOES_fire(name, monkeypatch):
    """Guard the guard above. With the deadline one checkpoint sooner the
    timeout must fire -- otherwise the previous test is passing because the
    mechanism is dead, which is exactly the state this whole file was written
    to end."""
    _skip_if_unavailable(name)
    n = _n_checkpoints(name)
    _checkpoint_clock(monkeypatch)
    ctx = _ctx(**{"train.backend": name, "train.pruning": "median_stop",
                  "train.timeout_s": n - 1.5})
    res = registry.get("train_backend", name)(ctx, _state(), _cand(), 1)
    assert res.timed_out is True and res.trained is False, (
        f"{name}: deadline at {n - 1.5} units over {n} checkpoints did not fire")
    assert len(res.checkpoints) < n


@pytest.mark.parametrize("name", [backend_param_unimplemented(n) for n in _backends()])
def test_a_timeout_is_flagged_not_just_described(name):
    """`error` is a string and every other failure fills it too. A wedged node
    and a reward that raises on the first transition both arrive as
    `trained=False` with prose, and the remedies are opposite -- one wants a
    re-run, the other wants a different reward. `timed_out` is the machine
    readable half, and it must reach the artifact.
    """
    _skip_if_unavailable(name)
    ctx = _ctx(**{"train.backend": name, "train.timeout_s": 0.001})
    res = registry.get("train_backend", name)(ctx, _state(), _cand(), 1)
    assert res.timed_out is True
    assert res.trained is False and "timeout" in res.error
    assert any(s["timed_out"] for s in res.seed_metrics)
    assert res.pruned is False


def test_the_run_summary_keeps_the_four_populations_apart():
    """`summarise_reports` printed SKIPPED for anything with `trained=False`, so
    a timed-out candidate read as CARD deliberately not training one. Four
    outcomes, four marks."""
    from bird import artifacts
    from bird.types import CandidateReport, TrainResult

    def report(**kw):
        cand = _cand()
        res = TrainResult(cand_id=cand.cand_id, candidate=cand, **kw)
        return CandidateReport(cand_id=cand.cand_id, candidate=cand, result=res,
                               fitness=0.5)

    text = artifacts.summarise_reports([
        report(),
        report(pruned=True),
        report(trained=False, skip_reason="card"),
        report(trained=False, error="boom"),
        report(trained=False, error="timeout after 1s", timed_out=True),
    ])
    for mark in ("trained", "pruned", "SKIPPED", "FAILED", "TIMEOUT"):
        assert mark in text, f"{mark!r} missing from:\n{text}"


# --------------------------------------------------------------------------
# ... and no shipped config may pin a deadline SHORTER than its own training
# --------------------------------------------------------------------------
#
# Because `_sb3_run` honours `train.timeout_s`, every pin in `configs/` is a kill
# switch rather than a decoration. A 300 s pin (commented as "900 s is an XLand
# wall-clock cap") inherited against 100,000 Meta-World env steps would report
# every candidate FAILED while the whole suite stays green, since nothing else
# compares the two numbers.
#
# The check is a ratio rather than a measurement because a wall clock cannot be
# measured in CI: the constant encodes `max(3600, steps / 40.0 + 1800)` so the
# shipped configs and the guard cannot disagree about what "too slow" means.

#: Deliberately pessimistic: ~95 Meta-World steps/s measured on a workstation with
#: torch free to use the whole machine, against a worker pinned to ONE thread
#: by `_pin_one_thread`. Being generous costs nothing -- the guard only fires on
#: a training that has genuinely stopped making progress.
_STEPS_PER_S_FLOOR = 40.0

#: Checkpoint evaluation is charged to the same wall clock and is NOT in
#: `train.env_steps`: `_sb3_run` evaluates 3 episodes at each of up to
#: MAX_CHECKPOINTS chunks, which is ~30,000 extra env steps on a 100,000-step
#: Meta-World seed and ~27% on a 20,000-step Pendulum one. A deadline sized
#: against the learning steps alone is ~30% smaller than it reads.
_EVAL_OVERHEAD = 1.3


def _sb3_configs() -> list:
    """Every shipped (method, profile) point that will run this code path.

    A point is a PAIR, not a file. Tier directories that pin
    `train.backend: sb3` do not exist: `configs/methods/*.yaml` are the methods
    (and `configs/*.yaml` the ERA recipes),
    `configs/_profiles/*.yaml` say how expensively each one runs,
    and `train.backend` / `train.env_steps` / `train.timeout_s` are all in
    `PROFILE_KEY_PREFIXES`, so the PROFILE decides all three. Enumerating files
    would therefore find no sb3 config at all -- which is the empty-parametrize
    shape this file exists to refuse, hence the guard below.
    """
    from bird.config import available_profiles
    from conftest import PAPER_CONFIGS
    methods = [p.stem for p in PAPER_CONFIGS]
    out = []
    for name in methods:
        for profile in sorted(available_profiles()):
            try:
                cfg = load(name, profile=profile)
            except Exception:          # `--validate-all` owns that failure
                continue
            if cfg["train.backend"] == "sb3":
                out.append((name, profile))
    return out


_SB3_CONFIGS = _sb3_configs()
_SB3_IDS = [f"{n}@{p}" for n, p in _SB3_CONFIGS]


def test_the_corpus_has_sb3_configs_to_check():
    """Guard the guard: the enumeration finding nothing is a green parametrised
    test over an empty list, which is the exact shape of the bug this file is
    about -- and it is a live risk because no FILE says `sb3`, only a profile."""
    assert len(_SB3_CONFIGS) > 20, _SB3_CONFIGS


@pytest.mark.parametrize("name,profile", _SB3_CONFIGS, ids=_SB3_IDS)
def test_no_sb3_config_pins_a_deadline_shorter_than_its_own_training(name, profile):
    """`train.timeout_s` is a hung-training guard; below the training it guards
    it is a kill switch that fires on every healthy seed.

    `_sb3_run` sets `seed_error`, `timed_out=True` and then `trained=False`,
    so the candidate is reported BROKEN -- not slow, not pruned. At 100,000
    Meta-World steps against 300 s that fires around chunk 3-6 of 20 on every
    seed of every candidate.

    The margin is printed, not just asserted: a tight point can sit at ~1.1x
    once checkpoint evaluation is counted, which passes and is worth seeing.
    """
    cfg = load(name, profile=profile)
    steps = int(cfg["train.env_steps"] or 0)
    timeout = float(cfg["train.timeout_s"])
    need = _EVAL_OVERHEAD * steps / _STEPS_PER_S_FLOOR
    print(f"  {name}@{profile:<24} {steps:>9,} steps  timeout {timeout:>7.0f}s  "
          f"need ~{need:>8.0f}s  margin {timeout / need if need else float('inf'):5.2f}x")
    assert need < timeout, (
        f"{name}@{profile}: train.timeout_s is {timeout:g} s against a training "
        f"that needs "
        f"~{need:.0f} s ({steps:,} steps at a pessimistic {_STEPS_PER_S_FLOOR:g} "
        f"steps/s, x{_EVAL_OVERHEAD} for checkpoint evaluation). _sb3_run honours "
        f"the key, so every seed of every candidate is cut short and reported as "
        f"FAILED. Raise the pin, or drop it to inherit the 3600 s default.")
