"""`_ippo_reward_env` -- the candidate's reward inside upstream's trainer.

Offline, with a stub standing in for upstream's wrapper. The properties here
all fail SILENTLY if they are wrong: a reward wrapper in the wrong position is
never called at all, one that adds instead of overwriting puts the held-out
preference reward into the gradient, and a binding that is not restored
changes every later training in the process.

The decisive case: an inner env paying task reward T, a partner
preference reward P, and a candidate returning C. What arrives at the
training call site must be exactly C -- not C + P, and not T.
"""
from __future__ import annotations

import pytest

from bird.components import _ippo_reward_env as M

T_TASK, P_PREF, C_CAND = 2.0, 0.5, 7.0


class _FakeLoadAgentWrapper:
    """Upstream's shape: inner `step_env`, then `+ pref_reward` on every agent.

    Mirrors `aht.py:872-881` -- the inner call is `step_env`, which is why a
    wrapper placed inside and overriding `step` is never reached.
    """

    agents = ["robot"]

    def __init__(self, env):
        self._env = env

    @classmethod
    def load_from_zoo(cls, env, zoo, uuids):
        return cls(env)

    def step(self, key, state, actions, reset_state=None):
        obs, nxt, rewards, dones, infos = self._env.step_env(key, state, actions)
        rewards = {a: rewards[a] + P_PREF for a in rewards}
        return obs, nxt, rewards, dones, infos


class _FakeInner:
    def step_env(self, key, state, actions):
        return ({"robot": 0}, ("state", state[1] + 1),
                {"robot": T_TASK, "human": T_TASK, "__all__": T_TASK},
                {"__all__": False}, {})


def _row_fn(s, a, s_next):
    return C_CAND, {"c": C_CAND}


def _to_row(state, prev=None):
    # two-arg: the row's tool velocity is a finite difference
    return state[1]


def _wrapped():
    cls = M.candidate_reward_wrapper_class(_FakeLoadAgentWrapper, _row_fn, _to_row)
    return cls.load_from_zoo(_FakeInner(), None, None)


def test_the_reward_at_the_training_call_site_is_exactly_the_candidate():
    """The decisive case: C, never C+P, never T."""
    _obs, _st, rewards, _d, _i = _wrapped().step(None, ("state", 0), {"robot": 0.0})
    assert rewards["robot"] == pytest.approx(C_CAND), rewards
    assert rewards["robot"] != pytest.approx(C_CAND + P_PREF), (
        "the candidate reward was ADDED to the partner's preference reward; "
        "the robot would train on candidate + R_pref and the held-out metric "
        "would be inside the gradient")
    assert rewards["robot"] != pytest.approx(T_TASK + P_PREF), (
        "the wrapper did not take effect at all -- upstream's task reward "
        "reached the training call site")


def test_the_team_scalar_follows_the_ego():
    """`__all__` must not keep upstream's sum, or a logger reports a reward
    nobody was trained on."""
    _o, _s, rewards, _d, _i = _wrapped().step(None, ("state", 0), {"robot": 0.0})
    assert rewards["__all__"] == pytest.approx(C_CAND)


def test_the_partner_keeps_its_preference_reward():
    """The partner is frozen, and changing what it is paid would be a change
    to the AHT protocol smuggled in through a reward wrapper."""
    _o, _s, rewards, _d, _i = _wrapped().step(None, ("state", 0), {"robot": 0.0})
    assert rewards["human"] == pytest.approx(T_TASK + P_PREF)


def test_an_inner_wrapper_would_never_be_called():
    """Pins WHY the wrapper is the outer one, so the placement cannot be
    'simplified' back later.

    Upstream calls `self._env.step_env`, so a subclass overriding `step` on
    the INNER env is dead code. If upstream ever changes that call to `step`,
    this test fails and the placement argument has to be re-read.
    """
    called = []

    class _InnerOverridingStep(_FakeInner):
        def step(self, *a, **k):            # pragma: no cover - must not run
            called.append(True)
            raise AssertionError("inner step was called")

    _FakeLoadAgentWrapper(_InnerOverridingStep()).step(None, ("state", 0), {"robot": 0.0})
    assert not called


def test_the_binding_is_restored():
    import types

    mod = types.ModuleType("trainer")
    mod.LoadAgentWrapper = _FakeLoadAgentWrapper
    sub = M.candidate_reward_wrapper_class(_FakeLoadAgentWrapper, _row_fn, _to_row)
    with M.candidate_reward_binding(mod, sub):
        assert mod.LoadAgentWrapper is sub
    assert mod.LoadAgentWrapper is _FakeLoadAgentWrapper


def test_the_binding_is_restored_on_an_exception():
    """A trainer module left bound would change every later training AND the
    eval env, and the eval would then agree with training for the wrong
    reason."""
    import types

    mod = types.ModuleType("trainer")
    mod.LoadAgentWrapper = _FakeLoadAgentWrapper
    sub = M.candidate_reward_wrapper_class(_FakeLoadAgentWrapper, _row_fn, _to_row)
    with pytest.raises(RuntimeError):
        with M.candidate_reward_binding(mod, sub):
            raise RuntimeError("boom")
    assert mod.LoadAgentWrapper is _FakeLoadAgentWrapper


# -- the stack that the trainer actually binds ------------------------------

def test_the_bound_class_stacks_both_overrides():
    """ONE class carrying the reward override AND the preference-obs override.

    `make_train` binds a single name at ippo_ff_nps.py:259, so two sibling
    LoadAgentWrapper subclasses would mean whichever is bound second silently
    does nothing. The reward override alone, stacked on the BARE wrapper,
    leaves upstream's `_append_pref_to_obs` appending the 7 held-out
    preference dims to every agent -- including the robot, whose TRAINING
    observation would then carry the metric it is being measured against.
    That is the leak on the path that matters, and the eval-path adapter
    would have stayed clean and shown nothing.

    Structural, so it runs offline: the assertion is that both overrides are
    reachable on one MRO, which is what "stacked" means. The width acceptance
    (robot 42 unaugmented, human 49) is the device-side half and belongs with
    the training-path probe.
    """
    class _Base:
        def _append_pref_to_obs(self, obs, ag_idx):       # the override to keep
            return obs

        def step(self, key, state, actions, reset_state=None):
            return ({}, state, {"robot": 0.0}, {}, {})

    stacked = M.candidate_reward_wrapper_class(_Base, _row_fn, _to_row)
    assert "step" in stacked.__dict__, "the reward override is not on the bound class"
    assert any("_append_pref_to_obs" in c.__dict__ for c in stacked.__mro__), (
        "the preference-observation override is not reachable from the bound "
        "class: the robot's training observation would carry the held-out "
        "preference dims")
    assert stacked.__mro__[1] is _Base


# -- the guard that the candidate's reward actually bound --------------------
#
# A GUARD AGAINST A SILENT FAILURE THAT ITSELF HAS NO TEST IS A CLAIM, NOT A
# CHECK. Both directions are here deliberately: a guard that refuses
# everything also "catches" the bug, and nothing in a negative-only test set
# would ever say so.


class _Inner:
    agents = ["robot"]


class _Outer:
    """A wrapper chain link, as upstream's wrappers are: inner env on `_env`."""

    def __init__(self, inner):
        self._env = inner


def test_an_unbound_env_is_refused():
    """NEGATIVE. The failure is silent, so the refusal is the whole guard.

    Without it the trainer builds upstream's bare wrapper, trains on the task
    reward plus the partner's preference reward, and finishes with a plausible
    curve nothing downstream can distinguish from a candidate that happened to
    score like upstream's reward.
    """
    with pytest.raises(RuntimeError, match="no candidate-reward wrapper"):
        M.assert_candidate_rewarded(_Outer(_Inner()))


def test_a_properly_bound_env_passes():
    """POSITIVE. A guard that refused everything would pass the negative test.

    Built through the real factory and wrapped the way `make_train` wraps it
    (`LogWrapper(LoadAgentWrapper(base))`), so the guard is exercised at the
    depth it actually runs at rather than on the wrapper itself.
    """
    cls = M.candidate_reward_wrapper_class(_FakeLoadAgentWrapper, _row_fn, _to_row)
    bound = cls.load_from_zoo(_FakeInner(), None, None)
    M.assert_candidate_rewarded(_Outer(bound))          # LogWrapper's position
    M.assert_candidate_rewarded(bound)                  # and undecorated


def test_the_guard_reads_a_marker_not_a_class_name():
    """No coupling to the class name: renaming the class must not blind it.

    The marker is an attribute the factory sets and the guard reads. If this
    is ever reverted to a `__name__.startswith(...)` match, this test fails --
    which is the point, because that failure mode is invisible at runtime.
    """
    cls = M.candidate_reward_wrapper_class(_FakeLoadAgentWrapper, _row_fn, _to_row)
    assert cls.bird_candidate_rewarded is True
    cls.__name__ = "SomethingElseEntirely"
    M.assert_candidate_rewarded(cls.load_from_zoo(_FakeInner(), None, None))


def test_the_binding_helper_verifies_the_rebind_took():
    """The other half: the name `make_train` resolves must really be ours."""
    import types

    mod = types.ModuleType("trainer")
    mod.LoadAgentWrapper = _FakeLoadAgentWrapper
    sub = M.candidate_reward_wrapper_class(_FakeLoadAgentWrapper, _row_fn, _to_row)
    with M.candidate_reward_binding(mod, sub):
        assert mod.LoadAgentWrapper is sub
    assert mod.LoadAgentWrapper is _FakeLoadAgentWrapper


def test_gt_return_on_the_REAL_EvalInfo_shapes():
    """`(T, n_episodes)` arrays indexed out of per-agent DICTS.

    A FIXTURE OF FOUR-ELEMENT 1-D ARRAYS WOULD PASS while a reduction returned
    0.0 for every seed. `EvalInfo.reward` and `.done` are per-agent dicts
    (`ippo_ff_nps.py:857/860`), not the batchified arrays the train path
    builds at `:443`: `np.asarray(dict)` is a 0-d OBJECT array and
    `.astype(bool)` on it is `True`, so a naive mask drops every step and the
    return is zero -- a measurement of nothing, reported as a measurement.
    A hand-made 1-D fixture cannot express a dict, so no amount of arithmetic
    checking on it can fail.

    So the fixture is the real shape, and passing a dict is asserted to
    RAISE rather than silently reduce.
    """
    import numpy as np

    from bird.components._ippo_train import aligned_task_return

    T, n_eps, task, pref_paid = 5, 3, 2.0, 0.5
    rew_dict = {"robot": np.full((T, n_eps), task + pref_paid),
                "human": np.full((T, n_eps), task + pref_paid),
                "__all__": np.full((T, n_eps), task + pref_paid)}
    pref = np.full((T, n_eps), pref_paid)
    pref[-1, :] = 0.0                       # zero-padded on the done step
    done_dict = {"__all__": np.zeros((T, n_eps), bool)}
    done_dict["__all__"][-1, :] = True

    total, n = aligned_task_return(rew_dict["robot"], pref, done_dict["__all__"])
    assert total == pytest.approx(task * (T - 1)), (
        f"expected the per-episode task return {task * (T - 1)}, got {total}")
    assert n == n_eps, "one done step per episode column"

    # A DICT MUST RAISE. This is the shape that returned 0.0.
    with pytest.raises(TypeError, match="dict"):
        aligned_task_return(rew_dict, pref, done_dict["__all__"])
    with pytest.raises(TypeError, match="dict"):
        aligned_task_return(rew_dict["robot"], pref, done_dict)


def test_aligned_task_return_masks_both_series_not_just_the_reward():
    """Kills the mutant that masks one term: the pref must be dropped too."""
    import numpy as np

    from bird.components._ippo_train import aligned_task_return

    rewards = np.array([3.0, 3.0, 9.0])       # the done step pays much more
    pref = np.array([1.0, 1.0, 1.0])          # and its pref is NOT zeroed here
    done = np.array([False, False, True])
    total, n = aligned_task_return(rewards, pref, done)
    assert total == pytest.approx(4.0), "both series must drop the same step"
    assert n == 1
    # reward-only masking would give 6.0 - 3.0 = 3.0; pref-only would give 12.0
    assert total != pytest.approx(3.0) and total != pytest.approx(12.0)


def test_aligned_task_return_keeps_everything_when_no_step_is_done():
    """Kills `keep = ones_like(...)` only in company with the tests above, and
    pins that a done-free episode loses nothing."""
    import numpy as np

    from bird.components._ippo_train import aligned_task_return

    total, n = aligned_task_return(np.array([1.0, 2.0]), np.array([0.5, 0.5]),
                                   np.array([False, False]))
    assert total == pytest.approx(2.0) and n == 0


def test_the_eval_binding_carries_the_obs_override_but_not_the_reward_one():
    """The two overrides are separable, and eval needs exactly one of them.

    Binding NOTHING for eval fails on a device: upstream's bare
    `LoadAgentWrapper` appends the 7 preference dims to every agent, so a
    49-wide robot observation reaches a network trained on 42 and raises
    `ScopeParamShapeError: "kernel" ... expected (49, 128), existing
    (42, 128)`. Binding the REWARD override instead would be the opposite
    error -- `gt_return` reads `infos.reward`, so a candidate-rewarded eval
    env would have the run scoring itself.

    Structural, so it runs offline: the eval class must carry
    `_append_pref_to_obs` and must NOT carry the reward `step` override.
    """
    from bird.components.assistax_ppo import _human_only_pref_obs_class

    pytest.importorskip("assistax")
    eval_cls = _human_only_pref_obs_class()
    assert any("_append_pref_to_obs" in c.__dict__ for c in eval_cls.__mro__), (
        "the eval env would append the held-out dims to the robot")
    assert not getattr(eval_cls, "bird_candidate_rewarded", False), (
        "the eval env is candidate-rewarded: gt_return would score the run "
        "on the candidate's own reward instead of upstream's")

    train_cls = M.candidate_reward_wrapper_class(eval_cls, _row_fn, _to_row)
    assert train_cls.bird_candidate_rewarded is True
    assert any("_append_pref_to_obs" in c.__dict__ for c in train_cls.__mro__)


def test_the_eval_call_site_binds_the_obs_class_by_name():
    """The WIRING, not just the class choice -- read off the source.

    The test above pins which class is correct; it cannot see which class
    `_evaluate` actually passes. Swapping that one argument to `wrapper_cls`
    reintroduces the whole defect (the run scoring itself on the candidate's
    reward) and survives every other check here, because nothing offline
    calls `_evaluate` -- it needs jax, a zoo and a trained policy.

    So this asserts on the AST: inside `_evaluate`, the
    `candidate_reward_binding` call's second argument must be a call to
    `_human_only_pref_obs_class`. Structural rather than behavioural, and
    that is stated rather than implied -- the behaviour is the device run's
    to confirm.
    """
    import ast
    import inspect

    from bird.components import _ippo_train

    tree = ast.parse(inspect.getsource(_ippo_train))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_evaluate")
    binds = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "candidate_reward_binding"]
    assert len(binds) == 1, f"expected one binding in _evaluate, found {len(binds)}"
    arg = binds[0].args[1]
    # The eval class is the preference-observation subclass
    # WRAPPED by `row_logging_class` (the adapter's row rides in `infos` so the
    # rollout comes home; `tests/test_ippo_eval_rollouts.py`). Still not the
    # candidate-rewarded wrapper: `row_logging_class` touches `infos` only, and
    # `test_the_eval_binding_carries_the_obs_override_but_not_the_reward_one`
    # holds the built class to that.
    assert isinstance(arg, ast.Call) and getattr(arg.func, "attr", None) == \
        "row_logging_class", (
        "the eval binding must be row_logging_class(_human_only_pref_obs_class(), ...); "
        f"got {ast.dump(arg)[:80]}. A candidate-rewarded eval env makes "
        "gt_return the run scoring itself.")
    inner = arg.args[0]
    assert isinstance(inner, ast.Call) and getattr(inner.func, "id", None) == \
        "_human_only_pref_obs_class", (
        "the row logger must wrap the preference-observation subclass alone; "
        f"got {ast.dump(inner)[:80]}")


def test_the_two_markers_are_distinct_and_each_guard_reads_its_own():
    """Train env: both overrides. Eval env: the observation one only.

    One marker cannot express both requirements, and the obvious shortcut
    -- run `assert_candidate_rewarded` on the eval env -- would
    refuse exactly the correct configuration, because the reward override is
    the thing the eval env must NOT have. Pinned so the two guards cannot be
    collapsed into one later.
    """
    from bird.components.assistax_ppo import _human_only_pref_obs_class

    # OFFLINE, via the `base` seam. An `importorskip` here would skip in every
    # CI selector, and two independent marker mutants (drop the obs marker;
    # set the reward marker on the eval class) would survive a green suite
    # because the only test asserting those cells never ran.
    eval_cls = _human_only_pref_obs_class(base=_FakeLoadAgentWrapper)
    train_cls = M.candidate_reward_wrapper_class(eval_cls, _row_fn, _to_row)

    assert eval_cls.bird_human_only_pref_obs is True
    assert not getattr(eval_cls, "bird_candidate_rewarded", False)
    assert train_cls.bird_candidate_rewarded is True
    assert train_cls.bird_human_only_pref_obs is True      # inherited

    M.assert_human_only_pref_obs(eval_cls)                 # eval: passes
    M.assert_human_only_pref_obs(train_cls)                # train: also has it
    M.assert_candidate_rewarded(train_cls)                 # train: rewarded
    with pytest.raises(RuntimeError, match="no candidate-reward wrapper"):
        M.assert_candidate_rewarded(eval_cls)              # eval: correctly not


def test_the_eval_call_site_asserts_the_stack_not_only_the_width():
    """The width equality is corroboration; the stack check is the property."""
    import ast
    import inspect

    from bird.components import _ippo_train

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(_ippo_train)))
              if isinstance(n, ast.FunctionDef) and n.name == "_evaluate")
    called = {getattr(n.func, "attr", None) for n in ast.walk(fn)
              if isinstance(n, ast.Call)}
    assert "assert_human_only_pref_obs" in called, (
        "_evaluate checks only the observation WIDTH, which passes for any "
        "wrapper stack with a coincidentally equal width -- and a matching "
        "width would leak the held-out dims silently")


def test_the_eval_never_logs_the_pipeline_state():
    """No MJX pipeline state in the eval scan's carry. A STRUCTURAL pin.

    Logging `EvalInfo.env_state` keeps the whole `LoadAgentState` -- the
    entire MJX pipeline state -- for every step of every eval episode. XLA
    reports 18.2 GB of program I/O, cannot rematerialize below 16.9 GiB, and
    the eval OOMs allocating 4.14 GiB on a 23 GB L4. Reading `env_state`
    because it is logged by default treats "already logged" as "free".

    So this asserts on the source: `_evaluate` must construct an
    `EvalInfoLogConfig` with `env_state=False`, and must not read
    `infos.env_state`. A structural pin, so the cheap-looking read cannot
    come back -- a memory regression of this shape
    shows up as an OOM on a device, which is the most expensive place to
    find it and the one place no offline test looks.
    """
    import ast
    import inspect

    from bird.components import _ippo_train

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(_ippo_train)))
              if isinstance(n, ast.FunctionDef) and n.name == "_evaluate")

    cfgs = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
            and getattr(n.func, "attr", None) == "EvalInfoLogConfig"]
    assert cfgs, "_evaluate does not pass an EvalInfoLogConfig, so env_state is ON"
    off = [k for c in cfgs for k in c.keywords
           if k.arg == "env_state" and getattr(k.value, "value", None) is False]
    assert off, "EvalInfoLogConfig is passed without env_state=False"

    reads = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "env_state" not in reads, (
        "_evaluate reads infos.env_state; with logging off that is None, and "
        "with logging on it is the 18 GB")


def test_the_seed_row_reaches_the_fitness_reader_that_actually_reads_it():
    """Consume the PRODUCER's output, not a hand-built row.

    `fitness_native_reward` reads `NATIVE_REWARD_KEYS` out of
    `seed_metrics[i]["checkpoints"]`, NOT off the seed row -- so a correct
    `gt_return` on the row scores nothing under
    `evaluate.fitness.source: native`. This builds a real `SeedOutcome`
    through `checkpoint_row`, folds it with `build_train_result`, and asserts
    the key is where the reader looks. Consuming the producer's output is
    what makes it catch this rather than a stand-in dict that happens to have
    the right shape.
    """
    from bird.components import _ippo_backend as B
    from bird.native_signal import NATIVE_REWARD_KEYS
    from bird.types import Candidate

    row = B.checkpoint_row(step=196608.0, round_=0, seed=0, fitness=0.25,
                           success_rate=0.0, reward_return=12.0, gt_return=558.61)
    outcome = B.SeedOutcome(seed=0, checkpoints=[row], gt_return=558.61)
    result = B.build_train_result(Candidate(cand_id="c0", iteration=0, reward_code="x"), [outcome])

    assert result.seed_metrics, "no seed rows produced"
    cps = result.seed_metrics[0].get("checkpoints")
    assert cps, "seed row carries no checkpoints; native fitness reads nothing"
    for key in NATIVE_REWARD_KEYS:
        assert key in cps[-1], f"{key} missing from the checkpoint the reader uses"
    assert cps[-1]["gt_return"] == pytest.approx(558.61)


def test_the_built_payload_validates_in_episodes_first_orientation():
    """The BUILT payload through the real validator, not a shape assertion.

    `eval_rollout_payload` / `_as_2d` take `(n_episodes, T)`. Feeding the
    rollout's native `(T, n_episodes)` makes `n_episodes` read as 1000 and
    `validate_eval_payload` raises thirteen problems -- last_contact_force 1
    vs 1000, partner_id 0 vs 1000, ten missing `partner.*` fields -- all
    downstream of one transpose.

    Both orientations are built here and passed to the real validator, so
    the test states which one is correct rather than asserting a `.shape`
    that a future seam could satisfy while still being transposed.
    """
    import numpy as np

    from bird.components import _ippo_backend as B

    n_eps, T = 3, 7
    kw = dict(partner_id=[f"uuid-{i}" for i in range(n_eps)],
              last_contact_force=np.zeros(n_eps),
              # Distinct min/max: the validator rejects a degenerate range,
              # correctly -- a zero-width gaussian window is not a preference.
              partner={f: np.full(n_eps, 0.9 if f.endswith("_max") else 0.2)
                       for f in B.PARTNER_FIELDS},
              task="scratchitch", seed=0)

    good = B.eval_rollout_payload(tool_speed=np.zeros((n_eps, T)),
                                  contact_force=np.zeros((n_eps, T)), **kw)
    assert not B.validate_eval_payload(good), B.validate_eval_payload(good)

    # The builder validates unconditionally (there is no opt-out), so a
    # transposed payload is refused AT BUILD rather than returned for a
    # consumer to reject.
    with pytest.raises(B.EvalPayloadError):
        B.eval_rollout_payload(tool_speed=np.zeros((T, n_eps)),
                               contact_force=np.zeros((T, n_eps)), **kw)


# -- producer to consumer ---------------------------------------------------
#
# Each of these starts at what the backend BUILDS and finishes at what the
# harness READS. Every defect this file pins survives a test that stops in
# the middle: a reduction called on hand-made arrays, a seed row asserted
# without the reader, a payload's shape assumed rather than validated.


def _seed_outcome(*, payload, r_task, gt_return, trained_steps=196608.0):
    """Exactly what `_ippo_train.run` builds, including the empty-checkpoint path."""
    from bird.components import _ippo_backend as B

    checkpoints = []
    if isinstance(payload, dict):
        checkpoints = [B.checkpoint_row(
            step=trained_steps, round_=0, seed=0,
            fitness=float(payload.get("fitness", 0.0)), success_rate=0.0,
            reward_return=float(r_task) if r_task is not None else 0.0,
            gt_return=gt_return)]
    return B.SeedOutcome(seed=0, checkpoints=checkpoints, gt_return=gt_return,
                         r_task=r_task, eval_payload=payload)


def test_an_absent_payload_yields_an_unscored_seed_not_a_zero():
    """Follow the None to the READER, not just to the row.

    `else 0.0` would score an unscored seed as zero. Passing None into
    `checkpoint_row` instead would only move the failure -- it coerces
    `fitness` into four aliases with `float()` -- so an unscored seed emits
    NO checkpoint row, and the seed reads unscored rather than zero.
    """
    from bird.components import _ippo_backend as B
    from bird.types import Candidate

    out = _seed_outcome(payload=None, r_task=None, gt_return=None)
    res = B.build_train_result(Candidate(cand_id="c", iteration=0, reward_code="x"), [out])
    row = res.seed_metrics[0]

    assert row["checkpoints"] == [], "an unscored seed emitted a scored checkpoint"
    assert row["gt_return"] is None, "unscored must not read as 0.0"
    assert res.gt_reward_curve in ([], [None]), res.gt_reward_curve


def test_a_measured_zero_task_return_survives_as_zero():
    """`r_task or 0.0` cannot tell a measured 0.0 from an absent one."""
    out = _seed_outcome(payload={"fitness": 0.5}, r_task=0.0, gt_return=0.0)
    assert out.checkpoints, "a scored seed emitted no checkpoint"
    assert out.checkpoints[0]["reward_return"] == 0.0
    assert out.r_task == 0.0 and out.r_task is not None


def test_a_scored_seed_reaches_the_native_fitness_readers_key_path():
    """The row the backend builds, read the way `fitness_native_reward` reads."""
    from bird.components import _ippo_backend as B
    from bird.native_signal import NATIVE_REWARD_KEYS
    from bird.types import Candidate

    out = _seed_outcome(payload={"fitness": 0.25}, r_task=12.0, gt_return=558.61)
    res = B.build_train_result(Candidate(cand_id="c", iteration=0, reward_code="x"), [out])
    cps = res.seed_metrics[0]["checkpoints"]
    assert cps, "no curve; native fitness has nothing to read"
    for key in NATIVE_REWARD_KEYS:
        assert key in cps[-1]
    assert cps[-1]["gt_return"] == pytest.approx(558.61)


def test_the_reduction_refuses_a_per_agent_dict_and_a_0d_object_array():
    """Both guards, by the shapes that actually reach them."""
    import numpy as np

    from bird.components._ippo_train import aligned_task_return

    T, n = 4, 2
    good = np.ones((T, n))
    # OUR message, not numpy's. `np.asarray(dict, dtype=float)` raises a
    # TypeError whose text also contains "dict", so a loose match would pass
    # whether or not the isinstance guard existed.
    with pytest.raises(TypeError, match="indexed out by the caller"):
        aligned_task_return({"robot": good}, good, np.zeros((T, n), bool))
    # what `np.asarray` makes of a dict, arriving as an array rather than a dict
    obj = np.asarray({"robot": good})
    assert obj.ndim == 0 and obj.dtype == object       # the silent shape
    with pytest.raises(TypeError, match="0-d|object"):
        aligned_task_return(obj, good, np.zeros((T, n), bool))


def test_the_payload_channels_go_through_the_transposing_seam():
    """AST: both pref channels pass through `_eps_first`, and n_eps is checked.

    WEAKER THAN EXECUTING IT, and said so. `_eps_first` is nested inside
    `_evaluate`, so it cannot be imported without restructuring
    `_ippo_train.py`. The contract it must
    satisfy IS executed, by
    `test_the_built_payload_validates_in_episodes_first_orientation`, which
    passes both orientations to the real validator; this pins that the seam
    is actually applied to the channels on the way there. Together they
    cover what one alone does not: which orientation is correct, and that
    the code reaches it.
    """
    import ast
    import inspect

    from bird.components import _ippo_train

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(_ippo_train)))
              if isinstance(n, ast.FunctionDef) and n.name == "_evaluate")
    dumped = ast.dump(fn)
    assert "_eps_first" in dumped, "the payload channels never reach the transpose"
    assert dumped.count("_eps_first") >= 3, (
        "fewer than both channels go through the seam (definition + two calls)")
    assert "NUM_EVAL_EPISODES" in dumped, (
        "n_episodes is not checked against the configured episode count, so a "
        "transposed payload could pass silently")


def test_run_emits_a_checkpoint_only_when_the_payload_exists():
    """AST on `run`, because calling it needs a device, a zoo and a policy.

    Two mutants live here and nothing offline can execute them: making the
    emission unconditional, and emptying the list. A helper that MIRRORS
    run's logic cannot catch either -- a mirror moves with the original only
    when someone remembers to move it -- so the structure is asserted on the
    real source and labelled structural.
    """
    import ast
    import inspect

    from bird.components import _ippo_train

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(_ippo_train)))
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    guards = [n for n in ast.walk(fn) if isinstance(n, ast.If)
              and "eval_payload" in ast.dump(n.test)
              and "checkpoint_row" in ast.dump(n)]
    assert guards, (
        "run() emits its checkpoint row without guarding on eval_payload: an "
        "unscored seed would be recorded as a scored one")
    assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "checkpoint_row"
               for n in ast.walk(guards[0])), "the guard does not contain the emission"


# -- the pytree-carry class -------------------------------------------------
#
# GATED BY EXECUTION, NOT BY THIS SUITE. The defect these pin fires at TRACE
# on a device and cannot fire here: a `lax.scan` carry whose structure
# differs between input and output. Green on this tier does NOT cover the
# class; a device run is what covers it. Stated so the suite is not read as
# more than it is.


def test_the_metric_slots_written_at_step_are_exactly_those_seeded_at_reset():
    """PURE PYTHON, so it runs in every selector.

    The structural test below needs jax and assistax and skips wherever
    they are not installed, so a defect of this class reaches a device unseen: slots
    added on the step path and not the reset path change the carry's pytree
    structure, and `lax.scan` refuses at trace (`ippo_ff_nps.py:462`). Read
    off the named constants so a third slot
    added to one path fails here, in every CI job.
    """
    import ast
    import inspect

    from bird.components import assistax_ppo as A

    assert set(A.BIRD_METRIC_SLOTS) == set(A.BIRD_METRIC_SEEDS), (
        "the seeded slots and the declared slots disagree")

    src = inspect.getsource(A)
    written = {n.value for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and n.value.startswith("bird_")}
    assert written == set(A.BIRD_METRIC_SLOTS), (
        f"a `bird_` metric key appears in the module that is not a declared "
        f"slot, or vice versa: module {sorted(written)} vs declared "
        f"{sorted(A.BIRD_METRIC_SLOTS)}. A key written on one path and not "
        f"seeded on the other changes the scan carry's pytree structure.")


@pytest.mark.jax
def test_the_reset_state_and_the_stepped_state_have_one_tree_structure():
    """THE WHOLE STRUCTURE, not the two names.

    `jax.tree_util.tree_structure` over both states, so a third key added to
    either path fails regardless of what it is called.
    """
    pytest.importorskip("assistax")
    import jax
    import jax.numpy as jnp

    import assistax
    from bird.components.assistax_ppo import _human_only_pref_obs_class

    # `pref_configs=None`: with `load_agents={}` there is no drawn partner,
    # so `_append_pref_to_obs` would KeyError on `ag_idx["human"]` -- an
    # impossible configuration (preference configs, no agent to apply them
    # to) rather than a defect. The preference path is covered by the width
    # tests; what THIS test needs is the metrics slots, which are seeded and
    # written regardless of it.
    base = assistax.make("scratchitch", homogenisation_method="max")
    w = _human_only_pref_obs_class()(base, {}, pref_configs=None,
                                     agents_expect_pref_obs=False)
    _obs, reset_state = w.reset(jax.random.PRNGKey(0))
    acts = {a: jnp.zeros(base.action_space(a).shape) for a in base.agents}
    _o2, stepped, _r, _d, _i = w.step(jax.random.PRNGKey(1), reset_state, acts)

    assert (jax.tree_util.tree_structure(reset_state)
            == jax.tree_util.tree_structure(stepped)), (
        "reset and stepped states differ in pytree structure; a scan carry "
        "built from them raises at trace")
    assert (set(reset_state._state.metrics) == set(stepped._state.metrics))


def test_the_payload_carries_the_PRE_step_contact_force():
    """A WRONG-QUANTITY defect no shape test catches.

    `bird_prev_contact_force` records the value AFTER the step, so the
    series' row t is what step t+1 begins with. The payload defines
    `last_contact_force` as the force carried INTO step 0. The two have the
    same shape and different meanings, so the check is on VALUES with a
    transition where they differ.
    """
    import numpy as np

    from bird.components._ippo_train import pre_step_contact_force

    post = np.array([[7.0, 9.0], [11.0, 13.0]])      # post-step, all distinct
    pre = pre_step_contact_force(post)
    assert pre.shape == post.shape
    assert np.array_equal(pre[0], np.zeros(2)), (
        "row 0 must be the wrapper's initial force, not the first post-step one")
    assert np.array_equal(pre[1], post[0]), "row t must be post-step row t-1"
    assert not np.array_equal(pre[0], post[0]), (
        "pre and post agree on row 0, so this fixture cannot discriminate")


def test_partner_ids_are_one_per_episode_not_one_per_step():
    """`bird_ag_idx` is (T, n_episodes); the payload wants n_episodes.

    Flattening it with `reshape(-1)` produces T x n_episodes ids, which
    `validate_eval_payload` refuses.
    """
    import numpy as np

    from bird.components._ippo_train import _distinct_ids, _partner_ids

    class _Split:
        train = {"IPPO": {"human": ["u0", "u1", "u2", "u3"]}}

    T, n_eps = 5, 3
    drawn = np.zeros((T, n_eps), dtype=int)
    drawn[0] = [0, 1, 2]              # the partners each episode STARTED with
    drawn[3:] = [3, 3, 3]             # a mid-rollout resample

    ids = _partner_ids(_Split(), drawn)
    assert ids == ["u0", "u1", "u2"], ids
    assert len(ids) == n_eps, f"{len(ids)} ids for {n_eps} episodes"
    # the resampling is recorded rather than implied away
    assert _distinct_ids(drawn) == 4


def test_the_checkpoint_fitness_is_not_a_fabricated_zero():
    """The payload carries no "fitness" key, so `.get("fitness", 0.0)` would
    score every checkpoint 0.0 -- the None-vs-0.0 rule one hop over.

    Asserted on the source, because the value is chosen inside `run`, which
    needs a device: the fitness passed to `checkpoint_row` must derive from
    a measured quantity, not from a `.get` default on a key the payload
    never has.
    """
    import ast
    import inspect

    from bird.components import _ippo_train

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(_ippo_train)))
              if isinstance(n, ast.FunctionDef) and n.name == "run")
    call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", "") == "checkpoint_row")
    fit = next(k.value for k in call.keywords if k.arg == "fitness")
    dumped = ast.dump(fit)
    assert "r_task" in dumped, (
        f"checkpoint fitness does not derive from a measured quantity: {dumped[:90]}")
    assert "'fitness'" not in dumped and '"fitness"' not in dumped, (
        "fitness is read from the payload by key; the payload has no such key, "
        "so every checkpoint would score the default")


def test_the_seeding_wrapper_actually_replaces_the_inner_reset():
    """Runs in EVERY selector, unlike the tree-structure test it backs up.

    Removing `wrapped.reset = reset` reinstates the fatal -- the slots are
    written on the step path and absent on the reset path, the scan carry
    changes pytree structure, and `lax.scan` refuses at trace. The only
    test that catches it behaviourally is `@pytest.mark.jax`, which skips in
    the default CI job (no jax installed), so the defect would reach a device again. This
    is structural, and says so.
    """
    import ast
    import inspect

    from bird.components import assistax_ppo as A

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(A)))
              if isinstance(n, ast.FunctionDef) and n.name == "_metric_slot_env")
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Attribute) and t.attr == "reset"
                       for t in n.targets)]
    assert assigns, (
        "_metric_slot_env does not replace the inner env's reset, so the "
        "metric slots exist on the step path only and the scan carry changes "
        "pytree structure between input and output")


# -- the payload contract's one-partner premise ------------------------------


def _payload_kw(n_eps=3, **over):
    from bird.components import _ippo_backend as B
    import numpy as np
    kw = dict(partner_id=[f"u{i}" for i in range(n_eps)],
              last_contact_force=np.zeros(n_eps),
              partner={f: np.full(n_eps, 0.9 if f.endswith("_max") else 0.2)
                       for f in B.PARTNER_FIELDS},
              task="scratchitch", seed=0)
    kw.update(over)
    return kw


def test_a_scalar_partner_field_is_refused_not_broadcast():
    """A scalar partner field is not broadcast; a "broadcast" to (1,) is not one.

    Refused by name, which is the stronger form: under a partner resampled
    per episode a scalar is a MISSING per-episode measurement, not one to
    spread. At n_episodes == 1 a (1,) array and a per-episode array are
    indistinguishable, so only a pinned NUM_EVAL_EPISODES would keep a silent
    broadcast safe.
    """
    import numpy as np

    from bird.components import _ippo_backend as B

    kw = _payload_kw()
    kw["partner"] = {**kw["partner"], "w_speed": 0.5}      # a scalar
    with pytest.raises(B.EvalPayloadError, match="per episode"):
        B.eval_rollout_payload(tool_speed=np.zeros((3, 4)),
                               contact_force=np.zeros((3, 4)), **kw)


def test_a_missing_partner_field_is_refused_at_BUILD_time():
    """An `if k in partner` filter would drop it and leave the refusal to a consumer.

    A full device training would then report ten problems at consume time
    instead of one at the line where the block was assembled.
    """
    import numpy as np

    from bird.components import _ippo_backend as B

    kw = _payload_kw()
    kw["partner"] = {k: v for k, v in kw["partner"].items() if k != "touch_threshold"}
    with pytest.raises(B.EvalPayloadError, match="touch_threshold"):
        B.eval_rollout_payload(tool_speed=np.zeros((3, 4)),
                               contact_force=np.zeros((3, 4)), **kw)


def test_episodes_sharing_a_partner_id_must_share_its_parameters():
    """The join that accepting per-episode arrays does NOT buy.

    32 distinct ids with one partner's parameters broadcast across them is
    valid on every other check and scores every episode against the wrong
    partner. Built through the real builder, refused by the real validator.
    """
    import numpy as np

    from bird.components import _ippo_backend as B

    # Built VALID, then mutated: the builder refuses an invalid payload,
    # so the validator is exercised directly on the shape a bad emitter would
    # produce.
    payload = B.eval_rollout_payload(tool_speed=np.zeros((3, 4)),
                                     contact_force=np.zeros((3, 4)),
                                     **_payload_kw(n_eps=3))
    payload["partner_id"] = ["same", "same", "other"]
    payload["partner"] = {**payload["partner"],
                          "w_speed": np.array([0.1, 0.9, 0.5])}
    problems = B.validate_eval_payload(payload)
    assert any("share partner_id" in p for p in problems), problems


def test_many_ids_with_one_parameter_vector_is_not_refused():
    """Legal: two drawn partners may share parameters, and refusing that would
    refuse a real population."""
    import numpy as np

    from bird.components import _ippo_backend as B

    kw = _payload_kw(n_eps=3, partner_id=["a", "b", "c"])    # constant params
    payload = B.eval_rollout_payload(tool_speed=np.zeros((3, 4)),
                                     contact_force=np.zeros((3, 4)), **kw)
    assert not B.validate_eval_payload(payload), "a legal payload was refused"


def test_a_terminating_task_is_refused_until_lengths_are_carried():
    """Zero padding is finite and in range, so it scores as simulated time."""
    import numpy as np

    from bird.components import _ippo_backend as B

    payload = B.eval_rollout_payload(tool_speed=np.zeros((3, 4)),
                                     contact_force=np.zeros((3, 4)),
                                     **_payload_kw())
    payload["task"] = "bedbathing"          # the one task that terminates
    assert any("terminate early" in p for p in B.validate_eval_payload(payload))
    ok = B.eval_rollout_payload(tool_speed=np.zeros((3, 4)),
                                contact_force=np.zeros((3, 4)), **_payload_kw())
    assert not B.validate_eval_payload(ok)


def test_meta_partner_distinct_ids_must_agree_with_partner_id():
    """Two copies of one fact, made load-bearing rather than decorative.

    `partner_distinct_ids` is derivable from `partner_id`, which the
    validator already length-checks. Carrying it anyway is worth it only if a
    disagreement is an error -- otherwise the join check believes
    `partner_id` while a reader believes the meta number, and they can drift
    apart silently.
    """
    import numpy as np

    from bird.components import _ippo_backend as B

    payload = B.eval_rollout_payload(tool_speed=np.zeros((3, 4)),
                                     contact_force=np.zeros((3, 4)),
                                     **_payload_kw(n_eps=3))
    payload["meta"] = {"partner_distinct_ids": 3}
    assert not B.validate_eval_payload(payload), "agreeing values were refused"

    payload["meta"] = {"partner_distinct_ids": 7}       # partner_id holds 3
    problems = B.validate_eval_payload(payload)
    assert any("partner_distinct_ids" in p for p in problems), problems


def test_steps_dropped_must_be_a_multiple_of_the_episode_count():
    """An invariant enforced by the validator, not left in a comment.

    NOT "one per episode" -- that is the intuitive reading and it is wrong.
    Two episodes per column would drop two per column and still be a
    multiple. The invariant is that every column drops the SAME number, so
    this tests the RECTANGLE from another angle. It holds BECAUSE the
    terminating-task refusal excludes early termination; if per-episode
    lengths ever land, both go together.
    """
    import numpy as np

    from bird.components import _ippo_backend as B

    payload = B.eval_rollout_payload(tool_speed=np.zeros((4, 9)),
                                     contact_force=np.zeros((4, 9)),
                                     **_payload_kw(n_eps=4))
    for ok in (0, 4, 8):
        payload["meta"] = {"gt_return_steps_dropped": ok}
        assert not B.validate_eval_payload(payload), f"{ok} was refused"
    payload["meta"] = {"gt_return_steps_dropped": 5}
    assert any("multiple of" in p for p in B.validate_eval_payload(payload))


def test_the_two_partner_counts_are_different_quantities_and_both_validate():
    """57 all-step ids against 31 step-0 ids: two counts, not one.

    Produce `partner_distinct_ids` from `bird_ag_idx` over ALL steps while
    `require_valid_eval_payload` asserts it equals the distinct ids of
    `partner_id`, which holds row 0 only, and the two agree exactly until a
    rollout is long enough to auto-reset -- every short test passes and a
    real run fails. They are two quantities, so they are two keys, and this
    builds the shape where they DIFFER.
    """
    import numpy as np

    from bird.components import _ippo_backend as B
    from bird.components._ippo_train import _distinct_ids, _partner_ids

    class _Split:
        train = {"IPPO": {"human": ["u0", "u1", "u2", "u3"]}}

    T, n_eps = 5, 3
    drawn = np.zeros((T, n_eps), dtype=int)
    drawn[0] = [0, 1, 2]              # the partners each episode STARTED with
    drawn[3:] = [3, 3, 3]             # a mid-rollout resample, as auto_reset does

    partner_id = _partner_ids(_Split(), drawn)
    step0 = len({str(x) for x in partner_id})
    all_steps = _distinct_ids(drawn)
    assert (step0, all_steps) == (3, 4), (step0, all_steps)

    def _payload(meta):
        p = B.eval_rollout_payload(tool_speed=np.zeros((n_eps, 4)),
                                   contact_force=np.zeros((n_eps, 4)),
                                   **_payload_kw(n_eps=n_eps))
        p["meta"] = meta
        return p

    # what the producer emits: like for like, plus the resample count
    assert not B.validate_eval_payload(_payload({
        "partner_distinct_ids": step0,
        "partner_distinct_ids_all_steps": all_steps})), "the fixed pair was refused"

    # the failure shape -- the all-step count under the step-0 key
    problems = B.validate_eval_payload(_payload({
        "partner_distinct_ids": all_steps}))
    assert any("partner_distinct_ids" in p for p in problems), problems

    # and the one direction a resample cannot produce
    problems = B.validate_eval_payload(_payload({
        "partner_distinct_ids": step0,
        "partner_distinct_ids_all_steps": step0 - 1}))
    assert any("partner_distinct_ids_all_steps" in p for p in problems), problems


def test_the_step0_count_is_produced_from_partner_id_not_from_the_metrics():
    """The producer must derive the checked field from the checked list.

    Asserted on the source because the meta dict is built inside `_evaluate`,
    which needs a device. The validator's redundancy check is only meaningful
    if the two copies are the same KIND of quantity, and the way it breaks
    is the producer reading `bird_ag_idx` while the checker reads
    `partner_id` -- so what is pinned here is exactly which name each field
    is computed from.
    """
    import ast
    import inspect

    from bird.components import _ippo_train

    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(_ippo_train)))
              if isinstance(n, ast.FunctionDef) and n.name == "_evaluate")
    fields = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value in (
                        "partner_distinct_ids", "partner_distinct_ids_all_steps"):
                    fields[k.value] = {n.id for n in ast.walk(v)
                                       if isinstance(n, ast.Name)}
    assert set(fields) == {"partner_distinct_ids",
                           "partner_distinct_ids_all_steps"}, fields
    assert "partner_id" in fields["partner_distinct_ids"], (
        "partner_distinct_ids must be derived from `partner_id` -- the list the "
        f"validator compares it against; it reads {fields['partner_distinct_ids']}")
    assert "metrics" in fields["partner_distinct_ids_all_steps"], (
        "the resample count is the all-step `bird_ag_idx` read, from `metrics`; "
        f"it reads {fields['partner_distinct_ids_all_steps']}")


def test_run_loads_no_global_the_module_never_defines():
    """A name that is a local of `_evaluate`, read in `run`, is a NameError in wait.

    Such a line can be unreachable only because `require_valid_eval_payload`
    raises one line earlier, so every offline test and every gate passes
    while the one path that matters cannot complete.

    THIS INSPECTS THE COMPILED FUNCTION, not the source, and that is what
    makes it a test of the raising path rather than a proxy for one. Python
    decides `n_dropped` is a global AT COMPILE TIME -- the enclosing
    assignment is in another function, so the compiler emits `LOAD_GLOBAL`
    and no runtime input can make that resolve. The NameError is therefore
    already fully determined by this code object, and a test that drove 270
    lines of `run` through stubbed jax and a stubbed trainer to watch it
    raise would be reporting on the stubs. Every `LOAD_GLOBAL` name in
    `run`, less what the module and builtins define, must be empty.
    """
    import builtins
    import dis
    import types

    from bird.components import _ippo_train

    # WALK THE OPCODES, NOT `co_names`, AND RECURSE INTO `co_consts`.
    # `co_names` also holds attribute names, so a check over it flags every
    # attribute access and invites an allow-list that swallows the real
    # case; and a bad global inside a nested def lives in its own code
    # object, invisible to a top-level pass.
    def _globals_of(code):
        names = {i.argval for i in dis.get_instructions(code)
                 if i.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                names |= _globals_of(const)
        return names

    loaded = _globals_of(_ippo_train.run.__code__)
    defined = set(vars(_ippo_train)) | set(dir(builtins))
    unbound = sorted(loaded - defined)
    assert not unbound, (
        f"`run` compiles a global load of {unbound}, which the module does not "
        "define -- a NameError the moment that line is reached")


def test_the_per_episode_partner_parameters_are_read_off_step_zero():
    """`distinct_parameter_vectors` must share `partner_id`'s axis.

    A meta number is only comparable with another if both are over the same
    population, and the count above is exactly two axes under one name if it
    goes wrong. `_partner_params` indexes the stacked configs by the drawn
    partner, and if it took all steps rather than row 0 the parameter vectors
    would name partners the `partner_id` list does not -- incomparable in the
    same way.
    """
    import numpy as np

    from bird.components._ippo_train import _partner_params
    from bird.components import _ippo_backend as B

    class _Env:
        pref_configs = {"human": {f: np.arange(4, dtype=float) + (
            0.9 if f.endswith("_max") else 0.0) for f in B.PARTNER_FIELDS}}

    T, n_eps = 5, 3
    drawn = np.zeros((T, n_eps), dtype=int)
    drawn[0] = [0, 1, 2]              # step 0: the episodes' own partners
    drawn[3:] = [3, 3, 3]             # a later resample, which must NOT be read

    got = _partner_params(_Env(), drawn, n_eps, None)
    field = B.PARTNER_FIELDS[0]
    assert list(got[field]) == list(_Env.pref_configs["human"][field][[0, 1, 2]]), (
        f"{field} was not read off step 0: {got[field]}")
    assert all(len(v) == n_eps for v in got.values())


def test_evaluate_returns_exactly_what_run_unpacks():
    """The arity that carried `n_dropped` out of `_evaluate` is a contract.

    Returning it is what makes `run` correct; a later return site that forgets
    it restores the NameError in a form the symtable check above cannot see,
    because the name would then be bound from a short tuple and fail at
    runtime instead. Both return sites and the call site are pinned together.
    """
    import ast
    import inspect

    from bird.components import _ippo_train

    tree = ast.parse(inspect.getsource(_ippo_train))
    ev = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_evaluate")
    arities = {len(n.value.elts) for n in ast.walk(ev)
               if isinstance(n, ast.Return) and isinstance(n.value, ast.Tuple)}
    # 6: `trajectories` (the eval rollouts as `Trajectory` objects) travels
    # beside `n_dropped` on both return sites and at the call site.
    assert arities == {6}, f"_evaluate return arities are {arities}, not just 6"

    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run")
    unpacks = [len(t.elts) for a in ast.walk(run)
               if isinstance(a, ast.Assign)
               for t in a.targets
               if isinstance(t, ast.Tuple)
               and isinstance(a.value, ast.Call)
               and getattr(a.value.func, "id", None) == "_evaluate"]
    assert unpacks == [6], f"run unpacks _evaluate into {unpacks}, not 6"
