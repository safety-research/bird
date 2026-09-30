"""`train.reward_source`: which reward the LEARNER trains on.

WHY THE KEY EXISTS, stated here because the alternative looks cheaper and is
wrong. An ORACLE arm trains the same learner on the environment's shipped
reward, and it is a published baseline -- the "Oracle" curve of Text2Reward's
Fig. 2 (`text2reward/text/experiments.tex:55`, "the expert-written reward
function provided by the environment") and of CARD's Fig. 3. The obvious way to
build one is `llm.generator.provider: fixed` plus a hand transcription of the
shipped reward. That works for gym HalfCheetah-v5, whose reward is a function
of the observation alone. It does not generalise:

  * a reward program is handed `(s, a)` and `self` bound to `None`
    (`CompiledReward._apply_bind`), so it cannot reach the adapter at training
    time -- though it CAN during verification (`verification.call_reward` passes
    `_SelfProxy`), so such a program passes stage 2 and dies every step of
    stage 3; and
  * all six of CARD's Meta-World v2 rewards read simulator state the 39-D
    observation does not carry (`tcp_center`, `init_tcp`, `obj_init_pos`,
    `_gripper_caging_reward`), so a transcription could not be faithful even if
    it could be reached.

Six approximate transcriptions would produce a curve labelled "Oracle" that is
not the oracle -- plausible, and invisible. Hence a key, dispatching to the
adapter's own callable, with nothing transcribed.

THE DEFAULT IS EVERY PUBLISHED METHOD. `candidate` must stay byte-identical to
compiling the candidate's own program at the three training sites, which is
what the first test below asserts rather than assumes.
"""

import pytest

from bird import registry
from conftest import backend_param
from bird.components import training
from bird.config import ConfigError, load

KEY = "train.reward_source"

#: A Meta-World task whose spec claims its shipped reward, and a gym task whose
#: spec disowns it (`reward.human.kind: none`, `bird/envs/gym_mujoco.py:46-56`)
#: -- the pair is what makes the refusal a per-TASK check and not a per-family one.
ENV_WITH_REF = "mt10_drawer-open-v3"
ENV_WITHOUT_REF = "gym_half_cheetah_backward"
ENV_WITH_REF_GYM = "gym_half_cheetah"


def test_the_family_has_both_members_and_the_key_is_backed_by_it():
    """A config value is a registry key: the family exists with both members."""
    registry.load_all()
    assert "reward_source" in registry.KINDS
    assert set(registry.names("reward_source")) == {"candidate", "reference"}
    from bird.schema import SCHEMA

    assert SCHEMA[KEY].kind == "reward_source"


def test_every_config_in_the_corpus_resolves_to_candidate():
    """The default is the published behaviour, so nothing's number moves.

    This is the test that would fail if someone 'helpfully' pinned `reference`
    anywhere, or moved the default. Every published method trains on the reward
    its generator wrote; there is no published method that does not."""
    from conftest import PAPER_CONFIGS

    for path in PAPER_CONFIGS:
        cfg = load(str(path))
        assert cfg[KEY] == "candidate", f"{path} resolves {KEY}={cfg[KEY]!r}"


def test_the_candidate_member_is_the_call_it_replaced():
    """`candidate` compiles the candidate's own code and nothing else.

    Asserted on the RESULT rather than by reading the source: the member must
    return a CompiledReward over the candidate's program, with the candidate's
    weights, in the candidate's reward language."""
    code = "def compute_reward(s, a):\n    return 2.0, {'two': 2.0}\n"

    class _Cand:
        cand_id = "c0"
        reward_code = code
        weights = {}

    class _Ctx:
        cfg = load("card", profile="tester")
        env = None

        @staticmethod
        def event(*a, **k):
            raise AssertionError("candidate source must journal nothing")

    reward = training._training_reward(_Ctx(), None, _Cand())
    total, comps = reward(None, None, None)
    assert total == 2.0 and comps == {"two": 2.0}


# --------------------------------------------------------------------------
# the reference member
# --------------------------------------------------------------------------


class _FakeEnv:
    """An adapter that has a reference reward and counts the calls.

    A fake rather than Meta-World on purpose: this test is about the WIRING --
    does the learner's reward become the environment's function, is it called on
    the state the step ARRIVED in -- and none of that needs a simulator. The
    Meta-World-specific claim (that `reference_reward` is bitwise the v2 branch)
    is measured per task against the simulator, outside this suite.
    """

    has_reference_reward = True

    def __init__(self):
        self.calls = []

    def reference_reward(self, s, a=None):
        self.calls.append((s, a))
        return 7.5


class _NoRefEnv:
    has_reference_reward = False


def _ctx_for(env, **overrides):
    ov = {KEY: "reference", "generate.n_candidates": 1,
          "problem.env_id": ENV_WITH_REF}
    ov.update(overrides)

    class _Ctx:
        cfg = load("card", profile="tester", overrides=ov)
        events = []

        def event(self, *a, **k):
            self.events.append((a, k))

    ctx = _Ctx()
    ctx.env = env
    return ctx


class _Cand:
    cand_id = "c1"
    reward_code = "def compute_reward(s, a):\n    return -999.0, {}\n"
    weights = {}


def test_reference_trains_on_the_environments_reward_not_the_candidates():
    """The whole point, and the candidate's program must not leak into it.

    The candidate here returns -999; the env returns 7.5. A run that trained on
    -999 while its config said `reference` would be the invisible failure this
    mechanism exists to avoid, so the assertion is on the VALUE."""
    env = _FakeEnv()
    reward = training._training_reward(_ctx_for(env), None, _Cand())
    total, comps = reward([1.0], [0.1], [2.0])
    assert total == 7.5
    assert comps == {}, "a shipped reward has no BIRD-visible decomposition"


def test_it_scores_the_state_the_step_arrived_in():
    """`s_next` when there is one, `s` otherwise -- the BIRD convention, and the
    same choice `EnvAdapter.gt_reward` makes (`s2 or s`).

    If these disagreed, a run's training reward and the `gt_return` series it
    records alongside would be two different numbers under one name."""
    env = _FakeEnv()
    reward = training._training_reward(_ctx_for(env), None, _Cand())
    reward([1.0], [0.1], [2.0])
    reward([3.0], [0.2], None)
    assert [s for s, _ in env.calls] == [[2.0], [3.0]]
    assert [a for _, a in env.calls] == [[0.1], [0.2]]


def test_it_journals_that_the_learner_did_not_see_the_candidates_program():
    """A run whose reward is not the generator's must say so in the artifact.

    Without this, an oracle cell and a searched cell differ only in their
    resolved config, and `report.json` attributes the environment's reward to
    the generator."""
    ctx = _ctx_for(_FakeEnv())
    training._training_reward(ctx, None, _Cand())
    kinds = [k for _a, k in ctx.events]
    assert any(k.get("event") == "reward_source" and k.get("source") == "reference"
               for k in kinds), kinds


def test_an_adapter_without_a_reference_reward_raises_rather_than_returning_zero():
    """The runtime backstop for the load-time refusal.

    `0.0` would be a flat reward that trains happily and reports nothing wrong,
    which is the shape `EnvAdapter.reference_reward`'s own docstring refuses
    ("raising is the contract for absence")."""
    with pytest.raises(ValueError, match="reference"):
        training._training_reward(_ctx_for(_NoRefEnv()), None, _Cand())


# --------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------


def test_reference_is_refused_on_a_task_that_disowns_its_shipped_reward():
    """Per TASK, not per family -- the five gym specs with `reward.human.kind: none`.

    `gym_half_cheetah` and `gym_half_cheetah_backward` are the same adapter and
    the same simulator; only the second's spec disowns the built-in reward,
    because it pays "the exact thing this task penalises". A per-family check
    would admit both, and the oracle curve on the backward task would be a
    forward-running reward wearing the task's name."""
    with pytest.raises(ConfigError, match="reward_source=reference"):
        load("card", overrides={KEY: "reference", "generate.n_candidates": 1,
                                "problem.env_id": ENV_WITHOUT_REF})
    cfg = load("card", overrides={KEY: "reference", "generate.n_candidates": 1,
                                  "problem.env_id": ENV_WITH_REF_GYM})
    assert cfg[KEY] == "reference"


def test_reference_is_refused_with_a_population():
    """An oracle arm trains one program: the environment's.

    K cells would be K learner seeds of one reward wearing a search's clothes,
    and the reflection prompt would be built from a reward the generator did not
    write."""
    with pytest.raises(ConfigError, match="n_candidates"):
        load("card", overrides={KEY: "reference", "generate.n_candidates": 8,
                                "problem.env_id": ENV_WITH_REF})


def test_the_population_refusal_does_not_fire_under_candidate():
    """K > 1 is only an error for `reference`. `eureka` carries K = 16 unchanged."""
    cfg = load("eureka")
    assert cfg[KEY] == "candidate"
    assert cfg["generate.n_candidates"] > 1


def test_the_shipped_reward_refusal_does_not_fire_under_candidate():
    """A task that disowns its shipped reward is fine to SEARCH on -- it is only
    the oracle arm that needs one. `card` on the same env the oracle arm is
    refused for."""
    cfg = load("card", overrides={"problem.env_id": ENV_WITHOUT_REF})
    assert cfg[KEY] == "candidate"


# BOTH NEGATIVE-SCOPE CHECKS ABOVE ARE SPLIT IN TWO, AND THE REASON IS WORTH
# KEEPING: one test that varies BOTH keys at once off one config fails for
# reasons that have nothing to do with this key. On `card` it hits
# `select.rule=none with generate.n_candidates=8` ("this rule ranks nothing and
# keeps the first candidate"); on `eureka` it hits
# `evaluate.fitness.source='native'` on a task that ships no native signal --
# the same underlying fact this key's own refusal reads, arriving through a
# different rule. Either failure would tell a reader the refusal was
# over-firing when it was not. A negative-scope test has to vary ONE key off a
# config that is otherwise legal, the same discipline an ablation demands.


def test_an_unregistered_value_is_refused_with_the_family():
    """The `kind=` path through the real validator."""
    with pytest.raises(ConfigError, match="reference"):
        load("card", overrides={KEY: "oracle"})


# --------------------------------------------------------------------------
# every learner backend routes through the one dispatch
# --------------------------------------------------------------------------
#
# `_training_reward` is the single dispatch, which is TRUE BY CONSTRUCTION and
# guarded by nothing unless a test enumerates the family. A new `train_backend`
# that called `compile_reward` directly would merge cleanly and leave the other
# backends honouring `train.reward_source` and one ignoring it -- a
# declared-but-unread key on one backend only, the worst-behaved member of that
# class, since every test that does not select that backend stays green.
#
# The pattern below is `tests/test_pruning.py`'s, deliberately, down to
# `_backends()` reading every registered backend off the registry: the risk is
# not one backend written wrongly but a second implementation of the same stage
# growing a second answer to the same config key. A new backend copying an old
# one would inherit the bug; parametrising means it inherits the test instead.

import types as pytypes  # noqa: E402  (grouped with the catalogue block)

from bird.budget import Budget  # noqa: E402
from bird.context import Context  # noqa: E402
from bird.types import Candidate  # noqa: E402

_REWARD = "def compute_reward(s, a):\n    return 0.0, {}\n"
_STEPS = 3000
_SB3_PPO_TINY = {"n_steps": 600, "batch_size": 100}
_FASTTD3_TINY = {"num_envs": 2, "batch_size": 32, "buffer_size": 64,
                 "critic_hidden_dim": 8, "actor_hidden_dim": 8, "num_atoms": 5,
                 "v_min": -10.0, "v_max": 10.0, "learning_starts": 1,
                 "compile": False}
# `simba_v2` is the FOURTH learner backend. Sized down for the same reason as
# the two above --
# the assertion is "was the spy called", and paying for 3000 updates through
# 512-wide hyperspherical blocks to learn that would put a minute of GPU-less
# matmul inside a test in the default (non-slow) selection. The keys are `tests/test_simba_v2.py::TINY`
# verbatim, which that file measures a real training with.
_SIMBA_V2_TINY = {"batch_size": 32, "buffer_size": 256,
                  "critic_hidden_dim": 16, "actor_hidden_dim": 16,
                  "critic_num_bins": 11, "learning_starts": 8, "compile": False,
                  "actor_num_blocks": 1, "critic_num_blocks": 1}


def _backends():
    registry.load_all()
    return sorted(n for (k, n) in registry._REGISTRY if k == "train_backend")


def _skip_if_unavailable(name: str) -> None:
    """NARROW on purpose: a missing optional package skips, and nothing else does.

    A broad skip here would turn the finding this file exists for into a green
    dot -- a silent `importorskip`, one level worse than usual because the
    assertion being skipped is about a key being
    unread. So this names packages, never swallows an assertion, and a backend
    whose dependency IS installed can only pass by routing through the dispatch.
    """
    if name == "sb3":
        pytest.importorskip("stable_baselines3")
        pytest.importorskip("gymnasium")
    # The jax tier's learner. `jax` IS in an extra, so this gate can open on
    # an install with that extra. Narrow, as the docstring requires -- it names
    # the package and swallows no assertion, so on a machine with the extra the
    # backend can only pass by routing through the dispatch.
    if name == "assistax_ppo":
        pytest.importorskip("jax")
    if name == "fasttd3":
        pytest.importorskip("torch")
    if name == "simba_v2":
        # One package, and the RIGHT one: `simba_v2` is a torch port of
        # `refs/code/SimbaV2`'s layers, not a wrapper round upstream's JAX, so
        # there is no `jax` or `flax` to gate on -- naming either would be the
        # never-opening gate `tests/test_no_silent_skip.py` refuses. `torch` is
        # in the `fasttd3` extra, so on any tree where that extra is synced this
        # skip does not fire and the backend can only pass by routing.
        pytest.importorskip("torch")


def _backend_ctx(name):
    registry.load_all()
    overrides = {"seed": 0, "train.backend": name, "train.env_steps": _STEPS,
                 "train.seeds_per_candidate": 1, "evaluate.rollouts_per_candidate": 1}
    if name == "fasttd3":
        overrides["train.hyperparameters"] = dict(_FASTTD3_TINY)
    if name == "sb3":
        overrides["train.hyperparameters"] = dict(_SB3_PPO_TINY)
    if name == "simba_v2":
        overrides["train.hyperparameters"] = dict(_SIMBA_V2_TINY)
    cfg = load("eureka", overrides=overrides, profile="tester")
    return Context(cfg=cfg, budget=Budget(),
                   env=registry.get("env", "toy_reacher")({}))


def test_there_are_backends_to_check():
    """Guard the guard: a parametrised test over an empty list passes vacuously."""
    assert _backends()


@pytest.mark.parametrize("name", [backend_param(n) for n in _backends()])
def test_every_backend_routes_its_reward_through_the_dispatch(name, monkeypatch):
    """`train.reward_source` must reach the reward of EVERY learner backend.

    Spies the dispatch itself rather than a fitness number, for `test_pruning`'s
    reason: if a backend stops consulting it the spy is never called and the
    assertion fails on that, not on a number that could have moved for other
    reasons. `fasttd3` imports the function as `_training._training_reward`, so
    patching the module attribute reaches it through the module object; a fifth
    backend that does the same inherits the test."""
    _skip_if_unavailable(name)
    seen = []
    real = training._training_reward

    def _spy(*a, **k):
        seen.append(name)
        return real(*a, **k)

    monkeypatch.setattr(training, "_training_reward", _spy)
    registry.get("train_backend", name)(_backend_ctx(name), _state_ns(), _cand_obj(), 1)
    assert seen, (
        f"train_backend {name!r} compiled its reward without consulting "
        "_training_reward, so train.reward_source is a declared-but-unread key "
        "on that backend alone: it validates, it is written into "
        "config.resolved.yaml, it changes the run id, and on that backend it "
        "changes nothing about the run.")


def _state_ns():
    return pytypes.SimpleNamespace(restart=0, iteration=0)


def _cand_obj(cid="c0000"):
    return Candidate(cand_id=cid, iteration=0, reward_code=_REWARD)


# --------------------------------------------------------------------------
# the journal row must survive the failure that most needs it
# --------------------------------------------------------------------------


def test_the_journal_row_is_written_even_when_the_member_raises():
    """The event ordering, as a falsifying probe.

    `reward_source_reference` raises when the adapter reports no reference
    reward, and all three call sites turn that into `TrainResult(trained=False)`
    plus a charged budget slot. With the event emitted AFTER the member call,
    the one failure that most needs the record would be the only one without
    it: a failed candidate and a spent slot, with nothing but
    `config.resolved.yaml` to say the learner was never going to see the
    generator's program.

    Fails if the event is emitted after the member call."""
    ctx = _ctx_for(_NoRefEnv())
    with pytest.raises(ValueError):
        training._training_reward(ctx, None, _Cand())
    assert any(k.get("event") == "reward_source" and k.get("source") == "reference"
               for _a, k in ctx.events), \
        "the member raised and no reward_source row reached the journal"
