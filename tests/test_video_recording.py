"""Rollout video survives to the recorder -- the arithmetic and the mechanism.

WHAT CAN GO WRONG, AND WHY A SUITE WOULD GO ON PASSING. `MetaWorld` cannot
invert a 39-D observation back to `(qpos, qvel)`, so it keeps an obs ->
simulator-snapshot LRU and `render(state)` is a cache lookup.
`observability.record_rollouts` replays a trajectory produced BEFORE everything
else the stage emitted, so with the LRU alone the recorder routinely asks for
rows already dropped: the raise is caught, `record_skipped` is journalled, and
the run completes with fewer videos than it picked. Nothing fails. Measured on
`mt10_window-open-v3` with real SB3/SAC, `train.env_steps: 2000`, two
candidates: 11,126 states emitted per candidate (2,108 training, 7,515
checkpoint evaluation, 1,503 rollout) against an 8,192-entry cache, candidate 0's
rollout `0/501` present at record time, one video from two.

Every test here is against the SHAPES the shipped configs actually declare,
because the failure is arithmetic between a config and a cache size and not a
code path anyone would read wrong:

  * `limen` -- 1 candidate, `rollouts_per_candidate: 50`. The candidate's OWN
    rollouts 1..49 (24,500 rows) evict its rollout 0. Zero videos, every
    iteration, with nothing else running.
  * `eureka` -- 16 candidates x 3 rollouts x 501 rows = 24,048 imported into an
    8,192 cache by the parallel path. Only the last five candidates keep their
    video; which five is decided by index, never by fitness.
  * any multi-candidate SEQUENTIAL shape -- the next candidate's training and
    checkpoint evaluation (87% of the flood, and it scales with
    `train.env_steps` while the cache does not) wipes the previous one.

The mechanism is `EnvAdapter.retain_states`: the caller, which knows which
rollout will be replayed, says so, and the LRU cannot touch it.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import PAPER_CONFIGS
from bird.config import ConfigError, load
from bird.envs.metaworld import MetaWorld, UnknownStateError

#: Not in the tester-tier smoke suite (rendering: writes and re-reads frames).
#: Deselected in CI by `-m "not slow"`; run it with `-m slow`. See pyproject.toml.
pytestmark = pytest.mark.slow

#: Same task the rest of the Meta-World suite uses; see `tests/test_metaworld.py`.
TASK = "drawer-open-v3"

#: A Meta-World run is a method config plus an execution profile plus an env
#: override:
#:
#:     load("eureka", profile="full", overrides={"problem.env_id": "mt10_..."})
#:
#: so the set of shapes this file has to check is the CROSS PRODUCT, not a glob. Every
#: assertion below reads only `generate.n_candidates` and
#: `evaluate.rollouts_per_candidate` against `MetaWorld`'s class constants, and
#: neither key is touched by `problem.env_id` -- the env axis fills TASK_SPEC_KEYS
#: (`problem.task_description`, `verify.forbidden_symbols`) and nothing else. One
#: representative mt10 id is therefore the whole tier for this file's purposes,
#: and crossing all ten would be ten identical shapes wearing different names.
MW_ENV_ID = f"mt10_{TASK}"

#: `tester` is excluded on purpose: it is the mock-learner profile, it never
#: builds a MuJoCo adapter, and no snapshot cache exists for it to overflow. The
#: two profiles that run on Meta-World are `dev` (dev-scale) and `full` (1M-step).
MW_PROFILES = ("dev", "full")


# --------------------------------------------------------------------------
# the arithmetic: a declared shape against the store that has to hold it
# --------------------------------------------------------------------------


#: The refusal `validate` raises for a method whose `train.algorithm` the
#: profile's SB3 learner cannot train (L2R's `none`, Singh's tabular
#: `q_learning`). Such a method cannot be pointed at Meta-World under these
#: profiles at all, so it has no shape for this file to check. Matched on the
#: message rather than a list of config names, so any OTHER refusal still fails
#: collection loudly.
_NO_SB3_IMPLEMENTATION = "has no sb3 implementation"


def _mw_configs():
    """Every method x profile that can be pointed at Meta-World, as (label, Config)."""
    out = []
    for path in PAPER_CONFIGS:
        for profile in MW_PROFILES:
            try:
                cfg = load(path, profile=profile,
                           overrides={"problem.env_id": MW_ENV_ID})
            except ConfigError as exc:
                if _NO_SB3_IMPLEMENTATION in str(exc):
                    continue
                raise
            assert str(cfg["problem.env_id"]).startswith("mt10_")
            out.append((f"{profile}/{path.stem}", cfg))
    return out


@pytest.mark.parametrize("label,cfg", _mw_configs(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_metaworld_shape_fits_the_retained_store(label, cfg):
    """The whole iteration's replayable rollouts must fit, with no LRU involved.

    `components.training` retains rollout 0 of EVERY candidate (it cannot know
    which one `output.video.record` will pick until §4 has run), so the peak
    claim is `generate.n_candidates x (horizon + 1)`. Overflow is FIFO, so
    exceeding this does not raise -- it silently gives the early candidates'
    videos back to the same failure this file exists to close.
    """
    rows = int(cfg["generate.n_candidates"]) * (MetaWorld.horizon + 1)
    assert rows <= MetaWorld.pinned_cache_size, (
        f"{label}: {cfg['generate.n_candidates']} candidates x "
        f"{MetaWorld.horizon + 1} rows = {rows} > pinned_cache_size "
        f"{MetaWorld.pinned_cache_size}. The earliest candidates lose their video "
        "and the loss is journalled, not raised.")


def _iteration_rows(cfg) -> int:
    """What one iteration imports: every candidate's every rollout, at full horizon.

    The two floods measured in the wild -- a parallel iteration's whole import, and a
    single candidate's own rollout set -- are both this product.
    """
    return (int(cfg["generate.n_candidates"])
            * int(cfg["evaluate.rollouts_per_candidate"])
            * (MetaWorld.horizon + 1))


#: The shipped shapes that do NOT fit the LRU, filtered HERE rather than skipped
#: inside the test. A config with nothing to prove is not a gated test -- it is a
#: parametrisation that should not have been generated, and skipping it would make
#: the suite's skip count a function of the shipped CONFIGS rather than of the
#: environment.
#:
#: A skip should mean a missing dependency; a skip whose condition is config DATA
#: could never be removed by installing anything, and would read as a missing
#: dependency while the cause was a `<=` on a config value.
_OVERSIZED = [(label, cfg) for label, cfg in _mw_configs()
              if _iteration_rows(cfg) > MetaWorld.snapshot_cache_size]


def test_some_shipped_shape_exceeds_the_lru():
    """Non-empty, so the parametrisation below cannot go vacuous.

    Deliberately NOT an equality on today's count. The specific guarantee -- that the
    two shapes measured broken are still broken -- is already asserted by name in
    `test_the_two_shapes_that_were_measured_broken_are_still_in_the_corpus`, and
    pinning a number here would track the config set and go stale: it would go
    red on a config being ADDED, which is not a defect.
    """
    assert _OVERSIZED, (
        "no shipped Meta-World config exceeds snapshot_cache_size, so the retained "
        "store is unfalsifiable here and the test below would pass with "
        "`retain_states` deleted")


@pytest.mark.parametrize("label,cfg", _OVERSIZED, ids=lambda v: v if isinstance(v, str) else "")
def test_the_lru_alone_could_not_have_held_these_shapes(label, cfg):
    """The negative half, and it is the one that keeps this file honest.

    If every shipped shape happened to fit in `snapshot_cache_size`, the test
    above would pass with `retain_states` deleted. This asserts that at least the
    two known-broken shapes do NOT fit the LRU, so the retained store is doing
    real work rather than shadowing a cache that was already big enough.
    """
    assert int(cfg["generate.n_candidates"]) * (MetaWorld.horizon + 1) \
        <= MetaWorld.pinned_cache_size


def test_the_two_shapes_that_were_measured_broken_are_still_in_the_corpus():
    """A guard on the guard: if `limen` stopped asking for 50 rollouts and
    `eureka` for its candidate count, the tests above would go green for the
    wrong reason and the mechanism could be deleted unnoticed.

    Loaded BY BARE NAME and with NO profile. Both are facts about the published
    method -- neither key is stated by any of `configs/_profiles/*.yaml`, and a
    profile that started stating one would be a profile that had become a
    method.
    """
    limen = load("limen")
    eureka = load("eureka")
    assert int(limen["evaluate.rollouts_per_candidate"]) * (MetaWorld.horizon + 1) \
        > MetaWorld.snapshot_cache_size, (
        "limen's own rollout set no longer overflows the LRU; the "
        "single-candidate half of this file no longer proves anything")
    assert int(eureka["generate.n_candidates"]) \
        * int(eureka["evaluate.rollouts_per_candidate"]) * (MetaWorld.horizon + 1) \
        > MetaWorld.snapshot_cache_size, (
        "eureka's iteration no longer overflows the LRU on import")


# --------------------------------------------------------------------------
# the mechanism, against the real simulator
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simulator():
    return pytest.importorskip("metaworld", reason="needs `uv sync --extra metaworld`")


@pytest.fixture(scope="module")
def env(simulator) -> MetaWorld:
    return MetaWorld(TASK)


def _episode(env: MetaWorld, seed: int, n: int) -> np.ndarray:
    """`n + 1` genuinely emitted states, the shape `_rollout` produces."""
    rng = np.random.default_rng(seed)
    s = env.reset(rng)
    rows = [np.asarray(s, dtype=float)]
    for _ in range(n):
        a = np.asarray(env.action_set[0], dtype=float)
        a = rng.uniform(env.action_low, env.action_high, size=np.shape(a))
        s, _done, _info = env.step(s, a)
        rows.append(np.asarray(s, dtype=float))
    return np.asarray(rows)


@pytest.fixture
def tiny_cache(env, monkeypatch):
    """Shrink both stores rather than emit 11k states.

    The bug is a ratio between what a stage emits and what the cache holds, so
    it reproduces identically at 1/128 scale and in ~2 s instead of ~90 s. The
    instance attributes shadow the class ones, so nothing leaks into the other
    Meta-World tests sharing this adapter.
    """
    monkeypatch.setattr(env, "snapshot_cache_size", 64, raising=False)
    monkeypatch.setattr(env, "pinned_cache_size", 128, raising=False)
    yield env
    env._cache.clear()
    env._pinned.clear()
    env._cur = None


def test_an_unretained_rollout_is_lost_to_the_flood(tiny_cache):
    """The failure, reproduced: what a Meta-World run does without
    `retain_states`, and why `record_skipped` exists at all."""
    env = tiny_cache
    rollout = _episode(env, seed=11, n=20)
    _episode(env, seed=12, n=90)  # the next candidate's training

    with pytest.raises(UnknownStateError):
        env.reference_reward(rollout[0])


def test_a_retained_rollout_survives_the_flood(tiny_cache):
    """The mechanism. `reference_reward` rather than `render` so the assertion holds
    on a machine with no GL context -- both go through the same `_lookup`, and the
    lookup is the whole mechanism."""
    env = tiny_cache
    rollout = _episode(env, seed=13, n=20)
    env.retain_states([rollout])
    _episode(env, seed=14, n=90)

    for i, row in enumerate(rollout):
        env.reference_reward(row)  # must not raise
    assert len(env._pinned) == len(rollout)
    assert i == len(rollout) - 1


def test_a_candidates_own_rollouts_cannot_evict_its_first_one(tiny_cache):
    """`limen`'s shape in miniature: retention has to happen when
    rollout 0 is produced, not after the rollout loop, because at
    `rollouts_per_candidate: 50` the loop itself is the flood."""
    env = tiny_cache
    first = _episode(env, seed=15, n=20)
    env.retain_states([first])
    for seed in range(16, 22):  # the remaining rollouts of the SAME candidate
        _episode(env, seed=seed, n=20)

    for row in first:
        env.reference_reward(row)


def test_retention_crosses_the_fork_boundary(simulator, tiny_cache):
    """`train.candidate_parallelism: parallel` computes the rollout in a child
    whose cache the parent never sees. A retention claim is a worker output like
    the budget delta: made in the child, meaningless unless it is sent home."""
    worker = tiny_cache
    rollout = _episode(worker, seed=31, n=20)
    worker.retain_states([rollout])
    blob = worker.export_states([rollout])
    assert blob["retained"], "export dropped the retention claim"

    parent = MetaWorld(TASK)
    parent.snapshot_cache_size = 64
    parent.pinned_cache_size = 128
    parent.import_states(blob)
    _episode(parent, seed=32, n=90)  # the next candidate, merged after this one

    for row in rollout:
        parent.reference_reward(row)


def test_a_blob_without_the_key_still_imports(tiny_cache):
    """`checkpoint.py` reuses this format, so a resume from a blob with no
    `retained` key must import rather than raise."""
    env = tiny_cache
    rollout = _episode(env, seed=41, n=5)
    blob = env.export_states([rollout])
    blob.pop("retained")
    env.import_states(blob)  # must not raise


def test_release_states_gives_the_rows_back(tiny_cache):
    """Retention is scoped to one iteration by `record_rollouts`; without the
    release the store would grow by `n_candidates x 501` rows per iteration for
    the length of the run."""
    env = tiny_cache
    rollout = _episode(env, seed=51, n=20)
    env.retain_states([rollout])
    assert env._pinned
    env.release_states()
    assert not env._pinned
    _episode(env, seed=52, n=90)
    with pytest.raises(UnknownStateError):
        env.reference_reward(rollout[0])


def test_retaining_a_state_this_adapter_never_emitted_is_a_no_op(tiny_cache):
    """A hint, never a claim of truth: retention must not be able to manufacture
    a snapshot, or `render` would replay physics that never happened."""
    env = tiny_cache
    env.retain_states([np.zeros((3, env.obs_dim))])
    assert not env._pinned
    with pytest.raises(UnknownStateError):
        env.reference_reward(np.zeros(env.obs_dim))


def test_the_retained_store_is_capped(tiny_cache):
    """Overflow drops the OLDEST claim rather than raising: a missing video must
    never cost a training run."""
    env = tiny_cache
    env.pinned_cache_size = 30
    first = _episode(env, seed=61, n=20)
    second = _episode(env, seed=62, n=20)
    env.retain_states([first])
    env.retain_states([second])
    assert len(env._pinned) == 30
    for row in second[-30:]:
        assert env._key(row) in env._pinned


# --------------------------------------------------------------------------
# the wiring: training retains, the recorder releases
# --------------------------------------------------------------------------


def test_training_retains_the_rollouts_it_is_handed():
    """`_retain_replay_states` is the handoff; WHICH rollouts is
    `_n_replayed_rollouts`' decision (tested below)."""
    from bird.components.training import _retain_replay_states
    from bird.types import Trajectory

    seen = []

    class Spy:
        def retain_states(self, arrays):
            seen.append(np.asarray(arrays[0]))

    rows = np.arange(12, dtype=float).reshape(4, 3)
    _retain_replay_states(Spy(), Trajectory(states=rows))
    assert len(seen) == 1 and np.array_equal(seen[0], rows)

    _retain_replay_states(Spy(), Trajectory(states=None))
    assert len(seen) == 1, "a stateless trajectory must not be retained"

    class Boom:
        def retain_states(self, arrays):
            raise RuntimeError("adapter said no")

    _retain_replay_states(Boom(), Trajectory(states=rows))  # a video is not worth a run
    _retain_replay_states(object(), Trajectory(states=rows))  # toy/control: no such method


def test_record_rollouts_releases_even_when_recording_is_disabled():
    """The release is in a `finally` around the whole call for a reason: the
    disabled-video, no-rundir and `record_timeout` paths all return early, and
    each would otherwise leak one iteration's claim."""
    from bird import observability

    released = []

    class Env:
        def release_states(self):
            released.append(1)

        def render(self, state):  # pragma: no cover - never reached
            raise AssertionError

    class Cfg(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    class Ctx:
        cfg = Cfg({"output.video.enabled": False})
        env = Env()
        rundir = None

    assert observability.record_rollouts(Ctx(), []) == []
    assert released == [1]


def test_release_failures_do_not_reach_the_search():
    from bird import observability

    class Env:
        def release_states(self):
            raise RuntimeError("no")

    class Cfg(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    class Ctx:
        cfg = Cfg({"output.video.enabled": False})
        env = Env()
        rundir = None

    assert observability.record_rollouts(Ctx(), []) == []


def test_only_the_rollouts_something_replays_are_retained():
    """Rollout 0 is what `record_rollouts` renders and what
    `preferences._clip_for` cuts. `evaluate.fitness.source: vlm_score` replays
    EVERY rollout it scores -- `evaluation._vlm_frames` samples frames from each
    of them -- and stage 4 runs after every candidate in the iteration has
    trained, by which point candidate 0's rollouts 1..K-1 are long evicted from
    the LRU by candidate 7's training. That eviction does not raise: the frames
    come back empty, the judgment is made from a text digest, and the fitness is
    indistinguishable. So the VLM configs have to retain the whole scored set.

    Not "always all of them", and the counterexample ships in `configs/`:
    `limen` runs `evaluate.rollouts_per_candidate: 50` x a
    500-step horizon = 25,050 rows per candidate against a 32,768-entry pinned
    store, so retaining all of them for a method that re-renders exactly one
    would FIFO-drop the rollout-0 rows its videos need."""
    from bird.components.training import _n_replayed_rollouts

    class _Cfg(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)

    vlm = _Cfg({"evaluate.fitness.source": "vlm_score",
                "evaluate.artifacts": ["scalar_metrics", "videos"],
                "evaluate.rollouts_per_candidate": 3})
    assert _n_replayed_rollouts(vlm) == 3

    # LIMEN: success_rate, 50 rollouts, nothing re-renders more than the first.
    limen = _Cfg({"evaluate.fitness.source": "success_rate",
                  "evaluate.artifacts": ["scalar_metrics"],
                  "evaluate.rollouts_per_candidate": 50})
    assert _n_replayed_rollouts(limen) == 1

    # A VLM fitness whose config does not call footage evidence gets no frames
    # either (`_vlm_frames` gates on the same key), so it retains one.
    no_video = _Cfg({"evaluate.fitness.source": "vlm_score",
                     "evaluate.artifacts": ["scalar_metrics"],
                     "evaluate.rollouts_per_candidate": 3})
    assert _n_replayed_rollouts(no_video) == 1


def test_the_shipped_vlm_configs_retain_what_their_judge_will_replay():
    """The pinned end of the same rule: whatever `configs/` actually ships must
    not be one of the silently-degrading shapes."""
    from pathlib import Path

    from bird import config as cfgmod
    from bird.components.training import _n_replayed_rollouts
    from bird.config import CONFIG_ROOT

    seen = 0
    for path in sorted(Path(CONFIG_ROOT).rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        cfg = cfgmod.load(path)
        if cfg.get("evaluate.fitness.source") != "vlm_score":
            continue
        if "videos" not in (cfg.get("evaluate.artifacts") or []):
            continue
        seen += 1
        assert _n_replayed_rollouts(cfg) == cfg["evaluate.rollouts_per_candidate"], path
    assert seen, "no vlm_score config found; this test is vacuous"


def test_best_and_worst_on_a_flat_column_records_two_clips():
    """`max` and `min` both return the FIRST extreme, so they are the same
    object exactly when every scored fitness is equal -- and a
    `[best] if worst is best` rule would then collapse the pair to ONE recording.

    That is backwards. A flat column is the case where the scalar has stopped
    discriminating, so footage of two different agents is the only evidence left
    about whether they behave differently at all. A saturating env can return
    every candidate at the same fitness, so such a rule would produce one clip
    per iteration precisely where two are needed."""
    import types

    from bird.observability import select_for_recording

    def rep(i, fit):
        return types.SimpleNamespace(
            cand_id=f"c{i}", fitness=fit,
            result=types.SimpleNamespace(trained=True, trajectories=[object()]))

    flat = [rep(i, 1.0) for i in range(9)]
    picks = select_for_recording("best_and_worst", flat)
    assert [p.cand_id for p in picks] == ["c0", "c8"], (
        "a flat column must still yield a best AND a worst, and deterministically "
        "-- candidate order is stable across train.candidate_parallelism")

    # One candidate genuinely has no counterpart, and one clip is right there.
    assert [p.cand_id for p in select_for_recording("best_and_worst", [rep(0, 1.0)])] == ["c0"]

    # The ordinary case is untouched.
    spread = [rep(0, 0.5), rep(1, 0.9), rep(2, 0.1)]
    assert [p.cand_id for p in select_for_recording("best_and_worst", spread)] == ["c1", "c2"]


def test_wandb_video_names_its_format():
    """wandb warns on `Video(path)` without `format=` and says the parameter
    becomes required in v0.20.0. Taken from the file's own suffix rather than
    from `output.video.format`, which degrades to `frames` when imageio is
    missing -- so the config key and the file on disk may disagree."""
    import inspect as _inspect

    from bird import observability

    src = _inspect.getsource(observability.WandbTracker.log_media)
    assert "format=fmt" in src and "p.suffix" in src
