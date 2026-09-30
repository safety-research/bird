"""The native-signal rule: a supervised search reads the env's OWN signal,
never the BIRD-authored task_metric. See bird/native_signal.py."""

import pytest

from bird.config import ConfigError, load
from bird import native_signal as ns


# -- the pure resolver -----------------------------------------------------

def test_discrete_kind_is_native_success():
    # kind==discrete holds ONLY where the spec pins a verified reimplementation
    # of the vendor's own success, so it is genuinely native.
    assert ns.has_native_success("discrete")
    assert ns.has_native_success("native_authored")
    assert not ns.has_native_success("continuous_only")
    assert not ns.has_native_success(None)


def test_reference_reward_presence_keys_native_reward():
    assert ns.has_native_reward("published_dense")
    assert ns.has_native_reward("tuned_dense")
    assert not ns.has_native_reward("none")   # the unsupervised-only tasks
    assert not ns.has_native_reward(None)


def test_native_resolves_success_then_reward_then_refuse():
    # ships a native success -> native_success (even if it also has a reward)
    assert ns.resolve_channel("discrete", "published_dense") == ns.NATIVE_SUCCESS
    # no native success but a shipped reward -> native_reward
    assert ns.resolve_channel("continuous_only", "published_dense") == ns.NATIVE_REWARD
    # neither -> REFUSE, never a custom channel
    assert ns.resolve_channel("continuous_only", "none") == ns.REFUSE
    assert ns.resolve_channel(None, None) == ns.REFUSE


def test_native_success_keys_exclude_custom_metric_aliases():
    # `task_success`/`score`/`consecutive_successes`/`fitness` ALIAS the BIRD
    # task_metric (custom_metric) in both backends' curve rows -- they must NOT
    # be readable as a native success flag (guards the LIMEN cascade / LaRes
    # thompson_success routing too, which share this tuple).
    for alias in ("task_success", "score", "consecutive_successes", "fitness"):
        assert alias not in ns.NATIVE_SUCCESS_KEYS
    assert "success_rate" in ns.NATIVE_SUCCESS_KEYS
    assert ns.NATIVE_REWARD_KEYS == ("gt_return",)


# -- coherence REFUSE (gym tasks are always available) ---------------------

def _load(env_id, source):
    return load("eureka", profile="tester", overrides={
        "problem.env_id": env_id, "evaluate.fitness.source": source})


def test_native_refuses_on_a_gym_none_task():
    # half_cheetah_backward: reward.human.kind: none AND continuous_only -> no
    # native signal at all; `native` must refuse, never fall back on task_metric.
    with pytest.raises(ConfigError, match="native"):
        _load("gym_half_cheetah_backward", "native")


def test_native_reward_refuses_when_no_reference_reward():
    with pytest.raises(ConfigError, match="reference reward"):
        _load("gym_half_cheetah_backward", "native_reward")


def test_native_success_refuses_on_a_continuous_only_task():
    # half_cheetah ships a reference reward but NO native success (continuous_only),
    # so an explicit native_success pin refuses rather than read a proxy.
    with pytest.raises(ConfigError, match="native success"):
        _load("gym_half_cheetah", "native_success")


def test_native_accepts_a_supervised_gym_task_as_native_reward():
    # half_cheetah (continuous_only, reward.human.kind != none): `native`
    # resolves to native_reward and validates without a native complaint.
    cfg = _load("gym_half_cheetah", "native")
    assert cfg["evaluate.fitness.source"] == "native"


def test_discrete_kind_implies_a_shipped_success_across_every_spec():
    """The provenance invariant that makes `kind: discrete` a SAFE native_success
    gate: a task is `discrete` ONLY when it pins a verified reimplementation of
    the vendor's own success (`discrete_success.shipped != null`), and
    `continuous_only` carries none. If a future spec set `kind: discrete` with
    `shipped: null` (a BIRD-authored discrete proxy), `has_native_success` would
    silently start scoring a supervised search on a non-native signal -- the exact
    bug the native-signal rule exists to stop. Pinned here so that can never happen
    unnoticed."""
    from bird.tasks import available, load as load_spec
    violations = []
    for tid in available():
        ds = load_spec(tid).discrete_success or {}
        kind, shipped = ds.get("kind"), ds.get("shipped")
        if kind == "discrete" and shipped is None:
            violations.append(f"{tid}: kind=discrete but shipped=null")
        if kind == "continuous_only" and shipped is not None:
            violations.append(f"{tid}: kind=continuous_only but shipped!=null")
        # native_authored (the toy family): repo-authored, so NO vendor-shipped
        # block -- the env's own success() is the native check. shipped must be null.
        if kind == "native_authored" and shipped is not None:
            violations.append(f"{tid}: kind=native_authored but shipped!=null "
                              "(repo-authored native success has no vendor block)")
    assert not violations, (
        "discrete_success.kind must satisfy discrete <=> shipped!=null (and "
        "native_authored => shipped null) so that native_success reads only a "
        "vendor-shipped or repo-authored success:\n  " + "\n  ".join(violations))


def test_toy_family_is_native_authored_and_resolves_native_success():
    """The rule: this repo is the vendor for the toy family, so their own
    success() IS native. Pinned so the flip can't silently regress (keeps CARD's
    TPE screen faithful on toy_reacher and the tester tier meaningful)."""
    from bird.tasks import load as load_spec
    for tid in ("toy_reacher", "toy_gridworld", "toy_hungry_thirsty"):
        spec = load_spec(tid)
        assert (spec.discrete_success or {}).get("kind") == "native_authored", tid
        ds, rw = ns.kinds_for_env(task_id=tid)
        assert ns.has_native_success(ds), tid
        assert ns.resolve_channel(ds, rw) == ns.NATIVE_SUCCESS, tid
