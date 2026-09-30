"""`policy_from_ref` must not hand a fasttd3 agent to the sb3 rebuilder.

THE HAZARD THIS FILE GUARDS. `training.policy_from_ref` is described --
in its own docstring and again at `envs/base.py::EnvAdapter.dr_probe` -- as the ONE place that
knows every blob kind, "so a fourth reader cannot drift from it". Shape and
dtype alone cannot discriminate: 1-D uint8 is sb3, a 2-D table is the
Q-learner, `(action_dim, obs_dim + 1)` is the CEM linear policy -- and

**a fasttd3 blob is also 1-D uint8.** `fasttd3._fasttd3_blob` writes
`torch.save({...})` into a `_PolicyBlob`, and `torch.save` writes a ZIP, so
even the magic bytes agree with an sb3 archive. A shape-and-dtype
discriminator would not reject the blob -- it would MATCH THE WRONG BRANCH,
which is worse than not matching, because `envs/base.py::EnvAdapter.dr_probe`
states the caller's safety property as "a blob that cannot be rebuilt RAISES
(`policy_from_ref` never degrades)".

WHO WOULD BE AFFECTED: `EnvAdapter.dr_probe` (`envs/base.py`) and
`components/phases.py` both call `policy_from_ref`, so on any fasttd3 run
DrEureka's RAPP sweep would reach the sb3 rebuild -- or, if `dr_probe` knew
only some kinds and returned NaN for the rest, RAPP would become the
uninformative-prior arm on the one backend that trains a real network.

WHERE THE FIXTURE COMES FROM, because a fixture the author invented agrees
with the author by construction. The archive shape below was READ OFF A REAL
fasttd3 result (~300 MB), whose `payload["policy"]` is one `_PolicyBlob` of
298,244,674 bytes -- dtype uint8, ndim 1, first bytes `PK\\x03\\x04`, and a zip
of 81 entries all under one top-level directory `archive/`, beginning
`archive/data.pkl`, `archive/.format_version`, `archive/byteorder`,
`archive/data/0`... That is `torch.save`'s container and nothing else writes
it. An sb3 archive has top-level members instead (`data`, `policy.pth`,
`_stable_baselines3_version`).

So the discriminator does not need torch, which matters: the default test
install (`--extra test`) has neither torch nor stable-baselines3, and a test
that `importorskip`s them would print as a dot and prove nothing.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

from bird.components import training as T

#: The measured member list of a real fasttd3 blob, trimmed to the entries
#: that identify the container. Every one of these was present in the real
#: result described in the module docstring.
_TORCH_ZIP_MEMBERS = ("archive/data.pkl", "archive/.format_version",
                      "archive/byteorder", "archive/data/0")

#: What an sb3 archive carries instead -- top-level members, no `archive/`.
_SB3_ZIP_MEMBERS = ("data", "policy.pth", "_stable_baselines3_version")


def _zip_blob(members) -> np.ndarray:
    """A `_PolicyBlob`-shaped 1-D uint8 array holding a zip with `members`."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in members:
            z.writestr(name, b"\x00")
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def test_a_fasttd3_blob_is_shaped_exactly_like_an_sb3_one():
    """The premise, asserted rather than assumed.

    If this ever stops holding -- a fasttd3 blob that is not 1-D uint8, or not
    a zip -- then the hazard is gone by construction and the guard can be
    reconsidered. While it holds, shape and dtype cannot separate the two and
    any discriminator that uses only those is wrong.
    """
    fasttd3 = _zip_blob(_TORCH_ZIP_MEMBERS)
    sb3 = _zip_blob(_SB3_ZIP_MEMBERS)
    for blob in (fasttd3, sb3):
        assert blob.dtype == np.uint8 and blob.ndim == 1
        assert bytes(blob[:4]) == b"PK\x03\x04", (
            "both containers are zips, so the magic bytes do not separate "
            "them either")


def test_a_fasttd3_blob_never_reaches_the_sb3_rebuild(monkeypatch):
    """The hazard: a fasttd3 blob must never be routed to the sb3 rebuild.

    Asserted as "did not enter the sb3 branch" rather than "raised", because
    the two are different failures and only one of them is this one: a raise
    from inside the sb3 loader would mean the blob got there and the loader
    happened to reject it, which is luck rather than a discriminator.
    """
    entered = []
    monkeypatch.setattr(T, "_sb3_policy_from_blob",
                        lambda *a, **k: entered.append(a) or "sb3 policy")

    blob = _zip_blob(_TORCH_ZIP_MEMBERS)
    # THE ROUTING IS ASSERTED FIRST, and the order matters. Written with
    # `pytest.raises` on the outside, a misrouting implementation reports "DID
    # NOT RAISE" -- true, and the wrong headline: with the sb3 loader stubbed
    # out, it RETURNS a policy for a fasttd3 blob. In production the real
    # loader would probably reject the archive, so the defect would read as an
    # sb3 loading failure rather than as a misroute, and a fix would be
    # attempted in the wrong function.
    outcome = "raised"
    try:
        outcome = T.policy_from_ref(None, object(), blob, None,
                                    ref="policy:c0003")
    except ValueError as exc:
        outcome = exc

    assert not entered, (
        "a fasttd3 blob entered `_sb3_policy_from_blob` and that call "
        f"returned {outcome!r}. It is 1-D uint8 like an sb3 blob, so shape "
        "and dtype cannot tell them apart -- the discriminator has to be the "
        "container's own members (torch.save writes everything under "
        "`archive/`) or the payload's `format: fasttd3/v1` marker, which "
        "`fasttd3._fasttd3_payload` already checks")
    assert isinstance(outcome, ValueError), (
        f"expected a refusal, got {outcome!r}")
    assert "policy:c0003" in str(outcome), (
        "`policy_from_ref` names the ref in every refusal it makes; a "
        "fasttd3 blob must be refused the same way rather than silently "
        "rebuilt as something else")


def test_an_sb3_blob_still_reaches_the_sb3_rebuild(monkeypatch):
    """The negative half. A discriminator that refused everything would pass
    the test above and break every sb3 warm start."""
    entered = []
    monkeypatch.setattr(T, "_sb3_policy_from_blob",
                        lambda *a, **k: entered.append(a) or "sb3 policy")

    blob = _zip_blob(_SB3_ZIP_MEMBERS)
    out = T.policy_from_ref(None, object(), blob, None, ref="policy:c0000")
    assert entered, "an sb3 blob must still be rebuilt by the sb3 branch"
    assert out == "sb3 policy"


# -- a real fasttd3 blob rebuilds and ACTS through the loop's own path --------------


def _tiny_fasttd3_run():
    """A real fasttd3 training on CPU, the shape `tests/test_fasttd3.py` uses, so the
    blob under test is one the backend wrote and not one this file invented."""
    import test_fasttd3 as FT_TESTS  # tests/ is on the path (conftest's own precedent)

    ctx = FT_TESTS._ctx(**{"train.hyperparameters": {**FT_TESTS.TINY, "num_envs": 2},
                           "train.env_steps": 200})
    res = FT_TESTS._train(ctx, "c0003")
    assert res.trained, res.error
    assert res.policy_ref and res.policy_ref in T._POLICY_STORE
    return ctx, res


def test_a_real_fasttd3_blob_rebuilds_through_its_own_branch_and_acts():
    """On a blob the backend wrote: `policy_from_ref` routes it to the
    fasttd3 branch (never the sb3 one), the rebuilt callable maps a raw state to a
    finite action in the adapter's box, deterministically, and it is the same
    numbers the loop's own action path produces from the same weights -- so the
    rolled-out policy IS the trained one, not a second implementation."""
    pytest.importorskip("torch")
    import bird.components.fasttd3 as FT

    ctx, res = _tiny_fasttd3_run()
    blob = T._POLICY_STORE[res.policy_ref]
    assert blob.dtype == np.uint8 and blob.ndim == 1
    # `"torch"`, not `"fasttd3"`: `simba_v2._blob` writes the same container,
    # so `_blob_kind` names the SERIALISER and the learner comes from the
    # payload's own `format` tag one level up.
    assert T._blob_kind(blob) == "torch"

    entered = []
    real_sb3 = T._sb3_policy_from_blob
    T._sb3_policy_from_blob = lambda *a, **k: entered.append(a)  # must not be reached
    try:
        policy = T.policy_from_ref(ctx.cfg, ctx.env, blob, None, ref=res.policy_ref)
    finally:
        T._sb3_policy_from_blob = real_sb3
    assert not entered, "a fasttd3 blob reached the sb3 rebuild"

    env = ctx.env
    s = env.reset(np.random.default_rng(0))
    a1, a2 = policy(s), policy(s)
    assert a1.shape == (int(np.asarray(env.action_low).size),)
    assert np.all(np.isfinite(a1)) and np.array_equal(a1, a2), "greedy: deterministic"
    lo, hi = np.asarray(env.action_low), np.asarray(env.action_high)
    center, half = FT._action_affine(env)
    assert np.all(a1 >= center - half - 1e-6) and np.all(a1 <= center + half + 1e-6), (
        "the actor's tanh output maps into the adapter's box, unclamped")
    assert np.all(lo <= center) and np.all(center <= hi)

    # THE SAME ACTION PATH: the loop's `_greedy_policy` over the payload's own
    # weights, built by the loop's own constructor, gives the same numbers.
    ns = FT._torch_classes()
    payload = FT._fasttd3_payload(ns, FT._resolve_arch(ctx.cfg), blob, res.policy_ref)
    hyper = FT._fasttd3_hyper(ctx.cfg)
    actor, _q, _qt, norm = FT._build_networks(ns, FT._resolve_arch(ctx.cfg), hyper,
                                              int(np.asarray(env.obs_low).size),
                                              int(np.asarray(env.action_low).size),
                                              int(hyper["num_envs"]), ns["torch"].device("cpu"))
    sd = dict(payload["actor"]); sd["noise_scales"] = actor.state_dict()["noise_scales"]
    actor.load_state_dict(sd)
    if payload.get("obs_normalizer") and norm is not None:
        norm.load_state_dict(payload["obs_normalizer"])
    actor.eval()
    direct = FT._greedy_policy(ns, actor, norm, lambda x: np.asarray(x, dtype=np.float32),
                               lambda a: center + np.asarray(a, dtype=float) * half,
                               ns["torch"].device("cpu"))
    assert np.allclose(direct(s), a1, atol=1e-6)

    # ... and the live caller: `EnvAdapter.dr_probe` returns a finite rate, not NaN.
    rate = env.dr_probe(res.policy_ref, {}, n_rollouts=1, cfg=ctx.cfg)
    assert np.isfinite(rate) and 0.0 <= rate <= 1.0, rate


def test_a_torch_container_of_neither_tag_is_refused_AT_THE_ROUTING_SITE():
    """The container says torch; the payload's own marker says neither learner.
    Refused with the ref and the marker named, and refused by the ROUTING rather
    than by whichever learner happened to be the fall-through.

    The distinction is the whole test. With fasttd3 as the else branch, an
    unknown tag would reach fasttd3's own format check and be refused there --
    a refusal either way, but not the one `blob_format`'s docstring and the
    routing comment describe. There are two branches and a raise, and this test
    pins the message that says so -- naming the tag it FOUND and the two tags
    that exist, rather than one learner's format for a blob that is not that
    learner's.
    """
    torch = pytest.importorskip("torch")
    import io as _io

    buf = _io.BytesIO()
    torch.save({"format": "somebody-else/v9", "actor": {}}, buf)
    blob = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    assert T._blob_kind(blob) == "torch", "torch's container, whatever the payload says"
    from bird.config import load

    cfg = load("rda", profile="dev", overrides={"problem.env_id": "pendulum",
                                                 "train.backend": "fasttd3",
                                                 "train.algorithm": "fasttd3",
                                                 "output.tracker": "none"})
    from bird import registry
    env = registry.get("env", "pendulum")(None)
    with pytest.raises(ValueError, match=r"policy:c0009.*somebody-else/v9"):
        T.policy_from_ref(cfg, env, blob, None, ref="policy:c0009")
    # and the message says which tags DO exist, so a reader of the failure knows
    # what a rebuildable blob looks like without opening the source.
    try:
        T.policy_from_ref(cfg, env, blob, None, ref="policy:c0009")
    except ValueError as exc:
        assert "fasttd3/v1" in str(exc) and "simba_v2/v1" in str(exc), str(exc)


def test_a_blob_of_neither_kind_is_refused_by_name():
    """1-D uint8, a zip, and neither learner's members: refused, naming the ref and
    both containers it is not."""
    blob = _zip_blob(("README", "weights.bin"))
    assert T._blob_kind(blob) == "unknown"
    with pytest.raises(ValueError, match=r"policy:c0007.*neither an sb3 archive.*nor a torch container"):
        T.policy_from_ref(None, object(), blob, None, ref="policy:c0007")
    garbage = np.frombuffer(b"not a zip at all", dtype=np.uint8)
    with pytest.raises(ValueError, match=r"policy:c0008.*neither"):
        T.policy_from_ref(None, object(), garbage, None, ref="policy:c0008")
