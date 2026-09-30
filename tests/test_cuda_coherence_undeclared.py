"""An undeclared `requires_cuda` is refused at config load, not quietly admitted.

THE ASYMMETRY. Two readers consult the same attribute on the same registered
object and can disagree about `None`:

    bird/xla_env.py   `if requires_cuda is False: return False`
                      -- None keeps the STRICT determinism call.
    bird/config.py    a bare `if needs_cuda:`
                      -- None is falsey, so such a rule would refuse nothing.

A jax-suite adapter whose author forgot the attribute would then be treated as
needing determinism by one reader and as cpu-capable by the other, while the
documentation describes both as strict -- which is what makes this worth a test
rather than a one-character fix: prose and code can disagree, and the prose is
the citable artifact.

WHY BOTH MUST AGREE ON STRICT, and not on permissive. The permissive side of a
disagreement is the quieter failure: an MJX adapter training on cpu beside an
idle GPU leaves every counter reading normal; being refused at config load is
loud, being admitted is not, and config load
is the earlier and quieter of the two readers. `False` is therefore the only
value that admits a cpu device -- an omission is not.

THE CATALOGUE TEST IS NOT THIS TEST. `tests/test_jax_toy.py::
test_every_jax_suite_env_declares_whether_it_needs_cuda` walks the registry
and fails if any REGISTERED jax-suite env omits the attribute, which makes
`None` unreachable for the envs that exist today. That is exactly why the
behaviour on `None` cannot be observed from the real registry, and why these
tests register stand-ins: without them the rule's `None` branch is dead code
that no run can reach, and dead code is where a wrong answer hides.
"""

from __future__ import annotations

import pytest

from bird import registry
from bird.config import Config, validate
from bird.envs.suites import env_suite

_UNDECLARED = "jax_standin_undeclared"
_DECLARED_FALSE = "jax_standin_cpu"
_DECLARED_TRUE = "jax_standin_gpu"
#: The `upstream_assistax_*` family. A jax-tier family whose ids do NOT begin
#: with `jax_`, which is the whole point: the refusal is entered through
#: `env_suite`, so a jax family the suite table does not place is never checked
#: at all.
_UPSTREAM_TRUE = "upstream_assistax_standin_gpu"
_UPSTREAM_UNDECLARED = "upstream_assistax_standin_undeclared"


class _Undeclared:
    """A jax-suite adapter whose author forgot `requires_cuda`. No attribute at all."""


class _DeclaredFalse:
    requires_cuda = False


class _DeclaredTrue:
    requires_cuda = True


@pytest.fixture
def standins():
    """Register the three stand-ins and REMOVE them again, whatever happens.

    The catalogue test above walks the whole registry, so a leaked
    `_UNDECLARED` would fail it -- in a different file, with a message about
    an adapter nobody wrote. The restore is a try/finally over a snapshot of
    the keys rather than a `del` of the three names, so a registration that
    raises part way through still cannot leave one behind.
    """
    registry.load_all()
    before = set(registry._REGISTRY)
    try:
        registry._REGISTRY[("env", _UNDECLARED)] = _Undeclared
        registry._REGISTRY[("env", _DECLARED_FALSE)] = _DeclaredFalse
        registry._REGISTRY[("env", _DECLARED_TRUE)] = _DeclaredTrue
        registry._REGISTRY[("env", _UPSTREAM_TRUE)] = _DeclaredTrue
        registry._REGISTRY[("env", _UPSTREAM_UNDECLARED)] = _Undeclared
        yield
    finally:
        for key in set(registry._REGISTRY) - before:
            registry._REGISTRY.pop(key, None)


def _problems(env_id, device=None):
    data = {"problem": {"env_id": env_id},
            "generate": {"reward_language": "jax"},
            "train": {"hyperparameters": {}}}
    if device is not None:
        data["train"]["hyperparameters"]["device"] = device
    try:
        validate(Config(data))
    except Exception as exc:  # noqa: BLE001 - ConfigError carries every problem at once
        return str(exc)
    return ""


def test_the_standins_are_jax_suite_envs(standins):
    """The premise, asserted rather than assumed.

    Every test below is about what the rule does to a JAX-SUITE env. If
    `env_suite` stopped mapping these ids to "jax" -- the mapping is by
    prefix -- the other tests would pass by never entering the branch at all,
    which is the failure mode this whole rule exists to prevent (a refusal
    keyed on the NAME doing the work of one keyed on CAPABILITY).
    """
    for env_id in (_UNDECLARED, _DECLARED_FALSE, _DECLARED_TRUE):
        assert env_suite(env_id) == "jax", (
            f"{env_id} is not classified into the jax suite, so the coherence "
            "rule's branch is never entered and these tests check nothing")


def test_an_undeclared_jax_env_is_refused_and_the_message_names_the_attribute(standins):
    out = _problems(_UNDECLARED, device="cuda")
    assert "requires_cuda" in out and _UNDECLARED in out, (
        "a jax-suite env declaring no `requires_cuda` was admitted (or refused "
        f"without naming the attribute or the env). Got: {out!r}")


def test_the_refusal_does_not_depend_on_the_device(standins):
    """cuda does not buy an undeclared adapter a pass.

    The missing DECLARATION is the defect, not the device that happens to be
    configured beside it -- an adapter nobody declared is one nothing can
    check, and a run that happens to be pointed at cuda today says nothing
    about the next one. This is the half a truthy test could not express even
    with the None case added: it would have refused only the cpu pairing.
    """
    for device in ("cuda", "cpu", "auto", None):
        out = _problems(_UNDECLARED, device=device)
        assert "requires_cuda" in out, (
            f"undeclared env admitted at device={device!r}: {out!r}")


def test_a_declared_false_jax_env_is_admitted_on_cpu(standins):
    """False is the value that admits cpu, and it must keep doing so.

    This is why the refusal keys on the declaration: one keyed on the env id's
    `jax_` prefix would catch `jax_toy`, an offline tester-tier env computing
    in jnp with no device hand-off, making `configs/examples/jax_reward.yaml`
    unloadable.
    """
    out = _problems(_DECLARED_FALSE, device="cpu")
    assert "requires_cuda" not in out, (
        f"a jax env declaring `requires_cuda = False` was refused a cpu device: {out!r}")


def test_a_non_jax_env_that_declares_nothing_is_still_admitted(standins):
    """THE BLAST RADIUS.

    `needs_cuda` is initialised to None and is only ever assigned inside
    `if is_jax_env:`, so it STAYS None for every non-jax env in the repo. A
    refusal reading a bare `if needs_cuda is None:` would fire for
    `toy_reacher`, `pendulum` and every config, and the four tests above would
    not notice, because they exercise jax-suite stand-ins only: they test the
    branch, not what it does to everything else.

    `toy_reacher` is the sharpest case available: `env_suite` puts it in the
    "toy" suite, it declares no `requires_cuda` and never should, and it is
    the env most of the tester tier runs on.
    """
    assert env_suite("toy_reacher") != "jax", (
        "toy_reacher is no longer a non-jax env, so this test no longer checks "
        "what it is named for -- pick another non-jax env id")
    out = _problems("toy_reacher", device="cpu")
    assert "requires_cuda" not in out, (
        f"a NON-jax env was refused for not declaring `requires_cuda`: {out!r}")


def test_a_declared_true_jax_env_still_needs_cuda(standins):
    """The declared case: `requires_cuda = True` on cpu is still refused."""
    out = _problems(_DECLARED_TRUE, device="cpu")
    assert "requires_cuda" in out and "cuda" in out, (
        f"a jax env declaring `requires_cuda = True` was admitted on cpu: {out!r}")
    assert "requires_cuda" not in _problems(_DECLARED_TRUE, device="cuda")


# ---------------------------------------------------------------------------
# The `upstream_assistax_*` family: a jax family NOT named `jax_*`.
#
# The rule is entered through `env_suite(problem.env_id) == "jax"`, and
# `env_suite` places by PREFIX. So the set of families this refusal covers is
# exactly the prefix tuple on the `jax` row of `bird/envs/suites.py` -- a
# family whose prefix is missing from it is not weakly checked, it is not
# checked at all, and `requires_cuda` on its adapter is read by nobody at load
# time. That is the idle-GPU failure one level up: the guard exists, the family
# is outside it, and every counter reads normal.
#
# These stand-ins are the `upstream_assistax_` prefix rather than a made-up one
# on purpose -- the covered set is the thing under test, so the test has to
# name a real member of it.
# ---------------------------------------------------------------------------


def test_the_upstream_assistax_family_is_in_the_jax_suite(standins):
    """The premise: the prefix has to be on the jax row.

    Asserted separately from the refusal below for the reason
    `test_the_standins_are_jax_suite_envs` exists: if this stops holding, the
    refusal test stops entering the branch and would go green by checking
    nothing.
    """
    for env_id in (_UPSTREAM_TRUE, _UPSTREAM_UNDECLARED):
        assert env_suite(env_id) == "jax", (
            f"{env_id} is not placed in the jax suite, so `_check_coherence` "
            "never reads its `requires_cuda` and a cuda-only adapter loads "
            "clean on a cpu device -- add the prefix to the `jax` row of "
            "bird/envs/suites.py")


@pytest.mark.parametrize("device", ["cpu", "auto", None])
def test_an_upstream_assistax_env_declaring_cuda_is_refused_on_a_non_cuda_device(
        standins, device):
    """A jax family outside the suite table, as a test.

    `requires_cuda` is asserted IN the message, not merely "it refused":
    `generate.reward_language: jax` on an env the suite table does not place
    is itself a refusal that names the env id, so a test asserting only that
    loading fails and that the id appears would pass with the prefix missing
    -- the exact thing this is here to catch.
    """
    out = _problems(_UPSTREAM_TRUE, device=device)
    assert "requires_cuda" in out, (
        f"an `upstream_assistax_*` env declaring `requires_cuda = True` was "
        f"admitted at device={device!r} (or refused for some other reason): {out!r}")
    assert _UPSTREAM_TRUE in out, (
        f"the refusal does not name the env, so an operator reading it cannot "
        f"tell which of the configured ids is the cuda-only one: {out!r}")


def test_an_upstream_assistax_env_declaring_cuda_is_admitted_on_cuda(standins):
    """The other side: the widened prefix must not refuse the correct pairing."""
    assert "requires_cuda" not in _problems(_UPSTREAM_TRUE, device="cuda")


def test_an_undeclared_upstream_assistax_env_is_refused(standins):
    """UNDECLARED stays strict on this family too: `False` is the only value
    that admits a cpu device, and an adapter that never declared is refused
    regardless of the device."""
    for device in ("cuda", "cpu", "auto", None):
        out = _problems(_UPSTREAM_UNDECLARED, device=device)
        assert "requires_cuda" in out and _UPSTREAM_UNDECLARED in out, (
            f"an undeclared `upstream_assistax_*` env was admitted at "
            f"device={device!r}: {out!r}")
