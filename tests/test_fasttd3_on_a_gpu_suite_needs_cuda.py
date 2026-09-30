"""A torch learner on a GPU suite must SAY cuda, before the spend.

THE FAILURE MODE. `bird/components/fasttd3.py` defaults
`train.hyperparameters.device` to `"cpu"`, so a launch that does not set it
trains on the CPU with a GPU idle beside it, and the downgrade also disables
AMP and `torch.compile` -- both derived from `device.type == "cuda"` two lines
later. Nothing else in the artifact says so: only the seed row's
`device`/`learner_device`, compared against intent, does.

`learner_device` on each seed row makes that VISIBLE after the fact,
compared against intent (`tests/test_learner_device_provenance.py`). This
refuses it BEFORE the fact, which is the half a report cannot do.

THE SCOPE. `_check_coherence` refuses `train.backend` in {fasttd3, simba_v2}
on a `humanoid_bench` or `assistax` suite env unless
`train.hyperparameters.device` starts with `cuda`; `auto` is refused there too.
On HumanoidBench it also requires `train.n_parallel_envs >= 8`, because an
unparallelised HumanoidBench run is env-bound. Every other suite and every
other backend is untouched, so `auto` keeps its fallback on a laptop or in CI.

`--validate-all` covers no fasttd3 config on these suites, so **these tests
are the only evidence the rule works**, which is why each one is written to
fail for a distinct reason rather than to cover a line.
"""

from __future__ import annotations

import pytest

from bird.config import Config, _GPU_SUITES, validate
from bird.envs.suites import env_suite


def _cfg(env_id, backend="fasttd3", device=None, n_parallel_envs=None):
    data = {
        "problem": {"env_id": env_id},
        "train": {"backend": backend, "hyperparameters": {}},
    }
    if device is not None:
        data["train"]["hyperparameters"]["device"] = device
    if n_parallel_envs is not None:
        data["train"]["n_parallel_envs"] = n_parallel_envs
    return Config(data)


def _problems(cfg):
    try:
        validate(cfg)
    except Exception as exc:  # noqa: BLE001 - ConfigError carries every problem
        return str(exc)
    return ""


def _device_problems(cfg):
    return [ln for ln in _problems(cfg).splitlines()
            if "hyperparameters.device" in ln]


def _parallel_problems(cfg):
    return [ln for ln in _problems(cfg).splitlines()
            if "n_parallel_envs >= 8" in ln]


# -- it fires ---------------------------------------------------------------


@pytest.mark.parametrize("backend", ["fasttd3", "simba_v2"])
@pytest.mark.parametrize("env_id,suite", [
    ("h1hand_basketball", "humanoid_bench"),
    ("assistax_scratchitch", "assistax"),
])
def test_the_default_cpu_device_is_refused_on_a_gpu_suite(env_id, suite, backend):
    """The exact failure shape: no device key, so the learner's `cpu` default.

    Parametrised over the suites and both torch backends, named individually
    so that a suite dropped from `_GPU_SUITES` fails as itself rather than
    lowering a count.
    """
    assert env_suite(env_id) == suite, "this test's premise moved"
    assert _device_problems(_cfg(env_id, backend=backend)), (
        f"a {backend} run on {suite} with no device key was accepted; that is "
        f"a cpu learner and a GPU reservation")


def test_an_explicit_cpu_is_refused_too_not_only_an_absent_key():
    """Saying `cpu` out loud on a GPU suite is not consent, it is a typo.

    A rule that only caught the ABSENT key would pass the one config whose
    author thought about the question and got it wrong.
    """
    assert _device_problems(_cfg("h1hand_basketball", device="cpu"))


def test_auto_is_refused_on_these_suites_and_the_narrowing_is_deliberate():
    """`auto` means "a GPU if there is one" -- i.e. cpu, silently, if not.

    THIS NARROWS `tests/test_learner_device_provenance.py::
    test_auto_keeps_its_fallback`, which is right in general and stays so:
    `auto` on a laptop or in CI is the entire point of the value, and
    `test_every_other_suite_is_untouched` asserts it still passes everywhere
    else. The narrowing is scoped to suites where a silent cpu run costs
    GPU-hours.
    """
    assert _device_problems(_cfg("h1hand_basketball", device="auto"))


def test_the_message_names_the_suite_and_the_value_it_got():
    """Whoever reads this is deciding whether to relaunch."""
    msg = "\n".join(_device_problems(_cfg("h1hand_basketball", device="auto")))
    assert "humanoid_bench" in msg
    assert "auto" in msg
    assert "cuda" in msg


# -- it does not fire -------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:3"])
def test_an_explicit_cuda_passes(device):
    assert not _device_problems(_cfg("h1hand_basketball", device=device))
    assert not _device_problems(_cfg("assistax_scratchitch", device=device))


@pytest.mark.parametrize("device", [None, "cpu", "auto"])
def test_every_other_suite_is_untouched(device):
    """The blast radius, asserted rather than assumed.

    `pendulum` is the suite the whole test tier runs on; if this rule reached
    it, every cheap config in the repo would need a device key.
    """
    assert env_suite("pendulum") not in _GPU_SUITES
    assert not _device_problems(_cfg("pendulum", device=device))
    assert not _device_problems(_cfg("pendulum", backend="simba_v2", device=device))


@pytest.mark.parametrize("backend", ["mock", "tabular", "sb3"])
def test_another_backend_on_a_gpu_suite_is_untouched(backend):
    """The rule is about the torch learners' `cpu` default, not about the suite.

    The surrogates are the tester tier and have no device key at all;
    refusing them would fail every mock run on an HB env, which is how the
    HumanoidBench tests are written. An sb3 run on HumanoidBench stays legal.
    """
    assert not _device_problems(_cfg("h1hand_basketball", backend=backend))
    assert not _parallel_problems(_cfg("h1hand_basketball", backend=backend))


# -- HumanoidBench must parallelise its envs --------------------------------


@pytest.mark.parametrize("backend", ["fasttd3", "simba_v2"])
@pytest.mark.parametrize("n", [None, 1, 7])
def test_an_unparallelised_humanoidbench_run_is_refused(backend, n):
    """One simulator feeding a GPU learner is env-bound; the floor is 8."""
    problems = _parallel_problems(
        _cfg("h1hand_basketball", backend=backend, device="cuda", n_parallel_envs=n))
    assert problems, f"{backend} on HumanoidBench with n_parallel_envs={n} was accepted"
    assert backend in problems[0]


@pytest.mark.parametrize("n", [8, 16, 24])
def test_a_parallelised_humanoidbench_run_passes(n):
    """16 is RDA's SimbaV2 recipe, 24 the paper's FastTD3 runs."""
    for backend in ("fasttd3", "simba_v2"):
        cfg = _cfg("h1hand_basketball", backend=backend, device="cuda", n_parallel_envs=n)
        assert not _parallel_problems(cfg)
        assert not _device_problems(cfg)


def test_the_parallelism_floor_is_humanoidbench_only():
    """Assistax is a GPU suite but not env-bound in the same way: only the
    device half of the rule applies there."""
    assert not _parallel_problems(_cfg("assistax_scratchitch", device="cuda"))
    assert not _parallel_problems(_cfg("assistax_scratchitch", device="cuda", n_parallel_envs=1))


# -- the suite set ----------------------------------------------------------


def test_the_suite_set_is_exactly_the_two_gpu_suites():
    """Pinned as a literal so a suite added or dropped is a visible decision.

    The `jax` suite (`upstream_assistax_*`, `jax_toy`) is deliberately absent:
    its cuda rule is the adapter's own `requires_cuda` declaration.
    """
    assert _GPU_SUITES == {"humanoid_bench", "assistax"}
    assert env_suite("jax_toy") not in _GPU_SUITES


def test_the_suite_set_is_resolved_through_env_suite_not_an_id_prefix():
    """A second copy of a prefix list goes stale when the suite grows.

    `humanoid_bench` spans `h1strong_` and `h1simplehand_` as well as
    `h1hand_`, and a hard-coded prefix guess here would miss ids like these.
    This asserts the wiring by using an id no prefix guess would catch.
    """
    assert env_suite("h1strong_highbar_hard") == "humanoid_bench"
    assert _device_problems(_cfg("h1strong_highbar_hard")), (
        "an id outside a narrow h1 prefix guess was not covered, which means "
        "this rule is reading a prefix somewhere instead of the suite table")
