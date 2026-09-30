"""LIMEN's cascade must actually get the short budget it asks for.

The failure this file exists to prevent: a live LIMEN run on Meta-World whose
every journalled `screen_pass` reads

    "screen": "cascade", "success_rate": 0.98, "short_budget_honoured": false

against `verify.cascade.short_budget_steps: 10000` and `train.env_steps: 100000`
-- the screen paying TEN TIMES its intended cost, on every candidate.

`screens._train_backend_short` passes the short budget only if the backend
advertises one of `env_steps`/`steps`/`max_steps`/`budget_steps` in its signature;
otherwise the backend reads `train.env_steps` for itself. A backend whose
signature is `(ctx, state, candidate, n_seeds=1)` never matches the probe, on ANY
env tier. (Whether the budget is then HONOURED is measured off the backend's
own rows, not read off the signature match -- that half lives
in `tests/test_cascade_measured_spend.py`, which is not slow-marked; this file
guards the half the signature can answer: that there is a parameter to pass.)

Why this rates a dedicated test rather than a line in an existing one: LIMEN's
contribution IS the cheap pre-filter, exactly as CARD's is the trainings it skips
(`budget.policy_trainings_skipped`) and Gran Turismo's is its alignment filter. A
method whose mechanism silently does not run still produces a fitness number and a
full artifact, so nothing fails -- the run completes, the dashboard fills, and the
method point is quietly measuring something else. `_train_backend_short`'s own
docstring names the hazard ("claiming a 1M-step screen that silently ran the full
25M would misreport exactly the quantity LIMEN's cascade exists to save"); a
docstring cannot fail, so the claim is asserted here.

The parametrisation over `registry.available("train_backend")` is deliberate: a
NEW backend added later inherits this test automatically, and fails it if it
copies a four-argument signature.
"""

from __future__ import annotations

import inspect

import pytest

from bird import registry

#: Not in the tester-tier smoke suite (heavyweight execution: repeated searches under a shrinking budget).
#: Deselected by `-m "not slow"`; run it with `-m slow`. See pyproject.toml.
pytestmark = pytest.mark.slow

#: The names `screens._train_backend_short` probes for, in its own order. Kept as a
#: literal rather than imported so that widening the probe in the screen does not
#: silently widen this test too -- if these drift apart, one of them is wrong and a
#: reader should have to look at both.
_BUDGET_PARAMS = ("env_steps", "steps", "max_steps", "budget_steps")


@pytest.fixture(scope="module", autouse=True)
def _loaded() -> None:
    registry.load_all()


def _backends():
    registry.load_all()
    # Every registered backend is a LEARNER: this file parametrises over them all.
    return sorted(n for (k, n) in registry._REGISTRY if k == "train_backend")


def test_there_are_backends_to_check() -> None:
    """Guard the guard: a parametrised test over an empty list passes vacuously."""
    assert _backends(), "no train_backend registered; the rest of this file is vacuous"


@pytest.mark.parametrize("name", _backends())
def test_every_backend_accepts_a_short_budget_override(name: str) -> None:
    """The probe in `_train_backend_short` must match, or the cascade is a no-op.

    Asserted against the signature rather than by running a search because that is
    precisely what the screen inspects when deciding whether it CAN pass the
    override -- testing the same predicate the production code branches on is the
    point, not an approximation of it. Whether the backend then honoured it is a
    different question, answered by measurement in
    `tests/test_cascade_measured_spend.py`.
    """
    fn = registry.get("train_backend", name)
    params = inspect.signature(fn).parameters
    matched = [p for p in _BUDGET_PARAMS if p in params]
    assert matched, (
        f"train_backend {name!r} has signature ({', '.join(params)}) and advertises "
        f"none of {_BUDGET_PARAMS}. screens._train_backend_short will therefore set "
        f"short_budget_honoured=False and the backend will read train.env_steps for "
        f"itself, so verify.cascade.short_budget_steps is silently ignored and "
        f"LIMEN's pre-filter costs a FULL training per candidate."
    )


@pytest.mark.parametrize("name", _backends())
def test_the_short_budget_override_is_optional(name: str) -> None:
    """`bird.py::train` calls the backend WITHOUT a budget override, so the parameter
    must default to reading the config. A required parameter would break every
    non-cascade config -- i.e. almost all of them."""
    fn = registry.get("train_backend", name)
    params = inspect.signature(fn).parameters
    for p in _BUDGET_PARAMS:
        if p in params:
            assert params[p].default is not inspect.Parameter.empty, (
                f"train_backend {name!r} takes {p!r} but it is REQUIRED; the normal "
                f"train stage passes only (ctx, state, candidate, n_seeds=...)"
            )
            break


def test_the_screen_reports_whether_it_got_what_it_asked_for() -> None:
    """`short_budget_honoured` must keep reaching the artifact.

    It is what makes this failure findable from a run directory rather than by
    reading source, so it is load-bearing evidence and not a debug line to tidy away.
    """
    from bird.components import screens

    src = inspect.getsource(screens)
    assert "short_budget_honoured" in src, (
        "screens.py no longer records short_budget_honoured; a future regression of "
        "this bug would then be invisible in the journal"
    )


# ==========================================================================
# ...and the budget it asks for has to MEAN something at the tier's scale.
# ==========================================================================

def test_the_cascade_budget_is_a_real_fraction_of_the_training_it_screens() -> None:
    """`verify.cascade.short_budget_steps` is a RATIO WEARING AN ABSOLUTE.

    LIMEN's published pairs are 500K/1M, 1M/2M and 3M/5M on the three XLand
    tasks where the cascade runs at all (`configs/methods/limen.yaml`, the comments on
    `verify.cascade.short_budget_steps` and `train.env_steps`) -- 50-60% of the full
    budget every time. The `dev` execution profile runs 20,000
    steps and a config pinning the ABSOLUTE 10,000 rides along, so the ratio
    silently falls to 1% under the `full` profile's 1,000,000 -- a candidate
    rejected for not reaching
    1% success in 1% of a training. At a smoke budget that throws out 15 of 24
    candidates (measured), so the population reaching `train` is not the one LIMEN
    selects from and its CHEAP PRE-FILTER becomes a different search.

    The floor is deliberately loose (5%) rather than the paper's 50%: the tester
    and dev profiles are smoke budgets where the screen only has to be
    exercised, and pinning the published ratio there would forbid a 2-second
    test. What it forbids is the ORDER-OF-MAGNITUDE drift that comes free with
    raising `train.env_steps` and touching nothing else.

    IT MEASURES THE EFFECTIVE BUDGET, NOT THE PINNED KEY. With
    `verify.cascade.short_budget_fraction`, a config can encode the pair either
    way, and reading `short_budget_steps` alone would make this test blind to
    exactly the configs that use the fraction: a config that inherits the paper point's
    `short_budget_steps: 3000000` and overrides the fraction to 0.5 reads 3M
    against the `dev` profile's 20,000 steps -- passing this floor by a factor
    of 150 while the screen actually runs 10,000. Ask
    `screens._cascade_budget` what will really be spent.
    """
    from pathlib import Path

    from bird import config as cfgmod
    from bird.config import CONFIG_ROOT
    from bird.components.screens import _cascade_budget

    class _Ctx:
        def __init__(self, cfg):
            self.cfg = cfg

    checked = []
    for path in sorted(Path(CONFIG_ROOT).rglob("*.yaml")):
        if path.name.startswith("_"):
            continue
        cfg = cfgmod.load(path)
        if cfg.get("verify.quality_screen") != "cascade":
            continue
        full = int(cfg["train.env_steps"])
        short = _cascade_budget(_Ctx(cfg))
        checked.append((path.name, short, full, short / full if full else 0))
        assert short >= 0.05 * full, (
            f"{path}: cascade screens on {short} steps against a {full}-step "
            f"training ({100 * short / full:.1f}%). LIMEN's own pairs are 50-60%; "
            f"below ~5% the screen rejects candidates for not having trained yet, "
            f"which is a different search and not a cheaper one.")
    assert checked, "no cascade config found; this test is vacuous"


# --------------------------------------------------------------------------
# verify.cascade.short_budget_fraction -- the ratio, pinned as a ratio
# --------------------------------------------------------------------------


def test_the_fraction_key_defaults_to_null_so_no_existing_hash_moves() -> None:
    """The key exists to make the ratio expressible, NOT to change any config
    that already runs. `Config.hash()` covers all of `_data` and the run
    directory is `<name>-<hash>-<stamp>/`, so a default that resolved to
    anything but `null` would rename every existing run dir and orphan
    every checkpoint under `loop.resume_from`."""
    from bird.config import load

    assert load("_default")["verify.cascade.short_budget_fraction"] is None


def test_null_fraction_means_the_absolute_is_used_verbatim() -> None:
    from bird.components.screens import _cascade_budget
    from bird.config import load

    cfg = load("limen")

    class _Ctx:
        pass
    ctx = _Ctx()
    ctx.cfg = cfg
    assert cfg["verify.cascade.short_budget_fraction"] is None
    assert _cascade_budget(ctx) == cfg["verify.cascade.short_budget_steps"] == 3_000_000


def test_a_fraction_sizes_the_screen_off_the_real_training_budget() -> None:
    """The whole point: the same config on a bigger budget screens proportionally
    harder in absolute steps and IDENTICALLY as a fraction.

    The budget comes from the execution profile, and no shipped config pins
    the fraction, so the 0.5 is supplied as an override: what is under test is
    `_cascade_budget`.
    """
    from bird.components.screens import _cascade_budget
    from bird.config import load

    class _Ctx:
        def __init__(self, cfg):
            self.cfg = cfg

    for profile, expect in (("dev", 10_000), ("full", 500_000)):
        cfg = load("limen_reward_only", profile=profile,
                   overrides={"verify.cascade.short_budget_fraction": 0.5})
        assert cfg["verify.cascade.short_budget_fraction"] == 0.5
        got = _cascade_budget(_Ctx(cfg))
        assert got == expect, f"{profile}: expected {expect} short steps, got {got}"
        assert got == 0.5 * cfg["train.env_steps"]


def test_the_fraction_survives_the_tier_boundary_where_an_absolute_does_not() -> None:
    """The argument for the key, asserted rather than trusted.

    An execution profile may move `train.env_steps` and may NOT touch
    `verify.cascade.short_budget_steps` (`config.PROFILE_KEY_PREFIXES` stops at
    the method), so a profile that legally raises the budget re-tunes the screen
    it is forbidden from touching. Under the fraction the same legal change
    leaves the ratio alone.
    """
    from bird.components.screens import _cascade_budget
    from bird.config import load

    class _Ctx:
        def __init__(self, cfg):
            self.cfg = cfg

    def ratio(name, profile, **over):
        cfg = load(name, profile=profile, overrides=over)
        return _cascade_budget(_Ctx(cfg)) / int(cfg["train.env_steps"])

    # THE COUNTERFACTUAL, constructed rather than read off a live config: the
    # absolute encoding, the SAME pinned 10,000 across a 50x budget jump the
    # profile is allowed to make -- screen silently 50x harsher, no method file
    # edited.
    absolute = {"verify.cascade.short_budget_fraction": None,
                "verify.cascade.short_budget_steps": 10_000}
    assert ratio("limen", "dev", **absolute) == pytest.approx(0.50)
    assert ratio("limen", "full", **absolute) == pytest.approx(0.01)

    # The fraction: same 50x budget jump, ratio unchanged.
    fraction = {"verify.cascade.short_budget_fraction": 0.5}
    for name in ("limen", "limen_reward_only"):
        for profile in ("tester", "dev", "full"):
            assert ratio(name, profile, **fraction) == pytest.approx(0.50), (
                f"{name} under {profile} no longer screens at LIMEN's "
                f"published ratio")


def test_an_out_of_range_fraction_is_a_config_error_not_a_step_count() -> None:
    """`0.5` and `500000` are both plausible-looking values for a key whose
    sibling IS a step count, so the two are one typo apart. Caught at validate,
    where every problem is reported at once, rather than as a screen that runs
    500,000x its training budget."""
    from bird.config import ConfigError, load

    for bad in (0, -0.5, 1.5, 10_000):
        with pytest.raises(ConfigError):
            load("limen", overrides={"verify.cascade.short_budget_fraction": bad})

    # the boundary is inclusive at 1.0: "screen on the whole budget" is a
    # legitimate (if pointless) request, not a typo.
    load("limen", overrides={"verify.cascade.short_budget_fraction": 1.0})
