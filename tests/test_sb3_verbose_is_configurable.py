"""`train.hyperparameters.verbose` must reach SB3 instead of killing the run.

THE HAZARD. A `verbose=0` literal at an SB3 construction site, placed BEFORE
`**sb3_hyper`:

    algo("MlpPolicy", _Spaces(), seed=seed, verbose=0, **sb3_hyper)

means `-s train.hyperparameters.verbose=1` does not turn SB3's progress on -- it
kills the run at construction with `TypeError: got multiple values for keyword
argument 'verbose'`. There are five such sites; the fifth takes `**seed_hyper`, a
copy made per seed, and is the one easiest to miss when editing by eye.

WHY A DEFAULT RATHER THAN A RESERVED ARGUMENT. `_sb3_algo_and_hyper`'s own
`reserved` loop exists for exactly this hazard -- "letting a config also supply
them would raise TypeError for a duplicate argument" -- and drops `policy`, `env`,
`seed` and `_init_setup_model` with a warning so the failure is never a crash.
`verbose` has the same hazard, but SB3's verbosity is genuinely worth setting, so
it is an ordinary default the config can override rather than a reserved name.

WHY IT MATTERS: a long Meta-World run otherwise has no progress telemetry of any
kind -- the log stops at `[2] verify`, nothing writes per-checkpoint state to
disk, and `budget.json` appears only at completion -- and this is the one knob
that turns it on.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from bird.components.training import _sb3_algo_and_hyper

TRAINING = pathlib.Path(__file__).resolve().parents[1] / "bird" / "components" / "training.py"


class _Cfg(dict):
    pass


def _cfg(**hyper):
    return _Cfg({"train.algorithm": "sac", "train.backend": "sb3",
                 "train.hyperparameters": dict(hyper)})


def test_the_default_is_still_zero():
    """Nothing set must behave exactly as the literal did, or every existing run's
    stdout changes for a reason nobody asked for."""
    # `_sb3_algo_and_hyper` resolves the ALGO CLASS out of `stable_baselines3`
    # before it builds the dict, so even a dict-only assertion needs the package.
    # The `tests` CI job installs `--extra test` and not sb3; without this the two
    # dict tests fail there with `ModuleNotFoundError` while passing under
    # `--extra all`. The two AST guards below are deliberately NOT skipped -- they
    # are static, they are the cheap regression net, and they must run in the job
    # that has no extras.
    pytest.importorskip("stable_baselines3")
    _algo, hyper, _anchor = _sb3_algo_and_hyper(_cfg())
    assert hyper["verbose"] == 0


def test_a_configured_verbose_reaches_sb3():
    """The whole point: the value must arrive, not raise."""
    pytest.importorskip("stable_baselines3")  # see the sibling above
    _algo, hyper, _anchor = _sb3_algo_and_hyper(_cfg(verbose=1))
    assert hyper["verbose"] == 1


def test_constructing_with_verbose_one_does_not_raise():
    """The behaviour, through SB3 itself rather than through the dict.

    The dict alone cannot show the hazard: the TypeError happens at the call.
    """
    gym = pytest.importorskip("gymnasium")
    sb3 = pytest.importorskip("stable_baselines3")
    import numpy as np

    class _Spaces(gym.Env):
        metadata: dict = {}

        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(-np.ones(3, np.float32),
                                                    np.ones(3, np.float32))
            self.action_space = gym.spaces.Box(-np.ones(2, np.float32),
                                               np.ones(2, np.float32))

        def reset(self, *, seed=None, options=None):
            return np.zeros(3, np.float32), {}

        def step(self, action):
            return np.zeros(3, np.float32), 0.0, False, False, {}

    _algo, hyper, _anchor = _sb3_algo_and_hyper(_cfg(verbose=1))
    # exactly the shape the five call sites use
    model = sb3.SAC("MlpPolicy", _Spaces(), seed=0, **hyper)
    assert model.verbose == 1

    _algo, hyper0, _anchor = _sb3_algo_and_hyper(_cfg())
    assert sb3.SAC("MlpPolicy", _Spaces(), seed=0, **hyper0).verbose == 0


def _literal_verbose_sites(tree: ast.AST) -> list:
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        has_star = any(isinstance(k.arg, type(None)) for k in node.keywords)
        for kw in node.keywords:
            if kw.arg == "verbose" and has_star:
                out.append(node.lineno)
    return out


def test_no_construction_site_passes_a_verbose_literal():
    """Asserted on CALLS: no call may pass `verbose=` beside a `**kwargs` splat.

    That combination is the defect's exact shape -- it is what makes the keyword
    un-overridable and turns a configured value into a TypeError. A later edit
    that reinstates `verbose=0` next to `**sb3_hyper` for tidiness fails here.
    """
    sites = _literal_verbose_sites(ast.parse(TRAINING.read_text(encoding="utf-8")))
    assert not sites, (
        f"`verbose=` is passed alongside a `**kwargs` splat at lines {sites}; a config "
        f"value for it then raises TypeError instead of overriding. Put the default in "
        f"`_sb3_algo_and_hyper`'s dict instead."
    )


def test_the_guard_above_can_actually_fire():
    """A detector reporting zero is indistinguishable from a broken one."""
    bad = ast.parse('algo("MlpPolicy", env, seed=0, verbose=0, **sb3_hyper)\n')
    good = ast.parse('algo("MlpPolicy", env, seed=0, **sb3_hyper)\n')
    assert _literal_verbose_sites(bad) == [1], "the guard missed the shape it exists to catch"
    assert _literal_verbose_sites(good) == [], "the guard fires on the fixed form"
