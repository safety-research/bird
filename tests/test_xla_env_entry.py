"""The determinism flag must be set BEFORE the env is constructed.

`bird/xla_env.py` is deliberately torch-free and jax-free: `bird.py`'s env
construction site imports it on every run of every tier, so a `torch` import
behind it would be paid by the numpy tiers too. That is a property worth a
test, because the natural place to have put this function -- beside its only
caller in `bird/components/fasttd3.py` -- imports torch at module scope, and
moving it back would be an easy "tidy-up".
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_xla_env_imports_without_torch_or_jax():
    """A FRESH INTERPRETER, because `sys.modules` in this one is polluted by
    every other test that has run.

    The claim is about what importing `bird.xla_env` PULLS IN, and it cannot
    be made in a process where torch is already loaded.
    """
    src = ("import sys;"
           "import bird.xla_env;"
           "print('torch' in sys.modules, 'jax' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", src], cwd=REPO,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-1500:]
    assert out.stdout.strip() == "False False", (
        f"bird.xla_env pulled in torch or jax: {out.stdout.strip()}. It is "
        f"imported from bird.py's env construction, which every run of every "
        f"tier passes, so what it drags in is paid by the numpy tiers too")


def test_the_flag_is_set_and_reports_in_time_before_jax_is_imported():
    """`in_time` is the honest half: it says whether the flag can still bite.

    A fresh interpreter again, and this time the point is the ORDER -- set
    the flag, then import jax, and the answer must be True. The same call
    after a jax import must raise under `strict=True`, because a run that
    proceeds there produces numbers under a flag that is not in force while
    every seed row says it is.
    """
    src = (
        "import sys\n"
        "from bird.xla_env import set_xla_determinism_flags as f\n"
        "assert f(strict=True) is True, 'jax not imported yet, must be in time'\n"
        "import os; assert '--xla_gpu_autotune_level=0' in os.environ['XLA_FLAGS']\n"
        "sys.modules['jax'] = type(sys)('jax')   # stand in for a real import\n"
        "assert f(strict=False) is False, 'after jax, must report NOT in time'\n"
        "try:\n"
        "    f(strict=True)\n"
        "except RuntimeError as exc:\n"
        "    assert 'after jax was imported' in str(exc)\n"
        "else:\n"
        "    raise AssertionError('strict=True must REFUSE after a jax import')\n"
        "print('ok')\n")
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    out = subprocess.run([sys.executable, "-c", src], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


def test_bird_py_sets_the_flag_before_constructing_a_jax_env():
    """THE CALL SITE, not the function.

    The function existed and was tested before this; what was missing was any
    real caller, so every real seed row would have read
    `autotune_flag_applied_before_jax_import: False` -- the defect recorded
    rather than fixed. This asserts the ordering at the one line every run
    passes: the flag is set, gated on the jax suite, BEFORE
    `registry.get("env", ...)`.

    Source-level, and that is a weaker instrument than running it -- but a
    run needs the jax extra and a GPU, so in CI this is what there is. The
    end-to-end version would read `applied_before_jax_import` off a real seed
    row; this suite runs no such training.
    """
    src = (REPO / "bird.py").read_text()
    i_flag = src.index("prepare_for_env(cfg[\"problem.env_id\"])")
    i_env = src.index('ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)')
    assert i_flag < i_env, (
        "the flag is set AFTER the env is constructed, which is too late: "
        "the adapter's mjx.put_model initialises the XLA backend in its own "
        "__init__")
    # The gate itself now lives in `prepare_for_env`, so this asserts the
    # ORDER here and the gate there -- one claim each, in the place that can
    # actually break it.
    gate = (REPO / "bird" / "xla_env.py").read_text()
    assert 'if env_suite(env_id) != "jax":' in gate, (
        "prepare_for_env no longer gates on the suite: XLA_FLAGS is "
        "process-global and a numpy-tier run has no business changing kernel "
        "selection for the rest of the process")
    body = gate.split("def prepare_for_env", 1)[1]
    # TWO CLAIMS SINCE THE FACTORIES CALL IT TOO. `strict` became a
    # keyword-only parameter so the registered jax factories can call the same
    # gate without raising from inside `__init__` (a late flag there is worth
    # a False on the row, not a dead candidate -- and a raise would error
    # every jax-tier fixture that `importorskip`s jax before constructing).
    # What must not change is that the DEFAULT is strict, which is what the
    # two entry points get, and that the parameter is actually threaded
    # through -- a signature carrying `strict` while the body hard-codes it
    # would pass a substring check and honour nothing.
    assert "strict: bool = True" in body, (
        "prepare_for_env must DEFAULT to strict: a silent non-application is "
        "the defect this whole path exists to remove, and the entry points "
        "pass no strict= of their own")
    assert "set_xla_determinism_flags(strict=strict)" in body, (
        "prepare_for_env declares `strict` but does not pass it on, so the "
        "parameter is a fabricated pin")


def test_the_row_reports_the_PROCESS_verdict_not_the_late_callers():
    """The field answers "was this process's flag set in time", and only the
    FIRST call can answer it.

    A view that reported its own call's verdict would read False on every
    real run, because it runs after the adapter imports jax: on a correctly
    wired process -- `prepare_for_env` called before the env, returning True
    -- the seed row would still read `autotune_flag_applied_before_jax_import:
    false`, regardless of the wiring.

    Three states, and `None` is not `False`: None means nothing ever asked
    (a numpy-tier run), False means someone asked too late. A row that
    cannot tell them apart invites "not applied" to be read as "failed".
    """
    src = (
        "import sys\n"
        "import bird.xla_env as X\n"
        "assert X.applied_in_time() is None, 'nothing has asked yet'\n"
        "assert X.set_xla_determinism_flags(strict=True) is True\n"
        "assert X.applied_in_time() is True\n"
        "sys.modules['jax'] = type(sys)('jax')      # the adapter imports jax\n"
        "assert X.set_xla_determinism_flags(strict=False) is False, 'this call is late'\n"
        "assert X.applied_in_time() is True, (\n"
        "    'a late call must NOT overwrite the process verdict: that is what "
        "made every real seed row read False')\n"
        "print('ok')\n")
    env = dict(os.environ)
    env.pop("XLA_FLAGS", None)
    out = subprocess.run([sys.executable, "-c", src], cwd=REPO, env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")
