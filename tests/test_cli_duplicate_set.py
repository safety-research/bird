"""`bird.py` refuses a repeated `--set` key however the flag is spelled.

`tests/test_config.py` pins `parse_overrides` directly. This pins the half that
only exists once argparse has run, and it is the half the guard was written
for: an operator does not call `parse_overrides`, they type a command line, and
the question is whether the second spelling of the same key can slip past.

WHY THE SPELLINGS MATTER. A shell `grep` over the command line, run before
launch, that matches a bare `-s` token lets `--set`, `-sK=V`, `--set=K=V` and a
mixed `-s` + `--set` pair all through -- and the mixed pair is the realistic
one, because an operator who has read the warning about `-s` may reach for
`--set` once for readability on the very line the warning was about.

Doing it in `main()` instead makes the spelling irrelevant by construction:
`add_argument("--set", "-s", action="append")` has already collapsed every form
into one list before anything looks for duplicates. That is the property this
file asserts, and it is the reason the guard is not a pre-flight script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Every way argparse accepts the flag, plus the mixed pair. Each entry is the
#: argv for ONE key given twice; all of them must be refused.
_DUPLICATE_SPELLINGS = {
    "short_spaced": ["-s", "loop.n_iterations=1", "-s", "loop.n_iterations=2"],
    "long_spaced": ["--set", "loop.n_iterations=1", "--set", "loop.n_iterations=2"],
    "short_attached": ["-sloop.n_iterations=1", "-sloop.n_iterations=2"],
    "long_equals": ["--set=loop.n_iterations=1", "--set=loop.n_iterations=2"],
    "mixed_short_then_long": ["-s", "loop.n_iterations=1", "--set", "loop.n_iterations=2"],
    "mixed_attached_then_spaced": ["-sloop.n_iterations=1", "--set", "loop.n_iterations=2"],
}


def _bird(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "bird.py", "-c", "eureka", "--profile", "tester",
         *args, "--print-config"],
        cwd=ROOT, capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("spelling", sorted(_DUPLICATE_SPELLINGS),
                         ids=sorted(_DUPLICATE_SPELLINGS))
def test_a_duplicate_key_is_refused_however_the_flag_is_written(spelling):
    proc = _bird(*_DUPLICATE_SPELLINGS[spelling])
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        f"{spelling}: a doubled key exited 0. The second value would have won "
        f"silently, which is what this refuses.\n{combined[-2000:]}")
    assert "loop.n_iterations" in combined and "given twice" in combined, (
        f"{spelling}: refused, but the message does not name the repeated key -- "
        f"so the operator is sent back to a command line they have already "
        f"misread once.\n{combined[-2000:]}")


def test_the_differing_value_message_names_both_values():
    """The reader is sent back to a command line they have already misread once."""
    proc = _bird("-s", "loop.n_iterations=1", "-s", "loop.n_iterations=2")
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "would have got 2, not 1" in combined, combined[-1500:]


def test_distinct_keys_still_resolve():
    """The guard must not be "more than one --set", which would break everything."""
    proc = _bird("-s", "loop.n_iterations=1", "-s", "generate.n_candidates=2")
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-2000:]
    assert "given twice" not in (proc.stdout + proc.stderr)


def test_the_same_value_twice_is_still_refused():
    """A repeat is a composition error even when the two values agree.

    Tempting to allow, and wrong: the two values agreeing today is a property of
    the two layers that happened to be composed, not of the command. The next
    edit to either layer makes them disagree and the guard would already have
    been taught to stay quiet.
    """
    proc = _bird("-s", "loop.n_iterations=1", "-s", "loop.n_iterations=1")
    combined = proc.stdout + proc.stderr
    assert proc.returncode != 0, combined[-2000:]
    # ...and it must not justify itself with a sentence that is false here. The
    # differing-value message says "the second would have won, silently", which on
    # equal values is true and immaterial -- the one input where the stated reason
    # does not apply. The refusal stands; the reason given
    # changes to the one that is actually true.
    assert "same value" in combined, (
        "the equal-value case is refused with the differing-value justification, "
        "which is the only input where that sentence is not the reason")
    assert "would have won" not in combined, combined[-1500:]
