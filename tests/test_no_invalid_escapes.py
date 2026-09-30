"""Every source file compiles with `DeprecationWarning` treated as an error.

An invalid escape sequence in a docstring (`"\\g<0>"` written as `"\\g<0>"` in a
non-raw string) is a DeprecationWarning at compile time that fires on every
import of the module, so it appears in the output of every test and every
`bird.py` invocation. Python
makes it a SyntaxError in a later release, so the warning is a countdown, not
noise. One test rather than one per file: the count is not the point.
"""

from __future__ import annotations

import warnings

from conftest import REPO


def _sources():
    yield REPO / "bird.py"
    yield from sorted((REPO / "bird").rglob("*.py"))


def test_every_source_file_compiles_without_deprecation_warnings() -> None:
    offenders = []
    for path in _sources():
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warnings.simplefilter("error", SyntaxWarning)
            try:
                compile(path.read_text(), str(path), "exec")
            except (SyntaxError, DeprecationWarning, SyntaxWarning) as exc:
                offenders.append(f"{path.relative_to(REPO)}: {exc}")
    assert not offenders, "\n".join(offenders)
