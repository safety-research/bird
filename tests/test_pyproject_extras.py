"""`pyproject.toml`'s `all` extra is hand-maintained, and this holds it to what it claims.

A full-deps install runs `--extra all`, and several per-tier extras justify a package by
saying `all` backstops it -- but `all` is a second, separately typed list, so a package
added to a tier extra and not to `all` makes the test it was added for `importorskip` in
the one configuration that was supposed to run it.

Stdlib only: `tomllib` on 3.11+, and `tomli` below it, which pytest itself depends on
there, so the parser is always present where this test runs.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: pytest depends on tomli there
    import tomli as tomllib  # type: ignore[no-redef]

REPO = Path(__file__).resolve().parents[1]

#: The extras whose comments in pyproject.toml say the tier "IS in `all`" / "is covered
#: by `all`" -- the ones a full-deps install is stated to backstop. `jax` is documented
#: there as deliberately NOT in `all`, so it is not listed here; adding a tier extra that
#: `all` should carry means adding it here too.
BACKSTOPPED_BY_ALL = ("metaworld", "assistax")


def _extras() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]


def _name(requirement: str) -> str:
    """`stable-baselines3[extra]>=2.3` -> `stable-baselines3`, normalised as PEP 503 does."""
    m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    assert m, f"unparseable requirement {requirement!r}"
    return re.sub(r"[-_.]+", "-", m.group(1)).lower()


def test_the_all_extra_carries_every_package_of_the_extras_it_backstops():
    extras = _extras()
    in_all = {_name(r) for r in extras["all"]}
    missing = sorted(f"{extra}: {_name(r)}" for extra in BACKSTOPPED_BY_ALL
                     for r in extras[extra] if _name(r) not in in_all)
    assert not missing, (
        f"pyproject.toml's `all` extra omits {missing}. A full-deps install uses "
        "`--extra all`, so a gated test whose package is only in its tier extra skips in "
        "the one install meant to run it. Add the package to `all` and re-lock (uv lock).")


def test_the_backstopped_extras_exist():
    """The list above names extras that exist; a renamed extra must not pass vacuously."""
    extras = _extras()
    for extra in BACKSTOPPED_BY_ALL:
        assert extra in extras, f"pyproject.toml has no `{extra}` extra; update BACKSTOPPED_BY_ALL"
