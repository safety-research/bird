"""Where `tasks/`, `configs/` and `policies/` are, and what to do when they are not.

WHY THIS MODULE EXISTS. Resolving a data root as
`Path(__file__).resolve().parent.parent` -- the directory above the `bird`
package -- is correct in a checkout, where that is the repo root. In a
NON-EDITABLE install it is `site-packages`, and `pyproject.toml`'s
`[tool.setuptools.packages.find] include = ["bird*"]` ships none of the three
directories, so such a lookup returns **a real path that exists, contains
other projects, and is not ours** -- with no error.

WHY THE THREE ARE NOT SIMPLY PACKAGED. `tasks/` and `policies/` are data
directories resolved from the checkout, not package data. `configs/` is shipped, as is `bird/envs/assets/` (already inside the package and imported at module scope by the env adapters -- without it `registry.load_all()` cannot even build). For the other two the supported
mechanism on a machine with no checkout is the OVERRIDE below.

TWO PREDICATES, ON PURPOSE, and the difference is the one thing to
understand here.

* `repo_root()` asks **"is this a CHECKOUT?"** and requires marker files.
  Its only caller is `wandb.log_code`, which uploads whatever directory it
  is given to an external service, so "a directory with our data in it" is
  not good enough -- a staged data tree can carry `tasks/`, `policies/` and
  even `pyproject.toml`, and uploading it would be this module's own defect
  relocated. It answers None on a staged tree, which is correct.
* `data_dir()` asks **"where is this DIRECTORY?"** and requires only that it
  exists. An installed package needs its task specs and does not need a
  checkout; making this one demand markers would refuse a data tree placed
  beside it and tell the user to set a variable that is already set.

They disagree about a staged data tree deliberately: it is not a checkout
and it does have the data.

THE RULE THIS MODULE ENFORCES: **never return a plausible wrong path.** A
caller gets the real directory, or an exception that names what is missing
and how to supply it. That is the whole difference between a bare
`parent.parent` and this module -- not that the roots resolve more often, but
that when they do not resolve, something says so.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Where to look when there is no checkout. Points at a directory that
#: CONTAINS `tasks/`, `configs/` and/or `policies/` -- not at one of them.
#:
#: COPIES OF THE DIRECTORIES INSIDE SITE-PACKAGES NEVER ANSWER. `data_dir`
#: returns the directory beside the package only when that parent is a
#: CHECKOUT, and site-packages never is, so `beside` is skipped before
#: existence is even tested. Measured on a simulated wheel layout with such
#: copies present and this variable unset: `tasks` None, `policies` None,
#: `configs` resolving to the SHIPPED `bird/_data/configs` rather than to the
#: copy. Without a checkout it is always this variable that answers, and a
#: missing `BIRD_DATA_ROOT` raises `DataRootMissing` rather than falling back
#: to any copy.
DATA_ROOT_ENV = "BIRD_DATA_ROOT"

#: The three directories this module knows how to find.
DATA_DIRS = ("tasks", "configs", "policies")

#: Files that mark a directory as a BIRD checkout rather than any directory
#: that happens to have a `configs/` in it. Checked together, because a bare
#: `configs/` is common and `pyproject.toml` alone is every Python project.
_CHECKOUT_MARKERS = ("pyproject.toml", "bird.py")


#: Which of the three candidates answered. Recorded rather than inferred:
#: the three point at equivalent content on a healthy machine, so a caller
#: that guessed would be right until the day they diverge -- and that day is
#: exactly when someone needs to know. Written onto the seed row as
#: `data_root_source` (`bird/components/fasttd3.py`), beside `learner_device`
#: and for the same reason: a value that resolves fine says nothing about
#: WHICH SOURCE produced it.
SOURCE_CHECKOUT = "checkout"
SOURCE_ENV = "env"
SOURCE_SHIPPED = "shipped"


class DataRootMissing(RuntimeError):
    """A data directory could not be located, and no guess was made.

    Its own class rather than a bare `RuntimeError` so a caller that wants to
    degrade (a provenance field, say) can catch exactly this and a caller
    that must not degrade does not have to.
    """


def _package_parent() -> Path:
    return Path(__file__).resolve().parent.parent


def _is_checkout(path: Path) -> bool:
    return all((path / m).is_file() for m in _CHECKOUT_MARKERS)


def program_path(raw: Any) -> Path:
    """`llm.generator.program` as a Path: absolute as given, else relative to the
    checkout -- ONE resolution, used by `config._check_coherence` at load and by
    `bird.llm.fixed` at construction, so the file the validator checked is the file
    the provider reads."""
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p
    root = repo_root(required=False)
    return (root / p) if root is not None else p


def repo_root(required: bool = True) -> Optional[Path]:
    """The checkout this package lives in, or None / an error if there is none.

    IDENTIFIED BY MARKER FILES, not by existence. Under a wheel
    `parent.parent` exists, so no check based on existence can catch the
    wrong answer. `pyproject.toml` plus
    `bird.py` is the pair -- the first is every Python project, the second is
    this repo's entrypoint, which a wheel never carries.
    """
    candidate = _package_parent()
    if _is_checkout(candidate):
        return candidate
    override = os.environ.get(DATA_ROOT_ENV)
    if override and _is_checkout(Path(override)):
        return Path(override)
    if not required:
        return None
    raise DataRootMissing(
        f"bird is not running from a checkout: {candidate} has no "
        f"{' and no '.join(_CHECKOUT_MARKERS)}. This is what a non-editable "
        f"install looks like. Set {DATA_ROOT_ENV} to a checkout, or install "
        f"with `pip install -e`.")


def data_dir(name: str, required: bool = True) -> Optional[Path]:
    """`tasks`, `configs` or `policies`, wherever it actually is.

    Order: the checkout beside the package; then `$BIRD_DATA_ROOT/<name>`;
    then package data shipped inside `bird/` (which today is `configs` only).
    Each candidate must EXIST to be returned -- an override naming a
    directory that is not there is not silently preferred over one that is.
    """
    path, _ = data_dir_with_source(name, required=required)
    return path


def data_dir_with_source(name: str, required: bool = True):
    """`(path, source)` -- the same lookup, saying which candidate answered.

    ONE resolution order, not two. `data_dir` is a thin wrapper over this,
    so a caller that wants provenance and a caller that does not cannot
    disagree about where the directory is. A second copy of the order that
    only the provenance path ran would be worse than no provenance: it would
    report confidently about a lookup nobody performed.

    `source` is one of `SOURCE_CHECKOUT`, `SOURCE_ENV`, `SOURCE_SHIPPED`, or
    None when nothing resolved and `required` is False.
    """
    if name not in DATA_DIRS:
        raise ValueError(f"unknown data directory {name!r}; expected one of "
                         f"{', '.join(DATA_DIRS)}")
    beside = _package_parent() / name
    if _is_checkout(_package_parent()) and beside.is_dir():
        return beside, SOURCE_CHECKOUT
    override = os.environ.get(DATA_ROOT_ENV)
    if override:
        candidate = Path(override) / name
        if candidate.is_dir():
            return candidate, SOURCE_ENV
    shipped = Path(__file__).resolve().parent / "_data" / name
    if shipped.is_dir():
        return shipped, SOURCE_SHIPPED
    if not required:
        return None, None
    raise DataRootMissing(
        f"cannot find the `{name}/` directory. Looked beside the package "
        f"({beside}), under ${DATA_ROOT_ENV}"
        f"{' (unset)' if not override else f' ({override})'}, and in the "
        f"package's own data ({shipped}). `tasks/` and `policies/` are NOT "
        f"shipped in the wheel, so on a machine with no checkout, set "
        f"{DATA_ROOT_ENV} to a directory containing them.")
