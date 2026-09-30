"""`bird.policy_api.import_beside`: a campaign's bare-name imports resolve to the
files beside the caller, or fail loudly -- never to a same-named module another
campaign already put in `sys.modules`."""
from __future__ import annotations

import sys

import pytest

from bird.policies import PolicyError
from bird.policy_api import import_beside


def _campaign(root, name, marker):
    d = root / name
    d.mkdir()
    (d / "lift.py").write_text(f"MARK = {marker!r}\n")
    (d / "attempts.py").write_text("from lift import MARK\nWHO = MARK\n")
    return d


@pytest.fixture(autouse=True)
def _clean_modules():
    for k in ("lift", "attempts"):
        sys.modules.pop(k, None)
    yield
    for k in ("lift", "attempts"):
        sys.modules.pop(k, None)


def test_imports_the_files_beside_the_caller_in_order(tmp_path):
    a = _campaign(tmp_path, "a", "A")
    lift, attempts = import_beside(a, "lift", "attempts")
    assert lift.MARK == "A" and attempts.WHO == "A"
    assert lift.__file__ == str(a / "lift.py")


def test_a_second_campaign_with_the_same_module_name_is_refused_not_shadowed(tmp_path):
    a = _campaign(tmp_path, "a", "A")
    b = _campaign(tmp_path, "b", "B")
    import_beside(a, "lift")
    with pytest.raises(PolicyError, match="already imported in this process"):
        import_beside(b, "lift", "attempts")
    assert sys.modules["lift"].MARK == "A", "the first campaign's module is left alone"


def test_a_foreign_module_under_the_name_is_refused_even_without_a_file(tmp_path):
    a = _campaign(tmp_path, "a", "A")
    sys.modules["lift"] = type(sys)("lift")        # no __file__ at all
    with pytest.raises(PolicyError, match="already imported"):
        import_beside(a, "lift")


def test_the_same_campaign_may_import_again(tmp_path):
    a = _campaign(tmp_path, "a", "A")
    assert import_beside(a, "lift") is import_beside(a, "lift")
