"""`load_policy` gives each campaign its OWN bare-named sibling modules, in one process.

Campaign code imports its siblings by bare name (`import hk`, `from rock import Rig`), and Python caches
those by that name for the whole process. Different campaign dirs can ship DIFFERENT contents under the same bare
module name. Loading two
such campaigns in one process -- a BC/DAgger caller's shape -- would hand the second campaign the first one's module,
silently. `bird.policy_api._evict_other_campaigns` is the guard, and these tests fail
without it (checked by deleting its call in `_import_file`: both collision tests fail), and the sys.path half by the
two tests at the end.

The campaigns here are synthetic (a `bird_control`-family closure over `toy_reacher`), so nothing needs a
simulator and CI runs them."""
from __future__ import annotations

import sys

import pytest

import bird.policies as policies_mod
import bird.policy_api as api_mod
from bird.policies import index

_MANIFEST = """campaign: {c}
family: bird_control
pattern: closure
code: [tc.py, helper.py]
source: {{archive: 'none: synthetic test campaign', date: '2026-09-23'}}
policies:
  - id: {c}/p
    task: null
    task_reason: synthetic
    env_id: toy_reacher
    entry: {{file: tc.py, symbol: make_policy}}
    status: reference
    score: {{value: null, reason: synthetic, unit: bird_task_metric,
            verified_through: bird_adapter, date: '2026-09-23'}}
"""


def _campaign(root, name, value):
    d = root / name
    d.mkdir()
    # `helper` is the colliding bare name: every campaign ships one, with different contents.
    (d / "helper.py").write_text(f"VALUE = {value!r}\n")
    (d / "tc.py").write_text("import helper\n\ndef make_policy(p):\n    return lambda s: helper.VALUE\n")
    (d / "policies.yaml").write_text(_MANIFEST.format(c=name))


@pytest.fixture
def two_campaigns(tmp_path, monkeypatch):
    _campaign(tmp_path, "camp_a", "from a")
    _campaign(tmp_path, "camp_b", "from b")
    monkeypatch.setattr(policies_mod, "policies_root", lambda start=None: tmp_path)
    index(tmp_path, refresh=True)
    # Nothing from a previous test may pre-seed the bare name or the per-file cache.
    monkeypatch.delitem(sys.modules, "helper", raising=False)
    monkeypatch.setattr(api_mod, "_MODULE_CACHE", {})
    saved_path = list(sys.path)
    yield tmp_path
    sys.modules.pop("helper", None)
    sys.path[:] = saved_path
    index(refresh=True)


def _helper_of(pid, root):
    rec = policies_mod.get(pid, root)
    make_policy = api_mod._resolve(rec, rec.entry)
    return make_policy.__globals__["helper"]


@pytest.mark.parametrize("order", [("camp_a", "camp_b"), ("camp_b", "camp_a")], ids=["a_then_b", "b_then_a"])
def test_each_campaign_gets_its_own_bare_named_sibling(two_campaigns, order):
    root = two_campaigns
    got = {c: _helper_of(f"{c}/p", root) for c in order}
    for c in order:
        assert got[c].__file__ == str((root / c / "helper.py").resolve()), (
            f"{c} resolved `helper` from {got[c].__file__}")
    assert got["camp_a"].VALUE == "from a" and got["camp_b"].VALUE == "from b"


def test_a_campaign_dir_already_on_sys_path_but_not_first_still_wins(two_campaigns):
    """camp_b's dir in FRONT of camp_a's (an earlier load left both there), then camp_a loaded -- eviction alone
    re-imports `helper` through camp_b's dir. The loader moves the loading campaign's dir to the front."""
    root = two_campaigns
    sys.path.insert(0, str((root / "camp_a").resolve()))
    sys.path.insert(0, str((root / "camp_b").resolve()))
    assert _helper_of("camp_a/p", root).VALUE == "from a"


def test_a_guarded_insert_campaign_can_still_import_a_sibling_lazily(tmp_path, monkeypatch):
    """Several campaigns insert their own dir only `if here not in sys.path` and import siblings inside a function, at
    reset or step. Removing the loader's entry after the import strands them -- such a policy raises
    ModuleNotFoundError at its first step -- so the entry stays, and a lazy sibling import made after load
    resolves."""
    d = tmp_path / "camp_c"
    d.mkdir()
    (d / "lazy_helper.py").write_text("VALUE = 'lazy from c'\n")
    (d / "tc.py").write_text(
        "import sys\nfrom pathlib import Path\nHERE = str(Path(__file__).resolve().parent)\n"
        "if HERE not in sys.path:\n    sys.path.insert(0, HERE)\n\n"
        "def make_policy(p):\n    def act(s):\n        import lazy_helper\n        return [0.0] if lazy_helper.VALUE else [1.0]\n    return act\n")
    (d / "policies.yaml").write_text(_MANIFEST.format(c="camp_c").replace("code: [tc.py, helper.py]", "code: [tc.py, lazy_helper.py]"))
    monkeypatch.setattr(policies_mod, "policies_root", lambda start=None: tmp_path)
    index(tmp_path, refresh=True)
    monkeypatch.delitem(sys.modules, "lazy_helper", raising=False)
    monkeypatch.setattr(api_mod, "_MODULE_CACHE", {})
    saved = list(sys.path)
    try:
        pol = api_mod.load_policy("camp_c/p", root=tmp_path)
        pol.reset()
        assert list(pol.act([0.0], t=0)) == [0.0]
    finally:
        sys.modules.pop("lazy_helper", None)
        sys.path[:] = saved
        index(refresh=True)
