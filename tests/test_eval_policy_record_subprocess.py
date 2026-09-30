"""The same-day collision, through the CLI rather than through ``main()``.

``tests/test_eval_policy_record.py`` proves the property in-process, which is where
every other check of the record shape belongs.  This file proves it the way the
defect arises in practice: two ``scripts/eval_policy.py`` invocations, two
interpreters, one day.  It is a separate file because ``pyproject.toml``
assigns the ``slow`` marker BY FILE and BY KIND -- "it forks a process" is one
of the kinds it names -- and a per-test mark has only two precedents in this
suite.  Being slow-marked, it is deselected from the default CI selection, so
nothing here may be the only cover for a rule.

Why it is worth running at all when the in-process half exists: `main()` is
called with an argv list and a keyword, and the CLI is called with a shell
line.  The two disagree the day an argument is added to one and not the other,
and the record whose name collides is written by the second.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import run_probe

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parent.parent

_MANIFEST = """campaign: subproccamp
family: bird_control
pattern: closure
code: [controller.py]
source: {archive: 'none: synthetic test campaign', date: '2026-09-06'}
policies:
  - id: subproccamp/candidate
    task: null
    task_reason: a synthetic record fixture, not a catalogue task
    env_id: toy_reacher
    entry: {file: controller.py, symbol: make_policy}
    status: reference
    score:
      value: null
      reason: measured in-test; this manifest records the artifact
      unit: bird_task_metric
      verified_through: bird_adapter
      date: '2026-09-06'
"""

_CONTROLLER = """import numpy as np


def make_policy(p):
    def act(s):
        return np.clip(1.5 * (s[4:6] - s[:2]), -1.0, 1.0)
    return act
"""


def _probe(root: Path) -> str:
    """Two full CLI runs in ONE child, so both land on the same calendar day.

    Splitting them across two children would let a run started at 23:59:59.9
    write into the next day and the check would pass for the wrong reason -- the
    property under test is that two records the SAME DAY are two objects.
    """
    return f'''
import json, pathlib, runpy, sys
root = pathlib.Path({str(root)!r})
camp = root / "subproccamp"
camp.mkdir(parents=True, exist_ok=True)
(camp / "policies.yaml").write_text({_MANIFEST!r})
(camp / "controller.py").write_text({_CONTROLLER!r})
sys.path.insert(0, {str(REPO)!r})
mod = runpy.run_path({str(REPO / "scripts" / "eval_policy.py")!r},
                     run_name="_eval_policy_probe")
argv = ["--policy", "subproccamp/candidate", "--seeds", "0-1", "--root", str(root)]
for _ in range(2):
    assert mod["main"](argv) == 0
names = sorted(p.name for p in (camp / "records").glob("*.json"))
dates = sorted({{json.loads(p.read_text())["date"]
               for p in (camp / "records").glob("*.json")}})
print("PROBE " + json.dumps({{"names": names, "dates": dates}}))
'''


def test_two_cli_runs_the_same_day_write_two_records(tmp_path):
    out = run_probe(_probe(tmp_path))
    assert len(out["dates"]) == 1, out["dates"]
    assert len(out["names"]) == 2, out["names"]
    assert out["names"][0] != out["names"][1]
    # Both names carry the same date and differ after it: the collision v1 had
    # is resolved by the clock and the digest, not by the day.
    day = out["dates"][0]
    assert all(n.startswith(f"eval_candidate_{day}_") for n in out["names"]), out["names"]
