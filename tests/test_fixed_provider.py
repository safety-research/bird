"""`llm.generator.provider: fixed` -- one authored program through stages 2-6 (bird/llm/fixed.py).

A control knob, so a searched number can be read against the same learner on a
known program. The generator returns the file verbatim;
everything downstream is the ordinary path. Offline: mock judge, mock learner.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from bird import registry
from bird.config import ConfigError, load

REPO = Path(__file__).resolve().parents[1]

registry.load_all()

PROGRAM = '''# a hand-written toy_reacher reward, for the fixed provider's tests
import numpy as np


def compute_reward(state, action=None, next_state=None):
    s = np.asarray(next_state if next_state is not None else state, dtype=float)
    dist = float(np.hypot(s[0] - s[4], s[1] - s[5]))
    effort = float(np.dot(np.asarray(action, dtype=float), np.asarray(action, dtype=float))) if action is not None else 0.0
    components = {"reach": -dist, "ctrl": -0.01 * effort}
    return components["reach"] + components["ctrl"], components
'''


def _program(tmp_path: Path) -> Path:
    p = tmp_path / "program.py"
    p.write_text(PROGRAM)
    return p


def _fixed(tmp_path: Path, **overrides):
    base = {"llm.generator.provider": "fixed",
            "llm.generator.program": str(_program(tmp_path)),
            "generate.n_candidates": 1, "seed": 0}
    base.update(overrides)
    return load("eureka", profile="tester", overrides=base)


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry_fixed", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- the three refusals ------------------------------------------------------


def test_k_candidates_are_the_same_program_under_k_learners(tmp_path):
    """`generate.n_candidates` is left to the config: K candidates under `fixed`
    are K identical programs, the same reward under K learner seeds. Two of them are two copies, not a refusal."""
    import random

    from bird.budget import Budget
    from bird.context import Context

    cfg = _fixed(tmp_path, **{"generate.n_candidates": 2})
    assert cfg["generate.n_candidates"] == 2
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(0))
    client = registry.get("llm", "fixed")(ctx, "generator")
    a, b = client("write a reward", n=2, tag="generate")
    assert a == b and "```python\n" + PROGRAM.rstrip() + "\n```" in a


def test_a_program_path_under_another_provider_is_refused(tmp_path):
    with pytest.raises(ConfigError, match=r"program is set but llm.generator.provider='mock'"):
        load("eureka", profile="tester",
             overrides={"llm.generator.program": str(_program(tmp_path)),
                        "generate.n_candidates": 1})


def test_a_fixed_judge_and_a_missing_file_are_refused(tmp_path):
    with pytest.raises(ConfigError, match=r"a judge is not a program"):
        _fixed(tmp_path, **{"llm.evaluator.provider": "fixed"})
    with pytest.raises(ConfigError, match=r"is not a file"):
        _fixed(tmp_path, **{"llm.generator.program": str(tmp_path / "nope.py")})
    with pytest.raises(ConfigError, match=r"needs llm.generator.program"):
        load("eureka", profile="tester",
             overrides={"llm.generator.provider": "fixed", "generate.n_candidates": 1})


# -- the program comes back verbatim, and the run is the ordinary path ----------


def test_the_provider_returns_the_file_verbatim_and_records_its_hash(tmp_path):
    import random

    from bird.budget import Budget
    from bird.context import Context

    cfg = _fixed(tmp_path)
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(0))
    client = registry.get("llm", "fixed")(ctx, "generator")
    assert client.sha256 == hashlib.sha256(PROGRAM.encode()).hexdigest()
    (one,) = client([{"role": "user", "content": "write a reward"}], n=1, tag="generate")
    assert "```python\n" + PROGRAM.rstrip() + "\n```" in one
    three = client("anything at all", n=3)
    assert three == [one, one, one], "the same program on every call, whatever the prompt"
    assert ctx.budget.llm_calls == 2 and ctx.budget.llm_prompt_tokens == 0
    with pytest.raises(Exception, match="judge is not a program"):
        registry.get("llm", "fixed")(ctx, "evaluator")


def test_a_tester_run_trains_the_authored_program_through_every_stage(tmp_path):
    """Eureka on toy_reacher, tester profile, two iterations: every candidate's
    stored reward IS the file, it trains, it is scored by the mock judge and
    selected, and the journal names the program's sha256 at generate time."""
    entry = _entry()
    cfg = _fixed(tmp_path)
    out = tmp_path / "runs"
    entry.run(cfg, out_root=str(out))
    (run,) = [p for p in out.iterdir() if p.is_dir()]
    assert json.loads((run / "status.json").read_text())["status"] == "ok"
    dirs = sorted(p for p in (run / "candidates").iterdir() if p.is_dir())
    assert len(dirs) == 2, "one candidate per iteration, two iterations"
    for d in dirs:
        assert (d / "reward.py").read_text().strip() == PROGRAM.strip()
        meta = json.loads((d / "meta.json").read_text())
        assert not meta.get("failure"), meta.get("failure")
        assert (d / "train_result.json").exists()
    events = [json.loads(x) for x in (run / "journal.jsonl").read_text().splitlines() if x.strip()]
    fixed = [e for e in events if e.get("event") == "fixed_program"]
    assert fixed and fixed[0]["sha256"] == hashlib.sha256(PROGRAM.encode()).hexdigest()
    budget = json.loads((run / "budget.json").read_text())
    assert budget.get("llm_prompt_tokens", 0) == 0 and budget.get("llm_calls", 0) >= 2

