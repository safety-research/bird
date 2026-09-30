"""`evaluate.rejudge_incumbent`: the incumbent is re-scored every round under the
round's own subtasks and repeats, before §5 compares anything against it.

Run under the tester profile (mock LLM/VLM, toy env), where `rda` is the one config
whose fitness IS a VLM score and whose subtask list moves between rounds.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bird.config import ConfigError, load

pytestmark = pytest.mark.usefixtures("tmp_path")


REPO = Path(__file__).resolve().parents[1]


def _entry():
    # `bird.py` is a script beside the package, not `bird/__init__.py`; load it by path
    # exactly as tests/test_parallelism.py does.
    import importlib.util
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _search(name, out, **over):
    over.setdefault("seed", 0)
    over.setdefault("loop.n_iterations", 3)
    _entry().run(load(name, profile="tester", overrides=over), out_root=str(out))
    return next(p for p in Path(out).iterdir() if p.is_dir())


def _events(run, stage):
    return [json.loads(l) for l in (run / "journal.jsonl").read_text().splitlines()
            if l.strip() and json.loads(l).get("stage") == stage]


def test_rejudge_needs_a_vlm_fitness_source():
    with pytest.raises(ConfigError, match="rejudge_incumbent"):
        load("eureka", profile="tester", overrides={"evaluate.rejudge_incumbent": True})


def test_off_by_default_leaves_no_trace(tmp_path):
    run = _search("rda", tmp_path)
    assert _events(run, "rejudge_incumbent") == []


def test_the_incumbent_is_rescored_each_round_after_the_first(tmp_path):
    run = _search("rda", tmp_path, **{"evaluate.rejudge_incumbent": True,
                                       "loop.carry": ["best_reward", "subtask_list", "policy_checkpoint"]})
    ev = _events(run, "rejudge_incumbent")
    done = [e for e in ev if not e.get("skipped")]
    # nothing to re-judge in round 0; every later round re-scores exactly one incumbent
    assert len(done) == 2, ev
    assert [e["iteration"] for e in done] == [1, 2]
    for e in done:
        assert e["cand_id"] and e["n_rollouts"] >= 1 and e["vlm_queries"] >= 1
        assert e["fitness_before"] is not None and e["fitness_after"] is not None
        assert isinstance(e["subtask_scores_after"], dict) and e["subtask_scores_after"]
    # the judge trace carries the re-scored queries under the incumbent's id, in the
    # round that re-judged it -- the same file the round's candidates land in
    trace = (run / "judgments" / "iter01.jsonl").read_text().splitlines()
    ids = {json.loads(l).get("cand_id") for l in trace if l.strip()}
    assert done[0]["cand_id"] in ids


def test_a_rescored_incumbent_is_what_select_compares_against(tmp_path):
    """With the incumbent rule on, the challenger is compared to the RE-JUDGED
    fitness: the `select` event of round k follows a `rejudge_incumbent` event of
    round k, and `update`'s best_fitness equals the re-judged value when no
    challenger beat it."""
    run = _search("rda", tmp_path, **{"evaluate.rejudge_incumbent": True,
                                       "select.require_improvement_over_incumbent": True,
                                       "loop.carry": ["best_reward", "subtask_list", "policy_checkpoint"]})
    lines = [json.loads(l) for l in (run / "journal.jsonl").read_text().splitlines() if l.strip()]
    for it in (1, 2):
        rj = next(e for e in lines if e.get("stage") == "rejudge_incumbent" and e.get("iteration") == it)
        i_rj = lines.index(rj)
        i_sel = next(i for i, e in enumerate(lines) if i > i_rj and e.get("stage") == "select")
        upd = next(e for e in lines[i_sel:] if e.get("stage") == "update")
        assert i_rj < i_sel, "re-judge precedes select"
        assert upd["best_fitness"] >= rj["fitness_after"] - 1e-9, \
            "the incumbent slot never holds less than the re-judged incumbent"
