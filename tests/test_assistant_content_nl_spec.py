"""`update.prompt.assistant_content: nl_spec`: GT's carried transcript
accumulates each round's ENGLISH design, and the latest program appears once.

App. B of the Gran Turismo paper (neurips_2025.tex:627-634) builds the
subsequent-round prompt from `{all_english_rewards}` -- "we have designed
{num_iters} round(s) of reward function components based on the following
instructions" -- and then "the most recent" `{reward_code}`, once. Carrying
only the winner's `reward_code` or its `raw_response` (`_turns`,
bird/components/update.py) cannot express that: under two-stage generation the
raw response IS the coder's code reply, so after K rounds a GT prompt would hold
K full programs and zero English designs, while the parent program appears twice
(the history turn and PARENT REWARD). The stage-1 English is on the record --
`Candidate.nl_spec`, written by `_two_stage_sample`, saved to meta.json.

`nl_spec` carries that field. When it is empty (a candidate whose spec was
never written, a resumed or legacy record) the turn falls back to the code and
the fallback is journalled as `assistant_content_fallback`; every turn is
stamped `content_kind` so `records/iterNN.json` says what it actually holds.
"""

from __future__ import annotations

import importlib.util
import json
import random

import pytest

from conftest import REPO
from bird import registry
from bird.budget import Budget
from bird.components import generation
from bird.config import load
from bird.context import Context
from bird.llm.base import messages_text
from bird.state import RunState
from bird.types import Candidate, CandidateReport, Selection, TrainResult

#: A program with a marker that occurs ONCE in it and nowhere in the prose spec
#: or the prompt boilerplate, so `text.count(MARKER)` is the number of copies of
#: THIS program in a prompt.
MARKER = "bonus_marker_7731"
CODE = (
    "def compute_reward(state, action=None, *_extra, **_kw):\n"
    f"    bonus = 0.25  # {MARKER}\n"
    "    return -abs(float(state[0])) + bonus\n"
)
assert CODE.count(MARKER) == 1
SPEC = (
    "Reward specification. One dense term in the distance to the goal, dominant; "
    "one small penalty on control effort, about a tenth of it; rules out the "
    "policy that parks far away and does nothing."
)
FEEDBACK = "Training statistics: the return plateaued early. feedback_marker_4412"


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ctx(**overrides):
    registry.load_all()
    cfg = load("gt", profile="tester", overrides=overrides)
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(cfg["seed"]))
    ctx.generator = registry.get("llm", "mock")(ctx, role="generator")
    ctx.evaluator = registry.get("llm", "mock")(ctx, role="evaluator")
    ctx.env = registry.get("env", cfg["problem.env_id"])(ctx)
    ctx.human = registry.get("phase", "human_oracle")(ctx)
    return ctx


def _report(nl_spec: str = SPEC, code: str = CODE, feedback: str = FEEDBACK,
            raw_response: str | None = None) -> CandidateReport:
    cand = Candidate(cand_id="c0001", iteration=0, reward_code=code, nl_spec=nl_spec,
                     raw_response=("```python\n" + code + "\n```") if raw_response is None
                     else raw_response)
    res = TrainResult(cand_id="c0001", candidate=cand)
    return CandidateReport(cand_id="c0001", candidate=cand, result=res,
                           fitness=None, feedback=feedback)


def _append(ctx, report) -> RunState:
    return registry.get("prompt_mode", "append")(ctx, RunState(), Selection(winners=[report]))


class _Recorder:
    """Stands in for `RunDir`: `Context.event` is a no-op without one."""

    def __init__(self):
        self.events = []

    def event(self, stage, **fields):
        self.events.append((stage, fields))


def test_under_nl_spec_the_carried_assistant_turn_is_the_english_design():
    """gt pins `nl_spec`; without that pin the carried turn would be the
    program."""
    ctx = _ctx()
    assert ctx.cfg["update.prompt.assistant_content"] == "nl_spec", "precondition: gt pins nl_spec"
    assert ctx.cfg["generate.output.two_stage_nl_then_code"] is True, "precondition: gt is two-stage"

    state = _append(ctx, _report())

    turn = state.dialogue[0]
    assert turn["role"] == "assistant"
    assert turn["content"] == SPEC
    assert MARKER not in turn["content"], "the carried turn is the program, not the English"
    assert turn["content_kind"] == "nl_spec"
    assert state.dialogue[1]["role"] == "user" and FEEDBACK in state.dialogue[1]["content"]


@pytest.mark.parametrize("call_index", [generation._CALL_THINKER, generation._CALL_CODER])
def test_the_next_prompt_shows_the_english_in_history_and_the_latest_code_once(call_index):
    """App. B's shape, as an assertion: `{all_english_rewards}` in the history,
    `{reward_code}` once. Carrying the program instead makes the marker appear
    TWICE (history turn + PARENT REWARD)."""
    ctx = _ctx()
    report = _report()
    state = _append(ctx, report)

    msgs = generation._build_messages(ctx, state, [report], tail=generation._SPEC_TAIL,
                                      call_index=call_index)
    hist = [m for m in msgs if m["role"] == "assistant"]
    assert len(hist) == 1 and hist[0]["content"] == SPEC
    text = messages_text(msgs)
    assert text.count(MARKER) == 1, (
        f"the parent program appears {text.count(MARKER)} times; App. B shows the "
        "most recent code once (PARENT REWARD), never also as a history turn")
    assert SPEC in text


def test_nl_spec_falls_back_to_the_code_and_journals_it():
    """A candidate with no spec (never written, or a legacy record) must still
    contribute a non-blank turn, and the record must say the turn is code."""
    ctx = _ctx()
    rec = ctx.rundir = _Recorder()

    state = _append(ctx, _report(nl_spec=""))
    assert state.dialogue[0]["content"] == CODE
    assert state.dialogue[0]["content_kind"] == "reward_code"
    assert rec.events == [("assistant_content_fallback",
                           {"cand_id": "c0001", "iteration": 0,
                            "wanted": "nl_spec", "carried": "reward_code"})]

    rec.events.clear()
    state = _append(ctx, _report(nl_spec="", code="", raw_response="raw reply 9913"))
    assert state.dialogue[0]["content"] == "raw reply 9913"
    assert state.dialogue[0]["content_kind"] == "raw_response"
    assert [e[1]["carried"] for e in rec.events] == ["raw_response"]


def test_a_spec_that_is_only_whitespace_counts_as_missing():
    ctx = _ctx()
    rec = ctx.rundir = _Recorder()
    state = _append(ctx, _report(nl_spec="  \n"))
    assert state.dialogue[0]["content"] == CODE
    assert state.dialogue[0]["content_kind"] == "reward_code"
    assert len(rec.events) == 1


@pytest.mark.parametrize("value, expected, kind", [
    ("reward_code", CODE, "reward_code"),
    ("raw_response", "```python\n" + CODE + "\n```", "raw_response"),
])
def test_reward_code_and_raw_response_are_unchanged(value, expected, kind):
    """Control: the two published pins (eureka, card) keep their meaning under
    the three-way selector, and no fallback is journalled when the slot is
    filled."""
    ctx = _ctx(**{"update.prompt.assistant_content": value})
    rec = ctx.rundir = _Recorder()
    state = _append(ctx, _report())
    assert state.dialogue[0]["content"] == expected
    assert state.dialogue[0]["content_kind"] == kind
    assert rec.events == []


def test_the_bookkeeping_key_never_reaches_the_wire():
    ctx = _ctx()
    state = _append(ctx, _report())
    cleaned = generation._clean_turns(state.dialogue)
    assert all(set(t) == {"role", "content"} for t in cleaned)
    assert cleaned[0]["content"] == SPEC


def test_a_two_iteration_gt_tester_run_carries_prose_not_programs(tmp_path):
    """End to end on the mock: every history assistant turn in an iteration-1
    prompt is prose, the iteration-0 winner's program appears exactly once
    (PARENT REWARD), and nothing fell back. Carrying the program would make the
    history assistant turn the full parent program, so it would appear twice."""
    cfg = load("gt", profile="tester", overrides={"loop.n_iterations": 2})
    _entry().run(cfg, out_root=str(tmp_path))
    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]

    journal = [json.loads(l) for l in (run / "journal.jsonl").read_text().splitlines() if l.strip()]
    winners = [e["winners"] for e in journal if e.get("stage") == "select"]
    assert winners, "no select event"
    (winner,) = winners[0]
    winner_code = (run / "candidates" / f"iter00_{winner}" / "reward.py").read_text().strip()
    assert winner_code
    winner_spec = json.loads(
        (run / "candidates" / f"iter00_{winner}" / "meta.json").read_text())["nl_spec"]
    assert winner_spec.strip() and "```" not in winner_spec

    later = sorted(d for d in (run / "candidates").iterdir()
                   if d.name.startswith("iter01") and (d / "prompt.json").exists())
    assert later, "no iteration-1 candidate wrote a prompt.json"
    for d in later:
        msgs = json.loads((d / "prompt.json").read_text())
        last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        history = [m for m in msgs[:last_user] if m["role"] == "assistant"]
        assert history, f"{d.name}: no carried assistant turn"
        for m in history:
            assert "```" not in m["content"] and "def " not in m["content"], (
                f"{d.name}: a history assistant turn holds a program, not the English design")
        assert history[0]["content"] == winner_spec
        text = "\n".join(m["content"] for m in msgs)
        assert text.count(winner_code) == 1, (
            f"{d.name}: the iter00 winner's program appears {text.count(winner_code)} times")

    assert not [e for e in journal if e.get("stage") == "assistant_content_fallback"], (
        "the spec existed on every winner; nothing should have fallen back")
    records = json.loads((run / "records" / "iter00.json").read_text())
    assert [t.get("content_kind") for t in records["dialogue"] if t["role"] == "assistant"] == ["nl_spec"]
