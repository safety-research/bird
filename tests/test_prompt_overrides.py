"""The five prompt-override knobs must be LIVE, not declared.

Why this file exists at all: a nullable prompt key nothing reads is exactly
the declared-but-unread class, and here the failure is silent
twice over -- a prompt-engineering arm whose override key is ignored runs the
BASELINE prompt under a treatment's config hash, and every number it produces
is a plausible one. Nothing but the prompt text itself can witness it.

Identity in the other direction (all five keys null == the prompt with no
override) is pinned by the config-wide tests: the tester determinism suite and
`test_eurekas_tips_reach_the_prompt_*` run the same configs with every knob at
its null default.
"""

import importlib.util
import json

from conftest import REPO

from bird.config import load


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


SYSTEM_OVERRIDE = "You are the override system prompt. Reply with code."
GUIDANCE = "Guidance marker: make the at-goal state the global argmax."
REFLECTION = "Reflection marker: estimate the plateau return before rewriting."


def test_generation_overrides_reach_the_prompt(tmp_path):
    """system_prompt REPLACES (tips and all); guidance and reflection_guidance
    render where their contracts say."""
    cfg = load("eureka", profile="tester", overrides={
        "generate.context.system_prompt": SYSTEM_OVERRIDE,
        "generate.context.guidance": GUIDANCE,
        "generate.context.reflection_guidance": REFLECTION,
    })
    _entry().run(cfg, out_root=str(tmp_path))
    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    later = [d for d in sorted((run / "candidates").iterdir())
             if not d.name.startswith("iter00") and (d / "prompt.json").exists()]
    assert later, "no iteration>=1 candidate wrote a prompt.json"
    msgs = json.loads((later[0] / "prompt.json").read_text())

    system = next(m["content"] for m in msgs if m["role"] == "system")
    assert system == SYSTEM_OVERRIDE, (
        "a non-null generate.context.system_prompt must replace the whole "
        "system message -- nothing appended, tips included")

    user = next(m["content"] for m in reversed(msgs) if m["role"] == "user")
    assert "## GUIDANCE\n" + GUIDANCE in user
    assert REFLECTION in user, (
        "reflection_guidance must ride with the feedback section on an "
        "iteration>=1 prompt")
    assert "Please carefully analyze the policy feedback" not in user, (
        "with tips on, reflection_guidance SUBSTITUTES for Eureka's "
        "code_feedback block; both at once is two competing instructions")
    # iteration-0 prompts have no parent, so the reflection text has no
    # feedback section to ride on and must be absent there.
    first = [d for d in sorted((run / "candidates").iterdir())
             if d.name.startswith("iter00") and (d / "prompt.json").exists()]
    msgs0 = json.loads((first[0] / "prompt.json").read_text())
    user0 = next(m["content"] for m in reversed(msgs0) if m["role"] == "user")
    assert REFLECTION not in user0


JUDGE_GUIDE = "Judge marker: frames are ground truth; component values only corroborate."
DECOMP_GUIDE = "Decomposition marker: early subtasks must separate candidates."


def test_judge_and_decomposition_guidance_reach_their_prompts(tmp_path):
    """rda's two §4/§1 prompt sites carry their overrides, witnessed by the
    judge trace (the artifact whose whole point is prompt parity)."""
    cfg = load("rda", profile="tester", overrides={
        "evaluate.vlm.judge_guidance": JUDGE_GUIDE,
        "generate.decomposition.guidance": DECOMP_GUIDE,
        "output.judge_trace.record": "inputs",
    })
    _entry().run(cfg, out_root=str(tmp_path))
    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]

    traces = sorted((run / "judgments").glob("*.jsonl")) if (run / "judgments").is_dir() else []
    assert traces, "output.judge_trace.record=inputs wrote no judgments/*.jsonl"
    rows = [json.loads(line)
            for p in traces for line in p.read_text().splitlines() if line.strip()]
    judge_prompts = [r.get("prompt", "") for r in rows
                     if r.get("role") == "vlm_subtask_score" and r.get("prompt")]
    assert judge_prompts, "no vlm_subtask_score query reached the judge trace"
    assert all(JUDGE_GUIDE in p for p in judge_prompts)
    assert all(p.index(JUDGE_GUIDE) < p.index("## Output") for p in judge_prompts), (
        "judge_guidance belongs inside the instructions block, before the "
        "output contract")


def test_decomposition_guidance_is_in_the_prompt_text():
    """`_decompose_prompt` is pure; the wiring test above cannot see its text
    (the decompose call goes to the generator, not the judge trace), so the
    contract -- guidance inside the constraints block, absent when unset -- is
    pinned on the function itself."""
    from bird.components.phases import _decompose_prompt

    base = _decompose_prompt("push the puck", None, "class Env: ...", False)
    assert DECOMP_GUIDE not in base
    guided = _decompose_prompt("push the puck", None, "class Env: ...", False,
                               guidance="\n" + DECOMP_GUIDE + "\n")
    assert DECOMP_GUIDE in guided
    assert guided.index(DECOMP_GUIDE) < guided.index("## Task Instruction")
    assert guided.replace("\n" + DECOMP_GUIDE + "\n", "", 1) == base, (
        "guidance must be an insertion, not a rewrite: with it removed the "
        "prompt must be byte-identical to the unguided one")
