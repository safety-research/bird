"""Artifact-writing invariants that are not covered by the pipeline tests."""



def test_execute_rate_does_not_count_a_reward_that_crashed():
    """A reward can compile and then raise the first time the learner calls it.

    `valid` means *compiled*; it does not mean *runnable*. Counting only the
    compile reported 1.0 on a real tester-profile eureka run where 2 of 24 rewards died
    with `FloatingPointError: reward returned nan` and
    `TypeError: compute_reward() takes 0 positional arguments`. This is the
    number that answers "is generation working at all", so it is the one place
    an inflated value costs the most.
    """
    from bird.artifacts import execute_rate
    from bird.observability import _execute_rate
    from bird.types import Candidate, CandidateReport, TrainResult

    def report(cand_id, valid, error="", screened=False, trained=True):
        cand = Candidate(cand_id=cand_id, iteration=0, reward_code="",
                         valid=valid, screened_out=screened)
        return CandidateReport(cand_id=cand_id, candidate=cand,
                               result=TrainResult(cand_id=cand_id, candidate=cand,
                                                  error=error, trained=trained))

    reports = [
        report("ok", valid=True),
        report("crashed", valid=True, error="FloatingPointError: reward returned nan"),
        report("uncompilable", valid=False),
        # Screened and skipped candidates are runnable code -- CARD's entire
        # contribution is training runs it did NOT launch, and counting those
        # as execution failures would turn its cost saving into a quality loss.
        report("screened", valid=True, screened=True, trained=False),
    ]
    assert execute_rate(reports) == 0.5
    assert _execute_rate(reports) == execute_rate(reports), (
        "the artifact writer and the tracker define execute_rate separately "
        "on purpose; they must not disagree")


def test_a_comparison_made_without_frames_is_counted():
    """A VLM judge with no video still returns a preference.

    `rda` and `gt` distinguish themselves by looking at rollouts. Run one
    where the renderer is unavailable -- `MUJOCO_GL=disable`, a missing system
    GL library, an env with no `render` -- and the comparator prompt degrades to
    `file=(not rendered)` plus the scalar lines, the judge answers anyway, and
    the run produces a fitness that is indistinguishable in the artifact from
    one produced with eyes. Nothing raises. `blind_comparisons` is the only
    thing standing between that and a published number.
    """
    from bird.budget import Budget

    b = Budget()
    assert b.blind_comparisons == 0
    b.record_blind_comparison()
    b.record_blind_comparison()
    assert b.blind_comparisons == 2
    assert "blind_comparisons" in b.report(), (
        "the counter has to reach the artifact; a caveat that lives only in "
        "memory is the same as no caveat")


def test_a_run_dir_says_whether_the_run_finished(tmp_path):
    """`status.json` is written `running` before anything can fail.

    A SIGSEGV -- which a MuJoCo run can reach through triton, and which no
    `except` in Python can see -- leaves `config.resolved.yaml`, a journal
    that simply stops, `candidates/` and `state/`. Without a marker written up
    front, that directory is structurally indistinguishable from a healthy one
    to any collector not specifically testing for the absence of `result.json`,
    and in a job array whose stdout lands on a compute node the run dir is the
    ONLY surviving evidence. The optimistic-write-then-clear convention is
    the only one a killed process can honour.
    """
    import json

    from bird.artifacts import RunDir

    rd = RunDir(tmp_path, "demo", "abc123")
    status = json.loads((rd.path / "status.json").read_text())
    assert status["status"] == "running", (
        "status.json must exist and read `running` from the moment the "
        "directory does; a marker written at the end cannot survive a crash")
    assert status["pid"] and status["host"], (
        "the marker has to say where the run was, because on a cluster the "
        "scheduler's log is on a filesystem the submitter cannot read")
    rd.close()


def test_a_search_that_raises_records_the_reason_in_the_artifact(tmp_path):
    """The traceback belongs in the run dir, not only on stdout."""
    import json

    from bird.artifacts import RunDir

    rd = RunDir(tmp_path, "demo", "abc123")
    try:
        raise RuntimeError("simulated mid-search failure")
    except RuntimeError as exc:
        rd.finish("failed", exc)
    rd.close()

    status = json.loads((rd.path / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert "simulated mid-search failure" in status["error"]
    assert "Traceback" in status["traceback"]

    events = [json.loads(l) for l in (rd.path / "journal.jsonl").read_text().splitlines()]
    assert any(e.get("stage") == "run_finished" and e.get("status") == "failed"
               for e in events), (
        "a collector that reads only the journal must see the crash too")


def test_finish_ok_clears_the_running_marker(tmp_path):
    import json

    from bird.artifacts import RunDir

    rd = RunDir(tmp_path, "demo", "abc123")
    rd.finish("ok")
    rd.close()
    assert json.loads((rd.path / "status.json").read_text())["status"] == "ok"


def test_the_journal_says_which_stage_a_run_is_inside(tmp_path):
    """Every other journal line is written when something FINISHED.

    `generate` emits its candidates only once the sampler has returned all of
    them; `train` emits one line per candidate as each completes. So a run that
    is twelve minutes into a healthy 16-candidate `generate` has written nothing
    at all, and its directory is byte-for-byte indistinguishable from one that
    died the moment it started.

    A run that wedges is just as ambiguous: one stopped inside `record_rollouts`
    produces nothing, and without an entry line it can be attributed to a stage
    only afterwards, by matching elapsed times against the source. An entry
    line makes the attribution a read rather than a reconstruction.
    """
    import importlib.util
    import json

    from conftest import REPO
    from bird.config import load

    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.run(load("eureka", overrides={"seed": 0, "loop.n_iterations": 2},
                   profile="tester"),
              out_root=str(tmp_path))

    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    events = [json.loads(x) for x in
              (run / "journal.jsonl").read_text().splitlines() if x.strip()]
    begins = [e for e in events if e["stage"] == "stage_begin"]

    # Every stage of the iteration, in the order the loop runs them. `record`
    # is in the list because it is where a run wedges, not because it is a
    # stage in the §1-§6 sense.
    order = ["generate", "verify", "train", "evaluate", "record", "select", "update"]
    assert [e["of"] for e in begins[:len(order)]] == order
    assert len(begins) == 2 * len(order), (
        f"two iterations must announce every stage twice, got {len(begins)}")

    # The iteration number is on the line itself. Without it a reader has to
    # count `update` events to know which iteration a gap belongs to, which is
    # exactly the reconstruction this event exists to remove.
    assert {e["iteration"] for e in begins} == {0, 1}
    assert all(e["restart"] == 0 for e in begins)

    # An entry line always precedes the completions of its own stage: that
    # ordering is the whole claim, and it is what lets a reader say "the last
    # entry line names the stage this run is inside".
    # The FIRST STAGE ENTRY, not journal line zero. A run-level line may legitimately
    # precede the first stage -- `task_spec_resolved` records which task definition the
    # run resolved and is written before anything can fail, for the same reason
    # `status.json` is. Asserting on index 0 encoded the claim as a position, and the
    # claim is an ORDERING: a stage's entry line precedes that stage's completions, which
    # is what lets a reader say "the last entry line names the stage this run is inside".
    first_generate = next(i for i, e in enumerate(events) if e["stage"] == "generate")
    first_begin = next(i for i, e in enumerate(events) if e["stage"] == "stage_begin")
    assert events[first_begin]["of"] == "generate"
    assert first_begin < first_generate
    # ...and nothing before the first entry claims to be a stage event at all.
    assert not [e for e in events[:first_begin] if e.get("of")], (
        f"stage-ish events precede the first stage entry: {events[:first_begin]}")


def test_per_subtask_scores_reach_the_journal_not_only_the_next_prompt(tmp_path):
    """The numbers §4 scores subtasks with must be IN the run directory.

    They already reached the generator via
    `generate.context.include_behavioural_analysis`, and reached
    `phases.run_subtask_reflection`, which revises the subtask the model names
    from the per-subtask analysis. So on a real RDA run these numbers decide
    both what the next prompt says and which subtask gets rewritten -- and
    without this field they appear nowhere in `journal.jsonl`.

    That is worse than a field nobody wrote. The evidence would be in the
    artifact, in the wrong object: recoverable only by reading a LATER
    candidate's `prompt.json`, and only by someone who already knew to look
    forward one iteration. Asked "which subtask was weakest, and why was that
    one revised", the journal could not answer and nothing would indicate it
    could not.

    Pinned on `rda` under the `tester` profile because it is the one config in the suite that
    resolves `evaluate.feedback.granularity: per_subtask` with a mock VLM, so
    the assertion runs offline in the ordinary suite rather than only under
    `--extra all`.
    """
    import importlib.util
    import json

    from conftest import REPO
    from bird.config import load

    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.run(load("rda", profile="tester"), out_root=str(tmp_path))

    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    events = [json.loads(line) for line in
              (run / "journal.jsonl").read_text().splitlines() if line.strip()]
    evaluated = [e for e in events if e.get("stage") == "evaluate"]
    assert evaluated, "no evaluate events at all"

    scored = [e for e in evaluated if e.get("subtask_scores")]
    assert scored, (
        "no `evaluate` event carries `subtask_scores`. rda resolves "
        "`evaluate.feedback.granularity: per_subtask`, so §4 computed them, the "
        "feedback rendered them and the reflection chose a target from them -- "
        "and the run directory kept none of it.")

    subtasks = set(next(e for e in events if e.get("stage") == "decompose")["subtasks"])
    # `generate.co_evolve.subtasks` is on, so the list is REVISED between
    # iterations and iteration 2's scores are keyed by text no `decompose` event
    # ever carried. Each revision is journaled as `subtask_reflection` with its
    # before/after, so the joinable set is the decomposition plus every accepted
    # revision -- if a revision stopped being recorded, that union goes stale and
    # this fails exactly as it would have on a missing `decompose`.
    subtasks |= {e["after"] for e in events
                 if e.get("stage") == "subtask_reflection" and not e.get("declined")}
    for event in scored:
        assert set(event["subtask_scores"]) <= subtasks, (
            "subtask_scores is keyed by something other than the subtask text "
            "the decompose/subtask_reflection events recorded, so the two cannot "
            "be joined -- which is the same unrecoverability with an extra step")
        assert all(isinstance(v, (int, float))
                   for v in event["subtask_scores"].values())


def test_a_config_without_subtasks_does_not_carry_an_empty_scores_field(tmp_path):
    """The field is absent, not `{}`, where the axis is off.

    `eureka` under the tester profile has no decomposition, so an empty dict on
    every evaluate event would be noise on the majority of runs and would make
    "this run scored no subtasks" indistinguishable from "this run has no
    subtasks to score"."""
    import importlib.util
    import json

    from conftest import REPO
    from bird.config import load

    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.run(load("eureka", profile="tester"), out_root=str(tmp_path))
    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    events = [json.loads(line) for line in
              (run / "journal.jsonl").read_text().splitlines() if line.strip()]
    for event in (e for e in events if e.get("stage") == "evaluate"):
        assert "subtask_scores" not in event


def test_eurekas_tips_reach_the_prompt_and_only_where_the_key_says_so(tmp_path):
    """`generate.context.include_reward_engineering_tips` must be VISIBLE in
    `prompt.json`, and invisible where it is off.

    Silent both ways: the tips' absence renders as a perfectly plausible eureka
    prompt (the numbers arrive without Eureka's instructions for reading them),
    and their presence on a method whose paper never sends them would be the
    same defect mirrored. Nothing but the prompt text itself
    can witness either.

    The same run also pins the single-source property: the NUMERIC
    REFLECTION section a child prompt carries must be the PARENT report's §4
    rendition (`report.meta["numeric_reflection"]` == `_section_numeric`'s
    text), not `generation._numeric_reflection`'s reconstruction. Two
    renditions of one curve disagreeing inside one prompt is exactly the wrong
    number that renders as a plausible one.
    """
    import importlib.util
    import json

    from conftest import REPO
    from bird.config import load

    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    analysis_head = "Please carefully analyze the policy feedback"
    writing_tell = "its own temperature variable"

    entry.run(load("eureka", profile="tester"), out_root=str(tmp_path / "eureka"))
    (run,) = [p for p in (tmp_path / "eureka").iterdir() if p.is_dir()]
    cands = sorted((run / "candidates").iterdir())
    later = [d for d in cands if not d.name.startswith("iter00")
             and (d / "prompt.json").exists()]
    assert later, "no iteration>=1 candidate wrote a prompt.json"
    msgs = json.loads((later[0] / "prompt.json").read_text())

    system = next(m["content"] for m in msgs if m["role"] == "system")
    assert writing_tell in system, (
        "eureka.py:57 appends code_output_tip to the system prompt; the key is "
        "on and the system message does not carry the writing tips")
    assert "not with commentary" not in system, (
        "the code-only sentence suppresses the analyze-first behaviour "
        "code_feedback elicits; the tips key must lift it")

    user = next(m["content"] for m in reversed(msgs) if m["role"] == "user")
    marker = "## NUMERIC REFLECTION\n"
    assert marker in user, "an iteration>=1 eureka prompt must reflect on its parent"
    start = user.index(marker) + len(marker)
    end = user.find("\n\n## ", start)
    section = user[start:end if end != -1 else len(user)]
    assert analysis_head in section, (
        "eureka.py:268-269 appends code_feedback after every successful round's "
        "statistics; the analysis tips are missing from NUMERIC REFLECTION")
    assert writing_tell in user.split("## OUTPUT CONTRACT")[-1], (
        "eureka.py:276 appends code_output_tip to every feedback message; our "
        "structural equivalent is the output contract's last block")

    # One rendition. The section body (minus the appended tips) is
    # §4's text -- it starts with _section_numeric's banner, never the
    # fallback's ("Component values at training checkpoints"), and appears
    # verbatim on a previous-iteration report, both in its feedback and in
    # meta["numeric_reflection"], the key generation reads.
    body = section.split("\n\n" + analysis_head)[0].strip()
    assert body.startswith("Training statistics, sampled at checkpoints:"), (
        "the NUMERIC REFLECTION body is not §4's rendition -- the "
        "head-truncating generation-side fallback fired, so the prompt carries "
        "two disagreeing renditions of the parent's curves")
    parent_prefix = f"iter{int(later[0].name[4:6]) - 1:02d}"
    parent_reports = [json.loads((d / "report.json").read_text())
                      for d in cands
                      if d.name.startswith(parent_prefix) and (d / "report.json").exists()]
    assert any(body == (r.get("meta") or {}).get("numeric_reflection")
               and body in (r.get("feedback") or "")
               for r in parent_reports), (
        "the prompt's NUMERIC REFLECTION body matches no previous-iteration "
        "report's meta['numeric_reflection'] -- the single source of truth is "
        "not single")

    # The negative control: a config that leaves the key at its default must
    # show none of it -- and keeps the code-only system sentence.
    entry.run(load("zeroshot", profile="tester"), out_root=str(tmp_path / "zeroshot"))
    (zrun,) = [p for p in (tmp_path / "zeroshot").iterdir() if p.is_dir()]
    for d in sorted((zrun / "candidates").iterdir()):
        if not (d / "prompt.json").exists():
            continue
        for m in json.loads((d / "prompt.json").read_text()):
            assert analysis_head not in m["content"]
            assert writing_tell not in m["content"]
            if m["role"] == "system":
                assert "not with commentary" in m["content"]


def test_the_scalar_gate_hides_the_bradley_terry_strength_and_tally_from_the_prompt(tmp_path):
    """`evaluate.feedback.state_selection_scalar: false` must remove the
    `Preference evidence: Bradley-Terry strength, N wins, N losses` block from
    every prompt, not only the `fitness=` header and `best so far` line
    (the config half is `configs/methods/gt.yaml` resolving the key false).

    Silent: for `evaluate.fitness.source: preference_bt` the Bradley-Terry
    strength IS the selection scalar (`_resolve_bt` copies it into
    `report.fitness`), so a fitness-blind method reading its own argmax
    quantity every round renders as a perfectly plausible GT prompt -- GT's
    App. B (neurips_2025.tex:618-662) shows the LLM no strength, rank or tally.
    Without the block gated, every iteration-1 gt tester prompt carries
    `Preference evidence: Bradley-Terry strength 0.534, 4 wins, 0 losses.`
    twice (once in the carried user turn, once in BEHAVIOURAL ANALYSIS) while
    carrying zero `fitness` tokens, so this block is the whole leak. The run
    also pins the header and `best so far` absences at the prompt level.

    The artifact must keep what the prompt loses: `report.json` meta still
    carries the strength and tally (or the agreement-rate figure becomes
    uncomputable), and `feedback_channel` honestly drops `+preference`.

    The control run pins that the gate is the KEY, not the method: a
    preference-scored config that pins the key true keeps the block.
    """
    import importlib.util
    import json
    import re

    from conftest import REPO
    from bird.config import load

    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)

    def later_prompts(root):
        (run,) = [p for p in root.iterdir() if p.is_dir()]
        later = [d for d in sorted((run / "candidates").iterdir())
                 if not d.name.startswith("iter00") and (d / "prompt.json").exists()]
        assert later, "no iteration>=1 candidate wrote a prompt.json"
        return run, [json.loads((d / "prompt.json").read_text()) for d in later]

    # Run 1: the shipped pin (the key resolves false on gt).
    cfg = load("gt", profile="tester")
    assert cfg.get("evaluate.feedback.state_selection_scalar") is False
    assert cfg.get("evaluate.fitness.source") == "preference_bt"
    entry.run(cfg, out_root=str(tmp_path / "pinned"))
    run, prompts = later_prompts(tmp_path / "pinned")
    text = "\n".join(m["content"] for msgs in prompts for m in msgs)
    assert "Bradley-Terry" not in text, (
        "the Bradley-Terry strength reached a fitness-blind method's prompt")
    assert "Preference evidence" not in text
    assert re.search(r"\b\d+ (wins|losses|comparisons)\b|strength \d", text) is None, (
        "the win/loss tally the strength was fitted from reached the prompt")
    # The header half of the gate, pinned at the prompt.
    assert "fitness=" not in text
    assert "fitness (" not in text
    assert "best so far" not in text

    d = json.loads((run / "candidates" / "iter00_c0000" / "report.json").read_text())
    assert "preference" not in d["feedback_channel"].split("+"), (
        "feedback_channel names a section the prose no longer carries")
    for key in ("bt_strength", "pref_wins", "pref_losses"):
        assert key in d["meta"], (
            f"meta[{key!r}] left report.json -- the gate is prose only; the "
            "artifact must keep what the prompt loses")

    # Run 2, the control: the same method with the key true shows the block.
    shown = load("gt", profile="tester",
                 overrides={"evaluate.feedback.state_selection_scalar": True})
    entry.run(shown, out_root=str(tmp_path / "shown"))
    # gt generates in two stages, so the LAST user message is the coder's
    # "implement that specification" turn; the carried feedback sits in an
    # earlier user message. Search the same text the pinned run was cleared on.
    _, prompts = later_prompts(tmp_path / "shown")
    text = "\n".join(m["content"] for msgs in prompts for m in msgs
                     if m["role"] == "user")
    assert "Preference evidence: Bradley-Terry strength" in text, (
        "the gate is the key, not the method: a config that pins true must "
        "keep the block")
    assert "fitness (preference_bt)" in text


def test_section_preference_is_prose_gated_by_the_scalar_key():
    """Unit form of the above: `_section_preference` returns "" under
    `state_selection_scalar: false`, renders under true, and pops nothing
    from `report.meta` either way (the numbers are the artifact's)."""
    from bird.budget import Budget
    from bird.components.evaluation import _section_preference
    from bird.config import load
    from bird.context import Context
    from bird.types import Candidate, CandidateReport, TrainResult

    def report():
        cand = Candidate("c1", 0, "def reward(s, a, s2): return 0.0, {}")
        return CandidateReport("c1", cand, TrainResult("c1", cand), fitness=0.6,
                               fitness_source="preference_bt",
                               meta={"bt_strength": 0.6, "pref_wins": 3,
                                     "pref_losses": 1, "pref_comparisons": 4})

    hidden = Context(cfg=load("gt", profile="tester"), budget=Budget())
    rep = report()
    assert _section_preference(hidden, rep) == "", (
        "the strength/tally block is the selection scalar and its inputs; "
        "state_selection_scalar: false must hide it as it hides the header")
    assert rep.meta["bt_strength"] == 0.6 and rep.meta["pref_wins"] == 3

    shown = Context(cfg=load("gt", profile="tester",
                             overrides={"evaluate.feedback.state_selection_scalar": True}),
                    budget=Budget())
    rep = report()
    block = _section_preference(shown, rep)
    assert "Bradley-Terry strength" in block and "3 wins" in block and "1 losses" in block
    assert rep.meta["bt_strength"] == 0.6


def test_the_similarity_score_reaches_the_journal_not_only_the_prompt(tmp_path):
    """`evaluate.similarity.metric` must be IN the run directory.

    The same omission `test_per_subtask_scores_reach_the_journal...` above pins,
    one field along. The metric is computed per candidate per iteration
    (`bird.py:179`) and reached two places, NEITHER of them the artifact:

      * `summarise_reports` renders it into the prose the generator reads;
      * `selection._objective_value` returns it as a ranking key, so under
        `evaluate.similarity.role: select` -- or any `select.rule` naming
        `similarity` as an objective -- the number picks the winner and leaves
        no record of having done so.

    Without the journal field, this exact config writes six `evaluate` events
    with keys `[cand_id, channel, feedback_chars, fitness, stage, t]`, and
    "similarity" appears nowhere in the run directory but the resolved config
    that asked for it.

    Pinned with `epic` because EPIC is the instrument the reward-design ablation
    study reads -- distance of each candidate against ground truth over
    iterations -- and a primary axis that is computed but not written down
    produces cells nobody can plot.

    POLARITY: `similarity` is HIGHER-IS-BETTER for every metric, `epic`
    included. `similarity_epic` returns `max(0.0, 1.0 - d)` so section 5's
    maximising rules can read it; the raw distance is `1 - similarity` and lives
    at `meta["epic_distance"]`, which this does not journal. Reading it the
    other way round would have a plotting harness invert the axis and name the
    worst candidate the best.
    """
    import importlib.util
    import json

    from conftest import REPO
    from bird.config import load

    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.run(load("eureka", profile="tester",
                   overrides={"evaluate.similarity.metric": "epic",
                              "evaluate.similarity.reference": "gt_reward"}),
              out_root=str(tmp_path))

    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    events = [json.loads(line) for line in
              (run / "journal.jsonl").read_text().splitlines() if line.strip()]
    evaluated = [e for e in events if e.get("stage") == "evaluate"]
    assert evaluated, "no evaluate events at all"

    scored = [e for e in evaluated if "similarity" in e]
    assert scored, (
        "no `evaluate` event carries `similarity`, though "
        "`evaluate.similarity.metric: epic` means one was computed for every "
        "candidate. Event keys seen: "
        f"{sorted({k for e in evaluated for k in e})}")
    assert all(isinstance(e["similarity"], float) for e in scored), (
        "similarity must be journalled as a number, not a rendered string")
    assert all(e.get("similarity_metric") == "epic" for e in scored), (
        "every journalled `similarity` must name the metric that produced it. "
        "Unlike `fitness`, whose quantity has one orientation, this key does "
        "not: `similarity_epic` registers `1 - d` because section 5 maximises, "
        "while the post-hoc pass writes `epic_distance`, i.e. `d`. A plot "
        "pooling the two reads half its points backwards, and nothing in the "
        "artifact says which orientation a row carries. NOTE the field records "
        "the CONFIGURED metric, not which implementation ran -- resolving that "
        "ambiguity needs a field set inside `_pseudometric`. "
        f"Rows seen: {[{k: e.get(k) for k in ('similarity', 'similarity_metric')} for e in scored][:3]}")


def test_similarity_is_absent_rather_than_null_when_no_metric_is_configured(tmp_path):
    """`metric: none` is the default, so the field must cost nothing there.

    Absent rather than `null`: every `evaluate` row in every existing run
    predates this field, so a reader treating "missing" and "null" as different
    states would see two populations where there is one. Same convention
    `subtask_scores` uses.

    `card` under the tester profile, NOT `eureka`: `eureka` resolves
    `pearson_curve`, not `none` -- and so does the PUBLISHED
    `configs/methods/eureka.yaml`. Which is the sharper form of the case this pair of
    tests exists for: every eureka run computes a similarity score against the
    ground-truth reward, and without the journal field it would throw it away.
    The gap is not hypothetical or specific to the ablation study.
    """
    import importlib.util
    import json

    from conftest import REPO
    from bird.config import load

    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    cfg = load("card", profile="tester")
    assert cfg["evaluate.similarity.metric"] == "none", (
        "this test is only meaningful while the config leaves the metric off")
    entry.run(cfg, out_root=str(tmp_path))

    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    events = [json.loads(line) for line in
              (run / "journal.jsonl").read_text().splitlines() if line.strip()]
    evaluated = [e for e in events if e.get("stage") == "evaluate"]
    assert evaluated, "no evaluate events at all"
    assert not any("similarity" in e for e in evaluated), (
        "`similarity` was written on a run that computed none")
    assert not any("similarity_metric" in e for e in evaluated), (
        "`similarity_metric` was written on a run that computed no similarity. "
        "The label must not outlive the value it labels -- a row carrying a "
        "metric name and no number reads as a failed measurement rather than "
        "an absent one")


def test_stage_4s_verdict_reaches_disk_in_the_shape_rdas_paper_prints(tmp_path):
    """`report.json` is the only record of what the evaluator concluded.

    Without it stage 4 would be the one stage with no artifact. `save_candidate`
    keeps what the generator wrote and `save_train_result` what the learner
    produced; the evaluation would exist only in memory and in a journal line
    recording `feedback_chars` -- three thousand characters of VLM analysis
    stored as an integer. It would survive indirectly, inside the NEXT
    iteration's `prompt.json`, so the final iteration's evaluation would be
    unrecoverable by construction.

    The field names are RDA's own (appendix 7.3: `number`, `name`, `behavior`,
    `score`, `analysis`) so the artifact is diffable against the paper rather
    than against a translation of it. `number` follows the SUBTASK ORDER the
    paper means, which is `subtask_scores`' insertion order -- sorting it
    silently renumbers the list `phases.run_subtask_reflection` indexes into.

    `behavior` -- the paper's fifth field -- is emitted when the report carries
    it and stays ABSENT rather than empty when it does not. The judge IS asked
    for App. 7.3's describe-behaviour step, so a present key is a real answer; an absent one means the judge did
    not supply it (an older run, a skipped field), and `"behavior": ""` would
    assert the VLM looked and saw nothing.
    """
    import json

    from bird.artifacts import RunDir
    from bird.types import Candidate, CandidateReport, TrainResult

    cand = Candidate(cand_id="c0007", iteration=3, reward_code="x = 1", valid=True)
    rep = CandidateReport(
        cand_id="c0007", candidate=cand,
        result=TrainResult(cand_id="c0007", candidate=cand, trained=True),
        fitness=0.5, fitness_source="vlm_score", feedback="prose the next prompt saw",
        feedback_channel="visual",
        # Deliberately NOT in alphabetical or score order: the artifact must
        # preserve the subtask order, not impose one.
        subtask_scores={"stand up": 1.0, "move to package": 0.5, "grasp": 0.0},
        subtask_rationales={"stand up": "upright by step 15", "grasp": "never closed"},
        # One subtask with a behavior, two without: present must land, absent
        # must stay a missing key.
        subtask_behaviors={"move to package": "walks upright, then crawls"})

    rd = RunDir(tmp_path, "t", "deadbeef")
    rd.save_report(rep)
    rd.close()
    out = json.loads((rd.path / "candidates" / "iter03_c0007" / "report.json").read_text())

    assert [s["name"] for s in out["subtasks"]] == ["stand up", "move to package", "grasp"]
    assert [s["number"] for s in out["subtasks"]] == [1, 2, 3]
    assert out["subtasks"][0]["analysis"] == "upright by step 15"
    # A subtask with no rationale drops the key rather than carrying "".
    assert "analysis" not in out["subtasks"][1]
    # The describe-behaviour output: present where supplied, absent
    # where not -- never "".
    assert out["subtasks"][1]["behavior"] == "walks upright, then crawls"
    assert "behavior" not in out["subtasks"][0]
    assert "behavior" not in out["subtasks"][2]
    # The prose, not merely its length -- the whole point of the file.
    assert out["feedback"] == "prose the next prompt saw"
    assert out["fitness"] == 0.5 and out["fitness_source"] == "vlm_score"


def test_a_method_with_no_subtasks_still_writes_a_report(tmp_path):
    """Empty and missing are different, here as everywhere else in the tree.

    Every method that is not RDA produces no per-subtask evidence, and a reader
    of the run directory distinguishes "this run predates `save_report`" from
    "this method has nothing per-subtask to say". It can only do that if the second case
    writes a file with an empty `subtasks` rather than no file at all.
    """
    import json

    from bird.artifacts import RunDir
    from bird.types import Candidate, CandidateReport, TrainResult

    cand = Candidate(cand_id="c0000", iteration=0, reward_code="", valid=True)
    rep = CandidateReport(cand_id="c0000", candidate=cand,
                          result=TrainResult(cand_id="c0000", candidate=cand),
                          fitness=None, fitness_source="none")
    rd = RunDir(tmp_path, "t", "deadbeef")
    rd.save_report(rep)
    rd.close()
    out = json.loads((rd.path / "candidates" / "iter00_c0000" / "report.json").read_text())
    assert out["subtasks"] == []
    # CARD computes no ranking scalar at all; null must not become 0.0.
    assert out["fitness"] is None


def test_status_json_records_the_code_version(tmp_path):
    """`status.json` gains `code_commit`/`code_dirty`: it lets a reader tell which
    code produced a run, so a missing or wrong value would silently mislabel
    it; hence a test."""
    import json

    from bird.artifacts import RunDir, code_version

    commit, dirty = code_version()
    assert commit == "unknown" or (len(commit) == 40
                                   and all(c in "0123456789abcdef" for c in commit)), commit
    assert isinstance(dirty, bool)
    # Cached: a second call is the same object, never a second `git` fork.
    assert code_version() is code_version()

    rd = RunDir(tmp_path, "t", "abc123def456")
    st = json.loads((rd.path / "status.json").read_text())
    assert st["code_commit"] == commit and st["code_dirty"] == dirty
    # On EVERY write, terminal ones included -- a run that died still names its code.
    rd._write_status("ok")
    st_ok = json.loads((rd.path / "status.json").read_text())
    assert st_ok["status"] == "ok"
    assert st_ok["code_commit"] == commit and st_ok["code_dirty"] == dirty
