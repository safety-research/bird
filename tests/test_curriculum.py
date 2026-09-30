"""The curriculum axis: an ORDERED progression of skills, gated stage by stage.

An unpublished direction of the design space, and the one that is not a
config edit: every other unexplored direction is reachable by editing keys that
already exist; this one needs `loop.curriculum.*`, a carry slot and three
registry families.

**The claim these tests exist to protect is a DISTINCTION, not a feature.** RDA
already decomposes, and at a glance that looks like this: `configs/methods/rda.yaml`
sets `generate.decomposition.enabled: true`, `subtask_list` is a carry slot,
`_vlm_trajectory_analysis` scores each subtask every iteration. But those are J
CONCURRENT ASPECTS of one task and the policy is always attempting the whole
thing. A curriculum is K SEQUENTIAL STAGES with a gate. If a refactor ever
collapses the two -- every stage in the prompt at once, or a gate that advances
regardless -- nothing else in the suite would notice, because the run would
still complete and every artifact would still say `curriculum: enabled`. That
is what `test_the_prompt_names_one_stage_...` and the gate tests are for.

Six further properties are pinned because each is an observed failure mode:

  1. the derived patience is FROZEN at authoring -- otherwise `on_stall:
     resplit` grows the stage list, which shrinks the derived patience, which
     stalls the new stages sooner: 7 -> 9 -> 13 stages in three iterations with
     nothing passed (observed);
  2. an infeasible curriculum (more stages than iterations) says so, since no
     coherence rule can -- the stage count does not exist until a model has been
     asked;
  3. a stage ADVANCED past is distinguishable in the record from one PASSED,
     because both are `index += 1`;
  4. an unanswered judge is not a failed stage (`None` is not `0.0`);
  5. a blind gate records that it was blind, as the other two judges do;
  6. frames do not ride out on the report into the checkpoint, which bloats
     rather than raises and would therefore be reported by nothing.

And one expressiveness test, which is this repo's actual acceptance criterion:
**Eureka's published pen-spinning curriculum is now a point in the space.**
"""

import importlib.util
import json
import random

import pytest

from conftest import REPO  # noqa: F401  (path setup)
from bird import config as C
from bird import registry
from bird.budget import Budget, BudgetExceeded
from bird.components import curriculum as CUR
from bird.components.generation import _curriculum_section
from bird.context import Context
from bird.state import CARRY_SLOTS, CurriculumState, RunState
from bird.types import Candidate, CandidateReport, Selection, TrainResult, Trajectory

CARRY = ["curriculum", "policy_checkpoint", "subtask_list", "dialogue", "best_reward"]


def _cfg(**overrides):
    base = {
        "loop.curriculum.enabled": True,
        "loop.carry": list(CARRY),
        "train.init": "warm_start_from_best",
        "generate.context.include_curriculum_stage": True,
        "loop.n_iterations": 6,
        # The judge gate, not the shipped default. `fixed_budget` is the
        # default because it is degenerate and cannot stall, but most of what
        # is under test here is what a judge decides -- and a helper that
        # silently left the schedule gate on would make the gate assertions
        # pass for the wrong reason.
        "loop.curriculum.gate": "vlm_ensemble",
    }
    base.update(overrides)
    return C.load("rda", overrides=base, profile="tester")


def _ctx(cfg=None, evaluator=None, generator=None):
    registry.load_all()
    cfg = cfg if cfg is not None else _cfg()
    ctx = Context(cfg=cfg, budget=Budget(), rundir=None, rng=random.Random(0))
    ctx.evaluator = evaluator
    ctx.generator = generator
    return ctx


def _report(cand_id="c0", fitness=1.0, policy_ref=None):
    traj = Trajectory(states=None, actions=None, rewards=[0.5] * 8, success=True,
                      length=8, ret=4.0)
    cand = Candidate(cand_id=cand_id, iteration=0,
                     reward_code="def compute_reward(state, action=None): return 0.0")
    res = TrainResult(cand_id=cand_id, candidate=cand, trajectories=[traj],
                      policy_ref=policy_ref)
    return CandidateReport(cand_id=cand_id, candidate=cand, result=res, fitness=fitness)


class _Judge:
    """A client that answers with a fixed queue of scores, then repeats the last."""

    modality = "text"

    def __init__(self, *scores):
        self.scores = list(scores) or [1.0]
        self.calls = []

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        self.calls.append({"prompt": messages, "tag": tag,
                           "n_images": len(images or ())})
        s = self.scores.pop(0) if len(self.scores) > 1 else self.scores[0]
        return [f"Score: {s}. Reason: because."] * n


class _Silent:
    """A client that answers nothing parseable -- a provider outage, not a fail."""

    modality = "text"

    def __init__(self):
        self.calls = 0

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        self.calls += 1
        return ["the model is unavailable"] * n


def _state(stages, index=0, iteration=0, **kw):
    cur = CurriculumState(stages=list(stages), index=index, **kw)
    st = RunState(iteration=iteration)
    st.curriculum = cur
    return st, cur


# --------------------------------------------------------------------------
# 1 -- a curriculum is not a decomposition
# --------------------------------------------------------------------------

def test_the_prompt_names_one_stage_and_marks_the_others():
    """The whole hypothesis: the generator is asked for ONE stage, not the task.

    A section that listed every stage would be RDA's SUBTASKS block with a new
    heading, and the axis would measure nothing -- so the marker on the current
    stage, and the explicit out-of-scope marker on the later ones, are the
    behaviour, not decoration.
    """
    st, cur = _state(["stand up", "grasp the bar", "lift it"], index=1)
    body = _curriculum_section(_ctx(), st)
    assert "[DONE] stand up" in body
    assert "[WRITE THE REWARD FOR THIS ONE] grasp the bar" in body
    assert "[later, not yet] lift it" in body
    assert body.count("WRITE THE REWARD FOR THIS ONE") == 1, (
        "exactly one stage is the target; more than one is a decomposition")
    assert "must not reward a later stage" in body


def test_the_section_is_empty_without_a_curriculum():
    """`_build_messages` drops empty bodies, so a heading over nothing would be
    a claim that a curriculum exists."""
    assert _curriculum_section(_ctx(), RunState()) == ""


def test_a_regression_warning_reaches_the_generator():
    """A rollback the prompt cannot see is a silent rewrite of what the next
    reward is being written against."""
    st, cur = _state(["a", "b"], index=1)
    cur.note = "REGRESSION: the agent has lost a skill."
    assert "REGRESSION" in _curriculum_section(_ctx(), st)


def test_the_curriculum_is_a_carry_slot_and_is_dropped_when_not_carried():
    """`loop.carry` decides; a curriculum that survived unnamed would make
    `history_mode: none` mean something different here than everywhere else."""
    assert CARRY_SLOTS["curriculum"] == "curriculum"
    st, _cur = _state(["a", "b"])
    st.apply_carry(["curriculum"])
    assert st.curriculum is not None
    st.apply_carry(["best_reward"])
    assert st.curriculum is None


# --------------------------------------------------------------------------
# 2 -- the gate
# --------------------------------------------------------------------------

def test_a_passed_stage_advances_and_records_the_policy_it_passed_with():
    ctx = _ctx(evaluator=_Judge(1.0))
    st, cur = _state(["a", "b"], iteration=0)
    st.policy_ref = "policy:c0"
    winner = _report(policy_ref="policy:c0")
    CUR.run_curriculum_step(ctx, st, Selection(winners=[winner], losers=[]))
    assert cur.index == 1
    assert cur.checkpoints["0"] == "policy:c0", (
        "the handover must record WHICH policy passed, or a rollback has "
        "nothing to restore")
    assert cur.pass_scores["0"] == pytest.approx(1.0)
    assert [h["event"] for h in cur.history] == ["passed"]


def test_the_gate_judges_and_carries_the_same_policy_under_elitism():
    """The gate scores the ROUND WINNER's rollout; the checkpoint, the
    `passed` event and the next stage's warm start must name that same
    policy -- not `state.policy_ref`, which topology's `_carry_inner_loop_refs`
    sets from `state.best`: the incumbent, which under the default
    `update.elitism.keep_global_best: true` moves only on whole-task fitness.
    Naming it would record, carry and later restore a winner that passed the
    stage without beating the incumbent as a policy the gate never saw.
    Driven through the real topology, as `bird.update()`
    orders it: incumbent A (0.21) at iteration 0, round winner B (0.18)
    passes the gate at iteration 1."""
    from bird.components import update as U

    judge = _Judge(0.2, 0.9)  # A fails the gate, B passes it
    cfg = _cfg(**{"loop.curriculum.gate_votes": 1})
    ctx = _ctx(cfg, evaluator=judge)
    assert cfg["update.elitism.keep_global_best"] is True
    st, cur = _state(["stand up", "lift"], iteration=0)

    a = _report("A", fitness=0.21, policy_ref="policy:A")
    sel = Selection(winners=[a], losers=[])
    U.topo_single_parent_hillclimb(ctx, st, sel)
    CUR.run_curriculum_step(ctx, st, sel)
    assert cur.index == 0 and st.policy_ref == "policy:A"

    st.iteration = 1
    b = _report("B", fitness=0.18, policy_ref="policy:B")
    sel = Selection(winners=[b], losers=[])
    U.topo_single_parent_hillclimb(ctx, st, sel)
    assert st.best.cand_id == "A" and st.policy_ref == "policy:A", (
        "elitism keeps the incumbent; this is the divergence under test")
    CUR.run_curriculum_step(ctx, st, sel)
    assert cur.index == 1
    passed = [h for h in cur.history if h["event"] == "passed"][0]
    assert cur.checkpoints["0"] == "policy:B", (
        "the checkpoint must be the policy the gate judged, not the incumbent's")
    assert passed["policy_ref"] == "policy:B"
    assert st.policy_ref == "policy:B", (
        "stage 2 must warm-start from the policy that passed stage 1")
    assert st.best.cand_id == "A", "fitness elitism is a different question and is untouched"


def test_a_winner_without_a_policy_ref_records_none_rather_than_the_incumbents():
    ctx = _ctx(evaluator=_Judge(1.0))
    st, cur = _state(["a", "b"], iteration=0)
    st.policy_ref = "policy:incumbent"
    CUR.run_curriculum_step(ctx, st, Selection(winners=[_report(policy_ref=None)], losers=[]))
    assert cur.index == 1
    assert cur.checkpoints["0"] == "", "no ref is not somebody else's ref"
    assert st.policy_ref == "policy:incumbent", "nothing better to carry; leave it"


def test_the_gate_is_an_ensemble_of_gate_votes_calls():
    """One VLM score is not evidence for a handover that is irreversible for the
    rest of the search."""
    judge = _Judge(1.0)
    ctx = _ctx(_cfg(**{"loop.curriculum.gate_votes": 5}), evaluator=judge)
    st, cur = _state(["a", "b"])
    CUR.gate_vlm_ensemble(ctx, st, _report(), "a", 0)
    assert len(judge.calls) == 5


def test_a_score_below_the_threshold_holds_the_stage():
    ctx = _ctx(_cfg(**{"loop.curriculum.gate_threshold": 0.9}), evaluator=_Judge(0.5))
    st, cur = _state(["a", "b"])
    CUR.run_curriculum_step(ctx, st, Selection(winners=[_report()], losers=[]))
    assert cur.index == 0 and not cur.checkpoints


def test_an_unanswered_judge_is_not_a_failed_stage():
    """`None` and `0.0` must stay apart. A gate that read a provider outage as
    "the agent cannot do it" would hold a curriculum at stage 0 for a whole run
    and report it as the agent's failure -- which needs the opposite response."""
    ctx = _ctx(evaluator=_Silent())
    st, cur = _state(["a", "b"])
    out = CUR.gate_vlm_ensemble(ctx, st, _report(), "a", 0)
    assert out["score"] is None and out["passed"] is False
    assert out["n_answered"] == 0, (
        "the record has to say nobody answered, or a stuck provider and a stuck "
        "policy are the same artifact")


def test_a_blind_gate_says_it_was_blind():
    """As in the other judges: a judgment made with no pixels must not be
    indistinguishable from one made with them."""
    ctx = _ctx(evaluator=_Judge(1.0))  # modality: text -> never sighted
    st, _cur = _state(["a", "b"])
    out = CUR.gate_vlm_ensemble(ctx, st, _report(), "a", 0)
    assert out["n_images"] == 0 and out["blind_reason"]


def test_fixed_budget_asks_the_judge_nothing_and_advances_on_the_budget():
    judge = _Judge(0.0)
    ctx = _ctx(_cfg(**{"loop.curriculum.gate": "fixed_budget",
                       "loop.curriculum.stage_patience": 2}), evaluator=judge)
    st, cur = _state(["a", "b"], iteration=0)
    CUR.run_curriculum_step(ctx, st, Selection(winners=[_report()], losers=[]))
    assert cur.index == 0, "one iteration of a two-iteration budget"
    st.iteration = 1
    CUR.run_curriculum_step(ctx, st, Selection(winners=[_report()], losers=[]))
    assert cur.index == 1
    assert judge.calls == [], "this gate is a schedule; asking would cost calls"


def test_fixed_budget_records_no_score_rather_than_zero():
    ctx = _ctx(_cfg(**{"loop.curriculum.gate": "fixed_budget"}))
    st, _cur = _state(["a", "b"])
    assert CUR.gate_fixed_budget(ctx, st, _report(), "a", 0)["score"] is None


# --------------------------------------------------------------------------
# 3 -- what happens when a stage never passes
# --------------------------------------------------------------------------

def test_a_stalled_stage_is_split_in_place():
    """The replacements go in AT THE SAME INDEX so the ordering claim either
    side of them still holds, and the first of them gets a fresh patience."""
    ctx = _ctx(_cfg(**{"loop.curriculum.stage_patience": 1}),
               evaluator=_Judge(0.0),
               generator=_ListClient(["get both feet under the hips",
                                      "rise until the torso is upright"]))
    st, cur = _state(["stand up", "lift the bar"], iteration=0)
    CUR.run_curriculum_step(ctx, st, Selection(winners=[_report()], losers=[]))
    assert cur.stages == ["get both feet under the hips",
                          "rise until the torso is upright", "lift the bar"]
    assert cur.index == 0 and cur.splits == 1
    assert cur.entered_at == 1, "the first replacement starts its own patience"
    assert cur.history[-1]["event"] == "stalled" and cur.history[-1]["taken"] == "resplit"


def test_a_stage_that_cannot_be_split_is_advanced_past():
    """The remedy must not reintroduce the disease: a stage that stalls and
    cannot be split is one to move past, not one to sit on for ever.

    An EMPTY stage list is the case this uses, and it is not contrived. It is
    what `_parse_list` returns fence artifacts for -- the JSON path yields
    nothing, the prose fallback then reads the fence itself -- so without
    `_clean_stages` this test splices the literal lines `json` and
    `{"stages": []}` into the curriculum and the judge is asked to score them."""
    ctx = _ctx(_cfg(**{"loop.curriculum.stage_patience": 1}),
               evaluator=_Judge(0.0), generator=_ListClient([]))
    st, cur = _state(["stand up", "lift the bar"], iteration=0)
    CUR.run_curriculum_step(ctx, st, Selection(winners=[_report()], losers=[]))
    assert cur.stages == ["stand up", "lift the bar"] and cur.index == 1
    assert cur.history[-1]["taken"] == "advance"


def test_fence_artifacts_never_become_stages():
    """The guard behind the test above, stated directly so a refactor that drops
    `_clean_stages` fails on the reason rather than on a downstream symptom."""
    assert CUR._clean_stages(["json", '{"stages": []}', "```", "py", "  ",
                              "stand up and stay upright"]) == \
        ["stand up and stay upright"]


def test_an_advanced_stage_is_distinguishable_from_a_passed_one():
    """Both are `index += 1`. A run whose every stage was advanced past is a
    FAILED curriculum that looks finished from the index alone, so the trace is
    the only thing that can tell them apart."""
    ctx = _ctx(_cfg(**{"loop.curriculum.stage_patience": 1,
                       "loop.curriculum.on_stall": "advance"}),
               evaluator=_Judge(0.0))
    st, cur = _state(["a", "b"], iteration=0)
    CUR.run_curriculum_step(ctx, st, Selection(winners=[_report()], losers=[]))
    assert cur.index == 1
    assert [h["event"] for h in cur.history] == ["stalled"]
    assert "0" not in cur.checkpoints, "advancing is not passing"
    assert "NOT completed" in cur.note


def test_abort_ends_the_search_the_way_a_spent_budget_does():
    """`BudgetExceeded` rather than a new exception type: it is the one signal
    `run_search` already treats as 'stop and write what you have', so a
    curriculum that gives up leaves the same complete artifact."""
    ctx = _ctx(_cfg(**{"loop.curriculum.stage_patience": 1,
                       "loop.curriculum.on_stall": "abort"}),
               evaluator=_Judge(0.0))
    st, _cur = _state(["a", "b"], iteration=0)
    with pytest.raises(BudgetExceeded):
        CUR.run_curriculum_step(ctx, st, Selection(winners=[_report()], losers=[]))


# --------------------------------------------------------------------------
# 4 -- the patience freeze, and feasibility
# --------------------------------------------------------------------------

def test_the_derived_patience_is_frozen_at_authoring():
    """OBSERVED, not imagined. With a live `n_iterations // n_stages`, a stalled
    stage splits, which raises n_stages, which lowers the patience, which stalls
    the new stages sooner: 7 -> 9 -> 13 in three iterations, nothing passed. The
    remedy shrinking the budget of the stages it creates is the disease it was
    written to cure."""
    ctx = _ctx(_cfg(**{"loop.n_iterations": 6}))
    cur = CurriculumState(stages=["a", "b", "c"])
    assert CUR._stage_patience(ctx, cur) == 2
    cur.stages = ["a1", "a2", "a3", "b", "c", "d"]  # as a re-split would leave it
    assert CUR._stage_patience(ctx, cur) == 2, (
        "a split must not shrink the budget of the stages it created")


def test_an_explicit_patience_is_read_live():
    """An operator who set a number means that number."""
    ctx = _ctx(_cfg(**{"loop.curriculum.stage_patience": 5}))
    assert CUR._stage_patience(ctx, CurriculumState(stages=["a"] * 20)) == 5


def test_an_infeasible_curriculum_says_so():
    """No coherence rule can catch this: the stage COUNT does not exist until a
    model has been asked, so the config is valid and the run is still one that
    cannot reach its last stage whatever the policy does."""
    ctx = _ctx(_cfg(**{"loop.n_iterations": 2}))
    st = RunState()
    st.subtasks = ["a", "b", "c", "d", "e"]
    cur = CUR.author_subtask_list(ctx, st)
    assert cur.history[-1]["feasible"] is False
    ctx2 = _ctx(_cfg(**{"loop.n_iterations": 20}))
    assert CUR.author_subtask_list(ctx2, st).history[-1]["feasible"] is True


# --------------------------------------------------------------------------
# 5 -- regression and rollback
# --------------------------------------------------------------------------

def test_forgetting_an_earlier_stage_restores_that_stage_s_policy():
    """Against the score the stage PASSED with, not the threshold: a stage that
    passed at 0.95 and now reads 0.75 has lost most of the skill while still
    clearing a 0.7 bar."""
    ctx = _ctx(_cfg(**{"loop.curriculum.regression_check": True,
                       "loop.curriculum.gate": "vlm_ensemble",
                       "loop.curriculum.regression_tolerance": 0.1}),
               evaluator=_Judge(0.4))
    st, cur = _state(["a", "b", "c"], index=2, iteration=5)
    cur.pass_scores = {"0": 0.9, "1": 0.9}
    cur.checkpoints = {"0": "policy:stage0", "1": "policy:stage1"}
    st.policy_ref = "policy:now"
    CUR._regression(ctx, st, cur, _report())
    assert st.policy_ref == "policy:stage0", (
        "the EARLIEST regressed stage, because a later checkpoint keeps a "
        "policy that has already forgotten something")
    rec = [h for h in cur.history if h["event"] == "regressed"][0]
    assert [f["stage_index"] for f in rec["findings"]] == [0, 1]
    assert "REGRESSION" in cur.note and "policy has been restored" in cur.note


def test_a_stage_still_within_tolerance_is_not_a_regression():
    ctx = _ctx(_cfg(**{"loop.curriculum.regression_check": True,
                       "loop.curriculum.gate": "vlm_ensemble",
                       "loop.curriculum.regression_tolerance": 0.3}),
               evaluator=_Judge(0.8))
    st, cur = _state(["a", "b"], index=1, iteration=3)
    cur.pass_scores = {"0": 0.9}
    cur.checkpoints = {"0": "policy:stage0"}
    st.policy_ref = "policy:now"
    assert CUR._regression(ctx, st, cur, _report()) is False
    assert st.policy_ref == "policy:now"


def test_an_unanswered_regression_check_is_not_evidence_of_forgetting():
    """A rollback throws away training. A provider that said nothing has not
    said the skill is gone."""
    ctx = _ctx(_cfg(**{"loop.curriculum.regression_check": True,
                       "loop.curriculum.gate": "vlm_ensemble"}),
               evaluator=_Silent())
    st, cur = _state(["a", "b"], index=1, iteration=3)
    cur.pass_scores = {"0": 0.9}
    cur.checkpoints = {"0": "policy:stage0"}
    st.policy_ref = "policy:now"
    assert CUR._regression(ctx, st, cur, _report()) is False
    assert st.policy_ref == "policy:now"


def test_the_regression_check_is_off_by_default():
    ctx = _ctx(evaluator=_Judge(0.0))
    st, cur = _state(["a", "b"], index=1, iteration=3)
    cur.pass_scores = {"0": 0.9}
    cur.checkpoints = {"0": "policy:stage0"}
    assert CUR._regression(ctx, st, cur, _report()) is False


# --------------------------------------------------------------------------
# 6 -- known traps
# --------------------------------------------------------------------------

def test_frames_do_not_ride_out_on_the_report():
    """`CandidateReport.meta` reaches `RunState`, `save_report` and
    `checkpoint.encode` -- which has a `bytes` tag, so twenty PNGs per candidate
    BLOAT a checkpoint rather than crash it, and nothing would report it
    (`preferences._drop_clips`, same trap)."""
    ctx = _ctx(evaluator=_Judge(1.0))
    st, _cur = _state(["a", "b"])
    winner, loser = _report("c0"), _report("c1")
    CUR.run_curriculum_step(ctx, st, Selection(winners=[winner], losers=[loser]))
    assert CUR._FRAMES_KEY not in winner.meta
    assert CUR._FRAMES_KEY not in loser.meta


def test_the_cache_is_dropped_even_when_the_round_raises():
    ctx = _ctx(_cfg(**{"loop.curriculum.stage_patience": 1,
                       "loop.curriculum.on_stall": "abort"}),
               evaluator=_Judge(0.0))
    st, _cur = _state(["a", "b"], iteration=0)
    winner = _report()
    with pytest.raises(BudgetExceeded):
        CUR.run_curriculum_step(ctx, st, Selection(winners=[winner], losers=[]))
    assert CUR._FRAMES_KEY not in winner.meta


def test_the_journal_event_name_does_not_collide_with_the_event_signature():
    """`Context.event(self, stage, **fields)` names its first positional
    parameter `stage`, so a field called `stage` is a TypeError raised at the one
    moment the journal is the only witness."""
    seen = []
    ctx = _ctx(evaluator=_Judge(1.0))
    ctx.event = lambda name, **kw: seen.append((name, kw))
    st, _cur = _state(["a", "b"])
    CUR.run_curriculum_step(ctx, st, Selection(winners=[_report()], losers=[]))
    names = [n for n, _ in seen]
    assert "curriculum_gate" in names and "curriculum_advance" in names
    assert all("stage" not in kw for _n, kw in seen), (
        "use stage_text; `stage` is Context.event's own parameter")


# --------------------------------------------------------------------------
# 7 -- the schema, and what the shipped configs may claim
# --------------------------------------------------------------------------

def test_no_published_config_selects_the_curriculum():
    """The curriculum is an unpublished part of the space, so no shipped config
    selects it."""
    from bird.config import CONFIG_ROOT

    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        if path.name.startswith("_") and path.parent == CONFIG_ROOT:
            continue
        if path.parent.name == "_profiles":
            continue
        cfg = C.load(str(path.relative_to(CONFIG_ROOT)).removesuffix(".yaml"))
        assert cfg["loop.curriculum.enabled"] is False, path


@pytest.mark.parametrize("overrides,fragment", [
    ({"loop.curriculum.enabled": True, "loop.carry": ["policy_checkpoint"]},
     "'curriculum' in loop.carry"),
    ({"loop.curriculum.enabled": True, "loop.carry": ["curriculum"]},
     "'policy_checkpoint' in loop.carry"),
    ({"loop.curriculum.enabled": True, "loop.carry": CARRY,
      "train.init": "from_scratch"}, "inert under train.init=from_scratch"),
    ({"generate.context.include_curriculum_stage": True},
     "requires loop.curriculum.enabled"),
])
def test_the_coherence_rules_fire(overrides, fragment):
    """Each of these is a config that would RUN and mean nothing: a curriculum
    that restarts at stage 0 every iteration, one whose stages inherit nothing
    from each other, one whose section renders nothing."""
    base = {"loop.carry": ["subtask_list", "dialogue"]}
    base.update(overrides)
    with pytest.raises(Exception) as exc:
        C.load("rda", overrides=base, profile="tester")
    assert fragment in str(exc.value)


def test_eurekas_published_curriculum_is_expressible():
    """The acceptance criterion, applied to the paper that provoked this axis.

    Eureka §4.3 / App. D.1: pen spinning as two stages -- re-orient the pen to
    random targets, then fine-tune THAT policy on the spinning task, against a
    `Scratch` control. Human-authored, two stages, no gate (each runs to its
    budget), one reward across both, warm-start as the handover. Without these
    keys BIRD could not express it; a method that is not a point in the space is
    this repo's definition of a missing knob.
    """
    cfg = _cfg(**{"loop.curriculum.author": "subtask_list",
                  "loop.curriculum.gate": "fixed_budget",
                  "loop.curriculum.stage_patience": 3,
                  "loop.n_iterations": 6})
    ctx = _ctx(cfg)
    st = RunState()
    st.subtasks = ["re-orient the pen to a random target configuration",
                   "spin the pen through the target configurations"]
    cur = CUR.author_subtask_list(ctx, st)
    st.curriculum = cur
    assert cur.stages == st.subtasks, "human-authored order, asked of no model"
    for it in range(6):
        st.iteration = it
        CUR.run_curriculum_step(ctx, st, Selection(winners=[_report()], losers=[]))
    assert cur.index == 2 and cur.complete
    assert [h["event"] for h in cur.history] == ["authored", "passed", "passed"]


def test_the_search_runs_end_to_end_with_the_curriculum_on(tmp_path):
    """The tester-tier curriculum point, run for real."""
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    cfg = _cfg(**{"loop.curriculum.gate": "vlm_ensemble",
                  "loop.curriculum.gate_threshold": 0.2,
                  "loop.curriculum.regression_check": True,
                  "generate.n_candidates": 2,
                  "generate.decomposition.n_subtasks": 3,
                  "loop.n_iterations": 4})
    result = entry.run(cfg, out_root=str(tmp_path))
    assert result["config_hash"] == cfg.hash()
    states = sorted((tmp_path).rglob("state/*.json"))
    assert states, "the run wrote no state"
    trace = json.loads(states[-1].read_text())["curriculum"]
    assert trace["n_stages"] >= 2 and trace["patience"] >= 1
    assert trace["history"][0]["event"] == "authored"
    assert any(h["event"] in ("passed", "stalled") for h in trace["history"]), (
        "a curriculum that neither passed nor stalled a stage in four "
        "iterations recorded nothing a reader could act on")


def test_the_curriculum_survives_a_checkpoint_round_trip():
    """Resume is where a silent loss here would cost most.

    `checkpoint.encode` raises on anything it has no tag for, so a
    `CurriculumState` missing from `_DECODABLE` fails at iteration 0 rather than
    at the resume at hour 18 -- that much is the codec's own design. What this
    pins is the other half: that every FIELD survives. A curriculum restored
    with its stages but not its `index`, `checkpoints` or `pass_scores` resumes
    at stage 0 with nothing to roll back to, and reads exactly like a fresh run
    of a config that has one.
    """
    from bird.checkpoint import _ArraySpill, decode, encode

    cur = CurriculumState(
        stages=["stand up", "lift the bar"], index=1, entered_at=2,
        history=[{"event": "passed", "stage_index": 0, "score": 0.9}],
        checkpoints={"0": "policy:c0"}, pass_scores={"0": 0.9},
        note="REGRESSION: the agent has lost a skill.", splits=1, patience=3)
    st = RunState(iteration=3)
    st.curriculum = cur

    back = decode(encode(st, _ArraySpill(None)), None).curriculum
    assert isinstance(back, CurriculumState), (
        "decoding resolves names only out of the closed `_DECODABLE` allow-list; "
        "a curriculum that came back as a dict would be a resumed run whose "
        "every stage lookup raises")
    assert back == cur, "a field lost here is a search that silently restarts"
    assert back.current == "lift the bar" and back.complete is False


class _ListClient:
    """A client that returns a JSON stage list -- what an author/resplit parses."""

    modality = "text"

    def __init__(self, stages):
        self.stages = list(stages)
        self.calls = []

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        self.calls.append(tag)
        return ['```json\n{"stages": %s}\n```' % json.dumps(self.stages)] * n
