"""The six stage interfaces.

They are the real API, so they are pinned here: if a signature changes, every
component in the repo is affected and the change should be deliberate.
"""

import inspect

import bird as pkg
from bird.types import Candidate, CandidateReport, Selection, TrainResult

import bird as _  # noqa: F401  (ensures sys.path from conftest)


def _stage_module():
    import importlib.util
    from conftest import REPO
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXPECTED = {
    "generate": ["ctx", "state"],
    "verify": ["ctx", "state", "candidates"],
    "train": ["ctx", "state", "candidates"],
    "evaluate": ["ctx", "state", "results"],
    "select": ["ctx", "state", "reports"],
    "update": ["ctx", "state", "selection"],
}


def test_all_six_stages_exist_with_pinned_signatures():
    mod = _stage_module()
    for name, params in EXPECTED.items():
        fn = getattr(mod, name, None)
        assert callable(fn), f"stage {name}() is missing from bird.py"
        got = list(inspect.signature(fn).parameters)
        assert got == params, f"{name}{tuple(got)} != {name}{tuple(params)}"


def test_report_carries_both_a_scalar_and_prose():
    """§4's two outputs. Stage 5 reads .fitness; stage 6 reads .feedback."""
    fields = CandidateReport.__dataclass_fields__
    assert "fitness" in fields and "feedback" in fields


def test_no_ranking_scalar_is_representable():
    """CARD computes no fitness at all -- None must be distinct from 0.0."""
    c = Candidate(cand_id="c0", iteration=0, reward_code="")
    r = CandidateReport(cand_id="c0", candidate=c, result=TrainResult("c0", c))
    assert r.fitness is None


def test_failure_is_recorded_not_dropped():
    """A program that never compiled and one a screen rejected are
    different populations, and both must survive into the artifact."""
    c = Candidate(cand_id="c0", iteration=0, reward_code="")
    invalid = c.failed("SyntaxError: unexpected EOF", kind="invalid")
    assert invalid.failure and invalid.failure_kind == "invalid"
    assert not invalid.trainable

    screened = Candidate(cand_id="c1", iteration=0, reward_code="",
                         screened_out=True, failure="TAC below top-n",
                         failure_kind="screened")
    assert screened.valid, "a screened candidate is still valid code"
    assert not screened.trainable
    assert screened.failure_kind != invalid.failure_kind


def test_screened_candidates_are_outside_the_bt_contest():
    """A TAC-screened, never-trained reward can never be R_best.

    GT Alg. 1 l.11-15: the preferences p_ij, the strengths b_1:N and the argmax
    are all over the N RACED agents. A BT rule that contested every report
    would, through `_bt_strengths`' prior anchor, hand each unjudged screened
    candidate roughly the pool-average strength -- so in an exactly-tied round
    (a regular tournament maps every contender to identical floats)
    `tie_break: first` would hand the WIN to a screened, never-trained reward.
    That winner reads as a perfectly plausible selection in every
    artifact -- the failure sentinel never applies because this rule ranks by
    strength, not fitness -- which is the silent-failure shape that justifies
    the test.
    """
    from bird import registry
    from bird.budget import Budget
    from bird.context import Context
    from bird.state import RunState
    from bird.types import Preference
    from conftest import load_gt_published

    registry.load_all()
    # The published GT point: gt + GT_PUBLISHED_OVERRIDES (one config per
    # method).
    cfg = load_gt_published(profile="tester")
    assert cfg["select.rule"] == "bradley_terry"
    assert cfg["select.tie_break"] == "first"
    events = []

    class _Rundir:
        def event(self, stage, **fields):
            events.append({"stage": stage, **fields})

    ctx = Context(cfg=cfg, budget=Budget())
    ctx.rundir = _Rundir()

    def rep(cid, *, screened):
        cand = Candidate(
            cand_id=cid, iteration=0, reward_code="def f(): ...",
            screened_out=screened,
            failure="tac: cold-start cut" if screened else "",
            failure_kind="screened" if screened else "")
        res = TrainResult(cand_id=cid, candidate=cand, trained=not screened)
        return CandidateReport(
            cand_id=cid, candidate=cand, result=res,
            fitness=-10000.0 if screened else None,
            fitness_source="preference_bt")

    # The screened pair sorts FIRST by cand_id: in a pool that included it, it
    # would sit on the prior anchor, tied exactly with the 1-1 raced agents,
    # and `first` would pick it.
    reports = [rep("c0000", screened=True), rep("c0001", screened=True),
               rep("c0002", screened=False), rep("c0003", screened=False)]
    state = RunState()
    # A regular tournament over the two raced agents: one win each, so the MM
    # fit gives them identical strengths -- the exactly-tied round.
    state.preferences = [
        Preference(left_id="c0002", right_id="c0003", label=1, source="vlm",
                   iteration=0),
        Preference(left_id="c0003", right_id="c0002", label=1, source="vlm",
                   iteration=0),
    ]
    sel = registry.get("select_rule", "bradley_terry")(ctx, state, reports)

    assert sel.winners and sel.winners[0].result.trained, (
        "the BT winner must be a raced agent; a screened, never-trained "
        f"candidate won: {sel.winners[0].cand_id}")
    strengths = [e for e in events if e["stage"] == "select_bt"][0]["strengths"]
    assert set(strengths) == {"c0002", "c0003"}, (
        "screened candidates must appear in no strengths payload: "
        f"{sorted(strengths)}")


def test_mean_per_step_return_corrects_length_bias():
    """Both TAC and TPE specify this; a length-biased return inverts them."""
    from bird.types import Trajectory
    short_success = Trajectory(rewards=[1, 1], success=True, length=2, ret=2.0)
    long_failure = Trajectory(rewards=[0.4] * 10, success=False, length=10, ret=4.0)
    assert long_failure.ret > short_success.ret
    assert short_success.mean_per_step_return > long_failure.mean_per_step_return


def test_selection_defaults_to_contested():
    assert Selection().contested is True
    assert Selection().winner is None


def test_public_api_is_importable():
    for name in ("Config", "Context", "RunState", "Candidate", "CandidateReport",
                 "Selection", "Trajectory", "TrainResult", "load"):
        assert hasattr(pkg, name), f"bird.{name} is not exported"
