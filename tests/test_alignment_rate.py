"""RDA's reported metric, and a silent degradation on the path to it.

`evaluate.fitness.source: vlm_score` is RDA's whole selection signal, and the
path that reaches the VLM depends on `_as_text` handling a list, which is exactly
what `bird/llm/base.py` says a client returns. Without that, every judgment comes
back empty and both callers degrade quietly on empty text -- the scorer to the
environment's success flag, the human oracle to None. A VLM-graded method becomes
a blind comparator and no artifact says so.

A suite can pass throughout that failure if nothing asserts that the VLM was
actually consulted. These tests do.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from bird.components.evaluation import _as_text, _client_text
from bird.config import CONFIG_ROOT


class _ContractClient:
    """A client honouring `bird/llm/base.py`: returns `list[str]`."""

    def __init__(self, text="Score: 4/5\nReason: mostly aligned."):
        self.text = text
        self.calls = 0

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        self.calls += 1
        return [self.text] * n


def test_as_text_unwraps_the_documented_list_return():
    assert _as_text(["hello"]) == "hello"
    assert _as_text(("hello", "second")) == "hello", "n=1 callers take the first sample"
    assert _as_text([]) == ""
    assert _as_text(["", "second"]) == "second", "an empty first sample is not the answer"
    # the shapes that already worked must keep working
    assert _as_text("plain") == "plain"
    assert _as_text({"text": "d"}) == "d"
    assert _as_text(types.SimpleNamespace(content="attr")) == "attr"


def test_a_contract_conforming_client_is_actually_read():
    """The failure itself: without the list branch this returns "", so every VLM
    judgment silently becomes a success-flag lookup."""
    client = _ContractClient()
    text = _client_text(client, "Rate this on a 1.0-5.0 scale")
    assert client.calls >= 1
    assert "4/5" in text, "the client answered and the caller threw it away"


def test_a_dead_client_still_degrades_quietly():
    """The fallback is correct behaviour, not a bug -- it just must not be
    reached by a client that answered."""
    assert _client_text(None, "anything") == ""
    assert _client_text(object(), "anything") == ""


# --------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------

def test_divide_by_max_is_not_min_max():
    """§10.1 normalises "by dividing by 5", so a Likert 1 is 0.2, not 0.0.
    Min-max would shift every published comparison by a fifth of the scale."""
    lo, hi = 1.0, 5.0
    mean = 1.0
    assert mean / hi == pytest.approx(0.2)
    assert (mean - lo) / (hi - lo) == pytest.approx(0.0)


def test_rda_pins_the_two_scorers_apart():
    """The in-loop scorer is 3-point on the [0,1] codomain (App. 7.3
    discretising §4.2's `s in [0,1]`); the reported
    metric (§5.1) is a 5-point Likert. One key for both would rescale the
    signal the search optimises -- which is why `alignment_rate.*` is a
    separate block."""
    from bird import config as cfgmod
    cfg = cfgmod.load(CONFIG_ROOT / "methods" / "rda.yaml")
    assert cfg.get("evaluate.feedback.score_scale") == "three_point", \
        "App. 7.3's in-loop rubric"
    assert cfg.get("alignment_rate.scale") == "likert", "§5.1 reported metric"
    assert cfg.get("alignment_rate.normalisation") == "divide_by_max", "§10.1"
    # §5.1 pins different budgets for the two evaluators.
    assert cfg.get("evaluate.vlm.repeats") == 1, "in-loop repeat count is unstated"
    assert cfg.get("alignment_rate.repeats") == 4, "§10.1: four queries per video"
    assert cfg.get("evaluate.vlm.images_per_query") == 20
    assert cfg.get("alignment_rate.images_per_query") == 50


def test_the_metric_rates_a_fresh_policy_not_a_carried_report():
    """`bird/state.py` forbids reading rollouts off a CARRIED report: they are
    dropped from the checkpoint, so a resumed run has none. Reading them made
    this metric answer `ok` after a straight run and `not_run` after a resume.
    The phase must take its rollouts from `final_retrain`'s live result."""
    src = Path("bird/components/phases.py").read_text()
    body = src[src.index('@register("phase", "alignment_rate")'):]
    body = body[:body.index('@register("phase", "real_world_eval")')]
    assert "final_retrain_result" in body, "the metric must rate a fresh policy"
    # Comments deliberately NAME the forbidden expression to explain why it is
    # forbidden, so match code lines only.
    code = "\n".join(ln for ln in body.splitlines()
                      if not ln.lstrip().startswith("#"))
    assert "report.result.trajectories" not in code, (
        "reading a carried report's rollouts makes a resumed run divergent "
        "(bird/state.py opening note)")


def test_rda_orders_final_retrain_before_the_metric():
    from bird import config as cfgmod
    post = list(cfgmod.load(CONFIG_ROOT / "methods" / "rda.yaml").get("post") or [])
    assert post.index("final_retrain") < post.index("alignment_rate"), (
        "alignment_rate rates the policy final_retrain produces")


# --------------------------------------------------------------------------
# The metric asked for more videos than the protocol could produce
# --------------------------------------------------------------------------

def test_the_retrain_produces_the_videos_the_metric_was_pinned_to_rate():
    """App. §10.1: "for each trained policy, we collect 5 trajectory videos ...
    we query the VLM four times per video, yielding 20 evaluations per policy
    ... each task is trained with 3 random seeds, this results in 60 total
    evaluations per task."

    Both numbers in the config are the paper's, and neither is reachable if
    `final_retrain` rolls out `evaluate.rollouts_per_candidate` (RDA's
    SEARCH-side K=3, Table 1): `alignment_rate` then slices `[:5]` of 3, every
    rda job logs "wanted 5 video(s), the retrained policy produced 3", and the
    headline metric is reported over 12 judgments instead of 20. The pin that
    matters is neither of them -- it is the retrain's own rollout count.
    """
    from bird import config as cfgmod
    from bird.components.phases import _report_rollouts

    cfg = cfgmod.load(CONFIG_ROOT / "methods" / "rda.yaml")
    n_seeds = cfg["final_retrain.n_seeds"]
    assert cfg["evaluate.rollouts_per_candidate"] == 3, "RDA's K, Table 1"
    assert cfg["alignment_rate.n_videos"] == 5, "App. §10.1, per policy"
    assert n_seeds == 3, "App. §10.1: three random seeds"

    n = _report_rollouts(cfg, n_seeds)
    assert n == 15, ("5 videos PER POLICY over 3 seed policies; `_sb3_run` cycles "
                     "its final rollouts over per_seed_policies[i % n_seeds], so "
                     "15 is 5 each and 5 would be 2/2/1")
    assert n * cfg["alignment_rate.repeats"] == 60, "App. §10.1's judgment count"


def test_the_search_side_rollout_count_is_not_disturbed():
    """`_report_rollouts` widens the RETRAIN only, and only upward. The search's
    own K decides what stage 4 ranks candidates on and must not move; nor may
    the retrain shrink below it, or `final_retrain`'s reported fitness would be
    computed over less evidence than the fitness it is compared against."""
    from bird import config as cfgmod
    from bird.components.phases import _report_rollouts

    for stem in ("eureka.yaml", "card.yaml", "limen.yaml"):
        cfg = cfgmod.load(CONFIG_ROOT / "methods" / stem)
        if cfg.get("alignment_rate.enabled"):
            continue
        assert _report_rollouts(cfg, cfg.get("final_retrain.n_seeds") or 1) == \
            cfg["evaluate.rollouts_per_candidate"], (
                f"{stem} has no alignment_rate and must be untouched")

    # ...and a config that enables the phase but never lists it in `post:` gets
    # nothing either: the phase that reads the rollouts is the one that pays.
    cfg = cfgmod.load(CONFIG_ROOT / "methods" / "rda.yaml", overrides={"post": ["final_retrain"]})
    from bird.components.phases import _report_rollouts as rr
    assert rr(cfg, 3) == cfg["evaluate.rollouts_per_candidate"]


def test_the_scoring_of_the_retrain_uses_the_searchs_own_k():
    """The widened count is applied around the BACKEND call only. Scoring the
    retrain over 15 rollouts while the selected fitness came from 3 would make
    `final_retrain.json`'s `gap` -- the whole point of the phase -- a comparison
    of two different estimators."""
    src = Path("bird/components/phases.py").read_text()
    body = src[src.index('@register("phase", "final_retrain")'):]
    body = body[:body.index('@register("phase", "alignment_rate")')]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    call = code.index("result = backend(")
    restore = code.index("ctx.cfg = prev_cfg", call)
    score = code.index("scored = score(", call)
    assert restore < score, (
        "the config must be restored between the retrain's rollouts and its scoring")
