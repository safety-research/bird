"""`evaluate.feedback.score_scale: staged` -- the anchored progress scale.

`staged` is an anchored, continuous progress scale: the judge is shown named
anchor points along [0, 1] and asked to interpolate between them rather than
pick one. What these tests pin is the part a refactor would break silently: the
anchors actually reach the prompt, the scale stays CONTINUOUS (an unanchored
judge collapses onto habitual values and equal scores are dropped as ties by
everything that consumes an ordering -- snapping would reintroduce exactly
that), and the enum is honest in both the schema and the loader.
"""
from __future__ import annotations


from bird.components.evaluation import _scale_bounds, _score_rubric, _snap


class _Ctx:
    def __init__(self, scale):
        self.cfg = type("C", (), {"get": lambda s, k, d=None:
                        {"evaluate.feedback.score_scale": scale}.get(k, d)})()


def test_staged_renders_every_anchor_and_asks_for_interpolation():
    text = _score_rubric("staged", 0.0, 1.0)
    for anchor in ("0.0", "0.25", "0.5", "0.75", "1.0"):
        assert anchor in text, f"anchor {anchor} missing from the rubric"
    # The stages, not just the numbers -- a bounds sentence with five floats
    # in it would pass a number-only check and anchor nothing.
    for stage in ("no progress", "approaches or orients", "clearly begins",
                  "most of the way", "goal state is achieved"):
        assert stage in text
    assert "interpolat" in text.lower()


def test_staged_is_continuous_never_snapped():
    # three_point quantises; staged must not, or the rank resolution the
    # scale exists for is quantised away.
    assert _snap(0.37, "staged") == 0.37
    assert _snap(0.37, "three_point") == 0.5
    assert _scale_bounds(_Ctx("staged")) == (0.0, 1.0)


def test_the_default_scale_prompt_is_unchanged():
    assert _score_rubric("[0,1]", 0.0, 1.0) == \
        "- Score each subtask on a 0.0-1.0 scale."


def test_the_loader_accepts_staged():
    from bird.config import load
    cfg = load("methods/rda.yaml",
               overrides={"evaluate.feedback.score_scale": "staged"})
    assert cfg["evaluate.feedback.score_scale"] == "staged"
