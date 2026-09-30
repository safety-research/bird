"""`bird/envs/hud.py` -- the font and the primitives that draw a command legend.

Deliberately NOT marked `slow` and importing no simulator. Every other test of the
command legend `importorskip`s `mujoco` and is therefore skipped under the default test
install, and the drawing is the half a reader notices when it breaks. This file is the
part that runs everywhere.

Only properties whose failure is silent: a glyph that renders blank, a caption that
vanishes off an edge instead of clipping, a panel that does not contain its text.
"""
from __future__ import annotations

import numpy as np
import pytest

from bird.envs import hud


def _blank(h: int = 40, w: int = 120) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_every_advertised_character_has_a_glyph_and_none_is_blank():
    for ch in hud._SHEET_CHARS:
        bits = hud.FONT[ch]
        assert bits.shape == (hud.GLYPH_H, hud.GLYPH_W), ch
        if ch != " ":
            assert bits.any(), f"{ch!r} is an all-blank glyph"
    assert len(hud.FONT) == len(set(hud._SHEET_CHARS))


def test_the_sheet_and_its_character_list_are_the_same_length():
    # The sheet is five bands of ten; a band that gains a glyph without the string
    # gaining a character would silently rename every glyph after it.
    bands = hud._SHEET.strip("\n").split("\n\n")
    assert len(bands) * 10 == len(hud._SHEET_CHARS)


def test_an_unknown_character_draws_a_visible_box_rather_than_nothing():
    # A silently blank glyph is indistinguishable from a space, so "the label lost a
    # character" would never be reported.
    assert hud.glyph("~").any()
    assert hud.glyph("~").tolist() == hud._TOFU.tolist()
    assert not hud.glyph(" ").any()


def test_lowercase_reaches_the_uppercase_glyph():
    assert hud.glyph("a").tolist() == hud.FONT["A"].tolist()


def test_draw_text_marks_pixels_and_reports_the_width_it_used():
    img = _blank()
    w = hud.draw_text(img, 2, 2, "AB", scale=1, shadow=None)
    assert w == hud.text_size("AB", 1)[0]
    assert img.any()
    assert int((img > 0).any(axis=2).sum()) == int(hud.FONT["A"].sum() + hud.FONT["B"].sum())


def test_scale_multiplies_the_inked_area_by_its_square():
    a, b = _blank(60, 200), _blank(60, 200)
    hud.draw_text(a, 2, 2, "W", scale=1, shadow=None)
    hud.draw_text(b, 2, 2, "W", scale=3, shadow=None)
    assert (b > 0).any(axis=2).sum() == 9 * (a > 0).any(axis=2).sum()


@pytest.mark.parametrize("x,y", [(-40, 4), (400, 4), (4, -40), (4, 400), (-4, -4)])
def test_text_clips_at_every_edge_instead_of_raising(x, y):
    # A legend is decoration on a diagnostic artifact. One that raises out of render()
    # takes down a rollout that is otherwise fine.
    img = _blank()
    hud.draw_text(img, x, y, "HELLO", scale=2)
    assert img.shape == (40, 120, 3)


def test_a_partly_offscreen_string_still_draws_the_part_that_fits():
    img = _blank()
    hud.draw_text(img, -6, 4, "MM", scale=1, shadow=None)
    assert img.any(), "a string overlapping the left edge drew nothing at all"


def test_the_shadow_is_what_makes_text_readable_on_a_pale_background():
    lit, plain = np.full((40, 120, 3), 255, np.uint8), np.full((40, 120, 3), 255, np.uint8)
    hud.draw_text(lit, 4, 4, "OK", (255, 255, 255), 2, shadow=(0, 0, 0))
    hud.draw_text(plain, 4, 4, "OK", (255, 255, 255), 2, shadow=None)
    assert (plain == 255).all(), "white on white should be invisible without a shadow"
    assert not (lit == 255).all(), "the shadow did not draw"


def test_draw_rect_blends_rather_than_replacing_when_alpha_is_partial():
    img = np.full((10, 10, 3), 200, np.uint8)
    hud.draw_rect(img, 0, 0, 10, 10, (0, 0, 0), alpha=0.5)
    assert int(img[5, 5, 0]) == 100


def test_a_rectangle_entirely_off_frame_changes_nothing():
    img = _blank()
    before = img.copy()
    hud.draw_rect(img, -50, -50, 10, 10, (255, 0, 0))
    hud.draw_rect(img, 500, 500, 10, 10, (255, 0, 0))
    assert (img == before).all()


def test_the_panel_reports_a_size_that_contains_its_longest_line():
    img = _blank(80, 300)
    w, h = hud.draw_panel(img, 4, 4, ["SHORT", "A MUCH LONGER LINE"], scale=1)
    assert w >= hud.text_size("A MUCH LONGER LINE", 1)[0]
    assert h >= 2 * hud.GLYPH_H
    assert hud.draw_panel(img, 0, 0, [], scale=1) == (0, 0)
