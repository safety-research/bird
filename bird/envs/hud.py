"""A 5x7 bitmap font and the few primitives needed to draw a legend onto a frame.

Pure numpy, no simulator and no GL. That is deliberate: tests of the simulator-backed
envs `importorskip` `mujoco` and are skipped wherever it is not installed, while the
drawing this module does is exactly the part a reader notices when it breaks. Keeping it
dependency-free means its tests run everywhere.

The font is stored as a SHEET -- five rows of ten glyphs, drawn with `#` and `.` -- so a
reader can see the letters in the source rather than decode a packed integer. A
character the sheet has no glyph for draws as a hollow box: a silently blank glyph and a
space are indistinguishable on screen, and "the label lost a character" is the kind of
defect nobody reports.

Every primitive CLIPS rather than raising. A legend is decoration on a diagnostic
artifact; a frame that renders with the caption half off the edge is a cosmetic bug, and
one that raises out of `render()` takes down a rollout that is otherwise fine.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

GLYPH_W, GLYPH_H = 5, 7
#: Space between glyphs, in unscaled pixels.
TRACKING = 1

_SHEET_CHARS = (
    "0123456789"
    "ABCDEFGHIJ"
    "KLMNOPQRST"
    "UVWXYZ.-+/"
    ":%<>=[]?, "
)

#: Five bands of ten glyphs. Read them with your eyes -- that is the point.
_SHEET = """
.###. ..#.. .###. .###. ...#. ##### ..##. ##### .###. .###.
#...# .##.. #...# #...# ..##. #.... .#... ....# #...# #...#
#..## ..#.. ....# ....# .#.#. ####. #.... ...#. #...# #...#
#.#.# ..#.. ...#. ..##. #..#. ....# ####. ..#.. .###. .####
##..# ..#.. ..#.. ....# ##### ....# #...# .#... #...# ....#
#...# ..#.. .#... #...# ...#. #...# #...# .#... #...# ...#.
.###. .###. ##### .###. ...#. .###. .###. .#... .###. .##..

..#.. ####. .###. ###.. ##### ##### .###. #...# .###. ..###
.#.#. #...# #...# #..#. #.... #.... #...# #...# ..#.. ...#.
#...# #...# #.... #...# #.... #.... #.... #...# ..#.. ...#.
#...# ####. #.... #...# ####. ####. #.### ##### ..#.. ...#.
##### #...# #.... #...# #.... #.... #...# #...# ..#.. ...#.
#...# #...# #...# #..#. #.... #.... #...# #...# ..#.. #..#.
#...# ####. .###. ###.. ##### #.... .###. #...# .###. .##..

#...# #.... #...# #...# .###. ####. .###. ####. .###. #####
#..#. #.... ##.## ##..# #...# #...# #...# #...# #...# ..#..
#.#.. #.... #.#.# #.#.# #...# #...# #...# #...# #.... ..#..
##... #.... #.#.# #..## #...# ####. #...# ####. .###. ..#..
#.#.. #.... #...# #...# #...# #.... #.#.# #.#.. ....# ..#..
#..#. #.... #...# #...# #...# #.... #..#. #..#. #...# ..#..
#...# ##### #...# #...# .###. #.... .##.# #...# .###. ..#..

#...# #...# #...# #...# #...# ##### ..... ..... ..... ....#
#...# #...# #...# #...# #...# ....# ..... ..... ..#.. ....#
#...# #...# #...# .#.#. .#.#. ...#. ..... ..... ..#.. ...#.
#...# #...# #.#.# ..#.. ..#.. ..#.. ..... ##### ##### ..#..
#...# #...# #.#.# .#.#. ..#.. .#... ..... ..... ..#.. .#...
#...# .#.#. ##.## #...# ..#.. #.... ..##. ..... ..#.. #....
.###. ..#.. #...# #...# ..#.. ##### ..##. ..... ..... #....

..... ##..# ...#. .#... ..... ..### ###.. .###. ..... .....
..##. ##..# ..#.. ..#.. ..... ..#.. ..#.. #...# ..... .....
..##. ...#. .#... ...#. ##### ..#.. ..#.. ....# ..... .....
..... ..#.. #.... ....# ..... ..#.. ..#.. ...#. ..... .....
..##. .#... .#... ...#. ##### ..#.. ..#.. ..#.. ..##. .....
..##. #..## ..#.. ..#.. ..... ..#.. ..#.. ..... ..##. .....
..... #..## ...#. .#... ..... ..### ###.. ..#.. .#... .....
"""


def _build() -> dict:
    bands = [b.split("\n") for b in _SHEET.strip("\n").split("\n\n")]
    out: dict = {}
    for band_i, rows in enumerate(bands):
        if len(rows) != GLYPH_H:
            raise ValueError(f"font band {band_i} has {len(rows)} rows, want {GLYPH_H}")
        for col in range(10):
            ch = _SHEET_CHARS[band_i * 10 + col]
            x0 = col * (GLYPH_W + 1)
            bits = np.zeros((GLYPH_H, GLYPH_W), dtype=bool)
            for r, row in enumerate(rows):
                cell = row[x0:x0 + GLYPH_W]
                if len(cell) != GLYPH_W:
                    raise ValueError(f"font glyph {ch!r} row {r} is {len(cell)} wide")
                bits[r] = [c == "#" for c in cell]
            out[ch] = bits
    return out


FONT = _build()

#: Drawn for a character the sheet has no glyph for -- visible, never silent.
_TOFU = np.zeros((GLYPH_H, GLYPH_W), dtype=bool)
_TOFU[0, :] = _TOFU[-1, :] = True
_TOFU[:, 0] = _TOFU[:, -1] = True


def glyph(ch: str) -> np.ndarray:
    """The bitmap for `ch`, upper-cased; the tofu box when there is no glyph."""
    return FONT.get(ch.upper(), _TOFU)


def text_size(s: str, scale: int = 1) -> Tuple[int, int]:
    """(width, height) in pixels that `draw_text` would occupy."""
    if not s:
        return 0, 0
    w = len(s) * GLYPH_W + (len(s) - 1) * TRACKING
    return w * scale, GLYPH_H * scale


def _blend(img: np.ndarray, y0: int, y1: int, x0: int, x1: int,
           rgb: Sequence[int], alpha: float) -> None:
    """Alpha-blend a solid colour into a clipped rectangle of `img`, in place."""
    h, w = img.shape[:2]
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y0 >= y1 or x0 >= x1:
        return
    patch = img[y0:y1, x0:x1].astype(np.float32)
    col = np.asarray(rgb, dtype=np.float32)
    img[y0:y1, x0:x1] = np.clip(patch * (1.0 - alpha) + col * alpha, 0, 255).astype(np.uint8)


def draw_rect(img: np.ndarray, x: int, y: int, w: int, h: int,
              rgb: Sequence[int] = (0, 0, 0), alpha: float = 1.0) -> None:
    """Filled rectangle, clipped to the frame."""
    _blend(img, int(y), int(y + h), int(x), int(x + w), rgb, float(alpha))


def draw_text(img: np.ndarray, x: int, y: int, s: str,
              rgb: Sequence[int] = (255, 255, 255), scale: int = 1,
              shadow: Optional[Sequence[int]] = (0, 0, 0)) -> int:
    """Draw `s` with its top-left at (x, y). Returns the width drawn.

    `shadow` offsets a second copy one scaled pixel down-right underneath, which is what
    keeps a legend readable over both the pale floor checker and the dark skybox; pass
    None to suppress it.
    """
    scale = max(1, int(scale))
    if shadow is not None:
        _draw_text_1(img, int(x) + scale, int(y) + scale, s, shadow, scale)
    _draw_text_1(img, int(x), int(y), s, rgb, scale)
    return text_size(s, scale)[0]


def _draw_text_1(img: np.ndarray, x: int, y: int, s: str,
                 rgb: Sequence[int], scale: int) -> None:
    h, w = img.shape[:2]
    col = np.asarray(rgb, dtype=np.uint8)
    step = (GLYPH_W + TRACKING) * scale
    for i, ch in enumerate(s):
        gx = x + i * step
        if gx >= w or gx + GLYPH_W * scale <= 0:
            continue
        bits = glyph(ch)
        if not bits.any():
            continue
        big = np.repeat(np.repeat(bits, scale, axis=0), scale, axis=1)
        gh, gw = big.shape
        # Clip the glyph against the frame on all four sides.
        sy0, sx0 = max(0, -y), max(0, -gx)
        sy1, sx1 = gh - max(0, (y + gh) - h), gw - max(0, (gx + gw) - w)
        if sy0 >= sy1 or sx0 >= sx1:
            continue
        m = big[sy0:sy1, sx0:sx1]
        img[y + sy0:y + sy1, gx + sx0:gx + sx1][m] = col


def draw_panel(img: np.ndarray, x: int, y: int, lines: Sequence[str],
               scale: int = 1, pad: int = 3, alpha: float = 0.45,
               rgb: Sequence[int] = (255, 255, 255),
               bg: Sequence[int] = (0, 0, 0)) -> Tuple[int, int]:
    """A translucent plate with `lines` of text on it. Returns its (width, height)."""
    if not lines:
        return 0, 0
    gw, gh = text_size(max(lines, key=len) or " ", scale)
    lead = gh + max(1, scale)
    w = gw + 2 * pad
    h = lead * len(lines) - (lead - gh) + 2 * pad
    draw_rect(img, x, y, w, h, bg, alpha)
    for i, line in enumerate(lines):
        draw_text(img, x + pad, y + pad + i * lead, line, rgb, scale)
    return w, h
