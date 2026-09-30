"""`frame_policy` -- how a rollout's sampled frames are DELIVERED to the judge.

Two entries in the `frame_policy` family (`evaluate.vlm.frame_policy`):

  even            one image per sampled frame. What every published method point
                  has always done; a strict identity, so the default path is
                  byte-for-byte what it was before this family existed.
  contact_sheet   the same frames tiled into ONE image, at reduced per-cell
                  resolution, with the episode step index burned into each cell
                  and repeated in a text manifest.

WHY PACKING IS NOT JUST COMPRESSION. Measured on three real rollout clips
(`gym_half_cheetah`, `mt10_reach-v3`, `h1hand_powerlift`), 64 frames at 96px as
one sheet against the default 16 frames at native 320px:

    half_cheetah          393 tok   vs  1,092 tok
    h1hand_powerlift      590 tok   vs  1,638 tok
    mt10_reach            786 tok   vs  2,185 tok

Four times the frames for about a third of the cost. The saving is NOT in the
pixels -- a 8x8 grid of 96px cells is the same pixel count as 64 separate 96px
images -- it is in collapsing 64 image blocks into one, and in being able to
drop per-cell resolution further than you would dare for a standalone frame
because neighbouring cells supply the context.

`cell_px` DEFAULTS TO 96 BECAUSE 96 IS A FLOOR, NOT A PREFERENCE. Same
measurement: `mt10_reach`'s red goal marker is a single faint pixel at 64px and
absent at 48px, while `h1hand_powerlift`'s standing/collapsed distinction
survives 48px and `half_cheetah`'s body pose survives 64px. A global constant
therefore either blinds the manipulation tasks or overpays for the locomotion
ones, which is why this is a config key and wants a per-task override.

THE LABEL AND THE MANIFEST CARRY THE SAME NUMBERS, AND THAT IS THE POINT. The
index is the EPISODE STEP, never the cell's position in the grid: positions
change when the grid changes and steps do not, and `judgments.frame_set` already
records steps for exactly this reason. Burned into the pixel AND stated in the
manifest, because a judge naming "frame 137" and a record resolving "frame 137"
must mean one instant. If those two ever disagree, every judgment built on them
is wrong and nothing complains.

NO NEW DEPENDENCY. Tiling is numpy and the encoder is `observability.encode_png`
-- the same hand-rolled one that lets `output.video.enabled` default to true.
The digit font below exists for the same reason that encoder does: drawing three
digits is not worth a Pillow dependency in a path that runs on every judgment.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..observability import encode_png
from ..registry import register

#: 3x5 bitmaps, one per digit. Scaled up when drawn; see `_draw_label`.
_FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}
_GLYPH_W, _GLYPH_H = 3, 5
_INK = np.array([255, 235, 0], dtype=np.uint8)      # yellow on black, both extremes
_PLATE = np.array([0, 0, 0], dtype=np.uint8)         # so the label survives any scene


def _resize_nearest(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbour resize of (H, W, 3) uint8. No interpolation on purpose.

    A judge is being asked what the robot did, not to admire the resampling, and
    nearest is the one filter that cannot invent a pixel value that was never
    rendered -- which matters when the thing being looked for is a small
    high-contrast marker a smoothing filter would blend into the table.
    """
    src_h, src_w = img.shape[:2]
    if (src_w, src_h) == (w, h):
        return img
    ys = (np.arange(h) * src_h // h).clip(0, src_h - 1)
    xs = (np.arange(w) * src_w // w).clip(0, src_w - 1)
    return img[ys][:, xs]


def _draw_label(cell: np.ndarray, text: str, scale: int) -> None:
    """Burn `text` into the top-left of `cell`, in place. Never raises."""
    if scale < 1 or not text:
        return
    gw, gh = _GLYPH_W * scale, _GLYPH_H * scale
    pad = max(1, scale)
    w = len(text) * (gw + pad) + pad
    h = gh + 2 * pad
    if h >= cell.shape[0] or w >= cell.shape[1]:
        return                                   # cell too small to label honestly
    cell[:h, :w] = _PLATE
    x = pad
    for ch in text:
        rows = _FONT.get(ch)
        if rows is not None:
            for r, row in enumerate(rows):
                for c, bit in enumerate(row):
                    if bit == "1":
                        y0, x0 = pad + r * scale, x + c * scale
                        cell[y0:y0 + scale, x0:x0 + scale] = _INK
        x += gw + pad


def _grid_shape(n: int, spec: str) -> Tuple[int, int]:
    """(rows, cols) for `n` cells. `auto` is the squarest fit that holds them all."""
    spec = (spec or "auto").strip().lower()
    if spec and spec != "auto":
        try:
            r, c = (int(p) for p in spec.replace("x", " ").split())
            if r > 0 and c > 0 and r * c >= n:
                return r, c
        except Exception:  # noqa: BLE001 - a malformed grid degrades to auto, see below
            pass
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    return rows, cols


def compose(rgb: Sequence[Any], steps: Optional[Sequence[int]], *,
            cell_px: int = 96, grid: str = "auto",
            label_frames: bool = True) -> Tuple[bytes, dict]:
    """Tile `rgb` into one sheet. Returns `(png_bytes, layout)`.

    `layout` is small, JSON-safe and describes what a reader is looking at:
    grid shape, reading order, cell size and the step index of every cell. It is
    what the manifest is rendered from and what the judgment record stores, so
    the prompt and the artifact cannot drift apart -- they are the same dict.
    """
    frames = [np.asarray(f) for f in rgb]
    n = len(frames)
    rows, cols = _grid_shape(n, grid)
    src_h, src_w = frames[0].shape[:2]
    cw = max(8, int(cell_px))
    ch = max(8, int(round(cw * src_h / src_w)))
    sheet = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    scale = max(1, cw // 32)
    placed: List[Optional[int]] = []
    for k in range(rows * cols):
        if k >= n:
            placed.append(None)
            continue
        cell = _resize_nearest(frames[k][..., :3].astype(np.uint8), cw, ch).copy()
        step = None
        if steps is not None and k < len(steps):
            step = int(steps[k])
        if label_frames and step is not None:
            _draw_label(cell, str(step), scale)
        r, c = divmod(k, cols)
        sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
        placed.append(step)
    layout = {
        "grid": f"{rows}x{cols}", "order": "row-major",
        "cell_px": [cw, ch], "sheet_px": [cols * cw, rows * ch],
        "n_cells": n, "steps": [s for s in placed if s is not None],
        # What was actually DRAWN, not what was asked for: with no step
        # indices `_draw_label` never runs, and `manifest` states this flag
        # to the judge -- claiming burned labels that are not in the pixels
        # would be a false statement about what the judge was shown.
        "labelled": bool(label_frames) and any(s is not None for s in placed),
    }
    return encode_png(sheet), layout


def manifest(layout: dict, *, episode_len: Optional[int] = None,
             fps: Optional[float] = None, truncated: Optional[bool] = None) -> str:
    """The text that travels WITH the sheet. States what is shown and what is not.

    The last line is not decoration. A judge shown 64 of 300 frames that is not
    told the other 236 exist will reason as though it watched the episode, and
    the whole failure this delivery format risks is a confident verdict about an
    instant nobody sampled.
    """
    steps = layout.get("steps") or []
    cw, ch = (layout.get("cell_px") or [0, 0])[:2]
    lines = [
        f"The image is a contact sheet: {layout.get('grid')} cells, "
        f"{layout.get('order')}, each {cw}x{ch} px.",
    ]
    # Every claim below is conditional on what was actually done: a manifest
    # that describes burned labels under `label_frames: false`, or episode
    # steps nobody recorded, is a false statement handed to the judge -- and
    # to every later human audit of the judgment.
    if layout.get("labelled"):
        lines.append(
            "Each cell is one frame of a single episode. The number burned "
            "into a cell's top-left corner is that frame's EPISODE STEP index"
            + (" (also listed below)." if steps else "."))
    elif steps:
        lines.append(
            "Each cell is one frame of a single episode. Cells are NOT "
            "labelled; the list below gives each cell's EPISODE STEP index "
            "in reading order.")
    else:
        lines.append(
            "Each cell is one frame of a single episode, in temporal order. "
            "Per-cell episode step indices were not recorded.")
    if steps:
        lines.append("Steps shown, in reading order: " + ", ".join(str(s) for s in steps) + ".")
    if episode_len:
        lines.append(f"The episode is {episode_len} steps long"
                     + (f" at {fps:g} fps." if fps else "."))
    if truncated is not None:
        lines.append("The episode ended early (terminated), it was not truncated at a cap."
                     if truncated is False else
                     "The episode ran to its step cap.")
    if steps and episode_len and len(steps) < episode_len:
        lines.append(f"Frames BETWEEN these steps exist and are not shown: you are seeing "
                     f"{len(steps)} of {episode_len} steps. Do not assume nothing happened "
                     f"in the gaps; if a judgment turns on an instant you cannot see, say so.")
    if steps:
        lines.append("Refer to any frame by its episode step index, never by its "
                     "position in the grid.")
    else:
        lines.append("Refer to any frame by its 1-based position in reading order.")
    return "\n".join(lines)


def sheet_cells(prov: Optional[dict]) -> Optional[int]:
    """How many frames the ONE delivered image tiles, or None when the frames
    went individually (`even`, or a composing policy that degraded).

    For the prompts: a judge handed a 4x5 grid must be told it is a grid and
    what the burned numbers mean, in the same sentence that counts the
    frames -- "1 evenly-spaced frames are attached" would describe a sheet as a
    frame. `evaluation._vlm_subtask_score` includes the manifest;
    `preferences._render` and `phases.run_alignment_rate` read these two."""
    comp = (prov or {}).get("composed") or {}
    n = comp.get("n_cells")
    return int(n) if comp.get("sheet_sha256") and n else None


def sheet_manifest(prov: Optional[dict]) -> str:
    """The composed sheet's manifest text, `""` when nothing was composed."""
    return str(((prov or {}).get("composed") or {}).get("manifest") or "").strip()


@register("frame_policy", "even")
def frame_policy_even(ctx: Any, png: Sequence[bytes], rgb: Optional[Sequence[Any]],
                      steps: Optional[Sequence[int]],
                      views: Optional[dict] = None) -> Tuple[List[bytes], str, dict]:
    """One image per frame. A strict identity -- `png` is returned untouched.

    Deliberately does not look at `rgb`, `steps` or `views`: this is the path every
    published config takes, and it must stay exactly what it was. A multi-view
    frame is still one image per instant here; `multiview.describe` tells the judge
    what its panels are, from the prompt, not from this policy.
    """
    return list(png), "", {}


@register("frame_policy", "contact_sheet")
def frame_policy_contact_sheet(ctx: Any, png: Sequence[bytes],
                               rgb: Optional[Sequence[Any]],
                               steps: Optional[Sequence[int]],
                               views: Optional[dict] = None
                               ) -> Tuple[List[bytes], str, dict]:
    """One tiled image plus a manifest.

    `views` is the panel layout of a multi-view frame (`output.video.n_views > 1`),
    passed by `observability._deliver` only when there is one. `cell_px` is a
    per-PANEL width, so a three-panel frame gets a cell three panels wide: the
    measured floor of 96 px is a floor on what one camera's view of the scene
    needs, and dividing it three ways would blind every panel at once.

    DEGRADES TO `even` RATHER THAN FAILING, and records why. Two cases reach
    here without usable arrays: frames read off disk as PNG bytes somebody else
    wrote (`sample_frames`' first source keeps no arrays), and a blind judgment
    with no frames at all. Neither is a reason to lose a judgment -- the repo's
    convention is that a degrade is recorded and the number is still produced --
    and composing would need a PNG *decoder*, which is a dependency this path
    does not have and should not acquire to salvage the rarer source.
    """
    if not png:
        return [], "", {"frame_policy": "contact_sheet", "degraded": "no frames"}
    if not rgb or len(rgb) != len(png):
        return list(png), "", {"frame_policy": "contact_sheet",
                               "degraded": "frames have no arrays in this process"}
    # No usable step indices is the same class of degrade as no arrays: the
    # whole addressing scheme -- burned labels, the manifest's step list, a
    # judge naming "frame 137" and a reader resolving it -- keys on the
    # EPISODE STEP, and a sheet without them would ship a manifest describing
    # an index that does not exist.
    if steps is None or len(steps) != len(png):
        return list(png), "", {"frame_policy": "contact_sheet",
                               "degraded": "no per-frame step indices in "
                                           "this process"}
    cfg = getattr(ctx, "cfg", None)

    def _get(key: str, default: Any) -> Any:
        try:
            v = cfg[key]
        except Exception:  # noqa: BLE001 - a missing key is a default, not a failure
            return default
        return default if v is None else v

    n_panels = 1
    try:
        n_panels = max(1, int((views or {}).get("n_views") or 1))
    except (TypeError, ValueError):
        n_panels = 1
    sheet, layout = compose(
        rgb, steps,
        cell_px=int(_get("evaluate.vlm.contact_sheet.cell_px", 96)) * n_panels,
        grid=str(_get("evaluate.vlm.contact_sheet.grid", "auto")),
        label_frames=bool(_get("evaluate.vlm.contact_sheet.label_frames", True)),
    )
    layout["frame_policy"] = "contact_sheet"
    text = manifest(layout)
    if n_panels > 1:
        layout["panels_per_cell"] = n_panels
        text += (f"\nEach cell is itself {n_panels} viewpoints of one instant placed "
                 "side by side; the viewpoint list below applies within every cell.")
    return [sheet], text, layout
