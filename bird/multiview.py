"""Several viewpoints in one recorded frame -- `output.video.n_views`.

WHAT THE KEY DOES. Every adapter's `render(state)` shows one camera (`primary_view`),
and every published config records exactly that: `n_views: 1`, the default, is
byte-for-byte the single-camera recording. `n_views: N` asks for the
first `N - 1` of the task spec's `judge.extra_views` BESIDE the primary, in one frame:
panels side by side, left to right, primary first, separated by a thin black gutter.
Each panel is `output.video.width` wide, so the primary panel of a multi-view frame is
pixel-identical to the single-view frame -- turning the key on ADDS pixels and moves
none, which is the property an ablation over it needs: the two arms differ only in
what the judge is additionally shown.

WHY ONE FRAME AND NOT N IMAGES. A frame is an instant. `judgments.frame_set` records
one episode step per frame, `save_trajectory_trace` pairs one trace point with one
frame, the contact sheet labels one step per cell, `clip_stats` measures one clip and
a clip is one directory of `frame_NNNN.png`. N images per instant would put
every one of those on a different footing; N panels in one image leaves all of them
exactly as they are. What changes is the frame's width, which is why the layout --
which panel is which camera, and where it starts and ends in pixels -- travels with
the frame (`render_stats.json`, the `record` journal line, the judgment record) and
is stated to the judge (`describe`).

WHAT THE JUDGE IS TOLD. `describe(layout)` renders the spec's own `mode` and `note`
for each panel, in reading order. A judge shown three panels and told nothing would
have to guess which is which, and a wrong guess is indistinguishable from a wrong
judgment. Under `n_views: 1` it renders the empty string, so every prompt that
includes it is byte-identical to the single-view prompt.

The cost is linear: Claude prices an image by its pixel count, so three 320 px panels
cost three 320 px frames. Panels at 480 px x 4 would exceed the 1568 px long edge the
API downscales at; keep `width x n_views` under it or the extra views arrive smaller
than the primary.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .envs.base import EnvAdapter, View

#: The config key, in one place. `select` reads it; the coherence check names it.
CONFIG_KEY = "output.video.n_views"

#: Black gutter between panels. Wide enough to read as a boundary at 320 px, narrow
#: enough that three panels still fit the API's long edge at 480 px.
GUTTER_PX = 4


def requested(cfg: Any) -> int:
    """`output.video.n_views` as an int, `1` when the key is absent or null."""
    try:
        value = cfg.get(CONFIG_KEY)
    except Exception:  # noqa: BLE001 - a config without the key is the single-view path
        return 1
    try:
        return max(1, int(value or 1))
    except (TypeError, ValueError):
        return 1


def select(env: Any, cfg: Any) -> Tuple[Tuple[View, ...], str]:
    """The views this run will draw, primary first, and a shortfall note or `""`.

    A shortfall is never an error: a sweep sets one `n_views` across tasks whose specs
    offer different numbers of extra views, and refusing would fail the sweep on the
    task with the fewest. What it must never be is silent -- the note is journalled by
    the recorder and carried in the judgment record, so a run that asked for three
    views and got one reads as that.
    """
    primary = getattr(env, "primary_view", None) or View(name="primary")
    n = requested(cfg)
    if n <= 1:
        return (primary,), ""
    extras = tuple(getattr(env, "extra_views", ()) or ())
    if not extras:
        return (primary,), (f"{CONFIG_KEY}={n} but the task spec offers no extra views; "
                            "recording the primary view only")
    # "Implements" means OVERRIDES: every `EnvAdapter` inherits a callable
    # `render_view` whose body is a refusal, so a `callable(...)` test would pass
    # an adapter that offers views it cannot draw and raise at render time
    # instead of degrading here with a note. A bound method
    # exposes the function it wraps as `__func__`; anything else duck-typed
    # (a staticmethod, an instance attribute) is taken at its word.
    method = getattr(env, "render_view", None)
    if not callable(method) or getattr(method, "__func__", method) is EnvAdapter.render_view:
        return (primary,), (f"{CONFIG_KEY}={n} but {type(env).__name__} implements no "
                            "render_view; recording the primary view only")
    chosen = extras[:n - 1]
    note = ""
    if len(chosen) < n - 1:
        note = (f"{CONFIG_KEY}={n} but the task spec offers {len(extras)} extra "
                f"view(s); recording {1 + len(chosen)}")
    return (primary,) + chosen, note


def layout(views: Sequence[View], panel_width: int, *, requested_n: int = 1,
           available: Optional[int] = None, gutter: int = GUTTER_PX) -> Dict[str, Any]:
    """Where each panel sits in the composite, and what it is. JSON-safe."""
    w, g = int(panel_width), int(gutter)
    panels = []
    for i, v in enumerate(views):
        x0 = i * (w + g)
        panels.append({**v.as_record(), "x0": x0, "x1": x0 + w})
    n = len(panels)
    return {
        "n_views": n,
        "panel_width": w,
        "gutter_px": g,
        "width": n * w + max(0, n - 1) * g,
        "requested": int(requested_n),
        "available": int(available if available is not None else n),
        "panels": panels,
    }


def compose(panels: Sequence[np.ndarray], gutter: int = GUTTER_PX) -> np.ndarray:
    """Panels side by side with black gutters; shorter panels are padded at the bottom.

    Padding rather than resizing, because a panel's pixels are the evidence: a
    humanoid model camera and a posed top-down camera can legitimately differ in
    aspect, and stretching one to match the other would move every pixel the judge
    is asked to read.
    """
    frames = [np.ascontiguousarray(np.asarray(p, dtype=np.uint8)) for p in panels]
    if not frames:
        raise ValueError("compose() needs at least one panel")
    if len(frames) == 1:
        return frames[0]
    height = max(f.shape[0] for f in frames)
    g = max(0, int(gutter))
    parts: List[np.ndarray] = []
    for i, f in enumerate(frames):
        if f.shape[0] < height:
            pad = np.zeros((height - f.shape[0], f.shape[1], 3), dtype=np.uint8)
            f = np.concatenate([f, pad], axis=0)
        if i and g:
            parts.append(np.zeros((height, g, 3), dtype=np.uint8))
        parts.append(f)
    return np.ascontiguousarray(np.concatenate(parts, axis=1))


def panel_count(lay: Optional[Dict[str, Any]]) -> int:
    """How many panels a frame carries under `lay`; `1` for no layout."""
    if not lay:
        return 1
    try:
        return max(1, int(lay.get("n_views") or len(lay.get("panels") or ()) or 1))
    except (TypeError, ValueError):
        return 1


def _panel_phrase(p: Dict[str, Any]) -> str:
    mode = str(p.get("mode") or "").strip()
    note = " ".join(str(p.get("note") or "").split())
    bits = [f"`{p.get('name')}`"]
    if mode:
        bits.append(f"({mode} camera)")
    head = " ".join(bits)
    return f"{head}: {note}" if note else head


def describe(lay: Optional[Dict[str, Any]], short: bool = False) -> str:
    """The prompt text that says what the panels are. `""` for a single-view frame."""
    if panel_count(lay) <= 1:
        return ""
    panels = list(lay.get("panels") or [])
    if short:
        names = ", ".join(f"`{p.get('name')}`" for p in panels)
        return (f"each attached frame shows the same instant from {len(panels)} "
                f"viewpoints side by side, left to right: {names}")
    lines = [
        f"Each attached image shows ONE instant of the episode from {len(panels)} "
        "viewpoints placed side by side and separated by a thin black gutter. All "
        "panels show the same simulator state; read them together. Left to right:"]
    for i, p in enumerate(panels, start=1):
        lines.append(f"  {i}. {_panel_phrase(p)}")
    return "\n".join(lines)
