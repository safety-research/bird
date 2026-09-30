"""`evaluate.vlm.frame_policy: contact_sheet` -- the delivery format, exercised.

The point of these tests is that the composing path runs with NO model involved.
A frame policy only a real VLM exercises is a policy nothing tests, and this one
sits on every judgment: a silent regression here changes what every judge saw
without changing a single number until the scores drift.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from bird.components.frames import (_grid_shape, compose, manifest,
                                    frame_policy_contact_sheet, frame_policy_even)


class _Ctx:
    """The two attributes the policies actually touch."""
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else {}


def _frames(n=16, h=240, w=320):
    return [np.full((h, w, 3), (i * 7 % 256, 40, 90), dtype=np.uint8) for i in range(n)]


def _png_size(png):
    """(width, height) out of the IHDR, so the assertion reads the FILE not our dict."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", png[16:24])


# --------------------------------------------------------------------------
# `even` is an identity, and that is the whole compatibility claim
# --------------------------------------------------------------------------

def test_even_returns_the_frames_untouched():
    """Every published method point takes this path; it must not transform anything."""
    png = [b"a", b"b", b"c"]
    out, note, layout = frame_policy_even(_Ctx(), png, _frames(3), [0, 5, 9])
    assert out == png and note == "" and layout == {}


# --------------------------------------------------------------------------
# the sheet
# --------------------------------------------------------------------------

def test_the_encoded_png_matches_the_layout_it_reports():
    """The file's own header, not our arithmetic about it.

    `sheet_px` is what the judgment record stores and what the manifest quotes,
    so a layout that disagrees with the actual image is a record that lies.
    """
    png, layout = compose(_frames(16), list(range(0, 160, 10)), cell_px=96)
    assert list(_png_size(png)) == layout["sheet_px"]
    assert layout["grid"] == "4x4" and layout["n_cells"] == 16


def test_cells_are_row_major_and_carry_episode_steps_not_positions():
    """The addressing scheme the judge and the record must agree on."""
    steps = [0, 3, 17, 42, 99, 100, 150, 299]
    _png, layout = compose(_frames(8), steps, cell_px=64)
    assert layout["order"] == "row-major"
    assert layout["steps"] == steps, "steps must survive composition in reading order"


def test_manifest_and_layout_cannot_disagree_about_the_steps():
    """The drift this format is most exposed to.

    A judge names a frame from the manifest; a reader resolves the named frame
    from the record. If those two lists ever diverge, every judgment built on them is
    wrong about which instant it was looking at, and nothing raises.
    """
    steps = [0, 11, 22, 33]
    _png, layout = compose(_frames(4), steps, cell_px=96)
    text = manifest(layout, episode_len=300, fps=20)
    for s in steps:
        assert str(s) in text
    assert "never by its position" in text


def test_the_manifest_states_that_unshown_frames_exist():
    """A judge told it saw 4 frames of a 300-step episode must not reason as
    though it watched the episode."""
    _png, layout = compose(_frames(4), [0, 1, 2, 3], cell_px=96)
    text = manifest(layout, episode_len=300)
    assert "not shown" in text and "300" in text


@pytest.mark.parametrize("spec,want", [
    ("auto", (4, 4)), ("2x8", (2, 8)), ("8x2", (8, 2)),
    ("nonsense", (4, 4)),        # malformed degrades to auto rather than raising
    ("1x2", (4, 4)),             # too small to hold 16: rejected, degrades to auto
])
def test_grid_shape(spec, want):
    assert _grid_shape(16, spec) == want


def test_a_cell_too_small_to_label_is_left_unlabelled_rather_than_corrupted():
    """The label is drawn into the pixels; there is no room to fail politely in.

    Better an unlabelled cell than a plate covering the frame it annotates.
    """
    png, layout = compose(_frames(4), [0, 1, 2, 3], cell_px=8, label_frames=True)
    assert list(_png_size(png)) == layout["sheet_px"]


# --------------------------------------------------------------------------
# degrading, which is the behaviour under the sources that keep no arrays
# --------------------------------------------------------------------------

def test_no_arrays_degrades_to_individual_frames_and_says_so():
    """`sample_frames`' read-off-disk source hands back bytes and no arrays.

    Composing would need a PNG decoder this path does not have, so it must fall
    back to individual frames -- and record that it did, or the run silently
    used a different delivery format than its config asked for.
    """
    png = [b"x", b"y"]
    out, note, layout = frame_policy_contact_sheet(_Ctx(), png, None, [0, 1])
    assert out == png and note == ""
    assert "degraded" in layout and "arrays" in layout["degraded"]


def test_a_blind_judgment_stays_blind():
    out, note, layout = frame_policy_contact_sheet(_Ctx(), [], None, None)
    assert out == [] and note == "" and layout["degraded"] == "no frames"


def test_config_values_drive_the_sheet():
    cfg = {"evaluate.vlm.contact_sheet.cell_px": 64,
           "evaluate.vlm.contact_sheet.grid": "2x8",
           "evaluate.vlm.contact_sheet.label_frames": False}
    frames = _frames(16)
    out, note, layout = frame_policy_contact_sheet(
        _Ctx(cfg), [b""] * 16, frames, list(range(16)))
    assert len(out) == 1, "a contact sheet is ONE image; that is the point"
    assert layout["grid"] == "2x8" and layout["cell_px"][0] == 64
    assert layout["labelled"] is False and note


def test_the_manifest_never_claims_labels_that_were_not_drawn():
    """Every manifest claim is conditional on what was actually done: a judge
    told numbers are burned into cells that carry none would be reasoning
    from a false statement about its own evidence."""
    _, unlabelled = compose(_frames(4), [0, 5, 9, 13], cell_px=64,
                            label_frames=False)
    text = manifest(unlabelled)
    assert "NOT labelled" in text and "burned" not in text
    assert "EPISODE STEP" in text          # the list below still names steps
    _, labelled = compose(_frames(4), [0, 5, 9, 13], cell_px=64,
                          label_frames=True)
    assert "burned" in manifest(labelled)


def test_no_steps_means_no_step_claims_anywhere():
    """`label_frames: true` with no step indices draws nothing, so `labelled`
    must say what is IN the pixels, and the manifest must not instruct the
    judge to address frames by an index that does not exist."""
    _, layout = compose(_frames(4), None, cell_px=64, label_frames=True)
    assert layout["labelled"] is False and layout["steps"] == []
    text = manifest(layout)
    assert "burned" not in text and "step index, never" not in text
    assert "position in reading order" in text


def test_no_step_indices_degrades_to_individual_frames():
    """The whole addressing scheme keys on the EPISODE STEP; a sheet composed
    without them would ship a manifest describing an index nobody recorded,
    so the policy degrades to `even` and records why."""
    frames = _frames(4)
    out, note, layout = frame_policy_contact_sheet(_Ctx(), [b""] * 4, frames, None)
    assert out == [b""] * 4 and note == ""
    assert "step indices" in layout["degraded"]
    out, _, layout = frame_policy_contact_sheet(_Ctx(), [b""] * 4, frames, [0, 1])
    assert len(out) == 4 and "step indices" in layout["degraded"]


# --------------------------------------------------------------------------
# end to end through `observability.sample_frames`
#
# Imported rather than rebuilt, for the reason the resume tests import the
# parallelism tests' scrubber: two harnesses that disagree about what a judge
# context is would let one of them pass while the real path is broken.
# --------------------------------------------------------------------------

from test_vlm_sees_frames import _ctx, _traj  # noqa: E402


def test_even_sends_one_image_per_frame_and_records_no_composition():
    """The default path, asserted rather than assumed."""
    from bird.observability import sample_frames
    ctx = _ctx(**{"evaluate.vlm.frame_policy": "even"})
    prov: dict = {}
    frames = sample_frames(ctx, _traj(), 8, prov=prov)
    assert len(frames) == 8
    assert "composed" not in prov


def test_contact_sheet_sends_one_image_and_records_what_it_composed():
    """One image out, and the record says which steps went into it.

    This is the whole delivery contract in one assertion: the judge receives a
    single sheet, the manifest names the steps, and `prov["composed"]` carries
    the same steps plus a hash of the bytes that actually crossed the wire --
    so a later reader can tell what was shown without re-deriving a stride.
    """
    from bird.observability import sample_frames
    ctx = _ctx(**{"evaluate.vlm.frame_policy": "contact_sheet",
                  "evaluate.vlm.contact_sheet.cell_px": 96})
    prov: dict = {}
    frames = sample_frames(ctx, _traj(), 8, prov=prov)
    assert len(frames) == 1, "a contact sheet is one image"
    comp = prov.get("composed") or {}
    assert comp.get("frame_policy") == "contact_sheet"
    assert comp.get("n_cells") == 8 and comp.get("grid") == "3x3"
    assert len(comp.get("sheet_sha256") or "") == 16
    assert "EPISODE STEP" in (comp.get("manifest") or "")
    # the per-frame record survives composition: the judge's frames resolve
    # through it, never through a re-derived stride
    assert prov.get("steps"), "individual frame steps must still be recorded"
    assert comp["steps"] == list(prov["steps"])


def test_a_policy_that_raises_degrades_and_the_record_says_so(monkeypatch):
    """"Crashed and fell back" must be auditable as itself: a config naming a
    policy beside a record with no `composed` entry is otherwise
    indistinguishable from a policy that was never invoked."""
    from bird import observability, registry

    def _boom(ctx, png, rgb, steps):
        raise RuntimeError("tiling exploded")

    # ctx FIRST: building it resolves the env through the real registry.
    ctx = _ctx(**{"evaluate.vlm.frame_policy": "contact_sheet"})
    monkeypatch.setattr(registry, "get", lambda kind, name: _boom)
    prov: dict = {}
    frames = observability.sample_frames(ctx, _traj(), 4, prov=prov)
    assert len(frames) == 4, "degrades to one image per frame"
    comp = prov.get("composed") or {}
    assert comp.get("frame_policy") == "contact_sheet"
    assert "RuntimeError" in (comp.get("degraded") or "")


# --------------------------------------------------------------------------
# what `output.judge_trace.record: inputs+frames` actually stores
# --------------------------------------------------------------------------

class _RunDir:
    def __init__(self, path):
        self.path = path


def _pixctx(tmp_path, record):
    return _Ctx({"output.judge_trace.record": record}), _RunDir(tmp_path)


def test_the_composed_sheet_is_stored_beside_its_source_frames(tmp_path):
    """`inputs+frames` must hold the judge's INPUT, not a reconstruction of it.

    The frames under `frames/<id>/` are what the sheet was built from. Under a
    composing policy they are not what crossed the wire, and every question the
    record exists to answer afterwards -- was that marker legible at this cell
    size, did the judge misread a cell -- is about the sheet.
    """
    from bird import judgments
    ctx, rundir = _pixctx(tmp_path, "inputs+frames")
    ctx.rundir = rundir
    png, _layout = compose(_frames(4), [0, 1, 2, 3], cell_px=64)
    judgments.note_composed(ctx, "abc123", png)
    out = tmp_path / "judgments" / "frames" / "abc123" / "sheet.png"
    assert out.is_file() and out.read_bytes() == png
    assert not list(out.parent.glob(".*tmp")), "no temp file may survive"


def test_the_sheet_is_not_stored_when_the_run_keeps_no_pixels(tmp_path):
    """`inputs` keeps prose and floats; it must not start writing PNGs."""
    from bird import judgments
    ctx, rundir = _pixctx(tmp_path, "inputs")
    ctx.rundir = rundir
    png, _layout = compose(_frames(4), [0, 1, 2, 3], cell_px=64)
    judgments.note_composed(ctx, "abc123", png)
    assert not (tmp_path / "judgments").exists()


def test_the_sheet_is_named_so_it_cannot_be_read_as_a_cell(tmp_path):
    """`sheet.png`, never `frame_NNNN.png`: a reader listing the set's frames
    must not pick up the composite as one of its own cells."""
    from bird import judgments
    ctx, rundir = _pixctx(tmp_path, "inputs+frames")
    ctx.rundir = rundir
    png, _layout = compose(_frames(4), [0, 1, 2, 3], cell_px=64)
    judgments.note_composed(ctx, "zz", png)
    names = sorted(p.name for p in (tmp_path / "judgments" / "frames" / "zz").iterdir())
    assert names == ["sheet.png"]
