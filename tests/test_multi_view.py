"""`output.video.n_views` -- several viewpoints in one recorded frame, and what it must not move.

The load-bearing claim is the first test's: the default (`n_views: 1`) hands the recorder
and the judge the env's own `render`, untouched, so every published config records the
frame it always has. The second is the claim an ablation over the key needs: turning it
on ADDS a panel to the right and moves no pixel of the primary. Everything after that is
plumbing that must not be able to fail silently -- the layout reaching the journal, the
clip's `render_stats.json`, the judgment record and the judge's prompt; a spec offering
fewer views than asked being recorded rather than refused; the spec loader refusing a
view block that would draw one camera twice.

Not `slow`, deliberately: the frames here are eight-pixel numpy arrays and no GL context
is opened, and the identity claim is one a PR must be able to break (the same argument
`tests/test_limen_observation.py` makes for its gate). `tests/test_metaworld.py` and
`tests/test_humanoid_hand.py` carry the simulator-backed half.
"""
from __future__ import annotations

import copy
import json
import struct

import numpy as np
import pytest

from bird import judgments, multiview, registry, tasks
from bird.artifacts import RunDir
from bird.budget import Budget
from bird.config import ConfigError, load
from bird.context import Context
from bird.envs import cameras
from bird.envs.base import View
from bird.envs.spec import views_of
from bird.observability import (_as_rgb8, _frames_from_disk, _resize_nearest, _view_plan,
                                record_rollouts, sample_frames)
from bird.state import RunState
from bird.types import Candidate, CandidateReport, TrainResult, Trajectory

W, H = 8, 6
PRIMARY = View(name="corner", mode="fixed", note="the fixed third-person camera")
EXTRAS = (View(name="topview", mode="fixed", note="straight down at the table"),
          View(name="behindGripper", mode="tracking", note="behind the gripper"))


class _ViewsEnv:
    """Renders a stored state from a primary and two extra viewpoints, distinguishably.

    Every panel is a function of the state (so two states never hash alike) AND of the
    viewpoint (the view's index lands in the blue channel), so a composite that put the
    wrong camera in a panel, or the same camera twice, fails on the pixels.
    """

    primary_view = PRIMARY
    extra_views = EXTRAS
    name = "views_env"

    def __init__(self, extras=EXTRAS):
        self.extra_views = tuple(extras)
        self.calls = []

    @staticmethod
    def _v(state):
        return int(abs(float(np.ravel(np.asarray(state, dtype=float))[0]))) % 200

    def render(self, state):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:, :, 0] = self._v(state)
        frame[0, 0, 1] = 255
        self.calls.append(("render", None))
        return frame

    def render_view(self, state, view):
        assert view in self.extra_views, view
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:, :, 0] = self._v(state)
        frame[:, :, 2] = 40 * (1 + self.extra_views.index(view))
        self.calls.append(("render_view", view.name))
        return frame


class _NoViewsEnv(_ViewsEnv):
    extra_views = ()

    def __init__(self):
        super().__init__(extras=())


class _VLM:
    """A client honouring `bird/llm/base.py` that records what it was sent."""

    modality = "vlm"

    def __init__(self, reply="Score: 0.5\nReason: it approached."):
        self.reply = reply
        self.calls = []

    def __call__(self, messages, n=1, temperature=None, images=None, tag=""):
        text = messages if isinstance(messages, str) else "\n".join(
            str(m.get("content", "")) for m in messages)
        self.calls.append({"prompt": text, "n_images": len(images or ())})
        return [self.reply] * n


def _traj(n=12):
    return Trajectory(states=[[float(i)] for i in range(n)], rewards=[1.0] * n,
                      success=False, length=n, ret=float(n))


def _report(i=0, n=12):
    cand = Candidate(cand_id=f"c{i:04d}", iteration=0,
                     reward_code="def compute_reward(state, action=None): return 0.0")
    res = TrainResult(cand_id=cand.cand_id, candidate=cand, trajectories=[_traj(n)])
    return CandidateReport(cand_id=cand.cand_id, candidate=cand, result=res, fitness=0.5)


def _ctx(tmp_path, env=None, **over):
    cfg = load("rda", profile="tester", overrides={
        "output.judge_trace.record": "inputs",
        "evaluate.vlm.images_per_query": 4,
        "output.video.timeout_s": None,
        "output.video.width": W,
        "verify.forbidden_symbols": [],
        **over,
    })
    ctx = Context(cfg=cfg, budget=Budget(), rundir=RunDir(tmp_path, "views", "hash0"),
                  env=env if env is not None else _ViewsEnv())
    ctx.generator = ctx.evaluator = _VLM()
    return ctx


def _journal(ctx, stage):
    lines = (ctx.rundir.path / "journal.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines if json.loads(x).get("stage") == stage]


def _png_size(png):
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", png[16:24])


# --------------------------------------------------------------------------
# the default is the single-view frame, untouched
# --------------------------------------------------------------------------

def test_the_default_records_the_envs_own_render_untouched(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.cfg["output.video.n_views"] == 1
    plan = _view_plan(ctx)
    assert plan.render == ctx.env.render, "n_views: 1 must hand the recorder env.render itself"
    assert plan.layout is None and plan.note == ""
    prov = {}
    png = sample_frames(ctx, _traj(), 3, prov=prov)
    assert len(png) == 3 and "views" not in prov and "views_note" not in prov
    assert _png_size(png[0]) == (W, H)
    assert all(c[0] == "render" for c in ctx.env.calls), "no extra view may be rendered"


def test_a_second_view_is_added_to_the_right_and_the_primary_does_not_move(tmp_path):
    ctx = _ctx(tmp_path, **{"output.video.n_views": 2})
    plan = _view_plan(ctx)
    state = np.array([37.0])
    composite = plan.render(state)
    assert plan.layout["n_views"] == 2
    assert composite.shape == (H, 2 * W + multiview.GUTTER_PX, 3)
    assert plan.width == composite.shape[1] == plan.layout["width"]
    single = _resize_nearest(_as_rgb8(ctx.env.render(state)), W)
    assert np.array_equal(composite[:, :W], single), "the primary panel moved"
    gutter = composite[:, W:W + multiview.GUTTER_PX]
    assert not gutter.any(), "the gutter is not black"
    right = _resize_nearest(_as_rgb8(ctx.env.render_view(state, EXTRAS[0])), W)
    assert np.array_equal(composite[:, W + multiview.GUTTER_PX:], right)
    p0, p1 = plan.layout["panels"]
    assert (p0["name"], p0["x0"], p0["x1"]) == ("corner", 0, W)
    assert (p1["name"], p1["x0"], p1["x1"]) == ("topview", W + multiview.GUTTER_PX,
                                                 2 * W + multiview.GUTTER_PX)


def test_three_views_are_three_panels_in_spec_order(tmp_path):
    ctx = _ctx(tmp_path, **{"output.video.n_views": 3})
    plan = _view_plan(ctx)
    frame = plan.render(np.array([5.0]))
    assert [p["name"] for p in plan.layout["panels"]] == ["corner", "topview", "behindGripper"]
    g = multiview.GUTTER_PX
    # the blue channel says which camera drew each panel
    assert frame[1, 1, 2] == 0 and frame[1, W + g + 1, 2] == 40 and frame[1, 2 * (W + g) + 1, 2] == 80


# --------------------------------------------------------------------------
# the layout travels with the frame: journal, render_stats.json, judgment record
# --------------------------------------------------------------------------

def test_the_recorder_writes_the_layout_beside_the_clip_and_in_the_journal(tmp_path):
    ctx = _ctx(tmp_path, **{"output.video.n_views": 2, "output.video.record": "all"})
    report = _report()
    written = record_rollouts(ctx, [report])
    assert len(written) == 1
    clip = written[0].path
    stats = json.loads((clip / "render_stats.json").read_text())
    assert [p["name"] for p in stats["views"]["panels"]] == ["corner", "topview"]
    frame = sorted(clip.glob("frame_*.png"))[0].read_bytes()
    assert _png_size(frame) == (2 * W + multiview.GUTTER_PX, H)
    views = _journal(ctx, "views")
    assert views and views[0]["requested"] == 2 and views[0]["n_views"] == 2
    assert views[0]["panels"] == ["corner", "topview"] and views[0]["note"] == ""
    rec = _journal(ctx, "record")
    assert rec and rec[0]["n_views"] == 2


def test_the_default_journals_no_views_event(tmp_path):
    ctx = _ctx(tmp_path, **{"output.video.record": "all"})
    record_rollouts(ctx, [_report()])
    assert _journal(ctx, "views") == []
    assert _journal(ctx, "record")[0]["n_views"] == 1


def test_frames_read_off_disk_carry_the_layout_to_the_judge(tmp_path):
    ctx = _ctx(tmp_path, **{"output.video.n_views": 2, "output.video.record": "all"})
    report = _report()
    record_rollouts(ctx, [report])
    traj = report.result.trajectories[0]
    assert traj.video_path
    png, rgb, source, lay = _frames_from_disk(traj.video_path, 3)
    assert source == "frames_dir" and len(png) == 3 and lay["n_views"] == 2
    prov = {}
    sample_frames(ctx, traj, 3, prov=prov)
    assert prov["source"] == "frames_dir"
    assert [p["name"] for p in prov["views"]["panels"]] == ["corner", "topview"]


def test_a_single_view_clip_on_disk_reads_as_one_view(tmp_path):
    ctx = _ctx(tmp_path, **{"output.video.record": "all"})
    report = _report()
    record_rollouts(ctx, [report])
    _png, _rgb, source, lay = _frames_from_disk(report.result.trajectories[0].video_path, 2)
    assert source == "frames_dir" and lay is None


def test_the_judgment_record_carries_the_layout(tmp_path):
    ctx = _ctx(tmp_path, **{"output.video.n_views": 2})
    prov = {}
    sample_frames(ctx, _traj(), 3, prov=prov)
    assert prov["views"]["n_views"] == 2 and prov["source"] == "render"
    assert prov["width"] == 2 * W + multiview.GUTTER_PX
    rec = judgments.frame_set(prov and [b"x"], None, views=prov["views"])
    assert rec["views"]["panels"][1]["name"] == "topview"
    assert "views" not in judgments.frame_set([b"x"], None)
    assert "views" not in judgments.frame_set([b"x"], None, views={"n_views": 1})


# --------------------------------------------------------------------------
# a shortfall is recorded, never refused
# --------------------------------------------------------------------------

def test_asking_for_more_views_than_the_spec_offers_records_the_shortfall(tmp_path):
    ctx = _ctx(tmp_path, **{"output.video.n_views": 5, "output.video.record": "all"})
    plan = _view_plan(ctx)
    assert plan.layout["n_views"] == 3 and plan.layout["requested"] == 5
    assert plan.layout["available"] == 3
    assert "offers 2 extra view(s)" in plan.note
    record_rollouts(ctx, [_report()])
    event = _journal(ctx, "views")[0]
    assert event["requested"] == 5 and event["n_views"] == 3 and "offers 2" in event["note"]
    prov = {}
    sample_frames(ctx, _traj(), 2, prov=prov)
    assert "offers 2" in prov["views_note"]


def test_an_env_with_no_extra_views_records_one_view_and_says_so(tmp_path):
    ctx = _ctx(tmp_path, env=_NoViewsEnv(), **{"output.video.n_views": 2,
                                               "output.video.record": "all"})
    plan = _view_plan(ctx)
    assert plan.render == ctx.env.render and plan.layout is None
    assert "offers no extra views" in plan.note
    record_rollouts(ctx, [_report()])
    event = _journal(ctx, "views")[0]
    assert event["n_views"] == 1 and event["panels"] == ["corner"] and "no extra views" in event["note"]


def test_an_adapter_that_only_inherits_render_view_is_treated_as_having_none(tmp_path):
    """Every `EnvAdapter` inherits a CALLABLE `render_view` whose body is a refusal, so
    `callable(...)` is not the test: an adapter that offers views it
    cannot draw must degrade here with a note, not raise NotImplementedError mid-record."""
    from bird.envs.base import EnvAdapter

    class _Inherits(EnvAdapter):
        primary_view = PRIMARY
        extra_views = EXTRAS
        name = "inherits"

        def __init__(self):  # the base constructor wants an action set; none is needed here
            pass

        def render(self, state):
            return _ViewsEnv().render(state)

    ctx = _ctx(tmp_path, env=_Inherits(), **{"output.video.n_views": 2,
                                             "output.video.record": "all"})
    plan = _view_plan(ctx)
    assert plan.layout is None and "implements no render_view" in plan.note
    assert plan.render(np.array([3.0])).shape == (H, W, 3)
    record_rollouts(ctx, [_report()])          # and the whole path degrades, never raises
    assert _journal(ctx, "views")[0]["n_views"] == 1


def test_an_env_without_render_view_is_not_asked_for_one(tmp_path):
    class _Bare:
        primary_view = PRIMARY
        extra_views = EXTRAS
        render = staticmethod(_ViewsEnv().render)
    ctx = _ctx(tmp_path, env=_Bare(), **{"output.video.n_views": 2})
    plan = _view_plan(ctx)
    assert plan.layout is None and "implements no render_view" in plan.note


# --------------------------------------------------------------------------
# the judge is told what the panels are, and nothing extra otherwise
# --------------------------------------------------------------------------

def _score(ctx):
    cand = Candidate(cand_id="c0", iteration=0,
                     reward_code="def compute_reward(state, action=None): return 0.0")
    res = TrainResult(cand_id="c0", candidate=cand, trajectories=[_traj()])
    return registry.get("fitness_source", "vlm_score")(ctx, RunState(), [res])


def test_the_vlm_judge_prompt_describes_the_panels_only_under_multi_view(tmp_path):
    one = _ctx(tmp_path / "one")
    _score(one)
    prompt_one = one.generator.calls[0]["prompt"]
    assert "viewpoint" not in prompt_one and "## Frames" not in prompt_one

    two = _ctx(tmp_path / "two", **{"output.video.n_views": 2})
    _score(two)
    prompt_two = two.generator.calls[0]["prompt"]
    assert "## Frames" in prompt_two and "2 viewpoints" in prompt_two
    assert "`corner`" in prompt_two and "`topview`" in prompt_two
    assert "straight down at the table" in prompt_two, "the spec's note is the judge's description"
    # the recorded query is the prompt that was sent, and its frame set carries the layout
    queries = [r for r in judgments.read(two.rundir) if r["kind"] == "query"]
    assert queries and queries[0]["prompt"] == prompt_two
    frames = [r for r in judgments.read(two.rundir) if r["kind"] == "frames"]
    assert frames and frames[0]["views"]["n_views"] == 2


def test_the_pairwise_side_names_its_panels(tmp_path):
    from bird.components import preferences

    two = _ctx(tmp_path / "two", **{"output.video.n_views": 2})
    text = preferences._render(two, _report(), "LEFT", attach=True)
    assert "2 viewpoints side by side" in text and "`topview`" in text

    one = _ctx(tmp_path / "one")
    text = preferences._render(one, _report(), "LEFT", attach=True)
    assert "viewpoint" not in text


def test_describe_is_empty_for_one_view_and_names_every_panel_otherwise():
    assert multiview.describe(None) == "" and multiview.describe({"n_views": 1}) == ""
    lay = multiview.layout((PRIMARY,) + EXTRAS, 320, requested_n=3)
    text = multiview.describe(lay)
    for v in (PRIMARY,) + EXTRAS:
        assert f"`{v.name}`" in text and v.note in text and f"({v.mode} camera)" in text
    assert "3 viewpoints" in text
    short = multiview.describe(lay, short=True)
    assert "\n" not in short and "`behindGripper`" in short


# --------------------------------------------------------------------------
# the contact sheet scales with the panel count
# --------------------------------------------------------------------------

def test_contact_sheet_cells_are_scaled_by_the_panel_count(tmp_path):
    from bird.components.frames import frame_policy_contact_sheet

    ctx = _ctx(tmp_path, **{"output.video.n_views": 3})
    plan = _view_plan(ctx)
    rgb = [plan.render(np.array([float(i)])) for i in range(4)]
    png = [b"\x89PNG\r\n\x1a\n" + bytes(8)] * 4
    steps = [0, 3, 6, 9]
    images, note, layout = frame_policy_contact_sheet(ctx, png, rgb, steps,
                                                      views=plan.layout)
    assert len(images) == 1 and layout["panels_per_cell"] == 3
    assert layout["cell_px"][0] == 96 * 3
    assert "3 viewpoints of one instant" in note
    images1, note1, layout1 = frame_policy_contact_sheet(ctx, png, rgb, steps)
    assert layout1["cell_px"][0] == 96 and "panels_per_cell" not in layout1


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------

def test_compose_pads_shorter_panels_and_places_gutters():
    a = np.full((4, 3, 3), 10, dtype=np.uint8)
    b = np.full((2, 3, 3), 20, dtype=np.uint8)
    out = multiview.compose([a, b], gutter=2)
    assert out.shape == (4, 8, 3)
    assert np.array_equal(out[:, :3], a)
    assert not out[:, 3:5].any()
    assert (out[:2, 5:] == 20).all() and not out[2:, 5:].any(), "padding is black, at the bottom"
    assert multiview.compose([a]) is not None and multiview.compose([a]).shape == a.shape
    with pytest.raises(ValueError):
        multiview.compose([])


def test_layout_offsets_account_for_the_gutter():
    lay = multiview.layout((PRIMARY,) + EXTRAS, 100, gutter=4)
    assert lay["width"] == 3 * 100 + 2 * 4
    assert [(p["x0"], p["x1"]) for p in lay["panels"]] == [(0, 100), (104, 204), (208, 308)]
    assert multiview.panel_count(lay) == 3 and multiview.panel_count(None) == 1


# --------------------------------------------------------------------------
# the config key and the View type
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [0, -1, "2", 1.5, True])
def test_n_views_must_be_a_positive_integer(bad):
    with pytest.raises(ConfigError):
        load("eureka", profile="tester", overrides={"output.video.n_views": bad})


def test_n_views_may_not_be_null():
    with pytest.raises(ConfigError):
        load("eureka", profile="tester", overrides={"output.video.n_views": None})


def test_requested_reads_the_key_and_defaults_to_one():
    assert multiview.requested(load("eureka", profile="tester")) == 1
    assert multiview.requested(load("eureka", profile="tester",
                                    overrides={"output.video.n_views": 3})) == 3

    class _NoCfg:
        def get(self, key):
            raise KeyError(key)
    assert multiview.requested(_NoCfg()) == 1


def test_view_is_hashable_and_round_trips_its_pose():
    v = View.from_mapping({"name": "top", "mode": "tracking", "note": " down ",
                           "pose": {"track_body": "pelvis", "lookat": [0, 0, 0.5],
                                    "distance": 4}})
    assert hash(v) == hash(View.from_mapping(v.as_record()))
    assert v.pose_dict == {"track_body": "pelvis", "lookat": [0, 0, 0.5], "distance": 4}
    assert v.note == "down"
    assert View(name="x").as_record() == {"name": "x", "mode": "fixed", "note": ""}


def test_pose_keys_are_one_list_in_two_files():
    """`tasks.py` may import nothing from `bird.envs`, so the allow-list is written twice;
    this is what keeps the two copies one fact."""
    assert tuple(tasks.VIEW_POSE_KEYS) == tuple(cameras.POSE_KEYS)


# --------------------------------------------------------------------------
# the spec loader
# --------------------------------------------------------------------------

def _doc():
    spec = tasks.load("drawer_close")
    return copy.deepcopy(dict(spec.raw)), spec.path


def _views(doc):
    return doc["judge"]["extra_views"]


@pytest.mark.parametrize("mutate, expect", [
    (lambda d: _views(d).append({"name": "topview", "mode": "fixed"}), "duplicate view name"),
    (lambda d: _views(d).append({"name": "corner", "mode": "fixed"}), "is the primary camera"),
    (lambda d: _views(d).append({"name": "x", "mode": "orbit"}), "mode must be one of"),
    (lambda d: _views(d).append({"name": "x", "mode": "fixed", "pose": {"azimut": 3}}),
     "unknown pose key"),
    (lambda d: _views(d).append({"name": "x", "mode": "fixed",
                                 "pose": {"track_body": "pelvis", "lookat": [0, 0, 0]}}),
     "both track_body and lookat"),
    (lambda d: _views(d).append({"name": "x", "mode": "fixed", "pose": {"distance": 3}}),
     "neither track_body nor lookat"),
    (lambda d: _views(d).append({"name": "", "mode": "fixed"}), "non-empty `name`"),
    (lambda d: _views(d).append({"name": "x", "mode": "fixed", "pose": {}}), "non-empty mapping"),
    (lambda d: d["judge"].__setitem__("extra_views", {"name": "x"}), "must be a list"),
], ids=["duplicate", "primary", "mode", "pose-key", "track-and-lookat", "neither", "name",
        "empty-pose", "not-a-list"])
def test_the_spec_loader_refuses_a_view_block_that_would_mislead_the_judge(mutate, expect):
    doc, path = _doc()
    mutate(doc)
    with pytest.raises(tasks.TaskSpecError, match=expect):
        tasks._check(doc, path)


def test_a_spec_with_no_extra_views_is_still_valid():
    doc, path = _doc()
    del doc["judge"]["extra_views"]
    tasks._check(doc, path)
    doc["judge"]["extra_views"] = None
    tasks._check(doc, path)


#: MT10 specs whose blind spot no model camera covers, and the POSED view each adds
#: FIRST (so `output.video.n_views: 2` picks it). peg-insert: the hole is on the box's
#: face away from `corner`, so insertion is structurally invisible from the primary and
#: from every fixed corner camera, which leaves the clips unjudgeable.
_MT10_POSED_FIRST = {"peg-insert-side-v3": "hole_face"}


def test_every_metaworld_spec_offers_the_two_measured_views():
    """The top view for planar progress, the behind-gripper camera
    for grasp and contact, ordered per task by which the task needs first. A task in
    `_MT10_POSED_FIRST` additionally leads with the one posed view that covers what no
    model camera can; every other extra view stays a model camera. Runs over all fifty
    (the forty generated specs carry exactly the two model cameras)."""
    from bird.envs.metaworld import _metaworld_specs

    seen_posed = set()
    for task, spec in _metaworld_specs().items():
        primary, extras = views_of(spec)
        assert primary.name == "corner" and primary.mode == "fixed", task
        names = [v.name for v in extras]
        assert {"topview", "behindGripper"} <= set(names), task
        assert all(v.note for v in extras), f"{task}: a view with no note tells the judge nothing"
        posed = [v.name for v in extras if v.pose is not None]
        want = _MT10_POSED_FIRST.get(task)
        assert posed == ([want] if want else []), (
            f"{task}: posed extra views {posed}; only {want!r} is documented for this task")
        if want:
            assert names[0] == want, f"{task}: {want!r} must lead so n_views: 2 records it"
            seen_posed.add(task)
        assert set(names) - set(posed) == {"topview", "behindGripper"}, (
            f"{task}: the model-camera views are exactly the two measured ones")
    assert seen_posed == set(_MT10_POSED_FIRST), (
        "a task named in _MT10_POSED_FIRST is missing from the Meta-World spec index")


# --------------------------------------------------------------------------
# the MuJoCo camera helper, without a GL context
# --------------------------------------------------------------------------

def test_camera_for_builds_the_camera_the_view_describes():
    mujoco = pytest.importorskip("mujoco")
    xml = """<mujoco><worldbody>
      <camera name="upright" pos="0 -3 1" xyaxes="1 0 0 0 0.3 1"/>
      <camera name="flipped" pos="0 -3 1" xyaxes="-1 0 0 0 -0.3 -1"/>
      <body name="pelvis" pos="0 0 1"><joint type="free"/><geom size="0.1" mass="1"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    views = cameras.MujocoViews(mujoco, m, d, 8, 8)
    assert views.camera_names() == ["upright", "flipped"]
    assert views.camera_for(View(name="upright")) == 0
    assert not views.inverted(0) and views.inverted(1), "the up-vector rule"
    with pytest.raises(KeyError, match="no camera of that name"):
        views.camera_for(View(name="nope"))
    cam = views.camera_for(View.from_mapping({"name": "top", "mode": "tracking",
                                              "pose": {"track_body": "pelvis", "distance": 4,
                                                       "azimuth": 90, "elevation": -80}}))
    # A FREE camera aimed at the body's centre of mass for THIS frame, never
    # mjCAMERA_TRACKING: that mode smooths its look-at from the previous update, so a
    # camera built per frame stays aimed at the origin (measured; the docstring).
    assert cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    assert np.allclose(cam.lookat, d.subtree_com[bid]) and d.subtree_com[bid][2] > 0.5
    assert (cam.distance, cam.azimuth, cam.elevation) == (4.0, 90.0, -80.0)
    # and it follows the body: move it, forward, ask again
    d.qpos[0] += 4.0
    mujoco.mj_forward(m, d)
    cam2 = views.camera_for(View.from_mapping({"name": "top", "mode": "tracking",
                                               "pose": {"track_body": "pelvis"}}))
    assert np.isclose(cam2.lookat[0] - cam.lookat[0], 4.0)
    free = views.camera_for(View.from_mapping({"name": "course", "mode": "fixed",
                                               "pose": {"lookat": [6, 0, 1], "distance": 12}}))
    assert free.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert list(free.lookat) == [6.0, 0.0, 1.0]
    with pytest.raises(KeyError, match="unknown pose key"):
        views.camera_for(View.from_mapping({"name": "x", "mode": "fixed", "pose": {"azimut": 1}}))
    with pytest.raises(KeyError, match="not a body"):
        views.camera_for(View.from_mapping({"name": "x", "mode": "tracking",
                                            "pose": {"track_body": "torso"}}))
    with pytest.raises(KeyError, match="neither track_body nor lookat"):
        views.camera_for(View.from_mapping({"name": "x", "mode": "fixed",
                                            "pose": {"distance": 3}}))
    with pytest.raises(KeyError, match="both track_body and lookat"):
        views.camera_for(View.from_mapping({"name": "x", "mode": "tracking",
                                            "pose": {"track_body": "pelvis", "lookat": [0, 0, 1]}}))
    assert views._renderer is None, "no GL context may be opened before the first render"
