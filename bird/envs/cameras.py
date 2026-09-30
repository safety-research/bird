"""`MujocoViews` -- extra viewpoints for the MuJoCo tiers, one helper they all share.

`EnvAdapter.render(state)` shows ONE camera, fixed at construction: Meta-World's
`corner`, HumanoidBench's `cam_default` (`trackcom` on the pelvis), and so on. Each of
those was chosen and measured (`metaworld.MetaWorld.camera_name`, `humanoid.py`'s render
notes), and each has a known blind spot the task spec's `judge.camera.note` records --
a tracking camera hides absolute travel, a single third-person corner hides whatever
the arm is in front of. `judge.extra_views` is where a spec names the viewpoints that
cover those blind spots, and this class is how the three MuJoCo adapters render them.

Two kinds of view, told apart by `View.pose`:

  * a MODEL camera -- `pose` is None and `View.name` is a `<camera name=...>` the MJCF
    defines (`corner2`, `topview`, `cam_hurdle`, ...). Its pose comes out of `data`
    after `mj_forward`, so a `trackcom` camera follows the body it tracks with no work
    here.
  * a POSED camera -- `pose` gives `{track_body | lookat, distance, azimuth, elevation}`
    and a FREE `MjvCamera` is built from it, aimed at `lookat`, or at the named body's
    subtree centre of mass read off `data` for THIS frame. This is how a humanoid gets
    a top-down or front-on view that follows it, which the asset never defined, without
    editing a vendored XML.

    NOT `mjCAMERA_TRACKING`, and the reason is measured: MuJoCo's tracking camera
    SMOOTHS its look-at point from the previous update, so a camera built fresh for
    one frame stays aimed at the world origin -- every "tracking" frame is then a
    view of the robot's feet, and a robot 4 m down the course is out of shot. A
    smoothed camera would also make
    `render_view` depend on what was rendered before it, which `render` must never
    do. A free camera whose look-at is set from `data` each frame is a pure function
    of the state, exactly as the model's own `trackcom` cameras are.

ORIENTATION IS CORRECTED HERE, BY MEASUREMENT, NOT BY NAME. Meta-World's `corner` is
defined upside down (`metaworld.MetaWorld.render` derives it from the XML's `xyaxes`)
and so are `corner2`..`corner4` -- measured: all four have a camera up-vector whose
world z is negative -- while `topview`, `behindGripper` and `gripperPOV` do not.
The rule is the general one: a camera's image-up axis is the second column of its
rotation matrix, and if that points into the floor the frame is rotated by 180 degrees
(`[::-1, ::-1]` -- a rotation, never a mirror, for the reason `MetaWorld.render` gives:
a mirror would make "is the puck left of the goal" answerable backwards). A posed
camera is built upright by MuJoCo and is never rotated.

THE CONTEXT IS CREATED LAZILY, AND NOT THROUGH `mujoco.Renderer`. It is a second GL
context per adapter, and GL context creation is a call that can hang on a headless
node (the reason `output.video.timeout_s` exists), so it must not happen at
construction for the many runs that record one view. It happens on the first
`render`, which only `output.video.n_views > 1` reaches, and it is armed by the same
watchdog as the primary render because it runs inside the same `record_rollouts` /
`sample_frames` call.

Not `mujoco.Renderer`, because that class binds its GL backend when `mujoco` is FIRST
IMPORTED (`mujoco.gl_context` reads `MUJOCO_GL` at import), and the adapters set
`MUJOCO_GL` inside their own `__init__`, deliberately after nothing has imported the
simulator (`metaworld.py`'s docstring). Anything that imports `mujoco` earlier -- the
test suite's `importorskip("metaworld")`, a notebook -- leaves `mujoco.Renderer` on GLFW,
which needs an X display and fails on a headless machine, while the
primary render keeps working because gymnasium picks ITS backend at construction. So
this class does what `mujoco.Renderer` does, with the backend read at construction:
`mujoco.osmesa` / `mujoco.egl` / `mujoco.glfw` by `MUJOCO_GL`, an `MjrContext` on the
model, `mjv_updateScene` + `mjr_render` + `mjr_readPixels`, rows flipped.

Deliberately NOT a segmentation renderer and NOT MSAA-free: this renders what a judge
is shown. A segmentation pass (visible pixels per camera by id colour) would disable
`offsamples`; a judge-facing frame keeps the model's anti-aliasing.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Union

import numpy as np

from .base import View

log = logging.getLogger(__name__)

#: `pose` keys a spec may carry. Anything else is refused at load (`tasks._check`), so
#: a typo (`azimut`) cannot silently leave a camera at MuJoCo's default pose.
POSE_KEYS = ("track_body", "lookat", "distance", "azimuth", "elevation")


class MujocoViews:
    """Render `View`s off a live `(model, data)` pair. See the module docstring."""

    def __init__(self, mujoco_module: Any, model: Any, data: Any,
                 height: int, width: int) -> None:
        self._mj = mujoco_module
        self._model = model
        self._data = data
        self._height, self._width = int(height), int(width)
        self._renderer: Optional[Any] = None

    # -- the model's cameras -------------------------------------------------

    def camera_names(self) -> List[str]:
        mj = self._mj
        return [mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_CAMERA, i) or f"#{i}"
                for i in range(int(self._model.ncam))]

    def camera_id(self, name: str) -> int:
        mj = self._mj
        return int(mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_CAMERA, str(name)))

    def inverted(self, camera_id: int) -> bool:
        """Does this model camera's image-up axis point into the floor?

        Read off `data.cam_xmat` AFTER `mj_forward`, so a tracking camera is judged in
        its live pose. The second column of the 3x3 is the camera's +y (image up) in
        world coordinates; `corner`'s `xyaxes="-1 1 0  -0.2 -0.2 -1"` gives it z < 0.
        """
        mat = np.asarray(self._data.cam_xmat[int(camera_id)], dtype=float).reshape(3, 3)
        return bool(mat[2, 1] < 0.0)

    # -- building a camera for a view ----------------------------------------

    def camera_for(self, view: View) -> Union[int, Any]:
        """A model camera id, or an `MjvCamera` built from `view.pose`."""
        mj = self._mj
        if view.pose is None:
            cid = self.camera_id(view.name)
            if cid < 0:
                raise KeyError(
                    f"view {view.name!r}: the model defines no camera of that name "
                    f"(it has {self.camera_names()}); give the view a `pose` or name "
                    "a camera the asset defines")
            return cid
        pose = view.pose_dict or {}
        unknown = sorted(set(pose) - set(POSE_KEYS))
        if unknown:
            raise KeyError(f"view {view.name!r}: unknown pose key(s) {unknown}; "
                           f"allowed: {list(POSE_KEYS)}")
        track = pose.get("track_body")
        lookat = pose.get("lookat")
        # The loader (`tasks._check_views`) refuses both of these for a spec; a View
        # built in code gets the same answer here rather than a camera aimed at the
        # world origin with nothing to say so.
        if track and lookat is not None:
            raise KeyError(f"view {view.name!r}: pose names both track_body and lookat")
        if not track and lookat is None:
            raise KeyError(f"view {view.name!r}: pose names neither track_body nor lookat; "
                           "distance, azimuth and elevation alone aim at the world origin")
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        if track:
            bid = int(mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_BODY, str(track)))
            if bid < 0:
                raise KeyError(f"view {view.name!r}: track_body {track!r} is not a body "
                               "of the model")
            # The body's subtree centre of mass, in the pose `data` holds NOW: for the
            # pelvis of a humanoid that is the whole robot's centre of mass, which is
            # what the model's own `trackcom` cameras follow.
            cam.lookat[:] = np.asarray(self._data.subtree_com[bid], dtype=float)
        else:
            cam.lookat[:] = np.asarray(lookat, dtype=float).ravel()[:3]
        cam.distance = float(pose.get("distance", 3.0))
        cam.azimuth = float(pose.get("azimuth", 90.0))
        cam.elevation = float(pose.get("elevation", -20.0))
        return cam

    # -- rendering ------------------------------------------------------------

    @staticmethod
    def _gl_context(width: int, height: int) -> Any:
        """A GL context for the backend `MUJOCO_GL` names NOW, not at mujoco's import."""
        backend = (os.environ.get("MUJOCO_GL") or "").lower().strip()
        if backend == "osmesa":
            from mujoco.osmesa import GLContext  # noqa: PLC0415 - backend-specific
        elif backend == "egl":
            from mujoco.egl import GLContext  # noqa: PLC0415 - backend-specific
        else:
            from mujoco.glfw import GLContext  # noqa: PLC0415 - backend-specific
        return GLContext(width, height)

    def _renderer_or_make(self) -> Any:
        """The `(gl, scene, mjr_context, rect, option, perturb)` bundle, built once.

        The offscreen buffer is the MODEL's (`vis.global_.offwidth/offheight`), which
        gymnasium raised to the primary render's size at construction; a request
        beyond it is clipped to it rather than refused, and the frame says so by its
        shape -- `observability._resize_nearest` scales whatever comes back.
        """
        if self._renderer is None:
            mj = self._mj
            width = min(self._width, int(self._model.vis.global_.offwidth))
            height = min(self._height, int(self._model.vis.global_.offheight))
            if (width, height) != (self._width, self._height):
                log.warning("extra views render at %dx%d: the model's offscreen buffer is "
                            "smaller than the %dx%d asked for", width, height,
                            self._width, self._height)
            gl = self._gl_context(width, height)
            gl.make_current()
            scene = mj.MjvScene(self._model, 10000)
            ctx = mj.MjrContext(self._model, mj.mjtFontScale.mjFONTSCALE_150)
            mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, ctx)
            self._renderer = {
                "gl": gl, "scene": scene, "ctx": ctx,
                "rect": mj.MjrRect(0, 0, width, height),
                "opt": mj.MjvOption(), "pert": mj.MjvPerturb(),
                "buf": np.empty((height, width, 3), dtype=np.uint8),
            }
        return self._renderer

    def render(self, view: View) -> np.ndarray:
        """One `(height, width, 3)` uint8 frame of the CURRENT `data` from `view`.

        The caller has already restored the state and run `mj_forward` -- this is a
        pure function of `data`, exactly as the primary `render` is after its restore.
        """
        mj = self._mj
        cam = self.camera_for(view)
        if isinstance(cam, int):
            camera = mj.MjvCamera()
            camera.type = mj.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = cam
        else:
            camera = cam
        r = self._renderer_or_make()
        # Contexts alternate with the primary renderer's inside one recording call,
        # so ours is made current on every frame, as gymnasium's is on its side.
        r["gl"].make_current()
        mj.mjv_updateScene(self._model, self._data, r["opt"], r["pert"], camera,
                           mj.mjtCatBit.mjCAT_ALL, r["scene"])
        mj.mjr_render(r["rect"], r["scene"], r["ctx"])
        mj.mjr_readPixels(r["buf"], None, r["rect"], r["ctx"])
        frame = r["buf"][::-1]                      # readPixels is bottom-up
        if isinstance(cam, int) and self.inverted(cam):
            frame = frame[::-1, ::-1]
        return np.ascontiguousarray(frame, dtype=np.uint8)

    def close(self) -> None:
        r, self._renderer = self._renderer, None
        if r is None:
            return
        for key in ("ctx", "gl"):
            try:
                r[key].free()
            except Exception:  # noqa: BLE001 - a context that will not close is not a failed run
                log.debug("freeing the extra-view %s raised", key, exc_info=True)
