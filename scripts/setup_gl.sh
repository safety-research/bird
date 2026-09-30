#!/usr/bin/env bash
# Install a userland OpenGL stack so MuJoCo can render, without root.
#
#   bash scripts/setup_gl.sh          # install into $HOME/.local/opt/gl
#   source scripts/setup_gl.sh --env  # just export the variables
#
#   BIRD_GL_PREFIX=/some/shared/dir/gl bash scripts/setup_gl.sh   # elsewhere
#
# Put the prefix on a filesystem every machine that renders can see (a shared
# mount on a cluster): a prefix the compute node cannot read fails with the SAME
# PyOpenGL error as no prefix at all, so the fix looks like it did not work.
#
# WHY THIS IS NEEDED AT ALL. MuJoCo renders through a real GL context, and a GL
# context comes from a system library that pip cannot install. On a typical GPU
# node without desktop GL, `ldconfig -p` reports:
#
#     libEGL_nvidia.so.0    yes    (the NVIDIA vendor driver)
#     libEGL.so.1           NO     (the vendor-neutral glvnd dispatch)
#     libOSMesa.so          NO     (software rendering)
#
# The vendor driver alone is not loadable: applications link `libEGL.so.1`, and
# glvnd is what routes that to a vendor. Without it MuJoCo's EGL backend dies in
# PyOpenGL with `AttributeError: 'NoneType' object has no attribute
# 'eglQueryString'`, which names neither EGL nor the missing package and sends
# you looking in the wrong place. The OSMesa backend fails the same way with
# `glGetError`. Both are missing-system-library errors wearing a Python
# traceback.
#
# The fix is a conda-forge Mesa in a prefix we own. `micromamba` is a single
# static binary and needs no root, no daemon and no existing conda.
#
# After this runs, EGL resolves in priority order: the NVIDIA vendor JSON that
# is already on the node (`10_nvidia.json`) is tried first and used on GPU
# nodes, and Mesa's software renderer (`50_mesa.json`) is the fallback on
# CPU-only nodes. Rendering therefore works on GPU and CPU-only machines alike.
#
# Measured cost of the fallback: ~0.67 s per 480x480 frame under llvmpipe. That
# is slow for a renderer and irrelevant here -- a run records a few hundred
# frames, not a few hundred thousand.
#
# THREE PIECES OF EXPECTED NOISE, and all three read as faults.
#
#   1. Once, before any frame is rendered:
#
#          Failed to load library ( 'libOpenGL.so.0' )
#
#      A PROBE REPORTING A NEGATIVE, not a fault, and it is not even logged as
#      an error. `OpenGL/platform/egl.py::EGLPlatform.GL` resolves desktop GL by
#      trying TWO names in order -- `for name in ('OpenGL','GL')` -- and
#      `ctypesloader.loadLibrary` tries `libOpenGL.so`, then `.so.9` down to
#      `.so.0`, before giving up on the first name with
#      `_log.info('Failed to load library ( %r )', filename)`. `libOpenGL.so.0`
#      is simply the LAST name it tried for `OpenGL`; the loop then asks for
#      `GL`, and the Mesa prefix this script installs provides `libGL.so.1`, so
#      the property resolves on the second name and nothing downstream is
#      missing. (Even if both failed it would fall back to GLES2/GLES1, which is
#      what MuJoCo's EGL backend actually calls.)
#
#      It reaches the log only because a BIRD job sets its root logger to INFO;
#      the level is `logging.INFO` on the `OpenGL.platform.ctypesloader` logger.
#      Silencing it would mean either installing a desktop-GL dispatch this
#      stack does not use -- a second glvnd library in a prefix whose EGL vendor
#      order is the delicate part -- or muting a
#      third-party logger, which would also mute a real load failure.
#
#   2. and 3. At interpreter SHUTDOWN, two
#      `OpenGL.raw.EGL._errors.EGLError: <exception str() failed>` tracebacks
#      from `OffScreenViewer.__del__` and `GLContext.__del__`. Those fire after
#      the frames are already written, and they are gymnasium tearing down a
#      context whose EGL display is gone.
#
# All three are harmless -- but (1) is near the TOP of the log and (2)/(3)
# are the LAST, so a job that fully succeeded is bracketed by what look like a
# missing library and a crash. Judge success by whether frames were written
# (or run `uv run python3 -m pytest tests -m gl`), not by a clean exit or by
# the absence of these messages.

set -euo pipefail

PREFIX="${BIRD_GL_PREFIX:-$HOME/.local/opt/gl}"
MAMBA_DIR="${BIRD_MAMBA_DIR:-$(dirname "$PREFIX")}"

_bird_gl_export() {
  export LD_LIBRARY_PATH="$PREFIX/lib:${LD_LIBRARY_PATH:-}"
  # Colon-separated, and the system directory comes FIRST so a GPU node picks
  # the NVIDIA vendor and only falls back to Mesa's llvmpipe when it cannot.
  export __EGL_VENDOR_LIBRARY_DIRS="/usr/share/glvnd/egl_vendor.d:$PREFIX/share/glvnd/egl_vendor.d"
  export LIBGL_DRIVERS_PATH="$PREFIX/lib/dri"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
}

if [[ "${1:-}" == "--env" ]]; then
  _bird_gl_export
  return 0 2>/dev/null || exit 0
fi

if [[ ! -f "$PREFIX/lib/libEGL.so.1" ]]; then
  mkdir -p "$MAMBA_DIR"
  if [[ ! -x "$MAMBA_DIR/bin/micromamba" ]]; then
    echo "fetching micromamba into $MAMBA_DIR"
    ( cd "$MAMBA_DIR" && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
        | tar -xj bin/micromamba )
  fi
  echo "installing mesa + glvnd into $PREFIX"
  "$MAMBA_DIR/bin/micromamba" create -y -q -p "$PREFIX" -c conda-forge mesalib libglvnd
else
  echo "GL prefix already present at $PREFIX"
fi

_bird_gl_export

echo "libEGL.so.1  : $(ls "$PREFIX/lib/libEGL.so.1" 2>/dev/null || echo MISSING)"
echo "egl vendors  : $__EGL_VENDOR_LIBRARY_DIRS"
echo
echo "add to a job script:   source scripts/setup_gl.sh --env"
