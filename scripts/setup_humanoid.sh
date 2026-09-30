#!/usr/bin/env bash
# Build the venv the `h1hand_*` HumanoidBench tier runs in (bird/envs/humanoid.py,
# bird/envs/humanoid_hand.py).
#
#   bash scripts/setup_humanoid.sh                        # .venv-humanoid, clone in .forks/
#   bash scripts/setup_humanoid.sh --venv DIR --clones DIR
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_humanoid.sh
#                                                         # CUDA torch, for GPU training
#
# Idempotent: a clone already at the pinned commit is kept, a venv already holding
# that commit is left alone, and the script ends with the same smoke test either way.
#
# WHY A SECOND VENV. The reason this tier cannot share the repo's venv is a single
# pin: **`mujoco==3.1.6`**, where every other MuJoCo tier here is on 3.3.0.
#
# THAT PIN IS REAL, AND NOT FOR THE REASON `setup.py` SUGGESTS. Upstream's
# `core_requirements` pins eleven packages and `bird/envs/humanoid.py` records which
# of them survived testing. The one that does: HumanoidBench VENDORS a slice of
# dm_control at `humanoid_bench/dmc_deps/`, and `dmc_sizes.py:241` maps
# `'flex_xvert0'` -- an `MjModel` attribute REMOVED after mujoco 3.1.6.
# `humanoid_bench/env.py:9` imports `dmc_deps.dmc_index` at module scope, so the
# failure is at import of the package, not at render time.
#
# **A physics-throughput benchmark is not evidence this works.** `mj_step` never
# touches `dmc_index`, so a green benchmark runs happily on 3.3.0 while every
# `gym.make` still fails. That is why the smoke test below asserts the version, the
# module and one real `gym.make` rather than timing anything
# (bird/envs/humanoid.py, the DEPENDENCIES section).
#
# `gymnasium==0.29.1` is pinned too, and for a NARROWER reason than upstream says:
# import and stepping are measured working on 1.3.0, but `Task.render` calls the
# 0.29-era `MujocoRenderer.render(mode, camera_id, camera_name)` and raises
# TypeError on 1.x. This tier renders, so 0.29.1 it is. A pin is real for a CODE
# PATH, not for a package.
#
# NOT AN `--extra` IN pyproject.toml. `humanoid_bench` is not on PyPI -- it installs
# from a git checkout -- so it cannot be a locked dependency. Its tests are marked
# (`-m humanoid`) and run BY HAND in this venv.
#
# GL IS REQUIRED EVEN FOR RUNS THAT NEVER RENDER: HumanoidBench builds its renderer
# during `gym.make`, so a headless box with no GL library fails at CONSTRUCTION.
# `MUJOCO_GL=disable` is not an escape hatch -- it removes the context the
# constructor asks for. Source the repo's GL helper before anything HB:
#
#   source scripts/setup_gl.sh --env
#   PYTHONPATH=. .venv-humanoid/bin/python -m pytest tests/test_humanoid_hand.py -m humanoid
#
# USE `uv run --no-sync`, NEVER A BARE `uv run`, for anything in this tier. A bare
# `uv run` syncs against `uv.lock` first, and this venv is not described by the lock
# -- the sync uninstalls the simulator. The commands printed at the end use the
# venv's python directly.
# THREE TRAPS, each caught only by an assertion being SPECIFIC. A version check and a
# physics benchmark would pass all of them. Detail at each point of use below.
#
#   1. UPSTREAM SHIPS A PACKAGE A NORMAL INSTALL BREAKS. `setup.py` calls a bare
#      `find_packages()` and `humanoid_bench/dmc_deps/` has no `__init__.py`, so
#      setuptools sees exactly two packages: a wheel built from this tree omits
#      `dmc_deps/` (imported unconditionally at `env.py:9`), `envs/`, and the whole
#      `assets/` tree -- while reporting success. Hence the editable install.
#   2. A clean-checkout guard trips on the install's OWN output (`build/`, which
#      upstream's .gitignore does not cover), so the script would work exactly once.
#   3. A stamp that records only the commit prints "already installed" after the
#      dependency list is edited and skips the install -- success reported over an
#      unchanged, still-broken venv. The stamp hashes the dep lists too.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv-humanoid"
CLONES="$REPO/.forks"
PYTHON_VERSION=3.11

# THE PIN, and why this commit. Upstream is https://github.com/carlosferrazza/humanoid-bench
# and an unpinned build takes whatever `main` is that day.
#
#   * `cb118903` is `main`'s tip (2025-09-18) and is CODE-IDENTICAL to `791405f5`
#     (2025-05-20), the last commit that touched code: `git diff 791405f5..cb118903`
#     is LICENSE, +3 lines, nothing else (verified against the GitHub compare API).
#     Pinning the tip therefore costs nothing in behaviour and lands on the commit a
#     fresh clone of `main` gets.
#   * THE TAGS ARE NOT USABLE AND THIS IS THE ONE THAT MATTERS. `v0.1.0` and
#     `v0.2.0` are lightweight tags that both PREDATE `2906e47d` (2025-05-19,
#     "Lighten dmc deps"), the commit that ADDED `humanoid_bench/dmc_deps/` --
#     `contents/humanoid_bench/dmc_deps?ref=v0.2.0` is a 404. So at either tag the
#     module this tier's mujoco pin exists to protect DOES NOT EXIST, and the
#     package imports real `dm_control` instead. A tag would look like the
#     conservative choice and would silently be a different dependency story.
FORK_URL="https://github.com/carlosferrazza/humanoid-bench"
FORK_COMMIT="cb1189039151c8aadaaa987b442da54383c87fab"
FORK_DIR_NAME="humanoid-bench"

usage() { sed -n '2,6p' "$0"; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
    case "$1" in
        --venv)   VENV="$(cd "$(dirname "$2")" 2>/dev/null && pwd)/$(basename "$2")"; shift 2 ;;
        --clones) CLONES="$(mkdir -p "$2" && cd "$2" && pwd)"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { echo "setup_humanoid: $*" >&2; exit 1; }

command -v uv  >/dev/null 2>&1 || die "uv is not on PATH (curl -LsSf https://astral.sh/uv/install.sh | sh)"
command -v git >/dev/null 2>&1 || die "git is not on PATH"

# Shallow fetch of ONE commit: `git clone --depth 1` can
# only shallow-fetch a branch tip, and a branch tip moves. Fetching by hash pins it.
clone_pinned() {   # clone_pinned DIR URL COMMIT
    local dir="$1" url="$2" commit="$3"
    if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
        # EXCLUDE THE INSTALL'S OWN OUTPUT, or this guard makes the script run
        # exactly once. `pip`/`setuptools` writes `build/` and `*.egg-info/` INTO the
        # source tree, and upstream's `.gitignore` covers `*.egg-info` and
        # `__pycache__/` but NOT `build/` -- so after one install the checkout is
        # dirty by 13 untracked files and every later run would die here. What the
        # guard is for is a SOURCE edit riding into a venv
        # that claims to be pinned; these paths are not that, and excluding them by
        # name keeps the check honest rather than weakening it to "tracked files
        # only", which would miss an untracked `humanoid_bench/patch.py`.
        if [ -n "$(git -C "$dir" status --porcelain --untracked-files=all -- . \
                     ':(exclude)build/' ':(exclude)dist/' ':(exclude)*.egg-info/')" ]; then
            die "$dir is at $commit but has local changes; a pinned upstream must be a clean checkout (git -C $dir status)"
        fi
        echo "  already at $commit: $dir (clean)"
        return
    fi
    if [ -e "$dir" ]; then
        die "$dir exists and is not a clean checkout of $commit; move it aside"
    fi
    mkdir -p "$dir"
    git -C "$dir" init -q
    git -C "$dir" remote add origin "$url"
    # ~390 MB: the repo carries Shadow-Hand meshes. One commit, no history.
    git -C "$dir" fetch -q --depth 1 origin "$commit"
    git -C "$dir" checkout -q FETCH_HEAD
    echo "  fetched $url @ $commit -> $dir"
}

say "1/4  clone: humanoid-bench @ ${FORK_COMMIT:0:8}"
mkdir -p "$CLONES"
FORK_DIR="$CLONES/$FORK_DIR_NAME"
clone_pinned "$FORK_DIR" "$FORK_URL" "$FORK_COMMIT"

say "2/4  venv: $VENV (python $PYTHON_VERSION)"
if [ -x "$VENV/bin/python" ]; then
    echo "  exists"
else
    uv venv -q "$VENV" -p "$PYTHON_VERSION"
    echo "  created"
fi
PY="$VENV/bin/python"

# Refuse to build on top of a 3.3.0 stack. Every other MuJoCo tier here (metaworld,
# gym_mujoco, assistax) is on 3.3.0, and `uv pip install mujoco==3.1.6`
# into one of those venvs would DOWNGRADE it silently and break that tier instead --
# the failure would surface later, somewhere else, as a tier that used to work.
if "$PY" - <<'EOF' 2>/dev/null
import importlib.util, sys
if importlib.util.find_spec("mujoco") is None:
    sys.exit(1)
import mujoco
sys.exit(0 if not str(mujoco.__version__).startswith("3.1.6") else 1)
EOF
then
    die "$VENV already holds mujoco $("$PY" -c 'import mujoco; print(mujoco.__version__)' 2>/dev/null), not 3.1.6 -- that is another tier's venv and this install would downgrade it. Pass --venv to another path"
fi
if "$PY" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("metaworld") else 1)' 2>/dev/null; then
    die "$VENV has metaworld on its sys.path; that is the metaworld tier's venv. Pass --venv to another path"
fi

say "3/4  install"
# WHY EACH ONE IS HERE -- every entry below is a MODULE-SCOPE import on the path from
# `import humanoid_bench`, traced rather than copied from upstream's
# `core_requirements`:
#   humanoid_bench/env.py:4,5   mujoco, gymnasium
#   humanoid_bench/env.py:9     humanoid_bench.dmc_deps.dmc_index  (vendored; the pin)
#   humanoid_bench/env.py:14    dm_control.utils.rewards
#   humanoid_bench/env.py:18 -> wrappers.py:5 -> mjx/flax_to_torch.py:1,3  torch, jax
# torch and jax are imported for `TorchModel`/`TorchPolicy`, which this tier never
# instantiates -- but the import is unconditional, so they are hard requirements of
# `gym.make`, not optional extras.
#
# pyyaml and pytest are BIRD's, not HumanoidBench's, and they are not optional: this tier
# is RUN from this venv (`PYTHONPATH=. .venv-humanoid/bin/python -m pytest
# tests/test_humanoid_hand.py`), so the interpreter must import `bird` and run the suite.
# `bird/config.py` imports yaml at module scope, so without pyyaml even `import bird`
# dies before it starts.
# `numpy<2` because mujoco 3.1.6 and dm_control 1.0.20 predate the numpy 2 ABI.
# anthropic is the reward generator / judge client; imageio-ffmpeg lets runs record mp4.
DEPS=(
    "mujoco==3.1.6" "gymnasium==0.29.1" "numpy<2" "dm_control==1.0.20"
    "jax[cpu]" imageio imageio-ffmpeg
    pyyaml pytest
    anthropic
)
# SB3 SEPARATELY AND `--no-deps`, because its resolve is what would break the pin:
# stable_baselines3 requires `gymnasium>=0.29.1,<1.3` and `numpy>=1.20`, and letting
# pip satisfy those is exactly how a venv wakes up on gymnasium 1.x with `Task.render`
# broken. Installed without deps and then checked by import, the three it actually
# needs at module scope (cloudpickle, pandas, matplotlib) added the same way.
#
# NOT OPTIONAL FOR THIS TIER: `configs/_profiles/full.yaml` and `dev.yaml` both
# set `train.backend: sb3`, so a BIRD search on any h1hand task needs it, and
# `tests/test_humanoid_hand.py` exercises that path. Without it the venv could
# construct environments it could not train in.
SB3_DEPS=( stable_baselines3 cloudpickle pandas matplotlib )
# CPU torch BY DEFAULT: the CPU index is ~200 MB against ~2.5 GB for a CUDA build,
# and constructing and stepping the environments, the tests and a `train.backend:
# sb3` run need no GPU. The paper's HumanoidBench learner (`train.backend: fasttd3`,
# `train.architecture: simba_v2`, `train.hyperparameters.device: cuda`) DOES need
# one, and `bird/config.py` refuses that config with a non-cuda device. For it, point
# this at a CUDA wheel index matching the machine's driver, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash scripts/setup_humanoid.sh
# The stamp below covers the index, so switching it re-installs torch.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"

# THE STAMP COVERS THE DEPENDENCY LIST, NOT JUST THE COMMIT. With `$FORK_COMMIT`
# alone, adding pyyaml to DEPS above and re-running would print "already installed"
# and skip the install entirely -- the script reporting success while the venv stays
# exactly as broken as before. A stamp that cannot see the thing you changed is worse
# than no stamp: it converts an edit into a no-op that looks like a cache hit.
STAMP="$VENV/.bird-humanoid-commit"
WANT="$FORK_COMMIT $(printf '%s\n' "${DEPS[@]}" "${SB3_DEPS[@]}" "$TORCH_INDEX" | sha256sum | cut -c1-12)"
if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$WANT" ] \
   && "$PY" -c 'import humanoid_bench, mujoco, yaml, stable_baselines3' 2>/dev/null; then
    echo "  already installed: humanoid-bench @ ${FORK_COMMIT:0:8}, deps unchanged"
else
    # ONE `uv pip install` for the explicit list, so the resolver sees mujoco==3.1.6
    # as a constraint rather than upgrading it to satisfy dm_control afterwards.
    uv pip install -q --python "$PY" "${DEPS[@]}"
    uv pip install -q --python "$PY" --index-url "$TORCH_INDEX" torch
    uv pip install -q --python "$PY" --no-deps "${SB3_DEPS[@]}"
    # --no-deps: upstream's core_requirements would pull mujoco-mjx, brax and gymnax
    # (MJX training, which this tier does not use) and would re-resolve mujoco.
    #
    # EDITABLE, AND NOT AS A CONVENIENCE -- A NON-EDITABLE INSTALL OF THIS PACKAGE IS
    # BROKEN. `setup.py` uses a bare `find_packages()`, and `humanoid_bench/dmc_deps/`
    # has NO `__init__.py`, so setuptools does not see it: `find_packages()` returns
    # exactly two packages (`humanoid_bench` and `humanoid_bench.mjx`). A wheel built
    # from this tree therefore ships neither `dmc_deps/` -- which `env.py:9` imports
    # unconditionally -- nor `envs/`, nor the `assets/` tree. Measured: the
    # non-editable install produces
    # `ModuleNotFoundError: No module named 'humanoid_bench.dmc_deps'` on the first
    # `import humanoid_bench`.
    #
    # Editable puts the CHECKOUT on sys.path, so every subpackage and asset resolves
    # from the pinned tree. The cost is that a local edit in $FORK_DIR rides into this
    # venv unseen, which is what `clone_pinned` refuses above by requiring a clean
    # checkout at the pin.
    uv pip install -q --python "$PY" --no-deps -e "$FORK_DIR"
    echo "$WANT" > "$STAMP"
    echo "  installed"
fi

say "4/4  smoke"
# Three assertions, and each one is a thing a weaker check would miss:
#   1. the mujoco version, because everything else about this venv is ordinary;
#   2. `dmc_deps.dmc_index` IMPORTS, which is the module the pin protects and the
#      thing a throughput benchmark never touches;
#   3. one real `gym.make` UNDER GL, because the renderer is built during
#      construction -- steps 1 and 2 both pass on a box with no GL at all.
if [ -z "${MUJOCO_GL:-}" ]; then
    echo "  MUJOCO_GL is unset; sourcing scripts/setup_gl.sh --env"
    # shellcheck disable=SC1091
    source "$REPO/scripts/setup_gl.sh" --env || die "setup_gl.sh --env failed; run bash scripts/setup_gl.sh first"
fi
PYTHONPATH="$REPO" "$PY" - <<'EOF'
import mujoco
assert mujoco.__version__ == "3.1.6", f"mujoco {mujoco.__version__}, need exactly 3.1.6"
print(f"  mujoco {mujoco.__version__}")

# The module the pin exists for. On 3.3.0 this raises at import, and nothing that
# only calls mj_step would ever find out.
import humanoid_bench.dmc_deps.dmc_index  # noqa: F401
print("  humanoid_bench.dmc_deps.dmc_index imports")

# sb3 is the one dependency here whose own requirements (`gymnasium>=0.29.1,<1.3`)
# could move the two pins if it were ever installed WITH deps. Asserting the
# versions after importing it is what would catch that, and costs nothing.
import gymnasium
import stable_baselines3  # noqa: F401
assert gymnasium.__version__ == "0.29.1", f"gymnasium {gymnasium.__version__}, need 0.29.1 for Task.render"
print(f"  gymnasium {gymnasium.__version__}, stable_baselines3 {stable_baselines3.__version__}, pins intact")

import os
import gymnasium as gym
import humanoid_bench  # noqa: F401  -- registers the ids
env = gym.make("h1hand-basketball-v0")
obs, _ = env.reset(seed=0)
env.close()
print(f"  gym.make('h1hand-basketball-v0') under MUJOCO_GL={os.environ.get('MUJOCO_GL')!r}: obs {obs.shape}")
EOF
echo
echo "Run the tier from the repo root as:"
echo "  source scripts/setup_gl.sh --env"
echo "  PYTHONPATH=. $VENV/bin/python -m pytest tests/test_humanoid_hand.py -m humanoid"
