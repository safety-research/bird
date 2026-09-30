#!/usr/bin/env bash
# The JAX tier's own venv: upstream Assistax (`upstream_assistax_*`,
# `train.backend: assistax_ppo`) and the offline `jax_toy`.
#
# ITS OWN VENV, not an extra in the shared one: `assistax` is a GIT-ONLY
# dependency at a pinned commit, so no `uv.lock` can hold it, and `jax[cuda13]`
# pulls ~3 GB of CUDA wheels that every other tier would carry for nothing. A
# shared venv would also make `uv sync` in one place prune another tier's
# packages.
#
# NEVER `uv run` BARE AGAINST THIS VENV. A bare `uv run` syncs first, and a sync
# against this hand-built venv would uninstall the packages the lock does not
# hold (assistax among them). Every invocation here and afterwards is
# `uv run --no-sync`.
#
#   bash scripts/setup_jax.sh                 # build .venv-jax beside the repo
#   uv run --no-sync --python .venv-jax/bin/python3 python3 -m pytest tests -m jax
#
# THE PARTNER ZOO (the Assistax ad-hoc-teamwork protocol). `train.backend:
# assistax_ppo` trains the robot against FROZEN human partners sampled from
# upstream's zoo of trained policies, published on the Hugging Face Hub as
# `leohink/assistax-zoo` (`zoo.tar.gz`, 728,257,000 bytes). Download and unpack
# it once, then point BIRD_ZOO_PATH at the INNER `zoo` folder -- the one holding
# `index.csv` beside `config/` and `params/`:
#
#   hf download leohink/assistax-zoo zoo.tar.gz --repo-type dataset --local-dir ~/assistax-zoo
#       # or: wget -P ~/assistax-zoo https://huggingface.co/datasets/leohink/assistax-zoo/resolve/main/zoo.tar.gz
#   tar -xzf ~/assistax-zoo/zoo.tar.gz -C ~/assistax-zoo
#   export BIRD_ZOO_PATH=~/assistax-zoo/zoo          # the folder holding index.csv
#   export BIRD_ZOO_TARBALL=~/assistax-zoo/zoo.tar.gz   # optional: also checked
#
# The zoo is checked BY CONTENT before training (`bird/envs/assistax_zoo.py::
# check_zoo`): the sha256 of the unpacked `index.csv`, and of the tarball when
# BIRD_ZOO_TARBALL is set, must match the pinned release, so a truncated extract
# or a different zoo refuses in a second rather than training against partners
# nobody pinned. The path is read from the environment, never the config, so
# the config hash does not depend on where a machine keeps it.
#
# NEEDS A GPU TO BE WORTH RUNNING. `jax[cuda13]` installs on a CPU box and
# imports, but MJX on CPU is slower than CPU MuJoCo -- the tier's whole
# argument is thousands of environments on one device. This script does not
# refuse a CPU box (a developer may want the import path) but says so.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${BIRD_JAX_VENV:-$HERE/.venv-jax}"

#: Pinned, not floating. MJX's API moved inside 0.4.x and the adapters are
#: written against the 0.4.37+ shape; a later `assistax` would silently change
#: the observation layout our specs read positionally.
ASSISTAX_GIT="${BIRD_ASSISTAX_GIT:-https://github.com/assistive-autonomy/assistax}"
ASSISTAX_REF="${BIRD_ASSISTAX_REF:-}"

command -v uv >/dev/null || { echo "uv is not on PATH" >&2; exit 2; }

# THE ONE-STEP SYNC, and it works only because `pyproject.toml`'s `[tool.uv]`
# declares `jax` in conflict with `all` and the mujoco==3.3.0 tiers. With the
# extra IN the lock, the whole venv comes from one `--frozen` sync, and the
# versions are the lock's rather than whatever a second resolver run picks.
#
# --frozen: resolve nothing, install exactly uv.lock. The alternative would
# re-resolve against PyPI and could hand this venv a different jax from the
# one the pins were measured against.
#
# WHY EACH EXTRA.
#
#   jax       the tier. jax[cuda13], mujoco-mjx, brax, gymnasium[mujoco].
#   fasttd3   torch, which the shared training helpers import.
#   sb3       the other learner, and stable-baselines3's gymnasium.
#   anthropic the generator. A jax run is still a BIRD search.
#   video     the recorder degrades loudly without it, and every jax task is
#             a MuJoCo scene somebody will want to watch.
#   test      `pytest -m jax` is the line this script prints at the end, and
#             without the extra there is no pytest in the venv to run it.
EXTRAS="${BIRD_JAX_EXTRAS:-jax fasttd3 sb3 anthropic video test}"
EXTRA_ARGS=()
for e in $EXTRAS; do EXTRA_ARGS+=(--extra "$e"); done

echo "==> syncing $VENV with: $EXTRAS"
echo "    (the jax half alone is ~3 GB of CUDA wheels)"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen "${EXTRA_ARGS[@]}"

if [ -n "$ASSISTAX_REF" ]; then
    echo "==> installing assistax at $ASSISTAX_REF"
    uv pip install --python "$VENV/bin/python3" \
        "assistax @ git+$ASSISTAX_GIT@$ASSISTAX_REF"
else
    # THE PIN, AND IT IS NOT A CHOICE.
    # `bird/envs/assets/assistax/PROVENANCE.json` records that every vendored
    # mesh and XML was extracted from assistive-autonomy/assistax at
    # a7d94f4e20636b9b0c07370344b8a2b521db58b2. Installing the PYTHON package
    # from any other commit would pair upstream code with assets from a
    # different upstream state -- a mismatch nothing downstream could see,
    # since both halves would load and the scene would merely be subtly not
    # the one the assets came from. So the pin is the assets' commit, by
    # construction rather than by selection, and it moves only when the
    # vendored assets are re-extracted.
    #
    # That sha is verified present on the remote. HEAD will move and the
    # assets will not, which is why the pin matters.
    ASSISTAX_REF="a7d94f4e20636b9b0c07370344b8a2b521db58b2"
    echo "==> installing assistax at the assets' commit $ASSISTAX_REF"
    echo "    (bird/envs/assets/assistax/PROVENANCE.json: upstream.commit)"
    uv pip install --python "$VENV/bin/python3" \
        "assistax @ git+$ASSISTAX_GIT@$ASSISTAX_REF"
fi

# VERIFY WHAT IS INSTALLED, NOT WHAT WAS ASKED FOR. pip records the resolved
# VCS commit in the dist-info's `direct_url.json`, so the installed package can
# be checked against the pin rather than trusted because this script asked for
# it. The case that matters is not this script: it is a LATER
# `uv pip install assistax` from HEAD, in this same venv, which would pair new
# upstream code with the vendored assets and leave no trace -- both halves
# import, the scene loads, and it is merely not the scene the assets came from.
# Refusing here is cheap; discovering it later from a number that moved is not.
if [ -n "$ASSISTAX_REF" ]; then
    INSTALLED_SHA="$("$VENV/bin/python3" - <<'PYEOF'
import glob, json, sys
hits = glob.glob(__import__("os").path.join(sys.prefix, "lib", "python*",
                                            "site-packages", "assistax-*.dist-info",
                                            "direct_url.json"))
if not hits:
    print("")                      # not a VCS install, or not installed
else:
    with open(hits[0]) as fh:
        print((json.load(fh).get("vcs_info") or {}).get("commit_id", ""))
PYEOF
)"
    if [ "$INSTALLED_SHA" != "$ASSISTAX_REF" ]; then
        echo "FATAL: the installed assistax is not the pinned commit." >&2
        echo "  pinned    (assets' commit): $ASSISTAX_REF" >&2
        echo "  installed (direct_url.json): ${INSTALLED_SHA:-<none: not a git install>}" >&2
        echo "  The vendored assets under bird/envs/assets/assistax/ were extracted" >&2
        echo "  from the pinned commit. Code from another commit loads fine against" >&2
        echo "  them and gives a scene nobody measured. Reinstall at the pin, or" >&2
        echo "  re-extract the assets and move the pin together." >&2
        exit 1
    fi
    echo "==> assistax verified at $INSTALLED_SHA (direct_url.json)"
fi

echo "==> checking the device"
"$VENV/bin/python3" - <<'PY'
import jax
devs = jax.devices()
print("  jax", jax.__version__, "devices:", devs)
if not any(d.platform == "gpu" for d in devs):
    print("  WARNING: no GPU visible. MJX on CPU is SLOWER than CPU MuJoCo, and")
    print("  `upstream_assistax_*` refuses a non-cuda device at config load. The")
    print("  import path and jax_toy work; the Assistax training path does not.")
PY

echo
echo "done. Use it with --no-sync, always:"
echo "  uv run --no-sync --python $VENV/bin/python3 python3 -m pytest tests -m jax"
if [ -z "${BIRD_ZOO_PATH:-}" ]; then
    echo
    echo "BIRD_ZOO_PATH is unset: train.backend=assistax_ppo needs the partner zoo"
    echo "(HF dataset leohink/assistax-zoo, zoo.tar.gz) -- see the header of this script."
fi
