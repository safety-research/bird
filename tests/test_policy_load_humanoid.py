"""Every runnable humanoidbench-family registry policy LOADS, in a fresh process.

A registered policy can still raise ModuleNotFoundError from
`bird.policy_api.load_policy`: a bare `import hk` fails when the loader does not put the
campaign dir on sys.path, and an import chain that resolves only through a directory outside this repo fails
everywhere but the machine it was written on. Nothing else would notice, because nothing loads a
HumanoidBench policy outside its own campaign: the tier needs `mujoco==3.1.6` and HumanoidBench,
which no `uv.lock` holds, so no CI job can construct one.

This test is therefore marked `humanoid` (deselected in every CI job) and gated on `humanoid_bench`. It says
nothing on a green PR. RUN IT BY HAND in the tier's venv (`bash scripts/setup_humanoid.sh`):

    source scripts/setup_gl.sh --env
    PYTHONPATH=. .venv-humanoid/bin/python -m pytest tests/test_policy_load_humanoid.py -m humanoid

One subprocess per CAMPAIGN, loading every runnable id in it: one campaign per process is the loader's
supported shape (`bird.policy_api.import_beside`), and a fresh interpreter is what a load in
`scripts/eval_policy.py` gets -- an in-process loop would let one campaign's sibling modules (a campaign
ships helpers such as `hk.py` beside its entry, and two campaigns may ship modules of the same name) satisfy
the next campaign's imports and hide exactly this defect.
Each policy is loaded, reset and stepped three times on its own env (seed 0): a sibling module imported lazily
inside a function -- at reset or at the first step -- fails there and not at load, so a load-only check can pass
a policy that raises at its first step."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import bird.policies as bp

REPO = Path(__file__).resolve().parents[1]


def _campaigns():
    out = {}
    for pid, rec in bp.index().items():
        if rec.family == "humanoidbench":
            out.setdefault(pid.split("/")[0], []).append(pid)
    return {k: sorted(v) for k, v in sorted(out.items())}


CAMPAIGNS = _campaigns()

#: Registered ids known broken for a reason outside this repo's loader: each must
#: still fail, so an entry cannot outlive its fix (the probe reports a known id that now runs as a failure).
KNOWN_BROKEN: dict = {}

_PROBE = """
import sys
import numpy as np
from bird import registry
from bird.policy_api import load_policy
import bird.policies as bp
registry.load_all()
known = set(sys.argv[1].split(",")) - {""}
bad, envs = [], {}
for pid in sys.argv[2:]:
    try:
        pol = load_policy(pid)
        env_id = bp.get(pid).env_id
        env = envs.get(env_id) or envs.setdefault(env_id, registry.get("env", env_id)(None))
        s = env.reset(np.random.default_rng(0))
        pol.reset(np.random.default_rng(0))
        for t in range(3):   # a sibling imported lazily at reset or at the first steps fails HERE, not at load
            s, done, _ = env.step(s, pol.act(s, t=t, env=env))
            if done:
                break
        if pid in known:
            bad.append(f"{pid}: listed in KNOWN_BROKEN but now loads, resets and steps -- remove it from the list")
    except Exception as e:
        if pid not in known:
            bad.append(f"{pid}: {type(e).__name__}: {e}")
print("\\n".join(bad))
sys.exit(1 if bad else 0)
"""


def test_the_catalogue_is_not_empty():
    """The parametrization below is keyed on the registry; an empty one would pass by collecting nothing. Every
    KNOWN_BROKEN id must be a registered id, or the exemption silently covers nothing."""
    # The release ships one humanoidbench-family campaign, the upstream-reward hacks.
    assert "h1hand_hb_hacks" in CAMPAIGNS, sorted(CAMPAIGNS)
    ids = {pid for v in CAMPAIGNS.values() for pid in v}
    assert set(KNOWN_BROKEN) <= ids, set(KNOWN_BROKEN) - ids


@pytest.mark.humanoid
@pytest.mark.slow
@pytest.mark.parametrize("campaign", sorted(CAMPAIGNS), ids=str)
def test_every_runnable_policy_in_the_campaign_loads_resets_and_steps_in_a_fresh_process(campaign):
    pytest.importorskip("humanoid_bench")
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.setdefault("MUJOCO_GL", "egl")
    known = ",".join(k for k in KNOWN_BROKEN if k in CAMPAIGNS[campaign])
    r = subprocess.run([sys.executable, "-c", _PROBE, known, *CAMPAIGNS[campaign]], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{campaign}: runnable policies that do not load:\n{r.stdout}\n{r.stderr[-2000:]}"
