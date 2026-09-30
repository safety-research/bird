"""Post phases act on the restart that OWNS the returned artifact.

`_execute` computes what the run returns as the max of `final_artifact` over
EVERY restart. Handing every `post:` phase `states[-1]` -- the last restart --
regardless would, with `loop.n_restarts > 1` (configs/methods/eureka.yaml pins 5), make
`run_final_retrain` read the last restart's `state.best` and retrain, score and
report a reward that is not the one `result.json` names as `returned_cand_id`.
On the eureka tester point with two restarts, every one of seeds 0-9 would
retrain a different candidate -- seed 0 returns a 1.0 reward and would retrain
a 0.0 one. Every shipped execution profile runs one restart; the published point
does not.

The state handed to the post phases is the restart whose final artifact is the
max (ties to the earliest, as `max` over the list chooses), the retrain records
that restart, and `result.json` names it too.

WHICH restart owns the best is a property of the mock's draws, and the mock
mixes the PROMPT into every draw's RNG (`bird/llm/mock.py::_rng_for`). Under
`generate.context.env_spec: full_source` the prompt carries `toy.py`'s class
body verbatim (`EnvAdapter._render_full_source`), so any edit to `ToyReacher`
-- a docstring, a new method -- re-rolls every candidate and can move the
owner with nothing about the retrain changed (adding one method to the toy
adapter is enough to flip a given seed from owner 1 to owner 0). So the
precondition is MEASURED rather than pinned:
each case scans seeds in order for the first run whose `result.json` names
that owner, and asserts the retrain acted on it. The scan is capped at
`_SEED_SCAN`: measured, restart 0 owns the best from seed 0 and restart 1 first
owns it at seed 14, so both owners are inside twenty.
"""
from __future__ import annotations

import json

import pytest

from test_final_retrain import _run, _search_seeds

#: Seeds tried, in order, for a run whose returned restart is the one a case
#: needs. Twenty because the measurement in the module docstring found both
#: owners inside twenty (restart 1 first at seed 14).
_SEED_SCAN = range(20)
_OVERRIDES = {"loop.n_restarts": 2, "final_retrain.enabled": True, "final_retrain.n_seeds": 3}
#: Fingerprints by seed, shared across the two cases so no seed is searched
#: twice. `_run` returns `{relative path: text}`, which does not depend on the
#: run's `tmp_path` outliving the case that made it.
_RUNS: dict = {}


def _run_owned_by(tmp_path, owner: int):
    """The first seed in `_SEED_SCAN` whose search RETURNED a candidate from
    restart `owner`, with its fingerprint -- read off `result.json`, the artifact
    the claim under test is about."""
    for seed in _SEED_SCAN:
        fp = _RUNS.get(seed)
        if fp is None:
            fp = _RUNS[seed] = _run("eureka", tmp_path / f"seed{seed}", seed=seed, **_OVERRIDES)
        if json.loads(fp["result.json"])["returned_restart"] == owner:
            return seed, fp
    pytest.fail(f"no seed in {list(_SEED_SCAN)} returned a candidate from restart {owner}, "
                "so this case cannot be exercised -- widen _SEED_SCAN rather than drop it")


@pytest.mark.parametrize("owner", [0, 1],
                         ids=["first_restart_owns_the_best", "last_restart_owns_the_best"])
def test_the_retrain_acts_on_the_reward_result_json_returns(tmp_path, owner):
    seed, fp = _run_owned_by(tmp_path, owner)
    res = json.loads(fp["result.json"])
    fr = json.loads(fp["phases/final_retrain.json"])
    assert fr["cand_id"] == res["returned_cand_id"], (
        f"seed {seed}: final_retrain retrained {fr['cand_id']} (fitness {fr['selected_fitness']}) "
        f"but the run returned {res['returned_cand_id']} (fitness {res['returned_fitness']})")
    assert fr["selected_fitness"] == res["returned_fitness"]
    # The precondition that makes the first case a test at all: the owner is NOT
    # the last restart, which is exactly the case `states[-1]` would get wrong.
    assert fr["restart"] == res["returned_restart"] == owner, (seed, fr["restart"], res["returned_restart"])
    # And the fresh-seed stream composes with it: the retrain's seeds are
    # disjoint from the winner's OWN search seeds in the owning restart -- the
    # ones that selected it -- and the artifact records the same.
    search = _search_seeds(fp)
    assert fr["cand_id"] in search and search[fr["cand_id"]]
    assert not (set(fr["seeds"]) & search[fr["cand_id"]]), (fr["seeds"], search[fr["cand_id"]])
    assert fr["seed_overlap"] == []
