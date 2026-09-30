"""`scripts/derive_humanoid_facts.py`'s check must be able to tell two tasks apart.

NOT `slow`, deliberately, and it is the file's kind that decides that (pyproject.toml
assigns the marker by kind, not by stopwatch): everything here is pure logic over a
synthetic facts dict. It imports no mujoco, loads no asset and constructs no environment,
so it belongs in the default (not `slow`) selection. The half that needs a real model is
the script's own `--check` CLI, which is run by hand.

WHAT THIS PINS, and why it is the only interesting property. Every h1hand model in
HumanoidBench shares one robot and one 61-actuator table -- verified across all ten
committed specs -- so the actuator comparison passes for ANY pairing of task and spec.
A check that treated the observation width as an informational note would be
structurally incapable of catching a spec spliced from the wrong task, which is the
single failure the script exists to prevent: under such a check, derived `walk` (151
wide) against the committed `door` spec (155) reports AGREES.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "derive_humanoid_facts.py"


def _module():
    """Import the script by path -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("derive_humanoid_facts", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _facts(nu: int = 2, width: int = 10) -> dict:
    return {
        "task": "synthetic", "robot": "h1hand",
        "qpos_qvel_width": width,
        "actuators": [{"name": f"act_{i}", "ctrlrange": [-0.5, 0.5]} for i in range(nu)],
    }


def _spec_dir(tmp_path: pathlib.Path, *, obs_dim: int, nu: int = 2,
              quote: str = "[-0.5, 0.5] rad", names: list[str] | None = None) -> pathlib.Path:
    import yaml

    names = names or [f"act_{i}" for i in range(nu)]
    doc = {
        "env": {"spaces": {
            "obs_dim": obs_dim,
            "action_fields": [{"name": n, "low": -1.0, "high": 1.0,
                               "description": f"a joint. range {quote}"} for n in names],
        }},
        "state_surface": {"flat_fields": [{"index": i, "name": f"s{i}"} for i in range(obs_dim)]},
    }
    d = tmp_path / "spec"
    d.mkdir()
    (d / "shared_spec.yaml").write_text(yaml.safe_dump(doc))
    return d


def test_a_spec_whose_width_matches_the_model_agrees(tmp_path):
    mod = _module()
    assert mod.check(_facts(width=10), str(_spec_dir(tmp_path, obs_dim=10))) == []


def test_a_spec_spliced_from_a_WIDER_task_is_caught(tmp_path):
    """The regression. Same robot, same actuators, different scene -- which is exactly
    what every wrong splice looks like, because the robot half is shared by construction.
    """
    mod = _module()
    problems = mod.check(_facts(width=10), str(_spec_dir(tmp_path, obs_dim=14)))
    assert problems, "a 14-wide spec against a 10-wide model must not pass"
    assert any("expected 0" in p for p in problems), problems


def test_a_declared_surplus_is_accepted_only_at_the_declared_size(tmp_path):
    """`extra_dims` is a claim with a number in it, so the number is checked.

    Both directions matter: push/package (3) and basketball (1) carry real task state
    outside qpos+qvel, and accepting ANY surplus would put the check straight back to
    where it could not discriminate.
    """
    mod = _module()
    spec = str(_spec_dir(tmp_path, obs_dim=13))
    assert mod.check(_facts(width=10), spec, expected_extra=3) == []
    assert mod.check(_facts(width=10), spec, expected_extra=1), "surplus 3 is not 1"
    assert mod.check(_facts(width=10), spec, expected_extra=0), "surplus 3 is not 0"


def test_a_description_that_does_not_quote_the_models_range_is_caught(tmp_path):
    """The action bound a spec records is the NORMALISED +/-1, so the only place the
    model's real range appears is the description text -- which makes that text load
    bearing rather than decorative, and worth checking.
    """
    mod = _module()
    problems = mod.check(_facts(width=10),
                         str(_spec_dir(tmp_path, obs_dim=10, quote="[-9, 9] rad")))
    assert any("does not quote" in p for p in problems), problems


def test_a_renamed_actuator_is_caught(tmp_path):
    mod = _module()
    problems = mod.check(_facts(width=10),
                         str(_spec_dir(tmp_path, obs_dim=10, names=["act_0", "WRONG"])))
    assert any("actuator name" in p for p in problems), problems


def test_an_unknown_robot_is_refused_not_guessed():
    """`--robot` accepts any string but the scene/robot split is computed from a
    MEASURED robot width. A robot with no measured entry must refuse rather than
    silently label robot joints as scene joints."""
    mod = _module()
    assert mod.robot_nq("h1hand") == 76
    assert mod.robot_nq("h1strong") == 76
    with pytest.raises(SystemExit, match="add it to _ROBOT_NQ"):
        mod.robot_nq("g1")

