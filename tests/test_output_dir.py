"""`output.dir` is honoured: it is the run root when the CLI is given no `--out`.

A key that is declared (configs/_default.yaml, bird/schema.py) and hashed but
read by nothing lets `-s output.dir=...` validate, move the hash and change
nothing. It is honoured rather than refused because a launcher that passes
`--out` explicitly still wins, so nothing such a launcher starts moves; and
`status.json` records which of the two chose the root (`out_root_source`), so a
run directory can say why it is where it is (the root itself is not written: an
absolute path in status.json would make two otherwise identical runs differ,
which the resume-parity test compares).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from conftest import REPO


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SMALL = ["-s", "loop.n_iterations=1", "-s", "generate.n_candidates=1",
          "-s", "evaluate.rollouts_per_candidate=1"]


def _only_run(root: Path) -> Path:
    dirs = [p for p in root.iterdir() if p.is_dir()]
    assert len(dirs) == 1, f"expected one run dir under {root}, found {dirs}"
    return dirs[0]


def test_output_dir_is_the_run_root_when_no_out_flag_is_given(tmp_path) -> None:
    root = tmp_path / "from_config"
    rc = _entry().main(["-c", "eureka", "-p", "tester",
                        "-s", f"output.dir={root}", *_SMALL])
    assert rc == 0
    assert root.is_dir(), "-s output.dir=... validated and changed nothing"
    run = _only_run(root)
    status = json.loads((run / "status.json").read_text())
    assert status["status"] == "ok"
    assert status["out_root_source"] == "config"
    assert run.parent.resolve() == root.resolve()


def test_an_explicit_out_flag_still_wins(tmp_path) -> None:
    """A launcher that passes `--out` keeps its run root regardless of the config."""
    ignored, cli = tmp_path / "ignored", tmp_path / "cli"
    rc = _entry().main(["-c", "eureka", "-p", "tester",
                        "-s", f"output.dir={ignored}", "--out", str(cli), *_SMALL])
    assert rc == 0
    assert not ignored.exists()
    run = _only_run(cli)
    status = json.loads((run / "status.json").read_text())
    assert status["out_root_source"] == "cli"
    assert run.parent.resolve() == cli.resolve()


def test_the_default_root_is_still_runs() -> None:
    """The key's default is the flag's default literal, so a config that never
    sets it writes under `runs/`."""
    from bird.config import load
    assert load("eureka", profile="tester")["output.dir"] == "runs"
