"""`era_s` is its own CONFIG FILE, so the variant is a config point: the `--diff`
against its parent `era_u` is the one key that selects the fitness source."""
from __future__ import annotations

from pathlib import Path

from bird.config import load

REPO = Path(__file__).resolve().parents[1]

PARENT = "era_u"
CHILD = "era_s"


def test_the_native_file_is_its_parent_one_key_apart():
    a = load(REPO / "configs" / f"{PARENT}.yaml", profile="full")
    b = load(REPO / "configs" / f"{CHILD}.yaml", profile="full")
    assert set(a.diff(b)) == {"name", "evaluate.fitness.source"}, sorted(a.diff(b))
    assert b["evaluate.fitness.source"] == "native" and b["name"] == CHILD
    assert b["problem.fitness_access"] == "none"
