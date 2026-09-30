"""The Assistax zoo partner split: a function of (index contents, scenario, seed) and nothing else."""
import csv
import json
import os
import random

import pytest

from bird.envs.assistax_zoo import (LOADED_AGENT, PartnerSplit, index_sha256, partner_pool, read_index,
                                    split_partners)

COLS = ["agent_uuid", "scenario", "scenario_agent_id", "algorithm", "is_rnn", "rnn_dim", "team_uuid",
        "w_speed", "w_force", "w_touch"]


def _write_zoo(path, rows):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)


def _rows(scenarios=("scratchitch", "feeding"), per_algo=4):
    rows, n = [], 0
    for sc in scenarios:
        for algo in ("IPPO", "MAPPO", "MASAC"):
            for _ in range(per_algo):
                team = f"team-{n:04d}"
                for role in ("robot", "human"):
                    rows.append({"agent_uuid": f"{sc[:2]}-{algo.lower()}-{role}-{n:04d}", "scenario": sc,
                                 "scenario_agent_id": role, "algorithm": algo, "is_rnn": "False", "rnn_dim": "0",
                                 "team_uuid": team, "w_speed": "0.1", "w_force": "0.4", "w_touch": "-0.03"})
                n += 1
    return rows


@pytest.fixture
def zoo(tmp_path):
    z = str(tmp_path / "zoo"); _write_zoo(z, _rows()); return z


def test_pool_is_the_scenarios_human_half_only(zoo):
    pool = partner_pool(read_index(zoo), "scratchitch")
    assert len(pool) == 12 and all(u.startswith("sc-") and "-human-" in u for u in pool)
    assert sorted(set(pool.values())) == ["IPPO", "MAPPO", "MASAC"]
    assert list(pool) == sorted(pool), "pool is uuid-ordered, not file-ordered"


def test_train_has_the_nested_shape_load_from_zoo_iterates(zoo):
    s = split_partners(zoo, "scratchitch", seed=3, n_train=5, stratify=False)
    for algo, agents in s.train.items():
        assert algo in ("IPPO", "MAPPO", "MASAC") and list(agents) == [LOADED_AGENT]
        assert all(isinstance(u, str) for u in agents[LOADED_AGENT])
    assert len(s.train_uuids) == 5 and s.pool_size == 12 and len(s.heldout) == 7


def test_train_and_heldout_are_disjoint_and_cover_the_draw(zoo):
    s = split_partners(zoo, "scratchitch", seed=11, n_train=4, n_heldout=3, stratify=False)
    assert not set(s.train_uuids) & set(s.heldout)
    assert len(s.train_uuids) == 4 and len(s.heldout) == 3


def test_same_seed_same_split_different_seed_different_split(zoo):
    a = split_partners(zoo, "scratchitch", seed=7, n_train=6)
    b = split_partners(zoo, "scratchitch", seed=7, n_train=6)
    c = split_partners(zoo, "scratchitch", seed=8, n_train=6)  # stratified default: 2 per algorithm
    assert a == b
    assert a.train_uuids != c.train_uuids or a.heldout != c.heldout


def test_split_is_independent_of_index_row_order(tmp_path):
    rows = _rows(); z1 = str(tmp_path / "z1"); z2 = str(tmp_path / "z2")
    _write_zoo(z1, rows); shuffled = list(rows); random.Random(0).shuffle(shuffled); _write_zoo(z2, shuffled)
    assert index_sha256(z1) != index_sha256(z2), "the two files really differ"
    a = split_partners(z1, "feeding", seed=5, n_train=6); b = split_partners(z2, "feeding", seed=5, n_train=6)
    assert a.train == b.train and a.heldout == b.heldout


def test_scenario_and_algorithm_filters(zoo):
    s = split_partners(zoo, "feeding", seed=1, n_train=3, algorithms=["MASAC"])
    assert list(s.train) == ["MASAC"] and all(u.startswith("fe-masac-human-") for u in s.train_uuids)
    assert s.algorithms == ["MASAC"] and s.seed_row()["algorithms"] == ["MASAC"]
    with pytest.raises(ValueError, match="no zoo agent plays"):
        split_partners(zoo, "bedbathing", seed=1, n_train=1)


def test_refuses_a_draw_larger_than_the_pool(zoo):
    with pytest.raises(ValueError, match="exceeds the pool"):
        split_partners(zoo, "scratchitch", seed=1, n_train=10, n_heldout=5)
    with pytest.raises(ValueError):
        split_partners(zoo, "scratchitch", seed=1, n_train=0)


def test_missing_index_and_missing_column_fail_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_index(str(tmp_path))
    bad = str(tmp_path / "bad"); os.makedirs(bad)
    with open(os.path.join(bad, "index.csv"), "w") as fh:
        fh.write("agent_uuid,scenario\nx,y\n")
    with pytest.raises(ValueError, match="lacks columns"):
        read_index(bad)


def test_record_is_json_with_the_seed_and_index_digest(zoo):
    s = split_partners(zoo, "scratchitch", seed=42, n_train=3)
    j = json.loads(s.to_json())
    assert j["seed"] == 42 and j["index_sha256"] == index_sha256(zoo) and j["scenario"] == "scratchitch"
    assert isinstance(PartnerSplit(**j), PartnerSplit)


def test_cli_prints_the_same_split(zoo, capsys):
    from bird.envs.assistax_zoo import main
    assert main([zoo, "scratchitch", "--seed", "9", "--n-train", "3"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out == json.loads(split_partners(zoo, "scratchitch", 9, 3).to_json())
    assert main([zoo, "scratchitch", "--seed", "9", "--n-train", "5", "--no-stratify"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(split_partners(zoo, "scratchitch", 9, 5, stratify=False).to_json())


def test_seed_row_carries_scalars_not_uuid_lists(zoo):
    s = split_partners(zoo, "scratchitch", seed=42, n_train=6, n_heldout=3)
    row = s.seed_row()
    assert row == {"scenario": "scratchitch", "agent": LOADED_AGENT, "split_seed": 42, "pool_size": 12, "n_train": 6,
                   "n_heldout": 3, "index_sha256": index_sha256(zoo), "algorithms": None, "stratify": True}
    assert not any(isinstance(v, list) and v and "-human-" in str(v[0]) for v in row.values())


def test_stratified_default_gives_every_algorithm_the_same_count_on_each_side(zoo):
    s = split_partners(zoo, "scratchitch", seed=0, n_train=6, n_heldout=6)     # pool 12 = 4 per algorithm
    assert {a: len(v[LOADED_AGENT]) for a, v in s.train.items()} == {"IPPO": 2, "MAPPO": 2, "MASAC": 2}
    held_by_algo = {}
    for u in s.heldout:
        held_by_algo[u.split("-")[1].upper()] = held_by_algo.get(u.split("-")[1].upper(), 0) + 1
    assert held_by_algo == {"IPPO": 2, "MAPPO": 2, "MASAC": 2}
    assert not set(s.train_uuids) & set(s.heldout)


def test_stratified_refuses_counts_that_do_not_divide_and_pooled_accepts_them(zoo):
    with pytest.raises(ValueError, match="divisible by the 3 algorithms"):
        split_partners(zoo, "scratchitch", seed=0, n_train=5)
    s = split_partners(zoo, "scratchitch", seed=0, n_train=5, stratify=False)
    assert len(s.train_uuids) == 5


def test_stratified_refuses_when_one_algorithm_is_short(tmp_path):
    rows = [r for r in _rows() if not (r["algorithm"] == "MASAC" and r["team_uuid"] in ("team-0010", "team-0011"))]
    z = str(tmp_path / "uneven"); _write_zoo(z, rows)          # scratchitch humans: IPPO 4, MAPPO 4, MASAC 2
    with pytest.raises(ValueError, match="fewer than"):
        split_partners(z, "scratchitch", seed=0, n_train=6, n_heldout=3)   # 2 + 1 per algorithm > MASAC's 2
    s = split_partners(z, "scratchitch", seed=0, n_train=6, n_heldout=0)  # 2 per algorithm fits
    assert {a: len(v[LOADED_AGENT]) for a, v in s.train.items()} == {"IPPO": 2, "MAPPO": 2, "MASAC": 2}


def test_pooled_draw_depends_on_the_seed_and_on_the_ranking(zoo):
    a = split_partners(zoo, "scratchitch", seed=1, n_train=5, stratify=False)
    b = split_partners(zoo, "scratchitch", seed=2, n_train=5, stratify=False)
    assert a.train_uuids != b.train_uuids, "the pooled path must rank by seed, not take the uuid-sorted head"
    pool_sorted = sorted(u for u in a.train_uuids + a.heldout)
    assert a.train_uuids != pool_sorted[:5], "not the first five of the uuid-sorted pool"


def test_seed_row_and_record_distinguish_stratified_from_pooled(zoo):
    a = split_partners(zoo, "scratchitch", seed=3, n_train=6)
    b = split_partners(zoo, "scratchitch", seed=3, n_train=6, stratify=False)
    assert a.train_uuids != b.train_uuids or a.heldout != b.heldout
    assert a.seed_row() != b.seed_row() and a.seed_row()["stratify"] is True and b.seed_row()["stratify"] is False
    assert json.loads(a.to_json())["stratify"] is True and json.loads(b.to_json())["stratify"] is False


def test_algorithms_records_the_filter_not_the_draw(zoo):
    s = split_partners(zoo, "scratchitch", seed=5, n_train=2, algorithms=["IPPO", "MAPPO"], stratify=True)
    assert s.algorithms == ["IPPO", "MAPPO"]                    # the filter, even though both algorithms drew
    t = split_partners(zoo, "scratchitch", seed=5, n_train=1, algorithms=["IPPO", "MAPPO"], stratify=False)
    assert t.algorithms == ["IPPO", "MAPPO"] and len(t.train) == 1, "one algorithm drew; the record still says the filter"
