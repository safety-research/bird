"""Partner population for Assistax ad-hoc teamwork: a deterministic train / held-out split of the zoo.

The zoo (`leohink/assistax-zoo`, ``zoo.tar.gz``) is upstream's directory: ``index.csv`` with one row per
trained agent (``agent_uuid, scenario, scenario_agent_id, algorithm, is_rnn, rnn_dim, team_uuid, w_speed,
w_force, w_touch``), ``config/<uuid>.yaml`` and ``params/<uuid>.safetensors``. Upstream's
``LoadAgentWrapper.load_from_zoo(env, zoo, load_agents_uuids)`` (refs/code/assistax/assistax/wrappers/aht.py)
does everything after the choice of partners: it loads each uuid, derives the preference configuration from
each uuid's own config, stacks them, and resamples the partner per episode. This module supplies ONLY the
population, in the nested shape that function iterates::

    {"IPPO": {"human": [uuid, ...]}, "MAPPO": {"human": [...]}, "MASAC": {"human": [...]}}

and keeps the held-out partners in a separate list that is never handed to the wrapper.

Why the split is ranked by ``sha256(seed || uuid)`` rather than by shuffling the file order: the partition
must be a function of the seed alone. A zoo re-extract that reorders ``index.csv`` rows, or a different Python
version's ``random.shuffle``, would otherwise change which partners are held out under the same recorded
seed, and the held-out metric would silently be measured on a different population. Sorting the pool by uuid
and ranking by a hash makes the draw independent of file order, platform and interpreter.

Stdlib only (csv, hashlib); the zoo's params are never read here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

INDEX_NAME = "index.csv"
REQUIRED_COLUMNS = ("agent_uuid", "scenario", "scenario_agent_id", "algorithm")
#: the agent every Assistax task loads from the zoo (upstream hardcodes ``self.loaded_agents = ['human']``)
LOADED_AGENT = "human"
#: Training partners drawn per scenario: half of the zoo's 630 partners per task, 105 IPPO / 105 MAPPO /
#: 105 MASAC, with the other 315 held out. The paper's protocol; ``split_partners`` is deterministic in
#: (seed, n_train), so the 315 uuids are DERIVED at construction rather than written into a config.
DEFAULT_N_TRAIN = 315

#: The environment variable naming the UNPACKED zoo: the inner ``zoo`` folder holding ``index.csv`` beside
#: ``config/`` and ``params/``. An execution fact (a per-machine path), so it is read from the environment and
#: never from the config, where it would fork one experiment's config hash across machines.
ZOO_ENV = "BIRD_ZOO_PATH"
#: Optional: the downloaded ``zoo.tar.gz`` itself, checked against ``ZOO_SHA256`` when set.
ZOO_TARBALL_ENV = "BIRD_ZOO_TARBALL"
#: sha256 of ``zoo.tar.gz`` from ``leohink/assistax-zoo`` on the Hugging Face Hub (728,257,000 bytes).
ZOO_SHA256 = "360cadf22ea8fda6b6315a5d4cecf0672720ddcc95d8c7ffdeb49f12dc401ce6"
#: sha256 of the unpacked zoo's ``index.csv`` (803,745 bytes, 6,300 rows) -- the stronger of the two checks for
#: what the adapter actually reads. The tarball digest proves the archive; this proves the extraction: a tarball
#: that verified and then unpacked onto a full disk leaves an index that is short, present and plausible, and
#: ``split_partners`` would draw its partners from whatever rows survived.
ZOO_INDEX_SHA256 = "39ca96ba78690554a5f30d5302704254b6b7ba390fd5b845ccc71162b6b57c14"


@dataclass(frozen=True)
class PartnerSplit:
    """The recorded partition. ``train`` is exactly what ``load_from_zoo`` takes as ``load_agents_uuids``."""

    scenario: str
    seed: int
    agent: str
    train: Dict[str, Dict[str, List[str]]]
    heldout: List[str]
    pool_size: int
    index_sha256: str
    #: the ``algorithms=`` FILTER the pool was restricted to (None = every algorithm in the index), not the draw
    algorithms: Optional[List[str]] = None
    stratify: bool = True

    @property
    def train_uuids(self) -> List[str]:
        return [u for algo in sorted(self.train) for u in self.train[algo][self.agent]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    def seed_row(self) -> Dict[str, object]:
        """The scalars that make the split reproducible, never the uuid lists -- a compact summary for a record.

        Same (scenario, agent, split_seed, index_sha256, n_train, n_heldout, algorithms filter, stratify) => the same
        partition; every one of those is a function argument or the index bytes, and ``stratify`` is here because
        True and False give different partitions from otherwise identical inputs. The lists themselves belong in the
        run artefact (``to_json``); a row carrying 210 uuids is unreadable.
        """
        return {"scenario": self.scenario, "agent": self.agent, "split_seed": self.seed, "pool_size": self.pool_size,
                "n_train": len(self.train_uuids), "n_heldout": len(self.heldout), "index_sha256": self.index_sha256,
                "algorithms": None if self.algorithms is None else list(self.algorithms), "stratify": self.stratify}


def read_index(zoo_path: str) -> List[Dict[str, str]]:
    """The zoo's index rows as dicts. Fails loudly on a missing file or a missing required column."""
    path = os.path.join(zoo_path, INDEX_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"zoo index not found: {path}")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"zoo index is empty: {path}")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"zoo index {path} lacks columns {missing}; has {sorted(rows[0])}")
    return rows


def index_sha256(zoo_path: str) -> str:
    with open(os.path.join(zoo_path, INDEX_NAME), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def partner_pool(
    rows: Sequence[Dict[str, str]],
    scenario: str,
    agent: str = LOADED_AGENT,
    algorithms: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """uuid -> algorithm for every zoo agent that can play ``agent`` in ``scenario``, in uuid order.

    Filtering on ``scenario_agent_id`` matters: the zoo holds the ROBOT half of every team too, and a robot
    policy loaded as the human would act on the wrong observation.
    """
    algos = set(algorithms) if algorithms else None
    pool = {
        r["agent_uuid"]: r["algorithm"]
        for r in rows
        if r["scenario"] == scenario and r["scenario_agent_id"] == agent and (algos is None or r["algorithm"] in algos)
    }
    if not pool:
        raise ValueError(f"no zoo agent plays {agent!r} in scenario {scenario!r}"
                         + (f" under algorithms {sorted(algos)}" if algos else ""))
    return dict(sorted(pool.items()))


def _rank_key(seed: int, uuid: str) -> str:
    return hashlib.sha256(f"{int(seed)}:{uuid}".encode()).hexdigest()


def split_partners(
    zoo_path: str,
    scenario: str,
    seed: int,
    n_train: int,
    n_heldout: Optional[int] = None,
    agent: str = LOADED_AGENT,
    algorithms: Optional[Sequence[str]] = None,
    stratify: bool = True,
) -> PartnerSplit:
    """Draw ``n_train`` training partners and ``n_heldout`` held-out partners (default: the rest) for a scenario.

    Deterministic in ``seed``: every uuid is ranked by ``sha256(seed:uuid)``; the first ``n_train`` in rank order
    train, the next ``n_heldout`` are held out; disjoint by construction. With ``stratify`` (the default, the AHT
    protocol's 50/50 by training algorithm: 105 IPPO / 105 MAPPO / 105 MASAC of 315 on each side) the ranking and
    the cut happen PER ALGORITHM, so ``n_train`` and ``n_heldout`` must divide evenly across the algorithms present
    and every algorithm contributes the same count to each side; ``stratify=False`` ranks the pool as one list.
    Refuses a draw larger than the pool (or than an algorithm's share).
    """
    if n_train < 1:
        raise ValueError("n_train must be >= 1")
    rows = read_index(zoo_path)
    pool = partner_pool(rows, scenario, agent=agent, algorithms=algorithms)
    if n_heldout is None:
        n_heldout = len(pool) - n_train
    if n_heldout < 0 or n_train + n_heldout > len(pool):
        raise ValueError(f"n_train {n_train} + n_heldout {n_heldout} exceeds the pool of {len(pool)} "
                         f"{agent!r} agents for {scenario!r}")
    if stratify:
        by_algo: Dict[str, List[str]] = {}
        for u, a in pool.items():
            by_algo.setdefault(a, []).append(u)
        k = len(by_algo)
        if n_train % k or n_heldout % k:
            raise ValueError(f"stratified split needs n_train {n_train} and n_heldout {n_heldout} divisible by the "
                             f"{k} algorithms present {sorted(by_algo)}; pass stratify=False for a pooled draw")
        per_t, per_h = n_train // k, n_heldout // k
        train_uuids, heldout = [], []
        for a in sorted(by_algo):
            ranked = sorted(by_algo[a], key=lambda u: _rank_key(seed, u))
            if per_t + per_h > len(ranked):
                raise ValueError(f"algorithm {a} has {len(ranked)} {agent!r} agents for {scenario!r}, "
                                 f"fewer than the {per_t} + {per_h} a stratified split needs")
            train_uuids += ranked[:per_t]
            heldout += ranked[per_t:per_t + per_h]
    else:
        ranked = sorted(pool, key=lambda u: _rank_key(seed, u))
        train_uuids = ranked[:n_train]
        heldout = ranked[n_train:n_train + n_heldout]
    train: Dict[str, Dict[str, List[str]]] = {}
    for u in train_uuids:
        train.setdefault(pool[u], {}).setdefault(agent, []).append(u)
    return PartnerSplit(
        scenario=scenario, seed=int(seed), agent=agent, train=dict(sorted(train.items())),
        heldout=heldout, pool_size=len(pool), index_sha256=index_sha256(zoo_path),
        algorithms=None if algorithms is None else sorted(set(algorithms)), stratify=bool(stratify),
    )


def _file_sha256(path: str) -> str:
    """A file's sha256, streamed. Empty string when it is not there."""
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_zoo(zoo_path: Optional[str] = None, tarball: Optional[str] = None) -> Optional[str]:
    """The reason to refuse a zoo, or None. Called BEFORE training, not during.

    BY CONTENT, NOT BY PRESENCE. A half-written tarball from an interrupted copy is present, has plausible size,
    and unpacks to a partial zoo; the partners load, the run proceeds, and the protocol is quietly not the one
    the paper describes. ``zoo_path`` defaults to ``$BIRD_ZOO_PATH`` and ``tarball`` to ``$BIRD_ZOO_TARBALL``;
    an unset tarball is not checked (the unpacked index is what the adapter reads).
    """
    zoo_dir = zoo_path if zoo_path is not None else (os.environ.get(ZOO_ENV) or "")
    if zoo_dir:
        index = os.path.join(str(zoo_dir), INDEX_NAME)
        if not os.path.isfile(index):
            return (f"zoo path {zoo_dir!r} has no {INDEX_NAME}. {ZOO_ENV} must name the INNER zoo folder -- "
                    f"the one holding {INDEX_NAME} beside config/ and params/ -- not the directory "
                    f"zoo.tar.gz was unpacked into.")
        got_index = _file_sha256(index)
        if got_index != ZOO_INDEX_SHA256:
            return (f"zoo path {zoo_dir!r} has an {INDEX_NAME} with sha256 {got_index}, not the "
                    f"{ZOO_INDEX_SHA256} of the pinned leohink/assistax-zoo release. A short or stale index "
                    f"is the failure a size check misses: the partners would be drawn from whatever rows "
                    f"survived, and the split would be deterministic, reproducible and wrong.")
    path = tarball if tarball is not None else (os.environ.get(ZOO_TARBALL_ENV) or "")
    if not path:
        return None
    got = _file_sha256(path)
    if not got:
        return f"the zoo tarball named by {ZOO_TARBALL_ENV} is not there: {path!r}."
    if got != ZOO_SHA256:
        return (f"the zoo tarball at {path!r} has sha256 {got}, not the {ZOO_SHA256} of the pinned "
                f"leohink/assistax-zoo release. A different zoo is a different experiment under the same name.")
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="deterministic Assistax zoo partner split (prints JSON)")
    p.add_argument("zoo_path"); p.add_argument("scenario"); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-train", type=int, required=True); p.add_argument("--n-heldout", type=int, default=None)
    p.add_argument("--agent", default=LOADED_AGENT); p.add_argument("--algorithms", nargs="*", default=None)
    p.add_argument("--no-stratify", action="store_true", help="rank the pool as one list instead of per algorithm")
    a = p.parse_args(argv)
    print(split_partners(a.zoo_path, a.scenario, a.seed, a.n_train, a.n_heldout, a.agent, a.algorithms,
                         stratify=not a.no_stratify).to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
