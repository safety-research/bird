"""Load and index ``policies/`` -- the direct-control policy registry.

``policies/<campaign>/policies.yaml`` records the hand-written policies from the
direct-control campaigns: where each one's verbatim code lives, its tuned
constants, and the score it was measured at (value + unit + instrument, per
seed).  This module is the reader; ``bird/policy_api.py`` is the uniform call
surface built on top of it; ``policies/README.md`` is the contract.

Two deliberate restrictions, both inherited from ``bird/tasks.py``:

- **pyyaml and the standard library only.**  The loader must be importable in
  any venv (neither benchmark family's runtime is installed everywhere -- the
  two are mutually exclusive), and full JSON-Schema validation belongs in a
  test behind an ``importorskip``, never on an import path.
- **Validation here is structural and loud.**  The checks below are the ones
  whose failure would make ``policy_api`` misbehave silently: a duplicate id,
  a campaign field disagreeing with its directory, a policy with nothing to
  run, a null score with no stated reason.  Everything the JSON
  Schema can say is said there instead, once.

The registry deliberately does NOT feed fitness normalisation: the anchors in
``tasks/<id>/shared_spec.yaml`` do that through ``baselines_of``, and these
scores must never be substituted into them (see the MT10 warning in
``policies/README.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

# libyaml's parser where the install has it, the pure-Python one where it does
# not -- `bird/tasks.py`'s arrangement, for the same reason: `CSafeLoader` is
# several times faster on these manifests than `yaml.safe_load`. Both loaders
# construct the same documents from the same events.
try:  # pragma: no cover - availability differs per install
    from yaml import CSafeLoader as _Loader
except ImportError:  # pragma: no cover
    from yaml import SafeLoader as _Loader  # type: ignore[assignment]

__all__ = [
    "PolicyError",
    "PolicyRecord",
    "policies_root",
    "index",
    "get",
    "policy_for_env",
    "reference_for_env",
]


class PolicyError(RuntimeError):
    """A manifest that cannot be trusted: named file, named defect."""


#: Manifest keys a policy entry may carry (mirrored from policies.schema.json;
#: kept here so a typo'd key fails in-process too, not only under jsonschema).
_POLICY_KEYS = {
    "id", "pattern", "task", "task_reason", "env_id",
    "gym_id", "entry", "rig", "params", "params_file", "status",
    "score", "caveats", "notes",
}
_TOP_KEYS = {
    "campaign", "family", "pattern", "runtime", "code", "source",
    "assets", "policies",
}
_PATTERNS = ("closure", "rig")
_FAMILIES = ("metaworld", "humanoidbench", "bird_control")
#: The one unit a score is stated in: the task spec's own metric as the BIRD adapter
#: scores it, which is also the unit a task spec's anchors are stated in.
TASK_METRIC_UNIT = "bird_task_metric"


@dataclass(frozen=True)
class PolicyRecord:
    """One policy, with its campaign context flattened in.

    ``path`` is the manifest that declared it and ``root`` the campaign
    directory every relative CODE reference -- ``entry``, ``params_file`` --
    resolves against.  The record is data only -- resolving ``entry`` into a
    callable is ``policy_api``'s job, and the containment rule (no file
    reference may escape its own campaign dir) is enforced there, where the
    import actually happens.
    """

    id: str
    campaign: str
    family: str
    pattern: str
    task: Optional[str]
    env_id: str
    gym_id: Optional[str]
    entry: Dict[str, Any]
    rig: Optional[Dict[str, Any]]
    params: Optional[Dict[str, Any]]
    params_file: Optional[str]
    status: str
    score: Dict[str, Any]
    caveats: Tuple[str, ...]
    notes: Optional[str]
    #: The manifest's stated reason for a null `task` (the schema REQUIRES one
    #: beside the null), carried so a reader of the record -- a table of
    #: policies, say -- can state the absence rather than draw a blank.
    task_reason: Optional[str] = None
    source: Dict[str, Any] = field(repr=False, default_factory=dict)
    root: Path = field(repr=False, default=Path("."))
    path: Path = field(repr=False, default=Path("."))

from . import paths



def policies_root(start: Optional[Path] = None) -> Path:
    """``policies/``, wherever it is, or a named error rather than a guess.

    An explicit ``start`` still wins and is not checked -- a caller passing a
    path has said where to look. Otherwise this goes through `bird.paths`,
    which finds a checkout, then ``$BIRD_DATA_ROOT``, and RAISES if neither
    has it. ``parent.parent / "policies"`` would be, under a non-editable
    install, a path inside site-packages that does not exist, so every caller
    would get an empty catalogue rather than an error -- and "no policies" and
    "no policies directory" are different facts.
    """
    if start is not None:
        return Path(start)
    return paths.data_dir("policies")


def _err(path: Path, msg: str) -> PolicyError:
    return PolicyError(f"{path}: {msg}")


def _load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.load(fh, Loader=_Loader)
    if not isinstance(doc, dict):
        raise _err(path, "manifest is not a mapping")
    unknown = sorted(set(doc) - _TOP_KEYS)
    if unknown:
        raise _err(path, f"unknown top-level keys {unknown}")
    for key in ("campaign", "family", "pattern", "code", "source", "policies"):
        if key not in doc:
            raise _err(path, f"missing required key `{key}`")
    if doc["campaign"] != path.parent.name:
        raise _err(path, f"campaign `{doc['campaign']}` != directory "
                         f"`{path.parent.name}` -- the id namespace would lie")
    if doc["family"] not in _FAMILIES:
        raise _err(path, f"unknown family `{doc['family']}`")
    if doc["pattern"] not in _PATTERNS:
        raise _err(path, f"unknown pattern `{doc['pattern']}`")
    return doc


def _check_policy(path: Path, camp: Dict[str, Any], pol: Dict[str, Any]) -> None:
    unknown = sorted(set(pol) - _POLICY_KEYS)
    if unknown:
        raise _err(path, f"{pol.get('id', '<no id>')}: unknown keys {unknown}")
    pid = pol.get("id")
    if not isinstance(pid, str) or "/" not in pid:
        raise _err(path, f"policy id `{pid}` is not <campaign>/<name>")
    if pid.split("/", 1)[0] != camp["campaign"]:
        raise _err(path, f"{pid}: id namespace != campaign "
                         f"`{camp['campaign']}`")
    pattern = pol.get("pattern", camp["pattern"])
    if pattern not in _PATTERNS:
        raise _err(path, f"{pid}: unknown pattern `{pattern}`")
    # Every policy must name the thing to run (its `entry`), and a rig policy
    # the Rig/Shim it runs over.
    if not pol.get("entry"):
        raise _err(path, f"{pid}: pattern {pattern} with no entry")
    if pattern == "rig" and not pol.get("rig"):
        raise _err(path, f"{pid}: pattern rig with no rig block -- RigPolicy "
                         "would fail at episode time with a worse message; "
                         "name the campaign's Rig/Shim")
    score = pol.get("score")
    if not isinstance(score, dict):
        raise _err(path, f"{pid}: missing score block")
    if score.get("unit") != TASK_METRIC_UNIT:
        raise _err(path, f"{pid}: score.unit `{score.get('unit')}` is not "
                         f"{TASK_METRIC_UNIT!r}, the one unit this registry scores in")
    if score.get("value") is None and not score.get("reason"):
        raise _err(path, f"{pid}: null score.value with no reason -- absence "
                         "is explicit here, never a silent zero")
    if pol.get("task") is None and not pol.get("task_reason"):
        raise _err(path, f"{pid}: null task with no task_reason")
    if not isinstance(pol.get("env_id"), str) or not pol["env_id"]:
        raise _err(path, f"{pid}: no env_id -- a score is measured through a BIRD "
                         "adapter, so the entry must name it")
    # Referenced files must exist -- a manifest naming a file that is not
    # there is the registry equivalent of the declared-but-unread config key.
    root = path.parent
    rel = pol.get("params_file")
    if rel is not None and not (root / rel).is_file():
        raise _err(path, f"{pid}: params_file `{rel}` does not exist")
    for key in ("entry", "rig"):
        ref = pol.get(key)
        if ref is not None:
            fname = ref.get("file")
            if not fname or not (root / fname).is_file():
                raise _err(path, f"{pid}: {key}.file `{fname}` does not exist")
            if not ref.get("symbol"):
                raise _err(path, f"{pid}: {key}.symbol missing")
    rec = score.get("record")
    if rec is not None and not (root / rec).is_file():
        raise _err(path, f"{pid}: score.record `{rec}` does not exist")


_INDEX: Optional["Dict[str, PolicyRecord]"] = None
_INDEX_ROOT: Optional[Path] = None


def index(root: Optional[Path] = None,
          refresh: bool = False) -> Mapping[str, PolicyRecord]:
    """id -> PolicyRecord over every ``policies/*/policies.yaml``.

    Memoised per root, like ``tasks.index``.  An empty ``policies/`` is a
    valid state (a fresh fork), a malformed manifest never is.
    """
    global _INDEX, _INDEX_ROOT
    base = policies_root(root)
    if not refresh and _INDEX is not None and _INDEX_ROOT == base:
        return _INDEX
    records: Dict[str, PolicyRecord] = {}
    if base.is_dir():
        for manifest in sorted(base.glob("*/policies.yaml")):
            camp = _load_manifest(manifest)
            pols = camp.get("policies")
            if not isinstance(pols, list) or not pols:
                raise _err(manifest, "empty policies list")
            for pol in pols:
                _check_policy(manifest, camp, pol)
                pid = pol["id"]
                if pid in records:
                    raise _err(manifest, f"duplicate policy id `{pid}` "
                                         f"(also in {records[pid].path})")
                records[pid] = PolicyRecord(
                    id=pid,
                    campaign=camp["campaign"],
                    family=camp["family"],
                    pattern=pol.get("pattern", camp["pattern"]),
                    task=pol.get("task"),
                    env_id=pol.get("env_id"),
                    gym_id=pol.get("gym_id"),
                    entry=pol.get("entry"),
                    rig=pol.get("rig"),
                    params=pol.get("params"),
                    params_file=pol.get("params_file"),
                    status=pol["status"],
                    score=pol["score"],
                    caveats=tuple(pol.get("caveats") or ()),
                    notes=pol.get("notes"),
                    task_reason=pol.get("task_reason"),
                    source=dict(camp.get("source") or {}),
                    root=manifest.parent,
                    path=manifest,
                )
    _INDEX, _INDEX_ROOT = records, base
    return records


#: The registry's four verdicts, best first: what `reference_for_env` ranks on before
#: score, and so what decides a task spec's `anchors.scripted`. ONE ranking, owned here,
#: and read only through `reference_for_env`. The order is the strength of the CLAIM each
#: verdict makes about the task:
#:   solved     -- solves the task on the standard reset, at the score shown;
#:   partial    -- part of the way, and the score says how far;
#:   reference  -- a controller kept for what it MEASURES, claiming nothing about solving
#:                 (where no expert anchor exists its number IS the reference), so it
#:                 ranks below a measured `partial`;
#:   negative   -- an honest negative, recorded so nobody re-derives it.
_STATUS_ORDER = {"solved": 0, "partial": 1, "reference": 2, "negative": 3}


def reference_for_env(env_id: str, root: Optional[Path] = None) -> Optional[Tuple[str, PolicyRecord]]:
    """The registry entry that stands for `env_id` in a task spec's `anchors.scripted`:
    the best-verdict entry (solved > partial > reference > negative) and,
    within a verdict, the highest score. Returns (policy_id, record) or None when no
    scored entry names the env."""
    cands = []
    for pid, rec in index(root).items():
        if rec.env_id != env_id or (rec.score or {}).get("value") is None:
            continue
        value = rec.score["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PolicyError(f"{pid}: score.value must be a number to rank on, got {value!r}")
        if rec.status not in _STATUS_ORDER:
            raise PolicyError(f"{pid}: status {rec.status!r} is not one of {sorted(_STATUS_ORDER)}")
        cands.append((_STATUS_ORDER[rec.status], pid, rec, float(value)))
    if not cands:
        return None
    # Every entry is scored in `TASK_METRIC_UNIT` (`_check_policy` refuses any other), so
    # within the best verdict tier the highest score wins.
    top = min(c[0] for c in cands)
    tier = [c for c in cands if c[0] == top]
    _, pid, rec, _ = max(tier, key=lambda c: c[3])
    return pid, rec


def policy_for_env(env_id: str, requested: str = "auto",
                   root: Optional[Path] = None) -> str:
    """The registry id of the scripted policy for ``env_id``.

    ``auto`` resolves the unique entry whose ``env_id`` matches; none or
    several is a refusal that names them, because a silently chosen one would make
    a `train.init: bc_prior` clone a function of manifest order. A named id is
    checked to exist and to belong to the env. Pure lookup over the index, so a
    config's coherence check can ask it at load time.
    """
    records = index(root)
    if requested and requested != "auto":
        rec = records.get(requested)
        if rec is None:
            raise ValueError(f"{requested!r} is not in the policy registry")
        if rec.env_id != env_id:
            raise ValueError(f"{requested!r} is a policy for {rec.env_id!r}, not for {env_id!r}")
        return requested
    hits = sorted(pid for pid, rec in records.items() if rec.env_id == env_id)
    if len(hits) != 1:
        raise ValueError(
            f"`auto` needs exactly one registry policy for env {env_id!r}; "
            f"found {hits or 'none'} -- name one")
    return hits[0]


def get(policy_id: str, root: Optional[Path] = None) -> PolicyRecord:
    recs = index(root)
    try:
        return recs[policy_id]
    except KeyError:
        known = ", ".join(sorted(recs)) or "<none>"
        raise PolicyError(f"unknown policy id `{policy_id}`; known: {known}") \
            from None

