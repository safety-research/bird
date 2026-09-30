"""Checkpoint + resume: `loop.resume_from`.

`loop.resume_from` is a **hash-stable sentinel** (`null | auto | auto_degraded`),
never a path. That is the whole trick. `Config.hash()` covers all of `_data`, so
a value that varied per leg (a directory name, a timestamp) would give each leg
of one search a different hash and therefore a different `runs/<name>-<H>-<t>/`.
A fixed string resolves identically on every leg, so a resumed search stays ONE
run directory whose name still contains the hash of the config that produced it.

Non-null also turns checkpoint writing ON. A later invocation of the *same
command line* adopts the directory and continues in place.

Two artifacts, and they must not be confused:

  * `state/iterNN.json` is `RunState.summary()` -- a lossy REPORT. Unchanged
    by this module.
  * `checkpoints/ckpt-NNNNN.json` is the CHECKPOINT -- everything needed to
    continue. Written here, read by nothing else.

WHAT THIS BUYS. Measured on Meta-World, with one batch job per whole search:
the LLM leg of an iteration is ~21 s and the RL leg is ~18 min, and a LIMEN
search runs 20+ h against a typical 24 h walltime. Without checkpoints, a key
rotation or a sustained 5xx at hour 18 raises `LLMError` and every hour of RL
compute is gone. There is no LLM conversation state to lose --
`bird/llm/anthropic_client.py` holds no message history and the Messages API is
stateless; every call is rebuilt from `RunState.dialogue`, which IS checkpointed.

THE FLOOR, and why it is a floor rather than a hope. `os.replace` after an
`fsync` is atomic against a torn write, but network and shared filesystems can
fail in ways atomicity does not cover (silently corrupted or stale data). So
every checkpoint
also carries a `digest` (sha256 over the body minus the digest field), and the
newest TWO are kept. That digest covers the JSON envelope only, so the spilled
`arrays/*.npy` -- which hold most of what a resumed run computes with -- are
fsynced like the envelope and verified against the sha256 their own filename
already asserts, in `_load_sidecar`. Integrity on the small part and none on
the bulk would be the worst of both: a torn sidecar that still `np.load`s
restores WRONG NUMBERS under an intact-looking checkpoint.
A checkpoint that will not parse, or whose digest does not
match, is REJECTED and named in `resume.jsonl` -- a silently skipped checkpoint
is indistinguishable from no checkpoint at all. The worst case is therefore *one
iteration behind*, never a crash on garbage and never a silent restart at zero.

THE CODEC IS STRICT ON PURPOSE. `encode()` raises `CheckpointEncodeError` on
anything it has no tag for; it never `str()`s the way `artifacts._json_default`
does. A component that stuffs a live object into `Candidate.meta` therefore
blows up at iteration 0 (second ~30) instead of at the resume at hour 18.
Decoding goes through a CLOSED allow-list of dataclasses -- never `importlib` on
a name out of the file and never `pickle`. That is not paranoia theatre: on a
shared filesystem with permissive directory modes, other users can write your
run directory. (The `pickle` in
`training.py::_write_payload` is fine -- private tmpdir, same interpreter,
lifetime of one fork. A checkpoint is another node, days later.)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import socket
import threading
import time
import traceback
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .state import ArchiveCell, CurriculumState, RunState, TreeNode
from .types import Candidate, CandidateReport, Preference, Selection, Trajectory, TrainResult

log = logging.getLogger("bird")

#: Bumped when the on-disk body changes shape. A file claiming a HIGHER version
#: is refused loudly rather than half-read.
SCHEMA_VERSION = 1

#: How long a lock heartbeat may go unrefreshed before the holder is presumed
#: dead and the directory may be stolen. Liveness is read from the lock, not
#: from `journal.jsonl`'s mtime (which ticks per stage event): a run can look
#: stalled by its journal while its lock is still live, and a refusal to steal
#: is the harmless answer.
RESUME_MIN_IDLE_S = 900

#: How often the heartbeat is rewritten, and it is a TIMER rather than a
#: checkpoint boundary. `Checkpointer.save` fires once per iteration, and one
#: iteration is ~18 min of RL on Meta-World (measured; a LIMEN iteration can
#: be ~50 min) -- so a heartbeat refreshed only there
#: ages far past `RESUME_MIN_IDLE_S` while its holder is perfectly healthy, and
#: `claim()` hands the directory to a second writer. That is the interleaving
#: the refusal message says it exists to prevent, arriving through the
#: mechanism meant to prevent it. 15x the margin, so a heartbeat can miss
#: fourteen beats on a wedged mount before the run reads dead.
HEARTBEAT_INTERVAL_S = 60

#: Arrays at or below this many elements go inline as base64; bigger ones spill
#: to a content-addressed `.npy` sidecar, so an unchanged trajectory is written
#: once rather than once per iteration.
INLINE_ARRAY_MAX = 1024

#: Rolling checkpoints kept. Two, so that a newest file which is complete but
#: semantically bad still leaves something to fall back to.
KEEP_CHECKPOINTS = 2

_RUNDIR_RE = re.compile(r"^.+-[0-9a-f]{6,}-\d{8}-\d{6}$")
_ARRAY_REF_RE = re.compile(r"arrays/[0-9a-f]+\.npy")


class CheckpointError(RuntimeError):
    """A checkpoint could not be used. Always loud, never swallowed."""


class ResumeLocked(CheckpointError):
    """Another live process holds this run directory."""


class CheckpointEncodeError(RuntimeError):
    """Something in the tree has no tag. Raised at iteration 0, not at hour 18."""

    def __init__(self, path_in_tree: str, type_name: str):
        super().__init__(f"no checkpoint encoding for {type_name} at {path_in_tree}")
        self.path_in_tree = path_in_tree
        self.type_name = type_name


# ==========================================================================
# Codec
# ==========================================================================

#: The CLOSED allow-list. Decoding never resolves a name outside this dict.
_DECODABLE: Dict[str, type] = {
    c.__name__: c for c in (Candidate, TrainResult, CandidateReport, Trajectory,
                            Preference, Selection, ArchiveCell, CurriculumState,
                            TreeNode, RunState)
}


class _ArraySpill:
    """Where ndarrays go, and the two counters the envelope reports."""

    def __init__(self, base: Optional[Path]):
        self.base = base
        self.written: set = set()
        self.dropped_trajectories = 0

    def put(self, arr: np.ndarray) -> Dict[str, Any]:
        buf = io.BytesIO()
        # `np.save` round-trips dtype, shape and order exactly, including a
        # non-contiguous slice; `tolist()` would silently upcast.
        np.save(buf, arr, allow_pickle=False)
        blob = buf.getvalue()
        if arr.size <= INLINE_ARRAY_MAX or self.base is None:
            return {"__ndarray__": base64.b64encode(blob).decode("ascii")}
        digest = hashlib.sha256(blob).hexdigest()[:16]
        rel = f"arrays/{digest}.npy"
        path = self.base / rel
        # Content-addressed, so an existing file of the RIGHT LENGTH is the same
        # file and skipping it is the whole point (an unchanged trajectory is
        # written once, not once per iteration). A file of the wrong length is
        # a torn write, and `not path.exists()` alone would leave it there
        # forever: every later checkpoint would name the same hash, find the
        # damaged sidecar, and be rejected by the load-side verification below.
        # One `stat` is what stops one bad write from poisoning the run.
        if not path.exists() or path.stat().st_size != len(blob):
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_bytes(path, blob)
        self.written.add(rel)
        return {"__ndarray_ref__": {"file": rel, "sha256_16": digest,
                                    "dtype": str(arr.dtype), "shape": list(arr.shape)}}


def _tagged(obj: Any, tag: str) -> bool:
    return isinstance(obj, dict) and len(obj) == 1 and tag in obj


def encode(obj: Any, spill: _ArraySpill, path: str = "$") -> Any:
    """Type-tagged, recursive, strict. Raises on anything with no tag."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        # `ArchiveCell.fitness` defaults to -inf and JSON has no literal for it.
        return float(obj) if math.isfinite(obj) else {"__float__": repr(float(obj))}
    if isinstance(obj, bytes):
        # MetaWorld's obs -> snapshot cache is keyed on `bytes`.
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, np.generic):
        return encode(obj.item(), spill, path)
    if isinstance(obj, np.ndarray):
        return spill.put(obj)
    if isinstance(obj, tuple):
        return {"__tuple__": [encode(v, spill, f"{path}[{i}]") for i, v in enumerate(obj)]}
    if isinstance(obj, list):
        return [encode(v, spill, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, dict):
        plain = all(isinstance(k, str) and not (k.startswith("__") and k.endswith("__"))
                    for k in obj)
        if plain:
            return {k: encode(v, spill, f"{path}.{k}") for k, v in obj.items()}
        # `RunState.archive` is keyed on a tuple; JSON cannot.
        return {"__map__": [[encode(k, spill, f"{path}.<key>"),
                             encode(v, spill, f"{path}[{k!r}]")] for k, v in obj.items()]}
    if is_dataclass(obj) and _DECODABLE.get(type(obj).__name__) is type(obj):
        name = type(obj).__name__
        out: Dict[str, Any] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if name == "TrainResult" and f.name in ("trajectories",
                                                    "store_trajectories"):
                # Dropped unconditionally and COUNTED. `RunDir.save_train_result`
                # already drops them from the artifact for the same reason. The
                # invariant this rests on is stated in `bird/state.py`: a carried
                # report is read for its scalar, its prose and its code -- never
                # for its rollouts. `state.trajectory_store` and
                # `state.preferences` DO keep theirs; those are carry slots and
                # they are the method. `store_trajectories` (CARD's dedicated
                # TPE pool) joins the drop for the same reasons plus size: §6
                # copies a passing pool INTO `state.trajectory_store` in the same
                # iteration, so a carried report encoding it again would write
                # the winner's 100 rollouts a second time per checkpoint -- and
                # the rewards/component lists are plain Python, which does not
                # dedupe the way ndarray sidecars do (`RunDir.save_train_result`
                # makes the same exclusion).
                spill.dropped_trajectories += len(value or ())
                out[f.name] = []
                continue
            out[f.name] = encode(value, spill, f"{path}.{f.name}")
        return {"__dataclass__": name, "fields": out}
    raise CheckpointEncodeError(path, type(obj).__name__)


def decode(obj: Any, base: Optional[Path] = None) -> Any:
    """Inverse of `encode`. Field-by-field and version tolerant."""
    if isinstance(obj, list):
        return [decode(v, base) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if _tagged(obj, "__tuple__"):
        return tuple(decode(v, base) for v in obj["__tuple__"])
    if _tagged(obj, "__map__"):
        return {decode(k, base): decode(v, base) for k, v in obj["__map__"]}
    if _tagged(obj, "__bytes__"):
        return base64.b64decode(obj["__bytes__"])
    if _tagged(obj, "__float__"):
        return float(obj["__float__"])
    if _tagged(obj, "__ndarray__"):
        return np.load(io.BytesIO(base64.b64decode(obj["__ndarray__"])), allow_pickle=False)
    if _tagged(obj, "__ndarray_ref__"):
        return _load_sidecar(obj["__ndarray_ref__"], base)
    if _tagged(obj, "__replay__"):
        return _decode_replay(obj["__replay__"], base)
    if "__dataclass__" in obj:
        name = obj["__dataclass__"]
        cls = _DECODABLE.get(name)
        if cls is None:
            raise CheckpointError(
                f"checkpoint names a type that is not on the decode allow-list: {name!r}")
        src = obj.get("fields") or {}
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(src) - known)
        if unknown:
            # Version tolerance in the other direction: a field this class no
            # longer has is dropped, loudly, rather than exploding in __init__.
            log.warning("checkpoint: %s carries field(s) %s this build does not know; dropped",
                        name, unknown)
        return cls(**{k: decode(v, base) for k, v in src.items() if k in known})
    return {k: decode(v, base) for k, v in obj.items()}


def _load_sidecar(ref: Dict[str, Any], base: Optional[Path]) -> np.ndarray:
    """Read a spilled array, and VERIFY it -- the envelope's digest cannot.

    `digest_of` hashes the JSON body, which holds only the sidecar's name,
    dtype and shape. Everything the resumed run actually computes with lives in
    the `.npy` file, and without this check a torn or mixed-block read that
    still parses would restore WRONG NUMBERS while the checkpoint reported
    itself intact and `degraded` stayed empty. The filename is already
    `sha256(blob)[:16]`, so the assertion is on disk and this recomputes it.
    The dtype/shape comparison is the residual case the hash cannot cover --
    an envelope REWRITTEN (with a fresh `digest`) to point a ref somewhere
    else, which on a world-writable run directory is exactly as available to
    another user as the decode allow-list assumes.

    Every failure is a `CheckpointError`, which is what makes the keep-2
    retention reachable: `np.load`'s own `EOFError` / `ValueError` are not
    caught by `load_plan`, so unconverted a corrupt sidecar would be FATAL
    while a missing one fell back cleanly -- the opposite of the stated floor.
    """
    path = (base or Path(".")) / ref["file"]
    if not path.exists():
        raise CheckpointError(f"checkpoint array sidecar missing: {path}")
    try:
        blob = path.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"checkpoint array sidecar unreadable: {path} ({exc})") from None
    want = ref.get("sha256_16") or Path(ref["file"]).stem
    got = hashlib.sha256(blob).hexdigest()[:16]
    if want != got:
        raise CheckpointError(
            f"checkpoint array sidecar {path.name} is corrupt: sha256 {got} != {want} "
            f"({len(blob)} bytes) -- the envelope's digest cannot see this")
    try:
        arr = np.load(io.BytesIO(blob), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - numpy raises EOFError/ValueError/OSError
        raise CheckpointError(
            f"checkpoint array sidecar {path.name} did not load "
            f"({type(exc).__name__}: {exc})") from None
    shape = ref.get("shape")
    dtype = ref.get("dtype")
    if (shape is not None and list(arr.shape) != list(shape)) or \
            (dtype is not None and str(arr.dtype) != str(dtype)):
        raise CheckpointError(
            f"checkpoint array sidecar {path.name} is not what the envelope named: "
            f"{arr.dtype}{list(arr.shape)} != {dtype}{list(shape or [])}")
    return arr


# -- replay buffers, columnar --------------------------------------------
#
# A replay buffer is up to 4096 `(obs, action, next_obs, done)` tuples. Encoded
# generically that is 16k base64 blobs; encoded columnar it is four arrays.


def _encode_replay(buf: Any, spill: _ArraySpill, path: str) -> Any:
    from .components.training import _ReplaySlice  # local: registry pulls in the world

    if isinstance(buf, _ReplaySlice):
        # Already columnar. `columnar: true` is what makes the round trip
        # type-preserving: decoding a `_ReplaySlice` back into 100,000 Python
        # tuples would be correct and would also be 80 MB and ~1.5 s, on the
        # resume path of the runs (`sb3`, 1M steps) this whole module exists for.
        return {"__replay__": {"n": len(buf), "columnar": True,
                               "obs": spill.put(np.asarray(buf.obs)),
                               "act": spill.put(np.asarray(buf.act)),
                               "next_obs": spill.put(np.asarray(buf.next_obs)),
                               "done": spill.put(np.asarray(buf.done))}}
    rows = list(buf or ())
    if not rows:
        return {"__replay__": {"n": 0}}
    try:
        if any(len(r) != 4 for r in rows):
            raise ValueError("not 4-tuples")
        # `int(r[1])` MUST NOT BE UNCONDITIONAL: on a continuous action space
        # `int(np.float64(0.37))` is 0 -- a silent, total loss of the action
        # column inside a `try` whose `except` never fires. Only take the int64
        # column when every action really is one.
        if any(np.ndim(r[1]) != 0 or not float(r[1]).is_integer() for r in rows):
            raise ValueError("non-integral actions")
        obs = np.asarray([np.asarray(r[0]) for r in rows])
        act = np.asarray([int(r[1]) for r in rows], dtype=np.int64)
        nxt = np.asarray([np.asarray(r[2]) for r in rows])
        done = np.asarray([bool(r[3]) for r in rows], dtype=bool)
        if obs.dtype == object or nxt.dtype == object:
            raise ValueError("ragged transitions")
    except Exception:  # noqa: BLE001 - fall back to the generic (correct) path
        return encode(rows, spill, path)
    return {"__replay__": {"n": len(rows), "obs": spill.put(obs), "act": spill.put(act),
                           "next_obs": spill.put(nxt), "done": spill.put(done)}}


def _decode_replay(blob: Dict[str, Any], base: Optional[Path]) -> Any:
    if not blob.get("n"):
        return []
    obs = decode(blob["obs"], base)
    act = decode(blob["act"], base)
    nxt = decode(blob["next_obs"], base)
    done = decode(blob["done"], base)
    if blob.get("columnar"):
        from .components.training import _ReplaySlice

        return _ReplaySlice(obs, act, nxt, np.asarray(done, dtype=bool))
    return [(obs[i], int(act[i]), nxt[i], bool(done[i])) for i in range(int(blob["n"]))]


# ==========================================================================
# Envelope
# ==========================================================================


def digest_of(body: Dict[str, Any]) -> str:
    payload = {k: v for k, v in body.items() if k != "digest"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _ckpt_dir(rundir: Any) -> Path:
    path = rundir.path if hasattr(rundir, "path") else Path(rundir)
    return Path(path) / "checkpoints"


def _atomic_bytes(path: Path, blob: bytes) -> None:
    """fsync, then `os.replace`, then fsync the directory.

    Used for the `.npy` SIDECARS as well as the envelope. A bare
    `write_bytes` + `os.replace` for them would be the asymmetry that matters:
    integrity paid for on the small part and skipped on the bulk -- a spilled
    replay buffer or policy weight array -- on exactly the mounts this
    module's docstring says it does not trust.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    # The rename itself has to reach the platter, or a node that loses power
    # after `os.replace` comes back with the directory entry missing.
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _atomic_json(path: Path, blob: str) -> None:
    _atomic_bytes(path, blob.encode())


def write(rundir: Any, body: Dict[str, Any]) -> Optional[Path]:
    """Atomic, fsynced, gc'd -- and it NEVER propagates an exception.

    Killing a 20-hour search because a checkpoint would not serialise is
    strictly worse than not being able to resume it. On failure the caller gets
    `None`, the reason lands in `checkpoints/last_error.txt`, and the search
    carries on un-resumable from that point.
    """
    d = _ckpt_dir(rundir)
    try:
        d.mkdir(parents=True, exist_ok=True)
        body = dict(body)
        body["digest"] = digest_of(body)
        final = d / f"ckpt-{int(body['seq']):05d}.json"
        # NOT `sort_keys`: a dict's INSERTION ORDER is data here.
        # `Candidate.weights` is rendered straight into the next prompt
        # (`weights: {"proximity": 1.781, "momentum": 1.306}`), so sorting it on
        # the way to disk changes the prompt the resumed leg sends and therefore
        # the program the model writes back. Measured: that one word cost
        # `rda` under the tester profile its byte-identical resume. The DIGEST is computed with
        # `sort_keys=True`, which is what makes it order-independent and the
        # file order free.
        _atomic_json(final, json.dumps(body))
        _gc(d)                                   # only AFTER the fsync
        return final
    except Exception:  # noqa: BLE001
        _record_write_failure(d)
        return None


def budget_blob(budget: Any) -> Dict[str, Any]:
    """A budget snapshot in exactly the shape `Budget.restore` reads back."""
    out = dict(budget.snapshot())
    out["_prior_elapsed_s"] = round(float(budget.wallclock_s), 3)
    return out


def write_restart(rundir: Any, restart: int, state: RunState,
                  budget: Optional[Dict[str, Any]] = None) -> Optional[Path]:
    """A completed restart is immutable, so it gets its own write-once file.

    Inlining every finished restart into the rolling checkpoint would rewrite
    `n_restarts x ~700 KB` every 18 minutes over a network filesystem for data
    that cannot change.

    It also carries the BUDGET AS OF ITS OWN END, which is the only place that
    number can be recovered from. The rolling checkpoint's budget is a running
    total; if a restart file is later unreadable, its restart is re-executed
    and that total is an over-count. `_plan_from` rewinds to the snapshot here.
    """
    d = _ckpt_dir(rundir)
    try:
        d.mkdir(parents=True, exist_ok=True)
        spill = _ArraySpill(d)
        body = {"schema": SCHEMA_VERSION, "restart": int(restart),
                "budget": dict(budget or {}),
                "state": encode(state, spill, "$.state")}
        body["digest"] = digest_of(body)
        _atomic_json(d / f"restart-{int(restart):02d}.json", json.dumps(body))
        return d / f"restart-{int(restart):02d}.json"
    except Exception:  # noqa: BLE001
        _record_write_failure(d)
        return None


def load_restart(rundir: Any, restart: int) -> Tuple[RunState, Dict[str, Any]]:
    """`(state, budget-at-its-end)`. The budget may be `{}` on an older file."""
    d = _ckpt_dir(rundir)
    path = d / f"restart-{int(restart):02d}.json"
    body = _read_verified(path)
    return decode(body["state"], d), dict(body.get("budget") or {})


def _record_write_failure(d: Path) -> None:
    log.warning("checkpoint write failed; this search is not resumable from here",
                exc_info=True)
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "last_error.txt").write_text(traceback.format_exc())
    except Exception:  # pragma: no cover - the mount itself is gone
        pass


def _gc(d: Path) -> None:
    ckpts = sorted(d.glob("ckpt-*.json"))
    for stale in ckpts[:-KEEP_CHECKPOINTS]:
        stale.unlink(missing_ok=True)
    live = set()
    for keep in sorted(d.glob("ckpt-*.json")) + sorted(d.glob("restart-*.json")):
        try:
            live.update(_ARRAY_REF_RE.findall(keep.read_text()))
        except OSError:  # pragma: no cover
            return
    for arr in (d / "arrays").glob("*.npy"):
        if f"arrays/{arr.name}" not in live:
            arr.unlink(missing_ok=True)


def _read_verified(path: Path) -> Dict[str, Any]:
    """Parse, version-check, digest-check. Every rejection is an exception with
    a reason a human can act on."""
    try:
        text = path.read_text()
    except OSError as exc:
        raise CheckpointError(f"{path.name}: unreadable ({exc})") from None
    try:
        body = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CheckpointError(f"{path.name}: not valid JSON ({exc}) -- truncated write") from None
    if not isinstance(body, dict):
        raise CheckpointError(f"{path.name}: not a checkpoint object")
    schema = body.get("schema")
    if schema != SCHEMA_VERSION:
        raise CheckpointError(
            f"{path.name}: schema {schema!r} but this build writes {SCHEMA_VERSION}; "
            "refusing to half-read it")
    want = body.get("digest")
    got = digest_of(body)
    if want != got:
        raise CheckpointError(f"{path.name}: digest mismatch ({want} != {got}) -- corrupt")
    return body


# ==========================================================================
# The cross-iteration stores
# ==========================================================================
#
# `_POLICY_STORE` / `_REPLAY_STORE` are module-level dicts of numpy arrays in
# `bird/components/training.py`; `state.policy_ref` / `.replay_ref` are keys
# into them and they die with the process. Only the entries REACHABLE from the
# checkpointed state are persisted -- 1-3 of them, never the store's 64.
# `loop.carry` has already declared that only `state.policy_ref` /
# `state.replay_ref` survive an iteration, so persisting more would persist
# state the algorithm calls dead and would perturb `_store`'s insertion-ordered
# eviction.
#
# THE SET DOES NOT DEPEND ON THE BACKEND. `reachable_refs` takes only `state`:
# every backend, `sb3` included, writes `result.policy_ref` / `.replay_ref`
# (otherwise `warm_start_from_best` and `secondary_replay_buffer` would be
# no-ops on it), so a per-backend branch here would be wrong.


def reachable_refs(state: Optional[RunState]) -> Tuple[set, set]:
    """(policy refs, replay refs) the restored state could ever dereference."""
    pol: set = set()
    rep: set = set()
    if state is None:
        return pol, rep
    if state.policy_ref:
        pol.add(state.policy_ref)
    if state.replay_ref:
        rep.add(state.replay_ref)
    # `returned` is a record, not a carry slot, so it can outlive the slot it
    # was copied from (`chain_end` after `op_restart` dropped `latest`; the
    # "never un-known" branch in `update()`). Walked here so a resumed leg's
    # return cannot dangle a ref to a policy the store no longer holds.
    reports = [state.best, state.latest, state.iteration_best, state.returned]
    reports += [cell.report for cell in (state.archive or {}).values()]
    for report in reports:
        result = getattr(report, "result", None)
        if result is None:
            continue
        if getattr(result, "policy_ref", None):
            pol.add(result.policy_ref)
        if getattr(result, "replay_ref", None):
            rep.add(result.replay_ref)
    return pol, rep


def export_stores(state: Optional[RunState], spill: _ArraySpill):
    """Encode the reachable store entries. Returns (policy, replay, degraded)."""
    from .components import training  # local: registry imports pull in the world

    pol_refs, rep_refs = reachable_refs(state)
    degraded: List[Dict[str, Any]] = []
    policy: Dict[str, Any] = {}
    replay: Dict[str, Any] = {}
    for ref in sorted(pol_refs):
        if ref in training._POLICY_STORE:
            policy[ref] = encode(training._POLICY_STORE[ref], spill, f"$.policy[{ref}]")
        else:
            degraded.append({"slot": "policy_ref", "ref": ref,
                             "reason": "not in _POLICY_STORE (evicted or never written)"})
    for ref in sorted(rep_refs):
        if ref in training._REPLAY_STORE:
            replay[ref] = _encode_replay(training._REPLAY_STORE[ref], spill, f"$.replay[{ref}]")
        else:
            degraded.append({"slot": "replay_ref", "ref": ref,
                             "reason": "not in _REPLAY_STORE (evicted or never written)"})
    return policy, replay, degraded


def import_stores(policy: Dict[str, Any], replay: Dict[str, Any]) -> None:
    """Replay decoded entries into the live stores, in key order.

    Order matters for the same reason `_merge_child` says it does: `_store`
    evicts oldest-first, so a deterministic insertion order keeps the eviction
    sequence a resumed run sees identical to the one it would have had.
    """
    from .components import training

    for key in sorted(policy):
        training._store(training._POLICY_STORE, key, policy[key])
    for key in sorted(replay):
        training._store(training._REPLAY_STORE, key, replay[key])


# ==========================================================================
# The plan
# ==========================================================================


@dataclass
class ResumePlan:
    path: Path
    seq: int
    phase: str
    restart: int
    next_iteration: int
    retries: int
    rng: tuple
    counters: dict
    budget: dict
    prior_elapsed_s: float
    state: Optional[RunState]
    completed_restarts: List[int]
    restart_states: List[RunState]
    pre_phases_done: List[str]
    policy_store: dict
    replay_store: dict
    env_states: Any
    degraded: List[dict]
    dropped: List[dict]
    rejected: List[dict]
    loop_start_budget: Optional[dict] = None
    journal_events: int = 0
    discarded_trainings: int = 0
    discarded_env_steps: int = 0


def load_plan(rundir: Any, strict: bool = True) -> Optional[ResumePlan]:
    """Newest usable checkpoint, or None. Every rejection is recorded.

    A checkpoint silently skipped because its digest failed looks identical to
    no checkpoint at all, so each rejection goes to `resume.jsonl` AND to the
    log, naming the file and the reason.
    """
    root = Path(rundir.path if hasattr(rundir, "path") else rundir)
    d = root / "checkpoints"
    rejected: List[Dict[str, Any]] = []
    # `.tmp` residue is never globbed: a SIGKILL mid-write must be
    # indistinguishable from "no new checkpoint".
    for path in sorted(d.glob("ckpt-*.json"), reverse=True):
        try:
            body = _read_verified(path)
        except CheckpointError as exc:
            log.warning("checkpoint rejected: %s", exc)
            rejected.append({"file": path.name, "reason": str(exc)})
            continue
        try:
            plan = _plan_from(root, d, body, rejected)
        except CheckpointError as exc:
            log.warning("checkpoint rejected: %s", exc)
            rejected.append({"file": path.name, "reason": str(exc)})
            continue
        for rec in rejected:
            append_resume_record(root, {"event": "checkpoint_rejected", **rec})
        if plan.degraded and strict:
            append_resume_record(root, {"event": "refused_degraded",
                                        "seq": plan.seq, "degraded": plan.degraded})
            raise CheckpointError(
                "checkpoint {} lost method-defining state and this run refuses a "
                "degraded resume:\n  - {}\nTo resume anyway, either pass "
                "`--resume-degraded` once (a flag, so it cannot move the config hash) "
                "or submit the search with `loop.resume_from: auto_degraded` from the "
                "start. Either way the loss is counted in `budget.resume_degradations` "
                "and named in result.json.".format(
                    plan.seq, "\n  - ".join(
                        f"{g.get('slot')}: {g.get('reason')}" for g in plan.degraded)))
        return plan
    for rec in rejected:
        append_resume_record(root, {"event": "checkpoint_rejected", **rec})
    append_resume_record(root, {"event": "no_candidate",
                                "reason": "no usable checkpoint", "rejected": rejected})
    return None


def _plan_from(root: Path, d: Path, body: Dict[str, Any],
               rejected: List[Dict[str, Any]]) -> ResumePlan:
    degraded = list(body.get("degraded") or [])
    state = decode(body["state"], d) if body.get("state") is not None else None
    policy = {k: decode(v, d) for k, v in (body.get("policy_store") or {}).items()}
    replay = {k: decode(v, d) for k, v in (body.get("replay_store") or {}).items()}
    env_states = decode(body["env_states"], d) if body.get("env_states") is not None else None

    # A ref with no payload must become None, never a stale key: a key that
    # `_POLICY_STORE.get()` misses is exactly the shape that reads as working.
    degraded += _null_dead_refs(state, policy, replay)

    declared = [int(k) for k in (body.get("completed_restarts") or [])]
    restart_states: List[RunState] = []
    restart_budgets: List[Dict[str, Any]] = []
    for k in declared:
        try:
            state_k, budget_k = load_restart(root, k)
        except CheckpointError as exc:
            degraded.append({"slot": f"restart-{k:02d}", "reason": str(exc)})
            break
        restart_states.append(state_k)
        restart_budgets.append(budget_k)
    completed = declared[:len(restart_states)]
    lost = declared[len(completed):]

    rng = body.get("rng")
    counters = decode(body.get("counters") or {}, d)
    loop_start = body.get("loop_start_budget")
    budget = dict(body.get("budget") or {})
    dropped = list(body.get("dropped") or [])
    if lost or body.get("restart_gap"):
        # A restart that cannot be restored is RE-EXECUTED, and the rolling
        # checkpoint's budget already holds what it spent. Restoring that total
        # would book the same trainings, env steps and LLM calls twice --
        # against `budget.max_*`, and in the cost column, which is a
        # first-class output. `Budget.restore`'s own docstring states the
        # invariant it would break ("iterations 0..k are never re-executed, so
        # nothing re-records them"), so the answer is to rewind rather than to
        # explain. Two ways in: a restart file that will not load (`lost`), and
        # a write that failed at the time (`restart_gap`, which stopped the
        # prefix there and left the budget running on past it).
        #
        # The wind-back point is the end of the last restart still readable,
        # or -- when there is none -- the total as the loop began, which is the
        # only reason `loop_start_budget` is in the envelope.
        rewind = restart_budgets[-1] if restart_budgets else loop_start
        if not rewind:
            raise CheckpointError(
                "restart state is missing and this checkpoint carries no budget to rewind "
                "to, so resuming would double-count what is about to be re-executed")
        budget = dict(rewind)
        at = ("restart-%02d" % completed[-1]) if completed else "the start of the loop"
        # `dropped`, NOT `degraded`. Re-executing a restart loses no
        # method-defining state -- it recomputes it -- and `degraded` is the
        # list strict `auto` REFUSES on. A swallowed write failure must not
        # become a refusal to resume at hour 18.
        dropped.append({"what": "budget rewound to " + at,
                        "why": f"lost={lost} gap={bool(body.get('restart_gap'))}; "
                               "those restarts are re-executed and must not be "
                               "counted twice"})
        if state is not None and int(body.get("restart", 0)) != len(completed):
            # `_execute` resumes at `len(restart_states)` and adopts `plan` only
            # when `plan.restart` matches, so the in-flight restart's state is
            # about to be thrown away. Named, because an artifact listing only
            # the unreadable restart file reads as if one restart were lost
            # when an iteration of another one went with it.
            dropped.append({
                "what": f"the in-flight state of restart {int(body.get('restart', 0))}",
                "why": "unreachable once the earlier restarts are re-executed"})
    prior = float(budget.pop("_prior_elapsed_s", 0.0) or 0.0)

    plan = ResumePlan(
        path=root, seq=int(body.get("seq", 0)), phase=str(body.get("phase", "")),
        restart=int(body.get("restart", 0)),
        next_iteration=int(body.get("next_iteration", 0)),
        retries=int(body.get("retries", 0)),
        rng=decode(rng, d), counters=counters, budget=budget, prior_elapsed_s=prior,
        state=state, completed_restarts=completed, restart_states=restart_states,
        pre_phases_done=list(body.get("pre_phases_done") or []),
        policy_store=policy, replay_store=replay, env_states=env_states,
        degraded=degraded, dropped=dropped, rejected=rejected,
        loop_start_budget=dict(loop_start) if loop_start else None,
        journal_events=int(body.get("journal_events", 0)),
    )
    plan.discarded_trainings, plan.discarded_env_steps = _discarded(root, plan.journal_events)
    return plan


def _null_dead_refs(state: Optional[RunState], policy: dict, replay: dict) -> List[dict]:
    """Set every unbacked ref to None -- in `RunState` AND in each restored
    `TrainResult` -- and say so.

    Nulling at the source is also what makes RAPP's `incumbent_best` and
    `final_retrain` take their existing `None` branch instead of dereferencing a
    dead key. Enumerating the phases that touch the store would be the hidden
    branch this repo bans.
    """
    if state is None:
        return []
    out: List[dict] = []

    def clear(holder: Any, attr: str, have: dict, where: str) -> None:
        ref = getattr(holder, attr, None)
        if ref and ref not in have:
            setattr(holder, attr, None)
            out.append({"slot": attr, "ref": ref, "where": where,
                        "reason": "no payload in the checkpoint; restored as None"})

    clear(state, "policy_ref", policy, "RunState")
    clear(state, "replay_ref", replay, "RunState")
    reports = [("best", state.best), ("latest", state.latest),
               ("iteration_best", state.iteration_best), ("returned", state.returned)]
    reports += [(f"archive[{k!r}]", c.report) for k, c in (state.archive or {}).items()]
    for where, report in reports:
        result = getattr(report, "result", None)
        if result is None:
            continue
        clear(result, "policy_ref", policy, where)
        clear(result, "replay_ref", replay, where)
    return out


def degradation_count(degraded: List[dict]) -> int:
    """`budget.resume_degradations`, which `budget.py` defines as ONE PER
    DROPPED SLOT -- so it counts distinct losses, not distinct sentences.

    One evicted `policy_ref` produces up to four entries: one written by
    `export_stores` at checkpoint time ("not in _POLICY_STORE"), and one per
    holder from `_null_dead_refs` at load time ("no payload ... restored as
    None"), and `rda`'s warm start reaches the same ref through both `best` and
    `latest`. Every one of those sentences is worth logging -- they say where
    the ref was reachable from -- and counting them all inflates by 4x a column
    a cross-method table is meant to filter on.
    """
    return len({(g.get("slot"), g.get("ref")) for g in degraded})


def _discarded(root: Path, journal_events: int) -> Tuple[int, int]:
    """Exactly what the iteration in flight spent after the last checkpoint.

    `bird.py`'s `train` event already carries `trained`, `env_steps` and
    `seeds`, so the two RL counters are exact for the leg that died. There is
    no journal event carrying TOKENS (`llm/base.py` records those into the
    Budget, not the journal), so LLM spend is NOT recoverable and is reported
    instead as a bound: `result.json["budget_unaccounted_iterations"]`.

    `journal_events` is a line COUNT rather than a timestamp on purpose -- the
    checkpoint body carries no clock reading of any kind.
    """
    path = root / "journal.jsonl"
    if not path.exists() or journal_events <= 0:
        return 0, 0
    trainings = 0
    steps = 0
    try:
        lines = path.read_text().splitlines()
    except OSError:  # pragma: no cover
        return 0, 0
    for line in lines[journal_events:]:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("stage") != "train" or not rec.get("trained"):
            continue
        trainings += int(rec.get("seeds") or 0)
        steps += int(rec.get("env_steps") or 0)
    return trainings, steps


# ==========================================================================
# Adoption
# ==========================================================================


def find_adoptable(out_root: Path, name: str, cfg_hash: str) -> Optional[Path]:
    """The newest unfinished directory this exact config could continue.

    Nothing found is NOT an error. The first submission and every requeue use
    the identical submit line, and the operator never makes a decision at 3 a.m.
    """
    pattern = f"{name}-{cfg_hash}-*"
    root = Path(out_root)
    if not root.is_dir():
        return None
    for path in sorted(root.glob(pattern), key=lambda q: q.name, reverse=True):
        if not _RUNDIR_RE.match(path.name):
            continue
        if (path / "result.json").exists():
            continue                                     # that search FINISHED
        status = _read_json(path / "status.json") or {}
        if status.get("status") == "ok":
            continue
        if not any((path / "checkpoints").glob("ckpt-*.json")):
            continue
        return path
    return None


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def append_resume_record(root: Any, record: Dict[str, Any]) -> None:
    """Append-only `resume.jsonl`: every attempt, every rejection, every steal."""
    path = Path(root.path if hasattr(root, "path") else root) / "resume.jsonl"
    try:
        with open(path, "a", buffering=1) as fh:
            fh.write(json.dumps({"at": round(time.time(), 3), **record}, default=str) + "\n")
    except OSError:  # pragma: no cover
        log.warning("could not append to %s", path)


# ==========================================================================
# The lock (one mechanism, two jobs)
# ==========================================================================
#
# A pure lock wedges after the walltime kill that is the whole use case; a pure
# heartbeat races two live writers. The lock file is REWRITTEN on a timer, so it
# is also the heartbeat, and one mechanism answers both questions honestly.
#
# ON A TIMER, and that word is the whole correctness argument. Rewriting it at
# every checkpoint reads as often enough and is not: a checkpoint is an
# ITERATION boundary and a Meta-World iteration can take hours, so a live
# holder would age past `RESUME_MIN_IDLE_S` in the middle of every one and any
# second process could steal the directory out from under it. Nothing
# Python of ours runs during an 18-minute `learn()` call, so only a thread can
# beat through it.


@dataclass
class Lock:
    path: Path
    stolen_from: Optional[dict] = None
    beat: Optional["_Beater"] = None
    #: Raised by `release()` BEFORE it touches the file. `heartbeat()` re-reads
    #: it immediately before writing, which is what catches a beat that READ
    #: the file before the unlink -- the one case no file check can see.
    released: bool = False


def _lock_payload() -> Dict[str, Any]:
    now = round(time.time(), 3)
    return {"pid": os.getpid(), "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            "taken": now, "heartbeat": now}


def _ours(holder: Optional[dict]) -> bool:
    """The identity check for "this lock is ours": pid AND host.

    pid alone is not an identity on a shared mount -- pids are per-node and
    run dirs are not, so a holder on another node can carry our pid without
    having stolen anything. Acting on it (a beat stamping it, `release()`
    unlinking it) would be one search operating another's lock."""
    return bool(holder) and holder.get("pid") == os.getpid() \
        and holder.get("host") == socket.gethostname()


def claim(rundir: Any, force: bool = False) -> Optional[Lock]:
    """Take `<rundir>/checkpoints/resume.lock`, or refuse with `ResumeLocked`.

    A lock whose heartbeat is older than `RESUME_MIN_IDLE_S` belongs to a
    holder presumed dead. For a SEARCH that is the resume seam working as
    designed -- the checkpoint is the record and the next leg continues it --
    so the stale lock is stolen and the steal is recorded.
    """
    root = Path(rundir.path if hasattr(rundir, "path") else rundir)
    d = root / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "resume.lock"
    payload = _lock_payload()
    try:
        # O_EXCL create: atomic on local filesystems and on NFSv4.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(payload))
        return Lock(path)
    except FileExistsError:
        pass
    holder = _read_json(path) or {}
    age = time.time() - float(holder.get("heartbeat") or 0.0)
    if not force and age < RESUME_MIN_IDLE_S:
        raise ResumeLocked(
            f"{root.name} is held by pid {holder.get('pid')} on {holder.get('host')} "
            f"(job {holder.get('slurm_job_id') or '-'}"
            f".{holder.get('slurm_array_task_id') or '-'}), heartbeat {age:.0f}s ago. "
            f"Refusing; a second writer would interleave two searches into one "
            f"directory. Wait {RESUME_MIN_IDLE_S - age:.0f}s or pass --resume-force.")
    payload["stolen_from"] = holder
    path.write_text(json.dumps(payload))
    reason = "forced" if force else f"heartbeat {age:.0f}s > {RESUME_MIN_IDLE_S}s"
    log.warning("stole the resume lock on %s (%s); previous holder %s", root.name, reason, holder)
    append_resume_record(root, {"event": "lock_stolen", "reason": reason, "holder": holder})
    return Lock(path, stolen_from=holder)


def heartbeat(lock: Optional[Lock]) -> None:
    """Rewrite the holder's `heartbeat`. ATOMIC, because someone reads it.

    `claim()` in another process reads this file and treats an unparseable one
    as an absent holder, i.e. as licence to steal -- so a reader catching a
    half-written lock would be a steal caused by the mechanism that proves the
    holder is alive.

    REFRESHES ONLY A LOCK THAT IS STILL OURS, and refuses in both directions of
    "not ours". Reading `_read_json(path) or _lock_payload()` and writing that
    back would make two separate false claims:

      absent   `release()` unlinks the file, so a beat that lands after it
               would RECREATE the lock from scratch -- a directory nobody holds,
               reading held, for the whole of `RESUME_MIN_IDLE_S`. The next
               requeue of a search that finished cleanly would then get
               `ResumeLocked` and need `--resume-force`, which is exactly the
               interleaving-refusal being spent on a phantom.
      foreign  after a steal the file carries the THIEF's payload, and stamping
               our clock onto it goes on certifying that holder as alive long
               after it is not -- the beater outliving what it vouches for.

    `_atomic_json` replaces rather than truncates, so the file is never
    momentarily absent for an honest reason; absence means gone.

    "Ours" is `_ours` -- pid AND host -- because pid alone collides across
    nodes on a shared mount. And both file checks see only a beat whose READ
    lands after the unlink or the steal: the beat `stop()`'s bounded join gave
    up on read BEFORE the unlink, still holds a payload that looks ours, and
    would write it back. `lock.released` is re-checked immediately before the
    write for exactly that beat. It is an in-process flag, not a second
    existence probe, on purpose: `release()` raises it before it touches the
    file, so the ordering is real, and a flag cannot stall on the sick mount
    that made the join time out in the first place. What remains open is a
    write already inside `_atomic_json` when the flag goes up -- one rename
    wide, and what it costs on loss is the phantom (`--resume-force`), not
    corruption.
    """
    if lock is None:
        return
    try:
        holder = _read_json(lock.path)
        if not _ours(holder):
            return
        holder["heartbeat"] = round(time.time(), 3)
        if lock.released:
            return
        _atomic_json(lock.path, json.dumps(holder))
    except OSError:  # pragma: no cover
        pass


#: How long `_Beater.stop` waits for the beat in flight. One beat is a read
#: plus an atomic rename of a few hundred bytes; anything approaching this
#: means the mount is in trouble, and a teardown path that blocks on a sick
#: mount is a task that holds its whole allocation and writes nothing.
_BEATER_JOIN_S = 5.0


class _Beater(threading.Thread):
    """One daemon thread whose entire job is `heartbeat()` on a timer.

    A thread, in a repo that refuses them for training -- and the reasons given
    there do not reach here. It shares nothing with the search: no `ctx`, no
    store, no RNG, no config; it writes one small file and reads the clock. It
    is the only writer of that file on the search's own path
    (`Checkpointer.save` does not beat). It is NOT the only writer full stop
    -- `release()` unlinks it -- and `stop()` below is where those two are
    ordered. `os.fork` copies only the calling thread, so a
    `train.candidate_parallelism: parallel` worker inherits the `Lock` object
    and no beater, and never touches the lock.
    """

    def __init__(self, lock: Lock, interval: float):
        super().__init__(name="bird-resume-heartbeat", daemon=True)
        self.lock = lock
        self.interval = float(interval)
        # NOT `self._stop`: `threading.Thread._stop` is a METHOD, and `join()`
        # calls it. Shadowing it with an Event would make every join below die
        # with `'Event' object is not callable`.
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            heartbeat(self.lock)

    def stop(self, timeout: float = _BEATER_JOIN_S) -> None:
        """Set the flag AND wait for the beat in flight to finish.

        Setting the flag alone only stops the NEXT beat. A beat already past
        `self._halt.wait()` still runs, and `release()` -- which calls this and
        then unlinks -- would race its own beater: the unlink lands first and
        the beat puts the file back. Observed on a loaded CI runner, not merely
        reasoned about.

        `heartbeat()` closes the same hole from the other side, and all three
        guards are kept: this join makes the ordering true, the absent-file
        check survives a beat that READS after the unlink, and `lock.released`
        survives the one that read before it -- the beat this bounded join
        gave up on, which no file check can refuse. Bounded,
        because a teardown path must not be able to hold the process: the
        thread is a daemon and one beat is a small atomic write.
        """
        self._halt.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout)


def start_heartbeat(lock: Optional[Lock],
                    interval: float = HEARTBEAT_INTERVAL_S) -> Optional[Lock]:
    """Begin beating. A no-op without a lock, and idempotent."""
    if lock is None or lock.beat is not None:
        return lock
    lock.beat = _Beater(lock, interval)
    lock.beat.start()
    return lock


def stop_heartbeat(lock: Optional[Lock]) -> None:
    if lock is None or lock.beat is None:
        return
    lock.beat.stop()
    lock.beat = None


def release(lock: Optional[Lock]) -> None:
    if lock is None:
        return
    # The flag FIRST, before the join and before any file operation: a beat
    # that already read a payload that looks ours -- the one the bounded join
    # below may give up on -- re-checks it in this process, where nothing can
    # stall. Raised after the unlink it would leave open the exact window it
    # exists to close.
    lock.released = True
    stop_heartbeat(lock)
    try:
        if _ours(_read_json(lock.path)):
            lock.path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        pass


# ==========================================================================
# Writing one checkpoint
# ==========================================================================


class Checkpointer:
    """Assembles the envelope. A no-op when `loop.resume_from` is null."""

    def __init__(self, rundir: Any, cfg: Any, lock: Optional[Lock] = None,
                 enabled: bool = False, seq: int = 0):
        self.rundir = rundir
        self.cfg = cfg
        self.lock = lock
        self.enabled = bool(enabled) and rundir is not None
        self.seq = int(seq)
        self.completed_restarts: List[int] = []
        self.pre_phases_done: List[str] = []
        self.loop_start_budget: Optional[Dict[str, Any]] = None
        self._restart_gap = False

    # -- the one public call --

    def save(self, ctx: Any, phase: str, state: Optional[RunState] = None,
             restart: int = 0, next_iteration: int = 0, retries: int = 0) -> Optional[Path]:
        if not self.enabled:
            return None
        self.seq += 1
        try:
            body = self._body(ctx, phase, state, restart, next_iteration, retries)
        except Exception as exc:  # noqa: BLE001 - never kill the search
            self.seq -= 1
            self._failed(ctx, exc)
            return None
        path = write(self.rundir, body)
        if path is None:
            self._failed(ctx, None)
            return None
        return path

    def record_restart(self, restart: int, path: Optional[Path]) -> None:
        """Count a finished restart -- only if its file reached disk, and only
        while the prefix is unbroken.

        `write_restart` returns `None` on failure and never raises (killing a
        20-hour search over a serialisation fault is worse than not resuming
        it), so appending regardless would make the next rolling checkpoint
        name a `restart-NN.json` that does not exist. Under strict
        `auto` that turns a swallowed write failure into a REFUSAL to resume at
        hour 18, and it takes the correctly-written restarts after it down too.

        Stopping at the first gap rather than skipping it is not fastidiousness:
        `_plan_from` maps `restart_states` POSITIONALLY, so a recorded `[1, 2]`
        would come back as restarts 0 and 1 and relabel the whole search.
        """
        if self._restart_gap or (self.enabled and path is None):
            if not self._restart_gap:
                log.warning("restart %d was not checkpointed; no later restart will be "
                            "recorded either, because the resumable prefix must stay "
                            "contiguous", restart)
            self._restart_gap = True
            return
        self.completed_restarts.append(restart)

    # -- internals --

    def _failed(self, ctx: Any, exc: Optional[BaseException]) -> None:
        d = _ckpt_dir(self.rundir)
        if exc is not None:
            log.warning("checkpoint could not be assembled (%s: %s); the search continues "
                        "but is not resumable from here", type(exc).__name__, exc)
            try:
                d.mkdir(parents=True, exist_ok=True)
                (d / "last_error.txt").write_text(
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            except Exception:  # pragma: no cover
                pass
        ctx.event("checkpoint_failed",
                  error=f"{type(exc).__name__}: {exc}" if exc is not None else "write failed")

    def _body(self, ctx: Any, phase: str, state: Optional[RunState],
              restart: int, next_iteration: int, retries: int) -> Dict[str, Any]:
        d = _ckpt_dir(self.rundir)
        d.mkdir(parents=True, exist_ok=True)
        spill = _ArraySpill(d)
        degraded: List[Dict[str, Any]] = []

        enc_state = None
        if state is not None:
            enc_state, deg = _encode_state(state, spill)
            degraded += deg
        policy, replay, deg = export_stores(state, spill)
        degraded += deg
        env_states, deg = _encode_env_states(ctx, state, spill)
        degraded += deg

        budget = budget_blob(ctx.budget)

        dropped: List[Dict[str, Any]] = []
        if spill.dropped_trajectories:
            dropped.append({"what": "TrainResult.trajectories",
                            "n": spill.dropped_trajectories})

        if degraded:
            log.warning("checkpoint %d is NOT strictly resumable under "
                        "`loop.resume_from: auto`: %s", self.seq,
                        "; ".join(f"{g.get('slot')}: {g.get('reason')}" for g in degraded))

        return {
            "schema": SCHEMA_VERSION,
            "seq": self.seq,
            "phase": phase,
            "name": self.cfg["name"],
            "config_hash": self.cfg.hash(),
            "run_dir": Path(self.rundir.path).name,
            "restart": int(restart),
            "next_restart": len(self.completed_restarts),
            "next_iteration": int(next_iteration),
            "retries": int(retries),
            "completed_restarts": list(self.completed_restarts),
            # `completed_restarts` is a PREFIX, so it cannot say "and one after
            # this failed to write". This flag can, and `_plan_from` needs it:
            # the budget kept running through a restart the prefix stops short
            # of, and that restart is about to be re-executed.
            "restart_gap": bool(self._restart_gap),
            # The only recoverable budget for "before restart 0", and therefore
            # the wind-back point when EVERY `restart-NN.json` is unreadable.
            "loop_start_budget": dict(self.loop_start_budget)
            if self.loop_start_budget else None,
            "pre_phases_done": list(self.pre_phases_done),
            "rng": encode(ctx.rng.getstate(), spill, "$.rng"),
            "counters": encode(ctx.counters, spill, "$.counters"),
            "budget": budget,
            "state": enc_state,
            "policy_store": policy,
            "replay_store": replay,
            "env_states": env_states,
            "journal_events": int(getattr(self.rundir, "n_events", 0)),
            "dropped": dropped,
            "degraded": degraded,
        }


def _encode_state(state: RunState, spill: _ArraySpill):
    """Field by field, so the two tolerant slots can degrade without taking the
    whole checkpoint with them.

    Everything else that fails to encode is a hard `CheckpointEncodeError` and
    the write is abandoned. Silent partial omission is not permitted.
    """
    degraded: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {}
    for f in fields(state):
        value = getattr(state, f.name)
        try:
            out[f.name] = encode(value, spill, f"$.state.{f.name}")
        except CheckpointEncodeError as exc:
            if f.name not in ("trajectory_store", "preferences"):
                raise
            out[f.name] = [] if isinstance(value, list) else None
            degraded.append({"slot": f.name, "reason": f"could not encode ({exc})",
                             "n": len(value or ())})
    return {"__dataclass__": "RunState", "fields": out}, degraded


def _encode_env_states(ctx: Any, state: Optional[RunState], spill: _ArraySpill):
    """The adapter's cross-process blob, for exactly the states the carried
    trajectories reference.

    Deliberate reuse of `EnvAdapter.export_states`/`import_states`: the parallel
    scheduler already had to answer "what does this adapter know that lives only
    in this address space". A checkpoint is the same question across time
    instead of across a fork.
    """
    degraded: List[Dict[str, Any]] = []
    if state is None or ctx.env is None:
        return None, degraded
    export = getattr(ctx.env, "export_states", None)
    if export is None:
        return None, degraded
    rows = [t.states for t in (state.trajectory_store or ()) if t.states is not None]
    for pref in state.preferences or ():
        for traj in (pref.left_traj, pref.right_traj):
            if traj is not None and traj.states is not None:
                rows.append(traj.states)
    if not rows:
        return None, degraded
    try:
        blob = export(rows)
    except Exception as exc:  # noqa: BLE001
        blob = None
        degraded.append({"slot": "env_states", "reason": f"export_states failed ({exc})"})
    if blob is None:
        return None, degraded
    try:
        return encode(blob, spill, "$.env_states"), degraded
    except CheckpointEncodeError as exc:
        # Named consequence: restored trajectories still SCORE (`task_metric` is
        # a pure function of the rows) but cannot be re-rendered or
        # re-`reference_reward`-ed, so a VLM-graded method silently becomes a
        # blind comparator. That downstream effect is what
        # `budget.blind_comparisons` already counts.
        degraded.append({"slot": "env_states", "reason": f"could not encode ({exc})"})
        return None, degraded


# ==========================================================================
# --resume-info
# ==========================================================================


def describe(rundir: Path) -> Dict[str, Any]:
    """Everything `--resume-info` prints. Runs nothing and costs nothing."""
    root = Path(rundir)
    d = root / "checkpoints"
    out: Dict[str, Any] = {"run_dir": str(root), "exists": root.is_dir(),
                           "checkpoints": sorted(p.name for p in d.glob("ckpt-*.json")),
                           "restarts": sorted(p.name for p in d.glob("restart-*.json")),
                           "finished": (root / "result.json").exists()}
    status = _read_json(root / "status.json") or {}
    out["status"] = status.get("status")
    out["leg"] = status.get("leg")
    holder = _read_json(d / "resume.lock")
    if holder:
        holder = dict(holder)
        holder["heartbeat_age_s"] = round(time.time() - float(holder.get("heartbeat") or 0), 1)
    out["lock"] = holder
    for path in sorted(d.glob("ckpt-*.json"), reverse=True):
        try:
            body = _read_verified(path)
        except CheckpointError as exc:
            out.setdefault("rejected", []).append(str(exc))
            continue
        out.update({
            "seq": body.get("seq"), "phase": body.get("phase"),
            "restart": body.get("restart"), "next_iteration": body.get("next_iteration"),
            "retries": body.get("retries"),
            "completed_restarts": body.get("completed_restarts"),
            "pre_phases_done": body.get("pre_phases_done"),
            "config_hash": body.get("config_hash"),
            "budget": body.get("budget"), "dropped": body.get("dropped"),
            "degraded": body.get("degraded"),
        })
        state = body.get("state") or {}
        carried = state.get("fields") or {}
        out["carried_sizes"] = {
            k: len(v) for k, v in carried.items()
            if isinstance(v, (list, dict)) and k in (
                "dialogue", "preferences", "trajectory_store", "subtasks",
                "failure_memory", "all_reports", "fitness_history")}
        break
    return out
