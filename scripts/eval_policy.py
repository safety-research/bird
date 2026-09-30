#!/usr/bin/env python3
"""Evaluate one registry policy through its BIRD adapter and write the record.

    uv run python3 scripts/eval_policy.py --policy mt10_suite/reach --seeds 0-24
    uv run python3 scripts/eval_policy.py --policy h1hand_hb_hacks/hop \
        --seeds 0,1,2 --out /tmp/reeval.json          # in the HumanoidBench venv
    uv run python3 scripts/eval_policy.py --policy my_campaign/my_policy \
        --root /path/to/other/policies --seeds 0-4

This is the single evaluation entry point for ``policies/`` -- a score enters
a manifest by being measured through this, never by being typed.  There is one
``run_episode``, so every record counts success and captures frames the same
way.

The environment must be the policy's own ``env_id`` resolved through the
registry: the adapter is the instrument of record (it zeroes qacc_warmstart
per step; a rig that builds its own gym env does not, and the difference is
measurable).
Running this needs the policy's family runtime installed -- ``metaworld`` and
HumanoidBench pin incompatible mujoco versions, so there is one venv per
family and no venv for both.

The output shape (``bird-policy-eval-v2``, ``policies/records.schema.json``) is
a SUPERSET of v1 (the shape of the committed simple_suite v1 records): per-seed
rows plus a summary carrying the constants, the commit and the date, with six
things v1 could not say added.  Every v1 key keeps its name and
meaning, so a v1 reader reads a v2 record unchanged.

- ``seed_block_hash`` the identity of the seed SET this record was measured
                      over.  Two records over the same seeds share it whatever
                      else moved, which is what makes "these are the seeds that
                      were spent" a checkable claim rather than a filename.
- ``engine``          the simulator version in force.  A ``bird_task_metric``
                      on a MuJoCo tier is a function of the integrator, and
                      ``mujoco==3.1.6`` and ``3.3.0`` share no interpreter here
                      -- a record that did not name its engine could not be
                      compared with one measured under the other.
- ``provenance``      ``measured``: this process ran the episodes.
- ``measured_at``     the wall clock, to the microsecond.  It is in the body
                      because it is in the FILENAME (below), and a name that
                      states a time the record does not is a fact with no
                      source.
- ``horizon``         the loop bound every row ran to -- the adapter's own,
                      or ``--max-steps`` when that was given
                      (``bird/policy_api.py::episode_horizon``).  The task
                      metric is a per-step fraction over THIS many steps.
- ``max_steps``       the ``--max-steps`` cap, or null when the episodes ran
                      to the adapter's horizon.  Without the pair, a 60-step
                      acrobot measurement and the 300-step one differ in no
                      key but ``rows`` and ``summary``, a capped row's
                      ``terminated_early`` is False (the cap is not the
                      environment), and the closed schema forbids adding the
                      fact by hand -- so a capped record could be cited as a
                      manifest's score and nothing could tell.  A record
                      whose ``max_steps`` is set is not a manifest's score.
                      Both keys are OPTIONAL in the schema -- older v2 records
                      lack them, are never rewritten, and read as "uncapped
                      as far as the record knows" -- and this writer emits
                      both on every record regardless
                      (``tests/test_eval_policy_record.py`` holds it to that).

**Two v1 defects fixed here, both of which lose information silently.**

1. ``params`` was written as ``gather_params(record) or None``, so a policy
   that genuinely runs with no constants recorded ``null`` -- the same value a
   record whose constants could not be read would carry.  It is now the merged
   dict, ``{}`` included: an empty parameter set is a reading, and only ``null``
   is an absence (``policies/policies.schema.json``'s rule for ``score.value``).
2. The output filename was date-only and written with a plain ``write_text``,
   so a second measurement on the same day OVERWROTE the first in place, with
   nothing anywhere saying a record had been replaced.  The name now
   carries ``HHMMSS`` and a short digest of the record itself, so a same-day
   re-run is a NEW OBJECT and two runs that differ in any byte differ in their
   name.  The digest covers ``measured_at``, which is why that field is at
   microsecond resolution: two runs inside one second are still two records.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: ``policies/policies.schema.json``'s own ``$defs.policy.id.pattern``, quoted
#: here because THIS file is where a policy id becomes a FILENAME and nothing
#: upstream holds an id to it.  ``bird/policies.py::_check_policy`` requires a
#: ``/`` and a first segment equal to the campaign (which equals the directory
#: name) and NOTHING about the second segment -- there is no id regex in that
#: module at all.  The schema does have one, and the schema is not consulted on
#: the ``--root`` path: ``index()`` is pyyaml + stdlib by contract and
#: ``jsonschema`` may never reach it, so the pattern is enforced over the repo's
#: committed manifests by ``tests/test_eval_policy_record.py`` and over a FOREIGN root
#: by nothing.  Under ``--root`` the manifest comes from a policies directory
#: outside this checkout (this module's own third usage example), which the
#: repo's schema tests do not cover, so the id is validated here before it
#: becomes part of a filename.
#: ``tests/test_eval_policy_record.py`` holds this literal equal to the schema's
#: pattern in both directions, so the copy cannot drift.
POLICY_ID_PATTERN = r"^[a-z0-9_]+/[a-z0-9_]+$"
_POLICY_ID_RE = re.compile(POLICY_ID_PATTERN)


def _resolve_under(root: Path, name: str) -> Optional[Path]:
    """``<root>/<name>`` or ``None`` -- the containment guard.

    Resolved on BOTH sides, so a ``..`` segment, an absolute segment and a
    symlink IN THE NAME that leaves the resolved root are all one check, and
    strictly BELOW the root -- a name that resolves to the records directory
    itself is not a record.

    **It cannot see a symlinked ROOT.**  Resolving both sides follows a link on
    the root exactly as it follows one in the name, so a ``records/`` that is
    a link to somewhere else resolves to its target and the target is
    contained in itself: exit 0, the record filed
    outside the tree, through a clean id.  What closes that is
    in the caller and not here -- ``records/`` and the campaign directory are
    refused BY NAME with ``is_symlink()``, which is the one question about a
    path that resolving destroys, and the write is contained against the
    ``--root`` captured at argument time rather than against anything derived
    from the campaign's own directory.  This helper stays as it is: it is the
    guard over the NAME half, which is the half that comes out of a manifest.
    """
    try:
        rootr = root.resolve()
        target = (rootr / name).resolve()
    except (OSError, ValueError):
        return None
    return target if rootr in target.parents else None


#: What ``provenance`` may say (``policies/records.schema.json``): this file
#: always ran the episodes it records.
PROVENANCE = ("measured",)
#: The word this writer writes.
PROVENANCE_MEASURED = "measured"
#: What this writer writes.  Record readers pin this word.
SCHEMA = "bird-policy-eval-v2"


def _parse_seeds(text: str):
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part[1:]:
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise SystemExit(f"no seeds in {text!r}")
    return seeds


def _commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _engine() -> dict:
    """The simulator version this record was measured under, or null.

    ``mujoco`` only: it is the one engine in this tree whose version moves a
    number (`mujoco==3.1.6` for HumanoidBench against `3.3.0` for Meta-World and
    Assistax, and no interpreter runs both).  A mapping rather than a bare
    string so a second engine can be named later without changing the type of a
    field readers already parse.  ``None`` is the honest reading for the
    `bird_control` tier, which is pure numpy and has no engine at all -- it is
    not a failed import, and the record does not pretend to distinguish them
    beyond "no mujoco was importable here"."""
    try:
        import mujoco  # noqa: PLC0415 - lazy: this tier does not always exist
        return {"mujoco": str(getattr(mujoco, "__version__", None) or "") or None}
    except Exception:
        return {"mujoco": None}


def _seed_block_hash(seeds: list[int]) -> str:
    """sha256 over the seed SET, through `bird.checkpoint.digest_of`.

    The seeds in the order they were run, which is the order the rows are in --
    a record whose rows and whose hash disagreed about the order would be two
    claims about one measurement.  `digest_of` is the repo's one JSON digest and
    is reused rather than re-derived: two digests over one quantity disagree in
    exactly the places nobody compares."""
    from bird.checkpoint import digest_of
    return digest_of({"seeds": [int(s) for s in seeds]})


def _dirty_paths(campaign_root: Path) -> list[str] | None:
    """Paths whose working-tree content the recorded `commit` does NOT describe:
    modified or untracked (non-ignored) files under the policy's own campaign
    directory, the adapters, the task specs (which supply the horizon), and this
    evaluator.  A record measured on such a
    tree pins a commit that cannot reproduce it.  None when
    git cannot answer (then `tree_dirty` is null, never false).

    UNDER `--root <somewhere outside this checkout>` git cannot answer, and that
    is the normal case rather than a fault: the campaign directory lives outside
    the repository, `git status -- <that path>` is `fatal: ... is
    outside repository`, and the function returns None.  So a record written
    through `--root` carries `tree_dirty: null` and `dirty_paths: null`.  Null,
    never false: nothing looked, and a reader that checks a record's commit pin
    judges only records that carry the marker, exactly so an unlooked-at tree is
    not read as a clean one."""
    scopes = [str(campaign_root), "bird", "tasks", "scripts/eval_policy.py"]   # tasks/ supplies the horizon
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", *scopes],
                             cwd=ROOT, capture_output=True, text=True,
                             timeout=10)
        if out.returncode != 0:
            return None
        # A campaign's own records/ is data this script writes, not code the policy
        # runs: measuring several policies in a row must not flag the second because
        # the first's record was written.
        rec_dir = str(campaign_root).rstrip("/") + "/records/"
        return sorted(line[3:] for line in out.stdout.splitlines()
                      if line.strip() and not line[3:].startswith(rec_dir))
    except Exception:
        return None


def _record_name(policy_id: str, doc: dict) -> str:
    """``eval_<name>_<date>_<HHMMSS>_<digest8>.json``.

    ``tests/test_eval_policy_record.py`` stubs this function with a two-argument
    lambda and must keep working.

    Three parts, and each is load-bearing:

    - the DATE keeps the v1 name readable and sortable, which is why it stays
      even though the timestamp subsumes it;
    - ``HHMMSS`` is what makes a same-day re-run a new object.  v1 stopped at
      the date and overwrote in place;
    - the DIGEST is over the record itself, so two records that differ in any
      byte have different names and two that do not are the same object.  It
      covers ``measured_at`` (microseconds), which is what stops two runs inside
      one second sharing a name -- the case ``HHMMSS`` alone cannot see.

    Not a counter, and not an `_1`/`_2` suffix search: a counter depends on what
    is already in the directory, so the same measurement gets a different name
    depending on what else happens to be beside it, and the collision it is
    resolving becomes invisible again the moment the directory is copied.

    **The id is validated against the schema's own pattern first, and REFUSED
    rather than sanitised.**  It is the one part of this name that came out of a
    data file, and under ``--root`` that file lies outside this checkout, where
    the repo's schema tests do not reach.  Without this guard
    ``recordcamp/../../../../../escaped`` reaches
    ``write_text``: exit 0, a record written outside the root, outside the
    campaign and outside the repo, and a junk ``records/eval_..`` directory
    left behind.  Sanitising -- replacing the offending characters -- would write a record
    under a name that is not the id it claims, which is the same class of lie as
    a filename stating a time the record does not; the closed allow-lists this
    repo already keeps for data-file-supplied names refuse instead
    (``bird/checkpoint.py::_DECODABLE``, ``metaworld._SUCCESS_CHECKS``).
    """
    from bird.checkpoint import digest_of
    if not isinstance(policy_id, str) or not _POLICY_ID_RE.match(policy_id):
        raise SystemExit(
            f"policy id {policy_id!r} does not match "
            f"policies.schema.json's `{POLICY_ID_PATTERN}`, and this record's "
            "FILENAME is built out of it -- a `..`, a `/` or an absolute "
            "segment in the name half writes the record somewhere other than "
            "the campaign's own records/. Refusing rather than rewriting the "
            "name: a record filed under a name that is not its id is worse "
            "than no record. Fix the id in the manifest that declares it, or "
            "pass --out to choose the path yourself.")
    stamp = _dt.datetime.fromisoformat(doc["measured_at"]).strftime("%H%M%S")
    return (f"eval_{policy_id.split('/', 1)[1]}_{doc['date']}_{stamp}_"
            f"{digest_of(doc)[:8]}.json")


def _measure(env, policy, seeds, max_steps):
    """Every seed on ONE adapter: ``run_episode``'s row per seed, with ``states``
    popped (it is large and already summarised by the metric)."""
    from bird.policy_api import run_episode  # lazy: numpy, like every bird import here
    rows = []
    for seed in seeds:
        row = run_episode(env, policy, seed, max_steps=max_steps)
        row.pop("states")
        rows.append(row)
        rr = row.get("reference_return")
        print(f"seed {seed:>3}: steps {row['steps']:>4}  "
              f"metric {row['metric']:.4f}  success {row['success']}"
              + (f"  reference_return {rr:.1f}" if rr is not None else ""))
    return rows


def _summary(rows: list) -> dict:
    """The record's ``summary`` block over its rows."""
    metrics = [r["metric"] for r in rows]
    returns = [r["reference_return"] for r in rows if r.get("reference_return") is not None]
    return {
        "n": len(rows),
        "mean": sum(metrics) / len(metrics),
        "min": min(metrics),
        "max": max(metrics),
        "successes": sum(r["success"] for r in rows),
        # The env's own reward summed per episode (rows[].reference_return), averaged over
        # the seeds that measured one. On HumanoidBench this is the benchmark's episode
        # return, the unit its published baselines use -- copy it into the manifest as
        # score.secondary.hb_return_mean so a results table can show the two on one scale.
        "reference_return_mean": (sum(returns) / len(returns)) if len(returns) == len(rows) and returns else None,
    }


def main(argv=None) -> int:
    """Measure one policy and write one record."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True,
                    help="registry id, e.g. mt10_suite/reach")
    ap.add_argument("--seeds", default="0-24",
                    help='"0-24" or "0,1,2" (default 0-24)')
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap per episode (default: the adapter's horizon)")
    ap.add_argument("--root", default=None,
                    help="policies directory to resolve --policy in (default: "
                         "the repo's own policies/). Use it to measure a "
                         "policies directory outside this checkout; the "
                         "directory is indexed as a whole, so one malformed "
                         "manifest anywhere under it fails the run")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: the policy dir's "
                         "records/eval_<name>_<date>_<HHMMSS>_<digest>.json)")
    args = ap.parse_args(argv)

    from bird import registry
    from bird.policies import (get as get_record, index as policy_index,
                               policies_root)
    from bird.policy_api import episode_horizon, gather_params, load_policy

    root = Path(args.root) if args.root else None
    # The containment ANCHOR, resolved once and HERE -- off the command line,
    # before a manifest has been read and before anything under the campaign
    # directory has been touched. `_resolve_under` resolves BOTH of its sides
    # at write time, so a check anchored on a path derived from the campaign's own
    # directory cannot see a symlink INSIDE the root: the anchor follows the
    # link to its target and the target is trivially contained in itself -- a
    # pre-created `records/` symlink would file the record outside the root,
    # exit 0, through a perfectly clean id. `--root` is the
    # one path on this branch that comes from the command line rather than
    # from the tree under it, which is what makes it the anchor. `policies_root`
    # supplies the `--root`-less default (the repo's own `policies/`) rather
    # than a second copy of it here.
    root_anchor = policies_root(root).resolve()
    # Unconditionally, and BEFORE the id is resolved: `index` memoises on the
    # root path with no content key, so a root this process has already read
    # would otherwise be resolved out of a snapshot the manifest on disk may no
    # longer declare -- and under `--root` that directory may have changed
    # since it was last read.
    policy_index(root, refresh=True)
    record = get_record(args.policy, root)

    registry.load_all()
    env = registry.get("env", record.env_id)(None)

    policy = load_policy(args.policy, root=root)
    # The bound every row below runs to, from the same expression the loop uses.
    horizon = episode_horizon(env, args.max_steps)

    seeds = _parse_seeds(args.seeds)
    rows = _measure(env, policy, seeds, args.max_steps)

    _dirty = _dirty_paths(record.root.relative_to(ROOT) if record.root.is_relative_to(ROOT) else record.root)
    now = _dt.datetime.now().astimezone()
    # v2, key for key and in the order it has always been written: the filename
    # digests these bytes, so the order is part of the contract.
    doc = {
        "schema": SCHEMA,
        "policy": args.policy,
        "env_id": record.env_id,
        "unit": "bird_task_metric",
        # What bounded the episodes. `max_steps` is the cap AS GIVEN and null when
        # there was none (0 means "the adapter's horizon" to the loop and is recorded
        # as that absence); `horizon` is what the rows actually ran to. A capped
        # measurement is a different quantity from the task metric, and this pair is
        # the only place the record says which one it is: rows[].terminated_early is
        # the environment's word and stays False when the cap ends an episode.
        "horizon": horizon,
        "max_steps": int(args.max_steps) if args.max_steps else None,
        "commit": _commit(),
        # A commit is only a pin if the tree it names is the tree that ran: a
        # `tree_dirty: true` record is not fit to back a manifest's score, so
        # commit first, then measure.
        "tree_dirty": None if _dirty is None else bool(_dirty),
        "dirty_paths": _dirty,
        "date": now.date().isoformat(),
        # To the microsecond, and in the body because it is in the filename:
        # see the module docstring. It is also what makes two runs inside one
        # second two records rather than one overwritten twice.
        "measured_at": now.isoformat(timespec="microseconds"),
        # This writer ran the episodes.
        "provenance": PROVENANCE_MEASURED,
        "engine": _engine(),
        "seed_block_hash": _seed_block_hash(seeds),
        # The MERGED constants the policy actually ran with (kwargs_from /
        # params_file / inline, later wins) -- record.params alone writes null
        # for policies whose constants live in a JSON file beside the code.
        # `{}` when the policy genuinely has none: an empty parameter set is a
        # reading, and only null is an absence. `or None` (v1) collapsed the two.
        "params": gather_params(record),
        "summary": _summary(rows),
        "rows": rows,
    }

    text = json.dumps(doc, indent=1) + "\n"
    if args.out:
        # An operator's explicit path, and it stays one: `--out
        # /tmp/reeval.json` (this module's own second usage example) is a
        # deliberate overwrite of a deliberate name. The
        # guards below are about the DERIVED path, whose name half comes out of
        # a manifest rather than off the command line.
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        records = record.root / "records"
        # Refused BY NAME, before any name is derived, any path resolved or
        # anything created. `is_symlink()` asks the one question about a path
        # that resolving destroys, and every other check in this branch
        # resolves -- so a `records/` (or campaign directory) that is a link
        # to somewhere else passes all of them:
        # `_resolve_under` resolves the link to its target and finds the target
        # contained in itself. Exit 0, the record outside the root, through a
        # clean id. Neither link is followed. Refusing, not repointing: a record
        # filed somewhere other than the campaign that claims it is the same class of
        # lie as one filed under a name that is not its id.
        for what, path in (("the campaign directory", record.root),
                           ("the campaign's records/ directory", records)):
            if path.is_symlink():
                raise SystemExit(
                    f"{path} ({what}) is a symbolic link to "
                    f"{str(path.readlink())!r} -- refusing to follow it. A "
                    "record is written into the campaign's OWN directory; a "
                    "link there files it somewhere else while every "
                    "check that resolves the path reads it as contained. "
                    "Remove the link, or pass --out to name the destination "
                    "yourself.")
        # The guards below run BEFORE anything is created: a refused id must
        # leave nothing behind, not even a junk `records/eval_..` directory.
        name = _record_name(args.policy, doc)
        # Belt as well as braces: `_record_name` refuses the ID, this refuses a
        # NAME that resolves out of records/, and the anchor below refuses a
        # PATH that leaves the tree `--root` named. Three guards over one hole
        # because they fail on different things -- an id the schema forbids, a
        # name a future edit builds some other way, and a directory that is no
        # longer where it says it is. None of them subsumes another, and in
        # particular NEITHER OF THE FIRST TWO SEES A SYMLINKED records/, which
        # is what the refusal above is for.
        out = _resolve_under(records, name)
        if out is None:
            raise SystemExit(
                f"the derived record name {name!r} does not resolve inside "
                f"{records} -- refusing to write a record outside the "
                "campaign's own records/ directory")
        if root_anchor not in out.parents:
            raise SystemExit(
                f"{out} is not inside {root_anchor} -- refusing to write a "
                "record outside the policies root this run was given. The "
                "anchor is the `--root` argument resolved once, before any "
                "manifest was read, so it holds even when a directory under "
                "the root has been made to point elsewhere.")
        records.mkdir(parents=True, exist_ok=True)
        # Exclusive create. The name is a function of the record's own bytes
        # (`_record_name`'s digest covers the whole doc), so a name that already
        # exists is either THIS record already written -- accepted, nothing is
        # lost -- or a short-digest collision, which is a refusal and not a
        # silent replacement. `write_text` alone could not tell the two apart,
        # which is v1's defect in miniature.
        try:
            with open(out, "x", encoding="utf-8") as fh:
                fh.write(text)
        except FileExistsError:
            if out.read_text(encoding="utf-8") != text:
                raise SystemExit(
                    f"{out} already exists with different content. The name is "
                    "a digest of the record, so this is a digest collision, not "
                    "a re-run: refusing to replace a record in place (that is "
                    "the v1 defect this name shape exists to prevent). Pass "
                    "--out to name the second one yourself.") from None
    print(f"\n{doc['summary']['n']} episodes  mean {doc['summary']['mean']:.4f}"
          f"  min {doc['summary']['min']:.4f}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
