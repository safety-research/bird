"""``bird-policy-eval-v2``: the record says what was measured, on what, and by whom.

``scripts/eval_policy.py`` is the single record writer for ``policies/`` -- a
score reaches a manifest by being measured through it, never by being typed.
This file holds v2 to the three things v1 could not do and the two it did
wrongly.

**The two defects, and why each is silent.**

- ``params`` was ``gather_params(record) or None``.  A policy that genuinely
  runs with no constants therefore recorded ``null``, which is the SAME value a
  record whose constants could not be read would carry.  Absence and emptiness
  are two states (``policies/policies.schema.json``'s rule for a null
  ``score.value``), and a reader cannot recover the difference afterwards.
- the output filename was date-only and written with a plain ``write_text``, so
  a second measurement on the same day overwrote the first IN PLACE with nothing
  saying so.  A suffix typed by hand onto the older file is the only trace such
  an overwrite can leave, and only when somebody noticed.

**The three additions** are ``seed_block_hash`` (the identity of the seed set,
so "these seeds were spent" is checkable against another record's seed set rather
than being a claim about a filename), ``engine`` (a ``bird_task_metric`` on a
MuJoCo tier is a function of the integrator, and ``3.1.6`` and ``3.3.0`` share
no interpreter here) and ``provenance`` (``measured``: the writer ran the
episodes it records).

Unmarked and in-process: the toy tier, pure numpy, no fork.  The half that
spawns ``sys.executable`` is ``tests/test_eval_policy_record_subprocess.py``, which is
``slow`` -- ``pyproject.toml`` assigns that marker BY FILE and BY KIND, and "it
forks a process" is one of the kinds.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


#: The keys ``bird-policy-eval-v1`` wrote, pinned as a LITERAL here rather than
#: read off the writer.  v2's whole claim is that it is a SUPERSET, and a list
#: derived from the code under test cannot fail when the code drops one of them.
V1_KEYS = ("schema", "policy", "env_id", "unit", "commit", "tree_dirty",
           "dirty_paths", "date", "params", "summary", "rows")

#: What v2 adds.  Also literal, for the same reason in the other direction: a
#: test that read this off the writer would pass a writer that added nothing.
#: The schema REQUIRES the first four.  The last two are optional there: they
#: were added after v2 records had already been committed, and a committed
#: record is never rewritten to fit a schema, so their absence is given a
#: meaning ("written before the keys existed, uncapped as far as the record
#: knows") instead.  The WRITER still emits both on every
#: record, and test_v2_carries_every_v1_key_and_its_additions holds it to that:
#: optional in the schema is not optional in the writer.
V2_REQUIRED_ADDED = ("measured_at", "provenance", "engine", "seed_block_hash")
V2_OPTIONAL_ADDED = ("horizon", "max_steps")
V2_ADDED = V2_REQUIRED_ADDED + V2_OPTIONAL_ADDED


def _eval_policy():
    """``scripts/eval_policy.py``, imported by path.

    ``scripts/`` is not a package and nothing may import it as one; every other
    test in this suite that exercises a script does the same.  A fresh module
    object per call would re-run the import for every test, so it
    is cached at module scope by the fixture below.
    """
    path = REPO / "scripts" / "eval_policy.py"
    spec = importlib.util.spec_from_file_location("_record_eval_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ep():
    return _eval_policy()


# ==========================================================================
# A synthetic campaign in tmp -- what `--root` is for
# ==========================================================================

#: A controller with NO constants at all: no `params`, no `params_file`, no
#: `entry.kwargs_from`.  That is the case v1 recorded as `null`, and it has to
#: be a policy that genuinely has none rather than one whose constants happen to
#: be empty, or the test would pass against `record.params or None` too.
_CONTROLLER = '''import numpy as np


def make_policy(p):
    def act(s):
        return np.clip(1.5 * (s[4:6] - s[:2]), -1.0, 1.0)
    return act
'''


def _write_campaign(root: Path, campaign: str = "recordcamp",
                    params_line: str = "", name: str = "candidate") -> Path:
    """A one-policy campaign under `root`.

    `name` is the id's SECOND segment and is interpolated verbatim, which is the
    point: `bird/policies.py` constrains the first segment (it must equal the
    campaign, which must equal the directory name) and constrains the second one
    not at all, so a manifest written under `--root` can put anything there --
    including a `..`. See the traversal test below.
    """
    camp = root / campaign
    camp.mkdir(parents=True)
    (camp / "controller.py").write_text(_CONTROLLER)
    (camp / "policies.yaml").write_text(
        f"campaign: {campaign}\n"
        "family: bird_control\n"
        "pattern: closure\n"
        "code: [controller.py]\n"
        "source: {archive: 'not distributed (synthetic test fixture)', date: '2026-09-06'}\n"
        "policies:\n"
        f"  - id: {campaign}/{name}\n"
        "    task: null\n"
        "    task_reason: a synthetic record fixture, not a catalogue task\n"
        "    env_id: toy_reacher\n"
        "    entry: {file: controller.py, symbol: make_policy}\n"
        + params_line +
        "    status: reference\n"
        "    score:\n"
        "      value: null\n"
        "      reason: measured in-test; this manifest records the artifact\n"
        "      unit: bird_task_metric\n"
        "      verified_through: bird_adapter\n"
        "      date: '2026-09-06'\n")
    return camp


def _run(ep, root: Path, *extra, seeds="0-2", policy="recordcamp/candidate"):
    argv = ["--policy", policy, "--seeds", seeds, "--root", str(root), *extra]
    assert ep.main(argv) == 0
    return sorted((root / policy.split("/", 1)[0] / "records").glob("*.json"))


def _only(paths):
    assert len(paths) == 1, [p.name for p in paths]
    return json.loads(paths[0].read_text())


def _new_paths(tmp_path: Path, before: set) -> list:
    """Everything created under `tmp_path` since `before`, bytecode aside.

    **The whole tree, not a filter over the artifact the guard exists to
    prevent.** Asserting only that no `.json` and no `eval_*` path appeared is a
    statement about the FILE and blind to a DIRECTORY the write site might
    create on its way to refusing: a bare `records.mkdir(parents=True,
    exist_ok=True)` moved above both guards would pass every such check. The
    comment at the write site says "before anything is created"; this is the
    assertion that can see it.

    `__pycache__` is the INTERPRETER's, written when `load_policy` imports the
    campaign's controller, and it appears whether the write site refuses or
    not. Excluded wherever that name is a path COMPONENT
    (``"__pycache__" not in p.parts``: the directory and everything under it,
    at any depth) and nothing else is, so a junk `records/` is still caught.
    """
    return sorted(str(p.relative_to(tmp_path))
                  for p in set(tmp_path.rglob("*")) - before
                  if "__pycache__" not in p.parts)


# ==========================================================================
# --root
# ==========================================================================

def test_root_evaluates_a_campaign_that_is_not_in_this_repo(tmp_path, ep):
    # The whole vertical under a foreign root: manifest -> index(root=) ->
    # load_policy(root=) -> episode on the adapter of record -> record written
    # beside the campaign.  A policy directory written outside the checkout is
    # exactly this shape, and it must never be the repo's own policies/ (index()
    # fails repo-wide on one malformed manifest anywhere under a root).
    _write_campaign(tmp_path)
    doc = _only(_run(ep, tmp_path))
    assert doc["policy"] == "recordcamp/candidate"
    assert doc["env_id"] == "toy_reacher"
    assert [r["seed"] for r in doc["rows"]] == [0, 1, 2]
    assert doc["summary"]["n"] == 3


def test_a_foreign_root_leaves_tree_dirty_null_rather_than_false(tmp_path, ep):
    """git cannot answer about a path outside the checkout, and the record says
    so rather than claiming a clean tree.

    `git status -- <tmp path>` is `fatal: ... is outside repository`, so
    `_dirty_paths` returns None and both markers are null.  Null is not false:
    a tree nobody looked at is never read as one that was looked at and found
    clean.
    """
    _write_campaign(tmp_path)
    doc = _only(_run(ep, tmp_path))
    assert doc["tree_dirty"] is None
    assert doc["dirty_paths"] is None


# ==========================================================================
# Defect 1: params is {} and never None
# ==========================================================================

def test_a_parameterless_policy_records_an_empty_params_and_not_null(tmp_path, ep):
    # v1 wrote `gather_params(record) or None`. This policy has no `params`, no
    # `params_file` and no `kwargs_from`, so v1 recorded `null` -- the same value
    # it would write if the constants existed and could not be read. `{}` is a
    # reading; only null is an absence.
    _write_campaign(tmp_path)
    doc = _only(_run(ep, tmp_path))
    assert doc["params"] == {}
    assert doc["params"] is not None


def test_constants_that_exist_still_reach_the_record(tmp_path, ep):
    # The control for the test above: if `params` were hard-wired to `{}` the
    # previous check would pass and this one would not.
    _write_campaign(tmp_path, params_line="    params: {gain: 2.0}\n")
    doc = _only(_run(ep, tmp_path))
    assert doc["params"] == {"gain": 2.0}


# ==========================================================================
# Defect 2: a same-day re-run is a NEW OBJECT
# ==========================================================================

def test_two_runs_the_same_day_write_two_records(tmp_path, ep):
    """v1's name was `eval_<name>_<date>.json` and the write was a plain
    `write_text`, so the second measurement of the day replaced the first with
    nothing anywhere recording that a record had been replaced.

    Both runs are identical invocations on the same day, which is the case v1
    lost.  They may also land inside the same SECOND, which `HHMMSS` alone
    cannot separate -- the digest covers `measured_at` at microsecond
    resolution, so they are still two names.
    """
    _write_campaign(tmp_path)
    first = _run(ep, tmp_path)
    second = _run(ep, tmp_path)
    assert len(first) == 1
    assert len(second) == 2, [p.name for p in second]
    docs = [json.loads(p.read_text()) for p in second]
    # Two objects, not one written twice: the rows are the same measurement and
    # the records are distinct documents that each say when they were taken.
    assert [d["rows"] for d in docs[:1]] == [docs[1]["rows"]]
    assert docs[0]["measured_at"] != docs[1]["measured_at"]


def test_the_name_carries_the_date_the_clock_and_a_digest(tmp_path, ep):
    _write_campaign(tmp_path)
    path = _run(ep, tmp_path)[0]
    doc = json.loads(path.read_text())
    stamp = _dt.datetime.fromisoformat(doc["measured_at"]).strftime("%H%M%S")
    assert path.name.startswith(f"eval_candidate_{doc['date']}_{stamp}_")
    digest = path.name.rsplit("_", 1)[1][: -len(".json")]
    assert len(digest) == 8 and all(c in "0123456789abcdef" for c in digest)


def test_the_digest_in_the_name_is_of_the_record(tmp_path, ep):
    # Not a counter and not an `_1` suffix search: a counter depends on what is
    # already in the directory, so the same measurement gets a different name
    # depending on what else happens to be beside it.
    from bird.checkpoint import digest_of
    _write_campaign(tmp_path)
    path = _run(ep, tmp_path)[0]
    doc = json.loads(path.read_text())
    assert path.name.endswith(digest_of(doc)[:8] + ".json")


# ==========================================================================
# v2 is a superset of v1
# ==========================================================================

def test_v2_carries_every_v1_key_and_its_additions(tmp_path, ep):
    _write_campaign(tmp_path)
    doc = _only(_run(ep, tmp_path))
    assert doc["schema"] == "bird-policy-eval-v2"
    missing = [k for k in V1_KEYS if k not in doc]
    assert not missing, f"v2 dropped v1 keys {missing}; it is a superset or it is a break"
    absent = [k for k in V2_ADDED if k not in doc]
    assert not absent, f"v2 is missing its own additions {absent}"
    assert set(doc) == set(V1_KEYS) | set(V2_ADDED), sorted(doc)


def test_a_capped_eval_records_the_cap_as_its_horizon(tmp_path, ep):
    """`--max-steps` shortens every episode, and without a horizon key nothing in the record
    would say so: `terminated_early` is false on a row the cap ended (it is the ENVIRONMENT's
    word, and the loop bound is not the environment), and the schema's
    `additionalProperties: false` forbids adding one by hand. A 60-step acrobot measurement
    (mean 0.0000) and the 300-step one (0.1595) would differ in no top-level key but rows and
    summary, and either could be cited as `score.record`. The record carries the loop bound
    every row ran to and the cap when one was set."""
    jsonschema = pytest.importorskip("jsonschema")
    from bird import registry
    registry.load_all()
    adapter_horizon = registry.get("env", "toy_reacher")(None).horizon
    _write_campaign(tmp_path)
    capped = _only(_run(ep, tmp_path, "--max-steps", "5"))
    assert capped["horizon"] == 5 and capped["max_steps"] == 5
    assert all(r["steps"] == 5 and r["terminated_early"] is False for r in capped["rows"]), (
        "a row the cap ended reads like a full one BY DESIGN; the record, not the row, says cap")
    jsonschema.validate(capped, _records_schema())
    for r in _run(ep, tmp_path):
        r.unlink()
    full = _only(_run(ep, tmp_path))
    assert full["horizon"] == adapter_horizon and full["max_steps"] is None
    assert all(r["steps"] == adapter_horizon for r in full["rows"])
    jsonschema.validate(full, _records_schema())
    # `--max-steps 0` means "the adapter's horizon" to the loop, and is recorded as the absence it is.
    for r in _run(ep, tmp_path):
        r.unlink()
    zero = _only(_run(ep, tmp_path, "--max-steps", "0"))
    assert zero["horizon"] == adapter_horizon and zero["max_steps"] is None


def _records_schema():
    """``records.schema.json``, the v2 contract, as a dict."""
    return json.loads((REPO / "policies" / "records.schema.json").read_text())


def test_the_record_validates_against_its_own_schema(tmp_path, ep):
    jsonschema = pytest.importorskip("jsonschema")
    _write_campaign(tmp_path)
    doc = _only(_run(ep, tmp_path))
    jsonschema.validate(doc, _records_schema())


def test_the_schema_refuses_a_record_that_lost_a_key(tmp_path, ep):
    # The control: `additionalProperties: false` plus a full `required` list is
    # only a contract if dropping a key fails. A schema that validated anything
    # would pass the test above with every addition deleted.
    jsonschema = pytest.importorskip("jsonschema")
    _write_campaign(tmp_path)
    doc = _only(_run(ep, tmp_path))
    schema = _records_schema()
    for key in V2_REQUIRED_ADDED:
        broken = {k: v for k, v in doc.items() if k != key}
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(broken, schema)
    # The two optional additions. Dropping them is the shape of every v2 record
    # committed before the keys existed and validates (tests/test_policy_records_corpus.py
    # holds committed records to it) -- but a WRONG value still fails: the keys
    # are optional, not unchecked, and 0 in particular is the writer's "use the
    # adapter's horizon", recorded as null, never as a number.
    without_optional_keys = {k: v for k, v in doc.items() if k not in V2_OPTIONAL_ADDED}
    jsonschema.validate(without_optional_keys, schema)
    for bad in ({"horizon": 0}, {"horizon": "300"}, {"horizon": None},
                {"max_steps": 0}, {"max_steps": "60"}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**doc, **bad}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**doc, "params": None}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**doc, "provenance": "guessed"}, schema)


# ==========================================================================
# seed_block_hash, engine, provenance
# ==========================================================================

def test_the_seed_block_hash_is_over_the_seeds_and_only_the_seeds(tmp_path, ep):
    from bird.checkpoint import digest_of
    _write_campaign(tmp_path)
    doc = _only(_run(ep, tmp_path))
    assert doc["seed_block_hash"] == digest_of({"seeds": [0, 1, 2]})


def test_a_different_seed_set_is_a_different_hash(tmp_path, ep):
    # The control: a hash that did not depend on the seeds would pass the check
    # above and fail this one. It is what makes "these seeds were spent" a
    # claim another record's seed set can be compared against.
    _write_campaign(tmp_path)
    _run(ep, tmp_path, seeds="0-2")
    paths = _run(ep, tmp_path, seeds="3-5")
    hashes = {json.loads(p.read_text())["seed_block_hash"] for p in paths}
    assert len(hashes) == 2


def test_provenance_is_measured_and_the_word_is_from_a_closed_set(tmp_path, ep):
    _write_campaign(tmp_path)
    doc = _only(_run(ep, tmp_path))
    assert doc["provenance"] == "measured"
    assert doc["provenance"] in ep.PROVENANCE
    assert ep.PROVENANCE == ("measured",)


def test_the_engine_names_mujoco_or_says_there_is_none(tmp_path, ep):
    _write_campaign(tmp_path)
    doc = _only(_run(ep, tmp_path))
    assert set(doc["engine"]) == {"mujoco"}
    version = doc["engine"]["mujoco"]
    assert version is None or isinstance(version, str)
    try:
        import mujoco
    except Exception:
        assert version is None
    else:
        assert version == mujoco.__version__


# ==========================================================================
# The name comes out of a manifest, and under --root that manifest is the
# artifact being measured
# ==========================================================================
#
# `_record_name` interpolates the id's second segment straight into a path.
# Nothing upstream constrains that segment: `policies/policies.schema.json`
# has the pattern `^[a-z0-9_]+/[a-z0-9_]+$`, but `bird/policies.py` never
# applies it -- there is no id regex in that module at all; `_check_policy`
# requires a `/` and a first segment equal to the
# campaign, and nothing about the second. jsonschema is not on `index()`'s
# import path by contract (pyyaml + stdlib only), so the pattern binds a
# FOREIGN root through nothing.
#
# `--root` is what makes that reachable rather than theoretical: it points at a
# policy directory outside this checkout, whose manifest is written by the same
# author as the policy under test. Without the guard: exit 0 and a record
# written outside the root, outside the campaign and outside the repo.

def test_the_id_pattern_is_the_policy_schemas_own():
    """One pattern, two files, held equal here.

    `scripts/eval_policy.py` quotes the schema's regex rather than importing
    it (the schema is JSON on disk and this is a script that must run with
    pyyaml + stdlib), so it is a copy -- and a copy nothing compares is the
    duplicate-enum defect. The literal below is pinned in THIS file so that
    both moving together is still a failure.
    """
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("_idpat_eval_policy",
                                   REPO / "scripts" / "eval_policy.py")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    schema = json.loads((REPO / "policies" / "policies.schema.json").read_text())
    in_schema = schema["$defs"]["policy"]["properties"]["id"]["pattern"]
    assert in_schema == r"^[a-z0-9_]+/[a-z0-9_]+$"
    assert mod.POLICY_ID_PATTERN == r"^[a-z0-9_]+/[a-z0-9_]+$"
    assert mod.POLICY_ID_PATTERN == in_schema


def test_every_committed_policy_id_already_matches_the_pattern():
    """The guard costs the repo nothing, and this is the measurement saying so.

    Every id across `policies/*/policies.yaml` matches. A
    refusal that would have refused something already committed is a different
    change from one that would not, and the difference is not visible from the
    guard.
    """
    import re

    import yaml
    pattern = re.compile(r"^[a-z0-9_]+/[a-z0-9_]+$")
    ids = []
    for manifest in sorted((REPO / "policies").glob("*/policies.yaml")):
        doc = yaml.safe_load(manifest.read_text())
        ids.extend(p["id"] for p in (doc.get("policies") or []))
    assert ids, "no committed policy manifests found -- the check is vacuous"
    assert [i for i in ids if not pattern.match(i or "")] == []


def test_a_traversing_policy_id_is_refused_and_nothing_is_written_outside(tmp_path, ep):
    """The traversal, end to end: refused, and the tree is untouched.

    `--root` is a SUBDIRECTORY of `tmp_path` so that there is an outside to
    escape to; the id climbs five levels, which without the guard lands the
    file beside `scripts/` (exit 0, `escaped_<date>_<HHMMSS>_<digest>.json`
    one directory up).
    """
    root = tmp_path / "sandbox" / "scripts"
    _write_campaign(root, name="../../../../../escaped")
    before = {p for p in tmp_path.rglob("*")}
    with pytest.raises(SystemExit) as exc:
        ep.main(["--policy", "recordcamp/../../../../../escaped",
                 "--seeds", "0-1", "--root", str(root)])
    # The refusal NAMES the id -- an operator staring at a foreign policy dir
    # has to be able to find which manifest to fix.
    assert "recordcamp/../../../../../escaped" in str(exc.value)
    assert "^[a-z0-9_]+/[a-z0-9_]+$" in str(exc.value)
    # NOTHING is created -- not a file and not a directory. A guard that
    # refused after `mkdir` would leave a junk `records/` behind, and an
    # assertion filtering on `.json` and on an `eval_` prefix could not see
    # that directory: `_new_paths` is what makes "before anything is created"
    # a checkable claim.
    assert _new_paths(tmp_path, before) == []


def test_a_shallow_traversal_is_refused_too(tmp_path, ep):
    # The two-level case stays INSIDE the root and still writes outside the
    # campaign's own records/ -- a record filed under a name that is not its id.
    root = tmp_path / "sandbox" / "scripts"
    _write_campaign(root, name="../../pwned")
    with pytest.raises(SystemExit, match="does not match"):
        ep.main(["--policy", "recordcamp/../../pwned", "--seeds", "0-1",
                 "--root", str(root)])
    assert list(tmp_path.rglob("*pwned*")) == []


def test_a_clean_id_still_writes_its_record(tmp_path, ep):
    # The control for the three above: the guard refuses traversals and not
    # policies. Without this, `_record_name` raising unconditionally passes them.
    root = tmp_path / "sandbox" / "scripts"
    _write_campaign(root)
    doc = _only(_run(ep, root))
    assert doc["policy"] == "recordcamp/candidate"


def test_the_containment_guard_refuses_a_name_that_leaves_the_records_dir(tmp_path, ep):
    """The second guard, called directly.

    Belt as well as braces: `_record_name` refuses the ID and this refuses the
    NAME -- a pattern the schema forbids against anything that resolves out of
    `records/`, including a `..` in a name some future edit builds another way.

    It does NOT refuse a symlinked `records/`. That case is the last check
    below; the two guards that close it
    are the `is_symlink()` refusal and the `--root` anchor.
    """
    records = tmp_path / "camp" / "records"
    records.mkdir(parents=True)
    assert ep._resolve_under(records, "eval_ok.json") == records / "eval_ok.json"
    assert ep._resolve_under(records, "../escaped.json") is None
    assert ep._resolve_under(records, "a/../../escaped.json") is None
    # The hole, stated as an assertion rather than left as prose: with
    # `records/` itself a link, the name resolves under the LINK'S TARGET and
    # this helper reports it contained. Nothing is wrong with the helper -- it
    # is the guard over the name half -- but a reader who believes it covers
    # links stops looking for the guard that is actually load-bearing.
    linked = tmp_path / "camp" / "linked_records"
    outside = tmp_path / "outside"
    outside.mkdir()
    linked.symlink_to(outside)
    assert ep._resolve_under(linked, "eval_ok.json") == outside.resolve() / "eval_ok.json"
    # Strictly BELOW: the directory itself is not a record.
    assert ep._resolve_under(records, ".") is None


# ==========================================================================
# A symlink inside the root, which resolving cannot see
# ==========================================================================

def test_a_symlinked_records_dir_is_refused_and_nothing_is_written_outside(tmp_path, ep):
    """A link, not a name.

    A campaign under `--root` is untrusted input, so `records/` can be a link
    to anywhere. Without the guard:
    exit 0, `outside_target/eval_candidate_<date>_<HHMMSS>_<digest>.json`,
    through a perfectly clean id. Every check in the
    write site resolved, and resolving is exactly what makes a link invisible:
    the anchor follows it and the target is contained in itself.

    Narrower than the id traversal above, because `open(..., "x")` can only
    create a NEW `eval_*.json` there -- but "narrower" is a statement about the
    blast radius, not about whether the record left the tree, and it left it.
    """
    root = tmp_path / "sandbox" / "scripts"
    camp = _write_campaign(root)
    outside = tmp_path / "outside_target"
    outside.mkdir()
    (camp / "records").symlink_to(outside)
    before = {p for p in tmp_path.rglob("*")}
    with pytest.raises(SystemExit, match="is a symbolic link"):
        ep.main(["--policy", "recordcamp/candidate", "--seeds", "0-1",
                 "--root", str(root)])
    # The refusal NAMES the link and its target: an operator looking at a
    # foreign policy dir has to be able to find what to delete.
    assert list(outside.iterdir()) == []
    assert _new_paths(tmp_path, before) == []


def test_a_symlinked_campaign_dir_is_refused_too(tmp_path, ep):
    """One level up, and the same refusal.

    `record.root` is `<root>/<campaign>` and the manifest lives in it, so a
    campaign directory that is a link is a readable, indexable campaign whose
    `records/` is somewhere else entirely -- and the link is one `mkdir` and one
    `symlink` from anything writing a script dir. Both names are checked
    because either alone leaves the other open.
    """
    root = tmp_path / "sandbox" / "scripts"
    root.mkdir(parents=True)
    outside = tmp_path / "outside_camp"
    _write_campaign(outside)
    (root / "recordcamp").symlink_to(outside / "recordcamp")
    before = {p for p in tmp_path.rglob("*")}
    with pytest.raises(SystemExit, match="is a symbolic link"):
        ep.main(["--policy", "recordcamp/candidate", "--seeds", "0-1",
                 "--root", str(root)])
    assert _new_paths(tmp_path, before) == []


def test_the_root_anchor_refuses_the_link_even_with_the_name_check_gone(tmp_path, ep, monkeypatch):
    """The second guard, made reachable.

    With `is_symlink()` answering False for everything, the write site is back
    to the unguarded state: `_resolve_under` resolves the
    linked `records/` to its target and reports the record contained. What
    refuses is the anchor -- `--root` resolved ONCE at argument time, before any
    manifest was read -- which is why it is not derived from `record.root` and
    why capturing it later would be the same bug in a new place.

    `pathlib.Path.is_symlink` is stubbed rather than the module's own helper,
    so the check being defeated is the exact call the guard makes.
    """
    monkeypatch.setattr(ep.Path, "is_symlink", lambda self: False)
    root = tmp_path / "sandbox" / "scripts"
    camp = _write_campaign(root)
    outside = tmp_path / "outside_target"
    outside.mkdir()
    (camp / "records").symlink_to(outside)
    before = {p for p in tmp_path.rglob("*")}
    with pytest.raises(SystemExit, match="is not inside"):
        ep.main(["--policy", "recordcamp/candidate", "--seeds", "0-1",
                 "--root", str(root)])
    assert list(outside.iterdir()) == []
    assert _new_paths(tmp_path, before) == []


def test_the_anchor_is_the_argument_and_not_the_campaigns_own_directory(tmp_path, ep, monkeypatch):
    """Why the anchor is captured at argument time, in one case.

    Anchoring on `record.root` instead -- the obvious-looking rewrite, and the
    one that keeps every other check in this file green -- passes a symlinked
    CAMPAIGN directory straight through: the anchor follows the link and the
    record is contained in the tree the link points at. `--root` is the only
    path in this branch the measured artifact cannot edit, so it is the only
    one a containment claim can rest on.

    `is_symlink` is stubbed again, so the anchor is the only guard left; without
    the stub the refusal above fires first and this mutation stays invisible.
    """
    monkeypatch.setattr(ep.Path, "is_symlink", lambda self: False)
    root = tmp_path / "sandbox" / "scripts"
    root.mkdir(parents=True)
    outside = tmp_path / "outside_camp"
    _write_campaign(outside)
    (root / "recordcamp").symlink_to(outside / "recordcamp")
    before = {p for p in tmp_path.rglob("*")}
    with pytest.raises(SystemExit, match="is not inside"):
        ep.main(["--policy", "recordcamp/candidate", "--seeds", "0-1",
                 "--root", str(root)])
    assert _new_paths(tmp_path, before) == []


def test_a_symlink_at_the_record_path_itself_is_already_closed(tmp_path, ep, monkeypatch):
    """The deeper case, and it needed no new code -- so it gets a check.

    A link one level below `records/`, at the record's own filename, is refused
    TWICE over, by refusals that exist independently of the link guard:

    - `_resolve_under` resolves the NAME, so a link pointing out of `records/`
      resolves out of `records/` and comes back `None`. That is the guard doing
      exactly what its docstring claims, on the half of the path it owns;
    - and `open(out, "x")` refuses any existing path, a dangling link included
      (`O_CREAT|O_EXCL` fails with `EEXIST` on a symlink whatever it points
      at), which is what `test_a_name_collision_over_different_bytes_is_refused_
      not_overwritten` pins.

    The first is what fires here, because it comes first. Pinned so that a
    future edit that stops resolving the name does not silently hand this case
    to the exclusive create, whose refusal would read as a digest collision.
    """
    monkeypatch.setattr(ep, "_record_name", lambda pid, doc: "eval_fixed.json")
    root = tmp_path / "sandbox" / "scripts"
    camp = _write_campaign(root)
    outside = tmp_path / "outside_target"
    outside.mkdir()
    target = outside / "victim.json"
    target.write_text('{"not": "a record"}\n')
    (camp / "records").mkdir()
    (camp / "records" / "eval_fixed.json").symlink_to(target)
    with pytest.raises(SystemExit, match="does not resolve inside"):
        ep.main(["--policy", "recordcamp/candidate", "--seeds", "0-1",
                 "--root", str(root)])
    assert target.read_text() == '{"not": "a record"}\n'


# ==========================================================================
# The name is a function of the record's bytes, so a collision is not a re-run
# ==========================================================================

class _FrozenClock:
    """`datetime` with `now()` pinned, so two runs produce one document."""

    class datetime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 6, 12, 34, 56, 789012,
                       tzinfo=_dt.timezone.utc)


def test_a_byte_identical_re_record_is_accepted_and_not_duplicated(tmp_path, ep, monkeypatch):
    """Two runs that produce the SAME document write one file, and exit 0.

    The name is a digest of the record, so a name that already exists is either
    this same record already written -- nothing is lost by leaving it -- or a
    short-digest collision, which is the next test. `write_text` could not tell
    those apart, which is v1's defect in miniature.
    """
    monkeypatch.setattr(ep, "_dt", _FrozenClock)
    _write_campaign(tmp_path)
    first = _run(ep, tmp_path)
    text = first[0].read_text()
    second = _run(ep, tmp_path)
    assert [p.name for p in second] == [p.name for p in first]
    assert second[0].read_text() == text


def test_a_name_collision_over_different_bytes_is_refused_not_overwritten(tmp_path, ep, monkeypatch):
    """The residual collision case, made reachable and made loud.

    A same-`measured_at` collision cannot silently replace a DIFFERENT record,
    because the name is a function of the bytes -- so the only way two distinct
    documents can share a name is an 8-hex-digest collision. Forced here by
    pinning the name, and the writer refuses rather than replacing in place.
    """
    monkeypatch.setattr(ep, "_record_name", lambda pid, doc: "eval_fixed.json")
    _write_campaign(tmp_path)
    first = _run(ep, tmp_path)
    assert [p.name for p in first] == ["eval_fixed.json"]
    with pytest.raises(SystemExit, match="digest collision"):
        ep.main(["--policy", "recordcamp/candidate", "--seeds", "0-2",
                 "--root", str(tmp_path)])
    # The first record is still the first record.
    assert json.loads(first[0].read_text())["measured_at"] \
        == json.loads((tmp_path / "recordcamp" / "records" / "eval_fixed.json").read_text())["measured_at"]


def test_out_stays_an_operators_explicit_path(tmp_path, ep):
    """`--out` is not guarded, and that is deliberate.

    This module's own usage example passes `--out /tmp/reeval.json`: a
    deliberate name, deliberately overwritten on a re-measure. The guards above
    are about the DERIVED path,
    whose name half comes out of a manifest.
    """
    _write_campaign(tmp_path)
    out = tmp_path / "elsewhere" / "chosen.json"
    assert ep.main(["--policy", "recordcamp/candidate", "--seeds", "0-1",
                    "--root", str(tmp_path), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["policy"] == "recordcamp/candidate"
    assert ep.main(["--policy", "recordcamp/candidate", "--seeds", "0-1",
                    "--root", str(tmp_path), "--out", str(out)]) == 0


def test_the_write_site_refuses_a_derived_name_that_escapes_records(tmp_path, ep, monkeypatch):
    """The containment guard's CALL SITE, not just the helper.

    With the id guard in place no CLI input can reach this branch -- which is
    what makes it belt as well as braces, and also what makes it invisible to
    every other test in this file. `_record_name` is stubbed to a traversing
    name so the second guard is the only thing standing between the manifest
    and a write outside `records/`; without the check in `main` this test
    writes a file one directory up and goes green.
    """
    monkeypatch.setattr(ep, "_record_name", lambda pid, doc: "../escaped.json")
    root = tmp_path / "sandbox" / "scripts"
    _write_campaign(root)
    before = {p for p in tmp_path.rglob("*")}
    with pytest.raises(SystemExit, match="does not resolve inside"):
        ep.main(["--policy", "recordcamp/candidate", "--seeds", "0-1",
                 "--root", str(root)])
    assert _new_paths(tmp_path, before) == []
