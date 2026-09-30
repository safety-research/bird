"""Every committed policy record validates against the schema it names.

`tests/test_eval_policy_record.py` validates only the record `scripts/eval_policy.py` has
JUST written -- so if `policies/records.schema.json` (`bird-policy-eval-v2`) gained a
`required` key, every v2 record already committed would stop validating against the
schema named in its own first line, and no other test would notice.  Each of them is
a manifest's `score.record`, so the schema would refuse exactly the records the
manifests rest on.  This file is that check, and the reason the later-added
`horizon` and `max_steps` keys are OPTIONAL in the schema: absent means "written
before the pair existed, uncapped as far as the record knows".

Two things this file deliberately does not do.

1. It does not `pytest.skip` the families it cannot validate: a permanent skip is a
   check that never runs.  The families are NAMED instead, in `UNVALIDATED`,
   each with the reason no validator exists -- and a record naming any schema id that
   is in neither table fails, because a record nobody can check is worse than a v1.
2. It does not validate v1 records against the v2 schema.  v2 is a superset, so every
   v1 record fails it by construction, and rewriting the committed v1 records to pass
   would retype numbers this repo's whole point is that nobody retypes
   (records.schema.json's own description).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from bird.policies import index, policies_root
from test_eval_policy_record import _records_schema

REPO = Path(__file__).resolve().parent.parent
ROOT = policies_root()

#: Every committed per-seed record: `policies/<campaign>/records/*.json`, whatever
#: shape it is in.  Collected once; the tests below partition it by the schema each
#: record names.
RECORDS = sorted(p for p in ROOT.rglob("*.json") if p.parent.name == "records")

#: Schema ids this repo ships a validator for, and the file that is that validator.
VALIDATED = {
    "bird-policy-eval-v2": "records.schema.json",
}

#: Schema ids (and the no-schema case, keyed None) this repo knows it CANNOT validate,
#: each with the reason.  Closed on purpose: a new id lands here with a sentence, or it
#: lands in VALIDATED with a schema file -- never in neither.
UNVALIDATED = {
    "bird-policy-eval-v1": (
        "the shape scripts/eval_policy.py wrote before v2; no schema file describes it, and "
        "records.schema.json says v2 must not"),
    None: (
        "no `schema` key at all -- a campaign's own per-seed log, an object or a bare list, "
        "whose shape is the campaign's"),
}


def _schema_name(doc):
    return doc.get("schema") if isinstance(doc, dict) else None


def _census():
    """{schema id or None: [(path, doc), ...]} over every committed record."""
    groups = defaultdict(list)
    for path in RECORDS:
        doc = json.loads(path.read_text())
        groups[_schema_name(doc)].append((path, doc))
    return groups


def test_the_corpus_is_where_this_file_looks():
    # The glob is the whole check's reach: a records/ dir it misses is a family of
    # records nothing validates. Every manifest-cited record must be inside it.
    assert RECORDS, "policies/*/records/ holds no JSON records at all"
    cited = {(rec.root / rec.score["record"]).resolve()
             for rec in index(refresh=True).values() if rec.score.get("record")}
    outside = sorted(str(p.relative_to(REPO)) for p in cited - {p.resolve() for p in RECORDS})
    assert not outside, (
        f"{len(outside)} manifest score.record file(s) lie outside the records/ dirs this "
        f"file validates:\n  " + "\n  ".join(outside))


def test_every_record_names_a_schema_this_file_has_a_verdict_on():
    census = _census()
    unknown = sorted((name for name in census if name not in VALIDATED and name not in UNVALIDATED),
                     key=str)
    assert not unknown, (
        f"records name schema id(s) {unknown} that this file can neither validate nor account "
        "for. Either add the schema file under policies/ and list the id in VALIDATED, or "
        "list it in UNVALIDATED with the reason -- a record nobody can check is not an option.\n"
        + "\n".join(f"  {name}: {len(census[name])} record(s), e.g. "
                    f"{census[name][0][0].relative_to(REPO)}" for name in unknown))


def test_every_committed_v2_record_validates_against_the_schema_it_names():
    jsonschema = pytest.importorskip("jsonschema")
    schema = _records_schema()
    v2 = _census().get("bird-policy-eval-v2", [])
    # 5 shipped in the release, under policies/h1hand_hb_hacks/records/ and
    # policies/simple_suite/records/. This count only grows; a drop is a manifest whose
    # backing record went away and needs a reason in the same change.
    assert len(v2) >= 5, (
        f"only {len(v2)} committed record(s) name bird-policy-eval-v2 (5 in the release); "
        "the check below exercises fewer committed records than it did when written")
    validator = jsonschema.validators.validator_for(schema)(schema)
    failures = []
    for path, doc in v2:
        for err in sorted(validator.iter_errors(doc), key=lambda e: list(map(str, e.absolute_path))):
            where = "/".join(map(str, err.absolute_path))
            failures.append(f"{path.relative_to(REPO)}: {err.message}" + (f" (at {where})" if where else ""))
    assert not failures, (
        f"{len(failures)} problem(s) across the {len(v2)} committed bird-policy-eval-v2 record(s), "
        "each of which names policies/records.schema.json as its contract. A committed record "
        "is never rewritten to fit a schema (the numbers are the point); the schema moves, and "
        "a key the writer gained after these were written is OPTIONAL with its absence given a "
        "meaning (horizon/max_steps: 'written before the pair existed, uncapped as far as the "
        "record knows').\n  " + "\n  ".join(failures))
