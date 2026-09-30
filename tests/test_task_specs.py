"""`tasks/` -- the base task definition, and the invariants that keep it honest.

Three layers, in the order they fail usefully:

  * STRUCTURE, in pure Python (pyyaml only, so it always runs). Every spec parses,
    declares the twelve required groups, and its `id` is its directory name.
  * THE HONESTY RULES, also pure Python. The schema's whole argument is that a value
    nobody obtained is `null` WITH a reason rather than omitted or zero-filled, so
    these are the rules BIRD actually leans on when it reads an anchor or a threshold.
    They are re-checked here rather than delegated to jsonschema because jsonschema is
    not a hard dependency and `--validate-all` must run on pyyaml + numpy alone.
  * FULL 2020-12 VALIDATION, behind `importorskip` (jsonschema is an optional
    dependency), mirroring how the sim-guarded tests here skip without mujoco.

And one thing that is not about the files at all: the COVERAGE PARTITION. A spec is not
a claim that this repo can run the task. The registry is the `problem.env_id` enum, so
every registered env must resolve to exactly one spec and every spec must either back a
registered env or be named in `_no_adapter.json` with a reason. Asserted in both
directions, because a spec that quietly gained an adapter and a spec that quietly lost
one are different bugs and neither should be silent.

And one more, for `env.reset` (tasks/SCHEMA_DELTA.md #14): THE BLOCK IS HELD AGAINST
THE ADAPTER. Every other field here is checked for shape and honesty; `env.reset` claims
what one seeded `_reset` MOVES, and a claim about code is checked against the code --
`entry` is resolved with `ast` to the class the registry binds, and wherever the
simulator is installed 24 seeded resets are stacked and the union of `fields` must equal
the set of columns that moved, in both directions. The failure this removes is the one
that renders best: a spec saying "the seed changes: goal" on a task where only y moves.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest
import yaml

#: Not in the tester-tier smoke suite (catalogue sweep: one invariant per task spec).
#: Deselected by `-m "not slow"`; run it with `-m slow`. See pyproject.toml.
pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"

#: The schema's top-level `required`. Duplicated here on purpose: this test must fail
#: with a readable message on a machine that has no jsonschema.
REQUIRED_GROUPS = (
    "schema_version", "id", "env", "description", "state_surface", "discrete_success",
    "continuous_success", "anchors", "budget", "reward", "judge", "provenance",
)

SPEC_FILE_NAME = "shared_spec.yaml"
SPEC_PATHS = sorted(TASKS.glob(f"*/{SPEC_FILE_NAME}"))
SPEC_IDS = [p.parent.name for p in SPEC_PATHS]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@pytest.fixture(scope="module")
def specs() -> dict:
    return {p.parent.name: _load(p) for p in SPEC_PATHS}


def test_the_catalogue_is_not_empty():
    """A glob that silently matches nothing would make every test below vacuous."""
    assert SPEC_PATHS, f"no shared_spec.yaml under {TASKS}"


@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_every_spec_parses_and_declares_the_required_groups(path):
    doc = _load(path)
    assert isinstance(doc, dict), f"{path} is not a mapping"
    missing = [g for g in REQUIRED_GROUPS if g not in doc]
    assert not missing, f"{path.parent.name} is missing {missing}"
    assert doc["schema_version"] == 1, (
        f"{path.parent.name}: schema_version {doc['schema_version']!r}, expected 1. "
        "Note REQUIRED fields have been added without bumping this, so it is a "
        "shape marker and not a compatibility signal -- see tasks/SCHEMA_DELTA.md")
    assert doc["id"] == path.parent.name, (
        f"{path}: id {doc['id']!r} != directory {path.parent.name!r}. The id is the "
        "catalogue's only unique key -- `env.env_id` is NOT unique (HalfCheetah-v5 "
        "backs three specs) -- so a mismatch makes every lookup ambiguous")


@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_an_unobtained_value_carries_a_reason(path):
    """Absence is explicit: `null` WITH a reason, never omitted and never zero-filled.

    This is the rule BIRD depends on most directly. `SpecEnvAdapter` turns a null anchor
    into `baselines = None`, which makes `_normalise_pool` leave fitness raw rather than
    invent a scale -- so a null anchor that lost its reason would become a silently
    unnormalised number rather than a stated gap.
    """
    doc = _load(path)

    for which in ("random", "expert"):
        anchor = doc["anchors"][which]
        if anchor.get("value") is None:
            assert (anchor.get("reason") or "").strip(), (
                f"{path.parent.name}: anchors.{which}.value is null with no reason")

    ds = doc["discrete_success"]
    if ds["kind"] == "continuous_only":
        assert (ds.get("no_discrete_success_because") or "").strip(), (
            f"{path.parent.name}: kind=continuous_only with no reason. A task with no "
            "pass/fail check is a fact about the task, not an unfilled field")
        assert ds.get("shipped") is None and ds.get("hardened") is None, (
            f"{path.parent.name}: kind=continuous_only but a success-check block is present")
    else:
        hardened = ds.get("hardened") or {}
        if hardened.get("status") == "pending":
            assert (hardened.get("pending_plan") or "").strip(), (
                f"{path.parent.name}: hardened oracle pending with no plan. An oracle "
                "that does not exist yet is a state, not an omission")
            assert hardened.get("module") is None, (
                f"{path.parent.name}: pending oracle names a module")
        elif hardened.get("tightened") is False:
            assert (hardened.get("not_tightened_because") or "").strip(), (
                f"{path.parent.name}: shipped success check left untightened with no reason. "
                "Declining to tighten is as much a decision as tightening")


@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_the_normalisation_has_exactly_one_source(path):
    """`method: anchors` derives the scale; `method: explicit` states it. Never both.

    The schema does not enforce this with an `if`/`then`; this test is that conditional.
    """
    norm = _load(path)["continuous_success"]["normalized"]
    if norm["method"] == "anchors":
        assert norm.get("formula") is None, (
            f"{path.parent.name}: method=anchors but a formula is given; two sources "
            "of truth for one scale")
    else:
        assert (norm.get("formula") or "").strip(), (
            f"{path.parent.name}: method=explicit with no formula")


@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_an_anchor_pair_that_cannot_normalise_is_rejected(path):
    """`clip((raw - random) / (expert - random), 0, 1)` needs a non-zero span."""
    a = _load(path)["anchors"]
    lo, hi = a["random"].get("value"), a["expert"].get("value")
    if lo is not None and hi is not None:
        assert lo != hi, (
            f"{path.parent.name}: random and expert anchors are both {lo}; the "
            "normalisation denominator is zero")


#: The specs carrying a `fasttd3` published baseline -- an EQUALITY, not a floor, since a
#: floor catches only shrinkage: FastTD3's released data covers 29 h1hand tasks, these are
#: the 24 of them with specs in this catalogue, and a quiet gain or loss of one should
#: fail loudly here rather than read as intended. The five paper tasks with no spec -- bookshelf_simple, insert_normal,
#: insert_small, spoon, truck -- get no entry anywhere, and `h1strong_highbar_hard` is
#: absent twice over: the paper reports no highbar task, and its robot variant is not
#: h1hand -- a number from one variant never fills the other's row. A NEW h1hand spec whose task FastTD3 reports fails
#: this equality on purpose: populate it from the released data (the method is in the
#: entries' own `source` text) or record here why it stays blank.
FASTTD3_SPECS = {
    "h1hand_balance_hard", "h1hand_balance_simple", "h1hand_basketball",
    "h1hand_bookshelf_hard", "h1hand_cabinet", "h1hand_crawl", "h1hand_cube",
    "h1hand_door", "h1hand_hurdle", "h1hand_maze", "h1hand_package",
    "h1hand_pole", "h1hand_powerlift", "h1hand_push", "h1hand_reach",
    "h1hand_room", "h1hand_run", "h1hand_sit_hard", "h1hand_sit_simple",
    "h1hand_slide", "h1hand_stair", "h1hand_stand", "h1hand_walk",
    "h1hand_window",
}


def test_the_fasttd3_baseline_is_on_exactly_the_reported_specs(specs):
    """Enter only tasks the source actually reports; leave the rest absent, never
    interpolated. Both directions matter: an entry appearing on a spec the paper does
    not report would be a fabricated pin, and one vanishing from a spec it does report
    would be a silent loss of the only expert-column source the HB tier has."""
    have = {tid for tid, doc in specs.items()
            if (doc["anchors"].get("published") or {}).get("fasttd3")}
    assert have == FASTTD3_SPECS, (
        f"unexpected: {sorted(have - FASTTD3_SPECS)}; missing: {sorted(FASTTD3_SPECS - have)}")


def test_a_published_baseline_is_verbatim_and_pinned(specs):
    """The block's whole value is provenance: a number without its unit, its spread, its
    budget, its success bar and its source down to figure/page is just a number, and
    per-stat sourcing depends on every one of those travelling with it. The substrings pinned here are the pins themselves -- the arXiv id, the figure,
    the released data file and its commit, and the benchmark commit the bar was read at."""
    for tid in sorted(FASTTD3_SPECS):
        entry = specs[tid]["anchors"]["published"]["fasttd3"]
        ctx = f"{tid}: anchors.published.fasttd3"
        assert isinstance(entry["value"], (int, float)) and \
            not isinstance(entry["value"], bool), f"{ctx}.value is not a number"
        assert entry["unit"] == "hb_episode_return", (
            f"{ctx}.unit: the FastTD3 numbers are HB episode returns; a different unit "
            "here means the entry was rewritten without re-reading the source")
        assert entry.get("algo") == "FastTD3", f"{ctx}.algo"
        assert entry.get("n_runs") == 3, (
            f"{ctx}.n_runs: the paper states mean and std across three runs")
        assert isinstance(entry.get("std"), (int, float)) and entry["std"] >= 0, f"{ctx}.std"
        assert (entry.get("env_steps") or 0) > 0, f"{ctx}.env_steps"
        assert (entry.get("success_bar") or 0) > 0, f"{ctx}.success_bar"
        for needle in ("2505.22642", "Figure 9", "humanoidbench_result.json", "77922d0"):
            assert needle in entry["source"], f"{ctx}.source lost the pin {needle!r}"
        assert "cb11890" in (entry.get("success_bar_source") or ""), (
            f"{ctx}.success_bar_source must pin the humanoid-bench commit the bar was "
            "read at -- the specs' own pinned commit, so the two cannot drift apart")
        assert "task_metric" in (entry.get("note") or ""), (
            f"{ctx}.note must keep saying what this number is NOT: a paper's episode "
            "return normalising a 0-1 task_metric is the exact confusion the separate "
            "block exists to prevent")


#: Wall-time in seconds at which every Figure 9 panel's x-axis ends.
FASTTD3_FIGURE_XLIM_S = 10800

#: task id -> (value, success_bar, final_wall_time_s), an EQUALITY on the numbers
#: themselves. The tests above check that an entry is SHAPED right and PINNED; neither
#: could fail if a value drifted, so a bad `sed`, a hand-edit or a re-run of a generator
#: against different data could change 24 numbers silently. A measured constant is an
#: equality, and this is not a floor or a digest because the failure has to name the task
#: and print both numbers: the fix is different depending on which side is wrong.
#:
#: All three are read off the SAME released curve -- the final point of
#: `data/humanoidbench_result.json` @ 77922d0 in younggyoseo/FastTD3 -- except the bar,
#: which is humanoid-bench source at cb11890. Re-derive from those two sources if you
#: need to move one, never from the table below.
#:
#: The third column is here rather than left to a file read at test time: the released
#: JSON is NOT committed, so a caveat check reading it would skip silently in every
#: checkout without it -- a guard that does not run, wearing the clothes of one that
#: passed. A pinned number that must equal its source is the honest form of the same fact.
FASTTD3_PINS = {
    "h1hand_balance_hard": (244.00, 800.0, 36090),
    "h1hand_balance_simple": (788.44, 800.0, 7029),
    "h1hand_basketball": (528.98, 1200.0, 15147),
    "h1hand_bookshelf_hard": (723.02, 2000.0, 16312),
    "h1hand_cabinet": (188.23, 2500.0, 9640),
    "h1hand_crawl": (956.24, 700.0, 4564),
    "h1hand_cube": (221.20, 370.0, 7674),
    "h1hand_door": (331.75, 600.0, 9484),
    "h1hand_hurdle": (886.83, 700.0, 11207),
    "h1hand_maze": (358.68, 1200.0, 5232),
    "h1hand_package": (-7487.25, 1500.0, 6816),
    "h1hand_pole": (867.14, 700.0, 11564),
    "h1hand_powerlift": (324.71, 800.0, 9926),
    "h1hand_push": (769.57, 700.0, 58772),
    "h1hand_reach": (8146.56, 12000.0, 11184),
    "h1hand_room": (176.97, 400.0, 9301),
    "h1hand_run": (902.02, 700.0, 4597),
    "h1hand_sit_hard": (726.08, 750.0, 10652),
    "h1hand_sit_simple": (929.01, 750.0, 3169),
    "h1hand_slide": (904.02, 700.0, 9279),
    "h1hand_stair": (689.62, 700.0, 11622),
    "h1hand_stand": (911.93, 800.0, 3256),
    "h1hand_walk": (928.26, 700.0, 3274),
    "h1hand_window": (613.33, 650.0, 18783),
}


def test_the_published_numbers_are_pinned(specs):
    """Every recorded value and bar, as an equality. See `FASTTD3_PINS`."""
    assert set(FASTTD3_PINS) == FASTTD3_SPECS, (
        "the pin table and the coverage set disagree about which specs carry an entry")
    for task_id, (value, bar, _) in sorted(FASTTD3_PINS.items()):
        entry = specs[task_id]["anchors"]["published"]["fasttd3"]
        assert entry["value"] == pytest.approx(value), (
            f"{task_id}: value {entry['value']} != pinned {value}")
        assert entry["success_bar"] == pytest.approx(bar), (
            f"{task_id}: success_bar {entry['success_bar']} != pinned {bar}")


def test_a_published_note_agrees_with_its_own_arithmetic(specs):
    """The note states a clears/below verdict; `value` and `success_bar` state the same
    thing as numbers. Two renderings of one fact, so they can disagree -- and a note is
    what a reader acts on, while the numbers are what a script reads.

    Also asserts the two caveats are applied by MEASUREMENT rather than by hand, since by
    hand they are applied inconsistently:

      * an entry whose released log runs past Figure 9's 10,800 s x-limit must say so;
      * an entry whose own `value +/- std` straddles the bar must not leave a bare
        verdict standing: the three runs behind the mean disagree about which side of the
        bar the task fell on, and "clears the 700 bar" is not a claim 769.57 +/- 115.01
        carries.
    """
    for task_id in sorted(FASTTD3_SPECS):
        entry = specs[task_id]["anchors"]["published"]["fasttd3"]
        note, value, bar, std = (entry["note"], entry["value"],
                                 entry["success_bar"], entry["std"])
        ctx = f"{task_id}: anchors.published.fasttd3"

        clears, below = "Clears the task's" in note, "Below the task's" in note
        assert clears != below, (
            f"{ctx}.note states neither exactly one of clears/below -- the verdict is "
            "the one sentence a reader takes away")
        assert clears == (value > bar), (
            f"{ctx}.note says {'clears' if clears else 'below'} but {value} vs bar {bar} "
            f"is {'clears' if value > bar else 'below'}")

        if value - std <= bar <= value + std:
            assert "INSIDE THE SPREAD" in note, (
                f"{ctx}: {value} +/- {std} straddles the bar {bar}, so the bare verdict "
                "overstates what three runs agreed on. Say so in the note")

        assert "hb_episode_return" == entry["unit"], ctx

    # The x-limit caveat, driven off the pinned wall-times so it runs in every checkout.
    # Asserted in BOTH directions: an entry inside the window must not claim a caveat it
    # has no basis for, which is the mirror of the omission.
    for task_id in sorted(FASTTD3_SPECS):
        note = specs[task_id]["anchors"]["published"]["fasttd3"]["note"]
        final_s = FASTTD3_PINS[task_id][2]
        claims = "x-limit" in note and "visible end" in note
        if final_s > FASTTD3_FIGURE_XLIM_S:
            assert claims, (
                f"{task_id}: its released log runs to {final_s} s, past Figure 9's "
                f"{FASTTD3_FIGURE_XLIM_S} s x-limit, so the figure a reader sees and this "
                "value differ. The note must say so")
        else:
            assert not claims, (
                f"{task_id}: its log ends at {final_s} s, inside Figure 9's window, so "
                "the note must not carry an extended-log caveat -- there is nothing "
                "beyond the plotted curve for it to be about")


def test_a_published_entry_cites_its_own_task(specs):
    """The `source` prose must name the task whose panel it read.

    An entry copied from another spec would still carry every substring pin checked
    above, because those pins are shared, but it would name the other task's panel.
    Requiring the entry to name its own task key is what catches a copied source.
    """
    for task_id in sorted(FASTTD3_SPECS):
        entry = specs[task_id]["anchors"]["published"]["fasttd3"]
        assert task_id in entry["source"], (
            f"{task_id}: its `source` does not name the {task_id} panel; an entry "
            "copied from another spec cites the wrong panel for its numbers")


def test_every_spec_validates(specs):
    """Full 2020-12 validation, gated: jsonschema is not a hard dependency here."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((TASKS / "shared_spec.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    problems = []
    for task_id, doc in specs.items():
        for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
            problems.append(f"{task_id}: {'/'.join(map(str, err.path)) or '<root>'}: "
                            f"{err.message}")
    assert not problems, "\n  - " + "\n  - ".join(problems)


# --------------------------------------------------------------------------
# the loader
# --------------------------------------------------------------------------
#
# The tests above check the CORPUS by reading YAML directly; these check
# `bird.tasks._check` by feeding it corpus documents with one field broken. Keeping the
# two independent is the point: a corpus test written on top of the loader could only
# ever confirm the loader agrees with itself.


def test_the_index_loads_every_spec():
    from bird import tasks

    assert set(tasks.index()) == set(SPEC_IDS)
    assert tasks.available() == tuple(sorted(SPEC_IDS))


def test_load_refuses_a_path_that_escapes_the_catalogue():
    """`problem.task_id` reaches this from a config file and from `-s` on a command line,
    and run dirs on a shared mount may be world-writable.

    `match="outside"`, NOT `match="outside|unknown"`. The looser regex is vacuous: every
    traversal input also fails the `index()` lookup that follows, with an "unknown task"
    message, so deleting the containment check entirely would leave the test green. It
    would assert that SOMETHING refused, which is never in doubt.

    Both layers are real and this pins the first one specifically. The second --
    `index()[task_id]` being a closed dict rather than a filesystem read -- is what makes
    the failure safe rather than merely reported, and is pinned by
    `test_an_unknown_task_names_what_is_available`.
    """
    from bird import tasks

    for bad in ("../../etc/passwd", "../door_open", "/etc", "sub/../../escape"):
        with pytest.raises(tasks.TaskSpecError, match="outside"):
            tasks.load(bad)


def test_an_unknown_task_names_what_is_available():
    from bird import tasks

    with pytest.raises(tasks.TaskSpecError) as exc:
        tasks.load("windo_open")
    assert "window_open" in str(exc.value), "should list the catalogue, as _near_miss does"


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d.pop("anchors"), "missing required group"),
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d.update(id="not-the-directory"), "does not match its directory"),
    (lambda d: d["anchors"]["expert"].update(value=None, reason=None), "null with no reason"),
    (lambda d: d["anchors"]["expert"].update(value=None, reason="  "), "null with no reason"),
    (lambda d: d["continuous_success"]["normalized"].update(
        method="anchors", formula="raw / 2"), "two sources of truth"),
    (lambda d: d["continuous_success"]["normalized"].update(
        method="explicit", formula=None), "no formula"),
])
def test_the_loader_rejects_a_broken_spec(mutate, expect):
    """Each mutation is a real failure mode, not a syntax error: a group that was never
    filled, a value nobody measured recorded as a bare null, two scales for one number."""
    import copy

    from bird import tasks

    doc = copy.deepcopy(_load(TASKS / "window_open" / SPEC_FILE_NAME))
    mutate(doc)
    with pytest.raises(tasks.TaskSpecError, match=expect):
        tasks._check(doc, TASKS / "window_open" / SPEC_FILE_NAME)


@pytest.mark.parametrize("task_id,mutate,expect", [
    ("half_cheetah", lambda d: d["discrete_success"].update(
        no_discrete_success_because=None), "continuous_only with no reason"),
    ("window_open", lambda d: d["discrete_success"]["hardened"].update(
        pending_plan=None), "pending with no plan"),
    ("window_open", lambda d: d["discrete_success"]["hardened"].update(
        status="written", tightened=False, not_tightened_because=None),
     "untightened with no reason"),
])
def test_the_loader_rejects_an_unexplained_absence(task_id, mutate, expect):
    """Aimed at a spec that really is in the state being tested, so the branch cannot
    quietly stop being exercised: `half_cheetah` is genuinely `continuous_only` (gym
    ships no success check) and `window_open`'s oracle is genuinely pending."""
    import copy

    from bird import tasks

    path = TASKS / task_id / SPEC_FILE_NAME
    doc = copy.deepcopy(_load(path))
    mutate(doc)
    with pytest.raises(tasks.TaskSpecError, match=expect):
        tasks._check(doc, path)


# --------------------------------------------------------------------------
# the coverage partition
# --------------------------------------------------------------------------


def test_no_adapter_entries_are_specs_and_carry_a_reason():
    from bird import tasks

    ledger = tasks.no_adapter_ledger()
    unknown = sorted(set(ledger) - set(tasks.index()))
    assert not unknown, f"_no_adapter.json names specs that do not exist: {unknown}"
    for task_id, entry in ledger.items():
        assert (entry.get("reason") or "").strip(), (
            f"_no_adapter.json: {task_id} has no reason. An unexplained exemption is "
            "how a gap becomes permanent")


def test_every_spec_either_backs_a_registered_env_or_says_why_not():
    """A spec is DATA, not a claim that this repo can run the task.

    The registry is the `problem.env_id` enum, so a spec with no adapter can never become
    a config value with no implementation behind it -- a fabricated pin. What it CAN do is
    sit here unexplained, and that is what `_no_adapter.json` prevents. Asserted in both directions: a spec that quietly gained an adapter and one
    that quietly lost its spec are different bugs and neither should be silent.
    """
    from bird import registry, tasks

    registry.load_all()
    envs = set(registry.names("env"))
    ledger = tasks.no_adapter_ledger()

    unexplained, wrongly_listed = [], []
    for task_id, spec in tasks.index().items():
        backed = spec.bird_env_id in envs
        if backed and task_id in ledger:
            wrongly_listed.append(task_id)
        elif not backed and task_id not in ledger:
            unexplained.append(task_id)

    assert not unexplained, (
        f"specs with no adapter and no entry in _no_adapter.json: {sorted(unexplained)}")
    assert not wrongly_listed, (
        f"specs listed in _no_adapter.json that DO back a registered env: "
        f"{sorted(wrongly_listed)} -- an adapter landed and the ledger was not updated")


def test_no_two_specs_claim_one_env_id():
    """`env.env_id` is NOT unique -- HalfCheetah-v5 backs three specs -- so `by_env_id`
    refuses an ambiguous match rather than picking. Nothing collides after the
    `mt10_` derivation; this pins that, and pins the refusal."""
    from bird import registry, tasks

    registry.load_all()
    for env_id in sorted(registry.names("env")):
        tasks.by_env_id(env_id)          # raises TaskSpecError on a collision

    raw = [s.env_id for s in tasks.index().values()]
    assert len(raw) > len(set(raw)), (
        "no benchmark env id is shared any more -- if the catalogue really lost its "
        "objective variants, drop this test; if not, the fixture stopped being loaded")


# --------------------------------------------------------------------------
# the fields BIRD authored (tasks/SCHEMA_DELTA.md)
# --------------------------------------------------------------------------
#
# These are BIRD's additions to the core schema, so nothing else tests them. They are also the fields a
# generated reward is written against, which makes a wrong one worse than a missing one:
# a hallucinated `s.` field fails loudly at verification, but a field documented with the
# wrong slice produces a reward that runs, trains, scores, and measures the wrong thing.


def _mt10():
    from bird import tasks

    return {s.id: s for s in tasks.index().values() if s.library == "metaworld"}


def test_the_fifty_metaworld_specs_are_the_fifty_registered_tasks():
    """MT10 as `mt10_<key>`, the other forty of MT50 as `mt50_<key>`, and
    the catalogue's Meta-World specs are exactly the registered set. An equality in both
    directions and on the counts: a spec without an adapter or an adapter without a spec
    fails here rather than in a config."""
    from bird import registry
    from bird.tasks import METAWORLD_MT10

    registry.load_all()
    mt10 = {n for n in registry.names("env") if n.startswith("mt10_")}
    mt50 = {n for n in registry.names("env") if n.startswith("mt50_")}
    assert {s.bird_env_id for s in _mt10().values()} == mt10 | mt50
    assert len(mt10) == 10 and len(mt50) == 40
    assert mt10 == {"mt10_" + k for k in METAWORLD_MT10}
    assert not {n[len("mt50_"):] for n in mt50} & set(METAWORLD_MT10), (
        "an MT10 task registered twice, under both prefixes")


@pytest.mark.parametrize("task_id", sorted(_mt10()))
def test_flat_fields_is_the_adapters_observation_table(task_id):
    """The identity. `flat_fields` is authored FROM `_STATE_FIELDS`, so while that table
    exists the two must agree exactly -- which is what makes the spec a copy rather than
    a fork."""
    from bird.envs.metaworld import _STATE_FIELDS

    spec = _mt10()[task_id]
    authored = [(e["name"], e["description"]) for e in spec.state_surface["flat_fields"]]
    assert authored == list(_STATE_FIELDS)
    assert len(authored) == spec.env["spaces"]["obs_dim"] == 39


@pytest.mark.parametrize("task_id", sorted(_mt10()))
def test_action_fields_is_the_adapters_action_table(task_id):
    from bird.envs.metaworld import _ACTION_FIELDS

    spec = _mt10()[task_id]
    fields = spec.env["spaces"]["action_fields"]
    assert [(f["name"], f["description"]) for f in fields] == list(_ACTION_FIELDS)
    assert len(fields) == spec.env["spaces"]["action_dim"] == 4


def _sliced():
    """Every spec whose `fields` carry a `slice`, keyed by id.

    WIDER THAN `_mt10()`, and the widening is the point rather than tidiness: guards
    hard-coded to the ten Meta-World specs' 39-D observation leave wider specs -- with a
    `fields` view over a `flat_fields` table -- outside them. Over the wider set both
    guards catch real defect shapes: `room` helpers whose `expression` carries a free
    variable `k`, and a `control_cost` written over `action` where every renderer in
    this repo binds `a`.

    Discovered by spec, not listed: a spec with no sliced field simply does not
    parametrise, so a future suite needs no edit here.
    """
    from bird import tasks

    return {tid: spec for tid, spec in tasks.index().items()
            if any(f.get("slice") for f in spec.state_surface.get("fields") or ())}


def _obs_dim(spec):
    return int((spec.env.get("spaces") or {})["obs_dim"])


@pytest.mark.parametrize("task_id", sorted(_sliced()))
def test_every_declared_slice_matches_its_declared_shape(task_id):
    """`fields` and `flat_fields` describe one observation at two granularities, and
    `slice` is the only thing that can catch them disagreeing. A slice whose width is not
    its shape means one of the two is wrong about the layout.

    The bound is the spec's own `obs_dim` rather than a literal 39, so the check means
    the same thing on a 229-D observation as on Meta-World's.
    """
    import re as _re

    spec = _sliced()[task_id]
    n = _obs_dim(spec)
    for f in spec.state_surface["fields"]:
        if not f.get("slice"):
            continue
        lo, hi = (int(x) for x in f["slice"].split(":"))
        shape = str(f["shape"]).strip()
        width = 1 if shape == "scalar" else int(_re.fullmatch(r"\((\d+),?\)", shape).group(1))
        assert width == hi - lo, f"{f['name']}: shape {shape} but slice {f['slice']}"
        assert 0 <= lo < hi <= n, (
            f"{f['name']}: slice {f['slice']} outside the {n}-D observation")


@pytest.mark.parametrize("task_id", sorted(_sliced()))
def test_every_advertised_expression_evaluates(task_id):
    """A helper or symbol whose right-hand side does not evaluate is worse than absent:
    it is pasted into a generated reward and fails at training time, one wasted policy
    run per candidate, misattributed to the reward.

    `s` and `a` are sized from the SPEC, and both names are what they are because that is
    what the rendered prompt binds -- `EnvAdapter._render_api_stub` writes `s[i]` and
    `a[i]`, so an expression over `action` names nothing. Both halves catch a real
    defect shape: `room` helpers whose expression carries a free `k` (a parametric helper is the fabricated callable
    `kind: inline_expression` exists to forbid -- a model writes `object_pos(s, 2)` and
    no such method exists), and a `control_cost` written over `action`.
    """
    import numpy as np

    spec = _sliced()[task_id]
    n = _obs_dim(spec)
    env = {"np": np, "s": np.arange(n, dtype=float) * 0.1,
           "a": np.arange(int((spec.env["spaces"])["action_dim"]), dtype=float) * 0.01}

    for helper in spec.state_surface["helpers"]:
        assert helper["kind"] == "inline_expression", (
            f"{helper['name']}: no method of that name exists on the adapter, and "
            "advertising one lets a `self.x(s)` candidate pass verification against a "
            "proxy self and then die in training against self=None")
        if helper["expression"] is not None:
            eval(helper["expression"], dict(env))          # noqa: S307 - our own data

    for name, expression in (spec.symbol_mapping or {}).items():
        eval(expression, dict(env))                        # noqa: S307 - our own data


_PEG_HEAD = ("(s[4:7] + (lambda u, w, v: v + 2 * w * np.cross(u, v) + 2 * np.cross(u, np.cross(u, v)))"
             "(s[7:10], s[10], np.array([-0.13, 0.0, -0.01])))")

#: T2R's `distance_to_goal` per task. The default is the observed object's distance to the
#: goal; the exceptions are the tasks where that quantity cannot reach the shipped success
#: region at all: `reach` scores the reconstructed tool centre (hand body +
#: (0, 0, -0.045), `_success_reach`) against the goal, so its distance is gripper-to-goal, with
#: s[0:3] -- the hand body itself -- as the observation-space proxy; the puck in its scene is a
#: distractor whose only role at reset is a rejection keeping it 0.15 m from the goal; the two window
#: tasks are scored on x alone and the goal marker sits 9.5 cm off the handle's line of
#: travel, so a 3-D norm has a floor of ~0.10 m against a 0.05 m radius; `peg_insert_side`
#: scores the peg HEAD, 13 cm from the observed grasp point, so the observed point is still
#: 13 cm out at full insertion. A table that made them uniform would hand a candidate a
#: "distance to goal" that no policy can drive below the threshold.
_DISTANCE_TO_GOAL = {
    "reach": "np.linalg.norm(s[0:3] - s[36:39])",
    "reach_wall": "np.linalg.norm(s[0:3] - s[36:39])",  # MT50; same case as reach
    "window_open": "np.abs(s[4] - s[36])",
    "window_close": "np.abs(s[4] - s[36])",
    "peg_insert_side": f"np.linalg.norm({_PEG_HEAD} - s[36:39])",
}

#: Per-task symbols beyond the shared table. `handle_to_goal_x` is the signed one-axis error
#: the window checks use; `gap_to_target_height` is what the button check compares (<= 0.024)
#: and `press_depth` the depth pressed from the 0.0935 m unpressed gap (not the gap, which is
#: its inverse); `peg_head` is the scored site, 13 cm from the observed
#: grasp point.
_TASK_SYMBOLS = {
    "window_open": {"handle_to_goal_x": "(s[4] - s[36])"},
    "window_close": {"handle_to_goal_x": "(s[4] - s[36])"},
    "button_press_topdown": {"gap_to_target_height": "np.abs(s[38] - s[6])",
                             "press_depth": "np.clip(0.0935 - np.abs(s[38] - s[6]), 0.0, None)"},
    "peg_insert_side": {"peg_head": _PEG_HEAD},
}


@pytest.mark.parametrize("task_id", sorted(_mt10()))
def test_distance_to_goal_is_the_quantity_the_success_check_can_reach(task_id):
    """See `_DISTANCE_TO_GOAL`: shared across the fifty except where the shared expression
    is structurally unable to enter the success region -- on the ten, reach, the windows and
    the peg; `reach_wall` is `reach`'s case again, no object to move,
    so its distance is gripper-to-goal while `object_to_goal` still names the (distractor)
    object. The forty generated specs carry the generator's generic object-to-goal norm and
    have not had the ten's per-task review; a one-axis MT50 check (button-press, door-lock,
    the handle tasks) may want its own row here."""
    spec = _mt10()[task_id]
    rhs = spec.symbol_mapping["distance_to_goal"]
    assert rhs == _DISTANCE_TO_GOAL.get(task_id, "np.linalg.norm(s[4:7] - s[36:39])"), task_id
    if task_id not in ("reach", "reach_wall"):
        assert spec.symbol_mapping["object_to_goal"] == rhs, "object_to_goal is the same quantity"
    # the task-specific symbols are pinned so a regeneration
    # or a hand edit cannot drop them silently: the generic evaluator above only proves they run
    for name, expr in _TASK_SYMBOLS.get(task_id, {}).items():
        assert spec.symbol_mapping.get(name) == expr, f"{task_id}: symbol_mapping.{name}"


@pytest.mark.parametrize("task_id", sorted(_mt10()))
def test_the_shipped_success_check_names_a_real_transcription(task_id):
    """The success check is REFERENCED, never carried as an evaluable string -- run dirs on
    a shared mount may be world-writable, so a data file that could name arbitrary code
    would be an execution path into a training run. The name must resolve, and it must
    resolve through the module's closed allow-list rather than by import."""
    import importlib

    from bird.envs.metaworld import _SUCCESS_CHECKS

    ref = _mt10()[task_id].discrete_success["shipped"]["reimplementation"]
    assert ref["repo"] == "bird"
    assert ref["symbol"] in _SUCCESS_CHECKS, (
        f"{ref['symbol']} is not in _SUCCESS_CHECKS; a name from a data file is resolved "
        "against that dict and nowhere else")
    module = importlib.import_module(ref["module"])
    assert getattr(module, ref["symbol"], None) is _SUCCESS_CHECKS[ref["symbol"]]


def test_a_name_the_allow_list_does_not_hold_is_refused_by_the_RESOLVER():
    """The test above checks the DATA; this one checks the CODE PATH.

    `test_the_shipped_success_check_names_a_real_transcription` asserts that the ten
    shipped symbols are in `_SUCCESS_CHECKS` -- and it does that with `importlib` itself, so it
    is structurally incapable of noticing if production STOPPED going through the
    allow-list. Replace the two lines in `MetaWorld.__init__` and in `success_check_for` with
    `getattr(import_module(reimpl["module"]), symbol)` and every other test in this file
    still passes, while an arbitrary `module`/`symbol` pair out of a YAML file becomes an
    import-and-call. That is the property the docstring above actually claims, and a
    test that verifies the data rather than the mechanism cannot see it. The precedent
    it follows is `checkpoint._DECODABLE`: run directories on a shared mount may be
    world-writable, and `tasks/` is ordinary tracked YAML.

    Runs without the metaworld extra -- `success_check_for` resolves through the spec and the
    allow-list using numpy alone, and `MetaWorld.__init__` performs the same check before
    `import metaworld`.
    """
    import copy

    from bird import tasks
    from bird.envs import metaworld as mw

    task_id = mw._EXPECTED_MT10[0]
    real = mw.by_env_id(mw._env_id(task_id))
    forged = copy.deepcopy(dict(real.raw))
    forged["discrete_success"]["shipped"]["reimplementation"]["symbol"] = "os.system"
    stub = tasks.TaskSpec(id=real.id, raw=forged, path=real.path, sha256=real.sha256)

    original = mw.by_env_id
    mw.by_env_id = lambda _env_id: stub
    try:
        with pytest.raises(tasks.TaskSpecError, match="_SUCCESS_CHECKS allow-list"):
            mw.success_check_for(task_id)
    finally:
        mw.by_env_id = original


def test_the_success_checks_the_specs_name_are_exactly_the_ones_the_adapter_has():
    """A surjection, not a bijection: window-open and window-close share `_success_window`
    (one goal-distance test, two goals), so nine success checks back ten tasks, and on
    the MT50 forty twenty-one tasks share four `_success_within_*` radii. Asserting a
    bijection here would fail on a correct catalogue."""
    from bird.envs.metaworld import _SUCCESS_CHECKS

    named = {s.discrete_success["shipped"]["reimplementation"]["symbol"]
             for s in _mt10().values()}
    assert named == set(_SUCCESS_CHECKS)
    # 9 checks for the MT10 ten; 20 more for the MT50 forty (ten bespoke, four radii
    # shared by twenty-one tasks and six single-axis bands shared by nine, each shared
    # by every task with that literal).
    assert len(named) == 29 and len(_mt10()) == 50


@pytest.mark.parametrize("task_id", sorted(_mt10()))
def test_the_domain_randomisation_axes_are_the_adapters(task_id):
    """`rapp.parameters` inherits from here, and DrEureka's entire mechanism is those
    axis names. An axis the adapter does not know is dropped silently by `set_dr`, which
    is how a sweep measures nothing and logs `degenerate`."""
    from bird.envs.metaworld import MetaWorld

    dr = _mt10()[task_id].domain_randomization
    assert dr is not None, "Meta-World implements dr_probe; a null block would hide it"
    assert {k: tuple(v) for k, v in dr["parameters"].items()} == \
        {k: tuple(v) for k, v in MetaWorld.dr_parameters.items()}
    assert dr["nominal"] == MetaWorld._dr_nominal
    for axis, (lo, hi) in dr["parameters"].items():
        assert lo < dr["nominal"][axis] < hi, (
            f"{axis}: nominal {dr['nominal'][axis]} is not inside [{lo}, {hi}]")


@pytest.mark.parametrize("task_id", sorted(_mt10()))
def test_the_two_prose_fields_are_at_different_granularities(task_id):
    """`env_prose` describes the ENVIRONMENT; `l_task` is the INSTRUCTION. The core
    schema overloads one field for both -- at environment granularity on the generated specs and
    at instruction granularity on the hand-written door_open -- so a consumer reading
    `natural_language` cannot tell which it is holding. Splitting them is only worth
    anything if they stay distinct."""
    d = _mt10()[task_id].description
    env_prose, l_task = str(d["env_prose"]).strip(), str(d["l_task"]).strip()
    assert env_prose and l_task
    assert env_prose != l_task
    assert len(l_task) > len(env_prose), (
        "the instruction states the task AND how the arm is commanded; the environment "
        "paragraph is a docstring")


@pytest.mark.parametrize("task_id", sorted(_mt10()))
def test_the_instruction_says_how_the_arm_is_commanded(task_id):
    """The 4-D action is a mocap DELTA, not a joint torque. A model that assumes torques
    writes a reward for a robot that is not in the room."""
    l_task = str(_mt10()[task_id].description["l_task"])
    assert "Cartesian displacement of the gripper" in l_task


@pytest.mark.parametrize("task_id", sorted(_mt10()))
def test_the_instruction_never_quotes_the_success_tolerance(task_id):
    """The goal displacement is the TASK and is stated; the tolerance is the METRIC and
    is not. `generate.context.strip_existing_reward` withholds `evaluate_state` from the
    environment text, and writing its threshold into the instruction would hand the same
    number over through the other door."""
    spec = _mt10()[task_id]
    l_task = str(spec.description["l_task"])
    criterion = str(spec.description.get("success_criterion_prose") or "")
    numbers = set(_re_numbers(criterion)) - set(_re_numbers(str(spec.description["env_prose"])))
    leaked = sorted(n for n in numbers if n in l_task)
    assert not leaked, (
        f"{task_id}: the instruction quotes {leaked}, which appears in the success "
        "criterion and nowhere in the environment description")


def _re_numbers(text):
    import re as _re

    # tolerances are written as decimals; bare integers are horizons and axis counts
    return [m.group(0) for m in _re.finditer(r"\d+\.\d+", text or "")]


# --------------------------------------------------------------------------
# the task definition as a frozen run input
# --------------------------------------------------------------------------
#
# `Config.hash()` covers the resolved config dict and nothing else, so the SPEC's content
# is not in the run id. Two runs whose spec differs would otherwise share a directory name
# and be adoptable as each other -- a silently wrong resume, which is worse than a crash.
# The spec is therefore copied into the run dir and checked on adopt, exactly as
# `config.resolved.yaml` is.


def _entry_module():
    import importlib.util

    from conftest import REPO

    spec = importlib.util.spec_from_file_location("bird_entry_task", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_spec_content_is_not_in_the_config_hash():
    """Stated as a test because it is the premise of everything below. If this ever
    becomes false the guard is redundant -- but silently keeping both would be worse."""
    from bird.config import load

    cfg = load("zeroshot", profile="dev",
               overrides={"problem.env_id": "mt10_window-open-v3"})
    before = cfg.hash()
    assert before == load("zeroshot", profile="dev",
                          overrides={"problem.env_id": "mt10_window-open-v3"}).hash()
    assert "window_open" not in cfg.to_yaml(), (
        "the config names an env id, not a task spec; if that changed, revisit whether "
        "the adopt guard below is still the mechanism")


def test_adopting_across_a_changed_task_spec_is_refused(tmp_path):
    from bird import checkpoint
    from bird.config import load

    mod = _entry_module()
    cfg = load("zeroshot", profile="dev",
               overrides={"problem.env_id": "mt10_window-open-v3"})
    target = tmp_path / "run"
    target.mkdir()
    (target / "config.resolved.yaml").write_text(cfg.to_yaml())

    spec = mod._task_spec_for(cfg)
    assert spec is not None, "zeroshot on mt10 must have a spec for this to test anything"

    # the honest case: the frozen copy is the current one
    (target / "task_spec.yaml").write_text(spec.path.read_text())
    mod._assert_same_config(target, cfg)

    # the case that matters: someone edited the catalogue between the legs
    (target / "task_spec.yaml").write_text(spec.path.read_text() + "\n# edited\n")
    with pytest.raises(checkpoint.CheckpointError, match="task definition"):
        mod._assert_same_config(target, cfg)


def test_adopting_a_run_dir_that_predates_the_catalogue_is_allowed(tmp_path):
    """A MISSING `task_spec.yaml` is not evidence of a changed task.

    It means the directory was written by a version that did not freeze the spec, and
    the config hash -- the primary identity -- still matched. Refusing there would strand
    every such run dir to guard against nothing. A copy that exists and DIFFERS is the case worth refusing, and the test
    above covers it.
    """
    from bird.config import load

    mod = _entry_module()
    cfg = load("zeroshot", profile="dev",
               overrides={"problem.env_id": "mt10_window-open-v3"})
    assert mod._task_spec_for(cfg) is not None
    target = tmp_path / "run"
    target.mkdir()
    (target / "config.resolved.yaml").write_text(cfg.to_yaml())
    mod._assert_same_config(target, cfg)          # no task_spec.yaml, and that is fine


def test_the_leak_gate_fires_on_an_authored_list_and_not_an_inherited_one(tmp_path):
    """`verify.forbidden_symbols` is inherit-by-default, and the gate is about a PIN
    NOTHING HONOURS -- so what matters is whether the CONFIG wrote the list, not whether
    one is present.

    The case: `-s problem.env_id=mt10_reach-v3` on a tester config. The tester profile
    turns verification off for speed, and the environment carries a leak gate, so the
    config acquires a non-empty list it never authored. Nothing is fabricated there, and
    reporting it as such
    trains people to ignore the one check `_check_coherence` itself calls dangerous
    rather than merely untidy.
    """
    from bird.config import ConfigError, load

    # inherited, verification off -> quiet
    cfg = load("eureka", profile="tester",
               overrides={"problem.env_id": "mt10_reach-v3"})
    assert cfg["verify.enabled"] is False
    assert cfg["verify.forbidden_symbols"], "the env must supply a non-empty list here"

    # authored, and nothing checks it -> still fatal
    p = tmp_path / "authored.yaml"
    p.write_text("name: authored\nverify:\n  enabled: false\n"
                 "  forbidden_symbols: [my_own_secret]\n")
    with pytest.raises(ConfigError, match="forbidden_symbols is set but"):
        load(p)

    # authored, verification on, but the check is not in the list -> still fatal
    p = tmp_path / "unwired.yaml"
    p.write_text("name: unwired\nverify:\n  enabled: true\n"
                 "  static_checks: [ast_syntax]\n  forbidden_symbols: [my_own_secret]\n")
    with pytest.raises(ConfigError, match="not in\\s+verify.static_checks"):
        load(p)


def test_the_leak_gate_covers_the_inherited_path_too(tmp_path):
    """Three arms, and authorship decides two of them.

    Scoping the whole check to authored lists removes a false positive on a tester config
    pointed at a Meta-World env, and is wrong in a way that is worse than that false
    positive. Inheriting is the DEFAULT, so both arms would stop covering the common path,
    and a check that fires only where nobody goes is deleted without anyone deciding to
    delete it.

      verify off  + AUTHORED  -> a pin nothing honours. Refuse.
      verify off  + inherited -> the method opted out of verification; the environment's
                                 gate does not apply. Coherent.
      verify on   + AUTHORED  -> incoherent, and there is a file to fix it in. Refuse.
      verify on   + inherited -> WIRED, not refused.

    WHY THE LAST ARM WIRES RATHER THAN REFUSES. The argument for refusing is that a
    config's §2 belongs to its method and the harness must not edit it. But the inherited
    path has NO FILE to be fixed in: the env is `-s problem.env_id=...`, so refusing would
    leave every method with `verify.enabled: true` -- `card`, `rda`, `limen` and
    `limen_reward_only` among them -- unable to resolve on the Meta-World tier AT ALL,
    and the only remedy would be
    `-s verify.static_checks=[...]` hand-carried onto every command line. `eureka` is the
    exception and not the counter-example: it passes only because `verify.enabled:
    false`, which is why spot-checking one config does not show it.

    And the premise is wrong on inspection. `verify.forbidden_symbols` is not one of the
    method's checks -- it is the HARNESS's anti-leakage gate, a property of the
    environment (what a candidate could reach to touch the metric it is scored on), which
    is exactly why an environment change may legitimately move it. Its reader is
    environment-driven for the same reason. Wiring it is installing the harness's own
    guard, not editing the method's §2 -- the same thing a per-environment overlay that
    pinned the two keys together would do.

    WHAT KEEPS THE CHECK ALIVE, since losing an arm is a real failure mode and an
    auto-wire could reintroduce it by making the whole gate unreachable: the AUTHORED
    arms are untouched. `_inherit_from_task_spec` only wires inside the branch that fires
    when `verify.forbidden_symbols` was null, so an author who writes a gate and forgets
    its reader still gets a validation error -- asserted below, and again in the
    `unwired.yaml` case above.
    """
    from bird.config import ConfigError, load

    inherited = load("eureka", profile="tester",
                     overrides={"problem.env_id": "mt10_reach-v3"})
    assert inherited["verify.enabled"] is False
    assert inherited["verify.forbidden_symbols"], "the env must supply a list here"
    assert "forbidden_symbols" not in inherited["verify.static_checks"], (
        "verification is off here, so there is nothing to wire and wiring anyway "
        "would put a check in a list no stage reads")

    authored = tmp_path / "authored.yaml"
    authored.write_text("name: authored\nverify:\n  enabled: false\n"
                        "  forbidden_symbols: [my_own_secret]\n")
    with pytest.raises(ConfigError, match="verify.enabled=false"):
        load(authored)

    # verify ON + AUTHORED: still refused, and this is the arm that keeps the
    # coherence check reachable with the inherited path wired.
    unwired_authored = tmp_path / "unwired_authored.yaml"
    unwired_authored.write_text(
        "name: unwired_authored\nverify:\n  enabled: true\n"
        "  static_checks: [ast_syntax]\n  forbidden_symbols: [my_own_secret]\n")
    with pytest.raises(ConfigError, match="in force but"):
        load(unwired_authored)

    # verify ON + inherited: WIRED. The gate came from the environment, so its
    # reader does too.
    wired = tmp_path / "wired.yaml"
    wired.write_text("name: wired\nproblem:\n  env_id: mt10_reach-v3\n"
                     "verify:\n  enabled: true\n  static_checks: [ast_syntax]\n")
    cfg = load(wired)
    assert cfg["verify.forbidden_symbols"], "precondition: the env supplied a gate"
    assert cfg["verify.static_checks"] == ["ast_syntax", "forbidden_symbols"], (
        "the spec-supplied gate did not bring its reader; this config would be "
        "refused outright, which makes the Meta-World tier unreachable")


def test_an_empty_consumer_gate_is_not_an_absent_one():
    """`forbidden_symbols_by_consumer: {bird: []}` means "no gate needed here".

    Resolved with `or`, that deliberate empty list is falsy and silently becomes the
    BENCHMARK's list -- a config would then reject symbols its task said were fine. The
    repo already treats this distinction as load-bearing: `singh_orp` pins
    `verify.forbidden_symbols: []` on purpose, and `artifacts.py` keeps `failure: ""`
    distinct from a reason.
    """
    from bird.config import forbidden_symbols_of

    class _Spec:
        def __init__(self, reward):
            self.reward = reward

    upstream = ["evaluate_state", "hardened_success"]
    assert forbidden_symbols_of(_Spec({"forbidden_symbols": upstream})) == upstream
    assert forbidden_symbols_of(_Spec({
        "forbidden_symbols": upstream,
        "forbidden_symbols_by_consumer": {"bird": ["task_metric"]}})) == ["task_metric"]
    assert forbidden_symbols_of(_Spec({
        "forbidden_symbols": upstream,
        "forbidden_symbols_by_consumer": {"bird": []}})) == [], (
        "an explicit empty gate fell through to the benchmark's list")


@pytest.mark.parametrize("bad", ["\x00", "a" * 5000, "../etc/passwd", "/etc"])
def test_every_unusable_task_id_raises_the_named_error(bad):
    """`_inherit_from_task_spec` catches TaskSpecError and nothing else, so anything that
    escapes it surfaces as a raw traceback from inside `load()` rather than as a config
    problem with the offending key named. A NUL byte made `pathlib` raise ValueError."""
    from bird import tasks

    with pytest.raises(tasks.TaskSpecError):
        tasks.load(bad)


# --------------------------------------------------------------------------
# the spec is the ONLY home, including for `full_source`
# --------------------------------------------------------------------------

_SPEC_OWNED = ("_prose", "_state_fields", "_action_fields", "_helpers")

_NATIVE = ("pendulum", "pendulum_discrete", "acrobot",
           "toy_reacher", "toy_gridworld", "toy_hungry_thirsty")


def _registered_env_ids():
    """Every registered env NAME. Names only -- this runs at collection.

    Wrapped in `env_param` so the HumanoidBench ids carry the `humanoid` marker
    and are DESELECTED under `-m "not humanoid"` rather than skipped in the body
    below -- see the comment on that skip.
    """
    from bird import registry
    from conftest import env_param

    registry.load_all()          # imports the env modules; constructs nothing
    return [env_param(n) for n in sorted(registry.names("env"))]


@pytest.mark.parametrize("env_id", _registered_env_ids())
def test_no_adapter_ships_a_declaration_the_spec_overwrites(env_id):
    """Dead as DATA is not dead as TEXT.

    A class-body declaration that `_apply_spec` overwrites is still SHIPPED TO THE MODEL
    under `generate.context.env_spec: full_source` -- which Eureka and DrEureka use --
    while the other three renderings show the spec. Two answers to one question, in one
    prompt, with nothing to say which is current, and it bites exactly when the spec
    moves: the catalogue is the task's single home (`tasks/SOURCE.md`), so the spec is the
    file expected to change.

    The symptom: fork a spec's horizon and three renderings follow it while
    `full_source` does not.

    THIS RENDERS THE STRING AND LOOKS IN IT, rather than asking which classes inherit
    `EnvAdapter._render_full_source`. An inheritance test is a PROXY for the claim and it
    is wrong: `PendulumDiscrete` overrides the renderer to concatenate `Pendulum`'s
    source with its own, so it ships MORE of its class body than the classes the proxy
    admits, and the proxy excludes it (27,016 characters with all three declarations
    inside, measured). The rendered text is the thing the model receives, so the
    rendered text is what gets asserted on.

    Scoped to `SpecEnvAdapter`, which is the actual condition rather than a narrowing:
    the defect is "declares what THE SPEC supplies", so a class no spec feeds has nothing
    to be overwritten by. `GymMujoco` is a plain `EnvAdapter` that legitimately owns
    its per-family observation tables, and flagging it would demand deleting the only
    copy there is.
    """
    from bird import registry
    from bird.envs.spec import SpecEnvAdapter

    registry.load_all()
    try:
        adapter = registry.get("env", env_id)(None)
    except _simulator_missing() as exc:
        # An uninstalled simulator drops THIS case, never collection -- see
        # `_registered_env_ids`. A job that installs only `--extra test` fires
        # this for every simulator-backed adapter.
        #
        # Under `--extra all` the HumanoidBench ids do not reach it:
        # HumanoidBench cannot share an interpreter with metaworld, so
        # `--extra all` cannot install it, and the answer is the `humanoid`
        # marker (deselect).
        pytest.skip(f"{env_id} needs a simulator this install lacks: {exc}")

    cls = type(adapter)
    owned = _SPEC_OWNED if isinstance(adapter, SpecEnvAdapter) else ()
    source = adapter._render_full_source()
    shipped = [attr for attr in owned
               if attr in vars(cls) and f"{attr} =" in source]
    assert not shipped, (
        f"{cls.__name__} declares {shipped} in its class body AND `_render_full_source` "
        f"puts them in the {len(source)}-char string the model reads, while `_apply_spec` "
        "overwrites them at runtime -- so the prompt carries the dead copy")


@pytest.mark.parametrize("env_id", _NATIVE)
def test_the_shape_the_class_builds_from_matches_the_spec(env_id):
    """`horizon`, `obs_dim` and `action_dim` legitimately live in both places -- the class
    builds its action set and bounds from them, the spec renders the API stub from them --
    so the contract is that they AGREE. `_apply_spec` asserts rather than preferring one,
    because silently preferring the spec lets a spec edit reach three renderings and
    not the fourth."""
    from bird import registry, tasks

    registry.load_all()
    adapter = registry.get("env", env_id)(None)
    spec = tasks.by_env_id(env_id)
    spaces = spec.env["spaces"]
    assert adapter.horizon == int(spec.env["horizon"])
    assert adapter.obs_dim == int(spaces["obs_dim"])
    assert adapter.action_dim == int(spaces["action_dim"])


def test_the_shape_guard_reads_the_instance_when_the_class_declares_none():
    """`Assistax` holds the ROW's horizon and obs_dim on the INSTANCE and `None` in the
    class body -- the base class's `horizon = 1` / `obs_dim = 0` would otherwise collide --
    and assigns them two lines before calling `_apply_spec`. A guard comparing the spec
    against `getattr(type(self), attr)` would, on such an adapter, compare against `None`,
    skip, and let `setattr` replace the row's value with the spec's: the silent preference
    the comment above it names as the bug, on exactly the adapters whose shapes vary per
    row (a spec copy with `env.horizon: 999` against a row of 300 would raise nothing and
    leave the instance at 999). Instance first, which falls back to the class through the
    MRO, so `Pendulum`, `MetaWorld` and every class-declared adapter see the same check; a
    class that declares `None` and sets nothing still adopts the spec's value. No
    committed spec disagrees with its row.
    """
    import dataclasses

    from bird import tasks
    from bird.envs.spec import SpecEnvAdapter

    spec = tasks.by_env_id("assistax_feeding")
    horizon, spaces = int(spec.env["horizon"]), spec.env["spaces"]
    drifted = dataclasses.replace(
        spec, raw={**spec.raw, "env": {**spec.env, "horizon": horizon + 1}})

    class RowShaped(SpecEnvAdapter):
        obs_dim = None
        horizon = None
        action_dim = int(spaces["action_dim"])

    def row_shaped(row_horizon):
        env = object.__new__(RowShaped)
        env.obs_dim, env.horizon = int(spaces["obs_dim"]), row_horizon
        return env

    agreeing = row_shaped(horizon)
    agreeing._apply_spec(spec)
    assert agreeing.horizon == horizon

    with pytest.raises(tasks.TaskSpecError, match=r"horizon"):
        row_shaped(horizon)._apply_spec(drifted)

    adopting = object.__new__(RowShaped)        # class None, instance unset: the spec supplies it
    adopting._apply_spec(drifted)
    assert adopting.horizon == horizon + 1 and adopting.obs_dim == int(spaces["obs_dim"])

    class ClassShaped(SpecEnvAdapter):
        obs_dim, action_dim = int(spaces["obs_dim"]), int(spaces["action_dim"])

    ClassShaped.horizon = horizon
    with pytest.raises(tasks.TaskSpecError, match=r"horizon"):
        object.__new__(ClassShaped)._apply_spec(drifted)


def test_a_flat_anchor_is_refused_rather_than_used():
    """A spec with no `by_reduction` never normalises anything, whether or not it carries
    a number -- because an anchor that cannot say WHICH reduction it measured cannot
    normalise (0.239 per-step vs 0.083 any-step is one policy on one task).

    THE REFUSAL IS ASSERTED OVER ALL OF THEM -- the loop below runs over `flat`, both
    populations, because the claim is true of both. What the split is for is everything
    else in this test, and the two populations differ in what they can go wrong as:

      MEASURED   the core-shaped Gymnasium MuJoCo specs, which carry a real `random`
                 (half_cheetah -4.58) and no reduction label. From inside `baselines_of`
                 those look overlooked, and reading them is the fallback this test
                 forbids -- so this is the only population that can TEMPT one, and the
                 synthetic-donor half at the bottom is built on it.
      UNMEASURED the HumanoidBench specs, where both ends are null with a reason. They
                 tempt nothing, but they take a DIFFERENT branch of `baselines_of`:
                 `anchors.get("random")` is the anchor DICT, which is never None, so they
                 reach the return-None path rather than the raise below it. If that
                 dict-vs-value distinction is ever "tidied" into a truthiness check, these
                 begin raising `TaskSpecError` at env construction and the whole tier
                 stops loading. That is what the extra assertion on this population is
                 watching, alongside the reason each null must carry.

    Both sets are asserted non-empty: a filter that quietly matched nothing would leave
    this passing having checked neither.
    """
    from bird import tasks
    from bird.envs.spec import baselines_of
    from bird.tasks import ANY_STEP, PER_STEP

    import dataclasses

    flat = [s for s in tasks.index().values() if not (s.anchors or {}).get("by_reduction")]
    assert flat, "no flat-anchor specs left -- this guard is now vacuous, delete or re-aim it"

    measured = [s for s in flat if (s.anchors.get("random") or {}).get("value") is not None]
    unmeasured = [s for s in flat if (s.anchors.get("random") or {}).get("value") is None]
    assert measured, (
        "no flat-anchor spec carries a measured random any more, so the tempting case "
        "this test is built on is gone -- re-aim it rather than deleting the loop")
    assert unmeasured, (
        "no flat-anchor spec has a null random any more. The HumanoidBench specs were "
        "that case; if they gained anchors, good -- delete this half rather than leaving "
        "it matching nothing")

    for spec in flat:
        for reduction in (PER_STEP, ANY_STEP):
            assert baselines_of(spec, reduction) is None, spec.id

    for spec in unmeasured:
        assert (spec.anchors["random"].get("reason") or "").strip(), (
            f"{spec.id}: a null anchor with no reason -- the absence-is-explicit rule")

    # THE LOOP ABOVE DISCRIMINATES ONLY WHERE A COMMITTED SPEC HAS BOTH ENDS. On a spec
    # whose `expert.value` is null the flat fallback this test forbids would return None
    # too, so the assertion there never discriminates -- a guard that cannot fail in the
    # direction it was written for, in a test written to prevent exactly that.
    #
    # So: a synthetic flat spec with BOTH ends present, independent of which committed
    # specs happen to carry an expert. That is the only shape under which the refusal and
    # the fallback give different answers, which makes it the shape this test is built on.
    donor = tasks.load("half_cheetah")
    both_ends = dataclasses.replace(donor, raw={
        **donor.raw,
        "anchors": {"random": {"value": 0.0833, "method": "random_policy"},
                    "expert": {"value": 0.95, "method": "synthetic_for_this_test"}}})
    for reduction in (PER_STEP, ANY_STEP):
        assert baselines_of(both_ends, reduction) is None, (
            "a flat anchor pair with both ends measured was used as a baseline. It still "
            "carries no reduction label, and 0.239 per-step against 0.083 any-step on "
            "drawer-close is one policy on one task")


@pytest.mark.parametrize("task_id", sorted(_mt10()))
def test_the_required_flat_pair_duplicates_the_any_step_pair(task_id):
    """Every nested spec ALSO carries the schema's required top-level `random`/`expert`,
    and that duplicate is an exact copy of the ANY-STEP pair -- never per-step, never a
    third number. On `door_open` that is 0.9667 against a per-step 0.0598.

    Latent: `baselines_of` is the only reader of `.anchors` in the tree and it routes
    through `by_reduction`. But a naive `spec.anchors["expert"]` returns the any-step figure
    whatever reduction was asked, so if a re-measurement ever updates one and not the other
    they diverge silently and the next reader gets a stale number with no tell. Pinning it
    makes the duplication an asserted invariant rather than a coincidence.
    """
    from bird.tasks import ANY_STEP

    anchors = _mt10()[task_id].anchors
    any_step = anchors["by_reduction"][ANY_STEP]
    for role in ("random", "expert"):
        assert anchors[role]["value"] == any_step[role]["value"], (
            f"{task_id}: the top-level {role} anchor has diverged from the any-step pair "
            "it duplicates")


# --------------------------------------------------------------------------
# env.reset -- what one seeded reset draws, held against the adapter (SCHEMA_DELTA #14)
# --------------------------------------------------------------------------
#
# Optional in the schema so a core-shaped spec stays valid; REQUIRED here on every
# spec in the catalogue (test A), the `_no_adapter.json` partition precedent. The block
# describes code, so the tests below read the code: B is the block's own arithmetic, C
# resolves `entry` and every `source` to the file and span they name, D holds the prompt
# quote to the field it cites, and F -- the only one that can catch a block that is internally consistent and
# wrong about the environment -- constructs the adapter and measures what moves.

#: The hooks `EnvAdapter.reset` reaches. A class that defines one of these ITSELF is the
#: class whose hook `entry.symbol` must name -- `_H1HandBase._reset` on a task that
#: overrides `_task_reset` is a block pasted from a sibling, and reads fine.
_RESET_HOOKS = ("_reset", "_task_reset", "_task_reset_state")

#: Specs whose `state_surface` CANNOT map a NAME to observation columns: no `flat_fields`
#: and no `fields[].slice` (the core shape, which the ten gym specs keep), over an
#: adapter whose state is raw `[qpos, qvel]` -- 18 columns on HalfCheetah against the
#: spec's gymnasium-shaped `obs_dim` of 17. For these a name token is checked by B and
#: never measured; an `a:b` slice token is a slice of the ADAPTER's flat state, so B
#: cannot bound it by the spec's `obs_dim` and F bounds it by the width `reset()`
#: actually returned -- and where every token is a slice, F measures the block in full.
#: A literal EQUALITY, pinned by `test_the_reset_name_check_set_is_exactly_the_
#: unmappable_specs`, so the set cannot grow silently: a new spec with no slices lands
#: HERE, by hand, or fails there.
RESET_NAME_CHECK_ONLY = frozenset({
    "half_cheetah", "half_cheetah_backward", "half_cheetah_target_speed",
    "hopper_hop", "hopper_hop_in_place", "inverted_pendulum_balance",
    "reacher_hold", "reacher_reach", "swimmer_forward", "swimmer_heading",
    # `humanoid_run` joins by hand, as this comment requires, and keeps
    # the family's shape rather than becoming the one gym spec with a different one:
    # `state_surface.fields` names `s.qpos`/`s.qvel` groups with no `slice`, the
    # core shape that the module docstring records the whole gym set as keeping
    # ("inertia now, not a rule") and that `tasks/SCHEMA_DELTA.md` owns changing.
    # Worth stating what is DIFFERENT here: unlike its ten siblings this spec's
    # `obs_dim` (47) EQUALS the adapter's flat state width, because Humanoid's
    # gymnasium observation is not what the adapter emits and the spec declares the
    # adapter's. So slices here would be soundly bounded and this spec could leave the
    # set -- it is the only member of which that is true. It stays because the
    # coverage leaving would buy is small: every one of this spec's `draws[].fields`
    # tokens is already an `a:b` slice, which F measures in full either way.
    "humanoid_run",
})

#: Adapter module -> the package `importorskip` asks for before constructing one of its
#: envs in F. Natives and toys are absent on purpose: they need nothing, and run in the
#: `--extra test` job. `humanoid` ids also carry their deselect marker via
#: `conftest.env_param`, so `-m "not humanoid"` deselects them rather than skipping.
#: `bird.envs.upstream_assistax` has no row: its factory raises `ImportError` on a machine
#: without the git-only `assistax` package, which `_simulator_missing()` turns into the
#: per-case skip below.
_SIM_PACKAGES = {
    "bird.envs.metaworld": "metaworld",
    "bird.envs.gym_mujoco": "gymnasium",
    "bird.envs.assistax": "mujoco",
    "bird.envs.humanoid_hand": "humanoid_bench",
    # The jax tier's offline env: CPU `jax` is the whole
    # requirement, and the id carries the `jax` deselect marker via `conftest.env_param`.
    "bird.envs.jax_toy": "jax",
}


def _simulator_missing() -> tuple:
    """The exceptions that mean "this install lacks the simulator" when an adapter is
    constructed, for the per-case skips below. `ImportError` is the obvious one, and it is
    not enough: gymnasium raises its own `gymnasium.error.DependencyNotInstalled` when
    mujoco is absent, and that class is NOT an ImportError subclass -- so a venv with
    gymnasium but no mujoco would FAIL the ten gym cases instead of skipping them. Looked
    up lazily: gymnasium itself is optional here."""
    kinds = [ImportError]
    try:
        from gymnasium.error import DependencyNotInstalled
    except ImportError:
        pass
    else:
        kinds.append(DependencyNotInstalled)
    return tuple(kinds)

#: How many seeded resets F stacks. The floor on a branch it can see: a field that moves
#: only in a branch of probability q is still at 24 resets with probability (1-q)^24,
#: which is 2% at q = 0.15 (and 0.02% at q = 0.3). A rarer branch than that needs this
#: raised, not the block loosened.
_RESET_SAMPLES = 24


def _reset_block(doc: dict):
    return (doc.get("env") or {}).get("reset")


def _roles(entries) -> set:
    return {e["role"] for e in entries or ()}


def _field_names(doc: dict) -> set:
    ss = doc["state_surface"]
    return ({f["name"] for f in ss.get("flat_fields") or ()}
            | {f["name"] for f in ss.get("fields") or ()})


def _is_mappable(doc: dict) -> bool:
    """Can a `fields` token be turned into observation columns for this spec?"""
    ss = doc["state_surface"]
    return bool(ss.get("flat_fields")) or any(f.get("slice") for f in ss.get("fields") or ())


def _columns_of(doc: dict, token: str) -> frozenset:
    """One `fields` token -> the observation columns it names.

    `a:b` is a slice; a `flat_fields[].name` is its index; a `fields[].name` with a
    `slice` is that slice (a group name stands for its slice). Raises on a token that
    resolves to nothing -- B has already checked every token resolves BY NAME, so here
    a miss means the spec is unmappable and belongs in `RESET_NAME_CHECK_ONLY`.
    """
    import re as _re

    if _re.fullmatch(r"\d+:\d+", token):
        lo, hi = (int(x) for x in token.split(":"))
        return frozenset(range(lo, hi))
    ss = doc["state_surface"]
    for f in ss.get("flat_fields") or ():
        if f["name"] == token:
            return frozenset({int(f["index"])})
    for f in ss.get("fields") or ():
        if f["name"] == token and f.get("slice"):
            lo, hi = (int(x) for x in f["slice"].split(":"))
            return frozenset(range(lo, hi))
    raise KeyError(token)


def _parse_lines(text) -> tuple:
    """`source_ref.lines`: '312' or '304-313' -> (lo, hi) inclusive."""
    import re as _re

    m = _re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", str(text))
    assert m, f"lines {text!r} is not `N` or `N-M`"
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    assert lo <= hi, f"lines {text!r} run backwards"
    return lo, hi


def _ast_span(path: Path, symbol: str) -> tuple:
    """The (first, last) line of the dotted `symbol` in `path`, by `ast`.

    `Pendulum._reset` is the `_reset` def inside the `Pendulum` class body;
    `_rs_scratchitch` (`bird/envs/assistax.py`) is a module-level def. Raises `LookupError` naming what was not found
    -- a symbol that does not resolve is the pasted-block signature this exists for.
    """
    import ast

    tree = ast.parse(path.read_text())
    scope = tree.body
    node = None
    for part in symbol.split("."):
        node = next((n for n in scope
                     if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == part), None)
        if node is None:
            raise LookupError(f"{path.relative_to(ROOT)} has no `{part}` (of `{symbol}`)")
        scope = getattr(node, "body", [])
    return int(node.lineno), int(node.end_lineno)


def _registered_class(factory) -> type:
    """The adapter class a registered `env` factory constructs, WITHOUT constructing it.

    Most factories here -- the natives' `def pendulum(ctx): return Pendulum()`, and
    closures `make(ctx)` returned by a family's factory builder -- end in
    `return <ClassName>(...)`, and `<ClassName>` resolves in the factory's own globals.
    Read off the source with `ast` rather than off a return annotation, because the
    annotations name base classes (`-> _H1HandBase`, `-> SpecEnvAdapter`) and the whole
    question is WHICH subclass.

    A FACTORY MAY BE THE CLASS ITSELF. `@register` takes any callable, so a family may
    decorate its adapter classes directly rather than writing a `def` that returns one.
    That is a registration shape, not a defect; without this branch it would fail with
    "expected one `return Class(...)`, found []", which reads as a defect in the adapter
    and is a gap in the reader. Answering `factory` itself is not a special case being
    tolerated: for a class factory it is the exact, unambiguous answer the `ast` walk is
    trying to reconstruct for everything else.
    """
    import ast
    import inspect
    import textwrap

    if isinstance(factory, type):
        return factory
    # A factory may DECLARE its class, and a declaration beats a reconstruction.
    # `upstream_assistax.py` builds one closure per task (`_make_factory(cls)`)
    # and returns `cls(ctx)`, so the walk below finds the name `cls` and looks
    # it up in `__globals__`, where a closure variable is not -- KeyError. The
    # same shape would hit any family that registers per-task closures rather
    # than one `def` per task. `adapter_cls` is set beside `requires_cuda` and
    # `batched` on the registered object for exactly this reason: it is the
    # unambiguous answer, and the walk is only ever trying to recover it.
    declared = getattr(factory, "adapter_cls", None)
    if isinstance(declared, type):
        return declared
    tree = ast.parse(textwrap.dedent(inspect.getsource(factory)))
    names = {n.value.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name)}
    assert len(names) == 1, (
        f"{factory.__module__}.{factory.__name__}: expected one `return Class(...)`, "
        f"found {sorted(names)}")
    cls = factory.__globals__[names.pop()]
    assert isinstance(cls, type), cls
    return cls


def _spec_backed_env_ids():
    """Every registered env that a spec backs, MARKED per simulator (`conftest.env_param`)."""
    from bird import registry, tasks
    from conftest import env_param

    registry.load_all()
    return [env_param(n) for n in sorted(registry.names("env")) if tasks.by_env_id(n)]


def test_every_spec_declares_its_reset(specs):
    """A: the schema keeps `env.reset` optional; this catalogue does not. Silent failure:
    an unauthored task renders "not recorded", one em dash away from "nothing varies"."""
    have = {tid for tid, doc in specs.items() if _reset_block(doc) is not None}
    assert have == set(specs), (
        f"specs with no env.reset: {sorted(set(specs) - have)}. Author it from the "
        "adapter's `_reset` (the block's `entry`), never from the prose -- see "
        "tasks/README.md `Adding a task`")


def test_the_reset_roles_in_tasks_py_are_the_schemas_enum():
    """`bird/tasks.py::RESET_ROLES` restates `$defs.reset_role` for rendering; two copies
    of one closed vocabulary must be EQUAL, and every role must carry a label, or a role
    the schema admits renders as its raw identifier."""
    from bird.tasks import RESET_ROLE_LABELS, RESET_ROLES

    schema = json.loads((TASKS / "shared_spec.schema.json").read_text())
    assert list(RESET_ROLES) == list(schema["$defs"]["reset_role"]["enum"])
    assert set(RESET_ROLE_LABELS) == set(RESET_ROLES)
    assert all(str(v).strip() for v in RESET_ROLE_LABELS.values())


@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_a_reset_names_fields_this_spec_has(path):
    """B: the block's own arithmetic, in pure Python. Silent failure: a `fields` entry
    that names nothing, a box of the wrong width, a probability vector summing to 0.9 --
    each renders as a plausible table row and F cannot reach them on a spec whose
    simulator is not installed."""
    from bird.tasks import RESET_ROLES

    doc = _load(path)
    tid = path.parent.name
    block = _reset_block(doc)
    if block is None:
        pytest.fail(f"{tid}: no env.reset (test A names every one missing)")

    names = _field_names(doc)
    n = int(doc["env"]["spaces"]["obs_dim"])

    # On a name-check-only spec a slice indexes the ADAPTER's flat state, whose width the
    # spec does not state (see RESET_NAME_CHECK_ONLY); F bounds it by the measured width.
    bound = None if tid in RESET_NAME_CHECK_ONLY else n

    def _check_tokens(where, tokens):
        for tok in tokens:
            if ":" in tok:
                lo, hi = (int(x) for x in tok.split(":"))
                assert 0 <= lo < hi and (bound is None or hi <= bound), (
                    f"{tid}: {where} slice {tok} outside the {n}-D observation")
            else:
                assert tok in names, (
                    f"{tid}: {where} names `{tok}`, which is neither a "
                    "state_surface.flat_fields[].name nor a fields[].name")

    draws, constants = block["draws"], block["constants"]
    for e in list(draws) + list(constants):
        assert e["role"] in RESET_ROLES, f"{tid}: role {e['role']!r} is not in RESET_ROLES"

    quantities = [d["quantity"] for d in draws]
    assert len(quantities) == len(set(quantities)), (
        f"{tid}: draws[].quantity is not unique: {quantities} -- `derived_from` refers to these")
    cq = [c["quantity"] for c in constants]
    assert len(cq) == len(set(cq)), f"{tid}: constants[].quantity is not unique: {cq}"

    for d in draws:
        q = d["quantity"]
        _check_tokens(f"draws[{q}].fields", d["fields"])
        assert bool(d["fields"]) != bool((d.get("not_observed_because") or "").strip()), (
            f"{tid}: draws[{q}]: `fields` is empty exactly when `not_observed_because` says "
            "where the drawn quantity lives instead")
        dist = d["distribution"]
        lo, hi = dist.get("low"), dist.get("high")
        assert (lo is None) == (hi is None), f"{tid}: draws[{q}]: low/high must both be given or both null"
        if lo is not None:
            widths = {len(d["fields"])}
            if _is_mappable(doc):
                widths.add(sum(len(_columns_of(doc, t)) for t in d["fields"]))
            assert len(lo) == len(hi) and len(lo) in widths, (
                f"{tid}: draws[{q}]: low/high of length {len(lo)}/{len(hi)} against "
                f"{len(d['fields'])} fields ({sorted(widths)} columns) -- one bound per entry")
            for a, b in zip(lo, hi):
                assert a < b, f"{tid}: draws[{q}]: low {a} is not below high {b}"
        probs = dist.get("probabilities")
        if probs is not None:
            assert abs(sum(probs) - 1.0) <= 1e-9, f"{tid}: draws[{q}]: probabilities sum to {sum(probs)}"
        if dist.get("family") == "categorical":
            assert probs is not None, f"{tid}: draws[{q}]: categorical without probabilities"
        if dist.get("family") == "bernoulli":
            assert dist.get("p") is not None, f"{tid}: draws[{q}]: bernoulli without p"
        derived = d.get("derived_from")
        assert (dist["family"] == "derived") == (derived is not None), (
            f"{tid}: draws[{q}]: family `derived` exactly when derived_from is set")
        if derived is not None:
            assert derived in quantities and derived != q, (
                f"{tid}: draws[{q}].derived_from = {derived!r} is not another draw's quantity")

    for c in constants:
        _check_tokens(f"constants[{c['quantity']}].fields", c.get("fields") or ())
        assert str(c["value"]).strip(), f"{tid}: constants[{c['quantity']}] has an empty value"

    drawn = _roles(draws)
    assert (block["kind"] == "deterministic") == (not draws), (
        f"{tid}: kind {block['kind']!r} with {len(draws)} draws -- deterministic is an "
        "empty `draws` with a `deterministic_because`, never an empty list on its own")
    if block["kind"] == "deterministic":
        assert (block.get("deterministic_because") or "").strip(), f"{tid}: deterministic without a reason"

    omits = set(block["prompt_omits"])
    assert omits <= drawn, f"{tid}: prompt_omits {sorted(omits - drawn)} are not drawn roles"
    if block["prompt_states"] is None:
        assert omits == drawn, (
            f"{tid}: prompt_states is null, so the prompt omits EVERY drawn role -- "
            f"prompt_omits {sorted(omits)} != draws {sorted(drawn)}")
        assert block.get("prompt_field") is None, f"{tid}: prompt_field set with no prompt_states"
    else:
        assert block.get("prompt_field"), f"{tid}: prompt_states without prompt_field"

    accounted = drawn | _roles(constants)
    assert {"agent_pose", "goal"} <= accounted, (
        f"{tid}: {sorted({'agent_pose', 'goal'} - accounted)} appear in neither draws nor "
        "constants. The two questions this block exists to answer are never answered by "
        "silence: a fixed or absent goal is a constant whose value starts `none --`")

    if block["mechanism"] == "pinned_instances":
        assert isinstance(block.get("n_instances"), int) and block["n_instances"] >= 2
        assert isinstance(block.get("instance_seed"), int)
    else:
        assert block.get("n_instances") is None and block.get("instance_seed") is None, (
            f"{tid}: n_instances/instance_seed are for pinned_instances only")


@pytest.mark.parametrize("task_id", SPEC_IDS)
def test_the_reset_entry_belongs_to_this_specs_adapter(task_id, specs):
    """C: `entry` names THE reset this task's adapter runs. Resolved from the registry
    (the class the factory constructs, by `ast`, constructing nothing) to the symbol and
    span the block cites. Silent failure: a walk block pasted onto reach is mostly right --
    the +/-0.01 noise entries are shared -- and reads fine; only the symbol says which
    `_task_reset` was actually read. Every repo `source.path` is resolved too; library
    paths only where the package is installed, and the rest are counted, not assumed."""
    import importlib.util
    import inspect

    from bird import registry, tasks

    doc = specs[task_id]
    block = _reset_block(doc)
    if block is None:
        pytest.fail(f"{task_id}: no env.reset (test A names every one missing)")

    spec = tasks.index()[task_id]
    registry.load_all()
    if spec.bird_env_id not in registry.names("env"):
        assert task_id in tasks.no_adapter_ledger(), f"{task_id}: no adapter and not in the ledger"
        pytest.skip(f"{task_id}: no adapter to resolve `entry` against")

    factory = registry.get("env", spec.bird_env_id)
    cls = _registered_class(factory)
    module = inspect.getmodule(cls)
    mro = {c.__name__ for c in cls.__mro__}

    # A family-dispatch adapter (assistax's `_FAMILIES[task]["reset"]`) may run a hook
    # DEFINED in another module (the row records it under "module"). The symbol is then a
    # module-level name of THAT file, and that file is what `entry.path` must name.
    families = getattr(module, "_FAMILIES", None)
    row = families.get(spec.env_id) if isinstance(families, dict) else None
    family_reset = row.get("reset") if isinstance(row, dict) else None
    hook_module = inspect.getmodule(family_reset) if callable(family_reset) else module
    module_names = set(vars(hook_module))

    entry = block["entry"]
    symbol = str(entry.get("symbol") or "")
    first = symbol.split(".")[0]
    assert first in mro or first in module_names, (
        f"{task_id}: entry.symbol {symbol!r} -- `{first}` is neither in "
        f"{cls.__name__}'s MRO {sorted(mro - {'object'})} nor a module-level name of "
        f"{hook_module.__name__}. This spec's adapter does not run that reset")

    # THE PASTED-BLOCK DETECTOR. A class that defines a hook itself is the class whose
    # hook must be named; a family-dispatch adapter is named by the family's own function.
    if callable(family_reset):
        assert symbol == family_reset.__name__, (
            f"{task_id}: entry.symbol {symbol!r}, but this task's reset is "
            f"`{module.__name__}._FAMILIES[{spec.env_id!r}]['reset']` = "
            f"`{family_reset.__name__}`")
    else:
        own = [h for h in _RESET_HOOKS if h in vars(cls)]
        if own:
            assert symbol in {f"{cls.__name__}.{h}" for h in own}, (
                f"{task_id}: entry.symbol {symbol!r}, but {cls.__name__} defines "
                f"{own} itself -- name that hook, not an ancestor's")

    entry_path = ROOT / str(entry["path"])
    assert entry_path.is_file(), f"{task_id}: entry.path {entry['path']} does not exist"
    # `symbol` is resolved INSIDE entry.path, so entry.path must be the file that defines
    # the named symbol -- the class in the MRO named by its first component, or the module
    # for a module-level hook. Without this a right symbol in a wrong file passes every
    # check above and fails only if that file happens to lack a same-named def.
    defining = next((c for c in cls.__mro__ if c.__name__ == first), hook_module)
    defined_in = Path(inspect.getsourcefile(defining)).resolve()
    assert entry_path.resolve() == defined_in, (
        f"{task_id}: entry.path {entry['path']} is not the file that defines `{first}` "
        f"({defined_in.relative_to(ROOT)})")
    try:
        span = _ast_span(entry_path, symbol)
    except LookupError as exc:
        pytest.fail(f"{task_id}: entry.symbol does not resolve in entry.path: {exc}")
    lo, hi = _parse_lines(entry["lines"])
    assert span[0] <= lo and hi <= span[1], (
        f"{task_id}: entry.lines {entry['lines']} is not inside `{symbol}` "
        f"({span[0]}-{span[1]} in {entry['path']}). Line numbers aged, or the wrong def")

    # Every source: repo paths resolved and range-checked; library paths where installed.
    unchecked = []
    for e in list(block["draws"]) + list(block["constants"]):
        src = e["source"]
        rel = str(src["path"])
        local = ROOT / rel
        if local.is_file():
            target = local
        else:
            top = rel.split("/")[0]
            found = importlib.util.find_spec(top) if top.isidentifier() else None
            if found is None or not found.origin:
                unchecked.append(rel)
                continue
            target = Path(found.origin).resolve().parent.parent / rel
            assert target.is_file(), (
                f"{task_id}: {e['quantity']}: source.path {rel} -- `{top}` is installed "
                f"and {target} does not exist")
        if src.get("lines") is not None:
            lo, hi = _parse_lines(src["lines"])
            n_lines = len(target.read_text(errors="replace").splitlines())
            assert hi <= n_lines, (
                f"{task_id}: {e['quantity']}: source.lines {src['lines']} beyond the "
                f"{n_lines} lines of {rel}")
    if unchecked:
        print(f"{task_id}: {len(unchecked)} library source path(s) unchecked "
              f"(package not installed): {sorted(set(unchecked))}")


# --------------------------------------------------------------------------
# Every `{path, lines[, symbol]}` citation of a file IN THIS REPO, held to the file
# --------------------------------------------------------------------------
#
# C above holds `entry.lines` to `entry.symbol`'s ast span, and every draw's and
# constant's `source.lines` only to the file's LENGTH. A length check lets citations go
# stale silently: an insertion ABOVE the methods the citations point into (a docstring
# in `reference_reward`, lines added to `_reset`) keeps every `hi` under the new line
# count, and the `entry.lines` that C does hold get re-derived while the
# `constants[].source` beside them -- same symbol, same span -- do not. A line number
# that is not held to what it names is a number the next insertion falsifies, and it
# falsifies it without a test moving.
#
# So every citation whose `path` is a file in this repo is held to the file's CONTENT,
# wherever in the spec it sits: a symbol-bearing one to the symbol's ast span; a
# symbol-less one into a `.py` file to what its own prose names -- the identifiers it
# quotes in backticks, its string literals, decimals and coordinate tuples -- of which
# at least one must occur in the cited lines (a draw may also be held by `rng`, since a
# draw's source is where the generator is consumed), and where a quoted name is a def
# the lines overlap, that def must contain the lines, be contained by them, or open on
# their first line: a range that names `task_metric` and straddles its boundary is a
# range the file moved under. Only names the FILE holds anywhere count, so quoting
# upstream's `reset_model` beside the adapter's lines does not fail a correct citation;
# a citation whose prose names nothing the file holds keeps the length check, and a
# `symbol` is how to hold it harder. Library paths (`humanoid_bench/...`,
# `metaworld/...`) stay with C's length check where the package is installed: their
# drift is upstream's to own.
#
# A symbol-less range is also refused when it OPENS inside a docstring: nobody cites
# the third line of a docstring as a draw's source, and a whole-def range shifted by
# a few dozen lines lands on the next class's docstring more often than anywhere else.
#
# Measured against a file moved under ~100 stale citations: all 13 stale
# symbol-bearing draw/constant citations fail here, and 34 of the 39 stale symbol-less
# ones do -- 18 for opening inside a docstring, 16 for naming nothing the cited lines
# hold; the five it cannot see are whole-class or multi-def ranges that landed on lines
# quoting the same names. It also refuses CORRECT citations whose prose names nothing
# their lines hold (e.g. `object_velocity` constants citing
# `qvel = self._init_qvel.copy()`); the remedy the message asks for is to name the def
# the range sits in as `symbol`. A three-line shift of one citation fails it by spec,
# key path and symbol -- the mutation this was checked by.

def _citations(doc: dict) -> list:
    """Every `{path, lines[, symbol]}` mapping in `doc`, as (key_path, mapping, owner)
    in document order. `owner` is the entry the citation belongs to -- the draw or
    constant whose `source` it is, or the mapping itself when it is a list element (an
    `authored_from` row) -- and is where the prose that names things lives."""
    out = []

    def walk(node, key, parent):
        if isinstance(node, dict):
            if isinstance(node.get("path"), str) and node.get("lines") is not None:
                out.append((key, node, parent if isinstance(parent, dict) else node))
            for k, v in node.items():
                walk(v, f"{key}.{k}" if key else str(k), node)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{key}[{i}]", node)

    walk(doc, "", None)
    return out


def _anchors(owner, file_text: str) -> set:
    """What `owner`'s prose names that `file_text` holds anywhere: identifiers inside
    backticks (`rng.uniform(-r, r, size=n_q)` names `rng`, `uniform`, `size`, `n_q`),
    double-quoted literals ("catch"), decimals (7.5, 0.35) and numeric tuples
    ((0.7, -0.5, 1.0)). Python keywords, builtins and `self` are not names; two
    characters or fewer is a variable letter, not a name. A draw is also anchored by
    `rng` -- its source is where the generator is consumed."""
    import builtins
    import keyword
    import re as _re

    stop = set(keyword.kwlist) | set(dir(builtins)) | {"self", "cls", "np", "numpy"}
    strings = []

    def collect(node):
        if isinstance(node, str):
            strings.append(node)
        elif isinstance(node, dict):
            for v in node.values():
                collect(v)
        elif isinstance(node, list):
            for v in node:
                collect(v)

    collect(owner)
    found = set()
    for s in strings:
        for frag in _re.findall(r"`([^`\n]+)`", s):
            found.update(t for t in _re.findall(r"[A-Za-z_]\w*", frag)
                         if len(t) >= 3 and t not in stop)
        found.update(_re.findall(r'"([^"\n]{3,40})"', s))
        found.update(_re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", s))
        found.update(_re.sub(r"\s+", " ", t) for t in _re.findall(r"\(\s*-?\d[\d.\s,-]*\)", s))
    anchors = {a for a in found if _names(a, file_text)}
    if "distribution" in owner:
        anchors.add("rng")
    return anchors


def _names(anchor: str, text: str) -> bool:
    """`anchor` occurs in `text` -- as a whole word when it is an identifier, so that
    `key` does not pass on `keyframe`."""
    import re as _re

    if _re.fullmatch(r"[A-Za-z_]\w*", anchor):
        return _re.search(rf"\b{_re.escape(anchor)}\b", text) is not None
    return anchor in text


@functools.lru_cache(maxsize=None)
def _def_nodes(path_str: str) -> tuple:
    """(name, first, last) for every def and class in the file at any depth, by `ast`."""
    import ast

    tree = ast.parse(Path(path_str).read_text())
    return tuple((n.name, int(n.lineno), int(n.end_lineno)) for n in ast.walk(tree)
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))


@functools.lru_cache(maxsize=None)
def _docstring_spans(path_str: str) -> tuple:
    """(first, last) of every docstring in the file -- the string that opens a module,
    class or def body -- by `ast`."""
    import ast

    tree = ast.parse(Path(path_str).read_text())
    spans = []
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body:
            first = body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                spans.append((int(first.lineno), int(first.end_lineno)))
    return tuple(spans)


def _repo_citation_cases() -> list:
    cases = []
    for path in SPEC_PATHS:
        for key, cite, owner in _citations(_load(path)):
            if (ROOT / str(cite["path"])).is_file():
                cases.append(pytest.param(path.parent.name, key, cite, owner,
                                          id=f"{path.parent.name}:{key}"))
    return cases


@pytest.mark.parametrize("task_id,key,cite,owner", _repo_citation_cases())
def test_a_citation_of_a_repo_file_still_names_what_it_cites(task_id, key, cite, owner):
    rel = str(cite["path"])
    target = ROOT / rel
    lo, hi = _parse_lines(cite["lines"])
    text = target.read_text(errors="replace")
    lines = text.splitlines()
    assert hi <= len(lines), (
        f"{task_id}: {key}: lines {cite['lines']} beyond the {len(lines)} lines of {rel}")
    if target.suffix != ".py":
        return   # an XML scene or a YAML config has no ast; the length check is all there is

    symbol = cite.get("symbol")
    if symbol:
        try:
            span = _ast_span(target, str(symbol))
        except LookupError as exc:
            pytest.fail(f"{task_id}: {key}: symbol does not resolve in {rel}: {exc}")
        assert span[0] <= lo and hi <= span[1], (
            f"{task_id}: {key}: lines {cite['lines']} is not inside `{symbol}` "
            f"({span[0]}-{span[1]} in {rel}). The file moved under the citation")
        return

    opened_in = [(a, b) for a, b in _docstring_spans(str(target)) if a <= lo <= b]
    assert not opened_in, (
        f"{task_id}: {key}: lines {cite['lines']} open inside a docstring "
        f"({opened_in[0][0]}-{opened_in[0][1]} in {rel}) -- not a line anyone cited on "
        "purpose. The file moved under the citation")
    anchors = _anchors(owner, text)
    if not anchors:
        return   # nothing named that the file holds; the length check above is all there is
    cited = "\n".join(lines[lo - 1:hi])
    assert any(_names(a, cited) for a in anchors), (
        f"{task_id}: {key}: none of what its prose names ({sorted(anchors)}) occurs in "
        f"{rel}:{cite['lines']}:\n{cited}\nThe file moved under the citation, or the prose "
        "names something the lines do not hold -- add a `symbol` if the range is a def's")
    named = [(n, a, b) for n, a, b in _def_nodes(str(target))
             if n in anchors and a <= hi and lo <= b]
    if named:
        assert any((a <= lo and hi <= b) or (lo <= a and b <= hi) or lo == a
                   for _, a, b in named), (
            f"{task_id}: {key}: lines {cite['lines']} straddle "
            f"{[f'{n} {a}-{b}' for n, a, b in named]} in {rel} -- neither inside the def "
            "its own prose names, nor containing it, nor opening on its first line")


def _normalise_ws(text) -> str:
    return " ".join(str(text).split())


@pytest.mark.parametrize("path", SPEC_PATHS, ids=SPEC_IDS)
def test_the_reset_prompt_claim_is_verbatim(path):
    """D: `prompt_states` is a substring of the field `prompt_field` names, whitespace
    aside. Silent failure: record drift -- a paraphrase that reads as a quote is a
    recurring class of defect, and here it is the citable artifact."""
    doc = _load(path)
    tid = path.parent.name
    block = _reset_block(doc)
    if block is None:
        pytest.fail(f"{tid}: no env.reset (test A names every one missing)")
    quote = block["prompt_states"]
    if quote is None:
        return
    field = block["prompt_field"]
    needle = _normalise_ws(quote)
    assert needle, f"{tid}: prompt_states is blank"
    if field == "flat_fields":
        haystacks = [_normalise_ws(f.get("description") or "")
                     for f in doc["state_surface"].get("flat_fields") or ()]
        assert any(needle in h for h in haystacks), (
            f"{tid}: prompt_states is not a substring of any flat_fields[].description:\n"
            f"  {quote!r}")
    else:
        source = doc["description"].get(field)
        assert source, f"{tid}: prompt_field {field!r} is empty or absent on this spec"
        assert needle in _normalise_ws(source), (
            f"{tid}: prompt_states is not a verbatim substring of description.{field}:\n"
            f"  {quote!r}")


def test_the_reset_name_check_set_is_exactly_the_unmappable_specs(specs):
    """`RESET_NAME_CHECK_ONLY` is an EQUALITY with the specs whose `state_surface` has no
    `flat_fields` and no `fields[].slice`. A spec that gains slices leaves the set (and F
    starts measuring it); a new sliceless spec fails here rather than joining it silently."""
    unmappable = {tid for tid, doc in specs.items() if not _is_mappable(doc)}
    assert unmappable == set(RESET_NAME_CHECK_ONLY), (
        f"unmappable but not listed: {sorted(unmappable - RESET_NAME_CHECK_ONLY)}; "
        f"listed but now mappable: {sorted(RESET_NAME_CHECK_ONLY - unmappable)}")


@pytest.mark.parametrize("env_id", _spec_backed_env_ids())
def test_the_declared_reset_draws_are_exactly_what_moves(env_id):
    """F: the measurement. `_RESET_SAMPLES` seeded resets stacked; the union of
    `draws[].fields` must EQUAL the set of columns that moved, in both directions;
    `constants[].fields` must not move; `kind` must agree with whether anything did; and
    `reset(default_rng(3))` twice must be bitwise equal -- the universal reproducibility
    check that replaces a per-spec boolean. Silent failure: the one the block exists to
    remove -- "seed changes: goal" on window_close when only y moves, "hand start fixed"
    when it is not. Gated per simulator; natives and toys run everywhere."""
    import numpy as np

    from bird import registry, tasks

    registry.load_all()
    factory = registry.get("env", env_id)
    package = _SIM_PACKAGES.get(getattr(factory, "__module__", ""))
    if package:
        pytest.importorskip(package)
    spec = tasks.by_env_id(env_id)
    doc = dict(spec.raw)
    block = _reset_block(doc)
    if block is None:
        pytest.fail(f"{spec.id}: no env.reset (test A names every one missing)")

    # THE SOLVER THIS BLOCK WAS MEASURED UNDER, or this test cannot evaluate its own
    # claim. What moves across seeded resets is a property of the SOLVER as much as of
    # the scene: the same scene under two mujoco wheels can move a column across the
    # 1e-6 line, and the verdict flips. Under a
    # wheel the spec was not measured on, a FAILURE would report a verdict this test does
    # not have, and "fixing" it by re-measuring would overwrite a correct record with one
    # taken under a wheel the tier does not run. So: skip, loudly, naming both versions.
    #
    # uv.lock carries both -- 3.3.0 pinned exactly by `assistax`/`metaworld`/`all`,
    # 3.13.0 by the `jax` extra -- so which one is installed is a property of the venv,
    # not of the tree. No CI job is affected (`--extra test` has no mujoco at all and
    # `--extra all` pins 3.3.0); the population that hits this is whoever installed the
    # jax extra (upstream Assistax's MJX backend).
    recorded = str((((doc.get("env") or {}).get("library") or {})
                    .get("stack") or {}).get("mujoco") or "")
    if recorded:
        # Keyed on what the SPEC records, not on `_SIM_PACKAGES`: a family with no row
        # there (one that gates through `_simulator_missing()` instead) would make a
        # `package == "mujoco"` test silently never fire for it. Ask the interpreter
        # directly.
        try:
            import mujoco
        except ImportError:
            mujoco = None
        if mujoco is not None and str(mujoco.__version__) != recorded:
            pytest.skip(
                f"{spec.id}: measured under mujoco {recorded}, installed is "
                f"{mujoco.__version__} -- this guard cannot judge a reset block under a "
                f"different solver ({tasks.__file__.rsplit('/', 2)[0]}/"
                f"tasks/{spec.id}/shared_spec.yaml). Re-run under the recorded wheel; "
                "do NOT re-measure.")

    try:
        adapter = factory(None)
    except _simulator_missing() as exc:
        pytest.skip(f"{env_id} needs a simulator this install lacks: {exc}")

    rows = [np.asarray(adapter.reset(np.random.default_rng(i)), dtype=float).ravel()
            for i in range(_RESET_SAMPLES)]
    S = np.stack(rows)
    # "Moved" is the constants' own tolerance below (1e-6), not exact inequality: on
    # box-close, disassemble and peg-unplug-side the hand settles with 1e-14 of float
    # jitter between instances (an object resting where the mocap weld pulls the hand),
    # which is not a draw and must not be declared as one.
    moving = {i for i in range(S.shape[1]) if np.ptp(S[:, i]) > 1e-6}

    a = np.asarray(adapter.reset(np.random.default_rng(3)))
    b = np.asarray(adapter.reset(np.random.default_rng(3)))
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), (
        f"{spec.id}: reset(default_rng(3)) twice is not bitwise equal -- the episode is "
        "not reproducible from the seed it was handed")

    assert (block["kind"] == "deterministic") == (not moving), (
        f"{spec.id}: kind {block['kind']!r} but {len(moving)} column(s) moved across "
        f"{_RESET_SAMPLES} seeded resets")

    if block["mechanism"] == "pinned_instances" and hasattr(adapter, "n_tasks"):
        assert block["n_instances"] == adapter.n_tasks, (
            f"{spec.id}: n_instances {block['n_instances']} against the adapter's "
            f"{adapter.n_tasks} pinned instances")

    tokens = [t for e in list(block["draws"]) + list(block["constants"])
              for t in (e.get("fields") or ())]
    if spec.id in RESET_NAME_CHECK_ONLY:
        if not all(":" in t for t in tokens):
            return               # a NAME on a sliceless spec: checked by B, unmeasurable
        # Every token is a slice of the adapter's flat state: measurable after all, and
        # bounded by the width `reset()` returned rather than the spec's `obs_dim`.
        for t in tokens:
            hi = int(t.split(":")[1])
            assert hi <= S.shape[1], (
                f"{spec.id}: fields slice {t} outside the {S.shape[1]}-column state reset() returns")
    else:
        n = int(doc["env"]["spaces"]["obs_dim"])
        assert S.shape[1] == n, f"{spec.id}: reset() returned {S.shape[1]} columns, spec says {n}"
    flat = {int(f["index"]): f["name"] for f in doc["state_surface"].get("flat_fields") or ()}

    def _label(cols):
        return [f"{i}={flat.get(i, '?')}" for i in sorted(cols)]

    declared = set()
    for d in block["draws"]:
        for tok in d["fields"]:
            declared |= _columns_of(doc, tok)
    assert declared == moving, (
        f"{spec.id}: draws[].fields != the columns that move across {_RESET_SAMPLES} "
        f"seeded resets.\n  move but undeclared: {_label(moving - declared)}\n"
        f"  declared but did not move: {_label(declared - moving)}")

    for c in block["constants"]:
        for tok in c.get("fields") or ():
            cols = sorted(_columns_of(doc, tok))
            spread = np.ptp(S[:, cols], axis=0)
            assert np.all(spread <= 1e-6), (
                f"{spec.id}: constants[{c['quantity']}].fields `{tok}` moved by up to "
                f"{spread.max():.3g} across seeded resets")
