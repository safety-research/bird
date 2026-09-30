"""`scripts/refresh_spec_lines.py` moves a spec's source-line citations with the code.

The failure it exists for is silent in one direction and loud in the other: a citation
that aged reads fine on the page and fails only
`test_the_reset_entry_belongs_to_this_specs_adapter`, every spec at once when a hook is
added to every adapter. What is pinned here is the part that must not be clever:

  * a cited line that survived an edit lands exactly where the diff put it;
  * a cited range whose own lines changed is LEFT ALONE and reported, never guessed;
  * only the digits of a moved `lines:` value change -- quotes, comments and every
    other byte of the spec stay as written;
  * the generated families are skipped by name.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("refresh_spec_lines",
                                                  ROOT / "scripts" / "refresh_spec_lines.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refresh_spec_lines"] = mod
    spec.loader.exec_module(mod)
    return mod


rsl = _load()

OLD = "\n".join(["import x", "", "class A:", "    def _reset(self):", "        a = 1",
                 "        b = 2", "        return a + b", "", "class B:", "    pass", ""])
# Three lines inserted above `A`, one line CHANGED inside `_reset`, `B` untouched.
NEW = "\n".join(["import x", "import y", "", "HELP = 1", "", "class A:", "    def _reset(self):",
                 "        a = 1", "        b = 3", "        return a + b", "", "class B:",
                 "    pass", ""])


def test_a_surviving_line_maps_to_exactly_where_the_diff_put_it():
    lm = rsl.LineMap(OLD, NEW)
    assert lm.map[3] == 6 and lm.map[4] == 7 and lm.map[5] == 8     # class A / def / a = 1
    assert lm.map[9] == 12 and lm.map[10] == 13                    # class B / pass
    assert 6 not in lm.map, "the changed line `b = 2` has no new home"
    assert lm.shift(3, 5) == (6, 8)
    assert lm.shift(9, 10) == (12, 13)
    assert lm.shift(4, 4) == (7, 7)


def test_a_range_whose_own_lines_changed_is_refused_not_guessed():
    lm = rsl.LineMap(OLD, NEW)
    assert lm.shift(4, 7) is None, "the range straddles the edited line"
    assert lm.shift(6, 6) is None
    assert rsl.LineMap(OLD, OLD).identical is True


SPEC = """\
env:
  reset:
    entry:
      path: bird/envs/fake.py
      symbol: A._reset
      lines: 4-5   # the draws
    constants:
    - source:
        path: bird/envs/fake.py
        lines: '9'
    - source:
        path: bird/envs/fake.py
        lines: 4-7
    - source:
        path: humanoid_bench/envs/x.py
        lines: 4-5
    - source:
        path: bird/envs/other.py
        lines: 4-5
"""


def test_only_the_digits_of_a_moved_citation_change_and_the_rest_is_reported():
    maps = {"bird/envs/fake.py": rsl.LineMap(OLD, NEW),
            "bird/envs/other.py": rsl.LineMap(OLD, OLD)}
    report = []
    out = rsl.refresh_spec(SPEC, maps, report, "fake_task")
    assert "      lines: 7-8   # the draws\n" in out, "moved, comment kept, no quotes added"
    assert "        lines: '12'\n" in out, "a single quoted line keeps its quotes"
    assert "        lines: 4-7\n" in out, "the straddling range is left as written"
    assert out.count("lines: 4-5") == 2, "the library path and the unchanged file are untouched"
    assert [r for r in report if "look by hand" in r] == [
        "fake_task: bird/envs/fake.py lines 4-7 -- the cited lines themselves changed; "
        "left as written, look by hand"]
    assert sorted(r for r in report if " -> " in r) == [
        "fake_task: bird/envs/fake.py lines 4-5 -> 7-8",
        "fake_task: bird/envs/fake.py lines 9-9 -> 12",
    ]
    # Byte-for-byte: every line that is not a rewritten `lines:` is unchanged.
    before = [ln for ln in SPEC.splitlines() if not ln.lstrip().startswith("lines:")]
    after = [ln for ln in out.splitlines() if not ln.lstrip().startswith("lines:")]
    assert before == after


AUTHORED_FROM = """\
provenance:
  authored_from:
  - path: bird/envs/fake.py
    lines: 9-10
    what: 'B: the class, cited as a list row the way every authored delta does'
  - path: bird/envs/fake.py
    lines: 4-7
    what: 'A._reset: straddles the edit, so it must be left alone and reported'
"""


def test_a_list_row_citation_moves_like_a_mapping_one():
    """`provenance.authored_from` rows are LIST elements -- `- path:` then `lines:` -- and a
    field pass that matched only a bare `path:` key would skip them, leaving such rows at
    line numbers every module edit ages (the citation test tolerates them: a `what:` of
    'the reward, ...' carries no backticked anchor). The list marker is part of the
    path line, not a reason to skip it."""
    maps = {"bird/envs/fake.py": rsl.LineMap(OLD, NEW)}
    report = []
    out = rsl.refresh_spec(AUTHORED_FROM, maps, report, "fake_task")
    assert "  - path: bird/envs/fake.py\n    lines: 12-13\n" in out, "the list row's range moved with class B"
    assert "    lines: 4-7\n" in out, "the straddling list row is left as written"
    assert [r for r in report if " -> " in r] == ["fake_task: bird/envs/fake.py lines 9-10 -> 12-13"]
    assert [r for r in report if "look by hand" in r] == [
        "fake_task: bird/envs/fake.py lines 4-7 -- the cited lines themselves changed; "
        "left as written, look by hand"]
    before = [ln for ln in AUTHORED_FROM.splitlines() if not ln.lstrip().startswith("lines:")]
    after = [ln for ln in out.splitlines() if not ln.lstrip().startswith("lines:")]
    assert before == after


PROSE = ("note: 'the draw is `fake.py:4`, see also fake.py:4-5 and x.py:4 and other.py:9; "
         "a range that straddles the edit, fake.py:4-7, stays.'\n")


def test_prose_mentions_move_with_the_code_and_library_files_do_not():
    maps = {"bird/envs/fake.py": rsl.LineMap(OLD, NEW),
            "bird/envs/other.py": rsl.LineMap(OLD, OLD)}
    report = []
    out = rsl.refresh_prose(PROSE, maps, report, "fake_task")
    assert "`fake.py:7`" in out and "fake.py:7-8 and" in out
    assert "x.py:4" in out, "a basename that is not a cited repo file is prose, not a citation"
    assert "other.py:9" in out, "an unchanged file keeps its numbers"
    assert "fake.py:4-7, stays" in out, "a straddling range is left as written"
    assert any("fake.py:4-7" in r and "look by hand" in r for r in report)
    assert sorted(r for r in report if " -> " in r) == [
        "fake_task: prose `fake.py:4-5` -> `fake.py:7-8`",
        "fake_task: prose `fake.py:4` -> `fake.py:7`",
    ]
    # A basename two cited repo files share is never guessed.
    two = {"bird/envs/fake.py": rsl.LineMap(OLD, NEW), "bird/other/fake.py": rsl.LineMap(OLD, NEW)}
    report = []
    assert rsl.refresh_prose("see fake.py:4", two, report, "t") == "see fake.py:4"
    assert report and "two cited files share" in report[0]


def test_a_bare_line_after_a_basename_means_that_file_and_a_library_name_resets_it():
    """`(metaworld.py:1302). ... UNSEEDED (:1303)` -- the bare number is the same file.
    After a library basename it is not ours to move; a time or a key never matches."""
    maps = {"bird/envs/fake.py": rsl.LineMap(OLD, NEW)}
    report = []
    text = ("first fake.py:4 then a bare one at :5 and (:4-5); then env.py:12 makes :4 a "
            "library line; time 12:30; key: value; and [:9] before any name is left")
    out = rsl.refresh_prose(text, maps, report, "t")
    assert out.startswith("first fake.py:7 then a bare one at :8 and (:7-8); then env.py:12 makes :4 a")
    assert "time 12:30; key: value; and [:9] before" in out
    assert [r for r in report if "bare" in r] == [
        "t: prose `:5` -> `:8` (bare, after fake.py)",
        "t: prose `:4-5` -> `:7-8` (bare, after fake.py)",
    ]
    # A bare number before any basename in the text is never moved.
    assert rsl.refresh_prose("at :4 then fake.py:4", maps, [], "t") == "at :4 then fake.py:7"


def test_a_bare_line_is_attributed_within_its_own_yaml_field_only():
    """A draw's `text:` cites the file its own `source.path` names -- a Meta-World
    wheel file that no `.py:` mention in the text ever names -- with bare numbers, and
    the `rng_note:` a few lines above names `base.py`. Attributed across that field
    boundary, `obj_low/obj_high :40-41` would move by base.py's offset in every such spec.
    The bare number in the SAME field as its name still moves (the case above)."""
    maps = {"bird/envs/fake.py": rsl.LineMap(OLD, NEW)}
    text = ("    rng_note: 'the instance is picked (fake.py:4); then `env.reset()` is called\n"
            "      UNSEEDED (:5) and the draw is read.'\n"
            "    draws:\n"
            "      - role: object_pose\n"
            "        distribution:\n"
            "          text: 'obj_low/obj_high :4-5, written to the body (:9); the goal\n"
            "            :10 is derived.'\n"
            "        source:\n"
            "          path: metaworld/envs/sawyer_fake_v3.py\n"
            "          lines: 4-5\n")
    report = []
    out = rsl.refresh_prose(text, maps, report, "t")
    assert "(fake.py:7); then `env.reset()` is called\n      UNSEEDED (:8) and" in out, \
        "the bare number in the field that named the file still moves"
    assert "text: 'obj_low/obj_high :4-5, written to the body (:9); the goal\n            :10 is" in out, \
        "a bare number in a field that names no file is not ours to move"
    assert report == ["t: prose `fake.py:4` -> `fake.py:7`",
                      "t: prose `:5` -> `:8` (bare, after fake.py)"]
    # The same numbers on one line, no field between them, move: the boundary is the
    # start of a FIELD, not the end of a line.
    one = "    note: 'see fake.py:4 and then\n      the bare :5 too'\n"
    assert rsl.refresh_prose(one, maps, [], "t") == "    note: 'see fake.py:7 and then\n      the bare :8 too'\n"

def test_a_bare_line_reaches_no_further_than_its_own_yaml_field():
    """`rng_note: '... base.py:475-476'` and, two fields later, `text: 'obj_low (:44-45)'`:
    the bare number is the metaworld task file's -- the one that draw's own `source.path`
    names -- and would stay put only as long as base.py's first fifty lines do not move.
    A `key:` line ends the reach of the last `.py:` mention. Mutation: drop `_FIELD_RE`
    from the events -- `(:4-5)` below becomes `(:7-8)` and `jointly (:5)` becomes `(:8)`."""
    maps = {"bird/envs/fake.py": rsl.LineMap(OLD, NEW)}
    text = ("    rng_note: 'reads the Generator (fake.py:4) and again\n"
            "      at (:5), on the next line of the same field'\n"
            "    draws:\n"
            "      - role: object_pose\n"
            "        distribution:\n"
            "          text: 'obj_low/obj_high (:4-5), drawn jointly (:5)'\n"
            "        source:\n"
            "          path: metaworld/envs/sawyer_xyz/v3/sawyer_push_v3.py\n")
    report = []
    out = rsl.refresh_prose(text, maps, report, "t")
    assert "(fake.py:7) and again\n      at (:8), on the next line" in out, out
    assert "'obj_low/obj_high (:4-5), drawn jointly (:5)'" in out, out
    assert [r for r in report if "bare" in r] == ["t: prose `:5` -> `:8` (bare, after fake.py)"]
    # The boundary only resets; a bare number after a `.py:` mention in the SAME field, past
    # a line break, still moves (the `rng_note` shape above), and a `key: value` inside prose
    # is not a boundary unless it starts the line.
    inline = "note: 'fake.py:4 then key: value then (:5)'"
    assert rsl.refresh_prose(inline, maps, [], "t") == "note: 'fake.py:7 then key: value then (:8)'"


def test_the_field_pass_and_the_prose_pass_compose_and_can_run_alone():
    maps = {"bird/envs/fake.py": rsl.LineMap(OLD, NEW)}
    text = SPEC + PROSE
    both = rsl.refresh_spec(text, maps, [], "t")
    assert "      lines: 7-8   # the draws\n" in both and "`fake.py:7`" in both
    fields_only = rsl.refresh_spec(text, maps, [], "t", prose=False)
    assert "      lines: 7-8   # the draws\n" in fields_only and "`fake.py:4`" in fields_only
    prose_only = rsl.refresh_spec(text, maps, [], "t", fields=False)
    assert "      lines: 4-5   # the draws\n" in prose_only and "`fake.py:7`" in prose_only


def test_cited_repo_paths_finds_every_repo_file_and_no_library_one():
    assert rsl.cited_repo_paths({"a": SPEC}) == ["bird/envs/fake.py", "bird/envs/other.py"]


def test_the_generated_families_are_skipped_by_prefix():
    assert {"assistax_", "upstream_assistax_", "jax_"} <= set(rsl.GENERATED_PREFIXES)
    for prefix in rsl.GENERATED_PREFIXES:
        assert (ROOT / "scripts" / rsl.generator_of(prefix)).is_file(), \
            f"{prefix}: skipped because a generator writes it, and that generator must exist"
    assert rsl.generator_of("assistax_") == "gen_assistax_specs.py"
    assert rsl.generator_of("upstream_assistax_") == "derive_jax_spec.py"
    assert rsl.generator_of("jax_") == "derive_jax_spec.py"


def test_check_mode_runs_against_head_and_reports():
    """A SMOKE test of the CLI, not a drift test: `--check --base HEAD` on a clean tree
    is a no-op, and on a tree with uncommitted adapter edits it legitimately reports
    drift (exit 1) -- which is exactly the state the script exists to fix. What is
    pinned is that it runs and reports either way; the drift itself is held by
    `tests/test_task_specs.py`'s citation tests."""
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "refresh_spec_lines.py"),
                        "--check", "--base", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    # A tree with uncommitted adapter edits legitimately reports drift; the assertion is
    # that the script RUNS and reports rather than raising.
    assert r.returncode in (0, 1), r.stderr
    assert "cited file(s) changed vs HEAD" in r.stderr
