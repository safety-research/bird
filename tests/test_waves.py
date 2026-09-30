"""`loop.waves`: one pass must be indistinguishable from no waves at all.

WHY THIS FILE EXISTS. The failure it guards is silent by construction.
`run_iteration` is a loop over passes that must reduce to the straight sequence
of six calls when there is one pass; if the loop changed anything -- an extra
`stage_begin`, a second `execute_rate` sample, reports appended to `all_reports`
at a different moment, candidate ids minted in a different order -- then EVERY
shipped config would still run, still pass its own tests, and still produce a
plausible number. The only thing that would have moved is the number. Looking at
the output cannot catch that; comparing whole run directories can.

It reuses `_scrub` / `_VOLATILE` / `_fingerprint` from `test_parallelism`
rather than restating them, for the reason `test_resume` gives for the same
import: two definitions of "volatile" that drift apart would let one test go
on passing while the property it names stopped holding.
"""

from pathlib import Path

import pytest

from test_parallelism import _fingerprint, _run  # noqa: F401


#: One wave that states the counts the config would have derived anyway. If
#: the wave machinery is a faithful generalisation, this is the identity.
_ONE_WAVE = [{"llm": 3, "crossover": 0, "when": "always"}]


def test_a_single_wave_is_the_unwaved_loop(tmp_path):
    """`waves: [one pass]` and `waves: []` must write identical run dirs."""
    plain = _run("eureka", tmp_path / "plain",
                 extra={"generate.n_candidates": 3, "loop.n_iterations": 2})
    waved = _run("eureka", tmp_path / "waved",
                 extra={"generate.n_candidates": 3, "loop.n_iterations": 2,
                        "loop.waves": _ONE_WAVE})
    assert set(plain) == set(waved), (
        "the two runs wrote different FILES: "
        f"only plain={sorted(set(plain) - set(waved))} "
        f"only waved={sorted(set(waved) - set(plain))}")
    differing = [k for k in sorted(plain) if plain[k] != waved[k]]
    assert not differing, f"a single wave changed {differing}"


def test_a_bootstrap_wave_runs_once_and_keeps_the_population_constant(tmp_path):
    """R*'s two-step first iteration: the second pass runs in iteration 1 only,
    and every iteration still trains `generate.n_candidates`.

    This is the property the budget match rests on. A bootstrap pass that also
    fired in iterations 2+ would train 4 where the baselines train 3, and the
    only visible symptom would be a better number.
    """
    import json

    out = tmp_path / "boot"
    # `seed=1` is MEASURED, not chosen for convenience: the mock seeds every sample
    # from its prompt, and at some seeds (0 among them, under the current toy_reacher
    # source) every module swap between the two mock parents is refused, so the
    # crossover makes nothing (`crossover_short`) and an iteration trains 2. That is
    # the crossover's own documented shortfall, not the wave schedule this test is
    # about. A prompt edit re-rolls this.
    _run("rstar", out,
         extra={"seed": 1,
                "generate.n_candidates": 3,
                "generate.crossover.n": 1,
                "loop.n_iterations": 3,
                "generate.alignment.iterations": 5,
                "loop.waves": [{"llm": 2, "crossover": 1, "when": "always"},
                               {"llm": 0, "crossover": 1, "when": "no_archive"}]})
    (run,) = [p for p in out.iterdir() if p.is_dir()]
    events = [json.loads(l) for l in (run / "journal.jsonl").read_text().splitlines()
              if l.strip()]

    waves = [e for e in events if e.get("stage") == "wave_begin"]
    by_iter = {}
    for e in waves:
        by_iter.setdefault(e["iteration"], []).append(e["wave"])
    assert by_iter.get(0) == [0, 1], (
        f"iteration 1 must run both passes, ran {by_iter.get(0)}")
    for it, seen in by_iter.items():
        if it == 0:
            continue
        assert seen == [0], f"iteration {it} must be single-pass, ran {seen}"

    # Counted from the candidate DIRECTORIES, whose names carry the iteration
    # (`iterNN_<id>`), not from `generate` events -- those record cand_id,
    # parent_id and chars and NO iteration, so a count keyed on one would be
    # zero for every iteration and the assertion below would pass while
    # measuring nothing. That vacuum is the failure mode this file is about,
    # so it is worth the extra line to not reproduce it here.
    per_iter = {}
    for d in (run / "candidates").iterdir():
        if not d.is_dir():
            continue
        it = d.name.split("_", 1)[0]
        per_iter[it] = per_iter.get(it, 0) + 1
    assert len(per_iter) == 3, f"expected 3 iterations of candidates, saw {per_iter}"
    assert set(per_iter.values()) == {3}, (
        f"every iteration must train generate.n_candidates=3, saw {per_iter}")
