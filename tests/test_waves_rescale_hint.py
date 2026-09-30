"""The `loop.waves` refusal must say what to write, not only what is wrong.

WHY THIS IS WORTH A TEST FILE. The refusal itself is correct and is not under
test here -- a wave list that disagrees with `generate.n_candidates` is an
unreported budget change, which is precisely what `_check_coherence` exists to
catch. What matters is that getting OUT of it is three coupled edits
(`generate.n_candidates`, `generate.crossover.n`, and the list), and a message
that states the mismatch and stops leaves the reader to derive them.
`configs/methods/rstar.yaml`'s header ("Running it budget-matched ... takes three
overrides") spells out exactly this override set.

The published point itself is NOT refused: probed across six load cases,
`configs/methods/rstar.yaml` loads under no profile, `tester` and `full`; only the
budget-matched `generate.n_candidates=8` line is refused, and that refusal is
the guard working. So the guard stays and the message carries the fix.

The suggestion is silent unless the rescale is EXACT, and that is the half most
worth pinning: a rounded suggestion would be a plausible wrong budget offered
BY the guard against plausible wrong budgets.
"""

import pytest

from bird.config import ConfigError, load
from bird.config import _rescaled_waves_hint as hint

RSTAR_WAVES = [
    {"llm": 12, "crossover": 4, "when": "always"},
    {"llm": 0, "crossover": 4, "when": "no_archive"},
]


def test_the_published_rstar_point_loads_under_every_profile():
    """The premise check: the published point loads.

    `rstar` carries its own wave list and the arithmetic holds at the published
    population of 16 -- steady 12+4, and a bootstrap iteration's 16+4 less the 4
    that cannot draw parents from an empty archive."""
    for profile in (None, "tester", "full"):
        cfg = load("rstar", profile=profile)
        assert cfg["loop.waves"] == RSTAR_WAVES
        assert cfg["generate.n_candidates"] == 16


def test_the_budget_matched_line_is_still_refused():
    """The guard is NOT relaxed: K=8 with the paper's waves stays a ConfigError.

    This is the assertion that would fail if someone 'fixed' the refusal by
    removing it."""
    with pytest.raises(ConfigError, match="generate.n_candidates=8"):
        load("rstar", overrides={"generate.n_candidates": 8})


def test_the_refusal_now_carries_the_line_to_write():
    """...and the message hands over the rescaled list and crossover share.

    A message that ends at "the two must agree" fails this. The numbers are
    the paper's 3:1 ratio at a population of 8 -- 6 LLM + 2 crossover -- which
    is what `configs/methods/rstar.yaml`'s header says."""
    with pytest.raises(ConfigError) as exc:
        load("rstar", overrides={"generate.n_candidates": 8})
    msg = str(exc.value)
    assert "loop.waves=[{llm:6,crossover:2,when:always},{llm:0,crossover:2,when:no_archive}]" in msg
    assert "generate.crossover.n=2" in msg


def test_the_whole_override_set_loads():
    """The line the message suggests is a line that works.

    Without this the suggestion is untested prose, which invites record drift:
    a comment that contradicts the code is worse than no comment, because the
    comment is the citable artifact."""
    cfg = load("rstar", overrides={
        "generate.n_candidates": 8,
        "generate.crossover.n": 2,
        "loop.waves": [{"llm": 6, "crossover": 2, "when": "always"},
                       {"llm": 0, "crossover": 2, "when": "no_archive"}],
    })
    assert cfg["generate.n_candidates"] == 8


def test_the_hint_is_silent_when_the_rescale_would_round():
    """A population the ratio does not divide gets no suggestion at all.

    12:4 at a population of 10 is 7.5:2.5. Offering 7:2 (sums to 9) or 8:3
    (sums to 11) would be the guard proposing a budget that fails its own
    check; offering 8:2 would silently change the crossover share, which is a
    method value. Silence is the only correct answer, and it is the clause a
    future 'helpful' edit is most likely to remove."""
    assert hint(RSTAR_WAVES, steady=16, want=10) == ""
    assert hint(RSTAR_WAVES, steady=16, want=6) == ""
    # A ratio that does divide, to prove the silence above is about the
    # arithmetic and not about the function having stopped working.
    assert "llm:3,crossover:1" in hint(RSTAR_WAVES, steady=16, want=4)


def test_the_hint_survives_a_malformed_wave_list_without_raising():
    """It runs inside the error path of `validate`, so it must never itself raise.

    `_check_coherence` collects EVERY problem rather than the first
    (`validate()`'s report-every-problem contract), so a hint that threw on a wave list that
    is malformed in a second way would replace a list of problems with a
    traceback -- losing the other problems and reporting the wrong one."""
    assert hint([{"llm": "twelve", "crossover": 4}], steady=16, want=8) == ""
    assert hint(["not a mapping"], steady=16, want=8) == ""
    assert hint([], steady=16, want=8) == ""
    assert hint(RSTAR_WAVES, steady=0, want=8) == ""
