"""R* parameter alignment reads and writes a program's float literals in ONE
index space.

WHY THIS FILE EXISTS. If `float_literals` enumerated with `ast.walk`
(breadth-first) while `_LiteralSwap` replaces depth-first, `theta[k]` would be
read from one literal and written to another whenever a literal sat deeper in an
earlier sibling than a shallower literal in a later sibling. Nothing would
crash: every SPSA run would start from a scrambled program, measure its
"before" accuracy on that scrambled program, and record `params_before` in an
order `params_after` did not share. The one visible symptom is a fitness note
such as `0.0 cannot be raised to a negative power`, from a `** 0.5` handed
another literal's value. A defect whose only symptom is a plausible number
needs a test.

The property pinned: `with_float_literals(code, float_literals(code))` is
`code` up to `ast.unparse` formatting, for every program -- compared as
`ast.dump`, which is formatting-blind and value-exact.

The last section pins the other input to the fit: the train/validation split
is a seeded shuffle of the labelled buffer, not a positional cut of it. A cut
`prefs[:n_train]` over a list that is iteration-major and rung-major would make
the validation set that selects the parameters always the newest, least-agreed
segments -- again a defect whose only symptom is a plausible number.
"""

from __future__ import annotations

import ast
import random
from types import SimpleNamespace

import numpy as np
import pytest

from bird.budget import Budget
from bird.components.alignment import (_split_preferences, alignment_bradley_terry,
                                       float_literals, with_float_literals)
from bird.config import load
from bird.context import Context
from bird.state import RunState
from bird.types import Candidate, Preference, Trajectory

#: Two programs on which the two orders disagree, then shapes that put a literal at
#: every depth `ast.walk` visits out of source order: nested calls, dict
#: values, a comprehension, a keyword default, unary minus, a nested `def`.
_PROGRAMS = {
    "probe": "def reward(s, a, s2):\n    return (1.0 + (2.0 * s[0])) + 3.0\n",
    "realistic": ("def reward(obs, action, s2):\n"
                  "    dist = obs[0]\n"
                  "    return -1.0 * dist + 0.25 * (obs[1] > 0.1) - 0.05 * action[0] ** 2\n"),
    "calls": ("def reward(s, a, s2):\n"
              "    return max(min(0.5 * s[0], 2.0), -1.5) + math.exp(-0.1 * abs(s[1]))\n"),
    "dict": ("def compute_reward(s, a, s2):\n"
             "    c = {'near': 1.0 / (1.0 + 2.0 * s[0]), 'fast': 3.0 * (s[1] ** 0.5)}\n"
             "    return sum(c.values()), c\n"),
    "comprehension": ("def reward(s, a, s2):\n"
                      "    return sum(0.5 * x ** 2.0 for x in a) - 0.01\n"),
    "default_and_nested_def": ("def compute_reward(state, action=None):\n"
                               "    def _f(v, default=0.0):\n"
                               "        return float(v) ** 0.5 if v > 0.0 else default\n"
                               "    return _f(state[0], -1.0) * 1.5\n"),
    "conditional": ("def reward(s, a, s2):\n"
                    "    bonus = 10.0 if s[0] > 0.9 else (0.5 if s[0] > 0.1 else 0.0)\n"
                    "    return bonus - 0.001 * float(np.sum(a ** 2))\n"),
}


def _same_program(a: str, b: str) -> bool:
    return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))


def _walk_order(code: str) -> list:
    """The contrasting order: `ast.walk`, breadth-first."""
    return [n.value for n in ast.walk(ast.parse(code))
            if isinstance(n, ast.Constant) and isinstance(n.value, float)]


@pytest.mark.parametrize("name", sorted(_PROGRAMS))
def test_writing_back_what_was_read_is_the_identity(name):
    code = _PROGRAMS[name]
    lits = float_literals(code)
    assert lits, f"{name}: the fixture exposes no float literal"
    back = with_float_literals(code, lits)
    assert back is not None
    assert _same_program(back, code), (
        f"{name}: float_literals -> with_float_literals changed the program\n"
        f"  read : {lits}\n  wrote: {back}")


def test_the_fixtures_are_where_breadth_first_and_depth_first_disagree():
    """If every fixture enumerated identically under `ast.walk`, the identity
    test above would pass under the breadth-first order too and pin nothing."""
    disagree = [n for n, code in _PROGRAMS.items() if _walk_order(code) != float_literals(code)]
    assert "probe" in disagree and "realistic" in disagree, disagree
    assert len(disagree) >= 5, f"only {disagree} distinguish the two orders"


@pytest.mark.parametrize("name", sorted(_PROGRAMS))
def test_the_kth_literal_read_is_the_kth_literal_written(name):
    """Per index, not only in aggregate: put a sentinel at position k and it
    must come back at position k, with every other literal untouched."""
    code = _PROGRAMS[name]
    lits = float_literals(code)
    sentinel = 12345.0
    assert sentinel not in lits
    for k in range(len(lits)):
        theta = list(lits)
        theta[k] = sentinel
        back = float_literals(with_float_literals(code, theta))
        assert back == theta, f"{name}: index {k} landed elsewhere: {back}"


# --- a property-style sweep --------------------------------------------------
#
# No `hypothesis` in the test extra, so a seeded generator. Expressions nest
# arbitrarily, mix literals with names and ints (which must NOT be enumerated),
# and are wrapped in the statement forms a reward program uses.

_OPS = ["+", "-", "*", "/"]


def _expr(rng: random.Random, depth: int) -> str:
    if depth <= 0 or rng.random() < 0.25:
        kind = rng.random()
        if kind < 0.5:
            return repr(round(rng.uniform(-3, 3), 3))
        if kind < 0.7:
            return f"s[{rng.randint(0, 5)}]"
        if kind < 0.85:
            return str(rng.randint(1, 9))  # an int: an index, not a parameter
        return f"a[{rng.randint(0, 2)}]"
    shape = rng.random()
    if shape < 0.5:
        return f"({_expr(rng, depth - 1)} {rng.choice(_OPS)} {_expr(rng, depth - 1)})"
    if shape < 0.65:
        return f"-{_expr(rng, depth - 1)}"
    if shape < 0.8:
        return f"max({_expr(rng, depth - 1)}, {_expr(rng, depth - 1)})"
    if shape < 0.9:
        return f"({_expr(rng, depth - 1)} if {_expr(rng, depth - 1)} > 0.0 else {_expr(rng, depth - 1)})"
    return f"abs({_expr(rng, depth - 1)}) ** {repr(round(rng.uniform(0.1, 2.5), 2))}"


def _program(rng: random.Random) -> str:
    n_stmts = rng.randint(1, 4)
    body = [f"    v{i} = {_expr(rng, rng.randint(1, 4))}" for i in range(n_stmts)]
    keys = ", ".join(f"'k{i}': v{i} * {repr(round(rng.uniform(0.5, 2.0), 2))}"
                     for i in range(n_stmts))
    body.append(f"    comps = {{{keys}}}")
    body.append("    return sum(comps.values()), comps")
    return "def compute_reward(s, a, s2):\n" + "\n".join(body) + "\n"


def test_round_trip_is_the_identity_on_generated_programs():
    rng = random.Random(20260911)
    programs = [_program(rng) for _ in range(300)]
    n_disagree = 0
    for code in programs:
        ast.parse(code)  # the generator must emit valid Python or this test measures nothing
        lits = float_literals(code)
        assert all(isinstance(v, float) for v in lits)
        back = with_float_literals(code, lits)
        assert back is not None and _same_program(back, code), (
            f"round trip changed the program\n--- wrote\n{back}\n--- read\n{code}")
        n_disagree += _walk_order(code) != lits
    # Vacuity guard: the sweep must include programs the two orders disagree
    # on, or it is a test of `ast.unparse` and not of the index space.
    assert n_disagree >= 100, f"only {n_disagree}/300 programs distinguish the orders"


def test_ints_and_bools_are_not_parameters():
    code = "def reward(s, a, s2):\n    return s[3] * 2 + (1.5 if True else 0) - range(10)[2]\n"
    assert float_literals(code) == [1.5]
    assert with_float_literals(code, [9.0]) is not None
    assert float_literals(with_float_literals(code, [9.0])) == [9.0]


# --- through the component ---------------------------------------------------


def _traj(x: float, n: int = 8) -> Trajectory:
    states = np.zeros((n, 2), dtype=float)
    states[:, 0] = x
    return Trajectory(states=states, actions=np.zeros((n - 1, 1), dtype=float),
                      rewards=[0.0] * (n - 1), length=n - 1)


def _prefs(n: int = 12):
    """Segments the candidate below ranks WRONGLY as written -- every label
    prefers the low-`s[0]` side while the program pays for a high one -- so a
    fit that flips the sign gains validation accuracy and `params_before` is
    recorded. `left_id`/`right_id` are provenance only."""
    out = []
    for i in range(n):
        hi, lo = _traj(1.0 + 0.1 * i), _traj(-1.0 - 0.1 * i)
        out.append(Preference(f"h{i}", f"l{i}", 0, left_traj=hi, right_traj=lo,
                              left_span=(0, 6), right_span=(0, 6)))
    return out


#: Nesting chosen so that `ast.walk` and the transformer disagree on the order:
#: `walk` reaches the outer `0.5` and `3.0` before the inner `1.0` and `2.0`.
_NESTED = ("def reward(s, a, s2):\n"
           "    return (1.0 * s[0] + (2.0 * s[1])) * 0.5 + 3.0\n")


def _ctx(iters: int = 30) -> Context:
    cfg = load("rstar", profile="tester",
               overrides={"generate.alignment.iterations": iters,
                          "generate.alignment.learning_rate": 0.5})
    return Context(cfg=cfg, budget=Budget(), rng=random.Random(0))


def test_params_before_is_float_literals_of_code_before():
    """`meta['alignment']` records the start point in the same index space as
    the end point, and the emitted code carries `params_after` at those indices."""
    assert _walk_order(_NESTED) != float_literals(_NESTED), "fixture does not nest"
    cand = Candidate(cand_id="c0001", iteration=1, reward_code=_NESTED)
    state = RunState(iteration=1)
    state.preferences = _prefs()
    (out,) = alignment_bradley_terry(_ctx(), state, [cand])
    info = out.meta["alignment"]
    assert "skipped" not in info, f"the fixture must produce a fit, got {info}"
    assert info["params_before"] == float_literals(info["code_before"])
    assert info["params_before"] == [1.0, 2.0, 0.5, 3.0]
    assert len(info["params_after"]) == len(info["params_before"])
    assert _same_program(out.reward_code,
                         with_float_literals(info["code_before"], info["params_after"]))
    # A fitted value that changed sign comes back through source as a unary
    # minus on a POSITIVE literal (`* -2.19` parses as `USub(2.19)`), so the
    # emitted program's literals match `params_after` in magnitude and count.
    assert [abs(v) for v in float_literals(out.reward_code)] == \
        [abs(v) for v in info["params_after"]]
    assert info["val_acc_after"] > info["val_acc_before"]


def test_the_before_accuracy_is_measured_on_the_program_as_written():
    """`val_acc_before` must be the accuracy of the model's program, not of a
    permutation of it. The program as written scores 0 on every pair (it
    prefers the side the labels reject); scrambled, `(1.0*s0 + 2.0*s1)*0.5 + 3.0`
    becomes `(0.5*s0 + 3.0*s1)*1.0 + 2.0` under the breadth-first order -- same sign, same
    accuracy -- so a program is used whose scramble FLIPS the sign instead."""
    code = ("def reward(s, a, s2):\n"
            "    return (2.0 * s[0] + (0.0 * s[1])) * -1.0\n")
    lits = float_literals(code)
    assert lits == [2.0, 0.0, 1.0]
    assert _walk_order(code) == [1.0, 2.0, 0.0], "fixture no longer distinguishes the orders"
    # As written the program pays -2*s0, which agrees with every label; the
    # breadth-first scramble `(1.0*s0 + 2.0*s1) * -0.0` pays nothing and ties every pair.
    cand = Candidate(cand_id="c0001", iteration=1, reward_code=code)
    state = RunState(iteration=1)
    state.preferences = _prefs()
    (out,) = alignment_bradley_terry(_ctx(iters=3), state, [cand])
    info = out.meta["alignment"]
    assert info["val_acc_before"] == 1.0, info


# --- the train/validation split ----------------------------------------------


def _rung_ordered_prefs():
    """The buffer as `bird.py` and the labeller build it: one batch per
    iteration, each batch strictest consensus first. `left_id` carries the
    provenance the assertions read; the trajectories are not executed here."""
    out = []
    for it in (1, 2):
        for rung, n in (("5of5", 6), ("4of5", 3), ("3of5", 1)):
            for k in range(n):
                out.append(Preference(f"i{it}-{rung}-{k}", "lo", 0, iteration=it,
                                      left_traj=_traj(1.0), right_traj=_traj(-1.0),
                                      left_span=(0, 6), right_span=(0, 6)))
    assert len(out) == 20
    return out


def _ids(prefs):
    return [p.left_id for p in prefs]


def test_the_split_is_a_seeded_permutation_not_a_positional_cut():
    """Same multiset, `train_fraction` honoured, the caller's list untouched,
    reproducible from the seed -- and NOT the positional cut, whose validation
    set on this buffer is iteration 2's tail and every one of its relaxed-rung
    labels."""
    prefs = _rung_ordered_prefs()
    before = _ids(prefs)
    train, valid = _split_preferences(random.Random(0), prefs, 0.7)
    assert _ids(prefs) == before, "the buffer itself must not be reordered"
    assert (len(train), len(valid)) == (14, 6)
    assert sorted(_ids(train) + _ids(valid)) == sorted(before)

    train2, valid2 = _split_preferences(random.Random(0), prefs, 0.7)
    assert (_ids(train2), _ids(valid2)) == (_ids(train), _ids(valid))

    positional_valid = _ids(prefs[14:])
    assert all(i.startswith("i2-") for i in positional_valid), "fixture lost its order"
    assert {i for i in positional_valid if "4of5" in i or "3of5" in i} == \
        {i for i in before if i.startswith("i2-") and "5of5" not in i}
    assert sorted(_ids(valid)) != sorted(positional_valid)
    # Measured with Random(0): validation reaches iteration 1 (3 of 6) and does
    # not swallow iteration 2's relaxed rungs whole -- the two things a
    # positional cut can never do on an accumulated buffer.
    assert any(p.iteration == 1 for p in valid)
    assert any(p.iteration == 2 and "5of5" not in p.left_id for p in train)


def test_a_single_pair_still_validates_on_itself():
    """`n_train` is at least 1 and an empty validation side falls back to the
    training side, the same edge semantics as a positional cut."""
    (only,) = _rung_ordered_prefs()[:1]
    train, valid = _split_preferences(random.Random(0), [only], 0.7)
    assert train == [only] and valid == [only]


def test_the_component_records_the_split_and_a_no_op_draws_nothing():
    """`meta['alignment']` says which split produced the numbers, on skipped
    candidates too. And the `iters <= 0` guard -- a direct-call path, since
    `_check_coherence` refuses that config -- returns BEFORE the shuffle, so a
    call that fits nothing leaves the run's RNG where it found it."""
    cand = Candidate(cand_id="c0001", iteration=1, reward_code=_NESTED)
    state = RunState(iteration=1)
    state.preferences = _prefs()
    (out,) = alignment_bradley_terry(_ctx(iters=1), state, [cand])
    info = out.meta["alignment"]
    assert info["split"] == "shuffled"
    assert (info["n_train"], info["n_valid"]) == (8, 4)  # round(12 * 0.7) = 8

    rng = random.Random(0)
    noop = SimpleNamespace(rng=rng, cfg={"generate.alignment.tunable": "float_literals",
                                         "generate.alignment.iterations": 0,
                                         "generate.alignment.learning_rate": 0.5,
                                         "generate.alignment.train_fraction": 0.7})
    kept = alignment_bradley_terry(noop, state, [Candidate(cand_id="c0002", iteration=1,
                                                           reward_code=_NESTED)])
    assert kept[0].reward_code == _NESTED and "alignment" not in kept[0].meta
    assert rng.getstate() == random.Random(0).getstate()


def test_a_program_compile_refuses_with_literals_replaced_is_named_not_scored():
    """A float literal inside a `match ... case 1.0:` pattern parses but cannot be
    parametrised: `compile` rejects a non-literal MatchValue, so `_Parametrised.code`
    is None and `bind()` returns None. Without a guard the fit would run SPSA over
    a callable that never evaluates and record `skipped: no_validation_gain,
    val_acc_before: 0.0` -- two statements about a program never scored. The
    artifact names the real reason and carries no accuracy."""
    import random
    import numpy as np
    from bird.budget import Budget
    from bird.components.alignment import alignment_bradley_terry
    from bird.config import load
    from bird.context import Context
    from bird.state import RunState
    from bird.types import Candidate, Preference, Trajectory

    def traj(x, n=8):
        st = np.zeros((n, 2)); st[:, 0] = x
        return Trajectory(states=st, actions=np.zeros((n - 1, 1)), rewards=[0.0] * (n - 1), length=n - 1)
    prefs = [Preference(f"h{i}", f"l{i}", 0, left_traj=traj(1.0 + 0.1 * i), right_traj=traj(-1.0 - 0.1 * i),
                        left_span=(0, 6), right_span=(0, 6)) for i in range(12)]
    cfg = load("rstar", profile="tester", overrides={"generate.alignment.iterations": 5,
                                                      "generate.alignment.learning_rate": 0.5})
    ctx = Context(cfg=cfg, budget=Budget(), rng=random.Random(0))
    code = ("def reward(s, a, s2):\n    match round(float(s[0]), 1):\n        case 1.0:\n"
            "            return -2.0 * s[0], {}\n        case _:\n            return 2.0 * s[0], {}\n")
    state = RunState(iteration=1); state.preferences = prefs
    (out,) = alignment_bradley_terry(ctx, state, [Candidate(cand_id="m", iteration=1, reward_code=code)])
    info = out.meta["alignment"]
    assert info["skipped"] == "not_parametrisable", info
    assert "val_acc_before" not in info and "val_acc_after" not in info, info
    assert out.reward_code == code, "the program is returned untouched"
