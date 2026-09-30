"""A zero-arity reward must never become a silent constant reward.

`def compute_reward():` -- the mock's `wrong_signature` archetype, and the
shape a model produces when it forgets the signature -- returns a number while
reading nothing. If the harness ever CALLED it correctly, it would score every
transition identically and the run would look healthy: a constant reward is a
plausible number, not a crash. That is the failure this file exists to keep
impossible, and the repo screens it at two layers that must both keep working.

**Verify on: refused before execution.** `signature_parse` reads the AST and
rejects it (§2 Validity), so it is `failure_kind: "invalid"` and costs no
training slot.

**Verify off: it must still be LOUD.** `CompiledReward._make_binder` ends
`return plan or [0]`, which hands one argument to a function that declared
nowhere to put it -- so the call raises instead of succeeding. That looks like
a bug and is deliberate: binding it "correctly" with zero arguments is the
silent-constant-reward case above. The comment on that line says so; this file
is what makes the claim checkable.

A config that resolves `verify.enabled: false` (`roska` under the tester
profile does) has only the second layer between it and a constant reward, and
any prompt edit that re-rolls the mock's draws can hand it the archetype.
"""
from __future__ import annotations

import pytest

from bird.components.training import CompiledReward
from bird.components.verification import static_signature_parse
from bird.types import Candidate

#: (source, accepted_by_signature_parse). `*args` is accepted on purpose: it
#: HAS a positional slot, so it can receive the state.
_PROGRAMS = [
    ("def compute_reward():\n    return -1.0\n", False),
    ("def compute_reward(**kw):\n    return -1.0\n", False),
    ("def compute_reward(*, state=None):\n    return -1.0\n", False),
    ("def compute_reward(*args):\n    return 0.0, {}\n", True),
    ("def compute_reward(state):\n    return 0.0, {}\n", True),
    ("def compute_reward(state, action=None, next_state=None):\n    return 0.0, {}\n", True),
]


def _compile(src):
    ns: dict = {}
    exec(compile(src, "<program>", "exec"), ns, ns)  # noqa: S102 - the subject
    return ns["compute_reward"]


@pytest.mark.parametrize("src,accepted", _PROGRAMS)
def test_signature_parse_is_the_verify_on_layer(src, accepted):
    """Driven, not restated: the check itself decides, on the real Candidate."""
    verdict = static_signature_parse(None, Candidate(cand_id="c", iteration=0, reward_code=src))
    assert (verdict == "") is accepted, f"{verdict!r} for:\n{src}"
    if not accepted:
        assert "takes no state argument" in verdict, verdict


@pytest.mark.parametrize("src", [p[0] for p in _PROGRAMS if not p[1]])
def test_verify_off_a_zero_arity_reward_raises_rather_than_scoring(src):
    """THE INVARIANT, and the one worth the file: not "it raises a TypeError"
    but "it does not return a number".

    Asserting the TypeError text would pin the mechanism -- the fabricated
    argument -- and go red on any future change that refused the program
    earlier and more accurately, which would be an improvement, not a
    regression. What must never change is that no value comes back.
    """
    r = CompiledReward(_compile(src), "compute_reward")
    with pytest.raises(Exception):  # noqa: B017 - ANY failure is the contract
        r(1.0, 2.0, 3.0)


@pytest.mark.parametrize("src", [p[0] for p in _PROGRAMS if p[1]])
def test_a_reward_with_a_positional_slot_still_scores(src):
    """The other half, so the guard above cannot be satisfied by refusing
    everything -- which is the way a check like this usually rots."""
    r = CompiledReward(_compile(src), "compute_reward")
    total, _components = r(2.0, 0.0, 0.0)
    assert isinstance(total, float)


def test_var_positional_receives_the_state_rather_than_nothing():
    """`*args` is the case the `or [0]` fallback exists to serve, and it is the
    reason the line cannot simply be deleted: the plan is empty here too, but
    this program CAN accept the state and must be given it."""
    seen = []

    def compute_reward(*args):
        seen.append(args)
        return 0.0, {}

    CompiledReward(compute_reward, "compute_reward")(7.0, 8.0, 9.0)
    assert seen and seen[0] == (7.0,), seen
