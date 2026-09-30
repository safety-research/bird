"""A task spec must not name one observation slot after a quantity that lives in another.

`tasks/*/shared_spec.yaml` describes the observation twice, and both halves are
hand-authored: `state_surface.fields` groups the slots into named, described
blocks (from the builder's `extra_groups`), and `state_surface.helpers` names
individual slots by expression (from `extra_helpers`). Nothing derives either
from the adapter, and nothing else checks them against each other, which is
exactly how they drift.

THE FAILURE THIS GUARDS. A `fields` entry `s.goal` declared at slot 36 while
the helpers put `goal` at slot 38 -- so the citable half of the spec names one
slot after a quantity that lives in another, and a reward written against the
declared name reads the wrong number.

WHY THIS COMPARISON AND NOT FIELD-NAME-VERSUS-CODE. The obvious test -- assert
the declared field names equal the attribute names `_extra_state` writes -- does
not work. Adapters name their
attributes as short handles (`d`, `g_x`) while the spec names them for a
reader (`distance_to_goal`, `goal_x`), so that comparison reports mostly
synonyms. A test whose failures are almost all false is a test
people learn to skip.

The signature that is not a synonym is a name that belongs to a DIFFERENT slot:
`s.goal` declared at one slot while the helpers put `goal` at another. Two
different quantities, both real, both in the same tail, one wearing the other's
slot. A failure here means a genuine swap rather than a wording difference.

WHAT IT DOES NOT CATCH, stated so the next reader does not over-trust it: two
slots whose names are both wrong in the same way, a group field whose prose
describes its members in the wrong order without renaming anything, and
anything about a single-slot name that no helper mentions at all. It is one
cheap invariant, not a proof that the surface is right.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

try:                                     # ~4x faster over the whole catalogue, and the
    from yaml import CSafeLoader as _Loader   # difference decides whether this test can
except ImportError:                      # live outside the `slow` mark at all.
    _Loader = yaml.SafeLoader

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
SPEC_PATHS = sorted(TASKS.glob("*/shared_spec.yaml"))

#: A helper that names exactly one slot, e.g. `expression: s[157]`. Helpers over
#: a range (`s[0:3]`) or an arithmetic expression name no single slot and are
#: skipped: this test is about which slot holds which quantity.
_ONE_SLOT = re.compile(r"s\[(\d+)\]\Z")


def _violations(path):
    """Every swapped-name finding in one spec, so a run names them all at once."""
    surface = (yaml.load(path.read_text(), Loader=_Loader) or {}).get("state_surface") or {}

    slot_of_helper = {}
    for helper in surface.get("helpers") or []:
        m = _ONE_SLOT.match(str(helper.get("expression", "")).strip())
        if m:
            slot_of_helper[helper.get("name", "")] = int(m.group(1))
    helper_at = {slot: name for name, slot in slot_of_helper.items()}

    out = []
    for field in surface.get("fields") or []:
        raw = str(field.get("slice", "")).strip().strip("'\"")
        if ":" not in raw:
            continue
        lo, hi = (int(x) for x in raw.split(":"))
        if hi - lo != 1:
            continue                      # only single-slot fields are unambiguous
        name = field.get("name", "").removeprefix("s.")
        elsewhere = slot_of_helper.get(name)
        if elsewhere is not None and elsewhere != lo:
            out.append(
                f"{path.parent.name}: field 's.{name}' declares slot {lo}, but the "
                f"helpers put '{name}' at slot {elsewhere}; slot {lo}'s helper is "
                f"'{helper_at.get(lo, '<none>')}'")
    return out


def test_no_field_is_named_for_another_slots_quantity():
    """One test over every spec, reporting every violation rather than the first.

    Parametrising per spec would be ~100 cases for one invariant for no extra
    signal; and the first failure is rarely the only one when a surface has
    drifted.
    """
    assert SPEC_PATHS, f"no shared_spec.yaml under {TASKS} -- a glob that " \
                       "silently matched nothing would make this vacuous"
    found = [v for path in SPEC_PATHS for v in _violations(path)]
    assert not found, (
        "a task spec names one slot after a quantity its helpers place in another:\n  "
        + "\n  ".join(found)
        + "\nCheck which half is wrong against the adapter's _extra_state write "
          "order before editing either -- the code is not automatically the "
          "guilty party."
    )
