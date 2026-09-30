"""A name an `env_spec` RENDERS as an identifier must be one a candidate can use.

The failure: an attribute-typed state rendering against an ndarray `state`
makes every candidate raise on attribute access --

    AttributeError: 'numpy.ndarray' object has no attribute 'x_velocity'

-- because the prompt shows the state as a class with typed attributes and
callable helpers while `## OUTPUT CONTRACT` pins
`compute_reward(state, action=None, next_state=None)` whose `state` is an array,
and `training._reward_namespace` binds only `{np, numpy, math}`
(`verification._restricted_globals` agrees; the training site passes no
`bindings`). The model obeyed every instruction it was given. A reward whose
signature disagrees with the prompt's (the CARD/T2R case) has the same shape.

WHY THIS IS KEYED ON (ADAPTER CLASS, MEMBER) AND NOT ON THE CONFIG. A test keyed
on the config and scoped by member NAME -- excluding `full_source` as "a span of
source, not an enumerated offer of identifiers" -- is wrong on both counts, and
together they hide the method that actually fails:

  * Two methods can fail through different members. `revolve` resolves
    `pythonic_class_abstraction`; **`eureka` resolves `full_source`** -- and
    `GymMujoco._render_full_source` PREPENDS the API stub, so `full_source`
    hands the model `@dataclass class State: x_position: float ...` too.
  * `eureka` at its DEFAULT env (`toy_reacher`) does not offend, because
    `full_source` there is `inspect.getsource(ToyReacher)` with no stub in it.
    It offends at `gym_humanoid_run`. A config-keyed test at default envs
    therefore cannot see this class of defect at all -- the default env is not
    the env a paper run uses.

So the invariant is a property of the RENDERING: for every adapter and every
`env_spec` member, a name the rendered text presents in an identifier-DEFINITION
position (`    name: <type>` or `def name(`) must be one the reward execution
namespace binds. Whether a particular config rescues itself with a
`generate.postprocess.symbol_mapping` is a separate, config-level question and
belongs to the load-time refusal, not here.

SCOPE, stated rather than left to be inferred. Members come from
`registry.names("env_spec")`, so a new member is covered the moment it is
registered rather than when somebody remembers this file. Adapters come from
`test_env_spec_leak._buildable()` -- reused rather than re-derived, because its
docstring already carries the reasoning for the heavy-tier exclusions and for
using a loop instead of `pytest.skip` (a skip would hide an unbuilt adapter).
One representative per adapter class rather than one per registered id, because
the rendering is a property of the class and a construction per id would price
coverage that does not vary. That reduction is the one real limitation of this file and is why it is
written down rather than left to be counted.

THE INSTRUMENT IS CHECKED BEFORE ANYTHING IS CONCLUDED FROM IT. The names are
read structurally off the adapter\'s own `_state_fields` / `_action_fields` /
`_helpers` -- the tables `_render_class_abstraction` and `_render_api_stub`
build their text from -- and then matched against the RENDERED text, so neither
half can drift alone. `test_the_detector_fires_on_a_planted_name` shows it can
report non-zero at all; `test_a_render_failure_is_never_read_as_no_offence`
exists because a sweep that swallows render errors in a bare `except`
confidently reports zero offences across every adapter.
"""

import functools
import re
import types

from test_env_spec_leak import _buildable

from bird import registry
from bird.components.training import _reward_namespace
from bird.config import load

#: A config is needed only as a RENDERING CONTEXT -- the stubs read
#: `problem.task_description`. It does not select the adapter or the member here,
#: and the field tables are identical with or without it (measured), so any
#: loadable config serves and the choice is not a hidden parameter.
_RENDER_CONTEXT = "revolve"

#: Measured: 22 (adapter class, member) pairs over 9 adapter classes, across
#: the 6 registered `env_spec` members, where the rendering offers a name
#: nothing binds. Value is the number of such names.
#:
#: `natural_language_only` and `none` appear NOWHERE in this table -- zero
#: offences across every class measured -- because they present a name as a label
#: beside its index (`- s[0] x_position: doc`) rather than as an identifier to
#: write. That is the shape a faithful rendering has, and it is why REvolve\'s
#: own release prompt (`refs/code/Revolve/prompts/env_input`, the Gymnasium
#: docstring index table) produces candidates that index `observation[22]`
#: rather than reaching for attributes.
#:
#: This table must SHRINK. Do NOT add a row to make a new
#: adapter or member pass -- a new offender is the thing this file exists to
#: stop, and the assertion below fails in BOTH directions so a row cannot
#: outlive its defect either.
KNOWN_OFFERS_UNBOUND = {
    ("Acrobot", "full_source"): 1,
    ("Acrobot", "pythonic_class_abstraction"): 10,
    ("Acrobot", "state_action_api_stub"): 7,
    #: Exemplar `assistax_armmanipulation` (the first buildable id in sorted
    #: registry order); the count is a property of the exemplar, see SCOPE.
    ("Assistax", "full_source"): 96,
    ("Assistax", "pythonic_class_abstraction"): 108,
    ("Assistax", "state_action_api_stub"): 96,
    ("GymMujoco", "full_source"): 25,
    ("GymMujoco", "pythonic_class_abstraction"): 28,
    ("GymMujoco", "state_action_api_stub"): 24,
    ("MetaWorld", "full_source"): 2,
    ("MetaWorld", "pythonic_class_abstraction"): 50,
    ("MetaWorld", "state_action_api_stub"): 43,
    ("Pendulum", "pythonic_class_abstraction"): 7,
    ("Pendulum", "state_action_api_stub"): 4,
    ("PendulumDiscrete", "pythonic_class_abstraction"): 7,
    ("PendulumDiscrete", "state_action_api_stub"): 4,
    ("ToyGridworld", "pythonic_class_abstraction"): 9,
    ("ToyGridworld", "state_action_api_stub"): 6,
    ("ToyHungryThirsty", "pythonic_class_abstraction"): 8,
    ("ToyHungryThirsty", "state_action_api_stub"): 5,
    ("ToyReacher", "pythonic_class_abstraction"): 11,
    ("ToyReacher", "state_action_api_stub"): 8,
}


def _identifier_definitions(text, names):
    """Names the rendered text presents in a position that offers them for use."""
    return [n for n in names
            if re.search(rf"^\s*{re.escape(n)}\s*:\s*\w", text, re.M)
            or re.search(rf"^\s*def\s+{re.escape(n)}\s*\(", text, re.M)]


def _names_of(env):
    return ([f for f, _ in getattr(env, "_state_fields", []) or []]
            + [f for f, _ in getattr(env, "_action_fields", []) or []]
            + [s.split("(")[0].strip() for s, _ in getattr(env, "_helpers", []) or []])


def _representatives():
    """One constructed adapter per adapter CLASS. See SCOPE in the module docstring."""
    reps = {}
    for name, env in _buildable():
        reps.setdefault(type(env).__name__, (name, env))
    return reps


@functools.lru_cache(maxsize=1)
def _sweep():
    """(offences, render_failures, classes) over every class x every member."""
    registry.load_all()
    cfg = load(_RENDER_CONTEXT)
    reps = _representatives()
    offences, failures = {}, []
    namespace = set(_reward_namespace("numpy"))
    for cls, (env_id, env) in sorted(reps.items()):
        names = _names_of(env)
        for member in sorted(registry.names("env_spec")):
            ctx = types.SimpleNamespace(cfg=cfg, env=env, rundir=None)
            try:
                text = registry.get("env_spec", member)(ctx, None)
            except Exception as exc:                      # noqa: BLE001 -- reported
                failures.append((cls, env_id, member, f"{type(exc).__name__}: {exc}"))
                continue
            unbound = sorted(n for n in set(_identifier_definitions(text, names))
                             if n not in namespace)
            if unbound:
                offences[(cls, member)] = (len(unbound), env_id, unbound[:6])
    # frozen: `lru_cache` hands the SAME objects to every caller, and a test
    # that mutated them would change what the next one measures.
    return (tuple(sorted(offences.items())), tuple(failures), tuple(sorted(reps)))


# -- the instrument, before anything is concluded from it --------------------

def test_the_detector_fires_on_a_planted_name():
    """A detector reporting zero is indistinguishable from a broken one."""
    text = ("class Env:\n"
            "    planted_attribute: float   # s[0]\n"
            "    def planted_helper(s): ...\n"
            "    not_offered_anywhere = 3\n")
    names = ["planted_attribute", "planted_helper", "not_offered_anywhere", "absent"]
    found = _identifier_definitions(text, names)
    assert found == ["planted_attribute", "planted_helper"], found


def test_a_render_failure_is_never_read_as_no_offence():
    """A sweep that swallows render errors reports zero.

    `_sweep` collects them instead, and this is the assertion that makes the
    collection load-bearing: a member that cannot render is an unchecked member,
    which is the same silence the whole file is about.
    """
    _offences, failures, reps = _sweep()
    assert reps, "no adapter class was constructed; the sweep checked nothing"
    assert not failures, (
        "these (class, member) pairs could not be rendered, so nothing was checked "
        f"for them: {failures}")


# -- the invariant ------------------------------------------------------------

def test_no_env_spec_offers_a_name_the_reward_namespace_does_not_bind():
    offences, _failures, reps = _sweep()
    offences = dict(offences)
    seen_classes = set(reps)

    measured = {k: v[0] for k, v in offences.items()}
    declared_here = {k: v for k, v in KNOWN_OFFERS_UNBOUND.items() if k[0] in seen_classes}

    new = sorted(set(measured) - set(KNOWN_OFFERS_UNBOUND))
    assert not new, (
        "these (adapter class, env_spec member) pairs render a name nothing binds and "
        "are not declared:\n"
        + "\n".join(f"  {c} x {m}: {offences[(c, m)][0]} name(s) via {offences[(c, m)][1]} "
                    f"-> {offences[(c, m)][2]}" for c, m in new)
        + "\nBind them, stop rendering them, or render them as a label beside an index "
          "the way `natural_language_only` does. Do NOT add a row to KNOWN_OFFERS_UNBOUND.")

    fixed = sorted(set(declared_here) - set(measured))
    assert not fixed, (
        f"declared but no longer offending: {fixed}. Delete the rows -- a gap list that "
        f"outlives its defect teaches the next reader that a solved problem is open.")

    # Safe to index `measured[k]` over `declared_here` ONLY because the `fixed`
    # assertion above has already established declared_here <= measured. Keep
    # them in this order; swapping them turns a stale row into a KeyError whose
    # traceback says nothing about the gap list.
    moved = {k: (declared_here[k], measured[k]) for k in declared_here
             if measured[k] != declared_here[k]}
    assert not moved, (
        f"declared count != measured (declared, measured): {moved}. Down is progress and "
        f"the row should follow it; up means something new is being offered unbound.")
