"""`problem.env_id` -> the BENCHMARK SUITE its environment belongs to.

**This file is AUTHORITATIVE for that taxonomy.** `bird/config.py`'s GPU rules
and `bird/xla_env.py` key on it.

Data, deliberately: `env_suite` is which BENCHMARK a run ran on, a claim about
what its numbers are comparable against. The label below is `mt10` and not
`metaworld` because MT10 is the protocol published numbers are quoted against.

PREFIX-OR-EXACT, ORDERED, FIRST MATCH WINS -- a table rather than a chain of
`if`s, for three reasons that are about extension rather than style:

  * A PREFIX IS ALREADY THIS REPO'S NAMING RULE for a new family.
    `bird/envs/metaworld.py` states it outright -- "the prefix names the
    BENCHMARK SUITE ... `metaworld_` would name the package; MT10 is the
    protocol". So a new benchmark family is one row. That is the whole
    extension story and it needs no code.
  * Three ids predate that rule and cannot be prefixed. `pendulum`,
    `pendulum_discrete` and `acrobot` share no prefix with each other. An
    exact set is the escape hatch for exactly those, and keeping it a set
    rather than a fourth prefix is what stops `pendulum` swallowing any
    later id that merely starts with the same word.
  * A REGEX table was rejected: an unanchored pattern matches in the middle and
    does it silently, and a set of regexes cannot be checked for mutual overlap
    cheaply. Prefixes and exact sets can, and the sync test does.

STDLIB ONLY, and no import of `bird.registry` or numpy. `bird/config.py` reads
it; `config -> registry -> components -> config` is a real cycle in this repo,
so the taxonomy stays a leaf module anything can import.

THE ORDER IS A RUNG LADDER rather than an alphabet: toy world models, then
numpy classic control, then a real simulator, then a benchmark carrying its own
success check. The LABELS are upstream's
own words for the same reason the env ids are -- `classic_control` is
gymnasium's directory name, `gym_mujoco` names gymnasium's `mujoco` family, and
`mt10` is the protocol string the benchmark is keyed by.
"""

from __future__ import annotations

from typing import Any, Tuple

#: (label, id prefixes, exact ids). See the module docstring for why it is a
#: table and why the order is the order.
ENV_SUITES: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    # (label,           id prefixes,   exact ids)
    ("toy",             ("toy_",),     ()),
    ("classic_control", (),            ("pendulum", "pendulum_discrete", "acrobot")),
    # The spec-backed gymnasium/MuJoCo tasks (`bird/envs/gym_mujoco.py`).
    ("gym_mujoco",      ("gym_",),     ()),
    ("mt10",            ("mt10_",),    ()),
    # The other forty Meta-World MT50 tasks. A separate suite from
    # `mt10` even though the simulator and the adapter are the same, for the reason
    # the prefix exists: MT10 is the protocol published numbers are quoted against,
    # and a grouping that pooled `mt10_reach-v3` with `mt50_hammer-v3` would invite
    # a comparison against a table that never contained the second.
    ("mt50",            ("mt50_",),    ()),
    # HumanoidBench, and it is its OWN suite rather than a `gym_mujoco` entry
    # even though it is a MuJoCo env. The tier cannot share an interpreter with the
    # rest of the repo -- HumanoidBench needs `mujoco==3.1.6`, metaworld pins
    # `3.3.0` -- so it runs in its own venv, and a
    # grouping that put it with the `gym_*` tasks would invite a comparison
    # whose two halves were produced by different simulator versions.
    # Prefixed rather than exact because the family is named for the robot and
    # `bird/envs/humanoid_hand.py` is a general adapter rather than one task: upstream
    # ships many tasks per robot, and an exact-id row would place the first and
    # drop every sibling into `other` -- the failure the `other` bucket exists
    # to make visible rather than to absorb quietly.
    #
    # SIX PREFIXES, NOT ONE: `h1_` alone is not enough. A HumanoidBench env id
    # is its gym id with `-v0` dropped and `-` turned into `_`
    # (`bird/tasks.py::_BIRD_ID_RULES`), and the gym id starts with the ROBOT --
    # so `h1hand_powerlift` and `h1strong_highbar_hard` do not begin with `h1_`.
    # The six are exactly the keys of `humanoid_bench.env.ROBOTS` with an
    # underscore appended, which is a rule a reader can check against upstream
    # rather than a hand-grown list. `g1_` is included for the same reason the
    # unregistered h1 variants are: a row that places only what is registered
    # today is a row that silently stops placing things.
    ("humanoid_bench",  ("h1_", "h1hand_", "h1simplehand_", "h1strong_",
                         "h1touch_", "g1_"), ()),
    # Assistax (assistive-autonomy/assistax): a Panda arm helping a passive human on a
    # wheelchair or a bed, five tasks, driven through plain `mujoco` from the vendored
    # upstream scenes. Its own suite: one robot-human setting with its own reward
    # structure, which nobody would pool with `gym_half_cheetah`.
    ("assistax",        ("assistax_",),  ()),
    # THE JAX TIER: envs stepped by JAX rather than CPU MuJoCo -- upstream
    # Assistax's own JAX implementation (`upstream_assistax_*`,
    # `bird/envs/upstream_assistax.py`) and the offline `jax_toy`. A different
    # solver is a different experiment, so it carries a different env id rather
    # than a flag on the same one, and a grouping that pooled `assistax_feeding`
    # with `upstream_assistax_feeding` would invite the comparison the prefix
    # exists to prevent.
    #
    # TWO PREFIXES, NOT ONE, and on THIS suite that is not only a labelling
    # question: `bird/config.py`'s `requires_cuda` refusal and
    # `bird/xla_env.py`'s determinism flags are both entered through
    # `env_suite(...) == "jax"`, so the set of families they apply to is
    # exactly this tuple. `upstream_assistax_*` is cuda-only; without its prefix
    # here it would resolve clean on cpu with nothing red.
    ("jax",             ("jax_", "upstream_assistax_"), ()),
    # A new family is ONE ROW HERE and nothing else, e.g.:
    #   ("mysuite",     ("mysuite_",),   ()),
)

#: An env id no row claims. A BUCKET rather than an error: an id this table has
#: never heard of still gets a label, one that says the table is behind.
ENV_SUITE_OTHER = "other"

#: An empty or missing `problem.env_id`, which is a different fact from an id no
#: row claims. Folding it into `other` would put "this table needs a new row"
#: and "there is no env id to read" in one bucket, and only the first is a to-do
#: for this file.
ENV_SUITE_UNKNOWN = "?"

def env_suite(env_id: Any) -> str:
    """Which benchmark suite `problem.env_id` belongs to.

    Never raises and never returns empty: every id lands in exactly one bucket,
    because a label that drops an id is worse than one that admits it cannot
    place it.
    """
    e = str(env_id or "").strip()
    if not e or e == "?":
        return ENV_SUITE_UNKNOWN
    for label, prefixes, exact in ENV_SUITES:
        if e in exact or any(e.startswith(p) for p in prefixes):
            return label
    return ENV_SUITE_OTHER
