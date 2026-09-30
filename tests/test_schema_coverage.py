"""The schema is split across two files on purpose; these tests stop it drifting.

  configs/_default.yaml -- the set of keys, and their defaults
  bird/schema.py        -- the allowed values for those keys

A single source of truth would be ideal. Two files is a readability
compromise, and it is only safe because the agreement is asserted here in both
directions.
"""

import yaml

from bird import registry
from bird.config import CONFIG_ROOT, flatten
from bird.schema import META_KEYS, SCHEMA


def _default_flat():
    data = yaml.safe_load((CONFIG_ROOT / "_default.yaml").read_text())
    return flatten(data)


def test_every_default_key_is_in_the_schema():
    extra = sorted(set(_default_flat()) - set(SCHEMA))
    assert not extra, f"_default.yaml declares keys the schema does not know: {extra}"


def test_every_schema_key_has_a_default():
    missing = sorted(set(SCHEMA) - set(_default_flat()) - META_KEYS)
    assert not missing, f"schema declares keys _default.yaml does not: {missing}"


def test_registry_kinds_match_schema_families():
    """Every kind named by a schema Field must be a real registry family."""
    registry.load_all()
    bad = sorted({f.kind for f in SCHEMA.values() if f.kind and f.kind not in registry.KINDS})
    bad += sorted({f.item_kind for f in SCHEMA.values()
                   if f.item_kind and f.item_kind not in registry.KINDS})
    assert not bad, f"schema references unknown registry kinds: {bad}"


def test_every_kind_backed_key_has_implementations():
    """A config value with no implementation behind it is a lie.

    This is the structural guarantee described in bird/registry.py: you cannot
    offer a config value the code cannot honour.
    """
    registry.load_all()
    empty = []
    for key, field in SCHEMA.items():
        for kind in (field.kind, field.item_kind):
            if kind and not registry.names(kind):
                empty.append(f"{key} -> kind {kind!r}")
    assert not empty, f"schema keys whose registry family is empty: {sorted(set(empty))}"


def test_defaults_validate_against_their_own_enums():
    """The defaults must themselves be legal values."""
    from bird.config import Config, validate

    validate(Config(yaml.safe_load((CONFIG_ROOT / "_default.yaml").read_text()),
                    source="_default.yaml"))


def test_every_schema_key_declares_a_section():
    """Each key belongs to a design-space section (§0-§6), so config_diff can group by it."""
    ok = {"§0", "§1", "§2", "§3", "§4", "§5", "§6"}
    bad = sorted(k for k, f in SCHEMA.items() if f.section not in ok)
    assert not bad, f"keys with an unrecognised section: {bad}"
