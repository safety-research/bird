"""`_partner_split` hands the trainer an object carrying EVERY field the seed row reads.

A view that carries fewer fields than the seed-row writer reads fails at the
first training with `'_PartnerSplitView' object has no attribute 'seed'`, and a
docstring saying only `.train` is read does not stop the code reading four
fields. These tests derive the read set from the SOURCE, so the claim cannot
drift from the code it describes.
"""
from __future__ import annotations

import pathlib
import re
import types

from bird.components import _ippo_train as T

_SRC = pathlib.Path(T.__file__).read_text()


def _fields_read_off_split() -> set:
    return set(re.findall(r"\bsplit\.([a-zA-Z_][a-zA-Z0-9_]*)", _SRC))


class _Cfg(dict):
    def get(self, k, d=None):  # the resolved config's .get shape
        return dict.get(self, k, d)


def test_the_source_reads_at_least_the_four_known_fields():
    got = _fields_read_off_split()
    assert {"train", "seed", "heldout", "index_sha256"} <= got, got


def test_the_view_carries_every_field_the_trainer_reads():
    view = T._PartnerSplitView({"IPPO": {"human": ["u1", "u2"]}}, seed=3)
    for name in _fields_read_off_split():
        assert hasattr(view, name), f"the view lacks `{name}`, which _ippo_train reads off `split`"
    assert view.seed == 3 and view.train == {"IPPO": {"human": ["u1", "u2"]}}
    assert view.heldout == [] and view.index_sha256 == ""   # not derived here, and said so


def test_the_adapter_s_full_split_is_returned_untouched():
    real = types.SimpleNamespace(seed=7, train={"IPPO": {"human": ["a"]}}, heldout=["b"],
                                 index_sha256="deadbeef", pool_size=2)
    env = types.SimpleNamespace(zoo_path="/z", zoo_split=real, zoo_partners=real.train, task="scratchitch")
    assert T._partner_split(env, _Cfg(seed=0)) is real


def test_a_hand_set_train_half_gets_the_view_with_the_run_seed():
    env = types.SimpleNamespace(zoo_path="/z", zoo_split=None,
                                zoo_partners={"MASAC": {"human": ["x"]}}, task="scratchitch")
    split = T._partner_split(env, _Cfg(seed=11))
    assert isinstance(split, T._PartnerSplitView)
    assert split.seed == 11 and split.train == {"MASAC": {"human": ["x"]}}
    for name in _fields_read_off_split():
        assert hasattr(split, name)


def test_no_zoo_means_none():
    env = types.SimpleNamespace(zoo_path=None, zoo_split=None, zoo_partners=None, task="scratchitch")
    assert T._partner_split(env, _Cfg(seed=0)) is None
