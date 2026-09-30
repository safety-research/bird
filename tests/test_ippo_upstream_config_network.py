"""`upstream_ippo_config` carries upstream's `network` block, or the learner dies.

Upstream's `MultiActorCritic` (`assistax/baselines/IPPO/ippo_ff_nps.py`) reads
`config["network"]["activation"]`, `["actor_hidden_dim"]` and
`["critic_hidden_dim"]`, values that hydra supplies from
`config/network/ff_nps.yaml`. Without that block in `UPSTREAM_IPPO_CONFIG`,
the first training call dies with `KeyError: 'network'`. These tests hold the
block in the config and, where the vendored upstream is checked out, hold it
equal to upstream's own file.
"""
from __future__ import annotations

import pathlib

import pytest

from bird.components import _ippo_backend as B

_UPSTREAM_KEYS_READ = ("activation", "actor_hidden_dim", "critic_hidden_dim")
_FF_NPS_YAML = (pathlib.Path(__file__).resolve().parents[1] / "refs" / "code" / "assistax"
                / "assistax" / "baselines" / "IPPO" / "config" / "network" / "ff_nps.yaml")


def _cfg():
    return B.upstream_ippo_config(env_name="scratchitch", env_kwargs={},
                                  total_timesteps=64 * 1024, zoo_path="/nonexistent/zoo")


def test_the_config_carries_every_network_key_upstream_reads():
    cfg = _cfg()
    assert "network" in cfg, "no `network` block: MultiActorCritic.__init__ raises KeyError('network')"
    for k in _UPSTREAM_KEYS_READ:
        assert k in cfg["network"], f"network block lacks {k!r}, which ippo_ff_nps.py reads"
    assert cfg["network"]["activation"] in ("relu", "tanh")
    assert int(cfg["network"]["actor_hidden_dim"]) > 0
    assert int(cfg["network"]["critic_hidden_dim"]) > 0


def test_the_network_block_is_upstream_ff_nps_verbatim():
    """Values, not just keys: this tier's claim is upstream's learner on upstream's settings."""
    net = _cfg()["network"]
    assert net["name"] == "ff_nps"
    assert net["recurrent"] is False and net["agent_param_sharing"] is False
    assert net["actor_hidden_dim"] == 128 and net["critic_hidden_dim"] == 128
    assert net["activation"] == "relu"


def test_the_network_block_equals_the_vendored_yaml_when_present():
    if not _FF_NPS_YAML.is_file():
        pytest.skip("refs/code/assistax not present; run scripts/fetch_refs.sh")
    import yaml  # pyyaml is a hard dependency; an importorskip here would be a silent skip
    up = yaml.safe_load(_FF_NPS_YAML.read_text())
    net = _cfg()["network"]
    assert set(up) <= set(net), f"upstream keys missing from the block: {set(up) - set(net)}"
    for k, v in up.items():
        assert net[k] == v, f"{k}: block has {net[k]!r}, upstream file has {v!r}"


def test_the_block_is_not_overwritten_by_the_per_run_fields():
    cfg = _cfg()
    for k in ("ENV_NAME", "ENV_KWARGS", "TOTAL_TIMESTEPS", "ZOO_PATH"):
        assert k in cfg
    assert cfg["network"] == B.UPSTREAM_IPPO_CONFIG["network"]
