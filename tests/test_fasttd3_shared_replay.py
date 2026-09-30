"""The `shared_population` replay half, on `train.backend: fasttd3`.

WHY THIS FILE EXISTS. A backend that reads the slice hand-off's INDEX (the seed
salt) and ignores its replay half has to be refused by `_check_coherence` for
`interaction_cfg.shared_buffer: true`, rather than let a declared pool not
execute. `fasttd3_backend` honours the replay half, so that refusal is lifted
for it -- which makes the tests below the only thing standing between a config
that says `shared_buffer: true` and an arm that trains on a buffer it never
pooled: a LaRes cell without its shared buffer and with a green config, which
nothing downstream can detect.

WHAT IS AND IS NOT COVERED HERE, stated because the gap matters. These run on
CPU and cover the CONTRACT: the lane layout, the refusals, the counts, the
store hand-off. They do NOT cover the device path at scale -- a `.permute` on
a cuda tensor, 128 lanes, a 1M-step budget -- which needs a GPU run. A green
run of this file is evidence about the semantics, not about the hardware, and
the two should not be confused when a GPU run's number is read.

THE ONE THAT IS NOT A UNIT TEST is `test_export_then_prefill_is_the_identity`.
The prefill's whole correctness rests on being the exact inverse of
`_fasttd3_export_replay`'s time-major flatten, and an assertion on either one
alone would pass with both wrong in the same direction -- the failure shape in
which a replacement instrument reproduces the original's numbers by inheriting
its defect. Composing them is the only check that cannot.
"""

from __future__ import annotations

import numpy as np
import pytest

from bird.config import load

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH,
                                 reason="train.backend: fasttd3 needs torch")

#: Heavyweight: builds real torch buffers and runs the backend end to end.
pytestmark = pytest.mark.slow


def _rows(n, n_obs=3, n_act=1, seed=0):
    """A pool in the shape `_ROUND_REPLAY` holds: raw (s, a, s', done)."""
    rng = np.random.default_rng(seed)
    from bird.components.training import _ReplaySlice
    return _ReplaySlice(
        rng.normal(size=(n, n_obs)).astype(np.float32),
        rng.uniform(-1, 1, size=(n, n_act)).astype(np.float32),
        rng.normal(size=(n, n_obs)).astype(np.float32),
        (rng.random(n) < 0.1))


@needs_torch
def test_export_then_prefill_is_the_identity():
    """The lane layout is the export's inverse, checked by composing them.

    `_fasttd3_export_replay` flattens `(n_env, rows)` TIME-MAJOR; the prefill
    must lay a flat pool back the same way. Asserting on either half alone
    would pass with both wrong in the same direction.
    """
    import torch
    from bird.components import fasttd3 as FT

    ns = FT._torch_classes()
    n_env, k, n_obs, n_act = 4, 5, 3, 1
    rb = ns["SimpleReplayBuffer"](n_env=n_env, buffer_size=k + 2, n_obs=n_obs,
                                  n_act=n_act, n_steps=1, gamma=0.99,
                                  device=torch.device("cpu"))
    pool = _rows(n_env * k, n_obs, n_act)

    class _View:
        n_clipped = 0
        n_steps = 0

        @staticmethod
        def to_agent_action(a):
            return np.asarray(a, dtype=np.float32)

        @staticmethod
        def to_env_action(a):
            return np.asarray(a, dtype=np.float32)

        @staticmethod
        def clip_env_action(a):
            return np.asarray(a, dtype=np.float32)

    written = FT._fasttd3_prefill_replay(
        ns, _View, rb, pool, lambda s, a, ns_: (0.0, {}), "none",
        torch.device("cpu"), n_env, 1)
    assert written == n_env * k

    back = FT._fasttd3_export_replay(_View, rb, n_env * k)
    assert back is not None
    np.testing.assert_allclose(back.obs, pool.obs, rtol=0, atol=1e-6)
    np.testing.assert_allclose(back.next_obs, pool.next_obs,
                               rtol=0, atol=1e-6)
    np.testing.assert_allclose(back.act, pool.act, rtol=0, atol=1e-6)


@needs_torch
def test_an_n_step_pool_is_refused_rather_than_summed():
    """`num_steps > 1` over pooled rows would add unrelated rewards.

    The n-step window walks forward assuming the lane is one episode stream
    and masks on `dones` alone -- truncations do not stop the sum. A pool is
    several arms' rows in arbitrary order, so the window would produce a
    number that is wrong and looks right. Refused, as `_sb3_prefill_replay`
    refuses `optimize_memory_usage`.
    """
    import torch
    from bird.components import fasttd3 as FT

    ns = FT._torch_classes()
    rb = ns["SimpleReplayBuffer"](n_env=2, buffer_size=8, n_obs=3, n_act=1,
                                  n_steps=3, gamma=0.99,
                                  device=torch.device("cpu"))

    class _View:
        @staticmethod
        def to_agent_action(a):
            return np.asarray(a, dtype=np.float32)

    written = FT._fasttd3_prefill_replay(
        ns, _View, rb, _rows(8), lambda s, a, ns_: (0.0, {}), "none",
        torch.device("cpu"), 2, 3)
    assert written == 0, (
        "an n-step pool was accepted; the window would sum rewards across "
        "transitions from different arms and call it an n-step return")
    assert int(rb.ptr) == 0


@needs_torch
def test_a_pool_shorter_than_one_lane_column_writes_nothing():
    """Fewer rows than `n_env` leaves no whole column, and a partial one would
    put padding lanes in the buffer that `sample` cannot tell from real rows."""
    import torch
    from bird.components import fasttd3 as FT

    ns = FT._torch_classes()
    rb = ns["SimpleReplayBuffer"](n_env=8, buffer_size=4, n_obs=3, n_act=1,
                                  n_steps=1, gamma=0.99,
                                  device=torch.device("cpu"))

    class _View:
        @staticmethod
        def to_agent_action(a):
            return np.asarray(a, dtype=np.float32)

    assert FT._fasttd3_prefill_replay(
        ns, _View, rb, _rows(3), lambda s, a, ns_: (0.0, {}), "none",
        torch.device("cpu"), 8, 1) == 0
    assert int(rb.ptr) == 0


def test_shared_buffer_on_fasttd3_is_no_longer_refused():
    """The coherence refusal that fasttd3's replay support lifts.

    No torch needed: this is a load-time rule. It is the cheapest possible
    regression test for the thing that would otherwise silently re-block every
    LaRes cell on this backend.
    """
    cfg = load("lares", overrides={"train.backend": "fasttd3",
                                   "train.interaction": "shared_population",
                                   "train.interaction_cfg.shared_buffer": True},
               profile="tester")
    assert cfg.get("train.interaction_cfg.shared_buffer") is True
    assert cfg.get("train.backend") == "fasttd3"

