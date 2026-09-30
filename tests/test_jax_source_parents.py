"""`source_parents()` and what `full_source` renders because of it.

WHY THIS FILE EXISTS. A batched adapter may hold a CPU adapter rather than
subclassing it, and then `inspect.getsource(type(self))` is the subclass alone
-- which defines no `task_metric`, no `success`, no `reference_reward`. The leak
guard in `tests/test_env_spec_leak.py` withholds ground-truth methods from the
rendered source, and a render with none to withhold PASSES WHILE CHECKING
NOTHING. `source_parents()` names the CPU classes `_render_full_source` prepends.

`jax_toy`, the one shipped batched adapter, is standalone: it is written
directly in `jax.numpy`, has no CPU parent, and its `source_parents()` is `()`.
The prepend path is therefore exercised here with a jax-free delegating stub.

The renderer is UNSTRIPPED and that is deliberate: `full_source` includes
`reference_reward` because that is what "full source" means (`base.py`'s
`describe` docstring), and `strip_existing_reward` cuts it in §1. The leak
test strips the render itself and asserts the defs were there to strip, so a
renderer that stripped would fail it.
"""
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

from bird import registry
from bird.envs.jax_base import BatchedEnvAdapter
from bird.envs.toy import ToyReacher

#: Recorded before `_render_full_source` prepended parents, so they pin that
#: the change did not move a CPU env's render. Literal ids, never a prefix
#: scan over the registry: a `[:1]` silently changes what is pinned when the
#: registry moves.
#: id -> (sha256, byte length, the packages its CONSTRUCTION needs).
#: The third element is not decoration: this case is deliberately NOT
#: jax-marked -- it is the CPU-side regression anchor and must run in the
#: default CI job, which installs only `--extra test` (pyyaml + numpy +
#: pytest, with no mujoco, no gymnasium and no metaworld). Without a per-id
#: skip it goes red there while passing in a jax venv, which is the reverse of
#: the blindness these tests exist to remove.
RECORDED_RENDER_SHAS = {
    # A TUPLE, because `gym_half_cheetah` needs BOTH: `gym_mujoco.py:978-979`
    # imports `gymnasium` AND `mujoco`, so naming one would still go red on an
    # install that had that one and not the other.
    # The full_source render quotes the adapter class body verbatim (docstrings
    # and comments included), so an edit anywhere in it moves every method's
    # prompt for that env. These are the renderings the paper's runs were
    # made with: an edit that moves one is to be reverted, not re-recorded.
    "toy_reacher": ("fa1237d9d660db308208c739153185dd9dec4944557772797b61445f3791476a",
                    7779, ()),
    # Recorded under gymnasium 1.3.0 and mujoco 3.3.0: this render includes upstream
    # source, so a different gymnasium version can move it.
    "gym_half_cheetah": ("1f95047a06973947ed6d162c1140684a8fae9b51ade08b532cbeb3691d2596fa",
                         26685, ("gymnasium", "mujoco")),
    "assistax_feeding": ("e2441ab3167fe376a51bbb396c0d84313eda71ae4b63184a988165e34f1cbfad",
                         13058, ("mujoco",)),
    "mt10_button-press-topdown-v3":
        ("95a03bcc24d9803675eb85b27a41143578b13c3a6e3fdb85c95189e3c9c1b595",
         19531, ("metaworld", "mujoco")),
}

GROUND_TRUTH_DEFS = ("task_metric", "success", "reference_reward")


def _jax_ids():
    registry.load_all()
    return sorted(n for n in registry.names("env") if n.startswith("jax_"))


def test_the_jax_family_is_not_empty():
    """Guards the enumeration: if the prefix stops matching, every case below
    vanishes and this file passes having tested nothing."""
    assert _jax_ids()


@pytest.mark.jax
@pytest.mark.parametrize("env_id", _jax_ids())
def test_a_jax_rendering_is_substantial_and_hides_no_ground_truth(env_id):
    """Non-empty, names its own class, and carries no ground-truth body.

    THE FIRST TWO CLAUSES ARE NOT PADDING. "Contains no ground-truth method
    body" is satisfied by rendering NOTHING AT ALL, which is exactly how the
    unguarded state would pass every check. A test whose assertion can be met
    by absence has the defect it is guarding against.
    """
    pytest.importorskip("jax")
    env = registry.get("env", env_id)(None)
    text = env.describe("full_source")
    assert text.strip(), f"{env_id}: rendered nothing"
    assert f"class {type(env).__name__}" in text, (
        f"{env_id}: the rendering does not contain its own class statement")


class _DelegatingStub(BatchedEnvAdapter):
    """A batched adapter that COMPOSES a CPU adapter instead of inheriting it,
    so its own class body carries none of the ground-truth methods."""

    def _build_action_set(self):
        return [[0.0]]

    def _bounds(self):
        return np.full(2, -1.0), np.full(2, 1.0)

    def source_parents(self):
        return (ToyReacher,)

    def random_state(self, rng):
        return np.zeros(2)

    def reset_batch(self, key, n):
        return np.zeros((n, 2), dtype=np.float32)

    def step_batch(self, state, action, physics):
        return state, np.array([False]), {}


def test_a_delegating_adapter_renders_its_parents_source_first():
    """The parent segment is the CPU class's own source, byte for byte, and it
    comes BEFORE the adapter's own class. Needs no jax: `BatchedEnvAdapter`
    imports none."""
    env = _DelegatingStub()
    text = env.describe("full_source")
    parent_src = inspect.getsource(ToyReacher)
    own_src = inspect.getsource(_DelegatingStub)
    assert parent_src in text, "the parent's source is not in the rendering"
    assert own_src in text, "the adapter's own source is not in the rendering"
    assert text.index(parent_src) < text.index(own_src), "the parent is not rendered first"
    # the ground truth is PRESENT -- that is what gives the stripper something
    # to cut, and its absence is the bug this whole contract addresses
    assert not [d for d in GROUND_TRUTH_DEFS if f"def {d}" in own_src]
    present = [d for d in GROUND_TRUTH_DEFS if f"def {d}" in text]
    assert present, (
        "no ground-truth def in the rendering, so the leak guard has nothing to "
        "withhold and passes while checking nothing")


@pytest.mark.parametrize("env_id", sorted(RECORDED_RENDER_SHAS))
def test_a_non_jax_rendering_did_not_move(env_id):
    """`_render_full_source` changed, and a CPU env's prompt must not have.

    The invariant is provable rather than hoped: the renderer only PREPENDS
    segments, `source_parents` exists only on `BatchedEnvAdapter`, and a
    plain `EnvAdapter` subclass has no such method -- so its rendering is
    unchanged by construction. These four shas are what turns that argument
    into a measurement.
    """
    want_sha, want_len, needs = RECORDED_RENDER_SHAS[env_id]
    for pkg in needs:
        pytest.importorskip(pkg)
    registry.load_all()
    env = registry.get("env", env_id)(None)
    text = env.describe("full_source")
    got = hashlib.sha256(text.encode()).hexdigest()
    assert (got, len(text)) == (want_sha, want_len), (
        f"{env_id}: rendering moved -- {len(text)} bytes / {got[:16]} against the "
        f"{want_len} / {want_sha[:16]} recorded. Every method's "
        f"prompt for this env just changed; that is either the bug or a decision "
        f"that needs the sha re-recorded with a reason.")
