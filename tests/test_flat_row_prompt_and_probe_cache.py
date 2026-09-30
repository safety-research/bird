"""The prompt says how the reward receives the state, and the probe pays the
env's compile once.

Both halves address a failure shape on a jax-traced (MJX) task, where a
candidate can fail verify for a reason that is not its own. A reward that reads
`s.tool_pos` fails with `AttributeError: BatchTracer has no attribute tool_pos`
under a method rendering `full_source` -- the ONE of the four renderings that
never emits the flat index map, so such a method is handed the adapter's source
and no statement of what a reward is called with. And a candidate can time out
on a single traced call against an MJX adapter whose first sampling compiles
for minutes: the candidate is not slow, the env is, and each candidate would
pay it again.
"""
from __future__ import annotations

import time

import numpy as np

from bird.components import verification as V
from bird.components.generation import _flat_row_interface


class _Env:
    """Only what the renderer reads."""

    _state_fields = (("tool_x", "the tool's x"), ("tool_y", "the tool's y"),
                     ("dist_tool_target", "tool-to-target distance"))
    _state_groups = (("tool_pose_velocity_force", 0, 2), ("task_fields", 2, 3))


class _Ctx:
    def __init__(self, env=None):
        import random
        self.env = _Env() if env is None else env
        self.rng = random.Random(0)


# -- A: the prompt states the interface --------------------------------------


def test_the_paragraph_names_the_fields_the_model_actually_guessed():
    """A RULE ALONE WOULD NOT HAVE HELPED. `tool_pos` and `dist` are not spec
    names -- the row calls them `tool_x/y/z` and `dist_tool_target` -- so a
    paragraph that only said "index it" would leave the model guessing the
    spelling. The map is the half that fixes that, so it is rendered in full.
    """
    out = _flat_row_interface(_Ctx())
    assert "tool_x: s[0]" in out
    assert "dist_tool_target: s[2]" in out
    assert "flat arrays of width 3" in out
    assert "tool_pose_velocity_force: s[0:2]" in out


def test_the_paragraph_forbids_attributes_on_the_state_and_says_nothing_of_self():
    """SCOPED TO `state` AND `next_state`. The flat-array contract is
    universal, so this is emitted on every tier -- but `self` is a different
    object (`call_reward` binds a `_SelfProxy`) and
    `pythonic_class_abstraction` advertises callable helpers ON PURPOSE. A
    blanket "there are no attributes" would deny T2R and CARD an interface
    they are deliberately given, manufacturing on those methods the very failure
    this fixes.
    """
    out = _flat_row_interface(_Ctx())
    assert "never `state.tool_x`" in out and "carry no attributes" in out
    assert "self" not in out, "the paragraph must say nothing about `self`"


def test_an_adapter_with_no_declared_fields_renders_nothing():
    """No fields, no paragraph -- rather than an empty, authoritative-looking
    table. `env_spec: none` routes around this seam entirely."""
    class _Bare:
        _state_fields = ()
        _state_groups = ()
    assert _flat_row_interface(_Ctx(_Bare())) == ""


def test_groups_are_omitted_when_the_spec_does_not_declare_them():
    """A tier without `state_surface.fields` still gets the positional table.
    The map alone is a complete interface; the groups are a convenience."""
    class _NoGroups:
        _state_fields = (("x", "d"), ("y", "d"))
        _state_groups = ()
    out = _flat_row_interface(_Ctx(_NoGroups()))
    assert "x: s[0]" in out and "contiguous groups" not in out


# -- B: the env's compile is paid once ---------------------------------------


class _SlowEnv:
    """Stands in for an MJX adapter: sampling is the expensive part."""

    def __init__(self, cost=0.05):
        self.calls, self._cost = 0, cost

    def sample_transitions(self, rng, n):
        self.calls += 1
        time.sleep(self._cost)
        return [(np.zeros(3), np.zeros(2), np.zeros(3)) for _ in range(n)]


def test_eight_candidates_sample_the_env_once_not_eight_times():
    """The measured shape of the failure: eight candidates, eight
    compiles. Wall time is asserted as a RATIO against the uncached path
    rather than an absolute, so it says the same thing on any box."""
    V.reset_probe_transitions()
    env = _SlowEnv()
    ctx = _Ctx(env)
    t0 = time.time()
    for _ in range(8):
        V.probe_transitions(ctx, 8)
    cached_s = time.time() - t0
    assert env.calls == 1, f"the env was sampled {env.calls} times, not once"

    V.reset_probe_transitions()
    env2 = _SlowEnv()
    ctx2 = _Ctx(env2)
    t0 = time.time()
    for _ in range(8):
        V.sample_transitions(ctx2, 8)
    uncached_s = time.time() - t0
    assert env2.calls == 8
    assert cached_s < uncached_s / 2, (cached_s, uncached_s)


def test_the_escape_hatch_restores_per_call_drawing(monkeypatch):
    """`BIRD_NO_PROBE_CACHE=1`, so a caller needing per-call drawing does not
    edit code -- and so this proves the cache is what makes the difference
    above, rather than the fixture being cheap."""
    monkeypatch.setenv("BIRD_NO_PROBE_CACHE", "1")
    V.reset_probe_transitions()
    env = _SlowEnv(cost=0.0)
    ctx = _Ctx(env)
    for _ in range(4):
        V.probe_transitions(ctx, 4)
    assert env.calls == 4


def test_the_cache_does_not_reach_the_screens_sampler():
    """SCOPE. `sample_transitions` is what the EPIC/STARC screens draw
    through, and they re-draw per call by design: a pseudometric over reward
    functions wants its own distribution. A cache inside it would silently
    change what the screens measure, so the cache sits beside it."""
    V.reset_probe_transitions()
    env = _SlowEnv(cost=0.0)
    ctx = _Ctx(env)
    V.sample_transitions(ctx, 4)
    V.sample_transitions(ctx, 4)
    assert env.calls == 2, "sample_transitions itself must not have been cached"


def test_the_traced_probe_draws_before_it_starts_the_clock():
    """The budget times the TRACE, not the env.

    Asserted on the source because reaching `_probe_traced` needs jax, which
    this suite does not have -- but the ORDER of two statements is the whole
    point: with `t0 = time.time()` above the sampling, a five-minute compile
    would be spent against a sixty-second budget and the candidate reported as
    having timed out.
    """
    import inspect

    src = inspect.getsource(V._probe_traced)
    assert src.index("probe_transitions(ctx, n)") < src.index("t0 = time.time()"), (
        "the clock starts before the transitions are drawn")
