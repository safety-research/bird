"""The two pre-training screens on a `generate.reward_language: jax` reward.

GT's TAC screen (`verify.quality_screen: tac`) and
CARD's TPE pruning (`tpe`) both RE-SCORE stored trajectories with the candidate's
own reward, one transition at a time, so on the JAX tier they are handed jnp
values where every other suite hands them numpy. Everything downstream --
`math.isfinite`, the `>` comparison that induces a preference, the rank sort --
assumes a Python float.

THE WHOLE PATH NARROWS TO ONE FUNCTION. `screens._rescore_transitions` and
`_score_transitions` both call `verification.reward_scalar`, which calls
`verification._as_float`. So the question "do the screens work on jnp
rewards" is the question "what does `_as_float` do with a jnp value", plus the
question of what the screens do with its refusals.

TWO TIERS OF TEST, and the split is deliberate rather than a shortcut:

  * `test_as_float_on_jax_values` is `importorskip("jax")`-gated, like every
    other jax test in this tree (the `jax` fixture in `test_jax_toy.py`). The
    jax extra pulls ~3 GB of CUDA wheels and is held out of `all` and out of
    every CI job (pyproject.toml, the `jax` extra), so THIS TEST DOES NOT RUN IN
    CI OR ON A CPU-ONLY INSTALL. It is the real check and it runs on a GPU
    machine with the extra installed.
  * `test_as_float_on_a_jax_shaped_stand_in` runs everywhere. It is NOT a
    substitute and must not be read as one: it exercises `_as_float` against an
    object with the INTERFACE jnp presents to it -- not an int/float/np scalar
    instance, convertible by `np.asarray`, carrying a dtype -- which is all
    `_as_float` actually touches. It would catch a regression that added an
    `isinstance(v, np.ndarray)` fast path or dropped the `np.asarray` fallback.
    It cannot catch anything about jax itself.
"""
import math

import numpy as np
import pytest

from bird.components.verification import RewardShapeError, _as_float


class _ArrayLike:
    """Duck-types what `_as_float` reads off a jnp value, and nothing more."""

    def __init__(self, value, dtype=np.float32):
        self._a = np.asarray(value, dtype=dtype)

    def __array__(self, dtype=None):
        return self._a if dtype is None else self._a.astype(dtype)


def test_as_float_on_a_jax_shaped_stand_in():
    # a scalar re-scores to a float -- the case every screen needs
    assert _as_float(_ArrayLike(1.5)) == pytest.approx(1.5)
    # a one-element array is still a scalar reward
    assert _as_float(_ArrayLike([1.5])) == pytest.approx(1.5)
    # A BATCH IS REFUSED, NOT SILENTLY REDUCED. This is the behaviour that
    # matters most on a batched tier: a reward that returns one value per
    # batch row must not be averaged into a scalar behind the screen's back,
    # because the screen would then rank candidates on a number no paper
    # defines. The refusal reaches the screen as `reward_error`.
    with pytest.raises(RewardShapeError):
        _as_float(_ArrayLike([1.0, 2.0, 3.0, 4.0]))
    # a bool-dtype reward is refused for its dtype, not silently cast to 0/1
    with pytest.raises(RewardShapeError):
        _as_float(_ArrayLike(True, dtype=bool))
    # NaN converts, and is then caught by `_rescore_transitions`'s
    # `math.isfinite` sweep -- the screens count it against the candidate
    # rather than dropping the trajectory. Asserted here so that contract is
    # pinned at the conversion boundary too.
    assert math.isnan(_as_float(_ArrayLike(float("nan"))))


@pytest.mark.jax
def test_as_float_on_jax_values():
    """The real check. Runs only where the jax extra is installed.

    THE GATE NAMES THE DISTRIBUTION, NOT THE SUBMODULE. An
    `importorskip("jax.numpy")` gate fails `test_no_silent_skip` -- correctly
    by its own rule and for a cause worth stating precisely, because the
    obvious reading is wrong. `jax` IS locked (`uv.lock`, from the `jax`
    extra in `pyproject.toml`); what cannot match is the literal
    `"jax.numpy"`, since the checker compares against a list of DISTRIBUTION
    names and a submodule is not one. So such a gate is not un-openable, only
    un-matchable, and an `_UNRESOLVABLE` exemption would write off coverage
    that the jax extra does in fact provide. The `jax` marker's description in
    `pyproject.toml` states the convention for this tier: tests
    `importorskip("jax")` on top of the marker.
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp
    assert _as_float(jnp.float32(1.5)) == pytest.approx(1.5)
    assert _as_float(jnp.asarray(1.5)) == pytest.approx(1.5)
    assert _as_float(jnp.asarray([1.5])) == pytest.approx(1.5)
    # a traced-looking weak-typed reduction is still a scalar
    assert _as_float(jnp.sum(jnp.asarray([1.0, 2.0]))) == pytest.approx(3.0)
    with pytest.raises(RewardShapeError):
        _as_float(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(RewardShapeError):
        _as_float(jnp.asarray(True))
    assert math.isnan(_as_float(jnp.asarray(float("nan"))))
