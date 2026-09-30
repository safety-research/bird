"""§2 and §3 must hand ONE program the SAME arguments, and a repair must be
postprocessed like a generation.

TWO DEFECTS, NEITHER VISIBLE FROM ANY SINGLE MODULE. They are tested together
because the second hides the first: a repair loop that keeps producing a
differently-broken program makes ten LLM calls per candidate report ten failures,
none of them the original one.

1. A `verification.call_reward` that binds POSITIONALLY, with a single name test
   -- whether the SECOND parameter contains "next" -- swaps the arguments of both
   papers' published Meta-World line, `def compute_dense_reward(self, action,
   obs)`: `action` receives the state and `obs` the action. Measured, one program
   through both stages with a 39-wide state of 0..38 and a 4-wide action of -1:

       VERIFY  call_reward     -> action_len=39  obs_len=4   obs[0]=-1.0
       TRAIN   CompiledReward  -> action_len=4   obs_len=39  obs[0]=0.0

   On Meta-World `obs` is then the 4-wide ACTION, and every candidate dies
   `ValueError: operands could not be broadcast together with shapes (3,) (0,)`
   at `obs[4:7]` -- deterministically, on three independent samples, which is
   what marks it a harness defect rather than a bad generation.

   The `self` half of the same split is noted in `verification.py`'s comment on
   the jax tier (`_make_binder` passes `self` as None; `call_reward` a
   `_SelfProxy`). The argument-order half is the one that changes results.

   `generate.output.signature` REVEALS this rather than causes it: the default
   OUTPUT CONTRACT pins one line, `(state, action=None, next_state=None)`, whose
   order positional binding happens to get right.

2. A `verification._resample_with_trace` that builds the repaired candidate
   straight from `_extract_code` and applies NO postprocess, while inheriting
   the parent's `meta`, sends every repaired program under a non-empty
   `generate.postprocess.symbol_mapping` to the harness with the prompt's own
   symbols unrewritten -- and the artifact records `symbol_mapping_applied: 7`
   against a program the mapping never touched.
"""

from __future__ import annotations

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components import verification
from bird.components.generation import _SYMBOLS_T2R_METAWORLD
from bird.components.training import _ARG_BY_NAME, CompiledReward
from bird.config import load
from bird.context import Context
from bird.types import Candidate

#: Distinguishable by WIDTH and by VALUE, so a swap cannot pass by coincidence:
#: 39 ascending values against 4 constant -1s is Meta-World's own pair of shapes.
STATE = np.arange(39, dtype=float)
ACTION = np.full(4, -1.0)
NEXT = STATE + 100.0

#: Every signature shape a published method emits, plus the one BIRD pins. The
#: second element is how the program names its STATE and ACTION parameters, so
#: one probe body can report both whatever they are called.
SIGNATURES = [
    ("state, action=None, next_state=None", "state", "action"),   # BIRD's default
    ("self, action, obs", "obs", "action"),                        # T2R / CARD
    ("self, action: np.ndarray, obs: np.ndarray", "obs", "action"),  # CARD, annotated
    ("s, a", "s", "a"),
    ("s, a, s2", "s", "a"),
    ("state, next_state", "state", None),                          # a published pair
]


def _probe(sig: str, state_name: str, action_name: str):
    """A program that REPORTS its arguments' shapes instead of computing a reward.

    The instrument that exposes the defect: `execution_smoke` records an
    exception's text in the candidate's verify record, so a program that raises
    its own argument shapes reads them back out of the artifact with no
    instrumentation of the harness at all.
    """
    seen: list = []
    src = (f"def compute_dense_reward({sig}):\n"
           f"    _state = {state_name}\n"
           f"    _action = {action_name if action_name else 'None'}\n"
           f"    seen.append((None if _state is None else np.shape(np.asarray(_state)),\n"
           f"                 None if _action is None else np.shape(np.asarray(_action))))\n"
           f"    return 0.0\n")
    ns = {"np": np, "seen": seen}
    exec(compile(src, "<probe>", "exec"), ns)
    return ns["compute_dense_reward"], seen


class _Ctx:
    env = None


@pytest.mark.parametrize("sig,state_name,action_name", SIGNATURES,
                         ids=[s[0][:28] for s in SIGNATURES])
def test_verify_and_train_bind_one_program_the_same_way(sig, state_name, action_name):
    """The invariant both defects violate, asserted directly.

    IT COMPARES THE ARGUMENTS EACH STAGE IS HANDED, NOT THE VALUES RETURNED, and
    that is not a stylistic choice: §3 puts a reward's return through
    `CompiledReward._unpack` and §2 does not, so comparing returns compares two
    different contracts and fails for every signature -- including the ones that
    bind correctly. Such a test is uniformly red, which reads like a
    catastrophic bug and is really a broken instrument. The probe therefore RECORDS its arguments into a list and
    returns a plain scalar both stages accept.

    A positional binder fails on `self, action, obs`: §2 is handed `((4,), (39,))` where §3
    is handed `((39,), (4,))`.
    """
    fn, seen = _probe(sig, state_name, action_name)

    # The ARGUMENTS each stage passes, not the values it returns: §3 unpacks a
    # reward's return through `_unpack` and §2 does not, so comparing returns
    # would compare two different contracts and prove nothing about binding.
    verification.call_reward(_Ctx(), fn, verification.Transition(STATE, ACTION, NEXT))
    CompiledReward(fn, "compute_dense_reward")(STATE, ACTION, NEXT)

    assert len(seen) == 2, seen
    verify_args, train_args = seen
    assert verify_args == train_args, (
        f"{sig!r}: stage 2 was handed {verify_args}, stage 3 {train_args}")
    # and it is the RIGHT way round, not merely consistent
    assert verify_args[0] == (39,), f"{sig!r}: the state argument is not the state"
    if action_name:
        assert verify_args[1] == (4,), f"{sig!r}: the action argument is not the action"


def test_the_published_metaworld_line_is_bound_by_name_not_position():
    """The exact defect, named, so a future positional shortcut fails here first.

    `(self, action, obs)` puts the ACTION first. Any binder that fills
    parameters in `(state, action, next_state)` order regardless of their names
    hands this program its arguments backwards, and the failure downstream is a
    numpy shape error inside the candidate -- which reads as the generator's
    fault."""
    fn, seen = _probe("self, action, obs", "obs", "action")
    verification.call_reward(_Ctx(), fn, verification.Transition(STATE, ACTION, NEXT))
    assert seen == [((39,), (4,))], f"arguments bound backwards: {seen}"
    # the table is the single source both stages read
    assert _ARG_BY_NAME["obs"] == 0 and _ARG_BY_NAME["action"] == 1


def test_a_name_the_table_does_not_know_still_binds_positionally():
    """The fallback, so name binding cannot refuse a program positional binding runs.

    Stage 2 exists to run programs, not to grade their naming."""
    fn, seen = _probe("x, y", "x", "y")
    verification.call_reward(_Ctx(), fn, verification.Transition(STATE, ACTION, NEXT))
    assert seen == [((39,), (4,))]


def test_an_unknown_trailing_default_does_not_reopen_the_swap():
    """One parameter away from the pinned line, and the reason the binder checks defaults.

    `(self, action, obs, extra=0)` cannot be bound by name alone -- `extra` is
    not in the table -- and an all-or-nothing rule would drop it to the
    positional path and swap it again. The rule is: fill the names the table
    knows, and stop at the first unknown that has a default."""
    fn, seen = _probe("self, action, obs, extra=0", "obs", "action")
    verification.call_reward(_Ctx(), fn, verification.Transition(STATE, ACTION, NEXT))
    assert seen == [((39,), (4,))], f"the trailing default reopened the swap: {seen}"


# --------------------------------------------------------------------------
# Defect 2: a repair is a new program and must be postprocessed like one
# --------------------------------------------------------------------------

#: Written in the abstraction's own vocabulary, which is what the prompt shows
#: and what the table exists to rewrite.
_REPAIRED = ("def compute_dense_reward(self, action, obs):\n"
             "    reach = self.robot.ee_position - self.obj1.position\n"
             "    return -float(np.linalg.norm(reach)) - float(self.goal_position[0]), {}\n")


class _FixedGenerator:
    """Returns one canned reply, however it is called."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def __call__(self, messages, tag=None, **kw):
        self.calls += 1
        return "```python\n" + self.text + "```"


def _repair_ctx() -> Context:
    registry.load_all()
    cfg = load("card", profile="tester")
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.generator = _FixedGenerator(_REPAIRED)
    return ctx


def _failed_parent() -> Candidate:
    return Candidate(
        cand_id="c0000", iteration=0,
        reward_code="def compute_dense_reward(self, action, obs):\n    return dist, {}\n",
        raw_response="```python\nbroken\n```",
        prompt_messages=[{"role": "user", "content": "write it"}],
        meta={"symbol_mapping_applied": 7, "parse_pattern": "```python(.*?)```",
              "sampling_mode": "iid_parallel", "sample_index": 0},
    )


def test_a_repaired_candidate_is_symbol_mapped():
    """The defect: a repaired program that keeps the prompt's raw symbols.

    `card` resolves `symbol_mapping: t2r_metaworld_global`, so a program written
    against the abstraction must come back with NO table key left in it.
    Without the postprocess every key survives and the candidate dies on
    `self.<symbol>` for as many attempts as the cap allows."""
    ctx = _repair_ctx()
    fresh, err = verification._regenerate(ctx, None, _failed_parent(), attempt=1)
    assert err == "" and fresh is not None, err
    survivors = [k for k in _SYMBOLS_T2R_METAWORLD if k in fresh.reward_code]
    assert survivors == [], f"unmapped keys survived the repair: {survivors}"
    assert "obs[" in fresh.reward_code, fresh.reward_code


def test_a_repaired_candidates_meta_describes_its_own_program():
    """The artifact must not inherit a claim about the parent's text.

    `symbol_mapping_applied` on a program the mapping never touched is worse
    than the bug it hid, because the bug is at least findable. Sampling
    provenance IS inherited -- a repair is a re-query of the same sample -- and
    `parse_pattern` is dropped rather than re-supplied, because `_extract_code`
    does its own extraction and does not report which pattern matched."""
    ctx = _repair_ctx()
    fresh, err = verification._regenerate(ctx, None, _failed_parent(), attempt=1)
    assert err == "" and fresh is not None, err
    assert fresh.meta.get("symbol_mapping_applied") == len(_SYMBOLS_T2R_METAWORLD)
    assert "parse_pattern" not in fresh.meta
    assert fresh.meta.get("sampling_mode") == "iid_parallel"
    assert fresh.meta.get("repaired_from") == "c0000"
