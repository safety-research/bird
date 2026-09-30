"""Feedback rollouts are env interaction and must reach `budget.json`.

THE SHAPE, three times over. Every backend ends a training with a loop of
`evaluate.rollouts_per_candidate` real rollouts and accumulates their steps into
`TrainResult.env_steps_used`; those steps must be charged to `ctx.budget` too.
The dedicated TPE-store loop twenty lines below each one records its own steps
(*"Collection steps must reach budget.json, not only env_steps_used"*), which
makes the feedback loop above it the one easy to leave uncharged -- in all
three copies: `_run_backend`, `_sb3_run` and `fasttd3`.

WHAT IT COSTS. Left uncharged on a short run, `env_steps_used` reads 19920
against a budget delta of 16920 -- the 3000 difference being this loop, against
a TRAINING of 1920. A rate or a cost row built on the budget under-reports by
the rollouts; one built on `env_steps_used` over-reports by them relative to
the budget. Two artifacts silently disagreeing about one quantity is exactly
what the store-loop charge exists to stop.

IT IS A CONSTANT, NOT A SCALING TERM, which is why it is easy to miss: the
loop runs `evaluate.rollouts_per_candidate` (default 3) times per candidate
regardless of `train.env_steps`, so at 1M steps it is 0.3% and at a 2000-step
pilot it is 15% and swamps the training. Short budgets -- pilots, cascade
screens -- are exactly where it matters and exactly where nobody looks.
"""

from __future__ import annotations

import inspect

from bird.components import fasttd3, training


#: Where the feedback loop starts, and where the STORE loop starts. The
#: window between them is the only region a feedback-loop charge can be in.
_FEEDBACK_MARK = "_retain_replay_states"
_STORE_MARK = "_n_store_rollouts(cfg)"

#: The CALL, with its receiver, not the bare name. A comment near the store
#: loop can contain the words "already calls `record_rollout_steps`", and a
#: substring count would read that comment as a charge. A test that counts
#: mentions cannot tell an implementation from a sentence about one.
_CHARGE = "ctx.budget.record_rollout_steps("


def _charges(src: str) -> int:
    """Calls to `record_rollout_steps`, ignoring comments and prose."""
    return sum(
        line.split("#", 1)[0].count(_CHARGE) for line in src.splitlines())


def _feedback_loop_src(fn_src: str, where: str) -> str:
    """The source BETWEEN the feedback loop and the store loop.

    BOUNDED BY THE STORE LOOP'S NAME, not by a character count. With a fixed
    window such as `src[i:i + 900]`, the store loop's own
    `record_rollout_steps` in `_sb3_run` and `fasttd3_backend` sits only a few
    characters outside it: one added line or two more words of comment would
    pull it inside, and the assertion would then pass on the STORE loop's
    charge while the feedback charge was deleted, in the two backends that
    have no other coverage.
    A window whose correctness depends on a character count is a window that
    is already wrong; it just has not been edited yet.

    Both markers are required rather than defaulted, so a rename fails here
    with a name instead of silently returning a window over the wrong code.
    """
    start = fn_src.find(_FEEDBACK_MARK)
    assert start >= 0, f"{where}: no {_FEEDBACK_MARK!r}; the loop was renamed"
    end = fn_src.find(_STORE_MARK, start)
    assert end > start, (
        f"{where}: no {_STORE_MARK!r} after the feedback loop, so this window "
        f"has no right-hand edge and would run to the end of the function")
    return fn_src[start:end]


def test_all_three_feedback_loops_charge_the_budget():
    """Named per backend, because they are three copies and not one call.

    A count would let one be fixed and another regress to the same total.
    Asserted on the source rather than by driving three learners, two of
    which need torch and an extra this suite does not install.
    """
    for src, where in (
        (inspect.getsource(training._run_backend), "_run_backend"),
        (inspect.getsource(training._sb3_run), "_sb3_run"),
        (inspect.getsource(fasttd3.fasttd3_backend), "fasttd3_backend"),
    ):
        seg = _feedback_loop_src(src, where)
        assert _charges(seg) >= 1, (
            f"{where}'s feedback-rollout loop does not charge the budget; "
            f"budget.json will under-report every run by "
            f"evaluate.rollouts_per_candidate x the horizon")


def test_the_store_loops_still_charge_it_too():
    """The store-loop charge must stay alongside the feedback-loop charge."""
    for src, where in (
        (inspect.getsource(training._run_backend), "_run_backend"),
        (inspect.getsource(training._sb3_run), "_sb3_run"),
        (inspect.getsource(fasttd3.fasttd3_backend), "fasttd3_backend"),
    ):
        assert _charges(src) >= 2, (
            f"{where} has fewer than two rollout charges; the feedback loop "
            f"and the TPE-store loop are both env interaction")


def test_record_rollout_steps_moves_env_steps_and_nothing_else():
    """It is steps, not a training: no `policy_trainings`, no cap check.

    A rollout charged as a training would corrupt CARD's headline, whose
    whole contribution is RL runs NOT launched.
    """
    from bird.budget import Budget

    b = Budget()
    before = b.snapshot() if hasattr(b, "snapshot") else None
    b.record_rollout_steps(75)
    assert b.env_steps == 75
    assert getattr(b, "policy_trainings", 0) == 0, (
        "a feedback rollout was counted as a policy training")
    if before is not None:
        assert b.delta_since(before).get("env_steps") == 75, (
            "the charge does not travel home from a fork")
