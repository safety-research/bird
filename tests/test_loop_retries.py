"""`loop.max_iteration_retries` caps the retries of ONE failed iteration.

The key's contract: "Maximum number of times a failed iteration is retried
under `loop.on_total_failure: retry_iteration`" -- per failed iteration. A
`retries` counter reset only after a SUCCESSFUL iteration would, once iteration
k had spent the cap and been advanced, advance every later all-failed iteration
in the same streak after a single attempt: with the cap at 3, all-fail would
give attempts {0: 4, 1: 1, 2: 1} where the contract gives {0: 4, 1: 4, 2: 4}.

No shipped config selects `retry_iteration` (eureka uses `continue`, matching
upstream), so this is reached by override only -- which is exactly why it needs
a pin: the path has no config to notice it.
"""

from __future__ import annotations

import collections
import importlib.util

from conftest import REPO
from bird import registry
from bird.budget import Budget
from bird.config import load
from bird.context import Context
from bird.types import Candidate, CandidateReport, Selection, TrainResult


def _entry():
    spec = importlib.util.spec_from_file_location("bird_entry", REPO / "bird.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _selection(it: int) -> Selection:
    cand = Candidate(cand_id=f"c{it:04d}", iteration=it,
                     reward_code="def compute_reward(s, a, s2):\n    return 0.0\n")
    rep = CandidateReport(cand_id=cand.cand_id, candidate=cand,
                          result=TrainResult(cand_id=cand.cand_id, candidate=cand),
                          fitness=1.0)
    return Selection(winners=[rep], rule="argmax")


def _run(script, n_iterations: int = 3, cap: int = 3):
    """`run_search` with `run_iteration` replaced by `script(iteration, attempt)`
    -> Selection | None, and every checkpoint call recorded."""
    registry.load_all()
    entry = _entry()
    cfg = load("eureka", profile="tester", overrides={
        "loop.n_iterations": n_iterations,
        "loop.on_total_failure": "retry_iteration",
        "loop.max_iteration_retries": cap,
    })
    ctx = Context(cfg=cfg, budget=Budget())
    attempts: collections.Counter = collections.Counter()
    ckpts = []

    def scripted(ctx_, state):
        attempts[state.iteration] += 1
        return script(state.iteration, attempts[state.iteration])

    entry.run_iteration = scripted
    entry._ckpt = (lambda ctx_, phase, state=None, restart=0, next_iteration=0, retries=0:
                   ckpts.append((phase, next_iteration, retries)))
    state = entry.run_search(ctx, restart=0)
    return state, dict(attempts), ckpts


def test_each_failed_iteration_gets_the_full_retry_budget() -> None:
    """Iteration 0 fails every time (1 + 3 retries, then advanced); iteration 1
    fails ONCE and must be retried -- a per-streak counter would already have
    spent its cap on iteration 0 and advanced it after one attempt."""
    def script(it, attempt):
        if it == 0:
            return None
        if it == 1 and attempt == 1:
            return None
        return _selection(it)

    state, attempts, ckpts = _run(script)

    assert attempts == {0: 4, 1: 2, 2: 1}, (
        f"attempts per iteration {attempts}: the retry cap is per failed "
        f"iteration, and iteration 1 was owed a retry")
    assert state.fitness_history == [1.0, 1.0]


def test_the_give_up_checkpoint_carries_the_reset_count() -> None:
    """A leg resumed from `iteration_failed` must agree with an uninterrupted
    one, so the checkpoint written when an iteration is given up carries the
    count the NEXT iteration starts from -- zero -- not the spent cap."""
    def script(it, attempt):
        return None if it == 0 else _selection(it)

    _state, attempts, ckpts = _run(script, n_iterations=2)

    assert attempts == {0: 4, 1: 1}
    retries_written = [(phase, retries) for phase, _n, retries in ckpts]
    assert ("iteration_retry", 3) in retries_written
    failed = [(n, r) for phase, n, r in ckpts if phase == "iteration_failed"]
    assert failed == [(1, 0)], (
        f"iteration_failed checkpoints {failed}: next_iteration 1 should start "
        f"with retries=0")


def test_all_failed_iterations_each_spend_the_cap() -> None:
    """The contract's literal reading: every iteration fails, every iteration
    gets 1 + cap attempts."""
    state, attempts, _ = _run(lambda it, attempt: None, n_iterations=3, cap=2)
    assert attempts == {0: 3, 1: 3, 2: 3}
    assert state.fitness_history == []
    assert state.consecutive_failures == 9
