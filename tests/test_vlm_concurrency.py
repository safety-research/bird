"""Concurrent VLM judging -- wall clock may move, nothing else.

`judge_concurrency` lets independent judge round-trips overlap when the
evaluator client declares it fans out (`concurrent_samples` +
`llm.max_concurrent_requests`, the same transport contract the sampler uses).
The judged quantities -- per-subtask scores, preference labels, budget
counters -- must be identical to the serial loops they replace, and the mock
provider never fans out, so every tester-tier determinism claim is untouched.

Why the loops are worth touching: rda's per-(rollout, subtask) scoring block
and gt's preferences block are serial judge round-trips that dominate an
iteration's wall clock while every worker core idles.
"""

import json
import re
import threading
import types

import numpy as np
import pytest

from bird import registry
from bird.budget import Budget
from bird.components.evaluation import judge_concurrency
from bird.config import load
from bird.context import Context
from bird.types import Candidate, CandidateReport, TrainResult, Trajectory


class _StubJudge:
    """Duck-typed evaluator: deterministic per-prompt answers, records calls.

    Answers through `complete(prompt)` -- the first name both `_client_text`
    (evaluation) and `_client_call` (preferences) probe -- so one stub serves
    both loops. The answer is a pure function of the prompt bytes, which is
    what makes serial and concurrent runs comparable verdict by verdict.
    """

    modality = "text"
    concurrent_samples = True

    def __init__(self, max_concurrent_requests=8):
        self.max_concurrent_requests = max_concurrent_requests
        self.calls = []
        self._lock = threading.Lock()

    def _answer(self, prompt: str) -> str:
        with self._lock:
            self.calls.append(prompt)
        h = (sum(prompt.encode()) % 97) / 97.0
        # Answer SHAPES, keyed on what was actually asked. Keying on "LEFT"
        # in prompt would answer the CAPTION request with the word "RIGHT" --
        # so both sides of every pair would get the same description, every
        # comparison would resolve the same way, and a reordered judge could
        # not change a single label. Answering each stage
        # in its own currency is what makes the labels vary at all.
        #
        # RDA's trajectory analysis (one paper-shaped JSON per trajectory
        # covering the prompt's whole subtask list) gets a real array: an
        # unparseable reply would send EVERY score to the env-flag fallback --
        # a constant per trajectory, and a constant cannot show a reordered
        # fold. Scores are a pure function of (prompt bytes, subtask index), on
        # the 3-point set the rda config pins, so serial and concurrent runs
        # stay comparable verdict by verdict.
        if '"subtasks"' in prompt and "## Subtask List" in prompt:
            section = prompt.split("## Subtask List", 1)[1].split("\n## ", 1)[0]
            subs = [m.strip() for m in re.findall(r"^\s*\d+\.\s+(.+)$", section,
                                                  flags=re.M)]
            entries = [{"number": i + 1, "name": s,
                        "behavior": f"the agent acts out phase {i}",
                        "score": [0.0, 0.5, 1.0][(sum(prompt.encode()) + 37 * i) % 3],
                        "analysis": f"evidence at h={h:.2f} for phase {i}"}
                       for i, s in enumerate(subs)]
            return "```json\n" + json.dumps({"subtasks": entries}) + "\n```"
        if "Answer with exactly one word" in prompt:
            return "RIGHT" if h < 0.5 else "LEFT"
        if "Score how well" in prompt:
            return f"Score: {h:.2f}"
        return f"the agent reached state {h:.2f} and stopped there"

    def complete(self, messages, images=None):
        if isinstance(messages, str):
            return self._answer(messages)
        return self._answer(messages[0]["content"])


def _ctx(name, judge, **overrides):
    """The judge lands on BOTH roles: `fitness_vlm_score` judges with
    `ctx.generator` (the Agent VLM), while the
    preference/caption paths still judge with `ctx.evaluator`. One stub on both
    keeps every call in one `calls` list, which is what the serial/concurrent
    equality below compares."""
    registry.load_all()
    cfg = load(name, profile="tester", overrides=overrides)
    ctx = Context(cfg=cfg, budget=Budget())
    ctx.evaluator = judge
    ctx.generator = judge
    return ctx


def _traj(seed: int) -> Trajectory:
    """Rollouts that a judge can TELL APART.

    `length` varies with the seed, and that is load-bearing rather than
    decorative: a blind judge's prompt carries "Rollout: length=N steps" and
    almost nothing else, so at a fixed length every prompt in the run is
    byte-identical, every answer is the same number, and reassembly order
    becomes unobservable. Measured at a fixed length: 12 judge calls, ONE
    distinct prompt, every subtask score 0.25.
    """
    rng = np.random.default_rng(seed)
    n = 4 + seed % 5
    states = rng.normal(size=(n + 1, 4))
    return Trajectory(states=states, actions=rng.normal(size=(n, 1)),
                      rewards=[float(x) for x in rng.normal(size=n)],
                      success=bool(seed % 2), length=n, ret=float(seed) / 7.0)


def _results(n):
    out = []
    for i in range(n):
        cand = Candidate(cand_id=f"c{i:04d}", iteration=0,
                         reward_code="def reward(s, a, s2):\n    return 0.0\n")
        out.append(TrainResult(cand_id=cand.cand_id, candidate=cand,
                               trajectories=[_traj(3 * i + j) for j in range(2)]))
    return out


def test_judge_concurrency_gates():
    """1 -- the serial loop every run always had -- for one job, for a client
    that never declared fan-out (the mock's shape), and for no client at all."""
    judge = _StubJudge(max_concurrent_requests=8)
    ctx = _ctx("rda", judge)
    assert judge_concurrency(ctx, 1) == 1
    assert judge_concurrency(ctx, 12) == 8
    assert judge_concurrency(ctx, 4) == 4
    ctx.evaluator = object()  # no concurrent_samples attribute: older bird/llm
    assert judge_concurrency(ctx, 12) == 1
    ctx.evaluator = None
    assert judge_concurrency(ctx, 12) == 1
    # `client=` names the judging client explicitly -- `fitness_vlm_score`
    # passes ctx.generator -- and overrides the evaluator default in
    # both directions.
    assert judge_concurrency(ctx, 12, client=judge) == 8
    ctx.evaluator = judge
    assert judge_concurrency(ctx, 12, client=object()) == 1


def _vlm_reports(concurrent: bool):
    judge = _StubJudge()
    judge.concurrent_samples = concurrent
    ctx = _ctx("rda", judge,
               **{"evaluate.vlm.repeats": 2, "evaluate.rollouts_per_candidate": 2})
    score = registry.get("fitness_source", "vlm_score")
    # TWO subtasks, not a `state=None` single pseudo-subtask. With one
    # bucket the fold is `np.mean` over that candidate's own
    # scores and is therefore ORDER-INVARIANT by construction: reversing the
    # concurrent reassembly could not change a number, so the comparison could
    # not fail. With two, a reordered result lands in the wrong subtask's
    # bucket and `subtask_scores` moves -- which is the whole claim.
    state = type("S", (), {"iteration": 0, "subtasks": [
        "swing the pendulum up to upright",
        "hold it balanced with little torque",
    ]})()
    reports = score(ctx, state, _results(3))
    return judge, ctx, reports


def test_vlm_score_concurrent_matches_serial():
    """Same subtask scores, same fitness, same budget -- per candidate, in
    order. The fold keeps the original triple-loop order, so `worst`'s
    first-lowest-rationale tie rule cannot move either."""
    sj, sctx, serial = _vlm_reports(concurrent=False)
    cj, cctx, conc = _vlm_reports(concurrent=True)
    assert [r.cand_id for r in serial] == [r.cand_id for r in conc]
    for a, b in zip(serial, conc):
        assert a.fitness == b.fitness
        assert a.subtask_scores == b.subtask_scores
        assert a.meta.get("vlm_queries") == b.meta.get("vlm_queries")
    assert sorted(sj.calls) == sorted(cj.calls), "different questions were asked"
    assert sctx.budget.llm_calls == cctx.budget.llm_calls
    assert sctx.budget.blind_judgments == cctx.budget.blind_judgments


def _pref_reports(concurrent: bool):
    judge = _StubJudge()
    judge.concurrent_samples = concurrent
    ctx = _ctx("gt", judge)
    results = _results(4)
    reports = [CandidateReport(cand_id=r.cand_id, candidate=r.candidate, result=r,
                               fitness=None)
               for r in results]
    state = type("S", (), {"iteration": 0, "preferences": []})()
    phase = registry.get("phase", "preferences")
    out = phase(ctx, state, reports)
    return judge, ctx, out


def test_preferences_concurrent_matches_serial():
    """Same Bradley-Terry strengths from the same labels in the same plan
    order; the fold (tie resolution, `_pref` construction, `D_pref` order)
    stays sequential and unchanged."""
    sj, sctx, serial = _pref_reports(concurrent=False)
    cj, cctx, conc = _pref_reports(concurrent=True)
    s_str = {r.cand_id: r.meta.get("bt_strength") for r in serial}
    c_str = {r.cand_id: r.meta.get("bt_strength") for r in conc}
    assert s_str == c_str
    assert sj.calls and sorted(sj.calls) == sorted(cj.calls)
    assert sctx.budget.llm_calls == cctx.budget.llm_calls
    assert sctx.budget.vlm_calls == cctx.budget.vlm_calls


def test_the_human_comparator_never_fans_out():
    """The interactive oracle is a person; `parallel_safe` is absent there and
    absent means serial, whatever the evaluator client declares."""
    registry.load_all()
    human = registry.get("comparator", "human")
    assert not getattr(human, "parallel_safe", False)
    for name in ("vlm", "llm_on_vlm_captions"):
        assert getattr(registry.get("comparator", name), "parallel_safe", False)


class _CountingLock:
    """A lock that records how many times it was ENTERED."""

    def __init__(self) -> None:
        self._inner = threading.Lock()
        self.entries = 0

    def __enter__(self):
        self._inner.acquire()
        self.entries += 1          # under the lock: this count is exact
        return self

    def __exit__(self, *exc) -> bool:
        self._inner.release()
        return False


def test_every_judge_budget_write_takes_the_lock(monkeypatch):
    """`_JUDGE_BUDGET_LOCK` must cover every direct `ctx.budget` write here.

    Asserting counter exactness cannot show this: `Budget`'s `+=` is exact on a
    GIL build whether or not the lock is there (measured on CPython 3.11.15 -- 16 threads x 20,000 unlocked `record_llm` calls lost ZERO,
    because CPython prefers to drop the GIL at a call boundary rather than
    between an attribute's LOAD and STORE). So an exactness assertion passes
    with the lock removed and proves nothing. Entry count is deterministic and
    does not.

    The expected count is derived, not magic: `_query` takes the lock twice per
    judge call (`record_llm`, then the completion-token add), and `_render`
    takes it once per blind comparison.
    """
    from bird.components import preferences

    counting = _CountingLock()
    monkeypatch.setattr(preferences, "_JUDGE_BUDGET_LOCK", counting)
    judge, ctx, _out = _pref_reports(concurrent=True)

    assert counting.entries > 0, "no judge budget write took the lock"
    assert counting.entries == 2 * len(judge.calls) + ctx.budget.blind_comparisons, (
        "a direct ctx.budget write in preferences.py bypassed _JUDGE_BUDGET_LOCK")
    # The counters must also be right. On a GIL build they would be even
    # unlocked, so this guards the arithmetic, not the lock.
    assert ctx.budget.llm_calls == len(judge.calls)


def _rundir(path):
    """`path` for `bird/judgments.py::_root` (`:175-183`), plus the no-op
    `event` `Context.event` calls (`bird/context.py:68`). Deliberately not a
    real `RunDir`: this test is about the judge trace, and a run directory
    would write a journal whose ordering is a different claim."""
    path.mkdir(parents=True, exist_ok=True)
    return types.SimpleNamespace(path=path, event=lambda *a, **k: None)


def _records(root):
    """Every judgment record, in file order."""
    d = root / "judgments"
    return [json.loads(line) for f in sorted(d.glob("*.jsonl"))
            for line in f.read_text(encoding="utf-8").splitlines() if line] \
        if d.is_dir() else []


def test_the_judgment_records_are_byte_identical_flattened_vs_per_candidate(tmp_path):
    """The gate on flattening `vlm_score`'s pool across candidates.

    One pool PER CANDIDATE would make the next candidate's frames wait on the
    previous candidate's last verdict, so `vlm_score` prepares every candidate
    serially (the frame phase drives the shared env renderer
    and cannot overlap -- `bird/components/evaluation.py:1158`) and runs all
    their judge round-trips through ONE pool.

    Equal fitness is not enough to pin that: the judge TRACE is an artifact,
    and `bird/judgments.py` records what the judge was shown and what it said.
    So this compares the BYTES of `judgments/*.jsonl` between

      A  one `vlm_score` call over N candidates      -- the flattened path
      B  N `vlm_score` calls of one candidate each   -- the per-candidate
                                                        shape, since a
                                                        single-candidate flatten
                                                        IS a per-candidate pool

    A record carries no timestamp and no uuid -- `query_id` is a hash of the
    prompt and its identity fields (`judgments.py:431-434`) -- so byte
    equality is the right assertion and not a flaky one.

    The mock never declares `concurrent_samples`, so `judge_concurrency`
    returns 1 here and both sides are serial: what this pins is the
    RESTRUCTURE, not the pool. Under a fanning-out provider the record ORDER
    inside a candidate is already the pool's schedule -- `_append` writes in
    arrival order under a lock
    (`judgments.py:510-523`) -- and flattening widens that interleaving from
    within-candidate to across-candidate. Content is unchanged either way;
    order was never guaranteed under concurrency and this test does not claim
    it is.
    """
    # max_concurrent_requests=1 ON BOTH SIDES: this test pins the RESTRUCTURE,
    # not the pool. `_StubJudge` declares `concurrent_samples` and defaults to
    # 8, so leaving it would compare two POOL SCHEDULES and fail for a reason
    # unrelated to the restructure -- records are written inside `_analyse`, so
    # their order under concurrency is arrival order (`judgments.py:510-523`).
    judge = _StubJudge(max_concurrent_requests=1)
    score = registry.get("fitness_source", "vlm_score")
    results = _results(3)
    state = types.SimpleNamespace(iteration=0)

    root_a = tmp_path / "flat"
    ctx_a = _ctx("rda", judge, **{"output.judge_trace.record": "inputs"})
    ctx_a.rundir = _rundir(root_a)
    reports_a = score(ctx_a, state, results)

    reports_b = []
    for i, res in enumerate(results):
        ctx_b = _ctx("rda", _StubJudge(max_concurrent_requests=1),
                      **{"output.judge_trace.record": "inputs"})
        ctx_b.rundir = _rundir(tmp_path / f"one{i}")
        reports_b.extend(score(ctx_b, state, [res]))

    flat_recs = _records(root_a)
    per_cand = [r for i in range(len(results))
                for r in _records(tmp_path / f"one{i}")]

    assert flat_recs, "no judgment records were written; the gate would pass vacuously"

    # (1) CONTENT: every record, byte for byte, exactly once on each side.
    assert sorted(json.dumps(r, sort_keys=True) for r in flat_recs) == \
           sorted(json.dumps(r, sort_keys=True) for r in per_cand), (
        "flattening changed WHAT the judge trace records. `judgments.py` "
        "writes what the judge was shown and what it said; a reader of "
        "`judgments/*.jsonl` is entitled to the same records for the same "
        "inputs, and this is the assertion that a fitness comparison misses.")

    # (2) ORDER WITHIN EACH KIND: unchanged, so a query still precedes its own
    #     response and rollouts still appear in job order.
    for kind in ("frames", "query", "response"):
        assert [json.dumps(r, sort_keys=True) for r in flat_recs if r["kind"] == kind] == \
               [json.dumps(r, sort_keys=True) for r in per_cand if r["kind"] == kind], (
            f"the {kind!r} records changed order")

    # (3) THE ONE ORDERING THAT DOES CHANGE, pinned rather than left implicit.
    #     Pass 1 prepares every candidate before pass 2 judges any, so ALL
    #     `frames` records precede ALL queries, where the per-candidate shape
    #     interleaves them (`ffffffqrqrqr...` against `ffqrqrffqrqr...`).
    #     That STRENGTHENS the property `bird/components/evaluation.py:1130`
    #     claims -- "written BEFORE the first query, so a run killed
    #     mid-judgment still says what its judge was looking at" -- from
    #     per-candidate to run-wide.
    kinds = "".join(r["kind"][0] for r in flat_recs)
    assert kinds == "f" * kinds.count("f") + kinds[kinds.count("f"):], (
        "a `frames` record was written after the first query; the two-pass "
        "split is supposed to make every candidate's frames precede all "
        "judging")

    assert [r.fitness for r in reports_a] == [r.fitness for r in reports_b]


@pytest.mark.parametrize("concurrent", [False, True])
def test_each_judgment_sees_its_own_candidates_reward_code(concurrent):
    """Every judge prompt carries the reward code of the candidate whose rollout
    it shows -- never the last candidate's.

    `_results` gives every candidate the SAME reward code, so no test above
    can see which candidate's code a prompt carried: an `_analyse` that read
    `rep` late from the loop (a late-binding closure) would give every
    judgment of every candidate the last one's code, reward trace and frame
    labels, while the parity test stayed green. Distinct code per
    candidate is what makes the pairing observable.
    """
    judge = _StubJudge()
    judge.concurrent_samples = concurrent
    ctx = _ctx("rda", judge,
               **{"evaluate.vlm.repeats": 2, "evaluate.rollouts_per_candidate": 2})
    results = _results(3)
    for i, res in enumerate(results):
        res.candidate.reward_code = (f"def reward(s, a, s2):\n"
                                     f"    return {i}.0  # MARK_CAND_{i}\n")
    state = type("S", (), {"iteration": 0, "subtasks": ["swing up", "balance"]})()
    registry.get("fitness_source", "vlm_score")(ctx, state, results)
    judged = [p for p in judge.calls if "MARK_CAND_" in p]
    marks = [re.findall(r"MARK_CAND_(\d)", p) for p in judged]
    assert all(len(set(m)) == 1 for m in marks), "a prompt carried two candidates' code"
    counts = {i: sum(1 for m in marks if m[0] == str(i)) for i in range(3)}
    assert counts == {0: 4, 1: 4, 2: 4}, counts
