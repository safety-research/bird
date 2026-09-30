"""`output.video.timeout_s`: a wedged renderer must not take the search with it.

A renderer can wedge on one compute node while the same GL path works on others:
the job stops one log line short of `record_rollouts`, produces nothing for many
minutes, writes no `videos/`, and is torn down by the batch scheduler with no
traceback while `status.json` still reads `running`. What prevents that is a
bound on how long recording may take.

Both halves are pinned here. The deadline checked between frames -- covering
the WHOLE call rather than each candidate, and visible in the journal
afterwards -- is the half Python can enforce. The faulthandler watchdog is the
half that actually diagnoses a render wedged in native code, and it is testable despite
firing when no Python in this process runs: its timer lives in a C thread, so a
subprocess that blocks inside a GIL-holding C call gets its dump and exits
normally. Untested, it degrades to a `log.debug` (no `fileno` on stderr, a
grace raised past the point the scheduler kills the step) in exactly the run where
it is the only diagnostic there is.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from bird import observability
from bird.artifacts import RunDir
from bird.budget import Budget
from bird.components import generation
from bird.config import load
from bird.context import Context
from bird.observability import record_rollouts
from bird.types import Candidate, CandidateReport, Trajectory, TrainResult

#: Not in the tester-tier smoke suite (rendering: real subprocesses and real timeouts).
#: Deselected in CI by `-m "not slow"`; run it with `-m slow`. See pyproject.toml.
pytestmark = pytest.mark.slow

#: Per-frame render cost. Tens of milliseconds, not seconds: the guard is being
#: tested, not the renderer, and the suite runs on every commit.
FRAME_S = 0.02


class SlowEnv:
    """An env whose `render` returns -- always -- just too slowly to finish.

    Deliberately NOT a hang: a hang is the case only the faulthandler watchdog
    can see, and it is reproduced in a subprocess below rather than here, where
    it would wedge the suite exactly the way a wedged renderer stalls a job.
    """

    def __init__(self, per_frame_s: float = FRAME_S):
        self.per_frame_s = per_frame_s
        self.calls = 0

    def render(self, state):
        self.calls += 1
        time.sleep(self.per_frame_s)
        return np.zeros((8, 8, 3), dtype=np.uint8)


def _report(i: int) -> CandidateReport:
    cand = Candidate(cand_id=f"c{i}", iteration=0, reward_code="def compute_reward(): ...")
    traj = Trajectory(states=[[0.0], [1.0], [2.0], [3.0]], rewards=[0.0] * 4, length=4)
    result = TrainResult(cand_id=cand.cand_id, candidate=cand, trained=True,
                         trajectories=[traj])
    return CandidateReport(cand_id=cand.cand_id, candidate=cand, result=result,
                           fitness=float(i))


def _ctx(tmp_path, timeout_s, n_frames=2, env=None) -> Context:
    cfg = load("eureka", profile="tester", overrides={
        "output.video.enabled": True,
        "output.video.record": "all",
        "output.video.format": "frames",
        "output.video.max_frames": n_frames,
        "output.video.width": 8,
        "output.video.timeout_s": timeout_s,
    })
    return Context(cfg=cfg, budget=Budget(), rundir=RunDir(tmp_path, "vid", "hash0"),
                   env=env if env is not None else SlowEnv())


def _events(ctx, stage: str) -> list:
    lines = (ctx.rundir.path / "journal.jsonl").read_text().splitlines()
    return [e for e in (json.loads(l) for l in lines) if e.get("stage") == stage]


def test_a_slow_renderer_is_abandoned_and_the_run_continues(tmp_path):
    """The failure mode the guard exists for, minus the uninterruptible part."""
    ctx = _ctx(tmp_path, timeout_s=FRAME_S, n_frames=8)
    reports = [_report(i) for i in range(4)]
    written = record_rollouts(ctx, reports)

    assert len(written) < 4, "the deadline elapsed and nothing stopped recording"
    assert all(r.path.exists() for r in written), (
        "abandoning must not leave a half-written recording in the return value")
    # The run goes on: every report the deadline cut short is scored without
    # frames, exactly as it is when an env has no `render` at all.
    recorded = {r.cand_id for r in written}
    unreached = [rep for rep in reports if rep.cand_id not in recorded]
    assert unreached and all(rep.result.trajectories[0].video_path is None
                             for rep in unreached)


def test_the_abandonment_reaches_the_journal(tmp_path):
    """A run that quietly stopped producing footage looks complete otherwise.

    Same reasoning as `record_skipped`: a VLM-graded method whose frames went
    missing falls through to a text-only comparison and returns a fitness that
    reads exactly like one produced with eyes.
    """
    ctx = _ctx(tmp_path, timeout_s=FRAME_S, n_frames=8)
    written = record_rollouts(ctx, [_report(i) for i in range(4)])

    events = _events(ctx, "record_timeout")
    assert len(events) == 1, "exactly one whole-call event, not one per candidate"
    ev = events[0]
    assert ev["timeout_s"] == pytest.approx(FRAME_S)
    assert ev["elapsed_s"] >= FRAME_S
    assert ev["n_abandoned"] >= 1
    assert ev["n_recorded"] == len(written)
    assert ev["n_frames"] >= 1, (
        "the tally must include frames rendered for the pick that was abandoned "
        "mid-trajectory; those are discarded and invisible to the caller")


def test_the_deadline_covers_the_whole_call_not_each_candidate(tmp_path):
    """`record: all` with N candidates must not multiply the cap by N.

    One candidate renders comfortably inside the cap; eight of the same
    candidate do not. A per-candidate deadline would pass both.
    """
    cap = 8 * FRAME_S  # 4x one candidate's two frames, 0.5x the eight-candidate call

    solo = _ctx(tmp_path / "solo", timeout_s=cap, n_frames=2)
    assert len(record_rollouts(solo, [_report(0)])) == 1, (
        "a single candidate is well inside the cap; the guard must not fire")
    assert not _events(solo, "record_timeout")

    many = _ctx(tmp_path / "many", timeout_s=cap, n_frames=2)
    written = record_rollouts(many, [_report(i) for i in range(8)])
    assert len(written) < 8
    assert _events(many, "record_timeout")


def test_no_timeout_records_normally(tmp_path):
    """`null` is the off switch, and off means the previous behaviour exactly."""
    env = SlowEnv(per_frame_s=0.001)
    ctx = _ctx(tmp_path, timeout_s=None, n_frames=3, env=env)
    written = record_rollouts(ctx, [_report(i) for i in range(3)])

    assert len(written) == 3
    assert env.calls == 9
    assert not _events(ctx, "record_timeout")
    assert len(_events(ctx, "record")) == 3


def test_a_generous_timeout_changes_nothing(tmp_path):
    """The default is a wedge detector, so a healthy render must not notice it."""
    env = SlowEnv(per_frame_s=0.001)
    ctx = _ctx(tmp_path, timeout_s=30, n_frames=3, env=env)
    assert len(record_rollouts(ctx, [_report(i) for i in range(3)])) == 3
    assert not _events(ctx, "record_timeout")


# --- the faulthandler half -------------------------------------------------
#
# Everything below pins `_arm_stack_dump`, which is the ONLY thing that produces
# a diagnosis of a render wedged in native code and the only thing whose failure mode is
# a `log.debug`. Nothing else in the suite can see it: the arming call already
# runs during the tests above, but its return value is never read, so both
# degrade branches could fire for the whole suite and every test stays green.


@pytest.fixture
def watchdog_calls(tmp_path, monkeypatch):
    """Record arm/cancel instead of really scheduling a process-global timer.

    `sys.stderr` is pointed at a real file because `_arm_stack_dump` gates on
    `fileno()` and pytest's own capture object does not always have one -- the
    gate is asserted on purpose in its own test below, so it must not silently
    decide these.
    """
    calls: list = []
    monkeypatch.setattr(observability.sys, "stderr", (tmp_path / "err").open("w"))
    monkeypatch.setattr(observability.faulthandler, "dump_traceback_later",
                        lambda t, exit=True: calls.append(("arm", t, exit)))
    monkeypatch.setattr(observability.faulthandler, "cancel_dump_traceback_later",
                        lambda: calls.append(("cancel",)))
    return calls


def test_the_watchdog_is_armed_past_the_deadline_and_cancelled_after(tmp_path, watchdog_calls):
    """The grace is what makes a dump mean something; `exit=False` is the contract.

    A grace of zero would dump on every clean abandonment. Raising it past the
    point the scheduler kills the step would mean it never dumps at all -- and
    neither mistake changes any other assertion in this file.
    """
    ctx = _ctx(tmp_path, timeout_s=30, n_frames=1, env=SlowEnv(per_frame_s=0.001))
    record_rollouts(ctx, [_report(0)])

    assert watchdog_calls == [("arm", 30 + observability._WATCHDOG_GRACE_S, False),
                              ("cancel",)]
    assert 0 < observability._WATCHDOG_GRACE_S <= 60, (
        "the grace has to fit inside the teardown window it is racing: a batch "
        "scheduler's kill wait is typically 30 s, so a grace of minutes is a watchdog "
        "that dumps after the step is already gone")


def test_the_watchdog_is_cancelled_on_the_deadline_path_too(tmp_path, watchdog_calls):
    """The early `return written` must not leak a live process-global timer.

    One is all there is, so a leaked one dumps every thread's stack in the
    middle of the NEXT iteration's training -- a wedge report about a run that
    is not wedged.
    """
    ctx = _ctx(tmp_path, timeout_s=FRAME_S, n_frames=8)
    record_rollouts(ctx, [_report(i) for i in range(4)])

    assert _events(ctx, "record_timeout"), "the deadline did not fire; wrong path"
    assert watchdog_calls[-1] == ("cancel",)
    assert [c[0] for c in watchdog_calls].count("arm") == 1


def test_null_arms_nothing(tmp_path, watchdog_calls):
    """`null` is the off switch for BOTH halves, not only the deadline."""
    ctx = _ctx(tmp_path, timeout_s=None, n_frames=2, env=SlowEnv(per_frame_s=0.001))
    record_rollouts(ctx, [_report(0)])
    assert watchdog_calls == []


def test_stderr_without_a_fileno_degrades_rather_than_raising(tmp_path, monkeypatch):
    """faulthandler needs a real fd; capture plumbing can leave stderr without one.

    The recording still has to happen -- a diagnostic that cannot arm must not
    take the video with it.
    """
    class NoFd:
        def write(self, s): return len(s)
        def flush(self): pass
        def fileno(self): raise OSError("no fd here")

    monkeypatch.setattr(observability.sys, "stderr", NoFd())
    monkeypatch.setattr(observability.faulthandler, "dump_traceback_later",
                        lambda *a, **k: pytest.fail("armed with no usable stderr"))
    assert observability._arm_stack_dump(1.0) is False

    ctx = _ctx(tmp_path, timeout_s=30, n_frames=2, env=SlowEnv(per_frame_s=0.001))
    assert len(record_rollouts(ctx, [_report(0)])) == 1


def test_the_generate_stage_render_is_watched_too(tmp_path, watchdog_calls):
    """`task_images()` renders, and on metaworld/rda it is the run's FIRST render.

    gymnasium builds `MujocoRenderer` lazily, so GL context creation happens
    there rather than in `record_rollouts` -- i.e. the likeliest place to wedge
    is in stage 1, where none of the assertions above reach. The production
    caller of `_images_for` is the DECOMPOSE phase rather than `backend_llm`
    (App. 7.2/7.5 take no image); it is still stage 1 and still the run's first
    render, and this test calls the function directly either way.
    """
    seen: list = []

    class ImageEnv:
        def task_images(self):
            seen.append(list(watchdog_calls))   # armed BEFORE the render, not after
            return [b"\x89PNG"]

    cfg = load("eureka", profile="tester", overrides={"problem.instruction_modality": "text+image",
                                           "output.video.timeout_s": 5})
    ctx = Context(cfg=cfg, budget=Budget(), env=ImageEnv())

    assert generation._images_for(ctx) == [b"\x89PNG"]
    assert seen == [[("arm", 5.0, False)]], "the generate-stage render is unguarded"
    assert watchdog_calls[-1] == ("cancel",)


#: The subprocess blocks inside `sleep(3)` called through `ctypes.PyDLL`, which
#: holds the GIL for its whole duration -- no bytecode runs, which is what a
#: wedged MuJoCo EGL call does and why the between-frames deadline cannot see
#: it. Short enough that this test costs ~1 s.
_WEDGED = """
import ctypes, sys
sys.path.insert(0, {root!r})
from bird.observability import render_watchdog
with render_watchdog(0.3):
    pass                             # returns at once; the timer must be cancelled
ctypes.PyDLL(None).sleep(1)          # an uncancelled timer would report THIS block
with render_watchdog(0.3):
    ctypes.PyDLL(None).sleep(1)      # the wedge: GIL held, no Python frames run
print("returned")
"""


def test_a_gil_holding_wedge_really_does_get_dumped():
    """The claim the whole watchdog rests on, executed rather than asserted about.

    Two windows, because `dump_traceback_later` is one-shot: the first is armed
    and left, so a missing cancel reports the innocent block between them and
    the dump count goes to two.
    """
    root = str(Path(observability.__file__).resolve().parents[1])
    proc = subprocess.run([sys.executable, "-c", _WEDGED.format(root=root)],
                          capture_output=True, text=True, timeout=60)

    assert proc.returncode == 0 and "returned" in proc.stdout, (
        "exit=False: the watchdog diagnoses, it must never kill the process")
    assert proc.stderr.count("Timeout (") == 1, (
        f"expected exactly one dump (armed, then cancelled); got:\n{proc.stderr}")
    assert "ctypes" in proc.stderr or "<string>" in proc.stderr, (
        f"the dump must name the frame the main thread is stuck in:\n{proc.stderr}")
