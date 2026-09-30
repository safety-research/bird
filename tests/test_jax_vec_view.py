"""`_JaxVecEnvView`'s own machinery, on a test double.

WHAT A DOUBLE CAN AND CANNOT SETTLE HERE, because the distinction decides
what this file is worth. It can settle the VIEW's logic -- the action maps,
the per-row auto-reset, the clip accounting, the device boundary, the cache
wiring -- because those are the view's own arithmetic and a double exercises
them exactly as a real adapter would. It CANNOT settle whether the tier
produces the same learning as the numpy tier on a real env: that needs an
end-to-end training run, which this suite does not include, and proving the
view against a stub written alongside it would only show the stub agrees with
it.

A THIRD THING NOTHING HERE SETTLES, and it is not about the double: the
SEED ROW's jax block is plumbed but unasserted. Every test in this file
reads a VIEW ATTRIBUTE -- `v.xla_env`, `v.jax_cache`, `v.dlpack_zero_copy`,
`v.jax_version` -- and nothing reads the dict at
`fasttd3.py`'s `**({"view": "jax", ...})` that copies them onto the row.
Deleting that whole block would leave this file green. All six fields are
in that position, so "the tests pass" must not be read as "the seed row
carries what the row says it carries".

Three closers are rejected, each for a known reason: a test on
`v.<field>` alone passes with the
plumbing line deleted; a helper plus a test of the helper goes stale the
moment the call site is inlined again; an AST check over the source is
satisfied by a comment mentioning the attribute. The form that catches a
deleted plumbing line is ONE assertion in an end-to-end training, which reads
a real seed row and compares it against the view that produced it:

    row = <seed row from an end-to-end training>
    for k in ("xla_env", "jax_cache", "dlpack_zero_copy", "device_peak_mib",
              "jax_torch_device", "jax_version"):
        assert k in row, f"the jax seed-row block lost {k}"
    assert row["jax_version"] == jax.__version__
    assert "--xla_gpu_autotune_level=0" in row["xla_env"]["XLA_FLAGS"]

No test in this suite runs that training, so the gap is written down here
rather than closed.

Everything here is skipped without jax and a GPU rather than silently
weakened, and the skip says which.
"""
from __future__ import annotations

import os
import numpy as np
import pytest

jax = pytest.importorskip("jax")
# `importorskip("jax.numpy")` is a gate that can NEVER OPEN: uv.lock
# resolves no package of that name and none ever will -- it is a
# submodule, not a distribution -- so the case skipped in every
# configuration and covered nothing, while printing as a dot.
# `tests/test_no_silent_skip.py` refuses it.
import jax.numpy as jnp          # `jax` is importorskip'd above; this is its submodule
torch = pytest.importorskip("torch")

from bird.components import fasttd3 as FT           # noqa: E402
from bird.envs.jax_base import BatchedEnvAdapter    # noqa: E402

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or jax.devices()[0].platform != "gpu",
    reason="the jax view is cuda-only by design; a cpu run is a refusal")


class _Double(BatchedEnvAdapter):
    """A batched adapter whose dynamics are trivial and whose TERMINATION IS
    CONTROLLABLE, so the auto-reset can be tested at a row of our choosing.

    `step_batch` adds the action to the state and terminates row 0 whenever
    its first coordinate exceeds `term_at`. Pure, jittable, no host callback:
    a double that broke those rules would test `vmap` rather than the view.
    """

    physics = "mjx"
    horizon = 4
    obs_low = np.full(3, -10.0)
    obs_high = np.full(3, 10.0)
    action_low = np.array([-1.0, -1.0])
    action_high = np.array([1.0, 1.0])

    def __init__(self, term_at: float = 100.0, dr: bool = False) -> None:
        self.term_at = float(term_at)
        #: When true the adapter declares a DR axis, so `physics_rows` returns
        #: a real per-row pytree instead of None and the view must carry it.
        self.dr = bool(dr)
        #: Incremented on every `physics_rows` call, so a test can tell a
        #: REDRAW from a carried-over pytree by value AND by call count.
        self.draws = 0
        #: `_sample_dr` calls -- a REDRAW, as against a rebroadcast of the
        #: same `_dr_now`, which is the failure the view must avoid.
        self.samples = 0
        self._dr_now = {"gravity": 9.81}
        self._rng = None
        #: Set truthy to make `step_batch` report row 0 as nonfinite.
        self.blow_up = False
        #: What the last `step_batch` actually received. The point of the
        #: double is to witness the call, so this is the witness.
        self.last_physics = "NEVER CALLED"

    def reset_batch(self, key, n):
        return jnp.zeros((n, 3), dtype=jnp.float32)

    def physics_rows(self, n, draw=None):
        """`None` without DR; otherwise one `(n,)` leaf carrying the DRAW's
        own value, so a test sees WHICH draw a row holds rather than only
        that something changed."""
        if not self.dr:
            return None
        self.draws += 1
        d = self._dr_now if draw is None else draw
        return {"gravity": jnp.full((n,), float(d["gravity"]), dtype=jnp.float32)}

    def _sample_dr(self, rng):
        """A NEW draw each call, which is the property the view depends on.
        The real adapter's `_sample_dr` draws from `_dr_ranges`; this counts
        so a test can tell a resample from a rebroadcast."""
        self.samples += 1
        return {"gravity": 100.0 + self.samples}

    def step_batch(self, state, action, physics):
        # THREE REQUIRED POSITIONALS, no default -- a default turns
        # "not implemented" into "works". The double matches the contract
        # exactly, so a caller that omits the argument raises here instead of
        # silently getting None.
        #
        # And never `**kwargs`: a kwargs double absorbs the argument in
        # silence, which is how the view could stop passing it and every
        # test here stay green. Recording it is what makes the double a
        # witness rather than a sink.
        self.last_physics = physics
        pad = jnp.pad(jnp.asarray(action), ((0, 0), (0, 1)))
        s2 = jnp.asarray(state) + pad
        done = s2[:, 0] > self.term_at
        info = {}
        if self.blow_up:
            # Row 0 only, so a test can see the OTHER rows keep going.
            info["nonfinite"] = jnp.asarray(
                [True] + [False] * (s2.shape[0] - 1), dtype=bool)
        return s2, done, info


def _view(n=3, term_at=100.0, norm="none", dr=False):
    from bird.components.training import compile_reward
    reward = compile_reward(
        # NO `float()`: under a trace that raises, which is the contract
        # the jax reward path's `row()` enforces deliberately (a host cast is
        # what kills a trace). A jax-language reward returns the device scalar.
        "def compute_reward(state, action=None, next_state=None):\n"
        "    return next_state[0]\n", None)
    return FT._JaxVecEnvView(_Double(term_at, dr=dr), n, reward, None, norm,
                             seed=5, obs_width=3)


@needs_gpu
def test_the_action_map_matches_the_numpy_view_exactly():
    """+/-1 in agent space IS the env bound, and `clip_env_action` clamps --
    the same two rules `_VecEnvView` states, because a tier that mapped
    actions differently would be a different experiment wearing the same
    config."""
    v = _view()
    ones = np.ones_like(v._low)
    assert np.allclose(np.asarray(v.to_env_action(jnp.asarray(ones))), v._high)
    assert np.allclose(np.asarray(v.to_env_action(jnp.asarray(-ones))), v._low)
    # round trip
    a = jnp.asarray(np.array([[0.3, -0.7]], dtype=np.float32))
    assert np.allclose(np.asarray(v.to_agent_action(v.to_env_action(a))),
                       np.asarray(a), atol=1e-6)
    # clipping is a clamp, not a wrap or a scale
    far = jnp.asarray(np.array([[5.0, -5.0]], dtype=np.float32))
    assert np.allclose(np.asarray(v.clip_env_action(far)),
                       np.array([[1.0, -1.0]]))


@needs_gpu
def test_a_finished_row_auto_resets_and_true_next_is_the_terminal_state():
    """`_step_slot`'s rule, per row: the buffer gets the TERMINAL observation
    in `true_next` while `next_obs` is already the fresh episode. Getting this
    backwards is invisible in a reward curve and corrupts every bootstrap."""
    v = _view(n=2, term_at=0.5)
    v.reset()
    # row 0 steps past the termination threshold, row 1 does not
    acts = torch.as_tensor(np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)).cuda()
    next_obs, rewards, dones, time_outs, true_next = v.step(acts)
    d = np.asarray(dones.cpu())
    assert bool(d[0]) and not bool(d[1]), f"expected only row 0 done, got {d}"
    assert np.asarray(true_next.cpu())[0, 0] == pytest.approx(1.0), (
        "true_next must carry the TERMINAL state, not the reset one")
    assert np.asarray(next_obs.cpu())[0, 0] == pytest.approx(0.0), (
        "next_obs for a done row must be the fresh episode's first state")
    assert np.asarray(next_obs.cpu())[1, 0] == pytest.approx(0.0)


@needs_gpu
def test_truncation_is_not_termination_at_the_horizon():
    """`time_outs` is truncation-AND-NOT-termination: the learner bootstraps
    differently on the two, so a row that terminates on its horizon step is a
    termination and must not also be flagged a time-out."""
    v = _view(n=1, term_at=100.0)     # never terminates
    v.reset()
    zero = torch.zeros((1, 2), dtype=torch.float32).cuda()
    for _ in range(int(_Double.horizon) - 1):
        _, _, dones, time_outs, _ = v.step(zero)
        assert not bool(np.asarray(dones.cpu())[0])
    _, _, dones, time_outs, _ = v.step(zero)
    assert bool(np.asarray(dones.cpu())[0]), "horizon reached: done"
    assert bool(np.asarray(time_outs.cpu())[0]), "and it is a TRUNCATION"


@needs_gpu
def test_clip_counting_syncs_on_read_and_counts_its_reads():
    """The clip counter lives on device and is read per chunk, not per step.
    `n_clipped_reads` is on the seed row so a future caller that reads it in
    the hot loop shows up in the artifact rather than only in wall time."""
    v = _view(n=2)
    v.reset()
    big = torch.as_tensor(np.array([[5.0, 0.0], [0.0, 0.0]],
                                   dtype=np.float32)).cuda()
    before = v.n_clipped_reads
    v.step(big)
    assert v.n_clipped_reads == before, "step() must not sync the counter"
    assert v.n_clipped == 1, "one row clipped"
    assert v.n_clipped_reads == before + 1, "the read is counted"


@needs_gpu
def test_the_device_boundary_is_zero_copy_in_both_directions():
    """`dlpack_zero_copy` is filled from real pointer comparisons, per
    direction, and both must be True on this stack -- a field that says zero
    copy on belief alone is the kind that misleads."""
    v = _view(n=2)
    v.reset()
    v.step(torch.zeros((2, 2), dtype=torch.float32).cuda())
    assert v.dlpack_zero_copy["obs"] is True
    assert v.dlpack_zero_copy["actions"] is True


@needs_gpu
def test_reward_norm_none_does_no_host_round_trip():
    """Under `train.reward_norm: none` the rewards reach torch through DLPack and `reward_sync_s`
    stays exactly zero. Under any other mode the sync happens and is timed."""
    v = _view(n=2, norm="none")
    v.reset()
    v.step(torch.zeros((2, 2), dtype=torch.float32).cuda())
    assert v.reward_sync_s == 0.0

    w = _view(n=2, norm="running_std")
    w.reset()
    w.step(torch.zeros((2, 2), dtype=torch.float32).cuda())
    assert w.reward_sync_s > 0.0, "a normalised run pays a sync and says so"


@needs_gpu
def test_the_compilation_cache_is_shared_across_processes():
    """A second process building the same view HITS the cache the first
    filled, and `jax_cache` records it from jax's own counters.

    TWO PROCESSES, NOT TWO VIEWS IN ONE. The cache exists to survive a
    process, and within one process jax's in-memory cache would make a
    second view look like a hit whether or not anything reached disk --
    a test that cannot distinguish the two would pass with the persistent
    cache switched off entirely.

    HIT AND MISS COME FROM `/jax/compilation_cache/cache_{hits,misses}`,
    never from how long a compile took: inferring a hit from a short compile
    and then citing the field as evidence for the cache is circular.
    """
    import json
    import subprocess
    import sys
    import tempfile

    src = (
        "import json, os, sys\n"
        "os.environ['BIRD_JAX_CACHE_DIR'] = sys.argv[1]\n"
        # The double's compiles take milliseconds, far under the 1.0 s
        # production threshold, so without this the cache declines to store
        # them and the run reports {hits: 0, misses: 0} -- which is the
        # correct reading of "nothing was worth caching" and tests nothing.
        # Lowering the floor is what makes the MECHANISM observable on a toy.
        "os.environ['BIRD_JAX_CACHE_MIN_COMPILE_S'] = '0.0'\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from tests.test_jax_vec_view import _view\n"
        "import torch\n"
        "v = _view(n=2)\n"
        "v.reset()\n"
        "v.step(torch.zeros((2, 2), dtype=torch.float32).cuda())\n"
        "print(json.dumps(v.jax_cache))\n"
    )
    repo = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    with tempfile.TemporaryDirectory() as cache:
        runs = []
        for _ in range(2):
            out = subprocess.run([sys.executable, "-c", src, cache, repo],
                                 capture_output=True, text=True, timeout=600)
            assert out.returncode == 0, out.stderr[-2000:]
            runs.append(json.loads(out.stdout.strip().splitlines()[-1]))

        cold, warm = runs
        assert cold["dir"] == cache and warm["dir"] == cache
        assert cold["misses"] > 0, (
            f"the first process must MISS and fill the cache: {cold}")
        assert warm["hits"] > 0, (
            f"the second process must HIT what the first wrote: {warm}. "
            f"A zero here with a non-empty dir means the key is varying "
            f"between processes -- the failure this test exists for.")


@needs_gpu
@pytest.mark.parametrize("preset,want_kept", [
    # An UNRELATED flag already in XLA_FLAGS. This is the case a
    # `setdefault` silently loses: the variable is non-empty, so
    # `setdefault` returns it untouched, the process keeps autotuning, and
    # a rerun-equality check becomes a coin flip about a third of the time.
    ("--xla_force_host_platform_device_count=1", True),
    # An operator who has already pinned a level keeps it. Our append must
    # not argue with a deliberate choice, and level 2 is not level 0, so a
    # run under this preset is NOT entitled to the equality -- which is why
    # the effective string goes on the seed row rather than a boolean.
    ("--xla_gpu_autotune_level=2", False),
])
def test_the_autotune_flag_is_appended_to_XLA_FLAGS_not_setdefault(preset, want_kept):
    """`XLA_FLAGS` is ONE string holding every XLA flag, so `setdefault` is
    the wrong verb and the difference is invisible at the call site.

    `--xla_gpu_autotune_level=0` is what makes the batched step bit-identical
    across launches (measured over six processes: 2 distinct results, spread
    2.03e-04, default; 1 distinct, spread 0.0, autotune off). A test that is
    an equality against a rerun rests entirely on the flag actually reaching
    XLA -- so the failure this guards is not a crash, it is an equality that
    passes four times and fails the fifth while every counter reads normal.

    A SUBPROCESS per case, because the flags are read at BACKEND
    INITIALISATION, once per process: an in-process test would measure
    whichever earlier test built a view first.
    """
    import json
    import subprocess
    import sys

    src = (
        "import json, os, sys\n"
        "os.environ['XLA_FLAGS'] = sys.argv[1]\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from tests.test_jax_vec_view import _view\n"
        "v = _view(n=2)\n"
        # BOTH: what the process has, and what the seed row will say. They
        # must agree -- a seed row that records a flag the process does not
        # have is the exact failure the read-back exists to prevent.
        "import jax\n"
        "print(json.dumps({'environ': os.environ['XLA_FLAGS'],\n"
        "                  'seed_row': v.xla_env.get('XLA_FLAGS'),\n"
        "                  'jax_version': v.jax_version,\n"
        "                  'installed': jax.__version__}))\n"
    )
    repo = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    out = subprocess.run([sys.executable, "-c", src, preset, repo],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])

    assert preset in got["environ"], (
        f"the pre-existing flag was destroyed: {got['environ']!r}. XLA_FLAGS "
        f"is one shared string and this view is not its only writer.")
    if want_kept:
        assert "--xla_gpu_autotune_level=0" in got["environ"], (
            f"the autotune flag never reached XLA: {got['environ']!r}. A "
            f"`setdefault` on a non-empty XLA_FLAGS produces exactly this.")
    else:
        assert "--xla_gpu_autotune_level=2" in got["environ"], (
            f"an operator's explicit level was overridden: {got['environ']!r}")
        assert "--xla_gpu_autotune_level=0" not in got["environ"], (
            f"both levels are present, so which one XLA honours is a parse "
            f"order nobody here has measured: {got['environ']!r}")

    assert got["seed_row"] == got["environ"], (
        f"the seed row must record the effective string, read back from the "
        f"process, not the one the view meant to build: "
        f"{got['seed_row']!r} vs {got['environ']!r}")

    # THE INSTALLED VERSION, NOT THE PIN. The tier pins jax <= 0.8.0 and a
    # venv can hold a later jax (0.10.2 in practice), so a reproducibility claim that does not name
    # the version is scoped to nothing -- XLA ships inside jaxlib and kernel
    # selection is its business. Asserted in the same subprocess because that
    # is the process whose flags were just measured.
    assert got["jax_version"] == got["installed"] and got["jax_version"], (
        f"the view must record the jax it is actually running under: "
        f"{got['jax_version']!r} vs {got['installed']!r}")


@needs_gpu
def test_the_per_row_physics_pytree_is_passed_and_redrawn_on_reset():
    """`step_batch`'s third argument, end to end: carried, passed, and redrawn PER ROW.

    WHY THIS TEST EXISTS AT ALL. `step_batch`'s `physics` parameter has a
    DEFAULT of `None`, so a view that never passes it compiles, runs, and
    passes every other test in this file while every DR family trains on the
    adapter's default physics. A keyword default turns "not implemented" into
    "works", so the wiring needs a test that goes red when it is deleted --
    which the whole rest of this file does not.

    The double's `physics_rows` returns the DRAW INDEX as its leaf value, so
    "which rows hold which draw" is readable rather than inferred.
    """
    v = _view(n=3, term_at=0.5, dr=True)
    env = v.env

    # Carried from construction, not first use: draw 1, every row.
    assert env.draws == 1
    assert np.allclose(np.asarray(v._physics["gravity"]), 9.81)

    # PASSED, and passed as the pytree we hold -- not None, not a default.
    v.step(torch.zeros((3, 2), dtype=torch.float32).cuda())
    assert env.last_physics is not None, (
        "step_batch received None: the view is not passing the pytree, which "
        "the default makes invisible everywhere else")
    assert np.allclose(np.asarray(env.last_physics["gravity"]), 9.81)

    # Terminate ROW 0 ONLY. `term_at=0.5` and +1 on the first coordinate, so
    # a +1 action on row 0 alone crosses it.
    act = torch.zeros((3, 2), dtype=torch.float32).cuda()
    act[0, 0] = 1.0
    v.step(act)

    phys = np.asarray(v._physics["gravity"])
    assert phys[0] == 101.0, (
        f"row 0 auto-reset and must hold the NEW draw (_sample_dr -> 101.0), "
        f"got {phys[0]}: a row that keeps its old draw is DR that randomises "
        f"once and then stops")
    assert phys[1] == 9.81 and phys[2] == 9.81, (
        f"rows 1 and 2 did not reset and must keep the original draw, got "
        f"{phys[1:]}: a whole-batch redraw would re-randomise mid-episode")
    assert env.samples == 1, (
        f"exactly one resample expected for one resetting row, saw "
        f"{env.samples}")

    # NO RESET, NO REDRAW -- the `any(done)` gate. Without it the host draw
    # is paid on every step of every training.
    v.step(torch.zeros((3, 2), dtype=torch.float32).cuda())
    assert env.samples == 1, (
        f"_sample_dr was called with no row done ({env.samples} samples): "
        f"the any(done) gate is not gating")


@needs_gpu
def test_a_family_with_no_dr_axes_passes_None_rather_than_skipping_the_argument():
    """`None` is the no-DR value and must still be PASSED.

    A family with no DR axes is why an end-to-end run cannot catch a missing
    third argument: on such an env, passing None and passing nothing are
    indistinguishable in behaviour. They are not indistinguishable here.
    """
    v = _view(n=2, dr=False)
    assert v._physics is None
    v.step(torch.zeros((2, 2), dtype=torch.float32).cuda())
    # `is None` carries the whole claim BECAUSE of the sentinel: the double
    # starts at "NEVER CALLED", so this passes only if step_batch was really
    # called and really received None.
    assert v.env.last_physics is None
    assert v.env.draws == 0


@needs_gpu
def test_the_device_agreement_says_skipped_rather_than_leaving_a_null():
    """`jax_pci: null` inside a populated dict reads as "checked and equal".

    That is the `learner_device: null` shape exactly -- a null meaning "not
    measured" sitting where a measurement goes, one key deep in a dict whose
    other three values are real. Such a null is easily read as evidence of a
    CPU run. So the row says which of the two states it is, in a key whose
    value cannot be mistaken for a measurement.
    """
    v = _view(n=2)
    d = v.device_check
    assert d["status"] in ("compared", "skipped"), d
    if d["status"] == "skipped":
        assert not (d["torch_pci"] and d["jax_pci"]), (
            f"status says skipped but both bus ids are present: {d}")
        assert d["skipped_because"], "a skip must carry its reason"
        assert "NOT compared" in d["skipped_because"], d["skipped_because"]
    else:
        assert d["torch_pci"] and d["jax_pci"], (
            f"status says compared but a bus id is missing: {d}")
        assert d["skipped_because"] == "", d


@needs_gpu
def test_a_nonfinite_row_is_terminated_and_counted_not_stepped_on():
    """The blow-up guard lives OUTSIDE `done`, on the promise that the view
    terminates the row.

    A view that never reads the flag keeps a row whose physics went nonfinite
    stepping on NaNs for the rest of the episode and writes every one of those
    transitions into the buffer. Nothing fails and nothing is logged: the
    numbers are finite-shaped garbage. So this asserts the three things the promise consists of --
    terminated, auto-reset, counted.

    TERMINATED and not truncated, because the learner must not bootstrap
    from a blown-up state: that value is meaningless and bootstrapping it
    walks the blow-up into the critic.
    """
    v = _view(n=3, term_at=1e9)          # nothing terminates on its own
    v.env.blow_up = True
    _obs, _r, dones, time_outs, _tn = v.step(
        torch.zeros((3, 2), dtype=torch.float32).cuda())

    d, t = dones.cpu().numpy(), time_outs.cpu().numpy()
    assert bool(d[0]), "the nonfinite row must be done"
    assert not bool(t[0]), (
        "the nonfinite row must be TERMINATED, not truncated: a truncation "
        "tells the learner to bootstrap from a blown-up state")
    assert not d[1] and not d[2], f"only row 0 blew up, got dones {d}"
    assert v.nonfinite_rows == 1, (
        f"nonfinite_rows={v.nonfinite_rows}: the count is what stops this "
        f"reading as zero-by-omission on the seed row")


@needs_gpu
def test_the_dr_redraw_is_a_NEW_draw_and_not_the_same_dict_again():
    """`physics_rows(n)` broadcasts the adapter's CURRENT `_dr_now`, so
    calling it again on reset returns the SAME values.

    That is domain randomisation which randomises once, at construction, and
    never again -- while `dr_parameters` and every other field say DR is on.
    The view must ask the adapter for a
    fresh per-episode draw, `_sample_dr`, exactly as the adapter's own
    `reset()` does, so a row that resets here gets the draw it would have
    got there.
    """
    v = _view(n=3, term_at=0.5, dr=True)
    env = v.env
    before = float(np.asarray(v._physics["gravity"])[0])
    samples_before = env.samples

    act = torch.zeros((3, 2), dtype=torch.float32).cuda()
    act[0, 0] = 1.0                       # row 0 alone crosses term_at
    v.step(act)

    phys = np.asarray(v._physics["gravity"])
    assert env.samples == samples_before + 1, (
        f"_sample_dr was called {env.samples - samples_before} times for one "
        f"resetting row: a redraw that never samples is a rebroadcast")
    assert phys[0] != before, (
        f"row 0 reset and still holds {phys[0]}: the redraw did not take")
    assert phys[1] == before and phys[2] == before, (
        f"rows 1 and 2 did not reset and must keep their draw, got {phys[1:]}")


@needs_gpu
def test_reset_batch_is_not_called_when_no_row_resets():
    """Calling `reset_batch` on EVERY step builds a batch `_tail` discards
    whenever nothing terminated.

    On an MJX adapter that is a host oracle draw plus a host-to-device
    transfer per step, for rows that reset a few times in a thousand. The
    select is `where(keep, s2, fresh)` and `keep` is all-True when nothing is
    done, so `fresh` is never read -- passing `s2` is the same arithmetic
    without the draw.
    """
    v = _view(n=3, term_at=1e9)
    env = v.env
    calls = {"n": 0}
    real = env.reset_batch
    env.reset_batch = lambda key, n: calls.__setitem__("n", calls["n"] + 1) or real(key, n)

    for _ in range(3):
        v.step(torch.zeros((3, 2), dtype=torch.float32).cuda())
    assert calls["n"] == 0, (
        f"reset_batch was called {calls['n']} times with no row done")

    v2 = _view(n=3, term_at=0.5)
    env2 = v2.env
    calls2 = {"n": 0}
    real2 = env2.reset_batch
    env2.reset_batch = lambda key, n: calls2.__setitem__("n", calls2["n"] + 1) or real2(key, n)
    act = torch.zeros((3, 2), dtype=torch.float32).cuda()
    act[0, 0] = 1.0
    v2.step(act)
    assert calls2["n"] == 1, (
        f"reset_batch must still be called when a row DOES reset; got "
        f"{calls2['n']} -- gating it away entirely would leave a terminated "
        f"row stepping from its terminal state")


def test_the_autotune_flag_has_a_process_entry_point_that_refuses_late():
    """Setting `XLA_FLAGS` in the view's `__init__` is TOO LATE in a real run.

    An MJX adapter calls `mjx.put_model` in its own `__init__`, which
    initialises the XLA backend before any view exists, so an append in the
    view changes nothing while `xla_env` reports the flag present. A flag
    recorded as present but applied late is worse than an absent one,
    because it is quoted in a reproducibility claim.

    No GPU needed: this is about import order and an environment variable.
    """
    import importlib
    import sys as _sys
    FT2 = importlib.import_module("bird.components.fasttd3")

    if "jax" in _sys.modules:
        with pytest.raises(RuntimeError, match="after jax was imported"):
            FT2.set_xla_determinism_flags(strict=True)
        assert FT2.set_xla_determinism_flags(strict=False) is False
    else:
        assert FT2.set_xla_determinism_flags(strict=True) is True
    assert "--xla_gpu_autotune_level" in os.environ.get("XLA_FLAGS", "")
