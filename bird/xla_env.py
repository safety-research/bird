"""XLA environment variables that must be set before the backend initialises.

TORCH-FREE AND DEPENDENCY-FREE ON PURPOSE. This is imported from the env
construction site, which every run passes through on every tier, so it must
not drag `torch` (or jax) in behind it -- `bird/components/fasttd3.py`, the
natural home for this logic, imports torch at module scope.

WHY IT CANNOT LIVE WITH ITS CALLER. XLA reads these variables at BACKEND
INITIALISATION -- the first jax operation in the process -- and on the jax
tier that is the ADAPTER's `mjx.put_model`, inside its `__init__`. Anything
set after the adapter is built is inert -- setting the flag in
`_JaxVecEnvView.__init__`, which runs later still, would leave it never in
effect in a real run while the seed row faithfully reported it present. A
flag recorded as applied but not in force is worse than an absent one,
because it is quoted in a reproducibility claim.
"""

from __future__ import annotations

import os
import sys
from typing import Any

#: Autotuning is what makes the batched step non-reproducible across launches.
#: Measured over six processes: with default flags the
#: n>=4 step returns one of TWO results per launch, spread 2.03e-04; with
#: `--xla_gpu_autotune_level=0` all six agree exactly, spread 0.0.
#: `--xla_gpu_deterministic_ops=true` and a `highest` matmul precision were
#: NOT needed.
AUTOTUNE_OFF = "--xla_gpu_autotune_level=0"

#: Whether the flags were set while jax was NOT yet imported -- recorded the
#: first time anyone sets them, PROCESS-WIDE.
#:
#: IT HAS TO BE RECORDED RATHER THAN RE-ASKED, or a correct setup reads
#: `False`. The view also
#: calls `set_xla_determinism_flags(strict=False)`, and by then the adapter
#: has imported jax -- so a view that reported ITS OWN call's verdict would
#: say "not in time" on every real run, no matter how early the process
#: entry point set the flag. The question the seed row asks is "was this
#: process's flag set in time", and only the FIRST call can answer it.
_APPLIED_IN_TIME: "bool | None" = None


#: The MJX zero-width-contracting-dim dot needs Triton's GEMM path OFF.
#: NOT `--xla_gpu_triton_gemm_any=True`, a look-alike that enables Triton for
#: MORE shapes -- the opposite of what this needs.
TRITON_GEMM_OFF = "--xla_gpu_enable_triton_gemm=false"


def set_xla_determinism_flags(strict: bool = True) -> bool:
    """Append `--xla_gpu_autotune_level=0` to `XLA_FLAGS`. Call before the env.

    Returns True if it was set while jax was NOT yet imported, i.e. while it
    can still affect backend initialisation. That boolean goes on the seed
    row: a row with False must not be quoted in an equality claim.

    APPENDED, NOT `setdefault`. `XLA_FLAGS` is one space-separated string
    holding every XLA flag, so `setdefault` is a no-op the moment anything
    else has set a single unrelated flag -- the process would keep autotuning
    while the row recorded the flag as present. An operator who has already
    pinned an autotune level keeps theirs.

    `strict=True` RAISES when jax is already imported, which is what the
    construction site wants: the run is about to produce numbers under a flag
    that is not in force. The view passes `strict=False`, because by then it
    is reporting rather than fixing.
    """
    in_time = "jax" not in sys.modules
    if strict and not in_time:
        raise RuntimeError(
            "set_xla_determinism_flags() called after jax was imported: XLA "
            "reads XLA_FLAGS at backend initialisation, so the flag would be "
            "inert while every seed row reported it present. It must be set "
            "before the env is constructed -- the adapter's mjx.put_model "
            "initialises the backend -- or exported in the environment.")
    global _APPLIED_IN_TIME
    if _APPLIED_IN_TIME is None:
        _APPLIED_IN_TIME = in_time
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_autotune_level" not in flags:
        flags = (flags + " " + AUTOTUNE_OFF).strip()
    # THE TRITON GEMM REFUSAL, and the flag name is exact rather than
    # approximate. MJX emits a dot whose contracting dimension is ZERO WIDE
    # when a scene has no constraints -- (0x30)*(30) on scratchitch -- and
    # Triton's GEMM lowering refuses it outright rather than returning an
    # empty result, so the first step dies with a kernel error naming no
    # environment and no candidate. Disabling Triton's GEMM path sends that
    # dot to the default lowering, which handles the zero-width case.
    # Diagnosed from StableHLO on CPU; identical with and without
    # homogenisation, so it is upstream physics rather than our port.
    #
    # Appended on the same terms as the autotune level: only when absent, so
    # an operator who has pinned it keeps their value.
    if "--xla_gpu_enable_triton_gemm" not in flags:
        flags = (flags + " " + TRITON_GEMM_OFF).strip()
    os.environ["XLA_FLAGS"] = flags
    return in_time


def prepare_for_env(env_id: Any, *, strict: bool = True) -> bool:
    """Set the determinism flags iff `env_id` is on the jax tier. Returns
    True when they were set in time, False when the tier does not need them.

    `strict=False` FOR THE CALL INSIDE ENV CONSTRUCTION, and the asymmetry is
    deliberate. The two entry-point calls (`bird.py::run`, the spawn child)
    run before anything in the process has touched jax, so a late one there
    means the RUN is about to produce numbers under an inert flag and raising
    is right. The registered env factories call this too -- they are the one
    place EVERY construction passes through, including the paths neither
    entry point covers -- but by then raising would be wrong twice over: it
    would turn a reproducibility warning into a dead candidate from inside
    `__init__`, and it would fail every jax-tier test whose fixture does
    `pytest.importorskip("jax")` before constructing an adapter. What the
    late call can still do, it does: `XLA_FLAGS` and (with it)
    `XLA_PYTHON_CLIENT_PREALLOCATE` get set, and the honest record of
    lateness stays where it has always been -- `applied_in_time()`, which
    reports the FIRST call's verdict and therefore still reads False on a
    process that only ever asked too late.

    Idempotent for the same reason: the flag append is conditional, the
    preallocate setting is a `setdefault`, and `_APPLIED_IN_TIME` is written
    once, so the factory call is a no-op in the normal case where an entry
    point already ran.

    THE GATE IS A FUNCTION SO THAT ANYTHING PRODUCING EVIDENCE CALLS THE SAME
    CODE THE RUN DOES. Were it inline in `bird.py`, a test could only
    reproduce it -- and a test that reproduces a call site proves the test
    works, not the call site. A caller should set no `XLA_FLAGS` of its own;
    it calls this.

    Gated on the SUITE rather than set unconditionally: `XLA_FLAGS` is
    process-global and a numpy-tier run has no business changing kernel
    selection for anything else in the process.
    """
    from bird.envs.suites import env_suite
    if env_suite(env_id) != "jax":
        return False

    # -- EVERY BIRD PROCESS THAT TOUCHES JAX STOPS PREALLOCATING ----------
    # jax grabs ~75% of the device at first use. MEASURED on an A100 40GB:
    # the PARENT sat on 30.17 of 39.49 GiB, the
    # first spawned worker got 8.81 GiB and the second found 2.75 MiB free
    # and died -- an out-of-memory failure that looked like a worker bug and
    # was a default. The parent's own measured jax need is 1-30 MiB
    # (`device_peak_mib.jax` on the seed rows), so the 30 GiB is reserved and
    # unused.
    #
    # HERE rather than in the view: `fasttd3.py`'s setdefault already makes
    # this the repo's policy, but it runs when a VIEW is built, which is
    # far too late for a parent that constructs the env -- and takes the
    # device -- before any view exists. This function is the one place that
    # runs at env construction and before jax is imported, which is the only
    # moment the variable still bites.
    #
    # `setdefault`, not assignment: an operator who has deliberately set it
    # keeps their value, which is the same contract `fasttd3.py` uses.
    # `XLA_PYTHON_CLIENT_MEM_FRACTION` is NOT set here -- it caps on-demand
    # growth and stays the worker's business.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # THE SUITE IS A PREFIX, NOT A CAPABILITY.
    # `env_suite` maps by the `jax_` prefix (envs/suites.py), so it answers
    # "what is this named", not "does this step on a device".
    # `jax_toy` is the tier's OFFLINE env -- zero `mjx` references in
    # `bird/envs/jax_toy.py` -- so `--xla_gpu_autotune_level` selects no
    # kernel for it and there is nothing for this call to make reproducible.
    # Asking anyway would raise `set_xla_determinism_flags() called after jax
    # was imported` in the tier's own end-to-end test, which imports jax
    # through a fixture before the run starts.
    #
    # THREE STATES, AND THE UNDECLARED ONE KEEPS THE STRICT BEHAVIOUR ON PURPOSE.
    # `requires_cuda` is declared on the registered env object (a class or a
    # factory carrying it). False -> this env does not step on the GPU, so no
    # flag and no strictness. True -> the MJX-backed path, unchanged. None ->
    # undeclared, and we keep the strict call rather than silently skipping,
    # because under skipping an MJX adapter whose author forgot the attribute
    # would run non-reproducibly with nothing red. The catalogue
    # test that makes None unreachable for jax-suite envs is the other half of
    # this and lives with the declaration.
    try:
        # ONLY IF THE REGISTRY IS ALREADY LOADED, and never forcing a load.
        # `registry.get` calls `load_all()`, which imports every component
        # module and the LLM clients; this helper runs before the env is
        # built and should not pull the whole registry in as a side effect.
        #
        # `requires_cuda` is only an OPT-OUT: an adapter that declares False
        # skips the flags. Unknown therefore means "set them", which is the
        # safe side, and reading it is not worth forcing a registry load.
        import bird.registry as _reg
        if getattr(_reg, "_LOADED", False):
            requires_cuda = getattr(_reg.get("env", env_id), "requires_cuda", None)
        else:
            requires_cuda = None
    except Exception:  # noqa: BLE001 - an unresolvable id is not this gate's business
        requires_cuda = None
    if requires_cuda is False:
        return False
    return set_xla_determinism_flags(strict=strict)


def preallocate_setting() -> str:
    """What `XLA_PYTHON_CLIENT_PREALLOCATE` is in THIS process, for the row.

    Recorded per process rather than assumed, so a run shows the parent and
    its workers agreeing -- the thing that is silently false when a 30 GiB
    parent meets a worker that needs 8. An empty string means the
    variable is unset, which for a jax-tier process means
    `prepare_for_env` never ran here.
    """
    import os as _os
    return _os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "")


def applied_in_time() -> "bool | None":
    """Was this PROCESS's determinism flag set before jax was imported?

    `None` when nothing has set it at all -- which is a third state and not
    the same as False: False means someone tried and was too late, None
    means the tier never asked. A seed row that cannot tell those apart
    invites "the flag was not applied" to be read as "the flag failed".
    """
    return _APPLIED_IN_TIME


def compilation_cache_dir() -> "str | None":
    """Where jax's PERSISTENT compilation cache lives, or None if it has none.

    Read from the environment first and from jax's own config only if jax is
    ALREADY IMPORTED -- `sys.modules`, never an import. This is called from
    the parent of a candidate wave, and importing jax there to ask a
    question about jax is the exact move the spawn worker exists to avoid.
    """
    import os
    import sys

    for var in ("JAX_COMPILATION_CACHE_DIR", "JAX_CACHE_DIR"):
        val = os.environ.get(var)
        if val:
            return val
    jax = sys.modules.get("jax")
    if jax is None:
        return None
    try:
        return getattr(jax.config, "jax_compilation_cache_dir", None) or None
    except Exception:  # noqa: BLE001 - a config attribute is not worth a failure
        return None


def compilation_cache_entries() -> "int | None":
    """How many entries jax's persistent compilation cache holds, or None.

    NONE IS NOT ZERO, and keeping them apart is the whole reason this
    returns an Optional. "No cache is configured" and "a cache is configured
    and empty" are different facts, and the measurement this feeds -- the
    parent's count before and after a wave, which is how we show the work
    compiled in the WORKERS and not in the parent -- is meaningless under
    the first. Reporting 0 there would make a run that measured nothing
    look like a run that proved the property.

    Counts files recursively rather than top-level entries: jax shards the
    cache into subdirectories, so a top-level count reads a constant.
    """
    import os

    d = compilation_cache_dir()
    if not d or not os.path.isdir(d):
        return None
    n = 0
    for _root, _dirs, files in os.walk(d):
        n += len(files)
    return n
