"""Per-run dependency container.

Everything a stage needs that is not `RunState`: the resolved config, the cost
counters, the output directory, the LLM clients, the environment adapter, and
the RNG. Passing it explicitly (rather than reaching for module globals) is
what lets the test suite run a whole method with a mock LLM and a toy env.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

from .artifacts import RunDir
from .budget import Budget
from .config import Config


class _NoTracker:
    """The tracker a Context has before `bird.run()` installs a real one.

    Defined here rather than imported from `bird.observability` to keep this
    module free of the registry (observability imports it, and the registry
    imports every component module, several of which import Context). A stage
    can therefore call `ctx.tracker.log_*` unconditionally -- including in a
    test that builds a Context by hand -- without a `None` guard at every call
    site, which is the same reason `human` and `env` are constructed eagerly.
    """

    def __getattr__(self, _name: str):
        return lambda *a, **k: None


@dataclass
class Context:
    cfg: Config
    budget: Budget
    rundir: Optional[RunDir] = None
    generator: Any = None  # LLMClient
    evaluator: Any = None  # LLMClient (often a different, cheaper model)
    env: Any = None  # EnvAdapter
    rng: random.Random = field(default_factory=random.Random)
    human: Any = None  # HumanOracle (scripted / interactive / none)
    tracker: Any = field(default_factory=_NoTracker)  # `output.tracker`; replaced in run()
    counters: dict = field(default_factory=dict)
    #: `bird.checkpoint.Checkpointer`; a no-op unless `loop.resume_from` is set.
    #: Lives here rather than in a stage argument because `run_search` and
    #: `_execute` both write checkpoints and neither is allowed to grow a second
    #: code path for the resumed case.
    checkpointer: Any = None

    #: Where `event()` writes when there is no run directory. `None` means
    #: drop, which is what a Context without a run does; a list means a forked
    #: child collecting its journal to ship home (`training.
    #: build_child_payload`), because the alternative -- silently discarding
    #: every event a worker emits -- loses K rows per iteration and leaves one
    #: surviving row that looks like the whole story.
    event_buffer: Optional[list] = None

    def next_id(self, prefix: str = "c") -> str:
        n = self.counters.get(prefix, 0)
        self.counters[prefix] = n + 1
        return f"{prefix}{n:04d}"

    def event(self, stage: str, **fields: Any) -> None:
        if self.rundir is not None:
            self.rundir.event(stage, **fields)
        elif self.event_buffer is not None:
            # A child with no run dir: collect rather than drop, so the parent
            # can replay into the one journal. Order is preserved and the
            # parent stamps nothing -- a replayed event must look like the
            # event the child emitted, or the journal describes the fold
            # instead of the work.
            self.event_buffer.append((stage, dict(fields)))
