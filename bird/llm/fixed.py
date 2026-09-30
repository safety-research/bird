"""`llm.generator.provider: fixed` -- the generator returns ONE authored program, verbatim.

WHAT IT IS FOR. A control: train a reward program somebody WROTE -- gym's own
HalfCheetah-v5 reward, a paper's, a hand baseline -- through the normal
stage-2-to-6 path, so a number from the search can be set beside a number from
the same learner on a known program. No other provider hands the pipeline a
fixed program: the mock draws its own, the real providers ask a model.

WHY A PROVIDER AND NOT A `generate.output` PATH. The `llm` registry family is
the seam where "what answers the prompt" is chosen (`mock`, `anthropic`,
`openai`); a program that answers every prompt with itself is one more member,
the mock's precedent, and every stage downstream -- the parser, the verifier,
training, evaluation, selection, the reflection prompt -- sees an ordinary
response and runs unchanged. No stage branches on it, which is the repo's
central rule.

TWO REFUSALS keep it a control (`config._check_coherence`): the file must
exist at load, and `fixed` on the evaluator role is refused -- a judge is not
a program, the judge stays mock or real. `generate.n_candidates` is LEFT TO
THE CONFIG: K candidates are K identical programs, which is the same reward
under K learner seeds -- a learner-variance control -- and a report should
state K.

THE PROFILE TRAP, stated because it will be hit: `configs/_profiles/full.yaml`
pins `llm.generator.provider: anthropic` and a profile OUTRANKS the method
config, so `-c <control> --profile full` silently swaps this provider out and
the load then fails on "program is set but the provider never reads it" --
loudly, which is the refusal's job. A launch under a profile that pins a
provider must put `-s llm.generator.provider=fixed -s llm.evaluator.provider=
mock` on the command line, where overrides sit above the profile.

PROVENANCE. `Config.hash()` carries the PATH, not the bytes, so an edited
program under the same path would resume into an old run's directory with a
different reward. The provider therefore journals the file's sha256 and first
line at construction (`event: fixed_program`), and a report should state what
the program is a transcription of; a reader of the artifact has the hash to
check the bytes against.

Zero tokens are recorded per call (`llm_calls` still counts the round-trips,
so the budget's call figure stays honest about how often the generator was
asked). Under `generate.output.format: component_dict_plus_weights` the
weights block is re-read from the program's own `weights = {...}` exactly as
the mock does, so the parser's fenced-json branch sees what the contract asks.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, List, Optional, Sequence

from ..paths import program_path
from ..registry import register
from .base import LLMClient, LLMError, Message

__all__ = ["FixedProgramLLM"]


class FixedProgramLLM(LLMClient):
    """Every generator call returns the same fenced program (module docstring)."""

    provider = "fixed"

    def __init__(self, ctx: Any, role: str) -> None:
        super().__init__(ctx, role)
        if role != "generator":
            raise LLMError("llm.evaluator.provider=fixed: a judge is not a program; the fixed "
                           "provider is the generator-side control only")
        raw = ctx.cfg.get("llm.generator.program")
        if not str(raw or "").strip():
            raise LLMError("llm.generator.provider=fixed needs llm.generator.program")
        self.program_path = program_path(raw)
        try:
            self.program = self.program_path.read_text()
        except OSError as exc:
            raise LLMError(f"llm.generator.program={raw!r}: cannot read {self.program_path}: "
                           f"{exc}") from exc
        if not self.program.strip():
            raise LLMError(f"llm.generator.program={raw!r}: {self.program_path} is empty")
        self.sha256 = hashlib.sha256(self.program.encode("utf-8")).hexdigest()
        first = next((ln.strip() for ln in self.program.splitlines() if ln.strip()), "")
        ctx.event("generate", event="fixed_program", path=str(raw),
                  resolved=str(self.program_path), sha256=self.sha256, first_line=first[:200],
                  n_lines=self.program.count("\n") + 1)

    # -- the contract ----------------------------------------------------

    def _response(self) -> str:
        head = (f"Fixed program {self.program_path.name} (sha256 {self.sha256[:12]}), "
                "returned verbatim; no model was called.")
        out = f"{head}\n\n```python\n{self.program.rstrip()}\n```"
        fmt = self.ctx.cfg.get("generate.output.format", "component_dict_return")
        if fmt == "component_dict_plus_weights":
            from .mock import _weights_of  # the same re-read the mock does

            weights = _weights_of(self.program)
            if weights:
                out += ("\n\nWeights applied by the code above:\n\n```json\n"
                        + json.dumps(weights, indent=2) + "\n```")
        return out

    def _complete(self, messages: List[Message], n: int, temperature: float,
                  images: Optional[Sequence[Any]], tag: str) -> List[str]:
        out = [self._response() for _ in range(n)]
        self._record(prompt_tokens=0, completion_tokens=0, n_images=len(images or ()))
        return out


@register("llm", "fixed", doc="One authored reward program, returned verbatim on every "
                              "generator call (a control; llm.generator.program).")
def fixed_factory(ctx: Any, role: str) -> FixedProgramLLM:
    """Factory for `llm.generator.provider: fixed`."""
    return FixedProgramLLM(ctx, role)
