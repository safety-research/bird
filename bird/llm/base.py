"""The LLM client contract every stage codes against.

One client object stands in for one *role* (`llm.generator.*` or
`llm.evaluator.*`). Splitting by role rather than by call site is what lets a
config price the two roles differently -- RDA's published point: its Agent VLM
(GPT-5) does ALL in-loop work, writing rewards and scoring videos alike, while
GPT-4.1 exists only for the post-hoc alignment-rate metric, the evaluator role
here (not "GPT-5 writes rewards, GPT-4.1 scores videos", a common misreading)
-- without any component knowing which model it is talking to.

The contract (fixed; providers implement `_complete` and nothing else):

    client(messages, n=1, temperature=None, images=None, tag="") -> list[str]
    client.chat_json(messages, schema_hint, **kw)                -> dict
    client.model / .role / .modality

`tag` is a free-text hint naming the *purpose* of the call ("generate",
"decompose", "score", "preference", "feedback", "repair"). It is advisory for a
real provider -- it never changes the request -- but the mock provider uses it
to decide what shape of answer to synthesise, which is what makes the offline
test suite able to exercise every branch of the loop.

Two conventions worth stating because the budget report depends on them:

  * a client records its OWN usage; no stage calls `budget.record_llm`. One
    `record_llm` per provider round-trip, so `budget.llm_calls` counts API
    calls, not samples -- `n=16` in four chunks is four calls.
  * when the provider does not report token counts we estimate them at
    `len(text) // 4` rather than leaving them at zero, because a zero token
    column silently makes a method look free.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Sequence

# The MODULE, not the symbol -- see the note above `chunk_sizes`.
from .. import parsing

log = logging.getLogger("bird")

Message = Dict[str, Any]

#: Crude but stable characters-per-token ratio for providers that do not
#: report usage. Deliberately a constant and not a tokenizer: a dependency on
#: tiktoken/transformers would break the "pure stdlib + numpy" rule.
CHARS_PER_TOKEN = 4

#: Rough prompt cost of one attached image, for `modality: vlm` accounting.
IMAGE_TOKENS = 800

VALID_ROLES = ("generator", "evaluator")


class LLMError(RuntimeError):
    """Provider failure that the caller is expected to see, not swallow."""


# --------------------------------------------------------------------------
# free functions -- shared by every provider and by the tests
# --------------------------------------------------------------------------


def estimate_tokens(text: Any) -> int:
    """`len // 4`, the standard back-of-envelope. Never returns 0 for non-empty
    text, so a short prompt still costs something in the budget report."""
    if not text:
        return 0
    s = text if isinstance(text, str) else str(text)
    return max(1, len(s) // CHARS_PER_TOKEN)


def _block_text(content: Any) -> str:
    """Flatten Anthropic-style content blocks to plain text for estimation."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(str(b.get("text", "")) if b.get("type") == "text" else f"[{b.get('type')}]")
            else:
                parts.append(str(b))
        return "\n".join(parts)
    return str(content)


def normalise_messages(messages: Any) -> List[Message]:
    """Accept a bare string, one message dict, or a list of them.

    Every stage that builds prompts does so slightly differently; normalising
    here means no component has to care, and a client can always assume
    `[{"role": ..., "content": ...}, ...]`.
    """
    if messages is None:
        return []
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if isinstance(messages, dict):
        messages = [messages]
    out: List[Message] = []
    for m in messages:
        if isinstance(m, str):
            out.append({"role": "user", "content": m})
        elif isinstance(m, dict):
            role = str(m.get("role", "user"))
            out.append({**m, "role": role, "content": m.get("content", "")})
        else:  # a Candidate-ish object with .role/.content, or junk
            out.append({"role": str(getattr(m, "role", "user")),
                        "content": str(getattr(m, "content", m))})
    return out


def messages_text(messages: Sequence[Message]) -> str:
    """The whole conversation as one string -- what the mock inspects and what
    token estimation measures."""
    return "\n".join(f"{m.get('role', 'user')}: {_block_text(m.get('content', ''))}"
                     for m in messages)


# `extract_json` lives in `bird/parsing.py`. It is a string scanner -- text
# in, value out -- with no tie to any model, and it must stay importable
# without importing `bird.llm*`: `registry.load_all()` imports every component
# module, and a training-only process need not import the LLM clients.
#
# `chat_json()` below calls it as `parsing.extract_json` through the MODULE
# rather than importing the name. The distinction is not style:
# `from ..parsing import extract_json` here would rebind the symbol into this
# module's namespace, so `from ..llm.base import extract_json` would keep
# working and components could depend on bird/llm again. The dependency points
# one way -- bird/llm may use bird/parsing, never the reverse -- and
# `tests/test_load_all_does_not_import_llm.py` pins both halves.
def chunk_sizes(n: int, chunk: int) -> List[int]:
    """Split `n` samples into per-call chunks (`generate.sampling.chunk_size`).

    Eureka's released code asks for 4 samples per API call; the knob exists so
    a provider that caps `n` does not silently return fewer candidates than
    the config asked for.
    """
    n = max(0, int(n))
    chunk = max(1, int(chunk))
    out = []
    while n > 0:
        out.append(min(chunk, n))
        n -= chunk
    return out


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------


class LLMClient:
    """Base class: role/config binding, chunking, budget accounting, JSON.

    Subclasses implement `_complete(messages, n, temperature, images, tag)`
    returning exactly `n` strings and recording their own usage via `_record`.
    """

    provider = "base"

    #: Whether this provider fans one `_complete(n)` out over concurrent
    #: requests when `llm.max_concurrent_requests` > 1. False for anything
    #: compute-local: the mock's per-sample RNG draws consume the run's stream
    #: IN ORDER, which concurrency would reorder -- so the mock stays serial
    #: whatever the key says, and the key documents itself as transport-only.
    concurrent_samples = False

    def __init__(self, ctx: Any, role: str):
        if role not in VALID_ROLES:
            raise LLMError(f"llm role must be one of {VALID_ROLES}, got {role!r}")
        cfg = ctx.cfg
        self.ctx = ctx
        self.role = role
        self.model = cfg.get(f"llm.{role}.model") or f"{self.provider}-{role}"
        self.temperature = float(cfg.get(f"llm.{role}.temperature", 0.0) or 0.0)
        self.modality = cfg.get(f"llm.{role}.modality") or "text"
        # Only the generator declares a reasoning effort in the schema; the
        # evaluator inherits "none" so both roles can be handled identically.
        self.reasoning_effort = cfg.get(f"llm.{role}.reasoning_effort") or "none"
        self.max_context_tokens = int(cfg.get("llm.max_context_tokens", 128000) or 128000)
        self.chunk_size = max(1, int(cfg.get("generate.sampling.chunk_size", 1) or 1))
        #: `llm.max_concurrent_requests` -- how many of one sampling call's
        #: round-trips a provider may have in flight at once. A transport
        #: schedule in the sense `train.candidate_parallelism` is: the samples
        #: are i.i.d., their order is by index either way, so the key must
        #: never move a number -- only the wall clock.
        self.max_concurrent_requests = max(
            1, int(cfg.get("llm.max_concurrent_requests", 1) or 1))
        self.n_calls = 0
        #: Serialises `_record` when a provider fans `_one` out over threads:
        #: `Budget.record_llm` is an unsynchronised `+=` (measured losing 72-78%
        #: of increments at 16 threads x 20k calls -- the number that sent the
        #: candidate fork to processes), so a fanned-out client must not write
        #: it bare. Uncontended cost on the serial path is nanoseconds.
        self._record_lock = threading.Lock()

    # -- rng -------------------------------------------------------------
    #
    # Read through to the context on every access, never captured at
    # construction: `run_search` installs a fresh `random.Random` per restart
    # (bird.py), and a client holding the old object would silently break the
    # "same seed, same run" guarantee.

    @property
    def rng(self):
        return self.ctx.rng

    # -- the contract ----------------------------------------------------

    def __call__(self, messages: Any, n: int = 1, temperature: Optional[float] = None,
                 images: Optional[Sequence[Any]] = None, tag: str = "") -> List[str]:
        msgs = normalise_messages(messages)
        n = max(1, int(n))
        temp = self.temperature if temperature is None else float(temperature)
        # THE CAP BINDS BEFORE THE CALL IS PAID FOR. `record_llm` increments
        # and only then `_check`s on `>`, so checking there alone would let
        # `max_llm_calls: 40` permit 41 API calls: the crossing one made,
        # billed and returned, and the run stopped afterwards. Asking here,
        # before `_complete`, is the difference between a cap and a report --
        # and a cap on a PAID external service that is only a report is the
        # worse half of the two.
        #
        # One round-trip per `_record`, and `n` samples may be one round-trip
        # or several depending on the provider, so this asks for ONE: the
        # conservative direction, refusing at the boundary rather than one
        # call past it.
        #
        # Reached through `getattr`, because `__call__` is also exercised by
        # doubles that bypass `__init__` and so have no `ctx` at all
        # (`tests/test_llm_transport.py::_Chunky`, which tests chunk
        # stitching and has no business owning a budget). The production
        # path always has one: `LLMClient.__init__` sets `self.ctx`
        # unconditionally, so a missing budget here means a test double
        # rather than an unenforced cap.
        _budget = getattr(getattr(self, "ctx", None), "budget", None)
        _why = _budget.would_exceed(llm_calls=1) if _budget is not None else None
        if _why:
            from ..budget import BudgetExceeded
            raise BudgetExceeded(_why)
        # A provider that fans its samples out over concurrent round-trips
        # receives the WHOLE n in one `_complete` call: chunking is meaningless
        # there (the Messages API has no `n`, every sample is its own request
        # regardless) and would serialise the very waves the fan-out overlaps.
        # Serial providers keep the exact chunk sequence they always saw.
        sizes = ([n] if self.concurrent_samples and self.max_concurrent_requests > 1
                 else chunk_sizes(n, self.chunk_size))
        out: List[str] = []
        #: Index-aligned with `out`: why sample i came back empty ("" when it did
        #: not). A provider that knows fills `_chunk_empty_reasons` per
        #: `_complete`; this is the one place the chunks are stitched back into
        #: the caller's order, so a reason can never migrate to a neighbouring
        #: slot across a chunk boundary.
        self.last_empty_reasons: List[str] = []
        for size in sizes:
            self._chunk_empty_reasons = []
            got = list(self._complete(msgs, size, temp, images, tag))
            reasons = [str(r) for r in (getattr(self, "_chunk_empty_reasons", None) or [])]
            reasons = (reasons + [""] * len(got))[:len(got)]
            out.extend(got)
            self.last_empty_reasons.extend(reasons)
        if len(out) < n:
            # A provider that returned short must not silently shrink the
            # candidate pool -- pad with empty strings so the parse stage sees
            # (and records) the forfeited slots. Eureka's `max_retries: 0`.
            log.warning("%s returned %d/%d samples for tag=%r", self.provider, len(out), n, tag)
            short = n - len(out)
            out.extend([""] * short)
            self.last_empty_reasons.extend(
                [f"provider returned {n - short}/{n} samples"] * short)
        self.last_empty_reasons = self.last_empty_reasons[:n]
        return out[:n]

    def chat_json(self, messages: Any, schema_hint: Any = "", **kw: Any) -> Dict[str, Any]:
        """One structured call. Returns `{}` on an unparseable answer.

        Used for VLM subtask scoring and preference labels (§4). The schema
        hint is appended to the prompt as text -- the shape is requested, never
        enforced, which is the honest model of what a provider guarantees.
        """
        msgs = normalise_messages(messages)
        hint = self._render_hint(schema_hint)
        if hint:
            msgs = msgs + [{"role": "user", "content":
                            "Reply with a single JSON object matching this shape, "
                            f"inside a ```json fence:\n{hint}"}]
        kw.setdefault("tag", "json")
        kw.pop("n", None)
        texts = self(msgs, n=1, **kw)
        data = parsing.extract_json(texts[0] if texts else "")
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"items": data}
        log.warning("%s: unparseable JSON for tag=%r", self.provider, kw.get("tag"))
        return {}

    # -- helpers for subclasses -----------------------------------------

    @staticmethod
    def _render_hint(schema_hint: Any) -> str:
        if not schema_hint:
            return ""
        if isinstance(schema_hint, str):
            return schema_hint
        try:
            return json.dumps(schema_hint, indent=2, default=str)
        except (TypeError, ValueError):
            return str(schema_hint)

    def _record(self, prompt_tokens: int, completion_tokens: int,
                n_images: int = 0, cache_read_tokens: int = 0,
                cache_write_tokens: int = 0, refused: bool = False,
                truncated: bool = False) -> None:
        """One provider round-trip's cost. `vlm` follows the client's declared
        modality, so `budget.vlm_calls` counts what the VLM was asked, not what
        happened to have an image attached.

        `prompt_tokens` is the TOTAL input the call consumed, cached or not; the
        two cache figures are a SPLIT of it and never an addition to it. Keeping
        the total whole is what lets a cached run and an uncached one be compared
        in the same cost column, and carrying the split is what makes the saving
        visible at all -- `llm_prompt_tokens` alone cannot distinguish "cheap
        method" from "cached prefix".
        """
        with self._record_lock:
            self.n_calls += 1
            self.ctx.budget.record_llm(
                prompt_tokens=int(prompt_tokens) + IMAGE_TOKENS * int(n_images),
                completion_tokens=int(completion_tokens),
                vlm=(self.modality == "vlm"),
                cache_read_tokens=int(cache_read_tokens),
                cache_write_tokens=int(cache_write_tokens),
                refused=bool(refused),
                truncated=bool(truncated),
            )

    def _complete(self, messages: List[Message], n: int, temperature: float,
                  images: Optional[Sequence[Any]], tag: str) -> List[str]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(role={self.role!r}, model={self.model!r}, "
                f"modality={self.modality!r})")
