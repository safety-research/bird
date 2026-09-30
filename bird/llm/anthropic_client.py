"""Anthropic Messages API provider. Registers `llm:anthropic`.

This is the only module `registry.load_all` is allowed to skip on ImportError,
so the SDK import happens lazily *inside the constructor* -- importing this
file must never require `anthropic` to be installed, or a laptop with no API
key could not run the mock configs.

Config mapping. Every declared key is honoured or the client says out loud
that it could not honour it -- a `temperature` field the client silently
ignored would be a fabricated pin:

    llm.<role>.model             -> `model` (never invented here; see DEFAULT_MODEL)
    llm.<role>.temperature       -> `temperature`, when the model still accepts it
    llm.<role>.reasoning_effort  -> adaptive thinking + `output_config.effort`
    llm.<role>.modality          -> whether image blocks are attached, and
                                    whether the budget counts this as a VLM call
    generate.sampling.chunk_size -> ignored by design: the Messages API has no
                                    `n`, so `n` samples are `n` round-trips
                                    (each one recorded separately in the budget)

Four ENVIRONMENT variables tune the transport, and none of them is a config key
-- see the comment on `MAX_RETRIES` for why (a config key here would move every
config hash to express something that cannot change what the model reads):

    BIRD_LLM_MAX_RETRIES     attempts after the first  (default 12)
    BIRD_LLM_BACKOFF_BASE_S  first backoff, doubling   (default 1.0)
    BIRD_LLM_BACKOFF_MAX_S   backoff ceiling           (default 180.0)
    BIRD_LLM_TIMEOUT_S       per-request wall clock    (default: the SDK's)

The first three default to a retry ladder MEASURED against real API overload
episodes, and exist so a long run can go further, never to walk the ladder
back down -- see the comment on `MAX_RETRIES`.

Current Claude models reject `temperature` outright, and reject `thinking` /
`output_config` on older ones. Rather than hard-coding a capability table that
goes stale, `_call` drops the offending parameter on a 400 that names it,
warns once, and retries -- so the run continues and the log records exactly
which pin the provider refused.
"""

from __future__ import annotations

import base64
import logging
import os
import random
import time
import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..registry import register
from .base import LLMClient, LLMError, Message, estimate_tokens, messages_text

log = logging.getLogger("bird")

#: Fallback when `llm.<role>.model` is unset. A DEFAULT, not a pin: configs
#: name their own model, and a method config that relies on this default is
#: under-specified.
DEFAULT_MODEL = "claude-opus-5"

#: Output cap per call.
#:
#: 8192 ("reward programs are short") is enough for Pendulum and not for
#: Meta-World. MEASURED: four identical real Opus 5 calls on the shipped
#: `mt10_peg-insert-side-v3` eureka prompt at an 8192 cap:
#:
#:     stop_reason    out_tokens   closing fence
#:     max_tokens          8,192   no
#:     end_turn            6,833   yes
#:     max_tokens          8,192   no
#:     end_turn            5,240   yes
#:
#: Half the samples hit the cap exactly, and the two that finished had almost no
#: margin. The cap is shared with adaptive-thinking tokens, which dominate on
#: hard tasks, so it binds hardest exactly where the task is hardest.
#:
#: The consequence of a too-small cap is not a visible error. A truncated
#: program has no closing fence, `_extract_code` falls through to its docstring
#: pattern and returns the DOCSTRING, and `verify` then reports `SyntaxError` --
#: i.e. the artifact blames the model for writing bad code when the client cut
#: it off mid-line. On the harder half of MT10 that is a coin flip per
#: candidate, and it would read as "these methods write worse rewards on hard
#: tasks".
DEFAULT_MAX_TOKENS = 32768

#: The retry ladder. MEASURED values, made operator-tunable.
#:
#: Making a mid-stream `overloaded_error` retryable at all (`_retryable_status`
#: below) is necessary and not sufficient: with 5 retries, long humanoid
#: searches still died with `failed after 5 retries` during real overload
#: episodes, repeatedly and late in a 20-iteration run. Retryability and retry
#: count are two independent requirements on one path.
#:
#: 12 retries capped at 180 s is ~16 minutes of BACKOFF (sum of min(2^i, 180)
#: for i<12) -- the base ladder only: `_sleep` adds jitter of up to
#: `llm.max_concurrent_requests` seconds per wait, and a server-named
#: `retry-after` is a floor that replaces the exponential term, so the wall
#: clock can run past the figure. It rides out an outage of the length
#: actually observed. The asymmetry is the whole argument: waiting 16 minutes
#: costs 16 minutes, while giving up costs a resume at best and, for a run that
#: cannot resume, a relaunch from iteration 0 and every hour of RL already
#: spent.
#:
#: WHY THEY ARE ENVIRONMENT VARIABLES AND NOT CONFIG KEYS. They are TRANSPORT,
#: on the same footing as `llm.prompt_caching` and `llm.max_concurrent_requests`
#: in `configs/_default.yaml`: how patiently this process re-sends a request
#: cannot change what the model reads, so two runs differing only here are the
#: same experiment. A config key would say otherwise in the one place it
#: matters -- `Config.hash()` covers all of `_data`, so three new keys in
#: `_default.yaml` would move EVERY config hash, rename every future run
#: directory, and make an in-flight run unresumable across the upgrade. An
#: operator lever that must not move a hash is an env var.
#:
#: THE DEFAULTS BELOW ARE THE MEASURED ONES. The variables exist so a run can go
#: FURTHER when its own evidence says to, not to quietly walk this ladder back
#: down: 12/180 was paid for in lost iterations, and anything smaller has to
#: beat that measurement.
#:
#: A malformed value is a WARNING and the default, never a crash: these are read
#: at import, and an env-var typo must not take out every task of a job array
#: before the search starts.


def _env_num(name: str, default: float, cast=float, minimum: float = 0.0) -> Any:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return cast(default)
    try:
        value = cast(str(raw).strip())
    except (TypeError, ValueError):
        log.warning("%s=%r is not a number; using the default %s", name, raw, default)
        return cast(default)
    if value < minimum:
        log.warning("%s=%r is below the minimum %s; using the default %s",
                    name, raw, minimum, default)
        return cast(default)
    return value


MAX_RETRIES = _env_num("BIRD_LLM_MAX_RETRIES", 12, int, 0)
#: Extra attempts when the model answers `stop_reason: refusal`. A refusal is not
#: an error -- the SDK raises nothing, so the ladder above never sees it -- and
#: left alone it becomes an empty sample on the first try, never retried, never
#: counted. Refusals are observed in transient bursts (16 of 16 samples across
#: two iterations empty in ~1.5 s each with zero output tokens, the identical
#: prompt answering normally a day later), and they are the cheapest failure to
#: retry: a refused call bills its input only.
REFUSAL_RETRIES = _env_num("BIRD_LLM_REFUSAL_RETRIES", 2, int, 0)
#: Pause between refusal attempts, seconds, times the attempt number. Fixed and
#: short on purpose: a refusal is a classifier verdict, not load, so the
#: transport ladder's exponential backoff and its concurrency-scaled jitter
#: (`_sleep`) would add tens of seconds -- and a "retry 1/12" log line -- to a
#: path that costs one input-token bill per attempt.
REFUSAL_PAUSE_S = _env_num("BIRD_LLM_REFUSAL_PAUSE_S", 2.0, float, 0.0)
BACKOFF_BASE_S = _env_num("BIRD_LLM_BACKOFF_BASE_S", 1.0, float, 0.0)
BACKOFF_MAX_S = _env_num("BIRD_LLM_BACKOFF_MAX_S", 180.0, float, 0.0)

#: Per-request wall clock handed to the SDK, or None for the SDK's own default.
#:
#: SEPARATE FROM THE RETRY LADDER ABOVE (see `_transport_errors`): a read
#: timeout is not a rate limit, and no number of retries helps a request whose
#: deadline is shorter than the answer takes. The ladder answers "the server
#: said no"; this answers "the server said nothing". `_call` STREAMS, so this is
#: the patience for the stream as a whole; RDA's evaluator attaches 20 frames
#: per query, which is the largest request shape any shipped config issues and
#: the one most likely to sit past a default.
#:
#: None by default, so an unset environment keeps exactly the SDK's behaviour --
#: this must not become a pin nobody asked for.
#: ON ONE LINE deliberately: `test_each_constant_is_still_bound_to_its_
#: environment_variable` reads this file as text (the module cannot be
#: re-imported under a different environment -- reload re-runs its
#: `@register` and `registry.register` raises on the duplicate), and a
#: binding wrapped across two lines reads as an unbound constant.
LLM_TIMEOUT_S: Optional[float] = _env_num("BIRD_LLM_TIMEOUT_S", 0.0, float, 0.0) or None

#: Error `type` strings that are transient no matter what HTTP status the
#: exception carries. This exists because of `_retryable_status` below, and it
#: is a list of error TYPES rather than statuses for the reason stated there.
RETRYABLE_ERROR_TYPES = frozenset({
    "overloaded_error",
    "api_error",
    "rate_limit_error",
    "timeout_error",
})

#: Parameters the client will drop-and-retry when a model rejects them.
DEGRADABLE = ("temperature", "thinking", "output_config")

#: Words a 400 may use for a parameter WITHOUT naming the request key.
#:
#: `_degrade` matches the key it sent against the error text, which only works
#: while the API and the request agree on the word. On `claude-sonnet-4-5`
#: with `llm.generator.reasoning_effort: medium` the reply is
#: `"This model does not support the effort parameter."` -- `effort` is the
#: field INSIDE `output_config`, so nothing the client sent by name appears in
#: the message, and without an alias `_degrade` returns False and the whole run
#: dies on its first generate call instead of degrading. That is the same shape as the
#: `temperature` TypeError already documented in `_call`: a rejected pin has to
#: be droppable however the far side phrases it.
#: Only the one that has actually been observed. A wider list would be guessing,
#: and a wrong entry silently drops a pin the model never objected to.
_ALIASES = {"effort": "output_config"}

_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}

#: Backoff jitter comes from a private stream: `ctx.rng` is the run's
#: reproducibility contract and network flakiness must not perturb it.
_JITTER = random.Random(0xC0FFEE)


class AnthropicClient(LLMClient):
    """Thin, budget-accounting wrapper over `client.messages.create`."""

    provider = "anthropic"

    #: `llm.max_concurrent_requests` is honoured here: the Messages API is
    #: stateless request/response, so n i.i.d. samples are n independent
    #: round-trips whatever happens, and overlapping them changes wall clock
    #: only. See `_complete` for the leader-then-followers shape.
    concurrent_samples = True

    def __init__(self, ctx: Any, role: str):
        super().__init__(ctx, role)
        if not self.model or self.model.startswith("anthropic-"):
            self.model = DEFAULT_MODEL
        self.max_tokens = DEFAULT_MAX_TOKENS
        self.prompt_caching = bool(ctx.cfg.get("llm.prompt_caching", True))
        #: Samples discarded because the provider cut them at `max_tokens`.
        #: Counted, not just logged: a run whose hard-task candidates were all
        #: truncated looks identical to one whose model wrote bad code.
        self._truncated = 0
        #: Samples the provider refused, after `REFUSAL_RETRIES` more attempts.
        self._refused = 0
        #: Why the samples of the CURRENT `_complete` call came back empty, ONE
        #: ENTRY PER SAMPLE and index-aligned with the list `_complete` returns
        #: ("" for a sample that arrived). `LLMClient.__call__` concatenates these
        #: across its chunks into `last_empty_reasons`, so the generate stage can
        #: attribute a reason to the exact forfeited slot -- not to whichever
        #: sample happened to be recorded last. `_one` never
        #: writes the list: it parks its reason in a thread-local and `_complete`
        #: collects it in the same thread, which is what keeps the fan-out's
        #: out-of-order completions from scrambling the alignment.
        self._chunk_empty_reasons: List[str] = []
        self._tl = threading.local()
        self._sdk: Any = None
        self._client: Any = None
        self._dropped: set = set()
        self._connect()

    # -- lazy SDK --------------------------------------------------------

    @staticmethod
    def _transport_errors(sdk: Any) -> Tuple[type, ...]:
        """The HTTP transport exceptions that escape the SDK's own wrapping.

        `_call` streams, and that is what makes this necessary. The SDK maps
        httpx failures onto `APIConnectionError` at the REQUEST layer, but a
        stream that dies while `get_final_message()` is iterating it raises the
        raw transport exception from inside the generator:

            anthropic/lib/streaming/_messages.py  __iter__
            anthropic/_streaming.py               _iter_events
            httpx2/_models.py                     iter_bytes
            httpx2.ReadTimeout: The read operation timed out

        `TransportError` is NOT a subclass of `APIConnectionError` (checked, not
        assumed), so without this tuple every `except` clause below misses it
        and a transient network blip ends a search holding hours of RL. No
        retry count helps this class: the exception type decides, not the count.

        The module is DISCOVERED rather than named. `anthropic` 1.0.0 imports
        `httpx2` 2.12.0, not `httpx`, and a bare `import httpx` raises
        ImportError there -- so a hardcoded name would either break now or
        break silently the day the SDK switches back. An empty tuple is a legal
        `except` target that never matches, so an unrecognised SDK degrades to
        not retrying transport errors instead of erroring at import.
        """
        base = getattr(sdk, "_base_client", None)
        if base is None:  # pragma: no cover -- SDK without the private module
            return ()
        for name in ("httpx2", "httpx"):
            mod = getattr(base, name, None)
            err = getattr(mod, "TransportError", None) if mod is not None else None
            if isinstance(err, type) and issubclass(err, BaseException):
                return (err,)
        return ()

    def _connect(self) -> Tuple[Any, Any]:
        if self._client is not None:
            return self._sdk, self._client
        try:
            import anthropic  # noqa: PLC0415 -- deliberately not at module import
        except ImportError as exc:
            raise LLMError(
                "llm provider 'anthropic' needs the anthropic SDK: pip install anthropic. "
                "Use provider 'mock' (e.g. `--profile tester`) to run offline."
            ) from exc
        key = os.environ.get("ANTHROPIC_API_KEY")
        # No key is not the same as no credentials -- the SDK also resolves an
        # `ant auth login` profile -- so only pass one when it is actually set.
        self._sdk = anthropic
        # `timeout` only when one was actually asked for: passing None would
        # override the SDK's own default with "no timeout" on some versions,
        # and this lever must be inert when unset.
        kwargs: Dict[str, Any] = {} if LLM_TIMEOUT_S is None else {"timeout": LLM_TIMEOUT_S}
        if key:
            kwargs["api_key"] = key
        try:
            self._client = anthropic.Anthropic(**kwargs)
        except Exception as exc:  # missing credentials, bad base_url, ...
            raise LLMError(
                f"could not construct the anthropic client ({exc}); set ANTHROPIC_API_KEY "
                "or switch llm.*.provider to 'mock'") from exc
        return self._sdk, self._client

    # -- request assembly ------------------------------------------------

    @staticmethod
    def _split_system(messages: Sequence[Message]) -> Tuple[str, List[Message]]:
        """Hoist system turns into the top-level `system` field: the Messages
        API takes user/assistant turns, and mid-conversation system messages
        are model-gated."""
        system_parts, turns = [], []
        for m in messages:
            content = m.get("content", "")
            if m.get("role") == "system":
                system_parts.append(content if isinstance(content, str) else str(content))
            else:
                role = "assistant" if m.get("role") == "assistant" else "user"
                turns.append({"role": role, "content": content})
        if not turns:
            turns = [{"role": "user", "content": "\n".join(system_parts) or "Continue."}]
        return "\n\n".join(system_parts), turns

    def _image_block(self, image: Any) -> Optional[Dict[str, Any]]:
        """Accept an already-formed block, raw bytes, a URL, a file path, or a
        base64 string. Reading a local file is the only filesystem touch."""
        if isinstance(image, dict):
            return image if image.get("type") == "image" else None
        if isinstance(image, (bytes, bytearray)):
            return {"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.standard_b64encode(bytes(image)).decode()}}
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                return {"type": "image", "source": {"type": "url", "url": image}}
            path = Path(image)
            if path.exists():
                media = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
                return {"type": "image", "source": {
                    "type": "base64", "media_type": media,
                    "data": base64.standard_b64encode(path.read_bytes()).decode()}}
            return {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": image}}
        return None

    def _attach_images(self, turns: List[Message], images: Sequence[Any]) -> List[Message]:
        blocks = [b for b in (self._image_block(i) for i in images) if b]
        if not blocks:
            return turns
        turns = [dict(t) for t in turns]
        last = turns[-1]
        content = last.get("content", "")
        text_blocks = content if isinstance(content, list) else \
            [{"type": "text", "text": str(content)}]
        # Images before text: the vision guidance is that the question reads
        # better after the thing it is about.
        last["content"] = blocks + text_blocks
        return turns

    # -- the provider entry point ---------------------------------------

    def _complete(self, messages: List[Message], n: int, temperature: float,
                  images: Optional[Sequence[Any]], tag: str) -> List[str]:
        system, turns = self._split_system(messages)
        if images and self.modality == "vlm":
            turns = self._attach_images(turns, images)
        elif images:
            log.warning("images passed to a text-modality %s client (tag=%r); ignoring",
                        self.role, tag)
            images = None

        n_img = len(images or ())
        if n <= 1 or self.max_concurrent_requests <= 1:
            out: List[str] = []
            reasons: List[str] = []
            for _ in range(n):  # the Messages API has no `n`
                out.append(self._one(system, turns, temperature, n_img))
                reasons.append(self._take_empty_reason())
            self._chunk_empty_reasons = reasons
            return out

        # LEADER, THEN FOLLOWERS -- `llm.max_concurrent_requests` > 1.
        #
        # The n samples are i.i.d. draws off ONE byte-identical request
        # (`iid_parallel`'s contract), so nothing orders them but the index,
        # and the index is preserved: sample 0 is the leader, `pool.map` keeps
        # 1..n-1 in submission order. What concurrency changes is WALL CLOCK
        # only -- serial stage-1 generation measures 255-546 s per iteration
        # on eureka/rda and 1,206-1,536 s on gt, every second of it 16 serial
        # round-trips into a stage no other core can start.
        #
        # The leader flies ALONE first, on purpose: it is the request that
        # writes the prompt cache (`_cache_breakpoints` marks the shared
        # prefix), and n-1 followers launched beside it would each re-process
        # -- and each re-bill -- the full prefix as a cache MISS. One extra
        # round-trip of latency buys the -61%-input-tokens figure measured for
        # `llm.prompt_caching`; with caching off the leader still costs only
        # that one round-trip against an unbounded herd. Failures keep their
        # serial meaning: `_one` raising (LLMError after retries) aborts this
        # `_complete` whether it was the leader or any follower.
        first = self._one(system, turns, temperature, n_img)
        first_reason = self._take_empty_reason()
        with ThreadPoolExecutor(
                max_workers=min(n - 1, self.max_concurrent_requests),
                thread_name_prefix=f"bird-llm-{self.role}") as pool:
            # Explicit futures, not `pool.map`: on the first failure everything
            # still QUEUED is cancelled, so a raise keeps as much of its serial
            # meaning as a pool allows -- followers that never started are
            # never sent, and therefore never billed or recorded. In-flight
            # requests finish (the API has no abort) and their usage is still
            # recorded by `_one`, which is the honest count of what was spent.
            # All n-1 futures are submitted up front, which is fine at this
            # call's scale -- n is `generate.n_candidates`, at most 16 in any
            # shipped config. A caller fanning out thousands would want a
            # bounded submission window here instead.
            def _sample() -> Tuple[str, str]:
                # `_one` parks its reason in a thread-local; read it back in the
                # SAME thread, so the (text, reason) pair can never be split by
                # a neighbouring follower finishing first.
                text = self._one(system, turns, temperature, n_img)
                return text, self._take_empty_reason()

            futures = [pool.submit(_sample) for _ in range(n - 1)]
            # Wait for FIRST_EXCEPTION, not in submission order: a late
            # follower's failure is noticed the moment it happens, and
            # everything still queued is cancelled then -- not after every
            # earlier future has drained first.
            _done, not_done = wait(futures, return_when=FIRST_EXCEPTION)
            for fut in not_done:
                fut.cancel()
            for fut in futures:
                if not fut.cancelled():
                    exc = fut.exception()
                    if exc is not None:
                        raise exc
            rest = [fut.result() for fut in futures]
        self._chunk_empty_reasons = [first_reason] + [r for _, r in rest]
        return [first] + [t for t, _ in rest]

    # -- prompt caching --------------------------------------------------

    def _cache_breakpoints(self, system: str, turns: List[Message]
                           ) -> Tuple[Any, List[Message]]:
        """Mark the stable prefix of this request as cacheable.

        TRANSPARENT BY CONSTRUCTION, and that is the only reason this belongs in
        the transport at all. `cache_control` is metadata about where the
        provider may reuse computation; it adds no token the model reads and
        removes none, so the sampled distribution is unchanged and no config in
        `configs/` means anything different because of it. If it ever stopped
        being transparent it would be a method key, not a §0 one.

        WHERE THE BREAKPOINTS GO, and each one earns its place against a call
        shape the shipped configs actually issue:

          * the SYSTEM block. Identical across every call a role ever makes
            (`generation._build_messages` builds it from `persona` or the
            default), so it is the one prefix that survives even a changed user
            turn.
          * the LAST IMAGE of the final turn, when there is one. RDA's §4 scorer
            sends the same 20 rollout frames to `len(subtasks)` consecutive
            queries that differ only in the sentence after them
            (`evaluation._vlm_trajectory_analysis`), and an image is the most
            expensive thing in this repo's prompts per unit of variability.
          * the END of the final turn. Eureka's shape: `n` samples off ONE
            prompt is `n` round-trips with byte-identical `messages`
            (the Messages API has no `n`), so the whole request is the prefix
            and calls 2..n read all of it.

        Three breakpoints against the API's limit of four, so nothing here
        crowds out a caller that wants one of its own.

        Below the provider's minimum cacheable length the marker is simply
        ignored -- no cache, no write premium -- so there is no size gate here.
        The premium only exists on a prefix long enough to cache, which is
        exactly the case where 1.25x once beats 1.0x eight times.
        """
        if not self.prompt_caching:
            return system, turns
        mark = {"type": "ephemeral"}
        sys_param: Any = system
        if system:
            sys_param = [{"type": "text", "text": system, "cache_control": mark}]
        if not turns:
            return sys_param, turns
        turns = [dict(t) for t in turns]
        last = turns[-1]
        content = last.get("content", "")
        blocks = ([dict(b) if isinstance(b, dict) else b for b in content]
                  if isinstance(content, list)
                  else [{"type": "text", "text": str(content)}])
        images = [b for b in blocks if isinstance(b, dict) and b.get("type") == "image"]
        if images:
            images[-1]["cache_control"] = dict(mark)
        if isinstance(blocks[-1], dict):
            blocks[-1]["cache_control"] = dict(mark)
        last["content"] = blocks
        return sys_param, turns

    def _one(self, system: str, turns: List[Message], temperature: float,
             n_images: int) -> str:
        system_param, turns = self._cache_breakpoints(system, turns)
        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": turns,
        }
        if system:
            params["system"] = system_param
        if self.reasoning_effort and self.reasoning_effort != "none":
            # Effort is the config's reasoning knob; adaptive thinking is what
            # makes it mean anything on current models.
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": self.reasoning_effort}
        else:
            params["temperature"] = float(temperature)

        message = self._call(params)
        for attempt in range(REFUSAL_RETRIES):
            if getattr(message, "stop_reason", None) != "refusal":
                break
            # Refused. Bill what was billed (the input) and ask again -- the
            # ladder in `_call` cannot: a refusal arrives as a normal message.
            self._account(message, turns, system, n_images)
            log.info("anthropic refused (category=%s); refusal retry %d/%d in %.0fs",
                     getattr(getattr(message, "stop_details", None), "category", None),
                     attempt + 1, REFUSAL_RETRIES, REFUSAL_PAUSE_S * (attempt + 1))
            time.sleep(REFUSAL_PAUSE_S * (attempt + 1))
            message = self._call(params)
        text = self._text_of(message)
        usage = getattr(message, "usage", None)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        # `usage.input_tokens` EXCLUDES both cache fields -- the API reports the
        # three as disjoint buckets. Summing them keeps `budget.llm_prompt_tokens`
        # meaning "input tokens this call consumed", the same thing it meant
        # before caching existed, so a cached run and an uncached one stay
        # comparable in the cost column; the split is carried alongside so the
        # SAVING is legible too. Adding them is the opposite of double-counting:
        # reading `input_tokens` alone would silently under-report a cached run
        # by ~4x and make caching look like a modelling change.
        stop = getattr(message, "stop_reason", None)
        self._record(
            prompt_tokens=(int(getattr(usage, "input_tokens", 0) or 0)
                           + cache_read + cache_write)
            or estimate_tokens(messages_text(turns) + system),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0)
            or estimate_tokens(text),
            n_images=0 if usage is not None else n_images,  # real usage already counts them
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            refused=(stop == "refusal"),
            truncated=(stop == "max_tokens"),
        )
        # A truncated sample must never be returned as if it were complete.
        # Returning it silently is what turned a client-side cap into an
        # apparent model failure (see DEFAULT_MAX_TOKENS). Empty, warned and
        # counted is the same contract `refusal` already uses, and it puts the
        # cause where the reader will look instead of in a SyntaxError.
        if getattr(message, "stop_reason", None) == "max_tokens":
            with self._record_lock:
                self._truncated += 1
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
            log.warning(
                "anthropic hit max_tokens=%d (%d output tokens); the sample is "
                "cut mid-program and is being discarded rather than parsed. "
                "Raise the cap if this recurs.",
                self.max_tokens, out_tok)
            self._park_empty_reason(f"truncated at max_tokens={self.max_tokens} ({out_tok} output tokens)")
            return ""
        if stop == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None)
            with self._record_lock:
                self._refused = getattr(self, "_refused", 0) + 1
            log.warning("anthropic refused (category=%s) on %d attempt(s); returning an "
                        "empty sample", category, REFUSAL_RETRIES + 1)
            self._park_empty_reason(f"refused by the provider (category={category}) "
                                    f"after {REFUSAL_RETRIES + 1} attempt(s)")
            return ""
        return text

    def _account(self, message: Any, turns: List[Message], system: str,
                 n_images: int) -> None:
        """Bill one round-trip that yielded no sample (a refusal being retried):
        its input tokens were consumed whether or not anything came back."""
        usage = getattr(message, "usage", None)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        self._record(
            prompt_tokens=(int(getattr(usage, "input_tokens", 0) or 0)
                           + cache_read + cache_write)
            or estimate_tokens(messages_text(turns) + system),
            completion_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            n_images=0 if usage is not None else n_images,
            cache_read_tokens=cache_read, cache_write_tokens=cache_write,
            refused=True,
        )

    # -- empty-sample reasons: parked per thread, collected per sample ------
    #
    # `_one` runs in whichever thread the fan-out gave it, so it cannot know its
    # sample index; it parks the reason in a thread-local and the caller that
    # DOES know the index reads it back in the same thread (`_take_empty_reason`)
    # right after `_one` returns. `getattr` throughout because the transport
    # tests build this client with `__init__` bypassed.

    def _park_empty_reason(self, reason: str) -> None:
        tl = getattr(self, "_tl", None)
        if tl is None:
            tl = self._tl = threading.local()
        tl.reason = reason

    def _take_empty_reason(self) -> str:
        tl = getattr(self, "_tl", None)
        reason = getattr(tl, "reason", "") if tl is not None else ""
        if tl is not None:
            tl.reason = ""
        return reason

    @staticmethod
    def _text_of(message: Any) -> str:
        blocks = getattr(message, "content", None) or []
        return "\n".join(getattr(b, "text", "") for b in blocks
                         if getattr(b, "type", "") == "text")

    # -- transport -------------------------------------------------------

    def _call(self, params: Dict[str, Any]) -> Any:
        """One request, with backoff on rate limits/5xx and drop-and-retry on
        a parameter this model no longer accepts.

        A rejected pin arrives in TWO shapes and both have to be caught:

        * a **400 from the API**, when the SDK is willing to send the parameter
          and the model refuses it -- `BadRequestError`, handled below; and
        * a **TypeError from the SDK itself**, when `Messages.create()` has
          dropped the keyword entirely and the request is never sent. Nothing
          reaches the network, so no `BadRequestError` is ever raised and the
          call dies with `got an unexpected keyword argument 'temperature'`.

        The second is what `anthropic` 1.0.0 does with `temperature`, `top_p`
        and `top_k`: they are gone from the signature and there is no `**kwargs`
        catch-all. Every published config here pins a temperature
        (Eureka 1.0, CARD and LIMEN 0.7, L2R 0.3), so catching only the first
        shape would crash every real run on its first LLM call instead of
        degrading. Both route through `_degrade`, so the pin is dropped
        once, warned about once, and remembered for the rest of the run.

        That record matters beyond plumbing: it is the run's own evidence that
        "Eureka at temperature 1.0" is not reproducible against this provider,
        rather than a silent substitution of the default.

        THE REQUEST IS STREAMED, and that is a precondition of `max_tokens`
        rather than a preference. `Messages.create()` refuses outright -- before
        any socket is opened -- when the output cap is large enough that the
        response *might* take more than ten minutes:

            expected = 3600 * max_tokens / 128_000      # anthropic 1.0.0,
            if expected > 600: raise ValueError(...)    # _base_client.py:748

        i.e. any `max_tokens > 21_333` is a hard client-side `ValueError`
        ("Streaming is required for operations that may take longer than 10
        minutes"), whatever the prompt is and however fast the model answers.
        `DEFAULT_MAX_TOKENS` is 32,768 (8,192 cuts half the Meta-World samples
        mid-program), so a non-streamed `create()` would kill EVERY real
        Anthropic call on its first request, in `generate` of iteration 1.
        (The ordering hides it: attempt 1 dies in `_degrade`'s TypeError on
        `temperature`, so the ValueError only surfaces on attempt 2 -- and the
        traceback names neither `max_tokens` nor the config key behind it.)

        `messages.stream(...).get_final_message()` returns the same `Message`
        -- `usage`, `stop_reason`, `content` -- with no ceiling, and its keyword
        signature is the same one `_degrade` already knows: `stream()` has no
        `temperature` either, so the drop-and-retry above is unchanged. Lowering
        the cap under 21,333 would work too and is the wrong trade: it buys the
        truncation bug back to keep a transport that the SDK documents as
        unsupported at this size.
        """
        sdk, client = self._connect()
        # Snapshot under the lock: concurrent followers (`_complete`'s fan-out)
        # read this set while another's `_degrade` may be adding to it. The
        # worst unsynchronised outcome is benign-looking (a second follower
        # re-learning a dropped pin, one extra 400 and a duplicate warning),
        # but benign-looking races are how the next refactor inherits a real
        # one -- and the lock is uncontended on the serial path.
        with self._record_lock:
            dropped = set(self._dropped)
        params = {k: v for k, v in params.items() if k not in dropped}
        last: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                with client.messages.stream(**params) as stream:
                    return stream.get_final_message()
            except sdk.BadRequestError as exc:
                dropped = self._degrade(params, exc)
                if not dropped:
                    raise
                last = exc
                continue  # a rejected pin is not a retry: fix and re-send
            except TypeError as exc:
                # The SDK refused the keyword before sending anything. Same
                # remedy, different messenger -- but only when the message
                # actually names a parameter we sent, so a genuine TypeError in
                # our own code still surfaces instead of being retried away.
                if not self._degrade(params, exc):
                    raise
                last = exc
                continue
            except sdk.RateLimitError as exc:
                last = exc
                self._sleep(attempt, self._retry_after(exc))
            except sdk.APIStatusError as exc:
                if not self._retryable_status(exc):
                    raise
                last = exc
                self._sleep(attempt, self._retry_after(exc))
            except (sdk.APIConnectionError, sdk.APITimeoutError) as exc:
                last = exc
                self._sleep(attempt, None)
            except self._transport_errors(sdk) as exc:
                # A stream that died mid-iteration. Same remedy as a connection
                # error -- it IS one, it just arrived unwrapped. Kept as its own
                # clause rather than widened into the one above so that what is
                # being caught, and why the SDK did not catch it, stays legible.
                last = exc
                self._sleep(attempt, None)

        raise LLMError(f"anthropic call failed after {MAX_RETRIES} retries: {last}")

    def _degrade(self, params: Dict[str, Any], exc: Exception) -> bool:
        """A 400 naming a parameter we sent means this model dropped support
        for it. Remove it, say so once, and carry on -- and remember, so the
        rest of the run does not re-learn it on every call.

        "Naming" includes naming a SUB-FIELD: see `_ALIASES`. Dropping
        `output_config` may leave `thinking` for the next attempt to reject,
        which is correct -- two round-trips is the price of not hard-coding a
        capability table that goes stale."""
        text = str(getattr(exc, "message", "") or exc).lower()
        named = {key for key in DEGRADABLE if key in text}
        named |= {target for word, target in _ALIASES.items() if word in text}
        for key in DEGRADABLE:
            if key in params and key in named:
                params.pop(key)
                with self._record_lock:
                    first_time = key not in self._dropped
                    self._dropped.add(key)
                if first_time:
                    log.warning("%s rejects %r; dropping it for the rest of this run "
                                "(the config pin cannot be honoured on this model)",
                                self.model, key)
                return True
        return False

    @staticmethod
    def _retryable_status(exc: Exception) -> bool:
        """Is this `APIStatusError` worth sending again?

        `status_code >= 500` is the obvious half and is not enough on its own:
        tested alone, a plain `overloaded_error` ends a run holding hours of
        RL. The status of a MID-STREAM failure describes the TRANSPORT, not the
        error:

            anthropic/_streaming.py:140
                raise self._client._make_status_error(err_msg, body=body,
                                                      response=self.response)

        `self.response` is the streaming response, which is **200 OK** -- the
        stream opened fine and then delivered an SSE `error` event -- and
        `APIStatusError.__init__` does `self.status_code = response.status_code`.
        So an overload arriving that way reads as 200, a `200 < 500` test takes
        the immediate-raise branch, and MAX_RETRIES never applies. It is also
        why the class is a bare `APIStatusError` rather than
        `InternalServerError`: the SDK's `_make_status_error` maps 4xx/5xx to
        subclasses and 200 to none of them, so catching the subclass would not
        help either.

        The same `__init__` sets `.type` from the SSE body, so the error's own
        name is available and is the right thing to read -- a status that
        describes a different layer is not. `.type` is `None` for a real HTTP
        error (no body of that shape), which is why this stays an OR and not a
        replacement: both halves are load-bearing.
        """
        if int(getattr(exc, "status_code", 0) or 0) >= 500:
            return True
        return str(getattr(exc, "type", "") or "") in RETRYABLE_ERROR_TYPES

    @staticmethod
    def _retry_after(exc: Exception) -> Optional[float]:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        try:
            return float(headers.get("retry-after"))
        except (TypeError, ValueError):
            return None

    def _sleep(self, attempt: int, retry_after: Optional[float]) -> None:
        """Back off, then spread -- and the spread is the half that matters here.

        Jitter on the exponential branch alone would put it on the one path
        that needs it least. `retry-after` is the rate-limit case, so every one
        of this call's in-flight followers is holding the SAME number from the
        same response, and without a spread they all wake in the same
        millisecond and re-hit the limit together. Serial cannot reach that
        state: there is never more than one request in flight to synchronise.
        It is the fan-out (`llm.max_concurrent_requests`) that creates the
        herd, so it is the fan-out width that sizes the spread -- the quantity
        to bound is the peak rather than the mean.

        Added, never subtracted: a `retry-after` is a floor the server named, so
        this can only ever wait longer than instructed. At the default
        `max_concurrent_requests: 1` the spread is `uniform(0, 1)` -- byte for
        byte the jitter the exponential branch always had -- so no serial run
        moves except by up to one extra second on a 429 it was going to lose
        anyway.
        """
        if attempt >= MAX_RETRIES:
            # `_call` makes MAX_RETRIES + 1 attempts, so a retryable failure on
            # the LAST one has no retry left to wait for. Sleeping here would
            # spend up to BACKOFF_MAX_S delaying the terminal LLMError -- and
            # log a "retry 13/12" that is never going to be made.
            return
        base = (retry_after if retry_after is not None
                else min(BACKOFF_MAX_S, BACKOFF_BASE_S * (2 ** attempt)))
        spread = float(max(1, int(getattr(self, "max_concurrent_requests", 1) or 1)))
        delay = base + _JITTER.uniform(0, spread)
        log.info("anthropic retry %d/%d in %.1fs", attempt + 1, MAX_RETRIES, delay)
        time.sleep(delay)


@register("llm", "anthropic", doc="Anthropic Messages API (needs ANTHROPIC_API_KEY).")
def anthropic_factory(ctx: Any, role: str) -> AnthropicClient:
    """Factory for `llm.generator.provider: anthropic` / `llm.evaluator.provider`."""
    return AnthropicClient(ctx, role)
