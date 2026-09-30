"""OpenAI-protocol client (`llm.<role>.provider: openai`).

Speaks `chat/completions`, which is the lingua franca every non-Anthropic
hosted model answers to. Written for OpenAI-compatible endpoints (e.g.
OpenRouter) but bound to the PROTOCOL, not the host:

  * `OPENAI_API_KEY` (+ optional `OPENAI_BASE_URL`) wins when set;
  * else `OPENROUTER_API_KEY`, with the base URL defaulting to
    `https://openrouter.ai/api/v1`.

The credential and endpoint live in the ENVIRONMENT, never in config, for the
same reason `anthropic_client` reads `ANTHROPIC_API_KEY`: a key is a
credential and a base URL is transport, neither is method identity, and a
resolved config must stay machine-independent. `llm.<role>.model` carries the
routed model id verbatim (`qwen/qwen3-max`, `qwen/qwen3-vl-235b-a22b-instruct`).

What this client deliberately does NOT copy from `anthropic_client`:

  * no prompt-cache breakpoints -- `cache_control` is Anthropic's dialect;
    OpenRouter forwards provider-side implicit caching where it exists, and
    the usage split is recorded when the response reports it;
  * no streaming -- `chat/completions` has no client-side max_tokens ceiling
    to route around, and `max_tokens` is left unset so the model's own output
    limit applies (the callers' outputs are reward programs and JSON verdicts,
    both self-limiting).  ONE EXCEPTION, and it is a repair rather than a
    policy change: a router may inject a DEFAULT completion allowance of its
    own when the request carries no `max_tokens`, and on a small-context model
    that invented allowance can push `prompt + allowance` past the window while
    the prompt itself fits.  `_degrade` recognises that specific 400 and pins
    `max_tokens` to the real headroom.  See `_degrade`;
  * `temperature` and reasoning effort are NOT mutually exclusive here --
    `_one` sends both and drops whichever the far side rejects, remembering
    the drop for the rest of the run exactly as `_degrade` does.
"""
from __future__ import annotations

import base64
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .base import IMAGE_TOKENS, LLMClient, LLMError, Message, estimate_tokens

log = logging.getLogger("bird")


def _int_after(text: str, phrase: str) -> Optional[int]:
    """The first integer following `phrase`, or None.

    Used only to read numbers back out of a provider's own 400 message, which
    is prose and may change wording at any time -- hence None on any surprise
    rather than a best guess.  See `OpenAIClient._pin_max_tokens`.
    """
    m = re.search(phrase + r"\D{0,20}(\d[\d,]*)", text)
    return int(m.group(1).replace(",", "")) if m else None


def _int_before(text: str, phrase: str) -> Optional[int]:
    """The integer immediately preceding `phrase`, or None."""
    m = re.search(r"(\d[\d,]*)\D{0,20}" + phrase, text)
    return int(m.group(1).replace(",", "")) if m else None

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: The anthropic client's retry ladder: 12 retries capped at 180 s rides out a
#: sustained-overload window instead of burning a short ladder (5 retries,
#: 60 s cap) inside it.
MAX_RETRIES = 12
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 180.0

_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}

#: Backoff jitter comes from a private stream: `ctx.rng` is the run's
#: reproducibility contract and network flakiness must not perturb it.
_JITTER = random.Random(0xC0FFEE)


def _retry_after(exc: Any) -> Optional[float]:
    """The server's own `retry-after` seconds, when the SDK exposes it."""
    try:
        value = exc.response.headers.get("retry-after")
        return float(value) if value else None
    except (AttributeError, TypeError, ValueError):
        return None


class OpenAIClient(LLMClient):
    """Budget-accounting wrapper over `client.chat.completions.create`."""

    provider = "openai"

    #: Same contract as the Anthropic client: `chat/completions` requests are
    #: stateless, n i.i.d. samples are n independent round-trips, overlapping
    #: them moves wall clock only.
    concurrent_samples = True

    def __init__(self, ctx: Any, role: str):
        super().__init__(ctx, role)
        if not self.model or self.model.startswith("openai-"):
            # No DEFAULT_MODEL here on purpose: this client exists to reach a
            # NAMED third-party model, and a silent fallback would run a
            # different model than the config hash claims.
            raise LLMError(
                f"llm.{role}.provider=openai requires an explicit llm.{role}.model "
                f"(e.g. qwen/qwen3-max); got {self.model!r}")
        #: Parameters a 400 told us this model rejects; dropped for the run.
        self._dropped: set[str] = set()
        #: Samples the model cut at its own output ceiling (`finish_reason:
        #: length`) and samples it refused -- counted, not just logged, for the
        #: reason anthropic_client gives at DEFAULT_MAX_TOKENS: a run whose
        #: hard-task candidates were all truncated looks identical to one whose
        #: model wrote bad code.
        self._truncated = 0
        self._refused = 0
        #: Per-sample "why empty", index-aligned with `_complete`'s output; the
        #: same contract as `anthropic_client`. `_one` parks
        #: its reason in a thread-local, `_complete` collects it in-thread.
        self._chunk_empty_reasons: List[str] = []
        self._tl = threading.local()
        self._client = self._connect()

    # -- connection --------------------------------------------------------

    def _connect(self) -> Any:
        try:
            from openai import OpenAI  # lazy: optional extra
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise LLMError(
                "llm.*.provider=openai needs the openai SDK: "
                "uv sync --extra openai (or --extra all)") from exc
        key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not key:
            key = os.environ.get("OPENROUTER_API_KEY")
            if key and not base_url:
                base_url = DEFAULT_OPENROUTER_BASE_URL
        if not key:
            raise LLMError(
                "provider=openai found neither OPENAI_API_KEY nor "
                "OPENROUTER_API_KEY in the environment")
        return OpenAI(api_key=key, base_url=base_url, max_retries=0)

    # -- message shaping ---------------------------------------------------

    @staticmethod
    def _image_part(img: Any) -> Optional[Dict[str, Any]]:
        """One `image_url` content part from bytes, a path, a data URL, or a
        bare base64 string. Returns None for something unrecognisable."""
        if isinstance(img, dict):
            return img if img.get("type") == "image_url" else None
        if isinstance(img, (bytes, bytearray)):
            b64 = base64.b64encode(bytes(img)).decode("ascii")
            return {"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}}
        if isinstance(img, str):
            if img.startswith(("http://", "https://", "data:")):
                return {"type": "image_url", "image_url": {"url": img}}
            p = Path(img)
            if p.suffix.lower() in _MEDIA_TYPES and p.is_file():
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                media = _MEDIA_TYPES[p.suffix.lower()]
                return {"type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{b64}"}}
            # assume bare base64 PNG, the shape `sample_frames` hands over
            return {"type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"}}
        return None

    def _payload(self, messages: List[Message],
                 images: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
        """`chat/completions` turns; images ride BEFORE the text of the last
        turn, mirroring `anthropic_client._attach_images`."""
        turns: List[Dict[str, Any]] = [dict(m) for m in messages]
        if images and self.modality == "vlm":
            parts = [p for p in (self._image_part(i) for i in images) if p]
            if parts and turns:
                last = turns[-1]
                last["content"] = parts + [
                    {"type": "text", "text": str(last.get("content", ""))}]
        elif images:
            log.warning("images passed to a text-modality %s client; ignoring",
                        self.role)
        return turns

    # -- transport ---------------------------------------------------------

    def _one(self, turns: List[Dict[str, Any]], temperature: float,
             n_img: int) -> str:
        import openai  # already imported by _connect; cheap

        params: Dict[str, Any] = {"model": self.model, "messages": turns}
        if "temperature" not in self._dropped:
            params["temperature"] = float(temperature)
        if self.reasoning_effort != "none" and "reasoning" not in self._dropped:
            # OpenRouter's unified reasoning parameter; forwarded per provider.
            params["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(**params)
            except openai.BadRequestError as exc:
                if self._degrade(params, str(exc)):
                    continue  # dropped a rejected pin; retry (consumes one
                    # attempt slot -- at most two pins exist, so the ladder
                    # loses nothing that matters)
                raise LLMError(f"openai 400: {exc}") from exc
            except openai.RateLimitError as exc:
                self._backoff(attempt, exc, retry_after=_retry_after(exc))
                continue
            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                self._backoff(attempt, exc)
                continue
            except openai.APIStatusError as exc:
                if exc.status_code >= 500:
                    self._backoff(attempt, exc)
                    continue
                raise LLMError(f"openai {exc.status_code}: {exc}") from exc

            choice = (resp.choices or [None])[0]
            message = getattr(choice, "message", None) if choice else None
            text = (getattr(message, "content", None) or "") if choice else ""
            finish = getattr(choice, "finish_reason", None) if choice else None
            # The empty-sample contract of `anthropic_client._one`, mirrored
            # here for the two ways chat/completions says "this is not a
            # complete answer": `finish_reason: length` is the model's own
            # output ceiling (this client sends no max_tokens, so there is no
            # cap of ours to raise), and a refusal arrives either as
            # `message.refusal` or as `finish_reason: content_filter`. Returned
            # as `text`, either would be parsed: a program cut mid-token has no
            # closing fence, `_extract_code` falls through to the docstring
            # pattern, and verification records a SyntaxError against the model
            # with `budget.llm_truncations` at 0.
            refusal = getattr(message, "refusal", None) if message is not None else None
            truncated = finish == "length"
            refused = bool(refusal) or finish == "content_filter"
            usage = getattr(resp, "usage", None)
            out_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
            if usage is not None:
                details = getattr(usage, "prompt_tokens_details", None)
                cached = int(getattr(details, "cached_tokens", 0) or 0)
                self._record(int(getattr(usage, "prompt_tokens", 0) or 0),
                             out_tokens, cache_read_tokens=cached,
                             refused=refused, truncated=truncated)
            else:
                # No usage block: estimate, matching base.py's convention that
                # images bill IMAGE_TOKENS apiece when the provider is silent.
                approx = sum(estimate_tokens(str(t.get("content", "")))
                             for t in turns)
                self._record(approx, estimate_tokens(text), n_images=n_img,
                             refused=refused, truncated=truncated)
            if truncated:
                # Never returned as if complete -- empty, warned and counted,
                # with the cause where the reader will look.
                with self._record_lock:
                    self._truncated = getattr(self, "_truncated", 0) + 1
                log.warning("openai(%s) stopped with finish_reason=length (%d output "
                            "tokens); the sample is cut mid-program and is being "
                            "discarded rather than parsed.", self.model, out_tokens)
                self._park_empty_reason(
                    f"truncated at the model's output limit (finish_reason=length, "
                    f"{out_tokens} output tokens)")
                return ""
            if refused:
                with self._record_lock:
                    self._refused = getattr(self, "_refused", 0) + 1
                why = (f"message.refusal={str(refusal)[:120]!r}" if refusal
                       else f"finish_reason={finish}")
                log.warning("openai(%s) refused (%s); returning an empty sample",
                            self.model, why)
                self._park_empty_reason(f"refused by the provider ({why})")
                return ""
            if not text:
                log.warning("openai(%s) returned an empty message "
                            "(finish_reason=%r)", self.model, finish)
                self._park_empty_reason(f"provider returned an empty message "
                                        f"(finish_reason={finish!r})")
            return text

        raise LLMError(
            f"openai: {MAX_RETRIES} retries exhausted for model {self.model}")

    # -- empty-sample reasons: parked per thread, collected per sample ------
    #
    # Verbatim the anthropic client's mechanism: `_one` runs in whichever
    # thread the fan-out gave it and cannot know its sample index, so it parks
    # the reason thread-locally and the caller that DOES know the index reads
    # it back in the same thread right after `_one` returns. `getattr`
    # throughout because the transport tests build this client with
    # `__init__` bypassed.

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

    def _degrade(self, params: Dict[str, Any], error_text: str) -> bool:
        """Drop a parameter the far side names in a 400, remember it, retry."""
        lowered = error_text.lower()
        if "temperature" in lowered and "temperature" in params:
            params.pop("temperature")
            self._dropped.add("temperature")
            log.warning("openai(%s) rejected temperature; dropping it for this "
                        "run: %s", self.model, error_text.strip()[:200])
            return True
        if ("reasoning" in lowered or "effort" in lowered) \
                and "extra_body" in params:
            params.pop("extra_body")
            self._dropped.add("reasoning")
            log.warning("openai(%s) rejected the reasoning parameter; dropping "
                        "it for this run: %s", self.model,
                        error_text.strip()[:200])
            return True
        if self._pin_max_tokens(params, error_text):
            return True
        return False

    #: Router-injected completion allowances are not a paper hyperparameter, so
    #: the pin leaves this much of the real headroom unused rather than asking
    #: for every last token: enough that a near-boundary prompt still gets a
    #: usable answer, small enough that it never costs a real completion.
    _HEADROOM_MARGIN_TOKENS = 64

    #: Below this, a pinned completion is not worth PAYING FOR: the answer would
    #: almost certainly come back cut, and a 400 that reaches the caller is a
    #: clearer signal than a billed call that returns nothing usable.
    #:
    #: The floor is NOT about truncated programs being misleading: `_one`
    #: already handles that one layer down -- `finish_reason == "length"` makes
    #: the sample EMPTY and counts it in `_truncated`, so a cut program is never
    #: parsed as a bad one. The floor is about cost and signal, not safety.
    _MIN_USEFUL_COMPLETION_TOKENS = 256

    def _pin_max_tokens(self, params: Dict[str, Any], error_text: str) -> bool:
        """Repair a `context_length_exceeded` caused by a completion allowance
        THIS CLIENT NEVER ASKED FOR.

        The client sends no `max_tokens` on purpose (see the module docstring).
        Some routers fill that in with a default of their own, and then reject
        the request for a total the caller never requested::

            This model's maximum context length is 8192 tokens. However, you
            requested 9281 tokens (5871 in the messages, 3410 in the
            completion).

        The prompt there is 5,871 against a 8,192 window -- it FITS, with 2,321
        to spare -- and the 3,410 is the router's invention.  Against the
        provider's own API the default is "the rest of the window" and the same
        call succeeds.  So this is not the model's context limit binding on the
        method; it is an artifact of the path the request took, and pinning
        `max_tokens` to the real headroom removes the artifact without changing
        anything the caller asked for.

        Measured on a Eureka search: that exact 400 ended the search 8.6 hours
        in, entering iteration 2 of 5, and the largest of the 16 completions
        that DID come back was 461 tokens -- 19.9% of the 2,321 headroom.
        Nothing in that run was close to needing what the router reserved.

        REFUSES rather than guesses.  Returns False -- letting the 400 through
        -- when the numbers are not both parseable, when `max_tokens` is already
        pinned (so this fires at most once and cannot loop), or when the real
        headroom is too small to hold a usable answer.  A prompt that genuinely
        does not fit its window must fail loudly; quietly shrinking the answer
        until it does is how a truncated reward function becomes a datapoint.
        """
        # PER-CALL, not per-run.  `params` is rebuilt by `_one` on every call, so
        # `"max_tokens" in params` is true only for a retry WITHIN this call -- which is
        # exactly the loop that must not happen -- and false for the next call, which
        # must be free to repair itself.  Latching on the run-scoped `self._dropped`
        # instead would make the repair fire ONCE per client and every later 400
        # raise, and since Eureka sends 16 candidates an iteration against a growing
        # prompt, a second occurrence is near-certain.  Re-deriving the headroom per
        # call is also the more correct behaviour, because it depends on THAT call's
        # prompt.
        if "max_tokens" in params:
            return False
        lowered = error_text.lower()
        if "context" not in lowered and "maximum context length" not in lowered:
            return False

        limit = _int_after(lowered, r"maximum context length is")
        used = _int_before(lowered, r"in the messages")
        if limit is None or used is None or used >= limit:
            return False

        # SANITY-CHECK THE PARSED PROMPT SIZE AGAINST OUR OWN ESTIMATE, because these
        # numbers come out of provider prose and the parse is positional.  A reworded
        # message that reads as "... (0 in the messages, N in the completion)" would
        # otherwise pin `max_tokens` to the ENTIRE window on a prompt that genuinely
        # fills it -- turning a clean 400 into a second doomed call.  The tolerance
        # is deliberately loose: tokenisers differ, and this is a sanity floor, not a
        # second opinion.
        #
        # Known limit, stated rather than left to be found: on a VLM request the
        # content is a list of parts whose image payloads are base64 data URIs, so
        # `estimate_tokens` over `str(...)` of them OVER-counts badly and this check
        # can refuse a repair it should have allowed.  The consequence is the 400
        # propagating -- exactly the behaviour before this method existed -- so the
        # failure mode is the status quo ante rather than a wrong number, and a
        # too-eager sanity floor is the right direction for one to fail in.
        try:
            ours = sum(estimate_tokens(m.get("content"))
                       for m in (params.get("messages") or [])
                       if isinstance(m, dict))
        except Exception:
            ours = None
        if ours and used < ours // 2:
            log.warning(
                "openai(%s) 400 names %d prompt tokens but this request estimates "
                "~%d; not trusting the parse, letting the 400 through",
                self.model, used, ours)
            return False

        room = limit - used - self._HEADROOM_MARGIN_TOKENS
        if room < self._MIN_USEFUL_COMPLETION_TOKENS:
            return False

        params["max_tokens"] = room
        log.warning(
            "openai(%s) 400 context_length_exceeded with a completion allowance "
            "this client did not send (window %d, prompt %d); pinning "
            "max_tokens=%d -- the real headroom -- and retrying once: %s",
            self.model, limit, used, room, error_text.strip()[:200])
        return True

    @staticmethod
    def _backoff(attempt: int, exc: Exception,
                 retry_after: Optional[float] = None) -> None:
        if attempt >= MAX_RETRIES:
            # raise BEFORE sleeping: a doomed attempt owes no farewell nap
            # (the same rule as `anthropic_client._sleep`).
            raise LLMError(f"openai: retries exhausted: {exc}") from exc
        if retry_after is not None and 0 < retry_after <= BACKOFF_MAX_S:
            delay = float(retry_after)
        else:
            delay = min(BACKOFF_MAX_S, BACKOFF_BASE_S * (2 ** attempt))
            delay *= 0.5 + _JITTER.random()
        log.warning("openai retry %d/%d in %.1fs: %s",
                    attempt + 1, MAX_RETRIES, delay, str(exc)[:200])
        time.sleep(delay)

    # -- the contract ------------------------------------------------------

    def _complete(self, messages: List[Message], n: int, temperature: float,
                  images: Optional[Sequence[Any]], tag: str) -> List[str]:
        turns = self._payload(messages, images)
        n_img = len(images or ()) if self.modality == "vlm" else 0

        if n <= 1 or self.max_concurrent_requests <= 1:
            out: List[str] = []
            reasons: List[str] = []
            for _ in range(n):
                out.append(self._one(turns, temperature, n_img))
                reasons.append(self._take_empty_reason())
            self._chunk_empty_reasons = reasons
            return out

        # Leader-then-followers, exactly as `anthropic_client._complete`: the
        # leader primes any provider-side implicit prompt cache, the followers
        # overlap, order is by index, and the first failure cancels whatever
        # has not started (in-flight requests finish and are recorded). Each
        # worker collects its own empty reason IN ITS THREAD, so the reason
        # stays with its sample through the out-of-order completions.
        first = self._one(turns, temperature, n_img)
        first_reason = self._take_empty_reason()

        def _sample():
            text = self._one(turns, temperature, n_img)
            return text, self._take_empty_reason()

        with ThreadPoolExecutor(
                max_workers=min(n - 1, self.max_concurrent_requests),
                thread_name_prefix=f"bird-llm-{self.role}") as pool:
            futures = [pool.submit(_sample) for _ in range(n - 1)]
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


from ..registry import register  # noqa: E402  (import kept beside its one use)


@register("llm", "openai",
          doc="OpenAI-protocol chat/completions client. Needs OPENAI_API_KEY "
              "(+ optional OPENAI_BASE_URL) or OPENROUTER_API_KEY; the openai "
              "SDK is imported lazily at client construction.")
def openai_factory(ctx: Any, role: str) -> OpenAIClient:
    """Factory for `llm.generator.provider: openai` / `llm.evaluator.provider`."""
    return OpenAIClient(ctx, role)
