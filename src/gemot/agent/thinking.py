"""Stop a model call whose thinking runs away, and hand its thinking back for an answer.

A reasoning model can spend its whole output budget thinking and return nothing: GLM 5.3 Flash
deliberated 65k tokens (~25 minutes) a turn without reaching a tool call, hy3 looped on the same
paragraphs. PEtabAgentGym replayed those calls (``work/models/runaway_replay.py``) and tested the
recovery used here (``work/models/handoff_test.py``, 2026-10-10): every hand-off answered with a
usable action, in seconds.

1. Every call is STREAMED, and the thinking text kept as it arrives. The stream is stopped when the
   thinking repeats itself (:class:`RepetitionDetector`: half of the trailing window's 8-word
   sequences already said, ``work/models/repetition.py``); a call the provider truncates
   (``finish_reason == "length"``, nothing said) is handed back too. There is no thinking budget:
   one would mostly cut calls that answer on their own (in sweep_indra_glm13, 64 of the 66 GLM
   calls that thought past 32k tokens ended in a tool call).
2. The HAND-OFF: the same conversation, the thinking so far as the model's own text, and one
   instruction (:data:`HANDOFF`) -- state the gist, then make the next tool call -- with thinking
   off and the call's own output budget: no cap of its own. Its answer becomes the model's turn,
   so the next turn need not re-derive the analysis.
3. A model whose host refuses to stream is called whole, once refused (:data:`NO_STREAM`): it cannot
   be stopped early, but a truncated call's thinking (OpenRouter's ``reasoning``) is handed back
   all the same.

The turn's ``response_metadata["thinking_handoff"]`` records what happened, which
``guards._reasoning_runaway_guard`` counts. Nothing here may end a run: a hand-off that fails hands
back the truncated turn, as before.
"""

from __future__ import annotations

import asyncio
import contextvars
import re
from dataclasses import dataclass, field
from typing import Any

#: Shingle length, trailing window (words) and check interval (characters) of the repetition
#: detector, and the repeated share at which it fires: repetition.py's settings, 0.5 the
#: threshold handoff_test.py cut hy3's loops at.
REPEAT_N, REPEAT_WINDOW, REPEAT_STEP, REPEAT_SHARE = 8, 2000, 2000, 0.5
#: Fewest shingles in the window before the share is read at all.
REPEAT_MIN = 50
HANDOFF = (
    "Your thinking budget for this step is used up; the text above is your reasoning so far. "
    "Without further deliberation: first state in a few sentences the gist of your analysis -- "
    "what you have established and what you will do next -- then make your next tool call."
)
#: OpenRouter's ways to turn thinking off, tried in order; the first a model's host accepts is
#: kept for it (:data:`THINKING_OFF`).
OFF_SETTINGS: tuple[dict, ...] = ({"enabled": False}, {"effort": "none"}, {"effort": "minimal"})

#: Models whose host refused a streamed call this process; they are called whole from then on.
NO_STREAM: set[str] = set()
#: The thinking-off setting each model's host accepted.
THINKING_OFF: dict[str, dict | None] = {}


@dataclass
class Capture:
    """What one call's raw responses carried beside the message langchain builds."""

    text: list[str] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)
    generation_id: str | None = None
    provider: str | None = None
    usage: dict | None = None
    #: The last value of each per-response field a stream repeats (OpenRouter sends
    #: ``finish_reason`` on its final chunk and again on the usage chunk, and langchain's chunk
    #: merge would concatenate the two: ``"tool_callstool_calls"``).
    last: dict = field(default_factory=dict)

    @property
    def thinking(self) -> str:
        return "".join(self.text)


#: The capture of the call in progress, filled by the model class's chunk and result hooks.
CAPTURE: contextvars.ContextVar[Capture | None] = contextvars.ContextVar("thinking_capture", default=None)


def record_chunk(chunk: dict) -> None:
    """Keep a streamed chunk's thinking, reasoning blocks, id, host and usage."""
    cap = CAPTURE.get()
    if cap is None or not isinstance(chunk, dict):
        return
    cap.generation_id = cap.generation_id or chunk.get("id")
    cap.provider = cap.provider or chunk.get("provider")
    if isinstance(chunk.get("usage"), dict):
        cap.usage = chunk["usage"]
    for key in ("model", "system_fingerprint", "service_tier"):
        if chunk.get(key):
            cap.last["model_name" if key == "model" else key] = chunk[key]
    for choice in chunk.get("choices") or []:
        if choice.get("finish_reason"):
            cap.last["finish_reason"] = choice["finish_reason"]
        delta = choice.get("delta") or {}
        text = delta.get("reasoning") or delta.get("reasoning_content")
        if isinstance(text, str):
            cap.text.append(text)
        if isinstance(delta.get("reasoning_details"), list):
            cap.details.extend(d for d in delta["reasoning_details"] if isinstance(d, dict))


def record_response(response: Any) -> None:
    """Keep a whole response's thinking, id and usage (a model that does not stream)."""
    cap = CAPTURE.get()
    if cap is None:
        return
    raw = response if isinstance(response, dict) else response.model_dump()
    cap.generation_id = raw.get("id")
    cap.provider = raw.get("provider")
    cap.usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else None
    for choice in raw.get("choices") or []:
        text = (choice.get("message") or {}).get("reasoning")
        if isinstance(text, str):
            cap.text.append(text)


def merge_details(deltas: list[dict]) -> list[dict]:
    """Streamed ``reasoning_details`` as one block per index: text and summary concatenated in
    order, every other field (type, format, signature, id) its last value."""
    blocks: dict[Any, dict] = {}
    for d in deltas:
        block = blocks.setdefault(d.get("index", 0), {})
        for key, value in d.items():
            if key in ("text", "summary", "data") and isinstance(value, str):
                block[key] = block.get(key, "") + value
            elif value is not None:
                block[key] = value
    return list(blocks.values())


class RepetitionDetector:
    """The share of the trailing window's ``REPEAT_N``-word sequences already said, read every
    ``REPEAT_STEP`` characters (``work/models/repetition.py``, made incremental)."""

    def __init__(self) -> None:
        self.words: list[str] = []
        self.seen: set[tuple[str, ...]] = set()
        self.repeated = [0]  # cumulative count of shingles said before, per shingle
        self.tail = ""
        self.chars = 0
        self.next_check = REPEAT_STEP

    def feed(self, text: str) -> float | None:
        """Read more thinking; the repeated share when a check is due, else None."""
        self.chars += len(text)
        parts = re.split(r"(\s+)", self.tail + text)
        self.tail = parts.pop() if parts and parts[-1] and not parts[-1].isspace() else ""
        for word in (p for p in parts if p and not p.isspace()):
            self.words.append(word)
            if len(self.words) >= REPEAT_N:
                gram = tuple(self.words[-REPEAT_N:])
                self.repeated.append(self.repeated[-1] + (gram in self.seen))
                self.seen.add(gram)
        if self.chars < self.next_check:
            return None
        self.next_check = self.chars + REPEAT_STEP
        n_grams = len(self.repeated) - 1
        lo = max(0, len(self.words) - REPEAT_WINDOW)
        if n_grams - lo < REPEAT_MIN:
            return None
        return (self.repeated[n_grams] - self.repeated[lo]) / (n_grams - lo)


class Watch:
    """Whether a call's thinking should be stopped: ``"repetition"``, or None."""

    def __init__(self) -> None:
        self.detector = RepetitionDetector()

    def feed(self, text: str) -> str | None:
        if not text:
            return None
        share = self.detector.feed(text)
        return "repetition" if share is not None and share >= REPEAT_SHARE else None


def is_stream_refusal(exc: BaseException) -> bool:
    """A host refusing to stream: a 400/422 that names streaming."""
    try:
        import openai
    except ImportError:  # pragma: no cover - the agent extra installs it
        return False
    refusal = (openai.BadRequestError, openai.UnprocessableEntityError)
    return isinstance(exc, refusal) and "stream" in str(exc).lower()


def is_runaway_message(message: Any) -> bool:
    """Truncated with nothing said: ``finish_reason == "length"``, no text, no tool call."""
    from .guards import _content_is_empty

    meta = getattr(message, "response_metadata", None) or {}
    return (
        meta.get("finish_reason") == "length"
        and not getattr(message, "tool_calls", None)
        and _content_is_empty(getattr(message, "content", None))
    )


async def generation_usage(llm: Any, generation_id: str | None, tries: int = 5) -> dict | None:
    """OpenRouter's account of a stopped stream, which never got to send its usage: its tokens and
    cost from ``/generation``, polled while the record is written. None if it never appears."""
    if not generation_id:
        return None
    try:
        import httpx

        key = llm.openai_api_key.get_secret_value() if llm.openai_api_key else None
        url = f"{str(llm.openai_api_base or 'https://openrouter.ai/api/v1').rstrip('/')}/generation"
        async with httpx.AsyncClient(timeout=20) as client:
            for attempt in range(tries):
                await asyncio.sleep(1.0 + attempt)
                r = await client.get(url, params={"id": generation_id}, headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 200:
                    data = r.json().get("data") or {}
                    return {
                        "prompt_tokens": int(data.get("tokens_prompt") or data.get("native_tokens_prompt") or 0),
                        "completion_tokens": int(
                            data.get("tokens_completion") or data.get("native_tokens_completion") or 0
                        ),
                        "cost": data.get("total_cost"),
                    }
    except Exception:  # pragma: no cover - accounting must never break a run
        return None
    return None


def _usage_metadata(usage: dict) -> dict:
    """langchain's usage record of an OpenRouter usage block (cache reads and writes kept)."""
    prompt, completion = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    out = {"input_tokens": prompt, "output_tokens": completion, "total_tokens": prompt + completion}
    cache = {k: int(details[v]) for k, v in (("cache_read", "cached_tokens"), ("cache_creation", "cache_write_tokens")) if details.get(v)}
    if cache:
        out["input_token_details"] = cache
    return out


def _add_usage(blocks: list[dict | None]) -> dict | None:
    """The usage blocks of a turn's calls summed. A cost only if every call reported one: a call
    with no account (a stopped stream OpenRouter never filed) leaves the turn uncosted rather
    than undercounted.

    It is the turn's ``llm_output`` usage, which the tracer reads its token counts from, so it
    carries a total: without one the span recorded prompt and completion alone, and a run's total
    came out short of the two (#125)."""
    known = [b for b in blocks if b]
    if not known:
        return None
    out = {k: sum(int(b.get(k) or 0) for b in known) for k in ("prompt_tokens", "completion_tokens")}
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    cached = sum(int((b.get("prompt_tokens_details") or {}).get("cached_tokens") or 0) for b in known)
    if cached:
        out["prompt_tokens_details"] = {"cached_tokens": cached}
    if len(known) == len(blocks) and all(b.get("cost") is not None for b in known):
        out["cost"] = sum(float(b["cost"]) for b in known)
    return out


async def _call(llm, plain, stream, messages, stop, run_manager, kwargs, watch: Watch | None):
    """One provider call: streamed and watched where the host streams, whole otherwise.

    Returns ``(generation, capture, stopped)``: ``stopped`` names why the stream was cut (the
    generation is then the partial turn, or None if nothing came), else None. The generation's
    info is merged into its message's ``response_metadata`` here, as langchain does only after
    ``_agenerate`` returns, so ``finish_reason`` can be read.
    """
    from langchain_core.language_models.chat_models import generate_from_stream

    cap = Capture()
    token = CAPTURE.set(cap)
    try:
        gen, stopped = None, None
        streamed = llm.model_name not in NO_STREAM
        if streamed:
            chunks, seen = [], 0
            it = stream(messages, stop, run_manager, **kwargs)
            try:
                async for chunk in it:
                    chunks.append(chunk)
                    new = "".join(cap.text[seen:])
                    seen = len(cap.text)
                    if watch is not None and (stopped := watch.feed(new)):
                        break
            except Exception as exc:
                if chunks or not is_stream_refusal(exc):
                    raise
                NO_STREAM.add(llm.model_name)
                streamed = False
                print(f"[thinking] {llm.model_name} refuses to stream ({exc}); calling it whole")
            finally:
                await it.aclose()
            if streamed and chunks:
                gen = generate_from_stream(iter(chunks)).generations[0]
                info = {k: v for k, v in (gen.generation_info or {}).items() if k not in cap.last}
                gen.generation_info = {**info, **cap.last} or None
                for key in cap.last:
                    gen.message.response_metadata.pop(key, None)
        if not streamed:
            gen = (await plain(messages, stop, run_manager, **kwargs)).generations[0]
        if gen is not None:
            gen.message.response_metadata = {**(gen.message.response_metadata or {}), **(gen.generation_info or {})}
        return gen, cap, stopped
    finally:
        CAPTURE.reset(token)


async def generate(llm: Any, plain, stream, messages, stop=None, run_manager=None, **kwargs):
    """The model class's ``_agenerate``: one turn, its runaway thinking handed off."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    gen, cap, stopped = await _call(llm, plain, stream, messages, stop, run_manager, kwargs, Watch())

    def finish(gen, usage, provider, combined: bool):
        msg = gen.message
        if provider:
            msg.response_metadata["openrouter_provider"] = provider
        if usage:
            msg.response_metadata["token_usage"] = usage
            if combined or not msg.usage_metadata:
                msg.usage_metadata = _usage_metadata(usage)
        llm_output = {"model_name": llm.model_name, "token_usage": usage or {}}
        if provider:
            llm_output["openrouter_provider"] = provider
        return ChatResult(generations=[gen], llm_output=llm_output)

    reason = stopped or ("length" if gen is not None and is_runaway_message(gen.message) else None)
    if reason is None:
        if cap.details:
            gen.message.additional_kwargs["reasoning_details"] = merge_details(cap.details)
        return finish(gen, cap.usage, cap.provider, combined=False)

    usage = cap.usage if stopped is None else await generation_usage(llm, cap.generation_id)
    thinking = cap.thinking
    record = {"reason": reason, "thinking_chars": len(thinking), "streamed": llm.model_name not in NO_STREAM}

    def empty(usages, recovered=False):
        msg = AIMessage(content="", response_metadata={"finish_reason": "length", "thinking_handoff": {**record, "recovered": recovered}})
        return finish(ChatGeneration(message=msg), _add_usage(usages), cap.provider, combined=True)

    if not thinking.strip():
        print(f"[thinking] {reason}: no thinking to hand back; handing the empty turn back")
        return empty([usage])
    print(f"[thinking] stopped on {reason} after {len(thinking)} characters of thinking; handing it back")
    handoff = [*messages, AIMessage(content=thinking), HumanMessage(content=HANDOFF)]
    known = llm.model_name in THINKING_OFF
    for setting in (THINKING_OFF[llm.model_name],) if known else OFF_SETTINGS:
        body = {**(llm.extra_body or {}), **(kwargs.get("extra_body") or {}), "reasoning": setting}
        try:
            reply, hcap, hstopped = await _call(
                llm, plain, stream, handoff, stop, run_manager,
                {**kwargs, "extra_body": body}, Watch(),
            )
        except Exception as exc:
            if not known and _is_bad_request(exc):
                continue  # this way of turning thinking off is refused: the next
            print(f"[thinking] the hand-off failed ({type(exc).__name__}: {exc}); handing the empty turn back")
            return empty([usage])
        THINKING_OFF[llm.model_name] = setting
        reply_usage = hcap.usage if hstopped is None else await generation_usage(llm, hcap.generation_id)
        if hstopped is not None or reply is None or is_runaway_message(reply.message):
            print("[thinking] the hand-off ran away too; handing the empty turn back")
            return empty([usage, reply_usage])
        reply.message.response_metadata["thinking_handoff"] = {**record, "recovered": True, "thinking_off": setting}
        return finish(reply, _add_usage([usage, reply_usage]), hcap.provider or cap.provider, combined=True)
    print("[thinking] no way of turning thinking off is accepted; handing the empty turn back")
    return empty([usage])


def _is_bad_request(exc: BaseException) -> bool:
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError))
