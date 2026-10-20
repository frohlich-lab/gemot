"""``run_deepagents`` — assemble the panel's graph, then drive it to a finish.

Everything the loop needs is built by this module's siblings (see the package docstring); what
lives here is the ASSEMBLY (curate the petab tools, discover the bio tools, wrap the breaker and
the literature recorder, build the subagents, attach the four middleware guards and the
checkpointer) and the RUN LOOP around ``astream``.

The loop is where a run's failure modes are absorbed rather than thrown away:

* a per-super-step **stall watchdog**, sized to the run's real fit budget so a long multistart
  is not mistaken for a hang;
* a **wall-clock deadline** and the server's own early-stop signal;
* **resume from the last checkpoint** on a transient provider failure, instead of abandoning the
  run;
* **bounced silent stops** — a graph that ends on a no-tool turn is pushed back into the finish
  gate (with the build nudge when nothing was authored at all), up to ``_MAX_SILENT_BOUNCES``;
* and an honest **relabel** at the end, so a degenerate run is reported as ``no_tools_executed``
  / ``no_model_built`` / ``silent_stop`` rather than as a "completed" one with nothing in it.

Returns the run's stop reason plus the accounting the runner writes into ``result.json``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
import functools
import inspect
import re
import time
from typing import Any

from .biocurator import (
    LITERATURE_BIO_TOOLS,
    _build_biocurator_subagent,
    discover_bio_tools,
)
from .circuit_breaker import (
    _disabled_tool_filter,
    _is_call_timeout,
    _new_breaker_state,
    _wrap_tools_with_circuit_breaker,
    call_outcomes,
)
from .finish_gate import _finish_tool
from .guards import (
    _empty_report_guard,
    _leaked_argument_repair,
    _statements_only_guard,
    _task_arg_guard,
    _tool_dispatch_guard,
    _reasoning_runaway_guard,
    new_guard_stats,
)
from .literature import _wrap_literature_recorder
from .probe import _model_committed, _structured
from .subagents import (
    _build_subagents,
    _builtin_tool_filter,
    _disable_general_purpose_subagent,
    _general_purpose_enabled,
)
from .token_usage import _LEAD_AGENT_NAME, _token_accountant
from .tools import ORCHESTRATOR_TOOLS, _select_tools, _trim_descriptions
from ..prompts import INDRA_STATEMENTS_ENABLED, _NO_MODEL_NUDGE, _SILENT_FINISH_NUDGE
from .. import models, prompts

# --- guards on the (input-dominated, occasionally-hanging) LLM loop -----------------------
# Transient provider failures (connection drops, malformed/non-JSON bodies, 5xx/429, hung
# calls) are common on OpenRouter and used to abandon the whole run. Now they RETRY instead of
# throwing the run away, in two layers:
#   * per-CALL: the OpenAI client's own ``max_retries`` retries connection errors, request
#     timeouts, and 5xx/429 (honouring Retry-After) transparently, so those never reach the loop.
#   * per-RUN resume: the agent is compiled with a LangGraph checkpointer, so anything that still
#     escapes — a malformed non-JSON body the client won't retry, exhausted call-retries, or a
#     STALL the client timeout can't catch — RESUMES ``astream`` from the last checkpoint instead
#     of aborting, so every committed model version and the conversation state are preserved.
_LLM_TIMEOUT_S = 120.0
_LLM_MAX_RETRIES = 4  # OpenAI-client retries for connection / timeout / 5xx / 429
_RUN_RESUME_ATTEMPTS = 3  # resume the graph from the last checkpoint after a stall/transient
# Harness-level stall watchdog on the agent loop. The client `timeout` above does NOT reliably
# catch a mid-stream provider stall (observed: a 24-min no-progress hang that the 120s timeout
# never fired on). astream emits once per graph super-step (a lead LLM call, a tool, or a whole
# analyst sub-agent), so if NO step completes within this window the loop is wedged — we cancel
# it and RESUME from the last checkpoint (or, once resume attempts run out, end gracefully).
# Floor for one super-step. run_deepagents takes max(this, fit_time_budget(run default)) as its
# baseline and raises it per step to the budget of the fit that step is about to run
# (fit_budget.py), since a lead commit's fit is one long super-step that legitimately streams
# nothing back for minutes. On the large examples the fit-derived term is bigger and this floor
# never binds; on the small ones it IS the window (blasi's fit term is ~1063s), so a genuinely
# wedged non-fit step now goes undetected for 1800s where it used to be 1063s -- the accepted
# price of telling "hung" from "thinking" on the slow provider measured below.
# Sized from MEASURED provider latency, not from a guess: over 282
# ChatOpenAI calls on z-ai/glm-5.3-flash the median is 58s, p90 352s, p99 772s and the slowest
# SUCCESSFUL call 842s -- against openai/gpt-5.6-luna at median 5.1s / max 44.5s over 162 calls.
# At the old 600s floor the watchdog could not tell "hung" from "thinking" on the slow provider,
# and cancelling mid-call is expensive: the cancel unwinds through LangGraph -> model node ->
# httpx (the innermost frame is a TLS read awaiting bytes that never came), which is where the
# CancelledError spans on `model`/`tools`/subagent nodes come from -- one hung call produces four
# to six of them, so they read as many failures when they are one.
#
# Retrying does NOT substitute for this. A stall is already retriable and resumes from the
# checkpoint `_RUN_RESUME_ATTEMPTS` times; measured on a glm blasi run, all three resumes stalled
# at the identical 1063s window because each resume re-sends the whole history and so re-hits the
# same large prompt. That run spent 4252s of its 4968s wall waiting. More retries buy nothing
# against a deterministic stall; a window wide enough for the provider's real tail does.
_STEP_STALL_TIMEOUT_S = 1800.0

#: Sentinel for "no emission has been observed yet", kept distinct from the ``None`` that
#: :func:`_state_message_count` returns for an UNREADABLE state. Conflating them let a run whose
#: very FIRST super-step hangs — nothing emitted, so no count — score the resume's checkpoint
#: replay as progress (`count != None`) and wipe the escalation it had just earned, retrying at
#: the same window that had already failed.
_NO_EMISSION_YET = object()

# Hard per-run token ceiling — the ONLY bound on how much a run can spend.
#
# `recursion_limit` does not bound this and cannot be made to: it caps LangGraph SUPER-STEPS, but
# a deepagents subagent makes all of its LLM calls inside ONE parent super-step, so its calls are
# invisible to the limit. Observed: a run with recursion_limit=250 made 1,493 LLM calls and
# consumed 107.9M prompt tokens (~$51 on one model, in one run). Two multipliers compound it:
# every resume re-enters astream with a FRESH step budget (so the effective ceiling is
# _RUN_RESUME_ATTEMPTS + 1 times whatever you set), and each resume re-sends the whole history
# because there is no trimmer — a deliberate choice that is near-free at ~98% cache_read but
# full price on a weakly-caching model.
#
# 20M is ~1.5x the largest legitimate run measured across 339 runs (13.2M, luna-pro on schwen at
# 155 calls) and below all three observed runaways (24.4M / 43.4M / 107.9M). Counted in tokens,
# not billed dollars.
_MAX_RUN_TOKENS = 20_000_000

# Provider content-policy refusals, matched on the response text because they arrive as a plain
# 400 with no distinguishing exception type: OpenAI/OpenRouter raise ValueError/BadRequestError
# carrying the provider's JSON. Kept deliberately NARROW — a stray 400 (a malformed request, a bad
# parameter) must NOT be relabelled as a content refusal, so a match needs an explicit
# moderation/policy phrase, not merely the status code.
_CONTENT_FILTER_MARKERS = (
    "flagged for possible biological risk",
    "content was flagged",
    "content_policy_violation",
    "content policy",
    "responsible_ai_policy",
    "flagged by the moderation",
    "moderation_blocked",
)


#: Tools whose call runs a full pyPESTO multistart, i.e. the ones whose `n_starts` argument
#: decides how long the next super-step legitimately takes.
#:
#: Derived from the lead's OWN tool list, not hand-written: `make_changes` has been renamed once
#: already (`set_model`/`change_parameters` -> `make_changes`, #130), and a hand-maintained copy
#: would silently stop matching on the next rename -- `_pending_fit_starts` would return None for
#: every step, every fit would run under the run-default window, and bachmann task 9's failure
#: would come back with a green test suite. Intersecting the roster means a rename empties this
#: set rather than half-matching, and tests/test_fit_timeouts asserts it is non-empty -- so the
#: rename fails a test instead of quietly disarming the watchdog. `make_changes` is the only
#: entry: there is no separate `fit` tool, and `get_fit_diagnosis` reads a fit rather than
#: running one.
#:
#: The analysts cannot fit at all -- no analyst toolset includes it, and deepagents' auto-added
#: `general-purpose` subagent that once called it from inside a delegation is disabled in
#: `subagents.py`. That is what makes reading the LEAD's own message a COMPLETE view of the
#: pending fit rather than a sample of it.
_FIT_TOOLS = frozenset({"make_changes"}) & frozenset(ORCHESTRATOR_TOOLS)

#: How far back to look for the AIMessage carrying the pending tool calls. Small on purpose: a
#: middleware node or two may emit after it, but anything deeper than this means the tool node
#: is not what runs next, and scanning the whole history would re-arm on ancient calls.
_PENDING_SCAN_DEPTH = 6


def _env_token_limit(name: str, default: int, *, minimum: int) -> int:
    """An int >= ``minimum`` from ``name``, or ``default`` for an unset, empty, or malformed
    override — a bad value must degrade, never crash the run before the graph is built (an empty
    ``NAME=`` in a submit script is the natural way to write "leave it alone"). Distinct from
    fit_budget._env_int, which enforces >= 1: the output cap documents 0 as its OFF switch, so
    it needs ``minimum=0``."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw.strip())
    except ValueError:
        print(f"[agent] ignoring {name}={raw!r} (not an integer); using {default}")
        return default
    if val < minimum:
        print(f"[agent] ignoring {name}={val} (< {minimum}); using {default}")
        return default
    return val


def _is_tool_result(msg: Any) -> bool:
    """True for a ToolMessage — matched by ``type``/``role``, not by class, so a serialized
    message reads the same as an object one and neither needs a langchain import here."""
    kind = getattr(msg, "type", None)
    if kind is None and isinstance(msg, dict):
        kind = msg.get("type") or msg.get("role")
    return str(kind or "") == "tool"


def _state_message_count(state: Any) -> int | None:
    """Number of messages in a ``stream_mode="values"`` emission, or ``None`` if unreadable.

    Never raises: this feeds the stall watchdog's checkpoint-replay test. ``None`` is distinct
    from ``0`` on purpose — an unreadable state must degrade to "treat this as progress" (the
    behaviour before the replay test existed), whereas a real empty state is genuinely no
    progress. Reporting 0 for both would leave the window permanently widened after one odd
    emission.
    """
    if isinstance(state, dict):
        msgs = state.get("messages")
        if isinstance(msgs, list):
            return len(msgs)
    return None


def _pending_fit_starts(state: Any, default: int, stats: dict | None = None) -> int | None:
    """``n_starts`` of the fit this state leaves PENDING, or ``None`` if no fit is pending.

    ``astream(..., stream_mode="values")`` emits the graph state after each node, so the
    ``AIMessage`` carrying ``make_changes(n_starts=300)`` arrives one super-step BEFORE the tool
    node runs that fit. Reading it here is what lets the stall watchdog size the very wait during
    which the fit happens, instead of guessing a multiple of the run default.

    With more than one fit pending the counts are SUMMED (they contend for one worker pool —
    see the loop). ``None`` and "pending at the default count" are deliberately DIFFERENT answers
    even though they size the same window. The server's own docstring tells the model to omit ``n_starts``
    unless it wants more than the default, so the commonest wedged-fit case is a pending fit with
    no count attached — reporting that as "no fit pending" would send every such post-mortem to
    the provider/LLM layer while a full multistart was in flight.

    Degrades to ``None`` when the state does not have the expected shape: a watchdog must not
    raise. That is precision loss in the direction of the old bug, so it is counted into
    ``stats["unreadable_graph_state"]`` (which reaches ``result.json``) rather than being
    absorbed silently, and the first occurrence prints.
    """
    try:
        if not isinstance(state, dict):
            # stream_mode="values" emits the graph state as a mapping. A langgraph that switches
            # to a pydantic state object would land here and disarm the watchdog for the whole
            # run, so this is a WARN path, not a quiet `return None`.
            raise TypeError(f"state is {type(state).__name__}, not a mapping")
        messages = state.get("messages")
        if not messages:
            return None  # start of the run / empty state
        # Walk BACK to the newest message that carries tool_calls, stopping at the first
        # ToolMessage. Reading only messages[-1] was too brittle: deepagents/langchain
        # middleware hooks (after_model, the 0.85 compaction node, the dispatch and
        # empty-report guards) are graph NODES, each with its own stream_mode="values"
        # emission, so anything landing a message between the AIMessage and the tool node
        # would silently drop the window to the floor -- bachmann task 9 again, and invisible,
        # because the state SHAPE is fine so `unreadable_graph_state` never fires.
        #
        # The ToolMessage stop is what keeps this from re-arming on a fit that already ran:
        # once results are back the fit is over, however many messages follow it.
        calls = None
        for msg in reversed(messages[-_PENDING_SCAN_DEPTH:]):
            if _is_tool_result(msg):
                break
            found = getattr(msg, "tool_calls", None)
            if found is None and isinstance(msg, dict):  # serialized message
                found = msg.get("tool_calls")
            if found:
                calls = found
                break
            if not isinstance(msg, dict) and not hasattr(msg, "content"):
                raise TypeError(f"unrecognised message object {type(msg).__name__}")
        if calls is None:
            return None  # no fit-shaped call pending
        pending = None
        for c in calls:
            # Per call: a malformed entry must not discard a VALID sibling's count. Everything
            # that can raise on a hostile shape goes inside -- `c.get`, and `args.get` when
            # `args` arrives as a JSON string rather than a dict (AttributeError, which is why
            # the suppress is not narrowed to TypeError/ValueError).
            with contextlib.suppress(Exception):
                if str(c.get("name") or "") not in _FIT_TOOLS:
                    continue
                n = (c.get("args") or {}).get("n_starts")
                # Only the COUNT is decided here, for the window and for the stall message; the
                # default floor and the ceiling live in fit_time_budget_for so the watchdog and
                # the transport cut cannot disagree. max(1, ...) matches the server's own
                # `_coerce_n_starts` lower bound -- without it a `n_starts=-5` the server will
                # reject still reaches the post-mortem as "-5-start fit pending".
                asked = default if n is None or isinstance(n, bool) else max(1, int(n))
                # SUM across a batch, not max. langgraph's ToolNode dispatches a message's
                # tool calls concurrently, but `make_changes` holds `session.lock` for its whole
                # change-and-fit critical section (petab_mcp/server.py), so two fits in one
                # emission SERIALIZE: the second waits out the first entirely before its own
                # starts. The super-step's wall clock is therefore the sum, and max() would size
                # the step for one of them and fire the watchdog mid-batch on a healthy run.
                pending = asked if pending is None else pending + asked
        return pending
    except Exception as exc:
        # Silent degradation here is the OLD BUG returning: every fit falls back to the
        # run-default window, a legitimately larger one is killed as "stalled", and the stall
        # line reports "no fit pending" while a full multistart is in flight. Nothing else in
        # the artifacts would distinguish that from a genuine stall -- so record it where a
        # sweep can count it (guard_stats is spread into result.json), and print the first one.
        if stats is not None:
            stats["unreadable_graph_state"] = stats.get("unreadable_graph_state", 0) + 1
            first = stats["unreadable_graph_state"] == 1
        else:
            first = False
        if first:
            print(
                f"[agent] cannot read pending fits from the graph state "
                f"({type(exc).__name__}: {exc}) — the stall watchdog falls back to the "
                "run-default window, which can kill a legitimately larger fit and will "
                "report it as 'no fit pending'"
            )
        return None


def _content_filter_stop_reason(exc: BaseException) -> str | None:
    """``content_filtered: <short reason>`` if ``exc`` is a provider content refusal, else None.

    The refusal is the model provider declining the request, not a harness or model failure, and
    it is not retriable — so the caller uses this to end the run with a countable outcome rather
    than a per-run-unique exception string.
    """
    text = str(exc)
    low = text.lower()
    if not any(m in low for m in _CONTENT_FILTER_MARKERS):
        return None
    # Surface the provider's own sentence, not the whole JSON envelope. The closing quote must
    # match the one that OPENED the message -- taking whichever of `',` / `",` comes first would
    # cut at a quote belonging to the message body.
    for opener, closer in (("'message': '", "'"), ('"message": "', '"')):
        i = text.find(opener)
        if i == -1:
            continue
        rest = text[i + len(opener) :]
        # The message ends at the first closing quote that is followed by the JSON punctuation
        # that can legally follow a value. `rfind` would run past the message and land on the
        # quote of a LATER key (`'code'`); a plain `find` would stop at a quote inside the
        # message body. Both were wrong in earlier versions of this function.
        end = -1
        for punct in (",", "}"):
            j = rest.find(closer + punct)
            if j != -1 and (end == -1 or j < end):
                end = j
        detail = (rest[:end] if end != -1 else rest)[:200]
        return f"content_filtered: {detail.strip()}"
    return f"content_filtered: {text.strip('{} ')[:200]}"


def _extra_body(model: str, reasoning_effort: str | None = None) -> dict[str, Any]:
    """OpenRouter-specific request fields: the billed-cost flag (``usage.include``, always), and
    provider pinning and reasoning where the model has an entry.

    Caching is not here: see :func:`_apply_cache_breakpoints`.
    """
    body: dict[str, Any] = {"usage": {"include": True}}
    if (routing := models.provider_routing(model)) is not None:
        body["provider"] = routing
    if (reasoning := models.reasoning_config(model)) is not None:
        body["reasoning"] = reasoning
    if reasoning_effort is not None:
        # The run's own effort (--reasoning-effort) replaces the model's setting.
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def _mark_cacheable(message: dict, config: dict) -> bool:
    """Put a cache breakpoint on one OpenAI-format message; ``False`` if it has no non-blank text
    to mark, in which case the caller walks further back."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        message["content"] = [{"type": "text", "text": content, "cache_control": config}]
        return True
    if isinstance(content, list):
        for block in reversed(content):
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                block["cache_control"] = config
                return True
    return False


def _apply_cache_breakpoints(payload: dict, config: dict) -> None:
    """Mark the system message and the last markable message of ``payload`` as cache breakpoints
    — two of Anthropic's four, on the CONVERTED payload (docs/RATIONALE.md)."""
    messages = payload.get("messages") or []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            _mark_cacheable(message, config)
            break
    for message in reversed(messages):
        if isinstance(message, dict) and _mark_cacheable(message, config):
            break


@functools.lru_cache(maxsize=None)
def _chat_model_class():
    """``ChatOpenAI``, subclassed to keep the UPSTREAM endpoint identity OpenRouter reports.

    OpenRouter returns a top-level ``provider`` naming the endpoint that actually served the call
    (``"OpenAI"``, ``"Azure"``, ``"Novita"``, …). ``langchain_openai`` drops it: its ``llm_output``
    hard-codes ``model_provider: "openai"`` — the name of the INTEGRATION, not of the endpoint —
    so the served endpoint is invisible downstream. That is the one field an A/B needs, because
    ``provider.only`` bounds routing to a SET and every member of that set is a different
    machine: an arm can drift across endpoints mid-sweep and nothing in the record would say so.

    ``response.model_dump()`` already carries ``provider`` (pydantic keeps unknown top-level
    fields as extras), so this only copies it where the callbacks can read it. Everything else is
    ``super()``'s. Best-effort by construction: any failure leaves the unmodified result, since
    provenance capture must never break a run.

    It also copies a stream's usage chunk onto ``response_metadata["token_usage"]``, where a
    non-streamed response already carries it.

    And it carries each turn's REASONING back to the provider. ``langchain_openai`` drops
    OpenRouter's ``reasoning_details`` both ways, so every request after the first sent the
    assistant's tool-call turns without the thinking that produced them; Anthropic then runs the
    next call without thinking at all (Haiku 4.5 reasoned on 0 of 1,466 calls after a tool result).
    The details ride the AIMessage's ``additional_kwargs`` and are put back, untouched (their
    signatures are verified), on that assistant message of every later request.

    ``lru_cache``d, and the caller passes ``name="ChatOpenAI"`` -- both for reasons that bite
    silently. LangChain derives a run's name (and openinference the SPAN name) from the class name,
    so an un-named subclass would rename every LLM span to ``_ChatOpenAIRecordingProvider``: every
    archived trace and every span query under ``slurm/trace_review`` keys on ``ChatOpenAI``, so the
    new runs would simply return no rows and could not be compared with the old ones. The cache is
    because each call would otherwise build -- and permanently register -- a fresh pydantic model
    class. Keeping the import inside the function is what ``tests/test_import_lightness`` needs.
    """
    from langchain_openai import ChatOpenAI

    from . import thinking

    class _ChatOpenAIRecordingProvider(ChatOpenAI):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            _return_reasoning(self._convert_input(input_).to_messages(), payload)
            config = models.caching_config(payload.get("model") or self.model_name)
            if config is not None:
                _apply_cache_breakpoints(payload, config)
            return payload

        # Every call goes through `_agenerate`, which streams it itself and stops a runaway's
        # thinking (agent.thinking); langchain's own streaming path would bypass that.
        def _should_stream(self, **kwargs):
            return False

        def _should_use_protocol_streaming(self, **kwargs):
            return False

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            return await thinking.generate(
                self, super()._agenerate, self._astream, messages, stop, run_manager, **kwargs
            )

        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            _keep_reasoning(response, result)
            thinking.record_response(response)
            try:
                served = getattr(response, "provider", None)
                if served is None and isinstance(response, dict):
                    served = response.get("provider")
                if not served:
                    return result
                # Both places: `llm_output` is what on_llm_end sees as an aggregate, and the
                # per-message metadata is what Phoenix exports onto the LLM span.
                if isinstance(result.llm_output, dict):
                    result.llm_output["openrouter_provider"] = served
                for gen in result.generations:
                    meta = getattr(getattr(gen, "message", None), "response_metadata", None)
                    if isinstance(meta, dict):
                        meta["openrouter_provider"] = served
            except Exception:  # pragma: no cover - provenance must never break a run
                pass
            return result

        def _convert_chunk_to_generation_chunk(self, chunk, *args, **kwargs):
            thinking.record_chunk(chunk)
            gen = super()._convert_chunk_to_generation_chunk(chunk, *args, **kwargs)
            try:
                usage = chunk.get("usage") if isinstance(chunk, dict) else None
                if gen is not None and isinstance(usage, dict):
                    gen.message.response_metadata["token_usage"] = usage
            except Exception:  # pragma: no cover - provenance must never break a run
                pass
            return gen

    return _ChatOpenAIRecordingProvider


def _keep_reasoning(response, result) -> None:
    """Copy each choice's OpenRouter ``reasoning_details`` onto its AIMessage's
    ``additional_kwargs`` (see :func:`_chat_model_class`). Best-effort."""
    try:
        raw = response if isinstance(response, dict) else response.model_dump()
        for choice, gen in zip(raw.get("choices") or [], result.generations):
            details = (choice.get("message") or {}).get("reasoning_details")
            if details:
                gen.message.additional_kwargs["reasoning_details"] = details
    except Exception:  # pragma: no cover - a lost reasoning block must never break a run
        pass


def _return_reasoning(messages, payload) -> None:
    """Put each AIMessage's ``reasoning_details`` back on its assistant message in ``payload``.
    The payload's messages are the input's, one for one; when they are not, nothing is touched."""
    sent = payload.get("messages")
    if not isinstance(sent, list) or len(sent) != len(messages):
        return
    for message, out in zip(messages, sent):
        details = (getattr(message, "additional_kwargs", None) or {}).get("reasoning_details")
        if details and isinstance(out, dict) and out.get("role") == "assistant":
            out["reasoning_details"] = details


# A provider 5xx that arrives UNCLASSIFIED. OpenRouter returns HTTP 200 with an `{"error": ...}`
# body for some upstream failures, so the OpenAI client raises a bare ValueError instead of an
# InternalServerError — which means such an error is invisible to BOTH retry layers: the client's
# own `max_retries` (it never saw a 5xx status) and `_TRANSIENT_ERRORS` (ValueError is not in it).
# It therefore reached the terminal handler and ended the run. Observed twice in 839 archived
# runs, each time on a HEALTHY run: median 5 fits, 62 tool calls and 25 minutes of work already
# done, with a model committed. The resume-from-checkpoint machinery already exists and preserves
# all of that; this classifier is only what lets the error reach it.
#
# Matched on a 5xx CODE, never on wording: a `ValueError` carrying a 400 is deterministic (a
# content refusal, a malformed request) and must keep falling through to the terminal path, or
# every content refusal would be retried three times before failing anyway.
_PROVIDER_CODE_RE = re.compile(r"['\"]code['\"]\s*:\s*(\d{3})")


def _transient_provider_reason(exc: BaseException) -> str | None:
    """A short reason if ``exc`` is an unclassified provider 5xx worth RESUMING for, else None.

    Two stream failures count too, now that every call streams (agent.thinking): a stream that
    went silent (langchain's ``StreamChunkTimeoutError``, no chunk for its chunk timeout), and an
    upstream failure OpenRouter relays as an error event mid-stream, which the OpenAI client raises
    as the bare ``APIError`` -- no status, so none of the retryable classes. The bare class only:
    its subclasses (a 400 for an over-long conversation, say) would fail the same way again.
    """
    try:
        import openai
        from langchain_openai import StreamChunkTimeoutError
    except ImportError:  # pragma: no cover - the agent extra installs both
        pass
    else:
        if isinstance(exc, StreamChunkTimeoutError) or type(exc) is openai.APIError:
            return f"stream: {type(exc).__name__}: {str(exc)[:120]}"
    m = _PROVIDER_CODE_RE.search(str(exc))
    if not m or not 500 <= int(m.group(1)) <= 599:
        return None
    return f"provider {m.group(1)}: {type(exc).__name__}: {str(exc)[:120]}"


# NOTE: the lead used to run a ContextEditingMiddleware (ClearToolUsesEdit) that cleared old
# tool results once its context passed ~90k tokens — a guard against an early failure mode where
# re-sent tool RESULTS ballooned a run to ~1.1M tokens and hanging calls. It was removed: all
# target models have a 1M-token window and the recursion limit already caps a run at ~120k tokens
# (a full crauste sweep of 17 runs topped out at 118k and the trimmer fired on only 2 of them,
# once each). With prompt caching the cleared history was ~98% near-free cache_read, while the
# clear itself invalidated the cached prefix — so trimming was net token-negative. The per-request
# timeout still backstops any slow call. Re-add a trimmer only if a model with a smaller window
# is introduced.


def _server_is_alive(server_poll) -> "bool | None":
    """``True`` if the MCP server PROCESS is still running, ``False`` if it has exited, ``None``
    when the caller wired no way to ask.

    Deliberately a process check and NOT a probe call. The obvious implementation -- read something
    cheap and see whether it answers -- cannot work here, because it cannot tell a dead server from
    a busy one, and the only moment this is consulted is the moment the server is guaranteed busy:
    `_is_call_timeout` documents that the abandoned multistart is STILL RUNNING (mcp sends no
    cancellation). Worse, `make_changes` is a sync `def`, and mcp runs sync tool functions directly
    on the event loop (`fastmcp.utilities.func_metadata`: `if fn_is_async: await fn(...) else:
    return fn(...)`), holding `session.lock` for the whole change-and-fit section. So NOTHING can be
    answered for the remainder of a multi-thousand-second fit. A read-based probe would report
    "dead" on a perfectly healthy server every single time it was asked, and end the run instead of
    resuming it.

    `Popen.poll()` has none of that ambiguity: it answers in microseconds, never confuses busy with
    gone, and returns the killing signal itself. That is exactly the OOM case this exists for --
    bachmann task 52859568_0, where the kill left every later call to wait out its own 4200s budget,
    ten times over, for 11.7 hours (see BENCHMARK_NOTES).

    It does not detect a server that is alive but wedged. That is on purpose: such a server is
    indistinguishable from one mid-fit, which is the confusion that makes a read-based check wrong.
    ``None`` is NOT treated as dead, so a caller that wires nothing keeps the old behaviour.
    """
    if server_poll is None:
        return None
    try:
        # `Popen.poll()` semantics: None means STILL RUNNING, an int is the exit code. Taking the
        # returncode rather than a bool is what makes `server_poll=proc.poll` -- the obvious
        # wiring, given this docstring -- correct instead of exactly inverted. A `bool()` of the
        # same call reads None as False and would report every healthy server dead.
        return server_poll() is None
    except Exception:  # pragma: no cover - never let a liveness check end a run by accident
        return None


# A silent stop (the graph ends on a no-tool turn) is not an accepted way to finish: the driver
# bounces it back into the finish gate this many times before relenting so a genuinely stuck run
# still terminates (at the step/wall limit) rather than looping forever.
_MAX_SILENT_BOUNCES = 3


async def run_deepagents(
    session,
    *,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    task: str,
    recursion_limit: int,
    temperature: float = 1.0,
    reasoning_effort: str | None = None,
    n_starts: int = 1,
    max_wall_seconds: float | None = None,
    max_run_tokens: int | None = _MAX_RUN_TOKENS,
    progress_probe: Any | None = None,
    server_poll: Any | None = None,
    bio_sessions: Any | None = None,
    bio_literature: bool = False,
    curated_entities: "list[str] | None" = None,
    biocurator_model: "str | None" = None,
    biology_blinded: bool = False,
    input_protocols: "list[tuple[str, str]] | None" = None,
) -> dict[str, Any]:
    """Drive the orchestrator (+ analyst subagents) against ``session``'s tools.

    ``max_wall_seconds`` (optional) is a self-imposed total-runtime budget, checked between
    graph super-steps: once exceeded, the loop stops gracefully (like running out of resume
    attempts) so ``_judge`` can score whatever is currently committed, INSTEAD of racing an
    outer resource limit (e.g. a Slurm ``--time`` allocation) that would hard-kill the process
    with no chance to write a result at all. A sweep should size this comfortably BELOW its
    Slurm time limit (see ``slurm/make_sweep.py``'s ``--max-wall-seconds`` / ``--time``) so this
    fires first. ``None`` (the default) means no self-imposed limit — only the per-step stall
    watchdog and the outer scheduler's own limit apply, as before.

    ``progress_probe`` (optional) is an un-timed ``await probe(tool, args)`` handle used to poll
    the server's ``get_progress`` between super-steps: when the early-stop chain reports
    ``should_stop`` (the non-improving-streak cap, or a plateau once the best model is adequate),
    the loop stops at the next clean boundary — the deterministic replacement for wandering to the
    recursion limit. It is passed un-timed so these internal polls don't inflate the tool clock /
    ``tool_calls`` metric. ``None`` disables the check (the loop runs as before).

    It is ALSO the finish gate's reader (``_finish_tool(..., probe=progress_probe)``), and those
    reads run INSIDE a tool super-step rather than between them — so they can queue behind a live
    or abandoned multistart. Worth stating because assuming otherwise is what made a flat per-call
    bound on this handle look safe: it 408s the gate's read, ``_current_fit`` comes back empty, and
    five refusals relabel a fully fitted run ``no_model_built``. The budget belongs in
    ``clock._wrap_call_timeout``, which keeps the queue term.

    ``biocurator_model`` (optional) puts the biocurator on its OWN model — an alias or a full
    OpenRouter id — leaving the lead and the three analysts on ``model``. ``None`` keeps the whole
    panel on one model, which is the default and what every earlier run did.

    ``server_poll`` (optional) is a zero-argument callable with ``Popen.poll`` semantics — ``None``
    while the MCP server is running, an exit code once it is not. The loop consults it before
    spending a resume attempt, because resuming into a dead server re-sends the whole history and
    re-issues the pending fit for another full timeout (see ``_server_is_alive``). ``None`` here
    means "no way to tell", and is NOT treated as death: omitting it silently leaves the run with
    the old behaviour rather than ending it early.

    The model is served by OpenRouter via its OpenAI-compatible API (``base_url`` +
    ``api_key``). Returns ``{"stop_reason": ..., "n_tool_errors": ..., "disabled_tools": ...,
    **guard_stats}`` — the stop reason, the circuit breaker's error tallies, what was sent and
    what served it, the token totals, and the guard counters (``guards.new_guard_stats``). ALL of
    it lands in ``result.json`` (``runner.provenance._agent_meta``), so a measurement added to
    this dict is recorded without a second edit in the runner.
    The orchestrator's messages, its ``task`` delegations, the analysts' private tool
    calls, and token usage are recorded by Phoenix tracing (set up by the runner), so this
    function just drives the loop to completion.
    """
    from deepagents import create_deep_agent
    from langchain_mcp_adapters.tools import load_mcp_tools

    biology_blind = prompts.BIOLOGY_BLIND_ENABLED
    biocurator_enabled = prompts.BIOCURATOR_ENABLED
    if biology_blind:
        if not biology_blinded:
            raise ValueError(
                "The biology-blind arm requires an anonymized server problem; "
                "launch it through the scenario runner"
            )
        system_prompt = prompts.BIOLOGY_BLIND_SYSTEM_PROMPT
        # Built here, never taken from the caller: the passed task is discarded whole, so no
        # scenario prose can reach a blinded agent. The problem's input protocols arrive as data
        # (``input_protocols``, from the anonymous view) and ``biology_blind_task`` refuses a rule
        # that is not made of input aliases, ``time`` and functions.
        task = prompts.biology_blind_task(input_protocols)
    if prompts.INDRA_STATEMENTS_ENABLED:
        if biology_blind:
            raise ValueError("the indra_statements arm has no biology-blind variant")
        # The arm's own lead prompt, whatever the scenario passed: the default teaches Antimony.
        system_prompt = prompts.INDRA_SYSTEM_PROMPT

    # Adapt the MCP tools bound to this already-initialized session, then curate to the
    # union of the lead's + analysts' tools. This pool is only used to resolve each analyst's
    # tool objects; it is NOT what the lead gets (see lead_tools below).
    petab_tools = _trim_descriptions(_select_tools(await load_mcp_tools(session)))
    # Blinded bio-knowledge tools from one or more SECONDARY MCP servers (biocurator joins the
    # panel when enabled; endpoints hardcoded in runner.BIO_MCP_URLS = BioContextAI + OmniPath;
    # the `bio_literature` ablation adds EuropePMC/bioRxiv). They are
    # RUNTIME-DISCOVERED (their names are unknown ahead of time), so they deliberately BYPASS
    # _select_tools/ALLOWED_TOOLS (which would warn+drop them as "unexpected") and are handed
    # ONLY to the biocurator subagent below — never to the lead or the other analysts. NOTE: the
    # runner's _ToolClock shadows only the PETAB session's call_tool, so bio-server latency is
    # counted as "LLM+overhead" in the metrics split rather than tool time (a known caveat now
    # that the biocurator runs by default; wrap each bio session's call_tool into the clock too
    # if that split matters).
    bio_tools: list = (
        []
        if not biocurator_enabled
        else await discover_bio_tools(
            bio_sessions,
            literature=bio_literature,
            exclude_names={n for t in petab_tools if (n := getattr(t, "name", None))},
        )
    )
    all_tools = petab_tools + bio_tools
    # Wrap the tool objects with the repeated-error circuit breaker BEFORE they are handed to
    # the lead and the analyst subagents (both reference these same objects), so a tool that
    # keeps failing identically is disabled for the run instead of being re-proposed endlessly.
    breaker_state = _new_breaker_state()
    # Only the EXTERNAL bio tools (flaky hosted servers) get the immediate transport-error disable;
    # a transport blip on a core petab tool must NOT silently strip it for the rest of the run.
    _bio_names = {nm for nm in (getattr(t, "name", None) for t in bio_tools) if nm}
    _wrap_tools_with_circuit_breaker(all_tools, breaker_state, transport_disable_names=_bio_names)
    # Provenance: record the biocurator's LITERATURE accesses (query + returned DOIs/PMIDs) for
    # audit + analysis (surfaced as biocurator_literature). Only literature tools are wrapped, and
    # only when the `bio_literature` ablation made them resolve into bio_tools; empty otherwise.
    literature_log: list = []
    _wrap_literature_recorder(
        [t for t in bio_tools if getattr(t, "name", None) in LITERATURE_BIO_TOOLS], literature_log
    )
    tools_by_name = {n: t for t in all_tools if (n := getattr(t, "name", None))}
    # Declared BEFORE the subagents are built: the panel shares this one dict, so the reasoning
    # guard's counters aggregate over the lead AND every analyst rather than reporting the lead's
    # share as the run's total.
    # One definition, in guards.new_guard_stats — see there for why it is seeded up front.
    guard_stats: dict[str, Any] = new_guard_stats()
    # each analyst carries its own tool OBJECTS, and its own runaway guard (see below)
    subagents = _build_subagents(tools_by_name, guard_stats)
    # When enabled, the biocurator stays on the panel even if no bio tools resolve: it then
    # answers from the model's own knowledge. Both biocurator=off and biology_blind=on omit it.
    # The lead's roster and guidance use the same effective switch in prompts.py.
    # Hand the biocurator the scenario's HAND-CURATED biological entity NAMES DIRECTLY (see
    # Scenario.biocurator_entities) — beyond list_observables it holds no petab tool and cannot self-serve like the
    # analysts, so this is its reliable input channel rather than a lead `description` /
    # silently-dropped `prompt=` (issue #91). Empty for a scenario with no lookupable biology (the
    # abstract toys): the biocurator then simply gets no injected entities (nothing to look up).
    biocurator_entities = list(curated_entities or []) if biocurator_enabled else []
    # `list_observables` is the ONE petab tool the biocurator now holds (owner decision,
    # 2026-08-26): the observable table's `observableName` states WHAT each assay measures in
    # biology-only terms — exactly the vocabulary a curator searches literature with (phosphosites,
    # proteoforms, assay type) — and it is a read the blinding already curates. The blinding
    # boundary that matters is unchanged: no model access, no fit access, no measurements.
    biocurator_spec = None
    if biocurator_enabled:
        biocurator_spec = _build_biocurator_subagent(
            bio_tools + [t for t in petab_tools if getattr(t, "name", None) == "list_observables"],
            biocurator_entities,
            guard_stats,
        )
        subagents.append(biocurator_spec)
    biocurator_tools = sorted(n for t in bio_tools if (n := getattr(t, "name", None)))
    if biology_blind:
        print("[agent] biology-blind panel: no biocurator or external knowledge tools")
    elif not biocurator_enabled:
        print(
            "[agent] biocurator disabled; biological context and three modeling analysts retained"
        )
    else:
        print(
            f"[agent] biocurator on the panel with {len(biocurator_tools)} bio tool(s)"
            + (
                f": {biocurator_tools}"
                if biocurator_tools
                else " (none resolved — running on the model's own knowledge)"
            )
        )
        print(
            f"[agent] biocurator handed {len(biocurator_entities)} entity name(s)"
            + (f": {biocurator_entities}" if biocurator_entities else " (none curated)")
        )
    # The LEAD gets ONLY the orchestrator tools (+ the finish gate) — NOT the analysts'
    # data/diagnostic tools (query_data, get_fitting_log, get_measurements, ...). Those heavy
    # reads run INSIDE analyst subagents, whose verbose results live in their own transient
    # contexts and never enter the lead's every-turn re-sent history — which is where this
    # input-dominated workload's token cost accumulates (the lead re-sends its whole history on
    # each of its 60-115 turns). The lead diagnoses by DELEGATING to the analysts, not by
    # querying data itself. deepagents' subagents hold their own resolved tool objects, so a
    # restricted top-level tool list does not starve them.
    lead_tools = [tools_by_name[n] for n in ORCHESTRATOR_TOOLS if n in tools_by_name]
    # Shared with the driver loop: the finish tool sets finish_state["confirmed"] on a
    # confirmed finish (so a legitimately-finished run is NOT bounced back), and "gave_up" after
    # too many no-fit finish attempts (so the driver ends the run promptly, not at recursion).
    finish_state = {"confirmed": False, "gave_up": False}
    lead_tools = [*lead_tools, _finish_tool(session, finish_state, probe=progress_probe)]

    # The model's REAL context window, as a langchain profile. Without it, deepagents'
    # summarization middleware cannot see a window and falls back to a flat 170,000-token
    # trigger (compute_summarization_defaults) — measured on the 37-run sneyd/fujita A/B:
    # every run compacted (81 events), each costing ~102k NOVEL tokens (cache hit 0.46 vs
    # 0.85-0.89 normal) and ~33s, at a median trigger prompt of 169,266 tokens — 16% of the
    # 1M window this driver already assumes everywhere else (it deliberately runs no history
    # trimmer, and models.py pins 1M-context endpoints for exactly that reason). With the
    # profile set, the trigger becomes 0.85 x window ≈ 891k. Long large-model runs DO reach it
    # now (lucarelli 845-927k, raimundez 862k on the 2026-08-20/21 sweeps), and that is by
    # design: one compaction is cheaper than the explicit stop that used to fire at 0.8 and
    # ended runs mid-improvement. The context-stop below is the true backstop, at 0.90.
    max_input_tokens = _env_token_limit("PETAB_MCP_MAX_INPUT_TOKENS", 1_048_576, minimum=1)
    # Output ceiling. An explicit 0 leaves `max_tokens` OFF so the provider default applies (the
    # behaviour before 2026-08-28); unset uses the 131_072 default — see the parameter below for
    # why the provider default is not enough. minimum=0 is what makes the off-switch reachable:
    # fit_budget's _env_int maps 0 back to the default, which would silently turn the cap ON.
    max_output_tokens = _env_token_limit("PETAB_MCP_MAX_OUTPUT_TOKENS", 131_072, minimum=0)
    # Bound once so the value sent to the provider is the value recorded in result.json — a
    # fresh uuid inlined at the call site could not be reported afterwards.
    prompt_cache_key = f"petab-{uuid.uuid4().hex[:12]}"

    def _build_llm(model_id: str):
        """One panel model, configured identically whichever agent holds it.

        The biocurator may run a DIFFERENT (cheaper) model from the lead — see
        ``biocurator_model`` — and everything below is per-model rather than per-agent: the
        provider pin comes from ``_extra_body(model_id)``, and the output ceiling is the ceiling
        of whatever model is being asked.
        """
        return _chat_model_class()(
            # Keeps the Phoenix span named `ChatOpenAI` despite the subclass -- see _chat_model_class.
            name="ChatOpenAI",
            profile={"max_input_tokens": max_input_tokens},
            model=model_id,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            timeout=_LLM_TIMEOUT_S,  # fail a hung provider call fast (was unbounded)
            max_retries=_LLM_MAX_RETRIES,
            # Ask for token usage even on streamed responses (adds stream_options.include_usage):
            # OpenAI-compatible providers omit usage from a stream unless requested, which would
            # leave _token_accountant recording 0 tokens for the whole run. Harmless when a call
            # isn't streamed.
            stream_usage=True,
            # ASK FOR THE FULL OUTPUT BUDGET. Omitting max_tokens does not mean "unbounded" -- the
            # provider substitutes its own default, and Z.AI's for z-ai/glm-5.3-flash is 65536,
            # exactly HALF the 131072 the model advertises as max_completion_tokens. That default is
            # what a reasoning runaway hits: measured across isensee and lucarelli, the lead returned
            # completion_tokens=65536 of which reasoning_tokens=65504, content "", finish_reason
            # "length" -- truncated mid-thought before emitting a single tool call, four turns in a
            # row, ~25 minutes each. Downstream that is indistinguishable from a model choosing to
            # stay silent, so the dispatch guard bounced it and the run ended `no_model_built` with
            # zero make_changes attempted. Asking for the model's real ceiling doubles the room it
            # has to reach a conclusion; `_reasoning_runaway_guard` covers the case where even the
            # full budget is not enough.
            **({"max_tokens": max_output_tokens} if max_output_tokens else {}),
            # Prompt-cache hint: OpenRouter's fallback STICKY-ROUTING key when no `session_id` is
            # sent, so repeat requests reach the same backend and its prefix cache (prompt tokens
            # dominate this workload). Without a key it hashes the opening messages instead, which an
            # agent loop rewrites. Harmless where a provider ignores it, redundant on a model pinned
            # to one endpoint.
            #
            # PER RUN, not a global constant. The key exists to give routing AFFINITY to requests
            # sharing a prefix family; the previous fixed "petab-mcp-agent" told the provider that
            # every concurrent sweep task (up to ~17 at once, x5 agents each, with completely
            # unrelated prompts) was ONE family — concentrating them on one backend where they evict
            # each other's prefixes. One uuid per run keeps a run's own lead+subagent traffic
            # together (they at least share tool schemas and skills) while isolating runs from each
            # other. Deliberately NOT per-agent: the lead and its subagents share this one llm
            # object, so a per-agent key would need per-call middleware for marginal gain.
            model_kwargs={"prompt_cache_key": prompt_cache_key},
            # OpenRouter request fields, sent verbatim in the body: see _extra_body and models.py.
            extra_body=_extra_body(model_id, reasoning_effort),
            default_headers=models.request_headers(model_id),
        )

    llm = _build_llm(model)
    # The biocurator's model, when the caller named one. It is the panel's heaviest reader —
    # entity lookups return whole database entries, which then ride its context — and its job is
    # retrieval and summary rather than modelling, so it is the one seat where a cheaper model is
    # worth measuring. ``None`` (the default) leaves it on the lead's model, so nothing changes
    # unless asked. Recorded in result.json either way.
    bio_model_id = (
        (models.resolve(biocurator_model) if biocurator_model else model)
        if biocurator_enabled
        else None
    )
    if biocurator_spec is not None and bio_model_id != model:
        # Set on the spec built above rather than passed to it: the spec is assembled before any
        # model exists, and deepagents reads `model` off it when it builds the subagent graph.
        biocurator_spec["model"] = _build_llm(bio_model_id)
        print(f"[agent] biocurator runs {bio_model_id} (lead: {model})")
    # Transient provider errors that should RESUME the run (below) rather than end it, when they
    # escape the OpenAI client's own retries — e.g. a malformed non-JSON body the client won't
    # retry. Non-transient errors (bad request, recursion-limit) are deliberately excluded.
    from json import JSONDecodeError

    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    _TRANSIENT_ERRORS: tuple = (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
        JSONDecodeError,
    )
    # Backstop: a transport error (a closed/hung MCP server) that ESCAPES the per-tool breaker —
    # e.g. raised at the `task` subagent boundary or the bio ClientSession itself — should RESUME
    # the run from the checkpoint rather than abort it (the petab server is still alive). Cover both
    # transport layers the breaker classifies in _TRANSPORT_ERROR_NAMES: the anyio stream errors and
    # httpx's TransportError base (ConnectError/ReadError/WriteError/ReadTimeout/ConnectTimeout/
    # RemoteProtocolError all derive from it).
    try:
        from anyio import BrokenResourceError, ClosedResourceError, EndOfStream

        _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (
            ClosedResourceError,
            BrokenResourceError,
            EndOfStream,
        )
    except Exception:  # pragma: no cover - anyio always present with mcp, but never fail on import
        pass
    try:
        import httpx

        _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (httpx.TransportError,)
    except Exception:  # pragma: no cover - httpx always present with mcp, but never fail on import
        pass

    # deepagents renamed the prompt kwarg between releases (instructions -> system_prompt).
    # Pass whichever the installed version accepts, plus the analyst subagents.
    params = inspect.signature(create_deep_agent).parameters
    prompt_kw = "system_prompt" if "system_prompt" in params else "instructions"
    kwargs: dict[str, Any] = {"model": llm, "tools": lead_tools, prompt_kw: system_prompt}
    if "name" in params:
        # Callback metadata identifies lead calls independently of the shared panel model.
        kwargs["name"] = _LEAD_AGENT_NAME
    if "subagents" in params:
        kwargs["subagents"] = subagents
    else:  # pragma: no cover - guards against a deepagents API drift
        print(
            "[agent] WARNING: installed deepagents has no `subagents` parameter; "
            "running a single agent without the analyst panel."
        )
    # Lead middleware: drop deepagents' built-in FS/todo tools (the analysts can't be reliably
    # filtered in this deepagents version — see _builtin_tool_filter; `task` is untouched), hide a
    # circuit-broken tool so it stops being re-proposed, and guard tool dispatch (dedupe duplicate
    # calls in a batch; push back on idle no-tool turns and label a run that never fires a tool as
    # no_tools_executed — see _tool_dispatch_guard). No history trimming: all target models have a
    # 1M-token window and the recursion limit caps a run at ~120k tokens, so the old tool-result
    # trimmer only ever fired on a rare over-long run and, with prompt caching, cost more (a clear
    # invalidates the cached prefix) than the near-free cache_read history it removed.
    guard_installed = "middleware" in params
    if guard_installed:
        kwargs["middleware"] = [
            _builtin_tool_filter(),
            # Outermost of the tool-call guards, so every guard after it sees the call repaired.
            _leaked_argument_repair(guard_stats),
            _disabled_tool_filter(breaker_state),
            # _task_arg_guard is placed OUTSIDE _tool_dispatch_guard (earlier in the list = outer
            # tool-call wrapper) so a REJECTED `task` call short-circuits before the dispatch guard
            # counts it — a bounced call never dispatched a subagent, so it must not inflate
            # tools_executed (which gates the no_tools_executed relabel / idle-nudge logic).
            # _empty_report_guard is placed INNERMOST (last) so its ONE-shot re-dispatch of a
            # subagent that returned "" hits the real tool directly, without re-entering
            # _tool_dispatch_guard and double-counting tools_executed for a single `task` call.
            # _reasoning_runaway_guard is the only guard here that wraps the MODEL call rather
            # than tool calls or tool results, so it does not compete with the others for
            # ordering: _tool_dispatch_guard counts idle turns in `after_model`, which runs once
            # per model NODE — after the whole wrap_model_call chain has returned — so it
            # necessarily judges the POST-ladder result whatever the list order. Its position is
            # therefore not load-bearing, and the note that used to be here had it backwards:
            # first-in-list is OUTERMOST (factory.py "First handler wraps all others"), so the
            # guards after it are INNER and each sees every call the ladder makes, truncated
            # ones included — not only the final retry. Benign today (a truncated turn carries
            # no tool calls, so _dedupe_response finds nothing to do), but stated correctly so a
            # future reorder is not reasoned from a false invariant.
            _task_arg_guard(guard_stats),
            # Outside _tool_dispatch_guard for the same reason as _task_arg_guard: a bounced call
            # applied nothing and must not count as an executed tool.
            *([_statements_only_guard(guard_stats)] if INDRA_STATEMENTS_ENABLED else []),
            _reasoning_runaway_guard(guard_stats),
            _tool_dispatch_guard(guard_stats),
            _empty_report_guard(guard_stats),
        ]
    else:  # pragma: no cover - guards against a deepagents API drift
        print(
            "[agent] WARNING: installed deepagents has no `middleware` parameter; "
            "the lead's built-in FS/todo tools + full history will remain, and proposed "
            "tool calls are not guarded."
        )
    # A checkpointer lets us RESUME the graph from its last completed super-step after a stall or
    # exhausted call-retry (below), instead of discarding the run. In-process MemorySaver is
    # enough (one run per process). Degrade gracefully if the graph can't take one.
    checkpointer = None
    if "checkpointer" in params:
        try:
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()
            kwargs["checkpointer"] = checkpointer
        except Exception as exc:  # pragma: no cover - resume is a best-effort enhancement
            print(
                f"[agent] no checkpointer ({type(exc).__name__}); a stall will end the run "
                "instead of resuming."
            )
    # Drop the uncurated `general-purpose` subagent BEFORE the graph is built — it is added
    # during create_deep_agent, so this must precede it (issue #112). Then verify through
    # deepagents' own resolver: a run whose panel still carries it is contaminated (the lead can
    # delegate model authoring), so record the verdict in result.json rather than only logging
    # it, and warn loudly. We do NOT abort — a harness-detail drift must not cost a sweep run.
    _disable_general_purpose_subagent()
    gp_enabled = _general_purpose_enabled(llm)
    guard_stats["general_purpose_subagent"] = gp_enabled
    guard_stats["subagent_names"] = [s["name"] for s in subagents]
    if gp_enabled is not False:
        print(
            "[agent] WARNING: deepagents' `general-purpose` subagent is "
            + ("still enabled" if gp_enabled else "of unknown status")
            + " — the lead may be able to delegate model authoring to an uncurated subagent "
            "holding the full tool pool (issue #112). Treat this run's containment as "
            "unverified."
        )
    agent = create_deep_agent(**kwargs)

    stop_reason = "completed"
    # Drive the agent to completion, iterating manually so a per-step STALL WATCHDOG can fire:
    # astream emits once per graph super-step, so if `__anext__` doesn't return within the stall
    # window the current step is wedged. On a stall — or a call-retry that ran out (below) — we
    # RESUME from the last checkpoint (astream with input=None continues pending work) rather than
    # abandoning the run, up to _RUN_RESUME_ATTEMPTS times. A non-transient error (e.g.
    # GraphRecursionError = step budget exhausted) is NOT retried; the judge assesses the model as
    # it stands. We consume the stream only to drive the loop; Phoenix spans are the record.
    # The longest single super-step is a lead commit whose make_changes runs an n_starts multistart
    # fit (astream emits nothing until it returns), so the stall window must exceed one whole fit
    # or a slow-but-healthy fit trips the watchdog. That used to force ONE global wait wide enough
    # for the largest possible fit, which is also how long a wedged NON-fit step (a lead LLM call,
    # an analyst sub-agent) went undetected. The window is sized PER STEP now: `step_wait` in the
    # loop below rises to the pending fit's own budget and otherwise sits at the floor below.
    #
    # Note what that does and does not buy. A wedged NON-fit step is still not detected promptly --
    # the floor is max(_STEP_STALL_TIMEOUT_S, fit_time_budget(run default)), so on the large
    # examples it is the run default's fit budget, hours not minutes. What per-step sizing removes
    # is the EXTRA widening: such a step used to be held open for the largest fit the run might
    # ever ask for (2x the default, and still short of the scenario floors). Detection of a wedged
    # non-fit step therefore roughly halves; it does not become prompt. Making it prompt needs the
    # floor itself sized per step kind, which is a separate change -- a hung provider CALL is
    # already caught in 120s by the client timeout, so what is left here is a genuinely wedged
    # step, and those are rare enough that a late detection costs one resume.
    from ..fit_budget import FIT_WATCHDOG_MARGIN, fit_time_budget, fit_time_budget_for

    # Sized to the n_starts the step about to run ACTUALLY asked for -- see _pending_fit_starts.
    #
    # `n_starts` here is only the run's DEFAULT, and a watchdog sized to it kills legitimate
    # fits: bachmann task 9 (job 52239407) ran a 300-start fit and died at exactly
    # fit_time_budget(200)=7560s as "stalled: no step completed", losing ~22 node-hours. That was
    # first patched with `fit_time_budget(2 * n_starts)`, which is a guess in both directions: it
    # still sits below scenario.build_true_model's floors (bruno 400, fujita 300), and on a run
    # whose agent never raises n_starts it doubles how long a genuinely wedged NON-fit step goes
    # undetected, for nothing.
    #
    # There is no server-side cap to lean on either (one was tried and reverted; any cap near the
    # default refuses the scenario floors above), so no multiplier can be right for a count the
    # agent chooses at runtime. Read the count instead: `step_limit` in the loop below is
    # fit_time_budget_for(pending fit, run default), which floors at the run default's own
    # budget and rises only for a step that actually asked for more.
    # Self-imposed total-runtime budget (see the docstring): checked between super-steps, never
    # mid-step, so we always stop at a clean boundary rather than interrupting a live commit/fit.
    deadline = (time.monotonic() + max_wall_seconds) if max_wall_seconds is not None else None
    # The finish gate holds a confirm on flagged residual structure while most of this budget is
    # left (finish_gate._FLAGGED_CONFIRM_BUDGET_FRACTION), so it reads the same clock.
    finish_state.update(deadline=deadline, max_wall_seconds=max_wall_seconds)
    # RESERVE ONE STEP. The budget is checked at step boundaries, but a super-step that commits a
    # model runs a full multistart, so "am I past the deadline?" asked AFTER the step is asked too
    # late: the overshoot is a whole fit. On the 2026-08-17 isensee sweep (budget 56400s inside a
    # 57600s Slurm limit, i.e. 1200s of margin against a ~5200s fit) 15 of 16 tasks were
    # TIMEOUT-killed rather than stopping cleanly, losing result.json, the judge verdict and the
    # token accounting for every one of them. So stop when there is not enough time left to
    # FINISH another step, using the same fit estimate the stall watchdog sizes itself to.
    # Capped at half the budget so a small budget still buys some work rather than stopping at 0.
    # The DEFAULT n_starts, not the 2x cap. Sizing to the cap looked safer -- an agent may raise
    # n_starts on its last step, and such a step is then CANCELLED at the hard deadline, losing
    # the fit -- but it charges every run for a raise most never make: at the recorded bachmann
    # config (n_starts=200, cap 400) that is fit_time_budget(400)-fit_time_budget(200) = 6720s
    # reserved unconditionally, 12% of an isensee budget. A certain 12% tax against an occasional
    # lost last fit is the wrong trade. The residual risk is real and accepted: an agent that
    # doubles n_starts on the final step can still have it cancelled.
    step_headroom = (
        min(fit_time_budget(n_starts), max_wall_seconds * 0.5)
        if max_wall_seconds is not None
        else 0.0
    )
    # THE deadline for every "may I start (or continue into) another step?" decision. The hard
    # `deadline` is the budget; this is the budget minus one step, i.e. the last moment at which
    # beginning a super-step can still finish inside it. Every such decision must use this one --
    # guarding only the step-boundary check left three paths (the stall/budget split, the
    # silent-finish bounce, and the retry-resume gate) able to start a fresh super-step inside the
    # reserved window, which is exactly the overshoot that TIMEOUT-killed 15/16 isensee tasks.
    soft_deadline = (deadline - step_headroom) if deadline is not None else None
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if checkpointer is not None:
        config["configurable"] = {"thread_id": "petab-mcp-run"}
    # Sum token usage across the lead + every subagent (config callbacks propagate into the
    # subgraphs), for result.json. One instance across all resume iterations = whole-run totals.
    token_accountant = _token_accountant()
    config["callbacks"] = [token_accountant]

    async def _early_stop_reason() -> str | None:
        """The server's early-stop reason if the run should stop now, else ``None`` (fail-open:
        any probe error just means "keep going")."""
        if progress_probe is None:
            return None
        try:
            data = _structured(await progress_probe("get_progress", {}))
            if isinstance(data, dict) and data.get("should_stop"):
                return data.get("stop_reason") or "early-stop condition met"
        except Exception:  # pragma: no cover - never let the probe break the run
            pass
        return None

    stream_input: Any = {"messages": [{"role": "user", "content": task}]}
    resume_left = _RUN_RESUME_ATTEMPTS if checkpointer is not None else 0
    # DOUBLE the stall window on each resume. A resume re-sends the whole history, so a stall
    # caused by the provider being SLOW on a large prompt reproduces at the same window every
    # time -- measured on a glm blasi run, all three resumes stalled at an identical 1063s and
    # the run spent 4252s of its 4968s wall waiting. Retrying at a fixed window can therefore
    # only ever confirm the first verdict. Widening it instead makes the retries diagnostic: a
    # merely slow call gets a real chance to land, while a genuinely wedged one still terminates.
    #
    # WORST CASE, stated honestly: with `--max-wall-seconds` unset (the CLI default) there is no
    # deadline to cap against, so a wedged run waits 1800+3600+7200+14400 = 7.5h before ending,
    # against 4x600s = 40min before this change. With a deadline set the waits are capped, but by
    # the HARD deadline -- so a wedged run can still consume the `step_headroom` reserve meant for
    # scoring. Escalation is therefore deliberately gated to stalls only, and reset on progress.
    stall_backoff = 1.0
    # Message count of the PREVIOUS emission, so a resume's checkpoint replay (which re-emits
    # state we have already seen, at the same count) is distinguishable from a real step. Lives
    # outside the resume loop for the same reason `pending_fit` does.
    #
    # Deliberately the last count, NOT a high-water mark: deepagents installs langchain's
    # summarization middleware by default, and compaction emits RemoveMessage(REMOVE_ALL_MESSAGES)
    # — the history collapses to a summary plus a short tail. Against a maximum, every step after
    # the first compaction looks like "no new messages", so stall_backoff could never return to
    # 1.0 and one early transient would leave the detector widened for the rest of the run: the
    # exact failure the reset exists to prevent, reintroduced from the other side. Any CHANGE in
    # count is progress; only an unchanged count is a replay.
    last_message_count: Any = _NO_EMISSION_YET
    # A silent stop (graph ends on a no-tool turn) without a confirmed finish is bounced back
    # into the finish gate this many times; separate from resume_left, which is for transients.
    silent_bounces = 0
    # Last clean-completion probe of whether a model was authored (see the silent-stop bounce);
    # captured so the post-loop relabel reuses it instead of re-probing. Only read when
    # stop_reason == "completed", which guarantees the intercept refreshed it that iteration.
    model_committed = True
    # n_starts of the fit pending for the next super-step, or None if none is (see
    # _pending_fit_starts). Lives OUTSIDE the resume loop so a resume re-enters with the last
    # count observed rather than dropping back to the default mid-fit: astream re-emits from the
    # checkpoint, but the first wait after `input=None` is armed before anything is emitted.
    pending_fit: int | None = None
    if progress_probe is None:
        # The no-model-built guard (and the deterministic early-stop) both need the probe; without
        # it a silent no-model stop scores as an empty "completed" run instead of resuming to
        # build one — so a caller that omits progress_probe reintroduces that gap silently.
        print(
            "[agent] no progress_probe wired — the no-model-built guard and deterministic "
            "early-stop are disabled for this run"
        )
    while True:
        retriable: str | None = None
        agen = None
        wall_budget_exceeded = False
        token_budget_exceeded = False
        early_stopped: str | None = None
        # Bound BEFORE the try: the `except asyncio.TimeoutError` handler reports it, and on
        # 3.11+ `asyncio.TimeoutError is TimeoutError`, so a TimeoutError raised anywhere in the
        # block -- including from astream() itself, before the inner loop assigns this -- reaches
        # that handler. Without a binding here the report raises UnboundLocalError and kills the
        # run instead of resuming.
        # Same expression as the in-loop floor, so a TimeoutError raised by astream() itself
        # reports the window that was actually in force rather than the un-backed-off one.
        step_wait = float(_STEP_STALL_TIMEOUT_S) * stall_backoff
        # On the input=None resume path, langgraph emits the CHECKPOINT state before scheduling
        # a single task (pregel/_loop.py: `if is_resuming: ... self._emit("values", ...)`), so
        # the first __anext__ below returns in milliseconds without the provider having done
        # anything. That emission must not count as progress: resetting stall_backoff on it
        # re-runs every retry at the base window, and a stall caused by a SLOW provider on a
        # large prompt then reproduces at the same window forever — the exact failure the
        # escalation exists to break.
        #
        # Detected from the STATE rather than by assuming the emission happens: a replay carries
        # no messages the previous attempt had not already produced. That keeps this correct
        # when langgraph does NOT emit (`is_resuming` is False because no checkpoint had been
        # written yet) — a bare "skip the first emission" flag would then swallow a real step —
        # and it degrades to "count it as progress" if a future langgraph stops replaying.
        # The count alone cannot cover ONE case — a resume after a stall that emitted nothing at
        # all, where there is no previous count to compare against — so a resume additionally
        # arms this for its first emission (see the check below).
        replay_pending = stream_input is None
        try:
            # Open the stream INSIDE the try so a synchronous raise from astream (e.g. on the
            # input=None resume path) is handled/degraded here, not left to abort the whole run.
            agen = agent.astream(stream_input, config=config, stream_mode="values").__aiter__()
            while True:
                try:
                    # How long this ONE step may take. fit_time_budget_for already floors at
                    # the RUN-DEFAULT fit budget and caps at FIT_BUDGET_CEILING_S, and it is the
                    # same function the per-call transport cut uses, so the two cannot disagree
                    # about how long a given fit is allowed to take. The only thing left to add
                    # is the absolute _STEP_STALL_TIMEOUT_S floor for a run whose fit budget is tiny;
                    # the run-default term must not be applied a second time -- the function
                    # already applies it.
                    #
                    # FIT_WATCHDOG_MARGIN keeps this STRICTLY ABOVE the transport cut for the
                    # same fit. Tied, the watchdog always wins (it is armed a graph-dispatch
                    # earlier) and an overrunning fit is cancelled mid-step as "stalled" instead
                    # of returning the clean 408 that resumes without telling the model to
                    # re-fit. Ordering: httpx ceiling > watchdog > per-call cut.
                    # `stall_backoff` widens ONLY the floor, never the fit-derived term. The
                    # fit term is already clamped at FIT_BUDGET_CEILING_S so a derived timeout
                    # cannot switch the watchdog off, and it must stay strictly below the session
                    # httpx ceiling (ordering above). Multiplying the whole max() breaks both: on
                    # isensee at the 12000s cap the fit term is ~20286s, so x8 would be 162288s
                    # (45h) -- past the 86400s clamp and past the httpx ceiling, leaving a wedged
                    # step undetected for half a multi-day run.
                    step_limit = max(
                        _STEP_STALL_TIMEOUT_S * stall_backoff,
                        fit_time_budget_for(pending_fit, n_starts) * FIT_WATCHDOG_MARGIN,
                    )
                    # Wait no longer than the time left in the wall-time budget. Without this the
                    # loop can only notice the deadline BETWEEN super-steps, so a step starting
                    # just before it runs the loop up to `step_limit` PAST it -- against a default
                    # margin of 20 min, that leaves ~2 min for the judge to restore and score a
                    # checkpoint. Observed: a fujita/v4-pro run was hard-killed by Slurm at
                    # exactly 06:00:02 with NO result.json, losing the whole run. Capping the wait
                    # hands control back at the deadline, so the entire margin is available for
                    # scoring.
                    # The HARD deadline, deliberately — NOT `soft_deadline`. This bounds a step
                    # that is already running, and the whole point of the reserve is to let that
                    # step finish: capping the wait at the soft deadline cancels a commit step up
                    # to `step_headroom` (as much as half the budget) early and throws its fit
                    # away. Only the "may I START another step?" decisions use the soft deadline.
                    step_wait = (
                        step_limit
                        if deadline is None
                        else max(1.0, min(step_limit, deadline - time.monotonic()))
                    )
                    state = await asyncio.wait_for(agen.__anext__(), timeout=step_wait)
                except StopAsyncIteration:
                    break
                # A step COMPLETED, so the provider is responsive: drop back to the base window.
                # Without this one early transient leaves the detector permanently widened for
                # the rest of the run -- a wedge 15 hours later would wait 2x or 8x by accident
                # rather than by escalation. The one emission that says nothing about the
                # provider is the checkpoint replay a resume starts with (see above) -- the
                # widened window must survive it so the retry actually runs wider.
                n_msgs = _state_message_count(state)
                # Progress = this emission differs from the previous one. Two exceptions:
                #   * an UNREADABLE state (None) degrades to "progress", the behaviour from
                #     before the replay test existed — one odd emission must not leave the
                #     window widened for good;
                #   * the FIRST emission of a resume, when nothing was ever emitted before
                #     (_NO_EMISSION_YET), is the checkpoint replay of work we never saw. It
                #     looks "different" from nothing, but the provider has done nothing since
                #     the stall, so it must not clear the escalation the stall just earned.
                replay = replay_pending and last_message_count is _NO_EMISSION_YET
                replay_pending = False
                if not replay and (n_msgs is None or n_msgs != last_message_count):
                    stall_backoff = 1.0
                last_message_count = n_msgs
                # Arm the NEXT wait from what this emission leaves pending. Deliberately after
                # the await and before every break below: a step that ends the loop does not need
                # a window, and a resume re-enters with whatever was last observed.
                pending_fit = _pending_fit_starts(state, n_starts, guard_stats)
                if soft_deadline is not None and time.monotonic() >= soft_deadline:
                    wall_budget_exceeded = True
                    break
                # Token ceiling, checked at the same clean boundary as the wall-time budget. The
                # accountant is ONE instance across every resume, so this bounds the whole run
                # rather than resetting each time astream is re-entered -- which is precisely how
                # the observed runaway escaped every other limit.
                if (
                    max_run_tokens is not None
                    and token_accountant.prompt + token_accountant.completion >= max_run_tokens
                ):
                    token_budget_exceeded = True
                    break
                # Context backstop, checked at the same clean boundary. This used to STOP at
                # 0.8 x window to pre-empt the summarization middleware's 0.85 compaction, on the
                # grounds that no call had ever approached it. Runs then reached it: on the
                # 2026-08-20/21 sweeps the 0.8 stop ended 3 of 9 lucarelli runs (prompts 845-927k)
                # and raimundez task 4 at dBIC +39.6 — still improving, 12h of budget left, ~30
                # BIC from passing. A compaction is a real quality event (~102k novel tokens,
                # history discarded mid-search) but it is strictly cheaper than ending the run:
                # so compaction at 0.85 is now the first line of defence and this stop sits above
                # it, at 0.90.
                #
                # `max_prompt` is an all-time monotonic maximum (token_usage never resets it),
                # so this cannot mean "post-compaction the history still fills the window" -- one
                # large prompt trips it permanently. The check also runs only at super-step
                # boundaries, so whatever the line, a single step can cross the real 1M window
                # before the next check and end as an unanalyzable provider 400.
                #
                # 0.90 is therefore a judgement, not a derivation: it keeps ~105k tokens of slack
                # (against ~102k for one compaction), which is enough for the compaction path but
                # NOT provably enough for a step that adds a large tool payload on top. It is
                # chosen to sit above the 0.85 compaction trigger while giving back most of the
                # margin the old 0.8 line cost -- that line ended 3 of 9 lucarelli runs and
                # raimundez t4 at dBIC +39.6, still improving. Size it from the largest observed
                # single-step prompt growth once that is measured.
                if token_accountant.max_prompt >= 0.90 * max_input_tokens:
                    early_stopped = (
                        "context_window_approached: largest prompt "
                        f"{token_accountant.max_prompt:,} tokens >= 90% of the "
                        f"{max_input_tokens:,}-token window"
                    )
                    break
                # The gate gave up (too many no-fit finish attempts) — end promptly at this
                # boundary rather than letting the model spam finish to the recursion limit.
                if finish_state["gave_up"]:
                    break
                # Deterministic early-stop: poll the server between super-steps and stop at this
                # clean boundary once the run has stopped improving (streak cap / plateau).
                early_stopped = await _early_stop_reason()
                if early_stopped is not None:
                    break
            stop_reason = (
                "wall_time_budget_exceeded"
                if wall_budget_exceeded
                # A BARE label, like wall_time_budget_exceeded beside it: the counts used to be
                # interpolated here, which made every such run's stop_reason unique AND collided
                # with `experiments._INFRA_ABORT_MARKERS` -- its "402" marker (out-of-credits) is a
                # substring of a comma-formatted token count like "20,402,551", so a runaway with
                # 0 fits was classified as infrastructure noise and dropped from the aggregation
                # entirely. The counts belong in the print below and in `token_usage`.
                else "token_budget_exceeded"
                if token_budget_exceeded
                else "no_model_built"
                if finish_state["gave_up"]
                else f"early_stop: {early_stopped}"
                if early_stopped
                else "completed"
            )
        except asyncio.TimeoutError:
            if soft_deadline is not None and time.monotonic() >= soft_deadline:
                # The wait was cut short by the wall-time cap above, not by a wedged step. This is
                # the budget expiring mid-super-step, so it must STOP (and let the judge score what
                # is committed) rather than resume — a resume here would spend the margin that the
                # scoring needs, which is how a run gets hard-killed with no result at all.
                wall_budget_exceeded = True
                stop_reason = "wall_time_budget_exceeded"
            else:
                # Report the window the step ACTUALLY had, not the run-default floor: with the
                # window sized per step, those differ whenever a fit was pending, and a stall
                # message naming the wrong number sends the next investigation to the wrong
                # timeout. Name the pending fit too -- "wedged fit" and "wedged LLM turn" want
                # different diagnoses, and this line is usually all a sweep post-mortem has.
                waiting_on = (
                    f"{pending_fit}-start fit pending"
                    if pending_fit is not None
                    else "no fit pending"
                )
                retriable = (
                    f"stalled: no step completed in {step_wait:.0f}s "
                    f"({waiting_on}; hung provider call or wedged fit)"
                )
        except _TRANSIENT_ERRORS as exc:  # call-level retries exhausted -> resume the run
            retriable = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            # A provider CONTENT-POLICY refusal is terminal and deterministic: the same transcript
            # re-sent trips it again, so it must not join _TRANSIENT_ERRORS and be resumed. Give it
            # its own short stop_reason instead of the raw exception text, so a sweep can count it
            # as one outcome rather than as a unique failure string per run. Seen live:
            # gpt-5.6-luna-pro on a PEtab immunology model -> 400 "This content was flagged for
            # possible biological risk." Nothing here can fix that; the point is to record it.
            # Order matters: a content refusal is checked FIRST and is terminal, so it can never
            # be routed to the retry path by a later branch.
            if (filtered := _content_filter_stop_reason(exc)) is not None:
                stop_reason = filtered
            elif _is_call_timeout(exc):
                # A per-call fit timeout that escaped the breaker (e.g. raised at the `task`
                # subagent boundary). Same reasoning as in the breaker: the server is still
                # running the abandoned multistart, so RESUME from the checkpoint rather than
                # ending the run -- and rather than letting the model re-issue a fit on top of
                # one still holding the worker pool. It cannot join _TRANSIENT_ERRORS because
                # that is a tuple of CLASSES and this is one status code within McpError.
                retriable = f"fit call timed out: {exc}"
            elif (transient := _transient_provider_reason(exc)) is not None:
                retriable = transient  # resume from the checkpoint, keeping every version
            else:
                stop_reason = f"{type(exc).__name__}: {exc}"
        finally:
            # Best-effort close; bound it so a truly wedged stream can't hang teardown too.
            if agen is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(agen.aclose(), timeout=10)
        if wall_budget_exceeded:
            print(
                f"[agent] wall-time budget ({max_wall_seconds:.0f}s) reached with less than "
                f"one step ({step_headroom:.0f}s) of headroom left — stopping at this step "
                "boundary; the judge scores whatever is currently committed."
            )
            break
        if token_budget_exceeded:
            # Deliberately NOT retriable: a resume would hand the loop a fresh step budget and the
            # whole history to re-send, which is the exact mechanism that turned one run into
            # 107.9M tokens. Ending here keeps everything committed so far scoreable.
            print(
                f"[agent] token budget exceeded: "
                f"{token_accountant.prompt + token_accountant.completion:,} tokens over "
                f"{max_run_tokens:,} in {token_accountant.n_calls} LLM calls — stopping at this "
                "step boundary; the judge scores whatever is currently committed."
            )
            break
        if early_stopped is not None:
            print(
                f"[agent] early-stop: {early_stopped} — stopping at this step boundary; "
                "the judge restores + scores the best-scoring checkpoint."
            )
            break
        if finish_state["gave_up"]:
            print(
                "[agent] finish gate gave up (no fittable model after repeated attempts) — "
                "ending the run as no-model-built."
            )
            break
        if retriable is None:
            # Clean graph termination. A "completed" that did NOT pass the finish gate is a SILENT
            # STOP — the lead ended on a no-tool turn, sidestepping the finish gate. A silent stop is
            # not an accepted exit: bounce it back into the gate so the run only ends by finishing
            # properly (a fitted model, diagnosis shown, confirmed) — with no coaching on WHAT to
            # fix, just enforcement of the protocol. Bounded (_MAX_SILENT_BOUNCES) so a genuinely
            # stuck run still terminates at the step/wall limit rather than looping. A CONFIRMED
            # finish is respected (not bounced) and needs neither the probe nor the relabel, so it
            # short-circuits first — saving a describe_problem call on the good path. The no-model
            # case gets the build nudge; a built-but-unfinished model gets the finish nudge. No
            # backoff: a silent stop is not a rate-limited transient, so there is nothing to wait
            # out. model_committed is captured for the post-loop relabel (no second probe).
            if stop_reason == "completed" and not finish_state["confirmed"]:
                model_committed = await _model_committed(progress_probe)
                past_deadline = soft_deadline is not None and time.monotonic() >= soft_deadline
                if (
                    silent_bounces < _MAX_SILENT_BOUNCES
                    and checkpointer is not None
                    and not past_deadline
                ):
                    silent_bounces += 1
                    no_model = (
                        prompts.INDRA_NO_MODEL_NUDGE if INDRA_STATEMENTS_ENABLED else _NO_MODEL_NUDGE
                    )
                    nudge = no_model if not model_committed else _SILENT_FINISH_NUDGE
                    what = (
                        "NO model authored" if not model_committed else "no finish through the gate"
                    )
                    print(
                        f"[agent] silent stop ({what}) — bouncing back to finish properly "
                        f"({silent_bounces}/{_MAX_SILENT_BOUNCES})"
                    )
                    stream_input = {"messages": [{"role": "user", "content": nudge}]}
                    continue
            break
        if soft_deadline is not None and time.monotonic() >= soft_deadline:
            # Out of time even before a resume would start — don't burn a backoff on a retry
            # the deadline will just cut off anyway.
            stop_reason = f"wall_time_budget_exceeded (during retry: {retriable})"
            print(
                f"[agent] wall-time budget ({max_wall_seconds:.0f}s) exceeded while handling "
                f"'{retriable}' — stopping instead of resuming."
            )
            break
        # The SAME guard for tokens. Without it a run already past the ceiling could still resume:
        # the inner check only runs at a completed step boundary, so a run that blew the budget and
        # then hit a stall / transient error arrives here with `token_budget_exceeded` still False.
        # Each resume re-enters astream with a fresh step budget AND re-sends the whole history, so
        # with `_RUN_RESUME_ATTEMPTS` retries the effective ceiling was several times the nominal
        # one -- the exact compounding `_MAX_RUN_TOKENS` exists to bound.
        if (
            max_run_tokens is not None
            and token_accountant.prompt + token_accountant.completion >= max_run_tokens
        ):
            stop_reason = "token_budget_exceeded"
            print(
                f"[agent] token budget ({max_run_tokens:,}) already exceeded at "
                f"{token_accountant.prompt + token_accountant.completion:,} while handling "
                f"'{retriable}' — stopping instead of resuming."
            )
            break
        # Is there still a server to resume against? Ask BEFORE spending a resume attempt: each
        # one re-sends the whole history and re-issues the pending fit, so resuming into a dead
        # server buys nothing and costs another full timeout per attempt. Checked here rather
        # than in the handlers above so it covers every retriable path (stall, transient,
        # per-call fit timeout) with one probe.
        if _server_is_alive(server_poll) is False:
            stop_reason = f"server_unreachable: {retriable}"
            print(
                f"[agent] {retriable} — and the MCP server process has exited, so it is gone "
                "rather than slow. Not resuming: each resume re-sends the whole history and "
                "re-issues the pending fit, and every call to a dead server waits out its own "
                "timeout."
            )
            break
        if resume_left <= 0:
            stop_reason = retriable  # out of resume attempts — end with whatever was built
            break
        resume_left -= 1
        # Escalate ONLY for a stall. The retry tail is shared with rate limits, connection drops
        # and fit-call 408s, and none of those says anything about how long a healthy step takes
        # -- widening the hang detector because the provider returned 429 three times would be
        # unrelated cause and effect.
        if retriable.startswith("stalled:"):
            stall_backoff *= 2.0
        backoff = min(5.0 * (_RUN_RESUME_ATTEMPTS - resume_left), 30.0)
        print(
            f"[agent] {retriable} — resuming from last checkpoint in {backoff:.0f}s "
            f"({resume_left} resume(s) left, stall window x{stall_backoff:.0f})"
        )
        await asyncio.sleep(backoff)
        stream_input = None  # resume pending work from the checkpoint

    # A "completed" stop_reason that did NOT pass the finish gate (bounces exhausted, or no
    # checkpointer to resume from) is relabelled honestly rather than read as a proper finish, so
    # evaluation can see a bypassed gate. Three cases, most specific first:
    #   * no_tools_executed — the lead only ever emitted no-tool turns, so the dispatch guard
    #     pushed back _MAX_IDLE_MODEL_STEPS times and GAVE UP (guard_stats["gave_up_no_tools"]):
    #     the degenerate "proposes tool calls but none fire" run. Distinct from a run that DID
    #     dispatch tools but built no model (e.g. delegation-only), which stays no_model_built.
    #     Gated on the guard actually being installed (without the `middleware` hook the counters
    #     are not trustworthy). This must come first: gave_up_no_tools implies no model, so it
    #     would otherwise be swallowed by the no_model_built branch below.
    #   * no_model_built — nothing was authored;
    #   * silent_stop — a model exists but the run never finished through the finish gate.
    # A confirmed finish keeps "completed". Reuses the intercept's probe (model_committed): a
    # "completed" stop_reason here means the final iteration ran the intercept, which refreshed it
    # — no second describe_problem call.
    if stop_reason == "completed" and not finish_state["confirmed"]:
        if guard_installed and guard_stats.get("gave_up_no_tools"):
            stop_reason = "no_tools_executed"
        else:
            stop_reason = "no_model_built" if not model_committed else "silent_stop"
    if guard_stats["dropped_dup_tool_calls"]:
        print(
            f"[agent] de-duplicated {guard_stats['dropped_dup_tool_calls']} identical tool "
            "call(s) within a batch over the run"
        )
    if breaker_state["n_tool_errors"]:
        disabled = sorted(breaker_state["disabled"])
        print(
            f"[agent] tool errors this run: {breaker_state['n_tool_errors']}"
            + (f"; disabled: {disabled}" if disabled else "")
        )
    token_usage = token_accountant.totals()
    # A run that dispatched tools necessarily made LLM calls; capturing zero of them means the
    # usage was not reported (a streamed response without usage, or a callback-propagation gap),
    # so the recorded token_usage understates the run. Make that loud rather than silently
    # writing zeros that read as a genuinely tiny run in the cost analysis.
    if token_usage["n_llm_calls"] == 0 and guard_stats.get("tools_executed", 0) > 0:
        print(
            "[agent] WARNING: token accounting captured 0 LLM calls despite tool activity — "
            "token_usage in result.json is understated (usage not reported for this run)."
        )
    elif token_usage["n_uncosted_calls"]:
        print(
            f"[agent] WARNING: {token_usage['n_uncosted_calls']} of "
            f"{token_usage['n_llm_calls']} LLM calls carried no billed cost — "
            "token_usage.cost_usd "
            + ("is absent" if token_usage["cost_usd"] is None else "understates the run")
            + "."
        )

    return {
        "stop_reason": stop_reason,
        "n_tool_errors": breaker_state["n_tool_errors"],
        "disabled_tools": sorted(breaker_state["disabled"]),
        # Call-level outcomes for the single editor, counted at the CLIENT wrapper. Distinct from
        # `edit_rejects`, which is server-side and therefore blind to a call that never reached
        # the server -- the case that mattered: a run making 134 failing make_changes calls
        # recorded `edit_rejects.attempted: 0`, so nothing in result.json showed the failure.
        "make_changes_calls": call_outcomes(breaker_state, "make_changes"),
        "openrouter_extra_body": _extra_body(model, reasoning_effort),
        "request_headers": models.request_headers(model),
        # What the request ASKED for is `openrouter_extra_body`; this is what it GOT. The two are
        # not the same claim -- `provider.only` bounds routing to a set, so a run can be served by
        # several of its members and still honour the pin.
        "openrouter_endpoints": token_accountant.endpoints(),
        "biocurator_tools": biocurator_tools,
        "biocurator_model": bio_model_id,
        "biocurator_entities": biocurator_entities,
        "biocurator_literature": literature_log,
        "token_usage": token_usage,
        # The output ceiling actually sent (0 = not sent, provider default applied). Reported
        # alongside openrouter_extra_body and for the same reason: what a run RAN WITH must be
        # readable from its own artifact, not inferred from when it was submitted.
        "max_output_tokens": max_output_tokens,
        "prompt_cache_key": prompt_cache_key,
        **guard_stats,
    }
