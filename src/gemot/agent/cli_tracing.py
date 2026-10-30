"""Observe native CLI events and export their evidence to the existing Phoenix trace.

This recorder never calls a model, builds a conversation, or dispatches a tool. Codex and
Claude keep their own long-lived sessions and cache behavior. JSONL is the evidence: tool
spans use native call IDs; Claude response spans use native message IDs and reported usage.
Codex's turn-completed usage is an aggregate, so it stays on the session under custom
attributes rather than being attributed to an invented API call. Native event logs remain
the complete record when a future CLI adds an event this observer does not understand.

Wall-clock timestamps describe when the driver observed events, not server inference
latency. Claude partial messages improve response boundaries; without them a response is
an arrival observation with zero duration. Full hidden system prompts and internal model
requests are not reconstructed. All OpenTelemetry imports are deferred until tracing is
enabled, and observation failures never interrupt the native agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import time
from typing import Any


_TOKEN_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "novel_tokens",
)


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _tokens(raw: Any, provider: str) -> dict[str, int | None] | None:
    """Preserve independently reported counts; absent cache counters are unknown.

    Claude's input count excludes cache reads and writes, so its inclusive prompt total
    needs all three components. Codex already includes caching in its input count.
    """
    if not isinstance(raw, dict):
        return None
    inp, out = _count(raw.get("input_tokens")), _count(raw.get("output_tokens"))
    if provider == "claude":
        read = _count(raw.get("cache_read_input_tokens"))
        write = _count(raw.get("cache_creation_input_tokens"))
        novel = inp + write if inp is not None and write is not None else None
        inp = novel + read if novel is not None and read is not None else None
    else:
        read = _count(raw.get("cached_input_tokens"))
        write = _count(raw.get("cache_write_input_tokens"))
        if inp is not None and read is not None and read > inp:
            read = None
        if inp is not None and write is not None and write > inp:
            write = None
        novel = inp - read if inp is not None and read is not None else None
    counts = {
        "prompt_tokens": inp,
        "completion_tokens": out,
        "total_tokens": inp + out if inp is not None and out is not None else None,
        "cache_read_tokens": read,
        "cache_write_tokens": write,
        "novel_tokens": novel,
    }
    return counts if any(value is not None for value in counts.values()) else None


@dataclass
class _ObservedSpan:
    span: Any
    start: int
    end: int | None = None
    data: dict[str, Any] = field(default_factory=dict)


class NativeCLIRecorder:
    """A synchronous ``on_event`` callback plus explicitly delimited session lifetime.

    Pass the existing ``TracingHandle`` and optionally the run's parent span. Without a
    tracing handle, usage accounting and final-response capture still work. ``usage()``
    returns ``None`` for quantities the native stream has not supplied, including Codex's
    internal model-call count. ``finish()`` is idempotent and closes interrupted spans.
    """

    def __init__(
        self, provider: str, model: str, tracing: Any = None, parent_span: Any = None
    ) -> None:
        if provider not in ("codex", "claude"):
            raise ValueError("Native CLI tracing provider must be codex or claude.")
        self.provider = provider
        self.model = model
        self.tracing = tracing
        self.parent_span = parent_span
        self.session_id: str | None = None
        self.final_text = ""
        self.error: str | None = None
        self.diagnostics: list[str] = []
        self._tracer = None
        self._trace = None
        self._root = None
        self._started = False
        self._finished = False
        self._items: dict[str, _ObservedSpan] = {}
        self._messages: dict[str, _ObservedSpan] = {}
        self._tools: dict[str, _ObservedSpan] = {}
        self._active_messages: dict[str, str] = {}
        self._aggregate: dict[str, int | None] | None = None
        self._usage_source: str | None = None
        self._cost: float | None = None
        self._missing_message_ids = False

    def _note(self, text: str) -> None:
        if text not in self.diagnostics and len(self.diagnostics) < 20:
            self.diagnostics.append(text)

    def start(self, prompt: str) -> None:
        """Record the single initial task; do not modify or resend it."""
        if self._started:
            return
        self._started = True
        if self.tracing is None:
            return
        try:
            from opentelemetry import trace

            self._trace = trace
            provider = getattr(self.tracing, "tracer_provider", None)
            self._tracer = (
                provider.get_tracer("gemot.native_cli")
                if provider
                else trace.get_tracer("gemot.native_cli")
            )
            context = trace.set_span_in_context(self.parent_span) if self.parent_span else None
            self._root = self._tracer.start_span(
                f"{self.provider}.session",
                context=context,
                attributes={
                    "openinference.span.kind": "AGENT",
                    "input.value": prompt,
                    "input.mime_type": "text/plain",
                    "gemot.cli.provider": self.provider,
                    "gemot.cli.requested_model": self.model,
                    "gemot.cli.timing_source": "observed_events",
                    "gemot.cli.usage_scope": "session_aggregate",
                    "gemot.cli.input_scope": "initial_task_only",
                },
            )
        except Exception as exc:
            self._note(f"Could not start native CLI trace: {type(exc).__name__}")
            self._tracer = None

    def _new(
        self, name: str, kind: str, now: int, attrs: dict[str, Any], parent: Any = None
    ) -> _ObservedSpan:
        span = None
        if self._tracer and self._root:
            try:
                span = self._tracer.start_span(
                    name,
                    context=self._trace.set_span_in_context(parent or self._root),
                    start_time=now,
                    attributes={
                        "openinference.span.kind": kind,
                        "gemot.cli.provider": self.provider,
                        "gemot.cli.timing_source": "observed_events",
                        **({"session.id": self.session_id} if self.session_id else {}),
                        **attrs,
                    },
                )
            except Exception as exc:
                self._note(f"Could not create native event span: {type(exc).__name__}")
        return _ObservedSpan(span=span, start=now)

    def _attrs(self, state: _ObservedSpan | None, attrs: dict[str, Any]) -> None:
        if state is not None and state.span is not None:
            for key, value in attrs.items():
                if value is not None:
                    state.span.set_attribute(key, value)

    def event(self, event: dict[str, Any]) -> None:
        """Observe one parsed native event. Unsupported events remain in the raw JSONL."""
        if self._finished:
            return
        try:
            if not isinstance(event, dict):
                self._note("Ignored a non-object native event.")
                return
            now = time.time_ns()
            session_id = event.get("session_id") or event.get("thread_id")
            if isinstance(session_id, str):
                self.session_id = session_id
                if self._root:
                    self._root.set_attribute("session.id", session_id)
            if self.provider == "codex":
                self._codex(event, now)
            else:
                self._claude(event, now)
        except Exception as exc:
            self._note(f"Could not observe {event.get('type')!r}: {type(exc).__name__}")

    __call__ = event

    def _codex(self, event: dict[str, Any], now: int) -> None:
        kind = event.get("type")
        if kind == "turn.completed":
            # Native retries can emit errors before the same turn completes successfully.
            self.error = None
            usage = _tokens(event.get("usage"), "codex")
            if usage is not None:
                self._usage_source = "codex_turn_completed"
            if usage is None or usage["total_tokens"] is None:
                self._note("Codex completed a turn without complete token usage.")
            counts = usage or dict.fromkeys(_TOKEN_KEYS)
            if self._aggregate is None:
                self._aggregate = counts
            else:
                # Once one turn lacks a count, later turns cannot make its total known.
                self._aggregate = {
                    key: self._aggregate[key] + value
                    if self._aggregate[key] is not None and value is not None
                    else None
                    for key, value in counts.items()
                }
        elif kind in ("error", "turn.failed"):
            detail = event.get("error")
            # ``turn.failed`` nests the provider's message: {"error": {"message": ...}}.
            if isinstance(detail, dict):
                detail = detail.get("message")
            self.error = str(detail or event.get("message") or kind)
            self._note(f"Codex reported {kind}: {self.error}")
        elif kind in ("item.started", "item.updated", "item.completed"):
            item = event.get("item")
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                return
            item_id, item_type = item["id"], item.get("type")
            model_output = item_type == "agent_message"
            tool = item_type in ("command_execution", "mcp_tool_call", "web_search", "file_change")
            if not model_output and not tool:
                return
            state = self._items.get(item_id)
            if state is None:
                name = item.get("tool") or item_type
                state = self._new(
                    f"codex.{name}",
                    "LLM" if model_output else "TOOL",
                    now,
                    {
                        "gemot.cli.item_id": item_id,
                        "gemot.cli.item_type": item_type,
                        "gemot.cli.observation_scope": "output_item"
                        if model_output
                        else "tool_lifecycle",
                        "gemot.cli.start_observed": kind == "item.started",
                    },
                )
                self._items[item_id] = state
            state.data.update(item)
            if model_output:
                text = item.get("text")
                if isinstance(text, str):
                    self._attrs(state, {"output.value": text, "output.mime_type": "text/plain"})
                    if kind == "item.completed":
                        self.final_text = text
            else:
                details = state.data
                arguments = details.get(
                    "arguments",
                    details.get("command", details.get("query", details.get("changes"))),
                )
                self._attrs(
                    state,
                    {
                        "tool.name": details.get("tool") or item_type,
                        "input.value": json.dumps(arguments, ensure_ascii=False),
                        "input.mime_type": "application/json",
                        "gemot.cli.mcp_server": details.get("server"),
                    },
                )
                if "result" in item or "aggregated_output" in item or "error" in item:
                    self._attrs(
                        state,
                        {
                            "output.value": json.dumps(
                                item.get(
                                    "result", item.get("aggregated_output", item.get("error"))
                                ),
                                ensure_ascii=False,
                            ),
                            "output.mime_type": "application/json",
                        },
                    )
            if kind == "item.completed":
                state.end = now
                if item.get("status") == "failed" or item.get("error"):
                    state.data["observation_error"] = str(item.get("error") or "Native item failed")

    def _message(
        self, message: dict[str, Any], parent_id: str, now: int, started: bool
    ) -> _ObservedSpan:
        message_id = message.get("id")
        if not isinstance(message_id, str):
            self._missing_message_ids = True
            message_id = f"observation-{len(self._messages)}"
        state = self._messages.get(message_id)
        if state is None:
            parent = self._tools.get(parent_id)
            state = self._new(
                "claude.response",
                "LLM",
                now,
                {
                    "gemot.cli.message_id": message_id,
                    "llm.model_name": message.get("model") or self.model,
                    "gemot.cli.start_observed": started,
                    "gemot.cli.input_scope": "not_exposed",
                    "gemot.cli.parent_tool_use_id": parent_id,
                },
                parent=parent.span if parent else None,
            )
            self._messages[message_id] = state
        for key, value in message.items():
            if key == "usage" and isinstance(value, dict):
                state.data.setdefault("usage", {}).update(value)
            else:
                state.data[key] = value
        self._active_messages[parent_id] = message_id
        return state

    def _claude(self, event: dict[str, Any], now: int) -> None:
        kind = event.get("type")
        parent_id = event.get("parent_tool_use_id") or ""
        if kind == "stream_event":
            inner = event.get("event", {})
            if inner.get("type") == "message_start" and isinstance(inner.get("message"), dict):
                self._message(inner["message"], parent_id, now, True)
            else:
                state = self._messages.get(self._active_messages.get(parent_id, ""))
                if (
                    state
                    and inner.get("type") == "message_delta"
                    and isinstance(inner.get("usage"), dict)
                ):
                    state.data.setdefault("usage", {}).update(inner["usage"])
                elif state and inner.get("type") == "message_stop":
                    state.end = now
        elif kind == "assistant" and isinstance(event.get("message"), dict):
            message = event["message"]
            state = self._message(message, parent_id, now, False)
            # With no message_start, this is an arrival observation, not API latency.
            if state.end is None:
                state.end = now
            content = message.get("content", [])
            self._attrs(
                state,
                {
                    "output.value": json.dumps(content, ensure_ascii=False),
                    "output.mime_type": "application/json",
                },
            )
            if event.get("error"):
                state.data["observation_error"] = str(event["error"])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if (
                        not parent_id
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                    ):
                        self.final_text = block["text"]
                    if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                        call_id = block["id"]
                        if call_id not in self._tools:
                            self._tools[call_id] = self._new(
                                str(block.get("name") or "claude.tool"),
                                "TOOL",
                                now,
                                {
                                    "tool.name": str(block.get("name") or "unknown"),
                                    "gemot.cli.tool_use_id": call_id,
                                    "input.value": json.dumps(
                                        block.get("input"), ensure_ascii=False
                                    ),
                                    "input.mime_type": "application/json",
                                },
                                parent=state.span,
                            )
        elif kind == "user" and isinstance(event.get("message"), dict):
            content = event["message"].get("content", [])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    state = self._tools.get(block.get("tool_use_id"))
                    if state:
                        state.end = now
                        self._attrs(
                            state,
                            {
                                "output.value": json.dumps(
                                    block.get("content"), ensure_ascii=False
                                ),
                                "output.mime_type": "application/json",
                            },
                        )
                        if block.get("is_error"):
                            state.data["observation_error"] = "Native tool returned an error"
        elif kind == "result":
            self._aggregate = _tokens(event.get("usage"), "claude")
            self._usage_source = "claude_result" if self._aggregate else None
            if isinstance(event.get("result"), str):
                self.final_text = event["result"]
            cost = event.get("total_cost_usd")
            self._cost = (
                float(cost)
                if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0
                else None
            )
            subtype = event.get("subtype")
            if event.get("is_error") or subtype not in (None, "success"):
                self.error = str(event.get("errors") or event.get("result") or event.get("subtype"))
                self._note(f"Claude reported unsuccessful result: {self.error}")
            elif subtype == "success":
                self.error = None
        elif kind == "error":
            self.error = str(event.get("error") or event.get("message") or kind)
            self._note(f"Claude reported error: {self.error}")

    def _response_usage(self) -> dict[str, int | None] | None:
        """Sum each Claude response's own counts; ``None`` where no response reported one."""
        reported = [
            counts
            for state in self._messages.values()
            if (counts := _tokens(state.data.get("usage"), "claude"))
        ]
        if not reported:
            return None
        return {
            key: sum(counts[key] for counts in reported if counts[key] is not None)
            if any(counts[key] is not None for counts in reported)
            else None
            for key in _TOKEN_KEYS
        }

    def usage(self) -> dict[str, Any]:
        """Return reported counts, never estimate tokens or internal API calls.

        Without the CLI's session total (a failed turn, a crash, a stopped process), the counts
        are its responses' own, a lower bound marked ``usage_source: "partial"``, or none at all,
        ``"unavailable"``: Codex reports nothing before its turn completes. A run that spent
        nothing is then not mistaken for one that was not counted (gemot#75).
        """
        counts, source = self._aggregate, self._usage_source
        if source is None:
            counts = self._response_usage()
            source = "partial" if counts else "unavailable"
        return {
            **(counts or dict.fromkeys(_TOKEN_KEYS)),
            "n_llm_calls": len(self._messages)
            if self.provider == "claude" and self._messages and not self._missing_message_ids
            else None,
            "usage_source": source,
            "reported_cost_usd": self._cost,
            "cost_source": "claude_cli_estimate" if self._cost is not None else None,
        }

    def finish(self, error: str | None = None) -> None:
        """Close observed spans, retaining partial evidence when the CLI was interrupted."""
        if self._finished:
            return
        self._finished = True
        now = time.time_ns()
        self.error = error or self.error
        try:
            from opentelemetry.trace import Status, StatusCode
        except ImportError:
            Status = StatusCode = None
        message_states = {id(state) for state in self._messages.values()}
        for state in [*self._items.values(), *self._messages.values(), *self._tools.values()]:
            if state.span is None:
                continue
            try:
                if id(state) in message_states:
                    usage = _tokens(state.data.get("usage"), "claude")
                    if usage:
                        self._attrs(
                            state,
                            {
                                "llm.token_count.prompt": usage["prompt_tokens"],
                                "llm.token_count.completion": usage["completion_tokens"],
                                "llm.token_count.total": usage["total_tokens"],
                                "llm.token_count.prompt_details.cache_read": usage[
                                    "cache_read_tokens"
                                ],
                                "llm.token_count.prompt_details.cache_write": usage[
                                    "cache_write_tokens"
                                ],
                            },
                        )
                incomplete = state.end is None
                if incomplete:
                    state.span.set_attribute("gemot.cli.incomplete", True)
                failure = state.data.get("observation_error") or (
                    self.error if incomplete else None
                )
                if failure and Status:
                    state.span.set_status(Status(StatusCode.ERROR, str(failure)))
                state.span.end(end_time=state.end or now)
            except Exception as exc:
                self._note(f"Could not finish native event span: {type(exc).__name__}")
        if self._root:
            try:
                for key, value in self.usage().items():
                    if value is not None:
                        self._root.set_attribute(f"gemot.cli.{key}", value)
                self._root.set_attribute("output.value", self.final_text)
                self._root.set_attribute("output.mime_type", "text/plain")
                if self.diagnostics:
                    self._root.set_attribute("gemot.cli.observer_diagnostics", self.diagnostics)
                if self.error and Status:
                    self._root.set_status(Status(StatusCode.ERROR, self.error))
                self._root.end(end_time=now)
            except Exception as exc:
                self._note(f"Could not finish native session span: {type(exc).__name__}")
