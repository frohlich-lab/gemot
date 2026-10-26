"""Native CLI observations preserve provider accounting without manufacturing API calls."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from gemot.agent.cli_tracing import NativeCLIRecorder


class _Span:
    def __init__(self, name, **kwargs):
        self.name = name
        self.attributes = kwargs.get("attributes", {})
        self.parent = kwargs.get("context")
        self.start = kwargs.get("start_time")
        self.end_time = None
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.status = status

    def end(self, end_time=None):
        self.end_time = end_time


def _recorder(monkeypatch, provider):
    spans = []

    def start_span(name, **kwargs):
        span = _Span(name, **kwargs)
        spans.append(span)
        return span

    tracer = SimpleNamespace(start_span=start_span)
    trace = SimpleNamespace(
        set_span_in_context=lambda span: span,
        get_tracer=lambda _: tracer,
        Status=lambda code, description: SimpleNamespace(code=code, description=description),
        StatusCode=SimpleNamespace(ERROR="ERROR"),
    )
    monkeypatch.setitem(sys.modules, "opentelemetry", SimpleNamespace(trace=trace))
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace)
    handle = SimpleNamespace(tracer_provider=SimpleNamespace(get_tracer=lambda _: tracer))
    parent = _Span("parent")
    recorder = NativeCLIRecorder(provider, "requested-model", handle, parent)
    recorder.start("one initial task")
    return recorder, spans, parent


def test_codex_items_are_not_fabricated_api_calls(monkeypatch):
    recorder, spans, parent = _recorder(monkeypatch, "codex")
    recorder.event({"type": "thread.started", "thread_id": "native-thread"})
    recorder.event(
        {
            "type": "item.started",
            "item": {
                "id": "tool1",
                "type": "mcp_tool_call",
                "server": "gemot",
                "tool": "get_model",
                "arguments": {"format": "sbml"},
            },
        }
    )
    recorder.event(
        {
            "type": "item.completed",
            "item": {
                "id": "tool1",
                "type": "mcp_tool_call",
                "tool": "get_model",
                "status": "completed",
                "result": {"model": "result"},
            },
        }
    )
    recorder.event(
        {
            "type": "item.completed",
            "item": {
                "id": "answer",
                "type": "agent_message",
                "text": "finished",
            },
        }
    )
    recorder.event(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 120,
                "cached_input_tokens": 100,
                "output_tokens": 30,
            },
        }
    )
    recorder.finish()
    assert spans[0].parent is parent
    assert spans[0].attributes["session.id"] == "native-thread"
    assert recorder.usage()["n_llm_calls"] is None
    assert recorder.usage()["prompt_tokens"] == 120
    assert recorder.usage()["cache_write_tokens"] is None
    assert recorder.final_text == "finished"
    assert spans[1].attributes["tool.name"] == "get_model"
    assert spans[1].end_time >= spans[1].start
    # A visible assistant item could be one of several pieces from one model request.
    assert spans[2].attributes["gemot.cli.start_observed"] is False
    assert spans[2].end_time == spans[2].start
    assert not any(key.startswith("llm.token_count") for key in spans[2].attributes)
    assert spans[0].attributes["gemot.cli.prompt_tokens"] == 120


def test_claude_deduplicates_partial_messages_and_parents_native_tools(monkeypatch):
    recorder, spans, _ = _recorder(monkeypatch, "claude")
    recorder.event({"type": "system", "subtype": "init", "session_id": "native-session"})
    recorder.event(
        {
            "type": "stream_event",
            "event": {
                "type": "message_start",
                "message": {
                    "id": "msg1",
                    "model": "actual-model",
                    "usage": {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 20,
                        "cache_creation_input_tokens": 5,
                        "output_tokens": 1,
                    },
                },
            },
        }
    )
    recorder.event(
        {
            "type": "stream_event",
            "event": {
                "type": "message_delta",
                "usage": {"output_tokens": 7},
            },
        }
    )
    recorder.event({"type": "stream_event", "event": {"type": "message_stop"}})
    recorder.event(
        {
            "type": "assistant",
            "message": {
                "id": "msg1",
                "model": "actual-model",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call1",
                        "name": "mcp__gemot__get_model",
                        "input": {},
                    },
                ],
            },
        }
    )
    recorder.event(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call1",
                        "content": "native tool result",
                    },
                ]
            },
        }
    )
    recorder.event(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "finished",
            "total_cost_usd": 0.04,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 7,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 5,
            },
        }
    )
    recorder.finish()
    assert len(spans) == 3
    assert spans[2].parent is spans[1]
    assert spans[1].attributes["llm.model_name"] == "actual-model"
    assert spans[1].attributes["llm.token_count.prompt"] == 35
    assert spans[1].attributes["llm.token_count.completion"] == 7
    assert spans[1].attributes["llm.token_count.prompt_details.cache_read"] == 20
    assert recorder.usage()["n_llm_calls"] == 1
    assert recorder.usage()["prompt_tokens"] == 35
    assert recorder.usage()["cost_source"] == "claude_cli_estimate"
    assert "llm.token_count.prompt" not in spans[0].attributes
    assert "input.value" not in spans[1].attributes  # Full native system prompt is not known.


def test_missing_usage_stays_unknown_and_finish_is_idempotent(monkeypatch):
    recorder, spans, _ = _recorder(monkeypatch, "codex")
    recorder.event(
        {
            "type": "item.started",
            "item": {
                "id": "running",
                "type": "command_execution",
                "command": "long-job",
            },
        }
    )
    recorder.event({"type": "turn.failed", "error": {"message": "interrupted"}})
    recorder.finish()
    first_end = spans[1].end_time
    recorder.finish()
    assert spans[1].end_time == first_end
    assert spans[1].attributes["gemot.cli.incomplete"] is True
    assert recorder.usage()["prompt_tokens"] is None
    assert recorder.usage()["n_llm_calls"] is None


def test_usage_collection_needs_no_tracing_and_unknown_events_do_not_break_agent():
    recorder = NativeCLIRecorder("claude", "sonnet")
    recorder.start("one task")
    recorder.event({"type": "future_event", "new_payload": {"whatever": True}})
    recorder.event(
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "answer"}]}}
    )
    recorder.event({"type": "result", "usage": {"input_tokens": -1, "output_tokens": 5}})
    recorder.finish()
    assert recorder.final_text == "answer"
    assert recorder.usage()["prompt_tokens"] is None
    assert recorder.usage()["n_llm_calls"] is None


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_successful_terminal_event_clears_transient_error(monkeypatch, provider):
    recorder, spans, _ = _recorder(monkeypatch, provider)
    recorder.event({"type": "error", "message": "Connection lost; retrying"})
    assert recorder.error == "Connection lost; retrying"
    recorder.event(
        {"type": "turn.completed"}
        if provider == "codex"
        else {"type": "result", "subtype": "success", "is_error": False}
    )
    recorder.finish()
    assert recorder.error is None
    assert spans[0].status is None
    assert any("Connection lost" in note for note in recorder.diagnostics)


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_error_after_success_and_process_failure_remain_failures(monkeypatch, provider):
    recorder, spans, _ = _recorder(monkeypatch, provider)
    recorder.event(
        {"type": "turn.completed"}
        if provider == "codex"
        else {"type": "result", "subtype": "success"}
    )
    recorder.event(
        {"type": "turn.failed", "error": "Quota exhausted"}
        if provider == "codex"
        else {"type": "result", "subtype": "error_during_execution", "errors": ["Quota exhausted"]}
    )
    assert "Quota exhausted" in recorder.error
    recorder.finish(error="Native CLI exited with code 1")
    assert recorder.error == "Native CLI exited with code 1"
    assert spans[0].status.code == "ERROR"
    assert spans[0].status.description == recorder.error


def test_claude_success_supersedes_failed_result_without_reusing_its_cost():
    recorder = NativeCLIRecorder("claude", "sonnet")
    recorder.event(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "errors": ["temporary connection failure"],
            "total_cost_usd": 0.01,
        }
    )
    assert recorder.error is not None
    recorder.event({"type": "result", "subtype": "success", "result": "recovered"})
    recorder.finish()
    assert recorder.error is None
    assert recorder.final_text == "recovered"
    assert recorder.usage()["reported_cost_usd"] is None
    assert recorder.usage()["n_llm_calls"] is None  # No response events were observed.


@pytest.mark.parametrize("cache_write", [None, -1, True])
def test_claude_unknown_cache_writes_do_not_become_zero(monkeypatch, cache_write):
    recorder, spans, _ = _recorder(monkeypatch, "claude")
    raw = {"input_tokens": 10, "output_tokens": 7, "cache_read_input_tokens": 20}
    if cache_write is not None:
        raw["cache_creation_input_tokens"] = cache_write
    recorder.event({"type": "assistant", "message": {"id": "m1", "usage": raw}})
    recorder.event({"type": "result", "subtype": "success", "usage": raw})
    recorder.finish()
    usage = recorder.usage()
    assert usage["completion_tokens"] == 7
    assert usage["cache_read_tokens"] == 20
    assert usage["cache_write_tokens"] is None
    assert usage["prompt_tokens"] is None
    assert usage["total_tokens"] is None
    assert usage["novel_tokens"] is None
    assert spans[1].attributes["llm.token_count.completion"] == 7
    assert "llm.token_count.prompt" not in spans[1].attributes
    assert "llm.token_count.prompt_details.cache_write" not in spans[1].attributes
    assert "gemot.cli.cache_write_tokens" not in spans[0].attributes


def test_explicit_zero_cache_counters_are_known():
    recorder = NativeCLIRecorder("claude", "sonnet")
    recorder.event(
        {
            "type": "result",
            "subtype": "success",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 7,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
    )
    assert recorder.usage()["prompt_tokens"] == 10
    assert recorder.usage()["total_tokens"] == 17
    assert recorder.usage()["cache_read_tokens"] == 0
    assert recorder.usage()["cache_write_tokens"] == 0


@pytest.mark.parametrize("missing_first", [True, False])
def test_codex_missing_turn_usage_cannot_be_hidden_by_another_turn(missing_first):
    recorder = NativeCLIRecorder("codex", "gpt-codex")
    events = [
        {"type": "turn.completed"},
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 100, "output_tokens": 5, "cached_input_tokens": 80},
        },
    ]
    for event in events if missing_first else reversed(events):
        recorder.event(event)
    assert recorder.usage()["prompt_tokens"] is None
    assert recorder.usage()["completion_tokens"] is None
    assert recorder.usage()["cache_read_tokens"] is None
    assert recorder.usage()["cache_write_tokens"] is None


def test_codex_partial_counts_aggregate_only_known_quantities():
    recorder = NativeCLIRecorder("codex", "gpt-codex")
    recorder.event(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 100, "output_tokens": 5, "cached_input_tokens": 80},
        }
    )
    recorder.event({"type": "turn.completed", "usage": {"input_tokens": 20}})
    assert recorder.usage()["prompt_tokens"] == 120
    assert recorder.usage()["completion_tokens"] is None
    assert recorder.usage()["total_tokens"] is None
    assert recorder.usage()["cache_read_tokens"] is None
    assert recorder.usage()["cache_write_tokens"] is None
