"""Tests for the deepagents driver's tool-dispatch guard (examples/_harness/agent.py).

Covers issue #63: a class of runs (first seen on the blasi zero-shot build) that proposed
tool calls every step but executed ZERO tools, terminating early with nothing built. The guard
(1) de-duplicates identical tool calls within a batch, (2) pushes back on idle no-tool turns and
labels a run that never fires a tool as ``no_tools_executed``.

The pure de-dup helpers are tested with no third-party dependency. The middleware itself is
exercised end-to-end against langchain's real ``create_agent`` graph driven by a scripted fake
chat model (no network, no MCP server, no deepagents) — skipped where langchain is absent, as it
is not a dependency of the petab-mcp package (only of the examples harness).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_AGENT_PY = Path(__file__).resolve().parents[1] / "examples" / "_harness" / "agent.py"


def _load_agent_module():
    """Load agent.py in isolation (by file path) so the heavy _harness package __init__
    (which pulls in petab_mcp / phoenix) is not imported. agent.py only touches stdlib at
    import time — every langchain/deepagents import is lazy, inside the driver functions."""
    spec = importlib.util.spec_from_file_location("_harness_agent_under_test", _AGENT_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = _load_agent_module()


# --------------------------------------------------------------------- pure de-dup helpers
def test_tool_call_key_ignores_id_and_arg_order():
    a = {"name": "set_model", "args": {"antimony": "x", "n_starts": 5}, "id": "call_1"}
    b = {"name": "set_model", "args": {"n_starts": 5, "antimony": "x"}, "id": "call_2"}
    assert agent._tool_call_key(a) == agent._tool_call_key(b)


def test_tool_call_key_distinguishes_name_and_args():
    base = {"name": "get_fit_summary", "args": {}}
    assert agent._tool_call_key(base) != agent._tool_call_key({"name": "get_errors", "args": {}})
    assert agent._tool_call_key({"name": "f", "args": {"x": 1}}) != \
        agent._tool_call_key({"name": "f", "args": {"x": 2}})


def test_tool_call_key_handles_non_serialisable_args():
    # Must not raise on args json can't encode (falls back to repr).
    key = agent._tool_call_key({"name": "f", "args": {1, 2, 3}})
    assert key.startswith("f\x00")


def test_dedupe_drops_exact_duplicates_keeping_first():
    calls = [
        {"name": "a", "args": {"x": 1}, "id": "1"},
        {"name": "a", "args": {"x": 1}, "id": "2"},   # dup of #1
        {"name": "a", "args": {"x": 2}, "id": "3"},   # different args
        {"name": "b", "args": {"x": 1}, "id": "4"},   # different name
    ]
    kept, dropped = agent._dedupe_tool_calls(calls)
    assert dropped == 1
    assert [c["id"] for c in kept] == ["1", "3", "4"]


def test_dedupe_empty_and_none():
    assert agent._dedupe_tool_calls([]) == ([], 0)
    assert agent._dedupe_tool_calls(None) == ([], 0)


def test_dedupe_collapses_a_full_batch_of_identical_calls():
    calls = [{"name": "describe_problem", "args": {}, "id": f"c{i}"} for i in range(5)]
    kept, dropped = agent._dedupe_tool_calls(calls)
    assert dropped == 4
    assert [c["id"] for c in kept] == ["c0"]


# --------------------------------------------------------------------- raw (additional_kwargs)
def test_dedupe_raw_keeps_first_by_name_and_arguments():
    raw = [
        {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "2", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "3", "type": "function", "function": {"name": "f", "arguments": '{"x":1}'}},
    ]
    kept = agent._dedupe_raw_tool_calls(raw)
    assert [c["id"] for c in kept] == ["1", "3"]


def test_dedupe_raw_collapses_a_repeated_id_duplicate():
    # A provider that repeats the SAME call with the SAME id must not leave a duplicate behind
    # (the bug an id-membership filter would miss).
    raw = [
        {"id": "dup", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "dup", "type": "function", "function": {"name": "f", "arguments": "{}"}},
    ]
    kept = agent._dedupe_raw_tool_calls(raw)
    assert len(kept) == 1
    assert kept[0]["id"] == "dup"


def test_dedupe_raw_empty_and_none():
    assert agent._dedupe_raw_tool_calls([]) == []
    assert agent._dedupe_raw_tool_calls(None) == []


# --------------------------------------------------------------------- middleware end-to-end
# langchain is an examples-harness dependency, NOT a dependency of the petab-mcp package, so the
# integration tests below skip cleanly where it is absent; the pure-helper tests above always run.
def _langchain_available() -> bool:
    # Treat only the genuine absence of the top-level ``langchain`` package as
    # "unavailable" (find_spec returns None for a missing top-level module). If
    # ``langchain`` is installed, probing the submodule may raise
    # ModuleNotFoundError from a broken install (e.g. a missing transitive
    # dependency) — let that surface rather than silently skipping the tests.
    if importlib.util.find_spec("langchain") is None:
        return False
    return importlib.util.find_spec("langchain.agents") is not None


requires_langchain = pytest.mark.skipif(
    not _langchain_available(), reason="langchain (examples-harness only) not installed"
)


def _fixtures():
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    class ToolFakeModel(FakeMessagesListChatModel):
        """A scripted fake model that accepts bind_tools (FakeMessagesListChatModel does not)."""

        def bind_tools(self, tools, **kwargs):
            return self

    @tool
    def describe_problem() -> str:
        """Orient: whether a model exists."""
        return "model_initialized: false"

    def run(scripted_msgs, *, max_idle_steps=3, return_messages=False):
        stats = {"tools_executed": 0, "dropped_dup_tool_calls": 0,
                 "idle_nudges": 0, "gave_up_no_tools": False}
        guard = agent._tool_dispatch_guard(stats, max_idle_steps=max_idle_steps)
        graph = create_agent(ToolFakeModel(responses=scripted_msgs),
                             tools=[describe_problem], middleware=[guard])
        out = graph.invoke({"messages": [HumanMessage(content="build a model")]},
                           config={"recursion_limit": 50})
        return (stats, out["messages"]) if return_messages else stats

    return run, AIMessage


def _call(id_="c1"):
    _, AIMessage = _fixtures()
    return AIMessage(content="", tool_calls=[
        {"name": "describe_problem", "args": {}, "id": id_, "type": "tool_call"}])


@requires_langchain
def test_duplicate_tool_calls_in_a_batch_execute_once():
    run, AIMessage = _fixtures()
    dup_batch = AIMessage(content="", tool_calls=[
        {"name": "describe_problem", "args": {}, "id": "c1", "type": "tool_call"},
        {"name": "describe_problem", "args": {}, "id": "c2", "type": "tool_call"},
    ])
    stats = run([dup_batch, AIMessage(content="done — fit adequate")])
    assert stats["dropped_dup_tool_calls"] == 1
    assert stats["tools_executed"] == 1
    assert stats["gave_up_no_tools"] is False


@requires_langchain
def test_idle_run_that_never_fires_a_tool_is_pushed_back_then_given_up():
    run, AIMessage = _fixtures()
    text_only = [AIMessage(content=f"I would build a model... ({i})") for i in range(10)]
    stats = run(text_only, max_idle_steps=3)
    assert stats["tools_executed"] == 0
    assert stats["idle_nudges"] == 3
    assert stats["gave_up_no_tools"] is True


@requires_langchain
def test_nudge_rescues_a_run_that_starts_by_stalling():
    # Two text-only turns, then the model finally emits a real tool call and finishes: the
    # push-back keeps the loop alive so the proposed tool call actually fires.
    run, AIMessage = _fixtures()
    stats = run([
        AIMessage(content="let me think about the mechanism"),
        AIMessage(content="still reasoning, no action yet"),
        _call(),
        AIMessage(content="fit adequate, finishing"),
    ], max_idle_steps=3)
    assert stats["tools_executed"] == 1
    assert stats["idle_nudges"] == 2
    assert stats["gave_up_no_tools"] is False


@requires_langchain
def test_normal_finish_after_a_tool_is_not_disturbed():
    # A no-tool assistant turn AFTER a tool has fired is a genuine finish — never nudged.
    run, AIMessage = _fixtures()
    stats = run([_call(), AIMessage(content="done, reporting the final fit")])
    assert stats["tools_executed"] == 1
    assert stats["idle_nudges"] == 0
    assert stats["gave_up_no_tools"] is False


@requires_langchain
def test_dedupe_also_collapses_the_raw_additional_kwargs_copy():
    # A duplicate batch whose raw additional_kwargs copy reuses the SAME id must be collapsed
    # there too — not just in .tool_calls — so a stale duplicate cannot resurface.
    run, AIMessage = _fixtures()
    dup_batch = AIMessage(
        content="",
        tool_calls=[
            {"name": "describe_problem", "args": {}, "id": "dup", "type": "tool_call"},
            {"name": "describe_problem", "args": {}, "id": "dup", "type": "tool_call"},
        ],
        additional_kwargs={"tool_calls": [
            {"id": "dup", "type": "function",
             "function": {"name": "describe_problem", "arguments": "{}"}},
            {"id": "dup", "type": "function",
             "function": {"name": "describe_problem", "arguments": "{}"}},
        ]},
    )
    stats, messages = run([dup_batch, AIMessage(content="done")], return_messages=True)
    assert stats["tools_executed"] == 1
    ai = next(m for m in messages if getattr(m, "tool_calls", None))
    assert len(ai.tool_calls) == 1
    assert len(ai.additional_kwargs.get("tool_calls", [])) == 1
