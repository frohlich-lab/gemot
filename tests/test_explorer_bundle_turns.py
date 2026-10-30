"""The explorer tells the lead's LLM calls from an analyst's by ``metadata.lc_agent_name``.

It used to read each span's system prompt from ``llm.input_messages``. The tracer now keeps only
the messages a call adds to its conversation's previous one (``gemot.tracing``), so a span past
the first of its conversation has no system prompt, and a still-running analyst's turns -- which
no finished ``task`` span parents yet -- would have been drawn as the lead's own
(``gemot.explorer.bundle``).
"""

from __future__ import annotations

import json

import pandas as pd

from gemot.explorer import bundle as B

_T0 = pd.Timestamp("2026-10-30T12:00:00Z")


def _llm(span_id, agent, second, *, task=None, input_messages=float("nan")):
    """An LLM span as Phoenix's spans dataframe returns it; ``task`` makes it delegate."""
    if task is None:
        output = [{"message.role": "assistant", "message.content": f"{agent} turn {second}"}]
    else:
        call = {"tool_call.function.name": "task", "tool_call.function.arguments": json.dumps(task)}
        output = [{"message.role": "assistant", "message.tool_calls": [call]}]
    return {
        "context.span_id": span_id,
        "parent_id": None,
        "name": "ChatOpenAI",
        "start_time": _T0 + pd.Timedelta(seconds=second),
        "attributes.openinference.span.kind": "LLM",
        "attributes.metadata": {"lc_agent_name": agent, "checkpoint_ns": "tools:1"},
        "attributes.llm.input_messages": input_messages,
        "attributes.llm.output_messages": output,
    }


def test_a_running_analyst_is_its_own_delegation_by_agent_name():
    delegation = {"subagent_type": "mathematician", "description": "check identifiability"}
    first = [{"message.role": "system", "message.content": "You are a mathematician"}]
    rows = [
        _llm("lead1", "lead", 0, task=delegation),
        _llm("math1", "mathematician", 1, input_messages=first),
        # Past its first call: only the new messages, no system prompt to tell it by.
        _llm("math2", "mathematician", 2),
    ]
    rows[2]["attributes.metadata"] = json.dumps(rows[2]["attributes.metadata"])  # older Phoenix
    df = pd.DataFrame(rows).set_index("context.span_id")

    turns, totals, status = B.load_turns("trace", "http://unused", "project", df=df)

    assert status["ok"]
    assert (totals["lead_turns"], totals["analyst_turns"]) == (1, 2)
    [block] = [
        tool["delegation"] for turn in turns for tool in turn["tools"] if "delegation" in tool
    ]
    assert block["in_progress"]
    assert (block["subagent_type"], block["description"]) == (
        "mathematician",
        "check identifiability",
    )
    assert [item["text"] for item in block["items"]] == [
        "mathematician turn 1",
        "mathematician turn 2",
    ]
