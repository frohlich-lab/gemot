"""The explorer reads a run's whole trace, marks the spans that failed, and sums tokens right.

Each test is a defect of #125, seen in the traces of sweep 63372750: a trace cut at Phoenix's
default of 1000 spans, a ``task`` that died on a dropped connection drawn as a duplicate with no
result, and a run's total short of its prompt plus completion. The frames have the shape Phoenix
21 returns (``gemot.explorer.bundle``).
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from gemot.explorer import bundle as B

_T0 = pd.Timestamp("2026-11-07T09:00:00Z")


def _span(span_id, parent, second, kind, *, name=None, status="OK", message="", **attributes):
    """One row of Phoenix 21's spans dataframe: its fixed columns and flattened attributes."""
    start = _T0 + pd.Timedelta(seconds=second)
    return {
        "name": name or kind.lower(),
        "span_kind": kind,
        "parent_id": parent,
        "start_time": start,
        "end_time": start + pd.Timedelta(milliseconds=500),
        "status_code": status,
        "status_message": message,
        "events": [],
        "context.span_id": span_id,
        "context.trace_id": "trace",
        "attributes.openinference.span.kind": kind,
        **{f"attributes.{key}": value for key, value in attributes.items()},
    }


def _llm(span_id, parent, second, agent, *, text="", task=None, tokens=(100, 20, 120), **kw):
    message = {"message.role": "assistant", "message.content": text}
    if task is not None:
        message["message.tool_calls"] = [
            {"tool_call.function.name": "task", "tool_call.function.arguments": json.dumps(task)}
        ]
    prompt, completion, total = tokens
    return _span(
        span_id,
        parent,
        second,
        "LLM",
        name="ChatOpenAI",
        **{
            "metadata": {"lc_agent_name": agent},
            "llm.output_messages": [message],
            "llm.token_count.prompt": prompt,
            "llm.token_count.completion": completion,
            "llm.token_count.total": total,
        },
        **kw,
    )


def _task(span_id, parent, second, delegation, **kw):
    return _span(
        span_id,
        parent,
        second,
        "TOOL",
        name="task",
        **{"tool.name": "task", "input.value": json.dumps(delegation)},
        **kw,
    )


def _frame(rows):
    return pd.DataFrame(rows).set_index("context.span_id", drop=False)


def _serve(monkeypatch, frame):
    """Stand in for Phoenix 21's client: the newest ``limit`` spans that started before
    ``end_time``, ties broken by the later-stored span first, as its server orders them."""
    requests = []

    def get_spans_dataframe(*, end_time=None, limit, **_):
        requests.append(limit)
        rows = frame if end_time is None else frame[frame["start_time"] < end_time]
        return rows.iloc[::-1].sort_values("start_time", ascending=False, kind="stable").head(limit)

    spans = SimpleNamespace(get_spans_dataframe=get_spans_dataframe)
    monkeypatch.setitem(
        sys.modules,
        "phoenix.client",
        SimpleNamespace(Client=lambda **_: SimpleNamespace(spans=spans)),
    )
    monkeypatch.setitem(
        sys.modules,
        "phoenix.client.types.spans",
        SimpleNamespace(SpanQuery=lambda: SimpleNamespace(where=lambda condition: condition)),
    )
    return requests


@pytest.mark.parametrize(
    "tied",
    [
        # Straddling the first page's end (the newest 1000 of 2500 reach back to span 1500): a
        # cursor at the earliest span read would skip the ten the page did not hold.
        range(1490, 1510),
        # More than a page, all at one instant: the cursor cannot move past them.
        range(200, 1700),
    ],
)
def test_a_trace_longer_than_a_page_is_read_whole(monkeypatch, tied):
    """fontaine runs 5 and 8 came back with exactly 1000 spans, the client's default, and without
    their earliest, the root among them (#125)."""
    seconds = [tied[0] if i in tied else i for i in range(2500)]
    frame = _frame(
        [_span("root", None, -1, "CHAIN", name="model_build_run")]
        + [_span(f"s{i:04d}", "root", s, "TOOL") for i, s in enumerate(seconds)]
    )
    requests = _serve(monkeypatch, frame)

    df = B._fetch_trace_df("trace", "http://phoenix.example", "project")

    assert df.index.is_unique
    assert sorted(df.index) == sorted(frame.index)
    assert len(requests) > 1


def _delegating_run(failed_message):
    """A finished run: the lead delegates once, the ``task`` fails and is run again."""
    delegation = {"subagent_type": "mathematician", "description": "check identifiability"}
    return [
        _span("root", None, 0, "CHAIN", name="model_build_run"),
        _llm("lead1", "root", 1, "lead", task=delegation),
        _task("task1", "root", 2, delegation, status="ERROR", message=failed_message),
        _llm("math1", "task1", 3, "mathematician", text="first try"),
        _task("task2", "root", 2103, delegation, **{"output.value": "identifiable"}),
        _llm("math2", "task2", 2104, "mathematician", text="second try"),
        _llm("lead2", "root", 2200, "lead", text="done"),
    ]


def test_a_failed_span_is_drawn_as_failed_with_its_status_message():
    """In lun 11 a ``task`` ended in ERROR after 35 min and was run again; the explorer drew the
    failed call as a duplicate with no result (#125)."""
    message = "OpenAIConnectionError: Connection error."
    df = _frame(_delegating_run(message))

    turns, _, status = B.load_turns("trace", "http://unused", "project", df=df, finished=True)

    assert status["ok"] and status["complete"]
    first, retry = turns[0]["tools"]
    assert (first["error"], first["delegation"]["items"][0]["text"]) == (message, "first try")
    assert (retry["error"], retry["result"]) == (None, "identifiable")


def test_a_span_without_a_total_counts_its_prompt_and_completion():
    """A turn the thinking hand-off combined recorded no total; the run's total was then short of
    its prompt plus completion (lun 11 and 14, one turn of 371k prompt and 132k completion)."""
    rows = _delegating_run("")
    rows[3] = _llm("math1", "task1", 3, "mathematician", tokens=(371_000, 132_000, float("nan")))
    df = _frame(rows)

    turns, totals, _ = B.load_turns("trace", "http://unused", "project", df=df, finished=True)

    analyst_turn = turns[0]["tools"][0]["delegation"]["items"][0]
    assert analyst_turn["tokens"]["total"] == 503_000
    assert totals["total"]["total"] == totals["total"]["prompt"] + totals["total"]["completion"]


_FIRST = {"subagent_type": "mathematician", "description": "check identifiability"}
_LATER = {"subagent_type": "mathematician", "description": "check version 13 against 15"}


def test_a_finished_run_missing_its_earliest_spans_says_so_and_draws_no_running_delegation():
    """A read without the root left the first analyst turns with no ``task`` span above them, and
    the explorer drew them as delegations still running, in turn 0, with a later delegation's
    description (#125)."""
    df = _frame(
        [
            # The root, the lead's first turn and its `task` span were not read.
            _llm("math1", "task1", 3, "mathematician", text="first look"),
            _llm("lead2", "root", 10, "lead", task=_LATER),
            _task("task2", "root", 11, _LATER, **{"output.value": "identifiable"}),
            _llm("math2", "task2", 12, "mathematician", text="second look"),
        ]
    )

    turns, totals, status = B.load_turns("trace", "http://unused", "project", df=df, finished=True)

    assert status["ok"] and not status["complete"]
    assert "not its root" in status["note"] and "1 analyst turn " in status["note"]
    [delegation] = [tool["delegation"] for t in turns for tool in t["tools"]]
    assert not delegation.get("in_progress")
    assert [item["text"] for item in delegation["items"]] == ["second look"]
    assert (totals["lead_turns"], totals["analyst_turns"]) == (1, 1)


def test_a_running_analyst_takes_the_description_of_its_types_latest_delegation():
    """The first delegation of a type is long finished when the lead delegates that type again."""
    df = _frame(
        [
            _llm("lead1", "root", 1, "lead", task=_FIRST),
            _task("task1", "root", 2, _FIRST, **{"output.value": "identifiable"}),
            _llm("math1", "task1", 3, "mathematician", text="first look"),
            _llm("lead2", "root", 10, "lead", task=_LATER),
            # Still running: its `task` span has not ended, so Phoenix does not hold it yet.
            _llm("math2", "task2", 11, "mathematician", text="second look"),
        ]
    )

    turns, _, _ = B.load_turns("trace", "http://unused", "project", df=df)

    [running] = [
        tool["delegation"]
        for t in turns
        for tool in t["tools"]
        if tool.get("delegation", {}).get("in_progress")
    ]
    assert running["description"] == _LATER["description"]
