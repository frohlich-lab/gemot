"""A turn the provider cuts off mid-answer, on the panel's real graph: the lead's and an analyst's.

gemot#161: a call that spends the whole output budget and stops with ``finish_reason == "length"``
after writing some text, with no tool call, was neither handed off by ``agent.thinking`` nor counted
by the runaway guard, since both asked for empty content. On the lead the graph then ended on a
no-tool turn and the driver bounced it as a silent stop (armistead 63612603_22: a 48-minute turn
lost, 18.5 more minutes to re-derive it); on an analyst the cut-off text became its report.

Built from the driver's own ``ChatOpenAI`` subclass over a mocked OpenRouter stream, so the turn
takes the production path from the SSE chunk to the hand-off, and from deepagents' ``create_deep_agent``
with the lead's guards and the analysts ``_build_subagents`` builds, each with its own runaway guard.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import random

import pytest

pytest.importorskip("deepagents", reason="harness-only dependency")
pytest.importorskip("langchain_openai", reason="harness-only dependency")

import openai  # noqa: E402
from deepagents import create_deep_agent  # noqa: E402
from langchain_core.messages import HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from gemot.agent import thinking  # noqa: E402
from gemot.agent.driver import _chat_model_class, _extra_body  # noqa: E402
from gemot.agent.guards import (  # noqa: E402
    _empty_report_guard,
    _reasoning_runaway_guard,
    _tool_dispatch_guard,
    new_guard_stats,
)
from gemot.agent.subagents import _build_subagents  # noqa: E402

# openai builds its client on `httpx` up to 2.x and on `httpx2` from 3.x (see test_billed_cost).
_httpx = importlib.import_module(openai.DefaultAsyncHttpxClient.__base__.__module__.split(".")[0])

MODEL = "xiaomi/mimo-v2.5"
LEAD = "PERSONA-LEAD"
#: 3000 words of thought that never repeats, so the repetition watch leaves it to the provider.
_RNG = random.Random(161)
THOUGHT = " ".join(f"w{_RNG.randrange(10**6)}" for _ in range(3000))
CUT_OFF = "So the readout needs a nuclear pool, and the next step is to"


def _event(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
    body = {
        "id": "gen",
        "provider": "Z.AI",
        "model": MODEL,
        "created": 1,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body)}\n\n"


def _cut_off() -> list[str]:
    """130k tokens of thought, a sentence of answer, then the provider's output cap."""
    pieces = [_event({"reasoning": THOUGHT[i : i + 2000]}) for i in range(0, len(THOUGHT), 2000)]
    usage = {"prompt_tokens": 1000, "completion_tokens": 131072, "total_tokens": 132072}
    return pieces + [_event({"content": CUT_OFF}, "length", usage)]


def _call(name: str, args: dict, text: str = "") -> list[str]:
    call = {"index": 0, "id": f"call-{name}", "type": "function"}
    call["function"] = {"name": name, "arguments": json.dumps(args)}
    usage = {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020}
    return ([_event({"content": text})] if text else []) + [
        _event({"tool_calls": [call]}, "tool_calls", usage)
    ]


def _say(text: str) -> list[str]:
    usage = {"prompt_tokens": 1000, "completion_tokens": 5, "total_tokens": 1005}
    return [_event({"content": text}, "stop", usage)]


def _system(body: dict) -> str:
    content = body["messages"][0].get("content")
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content or ""


def _run(lead: list, analyst: list, handoff: dict):
    """Run the panel on scripted replies: its final messages, guard stats, the hand-off requests
    as (agent, body), and the edits make_changes applied.

    ``lead`` and ``analyst`` are each agent's replies in turn; ``handoff`` maps an agent ("lead",
    "analyst") to its reply to the thinking hand-off, which is told apart by its instruction.
    """
    sent: list[tuple[str, dict]] = []

    def handler(request: _httpx.Request) -> _httpx.Response:
        body = json.loads(request.content)
        who = "lead" if LEAD in _system(body) else "analyst"
        last = body["messages"][-1]
        if last.get("role") == "user" and last.get("content") == thinking.HANDOFF:
            sent.append((who, body))
            events = handoff[who]
        else:
            script = lead if who == "lead" else analyst
            events = script.pop(0) if script else _say("done")
        text = "".join(events) + "data: [DONE]\n\n"
        return _httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=text.encode()
        )

    llm = _chat_model_class()(
        name="ChatOpenAI",
        model=MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key="x",
        stream_usage=True,
        extra_body=_extra_body(MODEL),
        http_async_client=openai.DefaultAsyncHttpxClient(transport=_httpx.MockTransport(handler)),
    )
    applied: list[dict] = []

    @tool
    def make_changes(note: str) -> str:
        """Apply a model edit."""
        applied.append({"note": note})
        return "applied"

    @tool
    def query_data(observable: str) -> str:
        """Read one observable's data."""
        return f"{observable}: rises, then settles"

    stats = new_guard_stats()
    agent = create_deep_agent(
        model=llm,
        tools=[make_changes, query_data],
        system_prompt=LEAD,
        subagents=_build_subagents({"query_data": query_data}, stats),
        middleware=[
            _reasoning_runaway_guard(stats),
            _tool_dispatch_guard(stats),
            _empty_report_guard(stats),
        ],
    )
    state = asyncio.run(
        agent.ainvoke({"messages": [HumanMessage("build a model")]}, config={"recursion_limit": 60})
    )
    return state["messages"], stats, sent, applied


@pytest.fixture(autouse=True)
def _fresh_thinking_state(monkeypatch):
    monkeypatch.setattr(thinking, "THINKING_OFF", {})
    monkeypatch.setattr(thinking, "NO_STREAM", set())


def test_a_lead_turn_cut_off_mid_answer_is_handed_off_not_bounced():
    # After a tool has run the idle nudge is off, so before the fix this turn ended the graph and
    # the driver bounced it as a silent stop; make_changes was never called.
    messages, stats, sent, applied = _run(
        lead=[_call("query_data", {"observable": "pSTAT5"}), _cut_off()],
        analyst=[],
        handoff={
            "lead": _call("make_changes", {"note": "nuclear pool"}, "Gist: add a nuclear pool.")
        },
    )
    assert applied == [{"note": "nuclear pool"}], "the hand-off's tool call must run"
    # The hand-off read the thinking AND the text the cut-off turn had written.
    ((who, body),) = sent
    assert who == "lead" and body["reasoning"] == {"enabled": False}
    assert body["messages"][-2] == {"role": "assistant", "content": f"{THOUGHT}\n\n{CUT_OFF}"}
    turn = next(
        m
        for m in messages
        if getattr(m, "tool_calls", None) and m.tool_calls[0]["name"] == "make_changes"
    )
    assert turn.response_metadata["thinking_handoff"]["text_chars"] == len(CUT_OFF)
    assert stats["reasoning_runaways"] == 1 and stats["reasoning_runaway_recovered"] == 1
    assert stats["thinking_cutoffs"] == {"length": 1}
    assert stats["output_cap_hits"] == 1
    assert stats["idle_nudges"] == 0


def test_an_analyst_turn_cut_off_mid_answer_is_handed_off_not_reported():
    # Before the fix the analyst's graph ended on the cut-off turn, and its half sentence went to
    # the lead as the report.
    task = {"description": "Which parameters are identifiable?", "subagent_type": "mathematician"}
    messages, stats, sent, applied = _run(
        lead=[_call("task", task)],
        analyst=[_cut_off(), _say("REPORT: k1 and k2 are not separately identifiable.")],
        handoff={"analyst": _call("query_data", {"observable": "pSTAT5"}, "Gist: check pSTAT5.")},
    )
    ((who, body),) = sent
    assert who == "analyst"
    assert body["messages"][-2] == {"role": "assistant", "content": f"{THOUGHT}\n\n{CUT_OFF}"}
    report = next(m for m in messages if isinstance(m, ToolMessage) and m.name == "task")
    assert "REPORT: k1 and k2" in str(report.content), report.content
    # Counted on the analyst's own guard, into the run's one stats dict.
    assert stats["reasoning_runaways"] == 1 and stats["reasoning_runaway_recovered"] == 1
    assert stats["thinking_cutoffs"] == {"length": 1}
    assert stats["output_cap_hits"] == 1
