"""A breaker strike is a model turn that failed, not a call (gemot#163).

In run 63612603_107 of sweep gemot_pools_20261107 a signal engineer sent three `query_data` calls
in one turn, all with the same bad cast (``No function matches '/(DOUBLE, VARCHAR)'``). Counted
per call, they disabled the tool for the rest of the task before the agent had read one of the
errors. The turn comes from the graph the calls run in, so these tests drive the real panel.
"""

from __future__ import annotations

import asyncio
import itertools
import json

import pytest

from gemot.agent.circuit_breaker import (
    _TOOL_ERROR_DISABLE_THRESHOLD,
    _new_breaker_state,
    _wrap_tools_with_circuit_breaker,
    disabled_tools,
    task_disabled_tools,
)

T = _TOOL_ERROR_DISABLE_THRESHOLD

_CAST_ERROR = (
    "SQL error: Binder Error: No function matches the given name and argument types "
    "'/(DOUBLE, VARCHAR)'."
)


def _run(caller: str, turns: list[list[str]], yields: bool = True) -> tuple[dict, list[str]]:
    """Drive a panel whose ``caller`` (``"lead"`` or ``"analyst"``) takes ``turns``.

    A turn is the `query_data` calls one model message issues together, one per SQL string; a
    query containing "bad" fails. With ``yields`` off the tool never yields, so a turn's calls run
    one after another. Returns the breaker state and the queries that reached the tool."""
    pytest.importorskip("deepagents", reason="harness-only dependency")
    from deepagents import create_deep_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import StructuredTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from gemot.agent.circuit_breaker import _disabled_tool_filter
    from gemot.agent.subagents import _subagent_prompt_key

    reached: list[str] = []

    async def _query_data(sql: str):
        reached.append(sql)
        # Yield, as the server round trip does, so a turn's calls are all in flight at once.
        if yields:
            await asyncio.sleep(0)
        ok = "bad" not in sql
        payload = {"ok": True, "rows": []} if ok else {"ok": False, "error": _CAST_ERROR}
        return ([{"type": "text", "text": json.dumps(payload)}], {"structured_content": payload})

    query_data = StructuredTool.from_function(
        coroutine=_query_data,
        name="query_data",
        description="SQL over the data",
        response_format="content_and_artifact",
    )
    state = _new_breaker_state()
    _wrap_tools_with_circuit_breaker([query_data], state)

    ids = itertools.count()

    def message(calls: list[tuple[str, dict]]) -> AIMessage:
        return AIMessage(
            "", tool_calls=[{"name": n, "args": a, "id": f"c{next(ids)}"} for n, a in calls]
        )

    batches = [message([("query_data", {"sql": sql}) for sql in turn]) for turn in turns]
    dispatch = message([("task", {"description": "read the data", "subagent_type": "analyst"})])
    scripts = {
        "PERSONA-LEAD": [*(batches if caller == "lead" else [dispatch]), AIMessage("done")],
        "PERSONA-ANALYST": [*batches, AIMessage("report")] if caller == "analyst" else [],
    }

    class Scripted(BaseChatModel):
        profile: dict | None = {"max_input_tokens": 1_000_000}

        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            script = next(v for k, v in scripts.items() if k in str(messages[0].content))
            return ChatResult(generations=[ChatGeneration(message=script.pop(0))])

    agent = create_deep_agent(
        model=Scripted(),
        tools=[query_data],
        system_prompt="PERSONA-LEAD",
        subagents=[
            {
                "name": "analyst",
                "description": "analyst",
                _subagent_prompt_key(): "PERSONA-ANALYST",
                "tools": [query_data],
            }
        ],
        middleware=[_disabled_tool_filter(state)],
    )
    asyncio.run(agent.ainvoke({"messages": [HumanMessage("build a model")]}))
    assert not any(scripts.values()), "every scripted turn was taken"
    return state, reached


def _disabled(state: dict, caller: str) -> bool:
    if caller == "lead":
        return disabled_tools(state) == ["query_data"]
    return task_disabled_tools(state) == {"query_data": 1}


@pytest.mark.parametrize("caller", ["lead", "analyst"])
def test_one_batch_of_identical_failures_is_one_strike(caller):
    """63612603_107: three calls that failed alike in one turn cut `query_data` before the agent
    had seen the error. Now they are one strike, and the corrected query reaches the tool."""
    state, reached = _run(caller, [["bad"] * T, ["good"]])

    assert reached == ["bad"] * T + ["good"]
    assert state["calls"]["query_data"] == {"tool_error": T, "ok": 1}
    assert not _disabled(state, caller)
    assert state["events"] == []


@pytest.mark.parametrize("caller", ["lead", "analyst"])
def test_the_same_failure_in_consecutive_turns_still_trips(caller):
    """A caller that sends the failing batch again after reading its errors is stuck: on the
    threshold's turn the tool is cut, and the next turn's call never reaches it."""
    state, reached = _run(caller, [["bad"] * T] * T + [["bad"]])

    assert reached == ["bad"] * T * T
    assert state["calls"]["query_data"] == {"tool_error": T * T, "disabled": 1}
    assert _disabled(state, caller)
    assert len(state["events"]) == 1 and f"after {T} consecutive turns" in state["events"][0]


@pytest.mark.parametrize("caller", ["lead", "analyst"])
def test_the_threshold_turn_reaches_the_tool_whatever_order_its_calls_run_in(caller):
    """The test above failed on main one run in a few, a call short: a call of the threshold turn
    that started after the one that tripped returned was short-circuited. Run one after another,
    the turn's calls still all reach the tool, and the next turn's does not. (From gemot#176, which
    fixed the same race another way.)"""
    state, reached = _run(caller, [["bad"] * T] * T + [["bad"]], yields=False)

    assert reached == ["bad"] * T * T
    assert state["calls"]["query_data"] == {"tool_error": T * T, "disabled": 1}
