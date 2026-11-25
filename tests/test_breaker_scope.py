"""The breaker cuts a tool for the caller whose calls failed, not for the run (gemot#133).

In sweep 63522750 one analyst's three identical SQL errors disabled `query_data` for the lead and
every later analyst, with about 2.5 days of a 3-day budget left (lun 103, froehlich 106). The
disabled set and the error streaks now belong to the lead (whose scope is the run) or to one
`task`; a server that went away is still cut for everyone.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import re

import pytest

from gemot import prompts
from gemot.agent import tools as T
from gemot.agent.circuit_breaker import (
    _TASK,
    _TOOL_ERROR_DISABLE_THRESHOLD,
    _new_breaker_state,
    _wrap_tools_with_circuit_breaker,
    disabled_tools,
    task_disabled_tools,
)

_SQL_ERROR = (
    'SQL error: Binder Error: Referenced column "residual" not found in FROM clause! '
    "Call `list_data_tables` for the exact table/column names."
)


def _query_data_result(sql: str):
    """What the server's `query_data` returns: rows for a query that works, and its
    ``{"ok": false, "error": ...}`` report for one that does not."""
    payload = {"ok": False, "error": _SQL_ERROR} if "bad" in sql else {"ok": True, "rows": []}
    return ([{"type": "text", "text": json.dumps(payload)}], {"structured_content": payload})


class _QueryData:
    """Enough of the `query_data` tool for the wrapper, recording which calls reached it."""

    name = "query_data"
    args = {"sql": {"type": "string"}}

    def __init__(self):
        self.reached: list[str] = []

        async def _run(sql: str):
            self.reached.append(sql)
            return _query_data_result(sql)

        self.coroutine = _run


def _text(result) -> str:
    content, _ = result
    return "".join(b["text"] for b in content) if isinstance(content, list) else content


async def _as(task: str | None, coro):
    """Run ``coro`` as `task` would, or as the lead for ``None``."""
    token = _TASK.set(task)
    try:
        return await coro
    finally:
        _TASK.reset(token)


def test_one_tasks_errors_leave_the_tool_to_the_lead_and_other_tasks():
    tool, state = _QueryData(), _new_breaker_state()
    _wrap_tools_with_circuit_breaker([tool], state)

    async def _run():
        stuck = [
            await _as("signal_engineer task 1", tool.coroutine(sql="bad"))
            for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD + 1)
        ]
        other = await _as("mathematician task 2", tool.coroutine(sql="good"))
        lead = await _as(None, tool.coroutine(sql="good"))
        return stuck, other, lead

    stuck, other, lead = asyncio.run(_run())

    assert "TOOL DISABLED" in _text(stuck[_TOOL_ERROR_DISABLE_THRESHOLD - 1])
    assert "the rest of this task" in _text(stuck[-1])
    assert tool.reached == ["bad"] * _TOOL_ERROR_DISABLE_THRESHOLD + ["good", "good"]
    assert "TOOL DISABLED" not in _text(other) + _text(lead)
    assert disabled_tools(state) == []
    assert task_disabled_tools(state) == {"query_data": 1}
    assert any("for signal_engineer task 1" in e for e in state["events"])


def test_a_new_task_starts_with_the_tool_back():
    """A task is a fresh conversation: it has never seen the errors that cut the tool before."""
    tool, state = _QueryData(), _new_breaker_state()
    _wrap_tools_with_circuit_breaker([tool], state)

    async def _run():
        for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD):
            await _as("signal_engineer task 1", tool.coroutine(sql="bad"))
        return await _as("signal_engineer task 3", tool.coroutine(sql="good"))

    assert "TOOL DISABLED" not in _text(asyncio.run(_run()))
    assert tool.reached[-1] == "good"


def test_parallel_tasks_do_not_add_up_to_a_streak():
    """lun 103's analyst found `query_data` disabled "after my first two calls": streaks were the
    run's, so errors from analysts running side by side counted toward one."""
    tool, state = _QueryData(), _new_breaker_state()
    _wrap_tools_with_circuit_breaker([tool], state)

    async def _analyst(task: str):
        token = _TASK.set(task)
        try:
            for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD - 1):
                await tool.coroutine(sql="bad")
                await asyncio.sleep(0)  # let the other analyst's call in between
        finally:
            _TASK.reset(token)

    async def _run():
        await asyncio.gather(_analyst("signal_engineer task 1"), _analyst("mathematician task 2"))

    asyncio.run(_run())
    assert len(tool.reached) == 2 * (_TOOL_ERROR_DISABLE_THRESHOLD - 1)
    assert task_disabled_tools(state) == {}


def test_the_lead_still_loses_a_tool_for_the_run():
    tool, state = _QueryData(), _new_breaker_state()
    _wrap_tools_with_circuit_breaker([tool], state)

    results = [asyncio.run(tool.coroutine(sql="bad")) for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD)]

    assert "the rest of this run" in _text(results[-1])
    assert disabled_tools(state) == ["query_data"]


# --- through the real panel ---------------------------------------------------------------------


def _panel_run() -> tuple[dict, list[str]]:
    """The graph the driver builds, in miniature: a lead holding `query_data` and the breaker's
    middleware, a signal engineer stuck on one bad query, then a mathematician and the lead each
    making a good one. Returns the breaker state and the queries that reached the tool."""
    pytest.importorskip("deepagents", reason="harness-only dependency")
    from deepagents import create_deep_agent
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import StructuredTool
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from gemot.agent.circuit_breaker import _disabled_tool_filter
    from gemot.agent.subagents import _subagent_prompt_key

    reached: list[str] = []

    async def _query_data(sql: str):
        reached.append(sql)
        return _query_data_result(sql)

    query_data = StructuredTool.from_function(
        coroutine=_query_data,
        name="query_data",
        description="SQL over the data",
        response_format="content_and_artifact",
    )
    state = _new_breaker_state()
    _wrap_tools_with_circuit_breaker([query_data], state)

    ids = itertools.count()

    def turn(name, args):
        return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c{next(ids)}"}])

    def task(analyst):
        return turn("task", {"description": "read the residuals", "subagent_type": analyst})

    scripts = {
        "PERSONA-LEAD": [
            task("signal_engineer"),
            task("mathematician"),
            turn("query_data", {"sql": "good lead"}),
            AIMessage("done"),
        ],
        "PERSONA-SIGNAL": [
            *[turn("query_data", {"sql": "bad"}) for _ in range(_TOOL_ERROR_DISABLE_THRESHOLD + 1)],
            AIMessage("report"),
        ],
        "PERSONA-MATH": [turn("query_data", {"sql": "good math"}), AIMessage("report")],
    }

    class Scripted(BaseChatModel):
        profile: dict | None = {"max_input_tokens": 1_000_000}

        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            assert isinstance(messages[0], SystemMessage)
            script = next(v for k, v in scripts.items() if k in str(messages[0].content))
            return ChatResult(generations=[ChatGeneration(message=script.pop(0))])

    def analyst(name, persona):
        return {
            "name": name,
            "description": name,
            _subagent_prompt_key(): persona,
            "tools": [query_data],
        }

    agent = create_deep_agent(
        model=Scripted(),
        tools=[query_data],
        system_prompt="PERSONA-LEAD",
        subagents=[
            analyst("signal_engineer", "PERSONA-SIGNAL"),
            analyst("mathematician", "PERSONA-MATH"),
        ],
        middleware=[_disabled_tool_filter(state)],
    )
    asyncio.run(agent.ainvoke({"messages": [HumanMessage("build a model")]}))
    assert not any(scripts.values()), "every scripted turn was taken"
    return state, reached


def test_an_analysts_errors_leave_query_data_to_the_lead_and_the_other_analysts():
    """Async, as the driver runs it (`astream`); the MCP tools have no sync path to take."""
    state, reached = _panel_run()

    assert reached == ["bad"] * _TOOL_ERROR_DISABLE_THRESHOLD + ["good math", "good lead"]
    assert state["calls"]["query_data"] == {"tool_error": 3, "disabled": 1, "ok": 2}
    assert disabled_tools(state) == []
    assert task_disabled_tools(state) == {"query_data": 1}
    assert any("for signal_engineer task 1" in e for e in state["events"])


# --- what the notice tells the caller to do instead ----------------------------------------------


def _named_tools(text: str) -> set[str]:
    return set(re.findall(r"`(\w+)`", text))


@pytest.mark.parametrize("disabled", sorted(prompts._DISABLED_TOOL_HINTS))
def test_a_hint_names_only_tools_every_holder_of_the_disabled_tool_holds(disabled):
    """The query_data hint sent the signal engineer to `get_measurements`, which it does not hold."""
    named = _named_tools(prompts._DISABLED_TOOL_HINTS[disabled])
    holders = [set(ts) for ts in T._TOOLSETS if disabled in ts]
    assert holders, f"no agent holds {disabled}"
    assert all(named <= held for held in holders)


def test_the_default_hint_names_no_tool_an_agent_may_lack():
    """It named `get_errors`, which no agent on the panel holds."""
    named = _named_tools(prompts._DISABLED_TOOL_HINT_DEFAULT)
    assert all(named <= set(ts) for ts in T._TOOLSETS)
