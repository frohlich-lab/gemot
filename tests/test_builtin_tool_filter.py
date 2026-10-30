"""The lead's model call carries its own tools and ``task``, and no deepagents built-in.

deepagents adds filesystem + todo tools to every agent on top of the tools we pass, and
:func:`gemot.agent.subagents._builtin_tool_filter` strips them from the lead's model call by
name. That name list goes stale silently: FilesystemMiddleware's ``delete`` was never on it, so
the lead was sent its schema on every turn, and could call it, on deepagents 0.7.1 and 0.7.32
alike. These tests build the real ``create_deep_agent`` graph and record what the lead's model is
bound to, so the next built-in deepagents adds fails here instead of riding along.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("deepagents", reason="harness-only dependency")

from gemot.agent import subagents as S  # noqa: E402
from gemot.agent.tools import ORCHESTRATOR_TOOLS  # noqa: E402


def _lead_bound_tools(run) -> list[list[str]]:
    """The tool names the lead's model is bound to, one sorted list per model call."""
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import StructuredTool

    bound: list[list[str]] = []

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            bound.append(sorted(t["name"] if isinstance(t, dict) else t.name for t in tools))
            return self

    graph = create_deep_agent(
        model=ToolModel(responses=[AIMessage(content="done")]),
        tools=[
            StructuredTool.from_function(lambda: "", name=name, description=name)
            for name in ORCHESTRATOR_TOOLS
        ],
        system_prompt="lead",
        subagents=[
            {
                "name": "analyst",
                "description": "analyst",
                S._subagent_prompt_key(): "analyst",
                "tools": [],
            }
        ],
        middleware=[S._builtin_tool_filter()],
    )
    run(graph, {"messages": [HumanMessage(content="inspect")]})
    return bound


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(lambda graph, state: graph.invoke(state), id="sync"),
        # What the driver runs: it streams with `astream`, so `awrap_model_call` is the path.
        pytest.param(lambda graph, state: asyncio.run(graph.ainvoke(state)), id="async"),
    ],
)
def test_lead_is_bound_to_its_own_tools_and_task_only(run):
    """The lead was bound to `delete` on every turn because it was missing from
    `_BUILTIN_TOOLS_TO_HIDE`. Comparing against the lead's real tool names also catches the
    converse: one of ORCHESTRATOR_TOOLS named like a built-in would be stripped with it."""
    assert _lead_bound_tools(run) == [sorted([*ORCHESTRATOR_TOOLS, "task"])]
