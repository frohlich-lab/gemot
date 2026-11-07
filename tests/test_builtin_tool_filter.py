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


def _analysts_bound_tools(run) -> tuple[dict[str, list[list[str]]], dict[str, list[str]]]:
    """The tool names each analyst's model is bound to, one sorted list per model call, with the
    lead delegating to every analyst in turn; and the tools each one's spec holds."""
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import StructuredTool

    from gemot.agent.biocurator import _build_biocurator_subagent
    from gemot.agent.tools import ALLOWED_TOOLS

    def tool(name):
        return StructuredTool.from_function(lambda: "", name=name, description=name)

    by_name = {name: tool(name) for name in ALLOWED_TOOLS}
    # The biocurator holds its bio-server tools and the one petab read its caller appends.
    specs = S._build_subagents(by_name) + [
        _build_biocurator_subagent([tool("bio_lookup"), by_name["list_observables"]], [])
    ]
    calls: list[tuple[str, list[str]]] = []  # (system prompt, bound tool names) per model call
    last: list[str] = []

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            last[:] = sorted(t["name"] if isinstance(t, dict) else t.name for t in tools)
            return self

        def _generate(self, messages, *args, **kwargs):
            calls.append((next(m.content for m in messages if m.type == "system"), list(last)))
            return super()._generate(messages, *args, **kwargs)

    delegate = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "id": f"call_{i}",
                    "args": {"description": "inspect", "subagent_type": spec["name"]},
                }
            ],
        )
        for i, spec in enumerate(specs)
    ]
    graph = create_deep_agent(
        model=ToolModel(
            responses=[m for d in delegate for m in (d, AIMessage(content="report"))]
            + [AIMessage(content="done")]
        ),
        tools=[by_name[name] for name in ORCHESTRATOR_TOOLS],
        system_prompt="lead",
        subagents=specs,
        middleware=[S._builtin_tool_filter()],
    )
    run(graph, {"messages": [HumanMessage(content="inspect")]})
    key = S._subagent_prompt_key()
    bound = {
        spec["name"]: [names for system, names in calls if system.startswith(spec[key])]
        for spec in specs
    }
    return bound, {spec["name"]: sorted(t.name for t in spec["tools"]) for spec in specs}


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(lambda graph, state: graph.invoke(state), id="sync"),
        pytest.param(lambda graph, state: asyncio.run(graph.ainvoke(state)), id="async"),
    ],
)
def test_each_analyst_is_bound_to_its_own_tools_only(run):
    """gemot#122: the analysts were left deepagents' filesystem built-ins, and in sweep 63372750
    78 of their 909 tool calls went to them, hunting for simulation files in a filesystem that is
    empty by design. deepagents builds each subagent its own middleware stack, so the lead's
    filter never reached them; each spec now carries its own."""
    bound, holds = _analysts_bound_tools(run)
    assert bound == {name: [held] for name, held in holds.items()}
