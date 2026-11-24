"""Tests for the task-arg guard's wrapped arguments (gemot.agent.guards) — gemot#158.

GPT-6 luna calls ``task(subagent_type=X, parameters={"description": ...})``. The guard bounced that
three times, then relented by keeping the top-level keys only, which dispatched ``{subagent_type}``
alone; deepagents' ``TaskToolSchema`` refused every one of those calls ("description: Field
required"), 108 in 16 luna runs. The guard now lifts an allowed key out of such a wrapper, and
never dispatches a call it changed without both ``description`` and ``subagent_type``.

``_LUNA_CALL`` is luna's call as the lead LLM span of run 63782292_28 recorded it (description
abridged). The guard's ``#91`` behaviour (``prompt=`` next to a short ``description``) is tested in
tests/test_biocurator_input.py.
"""

from __future__ import annotations

import asyncio

from conftest import requires_langchain

from gemot import prompts
from gemot.agent import guards as G

_LUNA_DESCRIPTION = "Please do a read-only signal and residual audit of v5."
_LUNA_CALL = {"subagent_type": "signal_engineer", "parameters": {"description": _LUNA_DESCRIPTION}}
_CLEAN_CALL = {"subagent_type": "signal_engineer", "description": _LUNA_DESCRIPTION}


# --------------------------------------------------------------------- _hoist_task_args (pure)
def test_hoist_lifts_luna_wrapped_description_losslessly():
    # The regression: the relent kept {"subagent_type": ...} alone and the description was lost.
    assert G._hoist_task_args(_LUNA_CALL) == (_CLEAN_CALL, True)


def test_hoist_lifts_both_arguments_out_of_one_wrapper():
    args = {"args": {"description": "d", "subagent_type": "mathematician"}}
    assert G._hoist_task_args(args) == ({"description": "d", "subagent_type": "mathematician"}, True)


def test_hoist_drops_what_is_not_an_allowed_key_and_says_so():
    # A wrapper's other keys, and a non-dict extra (#91's prompt=), cannot be delivered.
    args = {"subagent_type": "s", "parameters": {"description": "d", "focus": "sigma"}}
    assert G._hoist_task_args(args) == ({"subagent_type": "s", "description": "d"}, False)
    args = {"subagent_type": "s", "description": "d", "prompt": "the real request"}
    assert G._hoist_task_args(args) == ({"subagent_type": "s", "description": "d"}, False)


def test_hoist_keeps_the_top_level_key_over_a_disagreeing_nested_one():
    args = {"subagent_type": "s", "description": "short", "parameters": {"description": "long"}}
    assert G._hoist_task_args(args) == ({"subagent_type": "s", "description": "short"}, False)
    # ... and an agreeing one loses nothing.
    args = {"subagent_type": "s", "description": "d", "parameters": {"description": "d"}}
    assert G._hoist_task_args(args) == ({"subagent_type": "s", "description": "d"}, True)


# --------------------------------------------------------------------- _missing_task_args (pure)
def test_missing_task_args_names_absent_blank_and_non_string_arguments():
    assert G._missing_task_args(_CLEAN_CALL) == []
    assert G._missing_task_args({"subagent_type": "s"}) == ["description"]
    assert G._missing_task_args({"subagent_type": "s", "description": "  "}) == ["description"]
    assert G._missing_task_args({"description": {"text": "d"}}) == ["description", "subagent_type"]


def test_incomplete_message_names_what_is_missing_and_what_was_dropped():
    msg = G._task_arg_incomplete_message(["parameters"], ["description"])
    assert msg.startswith("TASK CALL NOT DISPATCHED")
    assert "['parameters']" in msg and "no `description`." in msg


# --------------------------------------------------------------------- the middleware
class _FakeRequest:
    def __init__(self, args, id_="c1"):
        self.tool_call = {"name": "task", "args": args, "id": id_, "type": "tool_call"}

    def override(self, *, tool_call):
        return _FakeRequest(tool_call["args"], tool_call["id"])


def _run(mw, args, dispatched):
    def handler(req):
        dispatched.append(req.tool_call["args"])
        return "DISPATCHED"

    return mw.wrap_tool_call(_FakeRequest(args), handler)


@requires_langchain
def test_luna_call_is_dispatched_at_once_with_its_description():
    # No bounce: lifting the description out of `parameters` delivers everything luna sent, so the
    # three rejections that used to precede the (failing) relent cost lead turns for nothing.
    stats = G.new_guard_stats()
    dispatched: list = []
    assert _run(G._task_arg_guard(stats), _LUNA_CALL, dispatched) == "DISPATCHED"
    assert dispatched == [_CLEAN_CALL]
    assert stats["task_arg_hoists"] == 1
    assert stats["task_arg_rejections"] == 0 and stats["task_arg_relents"] == 0
    assert stats["unexpected_task_kwargs"] == ["parameters"]


@requires_langchain
def test_relent_without_a_description_goes_back_to_the_lead():
    # A wrapper holding no allowed key: bounced to the cap, then refused with what is missing,
    # never dispatched as `{subagent_type}` alone.
    from langchain_core.messages import ToolMessage

    stats = G.new_guard_stats()
    mw = G._task_arg_guard(stats, max_rejections=2)
    dispatched: list = []
    args = {"subagent_type": "signal_engineer", "parameters": {"prompt": _LUNA_DESCRIPTION}}
    outs = [_run(mw, args, dispatched) for _ in range(4)]
    assert dispatched == []
    assert all(isinstance(o, ToolMessage) and o.status == "error" for o in outs)
    rejected = G._task_arg_reject_message(["parameters"])
    incomplete = G._task_arg_incomplete_message(["parameters"], ["description"])
    assert [o.content for o in outs] == [rejected, rejected, incomplete, incomplete]
    assert stats["task_arg_rejections"] == 2
    assert stats["task_arg_relents"] == 2 and stats["task_arg_relent_failures"] == 2


@requires_langchain
def test_relent_lifts_the_description_out_of_a_wrapper_it_cannot_deliver_whole():
    stats = G.new_guard_stats()
    mw = G._task_arg_guard(stats, max_rejections=1)
    dispatched: list = []
    args = {"subagent_type": "signal_engineer", "parameters": {**_LUNA_CALL["parameters"], "n": 3}}
    assert _run(mw, args, dispatched).status == "error"  # `n` would be lost: bounce
    assert _run(mw, args, dispatched) == "DISPATCHED"
    assert dispatched == [_CLEAN_CALL]
    assert stats["task_arg_relents"] == 1 and stats["task_arg_relent_failures"] == 0
    assert stats["task_arg_hoists"] == 0


@requires_langchain
def test_async_path_lifts_and_refuses_alike():
    stats = G.new_guard_stats()
    mw = G._task_arg_guard(stats, max_rejections=0)
    dispatched: list = []

    async def handler(req):
        dispatched.append(req.tool_call["args"])
        return "DISPATCHED"

    assert asyncio.run(mw.awrap_tool_call(_FakeRequest(_LUNA_CALL), handler)) == "DISPATCHED"
    refused = asyncio.run(mw.awrap_tool_call(_FakeRequest({"parameters": {}}), handler))
    assert refused.status == "error"
    assert refused.content == G._task_arg_incomplete_message(
        ["parameters"], ["description", "subagent_type"]
    )
    assert dispatched == [_CLEAN_CALL]


def test_incomplete_template_is_versioned():
    # Agent-facing text moves prompt_version only if the registry holds it.
    assert prompts.TASK_ARG_INCOMPLETE_TMPL in prompts.tracked_leaves()


@requires_langchain
def test_luna_call_reaches_deepagents_task_schema_end_to_end():
    """Luna's call through langchain's real agent graph, against a `task` tool validated by
    deepagents' own ``TaskToolSchema``: the schema that refused all 108 relented calls."""
    from deepagents.middleware.subagents import TaskToolSchema
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import StructuredTool

    class ToolFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    ran: list = []

    def task(description: str, subagent_type: str) -> str:
        ran.append((description, subagent_type))
        return "subagent report"

    task_tool = StructuredTool.from_function(
        task, name="task", description="Delegate to a subagent.", args_schema=TaskToolSchema
    )
    call = {"name": "task", "id": "c1", "type": "tool_call", "args": _LUNA_CALL}
    stats = G.new_guard_stats()
    graph = create_agent(
        ToolFakeModel(responses=[AIMessage(content="", tool_calls=[call]), AIMessage("done")]),
        tools=[task_tool],
        middleware=[G._task_arg_guard(stats), G._tool_dispatch_guard(stats)],
    )
    out = graph.invoke({"messages": [HumanMessage(content="audit v5")]})

    assert ran == [(_LUNA_DESCRIPTION, "signal_engineer")]
    reply = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert reply.status == "success" and reply.content == "subagent report"
    assert stats["task_arg_hoists"] == 1 and stats["tools_executed"] == 1
