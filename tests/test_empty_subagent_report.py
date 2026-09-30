"""Tests for the empty-subagent-report guard (gemot.agent.guards).

An analyst subagent can end its run with an empty final completion (finish_reason=stop, no
content). deepagents' ``task`` tool extracts that last AI message's content and returns ``""`` to
the lead, silently discarding the whole subagent run (confirmed live on a blasi run: signal_engineer
ran 27 turns of residual analysis, then stopped with empty content). The guard catches the blank
``task`` return, re-dispatches the subagent ONCE with a corrective nudge, and — if the retry is also
blank — substitutes a visible marker so the loss is legible instead of a silent ``""``.

The pure detection helpers are tested with no third-party dependency, consistent with how
``_unexpected_task_kwargs`` is tested. The middleware itself is exercised directly (its
``wrap_tool_call``/``awrap_tool_call`` methods against a scripted handler that returns the
``Command`` shape the real ``task`` tool produces) — skipped where langchain is absent, as it is
only an examples-harness dependency (mirrors tests/test_biocurator_input.py).
"""

from __future__ import annotations

import asyncio

from gemot import agent


# --------------------------------------------------------------------- _report_text (pure)
def test_report_text_passes_through_a_plain_string():
    assert agent._report_text("a report") == "a report"
    assert agent._report_text("") == ""


def test_report_text_treats_none_as_empty():
    assert agent._report_text(None) == ""


def test_report_text_flattens_content_blocks():
    # langchain also allows list content: bare strings and {"type": "text", "text": ...} dicts.
    content = ["hello ", {"type": "text", "text": "world"}, {"type": "image", "url": "x"}]
    assert agent._report_text(content) == "hello world"


# --------------------------------------------------------------------- _is_blank_report (pure)
def test_is_blank_report_flags_empty_and_whitespace():
    assert agent._is_blank_report("") is True
    assert agent._is_blank_report("   ") is True
    assert agent._is_blank_report("\n\t  \n") is True
    assert agent._is_blank_report(None) is True
    assert agent._is_blank_report([]) is True
    assert agent._is_blank_report([{"type": "text", "text": "  "}]) is True


def test_is_blank_report_accepts_a_real_report():
    assert agent._is_blank_report("Residuals show structure at t>5.") is False
    assert agent._is_blank_report(["  ", "x"]) is False
    assert agent._is_blank_report([{"type": "text", "text": "found it"}]) is False


# --------------------------------------------------------------------- _empty_report_guard middleware
# Building the middleware imports langchain (examples-harness only), so guard these tests.
from conftest import requires_langchain  # noqa: E402

_MARKER = agent.SUBAGENT_EMPTY_REPORT_MARKER
_NUDGE = agent.SUBAGENT_EMPTY_REPORT_NUDGE


def _fresh_stats() -> dict:
    return {"empty_subagent_reports": 0, "empty_subagent_retries": 0, "empty_subagent_markers": 0}


class _FakeRequest:
    """Mirrors the ToolCallRequest surface the guard uses: `.tool_call` and immutable `.override`."""

    def __init__(self, name, args, id_="c1"):
        self.tool_call = {"name": name, "args": args, "id": id_, "type": "tool_call"}

    def override(self, *, tool_call):
        new = _FakeRequest.__new__(_FakeRequest)
        new.tool_call = tool_call
        return new


class _ScriptedHandler:
    """A handler that returns successive scripted results and records the description it saw."""

    def __init__(self, *results):
        self._results = list(results)
        self.descriptions: list = []

    def __call__(self, request):
        self.descriptions.append(request.tool_call["args"].get("description"))
        return self._results.pop(0)


def _cmd(content, *, tool_call_id="c1", **extra):
    """A `Command` the shape deepagents' `task` tool returns: one ToolMessage, plus any extra
    state-update keys the subagent propagated."""
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    return Command(update={**extra, "messages": [ToolMessage(content, tool_call_id=tool_call_id)]})


def _content_of(result):
    """The ToolMessage content inside a `Command` (or a bare ToolMessage)."""
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    if isinstance(result, Command):
        return next(m.content for m in result.update["messages"] if isinstance(m, ToolMessage))
    return result.content


@requires_langchain
def test_empty_report_is_re_dispatched_and_recovers():
    stats = _fresh_stats()
    mw = agent._empty_report_guard(stats)
    handler = _ScriptedHandler(_cmd(""), _cmd("Residuals structured at late times."))
    req = _FakeRequest(
        "task", {"description": "analyse residuals", "subagent_type": "signal_engineer"}
    )

    out = mw.wrap_tool_call(req, handler)

    assert _content_of(out) == "Residuals structured at late times."  # the recovered report
    assert stats == {
        "empty_subagent_reports": 1,
        "empty_subagent_retries": 1,
        "empty_subagent_markers": 0,
    }
    # the re-dispatch carried the corrective nudge appended to the original description
    assert len(handler.descriptions) == 2
    assert handler.descriptions[0] == "analyse residuals"
    assert handler.descriptions[1].startswith("analyse residuals")
    assert _NUDGE in handler.descriptions[1]


@requires_langchain
def test_retry_also_blank_gets_the_visible_marker():
    stats = _fresh_stats()
    mw = agent._empty_report_guard(stats)
    handler = _ScriptedHandler(_cmd("   "), _cmd("\n\n"))  # blank both times
    req = _FakeRequest("task", {"description": "d", "subagent_type": "mathematician"})

    out = mw.wrap_tool_call(req, handler)

    assert _content_of(out) == _MARKER
    assert stats == {
        "empty_subagent_reports": 1,
        "empty_subagent_retries": 1,
        "empty_subagent_markers": 1,
    }


@requires_langchain
def test_no_retry_budget_substitutes_the_marker_immediately():
    stats = _fresh_stats()
    mw = agent._empty_report_guard(stats, retries=0)
    handler = _ScriptedHandler(_cmd(""))
    req = _FakeRequest("task", {"description": "d", "subagent_type": "signal_engineer"})

    out = mw.wrap_tool_call(req, handler)

    assert _content_of(out) == _MARKER
    assert len(handler.descriptions) == 1  # never re-dispatched
    assert stats == {
        "empty_subagent_reports": 1,
        "empty_subagent_retries": 0,
        "empty_subagent_markers": 1,
    }


@requires_langchain
def test_a_real_report_passes_through_untouched():
    stats = _fresh_stats()
    mw = agent._empty_report_guard(stats)
    result = _cmd("A full analyst report.")
    handler = _ScriptedHandler(result)
    req = _FakeRequest("task", {"description": "d", "subagent_type": "systems_scientist"})

    out = mw.wrap_tool_call(req, handler)

    assert out is result  # identical object, no rewrap
    assert len(handler.descriptions) == 1  # never re-dispatched
    assert stats == _fresh_stats()


@requires_langchain
def test_a_blank_non_task_call_is_none_of_this_guards_business():
    # An empty result from some OTHER tool must pass through — this guard only acts on `task`.
    stats = _fresh_stats()
    mw = agent._empty_report_guard(stats)
    result = _cmd("")
    handler = _ScriptedHandler(result)
    req = _FakeRequest("get_errors", {})

    out = mw.wrap_tool_call(req, handler)

    assert out is result
    assert stats == _fresh_stats()


@requires_langchain
def test_a_bare_toolmessage_result_is_handled_too():
    # The task tool normally returns a Command, but the guard also copes with a bare ToolMessage.
    from langchain_core.messages import ToolMessage

    stats = _fresh_stats()
    mw = agent._empty_report_guard(stats, retries=0)
    handler = _ScriptedHandler(ToolMessage("", tool_call_id="c1"))
    req = _FakeRequest("task", {"description": "d", "subagent_type": "signal_engineer"})

    out = mw.wrap_tool_call(req, handler)

    assert isinstance(out, ToolMessage)
    assert out.content == _MARKER
    assert out.tool_call_id == "c1"
    assert stats["empty_subagent_markers"] == 1


@requires_langchain
def test_a_plain_string_result_is_left_alone():
    # The task tool returns a plain str for an unknown subagent_type — not a report, so ignore it.
    stats = _fresh_stats()
    mw = agent._empty_report_guard(stats)
    handler = _ScriptedHandler("We cannot invoke subagent x because it does not exist")
    req = _FakeRequest("task", {"description": "d", "subagent_type": "x"})

    out = mw.wrap_tool_call(req, handler)

    assert out == "We cannot invoke subagent x because it does not exist"
    assert stats == _fresh_stats()


@requires_langchain
def test_marker_substitution_preserves_the_rest_of_the_command_state():
    # A blank Command may still carry other state updates (e.g. files the subagent wrote) and the
    # ToolMessage's tool_call_id; both must survive the marker swap.
    stats = _fresh_stats()
    mw = agent._empty_report_guard(stats, retries=0)
    handler = _ScriptedHandler(_cmd("", tool_call_id="call_42", files={"a.py": "x"}))
    req = _FakeRequest(
        "task", {"description": "d", "subagent_type": "signal_engineer"}, id_="call_42"
    )

    out = mw.wrap_tool_call(req, handler)

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    assert isinstance(out, Command)
    assert out.update["files"] == {"a.py": "x"}  # unrelated state preserved
    tm = next(m for m in out.update["messages"] if isinstance(m, ToolMessage))
    assert tm.content == _MARKER
    assert tm.tool_call_id == "call_42"  # id preserved for pairing


@requires_langchain
def test_async_paths_recover_then_marker():
    stats = _fresh_stats()
    mw = agent._empty_report_guard(stats)

    # recovers on the async retry
    sync = _ScriptedHandler(_cmd(""), _cmd("recovered async report"))

    async def ahandler(request):
        return sync(request)

    req = _FakeRequest("task", {"description": "d", "subagent_type": "signal_engineer"})
    out = asyncio.run(mw.awrap_tool_call(req, ahandler))
    assert _content_of(out) == "recovered async report"
    assert _NUDGE in sync.descriptions[1]
    assert stats == {
        "empty_subagent_reports": 1,
        "empty_subagent_retries": 1,
        "empty_subagent_markers": 0,
    }

    # and the async retry that stays blank falls back to the marker
    stats2 = _fresh_stats()
    mw2 = agent._empty_report_guard(stats2)
    sync2 = _ScriptedHandler(_cmd(""), _cmd(""))

    async def ahandler2(request):
        return sync2(request)

    out2 = asyncio.run(mw2.awrap_tool_call(req, ahandler2))
    assert _content_of(out2) == _MARKER
    assert stats2 == {
        "empty_subagent_reports": 1,
        "empty_subagent_retries": 1,
        "empty_subagent_markers": 1,
    }


@requires_langchain
def test_end_to_end_against_langchains_real_agent_graph():
    """The full loop through langchain's real ToolNode/middleware stack: the `task` tool returns a
    Command whose ToolMessage content is empty; the guard re-dispatches once (the tool then returns
    a real report) so the lead's message history carries the recovered report, never the "".

    Mirrors tests/test_biocurator_input.py's end-to-end test: the guard sits INNERMOST, exactly as
    run_deepagents installs it, so a re-dispatch does not double-count tools_executed."""
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.types import Command

    class ToolFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    seen_descriptions: list = []
    # the `task` tool: first call returns an empty report (the bug), the re-dispatch a real one.
    # It returns a Command carrying a ToolMessage — the exact shape deepagents' real task tool does.
    # (tool_call_id is hardcoded to the single scripted call id 'c1'; the return type is left
    # unannotated so langchain's schema builder doesn't have to resolve Command under the module's
    # `from __future__ import annotations`.)
    reports = ["", "Residual structure at late times; add a delay."]

    @tool
    def task(description: str, subagent_type: str):
        """Delegate to a subagent."""
        seen_descriptions.append(description)
        content = reports.pop(0)
        return Command(update={"messages": [ToolMessage(content, tool_call_id="c1")]})

    msgs = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "id": "c1",
                    "type": "tool_call",
                    "args": {
                        "description": "analyse residuals",
                        "subagent_type": "signal_engineer",
                    },
                }
            ],
        ),
        AIMessage(content="done"),
    ]
    stats = _fresh_stats()
    graph = create_agent(
        ToolFakeModel(responses=msgs), tools=[task], middleware=[agent._empty_report_guard(stats)]
    )
    out = graph.invoke(
        {"messages": [HumanMessage(content="build a model")]}, config={"recursion_limit": 30}
    )

    # the subagent was re-dispatched once, with the nudge appended
    assert len(seen_descriptions) == 2
    assert _NUDGE in seen_descriptions[1]
    # the recovered report reached the message history; the "" never did
    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.content for m in tool_msgs] == ["Residual structure at late times; add a delay."]
    assert stats == {
        "empty_subagent_reports": 1,
        "empty_subagent_retries": 1,
        "empty_subagent_markers": 0,
    }
