"""The per-delegation turn limit on analyst/biocurator subagents, and its approach warnings.

Regression cover for the runaway that motivated it: one run made 1,493 LLM calls under
recursion_limit=250 with zero fits, because deepagents gives each `task` call a FRESH step budget
and every progress-based early stop needs fitted ops to fire (session.py `should_stop`).
"""

import pytest

pytest.importorskip("langchain.agents.middleware.types")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from gemot.agent.guards import _subagent_turn_limit, _turn_notice  # noqa: E402

CAP = 20


def _state(n_ai: int, *, tool_calls: bool = True):
    """A subagent state after `n_ai` model turns, shaped like a real tool-using exchange."""
    msgs = [HumanMessage(content="investigate the residuals")]
    for i in range(n_ai):
        if tool_calls:
            msgs.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": "query_data", "args": {"sql": "select 1"}, "id": f"c{i}"}],
                )
            )
            msgs.append(ToolMessage(content="rows", tool_call_id=f"c{i}"))
        else:
            msgs.append(AIMessage(content="thinking"))
    return {"messages": msgs}


# --- the hard stop ------------------------------------------------------------------------
def test_under_the_limit_does_not_interfere():
    assert _subagent_turn_limit(CAP).after_model(_state(CAP - 1), None) is None


def test_past_the_limit_ends_the_delegation():
    assert _subagent_turn_limit(CAP).after_model(_state(CAP + 1), None)["jump_to"] == "end"


def test_the_hard_stop_never_returns_to_the_model():
    """`jump_to: model` with a tool call outstanding would leave an AIMessage carrying
    `tool_calls` followed by a non-tool message — a sequence providers reject. Ending is the
    only safe jump at a boundary that can land mid tool-call; the report it appends is never
    sent to a provider."""
    out = _subagent_turn_limit(CAP).after_model(_state(CAP + 1), None)
    assert out["jump_to"] == "end"
    assert [m.tool_calls for m in out["messages"]] == [[]]


def test_budget_is_per_delegation_not_cumulative():
    """The spec is built ONCE and reused for every `task` call, so a counter on the middleware
    instance would leak the first delegation's turns into the second and starve it."""
    mw = _subagent_turn_limit(CAP)
    assert mw.after_model(_state(CAP + 1), None)["jump_to"] == "end"
    assert mw.after_model(_state(2), None) is None


@pytest.mark.asyncio
async def test_async_hook_matches_the_sync_one():
    mw = _subagent_turn_limit(CAP)
    assert await mw.aafter_model(_state(3), None) is None
    assert await mw.aafter_model(_state(CAP + 1), None) == mw.after_model(_state(CAP + 1), None)


# --- what the lead receives from a delegation cut off at the limit -------------------------
# Without the turn notices (progress_signals off) delegations hit the cap 24 times in 15 of 35
# runs, and each report was the last text written between tool calls ("Now the high-dose
# long-observation series ..."), which leads read as a finding ("All four reads are in";
# gemot#159). _EmptyReportGuard passes such a report: it is not blank.
def test_a_cut_off_delegation_reports_itself_incomplete_with_its_last_words():
    state = _state(CAP)
    state["messages"][-2].content = "Now the high-dose long-observation series"
    (report,) = _subagent_turn_limit(CAP).after_model(state, None)["messages"]
    assert report.text.startswith("[REPORT INCOMPLETE") and f"all {CAP} of its turns" in report.text
    assert report.text.endswith("Now the high-dose long-observation series")


def test_a_cut_off_delegation_that_wrote_nothing_says_so():
    (report,) = _subagent_turn_limit(CAP).after_model(_state(CAP), None)["messages"]
    assert report.text.startswith("[REPORT INCOMPLETE")
    assert report.text.endswith("(It had written nothing.)")


def test_a_report_written_on_the_last_turn_is_not_marked():
    """A delegation whose last turn is its report has finished, not been cut off."""
    state = _state(CAP - 1)
    state["messages"].append(AIMessage(content="The residuals are white noise."))
    assert _subagent_turn_limit(CAP).after_model(state, None) == {"jump_to": "end"}


def test_the_lead_gets_the_incomplete_report_from_a_real_agent_loop():
    """Through langchain's own agent loop, with a model that never stops calling tools: the state
    update and the jump land together, and the delegation's report (deepagents forwards the LAST
    AIMessage with text, ``middleware/subagents.py``) is the one marked incomplete."""
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.tools import tool

    @tool
    def query_data(sql: str) -> str:
        """Run a query."""
        return "rows"

    class _Model(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    turns = (
        AIMessage(
            content=f"step {i}",
            tool_calls=[{"name": "query_data", "args": {"sql": "x"}, "id": f"c{i}"}],
        )
        for i in range(100)
    )
    agent = create_agent(
        _Model(messages=turns), tools=[query_data], middleware=[_subagent_turn_limit(3)]
    )
    messages = agent.invoke({"messages": [HumanMessage(content="investigate")]})["messages"]
    report = next(m.text for m in reversed(messages) if isinstance(m, AIMessage) and m.text)
    assert sum(isinstance(m, AIMessage) for m in messages) == 4  # three turns and the report
    assert report.startswith("[REPORT INCOMPLETE") and report.endswith("step 2")


# --- the approach warnings ----------------------------------------------------------------
@pytest.mark.parametrize(
    "used,expect",
    [
        (0, None),
        (5, None),
        (9, None),  # 95% of real delegations finish in here
        (10, "NOTE"),
        (14, "NOTE"),
        (15, "WARNING"),
        (18, "WARNING"),
        (19, "FINAL TURN"),
        (20, "FINAL TURN"),
    ],
)
def test_notice_escalates_at_ten_five_and_one(used, expect):
    got = _turn_notice(used, CAP)
    assert (got is None) if expect is None else (expect in got)


def test_notice_never_reports_a_negative_budget():
    assert "-1" not in (_turn_notice(CAP + 1, CAP) or "")


def test_notices_reach_the_model_via_the_request_not_the_state():
    """Warnings go into the outgoing REQUEST, never into state — appending them to state would
    hit the same invalid-sequence problem as the hard stop, and would accumulate."""
    seen = {}

    class _Req:
        messages = _state(15)["messages"]
        system_prompt = "you are a signal engineer"

        def override(self, **kw):
            seen.update(kw)
            return self

    mw = _subagent_turn_limit(CAP)
    mw.wrap_model_call(_Req(), lambda r: r)
    assert "system_prompt" not in seen, "the notice must not touch the cached prefix"
    assert "WARNING" in seen["messages"][-1].content


def test_the_notice_never_mutates_the_cached_prefix():
    """THE cache bug this guards: the notice carries a countdown, so putting it in the system
    prompt made message[0] a different string every turn — measured on the 2026-08-14 A/B as a
    0.86-0.98 -> 0.72 cache-hit collapse on 361/361 calls past the threshold. Appended at the
    END, everything before it stays cacheable."""
    from langchain_core.messages import HumanMessage

    seen = {}
    base_msgs = _state(15)["messages"]

    class _Req:
        messages = base_msgs
        system_prompt = "you are a signal engineer"

        def override(self, **kw):
            seen.update(kw)
            return self

    _subagent_turn_limit(CAP).wrap_model_call(_Req(), lambda r: r)
    out = seen["messages"]
    assert out[: len(base_msgs)] == list(base_msgs), "the prefix must be byte-identical"
    assert len(out) == len(base_msgs) + 1
    assert isinstance(out[-1], HumanMessage)


def test_no_notice_leaves_the_request_untouched():
    class _Req:
        messages = _state(2)["messages"]
        system_prompt = "unchanged"

        def override(self, **kw):  # pragma: no cover - must not be called
            raise AssertionError("request was overridden with no notice due")

    assert (
        _subagent_turn_limit(CAP).wrap_model_call(_Req(), lambda r: r).system_prompt == "unchanged"
    )


# --- wiring -------------------------------------------------------------------------------
def test_wired_into_every_subagent_spec():
    from gemot.agent.biocurator import _build_biocurator_subagent
    from gemot.agent.subagents import _build_subagents

    specs = _build_subagents({}) + [_build_biocurator_subagent([], [])]
    assert len(specs) == 4
    for spec in specs:
        names = [type(m).__name__ for m in spec.get("middleware") or []]
        assert "_SubagentTurnLimit" in names, f"{spec['name']} is unbounded"
