"""Tests for the deepagents driver's tool-dispatch guard (petab_harness.agent).

Covers issue #63: a class of runs (first seen on the blasi zero-shot build) that proposed
tool calls every step but executed ZERO tools, terminating early with nothing built. The guard
(1) de-duplicates identical tool calls within a batch, (2) pushes back on idle no-tool turns and
labels a run that never fires a tool as ``no_tools_executed``.

The pure de-dup helpers are tested with no third-party dependency. The middleware itself is
exercised end-to-end against langchain's real ``create_agent`` graph driven by a scripted fake
chat model (no network, no MCP server, no deepagents) — skipped where langchain is absent, as it
is not a dependency of the petab-mcp package (only of the examples harness).
"""

from __future__ import annotations



from petab_harness import agent, scenario




# --------------------------------------------------------------------- pure de-dup helpers
def test_tool_call_key_ignores_id_and_arg_order():
    a = {"name": "make_changes", "args": {"antimony": "x", "n_starts": 5}, "id": "call_1"}
    b = {"name": "make_changes", "args": {"n_starts": 5, "antimony": "x"}, "id": "call_2"}
    assert agent._tool_call_key(a) == agent._tool_call_key(b)


def test_tool_call_key_distinguishes_name_and_args():
    base = {"name": "get_fit_summary", "args": {}}
    assert agent._tool_call_key(base) != agent._tool_call_key({"name": "get_errors", "args": {}})
    assert agent._tool_call_key({"name": "f", "args": {"x": 1}}) != \
        agent._tool_call_key({"name": "f", "args": {"x": 2}})


def test_tool_call_key_handles_non_serialisable_args():
    # Must not raise on args json can't encode (falls back to repr).
    key = agent._tool_call_key({"name": "f", "args": {1, 2, 3}})
    assert key.startswith("f\x00")


def test_dedupe_drops_exact_duplicates_keeping_first():
    calls = [
        {"name": "a", "args": {"x": 1}, "id": "1"},
        {"name": "a", "args": {"x": 1}, "id": "2"},   # dup of #1
        {"name": "a", "args": {"x": 2}, "id": "3"},   # different args
        {"name": "b", "args": {"x": 1}, "id": "4"},   # different name
    ]
    kept, dropped = agent._dedupe_tool_calls(calls)
    assert dropped == 1
    assert [c["id"] for c in kept] == ["1", "3", "4"]


def test_dedupe_empty_and_none():
    assert agent._dedupe_tool_calls([]) == ([], 0)
    assert agent._dedupe_tool_calls(None) == ([], 0)


def test_dedupe_collapses_a_full_batch_of_identical_calls():
    calls = [{"name": "describe_problem", "args": {}, "id": f"c{i}"} for i in range(5)]
    kept, dropped = agent._dedupe_tool_calls(calls)
    assert dropped == 4
    assert [c["id"] for c in kept] == ["c0"]


# --------------------------------------------------------------------- raw (additional_kwargs)
def test_dedupe_raw_keeps_first_by_name_and_arguments():
    raw = [
        {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "2", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "3", "type": "function", "function": {"name": "f", "arguments": '{"x":1}'}},
    ]
    kept = agent._dedupe_raw_tool_calls(raw)
    assert [c["id"] for c in kept] == ["1", "3"]


def test_dedupe_raw_collapses_a_repeated_id_duplicate():
    # A provider that repeats the SAME call with the SAME id must not leave a duplicate behind
    # (the bug an id-membership filter would miss).
    raw = [
        {"id": "dup", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "dup", "type": "function", "function": {"name": "f", "arguments": "{}"}},
    ]
    kept = agent._dedupe_raw_tool_calls(raw)
    assert len(kept) == 1
    assert kept[0]["id"] == "dup"


def test_dedupe_raw_empty_and_none():
    assert agent._dedupe_raw_tool_calls([]) == []
    assert agent._dedupe_raw_tool_calls(None) == []


# --------------------------------------------------------------------- middleware end-to-end
# langchain is an examples-harness dependency, NOT a dependency of the petab-mcp package, so the
# integration tests below skip cleanly where it is absent; the pure-helper tests above always run.
from conftest import requires_langchain  # noqa: E402


def _fixtures():
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    class ToolFakeModel(FakeMessagesListChatModel):
        """A scripted fake model that accepts bind_tools (FakeMessagesListChatModel does not)."""

        def bind_tools(self, tools, **kwargs):
            return self

    @tool
    def describe_problem() -> str:
        """Orient: whether a model exists."""
        return "model_authored: false"

    def run(scripted_msgs, *, max_idle_steps=3, return_messages=False):
        stats = {"tools_executed": 0, "dropped_dup_tool_calls": 0,
                 "idle_nudges": 0, "gave_up_no_tools": False}
        guard = agent._tool_dispatch_guard(stats, max_idle_steps=max_idle_steps)
        graph = create_agent(ToolFakeModel(responses=scripted_msgs),
                             tools=[describe_problem], middleware=[guard])
        out = graph.invoke({"messages": [HumanMessage(content="build a model")]},
                           config={"recursion_limit": 50})
        return (stats, out["messages"]) if return_messages else stats

    return run, AIMessage


def _call(id_="c1"):
    _, AIMessage = _fixtures()
    return AIMessage(content="", tool_calls=[
        {"name": "describe_problem", "args": {}, "id": id_, "type": "tool_call"}])


@requires_langchain
def test_duplicate_tool_calls_in_a_batch_execute_once():
    run, AIMessage = _fixtures()
    dup_batch = AIMessage(content="", tool_calls=[
        {"name": "describe_problem", "args": {}, "id": "c1", "type": "tool_call"},
        {"name": "describe_problem", "args": {}, "id": "c2", "type": "tool_call"},
    ])
    stats = run([dup_batch, AIMessage(content="done — fit adequate")])
    assert stats["dropped_dup_tool_calls"] == 1
    assert stats["tools_executed"] == 1
    assert stats["gave_up_no_tools"] is False


@requires_langchain
def test_idle_run_that_never_fires_a_tool_is_pushed_back_then_given_up():
    run, AIMessage = _fixtures()
    text_only = [AIMessage(content=f"I would build a model... ({i})") for i in range(10)]
    stats = run(text_only, max_idle_steps=3)
    assert stats["tools_executed"] == 0
    assert stats["idle_nudges"] == 3
    assert stats["gave_up_no_tools"] is True


@requires_langchain
def test_nudge_rescues_a_run_that_starts_by_stalling():
    # Two text-only turns, then the model finally emits a real tool call and finishes: the
    # push-back keeps the loop alive so the proposed tool call actually fires.
    run, AIMessage = _fixtures()
    stats = run([
        AIMessage(content="let me think about the mechanism"),
        AIMessage(content="still reasoning, no action yet"),
        _call(),
        AIMessage(content="fit adequate, finishing"),
    ], max_idle_steps=3)
    assert stats["tools_executed"] == 1
    assert stats["idle_nudges"] == 2
    assert stats["gave_up_no_tools"] is False


@requires_langchain
def test_normal_finish_after_a_tool_is_not_disturbed():
    # A no-tool assistant turn AFTER a tool has fired is a genuine finish — never nudged.
    run, AIMessage = _fixtures()
    stats = run([_call(), AIMessage(content="done, reporting the final fit")])
    assert stats["tools_executed"] == 1
    assert stats["idle_nudges"] == 0
    assert stats["gave_up_no_tools"] is False


@requires_langchain
def test_dedupe_also_collapses_the_raw_additional_kwargs_copy():
    # A duplicate batch whose raw additional_kwargs copy reuses the SAME id must be collapsed
    # there too — not just in .tool_calls — so a stale duplicate cannot resurface.
    run, AIMessage = _fixtures()
    dup_batch = AIMessage(
        content="",
        tool_calls=[
            {"name": "describe_problem", "args": {}, "id": "dup", "type": "tool_call"},
            {"name": "describe_problem", "args": {}, "id": "dup", "type": "tool_call"},
        ],
        additional_kwargs={"tool_calls": [
            {"id": "dup", "type": "function",
             "function": {"name": "describe_problem", "arguments": "{}"}},
            {"id": "dup", "type": "function",
             "function": {"name": "describe_problem", "arguments": "{}"}},
        ]},
    )
    stats, messages = run([dup_batch, AIMessage(content="done")], return_messages=True)
    assert stats["tools_executed"] == 1
    ai = next(m for m in messages if getattr(m, "tool_calls", None))
    assert len(ai.tool_calls) == 1
    assert len(ai.additional_kwargs.get("tool_calls", [])) == 1


# -- the lead's curated tool surface (issue #113 follow-up) -------------------
def test_lead_does_not_carry_get_fit_summary_but_the_analysts_do():
    """`make_changes` returns the whole fit block — metrics, bic, the noise-sigma verdict, the
    residual_structure verdict — so for the LEAD, the only agent that fits, get_fit_summary reads
    back what it was just handed (9 calls against 451 fits in the measured sweep). It is dropped
    from its list, since every tool's schema is re-sent every turn.

    It must NOT be removed from the server: the analysts never fit, so it is their only score
    read, and the harness's own finish gate and final scoring go through it too.
    """
    mod = agent
    assert "get_fit_summary" not in mod.ORCHESTRATOR_TOOLS
    assert "make_changes" in mod.ORCHESTRATOR_TOOLS
    # still reachable for everyone who cannot get the score from a fit result
    assert "get_fit_summary" in mod.MATHEMATICIAN_TOOLS
    assert "get_fit_summary" in mod.SIGNAL_ENGINEER_TOOLS
    assert "get_fit_summary" in mod.ALLOWED_TOOLS
    # and the panel still has a description for it (ALLOWED_TOOLS is what the panel ever sees)
    assert "get_fit_summary" in mod.TOOL_DESCRIPTIONS


def test_the_lead_is_never_told_to_call_a_tool_it_does_not_have():
    """A prompt that points the lead at a tool missing from its list burns a step on a call that
    cannot dispatch. Guards the get_fit_summary removal above against the prompt text drifting
    back — the lead's system prompt lives in prompts.py, separately from its tool list."""
    mod = agent
    lead_prompt = scenario.DEFAULT_SYSTEM_PROMPT
    assert "get_fit_diagnosis" in lead_prompt, "sanity: this IS the lead's tool-naming prompt"

    # The general rule: no tool the panel has but the LEAD does not may be named in the lead's
    # prompt. Delegated reads (get_measurements, query_data, get_fitting_log, list_data_tables)
    # and get_fit_summary belong to the analysts / the harness's own polls, so the prompt must
    # describe them as work to DELEGATE, never as a call for the lead to make.
    not_the_leads = sorted(t for t in mod.ALLOWED_TOOLS if t not in mod.ORCHESTRATOR_TOOLS)
    assert not_the_leads, "sanity: some tools are analyst-only"
    named = [t for t in not_the_leads if t in lead_prompt]
    assert named == [], f"lead prompt names tools it cannot call: {named}"

    # the panel-facing override for the editor is lead-facing too
    assert "get_fit_summary" not in mod.TOOL_DESCRIPTIONS["make_changes"]
    # (the harness's OWN internal polls still call get_fit_summary — see _current_fit — which is
    # why this checks the lead-facing text rather than the module source.)


def test_get_parameters_description_does_not_contradict_the_parameter_table():
    """The lead-facing description REPLACES the server docstring, so a wrong claim there is what
    the model believes. It used to state that sigma_* and *_init rows are fixed (estimate=0):
    initial conditions have been estimated by DEFAULT since #90 (only an IC of exactly 0 or 1
    stays fixed), and an estimated noise sigma is the whole premise of the noise-sigma verdict.
    """
    mod = agent
    desc = mod.TOOL_DESCRIPTIONS["get_parameters"]

    # it must not assert the fixed-ness that reconcile contradicts
    assert "estimate=0" not in desc
    assert "estimated by DEFAULT" in desc or "estimated by default" in desc
    # and it must name the two annotations the tool exists to surface
    assert "role" in desc and "status" in desc
    for role in ("kinetic", "noise_sigma", "observable_scaling", "initial_condition"):
        assert role in desc, role
