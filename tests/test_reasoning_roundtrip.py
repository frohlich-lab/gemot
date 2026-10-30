"""Reasoning goes back to the provider only where it is needed: the tool-use turn in progress, for
the models that take it back.

langchain_openai drops OpenRouter's ``reasoning_details`` both ways, so every request after the
first sent the lead's tool-call turns without the thinking that produced them, and Anthropic ran
the next call without thinking: Claude Haiku 4.5 reasoned on 0 of 1,466 calls that followed a tool
result. Haiku also needs Anthropic's interleaved-thinking beta to think between tool calls at all.

Putting EVERY turn's reasoning back for EVERY model was gemot#76: GLM 5.3 Flash at effort max
re-sent up to 75k tokens of old reasoning per turn, 25-37% of every prompt.
"""

from __future__ import annotations

import types

import pytest

from gemot import models
from gemot.agent import driver

_DETAILS = [{"type": "reasoning.text", "text": "check the readout", "signature": "sig", "index": 0}]
_HAIKU = "anthropic/claude-haiku-5.5"


def test_haiku_asks_for_interleaved_thinking():
    assert models.request_headers(_HAIKU) == {"x-anthropic-beta": "interleaved-thinking-2025-05-14"}
    assert models.request_headers("z-ai/glm-5.3-flash") is None


def test_a_responses_reasoning_rides_its_message():
    message = types.SimpleNamespace(additional_kwargs={})
    result = types.SimpleNamespace(generations=[types.SimpleNamespace(message=message)])
    response = {"choices": [{"message": {"content": "", "reasoning_details": _DETAILS}}]}
    driver._keep_reasoning(response, result)
    assert message.additional_kwargs["reasoning_details"] == _DETAILS


def test_every_anthropic_model_and_no_other_takes_reasoning_back():
    """An Anthropic alias added without its entry would run every call after a tool result
    without thinking; any other model opted in silently would re-send what gemot#76 removed."""
    for model_id in models.MODELS.values():
        assert models.returns_reasoning(model_id) is model_id.startswith("anthropic/"), model_id


def _turn(text: str, call: str | None):
    from langchain_core.messages import AIMessage

    details = [{**_DETAILS[0], "text": text}]
    calls = [{"name": "describe_problem", "args": {}, "id": call}] if call else []
    return AIMessage(content="", tool_calls=calls, additional_kwargs={"reasoning_details": details})


def _returned(model: str, messages) -> list:
    """The reasoning text on each assistant message of the request ``model`` would be sent."""
    llm = driver._chat_model_class()(
        name="ChatOpenAI", model=model, api_key="x", base_url="http://x"
    )
    payload = llm._get_request_payload(messages)
    assert all(
        "reasoning_details" not in m for m in payload["messages"] if m["role"] != "assistant"
    )
    return [
        (m.get("reasoning_details") or [{}])[0].get("text")
        for m in payload["messages"]
        if m["role"] == "assistant"
    ]


def test_the_turn_in_progress_carries_its_reasoning_back():
    pytest.importorskip("langchain_openai", reason="agent stack not installed")
    from langchain_core.messages import HumanMessage, ToolMessage

    messages = [
        HumanMessage("build"),
        _turn("one", "c1"),
        ToolMessage("ok", tool_call_id="c1"),
        _turn("two", "c2"),
        ToolMessage("ok", tool_call_id="c2"),
    ]
    assert _returned(_HAIKU, messages) == ["one", "two"]


def test_a_model_not_opted_in_gets_no_reasoning_back():
    """gemot#76: the GLM sweep's prompts were a quarter to a third old reasoning."""
    pytest.importorskip("langchain_openai", reason="agent stack not installed")
    from langchain_core.messages import HumanMessage, ToolMessage

    messages = [HumanMessage("build"), _turn("one", "c1"), ToolMessage("ok", tool_call_id="c1")]
    assert _returned("z-ai/glm-5.3-flash", messages) == [None]


def test_a_plain_user_message_ends_the_turn():
    """A nudge after a text-only turn starts a new one: what came before it is not sent."""
    pytest.importorskip("langchain_openai", reason="agent stack not installed")
    from langchain_core.messages import HumanMessage, ToolMessage

    messages = [
        HumanMessage("build"),
        _turn("one", "c1"),
        ToolMessage("ok", tool_call_id="c1"),
        _turn("two", None),
        HumanMessage("you stopped without finishing"),
        _turn("three", "c3"),
        ToolMessage("ok", tool_call_id="c3"),
    ]
    assert _returned(_HAIKU, messages) == [None, None, "three"]


def test_a_notice_after_tool_results_does_not_end_the_turn():
    """The subagent turn limit appends its notice right after the tool results, where the
    provider merges it into the message returning them; ending the turn there would strip an
    analyst's thinking for its last calls."""
    pytest.importorskip("langchain_openai", reason="agent stack not installed")
    from langchain_core.messages import HumanMessage, ToolMessage

    messages = [
        HumanMessage("analyse"),
        _turn("one", "c1"),
        ToolMessage("ok", tool_call_id="c1"),
        HumanMessage("5 turns left"),
    ]
    assert _returned(_HAIKU, messages) == ["one"]


def test_the_thinking_hand_off_sends_no_reasoning_back():
    """agent.thinking's hand-off ends on its own instruction after the runaway's thinking as
    text, so nothing earlier rides along (it used to carry every turn's)."""
    pytest.importorskip("langchain_openai", reason="agent stack not installed")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from gemot.agent import thinking

    messages = [
        HumanMessage("build"),
        _turn("one", "c1"),
        ToolMessage("ok", tool_call_id="c1"),
        AIMessage(content="the runaway's thinking"),
        HumanMessage(thinking.HANDOFF),
    ]
    assert _returned(_HAIKU, messages) == [None, None]
