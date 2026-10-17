"""Each turn's reasoning goes back to the provider with that turn.

langchain_openai drops OpenRouter's ``reasoning_details`` both ways, so every request after the
first sent the lead's tool-call turns without the thinking that produced them, and Anthropic ran
the next call without thinking: Claude Haiku 4.5 reasoned on 0 of 1,466 calls that followed a tool
result. Haiku also needs Anthropic's interleaved-thinking beta to think between tool calls at all.
"""

from __future__ import annotations

import types

import pytest

from gemot import models
from gemot.agent import driver

_DETAILS = [{"type": "reasoning.text", "text": "check the readout", "signature": "sig", "index": 0}]


def test_haiku_asks_for_interleaved_thinking():
    assert models.request_headers("anthropic/claude-haiku-4.5") == {
        "x-anthropic-beta": "interleaved-thinking-2025-05-14"
    }
    assert models.request_headers("z-ai/glm-5.3-flash") is None


def test_a_responses_reasoning_rides_its_message():
    message = types.SimpleNamespace(additional_kwargs={})
    result = types.SimpleNamespace(generations=[types.SimpleNamespace(message=message)])
    response = {"choices": [{"message": {"content": "", "reasoning_details": _DETAILS}}]}
    driver._keep_reasoning(response, result)
    assert message.additional_kwargs["reasoning_details"] == _DETAILS


def test_the_request_carries_each_turns_reasoning_back():
    pytest.importorskip("langchain_openai", reason="agent stack not installed")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    llm = driver._chat_model_class()(
        name="ChatOpenAI", model="anthropic/claude-haiku-4.5", api_key="x", base_url="http://x"
    )
    turn = AIMessage(
        content="",
        tool_calls=[{"name": "describe_problem", "args": {}, "id": "c1"}],
        additional_kwargs={"reasoning_details": _DETAILS},
    )
    payload = llm._get_request_payload(
        [HumanMessage("build"), turn, ToolMessage("ok", tool_call_id="c1")]
    )
    assistant = [m for m in payload["messages"] if m["role"] == "assistant"]
    assert assistant[0]["reasoning_details"] == _DETAILS
    assert all("reasoning_details" not in m for m in payload["messages"] if m["role"] != "assistant")
