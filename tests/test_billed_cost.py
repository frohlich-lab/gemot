"""What the driver's ``ChatOpenAI`` subclass puts on the wire, and what comes back off it, over a
mocked HTTP transport: OpenRouter's ``usage.cost`` through to ``token_usage.cost_usd`` (non-streamed
and streamed, which langchain_openai shapes differently) and the cache breakpoints of
``models._CACHING``. test_token_accounting covers the callback on hand-built records and
test_model_routing the tables; this file is the assembly, where a langchain rename shows up."""

from __future__ import annotations

import importlib
import json

import pytest

pytest.importorskip("langchain_openai", reason="harness-only dependency")

import openai  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from petab_harness import models  # noqa: E402
from petab_harness.agent.driver import _chat_model_class, _extra_body  # noqa: E402
from petab_harness.agent.token_usage import _token_accountant  # noqa: E402

# openai builds its client on `httpx` up to 2.x and on `httpx2` from 3.x, and rejects a response
# produced by the other flavour's transport. Take the flavour from openai; do not import it here.
_httpx = importlib.import_module(openai.DefaultAsyncHttpxClient.__base__.__module__.split(".")[0])

MODEL = "xiaomi/mimo-v2.5"
# What OpenRouter returns with `usage: {include: true}`: the OpenAI fields plus `cost`.
USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 50,
    "total_tokens": 1050,
    "prompt_tokens_details": {"cached_tokens": 800},
    "cost": 0.00123,
    "cost_details": {"upstream_inference_cost": None},
    "is_byok": False,
}


def _completion(request: _httpx.Request) -> _httpx.Response:
    assert json.loads(request.content)["usage"] == {"include": True}, "the flag must be on the wire"
    return _httpx.Response(
        200,
        json={
            "id": "gen-1",
            "provider": "Novita",
            "model": MODEL,
            "object": "chat.completion",
            "created": 1,
            "usage": USAGE,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
        },
    )


def _stream(request: _httpx.Request) -> _httpx.Response:
    body = json.loads(request.content)
    assert body["stream"] is True and body["usage"] == {"include": True}

    def event(delta=None, finish=None, usage=None):
        d = {
            "id": "gen-2",
            "provider": "Novita",
            "model": MODEL,
            "created": 1,
            "object": "chat.completion.chunk",
            "choices": []
            if delta is None
            else [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage is not None:
            d["usage"] = usage
        return f"data: {json.dumps(d)}\n\n"

    # OpenRouter puts usage on a final, choice-less chunk.
    text = (
        event({"role": "assistant", "content": "h"})
        + event({"content": "i"}, finish="stop")
        + event(usage=USAGE)
        + "data: [DONE]\n\n"
    )
    return _httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=text.encode()
    )


def _llm(handler, model=MODEL):
    return _chat_model_class()(
        name="ChatOpenAI",
        model=model,
        base_url="https://openrouter.ai/api/v1",
        api_key="x",
        stream_usage=True,
        extra_body=_extra_body(model),
        http_async_client=openai.DefaultAsyncHttpxClient(transport=_httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_non_streamed_response_records_the_billed_cost():
    acc = _token_accountant()
    await _llm(_completion).ainvoke([HumanMessage("x")], config={"callbacks": [acc]})
    t = acc.totals()
    assert t["cost_usd"] == pytest.approx(0.00123)
    assert t["n_llm_calls"] == 1 and t["n_uncosted_calls"] == 0
    assert t["cache_read_tokens"] == 800, "the token counts must be untouched by the cost path"


@pytest.mark.asyncio
async def test_streamed_response_records_the_billed_cost():
    """Without the subclass copying the usage chunk across, a stream records tokens and no cost."""
    acc = _token_accountant()
    async for _ in _llm(_stream).astream([HumanMessage("x")], config={"callbacks": [acc]}):
        pass
    t = acc.totals()
    assert t["cost_usd"] == pytest.approx(0.00123)
    assert t["n_llm_calls"] == 1 and t["n_uncosted_calls"] == 0
    assert t["cache_read_tokens"] == 800


@pytest.mark.parametrize("model_id", sorted(models._CACHING))
@pytest.mark.asyncio
async def test_a_cached_model_sends_its_breakpoints_on_the_wire(model_id):
    """Placed by overriding ``_get_request_payload``, so a langchain rename would leave the
    table tests green while every request went out unmarked."""
    sent: dict = {}

    def handler(request: _httpx.Request) -> _httpx.Response:
        sent.update(json.loads(request.content))
        return _completion(request)

    await _llm(handler, model=model_id).ainvoke(
        [
            SystemMessage("sys text"),
            HumanMessage("task"),
            AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
            ToolMessage("tool output", tool_call_id="c1"),
        ]
    )

    marked = {
        m["role"]: b
        for m in sent["messages"]
        if isinstance(m.get("content"), list)
        for b in m["content"]
        if isinstance(b, dict) and "cache_control" in b
    }
    assert set(marked) == {"system", "tool"}, "the system prompt and the end of the history"
    for role, block in marked.items():
        assert block["cache_control"] == models.caching_config(model_id), f"{role}: wrong ttl/type"
    assert "cache_control" not in sent, "the top-level field is never sent"
