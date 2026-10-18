"""agent.thinking: a runaway's thinking is stopped on the stream and handed back for an answer."""

import asyncio
import random

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from gemot.agent import thinking
from gemot.agent.guards import _reasoning_runaway_guard, new_guard_stats


def _words(n, seed=0):
    rng = random.Random(seed)
    return " ".join(f"w{rng.randrange(10**6)}" for _ in range(n))


def test_a_loop_trips_the_detector_and_fresh_thinking_does_not():
    loop = " ".join(["I should check the fit of the model again before I commit anything."] * 400)
    fresh = _words(6000)
    assert max(s for s in _shares(loop) if s is not None) >= thinking.REPEAT_SHARE
    assert all(s is None or s < 0.05 for s in _shares(fresh))


def _shares(text, piece=500):
    d = thinking.RepetitionDetector()
    return [d.feed(text[i : i + piece]) for i in range(0, len(text), piece)]


def test_streamed_reasoning_blocks_are_merged_per_index():
    deltas = [
        {"type": "reasoning.text", "index": 0, "text": "Fir", "format": "x"},
        {"type": "reasoning.text", "index": 0, "text": "st."},
        {"type": "reasoning.text", "index": 0, "signature": "sig"},
    ]
    assert thinking.merge_details(deltas) == [
        {"type": "reasoning.text", "index": 0, "text": "First.", "format": "x", "signature": "sig"}
    ]


class _LLM:
    model_name = "test/model"
    extra_body = {"reasoning": {"effort": "max"}}
    openai_api_key = None
    openai_api_base = None


def _chunk(raw):
    thinking.record_chunk(raw)
    delta = raw["choices"][0]["delta"] if raw.get("choices") else {}
    info = {"finish_reason": raw["choices"][0]["finish_reason"]} if raw.get("choices") and raw["choices"][0].get("finish_reason") else None
    msg = AIMessageChunk(content=delta.get("content") or "", tool_call_chunks=delta.get("tool_call_chunks") or [])
    return ChatGenerationChunk(message=msg, generation_info=info)


def _thinking_stream(text, piece=400):
    async def gen():
        for i in range(0, len(text), piece):
            yield _chunk({"id": "gen-1", "provider": "Z.AI", "choices": [{"delta": {"reasoning": text[i : i + piece]}}]})
    return gen()


def _answer_stream(calls):
    async def gen():
        yield _chunk({"id": "gen-2", "choices": [{"delta": {"content": "Gist: the fit lacks a nuclear pool."}}]})
        yield _chunk({
            "id": "gen-2",
            "choices": [{"delta": {"tool_call_chunks": [{"name": "make_changes", "args": "{}", "id": "c1", "index": 0}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.002},
        })
        calls.append("answered")
    return gen()


def test_a_looping_stream_is_stopped_and_its_thinking_handed_back(monkeypatch):
    loop = " ".join(["Let me reconsider whether the readout should be phosphorylation weighted."] * 600)
    sent = []

    def stream(messages, stop, run_manager, **kwargs):
        sent.append((messages, kwargs))
        return _thinking_stream(loop) if len(sent) == 1 else _answer_stream([])

    async def usage(llm, gid, tries=5):
        return {"prompt_tokens": 1000, "completion_tokens": 5000, "cost": 0.01}

    monkeypatch.setattr(thinking, "generation_usage", usage)
    thinking.THINKING_OFF.pop("test/model", None)
    result = asyncio.run(thinking.generate(_LLM(), None, stream, [HumanMessage(content="build it")]))
    msg = result.generations[0].message
    assert [c["name"] for c in msg.tool_calls] == ["make_changes"]
    record = msg.response_metadata["thinking_handoff"]
    assert record["reason"] == "repetition" and record["recovered"] and record["thinking_off"] == {"enabled": False}
    assert record["thinking_chars"] < len(loop)  # stopped, not read to the end
    handoff, kwargs = sent[1]
    assert isinstance(handoff[-2], AIMessage) and handoff[-2].content.startswith("Let me reconsider")
    assert handoff[-1].content == thinking.HANDOFF
    assert kwargs["extra_body"]["reasoning"] == {"enabled": False}
    assert kwargs["max_tokens"] == thinking.HANDOFF_MAX_TOKENS
    # Both calls are on the bill: the stopped stream's from /generation, the hand-off's own.
    assert msg.response_metadata["token_usage"]["cost"] == pytest.approx(0.012)
    assert msg.usage_metadata["output_tokens"] == 5020


def _bad_request(text):
    response = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    return openai.BadRequestError(text, response=response, body=None)


def test_a_long_fresh_deliberation_is_never_cut():
    # No thinking budget: 120k words of thought that does not repeat itself run to the end.
    watch = thinking.Watch()
    assert all(watch.feed(_words(3000, seed=i)) is None for i in range(40))


def test_a_refused_way_of_turning_thinking_off_moves_to_the_next(monkeypatch):
    sent = []

    def stream(messages, stop, run_manager, **kwargs):
        sent.append(kwargs)
        if len(sent) == 1:
            return _thinking_stream(" ".join(["Perhaps the readout needs a scaling factor after all."] * 600))
        if kwargs["extra_body"]["reasoning"] == {"enabled": False}:
            async def refused():
                raise _bad_request("reasoning cannot be disabled for this model")
                yield
            return refused()
        return _answer_stream([])

    async def usage(llm, gid, tries=5):
        return None  # OpenRouter never filed the stopped stream: the turn stays uncosted

    monkeypatch.setattr(thinking, "generation_usage", usage)
    thinking.THINKING_OFF.pop("test/model", None)
    msg = asyncio.run(thinking.generate(_LLM(), None, stream, [HumanMessage(content="x")])).generations[0].message
    record = msg.response_metadata["thinking_handoff"]
    assert record["reason"] == "repetition" and record["thinking_off"] == {"effort": "none"}
    assert thinking.THINKING_OFF["test/model"] == {"effort": "none"}
    assert "cost" not in msg.response_metadata["token_usage"]


def test_a_model_that_will_not_stream_is_called_whole_and_its_truncated_thinking_handed_back(monkeypatch):
    thinking.NO_STREAM.discard("test/model")
    thinking.THINKING_OFF["test/model"] = {"enabled": False}

    def stream(messages, stop, run_manager, **kwargs):
        async def refused():
            raise _bad_request("Streaming is not supported for this model")
            yield
        return refused()

    plain_calls = []

    async def plain(messages, stop, run_manager, **kwargs):
        plain_calls.append(messages)
        if len(plain_calls) == 1:
            thinking.record_response({"id": "g", "choices": [{"message": {"reasoning": "deliberating..."}}],
                                      "usage": {"prompt_tokens": 10, "completion_tokens": 65536, "cost": 0.1}})
            msg = AIMessage(content="")
            return ChatResult(generations=[ChatGeneration(message=msg, generation_info={"finish_reason": "length"})])
        thinking.record_response({"id": "h", "choices": [{"message": {}}], "usage": {"prompt_tokens": 20, "completion_tokens": 5, "cost": 0.001}})
        msg = AIMessage(content="", tool_calls=[{"name": "query_data", "args": {}, "id": "q"}])
        return ChatResult(generations=[ChatGeneration(message=msg, generation_info={"finish_reason": "tool_calls"})])

    try:
        msg = asyncio.run(thinking.generate(_LLM(), plain, stream, [HumanMessage(content="x")])).generations[0].message
        assert "test/model" in thinking.NO_STREAM
        assert plain_calls[1][-2].content == "deliberating..."
        record = msg.response_metadata["thinking_handoff"]
        assert record == {"reason": "length", "thinking_chars": 15, "streamed": False, "recovered": True, "thinking_off": {"enabled": False}}
        assert msg.response_metadata["token_usage"]["cost"] == pytest.approx(0.101)
    finally:
        thinking.NO_STREAM.discard("test/model")


def test_a_hand_off_that_runs_away_too_hands_the_empty_turn_back(monkeypatch):
    thinking.THINKING_OFF["test/model"] = {"enabled": False}

    def stream(messages, stop, run_manager, **kwargs):
        return _thinking_stream(" ".join(["round and round we go again"] * 2000))

    async def usage(llm, gid, tries=5):
        return {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}

    monkeypatch.setattr(thinking, "generation_usage", usage)
    msg = asyncio.run(thinking.generate(_LLM(), None, stream, [HumanMessage(content="x")])).generations[0].message
    assert msg.content == "" and msg.response_metadata["finish_reason"] == "length"
    assert msg.response_metadata["thinking_handoff"]["recovered"] is False


def test_a_healthy_streamed_turn_passes_through_with_its_reasoning_blocks():
    def stream(messages, stop, run_manager, **kwargs):
        async def gen():
            yield _chunk({"id": "g", "provider": "Anthropic", "choices": [{"delta": {"reasoning": "short", "reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "short"}]}}]})
            async for c in _answer_stream([]):
                yield c
            # OpenRouter repeats the finish reason on its usage chunk (seen live, GLM and Haiku).
            yield _chunk({"id": "g", "choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        return gen()

    msg = asyncio.run(thinking.generate(_LLM(), None, stream, [HumanMessage(content="x")])).generations[0].message
    assert "thinking_handoff" not in msg.response_metadata
    assert msg.response_metadata["finish_reason"] == "tool_calls"
    assert msg.additional_kwargs["reasoning_details"] == [{"type": "reasoning.text", "index": 0, "text": "short"}]
    assert msg.response_metadata["openrouter_provider"] == "Anthropic"
    assert msg.response_metadata["token_usage"]["cost"] == 0.002


def test_the_guard_counts_a_hand_off_once():
    stats = new_guard_stats()
    guard = _reasoning_runaway_guard(stats)

    class _Req:
        model_settings: dict = {}
        model = None

    recovered = AIMessage(content="gist", tool_calls=[{"name": "x", "args": {}, "id": "1"}],
                          response_metadata={"thinking_handoff": {"reason": "length", "recovered": True, "streamed": True}})
    failed = AIMessage(content="", response_metadata={"finish_reason": "length",
                       "thinking_handoff": {"reason": "repetition", "recovered": False, "streamed": True}})

    async def run(msg):
        async def handler(request):
            return msg
        return await guard.awrap_model_call(_Req(), handler)

    asyncio.run(run(recovered))
    asyncio.run(run(failed))
    assert stats["reasoning_runaways"] == 2
    assert stats["reasoning_runaway_recovered"] == 1
    assert stats["thinking_cutoffs"] == {"length": 1, "repetition": 1}
