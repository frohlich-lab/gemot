"""Unit tests for agent._token_accountant — the callback that sums per-run token usage
(prompt / completion / cache_read / cache_write) written into result.json. langchain_core is a
harness-only dependency, so skip where absent."""

from __future__ import annotations

from uuid import uuid4

import pytest

pytest.importorskip("langchain_core", reason="harness-only dependency")

from petab_harness.agent import token_usage as _token_usage  # noqa: E402


def _resp(usage_list=None, llm_output=None, metas=None):
    """A fake LLMResult: generations carry AIMessage.usage_metadata (and response_metadata, where
    a stream's usage block lands); llm_output is the fallback."""
    generations = None
    if usage_list is not None:
        metas = metas or [None] * len(usage_list)
        generations = [
            [
                type(
                    "G",
                    (),
                    {"message": type("M", (), {"usage_metadata": um, "response_metadata": meta})()},
                )()
            ]
            for um, meta in zip(usage_list, metas)
        ]
    return type("R", (), {"generations": generations, "llm_output": llm_output})()


def _um(inp, out, cache=0, written=0):
    """langchain's UsageMetadata. `cache_creation` is where langchain_openai puts OpenRouter's
    `prompt_tokens_details.cache_write_tokens` — the premium-priced column."""
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "input_token_details": {"cache_read": cache, "cache_creation": written},
    }


@pytest.mark.asyncio
async def test_sums_usage_metadata_including_cache_read():
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([_um(1000, 50, cache=800)]))
    await acc.on_llm_end(_resp([_um(2000, 40, cache=1500)]))  # accumulates across calls
    t = acc.totals()
    assert t["prompt_tokens"] == 3000
    assert t["completion_tokens"] == 90
    assert t["cache_read_tokens"] == 2300
    assert t["total_tokens"] == 3090
    assert t["n_llm_calls"] == 2


@pytest.mark.asyncio
async def test_missing_cache_read_defaults_to_zero():
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([{"input_tokens": 500, "output_tokens": 20}]))  # no details
    t = acc.totals()
    assert t["prompt_tokens"] == 500 and t["cache_read_tokens"] == 0
    # Zero rather than a missing key: "never writes" and "we failed to record it" must differ.
    assert t["cache_write_tokens"] == 0


@pytest.mark.asyncio
async def test_sums_cache_writes_separately_from_reads():
    """Reads bill ~0.1x input and writes 1.25-2x, so summing them hides a cache that kept going
    cold — which `cost_usd` shows only as a bigger bill."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([_um(10_000, 100, cache=8_000, written=1_500)]))
    await acc.on_llm_end(_resp([_um(12_000, 80, cache=11_000, written=900)]))
    t = acc.totals()
    assert t["cache_read_tokens"] == 19_000
    assert t["cache_write_tokens"] == 2_400
    # novel = prompt - cache_read: the writes are the part of it charged at the premium, not a
    # third disjoint bucket, so they are NOT subtracted twice.
    assert t["novel_tokens"] == 3_000


@pytest.mark.asyncio
async def test_falls_back_to_llm_output_when_no_usage_metadata():
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(
        _resp(
            llm_output={
                "token_usage": {
                    "prompt_tokens": 700,
                    "completion_tokens": 30,
                    "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 250},
                }
            }
        )
    )
    t = acc.totals()
    assert t["prompt_tokens"] == 700 and t["completion_tokens"] == 30
    assert t["cache_read_tokens"] == 400 and t["n_llm_calls"] == 1
    # The raw OpenRouter spelling, not langchain's: this path reads the provider's own record.
    assert t["cache_write_tokens"] == 250


@pytest.mark.asyncio
async def test_no_usage_records_zero_and_no_call():
    """A response with NO usage at all (the streaming-without-usage case) records zero and, in
    particular, does not count as an LLM call — so the run-level n_llm_calls==0 warning can fire
    rather than the zeros passing as a genuinely tiny run."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(
        _resp([{"input_tokens": 0, "output_tokens": 0}])
    )  # empty-but-present usage
    await acc.on_llm_end(_resp(llm_output={}))  # no usage anywhere
    t = acc.totals()
    assert t["total_tokens"] == 0
    # the empty-but-present usage_metadata counts as a call; the usage-less one does not
    assert t["n_llm_calls"] == 1


@pytest.mark.asyncio
async def test_malformed_field_does_not_discard_other_records():
    """A bad value in one usage field must contribute 0 without aborting the rest of the
    response's records — a whole-method try/except would have dropped the good one too."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(
        _resp(
            [
                {
                    "input_tokens": "oops",
                    "output_tokens": 10,
                },  # bad prompt -> 0, completion still 10
                _um(500, 25, cache=200),  # fully valid, must still be counted
            ]
        )
    )
    t = acc.totals()
    assert t["prompt_tokens"] == 500  # 0 (bad) + 500 (good)
    assert t["completion_tokens"] == 35  # 10 + 25 — the good record was NOT discarded
    assert t["cache_read_tokens"] == 200
    assert t["n_llm_calls"] == 2


@pytest.mark.asyncio
async def test_never_raises_on_malformed_response():
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp(llm_output=None))  # nothing usable
    await acc.on_llm_end(type("R", (), {"generations": "garbage", "llm_output": 5})())
    assert acc.totals()["total_tokens"] == 0
    assert acc.totals()["cost_usd"] is None


# ------------------------------------------------------------------ billed cost (#178)


def _tu(cost, prompt=1000, completion=50):
    """OpenRouter's raw usage block as langchain_openai keeps it in ``token_usage``."""
    return {"prompt_tokens": prompt, "completion_tokens": completion, "cost": cost}


@pytest.mark.asyncio
async def test_billed_cost_is_summed_from_the_usage_block():
    """The non-streamed shape: the raw usage block in llm_output, summed over calls."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([_um(1000, 50)], llm_output={"token_usage": _tu(0.0012)}))
    await acc.on_llm_end(_resp([_um(2000, 40)], llm_output={"token_usage": _tu(0.0030)}))
    t = acc.totals()
    assert t["cost_usd"] == pytest.approx(0.0042)
    assert t["n_uncosted_calls"] == 0


@pytest.mark.asyncio
async def test_cost_is_counted_once_per_response():
    """The same block sits in llm_output AND on response_metadata, and on every candidate when
    n>1; it is the bill for one call and is added once."""
    acc = _token_usage._token_accountant()
    tu = _tu(0.005)
    await acc.on_llm_end(
        _resp(
            [_um(1000, 50), _um(1000, 50)],
            llm_output={"token_usage": tu},
            metas=[{"token_usage": tu}, {"token_usage": tu}],
        )
    )
    assert acc.totals()["cost_usd"] == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_streamed_response_cost_is_read_from_response_metadata():
    """A stream arrives with llm_output=None and its usage block on response_metadata only."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(
        _resp([_um(1000, 50)], llm_output=None, metas=[{"token_usage": _tu(0.002)}])
    )
    t = acc.totals()
    assert t["cost_usd"] == pytest.approx(0.002) and t["n_uncosted_calls"] == 0


@pytest.mark.asyncio
async def test_no_billed_cost_records_none_not_zero():
    """Usage without `cost` (a non-OpenRouter endpoint) records None, not 0.0."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(
        _resp([_um(1000, 50)], llm_output={"token_usage": {"prompt_tokens": 1000}})
    )
    t = acc.totals()
    assert t["cost_usd"] is None
    assert t["n_uncosted_calls"] == t["n_llm_calls"] == 1


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -float("inf"), True])
@pytest.mark.asyncio
async def test_a_non_finite_cost_is_absent_rather_than_summed(cost):
    """A NaN summed into `cost_usd` makes every mean over it NaN, and reports the run as uncosted
    nowhere. `True` would silently bill $1.00. Both are absent figures, like a missing field."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([_um(1000, 50)], llm_output={"token_usage": _tu(cost)}))
    t = acc.totals()
    assert t["cost_usd"] is None
    assert t["n_uncosted_calls"] == t["n_llm_calls"] == 1


@pytest.mark.asyncio
async def test_partial_cost_coverage_is_visible():
    """The sum covers the calls that reported a figure and n_uncosted_calls counts the rest; a
    malformed cost is absent, not 0."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([_um(1000, 50)], llm_output={"token_usage": _tu(0.004)}))
    await acc.on_llm_end(_resp([_um(1000, 50)], llm_output={"token_usage": _tu("oops")}))
    await acc.on_llm_end(_resp([_um(1000, 50)], llm_output={"token_usage": _tu(None)}))
    t = acc.totals()
    assert t["cost_usd"] == pytest.approx(0.004)
    assert t["n_llm_calls"] == 3 and t["n_uncosted_calls"] == 2


async def _start(acc, agent="lead", node="model"):
    run_id = uuid4()
    await acc.on_chat_model_start(
        {}, [], run_id=run_id, metadata={"lc_agent_name": agent, "langgraph_node": node}
    )
    return run_id


@pytest.mark.asyncio
async def test_lead_final_is_by_request_order_and_peak_survives_smaller_prompts():
    """A late completion must not overwrite the newest lead input; subagents and helper
    calls can exceed both lead sizes without changing them."""
    acc = _token_usage._token_accountant()
    older, newer = await _start(acc), await _start(acc)
    await acc.on_llm_end(_resp([_um(300, 10, cache=200)]), run_id=newer)
    await acc.on_llm_end(_resp([_um(1000, 10, cache=800)]), run_id=older)
    for name, node in [("biocurator", "model"), ("lead", "SummarizationMiddleware.before_model")]:
        run_id = await _start(acc, name, node)
        await acc.on_llm_end(_resp([_um(5000, 10)]), run_id=run_id)
    t = acc.totals()
    assert t["final_lead_prompt_tokens"] == 300
    assert t["max_lead_prompt_tokens"] == 1000
    assert t["max_prompt_tokens"] == 5000
    assert t["total_tokens"] == 11340 and t["cache_read_tokens"] == 1000


@pytest.mark.parametrize("size", [None, "oops", -1, True, 1.5, float("inf"), 0, 700])
@pytest.mark.parametrize("raw_usage", [False, True])
@pytest.mark.asyncio
async def test_lead_final_does_not_reuse_an_earlier_count_when_usage_is_missing(size, raw_usage):
    """Both provider usage shapes preserve unknown vs zero, even after a measured lead call."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([_um(1000, 10)]), run_id=await _start(acc))
    result = (
        _resp(llm_output={"token_usage": {"prompt_tokens": size, "completion_tokens": 10}})
        if raw_usage
        else _resp([_um(size, 10)])
    )
    await acc.on_llm_end(result, run_id=await _start(acc))
    expected = size if type(size) is int and size >= 0 else None
    assert acc.totals()["final_lead_prompt_tokens"] == expected
    assert acc.totals()["max_lead_prompt_tokens"] == 1000


@pytest.mark.asyncio
async def test_failed_or_usage_less_final_lead_request_is_unknown_until_retry_reports_usage():
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([_um(1000, 10)]), run_id=await _start(acc))
    # An unrelated failure must not erase the lead's last measured input.
    await acc.on_llm_error(RuntimeError("subagent failed"), run_id=await _start(acc, "biocurator"))
    assert acc.totals()["final_lead_prompt_tokens"] == 1000
    await acc.on_llm_error(RuntimeError("lead failed"), run_id=await _start(acc))
    assert acc.totals()["final_lead_prompt_tokens"] is None
    await acc.on_llm_end(_resp([_um(200, 10)]), run_id=await _start(acc))
    assert acc.totals()["final_lead_prompt_tokens"] == 200
    await acc.on_llm_end(_resp(), run_id=await _start(acc))
    assert acc.totals()["final_lead_prompt_tokens"] is None
    assert acc.totals()["max_lead_prompt_tokens"] == 1000


@pytest.mark.asyncio
async def test_unattributed_usage_does_not_invent_lead_measurements():
    acc = _token_usage._token_accountant()
    run_id = uuid4()
    await acc.on_chat_model_start({}, [], run_id=run_id)
    await acc.on_llm_end(_resp([_um(1000, 10)]), run_id=run_id)
    assert acc.totals()["max_prompt_tokens"] == 1000
    assert acc.totals()["final_lead_prompt_tokens"] is None
    assert acc.totals()["max_lead_prompt_tokens"] is None


@pytest.mark.asyncio
async def test_real_deepagents_callbacks_separate_shared_model_and_summary_calls():
    """Real graph metadata must override the parent's name in concurrent subagents. A
    before_model helper uses the lead's name but must not become its final/peak prompt."""
    from deepagents import create_deep_agent
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    def message(size, content="done", **kwargs):
        return AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": size,
                "output_tokens": 10,
                "total_tokens": size + 10,
                "input_token_details": {"cache_read": size // 2},
            },
            **kwargs,
        )

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    helper = ToolModel(responses=[message(20000)])

    class SummaryProbe(AgentMiddleware):
        async def abefore_model(self, state, runtime):
            await helper.ainvoke([HumanMessage(content="summarize")])

    llm = ToolModel(
        responses=[
            message(
                1000,
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"subagent_type": name, "description": "inspect"},
                        "id": name,
                        "type": "tool_call",
                    }
                    for name in ["analyst_a", "analyst_b"]
                ],
            ),
            message(5000),
            message(9000),
            message(300),
            message(400),
        ]
    )
    graph = create_deep_agent(
        model=llm,
        name=_token_usage._LEAD_AGENT_NAME,
        system_prompt="lead",
        tools=[],
        subagents=[
            {"name": name, "description": "analyst", "system_prompt": "analyst", "tools": []}
            for name in ["analyst_a", "analyst_b"]
        ],
        middleware=[SummaryProbe()],
        checkpointer=MemorySaver(),
    )
    acc = _token_usage._token_accountant()
    config = {"callbacks": [acc], "configurable": {"thread_id": "lead-prompt-test"}}
    async for _ in graph.astream({"messages": [HumanMessage(content="inspect")]}, config):
        pass
    t = acc.totals()
    assert t["n_llm_calls"] == 6  # two lead, two subagent, two summary helpers
    assert t["total_tokens"] == 55360
    assert t["max_prompt_tokens"] == 20000
    assert t["final_lead_prompt_tokens"] == 300
    assert t["max_lead_prompt_tokens"] == 1000
    # The same callback instance spans a resumed graph and retains its earlier peak.
    await graph.ainvoke({"messages": [HumanMessage(content="continue")]}, config)
    assert acc.totals()["final_lead_prompt_tokens"] == 400
    assert acc.totals()["max_lead_prompt_tokens"] == 1000
