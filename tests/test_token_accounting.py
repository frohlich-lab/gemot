"""Unit tests for agent._token_accountant — the callback that sums per-run token usage
(prompt / completion / cache_read) written into result.json. langchain_core is a
harness-only dependency, so skip where absent."""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core", reason="harness-only dependency")

from petab_harness.agent import token_usage as _token_usage  # noqa: E402


def _load_agent_module():
    return _token_usage


def _resp(usage_list=None, llm_output=None):
    """A fake LLMResult: generations carry AIMessage.usage_metadata; llm_output is the fallback."""
    generations = None
    if usage_list is not None:
        generations = [[type("G", (), {"message": type("M", (), {"usage_metadata": um})()})()]
                       for um in usage_list]
    return type("R", (), {"generations": generations, "llm_output": llm_output})()


def _um(inp, out, cache=0):
    return {"input_tokens": inp, "output_tokens": out,
            "input_token_details": {"cache_read": cache}}


@pytest.mark.asyncio
async def test_sums_usage_metadata_including_cache_read():
    acc = _load_agent_module()._token_accountant()
    await acc.on_llm_end(_resp([_um(1000, 50, cache=800)]))
    await acc.on_llm_end(_resp([_um(2000, 40, cache=1500)]))   # accumulates across calls
    t = acc.totals()
    assert t["prompt_tokens"] == 3000
    assert t["completion_tokens"] == 90
    assert t["cache_read_tokens"] == 2300
    assert t["total_tokens"] == 3090
    assert t["n_llm_calls"] == 2


@pytest.mark.asyncio
async def test_missing_cache_read_defaults_to_zero():
    acc = _load_agent_module()._token_accountant()
    await acc.on_llm_end(_resp([{"input_tokens": 500, "output_tokens": 20}]))  # no details
    t = acc.totals()
    assert t["prompt_tokens"] == 500 and t["cache_read_tokens"] == 0


@pytest.mark.asyncio
async def test_falls_back_to_llm_output_when_no_usage_metadata():
    acc = _load_agent_module()._token_accountant()
    await acc.on_llm_end(_resp(llm_output={"token_usage": {
        "prompt_tokens": 700, "completion_tokens": 30,
        "prompt_tokens_details": {"cached_tokens": 400}}}))
    t = acc.totals()
    assert t["prompt_tokens"] == 700 and t["completion_tokens"] == 30
    assert t["cache_read_tokens"] == 400 and t["n_llm_calls"] == 1


@pytest.mark.asyncio
async def test_no_usage_records_zero_and_no_call():
    """A response with NO usage at all (the streaming-without-usage case) records zero and, in
    particular, does not count as an LLM call — so the run-level n_llm_calls==0 warning can fire
    rather than the zeros passing as a genuinely tiny run."""
    acc = _load_agent_module()._token_accountant()
    await acc.on_llm_end(_resp([{"input_tokens": 0, "output_tokens": 0}]))  # empty-but-present usage
    await acc.on_llm_end(_resp(llm_output={}))                              # no usage anywhere
    t = acc.totals()
    assert t["total_tokens"] == 0
    # the empty-but-present usage_metadata counts as a call; the usage-less one does not
    assert t["n_llm_calls"] == 1


@pytest.mark.asyncio
async def test_malformed_field_does_not_discard_other_records():
    """A bad value in one usage field must contribute 0 without aborting the rest of the
    response's records — a whole-method try/except would have dropped the good one too."""
    acc = _load_agent_module()._token_accountant()
    await acc.on_llm_end(_resp([
        {"input_tokens": "oops", "output_tokens": 10},   # bad prompt -> 0, completion still 10
        _um(500, 25, cache=200),                          # fully valid, must still be counted
    ]))
    t = acc.totals()
    assert t["prompt_tokens"] == 500          # 0 (bad) + 500 (good)
    assert t["completion_tokens"] == 35       # 10 + 25 — the good record was NOT discarded
    assert t["cache_read_tokens"] == 200
    assert t["n_llm_calls"] == 2


@pytest.mark.asyncio
async def test_never_raises_on_malformed_response():
    acc = _load_agent_module()._token_accountant()
    await acc.on_llm_end(_resp(llm_output=None))                 # nothing usable
    await acc.on_llm_end(type("R", (), {"generations": "garbage", "llm_output": 5})())
    assert acc.totals()["total_tokens"] == 0
