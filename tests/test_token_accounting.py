"""Unit tests for agent._token_accountant — the callback that sums per-run token usage
(prompt / completion / cache_read) written into result.json. langchain_core is a
harness-only dependency, so skip where absent."""

from __future__ import annotations

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
            [type("G", (), {"message": type("M", (), {"usage_metadata": um,
                                                       "response_metadata": meta})()})()]
            for um, meta in zip(usage_list, metas)]
    return type("R", (), {"generations": generations, "llm_output": llm_output})()


def _um(inp, out, cache=0):
    return {"input_tokens": inp, "output_tokens": out,
            "input_token_details": {"cache_read": cache}}


@pytest.mark.asyncio
async def test_sums_usage_metadata_including_cache_read():
    acc = _token_usage._token_accountant()
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
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([{"input_tokens": 500, "output_tokens": 20}]))  # no details
    t = acc.totals()
    assert t["prompt_tokens"] == 500 and t["cache_read_tokens"] == 0


@pytest.mark.asyncio
async def test_falls_back_to_llm_output_when_no_usage_metadata():
    acc = _token_usage._token_accountant()
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
    acc = _token_usage._token_accountant()
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
    acc = _token_usage._token_accountant()
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
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp(llm_output=None))                 # nothing usable
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
    await acc.on_llm_end(_resp([_um(1000, 50), _um(1000, 50)], llm_output={"token_usage": tu},
                               metas=[{"token_usage": tu}, {"token_usage": tu}]))
    assert acc.totals()["cost_usd"] == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_streamed_response_cost_is_read_from_response_metadata():
    """A stream arrives with llm_output=None and its usage block on response_metadata only."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([_um(1000, 50)], llm_output=None,
                               metas=[{"token_usage": _tu(0.002)}]))
    t = acc.totals()
    assert t["cost_usd"] == pytest.approx(0.002) and t["n_uncosted_calls"] == 0


@pytest.mark.asyncio
async def test_no_billed_cost_records_none_not_zero():
    """Usage without `cost` (a non-OpenRouter endpoint) records None, not 0.0."""
    acc = _token_usage._token_accountant()
    await acc.on_llm_end(_resp([_um(1000, 50)],
                               llm_output={"token_usage": {"prompt_tokens": 1000}}))
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
