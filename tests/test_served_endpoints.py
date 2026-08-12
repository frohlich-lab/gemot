"""The upstream endpoint that SERVED a run is recorded, so endpoint drift is visible.

`provider.only` bounds routing to a SET of endpoints, not to one, so honouring the pin and being
served by a single machine are different claims. These tests pin the second one: that OpenRouter's
top-level `provider` survives langchain (which drops it), that it is tallied per run, and that a
run split across two endpoints is flagged rather than pooled.
"""

from __future__ import annotations

import asyncio

from petab_harness.agent.token_usage import _token_accountant
from petab_harness.metrics import _flatten_served_endpoints


class _Msg:
    def __init__(self, meta):
        self.response_metadata = meta


class _Gen:
    def __init__(self, meta):
        self.message = _Msg(meta)


class _Resp:
    """The shape LangChain hands to `on_llm_end`."""

    def __init__(self, llm_output=None, metas=()):
        self.llm_output = llm_output
        self.generations = [[_Gen(m)] for m in metas]


def _feed(*responses):
    acc = _token_accountant()

    async def go():
        for r in responses:
            await acc.on_llm_end(r)

    asyncio.run(go())
    return acc


def test_provider_read_from_llm_output():
    acc = _feed(_Resp({"openrouter_provider": "OpenAI", "service_tier": "default"}))
    ep = acc.endpoints()
    assert ep["served"] == [{"provider": "OpenAI", "service_tier": "default", "n_calls": 1}]
    assert ep["n_distinct"] == 1 and ep["drifted"] is False


def test_provider_falls_back_to_message_metadata():
    """A response whose aggregate lost the field is still attributed from the per-message copy."""
    acc = _feed(_Resp(llm_output={}, metas=[{"openrouter_provider": "Novita",
                                             "service_tier": "default"}]))
    assert acc.endpoints()["served"][0]["provider"] == "Novita"


def test_two_endpoints_are_flagged_as_drift():
    acc = _feed(_Resp({"openrouter_provider": "OpenAI", "service_tier": "default"}),
                _Resp({"openrouter_provider": "OpenAI", "service_tier": "default"}),
                _Resp({"openrouter_provider": "Azure", "service_tier": "default"}))
    ep = acc.endpoints()
    assert ep["drifted"] is True and ep["n_distinct"] == 2
    # Busiest first, so the majority endpoint reads off the top.
    assert [s["provider"] for s in ep["served"]] == ["OpenAI", "Azure"]
    assert [s["n_calls"] for s in ep["served"]] == [2, 1]


def test_same_provider_different_tier_is_drift():
    """`openai` and `openai/flex` are different prices and different machines."""
    acc = _feed(_Resp({"openrouter_provider": "OpenAI", "service_tier": "default"}),
                _Resp({"openrouter_provider": "OpenAI", "service_tier": "flex"}))
    assert acc.endpoints()["drifted"] is True


def test_unattributed_calls_are_counted_not_dropped():
    """Counts must sum to the calls made, or `n_calls` and the endpoint tally disagree."""
    acc = _feed(_Resp(None), _Resp({"openrouter_provider": "OpenAI", "service_tier": "default"}))
    ep = acc.endpoints()
    assert sum(s["n_calls"] for s in ep["served"]) == 2
    assert any(s["provider"] is None for s in ep["served"])


def test_an_unattributed_call_does_not_fake_drift():
    """A gap in OUR recording is not a second machine.

    Counting it as one flags an ordinary single-endpoint run as drifted, which would exclude a
    healthy run from the very A/B `served_drifted` exists to filter.
    """
    acc = _feed(_Resp(None), _Resp({"openrouter_provider": "OpenAI", "service_tier": "default"}))
    ep = acc.endpoints()
    assert ep["drifted"] is False
    assert ep["n_distinct"] == 1
    assert ep["n_unattributed_calls"] == 1


def test_accounting_survives_a_malformed_response():
    """Provenance capture must never break a run."""
    acc = _feed(_Resp("not-a-dict"))
    assert isinstance(acc.endpoints()["served"], list)


def test_a_truthy_non_dict_llm_output_is_still_tallied():
    """`or {}` let a bare string through to `.get`, raising and DROPPING the call from the tally —
    so the counts silently stopped summing to the calls made."""
    acc = _feed(_Resp("not-a-dict"), _Resp("also-not-a-dict"))
    ep = acc.endpoints()
    assert sum(s["n_calls"] for s in ep["served"]) == 2, "a malformed response must not vanish"
    assert ep["n_unattributed_calls"] == 2


def test_metrics_flattens_served_endpoints():
    row = _flatten_served_endpoints({"openrouter_endpoints": {
        "served": [{"provider": "OpenAI", "service_tier": "default", "n_calls": 3}],
        "n_distinct": 1, "drifted": False}})
    # provider/tier, not the bare name: the tier is half of what makes two endpoints distinct.
    assert row["served_provider"] == "OpenAI/default"
    assert row["served_n_distinct"] == 1 and row["served_drifted"] is False


def test_metrics_joins_multiple_providers_deterministically():
    row = _flatten_served_endpoints({"openrouter_endpoints": {
        "served": [{"provider": "OpenAI", "n_calls": 2}, {"provider": "Azure", "n_calls": 1}],
        "n_distinct": 2, "drifted": True}})
    assert row["served_provider"] == "Azure+OpenAI"   # sorted, so the key is stable
    assert row["served_drifted"] is True


def test_metrics_key_keeps_the_tier_that_made_two_endpoints_distinct():
    """A name-only key renders the tier-drift case as the useless "OpenAI+OpenAI" and drops the
    one field naming what actually differed."""
    row = _flatten_served_endpoints({"openrouter_endpoints": {
        "served": [{"provider": "OpenAI", "service_tier": "default", "n_calls": 2},
                   {"provider": "OpenAI", "service_tier": "flex", "n_calls": 1}],
        "n_distinct": 2, "drifted": True}})
    assert row["served_provider"] == "OpenAI/default+OpenAI/flex"


def test_metrics_never_writes_the_literal_string_None():
    """`str(None)` is indistinguishable from a real endpoint name when grouping a sweep."""
    row = _flatten_served_endpoints({"openrouter_endpoints": {
        "served": [{"provider": None, "service_tier": None, "n_calls": 3}],
        "n_distinct": 0, "drifted": False}})
    assert row["served_provider"] is None
    row2 = _flatten_served_endpoints({"openrouter_endpoints": {
        "served": [{"provider": "OpenAI", "service_tier": "default", "n_calls": 3},
                   {"provider": None, "service_tier": None, "n_calls": 1}],
        "n_distinct": 1, "drifted": False}})
    assert row2["served_provider"] == "OpenAI/default"


def test_metrics_leaves_pre_existing_rows_untouched():
    """A run recorded before this field existed must not gain columns that read as 'no drift'."""
    assert _flatten_served_endpoints({"scenario": "sneyd"}) == {"scenario": "sneyd"}


def test_served_columns_are_declared_in_the_history_header():
    """`DictWriter(extrasaction="ignore")` drops undeclared keys SILENTLY."""
    from petab_harness.metrics import HISTORY_FIELDS

    for col in ("served_provider", "served_n_distinct", "served_drifted"):
        assert col in HISTORY_FIELDS


def test_nested_source_is_carried_so_the_flattener_can_see_it():
    """build_row copies only flat fields; without this the columns stay empty on collect()."""
    from petab_harness.metrics import _NESTED_SOURCE_FIELDS

    assert "openrouter_endpoints" in _NESTED_SOURCE_FIELDS


def test_span_name_is_preserved_despite_the_subclass():
    """LangChain derives the run name — and openinference the SPAN name — from the class name.

    An un-named subclass renames every LLM span to `_ChatOpenAIRecordingProvider`, so every span
    query keyed on `ChatOpenAI` (the whole archive, and slurm/trace_review) returns no rows for new
    runs and cannot be compared with old ones.
    """
    from petab_harness.agent.driver import _chat_model_class

    kw = dict(model="openai/gpt-5.6-luna-pro", base_url="https://openrouter.ai/api/v1",
              api_key="not-a-real-key")
    assert _chat_model_class()(name="ChatOpenAI", **kw).get_name() == "ChatOpenAI"


def test_the_subclass_is_built_once():
    """Each call would otherwise construct — and permanently register — a fresh pydantic class."""
    from petab_harness.agent.driver import _chat_model_class

    assert _chat_model_class() is _chat_model_class()
