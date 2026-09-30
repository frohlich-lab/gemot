"""Provider pinning in ``models.py``.

Routing exists to stop a request silently landing on an endpoint that is far more expensive
(Luna: 11x on prompt price) or that bills a different cache-read rate (v4-flash: 26x across
providers, on a workload that is ~97% prompt tokens). None of that surfaces as a failure --
the run succeeds and the bill arrives later -- so the invariants are pinned here.
"""

import pytest

from gemot.models import (
    MODELS,
    _CACHING,
    _ROUTING,
    caching_config,
    provider_routing,
    resolve,
)


def test_every_pinned_id_is_one_we_actually_use():
    """_ROUTING is keyed by model ID, not by alias.

    So swapping an alias to a different id -- e.g. `v4-flash` to the date-pinned
    `deepseek-v4-flash-0731` -- silently orphans its routing entry and the next sweep runs
    unpinned at whatever the default route costs. Nothing else would report that.
    """
    in_use = set(MODELS.values())
    orphaned = set(_ROUTING) - in_use
    assert not orphaned, (
        f"{sorted(orphaned)} have routing but no alias points at them -- an alias was "
        "repointed and its pin was left behind, so it now routes unpinned"
    )


@pytest.mark.parametrize("model_id", sorted(_ROUTING))
def test_fallback_is_bounded_by_an_allowlist(model_id):
    """`order` alone still permits fallback to ANY other provider; `only` is what bounds it."""
    block = _ROUTING[model_id]
    assert "only" in block, f"{model_id} has no `only` allowlist -- fallback is unbounded"
    if "order" in block:
        assert set(block["order"]) <= set(block["only"]), (
            f"{model_id} prefers a provider its allowlist excludes"
        )


def test_deepseek_flash_excludes_the_fp4_endpoint():
    """DeepInfra is the cheapest ZDR-eligible endpoint but serves fp4.

    Benchmarking across differently-quantized endpoints confounds the model comparison, so
    the allowlist must keep it out even though it would be cheaper.
    """
    block = provider_routing("deepseek/deepseek-v4-flash")
    assert block is not None
    assert "deepinfra" not in block["only"]
    assert "deepseek" not in block["only"], "DeepSeek-direct is not ZDR-eligible for this account"


def test_unknown_models_return_none():
    """Callers omit the `provider` field entirely rather than sending an empty block."""
    assert provider_routing("some/model-we-never-heard-of") is None


def test_every_panel_model_is_pinned():
    """No alias may route by default.

    Default routing is not a neutral fallback: it landed minimax-m3 on a 524K-context endpoint
    and picked neither the cheapest nor a consistent provider for the others. A new alias with
    no routing entry is the regression this catches.
    """
    unpinned = sorted(a for a, e in MODELS.items() if provider_routing(e) is None)
    assert not unpinned, (
        f"{unpinned} route by default -- pick an endpoint deliberately (ZDR eligibility, "
        "context length and cache-read price all have to be checked live)"
    )


def test_resolve_maps_the_alias_to_the_pinned_id():
    assert resolve("v4-flash") == "deepseek/deepseek-v4-flash"
    assert provider_routing(resolve("v4-flash")) is not None, (
        "the alias resolves to an id with no routing -- see test_every_pinned_id_is_one_we_actually_use"
    )


# ------------------------------------------------------------------ reasoning config


def test_every_reasoning_entry_names_a_model_on_the_panel():
    """Same orphan check as the routing table: `_REASONING` is keyed by model ID, so an entry for
    a model no longer in MODELS is dead config that silently does nothing."""
    from gemot.models import MODELS, _REASONING

    in_use = set(MODELS.values())
    orphaned = set(_REASONING) - in_use
    assert not orphaned, f"{sorted(orphaned)} have reasoning config but are not on the panel"


def test_reasoning_effort_is_a_value_openrouter_accepts():
    from gemot.models import _REASONING

    for model_id, cfg in _REASONING.items():
        assert cfg.get("effort") in {"low", "medium", "high"} or "max_tokens" in cfg, (
            f"{model_id}: reasoning must set a valid `effort` or a `max_tokens` budget"
        )


def test_models_without_an_entry_keep_the_provider_default():
    """Omitted entirely, not sent as an empty block — a model whose thinking is already on must
    get exactly the request it got before."""
    from gemot.models import reasoning_config

    assert reasoning_config("openai/gpt-5.6-luna-pro") is None
    assert reasoning_config("some/unknown-model") is None


def test_extra_body_carries_routing_and_reasoning_independently():
    """Routing and reasoning appear only when their table has an entry; the billed-cost flag is on
    every request, so a model with neither sends exactly that."""
    from gemot.agent.driver import _extra_body

    luna = _extra_body("openai/gpt-5.6-luna-pro")
    assert set(luna) == {"provider", "usage"}, (
        "pinned but no reasoning entry -> the pin and the cost flag, nothing else"
    )
    assert luna["provider"]["only"] == ["openai"]
    assert luna["usage"] == {"include": True}

    assert _extra_body("some/unpinned-model") == {"usage": {"include": True}}, (
        "neither -> the cost flag alone"
    )


def test_extra_body_includes_reasoning_when_a_model_has_it(monkeypatch):
    from gemot import models as M
    from gemot.agent.driver import _extra_body

    monkeypatch.setitem(M._REASONING, "openai/gpt-5.6-luna-pro", {"effort": "high"})
    body = _extra_body("openai/gpt-5.6-luna-pro")
    assert body["reasoning"] == {"effort": "high"}
    assert body["provider"]["only"] == ["openai"], "reasoning must not displace the pin"


# ------------------------------------------------------------------ prompt caching


def test_every_caching_entry_names_a_model_on_the_panel():
    """An entry for a model no longer in MODELS is dead config that silently does nothing."""
    in_use = set(MODELS.values())
    orphaned = set(_CACHING) - in_use
    assert not orphaned, f"{sorted(orphaned)} have caching config but are not on the panel"


def test_implicitly_caching_models_ask_for_nothing():
    """An entry is not free on a provider that already caches: Gemini would start billing the
    writes that implicit caching gives us at zero."""
    for model_id in (
        "z-ai/glm-5.3-flash",
        "openai/gpt-5.6-luna",
        "deepseek/deepseek-v4-flash",
        "google/gemini-3.6-flash",
        "google/gemini-3.1-pro-preview",
    ):
        assert caching_config(model_id) is None, (
            f"{model_id} caches implicitly -- asking explicitly can only add cost"
        )
    assert caching_config("some/model-we-never-heard-of") is None


def test_cache_control_asks_for_the_one_hour_ttl():
    """The 5-minute default expires during a fit, and a cold prefix is re-written at the write
    rate — more expensive than never caching."""
    for model_id, block in _CACHING.items():
        assert block.get("type") == "ephemeral", f"{model_id}: unknown cache_control type"
        assert block.get("ttl") == "1h", (
            f"{model_id} would fall back to the 5-minute TTL, which expires inside a single fit"
        )


@pytest.mark.parametrize("model_id", sorted(_CACHING))
def test_an_explicitly_cached_model_is_pinned_to_exactly_one_endpoint(model_id):
    """Each provider keeps its own cache, so a fallback endpoint is a cold one, not a cheap
    outage."""
    block = provider_routing(model_id)
    assert block is not None, f"{model_id} asks for caching but routes by default"
    assert block.get("only") and len(block["only"]) == 1, (
        f"{model_id} may fail over to a second endpoint, which is a second (cold) cache"
    )
    assert "order" not in block, f"{model_id} has a fallback order, so it has a fallback cache"


def test_a_blank_text_block_does_not_capture_the_breakpoint():
    """An assistant turn that is only tool calls can arrive as a blank text block rather than
    `content: None`; marking it would leave the tool result — the real end of the history —
    unmarked."""
    from gemot.agent.driver import _apply_cache_breakpoints

    cfg = {"type": "ephemeral", "ttl": "1h"}
    payload = {
        "messages": [
            {"role": "system", "content": "system text"},
            {"role": "tool", "tool_call_id": "c1", "content": "tool output"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "  "}],
                "tool_calls": [{"id": "c2"}],
            },
        ]
    }
    _apply_cache_breakpoints(payload, cfg)
    system, tool, assistant = payload["messages"]
    assert assistant["content"] == [{"type": "text", "text": "  "}], "blank block left alone"
    assert tool["content"] == [{"type": "text", "text": "tool output", "cache_control": cfg}]
    assert system["content"][0]["cache_control"] == cfg


def test_caching_is_not_asked_for_in_the_request_body():
    """The top-level `cache_control` field is never sent (docs/RATIONALE.md)."""
    from gemot.agent.driver import _extra_body

    for model in ("anthropic/claude-sonnet-5", "openai/gpt-5.6-luna-pro"):
        assert "cache_control" not in _extra_body(model)
    assert _extra_body("anthropic/claude-sonnet-5")["provider"]["only"] == ["google-vertex/global"]


def test_breakpoints_land_on_the_system_message_and_the_last_message():
    """The last message in this loop is a TOOL result, and a string content is rewritten into a
    text block to carry the marker."""
    from gemot.agent.driver import _apply_cache_breakpoints

    cfg = {"type": "ephemeral", "ttl": "1h"}
    payload = {
        "messages": [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "tool output"},
        ]
    }
    _apply_cache_breakpoints(payload, cfg)
    system, user, assistant, tool = payload["messages"]
    assert system["content"] == [{"type": "text", "text": "system text", "cache_control": cfg}]
    assert tool["content"] == [{"type": "text", "text": "tool output", "cache_control": cfg}]
    assert user["content"] == "task", "only two breakpoints are placed"
    assert assistant["content"] is None, "a message with no text cannot carry one"


def test_breakpoints_skip_a_contentless_tail_and_mark_existing_blocks():
    """A trailing assistant turn carrying only tool calls has nothing to mark, so the breakpoint
    walks back to the last message that has text; content already in block form is marked in
    place rather than rewritten."""
    from gemot.agent.driver import _apply_cache_breakpoints

    cfg = {"type": "ephemeral"}
    payload = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "s"}]},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        ]
    }
    _apply_cache_breakpoints(payload, cfg)
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "s", "cache_control": cfg}
    ]
    assert payload["messages"][1]["content"] == [
        {"type": "text", "text": "task", "cache_control": cfg}
    ]


def test_the_request_langchain_builds_actually_carries_the_breakpoints():
    """The table being right is not the same as the marker reaching the provider.

    `langchain_openai` keeps `cache_control` on system and human content blocks and DROPS it from
    assistant and tool ones, which is why the breakpoints are applied to the converted payload.
    An upgrade that changed `_get_request_payload` would leave every test above green while every
    request went out uncached — silently, since an uncached request still succeeds.
    """
    pytest.importorskip("langchain_openai", reason="harness-only dependency")
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    from gemot.agent.driver import _chat_model_class, _extra_body

    model_id = "anthropic/claude-sonnet-5"
    llm = _chat_model_class()(
        name="ChatOpenAI",
        model=model_id,
        api_key="sk-not-a-real-key",
        base_url="https://openrouter.ai/api/v1",
        extra_body=_extra_body(model_id),
    )
    payload = llm._get_request_payload(
        [
            SystemMessage(content="system text"),
            HumanMessage(content="task"),
            AIMessage(
                content="",
                tool_calls=[{"name": "execute", "args": {}, "id": "c1", "type": "tool_call"}],
            ),
            ToolMessage(content="tool output", tool_call_id="c1"),
        ],
        stop=None,
    )
    marked = [
        m["role"]
        for m in payload["messages"]
        if isinstance(m.get("content"), list) and any(b.get("cache_control") for b in m["content"])
    ]
    assert marked == ["system", "tool"]
    assert payload["extra_body"]["provider"] == {"only": ["google-vertex/global"]}
    # a model with no caching entry is untouched
    plain = _chat_model_class()(
        name="ChatOpenAI",
        model="openai/gpt-5.6-luna-pro",
        api_key="sk-not-a-real-key",
        base_url="https://openrouter.ai/api/v1",
    )
    out = plain._get_request_payload([SystemMessage(content="s")], stop=None)
    assert out["messages"][0]["content"] == "s"
