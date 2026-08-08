"""Provider pinning in ``models.py``.

Routing exists to stop a request silently landing on an endpoint that is far more expensive
(Luna: 11x on prompt price) or that bills a different cache-read rate (v4-flash: 26x across
providers, on a workload that is ~97% prompt tokens). None of that surfaces as a failure --
the run succeeds and the bill arrives later -- so the invariants are pinned here.
"""
import pytest

from petab_harness.models import MODELS, _ROUTING, provider_routing, resolve


def test_every_pinned_id_is_one_we_actually_use():
    """_ROUTING is keyed by model ID, not by alias.

    So swapping an alias to a different id -- e.g. `v4-flash` to the date-pinned
    `deepseek-v4-flash-0731` -- silently orphans its routing entry and the next sweep runs
    unpinned at whatever the default route costs. Nothing else would report that.
    """
    in_use = {entry[0] for entry in MODELS.values()}
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
    unpinned = sorted(a for a, e in MODELS.items() if provider_routing(e[0]) is None)
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
    from petab_harness.models import MODELS, _REASONING

    in_use = {entry[0] for entry in MODELS.values()}
    orphaned = set(_REASONING) - in_use
    assert not orphaned, f"{sorted(orphaned)} have reasoning config but are not on the panel"


def test_reasoning_effort_is_a_value_openrouter_accepts():
    from petab_harness.models import _REASONING

    for model_id, cfg in _REASONING.items():
        assert cfg.get("effort") in {"low", "medium", "high"} or "max_tokens" in cfg, (
            f"{model_id}: reasoning must set a valid `effort` or a `max_tokens` budget"
        )


def test_models_without_an_entry_keep_the_provider_default():
    """Omitted entirely, not sent as an empty block — a model whose thinking is already on must
    get exactly the request it got before."""
    from petab_harness.models import reasoning_config

    assert reasoning_config("openai/gpt-5.6-luna-pro") is None
    assert reasoning_config("some/unknown-model") is None


def test_extra_body_carries_routing_and_reasoning_independently():
    """Each field appears only when its table has an entry, so a model with neither sends a bare
    request exactly as before this mechanism existed."""
    from petab_harness.agent.driver import _extra_body

    luna = _extra_body("openai/gpt-5.6-luna-pro")
    assert set(luna) == {"provider"}, "pinned but no reasoning entry -> provider only"
    assert luna["provider"]["only"] == ["openai"]

    assert _extra_body("some/unpinned-model") == {}, "neither -> send nothing"


def test_extra_body_includes_reasoning_when_a_model_has_it(monkeypatch):
    from petab_harness import models as M
    from petab_harness.agent.driver import _extra_body

    monkeypatch.setitem(M._REASONING, "openai/gpt-5.6-luna-pro", {"effort": "high"})
    body = _extra_body("openai/gpt-5.6-luna-pro")
    assert body["reasoning"] == {"effort": "high"}
    assert body["provider"]["only"] == ["openai"], "reasoning must not displace the pin"
