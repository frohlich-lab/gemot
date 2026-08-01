"""Go-to lead models for the deepagents panel, and the OpenRouter routing that pins the price.

OpenRouter serves a single model across SEVERAL providers at different prices. GPT-5.6 Luna, for
instance, is offered OpenAI-direct at ~$0.10/M prompt but ALSO via Amazon Bedrock ($0.22/M) and
Azure ($1.00-1.10/M) — up to ~11x more. Left to default routing a request can silently land on an
expensive copy, so the Luna family is pinned to OpenAI via provider routing (``only: [openai]``):
it serves OpenAI's documented standard tier, never the Bedrock/Azure copies. (We avoid
``sort: price``, which would opt into OpenAI's cheaper but undocumented ``flex`` tier.)

``MODELS`` maps a short alias to its OpenRouter id so ``--model luna`` works; :func:`resolve`
passes a full id through unchanged. The per-alias prices ($/1M tokens, prompt / completion) are
the OpenAI-direct listing at the time of writing and are documentation only — OpenRouter's live
catalog is the source of truth for what a request actually costs.
"""

from __future__ import annotations

#: alias -> (OpenRouter id, prompt $/M, completion $/M). The panel's go-to leads.
MODELS: dict[str, tuple[str, float, float]] = {
    "luna":       ("openai/gpt-5.6-luna",         0.10, 0.60),
    "luna-pro":   ("openai/gpt-5.6-luna-pro",     0.10, 0.60),
    "v4-flash":   ("deepseek/deepseek-v4-flash",  0.14, 0.28),
    "v4-pro":     ("deepseek/deepseek-v4-pro",    0.435, 0.87),
    "minimax-m3": ("minimax/minimax-m3",          0.30, 1.20),
}

#: Default lead. 1M-token context; on this prompt-heavy workload (~95% prompt tokens) the blended
#: cost tracks the $0.10/M prompt price of OpenAI's standard tier (see :func:`provider_routing`).
DEFAULT = "luna-pro"

#: Per-model OpenRouter provider routing, needed only where a model is MIRRORED across providers at
#: very different prices. OpenRouter serves the Luna family via OpenAI *and* Amazon Bedrock
#: ($0.22/M) and Azure ($1.00-1.10/M — a separate ``azure`` and ``azure/eu`` endpoint each). We use
#: an ALLOWLIST — ``only: [openai]`` — NOT a denylist: routing keys off each endpoint's ``tag``, and
#: a denylist would have to enumerate every expensive tag (``azure``, ``azure/eu``,
#: ``amazon-bedrock/us-east-1``, …) and would still miss any provider added later. An allowlist
#: excludes all of them by construction, with nothing to track. We deliberately do NOT add
#: ``sort: price`` (that opts into OpenAI's undocumented ``openai/flex`` $0.05/M tier); left alone,
#: ``only: [openai]`` serves the standard ``openai`` tier at the documented $0.10/M — verified live
#: (served by OpenAI, completion billed $0.60/M, not flex's $0.30/M). ``flex``/``priority`` are
#: opt-in. Single-provider models (deepseek, minimax) need no routing and are omitted.
_ROUTING: dict[str, dict] = {
    "openai/gpt-5.6-luna":     {"only": ["openai"]},
    "openai/gpt-5.6-luna-pro": {"only": ["openai"]},
}


def resolve(name: str) -> str:
    """Expand a go-to alias (``luna``) to its OpenRouter id; return a full id unchanged."""
    entry = MODELS.get(name)
    return entry[0] if entry else name


def provider_routing(model_id: str) -> dict | None:
    """The OpenRouter ``provider`` block for ``model_id``, or ``None`` when no pinning is needed.

    For the Luna family this restricts routing to OpenAI (an allowlist), so a request can never
    silently route off OpenAI's standard $0.10/M tier onto the ~2-11x Bedrock/Azure copies — nor
    onto any provider added later. Returns ``None`` for single-provider models (nothing to pin), so
    the caller omits the ``provider`` field entirely."""
    return _ROUTING.get(model_id)
