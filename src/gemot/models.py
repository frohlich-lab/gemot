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
#:
#: Every entry here is PINNED (see ``_ROUTING``), so each price is the price of the endpoint the
#: pin actually reaches — NOT the catalog's default-route price, which for several of these is a
#: different number entirely (deepseek v4-flash lists $0.14 but DigitalOcean serves it at $0.084).
#: v4-pro and minimax-m3 are the two where the pin costs MORE than the catalog suggests: their
#: cheapest endpoints are not available under a zero-data-retention policy, so the reachable
#: floor is Novita's, and both are ~$0.37-0.39 per run rather than the ~$0.15 the headline
#: prompt price implies. Prices are documentation only — nothing computes with them.
MODELS: dict[str, tuple[str, float, float]] = {
    "luna":       ("openai/gpt-5.6-luna",         0.10, 0.60),
    "luna-pro":   ("openai/gpt-5.6-luna-pro",     0.10, 0.60),
    "v4-flash":   ("deepseek/deepseek-v4-flash",  0.084, 0.168),
    "v4-pro":     ("deepseek/deepseek-v4-pro",    0.435, 0.87),
    "minimax-m3": ("minimax/minimax-m3",          0.30, 1.20),
    "mimo":       ("xiaomi/mimo-v2.5",            0.168, 0.336),
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
#: opt-in.
#:
#: DeepSeek v4-flash needs pinning for a DIFFERENT reason, and the earlier claim here that it was
#: single-provider was simply wrong — OpenRouter serves it from 20 endpoints. What matters is not
#: the prompt price but the CACHE-READ price: this workload is ~97% prompt tokens and the driver
#: runs no history trimmer because the resent history is ~98% near-free ``cache_read`` (see
#: ``agent/driver.py``). Cache-read varies far more across providers than prompt price does —
#: $0.003/M to $0.078/M, a 26x spread, against 2.9x on prompt — so a route chosen on prompt price
#: can still be an order of magnitude wrong on the number that actually bills. Effective cost at
#: our median run (3.63M prompt / 125k completion) ranges $0.055 to $0.145 across endpoints.
#:
#: DeepSeek's own endpoint is the cheapest ($0.055/run, the only one with implicit caching) but is
#: NOT available under a zero-data-retention policy — verified live: ``only: [deepseek]`` returns
#: 404 "No endpoints available matching your ... data policy". Of the ZDR-eligible ones we exclude
#: DeepInfra despite it being cheapest ($0.093/run): it serves **fp4**, and comparing a benchmark
#: across differently-quantized endpoints confounds the result the benchmark exists to measure.
#: That leaves DigitalOcean ($0.087/run) and Novita (fp8, $0.145/run).
#:
#: ``order`` sets the priority, ``only`` bounds the fallback to the same two — ``order`` alone
#: still permits fallback to any other provider. Verified live: three consecutive calls all served
#: by DigitalOcean, with Novita as the standby if it goes down. AkashML is deliberately absent —
#: it serves the date-pinned ``-0731`` id but not this floating one.
_ROUTING: dict[str, dict] = {
    "openai/gpt-5.6-luna":      {"only": ["openai"]},
    "openai/gpt-5.6-luna-pro":  {"only": ["openai"]},
    "deepseek/deepseek-v4-flash": {"order": ["digitalocean", "novita"],
                                   "only":  ["digitalocean", "novita"]},
    # mimo: the $0.003-cache-read endpoints at Xiaomi/GMICloud are ZDR-blocked, so Novita is the
    # cheapest reachable one ($0.066/run) and still carries that cache rate. DeepInfra is excluded
    # for CONTEXT, not price: it serves this model at 262,144 tokens, and driver.py runs no history
    # trimmer on the assumption of a 1M window — a run that landed there could wedge on context.
    "xiaomi/mimo-v2.5": {"order": ["novita/fp8", "parasail/fp8"],
                         "only":  ["novita/fp8", "parasail/fp8"]},
    # v4-pro: Baidu/StreamLake/DeepSeek serve it at ~$0.15/run but are all ZDR-blocked, so Novita
    # ($0.374) is the floor we can actually reach — the 120x cache discount is NOT available here.
    "deepseek/deepseek-v4-pro": {"order": ["novita/fp8", "parasail/fp8"],
                                 "only":  ["novita/fp8", "parasail/fp8"]},
    # minimax-m3: pinned for CONTEXT as much as price. Five of its nine endpoints serve 524,288
    # tokens and Morph 256,000; only GMICloud/Novita/Parasail reach 1M, and GMICloud is ZDR-blocked.
    # Default routing lands on Venice (524K), so this pin closes a live context exposure. The 1M
    # pair leads; DeepInfra/Together (524K) are standbys only, so an outage degrades the sweep
    # instead of stalling it. 524K is deliberate headroom, not a guess — the largest single call
    # across 210 archived runs was 171k tokens, and the run-level cap is the recursion limit, not
    # the window. Cost is not the reason for the order: the 524K endpoints save ~$0.03/run.
    "minimax/minimax-m3": {"order": ["novita/fp8", "parasail/fp8", "deepinfra/fp8", "together"],
                           "only":  ["novita/fp8", "parasail/fp8", "deepinfra/fp8", "together"]},
}


#: Per-model OpenRouter ``reasoning`` config, for models whose thinking is OFF by default.
#:
#: Empty today — see the Flash-Lite note below for the one model this was built for, and what
#: happened when it was tried. The mechanism is kept because some models ship with thinking OFF
#: and a benchmark that never enables it is measuring the wrong thing. Verified live that setting
#: `reasoning` yields non-zero `reasoning_tokens` AND preserves tool calls (a reasoning mode that
#: broke tool use would be worse than none). Reasoning tokens bill as COMPLETION tokens: ~250 per
#: call measured, negligible per run but worth watching on a long one.
_REASONING: dict[str, dict] = {}


def reasoning_config(model_id: str) -> dict | None:
    """The OpenRouter ``reasoning`` block for ``model_id``, or ``None`` to leave it at the
    provider default (which for most models already means thinking is on)."""
    return _REASONING.get(model_id)


# ---------------------------------------------------------------- why gemini-2.5-flash-lite is out
#
# It was added to the panel in August 2026 and removed the same day. Recorded here because the
# work to qualify it was real and someone will otherwise redo it: on paper it is a strong
# candidate — ZDR-viable, tool-capable, 1,048,576-token context on every Google endpoint, and
# $0.093/run by the cache-read arithmetic that governs this workload.
#
# Routing, if it is ever revisited: `google-ai-studio` is ZDR-BLOCKED (verified live, 404 on data
# policy); `google-vertex` and `google-vertex/eu` are not. Pin the bare tags — they exclude
# `google-ai-studio/flex` ($0.046/run), the same undocumented-cheap-tier trap we refuse for Luna.
# Verified live: 3/3 calls served standard.
#
# It was dropped on measured behaviour, not on price. Across 7 runs on perelson/blasi/rahman it
# scored 0, against mimo's 3 of 4 on the same examples, and cost 4.3x more per run ($0.42 vs
# $0.099) because a failing run now burns its whole recursion budget.
#
# Two rounds, because the first failure was ours and had to be excluded before judging the model:
#   1. Thinking OFF (its default). It degenerated — 42 of 75 `make_changes` calls named a
#      parameter it never declared, the name growing suffixes (`k_burst_rate_final_final_attempt`)
#      instead of reading the rejection, which lists the ids that exist. Edit reject rate 1.00.
#   2. Thinking ON (`reasoning: {effort: high}`). This genuinely helped: reject rate 1.00 -> 0.13,
#      median dBIC 629 -> 70, no more name degeneration. It still never produced a usable model.
#      The failure underneath is that it cannot tell it has ALREADY tried a structure: 112 of its
#      accepted edits came back `no_change` ("this structure already IS a version"), and it
#      re-authored a variant anyway. One run: 48 make_changes, 40 accepted, 2 fits.
#
# That last mode is also why it always ended at the recursion limit rather than being stopped
# earlier — a `no_change` op produces no new version and no new BIC, so the early-stop chain
# accumulates no evidence of a plateau. Worth fixing on its own merits (it is the wasted-iteration
# class from FINDINGS.md #5-#7), but it is not what made this model unusable.


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
