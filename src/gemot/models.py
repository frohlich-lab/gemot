"""Go-to lead models for the deepagents panel, and the OpenRouter routing that pins the price.

OpenRouter serves a single model across SEVERAL providers at different prices. GPT-5.6 Luna, for
instance, is offered OpenAI-direct at ~$0.10/M prompt but ALSO via Amazon Bedrock ($0.22/M) and
Azure ($1.00-1.10/M) — up to ~11x more. Left to default routing a request can silently land on an
expensive copy, so the Luna family is pinned to OpenAI via provider routing (``only: [openai]``):
it serves OpenAI's documented standard tier, never the Bedrock/Azure copies. (We avoid
``sort: price``, which would opt into OpenAI's cheaper but undocumented ``flex`` tier.)

``MODELS`` maps a short alias to its OpenRouter id so ``--model luna`` works; :func:`resolve`
passes a full id through unchanged.

Three per-model request tables live here, each keyed by OpenRouter id and omitted from the request
when a model has no entry: ``_ROUTING`` (:func:`provider_routing`) pins the endpoint,
``_REASONING`` (:func:`reasoning_config`) turns thinking on where it ships off, and ``_CACHING``
(:func:`caching_config`) asks for prompt caching where the provider does not do it implicitly.
"""

from __future__ import annotations

#: alias -> OpenRouter id. The panel's go-to leads, every one pinned to an endpoint (``_ROUTING``).
MODELS: dict[str, str] = {
    "luna": "openai/gpt-5.6-luna",
    "luna-pro": "openai/gpt-5.6-luna-pro",
    "v4-flash": "deepseek/deepseek-v4-flash",
    "v4-pro": "deepseek/deepseek-v4-pro",
    "minimax-m3": "minimax/minimax-m3",
    "mimo": "xiaomi/mimo-v2.5",
    "glm-flash": "z-ai/glm-5.3-flash",
    # The tier above the panel: ~10x luna per run, added to test whether capability buys
    # anything on the examples the panel cannot solve. Not defaults -- opt in with --models.
    "terra": "openai/gpt-5.6-terra",
    "astra": "openai/gpt-6-astra",
    "grok": "x-ai/grok-4.20",
    "gemini-flash": "google/gemini-3.6-flash",
    "gemini-pro": "google/gemini-3.1-pro-preview",
    # Anthropic, also opt-in: the only models whose caching is not implicit (``_CACHING``).
    "sonnet": "anthropic/claude-sonnet-5",
    "opus": "anthropic/claude-opus-5",
    "haiku": "anthropic/claude-haiku-4.5",
}

#: Default lead. 1M-token context, single pinned provider (Z.AI), and the best measured accuracy
#: on this benchmark: the 2026-08-28/29 campaign scored 71/78 (91%) across 18 examples, against
#: luna's 55/90 (61%) on the same post-fix harness. It is also the only model to have passed
#: bachmann (3/5) and lucarelli (4/4) on their current versions, both flat 0 for every other
#: model tested. Headline prices are 25-58% below luna's, but cost parity is NOT the reason and
#: is NOT established — cache reads are $0.015/M against luna's $0.010, and on a workload that is
#: ~89% cache reads that one rate decides the comparison (see the routing note below).
#:
#: The accuracy figure depends on the reasoning-runaway guard: unguarded, this model scored 0/15
#: on bachmann/isensee/boehm because it burned whole completion budgets thinking and returned
#: nothing. Do not roll the default forward onto a harness without agent/guards.py.
DEFAULT = "glm-flash"

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
    # glm-flash: ONE provider (Z.AI itself), so the pin is not about price mirrors -- it is a
    # standing guarantee that a provider added later cannot silently take the traffic. Verified
    # live 2026-08-28: `only: ["z-ai"]` returns HTTP 200 served by Z.AI, so unlike deepseek/mimo
    # the cheap endpoint IS reachable under this account's data policy. Context is 1M+, clearing
    # driver.py's 1M assumption with no fallback endpoint to worry about.
    #
    # Cache read is $0.015/M -- 1.5x luna's $0.010 -- and on this workload cache reads are ~89% of
    # prompt tokens, so that one rate decides the comparison: measured over the 66-run luna
    # campaign, GLM lands ~17% cheaper ($0.033 vs $0.038/run median) despite headline prices 25%
    # and 58% below luna's. It needs a >=78% cache hit rate just to break even, and can only
    # afford 1.16x luna's token consumption before it costs MORE. Treat cost parity as unproven
    # until a real sweep reports its own cache_read_tokens.
    "z-ai/glm-5.3-flash": {"only": ["z-ai"]},
    "openai/gpt-5.6-luna": {"only": ["openai"]},
    "openai/gpt-5.6-luna-pro": {"only": ["openai"]},
    "deepseek/deepseek-v4-flash": {
        "order": ["digitalocean", "novita"],
        "only": ["digitalocean", "novita"],
    },
    # mimo: the $0.003-cache-read endpoints at Xiaomi/GMICloud are ZDR-blocked, so Novita is the
    # cheapest reachable one ($0.066/run) and still carries that cache rate. DeepInfra is excluded
    # for CONTEXT, not price: it serves this model at 262,144 tokens, and driver.py runs no history
    # trimmer on the assumption of a 1M window — a run that landed there could wedge on context.
    "xiaomi/mimo-v2.5": {
        "order": ["novita/fp8", "parasail/fp8"],
        "only": ["novita/fp8", "parasail/fp8"],
    },
    # v4-pro: Baidu/StreamLake/DeepSeek serve it at ~$0.15/run but are all ZDR-blocked, so Novita
    # ($0.374) is the floor we can actually reach — the 120x cache discount is NOT available here.
    "deepseek/deepseek-v4-pro": {
        "order": ["novita/fp8", "parasail/fp8"],
        "only": ["novita/fp8", "parasail/fp8"],
    },
    # minimax-m3: pinned for CONTEXT as much as price. Five of its nine endpoints serve 524,288
    # tokens and Morph 256,000; only GMICloud/Novita/Parasail reach 1M, and GMICloud is ZDR-blocked.
    # Default routing lands on Venice (524K), so this pin closes a live context exposure. The 1M
    # pair leads; DeepInfra/Together (524K) are standbys only, so an outage degrades the sweep
    # instead of stalling it. 524K is deliberate headroom, not a guess — the largest single call
    # across 210 archived runs was 171k tokens, and the run-level cap is the recursion limit, not
    # the window. Cost is not the reason for the order: the 524K endpoints save ~$0.03/run.
    "minimax/minimax-m3": {
        "order": ["novita/fp8", "parasail/fp8", "deepinfra/fp8", "together"],
        "only": ["novita/fp8", "parasail/fp8", "deepinfra/fp8", "together"],
    },
    # terra: same Azure trap as the rest of the OpenAI family (OpenAI $1.00 vs Azure $2.75, a
    # 5.5x spread), so the bare `openai` tag is mandatory -- it also excludes `openai/flex` and
    # `openai/priority`, the undocumented cheap and dear tiers we refuse everywhere else.
    "openai/gpt-5.6-terra": {"only": ["openai"]},
    # astra: the family's standing `openai` tag, which also excludes `openai/flex` (half price,
    # undocumented tier), `openai/fast` (2x) and the Azure mirrors. Verified live: `openai` and
    # `azure` both answer for this account at the same price, so the pin fixes the endpoint
    # rather than the bill.
    "openai/gpt-6-astra": {"only": ["openai"]},
    # grok: `xai/zdr` and `xai` are the SAME price ($1.25/$0.20/$2.50); pin the zdr tag because it
    # names the data guarantee in the request rather than leaning on the account policy, and it
    # excludes the `/priority` tiers which cost exactly double for no benefit here.
    "x-ai/grok-4.20": {"order": ["xai/zdr", "xai"], "only": ["xai/zdr", "xai"]},
    # Gemini: READ THIS BEFORE ADDING ANOTHER GOOGLE MODEL. Google publishes no ZDR-tagged endpoint
    # at all -- grok's `xai/zdr` has no counterpart here, and `data_policy` comes back `{}` for
    # every endpoint (terra's included), so the API cannot answer the retention question either
    # way. These two are reachable only because ZDR enforcement is relaxed account-side, the same
    # concession that admitted terra. That is a policy decision, not a property of the pin.
    #
    # All six endpoints are byte-identical on price and context (1M, so no context trap of the kind
    # mimo and m3 have). The standby is the OTHER TWO VERTEX tags, not `google-ai-studio`: the
    # Flash-Lite note below records `google-ai-studio` as ZDR-BLOCKED (verified live, 404 on data
    # policy) while `google-vertex` and `google-vertex/eu` are not. Pinning AI Studio as the fallback
    # -- as this did -- meant a `google-vertex/global` outage left an effectively EMPTY allowlist, so
    # every gemini task in the array job would error out instead of degrading, which is the opposite
    # of what the fallback is for. flex is half price and priority 1.8x, and is excluded by the same
    # standing rule we apply to `openai/flex`.
    #
    # Unlike every incumbent, Gemini prices cache WRITES ($0.083/M flash, $0.375/M pro). That rate
    # bills only for explicit `cache_control` breakpoints; we set none anywhere in the harness, so
    # implicit caching applies and the write column stays at zero. Adding cache_control later would
    # start billing it -- on pro the write rate is 1.9x its own cache-read rate.
    #
    # No _REASONING entry on purpose: both models think by DEFAULT (verified live, 140-210
    # reasoning tokens on a one-shot tool call, tool calls intact). Contrast Flash-Lite below,
    # where thinking was off and had to be asked for.
    "google/gemini-3.6-flash": {
        "order": ["google-vertex/global", "google-vertex", "google-vertex/eu"],
        "only": ["google-vertex/global", "google-vertex", "google-vertex/eu"],
    },
    "google/gemini-3.1-pro-preview": {
        "order": ["google-vertex/global", "google-vertex", "google-vertex/eu"],
        "only": ["google-vertex/global", "google-vertex", "google-vertex/eu"],
    },
    # Anthropic: one tag, no `order` fallback -- pinned for the cache namespace, not the price.
    # `anthropic`, `claude-on-aws` and `azure/*` are ZDR-blocked here (404, verified per tag).
    # Through this account's own Anthropic key (BYOK), which the ZDR block above does not apply to.
    "anthropic/claude-sonnet-5": {"only": ["anthropic"]},
    "anthropic/claude-opus-5": {"only": ["google-vertex/global"]},
    # The Anthropic endpoint, through this account's own key (BYOK).
    "anthropic/claude-haiku-4.5": {"only": ["anthropic"]},
}


#: Per-model OpenRouter ``reasoning`` config, for models whose thinking is OFF by default.
#:
#: Empty today — see the Flash-Lite note below for the one model this was built for, and what
#: happened when it was tried. The mechanism is kept because some models ship with thinking OFF
#: and a benchmark that never enables it is measuring the wrong thing. Verified live that setting
#: `reasoning` yields non-zero `reasoning_tokens` AND preserves tool calls (a reasoning mode that
#: broke tool use would be worse than none). Reasoning tokens bill as COMPLETION tokens: ~250 per
#: call measured, negligible per run but worth watching on a long one.
_REASONING: dict[str, dict] = {
    # Haiku 4.5 thinks only when asked; "high" is OpenRouter's largest thinking budget, a share of
    # max_tokens (set PETAB_MCP_MAX_OUTPUT_TOKENS to Haiku's 64000). Thinking needs temperature 1.
    "anthropic/claude-haiku-4.5": {"effort": "high"},
    "anthropic/claude-sonnet-5": {"effort": "high"},
}


def reasoning_config(model_id: str) -> dict | None:
    """The OpenRouter ``reasoning`` block for ``model_id``, or ``None`` to leave it at the
    provider default (which for most models already means thinking is on)."""
    return _REASONING.get(model_id)


#: Per-model ``cache_control`` block for a provider that caches nothing unless asked, placed as
#: message breakpoints by ``driver._apply_cache_breakpoints``. See docs/RATIONALE.md.
_CACHING: dict[str, dict] = {
    "anthropic/claude-sonnet-5": {"type": "ephemeral", "ttl": "1h"},
    "anthropic/claude-opus-5": {"type": "ephemeral", "ttl": "1h"},
    "anthropic/claude-haiku-4.5": {"type": "ephemeral", "ttl": "1h"},
}


def caching_config(model_id: str) -> dict | None:
    """The ``cache_control`` block for ``model_id``, or ``None`` where the provider caches on its
    own — which is not "do not cache". Do not add an entry for one of those: Gemini bills cache
    writes only against explicit breakpoints, so it would start charging for what is free."""
    return _CACHING.get(model_id)


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
    return MODELS.get(name, name)


def provider_routing(model_id: str) -> dict | None:
    """The OpenRouter ``provider`` block for ``model_id``, or ``None`` when no pinning is needed.

    For the Luna family this restricts routing to OpenAI (an allowlist), so a request can never
    silently route off OpenAI's standard $0.10/M tier onto the ~2-11x Bedrock/Azure copies — nor
    onto any provider added later.

    EVERY panel model is pinned, single-provider ones included (``test_every_panel_model_is_pinned``
    enforces it): the pin on a model with one endpoint today, like ``z-ai/glm-5.3-flash``, is a
    standing guard against a second provider appearing later and silently taking the traffic at a
    different price or data policy. ``None`` therefore means "not a panel model", not "nothing to
    pin", and the caller omits the ``provider`` field for it."""
    return _ROUTING.get(model_id)
