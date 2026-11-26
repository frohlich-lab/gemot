"""Go-to lead models for the deepagents panel, and the OpenRouter routing that pins the price.

OpenRouter serves a single model across SEVERAL providers at different prices. GPT-5.6 Luna, for
instance, is offered OpenAI-direct at ~$0.10/M prompt but ALSO via Amazon Bedrock ($0.22/M) and
Azure ($1.00-1.10/M) — up to ~11x more. Left to default routing a request can silently land on an
expensive copy, so every go-to model is pinned to its endpoints by an OpenRouter ``provider``
block: the Luna family to OpenAI (``only: [openai]``), its documented standard tier, never the
Bedrock/Azure copies. (We avoid ``sort: price``, which would opt into OpenAI's cheaper but
undocumented ``flex`` tier.)

**The aliases and pins are data, in a models file.** Which endpoint a model is held to is the
operator's policy -- this lab's is zero-data-retention endpoints only, fp8 or better, Anthropic
through its own key -- and endpoints come and go (six pinned tags were gone on 2026-11-07). In a
public package a pin hard-coded here is a 404 for every user until a release, and a new model is a
release too. So ``models.toml`` beside this module holds each alias's OpenRouter id and
``provider`` block, and the default lead, with the notes on how each pin was chosen. ``MODELS``,
``_ROUTING`` and ``DEFAULT`` are read from it once, at import.

``GEMOT_MODELS`` names a file that REPLACES it whole. Deliberately not merged: a run's aliases and
pins are then the one file it names, never a mix of two that nobody wrote down. A sweep task
inherits the variable through sbatch's ``--export=ALL``, so set it, as an absolute path, before
submitting. Each run's ``result.json`` records what its requests asked for
(``openrouter_extra_body``) and which endpoints served them (``openrouter_endpoints``), whichever
file it read.

**What stays here is what a model's API needs**, the same for every operator. Four per-model
request tables, keyed by OpenRouter id and omitted from the request when a model has no entry:
``_REASONING`` (:func:`reasoning_config`) turns thinking on where it ships off, ``_CACHING``
(:func:`caching_config`) asks for prompt caching where the provider does not do it implicitly,
``_HEADERS`` (:func:`request_headers`) sends a beta header, and ``_RETURNS_REASONING``
(:func:`returns_reasoning`) names the models whose earlier reasoning goes back to them at all. A
model that needs one of these is a change here, not in a models file.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

#: Names a models file that replaces the packaged one whole.
MODELS_ENV = "GEMOT_MODELS"

#: The packaged models file: this lab's aliases, their pins and the default lead.
PACKAGED_MODELS = Path(__file__).with_name("models.toml")


def load_models(path: Path) -> tuple[dict[str, str], dict[str, dict], str]:
    """Read a models file into its aliases (alias -> OpenRouter id), its routing (OpenRouter id ->
    ``provider`` block) and its default alias.

    Strict about its own keys and nothing else: a misspelt ``provder`` would otherwise drop a pin,
    and an unpinned request still succeeds -- the bill is where it would surface. The ``provider``
    block itself is OpenRouter's and passes through as written. One id takes one pin, so two
    aliases for the same id are refused rather than one pin silently winning."""
    with path.open("rb") as fh:
        table = tomllib.load(fh)
    if unknown := set(table) - {"default", "models"}:
        raise ValueError(f"{path}: unknown top-level key(s) {sorted(unknown)}")
    aliases: dict[str, str] = {}
    routing: dict[str, dict] = {}
    for alias, entry in table.get("models", {}).items():
        if not isinstance(entry, dict) or "id" not in entry:
            raise ValueError(f"{path}: [models.{alias}] needs an `id`")
        if unknown := set(entry) - {"id", "provider"}:
            raise ValueError(f"{path}: [models.{alias}] has unknown key(s) {sorted(unknown)}")
        model_id = entry["id"]
        if model_id in aliases.values():
            raise ValueError(f"{path}: {model_id} has two aliases; one id takes one pin")
        aliases[alias] = model_id
        if "provider" in entry:
            routing[model_id] = entry["provider"]
    default = table.get("default")
    if default not in aliases:
        raise ValueError(f"{path}: default {default!r} is not one of its models")
    return aliases, routing, default


#: The models file this process read: ``GEMOT_MODELS`` when set, else the packaged one.
MODELS_FILE = Path(os.environ.get(MODELS_ENV) or PACKAGED_MODELS)

#: ``MODELS`` maps an alias to its OpenRouter id so ``--model luna`` works (:func:`resolve`),
#: ``_ROUTING`` an OpenRouter id to its ``provider`` block (:func:`provider_routing`), and
#: ``DEFAULT`` is the lead when ``--model`` is not given.
MODELS, _ROUTING, DEFAULT = load_models(MODELS_FILE)


#: Per-model OpenRouter ``reasoning`` config, for models whose thinking is OFF by default.
#:
#: Empty today — see the Flash-Lite note below for the one model this was built for, and what
#: happened when it was tried. The mechanism is kept because some models ship with thinking OFF
#: and a benchmark that never enables it is measuring the wrong thing. Verified live that setting
#: `reasoning` yields non-zero `reasoning_tokens` AND preserves tool calls (a reasoning mode that
#: broke tool use would be worse than none). Reasoning tokens bill as COMPLETION tokens: ~250 per
#: call measured, negligible per run but worth watching on a long one.
_REASONING: dict[str, dict] = {
    # A run at --reasoning-effort default: the effort these think at, a share of max_tokens (set
    # PETAB_MCP_MAX_OUTPUT_TOKENS to their 128000). Any other --reasoning-effort replaces it.
    "anthropic/claude-haiku-5.5": {"effort": "high"},
    "anthropic/claude-sonnet-5.5": {"effort": "high"},
}


#: Per-model request headers. Haiku 4.5 thinks between tool calls only under Anthropic's
#: interleaved-thinking beta, which OpenRouter forwards: without it Haiku reasoned on the first call
#: of each turn and on none of the 1,466 calls that followed a tool result (October 2026 sweep, 6%
#: of calls reasoning). Sonnet 4.6 and later interleave on their own (adaptive thinking); kept for
#: Haiku 5.5 until a sweep shows it does too.
_HEADERS: dict[str, dict[str, str]] = {
    "anthropic/claude-haiku-5.5": {"x-anthropic-beta": "interleaved-thinking-2025-05-14"},
}


def request_headers(model_id: str) -> dict[str, str] | None:
    """Extra HTTP headers for ``model_id``'s requests, or ``None``."""
    return _HEADERS.get(model_id)


def reasoning_config(model_id: str) -> dict | None:
    """The OpenRouter ``reasoning`` block for ``model_id``, or ``None`` to leave it at the
    provider default (which for most models already means thinking is on)."""
    return _REASONING.get(model_id)


#: Per-model ``cache_control`` block for a provider that caches nothing unless asked, placed as
#: message breakpoints by ``driver._apply_cache_breakpoints``. See docs/RATIONALE.md.
_CACHING: dict[str, dict] = {
    "anthropic/claude-sonnet-5.5": {"type": "ephemeral", "ttl": "1h"},
    "anthropic/claude-opus-5.5": {"type": "ephemeral", "ttl": "1h"},
    "anthropic/claude-haiku-5.5": {"type": "ephemeral", "ttl": "1h"},
    "anthropic/claude-fable-5.2": {"type": "ephemeral", "ttl": "1h"},
}


def caching_config(model_id: str) -> dict | None:
    """The ``cache_control`` block for ``model_id``, or ``None`` where the provider caches on its
    own — which is not "do not cache". Do not add an entry for one of those: Gemini bills cache
    writes only against explicit breakpoints, so it would start charging for what is free."""
    return _CACHING.get(model_id)


#: Models whose earlier reasoning a request carries back (``driver._return_reasoning``): the
#: ``reasoning_details`` of each assistant message since the last plain user message, the tool-call
#: loop in progress. Anthropic counts that loop as one assistant turn and requires its thinking back
#: ("Required: within a tool-use turn, pass thinking blocks back", Claude docs, Preserving thinking
#: blocks); without it Haiku 4.5 reasoned on 0 of 1,466 calls after a tool result. In a gemot run
#: the lead's turn ends only at a bounce nudge or a thinking hand-off, so these keep nearly all of it,
#: append-only, which leaves their prompt cache intact.
#:
#: Every other model gets NONE back, deliberately. Carrying it to all of them was gemot#76: GLM 5.3
#: Flash at effort max thought up to 75k tokens a call, and earlier turns' reasoning was 25-37% of
#: every prompt on perica_egf (31-65 calls a run). No other provider asks for it: OpenRouter calls it
#: useful for tool calling, its own Gemini 3 tool-calling example sends the tool-call turn back bare,
#: and Gemini ran that way here from 2026-08-10 until the carry-back landed on 2026-10-17. Opt a
#: model in only on a measured gain.
_RETURNS_REASONING: frozenset[str] = frozenset(
    {
        "anthropic/claude-sonnet-5.5",
        "anthropic/claude-opus-5.5",
        "anthropic/claude-haiku-5.5",
        "anthropic/claude-fable-5.2",
    }
)


def returns_reasoning(model_id: str) -> bool:
    """Whether ``model_id``'s requests carry back the reasoning of the tool-use turn in progress."""
    return model_id in _RETURNS_REASONING


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
    """The OpenRouter ``provider`` block for ``model_id`` in the models file, or ``None``.

    For the Luna family this restricts routing to OpenAI (an allowlist), so a request can never
    silently route off OpenAI's standard $0.10/M tier onto the ~2-11x Bedrock/Azure copies — nor
    onto any provider added later.

    EVERY model in the packaged file is pinned, single-provider ones included
    (``test_every_panel_model_is_pinned`` enforces it): the pin on a model with one endpoint today,
    like ``z-ai/glm-5.3-flash``, is a standing guard against a second provider appearing later and
    silently taking the traffic at a different price or data policy. ``None`` therefore means "not
    pinned by the file this process read", and the caller omits the ``provider`` field for it."""
    return _ROUTING.get(model_id)
