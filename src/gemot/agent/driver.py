"""``run_deepagents`` — assemble the panel's graph, then drive it to a finish.

Everything the loop needs is built by this module's siblings (see the package docstring); what
lives here is the ASSEMBLY (curate the petab tools, discover the bio tools, wrap the breaker and
the literature recorder, build the subagents, attach the four middleware guards and the
checkpointer) and the RUN LOOP around ``astream``.

The loop is where a run's failure modes are absorbed rather than thrown away:

* a per-super-step **stall watchdog**, sized to the run's real fit budget so a long multistart
  is not mistaken for a hang;
* a **wall-clock deadline** and the server's own early-stop signal;
* **resume from the last checkpoint** on a transient provider failure, instead of abandoning the
  run;
* **bounced silent stops** — a graph that ends on a no-tool turn is pushed back into the finish
  gate (with the build nudge when nothing was authored at all), up to ``_MAX_SILENT_BOUNCES``;
* and an honest **relabel** at the end, so a degenerate run is reported as ``no_tools_executed``
  / ``no_model_built`` / ``silent_stop`` rather than as a "completed" one with nothing in it.

Returns the run's stop reason plus the accounting the runner writes into ``result.json``.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import re
import time
from typing import Any

from .biocurator import (
    DEFAULT_BIO_TOOLS,
    LITERATURE_BIO_TOOLS,
    _BIO_DISCOVERY_TIMEOUT_S,
    _build_biocurator_subagent,
)
from .circuit_breaker import (
    _disabled_tool_filter,
    _new_breaker_state,
    _wrap_tools_with_circuit_breaker,
    call_outcomes,
)
from .finish_gate import _finish_tool
from .guards import _empty_report_guard, _task_arg_guard, _tool_dispatch_guard
from .literature import _wrap_literature_recorder
from .probe import _model_committed, _structured
from .subagents import (
    _build_subagents,
    _builtin_tool_filter,
    _disable_general_purpose_subagent,
    _general_purpose_enabled,
)
from .token_usage import _token_accountant
from .tools import ORCHESTRATOR_TOOLS, _select_tools, _trim_descriptions
from ..prompts import _NO_MODEL_NUDGE, _SILENT_FINISH_NUDGE
from .. import models

# --- guards on the (input-dominated, occasionally-hanging) LLM loop -----------------------
# Transient provider failures (connection drops, malformed/non-JSON bodies, 5xx/429, hung
# calls) are common on OpenRouter and used to abandon the whole run. Now they RETRY instead of
# throwing the run away, in two layers:
#   * per-CALL: the OpenAI client's own ``max_retries`` retries connection errors, request
#     timeouts, and 5xx/429 (honouring Retry-After) transparently, so those never reach the loop.
#   * per-RUN resume: the agent is compiled with a LangGraph checkpointer, so anything that still
#     escapes — a malformed non-JSON body the client won't retry, exhausted call-retries, or a
#     STALL the client timeout can't catch — RESUMES ``astream`` from the last checkpoint instead
#     of aborting, so every committed model version and the conversation state are preserved.
_LLM_TIMEOUT_S = 120.0
_LLM_MAX_RETRIES = 4           # OpenAI-client retries for connection / timeout / 5xx / 429
_RUN_RESUME_ATTEMPTS = 3       # resume the graph from the last checkpoint after a stall/transient
# Harness-level stall watchdog on the agent loop. The client `timeout` above does NOT reliably
# catch a mid-stream provider stall (observed: a 24-min no-progress hang that the 120s timeout
# never fired on). astream emits once per graph super-step (a lead LLM call, a tool, or a whole
# analyst sub-agent), so if NO step completes within this window the loop is wedged — we cancel
# it and RESUME from the last checkpoint (or, once resume attempts run out, end gracefully).
# This is only the FLOOR for the non-fit steps: run_deepagents raises the effective window to the
# full multistart fit budget (fit_budget.py) for the run, since a lead commit's fit is one long
# super-step that legitimately streams nothing back for minutes.
_STEP_STALL_TIMEOUT_S = 600.0

# Hard per-run token ceiling — the ONLY bound on how much a run can spend.
#
# `recursion_limit` does not bound this and cannot be made to: it caps LangGraph SUPER-STEPS, but
# a deepagents subagent makes all of its LLM calls inside ONE parent super-step, so its calls are
# invisible to the limit. Observed: a run with recursion_limit=250 made 1,493 LLM calls and
# consumed 107.9M prompt tokens (~$51 on one model, in one run). Two multipliers compound it:
# every resume re-enters astream with a FRESH step budget (so the effective ceiling is
# _RUN_RESUME_ATTEMPTS + 1 times whatever you set), and each resume re-sends the whole history
# because there is no trimmer — a deliberate choice that is near-free at ~98% cache_read but
# full price on a weakly-caching model.
#
# 20M is ~1.5x the largest legitimate run measured across 339 runs (13.2M, luna-pro on schwen at
# 155 calls) and below all three observed runaways (24.4M / 43.4M / 107.9M). Counting TOTAL
# tokens rather than estimated dollars deliberately: the per-model prices in models.py are
# documentation and carry no cache rates, so a cost estimate would be wrong by up to 10x on
# exactly the cache-friendly models least in need of a guard.
_MAX_RUN_TOKENS = 20_000_000

# Provider content-policy refusals, matched on the response text because they arrive as a plain
# 400 with no distinguishing exception type: OpenAI/OpenRouter raise ValueError/BadRequestError
# carrying the provider's JSON. Kept deliberately NARROW — a stray 400 (a malformed request, a bad
# parameter) must NOT be relabelled as a content refusal, so a match needs an explicit
# moderation/policy phrase, not merely the status code.
_CONTENT_FILTER_MARKERS = (
    "flagged for possible biological risk",
    "content was flagged",
    "content_policy_violation",
    "content policy",
    "responsible_ai_policy",
    "flagged by the moderation",
    "moderation_blocked",
)


def _content_filter_stop_reason(exc: BaseException) -> str | None:
    """``content_filtered: <short reason>`` if ``exc`` is a provider content refusal, else None.

    The refusal is the model provider declining the request, not a harness or model failure, and
    it is not retriable — so the caller uses this to end the run with a countable outcome rather
    than a per-run-unique exception string.
    """
    text = str(exc)
    low = text.lower()
    if not any(m in low for m in _CONTENT_FILTER_MARKERS):
        return None
    # Surface the provider's own sentence, not the whole JSON envelope. The closing quote must
    # match the one that OPENED the message -- taking whichever of `',` / `",` comes first would
    # cut at a quote belonging to the message body.
    for opener, closer in (("'message': '", "'"), ('"message": "', '"')):
        i = text.find(opener)
        if i == -1:
            continue
        rest = text[i + len(opener):]
        # The message ends at the first closing quote that is followed by the JSON punctuation
        # that can legally follow a value. `rfind` would run past the message and land on the
        # quote of a LATER key (`'code'`); a plain `find` would stop at a quote inside the
        # message body. Both were wrong in earlier versions of this function.
        end = -1
        for punct in (",", "}"):
            j = rest.find(closer + punct)
            if j != -1 and (end == -1 or j < end):
                end = j
        detail = (rest[:end] if end != -1 else rest)[:200]
        return f"content_filtered: {detail.strip()}"
    return f"content_filtered: {text.strip('{} ')[:200]}"


def _extra_body(model: str) -> dict[str, Any]:
    """OpenRouter-specific request fields: provider pinning, and reasoning where it is not on
    by default. Both are omitted entirely when unset, so a model with neither sends a bare
    request exactly as before."""
    body: dict[str, Any] = {}
    if (routing := models.provider_routing(model)) is not None:
        body["provider"] = routing
    if (reasoning := models.reasoning_config(model)) is not None:
        body["reasoning"] = reasoning
    return body


@functools.lru_cache(maxsize=None)
def _chat_model_class():
    """``ChatOpenAI``, subclassed to keep the UPSTREAM endpoint identity OpenRouter reports.

    OpenRouter returns a top-level ``provider`` naming the endpoint that actually served the call
    (``"OpenAI"``, ``"Azure"``, ``"Novita"``, …). ``langchain_openai`` drops it: its ``llm_output``
    hard-codes ``model_provider: "openai"`` — the name of the INTEGRATION, not of the endpoint —
    so the served endpoint is invisible downstream. That is the one field an A/B needs, because
    ``provider.only`` bounds routing to a SET and every member of that set is a different
    machine: an arm can drift across endpoints mid-sweep and nothing in the record would say so.

    ``response.model_dump()`` already carries ``provider`` (pydantic keeps unknown top-level
    fields as extras), so this only copies it where the callbacks can read it. Everything else is
    ``super()``'s. Best-effort by construction: any failure leaves the unmodified result, since
    provenance capture must never break a run.

    ``lru_cache``d, and the caller passes ``name="ChatOpenAI"`` -- both for reasons that bite
    silently. LangChain derives a run's name (and openinference the SPAN name) from the class name,
    so an un-named subclass would rename every LLM span to ``_ChatOpenAIRecordingProvider``: every
    archived trace and every span query under ``slurm/trace_review`` keys on ``ChatOpenAI``, so the
    new runs would simply return no rows and could not be compared with the old ones. The cache is
    because each call would otherwise build -- and permanently register -- a fresh pydantic model
    class. Keeping the import inside the function is what ``tests/test_import_lightness`` needs.
    """
    from langchain_openai import ChatOpenAI

    class _ChatOpenAIRecordingProvider(ChatOpenAI):
        def _create_chat_result(self, response, generation_info=None):
            result = super()._create_chat_result(response, generation_info)
            try:
                served = getattr(response, "provider", None)
                if served is None and isinstance(response, dict):
                    served = response.get("provider")
                if not served:
                    return result
                # Both places: `llm_output` is what on_llm_end sees as an aggregate, and the
                # per-message metadata is what Phoenix exports onto the LLM span.
                if isinstance(result.llm_output, dict):
                    result.llm_output["openrouter_provider"] = served
                for gen in result.generations:
                    meta = getattr(getattr(gen, "message", None), "response_metadata", None)
                    if isinstance(meta, dict):
                        meta["openrouter_provider"] = served
            except Exception:  # pragma: no cover - provenance must never break a run
                pass
            return result

    return _ChatOpenAIRecordingProvider


# A provider 5xx that arrives UNCLASSIFIED. OpenRouter returns HTTP 200 with an `{"error": ...}`
# body for some upstream failures, so the OpenAI client raises a bare ValueError instead of an
# InternalServerError — which means such an error is invisible to BOTH retry layers: the client's
# own `max_retries` (it never saw a 5xx status) and `_TRANSIENT_ERRORS` (ValueError is not in it).
# It therefore reached the terminal handler and ended the run. Observed twice in 839 archived
# runs, each time on a HEALTHY run: median 5 fits, 62 tool calls and 25 minutes of work already
# done, with a model committed. The resume-from-checkpoint machinery already exists and preserves
# all of that; this classifier is only what lets the error reach it.
#
# Matched on a 5xx CODE, never on wording: a `ValueError` carrying a 400 is deterministic (a
# content refusal, a malformed request) and must keep falling through to the terminal path, or
# every content refusal would be retried three times before failing anyway.
_PROVIDER_CODE_RE = re.compile(r"['\"]code['\"]\s*:\s*(\d{3})")


def _transient_provider_reason(exc: BaseException) -> str | None:
    """A short reason if ``exc`` is an unclassified provider 5xx worth RESUMING for, else None."""
    m = _PROVIDER_CODE_RE.search(str(exc))
    if not m or not 500 <= int(m.group(1)) <= 599:
        return None
    return f"provider {m.group(1)}: {type(exc).__name__}: {str(exc)[:120]}"


# NOTE: the lead used to run a ContextEditingMiddleware (ClearToolUsesEdit) that cleared old
# tool results once its context passed ~90k tokens — a guard against an early failure mode where
# re-sent tool RESULTS ballooned a run to ~1.1M tokens and hanging calls. It was removed: all
# target models have a 1M-token window and the recursion limit already caps a run at ~120k tokens
# (a full crauste sweep of 17 runs topped out at 118k and the trimmer fired on only 2 of them,
# once each). With prompt caching the cleared history was ~98% near-free cache_read, while the
# clear itself invalidated the cached prefix — so trimming was net token-negative. The per-request
# timeout still backstops any slow call. Re-add a trimmer only if a model with a smaller window
# is introduced.


# A silent stop (the graph ends on a no-tool turn) is not an accepted way to finish: the driver
# bounces it back into the finish gate this many times before relenting so a genuinely stuck run
# still terminates (at the step/wall limit) rather than looping forever.
_MAX_SILENT_BOUNCES = 3


async def run_deepagents(
    session,
    *,
    model: str,
    base_url: str,
    api_key: str,
    system_prompt: str,
    task: str,
    recursion_limit: int,
    temperature: float = 0.0,
    n_starts: int = 1,
    max_wall_seconds: float | None = None,
    max_run_tokens: int | None = _MAX_RUN_TOKENS,
    progress_probe: Any | None = None,
    bio_sessions: Any | None = None,
    bio_literature: bool = False,
    curated_entities: "list[str] | None" = None,
) -> dict[str, Any]:
    """Drive the orchestrator (+ analyst subagents) against ``session``'s tools.

    ``max_wall_seconds`` (optional) is a self-imposed total-runtime budget, checked between
    graph super-steps: once exceeded, the loop stops gracefully (like running out of resume
    attempts) so ``_judge`` can score whatever is currently committed, INSTEAD of racing an
    outer resource limit (e.g. a Slurm ``--time`` allocation) that would hard-kill the process
    with no chance to write a result at all. A sweep should size this comfortably BELOW its
    Slurm time limit (see ``slurm/make_sweep.py``'s ``--max-wall-seconds`` / ``--time``) so this
    fires first. ``None`` (the default) means no self-imposed limit — only the per-step stall
    watchdog and the outer scheduler's own limit apply, as before.

    ``progress_probe`` (optional) is an un-timed ``await probe(tool, args)`` handle used to poll
    the server's ``get_progress`` between super-steps: when the early-stop chain reports
    ``should_stop`` (the non-improving-streak cap, or a plateau once the best model is adequate),
    the loop stops at the next clean boundary — the deterministic replacement for wandering to the
    recursion limit. It is passed un-timed so these internal polls don't inflate the tool clock /
    ``tool_calls`` metric. ``None`` disables the check (the loop runs as before).

    The model is served by OpenRouter via its OpenAI-compatible API (``base_url`` +
    ``api_key``). Returns ``{"stop_reason": ..., "n_tool_errors": ..., "disabled_tools": ...,
    **guard_stats}`` — the stop reason, the circuit breaker's error tallies, and the guard
    counters (``tools_executed``, ``dropped_dup_tool_calls``, ``idle_nudges``, ``gave_up_no_tools``,
    and the empty-report guard's ``empty_subagent_reports`` / ``empty_subagent_retries`` /
    ``empty_subagent_markers``); the caller reads ``stop_reason`` and may ignore the
    rest. The orchestrator's messages, its ``task`` delegations, the analysts' private tool
    calls, and token usage are recorded by Phoenix tracing (set up by the runner), so this
    function just drives the loop to completion.
    """
    from deepagents import create_deep_agent
    from langchain_mcp_adapters.tools import load_mcp_tools

    # Adapt the MCP tools bound to this already-initialized session, then curate to the
    # union of the lead's + analysts' tools. This pool is only used to resolve each analyst's
    # tool objects; it is NOT what the lead gets (see lead_tools below).
    petab_tools = _trim_descriptions(_select_tools(await load_mcp_tools(session)))
    # Blinded bio-knowledge tools from one or more SECONDARY MCP servers (biocurator joins the
    # panel on every deepagents run; endpoints hardcoded in runner.BIO_MCP_URLS = BioContextAI + OmniPath;
    # the `bio_literature` ablation adds EuropePMC/bioRxiv). They are
    # RUNTIME-DISCOVERED (their names are unknown ahead of time), so they deliberately BYPASS
    # _select_tools/ALLOWED_TOOLS (which would warn+drop them as "unexpected") and are handed
    # ONLY to the biocurator subagent below — never to the lead or the other analysts. NOTE: the
    # runner's _ToolClock shadows only the PETAB session's call_tool, so bio-server latency is
    # counted as "LLM+overhead" in the metrics split rather than tool time (a known caveat now
    # that the biocurator runs by default; wrap each bio session's call_tool into the clock too
    # if that split matters).
    bio_tools: list = []
    if bio_sessions:
        petab_names = {n for t in petab_tools if (n := getattr(t, "name", None))}
        # The biocurator's tool selection: the curated DEFAULT_BIO_TOOLS (a lean, deliberate set
        # spanning the default BioContextAI server + OmniPath: identity/domains/interactions/
        # pathways/GO/ontology + OmniPath's SQL tool), plus the literature tools when the operator
        # opted in with PETAB_BIO_LITERATURE=1. It is an allowlist of NAMES, not a numeric cap, so it can
        # never drop the one tool the biocurator needs.
        allow = set(DEFAULT_BIO_TOOLS)
        if bio_literature:
            allow |= set(LITERATURE_BIO_TOOLS)
        # Discover tools across ALL bio servers, dropping any whose name COLLIDES with a petab tool
        # (would shadow it in tools_by_name for the lead/analysts) or DUPLICATES one already seen
        # from another bio server (keep the first). Bio tools keep their FULL server descriptions
        # (not _trim_descriptions, which is petab-oriented and would truncate usage/arg notes).
        discovered: list = []
        seen: set = set()
        for bs in bio_sessions:
            # Bounded + guarded: a bio server that is slow or whose (isolated) session has already
            # died must drop only its own tools, never abort the run (issue #86). A discovery
            # failure here is an ordinary Exception (the session is isolated in its own task, so it
            # never surfaces as a run-cancelling anyio group) — skip that server and continue.
            try:
                bs_tools = await asyncio.wait_for(load_mcp_tools(bs),
                                                  timeout=_BIO_DISCOVERY_TIMEOUT_S)
            except Exception as exc:
                print(f"[bio] WARNING: tool discovery on a bio server failed "
                      f"({type(exc).__name__}: {exc}); skipping that server's tools.")
                continue
            for t in bs_tools:
                nm = getattr(t, "name", None)
                if nm in petab_names:
                    print(f"[bio] WARNING: bio tool {nm!r} collides with a petab tool name; "
                          "dropping it so it cannot shadow the petab tool.")
                    continue
                if nm in seen:
                    print(f"[bio] duplicate bio tool {nm!r} from another bio server; keeping the "
                          "first.")
                    continue
                seen.add(nm)
                discovered.append((nm, t))
        names = [nm for nm, _ in discovered]
        if not discovered:
            print("[bio] the bio server(s) exposed no tools; the biocurator will run 'dumb' "
                  "(the model's own biology knowledge).")
        else:
            # No all-tools fallback: hand the biocurator ONLY the curated tools that actually
            # resolved on the connected server(s). If none match it still runs (below) — just
            # 'dumb' — and the empty biocurator_tools in result.json flags that for analysis.
            bio_tools = [t for nm, t in discovered if nm in allow]
            if not bio_tools:
                shown = f"{names[:12]}{' …' if len(names) > 12 else ''}"
                print(f"[bio] none of the curated tool names matched the connected server(s)' "
                      f"{len(names)} tools ({shown}); the biocurator will run 'dumb'.")
    all_tools = petab_tools + bio_tools
    # Wrap the tool objects with the repeated-error circuit breaker BEFORE they are handed to
    # the lead and the analyst subagents (both reference these same objects), so a tool that
    # keeps failing identically is disabled for the run instead of being re-proposed endlessly.
    breaker_state = _new_breaker_state()
    # Only the EXTERNAL bio tools (flaky hosted servers) get the immediate transport-error disable;
    # a transport blip on a core petab tool must NOT silently strip it for the rest of the run.
    _bio_names = {nm for nm in (getattr(t, "name", None) for t in bio_tools) if nm}
    _wrap_tools_with_circuit_breaker(all_tools, breaker_state, transport_disable_names=_bio_names)
    # Provenance: record the biocurator's LITERATURE accesses (query + returned DOIs/PMIDs) for
    # audit + analysis (surfaced as biocurator_literature). Only literature tools are wrapped, and
    # only when the `bio_literature` ablation made them resolve into bio_tools; empty otherwise.
    literature_log: list = []
    _wrap_literature_recorder(
        [t for t in bio_tools if getattr(t, "name", None) in LITERATURE_BIO_TOOLS], literature_log
    )
    tools_by_name = {n: t for t in all_tools if (n := getattr(t, "name", None))}
    subagents = _build_subagents(tools_by_name)  # each analyst carries its own tool OBJECTS
    # The biocurator joins the panel on every deepagents run (its lack of any petab tool is the
    # structural half of the task-blinding — see _build_biocurator_subagent). It is never dropped:
    # if bio tools resolved it uses them, otherwise it runs "dumb", answering from the model's own
    # biology knowledge (see BIOCURATOR_PROMPT). The lead's roster + biocurator guidance are static
    # in scenario.DEFAULT_SYSTEM_PROMPT — no runtime prompt splice, since the biocurator is
    # unconditional. biocurator_tools records what it actually got (empty = a dumb run) for analysis.
    # Hand the biocurator the scenario's HAND-CURATED biological entity NAMES DIRECTLY (see
    # Scenario.biocurator_entities) — it holds no petab tool and cannot self-serve like the
    # analysts, so this is its reliable input channel rather than a lead `description` /
    # silently-dropped `prompt=` (issue #91). Empty for a scenario with no lookupable biology (the
    # abstract toys): the biocurator then simply gets no injected entities (nothing to look up).
    biocurator_entities = list(curated_entities or [])
    subagents.append(_build_biocurator_subagent(bio_tools, biocurator_entities))
    biocurator_tools = sorted(n for t in bio_tools if (n := getattr(t, "name", None)))
    print(f"[agent] biocurator on the panel with {len(biocurator_tools)} bio tool(s)"
          + (f": {biocurator_tools}" if biocurator_tools
             else " (none resolved — running on the model's own knowledge)"))
    print(f"[agent] biocurator handed {len(biocurator_entities)} entity name(s)"
          + (f": {biocurator_entities}" if biocurator_entities else " (none curated)"))
    # The LEAD gets ONLY the orchestrator tools (+ the finish gate) — NOT the analysts'
    # data/diagnostic tools (query_data, get_fitting_log, get_measurements, ...). Those heavy
    # reads run INSIDE analyst subagents, whose verbose results live in their own transient
    # contexts and never enter the lead's every-turn re-sent history — which is where this
    # input-dominated workload's token cost accumulates (the lead re-sends its whole history on
    # each of its 60-115 turns). The lead diagnoses by DELEGATING to the analysts, not by
    # querying data itself. deepagents' subagents hold their own resolved tool objects, so a
    # restricted top-level tool list does not starve them.
    lead_tools = [tools_by_name[n] for n in ORCHESTRATOR_TOOLS if n in tools_by_name]
    # Shared with the driver loop: the finish tool sets finish_state["confirmed"] on a
    # confirmed finish (so a legitimately-finished run is NOT bounced back), and "gave_up" after
    # too many no-fit finish attempts (so the driver ends the run promptly, not at recursion).
    finish_state = {"confirmed": False, "gave_up": False}
    lead_tools = [*lead_tools, _finish_tool(session, finish_state, probe=progress_probe)]

    llm = _chat_model_class()(
        # Keeps the Phoenix span named `ChatOpenAI` despite the subclass -- see _chat_model_class.
        name="ChatOpenAI",
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        timeout=_LLM_TIMEOUT_S,          # fail a hung provider call fast (was unbounded)
        max_retries=_LLM_MAX_RETRIES,
        # Ask for token usage even on streamed responses (adds stream_options.include_usage):
        # OpenAI-compatible providers omit usage from a stream unless requested, which would
        # leave _token_accountant recording 0 tokens for the whole run. Harmless when a call
        # isn't streamed.
        stream_usage=True,
        # Best-effort prompt-cache hint: asks the provider to route repeat requests to the
        # same backend so its prefix cache can hit (prompt tokens dominate this workload).
        # Whether it takes effect is provider-dependent (OpenRouter -> the served model) and
        # not guaranteed; it is harmless when ignored.
        model_kwargs={"prompt_cache_key": "petab-mcp-agent"},
        # OpenRouter provider routing: for a model mirrored across providers (e.g. Luna is also on
        # Bedrock/Azure at ~2-11x) this excludes the pricey copies so the request stays on OpenAI's
        # standard tier. Sent verbatim in the request body via extra_body; included ONLY when the
        # model needs pinning (models.provider_routing returns None otherwise). See models.py.
        extra_body=_extra_body(model),
    )
    # Transient provider errors that should RESUME the run (below) rather than end it, when they
    # escape the OpenAI client's own retries — e.g. a malformed non-JSON body the client won't
    # retry. Non-transient errors (bad request, recursion-limit) are deliberately excluded.
    from json import JSONDecodeError

    from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                        RateLimitError)
    _TRANSIENT_ERRORS: tuple = (APIConnectionError, APITimeoutError, InternalServerError,
                                RateLimitError, JSONDecodeError)
    # Backstop: a transport error (a closed/hung MCP server) that ESCAPES the per-tool breaker —
    # e.g. raised at the `task` subagent boundary or the bio ClientSession itself — should RESUME
    # the run from the checkpoint rather than abort it (the petab server is still alive). Cover both
    # transport layers the breaker classifies in _TRANSPORT_ERROR_NAMES: the anyio stream errors and
    # httpx's TransportError base (ConnectError/ReadError/WriteError/ReadTimeout/ConnectTimeout/
    # RemoteProtocolError all derive from it).
    try:
        from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
        _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (ClosedResourceError, BrokenResourceError, EndOfStream)
    except Exception:  # pragma: no cover - anyio always present with mcp, but never fail on import
        pass
    try:
        import httpx
        _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (httpx.TransportError,)
    except Exception:  # pragma: no cover - httpx always present with mcp, but never fail on import
        pass

    # deepagents renamed the prompt kwarg between releases (instructions -> system_prompt).
    # Pass whichever the installed version accepts, plus the analyst subagents.
    params = inspect.signature(create_deep_agent).parameters
    prompt_kw = "system_prompt" if "system_prompt" in params else "instructions"
    kwargs: dict[str, Any] = {"model": llm, "tools": lead_tools, prompt_kw: system_prompt}
    if "subagents" in params:
        kwargs["subagents"] = subagents
    else:  # pragma: no cover - guards against a deepagents API drift
        print("[agent] WARNING: installed deepagents has no `subagents` parameter; "
              "running a single agent without the analyst panel.")
    # Lead middleware: drop deepagents' built-in FS/todo tools (the analysts can't be reliably
    # filtered in this deepagents version — see _builtin_tool_filter; `task` is untouched), hide a
    # circuit-broken tool so it stops being re-proposed, and guard tool dispatch (dedupe duplicate
    # calls in a batch; push back on idle no-tool turns and label a run that never fires a tool as
    # no_tools_executed — see _tool_dispatch_guard). No history trimming: all target models have a
    # 1M-token window and the recursion limit caps a run at ~120k tokens, so the old tool-result
    # trimmer only ever fired on a rare over-long run and, with prompt caching, cost more (a clear
    # invalidates the cached prefix) than the near-free cache_read history it removed.
    guard_stats: dict[str, Any] = {"tools_executed": 0, "dropped_dup_tool_calls": 0,
                                   "idle_nudges": 0, "gave_up_no_tools": False,
                                   "unexpected_task_kwargs": [], "task_arg_rejections": 0,
                                   "empty_subagent_reports": 0, "empty_subagent_retries": 0,
                                   "empty_subagent_markers": 0,
                                   "subagent_names": [], "general_purpose_subagent": None}
    guard_installed = "middleware" in params
    if guard_installed:
        kwargs["middleware"] = [
            _builtin_tool_filter(), _disabled_tool_filter(breaker_state),
            # _task_arg_guard is placed OUTSIDE _tool_dispatch_guard (earlier in the list = outer
            # tool-call wrapper) so a REJECTED `task` call short-circuits before the dispatch guard
            # counts it — a bounced call never dispatched a subagent, so it must not inflate
            # tools_executed (which gates the no_tools_executed relabel / idle-nudge logic).
            # _empty_report_guard is placed INNERMOST (last) so its ONE-shot re-dispatch of a
            # subagent that returned "" hits the real tool directly, without re-entering
            # _tool_dispatch_guard and double-counting tools_executed for a single `task` call.
            _task_arg_guard(guard_stats), _tool_dispatch_guard(guard_stats),
            _empty_report_guard(guard_stats),
        ]
    else:  # pragma: no cover - guards against a deepagents API drift
        print("[agent] WARNING: installed deepagents has no `middleware` parameter; "
              "the lead's built-in FS/todo tools + full history will remain, and proposed "
              "tool calls are not guarded.")
    # A checkpointer lets us RESUME the graph from its last completed super-step after a stall or
    # exhausted call-retry (below), instead of discarding the run. In-process MemorySaver is
    # enough (one run per process). Degrade gracefully if the graph can't take one.
    checkpointer = None
    if "checkpointer" in params:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
            kwargs["checkpointer"] = checkpointer
        except Exception as exc:  # pragma: no cover - resume is a best-effort enhancement
            print(f"[agent] no checkpointer ({type(exc).__name__}); a stall will end the run "
                  "instead of resuming.")
    # Drop the uncurated `general-purpose` subagent BEFORE the graph is built — it is added
    # during create_deep_agent, so this must precede it (issue #112). Then verify through
    # deepagents' own resolver: a run whose panel still carries it is contaminated (the lead can
    # delegate model authoring), so record the verdict in result.json rather than only logging
    # it, and warn loudly. We do NOT abort — a harness-detail drift must not cost a sweep run.
    _disable_general_purpose_subagent()
    gp_enabled = _general_purpose_enabled(llm)
    guard_stats["general_purpose_subagent"] = gp_enabled
    guard_stats["subagent_names"] = [s["name"] for s in subagents]
    if gp_enabled is not False:
        print("[agent] WARNING: deepagents' `general-purpose` subagent is "
              + ("still enabled" if gp_enabled else "of unknown status")
              + " — the lead may be able to delegate model authoring to an uncurated subagent "
                "holding the full tool pool (issue #112). Treat this run's containment as "
                "unverified.")
    agent = create_deep_agent(**kwargs)

    stop_reason = "completed"
    # Drive the agent to completion, iterating manually so a per-step STALL WATCHDOG can fire:
    # astream emits once per graph super-step, so if `__anext__` doesn't return within the stall
    # window the current step is wedged. On a stall — or a call-retry that ran out (below) — we
    # RESUME from the last checkpoint (astream with input=None continues pending work) rather than
    # abandoning the run, up to _RUN_RESUME_ATTEMPTS times. A non-transient error (e.g.
    # GraphRecursionError = step budget exhausted) is NOT retried; the judge assesses the model as
    # it stands. We consume the stream only to drive the loop; Phoenix spans are the record.
    # The longest single super-step is a lead commit whose make_changes runs an n_starts multistart
    # fit (astream emits nothing until it returns), so the stall window must exceed one whole fit
    # or a slow-but-healthy fit trips the watchdog. Size it to the same fit budget the transport's
    # sse_read_timeout uses (fit_budget.py), floored at _STEP_STALL_TIMEOUT_S for the non-fit steps
    # (a lead LLM call, an analyst sub-agent). A hung provider CALL is still caught fast by the
    # 120s client timeout; this watchdog only backstops a wedged step.
    from ..fit_budget import fit_time_budget
    stall_timeout = max(_STEP_STALL_TIMEOUT_S, fit_time_budget(n_starts))
    # Self-imposed total-runtime budget (see the docstring): checked between super-steps, never
    # mid-step, so we always stop at a clean boundary rather than interrupting a live commit/fit.
    deadline = (time.monotonic() + max_wall_seconds) if max_wall_seconds is not None else None
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if checkpointer is not None:
        config["configurable"] = {"thread_id": "petab-mcp-run"}
    # Sum token usage across the lead + every subagent (config callbacks propagate into the
    # subgraphs), for result.json. One instance across all resume iterations = whole-run totals.
    token_accountant = _token_accountant()
    config["callbacks"] = [token_accountant]

    async def _early_stop_reason() -> str | None:
        """The server's early-stop reason if the run should stop now, else ``None`` (fail-open:
        any probe error just means "keep going")."""
        if progress_probe is None:
            return None
        try:
            data = _structured(await progress_probe("get_progress", {}))
            if isinstance(data, dict) and data.get("should_stop"):
                return data.get("stop_reason") or "early-stop condition met"
        except Exception:  # pragma: no cover - never let the probe break the run
            pass
        return None

    stream_input: Any = {"messages": [{"role": "user", "content": task}]}
    resume_left = _RUN_RESUME_ATTEMPTS if checkpointer is not None else 0
    # A silent stop (graph ends on a no-tool turn) without a confirmed finish is bounced back
    # into the finish gate this many times; separate from resume_left, which is for transients.
    silent_bounces = 0
    # Last clean-completion probe of whether a model was authored (see the silent-stop bounce);
    # captured so the post-loop relabel reuses it instead of re-probing. Only read when
    # stop_reason == "completed", which guarantees the intercept refreshed it that iteration.
    model_committed = True
    if progress_probe is None:
        # The no-model-built guard (and the deterministic early-stop) both need the probe; without
        # it a silent no-model stop scores as an empty "completed" run instead of resuming to
        # build one — so a caller that omits progress_probe reintroduces that gap silently.
        print("[agent] no progress_probe wired — the no-model-built guard and deterministic "
              "early-stop are disabled for this run")
    while True:
        retriable: str | None = None
        agen = None
        wall_budget_exceeded = False
        token_budget_exceeded = False
        early_stopped: str | None = None
        try:
            # Open the stream INSIDE the try so a synchronous raise from astream (e.g. on the
            # input=None resume path) is handled/degraded here, not left to abort the whole run.
            agen = agent.astream(stream_input, config=config, stream_mode="values").__aiter__()
            while True:
                try:
                    # Wait no longer than the time left in the wall-time budget. Without this the
                    # loop can only notice the deadline BETWEEN super-steps, so a step starting
                    # just before it runs the loop up to `stall_timeout` (18 min at any n_starts)
                    # PAST it -- against a default margin of 20 min, that leaves ~2 min for the
                    # judge to restore and score a checkpoint. Observed: a fujita/v4-pro run was
                    # hard-killed by Slurm at exactly 06:00:02 with NO result.json, losing the
                    # whole run. Capping the wait hands control back at the deadline, so the
                    # entire margin is available for scoring.
                    step_wait = stall_timeout if deadline is None else max(
                        1.0, min(stall_timeout, deadline - time.monotonic()))
                    await asyncio.wait_for(agen.__anext__(), timeout=step_wait)
                except StopAsyncIteration:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    wall_budget_exceeded = True
                    break
                # Token ceiling, checked at the same clean boundary as the wall-time budget. The
                # accountant is ONE instance across every resume, so this bounds the whole run
                # rather than resetting each time astream is re-entered -- which is precisely how
                # the observed runaway escaped every other limit.
                if (max_run_tokens is not None
                        and token_accountant.prompt + token_accountant.completion
                        >= max_run_tokens):
                    token_budget_exceeded = True
                    break
                # The gate gave up (too many no-fit finish attempts) — end promptly at this
                # boundary rather than letting the model spam finish to the recursion limit.
                if finish_state["gave_up"]:
                    break
                # Deterministic early-stop: poll the server between super-steps and stop at this
                # clean boundary once the run has stopped improving (streak cap / plateau).
                early_stopped = await _early_stop_reason()
                if early_stopped is not None:
                    break
            stop_reason = (
                "wall_time_budget_exceeded" if wall_budget_exceeded
                # A BARE label, like wall_time_budget_exceeded beside it: the counts used to be
                # interpolated here, which made every such run's stop_reason unique AND collided
                # with `experiments._INFRA_ABORT_MARKERS` -- its "402" marker (out-of-credits) is a
                # substring of a comma-formatted token count like "20,402,551", so a runaway with
                # 0 fits was classified as infrastructure noise and dropped from the aggregation
                # entirely. The counts belong in the print below and in `token_usage`.
                else "token_budget_exceeded" if token_budget_exceeded
                else "no_model_built" if finish_state["gave_up"]
                else f"early_stop: {early_stopped}" if early_stopped
                else "completed"
            )
        except asyncio.TimeoutError:
            if deadline is not None and time.monotonic() >= deadline:
                # The wait was cut short by the wall-time cap above, not by a wedged step. This is
                # the budget expiring mid-super-step, so it must STOP (and let the judge score what
                # is committed) rather than resume — a resume here would spend the margin that the
                # scoring needs, which is how a run gets hard-killed with no result at all.
                wall_budget_exceeded = True
                stop_reason = "wall_time_budget_exceeded"
            else:
                retriable = (f"stalled: no step completed in {stall_timeout:.0f}s "
                             "(hung provider call or wedged fit)")
        except _TRANSIENT_ERRORS as exc:  # call-level retries exhausted -> resume the run
            retriable = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            # A provider CONTENT-POLICY refusal is terminal and deterministic: the same transcript
            # re-sent trips it again, so it must not join _TRANSIENT_ERRORS and be resumed. Give it
            # its own short stop_reason instead of the raw exception text, so a sweep can count it
            # as one outcome rather than as a unique failure string per run. Seen live:
            # gpt-5.6-luna-pro on a PEtab immunology model -> 400 "This content was flagged for
            # possible biological risk." Nothing here can fix that; the point is to record it.
            # Order matters: a content refusal is checked FIRST and is terminal, so it can never
            # be routed to the retry path by a later branch.
            if (filtered := _content_filter_stop_reason(exc)) is not None:
                stop_reason = filtered
            elif (transient := _transient_provider_reason(exc)) is not None:
                retriable = transient          # resume from the checkpoint, keeping every version
            else:
                stop_reason = f"{type(exc).__name__}: {exc}"
        finally:
            # Best-effort close; bound it so a truly wedged stream can't hang teardown too.
            if agen is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(agen.aclose(), timeout=10)
        if wall_budget_exceeded:
            print(f"[agent] wall-time budget ({max_wall_seconds:.0f}s) exceeded — stopping "
                  "at this step boundary; the judge scores whatever is currently committed.")
            break
        if token_budget_exceeded:
            # Deliberately NOT retriable: a resume would hand the loop a fresh step budget and the
            # whole history to re-send, which is the exact mechanism that turned one run into
            # 107.9M tokens. Ending here keeps everything committed so far scoreable.
            print(f"[agent] token budget exceeded: "
                  f"{token_accountant.prompt + token_accountant.completion:,} tokens over "
                  f"{max_run_tokens:,} in {token_accountant.n_calls} LLM calls — stopping at this "
                  "step boundary; the judge scores whatever is currently committed.")
            break
        if early_stopped is not None:
            print(f"[agent] early-stop: {early_stopped} — stopping at this step boundary; "
                  "the judge restores + scores the best-BIC checkpoint.")
            break
        if finish_state["gave_up"]:
            print("[agent] finish gate gave up (no fittable model after repeated attempts) — "
                  "ending the run as no-model-built.")
            break
        if retriable is None:
            # Clean graph termination. A "completed" that did NOT pass the finish gate is a SILENT
            # STOP — the lead ended on a no-tool turn, sidestepping the finish gate. A silent stop is
            # not an accepted exit: bounce it back into the gate so the run only ends by finishing
            # properly (a fitted model, diagnosis shown, confirmed) — with no coaching on WHAT to
            # fix, just enforcement of the protocol. Bounded (_MAX_SILENT_BOUNCES) so a genuinely
            # stuck run still terminates at the step/wall limit rather than looping. A CONFIRMED
            # finish is respected (not bounced) and needs neither the probe nor the relabel, so it
            # short-circuits first — saving a describe_problem call on the good path. The no-model
            # case gets the build nudge; a built-but-unfinished model gets the finish nudge. No
            # backoff: a silent stop is not a rate-limited transient, so there is nothing to wait
            # out. model_committed is captured for the post-loop relabel (no second probe).
            if stop_reason == "completed" and not finish_state["confirmed"]:
                model_committed = await _model_committed(progress_probe)
                past_deadline = deadline is not None and time.monotonic() >= deadline
                if (silent_bounces < _MAX_SILENT_BOUNCES and checkpointer is not None
                        and not past_deadline):
                    silent_bounces += 1
                    nudge = _NO_MODEL_NUDGE if not model_committed else _SILENT_FINISH_NUDGE
                    what = "NO model authored" if not model_committed else "no finish through the gate"
                    print(f"[agent] silent stop ({what}) — bouncing back to finish properly "
                          f"({silent_bounces}/{_MAX_SILENT_BOUNCES})")
                    stream_input = {"messages": [{"role": "user", "content": nudge}]}
                    continue
            break
        if deadline is not None and time.monotonic() >= deadline:
            # Out of time even before a resume would start — don't burn a backoff on a retry
            # the deadline will just cut off anyway.
            stop_reason = f"wall_time_budget_exceeded (during retry: {retriable})"
            print(f"[agent] wall-time budget ({max_wall_seconds:.0f}s) exceeded while handling "
                  f"'{retriable}' — stopping instead of resuming.")
            break
        # The SAME guard for tokens. Without it a run already past the ceiling could still resume:
        # the inner check only runs at a completed step boundary, so a run that blew the budget and
        # then hit a stall / transient error arrives here with `token_budget_exceeded` still False.
        # Each resume re-enters astream with a fresh step budget AND re-sends the whole history, so
        # with `_RUN_RESUME_ATTEMPTS` retries the effective ceiling was several times the nominal
        # one -- the exact compounding `_MAX_RUN_TOKENS` exists to bound.
        if (max_run_tokens is not None
                and token_accountant.prompt + token_accountant.completion >= max_run_tokens):
            stop_reason = "token_budget_exceeded"
            print(f"[agent] token budget ({max_run_tokens:,}) already exceeded at "
                  f"{token_accountant.prompt + token_accountant.completion:,} while handling "
                  f"'{retriable}' — stopping instead of resuming.")
            break
        if resume_left <= 0:
            stop_reason = retriable  # out of resume attempts — end with whatever was built
            break
        resume_left -= 1
        backoff = min(5.0 * (_RUN_RESUME_ATTEMPTS - resume_left), 30.0)
        print(f"[agent] {retriable} — resuming from last checkpoint in {backoff:.0f}s "
              f"({resume_left} resume(s) left)")
        await asyncio.sleep(backoff)
        stream_input = None  # resume pending work from the checkpoint

    # A "completed" stop_reason that did NOT pass the finish gate (bounces exhausted, or no
    # checkpointer to resume from) is relabelled honestly rather than read as a proper finish, so
    # evaluation can see a bypassed gate. Three cases, most specific first:
    #   * no_tools_executed — the lead only ever emitted no-tool turns, so the dispatch guard
    #     pushed back _MAX_IDLE_MODEL_STEPS times and GAVE UP (guard_stats["gave_up_no_tools"]):
    #     the degenerate "proposes tool calls but none fire" run. Distinct from a run that DID
    #     dispatch tools but built no model (e.g. delegation-only), which stays no_model_built.
    #     Gated on the guard actually being installed (without the `middleware` hook the counters
    #     are not trustworthy). This must come first: gave_up_no_tools implies no model, so it
    #     would otherwise be swallowed by the no_model_built branch below.
    #   * no_model_built — nothing was authored;
    #   * silent_stop — a model exists but the run never finished through the finish gate.
    # A confirmed finish keeps "completed". Reuses the intercept's probe (model_committed): a
    # "completed" stop_reason here means the final iteration ran the intercept, which refreshed it
    # — no second describe_problem call.
    if stop_reason == "completed" and not finish_state["confirmed"]:
        if guard_installed and guard_stats.get("gave_up_no_tools"):
            stop_reason = "no_tools_executed"
        else:
            stop_reason = "no_model_built" if not model_committed else "silent_stop"
    if guard_stats["dropped_dup_tool_calls"]:
        print(f"[agent] de-duplicated {guard_stats['dropped_dup_tool_calls']} identical tool "
              "call(s) within a batch over the run")
    if breaker_state["n_tool_errors"]:
        disabled = sorted(breaker_state["disabled"])
        print(f"[agent] tool errors this run: {breaker_state['n_tool_errors']}"
              + (f"; disabled: {disabled}" if disabled else ""))
    token_usage = token_accountant.totals()
    # A run that dispatched tools necessarily made LLM calls; capturing zero of them means the
    # usage was not reported (a streamed response without usage, or a callback-propagation gap),
    # so the recorded token_usage understates the run. Make that loud rather than silently
    # writing zeros that read as a genuinely tiny run in the cost analysis.
    if token_usage["n_llm_calls"] == 0 and guard_stats.get("tools_executed", 0) > 0:
        print("[agent] WARNING: token accounting captured 0 LLM calls despite tool activity — "
              "token_usage in result.json is understated (usage not reported for this run).")

    return {
        "stop_reason": stop_reason,
        "n_tool_errors": breaker_state["n_tool_errors"],
        "disabled_tools": sorted(breaker_state["disabled"]),
        # Call-level outcomes for the single editor, counted at the CLIENT wrapper. Distinct from
        # `edit_rejects`, which is server-side and therefore blind to a call that never reached
        # the server -- the case that mattered: a run making 134 failing make_changes calls
        # recorded `edit_rejects.attempted: 0`, so nothing in result.json showed the failure.
        "make_changes_calls": call_outcomes(breaker_state, "make_changes"),
        "openrouter_extra_body": _extra_body(model),
        # What the request ASKED for is `openrouter_extra_body`; this is what it GOT. The two are
        # not the same claim -- `provider.only` bounds routing to a set, so a run can be served by
        # several of its members and still honour the pin.
        "openrouter_endpoints": token_accountant.endpoints(),
        "biocurator_tools": biocurator_tools,
        "biocurator_entities": biocurator_entities,
        "biocurator_literature": literature_log,
        "token_usage": token_usage,
        **guard_stats,
    }
