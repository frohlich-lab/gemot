"""Per-run token accounting for the lead AND its subagents."""

from __future__ import annotations

import math
from uuid import UUID

#: Explicit graph name used by the driver and reported in LangChain callback metadata.
_LEAD_AGENT_NAME = "lead"

def _token_accountant():
    """A LangChain callback that sums token usage across EVERY LLM call in a run.

    Attached via the run config's ``callbacks``, which LangChain/LangGraph propagate into the
    analyst + biocurator subgraphs — so this totals the lead AND the subagents, matching
    Phoenix's per-trace rollup, but in-process so the counts can be written straight to
    result.json without re-deriving them from the (huge) span database. Records prompt,
    completion, both cache columns (``usage_metadata.input_token_details.cache_read`` and
    ``.cache_creation``) and the billed cost (OpenRouter's per-response ``usage.cost``, summed
    into ``cost_usd``). The cache columns explain the bill; ``cost_usd`` is the bill. One instance
    is reused across resume iterations so the totals cover the whole run. Never raises — token
    accounting must not break a run.

    Lead prompt sizes use the named lead graph's ``model`` calls, excluding subagents and
    helper calls such as summarization. ``final_lead_prompt_tokens`` describes the last
    lead request started, including retries: it is None if that request never reports usage.
    ``max_lead_prompt_tokens`` keeps the largest reported lead input across resumes and
    compaction. Both include cached input tokens and exclude the response's output tokens.
    """
    from langchain_core.callbacks import AsyncCallbackHandler

    # AsyncCallbackHandler (not the sync base): the run is driven with astream, so LangChain
    # awaits the async on_llm_end INLINE on the event loop instead of dispatching a sync handler
    # to a threadpool. That serialises the counter updates — parallel subagent LLM completions
    # cannot interleave a non-atomic `+=` and lose an increment.
    def _as_int(x) -> int:
        """A token count coerced to a non-negative int, or 0 for anything unparseable — so one
        malformed field never aborts accounting for the rest of a response's records."""
        try:
            return max(0, int(x or 0))
        except (TypeError, ValueError):
            return 0

    def _as_float(x) -> float | None:
        """A billed amount, or None when absent, unparseable, boolean or non-finite — a NaN or an
        inf would propagate into ``cost_usd`` and poison every aggregate taken over it."""
        if x is None or isinstance(x, bool):
            return None
        try:
            value = float(x)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    def _reported_prompt(x) -> int | None:
        """Missing or malformed lead usage is unknown, not a zero-sized prompt."""
        if x is None or isinstance(x, bool):
            return None
        try:
            value = int(x)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if value >= 0 and (not isinstance(x, float) or x == value) else None

    class _Accountant(AsyncCallbackHandler):
        def __init__(self):
            self.prompt = self.completion = self.cache_read = self.n_calls = 0
            self.cost, self.n_costed = 0.0, 0     # billed USD, and the calls that reported one
            self.cache_write = 0         # prompt tokens WRITTEN to the cache (billed at a premium)
            self.max_prompt = 0          # largest single-call prompt — the context-stop signal
            self.final_lead_prompt: int | None = None
            self.max_lead_prompt: int | None = None
            self._lead_runs: set[UUID] = set()
            self._last_lead_run: UUID | None = None
            # (upstream provider, service tier) -> calls served by it. See `endpoints()`.
            self.served: dict[tuple, int] = {}

        async def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None,
                                      **kwargs):
            # Subagent graphs replace lc_agent_name with their own name even when they share
            # the lead's model. before_model helpers inherit the name but have another node.
            if (isinstance(metadata, dict)
                    and metadata.get("lc_agent_name") == _LEAD_AGENT_NAME
                    and metadata.get("langgraph_node") == "model"):
                self._lead_runs.add(run_id)
                self._last_lead_run = run_id
                self.final_lead_prompt = None

        async def on_llm_error(self, error, *, run_id, **kwargs):
            self._lead_runs.discard(run_id)

        def _record_endpoint(self, response):
            """Tally which upstream endpoint served this call, from whichever place carries it.

            ``openrouter_provider`` is put there by the ChatOpenAI subclass in ``agent.driver``;
            ``service_tier`` is langchain's own passthrough. Both may be absent — a non-OpenRouter
            base_url, or a stream — and an unattributed call is tallied under ``None`` rather than
            dropped, so the counts always sum to the calls actually made.
            """
            # isinstance, not `or {}`: a truthy NON-dict llm_output (a provider returning a bare
            # string) would sail through `or {}` and raise on `.get`, and the caller's except would
            # then drop the call from the tally entirely -- breaking the one invariant below.
            out = getattr(response, "llm_output", None)
            out = out if isinstance(out, dict) else {}
            provider, tier = out.get("openrouter_provider"), out.get("service_tier")
            if provider is None:
                for gens in getattr(response, "generations", None) or []:
                    for gen in gens or []:
                        meta = getattr(getattr(gen, "message", None), "response_metadata", None)
                        if isinstance(meta, dict) and meta.get("openrouter_provider"):
                            provider = meta["openrouter_provider"]
                            tier = tier or meta.get("service_tier")
                            break
                    if provider is not None:
                        break
            key = (provider, tier)
            self.served[key] = self.served.get(key, 0) + 1

        def _response_cost(self, response) -> float | None:
            """OpenRouter's ``usage.cost`` for this response, from the ``token_usage`` block of
            ``llm_output`` or of a generation's ``response_metadata``, whichever carries it first;
            None if none does."""
            out = getattr(response, "llm_output", None)
            blocks = [out.get("token_usage")] if isinstance(out, dict) else []
            for gens in getattr(response, "generations", None) or []:
                for gen in gens or []:
                    meta = getattr(getattr(gen, "message", None), "response_metadata", None)
                    if isinstance(meta, dict):
                        blocks.append(meta.get("token_usage"))
            for tu in blocks:
                if isinstance(tu, dict) and (cost := _as_float(tu.get("cost"))) is not None:
                    return cost
            return None

        async def on_llm_end(self, response, *, run_id=None, **kwargs):  # noqa: D401 - LangChain hook
            is_lead = run_id in self._lead_runs
            self._lead_runs.discard(run_id)
            lead_prompt = None
            try:
                self._record_endpoint(response)
            except Exception:  # pragma: no cover - accounting must never break a run
                pass
            try:
                if (cost := self._response_cost(response)) is not None:
                    self.cost += cost
                    self.n_costed += 1
            except Exception:  # pragma: no cover - accounting must never break a run
                pass
            try:
                found = False
                # `generations` is List[List[Generation]]: the OUTER list is one entry per
                # PROMPT, the inner one per candidate when n > 1. A call is a prompt, so
                # n_calls advances once per outer entry that carried usage — not once per
                # candidate, which counted a single n=2 response as two calls and inflated the
                # metric the driver reads to decide whether usage was captured at all. The
                # token sums stay per-record: those really do add up.
                for gens in getattr(response, "generations", None) or []:
                    counted = False
                    for gen in gens or []:
                        msg = getattr(gen, "message", None)
                        um = getattr(msg, "usage_metadata", None) if msg is not None else None
                        if not um:
                            continue
                        found = True
                        if not counted:
                            counted = True
                            self.n_calls += 1
                        # Per-field coercion (not one int() over the whole record): a bad value
                        # in one field contributes 0 but does not discard the other fields, the
                        # other generations, or the llm_output fallback below.
                        self.prompt += _as_int(um.get("input_tokens"))
                        self.max_prompt = max(self.max_prompt, _as_int(um.get("input_tokens")))
                        if is_lead and (size := _reported_prompt(um.get("input_tokens"))) is not None:
                            lead_prompt = max(lead_prompt or 0, size)
                        self.completion += _as_int(um.get("output_tokens"))
                        self.cache_read += _as_int(
                            (um.get("input_token_details") or {}).get("cache_read"))
                        self.cache_write += _as_int(
                            (um.get("input_token_details") or {}).get("cache_creation"))
                if not found:
                    # Older/non-usage_metadata responses carry an aggregate in llm_output.
                    # isinstance, not `or {}` -- see _record_endpoint.
                    out = getattr(response, "llm_output", None)
                    tu = (out if isinstance(out, dict) else {}).get("token_usage") or {}
                    if tu:
                        self.prompt += _as_int(tu.get("prompt_tokens"))
                        self.max_prompt = max(self.max_prompt, _as_int(tu.get("prompt_tokens")))
                        if is_lead:
                            lead_prompt = _reported_prompt(tu.get("prompt_tokens"))
                        self.completion += _as_int(tu.get("completion_tokens"))
                        self.cache_read += _as_int(
                            (tu.get("prompt_tokens_details") or {}).get("cached_tokens"))
                        self.cache_write += _as_int(
                            (tu.get("prompt_tokens_details") or {}).get("cache_write_tokens"))
                        self.n_calls += 1
            except Exception:  # pragma: no cover - accounting must never break a run
                pass

            if is_lead:
                if run_id == self._last_lead_run:
                    self.final_lead_prompt = lead_prompt
                if lead_prompt is not None:
                    self.max_lead_prompt = max(self.max_lead_prompt or 0, lead_prompt)

        def totals(self):
            return {
                "prompt_tokens": self.prompt,
                "completion_tokens": self.completion,
                "cache_read_tokens": self.cache_read,    # cached prompt tokens
                # 0 wherever caching is implicit; elsewhere the tokens billed at the premium.
                "cache_write_tokens": self.cache_write,
                "total_tokens": self.prompt + self.completion,
                # The COMPUTE-priced number. Every estimate in the 2026-08-14 token review that
                # quoted prompt tokens overstated its saving 5-10x because ~80-90% are cache
                # reads; this column is what rankings should use.
                "novel_tokens": max(self.prompt - self.cache_read, 0),
                "max_prompt_tokens": self.max_prompt,
                "final_lead_prompt_tokens": self.final_lead_prompt,
                "max_lead_prompt_tokens": self.max_lead_prompt,
                "n_llm_calls": self.n_calls,
                "cost_usd": self.cost if self.n_costed else None,   # billed USD; None if none reported
                "n_uncosted_calls": max(self.n_calls - self.n_costed, 0),
            }

        def endpoints(self):
            """Which upstream endpoints served this run, for ``result.json``.

            ``drifted`` is the field to group an A/B by: ``provider.only`` bounds routing to a SET,
            and every member of that set is a different machine, so a run served by two of them is
            not one measurement. Recorded for EVERY run, not only drifted ones — "absent" would
            otherwise have to be read as "probably fine", which is how provenance rots.
            """
            served = sorted(
                ({"provider": p, "service_tier": t, "n_calls": n}
                 for (p, t), n in self.served.items()),
                key=lambda r: (-r["n_calls"], str(r["provider"])),
            )
            # Drift is judged over ATTRIBUTED endpoints only. The unattributed bucket is a gap in
            # our own recording, not a second machine, and counting it would flag an ordinary
            # single-endpoint run as drifted the moment one response arrived without a provider --
            # excluding a healthy run from the very A/B this field exists to filter. It stays in
            # `served` (so the counts still sum to the calls made) and is surfaced separately.
            attributed = [s for s in served if s["provider"] is not None]
            return {
                "served": served,
                "n_distinct": len(attributed),
                "n_unattributed_calls": sum(s["n_calls"] for s in served
                                            if s["provider"] is None),
                # One endpoint = no drift. Zero happens when nothing reported a provider (a
                # non-OpenRouter base_url), which is not drift either.
                "drifted": len(attributed) > 1,
            }

    return _Accountant()
