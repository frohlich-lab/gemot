"""Per-run token accounting for the lead AND its subagents."""

from __future__ import annotations

def _token_accountant():
    """A LangChain callback that sums token usage across EVERY LLM call in a run.

    Attached via the run config's ``callbacks``, which LangChain/LangGraph propagate into the
    analyst + biocurator subgraphs — so this totals the lead AND the subagents, matching
    Phoenix's per-trace rollup, but in-process so the counts can be written straight to
    result.json without re-deriving them from the (huge) span database. Records prompt,
    completion, and cache-read tokens; ``cache_read`` is the provider-reported cached-prompt
    count (``usage_metadata.input_token_details.cache_read``), which is what makes an offline
    cost estimate cache-aware. One instance is reused across resume iterations so the totals
    cover the whole run. Never raises — token accounting must not break a run.
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

    class _Accountant(AsyncCallbackHandler):
        def __init__(self):
            self.prompt = self.completion = self.cache_read = self.n_calls = 0

        async def on_llm_end(self, response, **kwargs):  # noqa: D401 - LangChain hook
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
                        self.completion += _as_int(um.get("output_tokens"))
                        self.cache_read += _as_int(
                            (um.get("input_token_details") or {}).get("cache_read"))
                if not found:
                    # Older/non-usage_metadata responses carry an aggregate in llm_output.
                    tu = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
                    if tu:
                        self.prompt += _as_int(tu.get("prompt_tokens"))
                        self.completion += _as_int(tu.get("completion_tokens"))
                        self.cache_read += _as_int(
                            (tu.get("prompt_tokens_details") or {}).get("cached_tokens"))
                        self.n_calls += 1
            except Exception:  # pragma: no cover - accounting must never break a run
                pass

        def totals(self):
            return {
                "prompt_tokens": self.prompt,
                "completion_tokens": self.completion,
                "cache_read_tokens": self.cache_read,   # cached prompt tokens (for cache-aware cost)
                "total_tokens": self.prompt + self.completion,
                "n_llm_calls": self.n_calls,
            }

    return _Accountant()
