"""Where one analyst's provider refusal stops: at that analyst, not the run.

Every analyst (and the biocurator) runs inside a ``task`` call of the lead's, and the ``task``
calls of one lead turn run as ONE super-step of the lead's graph. An exception escaping any of them
fails the super-step: langgraph cancels its siblings, and the driver's run loop decides what
happens next (:func:`gemot.agent.driver.run_deepagents`). That is right for a TRANSIENT failure: the
driver waits the outage out and resumes, and each cancelled analyst picks its round up from its
checkpoint (``tests/test_stall_pulses.py``). It is wrong for a provider REFUSAL, which the driver
treats as terminal because the same request is refused again: one analyst's content filter or
rejected request cancelled the other analysts' rounds and ended the whole run (gemot#166, luna
lucarelli 63782292_5/6).

So each analyst's model calls are wrapped (:func:`_provider_refusal_report`, innermost in its
middleware stack): a refusal ends that analyst's round with
:data:`gemot.prompts.SUBAGENT_PROVIDER_REFUSAL_TMPL` as its final message, which deepagents hands the
lead as the ``task`` result. Anything else propagates as before. A separate module because the line
it draws, an analyst's failure versus the run's, is the driver's run-loop classification read from
the other side; the predicates are the driver's own, so the two cannot disagree on what a refusal is.
"""

from __future__ import annotations

from typing import Any

from .. import prompts

#: Statuses of a request refused as such: malformed, too large, or unprocessable. Deliberately not
#: 401/402/404 (a key, credit or model that fails every call, which must end the run as it does)
#: nor 408/409/429 (transient, which the driver waits out and resumes).
_REFUSED_STATUSES = frozenset({400, 413, 422})


def analyst_refusal(exc: BaseException) -> str | None:
    """A short reason if ``exc`` is a provider refusal of one request, else ``None``.

    A content filter (by the driver's own markers), or a 400/413/422 read from the client's
    ``status_code``, a mid-stream error event's ``body["code"]``, or an HTTP-200 error body's
    quoted ``code``. Never OpenRouter's 402: credit fails every later call, so it ends the run.
    """
    from .driver import _PROVIDER_CODE_RE, _content_filter_stop_reason, _is_provider_credit

    if _is_provider_credit(exc):
        return None
    if (filtered := _content_filter_stop_reason(exc)) is not None:
        return filtered
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    if not isinstance(status, int) and isinstance(body, dict):
        status = body.get("code")
    if not isinstance(status, int) and (m := _PROVIDER_CODE_RE.search(str(exc))):
        status = int(m.group(1))
    if status not in _REFUSED_STATUSES:
        return None
    return f"HTTP {status} {type(exc).__name__}: {str(exc)[:200]}"


def _provider_refusal_report(stats: dict[str, Any], name: str):
    """Analyst middleware: a provider refusal of one of ``name``'s model calls ends its round with
    the refusal as its report (see the module docstring). Counted in ``stats`` as
    ``subagent_provider_refusals``, which reaches ``result.json`` with the other guard counters."""
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.messages import AIMessage

    def _report(reason: str) -> AIMessage:
        stats["subagent_provider_refusals"] = stats.get("subagent_provider_refusals", 0) + 1
        print(
            f"[provider refusal] {name}: {reason} — ending its round with the refusal as its report"
        )
        return AIMessage(prompts.SUBAGENT_PROVIDER_REFUSAL_TMPL.format(reason=reason))

    class _ProviderRefusalReport(AgentMiddleware):
        # The sync hook is unreachable under this driver (it only calls agent.astream) but is part
        # of the middleware contract, as on the other guards.
        def wrap_model_call(self, request, handler):
            try:
                return handler(request)
            except Exception as exc:
                if (reason := analyst_refusal(exc)) is None:
                    raise
                return _report(reason)

        async def awrap_model_call(self, request, handler):
            try:
                return await handler(request)
            except Exception as exc:
                if (reason := analyst_refusal(exc)) is None:
                    raise
                return _report(reason)

    return _ProviderRefusalReport()
