"""Lead-only middleware that keeps the agent loop honest.

Three guards:

* the **tool-dispatch guard** collapses duplicate calls within a batch and pushes back on
  no-tool turns, so a run cannot silently end having executed nothing;
* the **task-arg guard** rejects a ``task`` call carrying kwargs the deepagents tool would
  silently drop, so a subagent is never dispatched on truncated input;
* the **empty-report guard** catches a ``task`` subagent that returns an empty/whitespace report
  (its final completion was empty), re-dispatching it once and then marking the loss visibly, so
  the lead is never silently handed ``""`` in place of an analyst's work.

The first two operate on the tool calls the model PROPOSES; the empty-report guard operates on the
``task`` RESULT (repeated *tool errors* remain the circuit breaker's job in :mod:`.circuit_breaker`).
All three mutate a caller-owned ``stats`` dict the driver reads after the run.
"""

from __future__ import annotations

from typing import Any

from .. import prompts
from ..prompts import _IDLE_NUDGE
from .dedup import _dedupe_raw_tool_calls, _dedupe_tool_calls

# --- tool-dispatch guard: make sure proposed tool calls actually FIRE -----------------------
# A whole class of runs (originally seen on the blasi zero-shot build) never executed a single
# tool: the lead kept emitting text-only assistant turns — narrating what it *would* do — and the
# ReAct loop, seeing an assistant message with no tool_calls, treated each one as "done" and ended
# with nothing built (n_tool=0, terminating in a handful of steps, nowhere near the recursion
# limit). Two things harden the loop against that here (orthogonal to the repeated-error circuit
# breaker above, which handles tools that keep FAILING — this handles tools that never fire AT ALL):
#   1. de-duplicate identical tool calls WITHIN a batch: some providers emit the same (name, args)
#      twice — occasionally with a repeated id — in one assistant turn, which wastes a fit and can
#      break the tool-call/result pairing on the follow-up request. We collapse exact duplicates to
#      the first occurrence before they dispatch, so results feed back cleanly.
#   2. don't let an idle (no-tool) turn silently END the run while NOTHING has fired yet: push back
#      with a corrective message and loop back to the model, up to K times, so the model gets a real
#      chance to emit an actual tool call. If it still hasn't after K nudges we stop cleanly and the
#      run is labelled ``no_tools_executed`` (see run_deepagents) — a degenerate run that is
#      EXCLUDED as such rather than left as an ambiguous, silent early "completion".
# The nudge fires ONLY while zero tools have executed, so it can never interfere with a genuine
# finish (which always comes AFTER make_changes has run).
_MAX_IDLE_MODEL_STEPS = 3   # consecutive no-tool turns to push back on before giving up


def _tool_dispatch_guard(stats: dict[str, Any], *, max_idle_steps: int = _MAX_IDLE_MODEL_STEPS):
    """Lead-only middleware implementing the two hardening steps described above.

    ``stats`` is a caller-owned dict the middleware mutates so run_deepagents can inspect the
    outcome after the run: ``tools_executed`` (real tool dispatches, incl. ``task`` delegations),
    ``dropped_dup_tool_calls`` (batch duplicates collapsed), ``idle_nudges`` (no-tool turns pushed
    back on), and ``gave_up_no_tools`` (True once the push-back budget was exhausted). Defined
    lazily so importing this module doesn't require langchain — only the deepagents driver does.
    """
    from langchain.agents.middleware.types import AgentMiddleware, hook_config
    from langchain_core.messages import AIMessage, HumanMessage

    def _dedupe_response(response):
        # response is a ModelResponse (has .result: list[BaseMessage]) or a bare AIMessage.
        msgs = getattr(response, "result", None)
        target = msgs[-1] if msgs else (response if isinstance(response, AIMessage) else None)
        if isinstance(target, AIMessage) and getattr(target, "tool_calls", None):
            kept, dropped = _dedupe_tool_calls(target.tool_calls)
            if dropped:
                stats["dropped_dup_tool_calls"] += dropped
                # Keep additional_kwargs["tool_calls"] (the raw provider copy) consistent with the
                # trimmed .tool_calls — langchain-openai serialises from .tool_calls when it is
                # non-empty, but collapse the raw copy the same way so nothing can resurrect a
                # duplicate (de-duping by name+args, not id, since a repeated call may reuse its id).
                ak = dict(getattr(target, "additional_kwargs", None) or {})
                if isinstance(ak.get("tool_calls"), list):
                    ak["tool_calls"] = _dedupe_raw_tool_calls(ak["tool_calls"])
                new = target.model_copy(update={"tool_calls": kept, "additional_kwargs": ak})
                if msgs:
                    msgs[-1] = new
                else:
                    return new
        return response

    def _guard(state):
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        if not isinstance(last, AIMessage) or getattr(last, "tool_calls", None):
            return None  # a tool call was proposed (or not an AI turn) — let it dispatch/route
        if stats["tools_executed"] > 0:
            return None  # a no-tool turn AFTER tools have run is a genuine finish — allow it
        # Degenerate: nothing has fired yet and the model answered with text only.
        if stats["idle_nudges"] < max_idle_steps:
            stats["idle_nudges"] += 1
            print(f"[tool-dispatch guard] no tool has fired yet and the model replied with text "
                  f"only — pushing back ({stats['idle_nudges']}/{max_idle_steps})")
            return {"messages": [HumanMessage(content=_IDLE_NUDGE)], "jump_to": "model"}
        stats["gave_up_no_tools"] = True
        print("[tool-dispatch guard] the model never dispatched a tool after "
              f"{max_idle_steps} push-backs — ending the run (no_tools_executed)")
        return {"jump_to": "end"}

    class _ToolDispatchGuard(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return _dedupe_response(handler(request))

        async def awrap_model_call(self, request, handler):
            return _dedupe_response(await handler(request))

        def wrap_tool_call(self, request, handler):
            out = handler(request)
            stats["tools_executed"] += 1
            return out

        async def awrap_tool_call(self, request, handler):
            out = await handler(request)
            stats["tools_executed"] += 1
            return out

        @hook_config(can_jump_to=["model", "end"])
        def after_model(self, state, runtime):
            return _guard(state)

        @hook_config(can_jump_to=["model", "end"])
        async def aafter_model(self, state, runtime):
            return _guard(state)

    return _ToolDispatchGuard()


# --- task-arg guard: reject a `task` call carrying kwargs the tool silently drops --------------
# The deepagents `task` tool takes ONLY (description, subagent_type). The lead used to assemble the
# biocurator's entity names into a `prompt=` argument, which `task` SILENTLY DISCARDS — so the
# biocurator (which, unlike the analysts, holds no petab tool and cannot self-serve) received only
# the short `description` and its whole input channel was broken (issue #91). The entity names are
# now injected into the biocurator directly (hand-curated per scenario; _build_biocurator_subagent);
# this guard closes the other half: it makes the dropped-kwarg bug LOUD instead of silent by
# bouncing such a call back to the model with a corrective error, so the model re-issues it with
# everything in `description` (this protects the analysts' `prompt=` drops too).
_TASK_ALLOWED_ARGS = frozenset({"description", "subagent_type"})
_MAX_TASK_ARG_REJECTIONS = 3   # bounded push-back on unexpected task kwargs per run, then relent


def _unexpected_task_kwargs(args: "dict | None") -> "list[str]":
    """The keys a ``task(...)`` call carried BEYOND (description, subagent_type) — e.g. a ``prompt=``
    the deepagents task tool silently drops. Order-stable. Pure — unit-testable without langchain."""
    if not isinstance(args, dict):
        return []
    return [k for k in args if k not in _TASK_ALLOWED_ARGS]


def _task_arg_reject_message(bad_keys: "list[str]") -> str:
    """The corrective error returned to the model in place of dispatching a mis-called ``task``."""
    return prompts.TASK_ARG_REJECT_TMPL.format(bad_keys=bad_keys)


def _task_arg_guard(stats: dict[str, Any], *, max_rejections: int = _MAX_TASK_ARG_REJECTIONS):
    """Lead-only middleware: reject a ``task`` call that carries unexpected kwargs (see above).

    ``stats`` is a caller-owned dict the middleware mutates so run_deepagents can report the
    outcome: ``unexpected_task_kwargs`` (the offending keys seen across the run) and
    ``task_arg_rejections`` (how many calls were bounced). It works by returning an error
    ``ToolMessage`` in place of dispatching (the same short-circuit shape the circuit breaker
    uses) — it does NOT call ``handler``, so the subagent never runs on truncated input. The
    push-back is bounded: once ``max_rejections`` is hit it RELENTS and lets the call through (the
    task tool drops the extras — the original behavior) so a stubborn model still makes progress;
    the occurrence is still recorded. Defined lazily so importing this module doesn't require
    langchain — only the deepagents driver does.
    """
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.messages import ToolMessage

    def _reject(request):
        """Return an error ToolMessage to bounce this call, or None to dispatch normally."""
        tc = getattr(request, "tool_call", None) or {}
        if tc.get("name") != "task":
            return None
        bad = _unexpected_task_kwargs(tc.get("args"))
        if not bad:
            return None
        stats["unexpected_task_kwargs"] = sorted(
            set(stats.get("unexpected_task_kwargs") or []) | set(bad))
        if stats.get("task_arg_rejections", 0) >= max_rejections:
            print(f"[task-arg guard] unexpected task kwargs {bad} — relenting after "
                  f"{max_rejections} rejections; dispatching (the extras are dropped).")
            return None
        stats["task_arg_rejections"] = stats.get("task_arg_rejections", 0) + 1
        print(f"[task-arg guard] rejecting task(...) with unexpected kwargs {bad} "
              f"({stats['task_arg_rejections']}/{max_rejections})")
        return ToolMessage(content=_task_arg_reject_message(bad),
                           tool_call_id=tc.get("id") or "", status="error")

    class _TaskArgGuard(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            rejection = _reject(request)
            return rejection if rejection is not None else handler(request)

        async def awrap_tool_call(self, request, handler):
            rejection = _reject(request)
            return rejection if rejection is not None else await handler(request)

    return _TaskArgGuard()


# --- empty-report guard: don't hand the lead a subagent's "" as if it were a report ------------
# A subagent's whole output to the lead is the deepagents `task` tool's ToolMessage, whose content
# is the subagent's LAST non-empty assistant text. When a subagent's final turn is an empty
# completion (finish_reason=stop, no content) with no earlier turn carrying text either — seen live
# on a blasi run where signal_engineer did 27 tool-calling turns of residual analysis, then stopped
# with empty content (len 0) — that content is "". deepagents forwards the "" and the lead silently
# receives nothing: the whole subagent run is wasted, and a "" may read to the lead as "the analyst
# found nothing to report". It is intermittent — the same analyst returned full reports at other
# steps — so this cannot be fixed in the persona; it needs a guard on the RETURN. On a blank `task`
# return this middleware re-dispatches the subagent ONCE with a corrective nudge (the deepagents
# `execute` handler is documented as re-invokable for exactly this retry pattern); if the retry is
# ALSO blank it substitutes a visible marker, so the loss is legible rather than a silent "".
_MAX_EMPTY_REPORT_RETRIES = 1   # re-dispatch a subagent that returned an empty report at most once


def _report_text(content: Any) -> str:
    """A ToolMessage's ``content`` flattened to plain text. Content is normally a ``str``, but
    langchain also permits a list of blocks (a bare ``str`` or a ``{"type": "text", "text": ...}``
    dict); concatenate the text of those. Pure — unit-testable without langchain."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content)


def _is_blank_report(content: Any) -> bool:
    """True iff a subagent's returned report is empty or whitespace-only (see the guard above).
    Pure — unit-testable without langchain, like :func:`_unexpected_task_kwargs`."""
    return not _report_text(content).strip()


def _empty_report_guard(stats: dict[str, Any], *, retries: int = _MAX_EMPTY_REPORT_RETRIES):
    """Lead-only middleware: catch a ``task`` subagent that returns an empty/whitespace report.

    ``stats`` is a caller-owned dict the middleware mutates so run_deepagents can report the
    outcome in result.json: ``empty_subagent_reports`` (blank ``task`` returns detected),
    ``empty_subagent_retries`` (subagents re-dispatched over one), and ``empty_subagent_markers``
    (blank results replaced with the visible marker because the retry was also blank, or the retry
    budget was zero). It wraps the tool call so it can invoke the handler AGAIN — on a blank return
    it re-dispatches once with a corrective nudge folded into ``description`` and, failing that,
    substitutes the marker. It is placed INNERMOST (see run_deepagents) so a re-dispatch does not
    re-enter the tool-dispatch guard and double-count ``tools_executed``. Defined lazily so
    importing this module doesn't require langchain — only the deepagents driver does.
    """
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    def _tool_message(result):
        """The ToolMessage a ``task`` call returned — whether wrapped in a ``Command`` (the normal
        case) or returned bare. ``None`` for anything else (e.g. the tool's plain-``str``
        'subagent does not exist' error), which the guard leaves untouched."""
        if isinstance(result, Command):
            update = result.update if isinstance(result.update, dict) else {}
            for msg in reversed(update.get("messages") or []):
                if isinstance(msg, ToolMessage):
                    return msg
            return None
        return result if isinstance(result, ToolMessage) else None

    def _with_marker(result, tm):
        """``result`` with the blank ToolMessage ``tm`` swapped for the visible marker, preserving
        the ``Command`` wrapper and its other state updates and the original ``tool_call_id``."""
        stats["empty_subagent_markers"] = stats.get("empty_subagent_markers", 0) + 1
        print("[empty-report guard] report still empty — substituting the visible marker "
              f"{prompts.SUBAGENT_EMPTY_REPORT_MARKER!r}")
        marked = ToolMessage(prompts.SUBAGENT_EMPTY_REPORT_MARKER, tool_call_id=tm.tool_call_id)
        if isinstance(result, Command):
            update = dict(result.update)
            update["messages"] = [marked if m is tm else m for m in (update.get("messages") or [])]
            return Command(update=update)
        return marked

    def _nudged(tc):
        """A copy of the ``task`` tool call with the corrective nudge appended to ``description``."""
        args = dict(tc.get("args") or {})
        desc = str(args.get("description") or "")
        args["description"] = f"{desc}\n\n{prompts.SUBAGENT_EMPTY_REPORT_NUDGE}".strip()
        return {**tc, "args": args}

    def _is_blank_task(request, result):
        """The ToolMessage to act on if this is a ``task`` call that came back blank, else None."""
        tc = getattr(request, "tool_call", None) or {}
        if tc.get("name") != "task":
            return None
        tm = _tool_message(result)
        if tm is None or not _is_blank_report(tm.content):
            return None
        stats["empty_subagent_reports"] = stats.get("empty_subagent_reports", 0) + 1
        subagent = (tc.get("args") or {}).get("subagent_type")
        print(f"[empty-report guard] subagent {subagent!r} returned an empty report")
        return tm

    class _EmptyReportGuard(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            result = handler(request)
            tm = _is_blank_task(request, result)
            if tm is None:
                return result
            if retries > 0:
                stats["empty_subagent_retries"] = stats.get("empty_subagent_retries", 0) + 1
                tc = request.tool_call
                print(f"[empty-report guard] re-dispatching {(tc.get('args') or {}).get('subagent_type')!r} "
                      "once with a corrective nudge")
                result = handler(request.override(tool_call=_nudged(tc)))
                tm = _tool_message(result)
                if tm is None or not _is_blank_report(tm.content):
                    return result  # the nudge produced a real report — forward it
            return _with_marker(result, tm)

        async def awrap_tool_call(self, request, handler):
            result = await handler(request)
            tm = _is_blank_task(request, result)
            if tm is None:
                return result
            if retries > 0:
                stats["empty_subagent_retries"] = stats.get("empty_subagent_retries", 0) + 1
                tc = request.tool_call
                print(f"[empty-report guard] re-dispatching {(tc.get('args') or {}).get('subagent_type')!r} "
                      "once with a corrective nudge")
                result = await handler(request.override(tool_call=_nudged(tc)))
                tm = _tool_message(result)
                if tm is None or not _is_blank_report(tm.content):
                    return result  # the nudge produced a real report — forward it
            return _with_marker(result, tm)

    return _EmptyReportGuard()
