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

#: Set once if appending the turn-budget notice ever fails, so the warning in `_augment` prints
#: one line per process rather than one per turn. See the comment there for why there is no
#: system-prompt fallback.
_TURN_NOTICE_BROKEN = False

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
_MAX_IDLE_MODEL_STEPS = 3  # consecutive no-tool turns to push back on before giving up


#: Effort ladder for a reasoning runaway. Each retry steps DOWN one rung; the run keeps its
#: normal (provider-default) effort everywhere else.
#: Starts at "low", NOT "medium". The driver sets no reasoning_effort, so the baseline is the
#: provider default — and for the gpt-5 family (luna, luna-pro, terra) that default IS medium.
#: A "medium" first rung would re-issue a semantically identical request and burn a second
#: full-budget call (~25 min, ~65k completion tokens on the measured runaways) with no chance
#: of recovering from it. "low" is a real change on every model in the panel; the second rung
#: is "minimal" for providers that accept it, and a provider that rejects an unknown value is
#: handled by the retry's own except (see _retry_failed).
_RUNAWAY_EFFORT_LADDER = ("low", "minimal")


def _is_reasoning_runaway(result: Any) -> bool:
    """Did the model burn its whole completion budget thinking and emit NOTHING?

    The signature is exact and cheap to test: ``finish_reason == "length"`` with empty content
    AND no tool calls. Measured on z-ai/glm-5.3-flash over isensee and lucarelli: four
    consecutive turns each returning completion_tokens=65536 of which reasoning_tokens=65504,
    content "", no tool call, ~25 minutes apiece. The provider truncates mid-thought at its own
    65536 output cap (half the 131072 it advertises) before the model reaches a conclusion.

    Downstream this is indistinguishable from a model that CHOSE to say nothing, so the dispatch
    guard bounces it as an idle turn and the run ends `no_model_built` with zero make_changes
    attempted -- a silent, expensive nothing. It is worth catching precisely rather than by
    lowering reasoning effort globally, which would also remove the reasoning that makes this
    model good on the examples where it does converge.
    """
    # `result` is a langchain ModelResponse (fields: .result — a list[BaseMessage] — and
    # .structured_response) or, defensively, a bare AIMessage. Same contract as
    # _dedupe_response below. Reading the wrong attribute here is silent: getattr degrades to
    # the [result] fallback and the detector never fires — so this is pinned by
    # tests/test_reasoning_runaway.py against the real ModelResponse class.
    msgs = getattr(result, "result", None) or ([result] if result is not None else [])
    # The verdict belongs to the TERMINAL message — the last one carrying a finish_reason —
    # and to nothing before it. `.result` may hold more than one message (langchain appends a
    # ToolMessage when a tool produced the structured output), and scanning past a healthy
    # final message to convict on an earlier truncated one made the ladder discard a good
    # answer and re-call the provider. Verified: [truncated(length,""), healthy("...", stop)]
    # read as a runaway before this.
    for m in reversed(list(msgs)):
        # Skip on MESSAGE TYPE, not on whether finish_reason happens to be present: a healthy
        # final AI turn whose provider/middleware set no finish_reason would otherwise be
        # skipped, letting an earlier truncated turn decide — reinstating the very bug the
        # terminal-message rule fixed. Only a trailing non-AI message (the ToolMessage
        # langchain appends for structured output) is skippable.
        if getattr(m, "type", None) != "ai":
            continue
        if getattr(m, "tool_calls", None):
            return False
        meta = getattr(m, "response_metadata", None) or {}
        return meta.get("finish_reason") == "length" and _content_is_empty(
            getattr(m, "content", None)
        )
    return False


def _content_is_empty(content: Any) -> bool:
    """Nothing a downstream consumer could act on: None, whitespace, or blocks with no text.

    Reasoning models return content as a LIST of blocks as often as a bare string, and a
    truncated turn can carry a reasoning block with no text field — both must read as empty or
    the runaway detector misses exactly the responses it exists to catch.
    """
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                if block.strip():
                    return False
            elif isinstance(block, dict):
                if str(block.get("text") or "").strip():
                    return False
        return True
    return False


def _reasoning_runaway_guard(
    stats: dict[str, Any], *, ladder: "tuple[str, ...]" = _RUNAWAY_EFFORT_LADDER
):
    """Retry a truncated-mid-thought turn at reduced reasoning effort, then restore.

    Only the offending CALL is retried, and only while it keeps coming back empty; the request
    is otherwise untouched, so a run that never trips this never sees a changed effort. Bounded
    by the ladder length, so a model that returns empty at every effort costs a fixed number of
    extra calls rather than looping.
    """
    from langchain.agents.middleware.types import AgentMiddleware

    def _retry_settings(request, effort):
        settings = dict(getattr(request, "model_settings", None) or {})
        settings["reasoning_effort"] = effort
        return settings

    def _ladder_step(rung, request, result):
        """One decision of the retry ladder, shared verbatim by the sync and async hooks so the
        detection/counting/announce logic cannot diverge between them.

        Given the result of the call at ``rung`` (0 = the original), return the override request
        for the next rung, or None to hand ``result`` back — either because it is healthy (a
        recovery, when rung > 0) or because the ladder is exhausted. A runaway is counted ONCE,
        at detection, so ``reasoning_runaways`` counts events and ``reasoning_runaway_recovered``
        the subset that a retry rescued.
        """
        if not _is_reasoning_runaway(result):
            if rung > 0:
                stats["reasoning_runaway_recovered"] += 1
            return None
        if rung == 0:
            stats["reasoning_runaways"] += 1
        if rung >= len(ladder):
            return None  # empty at every effort: hand it back honestly rather than loop
        effort = ladder[rung]
        print(
            f"[reasoning guard] the model spent its whole completion budget thinking "
            f"and returned nothing — retrying this call at reasoning_effort={effort}"
        )
        try:
            return request.override(model_settings=_retry_settings(request, effort))
        except Exception as exc:  # pragma: no cover - older/newer ModelRequest shapes
            # Building the retry must not be able to end the run either. `override` is a
            # langchain API (dataclasses.replace over a fixed override key set), so a rename or
            # a dropped field would otherwise raise out of wrap_model_call into the driver's
            # generic handler — the very outcome _retry_failed exists to prevent, reached one
            # line before its try block. _subagent_turn_limit._augment guards the identical
            # call the same way.
            _retry_failed(rung, exc, result)
            return None

    def _retry_failed(rung, exc, original):
        """A retry raised: hand back the ORIGINAL truncated result instead of propagating.

        The guard must be strictly non-worsening. `reasoning_effort` is a request field not
        every OpenRouter-served model accepts, and an exception escaping here would leave the
        model node -> astream -> the driver's generic handler and end the whole run on a raw
        exception string. Without the guard that same turn is merely an idle turn the dispatch
        guard nudges, so swallowing the retry failure keeps the un-guarded behaviour as the
        floor. Counted so an endpoint that rejects the parameter is visible rather than silent.
        """
        stats["reasoning_runaway_retry_errors"] = stats.get("reasoning_runaway_retry_errors", 0) + 1
        print(
            f"[reasoning guard] retry at rung {rung} failed ({type(exc).__name__}: {exc}); "
            "keeping the original truncated turn and letting the dispatch guard handle it"
        )
        return original

    class _ReasoningRunawayGuard(AgentMiddleware):
        # The two hook bodies are the minimal sync/async shells around _ladder_step; only the
        # await differs. The sync one is unreachable under this driver (it only ever calls
        # agent.astream) but is part of the middleware contract, so it stays — as a shell.
        def wrap_model_call(self, request, handler):
            result, rung = handler(request), 0
            while (retry := _ladder_step(rung, request, result)) is not None:
                rung += 1
                try:
                    result = handler(retry)
                except Exception as exc:
                    return _retry_failed(rung, exc, result)
            return result

        async def awrap_model_call(self, request, handler):
            result, rung = await handler(request), 0
            while (retry := _ladder_step(rung, request, result)) is not None:
                rung += 1
                try:
                    result = await handler(retry)
                except Exception as exc:
                    return _retry_failed(rung, exc, result)
            return result

    return _ReasoningRunawayGuard()


def new_guard_stats() -> dict[str, Any]:
    """A fresh, fully-seeded guard-stats dict — the ONE definition of what the guards record.

    Seeded up front rather than by each guard as it first fires, so ``result.json`` carries an
    explicit zero for every counter even when a guard never triggers (or, for a deepagents build
    without the ``middleware`` parameter, is never constructed at all). The guards increment these
    keys in place, so a caller that builds one by hand seeds it from here; the driver spreads the
    whole dict into its return, and ``runner.provenance._agent_meta`` spreads that into
    ``result.json``, so a counter added here needs no second edit to be recorded.
    """
    return {
        # tool-dispatch guard
        "tools_executed": 0,
        "dropped_dup_tool_calls": 0,
        "idle_nudges": 0,
        "gave_up_no_tools": False,
        # task-arg guard
        "unexpected_task_kwargs": [],
        "task_arg_rejections": 0,
        # empty-report guard
        "empty_subagent_reports": 0,
        "empty_subagent_retries": 0,
        "empty_subagent_markers": 0,
        # reasoning-runaway guard
        "reasoning_runaways": 0,
        "reasoning_runaway_recovered": 0,
        "reasoning_runaway_retry_errors": 0,
        # driver-side: how often the stall watchdog could not read the graph state (it then
        # falls back to the run-default fit window, so a non-zero count means some step was
        # sized by the default rather than by its pending fit).
        "unreadable_graph_state": 0,
        # panel composition, not a counter
        "subagent_names": [],
        "general_purpose_subagent": None,
    }


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
            print(
                f"[tool-dispatch guard] no tool has fired yet and the model replied with text "
                f"only — pushing back ({stats['idle_nudges']}/{max_idle_steps})"
            )
            return {"messages": [HumanMessage(content=_IDLE_NUDGE)], "jump_to": "model"}
        stats["gave_up_no_tools"] = True
        print(
            "[tool-dispatch guard] the model never dispatched a tool after "
            f"{max_idle_steps} push-backs — ending the run (no_tools_executed)"
        )
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
_MAX_TASK_ARG_REJECTIONS = 3  # bounded push-back on unexpected task kwargs per run, then relent


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
            set(stats.get("unexpected_task_kwargs") or []) | set(bad)
        )
        if stats.get("task_arg_rejections", 0) >= max_rejections:
            print(
                f"[task-arg guard] unexpected task kwargs {bad} — relenting after "
                f"{max_rejections} rejections; dispatching (the extras are dropped)."
            )
            return None
        stats["task_arg_rejections"] = stats.get("task_arg_rejections", 0) + 1
        print(
            f"[task-arg guard] rejecting task(...) with unexpected kwargs {bad} "
            f"({stats['task_arg_rejections']}/{max_rejections})"
        )
        return ToolMessage(
            content=_task_arg_reject_message(bad), tool_call_id=tc.get("id") or "", status="error"
        )

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
_MAX_EMPTY_REPORT_RETRIES = 1  # re-dispatch a subagent that returned an empty report at most once


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
        print(
            "[empty-report guard] report still empty — substituting the visible marker "
            f"{prompts.SUBAGENT_EMPTY_REPORT_MARKER!r}"
        )
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
                print(
                    f"[empty-report guard] re-dispatching {(tc.get('args') or {}).get('subagent_type')!r} "
                    "once with a corrective nudge"
                )
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
                print(
                    f"[empty-report guard] re-dispatching {(tc.get('args') or {}).get('subagent_type')!r} "
                    "once with a corrective nudge"
                )
                result = await handler(request.override(tool_call=_nudged(tc)))
                tm = _tool_message(result)
                if tm is None or not _is_blank_report(tm.content):
                    return result  # the nudge produced a real report — forward it
            return _with_marker(result, tm)

    return _EmptyReportGuard()


# --- subagent turn limit ------------------------------------------------------------------
# A subagent is invoked with a FRESH `recursion_limit` on every `task` call (deepagents seeds
# the child run from the ambient parent config, and the child's own bound config wins), so the
# lead's limit bounds how many times it may DELEGATE, not how long each delegation runs. One
# observed run made 1,493 LLM calls under recursion_limit=250 with zero fits and zero edits:
# the work all happened inside subagents that were free to explore indefinitely, and every
# progress-based early stop stayed silent because those depend on FITTED ops (session.py's
# should_stop), of which there were none.
#
# The limit is counted from the STATE (AIMessages so far), not an instance counter: the spec is
# built once and shared across every invocation, so a counter on the middleware would leak the
# first delegation's turns into the second and shrink each successive call's budget to zero.
# Escalating notice as a delegation nears its cap. Injected into the OUTGOING REQUEST
# (wrap_model_call), never appended to state: at the cap the model's last turn is usually a tool
# call, and adding a HumanMessage after an AIMessage that carries `tool_calls` -- then returning to
# the model without running those tools -- produces a message sequence providers reject outright
# ("an assistant message with tool_calls must be followed by tool messages"). The existing
# _ToolDispatchGuard sidesteps this by nudging only on text-only turns; a turn limit cannot, since
# it must land wherever the budget runs out. Modifying the request leaves state well-formed.
# Sourced from prompts.py so `prompt_version()` moves when the wording does — see
# prompts.SUBAGENT_TURN_NOTICES for why this text cannot live here.
_TURN_NOTICES = prompts.SUBAGENT_TURN_NOTICES


def _turn_notice(turns_used: int, cap: int) -> "str | None":
    """The sternest notice earned at ``turns_used``, or None while there is room to work.

    Thresholds are clamped to at most HALF the cap, so a small cap cannot warn from turn zero. The
    10/5/1 numbers are sized against a 20-turn budget; applied absolutely, a cap of 8 would fire
    "start converging" on the delegation's very first turn (left = 8 <= 10) and a cap of 5 would
    open on the WARNING. `cap // 2` leaves the shipped 20-turn caps EXACTLY as they were (10 -> 10)
    while keeping the notices a genuine approach signal at any smaller cap.
    """
    left = cap - turns_used
    for threshold, text in sorted(_TURN_NOTICES, key=lambda t: t[0]):
        earned = min(threshold, max(1, cap // 2))
        if left <= earned:
            return text.format(left=max(left, 0), cap=cap)
    return None


def _subagent_turn_limit(max_turns: int):
    """Bound one subagent delegation to ``max_turns`` model calls, with warnings on approach.

    Counted from the STATE (AIMessages so far), not an instance counter: the spec is built once
    and reused for every `task` call, so a counter on the middleware would leak the first
    delegation's turns into the second and starve it.
    """
    from langchain.agents.middleware.types import AgentMiddleware, hook_config
    from langchain_core.messages import AIMessage

    def _turns(messages) -> int:
        return sum(1 for m in messages or [] if isinstance(m, AIMessage))

    def _augment(request):
        """Append the notice as a TRAILING message, never to the system prompt.

        The notice carries a countdown (``left`` decrements every turn), so putting it in the
        system prompt made message[0] a DIFFERENT STRING on every turn once the first notice
        fired — invalidating the whole prefix cache from the very first token, every turn, for
        the rest of the delegation. Measured on the 2026-08-14 draft-editing A/B: at the 11th
        call of a delegation (turns_used=10, the first notice) the model_editor's cache-hit rate
        fell 0.86-0.98 -> 0.72 and per-call excess novel tokens jumped ~1.6k -> ~4.9k, on 361/361
        calls at that depth; cache_write jumped 248 -> 1,594 in lockstep. Appended at the END the
        same text costs its own length once and leaves every earlier token cacheable.

        Injected into the OUTGOING REQUEST only (never into state), so it is not persisted and
        cannot accumulate across turns — the property the system-prompt version also had, and the
        reason this is a request hook rather than a state write.
        """
        notice = _turn_notice(_turns(getattr(request, "messages", None)), max_turns)
        if notice is None:
            return request
        from langchain_core.messages import HumanMessage

        try:
            messages = list(getattr(request, "messages", None) or [])
            return request.override(messages=[*messages, HumanMessage(content=notice)])
        except Exception as exc:  # pragma: no cover - older/newer request shapes
            # NO system-prompt fallback. Injecting the notice there is the regression this
            # function exists to remove: the countdown makes message[0] a different string every
            # turn, which invalidates the whole prefix cache from the first token (measured above:
            # hit rate 0.86-0.98 -> 0.72, ~1.6k -> ~4.9k excess novel tokens per call, 361/361
            # calls at that depth). Paying that on EVERY remaining turn to deliver a courtesy
            # countdown is a far worse trade than dropping the countdown — the hard cap in
            # `_check` still fires, so the delegation is bounded either way; it just gets cut off
            # without advance warning.
            #
            # And it is LOUD. A silent `return request` here would mean a langchain request-shape
            # change quietly disables turn notices for every delegation in every run, with the
            # only symptom being subagents that stop tapering — indistinguishable from a prompt
            # regression, and invisible in the trace. Printed once per process so a shape change
            # is obvious in the first run's log without flooding it every turn.
            global _TURN_NOTICE_BROKEN
            if not _TURN_NOTICE_BROKEN:
                _TURN_NOTICE_BROKEN = True
                print(
                    f"[agent] WARNING: could not append the turn-budget notice to the request "
                    f"({type(exc).__name__}: {exc}). Delegations will run to their hard cap "
                    f"with no countdown; `_augment` needs updating for this request shape. "
                    f"NOT falling back to the system prompt — that destroys the prefix cache."
                )
            return request

    def _check(state):
        # The hard stop. `jump_to: end` is safe with a tool call outstanding (nothing further is
        # sent to the provider), unlike returning to the model; a delegation cut off mid-call
        # yields a thin report, which _EmptyReportGuard already covers.
        # `>=`, not `>`: at `max_turns` AIMessages the budget is SPENT, and `>` allowed a
        # (max_turns + 1)-th model call -- an off-by-one against this factory's own contract
        # ("bound one delegation to `max_turns` model calls") and against the measured sizing.
        if _turns(state.get("messages")) >= max_turns:
            print(f"[subagent] turn limit ({max_turns}) reached — ending this delegation")
            return {"jump_to": "end"}
        return None

    class _SubagentTurnLimit(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return handler(_augment(request))

        async def awrap_model_call(self, request, handler):
            return await handler(_augment(request))

        @hook_config(can_jump_to=["end"])
        def after_model(self, state, runtime):
            return _check(state)

        @hook_config(can_jump_to=["end"])
        async def aafter_model(self, state, runtime):
            return _check(state)

    return _SubagentTurnLimit()
