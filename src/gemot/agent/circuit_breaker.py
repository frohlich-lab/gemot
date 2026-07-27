"""Cut a tool that keeps failing, or whose server has gone away."""

from __future__ import annotations

from typing import Any

from .. import prompts
from ..prompts import _DISABLED_TOOL_HINT_DEFAULT, _DISABLED_TOOL_HINTS

# --- repeated-tool-error circuit breaker --------------------------------------------------
# A broken tool used to be re-proposed cumulatively hundreds of times across a run (a
# trace-review swarm saw one query_data call re-proposed ~798× — up to ~100 pending copies in
# a single step) while the agent learned nothing from the identical failures. This breaker
# counts every tool failure and, after N CONSECUTIVE IDENTICAL errors from the same tool,
# DISABLES that tool for the rest of the run: further calls are short-circuited with a terminal
# message pointing at the working alternative, so the agent stops hammering a dead tool and
# moves on. A SUCCESS resets a tool's streak; DIFFERENT errors reset it too (the agent is still
# exploring), so only a genuinely-stuck tool is cut off.
#
# It works at the TOOL-OBJECT layer (wrapping each loaded tool's coroutine), not as agent
# middleware, deliberately: the heavy data/diagnostic tools (query_data, get_fitting_log, …)
# run INSIDE the analyst subagents, and deepagents 0.6.12's declarative subagents don't
# reliably receive a middleware list (see _builtin_tool_filter). Wrapping the tool object makes
# the breaker travel WITH the tool into whichever agent (lead or analyst) calls it.
_TOOL_ERROR_DISABLE_THRESHOLD = 3   # consecutive identical errors from one tool before it is cut


# Transport/connection failures — a hung or closed MCP server (in practice a flaky HOSTED bio
# server: BioContextAI / OmniPath). These are NOT retryable tool errors: the model can't fix an
# unreachable server, and letting one propagate aborts the whole run (9 runs across sweeps died as
# `stop_reason: ClosedResourceError`, several at n_fits=0). anyio stream errors are matched by class
# NAME (so we don't hard-import anyio internals that vary across versions); the httpx family is
# matched against its public TransportError base in _is_transport_error, so the breaker classifies
# the SAME httpx surface the resume backstop does rather than a hand-listed subset that can drift.
_TRANSPORT_ERROR_NAMES = frozenset({
    "ClosedResourceError", "BrokenResourceError", "EndOfStream",           # anyio streams
})


def _is_transport_error(exc: BaseException) -> bool:
    """True for a transport/connection failure (an unreachable/closed MCP server) as opposed to a
    normal tool error the model should see and retry. Covers the anyio stream errors (by name), the
    builtin ConnectionError, and the WHOLE httpx.TransportError family (ConnectError/ReadError/
    WriteError/ReadTimeout/ConnectTimeout/RemoteProtocolError/PoolTimeout/WriteTimeout/… ) — the same
    surface the run-loop resume backstop matches, so the two never disagree on what is 'transport'."""
    if type(exc).__name__ in _TRANSPORT_ERROR_NAMES or isinstance(exc, ConnectionError):
        return True
    try:
        import httpx
    except Exception:  # pragma: no cover - httpx always present with mcp
        return False
    return isinstance(exc, httpx.TransportError)


def _new_breaker_state() -> dict[str, Any]:
    """Mutable, run-scoped state shared by the wrapped tools and the lead's tool filter."""
    return {"disabled": set(), "n_tool_errors": 0, "streaks": {}, "events": []}


def _breaker_message(name: str, hint: str) -> list[dict[str, str]]:
    """The terminal content block returned in place of a disabled tool's result."""
    return [{"type": "text",
             "text": prompts.TOOL_DISABLED_REPEATED_TMPL.format(name=name, hint=hint)}]


def _transport_breaker_message(name: str, hint: str) -> list[dict[str, str]]:
    """Degraded block for a tool disabled after a transport failure (an unreachable/closed server).

    Distinct from :func:`_breaker_message`: a transport error disables on the FIRST failure, so the
    "failed repeatedly with the same error" wording would misdescribe it."""
    return [{"type": "text",
             "text": prompts.TOOL_DISABLED_TRANSPORT_TMPL.format(name=name, hint=hint)}]


def _result_error(content: Any, artifact: Any) -> str | None:
    """Return the error string if a (content, artifact) tool result is a FAILURE, else None.

    The petab-mcp server returns a failed read/diagnostic tool as a normal result carrying
    ``{"ok": false, "error": ...}`` (not an MCP protocol error), which langchain-mcp-adapters
    exposes in the artifact's ``structured_content`` and in the text content. A rejected
    *operation* (``{"committed": false}`` — e.g. a lint failure the agent should fix and retry)
    is deliberately NOT treated as a breaker error: that is the normal edit/retry feedback loop.
    """
    if isinstance(artifact, dict):
        sc = artifact.get("structured_content")
        if isinstance(sc, dict) and sc.get("ok") is False:
            return str(sc.get("error") or "tool error")
        if artifact.get("isError") or artifact.get("is_error"):
            return "tool error"
    # Fallback: concatenate any text content blocks and look for the {"ok": false} shape.
    text = ""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += str(block.get("text") or "")
            elif isinstance(block, str):
                text += block
    elif isinstance(content, str):
        text = content
    try:
        import json
        obj = json.loads(text)
    except Exception:
        return None
    if isinstance(obj, dict) and obj.get("ok") is False:
        return str(obj.get("error") or "tool error")
    return None


def _record_error(state: dict[str, Any], name: str, err: str) -> bool:
    """Count a tool error; return True iff it TRIPS the breaker (disables the tool now)."""
    state["n_tool_errors"] += 1
    sig = err[:200]
    streak = state["streaks"].get(name)
    if streak and streak["sig"] == sig:
        streak["count"] += 1
    else:
        streak = {"sig": sig, "count": 1}
        state["streaks"][name] = streak
    if streak["count"] >= _TOOL_ERROR_DISABLE_THRESHOLD and name not in state["disabled"]:
        state["disabled"].add(name)
        msg = (f"disabled `{name}` after {streak['count']} consecutive identical errors: "
               f"{sig}")
        state["events"].append(msg)
        print(f"[circuit-breaker] {msg}")
        return True
    return False


def _wrap_tools_with_circuit_breaker(tools: list, state: dict[str, Any],
                                     transport_disable_names: set[str] | None = None) -> list:
    """Wrap each tool's coroutine so repeated identical failures disable it (see above).

    Mutates the tool objects in place (and returns the list) so the breaker rides along into
    the analyst subagents, which hold these same objects. Best-effort: a tool without a
    coroutine is left untouched.

    ``transport_disable_names`` is the set of tool names for which a SINGLE transport error
    (unreachable/closed server) disables the tool immediately, without waiting for the 3-strike
    breaker. This is scoped to the EXTERNAL, optional bio tools (flaky hosted servers the model
    can't fix); a transport error on a CORE petab tool (the local server) instead falls through to
    the normal 3-strike / re-raise path, so one localhost blip can't silently strip ``fit`` /
    ``set_model`` for the rest of the run.
    """
    import inspect
    from functools import wraps

    transport_disable = transport_disable_names or set()

    for tool in tools:
        name = getattr(tool, "name", None)
        orig = getattr(tool, "coroutine", None)
        if name is None or orig is None:
            continue
        hint = _DISABLED_TOOL_HINTS.get(name, _DISABLED_TOOL_HINT_DEFAULT)

        def _make(name: str, orig, hint: str):
            @wraps(orig)
            async def _breaker(runtime=None, **arguments):
                if name in state["disabled"]:
                    return (_breaker_message(name, hint), None)  # short-circuit, no server hit
                try:
                    result = await orig(runtime=runtime, **arguments)
                except Exception as exc:  # a raised tool error (e.g. MCP isError -> ToolException)
                    tripped = _record_error(state, name, f"{type(exc).__name__}: {exc}")
                    # A transport error (unreachable/closed server) on an EXTERNAL bio tool can't be
                    # fixed by the model and aborts the whole run if it propagates — disable that
                    # tool for the rest of the run and return the degraded message NOW, without
                    # waiting for the 3-strike breaker. (Mid-run counterpart to the bio-session
                    # isolation that already handles connect/discovery failures.) A transport error
                    # on a CORE petab tool is NOT auto-disabled here: it falls through to the normal
                    # 3-strike / re-raise path, so a single localhost blip can't strip `fit` or
                    # `set_model` for the rest of the run.
                    if name in transport_disable and _is_transport_error(exc):
                        state["disabled"].add(name)
                        return (_transport_breaker_message(name, hint), None)
                    if tripped:
                        return (_breaker_message(name, hint), None)
                    raise
                content, artifact = result if isinstance(result, tuple) else (result, None)
                err = _result_error(content, artifact)
                if err is None:
                    state["streaks"].pop(name, None)  # a success clears the tool's error streak
                    return result
                if _record_error(state, name, err) and isinstance(content, list):
                    return (content + _breaker_message(name, hint), artifact)
                return result

            try:  # keep the original signature so langchain still injects `runtime` correctly
                _breaker.__signature__ = inspect.signature(orig)
            except (ValueError, TypeError):  # pragma: no cover - defensive
                pass
            return _breaker

        tool.coroutine = _make(name, orig, hint)
    return tools


def _disabled_tool_filter(state: dict[str, Any]):
    """Lead-only middleware: hide a disabled tool from the model so it stops being re-proposed.

    Complements the coroutine wrapper (the wrapper is what actually reaches the analyst
    subagents): for the LEAD's own tools this also drops the tool's schema from the re-sent
    request once it is disabled, saving tokens and removing the temptation to call it again.
    """
    from langchain.agents.middleware.types import AgentMiddleware

    def _name(tool) -> str | None:
        name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        return name if isinstance(name, str) else None

    def _keep(request):
        return [t for t in request.tools if _name(t) not in state["disabled"]]

    class _DropDisabledTools(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return handler(request.override(tools=_keep(request)))

        async def awrap_model_call(self, request, handler):
            return await handler(request.override(tools=_keep(request)))

    return _DropDisabledTools()
