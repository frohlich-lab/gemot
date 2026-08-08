"""Cut a tool that keeps failing, or whose server has gone away."""

from __future__ import annotations

import json
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

# Tools the breaker may NEVER disable, however often they fail. `make_changes` is the run's only
# editor: disabling it does not degrade the run, it GUARANTEES `no_model_built`, because nothing
# else can author a model or move a bound. Observed in the August sweep — xiaomi/mimo-v2.5 sent
# `parameters` as a JSON *string*, three identical client-side validation errors tripped the
# breaker, and the run then received "TOOL DISABLED ... Do NOT call it again" 98 times and scored
# nothing. The transport path already refuses to strip this tool (see the note in
# `_wrap_tools_with_circuit_breaker`); the 3-strike path simply never got the same guard.
# These tools escalate instead: the error is returned with an added instruction to CHANGE the
# call's shape, which is recoverable, where a disable is not.
_NEVER_DISABLE = frozenset({"make_changes"})

_ESCALATION = (
    " — this is the {n}th identical failure. `{name}` is the only tool that can change the model, "
    "so it stays available, but REPEATING THIS CALL WILL FAIL AGAIN. Change the SHAPE of the "
    "arguments, not just their values: check each argument's type against the tool schema "
    "(a list must be a JSON array, not a string containing one), and re-read the error above."
)

# JSON-schema types that a bare string is never a valid value for. A model that emits
# `parameters: "[{...}]"` (a JSON-encoded string where an array is required) fails validation
# BEFORE the server is reached — the pydantic error is titled `call[<tool>]`, i.e. raised against
# the published tool schema on the client, so neither FastMCP's `pre_parse_json` nor any
# server-side validator can rescue it. Coercing here, in the client-side wrapper, is the only
# layer that can.
_JSON_CONTAINER_TYPES = frozenset({"array", "object"})


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
    return {"disabled": set(), "n_tool_errors": 0, "streaks": {}, "events": [],
            # Separate from `streaks`: an operation REJECTION is not a tool error (see
            # `_result_error`) and must never disable anything, but a run can still spend its
            # whole budget re-sending one -- gemini 106 identical rejections, luna 246 of 265.
            "reject_streaks": {}, "n_rejection_escalations": 0,
            # Per-tool CALL outcomes, counted here because this wrapper is the only place that
            # sees every attempt. `_ToolClock.note_result` (runner/clock.py) tallies only calls
            # that reach `session.call_tool` AND return a parseable payload, so it is blind to the
            # three ways a call can fail without ever touching the server -- which is how a
            # mimo-v2.5 run that made 134 failing make_changes calls wrote
            # `edit_rejects: {attempted: 0, rejected: 0, reject_rate: null}`.
            "calls": {}}


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
    if name in _NEVER_DISABLE:
        # Not disabled — but still RECORDED. Returning silently here would leave a run that
        # failed this call 200 times reporting `disabled_tools: []`, no breaker line in the log
        # and nothing in `events`: a run that scored nothing with no recorded reason, which is
        # the exact invisibility the August sweep had to be reverse-engineered from.
        if streak["count"] == _TOOL_ERROR_DISABLE_THRESHOLD:
            msg = (f"`{name}` has failed {streak['count']} times with the same error and is the "
                   f"only editor, so it is NOT disabled; escalating instead: {sig}")
            state["events"].append(msg)
            print(f"[circuit-breaker] {msg}")
        return False
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
    ``make_changes`` for the rest of the run.
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
        containers = _container_arg_names(tool)

        def _make(name: str, orig, hint: str, containers: dict):
            @wraps(orig)
            async def _breaker(runtime=None, **arguments):
                if name in state["disabled"]:
                    _note_call(state, name, _CALL_DISABLED)
                    return (_breaker_message(name, hint), None)  # short-circuit, no server hit
                arguments = _coerce_json_string_args(arguments, containers)
                try:
                    result = await orig(runtime=runtime, **arguments)
                except Exception as exc:  # a raised tool error (e.g. MCP isError -> ToolException)
                    _note_call(state, name, _CALL_CLIENT_INVALID
                               if _is_client_validation_error(exc) else _CALL_TOOL_ERROR)
                    sig = f"{type(exc).__name__}: {exc}"
                    tripped = _record_error(state, name, sig)
                    if name in _NEVER_DISABLE:
                        # Never stripped, so the model keeps the tool — but a bare repeat of the
                        # same error teaches it nothing. Escalate the wording once the streak is
                        # real. Raise a plain ToolException rather than rebuilding `type(exc)`:
                        # the exception this path actually receives is langchain-mcp-adapters'
                        # `_MCPToolExecutionError`, whose __init__ takes a list of content blocks,
                        # so `type(exc)(some_string)` yields an object whose message is
                        # "string indices must be integers" — losing the escalation AND the
                        # original error. ToolException takes a string and keeps the tool-error
                        # handling path that every caller here already expects.
                        n = state["streaks"].get(name, {}).get("count", 1)
                        if n >= _TOOL_ERROR_DISABLE_THRESHOLD:
                            raise _tool_exception(
                                sig + _ESCALATION.format(n=n, name=name)) from exc
                        raise
                    # A transport error (unreachable/closed server) on an EXTERNAL bio tool can't be
                    # fixed by the model and aborts the whole run if it propagates — disable that
                    # tool for the rest of the run and return the degraded message NOW, without
                    # waiting for the 3-strike breaker. (Mid-run counterpart to the bio-session
                    # isolation that already handles connect/discovery failures.) A transport error
                    # on a CORE petab tool is NOT auto-disabled here: it falls through to the normal
                    # 3-strike / re-raise path, so a single localhost blip can't strip `fit` or
                    # `make_changes` for the rest of the run.
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
                    escalated = _note_repeated_rejection(state, name, content, artifact)
                    _note_call(state, name, _CALL_SERVER_REJECTED
                               if _rejection_error(content, artifact) is not None else _CALL_OK)
                    return escalated or result
                _note_call(state, name, _CALL_TOOL_ERROR)
                if _record_error(state, name, err) and isinstance(content, list):
                    return (content + _breaker_message(name, hint), artifact)
                return result

            try:  # keep the original signature so langchain still injects `runtime` correctly
                _breaker.__signature__ = inspect.signature(orig)
            except (ValueError, TypeError):  # pragma: no cover - defensive
                pass
            return _breaker

        tool.coroutine = _make(name, orig, hint, containers)
    return tools


_CALL_OK = "ok"
_CALL_SERVER_REJECTED = "server_rejected"   # reached the server, `applied: false`
_CALL_CLIENT_INVALID = "client_invalid"     # never reached it: argument shape failed validation
_CALL_TOOL_ERROR = "tool_error"             # raised anything else
_CALL_DISABLED = "disabled"                 # short-circuited by the breaker


def _note_call(state: dict[str, Any], name: str, outcome: str) -> None:
    state["calls"].setdefault(name, {})
    state["calls"][name][outcome] = state["calls"][name].get(outcome, 0) + 1


def _is_client_validation_error(exc: BaseException) -> bool:
    """True for the pydantic error raised against the PUBLISHED tool schema, client-side.

    langchain titles that model `call[<tool>]`, which is what distinguishes it from a validation
    error raised inside the server. Worth its own bucket: it means the request never left the
    process, so no server-side change can affect it.
    """
    return type(exc).__name__ == "ValidationError" and "validation error" in str(exc)


def call_outcomes(state: dict[str, Any], name: str) -> dict[str, Any]:
    """``{attempted, failed, fail_rate, by_outcome}`` for one tool, for ``result.json``."""
    by = dict(state.get("calls", {}).get(name) or {})
    attempted = sum(by.values())
    failed = attempted - by.get(_CALL_OK, 0)
    return {
        "attempted": attempted,
        "failed": failed,
        "fail_rate": round(failed / attempted, 3) if attempted else None,
        "by_outcome": dict(sorted(by.items(), key=lambda kv: -kv[1])),
    }


def _tool_exception(message: str) -> Exception:
    """A string-carrying tool error langchain's error handling understands.

    NOT ``type(original)(message)``: the exception this path receives from
    langchain-mcp-adapters is ``_MCPToolExecutionError``, whose ``__init__`` takes a list of
    content blocks. Passing it a string produces an object whose message is "string indices must
    be integers" — the escalation AND the original error both lost, so the model would end up
    with strictly less information than before.
    """
    try:
        from langchain_core.tools import ToolException
    except Exception:  # pragma: no cover - langchain_core is a harness dep
        return RuntimeError(message)
    return ToolException(message)


_REJECTION_ESCALATE_AT = 5   # identical operation rejections before the wording escalates

_REJECTION_ESCALATION = (
    "\n\nNOTE: this is the {n}th time this call was rejected for the SAME reason. Re-sending it "
    "with different values will not help — the rejection is about the FORM of the change, not "
    "its numbers. Re-read the error above, and if you cannot satisfy it, change approach: edit a "
    "different part of the model, or accept the current structure and stop."
)


def _rejection_error(content: Any, artifact: Any) -> str | None:
    """The error text of an operation REJECTION (``applied: false``), else None.

    Deliberately distinct from :func:`_result_error`: a rejection is the normal edit/retry
    feedback loop and must never disable a tool. It is tracked only so that a run cannot spend
    its whole budget re-sending one identical rejected edit without ever being told so.
    """
    obj = None
    if isinstance(artifact, dict) and isinstance(artifact.get("structured_content"), dict):
        obj = artifact["structured_content"]
    else:
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
            obj = json.loads(text)
        except Exception:
            return None
    if isinstance(obj, dict) and obj.get("applied") is False:
        return str(obj.get("error") or "rejected")
    return None


def _note_repeated_rejection(state: dict[str, Any], name: str, content: Any, artifact: Any):
    """Append an escalation block once the SAME rejection repeats, else return None.

    The alternative — counting rejections toward the breaker — would disable the editor, which is
    exactly the failure this module now guards against. So the streak only changes the WORDING.
    """
    err = _rejection_error(content, artifact)
    if err is None:
        state["reject_streaks"].pop(name, None)
        return None
    sig = err[:200]
    streak = state["reject_streaks"].get(name)
    if streak and streak["sig"] == sig:
        streak["count"] += 1
    else:
        streak = {"sig": sig, "count": 1}
        state["reject_streaks"][name] = streak
    if streak["count"] < _REJECTION_ESCALATE_AT or not isinstance(content, list):
        return None
    if streak["count"] == _REJECTION_ESCALATE_AT:
        state["n_rejection_escalations"] += 1
        msg = f"`{name}` rejected {streak['count']}x with the same error; escalating: {sig}"
        state["events"].append(msg)
        print(f"[circuit-breaker] {msg}")
    note = [{"type": "text",
             "text": _REJECTION_ESCALATION.format(n=streak["count"])}]
    return (content + note, artifact)


def _container_arg_names(tool: Any) -> dict[str, str]:
    """``{argument: declared container type}`` for this tool, read off its published schema.

    Only these are candidates for JSON-string coercion — never a plain `str` argument, so an
    Antimony model or a justification that happens to start with `[` is left alone. The declared
    type is kept so a parse can be checked against the container the schema actually asked for.
    """
    # `BaseTool.args` is a PROPERTY that indexes `args_schema["properties"]`, so it raises
    # KeyError for a zero-argument MCP tool published as `{"type": "object"}` — and getattr's
    # default only swallows AttributeError. The hosted bio servers' schemas are not ours to fix,
    # and this runs for EVERY tool at wrap time, so an unguarded read would abort run setup on
    # someone else's schema. Any failure here just means "no coercion for this tool".
    try:
        schema = tool.args
    except Exception:
        return {}
    if not isinstance(schema, dict):
        return {}
    names = {}
    for arg, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        branches = spec.get("anyOf") or spec.get("oneOf") or [spec]
        for branch in branches:
            if isinstance(branch, dict) and branch.get("type") in _JSON_CONTAINER_TYPES:
                names[arg] = branch["type"]      # remember WHICH container was declared
                break
    return names


def _coerce_json_string_args(arguments: dict, containers: dict) -> dict:
    """Parse a JSON-encoded STRING passed where the schema declares an array/object.

    Some models emit `parameters="[{...}]"` instead of a real list. That fails validation on the
    CLIENT, against the published tool schema, so the request never reaches the server and no
    server-side coercion can help. The intent is unambiguous and the repair is exact, so accept
    it rather than burn a whole edit cycle on a quoting artefact. Anything that does not parse,
    or parses to the wrong container, is passed through untouched for normal validation to
    reject — this widens what is accepted, it never changes what a valid call means.
    """
    if not containers:
        return arguments
    want = {"array": list, "object": dict}
    out = None
    for arg, declared in containers.items():
        value = arguments.get(arg)
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped[:1] not in ("[", "{"):
            continue
        try:
            parsed = json.loads(stripped)
        except (ValueError, TypeError):
            continue
        # Must match the container the schema DECLARED. An object string where an array is
        # required is not a quoting artefact we can repair — forwarding it would only trade a
        # client-side failure for a server round trip that fails anyway, and add a streak
        # increment on a tool that is never allowed to be disabled.
        if not isinstance(parsed, want.get(declared, (list, dict))):
            continue
        if out is None:
            out = dict(arguments)
        out[arg] = parsed
    return out if out is not None else arguments


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
