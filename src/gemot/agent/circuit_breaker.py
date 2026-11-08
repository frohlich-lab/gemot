"""Cut a tool that keeps failing, or whose server has gone away."""

from __future__ import annotations

import itertools
import json
import re
from contextvars import ContextVar
from typing import Any

from .. import infrastructure, prompts
from ..infrastructure import Evidence, InfrastructureError
from ..prompts import _DISABLED_TOOL_HINT_DEFAULT, _DISABLED_TOOL_HINTS

# --- repeated-tool-error circuit breaker --------------------------------------------------
# A broken tool used to be re-proposed cumulatively hundreds of times across a run (a
# trace-review swarm saw one query_data call re-proposed ~798× — up to ~100 pending copies in
# a single step) while the agent learned nothing from the identical failures. This breaker
# counts every tool failure and, after N CONSECUTIVE IDENTICAL errors from the same tool,
# DISABLES that tool for the caller (see below): its further calls are short-circuited with a
# terminal message pointing at the working alternative, so the agent stops hammering a dead tool
# and moves on. A SUCCESS resets a tool's streak; DIFFERENT errors reset it too (the agent is
# still exploring), so only a genuinely-stuck tool is cut off.
#
# It works at the TOOL-OBJECT layer (wrapping each loaded tool's coroutine), not as agent
# middleware, deliberately: the heavy data/diagnostic tools (query_data, get_fitting_log, …)
# run INSIDE the analyst subagents, and deepagents 0.6.12's declarative subagents don't
# reliably receive a middleware list (see _builtin_tool_filter). Wrapping the tool object makes
# the breaker travel WITH the tool into whichever agent (lead or analyst) calls it.
#
# What it cuts is a CALLER that is stuck, so the streak and the disable belong to the caller: the
# lead, whose scope is the run, or one `task` (one analyst's dispatch, from its first turn to its
# report). They used to belong to the run, and one analyst's three identical SQL errors took
# `query_data` from the lead and every later analyst for the remaining 2.5 days of a 3-day budget
# (gemot#133: lun 103 and froehlich 106 of sweep 63522750, both in the first analyst round). A
# tool that is really dead costs a later task three calls to find out again. A server that is
# gone is the run's, not a caller's, so the transport disable below stays run-wide.
_TOOL_ERROR_DISABLE_THRESHOLD = 3  # consecutive identical errors from one caller before it is cut

#: The `task` whose subagent is making the call, or ``None`` for the lead. Set by the lead's breaker
#: middleware around each `task` call (:func:`_disabled_tool_filter`). A context variable, because
#: the subagent's tool calls run inside that call, and asyncio and langgraph hand each task they
#: start a copy of the context it was started in: parallel tasks each see their own.
_TASK: ContextVar[str | None] = ContextVar("gemot_breaker_task", default=None)

#: What a disable notice says it lasts, per caller.
_FOR_THE_RUN = "the rest of this run"
_FOR_THE_TASK = "the rest of this task; other tasks keep it"

# Tools the breaker may NEVER disable, however often they fail. `make_changes` is the run's only
# committer: disabling it does not degrade the run, it GUARANTEES `no_model_built`, because nothing
# else can author a model or move a bound. Observed in the August sweep — xiaomi/mimo-v2.5 sent
# `parameters` as a JSON *string*, three identical client-side validation errors tripped the
# breaker, and the run then received "TOOL DISABLED ... Do NOT call it again" 98 times and scored
# nothing. The transport path already refuses to strip this tool (see the note in
# `_wrap_tools_with_circuit_breaker`); the 3-strike path simply never got the same guard.
# These tools escalate instead: the error is returned with an added instruction to CHANGE the
# call's shape, which is recoverable, where a disable is not.
#
# The same sole-editor argument covers every rejection `make_changes` can raise, and they are all
# REQUEST-SPECIFIC: three `unknown_reaction_id`/`malformed_reaction`/`duplicate_reaction_id`
# rejections in a row mean three bad edits, not a dead tool. (The draft era reached the same
# conclusion for lint_draft/edit_draft/write_draft, and the substring-patch surface that briefly
# replaced them; both are gone — editing is `make_changes` alone, which was already never-disable.)
#
# `get_model` is here by the VERIFICATION-ONLY rule: a read cannot be "stuck" in the sense this
# breaker models, so the 3-strike heuristic does not apply to it. It is the read of the model as
# WRITTEN, and while `model_edits` addresses reactions by id — so a lost read is less crippling
# than when edits had to quote text — it is still what a `base_version=N` fork is inspected with.
# It is registered without `ctx.run`, so its errors RAISE and would otherwise reach the 3-strike
# path.
_NEVER_DISABLE = frozenset({"make_changes", "get_model"})

_ESCALATION = (
    " — this is the {n}th identical failure. `{name}` is part of the model-editing loop, "
    "so it stays available, but REPEATING THIS CALL WILL FAIL AGAIN. Change the SHAPE of the "
    "arguments, not just their values: check each argument's type against the tool schema "
    "(a list must be a JSON array, not a string containing one), and re-read the error above."
)

# The same escalation for a never-disabled READ. `_ESCALATION` is written for the editor: it says
# the tool "is part of the model-editing loop" and directs the model to change argument SHAPE
# rather than values, which is right for a `parameters: "[{...}]"` type error and exactly WRONG
# for `get_model(version=7)` on a five-version run — there the fix IS a different value, and the
# editor text tells the model not to try it. With the read never disabled either, that could loop
# to the turn limit where the 3-strike disable would at least have stopped it.
_ESCALATION_READ = (
    " — this is the {n}th identical failure. `{name}` is a READ and stays available, but "
    "REPEATING THIS CALL WILL FAIL AGAIN. Re-read the error above and change the ARGUMENTS: an "
    "id or version that does not exist needs a different VALUE (list_versions / "
    "get_model_structure show what exists), and a type error needs a different shape. If the "
    "error is about the model itself rather than your arguments, no argument will fix it — stop "
    "calling this tool and proceed with what you already have."
)

#: Never-disabled tools that are READS, so the escalation names the right remedy.
_NEVER_DISABLE_READS = frozenset({"get_model"})

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
_TRANSPORT_ERROR_NAMES = frozenset(
    {
        "ClosedResourceError",
        "BrokenResourceError",
        "EndOfStream",  # anyio streams
    }
)


def _is_call_timeout(exc: BaseException) -> bool:
    """True for mcp's per-call read timeout — a bare ``MCPError`` carrying ``REQUEST_TIMEOUT``.

    ``runner.clock._wrap_call_timeout`` cuts a call the server has sent no heartbeat on for
    ``CALL_IDLE_WINDOW_S`` and raises ``MCPError(code=REQUEST_TIMEOUT)``, as mcp's own session
    clock does on an expiry (``mcp/shared/jsonrpc_dispatcher.py``), rather than anything in the
    httpx2 family. The code is the SDK's constant, -32001 since mcp 2.0; mcp 1.x sent HTTP's 408,
    and a match on that never fires now (tests/test_judge_call_timeout.py pins the real shape). That distinction matters
    here: a timed-out call is NOT a tool error the model should see and retry. The server is
    still running the abandoned multistart: gemot never cancels it, though mcp 2 would by default
    (``runner.main._ToolCallsRunOn``). A fit tool's timeout is handed to
    the lead as a note that says so, and a further timeout behind it ends the run (see
    :func:`_fit_wedged`). Every other call's is classified as transport, which routes it to the
    checkpoint resume, as the session-wide read timeout did by raising
    ``httpx.ReadTimeout`` before the timeout moved per call.
    """
    if type(exc).__name__ != "MCPError":
        return False
    from mcp.types import REQUEST_TIMEOUT

    return getattr(exc, "code", None) == REQUEST_TIMEOUT


#: The message mcp's dispatcher gives every call once the session's connection has closed
#: (``_CLOSED_OUTCOME`` in ``mcp/shared/jsonrpc_dispatcher.py``, and the same text for a call
#: sent after the close). Its code, ``CONNECTION_CLOSED``, is not enough on its own: the
#: transport also uses it for a single reply it could not deliver while the session lives on.
_SESSION_CLOSED = "Connection closed"


def _is_session_closed(exc: BaseException) -> bool:
    """True for the ``MCPError`` a call raises because its session has closed.

    The calls still waiting when the connection closed get it, and so does every call after
    (tests/test_client_mcp_error.py pins it on a real session). In mcp 2 that is how a server gone
    away reaches a session held open in a task of its own, as the bio sessions are: the transport's
    httpx2 error ends that task, not the call. Nothing the model sends reopens the session, so it
    is a transport error, unlike the transport's other ``CONNECTION_CLOSED`` errors
    (:func:`_is_client_mcp_error`).
    """
    if not _is_client_mcp_error(exc):
        return False
    from mcp.types import CONNECTION_CLOSED

    return exc.code == CONNECTION_CLOSED and exc.message == _SESSION_CLOSED


def _is_client_mcp_error(exc: BaseException) -> bool:
    """True for an ``MCPError`` raised by the call itself, as opposed to an ``is_error`` result.

    The session raises one when the server answers the request with a JSON-RPC error, and when the
    client gives the request up: a call timeout (:func:`_is_call_timeout`), a closed session
    (:func:`_is_session_closed`), or one reply the transport could not read. That last one is what
    ended run f2e2b86d after 15.8 h (gemot#103): ``SSE stream failed: Server-sent event exceeded
    the 1048576 byte limit``, for one ``get_version`` reply, on a session that went on working.
    """
    try:
        from mcp import MCPError
    except ImportError:  # pragma: no cover - mcp always present with the agent stack
        return False
    return isinstance(exc, MCPError)


#: How the server says a fit is still running when another call arrives behind it
#: (PEtab-MCP#108). Read only while a fit's call timeout is outstanding: before one, it can only
#: answer a fit batched beside another, and goes to the lead like any other reply.
_FIT_IN_PROGRESS = re.compile(r"\bfit_in_progress\b")


def _fit_wedged(state: dict[str, Any], text: str, timed_out: bool) -> Evidence | None:
    """The evidence that the server is wedged on a fit, or ``None``.

    After a fit's call times out, the server keeps that fit, and the session lock it holds, until
    the fit ends. A resume used to re-send the identical fit into it, and each resend waited out
    the full timeout again: 8.4 h of one INDRA run's last 9 h, against a fit that had hung on the
    server (gemot#56, PEtab-MCP#108). The wrapper hands the lead that first timeout instead, as
    its tool result. Once one is outstanding, a further call that times out (``timed_out``), or
    the server answering that the fit is still running, means the server is holding it still:
    the run stops with ``infrastructure_error: fit_wedged``. A fit call that returns clears it.
    """
    held = state["fit_timeout"]
    if held is None or not (timed_out or _FIT_IN_PROGRESS.search(text)):
        return None
    return Evidence(infrastructure.FIT_WEDGED, " ".join(f"{text[:300]}; behind {held}".split()))


def _is_transport_error(exc: BaseException) -> bool:
    """True for a transport/connection failure (an unreachable/closed MCP server, or a call that
    out-idled its own timeout) as opposed to a normal tool error the model should see and retry.
    Covers the anyio stream errors (by name), the builtin ConnectionError, the WHOLE
    httpx2.TransportError family (ConnectError/ReadError/WriteError/ReadTimeout/ConnectTimeout/
    RemoteProtocolError/PoolTimeout/WriteTimeout/… ), and mcp's own timeout (see _is_call_timeout)
    — the same surface the run-loop resume backstop matches, so the two never disagree on what is
    'transport'. httpx2, not httpx: mcp 2 runs its transport on it, and httpx2's errors do not
    derive from httpx's.

    Also a closed session (:func:`_is_session_closed`), which the backstop does not resume: the
    run ends on it, and is scored as an infrastructure stop."""
    if type(exc).__name__ in _TRANSPORT_ERROR_NAMES or isinstance(exc, ConnectionError):
        return True
    if _is_call_timeout(exc) or _is_session_closed(exc):
        return True
    try:
        import httpx2
    except Exception:  # pragma: no cover - httpx2 always present with mcp
        return False
    return isinstance(exc, httpx2.TransportError)


def _new_breaker_state() -> dict[str, Any]:
    """Mutable, run-scoped state shared by the wrapped tools and the lead's tool filter."""
    return {
        # The LEAD's scope: the tools it has had disabled, and its error streaks. Each `task`
        # has the same two under `tasks` (see `_scope`).
        "disabled": set(),
        "streaks": {},
        "tasks": {},
        # Tools whose server went away (a transport error on a bio tool): cut for every caller.
        "unreachable": set(),
        "n_tool_errors": 0,
        "events": [],
        # Separate from `streaks`: an operation REJECTION is not a tool error (see
        # `_result_error`) and must never disable anything, but a run can still spend its
        # whole budget re-sending one -- gemini 106 identical rejections, luna 246 of 265.
        "reject_streaks": {},
        "n_rejection_escalations": 0,
        # Per-tool CALL outcomes, counted here because this wrapper is the only place that
        # sees every attempt. `_ToolClock.note_result` (runner/clock.py) tallies only calls
        # that reach `session.call_tool` AND return a parseable payload, so it is blind to the
        # three ways a call can fail without ever touching the server -- which is how a
        # mimo-v2.5 run that made 134 failing make_changes calls wrote
        # `edit_rejects: {attempted: 0, rejected: 0, reject_rate: null}`.
        "calls": {},
        # Core-tool calls whose result or error carried an infrastructure failure (a full disk),
        # and the first one's `infrastructure.Evidence`. Once there is one, every later call is
        # refused before it reaches the server (gemot#39).
        "infrastructure": {"tool_calls": 0, "evidence": None},
        # The timed-out fit call the server may still be running, until a fit is answered: what
        # tells a wedged server from a slow fit (`_fit_wedged`, gemot#56).
        "fit_timeout": None,
    }


def _scope(state: dict[str, Any]) -> dict[str, Any]:
    """The calling scope's ``{"disabled", "streaks"}``: the lead's (the run's state itself), or
    the current `task`'s."""
    task = _TASK.get()
    if task is None:
        return state
    return state["tasks"].setdefault(task, {"disabled": set(), "streaks": {}})


def disabled_tools(state: dict[str, Any]) -> list[str]:
    """The tools the run lost: the lead's disables and the tools of a server that went away."""
    return sorted(state["disabled"] | state["unreachable"])


def task_disabled_tools(state: dict[str, Any]) -> dict[str, int]:
    """``{tool: how many tasks had it disabled}``. Each task lost it only for itself."""
    counts: dict[str, int] = {}
    for scope in state["tasks"].values():
        for name in scope["disabled"]:
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _breaker_message(name: str, hint: str) -> list[dict[str, str]]:
    """The terminal content block returned in place of a disabled tool's result."""
    text = prompts.TOOL_DISABLED_REPEATED_TMPL.format(
        name=name, hint=hint, scope=_FOR_THE_RUN if _TASK.get() is None else _FOR_THE_TASK
    )
    return [{"type": "text", "text": text}]


def _transport_breaker_message(name: str, hint: str) -> list[dict[str, str]]:
    """Degraded block for a tool disabled after a transport failure (an unreachable/closed server).

    Distinct from :func:`_breaker_message`: a transport error disables on the FIRST failure, so the
    "failed repeatedly with the same error" wording would misdescribe it."""
    return [
        {"type": "text", "text": prompts.TOOL_DISABLED_TRANSPORT_TMPL.format(name=name, hint=hint)}
    ]


def _result_error(content: Any, artifact: Any) -> str | None:
    """Return the error string if a (content, artifact) tool result is a FAILURE, else None.

    The petab-mcp server returns a failed read/diagnostic tool as a normal result carrying
    ``{"ok": false, "error": ...}`` (not an MCP protocol error), which the tool bridge
    (``agent.mcp_tools``) exposes in the artifact's ``structured_content`` and in the text content. A rejected
    *operation* (``{"committed": false}`` — e.g. a lint failure the agent should fix and retry)
    is deliberately NOT treated as a breaker error: that is the normal edit/retry feedback loop.

    ``ok: false`` alone is NOT a failure either — it must carry an ``error`` key. A VERDICT
    payload (``{"ok": false, "issues": [...]}`` with no error — the draft-era lint shape) is a
    tool doing its job: treating it as an error collapsed every fix-me verdict to the constant
    signature "tool error", so the documented different-errors-reset-the-streak protection could
    never fire, and three consecutive verdicts — three DIFFERENT issues, in the observed trips —
    disabled the tool (7 of 18 draft-arm runs, sneyd/fujita A/B). No current tool emits the
    shape, but the carve-out stays: it is the correct reading of any future verdict-style payload.
    """
    if isinstance(artifact, dict):
        sc = artifact.get("structured_content")
        if isinstance(sc, dict) and sc.get("ok") is False:
            return _verdict_or_error(sc)
        if artifact.get("isError") or artifact.get("is_error"):
            return "tool error"
    # Fallback: look for the {"ok": false} shape in the text content blocks.
    try:
        obj = json.loads(_content_text(content))
    except Exception:
        return None
    if isinstance(obj, dict) and obj.get("ok") is False:
        return _verdict_or_error(obj)
    return None


def _content_text(content: Any) -> str:
    """A tool result's text content blocks, concatenated."""
    if isinstance(content, str):
        return content
    text = ""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += str(block.get("text") or "")
            elif isinstance(block, str):
                text += block
    return text


def _infrastructure_error(state: dict[str, Any], name: str, hit: Evidence) -> InfrastructureError:
    """Record a call that met an infrastructure failure; return the error that ends the run in
    its place.

    Raised, not returned. The model cannot free disk space, and handed the error it does what the
    INDRA sweep's leads did with the same failures disguised as "ill-posed": re-author models that
    were fine (gemot#39).
    """
    record = state["infrastructure"]
    record["tool_calls"] += 1
    evidence = Evidence(hit.cause, f"{name}: {hit.text}")
    record["evidence"] = record["evidence"] or evidence
    _note_call(state, name, _CALL_INFRASTRUCTURE)
    print(
        f"[circuit-breaker] infrastructure failure ({hit.cause}); stopping the run: {evidence.text}"
    )
    return InfrastructureError(evidence)


def _verdict_or_error(payload: dict) -> str | None:
    """For an ``ok: false`` payload: None if it is a VERDICT, else its error string.

    A verdict is identified POSITIVELY, by carrying ``issues`` without an ``error``.
    Keying on "no ``error`` key" instead would exempt two shapes that are real failures: an
    empty ``error`` (which then signs the streak with "", handing the model an escalation with
    no error text), and a bare ``{"ok": false}`` — the shape a HOSTED bio server may use, and
    those are the flaky ones the breaker exists for.
    """
    err = payload.get("error")
    if err:
        return str(err)
    if "issues" in payload:
        return None
    return "tool error"


def _record_error(state: dict[str, Any], name: str, err: str) -> bool:
    """Count a tool error; return True iff it TRIPS the breaker (disables the tool now, for the
    calling scope)."""
    state["n_tool_errors"] += 1
    scope = _scope(state)
    sig = err[:200]
    streak = scope["streaks"].get(name)
    if streak and streak["sig"] == sig:
        streak["count"] += 1
    else:
        streak = {"sig": sig, "count": 1}
        scope["streaks"][name] = streak
    if name in _NEVER_DISABLE:
        # Not disabled — but still RECORDED. Returning silently here would leave a run that
        # failed this call 200 times reporting `disabled_tools: []`, no breaker line in the log
        # and nothing in `events`: a run that scored nothing with no recorded reason, which is
        # the exact invisibility the August sweep had to be reverse-engineered from.
        if streak["count"] == _TOOL_ERROR_DISABLE_THRESHOLD:
            msg = (
                f"`{name}` has failed {streak['count']} times with the same error but is part "
                f"of the model-editing loop, so it is NOT disabled; escalating instead: {sig}"
            )
            state["events"].append(msg)
            print(f"[circuit-breaker] {msg}")
        return False
    if streak["count"] >= _TOOL_ERROR_DISABLE_THRESHOLD and name not in scope["disabled"]:
        scope["disabled"].add(name)
        msg = (
            f"disabled `{name}` for {_TASK.get() or 'the lead'} after {streak['count']} "
            f"consecutive identical errors: {sig}"
        )
        state["events"].append(msg)
        print(f"[circuit-breaker] {msg}")
        return True
    return False


def _wrap_tools_with_circuit_breaker(
    tools: list,
    state: dict[str, Any],
    transport_disable_names: set[str] | None = None,
    fit_tools: frozenset[str] = frozenset(),
) -> list:
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

    Every OTHER tool's result and error is also read for an infrastructure failure
    (:func:`gemot.infrastructure.find`: the server's ``infrastructure_error`` report, or a full
    disk's raw text), and one that carries it raises
    :class:`~gemot.infrastructure.InfrastructureError` instead of returning, which ends the run.
    The external tools are exempt: a hosted server's disk is not this run's to stop on.

    ``fit_tools`` are the tools whose call runs a multistart. The first of their calls to time out
    returns a note saying so, and a server still holding that fit raises the same error
    (:func:`_fit_wedged`).
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
            ours = name not in transport_disable  # a tool of the run's own server

            @wraps(orig)
            async def _breaker(**arguments):
                if state["infrastructure"]["tool_calls"]:
                    # Another call already met the failure and the run is stopping; a call still
                    # in flight beside it must not reach the server either.
                    _note_call(state, name, _CALL_DISABLED)
                    raise InfrastructureError(state["infrastructure"]["evidence"])
                if name in state["unreachable"]:
                    _note_call(state, name, _CALL_DISABLED)
                    return (_transport_breaker_message(name, hint), None)
                if name in _scope(state)["disabled"]:
                    _note_call(state, name, _CALL_DISABLED)
                    return (_breaker_message(name, hint), None)  # short-circuit, no server hit
                arguments = _coerce_json_string_args(arguments, containers)
                try:
                    result = await orig(**arguments)
                except Exception as exc:  # a raised tool error (e.g. MCP isError -> ToolException)
                    # The server's own report arrives here: an `isError` result, which the
                    # adapter raises and its `handle_tool_error` would hand the model as a result.
                    if ours and (hit := infrastructure.find_in_error(exc)) is not None:
                        raise _infrastructure_error(state, name, hit) from exc
                    sig = f"{type(exc).__name__}: {exc}"
                    timed_out = _is_call_timeout(exc)
                    if ours and (hit := _fit_wedged(state, sig, timed_out)) is not None:
                        raise _infrastructure_error(state, name, hit) from exc
                    _note_call(
                        state,
                        name,
                        _CALL_CLIENT_INVALID
                        if _is_client_validation_error(exc)
                        else _CALL_TOOL_ERROR,
                    )
                    tripped = _record_error(state, name, sig)
                    if ours and timed_out and name in fit_tools:
                        # The lead hears of it rather than a resume re-sending the same fit into
                        # the server still running it (_fit_wedged).
                        state["fit_timeout"] = f"{name}: {sig}"
                        return (prompts.FIT_CALL_TIMEOUT_TMPL.format(name=name, error=sig), None)
                    # A TRANSPORT failure is exempt from the never-disable return below. The model
                    # cannot fix a closed session or an unreachable server by changing its
                    # arguments, so handing it "ClosedResourceError: ..." as a successful tool
                    # result on every subsequent call just burns the run to the recursion or token
                    # limit — and it is then recorded as a genuine modelling failure with 0 fits.
                    # Letting it propagate is what reaches the driver's `_TRANSIENT_ERRORS` resume
                    # path, and failing that, what puts the error class in `stop_reason` where
                    # `experiments._is_infra_abort` / `is_noop_infra_abort` can exclude it from the
                    # success/dBIC aggregation. Swallowing it silently reclassified infrastructure
                    # noise as model incapability, which biases every model comparison.
                    if name in _NEVER_DISABLE and not _is_transport_error(exc):
                        # Never stripped, so the model keeps the tool — but a bare repeat of the
                        # same error teaches it nothing. Escalate the wording once the streak is
                        # real, and RETURN it as the tool's result rather than raising.
                        #
                        # Raising here used to END THE RUN. The tool bridge (`langchain.mcp`, as
                        # langchain-mcp-adapters before it) installs `_handle_mcp_tool_error` as
                        # the tools' `handle_tool_error`, and it converts the error into a
                        # ToolMessage for exactly ONE type:
                        #
                        #     if isinstance(error, _MCPToolExecutionError): return ...content
                        #     raise error          # <- everything else propagates
                        #
                        # so a plain ToolException carrying the escalation took the `raise error`
                        # branch and aborted the whole run — as did any client-side pydantic
                        # validation error, which is not a ToolException subclass and so never
                        # reaches that callback at all. Observed: 3 of minimax-m3's 5 sneyd runs
                        # died this way with 0 fits.
                        #
                        # Returning (content, artifact) is the shape these tools already use
                        # (`response_format="content_and_artifact"`) and the same one the breaker
                        # and transport paths return, so the model reliably SEES the error and can
                        # correct the call — which is the entire point of never disabling this tool.
                        n = _scope(state)["streaks"].get(name, {}).get("count", 1)
                        esc = _ESCALATION_READ if name in _NEVER_DISABLE_READS else _ESCALATION
                        detail = sig + (
                            esc.format(n=n, name=name) if n >= _TOOL_ERROR_DISABLE_THRESHOLD else ""
                        )
                        return (detail, None)
                    # A transport error (unreachable/closed server) on an EXTERNAL bio tool can't be
                    # fixed by the model and aborts the whole run if it propagates — disable that
                    # tool for the rest of the run and return the degraded message NOW, without
                    # waiting for the 3-strike breaker. (Mid-run counterpart to the bio-session
                    # isolation that already handles connect/discovery failures.) A transport error
                    # on a CORE petab tool is NOT auto-disabled here: it falls through to the normal
                    # 3-strike / re-raise path, so a single localhost blip can't strip `fit` or
                    # `make_changes` for the rest of the run.
                    if name in transport_disable and _is_transport_error(exc):
                        state["unreachable"].add(name)
                        return (_transport_breaker_message(name, hint), None)
                    if tripped:
                        return (_breaker_message(name, hint), None)
                    if _is_client_mcp_error(exc) and not _is_transport_error(exc):
                        # The same `raise error` as above: an MCPError is no ToolException, so it
                        # ended the run (gemot#103: f2e2b86d, 15.8 h into a 33 h budget, on one
                        # reply too large for the client to read). The session is still up, so
                        # the model gets the error as the result and can ask for less; the
                        # 3-strike count above still applies.
                        return (sig, None)
                    raise
                content, artifact = result if isinstance(result, tuple) else (result, None)
                # The whole text, not only an error's: before PEtab-MCP#95 a fit that crashed on
                # the errno came back as a REJECTION ("the edit applied but the fit crashed
                # unexpectedly: [Errno 122] ..."), which `_result_error` rightly does not call an
                # error. A build that failed on it said only "ill-posed"; the server's logs carry
                # that one (LogWatch).
                text = _content_text(content)
                if ours and (hit := infrastructure.find(text)) is not None:
                    raise _infrastructure_error(state, name, hit)
                if ours and (hit := _fit_wedged(state, text, False)) is not None:
                    raise _infrastructure_error(state, name, hit)
                if name in fit_tools:
                    state["fit_timeout"] = None  # answered, so the timed-out fit has ended
                err = _result_error(content, artifact)
                if err is None:
                    _scope(state)["streaks"].pop(name, None)  # a success clears the streak
                    escalated = _note_repeated_rejection(state, name, content, artifact)
                    _note_call(
                        state,
                        name,
                        _CALL_SERVER_REJECTED
                        if _rejection_error(content, artifact) is not None
                        else _CALL_OK,
                    )
                    return escalated or result
                _note_call(state, name, _CALL_TOOL_ERROR)
                if _record_error(state, name, err) and isinstance(content, list):
                    return (content + _breaker_message(name, hint), artifact)
                if name in _NEVER_DISABLE:
                    # These tools are never cut, so a repeated failure has NO other feedback
                    # path: the escalation below is applied in the `except` branch, but these
                    # tools report their errors as RESULTS (`ctx.run` converts OperationError to
                    # `{"ok": false, "error": ...}`), which never raises. Without this, an editor
                    # stuck on one bad `old_string` loops on byte-identical feedback until the
                    # turn limit — no disable, no escalation, only a stdout line it cannot see.
                    n = _scope(state)["streaks"].get(name, {}).get("count", 1)
                    if n >= _TOOL_ERROR_DISABLE_THRESHOLD and isinstance(content, list):
                        note = [
                            {
                                "type": "text",
                                "text": (
                                    _ESCALATION_READ
                                    if name in _NEVER_DISABLE_READS
                                    else _ESCALATION
                                ).format(n=n, name=name),
                            }
                        ]
                        return (content + note, artifact)
                return result

            try:  # keep the original signature: StructuredTool reads it for `callbacks` and `config`
                _breaker.__signature__ = inspect.signature(orig)
            except (ValueError, TypeError):  # pragma: no cover - defensive
                pass
            return _breaker

        tool.coroutine = _make(name, orig, hint, containers)
    return tools


_CALL_OK = "ok"
_CALL_SERVER_REJECTED = "server_rejected"  # reached the server, `applied: false`
_CALL_CLIENT_INVALID = "client_invalid"  # never reached it: argument shape failed validation
_CALL_TOOL_ERROR = "tool_error"  # raised anything else
_CALL_DISABLED = "disabled"  # short-circuited by the breaker
_CALL_INFRASTRUCTURE = "infrastructure_error"  # e.g. a full disk; the run stops at this call


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


# NOTE: there is deliberately no `_tool_exception` helper any more. It existed to wrap the
# escalation in a plain ToolException, which is precisely what killed runs -- langchain-mcp-adapters'
# `_handle_mcp_tool_error` re-raises every ToolException that is not an `_MCPToolExecutionError`.
# Errors on a never-disabled tool are RETURNED as tool content instead (see `_make` above); do not
# reintroduce a raise here without checking that callback first.


_REJECTION_ESCALATE_AT = 5  # identical operation rejections before the wording escalates

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
        try:
            obj = json.loads(_content_text(content))
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
    note = [{"type": "text", "text": _REJECTION_ESCALATION.format(n=streak["count"])}]
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
                names[arg] = branch["type"]  # remember WHICH container was declared
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
    """Lead-only middleware: hide a disabled tool from the model so it stops being re-proposed,
    and give each `task` call its own breaker scope.

    Complements the coroutine wrapper (the wrapper is what actually reaches the analyst
    subagents): for the LEAD's own tools this also drops the tool's schema from the re-sent
    request once it is disabled, saving tokens and removing the temptation to call it again.

    Around a `task` call it sets :data:`_TASK`, so the subagent's tool calls count their errors,
    and lose a tool, only within that task. The label is the analyst and the task's number in the
    run, not the tool call id: some providers repeat ids across turns.
    """
    from langchain.agents.middleware.types import AgentMiddleware

    numbers = itertools.count(1)

    def _name(tool) -> str | None:
        name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        return name if isinstance(name, str) else None

    def _keep(request):
        gone = state["disabled"] | state["unreachable"]
        return [t for t in request.tools if _name(t) not in gone]

    def _task(request) -> str | None:
        """This call's scope label if it is a `task` call, else None."""
        tc = getattr(request, "tool_call", None) or {}
        if tc.get("name") != "task":
            return None
        analyst = (tc.get("args") or {}).get("subagent_type") or "subagent"
        return f"{analyst} task {next(numbers)}"

    class _DropDisabledTools(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            return handler(request.override(tools=_keep(request)))

        async def awrap_model_call(self, request, handler):
            return await handler(request.override(tools=_keep(request)))

        def wrap_tool_call(self, request, handler):
            if (task := _task(request)) is None:
                return handler(request)
            token = _TASK.set(task)
            try:
                return handler(request)
            finally:
                _TASK.reset(token)

        async def awrap_tool_call(self, request, handler):
            if (task := _task(request)) is None:
                return await handler(request)
            token = _TASK.set(task)
            try:
                return await handler(request)
            finally:
                _TASK.reset(token)

    return _DropDisabledTools()
