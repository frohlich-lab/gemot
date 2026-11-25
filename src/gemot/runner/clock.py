"""The MCP call handle the pipeline uses, and the two shadows wrapped around ``call_tool``.

``_ToolClock`` measures the wall-clock the run actually spent INSIDE tool calls. It tracks the
UNION of busy intervals rather than their sum, so concurrent analyst calls are not double-counted
and the reported compute time stays comparable across runs with different panel concurrency.

``_wrap_call_timeout`` cuts a call the server has gone silent on: every call lives on the server's
heartbeats (PEtab-MCP#224) rather than on a budget sized from its fit, which a fit queued for a
shared pool outlasts (gemot#106). What the heartbeats say across the run, :class:`_Heartbeats`,
feeds the driver's stall watchdog and takes the run's queue time out of its wall budget.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from collections.abc import Callable, Iterator

import anyio
from mcp import ClientSession, MCPError
from mcp.types import REQUEST_TIMEOUT

from .. import infrastructure
from ..causal_constraints import CONSTRAINT_VIOLATED, broken_constraints
from ..fit_budget import FIT_COMPILE_MARGIN_S


# --------------------------------------------------------------------------- MCP glue
def _unwrap(result):
    """Turn a CallToolResult into the tool's Python return value."""
    sc = getattr(result, "structured_content", None)
    if sc is not None:
        # FastMCP wraps non-dict returns as {"result": <value>}.
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    texts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except Exception:
        return joined


#: How long a call may go without a live heartbeat before :func:`_wrap_call_timeout` cuts it. The
#: server sends one every 30 s while a call runs (PEtab-MCP#224's ``HEARTBEAT_S``), so this is ten
#: missed in a row: far past the 3 s a GIL-holding AMICI build silenced the server's event loop in
#: PEtab-MCP#224, and far inside the stall watchdog's floor (``agent.driver._STEP_STALL_TIMEOUT_S``),
#: so a silent server's fit call returns its timeout to the lead before the watchdog cancels the step.
CALL_IDLE_WINDOW_S = 300.0

#: The judge's held-out score (:mod:`petab_mcp.validation_data`).
HELD_OUT_TOOL = "evaluate_validation_data"


def _make_call(session: ClientSession):
    """Return an ``await call(tool, **args)`` bound to ``session`` (unwraps the result).

    Carries no timeout of its own: every call is cut by `_wrap_call_timeout`, on the server's
    heartbeats, whoever makes it. Deliberately not an explicit ``read_timeout_seconds`` either:
    mcp's clock would then cut a read waiting behind a fit the heartbeats keep alive, as the flat
    per-handle bounds tried before did to the finish gate's reads (see ``run_deepagents``'s
    ``progress_probe``).

    An error result that reports an infrastructure failure (PEtab-MCP#95, or a raw full disk)
    raises :class:`~gemot.infrastructure.InfrastructureError` instead of being returned: unwrapped,
    it is a string where the judge expects a dict, and the run would end as a crash or a verdict
    on a broken server rather than as the infrastructure stop the sweep resubmits (gemot#39).
    """

    async def call(tool: str, /, **args):
        result = await session.call_tool(tool, args or {})
        value = _unwrap(result)
        if (
            getattr(result, "is_error", False)
            and isinstance(value, str)
            and (hit := infrastructure.find(value)) is not None
        ):
            raise infrastructure.InfrastructureError(
                infrastructure.Evidence(hit.cause, f"{tool}: {hit.text}")
            )
        return value

    return call


#: Tools whose result carries an ``applied`` flag, i.e. an edit the server can refuse.
#: ``set_model``/``change_parameters`` are the pre-#130 names, kept so old traces and any
#: still-pinned server version are counted the same way.
_EDIT_TOOLS = frozenset({"make_changes", "set_model", "change_parameters"})

#: Rejection buckets, matched in order. Deliberately coarse: the point is to show WHICH KIND of
#: refusal dominates a run at a glance, so a regression (or a skill gap) is visible from
#: result.json without mining the trace DB. `antimony_syntax` and `invalid_problem` are the two
#: that mattered on the sweep this was built from — 37% and 59% of set_model refusals.
_REJECT_PATTERNS = (
    ("antimony_syntax", "failed to parse antimony"),
    ("invalid_problem", "resulting problem is invalid"),
    ("repeated_widening", "refusing to widen"),
    # A one-sided bound edit keeps the other side (PEtab-MCP#293) and is refused only when the
    # kept side would invert the box.
    ("bad_bounds_usage", "the box would be"),
    ("unknown_key", "unknown key"),
    ("unknown_parameter", "unknown parameter id"),
    ("bad_scale", "invalid parameterscale"),
    ("bad_estimate", "cannot interpret estimate value"),
    ("missing_id", "must include an 'id'"),
    # NOTE there are deliberately no needles for the structured-op rejections. Those carry a
    # machine-readable `error_cause` (petab_mcp.operations.OperationError.cause), which
    # `note_result` prefers, so prose entries for them would be dead code — and a generic one
    # like "already exists" would additionally steal any FUTURE rejection containing the phrase,
    # which is the needle-shadows-the-bucket failure the labels were introduced to end.
    ("op_malformed", "nothing was applied"),  # residual, for anything unlabelled
)


def _result_payload(result: object) -> "dict | None":
    """The tool's JSON body out of an MCP result envelope, or ``None`` if it is not one."""
    try:
        content = getattr(result, "content", None)
        if content is None and isinstance(result, dict):
            content = (result.get("data") or {}).get("content")
        text = getattr(content[0], "text", None) if content else None
        if text is None and content:
            text = content[0].get("text")
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except Exception:  # noqa: BLE001 - a metric must never break a run
        return None


def _reject_cause(error: str) -> str:
    low = error.lower()
    for label, needle in _REJECT_PATTERNS:
        if needle in low:
            return label
    return "other"


class _ToolClock:
    """Client-observed wall-time during which at least one MCP tool call is in flight.

    This is time *as the runner sees it* — it includes the HTTP round-trip and any
    server-side queueing, not purely server execution — but for these demos it is
    dominated by the work inside the tool calls (AMICI compile + pyPESTO fits + SQL),
    so subtracting it from the wall clock is a good proxy for isolating LLM time.


    Tracks the UNION of busy intervals, not the sum of per-call durations: the LangGraph
    agent (deepagents' ToolNode) dispatches the tool calls in a single turn concurrently, so
    several ``session.call_tool`` coroutines overlap. Summing each one's duration double-counts the
    overlap and can exceed the wall clock (a fit + a SQL query running at once counts twice).
    A busy-interval measure instead answers "how much wall time was the server working," which
    is what we subtract from the wall clock to isolate LLM/inference time, and it can never
    exceed the wall clock.

    Also tallies REJECTED edits (see :meth:`note_result`), which is a different question from
    ``n_tool_errors``: a rejected edit is a SUCCESSFUL tool call returning ``applied: false``, so
    nothing upstream counts it. A Phoenix scan of one sweep found 518/2691 ``set_model`` calls
    (19%) and 111/1475 ``change_parameters`` calls (7.5%) rejected, dominated by causes the
    skills never mention — invisible in every result.json, and only discoverable by mining the
    36 GB trace DB after the fact. Carrying the rate per run makes it visible during a sweep.
    """

    __slots__ = (
        "seconds",
        "calls",
        "_active",
        "_since",
        "edits",
        "rejects",
        "fit_reverts",
        "reject_causes",
        "by_constraint",
        "first_commit_edit",
        "by_tool",
    )

    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0
        self._active = 0  # in-flight call count
        self._since = 0.0  # monotonic time the busy window (0 -> 1) opened
        self.edits = 0
        self.rejects = 0
        self.fit_reverts = 0
        self.reject_causes: dict[str, int] = {}
        # Refusals per causal constraint, on the causal_constraints arm: which of the user's claims
        # the agent kept breaking. A refusal naming two counts for each.
        self.by_constraint: dict[str, int] = {}
        # Which editing call committed the run's first version (1-based), or None: how many calls
        # it took the agent to get a model past the server's checks.
        self.first_commit_edit: int | None = None
        # Per-tool CALL COUNTS. `calls` alone is a bare total, so "was this tool used at all?" was
        # answerable only from the Phoenix traces -- which need a shared collector running, and
        # otherwise land in `.slurm_scratch`, which is deleted when a task completes. A sweep run
        # without the collector is therefore permanently unable to answer the question, and one was:
        # the whole point of job 51614722 was whether the agent called `get_data_regimes`, and the
        # answer is unrecoverable. The name is already in hand at the single chokepoint every call
        # routes through; keeping it costs one dict and makes tool usage part of result.json, which
        # survives.
        self.by_tool: dict[str, int] = {}

    def enter(self, tool_name: "str | None" = None) -> None:
        if self._active == 0:
            self._since = time.monotonic()
        self._active += 1
        self.calls += 1
        if tool_name:
            self.by_tool[tool_name] = self.by_tool.get(tool_name, 0) + 1

    def leave(self) -> None:
        self._active -= 1
        if self._active == 0:
            self.seconds += time.monotonic() - self._since

    def note_result(self, tool_name: str, result: object) -> None:
        """Tally an EDITING tool call and, if the edit was refused, bucket why.

        Best-effort by construction: the result shape is the MCP envelope, and a change to it must
        degrade this metric, never break a run. Unparseable results simply are not counted.
        """
        if tool_name not in _EDIT_TOOLS:
            return
        payload = _result_payload(result)
        if payload is None:
            return
        self.edits += 1
        if self.first_commit_edit is None and payload.get("committed") is True:
            self.first_commit_edit = self.edits
        # `applied is False` ALONE is the edit verdict. The payload flattens the FIT's keys in
        # too, so an `error` can ride on a change that applied and was even snapshotted — testing
        # it counted those as refusals and polluted `by_cause` with fit failures, in the one
        # metric whose job is telling the two apart.
        if payload.get("applied", payload.get("committed")) is not False:
            return
        # Since #130 one call edits AND fits, so `applied: false` covers two different events. A
        # pure edit refusal is `{applied, error[, issues]}` straight from the tool's error wrapper;
        # once the fit stage has run, `bundle_result` always sets `committed`. Counting the fit
        # reverts as refusals made the rate an edit-or-fit failure rate and broke comparability
        # with the pre-#130 baseline — so they are tallied separately instead of discarded.
        if "committed" in payload:
            self.fit_reverts += 1
            return
        self.rejects += 1
        # The server's own label when it set one (petab_mcp.operations.OperationError.cause);
        # the prose match is the fallback for rejections raised deeper in the pipeline, which
        # carry no label.
        cause = payload.get("error_cause") or _reject_cause(str(payload.get("error") or ""))
        self.reject_causes[cause] = self.reject_causes.get(cause, 0) + 1
        if cause == CONSTRAINT_VIOLATED:
            for cid in broken_constraints(str(payload.get("error") or "")):
                self.by_constraint[cid] = self.by_constraint.get(cid, 0) + 1

    def reject_summary(self) -> dict:
        """``{attempted, rejected, reject_rate, by_cause}`` for ``result.json``."""
        return {
            "attempted": self.edits,
            "rejected": self.rejects,
            "fit_reverted": self.fit_reverts,
            "reject_rate": round(self.rejects / self.edits, 3) if self.edits else None,
            "by_cause": dict(sorted(self.reject_causes.items(), key=lambda kv: -kv[1])),
        }


#: A heartbeat's ``message`` (PEtab-MCP#224, ``server._Heartbeat``): the session's running change
#: as ``"<phase> for <s>s, <done> of <n> starts done"``, ``<s>`` its seconds in that phase and the
#: starts left out before its fit has begun. With no change running there is no message.
_HEARTBEAT = re.compile(r"(?P<phase>\w+) for (?P<age>\d+)s")


class _Heartbeats:
    """What the server's heartbeats have said, across every call of the run.

    A heartbeat is live unless it shows a change ``compiling`` for longer than
    ``fit_budget.FIT_COMPILE_MARGIN_S``: the server keeps sending them while a build hangs, so a
    hung compile would otherwise hold its call for good. Each live one pushes out its own call's
    cut (:func:`_wrap_call_timeout`) and calls every listener (:meth:`listen`), which is how the
    driver's stall watchdog is fed.

    ``queue_seconds`` is the run's time with a change ``queued`` for the shared fit pool, which the
    driver leaves out of the wall budget. A ``queued`` heartbeat adds the time since the previous
    heartbeat on any call, but no more than its change has been queued: calls beside each other (a
    read waiting behind a fit gets the fit's heartbeats) are counted once, and the count begins
    where the wait did. It ends at the wait's last heartbeat, so it falls short of the server's own
    ``queue_seconds`` (``get_run_timing``) by less than one heartbeat per wait.
    """

    def __init__(self) -> None:
        self.queue_seconds = 0.0
        self._last: float | None = None
        self._listeners: list[Callable[[], None]] = []

    def hear(self, message: str | None) -> bool:
        """Note one heartbeat whose text is ``message``; whether it is live."""
        now = time.monotonic()
        match = _HEARTBEAT.match(message or "")
        phase, age = (match["phase"], float(match["age"])) if match else (None, 0.0)
        if phase == "queued":
            self.queue_seconds += age if self._last is None else min(age, now - self._last)
        self._last = now
        if phase == "compiling" and age > FIT_COMPILE_MARGIN_S:
            return False
        for listener in tuple(self._listeners):
            with contextlib.suppress(Exception):  # a listener never costs a call its heartbeat
                listener()
        return True

    @contextlib.contextmanager
    def listen(self, listener: Callable[[], None]) -> Iterator[None]:
        """Call ``listener`` on every live heartbeat while the block is open."""
        self._listeners.append(listener)
        try:
            yield
        finally:
            self._listeners.remove(listener)


def _wrap_call_timeout(session: ClientSession) -> _Heartbeats:
    """Cut a ``session.call_tool`` once the server has gone :data:`CALL_IDLE_WINDOW_S` without a
    live heartbeat on it, and return the run's :class:`_Heartbeats`.

    A call has no budget of its own. On a shared fit pool a fit's wall time includes its wait
    behind other runs' fits, which no formula bounds, so the waves-times-cap budget this replaced
    (``fit_budget.fit_time_budget_for``) would cut a fit that is only queued, as would the budget a
    read got for the fits ahead of it (gemot#106). Each call runs in an anyio cancel scope instead,
    whose deadline every live heartbeat pushes out. mcp's own per-request clock cannot do this: it
    pauses only while a request waits on a person (``mcp.shared._request_clock``). So it is set to
    ``math.inf`` unless the caller set it. A read waiting behind a fit gets the fit's heartbeats,
    so it lives as long as the fit does.

    A cut call raises ``MCPError(REQUEST_TIMEOUT)``, the error mcp's clock raised. So it takes the
    path a timed-out fit always took (``agent.circuit_breaker._fit_wedged``), and the server is not
    told to stop the fit: the cut cancels the call's task, and ``runner.main._ToolCallsRunOn``
    sends no cancel for a ``tools/call`` abandoned that way either. A call cut while the server's
    heartbeats still came, showing a compile past its margin, carries the last one as the error's
    ``data["heartbeat"]``: the server is alive and goes on with the change, so the tool wrapper
    hands the cut to its caller as such and never counts it towards a wedge (gemot#160).

    The caller's own ``progress_callback`` is still called with every heartbeat. Shadowing the
    instance method is how the runner already reaches every call site (see
    :func:`_wrap_tool_clock`): the runner's own calls and the deepagents panel's route through the
    bound session, so no caller bypasses this.
    """
    heartbeats = _Heartbeats()
    original = session.call_tool

    async def idle_timeout(*args, **kwargs):
        # ClientSession.call_tool(name, arguments, read_timeout_seconds, progress_callback, *, ...):
        # both may arrive positionally, and a keyword on top of a positional one raises "got
        # multiple values for argument".
        args = list(args)
        if len(args) > 2:
            args[2] = math.inf if args[2] is None else args[2]
        elif kwargs.get("read_timeout_seconds") is None:
            kwargs["read_timeout_seconds"] = math.inf
        theirs = args[3] if len(args) > 3 else kwargs.get("progress_callback")
        scope = anyio.CancelScope(deadline=anyio.current_time() + CALL_IDLE_WINDOW_S)
        stuck: str | None = None  # the heartbeat of a compile past its margin, if the last one was

        async def heard(progress: float, total: float | None, message: str | None) -> None:
            nonlocal stuck
            stuck = None if heartbeats.hear(message) else message
            if stuck is None:
                scope.deadline = anyio.current_time() + CALL_IDLE_WINDOW_S
            if theirs is not None:
                await theirs(progress, total, message)

        if len(args) > 3:
            args[3] = heard
        else:
            kwargs["progress_callback"] = heard
        with scope:
            return await original(*args, **kwargs)
        # Reached only when the scope's own deadline passed: it swallowed that cancellation.
        name = args[0] if args else kwargs.get("name")
        if stuck is None:
            why = f"no heartbeat from the server in {CALL_IDLE_WINDOW_S:g}s"
            raise MCPError(code=REQUEST_TIMEOUT, message=f"Request {name!r} timed out: {why}")
        why = f"its change has been {stuck} (the compile margin is {FIT_COMPILE_MARGIN_S:g}s)"
        raise MCPError(
            code=REQUEST_TIMEOUT,
            message=f"Request {name!r} timed out: {why}",
            data={"heartbeat": stuck},
        )

    try:
        session.call_tool = idle_timeout  # type: ignore[method-assign]
    except Exception as exc:
        # Best-effort like the clock, but say so. Without the shadow every call falls back to the
        # ClientSession-wide read_timeout_seconds `runner.main` sets, which is sized to the run
        # DEFAULT fit -- safe off a pool, but blind to a fit that queues. Print it so a sweep
        # post-mortem can tell that apart from a genuinely silent server.
        print(
            f"[runner] could not install the heartbeat timeout ({type(exc).__name__}: {exc}) — "
            "tool calls fall back to the session-wide default-sized timeout"
        )
    return heartbeats


def _wrap_tool_clock(session: ClientSession) -> _ToolClock:
    """Time every ``session.call_tool`` into a returned clock (best-effort).

    Both the runner's own calls and the deepagents panel's calls route through
    ``session.call_tool`` (the tool bridge, ``agent.mcp_tools``, calls through it), so shadowing
    the instance method captures ALL tool time — which for these demos is dominated by the
    AMICI compile + every pyPESTO fit + the SQL queries. Subtracting it from the wall clock
    isolates the LLM/inference time. If the method can't be shadowed the clock simply stays
    at zero (timing is reported as unavailable) rather than breaking the run.

    asyncio is single-threaded and cooperative, so the ``enter``/``leave`` counter updates
    (no ``await`` between reading and mutating ``_active``) are atomic w.r.t. each other; the
    overlap is only ever inside the awaited ``original(...)``.
    """
    clock = _ToolClock()
    original = session.call_tool

    async def timed(*args, **kwargs):
        _name = kwargs.get("name") or (args[0] if args else None)
        clock.enter(_name if isinstance(_name, str) else None)
        try:
            result = await original(*args, **kwargs)
        finally:
            clock.leave()
        # Tally edit rejections here rather than at a call site: EVERY tool call routes through
        # this shadow (the runner's own and the agent panel's), so nothing can bypass the metric.
        with contextlib.suppress(Exception):
            # `_name` from above, not a second copy of the same expression: two derived metrics
            # (`by_tool` and `reject_causes`) must never disagree about what a call was called.
            if isinstance(_name, str):
                clock.note_result(_name, result)
        return result

    with contextlib.suppress(Exception):
        session.call_tool = timed  # type: ignore[method-assign]
    return clock
