"""The MCP call handle the pipeline uses, and the tool clock wrapped around it.

``_ToolClock`` measures the wall-clock the run actually spent INSIDE tool calls. It tracks the
UNION of busy intervals rather than their sum, so concurrent analyst calls are not double-counted
and the reported compute time stays comparable across runs with different panel concurrency.
"""

from __future__ import annotations

import contextlib
import json
import time

from mcp import ClientSession

# --------------------------------------------------------------------------- MCP glue
def _unwrap(result):
    """Turn a CallToolResult into the tool's Python return value."""
    sc = getattr(result, "structuredContent", None)
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


def _make_call(session: ClientSession):
    """Return an ``await call(tool, **args)`` bound to ``session`` (unwraps the result)."""

    async def call(tool: str, /, **args):
        result = await session.call_tool(tool, args or {})
        return _unwrap(result)

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
    ("bad_bounds_usage", "sets only one bound"),
    ("unknown_key", "unknown key"),
    ("unknown_parameter", "unknown parameter id"),
    ("bad_scale", "invalid parameterscale"),
    ("bad_estimate", "cannot interpret estimate value"),
    ("missing_id", "must include an 'id'"),
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

    __slots__ = ("seconds", "calls", "_active", "_since", "edits", "rejects", "fit_reverts",
                 "reject_causes", "by_tool")

    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0
        self._active = 0  # in-flight call count
        self._since = 0.0  # monotonic time the busy window (0 -> 1) opened
        self.edits = 0
        self.rejects = 0
        self.fit_reverts = 0
        self.reject_causes: dict[str, int] = {}
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
        cause = _reject_cause(str(payload.get("error") or ""))
        self.reject_causes[cause] = self.reject_causes.get(cause, 0) + 1

    def reject_summary(self) -> dict:
        """``{attempted, rejected, reject_rate, by_cause}`` for ``result.json``."""
        return {
            "attempted": self.edits,
            "rejected": self.rejects,
            "fit_reverted": self.fit_reverts,
            "reject_rate": round(self.rejects / self.edits, 3) if self.edits else None,
            "by_cause": dict(sorted(self.reject_causes.items(), key=lambda kv: -kv[1])),
        }


def _wrap_tool_clock(session: ClientSession) -> _ToolClock:
    """Time every ``session.call_tool`` into a returned clock (best-effort).

    Both the runner's own calls and the deepagents panel's calls route through
    ``session.call_tool`` (langchain-mcp-adapters invokes the bound session), so shadowing
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
