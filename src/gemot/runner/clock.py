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
    """

    __slots__ = ("seconds", "calls", "_active", "_since")

    def __init__(self) -> None:
        self.seconds = 0.0
        self.calls = 0
        self._active = 0  # in-flight call count
        self._since = 0.0  # monotonic time the busy window (0 -> 1) opened

    def enter(self) -> None:
        if self._active == 0:
            self._since = time.monotonic()
        self._active += 1
        self.calls += 1

    def leave(self) -> None:
        self._active -= 1
        if self._active == 0:
            self.seconds += time.monotonic() - self._since


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
        clock.enter()
        try:
            return await original(*args, **kwargs)
        finally:
            clock.leave()

    with contextlib.suppress(Exception):
        session.call_tool = timed  # type: ignore[method-assign]
    return clock
