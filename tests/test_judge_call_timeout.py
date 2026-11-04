"""End-to-end cover for the per-call timeout, over a REAL ClientSession.

Written after a post-mortem of bachmann task 52859568_0 went wrong twice for want of exactly
this. The abort read ``McpError: ... Waited 4200.0 seconds`` (mcp 1.x's name), and
`runner.failure._summarize_exc` unwraps an exception group to its first leaf — so that text is
equally consistent with a bare ``McpError`` and with ``BaseExceptionGroup(McpError,
CancelledError)``. Reasoning from the text
alone produced a confident, wrong diagnosis (that the group shape was defeating
`_is_call_timeout`, whose first line is ``type(exc).__name__ != "MCPError"``). It is not: mcp
raises the timeout BARE, which the classifier already handles.

So these tests pin the real shape against the real client rather than a hand-built exception,
and they do it behind a genuine CPU-bound optimisation run off-thread — the way a fitting tool
actually blocks — instead of an ``asyncio.sleep`` that could hide a scheduling difference.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from gemot.agent.circuit_breaker import _is_call_timeout, _is_transport_error
from gemot.runner.clock import _make_call, _wrap_call_timeout


def _rosenbrock_multistart(n_starts: int) -> float:
    """A real (small) multistart on the Rosenbrock function — CPU-bound, no I/O, like a fit."""
    import numpy as np
    from scipy.optimize import minimize

    rng = np.random.default_rng(0)
    best = float("inf")
    for _ in range(n_starts):
        x0 = rng.uniform(-2.0, 2.0, size=6)
        res = minimize(
            lambda x: sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2), x0, method="BFGS"
        )
        best = min(best, float(res.fun))
    return best


def _server():
    """A one-tool MCP server whose `make_changes` runs the multistart OFF-THREAD.

    This DIFFERS from production on purpose, and the difference matters enough to state: the real
    `make_changes` is a sync `def` and mcp runs sync tool functions directly on the event loop
    (`func_metadata`: ``if fn_is_async: await fn(...) else: return fn(...)``), so a real server
    answers nothing at all for the duration of a fit. Off-thread here keeps the loop responsive so
    these tests can measure the CLIENT-side cut — which is what they are about — instead of
    deadlocking on a server that cannot reply. What that server-side blocking implies for liveness
    is covered separately by `test_a_BUSY_server_must_not_read_as_dead`; do not read these tests
    as evidence that a busy server stays reachable, because it does not."""
    import anyio
    from mcp.server import MCPServer

    srv = MCPServer("rosenbrock")

    @srv.tool()
    async def make_changes(n_starts: int = 1) -> str:
        best = await anyio.to_thread.run_sync(_rosenbrock_multistart, n_starts)
        return f"fitted {n_starts} starts, best={best:.3g}"

    @srv.tool()
    async def get_progress() -> dict:
        """The trivial read the liveness probe uses."""
        return {"should_stop": False}

    return srv


@contextlib.asynccontextmanager
async def _session():
    """A real ClientSession on `_server()`, in the handshake era the runner's own session speaks."""
    from mcp import Client

    async with Client(_server(), mode="legacy") as client:
        yield client.session


@pytest.mark.parametrize("n_starts", [1, 4])
def test_a_fit_that_finishes_inside_the_window_returns_normally(n_starts):
    """The control. Without this, a test asserting "it times out" also passes when the tool is
    simply broken."""

    async def go():
        async with _session() as s:
            _wrap_call_timeout(s)
            out = await _make_call(s)("make_changes", n_starts=n_starts)
            assert "fitted" in str(out)

    asyncio.run(go())


def test_an_overrunning_fit_raises_a_BARE_MCPError_that_the_classifier_catches():
    """The shape assertion. `_is_call_timeout` reads `type(exc).__name__` and `exc.code`, so it
    only works if mcp hands the timeout up unwrapped — assert that directly, because the
    stop_reason recorded in result.json cannot distinguish it. The code is checked against the
    SDK's constant: mcp 1.x sent HTTP's 408 here, mcp 2.0 sends -32001, and a classifier still
    matching 408 never fired."""
    from mcp.types import REQUEST_TIMEOUT

    async def go():
        async with _session() as s:
            with pytest.raises(BaseException) as ei:
                # 200 starts against a 50ms budget: the cut fires long before the fit ends.
                await s.call_tool("make_changes", {"n_starts": 200}, read_timeout_seconds=0.05)
            return ei.value

    exc = asyncio.run(go())
    assert type(exc).__name__ == "MCPError", f"shape changed: {type(exc).__name__}"
    # Bare, NOT wrapped in a group — an ExceptionGroup would slip past the driver's classifiers.
    assert isinstance(exc, Exception) and not isinstance(exc, BaseExceptionGroup)
    assert exc.code == REQUEST_TIMEOUT
    assert _is_call_timeout(exc)
    assert _is_transport_error(exc)


# --- liveness -------------------------------------------------------------------------------
# Nothing in the run loop asked whether the server was still there, so bachmann 52859568_0 kept
# resuming into one the OOM killer had taken down: ten consecutive 4200s timeouts, 11.7 hours,
# and a run with 14 healthy fits on disk discarded with delta_bic None.


def test_liveness_reads_popen_semantics_not_truthiness():
    """`Popen.poll()` returns None while RUNNING and an int once exited. Taking the returncode is
    what makes `server_poll=proc.poll` — the obvious wiring — correct rather than exactly
    inverted; `bool(poll())` would read None as False and call every healthy server dead."""
    from gemot.agent.driver import _server_is_alive

    assert _server_is_alive(lambda: None) is True  # running
    assert _server_is_alive(lambda: 0) is False  # exited cleanly
    assert _server_is_alive(lambda: -9) is False  # SIGKILL — the OOM case


def test_no_signal_is_not_evidence_of_death():
    """`None`, not False: a caller that wires nothing must not have its runs ended on no
    evidence. The call site checks `is False` for exactly this reason."""
    from gemot.agent.driver import _server_is_alive

    assert _server_is_alive(None) is None

    def boom():
        raise OSError("no such process")

    assert _server_is_alive(boom) is None


def test_liveness_against_a_real_process():
    """End-to-end on the actual object production passes: `Popen.poll` itself, not a lambda."""
    import signal
    import subprocess
    import sys

    from gemot.agent.driver import _server_is_alive

    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert _server_is_alive(proc.poll) is True
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    assert proc.poll() == -9  # what bachmann 52859568_0 recorded as server_returncode
    assert _server_is_alive(proc.poll) is False
