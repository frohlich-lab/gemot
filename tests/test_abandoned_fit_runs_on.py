"""A fit call gemot stops waiting for finishes on the server instead of being cancelled.

mcp 2 sends ``notifications/cancelled`` when a call is abandoned -- its read timeout fires, or the
task awaiting it is cancelled -- and petab-mcp then stops the fit. gemot keeps mcp 1.x's behaviour
of telling the server nothing (``runner.main._ToolCallsRunOn``), so the fit commits and the agent
(``wait_for_fit``) and the judge (gemot#79) can use its version. Behind the native CLI driver's
gateway (gemot#69) EVERY fit past the CLI's 600 s tool timeout is abandoned by a cancelled caller,
so losing this would stop all of them.

Against a real server over streamable HTTP, through the session ``runner.main`` opens. Each test
keeps its session open past the fit's end, as the runner keeps its own for the whole run: closing a
session ends it on the server, which stops whatever it still runs, cancel or no cancel. The control
shows a plain mcp 2 session does cancel, so these tests can tell the two apart.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time

import pytest

pytest.importorskip("fastmcp", reason="dev dependency")

from gemot.runner.main import _open_session  # noqa: E402
from gemot.runner.server import _free_port, _wait_until_listening  # noqa: E402

_SERVER = r"""
import sys, anyio
from pathlib import Path
from fastmcp import FastMCP
marker = Path(sys.argv[1])
srv = FastMCP("slow")

@srv.tool()
async def fit(tag: str, seconds: float = 1.5) -> str:
    try:
        await anyio.sleep(seconds)
    except anyio.get_cancelled_exc_class():
        (marker / f"{tag}.cancelled").write_text("1")
        raise
    (marker / f"{tag}.finished").write_text("1")
    return "fitted"

srv.run(transport="streamable-http", host="127.0.0.1", port=int(sys.argv[2]))
"""


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    marker = tmp_path_factory.mktemp("marker")
    port = _free_port()
    log = marker / "server.log"
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(marker), str(port)],
        stdout=subprocess.DEVNULL,
        stderr=log.open("w"),
    )
    try:
        _wait_until_listening("127.0.0.1", port, proc, log, timeout=60)
        yield f"http://127.0.0.1:{port}/mcp", marker
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _outcome(marker, tag, wait_s=10.0):
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        for outcome in ("finished", "cancelled"):
            if (marker / f"{tag}.{outcome}").exists():
                return outcome
        time.sleep(0.05)
    return None


#: Longer than the fit (1.5 s): how long a session stays open after it gave up on the call.
_HOLD_S = 2.5


def _open(url):
    return _open_session(url, read_ceiling_s=60.0, call_timeout_s=30.0)


def test_a_fit_call_that_times_out_still_finishes_on_the_server(server):
    from mcp import MCPError
    from mcp.types import REQUEST_TIMEOUT

    url, marker = server

    async def go():
        async with _open(url) as session:
            with pytest.raises(MCPError) as raised:
                await session.call_tool("fit", {"tag": "timeout"}, read_timeout_seconds=0.3)
            await asyncio.sleep(_HOLD_S)
            return raised.value

    assert asyncio.run(go()).code == REQUEST_TIMEOUT
    assert _outcome(marker, "timeout") == "finished"


def test_a_fit_call_whose_caller_is_cancelled_still_finishes(server):
    """The gateway's shape: the CLI gives up, its handler is cancelled, and with it the task
    awaiting this session's call. Also a sibling tool raising in the same agent step."""
    url, marker = server

    async def go():
        async with _open(url) as session:
            call = asyncio.create_task(session.call_tool("fit", {"tag": "caller"}))
            await asyncio.sleep(0.3)
            call.cancel()
            with pytest.raises(asyncio.CancelledError):
                await call
            await asyncio.sleep(_HOLD_S)

    asyncio.run(go())
    assert _outcome(marker, "caller") == "finished"


def test_the_control_a_plain_mcp_2_session_cancels_the_fit(server):
    import httpx2
    from mcp import ClientSession, MCPError
    from mcp.client.streamable_http import streamable_http_client

    url, marker = server

    async def go():
        async with httpx2.AsyncClient() as http:
            async with streamable_http_client(url, http_client=http) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    with pytest.raises(MCPError):
                        await session.call_tool("fit", {"tag": "plain"}, read_timeout_seconds=0.3)
                    await asyncio.sleep(_HOLD_S)

    asyncio.run(go())
    assert _outcome(marker, "plain") == "cancelled"
