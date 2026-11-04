"""A client-side ``MCPError`` on one call reaches the model as that tool's result (gemot#103).

Run f2e2b86d stopped 15.8 h into a 33 h budget on ``MCPError: SSE stream failed: Server-sent event
exceeded the 1048576 byte limit.``: one ``get_version`` reply was larger than mcp's default limit
for one server-sent event. The server and the session were fine. The circuit breaker re-raised the
error, which is no ``ToolException``, so nothing turned it into a ToolMessage and the run ended.

The error's real shape is taken from a real server over streamable HTTP, so a change in how mcp
reports it fails here. A closed session is the exception to the rule and is pinned too: it stays a
transport error, never handed to the model. The driver-level case is in
tests/test_run_loop_no_model.py, the classification in tests/test_experiments_summary.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys

import pytest

pytest.importorskip("fastmcp", reason="dev dependency")
pytest.importorskip("langchain.mcp", reason="agent stack not installed")

from mcp import MCPError  # noqa: E402
from mcp.types import CONNECTION_CLOSED  # noqa: E402

from gemot.agent import circuit_breaker as B  # noqa: E402
from gemot.agent.mcp_tools import load_mcp_tools  # noqa: E402
from gemot.runner.main import _open_session  # noqa: E402
from gemot.runner.server import _free_port, _wait_until_listening  # noqa: E402

#: f2e2b86d's stop reason, without the class name the driver prefixes.
_TOO_LARGE = "SSE stream failed: Server-sent event exceeded the 1048576 byte limit."

#: Over mcp's default limit for one server-sent event (1 MiB), with the JSON-RPC framing on top.
_OVER_THE_LIMIT = 2 * 1024 * 1024

_SERVER = r"""
import sys
from fastmcp import FastMCP
srv = FastMCP("big")

@srv.tool()
def get_version(n_bytes: int) -> str:
    return "x" * n_bytes

srv.run(transport="streamable-http", host="127.0.0.1", port=int(sys.argv[1]))
"""


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    log = tmp_path_factory.mktemp("server") / "server.log"
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVER, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=log.open("w"),
    )
    try:
        _wait_until_listening("127.0.0.1", port, proc, log, timeout=60)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _too_large() -> MCPError:
    return MCPError(code=CONNECTION_CLOSED, message=_TOO_LARGE)


# --- the error's shape, on a real session ------------------------------------------------------


def test_a_plain_mcp_2_session_loses_a_reply_over_1_mib_and_keeps_working(server):
    """The control: what f2e2b86d's session did. The error carries CONNECTION_CLOSED, the code a
    closed session uses too, and the session answers the next call: the breaker must tell the two
    apart by more than the code."""
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def go():
        async with httpx2.AsyncClient() as http:
            async with streamable_http_client(server, http_client=http) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    with pytest.raises(MCPError) as raised:
                        await session.call_tool("get_version", {"n_bytes": _OVER_THE_LIMIT})
                    after = await session.call_tool("get_version", {"n_bytes": 3})
                    return raised.value, after.content[0].text

    error, after = asyncio.run(go())
    assert (error.code, error.message) == (CONNECTION_CLOSED, _TOO_LARGE)
    assert after == "xxx"
    assert B._is_client_mcp_error(error)
    assert not B._is_transport_error(error)


def test_the_runners_session_reads_a_reply_over_1_mib(server):
    """gemot#103 item 2: the runner's session to its own server sets no limit on one event."""

    async def go():
        async with _open_session(server, read_ceiling_s=60.0, call_timeout_s=30.0) as session:
            return await session.call_tool("get_version", {"n_bytes": _OVER_THE_LIMIT})

    result = asyncio.run(go())
    assert not result.is_error
    assert len(result.content[0].text) == _OVER_THE_LIMIT


def test_a_call_on_a_closed_session_stays_a_transport_error():
    """A closed session raises an MCPError with the same code. Handed to the model like the one
    above, it would come back on every call until the breaker cut the tool, or, for the
    never-disabled `make_changes`, until the turn limit: a dead server scored as a model that
    could not build anything."""
    from fastmcp import Client, FastMCP

    srv = FastMCP("closed")

    @srv.tool()
    def ping() -> str:
        return "pong"

    async def go():
        async with Client(srv, mode="legacy") as client:
            session = client.session
        with pytest.raises(MCPError) as raised:
            await session.call_tool("ping", {})
        return raised.value

    error = asyncio.run(go())
    assert error.code == CONNECTION_CLOSED
    assert B._is_session_closed(error)
    assert B._is_transport_error(error)


# --- the breaker -------------------------------------------------------------------------------


class _Tool:
    def __init__(self, name, coro):
        self.name, self.coroutine = name, coro


def _raising(error):
    calls = {"n": 0}

    async def coro(**kw):
        calls["n"] += 1
        raise error

    return coro, calls


def test_a_client_mcp_error_comes_back_as_the_tools_result():
    coro, _ = _raising(_too_large())
    state = B._new_breaker_state()
    tool = _Tool("get_version", coro)
    B._wrap_tools_with_circuit_breaker([tool], state)

    content, artifact = asyncio.run(tool.coroutine(version=23, include=["simulation"]))

    assert content == f"MCPError: {_TOO_LARGE}"
    assert artifact is None
    assert state["n_tool_errors"] == 1
    assert B.call_outcomes(state, "get_version")["by_outcome"] == {"tool_error": 1}


def test_a_repeated_client_mcp_error_still_trips_the_breaker():
    """Returned rather than raised, it still counts: three identical errors cut the tool."""
    coro, calls = _raising(_too_large())
    state = B._new_breaker_state()
    tool = _Tool("get_version", coro)
    B._wrap_tools_with_circuit_breaker([tool], state)

    results = [asyncio.run(tool.coroutine(version=23)) for _ in range(4)]

    assert [r[0] for r in results[:2]] == [f"MCPError: {_TOO_LARGE}"] * 2
    assert "TOOL DISABLED" in results[2][0][0]["text"]
    assert "get_version" in state["disabled"]
    assert calls["n"] == 3  # the fourth call never reached the session


def test_a_server_error_response_comes_back_too():
    """A JSON-RPC error the server answers with is an MCPError as well, and as much the model's to
    act on (tests/test_fit_timeouts.py has said so of a 400 since the timeout moved per call)."""
    coro, _ = _raising(MCPError(code=-32602, message="Unknown tool: get_versoin"))
    tool = _Tool("get_versoin", coro)
    B._wrap_tools_with_circuit_breaker([tool], B._new_breaker_state())

    assert asyncio.run(tool.coroutine())[0] == "MCPError: Unknown tool: get_versoin"


@pytest.mark.parametrize("name", ["get_version", "make_changes"])
def test_a_closed_session_is_still_raised(name):
    """Before this, `make_changes` (never disabled) handed "Connection closed" to the model as a
    result; a closed session now reaches the driver from every tool, as it did from the rest."""
    coro, _ = _raising(MCPError(code=CONNECTION_CLOSED, message="Connection closed"))
    tool = _Tool(name, coro)
    B._wrap_tools_with_circuit_breaker([tool], B._new_breaker_state())

    with pytest.raises(MCPError, match="Connection closed"):
        asyncio.run(tool.coroutine())


def test_a_closed_bio_session_disables_its_tool():
    """A hosted bio server gone away: its session closes, and the tool goes at the first failure,
    as for any transport error on an external tool."""
    coro, calls = _raising(MCPError(code=CONNECTION_CLOSED, message="Connection closed"))
    state = B._new_breaker_state()
    tool = _Tool("query_omnipath", coro)
    B._wrap_tools_with_circuit_breaker([tool], state, transport_disable_names={"query_omnipath"})

    content, _ = asyncio.run(tool.coroutine())

    assert "TOOL DISABLED" in content[0]["text"]
    assert "query_omnipath" in state["disabled"]
    assert calls["n"] == 1


# --- through the real tool bridge --------------------------------------------------------------


@contextlib.asynccontextmanager
async def _bridged_session():
    """A real ClientSession whose `get_version` is an in-memory fastmcp tool."""
    from fastmcp import Client, FastMCP

    srv = FastMCP("bridge")

    @srv.tool()
    def get_version(version: int) -> str:
        """A version."""
        return f"version {version}"

    async with Client(srv, mode="legacy") as client:
        yield client.session


def test_through_the_bridge_the_error_is_a_tool_message_not_a_raise():
    """Where f2e2b86d ended: `langchain.mcp` hands the model only an `is_error` result, and
    LangChain re-raises any exception that is no ToolException. The session raises the exact
    MCPError the run stopped on; the tool call must answer with a ToolMessage, and the next call
    must reach the server."""

    async def go():
        async with _bridged_session() as session:
            tools = await load_mcp_tools(session)
            B._wrap_tools_with_circuit_breaker(tools, B._new_breaker_state())
            get_version = {t.name: t for t in tools}["get_version"]
            original = session.call_tool

            async def too_large_once(name, arguments, *args, **kwargs):
                session.call_tool = original
                raise _too_large()

            session.call_tool = too_large_once
            call = {"name": "get_version", "args": {"version": 23}, "type": "tool_call"}
            first = await get_version.ainvoke({**call, "id": "c1"})
            second = await get_version.ainvoke({**call, "id": "c2"})
            return first, second

    first, second = asyncio.run(go())
    assert first.content == f"MCPError: {_TOO_LARGE}"
    assert first.tool_call_id == "c1"
    assert second.content[0]["text"] == "version 23"
