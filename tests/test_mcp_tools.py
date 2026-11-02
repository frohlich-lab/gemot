"""The MCP -> LangChain tool bridge (``agent.mcp_tools``), over a REAL ClientSession.

``langchain.mcp`` is a beta namespace and the bridge hands its ``as_langchain_tool`` an mcp
session where it expects a fastmcp client. These tests are where a langchain release that reads
something new off that client fails, instead of a sweep.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

pytest.importorskip("langchain.mcp", reason="agent stack not installed")

from gemot.agent.mcp_tools import load_mcp_tools  # noqa: E402


def _server():
    """Built the way petab-mcp builds its server: fastmcp, a `ToolError` for a reported failure."""
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError

    srv = FastMCP("bridge")

    @srv.tool()
    def fit(n_starts: int = 1) -> dict:
        """Fit the model."""
        return {"n_starts": n_starts, "nll": 1.5}

    @srv.tool()
    def broken() -> str:
        """Always fails."""
        raise ToolError("infrastructure_error: disk_full: no space left")

    return srv


@contextlib.asynccontextmanager
async def _session():
    """A real ClientSession on `_server()`, in the handshake era the runner's own session speaks."""
    from fastmcp import Client

    async with Client(_server(), mode="legacy") as client:
        yield client.session


def _call(tool, **args):
    return {"name": tool.name, "args": args, "id": "c1", "type": "tool_call"}


def test_a_shadow_installed_after_the_tools_were_built_sees_every_call():
    """The runner shadows `session.call_tool` (tool clock, per-call timeout) and the bio cache
    does too; a bridge that bound the method once, or called through a client of its own, would
    let the panel's calls bypass all three."""

    async def go():
        async with _session() as s:
            tools = {t.name: t for t in await load_mcp_tools(s)}
            seen = []
            original = s.call_tool

            async def shadow(*args, **kwargs):
                seen.append(args[0])
                return await original(*args, **kwargs)

            s.call_tool = shadow
            message = await tools["fit"].ainvoke(_call(tools["fit"], n_starts=3))
            return seen, message

    seen, message = asyncio.run(go())
    assert seen == ["fit"]
    # The shape langchain-mcp-adapters produced: text blocks, structured content as the artifact.
    assert message.content[0]["type"] == "text"
    assert message.artifact == {"structured_content": {"n_starts": 3, "nll": 1.5}}


def test_a_wrapped_tool_sends_the_server_only_its_own_arguments():
    """gemot#90: the circuit breaker and the literature recorder passed `runtime=None` on to the
    coroutine they wrap. langchain-mcp-adapters' coroutine took `runtime` as a parameter of its
    own; `langchain.mcp`'s takes `**arguments`, so `runtime` went to the server, which refused
    every call of every tool, and runs built no model. The tests that wrap fake coroutines could
    not see it. Both wrappers are layered here as the driver layers them."""
    from gemot.agent.circuit_breaker import _new_breaker_state, _wrap_tools_with_circuit_breaker
    from gemot.agent.literature import _wrap_literature_recorder

    async def go():
        async with _session() as s:
            tools = await load_mcp_tools(s)
            _wrap_tools_with_circuit_breaker(tools, _new_breaker_state())
            log: list = []
            _wrap_literature_recorder(tools, log)
            fit = {t.name: t for t in tools}["fit"]
            sent = []
            original = s.call_tool

            async def shadow(name, arguments, *args, **kwargs):
                sent.append(arguments)
                return await original(name, arguments, *args, **kwargs)

            s.call_tool = shadow
            message = await fit.ainvoke(_call(fit, n_starts=3))
            return sent, message, log

    sent, message, log = asyncio.run(go())
    assert sent == [{"n_starts": 3}]
    assert message.status == "success"
    assert message.artifact == {"structured_content": {"n_starts": 3, "nll": 1.5}}
    assert log[0]["query"] == {"n_starts": 3}


def test_an_error_result_reaches_the_model_as_a_failed_tool_message():
    """PEtab-MCP#95 reports a full disk as an `is_error` result; the circuit breaker reads it off
    a failed ToolMessage carrying the server's text, as it did through the adapters."""

    async def go():
        async with _session() as s:
            tools = {t.name: t for t in await load_mcp_tools(s)}
            return await tools["broken"].ainvoke(_call(tools["broken"]))

    message = asyncio.run(go())
    assert message.status == "error"
    assert "disk_full" in message.content[0]["text"]
