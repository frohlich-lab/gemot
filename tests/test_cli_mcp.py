"""Real loopback MCP clients reach only the native gateway's advertised training tools.

The upstream is an initialized-session stand-in, so these tests exercise HTTP initialization,
tool discovery, forwarding, errors, tracing parentage and shutdown without starting a PEtab
server, fitting a model or invoking either native CLI.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from types import SimpleNamespace

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from gemot.agent import cli_mcp as C


def _result(payload, *, error=False):
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=error,
    )


class _Upstream:
    def __init__(self):
        self.calls = []
        self.cursors = []
        self.tools = [
            Tool(
                name="future_diagnostic",
                description="A server read outside the former panel's curated roster.",
                inputSchema={
                    "type": "object",
                    "properties": {"version": {"type": "integer"}},
                    "required": ["version"],
                    "additionalProperties": False,
                },
                outputSchema={"type": "object", "properties": {"version": {"type": "integer"}}},
                _meta={"upstream-marker": "preserved"},
            ),
            Tool(
                name="make_changes",
                description="Author, fit and snapshot one model.",
                inputSchema={"type": "object", "properties": {"justification": {"type": "string"}}},
            ),
            Tool(name="raises", inputSchema={"type": "object"}),
            Tool(name="evaluate_validation_data", inputSchema={"type": "object"}),
        ]

    async def list_tools(self, cursor=None):
        self.cursors.append(cursor)
        if cursor is None:
            return ListToolsResult(tools=self.tools[:2], nextCursor="page-2")
        assert cursor == "page-2"
        return ListToolsResult(tools=self.tools[2:])

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "raises":
            raise RuntimeError("upstream disconnected")
        if arguments.get("justification") == "reject":
            return _result({"applied": False, "error": "invalid model"})
        if arguments.get("justification") == "error":
            return _result({"ok": False, "error": "fit storage failed"}, error=True)
        return _result({"version": arguments.get("version", 4), "applied": True})


@contextlib.asynccontextmanager
async def _client(url):
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as client:
            initialized = await client.initialize()
            yield client, initialized


def test_gateway_preserves_schemas_and_forwards_calls_to_existing_session():
    async def run():
        upstream = _Upstream()
        async with C.serve_cli_tools(upstream) as gateway:
            assert upstream.cursors == [None, "page-2"]
            assert gateway.url.startswith("http://127.0.0.1:")
            async with _client(gateway.url) as (client, initialized):
                assert initialized.capabilities.tools is not None
                assert initialized.capabilities.resources is None
                assert initialized.capabilities.prompts is None
                listed = (await client.list_tools()).tools
                assert [tool.name for tool in listed] == list(gateway.tool_names)
                assert listed[0].model_dump(exclude={"meta"}) == upstream.tools[0].model_dump(
                    exclude={"meta"}
                )
                assert listed[0].meta == {
                    "upstream-marker": "preserved",
                    "anthropic/maxResultSizeChars": 500_000,
                }
                assert upstream.tools[0].meta == {"upstream-marker": "preserved"}
                assert "evaluate_validation_data" not in gateway.tool_names
                result = await client.call_tool("future_diagnostic", {"version": 7})
                assert result.model_dump() == _result({"version": 7, "applied": True}).model_dump()
                result = await client.call_tool("make_changes", {"justification": "measured rate"})
                assert result.structuredContent["applied"] is True
            assert upstream.calls == [
                ("future_diagnostic", {"version": 7}),
                ("make_changes", {"justification": "measured rate"}),
            ]
        assert gateway.stats == {
            "native_tool_calls": 2,
            "n_tool_errors": 0,
            "make_changes_calls": {
                "attempted": 1,
                "failed": 0,
                "fail_rate": 0.0,
                "by_outcome": {"ok": 1},
            },
        }
        # Exiting the context closes the listener, rather than leaving a second way to edit.
        async with httpx.AsyncClient() as http:
            with pytest.raises(httpx.ConnectError):
                await http.get(gateway.url)

    asyncio.run(run())


def test_hidden_tools_cannot_be_called_and_errors_keep_their_results():
    async def run():
        upstream = _Upstream()
        async with C.serve_cli_tools(upstream) as gateway:
            async with _client(gateway.url) as (client, _):
                hidden = await client.call_tool("evaluate_validation_data", {})
                assert hidden.isError
                assert upstream.calls == []
                rejected = await client.call_tool("make_changes", {"justification": "reject"})
                assert not rejected.isError
                assert rejected.structuredContent == {"applied": False, "error": "invalid model"}
                failed = await client.call_tool("make_changes", {"justification": "error"})
                assert failed.isError and failed.structuredContent["error"] == "fit storage failed"
                raised = await client.call_tool("raises", {})
                assert raised.isError and "upstream disconnected" in raised.content[0].text
            assert gateway.stats["n_tool_errors"] == 3
            assert gateway.stats["make_changes_calls"] == {
                "attempted": 2,
                "failed": 2,
                "fail_rate": 1.0,
                "by_outcome": {"server_rejected": 1, "tool_error": 1},
            }

    asyncio.run(run())


def test_tools_keep_the_run_span_as_parent_across_http_request_tasks(monkeypatch):
    spans = []
    parent = object()

    class Span:
        def __init__(self, name):
            self.name = name
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            self.status = status

    class Tracer:
        @contextlib.contextmanager
        def start_as_current_span(self, name, *, context):
            assert context is parent
            span = Span(name)
            spans.append(span)
            yield span

    handle = object()

    def trace_context(tracing):
        assert tracing is handle
        return Tracer(), parent

    monkeypatch.setattr(C, "_trace_context", trace_context)

    async def run():
        async with C.serve_cli_tools(_Upstream(), tracing=handle) as gateway:
            async with _client(gateway.url) as (client, _):
                await client.call_tool("future_diagnostic", {"version": 3})
                await client.call_tool("raises", {})

    asyncio.run(run())
    assert [span.name for span in spans] == ["future_diagnostic", "raises"]
    assert all(span.attributes["openinference.span.kind"] == "TOOL" for span in spans)
    assert json.loads(spans[0].attributes["input.value"]) == {"version": 3}
    assert json.loads(spans[0].attributes["output.value"])["structuredContent"]["version"] == 3
    assert spans[0].attributes["petab.tool_outcome"] == "ok"
    assert spans[1].attributes["petab.tool_outcome"] == "tool_error"


def test_gateway_rejects_wrong_path_host_and_browser_origin():
    async def run():
        async with C.serve_cli_tools(_Upstream()) as gateway:
            async with httpx.AsyncClient() as http:
                assert (await http.get(gateway.url + "/wrong")).status_code == 404
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                }
                response = await http.post(
                    gateway.url, json={}, headers={**headers, "Host": "attacker.test"}
                )
                assert response.status_code == 421
                response = await http.post(
                    gateway.url, json={}, headers={**headers, "Origin": "https://attacker.test"}
                )
                assert response.status_code == 403

    asyncio.run(run())


def test_cancelling_a_run_closes_its_gateway():
    async def run():
        ready = asyncio.Event()
        state = SimpleNamespace(url=None)

        async def hold():
            async with C.serve_cli_tools(_Upstream()) as gateway:
                state.url = gateway.url
                ready.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(hold())
        while not ready.is_set():
            if task.done():
                await task
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with httpx.AsyncClient() as http:
            with pytest.raises(httpx.ConnectError):
                await http.get(state.url)

    asyncio.run(run())


@pytest.mark.parametrize("exception", [TimeoutError("run deadline"), RuntimeError("CLI failed")])
def test_gateway_does_not_wrap_the_native_runs_failure(exception):
    async def run():
        with pytest.raises(type(exception)) as caught:
            async with C.serve_cli_tools(_Upstream()):
                raise exception
        assert caught.value is exception

    asyncio.run(run())


def test_blind_descriptions_do_not_mutate_the_upstream_tools():
    from gemot.prompts import BIOLOGY_BLIND_TOOL_DESCRIPTIONS

    async def run():
        upstream = _Upstream()
        originals = [tool.model_dump() for tool in upstream.tools]
        tools = await C._listed_tools(upstream, biology_blinded=True)
        editor = next(tool for tool in tools if tool.name == "make_changes")
        assert editor.description == BIOLOGY_BLIND_TOOL_DESCRIPTIONS["make_changes"]
        assert editor.inputSchema == upstream.tools[1].inputSchema
        assert [tool.model_dump() for tool in upstream.tools] == originals
        assert next(tool for tool in tools if tool.name == "future_diagnostic").description == ""

    asyncio.run(run())
