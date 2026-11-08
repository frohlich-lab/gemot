"""A loopback MCP gateway for a native CLI's whole model-building run.

The CLI owns its conversation and tool loop. This module only exposes the already connected
PEtab session over HTTP: calls retain the runner's timing, timeouts and anonymous-session
wrappers, and ``make_changes`` remains the server's scored, versioned editing boundary.
Tool schemas are copied from the server without constructing a LangChain agent or a panel.

The judge's held-out-data tool is neither listed nor callable. No resources, resource templates
or prompts are proxied. Each run binds an ephemeral loopback port and an unguessable URL path;
the gateway lasts only as long as the CLI run and records its calls beneath that run's Phoenix
span even though HTTP request tasks arrive independently of the runner task.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .. import ablations


#: The server registers this read solely for the harness's independent held-out evaluation.
_JUDGE_TOOLS = frozenset({"evaluate_validation_data"})
#: The early-stop and best-version probe, the harness's alone where the agent is not told how it is
#: doing (the progress_signals arm).
_PROGRESS_TOOLS = frozenset({"get_progress"})


@dataclass
class CliToolGateway:
    """The temporary endpoint and live counters shared with the native CLI driver."""

    url: str
    tool_names: tuple[str, ...]
    stats: dict[str, Any]


def _trace_context(tracing: Any) -> tuple[Any, Any]:
    """Capture the run's parent once; HTTP request tasks need not inherit its OTel context."""
    if tracing is None:
        return None, None
    try:
        from opentelemetry import context, trace

        provider = getattr(tracing, "tracer_provider", None)
        tracer = (
            provider.get_tracer("gemot.cli_mcp")
            if provider is not None
            else trace.get_tracer("gemot.cli_mcp")
        )
        return tracer, context.get_current()
    except Exception:
        return None, None


def _span_attributes(span: Any, attributes: dict[str, Any]) -> None:
    if span is not None:
        for key, value in attributes.items():
            with contextlib.suppress(Exception):
                span.set_attribute(key, value)


def _span_error(span: Any, message: str) -> None:
    if span is not None:
        with contextlib.suppress(Exception):
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, message))


@contextlib.contextmanager
def _tool_span(tracer: Any, parent: Any, name: str, arguments: dict[str, Any]):
    with contextlib.ExitStack() as stack:
        span = None
        if tracer is not None:
            with contextlib.suppress(Exception):
                span = stack.enter_context(tracer.start_as_current_span(name, context=parent))
        _span_attributes(
            span,
            {
                "openinference.span.kind": "TOOL",
                "tool.name": name,
                "input.value": json.dumps(arguments, ensure_ascii=False, default=str),
                "input.mime_type": "application/json",
            },
        )
        yield span


def _outcome(result: Any) -> str:
    """Distinguish a scored edit rejection from a failed tool, without changing either reply."""
    if result.is_error:
        return "tool_error"
    payload = result.structured_content
    if payload is None:
        text = "".join(
            block.text for block in result.content if getattr(block, "type", None) == "text"
        )
        with contextlib.suppress(ValueError):
            payload = json.loads(text)
    if isinstance(payload, dict):
        if payload.get("ok") is False and "error" in payload:
            return "tool_error"
        if payload.get("applied") is False:
            return "server_rejected"
    return "ok"


def _record(stats: dict[str, Any], name: str, outcome: str) -> None:
    if outcome in {"tool_error", "cancelled"}:
        stats["n_tool_errors"] += 1
    if name == "make_changes":
        calls = stats["make_changes_calls"]
        by = calls["by_outcome"]
        by[outcome] = by.get(outcome, 0) + 1
        calls["attempted"] += 1
        calls["failed"] += outcome != "ok"
        calls["fail_rate"] = round(calls["failed"] / calls["attempted"], 3)


async def _listed_tools(session: Any, *, biology_blinded: bool) -> list[Any]:
    """Read every tools/list page before publishing the stable surface for this run."""
    from mcp.types import PaginatedRequestParams

    hidden = _JUDGE_TOOLS | (
        frozenset() if ablations.enabled("progress_signals") else _PROGRESS_TOOLS
    )
    tools: dict[str, Any] = {}
    cursor = None
    seen_cursors = set()
    while True:
        page = await session.list_tools(
            params=PaginatedRequestParams(cursor=cursor) if cursor else None
        )
        for tool in page.tools:
            if tool.name not in hidden:
                tools[tool.name] = tool
        cursor = page.next_cursor
        if not cursor:
            break
        if cursor in seen_cursors:
            raise RuntimeError("PEtab MCP repeated a tools/list cursor")
        seen_cursors.add(cursor)
    if biology_blinded:
        from ..prompts import BIOLOGY_BLIND_TOOL_DESCRIPTIONS

        tools = {
            name: tool.model_copy(
                update={"description": BIOLOGY_BLIND_TOOL_DESCRIPTIONS.get(name, "")}
            )
            for name, tool in tools.items()
        }
    # Claude spills successful text replies above its default 50,000-character threshold
    # into files. Its Read tool is unavailable in these runs, so request the documented
    # maximum inline size. Claude still caps this at 500,000 characters; the gateway itself
    # forwards the original reply and records it without truncating scientific output.
    return [
        tool.model_copy(
            update={"meta": {**(tool.meta or {}), "anthropic/maxResultSizeChars": 500_000}}
        )
        for tool in tools.values()
    ]


@contextlib.asynccontextmanager
async def serve_cli_tools(
    session: Any,
    *,
    tracing: Any = None,
    biology_blinded: bool = False,
) -> AsyncIterator[CliToolGateway]:
    """Expose ``session`` to one native CLI run and close its endpoint on exit.

    ``session`` must already be initialized and carry the runner's desired wrappers. Its tool
    descriptions and JSON schemas pass through unchanged, except for the existing anonymous
    descriptions when ``biology_blinded`` is set. Tool metadata requests Claude's largest
    supported inline result size. ``stats`` remains readable after shutdown.
    ``tracing`` is the normal :class:`gemot.tracing.TracingHandle`; create this gateway while the
    run span is current so the tool spans retain that parent.
    """
    import uvicorn
    from mcp.server.lowlevel import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import CallToolResult, ListToolsResult, TextContent

    tools = await _listed_tools(session, biology_blinded=biology_blinded)
    names = frozenset(tool.name for tool in tools)
    stats = {
        "native_tool_calls": 0,
        "n_tool_errors": 0,
        "make_changes_calls": {
            "attempted": 0,
            "failed": 0,
            "fail_rate": None,
            "by_outcome": {},
        },
    }
    tracer, parent = _trace_context(tracing)

    async def list_tools(ctx, params):
        return ListToolsResult(tools=tools)

    # Validation belongs to the real server, and this low-level server does none: that keeps its
    # normal diagnostics, and every forwarded attempt is timed and traced, a malformed one too.
    async def call_tool(ctx, params):
        name, arguments = params.name, params.arguments or {}
        stats["native_tool_calls"] += 1
        with _tool_span(tracer, parent, name, arguments) as span:
            try:
                if name not in names:
                    raise ValueError(f"Tool is not exposed by the CLI gateway: {name}")
                result = await session.call_tool(name, arguments=arguments)
            except asyncio.CancelledError:
                _record(stats, name, "cancelled")
                _span_attributes(span, {"petab.tool_outcome": "cancelled"})
                _span_error(span, "Tool call cancelled")
                raise
            except Exception as exc:
                result = CallToolResult(
                    content=[TextContent(type="text", text=f"{type(exc).__name__}: {exc}")],
                    is_error=True,
                )
            outcome = _outcome(result)
            _record(stats, name, outcome)
            _span_attributes(
                span,
                {
                    "output.value": result.model_dump_json(by_alias=True),
                    "output.mime_type": "application/json",
                    "petab.tool_outcome": outcome,
                },
            )
            if outcome == "tool_error":
                _span_error(span, "Tool returned an error")
            return result

    server = Server("gemot-petab", on_list_tools=list_tools, on_call_tool=call_tool)
    # mcp 2 wraps each message in an OTel SERVER span of its own. The CLI sends no trace context,
    # so those would be parentless duplicates of the tool spans above, which keep the run's parent.
    server.middleware = []

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    http_task = None
    http_server = None
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.setblocking(False)
        port = sock.getsockname()[1]
        route = f"/{uuid.uuid4().hex}/mcp"
        manager = StreamableHTTPSessionManager(
            server,
            json_response=True,
            stateless=True,
            security_settings=TransportSecuritySettings(
                allowed_hosts=[f"127.0.0.1:{port}"], allowed_origins=[]
            ),
        )

        async def app(scope, receive, send):
            if scope["type"] == "http" and scope["path"] == route:
                await manager.handle_request(scope, receive, send)
                return
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"Not found"})

        class _EmbeddedServer(uvicorn.Server):
            # The runner owns Ctrl-C. Embedding uvicorn must not install or replay its own
            # process-wide handlers (the two hooks cover the supported uvicorn releases).
            def install_signal_handlers(self):
                pass

            @contextlib.contextmanager
            def capture_signals(self):
                yield

        http_server = _EmbeddedServer(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                interface="asgi3",
                lifespan="off",
                loop="asyncio",
                http="h11",
                ws="none",
                access_log=False,
                log_config=None,
                log_level="warning",
                timeout_graceful_shutdown=2,
            )
        )
        # The manager owns an anyio TaskGroup. Do not pass the caller's CLI exception through
        # its __aexit__: TaskGroup would wrap an ordinary timeout or process error in an
        # ExceptionGroup and change the native driver's stop classification.
        running = manager.run()
        await running.__aenter__()
        try:
            http_task = asyncio.create_task(http_server.serve(sockets=[sock]))
            try:
                async with asyncio.timeout(10):
                    while not http_server.started:
                        if http_task.done():
                            await http_task
                            raise RuntimeError("CLI MCP gateway stopped before startup")
                        await asyncio.sleep(0.01)
                yield CliToolGateway(
                    url=f"http://127.0.0.1:{port}{route}",
                    tool_names=tuple(tool.name for tool in tools),
                    stats=stats,
                )
            finally:
                http_server.should_exit = True
                try:
                    await asyncio.wait_for(http_task, timeout=5)
                except TimeoutError:
                    http_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await http_task
                except asyncio.CancelledError:
                    http_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await http_task
                    raise
        finally:
            await running.__aexit__(None, None, None)
    finally:
        sock.close()
