"""The native orchestrator launches one session and observes its own event stream.

CLI subprocess behavior and real HTTP forwarding have their own tests. Here the process and
gateway are inert stand-ins, making it observable if the orchestration starts a model per
turn, modifies the supplied task, loses the event callback, or leaks its temporary directory.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from contextlib import asynccontextmanager
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest

from gemot.agent import native_cli as N
from gemot import infrastructure, prompts
from gemot.agent import cli_mcp, cli_process
from gemot.agent.cli_process import CLIProcessError, CLIProcessResult
from gemot.agent.codex import CODEX_TOKENS_PER_TOOL_CALL


_TASK = "Build a model for this measured time course. Native session test task."
_STATS = {
    "native_tool_calls": 2,
    "n_tool_errors": 0,
    "make_changes_calls": {
        "attempted": 1,
        "failed": 0,
        "fail_rate": 0.0,
        "by_outcome": {"ok": 1},
    },
}


def _events(driver):
    if driver == "codex":
        return [
            {"type": "thread.started", "thread_id": "native-session"},
            {
                "type": "item.completed",
                "item": {"id": "answer", "type": "agent_message", "text": "done"},
            },
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 10, "cached_input_tokens": 5, "output_tokens": 4},
            },
        ]
    return [
        {"type": "system", "subtype": "init", "session_id": "native-session"},
        {
            "type": "assistant",
            "message": {
                "id": "answer",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "usage": {
                    "input_tokens": 3,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 2,
                    "output_tokens": 4,
                },
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "done",
            "session_id": "native-session",
            "usage": {
                "input_tokens": 3,
                "cache_read_input_tokens": 5,
                "cache_creation_input_tokens": 2,
                "output_tokens": 4,
            },
        },
    ]


def _stub_boundaries(monkeypatch, driver, *, process_error=None, events=None, tool_calls=2):
    seen = []
    state = SimpleNamespace(open=False, cwd=None, event_log=None)
    session = object()
    parent = object()

    @asynccontextmanager
    async def gateway(upstream, *, tracing=None, biology_blinded=False):
        assert upstream is session
        state.open = True
        seen.append("gateway-open")
        try:
            yield SimpleNamespace(
                url="http://127.0.0.1:1234/run/mcp",
                tool_names=("get_fit_summary", "make_changes"),
                stats={**_STATS, "native_tool_calls": tool_calls},
            )
        finally:
            state.open = False
            seen.append("gateway-close")

    def command(**kwargs):
        assert state.open
        assert kwargs == {
            "executable": "native-executable",
            "model": "raw-native-model",
            "reasoning_effort": "high",
            "mcp_url": "http://127.0.0.1:1234/run/mcp",
            "tool_timeout": 1234.0,
        }
        seen.append("command")
        return ["native-executable", driver]

    async def process(argv, *, prompt, cwd, event_log, timeout, on_event):
        assert state.open and argv == ["native-executable", driver]
        assert prompt == prompts.NATIVE_CLI_INSTRUCTIONS + _TASK and timeout is None
        state.cwd, state.event_log = Path(cwd), Path(event_log)
        assert state.cwd.is_dir()
        seen.append("process")
        native_events = _events(driver) if events is None else events
        state.event_log.write_text("".join(json.dumps(event) + "\n" for event in native_events))
        for event in native_events:
            result = on_event(event)
            if inspect.isawaitable(result):
                await result
        if process_error is not None:
            raise process_error
        return CLIProcessResult(returncode=0, stderr="", event_count=len(native_events))

    transport = importlib.import_module(f"gemot.agent.{driver}")
    monkeypatch.setattr(transport, f"{driver}_command", command)
    monkeypatch.setattr(cli_mcp, "serve_cli_tools", gateway)
    monkeypatch.setattr(cli_process, "run_cli", process)
    return session, parent, state, seen


@pytest.mark.parametrize("driver", ["codex", "claude"])
def test_one_native_process_owns_all_turns_and_keeps_gateway_alive(monkeypatch, tmp_path, driver):
    session, parent, state, seen = _stub_boundaries(monkeypatch, driver)
    result = asyncio.run(
        N.run_native_cli(
            session,
            driver=driver,
            model="raw-native-model",
            task=_TASK,
            work=tmp_path,
            executable="native-executable",
            reasoning_effort="high",
            timeout=75.0,
            tool_timeout=1234.0,
            parent_span=parent,
        )
    )
    assert seen == ["gateway-open", "command", "process", "gateway-close"]
    assert state.open is False and not state.cwd.exists()
    assert state.event_log.parent == tmp_path and state.event_log.is_file()
    assert result["stop_reason"] == "completed"
    assert result["token_usage"]["total_tokens"] == 14
    assert all(result[key] == value for key, value in _STATS.items())


@pytest.mark.parametrize("driver", ["codex", "claude"])
@pytest.mark.parametrize("failure", ["missing_terminal", "process", "cancelled", "no_tools"])
def test_native_failure_preserves_logs_and_closes_boundaries(
    monkeypatch, tmp_path, driver, failure
):
    error = None
    if failure == "process":
        error = CLIProcessError("native process failed", returncode=23)
    elif failure == "cancelled":
        error = asyncio.CancelledError()
    session, _, state, seen = _stub_boundaries(
        monkeypatch,
        driver,
        events=_events(driver)[:-1] if failure == "missing_terminal" else None,
        process_error=error,
        tool_calls=0 if failure == "no_tools" else 2,
    )
    run = N.run_native_cli(
        session,
        driver=driver,
        model="raw-native-model",
        task=_TASK,
        work=tmp_path,
        executable="native-executable",
        reasoning_effort="high",
        timeout=75.0,
        tool_timeout=1234.0,
    )
    if failure == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(run)
    else:
        result = asyncio.run(run)
        assert result["stop_reason"] == (
            "no_tools_executed" if failure == "no_tools" else "cli_error"
        )
        if failure == "process":
            assert result["cli_returncode"] == 23
            assert result["token_usage"]["total_tokens"] == 14
            assert result["make_changes_calls"] == _STATS["make_changes_calls"]
    assert seen[-1] == "gateway-close"
    assert not state.open and not state.cwd.exists()
    assert state.event_log.is_file()
    assert (tmp_path / f"{driver}_final.txt").read_text() == "done"


#: Codex's refusal of fontaine run 386d9522's first ``make_changes`` (codex-cli 0.154.0,
#: 2026-10-26), as its ``codex_events.jsonl`` recorded it, with ``arguments`` cut to one statement.
_CODEX_REFUSAL = {
    "type": "item.completed",
    "item": {
        "id": "item_20",
        "type": "mcp_tool_call",
        "server": "gemot",
        "tool": "make_changes",
        "arguments": {
            "statements": [
                {
                    "prefix": "n1",
                    "statement": {
                        "type": "PopulationTransition",
                        "source": {"name": "NMP"},
                        "target": {"name": "Early"},
                    },
                }
            ]
        },
        "result": None,
        "error": {"message": "MCP tool call requires approval, but approval policy is never"},
        "status": "failed",
    },
}


def _codex_call(**item):
    return {**_CODEX_REFUSAL, "item": {**_CODEX_REFUSAL["item"], **item}}


@pytest.mark.parametrize(
    "driver, event, stops",
    [
        ("codex", _CODEX_REFUSAL, True),
        # The wording is Codex's to change; a refusal it renames must stop the run all the same.
        ("codex", _codex_call(error={"message": "MCP tool call blocked by a hook"}), True),
        # Not refusals: a call Codex sent that went unanswered, and a tool error the server returned.
        ("codex", _codex_call(error={"message": "tool call error: timed out"}), False),
        (
            "codex",
            _codex_call(error=None, result={"content": [], "isError": True}),
            False,
        ),
        # Claude records its denials on the session's result (the shape Claude Code 2.1.302 emits).
        (
            "claude",
            {
                **_events("claude")[-1],
                "permission_denials": [
                    {
                        "tool_name": "mcp__gemot__make_changes",
                        "tool_use_id": "toolu_01",
                        "tool_input": {},
                    }
                ],
            },
            True,
        ),
    ],
)
def test_a_call_the_cli_refused_is_an_infrastructure_stop(
    monkeypatch, tmp_path, driver, event, stops
):
    """gemot#69: Codex refused every ``make_changes`` of seven fontaine runs; they committed no
    model and were recorded ``completed``. A refusal now ends the session at once as an
    infrastructure stop, which the runner does not judge and the analysis keeps out of every
    success rate."""
    events = _events(driver)
    events = [*events[:1], event, *events[1:]] if driver == "codex" else [*events[:-1], event]
    session, _, state, seen = _stub_boundaries(monkeypatch, driver, events=events)
    result = asyncio.run(
        N.run_native_cli(
            session,
            driver=driver,
            model="raw-native-model",
            task=_TASK,
            work=tmp_path,
            executable="native-executable",
            reasoning_effort="high",
            tool_timeout=1234.0,
        )
    )
    assert result["stop_reason"] == (
        infrastructure.stop_reason(infrastructure.CLI_REFUSED) if stops else "completed"
    )
    assert seen[-1] == "gateway-close" and not state.cwd.exists()
    if stops:
        assert infrastructure.is_stop(result["stop_reason"])
        assert "make_changes" in result["infrastructure_evidence"]
        assert result["cli_error"] == result["infrastructure_evidence"]
        # Nothing after the refusal is observed: the session ends there.
        assert result["native_event_count"] == events.index(event) + 1
    else:
        assert result["infrastructure_evidence"] is None


def test_whole_session_deadline_includes_gateway_startup(monkeypatch, tmp_path):
    state = SimpleNamespace(closed=False)

    @asynccontextmanager
    async def stalled_gateway(*args, **kwargs):
        try:
            await asyncio.Event().wait()
            yield
        finally:
            state.closed = True

    monkeypatch.setattr(cli_mcp, "serve_cli_tools", stalled_gateway)
    result = asyncio.run(
        N.run_native_cli(
            object(),
            driver="codex",
            model="native",
            task=_TASK,
            work=tmp_path,
            executable="unused",
            timeout=0.01,
        )
    )
    assert state.closed
    assert result["stop_reason"] == "wall_time_budget_exceeded"
    assert result["native_event_count"] == 0 and result["native_tool_names"] == []
    assert (tmp_path / result["native_event_log"]).is_file()


@pytest.mark.parametrize("driver", ["codex", "claude"])
def test_real_native_subprocess_calls_real_mcp_gateway(tmp_path, driver):
    """Run the complete bridge locally; the only fake is the model's executable."""
    from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

    executable = tmp_path / "fake-cli"
    program = tmp_path / "fake_cli.py"
    program.write_text(
        """import asyncio, json, os, sys
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

args = sys.argv[1:]
if args[0] == 'exec':
    setting = next(arg for arg in args if arg.startswith('mcp_servers.gemot.url='))
    url = json.loads(setting.split('=', 1)[1])
else:
    config = json.loads(args[args.index('--mcp-config') + 1])
    url = config['mcpServers']['gemot']['url']
prompt = sys.stdin.read()
assert os.listdir() == []

async def run():
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            assert [tool.name for tool in tools] == ['make_changes']
            result = await session.call_tool('make_changes', {
                'justification': 'measured rate', 'prompt': prompt,
                'cwd': os.getcwd(), 'gateway': url,
            })
            assert result.structuredContent == {'applied': True, 'version': 1}
asyncio.run(run())
"""
        + f"\nfor event in {repr(_events(driver))}:\n    print(json.dumps(event), flush=True)\n"
    )
    executable.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(program))
        + ' "$@"\n'
    )
    executable.chmod(0o700)
    calls = []

    async def list_tools():
        return ListToolsResult(
            tools=[
                Tool(name="make_changes", inputSchema={"type": "object"}),
                Tool(name="evaluate_validation_data", inputSchema={"type": "object"}),
            ]
        )

    async def call_tool(name, arguments):
        calls.append((name, arguments))
        assert Path(arguments["cwd"]).is_dir()
        payload = {"applied": True, "version": 1}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(payload))],
            structuredContent=payload,
        )

    async def run():
        import httpx

        work = tmp_path / "work"
        result = await N.run_native_cli(
            SimpleNamespace(list_tools=list_tools, call_tool=call_tool),
            driver=driver,
            model="native-model",
            task=_TASK,
            work=work,
            executable=str(executable),
            timeout=20,
        )
        assert result["stop_reason"] == "completed" and result["cli_returncode"] == 0
        assert result["native_session_id"] == "native-session"
        assert result["native_final_text"] == "done"
        assert result["native_tool_calls"] == 1
        assert result["make_changes_calls"]["by_outcome"] == {"ok": 1}
        assert result["token_usage"]["total_tokens"] == 14
        assert len(calls) == 1 and calls[0][0] == "make_changes"
        args = calls[0][1]
        assert args["prompt"] == prompts.NATIVE_CLI_INSTRUCTIONS + _TASK
        assert not Path(args["cwd"]).exists()
        assert [
            json.loads(line)
            for line in (work / result["native_event_log"]).read_text().splitlines()
        ] == _events(driver)
        assert (work / result["native_prompt_file"]).read_text() == args["prompt"]
        assert (work / result["native_final_file"]).read_text() == "done"
        async with httpx.AsyncClient() as http:
            with pytest.raises(httpx.ConnectError):
                await http.get(args["gateway"])

    asyncio.run(run())


def _native(session, driver, work, **kwargs):
    return asyncio.run(
        N.run_native_cli(
            session,
            driver=driver,
            model="raw-native-model",
            task=_TASK,
            work=work,
            executable="native-executable",
            reasoning_effort="high",
            tool_timeout=1234.0,
            **kwargs,
        )
    )


def _claude_response(message_id, *content):
    """One streamed Claude response: 100 prompt tokens at its start, 50 output by its end."""
    usage = {"input_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    return [
        {
            "type": "stream_event",
            "event": {
                "type": "message_start",
                "message": {"id": message_id, "usage": {**usage, "output_tokens": 1}},
            },
        },
        {
            "type": "stream_event",
            "event": {"type": "message_delta", "usage": {"output_tokens": 50}},
        },
        {"type": "assistant", "message": {"id": message_id, "content": list(content)}},
    ]


def _codex_tool_call(call_id):
    return {
        "type": "item.completed",
        "item": {
            "id": call_id,
            "type": "mcp_tool_call",
            "server": "gemot",
            "tool": "get_fit_summary",
            "arguments": {},
            "result": {"content": [], "isError": False},
            "error": None,
            "status": "completed",
        },
    }


#: Claude: the second response starts with 150 + 101 tokens reported. Codex: its second gateway
#: call returns with two calls' worth of budget counted.
_BUDGET_RUNS = {
    "claude": (
        [
            _events("claude")[0],
            *_claude_response("m1", {"type": "tool_use", "id": "call1", "name": "x", "input": {}}),
            {
                "type": "user",
                "message": {
                    "content": [{"type": "tool_result", "tool_use_id": "call1", "content": "ok"}]
                },
            },
            *_claude_response("m2", {"type": "text", "text": "done"}),
            _events("claude")[-1],
        ],
        5,
        251,
    ),
    "codex": (
        [
            _events("codex")[0],
            _codex_tool_call("call1"),
            _codex_tool_call("call2"),
            *_events("codex")[1:],
        ],
        2,
        2 * CODEX_TOKENS_PER_TOOL_CALL,
    ),
}


@pytest.mark.parametrize("driver", ["codex", "claude"])
@pytest.mark.parametrize("budget", ["spent", "left", "none"])
def test_run_token_budget_stops_the_cli_between_model_calls(monkeypatch, tmp_path, driver, budget):
    """gemot#75: the native driver enforced only a wall-clock timeout, and three perica_egf Codex
    runs finished at 29.6-43.0M tokens against the panel's 20M ``max_run_tokens``. The budget now
    ends the process where it is spent, at the start of a model call, and nothing after it runs."""
    events, trip, spent = _BUDGET_RUNS[driver]
    session, _, state, seen = _stub_boundaries(monkeypatch, driver, events=events)
    result = _native(
        session,
        driver,
        tmp_path,
        max_run_tokens={"spent": spent, "left": spent + 1, "none": None}[budget],
    )
    assert seen[-1] == "gateway-close" and not state.cwd.exists()
    if budget != "spent":
        assert result["stop_reason"] == "completed"
        assert result["native_event_count"] == len(events)
        return
    assert result["stop_reason"] == "token_budget_exceeded"
    assert result["cli_error"] is None and result["cli_returncode"] is None
    assert result["native_event_count"] == trip + 1
    usage = result["token_usage"]
    if driver == "claude":
        # The responses' own counts, reported as they streamed: the session total never came.
        assert usage["usage_source"] == "partial"
        assert usage["total_tokens"] == spent and usage["n_llm_calls"] == 2
    else:
        # Codex reports usage only when its turn completes, and a stopped turn never does.
        assert usage["usage_source"] == "unavailable" and usage["total_tokens"] is None


#: What three perica_egf Codex runs (sweeps 61679135/61679136, gemot 2db3d16) ended with after
#: ~6 h, and the stderr warning that was recorded as their ``cli_error`` instead.
_OUT_OF_CREDITS = "Your workspace is out of credits. Add credits to continue."
_STDERR = "WARNING: failed to clean up stale arg0 temp dirs: Not a directory (os error 20)"
_STDERR_EXIT = f"Native CLI exited with status 1: {_STDERR}"


@pytest.mark.parametrize("reported", [True, False])
def test_failed_codex_turn_records_the_providers_error_and_no_usage(
    monkeypatch, tmp_path, reported
):
    """gemot#75: a failed turn recorded the stderr warning as its ``cli_error`` and all-``None``
    usage that read like a run that spent nothing. The provider's error is the cause; stderr is
    the fallback only when the event stream gave none."""
    events = [_events("codex")[0], _codex_tool_call("call1")]
    if reported:
        events += [
            {"type": "error", "message": _OUT_OF_CREDITS},
            {"type": "turn.failed", "error": {"message": _OUT_OF_CREDITS}},
        ]
    session, _, _, seen = _stub_boundaries(
        monkeypatch,
        "codex",
        events=events,
        process_error=CLIProcessError(_STDERR_EXIT, returncode=1, stderr=_STDERR),
    )
    result = _native(session, "codex", tmp_path)
    assert seen[-1] == "gateway-close"
    assert result["stop_reason"] == "cli_error" and result["cli_returncode"] == 1
    assert result["cli_error"] == (_OUT_OF_CREDITS if reported else _STDERR_EXIT)
    assert result["token_usage"]["usage_source"] == "unavailable"
    assert result["token_usage"]["prompt_tokens"] is None


def test_crashed_claude_session_keeps_the_usage_its_responses_reported(monkeypatch, tmp_path):
    """gemot#75: usage came only from the session's final event, so a crashed run recorded none.
    Claude's responses report their own, which are kept and marked a lower bound."""
    events = [
        _events("claude")[0],
        *_claude_response("m1", {"type": "tool_use", "id": "call1", "name": "x", "input": {}}),
    ]
    session, _, _, _ = _stub_boundaries(
        monkeypatch,
        "claude",
        events=events,
        process_error=CLIProcessError("Native CLI exited with status 1: ", returncode=1),
    )
    usage = _native(session, "claude", tmp_path)["token_usage"]
    assert usage["usage_source"] == "partial"
    assert usage["prompt_tokens"] == 100 and usage["completion_tokens"] == 50
    assert usage["total_tokens"] == 150 and usage["n_llm_calls"] == 1
