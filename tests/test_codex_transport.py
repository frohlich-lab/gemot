"""Native Codex command construction and the one-process session boundary.

Real fake-executable processes verify that the task is sent once, native MCP events retain
order, logs survive failure, and timeout/cancellation do not leave processes running.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys

import pytest

from gemot.agent import codex as C
from gemot.agent import cli_process as P


_BASE_FAKE = r"""#!{python}
import json
import os
from pathlib import Path
import sys
import time
args = sys.argv[1:]
if args == ["--version"]:
    print("codex-cli test-1.0")
    sys.exit(0)
if args == ["exec", "--help"]:
    print("--ignore-user-config --ephemeral --skip-git-repo-check --json --sandbox --model")
    sys.exit(0)
if args == ["features", "list"]:
    print("shell_tool stable true")
    sys.exit(0)
record = Path(os.environ["GEMOT_TEST_CLI_RECORD"])
record.with_suffix(".pid").write_text(str(os.getpid()))
mode = os.environ.get("GEMOT_TEST_CLI_MODE", "success")
if mode == "startup-error":
    print("Authenticate with codex login", file=sys.stderr)
    sys.exit(17)
if mode == "hang":
    time.sleep(30)
if mode == "blocked-input":
    print(json.dumps({"type":"abort"}), flush=True)
    time.sleep(30)
prompt = sys.stdin.read()
record.write_text(json.dumps({"prompt":prompt,"args":args,"has_openrouter":"OPENROUTER_API_KEY" in os.environ,
                             "codex_home":os.environ.get("CODEX_HOME")}))
if mode == "malformed":
    print("not JSON")
    sys.exit(0)
if mode == "oversized":
    print(json.dumps({"type":"start"}), flush=True)
    print(json.dumps({"type":"large","content":"x" * 1000}), flush=True)
    sys.exit(0)
for event in [
    {"type":"thread.started","thread_id":"native-session"},
    {"type":"item.started","item":{"id":"mcp1","type":"mcp_tool_call","server":"gemot","tool":"inspect_model"}},
    {"type":"item.completed","item":{"id":"mcp1","type":"mcp_tool_call","result":{"value":3}}},
    {"type":"item.completed","item":{"id":"message1","type":"agent_message","text":"done"}},
    {"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":80,"output_tokens":12}}
]:
    print(json.dumps(event), flush=True)
"""


def _fake(tmp_path, monkeypatch):
    executable = tmp_path / "fake-codex"
    executable.write_text(_BASE_FAKE.replace("{python}", sys.executable))
    executable.chmod(0o755)
    record = tmp_path / "request.json"
    monkeypatch.setenv("GEMOT_TEST_CLI_RECORD", str(record))
    return str(executable), record


def _assert_reaped(record: Path) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(int(record.with_suffix(".pid").read_text()), 0)


def test_native_command_preserves_mcp_and_has_no_model_adapter(tmp_path, monkeypatch):
    executable, record = _fake(tmp_path, monkeypatch)
    assert C.check_codex_cli(executable) == "codex-cli test-1.0"
    command = C.codex_command(
        executable=executable,
        model="native-model",
        reasoning_effort="high",
        mcp_url="http://127.0.0.1:1234/mcp",
        tool_timeout=30000,
    )
    assert command[:2] == [executable, "exec"]
    assert command[-1] == "-"
    assert 'mcp_servers.gemot.url="http://127.0.0.1:1234/mcp"' in command
    assert "mcp_servers.gemot.tool_timeout_sec=30000" in command
    assert "mcp_servers.gemot.required=true" in command
    assert 'mcp_servers.gemot.default_tools_approval_mode="approve"' in command
    assert 'model_reasoning_effort="high"' in command
    assert "features.shell_tool=false" in command
    assert "features.hooks=false" not in command
    assert "--output-schema" not in command and "--output-last-message" not in command
    assert not record.exists()  # capability check never starts a native agent session


@pytest.mark.asyncio
async def test_one_task_one_process_native_tool_events_and_archive(tmp_path, monkeypatch):
    executable, record = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-forward")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "original-cli-auth"))
    events = []
    command = C.codex_command(
        executable=executable, model="native", mcp_url="http://localhost/mcp", tool_timeout=3600
    )
    log = tmp_path / "native_events.jsonl"
    result = await P.run_cli(
        command,
        prompt="One modelling task",
        cwd=tmp_path,
        event_log=log,
        timeout=5,
        on_event=events.append,
    )
    assert result.returncode == 0 and result.event_count == 5
    assert [event["type"] for event in events] == [
        "thread.started",
        "item.started",
        "item.completed",
        "item.completed",
        "turn.completed",
    ]
    assert events[1]["item"]["type"] == "mcp_tool_call"
    assert [json.loads(line) for line in log.read_text().splitlines()] == events
    request = json.loads(record.read_text())
    assert request["prompt"] == "One modelling task"
    assert not request["has_openrouter"]
    assert request["codex_home"] == str(tmp_path / "original-cli-auth")
    _assert_reaped(record)


@pytest.mark.parametrize(
    "mode,match",
    [
        ("startup-error", "status 17: Authenticate"),
        ("malformed", "malformed JSONL"),
        ("hang", "timeout"),
    ],
)
@pytest.mark.asyncio
async def test_failure_keeps_log_and_reaps_process(tmp_path, monkeypatch, mode, match):
    executable, record = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("GEMOT_TEST_CLI_MODE", mode)
    log = tmp_path / "events.jsonl"
    with pytest.raises(P.CLIProcessError, match=match):
        await P.run_cli(
            [executable],
            prompt="task",
            cwd=tmp_path,
            event_log=log,
            timeout=0.25 if mode == "hang" else 5,
        )
    assert log.exists()
    if mode == "malformed":
        assert log.read_text() == "not JSON\n"
    _assert_reaped(record)


@pytest.mark.asyncio
async def test_callback_failure_interrupts_blocked_stdin(tmp_path, monkeypatch):
    executable, record = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("GEMOT_TEST_CLI_MODE", "blocked-input")

    async def stop(event):
        raise ValueError("gateway stopped")

    with pytest.raises(ValueError, match="gateway stopped"):
        await P.run_cli(
            [executable],
            prompt="x" * 1_000_000,
            cwd=tmp_path,
            event_log=tmp_path / "events.jsonl",
            timeout=2,
            on_event=stop,
        )
    _assert_reaped(record)


@pytest.mark.asyncio
async def test_large_event_fails_without_losing_earlier_events(tmp_path, monkeypatch):
    executable, record = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("GEMOT_TEST_CLI_MODE", "oversized")
    monkeypatch.setattr(P, "_EVENT_LINE_LIMIT", 128)
    log = tmp_path / "events.jsonl"
    with pytest.raises(P.CLIProcessError, match="128-byte limit"):
        await P.run_cli([executable], prompt="task", cwd=tmp_path, event_log=log, timeout=5)
    assert [json.loads(line) for line in log.read_text().splitlines()] == [{"type": "start"}]
    _assert_reaped(record)


@pytest.mark.asyncio
async def test_callback_timeout_is_not_mislabeled_as_cli_wall_timeout(tmp_path, monkeypatch):
    executable, record = _fake(tmp_path, monkeypatch)

    def callback(event):
        raise TimeoutError("trace collector unavailable")

    with pytest.raises(TimeoutError, match="trace collector unavailable"):
        await P.run_cli(
            [executable],
            prompt="task",
            cwd=tmp_path,
            event_log=tmp_path / "events.jsonl",
            timeout=None,
            on_event=callback,
        )
    _assert_reaped(record)


@pytest.mark.asyncio
async def test_cancellation_reaps_native_session(tmp_path, monkeypatch):
    executable, record = _fake(tmp_path, monkeypatch)
    monkeypatch.setenv("GEMOT_TEST_CLI_MODE", "hang")
    task = asyncio.create_task(
        P.run_cli(
            [executable],
            prompt="task",
            cwd=tmp_path,
            event_log=tmp_path / "events.jsonl",
            timeout=None,
        )
    )
    async with asyncio.timeout(5):
        while not record.with_suffix(".pid").exists():
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_reaped(record)


def test_missing_cli_is_actionable(tmp_path):
    with pytest.raises(C.CodexCLIError, match="codex login"):
        C.check_codex_cli(str(tmp_path / "missing"))
