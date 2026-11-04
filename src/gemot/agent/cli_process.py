"""Own one native CLI process for an entire modelling run.

The CLI receives the task once and owns its conversation, tool loop, caching, compaction,
and internal retries. This module only supplies stdin, archives native JSONL events, forwards
those events to tracing, and reaps the process on exit or cancellation. It never constructs
model requests, interprets tool calls, reads credentials, or restarts a failed session.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from typing import Any


#: Native events can contain full MCP results; bound a single line without truncating it.
_EVENT_LINE_LIMIT = 16 * 1024 * 1024


class CLIProcessError(RuntimeError):
    """A native CLI startup, process, or event-stream failure."""

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class CLIProcessResult:
    """Process-level outcome; native event consumers own usage and final-answer extraction."""

    returncode: int
    stderr: str
    event_count: int


def cli_environment() -> dict[str, str]:
    """Leave native CLI authentication intact while excluding OpenRouter credentials."""
    return {key: value for key, value in os.environ.items() if not key.startswith("OPENROUTER_")}


def diagnostic(value: str) -> str:
    """Bound errors and redact secret environment values if the executable echoes them."""
    for name, secret in os.environ.items():
        if len(secret) >= 8 and any(
            marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            value = value.replace(secret, "[redacted]")
    return value.strip()[-2000:]


def executable_path(executable: str, *, label: str, login: str) -> str:
    """Resolve once before switching to an empty working directory."""
    path = shutil.which(executable)
    if path is None:
        raise CLIProcessError(
            f"{label} executable {executable!r} was not found. Install the official CLI, "
            f"authenticate with `{login}`, or select its executable path."
        )
    return str(Path(path).absolute())


def read_capability(command: list[str]) -> str:
    """Run a bounded help/version command, without starting an authenticated model session."""
    with tempfile.TemporaryDirectory(prefix="gemot-cli-check-") as cwd:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=cli_environment(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise CLIProcessError(f"CLI capability check failed: {exc}") from exc
        if result.returncode:
            raise CLIProcessError(
                "CLI capability check failed; update the official CLI. "
                + diagnostic(result.stderr or result.stdout),
                returncode=result.returncode,
            )
        return result.stdout


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.returncode is None:
            process.kill()
    except ProcessLookupError:
        pass
    await process.wait()


async def _write_prompt(stream: asyncio.StreamWriter, prompt: str) -> None:
    try:
        stream.write(prompt.encode("utf-8"))
        await stream.drain()
    except (BrokenPipeError, ConnectionResetError):
        # The process's exit status and stderr explain startup failures more usefully.
        pass
    finally:
        stream.close()


async def _read_stderr(stream: asyncio.StreamReader) -> str:
    tail = b""
    while chunk := await stream.read(8192):
        tail = (tail + chunk)[-8192:]
    return tail.decode("utf-8", errors="replace")


async def run_cli(
    command: list[str],
    *,
    prompt: str,
    cwd: str | Path,
    event_log: str | Path,
    timeout: float | None,
    on_event: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
) -> CLIProcessResult:
    """Send one task to one CLI and stream its native events until the whole session ends.

    ``event_log`` retains events even when parsing or the process fails. ``on_event`` observes
    each parsed event in order and may be synchronous or asynchronous. Cancellation propagates
    after killing and reaping the process group. A timeout bounds the entire run, not each
    native model turn or MCP tool call. No process or authentication failure is retried here.
    """
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("CLI run timeout must be a finite positive number or None.")
    if not command:
        raise ValueError("CLI command must not be empty.")
    log_path = Path(event_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as archive:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=cli_environment(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
                limit=_EVENT_LINE_LIMIT,
            )
        except OSError as exc:
            raise CLIProcessError(f"Could not start native CLI: {exc}") from exc

        async def read_events() -> int:
            count = 0
            while True:
                try:
                    line = await process.stdout.readline()
                except ValueError as exc:
                    raise CLIProcessError(
                        f"Native CLI JSONL event exceeds the {_EVENT_LINE_LIMIT}-byte limit; "
                        "earlier events remain in its event log."
                    ) from exc
                if not line:
                    break
                archive.write(line)
                archive.flush()
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
                    raise CLIProcessError(
                        "Native CLI emitted malformed JSONL; see its event log."
                    ) from exc
                if not isinstance(event, dict):
                    raise CLIProcessError("Native CLI emitted a non-object JSONL event.")
                count += 1
                if on_event is not None:
                    result = on_event(event)
                    if inspect.isawaitable(result):
                        await result
            return count

        tasks = [
            asyncio.create_task(read_events()),
            asyncio.create_task(_read_stderr(process.stderr)),
            asyncio.create_task(_write_prompt(process.stdin, prompt)),
        ]
        try:
            async with asyncio.timeout(timeout) as watchdog:
                count, stderr, _ = await asyncio.gather(*tasks)
                await process.wait()
            stderr = diagnostic(stderr)
            if process.returncode:
                raise CLIProcessError(
                    f"Native CLI exited with status {process.returncode}: {stderr}",
                    returncode=process.returncode,
                    stderr=stderr,
                )
            if not count:
                raise CLIProcessError("Native CLI exited without JSONL events.", stderr=stderr)
            return CLIProcessResult(returncode=process.returncode, stderr=stderr, event_count=count)
        except TimeoutError as exc:
            if not watchdog.expired():
                raise
            raise CLIProcessError(f"Native CLI session exceeded its {timeout:g}s timeout.") from exc
        finally:
            await _kill_process(process)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
