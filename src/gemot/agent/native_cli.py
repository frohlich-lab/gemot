"""Run one native coding-agent session against the scored PEtab tool surface.

The task is sent once. Codex or Claude owns the conversation, tool choices, context management,
and prompt caching; gemot supplies neither a panel nor a replacement model-call protocol.
The MCP gateway and native event recorder are independent observers. They preserve the same
server editing boundary, run trace and independent judging used by the other drivers.

An empty temporary working directory keeps the benchmark's reference files out of the CLI's
project context. Native JSONL and the initial/final text live in ``work/`` for archival, outside
that directory. Process errors retain partial accounting so the runner can judge committed
work; cancellation propagates after the process, gateway and spans have been closed. A tool
call the CLI refused itself ends the session at once as an infrastructure stop
(:data:`gemot.infrastructure.CLI_REFUSED`), which the runner does not judge.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
import tempfile
from typing import Any


async def run_native_cli(
    session: Any,
    *,
    driver: str,
    model: str,
    task: str,
    work: Path,
    executable: str,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    tracing: Any = None,
    parent_span: Any = None,
    biology_blinded: bool = False,
    tool_timeout: float = 600.0,
) -> dict:
    """Drive the whole task once and return native observations for the existing judge.

    ``timeout`` bounds the complete session. The runner supplies the remaining wall budget,
    where applicable; MCP fitting timeouts are separate and can be much longer than ordinary
    HTTP defaults. Success requires the CLI's terminal success event, not just exit status zero.
    """
    from .. import infrastructure, prompts
    from .claude import claude_command, claude_refusal
    from .cli_mcp import serve_cli_tools
    from .cli_process import CLIProcessError, diagnostic, run_cli
    from .cli_tracing import NativeCLIRecorder
    from .codex import codex_command, codex_refusal

    builders = {"codex": codex_command, "claude": claude_command}
    refusals = {"codex": codex_refusal, "claude": claude_refusal}
    if driver not in builders:
        raise ValueError(f"Unknown native CLI driver: {driver!r}")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("CLI run timeout must be a finite positive number or None.")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    prompt = prompts.NATIVE_CLI_INSTRUCTIONS + task
    prompt_path = work / f"{driver}_prompt.txt"
    event_path = work / f"{driver}_events.jsonl"
    final_path = work / f"{driver}_final.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    event_path.write_bytes(b"")
    recorder = NativeCLIRecorder(driver, model, tracing=tracing, parent_span=parent_span)
    recorder.start(prompt)
    event_count, terminal_success = 0, False
    error = evidence = None
    stop_reason = "completed"
    returncode = None
    gateway = None

    def observe(event: dict) -> None:
        nonlocal event_count, terminal_success
        event_count += 1
        recorder.event(event)
        if driver == "codex":
            if event.get("type") in {"turn.completed", "turn.failed"}:
                terminal_success = event["type"] == "turn.completed"
        elif event.get("type") == "result":
            terminal_success = event.get("subtype") == "success" and not event.get("is_error")
        # The CLI tells the model why it refused, and the model writes a report on what it could
        # not do: a run that "completes" with nothing committed (gemot#69). Raised here, it ends
        # the process, as an infrastructure failure ends a deepagents run.
        if (refusal := refusals[driver](event)) is not None:
            raise infrastructure.InfrastructureError(
                infrastructure.Evidence(infrastructure.CLI_REFUSED, refusal)
            )

    try:
        # Include tools/list and gateway startup: a server stalled before the first model
        # request must not evade the session deadline. Process cleanup still runs on expiry.
        async with asyncio.timeout(timeout) as watchdog:
            async with serve_cli_tools(
                session, tracing=tracing, biology_blinded=biology_blinded
            ) as gateway:
                command = builders[driver](
                    executable=executable,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    mcp_url=gateway.url,
                    tool_timeout=tool_timeout,
                )
                with tempfile.TemporaryDirectory(prefix=f"gemot-{driver}-") as cwd:
                    result = await run_cli(
                        command,
                        prompt=prompt,
                        cwd=cwd,
                        event_log=event_path,
                        timeout=None,
                        on_event=observe,
                    )
                returncode = result.returncode
                if recorder.error or not terminal_success:
                    error = diagnostic(
                        recorder.error or "Native CLI ended without a success event."
                    )
                    stop_reason = "cli_error"
                elif gateway.stats["native_tool_calls"] == 0:
                    stop_reason = "no_tools_executed"
    except TimeoutError:
        if not watchdog.expired():
            error = "Native CLI gateway or process timed out."
            raise
        error = f"Native CLI session exceeded its {timeout:g}s timeout."
        stop_reason = "wall_time_budget_exceeded"
    except CLIProcessError as exc:
        error = diagnostic(str(exc))
        returncode = exc.returncode
        stop_reason = (
            "wall_time_budget_exceeded" if isinstance(exc.__cause__, TimeoutError) else "cli_error"
        )
    except infrastructure.InfrastructureError as exc:
        error = evidence = diagnostic(exc.evidence.text)
        stop_reason = infrastructure.stop_reason(exc.evidence.cause)
    except asyncio.CancelledError:
        error = "Native CLI session cancelled."
        raise
    except BaseException as exc:
        error = diagnostic(str(exc))
        raise
    finally:
        recorder.finish(error=error)
        final_path.write_text(recorder.final_text, encoding="utf-8")

    return {
        "stop_reason": stop_reason,
        "token_usage": recorder.usage(),
        "native_session_id": recorder.session_id,
        "native_final_text": recorder.final_text,
        "native_prompt_file": prompt_path.name,
        "native_event_log": event_path.name,
        "native_final_file": final_path.name,
        "native_event_count": event_count,
        "native_tool_names": list(gateway.tool_names) if gateway else [],
        "native_trace_diagnostics": recorder.diagnostics,
        "cli_returncode": returncode,
        "cli_error": error,
        "infrastructure_evidence": evidence,
        **(gateway.stats if gateway else {}),
    }
