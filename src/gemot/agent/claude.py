"""Configure one native Claude Code print session for a complete gemot run.

Claude owns the conversation, prompt caching, compaction and MCP tool loop. This module checks
its CLI capabilities, connects the run's gemot gateway, recognizes the calls Claude denied
(:func:`claude_refusal`) and where its run budget is checked (:func:`claude_response_started`),
without a LangChain chat-model adapter or structured-output shim.
Authentication stays with the installed official CLI. Native JSON events, including partial messages, are observed by gemot's tracing layer.
"""

from __future__ import annotations

import json
import math

from .cli_process import CLIProcessError, executable_path, read_capability


class ClaudeCLIError(CLIProcessError):
    """The installed Claude Code CLI cannot run the requested native session."""


_REQUIRED_FLAGS = (
    "--print",
    "--output-format",
    "--verbose",
    "--include-partial-messages",
    "--tools",
    "--strict-mcp-config",
    "--mcp-config",
    "--restricted",
    "--no-session-persistence",
    "--setting-sources",
    "--model",
    "--effort",
    "--allowedTools",
)
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def check_claude_cli(executable: str = "claude") -> str:
    """Check installed capabilities without making a model request or accessing credentials."""
    try:
        path = executable_path(executable, label="Claude Code CLI", login="claude auth login")
        version = read_capability([path, "--version"]).strip()
        help_text = read_capability([path, "--help"])
        missing = [flag for flag in _REQUIRED_FLAGS if flag not in help_text]
        if missing:
            raise ClaudeCLIError(
                f"Claude CLI {version} lacks required options: {', '.join(missing)}. Update the official CLI."
            )
        return version
    except CLIProcessError as exc:
        if isinstance(exc, ClaudeCLIError):
            raise
        raise ClaudeCLIError(str(exc), returncode=exc.returncode, stderr=exc.stderr) from exc


def claude_command(
    *,
    executable: str = "claude",
    model: str,
    reasoning_effort: str | None = None,
    mcp_url: str,
    tool_timeout: float,
) -> list[str]:
    """Build one native ``-p`` session against the authorized gemot MCP gateway.

    Built-in tools are disabled for benchmark blindness; explicit MCP tools are preserved.
    The per-server timeout is in milliseconds and also raises Claude's HTTP request and idle
    limits, which would otherwise interrupt long scientific fits before gemot's own watchdog.
    """
    if not model.strip():
        raise ValueError("A nonempty native Claude model ID is required.")
    if reasoning_effort is not None and reasoning_effort not in _EFFORTS:
        raise ValueError("Claude reasoning effort must be low, medium, high, xhigh, or max.")
    if not math.isfinite(tool_timeout) or tool_timeout <= 0:
        raise ValueError("MCP tool timeout must be finite and positive.")
    path = executable_path(executable, label="Claude Code CLI", login="claude auth login")
    gateway = {
        "mcpServers": {
            "gemot": {
                "type": "http",
                "url": mcp_url,
                "timeout": math.ceil(tool_timeout * 1000),
                # ToolSearch is a built-in and is excluded by --tools "". Keep the actual
                # gateway tools visible from the first prompt instead of deferring them.
                "alwaysLoad": True,
            }
        }
    }
    command = [
        path,
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--tools",
        "",
        "--allowedTools",
        "mcp__gemot__*",
        "--strict-mcp-config",
        "--mcp-config",
        json.dumps(gateway),
        "--restricted",
        "--no-session-persistence",
        "--setting-sources",
        "",
        "--model",
        model,
    ]
    if reasoning_effort is not None:
        command.extend(["--effort", reasoning_effort])
    return command


def claude_refusal(event: dict) -> str | None:
    """Which calls Claude denied, if ``event`` is the session's result and records any.

    ``permission_denials`` is the CLI's own record of the calls it did not run. ``--allowedTools``
    grants every gateway tool, so a denial is that grant failing, never the model's doing.
    """
    denials = event.get("permission_denials") if event.get("type") == "result" else None
    if not denials or not isinstance(denials, list):
        return None
    tools = sorted({str(d.get("tool_name")) for d in denials if isinstance(d, dict)})
    return f"Claude denied {len(denials)} call(s) to {', '.join(tools) or 'unnamed tools'}"


def claude_response_started(event: dict) -> bool:
    """Whether ``event`` opens a model response, the point to check Claude's run budget.

    ``--include-partial-messages`` streams each response's usage as it starts, so the budget is
    counted in reported tokens. Every tool call the previous response asked for has returned by
    then, so stopping here interrupts no fit and no commit.
    """
    inner = event.get("event")
    return (
        event.get("type") == "stream_event"
        and isinstance(inner, dict)
        and inner.get("type") == "message_start"
    )
