"""Configure one native Codex CLI session for a complete gemot run.

Codex owns its agent loop, history, prompt caching and MCP calls. This module only checks
its documented command-line capabilities, points it at the run's gemot MCP gateway, and
recognizes a gateway call Codex refused to send (:func:`codex_refusal`).
There is no LangChain model adapter, serialized transcript protocol, or per-turn subprocess.
Authentication stays in the official CLI and administrator-managed policy remains effective.
"""

from __future__ import annotations

import json
import math

from .cli_process import CLIProcessError, executable_path, read_capability


class CodexCLIError(CLIProcessError):
    """The installed Codex CLI cannot run the requested native session."""


#: Options that keep the native session reproducible and machine-readable.
_REQUIRED_EXEC_FLAGS = (
    "--ignore-user-config",
    "--ephemeral",
    "--skip-git-repo-check",
    "--json",
    "--sandbox",
    "--model",
)
_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}

#: How Codex reports an MCP call it sent that got no answer (``handle_approved_mcp_tool_call``
#: in ``codex-rs/core/src/mcp_tool_call.rs``). Every other call it completes with no result is
#: one it refused to send (``notify_mcp_tool_call_skip``). A tool error is a result: the gateway
#: returns every upstream failure as one.
_SENT_CALL_ERROR = "tool call error:"


def check_codex_cli(executable: str = "codex") -> str:
    """Check version and required options with read-only commands, never a model request."""
    try:
        path = executable_path(executable, label="Codex CLI", login="codex login")
        version = read_capability([path, "--version"]).strip()
        help_text = read_capability([path, "exec", "--help"])
        missing = [flag for flag in _REQUIRED_EXEC_FLAGS if flag not in help_text]
        if missing:
            raise CodexCLIError(
                f"Codex CLI {version} lacks required options: {', '.join(missing)}. Update the official CLI."
            )
        features = read_capability([path, "features", "list"])
        if "shell_tool" not in {line.split()[0] for line in features.splitlines() if line.split()}:
            raise CodexCLIError(
                "Codex CLI lacks the shell_tool feature control; update the official CLI."
            )
        return version
    except CLIProcessError as exc:
        if isinstance(exc, CodexCLIError):
            raise
        raise CodexCLIError(str(exc), returncode=exc.returncode, stderr=exc.stderr) from exc


def codex_command(
    *,
    executable: str = "codex",
    model: str,
    reasoning_effort: str | None = None,
    mcp_url: str,
    tool_timeout: float,
) -> list[str]:
    """Build one native ``exec`` command; the caller supplies its task once on stdin.

    File and web tools are disabled for benchmark blindness. The MCP gateway remains available
    for the full session, with a timeout large enough for the harness's longest fit. Native
    reasoning, context management and tool-call generation are left to Codex.
    """
    if not model.strip():
        raise ValueError("A nonempty native Codex model ID is required.")
    if reasoning_effort is not None and reasoning_effort not in _EFFORTS:
        raise ValueError(
            "Codex reasoning effort must be none, minimal, low, medium, high, or xhigh."
        )
    if not math.isfinite(tool_timeout) or tool_timeout <= 0:
        raise ValueError("MCP tool timeout must be finite and positive.")
    path = executable_path(executable, label="Codex CLI", login="codex login")
    command = [
        path,
        "exec",
        "--json",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--model",
        model,
        "--ephemeral",
    ]
    settings = [
        'approval_policy="never"',
        'web_search="disabled"',
        "features.shell_tool=false",
        "tools.view_image=false",
        "project_doc_max_bytes=0",
        "mcp_servers.gemot.url=" + json.dumps(mcp_url),
        "mcp_servers.gemot.required=true",
        f"mcp_servers.gemot.tool_timeout_sec={tool_timeout:g}",
        # Codex's default (`auto`) asks before a tool that writes, and under approval_policy
        # "never" an asked call is refused: `make_changes` failed with "MCP tool call requires
        # approval, but approval policy is never". The gateway serves only the scored tools.
        'mcp_servers.gemot.default_tools_approval_mode="approve"',
    ]
    if reasoning_effort is not None:
        settings.append("model_reasoning_effort=" + json.dumps(reasoning_effort))
    for setting in settings:
        command.extend(["-c", setting])
    command.append("-")
    return command


def codex_refusal(event: dict) -> str | None:
    """Which call Codex refused and why, if ``event`` completes one it never sent, else ``None``.

    Deliberately NOT a match on the one refusal seen so far ("MCP tool call requires approval,
    but approval policy is never"): Codex also refuses for a hook, an app policy or a tool it no
    longer lists, and its wording and the setting that avoids it are Codex's to rename. Any call
    completed with neither a result nor the sent-call error counts, so a renamed refusal still
    stops the run instead of leaving it ``completed`` with nothing committed.
    """
    item = event.get("item")
    if event.get("type") != "item.completed" or not isinstance(item, dict):
        return None
    if item.get("type") != "mcp_tool_call" or item.get("status") == "completed":
        return None
    if item.get("result") is not None:
        return None
    error = item.get("error")
    message = str((error.get("message") if isinstance(error, dict) else error) or "no reason given")
    if message.startswith(_SENT_CALL_ERROR):
        return None
    return f"Codex refused {item.get('server')}/{item.get('tool')}: {message}"
