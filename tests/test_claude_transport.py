"""Verify native Claude CLI configuration without opening an authenticated session.

The gateway must survive benchmark isolation and keep enough time for scientific fits.
A print-mode task uses native MCP and message events, with no per-turn output schema.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from gemot.agent import claude as C


_BASE_FAKE = """#!{python}
import sys
if sys.argv[1:] == ["--version"]:
    print("2.1.302 (Claude Code)")
else:
    print({flags!r})
"""


def _fake(tmp_path, *, missing=None):
    path = tmp_path / "fake-claude"
    flags = " ".join(flag for flag in C._REQUIRED_FLAGS if flag != missing)
    path.write_text(_BASE_FAKE.format(python=sys.executable, flags=flags))
    path.chmod(0o755)
    return str(path)


def test_native_claude_gateway_configuration(tmp_path):
    executable = _fake(tmp_path)
    assert C.check_claude_cli(executable) == "2.1.302 (Claude Code)"
    command = C.claude_command(
        executable=executable,
        model="sonnet",
        reasoning_effort="high",
        mcp_url="http://localhost:1234/mcp",
        tool_timeout=30000,
    )
    config = json.loads(command[command.index("--mcp-config") + 1])
    assert config == {
        "mcpServers": {
            "gemot": {
                "type": "http",
                "url": "http://localhost:1234/mcp",
                "timeout": 30000000,
                "alwaysLoad": True,
            }
        }
    }
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--allowedTools") + 1] == "mcp__gemot__*"
    assert command[command.index("--effort") + 1] == "high"
    assert "--include-partial-messages" in command and "--strict-mcp-config" in command
    assert "--restricted" in command
    assert "--safe-mode" not in command  # safe mode disables the required MCP gateway
    assert "--json-schema" not in command and "--system-prompt" not in command
    assert "--dangerously-skip-permissions" not in command


def test_missing_capability_fails_before_session(tmp_path):
    executable = _fake(tmp_path, missing="--include-partial-messages")
    with pytest.raises(C.ClaudeCLIError, match="include-partial-messages"):
        C.check_claude_cli(executable)


def test_missing_executable_is_actionable(tmp_path):
    with pytest.raises(C.ClaudeCLIError, match="claude auth login"):
        C.check_claude_cli(str(tmp_path / "missing"))


@pytest.mark.parametrize("effort", ["minimal", "none", "ultra"])
def test_invalid_claude_effort_fails_before_launch(tmp_path, effort):
    executable = _fake(tmp_path)
    with pytest.raises(ValueError, match="reasoning effort"):
        C.claude_command(
            executable=executable,
            model="sonnet",
            reasoning_effort=effort,
            mcp_url="http://localhost/mcp",
            tool_timeout=300,
        )


def test_native_modules_import_without_langchain():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gemot.agent.claude; "
            "import gemot.agent.codex; import gemot.agent.cli_process; "
            "assert not any(x == 'langchain_core' or x.startswith('langchain_core.') for x in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
