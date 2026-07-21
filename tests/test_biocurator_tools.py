"""CI guard: the curated biocurator tool names must still resolve on the live bio MCP servers.

The biocurator (examples/_harness/agent.py) is handed a CURATED allowlist of tool names
(DEFAULT_BIO_TOOLS + LITERATURE_BIO_TOOLS) resolved against the hardcoded bio MCP endpoints
(runner.BIO_MCP_URLS: BioContextAI + OmniPath). There is no all-tools fallback, so if an upstream
server RENAMES or drops a tool the curated names silently stop matching and the biocurator runs
"dumb" (tool-less) on every run. This test connects to the live servers and asserts every curated
name is still exposed, so a rename is caught in CI rather than silently degrading the benchmark.

It SKIPS (never fails) when ``mcp`` isn't installed or the servers are unreachable/slow, so network
flakiness can't break CI; it only FAILS when it connects to every server and a curated name is
missing.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
from pathlib import Path

import pytest

_HARNESS = Path(__file__).resolve().parents[1] / "examples" / "_harness"
_CONNECT_TIMEOUT_S = 30.0


def _load_agent():
    """Load agent.py in isolation (by path) to read its curated tool-name sets without importing
    the heavy ``_harness`` package (mirrors tests/test_agent_tool_dispatch.py)."""
    spec = importlib.util.spec_from_file_location(
        "_biocurator_agent_under_test", _HARNESS / "agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bio_urls() -> "list[str]":
    """The bio MCP endpoint URLs, read from runner.py's constants so this stays in sync."""
    src = (_HARNESS / "runner.py").read_text()
    return re.findall(r'_(?:BIOCONTEXT|OMNIPATH)_MCP_URL\s*=\s*"([^"]+)"', src)


async def _discover_tool_names(urls: "list[str]") -> "set[str]":
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    names: "set[str]" = set()
    for url in urls:
        async def _one(u=url):
            async with streamablehttp_client(u) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    res = await session.list_tools()
                    return {t.name for t in res.tools}

        names |= await asyncio.wait_for(_one(), timeout=_CONNECT_TIMEOUT_S)
    return names


def test_curated_biocurator_tool_names_resolve_on_live_servers():
    agent = _load_agent()
    curated = set(agent.DEFAULT_BIO_TOOLS) | set(agent.LITERATURE_BIO_TOOLS)
    assert curated, "no curated biocurator tool names defined"

    urls = _bio_urls()
    assert urls, "no bio MCP URLs found in runner.py (constant renamed?)"

    try:
        import mcp  # noqa: F401
    except Exception as exc:  # mcp not installed in this env -> can't check
        pytest.skip(f"mcp not installed: {exc}")

    try:
        # All-or-nothing: if ANY server is unreachable/slow the union would be incomplete and give
        # a false 'missing', so treat any connect failure as 'can't check' and skip.
        discovered = asyncio.run(_discover_tool_names(urls))
    except Exception as exc:
        pytest.skip(f"bio MCP servers unreachable ({type(exc).__name__}: {exc})")

    missing = sorted(n for n in curated if n not in discovered)
    assert not missing, (
        "curated biocurator tool names are no longer exposed by the live bio servers "
        f"(upstream rename/removal?): {missing}. Update agent.DEFAULT_BIO_TOOLS / "
        "LITERATURE_BIO_TOOLS to match runner.BIO_MCP_URLS."
    )
