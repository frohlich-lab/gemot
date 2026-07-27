"""CI guard: the curated biocurator tool names must still resolve on the live bio MCP servers.

The biocurator (petab_harness.agent) is handed a CURATED allowlist of tool names
(DEFAULT_BIO_TOOLS + LITERATURE_BIO_TOOLS) resolved against the hardcoded bio MCP endpoints
(runner.endpoints.BIO_MCP_URLS: BioContextAI + OmniPath). There is no all-tools fallback, so if an upstream
server RENAMES or drops a tool the curated names silently stop matching and the biocurator runs
"dumb" (tool-less) on every run. This test connects to the live servers and asserts every curated
name is still exposed, so a rename is caught in CI rather than silently degrading the benchmark.

It SKIPS (never fails) when ``mcp`` isn't installed or the servers are unreachable/slow, so network
flakiness can't break CI; it only FAILS when it connects to every server and a curated name is
missing.
"""

from __future__ import annotations

import asyncio

import pytest

_CONNECT_TIMEOUT_S = 30.0


def _load_agent():
    """The agent package, for its curated tool-name sets. Import-light: `petab_harness.__init__`
    is lazy and the agent submodules only touch stdlib at import time."""
    from petab_harness import agent

    return agent


def _bio_urls() -> "list[str]":
    """The bio MCP endpoint URLs the biocurator is pointed at.

    ``runner.endpoints`` holds nothing but these constants and imports nothing, so this is a real
    import rather than the regex-over-runner.py-source it used to be — which would go silently
    empty, and so vacuously green, the moment that file was reorganised."""
    from petab_harness.runner.endpoints import BIO_MCP_URLS

    return list(BIO_MCP_URLS)


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
    assert urls, "no bio MCP URLs found (constant renamed?)"

    # No `mcp` availability probe: it is a hard dependency (pyproject [project.dependencies]),
    # so the skip it guarded could never fire.
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
