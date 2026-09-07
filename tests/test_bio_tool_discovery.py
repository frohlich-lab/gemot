"""Bio-tool discovery: the curated allowlist, offline.

``tests/test_biocurator_tools.py`` asks the LIVE servers whether the curated names still exist.
This asks the selection logic itself — allowlist, ablation, name collision, duplicate, dead
server — with no network, so it runs on every CI job rather than skipping when a server is slow.
"""

from __future__ import annotations

import asyncio

import pytest


def test_discovery_is_the_curated_allowlist(monkeypatch):
    """The curated names resolve; a literature tool only under the ablation; a petab-name
    collision and a second server's duplicate are dropped; a server whose discovery raises costs
    only its own tools."""
    pytest.importorskip("langchain_mcp_adapters")
    from langchain_core.tools import StructuredTool
    import langchain_mcp_adapters.tools as lmt
    from petab_harness.agent import biocurator as B

    def _tool(name):
        return StructuredTool.from_function(func=(lambda: ""), name=name, description=name)

    curated = B.DEFAULT_BIO_TOOLS[0]
    literature = B.LITERATURE_BIO_TOOLS[0]
    servers = {
        "a": [_tool(curated), _tool(literature), _tool("describe_problem"), _tool("unrelated")],
        "b": [_tool(curated)],            # duplicate of a's
        "dead": RuntimeError("tools/list failed"),
    }

    async def _fake_load(session):
        out = servers[session]
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(lmt, "load_mcp_tools", _fake_load)
    got = asyncio.run(B.discover_bio_tools(["a", "b", "dead"], literature=False,
                                           exclude_names={"describe_problem"}))
    assert [t.name for t in got] == [curated]
    got = asyncio.run(B.discover_bio_tools(["a"], literature=True))
    assert sorted(t.name for t in got) == sorted([curated, literature])
    assert asyncio.run(B.discover_bio_tools([], literature=True)) == []
    assert asyncio.run(B.discover_bio_tools(["dead"], literature=True)) == []


def test_a_server_exposing_nothing_curated_leaves_the_biocurator_dumb(monkeypatch):
    """No all-tools fallback: an unrecognised roster yields no tools, not every tool."""
    pytest.importorskip("langchain_mcp_adapters")
    from langchain_core.tools import StructuredTool
    import langchain_mcp_adapters.tools as lmt
    from petab_harness.agent import biocurator as B

    async def _fake_load(session):
        return [StructuredTool.from_function(func=(lambda: ""), name=f"renamed_{i}",
                                             description="x") for i in range(20)]

    monkeypatch.setattr(lmt, "load_mcp_tools", _fake_load)
    assert asyncio.run(B.discover_bio_tools(["a"], literature=True)) == []
