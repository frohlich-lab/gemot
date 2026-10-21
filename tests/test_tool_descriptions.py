"""The panel reads each tool's description as the server states it.

gemot used to replace them with its own copies (``TOOL_DESCRIPTIONS``), and the copies drifted:
the `statements` route reached the panel only through a stopgap that appended the server's
paragraph (frohlich-lab/PEtab-MCP#40). These load the tools the way the driver does, through the
MCP client and the langchain adapter, from an in-process server.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langchain_mcp_adapters", reason="agent stack not installed")

from conftest import YAML_PATH  # noqa: E402
from fastmcp import Client  # noqa: E402
from langchain_mcp_adapters.tools import load_mcp_tools  # noqa: E402
from petab_mcp.references import StubReferenceResolver  # noqa: E402
from petab_mcp.server import create_server  # noqa: E402

from gemot.agent.tools import ALLOWED_TOOLS, _blind_descriptions, _select_tools  # noqa: E402


def _served_and_panel(tmp_path, **kwargs):
    """The server's own tools by name, and what the panel holds of them."""
    server = create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        **kwargs,
    )

    async def load():
        async with Client(server) as client:
            served = {t.name: t for t in await client.list_tools()}
            return served, await load_mcp_tools(client.session)

    served, tools = asyncio.run(load())
    return served, {t.name: t for t in _blind_descriptions(_select_tools(tools))}


def test_the_panel_reads_the_servers_descriptions(tmp_path):
    served, panel = _served_and_panel(tmp_path)
    assert set(panel) == ALLOWED_TOOLS & set(served)
    assert {n: t.description for n, t in panel.items()} == {n: served[n].description for n in panel}


def test_the_statements_arm_reads_the_servers_statements_editor(tmp_path):
    """The arm's server offers the route alone, so the editor's schema and description carry no
    Antimony route, and both are the server's."""
    entity_map = tmp_path / "entity_map.json"
    entity_map.write_text("{}")
    served, panel = _served_and_panel(tmp_path, entity_map=str(entity_map), statements_only=True)
    editor = panel["make_changes"]
    assert editor.description == served["make_changes"].description
    assert "`statements` authors the WHOLE model" in editor.description
    args = set(editor.args_schema["properties"])
    assert {"statements", "statement_edits"} <= args
    assert not {"antimony", "model_edits"} & args
