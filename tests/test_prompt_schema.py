"""The tables the panel is told about are the tables the server serves.

The panel never reads a server docstring: ``_trim_descriptions`` replaces each with the one in
``prompts.py``. So when the server's tables change -- PEtab v1's wide conditions and
``simulationConditionId`` became v2's long conditions, experiments and ``experimentId`` -- the
panel goes on reading the old schema until these say otherwise. Guessing column names was the
largest non-editing error class in a measured sweep, so the schema is checked against the server's
own text, at the pinned server.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from gemot import prompts as P


def _server_docstring(tool: str) -> str:
    """The docstring of the server tool ``tool``, read from the installed server's source."""
    import petab_mcp.server

    tree = ast.parse(Path(petab_mcp.server.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == tool:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"the server defines no tool {tool!r}")


def _words(text: str) -> str:
    return " ".join(text.split())


def test_query_data_states_the_servers_schema_verbatim():
    """The schema paragraph is copied, not paraphrased: under v2 `experimentId` (the PEtab
    tables) and `experiment_id` (the derived ones) are both real, and only the exact text says
    which table has which."""
    doc = _server_docstring("query_data")
    match = re.search(r"TWO NAMING CONVENTIONS.*?(?=\n\s*\n)", doc, re.S)
    assert match, "the server's query_data docstring no longer states its naming conventions"
    assert _words(match[0]) in _words(P.TOOL_DESCRIPTIONS["query_data"])


@pytest.mark.parametrize("tool", ["list_conditions", "list_experiments"])
def test_the_design_tools_name_the_servers_columns(tool):
    """Every column the server's docstring names is one the panel's description names too."""
    columns = set(re.findall(r"`([a-z]+[A-Z]\w*)`", _server_docstring(tool)))
    assert columns, f"the server's {tool} docstring names no columns"
    assert not sorted(c for c in columns if c not in P.TOOL_DESCRIPTIONS[tool])


def test_no_panel_text_names_a_v1_table_key():
    """v1's condition keys are gone from every table the server serves; a prompt naming one sends
    the agent to a column that does not exist."""
    v1 = ("simulationConditionId", "preequilibrationConditionId", "condition_id")
    for text in P.tracked_leaves():
        assert not any(key in text for key in v1), [k for k in v1 if k in text]


def test_the_lead_reads_the_experiments():
    """The lead holds `list_conditions`; under v2 a condition says WHAT is set and only the
    experiments say WHEN, so it holds `list_experiments` too, and is told to read both."""
    pytest.importorskip("langchain_core", reason="agent stack not installed")
    from gemot.agent.tools import ORCHESTRATOR_TOOLS

    assert {"list_conditions", "list_experiments"} <= set(ORCHESTRATOR_TOOLS)
    for prompt in (P.DEFAULT_SYSTEM_PROMPT, P.BIOLOGY_BLIND_SYSTEM_PROMPT):
        assert "list_conditions" in prompt and "list_experiments" in prompt
