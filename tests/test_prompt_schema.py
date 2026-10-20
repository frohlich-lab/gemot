"""The tables the panel is told about are the tables the server serves.

The tool descriptions are the server's own (frohlich-lab/PEtab-MCP#40), so the schema they state
moves with the server; what the prompts here say about the tables must not name one it no longer
serves.
"""

from __future__ import annotations

import pytest

from gemot import prompts as P


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
