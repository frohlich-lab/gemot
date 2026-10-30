"""The deepagents panel must contain ONLY our four curated personas (issue #112).

deepagents auto-inserts a `general-purpose` subagent holding the FULL tool pool unless the
active harness profile disables it. We never curated it, and it broke the panel two ways in
sweep 50751606: the lead delegated to it and it called `set_model` (9x), `fit` and
`change_parameters` INSIDE a subagent — bypassing lead-only authoring — and 39% of its tool
calls were unused filesystem builtins. ``gemot.agent`` now registers a
``HarnessProfile`` turning that auto-add off.

The roster + lead-only-authoring assertions are dependency-free and always run. The last test
exercises the real ``create_deep_agent`` and needs deepagents installed (CI installs the
examples-harness requirements; it skips anywhere they are absent).
"""

from __future__ import annotations

import pytest

from gemot import agent

# The one editor. It authors the model AND fits, so "who may change the model" and "who may
# score it" are now the same question — and the answer is still: the lead alone.
_EDITORS = ("make_changes",)


class _FakeTool:
    """Stand-in for a loaded MCP tool — _build_subagents only reads the name key."""

    def __init__(self, name: str):
        self.name = name


# --------------------------------------------------------------------- the declared roster
def test_subagent_names_are_exactly_the_four_personas():
    assert agent.SUBAGENT_NAMES == (
        "mathematician",
        "systems_scientist",
        "signal_engineer",
        "biocurator",
    )
    assert "general-purpose" not in agent.SUBAGENT_NAMES


def test_built_panel_matches_the_declared_roster():
    """The specs handed to create_deep_agent must be exactly SUBAGENT_NAMES, in order — no
    uncurated fifth subagent, and none of ours silently dropped."""
    tools_by_name = {n: _FakeTool(n) for n in agent.ALLOWED_TOOLS}
    panel = agent._build_subagents(tools_by_name)
    panel.append(agent._build_biocurator_subagent([], []))
    assert [spec["name"] for spec in panel] == list(agent.SUBAGENT_NAMES)


# ------------------------------------------------------------------- lead-only authoring
def test_only_the_lead_holds_the_model_editors():
    """The invariant `general-purpose` was breaking: the editors live on the lead alone."""
    for editor in _EDITORS:
        assert editor in agent.ORCHESTRATOR_TOOLS, f"{editor} missing from the lead's tools"
    for spec in agent.ANALYSTS:
        leaked = [e for e in _EDITORS if e in spec["tools"]]
        assert not leaked, f"analyst {spec['name']!r} holds model editor(s) {leaked}"
    # The biocurator holds the bio-server tools plus exactly ONE petab read — list_observables,
    # appended by the CALLER (run_deepagents), so the builder itself adds nothing: handing it an
    # empty tool list must yield an empty panel, or a petab tool could ride in unnoticed.
    assert agent._build_biocurator_subagent([], [])["tools"] == []


# ------------------------------------------------------------------- tools actually resolve
def test_every_subagent_tool_is_resolvable():
    """Each spec's tools must be a SUBSET of ALLOWED_TOOLS, or `_select_tools` drops them before
    `_build_subagents` can resolve them — silently, with only a stdout print. The draft era lost a
    whole arm this way with every roster test green, because those compare subagent NAMES."""
    for spec in agent.ACTIVE_SPECS:
        missing = [t for t in spec["tools"] if t not in agent.ALLOWED_TOOLS]
        assert not missing, (spec["name"], missing)


def test_only_the_lead_holds_get_model():
    """The raw committed text is for the lead's `base_version` forks; only the lead edits."""
    assert "get_model" in agent.ORCHESTRATOR_TOOLS
    for spec in agent.ANALYSTS:
        assert "get_model" not in spec["tools"], spec["name"]


# ------------------------------------------------------------- the real create_deep_agent
def test_create_deep_agent_does_not_add_the_general_purpose_subagent(monkeypatch):
    """End-to-end: with the profile registered, deepagents builds the panel we passed and
    nothing more. Asserts on what `create_deep_agent` actually hands `SubAgentMiddleware`."""
    pytest.importorskip("deepagents", reason="examples-harness dependency")
    import deepagents.graph as da_graph
    from langchain_openai import ChatOpenAI

    assert agent._disable_general_purpose_subagent() is True

    captured: dict = {}

    # A subclass, not a recording function: deepagents >= 0.7.15 also runs `isinstance(...,
    # SubAgentMiddleware)` against the name patched here. Same class name, because deepagents
    # places middleware by `.name`, which defaults to it.
    class SubAgentMiddleware(da_graph.SubAgentMiddleware):
        def __init__(self, *args, **kwargs):
            captured["names"] = [spec["name"] for spec in kwargs["subagents"]]
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(da_graph, "SubAgentMiddleware", SubAgentMiddleware)

    llm = ChatOpenAI(model="test-model", api_key="not-a-real-key")
    assert agent._general_purpose_enabled(llm) is False
    prompt_key = agent._subagent_prompt_key()
    da_graph.create_deep_agent(
        model=llm,
        tools=[],
        subagents=[
            {"name": name, "description": f"{name} stub", prompt_key: "stub", "tools": []}
            for name in agent.SUBAGENT_NAMES
        ],
    )

    assert captured["names"] == list(agent.SUBAGENT_NAMES)
