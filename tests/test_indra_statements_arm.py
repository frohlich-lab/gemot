"""The indra_statements arm: the lead builds the model through make_changes' INDRA `statements`.

Three things have to hold for the arm to measure what it says. The server is offered the route
with an EMPTY entity map, so no reference structure reaches the agent. The agent is told how the
route works, although `_trim_descriptions` replaces the server's `make_changes` description (the
first run on the arm authored Antimony because the route's own paragraph never reached it). And
the model stays the statements' after its first build: a `make_changes` carrying `antimony` or
`model_edits` is bounced (a run that did not bounce them refined through Antimony from v1 on).

The pure helpers are tested with no third-party dependency; the middleware where langchain is
installed.
"""

from __future__ import annotations

import json
import types

import pytest

from gemot import ablations, prompts
from gemot.agent import guards, tools
from gemot.runner import main as runner_main

_ROUTE = (
    "`statements` is a third model route, and still exactly ONE route per call: it authors the "
    "WHOLE model from mechanism assertions. The result is an ordinary model."
)
_SERVER_MAKE_CHANGES = f"THE editor.\n\nPass `antimony` or `model_edits`.\n\n{_ROUTE}\n\nEvery version is kept."


def test_the_arm_is_registered_off_and_moves_the_prompt():
    arm = next(a for a in ablations.ABLATIONS if a.name == "indra_statements")
    assert arm.env == "PETAB_INDRA_STATEMENTS"
    assert arm.default_on is False and arm.changes_prompt is True


def test_the_lead_section_is_tracked_on_its_own():
    leaves = prompts.tracked_leaves()
    assert prompts._INDRA_STATEMENTS in leaves
    assert prompts.STATEMENTS_ONLY_REJECT_TMPL in leaves


def test_the_server_gets_an_empty_entity_map(tmp_path):
    path = runner_main._empty_entity_map(tmp_path / "build")
    assert json.loads(path.read_text()) == {}
    assert path.parent == tmp_path / "build"  # beside the build, not in the agent's work/


# ------------------------------------------------------------------ the route's description
def test_the_route_paragraph_is_the_servers_own():
    tool = types.SimpleNamespace(name="make_changes", description=_SERVER_MAKE_CHANGES)
    assert tools._statements_route(tool) == _ROUTE


def test_a_server_without_the_route_is_refused():
    tool = types.SimpleNamespace(name="make_changes", description="THE editor.")
    with pytest.raises(RuntimeError, match="without an entity map"):
        tools._statements_route(tool)


@pytest.mark.parametrize("arm", [True, False])
def test_make_changes_keeps_the_route_only_on_the_arm(monkeypatch, arm):
    monkeypatch.setattr(tools, "INDRA_STATEMENTS_ENABLED", arm)
    tool = types.SimpleNamespace(name="make_changes", description=_SERVER_MAKE_CHANGES)
    tools._trim_descriptions([tool])
    assert tool.description.startswith(prompts.TOOL_DESCRIPTIONS["make_changes"])
    assert (_ROUTE in tool.description) is arm


# ------------------------------------------------------------------ the statements-only guard
def test_only_the_other_model_routes_are_flagged():
    assert guards._non_statement_routes({"statements": [1], "parameters": [1]}) == []
    assert guards._non_statement_routes({"antimony": "model m end"}) == ["antimony"]
    assert guards._non_statement_routes({"model_edits": [{}], "antimony": ""}) == ["model_edits"]
    assert guards._non_statement_routes(None) == []


def test_the_guard_bounces_antimony_and_passes_statements():
    pytest.importorskip("langchain.agents.middleware.types", reason="agent stack not installed")
    stats: dict = {}
    guard = guards._statements_only_guard(stats)

    def call(name, args):
        request = types.SimpleNamespace(tool_call={"name": name, "args": args, "id": "c1"})
        return guard.wrap_tool_call(request, lambda r: "dispatched")

    assert call("make_changes", {"statements": ["r1: A binds B"]}) == "dispatched"
    assert call("get_model", {"format": "statements"}) == "dispatched"
    bounced = call("make_changes", {"antimony": "model m end", "justification": "j"})
    assert bounced != "dispatched" and bounced.status == "error"
    assert "['antimony']" in bounced.content and "`statements`" in bounced.content
    assert stats["statements_only_rejections"] == 1
