"""The indra_statements arm: the lead builds the model through make_changes' INDRA `statements`.

Three things have to hold for the arm to measure what it says. The server is offered the route
with an EMPTY entity map, so no reference structure reaches the agent. The agent is told how the
route works: the server offers it alone and its `make_changes` description documents only it
(tests/test_tool_descriptions.py; the first run on the arm authored Antimony because the route's
own paragraph never reached it). And the model stays the statements' after its first build: a
`make_changes` carrying `antimony` or `model_edits` is bounced (a run that did not bounce them
refined through Antimony from v1 on).

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
from gemot.runner.server import _SERVER_BOOTSTRAP


def test_the_arm_is_registered_off_and_moves_the_prompt():
    arm = next(a for a in ablations.ABLATIONS if a.name == "indra_statements")
    assert arm.env == "PETAB_INDRA_STATEMENTS"
    assert arm.default_on is False and arm.changes_prompt is True


def test_the_arm_texts_are_tracked_on_their_own():
    leaves = prompts.tracked_leaves()
    assert prompts.INDRA_SYSTEM_PROMPT in leaves
    assert prompts.INDRA_NO_MODEL_NUDGE in leaves
    assert prompts.STATEMENTS_ONLY_REJECT_TMPL in leaves


def test_the_arm_texts_do_not_teach_antimony():
    """The first runs on the arm kept the default lead prompt, with an INDRA note appended: Claude
    Haiku authored Antimony against the guard until the circuit breaker ended the run."""
    import re

    for text in (prompts.INDRA_SYSTEM_PROMPT, prompts.INDRA_NO_MODEL_NUDGE):
        assert not re.search(r"(?i)antimony", text)
    assert len(prompts.INDRA_SYSTEM_PROMPT) < len(prompts.DEFAULT_SYSTEM_PROMPT) / 2


def test_the_default_lead_prompt_is_not_the_arms():
    assert "INDRA" not in prompts.DEFAULT_SYSTEM_PROMPT


def test_the_server_gets_an_empty_entity_map(tmp_path):
    path = runner_main._empty_entity_map(tmp_path / "build")
    assert json.loads(path.read_text()) == {}
    assert path.parent == tmp_path / "build"  # beside the build, not in the agent's work/


def test_the_server_offers_the_route_alone_where_it_has_a_map(monkeypatch):
    """The map is given only on the arm, so it is what switches the text routes off. The bootstrap
    runs here with the server and uvicorn stubbed."""
    import sys

    seen: dict = {}
    server = types.ModuleType("petab_mcp.server")
    server.create_server = lambda *a, **kw: (
        seen.update(kw) or types.SimpleNamespace(run=lambda **_: None)
    )
    uvicorn = types.ModuleType("uvicorn")
    uvicorn.Config = object
    monkeypatch.setitem(sys.modules, "petab_mcp.server", server)
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    for entity_map in ("build/entity_map.json", ""):
        argv = ["-c", "p.yaml", "build", "off", "127.0.0.1", "0", "5", "", "0", entity_map, ""]
        monkeypatch.setattr(sys, "argv", argv)
        exec(_SERVER_BOOTSTRAP, {})
        assert seen["entity_map"] == (entity_map or None)


def _make_changes_tool(*args):
    schema = {"properties": dict.fromkeys(args, {}), "required": ["justification"]}
    return types.SimpleNamespace(name="make_changes", args_schema=schema)


def test_on_the_arm_only_the_statements_editor_is_accepted(monkeypatch):
    """`_statements_route` refused a server without the route; its successor must still fail at
    startup, not bounce every call of a run against the wrong server."""
    monkeypatch.setattr(tools, "INDRA_STATEMENTS_ENABLED", True)
    tools._check_statements_editor(
        [_make_changes_tool("statements", "bindings", "statement_edits")]
    )
    for wrong in (("antimony", "model_edits"), ("statements", "bindings", "antimony")):
        with pytest.raises(RuntimeError, match="statements-only"):
            tools._check_statements_editor([_make_changes_tool(*wrong)])
    # A server from before rows and their bindings were given apart: the lead's rows would not
    # match the editor it is prompted for.
    with pytest.raises(RuntimeError, match="no `bindings`"):
        tools._check_statements_editor([_make_changes_tool("statements", "statement_edits")])
    monkeypatch.setattr(tools, "INDRA_STATEMENTS_ENABLED", False)
    tools._check_statements_editor([_make_changes_tool("antimony")])


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

    rows = {"statements": ["r1: A binds B"], "bindings": {"r1": "default"}}
    assert call("make_changes", rows) == "dispatched"
    assert call("get_model", {"format": "statements"}) == "dispatched"
    bounced = call("make_changes", {"antimony": "model m end", "justification": "j"})
    assert bounced != "dispatched" and bounced.status == "error"
    assert "['antimony']" in bounced.content and "`statements`" in bounced.content
    assert stats["statements_only_rejections"] == 1
