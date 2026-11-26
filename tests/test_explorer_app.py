"""The run explorer (``gemot.explorer.app``, ``.conversation``): runs are found with the problem
their server scored against, and each is petab-mcp's viewer with the conversation that produced
the version picked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pn = pytest.importorskip("panel", reason="the explorer extra, until the petab-mcp pin ships viz")

from gemot.explorer import app as A  # noqa: E402
from gemot.explorer import conversation as C  # noqa: E402


def _example(examples: Path, name: str) -> Path:
    problem = examples / name / "problem"
    problem.mkdir(parents=True)
    (problem / "problem.yaml").write_text("format_version: 2.0.0\n")
    return problem


def test_an_archived_run_is_shown_against_its_examples_problem(tmp_path):
    """The archive keeps the store but not the problem, whose design tables the server never
    changes: the example's pristine problem/ is the one it scored against."""
    examples = tmp_path / "examples"
    problem = _example(examples, "fujita")
    run = examples / "_runs_archive" / "trace-1"
    (run / ".petab_mcp" / "versions").mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"scenario": "fujita", "success": True}))
    found = A.locate("trace-1", examples / "_runs_archive", examples)
    assert (found.store, found.problem, found.live) == (
        run / ".petab_mcp",
        problem / "problem.yaml",
        False,
    )
    assert A.locate("no-such-run", examples / "_runs_archive", examples) is None
    listed = A.runs(examples / "_runs_archive", examples)
    assert list(listed["run"]) == ["trace-1"] and list(listed["status"]) == ["succeeded"]


def test_a_live_run_is_shown_from_its_work_dir_and_listed_first(tmp_path):
    examples = tmp_path / "examples"
    _example(examples, "fujita")
    work = examples / "fujita" / "work"
    work.mkdir()
    (work / "problem.yaml").write_text("format_version: 2.0.0\n")
    (work / "run_live.json").write_text(json.dumps({"run_id": "trace-2", "scenario": "fujita"}))
    archived = examples / "_runs_archive" / "trace-1"
    archived.mkdir(parents=True)
    (archived / "result.json").write_text(json.dumps({"scenario": "fujita"}))
    found = A.locate("trace-2", examples / "_runs_archive", examples)
    assert (found.store, found.problem, found.live) == (
        work / ".petab_mcp",
        work / "problem.yaml",
        True,
    )
    assert list(A.runs(examples / "_runs_archive", examples)["run"]) == ["trace-2", "trace-1"]


#: Five lead turns: v0 committed by turn 1, v1 by turn 3 (with an analyst's delegation).
_BUNDLE = {
    "meta": {"scenario": "fujita", "model": "m", "success": True},
    "variants": [{"index": 0, "version": 0}, {"index": 1, "version": 1}],
    "turns": [
        {
            "i": 0,
            "text": "look at the data",
            "tools": [{"name": "query_data", "args": {"sql": "x"}}],
        },
        {"i": 1, "text": "first model", "variant_index": 0, "tools": []},
        {
            "i": 2,
            "text": "ask an analyst",
            "tools": [
                {
                    "name": "task",
                    "args": {},
                    "delegation": {
                        "subagent_type": "reviewer",
                        "description": "check residuals",
                        "items": [{"type": "assistant", "text": "they trend"}],
                    },
                }
            ],
        },
        {
            "i": 3,
            "text": "second model",
            "variant_index": 1,
            "tools": [{"name": "make_changes", "args": {}, "result": {"committed": True}}],
        },
        {"i": 4, "text": "done", "tools": []},
    ],
    "totals": {"lead_turns": 5, "total": 1234},
    "phoenix": {"ok": True},
}


def test_the_turns_shown_are_those_that_produced_the_version():
    assert C.committed_versions(_BUNDLE) == {1: 0, 3: 1}
    assert [t["i"] for t in C.turns_for(_BUNDLE, 0)] == [0, 1]
    assert [t["i"] for t in C.turns_for(_BUNDLE, 1)] == [2, 3]
    assert [t["i"] for t in C.turns_for(_BUNDLE, 7)] == [0, 1, 2, 3, 4]


def test_the_conversation_tab_follows_the_viewers_version(monkeypatch, tmp_path):
    import param
    from bokeh.document import Document

    monkeypatch.setattr(C.B, "build_bundle", lambda *a, **k: _BUNDLE)

    class Viz(param.Parameterized):
        version = param.Integer(default=1, allow_None=True)
        tick = param.Integer(default=0)

    viz = Viz()
    tab = C.Conversation("trace-1", tmp_path, {}, "http://localhost:6006", live=False)
    view = tab.view(viz)
    view.get_root(Document())

    def titles():
        return [card.title for card in view.select(pn.Card) if card.title.startswith("turn")]

    assert titles() == ["turn 2", "turn 3 · committed v1"]
    viz.version = 0
    assert titles() == ["turn 0", "turn 1 · committed v0"]
    tab.whole.value = True
    assert len(titles()) == 5


def test_a_run_without_a_trace_says_only_that(monkeypatch, tmp_path):
    """A scripted run has no Phoenix trace: the tab said "no turn committed this version, every
    turn is shown" above no turns at all."""
    empty = {**_BUNDLE, "turns": [], "phoenix": {"ok": False, "note": "no phoenix_trace_id"}}
    monkeypatch.setattr(C.B, "build_bundle", lambda *a, **k: empty)
    tab = C.Conversation("trace-1", tmp_path, {}, "http://localhost:6006", live=False)
    text = tab._render(0, 0, False, 1).object
    assert "No conversation: no phoenix_trace_id" in text and "every turn" not in text
