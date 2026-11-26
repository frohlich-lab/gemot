"""The run explorer (``gemot.explorer.app``): runs are found with the problem their server scored
against, and each is petab-mcp's viewer of its store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pn = pytest.importorskip("panel", reason="the explorer extra, until the petab-mcp pin ships viz")

from gemot.explorer import app as A  # noqa: E402


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
