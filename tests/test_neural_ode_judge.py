"""Proposals reach the judge as any panel model does: a private example copy, the scripted fix, result.json."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from petab_harness.neural_ode import judge as J


def _example(root: Path) -> Path:
    example = root / "toy"
    (example / "problem").mkdir(parents=True)
    (example / "problem" / "problem.yaml").write_text("format_version: 1\n")
    (example / "problem" / "reference.json").write_text('{"nll": 1.0, "k": 2, "n": 10}')
    (example / "problem" / ".petab_mcp").mkdir()
    (example / "problem" / ".petab_mcp" / "old.json").write_text("{}")
    (example / "validation").mkdir()
    (example / "validation" / "measurements.tsv").write_text("observableId\tsimulationConditionId\ttime\tmeasurement\n")
    return example


def test_each_proposal_is_committed_and_judged_in_its_own_example_copy(tmp_path, monkeypatch):
    example = _example(tmp_path)
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps({"problem": "toy", "proposals": [
        {"antimony": "model a\n  A' = -c0*A;\n  c0 = 0.5;\nend", "state_names": ["A"], "model_name": "toy_node_selected",
         "n_constants": 1, "record": {"tag": "selected", "field_r2": [0.9]}},
        {"antimony": "model b\n  A' = -c0*A + c1;\n  c0 = 0.5;\n  c1 = 0.1;\nend", "state_names": ["A"],
         "model_name": "toy_node_c5", "n_constants": 2, "record": {"tag": "c5", "complexity_budget": 5, "field_r2": [0.95]}},
    ]}))
    seen = []

    def fake_run_scenario(scenario, argv):
        # The scripted fix issues one make_changes with the proposal's model, as the canonical
        # builds do; the judge then writes result.json beside the private problem copy.
        calls = []

        async def call(name, **kwargs):
            calls.append((name, kwargs))
            return {"applied": True, "committed": True, "version": 1}

        asyncio.run(scenario.scripted_fix(call, 7))
        seen.append((scenario.name, scenario.problem_dir, argv, calls))
        work = scenario.example_dir / "work"
        work.mkdir()
        (work / "result.json").write_text(json.dumps({"success": True, "delta_bic": -1.5, "final_nll": 0.5, "k": 3,
                                                       "bic": 8.0, "validation": {"status": "ok", "nll": 2.0}}))
        return 0

    import petab_harness.runner.cli as cli
    monkeypatch.setattr(cli, "run_scenario", fake_run_scenario)
    rows = J.score_proposals(proposals, example, tmp_path / "judge", n_starts=7, only=("selected", "c5"))

    assert [r["model_name"] for r in rows] == ["toy_node_selected", "toy_node_c5"]
    name, problem_dir, argv, calls = seen[0]
    assert name == "toy_node_selected" and problem_dir == tmp_path / "judge" / "toy_node_selected" / "problem"
    assert argv == ["--driver", "scripted", "--n-starts", "7"]
    assert [n for n, _ in calls] == ["make_changes"]
    assert calls[0][1]["antimony"].startswith("model a") and calls[0][1]["n_starts"] == 7
    assert calls[0][1]["justification"] == J.JUSTIFICATION
    # The private copy carries the reference, the held-out set, and no earlier provenance.
    assert (problem_dir / "reference.json").is_file() and (problem_dir.parent / "validation" / "measurements.tsv").is_file()
    assert not (problem_dir / ".petab_mcp").exists()
    assert rows[1]["complexity_budget"] == 5 and rows[1]["delta_bic"] == -1.5 and rows[1]["validation"]["nll"] == 2.0
    scores = json.loads((tmp_path / "judge" / "scores.json").read_text())
    assert scores["n_starts"] == 7 and [r["tag"] for r in scores["scores"]] == ["selected", "c5"]
