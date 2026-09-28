"""`petab_harness.analysis`: the shared primitives every sweep report and trace review reads.

Fixtures build a sweep's results tree, a run's version store and an in-memory Phoenix `spans` table
from scratch, so these run in milliseconds with nothing installed but the harness — the same
condition a login-node report runs under.
"""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from petab_harness import analysis as A


def _write_task(root: Path, task: str, result: dict | None, combo: dict) -> Path:
    d = root / task
    d.mkdir(parents=True)
    (d / "combo.env").write_text("".join(f"{k}={v}\n" for k, v in combo.items()))
    if result is not None:
        (d / "result.json").write_text(json.dumps(result))
    return d


def _usage(prompt=1_000_000, cached=900_000, completion=10_000, calls=40):
    return {
        "prompt_tokens": prompt,
        "cache_read_tokens": cached,
        "completion_tokens": completion,
        "n_llm_calls": calls,
    }


@pytest.fixture
def results_root(tmp_path):
    root = tmp_path / "results"
    luna = "openai/gpt-5.6-luna-pro"
    _write_task(
        root,
        "100_0",
        {
            "scenario": "sneyd",
            "model": luna,
            "success": True,
            "delta_bic": -3.2,
            "stop_reason": "completed",
            "token_usage": {**_usage(), "cost_usd": 0.31},
            "llm_seconds": 600.0,
            "compute_seconds": 300.0,
            "tool_wall_seconds": 420.0,
            "queue_seconds": 120.0,
            "wall_seconds": 900.0,
            "n_fits": 12,
            "server_mode": "remote",
        },
        {"example": "sneyd", "model": luna, "exit_code": 0},
    )
    _write_task(
        root,
        "100_1",
        {
            "scenario": "sneyd",
            "model": luna,
            "success": False,
            "delta_bic": 44.0,
            "stop_reason": "wall_time_budget_exceeded",
            "token_usage": {**_usage(calls=80), "cost_usd": 1.20},
            "llm_seconds": 3000.0,
            "compute_seconds": 500.0,
            "wall_seconds": 3500.0,
        },
        {"example": "sneyd", "model": luna, "exit_code": 1},
    )
    # Out of credits before any work: pure noise, must not count as a modelling failure.
    _write_task(
        root,
        "100_2",
        {
            "scenario": "fujita",
            "model": luna,
            "success": False,
            "delta_bic": 1.5e7,
            "stop_reason": "APIStatusError: 402 payment",
            "n_ops": 0,
            "n_fits": 0,
        },
        {"example": "fujita", "model": luna, "exit_code": 1},
    )
    # Out of credits AFTER real work: a partial result that stays in.
    _write_task(
        root,
        "100_3",
        {
            "scenario": "fujita",
            "model": luna,
            "success": False,
            "delta_bic": 16.0,
            "stop_reason": "APIStatusError: 402 payment",
            "n_ops": 26,
            "n_fits": 25,
        },
        {"example": "fujita", "model": luna, "exit_code": 1},
    )
    # Never reached the shared server: combo.env only, exit 97.
    _write_task(
        root,
        "100_4",
        None,
        {"example": "boehm", "model": luna, "exit_code": 97, "reason": "petab_server_unreachable"},
    )
    # A second job, no recorded cost, ends before index 10 to check numeric ordering.
    _write_task(
        root,
        "200_10",
        {
            "scenario": "boehm",
            "model": "someone/new-model",
            "success": True,
            "delta_bic": 0.5,
            "stop_reason": "early_stop: plateau",
        },
        {"example": "boehm", "model": "someone/new-model", "exit_code": 0},
    )
    _write_task(
        root,
        "200_2",
        {
            "scenario": "boehm",
            "model": "someone/new-model",
            "success": False,
            "delta_bic": None,
            "stop_reason": "no_model_built",
        },
        {"example": "boehm", "model": "someone/new-model", "exit_code": 1},
    )
    return root


# --- cost ------------------------------------------------------------------------
def test_run_cost_is_the_recorded_bill_not_an_estimate():
    # The price table that priced runs from their token counts is gone (#170): a stale entry
    # produced wrong dollars silently. The only cost a run has is the one it recorded (#178).
    assert A.run_cost({**_usage(), "cost_usd": 0.42}) == pytest.approx(0.42)
    assert A.run_cost({"cost_usd": "0.05"}) == pytest.approx(0.05)


def test_run_cost_is_none_not_zero_when_unrecorded():
    assert A.run_cost(_usage()) is None  # tokens counted, bill never recorded
    assert A.run_cost(None) is None
    assert A.run_cost({"cost_usd": "n/a"}) is None


def test_cache_fraction_pools_prompt_tokens():
    rows = [{"token_usage": _usage(1000, 900)}, {"token_usage": _usage(1000, 100)}]
    assert A.cache_fraction(rows) == pytest.approx(0.5)
    assert A.cache_fraction([{"token_usage": None}]) is None


# --- loading ---------------------------------------------------------------------
def test_load_sweep_results_joins_result_with_combo_and_orders_numerically(results_root):
    rows = A.load_sweep_results(["100", "200"], results_root)
    assert [r["task"] for r in rows] == ["100_0", "100_1", "100_2", "100_3", "200_2", "200_10"]
    assert (
        rows[0]["combo"]["exit_code"] == "0" and rows[0]["job"] == "100" and rows[0]["index"] == "0"
    )
    assert all("result_missing" not in r for r in rows)


def test_load_sweep_results_can_count_what_a_sweep_lost(results_root):
    rows = A.load_sweep_results("100", results_root, include_missing=True)
    lost = [r for r in rows if r.get("result_missing")]
    assert [r["task"] for r in lost] == ["100_4"]
    assert lost[0]["scenario"] == "boehm" and lost[0]["success"] is False
    assert A.stop_class(lost[0]) == "server_unreachable"


def test_a_killed_task_is_no_result_not_server_unreachable(tmp_path):
    """A missing result.json means the task died; only exit 97 means it never reached the server.

    Conflating the two would file every Slurm TIMEOUT under a server-reachability failure.
    """
    root = tmp_path / "results"
    _write_task(root, "300_0", None, {"example": "boehm", "model": "m", "exit_code": "1"})
    _write_task(
        root,
        "300_1",
        None,
        {"example": "boehm", "model": "m", "exit_code": "97", "reason": "petab_server_unreachable"},
    )
    rows = {r["task"]: r for r in A.load_sweep_results("300", root, include_missing=True)}
    assert A.stop_class(rows["300_0"]) == "no_result"
    assert A.stop_class(rows["300_1"]) == "server_unreachable"
    assert not A.is_scoreable(rows["300_0"]) and not A.is_scoreable(rows["300_1"])


def test_read_combo_env_tolerates_a_missing_file(tmp_path):
    assert A.read_combo_env(tmp_path / "nope") == {}
    (tmp_path / "combo.env").write_text("example=sneyd\nexit_code=97\nreason=a=b\n")
    assert A.read_combo_env(tmp_path / "combo.env") == {
        "example": "sneyd",
        "exit_code": "97",
        "reason": "a=b",
    }


def test_load_archive_results_reads_every_run_newest_first(tmp_path):
    for run_id, started in (("t1", "2026-08-01T00:00:00"), ("t2", "2026-08-02T00:00:00")):
        d = tmp_path / run_id
        d.mkdir()
        (d / "result.json").write_text(json.dumps({"scenario": "x", "run_started_at": started}))
    (tmp_path / "junk").mkdir()
    rows = A.load_archive_results(tmp_path)
    assert [r["run_id"] for r in rows] == ["t2", "t1"]
    assert A.load_archive_results(tmp_path / "absent") == []


# --- classification --------------------------------------------------------------
@pytest.mark.parametrize(
    "row,expected",
    [
        ({"stop_reason": "completed"}, "completed"),
        ({"stop_reason": None}, "completed"),
        ({"stop_reason": "early_stop: plateau"}, "early_stop"),
        ({"stop_reason": "wall_time_budget_exceeded"}, "budget"),
        ({"stop_reason": "token_budget_exceeded"}, "budget"),
        ({"stop_reason": "GraphRecursionError: ..."}, "budget"),
        ({"stop_reason": "no_model_built"}, "no_model"),
        ({"stop_reason": "content_filter: flagged for biological risk"}, "content_filter"),
        # the driver's actual prefix (agent/driver._content_filter_stop_reason)
        ({"stop_reason": "content_filtered: possible biological risk"}, "content_filter"),
        # a judged result.json says how much was committed through head_version, not n_ops
        (
            {"stop_reason": "APIStatusError: 402", "head_version": 4, "n_fits": None},
            "infra_partial",
        ),
        ({"stop_reason": "APIStatusError: 402", "head_version": -1, "n_fits": 0}, "infra_noop"),
        ({"stop_reason": "APIStatusError: 402", "n_ops": 0, "n_fits": 0}, "infra_noop"),
        ({"stop_reason": "APIStatusError: 402", "n_ops": 26, "n_fits": 25}, "infra_partial"),
        (
            {"stop_reason": "stalled: no step completed in 1080s", "n_ops": 3, "n_fits": 3},
            "infra_partial",
        ),
        ({"stop_reason": "aborted: ClosedResourceError", "n_ops": 0}, "infra_noop"),
        (
            {"stop_reason": "recovered_from_incomplete_run: ...", "reconstructed": True},
            "reconstructed",
        ),
        ({"stop_reason": "petab_server_unreachable", "result_missing": True}, "server_unreachable"),
        ({"result_missing": True, "combo": {"exit_code": "97"}}, "server_unreachable"),
        ({"stop_reason": "no result.json (exit 1)", "result_missing": True}, "no_result"),
        ({"stop_reason": "SomethingElseError: odd"}, "other"),
    ],
)
def test_stop_class(row, expected):
    assert A.stop_class(row) == expected


def test_is_scoreable_drops_only_noise(results_root):
    rows = {r["task"]: r for r in A.load_sweep_results("100", results_root, include_missing=True)}
    assert A.is_scoreable(rows["100_0"]) and A.is_scoreable(rows["100_1"])
    assert not A.is_scoreable(rows["100_2"]), "an abort before any work counted as a result"
    assert A.is_scoreable(rows["100_3"]), "a late abort's partial result was dropped"
    assert not A.is_scoreable(rows["100_4"])


# --- aggregation -----------------------------------------------------------------
def test_summarize_by_model_scores_only_the_scoreable_runs(results_root):
    rows = A.load_sweep_results(["100", "200"], results_root, include_missing=True)
    by = {r["model"]: r for r in A.summarize_by(rows, "model")}
    luna = by["openai/gpt-5.6-luna-pro"]
    assert luna["n"] == 5 and luna["n_scoreable"] == 3 and luna["n_success"] == 1
    assert luna["success_rate"] == pytest.approx(1 / 3)
    assert luna["median_delta"] == pytest.approx(16.0)  # of {-3.2, 44.0, 16.0}; 1.5e7 excluded
    assert luna["best_delta"] == pytest.approx(-3.2)
    assert luna["stops"] == {
        "budget": 1,
        "completed": 1,
        "infra_noop": 1,
        "infra_partial": 1,
        "server_unreachable": 1,
    }
    assert luna["total_cost"] is not None and luna["mean_cost"] is not None
    new = by["someone/new-model"]
    assert new["total_cost"] is None and new["n_scoreable"] == 1 and new["success_rate"] == 1.0


@pytest.mark.parametrize(
    "row,expected",
    [
        # BIC arm: the judge records the two identically, either key answers.
        ({"delta_bic": -4.0, "delta_criterion": -4.0, "criterion": "bic"}, -4.0),
        # A non-BIC arm: `delta_bic` is recorded but is NOT what the gate measured.
        ({"delta_bic": -4.0, "delta_criterion": 12.5, "criterion": "aicc"}, 12.5),
        # Pre-#167 results carry only `delta_bic`, which was the gate then.
        ({"delta_bic": -4.0}, -4.0),
        ({"delta_criterion": None, "delta_bic": -4.0}, -4.0),
        ({"delta_criterion": float("inf"), "delta_bic": -4.0}, -4.0),
        ({"delta_bic": None}, None),
        ({}, None),
    ],
)
def test_score_delta_reads_the_criterion_the_run_was_graded_on(row, expected):
    got = A.score_delta(row)
    assert got is None if expected is None else got == pytest.approx(expected)


def test_a_declined_verdict_is_not_a_failed_one():
    """`success: null` is `recover_run.py` refusing to grade, not a miss.

    It writes one for a run ranked under a criterion it cannot re-score the reference under.
    Counting it as a failure would charge the model for the recovery tool's limitation.
    """
    declined = {
        "reconstructed": True,
        "stop_reason": "recovered_from_incomplete_run: ...",
        "success": None,
        "criterion": "aicc",
        "delta_bic": -72.41,
    }
    graded = {**declined, "success": False, "criterion": "bic"}
    assert not A.is_scoreable(declined)
    assert A.is_scoreable(graded), "a reconstructed BIC run is scoreable, marked but in"
    by = {r["model"]: r for r in A.summarize_by([{**declined, "model": "m"}], "model")}
    assert by["m"]["n"] == 1 and by["m"]["n_scoreable"] == 0
    assert by["m"]["success_rate"] is None and by["m"]["median_delta"] is None
    assert by["m"]["stops"] == {"reconstructed": 1}


def test_summarize_by_medians_the_deciding_delta_not_the_recorded_bic():
    rows = [
        {
            "model": "m",
            "success": True,
            "criterion": "aicc",
            "delta_criterion": d,
            "delta_bic": d - 100.0,
            "stop_reason": "completed",
        }
        for d in (-6.0, -2.0, 4.0)
    ]
    got = A.summarize_by(rows, "model")[0]
    assert got["median_delta"] == pytest.approx(-2.0), "read the BIC it was not graded on"
    assert got["best_delta"] == pytest.approx(-6.0)


def test_summarize_by_scenario(results_root):
    rows = A.load_sweep_results(["100", "200"], results_root, include_missing=True)
    by = {r["scenario"]: r for r in A.summarize_by(rows, "scenario")}
    assert set(by) == {"sneyd", "fujita", "boehm"}
    assert by["boehm"]["n"] == 3 and by["boehm"]["stops"]["server_unreachable"] == 1


def test_timing_split_and_summary(results_root):
    rows = A.load_sweep_results("100", results_root)
    split = A.timing_split(rows[0])
    assert split["llm"] == 600.0 and split["queue"] == 120.0 and split["compile"] is None
    summary = A.timing_summary(rows)
    assert summary["median_llm"] == pytest.approx(1800.0)  # median of 600, 3000
    assert summary["runs_with_queue_wait"] == 1
    assert summary["median_fit"] is None


def test_format_table_handles_none_and_dicts():
    text = A.format_table(
        [{"a": 1.23456, "b": None, "c": {"x": 1}}],
        [("a", "A", ".2f"), ("b", "B", ""), ("c", "C", "")],
    )
    assert text.splitlines()[0].startswith("A")
    assert "1.23" in text and "-" in text and "x=1" in text


# --- the version store -----------------------------------------------------------
def _write_store(run_dir: Path, metas: list[dict], events: list[dict] | None = None) -> None:
    (run_dir / ".petab_mcp").mkdir(parents=True, exist_ok=True)
    for meta in metas:
        vdir = run_dir / ".petab_mcp" / "versions" / f"v{meta['version']}"
        vdir.mkdir(parents=True)
        (vdir / "meta.json").write_text(json.dumps(meta))
    if events is not None:
        (run_dir / ".petab_mcp" / "provenance.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events)
        )


def test_best_fitted_version_ignores_simulate_only_and_recomputes_a_missing_bic(tmp_path):
    n = 100
    _write_store(
        tmp_path,
        [
            {"version": 0, "fit_status": "simulate_only", "neg_log_likelihood": -999.0},
            {
                "version": 1,
                "fit_status": "fit",
                "neg_log_likelihood": 10.0,
                "fitted_parameters": {"a": 1, "b": 2},
            },  # bic = 20 + 2 ln 100
            {"version": 2, "fit_status": "fit", "neg_log_likelihood": 12.0, "bic": 25.0},
            {"version": 3, "fit_status": "fit", "neg_log_likelihood": None},
        ],
    )
    bic, meta, n_fitted = A.best_fitted_version(tmp_path, n)
    assert meta["version"] == 2 and bic == pytest.approx(25.0) and n_fitted == 2
    assert A.bic_of({"neg_log_likelihood": 10.0, "fitted_parameters": {"a": 1, "b": 2}}, n) == (
        pytest.approx(20 + 2 * math.log(n))
    )
    assert A.best_fitted_version(tmp_path / "absent") is None


def test_run_digest_measures_the_dry_streak(tmp_path):
    events = [
        {
            "type": "fit",
            "version": 1,
            "is_new_best": True,
            "fit_status": "fit",
            "staged_edits": [{"operation": "set_model"}],
            "justification": "x" * 400,
        },
        {"type": "fit", "version": 2, "is_new_best": False, "fit_status": "fit"},
        {"type": "fit", "version": 3, "is_new_best": False, "fit_status": "simulate_only"},
        {"type": "fit_noop", "duplicate_of_version": 3},
        {
            "type": "alteration",
            "version": 4,
            "is_new_best": True,
            "fit_status": "fit",
            "operation": "add_reaction",
            "parameter_diff": {"added": ["k1", "k2"], "removed": []},
        },
        {"type": "fit", "version": 5, "is_new_best": False, "fit_status": "fit"},
    ]
    _write_store(tmp_path, [], events)
    digest = A.run_digest(tmp_path)
    assert digest["n_ops"] == 5 and digest["n_new_best"] == 2 and digest["n_fits"] == 4
    assert digest["longest_non_improving_streak"] == 2
    ops = digest["ops"]
    assert ops[0]["op"] == "set_model" and len(ops[0]["justification"]) < 400
    assert ops[3]["op"] == "add_reaction" and ops[3]["param_diff"] == {"added": 2, "removed": 0}
    assert A.provenance_events(tmp_path / "absent") == []


def test_find_work_dir_prefers_the_archive_then_results_then_both_work_depths(tmp_path):
    results, work = tmp_path / "results", tmp_path / "work"
    archive = tmp_path / "archive" / "abc"
    _write_store(archive, [{"version": 0, "fit_status": "fit", "neg_log_likelihood": 1.0}])
    (results / "1_0").mkdir(parents=True)
    (results / "1_0" / "result.json").write_text(json.dumps({"archive_dir": str(archive)}))
    assert A.find_work_dir("1_0", results, work) == archive

    # An empty archive must not win over a tree that still holds the history.
    empty = tmp_path / "archive" / "empty"
    (empty / ".petab_mcp" / "versions").mkdir(parents=True)
    (results / "1_1").mkdir()
    (results / "1_1" / "result.json").write_text(json.dumps({"archive_dir": str(empty)}))
    deep = work / "sweep_x" / "1_1" / "sneyd" / "work"
    _write_store(deep, [{"version": 0, "fit_status": "fit", "neg_log_likelihood": 1.0}])
    assert A.find_work_dir("1_1", results, work) == deep

    shallow = work / "1_2" / "sneyd" / "work"
    _write_store(shallow, [{"version": 0, "fit_status": "fit", "neg_log_likelihood": 1.0}])
    assert A.find_work_dir("1_2", results, work) == shallow
    assert A.find_work_dir("1_9", results, work) is None

    # `array_task.sh` stages under `<tag>/examples/<example>/work`, one level deeper again.
    staged = work / "1_3" / "examples" / "sneyd" / "work"
    _write_store(staged, [{"version": 0, "fit_status": "fit", "neg_log_likelihood": 1.0}])
    assert A.find_work_dir("1_3", results, work) == staged


def test_live_rows_reads_bic_off_the_fit_EVENT_not_its_result(tmp_path):
    """`bic` is TOP-LEVEL on a provenance `fit` event.

    The superseded slurm/live_bic_local.py read `rec["result"]["bic"]`, which is populated on no
    event type — measured across 68 real runs / 372 fit events, every one of them a miss — so it
    silently reported "no fit yet" for every task. Score the wrong key and this test fails.
    """
    work = tmp_path / "runs" / "work" / "700_2" / "examples" / "sneyd" / "work"
    _write_store(
        work,
        [],
        events=[
            {"type": "edit", "result": {"model": {"n_species": 3}}},
            {
                "type": "fit",
                "version": 0,
                "fit_status": "fit",
                "neg_log_likelihood": 60.0,
                "bic": 140.0,
                "result": {},
            },
            {
                "type": "fit",
                "version": 1,
                "fit_status": "fit",
                "neg_log_likelihood": 50.0,
                "bic": 121.5,
                "result": {},
            },  # the best
            {"type": "fit_noop", "version": 1},
        ],
    )
    work.mkdir(parents=True, exist_ok=True)
    (work / "run_live.json").write_text(
        json.dumps(
            {
                "scenario": "sneyd",
                "bic_reference": 130.0,
                "bic_threshold": 10.0,
                "phoenix_trace_id": "abc",
            }
        )
    )

    rows = A.live_rows("700", tmp_path / "runs" / "work")
    assert len(rows) == 1
    r = rows[0]
    assert r["task"] == "700_2" and r["scenario"] == "sneyd"
    assert r["n_fits"] == 2, "fit_noop or the edit event was counted as a fit"
    assert r["best_bic"] == pytest.approx(121.5)
    assert r["delta_bic"] == pytest.approx(-8.5) and r["passing"] is True

    assert A.live_rows("999", tmp_path / "runs" / "work") == []


def test_live_rows_falls_back_to_the_version_store_and_reports_no_fit_yet(tmp_path):
    root = tmp_path / "work"
    # No provenance log at all: the store is the only record.
    store_only = root / "701_0" / "sneyd" / "work"
    _write_store(
        store_only, [{"version": 0, "fit_status": "fit", "neg_log_likelihood": 10.0, "bic": 33.0}]
    )
    (store_only / "run_live.json").write_text(
        json.dumps({"scenario": "sneyd", "bic_reference": 30.0, "bic_threshold": 10.0})
    )
    # Started, but nothing fitted yet.
    fresh = root / "701_1" / "sneyd" / "work"
    fresh.mkdir(parents=True)
    (fresh / "run_live.json").write_text(
        json.dumps({"scenario": "sneyd", "bic_reference": 30.0, "bic_threshold": 10.0})
    )

    rows = {r["task"]: r for r in A.live_rows("701", root)}
    assert rows["701_0"]["best_bic"] == pytest.approx(33.0)
    assert rows["701_0"]["delta_bic"] == pytest.approx(3.0) and rows["701_0"]["passing"] is True
    assert rows["701_1"]["best_bic"] is None and rows["701_1"]["delta_bic"] is None
    assert rows["701_1"]["passing"] is None, "an unstarted fit must not read as a failure"


def test_work_roots_covers_every_layout_a_sweep_has_used(tmp_path, monkeypatch):
    monkeypatch.delenv("PETAB_MCP_SCRATCH", raising=False)
    monkeypatch.delenv("PETAB_MCP_WORK_ROOT", raising=False)
    monkeypatch.setattr(A.paths, "repo_root", lambda: tmp_path)
    roots = A.work_roots()
    assert roots[0] == tmp_path / "slurm" / "runs" / "work"  # array_task.sh's default
    assert set(roots) == {
        tmp_path / "slurm" / "runs" / "work",
        tmp_path / "slurm" / "work",
        tmp_path / ".slurm_scratch",
    }
    assert len(roots) == len(set(roots)), "the default root was listed twice"

    # A tree found under the legacy scratch root, with no work_root passed in.
    legacy = tmp_path / ".slurm_scratch" / "9_0" / "examples" / "sneyd" / "work"
    _write_store(legacy, [{"version": 0, "fit_status": "fit", "neg_log_likelihood": 1.0}])
    assert A.find_work_dir("9_0", tmp_path / "results") == legacy

    monkeypatch.setenv("PETAB_MCP_SCRATCH", str(tmp_path / "nodelocal"))
    assert A.work_roots()[0] == tmp_path / "nodelocal"


# --- Phoenix ---------------------------------------------------------------------
def test_flatten_attrs_normalises_nested_and_dotted_forms():
    nested = {"llm": {"output_messages": [{"message": {"content": "hi"}}]}, "tool.name": "fit"}
    flat = A.flatten_attrs(nested)
    assert flat["llm.output_messages.0.message.content"] == "hi"
    assert flat["tool.name"] == "fit"
    assert A.first_attr(flat, "missing", "tool.name") == "fit"
    assert A.first_attr(flat, "missing", default="d") == "d"


def test_clip_truncates_with_a_count():
    assert A.clip(None) is None
    assert A.clip("abc") == "abc"
    assert A.clip("a" * 300, 10) == "a" * 10 + "…[+290 chars]"
    assert A.clip({"k": 1}) == '{"k": 1}'


@pytest.fixture
def phoenix_db(tmp_path):
    db = tmp_path / "phoenix.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "create table spans (id integer primary key, trace_rowid int, parent_id text, "
        "name text, span_kind text, attributes text, status_code text, "
        "status_message text, start_time text)"
    )
    rows = [
        (
            1,
            7,
            None,
            "run",
            "CHAIN",
            json.dumps({"petab": {"scenario": "sneyd", "model": "m", "delta_bic": 3.0}}),
            "OK",
            None,
            "t0",
        ),
        (
            2,
            7,
            "1",
            "make_changes",
            "TOOL",
            json.dumps(
                {
                    "tool.name": "make_changes",
                    "output.value": json.dumps({"applied": False, "error": "bad"}),
                }
            ),
            "OK",
            None,
            "t1",
        ),
        (3, 7, "1", "query_data", "TOOL", json.dumps({}), "ERROR", "Need a DataFrame", "t2"),
        (4, 8, None, "other", "CHAIN", json.dumps({"no": "identity"}), "OK", None, "t0"),
        (
            5,
            7,
            "1",
            "llm",
            "LLM",
            json.dumps({"llm": {"output_messages": [{"message": {"content": "thinking"}}]}}),
            "OK",
            None,
            "t3",
        ),
    ]
    conn.executemany("insert into spans values (?,?,?,?,?,?,?,?,?)", rows)
    # The join the collector actually indexes: spans.trace_rowid -> traces.id, keyed by trace_id.
    conn.execute("create table traces (id integer primary key, trace_id text unique)")
    conn.executemany("insert into traces values (?,?)", [(7, "aaaa"), (8, "bbbb")])
    conn.commit()
    conn.close()
    return db


def test_phoenix_helpers_stream_filtered_spans_read_only(phoenix_db):
    conn = A.phoenix_connect(phoenix_db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("insert into spans (id) values (99)")
    runs = A.petab_runs(conn)
    assert set(runs) == {7} and runs[7]["scenario"] == "sneyd" and runs[7]["delta_bic"] == 3.0

    tools = list(A.iter_spans(conn, kind="tool", trace_rowid=7))
    assert [s["name"] for s in tools] == ["make_changes", "query_data"]
    assert A.span_error(tools[0]) is None and A.span_error(tools[1]) == "Need a DataFrame"
    assert A.tool_output(tools[0]) == {"applied": False, "error": "bad"}
    assert [s["name"] for s in A.iter_spans(conn, name_like="query%")] == ["query_data"]
    llm = next(A.iter_spans(conn, kind="LLM"))
    assert llm["attributes"]["llm.output_messages.0.message.content"] == "thinking"
    conn.close()


def test_iter_spans_resolves_trace_ids_through_the_indexed_traces_table(phoenix_db):
    conn = A.phoenix_connect(phoenix_db)
    assert A.trace_rowids(conn, ["aaaa", "bbbb", "nope"]) == {"aaaa": 7, "bbbb": 8}
    assert A.trace_rowids(conn, []) == {} and A.trace_rowids(conn, [None, ""]) == {}

    got = [s["name"] for s in A.iter_spans(conn, trace_ids=["aaaa"])]
    assert got == ["run", "make_changes", "query_data", "llm"]
    assert [s["name"] for s in A.iter_spans(conn, trace_ids=["bbbb"])] == ["other"]
    assert len(list(A.iter_spans(conn, trace_ids=["aaaa", "bbbb"]))) == 5
    # Composes with the other filters rather than replacing them.
    assert [s["name"] for s in A.iter_spans(conn, trace_ids=["aaaa"], kind="TOOL")] == [
        "make_changes",
        "query_data",
    ]
    # An id no trace matches yields nothing — never the whole table.
    assert list(A.iter_spans(conn, trace_ids=["nope"])) == []


def test_local_phoenix_copy_takes_the_wal_and_leaves_the_shm(tmp_path, phoenix_db):
    """The three rules that make a cross-host read correct; see local_phoenix_copy."""
    phoenix_db.with_name(phoenix_db.name + "-wal").write_bytes(b"wal-bytes")
    phoenix_db.with_name(phoenix_db.name + "-shm").write_bytes(b"shm-bytes")
    dest = tmp_path / "nodelocal"
    copy = A.local_phoenix_copy(phoenix_db, dest)

    assert copy == dest / "phoenix.db" and copy.is_file()
    assert (dest / "phoenix.db-wal").read_bytes() == b"wal-bytes"
    assert not (dest / "phoenix.db-shm").exists(), "a foreign -shm is worse than none"
    # Read-WRITE openable, so SQLite can replay the WAL on first open.
    conn = sqlite3.connect(copy)
    conn.execute("insert into traces values (9, 'cccc')")
    conn.close()

    # A checkpointed store, with no -wal beside it, copies fine.
    phoenix_db.with_name(phoenix_db.name + "-wal").unlink()
    plain = A.local_phoenix_copy(phoenix_db, tmp_path / "d2")
    assert plain.is_file() and not (tmp_path / "d2" / "phoenix.db-wal").exists()


# --- the command line ------------------------------------------------------------
def test_cli_summary_and_rows(results_root, capsys):
    assert A.main(["summary", "100", "200", "--results-dir", str(results_root)]) == 0
    out = capsys.readouterr().out
    assert "openai/gpt-5.6-luna-pro" in out and "server_unreachable=1" in out
    assert "runs_with_queue_wait=1" in out
    assert A.main(["rows", "100", "--results-dir", str(results_root)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("task,scenario,model")
    assert any(line.startswith("100_4,boehm") for line in out.splitlines())
    assert A.main(["summary", "999", "--results-dir", str(results_root)]) == 1


def test_the_module_is_import_light():
    probe = (
        "import sys; import petab_harness.analysis; "
        "bad = [m for m in ('mcp','petab_mcp','phoenix','pandas','numpy','matplotlib') "
        "if m in {x.split('.')[0] for x in sys.modules}]; "
        "sys.exit('leaked: ' + ','.join(bad) if bad else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
