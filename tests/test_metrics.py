"""`gemot.metrics`: the cross-run history CSV."""

from __future__ import annotations

import csv

from gemot import metrics


def _rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_collecting_twice_does_not_append_the_same_run_twice(tmp_path):
    """`collect_metrics.py` re-reads each example's latest result.json, so a second collection
    with no new run in between appended every run again and each counted twice (#10)."""
    out = tmp_path / "history.csv"
    a = {"scenario": "toy", "run_started_at": "2026-10-20T10:00:00.000001+00:00", "success": True}
    b = {"scenario": "toy", "run_started_at": "2026-10-20T11:00:00.000001+00:00", "success": False}

    metrics.append_history([a], out)
    metrics.append_history([a, b, b], out)

    assert [r["run_started_at"] for r in _rows(out)] == [a["run_started_at"], b["run_started_at"]]
    # The same instant in another example is another run.
    metrics.append_history([{**a, "scenario": "other"}], out)
    assert len(_rows(out)) == 3


def test_a_row_without_a_start_time_is_always_appended(tmp_path):
    """Nothing identifies it, so nothing says it repeats an earlier row."""
    out = tmp_path / "history.csv"
    metrics.append_history([{"scenario": "toy"}], out)
    metrics.append_history([{"scenario": "toy"}], out)
    assert len(_rows(out)) == 2
