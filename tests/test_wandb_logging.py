"""`gemot.wandb_logging`: what one run's result.json becomes in W&B."""

from __future__ import annotations

import sys
import types

import pytest

from gemot import wandb_logging as W


@pytest.fixture
def logged(monkeypatch):
    """The metrics `log_result` sends, through a stand-in `wandb` module."""
    sent: list[dict] = []

    class _Run:
        def log(self, data):
            sent.append(data)

        def finish(self):
            pass

    fake = types.SimpleNamespace(init=lambda **_: _Run(), Settings=lambda **_: None)
    monkeypatch.setitem(sys.modules, "wandb", fake)

    def log(result):
        assert W.log_result({"scenario": "toy", "model": "m", **result})
        return sent[-1]

    return log


def test_an_infrastructure_abort_before_any_work_is_not_logged_as_a_failure(logged):
    """The degraded `aborted:` result carries `success: False` though the judge never ran, and
    `passed = 0.0` counted every such abort as a failure in the group's W&B mean (#10). A stop on
    a transport marker with nothing committed is the same noise, though the judge then scored the
    baseline; the sweep rate (`analysis.is_scoreable`) leaves both out."""
    metrics = logged({"stop_reason": "aborted: ReadError", "success": False})
    assert "passed" not in metrics and "success" not in metrics
    timed_out = {"stop_reason": "fit call timed out", "success": False, "head_version": -1}
    assert "passed" not in logged(timed_out)


def test_a_late_abort_that_did_work_stays_a_failure(logged):
    """Its partial result is kept wherever a sweep is scored (`experiments.is_noop_infra_abort`);
    dropping it here alone would make the W&B rate disagree with the sweep's."""
    late = {"stop_reason": "aborted: ReadError", "success": False, "n_ops": 14}
    assert logged(late)["passed"] == 0.0


def test_a_declined_verdict_is_not_logged_as_a_failure(logged):
    """`recover_run.py` writes `success: null` for a run it cannot grade."""
    assert "passed" not in logged({"stop_reason": "recovered_from_incomplete_run", "success": None})


def test_a_verdict_is_logged_as_zero_or_one(logged):
    assert logged({"stop_reason": "completed", "success": True})["passed"] == 1.0
    assert logged({"stop_reason": "completed", "success": False})["passed"] == 0.0
