"""`gemot.wandb_logging`: what one run's result.json becomes in W&B."""

from __future__ import annotations

import sys
import types

import pytest

from gemot import wandb_logging


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
        assert wandb_logging.log_result({"scenario": "toy", "model": "m", **result})
        return sent[-1]

    return log


def test_an_infrastructure_abort_is_not_logged_as_a_failure(logged):
    """The degraded `aborted:` result carries `success: False` though the judge never ran, and
    `passed = 0.0` counted every such abort as a failure in the group's W&B mean (#10)."""
    metrics = logged({"stop_reason": "aborted: ReadError", "success": False})
    assert "passed" not in metrics and "success" not in metrics


def test_a_declined_verdict_is_not_logged_as_a_failure(logged):
    """`recover_run.py` writes `success: null` for a run it cannot grade."""
    assert "passed" not in logged({"stop_reason": "recovered_from_incomplete_run", "success": None})


def test_a_verdict_is_logged_as_zero_or_one(logged):
    assert logged({"stop_reason": "completed", "success": True})["passed"] == 1.0
    assert logged({"stop_reason": "completed", "success": False})["passed"] == 0.0
