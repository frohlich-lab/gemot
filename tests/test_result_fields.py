"""Test the residual-structure block the judge writes into result.json
(examples/_harness/runner.py `_residual_structure_fields`).

Pure projection of a get_fit_diagnosis payload; importing the harness runner pulls in `mcp`,
so skip where that harness-only dependency is absent."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

pytest.importorskip("mcp", reason="harness-only dependency")
from _harness.runner import _residual_structure_fields  # noqa: E402


def test_extracts_verdict_and_per_observable():
    diag = {
        "residual_structure": {"flagged_observables": ["obs_A"], "low_power": False},
        "per_observable": [
            {"observable_id": "obs_A", "mean_weighted_residual": 2.0,
             "structure_low_power": False, "sign_test": {"p": 0.002}, "runs_test": {"p": 0.3}},
            {"observable_id": "obs_B", "mean_weighted_residual": 0.1,
             "structure_low_power": True, "sign_test": {"p": 0.8}, "runs_test": {"p": None}},
        ],
    }
    out = _residual_structure_fields(diag)
    assert out["flagged_observables"] == ["obs_A"]
    assert out["low_power"] is False
    rows = {r["observable_id"]: r for r in out["per_observable"]}
    assert rows["obs_A"]["sign_test_p"] == 0.002 and rows["obs_A"]["runs_test_p"] == 0.3
    assert rows["obs_B"]["low_power"] is True and rows["obs_B"]["runs_test_p"] is None


def test_none_when_no_diagnosis():
    # model-less / no-simulation run: get_fit_diagnosis returns a status-only payload
    assert _residual_structure_fields(None) is None
    assert _residual_structure_fields({}) is None
    assert _residual_structure_fields({"status": "no simulation yet",
                                       "per_observable": [], "worst_points": []}) == {
        "flagged_observables": [], "low_power": False, "per_observable": []}
