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
from _harness.runner import (  # noqa: E402
    _NOISE_SIGMA_COLLAPSE_FRACTION,
    _noise_sigma_flag,
    _residual_structure_fields,
)


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


# --- noise-sigma plausibility flag (the hidden success gate) ------------------------------
def _sigma_entry(fitted, cap, frac):
    return {"fitted": fitted, "upper_bound_cap": cap, "data_std": 1.0, "fraction_of_data_std": frac}


def test_noise_sigma_flag_none_when_no_estimated_sigma():
    assert _noise_sigma_flag(None) is None
    assert _noise_sigma_flag({}) is None


def test_noise_sigma_flag_plausible_sigma_not_flagged():
    # fitted below its cap and well above the collapse floor -> neither inflated nor collapsed.
    f = _noise_sigma_flag({"sigma": _sigma_entry(0.08, 0.1, 0.4)})
    assert f["inflated"] is False and f["collapsed"] is False


def test_noise_sigma_flag_detects_inflation():
    # fitted above the data-driven cap -> inflated (absorbing misfit).
    f = _noise_sigma_flag({"sigma": _sigma_entry(0.5, 0.1, 2.5)})
    assert f["inflated"] is True and f["collapsed"] is False
    assert f["max_fraction_of_data_std"] == 2.5


def test_noise_sigma_flag_detects_collapse():
    # fraction below the collapse floor -> collapsed (overfitting the noise).
    frac = _NOISE_SIGMA_COLLAPSE_FRACTION / 2
    f = _noise_sigma_flag({"sigma": _sigma_entry(frac, 0.5, frac)})
    assert f["collapsed"] is True and f["inflated"] is False
    assert f["min_fraction_of_data_std"] == frac


def test_noise_sigma_flag_detects_both_inflated_and_collapsed():
    # one inflated + one collapsed estimated sigma in the same run -> BOTH flags set (the verdict
    # must then report both, not just one).
    f = _noise_sigma_flag({
        "sig_hi": _sigma_entry(0.5, 0.1, 2.5),                                  # inflated
        "sig_lo": _sigma_entry(0.01, 0.5, _NOISE_SIGMA_COLLAPSE_FRACTION / 2),  # collapsed
    })
    assert f["inflated"] is True and f["collapsed"] is True
