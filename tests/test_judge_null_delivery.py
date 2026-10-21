"""The judge's two guards for a delivery it did not watch being built (issue #1).

The null arm's model is scored as shipped, so the judge must count the parameters the panel's
server would estimate for it (``estimated``, from ``runner.barebones._reconciled``) and must not
grade a candidate scored on other data than the reference: before, a delivery listing extra
measurements its model reproduced lowered its ΔBIC without bound. Driven through the whole ``_judge`` with
fakes, as in test_judge_sigma_gate.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

from gemot.runner import _judge, _ToolClock

#: The candidate matches the reference exactly at n = 50, so ΔBIC is 0 and the run passes.
_REF = {"nll": 100.0, "k": 2, "n": 50}
_PARAMS = [{"estimate": 1}, {"estimate": 1}, {"estimate": 0}]
_PLAUSIBLE = {
    "sigma": {"fitted": 0.08, "upper_bound_cap": 0.1, "data_std": 1.0, "fraction_of_data_std": 0.4}
}


class _Scenario:
    name = "null"
    bic_threshold = 10.0
    diagnostics = None

    def __init__(self, work: Path):
        self.example_dir = work / "ex"

    def reference(self):
        return dict(_REF)


class _Baseline:
    nll = None
    species: set = set()


def _run_judge(work: Path, *, n: int, estimated=None) -> dict:
    summary = {
        "neg_log_likelihood": 100.0,
        "n_data_points": n,
        "chi2": 45.0,
        "noise_sigma_report": _PLAUSIBLE,
    }
    canned = {
        "get_progress": {"best_version": 1},
        "describe_problem": {"current_version": 1, "model_authored": True},
        "get_model_structure": {"reactions": [{"id": "r1"}], "species": ["A"]},
        "get_fit_summary": summary,
        "get_parameters": _PARAMS,
    }

    async def call(name, **_):
        return canned.get(name, {})

    _, result = asyncio.run(
        _judge(
            _Scenario(work),
            call,
            _Baseline(),
            work,
            {},
            _ToolClock(),
            time.monotonic(),
            estimated=estimated,
        )
    )
    assert json.loads((work / "result.json").read_text())["success"] == result["success"]
    return result


def test_a_delivery_matching_the_reference_passes(tmp_path):
    result = _run_judge(tmp_path, n=50)
    assert result["success"] is True and result["n_mismatch"] is False
    assert result["k"] == 2 and result["delta_bic"] == 0.0


def test_a_candidate_scored_on_other_data_has_no_delta(tmp_path):
    """Ten extra points the model reproduces would make the candidate's BIC lower than the truth's;
    the Δ is withheld instead, under every criterion, and the run fails."""
    result = _run_judge(tmp_path, n=60)
    assert result["n_mismatch"] is True and result["success"] is False
    assert result["delta_bic"] is None and result["delta_criterion"] is None
    assert all(row["delta"] is None for row in result["criteria"].values())


def test_the_reconciled_estimates_are_the_nulls_k(tmp_path):
    """Two parameters the reconcile estimates beyond the delivered rows cost 2·ln(50) ≈ 7.8, as
    they would the panel."""
    result = _run_judge(tmp_path, n=50, estimated=["a", "b", "kx", "C_init"])
    assert result["k"] == 4 and result["k_estimated"] == 4
    assert abs(result["delta_bic"] - 2 * math.log(50)) < 1e-9
