"""Wiring test for the hidden sigma gate in ``runner._judge``:

    success = bic_pass AND not sigma_implausible

``_noise_sigma_flag`` is unit-tested in isolation (test_result_fields); THIS test drives the WHOLE
``_judge`` with fakes to prove the flag is actually wired into the success verdict end-to-end — a
good BIC bought with an inflated (or collapsed) estimated sigma must score ``success=False`` even
though the BIC gate passes, and a rejectable BIC must fail regardless of a plausible sigma. The gate
is post-run and never shown to the agent, so a regression here would silently start counting gamed
runs as passes. Importing the harness runner pulls in ``mcp``, so skip where that dep is absent.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

pytest.importorskip("mcp", reason="harness-only dependency")
from _harness.runner import _NOISE_SIGMA_COLLAPSE_FRACTION, _ToolClock, _judge  # noqa: E402


def _sigma_entry(fitted, cap, frac):
    return {"fitted": fitted, "upper_bound_cap": cap, "data_std": 1.0, "fraction_of_data_std": frac}


_COLLAPSE_FRAC = _NOISE_SIGMA_COLLAPSE_FRACTION / 2     # well below the collapse floor
_PLAUSIBLE = {"sigma": _sigma_entry(0.08, 0.1, 0.4)}   # below its cap, above the collapse floor
_INFLATED = {"sigma": _sigma_entry(0.5, 0.1, 2.5)}     # past the data-driven cap (absorbs misfit)
_COLLAPSED = {"sigma": _sigma_entry(_COLLAPSE_FRAC, 0.5, _COLLAPSE_FRAC)}  # overfits the noise

# k = 3 estimated params (the estimate=0 row is the fixed noise sigma), n = 50 data points. The
# reference carries the SAME (nll, k, n) as the passing candidate, so a candidate nll of 100 gives
# ΔBIC = 0 (BIC passes) and an nll of 200 gives ΔBIC = +200 (rejectable).
_REF = {"nll": 100.0, "k": 3, "n": 50}
_PARAMS = [{"estimate": 1}, {"estimate": 1}, {"estimate": 1}, {"estimate": 0}]


class _Scenario:
    name = "wiring"
    bic_threshold = 10.0
    diagnostics = None

    def __init__(self, work: Path):
        # _judge writes result.json into `work` and prints it relative to example_dir.parent.
        self.example_dir = work / "ex"

    def reference(self):
        return dict(_REF)


class _Baseline:
    nll = 150.0
    species = {"A", "B"}


def _make_call(summary):
    """A fake async MCP `call`: canned per tool name. get_progress/current_version agree on v1 so
    _restore_best_checkpoint is a no-op (already on the best checkpoint); get_fit_summary carries
    the nll + noise_sigma_report that drive both gates; everything else is inert."""
    canned = {
        "get_progress": {"best_version": 1},
        "current_version": {"version": 1},
        "get_model_structure": {"reactions": [{"id": "r1"}], "species": ["A", "B"]},
        "get_fit_summary": summary,
        "get_parameters": _PARAMS,
        "get_fit_diagnosis": {},
        "diff_versions": {},
        "get_errors": [],
        "get_run_timing": {},
    }

    async def call(name, **kwargs):
        return canned.get(name, {})

    return call


def _run_judge(work: Path, nll: float, sigma_report):
    summary = {"neg_log_likelihood": nll, "n_data_points": 50, "chi2": 45.0,
               "noise_sigma_report": sigma_report}
    return asyncio.run(_judge(_Scenario(work), _make_call(summary), _Baseline(),
                              work, {}, _ToolClock(), time.monotonic()))


def test_good_bic_plausible_sigma_succeeds(tmp_path):
    code, result = _run_judge(tmp_path, 100.0, _PLAUSIBLE)
    assert result["delta_bic"] == pytest.approx(0.0)        # BIC gate passes
    assert result["noise_sigma_flag"]["inflated"] is False  # ... and the sigma is plausible
    assert result["success"] is True and code == 0


def test_good_bic_implausible_sigma_fails(tmp_path):
    # SAME passing BIC, but the estimated sigma is inflated -> the hidden gate vetoes the pass.
    code, result = _run_judge(tmp_path, 100.0, _INFLATED)
    assert result["delta_bic"] == pytest.approx(0.0)        # BIC still passes ...
    assert result["noise_sigma_flag"]["inflated"] is True   # ... but sigma is implausible ...
    assert result["success"] is False and code == 1         # ... so the run does NOT count.


def test_good_bic_collapsed_sigma_fails(tmp_path):
    # The OTHER implausible branch: same passing BIC, but the estimated sigma has collapsed
    # (overfitting the noise) -> the hidden gate must veto it too, not only the inflated case.
    code, result = _run_judge(tmp_path, 100.0, _COLLAPSED)
    assert result["delta_bic"] == pytest.approx(0.0)        # BIC still passes ...
    assert result["noise_sigma_flag"]["collapsed"] is True  # ... but sigma has collapsed ...
    assert result["success"] is False and code == 1         # ... so the run does NOT count.


def test_bad_bic_fails_despite_plausible_sigma(tmp_path):
    # The other arm of the AND: a plausible sigma cannot rescue a rejectable BIC.
    code, result = _run_judge(tmp_path, 200.0, _PLAUSIBLE)
    assert result["delta_bic"] > _Scenario(tmp_path).bic_threshold
    assert result["noise_sigma_flag"]["inflated"] is False
    assert result["success"] is False and code == 1
