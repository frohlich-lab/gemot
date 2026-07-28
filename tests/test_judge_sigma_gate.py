"""Wiring test for the hidden sigma gate in ``runner.judge._judge``:

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
import time
from pathlib import Path

import pytest
from petab_harness.runner import (  # noqa: E402
    _NOISE_SIGMA_COLLAPSE_FRACTION,
    _judge,
    _reference_sigma_flag,
    _ToolClock,
)


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
    """A fake async MCP `call`: canned per tool name. get_progress and describe_problem agree on
    v1 so _restore_best_checkpoint is a no-op (already on the best checkpoint); get_fit_summary
    carries the nll + noise_sigma_report that drive both gates; everything else is inert."""
    canned = {
        "get_progress": {"best_version": 1},
        # the HEAD version comes off describe_problem — there is no current_version tool
        "describe_problem": {"current_version": 1, "model_authored": True},
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


# --- Design A: judge the sigma against the TRUE model's own fitted sigma (#118) --------------
def _ref_flag(fitted, ref_sigma):
    return _reference_sigma_flag({"s": {"fitted": fitted}}, {"noise_sigma": {"s": ref_sigma}})


def test_reference_sigma_flag_accepts_a_sigma_near_the_truth():
    """A candidate whose sigma matches the reference fit's is plausible regardless of how it
    compares to the MARGINAL data std — the comparison #118 showed no fixed band can make."""
    f = _ref_flag(0.10, 0.10)
    assert (f["inflated"], f["collapsed"], f["max_ratio"], f["min_ratio"]) == (False, False, 1.0, 1.0)
    # The reference basis must emit the SAME keys as the data-std fallback, so a consumer of
    # result.json's noise_sigma_flag sees one shape regardless of which basis produced it.
    assert "max_fraction_of_data_std" in f and "min_fraction_of_data_std" in f
    for ratio in (0.5, 2.0):        # comfortably inside the tolerance factor
        f = _ref_flag(0.10 * ratio, 0.10)
        assert not f["inflated"] and not f["collapsed"], ratio


def test_reference_sigma_flag_catches_both_tails():
    from petab_mcp.constants import NOISE_SIGMA_REFERENCE_FACTOR as F
    inflated = _ref_flag(0.10 * (F * 2), 0.10)
    assert inflated["inflated"] and not inflated["collapsed"]
    collapsed = _ref_flag(0.10 / (F * 2), 0.10)
    assert collapsed["collapsed"] and not collapsed["inflated"]


def test_reference_sigma_flag_is_none_without_recorded_reference_sigmas():
    """An older reference.json carries only {nll,k,n}; the judge must fall back to the data-std
    band rather than silently passing everything."""
    assert _reference_sigma_flag({"s": {"fitted": 0.1}}, {"nll": 1.0, "k": 2, "n": 3}) is None
    assert _reference_sigma_flag({"s": {"fitted": 0.1}}, {"noise_sigma": {}}) is None
    assert _reference_sigma_flag({}, {"noise_sigma": {"s": 0.1}}) is None


def test_reference_sigma_pairs_a_lone_sigma_across_an_id_mismatch():
    """The reference is fit on the PRISTINE collection problem, whose ids can differ from the
    materialized ones (perelson ships `sd_viral_load`; the collection calls it
    `sd_task0_model0_perelson1_V`). With exactly ONE sigma on each side the pairing is unambiguous,
    so match positionally rather than silently losing the comparison."""
    f = _reference_sigma_flag({"sd_viral_load": {"fitted": 0.12}},
                              {"noise_sigma": {"sd_task0_model0_perelson1_V": 0.1228}})
    assert f is not None and not f["inflated"] and not f["collapsed"]


def test_reference_sigma_does_not_guess_when_several_sigmas_mismatch():
    """More than one sigma on either side -> stay id-matched: a WRONG pairing is worse than none."""
    assert _reference_sigma_flag(
        {"a": {"fitted": 0.1}, "b": {"fitted": 0.2}},
        {"noise_sigma": {"x": 0.1, "y": 99.0}}) is None


def test_reference_gate_judges_effective_sigma_when_available():
    """A composite sd (scale*sigma) is only comparable across models as the PRODUCT: a candidate at
    a different amplitude trades scale against sigma and reaches an identical sd with a raw sigma
    several-fold off. Judged raw it fails for matching the truth; judged effective it passes."""
    # truth: scale 4.0 * sigma 0.25 = 1.0. Candidate: scale 0.8 * sigma 1.25 = 1.0 (same sd).
    report = {"sd_o": {"fitted": 1.25, "effective_sigma": 1.0}}
    reference = {"noise_sigma": {"sd_o": 0.25}, "noise_sigma_effective": {"sd_o": 1.0}}
    flag = _reference_sigma_flag(report, reference)
    assert flag is not None and not flag["inflated"] and not flag["collapsed"]
    assert flag["max_ratio"] == pytest.approx(1.0)
    # Without the effective map the same pair is judged raw (1.25/0.25 = 5x) and IS flagged.
    raw_only = _reference_sigma_flag(report, {"noise_sigma": {"sd_o": 0.25}})
    assert raw_only is not None and raw_only["inflated"]
