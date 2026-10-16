"""The reference-free judge: a model-free data package graded by the absolute goodness-of-fit test
its scenario declares, since there is no true model to take a Δcriterion against.

Two layers. ``scoring._reference_free_adequacy`` is a pure function and is pinned branch by branch,
for each of the three declarable tests. ``runner.judge._judge`` is then driven end to end with a
fake ``call`` (the shape ``test_judge_sigma_gate`` uses), once per verdict it can reach in
reference-free mode, plus the check that a reference-based scenario is judged exactly as before.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time

import pytest

from gemot import metrics, tracing
from gemot import scenario as S
from gemot.paths import validation_measurements_path
from gemot.runner import REFERENCE_FREE_GOF_ALPHA, _judge, _reference_free_adequacy, _ToolClock

#: A fit that estimated no noise sigma: the problem's own noise model is fully specified.
_KNOWN = {}
#: A fit that estimated one, plausibly (under its data-driven cap, above the collapse floor).
_PLAUSIBLE = {
    "sd_obs": {"fitted": 0.08, "upper_bound_cap": 0.1, "data_std": 1.0, "fraction_of_data_std": 0.4}
}
#: ...and one inflated past its cap, absorbing misfit.
_INFLATED = {
    "sd_obs": {"fitted": 0.5, "upper_bound_cap": 0.1, "data_std": 1.0, "fraction_of_data_std": 2.5}
}


def _rolled(*, flagged=(), low_power=False, lof=None, read=True):
    """The judge's residual-structure block, as ``_residual_structure_fields`` writes it."""
    if not read:
        return {"rollup_status": "unavailable", "rows_status": "unavailable", "per_observable": []}
    out = {
        "rollup_status": "read",
        "rows_status": "read",
        "per_observable": [],
        "flagged_observables": list(flagged),
        "low_power": low_power,
    }
    if lof is not None:
        out["replicate_lack_of_fit"] = lof
    return out


def _lof(status, **extra):
    """The replicate lack-of-fit verdict as it rides on a fit (``data_access._lof_reported``)."""
    return {
        "available": status in ("flagged", "pass"),
        "status": status,
        "n_points": 12,
        "n_flagged": 0,
        **extra,
    }


# --------------------------------------------------------------------------- chi2
@pytest.mark.parametrize(
    "p, passed", [(0.5, True), (REFERENCE_FREE_GOF_ALPHA, True), (0.001, False)]
)
def test_chi2_with_fully_known_noise_is_the_textbook_test(p, passed):
    out = _reference_free_adequacy(S.ADEQUACY_CHI2, _KNOWN, p, None)
    assert (out["test"], out["basis"], out["passed"]) == ("chi2", "chi2", passed)
    assert out["chi2_gof_pvalue"] == p and out["alpha"] == REFERENCE_FREE_GOF_ALPHA


def test_chi2_without_a_p_value_is_unassessable():
    out = _reference_free_adequacy(S.ADEQUACY_CHI2, _KNOWN, None, None)
    assert out["basis"] == "chi2" and out["passed"] is None


def test_chi2_is_invalid_once_the_fit_estimated_a_sigma():
    """An ML sigma pins chi2 near n, so the declared test would pass ANY model. It is reported as
    invalid for this fit and never answered by another test instead — not even one whose passing
    evidence is right there in the same payload."""
    out = _reference_free_adequacy(
        S.ADEQUACY_CHI2, _PLAUSIBLE, 0.9, _rolled(lof=_lof("pass", lof_df=8))
    )
    assert out["passed"] is None and out["basis"] is None
    assert "invalid" in out["reason"] and "sd_obs" in out["reason"]
    assert out["lof_status"] is None


def test_an_unreadable_noise_report_is_not_known_noise():
    """``None`` means the report could not be read, which says nothing about whether a sigma was
    estimated; only an EMPTY report is known noise."""
    out = _reference_free_adequacy(S.ADEQUACY_CHI2, None, 0.9, None)
    assert out["passed"] is None and out["basis"] is None


# --------------------------------------------------------------------------- replicate_lof
def test_a_replicate_lack_of_fit_flag_fails_at_any_df():
    out = _reference_free_adequacy(
        S.ADEQUACY_REPLICATE_LOF,
        _PLAUSIBLE,
        None,
        _rolled(lof=_lof("flagged", n_flagged=3, lof_df=1)),
    )
    assert out["passed"] is False and out["basis"] == "replicate_lof"
    assert (out["lof_status"], out["lof_df"], out["lof_weak"]) == ("flagged", 1, False)
    assert "3 of 12" in out["reason"]


def test_a_weak_replicate_lack_of_fit_pass_still_passes_and_says_so():
    """A pass on few lack-of-fit df is weaker evidence, not a failure: every other gate that reads
    this arm lets it through, and the bar must not be stricter on a data package than elsewhere."""
    solid = _reference_free_adequacy(
        S.ADEQUACY_REPLICATE_LOF, _PLAUSIBLE, None, _rolled(lof=_lof("pass", lof_df=40))
    )
    assert solid["passed"] is True and solid["lof_weak"] is False
    weak = _reference_free_adequacy(
        S.ADEQUACY_REPLICATE_LOF, _PLAUSIBLE, None, _rolled(lof=_lof("pass", lof_df=1, weak=True))
    )
    assert weak["passed"] is True and weak["lof_weak"] is True and "weak" in weak["reason"]


def test_a_partial_replicate_lack_of_fit_pass_records_the_untested_family():
    out = _reference_free_adequacy(
        S.ADEQUACY_REPLICATE_LOF,
        _PLAUSIBLE,
        None,
        _rolled(lof=_lof("pass", lof_df=8, untested_status="too_few_usable_points")),
    )
    assert out["passed"] is True
    assert out["lof_untested_status"] == "too_few_usable_points"
    assert "too_few_usable_points" in out["reason"]


@pytest.mark.parametrize(
    "status",
    [
        "no_replicated_points",
        "replicates_too_thin",
        "too_few_usable_points",
        "vacuous",
        "no_noise_parameter",
        "unavailable",
    ],
)
def test_a_replicate_lack_of_fit_arm_that_could_not_run_is_unassessable(status):
    out = _reference_free_adequacy(
        S.ADEQUACY_REPLICATE_LOF, _PLAUSIBLE, None, _rolled(lof=_lof(status))
    )
    assert out["passed"] is None and out["lof_status"] == status
    # no df was counted, so "weak" has nothing to qualify (metrics._flatten_replicate_lof's rule)
    assert out["lof_weak"] is None and status in out["reason"]


@pytest.mark.parametrize("rolled", [None, _rolled(read=False), _rolled()])
def test_no_replicate_lack_of_fit_verdict_is_unassessable(rolled):
    out = _reference_free_adequacy(S.ADEQUACY_REPLICATE_LOF, _PLAUSIBLE, None, rolled)
    assert out["passed"] is None and out["basis"] is None


# --------------------------------------------------------------------------- residual_structure
def test_residual_structure_fails_and_names_the_observables():
    out = _reference_free_adequacy(
        S.ADEQUACY_RESIDUAL_STRUCTURE, _PLAUSIBLE, None, _rolled(flagged=["obs_B", "obs_A"])
    )
    assert out["passed"] is False and out["flagged_observables"] == ["obs_B", "obs_A"]
    assert "obs_A" in out["reason"]


def test_residual_structure_unflagged_on_underpowered_tests_is_unassessable():
    """Nothing flagged by a test that could not have flagged anything is no evidence of adequacy."""
    out = _reference_free_adequacy(
        S.ADEQUACY_RESIDUAL_STRUCTURE, _PLAUSIBLE, None, _rolled(low_power=True)
    )
    assert out["passed"] is None and out["basis"] == "residual_structure"
    assert out["flagged_observables"] == []


def test_residual_structure_unflagged_and_powered_passes():
    out = _reference_free_adequacy(S.ADEQUACY_RESIDUAL_STRUCTURE, _PLAUSIBLE, None, _rolled())
    assert out["passed"] is True


@pytest.mark.parametrize("rolled", [None, _rolled(read=False)])
def test_an_unread_residual_structure_verdict_is_unassessable(rolled):
    """`_residual_structure_fields` OMITS the verdict keys when it did not read them; an absent
    `flagged_observables` must not read as "nothing flagged"."""
    out = _reference_free_adequacy(S.ADEQUACY_RESIDUAL_STRUCTURE, _PLAUSIBLE, None, rolled)
    assert out["passed"] is None and out["basis"] is None


@pytest.mark.parametrize(
    "test, rolled",
    [
        (S.ADEQUACY_REPLICATE_LOF, _rolled(lof=_lof("pass", lof_df=8))),
        (S.ADEQUACY_RESIDUAL_STRUCTURE, _rolled()),
    ],
)
def test_the_sigma_free_tests_do_not_depend_on_the_noise_report(test, rolled):
    known = _reference_free_adequacy(test, _KNOWN, 0.9, rolled)
    estimated = _reference_free_adequacy(test, _PLAUSIBLE, 0.9, rolled)
    assert known == estimated and known["passed"] is True


def test_an_unknown_test_is_unassessable():
    out = _reference_free_adequacy("bic", _KNOWN, 0.9, _rolled())
    assert out["passed"] is None and out["basis"] is None


# --------------------------------------------------------------------------- the judge, end to end
_REF = {"nll": 100.0, "k": 3, "n": 50}


async def _no_fix(call, n_starts):  # pragma: no cover - never driven here
    return None


class _Baseline:
    nll = None  # zero-shot: nothing shipped to score
    species = {"A", "B"}


def _data_package(work, adequacy):
    return S.Scenario(
        name="pkg",
        problem_dir=work / "ex" / "problem",
        scripted_fix=_no_fix,
        diagnostics=None,
        reference_free=True,
        adequacy=adequacy,
    )


def _summary(*, nll=100.0, chi2=47.0, report=None, rollup=None):
    """k = 3, n = 50, so chi2 = 47 sits at its 47 dof (p ~ 0.5) and chi2 = 200 is rejected."""
    out = {
        "neg_log_likelihood": nll,
        "n_data_points": 50,
        "chi2": chi2,
        "noise_sigma_report": _KNOWN if report is None else report,
    }
    if rollup is not None:
        out["residual_structure"] = rollup
    return out


def _run(scenario, work, summary, **tools):
    canned = {
        "get_progress": {"best_version": 1},
        "describe_problem": {"current_version": 1, "model_authored": True},
        "get_model_structure": {"reactions": [{"id": "r1"}], "species": ["A", "B"]},
        "get_fit_summary": summary,
        "get_parameters": [{"estimate": 1}] * 3,
        "get_fit_diagnosis": {},
        "diff_versions": {},
        "get_errors": [],
        "get_run_timing": {},
        **tools,
    }

    async def call(name, **kwargs):
        return canned.get(name, {})

    work.mkdir(parents=True, exist_ok=True)
    return asyncio.run(
        _judge(scenario, call, _Baseline(), work, {}, _ToolClock(), time.monotonic())
    )


def test_an_adequate_reference_free_fit_succeeds_with_no_reference_numbers(tmp_path, capsys):
    code, result = _run(_data_package(tmp_path, S.ADEQUACY_CHI2), tmp_path, _summary())
    assert result["success"] is True and code == 0
    assert result["judge_mode"] == "reference_free"
    assert (result["adequacy"]["test"], result["adequacy"]["passed"]) == ("chi2", True)
    for key in (
        "reference",
        "bic_reference",
        "delta_bic",
        "criterion_reference",
        "delta_criterion",
    ):
        assert result[key] is None, key
    assert all(row["reference"] is None for row in result["criteria"].values())
    # the candidate's own scores are still recorded — they are what ranks runs on this problem
    assert result["bic"] is not None and result["criterion_value"] is not None
    assert result["bic_threshold"] is None  # there is no Δ for it to bound
    assert result["noise_sigma_basis"] == "data_std"  # no reference sigma to compare against
    out = capsys.readouterr().out
    assert "SUCCESS: adequate by its declared chi2 test" in out
    assert "true-model" not in out and "reference missing" not in out
    assert json.loads((tmp_path / "result.json").read_text())["judge_mode"] == "reference_free"


def test_a_reference_free_run_without_a_valid_fit_fails(tmp_path, capsys):
    code, result = _run(
        _data_package(tmp_path, S.ADEQUACY_CHI2), tmp_path, _summary(nll=None, chi2=None)
    )
    assert result["success"] is False and code == 1 and result["final_nll"] is None
    assert "NOT FIXED: no valid fit" in capsys.readouterr().out


def test_an_implausible_sigma_vetoes_an_adequate_reference_free_fit(tmp_path, capsys):
    """The integrity gate survives the switch of bar: the replicate arm passes, but the estimated
    sigma is inflated past its data-driven cap — the same veto a ΔBIC pass gets."""
    rollup = {
        "flagged_observables": [],
        "low_power": False,
        "replicate_lack_of_fit": _lof("pass", lof_df=9),
    }
    code, result = _run(
        _data_package(tmp_path, S.ADEQUACY_REPLICATE_LOF),
        tmp_path,
        _summary(report=_INFLATED, rollup=rollup),
    )
    assert result["adequacy"]["passed"] is True
    assert result["noise_sigma_flag"]["inflated"] is True
    assert result["success"] is False and code == 1
    assert "noise sigma is inflated" in capsys.readouterr().out


def test_an_inadequate_reference_free_fit_fails(tmp_path, capsys):
    code, result = _run(_data_package(tmp_path, S.ADEQUACY_CHI2), tmp_path, _summary(chi2=200.0))
    assert result["adequacy"]["passed"] is False
    assert result["success"] is False and code == 1
    assert "NOT FIXED: inadequate by its declared chi2 test" in capsys.readouterr().out


def test_unassessable_adequacy_is_not_a_pass(tmp_path, capsys):
    """chi2 declared, but this fit estimated a (plausible) sigma: the test cannot speak, and a fit
    nobody could assess is not counted as one that passed — as a missing reference.json is not."""
    code, result = _run(
        _data_package(tmp_path, S.ADEQUACY_CHI2), tmp_path, _summary(report=_PLAUSIBLE)
    )
    assert result["adequacy"]["passed"] is None
    assert result["noise_sigma_flag"]["inflated"] is False
    assert result["success"] is False and code == 1
    assert "NOT FIXED: adequacy not assessable" in capsys.readouterr().out


def test_reference_free_held_out_is_absolute_and_report_only(tmp_path, capsys):
    """With no reference there is nothing to subtract, so the held-out numbers are printed as they
    are — and, as on every other example, they never enter the verdict."""
    scenario = _data_package(tmp_path, S.ADEQUACY_CHI2)
    held_out = validation_measurements_path(scenario.example_dir)
    held_out.parent.mkdir(parents=True)
    held_out.write_text("observableId\tsimulationConditionId\ttime\tmeasurement\n")
    payload = {"status": "ok", "nll": 900.0, "n": 12, "chi2": 400.0, "n_conditions": 4}
    code, result = _run(scenario, tmp_path, _summary(), evaluate_validation_data=payload)
    assert result["success"] is True
    assert result["validation"]["nll"] == 900.0 and "delta_nll" not in result["validation"]
    line = next(x for x in capsys.readouterr().out.splitlines() if "held-out (validation)" in x)
    assert "nll 900" in line and "chi2 400" in line and "12 point(s)" in line
    assert "true-model" not in line


class _DuckScenario:
    """The shape every older judge test hands in: no reference-free fields at all."""

    name = "wiring"
    bic_threshold = 10.0
    diagnostics = None

    def __init__(self, work):
        self.example_dir = work / "ex"

    def reference(self):
        return dict(_REF)


def test_a_reference_based_scenario_is_judged_exactly_as_before(tmp_path):
    """None of the reference-free arm may leak into the ΔBIC gate. A real Scenario with a reference
    and the duck-typed one (which never had the new fields) must produce the same result, and that
    result is the ΔBIC verdict with `judge_mode: reference` and no adequacy block."""
    real = S.Scenario(
        name="wiring",
        problem_dir=tmp_path / "real" / "ex" / "problem",
        scripted_fix=_no_fix,
        diagnostics=None,
    )
    real.problem_dir.mkdir(parents=True)
    (real.problem_dir / "reference.json").write_text(json.dumps(_REF))
    summary = _summary(report=_PLAUSIBLE)
    _, a = _run(real, tmp_path / "real", summary)
    _, b = _run(_DuckScenario(tmp_path / "duck"), tmp_path / "duck", summary)
    volatile = {"wall_seconds", "compute_seconds", "llm_seconds", "run_ended_at"}
    assert {k: v for k, v in a.items() if k not in volatile} == {
        k: v for k, v in b.items() if k not in volatile
    }
    assert a["judge_mode"] == "reference" and a["adequacy"] is None
    assert a["delta_bic"] == pytest.approx(0.0) and a["bic_threshold"] == 10.0
    assert a["success"] is True


def test_a_reference_based_scenario_missing_its_reference_still_fails(tmp_path, capsys):
    """The safety property, end to end: this fit would PASS the reference-free chi2 bar, but its
    scenario never declared reference-free — so a deleted reference.json still fails it."""
    scenario = S.Scenario(
        name="wiring",
        problem_dir=tmp_path / "ex" / "problem",
        scripted_fix=_no_fix,
        diagnostics=None,
    )
    code, result = _run(scenario, tmp_path, _summary())
    assert result["success"] is False and code == 1
    assert result["judge_mode"] == "reference" and result["adequacy"] is None
    assert "could not evaluate" in capsys.readouterr().out


# --------------------------------------------------------------------------- sweeps
def test_the_mode_and_the_adequacy_verdict_reach_the_history_csv_and_the_span(
    tmp_path, monkeypatch
):
    """Flat columns, because a sweep groups by flat columns: without `judge_mode` beside `success`
    a reference-free pass and a ΔBIC pass land in one pooled rate."""
    result = {
        "scenario": "lun",
        "success": True,
        "judge_mode": "reference_free",
        "adequacy": {"test": "chi2", "passed": True, "reason": "chi2 GoF p=0.5 >= 0.05"},
    }
    path = metrics.append_history([metrics.build_row(result)], tmp_path / "h.csv")
    with path.open(newline="") as fh:
        row = next(csv.DictReader(fh))
    assert (row["judge_mode"], row["adequacy_test"], row["adequacy_passed"]) == (
        "reference_free",
        "chi2",
        "True",
    )
    # a reference-judged run contributes no adequacy columns at all
    assert metrics.flatten_adequacy({"scenario": "boehm", "adequacy": None}) == {}

    stamped: dict = {}
    monkeypatch.setattr(tracing, "_set_attrs", lambda span, attrs: stamped.update(attrs))
    tracing.record_run_result(object(), result)
    assert stamped["petab.judge_mode"] == "reference_free"
    assert stamped["petab.adequacy_test"] == "chi2" and stamped["petab.adequacy_passed"] is True


def test_batch_summaries_never_pool_the_two_kinds_of_pass():
    reference = {"scenario": "boehm", "success": True}  # predates judge_mode: reference-judged
    free = [
        {"scenario": "lun", "success": False, "judge_mode": "reference_free"},
        {"scenario": "lun", "success": True, "judge_mode": "reference_free"},
    ]
    out = metrics.summarize([reference, *free])
    assert "overall (reference): 1/1 passed" in out
    assert "overall (reference_free): 1/2 passed" in out
    assert "overall: " not in out
    # one mode reads exactly as it always did
    assert "overall: 1/1 passed" in metrics.summarize([reference])

    # The per-scenario rate too: ONE scenario under both modes is two rates, never one.
    same = [
        {"scenario": "lun", "success": True},
        {"scenario": "lun", "success": True},
        {"scenario": "lun", "success": False, "judge_mode": "reference_free"},
        {"scenario": "lun", "success": False, "judge_mode": "reference_free"},
    ]
    rates = metrics.summarize(same).split("success rate (this batch):")[1]
    assert "lun (reference)" in rates and "2/2" in rates
    assert "lun (reference_free)" in rates and "0/2" in rates
    assert "2/4" not in rates
    assert "  lun              2/2" in metrics.summarize(same[:2])  # one mode: unlabelled

    from gemot import experiments

    rows = [
        {"example": "boehm", "model": "m", "temperature": 0.0, "n_ops": 3, "n_fits": 3, **r}
        for r in ({"success": True}, {"success": False, "judge_mode": "reference_free"})
    ]
    out = experiments.summarize_sweep(rows)
    assert "overall (reference): 1/1 runs passed" in out
    assert "overall (reference_free): 0/1 runs passed" in out
    # The per-cell table: the same (example, model, temperature) under two modes is two cells.
    per_cell = out.split("Pass rate by model")[0]
    assert "× judge mode" in per_cell
    cell_rows = [line for line in per_cell.splitlines() if line.startswith("boehm")]
    assert len(cell_rows) == 2
    assert any("reference_free" in line and "0/1" in line for line in cell_rows)
    assert any("reference " in line and "1/1" in line for line in cell_rows)
    assert "1/2" not in per_cell
    # one mode: no mode column, as before
    assert "judge mode" not in experiments.summarize_sweep(rows[:1])
