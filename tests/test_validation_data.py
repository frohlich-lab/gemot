"""The fixed-θ HELD-OUT check, from gemot's side: the judge's projection of the score into the
verdict, the history CSV and the run span, and the guards that refuse a held-out problem before
the server is ever asked.

The scorer itself (:func:`petab_mcp.validation_data.score_problem`) and gemot's materialisation
of a held-out set on top of it read the server's held-out format, so they are tested against the
real server in ``tests/server/test_validation_data.py``.
"""

from __future__ import annotations


import pandas as pd
import pytest

from gemot import benchmark
from gemot.runner.scoring import _validation_fields


def _problem(measurements, *, observables=("obs_A",), conditions=("c_train", "c_test")):
    """A minimal petab problem stand-in with just the tables the guards read."""
    import petab.v1 as petab

    return petab.Problem(
        model=None,
        measurement_df=measurements,
        observable_df=pd.DataFrame(
            {
                "observableFormula": ["x"] * len(observables),
                "noiseFormula": ["1"] * len(observables),
            },
            index=pd.Index(list(observables), name="observableId"),
        ),
        condition_df=pd.DataFrame(
            {"dose": [0.0] * len(conditions)}, index=pd.Index(list(conditions), name="conditionId")
        ),
        parameter_df=pd.DataFrame(
            {
                "parameterScale": ["lin"],
                "lowerBound": [0.0],
                "upperBound": [1.0],
                "nominalValue": [0.5],
                "estimate": [1],
            },
            index=pd.Index(["k"], name="parameterId"),
        ),
    )


def _rows(condition, times=(0.0, 1.0), observable="obs_A"):
    return pd.DataFrame(
        {
            "observableId": [observable] * len(times),
            "simulationConditionId": [condition] * len(times),
            "time": list(times),
            "measurement": [1.0] * len(times),
        }
    )


# --------------------------------------------------------------------- the scoring projection
_REF = {
    "nll": -66.58,
    "k": 19,
    "n": 144,
    "validation": {"status": "ok", "nll": 500.0, "n": 276, "chi2": 1000.0},
}


_JUDGE_REF = {
    "nll": 100.0,
    "k": 3,
    "n": 50,
    "validation": {"status": "ok", "nll": 400.0, "n": 12, "chi2": 90.0},
}


_RAISES = object()


# ------------------------------------------------------------------- the agent cannot see it
def test_the_validation_tool_is_not_exposed_to_the_panel():
    """Two things have to stay out of the agent's context for the held-out score to mean anything:
    the data and the SCORE. The score would be a fitness signal — read it, edit, read it again, and
    the held-out set is training data. ``ALLOWED_TOOLS`` is a strict allowlist, so absence from it is
    the enforcement; this pins that it stays absent, since adding a tool to that set is a one-line
    change someone could make while tidying."""
    pytest.importorskip("langchain_core", reason="agent stack not installed")
    from gemot.agent.tools import ALLOWED_TOOLS

    assert "evaluate_validation_data" not in ALLOWED_TOOLS


def test_a_held_out_problem_that_changed_the_model_tables_is_refused(monkeypatch, tmp_path):
    """A held-out set is the SAME problem's data under unseen conditions. If its condition or
    observable table differs, the two scores are computed over two different models and their
    difference means nothing — so this refuses rather than scoring it."""
    train = _rows("c_train")
    published = _problem(train)
    # same measurements-are-disjoint story, but the held-out problem declares an extra condition
    held = _problem(_rows("c_test"), conditions=("c_train", "c_test", "c_extra"))

    monkeypatch.setattr(benchmark, "benchmark_dir", lambda *a, **k: tmp_path)
    (tmp_path / "held_out.yaml").write_text("format_version: 1\n")
    import petab.v1 as petab

    monkeypatch.setattr(petab.Problem, "from_yaml", staticmethod(lambda *a, **k: held))

    with pytest.raises(ValueError, match="condition table differs"):
        benchmark._validation_measurement_df(
            name="X",
            validation_yaml="held_out.yaml",
            collection=None,
            published=published,
            training_measurement_df=train,
        )


def test_an_absent_validation_yaml_names_itself(monkeypatch, tmp_path):
    """A typo'd path must say which path, at setup time. Falling through to "no held-out data" would
    leave the example silently unvalidated for as long as nobody read the setup log."""
    monkeypatch.setattr(benchmark, "benchmark_dir", lambda *a, **k: tmp_path)
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        benchmark._validation_measurement_df(
            name="X",
            validation_yaml="nope.yaml",
            collection=None,
            published=_problem(_rows("c_train")),
            training_measurement_df=_rows("c_train"),
        )


def test_the_held_out_table_lands_outside_the_problem_directory():
    """Structural, not incidental: the runner copies ``problem/`` into ``work/`` and points the MCP
    server at it, so a held-out table inside ``problem/`` would sit in the tree the agent's server
    reads. The path helper is what every writer and reader goes through, so pin it."""
    path = benchmark.validation_measurements_path(benchmark.Path("/tmp/ex"))
    assert path.parent.name == benchmark.VALIDATION_DIR
    assert path.parent.name != "problem"
    assert path.name == benchmark.VALIDATION_MEASUREMENTS


def test_delta_is_candidate_minus_reference_and_positive_means_worse():
    out = _validation_fields({"status": "ok", "nll": 560.0, "n": 276, "chi2": None}, _REF)
    assert out["delta_nll"] == pytest.approx(60.0)
    assert out["reference_nll"] == 500.0 and out["n"] == 276


def test_a_held_out_n_that_disagrees_with_the_reference_withholds_the_difference():
    """The two sides are computed at different times from different files — the reference at
    ``setup.py --refit``, the candidate at run time. A held-out set that changed in between would
    otherwise yield a plausible difference of two likelihoods over different data."""
    out = _validation_fields({"status": "ok", "nll": 560.0, "n": 200, "chi2": None}, _REF)
    assert out["n_mismatch"] is True
    assert "delta_nll" not in out


def test_a_model_that_cannot_integrate_the_unseen_conditions_keeps_its_own_status():
    """`non_finite` is a RESULT about the model — the ODEs blew up under an EGF pulse — not an
    infrastructure gap. Folding it into a null nll would make it indistinguishable from an example
    that ships no held-out data at all."""
    out = _validation_fields({"status": "non_finite", "reason": "could not integrate"}, _REF)
    assert out["status"] == "non_finite" and out["reason"] == "could not integrate"
    assert "delta_nll" not in out and out["reference_nll"] == 500.0


def test_no_reference_block_and_no_payload_is_no_check_at_all():
    """An example with no held-out data must produce None, so `val_status` stays EMPTY in the
    history CSV rather than claiming a check ran."""
    assert _validation_fields(None, {"nll": 1.0, "k": 2, "n": 10}) is None
    assert _validation_fields({}, {}) is None


def test_a_missing_tool_on_an_example_that_HAS_a_bar_is_reported_as_unavailable():
    """The other direction: the reference has a held-out bar but the run produced no score. That is
    a broken run, not an example without validation, and it must not read as one."""
    out = _validation_fields(None, _REF)
    assert out["status"] == "unavailable" and out["reference_nll"] == 500.0


def test_the_held_out_chi2_is_carried_for_both_sides_and_no_p_value_is_invented():
    """The truth's own held-out reduced chi2 on fujita is 36.5, so an absolute goodness-of-fit test
    against the training noise model rejects the TRUE model too and every run would record p = 0.
    Both chi2 values are carried instead, so the candidate reads against the truth on one scale."""
    out = _validation_fields({"status": "ok", "nll": 500.0, "n": 276, "chi2": 4000.0}, _REF)
    assert out["chi2"] == 4000.0 and out["reference_chi2"] == 1000.0
    assert "chi2_gof_pvalue" not in out and "chi2_dof" not in out


def test_a_reference_block_that_does_not_claim_success_is_not_reported_as_ok():
    """`status` is carried to separate "the refit's held-out scoring FAILED" from "no bar exists".
    Defaulting a silent block to `ok` would fold a third case — "the answer key does not say" —
    into the affirmative one, and then read its `nll` on the strength of that default."""
    silent = {"nll": 1.0, "k": 2, "n": 10, "validation": {"nll": 400.0, "n": 12}}
    out = _validation_fields({"status": "ok", "nll": 500.0, "n": 12, "chi2": 1.0}, silent)
    assert out["reference_status"] is None
    assert out["reference_nll"] is None and "delta_nll" not in out


# ------------------------------------------------------------------------- the history CSV
def test_the_history_csv_has_columns_for_the_held_out_score(tmp_path):
    """result.json carrying the block is not the same as a sweep being able to read it:
    `DictWriter(extrasaction="ignore")` drops a nested dict silently — how the edit-reject metric
    stayed invisible for a release."""
    from gemot import metrics

    result = {
        "scenario": "fujita",
        "validation": {
            "status": "ok",
            "nll": 560.0,
            "reference_nll": 500.0,
            "delta_nll": 60.0,
            "n": 276,
            "chi2": 300.0,
            "reference_chi2": 250.0,
        },
    }
    path = metrics.append_history([metrics.build_row(result)], tmp_path / "h.csv")
    row = dict(zip(*(line.split(",") for line in path.read_text().splitlines()[:2])))
    assert row["val_status"] == "ok"
    assert row["val_delta_nll"] == "60.0" and row["val_n"] == "276"
    assert row["val_reference_nll"] == "500.0"

    # An example with no held-out data leaves every column EMPTY rather than fabricating a status.
    p2 = metrics.append_history([metrics.build_row({"scenario": "boehm"})], tmp_path / "h2.csv")
    row2 = dict(zip(*(line.split(",") for line in p2.read_text().splitlines()[:2])))
    assert row2["val_status"] == "" and row2["val_delta_nll"] == ""

    # ...and a status without a usable difference writes the status but NOT a 0.0 delta, which a
    # sweep would average as "predicts exactly as well as the truth".
    p3 = metrics.append_history(
        [
            metrics.build_row(
                {"scenario": "fujita", "validation": {"status": "non_finite", "n": 276}}
            )
        ],
        tmp_path / "h3.csv",
    )
    row3 = dict(zip(*(line.split(",") for line in p3.read_text().splitlines()[:2])))
    assert row3["val_status"] == "non_finite" and row3["val_delta_nll"] == ""


class _Scenario:
    name = "validation-wiring"
    bic_threshold = 10.0
    diagnostics = None

    def __init__(self, work, reference=None):
        self.example_dir = work / "ex"
        self._reference = reference if reference is not None else dict(_JUDGE_REF)

    def reference(self):
        return dict(self._reference)


class _Baseline:
    nll = 150.0
    species = {"A", "B"}


def _run_judge(work, *, has_held_out, payload, reference=None, nll=100.0):
    import asyncio
    import time

    from gemot import benchmark as bm
    from gemot.runner import _judge, _ToolClock

    scenario = _Scenario(work, reference)
    if has_held_out:
        p = bm.validation_measurements_path(scenario.example_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("observableId\tsimulationConditionId\ttime\tmeasurement\n")

    canned = {
        "get_progress": {"best_version": 1},
        "describe_problem": {"current_version": 1, "model_authored": True},
        "get_model_structure": {"reactions": [{"id": "r1"}], "species": ["A", "B"]},
        "get_fit_summary": {"neg_log_likelihood": nll, "n_data_points": 50, "chi2": 45.0},
        "get_parameters": [{"estimate": 1}] * 3,
        "get_fit_diagnosis": {},
        "diff_versions": {},
        "get_errors": [],
        "get_run_timing": {},
        "evaluate_validation_data": payload,
    }
    asked = []

    async def call(name, **kwargs):
        asked.append(name)
        if name == "evaluate_validation_data" and payload is _RAISES:
            raise RuntimeError("Unknown tool: evaluate_validation_data")
        return canned.get(name, {})

    code, result = asyncio.run(
        _judge(scenario, call, _Baseline(), work, {}, _ToolClock(), time.monotonic())
    )
    return code, result, asked


def test_the_held_out_score_reaches_result_json_without_changing_the_verdict(tmp_path):
    """ΔBIC is 0 here, so the run passes. A held-out score that is far WORSE than the true model's
    must be recorded and must NOT flip that: the check is report-only, and a gate on it would be
    calibrated against nothing (see scoring._validation_fields)."""
    code, result, _ = _run_judge(
        tmp_path,
        has_held_out=True,
        payload={"status": "ok", "nll": 900.0, "n": 12, "chi2": 400.0, "n_conditions": 4},
    )
    assert result["delta_bic"] == pytest.approx(0.0)
    assert result["validation"]["delta_nll"] == pytest.approx(500.0)  # 900 - 400, much worse
    assert result["validation"]["reference_chi2"] == 90.0
    assert result["success"] is True and code == 0


def test_an_example_with_no_held_out_data_is_never_asked_for_a_score(tmp_path):
    """The other seventeen examples have no held-out table. Asking anyway printed an "unavailable"
    line on every one of their runs for a check they were never meant to have — so the judge gates
    on the file, and `validation` stays None rather than becoming a status."""
    code, result, asked = _run_judge(
        tmp_path, has_held_out=False, payload=_RAISES, reference={"nll": 100.0, "k": 3, "n": 50}
    )
    assert "evaluate_validation_data" not in asked
    assert result["validation"] is None
    assert result["success"] is True and code == 0


def test_a_failing_held_out_call_costs_the_run_nothing(tmp_path):
    """Report-only means report-only: if the tool raises on an example that HAS held-out data, the
    judge still writes result.json and still returns its BIC verdict. Losing result.json to a
    diagnostic is the unscored-run outcome the harness exists to avoid."""
    code, result, asked = _run_judge(tmp_path, has_held_out=True, payload=_RAISES)
    assert "evaluate_validation_data" in asked
    # the reference HAS a bar, so the failure is recorded as unavailable rather than dropped
    assert result["validation"]["status"] == "unavailable"
    assert result["validation"]["reference_nll"] == 400.0
    assert result["success"] is True and code == 0


def test_the_span_and_the_csv_cannot_disagree_about_the_held_out_columns(monkeypatch):
    """One naming authority. `tracing.record_run_result` stamps whatever
    `metrics.flatten_validation` returns, so a tenth `val_*` column reaches the Phoenix span and
    the history CSV together or not at all — it used to restate the nine names as its own tuple."""
    from gemot import metrics, tracing

    flat = metrics.flatten_validation({"validation": {"status": "ok", "nll": 1.0}})
    assert set(flat) == {c for c in metrics.HISTORY_FIELDS if c.startswith("val_")}
    # ...and a result with no held-out block contributes NOTHING, rather than nine empty columns
    assert metrics.flatten_validation({"scenario": "boehm"}) == {}
    # The span stamps exactly what the flattener returns, so it cannot hold its own key list. Pinned
    # BEHAVIOURALLY: a captured `_set_attrs` sees every `val_*` name and no others. (Asserting the
    # absence of the string "val_status" from the source would also fail on a comment that merely
    # mentions the column.)
    stamped: dict = {}
    monkeypatch.setattr(
        tracing, "_set_attrs", lambda span, attrs: stamped.update(attrs), raising=True
    )
    tracing.record_run_result(
        object(),
        {
            "scenario": "fujita",
            "validation": {
                "status": "ok",
                "nll": 5.0,
                "reference_nll": 4.0,
                "delta_nll": 1.0,
                "n": 12,
            },
        },
    )
    assert {k.removeprefix("petab.") for k in stamped if k.startswith("petab.val_")} <= set(
        metrics.HISTORY_FIELDS
    )
    assert "petab.val_delta_nll" in stamped and stamped["petab.val_delta_nll"] == 1.0


# --------------------------------------------- the guards the last two reviews added
def test_the_recorded_reference_vector_must_be_full_dimensional():
    """`_recorded_vector`'s two callers zip it against `pp.x_names` and index it by free position,
    so a reduced or reordered `x` would surface as wrong sigmas and a wrong held-out score rather
    than as an error."""
    from gemot.simulate import _recorded_vector

    class _PP:
        x_names = ["a", "b", "c"]

    class _Result:
        class optimize_result:
            list = [{"x": [1.0, 2.0]}]  # pypesto returning a REDUCED vector

    assert list(_recorded_vector(_Result(), [1.0, 2.0, 3.0], _PP())) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="must be the FULL vector"):
        _recorded_vector(_Result(), None, _PP())


def test_a_failed_reference_refit_is_distinguishable_from_no_bar_at_all(tmp_path):
    """Both leave `val_reference_nll` blank. Without `val_reference_status` the two are byte-
    identical in the history CSV, which would make the guarded refit's whole point — record the
    failure rather than drop it — invisible to every sweep."""
    from gemot import metrics

    failed = _validation_fields(
        {"status": "ok", "nll": 500.0, "n": 12, "chi2": 1.0},
        {
            "nll": 1.0,
            "k": 2,
            "n": 10,
            "validation": {"status": "error", "reason": "ValueError: boom"},
        },
    )
    assert failed["reference_status"] == "error" and failed["reference_nll"] is None

    rows = [metrics.build_row({"scenario": "fujita", "validation": failed})]
    path = metrics.append_history(rows, tmp_path / "h.csv")
    row = dict(zip(*(line.split(",") for line in path.read_text().splitlines()[:2])))
    assert row["val_reference_status"] == "error"
    assert row["val_reference_nll"] == "" and row["val_delta_nll"] == ""


def test_an_attempted_check_with_no_reference_bar_is_reported_not_skipped():
    """A judge that CALLED the tool and got nothing back must not look like an example with no
    held-out set. Without `asked` the two are the same `None`, so a reference that has not been
    refit yet turns every failed call into a silent clean skip — undiagnosable from result.json."""
    assert _validation_fields(None, {}) is None  # genuinely no check exists
    out = _validation_fields(None, {}, asked=True)
    assert out["status"] == "unavailable" and "no held-out score" in out["reason"]
    # and the shape does not depend on whether the tool answered
    assert "reference_chi2" in _validation_fields(
        None, {"validation": {"status": "ok", "nll": 1.0, "n": 4, "chi2": 2.0}}
    )
