"""The fixed-θ HELD-OUT check: :func:`petab_mcp.validation_data.score_problem` and its guards.

Tested at the boundary rather than through an end-to-end run: which tables the scorer copies, what
it declines before paying for a build, and how it refuses a parameter vector it cannot line up.
The objective evaluation past those guards needs an AMICI build; its correctness is pinned by the
reference's own held-out score in BENCHMARK_NOTES.md, re-derived by ``setup.py --refit``.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from petab_harness import benchmark
from petab_harness.runner.scoring import _validation_fields


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


# ------------------------------------------------------- the scorer does not disturb the session
def test_scoring_never_mutates_the_problem_it_was_handed(monkeypatch):
    """``score_session`` runs against the LIVE session problem, so a table it swapped in place would
    change what the agent reads and what the judge scores on the training data — the held-out check
    would corrupt the very run it is reporting on. The scorer builds a shallow view with its own
    dataframe objects; this pins that, without paying for an AMICI build, by intercepting the
    importer and inspecting the view it was handed.

    Also pins that the visualization table is DROPPED: petab's lint rejects one whose datasetIds are
    absent from the measurement table, which is exactly what swapping in held-out data does, and
    ``PetabImporter`` raises a bare "Invalid PEtab problem." for it (observed on the collection's
    fujita).
    """
    import pypesto.petab

    from petab_mcp import validation_data

    problem = _problem(_rows("c_train"))
    problem.visualization_df = pd.DataFrame({"plotId": ["p1"], "datasetId": ["c_train_obs_A"]})
    original_measurements = problem.measurement_df
    held_out = _rows("c_test")

    captured = {}

    class _Stop(RuntimeError):
        pass

    def _importer(petab_problem, **kwargs):
        captured["view"] = petab_problem
        raise _Stop

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _importer)
    with pytest.raises(_Stop):
        validation_data.score_problem(problem, held_out, build_dir="/tmp/x", model_name="m")

    view = captured["view"]
    assert view is not problem
    assert list(view.measurement_df["simulationConditionId"]) == ["c_test", "c_test"]
    assert view.visualization_df is None, "the training visualization table would fail petab lint"
    # ...and the caller's own problem is exactly as it was
    assert problem.measurement_df is original_measurements
    assert list(problem.measurement_df["simulationConditionId"]) == ["c_train", "c_train"]
    assert problem.visualization_df is not None


def test_the_held_out_view_keeps_the_training_estimate_split(monkeypatch):
    """The view handed to the importer must mark exactly the parameters the training table marks.

    AMICI compiles a NON-estimated parameter as a model constant, so a compiled module is a
    function of the SBML plus the free/fixed split — not of the SBML alone. ``score_session``
    therefore hands over the build target the training problem was compiled with and reuses that
    module, which is correct only while the view's split matches. This pins the copy that keeps it
    matching; a view that freed or fixed a parameter would silently evaluate against constants
    baked in for a different split.
    """
    import pypesto.petab

    from petab_mcp import validation_data

    problem = _problem(_rows("c_train"))
    captured = {}

    class _Stop(RuntimeError):
        pass

    def _importer(petab_problem, **kwargs):
        captured["view"] = petab_problem
        captured["kwargs"] = kwargs
        raise _Stop

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _importer)
    with pytest.raises(_Stop):
        validation_data.score_problem(
            problem, _rows("c_test"), build_dir="/tmp/train_build", model_name="train_module"
        )

    view = captured["view"]
    assert view.parameter_df["estimate"].to_dict() == problem.parameter_df["estimate"].to_dict()
    # ...and the target is passed through untouched, so score_session naming the TRAINING pair is
    # what makes the reuse land on the training module.
    assert captured["kwargs"]["output_folder"] == "/tmp/train_build"
    assert captured["kwargs"]["model_name"] == "train_module"


def test_an_empty_or_foreign_held_out_table_is_declined_before_any_build(monkeypatch):
    """The guards run BEFORE the importer, so a bad table costs nothing and names its own problem
    rather than surfacing from inside AMICI's parameter mapping."""
    import pypesto.petab

    from petab_mcp import validation_data

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("the importer must not be reached for a declined table")

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _boom)
    problem = _problem(_rows("c_train"))

    empty = validation_data.score_problem(
        problem, _rows("c_test").iloc[0:0], build_dir="/tmp/x", model_name="m"
    )
    assert empty["status"] == validation_data.STATUS_NO_DATA

    foreign = validation_data.score_problem(
        problem, _rows("c_elsewhere"), build_dir="/tmp/x", model_name="m"
    )
    assert foreign["status"] == validation_data.STATUS_TABLE_MISMATCH
    assert "c_elsewhere" in foreign["reason"]

    unknown_obs = validation_data.score_problem(
        problem, _rows("c_test", observable="obs_Z"), build_dir="/tmp/x", model_name="m"
    )
    assert unknown_obs["status"] == validation_data.STATUS_TABLE_MISMATCH
    assert "obs_Z" in unknown_obs["reason"]


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
    from petab_harness.agent.tools import ALLOWED_TOOLS

    assert "evaluate_validation_data" not in ALLOWED_TOOLS


def test_the_server_only_registers_the_tool_when_given_held_out_data(
    monkeypatch, yaml_path, tmp_path
):
    """No held-out file, no tool — so an example without a validation set cannot accidentally be
    scored against a stale or unrelated table, and the judge's "unavailable" status is reached by
    the tool genuinely not existing.

    Asserted through the registrar actually being CALLED, not by grepping ``create_server``'s
    source: a reformat must not fail this, and an equivalent rewrite must not pass it while
    dropping the gate."""
    import inspect

    from petab_mcp import server as server_mod
    from petab_mcp import validation_data
    from petab_mcp.server import create_server

    calls = []
    monkeypatch.setattr(
        server_mod,
        "_register_validation_data_tools",
        lambda mcp, ctx, held_out: calls.append(held_out),
    )
    assert "validation_measurements" in inspect.signature(create_server).parameters

    create_server(
        yaml_path, build_root=str(tmp_path / "b1"), provenance_dir=str(tmp_path / "prov1")
    )
    assert calls == []

    meas = tmp_path / "held.tsv"
    _rows("c_test").to_csv(meas, sep="\t", index=False)
    create_server(
        yaml_path,
        build_root=str(tmp_path / "b2"),
        provenance_dir=str(tmp_path / "prov2"),
        validation_measurements=str(meas),
    )
    assert len(calls) == 1
    assert isinstance(calls[0], validation_data.HeldOut)
    assert len(calls[0].measurement_df) == 2


def test_a_held_out_row_that_is_also_a_training_row_is_refused(monkeypatch, tmp_path):
    """The whole claim of the number is that the model never saw this data. An overlap makes it
    partly in-sample, which no downstream check could detect — the likelihoods would simply be a bit
    too good, indistinguishable from a model that predicts well."""
    train = _rows("c_train")
    published = _problem(train)
    overlapping = pd.concat([_rows("c_test"), _rows("c_train")], ignore_index=True)

    monkeypatch.setattr(benchmark, "benchmark_dir", lambda *a, **k: tmp_path)
    (tmp_path / "held_out.yaml").write_text("format_version: 1\n")

    import petab.v1 as petab

    held = _problem(overlapping)
    monkeypatch.setattr(petab.Problem, "from_yaml", staticmethod(lambda *a, **k: held))

    with pytest.raises(ValueError, match="also in the TRAINING table"):
        benchmark._validation_measurement_df(
            name="X",
            validation_yaml="held_out.yaml",
            collection=None,
            published=published,
            training_measurement_df=train,
        )


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
    from petab_harness import metrics

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

    from petab_harness import benchmark as bm
    from petab_harness.runner import _judge, _ToolClock

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
    from petab_harness import metrics, tracing

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


def test_an_explicit_point_must_come_with_matching_names(monkeypatch):
    """A caller-supplied vector is in ITS problem's x_names order. The two problems are built from
    the same parameter table so they agree today, but a silent re-ordering would evaluate a permuted
    vector and report a plausible, wrong likelihood — so the names are checked, not assumed."""
    from petab_mcp import validation_data

    problem = _problem(_rows("c_train"))
    with pytest.raises(ValueError, match="both x_full and x_names, or neither"):
        validation_data.score_problem(
            problem, _rows("c_test"), build_dir="/tmp/x", model_name="m", x_full=[0.5]
        )


# --------------------------------------------- the guards the last two reviews added
# Each of these exists because its ABSENCE was a silent-wrong-number risk, so each needs a test
# that fails if it is relaxed back. (The repo's own lesson: 15 defects survived a green suite
# because the plumbing around the statistic went untested — this IS the plumbing.)
def test_a_short_x_full_is_refused_rather_than_evaluated_misaligned(monkeypatch):
    """The name check compares `x_names` against the problem's; it says NOTHING about how many
    values arrived with them. A shorter vector whose free indices all happen to be in range would
    otherwise be silently indexed and reported as a plausible held-out likelihood for a parameter
    point nobody chose."""
    import pypesto.petab

    from petab_mcp import validation_data

    class _FakePP:
        x_names = ["k", "sigma", "scale"]
        x_free_indices = [0]

        def objective(self, *a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("the objective must not be evaluated on a bad vector")

    class _FakeImporter:
        def __init__(self, *a, **k):
            pass

        def create_problem(self):
            return _FakePP()

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _FakeImporter)
    problem = _problem(_rows("c_train"))

    with pytest.raises(ValueError, match="must be the FULL vector"):
        validation_data.score_problem(
            problem,
            _rows("c_test"),
            build_dir="/tmp/x",
            model_name="m",
            x_full=[0.5],  # one value for three parameters
            x_names=["k", "sigma", "scale"],
        )  # ...but the NAMES match, so only length catches it


def test_the_recorded_reference_vector_must_be_full_dimensional():
    """`_recorded_vector`'s two callers zip it against `pp.x_names` and index it by free position,
    so a reduced or reordered `x` would surface as wrong sigmas and a wrong held-out score rather
    than as an error."""
    from petab_harness.simulate import _recorded_vector

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
    from petab_harness import metrics

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


def test_a_time_written_as_1_and_1_0_is_the_same_held_out_row(monkeypatch, tmp_path):
    """The two tables are parsed from different TSVs, so one can render a timepoint as `1` and the
    other as `1.0`. Stringifying the key made those different rows and let a genuinely in-sample
    row past the disjointness check — the one guard that keeps the held-out score out-of-sample."""
    import pandas as pd

    def _table(time_value):
        return pd.DataFrame(
            {
                "observableId": ["obs_A"],
                "simulationConditionId": ["c1"],
                "time": [time_value],
                "measurement": [1.0],
            }
        )

    class _V:
        measurement_df = _table(1.0)
        observable_df = _problem(_rows("c1")).observable_df

    monkeypatch.setattr("petab.v1.lint.check_measurement_df", lambda *a, **k: None)
    monkeypatch.setattr("petab_mcp.fitting.needs_flattening", lambda *a, **k: False)

    with pytest.raises(ValueError, match="also in the TRAINING table"):
        benchmark._check_held_out_rows("Toy", _V(), _problem(_rows("c1")), _table(1))


# ------------------------------------------- the point a caller hands in, matched by name
def test_a_view_with_fewer_parameter_rows_is_served_not_refused():
    """The caller fitted the TRAINING problem; the held-out view can carry fewer parameter rows
    than that (petab refuses rows nothing references, and swapping a curated held-out table in
    retires the overrides the training rows carried). `pp.x_names` is then a strict subset of the
    caller's, which list equality would refuse outright — so it is matched by name."""
    from petab_mcp.validation_data import _reduced_from_named

    class _PP:
        x_names = ["k", "sigma"]  # the view dropped the training scalings
        x_free_indices = [0, 1]

    reduced = _reduced_from_named(_PP(), [1.0, 2.0, 3.0], ["k", "sigma", "s_blot_retired"])
    assert reduced == [1.0, 2.0]  # in pp.x_names order, the extra name ignored


def test_a_name_the_view_needs_and_the_caller_omitted_is_refused():
    """The other direction is still a hard error: a vector that does not cover what the problem
    needs would be evaluated permuted and report a plausible, wrong likelihood."""
    from petab_mcp.validation_data import _reduced_from_named

    class _PP:
        x_names = ["k", "sigma", "scale"]
        x_free_indices = [0, 1, 2]

    with pytest.raises(ValueError, match="do not match the held-out problem's"):
        _reduced_from_named(_PP(), [1.0, 2.0, 3.0], ["k", "sigma", "unrelated"])


def test_arity_is_reported_as_arity_not_as_a_name_mismatch():
    """A vector already reduced to free coordinates is a LENGTH mistake. Zipping it against the
    full name list would truncate silently and then blame the names."""
    from petab_mcp.validation_data import _reduced_from_named

    class _PP:
        x_names = ["k", "sigma", "scale"]
        x_free_indices = [0]

    with pytest.raises(ValueError, match="must be the FULL vector"):
        _reduced_from_named(_PP(), [0.5], ["k", "sigma", "scale"])


# ------------------------------------------------------- a held-out table the EXAMPLE ships
def test_a_curated_table_is_read_against_the_published_problems_own_tables(tmp_path):
    """The curated path has no second problem to compare against, so the published problem IS the
    reference for the table: read it, check it, hand it back unchanged. Pinned on the value, not on
    the reader, because a table silently returned empty would score a model against nothing."""
    (tmp_path / "curated").mkdir()
    _rows("c_test").to_csv(tmp_path / "curated" / "measurements.tsv", sep="\t", index=False)

    held = benchmark._curated_held_out(
        name="X",
        validation_table="curated/measurements.tsv",
        example_dir=tmp_path,
        published=_problem(_rows("c_train")),
    )

    assert list(held.measurement_df["simulationConditionId"]) == ["c_test", "c_test"]
    assert held.condition_df is None and held.mode == "fixed"


def test_a_curated_table_naming_an_unpublished_condition_is_refused(tmp_path):
    """Fixed-θ scores at the training point, so every held-out condition has to be one the model was
    told about. A row on a condition neither the problem nor a conditions.tsv beside the table
    defines asks for a prediction nobody could have made — that data is a TRANSFER set, which
    refits, and is not scored here."""
    (tmp_path / "curated").mkdir()
    _rows("c_knockdown").to_csv(tmp_path / "curated" / "measurements.tsv", sep="\t", index=False)

    with pytest.raises(ValueError, match="c_knockdown"):
        benchmark._curated_held_out(
            name="X",
            validation_table="curated/measurements.tsv",
            example_dir=tmp_path,
            published=_problem(_rows("c_train")),
        )


def test_a_curated_row_that_is_also_a_training_row_is_refused(tmp_path):
    """Same claim as for the yaml path, same check: the curated table goes through
    ``_check_held_out_rows``, so shipping a training row by hand is caught at setup."""
    (tmp_path / "curated").mkdir()
    _rows("c_train").to_csv(tmp_path / "curated" / "measurements.tsv", sep="\t", index=False)

    with pytest.raises(ValueError, match="also in the TRAINING table"):
        benchmark._curated_held_out(
            name="X",
            validation_table="curated/measurements.tsv",
            example_dir=tmp_path,
            published=_problem(_rows("c_train")),
        )


def test_an_absent_curated_table_names_itself(tmp_path):
    """As for a typo'd yaml: say which path, at setup, rather than leaving the example silently
    unvalidated."""
    with pytest.raises(FileNotFoundError, match="nope.tsv"):
        benchmark._curated_held_out(
            name="X",
            validation_table="nope.tsv",
            example_dir=tmp_path,
            published=_problem(_rows("c_train")),
        )


def test_a_malformed_curated_table_reports_lint_before_reading_condition_ids(tmp_path):
    """A missing required column must name the malformed table, not leak a raw KeyError."""
    _rows("c_test").drop(columns="simulationConditionId").to_csv(
        tmp_path / "measurements.tsv", sep="\t", index=False
    )

    with pytest.raises(ValueError, match="validation measurement table fails petab's lint"):
        benchmark._curated_held_out(
            name="X",
            validation_table="measurements.tsv",
            example_dir=tmp_path,
            published=_problem(_rows("c_train")),
        )


def test_declaring_both_held_out_sources_is_refused(tmp_path):
    """One example, one held-out file. Two declarations write to the same path, so whichever ran
    last would silently win and the example would be scored against a set nobody chose."""
    with pytest.raises(ValueError, match="BOTH"):
        benchmark._held_out_source(
            name="X",
            validation_yaml="test_data/held.yaml",
            validation_table="curated/measurements.tsv",
            example_dir=tmp_path,
            collection=None,
            published=_problem(_rows("c_train")),
        )


def test_no_declaration_means_no_held_out_table(tmp_path):
    """The default: an example without held-out data returns None here, which is what leaves the
    server's scoring tool unregistered."""
    assert (
        benchmark._held_out_source(
            name="X",
            validation_yaml=None,
            validation_table=None,
            example_dir=tmp_path,
            collection=None,
            published=_problem(_rows("c_train")),
        )
        is None
    )


# ----------------------------------- a fixed-θ set that brings new rows of the training inputs
def _curated(tmp_path, measurements, conditions=None):
    (tmp_path / "curated").mkdir(exist_ok=True)
    measurements.to_csv(tmp_path / "curated" / "measurements.tsv", sep="\t", index=False)
    if conditions is not None:
        conditions.to_csv(tmp_path / "curated" / "conditions.tsv", sep="\t", index=False)
    return "curated/measurements.tsv"


def test_a_curated_set_may_bring_new_values_of_the_training_inputs(tmp_path):
    """A conditions.tsv beside the table adds rows on the training table's own columns — another
    cell type's Smad abundances, a knockdown level — and the conditions it names are then known.
    Still fixed-θ: no parameter table, so the fitted point is what gets scored."""
    table = _curated(
        tmp_path,
        _rows("c_knockdown"),
        pd.DataFrame({"conditionId": ["c_knockdown"], "dose": [0.5]}),
    )

    held = benchmark._curated_held_out(
        name="X", validation_table=table, example_dir=tmp_path, published=_problem(_rows("c_train"))
    )

    assert held.mode == "fixed"
    assert list(held.condition_df.index) == ["c_knockdown"]
    assert float(held.condition_df.loc["c_knockdown", "dose"]) == 0.5


@pytest.mark.parametrize(
    "conditions, needle",
    [
        (
            pd.DataFrame({"conditionId": ["c_new"], "dose": [0.5], "temperature": [37.0]}),
            "different models",
        ),
        (pd.DataFrame({"conditionId": ["c_train"], "dose": [0.5]}), "does not redefine"),
    ],
)
def test_a_curated_condition_table_may_only_add_rows(tmp_path, conditions, needle):
    """A new column is a new input, i.e. a different model; a training id redefines a condition
    the model was fitted under. Both are refused at setup, in the score-time rule's own words."""
    table = _curated(tmp_path, _rows("c_test"), conditions)

    with pytest.raises(ValueError, match=needle):
        benchmark._curated_held_out(
            name="X",
            validation_table=table,
            example_dir=tmp_path,
            published=_problem(_rows("c_train")),
        )


def test_a_curated_preequilibration_condition_must_be_known_too(tmp_path):
    """The knockdown rows pre-equilibrate at the knockdown, so the preeq id is a condition the
    table names and is checked like the simulation one."""
    table = _curated(tmp_path, _rows("c_test").assign(preequilibrationConditionId="c_basal"))

    with pytest.raises(ValueError, match="c_basal"):
        benchmark._curated_held_out(
            name="X",
            validation_table=table,
            example_dir=tmp_path,
            published=_problem(_rows("c_train")),
        )


def test_materialize_writes_a_fixed_sets_condition_rows_beside_its_measurements(
    tmp_path, monkeypatch
):
    """What lands on disk is the contract with the server: conditions.tsv beside measurements.tsv,
    in the shape of the problem's own (no conditionName), and gone again when the set stops
    bringing any. The directory is gitignored and outlives a branch switch, so every file an
    earlier definition of the example wrote is removed first: a leftover parameters.tsv would flip
    the set to TRANSFER and refit, a leftover set would score an example that declares none."""
    from petab_mcp import validation_data

    published = _problem(_rows("c_train"))
    published.condition_df["conditionName"] = ["train", "test"]
    monkeypatch.setattr(benchmark, "_load_problem", lambda *a, **k: copy.deepcopy(published))
    monkeypatch.setattr(benchmark, "_write_empty_sbml", lambda d: None)
    monkeypatch.setattr(benchmark, "_write_problem_yaml", lambda d: None)
    table = _curated(
        tmp_path,
        _rows("c_knockdown"),
        pd.DataFrame({"conditionId": ["c_knockdown"], "conditionName": ["kd"], "dose": [0.5]}),
    )
    vpath = benchmark.validation_measurements_path(tmp_path)
    vpath.parent.mkdir()
    stale = vpath.with_name(validation_data.HELD_OUT_PARAMETERS)
    stale.write_text("parameterId\tnominalValue\testimate\nk\t0.5\t1\n")

    out = benchmark.materialize(
        name="X", shipped_param_ids=["k"], example_dir=tmp_path, validation_table=table
    )
    held = validation_data.read_held_out(vpath)

    assert out["validation_conditions"] == 1
    assert held.mode == validation_data.MODE_FIXED and not stale.exists()
    assert list(held.condition_df.index) == ["c_knockdown"]
    assert list(held.condition_df.columns) == ["dose"]

    _curated(tmp_path, _rows("c_test"))
    (tmp_path / "curated" / "conditions.tsv").unlink()
    benchmark.materialize(
        name="X", shipped_param_ids=["k"], example_dir=tmp_path, validation_table=table
    )
    assert validation_data.read_held_out(vpath).condition_df is None

    benchmark.materialize(name="X", shipped_param_ids=["k"], example_dir=tmp_path)
    assert not vpath.exists()


def test_score_problem_appends_a_fixed_sets_condition_rows_before_building(monkeypatch):
    """The scorer's view carries the training conditions AND the set's own, so a row on a new
    condition is a table the importer can build, not a mismatch; the caller's problem keeps its
    table."""
    import pypesto.petab

    from petab_mcp import validation_data

    seen = []

    class _Objective:
        def __call__(self, x, **kwargs):
            return {"fval": 1.0, "rdatas": []}

    class _PP:
        x_names = ["k"]
        x_free_indices = [0]
        x_scales = ["lin"]
        objective = _Objective()

    class _Importer:
        def __init__(self, petab_problem, **kwargs):
            seen.append(petab_problem)

        def create_problem(self):
            return _PP()

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _Importer)
    import petab_mcp.fitting as fitting

    monkeypatch.setattr(fitting, "_apply_amici_solver_defaults", lambda pp: None)
    problem = _problem(_rows("c_train"))
    new = pd.DataFrame({"dose": [0.5]}, index=pd.Index(["c_knockdown"], name="conditionId"))

    out = validation_data.score_problem(
        problem, _rows("c_knockdown"), build_dir="/tmp/x", model_name="m", condition_df=new
    )

    assert out["status"] == validation_data.STATUS_OK and out["n_conditions"] == 1
    assert list(seen[0].condition_df.index) == ["c_train", "c_test", "c_knockdown"]
    assert list(problem.condition_df.index) == ["c_train", "c_test"]


def test_a_fixed_sets_condition_table_is_checked_before_any_build(monkeypatch):
    """Same contract as for a foreign measurement table: the guards run before the importer."""
    import pypesto.petab

    from petab_mcp import validation_data

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("the importer must not be reached for a declined table")

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _boom)
    problem = _problem(_rows("c_train"))

    other_column = pd.DataFrame(
        {"temperature": [37.0]}, index=pd.Index(["c_new"], name="conditionId")
    )
    out = validation_data.score_problem(
        problem, _rows("c_new"), build_dir="/tmp/x", model_name="m", condition_df=other_column
    )
    assert out["status"] == validation_data.STATUS_TABLE_MISMATCH
    assert "different models" in out["reason"]

    redefined = pd.DataFrame({"dose": [0.5]}, index=pd.Index(["c_train"], name="conditionId"))
    out = validation_data.score_problem(
        problem, _rows("c_test"), build_dir="/tmp/x", model_name="m", condition_df=redefined
    )
    assert out["status"] == validation_data.STATUS_TABLE_MISMATCH
    assert "does not redefine" in out["reason"]


def test_score_session_hands_a_fixed_sets_condition_rows_to_the_scorer(monkeypatch):
    """`score_session` is the server's one entry point; a fixed set's own rows reach the scorer
    through it, or the tool reports `table_mismatch` for every curated set that brings any."""
    import contextlib
    import types

    from petab_mcp import validation_data

    seen = {}

    def _score(problem, measurement_df, **kwargs):
        seen.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(validation_data, "score_problem", _score)
    session = types.SimpleNamespace(
        lock=contextlib.nullcontext(), problem=object(), sbml_hash=lambda: "h"
    )
    engine = types.SimpleNamespace(
        pypesto_importer_kwargs=lambda h: {"output_folder": "/tmp/x", "model_name": "m"}
    )
    rows = pd.DataFrame({"dose": [0.5]}, index=pd.Index(["c_knockdown"], name="conditionId"))

    validation_data.score_session(
        session, engine, validation_data.HeldOut(_rows("c_knockdown"), condition_df=rows)
    )

    assert seen["condition_df"] is rows


def test_read_held_out_gives_a_fixed_set_its_condition_rows(tmp_path):
    """On disk: measurements.tsv + conditions.tsv and NO parameters.tsv is a fixed-θ set with rows
    of its own — what `materialize` writes for such a set and what the server reads back."""
    from petab_mcp import validation_data

    _rows("c_knockdown").to_csv(tmp_path / "measurements.tsv", sep="\t", index=False)
    pd.DataFrame({"conditionId": ["c_knockdown"], "dose": [0.5]}).to_csv(
        tmp_path / validation_data.HELD_OUT_CONDITIONS, sep="\t", index=False
    )

    held = validation_data.read_held_out(tmp_path / "measurements.tsv")

    assert held.mode == validation_data.MODE_FIXED
    assert list(held.condition_df.index) == ["c_knockdown"]


def test_score_session_scores_the_toy_at_its_fitted_point_on_a_condition_of_the_sets_own(
    session, engine
):
    """The one AMICI-backed check of the fixed-θ side: the toy's measurements moved onto a condition
    `c1` the training table does not have, brought by the set. Scored at the training point, not
    refit — `mode: fixed`, no `k` — and the session's own table is left as it was."""
    import math
    from pathlib import Path

    from petab_mcp import validation_data

    toy = Path(__file__).parent / "data" / "toy_v1"
    held_meas = pd.read_csv(toy / "measurements.tsv", sep="\t").assign(simulationConditionId="c1")
    held = validation_data.HeldOut(
        measurement_df=held_meas,
        condition_df=pd.DataFrame(index=pd.Index(["c1"], name="conditionId")),
    )

    out = validation_data.score_session(session, engine, held)

    assert out["status"] == validation_data.STATUS_OK, out
    assert out["mode"] == validation_data.MODE_FIXED and "k" not in out
    assert out["n"] == len(held_meas) and out["n_conditions"] == 1
    assert math.isfinite(out["nll"])
    assert list(session.problem.condition_df.index) == ["c0"]


# ------------------------------------------- a held-out table's own per-blot scale factors
def _scaled_problem(formula="observableParameter1_obs_A * x"):
    """A problem whose one observable carries a scaling placeholder, like every blot readout."""
    import petab.v1 as petab

    return petab.Problem(
        model=None,
        measurement_df=_rows("c_train").assign(observableParameters="s_train"),
        observable_df=pd.DataFrame(
            {"observableFormula": [formula], "noiseFormula": ["noiseParameter1_obs_A"]},
            index=pd.Index(["obs_A"], name="observableId"),
        ),
        condition_df=pd.DataFrame(
            {"dose": [0.0, 0.0]}, index=pd.Index(["c_train", "c_test"], name="conditionId")
        ),
        parameter_df=pd.DataFrame(
            {
                "parameterScale": ["lin", "lin"],
                "lowerBound": [0.0, 0.0],
                "upperBound": [1.0, 10.0],
                "nominalValue": [0.5, 1.0],
                "estimate": [1, 1],
            },
            index=pd.Index(["k", "s_train"], name="parameterId"),
        ),
    )


def test_held_out_scalings_are_the_overrides_the_problem_does_not_define():
    """A held-out blot's scale factor is whatever override the parameter table lacks; a shipped
    scaling (fujita's rows) or a number is an ordinary PEtab override and not touched."""
    from petab_mcp import validation_data

    held = pd.concat(
        [
            _rows("c_test").assign(observableParameters="s_blot"),
            _rows("c_test", times=(2.0,)).assign(observableParameters="s_train"),
            _rows("c_test", times=(3.0,)).assign(observableParameters="2.5"),
        ]
    )
    assert validation_data.held_out_scalings(_scaled_problem(), held) == {"s_blot"}


def test_held_out_scalings_are_solved_hierarchically_then_written_in_as_numbers(monkeypatch):
    """The two-pass shape, pinned without an AMICI build: pass one hands pyPESTO a HIERARCHICAL
    problem whose only estimated rows are the held-out scalings tagged `parameterType: scaling`
    (the model's own parameters keep their values, the training-only overrides are gone); pass two
    is the ordinary joint objective with the solved values substituted as plain numbers, so the
    reported nll/chi2 come off the same code path as for an example without held-out scalings."""
    import pypesto.petab

    from petab_mcp import validation_data

    seen = []

    class _Calc:
        def get_interpretable_inner_par_ids(self):
            return ["s_blot"]

    class _Objective:
        calculator = _Calc()

        def __call__(self, x, **kwargs):
            return {"fval": 1.0, "inner_parameters": [3.5], "rdatas": []}

    class _PP:
        x_names = ["k"]
        x_free_indices = [0]
        x_scales = ["lin"]
        objective = _Objective()

    class _Importer:
        def __init__(self, petab_problem, **kwargs):
            seen.append((petab_problem, kwargs.get("hierarchical", False)))

        def create_problem(self):
            return _PP()

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _Importer)
    monkeypatch.setattr(
        validation_data, "_apply_amici_solver_defaults", lambda pp: None, raising=False
    )
    import petab_mcp.fitting as fitting

    monkeypatch.setattr(fitting, "_apply_amici_solver_defaults", lambda pp: None)

    held = _rows("c_test").assign(observableParameters="s_blot")
    out = validation_data.score_problem(_scaled_problem(), held, build_dir="/tmp/x", model_name="m")

    assert [h for _, h in seen] == [True, False], (
        "hierarchical solve first, joint evaluation second"
    )
    hier_view, joint_view = seen[0][0], seen[1][0]
    row = hier_view.parameter_df.loc["s_blot"]
    assert row["parameterType"] == "scaling" and int(row["estimate"]) == 1
    assert float(row["lowerBound"]) == 0.0, "a gel's scale factor is non-negative"
    assert list(joint_view.measurement_df["observableParameters"]) == ["3.5", "3.5"]
    assert out["status"] == validation_data.STATUS_OK
    assert out["held_out_scalings"] == {"s_blot": 3.5}


def test_a_callers_vector_is_matched_by_name_so_a_smaller_held_out_problem_still_scores(
    monkeypatch,
):
    """The held-out view drops the training-only overrides and its hierarchical twin drops the inner
    scalings, so the problem pyPESTO builds has FEWER names than the reference fit's vector. Matching
    by name — not position — is what lets `_reference_validation` hand over its full vector."""
    import pypesto.petab

    from petab_mcp import validation_data

    class _PP:
        x_names = ["k"]
        x_free_indices = [0]
        x_scales = ["lin"]

        class objective:
            @staticmethod
            def __call__(x, **kwargs):
                raise AssertionError("not reached")

    assert validation_data._reduced_from_named(_PP(), [0.5, 2.0], ["s_train", "k"]) == [2.0]
    with pytest.raises(ValueError, match="do not match"):
        validation_data._reduced_from_named(_PP(), [0.5], ["s_train"])
    # ...and a caller whose vector is shorter than its own names is caught before any build
    monkeypatch.setattr(pypesto.petab, "PetabImporter", lambda *a, **k: None)
    with pytest.raises(ValueError, match="must be the FULL vector"):
        validation_data.score_problem(
            _scaled_problem(),
            _rows("c_test"),
            build_dir="/tmp/x",
            model_name="m",
            x_full=[0.5],
            x_names=["k", "s_train"],
        )


# ------------------------------------------------ inputs the collection's export baked in


def _sbml_scaled_problem():
    """`_scaled_problem` with a REAL (tiny) SBML model, which the lint below needs: petab derives
    the set of legitimate parameter ids from the model, so with `model=None` it cannot tell an
    extraneous row from a kinetic one — and `_without_extraneous` short-circuits for the same
    reason. Returns `(problem, doc)`; the document is returned to keep libsbml's owner alive."""
    import libsbml
    import petab.v1 as petab
    from petab.v1.models.sbml_model import SbmlModel

    doc = libsbml.SBMLDocument(3, 2)
    sm = doc.createModel()
    sm.setId("m")
    comp = sm.createCompartment()
    comp.setId("cell"), comp.setConstant(True), comp.setSize(1.0), comp.setSpatialDimensions(3)
    sp_ = sm.createSpecies()
    sp_.setId("x"), sp_.setCompartment("cell"), sp_.setInitialConcentration(1.0)
    sp_.setHasOnlySubstanceUnits(False), sp_.setBoundaryCondition(False), sp_.setConstant(False)
    par = sm.createParameter()  # the kinetic the parameter table declares
    par.setId("k"), par.setValue(0.5), par.setConstant(True)
    base = _scaled_problem()
    return petab.Problem(
        model=SbmlModel(sbml_model=sm),
        observable_df=base.observable_df,
        condition_df=base.condition_df,
        measurement_df=base.measurement_df,
        parameter_df=base.parameter_df,
    ), doc


def test_the_extraneous_prune_is_petabs_lint_not_ours():
    """Why `_without_extraneous` exists at all. Swapping a held-out table in retires the overrides
    the TRAINING rows referenced, and petab's own parameter-table lint — which
    `PetabImporter.create_problem` runs, hierarchical or not — refuses the rows left behind. The
    hierarchical solver cannot help: the lint fires before there is an objective to solve."""
    from petab.v1.lint import check_parameter_df

    problem, _doc = _sbml_scaled_problem()
    held = _rows("c_test").assign(observableParameters="s_blot")
    view = copy.copy(problem)
    view.measurement_df = held
    view.parameter_df = pd.concat(
        [
            problem.parameter_df,
            pd.DataFrame(
                {
                    "parameterScale": ["lin"],
                    "lowerBound": [0.0],
                    "upperBound": [1e12],
                    "nominalValue": [1.0],
                    "estimate": [1],
                },
                index=pd.Index(["s_blot"], name="parameterId"),
            ),
        ]
    )

    with pytest.raises((AssertionError, ValueError), match="[Ee]xtraneous"):
        check_parameter_df(
            view.parameter_df,
            view.model,
            view.observable_df,
            view.measurement_df,
            view.condition_df,
        )

    from petab_mcp import validation_data

    kept = validation_data._without_extraneous(view)
    assert "s_train" not in kept.index, "the retired training override is dropped"
    assert {"k", "s_blot"} <= set(kept.index), "the kinetics and the held-out scaling stay"
    check_parameter_df(kept, view.model, view.observable_df, view.measurement_df, view.condition_df)
