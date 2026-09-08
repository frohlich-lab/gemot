"""The fixed-θ HELD-OUT check: :func:`petab_mcp.validation_data.score_problem` and its guards.

Tested at the boundary rather than through an end-to-end run: which tables the scorer copies, what
it declines before paying for a build, and how it refuses a parameter vector it cannot line up.
The objective evaluation past those guards needs an AMICI build; its correctness is pinned by the
reference's own held-out score in BENCHMARK_NOTES.md, re-derived by ``setup.py --refit``.
"""

from __future__ import annotations

import pandas as pd
import pytest



def _problem(measurements, *, observables=("obs_A",), conditions=("c_train", "c_test")):
    """A minimal petab problem stand-in with just the tables the guards read."""
    import petab.v1 as petab

    return petab.Problem(
        model=None,
        measurement_df=measurements,
        observable_df=pd.DataFrame(
            {"observableFormula": ["x"] * len(observables),
             "noiseFormula": ["1"] * len(observables)},
            index=pd.Index(list(observables), name="observableId")),
        condition_df=pd.DataFrame(
            {"dose": [0.0] * len(conditions)},
            index=pd.Index(list(conditions), name="conditionId")),
        parameter_df=pd.DataFrame(
            {"parameterScale": ["lin"], "lowerBound": [0.0], "upperBound": [1.0],
             "nominalValue": [0.5], "estimate": [1]},
            index=pd.Index(["k"], name="parameterId")),
    )


def _rows(condition, times=(0.0, 1.0), observable="obs_A"):
    return pd.DataFrame({"observableId": [observable] * len(times),
                         "simulationConditionId": [condition] * len(times),
                         "time": list(times),
                         "measurement": [1.0] * len(times)})


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
        problem, _rows("c_test").iloc[0:0], build_dir="/tmp/x", model_name="m")
    assert empty["status"] == validation_data.STATUS_NO_DATA

    foreign = validation_data.score_problem(
        problem, _rows("c_elsewhere"), build_dir="/tmp/x", model_name="m")
    assert foreign["status"] == validation_data.STATUS_TABLE_MISMATCH
    assert "c_elsewhere" in foreign["reason"]

    unknown_obs = validation_data.score_problem(
        problem, _rows("c_test", observable="obs_Z"), build_dir="/tmp/x", model_name="m")
    assert unknown_obs["status"] == validation_data.STATUS_TABLE_MISMATCH
    assert "obs_Z" in unknown_obs["reason"]


# --------------------------------------------------------------------- the scoring projection
_REF = {"nll": -66.58, "k": 19, "n": 144,
        "validation": {"status": "ok", "nll": 500.0, "n": 276, "chi2": 1000.0}}


_JUDGE_REF = {"nll": 100.0, "k": 3, "n": 50,
              "validation": {"status": "ok", "nll": 400.0, "n": 12, "chi2": 90.0}}


_RAISES = object()
