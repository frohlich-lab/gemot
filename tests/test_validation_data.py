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
        validation_data.score_problem(problem, _rows("c_test"),
                                      build_dir="/tmp/train_build", model_name="train_module")

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



def test_the_server_only_registers_the_tool_when_given_held_out_data(monkeypatch, yaml_path,
                                                                    tmp_path):
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
    monkeypatch.setattr(server_mod, "_register_validation_data_tools",
                        lambda mcp, ctx, held_out: calls.append(held_out))
    assert "validation_measurements" in inspect.signature(create_server).parameters

    create_server(yaml_path, build_root=str(tmp_path / "b1"),
                  provenance_dir=str(tmp_path / "prov1"))
    assert calls == []

    meas = tmp_path / "held.tsv"
    _rows("c_test").to_csv(meas, sep="\t", index=False)
    create_server(yaml_path, build_root=str(tmp_path / "b2"),
                  provenance_dir=str(tmp_path / "prov2"),
                  validation_measurements=str(meas))
    assert len(calls) == 1
    assert isinstance(calls[0], validation_data.HeldOut)
    assert len(calls[0].measurement_df) == 2
