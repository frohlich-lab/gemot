"""The fixed-θ HELD-OUT check: :func:`petab_mcp.validation_data.score_problem` and its guards.

Tested at the boundary rather than through an end-to-end run: which tables the scorer copies, what
it declines before paying for a build, and how it refuses a parameter vector it cannot line up.
The objective evaluation past those guards needs an AMICI build; its correctness is pinned by the
reference's own held-out score in BENCHMARK_NOTES.md, re-derived by ``setup.py --refit``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from petab_harness import benchmark


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
            name="X", validation_yaml="held_out.yaml", collection=None,
            published=published, training_measurement_df=train)



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
            name="X", validation_yaml="held_out.yaml", collection=None,
            published=published, training_measurement_df=train)



def test_an_absent_validation_yaml_names_itself(monkeypatch, tmp_path):
    """A typo'd path must say which path, at setup time. Falling through to "no held-out data" would
    leave the example silently unvalidated for as long as nobody read the setup log."""
    monkeypatch.setattr(benchmark, "benchmark_dir", lambda *a, **k: tmp_path)
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        benchmark._validation_measurement_df(
            name="X", validation_yaml="nope.yaml", collection=None,
            published=_problem(_rows("c_train")), training_measurement_df=_rows("c_train"))



def test_the_held_out_table_lands_outside_the_problem_directory():
    """Structural, not incidental: the runner copies ``problem/`` into ``work/`` and points the MCP
    server at it, so a held-out table inside ``problem/`` would sit in the tree the agent's server
    reads. The path helper is what every writer and reader goes through, so pin it."""
    path = benchmark.validation_measurements_path(benchmark.Path("/tmp/ex"))
    assert path.parent.name == benchmark.VALIDATION_DIR
    assert path.parent.name != "problem"
    assert path.name == benchmark.VALIDATION_MEASUREMENTS


def test_a_time_written_as_1_and_1_0_is_the_same_held_out_row(monkeypatch, tmp_path):
    """The two tables are parsed from different TSVs, so one can render a timepoint as `1` and the
    other as `1.0`. Stringifying the key made those different rows and let a genuinely in-sample
    row past the disjointness check — the one guard that keeps the held-out score out-of-sample."""
    import pandas as pd

    def _table(time_value):
        return pd.DataFrame({"observableId": ["obs_A"], "simulationConditionId": ["c1"],
                             "time": [time_value], "measurement": [1.0]})

    class _V:
        measurement_df = _table(1.0)
        observable_df = _problem(_rows("c1")).observable_df
    monkeypatch.setattr("petab.v1.lint.check_measurement_df", lambda *a, **k: None)
    monkeypatch.setattr("petab_mcp.fitting.needs_flattening", lambda *a, **k: False)

    with pytest.raises(ValueError, match="also in the TRAINING table"):
        benchmark._check_held_out_rows("Toy", _V(), _problem(_rows("c1")), _table(1))
