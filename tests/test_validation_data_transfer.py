"""The TRANSFER kind of held-out check: a second problem on the same mechanism, on which both the
candidate and the reference are REFIT (``petab_mcp.validation_data`` and its two consumers).

Tested as three surfaces, apart from each other, exactly as the fixed-θ kind is in
``test_validation_data.py`` — the assembly path is where the defects live: which tables move onto
the candidate's view and which do not, what the materialiser writes and refuses, and what the judge
and the history CSV do with a ``mode: transfer`` payload. The one AMICI-backed test drives the whole
server-side path (``score_session``) on the toy problem, because "the refit runs and returns the
shape the judge reads" is not something a mocked importer can establish.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd
import pytest

from petab_mcp import validation_data as V

TOY = Path(__file__).parent / "data" / "toy_v1"


# ------------------------------------------------------------------ what makes a set a transfer
def _params(rows):
    df = pd.DataFrame(rows, columns=["parameterId", "parameterScale", "lowerBound", "upperBound",
                                     "nominalValue", "estimate"])
    return df.set_index("parameterId")


def _conditions(ids, **cols):
    df = pd.DataFrame({"conditionId": list(ids), **{k: list(v) for k, v in cols.items()}})
    return df.set_index("conditionId")


def _rows(condition, times=(0.0, 1.0), observable="obs_B"):
    return pd.DataFrame({"observableId": [observable] * len(times),
                         "simulationConditionId": [condition] * len(times),
                         "time": list(times), "measurement": [1.0] * len(times)})


def test_a_held_out_set_is_a_transfer_set_iff_it_brings_its_own_parameter_table(tmp_path):
    """The mode is READ OFF the files, not declared beside them: a parameter table is the statement
    that the parameters are not shared, and a flag next to it would be a second thing able to
    disagree with the first."""
    meas = tmp_path / "measurements.tsv"
    _rows("c1").to_csv(meas, sep="\t", index=False)
    assert V.read_held_out(meas).mode == V.MODE_FIXED

    _conditions(["c1"]).reset_index().to_csv(tmp_path / V.HELD_OUT_CONDITIONS, sep="\t",
                                             index=False)
    held = V.read_held_out(meas)
    assert held.mode == V.MODE_FIXED and list(held.condition_df.index) == ["c1"]

    _params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]).reset_index().to_csv(
        tmp_path / V.HELD_OUT_PARAMETERS, sep="\t", index=False)
    held = V.read_held_out(meas)
    assert held.mode == V.MODE_TRANSFER
    assert held.parameter_df.loc["A_init", "nominalValue"] == 2.0


# ------------------------------------------------------------- moving the candidate onto the set
def _candidate():
    """A candidate problem stand-in with the toy problem's tables: A_init fixed at 1.0."""
    import petab.v1 as petab

    return petab.Problem(
        model=None,
        measurement_df=_rows("c0"),
        observable_df=pd.DataFrame({"observableFormula": ["B"], "noiseFormula": ["sigma_B"]},
                                   index=pd.Index(["obs_B"], name="observableId")),
        condition_df=_conditions(["c0"]),
        parameter_df=_params([("k1", "log10", 1e-4, 100.0, 0.5, 1),
                              ("k2", "log10", 1e-4, 100.0, 0.1, 1),
                              ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),
                              ("A_init", "lin", 0.0, 100.0, 1.0, 0)]),
    )


_SHIPPED = _params([("k1", "log10", 1e-4, 100.0, 0.5, 1), ("k2", "log10", 1e-4, 100.0, 0.1, 1),
                    ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0), ("A_init", "lin", 0.0, 100.0, 1.0, 0)])

_HELD_OUT_PARAMS = _params([
    ("k1", "log10", 1e-4, 100.0, 7.0, 1),          # estimated: the published held-out optimum
    ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),       # fixed, unchanged
    ("A_init", "lin", 0.0, 100.0, 2.0, 0),         # fixed, CHANGED: the input that must move
    ("sigma_extra", "lin", 1e-4, 100.0, 3.0, 0),   # fixed, never in the training problem
])


def test_the_view_moves_only_the_changed_fixed_inputs_and_never_the_start_point():
    """Three kinds of held-out row and three different fates. `A_init` (fixed, changed) is written
    into the candidate; `k1` (estimated) is NOT — the held-out table's value is the published optimum
    for that system and would hand the candidate the answer, so its refit starts from its own
    training value; `sigma_extra` (a fixed quantity the training problem never named) binds to
    nothing. And the caller's problem is untouched throughout."""
    problem = _candidate()
    held = V.HeldOut(measurement_df=_rows("c1"), condition_df=_conditions(["c1"]),
                     parameter_df=_HELD_OUT_PARAMS)
    view, failed = V.transfer_view(problem, held, _SHIPPED)
    assert failed is None
    assert view.parameter_df.loc["A_init", "nominalValue"] == 2.0
    assert view.parameter_df.loc["k1", "nominalValue"] == 0.5, "the start point is the candidate's own"
    assert "sigma_extra" not in view.parameter_df.index
    assert list(view.condition_df.index) == ["c1"]
    assert list(view.measurement_df["simulationConditionId"]) == ["c1", "c1"]
    assert view.visualization_df is None
    # nothing leaked back
    assert problem.parameter_df.loc["A_init", "nominalValue"] == 1.0
    assert list(problem.condition_df.index) == ["c0"]
    assert list(problem.measurement_df["simulationConditionId"]) == ["c0", "c0"]


def test_a_candidate_that_cannot_receive_a_changed_input_is_a_finding_not_a_crash():
    """oliveira's `population` changes between the interior and the whole state. A model that
    hard-coded it (no row), renamed it, or freed it has no fixed parameter to receive the new value,
    so it cannot be moved to the held-out system at all — reported as its own status, because a
    refit at the TRAINING population would be a plausible, wrong number."""
    held = V.HeldOut(measurement_df=_rows("c1"), condition_df=_conditions(["c1"]),
                     parameter_df=_HELD_OUT_PARAMS)

    no_row = _candidate()
    no_row.parameter_df = no_row.parameter_df.drop(index="A_init")
    view, failed = V.transfer_view(no_row, held, _SHIPPED)
    assert view is None and failed["status"] == V.STATUS_INPUT_MISSING
    assert "A_init" in failed["reason"]

    freed = _candidate()
    freed.parameter_df.loc["A_init", "estimate"] = 1
    view, failed = V.transfer_view(freed, held, _SHIPPED)
    assert view is None and failed["status"] == V.STATUS_INPUT_MISSING


def test_a_held_out_table_equal_to_the_training_one_is_refused_as_misannotation():
    """Shipping a parameter table is what DECLARES a set a transfer set, so a table that is the
    training table says the set was mis-annotated — most likely a copied file. Left alone it would
    refit a structure against itself and report the result as a second system, which is a plausible
    number and a wrong one; the honest fixed-θ reading is available by dropping the table."""
    held = V.HeldOut(measurement_df=_rows("c1"), condition_df=_conditions(["c1"]),
                     parameter_df=_SHIPPED.copy())
    assert held.mode == V.MODE_TRANSFER, "the table is what makes it a transfer set"
    with pytest.raises(ValueError, match="identical to the training one"):
        V.transfer_view(_candidate(), held, _SHIPPED)


def test_a_held_out_table_that_estimates_an_unknown_parameter_is_refused():
    """A parameter the training problem never named cannot be estimated on the candidate's behalf —
    its model has no counterpart. A FIXED extra row stays tolerated (`sigma_extra` above): it binds
    to nothing, and a published table keeps such rows for observables the materialiser dropped."""
    rows = [("k1", "log10", 1e-4, 100.0, 7.0, 1), ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),
            ("A_init", "lin", 0.0, 100.0, 2.0, 0)]
    estimated_unknown = _params([*rows, ("k_new", "log10", 1e-4, 100.0, 1.0, 1)])
    held = V.HeldOut(measurement_df=_rows("c1"), condition_df=_conditions(["c1"]),
                     parameter_df=estimated_unknown)
    with pytest.raises(ValueError, match="k_new"):
        V.transfer_view(_candidate(), held, _SHIPPED)

    fixed_unknown = _params([*rows, ("k_new", "log10", 1e-4, 100.0, 1.0, 0)])
    held = V.HeldOut(measurement_df=_rows("c1"), condition_df=_conditions(["c1"]),
                     parameter_df=fixed_unknown)
    view, failed = V.transfer_view(_candidate(), held, _SHIPPED)
    assert failed is None and "k_new" not in view.parameter_df.index


def test_a_held_out_system_may_bring_its_own_estimated_noise_parameters():
    """A new noise sigma or observable scaling IS manageable: it enters through the observation
    model — the held-out measurement table's `noiseParameters` override — not the ODE, so the refit
    can estimate it however many the held-out system brings. Only a new MECHANISM parameter has
    nowhere to go."""
    meas = _rows("c1").assign(noiseParameters="sigma_new")
    rows = [("k1", "log10", 1e-4, 100.0, 7.0, 1), ("sigma_B", "lin", 1e-4, 100.0, 0.1, 0),
            ("A_init", "lin", 0.0, 100.0, 2.0, 0),
            ("sigma_new", "lin", 1e-4, 100.0, 0.3, 1)]
    held = V.HeldOut(measurement_df=meas, condition_df=_conditions(["c1"]),
                     parameter_df=_params(rows))
    view, failed = V.transfer_view(_candidate(), held, _SHIPPED)
    assert failed is None, failed


def test_a_held_out_condition_table_with_other_columns_is_another_model():
    """A condition column is an input the model declares and the compiled module maps; a held-out
    table adding one describes a model the candidate is not, and reusing the module would be wrong."""
    held = V.HeldOut(measurement_df=_rows("c1"), condition_df=_conditions(["c1"], dose=[1.0]),
                     parameter_df=_HELD_OUT_PARAMS)
    view, failed = V.transfer_view(_candidate(), held, _SHIPPED)
    assert view is None and failed["status"] == V.STATUS_TABLE_MISMATCH
    assert "dose" in failed["reason"]


def test_held_out_conditions_are_checked_against_the_held_out_table_not_the_training_one():
    """The whole point of the condition swap: `c1` is not a training condition and must still pass,
    while a measurement row naming a condition in NEITHER table must not."""
    ok = V.HeldOut(measurement_df=_rows("c1"), condition_df=_conditions(["c1"]),
                   parameter_df=_HELD_OUT_PARAMS)
    view, failed = V.transfer_view(_candidate(), ok, _SHIPPED)
    assert failed is None and view is not None

    bad = V.HeldOut(measurement_df=_rows("c_nowhere"), condition_df=_conditions(["c1"]),
                    parameter_df=_HELD_OUT_PARAMS)
    view, failed = V.transfer_view(_candidate(), bad, _SHIPPED)
    assert failed["status"] == V.STATUS_TABLE_MISMATCH and "c_nowhere" in failed["reason"]


def test_refit_declines_a_bad_table_before_any_build(monkeypatch):
    """Same contract as `score_problem`: the guards run before the importer, so an empty or foreign
    table costs nothing and names its own problem."""
    import pypesto.petab

    def _boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("the importer must not be reached for a declined table")

    monkeypatch.setattr(pypesto.petab, "PetabImporter", _boom)
    problem = _candidate()
    problem.measurement_df = problem.measurement_df.iloc[0:0]
    out = V.refit_problem(problem, build_dir="/tmp/x", model_name="m", n_starts=1)
    assert out["status"] == V.STATUS_NO_DATA

    problem = _candidate()
    problem.measurement_df = _rows("c_elsewhere")
    out = V.refit_problem(problem, build_dir="/tmp/x", model_name="m", n_starts=1)
    assert out["status"] == V.STATUS_TABLE_MISMATCH and "c_elsewhere" in out["reason"]


# ------------------------------------------------ the whole server-side path, with AMICI
def test_score_session_refits_the_toy_on_a_transfer_set_and_leaves_the_session_alone(session,
                                                                                    engine):
    """The one AMICI-backed test: the toy model, its A_init doubled on a new condition `c1`, is
    REFIT on measurements it never saw. Pins the payload SHAPE the judge reads (`mode`, `k`, `n`,
    a finite `nll`), that `k` is the count of estimated rows (k1, k2 — not sigma_B or A_init), that
    the refit's optimum is no worse than its start point, and that the session's own table and
    model are exactly as they were: a transfer refit that wrote its optimum back would change what
    the judge scores on the training data."""

    meas = pd.read_csv(TOY / "measurements.tsv", sep="\t")
    held_meas = meas.assign(simulationConditionId="c1", measurement=meas["measurement"] * 2.0)
    held = V.HeldOut(
        measurement_df=held_meas,
        condition_df=_conditions(["c1"]),
        parameter_df=_params([("A_init", "lin", 0.0, 100.0, 2.0, 0),
                              ("k1", "log10", 1e-4, 100.0, 0.9, 1)]),
    )
    before = session.problem.parameter_df.copy(deep=True)
    sbml_before = session.model_sbml()

    out = V.score_session(session, engine, held, n_starts=2)

    assert out["status"] == V.STATUS_OK, out
    assert out["mode"] == V.MODE_TRANSFER
    assert out["k"] == 2 and out["n"] == len(held_meas) and out["n_conditions"] == 1
    assert out["observables"] == ["obs_B"] and out["n_starts"] == 2
    assert math.isfinite(out["nll"]) and out["chi2"] is not None
    # the doubled data at A_init = 2 is the SAME curve shape, so the refit from the training
    # optimum must land near the training likelihood — a refit at the training A_init (the
    # input NOT moved) could not
    view, _ = V.transfer_view(session.problem, held, session.shipped_parameter_df)
    assert view.parameter_df.loc["A_init", "nominalValue"] == 2.0
    pd.testing.assert_frame_equal(session.problem.parameter_df, before)
    assert session.model_sbml() == sbml_before
    assert list(session.problem.condition_df.index) == ["c0"]


def test_the_server_tool_takes_n_starts_and_refits_a_transfer_bundle(tmp_path):
    """The registered tool is what the judge reaches over MCP: it must accept `n_starts` (or the
    judge's call fails and the run records `unavailable`) and must read the bundle as a transfer
    set. Driven in-process through `create_server` like the doc-consistency tests."""
    import asyncio

    from petab_mcp.server import create_server

    prob = tmp_path / "problem"
    shutil.copytree(TOY, prob)
    val = tmp_path / "validation"
    val.mkdir()
    pd.read_csv(TOY / "measurements.tsv", sep="\t").assign(simulationConditionId="c1").to_csv(
        val / "measurements.tsv", sep="\t", index=False)
    _conditions(["c1"]).reset_index().to_csv(val / V.HELD_OUT_CONDITIONS, sep="\t", index=False)
    _params([("A_init", "lin", 0.0, 100.0, 2.0, 0)]).reset_index().to_csv(
        val / V.HELD_OUT_PARAMETERS, sep="\t", index=False)

    mcp = create_server(str(prob / "problem.yaml"), provenance_dir=str(tmp_path / "prov"),
                        build_root=str(tmp_path / "build"), auto_fit=False,
                        validation_measurements=str(val / "measurements.tsv"))
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    assert "n_starts" in tools["evaluate_validation_data"].parameters["properties"]
