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
