"""An example may STATE its reference model differently from the published export — a declarative
patch that re-derives an initial value the export froze to a number, so a TRANSFER set that refits
can move it (``benchmark._stated_problem``, ``benchmark.patch_reference_model``). Tested on the
toy: the patch's own guarantees, and the reference routing for both kinds of held-out set."""

from __future__ import annotations

import json
from pathlib import Path

import libsbml
import pandas as pd
import pytest

from gemot import benchmark as B
from petab_mcp import validation_data as V

TOY = Path(__file__).parent / "data" / "toy_v1"


def _toy():
    import petab.v1 as petab

    return petab.Problem.from_yaml(str(TOY / "problem.yaml"))


PATCH = {"parameters": {"V0": 1.0}, "initial_assignments": {"k2": "0.1 * V0"}}


def test_the_patch_adds_constants_and_sets_initial_values_leaving_the_pristine_model_alone():
    """Declarative and readable in run.py: new constant parameters, and an initial value given as
    an expression over them where the export had a number."""
    problem = _toy()

    patched, touched = B.patch_reference_model(problem.model, PATCH)

    sbml = patched.sbml_model
    assert touched == ["k2(0)"]
    assert sbml.getParameter("V0").getValue() == 1.0 and sbml.getParameter("V0").getConstant()
    ia = sbml.getInitialAssignmentBySymbol("k2")
    assert libsbml.formulaToL3String(ia.getMath()) == "0.1 * V0"
    assert problem.model.sbml_model.getParameter("V0") is None
    assert problem.model.sbml_model.getInitialAssignmentBySymbol("k2") is None


@pytest.mark.parametrize(
    "patch, needle",
    [
        ({"parameters": {"k1": 1.0}}, "already in the reference model"),
        ({"initial_assignments": {"k2": "0.1 * V0"}}, "does not define"),
        ({"initial_assignments": {"nothing": "1"}}, "not in the reference model"),
        ({"initial_assignments": {"A": "2"}}, "already has an initial assignment"),
    ],
)
def test_a_patch_that_would_not_do_what_it_says_is_refused(patch, needle):
    with pytest.raises(ValueError, match=needle):
        B.patch_reference_model(_toy().model, patch)


def test_an_initial_assignment_cannot_override_an_assignment_rule():
    """libSBML would reject the document at build time; the patch says so instead."""
    problem = _toy()
    sbml = problem.model.sbml_model
    rule = sbml.createAssignmentRule()
    rule.setVariable("k2")
    rule.setMath(libsbml.parseL3Formula("0.2 * k1"))

    with pytest.raises(ValueError, match="assignment rule"):
        B.patch_reference_model(problem.model, {"initial_assignments": {"k2": "0.1"}})


def test_the_stated_problem_carries_the_patched_model_and_a_fixed_row_for_what_it_added():
    """The stated problem is what every held-out check and refit runs against; an example that
    declares no patch gets the published problem itself. The patch's new constants get FIXED rows
    at their patch value, so the published table has a counterpart for them."""
    problem = _toy()

    stated = B._stated_problem("X", problem, PATCH)

    assert stated.model.has_entity_with_id("V0") and not problem.model.has_entity_with_id("V0")
    assert "V0" in stated.parameter_df.index and "V0" not in problem.parameter_df.index
    assert list(stated.parameter_df.columns) == list(problem.parameter_df.columns)
    row = stated.parameter_df.loc["V0"]
    assert int(row["estimate"]) == 0 and float(row["nominalValue"]) == 1.0
    assert B._stated_problem("X", problem, None) is problem


def _curated(tmp_path, condition):
    d = tmp_path / "curated"
    d.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "observableId": ["obs_B", "obs_B"],
            "simulationConditionId": [condition] * 2,
            "time": [0.0, 1.0],
            "measurement": [1.0, 1.0],
        }
    ).to_csv(d / "measurements.tsv", sep="\t", index=False)
    pd.DataFrame({"conditionId": [condition]}).to_csv(
        d / V.HELD_OUT_CONDITIONS, sep="\t", index=False
    )
    return "curated/measurements.tsv"


def test_a_transfer_set_is_refit_on_the_stated_problem(tmp_path, monkeypatch):
    """The refit problem is the stated one, so the reference moves the patched initial value the
    way the transfer parameter table asks it to."""
    from gemot import simulate

    monkeypatch.setattr(B, "_load_problem", lambda *a, **k: _toy())
    (tmp_path / "problem").mkdir()
    seen = {}

    def _fake_training(problem, *, path, held_out=None, **_):
        seen["held_out"] = held_out
        ref = {"nll": 1.0, "k": 2, "n": 6}
        Path(path).write_text(json.dumps(ref))
        return ref

    def _fake_refit(problem, *, build_dir, model_name, n_starts, units=None):
        seen["refit_model"] = problem.model
        seen["refit_parameters"] = problem.parameter_df
        return {"status": "ok", "mode": "transfer", "nll": 3.0, "k": 2, "n": 2}

    monkeypatch.setattr(simulate, "write_reference_from_problem", _fake_training)
    monkeypatch.setattr(V, "refit_problem", _fake_refit)
    table = _curated(tmp_path, "c_new")
    (tmp_path / "curated" / V.HELD_OUT_PARAMETERS).write_text(
        "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
        "V0\tlin\t0.01\t100.0\t2.0\t1\n"
    )

    B.write_reference_for(
        name="X", example_dir=tmp_path, validation_table=table, reference_patch=PATCH
    )

    assert seen["held_out"] is None, "a transfer set is refit, not scored at the training vector"
    assert seen["refit_model"].has_entity_with_id("V0")
    assert seen["refit_model"].sbml_model.getInitialAssignmentBySymbol("k2") is not None
    # the held-out table may ESTIMATE a constant the patch added; the published TABLE has a
    # counterpart for it because `_stated_problem` states one
    assert int(seen["refit_parameters"].loc["V0", "estimate"]) == 1


def test_a_patch_with_a_fixed_theta_set_is_refused(tmp_path, monkeypatch):
    """A fixed-θ set is scored on the pristine structure, so declaring a patch beside one would
    drop it without saying so."""
    monkeypatch.setattr(B, "_load_problem", lambda *a, **k: _toy())
    (tmp_path / "problem").mkdir()
    table = _curated(tmp_path, "c_new")

    with pytest.raises(ValueError, match="only applied where the reference is REFIT"):
        B.write_reference_for(
            name="X", example_dir=tmp_path, validation_table=table, reference_patch=PATCH
        )


def test_materialize_checks_the_held_out_set_against_the_stated_problem(tmp_path, monkeypatch):
    """The other entry point. `materialize` runs the same held-out checks as `write_reference_for`
    and must hand them the same stated problem, or a transfer table that estimates a patch-added
    constant is refused on the setup path while passing on the refit one."""
    monkeypatch.setattr(B, "_load_problem", lambda *a, **k: _toy())
    table = _curated(tmp_path, "c_new")
    (tmp_path / "curated" / V.HELD_OUT_PARAMETERS).write_text(
        "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
        "V0\tlin\t0.01\t100.0\t2.0\t1\n"
    )

    B.materialize(
        name="X",
        shipped_param_ids=["sigma_B", "A_init", "B_init"],
        example_dir=tmp_path,
        validation_table=table,
        reference_patch=PATCH,
    )

    held = V.read_held_out(B.validation_measurements_path(tmp_path))
    assert held.mode == V.MODE_TRANSFER
    assert int(held.parameter_df.loc["V0", "estimate"]) == 1
