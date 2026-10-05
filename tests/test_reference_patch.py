"""An example may STATE its reference model differently from the published export — a declarative
patch that re-derives an initial value the export froze to a number, so a TRANSFER set that refits
can move it (``benchmark._stated_problem``, ``benchmark.patch_reference_model``). Tested on the
toy: the patch's own guarantees here, and the reference routing for both kinds of held-out set in
``tests/server/test_reference_patch.py``, which reads the server's held-out format."""

from __future__ import annotations

from pathlib import Path

import libsbml
import pytest

from gemot import benchmark as B


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
