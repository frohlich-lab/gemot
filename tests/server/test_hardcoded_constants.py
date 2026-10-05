"""Hardcoded constants are priced into ``k`` (``petab_mcp.literals``, issue #180).

Every complexity-penalised score charged ``k = n_estimated``, so a value written straight into a
rate law cost nothing however it was obtained: fujita run ``53685907_4`` had 21 reactions, 10
species and ``k = 4`` — its rate laws read ``0.000878``, ``2.69e-5``, ``1.333``… — and scored best
of 29 runs. Two halves are pinned here: the RULE (which literals count, on libsbml ASTs built the
way the server builds them, from Antimony) and the ASSEMBLY (every place ``k`` is derived reads the
same addend, from the fit payload to the stored version's bic to the judge).
"""

from __future__ import annotations

import asyncio
import math

import antimony
import libsbml
import pytest

from conftest import DATA_DIR

from petab_mcp import fitting
from petab_mcp import literals as L
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server
from gemot.runner.judge import _candidate_k


def _sbml_model(ant: str):
    antimony.clearPreviousLoads()
    assert antimony.loadAntimonyString(ant) >= 0, antimony.getLastError()
    doc = libsbml.readSBMLFromString(antimony.getSBMLString(antimony.getMainModuleName()))
    return doc.getModel()


def _rate_law(law: str) -> str:
    return f"model m\n  species A, B;\n  A = 5; B = 0;\n  kf = 0.5;\n  R1: A -> B; {law};\nend\n"


# ------------------------------------------------------------------------------ the rule
@pytest.mark.parametrize(
    "law, priced",
    [
        # the literals of the motivating run: fitted values, frozen in
        ("0.000878*A", ["0.000878"]),
        ("2.69e-5*A", ["2.69e-5"]),
        ("kf*A^1.333", ["1.333"]),
        ("kf*A/(0.0305 + A)", ["0.0305"]),
        # published-looking numbers, integer or not
        ("68190*kf", ["68190"]),
        ("kf*A/(12.5 + A)", ["12.5"]),
        # STRUCTURAL_VALUES stay free: zero, the identities, the oligomer multiplicities and their
        # reciprocals, however spelled
        ("0*A + 1*B", []),
        ("2*kf*A", []),
        ("kf*A^2/(1 + A^2)", []),
        ("kf*A^3", []),
        ("kf*A^4", []),
        ("6*kf*A", []),
        ("0.5*kf*A - 0.25*kf*B", []),  # 1/2 and 1/4 by value
        ("kf*A*(1/3) + kf*B*(1/6)", []),
        ("kf*A/6 - kf*B/4", []),
        ("100*kf*A", []),  # percent: a unit with nothing to absorb it into
        ("kf*A/100", []),
        # ...and nothing else is. A unit or scale factor is a magnitude like any other: name it, or
        # pay for it. These were all free under the old three-significant-digit threshold.
        ("1000000*kf*A", ["1000000"]),
        ("60*kf*A + 3600*kf*B", ["60", "3600"]),
        ("5*kf*A", ["5"]),  # 5 is not an oligomer size and not in the set
        ("5e-5*A", ["5e-5"]),
        # exact rational comparison: a rounded third is not a third
        ("kf*A*0.333", ["0.333"]),
    ],
)
def test_only_the_structural_multiplicities_are_free(law, priced):
    """The free set is enumerated (`literals.STRUCTURAL_VALUES`), not a threshold, so there is no
    rounding move onto it: the nearest free value to a fitted rate constant is `0`, which deletes
    the term. Under the old three-digit rule `0.878` could be written `0.88` for nothing, and
    `1000000`/`60`/`3600` were free for having few digits rather than for being structural.

    Membership is by exact rational VALUE, so `0.5` IS `1/2` and free while `0.333` is not `1/3`.
    """
    assert L.hardcoded_constants(_sbml_model(_rate_law(law))) == priced


def test_the_free_set_is_closed_under_sign_and_reciprocal():
    """Both directions of every multiplicity, because `2*x`, `x/2`, `-2*x` and `-x/2` are the same
    structural statement written four ways — and a rule that charged only some of them would push
    an author toward the spelling that happens to be free."""
    from fractions import Fraction

    for n in (1, 2, 3, 4, 6, 100):
        for v in (Fraction(n), -Fraction(n), Fraction(1, n), -Fraction(1, n)):
            assert v in L.STRUCTURAL_VALUES, v
    assert Fraction(0) in L.STRUCTURAL_VALUES
    # 5 and its reciprocal are deliberately absent: no oligomer is a pentamer here, and every
    # value admitted is a value a fitted number could be rounded onto.
    assert Fraction(5) not in L.STRUCTURAL_VALUES
    assert Fraction(1, 5) not in L.STRUCTURAL_VALUES


def test_each_distinct_value_is_charged_once_however_written():
    """`1.333` appears in three of the motivating run's rate laws as ONE shared exponent — one
    degree of freedom, one parameter. Two spellings of the same value are the same constant."""
    ant = (
        "model m\n  species A, B, C;\n  A = 5; B = 0; C = 0;\n  kf = 0.5;\n"
        "  R1: A -> B; kf*A^1.333;\n"
        "  R2: B -> C; kf*B^1.333 + 2.69e-5*B;\n"
        "  R3: C -> A; 0.0000269*C + kf*C^1.333;\nend\n"
    )
    assert L.hardcoded_constants(_sbml_model(ant)) == ["2.69e-5", "1.333"]


def test_every_math_container_is_scanned_but_species_initials_and_parameter_values_are_not():
    """A literal is priced wherever the model puts math — an assignment rule (the readout route),
    an initial assignment, a function body, an event — because each is a place a fitted value can
    be frozen. A species initial value is NOT scanned: the reconcile turns a nonzero, non-unit one
    into an ESTIMATED `<sid>_init` row already, so scanning it would double-charge. A parameter's
    model value is overridden by the table, so it is not scanned either. `2.5` in the assignment
    IS charged — two significant digits bought nothing once the free set became a fixed list."""
    ant = (
        "function hill(y)\n  y^1.333/(0.0305^1.333 + y^1.333)\nend\n"
        "model m\n  species A, B;\n  A = 5.678; B = 0;\n  kf = 0.000878;\n  scale = 0.5;\n"
        "  R1: A -> B; kf*hill(A);\n"
        "  readout_B := 0.0345*B;\n"
        "  scale = 2.5*kf;\n"
        "  E1: at (time > 10.5): A = 1.234;\nend\n"
    )
    sm = _sbml_model(ant)
    assert L.hardcoded_constants(sm) == ["0.0305", "0.0345", "1.234", "1.333", "2.5", "10.5"]


# --------------------------------------------------------------------------- the assembly
def test_complexity_k_adds_the_constants_to_the_estimated_rows():
    assert (
        fitting.complexity_k({"n_estimated": 4, "hardcoded_constants": ["0.000878", "1.333"]}) == 6
    )
    # a payload from before the field existed charges none; one without n_estimated has no k at all
    assert fitting.complexity_k({"n_estimated": 4}) == 4
    assert fitting.complexity_k({"hardcoded_constants": ["1.333"]}) is None
    assert fitting.complexity_k(None) is None


def test_the_fit_result_block_prices_the_constants_and_names_them():
    """The block's bic must charge the literals AND say so: a `k` above `n_estimated` with nothing
    to explain it reads as a bug, and the agent cannot act on a number it is not shown."""
    fitsim = {
        "status": "fit",
        "neg_log_likelihood": -50.0,
        "chi2": 18.0,
        "n_data_points": 100,
        "n_estimated": 4,
        "hardcoded_constants": ["0.000878", "2.69e-5", "1.333"],
        "fitted_parameters": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0},
    }
    block = fitting.fit_result_block(fitsim)
    assert block["n_estimated"] == 4 and block["k"] == 7
    assert block["hardcoded_constants"] == ["0.000878", "2.69e-5", "1.333"]
    assert block["bic"] == pytest.approx(7 * math.log(100) + 2 * -50.0)
    # ...and the list rides only when there is something to explain: it is re-sent every turn
    clean = fitting.fit_result_block({**fitsim, "hardcoded_constants": []})
    assert clean["k"] == 4 and "hardcoded_constants" not in clean


def test_a_new_best_is_judged_on_the_priced_bic(session):
    """`is_new_best` tracks BIC, so freezing four fitted values must not register as an
    improvement over estimating them: same nll, same k, same bic."""
    estimated = {
        "status": "fit",
        "neg_log_likelihood": -50.0,
        "chi2": 18.0,
        "n_data_points": 100,
        "n_estimated": 8,
        "hardcoded_constants": [],
    }
    frozen = {
        **estimated,
        "n_estimated": 4,
        "hardcoded_constants": ["0.000878", "2.69e-5", "1.333", "0.0305"],
    }
    a = fitting.compute_improvement(session, nllh=-50.0, objective_before=None, fitsim=estimated)
    b = fitting.compute_improvement(session, nllh=-50.0, objective_before=None, fitsim=frozen)
    assert a["bic"] == pytest.approx(b["bic"])


def test_the_streak_checkpoint_and_the_stored_version_use_the_same_k(session):
    """The early-stop checkpoint (`record_fit_progress`) and the version meta the judge scans
    (`_snapshot_version`) must price the literals too, or a frozen model would become the
    best-BIC checkpoint the judge restores while the fit block said otherwise."""
    fitsim = {
        "status": "fit",
        "neg_log_likelihood": -50.0,
        "chi2": 18.0,
        "n_data_points": 100,
        "n_estimated": 4,
        "hardcoded_constants": ["0.000878", "1.333"],
        "fitted_parameters": {"k1": 0.5, "k2": 0.1},
    }
    expected = 6 * math.log(100) + 2 * -50.0
    session.record_fit_progress(0, fitsim, {"is_new_best": True})
    # Checkpoint.score is the run's criterion — bic under the default the session resolves to.
    assert session.best.score == pytest.approx(expected)
    v = session._snapshot_version(meta={"neg_log_likelihood": -50.0}, sim=fitsim)
    assert session.provenance.get_meta(v)["bic"] == pytest.approx(expected)


_TOY_BASE = DATA_DIR / "problem.yaml"


def _server(tmp_path, auto_fit=False):
    return create_server(
        str(_TOY_BASE),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=auto_fit,
    )


_FROZEN_TOY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  r1: A => B; k1 * A;
  r2: B => A; 0.0987 * B;
end
"""


def test_the_server_reports_the_constants_on_the_edit_result_and_the_summary(tmp_path):
    """The real call path, no fit (auto_fit off): re-authoring the toy with `k2` frozen to
    `0.0987` shows up on the `make_changes` result and on `get_fit_summary` as one hardcoded
    constant, with `k` one above the estimated-row count on both reads."""
    mcp = _server(tmp_path)

    async def go():
        res = (
            await mcp.call_tool(
                "make_changes",
                {
                    "antimony": _FROZEN_TOY,
                    "justification": "freeze k2 at its fitted value",
                },
            )
        ).structured_content
        assert res["applied"] is True and res["committed"] is True
        summary = (await mcp.call_tool("get_fit_summary", {})).structured_content
        assert summary["hardcoded_constants"] == ["0.0987"]
        assert summary["k"] == summary["n_estimated"] + 1
        params = (await mcp.call_tool("get_parameters", {})).structured_content
        rows = params if isinstance(params, list) else params.get("result")
        assert _candidate_k(rows, summary)[0] == summary["k"]

    asyncio.run(go())


def test_a_fit_carries_the_constants_through_every_score(fit_session):
    """The fit path proper: the payload `fit_and_simulate` stores, the fit block the agent reads,
    the improvement it is ranked by, the stored version's meta and `get_fit_summary` all charge
    the frozen `0.0987`. This is the assembly that a green unit suite cannot vouch for."""
    from petab_mcp import operations as O

    sess = fit_session
    O.set_model(sess, _FROZEN_TOY, justification="freeze k2 at its fitted value")
    res = sess.fit(n_starts=2)
    assert res["committed"] is True
    sim = sess.simulation
    assert sim["hardcoded_constants"] == ["0.0987"]
    k = sim["n_estimated"] + 1
    assert res["k"] == k and res["hardcoded_constants"] == ["0.0987"]
    expected = k * math.log(sim["n_data_points"]) + 2 * sim["neg_log_likelihood"]
    assert res["bic"] == pytest.approx(expected)
    assert res["improvement"]["bic"] == pytest.approx(expected)
    assert sess.provenance.get_meta(res["version"])["bic"] == pytest.approx(expected)
    summary = fitting.fit_summary(sess, sess.engine)
    assert summary["k"] == k and summary["bic"] == pytest.approx(expected)
