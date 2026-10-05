"""The judge's half of pricing hardcoded constants into ``k`` (issue #180): it assembles ``k``
from its own reads the same way the server does, records the vector it scored, and the reference
pays for its own fitted literals.

The RULE (which literals count) and the server's side of the assembly are pinned against the real
server in ``tests/server/test_hardcoded_constants.py``, which also cross-checks the judge's ``k``
against the server's on a live edit.
"""

from __future__ import annotations

from gemot.runner.judge import _candidate_k, _fitted_parameters


# --------------------------------------------------------------------------- the assembly
def test_the_judge_derives_the_same_k_from_its_own_reads():
    """Estimated rows the model declares, plus the constants the summary lists. A shipped row the
    model never declared (`in_model: false`, appended by get_parameters for display) is not a free
    parameter of the model being scored and must not be charged — counting it made the judge's k
    exceed the server's by one per undeclared shipped row."""
    params = [
        {"parameter_id": "a", "estimate": 1},
        {"parameter_id": "b", "estimate": 1},
        {"parameter_id": "sigma", "estimate": 0},
        {"parameter_id": "shipped_unused", "estimate": 1, "in_model": False},
    ]
    final = {"neg_log_likelihood": -50.0, "hardcoded_constants": ["0.000878", "1.333"]}
    assert _candidate_k(params, final) == (4, 2, ["0.000878", "1.333"])
    # a summary without the list (an older server) or an unreadable read charges none
    assert _candidate_k(params, {"neg_log_likelihood": -50.0}) == (2, 2, [])
    assert _candidate_k("not a list", None) == (0, 0, [])


# ------------------------------------------------------------------- the reference side
def test_the_reference_pays_for_its_fitted_literals():
    """`setup.py --refit` charges the published model's FITTED literals to the bar's k (fujita's
    68190 is the author's fitted pro_EGFR, inlined by the collection) and records them, so a
    reference whose k exceeds its estimated-row count is auditable. Nothing else on the result
    moves: the published fit is unchanged, only its count."""
    from gemot.benchmark import charge_hardcoded_parameters

    ref = {"nll": -66.58, "k": 19, "n": 144, "noise_sigma": {}}
    charged = charge_hardcoded_parameters(ref, {"68190": "pro_EGFR, the author's fitted pool"})
    assert charged == {**ref, "k": 20, "hardcoded_constants": ["68190"]}
    assert ref["k"] == 19  # pure: the input is not mutated
    assert charge_hardcoded_parameters(ref, None) is ref
    assert charge_hardcoded_parameters(ref, {}) is ref


def test_the_judge_records_the_parameter_vector_it_scored():
    """Every criterion prices `k` parameters and none of them recorded what those parameters came
    out AS, so reading a finished run's fit back meant re-fitting it. A fit writes its result into
    `nominalValue`, and the vector is exactly the rows `_candidate_k` charges: a fixed row is not
    estimated, and an `in_model: false` row is a shipped parameter this model never declared."""
    params = [
        {"parameterId": "a", "estimate": 1, "nominalValue": 0.75},
        {"parameterId": "b", "estimate": 1, "nominalValue": 12.0},
        {"parameterId": "sigma", "estimate": 0, "nominalValue": 0.1},
        {"parameterId": "shipped_unused", "estimate": 1, "in_model": False, "nominalValue": 3.0},
    ]
    assert _fitted_parameters(params) == {"a": 0.75, "b": 12.0}
    # The same rows k charges, so the vector's length is auditable against `k_estimated`.
    assert len(_fitted_parameters(params)) == _candidate_k(params, None)[1]
    # None, not {}, where the read is unusable: "nothing recorded" is not "nothing estimated".
    assert _fitted_parameters("not a list") is None and _fitted_parameters([]) == {}
    # A row whose value is missing or non-numeric is skipped rather than crashing the judge.
    assert (
        _fitted_parameters(
            [
                {"parameterId": "a", "estimate": 1, "nominalValue": None},
                {"parameterId": "b", "estimate": 1, "nominalValue": "nan"},
            ]
        )
        == {}
    )
