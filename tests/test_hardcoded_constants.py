"""Hardcoded constants are priced into ``k`` (``petab_mcp.literals``, issue #180).

Every complexity-penalised score charged ``k = n_estimated``, so a value written straight into a
rate law cost nothing however it was obtained: fujita run ``53685907_4`` had 21 reactions, 10
species and ``k = 4`` — its rate laws read ``0.000878``, ``2.69e-5``, ``1.333``… — and scored best
of 29 runs. The RULE (which literals count) and the server's half of the ASSEMBLY (the fit payload
and the stored version's bic read the same addend) are pinned in PEtab-MCP. What is pinned here is
gemot's half: the judge derives the same ``k`` from its own reads. That the reference bar pays for its
fitted literals too is pinned in PEtabAgentBench, with the reference fit.
"""

from __future__ import annotations

import asyncio

from conftest import DATA_DIR

from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server
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
