"""The fit-budget invariants that span BOTH packages.

`petab_mcp` must import without `gemot` (tests/test_import_lightness), so the constants
the candidate fit and the reference refit share cannot be one object — they are two copies that
must agree. Nothing but these tests stops them drifting.
"""

from __future__ import annotations

import os
from pathlib import Path


# -- the two-copy tolerance invariant ------------------------------------------------------
def test_the_candidate_and_reference_fits_use_ONE_amici_tolerance():
    """`petab_mcp.fitting.FIT_AMICI_ATOL` (every candidate fit) and
    `gemot.fit_budget.FIT_AMICI_ATOL` (the reference refit) must be the same number.

    They are deliberately separate constants — petab_mcp must import without gemot — so
    nothing but this test stops them drifting. If they diverge, the reference and the candidate
    are optimised against DIFFERENT objectives and delta-BIC stops being a comparison: on
    lucarelli the gap between AMICI's stock 1e-16 and this 1e-10 was worth 175 BIC, which would
    read as a modelling result rather than a solver setting.
    """
    from gemot import fit_budget
    from petab_mcp import fitting

    assert fitting.FIT_AMICI_ATOL == fit_budget.FIT_AMICI_ATOL, (
        f"candidate atol {fitting.FIT_AMICI_ATOL} != reference atol "
        f"{fit_budget.FIT_AMICI_ATOL} — the two sides of every delta-BIC are being integrated "
        "at different tolerances"
    )
    # ...and it must NOT be AMICI's stock value: 1e-16 makes the forward-sensitivity solve
    # unable to differentiate on stiff models, reporting solver failure as failed starts.
    assert fitting.FIT_AMICI_ATOL > 1e-16


def test_both_tolerance_constants_honour_the_same_env_override():
    """One env var (PETAB_MCP_FIT_AMICI_ATOL) must move BOTH, or an operator tuning the
    tolerance for a sweep silently moves only the candidate side."""
    import subprocess
    import sys

    probe = (
        "import sys; sys.path.insert(0, %r);"
        "from gemot import fit_budget; from petab_mcp import fitting;"
        "print(fitting.FIT_AMICI_ATOL, fit_budget.FIT_AMICI_ATOL)"
        % str(Path(__file__).resolve().parent.parent / "src")
    )
    env = {**os.environ, "PETAB_MCP_FIT_AMICI_ATOL": "1e-8"}
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    cand, ref = out.stdout.split()
    assert float(cand) == float(ref) == 1e-8, f"env override moved only one side: {out.stdout!r}"


# ---------------------------------------------------------------- needs_flattening: numeric overrides
def _mdf(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def _problem(mdf):
    class _P:
        measurement_df = mdf

    return _P()


def test_numeric_per_timepoint_sigmas_do_not_force_flattening():
    """AMICI writes a numeric sigma straight into the ExpData, so per-timepoint numbers are
    already exact. MEASURED on upstream crauste: the nominal llh is bit-identical flattened and
    unflattened (-190.96512533983716). Flattening them would be pure cost -- it multiplies the
    observable count, and the same predicate gates the observable merge."""
    from petab_mcp.fitting import needs_flattening

    mdf = _mdf(
        [
            {
                "observableId": "a",
                "simulationConditionId": "c",
                "time": 0.0,
                "noiseParameters": 0.01,
            },
            {
                "observableId": "a",
                "simulationConditionId": "c",
                "time": 1.0,
                "noiseParameters": 0.02,
            },
        ]
    )
    assert needs_flattening(_problem(mdf)) is False


def test_numeric_per_timepoint_observable_parameters_do_force_flattening():
    """These DO become mapped parameters, one value per condition, so a number that varies
    between timepoints of one condition has nowhere to live."""
    from petab_mcp.fitting import needs_flattening

    mdf = _mdf(
        [
            {
                "observableId": "a",
                "simulationConditionId": "c",
                "time": 0.0,
                "observableParameters": 1.0,
            },
            {
                "observableId": "a",
                "simulationConditionId": "c",
                "time": 1.0,
                "observableParameters": 2.0,
            },
        ]
    )
    assert needs_flattening(_problem(mdf)) is True


def test_overrides_that_vary_only_across_conditions_are_taken_natively():
    """AMICI's parameter mapping is per condition, so this is not a flattening case."""
    from petab_mcp.fitting import needs_flattening

    mdf = _mdf(
        [
            {
                "observableId": "a",
                "simulationConditionId": "c1",
                "time": 0.0,
                "observableParameters": "s1",
            },
            {
                "observableId": "a",
                "simulationConditionId": "c2",
                "time": 0.0,
                "observableParameters": "s2",
            },
        ]
    )
    assert needs_flattening(_problem(mdf)) is False
