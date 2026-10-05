"""The harness's fit settings are the server's, applied the server's way.

``gemot.fit_budget`` reads the per-start cap, the worker count and the AMICI tolerance from
``petab_mcp.fitting`` (through ``gemot.server_api``), but applies them through helpers of its own,
which take per-call overrides the server's do not. These pin the helpers to the server's: if the
reference refit and a candidate fit stopped agreeing on tolerance, cap or parallelism, delta-BIC
would stop being a comparison, and a healthy fit could outrun the timeout sized for it.
"""

from __future__ import annotations


def test_the_reference_refit_integrates_at_the_servers_tolerance():
    """The reference refit sets its AMICI tolerance through ``apply_amici_solver_defaults``; it must
    land on the server's ``FIT_AMICI_ATOL``, which every candidate fit uses. On lucarelli the gap
    between AMICI's stock 1e-16 and the server's 1e-10 was worth 175 BIC, which would read as a
    modelling result rather than a solver setting."""
    from types import SimpleNamespace

    from gemot import fit_budget
    from petab_mcp import fitting

    seen = []
    problem = SimpleNamespace(
        objective=SimpleNamespace(amici_solver=SimpleNamespace(set_absolute_tolerance=seen.append))
    )
    fit_budget.apply_amici_solver_defaults(problem)
    assert seen == [fitting.FIT_AMICI_ATOL]
    # ...and it is NOT AMICI's stock value: 1e-16 makes the forward-sensitivity solve unable to
    # differentiate on stiff models, reporting solver failure as failed starts.
    assert fitting.FIT_AMICI_ATOL > 1e-16


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


# --------------------------------------------------------------- the server's rules, gemot's helpers
def test_fit_budget_matches_the_server():
    """The harness's helpers must derive what the server's do from the same settings, or the
    timeout sized for a fit stops matching the fit the server runs, and a healthy one gets killed.
    """
    from gemot import fit_budget

    from petab_mcp import fitting

    assert fit_budget.fides_maxtime_options() == fitting._fides_maxtime_options()

    # The worker count each side derives for a given multistart, including the edge cases
    # (0/negative starts, fewer starts than workers) where the two expressions differ in shape.
    for n_starts in (0, 1, 2, 3, 7, 50, 1000, -5):
        harness = fit_budget.fit_workers(n_starts)
        server = max(1, min(fitting.FIT_N_WORKERS, int(n_starts)))
        assert harness == server, f"n_starts={n_starts}: harness={harness} server={server}"
