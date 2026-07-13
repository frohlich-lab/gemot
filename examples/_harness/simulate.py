"""Synthetic-data generation for the examples.

Each example ships a deliberately BROKEN model, but its measurements must come from the
TRUE mechanism. These helpers simulate a truth model with the *same* engine the server
uses (AMICI's PEtab v1 path), so the data an agent is asked to fit is exactly what the
correctly-specified model produces — the only gap left for the fit to close is the noise
floor. A ``generate_data.py`` script per example calls :func:`simulate_true` on the truth
model, then :func:`add_noise_and_write` to emit ``measurements.tsv``.

Kept dependency-light on purpose: it needs only ``amici`` + ``petab`` (not the petab-mcp
server package), so data can be regenerated from a bare simulation environment.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from petab.v1.C import (
    MEASUREMENT,
    OBSERVABLE_ID,
    SIMULATION,
    SIMULATION_CONDITION_ID,
    TIME,
)


def _ensure_swig_on_path() -> None:
    """AMICI compiles generated C++ via CMake+SWIG; make sure ``swig`` is found."""
    if shutil.which("swig"):
        return
    try:
        import swig  # the pip `swig` package bundles the binary under data/bin

        cand = Path(swig.__file__).resolve().parent / "data" / "bin"
        if (cand / "swig").exists():
            os.environ["PATH"] = str(cand) + os.pathsep + os.environ.get("PATH", "")
    except Exception:  # best-effort; AMICI errors clearly if still missing
        pass


def measurement_skeleton(
    timepoints: dict[str, list[float]], observables: list[str]
) -> pd.DataFrame:
    """A PEtab measurement table for ``observables`` × ``timepoints`` (dummy values).

    ``timepoints`` maps a ``simulationConditionId`` to its list of sample times; every
    observable is measured at every time of every condition. The ``measurement`` column is
    filled with zeros — it is overwritten by :func:`add_noise_and_write` — but must be
    present for AMICI to build the simulation dataframe.
    """
    rows = []
    for cond_id, times in timepoints.items():
        for obs_id in observables:
            for t in times:
                rows.append({OBSERVABLE_ID: obs_id, SIMULATION_CONDITION_ID: cond_id,
                             TIME: float(t), MEASUREMENT: 0.0})
    return pd.DataFrame(rows)


def simulate_true(
    model_ant: str,
    parameter_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    observable_df: pd.DataFrame,
    measurement_df: pd.DataFrame,
    *,
    build_dir: str | Path,
    model_name: str = "truth",
) -> pd.DataFrame:
    """Simulate the truth model with AMICI and return a per-point simulation DataFrame.

    Returns the measurement skeleton merged with a ``simulation`` column (one noise-free
    value per observable/condition/time), ready for :func:`add_noise_and_write`.
    """
    import warnings

    from amici.sim.sundials.petab.v1 import RDATAS, rdatas_to_simulation_df, simulate_petab
    from petab.v1.C import NOMINAL_VALUE, PARAMETER_ID

    if parameter_df.index.name != PARAMETER_ID and PARAMETER_ID in parameter_df.columns:
        parameter_df = parameter_df.set_index(PARAMETER_ID)
    problem, amici_model = _import_and_problem(
        model_ant, parameter_df, condition_df, observable_df, measurement_df,
        build_dir, model_name)

    params: dict[str, float] = {}
    for pid in parameter_df.index:
        try:
            val = float(parameter_df.loc[pid, NOMINAL_VALUE])
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            params[str(pid)] = val

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The following problem parameters were not used",
            category=RuntimeWarning,
        )
        result = simulate_petab(
            problem, amici_model=amici_model, problem_parameters=params,
            scaled_parameters=False,
        )
    simdf = rdatas_to_simulation_df(result[RDATAS], amici_model, problem.measurement_df)
    keys = [OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME]
    merged = measurement_df.merge(simdf[keys + [SIMULATION]], on=keys, how="left")
    return merged


def _import_and_problem(model_ant, parameter_df, condition_df, observable_df,
                        measurement_df, build_dir, model_name):
    """Build a petab Problem + a compiled AMICI model (shared by simulate/fit)."""
    from amici.importers.petab.v1 import import_petab_problem
    from petab.v1 import Problem
    from petab.v1.C import OBSERVABLE_ID as OID, PARAMETER_ID
    from petab.v1.models.sbml_model import SbmlModel

    _ensure_swig_on_path()
    if parameter_df.index.name != PARAMETER_ID and PARAMETER_ID in parameter_df.columns:
        parameter_df = parameter_df.set_index(PARAMETER_ID)
    if observable_df.index.name != OID and OID in observable_df.columns:
        observable_df = observable_df.set_index(OID)
    if condition_df.index.name != "conditionId" and "conditionId" in condition_df.columns:
        condition_df = condition_df.set_index("conditionId")
    problem = Problem(
        model=SbmlModel.from_antimony(model_ant),
        condition_df=condition_df, observable_df=observable_df,
        measurement_df=measurement_df.copy(), parameter_df=parameter_df,
    )
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    amici_model = import_petab_problem(
        problem, output_dir=str(build_dir), model_name=model_name, compile_=True
    )
    return problem, amici_model


def fit_broken(
    model_ant: str,
    parameter_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    observable_df: pd.DataFrame,
    measurement_df: pd.DataFrame,
    *,
    build_dir: str | Path,
    model_name: str = "broken",
    n_starts: int = 24,
    seed: int = 0,
) -> tuple[dict[str, float], float]:
    """Best-fit the (broken) model to the measurements; return (fitted values, best nll).

    A lightweight multistart of scipy L-BFGS-B over the estimated parameters (on their PEtab
    scale), scored by AMICI's PEtab log-likelihood — the same objective the server's
    pyPESTO fit uses. The returned values become the broken model's ``nominalValue`` so it
    ships ALREADY fitted: the baseline the agent sees is the best its (wrong) structure can
    do, which is the whole point — re-tuning parameters then cannot help, only fixing the
    STRUCTURE can. ``parameter_df`` must carry ``estimate``, ``parameterScale``,
    ``lowerBound``, ``upperBound``, ``nominalValue`` columns.
    """
    from amici.sim.sundials.petab.v1 import LLH, simulate_petab
    from petab.v1 import scale, unscale
    from petab.v1.C import (
        ESTIMATE, LOWER_BOUND, NOMINAL_VALUE, PARAMETER_ID, PARAMETER_SCALE, UPPER_BOUND,
    )
    from scipy.optimize import minimize

    pdf = parameter_df
    if pdf.index.name != PARAMETER_ID and PARAMETER_ID in pdf.columns:
        pdf = pdf.set_index(PARAMETER_ID)
    problem, amici_model = _import_and_problem(
        model_ant, pdf, condition_df, observable_df, measurement_df, build_dir, model_name)

    free = [pid for pid in pdf.index if int(float(pdf.loc[pid, ESTIMATE])) == 1]
    scales = {pid: pdf.loc[pid, PARAMETER_SCALE] for pid in free}
    fixed = {pid: float(pdf.loc[pid, NOMINAL_VALUE]) for pid in pdf.index if pid not in free}
    lb = np.array([scale(float(pdf.loc[pid, LOWER_BOUND]), scales[pid]) for pid in free])
    ub = np.array([scale(float(pdf.loc[pid, UPPER_BOUND]), scales[pid]) for pid in free])

    import warnings

    def negllh(x_scaled: np.ndarray) -> float:
        params = dict(fixed)
        for pid, xs in zip(free, x_scaled):
            params[pid] = unscale(float(xs), scales[pid])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                res = simulate_petab(problem, amici_model=amici_model,
                                     problem_parameters=params, scaled_parameters=False)
            except Exception:
                return 1e12
        val = -float(res[LLH])
        return val if np.isfinite(val) else 1e12

    rng = np.random.default_rng(seed)
    x0_nominal = np.array([scale(float(pdf.loc[pid, NOMINAL_VALUE]), scales[pid]) for pid in free])
    best_x, best_f = None, np.inf
    for i in range(n_starts):
        x0 = x0_nominal if i == 0 else lb + rng.random(len(free)) * (ub - lb)
        r = minimize(negllh, x0, method="L-BFGS-B", bounds=list(zip(lb, ub)))
        if r.fun < best_f:
            best_f, best_x = r.fun, r.x
    fitted = {pid: unscale(float(x), scales[pid]) for pid, x in zip(free, best_x)}
    return fitted, best_f


def write_reference(
    model_ant: str,
    parameter_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    observable_df: pd.DataFrame,
    measurement_df: pd.DataFrame,
    *,
    build_dir: str | Path,
    model_name: str,
    path: str | Path,
    n_starts: int = 12,
    seed: int = 0,
) -> dict:
    """Fit the TRUE structure to the data and write a BIC reference to ``path``.

    The judge accepts a repaired model iff it can't be rejected against the true model by
    BIC, so each example ships ``problem/reference.json`` = ``{nll, k, n}`` for the true
    (data-generating) structure fitted with its natural free parameters: ``nll`` the best
    neg-log-likelihood, ``k`` the number of estimated parameters, ``n`` the number of data
    points. ``parameter_df`` should mark exactly the true model's free parameters as
    ``estimate = 1`` (seeded at their true values so the fit lands at the true optimum).
    """
    import json

    fitted, nll = fit_broken(
        model_ant, parameter_df, condition_df, observable_df, measurement_df,
        build_dir=build_dir, model_name=model_name, n_starts=n_starts, seed=seed,
    )
    ref = {"nll": float(nll), "k": int(len(fitted)), "n": int(len(measurement_df))}
    Path(path).write_text(json.dumps(ref, indent=2) + "\n")
    return ref


def add_noise_and_write(
    sim_df: pd.DataFrame,
    noise: dict[str, float],
    path: str | Path,
    *,
    seed: int,
    nonneg: bool = True,
    decimals: int = 4,
) -> pd.DataFrame:
    """Add Gaussian noise to the simulated values and write a PEtab measurements.tsv.

    ``noise`` maps ``observableId -> sigma`` (the standard deviation of the additive
    Gaussian error). ``seed`` makes the data reproducible. With ``nonneg`` the noised
    values are clipped at 0 (concentrations/populations can't be negative). Returns the
    written measurement DataFrame.
    """
    rng = np.random.default_rng(seed)
    out = sim_df.copy()
    values = out[SIMULATION].to_numpy(dtype=float)
    sigmas = out[OBSERVABLE_ID].map(noise).to_numpy(dtype=float)
    noised = values + rng.normal(0.0, 1.0, size=values.shape) * sigmas
    if nonneg:
        noised = np.clip(noised, 0.0, None)
    out[MEASUREMENT] = np.round(noised, decimals)
    out = out[[OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME, MEASUREMENT]]
    out.to_csv(path, sep="\t", index=False)
    return out
