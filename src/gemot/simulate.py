"""Synthetic-data generation for the examples.

Each example ships a deliberately BROKEN model, but its measurements must come from the
TRUE mechanism. These helpers simulate a truth model with the *same* engine the server
uses (AMICI's PEtab v1 path), so the data an agent is asked to fit is exactly what the
correctly-specified model produces — the only gap left for the fit to close is the noise
floor. A ``generate_data.py`` script per example calls :func:`simulate_true` on the truth
model, then :func:`add_noise_and_write` to emit ``measurements.tsv``. The reference bar they
write beside the data comes from ``petab_agent_bench.reference.write_reference``, next to the
benchmark problems' own reference fit.

Kept dependency-light on purpose: it needs only ``amici`` + ``petab`` (not the petab-mcp
server package), so data can be regenerated from a bare simulation environment.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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
                rows.append(
                    {
                        OBSERVABLE_ID: obs_id,
                        SIMULATION_CONDITION_ID: cond_id,
                        TIME: float(t),
                        MEASUREMENT: 0.0,
                    }
                )
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
        model_ant, parameter_df, condition_df, observable_df, measurement_df, build_dir, model_name
    )

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
            "ignore",
            message="The following problem parameters were not used",
            category=RuntimeWarning,
        )
        result = simulate_petab(
            problem,
            amici_model=amici_model,
            problem_parameters=params,
            scaled_parameters=False,
        )
    simdf = rdatas_to_simulation_df(result[RDATAS], amici_model, problem.measurement_df)
    keys = [OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME]
    # ``simdf`` carries exactly one simulation row per measurement row (see amici's
    # ``rdatas_to_measurement_df``), but replicate measurements share the same
    # (observableId, simulationConditionId, time) key. Merging on ``keys`` alone would
    # cross-join each size-r replicate group into r*r rows, inflating the returned
    # per-point frame. Tag each row with its within-group replicate index so the merge
    # stays one-to-one with the measurements: ``len(merged) == len(measurement_df)``. The
    # simulated value is identical across a replicate group, so pairing by index is exact
    # regardless of the (possibly reordered) row order in ``simdf``. Mirrors the fix in
    # ``petab_mcp.engine.AmiciEngine._simulation_records``.
    rep = "_petab_mcp_replicate"
    while rep in measurement_df.columns or rep in simdf.columns:  # avoid clobbering columns
        rep += "_"
    measurement_df = measurement_df.assign(
        **{rep: measurement_df.groupby(keys, sort=False).cumcount()}
    )
    simdf = simdf.assign(**{rep: simdf.groupby(keys, sort=False).cumcount()})
    merged = measurement_df.merge(simdf[keys + [rep, SIMULATION]], on=keys + [rep], how="left")
    return merged


def _import_and_problem(
    model_ant, parameter_df, condition_df, observable_df, measurement_df, build_dir, model_name
):
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
        condition_df=condition_df,
        observable_df=observable_df,
        measurement_df=measurement_df.copy(),
        parameter_df=parameter_df,
    )
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    amici_model = import_petab_problem(
        problem, output_dir=str(build_dir), model_name=model_name, compile_=True
    )
    return problem, amici_model


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
    # Every simulated observable must have a sigma. Series.map(dict) SILENTLY yields NaN for a
    # missing key, which would propagate through the noise draw (and clip/round) and write NaN
    # measurements to the TSV unnoticed — also poisoning the reference fit built from this data.
    # Fail loudly; the usual cause is adding an observable but not its NOISE entry.
    missing = sorted(set(map(str, out[OBSERVABLE_ID])) - set(map(str, noise)))
    if missing:
        raise ValueError(
            f"add_noise_and_write: no noise sigma for observable(s) {missing}; "
            f"add them to the noise map (have: {sorted(map(str, noise))})."
        )
    values = out[SIMULATION].to_numpy(dtype=float)
    sigmas = out[OBSERVABLE_ID].map(noise).to_numpy(dtype=float)
    if not np.all(np.isfinite(sigmas)):
        raise ValueError("add_noise_and_write: noise map has a non-finite sigma value.")
    noised = values + rng.normal(0.0, 1.0, size=values.shape) * sigmas
    if nonneg:
        noised = np.clip(noised, 0.0, None)
    out[MEASUREMENT] = np.round(noised, decimals)
    out = out[[OBSERVABLE_ID, SIMULATION_CONDITION_ID, TIME, MEASUREMENT]]
    out.to_csv(path, sep="\t", index=False)
    return out
