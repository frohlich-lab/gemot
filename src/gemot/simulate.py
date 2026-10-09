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


def fit_reference(
    model_ant: str,
    parameter_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    observable_df: pd.DataFrame,
    measurement_df: pd.DataFrame,
    *,
    build_dir: str | Path,
    model_name: str = "reference",
    n_starts: int = 24,
    n_workers: int | None = None,
    seed: int = 0,
    seed_at_nominal: bool = False,
) -> tuple[dict[str, float], float]:
    """Best-fit the TRUE structure with pyPESTO + **Fides** — the SAME optimizer the server
    runs on candidate models — and return (fitted values, best nll).

    The reference bar must be fit at least as well as any candidate, or ΔBIC measures the
    optimizer, not the structure. This project fits candidates with Fides everywhere; scipy
    L-BFGS-B is materially weaker on stiff/degenerate ODE likelihoods (it under-converged every
    reference by 2.5–5.7 nll, so the true structure "beat" its own reference by ΔBIC −5 to −11),
    so it is not used at all.

    ``seed_at_nominal`` (opt-in): pyPESTO's ``create_problem`` does NOT actually seed a start at
    ``nominalValue`` (its ``x_guesses`` is empty), so on a MULTIMODAL likelihood the random
    multistart can miss the basin the candidate settles in. When the caller has set the table's
    nominals to the intended (global) optimum, this includes that point in the reported best nll
    — so start-0 effectively lands in the true basin and ΔBIC is not at the mercy of which basin a
    random seed found. Off by default, so the well-behaved examples are unaffected.
    """
    import logging as _logging

    import numpy as np
    import pypesto.optimize
    import pypesto.petab
    from petab.v1 import Problem, scale, unscale
    from petab.v1.C import ESTIMATE, NOMINAL_VALUE
    from petab.v1.C import OBSERVABLE_ID as OID
    from petab.v1.C import PARAMETER_ID, PARAMETER_SCALE
    from petab.v1.models.sbml_model import SbmlModel

    _ensure_swig_on_path()
    np.random.seed(seed)  # best-effort determinism for start-point sampling

    pdf = parameter_df
    if pdf.index.name != PARAMETER_ID and PARAMETER_ID in pdf.columns:
        pdf = pdf.set_index(PARAMETER_ID)
    # Every criterion is likelihood-based, and the server scores the CANDIDATE on the pure AMICI
    # log-likelihood (engine.evaluate_objective -> -llh), NOT the MAP objective. So the reference
    # must be a pure-likelihood fit too: drop any objectivePrior columns here so pyPESTO's fval is
    # the neg-log-likelihood, not likelihood+prior. Without this the prior's normalization constant
    # (~½·log 2πσ² per prior) inflates only the reference nll and spuriously handicaps it — schwen
    # (the one example that ships priors) read ΔBIC ≈ −8 instead of ≈ 0. The priors still ship
    # VERBATIM in the problem (faithful) and still regularize the candidate's own fit; they just
    # don't enter the reference bar.
    pdf = pdf.drop(
        columns=[c for c in ("objectivePriorType", "objectivePriorParameters") if c in pdf.columns]
    )
    odf = observable_df
    if odf.index.name != OID and OID in odf.columns:
        odf = odf.set_index(OID)
    cdf = condition_df
    if cdf.index.name != "conditionId" and "conditionId" in cdf.columns:
        cdf = cdf.set_index("conditionId")

    problem = Problem(
        model=SbmlModel.from_antimony(model_ant),
        condition_df=cdf,
        observable_df=odf,
        measurement_df=measurement_df.copy(),
        parameter_df=pdf,
    )
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    importer = pypesto.petab.PetabImporter(
        problem, output_folder=str(build_dir), model_name=model_name
    )
    pp = importer.create_problem()
    # Cap each start's wall time exactly as the server's candidate fit does, so the reference and
    # candidate optimizers agree and ΔBIC parity holds, and parallelise the multistart the same way
    # (a one-time offline fit — MultiProcessEngine just makes it faster; see fit_budget.py).
    from .fit_budget import apply_amici_solver_defaults, fides_maxtime_options, fit_engine

    # Same reason: the integration tolerance is part of what the two optimizers have to agree on.
    apply_amici_solver_defaults(pp)
    optimizer = pypesto.optimize.FidesOptimizer(
        verbose=_logging.ERROR, options=fides_maxtime_options()
    )
    result = pypesto.optimize.minimize(
        pp,
        optimizer=optimizer,
        n_starts=n_starts,
        engine=fit_engine(n_starts, n_workers),
        progress_bar=False,
    )
    best = result.optimize_result.list[0]
    nll = float(best["fval"])
    # ``best["x"]`` / ``pp.x_names`` span ALL parameters (free + fixed, with fixed ones filled
    # back in), so key ``fitted`` to the ESTIMATED (free) params only — k = len(fitted) must be
    # the number of estimated parameters, not the full table size (else fixed sigmas/beta inflate k).
    free_names = [pid for pid in pdf.index if int(float(pdf.loc[pid, ESTIMATE])) == 1]
    xmap = dict(zip(pp.x_names, [float(z) for z in best["x"]]))
    fitted = {
        pid: float(unscale(xmap[pid], pdf.loc[pid, PARAMETER_SCALE]))
        for pid in free_names
        if pid in xmap
    }

    if seed_at_nominal:
        # Evaluate the objective at the (scaled) nominal point and adopt it if it beats the random
        # multistart. ``create_problem`` leaves ``x_guesses`` empty, so this is how the caller's
        # intended optimum actually enters the reference. pp.objective takes FREE-space coords, so
        # reduce the full nominal vector with pp.get_reduced_vector (the API that matches the
        # objective's expected ordering — see identifiability._compute_fim) rather than raw slicing.
        x_full = np.array(
            [
                scale(float(pdf.loc[name, NOMINAL_VALUE]), pdf.loc[name, PARAMETER_SCALE])
                for name in pp.x_names
            ]
        )
        fval_nom = float(pp.objective(pp.get_reduced_vector(x_full)))
        if np.isfinite(fval_nom) and fval_nom < nll:
            nll = fval_nom
            fitted = {pid: float(pdf.loc[pid, NOMINAL_VALUE]) for pid in free_names}

    return fitted, nll


def _effective_reference_sigmas(problem, raw: dict) -> dict:
    """``{sigma_id: effective sd}`` for the reference fit — the noiseFormula evaluated at those
    values, which is what a candidate's ``effective_sigma`` must be judged against.

    The RAW parameter is not comparable across models when the formula is composite or
    proportional: fiedler's sd is ``per-gel scale * sigma``, so a candidate predicting a different
    amplitude can trade scale against sigma and land on an IDENTICAL sd while its raw sigma differs
    several-fold — judging raw values fails a model whose noise level equals the truth's.

    Caveat worth knowing before relying on this for a composite formula: the value is taken in the
    AMPLITUDE-NORMALISED coordinate (see ``reconcile.sigma_effective_value``), because with one
    sigma shared across gels of differing gain there is no single sd in measurement units to report.
    That fixes the summarisability problem, not the scale-degeneracy one — a candidate that splits
    the same fit differently between gain and sigma still lands on a different number. The judge's
    tolerance is a generous ratio band, so a moderate split difference is absorbed; a large one
    is not. For fiedler the gain divides the formula exactly, so this equals the raw sigma.
    """
    from petab_mcp import reconcile as _reconcile

    out: dict = {}
    for pid, value in (raw or {}).items():
        try:
            effective = _reconcile.sigma_effective_value(
                problem, pid, float(value), fitted_values=raw
            )
        except Exception:  # noqa: BLE001 - advisory extra
            effective = None
        if effective is not None:
            out[pid] = float(effective)
    return out


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
    seed_at_nominal: bool = False,
) -> dict:
    """Fit the TRUE structure to the data and write the reference bar to ``path``.

    The judge accepts a repaired model iff it can't be rejected against the true model under the
    run's criterion, so each example ships ``problem/reference.json`` = ``{nll, k, n}`` for the true
    (data-generating) structure fitted with its natural free parameters: ``nll`` the best
    neg-log-likelihood, ``k`` the number of estimated parameters, ``n`` the number of data
    points. ``parameter_df`` should mark exactly the true model's free parameters as
    ``estimate = 1`` (seeded at their true values so the fit lands at the true optimum).
    """
    import json

    fitted, nll = fit_reference(
        model_ant,
        parameter_df,
        condition_df,
        observable_df,
        measurement_df,
        build_dir=build_dir,
        model_name=model_name,
        n_starts=n_starts,
        seed=seed,
        seed_at_nominal=seed_at_nominal,
    )
    ref = {"nll": float(nll), "k": int(len(fitted)), "n": int(len(measurement_df))}
    # Record the reference fit's own noise sigmas, symmetrically with write_reference_from_problem.
    # DEFENSIVE ONLY, and deliberately not load-bearing: every example that uses this writer today
    # (the synthetic enzyme_kinetics / predator_prey / repressilator) ships all of its sigmas FIXED
    # (estimate=0), so `fitted` contains none of them and both maps come out empty. A fixed sigma
    # cannot be inflated or collapsed by an agent, so there is nothing for the reference gate to
    # judge there — the gate's CI coverage comes from the unit tests over _reference_sigma_flag,
    # not from these examples. This exists so a future synthetic example that ESTIMATES a sigma is
    # recorded like any other, rather than silently falling back to the data-std band.
    try:
        import petab.v1 as _petab

        _problem = _petab.Problem(
            model=None,
            condition_df=condition_df,
            observable_df=observable_df,
            measurement_df=measurement_df,
            parameter_df=parameter_df,
        )
        from petab_mcp import reconcile as _reconcile

        sigma_ids = _reconcile.plausibility_sigma_ids(_problem)
        ref["noise_sigma"] = {
            pid: float(value) for pid, value in fitted.items() if pid in sigma_ids
        }
        ref["noise_sigma_effective"] = _effective_reference_sigmas(_problem, ref["noise_sigma"])
    except Exception as exc:  # noqa: BLE001 - advisory extra; never lose the nll/k/n bar
        print(f"[reference] could not record reference noise sigmas ({type(exc).__name__}: {exc})")
    Path(path).write_text(json.dumps(ref, indent=2) + "\n")
    return ref


def write_reference_from_problem(
    problem,
    *,
    build_dir: str | Path,
    model_name: str,
    path: str | Path,
    n_starts: int = 50,
    n_workers: int | None = None,
    per_start_timeout: float | None = None,
    seed: int = 0,
    seed_at_nominal: bool = True,
    held_out=None,
) -> dict:
    """Fit a PRISTINE collection PEtab problem EXACTLY as it ships and write ``{nll, k, n}``.

    The reference bar is the published benchmark's OWN best fit: its shipped SBML model + the
    published parameter table (the paper's true free/fixed split and bounds; ``nominalValue`` = the
    published optimum) + the published data — NOT a hand-authored ``TRUE_ANT`` and NOT the server's
    ``reconcile``. This anchors the reference to the genuine ground truth, independent of the IC /
    parametrisation conventions the CANDIDATE is scored under (so ΔBIC measures real closeness to
    the truth, not reproduction of a reconcile-mirrored reference).

    Fit with the same pyPESTO + Fides multistart the server runs on candidates. ``seed_at_nominal``
    seeds start-0 at the published optimum (adopted only if it beats the random multistart), so the
    reference reliably lands in the true basin. Every criterion is likelihood-based and the server
    scores the candidate on the pure AMICI ``-llh``, so any ``objectivePrior*`` columns are dropped
    here.

    ``per_start_timeout`` overrides the global Fides ``MAXTIME`` cap for this fit — the reference is
    the one fit that must be run to convergence rather than to a shared budget, and what "long
    enough" means is a property of the individual model (see
    :func:`~gemot.fit_budget.fides_maxtime_options`).

    ``held_out`` (a :class:`petab_mcp.validation_data.HeldOut`) additionally scores this reference
    against HELD-OUT measurements, on the condition rows the set brings if any, and records the
    result under ``validation`` — the predictive bar the judge compares a candidate
    against (see :mod:`petab_mcp.validation_data`). It is evaluated at the vector that produced the
    recorded ``nll``, whichever of the two it was, so the training and held-out numbers describe ONE
    fitted model. Only the SCORE is recorded, not the vector — re-deriving the held-out bar means
    re-running this refit, exactly as re-deriving the training bar does (see
    :func:`_reference_validation`). A TRANSFER set is not scored here: it is refit on its own,
    after this fit (``gemot.benchmark._transfer_reference``).
    """
    if held_out is not None and held_out.parameter_df is not None:
        raise ValueError(
            "write_reference_from_problem: a transfer set is REFIT, not scored at the "
            "reference's fitted point (gemot.benchmark._transfer_reference)"
        )
    import json
    import logging as _logging

    import numpy as np
    import pypesto.optimize
    import pypesto.petab
    from petab.v1 import scale
    from petab.v1.C import ESTIMATE, NOMINAL_VALUE, PARAMETER_ID, PARAMETER_SCALE

    from .fit_budget import apply_amici_solver_defaults, fides_maxtime_options, fit_engine

    _ensure_swig_on_path()
    np.random.seed(seed)

    pdf = problem.parameter_df
    if pdf.index.name != PARAMETER_ID and PARAMETER_ID in pdf.columns:
        pdf = pdf.set_index(PARAMETER_ID)
    pdf = pdf.drop(
        columns=[c for c in ("objectivePriorType", "objectivePriorParameters") if c in pdf.columns]
    )
    problem.parameter_df = pdf

    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    importer = pypesto.petab.PetabImporter(
        problem, output_folder=str(build_dir), model_name=model_name
    )
    pp = importer.create_problem()
    apply_amici_solver_defaults(pp)
    optimizer = pypesto.optimize.FidesOptimizer(
        verbose=_logging.ERROR, options=fides_maxtime_options(per_start_timeout)
    )
    result = pypesto.optimize.minimize(
        pp,
        optimizer=optimizer,
        n_starts=n_starts,
        engine=fit_engine(n_starts, n_workers),
        progress_bar=False,
    )
    nll = float(result.optimize_result.list[0]["fval"])

    # The vector the recorded nll belongs to, so the reference noise sigmas below are read off the
    # SAME fit (not the multistart best when the published optimum won, or vice versa).
    x_full_used = None
    if seed_at_nominal:
        x_full = np.array(
            [
                scale(float(pdf.loc[name, NOMINAL_VALUE]), pdf.loc[name, PARAMETER_SCALE])
                for name in pp.x_names
            ]
        )
        fval_nom = float(pp.objective(pp.get_reduced_vector(x_full)))
        if np.isfinite(fval_nom) and fval_nom < nll:
            nll = fval_nom
            x_full_used = x_full

    # k = the number of ESTIMATED parameters in the PUBLISHED table (fixed known-constants and any
    # fixed observation params are NOT counted), n = number of measurements.
    k = sum(int(float(pdf.loc[pid, ESTIMATE])) == 1 for pid in pdf.index)
    ref = {"nll": float(nll), "k": int(k), "n": int(len(problem.measurement_df))}
    # The reference fit's EFFECTIVE parameter count, at the point that produced the recorded nll
    # (petab_mcp.identifiability.effective_parameter_count) — the truth's own k vs k_eff beside the
    # candidate's, and the bar a TIC-ranked run is graded against (petab_mcp.criteria).
    # Advisory like the sigmas below: None when the objective cannot be evaluated
    # there, and the reference bar is written regardless.
    from petab_mcp.identifiability import effective_parameter_count

    # Through `_recorded_vector` like every other derived statistic, rather than indexing the
    # optimizer's `x` directly: that is where the full-dimensional invariant is enforced, so a
    # pyPESTO that ever returned a reduced or reordered vector fails fast here too instead of
    # computing k_eff at a silently wrong point.
    x_best = pp.get_reduced_vector(_recorded_vector(result, x_full_used, pp))
    ref["k_eff"] = effective_parameter_count(pp, x_best, k=int(k))
    # The reference fit's own NOISE SIGMAS — the only defensible "true sigma" for an ESTIMATED
    # sigma, whose nominalValue is merely a start point (8 of 12 benchmark examples estimate their
    # sigma, and e.g. perelson's nominalValue is a linear placeholder for a log10-space sd). The
    # judge compares a candidate's fitted sigma against these instead of against a fraction of the
    # marginal data std, which conflates variance the model EXPLAINS with variance it can't (#118).
    # Best-effort: a failure here must never cost us the nll/k/n bar.
    try:
        ref["noise_sigma"] = _reference_noise_sigmas(problem, pp, result, pdf, x_full_used)
        ref["noise_sigma_effective"] = _effective_reference_sigmas(problem, ref["noise_sigma"])
    except Exception as exc:  # noqa: BLE001 - advisory extra; the reference bar must be written
        print(f"[reference] could not record reference noise sigmas ({type(exc).__name__}: {exc})")
    if held_out is not None:
        # Guarded, for the same reason the sigmas above are: the reference bar is what this function
        # exists to produce, and it must not be lost to an advisory extra. Letting a raise through
        # here would discard a multistart that had already CONVERGED — hours of it on the large
        # examples — and write nothing at all, because `path` is written below.
        #
        # But not silently, either: a missing validation bar makes the judge's held-out comparison
        # unavailable on every future run of this example, and the only place an omitted key would
        # show is an empty CSV column. So the failure is RECORDED as a status rather than dropped —
        # `runner.scoring._validation_fields` reads any non-`ok` reference status as "no bar" and
        # withholds the difference instead of computing one against a number that isn't there.
        try:
            ref["validation"] = _reference_validation(
                problem,
                pp,
                result,
                x_full_used,
                held_out,
                build_dir=build_dir,
                model_name=model_name,
            )
        except Exception as exc:  # noqa: BLE001 - advisory extra; the reference bar must be written
            # `runner.scoring._validation_fields` carries this status into result.json as
            # `reference_status`, and `metrics` into the `val_reference_status` column — so the
            # failure is visible per-run, not only to whoever opens reference.json.
            ref["validation"] = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            print(
                f"[reference] could not score the held-out validation set "
                f"({type(exc).__name__}: {exc}); the nll/k/n bar is still written, but no "
                f"held-out bar — re-run --refit once the cause is fixed"
            )
        else:
            print(
                f"[reference] held-out validation: nll={ref['validation'].get('nll')} "
                f"n={ref['validation'].get('n')} status={ref['validation'].get('status')}"
            )
    Path(path).write_text(json.dumps(ref, indent=2) + "\n")
    return ref


def _recorded_vector(result, x_full_used, pp):
    """The FULL scaled parameter vector the recorded ``nll`` actually belongs to.

    ``write_reference_from_problem`` reports ``min(multistart best, published nominal point)``, so
    the committed ``nll`` belongs to one of two different parameter vectors and everything derived
    from that fit has to be read off the same one. ``x_full_used`` is non-``None`` exactly when the
    nominal point won; otherwise the best start's ``x`` is it.

    ONE definition, because every consumer of it makes the same mistake differently: pairing the
    recorded training nll with the OTHER vector's sigmas (:func:`_reference_noise_sigmas`) or with
    the other vector's held-out score (:func:`_reference_validation`) reports one model's fit beside
    another model's properties, which is precisely the mismatch that makes those numbers
    uninterpretable. If the ``min()`` above ever gains a third case, it has to be taught here and
    nowhere else.

    ``pp`` is taken only to CHECK the promise both callers rely on — that the vector is
    full-dimensional in ``pp.x_names`` order. They zip it against those names and index it by free
    position, so a pypesto that started returning a reduced or reordered ``x`` would surface as
    wrong sigmas and a wrong held-out score rather than as an error. Asserting the length is what
    makes the docstring enforceable instead of merely stated.
    """
    import numpy as np

    x = (
        np.asarray(x_full_used, dtype=float)
        if x_full_used is not None
        else np.asarray(result.optimize_result.list[0]["x"], dtype=float)
    )
    if x.shape != (len(pp.x_names),):
        raise ValueError(
            f"the recorded reference vector has shape {x.shape} for {len(pp.x_names)} parameters; "
            "it must be the FULL vector (fixed parameters included), in pp.x_names order"
        )
    return x


def _reference_validation(
    problem, pp, result, x_full_used, held_out, *, build_dir, model_name
) -> dict:
    """Score the reference fit against HELD-OUT measurements, at the vector its ``nll`` came from.

    Which vector that is matters and is easy to get wrong; :func:`_recorded_vector` owns that
    choice and says why.

    The VECTOR itself is deliberately not recorded. Writing it into ``reference.json`` would put the
    reference fit's full optimum into a committed file to serve a re-score path that does not exist
    — unread state in an answer key, free to go stale against the next refit. Re-deriving the
    held-out bar means re-running ``python -m petab_agent_bench --refit``, which is also the only
    way the training bar beside it can be re-derived, so the two cannot drift apart.
    """
    from petab_mcp import validation_data

    x = _recorded_vector(result, x_full_used, pp)
    return validation_data.score_problem(
        problem,
        held_out.measurement_df,
        build_dir=build_dir,
        model_name=model_name,
        x_full=[float(v) for v in x],
        x_names=[str(n) for n in pp.x_names],
        condition_df=held_out.condition_df,
    )


def _reference_noise_sigmas(problem, pp, result, pdf, x_full_used=None) -> dict:
    """``{sigma_parameter_id: fitted linear value}`` for the reference fit — the noise level the
    TRUE model needs to explain this data.

    Values come from whichever vector produced the recorded ``nll`` — see
    :func:`_recorded_vector`, which owns that choice. Returned on the LINEAR scale
    (un-scaled from each parameter's estimation scale), matching what the server reports for a
    candidate. Only parameters the observation model uses as a noise sigma are included.
    """
    from petab.v1.C import NOMINAL_VALUE
    from petab.v1.parameters import unscale

    from petab_mcp import reconcile as _reconcile

    sigma_ids = _reconcile.plausibility_sigma_ids(problem)
    if not sigma_ids:
        return {}
    x = _recorded_vector(result, x_full_used, pp)
    scales = list(pp.x_scales)
    out: dict[str, float] = {}
    for name, value, scale_name in zip(pp.x_names, x, scales):
        if name in sigma_ids:
            out[name] = float(unscale(float(value), scale_name))
    # Fixed sigmas never enter the optimizer vector; take their published value.
    for pid in sigma_ids - set(out):
        if pid in pdf.index:
            try:
                out[pid] = float(pdf.loc[pid, NOMINAL_VALUE])
            except (TypeError, ValueError):
                continue
    return out


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
