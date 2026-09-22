"""Score the trained field against a held-out set, in the set's own mode.

The counterpart of :mod:`petab_mcp.validation_data` for the neural field. **Fixed-θ** evaluates
the trained weights and scalars on conditions the training set did not cover: the one thing the
server cannot do for a field, since it fits PEtab v1 problems and a staged SciML problem is not
one.

**A transfer set is refused, not scored.** A transfer set asks whether a model structure can
describe a new system once its parameters are re-estimated, and a field's parameters ARE its
weights -- 90 to 1400 of them against 40 to 62 held-out points. Re-estimating them is
meaningless at that ratio; freezing them and refitting the starts and sigmas instead is not the
question the set asks, and the number it produces is not comparable with a reference's, which
re-estimates its own dynamics (crauste's bar refits all twelve rate constants per unit, perelson's
refits c and delta). The old code did exactly that and reported it as `transfer`, so the two sides
of every such comparison were doing different tasks and the field was doing the harder one.

The held-out measurements are staged as a PEtab SciML problem in the TRAINING design -- its
scales, horizon, latents, network and weights -- because the field is only defined in those
units; a set whose states or inputs differ from the training set's is ``table_mismatch``, exactly
as the server refuses a condition column it has not seen. Fixed-θ also needs every scalar the
held-out problem estimates to be one the training run fitted: shared starts, or starts the
held-out data pin. Where that fails, the scalars are refit and the payload says ``transfer``, so
nothing downstream subtracts it from a fixed-θ bar.

Report-only, like the server's: nothing here gates anything.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd

from . import problem as P
from . import stage as S
from . import train as T

STATUS_OK = "ok"
STATUS_TABLE_MISMATCH = "table_mismatch"
STATUS_NON_FINITE = "non_finite"
#: A set this field cannot be scored against at all -- see the module docstring.
STATUS_NO_TRANSFER = "transfer_not_answerable"
MODE_FIXED = "fixed"
MODE_TRANSFER = "transfer"


def score(result: T.TrainResult, held_out, directory: Path | str, *, training: P.Problem) -> dict:
    """``held_out`` (a :class:`petab_mcp.validation_data.HeldOut`) scored against the trained
    field, as ``{status, mode, nll, n, reason}``. Fixed-θ only; a transfer set is refused."""
    import equinox as eqx  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from amici.sim.jax import run_simulations  # noqa: PLC0415

    directory = Path(directory)
    n = int(len(held_out.measurement_df))
    if held_out.parameter_df is not None:
        return {"status": STATUS_NO_TRANSFER, "mode": MODE_TRANSFER, "n": n,
                "reason": "a transfer set re-estimates the model's parameters on the held-out "
                          "system; a field's parameters are its weights, and there are too few "
                          "held-out points to re-estimate them. Freezing them and refitting the "
                          "starts and sigmas answers a different question from the one the "
                          "reference bar answers, so no number is reported"}
    try:
        problem = _held_out_problem(training, held_out, directory / "v1")
        staged = S.stage(problem, directory / "staged", design=result.staged,
                         shared_initial_conditions=result.config.shared_initial_conditions)
    except (P.ProblemError, ValueError) as why:
        return {"status": STATUS_TABLE_MISMATCH, "mode": MODE_FIXED, "n": n, "reason": str(why)}
    imported = T._import(staged)
    imported = eqx.tree_at(lambda p: p.model.nns, imported, result.problem.model.nns)
    ids = list(imported.parameter_ids)
    known = {pid: result.parameters[pid] for pid in ids if pid in result.parameters}
    unknown = [pid for pid in ids if pid not in result.parameters]

    if not unknown:
        fitted = imported.update_parameters(jnp.asarray([known[pid] for pid in ids])) if ids else imported
        llh, _ = run_simulations(fitted, **T._solver(result.config))
        nll = -float(llh)
        if not math.isfinite(nll):
            return {"status": STATUS_NON_FINITE, "mode": MODE_FIXED, "n": n,
                    "reason": "the field could not be integrated on the held-out conditions"}
        return {"status": STATUS_OK, "mode": MODE_FIXED, "nll": nll, "n": n}

    return {"status": STATUS_NO_TRANSFER, "mode": MODE_FIXED, "n": n,
            "reason": f"fixed-theta needs every scalar the held-out problem estimates to be one "
                      f"the training run fitted; {unknown} are not. Shared starts or a t=0 datum "
                      f"would pin them. Refitting them instead would be a transfer, which a field "
                      f"cannot answer"}


def _held_out_problem(training: P.Problem, held_out, directory: Path) -> P.Problem:
    """The held-out set as a PEtab v1 problem beside the training tables it does not replace:
    the measurements are the set's; the conditions gain the set's rows (fixed-θ) or become the
    set's table (transfer); the parameters are the set's own when it brings them."""
    import petab.v1 as petab  # noqa: PLC0415
    from petab.v1 import C  # noqa: PLC0415
    from petab.v1.yaml import load_yaml  # noqa: PLC0415

    source = petab.Problem.from_yaml(str(training.source))
    directory.mkdir(parents=True, exist_ok=True)
    petab.write_measurement_df(held_out.measurement_df, str(directory / "measurements.tsv"))
    petab.write_observable_df(source.observable_df, str(directory / "observables.tsv"))
    conditions = source.condition_df
    theirs = held_out.condition_df
    if theirs is not None:
        theirs = theirs.drop(columns=[C.CONDITION_NAME], errors="ignore")
        mine = conditions.drop(columns=[C.CONDITION_NAME], errors="ignore")
        if set(map(str, theirs.columns)) != set(map(str, mine.columns)):
            raise P.ProblemError(
                f"the held-out condition table has columns {sorted(map(str, theirs.columns))} against "
                f"the training table's {sorted(map(str, mine.columns))}; a condition column is an input, "
                "so the two tables describe different fields")
        conditions = theirs[list(mine.columns)] if held_out.parameter_df is not None else pd.concat(
            [mine, theirs[list(mine.columns)]])
    petab.write_condition_df(conditions, str(directory / "conditions.tsv"))
    petab.write_parameter_df(held_out.parameter_df if held_out.parameter_df is not None else source.parameter_df,
                             str(directory / "parameters.tsv"))
    document = load_yaml(str(training.source))
    model = training.source.parent / document["problems"][0]["sbml_files"][0]
    shutil.copy(model, directory / "model.xml")
    protocols = training.source.parent / "inputs.tsv"
    if protocols.is_file():
        shutil.copy(protocols, directory / "inputs.tsv")
    (directory / "problem.yaml").write_text(
        "format_version: 1\nparameter_file: parameters.tsv\nproblems:\n- sbml_files: [model.xml]\n"
        "  condition_files: [conditions.tsv]\n  measurement_files: [measurements.tsv]\n"
        "  observable_files: [observables.tsv]\n")
    return P.load(directory, name=f"{training.name}_held_out", dosed=training.dosed)


def score_of(nll: float, k: int, n: int, *, criterion=None) -> float | None:
    """The field's own complexity-penalised score, from the server's criterion.

    :mod:`petab_mcp.criteria` owns scoring: six criteria, resolved from the explicit argument, else
    ``PETAB_MCP_CRITERION``, else BIC, with a ``k_eff`` slot for an effective parameter count. A
    local ``2 nll + k ln n`` would be a second copy that could only drift from it, and -- worse --
    could not be reparametrised with the rest, so a run ranked by AICc everywhere else would still
    report the field in BIC. Returns ``None`` rather than NaN where an input is missing, which is
    that module's contract and load-bearing downstream.

    Over the field's TRAINING nll and the scalars PEtab estimates, never the weights; the distilled
    models are scored downstream on what they really estimate.
    """
    from petab_mcp.criteria import resolve  # noqa: PLC0415

    return resolve(criterion).value(nll, k, n)
