"""Score the trained field against a held-out set, in the set's own mode.

The counterpart of :mod:`petab_mcp.validation_data` for the neural field. **Fixed-θ** evaluates
the trained weights and scalars on conditions the training set did not cover: the one thing the
server cannot do for a field, since it fits plain PEtab problems and a staged one carries the
SciML extension.

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
nothing downstream subtracts it from a fixed-θ bar. A transfer set's readout scale factors are
refit too: another system's scalings are not the training ones, which is the set's whole point.

Report-only, like the server's: nothing here gates anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class Prepared:
    """The held-out problem staged and imported ONCE, ready to score any field of its shape.

    A budget curve scores every checkpoint against the same held-out set; staging and importing
    it per checkpoint repeats the costliest part of scoring for each point. Fixed-theta only, as
    :func:`score` is: a transfer set is refused at :func:`prepare`.
    """

    mode: str
    n: int
    staged: object
    imported: object = field(repr=False)
    ids: list[str] = field(repr=False)
    config: T.TrainConfig = field(repr=False)


def prepare(
    result: T.TrainResult, held_out, directory: Path | str, *, training: P.Problem
) -> Prepared | dict:
    """``held_out`` staged against ``result``'s design, ready to score any field of that shape.
    Returns the failure payload instead if the tables cannot be reconciled."""
    directory = Path(directory)
    n = int(len(held_out.measurement_df))
    if held_out.parameter_df is not None:
        return {
            "status": STATUS_NO_TRANSFER,
            "mode": MODE_TRANSFER,
            "n": n,
            "reason": "a transfer set re-estimates the model's parameters on the held-out "
            "system; a field's parameters are its weights, and there are too few "
            "held-out points to re-estimate them. Freezing them and refitting the "
            "starts and sigmas answers a different question from the one the "
            "reference bar answers, so no number is reported",
        }
    try:
        problem = _held_out_problem(training, held_out, directory / "v1")
        staged = S.stage(
            problem,
            directory / "staged",
            design=result.staged,
            shared_initial_conditions=result.config.shared_initial_conditions,
        )
    except (P.ProblemError, ValueError) as why:
        return {"status": STATUS_TABLE_MISMATCH, "mode": MODE_FIXED, "n": n, "reason": str(why)}
    imported = T._import(staged)
    return Prepared(
        mode=MODE_FIXED,
        n=n,
        staged=staged,
        imported=imported,
        ids=list(imported.parameter_ids),
        config=result.config,
    )


def score_prepared(prepared: Prepared, networks, parameters: dict[str, float]) -> dict:
    """One field -- ``networks`` and the scalars it fitted -- against a :func:`prepare`d set."""
    import equinox as eqx  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from amici.sim.jax import run_simulations  # noqa: PLC0415

    n, ids = prepared.n, prepared.ids
    imported = eqx.tree_at(lambda p: p.model.nns, prepared.imported, networks)
    known = {pid: parameters[pid] for pid in ids if pid in parameters}
    unknown = [pid for pid in ids if pid not in parameters]

    if not unknown:
        fitted = (
            imported.update_parameters(jnp.asarray([known[pid] for pid in ids]))
            if ids
            else imported
        )
        llh, _ = run_simulations(fitted, **T._solver(prepared.config))
        nll = -float(llh)
        if not math.isfinite(nll):
            return {
                "status": STATUS_NON_FINITE,
                "mode": MODE_FIXED,
                "n": n,
                "reason": "the field could not be integrated on the held-out conditions",
            }
        return {"status": STATUS_OK, "mode": MODE_FIXED, "nll": nll, "n": n}

    return {
        "status": STATUS_NO_TRANSFER,
        "mode": MODE_FIXED,
        "n": n,
        "reason": f"fixed-theta needs every scalar the held-out problem estimates to be one "
        f"the training run fitted; {unknown} are not. Shared starts or a t=0 datum "
        f"would pin them. Refitting them instead would be a transfer, which a field "
        f"cannot answer",
    }


def score(result: T.TrainResult, held_out, directory: Path | str, *, training: P.Problem) -> dict:
    """``held_out`` (a :class:`petab_mcp.validation_data.HeldOut`) scored against the trained
    field, as ``{status, mode, nll, n, reason}``. Fixed-θ only; a transfer set is refused."""
    prepared = prepare(result, held_out, directory, training=training)
    if isinstance(prepared, dict):
        return prepared
    return score_prepared(prepared, result.problem.model.nns, result.parameters)


def _held_out_problem(training: P.Problem, held_out, directory: Path) -> P.Problem:
    """The held-out set as a PEtab v1 problem beside the training tables: the measurements are the
    set's, the conditions gain the set's rows, everything else is the training problem's. Fixed-θ
    only; :func:`prepare` refuses a transfer set before this.

    The set is read as the server reads it (:func:`petab_mcp.validation_data.read_held_out`), so
    its condition table is v2's long one whatever its files are. The route stages v1 problems
    (:mod:`.problem`), so those rows go back into v1's columns, one per target. A set keyed by
    experiment, a native v2 one, has no v1 form and is refused.
    """
    import petab.v1 as petab  # noqa: PLC0415
    from petab.v1 import C  # noqa: PLC0415
    from petab.v1.yaml import load_yaml  # noqa: PLC0415

    measurements = held_out.measurement_df
    if held_out.experiment_df is not None or C.SIMULATION_CONDITION_ID not in measurements:
        raise P.ProblemError(
            "the held-out set names experiments, not conditions (PEtab v2); the neural route "
            "stages PEtab v1 problems, so the set has no form it can score"
        )
    source = petab.Problem.from_yaml(str(training.source))
    directory.mkdir(parents=True, exist_ok=True)
    petab.write_measurement_df(measurements, str(directory / "measurements.tsv"))
    petab.write_observable_df(source.observable_df, str(directory / "observables.tsv"))
    conditions = source.condition_df
    if held_out.condition_df is not None:
        if conditions is None:
            raise P.ProblemError(
                "the held-out set brings a condition table but the training problem has none, so "
                "there are no inputs to line the two up by"
            )
        mine = conditions.drop(columns=[C.CONDITION_NAME], errors="ignore")
        named = {
            str(c)
            for column in (C.SIMULATION_CONDITION_ID, C.PREEQUILIBRATION_CONDITION_ID)
            if column in measurements
            for c in measurements[column].dropna()
            if str(c)
        }
        theirs = _v1_conditions(held_out.condition_df, named - set(map(str, mine.index)))
        new = sorted(set(map(str, theirs.columns)) - set(map(str, mine.columns)))
        if new:
            raise P.ProblemError(
                f"the held-out conditions set {new}, which no training condition sets; a "
                "condition target is an input, so the two tables describe different fields"
            )
        conditions = pd.concat([mine, theirs.reindex(columns=mine.columns)])
    petab.write_condition_df(conditions, str(directory / "conditions.tsv"))
    petab.write_parameter_df(source.parameter_df, str(directory / "parameters.tsv"))
    document = load_yaml(str(training.source))
    model = training.source.parent / document["problems"][0]["sbml_files"][0]
    shutil.copy(model, directory / "model.xml")
    protocols = training.source.parent / "inputs.tsv"
    if protocols.is_file():
        shutil.copy(protocols, directory / "inputs.tsv")
    (directory / "problem.yaml").write_text(
        "format_version: 1\nparameter_file: parameters.tsv\nproblems:\n- sbml_files: [model.xml]\n"
        "  condition_files: [conditions.tsv]\n  measurement_files: [measurements.tsv]\n"
        "  observable_files: [observables.tsv]\n"
    )
    return P.load(directory, name=f"{training.name}_held_out", dosed=training.dosed)


def _v1_conditions(condition_df, empty: "set[str]") -> pd.DataFrame:
    """A v2 condition table in v1's shape: indexed by condition, one column per target. v2 has no
    row for a condition that changes nothing; each id in ``empty`` comes back as an empty row, as
    v1 has it."""
    from petab.v2.C import CONDITION_ID, TARGET_ID, TARGET_VALUE  # noqa: PLC0415

    wide = condition_df.pivot(index=CONDITION_ID, columns=TARGET_ID, values=TARGET_VALUE)
    wide.columns.name = None
    return wide.reindex(wide.index.union(sorted(empty)))


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
