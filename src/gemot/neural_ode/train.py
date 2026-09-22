"""Fit the staged neural ODE through AMICI's JAX backend.

AMICI imports the staged problem as a ``JAXProblem``: the ODE with the network in its right-hand
side, PEtab's likelihood, its conditions, preequilibration and post-equilibration measurements,
all differentiable through diffrax. What is written here is only the loop around that: Adam over
the network weights and the log of every estimated scalar, restarts that redraw the weights,
patience, and the bookkeeping the route reads.

Three decisions are load-bearing. **Scalars are fitted on a log scale**: PEtab v2 has no
``parameterScale``, so the sigmas, scale factors and start states the staged problem estimates
would otherwise be searched on a linear scale where one Adam step can cross zero; every one of
them is positive by construction and spans decades. **A non-finite objective ends the restart,
not the run**: diffrax reports a failed solve as a non-finite likelihood and later steps cannot
recover from it, so the restart is recorded with its reason and the next one starts from fresh
weights. **Curriculum multiple shooting is a problem rewrite, not a loss** (arXiv:2608.05777,
five windows in the paper): ``petab_sciml.training`` exports stage problems in which every window
is an experiment with estimated start states and the continuity penalty is a set of measurements
of value zero, so AMICI's likelihood of a stage IS that stage's objective. This module runs the
stages in order, carries the weights and the scalars it recognises across them, and spends a
third of the step budget on the early stages and two thirds on the final, single-shooting one,
the allocation the paper found load-bearing. The rewrite refuses preequilibration, so under it
the windows are ignored and the result says so.

The reported ``nll`` is PEtab's likelihood of the final stage alone, the number AMICI's own
scorers reproduce.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # before the first array: float32 changes every number

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import pandas as pd  # noqa: E402

from . import stage as S  # noqa: E402
from .problem import Problem  # noqa: E402

#: A scalar's floor when its log is taken; PEtab lets an estimated start's bound sit at zero.
LOG_FLOOR = 1e-8
#: The files a stage problem needs beside the tables petab writes for it.
_EXTENSION_FILES = (f"{S.NETWORK_ID}.py", f"{S.NETWORK_ID}.yaml", f"{S.NETWORK_ID}_ps.hdf5", "hybridization.tsv")


@dataclass(frozen=True)
class TrainConfig:
    """One training configuration. Every field lands in the result record verbatim."""

    n_augmented: int = 0
    width: int = 32
    depth: int = 2
    #: A state counting time, read by the network: the field may then depend on the clock.
    clock: bool = False
    #: One start state for every experiment instead of one per experiment. A parameterisation
    #: choice BIC decides per problem; experiments that really start apart will reject it.
    shared_initial_conditions: bool = False
    #: Feed the network the pairwise products of its inputs as well (see `stage.product_features`).
    #: Widens the FIRST layer only; the distiller still regresses on the base variables.
    multiplicative: bool = False
    #: Windows of curriculum multiple shooting; 1 is single shooting. Stages merge one window at
    #: a time until the last stage is the staged problem itself.
    n_windows: int = 1
    #: Weight of the continuity penalty between adjacent windows.
    continuity_weight: float = 1.0
    steps: int = 1500
    learning_rate: float = 3e-3
    n_restarts: int = 4
    seed: int = 0
    patience: int = 300
    min_improvement: float = 1e-6
    rtol: float = 1e-6
    atol: float = 1e-8
    max_solver_steps: int = 8192

    def key(self) -> str:
        windows = f"_cms{self.n_windows}" if self.n_windows > 1 else ""
        clock = "_clock" if self.clock else ""
        shared = "_shared" if self.shared_initial_conditions else ""
        mult = "_mult" if self.multiplicative else ""
        return f"aug{self.n_augmented}_w{self.width}d{self.depth}{windows}{clock}{shared}{mult}_s{self.seed}"


@dataclass
class TrainResult:
    config: TrainConfig
    staged: S.Staged
    #: PEtab's negative log-likelihood at the best restart's optimum; ``inf`` if none trained.
    nll: float
    #: The scalars the staged problem estimates -- starts, sigmas, scale factors: the ODE's own
    #: ``k``, never the weights (docs/RATIONALE.md).
    n_ode_parameters: int
    n_network_weights: int
    solver_failed: bool
    steps_run: int
    restart_failures: tuple[str, ...]
    #: The best restart's objective at the end of every curriculum stage; the last is ``nll``,
    #: the earlier ones include the continuity penalties.
    stage_objectives: tuple[float, ...]
    notes: tuple[str, ...]
    wall_seconds: float
    #: The estimated scalars at the optimum, by PEtab id.
    parameters: dict[str, float]
    #: The trained ``JAXProblem``; simulate with it, do not serialise it.
    problem: object = field(repr=False)


def train(problem: Problem, directory: Path | str, config: TrainConfig) -> TrainResult:
    """Stage ``problem`` in ``directory`` and fit it under ``config``.

    Every restart starts from freshly drawn weights and the nominal scalars and runs every
    curriculum stage; the best finite final objective is kept. A run in which no restart produced
    one returns ``solver_failed`` with ``nll`` infinite rather than raising: the caller decides
    whether that ends a ladder or the whole route.
    """
    started = time.perf_counter()
    staged = stage_only(problem, directory, config)
    stages, notes = _curriculum(problem, staged, config)
    imported = [_import(stage) for stage in stages]
    steppers = [_stepper(jp, config) for jp in imported]
    budgets = _budget(config.steps, len(stages))
    network_class = type(imported[-1].model.nns[S.NETWORK_ID])

    best_value, best_trained, best_objectives, steps_run, failures = np.inf, None, (), 0, []
    for restart in range(config.n_restarts):
        networks = {S.NETWORK_ID: network_class(jr.PRNGKey(config.seed + restart))}
        scalars: dict[str, float] = {}
        objectives: list[float] = []
        for jp, stepper, budget in zip(imported, steppers, budgets):
            value, (networks, scalars), taken, failure = _fit(jp, stepper, networks, scalars, budget, config)
            steps_run += taken
            if failure is not None:
                failures.append(f"restart {restart}: {failure}")
                break
            objectives.append(value)
        else:
            if objectives[-1] < best_value:
                best_value, best_trained, best_objectives = objectives[-1], (networks, scalars), tuple(objectives)

    final = imported[-1]
    if best_trained is None:
        fitted, values = final, {}
    else:
        fitted = _with(final, best_trained[0], best_trained[1])
        values = dict(best_trained[1])
    return TrainResult(config=config, staged=staged, nll=float(best_value),
                       n_ode_parameters=len(final.parameter_ids), n_network_weights=staged.n_weights,
                       solver_failed=best_trained is None, steps_run=steps_run,
                       restart_failures=tuple(failures), stage_objectives=best_objectives, notes=notes,
                       wall_seconds=time.perf_counter() - started, parameters=values, problem=fitted)


def stage_only(problem: Problem, directory: Path | str, config: TrainConfig) -> S.Staged:
    """Stage ``problem`` under ``config`` without training; what :func:`train` does first."""
    return S.stage(problem, directory, n_augmented=config.n_augmented, width=config.width,
                   depth=config.depth, clock=config.clock,
                   shared_initial_conditions=config.shared_initial_conditions,
                   seed=config.seed, multiplicative=config.multiplicative)


def trajectories(result: TrainResult, n_times: int = 200) -> dict[str, dict[str, np.ndarray]]:
    """Dense state trajectories of the trained field, per experiment: ``{"times", "states"}``
    with ``states`` of shape ``(n_times, n_states)`` in the model's state order.

    The staged problem is copied with a measurement grid over ``[0, horizon]`` and re-imported, so
    dosing, preequilibration and protocol events are AMICI's own rather than re-implemented here.
    """
    from amici.sim.jax import ReturnValue, run_simulations  # noqa: PLC0415

    staged = result.staged
    dense = staged.directory / "dense"
    shutil.copytree(staged.directory, dense, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("dense", "curriculum", "jax_*", "v1", "__pycache__"))
    measurements = pd.read_csv(dense / "measurements.tsv", sep="\t")
    grid = np.linspace(0.0, staged.horizon, int(n_times))
    rows = []
    for experiment, group in measurements.groupby("experimentId", sort=False):
        # The observables take the grid in turn rather than one of them taking all of it: an
        # observable absent from the table takes its observableParameters out of use, and lint
        # reports those as extraneous parameters (fujita's scaling_pAkt_tot and scaling_pS6_tot).
        # Cycling keeps every one of them in use AND keeps one row per grid time, so the simulated
        # trajectory still has exactly `n_times` points. Which observable carries a time does not
        # matter: `ReturnValue.x` asks for states.
        templates = [rows_for.iloc[0].to_dict() for _, rows_for in group.groupby("observableId", sort=False)]
        for i, t in enumerate(grid):
            # 1.0, not 0.0: the grid's measured values are never read either, but `_import`
            # validates, and lint refuses a non-positive measurement under a `log`/`log10`
            # observable transformation (perelson's viral load is one).
            rows.append({**templates[i % len(templates)], "time": float(t), "measurement": 1.0})
        # A grid shorter than the observable count would still drop some; they join at the end.
        for template in templates[len(grid):]:
            rows.append({**template, "time": float(grid[-1]), "measurement": 1.0})
    pd.DataFrame(rows, columns=measurements.columns).to_csv(dense / "measurements.tsv", sep="\t", index=False)
    imported = _import(replace(staged, directory=dense, yaml=dense / "problem.yaml"))
    imported = eqx.tree_at(lambda p: p.model.nns, imported, result.problem.model.nns)
    imported = imported.update_parameters(result.problem.parameters)
    x, extra = run_simulations(imported, ret=ReturnValue.x, **_solver(result.config))
    times = np.asarray(extra["ts"])
    experiments = [e.id for e in imported._petab_problem.experiments]
    return {eid: {"times": times[i], "states": np.asarray(x)[i]} for i, eid in enumerate(experiments)}


def state_names(result: TrainResult) -> tuple[str, ...]:
    """The model's state order, which :func:`trajectories` reports in."""
    return tuple(result.problem.model.state_ids)


def save_bundle(result: TrainResult, directory: Path | str) -> Path:
    """The trained field as a complete PEtab SciML problem in ``directory``: the staged files with
    the trained weights in the array file and the fitted scalars as nominal values, so AMICI, or
    anyone reading the standard, re-imports exactly the model that was scored."""
    from petab_sciml import ArrayData, ArrayDataStandard  # noqa: PLC0415

    directory = Path(directory)
    shutil.copytree(result.staged.directory, directory, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("dense", "curriculum", "jax_*", "v1", "__pycache__"))
    network = result.problem.model.nns[S.NETWORK_ID]
    arrays = {name: {"weight": np.asarray(layer.weight), "bias": np.asarray(layer.bias)}
              for name, layer in network.layers.items()}
    ArrayDataStandard.save_data(ArrayData(metadata={"pytorch_format": True}, parameters={S.NETWORK_ID: arrays}),
                                str(directory / f"{S.NETWORK_ID}_ps.hdf5"))
    parameters = pd.read_csv(directory / "parameters.tsv", sep="\t")
    parameters["nominalValue"] = parameters["nominalValue"].astype(object)
    for i, pid in enumerate(parameters["parameterId"]):
        if pid in result.parameters:
            parameters.loc[i, "nominalValue"] = result.parameters[pid]
    parameters.to_csv(directory / "parameters.tsv", sep="\t", index=False)
    return directory


def _curriculum(problem: Problem, staged: S.Staged, config: TrainConfig) -> tuple[list[S.Staged], tuple[str, ...]]:
    """The stage problems to fit in order, ending with ``staged`` itself, and any note on why the
    curriculum is shorter than asked."""
    if config.n_windows <= 1:
        return [staged], ()
    if problem.uses_preequilibration:
        return [staged], (f"n_windows={config.n_windows} ignored: the multiple-shooting rewrite refuses "
                          "preequilibration",)
    from petab_sciml.training import CurriculumMultipleShootingProblem, UniformPartition  # noqa: PLC0415

    # Every window must have length; with few measured time points the partition cannot give
    # that many, and fewer windows beat a degenerate one.
    measurements = pd.read_csv(staged.directory / "measurements.tsv", sep="\t")
    t_min = float(measurements["time"].min())
    if t_min > 0.0:
        # The rewrite partitions the MEASURED span, so its first window would begin at the first
        # datum and the start states applied at t=0 would fall outside every window.
        return [staged], (f"n_windows={config.n_windows} ignored: the data start at t={t_min:g}, after the "
                          "experiments do, and the multiple-shooting rewrite would drop the starts",)
    n_windows, notes = int(config.n_windows), ()
    while n_windows > 1:
        ends = (UniformPartition(n_windows).get_time_points(measurements)
                if len(measurements["time"].unique()) >= n_windows else [])
        if ends and all(end > start for start, end in zip([t_min, *ends[:-1]], ends)):
            break
        n_windows -= 1
    if n_windows < config.n_windows:
        notes = (f"n_windows={config.n_windows} reduced to {n_windows}: the data have too few time points "
                 "for more windows of nonzero length",)
    if n_windows <= 1:
        return [staged], notes
    directory = staged.directory / "curriculum"
    shutil.rmtree(directory, ignore_errors=True)
    CurriculumMultipleShootingProblem(yaml=staged.yaml, partition=UniformPartition(n_windows),
                                      penalty=float(config.continuity_weight)).export(directory, validate=False)
    stages = sorted((p for p in directory.iterdir() if p.name.startswith("stage")), key=lambda p: int(p.name[5:]))
    for stage in stages[:-1]:
        for name in _EXTENSION_FILES:
            shutil.copy(staged.directory / name, stage / name)
        _seed_window_starts(stage, staged)
    return [*(replace(staged, directory=s, yaml=s / "problem.yaml") for s in stages[:-1]), staged], notes


def _seed_window_starts(stage: Path, staged: S.Staged) -> None:
    """A window's start parameters begin at the identity readout measured where the window starts,
    else at the start guess in the state's own units; the rewrite's one value for all would put a
    state of magnitude 300 at 1e-6."""
    parameters = pd.read_csv(stage / "parameters.tsv", sep="\t")
    parameters["nominalValue"] = parameters["nominalValue"].astype(object)  # the column also holds "array"
    experiments = pd.read_csv(stage / "experiments.tsv", sep="\t")
    measurements = pd.read_csv(staged.directory / "measurements.tsv", sep="\t")
    observables = pd.read_csv(staged.directory / "observables.tsv", sep="\t")
    identity = {str(f).strip(): oid for oid, f in zip(observables["observableId"], observables["observableFormula"])
                if str(f).strip() in staged.states}
    for i, pid in enumerate(parameters["parameterId"]):
        if not (str(pid).startswith("WINDOW") and "_PARAMETER_" in str(pid)):
            continue
        prefix, state = str(pid).rsplit("_PARAMETER_", 1)
        experiment = prefix.split("_EXPERIMENT_", 1)[1]
        t0 = float(experiments.loc[experiments["experimentId"] == prefix, "time"].astype(float).min())
        guess = S.START_GUESS_FRACTION * staged.scales.get(state, 1.0)
        if state in identity:
            rows = measurements[(measurements["experimentId"] == experiment) & (measurements["time"] == t0)
                                & (measurements["observableId"] == identity[state])]
            if len(rows):
                guess = float(rows["measurement"].astype(float).mean())
        parameters.loc[i, "nominalValue"] = max(guess, LOG_FLOOR)
    parameters.to_csv(stage / "parameters.tsv", sep="\t", index=False)


def _budget(steps: int, n_stages: int) -> list[int]:
    """A third of ``steps`` over the early stages, two thirds on the final one."""
    if n_stages == 1:
        return [int(steps)]
    early = max(1, round(steps / 3 / (n_stages - 1)))
    return [early] * (n_stages - 1) + [max(1, int(steps) - early * (n_stages - 1))]


def _stepper(imported, config: TrainConfig, *, freeze_network: bool = False):
    """The objective and one jitted Adam step for ``imported``, built once so every restart reuses
    the compilation. ``freeze_network`` fits the scalars alone, the weights held as they are."""
    from amici.sim.jax import run_simulations  # noqa: PLC0415

    solver = _solver(config)
    optimiser = optax.adam(config.learning_rate)

    def objective(trained, base):
        networks, log_scalars = trained
        if freeze_network:
            networks = jax.tree_util.tree_map(
                lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_inexact_array(leaf) else leaf, networks)
        llh, _ = run_simulations(_with(base, networks, log_scalars), **solver)
        return -llh

    @eqx.filter_jit
    def step(trained, state, base):
        value, gradient = eqx.filter_value_and_grad(objective)(trained, base)
        updates, state = optimiser.update(gradient, state, eqx.filter(trained, eqx.is_inexact_array))
        return value, eqx.apply_updates(trained, updates), state

    return objective, step, optimiser


def _fit(imported, stepper, networks: dict, scalars: dict[str, float], steps: int, config: TrainConfig):
    """Fit one stage from ``networks`` and the scalars it recognises by id (the rest start at
    their nominal values). Returns the best objective, ``(networks, scalars)`` at it, the steps
    taken and the failure, if any."""
    objective, step, optimiser = stepper
    ids = list(imported.parameter_ids)
    nominal = np.asarray(imported.parameters, dtype=float)
    start = jnp.log(jnp.maximum(jnp.asarray([scalars.get(pid, nominal[i]) for i, pid in enumerate(ids)]), LOG_FLOOR))
    trained = (networks, start)
    state = optimiser.init(eqx.filter(trained, eqx.is_inexact_array))
    best_value, best_trained, since_improvement, taken = np.inf, trained, 0, 0
    for i in range(steps):
        value, candidate, state = step(trained, state, imported)
        taken += 1
        value = float(value)
        if not np.isfinite(value):
            return best_value, _unpack(best_trained, ids), taken, f"non-finite objective at step {i}"
        if value < best_value - config.min_improvement:
            best_value, best_trained, since_improvement = value, trained, 0
        else:
            since_improvement += 1
        trained = candidate
        if since_improvement >= config.patience:
            break
    else:
        final = float(objective(trained, imported))
        if np.isfinite(final) and final < best_value:
            best_value, best_trained = final, trained
    return best_value, _unpack(best_trained, ids), taken, None


def _unpack(trained, ids: list[str]) -> tuple[dict, dict[str, float]]:
    networks, log_scalars = trained
    return networks, {pid: float(v) for pid, v in zip(ids, np.exp(np.asarray(log_scalars)))}


def _with(base, networks: dict, scalars):
    """``base`` carrying ``networks`` and the scalars, given as their logs or by id."""
    if isinstance(scalars, dict):
        values = jnp.asarray([scalars[pid] for pid in base.parameter_ids])
    else:
        values = jnp.exp(scalars)
    return eqx.tree_at(lambda p: p.model.nns, base, networks).update_parameters(values)


def _solver(config: TrainConfig) -> dict:
    import diffrax  # noqa: PLC0415

    return {"solver": diffrax.Kvaerno5(),
            "controller": diffrax.PIDController(rtol=config.rtol, atol=config.atol),
            "max_steps": int(config.max_solver_steps)}


def _import(staged: S.Staged):
    """The staged problem as AMICI's ``JAXProblem``. AMICI resolves the extension's files against
    the working directory, hence the change of directory for the import."""
    from amici.importers.petab import PetabImporter  # noqa: PLC0415
    from petab import v2  # noqa: PLC0415

    digest = hashlib.md5(str(staged.directory.resolve()).encode()).hexdigest()[:10]
    with _working_directory(staged.directory):
        petab_problem = v2.Problem.from_yaml(staged.yaml.name)
        importer = PetabImporter(petab_problem, module_name=f"node_{digest}", compile_=True, jax=True,
                                 validate=True, output_dir=str(staged.directory / f"jax_{digest}"))
        return importer.create_simulator(force_import=True)


@contextmanager
def _working_directory(path: Path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
