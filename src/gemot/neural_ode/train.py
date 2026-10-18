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

from . import regularise as R  # noqa: E402
from . import stage as S  # noqa: E402
from .problem import Problem  # noqa: E402

#: A scalar's floor when its log is taken; PEtab lets an estimated start's bound sit at zero.
LOG_FLOOR = 1e-8
#: The files a stage problem needs beside the tables petab writes for it.
_EXTENSION_FILES = (
    f"{S.NETWORK_ID}.py",
    f"{S.NETWORK_ID}.yaml",
    f"{S.NETWORK_ID}_ps.hdf5",
    "hybridization.tsv",
)


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
    #: Stages of a HORIZON curriculum: fit an early slice of the time course, then extend, each
    #: stage starting from the last one's weights. 1 is no curriculum. Unlike multiple shooting
    #: this rewrites nothing -- it withholds late measurements -- so it needs neither data at t=0
    #: nor the absence of preequilibration, and applies where `n_windows` cannot.
    #:
    #: MUTUALLY EXCLUSIVE with `n_windows`: both build a curriculum out of the same step budget
    #: and asking for both is a configuration error, not a preference to resolve. `train` raises
    #: rather than picking one, because the version that silently let `n_windows` win would run a
    #: sweep arm labelled as a horizon curriculum that had never built one.
    horizon_stages: int = 1
    #: Windows of curriculum multiple shooting; 1 is single shooting. Stages merge one window at
    #: a time until the last stage is the staged problem itself.
    n_windows: int = 1
    #: Weight of the continuity penalty between adjacent windows.
    continuity_weight: float = 1.0
    steps: int = 1500
    #: ``"adam"``, ``"fides"``, or ``"adam+fides"``. Adam takes a fixed step per iteration however curved the
    #: objective is, which is why crauste needed 80,000 of them; fides is a trust-region method
    #: that builds curvature from the gradients JAX already returns, so an iteration costs about
    #: the same and buys far more. ``steps`` is its iteration cap too.
    optimiser: str = "adam"
    #: Sparsity penalty on the field's WEIGHTS: one of `regularise.NAMES`. What a distillation has
    #: to reproduce is which VARIABLES the field reads, and that is decided in the first layer.
    penalty: str = "none"
    #: Weight of that penalty in the objective, and the strength of the proximal step under `l21`.
    penalty_weight: float = 0.0
    #: Share of the step budget Adam gets under ``"adam+fides"``; fides polishes with the rest.
    adam_share: float = 0.5
    learning_rate: float = 3e-3
    n_restarts: int = 4
    seed: int = 0
    patience: int = 300
    min_improvement: float = 1e-6
    rtol: float = 1e-6
    atol: float = 1e-8
    max_solver_steps: int = 8192
    #: Step counts at which to keep the best-so-far parameters, so a run can be scored as if its
    #: budget had ended there. Empty keeps nothing. It records what a run of that length WOULD have
    #: returned -- the best seen so far, not the current iterate -- so the checkpoints of one long
    #: run stand in for a series of shorter ones, and a budget sweep costs one training, not N.
    #: The trajectory is untouched: `key` ignores this, so a checkpointed run and a plain one at
    #: the same seed are the same run. To read the curve as a BUDGET, set `patience` at or above
    #: `steps`; otherwise early stopping ends the run and the later checkpoints only repeat it.
    #:
    #: **A DIAGNOSTIC, NOT A SELECTION.** Scoring these against `examples/*/validation` measures
    #: whether held-out loss turns upward as training proceeds -- a property of the method, fine to
    #: report as such. Picking the budget by which checkpoint scored best is different: that set is
    #: the benchmark's TEST set, and choosing on it puts it inside the training loop and voids every
    #: held-out number downstream. It also flatters itself, being a minimum over a dozen points of a
    #: noisy trace. The protocol is the campaign's: freeze the winner on a training-only criterion,
    #: then open validation once. An honest early stop needs an inner split of the TRAINING data --
    #: and note `bic_with_weights` cannot stand in, because `k` is constant along a trajectory, so
    #: it is monotone in the training nll and blind to this effect by construction.
    checkpoints: tuple[int, ...] = ()

    def key(self) -> str:
        windows = f"_cms{self.n_windows}" if self.n_windows > 1 else ""
        clock = "_clock" if self.clock else ""
        shared = "_shared" if self.shared_initial_conditions else ""
        mult = "_mult" if self.multiplicative else ""
        horizon = f"_cl{self.horizon_stages}" if self.horizon_stages > 1 else ""
        sparse = "" if self.penalty == "none" else f"_{self.penalty}{self.penalty_weight:g}"
        solver = "" if self.optimiser == "adam" else f"_{self.optimiser}"
        return (
            f"aug{self.n_augmented}_w{self.width}d{self.depth}{windows}{horizon}{clock}"
            f"{shared}{mult}{sparse}{solver}_s{self.seed}"
        )


@dataclass(frozen=True)
class Checkpoint:
    """The best-so-far optimum of one restart at one step count: what a run of that budget would
    have returned. ``networks`` is live equinox state, so it is held, never serialised."""

    restart: int
    steps: int
    #: The objective at the best step SEEN so far, not at step ``steps``.
    nll: float
    parameters: dict[str, float]
    networks: dict = field(repr=False)


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
    #: One entry per (restart, checkpoint) requested by ``config.checkpoints``, in the order taken.
    #: Score one with :func:`at_checkpoint`; a restart that failed before a checkpoint has none.
    checkpoints: tuple[Checkpoint, ...] = ()


class ConfigError(ValueError):
    """A training configuration that cannot do what it says."""


def train(problem: Problem, directory: Path | str, config: TrainConfig) -> TrainResult:
    """Stage ``problem`` in ``directory`` and fit it under ``config``.

    Every restart starts from freshly drawn weights and the nominal scalars and runs every
    curriculum stage; the best finite final objective is kept. A run in which no restart produced
    one returns ``solver_failed`` with ``nll`` infinite rather than raising: the caller decides
    whether that ends a ladder or the whole route.
    """
    if config.n_windows > 1 and config.horizon_stages > 1:
        raise ValueError(
            f"n_windows={config.n_windows} and horizon_stages={config.horizon_stages} both build a "
            "curriculum from one step budget; ask for one. Multiple shooting rewrites the problem "
            "into windows and refuses preequilibration; the horizon curriculum withholds late "
            "measurements and does not"
        )
    if config.penalty != "none" and config.optimiser != "adam":
        # Sparsity is the PENALTY plus the proximal step, and the proximal step belongs between
        # optimiser steps. fides owns its own: it chooses a step from a trust region it sizes from
        # the curvature it has accumulated, and projecting the iterate behind it invalidates both
        # the BFGS update and the radius. So `_fit_fides` does not threshold, and a penalised fides
        # run would add the penalty, never reach an exact zero, and still call itself sparse.
        raise ConfigError(
            f"penalty={config.penalty!r} needs optimiser='adam', not {config.optimiser!r}: the "
            "proximal step that makes a group exactly zero runs between optimiser steps, which a "
            "trust-region method does not expose"
        )
    if config.checkpoints:
        if config.optimiser != "adam":
            raise ValueError(
                f"checkpoints need optimiser='adam', not {config.optimiser!r}: a fides or hybrid "
                "iteration is a trust-region solve of varying cost, so its count is not a budget "
                "axis and the curve would not mean what it appears to"
            )
        if min(config.checkpoints) < 1 or max(config.checkpoints) > config.steps:
            raise ValueError(f"checkpoints {config.checkpoints} fall outside 1..{config.steps}")
    started = time.perf_counter()
    staged = stage_only(problem, directory, config)
    stages, notes = _curriculum(problem, staged, config)
    imported = [_import(stage) for stage in stages]
    steppers = [_stepper(jp, config) for jp in imported]
    budgets = _budget(config.steps, len(stages))
    network_class = type(imported[-1].model.nns[S.NETWORK_ID])

    best_value, best_trained, best_objectives, steps_run, failures = np.inf, None, (), 0, []
    checkpoints: list[Checkpoint] = []
    for restart in range(config.n_restarts):
        networks = {S.NETWORK_ID: network_class(jr.PRNGKey(config.seed + restart))}
        scalars: dict[str, float] = {}
        objectives: list[float] = []
        for jp, stepper, budget in zip(imported, steppers, budgets):
            try:
                fit = {"adam": _fit, "fides": _fit_fides, "adam+fides": _fit_hybrid}[
                    config.optimiser
                ]
            except KeyError as failure:
                raise ValueError(f"unsupported optimiser {config.optimiser!r}") from failure
            extra = {}
            if config.checkpoints and fit is _fit:

                def keep(taken, value, trained, _restart=restart):
                    nets, pars = trained
                    checkpoints.append(
                        Checkpoint(
                            restart=_restart,
                            steps=taken,
                            nll=float(value),
                            parameters=dict(pars),
                            networks=nets,
                        )
                    )

                extra["on_checkpoint"] = keep
            value, (networks, scalars), taken, failure = fit(
                jp, stepper, networks, scalars, budget, config, **extra
            )
            steps_run += taken
            if failure is not None:
                failures.append(f"restart {restart}: {failure}")
                break
            objectives.append(value)
        else:
            if objectives[-1] < best_value:
                best_value, best_trained, best_objectives = (
                    objectives[-1],
                    (networks, scalars),
                    tuple(objectives),
                )

    # RELAXED LASSO, WARM STARTED. The penalty selects which variables the field reads and shrinks
    # every surviving weight while doing it, so the penalised fit understates what that support can
    # reach. The refit therefore begins AT the penalised optimum -- `best_trained` carries both the
    # weights and the scalars into it -- rather than from a fresh draw: the support is already the
    # one worth keeping, and only the shrinkage has to come out. Starting over would re-solve a
    # problem that is already solved, and could land in a different basin on a different support.
    # The zeroed columns are held at zero throughout, so the refit cannot reintroduce a dropped
    # variable; what it recovers is the fit that support was always capable of.
    if best_trained is not None and config.penalty != "none" and config.penalty_weight:
        support = _support_mask(best_trained[0])
        free = replace(config, penalty="none", penalty_weight=0.0)
        stepper = _stepper(imported[-1], free)
        value, state, taken, failure = _fit(
            imported[-1], stepper, *best_trained, config.steps, free, support=support
        )
        steps_run += taken
        if failure is None and np.isfinite(value) and value < best_value:
            best_value, best_trained = value, state
            best_objectives = (*best_objectives[:-1], value)
        notes = (*notes, f"refit on the selected support: {failure or f'nll {value:.4f}'}")

    final = imported[-1]
    if best_trained is None:
        fitted, values = final, {}
    else:
        fitted = _with(final, best_trained[0], best_trained[1])
        values = dict(best_trained[1])
    return TrainResult(
        config=config,
        staged=staged,
        nll=float(best_value),
        n_ode_parameters=len(final.parameter_ids),
        n_network_weights=staged.n_weights,
        solver_failed=best_trained is None,
        steps_run=steps_run,
        restart_failures=tuple(failures),
        stage_objectives=best_objectives,
        notes=notes,
        wall_seconds=time.perf_counter() - started,
        parameters=values,
        problem=fitted,
        checkpoints=tuple(checkpoints),
    )


def at_checkpoint(result: TrainResult, checkpoint: Checkpoint) -> TrainResult:
    """``result`` as it would have been had the budget ended at ``checkpoint``. Everything the
    held-out scorer reads -- the staged design, the config, the networks and the scalars -- comes
    from the checkpoint; the run-level counters do not, and are left as the whole run's."""
    return replace(
        result,
        nll=checkpoint.nll,
        parameters=dict(checkpoint.parameters),
        problem=_with(result.problem, checkpoint.networks, checkpoint.parameters),
        checkpoints=(),
    )


def stage_only(problem: Problem, directory: Path | str, config: TrainConfig) -> S.Staged:
    """Stage ``problem`` under ``config`` without training; what :func:`train` does first."""
    return S.stage(
        problem,
        directory,
        n_augmented=config.n_augmented,
        width=config.width,
        depth=config.depth,
        clock=config.clock,
        shared_initial_conditions=config.shared_initial_conditions,
        seed=config.seed,
        multiplicative=config.multiplicative,
    )


def trajectories(result: TrainResult, n_times: int = 200) -> dict[str, dict[str, np.ndarray]]:
    """Dense state trajectories of the trained field, per experiment: ``{"times", "states"}``
    with ``states`` of shape ``(n_times, n_states)`` in the model's state order.

    The staged problem is copied with a measurement grid over ``[0, horizon]`` and re-imported, so
    dosing, preequilibration and protocol events are AMICI's own rather than re-implemented here.
    """
    from amici.sim.jax import ReturnValue, run_simulations  # noqa: PLC0415

    staged = result.staged
    dense = staged.directory / "dense"
    shutil.copytree(
        staged.directory,
        dense,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("dense", "curriculum", "cl_prob", "jax_*", "__pycache__"),
    )
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
        templates = [
            rows_for.iloc[0].to_dict() for _, rows_for in group.groupby("observableId", sort=False)
        ]
        for i, t in enumerate(grid):
            # 1.0, not 0.0: the grid's measured values are never read either, but `_import`
            # validates, and lint refuses a non-positive measurement under a `log`/`log10`
            # observable transformation (perelson's viral load is one).
            rows.append({**templates[i % len(templates)], "time": float(t), "measurement": 1.0})
        # A grid shorter than the observable count would still drop some; they join at the end.
        for template in templates[len(grid) :]:
            rows.append({**template, "time": float(grid[-1]), "measurement": 1.0})
    pd.DataFrame(rows, columns=measurements.columns).to_csv(
        dense / "measurements.tsv", sep="\t", index=False
    )
    imported = _import(replace(staged, directory=dense, yaml=dense / "problem.yaml"))
    imported = eqx.tree_at(lambda p: p.model.nns, imported, result.problem.model.nns)
    imported = imported.update_parameters(result.problem.parameters)
    x, extra = run_simulations(imported, ret=ReturnValue.x, **_solver(result.config))
    times = np.asarray(extra["ts"])
    experiments = [e.id for e in imported._petab_problem.experiments]
    return {
        eid: {"times": times[i], "states": np.asarray(x)[i]} for i, eid in enumerate(experiments)
    }


def state_names(result: TrainResult) -> tuple[str, ...]:
    """The model's state order, which :func:`trajectories` reports in."""
    return tuple(result.problem.model.state_ids)


def save_bundle(result: TrainResult, directory: Path | str) -> Path:
    """The trained field as a complete PEtab SciML problem in ``directory``: the staged files with
    the trained weights in the array file and the fitted scalars as nominal values, so AMICI, or
    anyone reading the standard, re-imports exactly the model that was scored."""
    from petab_sciml import ArrayData, ArrayDataStandard  # noqa: PLC0415

    directory = Path(directory)
    shutil.copytree(
        result.staged.directory,
        directory,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("dense", "curriculum", "cl_prob", "jax_*", "__pycache__"),
    )
    network = result.problem.model.nns[S.NETWORK_ID]
    arrays = {
        name: {"weight": np.asarray(layer.weight), "bias": np.asarray(layer.bias)}
        for name, layer in network.layers.items()
    }
    ArrayDataStandard.save_data(
        ArrayData(metadata={"pytorch_format": True}, parameters={S.NETWORK_ID: arrays}),
        str(directory / f"{S.NETWORK_ID}_ps.hdf5"),
    )
    parameters = pd.read_csv(directory / "parameters.tsv", sep="\t")
    parameters["nominalValue"] = parameters["nominalValue"].astype(object)
    for i, pid in enumerate(parameters["parameterId"]):
        if pid in result.parameters:
            parameters.loc[i, "nominalValue"] = result.parameters[pid]
    parameters.to_csv(directory / "parameters.tsv", sep="\t", index=False)
    return directory


def _curriculum(
    problem: Problem, staged: S.Staged, config: TrainConfig
) -> tuple[list[S.Staged], tuple[str, ...]]:
    """The stage problems to fit in order, ending with ``staged`` itself, and any note on why the
    curriculum is shorter than asked."""
    if config.horizon_stages > 1:
        return _horizon_curriculum(staged, config)
    if config.n_windows <= 1:
        return [staged], ()
    if problem.uses_preequilibration:
        return [staged], (
            f"n_windows={config.n_windows} ignored: the multiple-shooting rewrite refuses "
            "preequilibration",
        )
    from petab_sciml.training import CurriculumMultipleShootingProblem, UniformPartition  # noqa: PLC0415

    # Every window must have length; with few measured time points the partition cannot give
    # that many, and fewer windows beat a degenerate one.
    measurements = pd.read_csv(staged.directory / "measurements.tsv", sep="\t")
    t_min = float(measurements["time"].min())
    if t_min > 0.0:
        # The rewrite partitions the MEASURED span, so its first window would begin at the first
        # datum and the start states applied at t=0 would fall outside every window.
        return [staged], (
            f"n_windows={config.n_windows} ignored: the data start at t={t_min:g}, after the "
            "experiments do, and the multiple-shooting rewrite would drop the starts",
        )
    n_windows, notes = int(config.n_windows), ()
    while n_windows > 1:
        ends = (
            UniformPartition(n_windows).get_time_points(measurements)
            if len(measurements["time"].unique()) >= n_windows
            else []
        )
        if ends and all(end > start for start, end in zip([t_min, *ends[:-1]], ends)):
            break
        n_windows -= 1
    if n_windows < config.n_windows:
        notes = (
            f"n_windows={config.n_windows} reduced to {n_windows}: the data have too few time points "
            "for more windows of nonzero length",
        )
    if n_windows <= 1:
        return [staged], notes
    directory = staged.directory / "curriculum"
    shutil.rmtree(directory, ignore_errors=True)
    CurriculumMultipleShootingProblem(
        yaml=staged.yaml,
        partition=UniformPartition(n_windows),
        penalty=float(config.continuity_weight),
    ).export(directory, validate=False)
    stages = sorted(
        (p for p in directory.iterdir() if p.name.startswith("stage")),
        key=lambda p: int(p.name[5:]),
    )
    for stage in stages[:-1]:
        for name in _EXTENSION_FILES:
            shutil.copy(staged.directory / name, stage / name)
        _seed_window_starts(stage, staged)
    return [
        *(replace(staged, directory=s, yaml=s / "problem.yaml") for s in stages[:-1]),
        staged,
    ], notes


def _horizon_curriculum(
    staged: S.Staged, config: TrainConfig
) -> tuple[list[S.Staged], tuple[str, ...]]:
    """Stages that fit a growing slice of the time course, the last being the whole of it.

    `petab_sciml.training.CurriculumLearningProblem` builds these -- "training difficulty is
    progressively increased across stages by gradually extending the measurement time horizon" --
    and takes the same `UniformPartition` the multiple-shooting path two branches up already
    imports. An earlier version cut the measurement table here instead; the standard owns this and
    a second implementation could only drift from it.

    Nothing about the model, the conditions, the experiments or any preequilibration changes --
    only how much of the trajectory the likelihood sees. That is why this applies where the
    multiple-shooting rewrite cannot: it needs no datum at t=0 (crauste starts at t=4) and does not
    refuse preequilibration (perelson's). The trainer carries the weights across the stages, so an
    early stage is a cheap initialisation for the next, which is the point.

    What the export does not do is keep every observable present: it truncates, so an observable
    whose first datum comes late vanishes from the early stages. That is restored below.
    """
    from petab_sciml.training import CurriculumLearningProblem, UniformPartition  # noqa: PLC0415

    measurements = pd.read_csv(staged.directory / "measurements.tsv", sep="\t")
    finite = np.sort(measurements["time"].unique())
    finite = finite[np.isfinite(finite)]
    asked = int(config.horizon_stages)
    if len(finite) < 2:
        return [staged], (f"horizon_stages={asked} ignored: the data have one finite time point",)
    stages = min(asked, len(finite))

    directory = staged.directory / "cl_prob"
    shutil.rmtree(directory, ignore_errors=True)
    CurriculumLearningProblem(yaml=str(staged.yaml), partition=UniformPartition(n=stages)).export()
    written = sorted(
        (p for p in directory.iterdir() if p.is_dir()),
        key=lambda p: int("".join(c for c in p.name if c.isdigit()) or 0),
    )
    if len(written) != stages:
        raise ValueError(f"the curriculum asked for {stages} stages and got {len(written)}")

    # The partition is the library's; keeping every observable PRESENT in every stage is ours. A
    # cutoff drops an observable whose first datum comes later, and an observable with no row
    # takes its observable/noise parameters out of use, which lint reports as extraneous -- the
    # failure the dense sampling grid already hit. So each stage gets back the earliest row of any
    # (experiment, observable) pair its cutoff removed, taken from the last stage, which is the
    # whole time course written in the export's own schema.
    keys = ["experimentId", "observableId"]
    whole = pd.read_csv(written[-1] / "measurements.tsv", sep="\t")
    if set(map(tuple, whole[keys].to_numpy())) != set(map(tuple, measurements[keys].to_numpy())):
        raise ValueError("the last curriculum stage is not the whole time course")
    earliest = whole.loc[whole.groupby(keys, sort=False)["time"].idxmin()]
    for stage in written[:-1]:
        table = pd.read_csv(stage / "measurements.tsv", sep="\t")
        present = set(map(tuple, table[keys].to_numpy()))
        missing = earliest[[tuple(pair) not in present for pair in earliest[keys].to_numpy()]]
        if len(missing):
            rows = pd.concat([table, missing]).sort_values([*keys, "time"], kind="stable")
            rows.to_csv(stage / "measurements.tsv", sep="\t", index=False)

    # The export carries the PEtab tables; the SciML extension's files (the network, its weights
    # and the mapping) are the design's and are copied beside them where the export did not.
    for stage in written:
        for name in (p.name for p in staged.directory.iterdir() if p.is_file()):
            if not (stage / name).exists():
                shutil.copy2(staged.directory / name, stage / name)
    out = [replace(staged, directory=stage, yaml=stage / staged.yaml.name) for stage in written]
    notes = (
        ()
        if stages == asked
        else (f"horizon_stages={asked} reduced to {stages}: only {len(finite)} finite time points",)
    )
    return out, notes


def _seed_window_starts(stage: Path, staged: S.Staged) -> None:
    """A window's start parameters begin at the identity readout measured where the window starts,
    else at the start guess in the state's own units; the rewrite's one value for all would put a
    state of magnitude 300 at 1e-6."""
    parameters = pd.read_csv(stage / "parameters.tsv", sep="\t")
    parameters["nominalValue"] = parameters["nominalValue"].astype(
        object
    )  # the column also holds "array"
    experiments = pd.read_csv(stage / "experiments.tsv", sep="\t")
    measurements = pd.read_csv(staged.directory / "measurements.tsv", sep="\t")
    observables = pd.read_csv(staged.directory / "observables.tsv", sep="\t")
    identity = {
        str(f).strip(): oid
        for oid, f in zip(observables["observableId"], observables["observableFormula"])
        if str(f).strip() in staged.states
    }
    for i, pid in enumerate(parameters["parameterId"]):
        if not (str(pid).startswith("WINDOW") and "_PARAMETER_" in str(pid)):
            continue
        prefix, state = str(pid).rsplit("_PARAMETER_", 1)
        experiment = prefix.split("_EXPERIMENT_", 1)[1]
        t0 = float(
            experiments.loc[experiments["experimentId"] == prefix, "time"].astype(float).min()
        )
        guess = S.START_GUESS_FRACTION * staged.scales.get(state, 1.0)
        if state in identity:
            rows = measurements[
                (measurements["experimentId"] == experiment)
                & (measurements["time"] == t0)
                & (measurements["observableId"] == identity[state])
            ]
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


def _weight_matrices(networks: dict) -> list:
    """The field's dense weight matrices, FIRST LAYER FIRST. Biases are left out: a bias is a
    constant production term, not a dependency on an input variable."""
    layers = networks[S.NETWORK_ID].layers
    return [
        layers[name].weight for name in sorted(layers, key=lambda n: int(n.removeprefix("layer")))
    ]


def _variable_groups(imported, config: TrainConfig):
    """First-layer columns grouped by the input variable they carry, products folded into the
    variables they are built from (so dropping a variable drops its products too)."""
    columns = _weight_matrices(imported.model.nns).__getitem__(0).shape[1]
    if not config.multiplicative:
        return R.variable_groups(columns)
    base = int(round((-1 + (1 + 8 * columns) ** 0.5) / 2)) - 1  # columns = b + b(b+1)/2
    products = [(a, b) for a in range(base) for b in range(a, base)]
    return R.variable_groups(columns, products)


def _threshold(trained, config: TrainConfig, groups):
    """The proximal step: soft-threshold the first layer's variable groups to EXACT zeros. A
    penalty optimised with Adam only shrinks a group; this is what removes a variable."""
    if config.penalty != "l21" or not config.penalty_weight:
        return trained
    networks, scalars = trained
    net = networks[S.NETWORK_ID]
    first = sorted(net.layers, key=lambda n: int(n.removeprefix("layer")))[0]
    updated = R.proximal_group_threshold(
        net.layers[first].weight, config.penalty_weight * config.learning_rate, groups
    )
    net = eqx.tree_at(lambda n: n.layers[first].weight, net, updated)
    return {**networks, S.NETWORK_ID: net}, scalars


def _stepper(imported, config: TrainConfig, *, freeze_network: bool = False):
    """The objective and one jitted Adam step for ``imported``, built once so every restart reuses
    the compilation. ``freeze_network`` fits the scalars alone, the weights held as they are."""
    from amici.sim.jax import run_simulations  # noqa: PLC0415

    solver = _solver(config)
    optimiser = optax.adam(config.learning_rate)

    groups = _variable_groups(imported, config)

    def objective(trained, base):
        networks, log_scalars = trained
        if freeze_network:
            networks = jax.tree_util.tree_map(
                lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_inexact_array(leaf) else leaf,
                networks,
            )
        llh, _ = run_simulations(_with(base, networks, log_scalars), **solver)
        if config.penalty == "none" or not config.penalty_weight:
            return -llh
        return -llh + config.penalty_weight * R.penalty(
            config.penalty, _weight_matrices(networks), groups=groups
        )

    @eqx.filter_jit
    def step(trained, state, base):
        value, gradient = eqx.filter_value_and_grad(objective)(trained, base)
        updates, state = optimiser.update(
            gradient, state, eqx.filter(trained, eqx.is_inexact_array)
        )
        return value, eqx.apply_updates(trained, updates), state

    # Built here, once, so a fides restart reuses the executable instead of compiling its own.
    return objective, step, optimiser, eqx.filter_jit(eqx.filter_value_and_grad(objective))


#: Bound on every coordinate fides searches -- network weights and the LOGS of the scalars alike.
#: The method needs a finite box; exp(20) covers any sigma or start these problems carry, and a
#: weight of 20 is far outside anything a trained tanh field uses.
FIDES_BOUND = 20.0


def _fit_fides(
    imported, stepper, networks: dict, scalars: dict[str, float], steps: int, config: TrainConfig
):
    """One stage under fides, with the same signature and return as :func:`_fit`.

    The trained state is a pytree of network weights and the scalars' logs; fides wants one flat
    vector, so it is ravelled once and the unravel closed over. Curvature comes from a BFGS update
    over the gradients JAX returns -- no second derivatives are computed -- so an iteration costs
    one value-and-gradient, as an Adam step does.
    """
    import fides  # noqa: PLC0415
    import numpy as _np  # noqa: PLC0415
    from jax.flatten_util import ravel_pytree  # noqa: PLC0415

    _objective, _step, _optimiser, value_and_grad = stepper
    ids = list(imported.parameter_ids)
    nominal = np.asarray(imported.parameters, dtype=float)
    start = jnp.log(
        jnp.maximum(
            jnp.asarray([scalars.get(pid, nominal[i]) for i, pid in enumerate(ids)]), LOG_FLOOR
        )
    )
    params, static = eqx.partition((networks, start), eqx.is_inexact_array)
    flat, unravel = ravel_pytree(params)

    calls = {"n": 0}

    def fun(vector):
        calls["n"] += 1
        # The ravel happens OUTSIDE the jit deliberately. Closing a jitted function over `unravel`
        # would compile one XLA executable per restart, and fifty of them exhaust vm.max_map_count
        # -- the `LLVM compilation error: Cannot allocate memory` that `pysr_worker` documents.
        # `value_and_grad` comes from the stepper, built once for the whole run.
        trained = eqx.combine(unravel(jnp.asarray(vector)), static)
        value, grads = value_and_grad(trained, imported)
        gradient = ravel_pytree(eqx.filter(grads, eqx.is_inexact_array))[0]
        value, gradient = float(value), _np.asarray(gradient, dtype=float)
        if not _np.isfinite(value) or not _np.all(_np.isfinite(gradient)):
            # fides reads a non-finite objective as a step to reject, not a run to end -- the
            # trust region shrinks and it tries again, which is what a failed solve deserves.
            return _np.inf, _np.zeros_like(gradient)
        return value, gradient

    bound = _np.full(flat.size, FIDES_BOUND)
    optimiser = fides.Optimizer(
        fun,
        ub=bound,
        lb=-bound,
        verbose=0,
        hessian_update=fides.BFGS(),
        options={fides.Options.MAXITER: int(steps)},
    )
    try:
        value, vector, _grad, _hess = optimiser.minimize(_np.asarray(flat, dtype=float))
    except Exception as failure:  # noqa: BLE001 -- reported like a failed restart
        return (
            np.inf,
            _unpack(eqx.combine(unravel(flat), static), ids),
            calls["n"],
            f"fides raised {failure}",
        )
    if not np.isfinite(value):
        return (
            np.inf,
            _unpack(eqx.combine(unravel(flat), static), ids),
            calls["n"],
            "fides found no finite optimum",
        )
    # None, whatever fides exited on: FTOL/GTOL/XTOL are it CONVERGING and MAXITER is it spending
    # its budget, and NONE of those is a failed restart. The caller reads a non-None fourth value
    # as an aborted restart and throws the fit away -- reporting the exit flag there cost 106 array
    # tasks, each discarding a converged fit after forty minutes. The flag is not surfaced.
    return (
        float(value),
        _unpack(eqx.combine(unravel(jnp.asarray(vector)), static), ids),
        calls["n"],
        None,
    )


def _fit_hybrid(
    imported, stepper, networks: dict, scalars: dict[str, float], steps: int, config: TrainConfig
):
    """Adam, then fides from where it stopped.

    They fail in opposite directions on this objective. Adam takes a fixed step however curved the
    landscape is, so it wanders -- slowly, but into good basins; crauste needed 80,000 steps and
    still improved. fides descends its basin decisively and stops, which from a random field means
    a nearby poor optimum: on perelson it converged to 304.25 from every one of four starts, where
    Adam reached 222.12. So Adam spends `adam_share` of the budget finding the basin and fides
    spends the rest reaching its floor.
    """
    share = min(max(float(config.adam_share), 0.0), 1.0)
    adam_steps = max(1, int(round(steps * share)))
    value, (networks, scalars), taken, failure = _fit(
        imported, stepper, networks, scalars, adam_steps, config
    )
    if failure is not None and not np.isfinite(value):
        return value, (networks, scalars), taken, failure
    polished, state, more, why = _fit_fides(
        imported, stepper, networks, scalars, max(1, steps - adam_steps), config
    )
    if not np.isfinite(polished) or polished > value:
        # fides could not better Adam's point; keep the one that fits and say so.
        return value, (networks, scalars), taken + more, None
    return polished, state, taken + more, why


def _support_mask(networks: dict):
    """Which first-layer weights the penalty left alive, as a 0/1 array."""
    return (np.asarray(_weight_matrices(networks)[0]) != 0.0).astype(float)


def _apply_support(trained, support):
    """Hold the zeroed columns at zero, so a refit cannot reintroduce a dropped variable."""
    if support is None:
        return trained
    networks, scalars = trained
    net = networks[S.NETWORK_ID]
    first = sorted(net.layers, key=lambda n: int(n.removeprefix("layer")))[0]
    net = eqx.tree_at(
        lambda n: n.layers[first].weight, net, net.layers[first].weight * jnp.asarray(support)
    )
    return {**networks, S.NETWORK_ID: net}, scalars


def _fit(
    imported,
    stepper,
    networks: dict,
    scalars: dict[str, float],
    steps: int,
    config: TrainConfig,
    *,
    support=None,
    on_checkpoint=None,
):
    """Fit one stage from ``networks`` and the scalars it recognises by id (the rest start at
    their nominal values). Returns the best objective, ``(networks, scalars)`` at it, the steps
    taken and the failure, if any.

    ``on_checkpoint(taken, best_value, (networks, scalars))`` fires at each step count in
    ``config.checkpoints``, carrying the best optimum SEEN so far -- what this fit would return
    had ``steps`` been ``taken``. It cannot fire after an early stop, because there is no such run.
    """
    objective, step, optimiser, _value_and_grad = stepper
    wanted = set(config.checkpoints) if on_checkpoint is not None else set()
    ids = list(imported.parameter_ids)
    nominal = np.asarray(imported.parameters, dtype=float)
    start = jnp.log(
        jnp.maximum(
            jnp.asarray([scalars.get(pid, nominal[i]) for i, pid in enumerate(ids)]), LOG_FLOOR
        )
    )
    trained = (networks, start)
    state = optimiser.init(eqx.filter(trained, eqx.is_inexact_array))
    best_value, best_trained, since_improvement, taken = np.inf, trained, 0, 0
    for i in range(steps):
        value, candidate, state = step(trained, state, imported)
        taken += 1
        value = float(value)
        if not np.isfinite(value):
            return (
                best_value,
                _unpack(best_trained, ids),
                taken,
                f"non-finite objective at step {i}",
            )
        if value < best_value - config.min_improvement:
            best_value, best_trained, since_improvement = value, trained, 0
        else:
            since_improvement += 1
        trained = _apply_support(
            _threshold(candidate, config, _variable_groups(imported, config)), support
        )
        # The LAST step's checkpoint waits for the post-loop refinement below, or it would not be
        # what a run of that budget returns -- which is the one thing a checkpoint promises.
        if taken in wanted and taken < steps:
            on_checkpoint(taken, best_value, _unpack(best_trained, ids))
        if since_improvement >= config.patience:
            break
    else:
        final = float(objective(trained, imported))
        if np.isfinite(final) and final < best_value:
            best_value, best_trained = final, trained
        if steps in wanted:
            on_checkpoint(steps, best_value, _unpack(best_trained, ids))
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

    return {
        "solver": diffrax.Kvaerno5(),
        "controller": diffrax.PIDController(rtol=config.rtol, atol=config.atol),
        "max_steps": int(config.max_solver_steps),
    }


def _import(staged: S.Staged):
    """The staged problem as AMICI's ``JAXProblem``. AMICI resolves the extension's files against
    the working directory, hence the change of directory for the import."""
    from amici.importers.petab import PetabImporter  # noqa: PLC0415
    from petab import v2  # noqa: PLC0415

    digest = hashlib.md5(str(staged.directory.resolve()).encode()).hexdigest()[:10]
    with _working_directory(staged.directory):
        petab_problem = v2.Problem.from_yaml(staged.yaml.name)
        importer = PetabImporter(
            petab_problem,
            module_name=f"node_{digest}",
            compile_=True,
            jax=True,
            validate=True,
            output_dir=str(staged.directory / f"jax_{digest}"),
        )
        return importer.create_simulator(force_import=True)


@contextmanager
def _working_directory(path: Path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
