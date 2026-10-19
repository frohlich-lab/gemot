"""Write the neural ODE as a PEtab SciML problem AMICI's JAX backend can import.

A :class:`Problem` says which symbols are states, which condition targets dose them or clamp an
input, and how large everything is. Staging turns that into files: an SBML model whose species
are the states, each with a rate rule ``x' = scale_x / horizon * f_x`` fed by a tanh MLP over the
normalised states and inputs; the network in the standard's YAML plus the equinox module AMICI
loads; its initial weights as an HDF5 array file, so the problem is complete and lint-clean; the
hybridization and mapping tables that wire the two together; and the shipped PEtab v2 tables
around it, as the server holds them (:mod:`.problem`), so a v1 problem arrives already upgraded.
Nothing is converted here: that would lint the tables against the shipped model, and the zero-shot
problems ship an empty SBML against which every condition target and observable symbol is an
unknown entity.

The start states go into the condition table, where a state's start is a plain target: pinned to
a datum where the experiment starts, a setup-sheet value or a fixed ``<state>_init`` where the
tables give one, otherwise an estimated parameter, one per experiment or shared. Under
pre-equilibration the starts belong to the basal conditions and the settled state carries over
(:func:`_design`).

The network AMICI loads is the equinox module, because petab 0.9 reads a YAML-format network back
through PyTorch and PyTorch is not a dependency here; the YAML is written beside it as the
portable record of the architecture. Time is normalised in the rate rules rather than in the
network, so the field is O(1) in its own clock and derivatives come out in the problem's units.

A ``log10`` observable's noise formula comes out multiplied by ``ln 10``. PEtab v2 has no
log10-normal noise, so petab's upgrader writes log-normal, which alone would change the likelihood:
the residual ``(log10 y - log10 ŷ) / σ`` becomes ``(ln y - ln ŷ) / σ``. With ``σ`` scaled by
``ln 10`` the two densities are identical, the Jacobian term included, so the field's nll stays
comparable with the reference's. The server's upgrade applies that factor itself
(PEtab-dev/libpetab-python#535); this module applied it as well before that, which counted it
twice. The readout scale factors are FIXED at their nominal values, though the shipped table
estimates them. A neural state's magnitude is free, so a scale factor times that state is all the data
constrain, and estimating both let fujita's drift to 1e-8 and 1e5 while the states went the
other way -- a fit no better on the training data and worse off it. Fixed, the state carries the
magnitude, which is what the trainer this replaces did by making the states the readouts.
"""

from __future__ import annotations

import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import yaml

from . import problem as P
from .problem import Problem

#: The one network a staged problem carries, as the tables and the yaml name it.
NETWORK_ID = "net1"
#: Where an estimated start is initialised and how far its bounds reach, in units of the state's
#: scale. The paper seeds latent starts at 0.1.
START_GUESS_FRACTION = 0.1
START_UPPER_FACTOR = 10.0
#: The declared range of a network weight. Training is unconstrained; the bounds exist because
#: PEtab lint asks every estimated parameter of a problem with priors for finite ones.
WEIGHT_BOUND = 10.0
#: The v2 measurement columns staged; a held-out view's bookkeeping columns stay behind.
_MEASUREMENT_COLUMNS = (
    "experimentId",
    "observableId",
    "time",
    "measurement",
    "observableParameters",
    "noiseParameters",
)
#: The v2 parameter columns staged, after ``parameterId``.
_PARAMETER_COLUMNS = ("lowerBound", "upperBound", "nominalValue", "estimate")


@dataclass(frozen=True)
class Staged:
    """A PEtab v2 SciML problem on disk and what its network sees and drives."""

    directory: Path
    yaml: Path
    #: The network's outputs: observed, dosed, then latent states.
    states: tuple[str, ...]
    #: The network's inputs, in order: the states (and clock), clamped inputs, protocol targets.
    field_inputs: tuple[str, ...]
    #: Start parameters the staged problem estimates, besides the weights.
    estimated_starts: tuple[str, ...]
    n_weights: int
    scales: dict[str, float]
    horizon: float
    #: Pairwise products fed to the network as extra inputs; empty unless ``multiplicative``. NOT
    #: part of ``field_inputs``: the distiller regresses on the base variables and finds products
    #: itself, so these exist only to make the FIELD's job easier.
    products: tuple[tuple[str, str], ...] = ()


def stage(
    problem: Problem,
    directory: Path | str,
    *,
    n_augmented: int = 0,
    width: int = 32,
    depth: int = 2,
    clock: bool = False,
    shared_initial_conditions: bool = False,
    seed: int = 0,
    design: Staged | None = None,
    estimate_readout_scales: bool = False,
    multiplicative: bool = False,
) -> Staged:
    """Write ``problem`` as a neural ODE into ``directory`` and return what was written.

    ``n_augmented`` latent states are appended after the observed and dosed ones; ``clock`` adds
    a state that counts time as a network input; ``seed`` draws the initial weights. ``design``
    stages the problem in another staged problem's units instead -- its scales, horizon, latents,
    clock and network files -- which is how a held-out set is staged in the units the field was
    trained in; a problem whose states or inputs differ from the design's is refused.
    ``estimate_readout_scales`` leaves the readout scale factors estimated, for a transfer set
    whose scalings are not the training ones. Files already in ``directory`` are overwritten.

    The shipped preequilibration periods are always staged. Dropping them was once a knob, and it
    is not one: a problem that preequilibrates says its experiments begin at a steady state, so a
    run without them fits a DIFFERENT estimand -- free starts in place of a steady-state
    constraint -- whose likelihood is not comparable with the reference's. On perelson that knob
    moved the training nll from 3060 to 227 and none of it was method. Where a field has no
    steady state to reach (a random field under a zero input), the honest result is that the
    method cannot fit that problem, which belongs in a writeup and not in a config.
    """
    import antimony  # noqa: PLC0415

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if design is not None:
        n_augmented = len(design.states) - len(problem.state_names)
        clock = "clock" in design.field_inputs
        multiplicative = bool(design.products)
    latents = tuple(f"z{i + 1}" for i in range(int(n_augmented)))
    states = (*problem.state_names, *latents)
    integrated = (*states, "clock") if clock else states
    field_inputs = (*integrated, *problem.field_input_names)
    scales = {**problem.scales, **{z: 1.0 for z in latents}, "clock": problem.horizon}
    horizon = problem.horizon
    if design is not None:
        if states != design.states or field_inputs != design.field_inputs:
            raise ValueError(
                f"{problem.name} does not fit the design it is staged in: states {states} "
                f"against {design.states}, inputs {field_inputs} against {design.field_inputs}"
            )
        scales, horizon = dict(design.scales), design.horizon

    # The model, then the shipped tables around it.
    products = product_features(field_inputs) if multiplicative else []
    network_inputs = (*field_inputs, *(product_symbol(a, b) for a, b in products))
    text = _antimony(problem, states, integrated, scales, horizon, products)
    antimony.clearPreviousLoads()
    if antimony.loadAntimonyString(text) < 0:
        raise ValueError(f"the generated model is not valid Antimony: {antimony.getLastError()}")
    sbml = antimony.getSBMLString("neural_ode")
    (directory / "model.ant").write_text(text)
    (directory / "model.xml").write_text(sbml)

    # The network: its architecture, the module AMICI loads, and its initial weights.
    if design is None:
        n_weights = _write_network(directory, len(network_inputs), len(states), width, depth, seed)
    else:
        for name in (f"{NETWORK_ID}.py", f"{NETWORK_ID}.yaml", f"{NETWORK_ID}_ps.hdf5"):
            shutil.copy(design.directory / name, directory / name)
        n_weights = design.n_weights
    pd.DataFrame(
        {
            "targetId": [f"{NETWORK_ID}_input{i}" for i in range(len(network_inputs))]
            + [f"f_{s}" for s in states],
            "targetValue": [f"{s}_norm" for s in field_inputs]
            + [product_symbol(a, b) for a, b in products]
            + [f"{NETWORK_ID}_output{i}" for i in range(len(states))],
        }
    ).to_csv(directory / "hybridization.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "petabEntityId": [f"{NETWORK_ID}_input{i}" for i in range(len(network_inputs))]
            + [f"{NETWORK_ID}_output{i}" for i in range(len(states))]
            + [f"{NETWORK_ID}_ps"],
            "modelEntityId": [f"{NETWORK_ID}.inputs[0][{i}]" for i in range(len(network_inputs))]
            + [f"{NETWORK_ID}.outputs[0][{i}]" for i in range(len(states))]
            + [f"{NETWORK_ID}.parameters"],
        }
    ).to_csv(directory / "mapping.tsv", sep="\t", index=False)

    # The design: each experiment's periods, its first applying the starts (see `_design`).
    experiments, conditions, estimated = _design(
        problem, states, scales, shared=shared_initial_conditions
    )
    experiments.to_csv(directory / "experiments.tsv", sep="\t", index=False)
    conditions.to_csv(directory / "conditions.tsv", sep="\t", index=False)
    tables = problem.tables
    tables.observable_df.to_csv(directory / "observables.tsv", sep="\t")
    measurements = tables.measurement_df
    measurements = measurements[[c for c in _MEASUREMENT_COLUMNS if c in measurements]].assign(
        experimentId=[
            e if isinstance(e, str) and e.strip() else P._SINGLE
            for e in measurements["experimentId"]
        ]
    )
    measurements.to_csv(directory / "measurements.tsv", sep="\t", index=False)
    parameters = _parameters(problem, conditions, fix_readout_scales=not estimate_readout_scales)
    rows = [*estimated.values(), (f"{NETWORK_ID}_ps", -WEIGHT_BOUND, WEIGHT_BOUND, "array", "true")]
    extra = pd.DataFrame(
        rows, columns=["parameterId", "lowerBound", "upperBound", "nominalValue", "estimate"]
    )
    pd.concat([parameters, extra]).to_csv(directory / "parameters.tsv", sep="\t", index=False)

    document = {
        "format_version": "2.0.0",
        "parameter_files": ["parameters.tsv"],
        "model_files": {"model": {"location": "model.xml", "language": "sbml"}},
        "measurement_files": ["measurements.tsv"],
        "condition_files": ["conditions.tsv"],
        "experiment_files": ["experiments.tsv"],
        "observable_files": ["observables.tsv"],
        "mapping_files": ["mapping.tsv"],
    }
    document["extensions"] = {
        "sciml": {
            "version": "0.1.0",
            "required": True,
            "array_files": [f"{NETWORK_ID}_ps.hdf5"],
            "hybridization_files": ["hybridization.tsv"],
            "neural_networks": {
                NETWORK_ID: {
                    "location": f"{NETWORK_ID}.py",
                    "pre_initialization": False,
                    "format": "equinox",
                }
            },
        }
    }
    (directory / "problem.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    return Staged(
        directory=directory,
        yaml=directory / "problem.yaml",
        states=states,
        field_inputs=field_inputs,
        estimated_starts=tuple(estimated),
        n_weights=n_weights,
        products=tuple(products),
        scales=scales,
        horizon=horizon,
    )


def product_features(field_inputs) -> list[tuple[str, str]]:
    """The pairwise products of the field's inputs, squares included, as ``(a, b)`` pairs.

    Feature augmentation, NOT a structural constraint: a product is made easy to represent without
    stopping the network representing anything it could before, so the MLP stays universal. Mass
    action IS a product and saturating kinetics needs an x**2 in a denominator, and a tanh MLP has
    to build those out of sums -- approximately, and in a form symbolic regression then struggles
    to recover. They are computed as assignment rules in the model and fed in as extra network
    INPUTS, so the network stays a plain MLP the standard can express.
    """
    return [(a, b) for i, a in enumerate(field_inputs) for b in field_inputs[i:]]


def product_symbol(a: str, b: str) -> str:
    return f"{a}_x_{b}"


def _antimony(problem: Problem, states, integrated, scales, horizon: float, products=()) -> str:
    lines = [
        "model neural_ode",
        "  compartment cell = 1;",
        "  species " + ", ".join(f"{s} in cell" for s in integrated) + ";",
    ]
    lines += [f"  {s} = 0;" for s in integrated]
    lines += [f"  {u} = 0;" for u in problem.inputs]
    lines += [f"  {item.rule};" for item in problem.timed_inputs]
    lines += [
        f"  {s}_norm := {s} / {scales[s]!r};" for s in (*integrated, *problem.field_input_names)
    ]
    lines += [f"  {product_symbol(a, b)} := {a}_norm * {b}_norm;" for a, b in products]
    lines += [f"  f_{s} = 0;" for s in states]
    lines += [f"  {s}' = ({scales[s]!r} / {horizon!r}) * f_{s};" for s in states]
    if "clock" in integrated:
        lines.append("  clock' = 1;")
    return "\n".join((*lines, "end", ""))


def _parameters(problem: Problem, conditions: pd.DataFrame, *, fix_readout_scales: bool = True):
    """The shipped parameter table cut to what the observables, measurements and staged conditions
    reference, the readout scale factors fixed, and each estimated sigma started at its data's
    spread. The noise model stays the shipped one, flags included. v2 has no estimation scale, so
    the server's ``parameterScale`` extension is left behind: training works on logs anyway."""
    tables = problem.tables
    parameters = tables.parameter_df
    referenced = set(problem.noise_parameters)
    for column in ("observableFormula", "noiseFormula"):
        for formula in tables.observable_df[column].dropna().astype(str):
            referenced.update(parameters.index.intersection(_names(formula)))
    for column in ("observableParameters", "noiseParameters"):
        if column in tables.measurement_df:
            for cell in tables.measurement_df[column].dropna().astype(str):
                referenced.update(parameters.index.intersection(cell.split(";")))
    for value in conditions["targetValue"].astype(str):
        referenced.update(parameters.index.intersection(_names(value)))
    kept = parameters.loc[[p for p in parameters.index if p in referenced]].copy()
    kept["estimate"] = kept["estimate"].astype(bool)
    if fix_readout_scales:
        kept.loc[[p for p in kept.index if p in problem.observable_parameters], "estimate"] = False
    seeds = {pid: seed for pid, seed in _sigma_seeds(problem).items() if pid in kept.index}
    if seeds:
        kept["nominalValue"] = kept["nominalValue"].astype(float)
        for pid, seed in seeds.items():
            kept.loc[pid, "nominalValue"] = min(
                max(seed, float(kept.loc[pid, "lowerBound"])), float(kept.loc[pid, "upperBound"])
            )
    kept["estimate"] = kept["estimate"].map({True: "true", False: "false"})
    return kept.reset_index()[["parameterId", *_PARAMETER_COLUMNS]]


def _names(formula: str) -> list[str]:
    from .problem import _SYMBOL  # noqa: PLC0415

    return _SYMBOL.findall(formula)


def _write_network(
    directory: Path, n_in: int, n_out: int, width: int, depth: int, seed: int
) -> int:
    """The YAML architecture, the equinox module and the initial weights; returns the weight count."""
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    import amici  # noqa: PLC0415
    import jax.random as jr  # noqa: PLC0415
    from amici.exporters.jax.nn import generate_equinox  # noqa: PLC0415
    from petab_sciml import ArrayData, ArrayDataStandard, NNModelStandard  # noqa: PLC0415
    from petab_sciml.standard.nn_model import Input, Layer, NNModel, Node  # noqa: PLC0415

    sizes = [n_in, *([width] * depth), n_out]
    layers = [
        Layer(
            layer_id=f"layer{i + 1}",
            layer_type="Linear",
            args={"in_features": a, "out_features": b, "bias": True},
        )
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))
    ]
    forward = [Node(name="net_input", op="placeholder", target="net_input", args=[], kwargs={})]
    previous = "net_input"
    for i in range(len(layers)):
        forward.append(
            Node(
                name=f"layer{i + 1}",
                op="call_module",
                target=f"layer{i + 1}",
                args=[previous],
                kwargs={},
            )
        )
        previous = f"layer{i + 1}"
        if i < len(layers) - 1:
            forward.append(
                Node(
                    name=f"tanh{i + 1}", op="call_method", target="tanh", args=[previous], kwargs={}
                )
            )
            previous = f"tanh{i + 1}"
    forward.append(Node(name="output", op="output", target="output", args=[previous], kwargs={}))
    model = NNModel(
        nn_model_id=NETWORK_ID, inputs=[Input(input_id="input0")], layers=layers, forward=forward
    )
    NNModelStandard.save_data(model, str(directory / f"{NETWORK_ID}.yaml"))
    generate_equinox(model, directory / f"{NETWORK_ID}.py")
    module_name = f"{NETWORK_ID}_{hashlib.md5(str(directory.resolve()).encode()).hexdigest()[:10]}"
    net = amici._module_from_path(module_name, directory / f"{NETWORK_ID}.py").net(jr.PRNGKey(seed))
    arrays = {
        name: {"weight": np.asarray(layer.weight), "bias": np.asarray(layer.bias)}
        for name, layer in net.layers.items()
    }
    ArrayDataStandard.save_data(
        ArrayData(metadata={"pytorch_format": True}, parameters={NETWORK_ID: arrays}),
        str(directory / f"{NETWORK_ID}_ps.hdf5"),
    )
    return int(sum(a.size for d in arrays.values() for a in d.values()))


def _sigma_seeds(problem: Problem) -> dict[str, float]:
    """A start for each estimated sigma that alone scales some measurements: their spread, on the
    noise distribution's scale and in the sigma's own units. Adam in log space moves at most its
    learning rate per step, so a start decades off never arrives within the budget, and the
    table's nominal is not written to be a start.

    "In the sigma's own units" is the factor a noise formula multiplies its sigma by: a v1
    ``log10`` observable upgrades to log-normal noise ``sigma * ln(10)`` (see the module
    docstring), whose sigma is still in decades, so its start is the natural-log spread over
    ``ln 10``."""
    from petab.math import sympify_petab  # noqa: PLC0415
    from petab_mcp.problem import placeholders, split_overrides  # noqa: PLC0415

    tables = problem.tables
    observables = tables.observable_df
    factors: dict[tuple[str, str], tuple[str, float] | None] = {}

    def resolved(oid: str, cell) -> tuple[str, float] | None:
        """``(sigma, factor)`` where the row's noise formula is ``factor * sigma`` for one
        estimated sigma, else ``None``."""
        key = (oid, "" if cell is None or (isinstance(cell, float) and math.isnan(cell)) else cell)
        if key not in factors:
            expr = sympify_petab(str(observables.loc[oid, "noiseFormula"]))
            declared = placeholders(observables, oid, "noise")
            overrides = split_overrides(key[1])
            # By name: petab's symbols carry assumptions, so a bare `sp.Symbol` is another symbol.
            given = dict(zip(declared, overrides))
            expr = expr.subs(
                {s: sympify_petab(str(given[str(s)])) for s in expr.free_symbols if str(s) in given}
            )
            free = list(expr.free_symbols)
            factors[key] = None
            if len(free) == 1 and str(free[0]) in problem.noise_parameters:
                sigma = free[0]
                factor = expr.subs(sigma, 1)
                if factor.is_number and sp.simplify(expr - factor * sigma) == 0 and factor > 0:
                    factors[key] = (str(sigma), float(factor))
        return factors[key]

    pools: dict[str, list[float]] = {}
    for row in tables.measurement_df.itertuples(index=False):
        oid = str(row.observableId)
        found = resolved(oid, getattr(row, "noiseParameters", None))
        if found is None:
            continue
        sigma, factor = found
        y = float(row.measurement)
        if str(observables.loc[oid, "noiseDistribution"]) in ("log-normal", "log-laplace"):
            y = math.log(y)
        pools.setdefault(sigma, []).append(y / factor)
    return {
        pid: float(np.std(values))
        for pid, values in pools.items()
        if len(values) > 1 and np.std(values) > 0.0
    }


def _design(
    problem: Problem, states, scales, *, shared: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple]]:
    """The staged experiment and condition tables, and the start parameters to estimate.

    Every experiment's FIRST period -- its pre-equilibration where it has one -- applies one
    condition, ``start_<key>``: what the shipped conditions of that period set, their dosing
    targets renamed to the states they dose, and a start for every state they leave unset. The key
    is :meth:`Problem.start_key`, so experiments that settle in the same conditions share one start
    and the settled state carries over. A start the tables know is written as its number; a latent,
    or a state the tables leave open, gets one parameter per key or one shared across them. The
    later periods are the shipped ones.

    One condition rather than a start condition beside the shipped ones, so that no period the
    route stages depends on how an importer combines conditions applied together.
    """
    from petab.v2.C import TIME_PREEQUILIBRATION  # noqa: PLC0415

    renamed = {
        cid: {problem.start_columns.get(t, t): v for t, v in changes.items()}
        for cid, changes in P._changes(problem.tables.condition_df).items()
    }
    periods: list[tuple] = []
    starts: dict[str, dict[str, object]] = {}
    estimated: dict[str, tuple] = {}
    for experiment in problem.experiment_ids:
        key = problem.start_key(experiment)
        (first, applied), *rest = problem.periods[experiment]
        condition = f"start_{key}"
        periods.append((experiment, first, condition))
        # A later period that changes nothing -- no condition, or one with no rows, which is how
        # v2 writes a condition that sets nothing -- is not staged.
        periods += [(experiment, t, cid) for t, cids in rest for cid in cids if cid in renamed]
        if condition in starts:
            continue
        rows = starts[condition] = {
            t: v for cid in applied for t, v in renamed.get(cid, {}).items()
        }
        for state in states:
            if state in rows:
                continue
            if state in problem.pinned_starts:
                rows[state] = problem.pinned_starts[state]
                continue
            datum = (
                None
                if first == TIME_PREEQUILIBRATION
                else problem.measured_starts.get((experiment, state))
            )
            if datum is not None:
                rows[state] = float(datum)
                continue
            pid = f"{state}_0" if shared else f"{state}_0_{key}"
            estimated.setdefault(
                pid,
                (
                    pid,
                    0.0,
                    START_UPPER_FACTOR * scales[state],
                    START_GUESS_FRACTION * scales[state],
                    "true",
                ),
            )
            rows[state] = pid
    experiments = pd.DataFrame(periods, columns=["experimentId", "time", "conditionId"])
    later = dict.fromkeys(c for _, _, c in periods if c not in starts)
    conditions = pd.DataFrame(
        [(cid, t, v) for cid in later for t, v in renamed[cid].items()]
        + [(cid, t, v) for cid, rows in starts.items() for t, v in rows.items()],
        columns=["conditionId", "targetId", "targetValue"],
    )
    return experiments, conditions, estimated
