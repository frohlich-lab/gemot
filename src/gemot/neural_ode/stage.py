"""Write the neural ODE as a PEtab SciML problem AMICI's JAX backend can import.

A :class:`Problem` says which symbols are states, which condition columns dose them or clamp an
input, and how large everything is. Staging turns that into files: an SBML model whose species
are the states, each with a rate rule ``x' = scale_x / horizon * f_x`` fed by a tanh MLP over the
normalised states and inputs; the network in the standard's YAML plus the equinox module AMICI
loads; its initial weights as an HDF5 array file, so the problem is complete and lint-clean; the
hybridization and mapping tables that wire the two together; and the shipped measurement,
observable, condition and parameter tables, upgraded to PEtab v2 by petab's own converter.

Two orderings are deliberate. The upgrade runs AFTER the model is written: petab lints before it
converts, and the zero-shot problems ship an empty SBML, so against the shipped model every
condition column and observable symbol is an unknown entity and the stock converter refuses --
the reason an in-house wrapper once existed. Against the generated model every entity exists.
The start states are added AFTER the upgrade, in the v2 condition table, where a state's start is
a plain target: pinned to a t=0 datum, a setup-sheet value or a fixed ``<state>_init`` where the
tables give one, otherwise an estimated parameter, one per experiment or shared. Under
preequilibration the starts belong to the basal condition and the settled state carries over.

The network AMICI loads is the equinox module, because petab 0.9 reads a YAML-format network back
through PyTorch and PyTorch is not a dependency here; the YAML is written beside it as the
portable record of the architecture. Time is normalised in the rate rules rather than in the
network, so the field is O(1) in its own clock and derivatives come out in the problem's units.

A ``log10`` observable's noise formula comes out multiplied by ``ln 10``. PEtab v2 has no
log10-normal noise, so petab's upgrader writes log-normal, which alone would change the likelihood:
the residual ``(log10 y - log10 ŷ) / σ`` becomes ``(ln y - ln ŷ) / σ``. With ``σ`` scaled by
``ln 10`` the two densities are identical, the Jacobian term included, so the field's nll stays
comparable with the reference's. The upgrader applies that factor itself
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
import yaml

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
    import petab.v1 as petab  # noqa: PLC0415
    from petab.v2.petab1to2 import petab_files_1to2  # noqa: PLC0415

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = petab.Problem.from_yaml(str(problem.source))

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

    # The model, then the shipped tables around it, upgraded by petab.
    products = product_features(field_inputs) if multiplicative else []
    network_inputs = (*field_inputs, *(product_symbol(a, b) for a, b in products))
    text = _antimony(problem, states, integrated, scales, horizon, products)
    antimony.clearPreviousLoads()
    if antimony.loadAntimonyString(text) < 0:
        raise ValueError(f"the generated model is not valid Antimony: {antimony.getLastError()}")
    sbml = antimony.getSBMLString("neural_ode")
    v1 = directory / "v1"
    v1.mkdir(exist_ok=True)
    (v1 / "model.ant").write_text(text)
    (v1 / "model.xml").write_text(sbml)
    _stage_v1_tables(problem, source, v1, fix_readout_scales=not estimate_readout_scales)
    petab_files_1to2(v1 / "problem.yaml", directory)

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

    # Experiments are named after their simulation condition, the id the route knows; the
    # upgrader's own names depend on whether a condition has changes, and it names none at all
    # for a problem whose condition table has no columns.
    measurements = pd.read_csv(directory / "measurements.tsv", sep="\t")
    measurements["experimentId"] = (
        source.measurement_df[petab.C.SIMULATION_CONDITION_ID].astype(str).to_numpy()
    )
    measurements.to_csv(directory / "measurements.tsv", sep="\t", index=False)
    periods = []
    for experiment in problem.experiment_ids:
        if experiment in problem.preequilibration:
            periods.append((experiment, -np.inf, problem.preequilibration[experiment]))
        periods.append((experiment, 0.0, experiment))
    experiments = pd.DataFrame(periods, columns=["experimentId", "time", "conditionId"])
    experiments.to_csv(directory / "experiments.tsv", sep="\t", index=False)

    # Start states and the parameters the estimated ones need.
    starts, estimated = _starts(
        problem,
        experiments,
        states,
        latents,
        scales,
        shared=shared_initial_conditions,
        directory=directory,
    )
    conditions = pd.read_csv(directory / "conditions.tsv", sep="\t")
    conditions = pd.concat([conditions, pd.DataFrame(starts, columns=conditions.columns[:3])])
    conditions.to_csv(directory / "conditions.tsv", sep="\t", index=False)
    parameters = pd.read_csv(directory / "parameters.tsv", sep="\t")
    rows = [*estimated.values(), (f"{NETWORK_ID}_ps", -WEIGHT_BOUND, WEIGHT_BOUND, "array", "true")]
    extra = pd.DataFrame(
        rows, columns=["parameterId", "lowerBound", "upperBound", "nominalValue", "estimate"]
    )
    pd.concat([parameters, extra]).reindex(columns=list(parameters.columns)).to_csv(
        directory / "parameters.tsv", sep="\t", index=False
    )

    document = yaml.safe_load((directory / "problem.yaml").read_text())
    document["experiment_files"] = ["experiments.tsv"]
    document["mapping_files"] = ["mapping.tsv"]
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
    lines += [f"  {item.target} := {item.formula};" for item in problem.timed_inputs]
    lines += [
        f"  {s}_norm := {s} / {scales[s]!r};" for s in (*integrated, *problem.field_input_names)
    ]
    lines += [f"  {product_symbol(a, b)} := {a}_norm * {b}_norm;" for a, b in products]
    lines += [f"  f_{s} = 0;" for s in states]
    lines += [f"  {s}' = ({scales[s]!r} / {horizon!r}) * f_{s};" for s in states]
    if "clock" in integrated:
        lines.append("  clock' = 1;")
    return "\n".join((*lines, "end", ""))


def _stage_v1_tables(
    problem: Problem, source, v1: Path, *, fix_readout_scales: bool = True
) -> None:
    """The shipped tables beside the generated model: dosing columns renamed to the states they
    set, the parameter table cut to what the observables, measurements and conditions reference,
    the readout scale factors fixed, and each estimated sigma started at its data's spread. The
    noise model stays the shipped one, flags included."""
    conditions = source.condition_df
    if conditions is not None:
        conditions.rename(
            columns={c: s for c, s in problem.start_columns.items() if c != s}
        ).to_csv(v1 / "conditions.tsv", sep="\t")
    else:
        pd.DataFrame({"conditionId": list(problem.experiment_ids)}).to_csv(
            v1 / "conditions.tsv", sep="\t", index=False
        )
    source.measurement_df.to_csv(v1 / "measurements.tsv", sep="\t", index=False)
    # As shipped: the upgrader rescales a log10 observable's noise (see the module docstring).
    source.observable_df.to_csv(v1 / "observables.tsv", sep="\t")
    parameters = source.parameter_df
    referenced = set(problem.noise_parameters)
    for column in ("observableFormula", "noiseFormula"):
        for formula in source.observable_df[column].dropna().astype(str):
            referenced.update(parameters.index.intersection(_names(formula)))
    for column in ("observableParameters", "noiseParameters"):
        if column in source.measurement_df:
            for cell in source.measurement_df[column].dropna().astype(str):
                referenced.update(parameters.index.intersection(cell.split(";")))
    if conditions is not None:
        for column in conditions.columns:
            referenced.update(parameters.index.intersection(conditions[column].astype(str)))
    kept = parameters.loc[[p for p in parameters.index if p in referenced]].copy()
    if fix_readout_scales:
        kept.loc[[p for p in kept.index if p in problem.observable_parameters], "estimate"] = 0
    seeds = {pid: seed for pid, seed in _sigma_seeds(problem, source).items() if pid in kept.index}
    if seeds:
        kept["nominalValue"] = kept["nominalValue"].astype(float)
        for pid, seed in seeds.items():
            kept.loc[pid, "nominalValue"] = min(
                max(seed, float(kept.loc[pid, "lowerBound"])), float(kept.loc[pid, "upperBound"])
            )
    kept.to_csv(v1 / "parameters.tsv", sep="\t")
    (v1 / "problem.yaml").write_text(
        yaml.safe_dump(
            {
                "format_version": 1,
                "parameter_file": "parameters.tsv",
                "problems": [
                    {
                        "sbml_files": ["model.xml"],
                        "condition_files": ["conditions.tsv"],
                        "measurement_files": ["measurements.tsv"],
                        "observable_files": ["observables.tsv"],
                    }
                ],
            }
        )
    )


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


def _sigma_seeds(problem: Problem, source) -> dict[str, float]:
    """A start for each estimated sigma that alone scales some measurements: their spread, on the
    observable's scale (decades for ``log10``). Adam in log space moves at most its learning rate
    per step, so a start decades off never arrives within the budget, and the table's nominal is
    not written to be a start."""
    observables = source.observable_df
    pools: dict[str, list[float]] = {}
    for row in source.measurement_df.itertuples(index=False):
        oid = str(row.observableId)
        formula = str(observables.loc[oid, "noiseFormula"])
        cell = getattr(row, "noiseParameters", None)
        overrides = (
            []
            if cell is None or (isinstance(cell, float) and math.isnan(cell))
            else str(cell).split(";")
        )
        for index, value in enumerate(overrides, start=1):
            formula = formula.replace(f"noiseParameter{index}_{oid}", value.strip())
        if formula.strip() not in problem.noise_parameters:
            continue
        transformation = str(
            observables.get("observableTransformation", pd.Series(dtype=str)).get(oid, "lin")
        )
        y = float(row.measurement)
        if transformation == "log10":
            y = math.log10(y)
        elif transformation == "log":
            y = math.log(y)
        pools.setdefault(formula.strip(), []).append(y)
    return {
        pid: float(np.std(values))
        for pid, values in pools.items()
        if len(values) > 1 and np.std(values) > 0.0
    }


def _starts(
    problem: Problem,
    experiments: pd.DataFrame,
    states,
    latents,
    scales,
    *,
    shared: bool,
    directory: Path,
) -> tuple[list[tuple], dict[str, tuple]]:
    """Condition rows setting every state's start, and the parameters the estimated ones need.

    Per experiment the rows go on the preequilibration condition when there is one, else on the
    simulation condition. A start the tables know is written as its number; a latent, or a state
    the tables leave open, gets one parameter per condition or one shared across them.
    """
    existing = set(
        map(
            tuple,
            pd.read_csv(directory / "conditions.tsv", sep="\t")[
                ["conditionId", "targetId"]
            ].itertuples(index=False),
        )
    )
    rows: list[tuple] = []
    estimated: dict[str, tuple] = {}
    for experiment, periods in experiments.groupby("experimentId", sort=False):
        basal = periods[~np.isfinite(periods["time"])]
        target = basal["conditionId"].iloc[0] if len(basal) else str(experiment)
        for state in states:
            if (target, state) in existing:
                continue
            existing.add((target, state))
            if state in problem.pinned_starts:
                rows.append((target, state, problem.pinned_starts[state]))
                continue
            datum = None if len(basal) else problem.measured_starts.get((str(experiment), state))
            if datum is not None:
                rows.append((target, state, float(datum)))
                continue
            pid = f"{state}_0" if shared else f"{state}_0_{target}"
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
            rows.append((target, state, pid))
    return rows, estimated
