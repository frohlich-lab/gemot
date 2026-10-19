"""What a PEtab problem tells the neural route: its states, knobs, starts and scales.

The problem is read as the server reads it, by :meth:`petab_mcp.problem.WorkingProblem.from_yaml`:
PEtab v2 tables, a v1 problem upgraded on load by the server's own conversion. There is one
upgrade, the server's, so the route and the server name the same experiments and score the same
likelihood; a v1 reader beside it would be a second conversion that could only drift from it.

Only the data tables are read: measurements, observables, conditions, experiments and parameters
and, where a problem ships one, an ``inputs.tsv`` protocol table. The model is never opened. A
neural ODE's whole point here is to ask how much mechanism the DATA carry, so the species,
reactions and parameters of the shipped model would leak the answer being measured -- the same
reason the zero-shot examples ship an empty SBML.

The states are the symbols the observable formulas read that are neither parameters nor declared
placeholders: ``obs_P = P`` names the state ``P``, fujita's ``scaleFactor * (pAkt + pAkt_S6)``
names two. The formulas themselves are kept, so the staged problem's observation model is the
shipped one and AMICI evaluates it. Condition targets are experimental knobs. One that names a
state (``S2`` or ``init_S2``) sets that state's starting amount; any other ``init_*`` target is a
dosed species the data never observe, which becomes a named state of its own; everything else is
a clamped input the field reads as a constant. A fixed parameter named ``<state>_init`` or
``init_<state>`` pins that state's start too. These are conventions about experimental setup and
are labelled as such; ``dosed`` lets a caller declare a target dosed where the convention cannot
see it.

An experiment's knobs are what its periods have set by the time it is simulated: a v2 change
persists, so a target only the pre-equilibration sets still holds afterwards. A later period that
sets a knob again is a schedule, kept per experiment (:meth:`Problem.input_values`).

Magnitudes are read off the tables as well: the field works in units of each state's and
input's root-mean-square value and of the last measured time, so one width serves a problem
measured in molecules per cell over weeks as well as one in micromolar over minutes.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from petab.v1.math import sympify_petab
from petab.v2 import C

#: Symbols a protocol formula may use besides its declared condition targets.
PROTOCOL_ALLOWED = frozenset(
    {
        "time",
        "piecewise",
        "exp",
        "log",
        "log10",
        "ln",
        "sqrt",
        "pow",
        "abs",
        "min",
        "max",
        "true",
        "false",
        "pi",
    }
)
#: The experiment a measurement without one is staged under: its starts need a condition to sit on.
_SINGLE = "__single__"
#: The basal key of a pre-equilibration period that applies no condition.
_BASAL = "basal"
_SYMBOL = re.compile(r"[A-Za-z_]\w*")


class ProblemError(ValueError):
    """A table the loader cannot turn into one consistent problem."""


@dataclass(frozen=True)
class TimedInput:
    """An imposed protocol: ``target`` follows ``formula`` over ``time`` and its ``columns``."""

    target: str
    formula: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class Problem:
    name: str
    #: Where an ``inputs.tsv`` protocol is read from: the problem's own directory, or the training
    #: problem's for a held-out view of it.
    directory: Path
    observable_ids: tuple[str, ...]
    #: States the observable formulas read, in order of first appearance.
    observed: tuple[str, ...]
    #: Unobserved states whose starting amount the condition table doses.
    dosed: tuple[str, ...]
    #: Condition targets the field reads as clamped constants.
    inputs: tuple[str, ...]
    timed_inputs: tuple[TimedInput, ...]
    #: Experiment -> every target's value when it is first simulated, and basal key -> every
    #: target's value under that pre-equilibration; a parameter read as its nominal.
    condition_values: dict[str, dict[str, float]]
    #: Experiment -> ``(time, values)`` for each later period that sets a target again, the values
    #: in effect from that time on. Empty for an experiment simulated in one period.
    switches: dict[str, tuple[tuple[float, dict[str, float]], ...]]
    #: Condition target -> the state whose start it sets.
    start_columns: dict[str, str]
    #: ``(experiment, state)`` a start target actually pins. A target sets a start only where a
    #: condition the experiment applies before it is simulated carries a value for it: an unset
    #: one reads as zero like any other missing knob, and zero is a legitimate start, so the
    #: target's mere existence cannot stand in for it being set.
    column_starts: frozenset[tuple[str, str]]
    #: State -> the start a fixed ``<state>_init`` parameter pins in every experiment.
    pinned_starts: dict[str, float]
    #: ``(experiment, state)`` -> the datum of an identity readout where the experiment starts.
    measured_starts: dict[tuple[str, str], float]
    experiment_ids: tuple[str, ...]
    #: Experiment -> its periods in time order, each ``(time, condition ids)``.
    periods: dict[str, tuple[tuple[float, tuple[str, ...]], ...]]
    #: Experiment -> its basal key, which names the conditions its pre-equilibration applies;
    #: empty without pre-equilibration.
    preequilibration: dict[str, str]
    n_measurements: int
    #: Noise parameters the problem estimates, one entry per distinct symbol.
    noise_parameters: tuple[str, ...]
    #: Estimated parameters the observable formulas read, directly or through an override: the
    #: readout scale factors. The staged problem fixes them at their nominal values.
    observable_parameters: tuple[str, ...]
    #: Root-mean-square magnitude of every state and input: the field's units.
    scales: dict[str, float]
    #: The last finite measurement time: the field's unit of time.
    horizon: float
    warnings: tuple[str, ...]
    #: The tables as the server holds them, a :class:`petab_mcp.problem.WorkingProblem`. Staging
    #: writes them out again around the generated model.
    tables: Any = field(repr=False, compare=False)

    @property
    def state_names(self) -> tuple[str, ...]:
        return (*self.observed, *self.dosed)

    @property
    def field_input_names(self) -> tuple[str, ...]:
        """What the field reads besides its states: clamped targets no protocol consumes, then the
        protocol targets."""
        consumed = {c for item in self.timed_inputs for c in item.columns}
        return tuple(
            (
                *(n for n in self.inputs if n not in consumed),
                *(item.target for item in self.timed_inputs),
            )
        )

    @property
    def uses_preequilibration(self) -> bool:
        return bool(self.preequilibration)

    @property
    def basal_conditions(self) -> tuple[str, ...]:
        """Distinct basal keys, in order of first use."""
        return tuple(
            dict.fromkeys(
                self.preequilibration[e] for e in self.experiment_ids if e in self.preequilibration
            )
        )

    def start_key(self, experiment: str) -> str:
        """What an experiment's starts are keyed by: its basal key under pre-equilibration, where
        the experiments that settle in the same conditions share one start, else the experiment."""
        return self.preequilibration.get(experiment, experiment)

    def input_values(self, experiment: str, name: str, times) -> np.ndarray:
        """Target ``name``'s value in ``experiment`` at each of ``times``, following its schedule."""
        times = np.asarray(times, dtype=float)
        values = np.full(times.shape, self.condition_values[experiment][name])
        for start, later in self.switches.get(experiment, ()):
            values[times >= start] = later[name]
        return values

    def start_is_known(self, experiment: str, state: str) -> bool:
        """Whether the tables pin ``state``'s start in ``experiment`` at no parameter cost: a datum
        where it starts, a setup-sheet target or a fixed ``<state>_init``. bruno starts at t=5 and
        crauste at t=4, so this is read off the data rather than configured.

        The target clause is per experiment, not per target: an incompletely set start target leaves
        the experiments that do not set it unpinned, and counting those as fixed would understate
        ``k``, and an over-generous parameter budget is how a field gets latents the data cannot pay for.
        """
        return (
            state in self.pinned_starts
            or (experiment, state) in self.column_starts
            or (experiment, state) in self.measured_starts
        )

    def counts(self, *, shared_ics: bool) -> dict:
        """The counts the field-sizing rules read, off this problem's own tables. ``n_fixed_ics``
        is per experiment, or per state when starts are shared across experiments."""
        known = [[self.start_is_known(e, s) for s in self.state_names] for e in self.experiment_ids]
        n_fixed = (
            sum(all(column) for column in zip(*known))
            if shared_ics and known
            else sum(map(sum, known))
        )
        return {
            "n_measurements": self.n_measurements,
            "n_experiments": len(self.experiment_ids),
            "n_modelled_states": len(self.observed),
            "n_dosed": len(self.dosed),
            "n_inputs": len(self.field_input_names),
            "n_fixed_ics": int(n_fixed),
            "n_sigma": len(self.noise_parameters),
        }


def load(problem: Path | str, *, name: str | None = None, dosed: tuple[str, ...] = ()) -> Problem:
    """Read a PEtab problem directory (or its yaml), of either version, into a :class:`Problem`.

    ``dosed`` names condition targets that are dosed species although no naming convention says so
    (enzyme_kinetics' ``S`` is substrate added at t=0 and consumed). Raises :class:`ProblemError`
    on a problem the server cannot read or this reader cannot make consistent.
    """
    from petab_mcp.problem import WorkingProblem  # noqa: PLC0415

    path = Path(problem)
    yaml_path = path if path.is_file() else path / "problem.yaml"
    name = name or (
        yaml_path.parent.parent.name
        if yaml_path.parent.name == "problem"
        else yaml_path.parent.name
    )
    try:
        tables = WorkingProblem.from_yaml(yaml_path)
    except (OSError, KeyError, ValueError) as why:
        raise ProblemError(f"{name}: {yaml_path} is not a PEtab problem the server reads: {why}")
    return read(tables, name=name, directory=yaml_path.parent, dosed=dosed)


def read(tables, *, name: str, directory: Path, dosed: tuple[str, ...] = ()) -> Problem:
    """A :class:`Problem` from tables the server already holds, a
    :class:`petab_mcp.problem.WorkingProblem` such as the held-out view of a training problem;
    ``directory`` is where ``inputs.tsv`` is."""
    from petab_mcp.problem import placeholder_ids  # noqa: PLC0415

    warnings: list[str] = []
    meas = tables.measurement_df
    if meas is None or meas.empty:
        raise ProblemError(f"{name}: the problem has no measurements")
    meas = meas.reset_index(drop=True).copy()
    meas[C.EXPERIMENT_ID] = [
        str(e) if isinstance(e, str) and e.strip() else _SINGLE for e in meas[C.EXPERIMENT_ID]
    ]
    experiment_ids = tuple(pd.unique(meas[C.EXPERIMENT_ID]))

    observables = tables.observable_df
    observable_ids = tuple(str(o) for o in (observables.index if observables is not None else ()))
    unknown = set(meas[C.OBSERVABLE_ID].astype(str)) - set(observable_ids)
    if unknown:
        raise ProblemError(
            f"{name}: measurements reference undeclared observables {sorted(unknown)}"
        )
    nominal, estimated = _parameters(tables.parameter_df)
    placeholders = placeholder_ids(observables)
    formulas = {oid: str(f) for oid, f in zip(observable_ids, observables[C.OBSERVABLE_FORMULA])}
    observed: list[str] = []
    for formula in formulas.values():
        for symbol in _symbols(formula, placeholders):
            if symbol not in nominal and symbol not in estimated and symbol not in observed:
                observed.append(symbol)
    if not observed:
        raise ProblemError(f"{name}: no observable formula reads a state; nothing to model")

    changes = _changes(tables.condition_df)
    targets = list(dict.fromkeys(t for c in changes.values() for t in c))
    lower = {s.lower(): s for s in observed}
    start_columns: dict[str, str] = {}
    dosed_states: list[str] = []
    inputs: list[str] = []
    for target in targets:
        key = target.lower()
        state = lower.get(key) or lower.get(key.removeprefix("init_"))
        if state is not None:
            start_columns[target] = state
        elif target in dosed or (key.startswith("init_") and not key.endswith("_multiplier")):
            state = target.removeprefix("init_") if key.startswith("init_") else target
            start_columns[target] = state
            dosed_states.append(state)
        else:
            inputs.append(target)
    pinned_starts: dict[str, float] = {}
    for symbol, value in nominal.items():
        if estimated.get(symbol):
            continue
        key = symbol.lower()
        for stem in (key.removeprefix("init_"), key.removesuffix("_init")):
            if stem != key and stem in lower and lower[stem] not in pinned_starts:
                pinned_starts[lower[stem]] = float(value)

    periods = _periods(tables.experiment_df, experiment_ids, name)
    preequilibration = _preequilibration(periods, experiment_ids, name)
    values, given, switches = _knobs(changes, targets, periods, preequilibration, nominal, warnings)
    column_starts = frozenset(
        (experiment, state)
        for target, state in start_columns.items()
        for experiment in experiment_ids
        if target in given[experiment]
    )
    identity = {oid: f for oid, f in formulas.items() if f.strip() in observed}
    start_time = {
        e: next((t for t, _ in periods[e] if t != C.TIME_PREEQUILIBRATION), 0.0)
        for e in experiment_ids
    }
    measured_starts: dict[tuple[str, str], float] = {}
    at_start = meas[
        (meas[C.TIME].astype(float) == meas[C.EXPERIMENT_ID].map(start_time))
        & meas[C.OBSERVABLE_ID].isin(identity)
    ]
    for (experiment, oid), rows in at_start.groupby([C.EXPERIMENT_ID, C.OBSERVABLE_ID]):
        measured_starts[(str(experiment), identity[str(oid)].strip())] = float(
            rows[C.MEASUREMENT].astype(float).mean()
        )

    scales: dict[str, float] = {}
    for state in observed:
        scales[state] = _observed_scale(state, meas, formulas, nominal, observables)
    # Every value a target takes while an experiment is simulated, a switch's included.
    simulated = {
        target: [values[e][target] for e in experiment_ids]
        + [later[target] for e in experiment_ids for _, later in switches.get(e, ())]
        for target in targets
    }
    for target, state in start_columns.items():
        if state in dosed_states:
            scales[state] = _rms(simulated[target])
    for target in inputs:
        scales[target] = _rms(simulated[target])
    finite = meas[C.TIME].astype(float)
    finite = finite[np.isfinite(finite)]
    horizon = float(finite.max()) if len(finite) and finite.max() > 0 else 1.0
    timed = _timed_inputs(directory, inputs, warnings)
    for item in timed:
        scales[item.target] = 1.0

    return Problem(
        name=name,
        directory=Path(directory),
        observable_ids=observable_ids,
        observed=tuple(observed),
        dosed=tuple(dosed_states),
        inputs=tuple(inputs),
        timed_inputs=timed,
        condition_values=values,
        switches=switches,
        start_columns=start_columns,
        column_starts=column_starts,
        pinned_starts=pinned_starts,
        measured_starts=measured_starts,
        experiment_ids=experiment_ids,
        periods=periods,
        preequilibration=preequilibration,
        n_measurements=int(len(meas)),
        noise_parameters=_noise_parameters(meas, observables, nominal, estimated),
        observable_parameters=_observable_parameters(tables, estimated),
        scales=scales,
        horizon=horizon,
        warnings=tuple(dict.fromkeys(warnings)),
        tables=tables,
    )


def _symbols(formula: str, placeholders=frozenset()) -> list[str]:
    """Free symbols of a PEtab formula in order of appearance, ``placeholders`` excluded."""
    names = [str(s) for s in sympify_petab(formula).free_symbols if str(s) not in placeholders]
    return sorted(names, key=formula.find)


def _parameters(parameter_df) -> tuple[dict, dict]:
    """``{id: nominalValue}`` and ``{id: estimated}``; a row without a value counts as estimated."""
    from petab_mcp.problem import as_bool  # noqa: PLC0415

    nominal, estimated = {}, {}
    if parameter_df is None or parameter_df.empty:
        return nominal, estimated
    for pid, row in parameter_df.iterrows():
        value = _as_float(row.get(C.NOMINAL_VALUE))
        estimated[str(pid)] = value is None or as_bool(row.get(C.ESTIMATE))
        if value is not None:
            nominal[str(pid)] = value
    return nominal, estimated


def _noise_parameters(meas, observables, nominal, estimated) -> tuple[str, ...]:
    """Distinct symbols the noise formulas resolve to that the parameter table estimates: a
    ``noiseFormula`` naming a parameter, or a placeholder the measurements override with one."""
    from petab_mcp.problem import placeholder_ids, placeholders  # noqa: PLC0415

    found: dict[str, None] = {}
    declared = placeholder_ids(observables)
    overrides = (
        meas[C.NOISE_PARAMETERS].astype(str)
        if C.NOISE_PARAMETERS in meas.columns
        else pd.Series([""] * len(meas))
    )
    for oid, row in observables.iterrows():
        formula = "" if _blank(row.get(C.NOISE_FORMULA)) else str(row[C.NOISE_FORMULA])
        # Free symbols, not a regex over identifiers: `sigma * exp(-decay * time)` makes `exp` look
        # like a name, and since no parameter row carries it, the estimated-by-default rule below
        # would count it as a sigma -- inflating `n_sigma` and naming a parameter that does not
        # exist. `time` is PEtab's independent variable and reaches free_symbols as an ordinary
        # symbol, so it needs excluding too, for the same reason and with the same consequence.
        # The measurement overrides stay a regex: those cells are raw text, not a formula.
        names = (
            {str(s) for s in sympify_petab(formula).free_symbols} - {C.TIME}
            if formula.strip()
            else set()
        )
        if names & set(placeholders(observables, oid, "noise")):
            rows = (meas[C.OBSERVABLE_ID].astype(str) == str(oid)).to_numpy()
            for cell in overrides[rows]:
                names.update(
                    n
                    for n in str(cell).split(C.PARAMETER_SEPARATOR)
                    if _SYMBOL.fullmatch(n.strip())
                )
        for symbol in sorted(names):
            if symbol not in declared and estimated.get(symbol, symbol not in nominal):
                found.setdefault(symbol)
    return tuple(found)


def _observable_parameters(source, estimated) -> tuple[str, ...]:
    """The estimated readout scale factors, from the server's own resolver.

    :func:`petab_mcp.reconcile.observable_scaling_ids` already resolves these through
    ``observableFormula`` and the measurement-level ``observableParameters`` overrides, and its
    ``formula_symbols`` handles what a local regex would not: an empty cell, which
    ``str(nan)`` turns into a bogus ``"nan"`` symbol, and numeric literals in scientific notation.
    It is the reviewed definition of this set -- the same one the server classifies inner
    parameters and enforces observation-model roles with -- so the route reads it rather than
    keeping a second opinion that could drift from it.
    """
    from petab_mcp.reconcile import observable_scaling_ids  # noqa: PLC0415

    return tuple(sorted(s for s in observable_scaling_ids(source) if estimated.get(s)))


def _observed_scale(state, meas, formulas, nominal, observables) -> float:
    """RMS of the measurements reading ``state``, each divided by its observable's gain in the
    state, so the number is the state's own magnitude rather than the readout's."""
    from petab_mcp.problem import placeholders  # noqa: PLC0415

    parts = []
    for oid, formula in formulas.items():
        expr = sympify_petab(formula)
        if state not in {str(s) for s in expr.free_symbols}:
            continue
        rows = meas[meas[C.OBSERVABLE_ID].astype(str) == oid]
        cells = (
            rows[C.OBSERVABLE_PARAMETERS]
            if C.OBSERVABLE_PARAMETERS in rows.columns
            else pd.Series([None] * len(rows), index=rows.index)
        )
        declared = placeholders(observables, oid, "observable")
        gains = {
            cell: _gain(expr, state, nominal, cell, declared) for cell in cells.astype(str).unique()
        }
        parts.append(
            rows[C.MEASUREMENT].to_numpy(dtype=float) / cells.astype(str).map(gains).to_numpy()
        )
    return _rms(np.concatenate(parts)) if parts else 1.0


def _gain(expr, state: str, nominal, cell, placeholders=()) -> float:
    """The observable's slope in ``state``: the formula at ``state = 1`` less the formula at
    ``state = 0``, with every parameter at its nominal, every declared placeholder at the row's
    override and every other state at zero -- so an additive offset does not enter the scale."""
    placeholders = list(placeholders)
    overrides = (
        [] if _blank(cell) or cell in ("None", "nan") else str(cell).split(C.PARAMETER_SEPARATOR)
    )
    values = {}
    own = [symbol for symbol in expr.free_symbols if str(symbol) == state]
    for symbol in expr.free_symbols:
        name = str(symbol)
        if name == state:
            continue
        if name in placeholders:
            index = placeholders.index(name)
            override = overrides[index].strip() if index < len(overrides) else ""
            values[symbol] = (
                _as_float(override)
                if _as_float(override) is not None
                else nominal.get(override, 1.0)
            )
        else:
            values[symbol] = nominal.get(name, 0.0)
    try:
        at = [float(expr.subs({**values, **{symbol: x for symbol in own}})) for x in (1.0, 0.0)]
        gain = abs(at[0] - at[1])
    except (TypeError, ValueError):
        return 1.0
    return gain if math.isfinite(gain) and gain > 0.0 else 1.0


def _rms(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    magnitude = float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0
    return magnitude if magnitude > 0.0 else 1.0


def _changes(condition_df) -> dict[str, dict[str, Any]]:
    """Condition -> ``{target: value}``, the long v2 table folded per condition."""
    if condition_df is None or not len(condition_df):
        return {}
    return {
        str(cid): {str(t): v for t, v in zip(rows[C.TARGET_ID], rows[C.TARGET_VALUE])}
        for cid, rows in condition_df.groupby(C.CONDITION_ID, sort=False)
    }


def _periods(experiment_df, experiment_ids, name) -> dict:
    """Experiment -> its periods in time order, each ``(time, condition ids)``, the conditions of
    one period applied together. A measurement without an experiment is simulated from the model's
    own start, so its stand-in has one period at t=0 that applies nothing."""
    periods: dict[str, tuple] = {}
    if experiment_df is not None and len(experiment_df):
        for eid, rows in experiment_df.groupby(C.EXPERIMENT_ID, sort=False):
            by_time: dict[float, list[str]] = {}
            for t, cid in zip(rows[C.TIME].astype(float), rows[C.CONDITION_ID]):
                by_time.setdefault(float(t), [])
                if not _blank(cid):
                    by_time[float(t)].append(str(cid).strip())
            periods[str(eid)] = tuple((t, tuple(c)) for t, c in sorted(by_time.items()))
    missing = [e for e in experiment_ids if e != _SINGLE and e not in periods]
    if missing:
        raise ProblemError(
            f"{name}: measurements name experiments {missing} the experiment table does not define"
        )
    periods[_SINGLE] = ((0.0, ()),)
    return {e: periods[e] for e in experiment_ids}


def _knobs(changes, targets, periods, preequilibration, nominal, warnings):
    """Per experiment and per basal key, every target's value, a parameter's name read as its
    nominal; per experiment, the targets set by the time it is simulated, and its later switches.

    The values accumulate across periods, as v2 changes persist; a target nothing sets reads as
    zero, and says so where an experiment sets none at all.
    """

    def value(target, cell, where):
        number = _as_float(cell)
        if number is not None:
            return number
        if str(cell).strip() in nominal:
            return float(nominal[str(cell).strip()])
        if not _blank(cell):
            warnings.append(f"{target}={cell!r} in {where!r} is not numeric; read as 0")
        return 0.0

    values: dict[str, dict[str, float]] = {}
    given: dict[str, set[str]] = {}
    switches: dict[str, tuple] = {}
    for experiment, sequence in periods.items():
        applied: dict[str, float] = {}
        later: list[tuple[float, dict[str, float]]] = []
        for time, condition_ids in sequence:
            for cid in condition_ids:
                for target, cell in changes.get(cid, {}).items():
                    applied[target] = value(target, cell, cid)
            if time == C.TIME_PREEQUILIBRATION:
                values[preequilibration[experiment]] = {t: applied.get(t, 0.0) for t in targets}
            elif experiment not in values:
                values[experiment] = {t: applied.get(t, 0.0) for t in targets}
                given[experiment] = set(applied)
            elif condition_ids:
                later.append((time, {t: applied.get(t, 0.0) for t in targets}))
        if experiment not in values:
            raise ProblemError(f"experiment {experiment!r} has no period after pre-equilibration")
        # An empty first period followed by a switch is how the server's conversion starts a
        # converted protocol at t=0; only an experiment that never sets a knob is worth a word.
        if targets and not given[experiment] and not later:
            warnings.append(f"experiment {experiment!r} sets no condition target; knobs read as 0")
        if later:
            switches[experiment] = tuple(later)
    return values, given, switches


def _preequilibration(periods, experiment_ids, name) -> dict[str, str]:
    """Experiment -> its basal key, for every experiment or none: the conditions its
    pre-equilibration applies, joined, or :data:`_BASAL` where it applies none."""
    found = {
        e: "__".join(sorted(cids)) or _BASAL
        for e in experiment_ids
        for t, cids in periods[e]
        if t == C.TIME_PREEQUILIBRATION
    }
    if not found:
        return {}
    missing = [e for e in experiment_ids if e not in found]
    if missing:
        raise ProblemError(
            f"{name}: experiments {missing} preequilibrate in no condition while others do"
        )
    return found


def _timed_inputs(directory: Path, input_names, warnings) -> tuple[TimedInput, ...]:
    """``inputs.tsv``: ``inputId``, Antimony ``formula``, ``;``-separated ``conditionInputs``.

    A formula may name only ``time`` and its own condition columns. Anything else would carry
    model structure in through the protocol, so the row is refused and said so.
    """
    path = directory / "inputs.tsv"
    if not path.is_file():
        return ()
    out = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            target = str(row.get("inputId") or "").strip()
            formula = str(row.get("formula") or "").strip()
            columns = tuple(
                c.strip() for c in str(row.get("conditionInputs") or "").split(";") if c.strip()
            )
            if not target or not formula:
                continue
            symbols = {s for s in _SYMBOL.findall(formula) if s.lower() not in PROTOCOL_ALLOWED}
            outside = (symbols - set(columns)) | (set(columns) - set(input_names))
            if outside:
                warnings.append(
                    f"input {target!r} REFUSED: it names {sorted(outside)}, which are "
                    "not clamped condition columns of this problem"
                )
                continue
            out.append(TimedInput(target, formula, columns))
    return tuple(out)


def _blank(value) -> bool:
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or (isinstance(value, str) and not value.strip())
    )


def _as_float(value) -> float | None:
    if _blank(value):
        return None
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    text = str(value).strip()
    if C.PARAMETER_SEPARATOR in text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
